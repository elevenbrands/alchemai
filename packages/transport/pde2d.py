"""Steady 2-D scalar field problems written by the user, on structured grids (cell-centred finite volumes).

Equation, for one or several coupled variables u_i:

    div(D_i grad u_i) - f_i * v . grad u_i + R_i(coords, u, params) = 0

Coordinate systems (a, b) and sides ('<coord>_min' / '<coord>_max'):
    cartesian               (x, y)       rectangle, per unit depth
    axisymmetric            (r, z)       cylinder or annulus, full 2*pi
    polar                   (r, theta)   sector r_in..r_out, theta_0..theta_1, per unit depth
    spherical_axisymmetric  (r, theta)   shell or sphere, theta measured from the +z axis, full 2*pi

Method: tensor-product grid (optionally graded), two-point diffusive fluxes with the exact metric of each system,
convection in the non-conservative form F_f (u_f - u_P) (so a slightly non-solenoidal discrete velocity does not create
sources) with central, first-order upwind or van Leer TVD (deferred correction) face values, boundary-face unknowns so
that every boundary condition is one row, sparse LU and a damped Newton iteration for nonlinear sources and fluxes.

Evidence returned with every call: the problem is solved on three nested grids (n/2, n, 2n); every requested quantity
carries its change between the last two, the observed order and a `grid_converged` flag; the nonlinear residual and the
global balance (boundary fluxes, convection term, integrated source) evaluated with array formulas that do not use the
assembled matrix.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import splu

from packages.transport.pde2d_expr import RESERVED, Expr

SYSTEMS = {"cartesian": ("x", "y"), "axisymmetric": ("r", "z"), "polar": ("r", "theta"),
           "spherical_axisymmetric": ("r", "theta")}
BC_TYPES = ("dirichlet", "neumann", "robin", "flux", "symmetry")
SCHEMES = ("auto", "central", "upwind", "tvd")
DEFAULT_N = 64
MIN_N = 8
MAX_UNKNOWNS = 600_000        # unknowns of the finest grid (variables x padded cells)
MAX_POINTS = 200
NEWTON_TOL = 1e-10
TVD_TOL = 1e-8
PECLET_CENTRAL_LIMIT = 2.0
_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------------------------------------------------
# input parsing
# ---------------------------------------------------------------------------------------------------------------------
def _num(v, where, positive=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{where}: se esperaba un número finito")
    if positive and v <= 0:
        raise ValueError(f"{where}: debe ser > 0")
    return float(v)


def _ident(name, where):
    if not isinstance(name, str) or not name.isidentifier() or name in RESERVED:
        raise ValueError(f"{where}: '{name}' no es un nombre válido (identificador que no sea función ni 'pi')")
    return name


class _Problem:
    """Parsed, grid-independent description of the problem."""

    def __init__(self, coordinates, domain, equations, boundary_conditions, velocity, parameters, outputs, grid,
                 initial_guess, convection_scheme, refinement_tol, max_iterations):
        if coordinates not in SYSTEMS:
            raise ValueError(f"coordinates: '{coordinates}' no reconocido; opciones: {', '.join(SYSTEMS)}")
        self.system = coordinates
        self.cn = SYSTEMS[coordinates]
        ca, cb = self.cn
        self.sides = (f"{ca}_min", f"{ca}_max", f"{cb}_min", f"{cb}_max")
        self._domain(domain)
        # parameters and variables
        self.params = {}
        for k, v in (parameters or {}).items():
            self.params[_ident(k, "parameters")] = _num(v, f"parameters['{k}']")
        if not isinstance(equations, dict) or not equations:
            raise ValueError("equations: se espera {variable: {'D': ..., 'source': ...}} con al menos una variable")
        self.vars = [_ident(v, "equations") for v in equations]
        clash = (set(self.vars) | set(self.params)) & set(self.cn) or set(self.vars) & set(self.params)
        if clash:
            raise ValueError(f"nombres repetidos entre coordenadas, variables y parámetros: {', '.join(sorted(clash))}")
        self.nv = len(self.vars)
        self.n_geo = set(self.cn) | set(self.params)
        self.n_all = self.n_geo | set(self.vars)
        self.eq = [self._equation(v, equations[v]) for v in self.vars]
        self.vel = self._velocity(velocity)
        self.bcs = self._boundaries(boundary_conditions)
        self._outputs(outputs or {})
        self._grid(grid or {})
        self.guess = {}
        for v, e in (initial_guess or {}).items():
            if v not in self.vars:
                raise ValueError(f"initial_guess: '{v}' no es una variable de equations")
            self.guess[v] = Expr(e, self.n_geo, f"initial_guess['{v}']")
        if convection_scheme not in SCHEMES:
            raise ValueError(f"convection_scheme: '{convection_scheme}' no reconocido; opciones: {', '.join(SCHEMES)}")
        self.scheme_req = convection_scheme
        self.tol = _num(refinement_tol, "refinement_tol", positive=True)
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= 500:
            raise ValueError("max_iterations: entero entre 1 y 500")
        self.max_it = max_iterations

    # -- pieces -------------------------------------------------------------------------------------------------------
    def _domain(self, domain):
        ca, cb = self.cn
        if not isinstance(domain, dict):
            raise ValueError(f"domain: se espera {{'{ca}': [mín, máx], '{cb}': [mín, máx]}}")
        dom = dict(domain)
        if self.system == "spherical_axisymmetric" and "theta" not in dom:
            dom["theta"] = [0.0, math.pi]
        extra = set(dom) - set(self.cn)
        if extra:
            raise ValueError(f"domain: claves no reconocidas para '{self.system}': {', '.join(sorted(extra))}; "
                             f"se esperan '{ca}' y '{cb}'")
        lims = []
        for c in self.cn:
            if c not in dom or not isinstance(dom[c], (list, tuple)) or len(dom[c]) != 2:
                raise ValueError(f"domain: falta '{c}': [mín, máx]")
            lo, hi = _num(dom[c][0], f"domain['{c}'][0]"), _num(dom[c][1], f"domain['{c}'][1]")
            if hi <= lo:
                raise ValueError(f"domain['{c}']: el máximo debe ser mayor que el mínimo")
            lims.append((lo, hi))
        if ca == "r" and lims[0][0] < 0:
            raise ValueError("domain['r']: el radio mínimo no puede ser negativo")
        if self.system == "polar" and lims[1][1] - lims[1][0] > _TWO_PI * (1 + 1e-12):
            raise ValueError("domain['theta']: el sector no puede abarcar más de 2·pi rad")
        if self.system == "spherical_axisymmetric":
            lo, hi = lims[1]
            if lo < -1e-12 or hi > math.pi * (1 + 1e-12):
                raise ValueError("domain['theta']: en esféricas theta (desde el eje +z) va de 0 a pi rad")
            lims[1] = (max(lo, 0.0), min(hi, math.pi))
        self.lims = lims

    def _equation(self, v, spec):
        ca, cb = self.cn
        if not isinstance(spec, dict):
            raise ValueError(f"equations['{v}']: se espera un dict con 'D' y opcionalmente 'source'")
        extra = set(spec) - {"D", "source", "velocity_factor"}
        if extra:
            raise ValueError(f"equations['{v}']: claves no reconocidas: {', '.join(sorted(extra))}; se admiten 'D', "
                             "'source' y 'velocity_factor'")
        if "D" not in spec:
            raise ValueError(f"equations['{v}']: falta 'D' (difusividad: número, expresión o {{'{ca}': .., '{cb}': ..}})")
        d = spec["D"]
        if isinstance(d, dict):
            if set(d) != set(self.cn):
                raise ValueError(f"equations['{v}']['D']: como dict debe tener exactamente las claves '{ca}' y '{cb}'")
            da, db = (Expr(d[c], self.n_geo, f"equations['{v}']['D']['{c}']") for c in self.cn)
        else:
            da = db = Expr(d, self.n_geo, f"equations['{v}']['D']")
        src = spec.get("source")
        src = None if src is None else Expr(src, self.n_all, f"equations['{v}']['source']")
        fac = spec.get("velocity_factor", 1.0)
        fac = float(Expr(fac, set(self.params), f"equations['{v}']['velocity_factor']")(self.params))
        return {"Da": da, "Db": db, "src": src, "fac": fac}

    def _velocity(self, velocity):
        if velocity is None:
            return None
        ca, cb = self.cn
        if not isinstance(velocity, dict) or not velocity or set(velocity) - set(self.cn):
            raise ValueError(f"velocity: se espera {{'{ca}': expresión, '{cb}': expresión}} (componentes físicas de la "
                             "velocidad en función de las coordenadas; la que falte vale 0)")
        return tuple(Expr(velocity[c], self.n_geo, f"velocity['{c}']") if c in velocity else None for c in self.cn)

    def _bc(self, spec, where):
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError(f"{where}: se espera {{'type': ...}}; tipos: {', '.join(BC_TYPES)}")
        t = spec["type"]
        if t not in BC_TYPES:
            raise ValueError(f"{where}: tipo '{t}' no reconocido; tipos: {', '.join(BC_TYPES)}")
        allowed = {"dirichlet": {"value"}, "neumann": {"value"}, "robin": {"h", "u_ext"}, "flux": {"value"},
                   "symmetry": set()}[t] | {"type", "from", "to"}
        extra = set(spec) - allowed
        if extra:
            raise ValueError(f"{where}: claves no reconocidas para '{t}': {', '.join(sorted(extra))}")
        out = {"kind": t, "lo": -math.inf, "hi": math.inf}
        if "from" in spec:
            out["lo"] = _num(spec["from"], f"{where}['from']")
        if "to" in spec:
            out["hi"] = _num(spec["to"], f"{where}['to']")
        if out["hi"] <= out["lo"]:
            raise ValueError(f"{where}: 'to' debe ser mayor que 'from'")
        if t == "dirichlet":
            if "value" not in spec:
                raise ValueError(f"{where}: Dirichlet necesita 'value'")
            out["value"] = Expr(spec["value"], self.n_geo, f"{where}['value']")
        elif t == "neumann":
            out["value"] = Expr(spec.get("value", 0.0), self.n_geo, f"{where}['value']")
        elif t == "symmetry":
            out["kind"], out["value"] = "neumann", Expr(0.0, self.n_geo, where)
        elif t == "robin":
            if "h" not in spec or "u_ext" not in spec:
                raise ValueError(f"{where}: Robin necesita 'h' y 'u_ext' (flujo saliente −D·du/dn = h·(u − u_ext))")
            out["h"] = Expr(spec["h"], self.n_geo, f"{where}['h']")
            out["u_ext"] = Expr(spec["u_ext"], self.n_geo, f"{where}['u_ext']")
        else:
            if "value" not in spec:
                raise ValueError(f"{where}: 'flux' necesita 'value' (flujo difusivo saliente −D·du/dn, expresión de "
                                 "coordenadas, variables en la pared y parámetros)")
            out["value"] = Expr(spec["value"], self.n_all, f"{where}['value']")
        return out

    def _boundaries(self, bcs):
        if not isinstance(bcs, dict):
            raise ValueError(f"boundary_conditions: se espera un dict por lado; lados: {', '.join(self.sides)}")
        extra = set(bcs) - set(self.sides)
        if extra:
            raise ValueError(f"boundary_conditions: lados no reconocidos {', '.join(sorted(extra))}; en "
                             f"'{self.system}' los lados son {', '.join(self.sides)}")
        out = {}
        for side in self.sides:
            if side not in bcs:
                out[side] = {v: None for v in self.vars}
                continue
            spec = bcs[side]
            if isinstance(spec, list) or (isinstance(spec, dict) and "type" in spec):
                per_var = {v: spec for v in self.vars}
            elif isinstance(spec, dict):
                bad = set(spec) - set(self.vars)
                if bad:
                    raise ValueError(f"boundary_conditions['{side}']: se espera una condición ({{'type': ...}} o lista "
                                     f"de tramos) o un dict por variable; claves no reconocidas: {', '.join(sorted(bad))}")
                per_var = {v: spec.get(v) for v in self.vars}
            else:
                raise ValueError(f"boundary_conditions['{side}']: se espera una condición o un dict por variable")
            out[side] = {}
            for v, s in per_var.items():
                w = f"boundary_conditions['{side}']" + (f"['{v}']" if self.nv > 1 else "")
                if s is None:
                    out[side][v] = None
                elif isinstance(s, list):
                    if not s:
                        raise ValueError(f"{w}: lista de tramos vacía")
                    out[side][v] = [self._bc(x, f"{w}[{i}]") for i, x in enumerate(s)]
                else:
                    out[side][v] = [self._bc(s, w)]
        return out

    def _point(self, p, where):
        if isinstance(p, dict):
            if set(p) != set(self.cn):
                raise ValueError(f"{where}: como dict debe tener las claves '{self.cn[0]}' y '{self.cn[1]}'")
            p = [p[self.cn[0]], p[self.cn[1]]]
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"{where}: se espera [{self.cn[0]}, {self.cn[1]}]")
        out = []
        for c, x, (lo, hi) in zip(self.cn, p, self.lims):
            x = _num(x, f"{where} ({c})")
            eps = 1e-9 * (hi - lo)
            if x < lo - eps or x > hi + eps:
                raise ValueError(f"{where}: {c} = {x:g} está fuera del dominio [{lo:g}, {hi:g}]")
            out.append(min(max(x, lo), hi))
        return tuple(out)

    def _outputs(self, o):
        if not isinstance(o, dict):
            raise ValueError("outputs: se espera un dict")
        known = {"points", "derivative_points", "averages", "mixing_cup", "side_fluxes", "derived"}
        if set(o) - known:
            raise ValueError(f"outputs: claves no reconocidas {', '.join(sorted(set(o) - known))}; se admiten "
                             + ", ".join(sorted(known)))
        self.points = [self._point(p, f"outputs['points'][{i}]") for i, p in enumerate(o.get("points") or [])]
        self.dpoints = [self._point(p, f"outputs['derivative_points'][{i}]")
                        for i, p in enumerate(o.get("derivative_points") or [])]
        if len(self.points) + len(self.dpoints) > MAX_POINTS:
            raise ValueError(f"outputs: máximo {MAX_POINTS} puntos en total")
        names: dict[str, str] = {}

        def new_name(item, where):
            if not isinstance(item, dict) or "name" not in item:
                raise ValueError(f"{where}: se espera un dict con 'name'")
            n = _ident(item["name"], where)
            if n in names or n in self.n_all:
                raise ValueError(f"{where}: el nombre '{n}' ya está en uso")
            names[n] = where
            return n

        self.averages = []
        for i, it in enumerate(o.get("averages") or []):
            w = f"outputs['averages'][{i}]"
            n = new_name(it, w)
            if set(it) - {"name", "expression"} or "expression" not in it:
                raise ValueError(f"{w}: se espera {{'name', 'expression'}}")
            self.averages.append((n, Expr(it["expression"], self.n_all, f"{w}['expression']")))
        self.mixing = []
        for i, it in enumerate(o.get("mixing_cup") or []):
            w = f"outputs['mixing_cup'][{i}]"
            n = new_name(it, w)
            if set(it) - {"name", "expression", "coordinate", "at", "weight"} or not {"expression", "coordinate", "at"} <= set(it):
                raise ValueError(f"{w}: se espera {{'name', 'expression', 'coordinate', 'at'}} (y 'weight' opcional)")
            if it["coordinate"] not in self.cn:
                raise ValueError(f"{w}['coordinate']: debe ser '{self.cn[0]}' o '{self.cn[1]}' (la sección es "
                                 "coordenada = constante)")
            ax = self.cn.index(it["coordinate"])
            ats = it["at"] if isinstance(it["at"], (list, tuple)) else [it["at"]]
            lo, hi = self.lims[ax]
            vals = []
            for a in ats:
                a = _num(a, f"{w}['at']")
                if a < lo - 1e-9 * (hi - lo) or a > hi + 1e-9 * (hi - lo):
                    raise ValueError(f"{w}['at']: {a:g} está fuera del dominio [{lo:g}, {hi:g}]")
                vals.append(min(max(a, lo), hi))
            if not vals:
                raise ValueError(f"{w}['at']: falta la posición de la sección")
            wexp = None
            if it.get("weight") is not None:
                wexp = Expr(it["weight"], self.n_geo, f"{w}['weight']")
            elif self.vel is None or self.vel[ax] is None:
                raise ValueError(f"{w}: el promedio de mezcla necesita velocity['{it['coordinate']}'] (componente normal "
                                 "a la sección) o un 'weight' explícito")
            self.mixing.append({"name": n, "expr": Expr(it["expression"], self.n_all, f"{w}['expression']"),
                                "axis": ax, "at": vals, "weight": wexp, "scalar": not isinstance(it["at"], (list, tuple))})
        self.fluxes = []
        for i, it in enumerate(o.get("side_fluxes") or []):
            w = f"outputs['side_fluxes'][{i}]"
            n = new_name(it, w)
            if set(it) - {"name", "side", "variable", "from", "to", "at"} or "side" not in it:
                raise ValueError(f"{w}: se espera {{'name', 'side'}} y opcionalmente 'variable', 'from', 'to' o 'at'")
            if "at" in it and ("from" in it or "to" in it):
                raise ValueError(f"{w}: dé 'at' (valor local) o 'from'/'to' (tramo), no ambos")
            if it["side"] not in self.sides:
                raise ValueError(f"{w}['side']: '{it['side']}' no es un lado; lados: {', '.join(self.sides)}")
            var = it.get("variable", self.vars[0] if self.nv == 1 else None)
            if var not in self.vars:
                raise ValueError(f"{w}['variable']: indique una de {', '.join(self.vars)}")
            lo = _num(it["from"], f"{w}['from']") if "from" in it else -math.inf
            hi = _num(it["to"], f"{w}['to']") if "to" in it else math.inf
            if hi <= lo:
                raise ValueError(f"{w}: 'to' debe ser mayor que 'from'")
            at = None
            if "at" in it:
                l0, h0 = self.lims[1 - self.sides.index(it["side"]) // 2]
                at = _num(it["at"], f"{w}['at']")
                if at < l0 - 1e-9 * (h0 - l0) or at > h0 + 1e-9 * (h0 - l0):
                    raise ValueError(f"{w}['at']: {at:g} está fuera del lado [{l0:g}, {h0:g}]")
            self.fluxes.append({"name": n, "side": self.sides.index(it["side"]), "k": self.vars.index(var),
                                "lo": lo, "hi": hi, "at": at})
        avail = set(self.params)
        for n, _ in self.averages:
            avail |= {n, n + "_integral"}
        for m in self.mixing:
            if m["scalar"]:
                avail |= {m["name"], m["name"] + "_area_average"}
        for f in self.fluxes:
            avail |= {f["name"], f["name"] + "_gradient", f["name"] + "_mean"}
            if f["at"] is None:
                avail.add(f["name"] + "_integral")
        d = o.get("derived") or {}
        if not isinstance(d, dict):
            raise ValueError("outputs['derived']: se espera {nombre: expresión}")
        self.derived = []
        for n, e in d.items():
            _ident(n, "outputs['derived']")
            if n in avail or n in self.n_all:
                raise ValueError(f"outputs['derived']: el nombre '{n}' ya está en uso")
            self.derived.append((n, Expr(e, avail, f"outputs['derived']['{n}']")))

    def _grid(self, g):
        if not isinstance(g, dict) or set(g) - {"n", "grading"}:
            raise ValueError("grid: se espera {'n': [n1, n2], 'grading': {coordenada: ...}}")
        n = g.get("n", [DEFAULT_N, DEFAULT_N])
        if isinstance(n, int) and not isinstance(n, bool):
            n = [n, n]
        if (not isinstance(n, (list, tuple)) or len(n) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) for x in n)):
            raise ValueError("grid['n']: se esperan dos enteros (celdas de la malla base en cada coordenada)")
        if min(n) < MIN_N:
            raise ValueError(f"grid['n']: mínimo {MIN_N} celdas por dirección")
        n = [x + (x % 2) for x in n]
        if self.nv * (2 * n[0] + 2) * (2 * n[1] + 2) > MAX_UNKNOWNS:
            raise ValueError(f"grid['n']: la malla refinada ({2 * n[0]}×{2 * n[1]}, {self.nv} variable(s)) supera "
                             f"{MAX_UNKNOWNS} incógnitas; reduzca n")
        self.n = n
        gr = g.get("grading") or {}
        if not isinstance(gr, dict) or set(gr) - set(self.cn):
            raise ValueError(f"grid['grading']: se espera un dict con claves '{self.cn[0]}' y/o '{self.cn[1]}'")
        self.grading = []
        for ax, c in enumerate(self.cn):
            lo, hi = self.lims[ax]
            if c not in gr:
                s = {"type": "uniform"}
                if self.system == "polar" and ax == 0 and lo == 0.0:
                    s = {"type": "power", "cluster": "min", "strength": 2.0}
                elif self.system == "spherical_axisymmetric" and ax == 0 and lo > 0 and hi / lo > 3.0:
                    s = {"type": "log"}
                s["default"] = True
            else:
                x = gr[c]
                w = f"grid['grading']['{c}']"
                if x in ("uniform", "log"):
                    s = {"type": x}
                elif isinstance(x, dict) and not set(x) - {"type", "cluster", "strength"}:
                    t = x.get("type", "power")
                    if t in ("uniform", "log"):
                        s = {"type": t}
                    elif t == "power":
                        cl = x.get("cluster", "min")
                        if cl not in ("min", "max", "both"):
                            raise ValueError(f"{w}['cluster']: 'min', 'max' o 'both'")
                        p = _num(x.get("strength", 2.0), f"{w}['strength']")
                        if not 1.0 <= p <= 6.0:
                            raise ValueError(f"{w}['strength']: entre 1 (uniforme) y 6")
                        s = {"type": "power", "cluster": cl, "strength": p}
                    else:
                        raise ValueError(f"{w}['type']: 'uniform', 'power' o 'log'")
                else:
                    raise ValueError(f"{w}: se espera 'uniform', 'log' o {{'cluster': 'min'|'max'|'both', 'strength': p}}")
                if s["type"] == "log" and lo <= 0:
                    raise ValueError(f"{w}: la malla 'log' necesita un mínimo > 0")
            self.grading.append(s)

    def faces(self, ax, n):
        lo, hi = self.lims[ax]
        s = self.grading[ax]
        xi = np.linspace(0.0, 1.0, n + 1)
        if s["type"] == "log":
            f = lo * (hi / lo) ** xi
        else:
            if s["type"] == "power" and s["strength"] != 1.0:
                p, cl = s["strength"], s["cluster"]
                if cl == "min":
                    xi = xi ** p
                elif cl == "max":
                    xi = 1.0 - (1.0 - xi) ** p
                else:
                    xi = np.where(xi <= 0.5, 0.5 * (2 * xi) ** p, 1.0 - 0.5 * (2 * (1.0 - xi)) ** p)
            f = lo + (hi - lo) * xi
        f[0], f[-1] = lo, hi
        return f


# ---------------------------------------------------------------------------------------------------------------------
# one grid level
# ---------------------------------------------------------------------------------------------------------------------
def _lim(g, d):
    """van Leer limited slope psi(r)*d with r = g/d (harmonic mean of the two slopes when they have the same sign) and
    its partial derivatives with respect to g and d."""
    ok = g * d > 0.0
    den = np.where(ok, g + d, 1.0)
    return (np.where(ok, 2.0 * g * d / den, 0.0), np.where(ok, 2.0 * d * d / den ** 2, 0.0),
            np.where(ok, 2.0 * g * g / den ** 2, 0.0))


def _tvd_delta(Up, x, xf, F, jac=False):
    """High-order minus upwind face value on the interior faces along axis 0. Up (n+2, m), x padded nodes, xf faces.
    With jac=True also returns the padded indices (iC, iD, iU) along the axis and the partial derivatives of delta."""
    n, m = Up.shape[0] - 2, Up.shape[1]
    out = np.zeros((n + 1, m))
    if n < 2:
        return (out, None) if jac else out
    col = lambda v: v[:, None]
    f = np.arange(1, n)
    branches = []
    for iC, iD, iU in ((f, f + 1, f - 1), (f + 1, f, f + 2)):
        cg = col((x[iD] - x[iC]) / (x[iC] - x[iU]))
        frac = col((xf[f] - x[iC]) / (x[iD] - x[iC]))
        lim, lg, ld = _lim((Up[iC] - Up[iU]) * cg, Up[iD] - Up[iC])
        branches.append((lim * frac, (lg * cg - ld) * frac, ld * frac, -lg * cg * frac, iC, iD, iU))
    Fi = F[1:n]
    pos, neg = Fi > 0, Fi < 0
    sel = lambda a, b: np.where(pos, a, np.where(neg, b, 0.0))
    out[1:n] = sel(branches[0][0], branches[1][0])
    if not jac:
        return out
    ones = np.ones((1, m), dtype=int)
    d = {"dC": sel(branches[0][1], branches[1][1]), "dD": sel(branches[0][2], branches[1][2]),
         "dU": sel(branches[0][3], branches[1][3])}
    for name, j in (("iC", 4), ("iD", 5), ("iU", 6)):
        d[name] = np.where(pos, col(branches[0][j]) * ones, col(branches[1][j]) * ones)
    return out, d


class _Level:
    def __init__(self, P: _Problem, na: int, nb: int, scheme: str):
        self.P, self.na, self.nb, self.scheme = P, na, nb, scheme
        af, bf = P.faces(0, na), P.faces(1, nb)
        self.af, self.bf = af, bf
        ac, bc = 0.5 * (af[1:] + af[:-1]), 0.5 * (bf[1:] + bf[:-1])
        self.ac, self.bc = ac, bc
        dA, dB = np.diff(af), np.diff(bf)
        one_a, one_b = np.ones((na + 1, 1)), np.ones((1, nb + 1))
        sysm = P.system
        if sysm == "cartesian":
            Aa, Ab, V, hb = one_a * dB[None, :], dA[:, None] * one_b, dA[:, None] * dB[None, :], np.ones(na)
            self.sec_w = (lambda a: dB, lambda b: dA)
        elif sysm == "axisymmetric":
            ring = math.pi * (af[1:] ** 2 - af[:-1] ** 2)
            Aa, Ab, V, hb = _TWO_PI * af[:, None] * dB[None, :], ring[:, None] * one_b, ring[:, None] * dB[None, :], np.ones(na)
            self.sec_w = (lambda a: _TWO_PI * a * dB, lambda b: ring)
        elif sysm == "polar":
            half = 0.5 * (af[1:] ** 2 - af[:-1] ** 2)
            Aa, Ab, V, hb = af[:, None] * dB[None, :], dA[:, None] * one_b, half[:, None] * dB[None, :], ac
            self.sec_w = (lambda a: a * dB, lambda b: dA)
        else:
            dcos = np.cos(bf[:-1]) - np.cos(bf[1:])
            sb = np.sin(bf)
            sb[np.abs(sb) < 1e-12] = 0.0
            ring = math.pi * (af[1:] ** 2 - af[:-1] ** 2)
            Aa = _TWO_PI * (af ** 2)[:, None] * dcos[None, :]
            Ab = ring[:, None] * sb[None, :]
            V = (_TWO_PI / 3.0) * (af[1:] ** 3 - af[:-1] ** 3)[:, None] * dcos[None, :]
            hb = ac
            self.sec_w = (lambda a: _TWO_PI * a * a * dcos, lambda b: ring * math.sin(b))
        self.Aa, self.Ab, self.V = Aa, Ab, V
        self.ap = np.concatenate([af[:1], ac, af[-1:]])
        self.bp = np.concatenate([bf[:1], bc, bf[-1:]])
        self.da = np.diff(self.ap)[:, None] * np.ones((1, nb))
        self.db = hb[:, None] * np.diff(self.bp)[None, :]
        self.M = (na + 2) * (nb + 2)
        self.shape = (P.nv, na + 2, nb + 2)
        ca, cb = P.cn
        env = dict(P.params)
        self.env_c = {**env, ca: ac[:, None] * np.ones((1, nb)), cb: np.ones((na, 1)) * bc[None, :]}
        self.env_fa = {**env, ca: af[:, None] * np.ones((1, nb)), cb: np.ones((na + 1, 1)) * bc[None, :]}
        self.env_fb = {**env, ca: ac[:, None] * np.ones((1, nb + 1)), cb: np.ones((na, 1)) * bf[None, :]}
        # velocity fluxes through the faces (positive along +coordinate)
        self.Fa, self.Fb = np.zeros((na + 1, nb)), np.zeros((na, nb + 1))
        if P.vel is not None:
            if P.vel[0] is not None:
                self.Fa = self._finite(P.vel[0](self.env_fa, (na + 1, nb)), "velocity['%s']" % ca) * Aa
            if P.vel[1] is not None:
                self.Fb = self._finite(P.vel[1](self.env_fb, (na, nb + 1)), "velocity['%s']" % cb) * Ab
        self.has_conv = bool(np.any(self.Fa != 0) or np.any(self.Fb != 0))
        self.Ga, self.Gb, self.Dfa, self.Dfb = [], [], [], []
        for k, e in enumerate(P.eq):
            Da = self._finite(e["Da"](self.env_fa, (na + 1, nb)), f"D de '{P.vars[k]}'")
            Db = self._finite(e["Db"](self.env_fb, (na, nb + 1)), f"D de '{P.vars[k]}'")
            if np.any(Da < 0) or np.any(Db < 0):
                raise ValueError(f"equations['{P.vars[k]}']['D']: la difusividad es negativa en alguna parte del dominio")
            if not (np.any(Da > 0) or np.any(Db > 0)):
                raise ValueError(f"equations['{P.vars[k]}']['D']: la difusividad es cero en todo el dominio")
            self.Dfa.append(Da)
            self.Dfb.append(Db)
            self.Ga.append(Da * Aa / self.da)
            self.Gb.append(Db * Ab / self.db)

    @staticmethod
    def _finite(v, what):
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{what}: la expresión no da un valor finito en algún punto de la malla (división entre "
                             "cero o dominio de una función)")
        return v

    def peclet(self):
        pe = 0.0
        for k, e in enumerate(self.P.eq):
            for F, G in ((self.Fa[1:-1], self.Ga[k][1:-1]), (self.Fb[:, 1:-1], self.Gb[k][:, 1:-1])):
                F = np.abs(F) * abs(e["fac"])
                m = F > 0
                if np.any(m):
                    with np.errstate(divide="ignore"):
                        pe = max(pe, float(np.max(np.where(G[m] > 0, F[m] / np.where(G[m] > 0, G[m], 1.0), np.inf))))
        return pe

    # -- boundary sides -----------------------------------------------------------------------------------------------
    def side(self, s, k):
        """Geometry of side s (0 a_min, 1 a_max, 2 b_min, 3 b_max) for variable k."""
        na, nb, P = self.na, self.nb, self.P
        ca, cb = P.cn
        if s < 2:
            I = 0 if s == 0 else na + 1
            step = 1 if s == 0 else -1
            f = 0 if s == 0 else na
            J = np.arange(1, nb + 1)
            nB, nP, nP2 = I * (nb + 2) + J, (I + step) * (nb + 2) + J, (I + 2 * step) * (nb + 2) + J
            A, Aopp, dl, D = self.Aa[f], self.Aa[f + step], self.da[f], self.Dfa[k][f]
            xn = self.ap[[I, I + step, I + 2 * step]]
            along = self.bc
            env = {**P.params, ca: np.full(nb, self.af[f]), cb: self.bc}
            F_out = -self.Fa[0] if s == 0 else self.Fa[na]
        else:
            J = 0 if s == 2 else nb + 1
            step = 1 if s == 2 else -1
            f = 0 if s == 2 else nb
            I = np.arange(1, na + 1)
            nB, nP, nP2 = I * (nb + 2) + J, I * (nb + 2) + J + step, I * (nb + 2) + J + 2 * step
            A, Aopp, dl, D = self.Ab[:, f], self.Ab[:, f + step], self.db[:, f], self.Dfb[k][:, f]
            xn = self.bp[[J, J + step, J + 2 * step]]
            along = self.ac
            env = {**P.params, ca: self.ac, cb: np.full(na, self.bf[f])}
            F_out = -self.Fb[:, 0] if s == 2 else self.Fb[:, nb]
        degenerate = A <= 1e-14 * max(float(np.max(Aopp)), 1e-300)
        return {"nB": nB, "nP": nP, "nP2": nP2, "A": A, "Aeff": np.where(degenerate, Aopp, A), "d": dl, "D": D,
                "xn": xn, "along": along, "env": env, "deg": degenerate, "F_out": F_out}

    def assemble(self):
        P, na, nb, M = self.P, self.na, self.nb, self.M
        rows, cols, vals = [], [], []
        c0 = np.zeros(P.nv * M)

        def add(r, c, v):
            rows.append(np.asarray(r).ravel())
            cols.append(np.asarray(c).ravel())
            vals.append(np.asarray(v, dtype=float).ravel())

        nid = lambda I, J: I * (nb + 2) + J
        fa, ja = np.meshgrid(np.arange(na + 1), np.arange(nb), indexing="ij")
        ib, fb = np.meshgrid(np.arange(na), np.arange(nb + 1), indexing="ij")
        aL, aR = nid(fa, ja + 1), nid(fa + 1, ja + 1)
        bL, bR = nid(ib + 1, fb), nid(ib + 1, fb + 1)
        # face interpolation weight of the "right" node (boundary faces always take the boundary node)
        if self.scheme == "central":
            wa = ((self.af - self.ap[:-1]) / np.diff(self.ap))[:, None] * np.ones((1, nb))
            wb = np.ones((na, 1)) * ((self.bf - self.bp[:-1]) / np.diff(self.bp))[None, :]
        else:
            wa, wb = (self.Fa < 0).astype(float), (self.Fb < 0).astype(float)
        wa[0], wa[-1], wb[:, 0], wb[:, -1] = 0.0, 1.0, 0.0, 1.0
        self.wa, self.wb = wa, wb
        self.flux_faces = []          # nonlinear boundary rows: (k, node ids, Aeff, env, expr)
        self.bc_info = []             # per variable, per side: arrays used by the post-processing
        self.dir_values = [[] for _ in P.vars]
        self.level_fixed = [False] * P.nv
        for k, e in enumerate(P.eq):
            off = k * M
            for (nL, nR, G, F, w, mL, mR) in ((aL, aR, self.Ga[k], self.Fa, wa, fa >= 1, fa <= na - 1),
                                              (bL, bR, self.Gb[k], self.Fb, wb, fb >= 1, fb <= nb - 1)):
                cL = G - e["fac"] * F * w
                cR = G + e["fac"] * F * (1.0 - w)
                add(nL[mL] + off, nR[mL] + off, cL[mL])
                add(nL[mL] + off, nL[mL] + off, -cL[mL])
                add(nR[mR] + off, nL[mR] + off, cR[mR])
                add(nR[mR] + off, nR[mR] + off, -cR[mR])
            dref = max(float(np.max(self.Dfa[k])), float(np.max(self.Dfb[k])))
            info = []
            for s in range(4):
                sd = self.side(s, k)
                segs = P.bcs[P.sides[s]][P.vars[k]]
                nf = sd["A"].size
                kind = np.full(nf, "", dtype=object)
                kind[sd["deg"]] = "degenerate"
                live = ~sd["deg"]
                sc = sd["Aeff"] / sd["d"]
                where = f"boundary_conditions['{P.sides[s]}']" + (f"['{P.vars[k]}']" if P.nv > 1 else "")
                if np.any(live):
                    if segs is None:
                        raise ValueError(f"{where}: falta la condición de frontera de este lado (tipos: "
                                         + ", ".join(BC_TYPES) + ")")
                    cnt = np.zeros(nf, dtype=int)
                    for g in segs:
                        m = live & (sd["along"] >= g["lo"]) & (sd["along"] < g["hi"])
                        cnt += m
                        kind[m] = g["kind"]
                        if not np.any(m):
                            continue
                        r = sd["nB"][m] + off
                        envm = {n: (v[m] if isinstance(v, np.ndarray) else v) for n, v in sd["env"].items()}
                        cnt_m = int(m.sum())
                        if g["kind"] == "dirichlet":
                            val = self._finite(g["value"](envm, (cnt_m,)), where)
                            add(r, r, sc[m] * dref)
                            c0[r] = -sc[m] * dref * val
                            self.dir_values[k].append(val)
                            self.level_fixed[k] = True
                        elif g["kind"] == "neumann":
                            val = self._finite(g["value"](envm, (cnt_m,)), where)
                            add(r, r, sc[m] * dref)
                            add(r, sd["nP"][m] + off, -sc[m] * dref)
                            c0[r] = -sc[m] * dref * sd["d"][m] * val
                        elif g["kind"] == "robin":
                            h = self._finite(g["h"](envm, (cnt_m,)), where)
                            ue = self._finite(g["u_ext"](envm, (cnt_m,)), where)
                            diag = sc[m] * (sd["D"][m] + sd["d"][m] * h)
                            if np.any(diag == 0.0):
                                raise ValueError(f"{where}: condición Robin con D = 0 y h = 0 en la frontera (sin "
                                                 "sentido); use 'neumann' o 'symmetry'")
                            add(r, r, diag)
                            add(r, sd["nP"][m] + off, -sc[m] * sd["D"][m])
                            c0[r] = -sc[m] * sd["d"][m] * h * ue
                            self.dir_values[k].append(ue[h != 0])
                            self.level_fixed[k] = self.level_fixed[k] or bool(np.any(h != 0))
                        else:
                            if np.any(sd["D"][m] <= 0):
                                raise ValueError(f"{where}: condición de flujo con D = 0 normal a la frontera")
                            add(r, r, sc[m] * sd["D"][m])
                            add(r, sd["nP"][m] + off, -sc[m] * sd["D"][m])
                            self.flux_faces.append({"k": k, "nodes": sd["nB"][m], "A": sd["Aeff"][m], "env": envm,
                                                    "expr": g["value"]})
                            self.level_fixed[k] = self.level_fixed[k] or bool(g["value"].names & set(P.vars))
                    if np.any(live & (cnt == 0)):
                        x = sd["along"][live & (cnt == 0)]
                        raise ValueError(f"{where}: los tramos no cubren todo el lado (sin condición entre "
                                         f"{x.min():.6g} y {x.max():.6g}); revise 'from'/'to'")
                    if np.any(cnt > 1):
                        raise ValueError(f"{where}: hay tramos que se traslapan; revise 'from'/'to'")
                if np.any(sd["deg"]):
                    m = sd["deg"]
                    r = sd["nB"][m] + off
                    xB, x1, x2 = sd["xn"]
                    even = (P.system == "axisymmetric" and s == 0) or (P.system == "spherical_axisymmetric" and s >= 2)
                    if even:      # u = A + B (x - xB)^2 through the two nearest cells (symmetry axis)
                        s1, s2 = (x1 - xB) ** 2, (x2 - xB) ** 2
                        c1, c2 = s2 / (s2 - s1), -s1 / (s2 - s1)
                    else:         # linear extrapolation (the side has collapsed to a point; gradients may be singular)
                        t = (xB - x1) / (x1 - x2)
                        c1, c2 = 1.0 + t, -t
                    add(r, r, sc[m] * dref)
                    add(r, sd["nP"][m] + off, -c1 * sc[m] * dref)
                    add(r, sd["nP2"][m] + off, -c2 * sc[m] * dref)
                sd["kind"] = kind
                info.append(sd)
            self.bc_info.append(info)
            corners = np.array([nid(0, 0), nid(0, nb + 1), nid(na + 1, 0), nid(na + 1, nb + 1)]) + off
            add(corners, corners, np.ones(4))
        n = P.nv * M
        self.L = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(n, n)).tocsr()
        self.c0 = c0
        # interior node ids
        Ii, Jj = np.meshgrid(np.arange(1, na + 1), np.arange(1, nb + 1), indexing="ij")
        self.cell_nodes = nid(Ii, Jj)
        # dependencies of the local nonlinear terms
        self.src_deps = [sorted(P.vars.index(n) for n in (e["src"].names & set(P.vars))) if e["src"] is not None else []
                         for e in P.eq]
        self.nonlinear = any(self.src_deps) or any(f["expr"].names & set(P.vars) for f in self.flux_faces)
        self.truly_nonlinear = False
        if self.nonlinear:      # is the local term really nonlinear in u? compare its Jacobian at two states
            rs = np.random.default_rng(0)
            j1 = self.local_jacobian(rs.uniform(0.2, 1.0, self.shape))
            j2 = self.local_jacobian(rs.uniform(1.0, 3.0, self.shape))
            a1 = abs(j1 - j2).max() if j1.nnz or j2.nnz else 0.0
            self.truly_nonlinear = bool(not np.isfinite(a1) or a1 > 1e-5 * max(abs(j1).max(), abs(j2).max(), 1e-300))
        self.has_local = any(e["src"] is not None for e in P.eq) or bool(self.flux_faces)

    # -- residual -----------------------------------------------------------------------------------------------------
    def source(self, U, k):
        e = self.P.eq[k]
        if e["src"] is None:
            return np.zeros((self.na, self.nb))
        env = dict(self.env_c)
        for m, v in enumerate(self.P.vars):
            env[v] = U[m, 1:-1, 1:-1]
        return e["src"](env, (self.na, self.nb))

    def local(self, U):
        """Nonlinear/local terms: V*R at the cells, Aeff*q at the flux boundary faces."""
        out = np.zeros(U.size)
        Uf = U.reshape(self.P.nv, -1)
        for k in range(self.P.nv):
            if self.P.eq[k]["src"] is not None:
                out[k * self.M + self.cell_nodes.ravel()] = (self.V * self.source(U, k)).ravel()
        for f in self.flux_faces:
            env = dict(f["env"])
            for m, v in enumerate(self.P.vars):
                env[v] = Uf[m, f["nodes"]]
            out[f["k"] * self.M + f["nodes"]] = f["A"] * f["expr"](env, f["nodes"].shape)
        return out

    def local_jacobian(self, U):
        rows, cols, vals = [], [], []
        Uf = U.reshape(self.P.nv, -1)
        for k in range(self.P.nv):
            for m in self.src_deps[k]:
                eps = 1e-7 * (1.0 + np.abs(U[m]))
                Up, Um = U.copy(), U.copy()
                Up[m] += eps
                Um[m] -= eps
                d = self.V * (self.source(Up, k) - self.source(Um, k)) / (2.0 * eps[1:-1, 1:-1])
                rows.append(k * self.M + self.cell_nodes.ravel())
                cols.append(m * self.M + self.cell_nodes.ravel())
                vals.append(d.ravel())
        for f in self.flux_faces:
            for v in f["expr"].names & set(self.P.vars):
                m = self.P.vars.index(v)
                base = {x: Uf[i, f["nodes"]] for i, x in enumerate(self.P.vars)}
                eps = 1e-7 * (1.0 + np.abs(base[v]))
                ep, em = {**f["env"], **base}, {**f["env"], **base}
                ep[v], em[v] = base[v] + eps, base[v] - eps
                d = f["A"] * (f["expr"](ep, eps.shape) - f["expr"](em, eps.shape)) / (2.0 * eps)
                rows.append(f["k"] * self.M + f["nodes"])
                cols.append(m * self.M + f["nodes"])
                vals.append(d)
        n = self.P.nv * self.M
        if not rows:
            return sparse.csr_matrix((n, n))
        return sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()

    def tvd(self, U):
        out = np.zeros(self.shape)
        for k, e in enumerate(self.P.eq):
            da_, db_ = self.tvd_deltas(U[k])
            Ta, Tb = e["fac"] * self.Fa * da_, e["fac"] * self.Fb * db_
            res = out[k, 1:-1, 1:-1]
            res[:-1] -= Ta[1:-1]
            res[1:] += Ta[1:-1]
            res[:, :-1] -= Tb[:, 1:-1]
            res[:, 1:] += Tb[:, 1:-1]
        return out.ravel()

    def tvd_jacobian(self, U):
        """Jacobian of the deferred TVD term (semismooth Newton instead of a slowly converging fixed point)."""
        na, nb, M = self.na, self.nb, self.M
        rows, cols, vals = [], [], []
        for k, e in enumerate(self.P.eq):
            off = k * M
            _, da_ = _tvd_delta(U[k][:, 1:-1], self.ap, self.af, self.Fa, jac=True)
            _, db_ = _tvd_delta(U[k][1:-1, :].T, self.bp, self.bf, self.Fb.T, jac=True)
            if da_ is not None:
                f = np.arange(1, na)[:, None] * np.ones((1, nb), dtype=int)
                J = np.ones((na - 1, 1), dtype=int) * np.arange(1, nb + 1)[None, :]
                T = e["fac"] * self.Fa[1:na]
                for iN, dN in (("iC", "dC"), ("iD", "dD"), ("iU", "dU")):
                    c = da_[iN] * (nb + 2) + J + off
                    for r, sg in ((f * (nb + 2) + J + off, -1.0), ((f + 1) * (nb + 2) + J + off, 1.0)):
                        rows.append(r.ravel()); cols.append(c.ravel()); vals.append((sg * T * da_[dN]).ravel())
            if db_ is not None:
                f = np.arange(1, nb)[:, None] * np.ones((1, na), dtype=int)       # (nb-1, na): transposed layout
                I = np.ones((nb - 1, 1), dtype=int) * np.arange(1, na + 1)[None, :]
                T = e["fac"] * self.Fb.T[1:nb]
                for iN, dN in (("iC", "dC"), ("iD", "dD"), ("iU", "dU")):
                    c = I * (nb + 2) + db_[iN] + off
                    for r, sg in ((I * (nb + 2) + f + off, -1.0), (I * (nb + 2) + f + 1 + off, 1.0)):
                        rows.append(r.ravel()); cols.append(c.ravel()); vals.append((sg * T * db_[dN]).ravel())
        n = self.P.nv * M
        if not rows:
            return sparse.csr_matrix((n, n))
        return sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()

    def tvd_deltas(self, Uk):
        if self.scheme != "tvd":
            return np.zeros((self.na + 1, self.nb)), np.zeros((self.na, self.nb + 1))
        return (_tvd_delta(Uk[:, 1:-1], self.ap, self.af, self.Fa),
                _tvd_delta(Uk[1:-1, :].T, self.bp, self.bf, self.Fb.T).T)

    def residual(self, u):
        r = self.L @ u + self.c0
        U = u.reshape(self.shape)
        if self.has_local:
            r = r + self.local(U)
        if self.scheme == "tvd":
            r = r + self.tvd(U)
        return r

    def solve(self, U0, max_it):
        P = self.P
        u = U0.ravel().copy()
        tvd = self.scheme == "tvd"
        tol = TVD_TOL if tvd else NEWTON_TOL
        iterative = self.nonlinear or tvd
        lu, dscale, last_ratio, norm_prev = None, None, 0.0, None
        history = []
        r = self.residual(u)
        if not np.all(np.isfinite(r)):
            raise ValueError("el término fuente o un flujo de frontera no da un valor finito con la estimación inicial; "
                             "dé initial_guess dentro del dominio de las expresiones (p. ej. concentraciones > 0)")
        it = 0
        converged = False
        while True:
            if lu is None or (iterative and last_ratio > 0.3):
                J = self.L
                if self.nonlinear:
                    J = J + self.local_jacobian(u.reshape(self.shape))
                if tvd:
                    J = J + self.tvd_jacobian(u.reshape(self.shape))
                d = np.abs(J.diagonal())
                dscale = np.where(d > 0, d, 1.0)
                try:
                    lu = splu(J.tocsc())
                except RuntimeError:
                    raise ValueError("el sistema discreto es singular: revise que alguna condición (Dirichlet, Robin) o "
                                     "un término de reacción fije el nivel de cada variable") from None
            uscale = max(float(np.max(np.abs(u))), 1e-300)
            norm = float(np.max(np.abs(r) / dscale)) / uscale if np.any(r) else 0.0
            history.append(norm)
            if norm <= tol and (it > 0 or norm == 0.0):
                converged = True
                break
            if it >= max_it:
                break
            du = lu.solve(-r)
            if not np.all(np.isfinite(du)):
                raise ValueError("el sistema discreto es singular: revise que alguna condición (Dirichlet, Robin) o un "
                                 "término de reacción fije el nivel de cada variable")
            lam = 1.0
            rn0 = float(np.max(np.abs(r) / dscale))
            while True:
                un = u + lam * du
                rn = self.residual(un)
                ok = np.all(np.isfinite(rn))
                if not iterative or (ok and float(np.max(np.abs(rn) / dscale)) <= (1.0 - 1e-4 * lam) * rn0) \
                        or lam < 1e-3:
                    break
                lam *= 0.5
            if not np.all(np.isfinite(rn)):
                raise ValueError("la iteración de Newton salió del dominio de las expresiones (valor no finito); dé un "
                                 "initial_guess más cercano o revise la cinética")
            last_ratio = float(np.max(np.abs(rn) / dscale)) / rn0 if rn0 > 0 else 0.0
            u, r = un, rn
            it += 1
        self.U = u.reshape(self.shape).copy()
        self._fill_corners()
        self.iterations, self.res_norm, self.converged = it, history[-1], converged
        self.linear = not self.truly_nonlinear
        return self

    def _fill_corners(self):
        U, ap, bp = self.U, self.ap, self.bp
        for I, i1, i2 in ((0, 1, 2), (-1, -2, -3)):
            for J, j1, j2 in ((0, 1, 2), (-1, -2, -3)):
                ea = U[:, i1, J] + (U[:, i1, J] - U[:, i2, J]) * (ap[I] - ap[i1]) / (ap[i1] - ap[i2])
                eb = U[:, I, j1] + (U[:, I, j1] - U[:, I, j2]) * (bp[J] - bp[j1]) / (bp[j1] - bp[j2])
                U[:, I, J] = 0.5 * (ea + eb)

    # -- post-processing ----------------------------------------------------------------------------------------------
    def convection_cells(self, k):
        """Integral of f*v.grad(u) over every cell, from the face values (array formula, not the matrix)."""
        U, fac = self.U[k], self.P.eq[k]["fac"]
        out = np.zeros((self.na, self.nb))
        if not self.has_conv:
            return out
        da_, db_ = self.tvd_deltas(U)
        UL, UR = U[:-1, 1:-1], U[1:, 1:-1]
        uf = (1 - self.wa) * UL + self.wa * UR + da_
        out += fac * self.Fa[1:] * (uf[1:] - U[1:-1, 1:-1]) - fac * self.Fa[:-1] * (uf[:-1] - U[1:-1, 1:-1])
        UL, UR = U[1:-1, :-1], U[1:-1, 1:]
        uf = (1 - self.wb) * UL + self.wb * UR + db_
        out += fac * self.Fb[:, 1:] * (uf[:, 1:] - U[1:-1, 1:-1]) - fac * self.Fb[:, :-1] * (uf[:, :-1] - U[1:-1, 1:-1])
        return out

    def side_values(self, s, k):
        sd = self.bc_info[k][s]
        Uf = self.U[k].ravel()
        ub, up = Uf[sd["nB"]], Uf[sd["nP"]]
        grad = (ub - up) / sd["d"]                  # outward normal derivative (two-point, conservative)
        return sd, ub, grad

    def balance(self, k):
        P = self.P
        q_out, conv_out, gross = {}, {}, 0.0
        for s in range(4):
            sd, ub, grad = self.side_values(s, k)
            q_out[P.sides[s]] = float(np.sum(-sd["D"] * sd["A"] * grad))
            gross += float(np.sum(np.abs(sd["D"] * sd["A"] * grad)))
            conv_out[P.sides[s]] = float(np.sum(P.eq[k]["fac"] * sd["F_out"] * ub))
        src_c, conv_c = self.V * self.source(self.U, k), self.convection_cells(k)
        src, conv = float(np.sum(src_c)), float(np.sum(conv_c))
        imb = -sum(q_out.values()) - conv + src
        scale = max(float(np.sum(np.abs(src_c))), gross, float(np.sum(np.abs(conv_c))), 1e-300)
        return {"diffusive_flux_out_by_side": q_out, "convective_flux_out_by_side": conv_out,
                "convection_term_integral": conv, "source_integral": src, "imbalance": imb,
                "imbalance_relative": abs(imb) / scale}

    def measure(self):
        """Flat dict label -> (value, floor for the relative change) and the structured pieces for the report."""
        P, U = self.P, self.U
        ca, cb = P.cn
        q, struct = {}, {"points": [], "derivative_points": [], "averages": [], "mixing_cup": [], "side_fluxes": [],
                         "extrema": {}}
        rng = [max(float(np.ptp(U[k])), 1e-300) for k in range(P.nv)]
        span = [hi - lo for lo, hi in P.lims]
        interp = [RegularGridInterpolator((self.ap, self.bp), U[k]) for k in range(P.nv)]
        for p in P.points:
            item = {ca: p[0], cb: p[1]}
            for k, v in enumerate(P.vars):
                lab = f"{v}({p[0]:g},{p[1]:g})"
                q[lab] = (float(interp[k]([p])[0]), 0.1 * rng[k])
                item[v] = lab
            struct["points"].append(item)
        if P.dpoints:
            for k, v in enumerate(P.vars):
                Ga_, Gb_ = np.gradient(U[k], self.ap, axis=0), np.gradient(U[k], self.bp, axis=1)
                # on the boundary nodes: the conservative two-point normal derivative (second order as a flux)
                Ga_[0], Ga_[-1] = (U[k, 1] - U[k, 0]) / (self.ap[1] - self.ap[0]), (U[k, -1] - U[k, -2]) / (self.ap[-1] - self.ap[-2])
                Gb_[:, 0] = (U[k, :, 1] - U[k, :, 0]) / (self.bp[1] - self.bp[0])
                Gb_[:, -1] = (U[k, :, -1] - U[k, :, -2]) / (self.bp[-1] - self.bp[-2])
                ga = RegularGridInterpolator((self.ap, self.bp), Ga_)
                gb = RegularGridInterpolator((self.ap, self.bp), Gb_)
                for i, p in enumerate(P.dpoints):
                    if k == 0:
                        struct["derivative_points"].append({ca: p[0], cb: p[1]})
                    for g, c, sp in ((ga, ca, span[0]), (gb, cb, span[1])):
                        lab = f"d{v}_d{c}({p[0]:g},{p[1]:g})"
                        q[lab] = (float(g([p])[0]), 0.1 * rng[k] / sp)
                        struct["derivative_points"][i][f"d{v}_d{c}"] = lab
        named = dict(P.params)
        vol = float(np.sum(self.V))
        env = dict(self.env_c)
        for m, v in enumerate(P.vars):
            env[v] = U[m, 1:-1, 1:-1]
        for n, e in P.averages:
            val = e(env, (self.na, self.nb))
            self._finite(val, f"outputs['averages'] '{n}'")
            integ = float(np.sum(self.V * val))
            q[f"average:{n}"], q[f"integral:{n}"] = (integ / vol, 0.0), (integ, 0.0)
            named[n], named[n + "_integral"] = integ / vol, integ
            struct["averages"].append({"name": n, "expression": e.text, "average": f"average:{n}",
                                       "integral": f"integral:{n}"})
        for mc in P.mixing:
            ax = mc["axis"]
            xp = self.ap if ax == 0 else self.bp
            for at in mc["at"]:
                i = int(min(max(np.searchsorted(xp, at) - 1, 0), xp.size - 2))
                t = (at - xp[i]) / (xp[i + 1] - xp[i])
                if ax == 0:
                    sec = [(1 - t) * U[m, i, 1:-1] + t * U[m, i + 1, 1:-1] for m in range(P.nv)]
                    senv = {**P.params, ca: np.full(self.nb, at), cb: self.bc}
                else:
                    sec = [(1 - t) * U[m, 1:-1, i] + t * U[m, 1:-1, i + 1] for m in range(P.nv)]
                    senv = {**P.params, ca: self.ac, cb: np.full(self.na, at)}
                shape = sec[0].shape
                area = self.sec_w[ax](at)
                wexp = mc["weight"] if mc["weight"] is not None else P.vel[ax]
                w = self._finite(wexp(senv, shape), f"mixing_cup '{mc['name']}' (peso)") * area
                phi = self._finite(mc["expr"]({**senv, **dict(zip(P.vars, sec))}, shape), f"mixing_cup '{mc['name']}'")
                if abs(np.sum(w)) <= 1e-12 * max(float(np.sum(np.abs(w))), 1e-300) or np.sum(area) <= 0:
                    raise ValueError(f"outputs['mixing_cup'] '{mc['name']}': el caudal neto a través de la sección "
                                     f"{P.cn[ax]} = {at:g} es cero; el promedio de mezcla no está definido")
                lab, lab2 = f"mixing_cup:{mc['name']}@{P.cn[ax]}={at:g}", f"area_average:{mc['name']}@{P.cn[ax]}={at:g}"
                q[lab] = (float(np.sum(w * phi) / np.sum(w)), 0.0)
                q[lab2] = (float(np.sum(area * phi) / np.sum(area)), 0.0)
                struct["mixing_cup"].append({"name": mc["name"], "expression": mc["expr"].text, P.cn[ax]: at,
                                             "mixing_cup": lab, "area_average": lab2,
                                             "flow_through_section": float(np.sum(w))})
                if mc["scalar"]:
                    named[mc["name"]], named[mc["name"] + "_area_average"] = q[lab][0], q[lab2][0]
        for fl in P.fluxes:
            sd, ub, grad = self.side_values(fl["side"], fl["k"])
            if fl["at"] is not None:        # local values, interpolated along the side between face centres
                n, at = fl["name"], fl["at"]
                loc = lambda arr: float(np.interp(at, sd["along"], arr))
                vals = {"flux_out": loc(-sd["D"] * grad), "normal_derivative": loc(grad), "value": loc(ub)}
                item = {"name": n, "side": P.sides[fl["side"]], "variable": P.vars[fl["k"]], "at": at}
                for key, val in vals.items():
                    q[f"{key}:{n}"] = (val, 0.1 * rng[fl["k"]] if key == "value" else 0.0)
                    item[key] = f"{key}:{n}"
                named.update({n: vals["flux_out"], n + "_gradient": vals["normal_derivative"], n + "_mean": vals["value"]})
                struct["side_fluxes"].append(item)
                continue
            m = (sd["along"] >= fl["lo"]) & (sd["along"] < fl["hi"])
            area = float(np.sum(sd["A"][m]))
            n = fl["name"]
            if area <= 0:
                raise ValueError(f"outputs['side_fluxes'] '{n}': el tramo pedido del lado '{P.sides[fl['side']]}' no "
                                 "tiene área (lado degenerado o 'from'/'to' fuera del lado)")
            Q = float(np.sum(-sd["D"][m] * sd["A"][m] * grad[m]))
            vals = {"mean_flux_out": Q / area, "integral_flux_out": Q,
                    "mean_normal_derivative": float(np.sum(sd["A"][m] * grad[m])) / area,
                    "mean_value": float(np.sum(sd["A"][m] * ub[m])) / area}
            item = {"name": n, "side": P.sides[fl["side"]], "variable": P.vars[fl["k"]], "area": area}
            for key, val in vals.items():
                q[f"{key}:{n}"] = (val, 0.1 * rng[fl["k"]] if key == "mean_value" else 0.0)
                item[key] = f"{key}:{n}"
            named.update({n: vals["mean_flux_out"], n + "_integral": Q, n + "_gradient": vals["mean_normal_derivative"],
                          n + "_mean": vals["mean_value"]})
            struct["side_fluxes"].append(item)
        struct["derived"] = {}
        for n, e in P.derived:
            q[f"derived:{n}"] = (float(e(named)), 0.0)
            struct["derived"][n] = f"derived:{n}"
        A2, B2 = np.meshgrid(self.ap, self.bp, indexing="ij")
        mask = np.ones(A2.shape, dtype=bool)
        mask[[0, 0, -1, -1], [0, -1, 0, -1]] = False
        for k, v in enumerate(P.vars):
            ext = {}
            for name, fn in (("min", np.argmin), ("max", np.argmax)):
                i = fn(np.where(mask, U[k], np.inf if name == "min" else -np.inf))
                I, J = np.unravel_index(i, A2.shape)
                q[f"{name}:{v}"] = (float(U[k, I, J]), 0.1 * rng[k])
                ext[name] = {"value": f"{name}:{v}", ca: float(A2[I, J]), cb: float(B2[I, J])}
            struct["extrema"][v] = ext
        return q, struct


# ---------------------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------------------
def _compare(vals, floor, tol):
    """Convergence record of one quantity from its values on the three grids (coarse, base, fine)."""
    c, b, f = vals
    d1, d2 = b - c, f - b
    rec = {"value": f, "base_grid": b, "coarse_grid": c, "change": abs(d2)}
    scale = max(abs(f), floor)
    rel = abs(d2) / scale if scale > 0 else 0.0
    rec["relative_change"] = rel
    order = None
    if abs(d2) > 0 and abs(d1) > 0 and d1 * d2 > 0:
        order = math.log(abs(d1) / abs(d2), 2.0)
    rec["observed_order"] = order
    rec["grid_converged"] = bool(rel <= tol)
    if not rec["grid_converged"]:
        if abs(d2) >= 0.9 * abs(d1):
            rec["note"] = ("no converge con la malla: la diferencia entre mallas no disminuye al refinar (valor "
                           "posiblemente no acotado, p. ej. un gradiente en una esquina singular); no usar este número")
        else:
            rec["note"] = "aún cambia con la malla más que la tolerancia; aumente grid['n'] o gradúe la malla"
    elif order is not None and 0.8 <= order <= 3.5 and abs(d2) > 1e-13 * scale:
        rec["richardson_extrapolated"] = f + d2 / (2.0 ** order - 1.0)
    return rec


def solve_pde_2d(coordinates: str, domain: dict, equations: dict, boundary_conditions: dict,
                 velocity: dict | None = None, parameters: dict | None = None, outputs: dict | None = None,
                 grid: dict | None = None, initial_guess: dict | None = None, convection_scheme: str = "auto",
                 refinement_tol: float = 0.01, max_iterations: int = 60) -> dict:
    """Solve the steady 2-D problem and return the requested quantities with their grid-refinement evidence.

    See the module docstring for the equation and the agent tool (`packages.dwsim_client.tools_pde2d`) for the full
    description of the arguments. Raises ValueError with a Spanish message on bad input.
    """
    t0 = time.perf_counter()
    P = _Problem(coordinates, domain, equations, boundary_conditions, velocity, parameters, outputs, grid,
                 initial_guess, convection_scheme, refinement_tol, max_iterations)
    na, nb = P.n
    sizes = [(na // 2, nb // 2), (na, nb), (2 * na, 2 * nb)]
    scheme = P.scheme_req
    pe_base = _Level(P, na, nb, "upwind").peclet()
    if scheme == "auto":
        scheme = "central" if pe_base <= PECLET_CENTRAL_LIMIT else "tvd"
    if pe_base == 0.0:
        scheme = "central"
    levels, prev = [], None
    for (ma, mb) in sizes:
        lv = _Level(P, ma, mb, scheme)
        lv.assemble()
        for k, v in enumerate(P.vars):
            if not lv.level_fixed[k] and not lv.src_deps[k]:
                raise ValueError(f"el nivel de '{v}' no queda determinado: todas sus condiciones son de Neumann/simetría "
                                 "o de flujo fijo y la fuente no depende de la variable; dé al menos una condición "
                                 "Dirichlet o Robin")
        U0 = np.zeros(lv.shape)
        A2, B2 = np.meshgrid(lv.ap, lv.bp, indexing="ij")
        for k, v in enumerate(P.vars):
            if prev is not None:
                U0[k] = RegularGridInterpolator((prev.ap, prev.bp), prev.U[k])(np.stack([A2, B2], axis=-1))
            elif v in P.guess:
                U0[k] = lv._finite(P.guess[v]({**P.params, P.cn[0]: A2, P.cn[1]: B2}, A2.shape), f"initial_guess['{v}']")
            elif lv.dir_values[k] and sum(x.size for x in lv.dir_values[k]):
                U0[k] = float(np.mean(np.concatenate(lv.dir_values[k])))
        lv.solve(U0, P.max_it)
        levels.append(lv)
        prev = lv
    fine = levels[-1]
    measured = [lv.measure() for lv in levels]
    struct = measured[-1][1]
    conv = {lab: _compare([m[0][lab][0] for m in measured], measured[-1][0][lab][1], P.tol) for lab in measured[-1][0]}

    def rec(lab):
        r = conv[lab]
        out = {"value": r["value"], "relative_change_on_refinement": r["relative_change"],
               "grid_converged": r["grid_converged"]}
        if "note" in r:
            out["note"] = r["note"]
        return out

    ca, cb = P.cn
    res: dict = {"coordinates": P.system, "variables": list(P.vars)}
    res["points"] = [{ca: it[ca], cb: it[cb], **{v: rec(it[v]) for v in P.vars}} for it in struct["points"]]
    res["derivative_points"] = [{ca: it[ca], cb: it[cb], **{k: rec(v) for k, v in it.items() if k not in (ca, cb)}}
                                for it in struct["derivative_points"]]
    res["averages"] = {it["name"]: {"expression": it["expression"], "average": rec(it["average"]),
                                    "integral": rec(it["integral"])} for it in struct["averages"]}
    res["mixing_cup"] = [{"name": it["name"], "expression": it["expression"],
                          **{k: v for k, v in it.items() if k in (ca, cb)}, "mixing_cup": rec(it["mixing_cup"]),
                          "area_average": rec(it["area_average"]), "flow_through_section": it["flow_through_section"]}
                         for it in struct["mixing_cup"]]
    fkeys = ("mean_flux_out", "integral_flux_out", "mean_normal_derivative", "mean_value", "flux_out",
             "normal_derivative", "value")
    res["side_fluxes"] = {it["name"]: {**{k: v for k, v in it.items() if k in ("side", "variable", "area", "at")},
                                       **{k: rec(it[k]) for k in fkeys if k in it}}
                          for it in struct["side_fluxes"]}
    res["derived"] = {n: rec(lab) for n, lab in struct["derived"].items()}
    res["extrema"] = {v: {name: {**{k: x for k, x in e.items() if k != "value"}, **rec(e["value"])}
                          for name, e in ext.items()} for v, ext in struct["extrema"].items()}
    res["domain_measure"] = float(np.sum(fine.V))

    not_conv = [lab for lab, r in conv.items() if not r["grid_converged"]]
    balance = {v: fine.balance(k) for k, v in enumerate(P.vars)}
    div = 0.0
    if fine.has_conv:
        net = fine.Fa[1:] - fine.Fa[:-1] + fine.Fb[:, 1:] - fine.Fb[:, :-1]
        gross = np.abs(fine.Fa[1:]) + np.abs(fine.Fa[:-1]) + np.abs(fine.Fb[:, 1:]) + np.abs(fine.Fb[:, :-1])
        div = float(np.max(np.abs(net)) / max(float(np.max(gross)), 1e-300))
    scheme_text = {"central": "diferencias centrales (interpolación lineal en las caras, 2.º orden)",
                   "upwind": "upwind de primer orden (difusión numérica: comparar con 'tvd')",
                   "tvd": "TVD de van Leer por corrección diferida sobre upwind (2.º orden en zonas suaves, acotado)"}
    warnings = []
    if not all(lv.converged for lv in levels):
        warnings.append("La iteración (Newton / corrección diferida) NO alcanzó la tolerancia en alguna malla: los "
                        "resultados no son fiables; revise evidence.nonlinear.")
    if not_conv:
        warnings.append("Cantidades que NO han convergido con la malla (cambio relativo entre la malla base y la "
                        f"refinada > {P.tol:g}); no presentarlas como resultado: " + "; ".join(not_conv[:12])
                        + (" …" if len(not_conv) > 12 else ""))
    if fine.has_conv and div > 1e-3:
        warnings.append(f"El campo de velocidad dado no es solenoidal en la malla (divergencia relativa {div:.2g}); "
                        "se resolvió v·grad(u), que no equivale a div(v·u).")
    if scheme == "upwind" and pe_base > PECLET_CENTRAL_LIMIT:
        warnings.append("Upwind de primer orden con Péclet de celda alto: hay difusión numérica; compruebe con 'tvd'.")
    if scheme == "central" and pe_base > PECLET_CENTRAL_LIMIT:
        warnings.append(f"Diferencias centrales con Péclet de celda {pe_base:.3g} > 2: pueden aparecer oscilaciones; "
                        "use 'tvd' o refine.")
    for ax, c in enumerate(P.cn):
        if P.grading[ax].get("default") and P.grading[ax]["type"] != "uniform":
            warnings.append(f"Malla graduada por omisión en {c} ({P.grading[ax]['type']}): "
                            + ("concentrada hacia r = 0, donde el sector puede tener gradientes singulares."
                               if P.grading[ax]["type"] == "power" else "espaciado geométrico entre r mín y r máx."))
    if fine.truly_nonlinear:
        warnings.append("Problema no lineal: la solución hallada depende de initial_guess si hay multiplicidad de "
                        "estados estacionarios.")
    res["converged"] = bool(all(lv.converged for lv in levels))
    res["grid_converged"] = not not_conv
    res["not_grid_converged"] = not_conv
    res["evidence"] = {
        "grid_refinement": {
            "grids": [list(s) for s in sizes], "reported_from": list(sizes[-1]), "tolerance_relative": P.tol,
            "grading": {c: {k: v for k, v in P.grading[ax].items() if k != "default"} for ax, c in enumerate(P.cn)},
            "smallest_cell": {ca: float(np.min(np.diff(fine.af))), cb: float(np.min(np.diff(fine.bf)))},
            "quantities": conv,
        },
        "nonlinear": {"linear_problem": bool(fine.linear),
                      "iterations_by_grid": [lv.iterations for lv in levels],
                      "residual_norm_by_grid": [lv.res_norm for lv in levels],
                      "residual_definition": "máx |residuo_i| / (|diagonal_i| · máx|u|)",
                      "tolerance": TVD_TOL if scheme == "tvd" else NEWTON_TOL,
                      "converged_by_grid": [bool(lv.converged) for lv in levels]},
        "global_balance": {**balance,
                           "definition": "−Σ flujo difusivo saliente − ∫ f·v·grad(u) dV + ∫ R dV = imbalance, en la "
                                         "malla refinada (relativo a la suma de valores absolutos), con fórmulas de caras independientes de la matriz "
                                         "ensamblada (verifica ensamblado y solución; el error de malla lo mide "
                                         "grid_refinement)"},
        "convection": {"scheme": scheme, "scheme_description": scheme_text[scheme] if fine.has_conv else "sin convección",
                       "max_cell_peclet_base_grid": pe_base, "max_cell_peclet_fine_grid": fine.peclet(),
                       "velocity_divergence_relative": div},
        "elapsed_s": time.perf_counter() - t0,
    }
    res["warnings"] = warnings
    res["assumptions"] = [
        "Estado estacionario; D depende solo de las coordenadas y parámetros (no de la solución).",
        "Volúmenes finitos centrados en celda sobre malla estructurada; flujo difusivo de dos puntos; valores "
        "reportados de la malla refinada (2n).",
        "Flujos de frontera: fórmula conservativa de dos puntos (frontera–primera celda).",
        "Derivadas en puntos: diferencias de 2.º orden sobre los valores de celda, interpoladas bilinealmente; son "
        "derivadas respecto de la coordenada (la componente física tangencial en r-theta es (1/r)·du/dtheta).",
        {"cartesian": "Integrales y flujos por unidad de profundidad.",
         "axisymmetric": "Integrales y flujos sobre la vuelta completa (2·pi).",
         "polar": "Integrales y flujos por unidad de profundidad.",
         "spherical_axisymmetric": "Integrales y flujos sobre la vuelta completa (2·pi) alrededor del eje."}[P.system],
    ]
    res["not_included"] = ["transitorio", "geometrías no rectangulares en las coordenadas elegidas", "periodicidad en "
                           "theta (disco completo: usar la mitad con simetría)", "difusividad dependiente de la solución",
                           "resolución del campo de velocidad (se da como expresión)"]
    return res
