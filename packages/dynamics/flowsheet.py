"""Dynamic flowsheet simulator, independent of DWSIM: first-principles units with holdup connected by streams, solved as a
stiff ODE system with scipy (`solve_ivp`, Radau by default). Start-up, shutdown, disturbances and control loops.

What is solved (states in conservative form, so a tank may start empty or run dry):
  * liquid holdup (tank, mixing tank, heated/jacketed tank, CSTR, each cell of a lumped heat exchanger):
        dV/dt       = Σ q_in − Σ q_out
        d(V c_i)/dt = Σ q_in c_in,i − Σ q_out c_i + V Σ_r ν_ir r_r
        d(V T)/dt   = Σ q_in T_in − Σ q_out T + [Q + UA_j (T_j − T) + Σ links + V Σ_r (−ΔH_r) r_r]/(ρ cp)
    with constant ρ and cp (the classical stirred-tank model of Seborg, Luyben and Bequette);
  * gas vessel: dm/dt = Σ ṁ, dU/dt = Σ ṁ_in h_in − Σ ṁ_out h + UA (T_amb − T); ideal gas or CoolProp;
  * stream flow laws of `packages.dynamics.units`; a pipe with inertia adds one velocity state;
  * signal blocks: first-order lag and exact dead time (method of steps on the dense solution, no Padé);
  * PID (ideal ISA form, derivative on the filtered measurement) with back-calculation anti-windup.

Scheduled events (steps and ramps of any parameter) are integration restarts: the integrator never steps over one.
Hard limits of a tank (overflow at the declared maximum level, running dry) are located with root-finding events and
switch the tank between the modes normal / full (spilling) / empty (outflow limited to the inflow).

Evidence of each run: global volume, component and energy balances over the transient (boundary and source integrals
are integrated together with the state), residual of the final state, a second run with tolerances 100 times tighter,
and the list of scheduled events with the restart that hit each one.

Nothing physical has a default: volumes, areas, valve coefficients, pump curves, kinetics, UA, densities, heat capacities,
controller settings and initial conditions are the caller's. Numerical settings (tolerances, sample counts, bands) do."""
from __future__ import annotations

import bisect
import math

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

from packages.dynamics import units as U

SETTLE_BAND = 0.02           # ±2 % of the change after the last event (same band as packages.dynamics.loop)
BALANCE_TOL = 1e-6           # relative closure of the global balances
REFINE_TOL = 1e-4            # max difference between the run and the tight-tolerance run, relative to each output's span
STEADY_RATE_TOL = 1e-3       # normalised change over the time since the last event below which the end state is «steady»
EMPTY_SQRT = 1e-6            # a tank is empty when √(V/V_scale) falls to this (V/V_scale = 1e-12)
GAS_EMPTY = 1e-9             # a gas vessel with a fixed-flow outlet is empty when m/m0 falls to this
MAX_MODE_SWITCHES = 400
N_FINE = 2001
METHODS = ("Radau", "BDF", "LSODA")
TANK_TYPES = ("tank", "mixing_tank", "heated_tank", "cstr")


# ----------------------------------------------------------------------------------------------------------- helpers
def _req(d: dict, key: str, where: str, purpose: str):
    if not isinstance(d, dict) or d.get(key) is None:
        raise ValueError(f"falta '{key}' en {where}: se necesita para {purpose}")
    return d[key]


def _num(x, where: str, positive: bool = False, nonneg: bool = False) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{where} debe ser un número (se recibió {x!r})") from None
    if not math.isfinite(v):
        raise ValueError(f"{where} debe ser finito")
    if positive and v <= 0:
        raise ValueError(f"{where} debe ser > 0")
    if nonneg and v < 0:
        raise ValueError(f"{where} debe ser ≥ 0")
    return v


def clean(x):
    """JSON-safe copy: numpy scalars to Python, NaN to None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


class _Obj:
    """Plain attribute bag for compiled units, streams, blocks and controllers."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# ------------------------------------------------------------------------------------------------------------- model
class Model:
    """Compiled flowsheet: state layout, parameter table, right-hand side, events and variable getters."""

    def __init__(self, fs: dict, overrides: dict | None = None):
        if not isinstance(fs, dict) or not fs.get("units"):
            raise ValueError("falta 'units' en el flowsheet: lista de equipos con retención (tank, cstr, heat_exchanger, "
                             "gas_vessel…)")
        self.assumptions: list[str] = []
        self.warnings: list[str] = []
        self.comp = [str(c) for c in (fs.get("components") or [])]
        self.nc = len(self.comp)
        self.liquid = fs.get("liquid") or {}
        self.P: dict[str, float] = {}
        self.holdups: list[_Obj] = []
        self.vessels: list[_Obj] = []
        self.streams: list[_Obj] = []
        self.links: list[_Obj] = []
        self.blocks: list[_Obj] = []
        self.controllers: list[_Obj] = []
        self.flashes: list[_Obj] = []
        self.columns: list = []
        self.verbose: set[str] = set()                       # valid outputs left out of the default list (per stage)
        self.ports: dict[str, tuple[_Obj, _Obj]] = {}        # name usable in from/to -> (inlet unit, outlet unit)
        self.alias: dict[str, str] = {}
        self.fractions: set[str] = set()                     # parameters bounded to [0, 1] (valve openings)
        self.names: list[str] = []
        y0: list[float] = []
        scale: list[float] = []
        self._y0, self._scale = y0, scale
        ids = set()
        for u in fs["units"]:
            uid = str(_req(u, "id", "un equipo de 'units'", "identificarlo en corrientes, eventos y salidas"))
            if uid in ids:
                raise ValueError(f"identificador repetido: '{uid}'")
            ids.add(uid)
            typ = str(_req(u, "type", f"el equipo '{uid}'", "saber qué modelo usar")).lower()
            if typ in TANK_TYPES:
                h = self._add_holdup(uid, u, u.get("initial") or {})
                self.ports[uid] = (h, h)
            elif typ == "heat_exchanger":
                self._add_exchanger(uid, u)
            elif typ == "gas_vessel":
                self._add_vessel(uid, u)
            elif typ == "flash_drum":
                self._add_flash(uid, u)
            elif typ == "distillation_column":
                from packages.dynamics import column as COL
                self.columns.append(COL.compile_column(self, uid, u))
            else:
                raise ValueError(f"tipo de equipo no soportado en '{uid}': '{typ}'. Tipos: {', '.join(TANK_TYPES)}, "
                                 "heat_exchanger, gas_vessel, flash_drum, distillation_column")
        self.unit_by_id = {h.id: h for h in self.holdups} | {g.id: g for g in self.vessels} | \
            {f.id: f for f in self.flashes}
        for lk in fs.get("heat_links") or []:
            self._add_link(lk)
        for s in fs.get("streams") or []:
            self._add_stream(s)
        self._check_topology()
        self.getters: dict[str, callable] = {}
        self.pure: dict[str, callable] = {}
        self._register_unit_variables()
        for b in fs.get("blocks") or []:
            self._add_block(b)
        for c in fs.get("controllers") or []:
            self._add_controller(c)
        self.driven = {c.target: f"el controlador '{c.id}'" for c in self.controllers}
        self.driven.update({b.target: f"el bloque '{b.id}'" for b in self.blocks if b.target is not None})
        self.driven.update({f.kP: f"el equilibrio del tambor '{f.id}'" for f in self.flashes})
        for key, val in (overrides or {}).items():         # base value of a parameter replaced by the caller (step test)
            if key not in self.P:
                raise ValueError(f"'{key}' no es un parámetro del flowsheet")
            self.P[key] = _num(val, f"el valor impuesto a '{key}'")
        self.n_dyn = len(y0)
        self._add_quadratures()
        self.y0 = np.array(y0, dtype=float)
        self.scale = np.array(scale, dtype=float)
        self.t0 = 0.0
        self.sched_now: list[tuple[str, float, float]] = []
        self.hist_t: list[float] = []
        self.hist_sol: list = []
        self.delays = sorted({b.theta for b in self.blocks if b.theta > 0})
        self.no_delay = False
        self.oor = ""
        self._order_flows()
        self.finish()
        self._init_signals()

    # ------------------------------------------------------------------------------------------------ state helpers
    def _state(self, name: str, value: float, scale: float) -> int:
        self.names.append(name)
        self._y0.append(float(value))
        self._scale.append(float(max(abs(scale), 1e-300)))
        return len(self._y0) - 1

    def _liquid_prop(self, u: dict, key: str) -> float | None:
        v = u.get(key, self.liquid.get(key))
        return None if v is None else _num(v, f"'{key}'", positive=True)

    # ------------------------------------------------------------------------------------------------- liquid units
    def _add_holdup(self, uid: str, u: dict, init: dict, hidden: bool = False) -> _Obj:
        nc = self.nc
        const = bool(u.get("constant_volume", False))
        h = _Obj(id=uid, kind="liquid", ports=[], mode="normal", hidden=hidden, reactions=[], links=[], spill=0.0,
                 V=0.0, T=math.nan, c=np.zeros(nc), wet=True, level=None, phi=1.0, net_potential=0.0)
        h.rho = self._liquid_prop(u, "density_kg_m3")
        h.cp = self._liquid_prop(u, "heat_capacity_J_kgK")
        h.area = None if u.get("area_m2") is None else _num(u["area_m2"], f"'area_m2' de '{uid}'", positive=True)
        h.kP = None
        if u.get("pressure_Pa") is not None:
            h.kP = f"{uid}.pressure_Pa"
            self.P[h.kP] = _num(u["pressure_Pa"], f"'pressure_Pa' de '{uid}'", positive=True)
        if const:
            h.Vc = _num(_req(u, "volume_m3", f"el equipo '{uid}'", "la retención de líquido (volumen constante)"),
                        f"'volume_m3' de '{uid}'", positive=True)
            h.iV, h.vmax, v0 = -1, None, h.Vc
        else:
            h.Vc = None
            if h.area is None:
                raise ValueError(f"falta 'area_m2' en el tanque '{uid}': se necesita para convertir volumen en nivel "
                                 "(o declare 'constant_volume': true con 'volume_m3')")
            if init.get("level_m") is not None:
                v0 = _num(init["level_m"], f"'initial.level_m' de '{uid}'", nonneg=True) * h.area
            elif init.get("volume_m3") is not None:
                v0 = _num(init["volume_m3"], f"'initial.volume_m3' de '{uid}'", nonneg=True)
            else:
                raise ValueError(f"falta 'initial.level_m' (o 'initial.volume_m3') en el tanque '{uid}': el estado inicial "
                                 "no se supone (0 = arranque desde vacío)")
            h.vmax = None
            if u.get("max_level_m") is not None:
                h.vmax = _num(u["max_level_m"], f"'max_level_m' de '{uid}'", positive=True) * h.area
            elif u.get("max_volume_m3") is not None:
                h.vmax = _num(u["max_volume_m3"], f"'max_volume_m3' de '{uid}'", positive=True)
            if h.vmax is not None and v0 > h.vmax * (1 + 1e-12):
                raise ValueError(f"el volumen inicial de '{uid}' supera su máximo declarado")
        h.vscale = h.Vc or h.vmax or max(v0, 0.0) or 1.0
        if not const and h.vmax is None and not hidden:
            self.assumptions.append(f"'{uid}': no se dio 'max_level_m'; no se vigila el rebose de ese tanque")
        h.veps = 1e-13 * h.vscale
        if not const:
            h.iV = self._state(f"{uid}.volume_m3", v0, h.vscale)
        # composition
        h.iN = -1
        cscale = 1.0
        if nc:
            ci = _req(init, "concentration_mol_m3", f"'initial' de '{uid}'",
                      "el estado inicial de composición (dé 0 para un componente ausente)")
            c0 = [_num(_req(ci, k, f"'initial.concentration_mol_m3' de '{uid}'", "el estado inicial de composición"),
                       f"concentración inicial de {k} en '{uid}'", nonneg=True) for k in self.comp]
            cscale = max(max(c0), 1e-30)
            h.c0 = np.array(c0)
            h.iN = len(self._y0)
            for k, c in zip(self.comp, c0):
                self._state(f"{uid}.moles_mol.{k}", c * v0, 1.0)      # scale fixed later (needs feed concentrations)
        h.cscale = cscale
        # energy
        h.iE = -1
        h.kT = None
        if init.get("temperature_K") is not None:
            t0 = _num(init["temperature_K"], f"'initial.temperature_K' de '{uid}'", positive=True)
            h.iE = self._state(f"{uid}.VT_m3K", v0 * t0, h.vscale * t0)
            h.T0 = t0
        elif u.get("temperature_K") is not None:
            h.kT = f"{uid}.temperature_K"
            self.P[h.kT] = _num(u["temperature_K"], f"'temperature_K' de '{uid}'", positive=True)
            h.T0 = self.P[h.kT]
        else:
            h.T0 = math.nan
        h.kQ = h.kUA = h.kTj = None
        if u.get("heat_duty_W") is not None:
            h.kQ = f"{uid}.heat_duty_W"
            self.P[h.kQ] = _num(u["heat_duty_W"], f"'heat_duty_W' de '{uid}'")
        jk = u.get("jacket")
        if jk is not None:
            h.kUA, h.kTj = f"{uid}.jacket_UA_W_K", f"{uid}.jacket_temperature_K"
            self.P[h.kUA] = _num(_req(jk, "UA_W_K", f"'jacket' de '{uid}'", "el calor intercambiado con la chaqueta"),
                                 f"'jacket.UA_W_K' de '{uid}'", nonneg=True)
            self.P[h.kTj] = _num(_req(jk, "temperature_K", f"'jacket' de '{uid}'",
                                      "la temperatura del medio de la chaqueta"),
                                 f"'jacket.temperature_K' de '{uid}'", positive=True)
        for k, rx in enumerate(u.get("reactions") or []):
            h.reactions.append(self._compile_reaction(uid, k, rx, h))
        heat_terms = h.kQ or h.kUA or any(r["dH"] != 0.0 for r in h.reactions)
        if heat_terms and h.iE < 0:
            raise ValueError(f"'{uid}' tiene calor (heat_duty_W, jacket o calor de reacción) pero falta "
                             "'initial.temperature_K': se necesita para el balance de energía")
        h.heat_terms = bool(heat_terms)
        h.rhocp = None if (h.rho is None or h.cp is None) else h.rho * h.cp
        self.holdups.append(h)
        return h

    def _compile_reaction(self, uid: str, k: int, rx: dict, h: _Obj) -> dict:
        where = f"la reacción {k + 1} de '{uid}'"
        if not self.nc:
            raise ValueError(f"{where}: declare 'components' en el flowsheet para usar reacciones")
        st = _req(rx, "stoichiometry", where, "los coeficientes estequiométricos (negativos = reactivos)")
        for name in st:
            if name not in self.comp:
                raise ValueError(f"{where}: el componente '{name}' no está en 'components'")
        nu = np.array([float(st.get(c, 0.0)) for c in self.comp])
        orders = _req(rx, "orders", where, "los órdenes de reacción de la ley de potencias (no se suponen)")
        for name in orders:
            if name not in self.comp:
                raise ValueError(f"{where}: el componente '{name}' de 'orders' no está en 'components'")
        od = [(self.comp.index(c), float(n)) for c, n in orders.items() if float(n) != 0.0]
        if rx.get("rate_constant") is not None:
            k0, ea = _num(rx["rate_constant"], f"'rate_constant' de {where}", nonneg=True), 0.0
        else:
            k0 = _num(_req(rx, "pre_exponential", where, "la constante cinética (o dé 'rate_constant')"),
                      f"'pre_exponential' de {where}", nonneg=True)
            ea = _num(_req(rx, "activation_energy_J_mol", where, "la dependencia de Arrhenius con la temperatura"),
                      f"'activation_energy_J_mol' de {where}")
            if ea != 0.0 and h.iE < 0 and h.kT is None:
                raise ValueError(f"{where}: cinética de Arrhenius sin temperatura; dé 'initial.temperature_K' (balance de "
                                 f"energía) o 'temperature_K' del equipo '{uid}' (isotérmico)")
        dh = _num(rx.get("heat_of_reaction_J_mol", 0.0), f"'heat_of_reaction_J_mol' de {where}")
        if h.iE >= 0 and rx.get("heat_of_reaction_J_mol") is None:
            raise ValueError(f"falta 'heat_of_reaction_J_mol' en {where}: se necesita para el balance de energía (dé 0 si "
                             "la reacción es atérmica)")
        return {"id": str(rx.get("id", f"r{k + 1}")), "nu": nu, "orders": od, "k0": k0, "Ea": ea, "dH": dh}

    def _add_exchanger(self, uid: str, u: dict):
        n = int(u.get("cells", 1))
        if n < 1:
            raise ValueError(f"'cells' de '{uid}' debe ser ≥ 1")
        flow = str(_req(u, "flow", f"el intercambiador '{uid}'", "el arreglo de flujos ('countercurrent' o 'cocurrent')"))
        if flow not in ("countercurrent", "cocurrent"):
            raise ValueError(f"'flow' de '{uid}' debe ser 'countercurrent' o 'cocurrent'")
        ua = _num(_req(u, "UA_W_K", f"el intercambiador '{uid}'", "el calor transferido entre los dos lados"),
                  f"'UA_W_K' de '{uid}'", nonneg=True)
        key = f"{uid}.UA_W_K"
        self.P[key] = ua
        init = u.get("initial") or {}
        sides = {}
        for side in ("hot", "cold"):
            sd = _req(u, side, f"el intercambiador '{uid}'", f"la retención y propiedades del lado '{side}'")
            vol = _num(_req(sd, "volume_m3", f"'{side}' de '{uid}'", "la retención de líquido de ese lado"),
                       f"'{side}.volume_m3' de '{uid}'", positive=True)
            si = dict(init.get(side) or {})
            _req(si, "temperature_K", f"'initial.{side}' de '{uid}'", "la temperatura inicial de ese lado")
            cells = []
            for k in range(n):
                cu = {"constant_volume": True, "volume_m3": vol / n, "density_kg_m3": sd.get("density_kg_m3"),
                      "heat_capacity_J_kgK": sd.get("heat_capacity_J_kgK")}
                cell = self._add_holdup(f"{uid}.{side}.{k + 1}", cu, si, hidden=True)
                if cell.rhocp is None:
                    raise ValueError(f"faltan 'density_kg_m3' y 'heat_capacity_J_kgK' del lado '{side}' de '{uid}' (o en "
                                     "'liquid'): se necesitan para el balance de energía del intercambiador")
                cells.append(cell)
            for a, b in zip(cells[:-1], cells[1:]):
                self._raw_stream(f"{a.id}>", a, b, {"type": "overflow"}, hidden=True)
            sides[side] = cells
            self.ports[f"{uid}.{side}"] = (cells[0], cells[-1])
            self.alias[f"{uid}.{side}_outlet_temperature_K"] = f"{cells[-1].id}.temperature_K"
            self.alias[f"{uid}.{side}_inlet_cell_temperature_K"] = f"{cells[0].id}.temperature_K"
        for k in range(n):
            j = n - 1 - k if flow == "countercurrent" else k
            self.links.append(_Obj(id=f"{uid}.link.{k + 1}", a=sides["hot"][k], b=sides["cold"][j], key=key, factor=1.0 / n,
                                   Q=0.0))
        self.assumptions.append(f"'{uid}': intercambiador de {n} celda(s) por lado bien mezcladas, UA repartido por igual, "
                                "sin capacidad térmica de pared ni cambio de fase")

    def _add_link(self, lk: dict):
        lid = str(_req(lk, "id", "un elemento de 'heat_links'", "identificarlo"))
        bt = _req(lk, "between", f"el enlace térmico '{lid}'", "saber qué dos equipos intercambian calor")
        if len(bt) != 2 or any(b not in self.unit_by_id or self.unit_by_id[b].kind != "liquid" for b in bt):
            raise ValueError(f"'between' de '{lid}' debe nombrar dos equipos líquidos existentes")
        a, b = self.unit_by_id[bt[0]], self.unit_by_id[bt[1]]
        for h in (a, b):
            if h.iE < 0 or h.rhocp is None:
                raise ValueError(f"el enlace térmico '{lid}' necesita balance de energía en '{h.id}': dé "
                                 "'initial.temperature_K', 'density_kg_m3' y 'heat_capacity_J_kgK'")
        key = f"{lid}.UA_W_K"
        self.P[key] = _num(_req(lk, "UA_W_K", f"el enlace térmico '{lid}'", "el calor intercambiado"),
                           f"'UA_W_K' de '{lid}'",
                           nonneg=True)
        self.links.append(_Obj(id=lid, a=a, b=b, key=key, factor=1.0, Q=0.0))
        a.heat_terms = b.heat_terms = True

    # --------------------------------------------------------------------------------------------------- flash drum
    def _add_flash(self, uid: str, u: dict):
        """Isothermal flash drum (`packages.dynamics.flash_drum`) as a unit: liquid enters and leaves by the engine's
        streams (molar flows = q·c), the vapour leaves by its own outlet; the drum pressure is a variable that the
        valve and pump laws of its streams read."""
        from packages.dynamics import flash_drum as FD
        if self.nc < 2:
            raise ValueError(f"el tambor flash '{uid}' necesita 'components' (≥ 2) en el flowsheet")
        dr = FD.Drum(u, components=self.comp)
        fl = _Obj(id=uid, kind="liquid", is_flash=True, drum=dr, ports=[], rho=dr.rho, area=dr.area, iV=-1, iE=-1,
                  hidden=False, kT=None, kP=f"{uid}.pressure_Pa", V=0.0, T=dr.T, c=np.zeros(self.nc), wet=True,
                  eq=None, Vflow=0.0, rhocp=None, interacting=False)
        fl.iM = len(self._y0)
        for k, c in enumerate(self.comp):
            self._state(f"{uid}.moles_mol.{c}", dr.M0[k], dr.scale)
        eq0 = dr.equilibrium(dr.M0)
        self.P[fl.kP] = eq0["P"]
        fl.pkeys = {k: f"{uid}.{k}" for k in dr.P}           # vapour.opening, vapour.downstream_pressure_Pa, …
        for k, key in fl.pkeys.items():
            self.P[key] = dr.P[k]
            if k.endswith(".opening"):
                self.fractions.add(key)
        self.flashes.append(fl)
        self.ports[uid] = (fl, fl)
        self.assumptions.append(f"'{uid}': tambor flash isotérmico a {dr.T:g} K con equilibrio instantáneo ("
                                + ("K constantes" if dr.model == "k_values" else "volatilidad relativa constante, Raoult")
                                + "), vapor ideal, líquido de densidad molar constante; las corrientes de líquido llevan "
                                "c = x·ρ_molar; sin balance de energía en el tambor")

    def _flash_state(self, fl: _Obj, y) -> dict:
        return fl.drum.equilibrium(y[fl.iM:fl.iM + self.nc])

    # ---------------------------------------------------------------------------------------------------- gas units
    def _add_vessel(self, uid: str, u: dict):
        vol = _num(_req(u, "volume_m3", f"el recipiente '{uid}'", "el volumen de gas"), f"'volume_m3' de '{uid}'",
                   positive=True)
        gd = _req(u, "gas", f"el recipiente '{uid}'", "el modelo de gas ({'model': 'ideal', …} o {'model': 'coolprop', …})")
        thermal = str(_req(u, "thermal", f"el recipiente '{uid}'",
                           "el modo térmico: 'isothermal', 'adiabatic' o 'heat_transfer' (no se supone)"))
        if thermal not in ("isothermal", "adiabatic", "heat_transfer"):
            raise ValueError(f"'thermal' de '{uid}' debe ser 'isothermal', 'adiabatic' o 'heat_transfer'")
        model = str(_req(gd, "model", f"'gas' de '{uid}'", "elegir gas ideal o CoolProp"))
        if model == "ideal":
            mm = _num(_req(gd, "molar_mass_kg_mol", f"'gas' de '{uid}'", "la densidad del gas ideal"),
                      f"'molar_mass_kg_mol' de '{uid}'", positive=True)
            k = gd.get("heat_capacity_ratio")
            if k is not None and _num(k, f"'heat_capacity_ratio' de '{uid}'") <= 1.0:
                raise ValueError(f"'heat_capacity_ratio' de '{uid}' debe ser > 1")
            gas = U.IdealGas(mm, None if k is None else float(k))
        elif model == "coolprop":
            gas = U.CoolPropGas(str(_req(gd, "fluid", f"'gas' de '{uid}'", "el fluido de CoolProp")))
            self.assumptions.append(f"'{uid}': gas real {gas.fluid} con la ecuación de estado de Helmholtz de CoolProp "
                                    f"{gas.version}; el orificio usa la fórmula de tobera isentrópica de gas ideal con la "
                                    "densidad real aguas arriba y cp/cv real (aproximación, no validada frente a datos)")
        else:
            raise ValueError(f"'gas.model' de '{uid}' debe ser 'ideal' o 'coolprop'")
        init = u.get("initial") or {}
        p0 = _num(_req(init, "pressure_Pa", f"'initial' de '{uid}'", "la masa inicial de gas"),
                  f"'initial.pressure_Pa' de '{uid}'", positive=True)
        t0 = _num(_req(init, "temperature_K", f"'initial' de '{uid}'", "la masa inicial de gas"),
                  f"'initial.temperature_K' de '{uid}'", positive=True)
        if thermal != "isothermal" and model == "ideal" and gas.k is None:
            raise ValueError(f"falta 'gas.heat_capacity_ratio' en '{uid}': se necesita para el balance de energía del gas")
        rho0, u0, h0 = gas.from_TP(t0, p0)
        g = _Obj(id=uid, kind="gas", gas=gas, V=vol, thermal=thermal, ports=[], kUA=None, kTa=None, kT=None, iU=-1,
                 mode="normal", net=0.0, net_potential=0.0, fixed_out=False,
                 P=p0, T=t0, h=0.0, k=1.4, m=rho0 * vol, Qamb=0.0, m0=rho0 * vol,
                 escale=max(abs(u0), abs(h0), p0 / rho0) * rho0 * vol)
        g.im = self._state(f"{uid}.mass_kg", rho0 * vol, rho0 * vol)
        if thermal == "isothermal":
            g.kT = f"{uid}.temperature_K"
            self.P[g.kT] = t0
            self.assumptions.append(f"'{uid}': recipiente isotérmico a la temperatura inicial (intercambio de calor ideal "
                                    "con el entorno)")
        else:
            g.iU = self._state(f"{uid}.internal_energy_J", u0 * rho0 * vol, max(abs(u0), p0 / rho0) * rho0 * vol)
            if thermal == "heat_transfer":
                g.kUA, g.kTa = f"{uid}.UA_W_K", f"{uid}.ambient_temperature_K"
                self.P[g.kUA] = _num(_req(u, "UA_W_K", f"el recipiente '{uid}'", "el calor intercambiado con el ambiente"),
                                     f"'UA_W_K' de '{uid}'", nonneg=True)
                self.P[g.kTa] = _num(_req(u, "ambient_temperature_K", f"el recipiente '{uid}'", "la temperatura ambiente"),
                                     f"'ambient_temperature_K' de '{uid}'", positive=True)
        self.vessels.append(g)
        self.ports[uid] = (g, g)

    # ------------------------------------------------------------------------------------------------------ streams
    def _raw_stream(self, sid: str, frm, to, law: dict, hidden: bool = False) -> _Obj:
        s = _Obj(id=sid, frm=frm, to=to, law=law["type"], hidden=hidden, q=0.0, mdot=0.0, gas=False, cs=None, Ts=math.nan,
                 hs=0.0, iv=-1, end_of_curve=False, interacting=False, kT=None, kc=None)
        self.streams.append(s)
        if frm is not None:
            frm.ports.append((s, -1.0))
        if to is not None:
            to.ports.append((s, 1.0))
        return s

    def _add_stream(self, sd: dict):
        sid = str(_req(sd, "id", "una corriente de 'streams'", "identificarla en eventos y salidas"))
        if sid in self.unit_by_id or any(s.id == sid for s in self.streams):
            raise ValueError(f"identificador repetido: '{sid}'")
        frm = to = None
        if sd.get("from") is not None:
            if sd["from"] not in self.ports:
                raise ValueError(f"la corriente '{sid}' sale de '{sd['from']}', que no existe. Puertos: "
                                 f"{', '.join(sorted(self.ports))}")
            frm = self.ports[sd["from"]][1]
        if sd.get("to") is not None:
            if sd["to"] not in self.ports:
                raise ValueError(f"la corriente '{sid}' entra a '{sd['to']}', que no existe. Puertos: "
                                 f"{', '.join(sorted(self.ports))}")
            to = self.ports[sd["to"]][0]
        if frm is None and to is None:
            raise ValueError(f"la corriente '{sid}' no tiene 'from' ni 'to'")
        kinds = {x.kind for x in (frm, to) if x is not None}
        if len(kinds) > 1:
            raise ValueError(f"la corriente '{sid}' une un equipo de líquido con uno de gas: no está cubierto")
        gas = kinds == {"gas"}
        law = sd.get("flow")
        if law is None and frm is not None and frm.kind == "liquid" and frm.hidden:
            law = {"type": "overflow"}             # outlet of a heat-exchanger side: what comes in goes out
        law = _req({"flow": law}, "flow", f"la corriente '{sid}'", "la ley de caudal ('fixed', 'valve', 'pump'…)")
        typ = str(_req(law, "type", f"'flow' de '{sid}'", "elegir la ley de caudal"))
        valid = U.GAS_LAWS if gas else U.LIQUID_LAWS
        if typ not in valid:
            raise ValueError(f"ley de caudal '{typ}' no válida en '{sid}'. Opciones: {', '.join(valid)}")
        s = self._raw_stream(sid, frm, to, law)
        s.gas = gas
        where = f"'flow' de '{sid}'"
        P = self.P
        if gas:
            self._gas_stream(s, sd, law, where)
            return
        s.interacting = bool(law.get("interacting", False))
        if s.interacting and any(getattr(x, "is_flash", False) for x in (frm, to)):
            raise ValueError(f"'{sid}': 'interacting' no aplica a un tambor flash")
        if s.interacting and (frm is None or to is None):
            raise ValueError(f"'{sid}': 'interacting' necesita un tanque en cada extremo")
        needs_level = typ in ("linear_resistance", "gravity_orifice", "inertial_pipe")
        if needs_level:
            for x in ((frm, to) if s.interacting else (frm,)):
                if x is None or x.area is None or x.iV < 0:
                    raise ValueError(f"'{sid}': la ley '{typ}' necesita un tanque con nivel ('area_m2', volumen variable) "
                                     "aguas arriba" + (" y aguas abajo" if s.interacting else ""))
        if typ == "fixed":
            s.key = f"{sid}.volumetric_flow_m3_s"
            P[s.key] = _num(_req(law, "volumetric_flow_m3_s", where, "el caudal impuesto"), f"caudal de '{sid}'",
                            nonneg=True)
        elif typ == "linear_resistance":
            s.key = f"{sid}.resistance_s_m2"
            P[s.key] = _num(_req(law, "resistance_s_m2", where, "la resistencia lineal q = Δh/R"), f"R de '{sid}'",
                            positive=True)
        elif typ == "gravity_orifice":
            cd = _num(_req(law, "discharge_coefficient", where, "la descarga por gravedad q = Cd·a·√(2gh)"),
                      f"Cd de '{sid}'",
                      positive=True)
            s.key = f"{sid}.area_m2"
            P[s.key] = _num(_req(law, "area_m2", where, "el área del orificio de descarga"), f"área de '{sid}'", nonneg=True)
            s.cd = cd
        elif typ == "inertial_pipe":
            s.L = _num(_req(law, "length_m", where, "la inercia del líquido en la tubería"), f"longitud de '{sid}'",
                       positive=True)
            s.D = _num(_req(law, "diameter_m", where, "el área de paso y la fricción"), f"diámetro de '{sid}'",
                       positive=True)
            s.f = _num(_req(law, "darcy_friction_factor", where, "la pérdida por fricción h_f = f·(L/D)·v²/2g"),
                       f"factor de fricción de '{sid}'", nonneg=True)
            v0 = _num(_req(law, "initial_velocity_m_s", where, "el estado inicial de la velocidad en la tubería"),
                      f"velocidad inicial de '{sid}'")
            s.Ap = math.pi * s.D ** 2 / 4.0
            s.iv = self._state(f"{sid}.velocity_m_s", v0, max(abs(v0), 1.0))
            self.assumptions.append(f"'{sid}': tubería con inercia, flujo tapón incompresible, factor de fricción de Darcy "
                                    "constante (el del usuario), impulsada solo por la diferencia de nivel")
        elif typ in ("valve", "pump"):
            self._valve_or_pump(s, law, typ, where)
        elif typ == "overflow":
            if frm is None or frm.iV >= 0 or getattr(frm, "is_flash", False):
                raise ValueError(f"'{sid}': la ley 'overflow' (sale lo que entra) solo aplica a un equipo de volumen "
                                 "constante ('constant_volume': true)")
        # carried properties of a feed
        if frm is None:
            if to.iE >= 0:
                s.kT = f"{sid}.temperature_K"
                P[s.kT] = _num(_req(sd, "temperature_K", f"la alimentación '{sid}'", f"el balance de energía de '{to.id}'"),
                               f"'temperature_K' de '{sid}'", positive=True)
            if self.nc:
                cd_ = _req(sd, "concentration_mol_m3", f"la alimentación '{sid}'",
                           "el balance de componentes (dé 0 para un componente ausente)")
                s.kc = []
                for k in self.comp:
                    key = f"{sid}.concentration_mol_m3.{k}"
                    P[key] = _num(_req(cd_, k, f"'concentration_mol_m3' de '{sid}'", "el balance de componentes"),
                                  f"concentración de {k} en '{sid}'", nonneg=True)
                    s.kc.append(key)

    def _valve_or_pump(self, s: _Obj, law: dict, typ: str, where: str):
        P, sid = self.P, s.id
        frm, to = s.frm, s.to
        s.has_valve = typ == "valve" or law.get("Kv_m3_h") is not None or law.get("Cv_gpm_psi") is not None
        if s.has_valve:
            if law.get("Kv_m3_h") is not None:
                s.kv = _num(law["Kv_m3_h"], f"Kv de '{sid}'", positive=True)
            else:
                s.kv = U.KV_PER_CV * _num(_req(law, "Cv_gpm_psi", where, "el coeficiente de la válvula (o 'Kv_m3_h')"),
                                          f"Cv de '{sid}'", positive=True)
            s.kop = f"{sid}.opening"
            P[s.kop] = _num(_req(law, "opening", where, "la apertura de la válvula (0 a 1)"), f"apertura de '{sid}'")
            if not 0.0 <= P[s.kop] <= 1.0:
                raise ValueError(f"la apertura de '{sid}' debe estar entre 0 y 1 (fracción de carrera)")
            self.fractions.add(s.kop)
            s.char = str(_req(law, "characteristic", where, "la característica inherente de la válvula"))
            if s.char not in U.CHARACTERISTICS:
                raise ValueError(f"'characteristic' de '{sid}' debe ser una de {', '.join(U.CHARACTERISTICS)}")
            s.rang = None
            if s.char == "equal_percentage":
                s.rang = _num(_req(law, "rangeability", where, "la característica isoporcentual f = R^(x−1)"),
                              f"rangeabilidad de '{sid}'", positive=True)
                self.assumptions.append(f"'{sid}': válvula isoporcentual, f(0) = 1/R (no cierra hermético)")
            elif s.char == "quick_opening":
                self.assumptions.append(f"'{sid}': apertura rápida f = √x (convención de libro de texto; la norma no "
                                        "define la forma de esta curva: use la del fabricante si la tiene)")
        src = frm if frm is not None else None
        s.rho = (src.rho if src is not None else None) or (to.rho if to is not None else None)
        if s.rho is None:
            raise ValueError(f"falta 'density_kg_m3' (en 'liquid' o en el equipo) para la corriente '{sid}': se necesita "
                             "para convertir presión en carga")
        # upstream and downstream pressures
        s.kPu = s.kPd = None
        if frm is None:
            s.kPu = f"{sid}.upstream_pressure_Pa"
            P[s.kPu] = _num(_req(law, "upstream_pressure_Pa", where, "la presión aguas arriba de la alimentación"),
                            f"'upstream_pressure_Pa' de '{sid}'", positive=True)
        elif frm.kP is None:
            raise ValueError(f"falta 'pressure_Pa' en el equipo '{frm.id}': la corriente '{sid}' ({typ}) "
                             "necesita la presión "
                             "del espacio de gas aguas arriba")
        if to is None or law.get("downstream_pressure_Pa") is not None:
            s.kPd = f"{sid}.downstream_pressure_Pa"
            P[s.kPd] = _num(_req(law, "downstream_pressure_Pa", where, "la presión aguas abajo"),
                            f"'downstream_pressure_Pa' de '{sid}'", positive=True)
        elif to.kP is None:
            raise ValueError(f"falta 'pressure_Pa' en el equipo '{to.id}' (o 'downstream_pressure_Pa' en la corriente "
                             f"'{sid}'): se necesita la presión aguas abajo")
        if s.interacting and (to.area is None or to.iV < 0):
            raise ValueError(f"'{sid}': 'interacting' (descarga sumergida) necesita un tanque con nivel aguas abajo")
        if typ == "pump":
            cq = np.array(_req(law, "curve_flow_m3_s", where, "la curva de la bomba (caudales)"), dtype=float)
            ch = np.array(_req(law, "curve_head_m", where, "la curva de la bomba (cargas)"), dtype=float)
            if cq.size < 2 or cq.size != ch.size or np.any(np.diff(cq) <= 0) or cq[0] != 0.0 or np.any(np.diff(ch) > 0):
                raise ValueError(f"la curva de la bomba '{sid}' debe tener ≥ 2 puntos, empezar en caudal 0, con caudal "
                                 "creciente y carga no creciente")
            s.cq, s.ch = cq, ch
            s.krun, s.kspeed, s.kstatic, s.kloss = (f"{sid}.running", f"{sid}.speed_fraction", f"{sid}.static_head_m",
                                                    f"{sid}.loss_coefficient_s2_m5")
            P[s.krun] = float(bool(law.get("running", True))) if isinstance(law.get("running", True), bool) else _num(
                law.get("running"), f"'running' de '{sid}'")
            P[s.kspeed] = _num(law.get("speed_fraction", 1.0), f"'speed_fraction' de '{sid}'", nonneg=True)
            P[s.kstatic] = _num(_req(law, "static_head_m", where, "la elevación que vence la bomba (0 si no hay)"),
                                f"'static_head_m' de '{sid}'")
            P[s.kloss] = _num(law.get("loss_coefficient_s2_m5", 0.0), f"'loss_coefficient_s2_m5' de '{sid}'", nonneg=True)
            self.assumptions.append(f"'{sid}': bomba centrífuga con la curva del usuario interpolada linealmente, leyes de "
                                    "afinidad para 'speed_fraction', sin inercia del rotor; parada = caudal cero "
                                    "(válvula de "
                                    "retención en la descarga)")
            if law.get("running") is None:
                self.assumptions.append(f"'{sid}': no se dijo si la bomba arranca encendida; se tomó encendida desde t = 0 "
                                        "('running')")
            if law.get("speed_fraction") is None:
                self.assumptions.append(f"'{sid}': no se dio la velocidad; se tomó la velocidad nominal de la curva "
                                        "('speed_fraction' = 1)")
            if law.get("loss_coefficient_s2_m5") is None and not s.has_valve:
                self.assumptions.append(f"'{sid}': sin válvula ni 'loss_coefficient_s2_m5', la curva del sistema es solo "
                                        "carga estática y presiones")

    def _gas_stream(self, s: _Obj, sd: dict, law: dict, where: str):
        P, sid = self.P, s.id
        if s.law == "fixed":
            if s.frm is not None and s.to is not None:
                raise ValueError(f"'{sid}': caudal másico fijo entre dos recipientes no está cubierto; use 'gas_orifice'")
            s.key = f"{sid}.mass_flow_kg_s"
            P[s.key] = _num(_req(law, "mass_flow_kg_s", where, "el caudal másico impuesto"), f"caudal de '{sid}'",
                            nonneg=True)
        else:
            cd = _num(_req(law, "discharge_coefficient", where, "el caudal por el orificio"), f"Cd de '{sid}'",
                      positive=True)
            s.cd = cd
            s.key = f"{sid}.area_m2"
            P[s.key] = _num(_req(law, "area_m2", where, "el área del orificio"), f"área de '{sid}'", nonneg=True)
            for v in (s.frm, s.to):
                if v is not None and v.gas.model == "ideal" and v.gas.k is None:
                    raise ValueError(f"falta 'gas.heat_capacity_ratio' en '{v.id}': se necesita para el flujo por el "
                                     f"orificio '{sid}' (relación crítica de presiones)")
        s.kPu = s.kTu = s.kPd = None
        if s.frm is None:
            s.kTu = f"{sid}.upstream_temperature_K"
            P[s.kTu] = _num(_req(law, "upstream_temperature_K", where, "la entalpía del gas que entra"),
                            f"'upstream_temperature_K' de '{sid}'", positive=True)
            if s.law == "gas_orifice" or s.to.gas.model == "coolprop":
                s.kPu = f"{sid}.upstream_pressure_Pa"
                P[s.kPu] = _num(_req(law, "upstream_pressure_Pa", where, "la presión de la fuente de gas"),
                                f"'upstream_pressure_Pa' de '{sid}'", positive=True)
        if s.to is None and s.law == "gas_orifice":
            s.kPd = f"{sid}.downstream_pressure_Pa"
            P[s.kPd] = _num(_req(law, "downstream_pressure_Pa", where, "la contrapresión de la descarga"),
                            f"'downstream_pressure_Pa' de '{sid}'", nonneg=True)
        if s.frm is not None and s.to is not None and s.frm.gas.model != s.to.gas.model:
            raise ValueError(f"'{sid}' une recipientes con modelos de gas distintos")

    def _check_topology(self):
        for h in self.holdups:
            outs = [s for s, sg in h.ports if sg < 0 and s.law == "overflow"]
            if h.iV < 0 and h.ports and len(outs) != 1:
                raise ValueError(f"'{h.id}' es de volumen constante: necesita exactamente una corriente de salida con ley "
                                 "'overflow' (sale lo que entra); tiene " + str(len(outs)))
            if h.iE >= 0 and h.heat_terms and h.rhocp is None:
                raise ValueError(f"faltan 'density_kg_m3' y 'heat_capacity_J_kgK' para '{h.id}' (en el equipo o en "
                                 "'liquid'): se necesitan para convertir calor en cambio de temperatura")
            for s, sg in h.ports:
                other = s.frm if sg > 0 else s.to
                if other is None or other.kind != "liquid" or getattr(other, "is_flash", False):
                    continue
                if h.iE >= 0 and other.iE < 0 and (sg > 0 or s.interacting):
                    raise ValueError(f"'{h.id}' tiene balance de energía pero recibe '{s.id}' desde '{other.id}', que no "
                                     "tiene temperatura: dé 'initial.temperature_K' también en ese equipo")
                if h.iE >= 0 and other.iE >= 0 and h.rhocp and other.rhocp and abs(h.rhocp - other.rhocp) > 1e-9 * h.rhocp \
                        and sg > 0:
                    self.warnings.append(f"'{s.id}' une equipos con ρ·cp distintos ('{other.id}' y '{h.id}'): la energía "
                                         "transportada no se conserva en esa corriente; revise las propiedades")
        # scales of the mole states: the largest concentration seen in initial states and feeds
        if self.nc:
            cmax = max([h.cscale for h in self.holdups] + [self.P[k] for s in self.streams if s.kc for k in s.kc]
                       + [f.drum.rhom for f in self.flashes] + [1e-30])
            for h in self.holdups:
                for i in range(self.nc):
                    self._scale[h.iN + i] = cmax * h.vscale
            self.cmax = cmax

    def _order_flows(self):
        """Evaluation order: flow laws first, then 'overflow' streams once every other port of their unit is known."""
        self.law_streams = [s for s in self.streams if s.law != "overflow"]
        pending = [s for s in self.streams if s.law == "overflow"]
        known = {id(s) for s in self.law_streams}
        order = []
        while pending:
            progressed = False
            for s in list(pending):
                if all(id(o) in known for o, _ in s.frm.ports if o is not s):
                    order.append(s)
                    known.add(id(s))
                    pending.remove(s)
                    progressed = True
            if not progressed:
                raise ValueError("lazo algebraico de caudales entre equipos de volumen constante ("
                                 + ", ".join(s.id for s in pending) + "): ponga en el lazo un caudal fijo o un tanque con "
                                 "nivel")
        self.overflow_streams = order

    # ---------------------------------------------------------------------------------------------------- variables
    def _register_unit_variables(self):
        g, p, nc = self.getters, self.pure, self.nc
        for h in self.holdups:
            uid = h.id
            g[f"{uid}.volume_m3"] = lambda h=h: h.V
            if h.iV >= 0:
                p[f"{uid}.volume_m3"] = lambda y, h=h: y[h.iV]
                g[f"{uid}.level_m"] = lambda h=h: h.V / h.area
                p[f"{uid}.level_m"] = lambda y, h=h: y[h.iV] / h.area
            if h.iE >= 0:
                g[f"{uid}.temperature_K"] = lambda h=h: h.T
                p[f"{uid}.temperature_K"] = (lambda y, h=h: y[h.iE] / y[h.iV] if y[h.iV] > h.veps else math.nan) \
                    if h.iV >= 0 else (lambda y, h=h: y[h.iE] / h.Vc)
            for i in range(nc):
                name = f"{uid}.concentration_mol_m3.{self.comp[i]}"
                g[name] = lambda h=h, i=i: h.c[i]
                p[name] = (lambda y, h=h, i=i: y[h.iN + i] / y[h.iV] if y[h.iV] > h.veps else math.nan) \
                    if h.iV >= 0 else (lambda y, h=h, i=i: y[h.iN + i] / h.Vc)
        for v in self.vessels:
            g[f"{v.id}.pressure_Pa"] = lambda v=v: v.P
            g[f"{v.id}.temperature_K"] = lambda v=v: v.T
            g[f"{v.id}.mass_kg"] = lambda v=v: v.m
            p[f"{v.id}.mass_kg"] = lambda y, v=v: y[v.im]
            p[f"{v.id}.pressure_Pa"] = lambda y, v=v: self._vessel_state(v, y)[1]
            p[f"{v.id}.temperature_K"] = lambda y, v=v: self._vessel_state(v, y)[0]
        for s in self.streams:
            if s.gas:
                g[f"{s.id}.mass_flow_kg_s"] = lambda s=s: s.mdot
            else:
                g[f"{s.id}.volumetric_flow_m3_s"] = lambda s=s: s.q
                if s.iv >= 0:
                    g[f"{s.id}.velocity_m_s"] = lambda s=s: s.q / s.Ap
        for fl in self.flashes:
            uid = fl.id
            g[f"{uid}.pressure_Pa"] = lambda fl=fl: fl.eq["P"]
            p[f"{uid}.pressure_Pa"] = lambda y, fl=fl: self._flash_state(fl, y)["P"]
            g[f"{uid}.liquid_level_m"] = lambda fl=fl: fl.eq["level"]
            p[f"{uid}.liquid_level_m"] = lambda y, fl=fl: self._flash_state(fl, y)["level"]
            g[f"{uid}.holdup_vapour_fraction"] = lambda fl=fl: fl.eq["beta"]
            g[f"{uid}.vapour_flow_mol_s"] = lambda fl=fl: fl.Vflow
            for i, c in enumerate(self.comp):
                g[f"{uid}.x.{c}"] = lambda fl=fl, i=i: fl.eq["x"][i]
                p[f"{uid}.x.{c}"] = lambda y, fl=fl, i=i: self._flash_state(fl, y)["x"][i]
                g[f"{uid}.y.{c}"] = lambda fl=fl, i=i: fl.eq["y"][i]
        if self.columns:
            from packages.dynamics import column as COL
            for col in self.columns:
                COL.register_variables(self, col)
        for a, target in self.alias.items():
            g[a] = g[target]
            if target in p:
                p[a] = p[target]

    def _vessel_state(self, v: _Obj, y) -> tuple[float, float, float, float]:
        rho = y[v.im] / v.V
        if v.iU >= 0:
            return v.gas.from_rho_u(rho, y[v.iU] / y[v.im])
        return v.gas.from_rho_T(rho, self.P[v.kT])

    def variable_names(self) -> list[str]:
        return sorted(n for n in self.getters if n not in self.verbose
                      and not any(n.startswith(h.id + ".") for h in self.holdups if h.hidden))

    # ------------------------------------------------------------------------------------------ blocks / controllers
    def _add_block(self, b: dict):
        bid = str(_req(b, "id", "un elemento de 'blocks'", "identificarlo"))
        typ = str(b.get("type", "lag"))
        if typ not in ("lag", "dead_time", "sensor", "actuator"):
            raise ValueError(f"tipo de bloque no soportado en '{bid}': '{typ}' (use 'lag', con 'time_constant_s' y/o "
                             "'dead_time_s')")
        inp = str(_req(b, "input", f"el bloque '{bid}'", "saber qué señal retrasa"))
        tau = _num(b.get("time_constant_s", 0.0), f"'time_constant_s' de '{bid}'", nonneg=True)
        theta = _num(b.get("dead_time_s", 0.0), f"'dead_time_s' de '{bid}'", nonneg=True)
        if tau == 0.0 and theta == 0.0:
            raise ValueError(f"el bloque '{bid}' necesita 'time_constant_s' > 0 y/o 'dead_time_s' > 0 (no se suponen)")
        if inp not in self.getters:
            raise ValueError(f"la entrada '{inp}' del bloque '{bid}' no existe. Variables: "
                             + ", ".join(self.variable_names()[:40]))
        if theta > 0 and inp not in self.pure:
            raise ValueError(f"el tiempo muerto de '{bid}' solo puede aplicarse a una variable de estado de un equipo "
                             "o a la "
                             f"salida de un retardo de primer orden, no a '{inp}'")
        if tau == 0.0 and inp not in self.pure:
            raise ValueError(f"el bloque '{bid}' sin 'time_constant_s' necesita una variable de estado como entrada")
        tgt = b.get("target")
        if tgt is not None and tgt not in self.P:
            raise ValueError(f"el destino '{tgt}' del bloque '{bid}' no es un parámetro. Parámetros: "
                             + ", ".join(sorted(self.P)[:40]))
        blk = _Obj(id=bid, inp=inp, tau=tau, theta=theta, target=tgt, ix=-1, out=0.0, u0=0.0, x0=b.get("initial_output"))
        if tau > 0:
            blk.ix = self._state(f"{bid}.output", 0.0, 1.0)
            self.pure[f"{bid}.output"] = lambda y, blk=blk: y[blk.ix]
        self.getters[f"{bid}.output"] = lambda blk=blk: blk.out
        self.blocks.append(blk)

    def _add_controller(self, c: dict):
        cid = str(_req(c, "id", "un elemento de 'controllers'", "identificarlo"))
        where = f"el controlador '{cid}'"
        meas = str(_req(c, "measurement", where, "la variable medida"))
        if meas not in self.getters or (meas not in self.pure and not any(meas == f"{b.id}.output" for b in self.blocks)):
            raise ValueError(f"la medición '{meas}' de '{cid}' debe ser una variable de estado de un equipo o la salida de "
                             "un bloque (para medir un caudal, pase por un bloque 'lag'). Variables: "
                             + ", ".join(sorted(self.pure)[:40]))
        tgt = str(_req(c, "manipulated", where, "el parámetro que mueve el controlador"))
        if tgt not in self.P:
            raise ValueError(f"la variable manipulada '{tgt}' de '{cid}' no es un parámetro. Parámetros: "
                             + ", ".join(sorted(self.P)[:40]))
        P = self.P
        kc = _num(_req(c, "Kc", where, "la ganancia (u = bias + Kc·(setpoint − medición)…; signo = acción)"),
                  f"Kc de '{cid}'")
        ti = c.get("tau_I_s")
        ti = math.inf if ti is None else _num(ti, f"'tau_I_s' de '{cid}'", positive=True)
        td = _num(c.get("tau_D_s", 0.0), f"'tau_D_s' de '{cid}'", nonneg=True)
        alpha = _num(c.get("derivative_filter_alpha", 0.1), f"'derivative_filter_alpha' de '{cid}'", positive=True)
        ctl = _Obj(id=cid, meas=meas, target=tgt, Kc=kc, tau_I=ti, tau_D=td, tau_f=alpha * td, iI=-1, iF=-1, out=0.0,
                   err=0.0, raw=0.0, ksp=f"{cid}.setpoint", bias=c.get("bias"), bias_value=0.0,
                   umin=-math.inf if c.get("output_min") is None else _num(c["output_min"], f"'output_min' de '{cid}'"),
                   umax=math.inf if c.get("output_max") is None else _num(c["output_max"], f"'output_max' de '{cid}'"))
        if ctl.umin >= ctl.umax:
            raise ValueError(f"'output_min' debe ser menor que 'output_max' en '{cid}'")
        P[ctl.ksp] = _num(_req(c, "setpoint", where, "la consigna"), f"'setpoint' de '{cid}'")
        if math.isfinite(ti):
            tt = c.get("tracking_time_s")
            ctl.Tt = ti if tt is None else _num(tt, f"'tracking_time_s' de '{cid}'", positive=True)
            ctl.iI = self._state(f"{cid}.integral", 0.0, 1.0)
            if tt is None and (math.isfinite(ctl.umin) or math.isfinite(ctl.umax)):
                self.assumptions.append(f"'{cid}': anti-windup por retrocálculo con tiempo de seguimiento = tau_I_s (ajuste "
                                        "del algoritmo; cámbielo con 'tracking_time_s')")
        if td > 0:
            ctl.iF = self._state(f"{cid}.filtered_measurement", 0.0, 1.0)
            if c.get("derivative_filter_alpha") is None:
                self.assumptions.append(f"'{cid}': derivada sobre la medición filtrada con τ_f = 0.1·tau_D_s (ajuste del "
                                        "algoritmo; cámbielo con 'derivative_filter_alpha')")
        if not math.isfinite(ctl.umin) and not math.isfinite(ctl.umax):
            self.assumptions.append(f"'{cid}': sin 'output_min'/'output_max', la salida del controlador no satura")
        self.getters[f"{cid}.output"] = lambda ctl=ctl: ctl.out
        self.getters[f"{cid}.error"] = lambda ctl=ctl: ctl.err
        self.controllers.append(ctl)

    def _init_signals(self):
        """Initial block outputs and bumpless controller start from the initial unit states and base parameters."""
        y = np.array(self._y0, dtype=float)
        self._unit_variables(y)
        for b in self.blocks:
            b.u0 = float(self.pure[b.inp](y)) if b.inp in self.pure else math.nan
            if b.ix >= 0:
                if b.x0 is not None:
                    x0 = _num(b.x0, f"'initial_output' de '{b.id}'")
                elif b.inp in self.pure:
                    x0 = b.u0
                else:
                    x0 = None             # input is a flow or a controller output: fixed below, after one evaluation
                y[b.ix] = 0.0 if x0 is None else x0
                self._scale[b.ix] = max(abs(y[b.ix]), 1e-3)
                b.out = y[b.ix]
            else:
                b.out = b.u0
        for c in self.controllers:
            ym = self.getters[c.meas]() if c.meas not in self.pure else float(self.pure[c.meas](y))
            e0 = self.P[c.ksp] - ym
            u0 = self.P[c.target]
            if c.bias is not None:
                c.bias_value = _num(c.bias, f"'bias' de '{c.id}'")
            else:
                c.bias_value = u0 - c.Kc * e0
                self.assumptions.append(f"'{c.id}': sin 'bias', arranque sin salto (la salida inicial es el valor "
                                        "inicial de "
                                        f"'{c.target}')")
            if c.iI >= 0:
                y[c.iI] = c.bias_value
                self._scale[c.iI] = max(abs(c.bias_value), abs(u0), abs(c.Kc * e0), 1e-3)
            if c.iF >= 0:
                y[c.iF] = ym
                self._scale[c.iF] = max(abs(ym), 1e-3)
        self.y0 = y
        self.scale = np.array(self._scale + [1.0] * (len(y) - len(self._scale)), dtype=float)
        late = [b for b in self.blocks if b.ix >= 0 and b.x0 is None and b.inp not in self.pure]
        if late:
            self.rhs(0.0, self.y0)
            for b in late:
                self.y0[b.ix] = self.getters[b.inp]()
                self.scale[b.ix] = max(abs(self.y0[b.ix]), 1e-3)
        self._set_quadrature_scales()

    # -------------------------------------------------------------------------------------------------- quadratures
    def _add_quadratures(self):
        names = ["vol_in", "vol_out", "vol_spill"]
        names += [f"{p}.{c}" for c in self.comp for p in ("n_in", "n_out", "n_gen")]
        names += ["e_in", "e_out", "e_heat", "e_rxn", "m_in", "m_out", "eg_in", "eg_out", "eg_heat"]
        names += ["col_in", "col_in_light", "col_out", "col_out_light"]
        self.qi = {}
        for n in names:
            self.qi[n] = len(self._y0)
            self.names.append("∫" + n)
            self._y0.append(0.0)
            self._scale.append(1.0)
        self.dq = np.zeros(len(self._y0))

    def _set_quadrature_scales(self):
        sc, qi = self.scale, self.qi
        vs = max([h.vscale for h in self.holdups] + [1e-30]) if self.holdups else 1.0
        for n in ("vol_in", "vol_out", "vol_spill"):
            sc[qi[n]] = vs
        for c in self.comp:
            for p in ("n_in", "n_out", "n_gen"):
                sc[qi[f"{p}.{c}"]] = vs * self.cmax
        es = max([(h.rhocp or 1.0) * h.vscale * (h.T0 if math.isfinite(h.T0) else 300.0) for h in self.holdups] + [1e-30])
        for n in ("e_in", "e_out", "e_heat", "e_rxn"):
            sc[qi[n]] = es
        ms = max([v.m0 for v in self.vessels] + [1e-30])
        egs = max([v.escale for v in self.vessels] + [1e-30])      # also when isothermal (enthalpy still flows)
        for n in ("m_in", "m_out"):
            sc[qi[n]] = ms
        for n in ("eg_in", "eg_out", "eg_heat"):
            sc[qi[n]] = egs

    # ----------------------------------------------------------------------------------------------------- schedule
    def set_schedule(self, events: list[dict] | None, t_end: float) -> list[dict]:
        """Compile the event list into per-parameter piecewise-linear knots. Returns the normalised events."""
        self.base = dict(self.P)
        knots: dict[str, list[list[float]]] = {}
        out = []
        evs = []
        for k, e in enumerate(events or []):
            where = f"el evento {k + 1}"
            t = _num(_req(e, "time_s", where, "saber cuándo ocurre"), f"'time_s' de {where}", nonneg=True)
            tgt = str(_req(e, "target", where, "saber qué parámetro cambia"))
            if tgt not in self.P:
                raise ValueError(f"el destino '{tgt}' de {where} no es un parámetro del flowsheet. Parámetros: "
                                 + ", ".join(sorted(self.P)))
            val = _req(e, "value", where, "el nuevo valor del parámetro")
            if tgt in self.driven:
                raise ValueError(f"{where}: el destino '{tgt}' lo fija {self.driven[tgt]} en cada instante: el evento no "
                                 "tendría efecto (cambie la consigna '<controlador>.setpoint' o quite el controlador)")
            val = float(val) if isinstance(val, bool) else _num(val, f"'value' de {where}")
            if tgt in self.fractions and not 0.0 <= val <= 1.0:
                raise ValueError(f"{where}: el valor sobre '{tgt}' debe estar entre 0 y 1 (apertura de válvula)")
            ramp = _num(e.get("ramp_s", 0.0), f"'ramp_s' de {where}", nonneg=True)
            evs.append((t, k, tgt, val, ramp))
        evs.sort()
        last_end: dict[str, float] = {}
        cur: dict[str, float] = {}
        for t, k, tgt, val, ramp in evs:
            if t < last_end.get(tgt, -1.0) - 1e-12:
                raise ValueError(f"el evento {k + 1} sobre '{tgt}' empieza antes de que termine la rampa anterior")
            prev = cur.get(tgt, self.base[tgt])
            kn = knots.setdefault(tgt, [])
            if ramp > 0:
                kn.append([t, prev, prev])
                kn.append([t + ramp, val, val])
            else:
                kn.append([t, prev, val])
            cur[tgt] = val
            last_end[tgt] = t + ramp
            out.append({"time_s": t, "target": tgt, "value": val, "ramp_s": ramp, "inside_horizon": t < t_end})
        self.knots = knots
        bp = {0.0, float(t_end)}
        for kn in knots.values():
            for t, _, _ in kn:
                if 0.0 < t < t_end:
                    bp.add(t)
        for b in list(bp):                      # a discontinuity re-appears one dead time later in the delayed signal
            for th in self.delays:
                if 0.0 < b + th < t_end:
                    bp.add(b + th)
        self.breakpoints = sorted(bp)
        return out

    def _segment_schedule(self, ta: float):
        """Linear law a + b·t of every scheduled parameter on the interval that starts at ta."""
        now = []
        for tgt, kn in self.knots.items():
            ts = [k[0] for k in kn]
            i = bisect.bisect_right(ts, ta + 1e-12) - 1
            if i < 0:
                a, b = self.base[tgt], 0.0
            elif i == len(kn) - 1:
                a, b = kn[i][2], 0.0
            else:
                (t1, _, v1), (t2, v2, _) = kn[i], kn[i + 1]
                b = (v2 - v1) / (t2 - t1)
                a = v1 - b * t1
            now.append((tgt, a, b))
        self.sched_now = now

    def param_at(self, key: str, t: float) -> float:
        kn = self.knots.get(key) if hasattr(self, "knots") else None
        if not kn:
            return self.base[key] if hasattr(self, "base") else self.P[key]
        ts = [k[0] for k in kn]
        i = bisect.bisect_right(ts, t) - 1
        if i < 0:
            return self.base[key]
        if i == len(kn) - 1:
            return kn[i][2]
        (t1, _, v1), (t2, v2, _) = kn[i], kn[i + 1]
        return v1 + (v2 - v1) * (t - t1) / (t2 - t1)

    # ---------------------------------------------------------------------------------------------- right-hand side
    def _unit_variables(self, y):
        P, nc = self.P, self.nc
        for h in self.holdups:
            V = y[h.iV] if h.iV >= 0 else h.Vc
            h.V = V
            h.wet = V > h.veps
            if h.wet:
                if nc:
                    h.c = y[h.iN:h.iN + nc] / V
                if h.iE >= 0:
                    h.T = y[h.iE] / V
            else:
                if nc:
                    h.c = np.zeros(nc)
                if h.iE >= 0:
                    h.T = h.T0
            if h.kT is not None:
                h.T = P[h.kT]
        for v in self.vessels:
            v.m = y[v.im]
            v.T, v.P, v.h, v.k = self._vessel_state(v, y)
        for col in self.columns:                         # controllers read the column before its balances run
            col.M = y[col.iM:col.iM + col.NT]
            with np.errstate(divide="ignore", invalid="ignore"):
                col.x = y[col.iN:col.iN + col.NT] / col.M
        for fl in self.flashes:
            fl.eq = eq = self._flash_state(fl, y)          # FlashOutOfRange (a ValueError) outside two phases
            P[fl.kP] = eq["P"]
            fl.V = eq["L"] / fl.drum.rhom
            fl.c = eq["x"] * fl.drum.rhom

    def set_initial_dynamic_state(self, y_dyn: np.ndarray):
        """Replace the initial dynamic states (e.g. by a steady state) and the pre-t0 history of delayed signals."""
        self.y0[:self.n_dyn] = y_dyn[:self.n_dyn]
        for b in self.blocks:
            if b.inp in self.pure:
                b.u0 = float(self.pure[b.inp](self.y0))

    def _history(self, name: str, t: float, fallback: float) -> float:
        if t <= self.t0 or not self.hist_t:
            return fallback
        i = bisect.bisect_right(self.hist_t, t) - 1
        return float(self.pure[name](self.hist_sol[max(i, 0)](t)))

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        P, nc, qi = self.P, self.nc, self.qi
        for key, a, b in self.sched_now:
            P[key] = a + b * t
        if self.flashes:
            from packages.dynamics.flash_drum import FlashOutOfRange
            try:
                self._unit_variables(y)
            except FlashOutOfRange as e:            # a trial state outside two phases: NaN makes the solver step back
                self.oor = str(e)
                return np.full(len(y), np.nan)
        else:
            self._unit_variables(y)
        dy = np.zeros(len(y))
        # --- signal blocks (outputs) and controllers
        for b in self.blocks:
            if b.theta > 0 and not self.no_delay:
                b.ud = self._history(b.inp, t - b.theta, b.u0)
            elif b.inp in self.pure:
                b.ud = float(self.pure[b.inp](y))
            b.out = y[b.ix] if b.ix >= 0 else b.ud
            if b.target is not None:
                P[b.target] = b.out
        for c in self.controllers:
            ym = self.getters[c.meas]()
            e = P[c.ksp] - ym
            c.err = e
            d = 0.0
            if c.iF >= 0:
                dyf = (ym - y[c.iF]) / c.tau_f
                dy[c.iF] = dyf
                d = -c.Kc * c.tau_D * dyf
            raw = (y[c.iI] if c.iI >= 0 else c.bias_value) + c.Kc * e + d
            u = min(max(raw, c.umin), c.umax)
            if c.iI >= 0:
                dy[c.iI] = c.Kc * e / c.tau_I + (u - raw) / c.Tt
            c.raw, c.out = raw, u
            P[c.target] = u
        # --- liquid flows
        for s in self.law_streams:
            if s.gas:
                continue
            law = s.law
            if law == "fixed":
                s.q = P[s.key]
                continue
            frm, to = s.frm, s.to
            hu = frm.V / frm.area if (frm is not None and frm.area and (frm.iV >= 0 or getattr(frm, "is_flash", False))) \
                else 0.0
            hd = to.V / to.area if s.interacting else 0.0
            if law == "linear_resistance":
                s.q = (hu - hd) / P[s.key] if s.interacting else max(hu, 0.0) / P[s.key]
            elif law == "gravity_orifice":
                dh = hu - hd
                s.q = s.cd * P[s.key] * (U.ssqrt(2.0 * U.G * dh, 2.0 * U.G * U.REG_HEAD_M) if s.interacting
                                         else math.sqrt(2.0 * U.G * max(dh, 0.0)))
            elif law == "inertial_pipe":
                v = y[s.iv]
                s.q = s.Ap * v
                dy[s.iv] = U.G * (hu - hd) / s.L - s.f * v * abs(v) / (2.0 * s.D)
            else:
                pu = P[s.kPu] if s.kPu else P[frm.kP]
                pd = P[s.kPd] if s.kPd else P[to.kP]
                head = (pu - pd) / (s.rho * U.G) + hu - hd          # driving head available without the pump, m
                if law == "valve":
                    s.q = U.valve_flow(s.kv, U.valve_fraction(P[s.kop], s.char, s.rang), head * s.rho * U.G, s.rho,
                                       s.interacting)
                else:
                    loss = P[s.kloss]
                    if s.has_valve:
                        kvf = s.kv * U.valve_fraction(P[s.kop], s.char, s.rang) / 3600.0
                        if kvf <= 0.0:
                            s.q = 0.0
                            continue
                        loss += (s.rho / 1000.0) * 1e5 / (s.rho * U.G) / (kvf * kvf)
                    if P[s.krun] <= 0.0:
                        s.q = 0.0
                    else:
                        s.q, s.end_of_curve = U.pump_flow(s.cq, s.ch, P[s.kspeed], P[s.kstatic] - head, loss)
        for h in self.holdups:
            if h.mode != "empty":
                continue
            inflow = demand = 0.0
            for s, sg in h.ports:
                if s.law == "overflow":
                    continue
                f = sg * s.q
                if f > 0:
                    inflow += f
                else:
                    demand -= f
            h.net_potential = inflow - demand
            phi = min(1.0, inflow / demand) if demand > 0 else 1.0
            if phi < 1.0:
                for s, sg in h.ports:
                    if sg * s.q < 0:
                        s.q *= phi
        for s in self.overflow_streams:
            h = s.frm
            s.q = sum(sg * o.q for o, sg in h.ports if o is not s)
        # --- carried properties of every liquid stream (those of its actual source)
        for h in self.holdups:
            if not h.wet:                       # an empty tank passes its inflow straight through
                qin, cin, tin = 0.0, np.zeros(nc), 0.0
                for s, sg in h.ports:
                    f = sg * s.q
                    if f > 0:
                        src = s.frm if sg > 0 else s.to
                        qin += f
                        if nc:
                            cin = cin + f * (src.c if src is not None else np.array([P[k] for k in s.kc]))
                        if h.iE >= 0:
                            tin += f * (src.T if src is not None else P[s.kT])
                if qin > 0:
                    if nc:
                        h.c = cin / qin
                    if h.iE >= 0:
                        h.T = tin / qin
        dq = self.dq
        dq[:] = 0.0
        for s in self.streams:
            if s.gas:
                continue
            src = s.frm if s.q >= 0 else s.to
            if src is None:
                s.cs = np.array([P[k] for k in s.kc]) if s.kc else None
                s.Ts = P[s.kT] if s.kT else math.nan
            else:
                s.cs, s.Ts = src.c, src.T
            if s.frm is None:
                dq[qi["vol_in"]] += s.q
                for i in range(nc):
                    dq[qi[f"n_in.{self.comp[i]}"]] += s.q * s.cs[i]
            elif s.to is None:
                dq[qi["vol_out"]] += s.q
                for i in range(nc):
                    dq[qi[f"n_out.{self.comp[i]}"]] += s.q * s.cs[i]
            if getattr(s.to, "is_flash", False):        # liquid volume is not conserved through a flash: the drum is a
                dq[qi["vol_out"]] += s.q                  # boundary of the volume balance (moles are balanced in full)
            if getattr(s.frm, "is_flash", False):
                dq[qi["vol_in"]] += s.q
        for lk in self.links:
            a, b = lk.a, lk.b
            lk.Q = P[lk.key] * lk.factor * (a.T - b.T) if (a.wet and b.wet) else 0.0
        # --- liquid balances
        for h in self.holdups:
            net = 0.0
            dN = np.zeros(nc)
            dE = 0.0
            ein = eout = 0.0
            for s, sg in h.ports:
                f = sg * s.q
                net += f
                if nc:
                    dN += f * s.cs
                if h.iE >= 0:
                    dE += f * s.Ts
                    if f > 0:
                        ein += f * s.Ts
                    else:
                        eout -= f * s.Ts
            h.net = net
            spill = 0.0
            if h.mode == "full":
                spill = net
                dq[qi["vol_spill"]] += spill
                if nc:
                    dN -= spill * h.c
                    for i in range(nc):
                        dq[qi[f"n_out.{self.comp[i]}"]] += spill * h.c[i]
                if h.iE >= 0:
                    dE -= spill * h.T
                    eout += spill * h.T
            h.spill = spill
            if h.iV >= 0:
                dy[h.iV] = net - spill
            rc = h.rhocp or 1.0
            heat = 0.0
            if h.wet:
                for rx in h.reactions:
                    r = U.reaction_rate(rx, h.c, h.T) * h.V
                    dN += rx["nu"] * r
                    for i in range(nc):
                        if rx["nu"][i] != 0.0:
                            dq[qi[f"n_gen.{self.comp[i]}"]] += rx["nu"][i] * r
                    if h.iE >= 0:
                        dE += -rx["dH"] * r / rc
                        dq[qi["e_rxn"]] += -rx["dH"] * r
                if h.iE >= 0:
                    if h.kQ:
                        heat += P[h.kQ]
                    if h.kUA:
                        heat += P[h.kUA] * (P[h.kTj] - h.T)
            if h.iE >= 0:
                dq[qi["e_heat"]] += heat             # links are internal: they cancel in the global balance
                for lk in h.links_a:
                    heat -= lk.Q
                for lk in h.links_b:
                    heat += lk.Q
                dy[h.iE] = dE + heat / rc
                dq[qi["e_in"]] += rc * ein
                dq[qi["e_out"]] += rc * eout
            if nc:
                dy[h.iN:h.iN + nc] = dN
        for fl in self.flashes:
            dr = fl.drum
            for k, key in fl.pkeys.items():
                dr.P[k] = P[key]
            fl.Vflow = dr.vapour_flow(fl.eq)
            dN = -fl.Vflow * fl.eq["y"]
            for s, sg in fl.ports:
                dN = dN + sg * s.q * s.cs
            dy[fl.iM:fl.iM + nc] = dN
            for i in range(nc):
                dq[qi[f"n_out.{self.comp[i]}"]] += fl.Vflow * fl.eq["y"][i]
        if self.columns:
            from packages.dynamics import column as COL
            for col in self.columns:
                fi, fl_, po, pl = COL.column_rhs(col, y, dy, P)
                dq[qi["col_in"]] += fi
                dq[qi["col_in_light"]] += fl_
                dq[qi["col_out"]] += po
                dq[qi["col_out_light"]] += pl
        # --- gas
        for s in self.streams:
            if not s.gas:
                continue
            frm, to = s.frm, s.to
            if s.law == "fixed":
                s.mdot = P[s.key]
                up = frm
            else:
                pu = frm.P if frm is not None else P[s.kPu]
                pd = to.P if to is not None else P[s.kPd]
                if pu >= pd:
                    up, sign, p1, p2 = frm, 1.0, pu, pd
                elif to is not None and frm is not None:
                    up, sign, p1, p2 = to, -1.0, pd, pu
                else:
                    up, sign, p1, p2 = frm, 0.0, pu, pd          # no back-flow from or to a boundary
                if sign == 0.0:
                    s.mdot = 0.0
                elif up is None:
                    gas = to.gas
                    rho, _, hsrc = gas.from_TP(P[s.kTu], p1)
                    k = gas.k if gas.model == "ideal" else gas.from_rho_T(rho, P[s.kTu])[3]
                    s.mdot = U.orifice_mass_flow(s.cd * P[s.key], p1, rho, k, p2)
                    s.hs = hsrc
                else:
                    s.mdot = sign * U.orifice_mass_flow(s.cd * P[s.key], p1, up.m / up.V, up.k, p2)
            if s.law == "fixed" and frm is None:
                s.hs = to.gas.from_TP(P[s.kTu], P[s.kPu] if s.kPu else to.P)[2]
            elif s.law == "fixed" or (s.mdot >= 0 and frm is not None):
                s.hs = frm.h
            elif s.mdot < 0:
                s.hs = to.h
        for v in self.vessels:              # an empty vessel: a fixed-flow outlet can only take what comes in
            if v.mode != "empty":
                continue
            inflow = demand = 0.0
            for s, sg in v.ports:
                f = sg * s.mdot
                if f > 0:
                    inflow += f
                elif s.law == "fixed":
                    demand -= f
            v.net_potential = inflow - demand
            phi = min(1.0, inflow / demand) if demand > 0 else 1.0
            if phi < 1.0:
                for s, sg in v.ports:
                    if s.law == "fixed" and sg * s.mdot < 0:
                        s.mdot *= phi
        for s in self.streams:
            if not s.gas:
                continue
            if s.frm is None:
                dq[qi["m_in"]] += s.mdot
                dq[qi["eg_in"]] += s.mdot * s.hs
            elif s.to is None:
                dq[qi["m_out"]] += s.mdot
                dq[qi["eg_out"]] += s.mdot * s.hs
        for v in self.vessels:
            dm = dU = 0.0
            for s, sg in v.ports:
                dm += sg * s.mdot
                dU += sg * s.mdot * s.hs
            dy[v.im] = dm
            v.net = dm
            if v.iU >= 0:
                v.Qamb = P[v.kUA] * (P[v.kTa] - v.T) if v.kUA else 0.0
                dy[v.iU] = dU + v.Qamb
                dq[qi["eg_heat"]] += v.Qamb
        # --- block dynamics (their input may be a flow or a controller output, known by now)
        for b in self.blocks:
            if b.ix >= 0:
                u = b.ud if (b.theta > 0 or b.inp in self.pure) else self.getters[b.inp]()
                dy[b.ix] = (u - y[b.ix]) / b.tau
        dy[self.n_dyn:] = dq[self.n_dyn:]
        return dy

    # ------------------------------------------------------------------------------------------------------ modes
    def finish(self):
        for v in self.vessels:
            v.fixed_out = any(s.law == "fixed" and sg < 0 for s, sg in v.ports)
        self.moded = [h for h in self.holdups if h.iV >= 0] + [v for v in self.vessels if v.fixed_out]
        for h in self.holdups:
            h.links_a = [lk for lk in self.links if lk.a is h]
            h.links_b = [lk for lk in self.links if lk.b is h]

    def fix_modes(self, t: float, y: np.ndarray) -> list[tuple[str, str]]:
        """Mode of every tank that sits at a bound, consistent with the sign of its net inflow at (t, y)."""
        changes = []
        for _ in range(4):
            changed = False
            self.rhs(t, y)
            for h in self.holdups:
                if h.iV < 0:
                    continue
                new = h.mode
                at_top = h.vmax is not None and y[h.iV] >= h.vmax * (1 - 1e-12)
                at_bottom = y[h.iV] <= (EMPTY_SQRT ** 2) * h.vscale * (1 + 1e-9)
                if h.mode == "normal":
                    if at_top and h.net > 0:
                        new = "full"
                    elif at_bottom and h.net <= 0:
                        new = "empty"
                elif h.mode == "full" and h.net < 0:
                    new = "normal"
                elif h.mode == "empty" and h.net_potential > 0:
                    new = "normal"
                if new != h.mode:
                    h.mode = new
                    changes.append((h.id, new))
                    changed = True
            for v in self.vessels:
                if not v.fixed_out:
                    continue
                new = v.mode
                if v.mode == "normal" and y[v.im] <= GAS_EMPTY * v.m0 * (1 + 1e-9) and v.net <= 0:
                    new = "empty"
                elif v.mode == "empty" and v.net_potential > 0:
                    new = "normal"
                if new != v.mode:
                    v.mode = new
                    changes.append((v.id, new))
                    changed = True
            if not changed:
                break
        return changes

    def mode_events(self) -> tuple[list, list[tuple[_Obj, str]]]:
        fns, meta = [], []
        for h in self.holdups:
            if h.iV < 0:
                continue
            if h.mode == "normal":
                if h.vmax is not None:
                    f = (lambda t, y, h=h: y[h.iV] - h.vmax)
                    f.terminal, f.direction = True, 1.0
                    fns.append(f)
                    meta.append((h, "full"))
                f = (lambda t, y, h=h: U.ssqrt(y[h.iV] / h.vscale, 1e-30) - EMPTY_SQRT)
                f.terminal, f.direction = True, -1.0
                fns.append(f)
                meta.append((h, "empty"))
            elif h.mode == "full":
                f = (lambda t, y, h=h: (self.rhs(t, y), h.net)[1])
                f.terminal, f.direction = True, -1.0
                fns.append(f)
                meta.append((h, "normal"))
            else:
                f = (lambda t, y, h=h: (self.rhs(t, y), h.net_potential)[1])
                f.terminal, f.direction = True, 1.0
                fns.append(f)
                meta.append((h, "normal"))
        for v in self.vessels:
            if not v.fixed_out:
                continue
            if v.mode == "normal":
                f = (lambda t, y, v=v: y[v.im] / v.m0 - GAS_EMPTY)
                f.terminal, f.direction = True, -1.0
                meta.append((v, "empty"))
            else:
                f = (lambda t, y, v=v: (self.rhs(t, y), v.net_potential)[1])
                f.terminal, f.direction = True, 1.0
                meta.append((v, "normal"))
            fns.append(f)
        return fns, meta

    def values(self, t: float, y: np.ndarray, names: list[str]) -> list[float]:
        self.rhs(t, y)
        out = []
        for n in names:
            if n in self.getters:
                out.append(float(self.getters[n]()))
            else:
                out.append(float(self.P[n]))
        return out


# -------------------------------------------------------------------------------------------------------- integration
class Trajectory:
    """Piecewise dense solution: segments (t_start, t_end, callable, schedule snapshot, modes)."""

    def __init__(self, model: Model):
        self.m = model
        self.t0s: list[float] = []
        self.t1s: list[float] = []
        self.sols: list = []
        self.sched: list = []
        self.modes: list = []
        self.steps = 0
        self.rhs_evals = 0

    def add(self, sol, ta: float, tb: float):
        m = self.m
        self.t0s.append(ta)
        self.t1s.append(tb)
        self.sols.append(sol.sol)
        self.sched.append(list(m.sched_now))
        self.modes.append([h.mode for h in m.holdups + m.vessels])
        self.steps += len(sol.t) - 1
        self.rhs_evals += int(sol.nfev)
        m.hist_t.append(ta)
        m.hist_sol.append(sol.sol)

    def state(self, t: float, right: bool = True) -> tuple[np.ndarray, int]:
        """State at t; at a restart time `right` picks the segment that starts there (after the event)."""
        i = (bisect.bisect_right(self.t0s, t) - 1) if right else (bisect.bisect_left(self.t0s, t) - 1)
        i = min(max(i, 0), len(self.sols) - 1)
        return self.sols[i](min(max(t, self.t0s[i]), self.t1s[i])), i

    def activate(self, t: float, right: bool = True) -> np.ndarray:
        """Put the model in the schedule and tank modes of the segment that holds t; returns the state there."""
        y, i = self.state(t, right)
        m = self.m
        m.sched_now = self.sched[i]
        for h, mode in zip(m.holdups + m.vessels, self.modes[i]):
            h.mode = mode
        return y

    def values(self, t: float, names: list[str], right: bool = True) -> list[float]:
        return self.m.values(t, self.activate(t, right), names)


def integrate(m: Model, t_end: float, method: str, rtol: float, atol_factor: float) -> tuple[Trajectory, dict]:
    """Integrate from 0 to t_end restarting at every breakpoint, mode switch and (with dead time) every θ_min."""
    tr = Trajectory(m)
    y = m.y0.copy()
    atol = atol_factor * m.scale
    info = {"success": True, "message": "", "restarts": [], "incidents": [], "mode_switches": 0}
    chunk = min(m.delays) if m.delays else math.inf
    bps = m.breakpoints
    switches = 0
    for ta, tb in zip(bps[:-1], bps[1:]):
        m._segment_schedule(ta)
        info["restarts"].append(ta)
        for uid, mode in m.fix_modes(ta, y):
            info["incidents"].append({"time_s": ta, "unit": uid, "mode": mode})
        t = ta
        while t < tb:
            t_stop = min(tb, t + chunk)
            if tb - t_stop < 1e-9 * max(chunk, 1e-300) and math.isfinite(chunk):
                t_stop = tb
            fns, meta = m.mode_events()
            try:
                sol = solve_ivp(m.rhs, (t, t_stop), y, method=method, rtol=rtol, atol=atol, dense_output=True,
                                events=fns or None)
            except (ValueError, np.linalg.LinAlgError) as e:   # a property call failed (two-phase gas, NaN Jacobian)
                why = f" (estado de prueba fuera de la región de dos fases de un tambor flash: {m.oor})" if m.oor else ""
                info.update(success=False, message=str(e)[:200] + why, t_reached=t)
                return tr, info
            if sol.status < 0 or not sol.success:
                why = f" (estado de prueba fuera de la región de dos fases de un tambor flash: {m.oor})" if m.oor else ""
                info.update(success=False, message=str(sol.message)[:200] + why, t_reached=float(sol.t[-1]))
                if len(sol.t) > 1:
                    tr.add(sol, t, float(sol.t[-1]))
                return tr, info
            t_new = float(sol.t[-1])
            if t_new > t:
                tr.add(sol, t, t_new)
            y = sol.y[:, -1].copy()
            if sol.status == 1:
                fired = [meta[i] for i, te in enumerate(sol.t_events) if len(te)]
                for h, mode in fired:
                    h.mode = mode
                    info["incidents"].append({"time_s": t_new, "unit": h.id, "mode": mode})
                for uid, mode in m.fix_modes(t_new, y):
                    info["incidents"].append({"time_s": t_new, "unit": uid, "mode": mode})
                switches += 1
                if switches > MAX_MODE_SWITCHES:
                    info.update(success=False, t_reached=t_new,
                                message="demasiados cambios de modo (rebose/vacío) seguidos: "
                                "el tanque oscila en su límite; la simulación se detuvo")
                    return tr, info
            t = t_new
    info["mode_switches"] = switches
    info["t_reached"] = float(t_end)
    return tr, info


# ------------------------------------------------------------------------------------------------------ steady state
def steady_state(m: Model) -> dict:
    """Solve dy/dt = 0 for the dynamic states from the model's initial state (base parameters, modes as they are)."""
    m.sched_now = []
    m.no_delay = True
    n = m.n_dyn
    sc = m.scale[:n]
    y_full = m.y0.copy()
    m.fix_modes(0.0, y_full)

    def f(x):
        y_full[:n] = x * sc
        r = m.rhs(0.0, y_full)[:n] / sc
        return r if np.all(np.isfinite(r)) else np.full(n, 1e3)    # outside a flash's two-phase region

    x0 = m.y0[:n] / sc
    sol = root(f, x0, method="hybr", options={"xtol": 1e-14, "maxfev": 400 * (n + 1)})
    x = sol.x
    res = f(x)
    J = _jacobian(f, x, 1e-6)
    try:
        newton = np.linalg.solve(J, -res)
        dist = float(np.max(np.abs(newton) / np.maximum(np.abs(x), 1e-3)))
    except np.linalg.LinAlgError:
        dist = math.inf
    y_full[:n] = x * sc
    m.no_delay = False
    return {"y": y_full.copy(), "x": x, "f": f, "residual_max_per_s": float(np.max(np.abs(res))) if n else 0.0,
            "newton_correction_rel": dist, "converged": bool(dist < 1e-7), "solver_message": str(sol.message), "J": J}


def _jacobian(f, x: np.ndarray, rel: float) -> np.ndarray:
    n = len(x)
    J = np.zeros((n, n))
    for j in range(n):
        h = rel * max(abs(x[j]), 1e-3)
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (f(xp) - f(xm)) / (2 * h)
    return J


# ----------------------------------------------------------------------------------------------------------- metrics
def _resolve_outputs(m: Model, outputs: list[str] | None) -> list[str]:
    if not outputs:
        return m.variable_names()
    valid = set(m.getters) | set(m.P)
    for o in outputs:
        if o not in valid:
            raise ValueError(f"la salida '{o}' no existe. Variables: {', '.join(m.variable_names())}. Parámetros: "
                             f"{', '.join(sorted(m.P))}")
    return list(outputs)


def _time_grid(tr: Trajectory, t_end: float, n: int, marks: list[float]) -> np.ndarray:
    t_end = min(t_end, tr.t1s[-1])
    return np.unique(np.concatenate([np.linspace(0.0, t_end, n), [t for t in marks if 0.0 <= t <= t_end]]))


def sample(tr: Trajectory, names: list[str], grid: np.ndarray) -> np.ndarray:
    return np.array([tr.values(float(t), names) for t in grid], dtype=float).T


def _metrics(grid: np.ndarray, v: np.ndarray, t_last_event: float) -> dict:
    ok = np.isfinite(v)
    if not ok.any():
        return {"initial": None, "final": None}
    vi = np.where(ok)[0]
    imin, imax = vi[np.argmin(v[ok])], vi[np.argmax(v[ok])]
    out = {"initial": float(v[vi[0]]), "final": float(v[vi[-1]]), "min": float(v[imin]), "time_of_min_s": float(grid[imin]),
           "max": float(v[imax]), "time_of_max_s": float(grid[imax])}
    after = (grid >= t_last_event) & ok
    final = out["final"]
    start = float(v[after][0]) if after.any() else final
    change = final - start
    span = max(abs(change), 1e-12 * max(abs(final), 1.0))
    if abs(change) <= 1e-9 * max(abs(final), abs(start), 1e-30):
        out["settling_time_s"] = 0.0
        out["overshoot_fraction"] = 0.0
    else:
        outside = after & (np.abs(v - final) > SETTLE_BAND * span)
        idx = np.where(outside)[0]
        if len(idx) == 0:
            out["settling_time_s"] = 0.0
        elif idx[-1] + 1 < len(grid):
            out["settling_time_s"] = float(grid[idx[-1] + 1] - t_last_event)
        else:
            out["settling_time_s"] = None
        peak = (v[after].max() - final) if change > 0 else (final - v[after].min())
        out["overshoot_fraction"] = float(max(peak, 0.0) / abs(change))
    return out


def _limit_violations(tr: Trajectory, limits: list[dict] | None, m: Model, grid: np.ndarray) -> list[dict]:
    out = []
    for k, lim in enumerate(limits or []):
        var = str(_req(lim, "variable", f"el límite {k + 1}", "saber qué variable se vigila"))
        if var not in m.getters and var not in m.P:
            raise ValueError(f"la variable '{var}' del límite {k + 1} no existe. Variables: {', '.join(m.variable_names())}")
        if lim.get("max") is None and lim.get("min") is None:
            raise ValueError(f"el límite {k + 1} sobre '{var}' necesita 'max' y/o 'min' (no se suponen)")
        v = sample(tr, [var], grid)[0]
        for kind, sign in (("max", 1.0), ("min", -1.0)):
            if lim.get(kind) is None:
                continue
            bound = _num(lim[kind], f"'{kind}' del límite sobre '{var}'")
            g = sign * (v - bound)
            bad = np.isfinite(g) & (g > 0)
            if not bad.any():
                continue
            i = int(np.argmax(bad))
            t_first = float(grid[i])
            if i > 0 and np.isfinite(g[i - 1]) and g[i - 1] <= 0:
                try:
                    t_first = float(brentq(lambda t: sign * (tr.values(t, [var], right=False)[0] - bound), grid[i - 1],
                                           grid[i], xtol=1e-12 * max(grid[-1], 1.0)))
                except ValueError:
                    pass
            worst = float(v[np.nanargmax(g)])
            dt = np.diff(grid)
            out.append({"variable": var, "limit": kind, "bound": bound, "first_time_s": t_first, "worst_value": worst,
                        "time_beyond_s": float(np.sum(dt[bad[:-1] & bad[1:]]))})
    return out


def _balances(m: Model, y0: np.ndarray, y1: np.ndarray) -> dict:
    qi, nc = m.qi, m.nc
    q = {k: float(y1[i] - y0[i]) for k, i in qi.items()}
    out = {}

    def closure(acc: float, net: float, ref: list[float]) -> dict:
        scale = max([abs(acc), abs(net)] + [abs(r) for r in ref] + [1e-300])
        return {"accumulated": acc, "net_in_minus_out_plus_sources": net, "closure_rel": abs(acc - net) / scale}

    liquid = [h for h in m.holdups]
    if m.columns:
        tot = lambda y: sum(float(np.sum(y[c.iM:c.iM + c.NT])) for c in m.columns)     # noqa: E731
        lig = lambda y: sum(float(np.sum(y[c.iN:c.iN + c.NT])) for c in m.columns)     # noqa: E731
        out["column_total_mol"] = closure(tot(y1) - tot(y0), q["col_in"] - q["col_out"],
                                          [q["col_in"], q["col_out"], tot(y0) * 1e-9])
        out["column_light_mol"] = closure(lig(y1) - lig(y0), q["col_in_light"] - q["col_out_light"],
                                          [q["col_in_light"], q["col_out_light"], lig(y0) * 1e-9])
    if liquid or m.flashes:
        vol = lambda y: sum((y[h.iV] if h.iV >= 0 else h.Vc) for h in liquid)       # noqa: E731
        out["liquid_volume_m3"] = closure(vol(y1) - vol(y0), q["vol_in"] - q["vol_out"] - q["vol_spill"],
                                          [q["vol_in"], q["vol_out"], vol(y0) * 1e-9])
        out["liquid_volume_m3"].update(fed=q["vol_in"], withdrawn=q["vol_out"], spilled=q["vol_spill"])
        if m.flashes:
            out["liquid_volume_m3"]["note"] = "los tambores flash son frontera del balance de volumen (cambio de fase)"
        for i, c in enumerate(m.comp):
            mol = lambda y: sum(y[h.iN + i] for h in liquid) + sum(y[f.iM + i] for f in m.flashes)   # noqa: E731
            b = closure(mol(y1) - mol(y0), q[f"n_in.{c}"] - q[f"n_out.{c}"] + q[f"n_gen.{c}"],
                        [q[f"n_in.{c}"], q[f"n_out.{c}"], q[f"n_gen.{c}"], mol(y0) * 1e-9])
            b.update(fed=q[f"n_in.{c}"], withdrawn=q[f"n_out.{c}"], generated=q[f"n_gen.{c}"])
            out[f"component_mol.{c}"] = b
        eh = [h for h in liquid if h.iE >= 0]
        if eh:
            en = lambda y: sum((h.rhocp or 1.0) * y[h.iE] for h in eh)              # noqa: E731
            b = closure(en(y1) - en(y0), q["e_in"] - q["e_out"] + q["e_heat"] + q["e_rxn"],
                        [q["e_in"], q["e_out"], q["e_heat"], q["e_rxn"]])
            b.update(convected_in=q["e_in"], convected_out=q["e_out"], heat_added=q["e_heat"], reaction_heat=q["e_rxn"],
                     units="J (referencia T = 0 K)" if all(h.rhocp for h in eh) else "m3·K (no se dio ρ·cp)")
            out["liquid_energy"] = b
    if m.vessels:
        mass = lambda y: sum(y[v.im] for v in m.vessels)                             # noqa: E731
        out["gas_mass_kg"] = closure(mass(y1) - mass(y0), q["m_in"] - q["m_out"], [q["m_in"], q["m_out"], mass(y0) * 1e-9])
        ev = [v for v in m.vessels if v.iU >= 0]
        if ev:
            en = lambda y: sum(y[v.iU] for v in ev)                                  # noqa: E731
            if len(ev) == len(m.vessels):
                out["gas_energy_J"] = closure(en(y1) - en(y0), q["eg_in"] - q["eg_out"] + q["eg_heat"],
                                              [q["eg_in"], q["eg_out"], q["eg_heat"], en(y0) * 1e-9])
    return out


# -------------------------------------------------------------------------------------------------------- public API
def _check_numeric(method: str, rtol: float, t_end_s: float, n_points: int):
    if method not in METHODS:
        raise ValueError(f"'method' debe ser uno de {', '.join(METHODS)}")
    if not (1e-13 <= rtol <= 1e-3):
        raise ValueError("'rtol' debe estar entre 1e-13 y 1e-3")
    if t_end_s is None or not math.isfinite(float(t_end_s)) or float(t_end_s) <= 0:
        raise ValueError("falta 't_end_s' (> 0): el horizonte de simulación no se supone")
    if not (2 <= int(n_points) <= 5000):
        raise ValueError("'n_points' debe estar entre 2 y 5000")


def simulate(flowsheet: dict, t_end_s: float, events: list[dict] | None = None, outputs: list[str] | None = None,
             limits: list[dict] | None = None, start_from_steady_state: bool = False, n_points: int = 201,
             method: str = "Radau", rtol: float = 1e-8, atol_factor: float = 1e-12, refine: bool = True,
             parameter_overrides: dict | None = None) -> dict:
    """Run a transient. Returns time series of `outputs`, per-output metrics, incidents (overflow, dry tank), limit
    violations, the final state and the evidence (balances, final residual, tolerance refinement, events hit).
    `parameter_overrides` replaces base values of parameters (used by `step_test` to freeze a controller in manual)."""
    _check_numeric(method, rtol, t_end_s, n_points)
    t_end = float(t_end_s)

    def build() -> tuple[Model, list[dict]]:
        mm = Model(flowsheet, parameter_overrides)
        evs = mm.set_schedule(events, t_end)
        return mm, evs

    m, evs = build()
    names = _resolve_outputs(m, outputs)
    reasons: list[str] = []
    ss_info = None
    if start_from_steady_state:
        ss = steady_state(m)
        ss_info = {"converged": ss["converged"], "newton_correction_rel": ss["newton_correction_rel"],
                   "residual_max_per_s": ss["residual_max_per_s"]}
        if not ss["converged"]:
            raise ValueError("no se encontró el estado estacionario inicial desde el estado dado (corrección de Newton "
                             f"relativa {ss['newton_correction_rel']:.2e}; {ss['solver_message'][:120]}). Dé un estado "
                             "inicial más cercano o use start_from_steady_state=false. Un tanque con salida de caudal fijo "
                             "no tiene nivel estacionario único")
        m.set_initial_dynamic_state(ss["y"])
        m.P.update(m.base)
    y_start = m.y0.copy()
    tr, info = integrate(m, t_end, method, rtol, atol_factor)
    if not tr.sols:
        raise ValueError(f"la integración falló en el primer paso: {info['message']}")
    t_reached = tr.t1s[-1]
    marks = list(m.breakpoints) + [i["time_s"] for i in info["incidents"]]
    grid = _time_grid(tr, t_end, N_FINE, marks)
    fine = sample(tr, names, grid)
    scheduled = sorted({e["time_s"] for e in evs if e["inside_horizon"]} |
                       {e["time_s"] + e["ramp_s"] for e in evs if e["ramp_s"] > 0 and e["time_s"] + e["ramp_s"] <= t_end})
    t_last = max([t for t in scheduled if t < t_reached] + [0.0])
    metrics = {n: _metrics(grid, fine[k], t_last) for k, n in enumerate(names)}
    series_grid = _time_grid(tr, t_end, int(n_points), marks)
    series = sample(tr, names, series_grid)
    final_all = dict(zip(m.variable_names(), tr.values(t_reached, m.variable_names())))
    y_end = tr.activate(t_reached)
    # ---- evidence
    if not info["success"]:
        reasons.append(f"el integrador se detuvo en t = {info.get('t_reached', 0.0):.6g} s: {info['message']}")
    balances = _balances(m, y_start, y_end)
    worst = max([b["closure_rel"] for b in balances.values()] + [0.0])
    if worst > BALANCE_TOL:
        reasons.append(f"un balance global no cierra (peor cierre relativo {worst:.2e} > {BALANCE_TOL:.0e})")
    dy_end = m.rhs(t_reached, y_end)[:m.n_dyn]
    window = max(t_reached - t_last, 1e-300)
    norm_rate = np.abs(dy_end) / np.maximum(np.abs(y_end[:m.n_dyn]), m.scale[:m.n_dyn] * 1e-6)
    steady_change = float(np.max(norm_rate) * window) if m.n_dyn else 0.0
    hit = [{"time_s": t, "restart_at_s": min(info["restarts"], key=lambda r: abs(r - t)) if info["restarts"] else None}
           for t in scheduled]
    for h_ in hit:
        h_["hit_exactly"] = h_["restart_at_s"] is not None and h_["restart_at_s"] == h_["time_s"]
    if any(not h_["hit_exactly"] for h_ in hit if h_["time_s"] < t_reached):
        reasons.append("algún evento programado no coincidió con un reinicio del integrador")
    refinement = None
    if refine and info["success"]:
        m2, _ = build()
        if start_from_steady_state:
            m2.set_initial_dynamic_state(y_start)
        tr2, info2 = integrate(m2, t_end, method, max(rtol / 100.0, 1e-13), atol_factor / 100.0)
        if info2["success"]:
            fine2 = sample(tr2, names, grid)
            inc_dt = 0.0
            if len(info["incidents"]) == len(info2["incidents"]):
                inc_dt = max([abs(a["time_s"] - b["time_s"]) for a, b in zip(info["incidents"], info2["incidents"])] + [0.0])
            else:
                reasons.append("el número de cambios de modo (rebose/vacío) cambia al apretar la tolerancia")
            # a flow jumps at a located mode switch: grid points between the two runs' switch times are not compared
            # (the shift itself is reported and bounded below)
            near = np.zeros(len(grid), dtype=bool)
            for inc in info["incidents"] + info2["incidents"]:
                near |= np.abs(grid - inc["time_s"]) <= inc_dt + 1e-9 * max(t_end, 1.0)
            diffs = {}
            for k, n in enumerate(names):
                ok = np.isfinite(fine[k]) & np.isfinite(fine2[k]) & ~near
                if not ok.any():
                    continue
                span = max(np.ptp(fine2[k][ok]), 1e-9 * max(np.max(np.abs(fine2[k][ok])), 1e-300))
                diffs[n] = float(np.max(np.abs(fine[k][ok] - fine2[k][ok])) / span)
            worst_out = max(diffs, key=diffs.get) if diffs else None
            if inc_dt > 1e-6 * max(t_end, 1e-300):
                reasons.append(f"el instante de un cambio de modo se mueve {inc_dt:.2e} s al apretar la tolerancia")
            refinement = {"rtol": rtol, "rtol_tight": max(rtol / 100.0, 1e-13), "max_difference_rel_to_span":
                          diffs.get(worst_out, 0.0) if worst_out else 0.0, "worst_output": worst_out,
                          "max_incident_time_shift_s": inc_dt}
            if refinement["max_difference_rel_to_span"] > REFINE_TOL:
                reasons.append(f"la solución cambia {refinement['max_difference_rel_to_span']:.2e} del rango de "
                               f"'{worst_out}' al apretar la tolerancia 100 veces (> {REFINE_TOL:.0e})")
        else:
            reasons.append(f"la corrida de tolerancia apretada falló: {info2['message']}")
    incidents = []
    labels = {"full": "rebose: el tanque alcanzó su nivel máximo y derrama",
              "empty": "tanque vacío: la salida queda limitada "
              "a lo que entra", "normal": "vuelve a operar entre sus límites"}
    gas_ids = {v.id for v in m.vessels}
    for inc in info["incidents"]:
        desc = labels[inc["mode"]]
        if inc["unit"] in gas_ids and inc["mode"] == "empty":
            desc = "recipiente de gas vacío: la salida de caudal másico fijo queda limitada a lo que entra"
        incidents.append({"time_s": inc["time_s"], "unit": inc["unit"], "event": {"full": "overflow", "empty": "dry",
                         "normal": "back_in_range"}[inc["mode"]], "description": desc})
    violations = _limit_violations(tr, limits, m, grid)
    warnings = list(m.warnings)
    for s in m.streams:
        if getattr(s, "end_of_curve", False):
            warnings.append(f"la bomba '{s.id}' termina al final de su curva: el caudal está acotado por el último "
                            "punto dado")
    if any(i["event"] == "overflow" for i in incidents):
        warnings.append("hubo rebose: el volumen derramado sale del balance como 'spilled'")
    if any(i["event"] == "dry" and i["unit"] not in gas_ids for i in incidents):
        warnings.append("un tanque se vació: una bomba real cavitaría; aquí la salida se limita a la entrada")
    if any(i["event"] == "dry" and i["unit"] in gas_ids for i in incidents):
        warnings.append("un recipiente de gas se vació con una salida de caudal fijo: un compresor o regulador real no "
                        "podría sostener ese caudal; aquí la salida se limita a lo que entra")
    if violations:
        warnings.append("se violaron límites del usuario: ver 'limit_violations'")
    if any(not e["inside_horizon"] for e in evs):
        warnings.append("hay eventos programados en o después de t_end_s: no se aplicaron")
    at_steady = bool(steady_change < STEADY_RATE_TOL) and info["success"]
    if not at_steady and info["success"]:
        warnings.append("el estado final todavía cambia: no es un estado estacionario (alargue t_end_s si lo busca)")
    if m.delays and len(tr.sols) > 2000:
        warnings.append(f"tiempo muerto muy corto frente al horizonte: {len(tr.sols)} tramos de integración")
    assumptions = _base_assumptions(m) + m.assumptions
    return clean({
        "time_s": series_grid, "series": {n: series[k] for k, n in enumerate(names)}, "metrics": metrics,
        "incidents": incidents, "limit_violations": violations, "events": evs,
        "final_state": final_all, "at_steady_state": at_steady, "initial_steady_state": ss_info,
        "settling_band": SETTLE_BAND, "t_reached_s": t_reached,
        "warnings": warnings, "assumptions": assumptions, "not_included": _not_included(),
        "method": (f"EDO rígidas en forma conservativa (V, V·c, V·T; m, U) integradas con scipy {scipy.__version__} "
                   f"solve_ivp/{method} (rtol {rtol:g}, atol {atol_factor:g}·escala), reinicio en cada evento programado y "
                   "en cada cambio de modo localizado por búsqueda de raíz; tiempo muerto exacto por el método de pasos; "
                   "integrales de frontera integradas junto con el estado"),
        "evidence": {"verified": not reasons, "reasons": reasons, "balances": balances, "worst_balance_closure_rel": worst,
                     "final_normalized_change_over_window": steady_change, "tolerance_refinement": refinement,
                     "scheduled_events": hit, "integrator": {"segments": len(tr.sols), "steps": tr.steps,
                                                             "rhs_evaluations": tr.rhs_evals, "success": info["success"],
                                                             "mode_switches": info["mode_switches"]}},
    })


def _base_assumptions(m: Model) -> list[str]:
    out = []
    if m.holdups:
        out.append("líquido incompresible de densidad y cp constantes (los del usuario); mezcla perfecta en cada retención; "
                   "el volumen se conserva al mezclar")
        if any(h.reactions for h in m.holdups):
            out.append("cinética de ley de potencias r = k·Πc^n con k de Arrhenius o constante; calor de reacción constante")
    if any(v.gas.model == "ideal" for v in m.vessels):
        out.append("gas ideal de cp/cv constante; orificio como tobera isentrópica con coeficiente de descarga "
                   "(flujo crítico "
                   "cuando P_abajo/P_arriba ≤ (2/(k+1))^(k/(k−1))); sin contraflujo desde una frontera")
    if m.vessels:
        out.append("gas bien mezclado en el recipiente; sin capacidad térmica de pared")
    if m.delays:
        out.append("antes de t = 0 el proceso estaba en el estado inicial (historia de las señales con tiempo muerto)")
    if m.controllers:
        out.append("PID ideal (ISA): u = bias + Kc·[e + (1/τI)∫e − τD·d(medición filtrada)/dt], e = setpoint − medición; el "
                   "signo de Kc fija la acción")
    return out


def _not_included() -> list[str]:
    return ["tambor flash y columna sin balance de energía (isotérmico / flujos molares constantes); columnas "
            "multicomponente, con platos de eficiencia < 1 o con hidráulica de vertedero de Francis",
            "alimentación y productos de la columna conectados por corrientes (son parámetros de la columna)",
            "propiedades variables con temperatura o composición en el líquido (ρ, cp constantes)",
            "geometrías de tanque distintas de sección constante; capacidad térmica de paredes",
            "hidráulica de red con presión como incógnita (los caudales salen de leyes explícitas por corriente)",
            "mezclas de gases y condensación en recipientes de gas; golpe de ariete y dinámica de compresores",
            "tiempo muerto sobre la salida de un controlador o sobre un caudal (solo sobre variables de estado)"]


def steady_state_analysis(flowsheet: dict, outputs: list[str] | None = None) -> dict:
    """Steady state reached by Newton iteration from the given initial state, and its local stability (eigenvalues of
    the Jacobian by central differences)."""
    m = Model(flowsheet)
    m.set_schedule(None, 1.0)
    names = _resolve_outputs(m, outputs)
    ss = steady_state(m)
    reasons = []
    if not ss["converged"]:
        reasons.append(f"el sistema no convergió a un estado estacionario (corrección de Newton relativa "
                       f"{ss['newton_correction_rel']:.2e}; {ss['solver_message'][:160]})")
    n = m.n_dyn
    sc = m.scale[:n]
    eig = np.linalg.eigvals(ss["J"]) if n else np.array([])
    eig2 = np.linalg.eigvals(_jacobian(ss["f"], ss["x"], 1e-5)) if n else np.array([])
    eig = eig[np.argsort(-eig.real)]
    eig2 = eig2[np.argsort(-eig2.real)]
    eig_diff = float(np.max(np.abs(np.sort_complex(eig) - np.sort_complex(eig2))) / max(np.max(np.abs(eig)), 1e-300)) if n \
        else 0.0
    max_re = float(eig.real.max()) if n else 0.0
    rate_scale = float(np.max(np.abs(eig))) if n else 1.0
    stable = bool(max_re < -1e-9 * rate_scale)
    marginal = bool(abs(max_re) <= 1e-9 * rate_scale)
    max_re2 = float(eig2.real.max()) if n else 0.0
    abscissa_diff = abs(max_re - max_re2) / max(rate_scale, 1e-300)
    warnings = list(m.warnings)
    # the stability verdict rests on the spectral abscissa; the whole spectrum of a long chain of cells is
    # ill-conditioned (non-normal Jacobian), so its sensitivity is reported but only the abscissa decides
    if abscissa_diff > 1e-4 or (n and (max_re2 < -1e-9 * rate_scale) != stable):
        reasons.append(f"la parte real dominante cambia {abscissa_diff:.1e} con el paso de la diferencia finita: el "
                       "veredicto de estabilidad no es fiable")
    elif eig_diff > 1e-4:
        warnings.append(f"el espectro completo es sensible al paso de la diferencia finita ({eig_diff:.1e}; jacobiano no "
                        "normal, p. ej. muchas celdas en serie): constantes de tiempo secundarias inciertas; el veredicto "
                        "de estabilidad (parte real dominante) sí es robusto")
    listed = m.variable_names() + [k for k in names if k in m.getters and k in m.verbose]
    vals = dict(zip(listed, m.values(0.0, ss["y"], listed)))
    state_names = m.names[:n]
    loop = "en lazo cerrado" if m.controllers else "en lazo abierto"
    if ss["converged"] and not stable:
        warnings.append("estado estacionario " + ("marginal (un valor propio nulo: algún inventario no tiene nivel propio)"
                                                  if marginal else f"INESTABLE {loop}: una perturbación lo aleja"))
    return clean({
        "steady_state": {k: v for k, v in vals.items() if k not in m.verbose}, "outputs": {k: vals.get(k, m.P.get(k))
                                                                                         for k in names},
        "states": dict(zip(state_names, ss["y"][:n])), "converged": ss["converged"],
        "stable": stable if ss["converged"] else None, "marginal": marginal,
        "eigenvalues_per_s": [{"real": float(e.real), "imag": float(e.imag)} for e in eig],
        "time_constants_s": [float(-1.0 / e.real) for e in eig if e.real < 0],
        "oscillation_periods_s": sorted({float(2 * math.pi / abs(e.imag)) for e in eig if abs(e.imag) > 1e-12 * rate_scale}),
        "warnings": warnings, "assumptions": _base_assumptions(m) + m.assumptions + [
            "el estado estacionario hallado es el que Newton alcanza desde el estado inicial dado; con estados múltiples "
            "(CSTR exotérmico) dé un estado inicial cercano al que busca"],
        "not_included": _not_included(),
        "method": (f"dy/dt = 0 con scipy {scipy.__version__} optimize.root (hybr, Powell) sobre los estados escalados; "
                   "jacobiano por diferencias centrales y valores propios con numpy.linalg"),
        "evidence": {"verified": not reasons, "reasons": reasons, "residual_max_per_s": ss["residual_max_per_s"],
                     "newton_correction_rel": ss["newton_correction_rel"], "eigenvalue_step_sensitivity_rel": eig_diff,
                     "dominant_real_part_step_sensitivity_rel": abscissa_diff},
    })


def step_test(flowsheet: dict, input: str, step_size: float, output: str, t_end_s: float, step_time_s: float,
              n_points: int = 200, start_from_steady_state: bool = True, method: str = "Radau", rtol: float = 1e-8) -> dict:
    """Open-loop step test on the nonlinear flowsheet: step `input` (a parameter) by `step_size` at `step_time_s` and record
    `output`. A controller that manipulates `input` is put in manual: it is removed and `input` is frozen at the output
    the controller had at the start (closed-loop steady state if `start_from_steady_state`, else t = 0). The other
    controllers stay in automatic. The result carries the arguments of `fopdt_identify` ready to pass."""
    if step_size is None or not math.isfinite(float(step_size)) or float(step_size) == 0.0:
        raise ValueError("falta 'step_size' (≠ 0): el tamaño del escalón es decisión del usuario")
    if t_end_s is None or not math.isfinite(float(t_end_s)) or float(t_end_s) <= 0:
        raise ValueError("falta 't_end_s' (> 0): la duración de la prueba no se supone")
    if step_time_s is None or not (0.0 <= float(step_time_s) < float(t_end_s)):
        raise ValueError("falta 'step_time_s' o no está entre 0 y t_end_s: el instante del escalón es dato de la prueba")
    full = Model(flowsheet)
    if input not in full.P:
        raise ValueError(f"la entrada '{input}' no es un parámetro del flowsheet. Parámetros: {', '.join(sorted(full.P))}")
    if output not in full.getters:
        raise ValueError(f"la salida '{output}' no es una variable del flowsheet. Variables: "
                         f"{', '.join(full.variable_names())}")
    if any(b.target == input for b in full.blocks):
        raise ValueError(f"'{input}' lo fija un bloque ('target'): no se puede dar un escalón sobre él")
    manual = [c for c in full.controllers if c.target == input]
    extra_assumptions: list[str] = []
    overrides: dict[str, float] = {}
    fs = dict(flowsheet)
    if manual:
        full.set_schedule(None, 1.0)
        y_ref = full.y0
        if start_from_steady_state:
            ss = steady_state(full)
            if not ss["converged"]:
                raise ValueError("no se encontró el estado estacionario en lazo cerrado para pasar a manual el controlador "
                                 f"'{manual[0].id}' (corrección de Newton relativa {ss['newton_correction_rel']:.2e})")
            y_ref = ss["y"]
        full.set_initial_dynamic_state(y_ref)          # delayed measurements see the same state (no history yet)
        u_manual = full.values(0.0, full.y0, [f"{manual[0].id}.output"])[0]
        overrides[input] = u_manual
        ids = {c.id for c in manual}
        fs["controllers"] = [c for c in flowsheet.get("controllers") or [] if c.get("id") not in ids]
        extra_assumptions.append(f"controlador(es) {', '.join(sorted(ids))} en manual: '{input}' congelado en "
                                 f"{u_manual:.6g} "
                                 "(su salida " + ("en el estado estacionario en lazo cerrado)" if start_from_steady_state
                                                  else "en t = 0)"))
        try:
            probe = Model(fs, overrides)
        except ValueError as e:
            raise ValueError(f"al pasar a manual {', '.join(sorted(ids))} el flowsheet deja de ser válido: {e}") from e
    else:
        probe = full
    u0 = probe.P[input]
    res = simulate(fs, t_end_s, events=[{"time_s": float(step_time_s), "target": input, "value": u0 + float(step_size)}],
                   outputs=[output], start_from_steady_state=start_from_steady_state, n_points=n_points, method=method,
                   rtol=rtol, parameter_overrides=overrides or None)
    t = np.array(res["time_s"], dtype=float)
    y = res["series"][output]
    if any(v is None for v in y):
        raise ValueError(f"la salida '{output}' no está definida durante toda la prueba (¿tanque vacío?)")
    y = np.array(y, dtype=float)
    ts = float(step_time_s)
    y_base = float(np.interp(ts, t, y))
    dy = float(y[-1] - y_base)
    warnings = list(res["warnings"])
    pre = t <= ts
    drift = float(np.ptp(y[pre])) if pre.sum() > 1 else 0.0
    if drift > 0.01 * max(abs(dy), 1e-300):
        warnings.append(f"la salida ya cambiaba antes del escalón ({drift:.3g}, más del 1 % de la respuesta): la línea base "
                        "no es estacionaria y la identificación quedará sesgada (use start_from_steady_state)")
    if not res["at_steady_state"]:
        warnings.append("la respuesta no llegó a estado estacionario dentro de t_end_s: la ganancia aparente y el ajuste "
                        "FOPDT serán inciertos (o el proceso es integrador)")
    return clean({
        "time_s": t, "output": y, "input": input, "output_name": output, "input_initial": u0,
        "input_change": float(step_size), "step_time_s": ts, "baseline_output": y_base, "final_output": float(y[-1]),
        "apparent_gain": dy / float(step_size), "pre_step_drift": drift,
        "fopdt_identify_arguments": {"time_s": t, "output": y, "input_change": float(step_size), "step_time_s": ts,
                                     "baseline_output": y_base},
        "controllers_in_manual": sorted(c.id for c in manual), "at_steady_state": res["at_steady_state"],
        "incidents": res["incidents"], "metrics": res["metrics"][output],
        "warnings": warnings, "assumptions": res["assumptions"] + extra_assumptions, "not_included": res["not_included"],
        "method": res["method"] + "; prueba de escalón en lazo abierto sobre el modelo no lineal",
        "evidence": res["evidence"],
    })
