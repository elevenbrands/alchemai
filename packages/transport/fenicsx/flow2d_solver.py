"""Laminar incompressible Navier-Stokes in 2-D (planar or axisymmetric) with FEniCSx: Taylor-Hood P2-P1, Newton
(PETSc SNES + MUMPS) with continuation in Re, optional semi-implicit BDF2 transient and a steady passive scalar.

    (u·∇)u = −∇p + (1/Re) ∇²u,  ∇·u = 0      (solved non-dimensional: lengths / L_ref, velocities / U_ref, p / ρU_ref²)

Weak form with the Laplace (∇u:∇v) viscous term, so the natural condition at an outlet is ν ∂u/∂n − p n = −p0 n; with
the tangential velocity set to zero there it is exactly p = p0 with normal flow. Axisymmetric: (x, y) = (z, r), weight
r, extra u_r v_r / r² and u_r / r terms, u_r = 0 on the axis.

Forces on obstacles come from the reaction of the momentum residual with a lifting test function (consistent,
second-order) and, as a cross-check, from the boundary integral of the stress. The pressure part is ∫ p n ds.

Runs ONLY inside the FEniCSx environment as a separate process launched by `packages.transport.flow2d` (which
validates and documents the payload). Protocol: one JSON object on stdin, one on stdout; progress on stderr.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from dolfinx import fem, geometry, mesh  # noqa: E402
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem  # noqa: E402

import flow2d_geometry as G  # noqa: E402

T0 = time.perf_counter()
LU = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": 120,
      "ksp_error_if_not_converged": False}
REV_TOL = 1e-3                    # reversed flow: streamwise velocity below −REV_TOL·U_ref
_uid = [0]


def log(msg):
    sys.stderr.write(f"[flow2d {time.perf_counter() - T0:7.1f}s] {msg}\n")
    sys.stderr.flush()


def prefix(tag):
    _uid[0] += 1
    return f"f2d_{tag}_{_uid[0]}_"


class Sampler:
    """Point evaluation of dolfinx Functions with a bounding-box tree; points outside the mesh give NaN."""

    def __init__(self, dom):
        self.dom = dom
        self.tree = geometry.bb_tree(dom, 2)

    def __call__(self, f, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        P = np.zeros((len(pts), 3))
        P[:, :2] = pts[:, :2]
        col = geometry.compute_colliding_cells(self.dom, geometry.compute_collisions_points(self.tree, P), P)
        n = f.function_space.dofmap.index_map_bs if f.function_space.dofmap.index_map_bs > 1 else int(np.prod(f.ufl_shape) or 1)
        out = np.full((len(P), n), np.nan)
        idx = [i for i in range(len(P)) if len(col.links(i))]
        if idx:
            out[idx] = f.eval(P[idx], [col.links(i)[0] for i in idx]).reshape(len(idx), -1)
        return out


class Case:
    """One mesh: spaces, boundary tags, forms and solves. All fields non-dimensional."""

    def __init__(self, p, factor):
        self.p = p
        self.geo = geo = G.Geometry(p["mode"], p["rectangles"], p["obstacles"])
        self.axi = geo.axi
        self.bnds = p["boundaries"]
        self.dom, self.mesh_info = G.build_mesh(geo, self.bnds, p["mesh"], factor)
        dom = self.dom
        dom.topology.create_connectivity(1, 2)
        self.samp = Sampler(dom)
        cell = dom.basix_cell()
        self.W = fem.functionspace(dom, basix.ufl.mixed_element([basix.ufl.element("Lagrange", cell, 2, shape=(2,)),
                                                                 basix.ufl.element("Lagrange", cell, 1)]))
        self.V, self.vmap = self.W.sub(0).collapse()
        self.Q, self.qmap = self.W.sub(1).collapse()
        self.x = ufl.SpatialCoordinate(dom)
        self.wgt = self.x[1] if self.axi else 1.0
        self.two_pi = 2 * math.pi if self.axi else 1.0
        self.dx = ufl.Measure("dx", domain=dom, metadata={"quadrature_degree": 6})
        self.nu = fem.Constant(dom, 1.0)
        self.w = fem.Function(self.W)
        self._tag_facets()
        self._bcs()
        nloc = dom.topology.index_map(2).size_local
        hmin = float(np.min(dom.h(2, np.arange(nloc, dtype=np.int32))))
        self.mesh_info.update(cells=int(nloc), dofs=int(self.W.dofmap.index_map.size_local * self.W.dofmap.index_map_bs),
                              h_min=hmin * p["L_ref"], size_factor=factor)

    # ------------------------------------------------------------ boundaries
    def _tag_facets(self):
        dom, geo = self.dom, self.geo
        facets = np.sort(mesh.exterior_facet_indices(dom.topology))
        mid = mesh.compute_midpoints(dom, 1, facets)
        x, y = mid[:, 0], mid[:, 1]
        tag = np.zeros(len(facets), np.int32)
        if self.axi:
            tag[np.abs(y) < 1e-9] = -1                                              # axis
        for i, b in enumerate(self.bnds):
            if b.get("axis"):
                m = G.on_line(b, x, y)
            else:
                o = next(o for o in geo.obstacles if o["name"] == b["obstacle"])
                m = geo.in_obstacle(o, x, y, tol=1e-6)
            tag[(tag == 0) & m] = i + 1
        self.n_default_wall = int(np.sum(tag == 0))
        self.TAG_WALL, self.TAG_AXIS = len(self.bnds) + 1, len(self.bnds) + 2
        tag[tag == 0] = self.TAG_WALL
        tag[tag == -1] = self.TAG_AXIS
        self.facets, self.ftag = facets, tag
        self.ds = ufl.Measure("ds", domain=dom, subdomain_data=mesh.meshtags(dom, 1, facets, tag),
                              metadata={"quadrature_degree": 6})
        for i, b in enumerate(self.bnds):
            if not np.any(tag == i + 1):
                raise ValueError(f"la frontera '{b['name']}' no coincide con ningún tramo del contorno del dominio")

    def fac(self, tag):
        return self.facets[self.ftag == tag]

    def sym_ends(self, b):
        """Whether each end of a line selector touches a symmetry line / the axis (profile centre-line there)."""
        res = []
        for t in b["range"]:
            pt = (b["value"], t) if b["axis"] == "x" else (t, b["value"])
            s = self.axi and abs(pt[1]) < 1e-9 and b["axis"] == "x"
            for o in self.bnds:
                if o["type"] == "symmetry" and o is not b and bool(G.on_line(o, np.array(pt[0]), np.array(pt[1]))):
                    s = True
            res.append(bool(s))
        return res

    def velocity_callable(self, b):
        p, geo = self.p, self.geo
        L, U = p["L_ref"], p["U_ref"]
        if b.get("velocity") is not None:
            trees = [G.parse_expr(e, {"x", "y", "r", "z"}) for e in b["velocity"]]

            def f(X):
                env = {"x": X[0] * L, "y": X[1] * L, "z": X[0] * L, "r": X[1] * L}
                return np.vstack([np.broadcast_to(np.asarray(G.eval_expr(t, env), float) / U, X[0].shape) for t in trees])
            return f
        if b["type"] != "inlet":
            return lambda X: np.zeros((2, X.shape[1]))
        a = 0 if b["axis"] == "x" else 1
        lo, hi = b["range"]
        tm = 0.5 * (lo + hi)
        probe = [0.0, 0.0]
        probe[a], probe[1 - a] = b["value"] + 1e-4, tm
        sign = 1.0 if bool(geo.in_fluid(probe[0], probe[1])) else -1.0
        sl, sh = self.sym_ends(b)
        shape = G.profile_shape(geo, b, sl, sh, b.get("profile", "parabolic"))
        um = b["mean_velocity"] / U

        def f(X):
            v = np.zeros((2, X.shape[1]))
            v[a] = sign * um * np.maximum(shape(np.clip(X[1 - a], lo, hi)), 0.0)
            return v
        return f

    def _bcs(self):
        W, V = self.W, self.V
        self.outlets = []
        order = [(i + 1, b) for i, b in enumerate(self.bnds)]
        # priority among full-vector Dirichlet sets: walls (incl. default) first, then inlets / streams in list order
        walls = [(self.TAG_WALL, {"type": "wall"})] + [(t, b) for t, b in order if b["type"] == "wall"]
        others = [(t, b) for t, b in order if b["type"] in ("inlet", "stream")]
        taken = np.array([], np.int32)
        self.bcs = []
        for t, b in walls + others:
            g = fem.Function(V)
            g.interpolate(self.velocity_callable(b))
            dW, dV = fem.locate_dofs_topological((W.sub(0), V), 1, self.fac(t))
            keep = ~np.isin(dW, taken)
            taken = np.concatenate([taken, dW[keep]])
            if b["type"] == "inlet" and b.get("mean_velocity") is not None:
                # the wall wins at shared nodes; rescale so the DISCRETE flow is exactly mean_velocity × area
                g.x.array[dV[~keep]] = 0.0
                n = ufl.FacetNormal(self.dom)
                flow = fem.assemble_scalar(fem.form(ufl.dot(g, n) * self.wgt * self.ds(t)))
                area = fem.assemble_scalar(fem.form(fem.Constant(self.dom, 1.0) * self.wgt * self.ds(t)))
                target = -b["mean_velocity"] / self.p["U_ref"] * area
                if abs(flow) > 1e-14:
                    g.x.array[:] *= target / flow
            if keep.any():
                self.bcs.append(fem.dirichletbc(g, [dW[keep], dV[keep]], W.sub(0)))
        compsets = [(self.TAG_AXIS, 1)] if self.axi else []
        for t, b in order:
            if b["type"] == "symmetry":
                compsets.append((t, 0 if b["axis"] == "x" else 1))
            elif b["type"] == "outlet":
                self.outlets.append((t, b))
                if b.get("normal_flow", True):
                    compsets.append((t, 1 if b["axis"] == "x" else 0))
        for t, k in compsets:
            Vk, _ = W.sub(0).sub(k).collapse()
            zero = fem.Function(Vk)
            dW, dV = fem.locate_dofs_topological((W.sub(0).sub(k), Vk), 1, self.fac(t))
            keep = ~np.isin(dW, taken)
            if keep.any():
                self.bcs.append(fem.dirichletbc(zero, [dW[keep], dV[keep]], W.sub(0).sub(k)))
        self.pressure_pinned = not self.outlets
        if self.pressure_pinned:
            x0 = self.dom.geometry.x[0]
            Q = self.Q
            d = fem.locate_dofs_geometrical((W.sub(1), Q), lambda X: np.isclose(X[0], x0[0]) & np.isclose(X[1], x0[1]))
            self.bcs.append(fem.dirichletbc(fem.Function(Q), d, W.sub(1)))

    # ------------------------------------------------------------ forms
    def gin(self, u, v):
        e = ufl.inner(ufl.grad(u), ufl.grad(v))
        return e + u[1] * v[1] / self.x[1] ** 2 if self.axi else e

    def div(self, u):
        return ufl.div(u) + u[1] / self.x[1] if self.axi else ufl.div(u)

    def momentum(self, u, p, v, conv=None):
        """Momentum residual tested with v (without boundary terms): ((c·∇)u, v) + ν(∇u, ∇v) − (p, ∇·v)."""
        c = u if conv is None else conv
        return (ufl.inner(ufl.dot(ufl.grad(u), c), v) + self.nu * self.gin(u, v) - p * self.div(v)) * self.wgt * self.dx

    def outlet_terms(self, v):
        n = ufl.FacetNormal(self.dom)
        P = self.p["rho"] * self.p["U_ref"] ** 2
        return sum((fem.Constant(self.dom, float(b.get("pressure", 0.0)) / P) * ufl.dot(n, v) * self.wgt * self.ds(t)
                    for t, b in self.outlets), ufl.zero() * self.wgt * self.dx(domain=self.dom))

    def steady(self, re_target):
        """Newton at the target Re; on failure, continuation in Re from a low-Re solution. Returns a report dict."""
        u, p = ufl.split(self.w)
        v, q = ufl.TestFunctions(self.W)
        F = self.momentum(u, p, v) + q * self.div(u) * self.wgt * self.dx
        if self.outlets:
            F += self.outlet_terms(v)
        opts = dict(LU, snes_type="newtonls", snes_linesearch_type="bt", snes_rtol=1e-10, snes_atol=1e-11,
                    snes_stol=0.0, snes_max_it=25, snes_error_if_not_converged=False)
        prob = NonlinearProblem(F, self.w, bcs=self.bcs, petsc_options_prefix=prefix("ns"), petsc_options=opts)
        self._res_form = fem.form(F)
        rep = {"path": [], "newton_iterations": 0}

        def attempt(re):
            self.nu.value = 1.0 / re
            back = self.w.x.array.copy()
            try:
                prob.solver.setConvergenceHistory(60, reset=True)
                prob.solve()
                reason, its, fn = prob.solver.getConvergedReason(), prob.solver.getIterationNumber(), prob.solver.getFunctionNorm()
                h0 = prob.solver.getConvergenceHistory()[0]
                rep["residual_norm_initial"] = float(h0[0]) if len(h0) else None
            except Exception as ex:  # noqa: BLE001
                reason, its, fn = -99, 0, float("nan")
                log(f"SNES exception at Re {re:g}: {ex}")
            ok = reason > 0 and np.all(np.isfinite(self.w.x.array))
            rep["path"].append({"Re": re, "converged": bool(ok), "iterations": int(its), "residual_norm": float(fn)})
            rep["newton_iterations"] += int(its)
            log(f"Re {re:g}: reason {reason}, its {its}, |F| {fn:.2e}")
            if not ok:
                self.w.x.array[:] = back
                self.w.x.scatter_forward()
            return ok, fn

        ok, fn = attempt(re_target)
        if not ok:
            self.w.x.array[:] = 0.0
            re, step = min(re_target, 1.0), 4.0
            ok, fn = attempt(re)
            while ok and re < re_target * (1 - 1e-12) and step > 1.02 and len(rep["path"]) < 60:
                nxt = min(re * step, re_target)
                ok2, fn2 = attempt(nxt)
                if ok2:
                    re, fn = nxt, fn2
                else:
                    step = math.sqrt(step)
            rep["continuation_used"] = True
            ok = ok and re >= re_target * (1 - 1e-12)
            rep["Re_reached"] = re
            self.nu.value = 1.0 / re_target
        rep["converged"] = bool(ok)
        rep["residual_norm"] = float(fn)
        if self.pressure_pinned:                                      # no pressure boundary: report p with zero mean
            _, p_ = ufl.split(self.w)
            one = fem.Constant(self.dom, 1.0)
            mean = fem.assemble_scalar(fem.form(p_ * self.wgt * self.dx)) / fem.assemble_scalar(fem.form(one * self.wgt * self.dx))
            self.w.x.array[self.qmap] -= mean
        return rep

    def split(self):
        self.uh, self.ph = fem.Function(self.V), fem.Function(self.Q)
        self.uh.x.array[:] = self.w.x.array[self.vmap]
        self.ph.x.array[:] = self.w.x.array[self.qmap]
        return self.uh, self.ph

    # ------------------------------------------------------------ functionals
    def lifting(self, tag, comp):
        phi = fem.Function(self.V)
        bs = self.V.dofmap.index_map_bs
        d = fem.locate_dofs_topological(self.V, 1, self.fac(tag))
        phi.x.array[d * bs + comp] = 1.0
        return phi

    def boundary_flow(self, u, tag):
        n = ufl.FacetNormal(self.dom)
        return self.two_pi * fem.assemble_scalar(fem.form(ufl.dot(u, n) * self.wgt * self.ds(tag)))

    def boundary_area(self, tag):
        return self.two_pi * fem.assemble_scalar(fem.form(fem.Constant(self.dom, 1.0) * self.wgt * self.ds(tag)))

    def boundary_mean_p(self, p, tag):
        return self.two_pi * fem.assemble_scalar(fem.form(p * self.wgt * self.ds(tag))) / self.boundary_area(tag)

    def section(self, u, p, axis, value, lo, hi, n=241):
        """Flow through / mean pressure on an internal line (Simpson on point samples; outside the fluid counts as 0)."""
        t = np.linspace(lo, hi, n)
        t[0] += 1e-9 * (hi - lo)
        t[-1] -= 1e-9 * (hi - lo)
        pts = np.column_stack([np.full(n, value), t] if axis == "x" else [t, np.full(n, value)])
        uv, pv = self.samp(u, pts), self.samp(p, pts)[:, 0]
        ok = np.isfinite(pv)
        un = np.where(ok, uv[:, 0 if axis == "x" else 1], 0.0)
        wr = (2 * math.pi * (pts[:, 1])) if self.axi else np.ones(n)
        sw = np.ones(n)
        sw[1:-1:2], sw[2:-1:2] = 4, 2
        sw *= (hi - lo) / (n - 1) / 3
        area = float(np.sum(sw * wr * ok))
        flow = float(np.sum(sw * wr * un))
        pm = float(np.sum(sw * wr * np.where(ok, pv, 0.0)) / area) if area > 0 else float("nan")
        return {"flow": flow, "area": area, "mean_p": pm, "u_max": float(np.max(np.abs(un))), "u_signed_ext": float(un[np.argmax(np.abs(un))])}


def stream_axis(p):
    sd = p.get("stream_direction", "+x")
    return (0 if sd[1] == "x" else 1), (1.0 if sd[0] == "+" else -1.0)


class Scales:
    def __init__(self, p):
        self.L, self.U, self.rho = p["L_ref"], p["U_ref"], p["rho"]
        axi = p["mode"] == "axisymmetric"
        self.P = self.rho * self.U ** 2
        self.Q = self.U * self.L ** (2 if axi else 1)
        self.F = self.P * self.L ** (2 if axi else 1)
        self.T = self.L / self.U
        self.G = self.P / self.L


def obstacle_forces(c: Case, u, p, tag, o, extra=None):
    """Force of the fluid on an obstacle (non-dimensional): consistent total, pressure part, direct cross-check."""
    n = ufl.FacetNormal(c.dom)
    comps = [0] if c.axi else [0, 1]
    tot, pre, direct = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
    for i in comps:
        phi = c.lifting(tag, i)
        form = c.momentum(u, p, phi) if extra is None else extra(phi)
        tot[i] = -c.two_pi * fem.assemble_scalar(fem.form(form))
        pre[i] = c.two_pi * fem.assemble_scalar(fem.form(p * n[i] * c.wgt * c.ds(tag)))
        tau = c.nu * (ufl.grad(u) + ufl.grad(u).T)
        direct[i] = pre[i] - c.two_pi * fem.assemble_scalar(fem.form(ufl.dot(tau, n)[i] * c.wgt * c.ds(tag)))
    return tot, pre, direct


def wall_extent(c: Case, u, a, sgn, lo, hi, hc):
    """Separation / reattachment of a wall-attached zone from the sign change of the wall shear: the velocity along the
    wall direction `a`, sampled at a distance 0.02·h from an axis-aligned wall next to the zone. None if no such wall."""
    geo = c.geo
    edges = [(r[1 - a], r[3 - a]) for r in geo.rects] + [(o["box"][1 - a], o["box"][3 - a]) for o in geo.obstacles if o["type"] == "rectangle"]
    cands = sorted({e for pair in edges for e in pair})
    best = None
    for tw in cands:
        for side, inner in ((lo[1 - a], 1.0), (hi[1 - a], -1.0)):
            if not (-1e-9 <= inner * (side - tw) <= 2.5 * hc):
                continue
            ss = np.linspace(lo[a] - 6 * hc, hi[a] + 6 * hc, 1200)
            tin, tout = np.full_like(ss, tw + inner * 0.02 * hc), np.full_like(ss, tw - inner * 0.02 * hc)
            pin = np.column_stack([ss, tin] if a == 0 else [tin, ss])
            pout = np.column_stack([ss, tout] if a == 0 else [tout, ss])
            valid = geo.in_fluid(pin[:, 0], pin[:, 1]) & ~geo.in_fluid(pout[:, 0], pout[:, 1])
            v = sgn * c.samp(u, pin)[:, a]
            neg = valid & np.isfinite(v) & (v < 0)
            if not neg.any():
                continue
            ed = np.flatnonzero(np.diff(np.concatenate([[0], neg.astype(int), [0]])))
            runs = sorted(zip(ed[::2], ed[1::2]), key=lambda r_: -(r_[1] - r_[0]))
            i0, i1 = runs[0][0], runs[0][1] - 1
            s0, s1 = ss[i0], ss[i1]
            if i0 > 0 and valid[i0 - 1] and np.isfinite(v[i0 - 1]):
                s0 = ss[i0 - 1] + (ss[i0] - ss[i0 - 1]) * v[i0 - 1] / (v[i0 - 1] - v[i0])
            if i1 < len(ss) - 1 and valid[i1 + 1] and np.isfinite(v[i1 + 1]):
                s1 = ss[i1] + (ss[i1 + 1] - ss[i1]) * (-v[i1]) / (v[i1 + 1] - v[i1])
            if best is None or s1 - s0 > best[2] - best[1]:
                best = (tw, s0, s1)
    return best


def recirculation(c: Case, u, sc: Scales, p, secrects=()):
    """Zones of reversed velocity (connected sets of cells): extent, wake length behind obstacles (centre-line sign
    change) and separation / reattachment on walls (wall-shear sign change). 'Reversed' is measured against the flow
    direction of the straight section the node lies in, else against stream_direction."""
    ax, sg = stream_axis(p)
    V = c.V
    bs = V.dofmap.index_map_bs
    nloc = c.dom.topology.index_map(2).size_local
    Uv = u.x.array.reshape(-1, bs)
    us = sg * Uv[:, ax]
    X = V.tabulate_dof_coordinates()[:, :2]
    axn, sgn = np.full(len(us), ax), np.full(len(us), sg)
    for axis, s0, s1, t0, t1, d in secrects:
        a = 0 if axis == "x" else 1
        m = (X[:, a] >= s0) & (X[:, a] <= s1) & (X[:, 1 - a] >= t0 - 1e-9) & (X[:, 1 - a] <= t1 + 1e-9)
        us[m], axn[m], sgn[m] = d * Uv[m, a], a, d
    cd = V.dofmap.list[:nloc]
    rev = us < -REV_TOL
    cells = np.where(rev[cd].any(axis=1))[0]
    out = {"exists": False, "zones": [], "criterion": f"velocidad en la dirección del flujo principal < −{REV_TOL:g}·U_ref "
           "(dirección del tramo recto donde está el punto; fuera de tramos rectos, stream_direction)"}
    if not len(cells):
        return out
    lab = np.arange(len(cells))
    dl = np.full(len(us), len(cells) + 1)
    for _ in range(10000):
        dl[:] = len(cells) + 1
        np.minimum.at(dl, cd[cells].ravel(), np.repeat(lab, cd.shape[1]))
        new = dl[cd[cells]].min(axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
    gx = c.dom.geometry.x[:, :2]
    gd = c.dom.geometry.dofmap[:nloc]
    tri = gx[gd[:, :3]]
    area = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    L = sc.L
    for lb in np.unique(lab):
        zc = cells[lab == lb]
        if len(zc) < 3:
            continue
        nodes = np.unique(cd[zc].ravel())
        nodes = nodes[rev[nodes]]
        if len(nodes) < 6:
            continue
        xy = X[nodes]
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        worst = nodes[np.argmin(us[nodes])]
        za, zs = int(axn[worst]), float(sgn[worst])
        hc = float(np.sqrt(2 * area[zc].mean()))
        z = {"x_min": lo[0] * L, "x_max": hi[0] * L, "y_min": lo[1] * L, "y_max": hi[1] * L, "main_flow_direction": ("+" if zs > 0 else "-") + "xy"[za],
             "length_streamwise": (hi[za] - lo[za]) * L, "area": float(area[zc].sum()) * L ** 2, "n_cells": int(len(zc)),
             "min_streamwise_velocity": float(us[nodes].min()) * sc.U, "attached_to_obstacle": None,
             "resolution": hc * L}
        for o in c.geo.obstacles:
            if o["type"] != "circle" and not (c.axi and o["box"][1] <= 1e-9):
                continue                                              # wall-mounted plates: use the wall-shear extent
            b = c.geo.obstacle_box(o)
            hloc = 1.5 * hc
            if za != ax:
                continue
            if lo[0] <= b[2] + hloc and hi[0] >= b[0] - hloc and lo[1] <= b[3] + hloc and hi[1] >= b[1] - hloc:
                # wake length: last sign change of the streamwise velocity on the wake centre-line behind the body
                yc = 0.0 if c.axi else 0.5 * (b[1] + b[3])
                xc = 0.5 * (b[0] + b[2])
                rear = (b[2] if sg > 0 else b[0]) if ax == 0 else (b[3] if sg > 0 else b[1])
                far = (hi[ax] if sg > 0 else lo[ax]) + sg * 4 * hloc
                s = np.linspace(rear + sg * 1e-6, far, 600)
                pts = np.column_stack([s, np.full_like(s, yc)] if ax == 0 else [np.full_like(s, xc), s])
                v = sg * c.samp(u, pts)[:, ax]
                k = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
                d, _ = c.geo.obstacle_size(o, ax)
                if len(k):
                    j = k[-1]
                    xz = s[j] + (s[j + 1] - s[j]) * (-v[j]) / (v[j + 1] - v[j])
                    z.update(attached_to_obstacle=o["name"], length_from_obstacle_rear=abs(xz - rear) * L,
                             length_over_obstacle_size=abs(xz - rear) / d)
                else:
                    z.update(attached_to_obstacle=o["name"], length_from_obstacle_rear=abs((hi[ax] if sg > 0 else lo[ax]) - rear) * L,
                             length_over_obstacle_size=abs((hi[ax] if sg > 0 else lo[ax]) - rear) / d,
                             note="sin cambio de signo en la línea central; longitud tomada del extremo de la zona")
                break
        if z["attached_to_obstacle"] is None:
            we = wall_extent(c, u, za, zs, lo, hi, hc)
            if we is not None:
                z["wall"] = {"coordinate": we[0] * L, "from": we[1] * L, "to": we[2] * L, "length": (we[2] - we[1]) * L,
                             "method": "cambio de signo del esfuerzo cortante en la pared"}
        out["zones"].append(z)
    out["zones"].sort(key=lambda z: -z["area"])
    out["exists"] = bool(out["zones"])
    return out


def straight_sections(c: Case, u, p, sc: Scales):
    """Auto-detected straight sections (both long sides wall / symmetry / axis, no obstacle across): fully developed
    pressure gradient, comparison with the exact Poiseuille value for the measured flow, and 99 % entry length."""
    geo = c.geo

    def btype(x, y):
        if c.axi and abs(y) < 1e-9:
            return "symmetry"
        for b in c.bnds:
            if b.get("axis") and bool(G.on_line(b, np.array(x), np.array(y))):
                return "moving" if (b["type"] == "wall" and b.get("velocity") is not None) else b["type"]
        return "wall"

    cands = []
    for r in geo.rects:
        for axis in ("x", "y"):
            a0, a1, t0, t1 = (r[0], r[2], r[1], r[3]) if axis == "x" else (r[1], r[3], r[0], r[2])
            w = t1 - t0
            if a1 - a0 < 2 * w or (c.axi and axis == "y"):
                continue
            s = np.linspace(a0, a1, 801)[1:-1]
            ok = np.ones(len(s), bool)
            types = []
            for tt, outside in ((t0, t0 - 1e-5), (t1, t1 + 1e-5)):
                xs, ys = (s, np.full_like(s, outside)) if axis == "x" else (np.full_like(s, outside), s)
                ok &= ~geo.in_fluid(xs, ys)
                ty = np.array([btype(*((si, tt) if axis == "x" else (tt, si))) for si in s])
                ok &= np.isin(ty, ("wall", "symmetry"))
                types.append(ty)
            for tq in np.linspace(t0, t1, 13)[1:-1]:
                xs, ys = (s, np.full_like(s, tq)) if axis == "x" else (np.full_like(s, tq), s)
                ok &= geo.in_fluid(xs, ys)
            ok &= ~((types[0] == "symmetry") & (types[1] == "symmetry"))
            edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
            for i0, i1 in zip(edges[::2], edges[1::2]):
                if s[i1 - 1] - s[i0] >= 2 * w and len(set(types[0][i0:i1])) == 1 and len(set(types[1][i0:i1])) == 1:
                    cands.append((axis, float(s[i0]), float(s[i1 - 1]), t0, t1, types[0][i0] == "symmetry", types[1][i0] == "symmetry"))
    out, seen = [], set()
    for axis, s0, s1, t0, t1, syl, syh in cands:
        key = (axis, round(s0, 6), round(s1, 6), round(t0, 6), round(t1, 6))
        if key in seen:
            continue
        seen.add(key)
        w, Ls = t1 - t0, s1 - s0
        mid = c.section(u, p, axis, 0.5 * (s0 + s1), t0, t1)
        if abs(mid["flow"]) < 1e-9 * max(mid["area"], 1e-30):
            continue
        d = 1.0 if mid["flow"] > 0 else -1.0
        start, end = (s0, s1) if d > 0 else (s1, s0)
        a, b = start + d * 0.5 * Ls, end - d * min(w, 0.1 * Ls)
        st = np.linspace(a, b, 5)
        secs = [c.section(u, p, axis, v, t0, t1) for v in st]
        pm = [s_["mean_p"] for s_ in secs]
        g1, g2 = -(pm[2] - pm[0]) / abs(st[2] - st[0]), -(pm[4] - pm[2]) / abs(st[4] - st[2])
        g = -(pm[4] - pm[0]) / abs(st[4] - st[0])
        umean = abs(secs[-1]["flow"]) / secs[-1]["area"]
        ex = G.developed_gradient(geo, axis, t0, t1, syl, syh, float(c.nu.value), umean)
        sec = {"axis": axis, "from": start * sc.L, "to": end * sc.L, "across": [t0 * sc.L, t1 * sc.L],
               "sides": ["symmetry" if syl else "wall", "symmetry" if syh else "wall"],
               "flow_direction": ("+" if d > 0 else "-") + axis, "mean_velocity": umean * sc.U,
               "measured_between": [st[0] * sc.L, st[-1] * sc.L],
               "pressure_gradient": -g * sc.G, "pressure_drop_per_length": g * sc.G,
               "gradient_uniformity": abs(g1 - g2) / abs(g) if g else None,
               "u_max_over_mean_at_end": secs[-1]["u_max"] / umean}
        sec["developed"] = bool(g and abs(g1 - g2) <= 0.02 * abs(g))
        if ex is not None:
            ga, ratio = ex
            sec.update(poiseuille_pressure_drop_per_length=ga * sc.G, poiseuille_error_rel=g / ga - 1,
                       poiseuille_u_max_over_mean=ratio)
            # entry length: first station (from the section start, flow direction) where the centre-line velocity
            # is within 1 % of the developed value and stays there
            tc = t0 if syl else (t1 if syh else 0.5 * (t0 + t1))
            if not (c.axi and not syl):
                ss = np.linspace(start + d * 1e-6 * Ls, end - d * 1e-6 * Ls, 801)
                pts = np.column_stack([ss, np.full_like(ss, tc)] if axis == "x" else [np.full_like(ss, tc), ss])
                uc = np.abs(c.samp(u, pts)[:, 0 if axis == "x" else 1]) / (ratio * umean)
                bad = np.where(np.abs(uc - 1) > 0.01)[0]
                sec["centreline_velocity_start_over_developed"] = float(uc[0])
                if len(bad) == 0:
                    sec["entry_length_99"] = 0.0
                elif bad[-1] < len(ss) - 1:
                    sec["entry_length_99"] = abs(ss[bad[-1] + 1] - start) * sc.L
                else:
                    sec["entry_length_99"] = None
        sec["_rect"] = (axis, s0, s1, t0, t1, d)
        out.append(sec)
    return out


def scalar_solve(c: Case, u, sc: Scales, p):
    """Steady passive scalar u·∇c = (1/Pe) ∇²c on P2 with SUPG; Dirichlet values on named boundaries, zero flux elsewhere."""
    s = p["scalar"]
    kap = fem.Constant(c.dom, 1.0 / s["peclet"])
    S = fem.functionspace(c.dom, ("Lagrange", 2))
    ct, psi = ufl.TrialFunction(S), ufl.TestFunction(S)
    names = {b["name"]: i + 1 for i, b in enumerate(c.bnds)}
    bcs, dir_tags = [], {}
    for name, val in s["values"].items():
        t = names[name]
        bcs.append(fem.dirichletbc(fem.Constant(c.dom, float(val)), fem.locate_dofs_topological(S, 1, c.fac(t)), S))
        dir_tags[name] = (t, float(val))
    D0 = fem.functionspace(c.dom, ("DG", 0))                         # cell size as a DG0 field (CellDiameter is not
    hf = fem.Function(D0)                                            # available on quadratic geometry)
    ncell = c.dom.topology.index_map(2).size_local
    hf.x.array[:ncell] = c.dom.h(2, np.arange(ncell, dtype=np.int32))
    h = hf / 2
    un = ufl.sqrt(ufl.dot(u, u) + 1e-30)
    tau = 1.0 / ufl.sqrt((2 * un / h) ** 2 + (12 * kap / h ** 2) ** 2)

    def lap(f):
        return ufl.div(ufl.grad(f)) + (f.dx(1) / c.x[1] if c.axi else 0)

    def res(f, t):
        return (ufl.dot(u, ufl.grad(f)) * t + kap * ufl.dot(ufl.grad(f), ufl.grad(t))) * c.wgt * c.dx \
            + tau * ufl.dot(u, ufl.grad(t)) * (ufl.dot(u, ufl.grad(f)) - kap * lap(f)) * c.wgt * c.dx
    ch = LinearProblem(res(ct, psi), fem.Constant(c.dom, 0.0) * psi * c.wgt * c.dx, bcs=bcs, petsc_options=LU,
                       petsc_options_prefix=prefix("sc")).solve()
    if isinstance(ch, tuple):
        ch = ch[0]
    n = ufl.FacetNormal(c.dom)
    inflow = [v for nm, (t, v) in dir_tags.items() if c.bnds[t - 1]["type"] in ("inlet", "stream")]
    c_in = s.get("reference_inlet_value", inflow[0] if inflow else 0.0)
    out = {"peclet": s["peclet"], "inlet_value": c_in, "surfaces": {}, "outlets": {}, "min": float(ch.x.array.min()),
           "max": float(ch.x.array.max())}
    q = {}
    ax, _ = stream_axis(p)
    for nm, (t, val) in dir_tags.items():
        b = c.bnds[t - 1]
        if b["type"] in ("inlet", "stream") or val == c_in:
            continue
        psi_l = fem.Function(S)
        psi_l.x.array[fem.locate_dofs_topological(S, 1, c.fac(t))] = 1.0
        flux = c.two_pi * fem.assemble_scalar(fem.form(res(ch, psi_l)))             # surface → fluid, consistent
        direct = c.two_pi * fem.assemble_scalar(fem.form(kap * ufl.dot(ufl.grad(ch), n) * c.wgt * c.ds(t)))
        area = c.boundary_area(t)
        o = next((o for o in c.geo.obstacles if o["name"] == b.get("obstacle")), None)
        Lc = c.geo.obstacle_size(o, ax)[0] if o else 1.0
        nu_ = flux / area * Lc / (float(kap.value) * (val - c_in))
        out["surfaces"][nm] = {"nusselt_sherwood_mean": nu_, "characteristic_length": Lc * sc.L,
                               "nusselt_direct_gradient": direct / area * Lc / (float(kap.value) * (val - c_in)),
                               "surface_value": val, "area": area * sc.L ** (2 if c.axi else 1),
                               "peclet_on_characteristic_length": s["peclet"] * Lc}
        q[f"scalar.{nm}.nusselt_sherwood_mean"] = nu_
    for t, b in c.outlets:
        fl = fem.assemble_scalar(fem.form(ufl.dot(u, n) * c.wgt * c.ds(t)))
        if abs(fl) > 1e-14:
            mc = fem.assemble_scalar(fem.form(ch * ufl.dot(u, n) * c.wgt * c.ds(t))) / fl
            out["outlets"][b["name"]] = {"mixing_cup_value": mc}
            q[f"scalar.{b['name']}.mixing_cup_value"] = mc
    hmax = float(np.max(c.dom.h(2, np.arange(c.dom.topology.index_map(2).size_local, dtype=np.int32))))
    out["cell_peclet_max_estimate"] = float(np.abs(u.x.array).max()) * hmax * s["peclet"] / 2
    c.ch = ch
    return out, q


def postprocess(c: Case, p, sc: Scales, rep):
    """All reported quantities of one mesh (dimensional). Returns (result dict, flat dict of comparable numbers)."""
    u, ph = c.split()
    ax, sg = stream_axis(p)
    res, q = {}, {}
    fl = {}
    for i, b in enumerate(c.bnds):
        if b["type"] in ("inlet", "outlet", "stream"):
            t = i + 1
            Q = c.boundary_flow(u, t)
            A = c.boundary_area(t)
            fl[b["name"]] = {"type": b["type"], "flow_out": Q * sc.Q, "mean_normal_velocity_out": Q / A * sc.U,
                             "area": A * sc.L ** (2 if c.axi else 1), "mean_pressure": c.boundary_mean_p(ph, t) * sc.P}
            q[f"flow.{b['name']}"] = Q * sc.Q
            if b["type"] != "outlet":
                q[f"mean_pressure.{b['name']}"] = fl[b["name"]]["mean_pressure"]
    tin = sum(-f["flow_out"] for f in fl.values() if f["flow_out"] < 0)
    tout = sum(f["flow_out"] for f in fl.values() if f["flow_out"] > 0)
    res["boundaries"] = fl
    res["mass_balance"] = {"flow_in": tin, "flow_out": tout, "imbalance": tin - tout,
                           "imbalance_rel": abs(tin - tout) / max(tin, tout, 1e-300)}
    for i, b in enumerate(c.bnds):
        if b["type"] in ("wall", "symmetry") and b.get("axis"):
            fl_ = {"type": b["type"], "mean_pressure": c.boundary_mean_p(ph, i + 1) * sc.P}
            res["boundaries"][b["name"]] = fl_
    res["pressure_drops"] = {}
    for a, b in p.get("pressure_drops", []):
        v = res["boundaries"][a]["mean_pressure"] - res["boundaries"][b]["mean_pressure"]
        res["pressure_drops"][f"{a}->{b}"] = v
        q[f"pressure_drop.{a}->{b}"] = v
    if p.get("points"):
        pts = np.asarray(p["points"], float) / sc.L
        uv, pv = c.samp(u, pts), c.samp(ph, pts)[:, 0]
        res["points"] = []
        for k, (pt, a, b) in enumerate(zip(p["points"], uv, pv)):
            ins = bool(np.isfinite(b))
            res["points"].append({"point": pt, "inside_fluid": ins, "velocity": [a[0] * sc.U, a[1] * sc.U] if ins else None,
                                  "speed": float(np.hypot(a[0], a[1])) * sc.U if ins else None, "pressure": b * sc.P if ins else None})
            if ins:
                q[f"point{k}.velocity_x"], q[f"point{k}.velocity_y"], q[f"point{k}.pressure"] = a[0] * sc.U, a[1] * sc.U, b * sc.P
    res["sections"] = {}
    for s in p.get("sections", []):
        r = c.section(u, ph, s["axis"], s["value"] / sc.L, s["range"][0] / sc.L, s["range"][1] / sc.L)
        res["sections"][s["name"]] = {"flow_positive_axis": r["flow"] * sc.Q, "area": r["area"] * sc.L ** (2 if c.axi else 1),
                                      "mean_velocity": r["flow"] / r["area"] * sc.U if r["area"] else None,
                                      "max_normal_velocity": r["u_signed_ext"] * sc.U, "mean_pressure": r["mean_p"] * sc.P}
        q[f"section.{s['name']}.flow"] = r["flow"] * sc.Q
        q[f"section.{s['name']}.mean_pressure"] = r["mean_p"] * sc.P
    res["straight_sections"] = straight_sections(c, u, ph, sc)
    for k, s in enumerate(res["straight_sections"]):
        q[f"straight_section{k}.pressure_drop_per_length"] = s["pressure_drop_per_length"]
    res["obstacles"] = {}
    for i, b in enumerate(c.bnds):
        if b.get("obstacle") is None:
            continue
        o = next(o for o in c.geo.obstacles if o["name"] == b["obstacle"])
        tot, pre, direct = obstacle_forces(c, u, ph, i + 1, o)
        d, A = c.geo.obstacle_size(o, ax)
        drag, dpre = sg * tot[ax], sg * pre[ax]
        e = {"force": [tot[0] * sc.F, tot[1] * sc.F], "force_pressure": [pre[0] * sc.F, pre[1] * sc.F],
             "force_viscous": [(tot[0] - pre[0]) * sc.F, (tot[1] - pre[1]) * sc.F],
             "drag": drag * sc.F, "drag_pressure": dpre * sc.F, "drag_viscous": (drag - dpre) * sc.F,
             "drag_pressure_fraction": dpre / drag if drag else None,
             "drag_direct_stress_integral": sg * direct[ax] * sc.F,
             "drag_coefficient": drag / (0.5 * A), "drag_coefficient_pressure": dpre / (0.5 * A),
             "drag_coefficient_viscous": (drag - dpre) / (0.5 * A),
             "reference_area": A * sc.L ** (2 if c.axi else 1), "characteristic_length": d * sc.L,
             "reynolds_obstacle": d / float(c.nu.value)}
        if not c.axi:
            e["lift"] = tot[1 - ax] * sc.F
            e["lift_coefficient"] = tot[1 - ax] / (0.5 * A)
        res["obstacles"][o["name"]] = e
        q[f"obstacle.{o['name']}.drag"] = e["drag"]
        q[f"obstacle.{o['name']}.drag_pressure"] = e["drag_pressure"]
    res["recirculation"] = recirculation(c, u, sc, p, [s.pop("_rect") for s in res["straight_sections"]])
    for k, z in enumerate(res["recirculation"]["zones"][:3]):
        q[f"recirculation{k}.length"] = z.get("length_from_obstacle_rear", z.get("wall", {}).get("length", z["length_streamwise"]))
    q["recirculation.count"] = float(len(res["recirculation"]["zones"]))
    spd = np.hypot(*u.x.array.reshape(-1, 2).T)
    res["velocity_max"] = float(spd.max()) * sc.U
    res["pressure_min_max"] = [float(ph.x.array.min()) * sc.P, float(ph.x.array.max()) * sc.P]
    if p.get("scalar"):
        res["scalar"], qs = scalar_solve(c, u, sc, p)
        q.update(qs)
    return res, q


# ---------------------------------------------------------------- transient
def analyse_signal(t, s, scale):
    """Classify a signal as steady / periodic / not settled; frequency from upward zero crossings of the last cycles."""
    t, s = np.asarray(t), np.asarray(s)
    out = {"state": "not_settled", "frequency": None, "n_cycles": 0}
    if len(t) < 20:
        return out
    half = t >= t[0] + 0.5 * (t[-1] - t[0])
    last = t >= t[0] + 0.8 * (t[-1] - t[0])
    amp_last = 0.5 * (s[last].max() - s[last].min())
    out["amplitude_last_fifth"] = float(amp_last)
    if amp_last < 1e-4 * scale:
        prev = (t >= t[0] + 0.6 * (t[-1] - t[0])) & ~last
        if abs(s[last].mean() - s[prev].mean()) < 1e-4 * scale:
            out["state"] = "steady"
        return out
    th, sh = t[half], s[half]
    z = sh - 0.5 * (sh.max() + sh.min())
    k = np.where((z[:-1] < 0) & (z[1:] >= 0))[0]
    if len(k) < 4:
        return out
    tc = th[k] + (th[k + 1] - th[k]) * (-z[k]) / (z[k + 1] - z[k])
    per = np.diff(tc)
    peaks = np.array([sh[(th >= a) & (th < b)].max() - sh[(th >= a) & (th < b)].min() for a, b in zip(tc[:-1], tc[1:])]) / 2
    out["n_cycles"] = int(len(per))
    n = min(4, len(per))
    out["frequency"] = float(1 / per[-n:].mean())
    out["period_spread_rel"] = float(np.ptp(per[-n:]) / per[-n:].mean())
    out["amplitude_spread_rel"] = float(np.ptp(peaks[-n:]) / peaks[-n:].mean())
    out["amplitude"] = float(peaks[-n:].mean())
    out["window"] = [float(tc[-n - 1]), float(tc[-1])]
    if len(per) >= 4 and out["period_spread_rel"] < 0.01 and out["amplitude_spread_rel"] < 0.02:
        out["state"] = "periodic"
    return out


def transient(c: Case, p, sc: Scales, deadline):
    """Semi-implicit BDF2 (convection linearised with the extrapolated velocity), one LU per step, from the steady
    solution plus a transverse perturbation. Monitors drag / lift on the first obstacle (or v at the first point)."""
    tp = p["transient"]
    ax, sg = stream_axis(p)
    dt, t_end = tp["dt"] / sc.T, tp["t_end"] / sc.T
    W, V = c.W, c.V
    un, unm1, ustar, hist = (fem.Function(V) for _ in range(4))
    un.x.array[:] = c.w.x.array[c.vmap]
    obs = next(((i + 1, b) for i, b in enumerate(c.bnds) if b.get("obstacle") is not None), None)
    if obs:
        o = next(o for o in c.geo.obstacles if o["name"] == obs[1]["obstacle"])
        bx = c.geo.obstacle_box(o)
        d, A = c.geo.obstacle_size(o, ax)
        ctr = [0.5 * (bx[0] + bx[2]), 0.5 * (bx[1] + bx[3])]
        ctr[ax] += sg * 1.5 * d
    else:
        d, A = 1.0, 1.0
        ctr = [0.5 * (c.geo.bbox[0] + c.geo.bbox[2]), 0.5 * (c.geo.bbox[1] + c.geo.bbox[3])]
    eps = tp.get("perturbation", 0.2)
    pert = fem.Function(V)

    def bump(X):
        v = np.zeros((2, X.shape[1]))
        v[1 - ax] = eps * np.exp(-((X[0] - ctr[0]) ** 2 + (X[1] - ctr[1]) ** 2) / d ** 2)
        return v
    pert.interpolate(bump)
    un.x.array[:] += pert.x.array
    unm1.x.array[:] = un.x.array
    a0 = fem.Constant(c.dom, 1.0)
    idt = fem.Constant(c.dom, 1.0 / dt)
    (u, pr), (v, qq) = ufl.TrialFunctions(W), ufl.TestFunctions(W)
    a = a0 * idt * ufl.inner(u, v) * c.wgt * c.dx + c.momentum(u, pr, v, conv=ustar) + qq * c.div(u) * c.wgt * c.dx
    Lf = idt * ufl.inner(hist, v) * c.wgt * c.dx
    if c.outlets:
        Lf -= c.outlet_terms(v)
    wn = fem.Function(W)
    prob = LinearProblem(a, Lf, bcs=c.bcs, u=wn, petsc_options=LU, petsc_options_prefix=prefix("tr"))
    us, ps = ufl.split(wn)
    forms = None
    if obs:
        forms = []
        for i in ([0] if c.axi else [0, 1]):
            phi = c.lifting(obs[0], i)
            forms.append(fem.form(idt * ufl.inner(a0 * us - hist, phi) * c.wgt * c.dx + c.momentum(us, ps, phi, conv=ustar)))
    probe = np.asarray(p["points"][0], float) / sc.L if p.get("points") else None
    ts, cd, sig = [], [], []
    t, step, stopped = 0.0, 0, "t_end"
    use_lift = obs is not None and not c.axi
    t_step = None
    while t < t_end - 1e-12:
        t1 = time.perf_counter()
        if step == 0:
            a0.value, hist.x.array[:], ustar.x.array[:] = 1.0, un.x.array, un.x.array
        else:
            a0.value = 1.5
            hist.x.array[:] = 2.0 * un.x.array - 0.5 * unm1.x.array
            ustar.x.array[:] = 2.0 * un.x.array - unm1.x.array
        prob.solve()
        if not np.all(np.isfinite(wn.x.array)):
            stopped = "diverged"
            break
        t += dt
        step += 1
        unm1.x.array[:] = un.x.array
        un.x.array[:] = wn.x.array[c.vmap]
        ts.append(t)
        if forms:
            f = [-c.two_pi * fem.assemble_scalar(fm) for fm in forms]
            cd.append(sg * f[0 if c.axi else ax] / (0.5 * A))
            sig.append(f[1 - ax] / (0.5 * A) if use_lift else cd[-1])
        else:
            sig.append(float(c.samp(un, [probe])[0, 1 - ax]) if probe is not None else float(np.abs(un.x.array).max()))
        t_step = time.perf_counter() - t1 if t_step is None else 0.9 * t_step + 0.1 * (time.perf_counter() - t1)
        if step % 50 == 0:
            an = analyse_signal(ts, sig, 1.0)
            log(f"t* {t:.1f} step {step} sig {sig[-1]:+.4f} state {an['state']} cycles {an['n_cycles']} ({t_step:.2f} s/step)")
            if step >= tp.get("min_steps", 200) and an["state"] in ("periodic", "steady") and tp.get("stop_when_settled", True) \
                    and (an["state"] == "steady" or an["n_cycles"] >= tp.get("min_cycles", 8)):
                stopped = "settled"
                break
        if time.perf_counter() - T0 + 3 * t_step > deadline:
            stopped = "time_budget"
            break
    c.w.x.array[:] = wn.x.array
    an = analyse_signal(ts, sig, 1.0) if stopped != "diverged" else {"state": "diverged", "frequency": None, "n_cycles": 0}
    out = {"state": an["state"], "stopped_by": stopped, "steps": step, "dt": dt * sc.T, "time_simulated": t * sc.T,
           "signal": ("lift_coefficient" if use_lift else "drag_coefficient") if forms else "transverse_velocity_at_first_point",
           "analysis": an, "seconds_per_step": t_step, "characteristic_length": d * sc.L}
    if an.get("frequency"):
        out["frequency"] = an["frequency"] / sc.T
        out["strouhal"] = an["frequency"] * d                       # f d / U_ref
        out["steps_per_period"] = 1 / (an["frequency"] * dt)
        w0, w1 = an["window"]
        m = (np.asarray(ts) >= w0) & (np.asarray(ts) <= w1)
        if forms:
            out["drag_coefficient_mean"] = float(np.mean(np.asarray(cd)[m]))
            out["drag_coefficient_amplitude"] = float(0.5 * np.ptp(np.asarray(cd)[m]))
            if use_lift:
                out["lift_coefficient_amplitude"] = an["amplitude"]
                out["lift_coefficient_rms"] = float(np.std(np.asarray(sig)[m]))
    elif forms and an["state"] == "steady":
        out["drag_coefficient_mean"] = float(cd[-1])
    stride = max(1, len(ts) // 400)
    out["history"] = {"time": [x * sc.T for x in ts[::stride]], "signal": sig[::stride], "drag_coefficient": cd[::stride] if forms else None}
    return out


def save_figure(c: Case, p, sc: Scales, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    P1v = fem.functionspace(c.dom, ("Lagrange", 1, (2,)))
    u1 = fem.Function(P1v)
    u1.interpolate(c.uh)
    X = P1v.tabulate_dof_coordinates()[:, :2] * sc.L
    nloc = c.dom.topology.index_map(2).size_local
    tri = mtri.Triangulation(X[:, 0], X[:, 1], P1v.dofmap.list[:nloc])
    uv = u1.x.array.reshape(-1, 2) * sc.U
    ax_, sg = stream_axis(p)
    Q1 = fem.functionspace(c.dom, ("Lagrange", 1))
    p1 = fem.Function(Q1)
    p1.interpolate(c.ph)
    bb = c.geo.bbox
    asp = (bb[3] - bb[1]) / (bb[2] - bb[0])
    hgt = min(max(10 * asp, 1.6), 5.0)
    fig, axs = plt.subplots(2, 1, figsize=(10, 2 * hgt + 1.2), dpi=140)
    pc = axs[0].tripcolor(tri, np.hypot(uv[:, 0], uv[:, 1]), shading="gouraud", cmap="viridis")
    fig.colorbar(pc, ax=axs[0], label="|u|", pad=0.01)
    try:
        axs[0].tricontour(tri, sg * uv[:, ax_], levels=[0.0], colors="w", linewidths=0.8)
    except Exception:  # noqa: BLE001
        pass
    axs[0].set_title(title, fontsize=10)
    pc = axs[1].tripcolor(tri, p1.x.array * sc.P, shading="gouraud", cmap="coolwarm")
    fig.colorbar(pc, ax=axs[1], label="p", pad=0.01)
    for a in axs:
        if asp > 0.05:
            a.set_aspect("equal")
        a.set_xlabel("z" if c.axi else "x")
        a.set_ylabel("r" if c.axi else "y")
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    p = json.loads(sys.stdin.read())
    sc = Scales(p)
    L = sc.L
    # non-dimensional geometry (lengths / L_ref)
    p["rectangles"] = [[v / L for v in r] for r in p["rectangles"]]
    for o in p["obstacles"]:
        if o["type"] == "circle":
            o["center"], o["radius"] = [v / L for v in o["center"]], o["radius"] / L
        else:
            o["box"] = [v / L for v in o["box"]]
        o["mesh_size"] = o["mesh_size"] / L
    for b in p["boundaries"]:
        if b.get("axis"):
            b["value"], b["range"] = b["value"] / L, [v / L for v in b["range"]]
    mp = p["mesh"]
    mp["size"] /= L
    for bx in mp.get("boxes", []):
        bx["box"], bx["size"] = [v / L for v in bx["box"]], bx["size"] / L
    re = sc.rho * sc.U * sc.L / p["mu"]
    deadline = float(p.get("time_budget_s", 1e9))
    out = {"Re_reference": re, "dolfinx_version": __import__("dolfinx").__version__, "timing_s": {}}
    runs = []
    factors = [1.0, float(mp.get("refine_ratio", 1.5))] if mp.get("refinement_check", True) else [1.0]
    for k, fct in enumerate(factors):
        t1 = time.perf_counter()
        c = Case(p, fct)
        log(f"mesh x{fct:g}: {c.mesh_info['cells']} cells, {c.mesh_info['dofs']} dofs")
        rep = c.steady(re)
        if not rep["converged"] and not p.get("transient"):
            runs.append((c, rep, None, None))
            out["timing_s"][f"mesh{k}"] = time.perf_counter() - t1
            break
        res, q = postprocess(c, p, sc, rep)
        runs.append((c, rep, res, q))
        out["timing_s"][f"mesh{k}"] = time.perf_counter() - t1
    c, rep, res, q = runs[-1]
    out["steady"] = {"converged": rep["converged"], "newton": rep, "mesh": c.mesh_info}
    out["meshes"] = [r[0].mesh_info for r in runs]
    out["default_wall_facets"] = c.n_default_wall
    out["pressure_pinned"] = c.pressure_pinned
    if res is not None:
        out["result"] = res
        out["quantities"] = q
        if len(runs) == 2 and runs[0][3] is not None:
            out["quantities_coarse"] = runs[0][3]
            out["newton_coarse"] = runs[0][1]
        if p.get("figure_path"):
            save_figure(c, p, sc, p["figure_path"], p.get("figure_title", "Flujo laminar 2-D (FEniCSx)"))
            out["figure_path"] = p["figure_path"]
    if p.get("transient"):
        t1 = time.perf_counter()
        ct = runs[0][0] if p["transient"].get("mesh", "base") == "base" else c
        if not runs[0][1]["converged"]:
            log("steady solution not converged: transient starts from the last Newton state")
        tr = transient(ct, p, sc, deadline)
        tr["mesh"] = ct.mesh_info
        out["transient"] = tr
        out["timing_s"]["transient"] = time.perf_counter() - t1
    out["timing_s"]["total"] = time.perf_counter() - T0
    sys.stdout.write(json.dumps(out, default=float))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
