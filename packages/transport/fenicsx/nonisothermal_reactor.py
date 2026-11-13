"""Non-isothermal tubular reactor solved as a coupled 2-D axisymmetric field (FEniCSx): concentration and temperature,
first-order Arrhenius kinetics, heat of reaction, radial and axial dispersion/conduction, in ONE tube of the bundle.

    u(r) ∂c/∂z = D_r (1/r) ∂/∂r (r ∂c/∂r) + D_z ∂²c/∂z² − k(T) c
    ρ cp u(r) ∂T/∂z = λ_r (1/r) ∂/∂r (r ∂T/∂r) + λ_z ∂²T/∂z² + (−ΔH) k(T) c,      k(T) = A e^(−E/RT)

Inlet (z = 0): Danckwerts, u (c − C0) = D_z ∂c/∂z and ρ cp u (T − T0) = λ_z ∂T/∂z (total inflow = feed exactly).
Outlet: zero gradient. Axis: symmetry. Wall: no mass flux and one of
    fixed_T    T = T_wall (Dirichlet)
    coolant    −λ_r ∂T/∂r = U (T − T_coolant)          (U is the wall-to-coolant coefficient, coolant at constant T)
    adiabatic  ∂T/∂r = 0
Velocity: Poiseuille u(r) = 2 ū (1 − r²/R²) ("laminar") or u = ū ("plug", the pseudo-homogeneous fixed-bed model of
Froment with effective D_er, λ_er).

Unknowns are scaled, y = c/C0 and θ = T/T0. P2 elements so the SUPG residual carries the radial and axial
diffusion terms (with P1 they vanish element-wise and a large λ_r makes the stabilisation inconsistent). With ideal_gas (plug or frozen-mass-flux laminar profile, constant P,
dilute reactant) ρu = G is constant, y is the molar-flow ratio of A and c = C0·y/θ, so the rate carries a 1/θ factor
and u is the feed velocity; ρD and λ are taken constant. P2 × P2 mixed space, SUPG on both equations with
τ = [(2u/h)² + 9 (4κ/h²)²]^(−1/2), h = Δz/2 (axial κ: the streamline direction; vanishes where u → 0 at a laminar wall), Newton (PETSc SNES, LU) with continuation
on the heat of reaction from ΔH = 0 to the user's ΔH, halving the step when Newton fails; if that stalls (ignition),
pseudo-transient continuation (implicit Euler in pseudo-time, Δt growing) at the full ΔH, then a steady Newton solve.

Global balances. With test function 1 the SUPG term vanishes and Danckwerts adds no Dirichlet row, so the discrete
Galerkin solution conserves mass and energy over the whole tube. The energy closure H_out − H_in + Q_wall − Q_gen is
therefore a check that Newton converged and the boundary terms are coded right; the accuracy check is the mesh study.
Q_wall: coolant/adiabatic → U ∫ (T − Tc) dA (the boundary condition itself); fixed_T → consistent (residual) flux.

Runs ONLY inside the FEniCSx environment (dolfinx ≥ 0.11, LGPL-3) as a separate process launched by
`packages.transport.reactor_nonisothermal`. Protocol: one JSON object on stdin, one on stdout; progress on stderr.

Input keys: A_1_s, E_J_mol, dH_J_mol, C0_mol_m3, T0_K, rho_kg_m3, cp_J_kgK, D_r, D_z, lambda_r, lambda_z,
            tube_radius_m, tube_length_m, u_mean_m_s (at feed conditions), ideal_gas (bool), profile ("laminar"|"plug"), wall {"mode", "T_wall_K" |
            "U_W_m2K"+"T_coolant_K"}, nz, nr, mesh_study (bool), n_stations (axial), n_radial.
"""
from __future__ import annotations

import json
import sys
import time

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, geometry, la, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

R_GAS = 8.314462618
QDEG = 6
DEGREE = 2


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Case:
    def __init__(self, p: dict):
        self.A, self.E, self.dH = float(p["A_1_s"]), float(p["E_J_mol"]), float(p["dH_J_mol"])
        self.C0, self.T0 = float(p["C0_mol_m3"]), float(p["T0_K"])
        self.rho, self.cp = float(p["rho_kg_m3"]), float(p["cp_J_kgK"])
        self.Dr, self.Dz = float(p["D_r"]), float(p["D_z"])
        self.lr, self.lz = float(p["lambda_r"]), float(p["lambda_z"])
        self.R, self.L, self.um = float(p["tube_radius_m"]), float(p["tube_length_m"]), float(p["u_mean_m_s"])
        self.profile = p["profile"]
        self.wall = p["wall"]
        self.rcp = self.rho * self.cp
        self.ideal_gas = bool(p.get("ideal_gas", False))


def build(case: Case, nz: int, nr: int):
    dom = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [case.L, case.R]], [nz, nr], cell_type=mesh.CellType.triangle)
    Pk = basix.ufl.element("Lagrange", dom.basix_cell(), DEGREE)
    W = fem.functionspace(dom, basix.ufl.mixed_element([Pk, Pk]))
    w = fem.Function(W)
    w.sub(0).interpolate(lambda X: np.ones(X.shape[1]))
    w.sub(1).interpolate(lambda X: np.ones(X.shape[1]))
    fdim = 1
    inlet = mesh.locate_entities_boundary(dom, fdim, lambda X: np.isclose(X[0], 0.0))
    outlet = mesh.locate_entities_boundary(dom, fdim, lambda X: np.isclose(X[0], case.L))
    wallf = mesh.locate_entities_boundary(dom, fdim, lambda X: np.isclose(X[1], case.R))
    idx = np.concatenate([inlet, outlet, wallf])
    val = np.concatenate([np.full(len(inlet), 1), np.full(len(outlet), 2), np.full(len(wallf), 3)]).astype(np.int32)
    order = np.argsort(idx)
    tags = mesh.meshtags(dom, fdim, idx[order], val[order])
    return dom, W, w, tags, wallf


def forms(case: Case, dom, W, w, tags, s_dH: fem.Constant, nz: int):
    x = ufl.SpatialCoordinate(dom)
    r = x[1]
    u = 2 * case.um * (1 - (r / case.R) ** 2) if case.profile == "laminar" else fem.Constant(dom, case.um)
    y, th = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    dx = ufl.Measure("dx", domain=dom, metadata={"quadrature_degree": QDEG})
    ds = ufl.Measure("ds", domain=dom, subdomain_data=tags, metadata={"quadrature_degree": QDEG})
    h = fem.Constant(dom, case.L / nz / DEGREE)          # element length along the streamline (z), per node spacing
    ar, az = case.lr / case.rcp, case.lz / case.rcp
    thp = ufl.max_value(th, 0.2)
    k = case.A * ufl.exp(-case.E / (R_GAS * case.T0 * thp))
    if case.ideal_gas:   # ρu = G constant, c = C0·y·T0/T: the rate per volume carries 1/θ (y = molar-flow ratio)
        k = k / thp
    beta = s_dH * (-case.dH) * case.C0 / (case.rcp * case.T0)
    tau_c = 1.0 / ufl.sqrt((2 * u / h) ** 2 + 9 * (4 * case.Dz / h ** 2) ** 2)
    tau_t = 1.0 / ufl.sqrt((2 * u / h) ** 2 + 9 * (4 * az / h ** 2) ** 2)
    # full strong residuals (P2 has second derivatives): SUPG stays consistent even when radial diffusion is large
    lap_r = lambda f: f.dx(1).dx(1) + f.dx(1) / r
    Rc = u * y.dx(0) - case.Dz * y.dx(0).dx(0) - case.Dr * lap_r(y) + k * y
    Rt = u * th.dx(0) - az * th.dx(0).dx(0) - ar * lap_r(th) - beta * k * y
    Fc = (u * y.dx(0) * v + case.Dz * y.dx(0) * v.dx(0) + case.Dr * y.dx(1) * v.dx(1) + k * y * v) * r * dx \
        + tau_c * u * v.dx(0) * Rc * r * dx + u * (y - 1.0) * v * r * ds(1)
    Ft = (u * th.dx(0) * q + az * th.dx(0) * q.dx(0) + ar * th.dx(1) * q.dx(1) - beta * k * y * q) * r * dx \
        + tau_t * u * q.dx(0) * Rt * r * dx + u * (th - 1.0) * q * r * ds(1)
    mode = case.wall["mode"]
    if mode == "coolant":
        thc = case.wall["T_coolant_K"] / case.T0
        Ft += (case.wall["U_W_m2K"] / case.rcp) * (th - thc) * q * r * ds(3)
    return Fc + Ft, dict(u=u, y=y, th=th, k=k, r=r, dx=dx, ds=ds, beta=beta)


def solve(case: Case, nz: int, nr: int, tag: str):
    dom, W, w, tags, wallf = build(case, nz, nr)
    s = fem.Constant(dom, 0.0)
    F, aux = forms(case, dom, W, w, tags, s, nz)
    bcs = []
    if case.wall["mode"] == "fixed_T":
        W1, _ = W.sub(1).collapse()
        g = fem.Function(W1)
        g.x.array[:] = case.wall["T_wall_K"] / case.T0
        dofs = fem.locate_dofs_topological((W.sub(1), W1), 1, wallf)
        bcs = [fem.dirichletbc(g, dofs, W.sub(1))]
    # pseudo-transient term (implicit Euler toward steady state); inv_dt = 0 is the steady problem
    w_prev = fem.Function(W)
    inv_dt = fem.Constant(dom, 0.0)
    y, th = ufl.split(w)
    yp, thp = ufl.split(w_prev)
    v, q = ufl.TestFunctions(W)
    F_solve = F + inv_dt * ((y - yp) * v + (th - thp) * q) * aux["r"] * aux["dx"]
    opts = {"snes_type": "newtonls", "snes_linesearch_type": "bt", "snes_rtol": 1e-10, "snes_atol": 1e-12, "snes_stol": 1e-12,
            "snes_max_it": 60, "ksp_type": "preonly", "pc_type": "lu"}
    prob = NonlinearProblem(F_solve, w, bcs=bcs, petsc_options=opts, petsc_options_prefix=f"nirx_{tag}_{nz}_")
    snes = prob.solver
    ok = lambda: snes.getConvergedReason() > 0 and np.all(np.isfinite(w.x.array))
    s_val, ds_step, last_good, iters = 0.0, 1.0, w.x.array.copy(), 0
    steps, ptc = [], None
    while True:
        s.value = s_val
        prob.solve()
        if ok():
            last_good = w.x.array.copy()
            iters += snes.getIterationNumber()
            steps.append(round(s_val, 6))
            if s_val >= 1.0:
                break
            s_val = min(1.0, s_val + ds_step)
            ds_step = min(1.0, ds_step * 1.5)
            continue
        w.x.array[:] = last_good
        prev = steps[-1] if steps else 0.0
        ds_step = (s_val - prev) / 2
        if ds_step >= 1.0 / 32:
            s_val = prev + ds_step
            log(f"[{tag}] Newton falló; reintento s={s_val:.4f}")
            continue
        # natural continuation cannot pass the ignition: march in pseudo-time at the full ΔH from the last good state
        log(f"[{tag}] continuación detenida en s={prev:.4f}: pseudo-transitorio a ΔH completo")
        s.value = 1.0
        tau_res = case.L / case.um
        dt, n_ok, n_try = tau_res / 50, 0, 0
        while n_try < 150:
            n_try += 1
            w_prev.x.array[:] = w.x.array
            inv_dt.value = 1.0 / dt
            prob.solve()
            if ok():
                n_ok += 1
                iters += snes.getIterationNumber()
                change = float(np.max(np.abs(w.x.array - w_prev.x.array)))
                dt *= 2.0 if snes.getIterationNumber() <= 6 else 1.2
                if change < 1e-10 or dt > 1e6 * tau_res:
                    break
            else:
                w.x.array[:] = w_prev.x.array
                dt /= 4
                if dt < 1e-9 * tau_res:
                    break
        inv_dt.value = 0.0
        prob.solve()
        if not ok():
            raise RuntimeError(f"Newton no convergió en la malla {nz}×{nr}: la continuación en ΔH se detuvo en {prev:.4f} "
                               f"y el pseudo-transitorio ({n_ok} pasos) no llegó a un estado estacionario. Posible "
                               "desbocamiento: revisa los datos o refina la malla.")
        iters += snes.getIterationNumber()
        ptc = {"from_s": prev, "pseudo_time_steps": n_ok}
        steps.append(1.0)
        break
    reasons = {2: "atol", 3: "rtol", 4: "stol"}
    info_reason = reasons.get(snes.getConvergedReason(), str(snes.getConvergedReason()))
    # polish: a few Newton steps with tight tolerances; kept only if the residual goes down (closure is checked anyway)
    norm0, x0 = float(snes.getFunctionNorm()), w.x.array.copy()
    snes.setTolerances(rtol=1e-15, atol=1e-15, stol=1e-18, max_it=6)
    try:
        prob.solve()
        norm1 = float(snes.getFunctionNorm())
    except Exception:  # noqa: BLE001
        norm1 = float("inf")
    if not (np.all(np.isfinite(w.x.array)) and norm1 <= norm0):
        w.x.array[:] = x0
    else:
        info_reason += f"+pulido({norm0:.1e}→{norm1:.1e})"
    log(f"[{tag}] {nz}×{nr}: convergió, {iters} iteraciones de Newton, pasos de ΔH {steps}")
    return dom, W, w, tags, aux, F, bcs, {"newton_iterations": int(iters), "continuation_steps": steps, "pseudo_transient": ptc,
                                                       "snes_reason": info_reason, "residual_norm": float(snes.getFunctionNorm())}


def balances(case: Case, dom, W, w, aux, F, bcs):
    """Every term in W (energy) and mol/s (species), per tube."""
    u, y, th, k, r, dx, ds = aux["u"], aux["y"], aux["th"], aux["k"], aux["r"], aux["dx"], aux["ds"]
    two_pi = 2 * np.pi
    a = lambda e: two_pi * float(fem.assemble_scalar(fem.form(e)))
    rcpT0 = case.rcp * case.T0
    F_in = a(u * case.C0 * r * ds(1))
    F_out = a(u * case.C0 * y * r * ds(2))
    rate = a(k * case.C0 * y * r * dx)
    H_in = a(rcpT0 * u * r * ds(1))               # Danckwerts: total inflow at T0
    H_out = a(rcpT0 * u * th * r * ds(2))
    Q_gen = a((-case.dH) * k * case.C0 * y * r * dx)
    mode = case.wall["mode"]
    if mode == "coolant":
        Q_wall = a(case.wall["U_W_m2K"] * (case.T0 * th - case.wall["T_coolant_K"]) * r * ds(3))
    elif mode == "adiabatic":
        Q_wall = 0.0
    else:
        # consistent wall flux: energy residual (assembled without the Dirichlet rows) tested with q = 1 equals
        # ∮ α ∂θ/∂r r ds at the wall; heat leaving = −ρcp T0 · 2π · that sum
        b = fem.assemble_vector(fem.form(F))
        b.scatter_reverse(la.InsertMode.add)
        _, m1 = W.sub(1).collapse()
        Q_wall = -rcpT0 * two_pi * float(b.array[np.asarray(m1, dtype=np.int64)].sum())
    closure = H_out - H_in + Q_wall - Q_gen
    scale = max(abs(Q_gen), abs(H_out - H_in), abs(Q_wall), 1e-6 * abs(H_in), 1e-30)
    species = F_out - F_in + rate
    return {"F_in_mol_s": F_in, "F_out_mol_s": F_out, "reaction_rate_mol_s": rate,
            "species_closure_rel": species / max(F_in, 1e-30),
            "H_in_W": H_in, "H_out_W": H_out, "enthalpy_rise_W": H_out - H_in, "Q_gen_W": Q_gen, "Q_wall_out_W": Q_wall,
            "energy_closure_W": closure, "energy_closure_rel": closure / scale,
            "X_cup": 1 - F_out / F_in}


def sample(case: Case, dom, w, nzs: int, nrs: int):
    """Field on a (z, r) grid: y and θ; plus cup-mixing X(z), T(z)."""
    zs = np.linspace(0.0, case.L, nzs)
    zs[0], zs[-1] = 1e-9 * case.L, case.L * (1 - 1e-9)
    rs = np.linspace(0.0, case.R * 0.999999, nrs)
    Z, Rr = np.meshgrid(zs, rs, indexing="ij")
    pts = np.zeros((Z.size, 3))
    pts[:, 0], pts[:, 1] = Z.ravel(), Rr.ravel()
    tree = geometry.bb_tree(dom, 2)
    cand = geometry.compute_collisions_points(tree, pts)
    col = geometry.compute_colliding_cells(dom, cand, pts)
    cells = np.array([col.links(i)[0] for i in range(len(pts))], dtype=np.int32)
    y = w.sub(0).collapse().eval(pts, cells).ravel().reshape(Z.shape)
    th = w.sub(1).collapse().eval(pts, cells).ravel().reshape(Z.shape)
    ur = (2 * case.um * (1 - (rs / case.R) ** 2) if case.profile == "laminar" else np.full_like(rs, case.um)) * rs
    den = np.trapezoid(ur, rs)
    X_cup = 1 - np.trapezoid(y * ur, rs, axis=1) / den
    T_cup = case.T0 * np.trapezoid(th * ur, rs, axis=1) / den
    return zs, rs, y, th, X_cup, T_cup


def hot_spot(case: Case, dom, w, W):
    thf = w.sub(1).collapse()
    th = np.asarray(thf.x.array).ravel()
    xy = thf.function_space.tabulate_dof_coordinates()
    i = int(np.argmax(th))
    return {"T_max_K": float(case.T0 * th[i]), "z_m": float(xy[i, 0]), "r_m": float(xy[i, 1]),
            "T_min_K": float(case.T0 * th.min())}


def run_mesh(case: Case, nz: int, nr: int, tag: str, nzs: int, nrs: int, full: bool):
    t0 = time.perf_counter()
    dom, W, w, tags, aux, F, bcs, info = solve(case, nz, nr, tag)
    bal = balances(case, dom, W, w, aux, F, bcs)
    zs, rs, y, th, X_cup, T_cup = sample(case, dom, w, nzs, nrs)
    hs = hot_spot(case, dom, w, W)
    out = {"mesh": {"nz": nz, "nr": nr}, "balances": bal, "hot_spot": hs, "solver": info,
           "X_outlet": bal["X_cup"], "T_outlet_cup_K": float(T_cup[-1]), "seconds": time.perf_counter() - t0}
    if full:
        out["axial"] = {"z_m": zs.tolist(), "X_cup": X_cup.tolist(), "T_cup_K": T_cup.tolist(),
                        "T_centerline_K": (case.T0 * th[:, 0]).tolist(), "T_wall_side_K": (case.T0 * th[:, -1]).tolist()}
        izh = int(np.argmin(np.abs(zs - hs["z_m"])))
        out["radial_hot_spot"] = {"z_m": float(zs[izh]), "r_m": rs.tolist(), "T_K": (case.T0 * th[izh]).tolist(),
                                  "X": (1 - y[izh]).tolist()}
        out["radial_outlet"] = {"r_m": rs.tolist(), "T_K": (case.T0 * th[-1]).tolist(), "X": (1 - y[-1]).tolist()}
        out["grid"] = {"z_m": zs.tolist(), "r_m": rs.tolist(), "T_K": (case.T0 * th).tolist(), "X": (1 - y).tolist()}
    return out


def main() -> None:
    p = json.loads(sys.stdin.read())
    case = Case(p)
    nz, nr = int(p["nz"]), int(p["nr"])
    nzs, nrs = int(p.get("n_stations", 121)), int(p.get("n_radial", 41))
    res = {"coarse": run_mesh(case, nz, nr, "coarse", nzs, nrs, True)}
    if p.get("mesh_study", True):
        res["fine"] = run_mesh(case, 2 * nz, 2 * nr, "fine", nzs, nrs, False)
    res["dolfinx_version"] = __import__("dolfinx").__version__
    sys.stdout.write(json.dumps(res))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
