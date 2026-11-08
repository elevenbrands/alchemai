"""Tubular reactor solved as a 2-D axisymmetric field (FEniCSx): steady advection-diffusion-reaction, first order,
isothermal, in ONE tube of the bundle.

    u(r) ∂c/∂z = D [ (1/r) ∂/∂r (r ∂c/∂r) + ∂²c/∂z² ] − k c,   c(0, r) = C0,  zero flux at the wall and on the axis.

Velocity profile: Poiseuille u(r) = 2 ū (1 − r²/R²) (laminar) or u = ū (plug flow, what an ideal PFR assumes).
SUPG stabilisation with τ = h/(2u) (written as (h/2)·∂v/∂z so it stays finite where u → 0). P1 elements, direct LU.
Conversion is the mixing-cup value at the outlet: X = 1 − ∫ u c r dr / ∫ u C0 r dr.

Runs ONLY inside the FEniCSx environment (dolfinx ≥ 0.11, LGPL-3), as a separate process launched by
`packages.transport.reactor_field`; it has no dependency on the rest of AlchemAI. Protocol: one JSON object on stdin,
one JSON object on stdout. Everything else (progress) goes to stderr.

Input keys: k_1_s, tube_radius_m, tube_length_m, u_mean_m_s, diffusivity_m2_s, nz, nr, C0_mol_m3 (default 1.0),
            n_profile_points (default 60), figure_path (optional), figure_title (optional).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI


def solve_field(profile: str, *, k: float, C0: float, R: float, L: float, u_mean: float, D: float, nz: int, nr: int):
    dom = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, R]], [nz, nr], cell_type=mesh.CellType.triangle)
    Vh = fem.functionspace(dom, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(dom)
    z, r = x[0], x[1]
    u = 2 * u_mean * (1 - (r / R) ** 2) if profile == "laminar" else fem.Constant(dom, u_mean)
    c, v = ufl.TrialFunction(Vh), ufl.TestFunction(Vh)
    h = ufl.CellDiameter(dom)
    a = (u * c.dx(0) * v + D * ufl.inner(ufl.grad(c), ufl.grad(v)) + k * c * v) * r * ufl.dx \
        + (h / 2) * v.dx(0) * (u * c.dx(0) + k * c) * r * ufl.dx
    Lf = fem.Constant(dom, 0.0) * v * r * ufl.dx
    inlet = mesh.locate_entities_boundary(dom, 1, lambda X: np.isclose(X[0], 0.0))
    bc = fem.dirichletbc(fem.Constant(dom, C0), fem.locate_dofs_topological(Vh, 1, inlet), Vh)
    ch = LinearProblem(a, Lf, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                       petsc_options_prefix=f"adr_{profile}_{nz}_").solve()
    outlet = mesh.locate_entities_boundary(dom, 1, lambda X: np.isclose(X[0], L))
    tags = mesh.meshtags(dom, 1, outlet, np.full(len(outlet), 2, dtype=np.int32))
    ds = ufl.Measure("ds", domain=dom, subdomain_data=tags)
    num = fem.assemble_scalar(fem.form(u * ch * r * ds(2)))
    den = fem.assemble_scalar(fem.form(u * C0 * r * ds(2)))
    return dom, ch, float(1 - num / den)


def outlet_profile(dom, ch, *, C0: float, R: float, L: float, n: int):
    pts = np.zeros((n, 3))
    pts[:, 0] = L - 1e-9
    pts[:, 1] = np.linspace(0.0, R * 0.999, n)
    tree = geometry.bb_tree(dom, 2)
    cand = geometry.compute_collisions_points(tree, pts)
    col = geometry.compute_colliding_cells(dom, cand, pts)
    cells = [col.links(i)[0] for i in range(n)]
    c_out = ch.eval(pts, cells).ravel()
    return pts[:, 1].tolist(), (1 - c_out / C0).tolist()


def segregated_laminar(k: float, L: float, u_mean: float, R: float) -> float:
    """No radial diffusion: every annulus is its own PFR with residence time L/u(r)."""
    rr = np.linspace(0.0, R, 20001)[:-1]
    u = 2 * u_mean * (1 - (rr / R) ** 2)
    return float(1 - np.trapezoid(u * np.exp(-k * L / u) * rr, rr) / np.trapezoid(u * rr, rr))


def save_figure(path: str, title: str, dom, ch, *, C0, R, L, X_lam, X_seg, X_plug, r_out, X_out, k, u_mean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    xy = dom.geometry.x
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1] * 1000, dom.geometry.dofmaps[0])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), dpi=150, gridspec_kw={"height_ratios": [1, 1.1]})
    pc = ax1.tripcolor(tri, 1 - ch.x.array / C0, shading="gouraud", cmap="viridis", vmin=0, vmax=1)
    ax1.set_xlabel("z (m)")
    ax1.set_ylabel("r (mm)")
    ax1.set_title(title, fontsize=10)
    fig.colorbar(pc, ax=ax1, label="X local", pad=0.01)
    r_mm = np.asarray(r_out) * 1000
    rr = np.asarray(r_out)
    ax2.plot(r_mm, X_out, lw=2.2, color="#1C6E93", label=f"Laminar con difusión radial (FEniCSx) · X mezcla de copa = {X_lam:.3f}")
    ax2.plot(r_mm, 1 - np.exp(-k * L / (2 * u_mean * (1 - (rr / R) ** 2))), lw=1.3, ls="--", color="#1C6E93",
             label=f"Laminar segregado (analítico) · X = {X_seg:.3f}")
    ax2.axhline(X_plug, color="#C0392B", lw=2, label=f"Flujo pistón (PFR ideal) · X = {X_plug:.3f}")
    ax2.set_xlabel("r a la salida (mm)")
    ax2.set_ylabel("X(r) a la salida")
    ax2.set_ylim(0, 1.02)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    p = json.loads(sys.stdin.read())
    k, R, L, u_mean, D = p["k_1_s"], p["tube_radius_m"], p["tube_length_m"], p["u_mean_m_s"], p["diffusivity_m2_s"]
    nz, nr, C0 = int(p["nz"]), int(p["nr"]), float(p.get("C0_mol_m3", 1.0))
    kw = dict(k=k, C0=C0, R=R, L=L, u_mean=u_mean, D=D)
    tau = L / u_mean
    t0 = time.perf_counter()
    _, _, X_plug = solve_field("plug", nz=nz, nr=nr, **kw)
    t1 = time.perf_counter()
    dom, ch, X_lam = solve_field("laminar", nz=nz, nr=nr, **kw)
    t2 = time.perf_counter()
    _, _, X_lam_fine = solve_field("laminar", nz=2 * nz, nr=2 * nr, **kw)
    t3 = time.perf_counter()
    r_out, X_out = outlet_profile(dom, ch, C0=C0, R=R, L=L, n=int(p.get("n_profile_points", 60)))
    X_plug_analytic = float(1 - np.exp(-k * tau))
    X_seg = segregated_laminar(k, L, u_mean, R)
    out = {
        "tau_s": tau, "k_tau": k * tau,
        "Pe_axial": u_mean * L / D, "radial_diffusion_number": D * tau / R ** 2,
        "mesh": {"nz": nz, "nr": nr, "fine_nz": 2 * nz, "fine_nr": 2 * nr},
        "X_plug_analytic": X_plug_analytic, "X_plug_fenicsx": X_plug,
        "X_laminar_fenicsx": X_lam, "X_laminar_fenicsx_fine_mesh": X_lam_fine,
        "X_laminar_segregated_analytic": X_seg,
        "outlet_profile": {"r_m": r_out, "X": X_out},
        "timing_s": {"plug": t1 - t0, "laminar": t2 - t1, "laminar_fine": t3 - t2},
        "dolfinx_version": __import__("dolfinx").__version__,
    }
    if p.get("figure_path"):
        save_figure(p["figure_path"], p.get("figure_title", "Reactor tubular laminar (FEniCSx)"), dom, ch, C0=C0, R=R, L=L,
                    X_lam=X_lam, X_seg=X_seg, X_plug=X_plug_analytic, r_out=r_out, X_out=X_out, k=k, u_mean=u_mean)
        out["figure_path"] = p["figure_path"]
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
