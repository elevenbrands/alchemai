"""Tubular reactor as a resolved field (bloque M1 of docs/MAPA_MOTORES.md): the first external engine after DWSIM.

The ideal PFR of a process simulator assumes plug flow. A real tubular reactor is a bundle of tubes, often laminar,
where the centre-line fluid leaves under-converted and radial diffusion recovers part of the loss. This module asks
FEniCSx (dolfinx, LGPL-3, installed in its own micromamba environment) for the 2-D axisymmetric
advection-diffusion-reaction field of one tube and returns the mixing-cup conversion for the laminar and plug
profiles, the two analytic bounds (segregated laminar, plug 1 − e^(−kτ)) and the evidence that decides the seal.

FEniCSx runs as a SEPARATE PROCESS (`packages/transport/fenicsx/tubular_reactor.py`, JSON in/out) with a timeout
that kills the whole process group, like `claude -p` in the agent. Nothing from dolfinx is imported here.

Evidence → seal:
  * plug_matches_analytic: |X_plug_fenicsx − (1 − e^(−kτ))| < PLUG_TOL (the method reproduces the exact answer)
  * mesh_independent: |X_fine − X_coarse| < MESH_TOL (the laminar number does not depend on the mesh)
  * within_bounds: X_segregated − 0.01 ≤ X_laminar ≤ X_plug + 1e-3 (physics: radial diffusion helps, never beyond plug)
  * laminar_regime: Re < RE_LAMINAR (the Poiseuille profile is the right one). Needs ρ and μ (or Re) from the caller.
`verified` is true only when all four hold.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

R_GAS = 8.314462618
PLUG_TOL = 1e-3
MESH_TOL = 2e-3
RE_LAMINAR = 2300.0
DEFAULT_TIMEOUT_S = 600
SOLVER = Path(__file__).with_name("fenicsx") / "tubular_reactor.py"


def fenicsx_python() -> str | None:
    """Interpreter of the FEniCSx environment: $ALCHEMAI_FENICSX_PYTHON, else micromamba `fenicsx` under ~/mamba."""
    cands = [os.environ.get("ALCHEMAI_FENICSX_PYTHON"), str(Path.home() / "mamba" / "envs" / "fenicsx" / "bin" / "python")]
    for c in cands:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return None


def fenicsx_available() -> bool:
    return fenicsx_python() is not None


def _kill_group(proc: subprocess.Popen) -> None:
    import signal
    import time
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.communicate(timeout=5)
    except Exception:  # noqa: BLE001
        pass


def run_solver(payload: dict, timeout_s: int = DEFAULT_TIMEOUT_S, python: str | None = None) -> dict:
    """Run the FEniCSx script in its own process group with the payload on stdin; return its JSON."""
    py = python or fenicsx_python()
    if py is None:
        raise RuntimeError("FEniCSx no está instalado: falta el entorno micromamba `fenicsx` (~/mamba/envs/fenicsx) "
                           "o la variable ALCHEMAI_FENICSX_PYTHON. Ver transporte-lab/README.md para instalarlo.")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CONDA", "MAMBA", "PYTHON", "VIRTUAL_ENV"))}
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    proc = subprocess.Popen([py, "-I", str(SOLVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env, start_new_session=True, cwd=str(SOLVER.parent))
    try:
        out, err = proc.communicate(json.dumps(payload), timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        raise RuntimeError(f"FEniCSx no terminó en {timeout_s} s (malla {payload.get('nz')}×{payload.get('nr')}); "
                           "reduce nz/nr o sube timeout_s") from None
    if proc.returncode != 0:
        raise RuntimeError(f"FEniCSx falló (código {proc.returncode}): {err.strip()[-800:]}")
    i, j = out.find("{"), out.rfind("}")
    if i < 0 or j < 0:
        raise RuntimeError(f"FEniCSx no devolvió JSON: {out[-300:]!r} {err.strip()[-300:]!r}")
    return json.loads(out[i:j + 1])


def arrhenius_k(A_1_s: float, E_J_mol: float, T_K: float) -> float:
    return A_1_s * math.exp(-E_J_mol / (R_GAS * T_K))


def bundle_geometry(volumetric_flow_m3_s: float, tube_diameter_m: float, n_tubes: int, tube_length_m: float | None = None,
                    volume_m3: float | None = None) -> dict:
    """Per-tube mean velocity, length and residence time of a bundle. Give the tube length OR the total volume."""
    if (tube_length_m is None) == (volume_m3 is None):
        raise ValueError("give exactly one of tube_length_m or volume_m3")
    if n_tubes < 1 or tube_diameter_m <= 0 or volumetric_flow_m3_s <= 0:
        raise ValueError("n_tubes ≥ 1, tube_diameter_m > 0 and volumetric_flow_m3_s > 0 are required")
    A = math.pi * tube_diameter_m ** 2 / 4.0
    L = tube_length_m if tube_length_m is not None else volume_m3 / (n_tubes * A)
    u = volumetric_flow_m3_s / (n_tubes * A)
    return {"n_tubes": n_tubes, "tube_diameter_m": tube_diameter_m, "tube_length_m": L, "volume_m3": n_tubes * A * L,
            "u_mean_m_s": u, "tau_s": L / u}


def evidence_from(result: dict, Re: float | None) -> dict:
    plug = abs(result["X_plug_fenicsx"] - result["X_plug_analytic"])
    mesh = abs(result["X_laminar_fenicsx_fine_mesh"] - result["X_laminar_fenicsx"])
    lam = result["X_laminar_fenicsx"]
    bounds = (result["X_laminar_segregated_analytic"] - 0.01) <= lam <= (result["X_plug_analytic"] + 1e-3)
    ev = {
        "plug_matches_analytic": plug < PLUG_TOL, "plug_error": plug,
        "mesh_independent": mesh < MESH_TOL, "mesh_delta": mesh,
        "within_bounds": bounds,
        "laminar_regime": (Re < RE_LAMINAR) if Re is not None else None, "Re": Re,
    }
    reasons = []
    if not ev["plug_matches_analytic"]:
        reasons.append(f"el caso pistón no reproduce 1−e^(−kτ) (error {plug:.2e})")
    if not ev["mesh_independent"]:
        reasons.append(f"la conversión laminar cambia con la malla ({mesh:.2e}); sube nz/nr")
    if not bounds:
        reasons.append("la conversión laminar queda fuera de [segregado, pistón]")
    if Re is None:
        reasons.append("régimen no comprobado: da density_kg_m3 y viscosity_Pa_s (o reynolds) para confirmar laminar")
    elif Re >= RE_LAMINAR:
        reasons.append(f"Re = {Re:.0f} ≥ {RE_LAMINAR:.0f}: el perfil de Poiseuille no aplica; el flujo es turbulento o de transición")
    ev["verified"] = not reasons
    ev["reasons"] = reasons
    return ev


def tubular_reactor_field(volumetric_flow_m3_s: float, tube_diameter_m: float, n_tubes: int, diffusivity_m2_s: float,
                          tube_length_m: float | None = None, volume_m3: float | None = None,
                          k_1_s: float | None = None, A_1_s: float | None = None, E_J_mol: float | None = None, T_K: float | None = None,
                          density_kg_m3: float | None = None, viscosity_Pa_s: float | None = None, reynolds: float | None = None,
                          nz: int = 300, nr: int = 30, figure_path: str | None = None, timeout_s: int = DEFAULT_TIMEOUT_S) -> dict:
    """Conversion of a first-order, isothermal tubular reactor built as a bundle of n_tubes laminar tubes, solved as a
    2-D field with FEniCSx (separate process), against the ideal PFR (plug flow). See module docstring for the seal."""
    if k_1_s is None:
        if None in (A_1_s, E_J_mol, T_K):
            raise ValueError("give k_1_s, or A_1_s + E_J_mol + T_K (Arrhenius, E in J/mol)")
        k_1_s = arrhenius_k(A_1_s, E_J_mol, T_K)
    if k_1_s <= 0 or diffusivity_m2_s <= 0:
        raise ValueError("k_1_s and diffusivity_m2_s must be positive")
    geo = bundle_geometry(volumetric_flow_m3_s, tube_diameter_m, n_tubes, tube_length_m, volume_m3)
    Re = reynolds
    if Re is None and density_kg_m3 is not None and viscosity_Pa_s is not None:
        Re = density_kg_m3 * geo["u_mean_m_s"] * tube_diameter_m / viscosity_Pa_s
    payload = {"k_1_s": k_1_s, "tube_radius_m": tube_diameter_m / 2.0, "tube_length_m": geo["tube_length_m"],
               "u_mean_m_s": geo["u_mean_m_s"], "diffusivity_m2_s": diffusivity_m2_s, "nz": int(nz), "nr": int(nr)}
    if figure_path:
        payload["figure_path"] = str(Path(figure_path).expanduser().resolve())
        payload["figure_title"] = (f"Conversión local en un tubo laminar (FEniCSx) · {n_tubes} tubos × {tube_diameter_m*100:.1f} cm × "
                                   f"{geo['tube_length_m']:.2f} m" + (f" · Re {Re:.0f}" if Re is not None else ""))
    res = run_solver(payload, timeout_s=timeout_s)
    ev = evidence_from(res, Re)
    X_plug, X_lam = res["X_plug_analytic"], res["X_laminar_fenicsx"]
    return {
        "k_1_s": k_1_s, "geometry": geo, "Re": Re,
        "X_plug_flow": X_plug, "X_plug_flow_fenicsx": res["X_plug_fenicsx"],
        "X_laminar": X_lam, "X_laminar_fine_mesh": res["X_laminar_fenicsx_fine_mesh"],
        "X_laminar_segregated": res["X_laminar_segregated_analytic"],
        "laminar_penalty_points": 100.0 * (X_plug - X_lam),
        "radial_diffusion_number": res["radial_diffusion_number"], "Pe_axial": res["Pe_axial"], "k_tau": res["k_tau"],
        "outlet_profile": res["outlet_profile"], "mesh": res["mesh"], "timing_s": res["timing_s"],
        "figure_path": res.get("figure_path"), "engine": f"FEniCSx dolfinx {res.get('dolfinx_version', '?')}",
        "evidence": ev,
        "notes": ("Primer orden, isotérmico, un tubo del haz, perfil de Poiseuille, SUPG, P1, LU. D·τ/R² ≈ 0.1-1 es donde la "
                  "difusión radial importa; ≪ 0.1 tiende al segregado, ≫ 1 al pistón. La difusividad es dato del usuario "
                  "(ver diffusivity_estimate)."),
    }


def dependencies_report() -> dict:
    py = fenicsx_python()
    return {"fenicsx_python": py, "available": py is not None, "solver": str(SOLVER), "solver_exists": SOLVER.is_file(),
            "openfoam": shutil.which("openfoam")}
