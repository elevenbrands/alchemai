"""Residence time of a flow-through vessel by CFD (bloque M6a): is this tank really a CSTR?

OpenFOAM (GPL-3, separate process through the `openfoam -c` launcher, like FEniCSx in `reactor_field`) solves the
steady flow of a box-shaped vessel with one inlet and one outlet opening (simpleFoam, laminar or k-ε by the inlet
Reynolds number) and, on the converged flow, the steady equation of the local mean age of the fluid
(Spalding 1958; OpenFOAM's `age` function object): ∇·(φ a) − ∇·(ν_eff ∇a) = 1, a = 0 at the inlet.

From the age field alone, for a closed vessel at steady state (Danckwerts; Liu & Tilton 2010):
  * the flow-weighted age at the outlet equals the space time τ = V/Q (a theorem: it is the evidence check),
  * the volume-averaged age ⟨a⟩ = E[t²]/(2τ), so the RTD variance is σ² = 2τ⟨a⟩ − τ²,
  * CSTR ⇒ ⟨a⟩ = τ, σ²/τ² = 1; plug flow ⇒ ⟨a⟩ = τ/2, σ² = 0; dead zones or bypass ⇒ σ²/τ² > 1,
  * equivalent tanks in series N = τ²/σ² (gamma RTD) and the age histogram shows where the fluid stagnates.

Evidence → seal: the flow converged, the outlet age reproduces τ within THEOREM_TOL, σ² ≥ 0 and, with
`mesh_study`, N and ⟨a⟩ change less than MESH_TOL from the coarse to the fine mesh. The impeller of an agitated
tank is not modelled here (bloque M6b, MRF): this tool answers inlet/outlet placement, dead volume and bypass.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from packages.transport.reactor_field import _kill_group

THEOREM_TOL = 0.03
MESH_TOL = 0.10
RE_TURBULENT = 2000.0
DEFAULT_TIMEOUT_S = 1200
FACES = {"x-": (0, 0), "x+": (0, 1), "y-": (1, 0), "y+": (1, 1), "z-": (2, 0), "z+": (2, 1)}


def openfoam_available() -> bool:
    return shutil.which("openfoam") is not None


def _run(case: Path, cmd: str, timeout_s: int) -> None:
    proc = subprocess.Popen(["openfoam", "-c", f"cd {case} && {cmd}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        raise RuntimeError(f"OpenFOAM no terminó en {timeout_s} s: reduce cells_per_max_dimension o sube timeout_s") from None
    if proc.returncode != 0:
        logs = sorted(case.glob("log.*"), key=lambda p: p.stat().st_mtime)
        tail = logs[-1].read_text()[-1200:] if logs else (out + err)[-1200:]
        raise RuntimeError(f"OpenFOAM falló ({cmd.split()[0]}…): {tail}")


def _opening_box(op: dict, dims: tuple[float, float, float], eps: float) -> tuple[list[float], list[float]]:
    face = op["face"]
    if face not in FACES:
        raise ValueError(f"opening face must be one of {sorted(FACES)}")
    axis, side = FACES[face]
    others = [i for i in range(3) if i != axis]
    (c1, c2), (s1, s2) = op["center_m"], op["size_m"]
    lo, hi = [0.0] * 3, [0.0] * 3
    plane = dims[axis] * side
    lo[axis], hi[axis] = plane - eps, plane + eps
    for i, c, s in ((others[0], c1, s1), (others[1], c2, s2)):
        if c - s / 2 < -1e-12 or c + s / 2 > dims[i] + 1e-12:
            raise ValueError(f"opening on {face} falls outside the wall")
        lo[i], hi[i] = c - s / 2, c + s / 2
    return lo, hi


def _write_case(case: Path, dims, n_cells, Q, nu, inlet, outlet, turbulent, walls, k_in, eps_in) -> None:
    L, W, H = dims
    d = max(dims) / n_cells
    nx, ny, nz = (max(2, round(x / d)) for x in dims)
    for sub in ("0", "constant", "system"):
        (case / sub).mkdir(parents=True, exist_ok=True)
    hdr = "FoamFile { version 2.0; format ascii; class %s; object %s; }\n"
    w = lambda rel, cls, obj, body: (case / rel).write_text(hdr % (cls, obj) + body)   # noqa: E731
    w("system/blockMeshDict", "dictionary", "blockMeshDict", f"""scale 1;
vertices ((0 0 0) ({L} 0 0) ({L} {W} 0) (0 {W} 0) (0 0 {H}) ({L} 0 {H}) ({L} {W} {H}) (0 {W} {H}));
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1));
boundary ( walls {{ type wall; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }} );
""")
    eps = 0.25 * d
    acts = []
    for name, op in (("inlet", inlet), ("outlet", outlet)):
        lo, hi = _opening_box(op, dims, eps)
        acts.append(f"{{ name {name}Faces; type faceSet; action new; source patchToFace; patch walls; }}\n"
                    f"{{ name {name}Faces; type faceSet; action subset; source boxToFace; "
                    f"box ({lo[0]} {lo[1]} {lo[2]}) ({hi[0]} {hi[1]} {hi[2]}); }}")
    w("system/topoSetDict", "dictionary", "topoSetDict", "actions (\n" + "\n".join(acts) + "\n);\n")
    w("system/createPatchDict", "dictionary", "createPatchDict", """pointSync false;
patches ( { name inlet; patchInfo { type patch; } constructFrom set; set inletFaces; }
          { name outlet; patchInfo { type patch; } constructFrom set; set outletFaces; } );
""")
    wall_U = "slip" if walls == "slip" else "noSlip"
    w("0/U", "volVectorField", "U", f"""dimensions [0 1 -1 0 0 0 0]; internalField uniform (0 0 0);
boundaryField {{ inlet {{ type flowRateInletVelocity; volumetricFlowRate {Q}; value uniform (0 0 0); }}
  outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0); }}
  walls {{ type {wall_U}; }} }}
""")
    w("0/p", "volScalarField", "p", """dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;
boundaryField { inlet { type zeroGradient; } outlet { type fixedValue; value uniform 0; } walls { type zeroGradient; } }
""")
    w("constant/transportProperties", "dictionary", "transportProperties", f"transportModel Newtonian; nu {nu};\n")
    if turbulent:
        w("constant/turbulenceProperties", "dictionary", "turbulenceProperties",
          "simulationType RAS; RAS { RASModel kEpsilon; turbulence on; printCoeffs off; }\n")
        wk = "kqRWallFunction" if walls != "slip" else "zeroGradient"
        we = "epsilonWallFunction" if walls != "slip" else "zeroGradient"
        wn = "nutkWallFunction" if walls != "slip" else "calculated"
        for f, val, wall_t, dim in (("k", k_in, wk, "[0 2 -2 0 0 0 0]"), ("epsilon", eps_in, we, "[0 2 -3 0 0 0 0]")):
            w(f"0/{f}", "volScalarField", f, f"""dimensions {dim}; internalField uniform {val};
boundaryField {{ inlet {{ type fixedValue; value uniform {val}; }}
  outlet {{ type inletOutlet; inletValue uniform {val}; value uniform {val}; }}
  walls {{ type {wall_t}; value uniform {val}; }} }}
""")
        w("0/nut", "volScalarField", "nut", f"""dimensions [0 2 -1 0 0 0 0]; internalField uniform 0;
boundaryField {{ inlet {{ type calculated; value uniform 0; }} outlet {{ type calculated; value uniform 0; }}
  walls {{ type {wn}; value uniform 0; }} }}
""")
    else:
        w("constant/turbulenceProperties", "dictionary", "turbulenceProperties", "simulationType laminar;\n")
    w("system/controlDict", "dictionary", "controlDict", """application simpleFoam; startFrom startTime; startTime 0;
stopAt endTime; endTime 3000; deltaT 1; writeControl timeStep; writeInterval 3000; purgeWrite 0;
writeFormat ascii; writePrecision 10; timePrecision 6; runTimeModifiable false;
functions {
  age1 { type age; libs (fieldFunctionObjects); diffusion true; tolerance 1e-8; nCorr 50;
         executeControl writeTime; writeControl writeTime; }
  vol1 { type writeCellVolumes; libs (fieldFunctionObjects); executeControl writeTime; writeControl writeTime; }
  outAge { type surfaceFieldValue; libs (fieldFunctionObjects); regionType patch; name outlet;
           operation weightedAverage; weightField phi; fields (age); writeFields false;
           executeControl writeTime; writeControl writeTime; log true; }
  outFlow { type surfaceFieldValue; libs (fieldFunctionObjects); regionType patch; name outlet;
            operation sum; fields (phi); writeFields false; executeControl writeTime; writeControl writeTime; log true; }
}
""")
    w("system/fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); div(phi,k) bounded Gauss upwind;
  div(phi,epsilon) bounded Gauss upwind; div(phi,age) bounded Gauss upwind;
  div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; }
snGradSchemes { default corrected; } wallDist { method meshWave; }
""")
    w("system/fvSolution", "dictionary", "fvSolution", """solvers {
  p { solver GAMG; smoother GaussSeidel; tolerance 1e-8; relTol 0.05; }
  "(U|k|epsilon)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0.1; maxIter 50; }
  age { solver PBiCGStab; preconditioner DILU; tolerance 1e-10; relTol 0; } }
SIMPLE { nNonOrthogonalCorrectors 0; consistent yes;
  residualControl { p 1e-4; U 1e-5; "(k|epsilon)" 1e-5; } }
relaxationFactors { equations { U 0.9; "(k|epsilon)" 0.7; } fields { p 1; } }
""")


def _read_internal(path: Path) -> list[float]:
    txt = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(", txt)
    if m:
        n = int(m.group(1))
        vals = txt[m.end():txt.index(")", m.end())].split()
        return [float(v) for v in vals[:n]]
    m = re.search(r"internalField\s+uniform\s+([-+0-9.eE]+)", txt)
    if m:
        return [float(m.group(1))]
    raise RuntimeError(f"no pude leer el campo {path.name}")


def _last_number(dat: Path) -> float:
    rows = [ln.split() for ln in dat.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return float(rows[-1][-1])


def _solve(dims, n_cells, Q, nu, inlet, outlet, walls, timeout_s, keep: Path | None) -> dict:
    a_in = inlet["size_m"][0] * inlet["size_m"][1]
    Dh_in = 2 * inlet["size_m"][0] * inlet["size_m"][1] / (inlet["size_m"][0] + inlet["size_m"][1])
    U_in = Q / a_in
    Re_in = U_in * Dh_in / nu
    turbulent = Re_in > RE_TURBULENT
    k_in = 1.5 * (0.05 * U_in) ** 2                          # 5 % intensity at the inlet (assumption, reported)
    eps_in = 0.09 ** 0.75 * k_in ** 1.5 / (0.07 * Dh_in)    # mixing length 0.07·Dh (assumption, reported)
    d = max(dims) / n_cells
    for name, op in (("inlet", inlet), ("outlet", outlet)):
        if min(op["size_m"]) < 2 * d - 1e-12:
            raise ValueError(f"{name} opening {min(op['size_m']):.3g} m is under 2 cells ({d:.3g} m): raise "
                             "cells_per_max_dimension or enlarge the opening")
    case = Path(tempfile.mkdtemp(prefix=f"alchemai_vessel_{n_cells}_"))
    if " " in str(case):
        raise RuntimeError("OpenFOAM no acepta rutas con espacios: TMPDIR tiene un espacio")
    try:
        _write_case(case, dims, n_cells, Q, nu, inlet, outlet, turbulent, walls, k_in, eps_in)
        _run(case, "blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 && "
                   "createPatch -overwrite > log.createPatch 2>&1 && simpleFoam > log.simpleFoam 2>&1", timeout_s)
        log = (case / "log.simpleFoam").read_text()
        m = re.search(r"SIMPLE solution converged in (\d+) iterations", log)
        converged, iters = bool(m), int(m.group(1)) if m else None
        times = sorted((p for p in case.iterdir() if re.fullmatch(r"\d+(\.\d+)?", p.name) and p.name != "0"),
                       key=lambda p: float(p.name))
        last = times[-1]
        age = _read_internal(last / "age")
        V = _read_internal(last / "V")
        if len(age) != len(V):
            raise RuntimeError("age y V con tamaños distintos")
        pp = case / "postProcessing"
        t_out = _last_number(next(pp.glob("outAge/*/surfaceFieldValue.dat")))
        q_out = _last_number(next(pp.glob("outFlow/*/surfaceFieldValue.dat")))
        cont = re.findall(r"continuity errors : sum local = ([-+0-9.eE]+), global = ([-+0-9.eE]+)", log)
        bnd = (case / "constant" / "polyMesh" / "boundary").read_text()
        nf = {pn: int(mm.group(1)) for pn in ("inlet", "outlet")
              for mm in [re.search(pn + r"\s*\{[^}]*?nFaces\s+(\d+)", bnd)] if mm}
        if not nf.get("inlet") or not nf.get("outlet"):
            raise RuntimeError(f"no se crearon las aberturas (caras inlet/outlet: {nf}); revisa face/center/size")
        Vt = sum(V)
        a_mean = sum(a * v for a, v in zip(age, V)) / Vt
        res = {"n_cells": len(V), "cell_size_m": d, "converged": converged, "iterations": iters or 3000,
               "Re_inlet": Re_in, "turbulence": "k-epsilon" if turbulent else "laminar",
               "outlet_mean_age_s": t_out, "outlet_flow_m3_s": abs(q_out), "volume_mean_age_s": a_mean,
               "mesh_volume_m3": Vt, "continuity_sum_local": float(cont[-1][0]) if cont else None,
               "_age": age, "_V": V, "opening_faces": nf}
        if turbulent:
            res["turbulence_inlet"] = {"intensity": 0.05, "mixing_length_m": 0.07 * Dh_in}
        return res
    finally:
        if keep is not None:
            keep.mkdir(parents=True, exist_ok=True)
            dst = keep / case.name
            shutil.copytree(case, dst, dirs_exist_ok=True)
        shutil.rmtree(case, ignore_errors=True)


def vessel_rtd(length_m: float, width_m: float, height_m: float, volumetric_flow_m3_s: float,
               kinematic_viscosity_m2_s: float, inlet: dict, outlet: dict, cells_per_max_dimension: int = 30,
               mesh_study: bool = True, first_order_k_1_s: float | None = None, walls: str = "noSlip",
               keep_case_dir: str | None = None, timeout_s: int = DEFAULT_TIMEOUT_S) -> dict:
    if not openfoam_available():
        raise RuntimeError("OpenFOAM no está instalado (falta el lanzador `openfoam`; ver transporte-lab/README.md)")
    if min(length_m, width_m, height_m, volumetric_flow_m3_s, kinematic_viscosity_m2_s) <= 0:
        raise ValueError("dimensions, flow and viscosity must be positive")
    if walls not in ("noSlip", "slip"):
        raise ValueError("walls must be 'noSlip' (real tank) or 'slip' (idealised plug-flow check)")
    dims = (float(length_m), float(width_m), float(height_m))
    Q, nu = float(volumetric_flow_m3_s), float(kinematic_viscosity_m2_s)
    V = dims[0] * dims[1] * dims[2]
    tau = V / Q
    keep = Path(keep_case_dir).expanduser() if keep_case_dir else None
    meshes = [int(cells_per_max_dimension)]
    if mesh_study:
        meshes.append(round(1.5 * cells_per_max_dimension))
    runs = [_solve(dims, n, Q, nu, inlet, outlet, walls, timeout_s, keep) for n in meshes]
    fine = runs[-1]

    def stats(r):
        s2 = 2 * tau * r["volume_mean_age_s"] - tau ** 2
        return s2, (tau ** 2 / s2 if s2 > 0 else math.inf)

    s2, N = stats(fine)
    age, Vc = fine.pop("_age"), fine.pop("_V")
    for r in runs[:-1]:
        r.pop("_age"), r.pop("_V")
    bins = [0, 0.25, 0.5, 1, 2, 3, 5, math.inf]
    hist = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        vol = sum(v for a, v in zip(age, Vc) if lo * tau <= a < hi * tau)
        hist.append({"age_over_tau": f"{lo}-{hi if hi != math.inf else '∞'}", "volume_fraction": vol / sum(Vc)})
    stagnant = sum(v for a, v in zip(age, Vc) if a >= 3 * tau) / sum(Vc)
    out = {"space_time_tau_s": tau, "volume_m3": V, "outlet_mean_age_s": fine["outlet_mean_age_s"],
           "volume_mean_age_s": fine["volume_mean_age_s"], "mean_age_over_tau": fine["volume_mean_age_s"] / tau,
           "rtd_variance_s2": s2, "normalized_variance": s2 / tau ** 2,
           "tanks_in_series_N": N if math.isfinite(N) else None, "stagnant_volume_fraction_age_over_3tau": stagnant,
           "age_histogram": hist, "Re_inlet": fine["Re_inlet"], "turbulence": fine["turbulence"],
           "runs": runs, "assumptions": [], "warnings": []}
    out["assumptions"].append("difusión del trazador con ν_efectivo (Sc laminar y turbulento = 1, como el "
                              "function object age de OpenFOAM); en turbulento domina ν_t")
    if fine["turbulence"] == "k-epsilon":
        out["assumptions"].append("entrada turbulenta: intensidad 5 % y longitud de mezcla 0.07·Dh (supuesto "
                                  "estándar, no medido)")
    nv = s2 / tau ** 2
    if nv > 1.05:
        out["diagnosis"] = ("σ²/τ² > 1: peor que un CSTR, hay volumen muerto o cortocircuito entre entrada y salida; "
                            "mover entrada/salida o agitar")
    elif nv > 0.8:
        out["diagnosis"] = "σ²/τ² ≈ 1: se comporta como CSTR"
    else:
        out["diagnosis"] = f"σ²/τ² = {nv:.2f} < 1: entre CSTR y flujo pistón (≈ {N:.1f} tanques en serie)"
    if first_order_k_1_s is not None:
        k = float(first_order_k_1_s)
        Da = k * tau
        out["first_order_conversion"] = {
            "CSTR_ideal": Da / (1 + Da), "PFR_ideal": 1 - math.exp(-Da),
            "gamma_RTD_N": (1 - (1 + Da / N) ** (-N)) if math.isfinite(N) else 1 - math.exp(-Da),
            "note": "conversión con la RTD gamma de N tanques en serie (segregado, primer orden: exacto para esa RTD)"}
    reasons = []
    if not all(r["converged"] for r in runs):
        reasons.append("simpleFoam no alcanzó los residuales (p 1e-4, U 1e-5) en 3000 iteraciones")
    theorem = abs(fine["outlet_mean_age_s"] - tau) / tau
    if theorem > THEOREM_TOL:
        reasons.append(f"edad media a la salida {fine['outlet_mean_age_s']:.4g} s vs τ {tau:.4g} s "
                       f"({100 * theorem:.1f} % > {100 * THEOREM_TOL:.0f} %)")
    if s2 < -1e-3 * tau ** 2:
        reasons.append("σ² < 0: campo de edad inconsistente")
    flow_err = abs(fine["outlet_flow_m3_s"] - Q) / Q
    if flow_err > 1e-3:
        reasons.append(f"caudal de salida {fine['outlet_flow_m3_s']:.4g} vs {Q:.4g} m³/s")
    mesh = None
    if mesh_study:
        c = runs[0]
        s2c = 2 * tau * c["volume_mean_age_s"] - tau ** 2
        d_age = abs(c["volume_mean_age_s"] - fine["volume_mean_age_s"]) / fine["volume_mean_age_s"]
        d_nv = abs(s2c - s2) / tau ** 2                     # change of σ²/τ², absolute (σ² → 0 in plug flow)
        nv_lim = max(MESH_TOL * abs(s2) / tau ** 2, 0.05)
        mesh = {"mean_age_change": d_age, "normalized_variance_change": d_nv, "normalized_variance_limit": nv_lim}
        if d_age > MESH_TOL / 2 or d_nv > nv_lim:
            reasons.append(f"dependencia de malla: ⟨a⟩ cambia {100 * d_age:.1f} % (límite {100 * MESH_TOL / 2:.0f} %), "
                           f"σ²/τ² cambia {d_nv:.3f} (límite {nv_lim:.3f})")
    else:
        out["warnings"].append("sin estudio de malla: el resultado no se puede sellar")
        reasons.append("sin estudio de malla")
    out["evidence"] = {"outlet_age_vs_tau_rel": theorem, "outlet_flow_rel_error": flow_err, "mesh": mesh,
                       "converged": all(r["converged"] for r in runs), "verified": not reasons, "reasons": reasons}
    out["notes"] = ("Edad media local (Spalding): ⟨a⟩/τ = 1 en un CSTR, 0.5 en flujo pistón; σ² = 2τ⟨a⟩ − τ². "
                    "Tanque prismático con una entrada y una salida; sin impulsor (tanque agitado con MRF = bloque "
                    "M6b). La difusión numérica del esquema upwind acerca el resultado al CSTR: el estudio de malla "
                    "la acota.")
    return out
