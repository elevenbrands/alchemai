"""Stirred tank with a Rushton turbine by CFD (bloque M6b of docs/MAPA_MOTORES.md): power number from first principles.

OpenFOAM (GPL-3, separate process through the `openfoam -c` launcher, like M6a in `vessel_cfd`) solves the steady
turbulent flow of a baffled cylindrical tank stirred by a 6-blade Rushton disc turbine. The impeller region rotates
with a multiple reference frame (MRF): a cylindrical cell zone around the turbine is solved in the rotating frame,
the rest in the lab frame (simpleFoam, k-ε with wall functions). The geometry is generated here as STL and meshed
with snappyHexMesh; the case runs in a temporary directory without spaces (mpirun rejects them) on several cores.

What it returns:
  * Np = 2πN·|M_z| / (ρN³D⁵) from the torque the fluid exerts on impeller + shaft (forces function object),
  * the shaft power P = Np ρ N³ D⁵, the power per volume and per mass, Re, Fr and the tip speed,
  * the mixing time θ95 from Grenville's correlation using the CFD Np. It is a CORRELATION, labelled as such
    (Grenville 1992; Handbook of Industrial Mixing 2004, cap. 9): Nθ95 = 5.20 Np^(-1/3) (T/D)², H = T, turbulent
    when Re·Np^(1/3) > 6370. Outside H/T 0.95-1.05 it is not given.

Evidence → seal:
  * torque_balance: at steady state the torque the impeller puts in leaves through the tank wall and baffles
    (angular momentum is conserved): |M_rotor + M_stator| / |M_rotor| < TORQUE_TOL over the last WINDOW
    iterations (steady MRF leaves a ~2 % bias at the rotating-zone interface; a missing wall or baffle is ≫ 3 %),
  * steady: Np over the last WINDOW iterations oscillates less than OSC_TOL of its mean (steady MRF never settles
    exactly; the reported Np is that window's mean),
  * mesh_independent (with mesh_study): three meshes refined by ~1.3 converge monotonically and the grid
    convergence index of the fine mesh (Celik et al. 2008, ASME V&V procedure; Roache) is below GCI_TOL. The
    observed order is capped at the formal order 2 for the GCI and the Richardson-extrapolated Np (conservative),
  * mesh_ok: checkMesh passes and the impeller, baffles and rotating zone exist in the mesh,
  * turbulent: Re > RE_TURBULENT, where k-ε applies and Np of a baffled Rushton is constant,
  * plausible: NP_BAND (Rushton et al. 1950; Bates et al. 1963 report Np ≈ 4-6 for baffled disc turbines). A
    sanity band, not a proof: it catches a broken case, not a 10 % error.
`verified` is true only when all hold. k-ε + MRF typically lands within ~10 % of measured Np and underpredicts the
turbulent dissipation ε (reported as a diagnostic only).
"""
from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from packages.transport.vessel_cfd import _run, openfoam_available

TORQUE_TOL = 0.03   # window means of the standard case: 1.7 % bias ± 0.6 % (MRF interface), 250-iter windows 0.4-2.8 %
OSC_TOL = 0.08
GCI_TOL = 0.10
P_FORMAL = 2.0
MESH_RATIO = 1.3
RE_TURBULENT = 1.0e4
NP_BAND = (4.0, 6.0)
WINDOW = 500
G = 9.80665
DEFAULT_TIMEOUT_S = 4800
GRENVILLE_C = 5.20
GRENVILLE_TURB = 6370.0


# --------------------------------------------------------------------------- geometry (STL, metres, tank axis = z)
def _box(cx, cy, cz, lx, ly, lz, rot=0.0):
    """Closed box centred at (cx, cy, cz), sides lx, ly, lz, rotated by rot about the z axis through the origin."""
    v = []
    for dx in (-lx / 2, lx / 2):
        for dy in (-ly / 2, ly / 2):
            for dz in (-lz / 2, lz / 2):
                x, y = cx + dx, cy + dy
                v.append((x * math.cos(rot) - y * math.sin(rot), x * math.sin(rot) + y * math.cos(rot), cz + dz))
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    f = []
    for a, b, c, d in quads:
        f += [(v[a], v[b], v[c]), (v[a], v[c], v[d])]
    return f


def _cyl(r, z0, z1, n=96, bottom=True, top=True, side=True):
    f = []
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        p0, p1 = (r * math.cos(a0), r * math.sin(a0)), (r * math.cos(a1), r * math.sin(a1))
        if side:
            f += [((p0[0], p0[1], z0), (p1[0], p1[1], z0), (p1[0], p1[1], z1)),
                  ((p0[0], p0[1], z0), (p1[0], p1[1], z1), (p0[0], p0[1], z1))]
        if bottom:
            f.append(((0.0, 0.0, z0), (p1[0], p1[1], z0), (p0[0], p0[1], z0)))
        if top:
            f.append(((0.0, 0.0, z1), (p0[0], p0[1], z1), (p1[0], p1[1], z1)))
    return f


def _stl(path: Path, name: str, facets) -> None:
    out = [f"solid {name}"]
    for a, b, c in facets:
        u = [b[i] - a[i] for i in range(3)]
        w = [c[i] - a[i] for i in range(3)]
        n = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])
        m = math.sqrt(sum(x * x for x in n)) or 1.0
        out.append(f"facet normal {n[0]/m:.6g} {n[1]/m:.6g} {n[2]/m:.6g}\n outer loop")
        out += [f"  vertex {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}" for p in (a, b, c)]
        out.append(" endloop\nendfacet")
    out.append(f"endsolid {name}")
    path.write_text("\n".join(out) + "\n")


def rushton_geometry(T: float, H: float, D: float, C: float, n_baffles: int = 4, baffle_width_m: float | None = None,
                     blade_thickness_m: float | None = None) -> dict:
    """Dimensions of a standard Rushton turbine (6 blades D/5 high × D/4 long on a 0.75 D disc) in a baffled flat-
    bottom tank. Raises ValueError when the parts do not fit. Returns every dimension and the assumptions used."""
    if min(T, H, D, C) <= 0:
        raise ValueError("tank_diameter_m, liquid_height_m, impeller_diameter_m y clearance_m deben ser positivos")
    if n_baffles not in (0, 2, 3, 4, 6, 8):
        raise ValueError("n_baffles debe ser 0, 2, 3, 4, 6 u 8")
    assumptions = []
    wb = baffle_width_m
    if wb is None:
        wb = T / 10 if n_baffles else 0.0
        if n_baffles:
            assumptions.append("ancho de mampara T/10 (geometría estándar de Rushton 1950), pegada a la pared")
    tb = blade_thickness_m
    if tb is None:
        tb = 0.03 * D
        assumptions.append("espesor de aspa 0.03 D (Np baja con aspas más gruesas; Chapple et al. 2002)")
    Wb, Lb = D / 5, D / 4
    g = {"T": T, "H": H, "D": D, "C": C, "blade_height": Wb, "blade_length": Lb, "blade_thickness": tb,
         "disc_diameter": 0.75 * D, "disc_thickness": 0.02 * D, "hub_radius": 0.10 * D, "shaft_radius": 0.06 * D,
         "n_baffles": n_baffles, "baffle_width": wb, "baffle_thickness": T / 75, "pad": T / 30}
    gap = T / 2 - wb - D / 2
    if gap < 0.02 * T:
        raise ValueError(f"el impulsor no cabe: T/2 − mampara − D/2 = {gap*1000:.1f} mm; reduce D o el ancho de mampara")
    g["mrf_radius"] = D / 2 + min(0.36 * D / 2, 0.5 * gap)
    if C - Wb / 2 <= 0.02 * T or C + Wb / 2 >= H - 0.02 * T:
        raise ValueError("las aspas quedan fuera del líquido o tocan el fondo: revisa clearance_m y liquid_height_m")
    g["mrf_height"] = min(2.5 * Wb, 1.8 * C, 1.8 * (H - C))
    if tb < 0.005 * D or tb > 0.1 * D:
        raise ValueError("blade_thickness_m fuera de 0.005-0.1 D")
    g["assumptions"] = assumptions
    return g


# --------------------------------------------------------------------------- case
def _write_case(case: Path, g: dict, cells_across: int, N: float, nu: float, rho: float, iterations: int,
                n_procs: int) -> None:
    T, H, D, C, pad = g["T"], g["H"], g["D"], g["C"], g["pad"]
    Wb, Lb, tb = g["blade_height"], g["blade_length"], g["blade_thickness"]
    tri = case / "constant" / "triSurface"
    tri.mkdir(parents=True, exist_ok=True)
    for sub in ("0", "system"):
        (case / sub).mkdir(exist_ok=True)
    dt = g["disc_thickness"]
    imp = _cyl(g["disc_diameter"] / 2, C - dt / 2, C + dt / 2) + _cyl(g["hub_radius"], C - Wb / 2, C + Wb / 2)
    for k in range(6):
        imp += _box(D / 2 - Lb / 2, 0.0, C, Lb, tb, Wb, rot=2 * math.pi * k / 6)
    _stl(tri / "impeller.stl", "impeller", imp)
    _stl(tri / "shaft.stl", "shaft", _cyl(g["shaft_radius"], C, H + pad))
    baf = []
    nb, wbf, tbf = g["n_baffles"], g["baffle_width"], g["baffle_thickness"]
    for k in range(nb):
        baf += _box(T / 2 - wbf / 2 + tbf / 2, 0.0, H / 2, wbf + tbf, tbf, H + pad, rot=math.pi / 4 + 2 * math.pi * k / nb)
    _stl(tri / "tank.stl", "tank", _cyl(T / 2, 0.0, H, n=192, bottom=True, top=False))
    _stl(tri / "top.stl", "top", _cyl(T / 2, H, H, n=192, bottom=False, top=True, side=False))
    if nb:
        _stl(tri / "baffles.stl", "baffles", baf)

    hdr = "FoamFile { version 2.0; format ascii; class %s; object %s; }\n"

    def w(rel, cls, obj, body):
        (case / rel).write_text(hdr % (cls, obj) + body)

    bg = T / cells_across
    a, z0, z1 = T / 2 + pad, -pad, H + pad
    n_xy, n_z = math.ceil(2 * a / bg), math.ceil((z1 - z0) / bg)
    w("system/blockMeshDict", "dictionary", "blockMeshDict", f"""scale 1;
vertices ((-{a} -{a} {z0}) ({a} -{a} {z0}) ({a} {a} {z0}) (-{a} {a} {z0}) (-{a} -{a} {z1}) ({a} -{a} {z1}) ({a} {a} {z1}) (-{a} {a} {z1}));
blocks (hex (0 1 2 3 4 5 6 7) ({n_xy} {n_xy} {n_z}) simpleGrading (1 1 1));
boundary ( bg {{ type patch; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }} );
""")
    hm, rm = g["mrf_height"], g["mrf_radius"]
    baf_geo = "  baffles.stl { type triSurfaceMesh; name baffles; }\n" if nb else ""
    baf_ref = "    baffles { level (1 1); patchInfo { type wall; } }\n" if nb else ""
    zp = (C + hm / 2 + H) / 2
    w("system/snappyHexMeshDict", "dictionary", "snappyHexMeshDict", f"""castellatedMesh true; snap true; addLayers false;
geometry {{
  tank.stl {{ type triSurfaceMesh; name tank; }}
{baf_geo}  impeller.stl {{ type triSurfaceMesh; name impeller; }}
  shaft.stl {{ type triSurfaceMesh; name shaft; }}
  top.stl {{ type triSurfaceMesh; name top; }}
  mrf {{ type searchableCylinder; point1 (0 0 {C - hm / 2}); point2 (0 0 {C + hm / 2}); radius {rm}; }}
}};
castellatedMeshControls {{
  maxLocalCells 6000000; maxGlobalCells 12000000; minRefinementCells 0; nCellsBetweenLevels 3; maxLoadUnbalance 0.1;
  features ();
  refinementSurfaces {{
    tank {{ level (0 1); patchInfo {{ type wall; }} }}
{baf_ref}    impeller {{ level (2 2); patchInfo {{ type wall; }} }}
    shaft {{ level (1 1); patchInfo {{ type wall; }} }}
    top {{ level (0 0); patchInfo {{ type patch; }} }}
    mrf {{ level (2 2); faceZone mrfFaces; cellZone rotor; cellZoneInside inside; }}
  }};
  resolveFeatureAngle 30;
  refinementRegions {{ mrf {{ mode inside; levels ((1 2)); }} }};
  locationInMesh ({0.30 * T} {0.07 * T} {zp}); allowFreeStandingZoneFaces true;
}};
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap true; explicitFeatureSnap false; multiRegionFeatureSnap false; }};
addLayersControls {{ relativeSizes true; layers {{ }}; expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 60; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; }};
meshQualityControls {{ maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.05; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; relaxed {{ maxNonOrtho 75; }} }};
mergeTolerance 1e-6;
""")
    omega = 2 * math.pi * N
    w("constant/MRFProperties", "dictionary", "MRFProperties",
      f"MRF1 {{ cellZone rotor; active yes; nonRotatingPatches (); origin (0 0 0); axis (0 0 1); omega {omega}; }}\n")
    w("constant/transportProperties", "dictionary", "transportProperties", f"transportModel Newtonian; nu {nu};\n")
    w("constant/turbulenceProperties", "dictionary", "turbulenceProperties",
      "simulationType RAS; RAS { RASModel kEpsilon; turbulence on; printCoeffs off; }\n")
    walls = "tank|baffles|impeller|shaft" if nb else "tank|impeller|shaft"
    rot = f"type rotatingWallVelocity; origin (0 0 0); axis (0 0 1); omega {omega}; value uniform (0 0 0);"
    w("0/U", "volVectorField", "U", f"""dimensions [0 1 -1 0 0 0 0]; internalField uniform (0 0 0);
boundaryField {{ "(tank|baffles)" {{ type noSlip; }} "(impeller|shaft)" {{ {rot} }} top {{ type slip; }} bg {{ type slip; }} }}
""")
    u_tip = math.pi * N * D
    k0 = (0.00637 * u_tip) ** 2
    e0 = k0 ** 1.5 / (0.1 * D)
    for f, val, wf, dim in (("k", k0, "kqRWallFunction", "[0 2 -2 0 0 0 0]"),
                            ("epsilon", e0, "epsilonWallFunction", "[0 2 -3 0 0 0 0]")):
        w(f"0/{f}", "volScalarField", f, f"""dimensions {dim}; internalField uniform {val};
boundaryField {{ "({walls})" {{ type {wf}; value uniform {val}; }} top {{ type zeroGradient; }} bg {{ type zeroGradient; }} }}
""")
    w("0/nut", "volScalarField", "nut", f"""dimensions [0 2 -1 0 0 0 0]; internalField uniform 0;
boundaryField {{ "({walls})" {{ type nutkWallFunction; value uniform 0; }} top {{ type calculated; value uniform 0; }} bg {{ type calculated; value uniform 0; }} }}
""")
    w("0/p", "volScalarField", "p", 'dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;\nboundaryField { ".*" { type zeroGradient; } }\n')
    stator = "tank baffles top" if nb else "tank top"
    w("system/controlDict", "dictionary", "controlDict", f"""application simpleFoam; startFrom latestTime; startTime 0; stopAt endTime;
endTime {iterations}; deltaT 1; writeControl timeStep; writeInterval {iterations}; purgeWrite 1; writeFormat binary; runTimeModifiable false;
functions {{
  rotor {{ type forces; libs (forces); patches (impeller shaft); rho rhoInf; rhoInf {rho}; CofR (0 0 0);
          writeControl timeStep; writeInterval 10; log false; }}
  stator {{ type forces; libs (forces); patches ({stator}); rho rhoInf; rhoInf {rho}; CofR (0 0 0);
           writeControl timeStep; writeInterval 10; log false; }}
  dissipation {{ type volFieldValue; libs (fieldFunctionObjects); fields (epsilon); operation volAverage; regionType all;
                writeFields false; writeControl timeStep; writeInterval {iterations}; log false; }}
}}
""")
    w("system/fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; grad(p) Gauss linear; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); div(phi,k) bounded Gauss upwind;
  div(phi,epsilon) bounded Gauss upwind; div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear limited 0.5; } interpolationSchemes { default linear; }
snGradSchemes { default limited 0.5; } wallDist { method meshWave; }
""")
    w("system/fvSolution", "dictionary", "fvSolution", """solvers {
  p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.05; }
  "(U|k|epsilon)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; maxIter 50; } }
SIMPLE { nNonOrthogonalCorrectors 1; consistent no; pRefCell 0; pRefValue 0; residualControl { p 1e-5; U 1e-6; "(k|epsilon)" 1e-6; } }
relaxationFactors { fields { p 0.3; } equations { U 0.7; "(k|epsilon)" 0.7; } }
""")
    w("system/decomposeParDict", "dictionary", "decomposeParDict", f"numberOfSubdomains {n_procs}; method scotch;\n")


# --------------------------------------------------------------------------- post-processing
def read_moment_z(dat: Path) -> list[tuple[float, float]]:
    """(iteration, total M_z) rows of an OpenFOAM forces `moment.dat` (columns: time, total xyz, pressure xyz, …)."""
    rows = []
    for ln in dat.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        v = ln.replace("(", " ").replace(")", " ").split()
        rows.append((float(v[0]), float(v[3])))
    return rows


def _moment_file(case: Path, name: str) -> Path:
    files = sorted((case / "postProcessing" / name).glob("*/moment.dat"))
    if not files:
        raise RuntimeError(f"OpenFOAM no escribió postProcessing/{name}/*/moment.dat")
    return files[-1]


def _patch_faces(case: Path) -> dict[str, int]:
    txt = (case / "constant" / "polyMesh" / "boundary").read_text(errors="replace")
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"(\w+)\s*\{[^}]*?nFaces\s+(\d+);", txt)}


def _zone_cells(case: Path, zone: str) -> int:
    p = case / "constant" / "polyMesh" / "cellZones"
    if not p.is_file():
        return 0
    txt = p.read_bytes().decode("latin-1")
    m = re.search(zone + r"\s*\{\s*type\s+cellZone;\s*cellLabels\s+List<label>\s+(\d+)", txt)
    return int(m.group(1)) if m else 0


def _mesh_stats(case: Path) -> dict:
    log = (case / "log.checkMesh").read_text(errors="replace")
    m = re.search(r"^\s*cells:\s+(\d+)", log, re.M)
    return {"cells": int(m.group(1)) if m else None, "checkMesh_ok": "Mesh OK." in log,
            "patch_faces": _patch_faces(case), "rotor_cells": _zone_cells(case, "rotor")}


def _eps_avg(case: Path) -> float | None:
    files = sorted((case / "postProcessing" / "dissipation").glob("*/volFieldValue.dat"))
    if not files:
        return None
    rows = [ln.split() for ln in files[-1].read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return float(rows[-1][1]) if rows else None


def power_number_from(rows: list[tuple[float, float]], N: float, D: float, rho: float, window: int = WINDOW) -> dict:
    norm = rho * N ** 3 * D ** 5
    t_end = rows[-1][0]
    win = [abs(mz) for t, mz in rows if t > t_end - window]
    nps = [2 * math.pi * N * m / norm for m in win]
    mean = sum(nps) / len(nps)
    return {"Np": mean, "Np_min": min(nps), "Np_max": max(nps), "Np_last": nps[-1],
            "oscillation": (max(nps) - min(nps)) / mean, "torque_N_m": sum(win) / len(win), "iterations": t_end}


def grenville_mixing_time(Np: float, N: float, D: float, T: float, H: float, Re: float) -> dict:
    """θ95 from Grenville's turbulent correlation Nθ95 = 5.20 Np^(-1/3) (T/D)² (H = T). A correlation, not CFD."""
    out = {"source": "correlación de Grenville (1992); Handbook of Industrial Mixing (2004), cap. 9 — no es resultado del CFD",
           "theta95_s": None, "N_theta95": None}
    crit = Re * Np ** (1 / 3)
    out["Re_Np13"] = crit
    if not 0.95 <= H / T <= 1.05:
        out["note"] = f"H/T = {H/T:.2f}: la correlación es para H = T; no se da θ95"
    elif crit <= GRENVILLE_TURB:
        out["note"] = f"Re·Np^(1/3) = {crit:.0f} ≤ {GRENVILLE_TURB:.0f}: régimen de transición, fuera de la correlación turbulenta"
    else:
        nt = GRENVILLE_C * Np ** (-1 / 3) * (T / D) ** 2
        out.update({"N_theta95": nt, "theta95_s": nt / N})
    return out


def grid_convergence(phi: list[float], cells: list[int], p_formal: float = P_FORMAL) -> dict:
    """Celik et al. (2008) three-mesh procedure. phi and cells ordered fine → coarse. Representative size h ∝ N^(-1/3).
    Returns the convergence ratio R = e21/e32 (0 < R < 1 monotonic), the observed order, the Richardson value and
    GCI_fine computed with p = min(p_observed, p_formal)."""
    (f1, f2, f3), (n1, n2, n3) = phi, cells
    r21, r32 = (n1 / n2) ** (1 / 3), (n2 / n3) ** (1 / 3)
    e21, e32 = f2 - f1, f3 - f2
    out = {"r21": r21, "r32": r32, "e21": e21, "e32": e32, "monotonic": False, "p_observed": None, "p_used": None,
           "phi_extrapolated": None, "gci_fine": None}
    if e32 == 0 or e21 == 0:
        out["R"] = 0.0 if e21 == 0 else math.inf
        if e21 == 0:
            out.update({"monotonic": True, "phi_extrapolated": f1, "gci_fine": 0.0})
        return out
    R = e21 / e32
    out["R"] = R
    if not 0 < R < 1:
        return out
    s = 1.0
    p = p_formal
    for _ in range(100):
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    pu = min(p, p_formal)
    out.update({"monotonic": True, "p_observed": p, "p_used": pu,
                "phi_extrapolated": (r21 ** pu * f1 - f2) / (r21 ** pu - 1),
                "gci_fine": 1.25 * abs((f1 - f2) / f1) / (r21 ** pu - 1)})
    return out


def _solve(g: dict, cells_across: int, N: float, nu: float, rho: float, iterations: int, n_procs: int,
           timeout_s: int, keep: Path | None) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="alchemai_stirred_", dir=os.environ.get("TMPDIR") or "/tmp")).resolve()
    if " " in str(tmp):
        raise RuntimeError(f"la carpeta temporal tiene espacios ({tmp}); mpirun no la acepta. Define TMPDIR sin espacios")
    case = tmp / "case"
    try:
        _write_case(case, g, cells_across, N, nu, rho, iterations, n_procs)
        t0 = time.monotonic()
        _run(case, "blockMesh > log.blockMesh 2>&1 && snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 && "
                   "(checkMesh > log.checkMesh 2>&1 || true)", timeout_s)
        t_mesh = time.monotonic() - t0
        mesh = _mesh_stats(case)
        missing = [p for p in (["impeller", "shaft", "tank"] + (["baffles"] if g["n_baffles"] else []))
                   if mesh["patch_faces"].get(p, 0) == 0]
        if missing or mesh["rotor_cells"] == 0:
            raise RuntimeError(f"la malla perdió {missing or 'la zona rotatoria'}: sube cells_across_tank")
        left = max(60, int(timeout_s - t_mesh))
        if n_procs > 1:
            _run(case, f"decomposePar -force > log.decomposePar 2>&1 && "
                       f"mpirun -np {n_procs} simpleFoam -parallel > log.simpleFoam 2>&1", left)
        else:
            _run(case, "simpleFoam > log.simpleFoam 2>&1", left)
        t_solve = time.monotonic() - t0 - t_mesh
        rotor = read_moment_z(_moment_file(case, "rotor"))
        stator = read_moment_z(_moment_file(case, "stator"))
        res = power_number_from(rotor, N, g["D"], rho)
        t_end = rotor[-1][0]
        m_r = sum(m for t, m in rotor if t > t_end - WINDOW) / len([1 for t, _ in rotor if t > t_end - WINDOW])
        m_s = sum(m for t, m in stator if t > t_end - WINDOW) / len([1 for t, _ in stator if t > t_end - WINDOW])
        res.update({"M_rotor_N_m": m_r, "M_stator_N_m": m_s, "torque_imbalance": abs(m_r + m_s) / abs(m_r),
                    "eps_volume_avg_W_kg": _eps_avg(case), "mesh": mesh, "cells_across_tank": cells_across,
                    "timing_s": {"mesh": round(t_mesh, 1), "solve": round(t_solve, 1)}, "n_procs": n_procs})
        if keep is not None:
            keep.mkdir(parents=True, exist_ok=True)
            dest = keep / f"stirred_{cells_across}"
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(case, dest, ignore=shutil.ignore_patterns("processor*"))
            res["case_dir"] = str(dest)
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def stirred_tank_rushton(tank_diameter_m: float, liquid_height_m: float, impeller_diameter_m: float,
                         clearance_m: float, rotational_speed_rps: float, density_kg_m3: float, viscosity_Pa_s: float,
                         n_baffles: int = 4, baffle_width_m: float | None = None,
                         blade_thickness_m: float | None = None, cells_across_tank: int = 64, mesh_study: bool = True,
                         iterations: int = 3000, n_procs: int | None = None, keep_case_dir: str | None = None,
                         timeout_s: int = DEFAULT_TIMEOUT_S) -> dict:
    """Power number, power draw and mixing time of a baffled tank with a Rushton turbine by MRF CFD (see module)."""
    if not openfoam_available():
        raise RuntimeError("OpenFOAM no está instalado (falta el lanzador `openfoam`). Ver transporte-lab/README.md")
    if rotational_speed_rps <= 0 or density_kg_m3 <= 0 or viscosity_Pa_s <= 0:
        raise ValueError("rotational_speed_rps, density_kg_m3 y viscosity_Pa_s deben ser positivos")
    lo = 50 if mesh_study else 38
    if not lo <= cells_across_tank <= 90:
        raise ValueError(f"cells_across_tank entre {lo} y 90 (64 ≈ 0.9 M celdas, ~45 min en 4 núcleos; con estudio de "
                         "malla las otras dos mallas son 1.3 y 1.69 veces más gruesas)")
    if iterations < 2 * WINDOW:
        raise ValueError(f"iterations ≥ {2 * WINDOW}")
    T, H, D, C, N = tank_diameter_m, liquid_height_m, impeller_diameter_m, clearance_m, rotational_speed_rps
    g = rushton_geometry(T, H, D, C, n_baffles, baffle_width_m, blade_thickness_m)
    rho, mu = density_kg_m3, viscosity_Pa_s
    nu = mu / rho
    Re = rho * N * D ** 2 / mu
    n_procs = n_procs or max(1, min(4, os.cpu_count() or 1))
    keep = Path(keep_case_dir).expanduser().resolve() if keep_case_dir else None

    levels = [cells_across_tank]
    if mesh_study:
        levels += [round(cells_across_tank / MESH_RATIO), round(cells_across_tank / MESH_RATIO ** 2)]
    meshes = [_solve(g, n, N, nu, rho, iterations, n_procs, timeout_s, keep) for n in levels]
    fine = meshes[0]
    gc = grid_convergence([m["Np"] for m in meshes], [m["mesh"]["cells"] for m in meshes]) if mesh_study else None

    Np = fine["Np"]
    V = math.pi * T ** 2 / 4 * H
    P = Np * rho * N ** 3 * D ** 5
    eps_mean = P / (rho * V)
    mesh_ok = gc is not None and gc["monotonic"] and gc["gci_fine"] is not None and gc["gci_fine"] < GCI_TOL
    ev = {
        "torque_balance": all(m["torque_imbalance"] < TORQUE_TOL for m in meshes),
        "torque_imbalance": fine["torque_imbalance"],
        "steady": all(m["oscillation"] < OSC_TOL for m in meshes), "oscillation": fine["oscillation"],
        "mesh_independent": mesh_ok if gc else None, "grid_convergence": gc,
        "mesh_ok": all(m["mesh"]["checkMesh_ok"] for m in meshes),
        "turbulent": Re > RE_TURBULENT, "Re": Re,
        "plausible": NP_BAND[0] <= Np <= NP_BAND[1],
    }
    reasons = []
    if not ev["torque_balance"]:
        worst = max(m["torque_imbalance"] for m in meshes)
        reasons.append(f"el torque del impulsor no sale por las paredes (desbalance hasta {worst:.1%}); "
                       "el flujo no está en régimen")
    if not ev["steady"]:
        reasons.append(f"Np oscila más de {OSC_TOL:.0%} en las últimas {WINDOW} iteraciones; sube iterations")
    if gc is None:
        reasons.append("sin estudio de malla (mesh_study=False)")
    elif not gc["monotonic"]:
        reasons.append(f"Np no converge de forma monótona entre las 3 mallas (R = {gc['R']:.2f}); sube cells_across_tank")
    elif not mesh_ok:
        reasons.append(f"GCI de la malla fina {gc['gci_fine']:.1%} ≥ {GCI_TOL:.0%}; sube cells_across_tank")
    if not ev["mesh_ok"]:
        reasons.append("checkMesh reporta fallas de calidad")
    if not ev["turbulent"]:
        reasons.append(f"Re = {Re:.0f} ≤ {RE_TURBULENT:.0f}: k-ε no aplica y Np depende de Re; este resultado es orientativo")
    if not ev["plausible"]:
        reasons.append(f"Np = {Np:.2f} fuera de la banda {NP_BAND[0]}-{NP_BAND[1]} de la literatura para Rushton con "
                       "mamparas; revisa la geometría")
    ev["verified"] = not reasons
    ev["reasons"] = reasons
    eps_cfd = fine.get("eps_volume_avg_W_kg")
    return {
        "Np": Np, "Np_range_last_window": [fine["Np_min"], fine["Np_max"]],
        "Np_by_mesh": {m["cells_across_tank"]: m["Np"] for m in meshes},
        "Np_extrapolated": gc["phi_extrapolated"] if gc else None,
        "Np_uncertainty_fraction": gc["gci_fine"] if gc else None,
        "power_W": P, "power_per_volume_W_m3": P / V, "power_per_mass_W_kg": eps_mean,
        "torque_N_m": fine["torque_N_m"], "Re": Re, "Fr": N ** 2 * D / G, "tip_speed_m_s": math.pi * N * D,
        "liquid_volume_m3": V, "kolmogorov_scale_m": (nu ** 3 / eps_mean) ** 0.25,
        "mixing_time": grenville_mixing_time(Np, N, D, T, H, Re),
        "eps_cfd_over_mean": (eps_cfd / eps_mean) if eps_cfd else None,
        "geometry": {k: v for k, v in g.items() if k != "assumptions"},
        "runs": [{k: m[k] for k in ("cells_across_tank", "Np", "oscillation", "torque_imbalance", "M_rotor_N_m",
                                     "M_stator_N_m", "mesh", "timing_s", "n_procs", "iterations")} |
                 ({"case_dir": m["case_dir"]} if "case_dir" in m else {}) for m in meshes],
        "engine": "OpenFOAM simpleFoam + MRF, k-ε con funciones de pared, malla snappyHexMesh",
        "assumptions": g["assumptions"] + [
            "flujo estacionario con marco rotatorio múltiple (MRF): congela la posición aspa-mampara",
            "superficie libre plana (slip): válido sin vórtice, es decir con mamparas y Fr bajo",
            "fluido newtoniano, una sola fase, isotérmico"],
        "evidence": ev,
        "notes": ("k-ε + MRF suele quedar dentro de ~10 % del Np medido y subestima ε (eps_cfd_over_mean < 1 es "
                  "normal). El tiempo de mezclado sale de una correlación, no del CFD. Solo turbina Rushton por ahora."),
    }
