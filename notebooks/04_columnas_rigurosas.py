"""Week-4 study: rigorous distillation columns — DWSIM 10 (Fluent API) vs FluxoEngine (Wang-Henke).

Two cases, designed to separate *algorithm* error from *thermodynamic* error:
  A. Depropanizer C2/C3/nC4/nC5, PR in both engines   -> same thermo, differences = column algorithm.
  B. Ethanol-water, 20 stages: DWSIM NRTL (truth), DWSIM PR, Fluxo PR
       -> DWSIM-PR vs Fluxo-PR isolates the algorithm; DWSIM-NRTL vs DWSIM-PR isolates the thermo
          (quantifies what Fluxo loses for not having an activity model yet).

Fluxo numbers come from `swift test --filter LineaBaseDWSIM` (BASELINE_JSON lines) captured in
notebooks/fluxo_columnas_raw.jsonl. Output: notebooks/04_columnas_rigurosas.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.dwsim_client.fluent import fluent  # noqa: E402

Flowsheet, PropertyPackages, Q = fluent()
import System  # noqa: E402
from DWSIM.Automation.FluentAPI.Builders import CompositionBuilder  # noqa: E402

RAW = ROOT / "notebooks/fluxo_columnas_raw.jsonl"
fluxo = {}
if RAW.exists():
    for line in RAW.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            fluxo[d["caso"]] = d


def comp(pairs):
    def _set(cb):
        for name, z in pairs:
            cb.Mole(name, z)
    return System.Action[CompositionBuilder](_set)


def profile(col):
    """Stage profiles from a solved DWSIM column: T (K), P (Pa), L, V (mol/s), x, y per stage (top = 0)."""
    o = col.Object
    T = [float(v) for v in o.Tf]
    P = [float(v) for v in o.P0]
    L = [float(v) for v in o.Lf]
    V = [float(v) for v in o.Vf]
    x = [[float(v) for v in row] for row in o.xf]
    y = [[float(v) for v in row] for row in o.yf]
    return {"T": T, "P": P, "L": L, "V": V, "x": x, "y": y}


def run_column(name, compounds, pkg, feed_spec, stages, feed_stage, top_P, dP, reflux, spec_kind, spec_val, max_iter=None):
    fs = Flowsheet.Create(name)
    for c in compounds:
        fs = fs.WithCompound(c)
    fs = fs.WithPropertyPackage(getattr(PropertyPackages, pkg))
    feed = fs.AddMaterialStream("feed").WithPressure(Q.Pascal(feed_spec["P"])).WithMolarFlow(Q.MolPerSecond(feed_spec["F"]))
    if "T" in feed_spec:
        feed = feed.WithTemperature(Q.Kelvin(feed_spec["T"]))
    else:
        feed = feed.WithVaporFraction(feed_spec["VF"])
    feed = feed.WithComposition(comp(zip(compounds, feed_spec["z"])))
    dist, bot = fs.AddMaterialStream("dist"), fs.AddMaterialStream("bot")
    cd, rd = fs.AddEnergyStream("cd"), fs.AddEnergyStream("rd")
    # DWSIM's WithFeed index is 0-based (WithFeed(s, 2) lands on "Stage3"); feed_stage here is 1-based like Fluxo.
    col = (fs.AddDistillationColumn("T-1").WithNumberOfStages(stages).WithFeed(feed, feed_stage - 1)
           .WithDistillate(dist).WithBottoms(bot).WithCondenserDuty(cd).WithReboilerDuty(rd)
           .WithCondenserSpec("Reflux Ratio", reflux, "")
           .WithTopPressure(Q.Pascal(top_P)).WithColumnPressureDrop(Q.Pascal(dP)))
    # Second spec always on the reboiler as bottoms flow (setting the condenser spec twice hangs the solver).
    col = col.WithReboilerSpec("Product Molar Flow Rate", feed_spec["F"] - spec_val if spec_kind == "D" else spec_val, "mol/s")
    if max_iter:
        col = col.WithMaxIterations(int(max_iter))
    fs.AutoLayout()
    errs = list(fs.TrySolve())
    if errs:
        raise RuntimeError(f"{name}: {[str(e) for e in errs]}")
    o = col.Object
    out = {
        "iterations": int(o.ColumnSolverConvergenceReport.Count) if hasattr(o.ColumnSolverConvergenceReport, "Count") else None,
        "feed_T_K": feed.TemperatureK, "feed_VF": feed.VaporFraction,
        "D_mol_s": dist.MolarFlowMolPerSecond, "B_mol_s": bot.MolarFlowMolPerSecond,
        "T_dist_K": dist.TemperatureK, "T_bot_K": bot.TemperatureK,
        "xD": [dist.OverallMoleFraction(c) for c in compounds],
        "xB": [bot.OverallMoleFraction(c) for c in compounds],
        "Qc_kW": cd.EnergyFlowKW, "Qr_kW": rd.EnergyFlowKW,
        "solver": str(o.SolvingMethodName),
        "profile": profile(col),
    }
    return out


def rel(a, b):
    return (a - b) / b if b else float("nan")


rows = []


def add(case, var, f, d, unit="", fmt=".4g"):
    rows.append((case, var, f, d, unit, fmt))


# ------------------------------------------------------------------ DWSIM cases (each in its own process: `--case A|B_nrtl|B_pr`)
HC = ["Ethane", "Propane", "N-butane", "N-pentane"]
zA = [0.05, 0.45, 0.40, 0.10]
EW = ["Ethanol", "Water"]
zB = [0.5, 0.5]
CASES = {
    "A": lambda: run_column("depro", HC, "PengRobinson", {"T": 330.0, "P": 17e5, "F": 100.0, "z": zA},
                            stages=22, feed_stage=12, top_P=16e5, dP=1.5e5, reflux=1.84, spec_kind="D", spec_val=50.0),
    # Feed-stage index convention check (DWSIM WithFeed(stream, n): is n 0-based?)
    "A_f11": lambda: run_column("depro11", HC, "PengRobinson", {"T": 330.0, "P": 17e5, "F": 100.0, "z": zA},
                                stages=22, feed_stage=11, top_P=16e5, dP=1.5e5, reflux=1.84, spec_kind="D", spec_val=50.0),
    "A_f13": lambda: run_column("depro13", HC, "PengRobinson", {"T": 330.0, "P": 17e5, "F": 100.0, "z": zA},
                                stages=22, feed_stage=13, top_P=16e5, dP=1.5e5, reflux=1.84, spec_kind="D", spec_val=50.0),
    "B_nrtl": lambda: run_column("etoh_nrtl", EW, "NRTL", {"T": 300.0, "P": 101325.0, "F": 100.0, "z": zB},
                                 stages=20, feed_stage=10, top_P=101325.0, dP=0.0, reflux=2.0, spec_kind="B", spec_val=75.0),
    "B_pr": lambda: run_column("etoh_pr", EW, "PengRobinson", {"T": 300.0, "P": 101325.0, "F": 100.0, "z": zB},
                               stages=20, feed_stage=10, top_P=101325.0, dP=0.0, reflux=2.0, spec_kind="B", spec_val=75.0),
    # Same with the column capped at 30 outer iterations: tells whether DWSIM is stuck in the column loop or in the flash.
    "B_pr_30": lambda: run_column("etoh_pr30", EW, "PengRobinson", {"T": 300.0, "P": 101325.0, "F": 100.0, "z": zB},
                                  stages=20, feed_stage=10, top_P=101325.0, dP=0.0, reflux=2.0, spec_kind="B", spec_val=75.0,
                                  max_iter=30),
}
OUT = ROOT / "notebooks/04_dwsim_columnas_raw"
OUT.mkdir(exist_ok=True)

if len(sys.argv) > 2 and sys.argv[1] == "--case":
    import time
    t0 = time.time()
    res = CASES[sys.argv[2]]()
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / f"{sys.argv[2]}.json").write_text(json.dumps(res, default=float, indent=1))
    print(sys.argv[2], json.dumps({k: v for k, v in res.items() if k != "profile"}, default=float))
    print("T:", [round(t, 2) for t in res["profile"]["T"]])
    sys.exit(0)

A, B_nrtl = (json.loads((OUT / f"{c}.json").read_text()) for c in ("A", "B_nrtl"))
B_pr = json.loads((OUT / "B_pr.json").read_text()) if (OUT / "B_pr.json").exists() else None
B_pr_30 = json.loads((OUT / "B_pr_30.json").read_text()) if (OUT / "B_pr_30.json").exists() else None

# ------------------------------------------------------------------ Comparison table (Fluxo raw must exist)
def pct(f, d):
    return f"{100 * rel(f, d):+.2f} %" if d else "—"


def compare_column(case, F, D, compounds, lk_idx, hk_idx):
    """Rows: products, duties, temperatures, max |ΔT| over the stage profile, max |Δx| of the keys."""
    add(case, "D (mol/s)", F["D"], D["D_mol_s"])
    add(case, f"x_D {compounds[lk_idx]}", F["xD"][lk_idx], D["xD"][lk_idx])
    add(case, f"x_D {compounds[hk_idx]}", F["xD"][hk_idx], D["xD"][hk_idx])
    add(case, f"x_B {compounds[lk_idx]}", F["xB"][lk_idx], D["xB"][lk_idx])
    add(case, f"x_B {compounds[hk_idx]}", F["xB"][hk_idx], D["xB"][hk_idx])
    add(case, "T destilado (K)", F["tDestK"], D["T_dist_K"])
    add(case, "T fondo (K)", F["tFondoK"], D["T_bot_K"])
    add(case, "Qc (kW)", F["qCondW"] / 1e3, -abs(D["Qc_kW"]))
    add(case, "Qr (kW)", F["qRehW"] / 1e3, D["Qr_kW"])
    Tf, Td = F["T"], D["profile"]["T"]
    dT = [a - b for a, b in zip(Tf, Td)]
    j = max(range(len(dT)), key=lambda k: abs(dT[k]))
    add(case, f"ΔT máx. del perfil (etapa {j + 1})", Tf[j], Td[j], "K")
    xf, xd = F["x"], D["profile"]["x"]
    dx = [abs(xf[k][lk_idx] - xd[k][lk_idx]) for k in range(len(xf))]
    k = max(range(len(dx)), key=lambda q: dx[q])
    add(case, f"Δx máx. {compounds[lk_idx]} líquido (etapa {k + 1})", xf[k][lk_idx], xd[k][lk_idx])
    add(case, "iteraciones", F["iter"], D.get("iterations") or float("nan"), fmt="d")


lines = ["# Columnas rigurosas — DWSIM 10.2.12 vs FluxoEngine (Wang-Henke)", "",
         "Semana 4 del plan: ¿dónde falla Fluxo en destilación rigurosa? Dos casos, diseñados para separar el error",
         "del **algoritmo** del error de la **termodinámica**.", "",
         "- **A. Depropanizadora** C2/C3/nC4/nC5 (0.05/0.45/0.40/0.10), 100 mol/s a 330 K y 17 bar, 22 etapas, alimentación",
         "  en la 12, 16 → 17.5 bar, R = 1.84, D = 50 mol/s. **PR en ambos** → las diferencias son del algoritmo.",
         "- **B. Etanol-agua** 50/50, 100 mol/s a 300 K y 1 atm, 20 etapas, alimentación en la 10, R = 2, B = 75 mol/s.",
         "  DWSIM **NRTL** (verdad) vs DWSIM **PR** vs Fluxo **PR**: DWSIM-PR vs Fluxo-PR aísla el algoritmo; NRTL vs PR aísla",
         "  lo que Fluxo pierde por no tener modelo de actividad.", "",
         f"Solver de DWSIM: `{A['solver']}` (el mismo método que Fluxo); tolerancias 1e-5.", "",
         "**Hallazgo lateral**: `WithFeed(stream, n)` de la Fluent API cuenta etapas **desde 0** (`WithFeed(s, 2)` cae en",
         "«Stage3»). Con la alimentación «en la 12» pasada tal cual, la desviación máxima del perfil era 3.8 K justo bajo el",
         "plato de alimentación y la impureza clave difería 14 %; con el índice corregido (abajo) todo cae a < 1 K y ≈ 5 %.",
         "Corregido en `tools.py` (feed_stage 1-based como en los libros), en el caso de referencia 9 y en las tarjetas.", ""]

if "depro" in fluxo:
    compare_column("A depro", fluxo["depro"], A, HC, 1, 2)
if "etoh_pr" in fluxo and B_pr:
    compare_column("B etanol-agua PR", fluxo["etoh_pr"], B_pr, EW, 0, 1)
    compare_column("B Fluxo-PR vs DWSIM-NRTL", fluxo["etoh_pr"], B_nrtl, EW, 0, 1)

lines += ["| Caso | Variable | Fluxo | DWSIM | Desv. |", "|---|---|---|---|---|"]
for case, var, f, d, unit, fmt in rows:
    if fmt == "d":
        lines.append(f"| {case} | {var} | {int(f)} | {d if d == d else '—'} | — |")
    else:
        lines.append(f"| {case} | {var} | {f:{fmt}} {unit} | {d:{fmt}} {unit} | {pct(f, d)} |")

if "etoh_pr" not in fluxo:
    lines += ["", "## B en Fluxo: Wang-Henke **no converge** con PR", "",
              "Fluxo no llegó a 2000 iteraciones con amortiguamiento 1, 0.5, 0.25 ni 0.1 (error de equilibrio 0.07–0.84).",
              "La causa es termodinámica, no del algoritmo: la curva de burbuja a 1 atm que da el PR de Fluxo (k_ij = 0) para",
              "etanol-agua no es monótona y el vapor sale **más pobre** que el líquido en el centro (LLE espurio del cúbico",
              "con reglas de mezclado clásicas en una mezcla polar). Con esas K el punto de burbuja oscila.", "",
              "| x etanol | T burbuja Fluxo-PR (K) | y etanol |", "|---|---|---|"]
    txy = ROOT / "notebooks/fluxo_txy_etoh_pr.txt"
    if txy.exists():
        for ln in txy.read_text().splitlines():
            parts = dict(kv.split("=") for kv in ln.replace("TXY_PR ", "").split())
            lines.append(f"| {float(parts['x']):.1f} | {float(parts['T']):.2f} | {float(parts['y']):.4f} |")
    lines += ["", "Referencia: etanol puro hierve a 351.4 K y agua a 373.1 K; el azeótropo real está en x ≈ 0.89, 351.3 K (NRTL",
              "en `02_nrtl_etanol_agua.py`). **Conclusión para Fluxo 1.1/1.2: hace falta NRTL (E4) para cualquier mezcla polar;",
              "mientras tanto la app debe avisar que PR/SRK no son válidos para agua + alcoholes/ácidos.**"]

lines += ["", "## B en DWSIM: NRTL converge en 2.3 s; **PR tampoco converge**", ""]
if B_pr:
    lines += ["| Variable | NRTL | PR | PR − NRTL |", "|---|---|---|---|"]
    for var, a, b in [("x_D etanol", B_nrtl["xD"][0], B_pr["xD"][0]), ("x_B etanol", B_nrtl["xB"][0], B_pr["xB"][0]),
                      ("T destilado (K)", B_nrtl["T_dist_K"], B_pr["T_dist_K"]), ("T fondo (K)", B_nrtl["T_bot_K"], B_pr["T_bot_K"]),
                      ("Qc (kW)", B_nrtl["Qc_kW"], B_pr["Qc_kW"]), ("Qr (kW)", B_nrtl["Qr_kW"], B_pr["Qr_kW"])]:
        lines.append(f"| {var} | {a:.4g} | {b:.4g} | {b - a:+.4g} |")
else:
    lines += ["La misma columna con `PengRobinson` en DWSIM 10.2.12 (Wang-Henke, 100 iteraciones máx.) **no termina en 300 s**",
              "(matada; > 6 min de CPU), mientras que con NRTL resuelve en 2.3 s y la depropanizadora PR en 2.2 s.",
              "Es decir: el mismo algoritmo, en el simulador de referencia, falla con el mismo paquete en la misma mezcla. El",
              "problema es la **ecuación de estado para una mezcla polar**, no Wang-Henke ni Fluxo."]
    lines += ["", "Con `WithMaxIterations(30)` DWSIM termina y lanza `Solver reached the maximum number of iterations without",
              "converging` desde `Column.SolveBubblePointRetry` (RigorousColumn.vb:6818): el solver reintenta con otras",
              "estrategias y con el tope por defecto (100) ese bucle de reintentos es lo que pasa de 300 s. Ninguno de los dos",
              "motores entrega una solución falsa: DWSIM lanza excepción y Fluxo deja la columna en rojo con el error de equilibrio."]
lines += ["", f"Referencia NRTL (DWSIM): x_D etanol {B_nrtl['xD'][0]:.4f}, x_B {B_nrtl['xB'][0]:.4f}, T {B_nrtl['T_dist_K']:.2f} → "
          f"{B_nrtl['T_bot_K']:.2f} K, Qc {B_nrtl['Qc_kW']:.1f} kW, Qr {B_nrtl['Qr_kW']:.1f} kW."]

lines += ["", "## Perfiles de temperatura por etapa (K)", "", "| Etapa | A Fluxo | A DWSIM | B DWSIM-NRTL |", "|---|---|---|---|"]
nmax = max(len(A["profile"]["T"]), len(B_nrtl["profile"]["T"]))
for j in range(nmax):
    def g(seq):
        return f"{seq[j]:.2f}" if j < len(seq) else ""
    lines.append(f"| {j + 1} | {g(fluxo.get('depro', {}).get('T', []))} | {g(A['profile']['T'])} | {g(B_nrtl['profile']['T'])} |")

(ROOT / "notebooks/04_columnas_rigurosas.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
