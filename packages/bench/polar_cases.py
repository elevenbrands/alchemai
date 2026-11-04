"""Bench v2 — polar mixtures (block 13). The statements deliberately do NOT name a property package:
the agent has to recognise water + alcohol at 1 atm as a polar, non-ideal liquid and pick an activity
model (NRTL) instead of the cubic EOS it uses for hydrocarbons. With Peng-Robinson these cards come out
wrong by tens of kelvin (the PR bubble curve of ethanol-water is not monotone; see notebooks/04) or do
not converge at all, so the numbers alone grade the choice.

Oracles are DWSIM 10.2.12 with its own NRTL set (the same parameters FluxoEngine reproduces to 0.1 K,
notebooks/05 and 06). Run `uv run python packages/bench/polar_cases.py` to print them with PR alongside.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.dwsim_client.fluent import fluent  # noqa: E402
from packages.bench.cases_baseline import composition  # noqa: E402
from packages.bench.cases_ref import Case  # noqa: E402
from packages.bench.design_cases import DesignCase  # noqa: E402

Flowsheet, PropertyPackages, Q = fluent()

P_ATM = 101325.0


def _pp(name: str):
    return getattr(PropertyPackages, name)


# --------------------------------------------------------------------------- 1 bubble and dew points, ethanol-water 50/50
def b_burbuja_rocio(pp="NRTL"):
    names, z = ["Ethanol", "Water"], [0.5, 0.5]
    fs = Flowsheet.Create("bp").WithCompounds(*names).WithPropertyPackage(_pp(pp))
    bub = (fs.AddMaterialStream("bubble").WithPressure(Q.Pascal(P_ATM)).WithVaporFraction(0.0)
           .WithMolarFlow(Q.MolPerSecond(1.0)).WithComposition(composition(zip(names, z))))
    dew = (fs.AddMaterialStream("dew").WithPressure(Q.Pascal(P_ATM)).WithVaporFraction(1.0)
           .WithMolarFlow(Q.MolPerSecond(1.0)).WithComposition(composition(zip(names, z))))
    fs.AutoLayout()
    return fs, {"bub": bub, "dew": dew}


def r_burbuja_rocio(h):
    return {"T_bubble_K": h["bub"].TemperatureK, "T_dew_K": h["dew"].TemperatureK}


def c_burbuja_rocio(h, rep):
    tb, td = h["bub"].TemperatureK, h["dew"].TemperatureK
    return [("Burbuja entre Tb etanol (351.4) y Tb agua (373.2)", 351.0 < tb < 373.2, f"T_b={tb:.2f}"),
            ("Rocío por encima de la burbuja", td > tb, f"T_d={td:.2f}"),
            ("Rocío bajo Tb del agua", td < 373.2, "")]


# --------------------------------------------------------------------------- 2 PT flash, methanol-water 40/60 at 355 K
def b_flash_metanol(pp="NRTL"):
    names, z = ["Methanol", "Water"], [0.4, 0.6]
    fs = Flowsheet.Create("fl").WithCompounds(*names).WithPropertyPackage(_pp(pp))
    feed = (fs.AddMaterialStream("feed").At(Q.Kelvin(355.0), Q.Pascal(P_ATM)).WithMolarFlow(Q.MolPerSecond(100.0))
            .WithComposition(composition(zip(names, z))))
    vap, liq = fs.AddMaterialStream("vap"), fs.AddMaterialStream("liq")
    fs.AddSeparator("V-1").ConnectFeed(feed, 0).ConnectProduct(vap, 0).ConnectProduct(liq, 1)
    fs.AutoLayout()
    return fs, {"feed": feed, "vap": vap, "liq": liq}


def r_flash_metanol(h):
    return {"VF": h["feed"].VaporFraction, "y_MeOH": h["vap"].OverallMoleFraction("Methanol"),
            "x_MeOH": h["liq"].OverallMoleFraction("Methanol")}


def c_flash_metanol(h, rep):
    vf, y, x = h["feed"].VaporFraction, h["vap"].OverallMoleFraction("Methanol"), h["liq"].OverallMoleFraction("Methanol")
    return [("Bifásico", 0.02 < vf < 0.98, f"VF={vf:.4f}"),
            ("Vapor más rico en metanol que el líquido", y > 0.4 > x, f"y={y:.4f} x={x:.4f}"),
            ("Balance de metanol VF·y + (1−VF)·x = 0.4", abs(vf * y + (1 - vf) * x - 0.4) < 1e-3, f"{vf * y + (1 - vf) * x:.4f}")]


# --------------------------------------------------------------------------- 3 methanol-water column
def b_columna_metanol(pp="NRTL"):
    fs = Flowsheet.Create("colm").WithCompounds("Methanol", "Water").WithPropertyPackage(_pp(pp))
    feed = (fs.AddMaterialStream("feed").WithTemperature(Q.Kelvin(340.0)).WithPressure(Q.Pascal(P_ATM)).WithMolarFlow(Q.MolPerSecond(100.0))
            .SetCompoundMolarFlow("Methanol", 40.0).SetCompoundMolarFlow("Water", 60.0))
    dist, bot = fs.AddMaterialStream("dist"), fs.AddMaterialStream("bot")
    cd, rd = fs.AddEnergyStream("cd"), fs.AddEnergyStream("rd")
    col = (fs.AddDistillationColumn("T-201").WithNumberOfStages(20).WithFeed(feed, 10 - 1).WithDistillate(dist).WithBottoms(bot)
           .WithCondenserDuty(cd).WithReboilerDuty(rd).WithCondenserSpec("Reflux Ratio", 1.5, "")
           .WithReboilerSpec("Product Molar Flow Rate", 60.0, "mol/s").WithTopPressure(Q.Pascal(P_ATM)).WithColumnPressureDrop(Q.Pascal(0.0)))
    fs.AutoLayout()
    return fs, {"feed": feed, "dist": dist, "bot": bot, "cd": cd, "rd": rd, "col": col}


def r_columna_metanol(h):
    return {"x_MeOH_dist": h["dist"].OverallMoleFraction("Methanol"), "x_MeOH_bot": h["bot"].OverallMoleFraction("Methanol"),
            "Qc_kW": h["cd"].EnergyFlowKW, "Qr_kW": h["rd"].EnergyFlowKW}


def c_columna_metanol(h, rep):
    xd, xb = h["dist"].OverallMoleFraction("Methanol"), h["bot"].OverallMoleFraction("Methanol")
    return [("Destilado rico en metanol (sin azeótropo: puede pasar de 0.9)", xd > 0.9, f"x_D={xd:.4f}"),
            ("Fondo pobre en metanol", xb < 0.1, f"x_B={xb:.4f}"),
            ("D = F − B = 40 mol/s", abs(h["dist"].MolarFlowMolPerSecond - 40.0) < 0.05, f"D={h['dist'].MolarFlowMolPerSecond:.3f}"),
            ("Balance de metanol 40·x_D + 60·x_B = 40", abs(40.0 * xd + 60.0 * xb - 40.0) < 0.1, f"{40 * xd + 60 * xb:.3f}")]


# --------------------------------------------------------------------------- 4 ethanol-water azeotrope at 1 atm (design)
def _bubble_T(x_etoh: float, pp="NRTL") -> float:
    names = ["Ethanol", "Water"]
    fs = Flowsheet.Create("az").WithCompounds(*names).WithPropertyPackage(_pp(pp))
    s = (fs.AddMaterialStream("s").WithPressure(Q.Pascal(P_ATM)).WithVaporFraction(0.0).WithMolarFlow(Q.MolPerSecond(1.0))
         .WithComposition(composition(zip(names, [x_etoh, 1.0 - x_etoh]))))
    fs.Solve()
    return s.TemperatureK


def o_azeotropo(pp="NRTL"):
    # Minimum-boiling azeotrope = minimum of the bubble curve. Golden-section search on x in [0.7, 0.99].
    lo, hi = 0.70, 0.99
    g = (5 ** 0.5 - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = _bubble_T(a, pp), _bubble_T(b, pp)
    for _ in range(40):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = _bubble_T(a, pp)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = _bubble_T(b, pp)
        if hi - lo < 2e-4:
            break
    x = 0.5 * (lo + hi)
    return {"x_az_EtOH": x, "T_az_K": _bubble_T(x, pp)}


def c_azeotropo(a):
    return [("Azeótropo entre 0.85 y 0.92 (lit. 0.894; DWSIM-NRTL 0.868)", 0.85 < a["x_az_EtOH"] < 0.92, f"x={a['x_az_EtOH']:.4f}"),
            ("T del azeótropo bajo la Tb del etanol de DWSIM (351.8 K; exp. 351.4)", 350.0 < a["T_az_K"] < 351.8, f"T={a['T_az_K']:.3f}")]


NOTE_ES = " Elige tú el paquete termodinámico adecuado para esta mezcla y nómbralo en las notas."
NOTE_EN = " Choose the thermodynamic property package yourself and name it in the notes."

POLAR_SIM: list[Case] = [
    Case("polar_burbuja_rocio_etanol_agua", "Burbuja y rocío de etanol-agua 50/50", "Ethanol-water 50/50 bubble and dew points",
         "Etanol-agua 50/50 % mol a 1 atm (101325 Pa). Temperatura de burbuja y temperatura de rocío en K." + NOTE_ES,
         "Ethanol-water 50/50 mol % at 1 atm (101325 Pa). Bubble-point and dew-point temperatures in K." + NOTE_EN,
         b_burbuja_rocio, r_burbuja_rocio, c_burbuja_rocio, ["polares", "equilibrio"]),
    Case("polar_flash_metanol_agua", "Flash de metanol-agua a 355 K", "Methanol-water flash at 355 K",
         "100 mol/s de metanol-agua 40/60 % mol a 355 K y 1 atm en un separador flash. Fracción vapor y fracción mol de metanol en vapor y líquido." + NOTE_ES,
         "100 mol/s methanol-water 40/60 mol % at 355 K and 1 atm in a flash separator. Vapor fraction and methanol mole fraction in vapor and liquid." + NOTE_EN,
         b_flash_metanol, r_flash_metanol, c_flash_metanol, ["polares", "flash"]),
    Case("polar_columna_metanol_agua", "Columna metanol-agua", "Methanol-water column",
         "100 mol/s de metanol-agua 40/60 % mol a 340 K y 1 atm, columna de 20 etapas (1 = condensador total), alimentación en la 10, R = 1.5, fondo 60 mol/s, sin caída de presión. Fracción mol de metanol en destilado y fondo, cargas del condensador y rehervidor en kW." + NOTE_ES,
         "100 mol/s methanol-water 40/60 mol % at 340 K and 1 atm, 20-stage column (1 = total condenser), feed at stage 10, R = 1.5, bottoms 60 mol/s, no pressure drop. Methanol mole fraction in distillate and bottoms, condenser and reboiler duties in kW." + NOTE_EN,
         b_columna_metanol, r_columna_metanol, c_columna_metanol, ["polares", "columnas"]),
]

POLAR_DESIGN: list[DesignCase] = [
    DesignCase("polar_azeotropo_etanol_agua", "Azeótropo etanol-agua a 1 atm", "Ethanol-water azeotrope at 1 atm",
               "¿A qué fracción mol de etanol y a qué temperatura (K) hierve el azeótropo etanol-agua a 1 atm (101325 Pa)? Pista: es el mínimo de la curva de burbuja; búscalo con el simulador variando la composición." + NOTE_ES,
               "At what ethanol mole fraction and temperature (K) does the ethanol-water azeotrope boil at 1 atm (101325 Pa)? Hint: it is the minimum of the bubble curve; search it with the simulator by varying composition." + NOTE_EN,
               o_azeotropo, {"x_az_EtOH": {"abs": 0.02}, "T_az_K": {"abs": 0.5}}, {"x_az_EtOH": "-", "T_az_K": "K"}, c_azeotropo, ["polares", "diseno", "equilibrio"]),
]

POLAR_TOL = {"T_bubble_K": {"abs": 0.5}, "T_dew_K": {"abs": 0.5}, "VF": {"abs": 0.02}, "y_MeOH": {"abs": 0.01}, "x_MeOH": {"abs": 0.01},
             "x_MeOH_dist": {"abs": 0.01}, "x_MeOH_bot": {"abs": 0.01}, "Qc_kW": {"rel": 0.03}, "Qr_kW": {"rel": 0.03}}


def main():
    """Print NRTL (oracle) vs PR for each card: the evidence that the cards grade the package choice."""
    from packages.dwsim_client.verify import verify
    for c in POLAR_SIM:
        print(f"\n## {c.id}")
        for pp in ("NRTL", "PengRobinson"):
            fs, h = c.build(pp)
            rep = verify(fs)
            r = {k: round(float(v), 4) for k, v in c.readouts(h).items()} if rep.solved else {}
            chk = [(n, ok) for n, ok, _ in c.checks(h, rep)] if rep.solved else []
            print(f"- {pp:13s} solved={rep.solved} ok={rep.ok} {r} checks={[n for n, ok in chk if not ok] or 'all ok'}")
    for d in POLAR_DESIGN:
        print(f"\n## {d.id}")
        for pp in ("NRTL", "PengRobinson"):
            try:
                a = d.oracle(pp)
                print(f"- {pp:13s} { {k: round(v, 4) for k, v in a.items()} } fails={[n for n, ok, _ in d.checks(a) if not ok] or 'none'}")
            except Exception as ex:  # noqa: BLE001
                print(f"- {pp:13s} failed: {str(ex)[:120]}")


if __name__ == "__main__":
    main()
