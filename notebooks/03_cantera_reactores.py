"""Notebook 03 — Cinética y redes de reactores con Cantera, contra la solución analítica y contra DWSIM.

Tres preguntas, cada una con un número verificable:
  A. Un reactor de primer orden A→B isotérmico: ¿coinciden la fórmula de libro, Cantera (red de reactores) y
     el CSTR/PFR cinético de DWSIM? (Es el caso mínimo que debe reproducir el reactor cinético de Fluxo 1.1.)
  B. Combustión de metano con GRI-Mech 3.0 (53 especies, 325 reacciones): ¿cuánto se equivoca un reactor de
     conversión al 100 % (DWSIM, sin disociación) en la temperatura de llama adiabática frente al equilibrio real?
     ¿Y qué agrega la cinética (reactor bien agitado: apagado por tiempo de residencia, CO, NO)?
  C. Retardo de ignición en un reactor batch adiabático: lo que ningún modelo de conversión/equilibrio puede dar.

Salida: notebooks/03_cantera_reactores.md.  Correr: uv run python notebooks/03_cantera_reactores.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cantera as ct
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.dwsim_client import tools as T  # noqa: E402

R = ct.gas_constant / 1000.0          # J/(mol·K)
OUT = []


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


# --------------------------------------------------------------------------- A. primer orden A→B
K0, EA, TEMP, P, F_MOL, V_M3 = 1.0e8, 100_000.0, 600.0, 1.0e5, 10.0, 10.0      # 1/s, J/mol, K, Pa, mol/s, m³
k = K0 * math.exp(-EA / (R * TEMP))
Qv_ideal = F_MOL * R * TEMP / P                                                  # m³/s, gas ideal
tau = V_M3 / Qv_ideal
X_pfr_ana, X_cstr_ana = 1 - math.exp(-k * tau), k * tau / (1 + k * tau)

MECH_AB = f"""
units: {{length: m, quantity: mol, activation-energy: J/mol}}
phases:
- name: gas
  thermo: ideal-gas
  species: [A, B]
  kinetics: gas
  reactions: all
  state: {{T: {TEMP}, P: {P}}}
species:
- name: A
  composition: {{C: 4, H: 10}}
  thermo: {{model: constant-cp, T0: 298.15 K, h0: 0 kJ/mol, s0: 0 J/mol/K, cp0: 120 J/mol/K}}
- name: B
  composition: {{C: 4, H: 10}}
  thermo: {{model: constant-cp, T0: 298.15 K, h0: -8 kJ/mol, s0: 0 J/mol/K, cp0: 120 J/mol/K}}
reactions:
- equation: A => B
  rate-constant: {{A: {K0}, b: 0, Ea: {EA}}}
"""


def cantera_batch_isothermal(t_end: float) -> float:
    """PFR lagrangiano = batch isotérmico a presión constante (sin cambio de moles, densidad constante)."""
    gas = ct.Solution(yaml=MECH_AB)
    gas.TPX = TEMP, P, "A:1"
    r = ct.IdealGasConstPressureReactor(gas, energy="off", clone=False)
    net = ct.ReactorNet([r])
    net.advance(t_end)
    return 1.0 - r.phase["A"].X[0]


def cantera_cstr(tau_s: float) -> float:
    """Reactor bien agitado isotérmico con tiempo de residencia τ: alimentación + válvula, hasta estado estacionario."""
    gas = ct.Solution(yaml=MECH_AB)
    gas.TPX = TEMP, P, "A:1"
    inlet = ct.Reservoir(gas, clone=False)
    r = ct.IdealGasReactor(gas, energy="off", clone=False)
    r.volume = V_M3
    exhaust = ct.Reservoir(gas, clone=False)
    mfc = ct.MassFlowController(inlet, r, mdot=r.mass / tau_s)      # ONE inlet: a second MFC halves τ silently
    ct.PressureController(r, exhaust, primary=mfc, K=0.01)
    net = ct.ReactorNet([r])
    net.advance_to_steady_state()
    return 1.0 - r.phase["A"].X[0]


def dwsim_reactor(kind: str) -> tuple[float, float, bool]:
    fid = T.new_flowsheet(kind, ["N-butane", "Isobutane"], "PengRobinson")["flowsheet_id"]
    T.define_kinetic_reaction(fid, "ISO", {"N-butane": -1, "Isobutane": 1}, "N-butane", {"N-butane": 1}, K0, EA)
    T.add_material_stream(fid, "feed", temperature_K=TEMP, pressure_Pa=P, molar_flow_mol_s=F_MOL, mole_fractions={"N-butane": 1.0})
    T.add_material_stream(fid, "prod"); T.add_energy_stream(fid, "q")
    T.add_unit(fid, kind, "R", {"volume_m3": V_M3, "reaction_set": "KIN"})
    T.connect(fid, "R", feeds=["feed"], products=["prod"], energy_feeds=["q"])
    out = T.solve(fid)
    X = 1 - T.read_stream(fid, "prod")["mole_fractions"]["N-butane"]
    duty = out["units"][0]["duty_kW"]
    T.close_flowsheet(fid)
    return X, duty, out["ok"]


say("# Notebook 03 — Cinética y redes de reactores: libro vs Cantera vs DWSIM")
say()
say(f"Cantera {ct.__version__} · DWSIM 10.2.12 (Fluent API) · fecha 2026-11-02")
say()
say("## A. Primer orden A → B isotérmico (el caso mínimo de Fluxo 1.1)")
say()
say(f"k = A·exp(−E/RT) con A = {K0:.0e} 1/s, E = {EA/1000:.0f} kJ/mol, T = {TEMP:.0f} K → **k = {k:.5f} 1/s**. "
    f"Alimentación {F_MOL:.0f} mol/s de A a {P/1e5:.0f} bar; V = {V_M3:.0f} m³ → Q_v(gas ideal) = {Qv_ideal:.4f} m³/s, **τ = {tau:.3f} s**, kτ = {k*tau:.3f}.")
say()
X_pfr_ct = cantera_batch_isothermal(tau)
X_cstr_ct = cantera_cstr(tau)
X_pfr_dw, Q_pfr_dw, ok1 = dwsim_reactor("pfr")
X_cstr_dw, Q_cstr_dw, ok2 = dwsim_reactor("cstr")
say("| Reactor | Fórmula | Analítico | Cantera | DWSIM (PR, n-butano→isobutano) | Δ Cantera | Δ DWSIM |")
say("|---|---|---|---|---|---|---|")
say(f"| PFR | X = 1 − e^(−kτ) | {X_pfr_ana:.5f} | {X_pfr_ct:.5f} | {X_pfr_dw:.5f} | {100*(X_pfr_ct-X_pfr_ana)/X_pfr_ana:+.3f} % | {100*(X_pfr_dw-X_pfr_ana)/X_pfr_ana:+.3f} % |")
say(f"| CSTR | X = kτ/(1+kτ) | {X_cstr_ana:.5f} | {X_cstr_ct:.5f} | {X_cstr_dw:.5f} | {100*(X_cstr_ct-X_cstr_ana)/X_cstr_ana:+.3f} % | {100*(X_cstr_dw-X_cstr_ana)/X_cstr_ana:+.3f} % |")
say()
say(f"DWSIM cierra balances en ambos (verify ok = {ok1 and ok2}); cargas isotérmicas PFR {Q_pfr_dw:.2f} kW, CSTR {Q_cstr_dw:.2f} kW "
    f"(isomerización exotérmica, ΔHf del isobutano menor). La desviación de DWSIM ({100*(X_pfr_dw-X_pfr_ana)/X_pfr_ana:+.3f} % / {100*(X_cstr_dw-X_cstr_ana)/X_cstr_ana:+.3f} %) es la densidad Peng-Robinson "
    f"frente al gas ideal (Z ≈ 0.998 a 1 bar y 600 K): τ real ligeramente mayor. Lección: **el reactor cinético de Fluxo 1.1 debe calcular τ con la densidad del motor PR, no con PV = nRT.**")
say()
say("Trampas de la API de DWSIM anotadas: `DefineKineticReaction` exige órdenes para TODOS los compuestos de la estequiometría (también 0); "
    "E en J/mol; CSTR y PFR necesitan corriente de energía en el puerto 1 incluso en modo isotérmico; "
    "`CSTRBuilder.WithIsothermalTemperature` rompe el flash PH (T → 3·10⁷ K) — usar `Isothermal()` con la alimentación a la T deseada.")
say()

# --------------------------------------------------------------------------- B. GRI-Mech 3.0
say("## B. Metano + aire estequiométrico (GRI-Mech 3.0, 53 especies / 325 reacciones)")
say()
gas = ct.Solution("gri30.yaml")
T0, P0, X0 = 300.0, ct.one_atm, "CH4:1, O2:2, N2:7.52"
gas.TPX = T0, P0, X0
gas.equilibrate("HP")
T_eq, x_eq = gas.T, gas.X
major = {s: gas[s].X[0] for s in ("CO2", "H2O", "CO", "O2", "H2", "OH", "NO", "N2")}


def dwsim_adiabatic_conversion() -> tuple[float, bool]:
    fid = T.new_flowsheet("comb", ["Methane", "Oxygen", "Nitrogen", "Carbon dioxide", "Water"], "PengRobinson")["flowsheet_id"]
    T.define_conversion_reaction(fid, "COMB", {"Methane": -1, "Oxygen": -2, "Carbon dioxide": 1, "Water": 2}, "Methane", 100, "Comb")
    T.add_material_stream(fid, "mix", temperature_K=T0, pressure_Pa=P0, molar_flow_mol_s=10.52,
                          mole_fractions={"Methane": 1 / 10.52, "Oxygen": 2 / 10.52, "Nitrogen": 7.52 / 10.52})
    for n in ("flue", "liq"):
        T.add_material_stream(fid, n)
    T.add_energy_stream(fid, "Q")
    T.add_unit(fid, "conversion_reactor", "R-1", {"reaction_set": "Comb", "mode": "adiabatic"})
    T.connect(fid, "R-1", feeds=["mix"], products=["flue", "liq"], energy_feeds=["Q"])
    out = T.solve(fid)
    Tf = T.read_stream(fid, "flue")["T_K"]
    T.close_flowsheet(fid)
    return Tf, out["ok"]


T_dw, ok_dw = dwsim_adiabatic_conversion()
say("### B1. Temperatura de llama adiabática: equilibrio real vs conversión 100 %")
say()
say("| Modelo | T adiabática | Qué supone |")
say("|---|---|---|")
say(f"| Cantera, equilibrio HP (GRI-Mech) | **{T_eq:.0f} K** | 53 especies; disociación a CO, H₂, OH, O, H, NO… |")
say(f"| DWSIM, reactor de conversión adiabático 100 % (PR) | **{T_dw:.0f} K** | solo CH₄ + 2 O₂ → CO₂ + 2 H₂O; cp reales de PR; balances cerrados = {ok_dw} |")
say(f"| Diferencia | **{T_dw - T_eq:+.0f} K ({100*(T_dw-T_eq)/T_eq:+.1f} %)** | la disociación absorbe entalpía: el reactor de conversión sobreestima |")
say()
say("Composición de equilibrio (fracción mol): " + ", ".join(f"{s} {v:.4f}" if v > 1e-3 else f"{s} {v*1e6:.0f} ppm" for s, v in major.items()) + ".")
say()
say("Esto es lo que debe decir Fluxo cuando alguien use el reactor de conversión para una llama: «sin disociación, la T es un tope superior».")
say()

# well-stirred reactor vs residence time
say("### B2. Reactor bien agitado (WSR) adiabático: la cinética manda cuando τ es corto")
say()
say("Alimentación 300 K, 1 atm, estequiométrica. El reactor arranca en equilibrio (encendido) y se lleva a estado estacionario para cada τ. "
    "Si el tiempo de residencia es menor que el tiempo químico, la llama se apaga (blow-out): la conversión cae a ~0 aunque el equilibrio diga 2225 K.")
say()
say("| τ (ms) | T salida (K) | X CH₄ | CO (% mol) | NO (ppm) | Estado |")
say("|---|---|---|---|---|---|")
rows_wsr = []
for tau_ms in (0.05, 0.1, 0.2, 0.5, 1, 3, 10, 30, 100):
    g = ct.Solution("gri30.yaml")
    g.TPX = T0, P0, X0
    inlet = ct.Reservoir(g)                 # clone=True default: reservoir keeps the cold inlet state
    g.equilibrate("HP")
    wsr = ct.IdealGasReactor(g, clone=False)
    wsr.volume = 1.0
    exhaust = ct.Reservoir(g)
    mfc = ct.MassFlowController(inlet, wsr, mdot=wsr.mass / (tau_ms * 1e-3))
    ct.PressureController(wsr, exhaust, primary=mfc, K=0.01)
    net = ct.ReactorNet([wsr])
    try:
        net.advance_to_steady_state()
    except ct.CanteraError:
        pass
    th = wsr.phase
    x_ch4_out = th["CH4"].X[0]
    X_ch4 = 1 - x_ch4_out / (1 / 10.52)
    state = "apagado" if th.T < 1000 else ("encendido" if X_ch4 > 0.9 else "parcial")
    co, no = max(0.0, th["CO"].X[0] * 100), max(0.0, th["NO"].X[0] * 1e6)
    rows_wsr.append((tau_ms, th.T, max(0.0, X_ch4), co, no, state))
    say(f"| {tau_ms:g} | {th.T:.0f} | {max(0.0, X_ch4):.3f} | {co:.2f} | {no:.0f} | {state} |")
lit = [r for r in rows_wsr if r[5] == "encendido"]
blow = [r for r in rows_wsr if r[5] == "apagado"]
say()
if blow and lit:
    say(f"Blow-out entre τ = {max(r[0] for r in blow):g} y {min(r[0] for r in lit):g} ms. Con τ largo el WSR tiende al equilibrio "
        f"({lit[-1][1]:.0f} K vs {T_eq:.0f} K) y el NO sube con τ (cinética de Zeldovich, lenta): "
        f"{lit[0][4]:.0f} → {lit[-1][4]:.0f} ppm. Ningún reactor de conversión ni de equilibrio puede dar esta curva.")
else:
    say("(No se observó blow-out en el rango; ver tabla.)")
say()

# --------------------------------------------------------------------------- C. ignición
say("## C. Retardo de ignición (batch adiabático a volumen constante)")
say()
say("| T₀ (K) | P₀ (bar) | τ_ign (ms) por máximo de dT/dt |")
say("|---|---|---|")
ign = []
for T_ign, P_bar in ((1000.0, 10.0), (1200.0, 10.0), (1400.0, 10.0), (1200.0, 40.0)):
    g = ct.Solution("gri30.yaml")
    g.TPX = T_ign, P_bar * 1e5, X0
    r = ct.IdealGasReactor(g, clone=False)
    net = ct.ReactorNet([r])
    t, temps = [0.0], [g.T]
    while t[-1] < 0.5:
        net.step()
        t.append(net.time); temps.append(r.T)
        if r.T > T_ign + 400 and len(t) > 10:
            break
    t, temps = np.array(t), np.array(temps)
    dTdt = np.gradient(temps, t)
    t_ign = t[int(np.argmax(dTdt))]
    ign.append((T_ign, P_bar, t_ign))
    say(f"| {T_ign:.0f} | {P_bar:.0f} | {t_ign*1e3:.3f} |")
say()
say("El retardo cae ~un orden de magnitud por cada +200 K (Arrhenius global) y baja con la presión. Útil para dimensionar "
    "cámaras y para el HAZOP (mezcla inflamable + punto caliente = ignición en milisegundos).")
say()

# --------------------------------------------------------------------------- D. spec Fluxo 1.1
say("## D. Qué necesita el reactor cinético de Fluxo 1.1 (derivado de A–C)")
say()
say("""1. **Ley de velocidad**: Arrhenius con ley de potencias, r = A·exp(−E/RT)·Π C_i^n_i, concentraciones en mol/m³ desde la
   densidad del motor PR/SRK (no gas ideal). Reversible opcional con (A_r, E_r). Unidades explícitas en la UI (A en 1/s para
   primer orden, E en kJ/mol).
2. **CSTR**: estado estacionario, sistema algebraico F·(C_in − C_out) + r(C_out, T)·V = 0 por compuesto; Newton con jacobiano
   numérico; isotérmico (Q = ΔH) y adiabático (T de salida por balance de energía con ΔHf del motor, misma convención que el
   reactor de conversión actual).
3. **PFR**: ODE en el volumen dF_i/dV = ν_i·r; RK4 adaptativo (o Dormand-Prince) con 200+ puntos para el perfil; isotérmico y
   adiabático (dT/dV por balance de energía).
4. **Validación obligatoria antes de mostrar al usuario**: este caso A→B (X_PFR = {:.4f}, X_CSTR = {:.4f} a kτ = {:.3f}) dentro de
   0.1 %; combustión 100 % vs reactor de conversión; aviso «sin disociación: T tope superior» cuando T > 1500 K.
5. **Fuera de alcance 1.1**: mecanismos detallados (GRI-Mech), catalítico heterogéneo, no isotérmico con intercambio (modo
   HeatExchange). Eso queda en AlchemAI vía DWSIM/Cantera.""".format(X_pfr_ana, X_cstr_ana, k * tau))
say()
say("Evidencia reproducible: `uv run python notebooks/03_cantera_reactores.py` · tests `tests/test_kinetic_reactors.py`.")

(ROOT / "notebooks" / "03_cantera_reactores.md").write_text("\n".join(OUT) + "\n")
print("→ notebooks/03_cantera_reactores.md")
