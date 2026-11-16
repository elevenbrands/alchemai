"""Boiling-water-reactor (BWR) direct steam cycle and the Laguna Verde nuclear power station from public data (bloque M27_N).

What it answers: with PUBLIC data only (IAEA PRIS via WNA, IAEA/CNSNS/OSTI papers on the Laguna Verde power uprates, CFE data in
a published thermal-discharge study, CENACE public prices, NEI/EUCG and CFE COPAR costs, SEMARNAT grid factor), what the
plant does, its steam cycle (state points, efficiency, heat rate, condenser duty, cooling water), its generation per unit and
year against PRIS, its operating economics, the CO2 it avoids, a ProcessModel (block D draws it) and opportunities.

Cycle (per kg of steam leaving the reactor = feedwater returned, steady state; IAPWS-95 through CoolProp):
    reactor dome (saturated vapour, x = 1, published dome pressure) -> throttle (isenthalpic, to `throttle_pressure_Pa`)
    -> HP turbine (extraction for the top heater, wet exhaust at the crossover pressure, extraction for the next heater)
    -> moisture separator (all liquid removed, drains to the heater at the crossover pressure)
    -> reheater heated by live steam (drains to the top heater) -> LP turbines (extractions for the lower heaters)
    -> condenser cooled by once-through seawater -> feed pump -> N closed feedwater heaters (drains cascaded backwards,
    the lowest drains to the condenser) -> reactor.
    The flows (live steam to the reheater and each extraction) solve a LINEAR system (one heater balance per heater + the
    reheater balance), exact, then the energy balance is checked: q_reactor + w_pump = w_turbine + q_condenser.
Condenser coupled to the sea: T_cond = T_sea + Q_cond / (m_cw cp_sw) + TTD, cp and density of seawater from CoolProp
INCOMP::MITSW at the published salinity; solved with brentq.

Declared simplifications (the plant does not publish them; see `not_found` in the data file):
  * extraction pressures by the classical EQUAL-TEMPERATURE-RISE rule between the design condensing temperature and the
    saturation temperature of the dome, N heaters (default N = 6: the public EPU record names heaters «5 and 6»);
  * crossover (HP exhaust) pressure = extraction pressure of heater N-1; one constant isentropic efficiency for every turbine
    section (the ONE calibrated parameter); generator efficiency folded into it unless given; heater, reheater and
    condenser terminal differences 0 K (thermodynamic limit) unless given; one feed pump from the condenser to the dome
    pressure, isentropic unless given; no steam-line or valve pressure losses unless `throttle_pressure_Pa` is given.
"""
from __future__ import annotations

import json
import math
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

DATA = Path(__file__).parent / "data"
PSI_PA = 6894.757293168          # Pa per psi (NIST SP 811)
FLUID = "Water"                  # IAPWS-95 in CoolProp
HOURS_Y = 8760.0
FLAG = "VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN"
BAL_TOL = 1e-6
PLANTS = ("laguna_verde",)
OVERRIDE_KEYS = {
    "n_feedwater_heaters": "número de calentadores de agua de alimentación (2-9; por defecto 6)",
    "crossover_pressure_Pa": "presión de cruce HP→LP en Pa (por defecto la del calentador N-1)",
    "throttle_pressure_Pa": "presión a la entrada de la turbina en Pa (por defecto = presión de domo, sin pérdidas)",
    "reheat_ttd_K": "diferencia terminal del recalentador del MSR en K (por defecto 0, límite termodinámico)",
    "heater_ttd_K": "diferencia terminal de los calentadores en K (por defecto 0)",
    "condenser_ttd_K": "diferencia entre la temperatura de condensación y la salida del agua de mar en K (por defecto 0)",
    "seawater_inlet_K": "temperatura del agua de mar a la entrada en K (por defecto el promedio de las dos temporadas publicadas)",
    "cooling_flow_m3_s_unit": "gasto de agua de mar por unidad en m3/s (por defecto 63/2 publicado para la central)",
    "generator_efficiency": "eficiencia del generador (fracción; por defecto se incluye en la eficiencia calibrada de la turbina)",
    "pump_efficiency": "eficiencia de la bomba de agua de alimentación (fracción; por defecto 1, isentrópica)",
    "thermal_MWt": "potencia térmica del reactor en MWt (por defecto la licenciada publicada, 2317)",
    "gross_MWe_target": "potencia bruta publicada a la que se calibra la turbina en MWe (por defecto la de PRIS)",
    "heaters_out": "lista de calentadores fuera de servicio (1 = el de menor presión)",
}
PRICE_KEYS = {"energia_mxn_mwh": "MXN/MWh", "energia_usd_mwh": "USD/MWh", "fx_mxn_usd": "MXN/USD",
              "combustible_usd_mwh": "USD/MWh", "operacion_usd_mwh": "USD/MWh", "capital_usd_mwh": "USD/MWh"}

METHOD = ("Ciclo directo de un reactor de agua en ebullición (BWR) con CoolProp (IAPWS-95): vapor saturado del domo → turbina "
          "de alta → separador de humedad y recalentador con vapor vivo → turbinas de baja → condensador con agua de mar de un "
          "paso (CoolProp INCOMP::MITSW) → bomba → N calentadores cerrados con drenes en cascada. Gastos de extracción y "
          "del recalentador por un sistema LINEAL exacto (numpy); acople condensador-mar con brentq; UNA eficiencia "
          "isentrópica de turbina CALIBRADA (brentq) para reproducir la potencia bruta publicada por PRIS a la potencia "
          "térmica licenciada. Generación anual = neta × horas × factor de carga publicado; economía con precios "
          "públicos (CENACE PND, NEI/EUCG, COPAR CFE) como sugerencias marcadas; CO2 evitado con el factor oficial "
          "SEMARNAT/CRE. Python puro, sin simulador.")
NOT_INCLUDED = [
    "Neutrónica, termohidráulica del núcleo, recirculación, seguridad nuclear y protección física: fuera de alcance; sólo "
    "ingeniería general del ciclo de vapor y economía con datos públicos.",
    "Línea de expansión real de la turbina (eficiencia por etapa, humedad, Baumann), purgas del separador interno y del "
    "sistema de limpieza del agua del reactor, sellos y fugas.",
    "Servicios propios detallados: sólo la diferencia bruta − neta publicada por PRIS.",
    "Costo nivelado de la central existente: la inversión original y su financiamiento no son públicos.",
    "Ingresos por potencia, certificados de energías limpias o contratos: sólo energía al PND de la zona Veracruz.",
]


# --------------------------------------------------------------------------------------------------------- properties
def _load(name: str = "laguna_verde.json") -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=4096)
def _p(out: str, a: str, x: float, b: str, y: float) -> float:
    return float(PropsSI(out, a, x, b, y, FLUID))


def tsat(p: float) -> float:
    return _p("T", "P", p, "Q", 1.0)


def psat(t: float) -> float:
    return _p("P", "T", t, "Q", 0.0)


def hf(p: float) -> float:
    return _p("H", "P", p, "Q", 0.0)


def hg(p: float) -> float:
    return _p("H", "P", p, "Q", 1.0)


def _state(p: float, h: float) -> dict:
    t = _p("T", "P", p, "H", h)
    s = _p("S", "P", p, "H", h)
    lo, hi = hf(p), hg(p)
    x = (h - lo) / (hi - lo) if lo <= h <= hi else (None if h < lo else 1.0)
    return {"P_Pa": p, "T_K": t, "h_J_kg": h, "s_J_kgK": s, "x": x}


def _expand(p_in: float, h_in: float, p_out: float, eta: float) -> float:
    s_in = _p("S", "P", p_in, "H", h_in)
    h_s = _p("H", "P", p_out, "S", s_in)
    return h_in - eta * (h_in - h_s)


def seawater(t_k: float, salinity_g_kg: float) -> tuple[float, float]:
    """(cp J/kg/K, density kg/m3) of seawater from CoolProp INCOMP::MITSW (Sharqawy et al. correlations)."""
    fl = f"INCOMP::MITSW[{salinity_g_kg / 1000.0}]"
    return float(PropsSI("C", "T", t_k, "P", 101325.0, fl)), float(PropsSI("D", "T", t_k, "P", 101325.0, fl))


def equal_rise_pressures(t_cond_K: float, p_dome_Pa: float, n: int) -> list[float]:
    """Extraction pressures (ascending) of n heaters by the equal-temperature-rise rule between the condensing temperature
    and the saturation temperature of the dome: T_j = T_c + j (T_sat - T_c) / (n + 1), j = 1..n."""
    ts = tsat(p_dome_Pa)
    return [psat(t_cond_K + j * (ts - t_cond_K) / (n + 1)) for j in range(1, n + 1)]


# ------------------------------------------------------------------------------------------------------------ cycle
def bwr_cycle(p_dome_Pa: float, p_cond_Pa: float, extraction_pressures_Pa: list[float], eta_turbine: float,
              msr: bool = True, crossover_pressure_Pa: float | None = None, throttle_pressure_Pa: float | None = None,
              reheat_ttd_K: float = 0.0, heater_ttd_K: float = 0.0, pump_efficiency: float = 1.0,
              heaters_out: list[int] | None = None) -> dict:
    """Per kg of reactor steam. Heaters are numbered 1..N by ascending extraction pressure. Returns specific works and heats
    (J/kg), the solved flow fractions, the state points and the energy-balance residual."""
    pex = sorted(float(p) for p in extraction_pressures_Pa)
    n = len(pex)
    out_set = set(int(i) for i in (heaters_out or []))
    if any(i < 1 or i > n for i in out_set):
        raise ValueError(f"heaters_out: los calentadores van de 1 a {n}.")
    p_thr = float(throttle_pressure_Pa or p_dome_Pa)
    if not (p_cond_Pa < (pex[0] if n else p_thr) and (not n or pex[-1] < p_thr) and p_thr <= p_dome_Pa):
        raise ValueError("Presiones incongruentes: se requiere condensador < extracciones < entrada de turbina ≤ domo.")
    if not 0.3 <= eta_turbine <= 1.0 or not 0.3 <= pump_efficiency <= 1.0:
        raise ValueError("Las eficiencias de turbina y bomba deben estar entre 0.3 y 1.")
    if msr and n < 2 and crossover_pressure_Pa is None:
        raise ValueError("Con separador-recalentador (msr) se necesitan al menos 2 calentadores o una crossover_pressure_Pa.")
    p_x = None
    if msr:
        p_x = float(crossover_pressure_Pa) if crossover_pressure_Pa else pex[-2]
        if not p_cond_Pa < p_x < p_thr:
            raise ValueError("La presión de cruce debe estar entre la del condensador y la de entrada a la turbina.")
    nu = 1 + n                                   # unknowns: m_rh, m_1..m_n (m_rh = 0 row if no msr)
    dim = 1 + nu

    def const(c: float) -> np.ndarray:
        v = np.zeros(dim)
        v[0] = c
        return v

    def var(i: int) -> np.ndarray:               # i = 0 -> m_rh, i = j -> m_j
        v = np.zeros(dim)
        v[1 + i] = 1.0
        return v

    h_dome = hg(p_dome_Pa)
    h1 = h_dome                                  # isenthalpic throttle
    states = {"reactor_out": _state(p_dome_Pa, h_dome), "turbine_in": _state(p_thr, h1)}
    eqs: list[np.ndarray] = []
    flow = const(1.0) - (var(0) if msr else 0.0)
    w_t = const(0.0)
    h_cur, p_cur = h1, p_thr
    stations = sorted(set(pex + ([p_x] if msr else []) + [p_cond_Pa]), reverse=True)
    ext_h: dict[int, float] = {}
    drains: dict[int, list[tuple[np.ndarray, float]]] = {j: [] for j in range(1, n + 1)}
    to_cond: list[tuple[np.ndarray, float]] = []
    ms_liquid = None
    for p in stations:
        h_new = _expand(p_cur, h_cur, p, eta_turbine)
        w_t = w_t + flow * (h_cur - h_new)
        h_cur, p_cur = h_new, p
        for j in range(1, n + 1):
            if abs(pex[j - 1] - p) <= 1e-9 * p:
                ext_h[j] = h_cur
                states[f"extraction_{j}"] = _state(p, h_cur)
                flow = flow - var(j)
        if msr and abs(p - p_x) <= 1e-9 * p:
            states["hp_exhaust"] = _state(p, h_cur)
            x = states["hp_exhaust"]["x"]
            x = 1.0 if x is None else x
            ms_liquid = (flow * (1.0 - x), hf(p))
            vap = flow * x
            t_rh = tsat(p_thr) - reheat_ttd_K
            if t_rh <= tsat(p) + 1e-6:
                raise ValueError("reheat_ttd_K deja el recalentado por debajo de la saturación de la presión de cruce.")
            h_rh = _p("H", "P", p, "T", t_rh)
            eqs.append(vap * (h_rh - hg(p)) - var(0) * (h1 - hf(p_thr)))
            states["ms_vapour"] = _state(p, hg(p))
            states["reheater_out"] = _state(p, h_rh)
            h_cur = h_rh
            flow = vap
    h_exh = h_cur
    states["lp_exhaust"] = _state(p_cond_Pa, h_exh)
    exhaust_flow = flow
    # extra drains: reheater drains to the top active heater, moisture-separator liquid to the heater at/below crossover
    active = [j for j in range(1, n + 1) if j not in out_set]
    if msr:
        rh_drain = (var(0), hf(p_thr))
        (drains[active[-1]] if active else to_cond).append(rh_drain)
        below = [j for j in active if pex[j - 1] <= p_x * (1 + 1e-9)]
        (drains[below[-1]] if below else to_cond).append(ms_liquid)
    else:
        eqs.append(var(0))
    # water side
    h_fc = hf(p_cond_Pa)
    s_fc = _p("S", "P", p_cond_Pa, "Q", 0.0)
    h_ps = _p("H", "P", p_dome_Pa, "S", s_fc)
    h_pump = h_fc + (h_ps - h_fc) / pump_efficiency
    w_p = h_pump - h_fc
    states["condensate"] = _state(p_cond_Pa, h_fc)
    states["pump_out"] = _state(p_dome_Pa, h_pump)
    h_w = [h_pump]
    for j in range(1, n + 1):
        if j in out_set:
            h_w.append(h_w[-1])
        else:
            hj = _p("H", "P", p_dome_Pa, "T", tsat(pex[j - 1]) - heater_ttd_K)
            h_w.append(max(hj, h_w[-1]))
        states[f"feedwater_after_{j}"] = _state(p_dome_Pa, h_w[-1])
    for j in range(n, 0, -1):
        inc = drains[j]
        down = drains[j - 1] if j > 1 else to_cond
        if j in out_set:
            eqs.append(var(j))
            down.extend(inc)
            continue
        hd = hf(pex[j - 1])
        e = var(j) * (ext_h[j] - hd) - const(h_w[j] - h_w[j - 1])
        d = var(j)
        for f, h in inc:
            e = e + f * (h - hd)
            d = d + f
        eqs.append(e)
        down.append((d, hd))
    a = np.array([e[1:] for e in eqs])
    b = -np.array([e[0] for e in eqs])
    sol = np.linalg.solve(a, b)
    xv = np.concatenate([[1.0], sol])

    def ev(v) -> float:
        return float(np.dot(v, xv)) if isinstance(v, np.ndarray) else float(v)

    m_rh = float(sol[0])
    m_ext = {j: float(sol[j]) for j in range(1, n + 1)}
    q_cond = ev(exhaust_flow) * (h_exh - h_fc) + sum(ev(f) * (h - h_fc) for f, h in to_cond)
    mass_to_cond = ev(exhaust_flow) + sum(ev(f) for f, _ in to_cond)
    w_turb = ev(w_t)
    q_in = h_dome - h_w[-1]
    resid = (q_in + w_p - w_turb - q_cond) / q_in
    states["feedwater_to_reactor"] = _state(p_dome_Pa, h_w[-1])
    neg = [k for k, v in {"m_rh": m_rh, **{f"m_{j}": v for j, v in m_ext.items()}}.items() if v < -1e-9]
    return {"w_turbine_J_kg": w_turb, "w_pump_J_kg": w_p, "q_in_J_kg": q_in, "q_cond_J_kg": q_cond,
            "efficiency_cycle": (w_turb - w_p) / q_in, "efficiency_turbine_gross": w_turb / q_in,
            "m_reheater_frac": m_rh, "m_extraction_frac": m_ext, "lp_exhaust_frac": ev(exhaust_flow),
            "ms_liquid_frac": ev(ms_liquid[0]) if ms_liquid else 0.0, "mass_to_condenser_frac": mass_to_cond,
            "extraction_pressures_Pa": pex, "crossover_pressure_Pa": p_x, "throttle_pressure_Pa": p_thr,
            "p_cond_Pa": p_cond_Pa, "feedwater_T_K": states["feedwater_to_reactor"]["T_K"], "states": states,
            "energy_residual_rel": resid, "mass_residual": mass_to_cond - 1.0, "negative_flows": neg}


def cycle_with_sea(thermal_W: float, p_dome_Pa: float, extraction_pressures_Pa: list[float], eta_turbine: float,
                   sea_in_K: float, cooling_flow_m3_s: float, salinity_g_kg: float, condenser_ttd_K: float = 0.0,
                   generator_efficiency: float = 1.0, **kw) -> dict:
    """Cycle coupled to a once-through seawater condenser: solves T_cond = T_in + Q_cond/(m_cw cp) + TTD (brentq)."""
    cp, rho = seawater(sea_in_K, salinity_g_kg)
    m_cw = cooling_flow_m3_s * rho
    p_low = min(extraction_pressures_Pa) if extraction_pressures_Pa else p_dome_Pa

    def run(tc: float) -> dict:
        return bwr_cycle(p_dome_Pa, psat(tc), extraction_pressures_Pa, eta_turbine, **kw)

    def g(tc: float) -> float:
        c = run(tc)
        m_s = thermal_W / c["q_in_J_kg"]
        return sea_in_K + m_s * c["q_cond_J_kg"] / (m_cw * cp) + condenser_ttd_K - tc

    lo, hi = sea_in_K + condenser_ttd_K + 0.01, min(sea_in_K + 80.0, tsat(p_low) - 0.5)
    if g(hi) > 0:
        raise ValueError("El agua de mar no alcanza a condensar el vapor con ese gasto: revisa cooling_flow_m3_s_unit.")
    tc = brentq(g, lo, hi, xtol=1e-9)
    c = run(tc)
    m_s = thermal_W / c["q_in_J_kg"]
    q_cond = m_s * c["q_cond_J_kg"]
    gross = generator_efficiency * m_s * c["w_turbine_J_kg"]
    c.update({"thermal_W": thermal_W, "steam_flow_kg_s": m_s, "gross_W": gross, "turbine_shaft_W": m_s * c["w_turbine_J_kg"],
              "feed_pump_W": m_s * c["w_pump_J_kg"], "condenser_duty_W": q_cond, "T_cond_K": tc,
              "sea_in_K": sea_in_K, "sea_out_K": sea_in_K + q_cond / (m_cw * cp), "cooling_flow_kg_s": m_cw,
              "cooling_flow_m3_s": cooling_flow_m3_s, "seawater_cp_J_kgK": cp, "seawater_density_kg_m3": rho,
              "gross_efficiency": gross / thermal_W, "heat_rate_kJ_kWh": 3600.0 * thermal_W / gross,
              "generator_efficiency": generator_efficiency,
              "plant_balance_residual_rel": (thermal_W + m_s * c["w_pump_J_kg"] - m_s * c["w_turbine_J_kg"] - q_cond)
              / thermal_W})
    return c


def calibrate_eta(target_gross_W: float, thermal_W: float, **kw) -> tuple[float, dict]:
    """The one free parameter: the isentropic efficiency (all turbine sections) that gives the published gross output."""
    def f(eta: float) -> float:
        return cycle_with_sea(thermal_W, eta_turbine=eta, **kw)["gross_W"] - target_gross_W

    lo, hi = 0.40, 1.0
    flo, fhi = f(lo), f(hi)
    if flo > 0 or fhi < 0:
        raise ValueError(f"No hay eficiencia de turbina en [{lo}, {hi}] que dé {target_gross_W / 1e6:.1f} MWe brutos con "
                         f"{thermal_W / 1e6:.0f} MWt (rango del modelo {(flo + target_gross_W) / 1e6:.1f}–"
                         f"{(fhi + target_gross_W) / 1e6:.1f} MWe).")
    eta = brentq(f, lo, hi, xtol=1e-12)
    return eta, cycle_with_sea(thermal_W, eta_turbine=eta, **kw)


def rankine_example() -> dict:
    """The published ideal Rankine example (saturated steam 6 MPa, condenser 0.008 MPa) through the same engine."""
    ex = _load()["rankine_example"]
    c = bwr_cycle(ex["P_high_Pa"], ex["P_low_Pa"], [], 1.0, msr=False)
    return {"published": ex, "model": {"x4": c["states"]["lp_exhaust"]["x"], "w_turbine_kJ_kg": c["w_turbine_J_kg"] / 1e3,
                                       "w_pump_kJ_kg": c["w_pump_J_kg"] / 1e3, "q_in_kJ_kg": c["q_in_J_kg"] / 1e3,
                                       "efficiency": c["efficiency_cycle"]},
            "energy_residual_rel": c["energy_residual_rel"]}


# ----------------------------------------------------------------------------------------------------- plant runs
def _ov(overrides: dict | None) -> dict:
    ov = dict(overrides or {})
    for k in ov:
        if k not in OVERRIDE_KEYS:
            raise ValueError(f"overrides: clave desconocida {k!r}; claves válidas: " +
                             "; ".join(f"{a} = {b}" for a, b in OVERRIDE_KEYS.items()))
    return ov


def _design(d: dict, ov: dict) -> dict:
    seasons = d["cooling"]["seasons"]
    n = int(ov.get("n_feedwater_heaters", 6))
    if not 2 <= n <= 9:
        raise ValueError("n_feedwater_heaters debe estar entre 2 y 9.")
    p_dome = d["steam"]["dome_pressure_psia"] * PSI_PA
    t_dis = [s["discharge_T_C"] + 273.15 for s in seasons.values()]
    t_sea = [s["sea_T_C"] + 273.15 for s in seasons.values()]
    t_cd = sum(t_dis) / len(t_dis) + float(ov.get("condenser_ttd_K", 0.0))
    pex = equal_rise_pressures(t_cd, p_dome, n)
    kw = {"p_dome_Pa": p_dome, "extraction_pressures_Pa": pex,
          "sea_in_K": float(ov.get("seawater_inlet_K", sum(t_sea) / len(t_sea))),
          "cooling_flow_m3_s": float(ov.get("cooling_flow_m3_s_unit", d["cooling"]["flow_m3_s_plant"] / 2.0)),
          "salinity_g_kg": seasons["lluvias"]["salinity_g_kg"], "condenser_ttd_K": float(ov.get("condenser_ttd_K", 0.0)),
          "generator_efficiency": float(ov.get("generator_efficiency", 1.0)), "msr": bool(d["turbine_cycle"]["has_msr"]),
          "crossover_pressure_Pa": ov.get("crossover_pressure_Pa"), "throttle_pressure_Pa": ov.get("throttle_pressure_Pa"),
          "reheat_ttd_K": float(ov.get("reheat_ttd_K", 0.0)), "heater_ttd_K": float(ov.get("heater_ttd_K", 0.0)),
          "pump_efficiency": float(ov.get("pump_efficiency", 1.0)), "heaters_out": list(ov.get("heaters_out") or [])}
    return {"kw": kw, "n": n, "T_cond_design_K": t_cd, "p_dome_Pa": p_dome}


def _rows(c: dict) -> list[dict]:
    labels = {"reactor_out": "Salida del reactor (vapor saturado del domo)", "turbine_in": "Entrada a turbina de alta",
              "hp_exhaust": "Escape de turbina de alta (húmedo)", "ms_vapour": "Vapor seco del separador de humedad",
              "reheater_out": "Salida del recalentador (entrada a baja)", "lp_exhaust": "Escape de turbinas de baja",
              "condensate": "Condensado (líquido saturado)", "pump_out": "Descarga de la bomba de agua de alimentación",
              "feedwater_to_reactor": "Agua de alimentación al reactor"}
    rows = []
    for k, s in c["states"].items():
        lab = labels.get(k) or (f"Extracción al calentador {k.split('_')[1]}" if k.startswith("extraction") else
                                f"Agua de alimentación tras el calentador {k.split('_')[-1]}")
        rows.append({"id": k, "label_es": lab, **s})
    return rows


def _unit_list(unit) -> list[str]:
    d = _load()
    if unit is None:
        return sorted(d["units"])
    u = str(unit)
    if u not in d["units"]:
        raise ValueError(f"unit = {unit!r}: Laguna Verde tiene las unidades {sorted(d['units'])}.")
    return [u]


def _prices(d: dict, year: int, prices: dict | None, today: date, warnings: list, assumptions: list) -> dict:
    from packages.costing import reference_prices as RP
    prices = dict(prices or {})
    for k, v in prices.items():
        if k not in PRICE_KEYS:
            raise ValueError(f"prices: clave desconocida {k!r}; usa {PRICE_KEYS}.")
        if v is None or not math.isfinite(float(v)) or float(v) < 0:
            raise ValueError(f"prices[{k!r}] debe ser un número ≥ 0 en {PRICE_KEYS[k]}.")
    src = d["sources"]
    table: dict = {}

    def put(key, value, unit, sid, period, tier, basis):
        s = src.get(sid, {}) if sid else {}
        table[key] = {"value": value, "unit": unit, "source": s.get("what", "dato del usuario") if sid else "dato del usuario",
                      "url": s.get("url"), "period": period, "source_tier": tier, "basis": basis,
                      "flag": FLAG if sid else "dato del usuario", "date_read": d["date_read"] if sid else None}

    if "fx_mxn_usd" in prices:
        put("fx_mxn_usd", float(prices["fx_mxn_usd"]), "MXN/USD", None, None, "usuario", "tipo de cambio del usuario")
    else:
        fx = RP.lookup("fx_fix_mxn_usd", "MX", today=today)
        if fx.get("found"):
            table["fx_mxn_usd"] = {"value": fx["value"], "unit": "MXN/USD", "source": fx["source"], "url": fx["url"],
                                   "period": fx["period"], "source_tier": "primaria", "basis": "FIX del banco de referencia",
                                   "flag": FLAG, "date_read": fx.get("date_read")}
            warnings.append(f"El precio de la energía (MXN) se pasa a USD con el FIX {fx['value']} MXN/USD del "
                            f"{fx['period']['end']}, no con el promedio del año {year}: da prices['fx_mxn_usd'] si lo tienes.")
    mp = d["market_price"]
    if "energia_usd_mwh" in prices:
        put("energia_usd_mwh", float(prices["energia_usd_mwh"]), "USD/MWh", None, None, "usuario", "precio del usuario")
    else:
        if "energia_mxn_mwh" in prices:
            put("energia_mxn_mwh", float(prices["energia_mxn_mwh"]), "MXN/MWh", None, None, "usuario", "precio del usuario")
        elif str(year) in mp["by_year"]:
            y = mp["by_year"][str(year)]
            put("energia_mxn_mwh", y["value"], "MXN/MWh", mp["source"], {"start": y["start"], "end": y["end"]}, "primaria",
                f"{mp['statistic']} de la zona {mp['zone']}, {mp['market']}, {y['n_hours']} h; {y['note']}. {mp['caveat']}")
        if "energia_mxn_mwh" in table and "fx_mxn_usd" in table:
            e = table["energia_mxn_mwh"]
            table["energia_usd_mwh"] = {**e, "value": e["value"] / table["fx_mxn_usd"]["value"], "unit": "USD/MWh",
                                        "basis": e["basis"] + " Convertido con fx_mxn_usd de esta tabla."}
    nei = d["costs"]["nei_2025_bwr_usd_mwh"]
    for key, part in (("combustible_usd_mwh", "fuel"), ("operacion_usd_mwh", "operating"), ("capital_usd_mwh", "capital")):
        if key in prices:
            put(key, float(prices[key]), "USD/MWh", None, None, "usuario", "costo del usuario")
        else:
            put(key, nei[part], "USD/MWh", nei["source"], "2025", "secundaria (flota de EE. UU.)",
                f"NEI/EUCG 2025, promedio de {nei['n_plants']} centrales BWR de EE. UU. ({nei['dollars']}); no es el costo "
                f"de la CNLV, que CFE no publica")
    for k, t in table.items():
        if t["source"] != "dato del usuario":
            assumptions.append(f"Precio {k} = {t['value']:.4g} {t['unit']} ({t['basis']}; {t['url']}) — {FLAG}.")
    if "energia_usd_mwh" not in table:
        warnings.append(f"No hay precio público cargado para {year} (sólo {sorted(mp['by_year'])}): da prices['energia_usd_mwh'] "
                        "o prices['energia_mxn_mwh'] para calcular ingresos y margen.")
    return table


def _generation(d: dict, units: list[str], year: int, net_model: dict) -> dict:
    cols = d["pris_annual"]["columns"]
    out = {}
    for u in units:
        row = d["pris_annual"][u].get(str(year))
        if row is None:
            out[u] = {"year": year, "published": None, "note_es": f"PRIS no tiene {year} para la unidad {u}"}
            continue
        r = dict(zip(cols, row))
        hours = 8784.0 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else HOURS_Y
        model_gwh = r["reference_unit_power_MWe"] * hours * r["load_factor_pct"] / 100.0 / 1000.0
        out[u] = {"year": year, "hours": hours, "published_GWh": r["electricity_supplied_GWh"],
                  "reference_unit_power_MWe": r["reference_unit_power_MWe"], "load_factor_pct": r["load_factor_pct"],
                  "energy_availability_factor_pct": r["energy_availability_factor_pct"], "time_online_h": r["time_online_h"],
                  "model_GWh": model_gwh, "deviation_rel": (model_gwh - r["electricity_supplied_GWh"]) /
                  r["electricity_supplied_GWh"],
                  "net_MWe_used": r["reference_unit_power_MWe"],
                  "note_es": "modelo = potencia de referencia (neta) × horas × factor de carga publicado: identidad de la "
                             "definición de PRIS (consistencia, no validación independiente)",
                  "gross_MWe_model": net_model.get(u)}
    return out


def _economics(d, units, gen, table, year, warnings) -> dict:
    mwh = sum(g["published_GWh"] * 1000.0 for g in gen.values() if g.get("published_GWh"))
    if not mwh:
        return {"available": False, "reason_es": f"sin generación publicada para {year}"}
    c_f, c_o, c_k = (table[k]["value"] for k in ("combustible_usd_mwh", "operacion_usd_mwh", "capital_usd_mwh"))
    out = {"year": year, "net_generation_MWh": mwh, "fuel_usd_mwh": c_f, "operating_usd_mwh": c_o, "capital_usd_mwh": c_k,
           "total_generating_cost_usd_mwh": c_f + c_o + c_k, "fuel_usd_y": c_f * mwh, "operating_usd_y": c_o * mwh,
           "capital_usd_y": c_k * mwh,
           "prices": {k: v for k, v in table.items()},
           "lcoe_existing_plant": None,
           "lcoe_note_es": "No calculable con datos públicos: CFE no publica la inversión original ni su financiamiento. "
                           "total_generating_cost_usd_mwh es la definición NEI (combustible + operación + capital de "
                           "sostenimiento), no un costo nivelado.",
           "copar_2012_reference": d["costs"]["copar_2012_abwr_new"]}
    p = table.get("energia_usd_mwh", {}).get("value")
    if p is not None:
        rev = p * mwh
        out.update({"price_usd_mwh": p, "revenue_usd_y": rev,
                    "margin_over_fuel_usd_mwh": p - c_f,
                    "operating_margin_usd_mwh": p - c_f - c_o, "operating_margin_usd_y": (p - c_f - c_o) * mwh,
                    "margin_after_capital_usd_mwh": p - c_f - c_o - c_k,
                    "margin_after_capital_usd_y": (p - c_f - c_o - c_k) * mwh})
        hist = d["gross_output_history_MWe"]
        add_mw = (hist["epu_design_at_2317MWt"] - hist["pre_epu_at_2027MWt"]) * 2
        lf = [g["load_factor_pct"] for g in gen.values() if g.get("load_factor_pct")]
        lf_m = sum(lf) / len(lf) / 100.0
        add_mwh = add_mw * HOURS_Y * lf_m
        cost = d["costs"]["epu_project_cost_usd"]["value"]
        val = add_mwh * (p - c_f)
        out["power_uprate"] = {"added_gross_MWe_both_units": add_mw, "added_MWh_y": add_mwh, "load_factor_used": lf_m,
                               "value_usd_y": val, "project_cost_usd": cost,
                               "simple_payback_y": cost / val if val > 0 else None,
                               "basis_es": "aumento bruto de diseño 686.7 → 817.1 MWe por unidad (Iberdrola ENC 2010) × 2 × "
                                           "8760 h × factor de carga del año, valorado a (precio − combustible); costo del "
                                           "proyecto 650 MUSD (WNN 2013). No resta servicios propios adicionales."}
    else:
        out["price_usd_mwh"] = None
    if len(units) == 1:
        warnings.append("Economía de una sola unidad: los costos por MWh son promedios de flota, no de la unidad.")
    return out


def _sea_sensitivity(base_kw: dict, eta: float, thermal_W: float, d: dict) -> list[dict]:
    seasons = d["cooling"]["seasons"]
    cases = [("secas (feb-may), mar publicado", seasons["secas"]["sea_T_C"]),
             ("lluvias (jun-sep), mar publicado", seasons["lluvias"]["sea_T_C"]),
             ("lluvias, toma medida (pluma recirculada)", seasons["lluvias"]["intake_measured_mean_T_C"])]
    out = []
    for lab, tc in cases:
        c = cycle_with_sea(thermal_W, eta_turbine=eta, **{**base_kw, "sea_in_K": tc + 273.15})
        out.append({"case_es": lab, "sea_in_K": tc + 273.15, "T_cond_K": c["T_cond_K"], "p_cond_Pa": c["p_cond_Pa"],
                    "gross_W": c["gross_W"], "gross_efficiency": c["gross_efficiency"]})
    return out


def _opportunities(d, units, year, gen, econ, sea, heater_case, base, cal_target_W) -> list[dict]:
    opps = []
    p = econ.get("price_usd_mwh")
    c_f = econ.get("fuel_usd_mwh")
    tot = econ.get("total_generating_cost_usd_mwh")
    fb = d["fleet_benchmark"]
    ref_cf = fb["global_average_capacity_factor_2024_pct"]
    for u in units:
        g = gen.get(u, {})
        lf = g.get("load_factor_pct")
        if lf is None or lf >= ref_cf:
            continue
        mwh = g["reference_unit_power_MWe"] * g["hours"] * (ref_cf - lf) / 100.0
        val = mwh * (p - c_f) if p is not None else None
        opps.append({"id": f"factor_capacidad:unidad_{u}", "unit_ids": ["reactor", "generador"], "stream_ids": ["e_red"],
                     "category": "reliability",
                     "title_es": f"Unidad {u}: factor de carga {lf:.1f} % en {year} vs {ref_cf:.0f} % promedio mundial",
                     "explanation_es": f"La unidad {u} entregó {g['published_GWh']:.0f} GWh en {year} (PRIS). Llevarla al "
                                       f"promedio mundial de factor de capacidad de 2024 ({ref_cf:.0f} %, WNA) daría "
                                       f"{mwh / 1e3:.0f} GWh más (recargas más cortas, menos salidas forzadas).",
                     "value_usd_y": val, "value_low_usd_y": mwh * (p - tot) if p is not None else None,
                     "value_high_usd_y": val,
                     "basis_es": "MWh = potencia de referencia × horas × (83 % − factor del año); valor alto = MWh × (precio − "
                                 "combustible), bajo = MWh × (precio − costo total NEI)",
                     "assumptions": ["Promedio mundial de TODOS los tipos de reactor (la mediana BWR de PRIS no se pudo leer).",
                                     "El MWh adicional se vende al PND promedio de la zona Veracruz."],
                     "evidence": {"verified": True, "published_GWh": g["published_GWh"], "load_factor_pct": lf,
                                  "benchmark_pct": ref_cf, "benchmark_source": d["sources"]["WNPR_2025"]["url"]},
                     "confidence": "media"})
    if len(sea) >= 3:
        dp = sea[1]["gross_W"] - sea[2]["gross_W"]
        lf = [gen[u]["load_factor_pct"] / 100.0 for u in units if gen.get(u, {}).get("load_factor_pct")]
        hours = 122 * 24.0
        mwh = dp / 1e6 * hours * (sum(lf) / len(lf) if lf else 0.0) * len(units)
        opps.append({"id": "agua_de_mar:recirculacion_pluma", "unit_ids": ["condensador", "bombas_circulacion"],
                     "stream_ids": ["u_mar_entrada", "u_mar_descarga"], "category": "energy",
                     "title_es": "Recirculación de la pluma térmica a la toma en temporada de lluvias",
                     "explanation_es": f"El estudio publicado midió la toma a {sea[2]['sea_in_K'] - 273.15:.1f} °C con el mar "
                                       f"a {sea[1]['sea_in_K'] - 273.15:.0f} °C: la descarga caliente regresa a la toma. "
                                       f"Con el ciclo calibrado eso cuesta {dp / 1e6:.1f} MWe brutos por unidad "
                                       f"(condensador de {sea[1]['p_cond_Pa'] / 1e3:.2f} a {sea[2]['p_cond_Pa'] / 1e3:.2f} "
                                       "kPa). Separar toma y descarga o mejorar la dispersión lo recuperaría.",
                     "value_usd_y": mwh * (p - c_f) if p is not None else None, "value_low_usd_y": None,
                     "value_high_usd_y": None,
                     "basis_es": "ΔP bruta (mar 28 °C vs toma medida 30.02 °C, Ramírez-León et al. 2013, datos CFE de la "
                                 "época de 2027 MWt) × 122 días (jun-sep) × factor de carga × unidades × (precio − combustible)",
                     "assumptions": ["Datos de temperatura previos al cambio de condensador del EPU.",
                                     "Sensibilidad del modelo calibrado con diferencias terminales de 0 K."],
                     "evidence": {"verified": True, "cases": sea}, "confidence": "baja"})
    if heater_case:
        dmw = heater_case["delta_gross_W"]
        val_day = -dmw / 1e6 * 24.0 * (p - c_f) if p is not None else None
        opps.append({"id": "calentador_fuera:superior", "unit_ids": [f"calentador_{heater_case['heater']}"],
                     "stream_ids": ["s_agua_alimentacion"], "category": "energy",
                     "title_es": f"Calentador {heater_case['heater']} fuera de servicio: {-dmw / 1e6:.1f} MWe menos",
                     "explanation_es": "Con el calentador superior fuera, el agua entra más fría al reactor; a potencia "
                                       f"térmica constante el agua de alimentación baja de "
                                       f"{heater_case['T_fw_base_K'] - 273.15:.0f} a {heater_case['T_fw_out_K'] - 273.15:.0f} °C "
                                       "y la eficiencia bruta cae. Ilustrativo: la CNLV no publica el número real de "
                                       "calentadores ni la disponibilidad de cada uno.",
                     "value_usd_y": None, "value_low_usd_y": None, "value_high_usd_y": None,
                     "basis_es": "ILUSTRATIVO (no hay dato de días fuera de servicio): valor por día en evidence",
                     "assumptions": ["Ciclo simplificado de N calentadores por regla de elevación igual."],
                     "evidence": {"verified": True, "usd_per_day": val_day, **heater_case}, "confidence": "baja"})
    hist = d["gross_output_history_MWe"]
    gap = [hist["epu_design_at_2317MWt"] - d["units"][u]["gross_MWe"] for u in units]
    if p is not None and all(x > 0 for x in gap):
        lf = [gen[u]["load_factor_pct"] / 100.0 for u in units if gen.get(u, {}).get("load_factor_pct")]
        mwh = sum(gap) * HOURS_Y * (sum(lf) / len(lf) if lf else 0.0)
        opps.append({"id": "epu:salida_de_diseno", "unit_ids": ["turbina_ap", "turbina_bp", "generador"], "stream_ids": [],
                     "category": "energy",
                     "title_es": f"Bruta publicada (PRIS) {min(gap):.0f}–{max(gap):.0f} MWe por debajo del diseño del EPU",
                     "explanation_es": "El proyecto del EPU (Iberdrola, ENC 2010) declaró 817.1 MWe brutos por unidad; PRIS "
                                       "reporta 805 y 803 MWe. Si la diferencia es real (no sólo una clasificación "
                                       "conservadora), revisar eficiencia de turbinas, vacío del condensador y calentadores.",
                     "value_usd_y": mwh * (p - c_f), "value_low_usd_y": 0.0, "value_high_usd_y": mwh * (p - c_f),
                     "basis_es": "(817.1 − bruta PRIS) × 8760 h × factor de carga × (precio − combustible)",
                     "assumptions": ["La potencia de referencia de PRIS puede ser una clasificación, no la capacidad real."],
                     "evidence": {"verified": True, "gap_MWe": gap}, "confidence": "baja"})
    opps.sort(key=lambda o: -(o["value_usd_y"] or 0.0))
    for i, o in enumerate(opps, 1):
        o["rank"] = i
    return opps


def _types() -> set:
    try:
        from packages.diagram.symbols import EQUIPMENT_TYPES
        return set(EQUIPMENT_TYPES)
    except Exception:
        return set()


def _process_model(d: dict, u: str, c: dict, n_units: int) -> dict:
    types = _types()

    def t(want: str, fallback: str) -> str:
        return want if want in types else fallback

    st = c["states"]
    m = c["steam_flow_kg_s"]
    kgh = 3600.0
    n = len(c["extraction_pressures_Pa"])
    p_x = c["crossover_pressure_Pa"]
    sec = [{"id": "ciclo", "label_es": "Ciclo de vapor directo BWR (una sola sección: el ciclo es un lazo cerrado)"}]
    subsystem = {"nuclear": "Isla nuclear", "turbinas": "Turbogenerador", "condensacion": "Condensación y agua de mar",
                 "regeneracion": "Agua de alimentación (regeneración)"}
    lab_u = f"Unidad {u}" + (f" (×{n_units} idénticas)" if n_units > 1 else "")
    U = [
        {"id": "reactor", "label_es": f"Reactor BWR-5 — {lab_u}", "type": t("nuclear_reactor", "reactor"), "section": "nuclear",
         "params": {"thermal_W": c["thermal_W"], "dome_pressure_Pa": st["reactor_out"]["P_Pa"]},
         "duties_W": {"thermal_W": c["thermal_W"]}, "metrics": {"steam_flow_kg_s": m}},
        {"id": "turbina_ap", "label_es": "Turbina de alta presión", "type": t("steam_turbine", "expander"), "section": "turbinas",
         "params": {"inlet_P_Pa": c["throttle_pressure_Pa"], "exhaust_P_Pa": p_x}, "duties_W": {}, "metrics": {}},
        {"id": "msr", "label_es": "Separadores de humedad y recalentadores (MSR)",
         "type": t("moisture_separator_reheater", "separator_3phase"), "section": "turbinas",
         "params": {"P_Pa": p_x}, "duties_W": {}, "metrics": {"live_steam_frac": c["m_reheater_frac"]}},
        {"id": "turbina_bp", "label_es": "Turbinas de baja presión", "type": t("steam_turbine", "expander"), "section": "turbinas",
         "params": {"exhaust_P_Pa": c["p_cond_Pa"]}, "duties_W": {}, "metrics": {}},
        {"id": "generador", "label_es": "Generador eléctrico", "type": t("generator", "generic"), "section": "turbinas",
         "params": {}, "duties_W": {"gross_W": c["gross_W"]}, "metrics": {"gross_efficiency": c["gross_efficiency"]}},
        {"id": "condensador", "label_es": "Condensador principal", "type": t("condenser", "heat_exchanger"),
         "section": "condensacion", "params": {"P_Pa": c["p_cond_Pa"], "T_K": c["T_cond_K"]},
         "duties_W": {"condenser_W": c["condenser_duty_W"]}, "metrics": {}},
        {"id": "bombas_circulacion", "label_es": "Bombas de agua de circulación (mar)", "type": "pump",
         "section": "condensacion", "params": {"flow_m3_s": c["cooling_flow_m3_s"]}, "duties_W": {}, "metrics": {}},
        {"id": "bomba_alimentacion", "label_es": "Bombas de condensado y de agua de alimentación", "type": "pump",
         "section": "regeneracion", "params": {"discharge_P_Pa": st["pump_out"]["P_Pa"]},
         "duties_W": {"shaft_W": c["feed_pump_W"]}, "metrics": {}},
    ]
    for j in range(1, n + 1):
        U.append({"id": f"calentador_{j}", "label_es": f"Calentador de agua de alimentación {j}", "type": "heat_exchanger",
                  "section": "regeneracion", "params": {"extraction_P_Pa": c["extraction_pressures_Pa"][j - 1]},
                  "duties_W": {}, "metrics": {"extraction_frac": c["m_extraction_frac"][j]}})

    def ms(sid, label, f, to, key, frac, **extra):
        s = st[key]
        return {"id": sid, "label_es": label, "from": f, "to": to, "kind": "material", "flow_kg_h": frac * m * kgh,
                "T_K": s["T_K"], "P_Pa": s["P_Pa"], **extra}

    hp_ext = [j for j in range(1, n + 1) if c["extraction_pressures_Pa"][j - 1] > p_x * (1 + 1e-9)]
    x_ext = [j for j in range(1, n + 1) if abs(c["extraction_pressures_Pa"][j - 1] - p_x) <= 1e-9 * p_x]
    f_ms = 1.0 - c["m_reheater_frac"] - sum(c["m_extraction_frac"][j] for j in hp_ext + x_ext)
    S = [
        ms("s_vapor_principal", "Vapor principal", "reactor", "turbina_ap", "turbine_in", 1.0 - c["m_reheater_frac"],
           from_port="top"),
        ms("s_vapor_recalentador", "Vapor vivo al recalentador", "reactor", "msr", "turbine_in", c["m_reheater_frac"],
           from_port="top"),
        ms("s_escape_ap", "Escape de alta (húmedo)", "turbina_ap", "msr", "hp_exhaust", f_ms),
        ms("s_vapor_recalentado", "Vapor recalentado a baja", "msr", "turbina_bp", "reheater_out",
           f_ms * (1.0 - c["ms_liquid_frac"] / max(f_ms, 1e-12))),
        ms("s_escape_bp", "Escape de baja", "turbina_bp", "condensador", "lp_exhaust", c["lp_exhaust_frac"]),
        ms("s_condensado", "Condensado", "condensador", "bomba_alimentacion", "condensate", 1.0),
    ]
    prev = "bomba_alimentacion"
    for j in range(1, n + 1):
        src = "turbina_ap" if (j in hp_ext or j in x_ext) else "turbina_bp"
        S.append(ms(f"s_extraccion_{j}", f"Extracción {j}", src, f"calentador_{j}", f"extraction_{j}",
                    c["m_extraction_frac"][j]))
        S.append(ms(f"s_agua_{j}", f"Agua de alimentación {j}", prev, f"calentador_{j}",
                    "pump_out" if j == 1 else f"feedwater_after_{j - 1}", 1.0))
        prev = f"calentador_{j}"
    S.append(ms("s_agua_alimentacion", "Agua de alimentación al reactor", prev, "reactor", "feedwater_to_reactor", 1.0))
    S.append({"id": "s_drenes_msr", "label_es": "Drenes del separador y recalentador", "from": "msr",
              "to": f"calentador_{x_ext[0] if x_ext else n}", "kind": "material",
              "flow_kg_h": (c["ms_liquid_frac"] + c["m_reheater_frac"]) * m * kgh,
              "note_es": "drenes del separador al calentador de la presión de cruce y del recalentador al calentador superior"})
    S.append({"id": "s_drenes_cascada", "label_es": "Drenes de calentadores en cascada", "from": "calentador_1",
              "to": "condensador", "kind": "material", "note_es": "cada calentador drena al de menor presión; el 1 al condensador"})
    S.append({"id": "e_eje", "label_es": "Potencia en el eje", "from": "turbina_bp", "to": "generador", "kind": "energy",
              "power_W": c["turbine_shaft_W"]})
    S.append({"id": "e_red", "label_es": "Energía bruta a la red", "from": "generador", "to": None, "kind": "energy",
              "power_W": c["gross_W"]})
    sw = c["cooling_flow_kg_s"] * kgh
    S.append({"id": "u_mar_entrada", "label_es": "Agua de mar (toma)", "from": None, "to": "bombas_circulacion",
              "kind": "utility", "flow_kg_h": sw, "T_K": c["sea_in_K"], "P_Pa": 101325.0})
    S.append({"id": "u_mar_condensador", "label_es": "Agua de mar al condensador", "from": "bombas_circulacion",
              "to": "condensador", "kind": "utility", "flow_kg_h": sw, "T_K": c["sea_in_K"]})
    S.append({"id": "u_mar_descarga", "label_es": "Descarga al mar", "from": "condensador", "to": None, "kind": "utility",
              "flow_kg_h": sw, "T_K": c["sea_out_K"], "P_Pa": 101325.0})
    for x in U:
        x["params"]["subsystem_es"] = subsystem[x["section"]]
        x["section"] = "ciclo"
    return {"name": f"{d['name']} — ciclo de vapor, unidad {u}", "kind": "plant_model", "units": U, "streams": S,
            "sections": sec, "source": "packages.power.nuclear (bloque M27_N): IAEA PRIS (vía WNA), CNSNS/IAEA/OSTI, CFE "
            "(Ramírez-León et al. 2013), CoolProp IAPWS-95"}


def _today(today) -> date:
    if today is None:
        return date.today()
    if isinstance(today, date):
        return today
    try:
        return date.fromisoformat(str(today))
    except ValueError as exc:
        raise ValueError(f"today = {today!r}: usa formato ISO AAAA-MM-DD.") from exc


def run_plant(plant: str = "laguna_verde", year: int | None = None, unit=None, overrides: dict | None = None,
              prices: dict | None = None, today=None) -> dict:
    if plant not in PLANTS:
        raise ValueError(f"plant = {plant!r}: plantas disponibles {list(PLANTS)}.")
    d = _load()
    ov = _ov(overrides)
    units = _unit_list(unit)
    years = sorted({int(y) for u in units for y in d["pris_annual"][u]})
    year = int(year) if year is not None else years[-1]
    if year not in years:
        raise ValueError(f"year = {year}: PRIS tiene {years[0]}–{years[-1]} para Laguna Verde.")
    warnings: list[str] = []
    assumptions: list[str] = []
    des = _design(d, ov)
    kw = des["kw"]
    thermal_W = float(ov.get("thermal_MWt", d["units"][units[0]]["thermal_MWt"])) * 1e6
    cal, cyc = {}, {}
    for u in units:
        target = float(ov.get("gross_MWe_target", d["units"][u]["gross_MWe"])) * 1e6
        eta, c = calibrate_eta(target, thermal_W, **{**kw, "heaters_out": []})
        cal[u] = {"turbine_isentropic_efficiency": {
            "value": eta, "fitted_to": "potencia bruta publicada", "published_gross_W": target, "model_gross_W": c["gross_W"],
            "residual_rel": (c["gross_W"] - target) / target, "thermal_W": thermal_W, "bounds": [0.40, 1.0],
            "flag": "CALIBRADO a la potencia bruta de PRIS (no es validación)" +
                    ("" if "generator_efficiency" in ov else "; incluye la eficiencia del generador y pérdidas mecánicas")}}
        cyc[u] = c
        if c["negative_flows"]:
            warnings.append(f"Unidad {u}: gastos negativos en {c['negative_flows']}: el arreglo de calentadores no es físico.")
    u0 = units[0]
    c0 = cyc[u0]
    eta0 = cal[u0]["turbine_isentropic_efficiency"]["value"]
    if kw["heaters_out"]:
        c_user = cycle_with_sea(thermal_W, eta_turbine=eta0, **kw)
        warnings.append(f"Corrida con calentadores fuera de servicio {kw['heaters_out']}: bruta "
                        f"{c_user['gross_W'] / 1e6:.1f} MWe (la calibración se hizo con todos en servicio).")
        c0 = c_user
    # validation against published numbers NOT used in the fit
    hist = d["gross_output_history_MWe"]
    pre = cycle_with_sea(d["thermal_power_history_MWt"]["spu_105pct"] * 1e6, eta_turbine=eta0, **{**kw, "heaters_out": []})
    seas = d["cooling"]["seasons"]
    pre_s = {k: cycle_with_sea(d["thermal_power_history_MWt"]["spu_105pct"] * 1e6, eta_turbine=eta0,
                               **{**kw, "heaters_out": [], "sea_in_K": s["sea_T_C"] + 273.15}) for k, s in seas.items()}
    rows = [{"quantity": "bruta a 2027 MWt (antes del EPU)", "published": hist["pre_epu_at_2027MWt"],
             "model": pre["gross_W"] / 1e6, "unit": "MWe", "source": d["sources"]["IBERDROLA_ENC2010"]["url"],
             "role": "validación", "note_es": "el EPU cambió turbina de alta, condensador, MSR y calentadores 5-6: compara el "
                                              "ciclo calibrado de hoy a la potencia térmica anterior"},
            {"quantity": "bruta de diseño del EPU a 2317 MWt", "published": hist["epu_design_at_2317MWt"],
             "model": c0["gross_W"] / 1e6, "unit": "MWe", "source": d["sources"]["IBERDROLA_ENC2010"]["url"],
             "role": "otra cifra publicada", "note_es": "el modelo está calibrado a PRIS (805/803); la diferencia mide la "
                                                         "discrepancia entre fuentes, no el modelo"},
            {"quantity": "bruta tras el EPU (WNN 2013)", "published": hist["wnn_after_epu_MWe"], "model": c0["gross_W"] / 1e6,
             "unit": "MWe", "source": d["sources"]["WNN_2013"]["url"], "role": "otra cifra publicada"}]
    for k, c in pre_s.items():
        rows.append({"quantity": f"calentamiento del agua de mar (descarga − mar), temporada de {k} (2027 MWt)",
                     "published": seas[k]["discharge_T_C"] - seas[k]["sea_T_C"],
                     "model": c["sea_out_K"] - c["sea_in_K"], "unit": "K", "source": d["sources"]["RAMIREZ_LEON_2013"]["url"],
                     "role": "validación", "note_es": "63 m3/s tomado como gasto de toda la central (31.5 por unidad): el "
                                                      "artículo no lo precisa"})
    for r in rows:
        r["deviation_rel"] = (r["model"] - r["published"]) / r["published"]
    for u in units:
        rows.append({"quantity": f"bruta de PRIS, unidad {u}", "published": d["units"][u]["gross_MWe"],
                     "model": cyc[u]["gross_W"] / 1e6, "unit": "MWe", "source": d["sources"][f"PRIS_WNA_{u}"]["url"],
                     "role": "ajustado", "deviation_rel": cal[u]["turbine_isentropic_efficiency"]["residual_rel"]})
    # sensitivities
    sens_n = []
    for nh in (4, 5, 6, 7):
        try:
            dn = _design(d, {**ov, "n_feedwater_heaters": nh})
            e_n, c_n = calibrate_eta(float(ov.get("gross_MWe_target", d["units"][u0]["gross_MWe"])) * 1e6, thermal_W,
                                     **{**dn["kw"], "heaters_out": []})
            pre_n = cycle_with_sea(d["thermal_power_history_MWt"]["spu_105pct"] * 1e6, eta_turbine=e_n,
                                   **{**dn["kw"], "heaters_out": []})
            sens_n.append({"n_feedwater_heaters": nh, "eta_calibrated": e_n, "feedwater_T_K": c_n["feedwater_T_K"],
                           "steam_flow_kg_s": c_n["steam_flow_kg_s"], "gross_at_2027MWt_MWe": pre_n["gross_W"] / 1e6})
        except ValueError as exc:
            sens_n.append({"n_feedwater_heaters": nh, "error_es": str(exc)})
    sea = _sea_sensitivity({**kw, "heaters_out": []}, eta0, thermal_W, d)
    heater_case = None
    n = des["n"]
    try:
        c_out = cycle_with_sea(thermal_W, eta_turbine=eta0, **{**kw, "heaters_out": [n]})
        heater_case = {"heater": n, "gross_out_W": c_out["gross_W"], "gross_base_W": cyc[u0]["gross_W"],
                       "delta_gross_W": c_out["gross_W"] - cyc[u0]["gross_W"], "T_fw_base_K": cyc[u0]["feedwater_T_K"],
                       "T_fw_out_K": c_out["feedwater_T_K"]}
    except ValueError as exc:
        warnings.append(f"Sensibilidad de calentador fuera de servicio no calculada: {exc}")
    gen = _generation(d, units, year, {u: cyc[u]["gross_W"] / 1e6 for u in units})
    table = _prices(d, year, prices, _today(today), warnings, assumptions)
    econ = _economics(d, units, gen, table, year, warnings)
    mwh = econ.get("net_generation_MWh") or 0.0
    ef = d["emission_factor"]
    emis = {"net_generation_MWh": mwh, "grid_factor_t_mwh": ef["value_t_mwh"], "co2_avoided_t_y": mwh * ef["value_t_mwh"],
            "factor_year": ef["year"], "source": d["sources"]["FESEN_2024"]["url"],
            "note_es": "factor PROMEDIO del Sistema Eléctrico Nacional (SEMARNAT/CRE), no el marginal; sin ciclo de vida del "
                       "combustible"}
    opps = _opportunities(d, units, year, gen, econ, sea, heater_case, cyc[u0], None)
    pm = _process_model(d, u0, c0, len(units))
    rk = rankine_example()
    # evidence
    resid = max(abs(c["energy_residual_rel"]) for c in list(cyc.values()) + [pre, c0])
    presid = max(abs(c["plant_balance_residual_rel"]) for c in list(cyc.values()) + [pre, c0])
    mres = max(abs(c["mass_residual"]) for c in list(cyc.values()) + [pre, c0])
    cal_res = max(abs(v["turbine_isentropic_efficiency"]["residual_rel"]) for v in cal.values())
    gen_dev = max((abs(g["deviation_rel"]) for g in gen.values() if g.get("deviation_rel") is not None), default=0.0)
    neg = any(c["negative_flows"] for c in cyc.values())
    reasons = []
    if resid > BAL_TOL or presid > BAL_TOL:
        reasons.append(f"balance de energía no cierra (residuo {max(resid, presid):.2e})")
    if mres > BAL_TOL:
        reasons.append(f"balance de masa no cierra ({mres:.2e})")
    if cal_res > 1e-6:
        reasons.append(f"calibración no converge (residuo {cal_res:.2e})")
    if neg:
        reasons.append("gastos de extracción negativos")
    evidence = {"verified": not reasons, "energy_balance_residual_rel": resid, "plant_balance_residual_rel": presid,
                "mass_balance_residual": mres, "calibration_residual_rel": cal_res, "generation_identity_max_dev_rel": gen_dev,
                "rankine_example_energy_residual_rel": rk["energy_residual_rel"],
                "note": "verified certifica los balances y la calibración de ESTA corrida, no los datos públicos."}
    if reasons:
        evidence["reasons"] = reasons
    assumptions += [
        f"Presión de domo {d['steam']['dome_pressure_psia']:.0f} psia = {des['p_dome_Pa'] / 1e6:.4f} MPa (CNSNS-TR-25), "
        "constante en los aumentos de potencia (CPPU); vapor del domo saturado seco (x = 1).",
        f"{n} calentadores cerrados con extracciones por la regla de elevación igual de temperatura entre "
        f"{des['T_cond_design_K'] - 273.15:.1f} °C (promedio de las descargas publicadas + TTD) y "
        f"{tsat(des['p_dome_Pa']) - 273.15:.1f} °C: SIMPLIFICACIÓN DECLARADA (la CNLV no publica su arreglo); ver "
        "cycle.sensitivity_n_heaters.",
        "Presión de cruce = extracción del calentador N-1; recalentador con vapor vivo hasta la saturación del vapor vivo "
        "(TTD 0 K); calentadores y condensador con TTD 0 K; una sola bomba isentrópica condensador → domo; sin pérdidas de "
        "presión en líneas: límites termodinámicos declarados, cambiables en overrides.",
        f"Agua de mar a {kw['sea_in_K'] - 273.15:.2f} °C (promedio de las temporadas publicadas 25 y 28 °C), "
        f"{kw['cooling_flow_m3_s']:.1f} m3/s por unidad (63 m3/s publicados para la central ÷ 2), salinidad 33 g/kg.",
        "Una sola eficiencia isentrópica para todas las secciones de turbina, CALIBRADA a la bruta de PRIS.",
    ]
    if unit is None:
        warnings.append("Ambas unidades comparten diseño; el diagrama muestra la unidad 1 como representativa.")
    cycle = {"unit": u0, "states": _rows(c0), "steam_flow_kg_s": c0["steam_flow_kg_s"], "thermal_W": c0["thermal_W"],
             "gross_W": c0["gross_W"], "gross_efficiency": c0["gross_efficiency"], "heat_rate_kJ_kWh": c0["heat_rate_kJ_kWh"],
             "heat_rate_btu_kWh": c0["heat_rate_kJ_kWh"] / 1.05505585262,
             "cycle_efficiency_net_of_feed_pump": c0["efficiency_cycle"], "condenser_duty_W": c0["condenser_duty_W"],
             "condenser_pressure_Pa": c0["p_cond_Pa"], "condenser_T_K": c0["T_cond_K"],
             "cooling_water_flow_kg_s": c0["cooling_flow_kg_s"], "seawater_rise_K": c0["sea_out_K"] - c0["sea_in_K"],
             "feed_pump_W": c0["feed_pump_W"], "feedwater_T_K": c0["feedwater_T_K"],
             "extraction_pressures_Pa": c0["extraction_pressures_Pa"], "crossover_pressure_Pa": c0["crossover_pressure_Pa"],
             "extraction_fractions": c0["m_extraction_frac"], "reheater_live_steam_frac": c0["m_reheater_frac"],
             "published_aux_load_W": (d["units"][u0]["gross_MWe"] - d["units"][u0]["net_MWe"]) * 1e6,
             "duties_W": {"reactor": c0["thermal_W"], "turbine_shaft": c0["turbine_shaft_W"], "condenser": c0["condenser_duty_W"],
                          "feed_pump": c0["feed_pump_W"]},
             "sensitivity_n_heaters": sens_n, "sensitivity_seawater": sea,
             "heater_out_of_service": heater_case, "per_unit": {u: {"gross_W": cyc[u]["gross_W"],
                                                                    "eta": cal[u]["turbine_isentropic_efficiency"]["value"]}
                                                                for u in units}}
    plant = {"name": d["name"], "units": {u: d["units"][u] for u in units},
             "thermal_power_history_MWt": d["thermal_power_history_MWt"], "cooling": d["cooling"]["type"],
             "fuel": d["fuel"], "licences": {u: d["units"][u]["licence_until"] for u in units},
             "what_it_does_es": "Dos reactores de agua en ebullición (BWR-5, diseño GE) generan vapor saturado directamente en "
                                "el núcleo; el vapor mueve el turbogenerador y se condensa con agua del Golfo de México en "
                                "un solo paso; el condensado vuelve al reactor calentado con extracciones de turbina."}
    return {"plant": plant, "year": year, "process_model": pm, "cycle": cycle, "generation": gen, "economics": econ,
            "emissions_avoided": emis, "opportunities": opps, "calibrated_parameters": cal,
            "published_comparison": {"rows": rows, "rankine_example": rk,
                                     "level": "VALIDACIÓN en las filas «validación» y en el ejemplo de Rankine; «ajustado» "
                                              "reproduce el dato por construcción; «otra cifra publicada» mide discrepancias "
                                              "entre fuentes."},
            "not_found": d["not_found"], "assumptions": assumptions, "warnings": warnings, "not_included": NOT_INCLUDED,
            "method": METHOD, "evidence": evidence}
