"""Bare-module capital, total capital investment and discounted cash flow (bloque M3 of docs/MAPA_MOTORES.md).

Bare module (Guthrie method as in Seider et al.): C_BM = C_P,base · (F_BM + F_D·F_P·F_M − 1) = C_P + C_P,base·(F_BM − 1),
where C_P is the purchase cost (with material, pressure and type factors) and C_P,base the same equipment in the base
material and type (carbon steel / cast iron, type factor 1), both priced by the SAME Seider correlation in IDAES
(`packages.costing.sslw`). This is exactly BioSTEAM's formula (`Unit._update_installed_costs`). The bare-module factors
F_BM are READ from BioSTEAM's unit classes at run time (Seider et al. 2017), not transcribed: see `bare_module_factors`.

Total capital investment follows Seider's structure; every percentage is the user's input (no defaults):
  C_TBM = Σ C_BM (× location factor)  →  C_DPI = C_TBM·(1 + site + service facilities + allocated/offsites)
  →  C_TDC = C_DPI·(1 + contingency and contractor fee)  →  C_TPI = C_TDC·(1 + land + royalties + startup)
  →  TCI = C_TPI + working capital (fraction of C_TPI).

Cash flow (real terms, no loans): construction years with a schedule, working capital spent in the last construction
year and recovered at the end, land spent at the start and recovered at the end, depreciation (MACRS from BioSTEAM's
IRS Publication 946 table, or straight line / declining balance / sum of years digits) on the depreciable capital,
income tax with losses carried forward. Time 0 is the end of construction (BioSTEAM's convention). NPV, IRR, payback,
discounted payback, break-even revenue. The NPV is recomputed with BioSTEAM's own cash-flow functions as a cross-check.
"""
from __future__ import annotations

import math

import numpy as np

CAPITAL_FACTORS = ("site_preparation", "service_facilities", "allocated_costs", "contingency_and_fee", "land", "royalties",
                   "startup", "working_capital")


# --------------------------------------------------------------------------- bare-module factors (from BioSTEAM)

def bare_module_factors() -> dict:
    """F_BM by equipment type, read from BioSTEAM 2.x unit classes (Seider et al. 2017)."""
    import biosteam as bst
    from biosteam.units.design_tools import PressureVessel
    hx = bst.units.HX._F_BM_default
    pv = PressureVessel._F_BM_default
    col = bst.units.BinaryDistillation._F_BM_default
    table = {
        "heat_exchanger_floating_head": (hx["Floating head"], "biosteam.units.HX['Floating head']"),
        "heat_exchanger_fixed_head": (hx["Fixed head"], "biosteam.units.HX['Fixed head']"),
        "heat_exchanger_Utube": (hx["U tube"], "biosteam.units.HX['U tube']"),
        "heat_exchanger_kettle_vap": (hx["Kettle vaporizer"], "biosteam.units.HX['Kettle vaporizer']"),
        "double_pipe": (hx["Double pipe"], "biosteam.units.HX['Double pipe']"),
        "furnace": (hx["Furnace"], "biosteam.units.HX['Furnace']"),
        "pump": (bst.units.Pump._F_BM_default["Pump"], "biosteam.units.Pump['Pump'] (motor igual)"),
        "compressor": (bst.units.IsentropicCompressor._F_BM_default["Compressor(s)"], "biosteam.units.IsentropicCompressor"),
        "vessel_vertical": (pv["Vertical pressure vessel"], "biosteam PressureVessel['Vertical pressure vessel']"),
        "vessel_horizontal": (pv["Horizontal pressure vessel"], "biosteam PressureVessel['Horizontal pressure vessel']"),
        "tower": (col["Tower"], "biosteam.units.BinaryDistillation['Tower'] (platos igual)"),
        "storage_tank": (bst.units.StorageTank._F_BM_default["Tank"], "biosteam.units.StorageTank['Tank']"),
        "evaporator": (bst.units.MultiEffectEvaporator._F_BM_default["Evaporators"], "biosteam.units.MultiEffectEvaporator"),
        "platforms_ladders": (pv["Platform and ladders"], "biosteam PressureVessel['Platform and ladders']"),
    }
    return {"factors": {k: v[0] for k, v in table.items()}, "source": {k: v[1] for k, v in table.items()},
            "reference": f"Seider et al. (2017) vía BioSTEAM {bst.__version__}; C_BM = C_P + C_P,base·(F_BM − 1)",
            "not_listed": "Relleno, filtros, centrífugas, etc.: da F_BM tú (campo F_BM del equipo) con su fuente"}


# --------------------------------------------------------------------------- capital

def capital_investment(equipment: list[dict], factors: dict, location_factor: float = 1.0) -> dict:
    """equipment: [{name, type (key of bare_module_factors) or F_BM, purchase_cost_usd, base_purchase_cost_usd (optional),
    platforms_ladders_purchase_cost_usd (optional), basis (optional, e.g. 'CEPCI 797.9')}]. factors: all of
    CAPITAL_FACTORS as fractions (0 is valid when it does not apply)."""
    if not equipment:
        raise ValueError("equipment list is empty")
    missing = [k for k in CAPITAL_FACTORS if k not in (factors or {})]
    if missing:
        raise ValueError(f"factors missing: {missing}. They are the user's (fractions; 0 if it does not apply): "
                         "site_preparation, service_facilities, allocated_costs (of C_TBM); contingency_and_fee (of C_DPI); "
                         "land, royalties, startup (of C_TDC); working_capital (of C_TPI)")
    bad = [k for k in CAPITAL_FACTORS if not (0.0 <= float(factors[k]) <= 2.0)]
    if bad:
        raise ValueError(f"factors out of range [0, 2] (they are fractions, e.g. 0.18 = 18 %): {bad}")
    if location_factor <= 0:
        raise ValueError("location_factor must be positive")
    fbm_table = bare_module_factors()
    rows, reasons, assumptions = [], [], []
    for e in equipment:
        name = str(e.get("name", "?"))
        cp = float(e["purchase_cost_usd"])
        if cp < 0:
            raise ValueError(f"{name}: purchase_cost_usd < 0")
        if e.get("F_BM") is not None:
            fbm, src = float(e["F_BM"]), "dado por el usuario"
        elif e.get("type") in fbm_table["factors"]:
            fbm, src = fbm_table["factors"][e["type"]], fbm_table["source"][e["type"]]
        else:
            raise ValueError(f"{name}: type must be one of {sorted(fbm_table['factors'])} or give F_BM")
        pl = float(e.get("platforms_ladders_purchase_cost_usd") or 0.0)
        base = e.get("base_purchase_cost_usd")
        if base is not None:
            base = float(base)
            if base <= 0 or base > cp * (1 + 1e-9):
                raise ValueError(f"{name}: base_purchase_cost_usd must be in (0, purchase_cost_usd]")
            cbm = (cp - pl) + (base - pl) * (fbm - 1.0) + pl          # platforms and ladders keep F_BM = 1
            method = "C_P + C_P,base·(F_BM − 1)"
        else:
            cbm = (cp - pl) * fbm + pl
            method = "F_BM·C_P (sin costo base: exacto solo en acero al carbón; sobrestima en aleaciones)"
            reasons.append(f"{name}: falta base_purchase_cost_usd (lo da equipment_cost); C_BM = F_BM·C_P")
        rows.append({"name": name, "type": e.get("type"), "purchase_cost_usd": cp, "base_purchase_cost_usd": base,
                     "F_BM": fbm, "F_BM_source": src, "material_pressure_factor": (cp / base) if base else None,
                     "bare_module_cost_usd": cbm, "method": method, "basis": e.get("basis")})
        if src == "dado por el usuario":
            assumptions.append(f"{name}: F_BM {fbm} dado por el usuario")
    bases = {r["basis"] for r in rows if r["basis"]}
    if len(bases) > 1:
        reasons.append(f"costos en bases distintas {sorted(bases)}: llevar todo al mismo CEPCI antes de sumar")
    f = {k: float(factors[k]) for k in CAPITAL_FACTORS}
    c_tbm = sum(r["bare_module_cost_usd"] for r in rows) * location_factor
    site, serv, alloc = c_tbm * f["site_preparation"], c_tbm * f["service_facilities"], c_tbm * f["allocated_costs"]
    c_dpi = c_tbm + site + serv + alloc
    cont = c_dpi * f["contingency_and_fee"]
    c_tdc = c_dpi + cont
    land, roy, start = c_tdc * f["land"], c_tdc * f["royalties"], c_tdc * f["startup"]
    c_tpi = c_tdc + land + roy + start
    wc = c_tpi * f["working_capital"]
    tci = c_tpi + wc
    parts = [c_tbm, site, serv, alloc, cont, land, roy, start, wc]
    sum_ok = abs(sum(parts) - tci) <= 1e-9 * max(tci, 1.0)
    from packages.costing.annual import LANG
    purchase = sum(r["purchase_cost_usd"] for r in rows)
    fci_lang = purchase * LANG["lang_1948_fluid"] * location_factor
    ev = {"bare_module_factors_sourced": all(bool(r["F_BM_source"]) for r in rows),
          "base_costs_known": all(r["base_purchase_cost_usd"] is not None for r in rows),
          "same_cost_basis": len(bases) <= 1, "components_sum_to_total": bool(sum_ok)}
    if not sum_ok:
        reasons.append("los componentes no suman la inversión total")
    ev["verified"] = not reasons
    ev["reasons"] = reasons
    return {
        "equipment": rows, "location_factor": location_factor, "factors": f,
        "total_purchase_cost_usd": purchase, "total_bare_module_cost_usd": c_tbm,
        "site_preparation_usd": site, "service_facilities_usd": serv, "allocated_costs_usd": alloc,
        "direct_permanent_investment_usd": c_dpi, "contingency_and_fee_usd": cont,
        "total_depreciable_capital_usd": c_tdc, "land_usd": land, "royalties_usd": roy, "startup_usd": start,
        "total_permanent_investment_usd": c_tpi, "working_capital_usd": wc, "total_capital_investment_usd": tci,
        "lang_crosscheck": {"fixed_capital_lang_1948_fluid_usd": fci_lang,
                            "ratio_depreciable_capital_to_lang": c_tdc / fci_lang if fci_lang else None,
                            "note": "Lang 4.74 × compra (planta de fluidos, 1948): solo para ordenar magnitudes"},
        "assumptions": assumptions, "evidence": ev,
        "notes": ("Estructura de inversión de Seider et al.; F_BM de BioSTEAM. Los porcentajes de sitio, servicios, "
                  "contingencia, terreno, regalías, arranque y capital de trabajo son del usuario. Para el flujo de caja usa "
                  "total_depreciable_capital_usd (+ royalties y startup si se deprecian), land_usd y working_capital_usd."),
    }


# --------------------------------------------------------------------------- depreciation

def depreciation_schedule(key: str) -> np.ndarray:
    """'MACRS7' (IRS Pub. 946, half-year, from BioSTEAM), 'SL10', 'DDB10', 'SYD10'."""
    k = str(key).upper().replace("-", "").replace(" ", "")
    for prefix in ("MACRS", "SL", "DDB", "SYD"):
        if k.startswith(prefix) and k[len(prefix):].isdigit():
            n = int(k[len(prefix):])
            break
    else:
        raise ValueError("depreciation: 'MACRS3|5|7|10|15|20', 'SL<n>', 'DDB<n>' or 'SYD<n>'")
    if n < 1:
        raise ValueError("depreciation years must be ≥ 1")
    if prefix == "MACRS":
        from biosteam import TEA
        tab = TEA.depreciation_schedules
        if ("MACRS", n) not in tab:
            raise ValueError("MACRS years: 3, 5, 7, 10, 15 or 20")
        return np.array(tab[("MACRS", n)], dtype=float)
    if prefix == "SL":
        return np.full(n, 1.0 / n)
    if prefix == "SYD":
        s = n * (n + 1) / 2
        return np.array([(n - i) / s for i in range(n)])
    arr, val = np.zeros(n), 1.0          # double declining balance switching to straight line (fully depreciates;
    for i in range(n):                   # BioSTEAM's DDB never switches and leaves (1 − 2/n)^n undepreciated)
        arr[i] = max(val * 2.0 / n, val / (n - i))
        val -= arr[i]
    return arr


# --------------------------------------------------------------------------- cash flow

def _cashflow(dep_capital, land, wc, sched, life, revenue, opex, tax, dep, st, st_rev, st_opex):
    nc = len(sched)
    n = nc + life
    t = np.arange(-nc + 1, life + 1, dtype=float)
    cap, wca, landa, S, C, D = (np.zeros(n) for _ in range(6))
    cap[:nc] = dep_capital * np.asarray(sched)
    wca[nc - 1] += wc
    wca[-1] -= wc
    landa[0] += land
    landa[-1] -= land
    whole, part = int(st), st % 1
    for j in range(life):
        i = nc + j
        if j < whole:
            S[i], C[i] = st_rev * revenue, st_opex * opex
        elif j == whole:
            S[i] = part * st_rev * revenue + (1 - part) * revenue
            C[i] = part * st_opex * opex + (1 - part) * opex
        else:
            S[i], C[i] = revenue, opex
    D[nc:nc + len(dep)] = dep_capital * dep
    taxable = S - C - D
    carried, tx = 0.0, np.zeros(n)
    for i in range(n):
        x = taxable[i] + carried
        if x < 0:
            carried, tx[i] = x, 0.0
        else:
            carried, tx[i] = 0.0, tax * x
    cf = S - C - tx - cap - wca - landa
    return {"t": t, "capital": cap, "working_capital": wca, "land": landa, "revenue": S, "operating_cost": C,
            "depreciation": D, "taxable_income": taxable, "tax": tx, "cash_flow": cf}


def _npv(cf, t, r):
    return float(np.sum(cf / (1.0 + r) ** t))


def _payback(cf, t):
    cum = np.cumsum(cf)
    for i in range(1, len(cum)):
        if cum[i - 1] < 0 <= cum[i]:
            return float(t[i - 1] + (-cum[i - 1]) / (cum[i] - cum[i - 1]) * (t[i] - t[i - 1]))
    return None


def _biosteam_npv(dep_capital, land, wc, sched, life, revenue, opex, tax, dep, st, st_rev, st_opex, r):
    """Same cash flow through BioSTEAM's functions (independent implementation of the cash-flow mechanics: construction,
    startup, working capital, carried losses, discounting) for the cross-check. The depreciation array is ours."""
    from biosteam import _tea as BT
    nc = len(sched)
    n = nc + life
    D, C, S, C_FC, C_WC = (np.zeros(n) for _ in range(5))
    BT.fill_taxable_and_nontaxable_cashflows_without_loans(D, C, S, C_FC, C_WC, dep_capital, wc, dep_capital, opex, 0.0,
                                                          revenue, st, st_opex, 0.0, st_rev, np.asarray(sched, float), nc)
    D[nc:nc + dep.size] = dep_capital * dep
    taxable = S - C - D
    taxed = BT.taxable_earnings_with_fowarded_losses(taxable)
    cf = taxable + (D - C_FC - C_WC) - tax * taxed
    cf[0] -= land
    cf[-1] += land
    t = np.arange(-nc + 1, life + 1, dtype=float)
    return float(BT.NPV_at_IRR(r, cf, t))


def cash_flow_analysis(depreciable_capital_usd: float, revenue_usd_y: float, operating_cost_usd_y: float,
                       plant_life_years: int, income_tax_rate: float, discount_rate: float, depreciation: str,
                       construction_schedule: list[float], working_capital_usd: float = 0.0, land_usd: float = 0.0,
                       startup_years: float = 0.0, startup_revenue_fraction: float = 1.0,
                       startup_operating_cost_fraction: float = 1.0) -> dict:
    """Discounted cash flow of a plant, real terms, no loans. See module docstring."""
    life = int(plant_life_years)
    sched = [float(x) for x in construction_schedule]
    if life < 1 or not sched or any(x < 0 for x in sched) or abs(sum(sched) - 1.0) > 1e-6:
        raise ValueError("plant_life_years ≥ 1 and construction_schedule = fractions per year summing to 1 (e.g. [0.4, 0.6])")
    if not (0 <= income_tax_rate < 1) or not (-0.5 < discount_rate < 1):
        raise ValueError("income_tax_rate in [0, 1) and discount_rate in (-0.5, 1) as fractions (0.30 = 30 %)")
    if depreciable_capital_usd <= 0 or min(working_capital_usd, land_usd, revenue_usd_y, operating_cost_usd_y) < 0:
        raise ValueError("depreciable_capital_usd > 0; revenue, operating cost, working capital and land ≥ 0")
    if not (0 <= startup_years < life):
        raise ValueError("startup_years in [0, plant_life_years)")
    dep = depreciation_schedule(depreciation)
    if dep.size > life:
        raise ValueError(f"depreciation {depreciation} lasts {dep.size} years, longer than the plant life ({life})")
    args = (depreciable_capital_usd, land_usd, working_capital_usd, sched, life, revenue_usd_y, operating_cost_usd_y,
            income_tax_rate, dep, startup_years, startup_revenue_fraction, startup_operating_cost_fraction)
    tab = _cashflow(*args)
    t, cf = tab["t"], tab["cash_flow"]
    npv = _npv(cf, t, discount_rate)
    signs = np.sign(cf[np.abs(cf) > 1e-9 * depreciable_capital_usd])
    changes = int(np.sum(signs[1:] != signs[:-1]))
    irr = None
    from scipy.optimize import brentq
    f = lambda r: _npv(cf, t, r)
    grid = [-0.9, -0.5, -0.2, 0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 5.0]
    for a, b in zip(grid[:-1], grid[1:]):
        if f(a) * f(b) <= 0:
            irr = float(brentq(f, a, b, xtol=1e-12))
            break
    disc = cf / (1.0 + discount_rate) ** t
    # break-even revenue (NPV = 0 at the discount rate)
    g = lambda rev: _npv(_cashflow(*(args[:5] + (rev,) + args[6:]))["cash_flow"], t, discount_rate)
    hi = max(revenue_usd_y, operating_cost_usd_y, depreciable_capital_usd) * 2 + 1.0
    while g(hi) < 0 and hi < 1e15:
        hi *= 2
    be = float(brentq(g, 0.0, hi, xtol=1e-6)) if g(0.0) < 0 < g(hi) else (0.0 if g(0.0) >= 0 else None)
    npv_bst = _biosteam_npv(depreciable_capital_usd, land_usd, working_capital_usd, sched, life, revenue_usd_y,
                            operating_cost_usd_y, income_tax_rate, dep, startup_years,
                            startup_revenue_fraction, startup_operating_cost_fraction, discount_rate)
    scale = depreciable_capital_usd + working_capital_usd + land_usd
    reasons = []
    ev = {"depreciation_complete": abs(tab["depreciation"].sum() - depreciable_capital_usd) < 1e-4 * depreciable_capital_usd,
          "crosscheck_biosteam": abs(npv - npv_bst) < 1e-6 * scale, "npv_biosteam_usd": npv_bst,
          "irr_consistent": (abs(f(irr)) < 1e-6 * scale) if irr is not None else None,
          "single_sign_change": changes == 1, "sign_changes": changes}
    if not ev["depreciation_complete"]:
        reasons.append("la depreciación no suma el capital depreciable")
    if not ev["crosscheck_biosteam"]:
        reasons.append(f"el VPN no coincide con el de BioSTEAM ({npv:.6g} vs {npv_bst:.6g})")
    if irr is not None and not ev["irr_consistent"]:
        reasons.append("el VPN a la TIR no es cero")
    if changes > 1:
        reasons.append(f"el flujo cambia de signo {changes} veces: puede haber varias TIR; decide con el VPN")
    ev = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v) if isinstance(v, np.floating) else v) for k, v in ev.items()}
    ev["verified"] = not reasons
    ev["reasons"] = reasons
    rows = [{"t": int(t[i]), **{k: float(tab[k][i]) for k in ("capital", "working_capital", "land", "revenue", "operating_cost",
                                                               "depreciation", "taxable_income", "tax", "cash_flow")},
             "discounted_cash_flow": float(disc[i]), "cumulative_discounted": float(np.cumsum(disc)[i])} for i in range(len(t))]
    return {
        "npv_usd": float(npv), "irr": irr, "payback_years": _payback(cf, t), "discounted_payback_years": _payback(disc, t),
        "break_even_revenue_usd_y": be, "total_capital_spent_usd": depreciable_capital_usd + working_capital_usd + land_usd,
        "table": rows, "depreciation_schedule": dep.tolist(), "evidence": ev,
        "conventions": ("t = 0 al terminar la construcción; capital en años de construcción t ≤ 0 según el calendario; capital "
                        "de trabajo en el último año de construcción y recuperado al final; terreno al inicio y recuperado al "
                        "final; depreciación desde t = 1; impuesto con pérdidas trasladadas a años siguientes; términos reales "
                        "(sin inflación) y sin préstamos; payback en años desde t = 0."),
        "notes": ("Ingresos, costos, tasa de impuesto y de descuento son del usuario. MACRS es la tabla del IRS (EE. UU.); para "
                  "México usa SL con la tasa de la Ley del ISR que corresponda al activo (p. ej. SL10 = 10 % anual)."),
    }
