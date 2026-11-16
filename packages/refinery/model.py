"""Planning-level refinery model of the Pemex refineries, built for Miguel Hidalgo (Tula) (bloque M27_R).

What it answers: with PUBLIC data only, the crude run of a Pemex refinery is split into TBP cuts (published assays of Maya,
Istmo and Olmeca), routed through its published unit capacities (eBDI «capacidad ajustada») with fixed yield vectors taken
from published worked examples (FCC, catalytic reforming, delayed coking, alkylation), hydrotreated to the NOM-016-CRE-2016
sulfur specs, and closed in mass, volume, sulfur and hydrogen. It returns the ProcessModel JSON (units, streams, sections)
that block D draws and block O annotates, the product slate against Pemex's published production, energy from published
specific consumptions, CO2, and the operating margin at published prices.

This is how refinery PLANNERS model a refinery (unit yields + linear balances, the structure of a refinery LP), NOT a
rigorous simulation: no reactor kinetics, no column internals, no blending of octane/cetane, no optimisation. Every yield
vector is a published example at ONE set of conditions (reformer at RON 98, coker scaled by the Conradson carbon of its
feed) and is applied as is, EXCEPT the FCC, whose yields are a function of conversion (piecewise-linear in the three
severities of the KBR pilot-plant table, Niccum 2013; conversion calibrated to diesel only when identifiable, else the
table's central case). FCC light cycle oil reaches diesel only after straight-run distillate and within the published
hydrodesulfurization capacity; the rest goes to fuel oil (IMP, Laredo et al. 2002).

Balances (per unit, exact by construction and checked):
    mass:   sum(products) = feed (+ H2 consumed in hydrotreating)
    sulfur: sum(S in products) = S in feed (hydrotreaters remove S as H2S down to the product spec)
    volume: CDU cuts are vol% of crude (light ends by difference to 100 %); other units by mass / standard density.
Standard density = SG(60 °F) * density of water at 60 °F (IAPWS-95 via CoolProp); SG = 141.5 / (°API + 131.5).
"""
from __future__ import annotations

import json
import math
from datetime import date
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"
BBL_M3 = 0.158987304                 # m3 per barrel (Pemex eBDI conversion factors; exact definition 42 US gal)
BTU_J = 1055.05585262                # J per Btu (Pemex eBDI conversion factors)
M_S, M_H2, M_H2S = 32.065, 2.01588, 34.08088     # g/mol (IUPAC standard atomic weights)
M_C, M_CO2 = 12.011, 44.0095
T60F_K = 288.7055556
DEFAULT_PERIOD = "2026-01/2026-09"
CUTS = ("naphtha", "kerosene", "lgo", "hgo", "vr")
FLAG = "VALOR DE REFERENCIA PUBLICADO, NO TU COTIZACIÓN"
OVERRIDE_KEYS = ("capacity_mbd", "kerosene_to_jet_fraction", "asphalt_bpd", "butanes_to_gas_plants_fraction",
                 "fcc_conversion_wt_pct", "cutter_fraction")
MODES = ("as_operated", "nameplate")
UNIT_KEYS = ("cdu", "vdu", "coker", "visbreaker", "fcc", "reformer", "alkylation_isomerization", "hds_total")
PRICE_KEYS = {"maya": "USD/bbl", "istmo": "USD/bbl", "olmeca": "USD/bbl", "gasolina": "USD/bbl", "turbosina": "USD/bbl",
              "diesel": "USD/bbl", "combustoleo": "USD/bbl", "asfalto": "USD/bbl", "glp": "USD/bbl", "coque": "USD/t",
              "azufre": "USD/t", "gas_seco": "USD/GJ", "energia": "USD/GJ", "costo_fijo": "USD/bbl", "fx_mxn_usd": "MXN/USD"}

METHOD = ("Modelo de PLANEACIÓN de refinería (rendimientos por unidad + balances lineales, como un modelo de programación "
          "lineal de refinería sin optimizar): crudo → destilación atmosférica por cortes TBP de ensayes publicados → "
          "vacío → "
          "coquización / reductora de viscosidad / asfalto → FCC → hidrotratamiento de naftas y reformación → alquilación → "
          "hidrodesulfuración de turbosina, diésel y gasolina catalítica a la NOM-016-CRE-2016 → pools de productos. "
          "Vectores "
          "de rendimiento FIJOS de ejemplos resueltos publicados (Jechura, Colorado School of Mines, con correlaciones "
          "de Gary et al.) salvo el FCC, con rendimientos contra conversión de una tabla publicada de planta piloto "
          "(Niccum, KBR 2013) y LCO a diésel limitado por la capacidad de hidrodesulfuración; capacidades de la eBDI de Pemex; energía con intensidades del DOE (2015); balances de masa, "
          "volumen, azufre e "
          "hidrógeno cerrados por unidad y verificados. Python puro, sin simulador.")
NOT_INCLUDED = [
    "Cinética de reactores, diseño de columnas y efecto de las condiciones de operación: cada unidad usa UN vector de "
    "rendimiento publicado a una condición fija (reformadora a RON 98); el FCC usa rendimientos contra conversión de "
    "UNA tabla de planta piloto (una carga de 22.5 °API y 0.55 % S, un catalizador), no la carga ni el catalizador de "
    "Pemex.",
    "Diluente (cutter) del combustóleo: no se calcula por viscosidad (no hay viscosidad abierta del residuo de vacío ni "
    "receta publicada); es una fracción del destilado intermedio (cutter_fraction), calibrada al combustóleo sólo donde "
    "éste no identifica ya la coquizadora; donde sí, queda en 0 (no identificable).",
    "Optimización (programación lineal) de la carga, de los cortes y del destino de las corrientes: el ruteo es fijo.",
    "Calidad de mezclas: octano, cetano, presión de vapor, viscosidad del combustóleo y diluentes.",
    "Isomerización (su capacidad está sumada a la de alquilación en la eBDI y no hay reparto publicado), H-Oil, planta de "
    "hidrógeno y recuperación real de las plantas de azufre (se reporta el azufre potencial en el gas ácido).",
    "Consumo real de hidrógeno de los hidrotratamientos: se calcula sólo el estequiométrico de la desulfuración (cota "
    "inferior).",
    "Mermas, autoconsumos de producto, inventarios, paros y transferencias entre refinerías.",
    "Costo de construir la refinería (VPN): la refinería existe; la rentabilidad aquí es el margen de operación.",
]


# ---------------------------------------------------------------------------------------------------------------- data
@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def water_density_60F() -> float:
    """Density of liquid water at 60 °F and 1 atm (kg/m3), IAPWS-95 through CoolProp."""
    from CoolProp.CoolProp import PropsSI
    return float(PropsSI("D", "T", T60F_K, "P", 101325.0, "Water"))


def sg_from_api(api: float) -> float:
    return 141.5 / (api + 131.5)


def kg_per_bbl(sg: float) -> float:
    return sg * water_density_60F() * BBL_M3


@lru_cache(maxsize=1)
def fuel_constants() -> dict:
    """LHV of CH4 and H2 and kg CO2 per GJ of CH4 from the NASA polynomials of the combustion engine (bloque M18)."""
    from packages.energy import combustion as C
    h = lambda s: C.h_molar(s, 298.15)          # noqa: E731
    lhv_ch4 = (h("CH4") + 2 * h("O2") - h("CO2") - 2 * h("H2O")) / C._mw("CH4")
    lhv_h2 = (h("H2") + 0.5 * h("O2") - h("H2O")) / C._mw("H2")
    co2_per_GJ = (C._mw("CO2") / C._mw("CH4")) / (lhv_ch4 / 1e9)
    return {"lhv_ch4_J_kg": lhv_ch4, "lhv_h2_J_kg": lhv_h2, "co2_kg_GJ_ch4": co2_per_GJ}


# ------------------------------------------------------------------------------------------------------------- flows
def _flow(bpd: float = 0.0, kg_d: float = 0.0, s_kg_d: float = 0.0, ccr_kg_d: float = 0.0) -> dict:
    return {"bpd": bpd, "kg_d": kg_d, "s_kg_d": s_kg_d, "ccr_kg_d": ccr_kg_d}


def _mix(*flows: dict) -> dict:
    out = _flow()
    for f in flows:
        for k in out:
            out[k] += f.get(k, 0.0)
    return out


def _part(f: dict, frac: float) -> dict:
    return {k: v * frac for k, v in f.items()}


def _sg(f: dict) -> float | None:
    return f["kg_d"] / (f["bpd"] * BBL_M3 * water_density_60F()) if f["bpd"] > 0 else None


def _s_wt(f: dict) -> float:
    return 100.0 * f["s_kg_d"] / f["kg_d"] if f["kg_d"] > 0 else 0.0


def _take(f: dict, limit_bpd: float) -> tuple[dict, dict]:
    """Split a flow by volume: (taken up to limit_bpd, rest), both with the same composition."""
    if f["bpd"] <= 0 or limit_bpd >= f["bpd"]:
        return dict(f), _flow()
    x = max(limit_bpd, 0.0) / f["bpd"]
    return _part(f, x), _part(f, 1.0 - x)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


# ---------------------------------------------------------------------------------------------------------- inputs
def available_refineries() -> list[str]:
    return sorted(_load("pemex_refineries.json")["refineries"])


def _refinery(refinery: str) -> tuple[str, dict, dict]:
    key = str(refinery).strip().lower().replace(" ", "_").replace("á", "a").replace("é", "e").replace("í", "i")
    alias = {"miguel_hidalgo": "tula", "olmeca": "dos_bocas", "lazaro_cardenas": "minatitlan", "ciudad_madero": "madero"}
    key = alias.get(key, key)
    refs = _load("pemex_refineries.json")["refineries"]
    if key not in refs:
        raise ValueError(f"refinery = {refinery!r}: no hay datos públicos cargados. Disponibles: {sorted(refs)}.")
    extra = _load(f"{key}.json") if (DATA / f"{key}.json").exists() else {}
    return key, refs[key], extra


def _period(ref: dict, period: str | None, year: int | None) -> str:
    if period is None and year is not None:
        period = {2024: "2024", 2025: "2025", 2026: DEFAULT_PERIOD}.get(int(year))
        if period is None:
            raise ValueError(f"year = {year}: hay datos publicados para 2024, 2025 y 2026 (ene-sep); da otro año o period.")
    period = period or DEFAULT_PERIOD
    if period not in ref["periods"]:
        raise ValueError(f"period = {period!r}: periodos publicados disponibles {sorted(ref['periods'])}.")
    return period


def _capacities(ref: dict, extra: dict, unit_overrides: dict, warnings: list, assumptions: list) -> dict:
    caps = {}
    for k in UNIT_KEYS:
        v = ref["capacity_mbd_2025"].get(k)
        caps[k] = {"mbd": v, "source": "Pemex eBDI, capacidad ajustada 2025 (AEPSTRI5A)"}
    for k, o in extra.get("capacity_overrides_mbd", {}).items():
        caps[k] = {"mbd": o["value"], "source": f"{o['source']} ({o['url']})", "status": o.get("status")}
        assumptions.append(f"Capacidad de {k} = {o['value']} Mbd tomada de {o['source']} [fuente {o['source_tier']}]: "
                           f"no está en el cuadro de capacidades de la eBDI 2025. Estado: {o.get('status')}.")
    for k, v in (unit_overrides.get("capacity_mbd") or {}).items():
        if k not in UNIT_KEYS:
            raise ValueError(f"unit_overrides['capacity_mbd'] tiene la unidad desconocida {k!r}; usa una de {list(UNIT_KEYS)}.")
        if v is None or float(v) < 0:
            raise ValueError(f"unit_overrides['capacity_mbd'][{k!r}] debe ser ≥ 0 (miles de barriles diarios).")
        caps[k] = {"mbd": float(v), "source": "dato del usuario"}
    for k in UNIT_KEYS:
        if caps[k]["mbd"] is None:
            caps[k]["mbd"] = 0.0
            caps[k]["note"] = "la eBDI no registra esta unidad en esta refinería ('--' o N/D): se modela sin ella"
    return caps


def _slate(ref: dict, period: str, crude_slate: dict | None, utilization, cdu_mbd: float, warnings: list,
           assumptions: list) -> tuple[dict, dict, bool]:
    assays = _load("crude_assays.json")
    tmap = assays["pemex_type_map"]
    pub = ref["periods"][period]["crude_mbd"]
    published = {}
    for t, v in pub.items():
        if t == "total" or not v:
            continue
        if t in tmap:
            published[tmap[t]] = v * 1000.0
        else:
            warnings.append(f"La carga publicada incluye {v:.3f} Mbd de «{t}» sin ensaye abierto: se EXCLUYE del modelo "
                            f"(no se inventa su composición).")
    is_published = crude_slate is None and not isinstance(utilization, (int, float))
    if crude_slate is not None:
        slate = {}
        for k, v in crude_slate.items():
            kk = str(k).strip().lower()
            kk = tmap.get(kk, kk)
            if kk not in assays["crudes"]:
                raise ValueError(f"crude_slate: crudo {k!r} sin ensaye; usa {sorted(assays['crudes'])} (o pesado/ligero/"
                                 f"superligero) en barriles por día.")
            if v is None or float(v) < 0:
                raise ValueError(f"crude_slate[{k!r}] debe ser ≥ 0 barriles por día.")
            slate[kk] = slate.get(kk, 0.0) + float(v)
        assumptions.append("Carga de crudo dada por el usuario.")
    else:
        slate = dict(published)
        assumptions.append(f"Carga de crudo = proceso PUBLICADO por Pemex para el periodo {period} por tipo "
                           f"({', '.join(f'{k} {v / 1000:.3f} Mbd' for k, v in slate.items())}); "
                           f"{assays['pemex_type_map_note']}")
        if isinstance(utilization, (int, float)):
            tot = sum(slate.values())
            if tot <= 0:
                raise ValueError("No hay proceso de crudo publicado en ese periodo para repartir la utilización.")
            scale = float(utilization) * cdu_mbd * 1000.0 / tot
            slate = {k: v * scale for k, v in slate.items()}
            assumptions.append(f"Utilización de la destilación atmosférica = {float(utilization):.3f} de {cdu_mbd} Mbd, con "
                               f"el reparto de crudos publicado del periodo.")
    if sum(slate.values()) <= 0:
        raise ValueError("La carga de crudo es cero: da crude_slate en barriles por día o un periodo con proceso publicado.")
    return slate, published, is_published


def _utilization(utilization) -> dict:
    out = {k: 1.0 for k in UNIT_KEYS}
    if isinstance(utilization, dict):
        for k, v in utilization.items():
            if k not in UNIT_KEYS:
                raise ValueError(f"utilization: unidad desconocida {k!r}; usa {list(UNIT_KEYS)} (fracción de la capacidad).")
            if v is None or not (0.0 <= float(v) <= 1.5):
                raise ValueError(f"utilization[{k!r}] debe ser una fracción entre 0 y 1.5 de la capacidad.")
            out[k] = float(v)
    elif utilization is not None and not isinstance(utilization, (int, float)):
        raise ValueError("utilization: da un número (fracción de la capacidad atmosférica) o un dict por unidad.")
    elif isinstance(utilization, (int, float)) and not (0.0 < float(utilization) <= 1.5):
        raise ValueError("utilization (atmosférica) debe estar entre 0 y 1.5.")
    return out


# ------------------------------------------------------------------------------------------------------------- units
def crude_unit(slate: dict) -> tuple[dict, dict]:
    """TBP cuts of the blend (bbl/d, kg/d, S, CCR in VR) and the assay consistency of every crude."""
    a = _load("crude_assays.json")
    le_sg = a["light_ends_sg"]["value"]
    cuts = {c: _flow() for c in CUTS}
    cuts["light_ends"] = _flow()
    checks = {}
    for crude, bpd in slate.items():
        if bpd <= 0:
            continue
        x = a["crudes"][crude]
        vol_sum = sum(x["cuts"][c]["vol_pct"] for c in CUTS)
        le_pct = 100.0 - vol_sum
        m_api = bpd * kg_per_bbl(sg_from_api(x["api"]))
        m_cuts, s_cuts = 0.0, 0.0
        for c in CUTS:
            q = x["cuts"][c]
            b = bpd * q["vol_pct"] / 100.0
            m = b * kg_per_bbl(sg_from_api(q["api"]))
            s = m * q["sulfur_wt"] / 100.0
            f = _flow(b, m, s)
            if c == "vr":
                if "ccr_wt" in q:
                    ccr = q["ccr_wt"]
                else:          # all Conradson carbon of the crude assumed to report to the vacuum residue
                    ccr = x["ccr_wt"] * m_api / m
                f["ccr_kg_d"] = m * ccr / 100.0
            cuts[c] = _mix(cuts[c], f)
            m_cuts += m
            s_cuts += s
        b_le = bpd * le_pct / 100.0
        m_le = b_le * kg_per_bbl(le_sg)
        cuts["light_ends"] = _mix(cuts["light_ends"], _flow(b_le, m_le, 0.0))
        m_tot = m_cuts + m_le
        vr = x["cuts"]["vr"]
        m_vr = bpd * vr["vol_pct"] / 100.0 * kg_per_bbl(sg_from_api(vr["api"]))
        ccr_est = x["ccr_wt"] * m_api / m_vr
        checks[crude] = {
            "bpd": bpd, "light_ends_by_difference_vol_pct": le_pct,
            "light_ends_published_vol_pct": a["light_ends_check"]["values_vol_pct"].get(crude),
            "mass_from_cuts_vs_whole_crude_api_rel": (m_tot - m_api) / m_api,
            "sulfur_from_cuts_vs_whole_crude_rel": (100.0 * s_cuts / m_tot - x["sulfur_wt"]) / x["sulfur_wt"],
            "ar_651plus_vol_pct_published": x["ar_651plus"]["vol_pct"],
            "hgo_plus_vr_vol_pct": x["cuts"]["hgo"]["vol_pct"] + vr["vol_pct"],
            "vr_ccr_wt_used": vr.get("ccr_wt", ccr_est),
            "vr_ccr_wt_estimated_all_ccr_in_vr": ccr_est,
            "vr_ccr_wt_published": vr.get("ccr_wt"),
        }
    return cuts, checks


def fcc_conversion_bounds() -> tuple[float, float]:
    x = _load("unit_yields.json")["fcc_vs_conversion"]["conversion_wt_pct"]
    return float(x[0]), float(x[-1])


def _pwl(x: float, xs: list, ys: list) -> float:
    """Piecewise-linear interpolation through the published points (no extrapolation: caller checks the range)."""
    for i in range(len(xs) - 1):
        if x <= xs[i + 1] or i == len(xs) - 2:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def fcc_yields_wt(conversion_wt_pct: float) -> dict:
    """FCC yields (wt% of feed) and product gravities at a conversion, piecewise-linear in the published KBR table.

    conversion = 100 - LCO - slurry (wt%). Returns the seven published lumps, olefinicities and API gravities."""
    y = _load("unit_yields.json")["fcc_vs_conversion"]
    xs = y["conversion_wt_pct"]
    lo, hi = xs[0], xs[-1]
    x = float(conversion_wt_pct)
    if not lo - 1e-9 <= x <= hi + 1e-9:
        raise ValueError(f"Conversión del FCC = {x} % peso fuera del intervalo publicado {lo}-{hi} % peso (Niccum/KBR 2013): "
                         f"el modelo no extrapola la tabla.")
    wt = {k: _pwl(x, xs, v) for k, v in y["wt_pct"].items()}
    return {"wt_pct": wt, "api": {k: _pwl(x, xs, v) for k, v in y["api"].items()},
            "c3_olefinicity": _pwl(x, xs, y["c3_olefinicity_wt_pct"]) / 100.0,
            "c4_olefinicity": _pwl(x, xs, y["c4_olefinicity_wt_pct"]) / 100.0, "conversion_wt_pct": x}


def fcc_unit(feed: dict, conversion_wt_pct: float) -> dict:
    """FCC yields as a function of conversion (KBR pilot-plant table, piecewise linear); C3/C4 split by the published
    olefinicity, iC4/nC4 and sulfur split by the Jechura example; LPG component densities of the Jechura table."""
    ex = _load("unit_yields.json")["fcc"]
    yc = _load("unit_yields.json")["fcc_vs_conversion"]
    r = fcc_yields_wt(conversion_wt_pct)
    wt = r["wt_pct"]
    fi = yc["isobutane_fraction_of_c4_paraffins_wt"]["value"]
    c3, c4 = wt["c3_lpg"], wt["c4_lpg"]
    parts = {"light_gases": (wt["dry_gas"], None), "propane": (c3 * (1 - r["c3_olefinicity"]), ex["products"]["propane"]["sg"]),
             "propylene": (c3 * r["c3_olefinicity"], ex["products"]["propylene"]["sg"]),
             "isobutane": (c4 * (1 - r["c4_olefinicity"]) * fi, ex["products"]["isobutane"]["sg"]),
             "n_butane": (c4 * (1 - r["c4_olefinicity"]) * (1 - fi), ex["products"]["n_butane"]["sg"]),
             "butylenes": (c4 * r["c4_olefinicity"], ex["products"]["butylenes"]["sg"]),
             "gasoline": (wt["gasoline"], sg_from_api(r["api"]["gasoline"])), "lco": (wt["lco"], sg_from_api(r["api"]["lco"])),
             "hco": (wt["slurry"], sg_from_api(r["api"]["slurry"])), "coke": (wt["coke"], None)}
    rec = ex["sulfur_recovery_pct"]
    rsum = sum(rec.values())
    # The published dry gas is taken as hydrocarbon (pilot feed 0.55 % S); the H2S of the sulfur split is an extra lump
    # and every published lump is scaled by (1 - H2S fraction) so the unit closes on the feed.
    s_gas = feed["s_kg_d"] * rec["light_gases"] / rsum
    h2s = s_gas * M_H2S / M_S
    k = 1.0 - h2s / feed["kg_d"] if feed["kg_d"] > 0 else 1.0
    out = {}
    for p, (w, sg) in parts.items():
        if p == "light_gases":
            continue
        m = k * feed["kg_d"] * w / 100.0
        out[p] = _flow(m / kg_per_bbl(sg) if sg else 0.0, m, feed["s_kg_d"] * rec.get(p, 0.0) / rsum)
    out["acid_gas"] = _flow(0.0, h2s, s_gas)
    out["dry_gas"] = _flow(0.0, k * feed["kg_d"] * parts["light_gases"][0] / 100.0, 0.0)
    out["_h2s_scaling"] = k
    out["_sulfur_recovery_sum_pct"] = rsum
    out["_wt_pct"] = wt
    out["_conversion_wt_pct"] = r["conversion_wt_pct"]
    return out


def fcc_example_unit(feed: dict) -> dict:
    """Fixed FCC yield vector (wt% on feed) of the Jechura example at 72 vol% (kept as a cross-check of the
    conversion-dependent table and of the wt->vol arithmetic); sulfur by its published recovery split."""
    y = _load("unit_yields.json")["fcc"]
    rec = y["sulfur_recovery_pct"]
    rsum = sum(rec.values())
    out = {}
    s_feed = feed["s_kg_d"]
    for p, q in y["products"].items():
        m = feed["kg_d"] * q["wt_pct"] / 100.0
        s = s_feed * rec.get(p, 0.0) / rsum
        b = m / kg_per_bbl(q["sg"]) if q.get("sg") else 0.0
        out[p] = _flow(b, m, s)
    lg = out["light_gases"]
    h2s = lg["s_kg_d"] * M_H2S / M_S
    out["acid_gas"] = _flow(0.0, h2s, lg["s_kg_d"])
    out["dry_gas"] = _flow(0.0, lg["kg_d"] - h2s, 0.0)
    del out["light_gases"]
    out["_sulfur_recovery_sum_pct"] = rsum
    return out


def coker_unit(feed: dict) -> dict:
    """Coke = (coke/CCR of the published example) x CCR of the feed; the rest split like the example; S as published."""
    y = _load("unit_yields.json")["coker"]
    k = y["wt_pct"]["coke"] / y["conditions"]["feed_ccr_wt"]
    ccr = 100.0 * feed["ccr_kg_d"] / feed["kg_d"] if feed["kg_d"] > 0 else 0.0
    coke = min(k * ccr, 100.0)
    rest = {p: y["wt_pct"][p] for p in ("gas", "coker_gasoline", "coker_gas_oil")}
    rsum = sum(rest.values())
    wt = {p: (100.0 - coke) * v / rsum for p, v in rest.items()}
    wt["coke"] = coke
    sd = y["sulfur_distribution_pct"]
    s_split = {"gas": sd["gas"], "coker_gasoline": sd["light_naphtha"] + sd["heavy_naphtha"],
               "coker_gas_oil": sd["lcgo"] + sd["hcgo"], "coke": sd["coke"]}
    ssum = sum(s_split.values())
    out = {}
    for p, w in wt.items():
        m = feed["kg_d"] * w / 100.0
        s = feed["s_kg_d"] * s_split[p] / ssum
        sg = y["sg"].get(p)
        out[p] = _flow(m / kg_per_bbl(sg) if sg else 0.0, m, s)
    gas = out.pop("gas")
    h2s = gas["s_kg_d"] * M_H2S / M_S
    hc = gas["kg_d"] - h2s
    comp = {c: v for c, v in y["gas_lb_per_day"].items() if c != "h2s"}
    csum = sum(comp.values())
    dry = sum(comp[c] for c in ("methane", "ethene", "ethane", "h2", "co2")) / csum
    lpg_sg = {"propene": 0.5210, "propane": 0.5070, "butenes": 0.6013, "isobutane": 0.5629, "n_butane": 0.5840}
    out["acid_gas"] = _flow(0.0, h2s, gas["s_kg_d"])
    out["dry_gas"] = _flow(0.0, hc * dry, 0.0)
    out["lpg"] = _flow(sum(hc * comp[c] / csum / kg_per_bbl(sg) for c, sg in lpg_sg.items()), hc * (1.0 - dry), 0.0)
    c4 = ("butenes", "isobutane", "n_butane")
    out["_lpg_c4"] = _flow(sum(hc * comp[c] / csum / kg_per_bbl(lpg_sg[c]) for c in c4), sum(hc * comp[c] / csum for c in c4), 0.0)
    out["_wt_pct"] = wt
    out["_feed_ccr_wt"] = ccr
    return out


def reformer_unit(feed: dict) -> dict:
    """Reformer yield (wt%) of the published example at RON 98, H2S excluded and the rest renormalized."""
    y = _load("unit_yields.json")["reformer"]
    lb = {k: v for k, v in y["lb_per_day"].items() if k != "h2s"}
    tot = sum(lb.values())
    out = {}
    for p, v in lb.items():
        m = feed["kg_d"] * v / tot
        sg = y["sg"].get(p)
        out[p] = _flow(m / kg_per_bbl(sg) if sg else 0.0, m, feed["s_kg_d"] if p == "c5plus" else 0.0)
    return out


def hydrotreater(feed: dict, spec_mg_kg: float) -> dict:
    """Removes S down to the spec as H2S with stoichiometric H2 (S + H2 -> H2S); liquid volume kept."""
    s_target = feed["kg_d"] * spec_mg_kg * 1e-6
    s_rem = max(feed["s_kg_d"] - s_target, 0.0)
    # product mass m_p = feed - s_rem; spec applies to the product: solve s_left = spec * (feed - s_rem)
    if s_rem > 0:
        s_rem = (feed["s_kg_d"] - spec_mg_kg * 1e-6 * feed["kg_d"]) / (1.0 - spec_mg_kg * 1e-6)
    prod = _flow(feed["bpd"], feed["kg_d"] - s_rem, feed["s_kg_d"] - s_rem)
    return {"product": prod, "acid_gas": _flow(0.0, s_rem * M_H2S / M_S, s_rem), "h2_kg_d": s_rem * M_H2 / M_S,
            "s_removed_kg_d": s_rem}


def alkylation_unit(olefins_kg_d: float, ic4_kg_d: float, cap_bpd: float) -> dict:
    """Phillips HF alkylation mass balance of the published example: iC4 and acid oil per kg of olefin."""
    y = _load("unit_yields.json")["alkylation"]
    m = y["mass"]
    olef = m["olefin_feed_propylene"] + m["olefin_feed_butenes"]
    ic4 = (m["isobutane_in_olefin_feed"] + m["isobutane_in_saturated_butanes"] - m["isobutane_out_propane"]
           - m["isobutane_out_butane"] - m["isobutane_out_alkylate"])
    r_ic4, r_acid = ic4 / olef, m["acid_oils"] / olef
    r_alk = 1.0 + r_ic4 - r_acid                       # closed by difference (mass balance)
    kpb = kg_per_bbl(y["alkylate_sg"])
    limits = {"olefinas": olefins_kg_d, "isobutano": ic4_kg_d / r_ic4, "capacidad": cap_bpd * kpb / r_alk}
    used = max(min(limits.values()), 0.0)
    binding = min(limits, key=limits.get)
    alk = used * r_alk
    return {"olefin_used_kg_d": used, "ic4_used_kg_d": used * r_ic4, "alkylate": _flow(alk / kpb, alk, 0.0),
            "acid_oil_kg_d": used * r_acid, "binding": binding, "ratios": {"ic4_per_olefin": r_ic4, "alkylate_per_olefin": r_alk,
            "alkylate_per_olefin_published": m["alkylate"] / olef, "acid_oil_per_olefin": r_acid}}


# ------------------------------------------------------------------------------------------------------------ prices
def _price_table(prices: dict | None, today: date, warnings: list, assumptions: list) -> dict:
    from packages.costing import reference_prices as RP
    p = _load("prices.json")
    prices = prices or {}
    for k in prices:
        if k not in PRICE_KEYS:
            raise ValueError(f"prices: clave desconocida {k!r}; usa {sorted(PRICE_KEYS)} (unidades {PRICE_KEYS}).")
    table = {}

    def put(key, value, source, url, period, tier, basis):
        table[key] = {"value": value, "unit": PRICE_KEYS[key], "source": source, "url": url, "period": period,
                      "source_tier": tier, "basis": basis, "flag": FLAG, "date_read": p["date_read"]}

    fx = RP.lookup("fx_fix_mxn_usd", "MX", today=today)
    if "fx_mxn_usd" in prices:
        put("fx_mxn_usd", float(prices["fx_mxn_usd"]), "dato del usuario", None, None, "usuario", "tipo de cambio")
    elif fx.get("found"):
        put("fx_mxn_usd", fx["value"], fx["source"], fx["url"], fx["period"], "primaria", "FIX del banco de referencia")
        warnings.append(f"Los precios internos de Pemex (MXN, ene-sep 2026) se pasan a USD con el FIX {fx['value']} MXN/USD del "
                        f"{fx['period']['end']} (no es el promedio del periodo): da prices['fx_mxn_usd'] si tienes el "
                        f"promedio.")
    fxv = table.get("fx_mxn_usd", {}).get("value")
    c = p["crude"]
    for k, series in c["monthly_usd_bbl"].items():
        vals = [v for v in series if v is not None]
        put(k, sum(vals) / len(vals), c["source"], c["url"], c["period"], c["source_tier"],
            f"promedio simple de {len(vals)} meses del precio de exportación (costo de oportunidad del crudo)")
    d = p["products_domestic"]
    for k, it in d["items"].items():
        if k == "coque":
            continue
        if fxv:
            v = it["value_mmxn"] * 1e6 / it["volume_m3"] / fxv * BBL_M3
            put(k, v, d["source"], d["url"], d["period"], d["source_tier"],
                f"valor/volumen de ventas internas del SNR ({it['row']}) = "
                f"{it['value_mmxn'] * 1e6 / it['volume_m3']:.2f} MXN/m3, a USD con el tipo de cambio de la tabla")
    lp = p["lpg"]
    put("glp", sum(lp["monthly_usd_gal"]) / len(lp["monthly_usd_gal"]) * 42.0, lp["source"], lp["url"], lp["period"],
        lp["source_tier"], "promedio simple ene-sep 2026 del propano Mont Belvieu × 42 gal/bbl")
    ng = RP.lookup("natural gas", "MX", today=today)
    if ng.get("found"):
        for k in ("gas_seco", "energia"):
            put(k, ng["value"], ng["source"], ng["url"], ng["period"], "primaria",
                f"gas natural del banco de servicios ({ng.get('region_used')}): "
                + ("crédito del gas seco de refinería" if k == "gas_seco" else "precio de toda la energía en sitio"))
        if ng.get("fallback_region_used"):
            warnings.append(f"Gas natural: no hay precio para México en el banco; se usa {ng['region_used']} "
                            f"({ng['value']:.3f} USD/GJ).")
    fc = p["fixed_cost_benchmark"]
    put("costo_fijo", fc["value"], fc["source"], fc["url"], fc["period"], fc["source_tier"], fc["item"] + ". " + fc["notes"])
    for k, v in prices.items():
        if k == "fx_mxn_usd":
            continue
        if v is None or not math.isfinite(float(v)) or float(v) < 0:
            raise ValueError(f"prices[{k!r}] debe ser un número ≥ 0 en {PRICE_KEYS[k]}.")
        put(k, float(v), "dato del usuario", None, None, "usuario", "precio dado por el usuario")
    for k, t in table.items():
        if t["source"] != "dato del usuario":
            assumptions.append(f"Precio {k} = {t['value']:.4g} {t['unit']} ({t['basis']}; {t['source']}; {t['url']}; periodo "
                               f"{t['period']}) — {FLAG}.")
    table["_not_found"] = p["not_found"]
    return table


# -------------------------------------------------------------------------------------------------------------- model
def _run_core(refinery: str = "tula", period: str | None = None, year: int | None = None, crude_slate: dict | None = None,
              utilization=None, prices: dict | None = None, unit_overrides: dict | None = None,
              today: date | None = None) -> dict:
    today = today or date.today()
    warnings: list[str] = []
    assumptions: list[str] = []
    unit_overrides = dict(unit_overrides or {})
    for k in unit_overrides:
        if k not in OVERRIDE_KEYS:
            raise ValueError(f"unit_overrides: clave desconocida {k!r}; usa {list(OVERRIDE_KEYS)}.")
    key, ref, extra = _refinery(refinery)
    period = _period(ref, period, year)
    caps = _capacities(ref, extra, unit_overrides, warnings, assumptions)
    util = _utilization(utilization)
    for k, st in extra.get("unit_status", {}).items():
        if (isinstance(utilization, dict) and k in utilization) or not period.startswith(str(st["applies_to_year"])):
            continue
        util[k] = float(st["utilization"])
        caps[k]["status"] = st["status"]
        assumptions.append(f"{k}: utilización = {st['utilization']} por estado publicado «{st['status']}» ({st['source']}, "
                           f"{st['url']}); da utilization['{k}'] si ya opera.")
    cap_bpd = {k: caps[k]["mbd"] * 1000.0 * util[k] for k in UNIT_KEYS}
    slate, published_slate, is_published = _slate(ref, period, crude_slate, utilization, caps["cdu"]["mbd"], warnings,
                                                  assumptions)
    yields = _load("unit_yields.json")
    specs = yields["product_sulfur_specs_mg_kg"]
    crude_bpd = sum(slate.values())
    units: dict[str, dict] = {}
    if crude_bpd > caps["cdu"]["mbd"] * 1000.0 * util["cdu"] * (1 + 1e-9):
        warnings.append(f"El crudo ({crude_bpd / 1000:.1f} Mbd) excede la capacidad atmosférica utilizable "
                        f"({cap_bpd['cdu'] / 1000:.1f} Mbd): se procesa todo y se marca.")

    def unit(uid, feed, products, cap=None, extra_in=0.0, extra_out=0.0, **metrics):
        m_in = feed["kg_d"] + extra_in
        m_out = sum(f["kg_d"] for f in products.values()) + extra_out
        units[uid] = {"feed": feed, "products": products, "capacity_bpd": cap, "mass_in_kg_d": m_in, "mass_out_kg_d": m_out,
                      "mass_closure_rel": abs(m_in - m_out) / max(m_in, 1e-12), "metrics": metrics}

    # crude unit
    cuts, assay_checks = crude_unit(slate)
    crude = _mix(*cuts.values())
    unit("cdu", crude, dict(cuts), cap_bpd["cdu"])
    # vacuum unit
    ar = _mix(cuts["hgo"], cuts["vr"])
    ar_in, ar_bypass = _take(ar, cap_bpd["vdu"])
    x_in = ar_in["bpd"] / ar["bpd"] if ar["bpd"] > 0 else 0.0
    vgo, vr = _part(cuts["hgo"], x_in), _part(cuts["vr"], x_in)
    if ar_bypass["bpd"] > 0:
        warnings.append(f"Residuo atmosférico por encima de la capacidad de vacío: {ar_bypass['bpd'] / 1000:.1f} Mbd van "
                        f"directo a combustóleo.")
    unit("vdu", ar_in, {"vgo": vgo, "vr": vr}, cap_bpd["vdu"])
    # asphalt from vacuum residue
    if "asphalt_bpd" in unit_overrides:
        asph_bpd = float(unit_overrides["asphalt_bpd"])
        assumptions.append(f"Asfalto = {asph_bpd:.0f} bbl/d (dato del usuario).")
    else:
        asph_pub = ref["periods"][period]["products_mbd"].get("asfaltos")
        asph_bpd = 1000.0 * (asph_pub or 0.0)
        if asph_pub is None:
            warnings.append("Pemex no publica asfalto de esta refinería en el periodo (N/D o '--'): se modela sin asfalto "
                            "(supuesto marcado; da unit_overrides['asphalt_bpd'] si lo produce).")
        assumptions.append(f"Asfalto = {asph_bpd:.0f} bbl/d = producción PUBLICADA del periodo {period} (dato de entrada, no "
                           f"predicción): se retira del residuo de vacío.")
    asphalt, vr_rest = _take(vr, asph_bpd)
    if asphalt["bpd"] < asph_bpd - 1e-6:
        warnings.append("No alcanza el residuo de vacío para el asfalto publicado: se produce todo el disponible.")
    # coker, visbreaker
    cok_feed, vr_rest = _take(vr_rest, cap_bpd["coker"])
    cok = coker_unit(cok_feed) if cok_feed["kg_d"] > 0 else None
    if cok:
        unit("coker", cok_feed, {k: v for k, v in cok.items() if not k.startswith("_")}, cap_bpd["coker"],
             coke_wt_pct=cok["_wt_pct"]["coke"], feed_ccr_wt=cok["_feed_ccr_wt"])
    vis_feed, vr_rest = _take(vr_rest, cap_bpd["visbreaker"])
    if vis_feed["bpd"] > 0:
        unit("visbreaker", vis_feed, {"fuel_oil": dict(vis_feed)}, cap_bpd["visbreaker"])
        warnings.append("Reductora de viscosidad modelada como paso a combustóleo sin cambio de rendimientos: no se abrió hoy "
                        "un vector de rendimientos publicado (reduce viscosidad; su pequeña producción de gas y nafta "
                        "no se ve).")
    # FCC: straight-run VGO first, then coker gas oil
    cgo = cok["coker_gas_oil"] if cok else _flow()
    fcc_vgo, vgo_excess = _take(vgo, cap_bpd["fcc"])
    fcc_cgo, cgo_excess = _take(cgo, cap_bpd["fcc"] - fcc_vgo["bpd"])
    fcc_feed = _mix(fcc_vgo, fcc_cgo)
    yc = yields["fcc_vs_conversion"]
    if "fcc_conversion_wt_pct" in unit_overrides:
        conv = unit_overrides["fcc_conversion_wt_pct"]
        lo_c, hi_c = fcc_conversion_bounds()
        if conv is None or not lo_c <= float(conv) <= hi_c:
            raise ValueError(f"unit_overrides['fcc_conversion_wt_pct'] debe estar entre {lo_c} y {hi_c} % peso (intervalo de "
                             f"la tabla publicada de Niccum/KBR 2013; no se extrapola).")
        conv = float(conv)
    else:
        conv = yc["conversion_wt_pct"][1]
        assumptions.append(f"Conversión del FCC = {conv} % peso = caso «media» de la tabla publicada (Niccum/KBR 2013): Pemex "
                           f"no publica la conversión de sus FCC; en mode='as_operated' se calibra si hay un producto libre "
                           f"que la identifique.")
    fcc = fcc_unit(fcc_feed, conv) if fcc_feed["kg_d"] > 0 else None
    if fcc:
        unit("fcc", fcc_feed, {k: v for k, v in fcc.items() if not k.startswith("_")}, cap_bpd["fcc"],
             feed_sulfur_wt=_s_wt(fcc_feed), feed_sg=_sg(fcc_feed), conversion_wt_pct=conv,
             yields_wt_pct=fcc["_wt_pct"], yields_source=yc["source"], yields_url=yc["url"])
        if abs(_s_wt(fcc_feed) - yc["feed"]["sulfur_wt"]) > 0.25 or abs((141.5 / _sg(fcc_feed) - 131.5) - yc["feed"]["api"]) > 3:
            warnings.append(f"Rendimientos del FCC de una tabla de planta piloto con carga de {yc['feed']['api']} °API y "
                            f"{yc['feed']['sulfur_wt']} % S; la carga modelada tiene {141.5 / _sg(fcc_feed) - 131.5:.1f} °API y "
                            f"{_s_wt(fcc_feed):.2f} % S (se usan los mismos rendimientos; el azufre con el reparto del ejemplo de "
                            f"Jechura).")
    if vgo_excess["bpd"] + cgo_excess["bpd"] > 0:
        warnings.append(f"Gasóleos por encima de la capacidad del FCC: {(vgo_excess['bpd'] + cgo_excess['bpd']) / 1000:.1f} Mbd "
                        f"van a combustóleo.")
    # naphtha hydrotreating and reforming
    cgas = cok["coker_gasoline"] if cok else _flow()
    nht_feed = _mix(cuts["naphtha"], cgas)
    split = {k: v["value"] * 1000.0 * util["hds_total"] for k, v in extra.get("hds_split_mbd", {}).items()}

    def hdt(uid: str, feed: dict, spec: float, product_es: str) -> dict:
        """Hydrotreat up to the unit's published capacity; feed above it goes UNTREATED to its product (off-spec flag)."""
        cap = split.get(uid)
        treated, bypass = _take(feed, cap) if cap is not None else (feed, _flow())
        r = hydrotreater(treated, spec)
        r["product"] = _mix(r["product"], bypass)
        r["bypass"] = bypass
        unit(uid, feed, {"product": r["product"], "acid_gas": r["acid_gas"]}, cap, extra_in=r["h2_kg_d"],
             bypass_untreated_bpd=bypass["bpd"], capacity_source=extra.get("hds_split_mbd", {}).get(uid, {}).get("source"),
             capacity_basis=("capacidad propia publicada" if cap is not None else
                                                                 "sin capacidad propia publicada: comparte hds_total"))
        if bypass["bpd"] > 0:
            warnings.append(f"{uid}: la carga ({feed['bpd'] / 1000:.1f} Mbd) excede su capacidad publicada "
                            f"({cap / 1000:.1f} Mbd); {bypass['bpd'] / 1000:.1f} Mbd pasan SIN hidrotratar a {product_es}, "
                            f"que queda FUERA DE ESPECIFICACIÓN de azufre ({1e4 * _s_wt(r['product']):.0f} mg/kg).")
        return r

    nht = hdt("hds_naftas", nht_feed, specs["gasoline_avg"], "la nafta (reformadora / gasolinas)")
    ref_feed, naphtha_surplus = _take(nht["product"], cap_bpd["reformer"])
    refm = reformer_unit(ref_feed) if ref_feed["kg_d"] > 0 else None
    if refm:
        unit("reformer", ref_feed, refm, cap_bpd["reformer"])
    if naphtha_surplus["bpd"] > 0:
        warnings.append(f"Nafta hidrotratada por encima de la capacidad de reformación: {naphtha_surplus['bpd'] / 1000:.1f} Mbd "
                        f"van al pool de gasolinas sin reformar (bajo octano; en la refinería real parte iría a "
                        f"isomerización, "
                        f"no modelada).")
    # alkylation: FCC olefins with FCC + reformer isobutane
    olef = (fcc["propylene"]["kg_d"] + fcc["butylenes"]["kg_d"]) if fcc else 0.0
    ic4 = (fcc["isobutane"]["kg_d"] if fcc else 0.0) + (refm["isobutane"]["kg_d"] if refm else 0.0)
    alk = (alkylation_unit(olef, ic4, cap_bpd["alkylation_isomerization"])
           if olef > 0 and cap_bpd["alkylation_isomerization"] > 0 else None)
    if alk:
        fo = alk["olefin_used_kg_d"] / olef
        alk_feed = _flow(fo * (fcc["propylene"]["bpd"] + fcc["butylenes"]["bpd"]) + alk["ic4_used_kg_d"] / kg_per_bbl(0.5629),
                         alk["olefin_used_kg_d"] + alk["ic4_used_kg_d"], 0.0)
        unit("alkylation", alk_feed, {"alkylate": alk["alkylate"], "acid_oil": _flow(0.0, alk["acid_oil_kg_d"], 0.0)},
             cap_bpd["alkylation_isomerization"], binding_constraint=alk["binding"])
        assumptions.append("La capacidad publicada de «alquilación e isomerización» se asigna completa a alquilación (barriles "
                           "de alquilado); no hay reparto publicado.")
    # middle distillates and FCC gasoline hydrotreating
    fjet = float(unit_overrides.get("kerosene_to_jet_fraction", 1.0))
    if not 0.0 <= fjet <= 1.0:
        raise ValueError("unit_overrides['kerosene_to_jet_fraction'] debe estar entre 0 y 1.")
    if "kerosene_to_jet_fraction" not in unit_overrides:
        assumptions.append("Todo el corte de querosina (351-526 °F) va a turbosina (elección del modelo; en la refinería parte "
                           "se manda a diésel: ajústalo con unit_overrides['kerosene_to_jet_fraction']).")
    kero_jet, kero_dsl = _part(cuts["kerosene"], fjet), _part(cuts["kerosene"], 1.0 - fjet)
    jet = hdt("hds_turbosina", kero_jet, specs["jet_max"], "turbosina")
    # LCO fate (IMP 2002; Niccum 2013): LCO reaches diesel only after straight-run distillate and only within the
    # hydrodesulfurization capacity (the diesel HDS's own published capacity, else the free part of hds_total); the
    # excess goes to fuel oil as cutter stock.
    sr_dsl = _mix(cuts["lgo"], kero_dsl)
    fgas_feed = fcc["gasoline"] if fcc else _flow()
    lco = fcc["lco"] if fcc else _flow()
    # Cutter stock (Pemex eBDI glossary: fuel oil = vacuum residue + other residues, «se utilizan diluyentes para ajustar
    # las especificaciones»): a fraction of the middle-distillate pool (straight-run diesel + non-jet kerosene + LCO),
    # taken proportionally and untreated, goes to fuel oil. Pemex does not publish the share: 0 unless given/calibrated.
    fcut = float(unit_overrides.get("cutter_fraction", 0.0))
    if not 0.0 <= fcut <= 1.0:
        raise ValueError("unit_overrides['cutter_fraction'] debe estar entre 0 y 1 (fracción del destilado intermedio "
                         "enviada a combustóleo como diluente).")
    cutter = _mix(_part(sr_dsl, fcut), _part(lco, fcut))
    sr_dsl, lco = _part(sr_dsl, 1.0 - fcut), _part(lco, 1.0 - fcut)
    if fcut > 0:
        assumptions.append(f"Diluente del combustóleo = {fcut:.3f} del destilado intermedio (diésel primario + querosina no "
                           f"turbosina + LCO), proporcional y sin hidrotratar = {cutter['bpd'] / 1000:.1f} Mbd. Pemex documenta "
                           f"el uso de diluyentes (glosario eBDI) pero no publica cuánto ni cuáles.")
    if "hds_diesel" in split:
        lco_room, lco_basis = max(split["hds_diesel"] - sr_dsl["bpd"], 0.0), "capacidad propia publicada de la HDS de diésel"
    elif cap_bpd["hds_total"] > 0:
        lco_room = max(cap_bpd["hds_total"] - nht_feed["bpd"] - kero_jet["bpd"] - sr_dsl["bpd"] - fgas_feed["bpd"], 0.0)
        lco_basis = "capacidad total de hidrodesulfuración libre (no hay capacidad propia publicada de la HDS de diésel)"
    else:
        lco_room, lco_basis = float("inf"), "sin capacidad de hidrodesulfuración publicada: sin límite (marcado)"
    lco_dsl, lco_fo = _take(lco, lco_room)
    if lco_fo["bpd"] > 0:
        warnings.append(f"LCO del FCC por encima de la capacidad de hidrodesulfuración ({lco_basis}): {lco_fo['bpd'] / 1000:.1f} "
                        f"Mbd van a combustóleo como diluente, no a diésel.")
    dsl_feed = _mix(sr_dsl, lco_dsl)
    dsl = hdt("hds_diesel", dsl_feed, specs["diesel_uba_max"], "diésel")
    units["hds_diesel"]["metrics"].update(lco_to_diesel_bpd=lco_dsl["bpd"], lco_to_fuel_oil_bpd=lco_fo["bpd"],
                                          lco_limit_basis=lco_basis, cutter_fraction=fcut, cutter_to_fuel_oil_bpd=cutter["bpd"],
                                          lco_in_cutter_bpd=(fcc["lco"]["bpd"] * fcut if fcc else 0.0))
    fgas = hdt("hds_gasolina_fcc", fgas_feed, specs["gasoline_avg"], "gasolina catalítica")
    hds_feed_bpd = nht_feed["bpd"] + kero_jet["bpd"] + dsl_feed["bpd"] + fgas_feed["bpd"]
    for uid in ("hds_naftas", "hds_turbosina", "hds_diesel", "hds_gasolina_fcc"):
        units[uid]["metrics"].update(shared_capacity_key="hds_total", shared_capacity_bpd=cap_bpd["hds_total"],
                                     shared_utilization_of_capacity=(hds_feed_bpd / cap_bpd["hds_total"]
                                                                     if cap_bpd["hds_total"] > 0 else None))
    if cap_bpd["hds_total"] <= 0:
        warnings.append("No hay capacidad de hidrodesulfuración publicada: los destilados se hidrotratan a especificación "
                        "SIN límite de capacidad (supuesto del modelo, marcado); da unit_overrides['capacity_mbd']['hds_total'].")
    assumptions.append("Hidrotratadoras: toda su carga se hidrotrata a especificación salvo que exista una capacidad PROPIA "
                       "publicada para esa planta; lo que exceda esa capacidad pasa sin tratar a su producto, marcado fuera de "
                       "especificación. Nunca se pierde producto. Sin capacidad propia, se compara la carga sumada con la "
                       "capacidad total publicada (hds_total) y sólo se avisa.")
    if hds_feed_bpd > cap_bpd["hds_total"] * (1 + 1e-9):
        warnings.append(f"La carga total a hidrodesulfuración ({hds_feed_bpd / 1000:.1f} Mbd) excede la capacidad publicada "
                        f"({cap_bpd['hds_total'] / 1000:.1f} Mbd): se trata todo y se marca.")
    # pools
    pools = {
        "gasolina": _mix(refm["c5plus"] if refm else _flow(), naphtha_surplus, fgas["product"],
                         alk["alkylate"] if alk else _flow()),
        "turbosina": jet["product"],
        "diesel": dsl["product"],
        "combustoleo": _mix(ar_bypass, vr_rest, units.get("visbreaker", {}).get("products", {}).get("fuel_oil", _flow()),
                            fcc["hco"] if fcc else _flow(), vgo_excess, cgo_excess, lco_fo, cutter),
        "asfalto": asphalt,
        "coque": cok["coke"] if cok else _flow(),
    }
    lpg_parts = [cuts["light_ends"]]
    if fcc:
        lpg_parts += [fcc["propane"], fcc["n_butane"]]
        if alk:
            lo = 1.0 - alk["olefin_used_kg_d"] / olef
            lpg_parts += [_part(fcc["propylene"], lo), _part(fcc["butylenes"], lo)]
        else:
            lpg_parts += [fcc["propylene"], fcc["butylenes"]]
    if refm:
        lpg_parts += [refm["c3"], refm["n_butane"]]
    ic4_left = ic4 - (alk["ic4_used_kg_d"] if alk else 0.0)
    if ic4 > 0:
        lpg_parts.append(_flow(ic4_left / kg_per_bbl(0.5629), ic4_left, 0.0))
    if cok:
        lpg_parts.append(cok["lpg"])
    c4_parts = []
    if fcc:
        c4_parts.append(fcc["n_butane"])
        c4_parts.append(_part(fcc["butylenes"], lo) if alk else fcc["butylenes"])
    if refm:
        c4_parts.append(refm["n_butane"])
    if ic4 > 0:
        c4_parts.append(_flow(ic4_left / kg_per_bbl(0.5629), ic4_left, 0.0))
    if cok:
        c4_parts.append(cok["_lpg_c4"])
    fbut = float(unit_overrides.get("butanes_to_gas_plants_fraction", 0.0))
    if not 0.0 <= fbut <= 1.0:
        raise ValueError("unit_overrides['butanes_to_gas_plants_fraction'] debe estar entre 0 y 1.")
    butanes = _part(_mix(*c4_parts), fbut)
    pools["glp"] = _mix(*lpg_parts, _part(butanes, -1.0))
    pools["butanos"] = butanes
    # hydrogen
    h2_made = refm["h2"]["kg_d"] if refm else 0.0
    h2_used = nht["h2_kg_d"] + jet["h2_kg_d"] + dsl["h2_kg_d"] + fgas["h2_kg_d"]
    h2_external = max(h2_used - h2_made, 0.0)
    h2_surplus = max(h2_made - h2_used, 0.0)
    if h2_external > 0:
        warnings.append(f"El hidrógeno de reformación ({h2_made / 24:.0f} kg/h) no cubre ni el mínimo estequiométrico de la "
                        f"desulfuración ({h2_used / 24:.0f} kg/h): faltan {h2_external / 24:.0f} kg/h de una planta de "
                        f"hidrógeno "
                        f"(no modelada; se contabiliza como entrada externa).")
    dry = _mix(fcc["dry_gas"] if fcc else _flow(), cok["dry_gas"] if cok else _flow(), refm["c1_c2"] if refm else _flow())
    pools["gas_seco"] = _mix(dry, _flow(0.0, h2_surplus, 0.0))
    acid = _mix(nht["acid_gas"], jet["acid_gas"], dsl["acid_gas"], fgas["acid_gas"], fcc["acid_gas"] if fcc else _flow(),
                cok["acid_gas"] if cok else _flow())
    fcc_coke = fcc["coke"] if fcc else _flow()
    acid_oil = _flow(0.0, alk["acid_oil_kg_d"] if alk else 0.0, 0.0)
    # balances
    out_mass = sum(f["kg_d"] for f in pools.values()) + acid["kg_d"] + fcc_coke["kg_d"] + acid_oil["kg_d"]
    in_mass = crude["kg_d"] + h2_external
    out_s = sum(f["s_kg_d"] for f in pools.values()) + acid["s_kg_d"] + fcc_coke["s_kg_d"]
    balances = {
        "crude_kg_h": crude["kg_d"] / 24, "external_h2_kg_h": h2_external / 24, "outputs_kg_h": out_mass / 24,
        "overall_mass_closure_rel": abs(in_mass - out_mass) / in_mass,
        "sulfur_in_kg_h": crude["s_kg_d"] / 24, "sulfur_out_kg_h": out_s / 24,
        "sulfur_closure_rel": abs(crude["s_kg_d"] - out_s) / max(crude["s_kg_d"], 1e-12),
        "sulfur_distribution_pct": {**{k: 100 * f["s_kg_d"] / crude["s_kg_d"] for k, f in pools.items()},
                                    "gas_acido_a_azufre": 100 * acid["s_kg_d"] / crude["s_kg_d"],
                                    "coque_fcc_quemado_SOx": 100 * fcc_coke["s_kg_d"] / crude["s_kg_d"]},
        "sulfur_potential_t_d": acid["s_kg_d"] / 1000.0,
        "hydrogen": {"produced_reformer_kg_h": h2_made / 24, "consumed_stoichiometric_min_kg_h": h2_used / 24,
                     "surplus_to_fuel_gas_kg_h": h2_surplus / 24, "external_required_kg_h": h2_external / 24,
                     "note": "consumo = sólo S + H2 → H2S (cota inferior; la saturación de olefinas y aromáticos no se modela)"},
        "units_max_mass_closure_rel": max(u["mass_closure_rel"] for u in units.values()),
        "liquid_products_vol_pct_of_crude": 100 * (sum(pools[k]["bpd"] for k in ("gasolina", "turbosina", "diesel", "combustoleo",
                                                                          "asfalto", "glp", "butanos")) / crude_bpd),
    }
    fo_s = _s_wt(pools["combustoleo"])
    if fo_s > specs["fuel_oil_max_wt_pct"]:
        warnings.append(f"El combustóleo modelado tiene {fo_s:.2f} % de azufre > {specs['fuel_oil_max_wt_pct']} % máximo de la "
                        f"NOM-016-CRE-2016.")
    # energy and CO2
    e = yields["energy_intensity_btu_bbl"]["values"]
    cov = yields["energy_intensity_btu_bbl"]["coverage_of_sector_onsite_energy"]
    w = lambda btu_d: btu_d * BTU_J / 86400.0      # noqa: E731
    duties = {"cdu": w(e["cdu"] * crude_bpd), "vdu": w(e["vdu"] * ar_in["bpd"]),
              "coker": w(e["coking_visbreaking"] * cok_feed["bpd"]), "visbreaker": w(e["coking_visbreaking"] * vis_feed["bpd"]),
              "fcc": w(e["fcc"] * fcc_feed["bpd"]), "reformer": w(e["reformer"] * ref_feed["bpd"]),
              "alkylation": w(e["alkylation"] * (alk["alkylate"]["bpd"] if alk else 0.0)),
              "hds_naftas": w(e["hydrotreating"] * nht_feed["bpd"]), "hds_turbosina": w(e["hydrotreating"] * kero_jet["bpd"]),
              "hds_diesel": w(e["hydrotreating"] * dsl_feed["bpd"]), "hds_gasolina_fcc": w(e["hydrotreating"] * fgas_feed["bpd"])}
    e_units = sum(duties.values())
    e_total = e_units / cov
    fc = fuel_constants()
    fuel_gas_W = ((dry["kg_d"] * fc["lhv_ch4_J_kg"] + h2_surplus * fc["lhv_h2_J_kg"]) / 86400.0)
    co2_fuel = e_total * 86400 / 1e9 * fc["co2_kg_GJ_ch4"]
    co2_fcc = (fcc_coke["kg_d"] - fcc_coke["s_kg_d"]) * M_CO2 / M_C
    energy = {"duties_W": duties, "units_studied_W": e_units, "refinery_total_estimated_W": e_total,
              "specific_GJ_per_bbl": e_total * 86400 / 1e9 / crude_bpd, "fuel_gas_produced_W": fuel_gas_W,
              "fuel_gas_cover_fraction": fuel_gas_W / e_total if e_total > 0 else None,
              "basis": f"intensidad en sitio (combustible+vapor+electricidad) del DOE 2015 × carga de cada unidad; total de la "
                       f"refinería = unidades estudiadas / {cov} (cobertura publicada del 68 %)"}
    emissions = {"co2_fuel_t_d": co2_fuel / 1000, "co2_fcc_coke_burn_t_d": co2_fcc / 1000,
                 "co2_total_t_d": (co2_fuel + co2_fcc) / 1000,
                 "co2_t_per_kbbl": (co2_fuel + co2_fcc) / 1000 / (crude_bpd / 1000),
                 "co2_factor_kg_GJ": fc["co2_kg_GJ_ch4"],
                 "basis": "toda la energía en sitio como gas natural (metano, LHV y CO2 por estequiometría con polinomios NASA) + "
                          "coque del FCC quemado en el regenerador como carbono (sin el azufre): cota superior de esa parte"}
    bm = _load("prices.json")["benchmarks"]
    emissions["pemex_refining_target_2026_t_per_kbbl"] = {**bm["pemex_refining_co2_target_2026"]}
    # economics
    ptab = _price_table(prices, today, warnings, assumptions)
    econ = _economics(slate, pools, dry, h2_surplus, e_total, crude_bpd, ptab, balances, fc, warnings)
    econ["pemex_variable_margin_2025"] = bm["pemex_variable_refining_margin_2025"]
    # comparison with published production
    comp = _compare(ref, period, pools, is_published, warnings)
    pm = _process_model(key, ref, period, caps, util, units, cuts, pools, acid, dry, h2_made, h2_used, h2_external, h2_surplus,
                        fcc_coke, duties, crude, ar_bypass, vr_rest, vgo_excess, cgo_excess, naphtha_surplus, ptab, e_total,
                        co2_fuel + co2_fcc)
    product_slate = {k: {"bpd": f["bpd"], "kg_h": f["kg_d"] / 24, "sulfur_mg_kg": 1e4 * _s_wt(f),
                         "vol_pct_of_crude": 100 * f["bpd"] / crude_bpd} for k, f in pools.items()}
    unit_table = [{"unit": k, "capacity_bpd": u["capacity_bpd"], "feed_bpd": u["feed"]["bpd"], "feed_kg_h": u["feed"]["kg_d"] / 24,
                   "utilization_of_capacity": (u["feed"]["bpd"] / u["capacity_bpd"]) if u["capacity_bpd"] else None,
                   "feed_sulfur_wt_pct": _s_wt(u["feed"]), "mass_closure_rel": u["mass_closure_rel"],
                   "products_bpd": {p: f["bpd"] for p, f in u["products"].items()},
                   "products_kg_h": {p: f["kg_d"] / 24 for p, f in u["products"].items()}, **u["metrics"]}
                  for k, u in units.items()]
    for k in ("cdu", "vdu", "coker", "fcc", "reformer"):
        if k in units and caps.get(k, {}).get("status"):
            unit_table[[r["unit"] for r in unit_table].index(k)]["status"] = caps[k]["status"]
    # evidence
    reasons = []
    ok_units = balances["units_max_mass_closure_rel"] < 1e-9
    ok_s = balances["sulfur_closure_rel"] < 1e-9
    ok_m = balances["overall_mass_closure_rel"] < 1e-9
    if not ok_units:
        reasons.append("algún balance de masa por unidad no cierra")
    if not ok_s:
        reasons.append("el balance de azufre no cierra")
    if not ok_m:
        reasons.append("el balance global de masa no cierra")
    maya = assay_checks.get("maya")
    evidence = {"verified": ok_units and ok_s and ok_m, "unit_mass_closure_max_rel": balances["units_max_mass_closure_rel"],
                "overall_mass_closure_rel": balances["overall_mass_closure_rel"], "sulfur_closure_rel": balances["sulfur_closure_rel"],
                "assay_checks": assay_checks,
                "maya_vr_ccr_estimate_vs_published_rel": (_rel(maya["vr_ccr_wt_estimated_all_ccr_in_vr"],
                                                               maya["vr_ccr_wt_published"]) if maya else None),
                "fcc_sulfur_recovery_sum_pct": fcc["_sulfur_recovery_sum_pct"] if fcc else None,
                "reasons": reasons,
                "note": "verified certifica el cierre numérico de los balances de ESTA corrida, no que los rendimientos fijos "
                        "publicados representen a la refinería: eso lo mide published_comparison."}
    for c, chk in assay_checks.items():
        if abs(chk["mass_from_cuts_vs_whole_crude_api_rel"]) > 0.002:
            warnings.append(f"Ensaye {c}: la masa sumada de los cortes difiere {100 * chk['mass_from_cuts_vs_whole_crude_api_rel']:+.2f} % "
                            f"de la del crudo entero por su °API (inconsistencia de la tabla publicada); el modelo usa "
                            f"la suma "
                            f"de los cortes.")
        if abs(chk["ar_651plus_vol_pct_published"] - chk["hgo_plus_vr_vol_pct"]) > 0.05:
            warnings.append(f"Ensaye {c}: el residuo 651+ °F publicado ({chk['ar_651plus_vol_pct_published']} % vol) no es la "
                            f"suma de gasóleo pesado + residuo de vacío ({chk['hgo_plus_vr_vol_pct']:.1f} % vol); el "
                            f"modelo usa "
                            f"los dos cortes.")
    maya_est = (f"{assay_checks['maya']['vr_ccr_wt_estimated_all_ccr_in_vr']:.2f}" if "maya" in assay_checks else "n/d")
    for c, chk in assay_checks.items():
        if chk["vr_ccr_wt_published"] is None:
            assumptions.append(f"CCR del residuo de vacío de {c} = {chk['vr_ccr_wt_estimated_all_ccr_in_vr']:.1f} % peso, "
                               f"estimado suponiendo que todo el carbón Conradson del crudo "
                               f"({_load('crude_assays.json')['crudes'][c]['ccr_wt']} %) "
                               f"queda en el residuo de vacío (para Maya ese supuesto da "
                               f"{maya_est} vs 25.8 publicado).")
    assumptions += [
        "Coquizadora: coque = (46.88/29.3) × CCR de su carga (relación del ejemplo publicado, forma de Gary-Handwerk); gas, "
        "gasolina y gasóleo de coquización en las proporciones del ejemplo; azufre con el reparto publicado del ejemplo.",
        "FCC: rendimientos (% peso) contra conversión, interpolación lineal entre los tres casos de la Tabla 3 de Niccum "
        "(KBR, 2013), planta piloto con carga de 22.5 °API y 0.55 % S; propileno/butilenos por la olefinicidad publicada; "
        "iC4/nC4 y reparto de azufre del ejemplo de Jechura; el H2S es un lote adicional y los demás se escalan para "
        "cerrar masa. La carga real de Pemex es más pesada y más azufrada.",
        "LCO: entra a diésel sólo después del destilado primario y dentro de la capacidad de hidrodesulfuración publicada "
        "(propia de la HDS de diésel si existe, si no la libre de hds_total); el exceso va a combustóleo (Laredo et al., "
        "IMP 2002: el uso de ACL en diésel en México está restringido).",
        "Reformadora: vector fijo del ejemplo publicado a RON 98 (carga 51.7 °API); el H2S del ejemplo se excluye.",
        "Hidrotratamientos: quitan azufre hasta la NOM-016-CRE-2016 (gasolina 30 mg/kg promedio, diésel 15 mg/kg, turbosina "
        "3000 mg/kg) con H2 estequiométrico; el volumen líquido se conserva (sin ganancia de volumen por hidrogenación).",
        f"Densidad de los ligeros C4- = {_load('crude_assays.json')['light_ends_sg']['value']} (n-butano); el ensaye no "
        f"la publica.",
        "Gas licuado = ligeros C4- del crudo + C3/C4 del FCC, reformadora y coquizadora no alquilados (todo "
        "propano-butano).",
        "Las unidades aguas abajo toman toda la carga disponible hasta su capacidad publicada × utilization (ruteo de "
        "máxima "
        "conversión), salvo que des utilization por unidad.",
    ]
    return {"refinery": key, "name": ref["name"], "period": period, "crude_slate_bpd": slate, "crude_bpd": crude_bpd,
            "published_crude_slate_bpd": published_slate, "process_model": pm, "product_slate": product_slate,
            "unit_table": unit_table, "capacities_mbd": caps, "balances": balances, "energy": energy, "emissions": emissions,
            "economics": econ, "published_comparison": comp, "assumptions": assumptions, "warnings": warnings,
            "not_included": list(NOT_INCLUDED), "method": METHOD, "evidence": evidence, "today": today.isoformat()}


def _economics(slate, pools, dry, h2_surplus, e_total_W, crude_bpd, ptab, balances, fc, warnings) -> dict:
    missing, rev = [], {}
    for k in ("gasolina", "turbosina", "diesel", "combustoleo", "asfalto", "glp", "butanos"):
        pk = "glp" if k == "butanos" else k
        if pk in ptab:
            rev[k] = pools[k]["bpd"] * ptab[pk]["value"]
        elif pools[k]["bpd"] > 0:
            missing.append(k)
    coke_t = pools["coque"]["kg_d"] / 1000.0
    if "coque" in ptab:
        rev["coque"] = coke_t * ptab["coque"]["value"]
    elif coke_t > 0:
        missing.append("coque")
    s_t = balances["sulfur_potential_t_d"]
    if "azufre" in ptab:
        rev["azufre"] = s_t * ptab["azufre"]["value"]
    elif s_t > 0:
        missing.append("azufre")
    gas_GJ_d = (dry["kg_d"] * fc["lhv_ch4_J_kg"] + h2_surplus * fc["lhv_h2_J_kg"]) / 1e9
    if "gas_seco" in ptab:
        rev["gas_seco"] = gas_GJ_d * ptab["gas_seco"]["value"]
    else:
        missing.append("gas_seco")
    crude_cost, cmiss = 0.0, []
    for c, b in slate.items():
        if c in ptab:
            crude_cost += b * ptab[c]["value"]
        else:
            cmiss.append(c)
    if cmiss:
        return {"complete": False, "missing_prices": cmiss + missing,
                "reason": f"Falta el precio de los crudos {cmiss}: sin él no hay margen. Dalo en prices."}
    revenue = sum(rev.values())
    gross = (revenue - crude_cost) / crude_bpd
    energy_GJ_d = e_total_W * 86400 / 1e9
    out = {"complete": not missing, "missing_prices": missing,
           "revenue_usd_d": rev, "crude_cost_usd_d": crude_cost, "gross_margin_usd_bbl": gross,
           "crude_value_usd_bbl": crude_cost / crude_bpd, "energy_GJ_d": energy_GJ_d}
    if missing:
        warnings.append(f"Sin precio publicado para {missing}: su valor NO se suma a los ingresos (margen conservador). "
                        f"Dalo en prices ({ {m: PRICE_KEYS.get(m) for m in missing} }).")
    if "energia" in ptab:
        out["energy_cost_usd_bbl"] = energy_GJ_d * ptab["energia"]["value"] / crude_bpd
    else:
        out["energy_cost_usd_bbl"] = None
        missing.append("energia")
    out["fixed_cost_usd_bbl"] = ptab["costo_fijo"]["value"] if "costo_fijo" in ptab else None
    if out["energy_cost_usd_bbl"] is not None and out["fixed_cost_usd_bbl"] is not None:
        out["cash_operating_cost_usd_bbl"] = out["energy_cost_usd_bbl"] + out["fixed_cost_usd_bbl"]
        out["net_margin_usd_bbl"] = gross - out["cash_operating_cost_usd_bbl"]
        out["annual_usd"] = out["net_margin_usd_bbl"] * crude_bpd * 365.0
        out["net_margin_if_benchmark_includes_energy_usd_bbl"] = gross - out["fixed_cost_usd_bbl"]
    else:
        out["net_margin_usd_bbl"] = None
        out["annual_usd"] = None
    out["annual_gross_usd"] = gross * crude_bpd * 365.0
    out["prices"] = {k: v for k, v in ptab.items() if not k.startswith("_")}
    out["prices_not_found"] = ptab.get("_not_found", [])
    out["basis"] = ("Rentabilidad = margen de OPERACIÓN (la refinería ya existe; no se calcula VPN de construirla). Margen bruto = "
                    "(ingresos por productos − costo del crudo) / barril de crudo; costo en efectivo = energía en "
                    "sitio estimada × "
                    "precio del gas natural + gasto de operación de referencia por barril (Valero, Costa del Golfo); "
                    "anual = "
                    "margen neto × crudo promedio del periodo × 365 (el promedio publicado ya es por día calendario). "
                    "El costo "
                    "de referencia puede ya incluir energía: por eso también se da net_margin_if_benchmark_includes_energy.")
    return out


def _compare(ref: dict, period: str, pools: dict, is_published: bool, warnings: list) -> dict:
    pub = ref["periods"][period]["products_mbd"]
    rows = []
    pairs = (("gasolina", "gasolinas"), ("turbosina", "turbosina"), ("diesel", "diesel"), ("combustoleo", "combustoleo"),
             ("glp", "gas_licuado"), ("asfalto", "asfaltos"))
    for m, p in pairs:
        v = pub.get(p)
        model = pools[m]["bpd"] / 1000.0
        rows.append({"product": m, "published_mbd": v, "model_mbd": model,
                     "deviation_pct": (100 * (model - v) / v) if v else None,
                     "note": "dato de entrada (no es predicción)" if m == "asfalto" else None})
    liq_pub = sum(pub.get(p) or 0.0 for _, p in pairs)
    liq_mod = sum(pools[m]["bpd"] for m, p in pairs if pub.get(p) is not None) / 1000.0     # same products on both sides
    out = {"period": period, "rows": rows, "liquids_published_mbd": liq_pub, "liquids_model_mbd": liq_mod,
           "liquids_deviation_pct": 100 * (liq_mod - liq_pub) / liq_pub if liq_pub else None,
           "liquids_basis": "suma de los productos con dato publicado (los N/D se excluyen de ambos lados)",
           "not_compared": {"coque": {"published_mbd": pub.get("coque"), "model_t_d": pools["coque"]["kg_d"] / 1000,
                                      "why": "Pemex publica el coque en barriles sin densidad; el modelo da toneladas."},
                            "gas_seco": {"published_mbpce": pub.get("gas_seco_mbpce"), "model_t_d": pools["gas_seco"]["kg_d"] / 1000,
                                         "why": "Pemex lo publica en barriles de crudo equivalente sin factor publicado."}},
           "comparable": is_published,
           "level": "VALIDACIÓN contra producción publicada (sin calibrar): los rendimientos no se ajustaron a estos datos."}
    if not is_published:
        out["note"] = "La carga no es la publicada del periodo: la comparación es sólo orientativa."
    return out


# ------------------------------------------------------------------------------------------------------- process model
def _types() -> set:
    try:
        from packages.diagram.symbols import EQUIPMENT_TYPES
        return set(EQUIPMENT_TYPES)
    except Exception:            # block D not present: the contract's list
        return {"pump", "compressor", "fan", "heat_exchanger", "air_cooler", "heater", "cooler", "fired_heater", "boiler",
                "cooling_tower", "vessel", "flash_drum", "separator_3phase", "decanter", "tank", "column", "packed_column",
                "reactor", "cstr", "pfr", "fixed_bed_reactor", "fluidized_bed_reactor", "fermenter", "mixer", "splitter",
                "valve", "filter", "centrifuge", "screw_press", "mill", "conveyor", "dryer", "evaporator", "crystallizer",
                "cyclone", "membrane", "adsorber", "flare", "stack", "generic"}


def _process_model(key, ref, period, caps, util, units, cuts, pools, acid, dry, h2_made, h2_used, h2_external, h2_surplus,
                   fcc_coke, duties, crude, ar_bypass, vr_rest, vgo_excess, cgo_excess, naphtha_surplus, ptab, e_total,
                   co2_kg_d) -> dict:
    sections = [{"id": "crudo", "label_es": "Recibo de crudo"}, {"id": "destilacion", "label_es": "Destilación"},
                {"id": "conversion", "label_es": "Conversión de residuos y gasóleos"},
                {"id": "gasolinas", "label_es": "Naftas y gasolinas"},
                {"id": "hidrotratamiento", "label_es": "Hidrodesulfuración de destilados"},
                {"id": "servicios", "label_es": "Hidrógeno, gas combustible y azufre"},
                {"id": "productos", "label_es": "Productos"}]
    U = []

    def add(uid, label, typ, section, cap_key=None, model_key=None, duty_key=None, bandwidth=None, **metrics):
        u = units.get(model_key or uid)
        params = {}
        if cap_key:
            params["capacity_bpd"] = caps[cap_key]["mbd"] * 1000.0
            params["capacity_source"] = caps[cap_key]["source"]
            params["max_utilization_fraction"] = util[cap_key]
            if caps[cap_key].get("status"):
                params["status"] = caps[cap_key]["status"]
        if u:
            params["feed_bpd"] = u["feed"]["bpd"]
            params["feed_kg_h"] = u["feed"]["kg_d"] / 24
            if u["capacity_bpd"]:
                params["utilization_of_capacity"] = u["feed"]["bpd"] / u["capacity_bpd"]
            m = u["metrics"]
            if "shared_capacity_key" in m:                     # hydrotreaters
                params["capacity_basis"] = m["capacity_basis"]
                params["shared_capacity_bpd"] = m["shared_capacity_bpd"]
                params["shared_utilization_of_capacity"] = m["shared_utilization_of_capacity"]
                params["bypass_untreated_bpd"] = m["bypass_untreated_bpd"]
                if u["capacity_bpd"] is not None:
                    params["capacity_bpd"] = u["capacity_bpd"]
                    params["capacity_source"] = m.get("capacity_source")
                else:
                    params.pop("capacity_bpd", None)
                    params["capacity_source"] = (f"sin capacidad propia publicada; capacidad total de hidrodesulfuración: "
                                                 f"{caps['hds_total']['source']}")
                    params["utilization_of_capacity"] = m["shared_utilization_of_capacity"]
                    params["utilization_basis"] = "compartida: carga de todas las hidrotratadoras / capacidad hds_total"
            metrics = {**metrics, "mass_closure_rel": u["mass_closure_rel"]}
        duty = {}
        if duty_key and u:
            duty = {"energy_W": duties[duty_key]}
            params["bandwidth_process"] = bandwidth
            params["energy_basis_es"] = ("NO medida: intensidad típica actual (CT) del DOE 2015 para este proceso × carga "
                                         "modelada; una comparación contra la misma banda CT da brecha cero por "
                                         "construcción")
            rate = u["products"]["alkylate"]["bpd"] if bandwidth == "alkylation" else u["feed"]["bpd"]
            metrics = {**metrics, ("production_bpd" if bandwidth == "alkylation" else "throughput_bpd"): rate}
        U.append({"id": uid, "label_es": label, "type": typ, "section": section, "params": params,
                  "duties_W": duty, "metrics": metrics})

    add("tanque_crudo", "Tanques de crudo", "tank", "crudo")
    add("horno_atm", "Calentador de carga atmosférica", "fired_heater", "destilacion")
    add("cdu", "Destilación atmosférica", "column", "destilacion", "cdu", duty_key="cdu",
        bandwidth="atmospheric_crude_distillation")
    add("horno_vacio", "Calentador de vacío", "fired_heater", "destilacion")
    add("vdu", "Destilación al vacío", "column", "destilacion", "vdu", duty_key="vdu", bandwidth="vacuum_crude_distillation")
    if "coker" in units:
        add("horno_coquizadora", "Calentador de coquizadora", "fired_heater", "conversion")
        add("coquizadora", "Coquización retardada (tambores)", "vessel", "conversion", "coker", "coker", duty_key="coker",
            bandwidth="coking_visbreaking")
    if "visbreaker" in units:
        add("reductora_viscosidad", "Reductora de viscosidad", "fired_heater", "conversion", "visbreaker", "visbreaker",
            duty_key="visbreaker", bandwidth="coking_visbreaking")
    if "fcc" in units:
        add("fcc", "Desintegración catalítica (FCC)", "fluidized_bed_reactor", "conversion", "fcc", duty_key="fcc",
            bandwidth="fluid_catalytic_cracking")
        add("fcc_fraccionadora", "Fraccionadora y planta de gas del FCC", "column", "conversion")
    add("hds_naftas", "Hidrotratadora de naftas", "fixed_bed_reactor", "gasolinas", "hds_total", duty_key="hds_naftas",
        bandwidth="hydrotreating")
    if "reformer" in units:
        add("reformadora", "Reformación catalítica", "fixed_bed_reactor", "gasolinas", "reformer", "reformer",
            duty_key="reformer", bandwidth="catalytic_reforming")
    if "alkylation" in units:
        add("alquilacion", "Alquilación", "reactor", "gasolinas", "alkylation_isomerization", "alkylation",
            duty_key="alkylation", bandwidth="alkylation")
    add("hds_gasolina_fcc", "Postratamiento de gasolina catalítica", "fixed_bed_reactor", "gasolinas", "hds_total",
        duty_key="hds_gasolina_fcc", bandwidth="hydrotreating")
    add("hds_turbosina", "Hidrodesulfuradora de turbosina", "fixed_bed_reactor", "hidrotratamiento", "hds_total",
        duty_key="hds_turbosina", bandwidth="hydrotreating")
    add("hds_diesel", "Hidrodesulfuradora de diésel", "fixed_bed_reactor", "hidrotratamiento", "hds_total",
        duty_key="hds_diesel", bandwidth="hydrotreating")
    add("cabezal_h2", "Cabezal de hidrógeno", "mixer", "servicios", produced_kg_h=h2_made / 24, consumed_min_kg_h=h2_used / 24,
        external_kg_h=h2_external / 24)
    add("gas_combustible", "Sistema de gas combustible", "mixer", "servicios")
    add("planta_azufre", "Aminas y plantas de azufre (Claus)", "reactor", "servicios",
        sulfur_potential_t_d=acid["s_kg_d"] / 1000)
    add("chimenea", "Chimeneas de calentadores y regenerador", "stack", "servicios", co2_t_d=co2_kg_d / 1000,
        total_energy_W=e_total)
    for pid, label, typ in (("pool_gasolinas", "Mezcla y tanques de gasolinas", "tank"), ("tanque_turbosina", "Turbosina", "tank"),
                            ("tanque_diesel", "Diésel UBA", "tank"), ("tanque_combustoleo", "Combustóleo", "tank"),
                            ("esferas_glp", "Gas licuado", "tank"), ("tanque_asfalto", "Asfalto", "tank"),
                            ("patio_coque", "Manejo de coque", "conveyor")):
        add(pid, label, typ, "productos")
    allowed = _types()
    for u in U:
        assert u["type"] in allowed, u["type"]
    ids = {u["id"] for u in U}
    S = []

    def price_of(k):
        t = ptab.get(k)
        return {"value": t["value"], "unit": t["unit"], "source": t["source"]} if t else None

    def st(sid, label, fr, to, f, kind="material", product=None, cut=None):
        if f["kg_d"] <= 0 and f["bpd"] <= 0:
            return
        fr = fr if fr in ids else None
        to = to if to in ids else None
        s = {"id": sid, "label_es": label, "from": fr, "to": to, "kind": kind, "flow_kg_h": f["kg_d"] / 24}
        if f["bpd"] > 0:
            s["flow_bpd"] = f["bpd"]
        if f["kg_d"] > 0:
            s["sulfur_wt_pct"] = _s_wt(f)
        if cut:
            s["cut"] = cut
        if product and price_of(product):
            s["price"] = price_of(product)
        S.append(s)

    st("s_crudo", "Crudo", None, "tanque_crudo", crude, product=None)
    st("s_crudo_carga", "Carga de crudo", "tanque_crudo", "horno_atm", crude)
    st("s_crudo_caliente", "Crudo caliente", "horno_atm", "cdu", crude)
    st("s_ligeros", "Ligeros C4-", "cdu", "esferas_glp", cuts["light_ends"], cut="C4-")
    st("s_nafta_primaria", "Nafta primaria", "cdu", "hds_naftas", cuts["naphtha"], cut="C5-351 °F")
    st("s_querosina", "Querosina", "cdu", "hds_turbosina", cuts["kerosene"], cut="351-526 °F")
    fcut = units["hds_diesel"]["metrics"].get("cutter_fraction", 0.0) if "hds_diesel" in units else 0.0
    st("s_gasoleo_ligero", "Gasóleo ligero primario", "cdu", "hds_diesel", _part(cuts["lgo"], 1.0 - fcut), cut="526-651 °F")
    if fcut > 0:
        kj = units["hds_turbosina"]["feed"]["bpd"] / cuts["kerosene"]["bpd"] if cuts["kerosene"]["bpd"] > 0 else 0.0
        st("s_diluente_primario", "Destilado intermedio primario a combustóleo (diluente)", "cdu", "tanque_combustoleo",
           _part(_mix(cuts["lgo"], _part(cuts["kerosene"], 1.0 - kj)), fcut))
    vdu_feed = units["vdu"]["feed"]
    st("s_residuo_atm", "Residuo atmosférico", "cdu", "horno_vacio", vdu_feed, cut="651+ °F")
    st("s_residuo_atm_caliente", "Residuo atmosférico caliente", "horno_vacio", "vdu", vdu_feed)
    st("s_residuo_atm_excedente", "Residuo atmosférico excedente", "cdu", "tanque_combustoleo", ar_bypass)
    vprod = units["vdu"]["products"]
    st("s_gasoleo_vacio", "Gasóleo de vacío", "vdu", "fcc", vprod["vgo"], cut="651-1000 °F")
    st("s_gasoleo_vacio_excedente", "Gasóleo excedente", "vdu", "tanque_combustoleo", vgo_excess)
    if "coker" in units:
        cf = units["coker"]["feed"]
        cp = units["coker"]["products"]
        st("s_residuo_vacio_coq", "Residuo de vacío a coquización", "vdu", "horno_coquizadora", cf, cut="1000+ °F")
        st("s_residuo_vacio_caliente", "Residuo de vacío caliente", "horno_coquizadora", "coquizadora", cf)
        st("s_gasolina_coquizadora", "Nafta de coquización", "coquizadora", "hds_naftas", cp["coker_gasoline"])
        st("s_gasoleo_coquizadora", "Gasóleo de coquización", "coquizadora", "fcc",
           _mix(cp["coker_gas_oil"], _part(cgo_excess, -1.0)))
        st("s_gasoleo_coq_excedente", "Gasóleo de coquización excedente", "coquizadora", "tanque_combustoleo", cgo_excess)
        st("s_coque", "Coque", "coquizadora", "patio_coque", cp["coke"], product="coque")
        st("s_gas_coquizadora", "Gas de coquización", "coquizadora", "gas_combustible", cp["dry_gas"])
        st("s_glp_coquizadora", "C3/C4 de coquización", "coquizadora", "esferas_glp", cp["lpg"])
        st("s_gas_acido_coq", "Gas ácido de coquización", "coquizadora", "planta_azufre", cp["acid_gas"])
    if "visbreaker" in units:
        st("s_residuo_vacio_vis", "Residuo de vacío a reductora", "vdu", "reductora_viscosidad", units["visbreaker"]["feed"])
        st("s_combustoleo_vis", "Residuo reducido", "reductora_viscosidad", "tanque_combustoleo", units["visbreaker"]["feed"])
    st("s_residuo_vacio_combustoleo", "Residuo de vacío a combustóleo", "vdu", "tanque_combustoleo", vr_rest)
    st("s_residuo_vacio_asfalto", "Residuo de vacío a asfalto", "vdu", "tanque_asfalto", pools["asfalto"], product="asfalto")
    if "fcc" in units:
        fp = units["fcc"]["products"]
        st("s_efluente_fcc", "Efluente del reactor FCC", "fcc", "fcc_fraccionadora",
           _mix(*[f for k, f in fp.items() if k != "coke"]))
        st("s_gasolina_fcc", "Gasolina catalítica", "fcc_fraccionadora", "hds_gasolina_fcc", fp["gasoline"])
        lco_fo_bpd = (units["hds_diesel"]["metrics"].get("lco_to_fuel_oil_bpd", 0.0)
                      + units["hds_diesel"]["metrics"].get("lco_in_cutter_bpd", 0.0))
        x_fo = lco_fo_bpd / fp["lco"]["bpd"] if fp["lco"]["bpd"] > 0 else 0.0
        st("s_aceite_ciclico_ligero", "Aceite cíclico ligero (LCO) a hidrodesulfuración de diésel", "fcc_fraccionadora",
           "hds_diesel", _part(fp["lco"], 1.0 - x_fo))
        st("s_lco_combustoleo", "LCO a combustóleo (diluente y excedente de hidrodesulfuración)", "fcc_fraccionadora", "tanque_combustoleo",
           _part(fp["lco"], x_fo))
        st("s_aceite_ciclico_pesado", "Aceite cíclico pesado (HCO)", "fcc_fraccionadora", "tanque_combustoleo", fp["hco"])
        st("s_olefinas", "Propileno, butilenos e isobutano", "fcc_fraccionadora", "alquilacion",
           _mix(fp["propylene"], fp["butylenes"], fp["isobutane"]))
        st("s_glp_fcc", "Propano y n-butano del FCC", "fcc_fraccionadora", "esferas_glp", _mix(fp["propane"], fp["n_butane"]))
        st("s_gas_seco_fcc", "Gas seco del FCC", "fcc_fraccionadora", "gas_combustible", fp["dry_gas"])
        st("s_gas_acido_fcc", "Gas ácido del FCC", "fcc_fraccionadora", "planta_azufre", fp["acid_gas"])
        st("s_humos_regenerador", "Coque quemado en el regenerador", "fcc", "chimenea", fcc_coke)
    nh = units["hds_naftas"]["products"]
    if "reformer" in units:
        rp = units["reformer"]["products"]
        st("s_nafta_hidrotratada", "Nafta hidrotratada", "hds_naftas", "reformadora", units["reformer"]["feed"])
        st("s_reformado", "Reformado", "reformadora", "pool_gasolinas", rp["c5plus"], product="gasolina")
        st("s_h2_reformadora", "Hidrógeno de reformación", "reformadora", "cabezal_h2", rp["h2"], kind="utility")
        st("s_gas_reformadora", "Gas C1-C2 de reformación", "reformadora", "gas_combustible", rp["c1_c2"])
        st("s_glp_reformadora", "C3/C4 de reformación", "reformadora", "esferas_glp", _mix(rp["c3"], rp["n_butane"]))
        st("s_isobutano_reformadora", "Isobutano de reformación", "reformadora", "alquilacion", rp["isobutane"])
    st("s_nafta_sin_reformar", "Nafta hidrotratada sin reformar", "hds_naftas", "pool_gasolinas", naphtha_surplus,
       product="gasolina")
    st("s_gas_acido_nht", "Gas ácido de hidrotratamiento de naftas", "hds_naftas", "planta_azufre", nh["acid_gas"])
    if "alkylation" in units:
        st("s_alquilado", "Alquilado", "alquilacion", "pool_gasolinas", units["alkylation"]["products"]["alkylate"],
           product="gasolina")
    for uid, prod, tgt, lab, pk in (("hds_gasolina_fcc", "product", "pool_gasolinas", "Gasolina catalítica desulfurada", "gasolina"),
                                    ("hds_turbosina", "product", "tanque_turbosina", "Turbosina", "turbosina"),
                                    ("hds_diesel", "product", "tanque_diesel", "Diésel UBA", "diesel")):
        st(f"s_{uid}_producto", lab, uid, tgt, units[uid]["products"][prod], product=pk)
        st(f"s_{uid}_gas_acido", f"Gas ácido ({lab.lower()})", uid, "planta_azufre", units[uid]["products"]["acid_gas"])
    st("s_gasolinas", "Gasolinas", "pool_gasolinas", None, pools["gasolina"], product="gasolina")
    st("s_turbosina", "Turbosina", "tanque_turbosina", None, pools["turbosina"], product="turbosina")
    st("s_diesel", "Diésel", "tanque_diesel", None, pools["diesel"], product="diesel")
    st("s_combustoleo", "Combustóleo", "tanque_combustoleo", None, pools["combustoleo"], product="combustoleo")
    st("s_glp", "Gas licuado", "esferas_glp", None, pools["glp"], product="glp")
    st("s_butanos", "Mezcla de butanos a complejos procesadores de gas", "esferas_glp", None, pools["butanos"], product="glp")
    st("s_asfalto", "Asfalto", "tanque_asfalto", None, pools["asfalto"], product="asfalto")
    st("s_coque_venta", "Coque", "patio_coque", None, pools["coque"], product="coque")
    st("s_azufre", "Azufre potencial (en H2S)", "planta_azufre", None, _flow(0.0, acid["s_kg_d"], acid["s_kg_d"]),
       product="azufre")
    st("s_h2_externo", "Hidrógeno de planta de H2 (faltante)", None, "cabezal_h2", _flow(0.0, h2_external, 0.0), kind="utility")
    for uid in ("hds_naftas", "hds_gasolina_fcc", "hds_turbosina", "hds_diesel"):
        h2 = units[uid]["mass_in_kg_d"] - units[uid]["feed"]["kg_d"]
        st(f"s_h2_a_{uid}", "Hidrógeno de hidrotratamiento", "cabezal_h2", uid, _flow(0.0, h2, 0.0), kind="utility")
    st("s_h2_excedente", "Hidrógeno excedente a gas combustible", "cabezal_h2", "gas_combustible",
       _flow(0.0, h2_surplus, 0.0), kind="utility")
    fg = pools["gas_seco"]
    for h in ("horno_atm", "horno_vacio", "horno_coquizadora"):
        if h in ids:
            S.append({"id": f"u_gas_comb_{h}", "label_es": "Gas combustible", "from": "gas_combustible", "to": h,
                      "kind": "utility", "note_es": "reparto del gas combustible entre calentadores no modelado"})
            S.append({"id": f"u_humos_{h}", "label_es": "Gases de combustión", "from": h, "to": "chimenea", "kind": "utility"})
    S.append({"id": "s_gas_seco", "label_es": "Gas seco de refinería", "from": "gas_combustible", "to": None, "kind": "material",
              "flow_kg_h": fg["kg_d"] / 24, "price": price_of("gas_seco"),
              "note_es": "se valora como gas natural; en la refinería se quema en sus calentadores"})
    return {"name": f"{ref['name']} — modelo de planeación ({period})", "kind": "refinery", "units": U, "streams": S,
            "sections": sections, "source": "packages.refinery.model (bloque M27_R): datos públicos de Pemex eBDI, ensayes OGJ "
            "2000, vectores de rendimiento CBEN 409 (Colorado School of Mines), intensidades DOE 2015"}


# ---------------------------------------------------------------------------------------------- as-operated calibration
CALIBRATION = (   # (parameter, how it is passed, product it is fitted to, published key, direction of the product with it)
    ("coker_utilization", "utilization.coker", "combustoleo", "combustoleo", -1),
    # cutter stock share is identified by fuel oil ONLY where fuel oil is not already used by the coker utilization
    ("cutter_fraction", "unit_overrides.cutter_fraction", "combustoleo", "combustoleo", +1),
    ("kerosene_to_jet_fraction", "unit_overrides.kerosene_to_jet_fraction", "turbosina", "turbosina", +1),
    # FCC conversion is identified by diesel: conversion = 100 - LCO - slurry, LCO lands in diesel and is the most
    # conversion-sensitive published product (KBR table: LCO 27.4 -> 16.0 wt%); gasoline is flat (43-47 wt%, not
    # identifiable), fuel oil and LPG are already claimed. It is calibrated only if no earlier parameter claimed diesel.
    ("fcc_conversion_wt_pct", "unit_overrides.fcc_conversion_wt_pct", "diesel", "diesel", -1),
    ("butanes_to_gas_plants_fraction", "unit_overrides.butanes_to_gas_plants_fraction", "glp", "gas_licuado", -1),
)
CAL_DEFAULT = {"coker_utilization": 1.0, "kerosene_to_jet_fraction": 1.0, "butanes_to_gas_plants_fraction": 0.0,
               "cutter_fraction": 0.0}


def _cal_bounds(name: str) -> tuple[float, float]:
    return fcc_conversion_bounds() if name == "fcc_conversion_wt_pct" else (0.0, 1.0)
CAL_TOL_REL = 1e-6


def _kwargs_with(base: dict, values: dict) -> dict:
    kw = {**base, "utilization": dict(base["utilization"] or {}), "unit_overrides": dict(base["unit_overrides"] or {})}
    for name, how, *_ in CALIBRATION:
        if name in values:
            where, key = how.split(".")
            kw[where][key] = values[name]
    return kw


def run_model(refinery: str = "tula", period: str | None = None, year: int | None = None, crude_slate: dict | None = None,
              utilization=None, prices: dict | None = None, unit_overrides: dict | None = None,
              today: date | None = None, mode: str = "as_operated") -> dict:
    """mode 'nameplate': every unit takes all its feed up to its published capacity (maximum conversion).
    mode 'as_operated': the quantities Pemex does not publish but whose effect shows in the published product slate of the
    period are SOLVED (bisection, one parameter per product) so the model reproduces that product; the other products are
    the validation."""
    if mode not in MODES:
        raise ValueError(f"mode = {mode!r}: usa 'as_operated' (como opera, calibrado a la producción publicada) o 'nameplate'.")
    if isinstance(utilization, (int, float)) and not isinstance(utilization, bool):
        util_dict, util_num = None, utilization
    else:
        util_dict, util_num = utilization, None
    if util_dict is not None and not isinstance(util_dict, dict):
        raise ValueError("utilization: da un número (fracción de la capacidad atmosférica) o un dict por unidad.")
    base = {"refinery": refinery, "period": period, "year": year, "crude_slate": crude_slate,
            "utilization": util_dict if util_num is None else None, "prices": prices, "unit_overrides": unit_overrides,
            "today": today}
    if util_num is not None:
        res = _run_core(**{**base, "utilization": util_num})
        if mode == "as_operated":
            res["warnings"].append("mode='as_operated' no aplica con una utilización atmosférica distinta de la publicada: "
                                   "corrida a capacidad (nameplate) con esa carga.")
        res["mode"] = "nameplate"
        res["calibrated_parameters"] = {}
        res["calibration_summary"] = {"at_bound": [], "unpublished_products": [], "all_resolved": None,
                                      "text_es": "Sin calibración (modo nameplate)."}
        return res
    first = _run_core(**base)
    if mode == "nameplate":
        first["mode"] = "nameplate"
        first["calibrated_parameters"] = {}
        first["calibration_summary"] = {"at_bound": [], "unpublished_products": [], "all_resolved": None,
                                      "text_es": "Sin calibración (modo nameplate)."}
        return first
    if crude_slate is not None:
        first["warnings"].append("mode='as_operated' necesita la carga PUBLICADA del periodo para calibrar contra su producción: "
                                 "con crude_slate propio no se calibra nada (corrida a capacidad).")
        first["mode"] = "nameplate"
        first["calibrated_parameters"] = {}
        first["calibration_summary"] = {"at_bound": [], "unpublished_products": [], "all_resolved": None,
                                      "text_es": "Sin calibración (modo nameplate)."}
        return first
    pub = _load("pemex_refineries.json")["refineries"][first["refinery"]]["periods"][first["period"]]["products_mbd"]
    user_util, user_ov = dict(util_dict or {}), dict(unit_overrides or {})
    free, claimed, not_identified = [], set(), []
    reserved_diesel = False
    has_fcc = any(u["unit"] == "fcc" for u in first["unit_table"])
    for name, how, product, pkey, sign in CALIBRATION:
        where, key = how.split(".")
        given = key in (user_util if where == "utilization" else user_ov)
        target = pub.get(pkey)
        if name == "kerosene_to_jet_fraction" and target is None and pub.get("diesel") is not None:
            product, target = "diesel", pub.get("diesel")     # jet not published (N/D): the split shows in diesel
        if given:
            continue
        if name == "coker_utilization" and first["capacities_mbd"]["coker"]["mbd"] <= 0:
            continue
        if name == "fcc_conversion_wt_pct" and not has_fcc:
            continue
        if name == "cutter_fraction" and first["capacities_mbd"]["coker"]["mbd"] > 0 and target is not None:
            # with a coker, fuel oil is the coker's signal (calibrated, or fixed by the user in a what-if): never re-fit
            # it with the cutter
            not_identified.append((name, product))
            continue
        if name == "fcc_conversion_wt_pct" and any(f[0] == "cutter_fraction" for f in free):
            # the cutter already moves middle distillate (and LCO) between diesel and fuel oil: diesel is kept as the
            # independent VALIDATION of that split, so the conversion is not fitted to it
            reserved_diesel = True
            continue
        if target is None:
            continue
        if product in claimed:                  # one parameter per published product, never more
            not_identified.append((name, product))
            continue
        claimed.add(product)
        free.append((name, product, target * 1000.0, sign))
    conv_mid = _load("unit_yields.json")["fcc_vs_conversion"]["conversion_wt_pct"][1]

    def solve(free_set: list) -> tuple[dict, dict]:
        values = {name: (conv_mid if name == "fcc_conversion_wt_pct" else CAL_DEFAULT[name]) for name, *_ in free_set}

        def product_at(name: str, x: float, product: str) -> float:
            return _run_core(**_kwargs_with(base, {**values, name: x}))["product_slate"][product]["bpd"]

        status = {}
        for _ in range(4):                  # Gauss-Seidel passes (couplings: coker LPG and gas oil, FCC LCO to fuel oil)
            for name, product, target, sign in free_set:
                lo_b, hi_b = _cal_bounds(name)
                f0, f1 = product_at(name, lo_b, product), product_at(name, hi_b, product)
                lo_v, hi_v = min(f0, f1), max(f0, f1)
                if target <= lo_v or target >= hi_v:
                    x = lo_b if abs(target - f0) < abs(target - f1) else hi_b
                    hit = abs(target - (f0 if x == lo_b else f1)) <= 1e-9 * max(target, 1.0)
                    status[name] = ("resuelto (en el límite)" if hit else
                                    f"en el límite: el producto publicado no se alcanza dentro de {lo_b:g}-{hi_b:g}")
                else:
                    a, b = lo_b, hi_b
                    for _ in range(48):
                        m = 0.5 * (a + b)
                        fm = product_at(name, m, product)
                        if (fm - target) * (f1 - f0) > 0:
                            b = m
                        else:
                            a = m
                    x = 0.5 * (a + b)
                    status[name] = "resuelto"
                values[name] = x
        return values, status

    values, status = solve(free)
    # Identifiability rule for the FCC conversion: its bounds are the RANGE OF THE SOURCE TABLE, not a physical limit (a
    # fraction of 0 or 1 is a real operating point; a conversion pinned at the table edge is not). If diesel cannot be
    # reproduced inside the table, the diesel gap is not an FCC-conversion effect: conversion is NOT calibrated, it stays
    # at the source's central case, diesel stays a validation product, and the other parameters are re-solved.
    conv_unidentified = None
    if "fcc_conversion_wt_pct" in status and not status["fcc_conversion_wt_pct"].startswith("resuelto"):
        conv_unidentified = {"value_at_bound": values["fcc_conversion_wt_pct"], "status": status["fcc_conversion_wt_pct"]}
        free = [f for f in free if f[0] != "fcc_conversion_wt_pct"]
        values, status = solve(free)
    res = _run_core(**_kwargs_with(base, values))
    cal = {}
    for name, product, target, sign in free:
        model = res["product_slate"][product]["bpd"]
        cal[name] = {"value": values[name], "fitted_to": product, "published_bpd": target, "model_bpd": model,
                     "residual_rel": (model - target) / target if target else model, "status": status.get(name),
                     "bounds": list(_cal_bounds(name)), "flag": "CALIBRADO a la producción publicada (no es validación)"}
        if not str(status.get(name)).startswith("resuelto"):
            res["warnings"].append(f"Calibración de {name}: {status.get(name)} ({product}: publicado {target / 1000:.2f} Mbd, "
                                   f"modelo {model / 1000:.2f} Mbd).")
    res["assumptions"].append("mode='as_operated': " + "; ".join(
        f"{n} = {c['value']:.4f} CALIBRADO para reproducir {c['fitted_to']} publicado ({c['published_bpd'] / 1000:.3f} Mbd)"
        for n, c in cal.items()) + ". Pemex no publica estos valores; son ajustes, no validación.")
    if conv_unidentified:
        probe = _run_core(**_kwargs_with(base, {**values, "fcc_conversion_wt_pct": conv_unidentified["value_at_bound"]}))
        conv_unidentified.update(
            diesel_model_bpd_at_bound=probe["product_slate"]["diesel"]["bpd"], diesel_published_bpd=pub["diesel"] * 1000.0,
            used_value=conv_mid, text_es=(
                f"La conversión del FCC NO se calibra: ni en el borde de la tabla publicada "
                f"({conv_unidentified['value_at_bound']:g} % peso) el diésel modelado "
                f"({probe['product_slate']['diesel']['bpd'] / 1000:.1f} Mbd) alcanza el publicado ({pub['diesel']:.1f} Mbd); la "
                f"diferencia no es efecto de la conversión. Se usa el caso central publicado ({conv_mid} % peso) y el diésel "
                f"queda como validación."))
        res["warnings"].append(conv_unidentified["text_es"])
    res["fcc_conversion_identification"] = conv_unidentified or (
        {"status": "calibrada", "value": values["fcc_conversion_wt_pct"]} if "fcc_conversion_wt_pct" in values else
        {"status": "no calibrada: el diésel ya lo ajusta otro parámetro (turbosina N/D → fracción de querosina)",
         "used_value": conv_mid} if any(n == "fcc_conversion_wt_pct" for n, _ in not_identified) else
        {"status": "no calibrada: el diésel se reserva como validación del diluente (cutter_fraction calibrado al "
                   "combustóleo)", "used_value": conv_mid} if reserved_diesel else
        {"status": "no calibrada (dato del usuario, sin FCC o sin diésel publicado)"})
    for name, product in not_identified:
        res["warnings"].append(f"{name} NO se calibra: su producto identificador ({product}) ya lo ajusta otro parámetro y no "
                               f"hay otro producto publicado que lo identifique; se usa el valor por defecto (ver "
                               f"assumptions).")
    fitted = {c["fitted_to"] for c in cal.values()}
    comp = res["published_comparison"]
    for row in comp["rows"]:
        row["role"] = ("sin dato publicado (N/D)" if row["published_mbd"] is None else
                       "dato de entrada" if row["product"] == "asfalto" else
                       "ajustado" if row["product"] in fitted else "validación")
    val = [r for r in comp["rows"] if r["role"] == "validación"]
    comp["validation_rows"] = val
    comp["fitted_products"] = sorted(fitted)
    comp["level"] = ("VALIDACIÓN sólo en los productos no ajustados (validation_rows) y en el total de líquidos; los marcados "
                     "«ajustado» reproducen el dato publicado por construcción.")
    ec = res["economics"]
    if ec.get("prices"):
        pairs = (("gasolina", "gasolinas"), ("turbosina", "turbosina"), ("diesel", "diesel"), ("combustoleo", "combustoleo"),
                 ("asfalto", "asfaltos"), ("glp", "gas_licuado"))
        nd = [k for k, p in pairs if pub.get(p) is None and not (k == "asfalto" and res["product_slate"]["asfalto"]["bpd"] == 0)]
        if nd:
            ec["gross_margin_at_published_slate_usd_bbl"] = None
            ec["gross_margin_at_published_slate_note"] = (f"No se calcula: Pemex no publica {nd} (N/D) en este periodo y "
                                                          f"no se rellenan.")
        elif all(k in ec["prices"] for k, _ in pairs):
            rev = sum((pub.get(p) or 0.0) * 1000.0 * ec["prices"][k]["value"] for k, p in pairs)
            ec["gross_margin_at_published_slate_usd_bbl"] = (rev - ec["crude_cost_usd_d"]) / res["crude_bpd"]
            ec["gross_margin_at_published_slate_note"] = (
                "Diagnóstico: mismo crudo y mismos precios, pero con la producción PUBLICADA de los seis productos líquidos en "
                "lugar de la modelada (sin gas seco, coque ni azufre). La diferencia con gross_margin_usd_bbl es el efecto de "
                "los productos no ajustados (gasolinas, diésel) y del volumen de líquidos que el modelo sobrestima.")
    at_bound = [n for n, c in cal.items() if not str(c["status"]).startswith("resuelto")]
    unpublished = [r["product"] for r in comp["rows"] if r["published_mbd"] is None]
    comp["unpublished_products"] = unpublished
    text = []
    if at_bound:
        text.append("Calibración EN EL LÍMITE (el modelo no reproduce el dato publicado): " + "; ".join(
            f"{n} = {cal[n]['value']:.2f} → {cal[n]['fitted_to']} modelo {cal[n]['model_bpd'] / 1000:.2f} vs publicado "
            f"{cal[n]['published_bpd'] / 1000:.2f} Mbd" for n in at_bound) + ".")
    if unpublished:
        text.append(f"Pemex no publica (N/D) {unpublished} para este periodo: N/D no es cero; esos productos no se comparan "
                    f"ni se calibran.")
    if any(c["fitted_to"] == "diesel" for c in cal.values()):
        text.append("Como la turbosina no está publicada, la fracción de querosina a turbosina se ajustó al diésel.")
    res["calibration_summary"] = {"at_bound": at_bound, "unpublished_products": unpublished,
                                  "all_resolved": not at_bound, "text_es": " ".join(text) or "Todas las calibraciones "
                                  "reproducen su producto publicado."}
    res["mode"] = "as_operated"
    res["calibrated_parameters"] = cal
    return res
