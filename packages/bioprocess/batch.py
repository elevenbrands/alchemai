"""Batch operations (bloque M26_E2): batch / fed-batch fermentation with vessel scheduling and cost, and batch distillation.

Fermentation (unstructured, well mixed, isothermal, constant density; volume changes only by the feed):
    dV/dt     = F(t)
    d(XV)/dt  = (mu - kd)·X·V                         mu = mu_max·S/(Ks + S)·[inhibition terms]
    d(SV)/dt  = F·Sf - (mu/Yxs + m + qP/Yps)·X·V      (the qP/Yps term only if Yps is given)
    d(PV)/dt  = qP·X·V                                qP = alpha·mu + beta   (Luedeking-Piret)
Inhibition terms (chosen by name, multiplied together): Andrews/Haldane substrate inhibition S/(Ks + S + S²/Ki) (replaces
the Monod term), Levenspiel (1 - P/Pmax)^n, Aiba exponential exp(-k·P), hyperbolic Kp/(Kp + P). Every kinetic parameter is
the caller's; `published_kinetic_sets()` lists published values with their source for the caller to pass explicitly.
Integration: scipy `solve_ivp` (LSODA) on the totals (kg), with the integrals of mu·X·V and X·V carried as extra states so
that the biomass, substrate and product balances can be closed independently of the integrator's own states.

Scheduling: cycle = fill + reaction + empty + clean; total working volume = harvested broth rate × cycle (the same
relation as `biosteam.units.design_tools.size_batch` with a given loading time, which is called as a cross-check).
Cost: BioSTEAM's `NRELAnaerobicBatchBioreactor` cost items (Humbird et al. 2011, NREL/TP-5100-47764), evaluated by
BioSTEAM's own cost routine at `bst.CE` and then scaled to the caller's CEPCI.

Batch distillation (constant molar overflow, negligible hold-up on the stages and in the condenser, total condenser,
perfectly mixed still; all on a molar basis):
    simple (Rayleigh)          dx_i/dxi = x_i - y_i(x)             xi = ln(W0/W)
    N stages, constant reflux  dx_i/dxi = x_i - xD_i(x; N, R)      xD from stage-to-stage (McCabe-Thiele) calculation
    N stages, constant xD      W = W0·(xD - x0)/(xD - x); dt/dx = -W0·(xD - x0)·(R(x) + 1)/(V·(xD - x)²)   (binary)
Equilibrium: constant relative volatilities, a tabulated binary x-y curve (monotone PCHIP interpolation), or K-values of
the repo flash engine (`packages.properties.flash.independent_flash`, bubble point at the given pressure).

BioSTEAM/thermosteam are imported lazily inside `_biosteam_guard`, which restores thermo's global error flags, `bst.CE`
and the utility prices."""
from __future__ import annotations

import math
import warnings as _warnings
from contextlib import contextmanager

import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, fsolve

DATE_READ = "2026-11-16"
ATOMIC_MASS_C = 12.011          # g/mol (IUPAC abridged), only for the carbon balance
MOLAR_MASS_CO2 = 44.009         # g/mol
BALANCE_TOL = 1e-6              # relative closure demanded of the integrated balances
SOLVER_TOL = 1e-4               # relative agreement demanded between two integrations (different tolerance / method)
ANALYTIC_TOL = 1e-5             # relative agreement with a closed form, when one applies
NREL_REFERENCE_VOLUME_M3 = 3785.0   # 1 MM gal: the single size of the NREL fermentor quote behind BioSTEAM's cost item
COST_VOLUME_BAND = (0.1, 1.0)   # × reference volume outside which the cost scaling is flagged (choice of this module)
NREL_SOURCE = ("Humbird et al. 2011, NREL/TP-5100-47764, lista de equipo del área 300 (F-300 fermentador de 1 000 000 gal, "
               "A-300 agitador de 30 hp, P-300 bomba de recirculación, M-910 sistema CIP), dólares de 2009; "
               "https://www.osti.gov/servlets/purl/1013269 (leído el 2026-11-16)")

INHIBITION_MODELS = ("andrews", "levenspiel", "aiba", "hyperbolic")


# ---------------------------------------------------------------------------------------------------------------
# small helpers
def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _need(d: dict | None, key: str, what: str, where: str) -> float:
    if d is None or d.get(key) is None:
        raise ValueError(f"Falta {where}['{key}']: {what}. Es un dato del usuario; no se supone.")
    try:
        v = float(d[key])
    except (TypeError, ValueError):
        raise ValueError(f"{where}['{key}'] debe ser un número ({what}).") from None
    if not math.isfinite(v):
        raise ValueError(f"{where}['{key}'] debe ser un número finito ({what}).")
    return v


def _positive(v: float, name: str, zero_ok: bool = False) -> float:
    if v < 0 or (v == 0 and not zero_ok):
        raise ValueError(f"{name} debe ser {'≥' if zero_ok else '>'} 0 (se recibió {v}).")
    return v


def _rel(a: float, b: float, scale: float = 0.0) -> float:
    return abs(a - b) / max(abs(b), abs(scale), 1e-300)


@contextmanager
def _biosteam_guard():
    """Lazy BioSTEAM import that leaves the process as it found it: thermo's RAISE_PROPERTY_CALCULATION_ERROR flags
    (thermosteam sets them to True on import), `bst.CE`, the electricity price and the heat-utility agents' prices;
    BioSTEAM warnings raised inside are recorded in the yielded list instead of leaking."""
    from thermo.utils.mixture_property import MixtureProperty
    from thermo.utils.t_dependent_property import TDependentProperty
    flags = (TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR)
    caught: list[str] = []
    bst = None
    saved = None
    try:
        with _warnings.catch_warnings(record=True) as rec:
            _warnings.simplefilter("always")
            import biosteam as bst
            saved = {"CE": bst.CE, "power_price": bst.PowerUtility.price, "flowsheet": bst.main_flowsheet.get_flowsheet(),
                     "agents": [(a, a.heat_transfer_price, a.regeneration_price)
                                for a in list(bst.HeatUtility.heating_agents) + list(bst.HeatUtility.cooling_agents)]}
            yield bst, caught
        caught.extend(str(w.message) for w in rec
                      if not issubclass(w.category, (SyntaxWarning, DeprecationWarning, ImportWarning)))
    finally:
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR, MixtureProperty.RAISE_PROPERTY_CALCULATION_ERROR = flags
        if bst is not None and saved is not None:
            bst.CE = saved["CE"]
            bst.PowerUtility.price = saved["power_price"]
            if bst.main_flowsheet.get_flowsheet() is not saved["flowsheet"]:
                bst.main_flowsheet.set_flowsheet(saved["flowsheet"])
            for a, htp, rp in saved["agents"]:
                a.heat_transfer_price, a.regeneration_price = htp, rp


def _cepci(year: int, cepci: float | None) -> tuple[float, str]:
    from packages.costing.sslw import CEPCI_IDAES
    if cepci is not None:
        return _positive(float(cepci), "cepci"), f"CEPCI {float(cepci):g} (dado por el usuario)"
    if year not in CEPCI_IDAES:
        raise ValueError(f"La tabla CEPCI del repositorio (IDAES) cubre {min(CEPCI_IDAES)}-{max(CEPCI_IDAES)}; para {year} "
                         "da cepci=<valor del índice>.")
    return CEPCI_IDAES[year], f"CEPCI {CEPCI_IDAES[year]} ({year}, tabla de IDAES)"


# ---------------------------------------------------------------------------------------------------------------
# published kinetic data (never applied silently)
def published_kinetic_sets() -> dict:
    """Published kinetic values for ethanol fermentation, each with its source, to be passed EXPLICITLY by the caller.
    Read on 2026-11-16 in the open-access review of Oliveira et al. (2017). They are literature values of other strains
    and conditions: a design must use kinetics measured for its own strain, medium and temperature."""
    review = ("Oliveira, Stremel, Dechechi & Pereira (2017), «Kinetic Modeling of 1-G Ethanol Fermentations», en "
              "Fermentation Processes, IntechOpen, doi:10.5772/65460")
    url = "https://www.intechopen.com/chapters/52695"
    return _clean({
        "date_read": DATE_READ,
        "units_note": ("valores como se publican (h⁻¹, g/L, g/g); la herramienta pide SI: 1 h⁻¹ = 1/3600 s⁻¹, "
                       "1 g/L = 1 kg/m³, g/g = kg/kg"),
        "sets": [
            {"id": "hojo_1999_flocculating_yeast",
             "organism": "levadura floculante (Saccharomyces cerevisiae), CSTR con y sin reciclo de células",
             "model": "Monod con inhibición por etanol de Levenspiel: mu = mu_max·S/(Ks+S)·(1 − P/P*)^n; muerte kd",
             "published": {"mu_max_1_h": 0.6, "Ks_g_L": 0.57, "P_max_g_L": 80.0, "n": 1.8, "kd_1_h": 0.0054},
             "as_tool_arguments": {"mu_max_1_s": 0.6 / 3600.0, "Ks_kg_m3": 0.57, "kd_1_s": 0.0054 / 3600.0,
                                   "inhibition": [{"model": "levenspiel", "P_max_kg_m3": 80.0, "n": 1.8}]},
             "source": f"Hojo, Hokka & Maior (1999), Appl. Biochem. Biotechnol. 77-79:535-545, según las ecs. 58 y 61 de "
                       f"{review}",
             "url": url,
             "caveats": ["ajustados en cultivo CONTINUO; la revisión no indica temperatura ni azúcar",
                         "la revisión imprime además Yg = 0.014 g/g y muP = 0.065 + 2.24·muS: 2.24 g/g supera el máximo "
                         "estequiométrico 0.511 g etanol/g glucosa, así que los rendimientos NO se ofrecen aquí: "
                         "tómalos del artículo original o de tus datos"]},
            {"id": "ethanol_growth_associated_alpha",
             "organism": "levadura, producción de etanol por lotes",
             "model": "Luedeking-Piret asociado al crecimiento: qP = alpha·mu (beta = 0)",
             "published": {"alpha_g_g": 4.17},
             "as_tool_arguments": {"alpha_kg_kg": 4.17},
             "source": f"Oliveira et al. (ref. 19 de la revisión), citado junto a la ec. 28 de {review}", "url": url,
             "caveats": ["un solo valor de un trabajo por lotes; depende de la cepa y del medio"]},
            {"id": "anaerobic_biomass_yield_on_glucose",
             "organism": "S. cerevisiae y Zymomonas mobilis, anaerobio, fuente de nitrógeno NH3",
             "model": "rendimiento de biomasa sobre glucosa; biomasa CH1.8O0.5N0.2",
             "published": {"Yxs_yeast_g_g": 0.11, "Yxs_zymomonas_g_g": 0.05, "Yps_stoichiometric_max_g_g": 0.511},
             "as_tool_arguments": {"Yxs_kg_kg": 0.11},
             "source": f"Doran (Bioprocess Engineering Principles), según {review}", "url": url,
             "caveats": ["0.511 g/g es el máximo estequiométrico (Gay-Lussac), no un rendimiento observado",
                         "la revisión imprime la fórmula de la biomasa como «C1.8O0.5N0.2» (falta la H): se entiende "
                         "CH1.8O0.5N0.2, la de Doran"]},
        ],
        "typical_ranges": {
            "what": "Tabla 3 de la revisión: intervalos de parámetros publicados para fermentación alcohólica (para "
                    "comprobar órdenes de magnitud, NO para usarlos como dato)",
            "mu_max_1_h": [0.11, 0.56], "Ks_g_L": [0.07, 0.57], "P_max_linear_model_g_L": [87.0, 95.0],
            "levenspiel_n": [0.41, 4.0], "X_max_g_L": [100.0, 330.0],
            "source": f"Tabla 3 de {review}", "url": url},
        "warning": "Ninguna herramienta aplica estos valores por su cuenta: hay que pasarlos como argumentos y quedan "
                   "como supuesto del usuario.",
    })


# ---------------------------------------------------------------------------------------------------------------
# fermentation kinetics
def _parse_kinetics(k: dict | None) -> dict:
    if not isinstance(k, dict):
        raise ValueError("Falta kinetics: {mu_max_1_s, Ks_kg_m3, Yxs_kg_kg, ...} son datos cinéticos del usuario "
                         "(published_kinetic_sets() lista valores publicados con su fuente para pasarlos explícitamente).")
    p = {"mu_max": _positive(_need(k, "mu_max_1_s", "velocidad específica máxima de crecimiento, 1/s", "kinetics"),
                             "kinetics['mu_max_1_s']", zero_ok=True),
         "Ks": _positive(_need(k, "Ks_kg_m3", "constante de saturación de Monod, kg/m³", "kinetics"), "kinetics['Ks_kg_m3']"),
         "Yxs": _positive(_need(k, "Yxs_kg_kg", "rendimiento de biomasa sobre sustrato, kg/kg", "kinetics"),
                          "kinetics['Yxs_kg_kg']")}
    for key, name in (("m", "maintenance_kg_kg_s"), ("kd", "kd_1_s"), ("alpha", "alpha_kg_kg"), ("beta", "beta_kg_kg_s")):
        p[key] = _positive(float(k[name]), f"kinetics['{name}']", zero_ok=True) if k.get(name) is not None else 0.0
        p[f"{key}_given"] = k.get(name) is not None
    p["Yps"] = _positive(float(k["Yps_kg_kg"]), "kinetics['Yps_kg_kg']") if k.get("Yps_kg_kg") is not None else None
    known = {"mu_max_1_s", "Ks_kg_m3", "Yxs_kg_kg", "maintenance_kg_kg_s", "kd_1_s", "alpha_kg_kg", "beta_kg_kg_s",
             "Yps_kg_kg", "inhibition"}
    extra = sorted(set(k) - known)
    if extra:
        raise ValueError(f"kinetics tiene claves desconocidas {extra}; las válidas son {sorted(known)}.")
    terms = []
    inh = k.get("inhibition") or []
    if isinstance(inh, dict):
        inh = [inh]
    for t in inh:
        model = str(t.get("model", "")).lower()
        if model not in INHIBITION_MODELS:
            raise ValueError(f"inhibition.model debe ser uno de {list(INHIBITION_MODELS)} (se recibió {t.get('model')!r}).")
        if model == "andrews":
            terms.append(("andrews", _positive(_need(t, "Ki_kg_m3", "constante de inhibición por sustrato, kg/m³",
                                                      "inhibition"), "Ki_kg_m3")))
        elif model == "levenspiel":
            terms.append(("levenspiel", _positive(_need(t, "P_max_kg_m3", "concentración de producto que detiene el "
                                                        "crecimiento, kg/m³", "inhibition"), "P_max_kg_m3"),
                          _positive(_need(t, "n", "exponente de Levenspiel", "inhibition"), "n")))
        elif model == "aiba":
            terms.append(("aiba", _positive(_need(t, "k_m3_kg", "constante exponencial de Aiba, m³/kg", "inhibition"),
                                            "k_m3_kg")))
        else:
            terms.append(("hyperbolic", _positive(_need(t, "Kp_kg_m3", "constante de inhibición por producto, kg/m³",
                                                        "inhibition"), "Kp_kg_m3")))
    if sum(1 for t in terms if t[0] == "andrews") > 1:
        raise ValueError("Solo puede haber un término de Andrews (inhibición por sustrato).")
    if any(t[0] != "andrews" for t in terms) and not (p["alpha"] > 0 or p["beta"] > 0):
        raise ValueError("Hay inhibición por producto pero no formación de producto: da alpha_kg_kg y/o beta_kg_kg_s "
                         "(Luedeking-Piret), que son datos del usuario.")
    p["terms"] = terms
    return p


def _mu(p: dict, S: float, P: float) -> float:
    if S <= 0.0:
        return 0.0
    ki = next((t[1] for t in p["terms"] if t[0] == "andrews"), None)
    mu = p["mu_max"] * S / (p["Ks"] + S + (S * S / ki if ki else 0.0))
    for t in p["terms"]:
        if t[0] == "levenspiel":
            mu *= max(0.0, 1.0 - max(P, 0.0) / t[1]) ** t[2]
        elif t[0] == "aiba":
            mu *= math.exp(-t[1] * max(P, 0.0))
        elif t[0] == "hyperbolic":
            mu *= t[1] / (t[1] + max(P, 0.0))
    return mu


def _kinetics_text(p: dict) -> str:
    parts = ["Monod"]
    names = {"andrews": "inhibición por sustrato de Andrews S/(Ks+S+S²/Ki)", "levenspiel": "inhibición por producto de "
             "Levenspiel (1−P/Pmax)^n", "aiba": "inhibición por producto de Aiba exp(−k·P)",
             "hyperbolic": "inhibición hiperbólica por producto Kp/(Kp+P)"}
    parts += [names[t[0]] for t in p["terms"]]
    if p["alpha"] > 0 or p["beta"] > 0:
        parts.append("producto de Luedeking-Piret qP = alpha·mu + beta")
    if p["m"] > 0:
        parts.append("mantenimiento")
    if p["kd"] > 0:
        parts.append("muerte celular de primer orden")
    return ", ".join(parts)


def _parse_feed(feed: dict | None, V0: float) -> dict | None:
    if feed is None:
        return None
    F0 = _positive(_need(feed, "flow_m3_s", "caudal de alimentación (inicial si es exponencial), m³/s", "feed"),
                   "feed['flow_m3_s']")
    Sf = _positive(_need(feed, "substrate_kg_m3", "concentración de sustrato en la alimentación, kg/m³", "feed"),
                   "feed['substrate_kg_m3']")
    f = {"F0": F0, "Sf": Sf, "start": float(feed.get("start_s") or 0.0),
         "stop": float(feed["stop_s"]) if feed.get("stop_s") is not None else math.inf,
         "k": float(feed.get("exponential_rate_1_s") or 0.0),
         "Vmax": float(feed["max_volume_m3"]) if feed.get("max_volume_m3") is not None else math.inf}
    if f["start"] < 0 or f["stop"] <= f["start"]:
        raise ValueError("feed: start_s ≥ 0 y stop_s > start_s.")
    if f["Vmax"] <= V0:
        raise ValueError("feed['max_volume_m3'] debe ser mayor que el volumen inicial.")
    return f


def _integrate_fermentation(p: dict, y0: np.ndarray, feed: dict | None, t_end: float, rtol: float, method: str):
    """Piecewise integration (feed switching instants, volume limit, substrate exhaustion). Returns the list of
    (t0, t1, dense solution), the stop reason and the flags."""
    def flow(t: float, on: bool) -> float:
        if not on or feed is None:
            return 0.0
        return feed["F0"] * (math.exp(feed["k"] * (t - feed["start"])) if feed["k"] else 1.0)

    def rhs(t, y, on):
        V, MX, MS, MP = y[0], max(y[1], 0.0), y[2], y[3]
        S, P = max(MS, 0.0) / V, max(MP, 0.0) / V
        mu = _mu(p, S, P)
        qp = p["alpha"] * mu + p["beta"]
        F = flow(t, on)
        sink = (mu / p["Yxs"] + p["m"] + (qp / p["Yps"] if p["Yps"] else 0.0)) * MX
        return [F, (mu - p["kd"]) * MX, (F * feed["Sf"] if F else 0.0) - sink, qp * MX, mu * MX, MX,
                (F * feed["Sf"] if F else 0.0)]

    s_scale = max(y0[2] / y0[0], feed["Sf"] if feed else 0.0, p["Ks"])
    zero_order_sink = p["m"] > 0 or (p["Yps"] is not None and p["beta"] > 0)

    def exhausted(t, y, on):
        return y[2] / y[0] - 1e-9 * s_scale
    exhausted.terminal, exhausted.direction = True, -1

    def full(t, y, on):
        return y[0] - feed["Vmax"]
    full.terminal, full.direction = True, 1

    breaks = [0.0, t_end]
    if feed is not None:
        breaks += [b for b in (feed["start"], feed["stop"]) if 0.0 < b < t_end]
    breaks = sorted(set(breaks))
    segments, y, t, reason, volume_limited = [], np.array(y0, float), 0.0, "t_end", False
    atol = np.array([1e-12 * y0[0], 1e-12 * max(y0[1], 1e-30), 1e-12 * s_scale * y0[0], 1e-12 * s_scale * y0[0],
                     1e-12 * max(y0[1], 1e-30), 1e-9 * max(y0[1], 1e-30), 1e-12 * s_scale * y0[0]])
    i = 1
    while i < len(breaks):
        t1 = breaks[i]
        on = feed is not None and feed["start"] <= t < feed["stop"] and not volume_limited
        events = [exhausted] if (zero_order_sink and y[2] > 0) else []
        if on and math.isfinite(feed["Vmax"]):
            events.append(full)
        sol = solve_ivp(rhs, (t, t1), y, method=method, rtol=rtol, atol=atol, dense_output=True, events=events, args=(on,))
        if not sol.success:
            return segments, f"integrador: {sol.message}", False
        segments.append((t, float(sol.t[-1]), sol.sol))
        y, t = sol.y[:, -1].copy(), float(sol.t[-1])
        if sol.status == 1:
            if events and events[0] is exhausted and len(sol.t_events[0]):
                y[2] = max(y[2], 0.0)
                if on and flow(t, True) * feed["Sf"] > 0 and p["mu_max"] > 0:
                    reason = "substrate_exhausted_while_feeding"
                else:
                    reason = "substrate_exhausted"
                break
            volume_limited = True
            continue                  # same break point, feed now off
        i += 1
    return segments, reason, True


def _sample(segments: list, times: np.ndarray) -> np.ndarray:
    out = np.empty((len(times), 7))
    for j, t in enumerate(times):
        seg = next((s for s in segments if s[0] <= t <= s[1]), segments[-1])
        out[j] = seg[2](min(max(t, seg[0]), seg[1]))
    return out


def _carbon_fraction(formula: str, what: str) -> tuple[float, float]:
    """(kg C per kg, molar mass kg/mol of the formula) from the elemental formula; atomic masses of `chemicals`."""
    from chemicals.elements import molecular_weight, simple_formula_parser
    if not formula:
        raise ValueError(f"Falta la fórmula elemental de {what} (p. ej. C6H12O6, CH1.8O0.5N0.2): es un dato del usuario.")
    try:
        atoms = simple_formula_parser(str(formula))
        mw = molecular_weight(atoms)
    except Exception:  # noqa: BLE001
        raise ValueError(f"No se pudo interpretar la fórmula de {what}: {formula!r} (ejemplos: C6H12O6, "
                         "CH1.8O0.5N0.2).") from None
    if not atoms or mw <= 0:
        raise ValueError(f"Fórmula de {what} vacía: {formula!r}.")
    return atoms.get("C", 0.0) * ATOMIC_MASS_C / mw, mw / 1000.0


def monod_batch_time_closed_form(mu_max: float, Ks: float, Yxs: float, X0: float, S0: float, S: float) -> float:
    """Closed form of the batch Monod problem without maintenance, death or inhibition (X = X0 + Yxs·(S0 − S)):
    mu_max·t = (1 + Ks·Yxs/C)·ln(X/X0) + (Ks·Yxs/C)·ln(S0/S), with C = X0 + Yxs·S0. Derived by partial fractions of
    −Yxs·(Ks + S)/(S·(C − Yxs·S)) dS = mu_max dt."""
    C = X0 + Yxs * S0
    X = C - Yxs * S
    a = Ks * Yxs / C
    return ((1.0 + a) * math.log(X / X0) + a * math.log(S0 / S)) / mu_max


def batch_fermentation(kinetics: dict, initial: dict, t_end_s: float, feed: dict | None = None, target: dict | None = None,
                       stoichiometry: dict | None = None, heat: dict | None = None, agitation_W_m3: float | None = None,
                       schedule: dict | None = None, year: int = 2023, cepci: float | None = None,
                       n_points: int = 61) -> dict:
    """Batch or fed-batch fermentation profile (see the module docstring for the model), time to a target, productivity,
    CO2 and heat, and optionally the vessel schedule and BioSTEAM purchase cost. All inputs SI; nothing kinetic,
    stoichiometric, thermal or operating is assumed."""
    warnings: list[str] = []
    assumptions: list[str] = []
    not_included = ["Limitación por oxígeno y transferencia gas-líquido (kLa), pH, temperatura variable y fase de latencia.",
                    "Modelos estructurados o segregados (viabilidad, plásmidos), lisis celular y consumo endógeno tras "
                    "agotarse el sustrato.",
                    "Cambios de volumen por evaporación, muestreo, adición de base/antiespumante o arrastre en el venteo.",
                    "Esterilización, tren de inóculo y recuperación del producto."]
    p = _parse_kinetics(kinetics)
    if not isinstance(initial, dict):
        raise ValueError("Falta initial: {volume_m3, biomass_kg_m3, substrate_kg_m3, product_kg_m3} (datos del usuario).")
    V0 = _positive(_need(initial, "volume_m3", "volumen inicial de caldo, m³", "initial"), "initial['volume_m3']")
    X0 = _positive(_need(initial, "biomass_kg_m3", "concentración inicial de biomasa (inóculo), kg/m³", "initial"),
                   "initial['biomass_kg_m3']")
    S0 = _positive(_need(initial, "substrate_kg_m3", "concentración inicial de sustrato, kg/m³", "initial"),
                   "initial['substrate_kg_m3']", zero_ok=True)
    P0 = _positive(_need(initial, "product_kg_m3", "concentración inicial de producto, kg/m³ (0 si no hay)", "initial"),
                   "initial['product_kg_m3']", zero_ok=True)
    if t_end_s is None or not t_end_s > 0:
        raise ValueError("Falta t_end_s > 0: duración a simular, s (dato de operación del usuario).")
    t_end = float(t_end_s)
    f = _parse_feed(feed, V0)
    if f is None and S0 <= 0:
        raise ValueError("Sin alimentación, initial['substrate_kg_m3'] debe ser > 0.")
    n_points = int(max(5, min(n_points, 2001)))

    for name, given, text in (("maintenance_kg_kg_s", p["m_given"], "sin término de mantenimiento"),
                              ("kd_1_s", p["kd_given"], "sin muerte celular"),
                              ("alpha_kg_kg / beta_kg_kg_s", p["alpha_given"] or p["beta_given"], "sin producto modelado")):
        if not given:
            assumptions.append(f"kinetics no trae {name}: modelo {text} (estructura del modelo elegida por el usuario).")
    if (p["alpha"] > 0 or p["beta"] > 0) and p["Yps"] is None:
        assumptions.append("Sin Yps_kg_kg: el sustrato gastado en formar producto se considera incluido en Yxs y en el "
                           "mantenimiento (Yxs aparente).")
    assumptions += ["Tanque perfectamente mezclado, isotérmico y de densidad constante; el volumen solo cambia por la "
                    "alimentación.",
                    "La alimentación no trae células ni producto." if f else "Operación por lotes: sin entradas ni salidas "
                    "de líquido."]

    y0 = np.array([V0, X0 * V0, S0 * V0, P0 * V0, 0.0, 0.0, 0.0])
    segs, reason, ok = _integrate_fermentation(p, y0, f, t_end, 1e-10, "LSODA")
    reasons: list[str] = []
    if not ok:
        return _clean({"status": "not_converged", "warnings": warnings, "assumptions": assumptions,
                       "not_included": not_included, "method": "scipy.integrate.solve_ivp (LSODA)",
                       "evidence": {"verified": False, "reasons": [reason]}})
    t_stop = segs[-1][1]
    if reason == "substrate_exhausted":
        warnings.append(f"El sustrato se agotó en t = {t_stop:.6g} s (antes de t_end_s): el perfil termina ahí porque el "
                        "modelo no describe el metabolismo endógeno posterior.")
    elif reason == "substrate_exhausted_while_feeding":
        warnings.append(f"En t = {t_stop:.6g} s el consumo de orden cero (mantenimiento o producto no asociado) supera a "
                        "la alimentación y el sustrato llega a cero: el perfil termina ahí.")
    times = np.linspace(0.0, t_stop, n_points)
    Y = _sample(segs, times)
    V, MX, MS, MP, G, B, FED = (Y[:, i] for i in range(7))
    MS = np.maximum(MS, 0.0)
    X, S, P = MX / V, MS / V, MP / V
    mu = np.array([_mu(p, s, pp) for s, pp in zip(S, P)])
    qp = p["alpha"] * mu + p["beta"]
    r_x = (mu - p["kd"]) * MX                       # kg/s, viable biomass
    r_dead = p["kd"] * MX
    r_s = (mu / p["Yxs"] + p["m"] + (qp / p["Yps"] if p["Yps"] else 0.0)) * MX
    r_p = qp * MX

    # ---- balances closed with the independent integrals G = ∫mu·X·V dt and B = ∫X·V dt
    yf = Y[-1]
    Gf, Bf = yf[4], yf[5]
    consumed = y0[2] + yf[6] - max(yf[2], 0.0)
    formed_p = yf[3] - y0[3]
    bal_x = _rel(yf[1] - y0[1], Gf - p["kd"] * Bf, scale=y0[1])
    cons_model = Gf / p["Yxs"] + p["m"] * Bf + ((p["alpha"] * Gf + p["beta"] * Bf) / p["Yps"] if p["Yps"] else 0.0)
    bal_s = _rel(consumed, cons_model, scale=max(y0[2], yf[6], 1e-30))
    bal_p = _rel(formed_p, p["alpha"] * Gf + p["beta"] * Bf, scale=max(abs(formed_p), 1e-30)) if formed_p else 0.0
    bal_v = _rel(yf[0] - y0[0], (yf[6] / f["Sf"]) if f else 0.0, scale=y0[0])
    # ---- second integration (stiff implicit method, looser tolerance): step / method independence
    segs2, reason2, ok2 = _integrate_fermentation(p, y0, f, t_stop, 1e-7, "Radau")
    if ok2:
        y2 = segs2[-1][2](segs2[-1][1])
        scale = np.array([y0[0], max(yf[1], y0[1]), max(y0[2] + yf[6], 1e-30), max(yf[3], y0[2] + yf[6], 1e-30)])
        solver_diff = float(np.max(np.abs(y2[:4] - yf[:4]) / scale))
        if abs(segs2[-1][1] - t_stop) > 1e-4 * t_stop:
            solver_diff = max(solver_diff, abs(segs2[-1][1] - t_stop) / t_stop)
    else:
        solver_diff = math.inf
        reasons.append(f"la segunda integración (Radau) no terminó: {reason2}")
    evidence: dict = {"biomass_balance_rel_residual": bal_x, "substrate_balance_rel_residual": bal_s,
                      "product_balance_rel_residual": bal_p, "volume_balance_rel_residual": bal_v,
                      "second_solver_max_rel_diff": solver_diff,
                      "second_solver": "Radau, rtol 1e-7 (la principal: LSODA, rtol 1e-10)"}
    for k_, v_ in (("biomasa", bal_x), ("sustrato", bal_s), ("producto", bal_p), ("volumen", bal_v)):
        if v_ > BALANCE_TOL:
            reasons.append(f"el balance integrado de {k_} cierra a {v_:.2e} (> {BALANCE_TOL:g})")
    if solver_diff > SOLVER_TOL:
        reasons.append(f"dos integraciones distintas difieren {solver_diff:.2e} (> {SOLVER_TOL:g})")
    plain_monod = (f is None and not p["terms"] and p["m"] == 0 and p["kd"] == 0 and p["mu_max"] > 0
                   and (p["Yps"] is None or (p["alpha"] == 0 and p["beta"] == 0)))
    if plain_monod:
        # compared in S (absolute, relative to S0): t(S) is ill-conditioned once S << Ks
        def gap(s_: float) -> float:
            return monod_batch_time_closed_form(p["mu_max"], p["Ks"], p["Yxs"], X0, S0, s_) - t_stop
        lo = S0 * 1e-300
        s_cf = float(brentq(gap, lo, S0, xtol=1e-300, rtol=1e-14)) if gap(lo) > 0 else 0.0
        evidence["monod_closed_form"] = {"S_closed_form_kg_m3": s_cf, "S_simulated_kg_m3": float(S[-1]),
                                         "abs_diff_over_S0": abs(float(S[-1]) - s_cf) / S0,
                                         "formula": "mu_max·t = (1+Ks·Yxs/C)·ln(X/X0) + (Ks·Yxs/C)·ln(S0/S), C = X0+Yxs·S0"}
        if evidence["monod_closed_form"]["abs_diff_over_S0"] > ANALYTIC_TOL:
            reasons.append("el sustrato simulado no coincide con la solución cerrada de Monod")

    # ---- target
    total_supplied = y0[2] + FED
    conversion = np.where(total_supplied > 0, 1.0 - MS / np.maximum(total_supplied, 1e-300), 0.0)
    t_target, target_out = None, None
    if target is not None:
        keys = {"substrate_conversion": conversion, "product_kg_m3": P, "biomass_kg_m3": X, "substrate_kg_m3": -S}
        given = [k_ for k_ in keys if target.get(k_) is not None]
        if len(given) != 1:
            raise ValueError(f"target debe traer exactamente una de {list(keys)}.")
        key = given[0]
        goal = float(target[key]) * (-1.0 if key == "substrate_kg_m3" else 1.0)
        if key == "substrate_conversion" and not 0 < goal < 1:
            raise ValueError("target['substrate_conversion'] debe estar entre 0 y 1.")

        def value_at(t: float) -> float:
            v, mx, ms, mp, _, _, fed = _sample(segs, np.array([t]))[0]
            return {"substrate_conversion": 1.0 - max(ms, 0.0) / max(y0[2] + fed, 1e-300), "product_kg_m3": mp / v,
                    "biomass_kg_m3": mx / v, "substrate_kg_m3": -max(ms, 0.0) / v}[key] - goal
        fine = np.linspace(0.0, t_stop, 2001)
        vals = np.array([value_at(t) for t in fine])
        idx = np.where((vals[:-1] < 0) & (vals[1:] >= 0))[0]
        if vals[0] >= 0:
            t_target = 0.0
        elif len(idx):
            t_target = float(brentq(value_at, fine[idx[0]], fine[idx[0] + 1], xtol=1e-12 * t_stop, rtol=1e-12))
        else:
            warnings.append(f"El objetivo {key} = {target[key]} no se alcanza en {t_stop:.6g} s simulados.")
        target_out = {"type": key, "value": float(target[key]), "reached": t_target is not None, "time_s": t_target}

    t_ref = t_target if t_target is not None else t_stop
    ref = _sample(segs, np.array([t_ref]))[0]
    ref_state = {"time_s": t_ref, "volume_m3": ref[0], "biomass_kg_m3": ref[1] / ref[0],
                 "substrate_kg_m3": max(ref[2], 0.0) / ref[0], "product_kg_m3": ref[3] / ref[0],
                 "substrate_conversion": 1.0 - max(ref[2], 0.0) / max(y0[2] + ref[6], 1e-300)}
    cons_ref = y0[2] + ref[6] - max(ref[2], 0.0)
    out_summary = {
        "basis": "al alcanzar el objetivo" if t_target is not None else "al final del perfil",
        **ref_state,
        "biomass_formed_kg": ref[1] - y0[1], "product_formed_kg": ref[3] - y0[3], "substrate_consumed_kg": cons_ref,
        "substrate_fed_kg": ref[6],
        "observed_yield_biomass_kg_kg": (ref[1] - y0[1]) / cons_ref if cons_ref > 0 else None,
        "observed_yield_product_kg_kg": (ref[3] - y0[3]) / cons_ref if cons_ref > 0 else None,
        "product_productivity_kg_m3_s": (ref[3] - y0[3]) / (ref[0] * t_ref) if t_ref > 0 else None,
        "biomass_productivity_kg_m3_s": (ref[1] - y0[1]) / (ref[0] * t_ref) if t_ref > 0 else None,
        "productivity_definition": "masa formada / (volumen de caldo al final × tiempo de reacción); no incluye tiempos "
                                   "muertos (ver schedule)"}

    # ---- CO2 and carbon balance
    co2_rate = None
    co2_out = None
    if stoichiometry is not None:
        wS, _ = _carbon_fraction(stoichiometry.get("substrate_formula") or "", "sustrato (stoichiometry['substrate_formula'])")
        wX, _ = _carbon_fraction(stoichiometry.get("biomass_formula") or "", "biomasa (stoichiometry['biomass_formula'])")
        has_p = p["alpha"] > 0 or p["beta"] > 0
        wP, mwP = (_carbon_fraction(stoichiometry.get("product_formula") or "", "producto (stoichiometry['product_formula'])")
                   if has_p else (0.0, 1.0))
        spec = stoichiometry.get("co2")
        if spec is None:
            raise ValueError("Falta stoichiometry['co2']: 'carbon_balance' (CO2 por diferencia de carbono) o "
                             "{'mol_per_mol_product': n} o {'mol_per_kg_substrate': n} (estequiometría del usuario).")
        c_in = r_s * wS
        c_fixed = (r_x + r_dead) * wX + r_p * wP                      # kg C/s into biomass (viable + dead) and product
        dead_f = p["kd"] * Bf
        c_tot_in = consumed * wS
        c_tot_fixed = (yf[1] - y0[1] + dead_f) * wX + formed_p * wP
        if spec == "carbon_balance":
            co2_rate = (c_in - c_fixed) / (ATOMIC_MASS_C / 1000.0)    # mol/s
            co2_total = (c_tot_in - c_tot_fixed) / (ATOMIC_MASS_C / 1000.0)
            basis = "balance de carbono: C del sustrato consumido − C en biomasa − C en producto"
            assumptions.append("CO2 por diferencia de carbono: se supone que no hay otros subproductos carbonados "
                               "(glicerol, ácidos, etc.); si los hay, este CO2 está sobrestimado.")
            recovery = None
            if co2_total < -1e-9 * max(c_tot_in, 1e-30) / (ATOMIC_MASS_C / 1000.0):
                warnings.append("El balance de carbono da CO2 NEGATIVO: los rendimientos dados fijan más carbono en "
                                "biomasa y producto que el que aporta el sustrato. Revisa Yxs, alpha/beta y las fórmulas.")
                reasons.append("rendimientos incompatibles con el balance de carbono (CO2 < 0)")
        elif isinstance(spec, dict) and spec.get("mol_per_mol_product") is not None:
            if not has_p:
                raise ValueError("co2.mol_per_mol_product necesita producto modelado (alpha_kg_kg y/o beta_kg_kg_s).")
            nu = _positive(float(spec["mol_per_mol_product"]), "co2['mol_per_mol_product']", zero_ok=True)
            co2_rate = nu * r_p / mwP
            co2_total = nu * formed_p / mwP
            basis = f"{nu:g} mol CO2 por mol de producto (estequiometría del usuario)"
            recovery = (c_tot_fixed + co2_total * ATOMIC_MASS_C / 1000.0) / c_tot_in if c_tot_in > 0 else None
        elif isinstance(spec, dict) and spec.get("mol_per_kg_substrate") is not None:
            nu = _positive(float(spec["mol_per_kg_substrate"]), "co2['mol_per_kg_substrate']", zero_ok=True)
            co2_rate = nu * r_s
            co2_total = nu * consumed
            basis = f"{nu:g} mol CO2 por kg de sustrato consumido (dato del usuario)"
            recovery = (c_tot_fixed + co2_total * ATOMIC_MASS_C / 1000.0) / c_tot_in if c_tot_in > 0 else None
        else:
            raise ValueError("stoichiometry['co2'] debe ser 'carbon_balance', {'mol_per_mol_product': n} o "
                             "{'mol_per_kg_substrate': n}.")
        co2_out = {"basis": basis, "total_mol": co2_total, "total_kg": co2_total * MOLAR_MASS_CO2 / 1000.0,
                   "peak_rate_mol_s": float(np.max(co2_rate)), "carbon_mass_fractions": {"substrate": wS, "biomass": wX,
                                                                                         "product": wP if has_p else None},
                   "carbon_recovery": recovery,
                   "carbon_recovery_definition": "(C en biomasa + C en producto + C en CO2) / C del sustrato consumido; "
                                                 "juzga los rendimientos del usuario, no la integración"}
        if recovery is not None:
            evidence["carbon_recovery_of_user_yields"] = recovery
            if abs(recovery - 1.0) > 0.05:
                warnings.append(f"Los rendimientos y la estequiometría dados recuperan el {100 * recovery:.1f} % del "
                                "carbono del sustrato: fuera de 95-105 % hay subproductos no contados o rendimientos inconsistentes.")

    # ---- heat
    heat_rate = None
    heat_out = None
    if heat is not None:
        basis_h = str(heat.get("basis", "")).lower()
        if basis_h == "substrate":
            h = _positive(_need(heat, "heat_J_per_kg_substrate", "calor liberado por kg de sustrato consumido, J/kg", "heat"),
                          "heat['heat_J_per_kg_substrate']", zero_ok=True)
            heat_rate = h * r_s
            heat_total = h * consumed
            h_basis = "calor de reacción dado por el usuario por kg de sustrato consumido"
        elif basis_h == "combustion":
            hc = heat.get("heats_of_combustion_J_kg") or {}
            has_p = p["alpha"] > 0 or p["beta"] > 0
            hS = _positive(_need(hc, "substrate", "calor de combustión del sustrato, J/kg (positivo)",
                                 "heat['heats_of_combustion_J_kg']"), "calor de combustión del sustrato")
            hX = _positive(_need(hc, "biomass", "calor de combustión de la biomasa seca, J/kg (positivo)",
                                 "heat['heats_of_combustion_J_kg']"), "calor de combustión de la biomasa")
            hP = (_positive(_need(hc, "product", "calor de combustión del producto, J/kg (positivo)",
                                  "heat['heats_of_combustion_J_kg']"), "calor de combustión del producto", zero_ok=True)
                  if has_p else 0.0)
            heat_rate = hS * r_s - hX * (r_x + r_dead) - hP * r_p
            heat_total = hS * consumed - hX * (yf[1] - y0[1] + p["kd"] * Bf) - hP * formed_p
            h_basis = ("estequiometría + calores de combustión del usuario (ley de Hess, equivalente a entalpías de "
                       "formación): Q = Δhc·sustrato consumido − Δhc·biomasa formada − Δhc·producto formado")
            assumptions.append("Calor por Hess con calores de combustión: CO2 y H2O no aportan y no se cuenta la fuente de "
                               "nitrógeno (exacto si el calor de combustión de la biomasa está referido a esa misma "
                               "fuente, p. ej. NH3; si no, el usuario debe corregirlo).")
        else:
            raise ValueError("heat['basis'] debe ser 'substrate' (con heat_J_per_kg_substrate) o 'combustion' (con "
                             "heats_of_combustion_J_kg = {substrate, biomass, product}); son datos del usuario.")
        ipk = int(np.argmax(heat_rate))
        heat_out = {"basis": h_basis, "total_J": heat_total, "peak_metabolic_W": float(heat_rate[ipk]),
                    "peak_time_s": float(times[ipk]), "peak_metabolic_W_m3": float(heat_rate[ipk] / V[ipk])}
        if heat_total < 0:
            warnings.append("El calor calculado es negativo (endotérmico): revisa los calores de combustión y rendimientos.")
    agit = None
    if agitation_W_m3 is not None:
        agit = _positive(float(agitation_W_m3), "agitation_W_m3", zero_ok=True)
    cooling = None
    if heat_rate is not None:
        duty = heat_rate + (agit * V if agit is not None else 0.0)
        ipk = int(np.argmax(duty))
        cooling = {"peak_W": float(duty[ipk]), "peak_time_s": float(times[ipk]), "peak_W_m3": float(duty[ipk] / V[ipk]),
                   "includes": "calor metabólico" + (" + potencia de agitación (P/V del usuario × volumen)" if agit is not None
                                                     else " (sin agitación: no se dio agitation_W_m3)")}
        not_included.append("En el enfriamiento: calor sensible de la alimentación, evaporación, pérdidas al ambiente y "
                            "calor de la aireación.")
    else:
        not_included.append("Calor liberado y carga de enfriamiento (no se dio heat).")
    if stoichiometry is None:
        not_included.append("CO2 desprendido y balance de carbono (no se dio stoichiometry).")

    result = {
        "status": "ok", "mode": "fed_batch" if f else "batch", "kinetic_model": _kinetics_text(p),
        "profile": {"t_s": times, "volume_m3": V, "biomass_kg_m3": X, "substrate_kg_m3": S, "product_kg_m3": P,
                    "mu_1_s": mu, "substrate_conversion": conversion,
                    **({"heat_rate_W": heat_rate} if heat_rate is not None else {}),
                    **({"co2_rate_mol_s": co2_rate} if co2_rate is not None else {})},
        "stop_reason": reason, "simulated_time_s": t_stop, "target": target_out, "summary": out_summary,
        "final": {"time_s": t_stop, "volume_m3": V[-1], "biomass_kg_m3": X[-1], "substrate_kg_m3": S[-1],
                  "product_kg_m3": P[-1], "dead_biomass_kg": p["kd"] * Bf},
        "co2": co2_out, "heat": heat_out, "cooling_duty": cooling,
        "agitation": ({"power_per_volume_W_m3": agit, "peak_power_W": agit * float(np.max(V))} if agit is not None
                      else None),
    }

    if schedule is not None:
        peak_duty = cooling["peak_W"] if cooling else None
        result["schedule"] = _schedule(schedule, t_ref, float(ref[0]), float(ref[3] - y0[3]), peak_duty, agit, year,
                                       cepci, warnings, assumptions, not_included, reasons, evidence)
    evidence["verified"] = not reasons
    if reasons:
        evidence["reasons"] = reasons
    result.update(warnings=warnings, assumptions=assumptions, not_included=not_included,
                  method=("Balances de biomasa, sustrato, producto y volumen de un fermentador mezclado integrados con "
                          "scipy.integrate.solve_ivp (LSODA, rtol 1e-10; contraste con Radau); cierre de balances con las "
                          "integrales ∫mu·X·V dt y ∫X·V dt llevadas como estados"
                          + ("; programación de tanques por tiempo de ciclo y costo con las correlaciones "
                             "NRELAnaerobicBatchBioreactor de BioSTEAM" if schedule is not None else "")),
                  evidence=evidence)
    return _clean(result)


def _schedule(sch: dict, t_reaction: float, v_final_sim: float, product_per_batch_sim: float,
              peak_duty_sim: float | None, agit: float | None, year: int, cepci: float | None,
              warnings: list, assumptions: list, not_included: list, reasons: list, evidence: dict) -> dict:
    where = "schedule"
    fill = _positive(_need(sch, "fill_time_s", "tiempo de llenado de un tanque, s", where), "fill_time_s", zero_ok=True)
    empty = _positive(_need(sch, "empty_time_s", "tiempo de vaciado, s", where), "empty_time_s", zero_ok=True)
    clean = _positive(_need(sch, "clean_time_s", "tiempo de limpieza/esterilización (CIP/SIP), s", where), "clean_time_s",
                      zero_ok=True)
    wf = _need(sch, "working_volume_fraction", "fracción del volumen del tanque ocupada por el caldo al final", where)
    if not 0 < wf <= 1:
        raise ValueError("schedule['working_volume_fraction'] debe estar en (0, 1].")
    has_q, has_p = sch.get("broth_rate_m3_s") is not None, sch.get("product_rate_kg_s") is not None
    if has_q == has_p:
        raise ValueError("schedule: da exactamente uno de broth_rate_m3_s (caldo cosechado, equivalente continuo) o "
                         "product_rate_kg_s (producto formado); es la capacidad requerida por el usuario.")
    if has_q:
        q = _positive(float(sch["broth_rate_m3_s"]), "broth_rate_m3_s")
    else:
        if product_per_batch_sim <= 0:
            raise ValueError("schedule['product_rate_kg_s'] necesita que la simulación forme producto.")
        q = _positive(float(sch["product_rate_kg_s"]), "product_rate_kg_s") * v_final_sim / product_per_batch_sim
    has_n, has_v = sch.get("n_vessels") is not None, sch.get("max_vessel_volume_m3") is not None
    if has_n == has_v:
        raise ValueError("schedule: da exactamente uno de n_vessels o max_vessel_volume_m3 (decisión de diseño del usuario).")
    cycle = fill + t_reaction + empty + clean
    v_total_working = q * cycle
    if has_n:
        n = int(sch["n_vessels"])
        if n < 1 or n != float(sch["n_vessels"]):
            raise ValueError("schedule['n_vessels'] debe ser un entero ≥ 1.")
    else:
        vmax = _positive(float(sch["max_vessel_volume_m3"]), "max_vessel_volume_m3")
        n = max(1, math.ceil(v_total_working / (vmax * wf) - 1e-12))
    v_work = v_total_working / n
    v_vessel = v_work / wf
    scale = v_work / v_final_sim
    out = {"cycle_time_s": cycle, "reaction_time_s": t_reaction, "fill_time_s": fill, "empty_time_s": empty,
           "clean_time_s": clean, "broth_rate_m3_s": q, "n_vessels": n, "vessel_volume_m3": v_vessel,
           "working_volume_m3": v_work, "working_volume_fraction": wf, "total_working_volume_m3": v_total_working,
           "batches_per_vessel_per_year_8000h": 8000.0 * 3600.0 / cycle, "stagger_between_starts_s": cycle / n,
           "vessels_in_reaction_on_average": n * t_reaction / cycle,
           "scale_factor_vs_simulation": scale,
           "capacity_check_m3_s": n * v_work / cycle}
    assumptions.append(f"Programación: ciclo = llenado + reacción + vaciado + limpieza = {cycle:.6g} s; el tiempo de "
                       "reacción es el de la simulación (al objetivo si se dio); arranques escalonados cada ciclo/N; sin "
                       "tiempos de espera ni paros.")
    assumptions.append(f"El lote simulado se escala ×{scale:.6g} en volumen a concentraciones iguales (misma cinética, "
                       "alimentación proporcional); el escalamiento de mezcla y transferencia no se evalúa.")
    if n == 1:
        warnings.append("Con un solo tanque la descarga es intermitente: hace falta almacenamiento aguas abajo para un "
                        "proceso continuo.")
    agit_kw = agit * v_work / 1000.0 if agit is not None else None
    duty_w = peak_duty_sim * scale if peak_duty_sim is not None else None
    out["per_vessel"] = {"agitator_power_kW": agit_kw, "peak_cooling_duty_W": duty_w}
    evidence["schedule_capacity_rel_residual"] = _rel(n * v_work / cycle, q)
    if sch.get("cost", True):
        out["cost"] = _cost_vessels(n, v_vessel, v_work, q, duty_w, agit_kw, fill, t_reaction, empty + clean, wf, year, cepci,
                                    warnings, assumptions, not_included, reasons, evidence)
    else:
        not_included.append("Costo de compra de los fermentadores (schedule['cost'] = false).")
    return out


def _cost_vessels(n: int, v_vessel: float, v_work: float, q_m3_s: float, duty_w: float | None, agit_kw: float | None,
                  fill: float, t_rxn: float, t_dead: float, wf: float, year: int, cepci: float | None, warnings: list,
                  assumptions: list, not_included: list, reasons: list, evidence: dict) -> dict:
    ce_target, basis = _cepci(year, cepci)
    with _biosteam_guard() as (bst, caught):
        from types import SimpleNamespace

        from biosteam.units.decorators._cost import _decorated_cost
        from biosteam.units.design_tools import compute_closed_vessel_turbine_purchase_cost, size_batch
        from biosteam.units.nrel_bioreactor import NRELAnaerobicBatchBioreactor as NREL
        ce_bst = float(bst.CE)
        items = {k: v for k, v in NREL.cost_items.items() if not (k == "Heat exchangers" and duty_w is None)}
        design = {"Reactor volume": v_vessel, "Number of reactors": n, "Recirculation flow rate": q_m3_s * 3600.0 / n,
                  "Reactor duty": (duty_w or 0.0) * 3.6}          # W -> kJ/h
        power: list[float] = []
        stub = SimpleNamespace(design_results=design, baseline_purchase_costs={}, parallel={}, cost_items=items,
                               add_power_utility=power.append)
        _decorated_cost(stub)
        fbm = {k: float(NREL._F_BM_default[k]) for k in items}
        refs = {k: {"reference_size": float(v.S), "units": v._units, "basis": v._basis, "exponent": float(v.n),
                    "reference_cost_usd": float(v.cost), "reference_cepci": float(v.CE), "lower_bound": v.lb,
                    "upper_bound": v.ub} for k, v in items.items()}
        sb = size_batch(q_m3_s, t_rxn, t_dead, wf, N_reactors=n, loading_time=fill)
        alt = (float(compute_closed_vessel_turbine_purchase_cost(agit_kw * 1.34102)) if agit_kw else None)
        nrel_agit_kw = float(NREL.cost_items["Agitators"].kW) * v_vessel / float(NREL.cost_items["Agitators"].S)
    evidence["biosteam_size_batch_volume_rel_diff"] = _rel(sb["Reactor volume"], v_vessel)
    if evidence["biosteam_size_batch_volume_rel_diff"] > 1e-9:
        reasons.append("el volumen de tanque no coincide con biosteam.size_batch")
    names = {"Reactors": "Fermentadores (tanque)", "Agitators": "Agitadores", "Cleaning in place": "Sistema CIP",
             "Recirculation pumps": "Bombas de recirculación", "Heat exchangers": "Enfriadores de fermentación"}
    f_ce = ce_target / ce_bst
    equipment = []
    for k, c_one in stub.baseline_purchase_costs.items():
        count = int(stub.parallel.get(k, 1))
        cp = float(c_one) * count * f_ce
        equipment.append({"name": names.get(k, k), "biosteam_item": k, "count": count, "F_BM": fbm[k],
                          "F_BM_source": "BioSTEAM NRELAnaerobicBatchBioreactor (factor de instalación de NREL 2011)",
                          "purchase_cost_usd": cp, "base_purchase_cost_usd": cp, "unit_purchase_cost_usd": cp / count,
                          "basis": basis, "correlation": refs[k]})
    total = sum(e["purchase_cost_usd"] for e in equipment)
    lo, hi = (b * NREL_REFERENCE_VOLUME_M3 for b in COST_VOLUME_BAND)
    if not lo <= v_vessel <= hi:
        warnings.append(f"Volumen de tanque {v_vessel:.4g} m³ fuera de {lo:.4g}-{hi:.4g} m³: la correlación de BioSTEAM es "
                        f"UN punto de NREL ({NREL_REFERENCE_VOLUME_M3:g} m³, 2009) escalado con exponente 0.5 sin límites "
                        "declarados; el costo es una extrapolación.")
    for msg in caught:
        warnings.append(f"Aviso de BioSTEAM: {msg}")
    if agit_kw is not None and nrel_agit_kw > 0 and not (1 / 3 <= agit_kw / nrel_agit_kw <= 3):
        warnings.append(f"El agitador costeado es el de NREL ({nrel_agit_kw:.4g} kW por tanque, 5.9 W/m³, fermentación "
                        f"anaerobia); tu P/V da {agit_kw:.4g} kW por tanque: el costo del agitador no es representativo "
                        "(ver agitator_alternative).")
    assumptions.append("Costo de compra: partidas de costo de biosteam.units.NRELAnaerobicBatchBioreactor evaluadas por la "
                       f"rutina de BioSTEAM a su CEPCI {ce_bst:g} y escaladas a {basis}. Acero inoxidable 304 como en la "
                       "cotización de NREL; base_purchase_cost_usd = purchase_cost_usd (C_BM = F_BM·C_P).")
    assumptions.append("El enfriador se costea con la carga PICO por tanque (BioSTEAM usa la carga media de toda la "
                       "unidad): decisión de este módulo." if duty_w is not None else
                       "Sin heat no hay carga térmica: el enfriador de fermentación no se costea.")
    if duty_w is None:
        not_included.append("Costo del enfriador de fermentación (falta heat para su carga).")
    not_included.append("Costo: tren de inóculo, tanques de retención, instrumentación, tuberías, instalación (usa F_BM en "
                        "capital_investment) y aireación/compresores.")
    return {"equipment": equipment, "total_purchase_cost_usd": total, "cost_year": year if cepci is None else None,
            "cepci": ce_target, "cepci_basis": basis, "biosteam_cepci": ce_bst, "source": NREL_SOURCE,
            "electricity_kW": {"agitators_from_user_P_V": agit_kw * n if agit_kw is not None else None,
                               "biosteam_nrel_basis_total": float(sum(power))},
            "agitator_alternative": ({"purchase_cost_usd_each": alt * f_ce, "power_hp_each": agit_kw * 1.34102,
                                      "correlation": "BioSTEAM compute_closed_vessel_turbine_purchase_cost (Seider): "
                                                     "4105·hp^0.57 a CEPCI 567; BioSTEAM no declara su intervalo",
                                      "basis": basis} if alt is not None else None),
            "range": f"punto único de NREL: {NREL_REFERENCE_VOLUME_M3:g} m³ por tanque; BioSTEAM no declara límites; aquí "
                     f"se avisa fuera de {lo:.4g}-{hi:.4g} m³"}


# ---------------------------------------------------------------------------------------------------------------
# batch distillation
class _Equilibrium:
    """y(x) and x(y) for the three equilibrium options. Binary tables refer to component 0."""

    def __init__(self, spec: dict | None, components: list[str], warnings: list, assumptions: list, x_charge: np.ndarray):
        if not isinstance(spec, dict) or not spec.get("type"):
            raise ValueError("Falta equilibrium: {'type': 'constant_alpha', 'alpha': [...]} | {'type': 'table', 'x': [...], "
                             "'y': [...]} | {'type': 'thermo', 'model': ..., 'P_Pa': ...}. El equilibrio es dato del usuario.")
        self.kind = str(spec["type"]).lower()
        self.nc = len(components)
        self.components = components
        self.flash_calls = 0
        self.flash_unverified = 0
        self.thermo = None
        if self.kind == "constant_alpha":
            a = spec.get("alpha")
            if a is None or len(a) != self.nc:
                raise ValueError("equilibrium['alpha']: una volatilidad relativa por componente (misma referencia para "
                                 "todas), dato del usuario.")
            self.alpha = np.array([_positive(float(v), "alpha") for v in a])
            assumptions.append("Volatilidades relativas constantes en todo el intervalo (dato del usuario).")
        elif self.kind == "table":
            if self.nc != 2:
                raise ValueError("equilibrium 'table' solo vale para mezclas binarias.")
            x, y = np.array(spec.get("x") or [], float), np.array(spec.get("y") or [], float)
            if len(x) < 3 or len(x) != len(y):
                raise ValueError("equilibrium 'table': da x e y (fracción molar del componente 1 en líquido y vapor) con al "
                                 "menos 3 puntos.")
            order = np.argsort(x)
            x, y = x[order], y[order]
            if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
                raise ValueError("equilibrium 'table': x e y deben ser estrictamente crecientes (sin puntos repetidos).")
            self._set_table(x, y)
            assumptions.append("Curva de equilibrio x-y interpolada con PCHIP monótona entre los puntos del usuario; no se "
                               "extrapola.")
        elif self.kind == "thermo":
            model = spec.get("model")
            if not model:
                raise ValueError("equilibrium 'thermo': falta model (PR, SRK, NRTL, UNIQUAC, WILSON, UNIFAC, "
                                 "UNIFAC_DORTMUND, IDEAL): es una decisión del usuario.")
            P = _positive(_need(spec, "P_Pa", "presión de operación de la destilación, Pa", "equilibrium"), "P_Pa")
            self.thermo = {"model": model, "P_Pa": P, "kij": spec.get("kij"),
                           "activity_parameters": spec.get("activity_parameters"),
                           "use_library_kij": bool(spec.get("use_library_kij", True))}
            self._flash_notes: set[str] = set()
            if self.nc == 2:
                n = int(spec.get("table_points", 41))
                xs = np.linspace(0.0, 1.0, n)
                ys = np.array([self._bubble(np.array([v, 1 - v]))[0][0] if 0 < v < 1 else v for v in xs])
                if np.any(np.diff(ys) <= 0):
                    k = int(np.argmax(np.diff(ys) <= 0))
                    raise ValueError("El modelo termodinámico no da una curva y(x) monótona (¿dos fases líquidas?) cerca de "
                                     f"x = {xs[k]:.3f}; este motor no cubre ese caso.")
                self._set_table(xs, ys)
                assumptions.append(f"Binario con 'thermo': curva x-y de {n} puntos de burbuja de independent_flash a "
                                   f"{P:g} Pa (modelo {model}), interpolada con PCHIP.")
            else:
                assumptions.append(f"Valores K de independent_flash (punto de burbuja a {P:g} Pa, modelo {model}) en cada "
                                   "evaluación.")
        else:
            raise ValueError("equilibrium['type'] debe ser 'constant_alpha', 'table' o 'thermo'.")
        self.warnings = warnings

    def _set_table(self, x: np.ndarray, y: np.ndarray) -> None:
        self.tx, self.ty = x, y
        self._f = PchipInterpolator(x, y)
        self._finv = PchipInterpolator(y, x)
        self.azeotrope = None
        d = y - x
        inner = d[(x > 0) & (x < 1)]
        if len(inner) and np.any(inner <= 0) and np.any(inner > 0):
            g = lambda v: float(self._f(v)) - v     # noqa: E731
            xs = np.linspace(x[0], x[-1], 801)
            gs = np.array([g(v) for v in xs])
            idx = [i for i in range(len(xs) - 1) if gs[i] * gs[i + 1] < 0 and 0 < xs[i] and xs[i + 1] < 1]
            if idx:
                self.azeotrope = float(brentq(g, xs[idx[0]], xs[idx[0] + 1]))

    def _bubble(self, x: np.ndarray):
        from packages.properties.flash import independent_flash
        t = self.thermo
        z = np.clip(x, 1e-12, None)          # the flash engine needs every component present; 1e-12 is "absent"
        z = z / z.sum()
        r = independent_flash(self.components, [float(v) for v in z], model=t["model"], P_Pa=t["P_Pa"], vapor_fraction=0.0,
                              kij=t["kij"], activity_parameters=t["activity_parameters"],
                              use_library_kij=t["use_library_kij"])
        self.flash_calls += 1
        if r.get("status") != "ok" or not (r.get("evidence") or {}).get("verified", False):
            self.flash_unverified += 1
        for w in r.get("warnings") or []:
            self._flash_notes.add(w)
        vap = next((ph for ph in r.get("phases", []) if ph.get("type") == "vapor"), None)
        liq = next((ph for ph in r.get("phases", []) if ph.get("type") == "liquid"), None)
        if vap is None or liq is None or r.get("liquid_liquid_split"):
            raise ValueError("independent_flash no devolvió un punto de burbuja líquido-vapor simple (¿dos líquidos o "
                             f"fuera de intervalo?) para x = {dict(zip(self.components, np.round(z, 5)))}.")
        y = np.array([vap["mole_fractions"].get(c, 0.0) for c in self.components])
        return y, float(r["T_K"]), float(vap["H_J_mol"]) - float(liq["H_J_mol"])

    def state(self, x: np.ndarray) -> tuple[float | None, float | None]:
        """(bubble T, latent heat) of a still composition, or (None, None) where the flash engine cannot give a simple
        bubble point (an essentially pure still)."""
        try:
            _, T, lam = self._bubble(x)
            return T, lam
        except ValueError:
            self._flash_notes.add("No se pudo evaluar el punto de burbuja de un hervidor prácticamente puro: esa "
                                  "temperatura (o calor de vaporización) queda sin dato.")
            return None, None

    def y(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "constant_alpha":
            w = self.alpha * np.clip(x, 0.0, None)
            return w / w.sum()
        if self.nc == 2:
            x1 = float(min(max(x[0], self.tx[0]), self.tx[-1]))
            y1 = float(self._f(x1))
            return np.array([y1, 1.0 - y1])
        return self._bubble(x)[0]

    def x_from_y(self, y: np.ndarray) -> np.ndarray:
        if self.kind == "constant_alpha":
            w = np.clip(y, 0.0, None) / self.alpha
            return w / w.sum()
        if self.nc == 2:
            y1 = float(min(max(y[0], self.ty[0]), self.ty[-1]))
            x1 = float(self._finv(y1))
            return np.array([x1, 1.0 - x1])
        raise ValueError("La rectificación por etapas multicomponente solo está disponible con volatilidades relativas "
                         "constantes.")

    def in_table(self, x1: float) -> bool:
        return self.kind == "constant_alpha" or self.nc != 2 or (self.tx[0] - 1e-12 <= x1 <= self.tx[-1] + 1e-12)


def _still_from_distillate(eq: _Equilibrium, xD: np.ndarray, n_stages: int, R: float) -> np.ndarray:
    """Stage-to-stage from the top (total condenser): y1 = xD; x_n in equilibrium with y_n; operating line
    y_{n+1} = (R·x_n + xD)/(R + 1). The N-th equilibrium stage is the still: returns its liquid composition."""
    y = xD
    x = eq.x_from_y(y)
    for _ in range(n_stages - 1):
        y = (R * x + xD) / (R + 1.0) if math.isfinite(R) else x
        x = eq.x_from_y(y)
    return x


def _distillate_from_still(eq: _Equilibrium, xW: np.ndarray, n_stages: int, R: float, guess: np.ndarray | None = None):
    """Invert `_still_from_distillate`: distillate composition in equilibrium-stage balance with the still liquid."""
    if n_stages == 1:
        return eq.y(xW)
    if eq.nc == 2:
        hi = (eq.azeotrope if getattr(eq, "azeotrope", None) and xW[0] < eq.azeotrope else
              (eq.tx[-1] if eq.kind != "constant_alpha" else 1.0))
        lo = float(xW[0])
        g = lambda v: _still_from_distillate(eq, np.array([v, 1.0 - v]), n_stages, R)[0] - xW[0]   # noqa: E731
        hi_eff = hi - 1e-13
        if g(hi_eff) <= 0:               # even the richest reachable distillate needs a leaner still: pinched at the top
            return np.array([hi_eff, 1.0 - hi_eff])
        v = brentq(g, lo, hi_eff, xtol=1e-14, rtol=1e-13)
        return np.array([v, 1.0 - v])
    fenske = xW * eq.alpha ** n_stages    # total-reflux (Fenske) distribution: always a feasible starting point
    fenske = fenske / fenske.sum()

    def unpack(v):
        e = np.exp(np.append(v, 0.0) - np.max(np.append(v, 0.0)))
        return e / e.sum()

    def res(v):
        return (_still_from_distillate(eq, unpack(v), n_stages, R) - xW)[:-1]
    best = None
    for g0 in ([guess] if guess is not None else []) + [fenske, eq.y(xW)]:
        g0 = np.clip(g0, 1e-300, None)
        v, _, ier, _ = fsolve(res, np.log(g0[:-1] / g0[-1]), full_output=True, xtol=1e-13)
        xd = unpack(v)
        err = float(np.max(np.abs(_still_from_distillate(eq, xd, n_stages, R) - xW)))
        if best is None or err < best[1]:
            best = (xd, err)
        if err < 1e-9:
            break
    if best[1] > 1e-7:
        raise RuntimeError("no convergió el cálculo etapa a etapa de la composición del destilado")
    return best[0]


def rayleigh_constant_alpha(alpha: list[float], x0: list[float], W0: float, W: float) -> list[float]:
    """Analytical still composition of simple (Rayleigh) distillation with constant relative volatilities:
    (W·x_i)/(W0·x_i0) = [(W·x_r)/(W0·x_r0)]^(alpha_i/alpha_r) for every i, with sum x_i = 1 fixing the reference
    component's remaining fraction (one scalar root). Derived from d(W·x_i)/d(W·x_r) = alpha_i·x_i/(alpha_r·x_r)."""
    a = np.array(alpha, float) / float(alpha[0])
    n0 = W0 * np.array(x0, float)
    g = lambda u: float(np.sum(n0 * u ** a) - W)      # u = remaining fraction of component 0   # noqa: E731
    u = brentq(g, 0.0, 1.0, xtol=1e-16, rtol=1e-15)
    n = n0 * u ** a
    return (n / n.sum()).tolist()


def batch_distillation(components: list[str], charge_mol: float, charge_mole_fractions: list[float], equilibrium: dict,
                       stop: dict, mode: str = "rayleigh", n_stages: int | None = None, reflux_ratio: float | None = None,
                       distillate_mole_fraction: float | None = None, boilup_rate_mol_s: float | None = None,
                       heat_of_vaporization_J_mol: float | None = None, n_points: int = 41) -> dict:
    """Batch distillation of a charge: simple (Rayleigh), multistage at constant reflux, or multistage at constant
    distillate composition (binary). See the module docstring for the model and its assumptions."""
    warnings: list[str] = []
    assumptions: list[str] = []
    not_included = ["Retención de líquido en platos, condensador y tambor de reflujo (reduce la separación real y alarga el "
                    "arranque).", "Balance de energía por etapa (se supone flujo molar constante), caída de presión y "
                    "eficiencia de plato (las etapas son teóricas).",
                    "Arranque a reflujo total, cortes intermedios (slop cuts) y política óptima de reflujo."]
    reasons: list[str] = []
    if not components or len(components) < 2:
        raise ValueError("components: al menos dos componentes (nombres del usuario).")
    nc = len(components)
    W0 = _positive(float(charge_mol) if charge_mol is not None else -1.0, "charge_mol (carga inicial del hervidor, mol)")
    if charge_mole_fractions is None or len(charge_mole_fractions) != nc:
        raise ValueError("charge_mole_fractions: una fracción molar por componente (dato del usuario).")
    x0 = np.array([float(v) for v in charge_mole_fractions])
    if np.any(x0 < 0) or abs(x0.sum() - 1.0) > 1e-6:
        raise ValueError(f"charge_mole_fractions debe ser ≥ 0 y sumar 1 (suma {x0.sum():.6g}).")
    x0 = x0 / x0.sum()
    mode = str(mode).lower()
    if mode not in ("rayleigh", "constant_reflux", "constant_distillate"):
        raise ValueError("mode debe ser 'rayleigh', 'constant_reflux' o 'constant_distillate'.")
    eq = _Equilibrium(equilibrium, list(components), warnings, assumptions, x0)
    if mode == "rayleigh":
        N, R = 1, 0.0
        if n_stages not in (None, 1) or reflux_ratio not in (None, 0, 0.0):
            raise ValueError("mode 'rayleigh' es una sola etapa sin reflujo: usa 'constant_reflux' para n_stages y "
                             "reflux_ratio.")
    else:
        if n_stages is None or int(n_stages) < 1 or int(n_stages) != float(n_stages):
            raise ValueError("Falta n_stages: número de etapas teóricas INCLUYENDO el hervidor (entero ≥ 1), dato del "
                             "equipo.")
        N = int(n_stages)
        if eq.nc > 2 and eq.kind != "constant_alpha" and N > 1:
            raise ValueError("La rectificación por etapas multicomponente solo está disponible con equilibrium "
                             "'constant_alpha'; con 'thermo' multicomponente usa mode 'rayleigh'.")
        if mode == "constant_reflux":
            if reflux_ratio is None:
                raise ValueError("Falta reflux_ratio (L/D, constante): dato de operación del usuario.")
            R = _positive(float(reflux_ratio), "reflux_ratio", zero_ok=True)
        assumptions.append(f"{N} etapas teóricas contando el hervidor; condensador total; reflujo como líquido saturado.")
    V = _positive(float(boilup_rate_mol_s), "boilup_rate_mol_s") if boilup_rate_mol_s is not None else None
    lam = (_positive(float(heat_of_vaporization_J_mol), "heat_of_vaporization_J_mol")
           if heat_of_vaporization_J_mol is not None else None)
    if not isinstance(stop, dict):
        raise ValueError("Falta stop: criterio de paro (dato de operación del usuario).")
    stop_keys = ("residue_mole_fraction", "distilled_fraction", "average_distillate_mole_fraction", "time_s",
                 "max_reflux_ratio")
    given = [k for k in stop_keys if stop.get(k) is not None]
    if len(given) != 1:
        raise ValueError(f"stop debe traer exactamente uno de {list(stop_keys)} (y 'component' si es una composición).")
    skey, sval = given[0], float(stop[given[0]])
    if skey in ("residue_mole_fraction", "average_distillate_mole_fraction"):
        comp = stop.get("component")
        if comp not in components:
            raise ValueError(f"stop['component'] debe ser uno de {components}: a qué componente se refiere la composición.")
        ic = components.index(comp)
        if not 0 < sval < 1:
            raise ValueError("La fracción molar de paro debe estar entre 0 y 1.")
    else:
        ic = 0
    if skey == "time_s" and V is None:
        raise ValueError("stop['time_s'] necesita boilup_rate_mol_s (vapor generado en el hervidor, mol/s).")
    if skey == "distilled_fraction" and not 0 < sval < 1:
        raise ValueError("stop['distilled_fraction'] (destilado/carga, molar) debe estar entre 0 y 1.")
    if skey == "max_reflux_ratio" and mode != "constant_distillate":
        raise ValueError("stop['max_reflux_ratio'] solo aplica a mode 'constant_distillate'.")
    if eq.kind != "constant_alpha" and eq.nc == 2 and not eq.in_table(x0[0]):
        raise ValueError("La composición de la carga queda fuera de la tabla de equilibrio dada.")
    n_points = int(max(5, min(n_points, 1001)))
    assumptions.append("Flujo molar constante (CMO), retención despreciable en la columna y el condensador, hervidor "
                       "perfectamente mezclado y en equilibrio con su vapor.")

    if mode == "constant_distillate":
        res = _constant_distillate(eq, components, W0, x0, N, distillate_mole_fraction, skey, sval, ic, V, lam, n_points,
                                   warnings, assumptions, reasons)
    else:
        res = _rayleigh_like(eq, components, W0, x0, N, R, skey, sval, ic, V, lam, n_points, warnings, assumptions, reasons,
                             mode)
    if eq.thermo is not None:
        res["evidence"]["flash_calls"] = eq.flash_calls
        res["evidence"]["flash_calls_not_verified"] = eq.flash_unverified
        if eq.flash_unverified:
            reasons.append(f"{eq.flash_unverified} de {eq.flash_calls} puntos de burbuja no quedaron verificados por "
                           "independent_flash")
        warnings.extend(sorted(eq._flash_notes))
    if lam is None and res.get("heat") is None:
        not_included.append("Calor del hervidor (no se dio heat_of_vaporization_J_mol ni hay modelo 'thermo' del que "
                            "tomarlo).")
    if V is None:
        not_included.append("Tiempo de lote y perfiles contra el tiempo (no se dio boilup_rate_mol_s).")
    res["evidence"]["verified"] = not reasons
    if reasons:
        res["evidence"]["reasons"] = reasons
    res.update(warnings=warnings, assumptions=assumptions, not_included=not_included)
    return _clean(res)


def _latent(eq: _Equilibrium, lam: float | None, xs: list[np.ndarray], assumptions: list) -> tuple[float | None, str | None]:
    if lam is not None:
        return lam, "calor de vaporización molar dado por el usuario (constante)"
    if eq.thermo is None:
        return None, None
    vals = [v for v in (eq.state(x)[1] for x in xs) if v is not None]
    if not vals:
        return None, None
    assumptions.append("Calor de vaporización: H(vapor incipiente) − H(líquido) del punto de burbuja de independent_flash, "
                       "promedio entre el estado inicial y el final del hervidor.")
    return float(np.mean(vals)), "independent_flash (promedio inicial-final)"


def _rayleigh_like(eq, components, W0, x0, N, R, skey, sval, ic, V, lam, n_points, warnings, assumptions, reasons, mode):
    nc = len(components)
    state = {"guess": None}

    def xD_of(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 1e-300, None)
        x = x / x.sum()
        if N == 1:
            return eq.y(x)
        xd = _distillate_from_still(eq, x, N, R, state["guess"])
        state["guess"] = xd
        return xd

    # states: x (nc), cumulative distillate of each component / W0 (nc)
    def rhs(xi, s):
        x = s[:nc]
        xd = xD_of(x)
        return np.concatenate([x - xd, math.exp(-xi) * xd])

    def avg_xd(xi, s):
        d = 1.0 - math.exp(-xi)
        return s[nc + ic] / d if d > 1e-14 else xD_of(s[:nc])[ic]

    xd0 = xD_of(x0)
    rising = xd0[ic] > x0[ic]                # the stop component is enriched in the distillate
    time_factor = (R + 1.0) / V if V else None
    if skey == "residue_mole_fraction":
        if (sval >= x0[ic]) == rising:
            raise ValueError(f"stop: el hervidor {'se empobrece' if rising else 'se enriquece'} en {components[ic]} "
                             f"(carga {x0[ic]:.4g}); {sval:.4g} no se alcanza destilando.")
        ev = lambda xi, s: s[ic] - sval                                     # noqa: E731
    elif skey == "distilled_fraction":
        ev = lambda xi, s: (1.0 - math.exp(-xi)) - sval                     # noqa: E731
    elif skey == "time_s":
        ev = lambda xi, s: W0 * (1.0 - math.exp(-xi)) * time_factor - sval  # noqa: E731
        if sval * V / (R + 1.0) >= W0:
            raise ValueError("stop['time_s'] destilaría toda la carga con ese boilup y reflujo.")
    else:
        if (sval >= xd0[ic]) == rising or (sval <= x0[ic]) == rising:
            raise ValueError(f"stop: la composición media del destilado en {components[ic]} va de {xd0[ic]:.4g} (primera "
                             f"gota) hacia {x0[ic]:.4g} (todo destilado); {sval:.4g} no está en ese intervalo.")
        ev = lambda xi, s: (avg_xd(xi, s) - sval) if xi > 1e-9 else (xd0[ic] - sval)   # noqa: E731
    ev.terminal = True
    xi_max = 30.0                                # W/W0 = 1e-13: the still is dry for any practical purpose
    s0 = np.concatenate([x0, np.zeros(nc)])
    table_floor = None
    events = [ev]
    if eq.kind != "constant_alpha" and nc == 2:
        lo_t, hi_t = eq.tx[0], eq.tx[-1]
        out_of_table = lambda xi, s: (s[0] - lo_t) * (hi_t - s[0]) + 1e-12   # noqa: E731
        out_of_table.terminal = True
        events.append(out_of_table)
        table_floor = (lo_t, hi_t)

    def run(rtol):
        state["guess"] = None
        return solve_ivp(rhs, (0.0, xi_max), s0, method="LSODA" if eq.thermo is None or nc == 2 else "RK45", rtol=rtol,
                         atol=1e-13 if rtol < 1e-8 else 1e-10, events=events, dense_output=True)
    tight = eq.thermo is None or nc == 2
    sol = run(1e-11 if tight else 1e-6)       # every evaluation of the multicomponent thermo route is a flash
    if not sol.success:
        return {"status": "not_converged", "method": "scipy.integrate.solve_ivp", "evidence": {"integrator": sol.message}}
    reached = sol.status == 1 and len(sol.t_events[0]) > 0
    xi_f = float(sol.t[-1])
    if not reached:
        why = ("la composición del hervidor salió de la tabla de equilibrio dada" if (len(events) > 1 and sol.status == 1)
               else "se destiló prácticamente toda la carga")
        warnings.append(f"El criterio de paro no se alcanzó: {why}. Se reporta el estado hasta ahí.")
        reasons.append("criterio de paro no alcanzado")
    xis = np.linspace(0.0, xi_f, n_points)
    S_ = sol.sol(xis).T
    Wp = W0 * np.exp(-xis)
    xW = np.clip(S_[:, :nc], 0.0, None)
    xW = xW / xW.sum(axis=1, keepdims=True)
    Dp = W0 - Wp
    Dcomp = W0 * S_[:, nc:]
    state["guess"] = None
    xD_inst = np.array([xD_of(x) for x in xW])
    with np.errstate(invalid="ignore", divide="ignore"):
        xD_avg = np.where(Dp[:, None] > 1e-12 * W0, Dcomp / np.maximum(Dp[:, None], 1e-300), xD_inst)
    Wf, Df, xWf = float(Wp[-1]), float(Dp[-1]), xW[-1]
    xDf = xD_avg[-1]
    # evidence: component balance with the independently integrated distillate, second tolerance, analytical form
    bal = float(np.max(np.abs(Wf * xWf + Dcomp[-1] - W0 * x0)) / W0)
    sol2 = run(1e-8 if tight else 1e-4)
    s2 = sol2.sol(min(xi_f, float(sol2.t[-1])))
    # second run: still composition at the same cut, and still amount (per mol of charge) at which it stopped
    tol_diff = float(max(np.max(np.abs(s2[:nc] - S_[-1, :nc])),
                         abs(math.exp(-float(sol2.t[-1])) - math.exp(-xi_f)) if reached else 0.0))
    evidence = {"component_balance_max_abs_mol_per_mol_charge": bal, "tolerance_independence_max_abs": tol_diff,
                "stop_reached": reached}
    evidence["tolerances"] = {"component_balance": 1e-7 if tight else 1e-5, "tolerance_independence": 1e-6 if tight else 5e-4,
                              "integration_rtol": [1e-11, 1e-8] if tight else [1e-6, 1e-4]}
    if bal > evidence["tolerances"]["component_balance"]:
        reasons.append(f"el balance por componente cierra a {bal:.2e} mol/mol de carga")
    if tol_diff > evidence["tolerances"]["tolerance_independence"]:
        reasons.append(f"dos tolerancias de integración difieren {tol_diff:.2e} en composición")
    if eq.kind == "constant_alpha" and N == 1:
        xa = np.array(rayleigh_constant_alpha(eq.alpha.tolist(), x0.tolist(), W0, Wf))
        evidence["rayleigh_analytical_max_abs_diff"] = float(np.max(np.abs(xa - xWf)))
        evidence["rayleigh_analytical_formula"] = "(W·x_i)/(W0·x_i0) = [(W·x_r)/(W0·x_r0)]^(alpha_i/alpha_r)"
        if evidence["rayleigh_analytical_max_abs_diff"] > 1e-7:
            reasons.append("la composición final no coincide con la solución analítica de Rayleigh")
    if eq.kind == "constant_alpha" and N > 1 and math.isfinite(R):
        fen = xWf * eq.alpha ** N
        evidence["total_reflux_fenske_distillate"] = dict(zip(components, (fen / fen.sum()).tolist()))
    T_prof = None
    if eq.thermo is not None:
        T_prof = [eq.state(x)[0] for x in xW]
    lam_v, lam_src = _latent(eq, lam, [x0, xWf], assumptions)
    t_prof = Dp * time_factor if V else None
    out = {
        "status": "ok", "mode": mode, "n_stages": N, "reflux_ratio": R if N > 1 or mode != "rayleigh" else 0.0,
        "components": list(components),
        "final": {"still_mol": Wf, "distillate_mol": Df, "fraction_distilled": Df / W0,
                  "still_mole_fractions": dict(zip(components, xWf.tolist())),
                  "average_distillate_mole_fractions": dict(zip(components, xDf.tolist())),
                  "instantaneous_distillate_mole_fractions": dict(zip(components, xD_inst[-1].tolist())),
                  "recovery_in_distillate": {c: (float(Dcomp[-1, i] / (W0 * x0[i])) if x0[i] > 0 else None)
                                             for i, c in enumerate(components)},
                  "rayleigh_integral_ln_W0_over_W": xi_f,
                  "batch_time_s": float(t_prof[-1]) if V else None,
                  "still_T_K": T_prof[-1] if T_prof else None},
        "profile": {"still_mol": Wp, "distillate_mol": Dp, **({"t_s": t_prof} if V else {}),
                    "still_mole_fractions": {c: xW[:, i] for i, c in enumerate(components)},
                    "instantaneous_distillate_mole_fractions": {c: xD_inst[:, i] for i, c in enumerate(components)},
                    "average_distillate_mole_fractions": {c: xD_avg[:, i] for i, c in enumerate(components)},
                    **({"still_T_K": T_prof} if T_prof else {})},
        "heat": None,
        "method": ("Ecuación de Rayleigh dx_i/dξ = x_i − xD_i con ξ = ln(W0/W), integrada con scipy.integrate.solve_ivp; "
                   + ("xD = y de equilibrio con el hervidor (una etapa)" if N == 1 else
                      f"xD por cálculo etapa a etapa (McCabe-Thiele) con {N} etapas y R = {R:g}, invertido con "
                      + ("brentq" if nc == 2 else "fsolve"))),
        "evidence": evidence,
    }
    if lam_v is not None:
        if V:
            out["heat"] = {"reboiler_duty_W": V * lam_v, "reboiler_heat_J": V * lam_v * float(t_prof[-1]),
                           "heat_of_vaporization_J_mol": lam_v, "source": lam_src}
        else:
            out["heat"] = {"reboiler_heat_J": Df * (R + 1.0) * lam_v, "heat_of_vaporization_J_mol": lam_v, "source": lam_src,
                           "note": "sin boilup_rate_mol_s solo se da el calor total = vapor generado × calor de vaporización"}
        assumptions.append("Calor del hervidor = vapor generado × calor de vaporización molar (sin calentamiento sensible "
                           "de la carga ni pérdidas).")
    if getattr(eq, "azeotrope", None):
        warnings.append(f"La curva de equilibrio cruza la diagonal en x ≈ {eq.azeotrope:.4f} (azeótropo): el destilado no "
                        "puede pasar de esa composición.")
    return out


def _constant_distillate(eq, components, W0, x0, N, xD_spec, skey, sval, ic, V, lam, n_points, warnings, assumptions,
                         reasons):
    if eq.nc != 2:
        raise ValueError("mode 'constant_distillate' solo está disponible para mezclas binarias.")
    if xD_spec is None:
        raise ValueError("Falta distillate_mole_fraction: fracción molar del componente 1 que se mantiene en el destilado "
                         "(especificación del usuario).")
    xD = float(xD_spec)
    if not x0[0] < xD < 1:
        raise ValueError("distillate_mole_fraction debe ser mayor que la del componente 1 en la carga y menor que 1.")
    if N < 2:
        raise ValueError("Con una sola etapa no se puede mantener constante el destilado: n_stages ≥ 2.")
    if eq.kind != "constant_alpha" and not (eq.ty[0] < xD <= eq.ty[-1]):
        raise ValueError("distillate_mole_fraction queda fuera de la tabla de equilibrio.")
    xDv = np.array([xD, 1.0 - xD])
    x_min = float(_still_from_distillate(eq, xDv, N, math.inf)[0])       # total reflux: leanest still that still gives xD
    if x_min >= x0[0]:
        raise ValueError(f"Con {N} etapas no se alcanza xD = {xD:g} ni a reflujo total desde una carga con x = {x0[0]:.4g} "
                         f"(a reflujo total el hervidor tendría que tener x ≥ {x_min:.4g}).")

    def R_of(x: float) -> float:
        g = lambda r: _still_from_distillate(eq, xDv, N, r)[0] - x     # decreasing in r   # noqa: E731
        lo, hi = 0.0, 1.0
        if g(lo) <= 0:
            return 0.0
        while g(hi) > 0:
            hi *= 2.0
            if hi > 1e9:
                return math.inf
        return float(brentq(g, lo, hi, xtol=1e-12, rtol=1e-12))

    W_of = lambda x: W0 * (xD - x0[0]) / (xD - x)      # noqa: E731
    if skey == "residue_mole_fraction":
        if ic != 0:
            sval, ic = 1.0 - sval, 0
        x_end = sval
    elif skey == "distilled_fraction":
        x_end = xD - (xD - x0[0]) / (1.0 - sval)
    elif skey == "max_reflux_ratio":
        Rmax = _positive(sval, "stop['max_reflux_ratio']")
        if R_of(x0[0]) >= Rmax:
            raise ValueError(f"Ya al inicio se necesita R = {R_of(x0[0]):.4g} ≥ max_reflux_ratio.")
        x_end = float(_still_from_distillate(eq, xDv, N, Rmax)[0])      # still composition that needs exactly Rmax
    elif skey == "time_s":
        x_end = None
    else:
        raise ValueError("Con destilado constante, stop es residue_mole_fraction, distilled_fraction, max_reflux_ratio o "
                         "time_s (la composición media del destilado ya es la especificada).")
    reached = True
    floor = x_min + 1e-6 * (x0[0] - x_min)
    if x_end is not None and not floor < x_end < x0[0]:
        if x_end >= x0[0]:
            raise ValueError("El criterio de paro ya se cumple con la carga inicial.")
        warnings.append(f"El paro pedido (x del hervidor = {x_end:.4g}) está por debajo del límite de reflujo total "
                        f"({x_min:.4g}): se reporta hasta R muy alto, sin alcanzarlo.")
        reasons.append("criterio de paro inalcanzable con estas etapas")
        x_end, reached = floor + 1e-3 * (x0[0] - floor), False

    def integrate(x_stop: float, n: int):
        xs = np.linspace(x0[0], x_stop, n)
        Rs = np.array([R_of(v) for v in xs])
        integrand = (Rs + 1.0) / (xD - xs) ** 2
        cum = np.concatenate([[0.0], [-(simpson(integrand[: i + 1], x=xs[: i + 1])) for i in range(1, n)]])
        return xs, Rs, W0 * (xD - x0[0]) * cum          # last array: total vapour boiled up to each point, mol
    if x_end is None:                                    # time stop: find the still composition that takes that time
        target_vap = sval * V
        hi_x, lo_x = x0[0], floor + 1e-3 * (x0[0] - floor)
        if integrate(lo_x, 401)[2][-1] < target_vap:
            warnings.append("En el tiempo pedido se llega al límite de reflujo total; se reporta hasta ahí.")
            reasons.append("criterio de paro inalcanzable con estas etapas")
            x_end, reached = lo_x, False
        else:
            x_end = float(brentq(lambda x: integrate(x, 201)[2][-1] - target_vap, lo_x, hi_x - 1e-12, xtol=1e-10))
    n_fine = 401
    xs, Rs, vap = integrate(x_end, n_fine)
    xs2, Rs2, vap2 = integrate(x_end, 2 * n_fine - 1)
    step_diff = _rel(vap[-1], vap2[-1])
    idx = np.unique(np.round(np.linspace(0, n_fine - 1, n_points)).astype(int))
    Wp = np.array([W_of(v) for v in xs])
    Dp = W0 - Wp
    mt_res = float(np.max(np.abs([_still_from_distillate(eq, xDv, N, r)[0] - v for r, v in zip(Rs[idx], xs[idx])
                                  if math.isfinite(r) and r > 0] or [0.0])))
    bal = abs(Wp[-1] * xs[-1] + Dp[-1] * xD - W0 * x0[0]) / W0
    evidence = {"component_balance_abs_mol_per_mol_charge": float(bal), "stage_calculation_max_abs_residual": mt_res,
                "step_independence_rel_diff_vapour": step_diff, "stop_reached": reached,
                "total_reflux_still_limit_mole_fraction": x_min}
    if eq.kind == "constant_alpha":
        a = float(eq.alpha[0] / eq.alpha[1])
        fen = (xD / (1 - xD)) / a ** N
        evidence["fenske_still_limit_mole_fraction"] = fen / (1 + fen)
        if abs(evidence["fenske_still_limit_mole_fraction"] - x_min) > 1e-9:
            reasons.append("el límite de reflujo total no coincide con Fenske")
    if step_diff > 1e-5:
        reasons.append(f"la cuadratura del vapor generado cambia {step_diff:.2e} al duplicar puntos")
    if mt_res > 1e-8:
        reasons.append("el cálculo etapa a etapa no reproduce la composición del hervidor")
    lam_v, lam_src = _latent(eq, lam, [x0, np.array([xs[-1], 1 - xs[-1]])], assumptions)
    T_prof = [eq.state(np.array([v, 1 - v]))[0] for v in xs[idx]] if eq.thermo is not None else None
    t_prof = vap / V if V else None
    c0, c1 = components
    out = {
        "status": "ok", "mode": "constant_distillate", "n_stages": N, "components": list(components),
        "distillate_mole_fractions": {c0: xD, c1: 1.0 - xD},
        "final": {"still_mol": float(Wp[-1]), "distillate_mol": float(Dp[-1]), "fraction_distilled": float(Dp[-1] / W0),
                  "still_mole_fractions": {c0: float(xs[-1]), c1: float(1 - xs[-1])},
                  "average_distillate_mole_fractions": {c0: xD, c1: 1.0 - xD},
                  "reflux_ratio_initial": float(Rs[0]), "reflux_ratio_final": float(Rs[-1]),
                  "vapour_boiled_mol": float(vap[-1]), "batch_time_s": float(t_prof[-1]) if V else None,
                  "recovery_in_distillate": {c0: float(Dp[-1] * xD / (W0 * x0[0])),
                                             c1: float(Dp[-1] * (1 - xD) / (W0 * x0[1]))},
                  "still_T_K": T_prof[-1] if T_prof else None},
        "profile": {"still_mol": Wp[idx], "distillate_mol": Dp[idx], "reflux_ratio": Rs[idx], "vapour_boiled_mol": vap[idx],
                    **({"t_s": t_prof[idx]} if V else {}),
                    "still_mole_fractions": {c0: xs[idx], c1: 1.0 - xs[idx]}, **({"still_T_K": T_prof} if T_prof else {})},
        "heat": None,
        "method": (f"Destilado constante: en cada composición del hervidor se busca (brentq) el reflujo con el que {N} "
                   "etapas teóricas (McCabe-Thiele) dan el destilado pedido; W = W0·(xD − x0)/(xD − x); vapor generado = "
                   "W0·(xD − x0)·∫(R+1)/(xD − x)² dx por Simpson"),
        "evidence": evidence,
    }
    if lam_v is not None:
        out["heat"] = {"reboiler_heat_J": float(vap[-1]) * lam_v, "heat_of_vaporization_J_mol": lam_v, "source": lam_src,
                       **({"reboiler_duty_W": V * lam_v} if V else {})}
        assumptions.append("Calor del hervidor = vapor generado × calor de vaporización molar (sin calentamiento sensible "
                           "de la carga ni pérdidas).")
    return out
