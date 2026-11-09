"""Detailed kinetics and combustion with Cantera (bloque M2 of docs/MAPA_MOTORES.md).

Two tools the process simulator cannot give:
  * adiabatic_flame: equilibrium (HP) adiabatic flame temperature and composition with dissociation, the complete-
    combustion temperature (no dissociation, what a 100 % conversion reactor computes) as the upper bound, and
    optionally the laminar flame speed (FreeFlame, 1-D, mixture-averaged transport).
  * kinetic_reactor_network: batch (constant volume or pressure), PFR (lagrangian, constant pressure) and well-stirred
    reactor (WSR / CSTR, sweep of residence times) on a detailed mechanism (GRI-Mech 3.0 by default) or on user power-law
    kinetics (isothermal). Gives ignition delay, blow-out, CO and NO, profiles and conversions.

Cantera (BSD-3) runs in process. Mechanisms are restricted to the files Cantera ships (no arbitrary paths).

Evidence → seal (each result carries `evidence.verified` and `evidence.reasons`):
  * element_balance_closed: elemental mass fractions in = out (< 1e-8)
  * energy_conserved: adiabatic runs keep h (constant P) or u (constant V) of the mixture (relative < 1e-6)
  * tolerance_independent: a rerun with integrator tolerances 1000× looser changes T by < 0.5 K and X by < 1e-3
  * steady_state (WSR): advancing 10 more residence times changes nothing
  * grid_independent (flame speed): a finer grid changes S_L by < 2 %
  * within_mechanism_range: GRI-Mech 3.0 (and its h2o2 subset) is validated for 1000-2500 K, 10 Torr-10 atm,
    φ 0.1-5 (premixed natural gas); outside that, or for a mechanism with no stated range, the result is not verified.
    User power-law kinetics are the user's data: the range check passes and the response says so.
"""
from __future__ import annotations

import math
import re

import cantera as ct
import numpy as np

KNOWN_RANGES = {   # published validation range of the mechanism (GRI-Mech 3.0 web page, "optimized for natural gas")
    "gri30.yaml": {"T_K": (1000.0, 2500.0), "P_Pa": (1333.2, 1.01325e6), "phi": (0.1, 5.0)},
    "h2o2.yaml": {"T_K": (1000.0, 2500.0), "P_Pa": (1333.2, 1.01325e6), "phi": (0.1, 5.0)},   # H/O subset of GRI 3.0
}
INERT = {"N2", "AR", "Ar", "HE", "He", "H2O", "CO2"}
TOL_LOOSE = (1e-6, 1e-12)
TOL_TIGHT = (1e-9, 1e-15)
REACTORS = ("batch_constant_volume", "batch_constant_pressure", "pfr", "wsr")


# --------------------------------------------------------------------------- mechanism

def _mechanism_files() -> set[str]:
    return {f for f in ct.list_data_files() if f.endswith(".yaml")}


def _check_mechanism(mechanism: str) -> None:
    if mechanism not in _mechanism_files():
        raise ValueError(f"mechanism {mechanism!r} is not a Cantera data file; use one of "
                         "gri30.yaml (natural gas, default), h2o2.yaml, gri30_highT.yaml, nDodecane_Reitz.yaml, ... "
                         "or give custom_species + custom_reactions")


_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_\-()]*$")


def custom_mechanism_yaml(species: list[dict], reactions: list[dict]) -> str:
    """YAML for user power-law kinetics: ideal gas, constant-cp placeholder thermo (valid only isothermal), irreversible
    reactions r = A·T^b·exp(−Ea/RT)·Π C_i^n_i with C in mol/m³ (A in the matching SI units)."""
    if not species or not reactions:
        raise ValueError("custom_species and custom_reactions must both be non-empty")
    sp_lines, names = [], []
    for s in species:
        n = str(s.get("name", ""))
        comp = s.get("composition")
        if not _NAME.match(n) or not isinstance(comp, dict) or not comp:
            raise ValueError(f"each custom species needs a name and a composition dict of elements, e.g. "
                             f"{{'name': 'A', 'composition': {{'C': 4, 'H': 10}}}}; got {s!r}")
        names.append(n)
        comp_s = ", ".join(f"{e}: {float(v):g}" for e, v in comp.items())
        sp_lines.append(f"- name: {n}\n  composition: {{{comp_s}}}\n"
                        "  thermo: {model: constant-cp, T0: 298.15 K, h0: 0 kJ/mol, s0: 0 J/mol/K, cp0: 100 J/mol/K}")
    rx_lines = []
    for r in reactions:
        eq = str(r.get("equation", ""))
        if "=>" not in eq or "<=>" in eq:
            raise ValueError(f"custom reactions must be irreversible ('A + B => C'); write a reversible one as two "
                             f"irreversible reactions with their own A and Ea. Got {eq!r}")
        line = (f"- equation: {eq}\n  rate-constant: {{A: {float(r['A']):.10g}, b: {float(r.get('b', 0.0)):g}, "
                f"Ea: {float(r.get('Ea_J_mol', 0.0)):.10g}}}")
        if r.get("orders"):
            line += "\n  orders: {" + ", ".join(f"{k}: {float(v):g}" for k, v in r["orders"].items()) + "}"
            line += "\n  nonreactant-orders: true"
        rx_lines.append(line)
    return ("units: {length: m, quantity: mol, activation-energy: J/mol}\n"
            "phases:\n- name: gas\n  thermo: ideal-gas\n  species: [" + ", ".join(names) + "]\n"
            "  kinetics: gas\n  reactions: all\n  state: {T: 300.0, P: 101325.0}\n"
            "species:\n" + "\n".join(sp_lines) + "\nreactions:\n" + "\n".join(rx_lines) + "\n")


def _solution(mechanism: str, custom_yaml: str | None) -> ct.Solution:
    return ct.Solution(yaml=custom_yaml) if custom_yaml else ct.Solution(mechanism)


def _check_species(gas: ct.Solution, comp) -> None:
    keys = comp.keys() if isinstance(comp, dict) else [p.split(":")[0].strip() for p in str(comp).split(",") if p.strip()]
    missing = [k for k in keys if k not in gas.species_names]
    if missing:
        raise ValueError(f"species not in the mechanism: {missing}. The mechanism has {gas.n_species} species, e.g. "
                         f"{gas.species_names[:25]}")


def _elements(gas: ct.Solution) -> dict:
    return {e: float(gas.elemental_mass_fraction(e)) for e in gas.element_names}


def _element_error(a: dict, b: dict) -> float:
    return max(abs(a[e] - b[e]) for e in a)


def _phi(gas: ct.Solution) -> float | None:
    try:
        v = float(gas.equivalence_ratio())
        return v if math.isfinite(v) else None
    except Exception:  # noqa: BLE001  (no fuel or no oxidizer in the mixture)
        return None


def _range_check(mechanism: str, custom: bool, T_K: float, P_Pa: float, phi: float | None) -> tuple[bool | None, list[str], str]:
    if custom:
        return True, [], "cinética del usuario: el rango de validez es responsabilidad de quien da A, Ea y órdenes"
    rng = KNOWN_RANGES.get(mechanism)
    if rng is None:
        return None, [f"el mecanismo {mechanism} no tiene rango de validez registrado aquí; resultado no verificado"], ""
    reasons = []
    lo, hi = rng["T_K"]
    if not (lo <= T_K <= hi):
        reasons.append(f"T = {T_K:.0f} K fuera del rango validado de {mechanism} ({lo:.0f}-{hi:.0f} K)")
    lo, hi = rng["P_Pa"]
    if not (lo <= P_Pa <= hi):
        reasons.append(f"P = {P_Pa/1e5:.3g} bar fuera del rango validado de {mechanism} ({lo/1e5:.3g}-{hi/1e5:.3g} bar)")
    if phi is not None:
        lo, hi = rng["phi"]
        if not (lo <= phi <= hi):
            reasons.append(f"φ = {phi:.2f} fuera del rango validado de {mechanism} ({lo}-{hi})")
    note = f"{mechanism}: validado para gas natural, 1000-2500 K, 10 Torr-10 atm, φ 0.1-5 (mezclas premezcladas)"
    return not reasons, reasons, note


# --------------------------------------------------------------------------- adiabatic flame

def _complete_combustion_T(gas: ct.Solution, h_mass: float, P_Pa: float) -> float | None:
    """T of the products of complete combustion (C→CO2, H→H2O, N→N2, excess O→O2), same h. None if rich or if the
    mixture has elements other than C, H, O, N, Ar, He, or if the mechanism lacks those products."""
    els = set(gas.element_names)
    if not els <= {"C", "H", "O", "N", "Ar", "He"}:
        return None
    n = {e: 0.0 for e in ("C", "H", "O", "N")}
    for k, x in enumerate(gas.X):
        if x <= 0:
            continue
        for e in n:
            if e in els:
                n[e] += x * gas.n_atoms(k, e)
    o2_left = n["O"] / 2 - n["C"] - n["H"] / 4
    if o2_left < -1e-12:
        return None
    prod = {"CO2": n["C"], "H2O": n["H"] / 2, "N2": n["N"] / 2, "O2": max(o2_left, 0.0)}
    for inert in ("AR", "Ar", "HE", "He"):
        if inert in gas.species_names:
            prod[inert] = float(gas[inert].X[0])
    prod = {k: v for k, v in prod.items() if v > 0}
    if any(k not in gas.species_names for k in prod):
        return None
    g = ct.Solution(thermo="ideal-gas", species=[gas.species(k) for k in prod])
    g.HPX = h_mass, P_Pa, prod
    return float(g.T)


def adiabatic_flame(fuel: str | dict, equivalence_ratio: float, T_K: float = 298.15, P_Pa: float = 101325.0,
                    oxidizer: str | dict = "O2:1, N2:3.76", mechanism: str = "gri30.yaml",
                    laminar_flame_speed: bool = False, report_species: list[str] | None = None) -> dict:
    """Adiabatic flame of a premixed fuel/oxidizer mixture (Cantera). See module docstring for the seal."""
    _check_mechanism(mechanism)
    if equivalence_ratio <= 0 or T_K <= 0 or P_Pa <= 0:
        raise ValueError("equivalence_ratio, T_K and P_Pa must be positive")
    gas = ct.Solution(mechanism)
    _check_species(gas, fuel)
    _check_species(gas, oxidizer)
    gas.set_equivalence_ratio(equivalence_ratio, fuel, oxidizer)
    gas.TP = T_K, P_Pa
    if not (gas.min_temp <= T_K <= gas.max_temp):
        raise ValueError(f"T_K outside the thermo range of {mechanism} ({gas.min_temp:.0f}-{gas.max_temp:.0f} K)")
    reactants = {s: float(x) for s, x in zip(gas.species_names, gas.X) if x > 0}
    h0, el0, rho0 = gas.enthalpy_mass, _elements(gas), gas.density
    T_complete = _complete_combustion_T(gas, h0, P_Pa)
    gas.equilibrate("HP")
    T_ad = float(gas.T)
    el_err = _element_error(el0, _elements(gas))
    h_err = abs(gas.enthalpy_mass - h0) / (abs(h0) + gas.cp_mass * T_ad)
    want = report_species or []
    top = [gas.species_names[i] for i in np.argsort(gas.X)[::-1][:8]]
    show = list(dict.fromkeys(top + [s for s in ("CO", "H2", "OH", "NO") if s in gas.species_names] +
                              [s for s in want if s in gas.species_names]))
    products = {s: float(gas[s].X[0]) for s in show}
    reasons = []
    if el_err >= 1e-8:
        reasons.append(f"balance de elementos no cierra ({el_err:.1e})")
    if h_err >= 1e-6:
        reasons.append(f"la entalpía no se conserva ({h_err:.1e})")
    if T_ad > gas.max_temp:
        reasons.append(f"T adiabática {T_ad:.0f} K sobre el límite de los polinomios de {mechanism} ({gas.max_temp:.0f} K)")
    out = {
        "mechanism": mechanism, "n_species": gas.n_species, "n_reactions": gas.n_reactions,
        "equivalence_ratio": equivalence_ratio, "T_in_K": T_K, "P_Pa": P_Pa, "reactants": reactants,
        "T_adiabatic_equilibrium_K": T_ad,
        "T_adiabatic_complete_combustion_K": T_complete,
        "dissociation_penalty_K": (T_complete - T_ad) if T_complete is not None else None,
        "products_mole_fractions": products,
        "density_unburnt_kg_m3": rho0, "density_burnt_kg_m3": float(gas.density),
        "engine": f"Cantera {ct.__version__}",
    }
    ev = {"element_balance_closed": el_err < 1e-8, "element_error": el_err, "energy_conserved": h_err < 1e-6,
          "energy_error": h_err, "within_thermo_range": T_ad <= gas.max_temp}
    if laminar_flame_speed:
        fs = _flame_speed(mechanism, fuel, oxidizer, equivalence_ratio, T_K, P_Pa)
        out["laminar_flame_speed_m_s"] = fs["S_L"]
        out["laminar_flame_speed_coarse_grid_m_s"] = fs["S_L_coarse"]
        out["flame_grid_points"] = fs["points"]
        out["flame_T_max_K"] = fs["T_max"]
        ev["grid_independent"] = fs["delta"] < 0.02
        ev["grid_delta"] = fs["delta"]
        if not ev["grid_independent"]:
            reasons.append(f"la velocidad de llama cambia {100*fs['delta']:.1f} % con la malla fina")
        ok, rs, note = _range_check(mechanism, False, fs["T_max"], P_Pa, equivalence_ratio)
        ev["within_mechanism_range"] = ok
        reasons += rs
        out["mechanism_validity"] = note
    else:
        out["mechanism_validity"] = "equilibrio: solo depende de la termodinámica (polinomios NASA del mecanismo), no de la cinética"
    ev["verified"] = not reasons
    ev["reasons"] = reasons
    out["evidence"] = ev
    out["notes"] = ("La T de combustión completa (sin disociación) es lo que da un reactor de conversión al 100 %: es un tope "
                    "superior. La T de equilibrio incluye CO, H2, OH, O, H y NO. " +
                    ("Velocidad de llama: llama libre 1-D premezclada, transporte promediado por mezcla." if laminar_flame_speed else ""))
    return out


def _flame_speed(mechanism, fuel, oxidizer, phi, T_K, P_Pa) -> dict:
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.TP = T_K, P_Pa
    f = ct.FreeFlame(gas, width=0.03)
    f.transport_model = "mixture-averaged"
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    f.solve(loglevel=0, auto=True)
    s1 = float(f.velocity[0])
    f.set_refine_criteria(ratio=2, slope=0.03, curve=0.06)
    f.solve(loglevel=0)
    s2 = float(f.velocity[0])
    return {"S_L": s2, "S_L_coarse": s1, "delta": abs(s2 - s1) / s2, "points": len(f.grid), "T_max": float(max(f.T))}


# --------------------------------------------------------------------------- reactor networks

def _mass_fractions(gas: ct.Solution) -> np.ndarray:
    return np.array(gas.Y, dtype=float)


def _batch(gas_factory, reactor: str, energy: str, t_end: float, tol: tuple[float, float]) -> dict:
    gas = gas_factory()
    cls = ct.IdealGasReactor if reactor == "batch_constant_volume" else ct.IdealGasConstPressureReactor
    r = cls(gas, energy="on" if energy == "adiabatic" else "off", clone=False)
    net = ct.ReactorNet([r])
    net.rtol, net.atol = tol
    t, T, Y = [0.0], [r.T], [_mass_fractions(r.phase)]
    while net.time < t_end:
        net.step()
        t.append(net.time)
        T.append(r.T)
        Y.append(_mass_fractions(r.phase))
    # exact end state: fresh run with advance(t_end)
    gas2 = gas_factory()
    r2 = cls(gas2, energy="on" if energy == "adiabatic" else "off", clone=False)
    net2 = ct.ReactorNet([r2])
    net2.rtol, net2.atol = tol
    net2.advance(t_end)
    return {"t": np.array(t), "T": np.array(T), "Y": np.array(Y), "end": r2.phase, "steps": len(t) - 1}


def _wsr(gas_factory, energy: str, tau: float, initial_state: str, tol: tuple[float, float]) -> dict:
    feed = gas_factory()
    inlet = ct.Reservoir(feed, clone=True)         # independent copy: the reservoir keeps the inlet state
    gas = gas_factory()
    if initial_state == "ignited":
        gas.equilibrate("HP" if energy == "adiabatic" else "TP")
    T_init = float(gas.T)
    r = ct.IdealGasReactor(gas, energy="on" if energy == "adiabatic" else "off", clone=False)
    r.volume = 1.0
    exhaust = ct.Reservoir(gas, clone=True)
    mfc = ct.MassFlowController(inlet, r, mdot=r.mass / tau)      # ONE inlet: a second MFC halves τ silently
    ct.PressureController(r, exhaust, primary=mfc, K=0.01)
    net = ct.ReactorNet([r])
    net.rtol, net.atol = tol
    converged = True
    try:
        net.advance_to_steady_state()
    except ct.CanteraError:
        converged = False
    T1, Y1 = float(r.T), _mass_fractions(r.phase)
    try:
        net.advance(net.time + 10 * tau)
    except ct.CanteraError:
        converged = False
    drift_T = abs(float(r.T) - T1)
    drift_Y = float(np.max(np.abs(_mass_fractions(r.phase) - Y1)))
    return {"phase": r.phase, "T_init": T_init, "converged": converged, "drift_T": drift_T, "drift_Y": drift_Y,
            "mdot_per_m3": float(mfc.mass_flow_rate)}


def _snapshot(gas: ct.Solution, Y_in: np.ndarray, key: list[str], show: list[str]) -> dict:
    Y = np.array(gas.Y)
    conv = {}
    for s in key:
        i = gas.species_index(s)
        conv[s] = float(1 - Y[i] / Y_in[i]) if Y_in[i] > 0 else None
    return {"T_K": float(gas.T), "P_Pa": float(gas.P), "conversion": conv,
            "mole_fractions": {s: float(gas[s].X[0]) for s in show}}


def kinetic_reactor_network(reactor: str, feed: str | dict, T_K: float, P_Pa: float, mechanism: str = "gri30.yaml",
                            energy: str = "adiabatic", end_time_s: float | None = None,
                            residence_times_s: list[float] | None = None, key_species: list[str] | None = None,
                            report_species: list[str] | None = None, initial_state: str = "ignited",
                            custom_species: list[dict] | None = None, custom_reactions: list[dict] | None = None,
                            n_profile_points: int = 60) -> dict:
    """Batch / PFR / WSR on a detailed mechanism or user power-law kinetics (Cantera). See module docstring."""
    if reactor not in REACTORS:
        raise ValueError(f"reactor must be one of {REACTORS}")
    if energy not in ("adiabatic", "isothermal"):
        raise ValueError("energy must be 'adiabatic' or 'isothermal'")
    if initial_state not in ("ignited", "feed"):
        raise ValueError("initial_state must be 'ignited' (WSR starts burning, finds the upper branch) or 'feed'")
    if T_K <= 0 or P_Pa <= 0:
        raise ValueError("T_K and P_Pa must be positive")
    custom = bool(custom_species or custom_reactions)
    cy = custom_mechanism_yaml(custom_species or [], custom_reactions or []) if custom else None
    if custom and energy != "isothermal":
        raise ValueError("custom power-law kinetics run isothermal only (the tool has no real thermo for your species: "
                         "an adiabatic run would invent the heat of reaction); use DWSIM's kinetic reactor for that")
    if not custom:
        _check_mechanism(mechanism)
    probe = _solution(mechanism, cy)
    _check_species(probe, feed)

    def gas_factory():
        g = _solution(mechanism, cy)
        g.TPX = T_K, P_Pa, feed
        return g

    g_in = gas_factory()
    Y_in, el_in = _mass_fractions(g_in), _elements(g_in)
    h_in, u_in, cpT = g_in.enthalpy_mass, g_in.int_energy_mass, g_in.cp_mass * T_K
    phi = None if custom else _phi(g_in)
    feed_species = [s for s, x in zip(g_in.species_names, g_in.X) if x > 0]
    key = key_species or [s for s in feed_species if s not in INERT and (custom or s not in ("O2",))] or feed_species
    _check_species(probe, {k: 1 for k in key})
    show_base = list(dict.fromkeys(key + [s for s in ("CO", "NO", "NO2", "OH") if s in g_in.species_names] +
                                   [s for s in (report_species or []) if s in g_in.species_names]))
    reasons: list[str] = []
    out: dict = {"reactor": reactor, "energy": energy, "mechanism": "custom (usuario)" if custom else mechanism,
                 "n_species": g_in.n_species, "n_reactions": g_in.n_reactions, "feed": {s: float(g_in[s].X[0]) for s in feed_species},
                 "T_in_K": T_K, "P_Pa": P_Pa, "equivalence_ratio": phi, "key_species": key, "engine": f"Cantera {ct.__version__}"}
    ev: dict = {}
    if reactor != "wsr":
        if not end_time_s or end_time_s <= 0:
            raise ValueError("end_time_s > 0 is required for batch and pfr (for pfr it is the residence time)")
        tight = _batch(gas_factory, reactor, energy, end_time_s, TOL_TIGHT)
        loose = _batch(gas_factory, reactor, energy, end_time_s, TOL_LOOSE)
        end = tight["end"]
        top = [end.species_names[i] for i in np.argsort(end.X)[::-1][:8]]
        show = list(dict.fromkeys(show_base + top))
        snap = _snapshot(end, Y_in, key, show)
        snap_l = _snapshot(loose["end"], Y_in, key, show)
        out["final"] = snap
        # profile on an even time grid, interpolated from the adaptive steps
        tg = np.linspace(0.0, end_time_s, max(2, int(n_profile_points)))
        prof = {"t_s": tg.tolist(), "T_K": np.interp(tg, tight["t"], tight["T"]).tolist()}
        for s in key[:4]:
            i = end.species_index(s)
            prof[f"X_{s}"] = (1 - np.interp(tg, tight["t"], tight["Y"][:, i]) / Y_in[i]).tolist() if Y_in[i] > 0 else None
        out["profile"] = prof
        out["integrator_steps"] = tight["steps"]
        T_op = float(np.max(tight["T"]))
        if energy == "adiabatic" and T_op - T_K > 100:
            dTdt = np.gradient(tight["T"], tight["t"])
            i = int(np.argmax(dTdt))
            dTl = np.gradient(loose["T"], loose["t"])
            t_ign, t_ign_l = float(tight["t"][i]), float(loose["t"][int(np.argmax(dTl))])
            out["ignition_delay_s"] = t_ign
            out["ignition_delay_loose_tol_s"] = t_ign_l
            ev["ignition_delay_tolerance_delta"] = abs(t_ign - t_ign_l) / t_ign
            if ev["ignition_delay_tolerance_delta"] >= 0.01:
                reasons.append(f"el retardo de ignición cambia {100*ev['ignition_delay_tolerance_delta']:.1f} % con la tolerancia")
        elif energy == "adiabatic":
            out["ignition_delay_s"] = None
            out["ignition_note"] = f"sin ignición antes de {end_time_s:g} s (ΔT < 100 K)"
        dT = abs(snap["T_K"] - snap_l["T_K"])
        dX = max((abs(a - b) for a, b in zip(snap["conversion"].values(), snap_l["conversion"].values())
                  if a is not None and b is not None), default=0.0)
        el_err = _element_error(el_in, _elements(end))
        if energy == "adiabatic":
            e_end = end.int_energy_mass if reactor == "batch_constant_volume" else end.enthalpy_mass
            e_in = u_in if reactor == "batch_constant_volume" else h_in
            e_err = abs(e_end - e_in) / (abs(e_in) + cpT)
        else:
            e_err = 0.0
        T_check = T_K if reactor.startswith("batch") else max(T_op, T_K)
        if reactor.startswith("batch") and energy == "adiabatic":
            T_check = T_K          # ignition chemistry is decided at the initial temperature
    else:
        taus = [float(t) for t in (residence_times_s or [])]
        if not taus or any(t <= 0 for t in taus) or len(taus) > 20:
            raise ValueError("residence_times_s: 1-20 positive values (s) are required for wsr")
        taus.sort()
        g_eq = gas_factory()
        g_eq.equilibrate("HP" if energy == "adiabatic" else "TP")
        T_eq = float(g_eq.T)
        rows, dT, dX, el_err, e_err, worst_drift = [], 0.0, 0.0, 0.0, 0.0, (0.0, 0.0)
        not_conv = []
        for tau in taus:
            a = _wsr(gas_factory, energy, tau, initial_state, TOL_TIGHT)
            b = _wsr(gas_factory, energy, tau, initial_state, TOL_LOOSE)
            ph = a["phase"]
            top = [ph.species_names[i] for i in np.argsort(ph.X)[::-1][:6]]
            snap = _snapshot(ph, Y_in, key, list(dict.fromkeys(show_base + top)))
            snap_l = _snapshot(b["phase"], Y_in, key, list(snap["mole_fractions"]))
            rise = (snap["T_K"] - T_K) / (T_eq - T_K) if energy == "adiabatic" and T_eq - T_K > 100 else None
            state = None if rise is None else ("encendido" if rise > 0.5 else "apagado" if rise < 0.1 else "parcial")
            rows.append({"residence_time_s": tau, **snap, "state": state, "mass_flow_kg_s_per_m3": a["mdot_per_m3"]})
            dT = max(dT, abs(snap["T_K"] - snap_l["T_K"]))
            dX = max([dX] + [abs(x - y) for x, y in zip(snap["conversion"].values(), snap_l["conversion"].values())
                              if x is not None and y is not None])
            el_err = max(el_err, _element_error(el_in, _elements(ph)))
            if energy == "adiabatic":
                e_err = max(e_err, abs(ph.enthalpy_mass - h_in) / (abs(h_in) + cpT))
            worst_drift = (max(worst_drift[0], a["drift_T"]), max(worst_drift[1], a["drift_Y"]))
            if not a["converged"] or a["drift_T"] > 0.5 or a["drift_Y"] > 1e-4:
                not_conv.append(tau)
        out["T_equilibrium_K"] = T_eq
        out["sweep"] = rows
        states = [r["state"] for r in rows]
        if "apagado" in states and "encendido" in states:
            last_off = max(r["residence_time_s"] for r in rows if r["state"] == "apagado")
            first_on = min(r["residence_time_s"] for r in rows if r["state"] == "encendido")
            out["blow_out_between_s"] = [last_off, first_on]
        ev["steady_state"] = not not_conv
        ev["steady_state_drift"] = {"T_K": worst_drift[0], "Y": worst_drift[1]}
        if not_conv:
            reasons.append(f"sin estado estacionario estable para τ = {not_conv} s")
        T_op = max([T_eq if initial_state == "ignited" else T_K] + [r["T_K"] for r in rows])
        T_check = T_op
    ev["tolerance_independent"] = dT < 0.5 and dX < 1e-3
    ev["tolerance_delta"] = {"T_K": dT, "X": dX}
    if not ev["tolerance_independent"]:
        reasons.append(f"el resultado cambia con la tolerancia del integrador (ΔT {dT:.2f} K, ΔX {dX:.1e})")
    ev["element_balance_closed"] = el_err < 1e-8
    ev["element_error"] = el_err
    if el_err >= 1e-8:
        reasons.append(f"balance de elementos no cierra ({el_err:.1e})")
    ev["energy_conserved"] = e_err < 1e-6 if energy == "adiabatic" else None
    ev["energy_error"] = e_err if energy == "adiabatic" else None
    if energy == "adiabatic" and e_err >= 1e-6:
        reasons.append(f"la energía no se conserva ({e_err:.1e})")
    ok, rs, note = _range_check(mechanism, custom, T_check, P_Pa, phi)
    ev["within_mechanism_range"] = ok
    reasons += rs
    out["mechanism_validity"] = note
    ev["verified"] = not reasons
    ev["reasons"] = reasons
    out["evidence"] = ev
    notes = {"batch_constant_volume": "Batch adiabático o isotérmico a volumen constante (bomba, cámara cerrada).",
             "batch_constant_pressure": "Batch a presión constante.",
             "pfr": "PFR ideal en marco lagrangiano: batch a presión constante con t = tiempo de residencia (sin ΔP ni "
                    "dispersión axial).",
             "wsr": "Reactor perfectamente agitado (WSR/CSTR) en estado estacionario por cada τ; con initial_state 'ignited' "
                    "arranca en equilibrio y encuentra la rama encendida (multiplicidad: desde 'feed' puede no encender)."}
    out["notes"] = notes[reactor] + (" Cinética del usuario con termo de cp constante: solo isotérmico." if custom else "")
    return out


def dependencies_report() -> dict:
    return {"cantera": ct.__version__, "mechanisms": sorted(_mechanism_files())}
