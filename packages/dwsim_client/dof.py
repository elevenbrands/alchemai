"""Deterministic degrees-of-freedom check for a flowsheet built with the tools layer, BEFORE solve().

Sequential-modular logic (what DWSIM actually needs to run a unit):
  * every feed stream that no unit produces must carry a composition, one flow and two of (T, P, vapor fraction);
  * a stream produced by a unit must NOT carry T/P/flow specs (DWSIM would overwrite them: over-specification);
  * each unit kind needs a fixed set of ports and a fixed set of specs; a missing one → "under-specified",
    an extra or mutually-exclusive one (outlet_T_K + heat_added_kW, adiabatic + polytropic efficiency) → "over-specified";
  * splitter ratios must match the number of products and sum to 1 (±1e-6); recycle needs a tear stream.

It is a bookkeeping check on the SPECIFICATION, not a physics check: a pump told to lower pressure is well-posed here and
fails later in solve() (negative power) — that is the agent's job to interpret. Output is the agent's/human's checklist.
"""
from __future__ import annotations

from typing import Any

ONE_OF = "one_of"

# ports: (min_feeds, max_feeds, n_products, n_energy_feeds, n_energy_products); specs: required keys, groups of ONE_OF
RULES: dict[str, dict[str, Any]] = {
    "mixer":              dict(ports=(2, 99, 1, 0, 0), required=[], one_of=[]),
    "splitter":           dict(ports=(1, 1, None, 0, 0), required=["split_ratios"], one_of=[]),
    "heater":             dict(ports=(1, 1, 1, None, None), required=[], one_of=[("outlet_T_K", "heat_added_kW")]),
    "cooler":             dict(ports=(1, 1, 1, None, None), required=[], one_of=[("outlet_T_K", "heat_removed_kW")]),
    "valve":              dict(ports=(1, 1, 1, 0, 0), required=["outlet_P_Pa"], one_of=[]),
    "pump":               dict(ports=(1, 1, 1, None, None), required=["outlet_P_Pa"], one_of=[], optional=["efficiency_pct"]),
    "compressor":         dict(ports=(1, 1, 1, None, None), required=["outlet_P_Pa"], one_of=[("adiabatic_eff_pct", "polytropic_eff_pct")], optional_one_of=True),
    "expander":           dict(ports=(1, 1, 1, None, None), required=["outlet_P_Pa"], one_of=[("adiabatic_eff_pct", "polytropic_eff_pct")], optional_one_of=True),
    "separator":          dict(ports=(1, 99, 2, None, None), required=[], one_of=[]),
    "heat_exchanger":     dict(ports=(2, 2, 2, 0, 0), required=["UA_W_K"], one_of=[], optional=["hot_dP_Pa", "cold_dP_Pa"]),
    "shortcut_column":    dict(ports=(1, 1, 2, 1, 1), required=["light_key", "heavy_key", "lk_mole_frac_in_bottoms", "hk_mole_frac_in_distillate",
                                                                "condenser_P_Pa", "reboiler_P_Pa", "reflux_ratio"], one_of=[]),
    "distillation_column": dict(ports=(1, 1, 2, None, None), required=["stages", "reflux_ratio", "top_P_Pa"],
                                one_of=[("bottoms_molar_flow_mol_s", "distillate_molar_flow_mol_s")], optional=["feed_stage", "dP_Pa"]),
    "conversion_reactor": dict(ports=(1, 99, 2, 1, 0), required=["reaction_set"], one_of=[], optional=["mode", "dP_Pa"]),
    "equilibrium_reactor": dict(ports=(1, 99, 2, 1, 0), required=["reaction_set"], one_of=[], optional=["mode", "dP_Pa"]),
    "gibbs_reactor":      dict(ports=(1, 99, 2, 1, 0), required=[], one_of=[], optional=["reaction_set", "mode", "dP_Pa"]),
    "cstr":               dict(ports=(1, 99, 1, 1, 0), required=["volume_m3", "reaction_set"], one_of=[], optional=["mode", "dP_Pa"]),
    "pfr":                dict(ports=(1, 99, 1, 1, 0), required=["volume_m3", "reaction_set"], one_of=[], optional=["mode", "dP_Pa"]),
    "recycle":            dict(ports=(1, 1, 1, 0, 0), required=[], one_of=[]),
}
STREAM_STATE = ("temperature_K", "pressure_Pa", "vapor_fraction")
STREAM_FLOW = ("molar_flow_mol_s", "mass_flow_kg_s")
STREAM_COMP = ("mole_fractions", "compound_molar_flows_mol_s")


def analyse(session: dict) -> dict:
    """Pure function over the session records (streams, specs, params, connections)."""
    specs, params, conn, kinds = session.get("stream_specs", {}), session.get("params", {}), session.get("conn", {}), session.get("kinds", {})
    produced = {p: u for u, c in conn.items() for p in c["products"]}
    consumed: dict[str, list[str]] = {}
    for u, c in conn.items():
        for f in c["feeds"]:
            consumed.setdefault(f, []).append(u)
    streams, units, problems = {}, {}, []
    for name in session.get("streams", {}):
        sp = specs.get(name, {})
        has_state = [k for k in STREAM_STATE if k in sp]
        has_flow = [k for k in STREAM_FLOW if k in sp]
        has_comp = [k for k in STREAM_COMP if k in sp]
        if "compound_molar_flows_mol_s" in sp and not has_flow:
            has_flow = ["compound_molar_flows_mol_s"]
        entry: dict[str, Any] = {"produced_by": produced.get(name), "consumed_by": consumed.get(name, [])}
        if name in produced:
            extra = has_state + [k for k in STREAM_FLOW if k in sp] + [k for k in STREAM_COMP if k in sp]
            entry["status"] = "over-specified" if extra else "ok"
            if extra:
                entry["extra"] = extra
                problems.append(f"stream {name}: produced by {produced[name]} but also specified ({', '.join(extra)}) — DWSIM will overwrite it")
        else:
            missing = []
            if len(has_state) < 2:
                missing.append(f"state: {2 - len(has_state)} more of (T, P, vapor_fraction)")
            if len(has_state) > 2:
                missing.append("state: T, P and vapor_fraction all given (over-specified)")
            if not has_flow:
                missing.append("flow (molar_flow_mol_s | mass_flow_kg_s)")
            if not has_comp:
                missing.append("composition (mole_fractions | compound_molar_flows_mol_s)")
            if "mole_fractions" in sp:
                tot = sum(float(v) for v in sp["mole_fractions"].values())
                if abs(tot - 1.0) > 1e-4:
                    missing.append(f"mole_fractions sum to {tot:.4f}, not 1")
            if not consumed.get(name):
                entry["note"] = "feed stream not connected to any unit"
            entry["status"] = "under-specified" if missing else "ok"
            if missing:
                entry["missing"] = missing
                if consumed.get(name) or not conn:
                    problems.append(f"feed stream {name}: missing {'; '.join(missing)}")
        streams[name] = entry
    for u, kind in kinds.items():
        rule = RULES.get(kind)
        p = params.get(u, {})
        c = conn.get(u)
        entry = {"kind": kind, "missing": [], "extra": [], "status": "ok"}
        if rule is None:
            entry["status"] = "unknown kind"; units[u] = entry; continue
        if c is None:
            entry["missing"].append("not connected (call connect())"); entry["status"] = "unconnected"; units[u] = entry
            problems.append(f"unit {u} ({kind}): not connected"); continue
        fmin, fmax, nprod, nef, nep = rule["ports"]
        nf, npd, nefa, nepa = len(c["feeds"]), len(c["products"]), len(c["energy_feeds"]), len(c["energy_products"])
        if nf < fmin:
            entry["missing"].append(f"{fmin - nf} feed stream(s)")
        if nf > fmax:
            entry["extra"].append(f"{nf - fmax} feed stream(s) too many")
        if kind == "splitter":
            nprod = len(p.get("split_ratios", []) or [])
            if nprod and abs(sum(map(float, p["split_ratios"])) - 1.0) > 1e-6:
                entry["extra"].append(f"split_ratios sum to {sum(map(float, p['split_ratios'])):.4f}, not 1 (mass balance)")
        if nprod is not None and npd != nprod:
            (entry["missing"] if npd < nprod else entry["extra"]).append(f"products: {npd} connected, {nprod} required")
        if nef is not None and nefa != nef:
            (entry["missing"] if nefa < nef else entry["extra"]).append(f"energy_feeds: {nefa} connected, {nef} required")
        if nep is not None and nepa != nep:
            (entry["missing"] if nepa < nep else entry["extra"]).append(f"energy_products: {nepa} connected, {nep} required")
        for k in rule["required"]:
            if k not in p:
                entry["missing"].append(f"param {k}")
        for group in rule["one_of"]:
            present = [k for k in group if k in p]
            if len(present) > 1:
                entry["extra"].append(f"mutually exclusive params both given: {', '.join(present)}")
            elif not present and not rule.get("optional_one_of"):
                entry["missing"].append(f"param one of ({' | '.join(group)})")
        known = set(rule["required"]) | {k for g in rule["one_of"] for k in g} | set(rule.get("optional", []))
        for k in p:
            if k not in known:
                entry["extra"].append(f"param {k} is not used by {kind}")
        if entry["missing"] and entry["extra"]:
            entry["status"] = "under- and over-specified"
        elif entry["missing"]:
            entry["status"] = "under-specified"
        elif entry["extra"]:
            entry["status"] = "over-specified"
        if entry["status"] != "ok":
            problems.append(f"unit {u} ({kind}): {entry['status']} — " + "; ".join(entry["missing"] + entry["extra"]))
        units[u] = entry
    dangling = [s for s, e in streams.items() if not e["produced_by"] and not e["consumed_by"]]
    return {"ok": not problems, "n_problems": len(problems), "problems": problems, "streams": streams, "units": units,
            "dangling_streams": dangling}


def dof_check(flowsheet_id: str) -> dict:
    """Degrees-of-freedom / specification check BEFORE solve(): for every feed stream (needs composition, one flow and two
    of T, P, vapor fraction; a stream produced by a unit must not be specified) and every unit (required ports and params by
    kind; mutually exclusive specs such as outlet_T_K + heat_added_kW or adiabatic + polytropic efficiency; splitter ratios
    summing to 1). Returns ok, a list of problems in plain words and per-stream/per-unit status
    (ok | under-specified | over-specified | unconnected). This is bookkeeping on the specification, not physics: a feasible
    DOF count can still be physically impossible (that shows up in solve()). Call it when a statement looks incomplete or
    contradictory, and whenever solve() fails."""
    from packages.dwsim_client import tools as T
    return analyse(T._s(flowsheet_id))
