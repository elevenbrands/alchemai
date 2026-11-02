"""Sweep / re-spec tools: let the agent search in ONE call instead of one model turn per iteration.

Design questions ("which UA gives 330 K?", "which T gives VF = 0.5?") need many solves. Re-specifying
a unit or a feed stream on the existing flowsheet and re-solving is cheap in DWSIM; what is expensive
is a model turn per iteration. `sweep` runs the whole scan server-side and returns the table (plus the
interpolated crossing if a target is given).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.dwsim_client import tools as T  # noqa: E402

Q = T.Q


def set_unit_param(flowsheet_id: str, unit: str, param: str, value: float) -> dict:
    """Re-specify one numeric parameter of an existing unit (same names as add_unit params: outlet_T_K,
    outlet_P_Pa, UA_W_K, reflux_ratio, adiabatic_eff_pct, efficiency_pct, heat_added_kW, heat_removed_kW,
    split_ratio_0 (first ratio; the second becomes 1-x)). Call solve() afterwards."""
    s = T._s(flowsheet_id)
    b, kind = s["units"][unit], s["kinds"][unit]
    v = float(value)
    if param == "outlet_T_K":
        b.WithOutletTemperature(Q.Kelvin(v))
    elif param == "outlet_P_Pa":
        b.WithOutletPressure(Q.Pascal(v))
    elif param == "UA_W_K":
        b.WithGlobalUA(v)
    elif param == "reflux_ratio":
        b.WithRefluxRatio(v) if kind == "shortcut_column" else b.WithCondenserSpec("Reflux Ratio", v, "")
    elif param == "adiabatic_eff_pct":
        b.WithAdiabaticEfficiencyPercent(v)
    elif param == "polytropic_eff_pct":
        b.WithPolytropicEfficiencyPercent(v)
    elif param == "efficiency_pct":
        b.WithEfficiencyPercent(v)
    elif param == "heat_added_kW":
        b.WithHeatAdded(Q.Kilowatt(v))
    elif param == "heat_removed_kW":
        b.WithHeatRemoved(Q.Kilowatt(v))
    elif param == "split_ratio_0":
        b.WithSplitRatios(v, 1.0 - v)
    else:
        raise ValueError(f"unsupported param {param!r}")
    return {"ok": True, "unit": unit, "param": param, "value": v}


def set_stream_spec(flowsheet_id: str, stream: str, temperature_K: float | None = None, pressure_Pa: float | None = None,
                    vapor_fraction: float | None = None, molar_flow_mol_s: float | None = None,
                    mass_flow_kg_s: float | None = None, mole_fractions: dict[str, float] | None = None,
                    compound_molar_flows_mol_s: dict[str, float] | None = None) -> dict:
    """Re-specify a feed stream (any subset of T, P, vapor fraction, flow, composition). Call solve() afterwards."""
    s = T._s(flowsheet_id)
    b = s["streams"][stream]
    if temperature_K is not None:
        b.WithTemperature(Q.Kelvin(float(temperature_K)))
    if pressure_Pa is not None:
        b.WithPressure(Q.Pascal(float(pressure_Pa)))
    if vapor_fraction is not None:
        b.WithVaporFraction(float(vapor_fraction))
    if molar_flow_mol_s is not None:
        b.WithMolarFlow(Q.MolPerSecond(float(molar_flow_mol_s)))
    if mass_flow_kg_s is not None:
        b.WithMassFlow(Q.KgPerSecond(float(mass_flow_kg_s)))
    if mole_fractions:
        b.WithComposition(T._composition(mole_fractions))
    if compound_molar_flows_mol_s:
        for k, v in compound_molar_flows_mol_s.items():
            b.SetCompoundMolarFlow(k, float(v))
    return {"ok": True, "stream": stream}


def _readout(flowsheet_id: str, kind: str, name: str, key: str) -> float:
    d = T.read_unit(flowsheet_id, name) if kind == "unit" else T.read_stream(flowsheet_id, name)
    if key.startswith("mole_fractions."):
        return float(d["mole_fractions"][key.split(".", 1)[1]])
    if key.startswith("energy_streams_kW."):
        return float(d["energy_streams_kW"][key.split(".", 1)[1]])
    return float(d[key])


def sweep(flowsheet_id: str, vary_kind: str, vary_name: str, vary_param: str, values: list[float],
          read_kind: str, read_name: str, read_key: str, target: float | None = None) -> dict:
    """Vary one spec over `values`, re-solve each time and read one quantity: returns a table and, if `target`
    is given, the value where the readout crosses the target (linear interpolation between the bracketing
    points). vary_kind/read_kind ∈ {"unit", "stream"}; vary_param as in set_unit_param, or a stream spec
    name (temperature_K, pressure_Pa, vapor_fraction, molar_flow_mol_s, mass_flow_kg_s); read_key as in
    read_unit/read_stream (e.g. "T_K", "vapor_fraction", "power_kW", "mole_fractions.Ethanol",
    "energy_streams_kW.Qr"). Only rows with ok=true are trustworthy."""
    rows: list[dict[str, Any]] = []
    for v in values:
        if vary_kind == "unit":
            set_unit_param(flowsheet_id, vary_name, vary_param, float(v))
        else:
            set_stream_spec(flowsheet_id, vary_name, **{vary_param: float(v)})
        sol = T.solve(flowsheet_id)
        try:
            y = _readout(flowsheet_id, read_kind, read_name, read_key)
        except Exception as ex:  # noqa: BLE001
            y = None
            sol["solver_errors"] = sol.get("solver_errors", []) + [f"readout failed: {ex}"]
        rows.append({"value": float(v), "readout": y, "ok": bool(sol["ok"]), "errors": sol["solver_errors"][:2]})
    out: dict[str, Any] = {"rows": rows}
    if target is not None:
        good = [r for r in rows if r["ok"] and r["readout"] is not None]
        for a, b in zip(good, good[1:]):
            da, db = a["readout"] - target, b["readout"] - target
            if da == 0 or da * db < 0:
                x = a["value"] + (b["value"] - a["value"]) * da / (da - db) if da != db else a["value"]
                out["crossing"] = {"value": x, "bracket": [a["value"], b["value"]],
                                   "note": "linear interpolation; re-solve at this value and verify, or sweep a finer range"}
                break
        else:
            out["crossing"] = None
    return out


SWEEP_TOOLS = [set_unit_param, set_stream_spec, sweep]
