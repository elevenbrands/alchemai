"""Generic one-variable design optimizer for the agent (bloque 21).

Varies one unit parameter or feed-stream spec between bounds, re-solves DWSIM at each point and minimizes (or
maximizes) an objective: the total annual cost from flowsheet_cost, or a weighted sum of readouts (e.g. total
compressor power). Process limits (typically from the HAZOP: maximum discharge temperature, minimum approach,
maximum pressure) are hard constraints: a point that violates one, or that does not solve with closed balances, is
infeasible. Method: scan on a linear or log grid, then golden-section on the bracket around the best feasible point.
The whole curve comes back as evidence, plus how flat the optimum is.
"""
from __future__ import annotations

import math
from typing import Any

from packages.dwsim_client import tools as T
from packages.dwsim_client.tools_sweep import _readout, _with_fraction, set_stream_spec, set_unit_param


def _set(fid: str, kind: str, name: str, param: str, v: float):
    if kind == "unit":
        set_unit_param(fid, name, param, v)
    elif param.startswith("mole_fraction."):
        set_stream_spec(fid, name, mole_fractions=_with_fraction(fid, name, param.split(".", 1)[1], v))
    else:
        set_stream_spec(fid, name, **{param: v})


def _eval(fid: str, x: float, var: dict, objective: dict, constraints: list[dict]) -> dict:
    _set(fid, var["kind"], var["name"], var["param"], x)
    sol = T.solve(fid)
    row: dict[str, Any] = {"x": x, "solved_ok": bool(sol["ok"]), "feasible": False, "objective": None, "violations": []}
    if not sol["ok"]:
        row["violations"].append(f"solve not ok: {sol.get('solver_errors', [])[:1]}")
        return row
    cons = {}
    for c in constraints:
        v = _readout(fid, c["kind"], c["name"], c["key"])
        cons[f"{c['name']}.{c['key']}"] = v
        if "max" in c and v > c["max"]:
            row["violations"].append(f"{c['name']}.{c['key']} ≤ {c['max']} (vecino: {v:.6g})")
        if "min" in c and v < c["min"]:
            row["violations"].append(f"{c['name']}.{c['key']} ≥ {c['min']} (vecino: {v:.6g})")
    row["constraints"] = cons
    if objective["type"] == "total_annual_cost":
        from packages.dwsim_client.tools_costing import flowsheet_cost
        a = {k: v for k, v in objective.items() if k != "type"}
        r = flowsheet_cost(fid, **a)
        y = r["total_annual_cost_usd_y"]
        row.update(capital_usd_y=r["annualized_capital_usd_y"], operating_usd_y=r["operating_cost_usd_y"], areas_m2=r.get("areas_m2"),
                   not_costed=r["not_costed"])
    elif objective["type"] == "sum":
        terms = {f"{t['name']}.{t['key']}": float(t.get("weight", 1.0)) * _readout(fid, t["kind"], t["name"], t["key"]) for t in objective["terms"]}
        row["terms"] = terms
        y = sum(terms.values())
    else:
        raise ValueError("objective.type must be total_annual_cost or sum")
    row["objective"] = float(y)
    row["feasible"] = not row["violations"]
    return row


def optimize_design(flowsheet_id: str, vary_kind: str, vary_name: str, vary_param: str, lower: float, upper: float,
                    objective: dict, constraints: list[dict] | None = None, maximize: bool = False, log_scale: bool = False,
                    n_scan: int = 9, refine_iterations: int = 12) -> dict:
    """Optimize ONE design variable of a built flowsheet in a single call.
    vary_kind "unit"|"stream", vary_name, vary_param as in sweep (outlet_P_Pa, UA_W_K, reflux_ratio, outlet_T_K,
    temperature_K, mass_flow_kg_s, mole_fraction.<C>, ...), between lower and upper (log_scale for ranges spanning decades).
    objective: {"type": "total_annual_cost", <flowsheet_cost args: prices, hours_per_year, interest, years,
    exchanger_specs, options, lang, country, year>} or {"type": "sum", "terms": [{"kind", "name", "key", "weight"}]}
    (e.g. both compressor powers: key "duty_or_power_kW"). constraints: [{"kind", "name", "key", "max"|"min"}], hard
    limits such as a HAZOP discharge-temperature limit; violating or unsolved points are infeasible.
    Scans n_scan points then golden-section refines the bracket of the best feasible point; leaves the flowsheet solved at
    the optimum. Returns best {x, objective, constraints}, the curve, `active_constraint` when the optimum sits on a limit,
    and `flat_range` (x values within 1 % of the optimum: the decision is not sensitive inside it)."""
    cons = constraints or []
    sgn = -1.0 if maximize else 1.0
    if log_scale:
        if lower <= 0:
            raise ValueError("log_scale needs lower > 0")
        grid = [lower * (upper / lower) ** (i / (n_scan - 1)) for i in range(n_scan)]
    else:
        grid = [lower + (upper - lower) * i / (n_scan - 1) for i in range(n_scan)]
    curve = [_eval(flowsheet_id, x, {"kind": vary_kind, "name": vary_name, "param": vary_param}, objective, cons) for x in grid]
    key = lambda r: sgn * r["objective"] if r["feasible"] else math.inf
    feas = [i for i, r in enumerate(curve) if r["feasible"]]
    if not feas:
        return {"feasible": False, "curve": curve, "note": "no feasible point in the range: widen the bounds or relax a limit"}
    k = min(feas, key=lambda i: key(curve[i]))
    a, b = grid[max(0, k - 1)], grid[min(n_scan - 1, k + 1)]
    tf = (lambda v: math.log(v)) if log_scale else (lambda v: v)
    itf = (lambda v: math.exp(v)) if log_scale else (lambda v: v)
    var = {"kind": vary_kind, "name": vary_name, "param": vary_param}
    la, lb = tf(a), tf(b)
    g = (math.sqrt(5) - 1) / 2
    lc, ld = lb - g * (lb - la), la + g * (lb - la)
    rc, rd = _eval(flowsheet_id, itf(lc), var, objective, cons), _eval(flowsheet_id, itf(ld), var, objective, cons)
    refine = [rc, rd]
    for _ in range(refine_iterations):
        if key(rc) <= key(rd):
            lb, ld, rd = ld, lc, rc
            lc = lb - g * (lb - la)
            rc = _eval(flowsheet_id, itf(lc), var, objective, cons)
            refine.append(rc)
        else:
            la, lc, rc = lc, ld, rd
            ld = la + g * (lb - la)
            rd = _eval(flowsheet_id, itf(ld), var, objective, cons)
            refine.append(rd)
    allpts = sorted(curve + refine, key=lambda r: r["x"])
    best = min((r for r in allpts if r["feasible"]), key=key)
    final = _eval(flowsheet_id, best["x"], var, objective, cons)          # leave the flowsheet at the optimum
    # an optimum next to an infeasible neighbour sits on a constraint
    idx = allpts.index(best)
    nb = [allpts[j] for j in (idx - 1, idx + 1) if 0 <= j < len(allpts)]
    active = sorted({v for r in nb if not r["feasible"] for v in r["violations"]})
    flat = [r["x"] for r in allpts if r["feasible"] and abs(r["objective"] - best["objective"]) <= 0.01 * abs(best["objective"])]
    at_bound = best["x"] in (grid[0], grid[-1])
    return {"feasible": True, "best": final, "curve": [{k: r[k] for k in ("x", "objective", "feasible", "violations")} for r in allpts],
            "active_constraint": active or None, "flat_range": [min(flat), max(flat)] if flat else None,
            "at_bound": at_bound, "n_evaluations": len(curve) + len(refine) + 1,
            "note": ("optimum at a bound of the range: widen it" if at_bound else
                     "re-check the optimum with solve() and report the curve; inside flat_range the choice is not sensitive")}


OPTIMIZE_TOOLS = [optimize_design]
