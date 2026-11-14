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


def _eval_current(fid: str, objective: dict, constraints: list[dict]) -> dict:
    """Solve at the current specs; objective, constraint values and a relative violation measure."""
    r = _eval(fid, None, None, objective, constraints)
    viol = 0.0
    for c in constraints:
        v = (r.get("constraints") or {}).get(f"{c['name']}.{c['key']}")
        if v is None:
            continue
        if "max" in c and v > c["max"]:
            viol += (v - c["max"]) / max(abs(c["max"]), 1e-9)
        if "min" in c and v < c["min"]:
            viol += (c["min"] - v) / max(abs(c["min"]), 1e-9)
    if not r["solved_ok"]:
        viol += 1.0
    r["violation"] = viol
    return r


def _eval(fid: str, x: float | None, var: dict | None, objective: dict, constraints: list[dict]) -> dict:
    if var is not None:
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


def optimize_design_multi(flowsheet_id: str, variables: list[dict], objective: dict, constraints: list[dict] | None = None,
                          maximize: bool = False, grid_points: int = 5, max_evaluations: int = 150, tolerance: float = 1e-4) -> dict:
    """Optimize 1-4 design variables together (they may interact, like the intermediate pressures of a multistage
    compressor). variables: [{"kind": "unit"|"stream", "name", "param", "lower", "upper", "log_scale": bool, "integer": bool}]
    with params as in optimize_design; "integer": true for counts (column stages: the column is rebuilt at each value).
    objective and constraints exactly as in optimize_design. Method: a coarse grid (grid_points per variable, fewer when
    there are more variables) picks the start, then Nelder-Mead in normalized coordinates over the continuous variables;
    with integer variables, every integer value in range (up to 12, else a coarse set) gets its own continuous
    sub-optimization and the best feasible wins. Constraint violations and unsolved points are penalized, and the
    reported optimum is always a feasible evaluated point.
    Returns best values, objective, active constraints, the path and how many solves it took; leaves the flowsheet solved
    at the optimum."""
    import itertools
    from scipy.optimize import minimize
    cons = constraints or []
    if not 1 <= len(variables) <= 4:
        raise ValueError("1 to 4 variables")
    sgn = -1.0 if maximize else 1.0
    ints = [i for i, v in enumerate(variables) if v.get("integer")]
    if ints:
        return _optimize_with_integers(flowsheet_id, variables, objective, cons, maximize, grid_points, max_evaluations, tolerance, ints)
    lo = [float(v["lower"]) for v in variables]
    hi = [float(v["upper"]) for v in variables]
    logs = [bool(v.get("log_scale")) for v in variables]
    to_x = lambda z: [math.exp(math.log(l) + zi * (math.log(h) - math.log(l))) if lg else l + zi * (h - l)
                      for zi, l, h, lg in zip(z, lo, hi, logs)]
    evals: list[dict] = []
    ref = {"scale": None}

    def run(z):
        z = [min(1.0, max(0.0, float(v))) for v in z]
        xs = to_x(z)
        for v, x in zip(variables, xs):
            _set(flowsheet_id, v["kind"], v["name"], v["param"], x)
        r = _eval_current(flowsheet_id, objective, cons)
        r["x"] = xs
        evals.append(r)
        return r

    def f(z):
        r = run(z)
        if r["objective"] is None:
            return 1e12
        y = sgn * r["objective"]
        if ref["scale"] is None:
            ref["scale"] = abs(y) + 1.0
        return y + 10.0 * ref["scale"] * r["violation"]

    n = len(variables)
    gp = max(3, grid_points if n <= 2 else 3)
    grid = [[i / (gp - 1) for i in range(gp)] for _ in range(n)]
    starts = [(f(list(z)), list(z)) for z in itertools.product(*grid)]
    z0 = min(starts)[1]
    budget = max(20, max_evaluations - len(starts))
    res = minimize(f, z0, method="Nelder-Mead", bounds=[(0.0, 1.0)] * n,
                   options={"maxfev": budget, "xatol": tolerance, "fatol": tolerance * ref["scale"] if ref["scale"] else 1e-8})
    feas = [r for r in evals if r["feasible"]]
    if not feas:
        return {"feasible": False, "n_evaluations": len(evals), "note": "no feasible point found: widen bounds or relax a limit"}
    best = min(feas, key=lambda r: sgn * r["objective"])
    for v, x in zip(variables, best["x"]):
        _set(flowsheet_id, v["kind"], v["name"], v["param"], x)
    final = _eval_current(flowsheet_id, objective, cons)
    final["x"] = best["x"]
    near = [r for r in evals if not r["feasible"] and r["objective"] is not None
            and all(abs(a - b) <= 0.02 * abs(h - l) for a, b, l, h in zip(r["x"], best["x"], lo, hi))]
    at_bound = [v["name"] + "." + v["param"] for v, x, l, h in zip(variables, best["x"], lo, hi) if min(x - l, h - x) <= 1e-3 * (h - l)]
    return {"feasible": True, "best": {"values": {f"{v['name']}.{v['param']}": x for v, x in zip(variables, best["x"])},
                                       "objective": final["objective"], "constraints": final.get("constraints"), "terms": final.get("terms")},
            "active_constraints": sorted({m.split(" (vecino")[0] for r in near for m in r["violations"]}) or None,
            "variables_at_bound": at_bound or None,
            "n_evaluations": len(evals) + 1, "optimizer_message": str(res.message),
            "path": [{"x": r["x"], "objective": r["objective"], "feasible": r["feasible"]} for r in evals[:: max(1, len(evals) // 40)]],
            "note": "variables at a bound: widen the range" if at_bound else "verify the optimum with solve() and report it with the path"}


def _optimize_with_integers(fid, variables, objective, cons, maximize, grid_points, max_evaluations, tolerance, ints) -> dict:
    """Enumerate the integer variables (full range when ≤ 12 values per variable, else ~6 evenly spaced values and a
    refinement around the best), optimizing the continuous ones at each combination."""
    import itertools
    sgn = -1.0 if maximize else 1.0
    cont = [v for i, v in enumerate(variables) if i not in ints]

    def int_values(v):
        lo, hi = int(round(v["lower"])), int(round(v["upper"]))
        vals = list(range(lo, hi + 1))
        if len(vals) > 12:
            step = max(1, (hi - lo) // 5)
            vals = sorted({lo + k * step for k in range(6)} | {hi})
        return vals

    combos = list(itertools.product(*[int_values(variables[i]) for i in ints]))
    per = max(8, max_evaluations // max(1, len(combos)))
    results, n_evals = [], 0

    def run_combo(c):
        nonlocal n_evals
        for i, val in zip(ints, c):
            _set(fid, variables[i]["kind"], variables[i]["name"], variables[i]["param"], float(val))
        if cont:
            r = optimize_design_multi(fid, cont, objective, cons, maximize, grid_points, per, tolerance)
            n_evals += r.get("n_evaluations", 0)
        else:
            e = _eval_current(fid, objective, cons)
            n_evals += 1
            r = ({"feasible": True, "best": {"values": {}, "objective": e["objective"], "constraints": e.get("constraints"), "terms": e.get("terms")},
                  "active_constraints": None, "variables_at_bound": None} if e["feasible"] else {"feasible": False})
        r["integers"] = {f"{variables[i]['name']}.{variables[i]['param']}": val for i, val in zip(ints, c)}
        return r

    for c in combos:
        results.append(run_combo(c))
    feas = [r for r in results if r["feasible"]]
    full = len(list(itertools.product(*[range(int(round(variables[i]["lower"])), int(round(variables[i]["upper"])) + 1) for i in ints])))
    rounds = 0
    while feas and len(combos) < full and rounds < 8:          # refine ±2 around the best until it is interior
        rounds += 1
        best = min(feas, key=lambda r: sgn * r["best"]["objective"])
        bc = tuple(best["integers"].values())
        seen = {tuple(r["integers"].values()) for r in results}
        new = [c for c in itertools.product(*[range(int(b) - 2, int(b) + 3) for b in bc])
               if c not in seen and all(variables[i]["lower"] <= val <= variables[i]["upper"] for i, val in zip(ints, c))]
        if not new:
            break
        for c in new:
            results.append(run_combo(c))
        feas = [r for r in results if r["feasible"]]
    if not feas:
        return {"feasible": False, "n_evaluations": n_evals, "note": "no feasible point found: widen bounds or relax a limit"}
    best = min(feas, key=lambda r: sgn * r["best"]["objective"])
    for i, val in best["integers"].items():
        name, param = i.rsplit(".", 1)
        kind = next(v["kind"] for v in variables if v["name"] == name and v["param"] == param)
        _set(fid, kind, name, param, float(val))
    for k, x in best["best"]["values"].items():
        name, param = k.rsplit(".", 1)
        kind = next(v["kind"] for v in variables if v["name"] == name and v["param"] == param)
        _set(fid, kind, name, param, x)
    final = _eval_current(fid, objective, cons)
    curve = sorted(({"integers": r["integers"], "objective": r["best"]["objective"] if r["feasible"] else None,
                     "continuous": r["best"]["values"] if r["feasible"] else None, "feasible": r["feasible"]} for r in results),
                   key=lambda d: tuple(d["integers"].values()))
    ilo = [variables[i]["lower"] for i in ints]; ihi = [variables[i]["upper"] for i in ints]
    at_bound = [k for (k, val), l, h in zip(best["integers"].items(), ilo, ihi) if val in (int(round(l)), int(round(h)))] + (best.get("variables_at_bound") or [])
    return {"feasible": True, "best": {"values": best["integers"] | best["best"]["values"], "objective": final["objective"],
                                       "constraints": final.get("constraints"), "terms": final.get("terms")},
            "active_constraints": best.get("active_constraints"), "variables_at_bound": at_bound or None,
            "n_evaluations": n_evals + 1, "integer_curve": curve,
            "note": "integer variable at a bound: widen the range" if at_bound else "integer_curve shows the objective at each stage count; report how flat it is"}


def eo_column_optimize(components: list[str], feed_flow_mol_s: float, feed_T_K: float, feed_P_Pa: float,
                       feed_mole_fractions: dict, n_trays: int, feed_tray: int, pressure_Pa: float, specs: list[dict],
                       reflux_ratio_min: float = 0.1, reflux_ratio_max: float = 20.0, boilup_ratio_min: float = 0.1,
                       boilup_ratio_max: float = 20.0, activity_model: str = "ideal",
                       binary_parameters: list | dict | None = None, excess_enthalpy: bool | None = None) -> dict:
    """Equation-oriented column optimisation (IDAES + ipopt, the whole column and the specs as ONE NLP) for a mixture
    of 2+ components named in English (chemicals names, e.g. "n-hexane", "methanol", "water"): minimum reboiler duty
    of a tray column (total condenser, partial reboiler, n_trays equilibrium trays, feed_tray counted from the top)
    meeting every spec, varying reflux and boil-up ratio. specs = [{"stream": "distillate" | "bottoms", "component":
    name, "kind": "mole_fraction" | "recovery", "min": value}, ...]; feed_mole_fractions = {name: z} summing to 1. The
    IDAES property package is generated from chemicals/thermo (Psat Wagner-McGarry, cp ideal gas Poling, liquid density
    DIPPR 105 or a cubic fit to VDI-PPDS, cp liquid fitted to thermo, ΔHvap = DIPPR 106 at Tsat) and checked against
    DIPPR in `properties.property_check`.
    activity_model="ideal" (default): Raoult + ideal gas; screened with Dortmund UNIFAC and REFUSED if an azeotrope or
    |ln γi/γj| > 0.10 is predicted.
    activity_model="NRTL": non-ideal liquid (modified Raoult with NRTL γ and NRTL excess enthalpy) + ideal-gas vapour.
    Binary parameters, ONE labelled source per pair, in this order: (a) the user's own, binary_parameters=[{"components":
    [i, j], "b_ij_K", "b_ji_K" (tau = a + b/T; or "dg_ij_J_mol"/"dg_ji_J_mol" or "dg_ij_cal_mol"/"dg_ji_cal_mol"),
    optional "a_ij", "a_ji", "alpha", "source": citation}]; (b) the full result dict of vle_binary_regression (pass it
    as is, alone or in the list: chain vle_binary_regression → eo_column_optimize when the user has VLE data);
    (c) the ChemSep NRTL databank shipped by thermo (the same table DWSIM's NRTL uses). A pair with none of the three
    is REFUSED: run vle_binary_regression on published data, never invent parameters. REFUSED too: specs on the far side
    of an azeotrope predicted with the chosen parameters, any binary azeotrope in a 3+ component mixture, and a vapour
    with |B·P/RT| > 0.10. UNIFAC is reported as a cross-check (γ∞, azeotrope) in properties.screening; the products'
    temperatures are re-checked as bubble points with thermo's NRTL (properties.thermo_bubble_check).
    Binary NRTL with two specs: a McCabe-Thiele check with the same NRTL runs first (`shortcut`: N_min, R_min) and
    REFUSES columns with too few trays, a reflux bound that cannot reach the specs, or a product closer than 0.03 K
    (dew − bubble) to an azeotrope; relay the Spanish reason. Excess enthalpy h^E of NRTL in the liquid enthalpy
    (Gibbs-Helmholtz, same parameters), excess_enthalpy: None (DEFAULT, conditional: the owner's policy) = ON only if
    EVERY pair's parameters come from a vle_binary_regression fit that included h^E data (excess_enthalpy_sets) and
    passed its verdict (representa_VLE_y_hE); in any other case (ChemSep bank, VLE-only regression, user values, a
    joint fit that failed the verdict, mixed sources) it is OFF and `warnings` carries a Spanish notice naming the
    pairs without calorimetric support and how to turn it on. True / False = the user's explicit choice, always
    honoured and labelled (True on VLE-only parameters stays labelled «no validado con calorimetría»: it can have the
    wrong sign, as in ethanol-water below ~363 K). properties.excess_enthalpy has `decision` (mode automatico|usuario,
    included, reason), `source` (where h^E comes from, with citations) and first_order_share_of_reboiler_duty (its
    share of the duty, or what it would be if off). The like-for-like DWSIM mode for the OFF case is «ExpData».
    A user pair may add "hE_source": a citation, kept as a label only (it does NOT turn the term on by default).
    Returns the optimum (RR, BR, duties, products), Lagrange multipliers = MW per unit of each spec, `warnings`
    (parameter extrapolation, Psat source of a regression) and `evidence`: ipopt optimal, specs met, properties
    verified, 8 neighbours ±1 % no better and feasible, multipliers = finite differences within 5 %, NRTL γ and h^E
    equal to thermo's → verified. 20-120 s. Use it for 'minimum energy/reflux for these purities/recoveries' and to
    price a spec; azeotropic separations (entrainers, pressure swing) go to the DWSIM column (optimize_design_multi)."""
    from packages.eo.column_optimize import eo_column_optimize as _eo
    return _eo(components, feed_flow_mol_s, feed_T_K, feed_P_Pa, feed_mole_fractions, n_trays, feed_tray, pressure_Pa,
               specs, (reflux_ratio_min, reflux_ratio_max), (boilup_ratio_min, boilup_ratio_max),
               activity_model=activity_model, binary_parameters=binary_parameters, excess_enthalpy=excess_enthalpy)


def optimize_design_surrogate(flowsheet_id: str, variables: list[dict], objective: dict, constraints: list[dict] | None = None,
                              maximize: bool = False, n_samples: int | None = None, n_holdout: int | None = None,
                              tolerance_pct: float = 0.5, objective_tolerance: float | None = None, seed: int = 0,
                              return_samples: bool = False) -> dict:
    """Surrogate-based optimisation of an existing DWSIM flowsheet (bloque M8c, OMLT + ipopt). variables = 1-6
    CONTINUOUS variables [{"kind": "unit"|"stream", "name", "param", "lower", "upper", "log_scale": bool}] with the
    user's bounds; objective and constraints exactly as in optimize_design (constraints may add "tolerance" = absolute
    accuracy wanted for that readout, e.g. 0.002 for a mole fraction). Samples the flowsheet in DWSIM over the bounds
    (Latin hypercube + corners; n_samples default 20 + 15·n, an independent held-out hypercube of n_holdout), trains one
    small tanh network per output (numpy, architecture by cross-validation), embeds them with OMLT
    (ReducedSpaceSmoothNNFormulation) and solves with ipopt from several starts, then RE-RUNS DWSIM at the proposed
    optimum. Returns surrogate_optimum, dwsim_at_optimum, holdout_error and error_at_optimum per output, the best DWSIM
    sample and `evidence`; status "verified" only if held-out and optimum errors are within tolerance (default 0.5 % of
    the median objective and of each limit), DWSIM meets the constraints within tolerance and no sample is better.
    Otherwise "not_verified": never report the surrogate value as the answer. Leaves the flowsheet solved at the
    proposed optimum. Use it when each DWSIM solve is slow or noisy, or to obtain a smooth algebraic model of the
    flowsheet; for 1-4 variables on a fast flowsheet optimize_design_multi is the direct alternative."""
    from packages.eo.surrogate import surrogate_optimize
    cons = constraints or []
    limits = {f"{c['name']}.{c['key']}": {k: c.get(k) for k in ("max", "min", "tolerance")} for c in cons}
    for k, lim in limits.items():
        if lim["max"] is None and lim["min"] is None:
            raise ValueError(f"la restricción {k} necesita max o min")

    def evaluate(x: list[float]) -> dict:
        for v, xi in zip(variables, x):
            _set(flowsheet_id, v["kind"], v["name"], v["param"], float(xi))
        r = _eval(flowsheet_id, None, None, objective, cons)
        return {"solved_ok": r["solved_ok"], "objective": r["objective"], "constraints": r.get("constraints") or {}}

    return surrogate_optimize(evaluate, variables, limits, maximize, n_samples, n_holdout, tolerance_pct,
                              objective_tolerance, seed, return_samples=return_samples)


OPTIMIZE_TOOLS = [optimize_design, optimize_design_multi, eo_column_optimize, optimize_design_surrogate]
