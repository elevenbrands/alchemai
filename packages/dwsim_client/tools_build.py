"""Whole-flowsheet tools: build a complete DWSIM flowsheet from one specification and meet design specifications.

Large processes (several recycles, a dozen units) cost one tool call per stream, unit and connection with the primitive
tools; `build_flowsheet` does the same work, with the same primitives and the same verifier, in one call.
`flowsheet_goal_seek` is the "design spec / adjust" of a sequential-modular simulator: it varies 1-3 specifications until
the same number of readouts (or ratios of readouts) hit their targets, re-solving the flowsheet at every trial.
"""
from __future__ import annotations

from typing import Any

from packages.dwsim_client import tools as T
from packages.dwsim_client import tools_sweep as TS
from packages.dwsim_client.dof import dof_check

_REACTIONS = {"conversion": T.define_conversion_reaction, "equilibrium": T.define_equilibrium_reaction,
              "kinetic": T.define_kinetic_reaction}
_PORT_KEYS = ("feeds", "products", "energy_feeds", "energy_products")


def _fail(fs_id: str | None, stage: str, item: Any, ex: Exception) -> dict:
    if fs_id:
        T.close_flowsheet(fs_id)
    return {"ok": False, "built": False, "failed_at": stage, "item": item, "error": f"{type(ex).__name__}: {ex}",
            "note": "No se creó el diagrama: corrige ese elemento de la especificación y vuelve a llamar a build_flowsheet."}


def build_flowsheet(name: str, compounds: list[str], property_package: str, streams: list[dict] | None = None,
                    units: list[dict] | None = None, reactions: list[dict] | None = None, solve: bool = True) -> dict:
    """Build a WHOLE flowsheet in one call and solve it: use this instead of one call per stream/unit/connection whenever
    the process has more than ~4 units. streams: [{name, temperature_K, pressure_Pa, vapor_fraction, molar_flow_mol_s,
    mass_flow_kg_s, mole_fractions, compound_molar_flows_mol_s}] — list only specified feeds; every other stream named in
    a unit is created empty. units: [{kind, name, params, feeds, products, energy_feeds, energy_products}] with kinds,
    params and port order exactly as in add_unit/connect (reactors: products [vapor, liquid]; columns: [distillate,
    bottoms] + both duties; recycle: feeds [tear], products [recycled]); energy streams are created from their names.
    reactions: [{type: "conversion"|"equilibrium"|"kinetic", name, stoichiometry, base_compound, + the arguments of the
    matching define_*_reaction, reaction_set}]. If an element is rejected nothing is kept and the error names it: fix the
    spec and call again. Returns flowsheet_id, dof {ok, problems} and solve (same report as solve(), with every stream):
    only ok=true is trustworthy. Afterwards refine with set_unit_param/set_stream_spec/sweep/flowsheet_goal_seek."""
    streams, units, reactions = list(streams or []), list(units or []), list(reactions or [])
    try:
        fs_id = T.new_flowsheet(name, compounds, property_package)["flowsheet_id"]
    except Exception as ex:  # noqa: BLE001
        return _fail(None, "new_flowsheet", {"compounds": compounds, "property_package": property_package}, ex)
    for r in reactions:
        try:
            kw = dict(r)
            _REACTIONS[str(kw.pop("type")).lower()](fs_id, **kw)
        except KeyError as ex:
            return _fail(fs_id, "reactions", r, ValueError(f"falta o no se reconoce {ex}; type ∈ {sorted(_REACTIONS)}"))
        except Exception as ex:  # noqa: BLE001
            return _fail(fs_id, "reactions", r, ex)
    listed: set[str] = set()
    for st in streams:
        try:
            kw = dict(st)
            T.add_material_stream(fs_id, kw.pop("name"), **kw)
            listed.add(st["name"])
        except Exception as ex:  # noqa: BLE001
            return _fail(fs_id, "streams", st, ex)
    energy: set[str] = set()
    for u in units:
        try:
            unknown = set(u) - {"kind", "name", "params", *_PORT_KEYS}
            if unknown:
                raise ValueError(f"claves no reconocidas {sorted(unknown)}; usa kind, name, params, {', '.join(_PORT_KEYS)}")
            for key in ("feeds", "products"):
                for sname in u.get(key) or []:
                    if sname not in listed:
                        T.add_material_stream(fs_id, sname)
                        listed.add(sname)
            for key in ("energy_feeds", "energy_products"):
                for sname in u.get(key) or []:
                    if sname not in energy:
                        T.add_energy_stream(fs_id, sname)
                        energy.add(sname)
            T.add_unit(fs_id, u["kind"], u["name"], u.get("params"))
        except Exception as ex:  # noqa: BLE001
            return _fail(fs_id, "units", u, ex)
    for u in units:                                   # connect after every unit exists (recycles close loops)
        try:
            T.connect(fs_id, u["name"], u.get("feeds"), u.get("products"), u.get("energy_feeds"), u.get("energy_products"))
        except Exception as ex:  # noqa: BLE001
            return _fail(fs_id, "connect", u, ex)
    out: dict[str, Any] = {"ok": True, "built": True, "flowsheet_id": fs_id,
                           "created": {"reactions": len(reactions), "material_streams": len(listed),
                                       "energy_streams": len(energy), "units": len(units)}}
    try:
        dof = dof_check(fs_id)
        out["dof"] = {"ok": dof.get("ok"), "problems": dof.get("problems", [])}
    except Exception as ex:  # noqa: BLE001
        out["dof"] = {"ok": None, "problems": [f"dof_check falló: {ex}"]}
    if solve:
        try:
            out["solve"] = T.solve(fs_id)
        except Exception as ex:  # noqa: BLE001
            out["solve"] = {"solved": False, "ok": False, "solver_errors": [f"{type(ex).__name__}: {ex}"]}
        out["ok"] = bool(out["solve"].get("ok"))
    return out


def _read(fs_id: str, spec: dict) -> float:
    return TS._readout(fs_id, spec["kind"], spec["name"], spec["key"])


def _apply(fs_id: str, var: dict, x: float) -> None:
    if var["kind"] == "unit":
        TS.set_unit_param(fs_id, var["name"], var["param"], float(x))
    elif str(var["param"]).startswith("mole_fraction."):
        TS.set_stream_spec(fs_id, var["name"], mole_fractions=TS._with_fraction(fs_id, var["name"], var["param"].split(".", 1)[1], float(x)))
    else:
        TS.set_stream_spec(fs_id, var["name"], **{var["param"]: float(x)})


def flowsheet_goal_seek(flowsheet_id: str, variables: list[dict], targets: list[dict], tolerance: float = 1e-4,
                        max_solves: int = 60) -> dict:
    """Design specification ("adjust") on a built flowsheet, in one call: vary N specs until N readouts hit their targets
    (N = 1 to 3 together), re-solving every trial. variables: [{"kind": "unit"|"stream", "name", "param", "lower",
    "upper"}] (param as in sweep: split_ratio_0, outlet_T_K, molar_flow_mol_s, reflux_ratio, ...). targets: [{"kind",
    "name", "key", "value"}] with key as in sweep ("component_flow.<C>" mol/s, "mole_fractions.<C>", "T_K",
    "molar_flow_mol_s", "energy_streams_kW.<E>"); add "divide_by": {"kind", "name", "key"} for a ratio (e.g. H2/toluene at
    a reactor inlet = 7 by varying the makeup flow; purge fraction for an inert/reactant ratio). tolerance is relative.
    Returns converged, the values found (at_bound flags), readouts, residuals, solves and the final solve report; the
    flowsheet stays solved there. converged=false: the targets are not met — do not report those values."""
    import numpy as np
    from scipy.optimize import brentq, least_squares

    n = len(variables)
    if not 1 <= n <= 3 or len(targets) != n:
        raise ValueError("se necesitan entre 1 y 3 variables y el mismo número de objetivos")
    for v in variables:
        if not float(v["lower"]) < float(v["upper"]):
            raise ValueError(f"límites inválidos para {v.get('name')}.{v.get('param')}: lower debe ser menor que upper")
    for t in targets:
        if "value" not in t:
            raise ValueError("cada objetivo necesita 'value'")
    lo = np.array([float(v["lower"]) for v in variables])
    hi = np.array([float(v["upper"]) for v in variables])
    scale = np.array([abs(float(t["value"])) or 1.0 for t in targets])
    count = {"solves": 0, "failed": 0}
    last: dict[str, Any] = {}

    def readouts() -> list[float]:
        vals = []
        for t in targets:
            y = _read(flowsheet_id, t)
            if t.get("divide_by"):
                y = y / _read(flowsheet_id, t["divide_by"])
            vals.append(float(y))
        return vals

    def residual(x) -> np.ndarray:
        if count["solves"] >= max_solves:
            raise StopIteration
        for v, xi in zip(variables, np.atleast_1d(x)):
            _apply(flowsheet_id, v, float(xi))
        sol = T.solve(flowsheet_id)
        count["solves"] += 1
        last.update(x=[float(xi) for xi in np.atleast_1d(x)], sol=sol)
        try:
            ys = readouts()
        except Exception:  # noqa: BLE001
            ys = None
        if not sol.get("solved") or ys is None or not all(np.isfinite(ys)):
            count["failed"] += 1
            last["readouts"] = None
            return np.full(n, 1e3)
        last["readouts"] = ys
        return (np.array(ys) - np.array([float(t["value"]) for t in targets])) / scale

    method, x_best = "", None
    try:
        if n == 1:
            fa, fb = residual(lo)[0], residual(hi)[0]
            if abs(fa) < 1e3 and abs(fb) < 1e3 and fa * fb < 0:
                method = "brentq (cambio de signo entre los límites)"
                x_best = np.array([brentq(lambda z: residual([z])[0], lo[0], hi[0], xtol=1e-10 * (hi[0] - lo[0]), rtol=1e-10,
                                          maxiter=max_solves)])
            elif abs(fa) <= tolerance:
                x_best = lo.copy()
            elif abs(fb) <= tolerance:
                x_best = hi.copy()
        if x_best is None:
            method = "mínimos cuadrados con límites (trf, jacobiano por diferencias)"
            x0 = np.array([float(v.get("start", (a + b) / 2)) for v, a, b in zip(variables, lo, hi)])
            res = least_squares(residual, x0, bounds=(lo, hi), x_scale=hi - lo, diff_step=1e-3, xtol=1e-12, ftol=1e-12,
                                gtol=1e-12, max_nfev=max_solves)
            x_best = res.x
    except StopIteration:
        method += " — detenido por max_solves"
        x_best = np.array(last.get("x", (lo + hi) / 2))
    count_before = count["solves"]
    count["solves"] = -10**9                              # the closing solve always runs
    r = residual(x_best)
    count["solves"] = count_before + 1
    sol = last["sol"]
    converged = bool(sol.get("ok")) and last["readouts"] is not None and bool(np.all(np.abs(r) <= tolerance))
    out = {"converged": converged, "method": method,
           "variables": [{"kind": v["kind"], "name": v["name"], "param": v["param"], "value": float(x),
                          "at_bound": bool(abs(x - a) < 1e-9 * (b - a) or abs(x - b) < 1e-9 * (b - a))}
                         for v, x, a, b in zip(variables, x_best, lo, hi)],
           "targets": [{"key": t["key"], "name": t["name"], "divide_by": t.get("divide_by"), "target": float(t["value"]),
                        "readout": (last["readouts"][i] if last["readouts"] else None),
                        "relative_residual": float(r[i]) if last["readouts"] else None} for i, t in enumerate(targets)],
           "solves": count["solves"], "failed_solves": count["failed"], "solve": sol}
    if not converged:
        out["note"] = ("No se cumplieron los objetivos dentro de los límites (o el diagrama no resolvió ahí): no reportar "
                       "estos valores como resultado; revisa los límites o las especificaciones.")
    return out


BUILD_TOOLS = [build_flowsheet, flowsheet_goal_seek]
