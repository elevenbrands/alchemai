"""Save and rebuild a flowsheet session (MIT). The Mac app stores a case as JSON and rebuilds it in a fresh DWSIM
session when the case is reopened.

The snapshot is the CURRENT state of the session, not the history of calls: units with their present parameters
(after any set_unit_param, optimizer or column rebuild), streams with their present specs, connections, the property
package and the reaction definitions. Rebuilding replays that state with the same tools the agent uses, then solves.
"""
from __future__ import annotations

from typing import Any

from packages.dwsim_client import tools as T

SNAPSHOT_VERSION = 1


def snapshot_flowsheet(flowsheet_id: str) -> dict[str, Any]:
    s = T._s(flowsheet_id)
    meta = s.get("meta") or {}
    if "property_package" not in meta:
        raise ValueError("this flowsheet was created before snapshots existed (no property package recorded)")
    units = []
    for name, kind in s["kinds"].items():
        params = dict(s["params"].get(name) or {})
        if name in s["column_feeds"]:
            params["feed_stage"] = s["column_feeds"][name]
        conn = dict(s["conn"].get(name) or {})
        units.append({"name": name, "kind": kind, "params": params, "connections": {k: list(conn.get(k) or []) for k in
                      ("feeds", "products", "energy_feeds", "energy_products")}})
    return {"version": SNAPSHOT_VERSION, "name": meta.get("name", "caso"), "property_package": meta["property_package"],
            "compounds": list(s["compounds"]), "reaction_defs": list(s.get("reaction_defs") or []),
            "streams": [{"name": n, "spec": dict(s["stream_specs"].get(n) or {})} for n in s["streams"]],
            "energy": list(s["energy"]), "units": units}


def restore_flowsheet(snapshot: dict[str, Any], solve: bool = True) -> dict[str, Any]:
    if int(snapshot.get("version", 0)) != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported snapshot version {snapshot.get('version')!r}")
    fid = T.new_flowsheet(snapshot["name"], snapshot["compounds"], snapshot["property_package"])["flowsheet_id"]
    try:
        for r in snapshot.get("reaction_defs") or []:
            getattr(T, r["tool"])(fid, **r["args"])
        for st in snapshot["streams"]:
            T.add_material_stream(fid, st["name"], **(st.get("spec") or {}))
        for e in snapshot.get("energy") or []:
            T.add_energy_stream(fid, e)
        for u in snapshot["units"]:
            T.add_unit(fid, u["kind"], u["name"], u.get("params") or {})
        for u in snapshot["units"]:
            c = u.get("connections") or {}
            if any(c.get(k) for k in ("feeds", "products", "energy_feeds", "energy_products")):
                T.connect(fid, u["name"], c.get("feeds") or [], c.get("products") or [], c.get("energy_feeds") or [], c.get("energy_products") or [])
    except Exception:
        T.close_flowsheet(fid)
        raise
    out: dict[str, Any] = {"flowsheet_id": fid}
    if solve:
        out["solve"] = T.solve(fid)
    return out
