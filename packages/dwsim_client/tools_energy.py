"""Agent tool for energy integration (bloque M5): pinch analysis. No flowsheet needed: the stream table comes from
the user or from the heaters/coolers of a solved flowsheet (inlet/outlet T and duty). Engine and evidence live in
`packages.energy.pinch`."""
from __future__ import annotations

from packages.energy import pinch as P


def pinch_analysis(streams: list[dict], dT_min_K: float, dT_min_sweep_K: list[float] | None = None,
                   current_hot_utility_W: float | None = None, current_cold_utility_W: float | None = None) -> dict:
    """Pinch analysis of a process stream table by the Problem Table Algorithm: minimum hot and cold utility (MER
    targets), pinch temperature, maximum heat recovery, problem table, heat cascade, grand composite curve (where to
    place utilities: low- vs high-pressure steam, cooling water vs refrigeration), hot and cold composite curves in
    real temperatures, minimum number of exchangers, and which streams cross the pinch. streams = [{'name',
    'T_supply_K', 'T_target_K', 'CP_W_K' (m·cp, assumed constant) or 'heat_load_W', optional 'dT_contribution_K'
    (default dT_min_K/2; use larger for gases/low h, smaller for boiling/condensing)}]; hot streams are those that
    cool (T_supply > T_target). A phase change at constant T goes as a 1 K span with heat_load_W = latent duty; a
    stream with strongly varying cp should be split into segments. dT_min_K is a design decision (typical 10-20 K
    liquids, 20-40 K gases, 3-5 K refrigeration): ask or state the one used, and give dT_min_sweep_K to show the
    trade-off. current_hot_utility_W / current_cold_utility_W (the plant's present consumption) give the saving
    potential = heat crossing the pinch. evidence: the shifted composite curves built independently must touch at the
    pinch without crossing, a smaller cold utility must make them cross, and the overall energy balance must close."""
    return P.pinch(streams, dT_min_K, dT_min_sweep_K, current_hot_utility_W, current_cold_utility_W)


ENERGY_TOOLS = [pinch_analysis]
