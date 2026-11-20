"""Signal diagnosis shared by the ODE and PDE solvers: is the tail of a computed history a sustained oscillation, a decaying
one, a steady value or still moving? Period and amplitude come from the located maxima (parabolic refinement)."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

SUSTAINED_TOL = 0.005      # |amplitude ratio per cycle − 1| below this → sustained
REGULAR_TOL = 0.02         # relative scatter of the peak-to-peak times below this → regular period
_TEXT = {
    "sustained_oscillation": "oscilación sostenida (amplitud y periodo constantes en la ventana analizada)",
    "decaying_oscillation": "oscilación amortiguada (la amplitud decrece ciclo a ciclo; tiende a un estado estacionario)",
    "growing_oscillation": "oscilación creciente (la amplitud aún aumenta ciclo a ciclo; no ha llegado a su ciclo límite)",
    "irregular_oscillation": "oscilación irregular (periodo o amplitud no repetibles en la ventana: transitorio, varias "
                             "frecuencias o caos)",
    "steady": "sin oscilación: valor constante en la ventana analizada",
    "not_settled": "sin oscilación, pero el valor todavía cambia al final de la ventana (no ha llegado al estacionario)",
}


def _refine(t, y, idx):
    """Parabolic refinement of the extrema at the uniform-grid indices idx → (times, values)."""
    idx = idx[(idx > 0) & (idx < len(y) - 1)]
    y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
    den = y0 - 2 * y1 + y2
    safe = np.where(den == 0, 1.0, den)
    off = np.where(den == 0, 0.0, 0.5 * (y0 - y2) / safe)
    return t[idx] + off * (t[1] - t[0]), y1 - 0.25 * (y0 - y2) * off


def diagnose(t: np.ndarray, y: np.ndarray, scale: float, noise_rel: float = 1e-6) -> dict:
    """Classify the history y(t) (uniform t). `scale` = typical magnitude of the variable over the whole run;
    `noise_rel` = relative numerical noise (integration tolerance) below which variations are ignored."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo
    scale = max(float(scale), abs(lo), abs(hi), 1e-300)
    floor = max(1e-6, 100 * noise_rel) * scale
    out = {"window": [float(t[0]), float(t[-1])], "min": lo, "max": hi, "mean": float(np.trapezoid(y, t) / (t[-1] - t[0]))}

    def finish(kind, **extra):
        out.update(kind=kind, description=_TEXT[kind], **extra)
        return out

    if span < floor:
        return finish("steady")
    tp = tv = np.array([])
    for prom in (0.25 * span, max(1e-3 * span, 10 * noise_rel * scale)):
        ip, _ = find_peaks(y, prominence=prom)
        iv, _ = find_peaks(-y, prominence=prom)
        if len(ip) >= 3 and len(iv) >= 2:
            (tp, yp), (tv, yv) = _refine(t, y, ip), _refine(t, y, iv)
            break
    if len(tp) < 3 or len(tv) < 2:
        n_tail = max(2, len(y) // 10)
        moving = abs(y[-1] - y[-n_tail]) > max(floor, 1e-4 * span)
        return finish("not_settled" if moving else "steady")
    periods = np.diff(tp)
    # amplitude of each cycle: peak to the FOLLOWING trough (a consistent pairing, so the ratio is exact for a damped wave)
    nxt = np.searchsorted(tv, tp)
    amps = np.maximum(np.array([0.5 * (yp[k] - yv[j]) for k, j in enumerate(nxt) if j < len(tv)]), 1e-300)
    if len(amps) < 2:
        return finish("not_settled")
    ratio = float((amps[-1] / amps[0]) ** (1.0 / (len(amps) - 1)))
    scatter = float(np.std(periods) / np.mean(periods))
    extra = {"period": float(np.mean(periods)), "period_scatter_rel": scatter, "amplitude": float(amps[-1]),
             "amplitude_definition": "mitad de la distancia del último pico al valle siguiente dentro de la ventana",
             "amplitude_ratio_per_cycle": ratio, "n_cycles": int(len(periods)),
             "last_peak": {"time": float(tp[-1]), "value": float(yp[-1])},
             "last_trough": {"time": float(tv[-1]), "value": float(yv[-1])}}
    if amps[-1] < floor:
        return finish("decaying_oscillation", **extra)
    if scatter > REGULAR_TOL or (len(amps) > 3 and np.std(amps) / np.mean(amps) > 0.05
                                 and not _monotone(amps)):
        return finish("irregular_oscillation", **extra)
    if abs(ratio - 1.0) < SUSTAINED_TOL:
        return finish("sustained_oscillation", **extra)
    return finish("decaying_oscillation" if ratio < 1 else "growing_oscillation", **extra)


def _monotone(a: np.ndarray) -> bool:
    d = np.diff(a)
    return bool(np.all(d <= 0) or np.all(d >= 0))
