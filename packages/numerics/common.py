"""Small helpers shared by the equation solvers."""
from __future__ import annotations

import math
import time

import numpy as np

MAX_SECONDS = 90.0          # wall-clock budget of one tool call (all runs, including the evidence reruns)
EVIDENCE_TOL = 1e-3         # refinement difference (relative to the scale of each quantity) accepted as verified


class Timeout(Exception):
    """Raised inside a right-hand side when the wall-clock budget is exhausted."""


class Deadline:
    def __init__(self, seconds: float = MAX_SECONDS):
        self.end = time.monotonic() + seconds
        self.calls = 0

    def tick(self) -> None:
        self.calls += 1
        if self.calls % 64 == 0 and time.monotonic() > self.end:
            raise Timeout()


TIMEOUT_TEXT = (f"el cálculo excedió el tiempo límite de {MAX_SECONDS:.0f} s: el sistema es demasiado rígido, explota o el "
                "intervalo es muy largo para las tolerancias pedidas; reducir el intervalo, relajar tolerancias o revisar "
                "las ecuaciones")


def number(value, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito")
    if positive and value <= 0:
        raise ValueError(f"{name} debe ser > 0")
    return float(value)


def number_list(values, name: str) -> np.ndarray:
    if values is None:
        return np.array([])
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        values = [values]
    try:
        arr = np.array([float(v) for v in values], float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} debe ser una lista de números") from None
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} debe contener números finitos")
    return arr


def full(value, like: np.ndarray) -> np.ndarray:
    """Broadcast an expression result (scalar or array) to the shape of `like` as float."""
    return np.broadcast_to(np.asarray(value, float), like.shape)


def rel_diff(a, b, scale) -> float:
    """max |a − b| / scale over arrays (scale is a positive scalar or array)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(scale, 1e-300)))
