from __future__ import annotations


def within(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


def relative_error(value: float, target: float) -> float:
    denom = max(abs(target), 1e-12)
    return abs(value - target) / denom

