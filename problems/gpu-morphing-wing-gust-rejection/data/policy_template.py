"""Minimal policy interface for GPU Morphing Wing Gust Rejection."""

from __future__ import annotations

import numpy as np


class Policy:
    def act(self, obs: dict) -> list[float]:
        _ = obs
        return np.zeros(7, dtype=float).tolist()
