from __future__ import annotations


import numpy as np
import pandas as pd



# --- Odds conversions ---


def american_to_decimal(odds: int) -> float:
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be > 1")
    return int(round((decimal_odds - 1) * 100)) if decimal_odds >= 2 else int(round(-100 / (decimal_odds - 1)))


def american_to_implied_prob(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


# --- Vig removal (2-way market only) ---


def remove_vig_two_way(p1_raw: float, p2_raw: float) -> tuple[float, float]:
    """
    p*_raw are implied probabilities (including vig) for two opposing selections.
    Returns vig-free (normalized) probabilities.
    """
    s = p1_raw + p2_raw
    if s <= 0:
    # Degenerate; guard against divide-by-zero
        return 0.5, 0.5
    return p1_raw / s, p2_raw / s


# Utility


def clip_prob(p: float) -> float:
    return float(np.clip(p, 0.001, 0.999))
