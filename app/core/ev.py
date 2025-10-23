from __future__ import annotations
from dataclasses import dataclass

def ev_per_unit(decimal_odds: float, model_prob: float) -> float:
    # Expected profit for $1 stake
    return model_prob * (decimal_odds - 1) - (1 - model_prob)

def kelly_fraction(decimal_odds: float, model_prob: float) -> float:
    b = decimal_odds - 1
    q = 1 - model_prob
    k = (b * model_prob - q) / b
    return max(0.0, k)  # don’t bet if negative EV

@dataclass
class BetEval:
    model_prob: float
    ev_per_dollar: float   # ✅ valid Python identifier
    kelly: float
    kelly_quarter: float

def evaluate(decimal_odds: float, model_prob: float) -> BetEval:
    ev = ev_per_unit(decimal_odds, model_prob)
    k = kelly_fraction(decimal_odds, model_prob)
    return BetEval(
        model_prob=model_prob,
        ev_per_dollar=ev,     # ✅ rename here too
        kelly=k,
        kelly_quarter=0.25 * k
    )