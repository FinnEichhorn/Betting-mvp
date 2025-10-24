from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

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

def evaluate_df(df: pd.DataFrame, model_col: str = "p_model", dec_col: str = "price_decimal") -> pd.DataFrame:
    """
    Vectorized evaluation of EV and Kelly for a DataFrame.
    Adds: ev_per_dollar, kelly, kelly_quarter
    """
    out = df.copy()
    out["ev_per_dollar"] = df.apply(
        lambda r: ev_per_unit(r[dec_col], r[model_col]), axis=1
    )
    out["kelly"] = df.apply(
        lambda r: kelly_fraction(r[dec_col], r[model_col]), axis=1
    )
    out["kelly_quarter"] = 0.25 * out["kelly"]
    return out