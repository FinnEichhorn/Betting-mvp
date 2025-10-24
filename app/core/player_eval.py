# app/core/player_eval.py
from __future__ import annotations
import numpy as np
import pandas as pd
from app.core.odds import american_to_decimal
from app.core.ev import evaluate


def _ensure_decimal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "decimal_odds" in df.columns:
        return df
    if "price_decimal" in df.columns:
        df["decimal_odds"] = df["price_decimal"].astype(float)
        return df
    # Fallbacks for american odds
    if "price_american" in df.columns:
        df["decimal_odds"] = df["price_american"].astype(float).map(american_to_decimal)
    elif "american_odds" in df.columns:
        df["decimal_odds"] = df["american_odds"].astype(float).map(american_to_decimal)
    elif "price" in df.columns:
        df["decimal_odds"] = df["price"].astype(float).map(american_to_decimal)
    return df


def add_fair_probs_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the bookmaker vig for each player prop market.
    Normalizes implied probabilities per bookmaker per player/market/point.
    """
    df = _ensure_decimal(df)
    out = df.copy()
    out["imp_raw"] = 1.0 / out["decimal_odds"]
    group_cols = ["event_id", "player", "market", "point", "bookmaker"]
    denom = out.groupby(group_cols)["imp_raw"].transform("sum")
    out["p_fair"] = np.where(denom > 0, out["imp_raw"] / denom, np.nan)
    return out


def compute_player_edges(
    df: pd.DataFrame,
    baseline_model,
    m: float,
    min_p: float = 0.01,
    max_p: float = 0.99,
    markets: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute expected value and Kelly for each player prop selection.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if markets:
        df = df[df["market"].isin(markets)].copy()
        if df.empty:
            return pd.DataFrame()

    df = add_fair_probs_players(df)

    # 1. Predict model probabilities for each row
    def _predict_row(r) -> float:
        return baseline_model.predict(
            market_prob_vigfree = r.get("p_fair", None),
            selection           = r.get("outcome", ""),
            market_type         = r.get("market", ""),
            point               = r.get("point", None),
            player              = r.get("player", None),  # optional argument
        )

    df["model_prob"] = df.apply(_predict_row, axis=1).astype(float)

    # 2. Apply your slider multiplier
    df["model_prob_adj"] = np.clip(df["model_prob"] * m, min_p, max_p)

    # 3. Compute EV & Kelly using your core evaluate()
    def _eval_row(r):
        be = evaluate(decimal_odds=float(r["decimal_odds"]), model_prob=float(r["model_prob_adj"]))
        return pd.Series({
            "ev_per_dollar": round(be.ev_per_dollar, 4),
            "kelly": round(be.kelly, 4),
            "kelly_quarter": round(be.kelly_quarter, 4),
        })

    df_eval = df.apply(_eval_row, axis=1)
    out = pd.concat([df, df_eval], axis=1)

    # 4. Optional: difference between model and vig-free market probs
    if "p_fair" in out.columns:
        out["edge_prob"] = (out["model_prob_adj"] - out["p_fair"]).round(4)

    # 5. Sort by EV descending for display
    sort_cols = [c for c in ["ev_per_dollar", "player", "market", "point"] if c in out.columns]
    return out.sort_values(sort_cols, ascending=[False, True, True, True]).reset_index(drop=True)
