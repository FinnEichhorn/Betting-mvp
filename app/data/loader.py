from __future__ import annotations


import pandas as pd
from pathlib import Path


from app.core.odds import american_to_decimal


REQUIRED_COLS = [
    "event_id",
    "league",
    "start_time",
    "home_team",
    "away_team",
    "market_type",
    "selection",
    "book",
    "american_odds",
    ]




def load_odds_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "decimal_odds" not in df.columns:
        df["decimal_odds"] = df["american_odds"].astype(int).map(american_to_decimal)
    return df
