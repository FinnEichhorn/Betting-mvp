from __future__ import annotations
from typing import Iterable, Optional, List
import pandas as pd
from dataclasses import asdict
from .providers.base import OddsProvider, NormalizedOdd

def to_dataframe(rows: Iterable[NormalizedOdd]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in rows])
    if not df.empty:
        # Consistent dtypes
        df["commence_time_utc"] = pd.to_datetime(df["commence_time_utc"], utc=True)
        if "last_update_utc" in df:
            df["last_update_utc"] = pd.to_datetime(df["last_update_utc"], utc=True)
    return df

def merge_providers(
    providers: List[OddsProvider],
    sport_key: str,
    region: str = "us",
    markets: Optional[List[str]] = None,
    bookmakers: Optional[List[str]] = None,
) -> pd.DataFrame:
    frames = []
    for p in providers:
        rows = p.fetch(sport_key=sport_key, region=region, markets=markets, bookmakers=bookmakers)
        frames.append(to_dataframe(rows))
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame()
    # stable column order
    cols = [
        "provider","bookmaker","sport_key","league","event_id","commence_time_utc",
        "home_team","away_team","market","outcome","point","price_american",
        "price_decimal","last_update_utc"
    ]
    return df.reindex(columns=cols)

def save_snapshot_parquet(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        return
    df.to_parquet(path, index=False)
