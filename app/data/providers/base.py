from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Dict, Any
import abc
import datetime as dt

# One row per market selection (e.g., Moneyline home/away, Spread home/away, Total over/under)
@dataclass(frozen=True)
class NormalizedOdd:
    provider: str                    # e.g., "theoddsapi"
    bookmaker: str                   # e.g., "Pinnacle", "DraftKings"
    sport_key: str                   # provider sport key
    league: Optional[str]            # optional, derived
    event_id: str
    commence_time_utc: dt.datetime
    home_team: str
    away_team: str
    market: Literal["h2h", "spreads", "totals"]
    outcome: str                     # "home"/"away"/"draw" OR "home"/"away" for spreads OR "over"/"under" for totals
    point: Optional[float]           # spread/total line when applicable
    price_american: int
    price_decimal: float
    last_update_utc: Optional[dt.datetime]

class OddsProvider(abc.ABC):
    name: str

    @abc.abstractmethod
    def fetch(
        self,
        sport_key: str,
        region: str = "us",
        markets: Optional[list[str]] = None,
        bookmakers: Optional[list[str]] = None,
    ) -> Iterable[NormalizedOdd]:
        ...
