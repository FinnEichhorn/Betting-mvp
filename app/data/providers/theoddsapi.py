from __future__ import annotations
import os
import math
import datetime as dt
from typing import Iterable, Optional, List
import httpx

from .base import OddsProvider, NormalizedOdd

API_BASE = "https://api.the-odds-api.com/v4"  # free/premium tiers available

def dec_to_amer(decimal: float) -> int:
    # safe convert (The Odds API returns decimal by default for many markets)
    if decimal <= 1.0:
        return 0
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100))
    # favorites
    return int(round(-100 / (decimal - 1.0)))

class TheOddsApiProvider(OddsProvider):
    name = "theoddsapi"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Set ODDS_API_KEY in your environment to use The Odds API adapter."
            )
        self.timeout = timeout

    def _get(self, path: str, params: dict) -> list[dict]:
        params = dict(params)
        params["apiKey"] = self.api_key
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(f"{API_BASE}{path}", params=params)
            r.raise_for_status()
            return r.json()

    def fetch(
        self,
        sport_key: str,
        region: str = "us",
        markets: Optional[List[str]] = None,
        bookmakers: Optional[List[str]] = None,
    ) -> Iterable[NormalizedOdd]:
        """
        sport_key example: 'soccer_epl', 'basketball_nba', 'icehockey_nhl', 'americanfootball_nfl'
        region: 'us', 'uk', 'eu', 'au'
        markets: e.g. ['h2h','spreads','totals']
        bookmakers: e.g. ['pinnacle','draftkings']
        """
        markets = markets or ["h2h"]  # minimal default
        params = {
            "regions": region,
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        events = self._get(f"/sports/{sport_key}/odds", params)

        out: list[NormalizedOdd] = []
        for ev in events:
            event_id = ev["id"]
            commence = dt.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).astimezone(dt.timezone.utc)
            home = ev["home_team"]
            away = ev["away_team"]
            for book in ev.get("bookmakers", []):
                bk_name = book.get("title") or book.get("key")
                last = book.get("last_update")
                last_dt = None
                if last:
                    try:
                        last_dt = dt.datetime.fromisoformat(last.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
                    except Exception:
                        last_dt = None
                for market in book.get("markets", []):
                    mkey = market.get("key")  # 'h2h', 'spreads', 'totals'
                    # Outcomes format differs per market
                    for outcome in market.get("outcomes", []):
                        outcome_name = outcome.get("name", "").lower()
                        price_dec = float(outcome.get("price"))
                        price_amer = outcome.get("price_american")
                        if price_amer is None:
                            price_amer = dec_to_amer(price_dec)

                        point = outcome.get("point")
                        norm_outcome = outcome_name
                        if mkey == "h2h":
                            # normalize "draw" "home" "away"
                            if outcome_name not in ("home", "away", "draw"):
                                # The Odds API often gives actual team names for h2h
                                if outcome_name == home.lower() or outcome.get("name") == home:
                                    norm_outcome = "home"
                                elif outcome_name == away.lower() or outcome.get("name") == away:
                                    norm_outcome = "away"
                                else:
                                    # fallback: keep as-is
                                    norm_outcome = outcome.get("name").lower()
                        elif mkey == "spreads":
                            # usually name is team name; normalize to home/away by matching team
                            nm = outcome.get("name")
                            if nm == home:
                                norm_outcome = "home"
                            elif nm == away:
                                norm_outcome = "away"
                            # 'point' is the spread (negative for favorite)
                        elif mkey == "totals":
                            # name is 'Over'/'Under'
                            norm_outcome = "over" if "over" in outcome_name else "under"

                        out.append(
                            NormalizedOdd(
                                provider=self.name,
                                bookmaker=bk_name,
                                sport_key=sport_key,
                                league=None,
                                event_id=event_id,
                                commence_time_utc=commence,
                                home_team=home,
                                away_team=away,
                                market=mkey,
                                outcome=norm_outcome,
                                point=float(point) if point is not None else None,
                                price_american=int(price_amer),
                                price_decimal=price_dec,
                                last_update_utc=last_dt,
                            )
                        )
        return out
