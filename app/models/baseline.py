from __future__ import annotations


from app.core.odds import clip_prob


# A toy model: small home edge; totals/others default to market-neutral 50%.




import math

def american_to_implied_prob(odds: float) -> float:
    """
    Convert American odds to implied win probability (includes vig).
    """
    try:
        odds = float(odds)
    except (ValueError, TypeError):
        return 0.5  # fallback

    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def baseline_probability(selection: str, home_team: str, away_team: str, market_type: str) -> float:
    selection = str(selection).upper().strip()
    market_type = str(market_type).upper().strip()

    # Mild, deterministic biases so you can see EV/Kelly vary
    if market_type in ("MONEYLINE", "MATCH", "WINNER"):
        return 0.54 if selection == "HOME" else 0.46
    if market_type in ("TOTALS", "OVER_UNDER", "O/U"):
        return 0.52 if selection == "UNDER" else 0.48

    return 0.5