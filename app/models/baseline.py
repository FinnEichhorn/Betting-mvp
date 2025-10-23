from __future__ import annotations


from app.core.odds import clip_prob


# A toy model: small home edge; totals/others default to market-neutral 50%.




def baseline_probability(selection: str, home_team: str, away_team: str, market_type: str) -> float:
    sel = selection.upper()
    mkt = market_type.upper()


# Home/away ML: give home ~2% bump
    if mkt == "ML":
        if sel == "HOME":
            return clip_prob(0.52)
        elif sel == "AWAY":
            return clip_prob(0.48)


# For spreads/totals you'd plug in your model; here we default to coin flip
    return 0.50