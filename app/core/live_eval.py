import numpy as np
import pandas as pd
from app.core.odds import american_to_implied_prob, american_to_decimal, remove_vig_two_way

def add_fair_probs(df_live: pd.DataFrame) -> pd.DataFrame:
    """
    For two-way markets (h2h without draw, spreads, totals),
    compute fair (de-vigged) probabilities per bookmaker/event/market.
    """
    df = df_live.copy()

    # raw implied prob from american odds
    df["imp_raw"] = df["price_american"].astype(int).map(american_to_implied_prob)

    # tag rows that are two-way (skip draws for now)
    is_two_way = (df["market"].isin(["h2h", "spreads", "totals"])) & (~df["outcome"].eq("draw"))
    df_tw = df[is_two_way].copy()

    # For two-way markets, we need the pair side-by-side to de-vig
    def _devig_grp(g: pd.DataFrame) -> pd.DataFrame:
        # Expect exactly two outcomes (home/away) or (over/under)
        if g.shape[0] != 2:
            g["p_fair"] = np.nan
            return g

        p1, p2 = g["imp_raw"].iloc[0], g["imp_raw"].iloc[1]
        fair1, fair2 = remove_vig_two_way(p1, p2)  # you already have this
        g = g.copy()
        g.loc[g.index[0], "p_fair"] = fair1
        g.loc[g.index[1], "p_fair"] = fair2
        return g

    df_tw = (
        df_tw.sort_values(["event_id","bookmaker","market","outcome"])
             .groupby(["event_id","bookmaker","market"], group_keys=False)
             .apply(_devig_grp)
    )

    # merge back
    df = df.merge(
        df_tw[["p_fair"]],
        left_index=True, right_index=True, how="left"
    )

    # helpful price columns
    df["price_decimal"] = df["price_american"].astype(int).map(american_to_decimal)
    df["b"] = df["price_decimal"] - 1.0  # profit multiple for Kelly
    return df
