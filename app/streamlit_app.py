from __future__ import annotations

# ── sys.path bootstrap so "app/..." imports work both locally and in deployment ──
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── standard / third-party ──
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import os
import requests

# ── project imports (adjust provider path if yours differs) ──
from app.models.baseline import BaselineModel, BaselineConfig
from app.core.odds import american_to_implied_prob, american_to_decimal
from app.core.ev import evaluate
from app.core.live_eval import add_fair_probs
# near the top of the file
import os
import contextlib

def _get_odds_api_key() -> str | None:
    # 1) env var takes precedence (works locally and on most hosts)
    key = os.getenv("ODDS_API_KEY")
    if key:
        return key
    
    # 2) streamlit secrets (works on Streamlit Cloud or if you add .streamlit/secrets.toml)
    with contextlib.suppress(Exception):
        # Access inside try/suppress so missing secrets.toml won't crash
        return st.secrets["ODDS_API_KEY"]  # type: ignore[attr-defined]
    return None


# Try to use your core arbitrage module; fall back to internal scan if not available.
try:
    from app.core.arb import is_two_way_arb, arb_stakes
    HAS_CORE_ARB = True
except Exception:
    HAS_CORE_ARB = False

@st.cache_data(show_spinner=False)
def fetch_player_event_odds(
    sport: str,
    region: str,
    markets: tuple[str, ...],
    bookmakers: tuple[str, ...] | None = None,
    max_events: int = 20,   # to control usage
    odds_format: str = "american",
) -> pd.DataFrame:
    """
    Player props must use the per-event odds endpoint.
    1) Get upcoming events
    2) For each event_id, call /events/{eventId}/odds with player_* markets
    """
    api_key = _get_odds_api_key()
    if not api_key:
        st.error("Missing ODDS_API_KEY.")
        st.stop()

    base = "https://api.the-odds-api.com/v4"
    # 1) list events
    evs_url = f"{base}/sports/{sport}/events"
    evs_params = {"apiKey": api_key, "regions": region}
    evs_resp = requests.get(evs_url, params=evs_params, timeout=30)
    evs_resp.raise_for_status()
    events = evs_resp.json()
    if not events:
        return pd.DataFrame()

    rows = []
    markets_param = ",".join(markets)
    bookies_param = ",".join(bookmakers) if bookmakers else None

    # 2) per-event odds for player markets
    for ev in events[:max_events]:
        event_id = ev.get("id")
        if not event_id:
            continue
        url = f"{base}/sports/{sport}/events/{event_id}/odds"
        params = {
            "apiKey": api_key,
            "regions": region,
            "oddsFormat": odds_format,
            "markets": markets_param,
        }
        if bookies_param:
            params["bookmakers"] = bookies_param

        r = requests.get(url, params=params, timeout=30)
        # Some events won't have props; skip cleanly
        if r.status_code == 404:
            continue
        r.raise_for_status()
        data = r.json()

        # Normalize: flatten bookmakers -> markets -> outcomes
        event_name = f"{ev.get('home_team','')} vs {ev.get('away_team','')}".strip()
        commence = ev.get("commence_time")

        for bk in data.get("bookmakers", []):
            book = bk.get("key") or bk.get("title")
            for mk in bk.get("markets", []):
                mkey = mk.get("key")
                point = mk.get("point")  # some props put line here; outcomes also carry point
                for oc in mk.get("outcomes", []):
                    # The Odds API usually puts Over/Under name in 'name' and player name in 'description' or 'participant'
                    outcome = oc.get("name")
                    player = oc.get("description") or oc.get("participant") or oc.get("player")
                    price_american = oc.get("price") or oc.get("american") or oc.get("price_american")
                    price_decimal = oc.get("price_decimal")

                    # derive decimal if needed
                    if price_decimal is None and price_american is not None:
                        try:
                            price_decimal = american_to_decimal(float(price_american))
                        except Exception:
                            price_decimal = None

                    # prefer outcome-level point if present
                    line = oc.get("point", point)

                    rows.append({
                        "event_id": event_id,
                        "event": event_name,
                        "commence_time": commence,
                        "player": player,
                        "market": mkey,
                        "point": line,
                        "outcome": outcome,
                        "bookmaker": book,
                        "price_american": price_american,
                        "price_decimal": price_decimal,
                        "decimal_odds": price_decimal,  # convenience alias
                    })

    return pd.DataFrame(rows)


def _scan_two_way_arbs_from_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple 2-way arb table using your pairwise core helpers.
    Expects df with columns: event_id, market, outcome, bookmaker, price_american (and/or price_decimal).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["event_id","market","arb_implied_sum","arb_exists"])

    # Keep only 2-outcome markets
    g = df.groupby(["event_id","market"])["outcome"].nunique().reset_index(name="n_outcomes")
    keep = g[g["n_outcomes"] == 2][["event_id","market"]]
    df2 = df.merge(keep, on=["event_id","market"], how="inner")

    # Best price per outcome across books (by highest decimal odds)
    def _american_to_decimal(a):
        a = float(a)
        return 1.0 + (a / 100.0) if a > 0 else 1.0 + (100.0 / abs(a))

    tmp = df2.copy()
    if "price_decimal" not in tmp.columns:
        tmp["price_decimal"] = tmp["price_american"].astype(float).map(_american_to_decimal)

    # For each event/market/outcome choose max decimal odds and remember which book had it
    idx = tmp.groupby(["event_id","market","outcome"])["price_decimal"].idxmax()
    best = tmp.loc[idx, ["event_id","market","outcome","bookmaker","price_decimal","price_american"]]

    # Pivot to two columns (one per outcome)
    piv = best.pivot_table(index=["event_id","market"],
                           columns="outcome",
                           values=["price_decimal","price_american","bookmaker"],
                           aggfunc="first")

    if piv.empty or len(piv.columns.get_level_values(1).unique()) != 2:
        return pd.DataFrame(columns=["event_id","market","arb_implied_sum","arb_exists"])

    outcomes = list(piv.columns.get_level_values(1).unique())
    o1, o2 = outcomes[0], outcomes[1]

    # Flatten columns
    piv.columns = [f"{lvl0}_{lvl1}" for (lvl0,lvl1) in piv.columns]
    piv = piv.reset_index()

    # Use your core pairwise tester
    ok_list, margin_list, invsum_list = [], [], []
    for _, r in piv.iterrows():
        d1 = float(r[f"price_decimal_{o1}"])
        d2 = float(r[f"price_decimal_{o2}"])
        ok, margin = is_two_way_arb(d1, d2)
        ok_list.append(ok)
        margin_list.append(margin)
        invsum_list.append(1.0/d1 + 1.0/d2)

    piv["arb_exists"] = ok_list
    piv["arb_margin"] = margin_list
    piv["arb_implied_sum"] = invsum_list

    # Rename columns to readable names
    out = piv.rename(columns={
        f"bookmaker_{o1}": f"best_book_{o1}",
        f"bookmaker_{o2}": f"best_book_{o2}",
        f"price_american_{o1}": f"best_american_{o1}",
        f"price_american_{o2}": f"best_american_{o2}",
        f"price_decimal_{o1}": f"best_decimal_{o1}",
        f"price_decimal_{o2}": f"best_decimal_{o2}",
    })
    # Sort by best opportunities first
    return out.sort_values(["arb_exists","arb_implied_sum"]).reset_index(drop=True)

def _add_stakes_from_core(arbs: pd.DataFrame, total_stake: float = 100.0) -> pd.DataFrame:
    """Add stake splits & guaranteed profit using your core arb_stakes()."""
    if arbs.empty:
        return arbs
    arbs = arbs.copy()
    # Find which two outcome columns we have
    dec_cols = [c for c in arbs.columns if c.startswith("best_decimal_")]
    if len(dec_cols) < 2:
        return arbs
    d1_col, d2_col = dec_cols[:2]
    name1 = d1_col.replace("best_decimal_", "")
    name2 = d2_col.replace("best_decimal_", "")

    s1_list, s2_list, ret_list, prof_list = [], [], [], []
    for _, r in arbs.iterrows():
        d1 = float(r[d1_col]); d2 = float(r[d2_col])
        s1, s2, guaranteed_return, profit = arb_stakes(d1, d2, total_stake)
        s1_list.append(round(s1, 2)); s2_list.append(round(s2, 2))
        ret_list.append(round(guaranteed_return, 2)); prof_list.append(round(profit, 2))

    arbs[f"stake_{name1}"] = s1_list
    arbs[f"stake_{name2}"] = s2_list
    arbs["guaranteed_return_$"] = ret_list
    arbs["guaranteed_profit_$"] = prof_list
    return arbs


load_dotenv()

st.set_page_config(page_title="Live Edges & Kelly", page_icon="💹", layout="wide")
st.title("Live Odds → Edges & Kelly")
st.caption("Real-time odds → vig-free probabilities → baseline model → EV & Kelly. Optional arbitrage across books.")

# =========================
# Odds fetch (live, cached) — direct JSON from The Odds API
# =========================
API_BASE = "https://api.the-odds-api.com/v4"

@st.cache_data(show_spinner=False, ttl=30)
def fetch_live_odds(sport: str, region: str, markets: tuple[str, ...], bookmakers: tuple[str, ...]) -> pd.DataFrame:
    """
    Calls The Odds API directly and returns a normalized DataFrame:
      event_id, event, commence_time_utc, market, outcome, bookmaker, price_american [, point, price_decimal]
    """
    api_key = _get_odds_api_key()
    if not api_key:
        st.error(
        "No API key found. Set ODDS_API_KEY as an environment variable "
        "or add it to `.streamlit/secrets.toml` as ODDS_API_KEY = \"...\""
        )
        st.stop()


    params = {
        "apiKey": api_key,
        "regions": region,                              # e.g., 'us'
        "markets": ",".join(markets) or "h2h",         # e.g., 'h2h,spreads,totals'
        "oddsFormat": "american",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)    # optional filter

    url = f"{API_BASE}/sports/{sport}/odds"
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    # Optional: expose rate limit info in the UI
    rl_used = resp.headers.get("x-requests-used")
    rl_remaining = resp.headers.get("x-requests-remaining")
    if rl_used and rl_remaining:
        st.caption(f"Requests used: {rl_used} • Remaining: {rl_remaining}")

    return _normalize_theoddsapi_payload(
        payload,
        want_markets=set(markets),
        want_books=set(bookmakers) if bookmakers else None,
    )

def _normalize_theoddsapi_payload(payload, want_markets: set[str], want_books: set[str] | None) -> pd.DataFrame:
    """
    Convert The Odds API JSON into a flat DataFrame:
      event_id, event, commence_time_utc, market, outcome, bookmaker, price_american [, point, price_decimal]
    """
    if payload is None:
        return pd.DataFrame()

    rows = []
    events = payload if isinstance(payload, list) else payload.get("data") or payload.get("events") or []
    for ev in events:
        event_id = ev.get("id") or ev.get("event_id") or ev.get("key")
        start_raw = ev.get("commence_time") or ev.get("commence_time_utc") or ev.get("start_time")
        try:
            commence_time_utc = pd.to_datetime(start_raw, utc=True).isoformat() if start_raw else None
        except Exception:
            commence_time_utc = start_raw

        home = ev.get("home_team") or ev.get("home")
        away = ev.get("away_team") or ev.get("away")
        event_name = ev.get("event") or (f"{home} vs {away}" if home and away else str(event_id))

        for bk in ev.get("bookmakers", []):
            book_key = (bk.get("key") or bk.get("title") or bk.get("bookmaker") or "").lower()
            if want_books and book_key not in want_books:
                continue

            for m in bk.get("markets", []):
                market = (m.get("key") or m.get("market") or "").lower()
                if want_markets and market not in want_markets:
                    continue

                for o in m.get("outcomes", []):
                    outcome = (o.get("name") or o.get("outcome") or "").lower()
                    price_am = o.get("price") or o.get("odds") or o.get("american")
                    price_dec = o.get("price_decimal") or o.get("decimal")
                    point = o.get("point")

                    # If only decimal provided, derive American
                    if price_am is None and price_dec is not None:
                        try:
                            d = float(price_dec)
                            price_am = int(round((d - 1.0) * 100)) if d >= 2.0 else int(round(-100 / (d - 1.0)))
                        except Exception:
                            pass

                    rows.append({
                        "event_id": event_id,
                        "event": event_name,
                        "commence_time_utc": commence_time_utc,
                        "market": market,
                        "outcome": outcome,
                        "bookmaker": book_key,
                        "price_american": pd.to_numeric(price_am, errors="coerce"),
                        "point": pd.to_numeric(point, errors="coerce"),
                    })

    df = pd.DataFrame.from_records(rows)
    if df.empty:
        return df

    # Normalize text columns
    df["market"] = df["market"].astype(str).str.lower()
    df["outcome"] = df["outcome"].astype(str).str.lower()

    # Derive decimal odds when American present
    def _american_to_decimal(a):
        try:
            a = float(a)
            return 1.0 + (a / 100.0) if a > 0 else 1.0 + (100.0 / abs(a))
        except Exception:
            return np.nan

    df["price_decimal"] = df["price_american"].map(_american_to_decimal)

    sort_cols = [c for c in ["commence_time_utc", "event", "bookmaker", "market", "outcome"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)

# EV/Kelly helpers vectorized

def ev_per_unit(decimal_odds, model_prob):
    """
    Works with scalars OR pandas Series/arrays.
    Returns expected profit per $1 stake.
    """
    d = pd.to_numeric(decimal_odds, errors="coerce")
    p = pd.to_numeric(model_prob, errors="coerce")
    b = d - 1.0
    q = 1.0 - p
    return p * b - q

def kelly_fraction(decimal_odds, model_prob):
    """
    Works with scalars OR pandas Series/arrays.
    Kelly = (b*p - q) / b, clipped at >= 0 and b>0.
    """
    d = pd.to_numeric(decimal_odds, errors="coerce")
    p = pd.to_numeric(model_prob, errors="coerce")
    b = d - 1.0
    q = 1.0 - p

    # avoid divide-by-zero and negative b
    k = np.where(b > 0, (b * p - q) / b, 0.0)
    # no negative Kelly
    return np.clip(k, 0.0, None)


# =========================
# Fallback 2-way arb scan
# =========================
def _scan_two_way_arbs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple internal two-way arb scan across books. Looks only at markets with exactly two outcomes
    (e.g., h2h without draw, spreads, totals). Returns best odds and whether implied sums < 1.
    """
    core = df.copy()
    core = core[core["market"].isin(["h2h", "spreads", "totals"])]
    core = core[~core["outcome"].eq("draw")]
    core["imp"] = core["price_american"].map(american_to_implied_prob)

    # groups with exactly two distinct outcomes
    n_outcomes = core.groupby(["event_id", "market"])["outcome"].nunique()
    valid = n_outcomes[n_outcomes == 2].index
    if len(valid) == 0:
        return pd.DataFrame(columns=["event_id","market","arb_implied_sum","arb_exists"])

    core = core.set_index(["event_id","market"]).loc[valid].reset_index()

    # best (lowest implied; i.e., highest price) per outcome
    idx_best = core.groupby(["event_id","market","outcome"])["imp"].idxmin()
    best = core.loc[idx_best].copy()

    piv_imp = best.pivot(index=["event_id","market"], columns="outcome", values="imp")
    if piv_imp.shape[1] != 2:
        return pd.DataFrame(columns=["event_id","market","arb_implied_sum","arb_exists"])

    outcomes = list(piv_imp.columns)
    piv_imp["arb_implied_sum"] = piv_imp[outcomes[0]] + piv_imp[outcomes[1]]
    piv_imp["arb_exists"] = piv_imp["arb_implied_sum"] < 1.0

    piv_book = best.pivot(index=["event_id","market"], columns="outcome", values="bookmaker")
    piv_price = best.pivot(index=["event_id","market"], columns="outcome", values="price_american")

    out = piv_imp.reset_index()
    for c in outcomes:
        out[f"best_book_{c}"]       = piv_book[c].values
        out[f"best_american_{c}"]   = piv_price[c].values
    return out.sort_values(["arb_exists","arb_implied_sum"]).reset_index(drop=True)


# =========================
# Sidebar (you’ve got 9 US books; feel free to expand defaults)
# =========================
st.sidebar.header("Live data")
sport = st.sidebar.selectbox(
    "Sport",
    ["basketball_nba", "icehockey_nhl", "americanfootball_nfl", "baseball_mlb", "soccer_epl"],
    index=0,
)
region = st.sidebar.selectbox("Region", ["us", "uk", "eu", "au"], index=0)
markets = st.sidebar.multiselect("Markets", ["h2h", "spreads", "totals"], default=["h2h", "spreads", "totals"])
bookmakers = st.sidebar.multiselect(
    "Bookmakers (optional)",
    # Add/adjust to match your provider’s book keys (you said ~9 for US)
    ["pinnacle","draftkings","fanduel","betmgm","caesars","williamhill_us","pointsbetus","betrivers","wynnbet"],
    default=[]
)

st.sidebar.header("Model")
m = st.slider(
    "Model multiplier (confidence)",
    min_value=0.85, max_value=1.30, value=1.00, step=0.01,
    help="Scales your model’s win probability: p' = clip(m * p, 0–1). Start at 1.00 and adjust slowly."
)

# Default prior strength (controls how heavily the baseline trusts prior_center)
prior_strength = 0.5     # try 0.25–1.0 range; smaller = more reactive, larger = more conservative
prior_center = 0.50
min_p, max_p = 0.01, 0.99

cfg = BaselineConfig(
    prior_strength=prior_strength,
    prior_center=prior_center,
    min_p=min_p,
    max_p=max_p
)
baseline = BaselineModel(cfg)


show_arb = st.sidebar.toggle("Enable Arbitrage tab", value=True)

# =========================
# Tabs
# =========================
tabs = ["Live odds", "Edges & Kelly","Players"]
if show_arb:
    tabs.append("Arbitrage")
tab_objs = st.tabs(tabs)

# ── Live odds ─────────────────────────────────────────────────────────────────
with tab_objs[0]:
    st.subheader("Live odds")
    df_live = fetch_live_odds(sport, region, tuple(markets), tuple(bookmakers))
    st.caption(f"{len(df_live)} selections loaded.")
    if df_live.empty:
        st.info("No odds returned for this selection.")
    else:
        st.dataframe(
            df_live.sort_values(["commence_time_utc","bookmaker","market","outcome"]).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )


# ── Edges & Kelly ────────────────────────────────────────────────────────────
with tab_objs[1]:
    st.subheader("Edges & Kelly")
    df_live = fetch_live_odds(sport, region, tuple(markets), tuple(bookmakers))

    if df_live.empty:
        st.info("No odds returned for this selection.")
    else:
        # 1) Vig-free probabilities per book/event/market
        df_fair = add_fair_probs(df_live)

        # 2) Ensure we have decimal odds available
        if "decimal_odds" not in df_fair.columns:
            if "price_decimal" in df_fair.columns:
                df_fair["decimal_odds"] = df_fair["price_decimal"].astype(float)
            elif "american_odds" in df_fair.columns:
                df_fair["decimal_odds"] = df_fair["american_odds"].astype(float).map(american_to_decimal)
            elif "price_american" in df_fair.columns:
                df_fair["decimal_odds"] = df_fair["price_american"].astype(float).map(american_to_decimal)
            else:
                st.error("No decimal odds found (need decimal_odds, price_decimal, american_odds, or price_american).")
                st.stop()

        # 3) Baseline model → RAW model probability (row-wise; uses p_fair when present)
        def _predict_row(r) -> float:
            return baseline.predict(
                market_prob_vigfree = r.get("p_fair", None),  # may be None; baseline can choose a fallback
                selection           = r.get("outcome", ""),
                market_type         = r.get("market", ""),
                point               = r.get("point", None),
            )

        df_fair["model_prob"] = df_fair.apply(_predict_row, axis=1).astype(float)

        # 4) Apply the slider multiplier m and clamp to [min_p, max_p]
        #    (m is the "Model multiplier (confidence)" slider you added earlier)
        df_fair["model_prob_adj"] = np.clip(df_fair["model_prob"] * m, min_p, max_p)

        # 5) EV & Kelly using the **adjusted** probability
        def _eval_row(r):
            be = evaluate(decimal_odds=float(r["decimal_odds"]), model_prob=float(r["model_prob_adj"]))
            # evaluate() returns a BetEval dataclass in your ev.py (ev_per_dollar, kelly, kelly_quarter)
            return pd.Series({
                "ev_per_dollar": round(be.ev_per_dollar, 4),
                "kelly": round(be.kelly, 4),
                "kelly_quarter": round(be.kelly_quarter, 4),
            })

        df_eval = df_fair.apply(_eval_row, axis=1)
        df_out = pd.concat([df_fair, df_eval], axis=1)

        # 6) Optional: show "edge" vs vig-free probability if available
        if "p_fair" in df_out.columns:
            df_out["edge_prob"] = (df_out["model_prob_adj"] - df_out["p_fair"]).round(4)

        # 7) Display — sort by EV per $ descending
        cols_show = [
            "event", "commence_time", "market", "outcome", "bookmaker",
            "decimal_odds", "p_fair", "model_prob", "model_prob_adj",
            "ev_per_dollar", "kelly", "kelly_quarter", "edge_prob"
        ]
        cols_show = [c for c in cols_show if c in df_out.columns]
        st.dataframe(
            df_out[cols_show].sort_values("ev_per_dollar", ascending=False),
            use_container_width=True, hide_index=True
        )

# ── Players (Edges & Kelly) ───────────────────────────────────────────────────
with tab_objs[2]:
    st.subheader("Player Props — Edges & Kelly")

    # Controls (re-use your existing sidebar selects for sport/region/bookmakers if you like)
    default_markets = ["player_points", "player_assists", "player_rebounds"]
    ply_markets = st.multiselect("Player markets", default_markets, default=default_markets)
    player_search = st.text_input("Filter by player name (optional)", value="")

   
    # Fetch player props by calling the same live-odds fetcher with player markets
    df_players = fetch_player_event_odds(sport, region, tuple(ply_markets), tuple(bookmakers))


    if df_players.empty:
        st.info("No player props returned for this selection.")
    else:
        # Optional filter by player name
        if player_search:
            mask = df_players["player"].str.contains(player_search, case=False, na=False) if "player" in df_players.columns else True
            df_players = df_players[mask].copy()

        # Compute edges using the SAME slider 'm' and the SAME baseline you already created
        from app.core.player_eval import compute_player_edges
        df_out = compute_player_edges(df_players, baseline_model=baseline, m=m, min_p=min_p, max_p=max_p, markets=ply_markets)

        if df_out.empty:
            st.success("No opportunities found at the moment.")
        else:
            cols = [
                "event", "commence_time", "player", "market", "point", "outcome",
                "bookmaker", "decimal_odds", "p_fair", "model_prob", "model_prob_adj",
                "ev_per_dollar", "kelly", "kelly_quarter", "edge_prob"
            ]
            cols = [c for c in cols if c in df_out.columns]
            st.dataframe(df_out[cols], use_container_width=True, hide_index=True)


# ── Arbitrage (optional) ─────────────────────────────────────────────────────
if show_arb:
    with tab_objs[2]:
        st.subheader("Two-way arbitrage across books")
        df_live = fetch_live_odds(sport, region, tuple(markets), tuple(bookmakers))
        if df_live.empty:
            st.info("No odds returned for this selection.")
        else:
            if HAS_CORE_ARB:
                arbs = _scan_two_way_arbs_from_core(df_live)
                if arbs.empty or not arbs.get("arb_exists", pd.Series([False])).any():
                    st.success("No 2-way arbs found right now.")
                else:
                    arbs = _add_stakes_from_core(arbs, total_stake=100.0)
                    st.warning("Potential arbs detected — verify limits and rules before placing bets.")
                    st.dataframe(arbs, use_container_width=True, hide_index=True)

            else:
                # Fallback internal scan
                arbs = _scan_two_way_arbs(df_live)
                if arbs.empty or not arbs["arb_exists"].any():
                    st.success("No 2-way arbs found right now.")
                else:
                    # Simple equalized stakes for a $100 notional using implied→decimal
                    recs = []
                    outcome_cols = [c for c in arbs.columns if c.startswith("best_american_")]
                    if len(outcome_cols) >= 2:
                        o1, o2 = outcome_cols[:2]
                        name1, name2 = o1.replace("best_american_", ""), o2.replace("best_american_", "")
                        for _, row in arbs.iterrows():
                            p1 = american_to_implied_prob(row[o1])
                            p2 = american_to_implied_prob(row[o2])
                            if p1 + p2 >= 1.0:
                                s1 = s2 = guaranteed = 0.0
                            else:
                                total = 100.0
                                s1 = total * p2 / (p1 + p2)
                                s2 = total * p1 / (p1 + p2)
                                d1, d2 = 1.0 / p1, 1.0 / p2
                                ret1 = s1 * (d1 - 1.0) - s2
                                ret2 = s2 * (d2 - 1.0) - s1
                                guaranteed = min(ret1, ret2)
                            rec = dict(row)
                            rec[f"stake_{name1}"] = round(s1, 2)
                            rec[f"stake_{name2}"] = round(s2, 2)
                            rec["guaranteed_profit_$"] = round(guaranteed, 2)
                            recs.append(rec)
                    arbs_stakes = pd.DataFrame.from_records(recs) if recs else arbs
                    st.warning("Potential arbs detected — verify limits and rules before placing bets.")
                    st.dataframe(arbs_stakes, use_container_width=True, hide_index=True)
