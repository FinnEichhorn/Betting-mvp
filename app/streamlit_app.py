from __future__ import annotations

# --- PATH BOOTSTRAP: ensure repo root is on sys.path ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # repo root (parent of 'app')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app.components.ui import section, subtle, warn, success
from app.core.odds import american_to_implied_prob, american_to_decimal, remove_vig_two_way
from app.core.ev import evaluate
from app.core.arb import is_two_way_arb, arb_stakes
from app.data.loader import load_odds_csv
from app.models.baseline import baseline_probability
import app.db as db

# -------- helpers & session init --------
load_dotenv()
st.set_page_config(page_title="Betting MVP", page_icon="💹", layout="wide")

def init_state():
    st.session_state.setdefault("odds_df", None)
    st.session_state.setdefault("use_sample", False)
    st.session_state.setdefault("only_pos_ev", False)
    st.session_state.setdefault("min_edge", 0.0)
    # track last loaded source (nice for debugging / UI)
    st.session_state.setdefault("odds_source", None)

init_state()

@st.cache_data(show_spinner=False)
def _load_from_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))

@st.cache_data(show_spinner=False)
def _load_sample_csv() -> pd.DataFrame:
    return load_odds_csv(Path("data/sample_odds.csv"))

# --- DB init ---
db.init_db()

# --- Sidebar: Bankroll & Settings ---
st.sidebar.header("Settings")
start_roll = st.sidebar.number_input(
    "Starting bankroll ($)",
    min_value=0.0,
    value=float(db.get_meta("starting_bankroll", "2000") or 2000.0),
    step=50.0,
)
if st.sidebar.button("Save settings", key="save_settings"):
    db.set_meta("starting_bankroll", str(start_roll))
    db.record_snapshot()
    st.success("Settings saved.")

# --- Sidebar: Odds loading controls ---
def _on_upload_change():
    f = st.session_state.get("odds_upload")
    if f is None:
        return
    try:
        st.session_state["odds_df"] = _load_from_bytes(f.getvalue())
        st.session_state["odds_source"] = f"name: {getattr(f, 'name', 'uploaded.csv')}"
    except Exception as e:
        warn(f"Failed to load odds: {e}")

uploaded = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
    key="odds_upload",
    on_change=_on_upload_change,
)

if st.sidebar.button("Load sample odds", key="btn_use_sample"):
    try:
        st.session_state["odds_df"] = _load_sample_csv()
        st.session_state["odds_source"] = "sample:data/sample_odds.csv"
    except Exception as e:
        warn(f"Failed to load sample: {e}")

st.title("Sports Betting MVP 💹")
subtle("Edges, Kelly staking, arbitrage, and bet logging — minimal but extensible.")

with st.sidebar.expander("Debug: session"):
    odf = st.session_state.get("odds_df", None)
    st.write({
        "has_df": isinstance(odf, pd.DataFrame),
        "rows": (len(odf) if isinstance(odf, pd.DataFrame) else 0),
        "source": st.session_state.get("odds_source")
    })


# ---- Guard: require odds in session_state ----
odf = st.session_state.get("odds_df")
if not isinstance(odf, pd.DataFrame) or odf.empty:
    st.info("Load some odds to begin (sidebar → upload or use sample).")
    st.stop()

# From here on, always work on a copy of session data
odds_df = odf.copy()


# --- Ensure required columns ---
required = ["event_id","league","start_time","home_team","away_team","market_type","selection","book","american_odds"]
missing = [c for c in required if c not in odds_df.columns]
if missing:
    warn(f"Missing required columns: {missing}")
    st.stop()

# --- Sanitize inputs ---
odds_df = odds_df.copy()

# Strip whitespace from text columns (helps with weird CSVs)
for col in ["event_id","league","home_team","away_team","market_type","selection","book"]:
    if col in odds_df.columns:
        odds_df[col] = odds_df[col].astype(str).str.strip()

# Parse odds to numeric, drop bad rows with a clear message
odds_df["american_odds"] = pd.to_numeric(odds_df["american_odds"], errors="coerce")
bad_mask = odds_df["american_odds"].isna()
if bad_mask.any():
    st.warning(f"Removed {int(bad_mask.sum())} row(s) with non-numeric american_odds.")
    odds_df = odds_df[~bad_mask]

# Now safe to cast to int
odds_df["american_odds"] = odds_df["american_odds"].astype(int)

# Derive decimal_odds if absent
if "decimal_odds" not in odds_df.columns:
    odds_df["decimal_odds"] = odds_df["american_odds"].map(american_to_decimal)

# Parse start_time for sorting
odds_df["start_time"] = pd.to_datetime(odds_df["start_time"], errors="coerce")

# Normalize selection and market_type early so models/EV behave
if "selection" in odds_df.columns:
    odds_df["selection"] = odds_df["selection"].astype(str).str.strip().str.upper()

if "market_type" in odds_df.columns:
    odds_df["market_type"] = odds_df["market_type"].astype(str).str.strip().str.upper()

# Normalize labels for the model
odds_df["selection"] = odds_df["selection"].astype(str).str.strip().str.upper()
odds_df["market_type"] = odds_df["market_type"].astype(str).str.strip().str.upper()


def compute_edges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # raw implied prob from american odds
    df["imp_raw"] = df["american_odds"].astype(int).map(american_to_implied_prob)

    # default: if we can't vig-strip, keep raw
    df["prob_vigfree"] = df["imp_raw"].values

    # Two-way vig removal per (event, market, book)
    for (eid, mkt, book), g in df.groupby(["event_id", "market_type", "book"], dropna=False):
        if len(g) == 2:
            p1, p2 = remove_vig_two_way(float(g.iloc[0]["imp_raw"]), float(g.iloc[1]["imp_raw"]))
            df.loc[g.index, "prob_vigfree"] = [p1, p2]

    # Baseline model probability (normalize inputs already uppercased)
    df["model_prob"] = [
        baseline_probability(sel, h, a, mkt)
        for sel, h, a, mkt in zip(df["selection"], df["home_team"], df["away_team"], df["market_type"])
    ]

    # Safety: coerce/clip to (0,1) to avoid NaNs/inf in EV
    df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce").fillna(0.5)
    df["model_prob"] = df["model_prob"].clip(1e-6, 1 - 1e-6)

    # Ensure decimal_odds valid
    df["decimal_odds"] = pd.to_numeric(df["decimal_odds"], errors="coerce")
    bad_dec = (df["decimal_odds"].isna()) | (df["decimal_odds"] <= 1.0)
    if bad_dec.any():
        st.info(f"Dropped {int(bad_dec.sum())} row(s) with invalid decimal_odds.")
        df = df[~bad_dec]

    # EV & Kelly
    evals = df.apply(lambda r: evaluate(r["decimal_odds"], r["model_prob"]), axis=1)
    df["ev_per_$"] = [e.ev_per_dollar for e in evals]
    df["kelly_25"] = [e.kelly_quarter for e in evals]

    # Convenience field for filtering
    df["edge_%"] = df["ev_per_$"] * 100.0

    return df


edges_df = compute_edges(odds_df)


# --- Best price per selection across books (robust to NaNs) ---
# Normalize selection just in case
edges_df["selection"] = edges_df["selection"].astype(str).str.upper().str.strip()

# Keep only known 2-way selections for arb logic; others still work for EV
known = {"HOME","AWAY","OVER","UNDER"}
unknown_mask = ~edges_df["selection"].isin(known)
if unknown_mask.any():
    st.info(f"Ignoring {int(unknown_mask.sum())} row(s) for arbitrage pairing due to unknown selection label(s). They still appear in the Edges table.")


best_prices = (
    edges_df
    .dropna(subset=["decimal_odds"])                         # drop rows where price is missing
    .sort_values(["event_id","market_type","selection","decimal_odds"], ascending=[True, True, True, False])
    .drop_duplicates(subset=["event_id","market_type","selection"], keep="first")
    .reset_index(drop=True)
)


# --- Tabs ---
T1, T2, T3, T4 = st.tabs(["Today/Edges", "Arbitrage", "Bets & Bankroll", "Raw Odds"])


with T1:
    section("Edges by Market (¼-Kelly suggestions)")

 # ---- Stable widget state (inside with T1:) ----
    st.session_state.setdefault("ui_only_pos_ev", True)
    st.session_state.setdefault("ui_min_edge", 0.0)

    st.session_state["ui_min_edge"] = st.slider(
        "Min edge (% per $1)",
        -10.0, 20.0,
        float(st.session_state["ui_min_edge"]),
        0.5,
        key="min_edge_sld",
    )
    st.session_state["ui_only_pos_ev"] = st.checkbox(
        "Only positive EV",
        value=bool(st.session_state["ui_only_pos_ev"]),
        key="only_pos_ev_chk",
    )


    # Apply filters using the stable state
    view = edges_df.copy()
    if st.session_state["ui_only_pos_ev"]:
        view = view[view["ev_per_$"] > 0]
    view = view[view["edge_%"] >= float(st.session_state["ui_min_edge"])]

    # ---- (#4) Empty-table explanation (debug) ----
    if view.empty:
        with st.expander("Why is this empty? (debug)"):
            st.write({
                "rows_loaded": len(edges_df),
                "positive_ev_rows": int((edges_df["ev_per_$"] > 0).sum()),
                "min_edge_filter": float(st.session_state["ui_min_edge"]),
                "only_pos_ev": bool(st.session_state["ui_only_pos_ev"]),
            })
            st.write("Model prob summary:", edges_df["model_prob"].describe())
            st.write("EV per $ summary:", edges_df["ev_per_$"].describe())



    cols = [
    "event_id","league","start_time","market_type","selection","book",
    "american_odds","decimal_odds","model_prob","ev_per_$","kelly_25"
    ]
    st.dataframe(view[cols].sort_values(["start_time","event_id","market_type","selection"]).reset_index(drop=True), use_container_width=True)


    st.divider()
    section("Place a Bet (log to DB)")
    with st.form("place_bet"):
        pick = st.selectbox("Pick (from filtered table above)", options=[
            f"{r.event_id} | {r.market_type} | {r.selection} | {r.book} | {int(r.american_odds)}"
            for _, r in view.iterrows()
        ]) if len(view) else None
        stake = st.number_input("Stake ($)", min_value=1.0, value=25.0, step=1.0)
        submitted = st.form_submit_button("Log bet")


    if submitted and pick:
        parts = [p.strip() for p in pick.split("|")]
        event_id, market_type, selection, book, am_str = parts
        row = view[(view["event_id"]==event_id) & (view["market_type"]==market_type) & (view["selection"]==selection) & (view["book"]==book) & (view["american_odds"]==int(am_str))].iloc[0]
        bet_id = db.log_bet(
            ts=datetime.utcnow().isoformat(),
            event_id=row.event_id,
            league=row.league,
            selection=row.selection,
            book=row.book,
            american_odds=int(row.american_odds),
            decimal_odds=float(row.decimal_odds),
            stake=float(stake),
        )
        success(f"Bet logged with id={bet_id}")


with T2:
    section("Two-way Arbitrage Scanner (best prices across books)")
    total_stake = st.number_input("Total stake for arb calc ($)", min_value=10.0, value=100.0, step=5.0)
    
    # Build pairs per event/market: HOME vs AWAY (or OVER vs UNDER)
    pairs = []
    for (eid, mkt), g in best_prices.groupby(["event_id","market_type"], dropna=False):
        if set(g["selection"]) >= {"HOME","AWAY"}:
            home = g[g["selection"]=="HOME"].iloc[0]
            away = g[g["selection"]=="AWAY"].iloc[0]
            pairs.append((eid, mkt, home, away))
        elif set(g["selection"]) >= {"OVER","UNDER"}:
            over = g[g["selection"]=="OVER"].iloc[0]
            under = g[g["selection"]=="UNDER"].iloc[0]
            pairs.append((eid, mkt, over, under))


    rows = []
    for eid, mkt, a, b in pairs:
        arb, margin = is_two_way_arb(float(a.decimal_odds), float(b.decimal_odds))
        if arb:
            s1, s2, ret, profit = arb_stakes(float(a.decimal_odds), float(b.decimal_odds), float(total_stake))
            rows.append({
                "event_id": eid,
                "market_type": mkt,
                "sel1": a.selection,
                "book1": a.book,
                "dec1": a.decimal_odds,
                "sel2": b.selection,
                "book2": b.book,
                "dec2": b.decimal_odds,
                "edge_margin": margin,
                "stake1": round(s1,2),
                "stake2": round(s2,2),
                "guaranteed_return": round(ret,2),
                "profit": round(profit,2),
            })


    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("profit", ascending=False), use_container_width=True)
    else:
        st.info("No two-way arbitrage found with current prices.")
    
with T3:
    st.header("Bankroll & Bets")

    start, open_risk, realized = db.bankroll_snapshot()
    current = start - open_risk + realized

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Starting", f"${start:,.2f}")
    c2.metric("Open Risk", f"${open_risk:,.2f}")
    c3.metric("Realized PnL", f"${realized:,.2f}")
    c4.metric("Current", f"${current:,.2f}")

    bets = db.list_bets()
    if bets:
        dfb = pd.DataFrame(
            bets,
            columns=["id","ts","event_id","league","selection","book","american_odds","decimal_odds","stake","result","pnl"]
        ).sort_values("ts", ascending=False)
        st.dataframe(dfb, use_container_width=True)

        st.subheader("Settle Bet")
        sel_id = st.selectbox("Bet ID", options=dfb["id"].tolist())
        res = st.selectbox("Result", options=["WIN","LOSE","PUSH"])
        if st.button("Settle"):
            db.settle_bet(int(sel_id), res)
            st.success("Bet settled.")
            st.rerun()

        st.download_button(
            label="Download bets CSV",
            data=dfb.to_csv(index=False).encode("utf-8"),
            file_name="bets_export.csv",
            mime="text/csv",
        )
    else:
        st.info("No bets yet. Log one on the Edges tab.")

    hist = db.history_rows()
    if hist:
        dh = pd.DataFrame(hist, columns=["ts","starting","open_risk","realized","current"])
        dh["ts"] = pd.to_datetime(dh["ts"])
        st.line_chart(dh.set_index("ts")["current"], height=220)


with T4:
    section("Raw odds (debug)")
    st.dataframe(odds_df, use_container_width=True)
    section("Computed edges (debug)")
    st.dataframe(edges_df, use_container_width=True)


    st.caption("Use this tab when something looks off — check inputs and derived columns.")