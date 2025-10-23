from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime

DB_PATH = Path("data/bets.db")

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS bets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  event_id TEXT NOT NULL,
  league TEXT,
  selection TEXT NOT NULL,
  book TEXT NOT NULL,
  american_odds INTEGER NOT NULL,
  decimal_odds REAL NOT NULL,
  stake REAL NOT NULL,
  result TEXT,
  pnl REAL,
  closing_odds REAL
);

CREATE TABLE IF NOT EXISTS bankroll_history (
  ts TEXT NOT NULL,
  starting REAL NOT NULL,
  open_risk REAL NOT NULL,
  realized REAL NOT NULL,
  current REAL NOT NULL
);
"""

def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db() -> None:
    with connect() as con:
        con.executescript(SCHEMA)
        cur = con.execute("SELECT value FROM meta WHERE key='starting_bankroll'")
        if cur.fetchone() is None:
            con.execute("INSERT INTO meta(key, value) VALUES('starting_bankroll', '2000')")
        con.commit()
        record_snapshot()

def get_meta(key: str, default: Optional[str] = None) -> str:
    with connect() as con:
        cur = con.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else default

def set_meta(key: str, value: str) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO meta(key,value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        con.commit()

def bankroll_snapshot() -> tuple[float, float, float]:
    start = float(get_meta("starting_bankroll", "2000") or 2000)
    with connect() as con:
        open_risk = con.execute(
            "SELECT SUM(stake) FROM bets WHERE result IS NULL"
        ).fetchone()[0] or 0.0
        realized = con.execute(
            "SELECT SUM(pnl) FROM bets WHERE result IS NOT NULL"
        ).fetchone()[0] or 0.0
    return start, open_risk, realized

def record_snapshot() -> None:
    start, open_risk, realized = bankroll_snapshot()
    current = start - open_risk + realized
    with connect() as con:
        con.execute(
            "INSERT INTO bankroll_history(ts, starting, open_risk, realized, current) VALUES(?,?,?,?,?)",
            (datetime.utcnow().isoformat(), start, open_risk, realized, current),
        )
        con.commit()

def log_bet(ts, event_id, league, selection, book, american_odds, decimal_odds, stake, closing_odds=None) -> int:
    with connect() as con:
        cur = con.execute(
            """
            INSERT INTO bets(ts,event_id,league,selection,book,american_odds,decimal_odds,stake,closing_odds)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (ts, event_id, league, selection, book, american_odds, decimal_odds, stake, closing_odds),
        )
        con.commit()
        bet_id = int(cur.lastrowid)
    record_snapshot()
    return bet_id

def settle_bet(bet_id: int, result: str) -> None:
    assert result in {"WIN", "LOSE", "PUSH"}
    with connect() as con:
        dec, stake = con.execute(
            "SELECT decimal_odds, stake FROM bets WHERE id=?", (bet_id,)
        ).fetchone()
        pnl = stake * (dec - 1) if result == "WIN" else (-stake if result == "LOSE" else 0.0)
        con.execute("UPDATE bets SET result=?, pnl=? WHERE id=?", (result, pnl, bet_id))
        con.commit()
    record_snapshot()

def list_bets() -> list[tuple]:
    with connect() as con:
        cur = con.execute(
            "SELECT id, ts, event_id, league, selection, book, american_odds, decimal_odds, stake, result, pnl "
            "FROM bets ORDER BY ts DESC, id DESC"
        )
        return cur.fetchall()

def history_rows() -> list[tuple]:
    with connect() as con:
        cur = con.execute(
            "SELECT ts, starting, open_risk, realized, current FROM bankroll_history ORDER BY ts ASC"
        )
        return cur.fetchall()

def reset_all(confirm: bool = False) -> None:
    if not confirm:
        return
    with connect() as con:
        con.execute("DELETE FROM bets")
        con.execute("DELETE FROM bankroll_history")
        con.commit()
    record_snapshot()
