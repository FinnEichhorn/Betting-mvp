# app/models/baseline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class BaselineConfig:
    """
    Lightweight, data-agnostic baseline model for sportsbook markets.
    Starts from market (vig-free) probability when available, applies tiny domain priors,
    then shrinks toward 50/50 to avoid overconfident EVs.

    Tunables are intentionally small so this model acts as a gentle prior rather than a
    predictive system that would require ratings/inputs.
    """
    # Shrinkage toward 50/50. Larger = more conservative.
    prior_strength: float = 6.0
    prior_center: float = 0.50

    # Domain priors (tiny nudges; keep small)
    home_edge: float = 0.02             # Moneyline: slight home advantage
    spread_home_edge: float = 0.005     # Spreads: even smaller home ATS lean
    totals_under_bias: float = 0.01     # Totals: slight lean to UNDER historically

    # Hard caps to keep probabilities sane
    min_p: float = 0.02
    max_p: float = 0.98

#baselineModel
    """
    A tiny baseline model that:
      1) Starts from vig-free market probability if provided (or 0.50 fallback)
      2) Applies small domain-specific priors (home ML, home ATS, UNDER totals)
      3) Shrinks toward 0.50 using a simple Beta-prior style update
      4) Clamps to [min_p, max_p]

    Supported market_type keys (case-insensitive):
      - "h2h" / "moneyline" / "match" / "winner"
      - "spreads" / "spread" / "ats"
      - "totals" / "total" / "over_under" / "o/u"
    Supported selection keys (case-insensitive):
      - H2H/SPREADS: "home", "away"
      - TOTALS: "over", "under"
      - (draw is not modeled here)
    """
# in app/models/baseline.py
import numpy as np

class BaselineModel:
    def __init__(self, cfg):
        self.cfg = cfg  # expects .prior_strength, .prior_center, .min_p, .max_p

    def predict(
        self,
        market_prob_vigfree: float | None,
        selection: str,
        market_type: str,
        point: float | None = None,
        **kwargs,  # future-proof: player, team, etc.
    ) -> float:
        # 1) start from prior
        p = float(self.cfg.prior_center)

        # 2) blend in vig-free market probability if present
        if market_prob_vigfree is not None:
            try:
                mkt = float(market_prob_vigfree)
                w = float(np.clip(self.cfg.prior_strength, 0.0, 1.0))  # 1.0 = stick to prior more
                p = w * self.cfg.prior_center + (1.0 - w) * mkt
            except Exception:
                pass  # keep prior if conversion fails

        # 3) for symmetric 2-way markets, flip if selection is the "other side"
        sel = (selection or "").strip().lower()
        if sel in {"under", "no", "away"}:
            p = 1.0 - p

        # 4) (optional) tiny line-based adjustment for spreads/totals if you want later
        # if market_type in {"spreads", "totals", "player_points"} and point is not None:
        #     p = logistic_shift(p, point)  # your future function

        # 5) clamp
        p = float(np.clip(p, self.cfg.min_p, self.cfg.max_p))
        return p

    
    def __init__(self, cfg: BaselineConfig | None = None) -> None:
        self.cfg = cfg or BaselineConfig()


# ---------- public API ----------

    def predict(
        self,
        market_prob_vigfree: Optional[float],
        selection: str,
        market_type: str,
        point: Optional[float] = None,
        **kwargs,   # <-- add this to swallow 'player', 'team', etc.
    ) -> float:
        # === your existing body ===
        p = float(self.cfg.prior_center)
        if market_prob_vigfree is not None:
            try:
                mkt = float(market_prob_vigfree)
                w = float(np.clip(self.cfg.prior_strength, 0.0, 1.0))
                p = w * self.cfg.prior_center + (1.0 - w) * mkt
            except Exception:
                pass

        sel = (selection or "").strip().lower()
        if sel in {"under", "no", "away"}:
            p = 1.0 - p

        p = float(np.clip(p, self.cfg.min_p, self.cfg.max_p))
        return p


    def predict_from_row(self, row: dict) -> float:
        """
        Convenience for DataFrame rows that follow your normalized schema:
          - row["market"] in {"h2h","spreads","totals"}
          - row["outcome"] in {"home","away"} or {"over","under"}
          - row.get("p_fair") = vig-free prob for THIS outcome (may be NaN/None)
          - row.get("point") = spread/total number (optional)

        Example usage:
            df["p_model"] = df.apply(baseline.predict_from_row, axis=1)
        """
        market_type = str(row.get("market", "")).lower()
        selection = str(row.get("outcome", "")).lower()

        p_fair = row.get("p_fair", None)
        if p_fair is not None:
            try:
                # Cope with pandas NaN
                if p_fair != p_fair:  # NaN check
                    p_fair = None
                else:
                    p_fair = float(p_fair)
            except Exception:
                p_fair = None

        point = row.get("point", None)
        try:
            point = float(point) if point is not None else None
        except Exception:
            point = None

        return self.predict(p_fair, selection, market_type, point=point)

    # ---------- internal helpers ----------

    def _normalize_keys(self, selection: str, market_type: str) -> tuple[str, str]:
        sel = str(selection).strip().lower()
        mkt = str(market_type).strip().lower()

        # Market aliases
        if mkt in ("moneyline", "match", "winner"):
            mkt = "h2h"
        elif mkt in ("spread", "ats"):
            mkt = "spreads"
        elif mkt in ("total", "over_under", "o/u", "o-u"):
            mkt = "totals"

        # Selection aliases
        if sel in ("home", "away", "draw"):
            pass
        elif sel in ("over", "under"):
            pass
        else:
            # Keep unknowns as-is; downstream bias map just won't add anything.
            sel = sel.lower()

        return sel, mkt

    def _start_from_market(self, p_market: Optional[float]) -> float:
        if p_market is None:
            return 0.50
        try:
            p = float(p_market)
        except Exception:
            p = 0.50
        return p

    def _apply_domain_bias(
        self,
        p: float,
        selection: str,
        market_type: str,
        *,
        point: Optional[float] = None,
    ) -> float:
        """
        Tiny domain priors:
          - H2H: +/- home_edge for HOME/AWAY
          - SPREADS: +/- spread_home_edge for HOME/AWAY
          - TOTALS: +/- totals_under_bias for UNDER/OVER
        'point' is accepted for future sophistication (e.g., decay by |point|), but not used now.
        """
        sel = selection.upper()
        mkt = market_type.upper()

        if mkt == "H2H":
            if sel == "HOME":
                p += self.cfg.home_edge
            elif sel == "AWAY":
                p -= self.cfg.home_edge

        elif mkt == "SPREADS":
            # Home ATS micro-lean; keep small to avoid overpowering market 50/50
            if sel == "HOME":
                p += self.cfg.spread_home_edge
            elif sel == "AWAY":
                p -= self.cfg.spread_home_edge

        elif mkt == "TOTALS":
            if sel == "UNDER":
                p += self.cfg.totals_under_bias
            elif sel == "OVER":
                p -= self.cfg.totals_under_bias

        return p

    def _shrink(self, p: float) -> float:
        """
        Simple Beta-prior shrinkage toward prior_center.
        Equivalent to (n*p_obs + s*prior) / (n + s) with n=1 here.
        """
        s = float(self.cfg.prior_strength)
        c = float(self.cfg.prior_center)
        return (p + s * c) / (1.0 + s)

    def _clamp(self, p: float) -> float:
        return max(self.cfg.min_p, min(self.cfg.max_p, p))
