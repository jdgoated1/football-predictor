"""High-level prediction API.

Pipeline:
    Elo + Pi-rating + multi-window form features  ->
       (Dixon-Coles outcome probs, XGBoost probs, LightGBM probs)  ->
       Logistic-regression META learner  ->  final outcome probabilities
    +
    Dixon-Coles scoreline matrix rescaled to match the final outcome probs
    -> top scorelines, most-likely score, expected goals
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .dixon_coles import DixonColes
from .elo import EloEngine
from .features import FEATURE_COLS, WINDOWS, XG_WINDOWS
from .pi_rating import PiRating

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


@dataclass
class PredictorBundle:
    name: str
    elo: EloEngine
    pi: PiRating
    dc: DixonColes
    xgb: object | None = None
    lgb: object | None = None
    cat: object | None = None
    meta: object | None = None
    last_seen: dict[str, pd.Timestamp] = field(default_factory=dict)
    last_form: dict[str, dict] = field(default_factory=dict)
    last_streak: dict[str, int] = field(default_factory=dict)
    last_h2h: dict[tuple, dict] = field(default_factory=dict)
    teams: list[str] = field(default_factory=list)
    metrics: dict | None = None       # report dicts from training

    def save(self) -> Path:
        MODELS.mkdir(parents=True, exist_ok=True)
        p = MODELS / f"{self.name}.joblib"
        joblib.dump(self, p)
        return p

    @classmethod
    def load(cls, name: str) -> "PredictorBundle":
        return joblib.load(MODELS / f"{name}.joblib")

    # ---------- low-level feature-row builder ----------

    def _form_default(self) -> dict:
        d = {}
        for w in WINDOWS:
            d[f"form{w}_pts"] = 1.5
            d[f"form{w}_gd"] = 0.0
        for w in XG_WINDOWS:
            d[f"xg_for{w}"] = np.nan
            d[f"xg_against{w}"] = np.nan
        return d

    def _form_for(self, team: str) -> dict:
        return self.last_form.get(team, self._form_default())

    def _rest_for(self, team: str, ref_date: pd.Timestamp) -> int:
        last = self.last_seen.get(team)
        if last is None:
            return 14
        return min((ref_date - last).days, 30)

    def _h2h_for(self, home: str, away: str) -> dict:
        key = tuple(sorted([home, away]))
        rec = self.last_h2h.get(key)
        if rec is None:
            return {"h2h_home_pts": 1.0, "h2h_home_gd": 0.0, "h2h_n": 0}
        first = key[0]
        pts = rec["pts_first"] if home == first else rec["pts_second"]
        gd = rec["gd_first"] if home == first else -rec["gd_first"]
        return {"h2h_home_pts": pts, "h2h_home_gd": gd, "h2h_n": rec["n"]}

    def feature_row(self, home: str, away: str, ref_date: pd.Timestamp,
                    neutral: bool = False) -> pd.DataFrame:
        elo_h = self.elo.rating(home)
        elo_a = self.elo.rating(away)
        pi_h_avg = 0.5 * (self.pi.get_h(home) + self.pi.get_a(home))
        pi_a_avg = 0.5 * (self.pi.get_h(away) + self.pi.get_a(away))
        h_form = self._form_for(home)
        a_form = self._form_for(away)
        h2h = self._h2h_for(home, away)
        row = {
            "elo_home_pre": elo_h, "elo_away_pre": elo_a, "elo_diff": elo_h - elo_a,
            "pi_home_avg": pi_h_avg, "pi_away_avg": pi_a_avg,
            "pi_home_home": self.pi.get_h(home),
            "pi_away_away": self.pi.get_a(away),
            "pi_expected_gd": self.pi.expected_gd(home, away, neutral=neutral),
            "home_rest_days": self._rest_for(home, ref_date),
            "away_rest_days": self._rest_for(away, ref_date),
            "h2h_home_pts": h2h["h2h_home_pts"], "h2h_home_gd": h2h["h2h_home_gd"],
            "h2h_n": h2h["h2h_n"],
            "home_streak": self.last_streak.get(home, 0),
            "away_streak": self.last_streak.get(away, 0),
            "is_international": int(self.name == "internationals"),
            "neutral": int(neutral),
        }
        for w in WINDOWS:
            row[f"home_form{w}_pts"] = h_form[f"form{w}_pts"]
            row[f"home_form{w}_gd"] = h_form[f"form{w}_gd"]
            row[f"away_form{w}_pts"] = a_form[f"form{w}_pts"]
            row[f"away_form{w}_gd"] = a_form[f"form{w}_gd"]
        for w in XG_WINDOWS:
            row[f"home_xg_for{w}"] = h_form.get(f"xg_for{w}", np.nan)
            row[f"home_xg_against{w}"] = h_form.get(f"xg_against{w}", np.nan)
            row[f"away_xg_for{w}"] = a_form.get(f"xg_for{w}", np.nan)
            row[f"away_xg_against{w}"] = a_form.get(f"xg_against{w}", np.nan)
        return pd.DataFrame([row], columns=FEATURE_COLS)

    # ---------- main predict ----------

    def predict(self, home: str, away: str, ref_date: pd.Timestamp | None = None,
                neutral: bool = False, max_goals: int = 8,
                blend: float = 1.0,
                odds: tuple[float, float, float] | None = None) -> dict:
        """`blend`: 0 = pure Dixon-Coles outcome probs, 1 = full stacked ensemble.
        `odds`: optional decimal bookmaker odds (home, draw, away) used as a
        meta-learner feature. When supplied, predictions tend to align more
        with market expectations."""
        if ref_date is None:
            ref_date = pd.Timestamp.today().normalize()

        # Dixon-Coles
        sm = self.dc.score_matrix(home, away, max_goals=max_goals, neutral=neutral)
        lam, mu = self.dc.expected_goals(home, away, neutral=neutral)
        dc_probs = {
            "H": float(np.tril(sm, -1).sum()),
            "D": float(np.trace(sm)),
            "A": float(np.triu(sm, 1).sum()),
        }
        dc_arr = np.array([dc_probs["H"], dc_probs["D"], dc_probs["A"]])

        # Convert optional odds to probabilities
        if odds is not None:
            try:
                inv = np.array([1.0 / float(odds[0]), 1.0 / float(odds[1]), 1.0 / float(odds[2])])
                odds_probs_arr = inv / inv.sum()
                has_odds = 1.0
            except Exception:
                odds_probs_arr = np.array([1/3, 1/3, 1/3])
                has_odds = 0.0
        else:
            odds_probs_arr = np.array([1/3, 1/3, 1/3])
            has_odds = 0.0

        xgb_probs = lgb_probs = cat_probs = stacked = None
        if (self.xgb is not None and self.lgb is not None and self.cat is not None
                and self.meta is not None):
            row = self.feature_row(home, away, ref_date, neutral=neutral)
            try:
                xgb_p = self.xgb.predict_proba(row)[0]
                lgb_p = self.lgb.predict_proba(row)[0]
                cat_p = self.cat.predict_proba(row.values)[0]
                meta_X = np.hstack([xgb_p, lgb_p, cat_p, dc_arr,
                                    odds_probs_arr, [has_odds]]).reshape(1, -1)
                stack_p = self.meta.predict_proba(meta_X)[0]
                xgb_probs = {"H": float(xgb_p[0]), "D": float(xgb_p[1]), "A": float(xgb_p[2])}
                lgb_probs = {"H": float(lgb_p[0]), "D": float(lgb_p[1]), "A": float(lgb_p[2])}
                cat_probs = {"H": float(cat_p[0]), "D": float(cat_p[1]), "A": float(cat_p[2])}
                stacked = {"H": float(stack_p[0]), "D": float(stack_p[1]), "A": float(stack_p[2])}
            except Exception:
                stacked = None

        # Blend Dixon-Coles outcome with the stacked ensemble
        if stacked is not None:
            blended = {k: (1 - blend) * dc_probs[k] + blend * stacked[k] for k in "HDA"}
            s = sum(blended.values())
            blended = {k: v / s for k, v in blended.items()}
        else:
            blended = dc_probs

        # Re-scale the scoreline matrix so each H/D/A region sums to the blended prob
        sm_adj = sm.copy()
        idx = np.indices(sm.shape)
        h_win = idx[0] > idx[1]
        draw = idx[0] == idx[1]
        a_win = idx[0] < idx[1]
        for region, target in [(h_win, blended["H"]), (draw, blended["D"]), (a_win, blended["A"])]:
            cur = sm_adj[region].sum()
            if cur > 1e-9:
                sm_adj[region] *= target / cur

        flat = [(int(i), int(j), float(sm_adj[i, j])) for i in range(sm_adj.shape[0])
                for j in range(sm_adj.shape[1])]
        flat.sort(key=lambda x: -x[2])
        top_scores = flat[:8]

        return {
            "home": home, "away": away, "neutral": neutral,
            "elo_home": self.elo.rating(home), "elo_away": self.elo.rating(away),
            "pi_home_h": self.pi.get_h(home), "pi_away_a": self.pi.get_a(away),
            "lambda_home": lam, "lambda_away": mu,
            "dc_outcome": dc_probs,
            "xgb_outcome": xgb_probs,
            "lgb_outcome": lgb_probs,
            "cat_outcome": cat_probs,
            "stacked_outcome": stacked,
            "outcome": blended,
            "score_matrix": sm_adj,
            "top_scores": top_scores,
            "most_likely": (top_scores[0][0], top_scores[0][1]),
        }


# ---------- post-training helpers ----------

def build_predict_state(elo: EloEngine, pi: PiRating, dc: DixonColes,
                        xgb_model, lgb_model, cat_model, meta_model,
                        matches: pd.DataFrame, name: str,
                        metrics: dict | None = None) -> PredictorBundle:
    """Walk matches once to capture each team's latest form/rest/streak/H2H state."""
    max_w = max(WINDOWS + XG_WINDOWS)
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_w))
    xg_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_w))
    last_seen: dict[str, pd.Timestamp] = {}
    last_streak: dict[str, int] = defaultdict(int)
    h2h: dict[tuple, dict] = {}
    has_xg_col = "home_xg" in matches.columns and "away_xg" in matches.columns

    for row in matches.sort_values("date").itertuples(index=False):
        hg, ag = int(row.home_goals), int(row.away_goals)
        h_pts = 3 if hg > ag else (1 if hg == ag else 0)
        a_pts = 3 if ag > hg else (1 if hg == ag else 0)
        history[row.home].append((hg, ag, h_pts))
        history[row.away].append((ag, hg, a_pts))
        if has_xg_col:
            hxg = getattr(row, "home_xg", None)
            axg = getattr(row, "away_xg", None)
            try:
                hxg = float(hxg) if hxg is not None and not pd.isna(hxg) else None
                axg = float(axg) if axg is not None and not pd.isna(axg) else None
            except (TypeError, ValueError):
                hxg = axg = None
            xg_history[row.home].append((hxg, axg))
            xg_history[row.away].append((axg, hxg))
        last_seen[row.home] = row.date
        last_seen[row.away] = row.date
        if hg > ag:
            last_streak[row.home] = last_streak[row.home] + 1 if last_streak[row.home] > 0 else 1
            last_streak[row.away] = last_streak[row.away] - 1 if last_streak[row.away] < 0 else -1
        elif hg < ag:
            last_streak[row.home] = last_streak[row.home] - 1 if last_streak[row.home] < 0 else -1
            last_streak[row.away] = last_streak[row.away] + 1 if last_streak[row.away] > 0 else 1
        else:
            last_streak[row.home] = 0
            last_streak[row.away] = 0
        key = tuple(sorted([row.home, row.away]))
        first = key[0]
        if key not in h2h:
            h2h[key] = {"pts_first": 1.0, "pts_second": 1.0, "gd_first": 0.0, "n": 0,
                        "records": deque(maxlen=6)}
        rec = h2h[key]
        pts_for_first = h_pts if row.home == first else a_pts
        pts_for_second = a_pts if row.home == first else h_pts
        gd_first = (hg - ag) if row.home == first else (ag - hg)
        rec["records"].append((pts_for_first, pts_for_second, gd_first))
        rec["n"] = len(rec["records"])
        rec["pts_first"] = float(np.mean([r[0] for r in rec["records"]]))
        rec["pts_second"] = float(np.mean([r[1] for r in rec["records"]]))
        rec["gd_first"] = float(np.mean([r[2] for r in rec["records"]]))

    last_form = {}
    for team, hist in history.items():
        records = list(hist)
        if records:
            form = {}
            for w in WINDOWS:
                recent = records[-w:]
                form[f"form{w}_pts"] = float(np.mean([r[2] for r in recent]))
                form[f"form{w}_gd"] = float(np.mean([r[0] - r[1] for r in recent]))
            # Rolling xG averages from last N games (where xG was recorded)
            xg_recs = [x for x in xg_history.get(team, []) if x[0] is not None and x[1] is not None]
            for w in XG_WINDOWS:
                recent_xg = xg_recs[-w:]
                if len(recent_xg) >= max(2, w // 2):
                    form[f"xg_for{w}"] = float(np.mean([x[0] for x in recent_xg]))
                    form[f"xg_against{w}"] = float(np.mean([x[1] for x in recent_xg]))
                else:
                    form[f"xg_for{w}"] = np.nan
                    form[f"xg_against{w}"] = np.nan
            last_form[team] = form

    teams = sorted(set(matches["home"]).union(matches["away"]))
    return PredictorBundle(
        name=name, elo=elo, pi=pi, dc=dc,
        xgb=xgb_model, lgb=lgb_model, cat=cat_model, meta=meta_model,
        last_seen=last_seen, last_form=last_form, last_streak=dict(last_streak),
        last_h2h=h2h, teams=teams, metrics=metrics,
    )
