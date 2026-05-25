"""Match-level feature engineering.

Inputs: a DataFrame of matches sorted chronologically with columns:
  date, home, away, home_goals, away_goals, competition, neutral, is_international
Plus Elo and Pi-rating columns already attached by their respective engines.

Outputs: same DataFrame with rolling form features (3/5/10-game windows),
streaks, rest days, head-to-head record, and target label.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

WINDOWS = [3, 5, 10]
XG_WINDOWS = [5, 10]
HISTORY_LEN = max(WINDOWS + XG_WINDOWS)


def _rolling_and_streaks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    has_xg = "home_xg" in df.columns and "away_xg" in df.columns

    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    xg_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    last_played: dict[str, pd.Timestamp] = {}
    streak: dict[str, int] = defaultdict(int)

    feats = []
    for row in df.itertuples(index=False):
        f: dict = {}
        for side, team in (("home", row.home), ("away", row.away)):
            hist = list(history[team])
            for w in WINDOWS:
                recent = hist[-w:]
                if recent:
                    f[f"{side}_form{w}_pts"] = float(np.mean([r[3] for r in recent]))
                    f[f"{side}_form{w}_gd"] = float(np.mean([r[1] - r[2] for r in recent]))
                else:
                    f[f"{side}_form{w}_pts"] = 1.5
                    f[f"{side}_form{w}_gd"] = 0.0
            f[f"{side}_streak"] = streak[team]
            # Rolling xG features (NaN when team has insufficient xG history)
            xg_hist = [x for x in xg_history[team] if x[0] is not None and x[1] is not None]
            for w in XG_WINDOWS:
                recent_xg = xg_hist[-w:]
                if len(recent_xg) >= max(2, w // 2):
                    f[f"{side}_xg_for{w}"] = float(np.mean([x[0] for x in recent_xg]))
                    f[f"{side}_xg_against{w}"] = float(np.mean([x[1] for x in recent_xg]))
                else:
                    f[f"{side}_xg_for{w}"] = np.nan
                    f[f"{side}_xg_against{w}"] = np.nan

        h_rest = (row.date - last_played[row.home]).days if row.home in last_played else 14
        a_rest = (row.date - last_played[row.away]).days if row.away in last_played else 14
        f["home_rest_days"] = min(h_rest, 30)
        f["away_rest_days"] = min(a_rest, 30)
        feats.append(f)

        # Update post-match
        hg, ag = int(row.home_goals), int(row.away_goals)
        h_pts = 3 if hg > ag else (1 if hg == ag else 0)
        a_pts = 3 if ag > hg else (1 if hg == ag else 0)
        history[row.home].append((row.date, hg, ag, h_pts))
        history[row.away].append((row.date, ag, hg, a_pts))
        if has_xg:
            hxg = getattr(row, "home_xg", None)
            axg = getattr(row, "away_xg", None)
            try:
                hxg = float(hxg) if hxg is not None and not pd.isna(hxg) else None
                axg = float(axg) if axg is not None and not pd.isna(axg) else None
            except (TypeError, ValueError):
                hxg = axg = None
            xg_history[row.home].append((hxg, axg))
            xg_history[row.away].append((axg, hxg))
        last_played[row.home] = row.date
        last_played[row.away] = row.date

        # Streaks
        if hg > ag:
            streak[row.home] = streak[row.home] + 1 if streak[row.home] > 0 else 1
            streak[row.away] = streak[row.away] - 1 if streak[row.away] < 0 else -1
        elif hg < ag:
            streak[row.home] = streak[row.home] - 1 if streak[row.home] < 0 else -1
            streak[row.away] = streak[row.away] + 1 if streak[row.away] > 0 else 1
        else:
            streak[row.home] = 0
            streak[row.away] = 0

    return pd.concat([df, pd.DataFrame(feats)], axis=1)


def _h2h(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    hist: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=window))
    h2h_pts, h2h_gd, h2h_n = [], [], []
    for row in df.itertuples(index=False):
        key = tuple(sorted([row.home, row.away]))
        records = list(hist[key])
        if records:
            pts = np.mean([r["home_pts"] if r["home"] == row.home else r["away_pts"]
                           for r in records])
            gd = np.mean([(r["hg"] - r["ag"]) if r["home"] == row.home else (r["ag"] - r["hg"])
                          for r in records])
            n = len(records)
        else:
            pts, gd, n = 1.0, 0.0, 0
        h2h_pts.append(pts); h2h_gd.append(gd); h2h_n.append(n)
        hg, ag = int(row.home_goals), int(row.away_goals)
        h_pts = 3 if hg > ag else (1 if hg == ag else 0)
        a_pts = 3 if ag > hg else (1 if hg == ag else 0)
        hist[key].append(dict(home=row.home, away=row.away, hg=hg, ag=ag,
                              home_pts=h_pts, away_pts=a_pts))
    df["h2h_home_pts"] = h2h_pts
    df["h2h_home_gd"] = h2h_gd
    df["h2h_n"] = h2h_n
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature transforms. Expects Elo + Pi columns already attached."""
    df = _rolling_and_streaks(df)
    df = _h2h(df)
    df["outcome"] = np.where(df["home_goals"] > df["away_goals"], 0,
                    np.where(df["home_goals"] < df["away_goals"], 2, 1))  # 0=H 1=D 2=A
    return df


FEATURE_COLS = [
    # Elo
    "elo_home_pre", "elo_away_pre", "elo_diff",
    # Pi-ratings
    "pi_home_avg", "pi_away_avg", "pi_home_home", "pi_away_away", "pi_expected_gd",
    # Multi-window form
    "home_form3_pts", "home_form3_gd",
    "home_form5_pts", "home_form5_gd",
    "home_form10_pts", "home_form10_gd",
    "away_form3_pts", "away_form3_gd",
    "away_form5_pts", "away_form5_gd",
    "away_form10_pts", "away_form10_gd",
    # Streaks (signed)
    "home_streak", "away_streak",
    # Rolling xG (NaN where unavailable; tree models handle missing)
    "home_xg_for5", "home_xg_against5",
    "home_xg_for10", "home_xg_against10",
    "away_xg_for5", "away_xg_against5",
    "away_xg_for10", "away_xg_against10",
    # Rest, H2H, context
    "home_rest_days", "away_rest_days",
    "h2h_home_pts", "h2h_home_gd", "h2h_n",
    "is_international", "neutral",
]
