"""Monte Carlo tournament simulator.

Two modes:
  * `simulate_league(bundle, teams, ...)` - double round-robin, returns probability
    table for final standings + expected points.
  * `simulate_knockout(bundle, fmt, teams_by_group, ...)` - groups + knockout
    bracket (World Cup / Euros / Copa / AFCON), returns stage-advancement
    probabilities.

Scorelines are sampled from the Dixon-Coles bivariate Poisson matrix. Knockout
draws are resolved with an Elo-weighted penalty shootout.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .predictor import PredictorBundle


# ---------- shared sampling ----------

def _precompute(bundle: PredictorBundle, teams: list[str], neutral: bool,
                max_goals: int = 7, fmt_name: str | None = None,
                wc_blend: float = 0.0) -> dict[tuple[str, str], np.ndarray]:
    """Cache the flattened scoreline distribution for every (home, away) pair.

    When `fmt_name` is WC 2026 and `wc_blend > 0`, the DC score matrix is
    rescaled so that H/D/A regions match a blend of (DC outcome) and
    (WC 2026 market prior). Sampling then naturally reflects the prior."""
    from .wc26_strength import fmt_uses_wc_prior, wc_outcome_prior
    apply_wc = wc_blend > 0 and fmt_uses_wc_prior(fmt_name)

    cache: dict[tuple[str, str], np.ndarray] = {}
    for h in teams:
        for a in teams:
            if h == a:
                continue
            m = bundle.dc.score_matrix(h, a, max_goals=max_goals, neutral=neutral)
            if apply_wc:
                prior = wc_outcome_prior(h, a)
                if prior is not None:
                    p_h_dc = float(np.tril(m, -1).sum())
                    p_d_dc = float(np.trace(m))
                    p_a_dc = float(np.triu(m, 1).sum())
                    target_h = (1 - wc_blend) * p_h_dc + wc_blend * prior["H"]
                    target_d = (1 - wc_blend) * p_d_dc + wc_blend * prior["D"]
                    target_a = (1 - wc_blend) * p_a_dc + wc_blend * prior["A"]
                    tot = target_h + target_d + target_a
                    target_h /= tot; target_d /= tot; target_a /= tot
                    idx = np.indices(m.shape)
                    if p_h_dc > 1e-9: m[idx[0] > idx[1]] *= target_h / p_h_dc
                    if p_d_dc > 1e-9: m[idx[0] == idx[1]] *= target_d / p_d_dc
                    if p_a_dc > 1e-9: m[idx[0] < idx[1]] *= target_a / p_a_dc
            flat = m.ravel()
            s = flat.sum()
            if s > 0:
                flat = flat / s
            cache[(h, a)] = flat
    return cache


def _sample_match(cache: dict, h: str, a: str, max_goals: int = 7,
                  rng: np.random.Generator | None = None) -> tuple[int, int]:
    rng = rng or np.random.default_rng()
    p = cache[(h, a)]
    idx = rng.choice(len(p), p=p)
    return divmod(idx, max_goals + 1)


def _penalty_winner(bundle: PredictorBundle, h: str, a: str,
                    rng: np.random.Generator) -> str:
    """Slight Elo lean - 100 pt diff ~ 53%; otherwise nearly 50/50."""
    diff = bundle.elo.rating(h) - bundle.elo.rating(a)
    p_h = 1.0 / (1.0 + 10.0 ** (-diff / 800.0))
    return h if rng.random() < p_h else a


# ---------- league simulation ----------

@dataclass
class StartingState:
    """Mid-season starting state for a league simulation."""
    pts: dict[str, int]
    gf: dict[str, int]
    ga: dict[str, int]
    played_pairs: set[tuple[str, str]]   # (home, away)
    teams: list[str]


def current_state_for(league: str, processed_dir: Path | None = None) -> StartingState | None:
    """Read the parquet and reconstruct the current season's standings + played pairs."""
    processed_dir = processed_dir or (Path(__file__).resolve().parent.parent / "data" / "processed")
    p = processed_dir / "leagues.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = df[df["competition"] == league]
    if df.empty:
        return None
    last = df["date"].max()
    season_start_year = last.year if last.month >= 8 else last.year - 1
    season_start = pd.Timestamp(year=season_start_year, month=7, day=1)
    df = df[df["date"] >= season_start].copy()
    teams = sorted(set(df["home"]).union(df["away"]))
    pts = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    played = set()
    for row in df.itertuples(index=False):
        h, a = row.home, row.away
        hg, ag = int(row.home_goals), int(row.away_goals)
        gf[h] += hg; ga[h] += ag
        gf[a] += ag; ga[a] += hg
        if hg > ag:
            pts[h] += 3
        elif hg < ag:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1
        played.add((h, a))
    return StartingState(pts=pts, gf=gf, ga=ga, played_pairs=played, teams=teams)


def simulate_league(bundle: PredictorBundle, teams: list[str], n_sims: int = 3000,
                    start: StartingState | None = None,
                    max_goals: int = 7, seed: int | None = None) -> pd.DataFrame:
    """Simulate a (continuation of a) league season.

    If `start` is None: simulate a full double round-robin from 0-0 standings.
    If `start` is given: only simulate the unplayed fixtures, on top of current standings.
    """
    rng = np.random.default_rng(seed)
    teams = sorted(teams)
    cache = _precompute(bundle, teams, neutral=False, max_goals=max_goals)

    # Build the list of fixtures to simulate
    all_pairs = [(h, a) for h in teams for a in teams if h != a]
    if start is not None:
        remaining = [(h, a) for (h, a) in all_pairs if (h, a) not in start.played_pairs]
        base_pts = start.pts
        base_gf = start.gf
        base_ga = start.ga
    else:
        remaining = all_pairs
        base_pts = {t: 0 for t in teams}
        base_gf = {t: 0 for t in teams}
        base_ga = {t: 0 for t in teams}

    n = len(teams)
    pos_counts = np.zeros((n, n), dtype=np.int32)
    pts_sum = np.zeros(n, dtype=np.float64)
    win_counts = np.zeros(n, dtype=np.int32)
    team_idx = {t: i for i, t in enumerate(teams)}

    for _ in range(n_sims):
        pts = base_pts.copy()
        gf = base_gf.copy()
        ga = base_ga.copy()
        for (h, a) in remaining:
            p = cache[(h, a)]
            idx = rng.choice(len(p), p=p)
            hg, ag = divmod(idx, max_goals + 1)
            gf[h] += hg; ga[h] += ag
            gf[a] += ag; ga[a] += hg
            if hg > ag:
                pts[h] += 3
            elif hg < ag:
                pts[a] += 3
            else:
                pts[h] += 1; pts[a] += 1
        # Rank by (pts desc, gd desc, gf desc)
        ranked = sorted(teams, key=lambda t: (-pts[t], -(gf[t] - ga[t]), -gf[t]))
        for pos, t in enumerate(ranked):
            pos_counts[team_idx[t], pos] += 1
            pts_sum[team_idx[t]] += pts[t]
        win_counts[team_idx[ranked[0]]] += 1

    df = pd.DataFrame({"Team": teams})
    df["Win %"]     = pos_counts[:, 0] / n_sims * 100
    df["Top 4 %"]   = pos_counts[:, :4].sum(axis=1) / n_sims * 100
    df["Top 6 %"]   = pos_counts[:, :6].sum(axis=1) / n_sims * 100
    df["Bottom 3 %"] = pos_counts[:, -3:].sum(axis=1) / n_sims * 100
    df["Expected pts"] = pts_sum / n_sims
    df["Expected pos"] = (pos_counts * np.arange(1, n + 1)).sum(axis=1) / n_sims
    return df.sort_values("Expected pts", ascending=False).reset_index(drop=True)


# ---------- knockout tournament simulation ----------

@dataclass
class TournamentFormat:
    name: str
    n_groups: int
    per_group: int = 4
    advance_per_group: int = 2
    best_thirds: int = 0     # how many best 3rd-place teams also advance

    @property
    def n_teams(self) -> int:
        return self.n_groups * self.per_group

    @property
    def n_knockout(self) -> int:
        return self.n_groups * self.advance_per_group + self.best_thirds


FORMATS: dict[str, TournamentFormat] = {
    "World Cup 2026 (48 teams)": TournamentFormat("World Cup 2026", n_groups=12, best_thirds=8),
    "World Cup 2022 (32 teams)": TournamentFormat("World Cup 2022", n_groups=8),
    "Euro 2024 (24 teams)":      TournamentFormat("Euro 2024", n_groups=6, best_thirds=4),
    "AFCON (24 teams)":          TournamentFormat("AFCON", n_groups=6, best_thirds=4),
    "Copa America (16 teams)":   TournamentFormat("Copa America", n_groups=4),
}


# ---------- real knockout bracket pairings ----------
#
# First-round pairings are listed in bracket-tree order so subsequent rounds can
# pair winners sequentially (winner[0] vs winner[1], winner[2] vs winner[3], ...)
# and the bracket structure is preserved through to the final.
#
# Qualifier ordering for groups-of-4 with top-2 advancing:
#     [G1_1, G1_2, G2_1, G2_2, ..., Gn_1, Gn_2, T1, T2, ...]
# where T are best-third qualifiers (when applicable).
BRACKETS: dict[str, list[tuple[int, int]]] = {
    # WC 2022: 8 groups, 16 qualifiers, R16 first round
    "World Cup 2022 (32 teams)": [
        (0, 3),    # 1A vs 2B
        (4, 7),    # 1C vs 2D
        (2, 1),    # 1B vs 2A
        (6, 5),    # 1D vs 2C
        (8, 11),   # 1E vs 2F
        (12, 15),  # 1G vs 2H
        (10, 9),   # 1F vs 2E
        (14, 13),  # 1H vs 2G
    ],
    # Copa America 2024: 4 groups, 8 qualifiers, QF first round
    "Copa America (16 teams)": [
        (0, 3),  # 1A vs 2B
        (2, 1),  # 1B vs 2A
        (4, 7),  # 1C vs 2D
        (6, 5),  # 1D vs 2C
    ],
}


def _first_round_pairs(fmt_name: str, n_qualifiers: int) -> list[tuple[int, int]]:
    if fmt_name in BRACKETS and 2 * len(BRACKETS[fmt_name]) == n_qualifiers:
        return BRACKETS[fmt_name]
    return [(i, i + 1) for i in range(0, n_qualifiers, 2)]


def _simulate_group(bundle, group_teams: list[str], cache, rng,
                    max_goals: int) -> dict[str, dict]:
    pts = {t: 0 for t in group_teams}
    gf = {t: 0 for t in group_teams}
    ga = {t: 0 for t in group_teams}
    for i, h in enumerate(group_teams):
        for j, a in enumerate(group_teams):
            if i == j:
                continue
            # Only schedule each pair once - pick whichever order is in cache (both are)
            if j < i:
                continue
            # Single match (tournament group stage = single fixture, not home/away)
            p = cache[(h, a)]
            idx = rng.choice(len(p), p=p)
            hg, ag = divmod(idx, max_goals + 1)
            gf[h] += hg; ga[h] += ag
            gf[a] += ag; ga[a] += hg
            if hg > ag:
                pts[h] += 3
            elif hg < ag:
                pts[a] += 3
            else:
                pts[h] += 1; pts[a] += 1
    return {t: {"pts": pts[t], "gd": gf[t] - ga[t], "gf": gf[t]} for t in group_teams}


def _rank(group_stats: dict) -> list[str]:
    return sorted(group_stats, key=lambda t: (-group_stats[t]["pts"],
                                              -group_stats[t]["gd"],
                                              -group_stats[t]["gf"]))


def simulate_knockout(bundle: PredictorBundle, fmt: TournamentFormat,
                      teams_by_group: list[list[str]], n_sims: int = 3000,
                      max_goals: int = 7, seed: int | None = None,
                      fmt_name: str | None = None,
                      wc_blend: float = 0.0) -> pd.DataFrame:
    """Simulate a group-stage + knockout tournament n_sims times."""
    if len(teams_by_group) != fmt.n_groups:
        raise ValueError(f"Need {fmt.n_groups} groups, got {len(teams_by_group)}")
    all_teams = sorted({t for g in teams_by_group for t in g})
    cache = _precompute(bundle, all_teams, neutral=True, max_goals=max_goals,
                        fmt_name=fmt_name, wc_blend=wc_blend)
    rng = np.random.default_rng(seed)

    # Determine knockout round names
    n_ko = fmt.n_knockout
    round_names = []
    n = n_ko
    while n >= 2:
        if n == 2:
            round_names.append("Final")
        elif n == 4:
            round_names.append("Semi")
        elif n == 8:
            round_names.append("Quarter")
        elif n == 16:
            round_names.append("R16")
        elif n == 32:
            round_names.append("R32")
        else:
            round_names.append(f"R{n}")
        n //= 2
    # Plus "Group" stage for everyone who started
    stages = ["Group", *round_names, "Champion"]

    counts = {t: defaultdict(int) for t in all_teams}

    for _ in range(n_sims):
        for t in all_teams:
            counts[t]["Group"] += 1
        # --- group stage ---
        qualified: list[str] = []
        thirds: list[tuple[str, dict]] = []
        for group in teams_by_group:
            stats = _simulate_group(bundle, group, cache, rng, max_goals)
            ranked = _rank(stats)
            for t in ranked[:fmt.advance_per_group]:
                qualified.append(t)
            if fmt.best_thirds > 0 and len(ranked) > fmt.advance_per_group:
                third = ranked[fmt.advance_per_group]
                thirds.append((third, stats[third]))
        if fmt.best_thirds > 0:
            thirds.sort(key=lambda kv: (-kv[1]["pts"], -kv[1]["gd"], -kv[1]["gf"]))
            for t, _s in thirds[:fmt.best_thirds]:
                qualified.append(t)
        # --- knockout: use real bracket if available, else sequential ---
        survivors = list(qualified)
        first_pairs = _first_round_pairs(fmt_name or fmt.name, len(survivors))
        for round_idx, rnd_name in enumerate(round_names):
            for t in survivors:
                counts[t][rnd_name] += 1
            next_survivors = []
            if round_idx == 0:
                pairings = [(survivors[i], survivors[j]) for (i, j) in first_pairs]
            else:
                pairings = [(survivors[i], survivors[i + 1]) for i in range(0, len(survivors), 2)]
            for (h, a) in pairings:
                p = cache[(h, a)]
                idx = rng.choice(len(p), p=p)
                hg, ag = divmod(idx, max_goals + 1)
                if hg > ag:
                    winner = h
                elif hg < ag:
                    winner = a
                else:
                    winner = _penalty_winner(bundle, h, a, rng)
                next_survivors.append(winner)
            survivors = next_survivors
        for t in survivors:
            counts[t]["Champion"] += 1

    rows = []
    for t in all_teams:
        row = {"Team": t}
        for stage in stages:
            row[stage + " %"] = counts[t][stage] / n_sims * 100
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("Champion %", ascending=False).reset_index(drop=True)


def top_n_by_elo(bundle: PredictorBundle, n: int) -> list[str]:
    return [t for t, _ in sorted(bundle.elo.ratings.items(), key=lambda kv: -kv[1])[:n]]


# ---------- deterministic "modal path" predictions ----------

def best_pick_score(pred: dict) -> tuple[int, int, float]:
    """Legacy: most-likely scoreline within most-likely result region.
    Kept for backward compatibility; `best_ev_score` is strictly better."""
    outcome = pred["outcome"]
    sm = pred["score_matrix"]
    if outcome["H"] >= outcome["D"] and outcome["H"] >= outcome["A"]:
        valid = lambda i, j: i > j
    elif outcome["A"] >= outcome["D"] and outcome["A"] >= outcome["H"]:
        valid = lambda i, j: i < j
    else:
        valid = lambda i, j: i == j
    best = (1, 1, 0.0)
    for i in range(sm.shape[0]):
        for j in range(sm.shape[1]):
            if valid(i, j) and sm[i, j] > best[2]:
                best = (i, j, float(sm[i, j]))
    return best


def get_actual_results(matches_df: pd.DataFrame,
                       start_date: str = "2026-06-11",
                       end_date: str = "2026-07-19") -> dict[tuple[str, str], tuple[int, int]]:
    """Return {(home, away): (home_goals, away_goals)} for played matches in the
    tournament's date window. Used to override predictions with actual results
    as the tournament progresses (e.g. after the daily auto-update fetches them).
    """
    if matches_df is None or len(matches_df) == 0:
        return {}
    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    df = matches_df[(matches_df["date"] >= s) & (matches_df["date"] <= e)].copy()
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for r in df.itertuples(index=False):
        try:
            out[(r.home, r.away)] = (int(r.home_goals), int(r.away_goals))
        except (AttributeError, TypeError, ValueError):
            continue
    return out


def best_ev_score(pred: dict, exact_pts: float = 3.0, result_pts: float = 1.0,
                  gd_pts: float = 0.0, draw_bonus: float = 0.0) -> tuple[int, int, float, float]:
    """Pick the scoreline that maximises expected points under a school
    competition scoring scheme.

    Scoring (Premier-League-Predictor style):
      exact_pts: points if the EXACT score is matched (default 3)
      result_pts: points if the result (W/D/L) is correct (default 1)
      gd_pts: bonus if goal difference matches but not exact (default 0)

    Expected points for picking (i, j) for a match with result R:
        EV = exact_pts × P(i,j)                           (exact score hit)
           + result_pts × (P(R) - P(i,j))                 (other scores with same result)
           + gd_pts × (P(|home-away|=|i-j|) - P(i,j))     (other scores with same GD)
    The gd_pts term excludes the exact-score case (already counted).

    Returns (home_goals, away_goals, P_exact, EV).
    """
    outcome = pred["outcome"]
    sm = pred["score_matrix"]
    n_rows, n_cols = sm.shape
    # Pre-compute GD totals (sum over cells with same |i-j|)
    gd_totals: dict[int, float] = {}
    for i in range(n_rows):
        for j in range(n_cols):
            gd_totals[i - j] = gd_totals.get(i - j, 0.0) + float(sm[i, j])

    best = (1, 1, 0.0, -1.0)
    for i in range(n_rows):
        for j in range(n_cols):
            p_exact = float(sm[i, j])
            if i > j:
                p_result = outcome["H"]
            elif i < j:
                p_result = outcome["A"]
            else:
                p_result = outcome["D"]
            p_same_gd = gd_totals.get(i - j, 0.0)
            ev = (exact_pts * p_exact
                  + result_pts * (p_result - p_exact)
                  + gd_pts * (p_same_gd - p_exact))
            # Optional draw bonus: bias picks toward draws to match real-world
            # draw frequency (~25-30% of WC games). Pure EV gives ~0 draws because
            # one team is always slightly favoured; this term gently corrects that.
            if i == j and draw_bonus > 0:
                ev += draw_bonus
            if ev > best[3]:
                best = (i, j, p_exact, ev)
    return best


def _round_name(n: int) -> str:
    return {2: "Final", 4: "Semi-final", 8: "Quarter-final",
            16: "Round of 16", 32: "Round of 32"}.get(n, f"Round of {n}")


def _matchday_pairs(group: list[str]) -> list[list[tuple[str, str]]]:
    """Standard FIFA 4-team round-robin matchday schedule.
    Each team plays exactly once per matchday. Position 1 is pot 1 (typically
    seeded / host), position 4 is the lowest pot."""
    if len(group) == 4:
        t1, t2, t3, t4 = group
        return [
            [(t1, t2), (t3, t4)],     # Matchday 1
            [(t1, t3), (t4, t2)],     # Matchday 2
            [(t4, t1), (t2, t3)],     # Matchday 3
        ]
    # Fallback for non-4-team groups: dump everything into one matchday
    pairs = [(group[i], group[j]) for i in range(len(group)) for j in range(i + 1, len(group))]
    return [pairs]


def predict_group_fixtures(bundle: PredictorBundle,
                           groups: list[list[str]],
                           fmt_name: str | None = None,
                           wc_blend: float = 0.0,
                           exact_pts: float = 3.0,
                           result_pts: float = 1.0,
                           gd_pts: float = 0.0,
                           draw_bonus: float = 0.0,
                           actual_results: dict | None = None) -> list[dict]:
    """For every group fixture, return the model's prediction.

    If `fmt_name` matches a tournament in `schedules.SCHEDULES`, the real
    published fixture list (with actual dates) is used. If `fmt_name` is the
    WC 2026 and `wc_blend > 0`, the WC 2026 squad-strength prior is blended
    into each prediction.
    """
    from .real_groups import ALIASES
    from .schedules import get_schedule
    from .wc26_strength import apply_wc_prior_to_prediction, fmt_uses_wc_prior

    actual_results = actual_results or {}

    def _do_predict(h: str, a: str) -> dict:
        pred = bundle.predict(h, a, neutral=True)
        if wc_blend > 0 and fmt_uses_wc_prior(fmt_name):
            pred = apply_wc_prior_to_prediction(pred, h, a, blend=wc_blend)
        return pred

    def _alt_scores(pred: dict, exclude: tuple[int, int]) -> str:
        """Top 3 alternative scorelines (excluding the primary). Each is tagged
        with the result it implies (H/D/A) so the user can see why they weren't
        picked as primary - alts from a non-favoured result lose the result
        point even if their raw probability matches the primary."""
        tops = pred.get("top_scores", [])
        parts = []
        for hg, ag, p in tops:
            if (hg, ag) == exclude:
                continue
            if hg > ag:
                tag = "H"
            elif hg < ag:
                tag = "A"
            else:
                tag = "D"
            parts.append(f"{hg}-{ag} ({p*100:.0f}% {tag})")
            if len(parts) >= 3:
                break
        return ", ".join(parts)

    schedule = get_schedule(fmt_name) if fmt_name else None
    if schedule and "group_fixtures" in schedule:
        team_to_group = {t: gi for gi, group in enumerate(groups) for t in group}
        out = []
        per_team_count: dict[str, int] = {}
        for date, raw_h, raw_a in schedule["group_fixtures"]:
            h = ALIASES.get(raw_h, raw_h)
            a = ALIASES.get(raw_a, raw_a)
            if h not in team_to_group or a not in team_to_group:
                continue
            md = max(per_team_count.get(h, 0), per_team_count.get(a, 0)) + 1
            per_team_count[h] = per_team_count.get(h, 0) + 1
            per_team_count[a] = per_team_count.get(a, 0) + 1
            # Use actual result if the match has been played
            if (h, a) in actual_results:
                act_h, act_a = actual_results[(h, a)]
                out.append({
                    "date": date,
                    "group_idx": team_to_group[h], "matchday": md,
                    "home": h, "away": a,
                    "score": (act_h, act_a),
                    "score_prob": 1.0,
                    "alt_scores": "",
                    "p_home": 1.0 if act_h > act_a else 0.0,
                    "p_draw": 1.0 if act_h == act_a else 0.0,
                    "p_away": 1.0 if act_h < act_a else 0.0,
                    "xg_home": float(act_h), "xg_away": float(act_a),
                    "is_actual": True,
                })
                continue
            pred = _do_predict(h, a)
            best_hg, best_ag, best_p, _ev = best_ev_score(
                pred, exact_pts=exact_pts, result_pts=result_pts, gd_pts=gd_pts,
                draw_bonus=draw_bonus,
            )
            out.append({
                "date": date,
                "group_idx": team_to_group[h], "matchday": md,
                "home": h, "away": a,
                "score": (best_hg, best_ag),
                "score_prob": best_p,
                "alt_scores": _alt_scores(pred, exclude=(best_hg, best_ag)),
                "p_home": pred["outcome"]["H"],
                "p_draw": pred["outcome"]["D"],
                "p_away": pred["outcome"]["A"],
                "xg_home": pred["lambda_home"],
                "xg_away": pred["lambda_away"],
                "is_actual": False,
            })
        out.sort(key=lambda x: (x["date"], x["group_idx"]))
        return out

    # ---- Fallback: generated round-robin matchday pattern ----
    out = []
    for gi, group in enumerate(groups):
        for md_idx, md_pairs in enumerate(_matchday_pairs(group)):
            for (h, a) in md_pairs:
                if (h, a) in actual_results:
                    act_h, act_a = actual_results[(h, a)]
                    out.append({
                        "group_idx": gi, "matchday": md_idx + 1,
                        "home": h, "away": a,
                        "score": (act_h, act_a), "score_prob": 1.0, "alt_scores": "",
                        "p_home": 1.0 if act_h > act_a else 0.0,
                        "p_draw": 1.0 if act_h == act_a else 0.0,
                        "p_away": 1.0 if act_h < act_a else 0.0,
                        "xg_home": float(act_h), "xg_away": float(act_a),
                        "is_actual": True,
                    })
                    continue
                pred = _do_predict(h, a)
                best_hg, best_ag, best_p, _ev = best_ev_score(
                    pred, exact_pts=exact_pts, result_pts=result_pts, gd_pts=gd_pts,
                    draw_bonus=draw_bonus,
                )
                out.append({
                    "group_idx": gi, "matchday": md_idx + 1,
                    "home": h, "away": a,
                    "score": (best_hg, best_ag),
                    "score_prob": best_p,
                    "alt_scores": _alt_scores(pred, exclude=(best_hg, best_ag)),
                    "p_home": pred["outcome"]["H"],
                    "p_draw": pred["outcome"]["D"],
                    "p_away": pred["outcome"]["A"],
                    "xg_home": pred["lambda_home"],
                    "xg_away": pred["lambda_away"],
                    "is_actual": False,
                })
    return out


def _modal_group_table(bundle: PredictorBundle, group: list[str],
                       fmt_name: str | None = None,
                       wc_blend: float = 0.0,
                       exact_pts: float = 3.0,
                       result_pts: float = 1.0,
                       gd_pts: float = 0.0,
                       draw_bonus: float = 0.0,
                       actual_results: dict | None = None) -> list[tuple[str, dict]]:
    """Predict the group standings using EV-maximising scoreline per match."""
    from .wc26_strength import apply_wc_prior_to_prediction, fmt_uses_wc_prior
    use_wc = wc_blend > 0 and fmt_uses_wc_prior(fmt_name)
    actual_results = actual_results or {}
    pts = {t: 0 for t in group}
    gf = {t: 0 for t in group}
    ga = {t: 0 for t in group}
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            h, a = group[i], group[j]
            # Use actual played result if available
            if (h, a) in actual_results:
                hg, ag = actual_results[(h, a)]
            elif (a, h) in actual_results:
                ag, hg = actual_results[(a, h)]
            else:
                pred = bundle.predict(h, a, neutral=True)
                if use_wc:
                    pred = apply_wc_prior_to_prediction(pred, h, a, blend=wc_blend)
                hg, ag, _, _ = best_ev_score(
                    pred, exact_pts=exact_pts, result_pts=result_pts, gd_pts=gd_pts,
                    draw_bonus=draw_bonus,
                )
            gf[h] += hg; ga[h] += ag
            gf[a] += ag; ga[a] += hg
            if hg > ag:
                pts[h] += 3
            elif hg < ag:
                pts[a] += 3
            else:
                pts[h] += 1; pts[a] += 1
    ranked = sorted(group, key=lambda t: (-pts[t], -(gf[t] - ga[t]), -gf[t]))
    return [(t, {"pts": pts[t], "gd": gf[t] - ga[t], "gf": gf[t]}) for t in ranked]


def sample_one_tournament(bundle: PredictorBundle, fmt: TournamentFormat,
                          groups: list[list[str]], seed: int | None = None,
                          max_goals: int = 7,
                          fmt_name: str | None = None,
                          wc_blend: float = 0.0) -> tuple[list[dict], list[list[tuple[str, dict]]],
                                                       list[list[dict]], str | None]:
    """Run ONE random tournament simulation. Returns (group_fixtures, group_standings,
    knockout_rounds, champion). Each call with a new seed gives a different tournament -
    including upsets. If a real schedule is available for `fmt_name`, group fixtures
    use real dates and the real fixture list."""
    from .real_groups import ALIASES
    from .schedules import get_schedule, ko_date

    rng = np.random.default_rng(seed)
    all_teams = sorted({t for g in groups for t in g})
    cache = _precompute(bundle, all_teams, neutral=True, max_goals=max_goals,
                        fmt_name=fmt_name, wc_blend=wc_blend)
    schedule = get_schedule(fmt_name) if fmt_name else None

    # --- group stage ---
    group_fixtures: list[dict] = []
    standings: list[list[tuple[str, dict]]] = []
    team_to_group = {t: gi for gi, group in enumerate(groups) for t in group}
    pts: dict[str, int] = {}
    gf: dict[str, int] = {}
    ga: dict[str, int] = {}
    for group in groups:
        for t in group:
            pts[t] = 0
            gf[t] = 0
            ga[t] = 0

    if schedule and "group_fixtures" in schedule:
        per_team_count: dict[str, int] = {}
        for date, raw_h, raw_a in schedule["group_fixtures"]:
            h = ALIASES.get(raw_h, raw_h)
            a = ALIASES.get(raw_a, raw_a)
            if h not in team_to_group or a not in team_to_group:
                continue
            md = max(per_team_count.get(h, 0), per_team_count.get(a, 0)) + 1
            per_team_count[h] = per_team_count.get(h, 0) + 1
            per_team_count[a] = per_team_count.get(a, 0) + 1
            p = cache[(h, a)]
            idx = rng.choice(len(p), p=p)
            hg, ag = divmod(idx, max_goals + 1)
            gf[h] += hg; ga[h] += ag
            gf[a] += ag; ga[a] += hg
            if hg > ag:
                pts[h] += 3
            elif hg < ag:
                pts[a] += 3
            else:
                pts[h] += 1; pts[a] += 1
            group_fixtures.append({
                "date": date, "group_idx": team_to_group[h], "matchday": md,
                "home": h, "away": a, "score": (hg, ag),
            })
        group_fixtures.sort(key=lambda x: (x["date"], x["group_idx"]))
    else:
        for gi, group in enumerate(groups):
            for md_idx, md_pairs in enumerate(_matchday_pairs(group)):
                for (h, a) in md_pairs:
                    p = cache[(h, a)]
                    idx = rng.choice(len(p), p=p)
                    hg, ag = divmod(idx, max_goals + 1)
                    gf[h] += hg; ga[h] += ag
                    gf[a] += ag; ga[a] += hg
                    if hg > ag:
                        pts[h] += 3
                    elif hg < ag:
                        pts[a] += 3
                    else:
                        pts[h] += 1; pts[a] += 1
                    group_fixtures.append({
                        "group_idx": gi, "matchday": md_idx + 1,
                        "home": h, "away": a, "score": (hg, ag),
                    })

    for group in groups:
        ranked = sorted(group, key=lambda t: (-pts[t], -(gf[t] - ga[t]), -gf[t]))
        standings.append([(t, {"pts": pts[t], "gd": gf[t] - ga[t], "gf": gf[t]}) for t in ranked])

    # --- determine knockout qualifiers ---
    qualifiers: list[str] = []
    for ranked in standings:
        for t, _ in ranked[:fmt.advance_per_group]:
            qualifiers.append(t)
    if fmt.best_thirds > 0:
        thirds = []
        for ranked in standings:
            if len(ranked) > fmt.advance_per_group:
                t, s = ranked[fmt.advance_per_group]
                thirds.append((t, s))
        thirds.sort(key=lambda kv: (-kv[1]["pts"], -kv[1]["gd"], -kv[1]["gf"]))
        for t, _ in thirds[:fmt.best_thirds]:
            qualifiers.append(t)

    # --- knockout cascade (real bracket if available, else sequential) ---
    rounds: list[list[dict]] = []
    current = qualifiers
    first_pairs = _first_round_pairs(fmt_name or fmt.name, len(current))
    round_idx = 0
    while len(current) >= 2:
        round_name = _round_name(len(current))
        round_date = ko_date(fmt_name, round_name) if fmt_name else None
        next_round = []
        matches = []
        if round_idx == 0:
            pair_iter = [(current[i], current[j]) for (i, j) in first_pairs]
        else:
            pair_iter = [(current[i], current[i + 1]) for i in range(0, len(current), 2)]
        for (h, a) in pair_iter:
            p = cache[(h, a)]
            idx = rng.choice(len(p), p=p)
            hg, ag = divmod(idx, max_goals + 1)
            went_to_pens = False
            if hg > ag:
                winner = h
            elif hg < ag:
                winner = a
            else:
                winner = _penalty_winner(bundle, h, a, rng)
                went_to_pens = True
            entry = {
                "round": round_name, "home": h, "away": a,
                "score": (hg, ag), "winner": winner, "pens": went_to_pens,
            }
            if round_date:
                entry["date"] = round_date
            matches.append(entry)
            next_round.append(winner)
        rounds.append(matches)
        current = next_round
        round_idx += 1

    champion = current[0] if current else None
    return group_fixtures, standings, rounds, champion


def predict_modal_bracket(bundle: PredictorBundle, fmt: TournamentFormat,
                          groups: list[list[str]],
                          fmt_name: str | None = None,
                          wc_blend: float = 0.0,
                          exact_pts: float = 3.0,
                          result_pts: float = 1.0,
                          gd_pts: float = 0.0,
                          draw_bonus: float = 0.0,
                          actual_results: dict | None = None) -> tuple[list[list[dict]], str | None,
                                                            list[list[tuple[str, dict]]]]:
    """Cascade most-likely outcomes through the tournament.

    Returns (rounds, champion, group_standings) where:
      * group_standings[i] is the modal table for group i (list of (team, stats))
      * rounds is a list of rounds, each a list of fixture dicts
      * champion is the predicted winner
    """
    from .wc26_strength import apply_wc_prior_to_prediction, fmt_uses_wc_prior
    use_wc = wc_blend > 0 and fmt_uses_wc_prior(fmt_name)
    # 1) Group stage modal standings (WC prior applied if enabled)
    actual_results = actual_results or {}
    standings = [_modal_group_table(bundle, g, fmt_name=fmt_name, wc_blend=wc_blend,
                                     exact_pts=exact_pts, result_pts=result_pts, gd_pts=gd_pts,
                                     draw_bonus=draw_bonus, actual_results=actual_results)
                 for g in groups]

    # 2) Determine qualifiers - top N per group, then best 3rds
    qualifiers: list[str] = []
    for ranked in standings:
        for t, _ in ranked[:fmt.advance_per_group]:
            qualifiers.append(t)
    if fmt.best_thirds > 0:
        thirds = []
        for ranked in standings:
            if len(ranked) > fmt.advance_per_group:
                team, stats = ranked[fmt.advance_per_group]
                thirds.append((team, stats))
        thirds.sort(key=lambda kv: (-kv[1]["pts"], -kv[1]["gd"], -kv[1]["gf"]))
        for t, _ in thirds[:fmt.best_thirds]:
            qualifiers.append(t)

    # 3) Knockout cascade (real bracket if available, else sequential)
    from .schedules import ko_date as _ko_date
    rounds: list[list[dict]] = []
    current = qualifiers
    first_pairs = _first_round_pairs(fmt_name or fmt.name, len(current))
    round_idx = 0
    while len(current) >= 2:
        round_name = _round_name(len(current))
        round_date = _ko_date(fmt_name, round_name) if fmt_name else None
        next_round = []
        matches = []
        if round_idx == 0:
            iter_pairs = [(current[i], current[j]) for (i, j) in first_pairs]
        else:
            iter_pairs = [(current[i], current[i + 1]) for i in range(0, len(current), 2)]
        for (h, a) in iter_pairs:
            actual = actual_results.get((h, a)) or (
                (actual_results[(a, h)][1], actual_results[(a, h)][0]) if (a, h) in actual_results else None
            )
            if actual is not None:
                hg, ag = actual
                if hg > ag:
                    winner, pens = h, False
                elif hg < ag:
                    winner, pens = a, False
                else:
                    # KO actual draw means we don't know who advanced from this data alone.
                    # Fall back to higher Elo as a best guess (real shootouts are ~50/50).
                    winner = h if bundle.elo.rating(h) >= bundle.elo.rating(a) else a
                    pens = True
                entry = {
                    "round": round_name, "home": h, "away": a,
                    "score": (hg, ag), "winner": winner, "pens": pens,
                    "p_home": 1.0 if hg > ag else 0.0,
                    "p_draw": 1.0 if hg == ag else 0.0,
                    "p_away": 1.0 if hg < ag else 0.0,
                    "is_actual": True,
                }
                if round_date:
                    entry["date"] = round_date
                matches.append(entry)
                next_round.append(winner)
                continue

            pred = bundle.predict(h, a, neutral=True)
            if use_wc:
                pred = apply_wc_prior_to_prediction(pred, h, a, blend=wc_blend)
            hg, ag, _, _ = best_ev_score(
                pred, exact_pts=exact_pts, result_pts=result_pts, gd_pts=gd_pts,
                draw_bonus=draw_bonus,
            )
            p_h = pred["outcome"]["H"]
            p_a = pred["outcome"]["A"]
            if hg > ag:
                winner = h
                pens = False
            elif hg < ag:
                winner = a
                pens = False
            else:
                # KO can't end as draw - pick winner by outcome probability
                winner = h if p_h >= p_a else a
                pens = True
            entry = {
                "round": round_name, "home": h, "away": a,
                "score": (hg, ag), "winner": winner,
                "pens": pens,
                "p_home": pred["outcome"]["H"],
                "p_draw": pred["outcome"]["D"],
                "p_away": pred["outcome"]["A"],
                "is_actual": False,
            }
            if round_date:
                entry["date"] = round_date
            matches.append(entry)
            next_round.append(winner)
        rounds.append(matches)
        current = next_round
        round_idx += 1
    champion = current[0] if current else None
    return rounds, champion, standings
