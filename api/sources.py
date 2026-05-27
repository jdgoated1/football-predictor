"""Read cached FBref data from parquet files.

The Streamlit Cloud container can't run soccerdata's Selenium-based scraper
(no Chrome, read-only venv, IP blocked by FBref). So scraping happens in the
daily GitHub Action via `scripts/refresh_live_data.py`, which commits parquet
files under `data/processed/live/`. This module just reads those files.

If the parquet files don't exist (fresh deploy before first workflow run),
all readers return empty results and the callers surface a friendly message.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
LIVE_DIR = ROOT / "data" / "processed" / "live"
SCHEDULES_PATH = LIVE_DIR / "schedules.parquet"
LINEUPS_PATH = LIVE_DIR / "lineups.parquet"


LEAGUE_CODES: dict[str, str] = {
    "EPL": "ENG-Premier League",
    "LaLiga": "ESP-La Liga",
    "Bundesliga": "GER-Bundesliga",
    "SerieA": "ITA-Serie A",
    "Ligue1": "FRA-Ligue 1",
}


def current_season() -> str:
    today = datetime.today()
    start = today.year if today.month >= 8 else today.year - 1
    return f"{str(start)[-2:]}{str(start + 1)[-2:]}"


def resolve_league(name: str) -> str:
    if name not in LEAGUE_CODES:
        raise ValueError(
            f"Unknown league '{name}'. Valid: {', '.join(LEAGUE_CODES)}"
        )
    return LEAGUE_CODES[name]


# `_mtime` keeps the in-process cache fresh against the parquet file - if the
# file is rewritten (e.g. the GitHub Action committed new data and the app
# restarted), the next read picks up the new content.
_SCHEDULE_CACHE: tuple[float, pd.DataFrame] | None = None
_LINEUPS_CACHE: tuple[float, pd.DataFrame] | None = None


def _read_cached(path: Path, slot: str) -> pd.DataFrame:
    global _SCHEDULE_CACHE, _LINEUPS_CACHE
    if not path.exists():
        return pd.DataFrame()
    mtime = path.stat().st_mtime
    cur = _SCHEDULE_CACHE if slot == "schedule" else _LINEUPS_CACHE
    if cur and cur[0] == mtime:
        return cur[1]
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if slot == "schedule":
        _SCHEDULE_CACHE = (mtime, df)
    else:
        _LINEUPS_CACHE = (mtime, df)
    return df


def _all_schedules() -> pd.DataFrame:
    return _read_cached(SCHEDULES_PATH, "schedule")


def _all_lineups() -> pd.DataFrame:
    return _read_cached(LINEUPS_PATH, "lineups")


def _schedule_for(league_name: str, season: str | None = None) -> pd.DataFrame:
    """Filter the cached schedules to one league + season."""
    df = _all_schedules()
    if df.empty:
        return df
    seas = season or current_season()
    mask = df["league"] == league_name
    if "season" in df.columns:
        mask &= df["season"].astype(str) == seas
    return df[mask].copy()


def data_freshness() -> dict:
    """Surface when the cached data was last refreshed (for UI display)."""
    out = {}
    for label, path in [("schedules", SCHEDULES_PATH), ("lineups", LINEUPS_PATH)]:
        if path.exists():
            ts = datetime.fromtimestamp(path.stat().st_mtime)
            out[label] = ts.isoformat()
        else:
            out[label] = None
    return out


def _parse_score(value) -> tuple[int, int] | None:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if not s or ("–" not in s and "-" not in s):
        return None
    sep = "–" if "–" in s else "-"
    parts = s.split(sep)
    if len(parts) != 2:
        return None
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        return None


def fixtures(league_name: str, days: int = 14, season: str | None = None) -> list[dict]:
    """Upcoming fixtures within `days` days, ordered by kickoff."""
    resolve_league(league_name)  # validate
    seas = season or current_season()
    df = _schedule_for(league_name, seas)
    today = pd.Timestamp(datetime.today().date())
    cutoff = today + pd.Timedelta(days=days)

    upcoming = df[df["date"].notna() & (df["date"] >= today) & (df["date"] <= cutoff)].copy()

    # A fixture is "upcoming" if there's no score yet
    if "score" in upcoming.columns:
        upcoming = upcoming[upcoming["score"].isna() | (upcoming["score"].astype(str).str.strip() == "")]

    upcoming = upcoming.sort_values("date")

    out = []
    for row in upcoming.itertuples(index=False):
        out.append({
            "date": pd.Timestamp(row.date).date().isoformat() if not pd.isna(row.date) else None,
            "time": str(getattr(row, "time", "") or "") or None,
            "home": str(getattr(row, "home_team", "") or ""),
            "away": str(getattr(row, "away_team", "") or ""),
            "league": league_name,
            "season": seas,
            "venue": str(getattr(row, "venue", "") or "") or None,
            "week": str(getattr(row, "week", "") or "") or None,
            "game_id": str(getattr(row, "game_id", "") or "") or None,
        })
    return out


def results(league_name: str, limit: int = 20, season: str | None = None) -> list[dict]:
    """Most recent completed matches, newest first."""
    resolve_league(league_name)
    seas = season or current_season()
    df = _schedule_for(league_name, seas)
    today = pd.Timestamp(datetime.today().date())

    completed = df[df["date"].notna() & (df["date"] <= today)].copy()
    if "score" in completed.columns:
        completed = completed[completed["score"].astype(str).str.contains(r"\d", na=False)]

    completed = completed.sort_values("date", ascending=False).head(limit)

    out = []
    for row in completed.itertuples(index=False):
        parsed = _parse_score(getattr(row, "score", None))
        if parsed is None:
            continue
        hg, ag = parsed
        out.append({
            "date": pd.Timestamp(row.date).date().isoformat(),
            "home": str(getattr(row, "home_team", "") or ""),
            "away": str(getattr(row, "away_team", "") or ""),
            "home_goals": hg,
            "away_goals": ag,
            "league": league_name,
            "season": seas,
            "game_id": str(getattr(row, "game_id", "") or "") or None,
        })
    return out


def standings(league_name: str, season: str | None = None) -> list[dict]:
    """Computed standings from completed games this season."""
    resolve_league(league_name)
    seas = season or current_season()
    df = _schedule_for(league_name, seas)

    table: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        parsed = _parse_score(getattr(row, "score", None))
        if parsed is None:
            continue
        hg, ag = parsed
        home = str(getattr(row, "home_team", "") or "")
        away = str(getattr(row, "away_team", "") or "")
        if not home or not away:
            continue
        for team in (home, away):
            table.setdefault(team, {
                "team": team, "played": 0, "wins": 0, "draws": 0, "losses": 0,
                "goals_for": 0, "goals_against": 0, "points": 0,
            })
        table[home]["played"] += 1
        table[away]["played"] += 1
        table[home]["goals_for"] += hg
        table[home]["goals_against"] += ag
        table[away]["goals_for"] += ag
        table[away]["goals_against"] += hg
        if hg > ag:
            table[home]["wins"] += 1
            table[home]["points"] += 3
            table[away]["losses"] += 1
        elif hg < ag:
            table[away]["wins"] += 1
            table[away]["points"] += 3
            table[home]["losses"] += 1
        else:
            table[home]["draws"] += 1
            table[away]["draws"] += 1
            table[home]["points"] += 1
            table[away]["points"] += 1

    rows = list(table.values())
    for r in rows:
        r["goal_diff"] = r["goals_for"] - r["goals_against"]
    rows.sort(key=lambda r: (-r["points"], -r["goal_diff"], -r["goals_for"], r["team"]))
    for idx, r in enumerate(rows, start=1):
        r["rank"] = idx
    return rows


def _s(v):
    if v is None or pd.isna(v):
        return None
    s = str(v).strip()
    return s or None


def lineup(game_id: str, league_name: str, season: str | None = None) -> dict:
    """Lineups for one match, looked up by `game_id` in the cached parquet."""
    resolve_league(league_name)
    all_lups = _all_lineups()
    if all_lups.empty:
        return {"home": None, "away": None}
    df = all_lups[all_lups["game_id"].astype(str) == str(game_id)].copy()
    if df.empty:
        return {"home": None, "away": None}

    teams_in_df = df["team"].unique().tolist() if "team" in df.columns else []
    if len(teams_in_df) < 2:
        return {"home": None, "away": None, "raw_teams": teams_in_df}

    # Prefer home_team/away_team columns committed by refresh_live_data; fall back
    # to parsing the 'game' string if those are missing.
    home_team = away_team = None
    if "home_team" in df.columns and "away_team" in df.columns:
        home_team = _s(df["home_team"].iloc[0])
        away_team = _s(df["away_team"].iloc[0])
    if (home_team is None or away_team is None) and "game" in df.columns:
        game_str = str(df["game"].iloc[0])
        after_date = game_str.split(" ", 1)[1] if " " in game_str else game_str
        for candidate in teams_in_df:
            if after_date.startswith(candidate + "-"):
                home_team = candidate
                away_team = after_date[len(candidate) + 1:]
                break
    if home_team is None or away_team is None:
        home_team, away_team = teams_in_df[0], teams_in_df[1]

    starter_mask = df["is_starter"].fillna(False).astype(bool) if "is_starter" in df.columns else None

    def players_for(team_name: str, kind: str) -> list[dict]:
        sub = df[df["team"] == team_name]
        if starter_mask is not None:
            team_starter = starter_mask.loc[sub.index]
            sub = sub[team_starter] if kind == "starting" else sub[~team_starter]
        players = []
        for r in sub.itertuples(index=False):
            num = getattr(r, "jersey_number", None)
            mins = getattr(r, "minutes_played", None)
            players.append({
                "number": str(int(num)) if pd.notna(num) else None,
                "name": _s(getattr(r, "player", None)) or "",
                "position": _s(getattr(r, "position", None)),
                "minutes": int(mins) if pd.notna(mins) else None,
            })
        return players

    return {
        "home": {
            "team": home_team,
            "formation": None,
            "starting": players_for(home_team, "starting"),
            "bench": players_for(home_team, "bench"),
        },
        "away": {
            "team": away_team,
            "formation": None,
            "starting": players_for(away_team, "starting"),
            "bench": players_for(away_team, "bench"),
        },
    }


def list_leagues() -> list[str]:
    return list(LEAGUE_CODES.keys())


def teams_in_league(league_name: str, season: str | None = None) -> list[str]:
    """Distinct team names appearing in this season's schedule (FBref naming)."""
    resolve_league(league_name)
    seas = season or current_season()
    df = _schedule_for(league_name, seas)
    names = set()
    for col in ("home_team", "away_team"):
        if col in df.columns:
            names.update(str(x) for x in df[col].dropna().unique())
    return sorted(names)


def _team_completed_matches(team: str, league_name: str, season: str | None,
                            limit: int | None = None) -> pd.DataFrame:
    """Helper: return this team's completed matches, newest first."""
    resolve_league(league_name)
    seas = season or current_season()
    df = _schedule_for(league_name, seas)
    if "home_team" not in df.columns or "away_team" not in df.columns:
        return df.iloc[0:0]
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    sub = df[mask & df["date"].notna()].copy()
    if "score" in sub.columns:
        sub = sub[sub["score"].astype(str).str.contains(r"\d", na=False)]
    sub = sub.sort_values("date", ascending=False)
    if limit:
        sub = sub.head(limit)
    return sub


def team_form(team: str, league_name: str, n: int = 10,
              season: str | None = None) -> dict:
    """Last `n` completed matches for `team` with W/D/L summary."""
    sub = _team_completed_matches(team, league_name, season, limit=n)

    matches = []
    wins = draws = losses = gf = ga = 0
    form_chars: list[str] = []
    for row in sub.itertuples(index=False):
        parsed = _parse_score(getattr(row, "score", None))
        if parsed is None:
            continue
        hg, ag = parsed
        home = str(getattr(row, "home_team", "") or "")
        away = str(getattr(row, "away_team", "") or "")
        is_home = home == team
        team_goals = hg if is_home else ag
        opp_goals = ag if is_home else hg
        opponent = away if is_home else home
        if team_goals > opp_goals:
            result = "W"; wins += 1
        elif team_goals < opp_goals:
            result = "L"; losses += 1
        else:
            result = "D"; draws += 1
        gf += team_goals
        ga += opp_goals
        form_chars.append(result)
        matches.append({
            "date": pd.Timestamp(row.date).date().isoformat(),
            "opponent": opponent,
            "venue": "home" if is_home else "away",
            "team_goals": int(team_goals),
            "opponent_goals": int(opp_goals),
            "result": result,
            "game_id": str(getattr(row, "game_id", "") or "") or None,
        })

    # form_chars currently newest-first; flip to oldest-first for "WWLDW" readability
    form_string = "".join(reversed(form_chars))
    return {
        "team": team,
        "league": league_name,
        "season": season or current_season(),
        "matches": matches,
        "summary": {
            "played": len(matches),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_for": gf,
            "goals_against": ga,
            "goal_diff": gf - ga,
            "points": wins * 3 + draws,
            "form": form_string,
        },
    }


def team_last_lineups(team: str, league_name: str, n: int = 5,
                      season: str | None = None) -> dict:
    """Pull the team's last `n` starting XIs by fetching the lineup for each
    of their recent matches.

    WARNING: this triggers `n` FBref calls on cold cache (~5-15s each).
    """
    sub = _team_completed_matches(team, league_name, season, limit=n)
    lineups_out = []
    for row in sub.itertuples(index=False):
        gid = str(getattr(row, "game_id", "") or "")
        if not gid:
            continue
        try:
            data = lineup(gid, league_name=league_name, season=season)
        except Exception:
            continue
        home = str(getattr(row, "home_team", "") or "")
        away = str(getattr(row, "away_team", "") or "")
        is_home = home == team
        opponent = away if is_home else home
        # Skip matches where lineup data wasn't fetched (FBref hiccup, etc.) -
        # both sides come back as None for those.
        if not data.get("home") and not data.get("away"):
            continue
        home_block = data.get("home") or {}
        away_block = data.get("away") or {}
        if home_block.get("team") == team:
            block = home_block
        elif away_block.get("team") == team:
            block = away_block
        else:
            # Names don't match exactly - fall back to the side from the schedule
            block = home_block if is_home else away_block
        if not block:
            continue
        lineups_out.append({
            "date": pd.Timestamp(row.date).date().isoformat(),
            "opponent": opponent,
            "venue": "home" if is_home else "away",
            "game_id": gid,
            "starting": block["starting"],
            "bench": block["bench"],
        })
    return {
        "team": team,
        "league": league_name,
        "season": season or current_season(),
        "lineups": lineups_out,
    }


# FBref position codes grouped into 4 buckets. The first matching prefix wins,
# so order matters - keep more-specific codes (e.g. "WB") before broader ones.
_POSITION_BUCKETS = [
    ("GK", "GK"),
    ("CB", "DEF"), ("RB", "DEF"), ("LB", "DEF"), ("WB", "DEF"), ("DF", "DEF"),
    ("DM", "MID"), ("CM", "MID"), ("AM", "MID"), ("LM", "MID"), ("RM", "MID"), ("MF", "MID"),
    ("LW", "FW"), ("RW", "FW"), ("ST", "FW"), ("CF", "FW"), ("FW", "FW"),
]


def _bucket_position(pos: str | None) -> str:
    if not pos:
        return "MID"
    p = pos.upper().strip()
    for prefix, bucket in _POSITION_BUCKETS:
        if prefix in p:
            return bucket
    return "MID"


def _is_realistic_formation(d: int, m: int, f: int) -> bool:
    """Sanity check for outfield bucket counts (DEF-MID-FW)."""
    return d + m + f == 10 and 3 <= d <= 5 and 2 <= m <= 5 and 1 <= f <= 3


def _pick_formation(per_match_formations: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """Pick the team's predicted shape from per-match bucket counts.

    Prefer the most-common realistic formation. If no match yielded a realistic
    shape (e.g. position tagging was odd), fall back to the per-bucket median
    clamped to valid ranges, summing to 10 by adjusting the largest bucket.
    """
    realistic = [f for f in per_match_formations if _is_realistic_formation(*f)]
    if realistic:
        counts: dict[tuple, int] = {}
        for f in realistic:
            counts[f] = counts.get(f, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    if not per_match_formations:
        return (4, 3, 3)

    import statistics
    defs = max(3, min(5, int(round(statistics.median(f[0] for f in per_match_formations)))))
    mids = max(2, min(5, int(round(statistics.median(f[1] for f in per_match_formations)))))
    fws  = max(1, min(3, int(round(statistics.median(f[2] for f in per_match_formations)))))
    total = defs + mids + fws
    if total != 10:
        diff = 10 - total
        # Adjust MID first (most flexible), then DEF, then FW, while staying in range.
        for name, lo, hi in [("mids", 2, 5), ("defs", 3, 5), ("fws", 1, 3)]:
            cur = {"defs": defs, "mids": mids, "fws": fws}[name]
            new = max(lo, min(hi, cur + diff))
            diff -= (new - cur)
            if name == "defs": defs = new
            elif name == "mids": mids = new
            else: fws = new
            if diff == 0:
                break
    return (defs, mids, fws)


def predicted_xi(team: str, league_name: str, lookback: int = 5,
                 season: str | None = None) -> dict:
    """Frequency-based predicted XI shaped to match the team's recent formation.

    Algorithm:
      1. Compute each match's actual formation (bucketing FBref position codes).
      2. Predict the team's shape: most-common realistic formation across those
         matches (e.g. 4-3-3), falling back to clamped medians if none qualify.
      3. For each bucket slot in the predicted shape, pick the most-frequent
         starter whose dominant bucket matches. If a bucket runs short, pull
         from the nearest neighbour bucket so we still field 11.

    Honest limits: no injury/suspension data, no opponent-specific tactics.
    """
    last = team_last_lineups(team, league_name, n=lookback, season=season)
    n_matches = len(last["lineups"])

    starts: dict[str, dict] = {}
    per_match_formations: list[tuple[int, int, int]] = []

    for match in last["lineups"]:
        match_counts = {"GK": 0, "DEF": 0, "MID": 0, "FW": 0}
        for p in match["starting"]:
            name = (p.get("name") or "").strip()
            if not name:
                continue
            pos = p.get("position") or ""
            rec = starts.setdefault(name, {"name": name, "positions": {}, "starts": 0})
            rec["starts"] += 1
            rec["positions"][pos] = rec["positions"].get(pos, 0) + 1
            match_counts[_bucket_position(pos)] += 1
        per_match_formations.append((match_counts["DEF"], match_counts["MID"], match_counts["FW"]))

    def_n, mid_n, fw_n = _pick_formation(per_match_formations)

    # Per-player summary
    ranked = []
    for name, rec in starts.items():
        most_common_pos = max(rec["positions"].items(), key=lambda kv: kv[1])[0] if rec["positions"] else ""
        ranked.append({
            "name": name,
            "starts": rec["starts"],
            "position": most_common_pos,
            "bucket": _bucket_position(most_common_pos),
            "start_freq": f"{rec['starts']}/{n_matches}" if n_matches else "0/0",
        })
    ranked.sort(key=lambda p: -p["starts"])

    # Allocate slots, falling back to adjacent buckets when a bucket runs short.
    NEIGHBOURS = {"DEF": ["MID"], "MID": ["FW", "DEF"], "FW": ["MID"]}
    chosen: list[dict] = []
    used: set[str] = set()

    def take(bucket: str, k: int) -> list[dict]:
        picked = []
        for p in ranked:
            if len(picked) >= k:
                break
            if p["name"] in used or p["bucket"] != bucket:
                continue
            picked.append(p)
            used.add(p["name"])
        return picked

    def take_with_fallback(bucket: str, k: int) -> list[dict]:
        picked = take(bucket, k)
        if len(picked) < k:
            for fallback_bucket in NEIGHBOURS.get(bucket, []):
                picked += take(fallback_bucket, k - len(picked))
                if len(picked) >= k:
                    break
        return picked

    gk = take("GK", 1)
    defs = take_with_fallback("DEF", def_n)
    mids = take_with_fallback("MID", mid_n)
    fws = take_with_fallback("FW", fw_n)
    xi = gk + defs + mids + fws

    # If we still don't have 11 (e.g. very thin squad in the data), top up by raw frequency.
    if len(xi) < 11:
        for p in ranked:
            if p["name"] in used:
                continue
            xi.append(p)
            used.add(p["name"])
            if len(xi) >= 11:
                break

    formation = f"{def_n}-{mid_n}-{fw_n}"

    avg_start_rate = (sum(p["starts"] for p in xi) / (len(xi) * n_matches)) if (xi and n_matches) else 0.0
    if avg_start_rate >= 0.85:
        confidence = "high"
    elif avg_start_rate >= 0.65:
        confidence = "moderate"
    else:
        confidence = "low"

    return {
        "team": team,
        "league": league_name,
        "season": season or current_season(),
        "based_on_matches": n_matches,
        "method": ("Frequency-based: predicted formation is the most-common realistic shape "
                   "across the team's last N league matches, with the highest-frequency "
                   "starter filling each role. Does NOT account for injuries, suspensions, "
                   "or opponent-specific tactical changes."),
        "predicted_xi": xi,
        "formation": formation,
        "confidence": confidence,
        "avg_start_rate": round(avg_start_rate, 3),
    }
