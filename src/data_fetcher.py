"""Download and unify match data from multiple sources.

Sources:
  * football-data.co.uk - season CSVs for the top 5 European leagues (free, no auth)
  * github.com/martj42/international_results - international match results since 1872
"""
from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

# football-data.co.uk league codes
LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

# Pull last 15 seasons - season string is "YYYY/YY", code is e.g. "2324"
def season_codes(start_year: int = 2010, end_year: int | None = None) -> list[str]:
    if end_year is None:
        # Football season starts in August. Up to July, "current" season is last year.
        today = datetime.today()
        end_year = today.year if today.month >= 8 else today.year - 1
        end_year += 1   # overshoot by one to grab the next season as soon as it appears
    codes = []
    for y in range(start_year, end_year + 1):
        a = str(y)[-2:]
        b = str(y + 1)[-2:]
        codes.append(f"{a}{b}")
    return codes


def fd_url(season: str, code: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"


def fetch_text(url: str, timeout: int = 30) -> str | None:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and len(r.content) > 200:
            # football-data.co.uk uses cp1252; force decode
            try:
                return r.content.decode("utf-8")
            except UnicodeDecodeError:
                return r.content.decode("cp1252", errors="replace")
    except Exception as e:
        print(f"  ! {url}: {e}", file=sys.stderr)
    return None


def fetch_leagues(start_year: int = 2010, end_year: int = 2025) -> pd.DataFrame:
    """Download every (season, league) CSV, normalise schema, concat."""
    frames = []
    for season in season_codes(start_year, end_year):
        for code, name in LEAGUES.items():
            url = fd_url(season, code)
            out = RAW / f"{code}_{season}.csv"
            text = None
            if out.exists() and out.stat().st_size > 200:
                text = out.read_text(encoding="utf-8", errors="replace")
            else:
                text = fetch_text(url)
                if text:
                    out.write_text(text, encoding="utf-8")
                    print(f"  + {name} {season}")
            if not text:
                continue
            try:
                df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
            except Exception as e:
                print(f"  ! parse {code}_{season}: {e}", file=sys.stderr)
                continue
            df = _normalise_league(df, name)
            if df is not None and len(df):
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["date", "home", "away", "home_goals", "away_goals"])
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def _normalise_league(df: pd.DataFrame, competition: str) -> pd.DataFrame | None:
    needed = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not needed.issubset(df.columns):
        return None
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        "home": df["HomeTeam"].astype(str).str.strip(),
        "away": df["AwayTeam"].astype(str).str.strip(),
        "home_goals": pd.to_numeric(df["FTHG"], errors="coerce"),
        "away_goals": pd.to_numeric(df["FTAG"], errors="coerce"),
        "competition": competition,
        "neutral": False,
        "is_international": False,
    })
    # bring across optional shot stats
    for src, dst in [("HS", "home_shots"), ("AS", "away_shots"),
                     ("HST", "home_sot"), ("AST", "away_sot")]:
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")
    # Bookmaker odds - keep all available for consensus aggregation
    # B365=Bet365, BW=Betway, IW=Interwetten, PS=Pinnacle (sharpest), WH=William Hill, VC=VC Bet
    BOOK_PREFIXES = ["B365", "BW", "IW", "PS", "WH", "VC", "LB", "GB", "BS", "SJ", "SB"]
    for prefix in BOOK_PREFIXES:
        for outcome in ["H", "D", "A"]:
            src = f"{prefix}{outcome}"
            dst = f"odds_{prefix.lower()}_{outcome.lower()}"
            if src in df.columns:
                out[dst] = pd.to_numeric(df[src], errors="coerce")
    # Back-compat: keep odds_h/d/a pointing at B365 specifically
    for outcome in ["h", "d", "a"]:
        bk_col = f"odds_b365_{outcome}"
        if bk_col in out.columns:
            out[f"odds_{outcome}"] = out[bk_col]
    return out.dropna(subset=["date", "home", "away", "home_goals", "away_goals"])


INT_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def fetch_internationals(start_year: int = 2000) -> pd.DataFrame:
    """Download international results (men's senior teams) - keeps last ~25y."""
    out = RAW / "international_results.csv"
    if out.exists() and out.stat().st_size > 1000:
        text = out.read_text(encoding="utf-8", errors="replace")
    else:
        text = fetch_text(INT_URL, timeout=60)
        if text:
            out.write_text(text, encoding="utf-8")
            print(f"  + internationals")
    if not text:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].dt.year >= start_year].copy()
    # Map tournament -> "competition" with a few useful collapsings
    df["competition"] = df["tournament"]
    out_df = pd.DataFrame({
        "date": df["date"],
        "home": df["home_team"].astype(str).str.strip(),
        "away": df["away_team"].astype(str).str.strip(),
        "home_goals": pd.to_numeric(df["home_score"], errors="coerce"),
        "away_goals": pd.to_numeric(df["away_score"], errors="coerce"),
        "competition": df["competition"],
        "neutral": df["neutral"].astype(bool) if "neutral" in df.columns else False,
        "is_international": True,
    })
    return out_df.dropna(subset=["date", "home", "away", "home_goals", "away_goals"])


def fetch_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Fetching league data...")
    leagues = fetch_leagues()
    print(f"  -> {len(leagues):,} league matches")
    print("Fetching international data...")
    internationals = fetch_internationals()
    print(f"  -> {len(internationals):,} international matches")
    leagues.to_parquet(PROCESSED / "leagues.parquet", index=False)
    internationals.to_parquet(PROCESSED / "internationals.parquet", index=False)
    return leagues, internationals


if __name__ == "__main__":
    fetch_all()
