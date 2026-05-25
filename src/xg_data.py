"""Understat xG fetcher and merger.

Understat covers the top 5 European leagues from ~2014-15 onwards.
We fetch once, cache to parquet, then merge xG values into the main matches
dataframe by (date + normalised team names).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XG_CACHE = ROOT / "data" / "processed" / "understat_xg.parquet"

UNDERSTAT_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

# Soccerdata Understat uses "YYYY/YY" or short format like "2425"
SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]

# Understat team name -> football-data.co.uk team name
TEAM_MAP = {
    # ---- Premier League ----
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Tottenham": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton": "Brighton",
    "West Ham": "West Ham",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Crystal Palace": "Crystal Palace",
    "Leicester": "Leicester",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Chelsea": "Chelsea",
    "Aston Villa": "Aston Villa",
    "Everton": "Everton",
    "Burnley": "Burnley",
    "Brentford": "Brentford",
    "Bournemouth": "Bournemouth",
    "Fulham": "Fulham",
    "Ipswich": "Ipswich",
    "Luton": "Luton",
    "Norwich": "Norwich",
    "Watford": "Watford",
    "West Bromwich Albion": "West Brom",
    "Cardiff": "Cardiff",
    "Huddersfield": "Huddersfield",
    "Southampton": "Southampton",
    "Leeds": "Leeds",
    # ---- La Liga ----
    "Atletico Madrid": "Ath Madrid",
    "Real Sociedad": "Sociedad",
    "Athletic Club": "Ath Bilbao",
    "Real Madrid": "Real Madrid",
    "Barcelona": "Barcelona",
    "Real Valladolid": "Valladolid",
    "Real Betis": "Betis",
    "Espanyol": "Espanol",
    "Cadiz": "Cadiz",
    "Almeria": "Almeria",
    "Mallorca": "Mallorca",
    "Las Palmas": "Las Palmas",
    "Celta Vigo": "Celta",
    "Sevilla": "Sevilla",
    "Villarreal": "Villarreal",
    "Valencia": "Valencia",
    "Getafe": "Getafe",
    "Osasuna": "Osasuna",
    "Rayo Vallecano": "Vallecano",
    "Girona": "Girona",
    "Granada": "Granada",
    "Alaves": "Alaves",
    "Elche": "Elche",
    "Levante": "Levante",
    "Eibar": "Eibar",
    "Leganes": "Leganes",
    "Huesca": "Huesca",
    # ---- Bundesliga ----
    "Bayern Munich": "Bayern Munich",
    "RB Leipzig": "RB Leipzig",
    "Borussia Dortmund": "Dortmund",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia M.Gladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Werder Bremen": "Werder Bremen",
    "Hertha Berlin": "Hertha",
    "FC Cologne": "FC Koln",
    "FC Koln": "FC Koln",
    "Hoffenheim": "Hoffenheim",
    "Mainz 05": "Mainz",
    "Schalke 04": "Schalke 04",
    "Augsburg": "Augsburg",
    "Wolfsburg": "Wolfsburg",
    "Union Berlin": "Union Berlin",
    "Stuttgart": "Stuttgart",
    "Heidenheim": "Heidenheim",
    "Bochum": "Bochum",
    "Darmstadt": "Darmstadt",
    "Greuther Furth": "Greuther Furth",
    "Arminia Bielefeld": "Bielefeld",
    "St Pauli": "St Pauli",
    "Holstein Kiel": "Holstein Kiel",
    "Fortuna Dusseldorf": "Fortuna Dusseldorf",
    "Hannover 96": "Hannover",
    "Hamburger SV": "Hamburg",
    "Nurnberg": "Nurnberg",
    "Paderborn": "Paderborn",
    # ---- Serie A ----
    "Internazionale": "Inter",
    "AC Milan": "Milan",
    "Hellas Verona": "Verona",
    "Roma": "Roma",
    "Juventus": "Juventus",
    "Napoli": "Napoli",
    "Lazio": "Lazio",
    "Atalanta": "Atalanta",
    "Fiorentina": "Fiorentina",
    "Bologna": "Bologna",
    "Torino": "Torino",
    "Sassuolo": "Sassuolo",
    "Sampdoria": "Sampdoria",
    "Spezia": "Spezia",
    "Salernitana": "Salernitana",
    "Empoli": "Empoli",
    "Cagliari": "Cagliari",
    "Genoa": "Genoa",
    "Lecce": "Lecce",
    "Udinese": "Udinese",
    "Monza": "Monza",
    "Cremonese": "Cremonese",
    "Frosinone": "Frosinone",
    "Como": "Como",
    "Parma": "Parma",
    "Venezia": "Venezia",
    "SPAL": "Spal",
    "Crotone": "Crotone",
    "Benevento": "Benevento",
    "Brescia": "Brescia",
    "Chievo": "Chievo",
    "Pisa": "Pisa",
    # ---- Ligue 1 ----
    "Paris Saint Germain": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "Saint-Etienne": "St Etienne",
    "Olympique Marseille": "Marseille",
    "Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "Lyon": "Lyon",
    "Lens": "Lens",
    "Lille": "Lille",
    "Strasbourg": "Strasbourg",
    "Reims": "Reims",
    "Nantes": "Nantes",
    "Nice": "Nice",
    "Rennes": "Rennes",
    "Brest": "Brest",
    "Toulouse": "Toulouse",
    "Clermont Foot": "Clermont",
    "Monaco": "Monaco",
    "Montpellier": "Montpellier",
    "Le Havre": "Le Havre",
    "Lorient": "Lorient",
    "Angers": "Angers",
    "Bordeaux": "Bordeaux",
    "Dijon": "Dijon",
    "Amiens": "Amiens",
    "Auxerre": "Auxerre",
    "Ajaccio": "Ajaccio",
    "Metz": "Metz",
    "Troyes": "Troyes",
    "Nimes": "Nimes",
}


def normalize_team(name: str) -> str:
    """Map Understat team name to football-data.co.uk convention."""
    return TEAM_MAP.get(name, name)


def fetch_understat_xg(force: bool = False) -> pd.DataFrame:
    """Fetch & cache Understat schedule (with xG) for top 5 leagues."""
    if XG_CACHE.exists() and not force:
        return pd.read_parquet(XG_CACHE)
    import soccerdata as sd
    warnings.filterwarnings("ignore")
    print(f"Fetching Understat xG for {len(UNDERSTAT_LEAGUES)} leagues x {len(SEASONS)} seasons...")
    us = sd.Understat(leagues=UNDERSTAT_LEAGUES, seasons=SEASONS, no_cache=False)
    games = us.read_schedule()
    games = games.reset_index()
    keep = ["date", "home_team", "away_team", "home_xg", "away_xg", "home_goals", "away_goals"]
    games = games[[c for c in keep if c in games.columns]].copy()
    games["home"] = games["home_team"].map(normalize_team)
    games["away"] = games["away_team"].map(normalize_team)
    games["date"] = pd.to_datetime(games["date"]).dt.normalize()
    games = games.dropna(subset=["home_xg", "away_xg", "date", "home", "away"])
    XG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    games[["date", "home", "away", "home_xg", "away_xg"]].to_parquet(XG_CACHE, index=False)
    print(f"  Cached {len(games):,} matches with xG to {XG_CACHE}")
    return games


def merge_xg(matches: pd.DataFrame) -> pd.DataFrame:
    """Left-merge xG into the matches frame. Returns matches with added
    home_xg and away_xg columns (NaN where unmatched)."""
    try:
        xg = fetch_understat_xg()
    except Exception as e:
        print(f"  xG fetch failed: {e} - continuing without xG")
        matches = matches.copy()
        matches["home_xg"] = pd.NA
        matches["away_xg"] = pd.NA
        return matches
    if "home_xg" in matches.columns:
        return matches
    matches = matches.copy()
    matches["_date_key"] = pd.to_datetime(matches["date"]).dt.normalize()
    xg_keyed = xg[["date", "home", "away", "home_xg", "away_xg"]].rename(
        columns={"date": "_date_key"}
    )
    merged = matches.merge(xg_keyed, on=["_date_key", "home", "away"], how="left")
    merged = merged.drop(columns=["_date_key"])
    matched = merged["home_xg"].notna().sum()
    print(f"  xG matched: {matched:,}/{len(merged):,} ({matched/len(merged)*100:.1f}%)")
    return merged


if __name__ == "__main__":
    df = fetch_understat_xg()
    print(df.head())
    print(f"\n{len(df):,} matches with xG")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
