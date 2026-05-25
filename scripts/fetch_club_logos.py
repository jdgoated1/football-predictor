"""Fetch club badge URLs from the luukhopman/football-logos GitHub repo.

Uses GitHub's contents API to list the logos directory once, then matches
football-data.co.uk team names against the file names with manual aliases
for the messy ones. No rate limits (GitHub raw URLs are CDN-cached).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predictor import PredictorBundle

OUT = ROOT / "data" / "processed" / "club_logos.json"
REPO = "luukhopman/football-logos"
LOGO_DIR_API = f"https://api.github.com/repos/{REPO}/contents/logos"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/master/logos/"

# Football-data.co.uk short name -> the file stem used in the logos repo
ALIASES = {
    # Premier League
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Tottenham": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest",
    "Leicester": "Leicester City",
    "West Brom": "West Bromwich Albion",
    "Norwich": "Norwich City",
    "Cardiff": "Cardiff City",
    "Huddersfield": "Huddersfield Town",
    "Stoke": "Stoke City",
    "Hull": "Hull City",
    "QPR": "Queens Park Rangers",
    "Sheffield United": "Sheffield United",
    "Ipswich": "Ipswich Town",
    "Luton": "Luton Town",
    "Swansea": "Swansea City",
    "Reading": "Reading",
    # La Liga
    "Ath Madrid": "Atletico de Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Sociedad": "Real Sociedad",
    "Betis": "Real Betis",
    "Vallecano": "Rayo Vallecano",
    "Espanol": "Espanyol",
    "Celta": "Celta Vigo",
    "Valladolid": "Real Valladolid",
    "Sp Gijon": "Sporting Gijon",
    "La Coruna": "Deportivo La Coruna",
    # Bundesliga
    "Dortmund": "Borussia Dortmund",
    "Leverkusen": "Bayer 04 Leverkusen",
    "M'gladbach": "Borussia Monchengladbach",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Hertha": "Hertha BSC",
    "FC Koln": "1. FC Koln",
    "Mainz": "1. FSV Mainz 05",
    "Stuttgart": "VfB Stuttgart",
    "Bielefeld": "Arminia Bielefeld",
    "Hamburg": "Hamburger SV",
    "Hannover": "Hannover 96",
    "Nurnberg": "1. FC Nurnberg",
    "Bochum": "VfL Bochum",
    "Wolfsburg": "VfL Wolfsburg",
    "Fortuna Dusseldorf": "Fortuna Dusseldorf",
    "Werder Bremen": "Werder Bremen",
    "Augsburg": "FC Augsburg",
    "Hoffenheim": "TSG Hoffenheim",
    "Schalke 04": "FC Schalke 04",
    "Union Berlin": "1. FC Union Berlin",
    "St Pauli": "FC St. Pauli",
    "Heidenheim": "1. FC Heidenheim",
    "Holstein Kiel": "Holstein Kiel",
    "Darmstadt": "SV Darmstadt 98",
    "Greuther Furth": "Greuther Furth",
    "Paderborn": "SC Paderborn 07",
    # Serie A
    "Inter": "Inter Milan",
    "Milan": "AC Milan",
    "Verona": "Hellas Verona",
    "Roma": "AS Roma",
    "Spal": "SPAL",
    "Juventus": "Juventus",
    "Napoli": "Napoli",
    "Lazio": "Lazio",
    # Ligue 1
    "Paris SG": "Paris Saint-Germain",
    "Marseille": "Olympique Marseille",
    "Lyon": "Olympique Lyonnais",
    "St Etienne": "Saint-Etienne",
    "Nimes": "Nimes Olympique",
    "Clermont": "Clermont Foot",
    "Le Havre": "Le Havre",
    "Lorient": "FC Lorient",
    "Nantes": "FC Nantes",
    "Bordeaux": "Bordeaux",
    "Rennes": "Stade Rennais",
    "Brest": "Stade Brestois 29",
    "Toulouse": "Toulouse FC",
    "Montpellier": "Montpellier HSC",
    "Strasbourg": "RC Strasbourg",
    "Reims": "Stade de Reims",
    "Lille": "LOSC Lille",
    "Lens": "RC Lens",
    "Monaco": "AS Monaco",
    "Nice": "OGC Nice",
    "Angers": "Angers SCO",
    "Metz": "FC Metz",
    "Auxerre": "AJ Auxerre",
    "Troyes": "ESTAC Troyes",
    "Ajaccio": "AC Ajaccio",
    "Amiens": "Amiens SC",
    "Dijon": "Dijon FCO",
}


def _norm(s: str) -> str:
    """Lowercase, strip accents, collapse non-alnum."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fetch_logo_index() -> dict[str, str]:
    """Return {normalised_stem: raw_url} for every PNG in the repo.
    The repo organises logos by league subfolder, so we walk one level deep."""
    print(f"Fetching file list from {REPO}...")
    out: dict[str, str] = {}
    # First level: list /logos -> league directories
    r = requests.get(LOGO_DIR_API, headers={"Accept": "application/vnd.github+json"},
                     timeout=30)
    r.raise_for_status()
    leagues = [f for f in r.json() if f.get("type") == "dir"]
    print(f"  walking {len(leagues)} league directories...")
    for league_dir in leagues:
        league_path = league_dir["path"]
        sub_url = f"https://api.github.com/repos/{REPO}/contents/{league_path}"
        rr = requests.get(sub_url, params={"per_page": 100},
                          headers={"Accept": "application/vnd.github+json"},
                          timeout=30)
        if rr.status_code != 200:
            continue
        for f in rr.json():
            name = f.get("name", "")
            if name.lower().endswith(".png"):
                stem = name[:-4]
                url = f.get("download_url") or (RAW_BASE + league_path.split("/", 1)[1] + "/" + name)
                # Don't overwrite an earlier league's match
                key = _norm(stem)
                if key not in out:
                    out[key] = url
    print(f"  found {len(out)} unique team logos")
    return out


def main():
    bundle = PredictorBundle.load("leagues")
    teams = sorted(set(bundle.teams))
    print(f"\n{len(teams)} unique club teams in bundle")

    index = fetch_logo_index()
    mapping: dict[str, str] = {}
    misses: list[str] = []

    for t in teams:
        aliased = ALIASES.get(t, t)
        norm = _norm(aliased)
        if norm in index:
            mapping[t] = index[norm]
            continue
        # Try without aliasing
        norm_raw = _norm(t)
        if norm_raw in index:
            mapping[t] = index[norm_raw]
            continue
        # Try substring fuzzy match - find any logo whose stem contains/is contained by ours
        for stem_norm, url in index.items():
            if (len(norm) >= 4 and norm in stem_norm) or (len(stem_norm) >= 4 and stem_norm in norm):
                mapping[t] = url
                break
        else:
            misses.append(t)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nMatched {len(mapping)} of {len(teams)} ({len(mapping)/len(teams)*100:.0f}%)")
    print(f"Saved -> {OUT}")
    if misses:
        print(f"\nMissed ({len(misses)}): {misses[:30]}{'...' if len(misses) > 30 else ''}")


if __name__ == "__main__":
    main()
