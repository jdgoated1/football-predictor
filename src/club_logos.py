"""Club badge URLs cached from TheSportsDB.

The JSON file is built by `scripts/fetch_club_logos.py`. At runtime we just
load the mapping and return URLs for the teams we have data for.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGOS_JSON = ROOT / "data" / "processed" / "club_logos.json"

_CACHE: dict[str, str] | None = None


def _load() -> dict[str, str]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if LOGOS_JSON.exists():
        try:
            _CACHE = json.loads(LOGOS_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _CACHE = {}
    else:
        _CACHE = {}
    return _CACHE


def logo(team: str) -> str | None:
    """Return badge URL for a club team, or None if not available."""
    return _load().get(team)


def has_logos() -> bool:
    return bool(_load())
