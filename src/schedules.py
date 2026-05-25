"""Real-world fixture schedules for tournaments where they exist.

When a schedule is present in `SCHEDULES`, the prediction code uses the
published fixture list (with actual dates) instead of a generated round-robin.

Team names match the conventions of the martj42 international_results dataset
(post-alias resolution). The `ALIASES` dict in `real_groups.py` handles common
variants like "USA" -> "United States".
"""
from __future__ import annotations

# 2026 FIFA World Cup group stage - all 72 fixtures with real dates
# Source: officially published FIFA / Sky Sports schedule
WC_2026_GROUP_FIXTURES: list[tuple[str, str, str]] = [
    # date, home, away
    ("2026-06-11", "Mexico", "South Africa"),
    ("2026-06-12", "South Korea", "Czech Republic"),
    ("2026-06-12", "Canada", "Bosnia and Herzegovina"),
    ("2026-06-13", "United States", "Paraguay"),
    ("2026-06-13", "Qatar", "Switzerland"),
    ("2026-06-13", "Brazil", "Morocco"),
    ("2026-06-14", "Haiti", "Scotland"),
    ("2026-06-14", "Australia", "Turkey"),
    ("2026-06-14", "Germany", "Curaçao"),
    ("2026-06-14", "Netherlands", "Japan"),
    ("2026-06-15", "Ivory Coast", "Ecuador"),
    ("2026-06-15", "Sweden", "Tunisia"),
    ("2026-06-15", "Spain", "Cape Verde"),
    ("2026-06-15", "Belgium", "Egypt"),
    ("2026-06-15", "Saudi Arabia", "Uruguay"),
    ("2026-06-16", "Iran", "New Zealand"),
    ("2026-06-16", "France", "Senegal"),
    ("2026-06-16", "Iraq", "Norway"),
    ("2026-06-17", "Argentina", "Algeria"),
    ("2026-06-17", "Austria", "Jordan"),
    ("2026-06-17", "Portugal", "DR Congo"),
    ("2026-06-17", "England", "Croatia"),
    ("2026-06-18", "Ghana", "Panama"),
    ("2026-06-18", "Uzbekistan", "Colombia"),
    # ---- Matchday 2 ----
    ("2026-06-18", "Czech Republic", "South Africa"),
    ("2026-06-18", "Switzerland", "Bosnia and Herzegovina"),
    ("2026-06-18", "Canada", "Qatar"),
    ("2026-06-19", "Mexico", "South Korea"),
    ("2026-06-19", "United States", "Australia"),
    ("2026-06-19", "Scotland", "Morocco"),
    ("2026-06-20", "Brazil", "Haiti"),
    ("2026-06-20", "Turkey", "Paraguay"),
    ("2026-06-20", "Netherlands", "Sweden"),
    ("2026-06-20", "Germany", "Ivory Coast"),
    ("2026-06-21", "Ecuador", "Curaçao"),
    ("2026-06-21", "Tunisia", "Japan"),
    ("2026-06-21", "Spain", "Saudi Arabia"),
    ("2026-06-21", "Belgium", "Iran"),
    ("2026-06-21", "Uruguay", "Cape Verde"),
    ("2026-06-22", "New Zealand", "Egypt"),
    ("2026-06-22", "Argentina", "Austria"),
    ("2026-06-22", "France", "Iraq"),
    ("2026-06-23", "Norway", "Senegal"),
    ("2026-06-23", "Jordan", "Algeria"),
    ("2026-06-23", "Portugal", "Uzbekistan"),
    ("2026-06-23", "England", "Ghana"),
    ("2026-06-24", "Panama", "Croatia"),
    ("2026-06-24", "Colombia", "DR Congo"),
    # ---- Matchday 3 (groups close simultaneously per group, FIFA tradition) ----
    ("2026-06-24", "Switzerland", "Canada"),
    ("2026-06-24", "Bosnia and Herzegovina", "Qatar"),
    ("2026-06-24", "Morocco", "Haiti"),
    ("2026-06-24", "Scotland", "Brazil"),
    ("2026-06-25", "South Africa", "South Korea"),
    ("2026-06-25", "Czech Republic", "Mexico"),
    ("2026-06-25", "Curaçao", "Ivory Coast"),
    ("2026-06-25", "Ecuador", "Germany"),
    ("2026-06-26", "Tunisia", "Netherlands"),
    ("2026-06-26", "Japan", "Sweden"),
    ("2026-06-26", "Turkey", "United States"),
    ("2026-06-26", "Paraguay", "Australia"),
    ("2026-06-26", "Norway", "France"),
    ("2026-06-26", "Senegal", "Iraq"),
    ("2026-06-27", "Cape Verde", "Saudi Arabia"),
    ("2026-06-27", "Uruguay", "Spain"),
    ("2026-06-27", "New Zealand", "Belgium"),
    ("2026-06-27", "Egypt", "Iran"),
    ("2026-06-27", "Panama", "England"),
    ("2026-06-27", "Croatia", "Ghana"),
    ("2026-06-28", "Colombia", "Portugal"),
    ("2026-06-28", "DR Congo", "Uzbekistan"),
    ("2026-06-28", "Algeria", "Austria"),
    ("2026-06-28", "Jordan", "Argentina"),
]


# Date ranges for each knockout round
# Round name -> (start, end) inclusive
WC_2026_KO_RANGES: dict[str, tuple[str, str]] = {
    "Round of 32": ("2026-06-28", "2026-07-03"),
    "Round of 16": ("2026-07-04", "2026-07-07"),
    "Quarter-final": ("2026-07-09", "2026-07-12"),
    "Semi-final": ("2026-07-14", "2026-07-15"),
    "Final": ("2026-07-19", "2026-07-19"),
}


SCHEDULES: dict[str, dict] = {
    "World Cup 2026 (48 teams)": {
        "group_fixtures": WC_2026_GROUP_FIXTURES,
        "ko_ranges": WC_2026_KO_RANGES,
    },
}


def get_schedule(fmt_name: str) -> dict | None:
    return SCHEDULES.get(fmt_name)


def ko_date(fmt_name: str, round_name: str) -> str | None:
    """Return the start date of a KO round (or None if unknown)."""
    sched = SCHEDULES.get(fmt_name)
    if not sched:
        return None
    ranges = sched.get("ko_ranges", {})
    if round_name in ranges:
        return ranges[round_name][0]
    # Try fuzzy matching round names
    for key, (start, _) in ranges.items():
        if key.lower().startswith(round_name.lower()) or round_name.lower().startswith(key.lower()):
            return start
    return None
