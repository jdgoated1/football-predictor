"""Real group-stage draws for recent / upcoming international tournaments.

Team names follow the martj42/international_results dataset conventions.
`resolve_groups()` checks every team is recognised in the bundle and applies
a small alias map for common naming differences.
"""
from __future__ import annotations

from .predictor import PredictorBundle


# Aliases: name we wrote -> name used in the dataset
ALIASES: dict[str, str] = {
    "USA": "United States",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
}


REAL_GROUPS: dict[str, list[list[str]]] = {
    # FIFA World Cup 2026 - drawn December 2025
    "World Cup 2026 (48 teams)": [
        ["Mexico", "South Korea", "South Africa", "Czechia"],
        ["Canada", "Switzerland", "Qatar", "Bosnia-Herzegovina"],
        ["Brazil", "Morocco", "Scotland", "Haiti"],
        ["USA", "Paraguay", "Australia", "Turkiye"],
        ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
        ["Netherlands", "Japan", "Tunisia", "Sweden"],
        ["Belgium", "Iran", "Egypt", "New Zealand"],
        ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
        ["France", "Senegal", "Norway", "Iraq"],
        ["Argentina", "Austria", "Algeria", "Jordan"],
        ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
        ["England", "Croatia", "Panama", "Ghana"],
    ],

    # FIFA World Cup 2022 (Qatar)
    "World Cup 2022 (32 teams)": [
        ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        ["England", "Iran", "USA", "Wales"],
        ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        ["France", "Australia", "Denmark", "Tunisia"],
        ["Spain", "Costa Rica", "Germany", "Japan"],
        ["Belgium", "Canada", "Morocco", "Croatia"],
        ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        ["Portugal", "Ghana", "Uruguay", "South Korea"],
    ],

    # UEFA Euro 2024 (Germany)
    "Euro 2024 (24 teams)": [
        ["Germany", "Scotland", "Hungary", "Switzerland"],
        ["Spain", "Croatia", "Italy", "Albania"],
        ["Slovenia", "Denmark", "Serbia", "England"],
        ["Poland", "Netherlands", "Austria", "France"],
        ["Belgium", "Slovakia", "Romania", "Ukraine"],
        ["Turkey", "Georgia", "Portugal", "Czech Republic"],
    ],

    # AFCON 2023 (held in Ivory Coast, Jan 2024)
    "AFCON (24 teams)": [
        ["Ivory Coast", "Nigeria", "Equatorial Guinea", "Guinea-Bissau"],
        ["Egypt", "Ghana", "Cape Verde", "Mozambique"],
        ["Senegal", "Cameroon", "Guinea", "Gambia"],
        ["Algeria", "Angola", "Burkina Faso", "Mauritania"],
        ["Tunisia", "Mali", "South Africa", "Namibia"],
        ["Morocco", "DR Congo", "Zambia", "Tanzania"],
    ],

    # Copa America 2024 (USA)
    "Copa America (16 teams)": [
        ["Argentina", "Peru", "Chile", "Canada"],
        ["Mexico", "Ecuador", "Venezuela", "Jamaica"],
        ["USA", "Uruguay", "Panama", "Bolivia"],
        ["Brazil", "Colombia", "Paraguay", "Costa Rica"],
    ],
}


def resolve_groups(name: str, bundle: PredictorBundle) -> tuple[list[list[str]], list[str]]:
    """Return (groups_after_aliasing, list_of_unknown_teams)."""
    raw = REAL_GROUPS.get(name)
    if raw is None:
        return [], []
    known = set(bundle.teams)
    resolved: list[list[str]] = []
    unknown: list[str] = []
    for group in raw:
        g_out = []
        for team in group:
            mapped = ALIASES.get(team, team)
            if mapped not in known:
                unknown.append(team)
            g_out.append(mapped)
        resolved.append(g_out)
    return resolved, unknown
