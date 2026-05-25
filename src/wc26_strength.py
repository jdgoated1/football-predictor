"""WC 2026 team strength priors.

For each of the 48 qualified teams, holds:
  - rank: ESPN squad ranking (1 = strongest, 48 = weakest)
  - odds: average bookmaker decimal odds to win the tournament outright
  - value_m: Transfermarkt total squad market value in € million (top teams only)

When predicting a WC 2026 match, we blend the historical-form-based model
output with a market-derived prior so the prediction reflects current squad
quality even when the model's recent-match data underestimates it (e.g. teams
that don't play many competitive matches outside major tournaments).

Sources (all public, May 2026):
  - ESPN: "2026 World Cup squads ranked: All 48 national teams"
  - RotoWire: 2026 World Cup Winner Odds (DraftKings consensus)
  - sportsorca.com / Transfermarkt: top-10 squad values
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np


# Per-team WC 2026 priors. Team names follow the bundle (martj42 dataset) naming.
WC_2026_DATA: dict[str, dict] = {
    "Spain":                  {"rank": 2,  "odds": 5.75,   "value_m": 920},
    "France":                 {"rank": 1,  "odds": 6.00,   "value_m": 1280},
    "England":                {"rank": 3,  "odds": 7.50,   "value_m": 1300},
    "Brazil":                 {"rank": 4,  "odds": 9.00,   "value_m": 1000},
    "Argentina":              {"rank": 7,  "odds": 10.00,  "value_m": 570},
    "Portugal":               {"rank": 5,  "odds": 11.00,  "value_m": 850},
    "Germany":                {"rank": 8,  "odds": 15.00,  "value_m": 850},
    "Netherlands":            {"rank": 6,  "odds": 21.00,  "value_m": 720},
    "Norway":                 {"rank": 9,  "odds": 31.00},
    "Belgium":                {"rank": 10, "odds": 36.00},
    "Senegal":                {"rank": 11, "odds": 91.00},
    "Turkey":                 {"rank": 12, "odds": 101.00, "value_m": 460},
    "Morocco":                {"rank": 13, "odds": 51.00},
    "Colombia":               {"rank": 14, "odds": 41.00},
    "Uruguay":                {"rank": 15, "odds": 66.00},
    "Ecuador":                {"rank": 16, "odds": 81.00},
    "Switzerland":            {"rank": 17, "odds": 66.00},
    "Croatia":                {"rank": 18, "odds": 81.00},
    "Ivory Coast":            {"rank": 19, "odds": 251.00},
    "Japan":                  {"rank": 20, "odds": 66.00},
    "Sweden":                 {"rank": 21, "odds": 101.00},
    "United States":          {"rank": 22, "odds": 61.00},
    "Austria":                {"rank": 23, "odds": 151.00},
    "Mexico":                 {"rank": 24, "odds": 81.00},
    "Algeria":                {"rank": 25, "odds": 401.00},
    "Scotland":               {"rank": 26, "odds": 201.00},
    "Paraguay":               {"rank": 27, "odds": 301.00},
    "Czech Republic":         {"rank": 28, "odds": 251.00},
    "Canada":                 {"rank": 29, "odds": 201.00},
    "South Korea":            {"rank": 30, "odds": 501.00},
    "DR Congo":               {"rank": 31, "odds": 1001.00},
    "Australia":              {"rank": 32, "odds": 501.00},
    "Egypt":                  {"rank": 33, "odds": 301.00},
    "Uzbekistan":             {"rank": 34, "odds": 1001.00},
    "Ghana":                  {"rank": 35, "odds": 1001.00},
    "Bosnia and Herzegovina": {"rank": 36, "odds": 501.00},
    "Panama":                 {"rank": 37, "odds": 1001.00},
    "Iran":                   {"rank": 38, "odds": 1001.00},
    "Jordan":                 {"rank": 39, "odds": 2001.00, "value_m": 16},
    "Tunisia":                {"rank": 40, "odds": 1001.00},
    "New Zealand":            {"rank": 41, "odds": 1001.00},
    "Haiti":                  {"rank": 42, "odds": 2001.00},
    "Saudi Arabia":           {"rank": 43, "odds": 2001.00},
    "Iraq":                   {"rank": 44, "odds": 2001.00},
    "South Africa":           {"rank": 45, "odds": 2001.00},
    "Cape Verde":             {"rank": 46, "odds": 2001.00},
    "Curaçao":                {"rank": 47, "odds": 5001.00},
    "Qatar":                  {"rank": 48, "odds": 5001.00},
}


def market_strength(team: str) -> float | None:
    """Return a log-odds strength score for a WC 2026 team (higher = stronger)."""
    data = WC_2026_DATA.get(team)
    if not data or "odds" not in data:
        return None
    return -math.log(data["odds"])


def wc_outcome_prior(home: str, away: str, draw_floor: float = 0.24) -> dict | None:
    """Market-implied H/D/A probabilities for a WC 2026 matchup.

    Both teams' outright-winner odds are converted to log-strength scores; the
    difference goes through a logistic to give a win probability share. A flat
    draw floor (typical for tournament football) takes some of the win share.

    Returns None if either team isn't a WC 2026 team or has no odds.
    """
    sh = market_strength(home)
    sa = market_strength(away)
    if sh is None or sa is None:
        return None
    # Strength scale tuned so that ~1-point diff in log-odds ~ +20% win prob
    diff = sh - sa
    p_h_market = 1.0 / (1.0 + math.exp(-diff))
    p_a_market = 1.0 - p_h_market
    win_share = 1.0 - draw_floor
    return {
        "H": float(p_h_market * win_share),
        "D": float(draw_floor),
        "A": float(p_a_market * win_share),
    }


def apply_wc_prior_to_prediction(pred: dict, home: str, away: str,
                                 blend: float = 0.30) -> dict:
    """Take a prediction dict (from bundle.predict()) and blend its outcome
    with the WC 2026 market prior. Re-scales the score matrix accordingly.

    `blend`: 0 = no adjustment, 1 = pure market prior. Default 0.30."""
    if blend <= 0:
        return pred
    prior = wc_outcome_prior(home, away)
    if prior is None:
        return pred
    pred = deepcopy(pred)
    original = pred["outcome"]
    blended = {k: (1 - blend) * original[k] + blend * prior[k] for k in "HDA"}
    s = sum(blended.values())
    blended = {k: v / s for k, v in blended.items()}
    pred["outcome_pre_wc"] = original
    pred["wc_market_prior"] = prior
    pred["outcome"] = blended

    # Re-scale the score matrix so each H/D/A region sums to the new outcome
    sm = pred["score_matrix"]
    idx = np.indices(sm.shape)
    regions = [
        (idx[0] > idx[1], blended["H"]),
        (idx[0] == idx[1], blended["D"]),
        (idx[0] < idx[1], blended["A"]),
    ]
    for region, target in regions:
        cur = sm[region].sum()
        if cur > 1e-9:
            sm[region] *= target / cur
    pred["score_matrix"] = sm
    # Recompute most_likely and top_scores from rescaled matrix
    flat = [(int(i), int(j), float(sm[i, j])) for i in range(sm.shape[0])
            for j in range(sm.shape[1])]
    flat.sort(key=lambda x: -x[2])
    pred["top_scores"] = flat[:8]
    pred["most_likely"] = (flat[0][0], flat[0][1])
    return pred


def fmt_uses_wc_prior(fmt_name: str | None) -> bool:
    """Whether to apply WC 2026 prior for this tournament format name."""
    return fmt_name == "World Cup 2026 (48 teams)"
