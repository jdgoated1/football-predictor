"""Pi-rating system (Constantinou & Fenton, 2013).

Each team has TWO ratings: one for when they play at home, one for when they
play away. Updates happen on each match using the discrepancy between observed
and expected goal difference. Cross-rating updates ensure both home and away
ratings learn from every match (with reduced weight).

Reference: Constantinou, A. C., & Fenton, N. E. (2013). Determining the level
of ability of football teams by dynamic ratings based on the relative
discrepancies in scores between adjacent divisions. JQAS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PiRating:
    # Learning rates
    lam: float = 0.054        # primary side rating update
    gamma: float = 0.79       # cross-rating update fraction (paper: 0.6-0.9)
    # Goal-diff -> rating diff transformation (paper uses b=10, c=3)
    b: float = 10.0
    c: float = 3.0
    init: float = 0.0
    home_rating: dict[str, float] = field(default_factory=dict)
    away_rating: dict[str, float] = field(default_factory=dict)

    def get_h(self, team: str) -> float:
        return self.home_rating.get(team, self.init)

    def get_a(self, team: str) -> float:
        return self.away_rating.get(team, self.init)

    def expected_gd(self, home: str, away: str, neutral: bool = False) -> float:
        # Convert each rating into an expected goal contribution
        if neutral:
            # Average the two ratings for each team when there's no home venue
            r_h_home = 0.5 * (self.get_h(home) + self.get_a(home))
            r_a_away = 0.5 * (self.get_h(away) + self.get_a(away))
        else:
            r_h_home = self.get_h(home)
            r_a_away = self.get_a(away)
        lam_h = self._rating_to_goals(r_h_home)
        lam_a = self._rating_to_goals(r_a_away)
        return lam_h - lam_a

    def _rating_to_goals(self, r: float) -> float:
        # b^(|r|/c) - 1, signed
        return math.copysign(self.b ** (abs(r) / self.c) - 1, r)

    def update(self, home: str, away: str, hg: int, ag: int,
               neutral: bool = False) -> None:
        expected = self.expected_gd(home, away, neutral=neutral)
        actual = hg - ag
        # Saturate actual goal difference to limit influence of blowouts
        sat = math.copysign(math.log1p(abs(actual) * 1.0), actual)
        err = sat - math.copysign(math.log1p(abs(expected) * 1.0), expected)

        if neutral:
            # Update both ratings equally for both teams
            self.home_rating[home] = self.get_h(home) + self.lam * err
            self.away_rating[home] = self.get_a(home) + self.lam * err
            self.home_rating[away] = self.get_h(away) - self.lam * err
            self.away_rating[away] = self.get_a(away) - self.lam * err
        else:
            # Primary update on the relevant side
            self.home_rating[home] = self.get_h(home) + self.lam * err
            self.away_rating[away] = self.get_a(away) - self.lam * err
            # Cross-update at smaller weight
            self.away_rating[home] = self.get_a(home) + self.lam * self.gamma * err
            self.home_rating[away] = self.get_h(away) - self.lam * self.gamma * err

    def run(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Replay matches chronologically, attach pre-match Pi-rating features."""
        df = matches.sort_values("date").reset_index(drop=True).copy()
        pi_h_pre, pi_a_pre, pi_h_home, pi_a_away, pi_egd = [], [], [], [], []
        for row in df.itertuples(index=False):
            neutral = bool(getattr(row, "neutral", False))
            pi_h_home.append(self.get_h(row.home))
            pi_a_away.append(self.get_a(row.away))
            pi_h_pre.append(0.5 * (self.get_h(row.home) + self.get_a(row.home)))
            pi_a_pre.append(0.5 * (self.get_h(row.away) + self.get_a(row.away)))
            pi_egd.append(self.expected_gd(row.home, row.away, neutral=neutral))
            self.update(row.home, row.away, int(row.home_goals), int(row.away_goals),
                        neutral=neutral)
        df["pi_home_avg"] = pi_h_pre
        df["pi_away_avg"] = pi_a_pre
        df["pi_home_home"] = pi_h_home   # the home team's home-strength rating
        df["pi_away_away"] = pi_a_away   # the away team's away-strength rating
        df["pi_expected_gd"] = pi_egd
        return df
