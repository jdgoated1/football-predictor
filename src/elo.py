"""Elo rating engine with goal-difference weighting and competition multipliers.

Inspired by the World Football Elo / FIFA Men's ranking models. Maintains a
single rolling rating per team. Internationals and clubs are kept in separate
namespaces because they never play each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

# Competition weights - roughly mirrors FIFA's K-factor importance scale
COMP_K = {
    "Premier League": 32, "La Liga": 32, "Bundesliga": 32,
    "Serie A": 32, "Ligue 1": 32,
    "FIFA World Cup": 60, "UEFA Euro": 50, "Copa America": 50,
    "African Cup of Nations": 50, "AFC Asian Cup": 50,
    "FIFA World Cup qualification": 40,
    "UEFA Euro qualification": 35, "Copa America qualification": 35,
    "African Cup of Nations qualification": 30, "AFC Asian Cup qualification": 30,
    "UEFA Nations League": 30,
    "Friendly": 15,
}
DEFAULT_K = 25
HOME_ADV = 65          # Elo points granted to home side
INIT_RATING = 1500.0


@dataclass
class EloEngine:
    home_adv: float = HOME_ADV
    init: float = INIT_RATING
    ratings: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.init)

    @staticmethod
    def _gd_multiplier(gd: int) -> float:
        # World Football Elo's classic goal-difference index
        gd = abs(gd)
        if gd <= 1:
            return 1.0
        if gd == 2:
            return 1.5
        return (11 + gd) / 8.0

    @staticmethod
    def expected(r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))

    def update(self, home: str, away: str, hg: int, ag: int,
               competition: str = "", neutral: bool = False) -> tuple[float, float]:
        k = COMP_K.get(competition, DEFAULT_K)
        h_eff = 0.0 if neutral else self.home_adv
        ra = self.rating(home) + h_eff
        rb = self.rating(away)
        ea = self.expected(ra, rb)
        if hg > ag:
            sa = 1.0
        elif hg < ag:
            sa = 0.0
        else:
            sa = 0.5
        g = self._gd_multiplier(hg - ag)
        delta = k * g * (sa - ea)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta
        return self.ratings[home], self.ratings[away]

    def run(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Replay every match in chronological order, attach pre-match Elo cols."""
        df = matches.sort_values("date").reset_index(drop=True).copy()
        elo_home_pre = []
        elo_away_pre = []
        for row in df.itertuples(index=False):
            elo_home_pre.append(self.rating(row.home))
            elo_away_pre.append(self.rating(row.away))
            self.update(row.home, row.away, int(row.home_goals), int(row.away_goals),
                        competition=getattr(row, "competition", ""),
                        neutral=bool(getattr(row, "neutral", False)))
        df["elo_home_pre"] = elo_home_pre
        df["elo_away_pre"] = elo_away_pre
        df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]
        return df

    def top(self, n: int = 20) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda kv: kv[1], reverse=True)[:n]
