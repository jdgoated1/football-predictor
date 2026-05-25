"""Dixon-Coles (1997) time-weighted bivariate Poisson model.

For each team we fit an attack parameter and a defence parameter. The model
predicts the expected goal rate for each side and applies a low-score
correlation correction to inflate draw probability for 0-0 / 1-1 etc.

Reference: Dixon, M. J., & Coles, S. G. (1997). Modelling Association Football
Scores and Inefficiencies in the Football Betting Market.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


@dataclass
class DixonColes:
    xi: float = 0.0019       # time decay per day (~half life of ~1 year)
    home_adv: float = 0.25   # log-scale home advantage
    rho: float = -0.05       # low-score correlation
    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    fitted: bool = False

    # ---------- fitting ----------

    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp | None = None,
            max_iter: int = 200) -> "DixonColes":
        """Fit on a long DataFrame of matches (one league at a time gives best results,
        but pooling top-5 still works because attack/defence parameters are per team)."""
        df = df.dropna(subset=["home", "away", "home_goals", "away_goals"]).copy()
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)
        if ref_date is None:
            ref_date = df["date"].max()
        days = (ref_date - df["date"]).dt.days.clip(lower=0).to_numpy()
        weights = np.exp(-self.xi * days)

        teams = sorted(set(df["home"]).union(df["away"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        h_idx = df["home"].map(idx).to_numpy()
        a_idx = df["away"].map(idx).to_numpy()
        hg = df["home_goals"].to_numpy()
        ag = df["away_goals"].to_numpy()

        # params: [attack_1..n-1, defence_1..n-1, home_adv, rho]
        # Last team is the reference (attack sum = 0 constraint baked in)
        def unpack(x):
            atk = np.concatenate([x[:n - 1], [-x[:n - 1].sum()]])
            dfn = np.concatenate([x[n - 1:2 * n - 2], [-x[n - 1:2 * n - 2].sum()]])
            return atk, dfn, x[-2], x[-1]

        def neg_log_lik(x):
            atk, dfn, hadv, rho = unpack(x)
            lam = np.exp(atk[h_idx] - dfn[a_idx] + hadv)
            mu = np.exp(atk[a_idx] - dfn[h_idx])
            # Cap to avoid overflow on bad params during early iters
            lam = np.clip(lam, 1e-6, 12)
            mu = np.clip(mu, 1e-6, 12)
            # Poisson log-pmf
            ll_h = hg * np.log(lam) - lam - _log_factorial(hg)
            ll_a = ag * np.log(mu) - mu - _log_factorial(ag)
            # Dixon-Coles low-score correction (only matters when both <=1)
            tau = np.ones_like(lam)
            mask00 = (hg == 0) & (ag == 0)
            mask01 = (hg == 0) & (ag == 1)
            mask10 = (hg == 1) & (ag == 0)
            mask11 = (hg == 1) & (ag == 1)
            tau = np.where(mask00, 1 - lam * mu * rho, tau)
            tau = np.where(mask01, 1 + lam * rho, tau)
            tau = np.where(mask10, 1 + mu * rho, tau)
            tau = np.where(mask11, 1 - rho, tau)
            tau = np.clip(tau, 1e-9, None)
            log_tau = np.log(tau)
            ll = ll_h + ll_a + log_tau
            return -np.sum(weights * ll)

        x0 = np.concatenate([
            np.zeros(n - 1),       # attack (n-1)
            np.zeros(n - 1),       # defence (n-1)
            [self.home_adv],
            [self.rho],
        ])
        bounds = [(-3, 3)] * (n - 1) + [(-3, 3)] * (n - 1) + [(0.0, 1.0), (-0.2, 0.2)]
        res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": max_iter, "ftol": 1e-7})
        atk, dfn, hadv, rho = unpack(res.x)
        self.teams = teams
        self.attack = {t: float(atk[i]) for i, t in enumerate(teams)}
        self.defence = {t: float(dfn[i]) for i, t in enumerate(teams)}
        self.home_adv = float(hadv)
        self.rho = float(rho)
        self.fitted = True
        return self

    # ---------- prediction ----------

    def expected_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        a_h = self.attack.get(home, 0.0)
        d_h = self.defence.get(home, 0.0)
        a_a = self.attack.get(away, 0.0)
        d_a = self.defence.get(away, 0.0)
        hadv = 0.0 if neutral else self.home_adv
        lam = float(np.exp(a_h - d_a + hadv))
        mu = float(np.exp(a_a - d_h))
        return lam, mu

    def score_matrix(self, home: str, away: str, max_goals: int = 8,
                     neutral: bool = False) -> np.ndarray:
        lam, mu = self.expected_goals(home, away, neutral=neutral)
        rng = np.arange(max_goals + 1)
        ph = poisson.pmf(rng, lam)
        pa = poisson.pmf(rng, mu)
        m = np.outer(ph, pa)
        # Dixon-Coles low-score correction
        m[0, 0] *= 1 - lam * mu * self.rho
        m[0, 1] *= 1 + lam * self.rho
        m[1, 0] *= 1 + mu * self.rho
        m[1, 1] *= 1 - self.rho
        # Renormalise
        m = np.clip(m, 0, None)
        s = m.sum()
        if s > 0:
            m /= s
        return m

    def outcome_probs(self, home: str, away: str, neutral: bool = False) -> dict[str, float]:
        m = self.score_matrix(home, away, neutral=neutral)
        p_home = float(np.tril(m, -1).sum())
        p_draw = float(np.trace(m))
        p_away = float(np.triu(m, 1).sum())
        return {"H": p_home, "D": p_draw, "A": p_away}


_LOG_FACT_CACHE: dict[int, float] = {}


def _log_factorial(arr):
    arr = np.asarray(arr, dtype=int)
    out = np.zeros_like(arr, dtype=float)
    unique = np.unique(arr)
    for u in unique:
        if u not in _LOG_FACT_CACHE:
            from math import lgamma
            _LOG_FACT_CACHE[int(u)] = lgamma(int(u) + 1)
        out[arr == u] = _LOG_FACT_CACHE[int(u)]
    return out
