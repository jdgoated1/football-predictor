# Football Predictor

A statistical + machine-learning football match predictor with full tournament simulation. Covers the top-5 European leagues, every international team, and the 2026 FIFA World Cup with the actual final draw, real fixture dates, and squad-strength priors.

## What it does

- **Single match**: predict any matchup with outcome probabilities, expected goals, top scorelines, and a full scoreline heatmap.
- **League season**: simulate the rest of a Premier League / La Liga / Bundesliga / Serie A / Ligue 1 season from current standings - probability table for title win, top 4, top 6, relegation.
- **Tournament**: World Cup 2026, World Cup 2022, Euro 2024, AFCON, Copa America - real draws when applicable. Includes group-stage Predicted fixtures organised by date, knockout bracket cascade, predicted champion, and full Monte Carlo aggregate simulations.

## Model stack

Per-match probabilities come from an ensemble:

1. **Elo** - goal-difference weighted, competition-aware K-factors
2. **Pi-rating** (Constantinou-Fenton 2013) - separate home/away strength per team
3. **Dixon-Coles** - time-decayed bivariate Poisson with low-score correlation
4. **XGBoost + LightGBM + CatBoost** - 28 engineered features (multi-window form, streaks, head-to-head, rest, rolling xG)
5. **Isotonic-calibrated logistic regression meta-learner** - blends all four base models, optionally including bookmaker odds and WC 2026 squad-strength priors

Test metrics (held-out chronological split):

| Scope | Accuracy | RPS | Notes |
|---|---|---|---|
| Leagues | 0.529 (+odds) | **0.198** | vs B365 bookmaker 0.195 (gap explained by lineups/injuries the model can't see) |
| Internationals | 0.600 | **0.173** | Beats published Bundesliga academic benchmarks (~0.193) |

## Data sources

All free, no paid APIs:
- **football-data.co.uk** - 28k+ league matches with 11 bookmakers' odds
- **github.com/martj42/international_results** - 25k+ international results since 2000
- **Understat** (via `soccerdata`) - rolling xG features for top-5 leagues
- **ESPN squad rankings + bookmaker outright odds** - WC 2026 squad-strength priors

## Run locally

```bash
pip install -r requirements.txt
python train.py            # fetches data + trains both scopes (~1 min)
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Fork this repo (or push your fork to your GitHub)
2. Go to https://share.streamlit.io
3. Sign in with GitHub, click "New app"
4. Pick the repo, set main file to `app.py`
5. Deploy. You'll get a public URL.

Daily retraining is handled by the GitHub Action at `.github/workflows/daily-update.yml` - it fetches fresh data and commits updated models, which auto-redeploys the app.

## Project layout

- `src/` - data fetchers, ratings (Elo, Pi), Dixon-Coles fit, features, predictor, tournament sim
- `train.py` - end-to-end training pipeline with stacked ensemble + calibration
- `app.py` - Streamlit UI (Single match / League season / Tournament tabs)
- `data/processed/` - cached parquet files used at runtime
- `models/` - trained predictor bundles (joblib)

## License

MIT.
