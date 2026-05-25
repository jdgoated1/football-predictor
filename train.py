"""End-to-end training with stacked ensemble + calibration metrics.

Pipeline per scope:
  1. Replay Elo and Pi-rating to attach pre-match rating features
  2. Build form / streak / H2H / rest features
  3. Fit Dixon-Coles on the training portion
  4. Compute DC outcome probabilities for every match (post-hoc on test)
  5. Train XGBoost and LightGBM base models on train slice only
  6. Compute base predictions on the stacking-val slice
  7. Train logistic-regression meta learner on those
  8. Evaluate every model + bookmaker baseline + stacked ensemble on test
  9. Re-fit base models on train+val for production, save bundle
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=UserWarning)

from src.data_fetcher import fetch_all, PROCESSED
from src.xg_data import merge_xg
from src.dixon_coles import DixonColes
from src.elo import EloEngine
from src.features import FEATURE_COLS, build_features
from src.metrics import odds_to_probs, print_report, report

# All possible bookmaker prefixes - we'll use whichever are populated per row
BOOK_PREFIXES = ["b365", "bw", "iw", "ps", "wh", "vc", "lb", "gb", "bs", "sj", "sb"]


def _consensus_odds_probs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row, average the implied probabilities across every available bookmaker.

    Returns (consensus_probs[N,3], has_odds[N], n_books[N]) where:
      - consensus_probs is the per-match average of normalised implied probs
      - has_odds is 1 if at least one bookmaker provided odds, else 0
      - n_books is the count of bookmakers contributing (for diagnostics)
    """
    n = len(df)
    probs_sum = np.zeros((n, 3))
    n_books = np.zeros(n)
    for prefix in BOOK_PREFIXES:
        cols = [f"odds_{prefix}_h", f"odds_{prefix}_d", f"odds_{prefix}_a"]
        if not all(c in df.columns for c in cols):
            continue
        odds = df[cols].values.astype(float)
        valid = ~np.isnan(odds).any(axis=1) & (odds > 1.0).all(axis=1)
        if valid.any():
            inv = 1.0 / odds[valid]
            s = inv.sum(axis=1, keepdims=True)
            probs_sum[valid] += inv / s
            n_books[valid] += 1
    has_odds = (n_books > 0).astype(float)
    consensus = np.full((n, 3), 1/3, dtype=float)
    mask = n_books > 0
    if mask.any():
        consensus[mask] = probs_sum[mask] / n_books[mask, None]
    return consensus, has_odds, n_books


def _odds_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Back-compat wrapper - returns (probs, has_odds) from consensus."""
    probs, has_odds, _ = _consensus_odds_probs(df)
    return probs, has_odds
from src.pi_rating import PiRating
from src.predictor import build_predict_state


def _dc_outcome_probs(dc: DixonColes, df: pd.DataFrame, neutral_col: str = "neutral",
                      max_goals: int = 8) -> np.ndarray:
    """For each row, compute (H, D, A) outcome probabilities under the fitted DC model."""
    out = np.zeros((len(df), 3), dtype=np.float64)
    for i, row in enumerate(df.itertuples(index=False)):
        sm = dc.score_matrix(row.home, row.away,
                             max_goals=max_goals,
                             neutral=bool(getattr(row, neutral_col, False)))
        out[i, 0] = float(np.tril(sm, -1).sum())
        out[i, 1] = float(np.trace(sm))
        out[i, 2] = float(np.triu(sm, 1).sum())
    return out


def _make_xgb(num_class: int = 3):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=num_class,
        eval_metric="mlogloss", tree_method="hist", n_jobs=-1,
    )


def _make_lgb():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=800, max_depth=-1, num_leaves=31, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multiclass", num_class=3,
        n_jobs=-1, verbosity=-1,
    )


def _make_cat():
    """CatBoost - ordered boosting + categorical handling; consistently strong on football."""
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=800, depth=6, learning_rate=0.04,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=42,
        verbose=0, allow_writing_files=False,
    )


def train_scope(name: str, matches: pd.DataFrame):
    print(f"\n=== {name.upper()} ({len(matches):,} matches) ===")
    t0 = time.time()

    # ---- merge xG (leagues only) ----
    if name == "leagues":
        print("Merging Understat xG...")
        matches = merge_xg(matches)

    # ---- ratings ----
    print("Replay Elo + Pi-rating...")
    elo = EloEngine()
    with_elo = elo.run(matches)
    pi = PiRating()
    with_pi = pi.run(with_elo)
    print(f"  Elo top: {[(t, round(r)) for t, r in elo.top(5)]}")
    pi_top = sorted(pi.home_rating.items(), key=lambda kv: -kv[1])[:5]
    print(f"  Pi (home) top: {[(t, round(r, 2)) for t, r in pi_top]}")

    # ---- features ----
    print("Build features...")
    feats = build_features(with_pi)
    # Filter to rows where rolling windows are populated enough
    labeled = feats[feats["home_form3_pts"].notna()].copy()
    labeled = labeled[labeled["away_form3_pts"].notna()].copy()
    print(f"  {len(labeled):,} usable rows (after warm-up)")

    # ---- chronological 70/15/15 split ----
    n = len(labeled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)
    train_df = labeled.iloc[:n_train].copy()
    val_df = labeled.iloc[n_train:n_val].copy()
    test_df = labeled.iloc[n_val:].copy()
    print(f"  Split: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # ---- Dixon-Coles on train ----
    print("Fit Dixon-Coles (train only, time-decayed)...")
    dc_train = train_df[train_df["date"] >= train_df["date"].max() - pd.Timedelta(days=5 * 365)]
    dc_eval = DixonColes(xi=0.0019)
    dc_eval.fit(dc_train, ref_date=train_df["date"].max(), max_iter=300)
    print(f"  home_adv={dc_eval.home_adv:.3f}  rho={dc_eval.rho:.4f}  teams={len(dc_eval.teams)}")

    # DC outcome probabilities for val + test
    print("Computing DC outcome probs for val + test...")
    dc_val_probs = _dc_outcome_probs(dc_eval, val_df)
    dc_test_probs = _dc_outcome_probs(dc_eval, test_df)

    # ---- XGBoost + LightGBM on train ----
    X_train = train_df[FEATURE_COLS].astype(float)
    y_train = train_df["outcome"].astype(int).values
    X_val = val_df[FEATURE_COLS].astype(float)
    y_val = val_df["outcome"].astype(int).values
    X_test = test_df[FEATURE_COLS].astype(float)
    y_test = test_df["outcome"].astype(int).values

    print("Train XGBoost + LightGBM + CatBoost...")
    xgb_model = _make_xgb()
    xgb_model.fit(X_train, y_train, verbose=False)
    lgb_model = _make_lgb()
    lgb_model.fit(X_train, y_train)
    cat_model = _make_cat()
    cat_model.fit(X_train.values, y_train)

    # ---- meta learner on val (isotonic-calibrated logistic regression) ----
    print("Build stacking features + calibrate meta...")
    xgb_val = xgb_model.predict_proba(X_val)
    lgb_val = lgb_model.predict_proba(X_val)
    cat_val = cat_model.predict_proba(X_val.values)

    # Add bookmaker consensus odds (average across all available bookmakers) as
    # meta-learner features. 30% of with-odds val rows are masked back to "no
    # odds" so the meta learns to handle prediction with and without odds.
    odds_val, has_odds_val, n_books_val = _consensus_odds_probs(val_df)
    if has_odds_val.sum() > 0:
        print(f"  Bookmakers per match (avg): {n_books_val[has_odds_val == 1].mean():.1f}")
    odds_test, has_odds_test, _ = _consensus_odds_probs(test_df)
    rng = np.random.default_rng(42)
    mask = (rng.random(len(val_df)) < 0.3) & (has_odds_val == 1)
    odds_val_aug = odds_val.copy()
    has_odds_aug = has_odds_val.copy()
    odds_val_aug[mask] = 1/3
    has_odds_aug[mask] = 0.0

    meta_X_val = np.hstack([xgb_val, lgb_val, cat_val, dc_val_probs,
                            odds_val_aug, has_odds_aug.reshape(-1, 1)])
    print(f"  Val odds coverage: {has_odds_val.mean()*100:.0f}% (after masking: {has_odds_aug.mean()*100:.0f}%)")

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    meta_base = LogisticRegression(C=1.0, max_iter=2000)
    meta_model = CalibratedClassifierCV(meta_base, method="isotonic", cv=3)
    meta_model.fit(meta_X_val, y_val)

    # ---- evaluation on test (both modes: with available odds AND with all odds masked) ----
    print("\nEvaluating on test...")
    xgb_test = xgb_model.predict_proba(X_test)
    lgb_test = lgb_model.predict_proba(X_test)
    cat_test = cat_model.predict_proba(X_test.values)
    no_odds_test = np.tile([1/3, 1/3, 1/3], (len(X_test), 1))
    no_has_odds = np.zeros((len(X_test), 1))

    meta_X_test_no = np.hstack([xgb_test, lgb_test, cat_test, dc_test_probs,
                                no_odds_test, no_has_odds])
    meta_X_test_yes = np.hstack([xgb_test, lgb_test, cat_test, dc_test_probs,
                                 odds_test, has_odds_test.reshape(-1, 1)])
    stack_no = meta_model.predict_proba(meta_X_test_no)
    stack_yes = meta_model.predict_proba(meta_X_test_yes)

    reports = []
    for label, probs in [("Dixon-Coles", dc_test_probs),
                         ("XGBoost", xgb_test),
                         ("LightGBM", lgb_test),
                         ("CatBoost", cat_test),
                         ("STACKED (no odds)", stack_no),
                         ("STACKED (+ odds)", stack_yes)]:
        rep = report(y_test, probs, label)
        print_report(rep)
        reports.append(rep)

    # ---- bookmaker baselines (B365 alone vs multi-book consensus) ----
    bk_b365_report = bk_cons_report = None
    if "odds_b365_h" in test_df.columns:
        b365 = test_df[["odds_b365_h", "odds_b365_d", "odds_b365_a"]].values
        valid_b = ~np.isnan(b365).any(axis=1)
        if valid_b.sum() > 100:
            bk_b365_probs = odds_to_probs(b365[valid_b, 0], b365[valid_b, 1], b365[valid_b, 2])
            bk_b365_report = report(y_test[valid_b], bk_b365_probs, "B365 only")
            print(f"\nOn {valid_b.sum():,} matches with B365 odds:")
            print_report(bk_b365_report)
            reports.append(bk_b365_report)
    valid_c = has_odds_test == 1
    if valid_c.sum() > 100:
        bk_cons_report = report(y_test[valid_c.astype(bool)], odds_test[valid_c.astype(bool)],
                                "Consensus odds")
        stack_no_subset = report(y_test[valid_c.astype(bool)],
                                 stack_no[valid_c.astype(bool)], "STACKED no-odds (same)")
        stack_yes_subset = report(y_test[valid_c.astype(bool)],
                                  stack_yes[valid_c.astype(bool)], "STACKED +odds (same)")
        print(f"\nOn {int(valid_c.sum()):,} matches with multi-book consensus odds:")
        print_report(bk_cons_report)
        print_report(stack_no_subset)
        print_report(stack_yes_subset)
        reports.extend([bk_cons_report, stack_no_subset, stack_yes_subset])

    # ---- refit on full data (train + val) for production ----
    print("\nRefit base models on train+val for production...")
    Xall = pd.concat([X_train, X_val], ignore_index=True)
    yall = np.concatenate([y_train, y_val])
    xgb_final = _make_xgb()
    xgb_final.fit(Xall, yall, verbose=False)
    lgb_final = _make_lgb()
    lgb_final.fit(Xall, yall)
    cat_final = _make_cat()
    cat_final.fit(Xall.values, yall)

    # Re-fit DC on all-but-test data, using full date range (better attack/defence estimates)
    print("Refit Dixon-Coles on train+val...")
    refit_df = pd.concat([train_df, val_df], ignore_index=True)
    dc_final_train = refit_df[refit_df["date"] >= refit_df["date"].max() - pd.Timedelta(days=5 * 365)]
    dc_final = DixonColes(xi=0.0019)
    dc_final.fit(dc_final_train, ref_date=refit_df["date"].max(), max_iter=300)

    # ---- build bundle ----
    print("Building prediction bundle...")
    bundle = build_predict_state(
        elo, pi, dc_final, xgb_final, lgb_final, cat_final, meta_model,
        matches, name, metrics={"reports": reports},
    )
    p = bundle.save()
    print(f"Saved -> {p}    ({time.time() - t0:.1f}s total)\n")
    return bundle


def main():
    leagues_path = PROCESSED / "leagues.parquet"
    intl_path = PROCESSED / "internationals.parquet"
    if leagues_path.exists() and intl_path.exists():
        print("Using cached parquet data.")
        leagues = pd.read_parquet(leagues_path)
        intl = pd.read_parquet(intl_path)
    else:
        leagues, intl = fetch_all()

    if len(leagues):
        train_scope("leagues", leagues)
    if len(intl):
        train_scope("internationals", intl)
    print("Done.")


if __name__ == "__main__":
    main()
