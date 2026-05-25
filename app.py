"""Streamlit web UI for the football score predictor."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.predictor import PredictorBundle
from src.real_groups import REAL_GROUPS, resolve_groups
from src.tournament import (
    FORMATS, current_state_for, get_actual_results, predict_group_fixtures,
    predict_modal_bracket, sample_one_tournament, simulate_knockout, simulate_league,
    top_n_by_elo,
)

st.set_page_config(page_title="Football Score Predictor", layout="wide",
                   page_icon=":soccer:")

# ============================================================================
# Sidebar (shared across tabs)
# ============================================================================
st.sidebar.title("Football predictor")
st.sidebar.caption("Elo + Dixon-Coles + XGBoost ensemble")

available_scopes = []
for name in ["leagues", "internationals"]:
    if (ROOT / "models" / f"{name}.joblib").exists():
        available_scopes.append(name)

if not available_scopes:
    st.error("No trained models found. Run `python train.py` first.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_bundle(name: str, mtime: float) -> PredictorBundle:
    return PredictorBundle.load(name)


def get_bundle(name: str) -> PredictorBundle:
    p = ROOT / "models" / f"{name}.joblib"
    return load_bundle(name, p.stat().st_mtime)


# Show "model updated" timestamps + the universal update button
for s in available_scopes:
    p = ROOT / "models" / f"{s}.joblib"
    st.sidebar.caption(f"{s.title()} model: "
                       f"{pd.Timestamp(p.stat().st_mtime, unit='s'):%Y-%m-%d %H:%M}")

if st.sidebar.button("Update now",
                     help="Re-download latest match data and retrain (~30 sec)"):
    progress = st.sidebar.empty()
    progress.info("Updating... please wait")
    try:
        result = subprocess.run([sys.executable, "update.py"], cwd=str(ROOT),
                                capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            progress.success("Updated. Reloading...")
            load_bundle.clear()
            st.rerun()
        else:
            progress.error(f"Update failed: {result.stderr[-300:] or result.stdout[-300:]}")
    except subprocess.TimeoutExpired:
        progress.error("Update timed out (>5 min)")

st.sidebar.divider()

# ============================================================================
# Tabs
# ============================================================================
tab_match, tab_league, tab_cup = st.tabs(["Single match", "League season", "Tournament"])


# ----------------------------------------------------------------------------
# TAB 1: single match
# ----------------------------------------------------------------------------
def render_match():
    scope = st.radio("Scope", available_scopes, format_func=str.title, horizontal=True,
                     key="match_scope")
    bundle = get_bundle(scope)
    teams = sorted(bundle.teams)

    c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
    home = c1.selectbox("Home team", teams,
                        index=teams.index("Arsenal") if "Arsenal" in teams else 0,
                        key="m_home")
    away_options = [t for t in teams if t != home]
    default_away = "Liverpool" if "Liverpool" in away_options else away_options[0]
    away = c2.selectbox("Away team", away_options,
                        index=away_options.index(default_away), key="m_away")
    neutral = c3.checkbox("Neutral venue", value=(scope == "internationals"),
                          key="m_neutral")
    blend = c4.slider("Ensemble weight", 0.0, 1.0, 1.0, 0.05, key="m_blend",
                      help="0 = pure Dixon-Coles, 1 = full stacked ensemble.")

    # Optional: bookmaker odds boost
    use_odds = st.checkbox("Boost with current bookmaker odds (optional)", key="m_use_odds",
                           help="When the model has bookmaker odds, RPS drops from ~0.20 to ~0.198 - "
                                "almost matching the bookmaker baseline.")
    odds_arg = None
    if use_odds:
        oc1, oc2, oc3 = st.columns(3)
        oh = oc1.number_input("Home odds (decimal)", min_value=1.01, max_value=200.0,
                              value=2.10, step=0.01, key="m_odds_h")
        od = oc2.number_input("Draw odds (decimal)", min_value=1.01, max_value=200.0,
                              value=3.40, step=0.01, key="m_odds_d")
        oa = oc3.number_input("Away odds (decimal)", min_value=1.01, max_value=200.0,
                              value=3.50, step=0.01, key="m_odds_a")
        odds_arg = (oh, od, oa)

    pred = bundle.predict(home, away, neutral=neutral, blend=blend, odds=odds_arg)

    st.markdown(f"### {home}  vs  {away}")
    hg, ag = pred["most_likely"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"{home} win", f"{pred['outcome']['H']*100:.1f}%")
    k2.metric("Draw",        f"{pred['outcome']['D']*100:.1f}%")
    k3.metric(f"{away} win", f"{pred['outcome']['A']*100:.1f}%")
    k4.metric("Predicted score", f"{hg} - {ag}")
    k5.metric("Expected goals", f"{pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}")

    st.divider()

    col_bar, col_top = st.columns([1, 1])
    with col_bar:
        st.subheader("Outcome probabilities")
        rows = [{"Result": f"{home} win", "Probability": pred["outcome"]["H"], "Model": "Ensemble"},
                {"Result": "Draw",        "Probability": pred["outcome"]["D"], "Model": "Ensemble"},
                {"Result": f"{away} win", "Probability": pred["outcome"]["A"], "Model": "Ensemble"}]
        if pred["xgb_outcome"] is not None:
            for k, v in [("H", f"{home} win"), ("D", "Draw"), ("A", f"{away} win")]:
                rows.append({"Result": v, "Probability": pred["dc_outcome"][k], "Model": "Dixon-Coles"})
                rows.append({"Result": v, "Probability": pred["xgb_outcome"][k], "Model": "XGBoost"})
        odf = pd.DataFrame(rows)
        fig = px.bar(odf, x="Result", y="Probability", color="Model", barmode="group",
                     text=odf["Probability"].map(lambda v: f"{v*100:.1f}%"))
        fig.update_layout(yaxis_tickformat=".0%", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_top:
        st.subheader("Most likely scorelines")
        top_df = pd.DataFrame([
            {"Score": f"{h}-{a}", "p": p, "Probability": f"{p*100:.1f}%"}
            for (h, a, p) in pred["top_scores"]
        ])
        fig2 = px.bar(top_df, x="Score", y="p", text="Probability", color="p",
                      color_continuous_scale="Blues")
        fig2.update_layout(yaxis_tickformat=".0%", yaxis_title="Probability",
                           height=380, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Full scoreline distribution")
    sm = pred["score_matrix"]
    n = min(7, sm.shape[0])
    heat = sm[:n, :n]
    hm = go.Figure(data=go.Heatmap(
        z=heat * 100,
        x=[str(i) for i in range(n)], y=[str(i) for i in range(n)],
        text=[[f"{heat[i, j]*100:.1f}%" for j in range(n)] for i in range(n)],
        texttemplate="%{text}", colorscale="Blues", colorbar=dict(title="%"),
    ))
    hm.update_layout(xaxis_title=f"{away} goals", yaxis_title=f"{home} goals",
                     yaxis_autorange="reversed", height=480)
    st.plotly_chart(hm, use_container_width=True)


# ----------------------------------------------------------------------------
# TAB 2: League season
# ----------------------------------------------------------------------------
def render_league():
    if "leagues" not in available_scopes:
        st.warning("League model not available. Train it first.")
        return
    bundle = get_bundle("leagues")

    LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
    c1, c2, c3 = st.columns([3, 2, 2])
    league = c1.selectbox("League", LEAGUES, key="l_league")
    continue_season = c2.toggle("Continue from current standings", value=True,
                                key="l_continue",
                                help="If on, only the remaining fixtures are simulated.")
    n_sims = c3.number_input("Simulations", min_value=200, max_value=20000, value=3000,
                             step=500, key="l_nsims")

    state = current_state_for(league)
    if state is None or not state.teams:
        st.error(f"No data found for {league}.")
        return

    if continue_season:
        played = len(state.played_pairs)
        total = len(state.teams) * (len(state.teams) - 1)
        st.caption(f"{len(state.teams)} teams. {played}/{total} fixtures played. "
                   f"Simulating the remaining {total - played}.")
    else:
        st.caption(f"{len(state.teams)} teams. Simulating a full {len(state.teams)*(len(state.teams)-1)} match season from scratch.")

    if continue_season:
        # Show the current table
        with st.expander("Current standings (real)", expanded=False):
            cur = pd.DataFrame([{
                "Team": t, "Pts": state.pts[t], "GF": state.gf[t], "GA": state.ga[t],
                "GD": state.gf[t] - state.ga[t],
            } for t in state.teams]).sort_values(["Pts", "GD", "GF"], ascending=False).reset_index(drop=True)
            cur.index += 1
            st.dataframe(cur, use_container_width=True)

    if st.button("Run season simulation", key="l_run", type="primary"):
        with st.spinner(f"Running {n_sims:,} simulations..."):
            result = simulate_league(bundle, state.teams, n_sims=int(n_sims),
                                     start=state if continue_season else None)
        st.success(f"Done - {n_sims:,} seasons simulated.")
        st.subheader("Predicted final table")
        # Pretty formatting
        styled = result.copy()
        for col in ["Win %", "Top 4 %", "Top 6 %", "Bottom 3 %"]:
            styled[col] = styled[col].map(lambda v: f"{v:.1f}%")
        styled["Expected pts"] = styled["Expected pts"].map(lambda v: f"{v:.1f}")
        styled["Expected pos"] = styled["Expected pos"].map(lambda v: f"{v:.1f}")
        st.dataframe(styled, use_container_width=True, hide_index=True)

        cA, cB = st.columns(2)
        with cA:
            st.subheader("Title race")
            top = result.nlargest(8, "Win %")
            fig = px.bar(top, x="Team", y="Win %", text=top["Win %"].map(lambda v: f"{v:.1f}%"),
                         color="Win %", color_continuous_scale="Greens")
            fig.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with cB:
            st.subheader("Relegation candidates")
            bot = result.nlargest(8, "Bottom 3 %")
            fig = px.bar(bot, x="Team", y="Bottom 3 %",
                         text=bot["Bottom 3 %"].map(lambda v: f"{v:.1f}%"),
                         color="Bottom 3 %", color_continuous_scale="Reds")
            fig.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# TAB 3: Tournament (group + knockout)
# ----------------------------------------------------------------------------
def _snake_seed(teams_ranked: list[str], n_groups: int) -> list[list[str]]:
    """Pot-based snake draft so each group has one strong, one medium, etc."""
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    for i, t in enumerate(teams_ranked):
        pot = i // n_groups
        slot = i % n_groups
        if pot % 2 == 1:
            slot = n_groups - 1 - slot
        groups[slot].append(t)
    return groups


def render_tournament():
    if "internationals" not in available_scopes:
        st.warning("Internationals model not available.")
        return
    bundle = get_bundle("internationals")
    all_teams = sorted(bundle.teams)

    c1, c2, c3 = st.columns([3, 3, 2])
    fmt_name = c1.selectbox("Format", list(FORMATS.keys()), key="t_format")
    fmt = FORMATS[fmt_name]

    # Show "Real" as the default option whenever we have an official draw on file
    has_real = fmt_name in REAL_GROUPS
    draw_options = (["Real (official draw)", "Snake (balanced)", "Random"]
                    if has_real else ["Snake (balanced)", "Random"])
    seed_mode = c2.radio("Group draw", draw_options, horizontal=True, key="t_seed")
    n_sims = c3.number_input("Simulations", min_value=200, max_value=20000, value=3000,
                             step=500, key="t_nsims")

    st.caption(f"{fmt.n_groups} groups of {fmt.per_group} = {fmt.n_teams} teams. "
               f"{fmt.advance_per_group} per group{(' + ' + str(fmt.best_thirds) + ' best 3rds') if fmt.best_thirds else ''} advance to "
               f"{fmt.n_knockout}-team knockout.")

    # WC 2026 squad-strength prior (only relevant for that tournament)
    wc_blend = 0.0
    if fmt_name == "World Cup 2026 (48 teams)":
        wc_blend = st.slider(
            "WC 2026 squad-strength prior",
            0.0, 0.7, 0.30, 0.05, key="t_wc_blend",
            help=("Blend the model's historical-form predictions with current squad "
                  "quality (Transfermarkt market values + bookmaker outright odds). "
                  "0 = pure historical model, 0.3 = balanced (recommended), "
                  "0.7 = mostly market view. Compensates for the model not knowing "
                  "current squad / star-player availability."),
        )

    # Build groups
    if seed_mode.startswith("Real"):
        groups, unknown = resolve_groups(fmt_name, bundle)
        if unknown:
            st.warning(f"Unrecognised teams (treated as default-strength): {unknown}")
        teams = [t for g in groups for t in g]
        st.info(f"Using the actual official draw for **{fmt_name}**.")
    else:
        default_teams = top_n_by_elo(bundle, fmt.n_teams)
        teams = st.multiselect(
            f"Pick {fmt.n_teams} teams (default: top {fmt.n_teams} by Elo)",
            all_teams, default=default_teams, key="t_teams"
        )
        if len(teams) != fmt.n_teams:
            st.warning(f"Need exactly {fmt.n_teams} teams. Currently {len(teams)}.")
            return
        if seed_mode.startswith("Snake"):
            ranked = sorted(teams, key=lambda t: -bundle.elo.rating(t))
            groups = _snake_seed(ranked, fmt.n_groups)
        else:
            rng = np.random.default_rng(42)
            shuffled = list(teams)
            rng.shuffle(shuffled)
            groups = [shuffled[i*fmt.per_group:(i+1)*fmt.per_group] for i in range(fmt.n_groups)]

    with st.expander("Group draw preview", expanded=seed_mode.startswith("Real")):
        cols = st.columns(min(fmt.n_groups, 4))
        for i, g in enumerate(groups):
            col = cols[i % len(cols)]
            with col:
                st.markdown(f"**Group {chr(ord('A') + i)}**")
                rows = [{"Team": t, "Elo": int(bundle.elo.rating(t))} for t in g]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ---- predicted fixtures ----
    with st.expander("Predicted fixtures", expanded=False):
        mode = st.radio(
            "Mode",
            ["Most-likely outcomes (deterministic)", "Sampled tournament (re-roll for upsets)"],
            horizontal=True, key="t_fix_mode",
            help=("Most-likely = each fixture's score is the expected-points-maximising pick "
                  "under the scoring rules below. Sampled = one random tournament drawn from "
                  "the probability distribution; click Re-roll for a different one."),
        )
        # EV-optimisation tuning (collapsed by default; sensible defaults below work for
        # standard 3-for-exact-1-for-result scoring used by most prediction contests)
        exact_pts, result_pts, gd_pts = 3.0, 1.0, 0.0
        draw_bonus = 0.15   # default catches the genuinely tight matches as draws
        if mode.startswith("Most-likely"):
            with st.expander("Tune scoring (advanced)", expanded=False):
                st.caption(
                    "Defaults match the most common prediction contests (3 pts exact, 1 pt result). "
                    "Adjust if your contest uses different rules."
                )
                sc1, sc2, sc3, sc4 = st.columns(4)
                exact_pts = sc1.number_input("Exact score pts", min_value=0.0, max_value=20.0,
                                             value=3.0, step=0.5, key="t_exact_pts")
                result_pts = sc2.number_input("Correct result pts", min_value=0.0, max_value=10.0,
                                              value=1.0, step=0.5, key="t_result_pts")
                gd_pts = sc3.number_input("Correct goal-diff bonus", min_value=0.0,
                                           max_value=10.0, value=0.0, step=0.5, key="t_gd_pts",
                                           help="Extra points if you got the goal-difference right "
                                                "(e.g. predicted 2-1 and actual was 3-2). 0 if "
                                                "your contest doesn't use this.")
                draw_bonus = sc4.number_input("Draw bias", min_value=0.0, max_value=1.0,
                                              value=0.15, step=0.05, key="t_draw_bonus",
                                              help="Pure EV under default scoring picks zero "
                                                   "draws (no WC match has Draw as the most likely "
                                                   "result). 0.15 catches ~6 draws on the tightest "
                                                   "matches; 0.30 matches real WC draw frequency "
                                                   "(~25%) at the cost of some EV.")
                st.caption(
                    "Picks are EV-optimal across **all** scorelines for these rules. "
                    "If a pick lands in a different result region than the headline H/D/A "
                    "favourite, that's correct: the alternative result's modal scoreline is "
                    "concentrated enough to outweigh the lost result point. Each alt score is "
                    "tagged (H/D/A) so you can see which result it implies."
                )

        with st.spinner("Computing fixtures..."):
            # Detect actual played WC 2026 matches from the latest data
            actual_results = {}
            n_actual = 0
            if fmt_name == "World Cup 2026 (48 teams)":
                intl_path = ROOT / "data" / "processed" / "internationals.parquet"
                if intl_path.exists():
                    try:
                        intl_df = pd.read_parquet(intl_path)
                        actual_results = get_actual_results(intl_df)
                        n_actual = len(actual_results)
                    except Exception:
                        actual_results = {}
                if n_actual > 0:
                    st.success(f"✓ {n_actual} WC 2026 match{'es' if n_actual != 1 else ''} "
                               f"detected with real results - using those instead of predictions.")
            if mode.startswith("Most-likely"):
                fixtures = predict_group_fixtures(
                    bundle, groups, fmt_name=fmt_name, wc_blend=wc_blend,
                    exact_pts=exact_pts, result_pts=result_pts, gd_pts=gd_pts,
                    draw_bonus=draw_bonus, actual_results=actual_results,
                )
                rounds, champion, standings = predict_modal_bracket(
                    bundle, fmt, groups, fmt_name=fmt_name, wc_blend=wc_blend,
                    exact_pts=exact_pts, result_pts=result_pts, gd_pts=gd_pts,
                    draw_bonus=draw_bonus, actual_results=actual_results,
                )
                sampled_xg_map = None
            else:
                import time as _time
                if "sample_seed" not in st.session_state:
                    st.session_state["sample_seed"] = int(_time.time())
                cols = st.columns([1, 4])
                if cols[0].button("Re-roll", key="t_reroll", type="primary"):
                    st.session_state["sample_seed"] = int(_time.time() * 1000) % (2**32)
                cols[1].caption(f"Sample #{st.session_state['sample_seed']} - click Re-roll for a different tournament")
                fixtures, standings, rounds, champion = sample_one_tournament(
                    bundle, fmt, groups, seed=st.session_state["sample_seed"],
                    fmt_name=fmt_name, wc_blend=wc_blend,
                )

        view_by_stage, view_chrono, view_by_group = st.tabs(
            ["By stage", "Chronological", "By group"]
        )

        # ---- By stage view: one section per tournament stage ----
        with view_by_stage:
            st.caption(
                "Predictions organised by tournament stage "
                "(Group Stage → R32 → R16 → QF → SF → Final). Each section is "
                "self-contained. Played matches show ✓ actual results; everything else "
                "is the model's predicted score."
            )

            # ---- Group stage section ----
            st.markdown("### Group Stage")
            n_played = sum(1 for f in fixtures if f.get("is_actual"))
            n_total = len(fixtures)
            if n_played > 0:
                st.caption(f"{n_played} of {n_total} matches played (real results); "
                           f"{n_total - n_played} predicted.")
            else:
                st.caption(f"All {n_total} matches predicted (tournament hasn't started yet).")
            gs_rows = []
            for f in fixtures:
                marker = "✓ " if f.get("is_actual") else ""
                score_text = f"{marker}{f['score'][0]}–{f['score'][1]}"
                if not f.get("is_actual") and f.get("score_prob"):
                    score_text += f" ({f['score_prob']*100:.0f}%)"
                row = {}
                if "date" in f:
                    row["Date"] = pd.Timestamp(f["date"]).strftime("%a %d %b")
                row.update({
                    "Group": chr(ord('A') + f["group_idx"]),
                    "Home": f["home"],
                    "Score": score_text,
                    "Away": f["away"],
                })
                gs_rows.append(row)
            st.dataframe(pd.DataFrame(gs_rows), hide_index=True, use_container_width=True,
                         height=min(450, 38 + 35 * len(gs_rows)))

            # ---- One section per knockout round ----
            for round_matches in rounds:
                round_name = round_matches[0]["round"]
                round_date = round_matches[0].get("date")
                n_actual_round = sum(1 for m in round_matches if m.get("is_actual"))
                header = f"### {round_name}"
                if round_date:
                    header += f"  _(from {pd.Timestamp(round_date).strftime('%d %b')})_"
                st.markdown(header)
                if n_actual_round > 0:
                    st.caption(f"{n_actual_round} of {len(round_matches)} matches played.")
                ko_rows = []
                for m in round_matches:
                    marker = "✓ " if m.get("is_actual") else ""
                    score_text = f"{marker}{m['score'][0]}–{m['score'][1]}"
                    if m.get("pens"):
                        score_text += f" → {m['winner']} (ET/pens)"
                    elif not m.get("is_actual") and m.get("p_home") is not None:
                        # Show winning side probability for context
                        pass
                    ko_rows.append({
                        "Home": m["home"],
                        "Score": score_text,
                        "Away": m["away"],
                        "Winner": m["winner"],
                    })
                st.dataframe(pd.DataFrame(ko_rows), hide_index=True, use_container_width=True)

            if champion:
                st.success(f"🏆 **Predicted Champion: {champion}**")

        # ---- Chronological view (by real date when available, else by matchday) ----
        with view_chrono:
            has_dates = any("date" in f for f in fixtures)
            if has_dates:
                # Group fixtures by date, render each day as its own block
                by_date: dict[str, list[dict]] = {}
                for f in fixtures:
                    by_date.setdefault(f.get("date", "?"), []).append(f)
                for date in sorted(by_date):
                    label = pd.Timestamp(date).strftime("%a %d %b %Y")
                    st.markdown(f"**{label}**")
                    rows = []
                    for f in by_date[date]:
                        marker = "✓ " if f.get("is_actual") else ""
                        score_text = f"{marker}{f['score'][0]}–{f['score'][1]}"
                        if not f.get("is_actual") and "score_prob" in f and f["score_prob"]:
                            score_text += f" ({f['score_prob']*100:.0f}%)"
                        row = {
                            "Group":  chr(ord('A') + f["group_idx"]),
                            "Home":   f["home"],
                            "Score":  score_text,
                            "Away":   f["away"],
                        }
                        if "p_home" in f:
                            row["H %"] = f"{f['p_home']*100:.0f}%"
                            row["D %"] = f"{f['p_draw']*100:.0f}%"
                            row["A %"] = f"{f['p_away']*100:.0f}%"
                        if f.get("alt_scores"):
                            row["Alt scores"] = f["alt_scores"]
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                for md in [1, 2, 3]:
                    md_fixtures = [f for f in fixtures if f["matchday"] == md]
                    if not md_fixtures:
                        continue
                    st.markdown(f"**Matchday {md}**")
                    rows = []
                    for f in md_fixtures:
                        row = {
                            "Group":  chr(ord('A') + f["group_idx"]),
                            "Home":   f["home"],
                            "Score":  f"{f['score'][0]}–{f['score'][1]}",
                            "Away":   f["away"],
                        }
                        if "p_home" in f:
                            row["H %"] = f"{f['p_home']*100:.0f}%"
                            row["D %"] = f"{f['p_draw']*100:.0f}%"
                            row["A %"] = f"{f['p_away']*100:.0f}%"
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            for round_matches in rounds:
                round_name = round_matches[0]["round"]
                date_suffix = ""
                if "date" in round_matches[0]:
                    date_suffix = f"  ({pd.Timestamp(round_matches[0]['date']).strftime('%d %b')} onwards)"
                st.markdown(f"**{round_name}**{date_suffix}")
                rows = []
                for m in round_matches:
                    score_text = f"{m['score'][0]}–{m['score'][1]}"
                    if m.get("pens"):
                        score_text += f" → {m['winner']} (ET/pens)"
                    row = {
                        "Home":   m["home"],
                        "Score":  score_text,
                        "Away":   m["away"],
                        "Win":    m["winner"],
                    }
                    if "p_home" in m:
                        row["H %"] = f"{m['p_home']*100:.0f}%"
                        row["D %"] = f"{m['p_draw']*100:.0f}%"
                        row["A %"] = f"{m['p_away']*100:.0f}%"
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            if champion:
                st.success(f"Champion: **{champion}**")

        # ---- By group view (each group + its 6 fixtures + standings) ----
        with view_by_group:
            group_tabs = st.tabs([f"Group {chr(ord('A') + i)}" for i in range(fmt.n_groups)])
            for gi, tab in enumerate(group_tabs):
                with tab:
                    rows = []
                    for f in [x for x in fixtures if x["group_idx"] == gi]:
                        row = {}
                        if "date" in f:
                            row["Date"] = pd.Timestamp(f["date"]).strftime("%d %b")
                        else:
                            row["MD"] = f["matchday"]
                        score_text = f"{f['score'][0]}–{f['score'][1]}"
                        if "score_prob" in f and f["score_prob"]:
                            score_text += f" ({f['score_prob']*100:.0f}%)"
                        row.update({
                            "Home":   f["home"],
                            "Score":  score_text,
                            "Away":   f["away"],
                        })
                        if "p_home" in f:
                            row["H %"] = f"{f['p_home']*100:.0f}%"
                            row["D %"] = f"{f['p_draw']*100:.0f}%"
                            row["A %"] = f"{f['p_away']*100:.0f}%"
                            row["xG"]  = f"{f['xg_home']:.2f}–{f['xg_away']:.2f}"
                        if f.get("alt_scores"):
                            row["Alt scores"] = f["alt_scores"]
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                    st.caption("Final standings:")
                    srows = [{"Pos": i + 1, "Team": t, "Pts": s["pts"],
                              "GD": s["gd"], "GF": s["gf"]}
                             for i, (t, s) in enumerate(standings[gi])]
                    st.dataframe(pd.DataFrame(srows), hide_index=True, use_container_width=True)

    if st.button("Run tournament simulation", key="t_run", type="primary"):
        with st.spinner(f"Running {n_sims:,} tournament simulations..."):
            result = simulate_knockout(bundle, fmt, groups, n_sims=int(n_sims),
                                       fmt_name=fmt_name, wc_blend=wc_blend)
        st.success(f"Done - {n_sims:,} tournaments simulated.")
        st.subheader("Stage advancement probabilities")
        # Pretty-format every '... %' column
        styled = result.copy()
        for col in [c for c in styled.columns if c.endswith(" %")]:
            styled[col] = styled[col].map(lambda v: f"{v:.1f}%")
        st.dataframe(styled, use_container_width=True, hide_index=True)

        cA, cB = st.columns(2)
        with cA:
            st.subheader("Title favourites")
            top = result.nlargest(10, "Champion %")
            fig = px.bar(top, x="Team", y="Champion %",
                         text=top["Champion %"].map(lambda v: f"{v:.1f}%"),
                         color="Champion %", color_continuous_scale="Viridis")
            fig.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with cB:
            st.subheader("Reach the final")
            if "Final %" in result.columns:
                top = result.nlargest(10, "Final %")
                fig = px.bar(top, x="Team", y="Final %",
                             text=top["Final %"].map(lambda v: f"{v:.1f}%"),
                             color="Final %", color_continuous_scale="Cividis")
                fig.update_layout(height=400, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Dispatch
# ============================================================================
with tab_match:
    render_match()

with tab_league:
    render_league()

with tab_cup:
    render_tournament()
