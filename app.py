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

from src.club_logos import logo as club_logo, has_logos
from src.flags import flagged
from src.predictor import PredictorBundle
from src.real_groups import REAL_GROUPS, resolve_groups
from src.tournament import (
    FORMATS, current_state_for, get_actual_results, predict_group_fixtures,
    predict_modal_bracket, sample_one_tournament, simulate_knockout, simulate_league,
    top_n_by_elo,
)

st.set_page_config(page_title="Football Predictor", layout="wide",
                   page_icon=":soccer:")

# ============================================================================
# Custom styling - dark theme, custom fonts, hero section, footer
# ============================================================================
_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, p, label, div {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background: radial-gradient(ellipse at top, #131c2e 0%, #0a0e1a 50%) !important;
}

h1, h2, h3, h4, h5 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: -0.02em !important;
    font-weight: 700 !important;
}

/* ----- Hero ----- */
.hero {
    background: linear-gradient(135deg, #0e2a3f 0%, #0d4d3a 100%);
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    margin: 0 0 2rem 0;
    border: 1px solid rgba(16, 185, 129, 0.18);
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 32px rgba(0, 0, 0, 0.3);
}
.hero::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 380px;
    height: 380px;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.18) 0%, transparent 65%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -50%;
    left: -10%;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(251, 191, 36, 0.10) 0%, transparent 65%);
    pointer-events: none;
}
.hero-content { position: relative; z-index: 2; }
.hero h1 {
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    margin: 0 0 0.4rem 0 !important;
    background: linear-gradient(95deg, #34d399 0%, #fbbf24 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.hero-subtitle {
    color: #cbd5e1;
    font-size: 1.05rem;
    margin: 0 0 1.1rem 0;
    max-width: 720px;
    line-height: 1.5;
}
.hero-stats { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.stat-pill {
    background: rgba(16, 185, 129, 0.10);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
    padding: 0.38rem 0.95rem;
    border-radius: 100px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}
.stat-pill.gold {
    background: rgba(251, 191, 36, 0.08);
    border-color: rgba(251, 191, 36, 0.3);
    color: #fbbf24;
}

/* ----- Metric cards ----- */
[data-testid="stMetric"], [data-testid="metric-container"] {
    background: rgba(19, 24, 37, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    transition: border-color 0.2s ease;
}
[data-testid="stMetric"]:hover, [data-testid="metric-container"]:hover {
    border-color: rgba(16, 185, 129, 0.3);
}
[data-testid="stMetricValue"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
    color: #f8fafc !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ----- Buttons ----- */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.18s ease !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.22);
    border-color: rgba(16, 185, 129, 0.4) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    border-color: rgba(16, 185, 129, 0.55) !important;
}

/* ----- Tabs ----- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0 !important;
    padding: 0.55rem 1.1rem !important;
    font-weight: 600 !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(16, 185, 129, 0.1) !important;
    color: #34d399 !important;
}

/* ----- Sidebar ----- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d121e 0%, #0a0e1a 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.5rem !important;
    background: linear-gradient(95deg, #34d399 0%, #fbbf24 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ----- DataFrames ----- */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

/* ----- Success/warning/info banners ----- */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* ----- Expanders ----- */
[data-testid="stExpander"] details {
    background: rgba(19, 24, 37, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 10px !important;
}

/* ----- Numbers in mono ----- */
.score-mono {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700;
    letter-spacing: 0.02em;
}

/* ----- Footer ----- */
.custom-footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    margin-top: 4rem;
    color: #64748b;
    font-size: 0.82rem;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    line-height: 1.7;
}
.custom-footer a { color: #34d399; text-decoration: none; font-weight: 500; }
.custom-footer a:hover { text-decoration: underline; }

/* ----- Hide default chrome ----- */
#MainMenu, footer:not(.custom-footer), header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}
.block-container { padding-top: 1.5rem !important; }

/* ----- Group stage cards ----- */
.group-card {
    background: linear-gradient(180deg, rgba(19,24,37,0.85) 0%, rgba(13,18,30,0.95) 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 0.9rem 1rem 1rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 14px rgba(0,0,0,0.18);
}
.group-card .gc-header {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 0.6rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.group-card .gc-letter {
    font-weight: 800;
    font-size: 1.05rem;
    background: linear-gradient(95deg, #34d399 0%, #fbbf24 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.03em;
}
.group-card table.gc-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 0.75rem;
    font-size: 0.82rem;
}
.group-card table.gc-table {
    table-layout: fixed;
}
.group-card table.gc-table th, .group-card table.gc-table td {
    padding: 5px 4px;
    text-align: right;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.group-card table.gc-table th { color: #94a3b8; font-weight: 600; font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.04em; }
.group-card table.gc-table th.team-th, .group-card table.gc-table td.team-cell {
    text-align: left; color: #f1f5f9; font-weight: 500; width: auto;
}
.group-card table.gc-table td.pos-cell { text-align: center; color: #94a3b8; width: 22px;
    font-family: 'JetBrains Mono', monospace; padding-left: 0; padding-right: 2px; }
.group-card table.gc-table th.num, .group-card table.gc-table td.num { width: 24px; }
.group-card table.gc-table th.gfga, .group-card table.gc-table td.gfga { width: 48px; }
.group-card table.gc-table th.gd-col, .group-card table.gc-table td.gd-col { width: 30px; }
.group-card table.gc-table th.pts-col, .group-card table.gc-table td.pts-col { width: 30px; }
.group-card table.gc-table td.pts-cell { font-weight: 700; color: #f8fafc;
    font-family: 'JetBrains Mono', monospace; }
.group-card tr.adv-q td.pos-cell { color: #34d399; }
.group-card tr.adv-q td.pos-cell::before { content: ""; display: inline-block; width: 3px;
    height: 14px; background: #10b981; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.group-card tr.adv-t td.pos-cell { color: #fbbf24; }
.group-card tr.adv-t td.pos-cell::before { content: ""; display: inline-block; width: 3px;
    height: 14px; background: #f59e0b; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.group-card .gc-fixtures { font-size: 0.78rem; color: #cbd5e1; }
.group-card .gc-fixtures .fx {
    display: flex; align-items: center; gap: 0.35rem;
    padding: 4px 0; border-bottom: 1px dashed rgba(255,255,255,0.05);
}
.group-card .gc-fixtures .fx:last-child { border-bottom: none; }
.group-card .gc-fixtures .fx .fx-date { flex: 0 0 50px; color: #64748b; font-size: 0.7rem; }
.group-card .gc-fixtures .fx .fx-home,
.group-card .gc-fixtures .fx .fx-away  {
    flex: 1 1 0; min-width: 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.group-card .gc-fixtures .fx .fx-home  { text-align: right; }
.group-card .gc-fixtures .fx .fx-away  { text-align: left; }
.group-card .gc-fixtures .fx .fx-score {
    font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.85rem;
    color: #f8fafc; padding: 1px 8px; min-width: 44px; text-align: center;
    background: rgba(255,255,255,0.04); border-radius: 5px;
}
.group-card .gc-fixtures .fx.played .fx-score { background: rgba(16,185,129,0.18);
    color: #34d399; }

/* ----- Knockout bracket ----- */
.bracket-wrap {
    display: flex; gap: 0.6rem; overflow-x: auto; padding: 1rem 0.4rem 0.5rem;
    align-items: stretch;
}
.bracket-round {
    display: flex; flex-direction: column; justify-content: space-around;
    min-width: 195px; flex: 1; gap: 0.55rem;
}
.bracket-round-title {
    text-align: center; color: #94a3b8; font-size: 0.74rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem;
}
.bracket-match {
    background: linear-gradient(180deg, rgba(19,24,37,0.85) 0%, rgba(13,18,30,0.95) 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 3px solid rgba(16,185,129,0.5);
    border-radius: 8px;
    padding: 0.45rem 0.6rem;
    font-size: 0.8rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.18);
}
.bracket-match.pens { border-left-color: rgba(251,191,36,0.7); }
.bracket-match.played { border-left-color: rgba(34,211,238,0.7); }
.bracket-match .bm-side {
    display: flex; justify-content: space-between; align-items: center;
    padding: 2px 0; gap: 0.4rem;
}
.bracket-match .bm-side .team {
    flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    color: #cbd5e1; font-weight: 500;
}
.bracket-match .bm-side.winner .team { color: #f8fafc; font-weight: 700; }
.bracket-match .bm-side.loser  .team { color: #64748b; }
.bracket-match .bm-side .gs {
    font-family: 'JetBrains Mono', monospace; font-weight: 700;
    min-width: 20px; text-align: right;
}
.bracket-match .bm-side.winner .gs { color: #34d399; }
.bracket-match .bm-side.loser  .gs { color: #64748b; }
.bracket-match .bm-meta {
    margin-top: 4px; padding-top: 4px; border-top: 1px dashed rgba(255,255,255,0.05);
    font-size: 0.66rem; color: #94a3b8; text-align: center; letter-spacing: 0.04em;
}
.bracket-champion {
    background: linear-gradient(135deg, #14532d 0%, #422006 100%);
    border: 1px solid rgba(251,191,36,0.5);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    font-weight: 700;
    margin-top: 1rem;
}
.bracket-champion .ch-label { color: #fbbf24; text-transform: uppercase;
    letter-spacing: 0.1em; font-size: 0.7rem; }
.bracket-champion .ch-team { color: #fef3c7; font-size: 1.3rem; margin-top: 0.3rem; }

/* ----- Mobile tweaks ----- */
@media (max-width: 768px) {
    .hero { padding: 1.5rem 1.3rem; }
    .hero h1 { font-size: 1.8rem !important; }
    .hero-subtitle { font-size: 0.95rem; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    .bracket-round { min-width: 165px; }
}
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

_HERO_HTML = """
<div class="hero">
  <div class="hero-content">
    <h1>Football Predictor</h1>
    <p class="hero-subtitle">
      Statistical match and tournament predictions across the top-5 European
      leagues and every international team — built on a stacked Elo +
      Dixon-Coles + gradient-boosted ensemble. Tournament mode covers the
      <strong style="color:#fbbf24">2026 FIFA World Cup</strong> with the
      official draw, real fixture dates, and squad-strength priors.
    </p>
    <div class="hero-stats">
      <span class="stat-pill">Stacked ML ensemble</span>
      <span class="stat-pill">RPS 0.173 internationals</span>
      <span class="stat-pill">RPS 0.198 leagues</span>
      <span class="stat-pill gold">WC 2026 ready</span>
    </div>
  </div>
</div>
"""
st.markdown(_HERO_HTML, unsafe_allow_html=True)

# ============================================================================
# Sidebar (shared across tabs)
# ============================================================================
st.sidebar.markdown("# ⚽ Football Predictor")
st.sidebar.caption("Stacked ML ensemble · daily auto-updated")

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

# Feedback / feature request section
with st.sidebar.expander("Feedback / feature request"):
    st.caption("Spot a bug or have an idea? Drop it here.")
    fb_type = st.selectbox("Type", ["Feature request", "Bug report", "Other"],
                            key="fb_type")
    fb_title = st.text_input("Short title", key="fb_title",
                              placeholder="e.g. Add Champions League")
    fb_desc = st.text_area("Details", key="fb_desc",
                            placeholder="Describe the issue or idea...")
    if st.button("Create on GitHub", key="fb_submit", type="secondary"):
        if fb_title and fb_desc:
            from urllib.parse import quote
            label_map = {
                "Feature request": "enhancement",
                "Bug report": "bug",
                "Other": "feedback",
            }
            prefix = {"Feature request": "[Feature] ", "Bug report": "[Bug] ", "Other": ""}[fb_type]
            url = (
                "https://github.com/jdgoated1/football-predictor/issues/new"
                f"?title={quote(prefix + fb_title)}"
                f"&body={quote(fb_desc)}"
                f"&labels={label_map[fb_type]}"
            )
            st.markdown(f"[**Click here to post on GitHub →**]({url})")
            st.caption("Pre-fills your text. You'll need a free GitHub account "
                        "to publish. Or just share the link directly.")
        else:
            st.warning("Please fill in both title and details first.")
    st.caption("Or browse [existing issues](https://github.com/jdgoated1/football-predictor/issues).")

st.sidebar.divider()

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

    # Format team names with flag emojis (for internationals; clubs unchanged)
    fmt_team = (lambda t: flagged(t)) if scope == "internationals" else (lambda t: t)
    c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
    home = c1.selectbox("Home team", teams,
                        index=teams.index("Arsenal") if "Arsenal" in teams else 0,
                        key="m_home", format_func=fmt_team)
    away_options = [t for t in teams if t != home]
    default_away = "Liverpool" if "Liverpool" in away_options else away_options[0]
    away = c2.selectbox("Away team", away_options,
                        index=away_options.index(default_away), key="m_away",
                        format_func=fmt_team)
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
    home_disp = fmt_team(home)
    away_disp = fmt_team(away)

    # Show club crests for leagues scope when logos are available
    if scope == "leagues" and has_logos():
        crest_cols = st.columns([1, 4, 1])
        h_logo = club_logo(home)
        a_logo = club_logo(away)
        with crest_cols[0]:
            if h_logo:
                st.image(h_logo, width=80)
        with crest_cols[1]:
            st.markdown(f"<h3 style='text-align:center;margin-top:1.5rem'>"
                        f"{home_disp}  vs  {away_disp}</h3>",
                        unsafe_allow_html=True)
        with crest_cols[2]:
            if a_logo:
                st.image(a_logo, width=80)
    else:
        st.markdown(f"### {home_disp}  vs  {away_disp}")
    hg, ag = pred["most_likely"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"{home_disp} win", f"{pred['outcome']['H']*100:.1f}%")
    k2.metric("Draw",        f"{pred['outcome']['D']*100:.1f}%")
    k3.metric(f"{away_disp} win", f"{pred['outcome']['A']*100:.1f}%")
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
LEAGUE_BADGES = {
    "Premier League": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "La Liga":        "🇪🇸",
    "Bundesliga":     "🇩🇪",
    "Serie A":        "🇮🇹",
    "Ligue 1":        "🇫🇷",
}


def _render_one_league(bundle, league: str, key_prefix: str) -> None:
    """Render the full league simulator for one league. Called per tab."""
    state = current_state_for(league)
    if state is None or not state.teams:
        st.error(f"No data found for {league}.")
        return

    c1, c2 = st.columns([2, 2])
    continue_season = c1.toggle("Continue from current standings", value=True,
                                key=f"{key_prefix}_continue",
                                help="If on, only the remaining fixtures are simulated.")
    n_sims = c2.number_input("Simulations", min_value=200, max_value=20000, value=3000,
                             step=500, key=f"{key_prefix}_nsims")

    if continue_season:
        played = len(state.played_pairs)
        total = len(state.teams) * (len(state.teams) - 1)
        st.caption(f"{len(state.teams)} teams. {played}/{total} fixtures played. "
                   f"Simulating the remaining {total - played}.")
    else:
        st.caption(f"{len(state.teams)} teams. Simulating a full "
                   f"{len(state.teams)*(len(state.teams)-1)} match season from scratch.")

    use_logos = has_logos()

    if continue_season:
        with st.expander("Current standings (real)", expanded=False):
            cur_rows = []
            for t in sorted(state.teams,
                             key=lambda x: (-state.pts[x],
                                            -(state.gf[x] - state.ga[x]),
                                            -state.gf[x])):
                row = {}
                if use_logos:
                    row[" "] = club_logo(t) or ""
                row.update({"Team": t, "Pts": state.pts[t], "GF": state.gf[t],
                            "GA": state.ga[t], "GD": state.gf[t] - state.ga[t]})
                cur_rows.append(row)
            cur = pd.DataFrame(cur_rows)
            cur.index = range(1, len(cur) + 1)
            cfg = {" ": st.column_config.ImageColumn("", width="small")} if use_logos else None
            st.dataframe(cur, use_container_width=True, column_config=cfg)

    if st.button("Run season simulation", key=f"{key_prefix}_run", type="primary"):
        with st.spinner(f"Running {n_sims:,} simulations..."):
            result = simulate_league(bundle, state.teams, n_sims=int(n_sims),
                                     start=state if continue_season else None)
        st.success(f"Done - {n_sims:,} seasons simulated.")
        st.subheader("Predicted final table")
        styled = result.copy()
        for col in ["Win %", "Top 4 %", "Top 6 %", "Bottom 3 %"]:
            styled[col] = styled[col].map(lambda v: f"{v:.1f}%")
        styled["Expected pts"] = styled["Expected pts"].map(lambda v: f"{v:.1f}")
        styled["Expected pos"] = styled["Expected pos"].map(lambda v: f"{v:.1f}")
        if use_logos:
            styled.insert(0, " ", styled["Team"].map(lambda t: club_logo(t) or ""))
        cfg = {" ": st.column_config.ImageColumn("", width="small")} if use_logos else None
        st.dataframe(styled, use_container_width=True, hide_index=True,
                     column_config=cfg)

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


def render_league():
    if "leagues" not in available_scopes:
        st.warning("League model not available. Train it first.")
        return
    bundle = get_bundle("leagues")

    leagues = list(LEAGUE_BADGES.keys())
    tab_labels = [f"{LEAGUE_BADGES[lg]}  {lg}" for lg in leagues]
    league_tabs = st.tabs(tab_labels)
    for tab, lg in zip(league_tabs, leagues):
        with tab:
            _render_one_league(bundle, lg, key_prefix=f"l_{lg.replace(' ', '_').lower()}")


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


# ----------------------------------------------------------------------------
# Tournament visualisation helpers
# ----------------------------------------------------------------------------
import html as _html


def _team_label(team: str, height: int = 12) -> str:
    """Short team label with a leading flag IMG (works on Linux servers).

    Falls back to plain name when no flag mapping exists. Uses flagcdn PNGs
    rather than Unicode emojis because Streamlit Cloud's Linux fonts don't
    render regional-indicator pairs as flag glyphs."""
    from src.flags import flag_img_html
    img = flag_img_html(team, height=height)
    return f"{img}{_html.escape(team)}"


def _stats_from_fixtures(group_teams: list[str],
                          group_fixtures: list[dict]) -> dict[str, dict]:
    """Recover P/W/D/L/GF/GA from the predicted fixture list for one group."""
    stats = {t: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0} for t in group_teams}
    for f in group_fixtures:
        h, a = f["home"], f["away"]
        hg, ag = f["score"]
        if h not in stats or a not in stats:
            continue
        stats[h]["P"] += 1; stats[a]["P"] += 1
        stats[h]["GF"] += hg; stats[h]["GA"] += ag
        stats[a]["GF"] += ag; stats[a]["GA"] += hg
        if hg > ag:
            stats[h]["W"] += 1; stats[a]["L"] += 1
        elif hg < ag:
            stats[a]["W"] += 1; stats[h]["L"] += 1
        else:
            stats[h]["D"] += 1; stats[a]["D"] += 1
    return stats


def _render_group_cards(fixtures: list[dict], standings: list[list[tuple]],
                         groups: list[list[str]], fmt) -> None:
    """Render each group as a styled card with standings + fixtures.

    Grid: 3 columns when there are 6+ groups, 4 when 4, else 2.
    """
    n_groups = len(groups)
    if n_groups >= 6:
        cols_per_row = 3
    elif n_groups == 4:
        cols_per_row = 4
    elif n_groups in (2, 3):
        cols_per_row = n_groups
    else:
        cols_per_row = 2

    # Best-3rds qualifiers (highlighted yellow) if format uses them
    best_thirds_set: set[str] = set()
    if fmt.best_thirds > 0:
        thirds = []
        for ranked in standings:
            if len(ranked) > fmt.advance_per_group:
                team, st_dict = ranked[fmt.advance_per_group]
                thirds.append((team, st_dict))
        thirds.sort(key=lambda kv: (-kv[1]["pts"], -kv[1]["gd"], -kv[1]["gf"]))
        best_thirds_set = {t for t, _ in thirds[:fmt.best_thirds]}

    def card_html(gi: int) -> str:
        letter = chr(ord("A") + gi)
        stats = _stats_from_fixtures(groups[gi], [f for f in fixtures if f["group_idx"] == gi])
        # Standings table rows
        rows_html = []
        for pos, (team, sd) in enumerate(standings[gi], start=1):
            s = stats[team]
            row_class = ""
            if pos <= fmt.advance_per_group:
                row_class = "adv-q"
            elif team in best_thirds_set:
                row_class = "adv-t"
            rows_html.append(
                f'<tr class="{row_class}">'
                f'<td class="pos-cell">{pos}</td>'
                f'<td class="team-cell">{_team_label(team)}</td>'
                f'<td class="num">{s["P"]}</td>'
                f'<td class="num">{s["W"]}</td>'
                f'<td class="num">{s["D"]}</td>'
                f'<td class="num">{s["L"]}</td>'
                f'<td class="gfga">{s["GF"]}–{s["GA"]}</td>'
                f'<td class="gd-col">{(sd["gd"]):+d}</td>'
                f'<td class="pts-cell pts-col">{sd["pts"]}</td>'
                f'</tr>'
            )
        # Fixtures list (sorted by date if available, else by matchday)
        gfx = sorted(
            [f for f in fixtures if f["group_idx"] == gi],
            key=lambda f: (f.get("date") or "", f.get("matchday", 0)),
        )
        fx_html = []
        for f in gfx:
            played = f.get("is_actual", False)
            date_label = ""
            if f.get("date"):
                date_label = pd.Timestamp(f["date"]).strftime("%d %b")
            elif "matchday" in f:
                date_label = f"MD{f['matchday']}"
            sc = f"{f['score'][0]}–{f['score'][1]}"
            fx_html.append(
                f'<div class="fx{" played" if played else ""}">'
                f'<span class="fx-date">{date_label}</span>'
                f'<span class="fx-home">{_team_label(f["home"])}</span>'
                f'<span class="fx-score">{sc}</span>'
                f'<span class="fx-away">{_team_label(f["away"])}</span>'
                f'</div>'
            )
        return (
            f'<div class="group-card">'
            f'<div class="gc-header"><span class="gc-letter">GROUP {letter}</span></div>'
            f'<table class="gc-table">'
            f'<thead><tr><th class="pos-cell"></th><th class="team-th">Team</th>'
            f'<th class="num">P</th><th class="num">W</th><th class="num">D</th>'
            f'<th class="num">L</th><th class="gfga">GF–GA</th>'
            f'<th class="gd-col">GD</th><th class="pts-col">Pts</th>'
            f'</tr></thead><tbody>{"".join(rows_html)}</tbody></table>'
            f'<div class="gc-fixtures">{"".join(fx_html)}</div>'
            f'</div>'
        )

    # Render in rows of `cols_per_row`
    for row_start in range(0, n_groups, cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            gi = row_start + col_idx
            if gi >= n_groups:
                break
            with cols[col_idx]:
                st.markdown(card_html(gi), unsafe_allow_html=True)

    # Legend
    legend_parts = []
    legend_parts.append('<span style="color:#34d399">▌</span> Advances')
    if fmt.best_thirds > 0:
        legend_parts.append('<span style="color:#fbbf24">▌</span> Best 3rd-place')
    st.markdown(
        f'<div style="color:#94a3b8;font-size:0.78rem;margin-top:0.2rem">'
        f'{" &nbsp; ".join(legend_parts)}</div>',
        unsafe_allow_html=True,
    )


def _render_bracket(rounds: list[list[dict]], champion: str | None) -> None:
    """Render the knockout rounds as a horizontal bracket of match cards."""
    if not rounds:
        return
    round_cols_html = []
    for round_matches in rounds:
        round_name = round_matches[0]["round"]
        date_label = ""
        if round_matches[0].get("date"):
            date_label = f' · {pd.Timestamp(round_matches[0]["date"]).strftime("%d %b")}'
        match_html = []
        for m in round_matches:
            hg, ag = m["score"]
            winner = m["winner"]
            played = m.get("is_actual", False)
            pens = m.get("pens", False)
            home_class = "winner" if winner == m["home"] else "loser"
            away_class = "winner" if winner == m["away"] else "loser"
            classes = ["bracket-match"]
            if played: classes.append("played")
            if pens:   classes.append("pens")
            meta = ""
            if pens:
                meta = f'<div class="bm-meta">ET / pens → {_html.escape(winner)}</div>'
            elif played:
                meta = '<div class="bm-meta">✓ played</div>'
            match_html.append(
                f'<div class="{" ".join(classes)}">'
                f'<div class="bm-side {home_class}">'
                f'<span class="team">{_team_label(m["home"])}</span>'
                f'<span class="gs">{hg}</span></div>'
                f'<div class="bm-side {away_class}">'
                f'<span class="team">{_team_label(m["away"])}</span>'
                f'<span class="gs">{ag}</span></div>'
                f'{meta}'
                f'</div>'
            )
        round_cols_html.append(
            f'<div class="bracket-round">'
            f'<div class="bracket-round-title">{_html.escape(round_name)}{date_label}</div>'
            f'{"".join(match_html)}'
            f'</div>'
        )
    st.markdown(
        f'<div class="bracket-wrap">{"".join(round_cols_html)}</div>',
        unsafe_allow_html=True,
    )
    if champion:
        st.markdown(
            f'<div class="bracket-champion">'
            f'<div class="ch-label">🏆 Predicted Champion</div>'
            f'<div class="ch-team">{_team_label(champion)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


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
                rows = [{"Team": flagged(t), "Elo": int(bundle.elo.rating(t))} for t in g]
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
                "Predictions organised by stage. Group cards show standings + every "
                "fixture; knockout rounds render as a bracket. Played matches are "
                "highlighted; everything else is the model's predicted score."
            )

            # ---- Group stage (visual cards) ----
            st.markdown("### Group Stage")
            n_played = sum(1 for f in fixtures if f.get("is_actual"))
            n_total = len(fixtures)
            if n_played > 0:
                st.caption(f"{n_played} of {n_total} matches played (real results); "
                           f"{n_total - n_played} predicted.")
            else:
                st.caption(f"All {n_total} matches predicted (tournament hasn't started yet).")
            _render_group_cards(fixtures, standings, groups, fmt)

            # ---- Knockout bracket (visual) ----
            if rounds:
                st.markdown("### Knockout Bracket")
                ko_total = sum(len(r) for r in rounds)
                ko_played = sum(1 for r in rounds for m in r if m.get("is_actual"))
                if ko_played > 0:
                    st.caption(f"{ko_played} of {ko_total} knockout matches played.")
                _render_bracket(rounds, champion)

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
                            "Home":   flagged(f["home"]),
                            "Score":  score_text,
                            "Away":   flagged(f["away"]),
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
                            "Home":   flagged(f["home"]),
                            "Score":  f"{f['score'][0]}–{f['score'][1]}",
                            "Away":   flagged(f["away"]),
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
                        "Home":   flagged(m["home"]),
                        "Score":  score_text,
                        "Away":   flagged(m["away"]),
                        "Win":    flagged(m["winner"]),
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
                            "Home":   flagged(f["home"]),
                            "Score":  score_text,
                            "Away":   flagged(f["away"]),
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
                    srows = [{"Pos": i + 1, "Team": flagged(t), "Pts": s["pts"],
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


# ============================================================================
# Footer
# ============================================================================
_FOOTER_HTML = """
<div class="custom-footer">
  <div>
    Built with <a href="https://streamlit.io" target="_blank">Streamlit</a> ·
    Models: Elo + Pi-rating + Dixon-Coles + (XGBoost + LightGBM + CatBoost)
    ensemble with isotonic-calibrated meta-learner
  </div>
  <div>
    Data: <a href="https://www.football-data.co.uk" target="_blank">football-data.co.uk</a> ·
    <a href="https://github.com/martj42/international_results" target="_blank">martj42/international_results</a> ·
    <a href="https://understat.com" target="_blank">Understat</a> ·
    public bookmakers
  </div>
  <div style="margin-top: 0.8rem;">
    <a href="https://github.com/jdgoated1/football-predictor" target="_blank">
      ★ View source on GitHub
    </a>
  </div>
</div>
"""
st.markdown(_FOOTER_HTML, unsafe_allow_html=True)
