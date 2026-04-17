"""
app.py — FootballIQ Streamlit Dashboard
========================================
Run with:  streamlit run app.py
"""

import sys
import json
import logging
import pickle
from pathlib import Path
from datetime import date, datetime

import streamlit as st
import plotly.graph_objects as go

# ── Path setup (so src.* imports resolve) ────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FootballIQ — Match Intelligence Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0e1a; color: #e8eaf0; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1a2744 0%, #0f1629 40%, #162038 100%);
    border: 1px solid #2a3a5c;
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(99,179,237,0.08) 0%, transparent 65%);
}
.hero-title {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #63b3ed, #a78bfa, #f687b3);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem;
}
.hero-sub {
    font-size: 0.95rem; color: #718096; font-weight: 400;
    letter-spacing: 0.4px;
}

/* Cards */
.card {
    background: #111827;
    border: 1px solid #1f2d47;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: #4a6fa5; margin-bottom: 0.6rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px;
    background: #111827;
    border: 1px solid #1f2d47;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-label { font-size: 0.65rem; letter-spacing: 1px; text-transform: uppercase; color: #4a6fa5; margin-bottom: 0.3rem; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #e8eaf0; }
.metric-sub   { font-size: 0.72rem; color: #718096; margin-top: 0.15rem; }

/* Outcome badge */
.outcome-badge {
    display: inline-block;
    font-size: 1.8rem; font-weight: 800;
    padding: 0.6rem 1.8rem;
    border-radius: 50px;
    margin-bottom: 0.5rem;
}
.outcome-home { background: linear-gradient(135deg, #1a4731, #22543d); color: #68d391; border: 1px solid #276749; }
.outcome-draw { background: linear-gradient(135deg, #44337a, #553c9a); color: #b794f4; border: 1px solid #6b46c1; }
.outcome-away { background: linear-gradient(135deg, #742a2a, #9b2c2c); color: #fc8181; border: 1px solid #c53030; }

/* Confidence pill */
.conf-high   { background:#1c4532; color:#68d391; border:1px solid #276749; }
.conf-medium { background:#44337a; color:#d6bcfa; border:1px solid #6b46c1; }
.conf-low    { background:#742a2a; color:#fc8181; border:1px solid #c53030; }
.conf-badge  { display:inline-block; padding:0.25rem 0.9rem; border-radius:50px; font-size:0.8rem; font-weight:600; }

/* Factor/risk bullets */
.factor-item {
    display: flex; align-items: flex-start; gap: 0.5rem;
    padding: 0.45rem 0; border-bottom: 1px solid #1a2235;
    font-size: 0.88rem; color: #a0aec0; line-height: 1.5;
}
.factor-item:last-child { border-bottom: none; }
.bullet-green { color: #68d391; font-size: 1rem; flex-shrink: 0; }
.bullet-red   { color: #fc8181; font-size: 1rem; flex-shrink: 0; }

/* Divider */
.divider { border: none; border-top: 1px solid #1f2d47; margin: 1.2rem 0; }

/* Form badge */
.form-badge {
    display: inline-block; width: 28px; height: 28px;
    border-radius: 6px; text-align: center; line-height: 28px;
    font-size: 0.75rem; font-weight: 700; margin: 0 2px;
}
.fb-W { background:#1a4731; color:#68d391; }
.fb-D { background:#2d3748; color:#a0aec0; }
.fb-L { background:#742a2a; color:#fc8181; }

/* Generate button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #3b5fc0, #6b46c1);
    color: white; font-weight: 700; font-size: 1rem;
    border: none; border-radius: 10px; padding: 0.75rem 2.5rem;
    width: 100%; cursor: pointer; transition: all 0.2s ease;
    letter-spacing: 0.3px;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #4c6ef5, #7c3aed);
    transform: translateY(-1px); box-shadow: 0 6px 20px rgba(99,102,241,0.35);
}

/* Select boxes and date input */
.stSelectbox label, .stDateInput label { color: #a0aec0 !important; font-size: 0.82rem !important; }
.stSelectbox > div > div, .stDateInput > div > div {
    background: #111827 !important; border: 1px solid #2a3a5c !important;
    border-radius: 8px !important; color: #e8eaf0 !important;
}

/* Source badge */
.source-tag {
    display: inline-block; font-size: 0.68rem; padding: 0.15rem 0.6rem;
    border-radius: 4px; font-weight: 600; letter-spacing: 0.5px;
    text-transform: uppercase; margin-left: 0.4rem;
}
.src-fbref      { background:#1a3a5c; color:#63b3ed; }
.src-historical { background:#1a2d1a; color:#68d391; }
.src-simulated  { background:#2d2d0f; color:#ecc94b; }
</style>
""", unsafe_allow_html=True)

# ── Load team list ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_team_list():
    model_dir = ROOT / "models"
    pkl = model_dir / "team_list.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            teams = pickle.load(f)
        return sorted(teams)
    # Fallback: pull from Matches.csv
    fallback = ROOT.parent / "Data" / "Matches.csv"
    if fallback.exists():
        import pandas as pd
        df = pd.read_csv(fallback, usecols=["HomeTeam", "AwayTeam"], low_memory=False)
        teams = sorted(set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna()))
        return teams
    return ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Tottenham"]


@st.cache_resource(show_spinner=False)
def model_ready() -> bool:
    return (ROOT / "models" / "xgb_model.pkl").exists()


# ── UI helpers ────────────────────────────────────────────────────────────────

def form_badges_html(form_list: list) -> str:
    if not form_list:
        return "<span style='color:#4a6fa5;font-size:0.82rem;'>No form data</span>"
    badges = "".join(f"<span class='form-badge fb-{r}'>{r}</span>" for r in form_list)
    return badges


def probability_chart(blended: dict, home_team: str, away_team: str) -> go.Figure:
    labels = [f"🏠 {home_team}", "🤝 Draw", f"✈️ {away_team}"]
    values = [
        blended.get("home", 0) * 100,
        blended.get("draw", 0) * 100,
        blended.get("away", 0) * 100,
    ]
    colours = ["#3b82f6", "#8b5cf6", "#ef4444"]

    fig = go.Figure()
    for i, (label, value, colour) in enumerate(zip(labels, values, colours)):
        fig.add_trace(go.Bar(
            x=[value], y=[label], orientation="h",
            marker=dict(
                color=colour,
                opacity=0.85,
                line=dict(color=colour, width=0),
            ),
            text=f"{value:.1f}%",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=13, family="Inter"),
            name=label,
            hovertemplate=f"<b>{label}</b><br>Probability: {value:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.8)",
        height=210,
        margin=dict(l=10, r=20, t=10, b=10),
        showlegend=False,
        font=dict(family="Inter", color="#a0aec0"),
        xaxis=dict(
            range=[0, 100],
            ticksuffix="%",
            gridcolor="#1f2d47",
            zerolinecolor="#1f2d47",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#e8eaf0"),
            gridcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def outcome_class(outcome: str) -> str:
    o = outcome.lower()
    if "home" in o: return "outcome-home"
    if "draw" in o: return "outcome-draw"
    return "outcome-away"


def confidence_class(conf: str) -> str:
    c = conf.lower()
    if c == "high":   return "conf-high"
    if c == "medium": return "conf-medium"
    return "conf-low"


def source_badge(source: str) -> str:
    cls_map = {"fbref": "src-fbref", "historical": "src-historical", "simulated": "src-simulated"}
    cls = cls_map.get(source, "src-historical")
    return f"<span class='source-tag {cls}'>{source}</span>"


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # Hero
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">⚽ FootballIQ</div>
        <div class="hero-sub">Elo + Machine Learning + LLM Match Intelligence Engine</div>
    </div>
    """, unsafe_allow_html=True)

    if not model_ready():
        st.error(
            "⚠️ Model artefacts not found. "
            "Please run **`python train.py`** from the `footballiq/` directory first.",
            icon="🚨",
        )
        st.stop()

    teams = load_team_list()

    # ── Input section ─────────────────────────────────────────────────────
    st.markdown("<div class='card'><div class='card-title'>📋 Match Setup</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 1.2])
    with col1:
        default_home = teams.index("Arsenal") if "Arsenal" in teams else 0
        home_team = st.selectbox("🏠 Home Team", teams, index=default_home, key="home_team")
    with col2:
        # Default away = next team after home
        default_away = (default_home + 10) % len(teams)
        away_team = st.selectbox("✈️ Away Team", teams, index=default_away, key="away_team")
    with col3:
        match_date = st.date_input("📅 Match Date", value=date.today(), key="match_date")

    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        rf_weight  = st.slider("XGB Model weight", 0.0, 1.0, 0.60, 0.05,
                               help="Remaining weight goes to Elo model")
    with adv_col2:
        use_fbref = st.toggle("🌐 Try FBref live scrape", value=False,
                              help="Falls back to historical CSV if unavailable")
    elo_weight = round(1.0 - rf_weight, 2)

    st.markdown("</div>", unsafe_allow_html=True)

    if home_team == away_team:
        st.warning("⚠️ Home and Away teams must be different.", icon="⚠️")
        st.stop()

    generate = st.button("⚡ Generate Match Analysis", use_container_width=True)

    if not generate:
        st.markdown("""
        <div style='text-align:center; padding: 3rem 0; color:#4a6fa5;'>
            <div style='font-size:3rem;margin-bottom:1rem;'>📊</div>
            <div style='font-size:1rem;'>Select teams and click <b>Generate Match Analysis</b></div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Run pipeline ──────────────────────────────────────────────────────
    with st.spinner("🔍 Running FootballIQ pipeline …"):
        try:
            from src.predict  import run_prediction
            from src.tracking import log_prediction

            result = run_prediction(
                home_team=home_team,
                away_team=away_team,
                match_date=str(match_date),
                rf_weight=rf_weight,
                elo_weight=elo_weight,
                use_llm=True,
                use_fbref=use_fbref,
            )
            log_prediction(result)

        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

    # ── Unpack result ─────────────────────────────────────────────────────
    blended   = result["blended_probabilities"]
    elo_r     = result["elo_ratings"]
    rf_probs  = result["rf_probabilities"]
    elo_probs = result["elo_probabilities"]
    context   = result["scraped_context"]
    llm       = result.get("llm_report", {})

    predicted_outcome = llm.get("predicted_outcome", "—")
    confidence        = llm.get("confidence", "—")
    key_factors       = llm.get("key_factors", [])
    risk_flags        = llm.get("risk_flags", [])
    reasoning         = llm.get("reasoning_summary", "")
    llm_error         = llm.get("_llm_error", False)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Layout: Left column (probs + context) | Right column (LLM report) ──
    left, right = st.columns([1.1, 1], gap="large")

    # ══ LEFT ══════════════════════════════════════════════════════════════

    with left:
        # Probability bar chart
        st.markdown("<div class='card'><div class='card-title'>📈 Blended Win Probabilities</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(
            probability_chart(blended, home_team, away_team),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Elo metric cards
        elo_diff = elo_r["home"] - elo_r["away"]
        diff_col = "#68d391" if elo_diff >= 0 else "#fc8181"
        diff_side = "Home advantage" if elo_diff >= 0 else "Away advantage"

        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>🏆 Elo Ratings</div>
          <div class='metric-row'>
            <div class='metric-card'>
              <div class='metric-label'>🏠 {home_team}</div>
              <div class='metric-value'>{elo_r['home']:.0f}</div>
              <div class='metric-sub'>Home Elo</div>
            </div>
            <div class='metric-card'>
              <div class='metric-label'>Δ Differential</div>
              <div class='metric-value' style='color:{diff_col};'>{elo_diff:+.0f}</div>
              <div class='metric-sub'>{diff_side}</div>
            </div>
            <div class='metric-card'>
              <div class='metric-label'>✈️ {away_team}</div>
              <div class='metric-value'>{elo_r['away']:.0f}</div>
              <div class='metric-sub'>Away Elo</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Form & context
        src_tag   = source_badge(context.get("source", "historical"))
        home_form = form_badges_html(context.get("home_form", []))
        away_form = form_badges_html(context.get("away_form", []))
        h2h       = context.get("h2h_record", {})
        hga       = context.get("home_goals_avg")
        aga       = context.get("away_goals_avg")

        h2h_text = (
            f"{home_team} {h2h.get('home_wins',0)}W / "
            f"{h2h.get('draws',0)}D / "
            f"{h2h.get('away_wins',0)}W {away_team}"
            if (h2h.get("home_wins",0)+h2h.get("draws",0)+h2h.get("away_wins",0)) > 0
            else "No H2H data in dataset"
        )

        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>📋 Pre-Match Context {src_tag}</div>
          <table style='width:100%;border-collapse:collapse;'>
            <tr>
              <td style='padding:0.5rem 0;color:#718096;font-size:0.78rem;width:38%;'>🏠 {home_team} form</td>
              <td>{home_form}</td>
            </tr>
            <tr>
              <td style='padding:0.5rem 0;color:#718096;font-size:0.78rem;'>✈️ {away_team} form</td>
              <td>{away_form}</td>
            </tr>
            <tr>
              <td style='padding:0.5rem 0;color:#718096;font-size:0.78rem;'>Avg goals (home)</td>
              <td style='color:#e8eaf0;font-size:0.88rem;'>{f"{hga:.2f}" if hga else "N/A"} per game</td>
            </tr>
            <tr>
              <td style='padding:0.5rem 0;color:#718096;font-size:0.78rem;'>Avg goals (away)</td>
              <td style='color:#e8eaf0;font-size:0.88rem;'>{f"{aga:.2f}" if aga else "N/A"} per game</td>
            </tr>
            <tr>
              <td style='padding:0.5rem 0;color:#718096;font-size:0.78rem;'>H2H record</td>
              <td style='color:#e8eaf0;font-size:0.85rem;'>{h2h_text}</td>
            </tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

        # Prob breakdown table
        with st.expander("📊 Probability Breakdown (XGB vs Elo vs Blended)"):
            st.markdown(f"""
            | Outcome | XGB Model | Elo Model | Blended ({rf_weight:.0%} / {elo_weight:.0%}) |
            |---------|-----------|-----------|------------|
            | 🏠 Home Win | {rf_probs['home']:.1%} | {elo_probs['home']:.1%} | **{blended['home']:.1%}** |
            | 🤝 Draw | {rf_probs['draw']:.1%} | {elo_probs['draw']:.1%} | **{blended['draw']:.1%}** |
            | ✈️ Away Win | {rf_probs['away']:.1%} | {elo_probs['away']:.1%} | **{blended['away']:.1%}** |
            """)

    # ══ RIGHT ═════════════════════════════════════════════════════════════

    with right:
        if llm_error:
            st.warning("⚠️ LLM analysis unavailable — showing model-based prediction only.", icon="⚠️")

        # Predicted outcome
        oc_class = outcome_class(predicted_outcome)
        cf_class = confidence_class(confidence)

        st.markdown(f"""
        <div class='card' style='text-align:center;'>
          <div class='card-title'>🎯 Predicted Outcome</div>
          <div class='outcome-badge {oc_class}'>{predicted_outcome}</div>
          <br>
          <span class='conf-badge {cf_class}'>● {confidence} Confidence</span>
          <div style='margin-top:0.8rem;color:#4a6fa5;font-size:0.75rem;'>
            {home_team} vs {away_team} · {match_date}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Reasoning summary
        if reasoning:
            st.markdown(f"""
            <div class='card'>
              <div class='card-title'>🧠 Analyst Reasoning</div>
              <p style='color:#a0aec0;font-size:0.88rem;line-height:1.7;margin:0;'>{reasoning}</p>
            </div>
            """, unsafe_allow_html=True)

        # Key factors
        if key_factors:
            factors_html = "".join(
                f"<div class='factor-item'><span class='bullet-green'>▸</span>{f}</div>"
                for f in key_factors
            )
            st.markdown(f"""
            <div class='card'>
              <div class='card-title'>✅ Key Factors</div>
              {factors_html}
            </div>
            """, unsafe_allow_html=True)

        # Risk flags
        if risk_flags:
            risks_html = "".join(
                f"<div class='factor-item'><span class='bullet-red'>⚠</span>{r}</div>"
                for r in risk_flags
            )
            st.markdown(f"""
            <div class='card'>
              <div class='card-title'>🚩 Risk Flags</div>
              {risks_html}
            </div>
            """, unsafe_allow_html=True)

        # Raw JSON expander
        with st.expander("🔧 Raw JSON Report (technical view)"):
            safe_result = {k: v for k, v in result.items() if k != "rag_context"}
            st.code(json.dumps(safe_result, indent=2, default=str), language="json")

        with st.expander("📝 RAG Context Block (injected into LLM)"):
            st.code(result.get("rag_context", "No context available"), language="text")


if __name__ == "__main__":
    main()
