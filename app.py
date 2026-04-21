"""
app.py — FootballIQ Streamlit Dashboard
Run with:  streamlit run app.py
"""

import sys
import json
import logging
import pickle
from pathlib import Path
from datetime import date

import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="FootballIQ — Match Intelligence Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0e1a; color: #e8eaf0; }
#MainMenu, footer, header { visibility: hidden; }

.hero-banner {
    background: linear-gradient(135deg, #1a2744 0%, #0f1629 40%, #162038 100%);
    border: 1px solid #2a3a5c; border-radius: 16px;
    padding: 2.5rem 2rem 2rem; margin-bottom: 2rem;
    text-align: center; position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(99,179,237,0.08) 0%, transparent 65%);
}
.hero-title {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #63b3ed, #a78bfa, #f687b3);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem;
}
.hero-sub { font-size: 0.95rem; color: #718096; font-weight: 400; letter-spacing: 0.4px; }

.card {
    background: #111827; border: 1px solid #1f2d47;
    border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
}
.card-title {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: #4a6fa5; margin-bottom: 0.6rem;
}

.metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px; background: #111827;
    border: 1px solid #1f2d47; border-radius: 10px;
    padding: 1rem 1.2rem; text-align: center;
}
.metric-label { font-size: 0.65rem; letter-spacing: 1px; text-transform: uppercase; color: #4a6fa5; margin-bottom: 0.3rem; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #e8eaf0; }
.metric-sub   { font-size: 0.72rem; color: #718096; margin-top: 0.15rem; }

.outcome-badge {
    display: inline-block; font-size: 1.8rem; font-weight: 800;
    padding: 0.6rem 1.8rem; border-radius: 50px; margin-bottom: 0.5rem;
}
.outcome-home { background: linear-gradient(135deg,#1a4731,#22543d); color:#68d391; border:1px solid #276749; }
.outcome-draw { background: linear-gradient(135deg,#44337a,#553c9a); color:#b794f4; border:1px solid #6b46c1; }
.outcome-away { background: linear-gradient(135deg,#742a2a,#9b2c2c); color:#fc8181; border:1px solid #c53030; }

.conf-high   { background:#1c4532; color:#68d391; border:1px solid #276749; }
.conf-medium { background:#44337a; color:#d6bcfa; border:1px solid #6b46c1; }
.conf-low    { background:#742a2a; color:#fc8181; border:1px solid #c53030; }
.conf-badge  { display:inline-block; padding:0.25rem 0.9rem; border-radius:50px; font-size:0.8rem; font-weight:600; }

.factor-item {
    display:flex; align-items:flex-start; gap:0.5rem;
    padding:0.45rem 0; border-bottom:1px solid #1a2235;
    font-size:0.88rem; color:#a0aec0; line-height:1.5;
}
.factor-item:last-child { border-bottom:none; }
.bullet-green { color:#68d391; font-size:1rem; flex-shrink:0; }
.bullet-red   { color:#fc8181; font-size:1rem; flex-shrink:0; }

.divider { border:none; border-top:1px solid #1f2d47; margin:1.2rem 0; }

.form-badge {
    display:inline-block; width:28px; height:28px;
    border-radius:6px; text-align:center; line-height:28px;
    font-size:0.75rem; font-weight:700; margin:0 2px;
}
.fb-W { background:#1a4731; color:#68d391; }
.fb-D { background:#2d3748; color:#a0aec0; }
.fb-L { background:#742a2a; color:#fc8181; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #3b5fc0, #6b46c1);
    color: white; font-weight: 700; font-size: 1rem;
    border: none; border-radius: 10px; padding: 0.75rem 2.5rem;
    width: 100%; cursor: pointer; transition: all 0.2s ease;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #4c6ef5, #7c3aed);
    transform: translateY(-1px); box-shadow: 0 6px 20px rgba(99,102,241,0.35);
}

.stSelectbox label { color:#a0aec0 !important; font-size:0.82rem !important; }
.stSelectbox > div > div {
    background:#111827 !important; border:1px solid #2a3a5c !important;
    border-radius:8px !important; color:#e8eaf0 !important;
}

.source-tag {
    display:inline-block; font-size:0.68rem; padding:0.15rem 0.6rem;
    border-radius:4px; font-weight:600; letter-spacing:0.5px;
    text-transform:uppercase; margin-left:0.4rem;
}
.src-fbref      { background:#1a3a5c; color:#63b3ed; }
.src-livescore  { background:#1a3a5c; color:#63b3ed; }
.src-historical { background:#1a2d1a; color:#68d391; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_teams():
    pkl = ROOT / "models" / "team_list.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return sorted(pickle.load(f))
    # Fallback: read directly from CSV
    csv = ROOT / "Data" / "Matches.csv"
    if csv.exists():
        import pandas as pd
        df = pd.read_csv(csv, usecols=["HomeTeam", "AwayTeam"], low_memory=False)
        return sorted(set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna()))
    return ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Tottenham"]


def model_ready() -> bool:
    return (ROOT / "models" / "xgb_model.pkl").exists()


# ── Chart ─────────────────────────────────────────────────────────────────────

def prob_chart(blended: dict, home: str, away: str) -> go.Figure:
    labels  = [f"🏠 {home}", "🤝 Draw", f"✈️ {away}"]
    values  = [blended["home"] * 100, blended["draw"] * 100, blended["away"] * 100]
    colours = ["#3b82f6", "#8b5cf6", "#ef4444"]

    fig = go.Figure()
    for label, value, colour in zip(labels, values, colours):
        fig.add_trace(go.Bar(
            x=[value], y=[label], orientation="h",
            marker=dict(color=colour, opacity=0.85),
            text=f"{value:.1f}%", textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=13, family="Inter"),
            name=label,
            hovertemplate=f"<b>{label}</b><br>{value:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.8)",
        height=210, margin=dict(l=10, r=20, t=10, b=10),
        showlegend=False, font=dict(family="Inter", color="#a0aec0"),
        xaxis=dict(range=[0,100], ticksuffix="%", gridcolor="#1f2d47",
                   zerolinecolor="#1f2d47", tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=12, color="#e8eaf0"), gridcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def form_html(lst: list) -> str:
    if not lst:
        return "<span style='color:#4a6fa5;font-size:0.82rem;'>No form data</span>"
    return "".join(f"<span class='form-badge fb-{r}'>{r}</span>" for r in lst)

def outcome_cls(o: str) -> str:
    o = o.lower()
    return "outcome-home" if "home" in o else ("outcome-draw" if "draw" in o else "outcome-away")

def conf_cls(c: str) -> str:
    return {"high": "conf-high", "medium": "conf-medium", "low": "conf-low"}.get(c.lower(), "conf-low")

def src_badge(s: str) -> str:
    cls = "src-livescore" if s in ["livescore", "fbref", "sofascore"] else "src-historical"
    return f"<span class='source-tag {cls}'>{s.upper()}</span>"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">⚽ FootballIQ</div>
        <div class="hero-sub">Elo + Machine Learning + LLM Match Intelligence Engine</div>
    </div>""", unsafe_allow_html=True)

    if not model_ready():
        st.error("⚠️ Model not found. Run **`python train.py`** first.", icon="🚨")
        st.stop()

    teams = load_teams()

    # ── Inputs ────────────────────────────────────────────────────────────
    st.markdown("<div class='card'><div class='card-title'>📋 Match Setup</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        idx_h   = teams.index("Arsenal") if "Arsenal" in teams else 0
        home_t  = st.selectbox("🏠 Home Team", teams, index=idx_h)
    with c2:
        idx_a   = (idx_h + 10) % len(teams)
        away_t  = st.selectbox("✈️ Away Team", teams, index=idx_a)
    m_date = date.today()

    ac1, ac2 = st.columns(2)
    with ac1:
        rf_w = st.slider("XGB Model weight", 0.0, 1.0, 0.60, 0.05)
    with ac2:
        use_live_data = st.toggle("🌐 Try LiveScore API", value=False,
                              help="Uses LiveScore RapidAPI for real-time form data. Falls back to historical CSV if unavailable.")
    elo_w = round(1.0 - rf_w, 2)
    st.markdown("</div>", unsafe_allow_html=True)

    if home_t == away_t:
        st.warning("Select different teams.", icon="⚠️"); st.stop()

    generate = st.button("⚡ Generate Match Analysis", use_container_width=True)

    if not generate:
        st.markdown("""
        <div style='text-align:center;padding:3rem 0;color:#4a6fa5;'>
            <div style='font-size:3rem;margin-bottom:1rem;'>📊</div>
            <div style='font-size:1rem;'>Select teams and click <b>Generate Match Analysis</b></div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Run pipeline ──────────────────────────────────────────────────────
    with st.spinner("🔍 Running FootballIQ pipeline …"):
        try:
            from src.predict  import run_prediction
            from src.tracking import log_prediction
            result = run_prediction(
                home_team=home_t, away_team=away_t,
                match_date=str(m_date),
                rf_weight=rf_w, elo_weight=elo_w,
                use_llm=True, use_live_data=use_live_data,
            )
            log_prediction(result)
        except FileNotFoundError as e:
            st.error(str(e)); st.stop()
        except Exception as e:
            st.error(f"Pipeline error: {e}"); st.stop()

    blended  = result["blended_probabilities"]
    elo_r    = result["elo_ratings"]
    rf_p     = result["rf_probabilities"]
    elo_p    = result["elo_probabilities"]
    ctx      = result["scraped_context"]
    llm      = result.get("llm_report", {})

    predicted = llm.get("predicted_outcome", "—")
    conf      = llm.get("confidence", "—")
    factors   = llm.get("key_factors", [])
    risks     = llm.get("risk_flags", [])
    reasoning = llm.get("reasoning_summary", "")
    llm_err   = llm.get("_llm_error", False)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    left, right = st.columns([1.1, 1], gap="large")

    # ══ LEFT ═════════════════════════════════════════════════════════════
    with left:
        # Prob chart
        st.markdown("<div class='card'><div class='card-title'>📈 Blended Win Probabilities</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(prob_chart(blended, home_t, away_t),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # Elo cards
        diff = elo_r["home"] - elo_r["away"]
        dcol = "#68d391" if diff >= 0 else "#fc8181"
        side = "Home advantage" if diff >= 0 else "Away advantage"
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>🏆 Elo Ratings</div>
          <div class='metric-row'>
            <div class='metric-card'>
              <div class='metric-label'>🏠 {home_t}</div>
              <div class='metric-value'>{elo_r['home']:.0f}</div>
              <div class='metric-sub'>Home Elo</div>
            </div>
            <div class='metric-card'>
              <div class='metric-label'>Δ Differential</div>
              <div class='metric-value' style='color:{dcol};'>{diff:+.0f}</div>
              <div class='metric-sub'>{side}</div>
            </div>
            <div class='metric-card'>
              <div class='metric-label'>✈️ {away_t}</div>
              <div class='metric-value'>{elo_r['away']:.0f}</div>
              <div class='metric-sub'>Away Elo</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Form & context
        h2h = ctx.get("h2h_record", {})
        hw  = h2h.get("home_wins", 0)
        aw2 = h2h.get("away_wins", 0)
        dr  = h2h.get("draws", 0)
        h2h_str = (f"{home_t} {hw}W / {dr}D / {aw2}W {away_t}"
                   if hw + aw2 + dr > 0 else "No H2H in dataset")
        hga = ctx.get("home_goals_avg")
        aga = ctx.get("away_goals_avg")

        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>📋 Pre-Match Context {src_badge(ctx.get('source','historical'))}</div>
          <table style='width:100%;border-collapse:collapse;'>
            <tr>
              <td style='padding:0.5rem 0;color:#718096;font-size:0.78rem;width:40%;'>🏠 {home_t} form</td>
              <td>{form_html(ctx.get('home_form',[]))}</td>
            </tr>
            <tr>
              <td style='padding:0.5rem 0;color:#718096;font-size:0.78rem;'>✈️ {away_t} form</td>
              <td>{form_html(ctx.get('away_form',[]))}</td>
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
              <td style='color:#e8eaf0;font-size:0.85rem;'>{h2h_str}</td>
            </tr>
          </table>
        </div>""", unsafe_allow_html=True)

        with st.expander("📊 Probability Breakdown (XGB vs Elo vs Blended)"):
            st.markdown(f"""
| Outcome | XGB Model | Elo Model | Blended ({rf_w:.0%} / {elo_w:.0%}) |
|---------|-----------|-----------|------------|
| 🏠 Home Win | {rf_p['home']:.1%} | {elo_p['home']:.1%} | **{blended['home']:.1%}** |
| 🤝 Draw     | {rf_p['draw']:.1%} | {elo_p['draw']:.1%} | **{blended['draw']:.1%}** |
| ✈️ Away Win  | {rf_p['away']:.1%} | {elo_p['away']:.1%} | **{blended['away']:.1%}** |
""")

    # ══ RIGHT ════════════════════════════════════════════════════════════
    with right:
        if llm_err:
            st.warning("⚠️ LLM unavailable — showing model-only prediction.", icon="⚠️")

        oc = outcome_cls(predicted)
        cc = conf_cls(conf)

        st.markdown(f"""
        <div class='card' style='text-align:center;'>
          <div class='card-title'>🎯 Predicted Outcome</div>
          <div class='outcome-badge {oc}'>{predicted}</div><br>
          <span class='conf-badge {cc}'>● {conf} Confidence</span>
          <div style='margin-top:0.8rem;color:#4a6fa5;font-size:0.75rem;'>
            {home_t} vs {away_t} · {m_date}
          </div>
        </div>""", unsafe_allow_html=True)

        if reasoning:
            st.markdown(f"""
            <div class='card'>
              <div class='card-title'>🧠 Analyst Reasoning</div>
              <p style='color:#a0aec0;font-size:0.88rem;line-height:1.7;margin:0;'>{reasoning}</p>
            </div>""", unsafe_allow_html=True)

        if factors:
            st.markdown(
                "<div class='card'><div class='card-title'>✅ Key Factors</div>" +
                "".join(f"<div class='factor-item'><span class='bullet-green'>▸</span>{f}</div>"
                        for f in factors) +
                "</div>", unsafe_allow_html=True)

        if risks:
            st.markdown(
                "<div class='card'><div class='card-title'>🚩 Risk Flags</div>" +
                "".join(f"<div class='factor-item'><span class='bullet-red'>⚠</span>{r}</div>"
                        for r in risks) +
                "</div>", unsafe_allow_html=True)

        with st.expander("🔧 Raw JSON Report"):
            st.code(json.dumps(
                {k: v for k, v in result.items() if k != "rag_context"},
                indent=2, default=str), language="json")

        with st.expander("📝 RAG Context Block"):
            st.code(result.get("rag_context", ""), language="text")


if __name__ == "__main__":
    main()
