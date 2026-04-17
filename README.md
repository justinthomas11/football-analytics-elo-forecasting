# FootballIQ — Elo + LLM Match Intelligence Engine

> **XGBoost + Elo ratings + Claude claude-sonnet-4-20250514 → structured pre-match briefings**

FootballIQ upgrades the existing football analytics project into a production-grade
prediction system that layers a RAG-augmented LLM reasoning engine on top of a
calibrated XGBoost classifier and a live Elo rating system.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FootballIQ Pipeline                         │
└─────────────────────────────────────────────────────────────────────┘

 User Input: home_team, away_team, match_date
       │
       ▼
 ┌─────────────┐    ┌──────────────────────────────────────┐
 │  Elo Lookup │    │  XGBClassifier (CalibratedClassifierCV)│
 │  elo_state  │    │  Features: EloDiff, FormDiff,         │
 │  .pkl       │    │  ShotDiff, OddHome/Draw/Away, …       │
 └──────┬──────┘    └──────────────────┬───────────────────┘
        │ 40%                          │ 60%
        └──────────────┬───────────────┘
                       │  Weighted Blend
                       ▼
              ┌─────────────────┐
              │ Blended Probs   │  home / draw / away
              └────────┬────────┘
                       │
                       ▼
             ┌──────────────────┐
             │  src/scraper.py  │  FBref last-5 form, goals,
             │  (FBref or CSV)  │  H2H from Matches.csv
             └────────┬─────────┘
                      │
                      ▼
           ┌───────────────────────┐
           │  src/rag_context.py   │  Structured natural-language
           │  Context builder      │  context block
           └──────────┬────────────┘
                      │
                      ▼
           ┌──────────────────────────────────┐
           │  src/llm_analyst.py              │
           │  Claude claude-sonnet-4-20250514                 │
           │  → structured JSON briefing      │
           └──────────┬───────────────────────┘
                      │
            ┌─────────┴──────────┐
            ▼                    ▼
     ┌─────────────┐    ┌────────────────┐
     │   app.py    │    │ src/tracking.py│
     │  Streamlit  │    │  MLflow logger │
     │  Dashboard  │    │  (mlflow.db)   │
     └─────────────┘    └────────────────┘
```

---

## Project Structure

```
Football_Analytics_Dashboard/
├── app.py                    # Streamlit dashboard
├── train.py                  # XGB training + artefact export
├── requirements.txt
├── .env.example              # API key template
├── Data/                     # Existing match CSVs
│   ├── Matches.csv           # 230k+ historical matches
│   └── EloRatings.csv        # Club Elo snapshots
├── models/                   # Auto-created by train.py
│   ├── xgb_model.pkl
│   ├── elo_state.pkl
│   ├── team_list.pkl
│   └── feature_names.pkl
├── src/
│   ├── __init__.py
│   ├── scraper.py
│   ├── rag_context.py
│   ├── llm_analyst.py
│   ├── predict.py
│   └── tracking.py
├── FootballAnalytics.ipynb   # Original EDA & modelling notebook
├── mlflow.db                 # MLflow experiment store
└── mlruns/
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API key
cp .env.example .env
# Edit .env → add your ANTHROPIC_API_KEY

# 3. Train the model (saves artefacts to models/)
python train.py

# 4. Launch dashboard
streamlit run app.py
```

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `train.py` | Trains calibrated XGBClassifier; exports `models/*.pkl` |
| `src/scraper.py` | FBref form scraper + Matches.csv H2H fallback |
| `src/rag_context.py` | Builds structured natural-language LLM context |
| `src/llm_analyst.py` | Claude claude-sonnet-4-20250514 wrapper → structured JSON |
| `src/predict.py` | End-to-end pipeline orchestrator |
| `src/tracking.py` | MLflow experiment logger |
| `app.py` | Streamlit dashboard (charts, cards, expanders) |

---

## LLM Output Schema

```json
{
  "predicted_outcome":  "Home Win | Draw | Away Win",
  "confidence":         "High | Medium | Low",
  "key_factors":        ["factor1", "factor2"],
  "risk_flags":         ["risk1"],
  "reasoning_summary":  "2-3 sentence plain English.",
  "elo_probability":    {"home": 0.48, "draw": 0.28, "away": 0.24}
}
```

Confidence thresholds: **High** > 55% · **Medium** 40–55% · **Low** < 40%

---

## Blending Formula

```
blended = (XGB_prob × rf_weight + Elo_prob × elo_weight) / (rf_weight + elo_weight)
```

Default: **60% XGB / 40% Elo** — adjustable via slider in dashboard.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost + Scikit-learn |
| Elo | Custom logistic formula with draw adjustment |
| LLM | Anthropic Claude claude-sonnet-4-20250514 |
| Scraping | Requests + BeautifulSoup4 |
| Dashboard | Streamlit + Plotly |
| Tracking | MLflow (SQLite) |

---

## View MLflow Experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
