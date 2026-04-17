# FootballIQ — LLM Match Intelligence Engine

> **Elo ratings + XGBoost + Claude claude-sonnet-4-20250514 → structured pre-match briefings**

FootballIQ is a production-grade football match prediction system that layers a RAG-augmented LLM reasoning engine on top of a calibrated XGBoost classifier and a live Elo rating system.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FootballIQ Pipeline                         │
└─────────────────────────────────────────────────────────────────────┘

 User Input: home_team, away_team, match_date
       │
       ▼
 ┌─────────────┐    ┌──────────────────┐
 │  Elo Lookup │    │  XGB Classifier  │   ← trained on 230k+ matches
 │  (elo_state │    │  (calibrated,    │      (EloDiff, FormDiff,
 │   .pkl)     │    │   isotonic CV)   │       OddHome, ShotDiff …)
 └──────┬──────┘    └────────┬─────────┘
        │                   │
        │  Elo probs (40%)  │  XGB probs (60%)
        └─────────┬─────────┘
                  │ Weighted Blend
                  ▼
          ┌──────────────┐
          │  Blended     │  home / draw / away probabilities
          │  Probs       │
          └──────┬───────┘
                 │
                 ▼
       ┌─────────────────┐
       │  scraper.py     │  FBref last-5 form, goals avg, H2H
       │  (live scrape   │  → falls back to Matches.csv if blocked
       │   or CSV)       │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │  rag_context.py │  Structured natural-language context block
       └────────┬────────┘
                │
                ▼
       ┌──────────────────────────────────┐
       │  llm_analyst.py                  │
       │  Claude claude-sonnet-4-20250514                 │
       │  System: professional analyst    │
       │  User: RAG block + blended probs │
       │  Output: structured JSON         │
       └────────┬─────────────────────────┘
                │
                ▼
       ┌─────────────────┐        ┌──────────────────┐
       │  app.py         │        │  tracking.py     │
       │  Streamlit UI   │        │  MLflow logger   │
       │  (Plotly charts │        │  (params,metrics │
       │   LLM cards)    │        │   tags, run_id)  │
       └─────────────────┘        └──────────────────┘
```

---

## Project Structure

```
footballiq/
├── app.py                    # Streamlit dashboard
├── train.py                  # XGB training + artefact export
├── requirements.txt
├── .env.example              # API key template
├── data/                     # Symlink or copy of match CSVs
│   ├── Matches.csv           # 230k+ historical matches
│   └── EloRatings.csv        # Club Elo snapshots
├── models/                   # Auto-created by train.py
│   ├── xgb_model.pkl         # CalibratedClassifierCV
│   ├── elo_state.pkl         # {team → latest Elo}
│   ├── team_list.pkl         # sorted team names
│   └── feature_names.pkl     # feature column order
└── src/
    ├── __init__.py
    ├── scraper.py            # FBref + CSV context scraper
    ├── rag_context.py        # Natural-language context builder
    ├── llm_analyst.py        # Claude API wrapper
    ├── predict.py            # End-to-end pipeline orchestrator
    └── tracking.py           # MLflow experiment logger
```

---

## Quick Start

### 1. Install dependencies
```bash
cd footballiq
pip install -r requirements.txt
```

### 2. Set your API key
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 3. Link data (or copy CSVs)
The training script auto-discovers CSVs from `../Data/` (the parent project folder).
If you want a self-contained setup:
```bash
mkdir -p data
# Copy Matches.csv and EloRatings.csv into footballiq/data/
```

### 4. Train the model
```bash
python train.py
```
This generates all `.pkl` artefacts in `models/`.

### 5. Launch the dashboard
```bash
streamlit run app.py
```

---

## Module Reference

### `train.py`
Loads `Matches.csv`, engineers 13 features (EloDiff, FormDiff, ShotDiff,
etc.), trains a calibrated XGBClassifier, and persists model artefacts.

| Artefact | Contents |
|----------|----------|
| `xgb_model.pkl` | `CalibratedClassifierCV` (isotonic, cv=3) |
| `elo_state.pkl` | `{team_name: float}` latest Elo per team |
| `team_list.pkl` | Sorted list of all known teams |

### `src/scraper.py`
```python
from src.scraper import scrape_prematch_context
ctx = scrape_prematch_context("Arsenal", "Chelsea", use_fbref=True)
# Returns: home_form, away_form, h2h_record, home_goals_avg, ...
```
- Attempts FBref.com with polite rate-limiting (2 s + jitter delay)
- Falls back to historical `Matches.csv` automatically

### `src/rag_context.py`
```python
from src.rag_context import build_rag_context
block = build_rag_context(ctx, "Arsenal", "Chelsea", home_elo=1820, away_elo=1790)
```

### `src/llm_analyst.py`
```python
from src.llm_analyst import analyse
report = analyse(block, {"home": 0.48, "draw": 0.28, "away": 0.24})
# Returns structured dict: predicted_outcome, confidence, key_factors, ...
```

### `src/predict.py`
```python
from src.predict import run_prediction
result = run_prediction("Arsenal", "Chelsea", "2024-05-01")
```

### `src/tracking.py`
```python
from src.tracking import log_prediction
run_id = log_prediction(result)
```
View experiments:
```bash
cd ..   # back to project root
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## LLM Output Schema

```json
{
  "predicted_outcome":  "Home Win | Draw | Away Win",
  "confidence":         "High | Medium | Low",
  "key_factors":        ["factor1", "factor2", "factor3"],
  "risk_flags":         ["risk factor 1"],
  "reasoning_summary":  "Plain English 2-3 sentence explanation.",
  "elo_probability":    {"home": 0.48, "draw": 0.28, "away": 0.24}
}
```

Confidence thresholds:
- **High** → top outcome probability > 55%
- **Medium** → 40–55%
- **Low** → < 40%

---

## Blending Logic

```
blended_prob = (XGB_prob × rf_weight + Elo_prob × elo_weight)
                / (rf_weight + elo_weight)
```

Default weights: **XGB 60% / Elo 40%** — configurable via slider in the dashboard or `rf_weight` parameter in `run_prediction()`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost + Scikit-learn (CalibratedClassifierCV) |
| Elo System | Custom logistic Elo formula with draw adjustment |
| LLM | Anthropic Claude claude-sonnet-4-20250514 |
| Scraping | Requests + BeautifulSoup4 (FBref.com) |
| Dashboard | Streamlit + Plotly |
| Tracking | MLflow (SQLite backend) |
| Data | Pandas + NumPy |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |

---

## Notes

- **Rate limiting**: The FBref scraper introduces a 2–3.5 s delay per request and handles HTTP 429 responses with `Retry-After` back-off.
- **Graceful degradation**: Every module has try/except fallback — the dashboard never crashes, it degrades to model-only output.
- **Data privacy**: The `.env` file containing your API key is excluded from git via `.gitignore`.
