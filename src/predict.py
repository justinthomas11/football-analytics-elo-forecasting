"""
src/predict.py — FootballIQ Prediction Pipeline
================================================
Orchestrates: model load → XGB probs → Elo probs → blend → scrape → RAG → LLM → output.

Usage:
    from src.predict import run_prediction
    result = run_prediction("Arsenal", "Chelsea", "2024-04-20")
"""

import math
import pickle
import logging
import warnings
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent   # project root
MODELS_DIR = BASE_DIR / "models"

FEATURES = [
    "EloDiff", "AbsEloDiff",
    "Form3Diff", "AbsForm5Diff",
    "ShotDiff", "AbsShotDiff",
    "TargetDiff",
    "CornerDiff", "CardDiff",
    "LowScoringBias",
    "OddHome", "OddDraw", "OddAway",
]

LABEL_DECODE = {0: "Home Win", 1: "Draw", 2: "Away Win"}
RF_WEIGHT    = 0.60
ELO_WEIGHT   = 0.40
ELO_BASE     = 1500.0

# ── Model cache ───────────────────────────────────────────────────────────────
_model_cache = _elo_cache = _teams_cache = None


def _load_artefacts():
    global _model_cache, _elo_cache, _teams_cache
    if _model_cache is not None:
        return _model_cache, _elo_cache, _teams_cache
    try:
        with open(MODELS_DIR / "xgb_model.pkl",  "rb") as f: _model_cache = pickle.load(f)
        with open(MODELS_DIR / "elo_state.pkl",  "rb") as f: _elo_cache   = pickle.load(f)
        with open(MODELS_DIR / "team_list.pkl",  "rb") as f: _teams_cache = pickle.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model artefacts missing at {MODELS_DIR}/. Run train.py first."
        ) from exc
    return _model_cache, _elo_cache, _teams_cache


# ── Elo probability ───────────────────────────────────────────────────────────

def _elo_probs(home_elo: float, away_elo: float) -> dict:
    diff     = home_elo - away_elo
    expected = 1 / (1 + 10 ** (-diff / 400))
    draw     = 0.28 * math.exp(-abs(diff) / 600)
    hw       = expected * (1 - draw)
    aw       = (1 - expected) * (1 - draw)
    dr       = 1 - hw - aw
    total    = hw + dr + aw
    return {
        "home": round(hw / total, 4),
        "draw": round(dr / total, 4),
        "away": round(aw / total, 4),
    }


# ── Feature builder ───────────────────────────────────────────────────────────

def _build_row(
    home_elo, away_elo,
    home_form=1.5, away_form=1.5,
    home_shots=12.0, away_shots=10.0,
    home_target=5.0, away_target=4.0,
    home_corners=5.5, away_corners=4.5,
    home_cards=1.5, away_cards=1.5,
    odd_home=2.2, odd_draw=3.4, odd_away=3.2,
) -> pd.DataFrame:
    ed  = home_elo - away_elo
    f3d = home_form - away_form
    sd  = home_shots - away_shots
    row = {
        "EloDiff":       ed,       "AbsEloDiff":   abs(ed),
        "Form3Diff":     f3d,      "AbsForm5Diff": abs(f3d),
        "ShotDiff":      sd,       "AbsShotDiff":  abs(sd),
        "TargetDiff":    home_target - away_target,
        "CornerDiff":    home_corners - away_corners,
        "CardDiff":      home_cards - away_cards,
        "LowScoringBias": (odd_home + odd_away) / max(odd_draw, 0.01),
        "OddHome": odd_home, "OddDraw": odd_draw, "OddAway": odd_away,
    }
    return pd.DataFrame([row])[FEATURES]


def _form_pts(form_list: list) -> float:
    if not form_list:
        return 1.5
    return sum(3 if r == "W" else (1 if r == "D" else 0) for r in form_list) / len(form_list)


# ── Fuzzy team name match ─────────────────────────────────────────────────────

def _fuzzy_match(query: str, candidates: list, threshold: float = 0.55) -> Optional[str]:
    q = query.lower()
    best, best_score = None, 0.0
    for c in candidates:
        cl  = c.lower()
        qbi = set(q[i:i+2] for i in range(len(q)-1))
        cbi = set(cl[i:i+2] for i in range(len(cl)-1))
        if not qbi or not cbi:
            continue
        score = len(qbi & cbi) / len(qbi | cbi)
        if score > best_score:
            best_score, best = score, c
    return best if best_score >= threshold else None


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_prediction(
    home_team: str,
    away_team: str,
    match_date: Optional[str] = None,
    rf_weight: float = RF_WEIGHT,
    elo_weight: float = ELO_WEIGHT,
    use_llm: bool = True,
    use_fbref: bool = False,
) -> dict:
    """
    Full FootballIQ prediction pipeline.

    Returns a unified dict with: fixture, elo_ratings, rf_probabilities,
    elo_probabilities, blended_probabilities, scraped_context,
    rag_context, llm_report.
    """
    from src.scraper     import scrape_prematch_context
    from src.rag_context import build_rag_context
    from src.llm_analyst import analyse

    match_date             = match_date or str(date.today())
    model, elo_state, _   = _load_artefacts()

    # Step 1 — Elo lookup with fuzzy fallback
    def _lookup_elo(team):
        if team in elo_state:
            return elo_state[team]
        m = _fuzzy_match(team, list(elo_state.keys()))
        if m:
            logger.info(f"Fuzzy match: '{team}' → '{m}' (Elo={elo_state[m]:.0f})")
            return elo_state[m]
        return ELO_BASE

    home_elo = _lookup_elo(home_team)
    away_elo = _lookup_elo(away_team)

    # Step 2 — Scrape context
    logger.info(f"Scraping context for {home_team} vs {away_team} …")
    context = scrape_prematch_context(home_team, away_team, use_fbref=use_fbref)

    # Step 3 — Build feature row
    feat_row = _build_row(
        home_elo=home_elo, away_elo=away_elo,
        home_form=_form_pts(context.get("home_form", [])),
        away_form=_form_pts(context.get("away_form", [])),
    )

    # Step 4 — XGB probabilities
    rf_raw  = model.predict_proba(feat_row)[0]
    rf_probs = {"home": float(rf_raw[0]), "draw": float(rf_raw[1]), "away": float(rf_raw[2])}

    # Step 5 — Elo probabilities
    elo_probs = _elo_probs(home_elo, away_elo)

    # Step 6 — Blend
    w = rf_weight + elo_weight
    blended = {
        k: round((rf_probs[k] * rf_weight + elo_probs[k] * elo_weight) / w, 4)
        for k in ("home", "draw", "away")
    }

    # Step 7 — RAG context
    rag_block = build_rag_context(context, home_team, away_team, home_elo, away_elo)

    # Step 8 — LLM
    llm_report = {}
    if use_llm:
        logger.info("Calling Claude for analysis …")
        llm_report = analyse(rag_block, blended)

    return {
        "fixture":               {"home_team": home_team, "away_team": away_team, "match_date": match_date},
        "elo_ratings":           {"home": round(home_elo, 1), "away": round(away_elo, 1)},
        "rf_probabilities":      rf_probs,
        "elo_probabilities":     elo_probs,
        "blended_probabilities": blended,
        "blend_weights":         {"rf": rf_weight, "elo": elo_weight},
        "scraped_context":       context,
        "rag_context":           rag_block,
        "llm_report":            llm_report,
    }
