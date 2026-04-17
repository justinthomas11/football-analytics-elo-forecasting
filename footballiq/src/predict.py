"""
src/predict.py — FootballIQ Prediction Pipeline
================================================
Orchestrates: model load → RF probs → Elo probs → blend → scrape → RAG → LLM → output.

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
BASE_DIR   = Path(__file__).resolve().parent.parent   # footballiq/
MODELS_DIR = BASE_DIR / "models"

# ── Feature list (must match train.py) ───────────────────────────────────────
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

# Blending weights (configurable)
RF_WEIGHT  = 0.60
ELO_WEIGHT = 0.40

# Elo parameters
ELO_K      = 32
ELO_BASE   = 1500.0


# ── Loaders ───────────────────────────────────────────────────────────────────
_model_cache     = None
_elo_state_cache = None
_team_list_cache = None


def _load_artefacts():
    global _model_cache, _elo_state_cache, _team_list_cache
    if _model_cache is not None:
        return _model_cache, _elo_state_cache, _team_list_cache

    try:
        with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
            _model_cache = pickle.load(f)
        with open(MODELS_DIR / "elo_state.pkl", "rb") as f:
            _elo_state_cache = pickle.load(f)
        with open(MODELS_DIR / "team_list.pkl", "rb") as f:
            _team_list_cache = pickle.load(f)
        logger.info("Model artefacts loaded successfully.")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model artefacts not found at {MODELS_DIR}. "
            "Run train.py first to generate them."
        ) from exc

    return _model_cache, _elo_state_cache, _team_list_cache


# ── Elo probability calculation ───────────────────────────────────────────────

def _elo_win_probability(home_elo: float, away_elo: float) -> dict:
    """
    Convert Elo ratings to match outcome probabilities using the
    standard Elo formula with a draw adjustment factor.
    """
    diff     = home_elo - away_elo
    # Expected score for home team
    expected = 1 / (1 + 10 ** (-diff / 400))

    # Estimate draw probability via a logistic sigmoid on |diff|
    abs_diff  = abs(diff)
    draw_base = 0.28 * math.exp(-abs_diff / 600)  # draws more likely when close
    home_win  = expected * (1 - draw_base)
    away_win  = (1 - expected) * (1 - draw_base)
    draw      = 1 - home_win - away_win

    # Normalise to sum to 1
    total = home_win + draw + away_win
    return {
        "home": round(home_win / total, 4),
        "draw": round(draw      / total, 4),
        "away": round(away_win  / total, 4),
    }


# ── Feature builder for model inference ──────────────────────────────────────

def _build_feature_row(
    home_elo: float,
    away_elo: float,
    home_form3: float = 1.0,
    away_form3: float = 1.0,
    home_form5: float = 1.0,
    away_form5: float = 1.0,
    home_shots: float = 12.0,
    away_shots: float = 10.0,
    home_target: float = 5.0,
    away_target: float = 4.0,
    home_corners: float = 5.5,
    away_corners: float = 4.5,
    home_cards: float = 1.5,
    away_cards: float = 1.5,
    odd_home: float = 2.2,
    odd_draw: float = 3.4,
    odd_away: float = 3.2,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame for model.predict_proba()."""
    elo_diff     = home_elo - away_elo
    form3_diff   = home_form3 - away_form3
    form5_diff   = home_form5 - away_form5
    shot_diff    = home_shots - away_shots
    target_diff  = home_target - away_target
    corner_diff  = home_corners - away_corners
    card_diff    = home_cards - away_cards
    low_scoring  = (odd_home + odd_away) / max(odd_draw, 0.01)

    row = {
        "EloDiff":       elo_diff,
        "AbsEloDiff":    abs(elo_diff),
        "Form3Diff":     form3_diff,
        "AbsForm5Diff":  abs(form5_diff),
        "ShotDiff":      shot_diff,
        "AbsShotDiff":   abs(shot_diff),
        "TargetDiff":    target_diff,
        "CornerDiff":    corner_diff,
        "CardDiff":      card_diff,
        "LowScoringBias":low_scoring,
        "OddHome":       odd_home,
        "OddDraw":       odd_draw,
        "OddAway":       odd_away,
    }
    return pd.DataFrame([row])[FEATURES]


def _form_to_points(form_list: list) -> float:
    """Convert ['W','D','L',...] to a normalised 0-3 points-per-game proxy."""
    if not form_list:
        return 1.5  # neutral default
    pts = sum(3 if r == "W" else (1 if r == "D" else 0) for r in form_list)
    return pts / len(form_list)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_prediction(
    home_team: str,
    away_team: str,
    match_date: Optional[str] = None,
    rf_weight: float = RF_WEIGHT,
    elo_weight: float = ELO_WEIGHT,
    use_llm: bool = True,
    use_fbref: bool = True,
) -> dict:
    """
    Full FootballIQ prediction pipeline.

    Parameters
    ----------
    home_team   : Home team display name.
    away_team   : Away team display name.
    match_date  : ISO date string (YYYY-MM-DD). Defaults to today.
    rf_weight   : Weight for XGB model probabilities (default 0.60).
    elo_weight  : Weight for Elo probabilities (default 0.40).
    use_llm     : Whether to call Claude for narrative analysis.
    use_fbref   : Whether to attempt FBref live scraping.

    Returns
    -------
    dict — full prediction report.
    """
    from src.scraper     import scrape_prematch_context
    from src.rag_context import build_rag_context
    from src.llm_analyst import analyse

    match_date = match_date or str(date.today())
    model, elo_state, team_list = _load_artefacts()

    # ── Step 1: Elo ratings ───────────────────────────────────────────────
    home_elo = elo_state.get(home_team, ELO_BASE)
    away_elo = elo_state.get(away_team, ELO_BASE)

    # Fuzzy match if exact key not found
    if home_elo == ELO_BASE and home_team not in elo_state:
        match = _fuzzy_match(home_team, list(elo_state.keys()))
        if match:
            home_elo = elo_state[match]
            logger.info(f"Fuzzy-matched '{home_team}' → '{match}' (Elo={home_elo:.0f})")

    if away_elo == ELO_BASE and away_team not in elo_state:
        match = _fuzzy_match(away_team, list(elo_state.keys()))
        if match:
            away_elo = elo_state[match]
            logger.info(f"Fuzzy-matched '{away_team}' → '{match}' (Elo={away_elo:.0f})")

    # ── Step 2: Scrape context ────────────────────────────────────────────
    logger.info(f"Scraping context for {home_team} vs {away_team} …")
    context = scrape_prematch_context(home_team, away_team, use_fbref=use_fbref)

    # ── Step 3: Form proxies for features ─────────────────────────────────
    home_form_pts = _form_to_points(context.get("home_form", []))
    away_form_pts = _form_to_points(context.get("away_form", []))

    # ── Step 4: Build feature row + RF prediction ─────────────────────────
    feat_row = _build_feature_row(
        home_elo=home_elo,
        away_elo=away_elo,
        home_form3=home_form_pts,
        away_form3=away_form_pts,
        home_form5=home_form_pts,
        away_form5=away_form_pts,
    )
    rf_probs_raw = model.predict_proba(feat_row)[0]  # [home, draw, away]
    rf_probs = {
        "home": float(rf_probs_raw[0]),
        "draw": float(rf_probs_raw[1]),
        "away": float(rf_probs_raw[2]),
    }

    # ── Step 5: Elo probabilities ─────────────────────────────────────────
    elo_probs = _elo_win_probability(home_elo, away_elo)

    # ── Step 6: Blend ─────────────────────────────────────────────────────
    w_sum = rf_weight + elo_weight
    blended = {
        "home": round((rf_probs["home"] * rf_weight + elo_probs["home"] * elo_weight) / w_sum, 4),
        "draw": round((rf_probs["draw"] * rf_weight + elo_probs["draw"] * elo_weight) / w_sum, 4),
        "away": round((rf_probs["away"] * rf_weight + elo_probs["away"] * elo_weight) / w_sum, 4),
    }

    # ── Step 7: RAG context block ─────────────────────────────────────────
    rag_block = build_rag_context(
        context, home_team, away_team, home_elo, away_elo
    )

    # ── Step 8: LLM analysis ─────────────────────────────────────────────
    llm_report = {}
    if use_llm:
        logger.info("Sending to Claude for analysis …")
        llm_report = analyse(rag_block, blended)

    # ── Assemble final output ─────────────────────────────────────────────
    return {
        "fixture": {
            "home_team":  home_team,
            "away_team":  away_team,
            "match_date": match_date,
        },
        "elo_ratings": {
            "home": round(home_elo, 1),
            "away": round(away_elo, 1),
        },
        "rf_probabilities":    rf_probs,
        "elo_probabilities":   elo_probs,
        "blended_probabilities": blended,
        "blend_weights": {
            "rf":  rf_weight,
            "elo": elo_weight,
        },
        "scraped_context": context,
        "rag_context":     rag_block,
        "llm_report":      llm_report,
    }


def _fuzzy_match(query: str, candidates: list, threshold: float = 0.6) -> Optional[str]:
    """
    Simple character-overlap fuzzy match — avoids pulling in fuzzywuzzy.
    Returns the best match above threshold, or None.
    """
    query_lower = query.lower()
    best, best_score = None, 0.0
    for c in candidates:
        c_lower = c.lower()
        # Jaccard similarity on bigrams
        q_bi = set(query_lower[i:i+2] for i in range(len(query_lower)-1))
        c_bi = set(c_lower[i:i+2] for i in range(len(c_lower)-1))
        if not q_bi or not c_bi:
            continue
        score = len(q_bi & c_bi) / len(q_bi | c_bi)
        if score > best_score:
            best_score, best = score, c
    return best if best_score >= threshold else None
