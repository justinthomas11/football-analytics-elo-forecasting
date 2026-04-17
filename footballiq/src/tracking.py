"""
src/tracking.py — FootballIQ MLflow Experiment Tracker
========================================================
Logs each prediction run as a versioned MLflow experiment.

Tracked parameters / metrics:
  - home_team, away_team, match_date
  - blended home/draw/away probabilities
  - predicted_outcome, confidence
  - model_version (git hash or "local")
  - data_source (fbref / historical / simulated)
"""

import hashlib
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# MLflow db lives next to the footballiq folder (reuse existing mlruns)
BASE_DIR     = Path(__file__).resolve().parent.parent        # footballiq/
MLFLOW_DIR   = BASE_DIR.parent / "mlruns"                    # ../mlruns
EXPERIMENT_NAME = "FootballIQ-Predictions"


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BASE_DIR),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha
    except Exception:
        return "local"


def log_prediction(result: dict) -> str:
    """
    Log a prediction run to MLflow.

    Parameters
    ----------
    result : dict returned by predict.run_prediction()

    Returns
    -------
    str — MLflow run_id (or empty string on failure)
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR.parent / 'mlflow.db'}")
        mlflow.set_experiment(EXPERIMENT_NAME)

        fixture  = result.get("fixture", {})
        blended  = result.get("blended_probabilities", {})
        llm      = result.get("llm_report", {})
        elo      = result.get("elo_ratings", {})
        context  = result.get("scraped_context", {})

        predicted_outcome = llm.get("predicted_outcome", "Unknown")
        confidence        = llm.get("confidence", "Unknown")

        with mlflow.start_run(run_name=f"{fixture.get('home_team','?')} vs {fixture.get('away_team','?')}"):
            # ── Parameters ────────────────────────────────────────────────
            mlflow.log_param("home_team",        fixture.get("home_team", "Unknown"))
            mlflow.log_param("away_team",        fixture.get("away_team", "Unknown"))
            mlflow.log_param("match_date",       fixture.get("match_date", "Unknown"))
            mlflow.log_param("predicted_outcome",predicted_outcome)
            mlflow.log_param("confidence",       confidence)
            mlflow.log_param("model_version",    _git_sha())
            mlflow.log_param("data_source",      context.get("source", "unknown"))
            mlflow.log_param("rf_weight",        result.get("blend_weights", {}).get("rf",  0.6))
            mlflow.log_param("elo_weight",       result.get("blend_weights", {}).get("elo", 0.4))

            # ── Metrics ───────────────────────────────────────────────────
            mlflow.log_metric("prob_home",       blended.get("home", 0.0))
            mlflow.log_metric("prob_draw",       blended.get("draw", 0.0))
            mlflow.log_metric("prob_away",       blended.get("away", 0.0))
            mlflow.log_metric("home_elo",        elo.get("home", 0.0))
            mlflow.log_metric("away_elo",        elo.get("away", 0.0))
            mlflow.log_metric("elo_diff",        elo.get("home", 0.0) - elo.get("away", 0.0))

            # ── Tags ──────────────────────────────────────────────────────
            mlflow.set_tag("pipeline",  "FootballIQ")
            mlflow.set_tag("llm_model", "claude-sonnet-4-20250514")

            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow run logged: {run_id}")
            return run_id

    except Exception as exc:
        logger.warning(f"MLflow logging failed (non-fatal): {exc}")
        return ""
