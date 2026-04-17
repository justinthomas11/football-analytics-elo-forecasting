"""
train.py — FootballIQ Training Script
======================================
Trains a calibrated XGBClassifier on historical match data and saves:
  - models/xgb_model.pkl       : CalibratedClassifierCV
  - models/elo_state.pkl       : dict mapping team → latest Elo rating
  - models/team_list.pkl       : sorted list of known teams
  - models/feature_names.pkl   : list of feature column names
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MATCHES_CSV = DATA_DIR / "Matches.csv"
ELO_CSV     = DATA_DIR / "EloRatings.csv"

# Fallback: use parent project's Data/ folder if footballiq/data/ is empty
if not MATCHES_CSV.exists():
    MATCHES_CSV = BASE_DIR.parent / "Data" / "Matches.csv"
    ELO_CSV     = BASE_DIR.parent / "Data" / "EloRatings.csv"

FEATURES = [
    "EloDiff", "AbsEloDiff",
    "Form3Diff", "AbsForm5Diff",
    "ShotDiff", "AbsShotDiff",
    "TargetDiff",
    "CornerDiff", "CardDiff",
    "LowScoringBias",
    "OddHome", "OddDraw", "OddAway",
]

LABEL_MAP    = {"H": 0, "D": 1, "A": 2}
LABEL_DECODE = {0: "Home Win", 1: "Draw", 2: "Away Win"}


def load_and_prepare(matches_path: Path) -> pd.DataFrame:
    print(f"📂 Loading matches from {matches_path} …")
    df = pd.read_csv(matches_path, parse_dates=["MatchDate"], low_memory=False)

    # Drop rows without a result
    df = df.dropna(subset=["FTResult"])

    # Fill numeric NaNs with column medians
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # ── Feature engineering ───────────────────────────────────────────────
    df["EloDiff"]        = df["HomeElo"]    - df["AwayElo"]
    df["Form3Diff"]      = df["Form3Home"]  - df["Form3Away"]
    df["Form5Diff"]      = df["Form5Home"]  - df["Form5Away"]
    df["ShotDiff"]       = df["HomeShots"]  - df["AwayShots"]
    df["TargetDiff"]     = df["HomeTarget"] - df["AwayTarget"]
    df["CornerDiff"]     = df["HomeCorners"]- df["AwayCorners"]
    df["CardDiff"]       = (df["HomeYellow"] + df["HomeRed"]) - \
                           (df["AwayYellow"] + df["AwayRed"])

    df["AbsEloDiff"]     = df["EloDiff"].abs()
    df["AbsForm5Diff"]   = df["Form5Diff"].abs()
    df["AbsShotDiff"]    = df["ShotDiff"].abs()

    df["LowScoringBias"] = (df["OddHome"] + df["OddAway"]) / df["OddDraw"]

    df["ResultLabel"]    = df["FTResult"].map(LABEL_MAP)
    df = df.dropna(subset=["ResultLabel"] + FEATURES)
    df["ResultLabel"]    = df["ResultLabel"].astype(int)

    return df


def build_elo_state(df: pd.DataFrame) -> dict:
    """Return most-recent Elo per team from the Matches dataset."""
    df_sorted = df.sort_values("MatchDate")
    elo_state = {}
    for _, row in df_sorted.iterrows():
        if not pd.isna(row["HomeElo"]):
            elo_state[row["HomeTeam"]] = float(row["HomeElo"])
        if not pd.isna(row["AwayElo"]):
            elo_state[row["AwayTeam"]] = float(row["AwayElo"])
    return elo_state


def train(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["ResultLabel"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class weights
    classes = np.unique(y_train)
    weights  = compute_class_weight("balanced", classes=classes, y=y_train)
    cw       = dict(zip(classes, weights))
    sample_w = y_train.map(cw)

    print("⚙️  Training XGBClassifier …")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_w)

    print("🔧 Calibrating probabilities …")
    calibrated = CalibratedClassifierCV(estimator=xgb, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)

    pred  = calibrated.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, pred, target_names=["Home Win", "Draw", "Away Win"]))

    return calibrated


def main():
    df        = load_and_prepare(MATCHES_CSV)
    elo_state = build_elo_state(df)
    team_list = sorted(elo_state.keys())

    model = train(df)

    # ── Save artefacts ────────────────────────────────────────────────────
    with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "elo_state.pkl", "wb") as f:
        pickle.dump(elo_state, f)
    with open(MODELS_DIR / "team_list.pkl", "wb") as f:
        pickle.dump(team_list, f)
    with open(MODELS_DIR / "feature_names.pkl", "wb") as f:
        pickle.dump(FEATURES, f)

    print(f"\n✅ Saved model artefacts to {MODELS_DIR}/")
    print(f"   Teams tracked: {len(team_list)}")
    print(f"   Elo state entries: {len(elo_state)}")


if __name__ == "__main__":
    main()
