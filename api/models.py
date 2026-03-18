from pydantic import BaseModel, Field


# ── INPUT MODEL (what the client sends us) ──────────────

class MatchInput(BaseModel):

    # Elo rating gap
    EloDiff: float = Field(..., example=120.5)
    AbsEloDiff: float = Field(..., example=120.5)

    # Recent form gap
    Form3Diff: float = Field(..., example=0.5)
    AbsForm5Diff: float = Field(..., example=0.3)

    # Shot-based stats
    ShotDiff: float = Field(..., example=3.0)
    AbsShotDiff: float = Field(..., example=3.0)
    TargetDiff: float = Field(..., example=1.5)

    # Physical play
    CornerDiff: float = Field(..., example=2.0)
    CardDiff: float = Field(..., example=-1.0)

    # Betting market
    LowScoringBias: float = Field(..., example=1.8)
    OddHome: float = Field(..., example=2.1)
    OddDraw: float = Field(..., example=3.4)
    OddAway: float = Field(..., example=3.6)


# ── OUTPUT MODEL (what we send back) ────────────────────

class PredictionOutput(BaseModel):

    prediction: str        # "Home", "Draw", or "Away"
    probabilities: dict    # e.g. {"Home": 0.55, "Draw": 0.20, "Away": 0.25}
    confidence: float      # the highest probability value

