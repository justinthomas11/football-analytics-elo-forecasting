import pickle
import pandas as pd
from contextlib import asynccontextmanager


from fastapi import FastAPI, HTTPException
#We removed Pydantic import as the Pydantic models now live in models.py
from api.models import MatchInput, PredictionOutput

app = FastAPI()

class MatchInput(BaseModel):
    EloDiff: float
    AbsEloDiff: float
    Form3Diff: float
    AbsForm5Diff: float
    ShotDiff: float
    AbsShotDiff: float
    TargetDiff: float
    CornerDiff: float
    CardDiff: float
    LowScoringBias: float
    OddHome: float
    OddDraw: float
    OddAway: float

@app.get("/")
def root():
    return {"message": "Football Predictor API is running"}

@app.get("/odds")
def get_odds(home:float, away:float):
  diff= home-away
  return {"Home":home, "Away":away,"Difference":diff}

@app.post("/predict")
def predict_match(data:MatchInput):
  return {
    "received":data.model_dump()
    
  }
