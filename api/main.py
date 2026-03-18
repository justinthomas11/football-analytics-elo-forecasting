from fastapi import FastAPI
from pydantic import BaseModel

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
