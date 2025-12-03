from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import predict
import traceback

app = FastAPI()

class MatchInput(BaseModel):
    team1: str
    team2: str
    venue: str
    toss_winner: str
    toss_decision: str

@app.post("/predict")
def predict_match(data: MatchInput):
    try:
        result = predict(data.dict())
        return {"winner": result}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
