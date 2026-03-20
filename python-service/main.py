from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import predict
import traceback
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


class MatchInput(BaseModel):
    team1: str
    team2: str
    venue: str
    toss_winner: str
    toss_decision: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_match(data: MatchInput):
    try:
        winner, confidence = predict(data.dict())

        return {
            "team1": data.team1,
            "team2": data.team2,
            "predicted_winner": winner,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))