
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("mental_health_predictor.pkl")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class InputData(BaseModel):
    age : int = Field(..., ge = 12, le = 95)
    gender : Literal["Male", "Female"]
    stress_level : Literal["Low", "Medium", "High", "Very High"]
    country : str
    Academic_Level : Literal["High School", "Undergraduate", "Graduate"]
    Most_Used_Platform : Literal["Instagram", "TikTok", "Facebook", "LinkedIn", "YouTube", "Twitter", "Snapchat", "WhatsApp", "LINE", "VKontakte", "KakaoTalk", "WeChat"]
    Purpose_Of_Use : Literal["Entertainment", "Education", "Networking", "News"]
    Avg_Daily_Usage_Hours : float = Field(..., ge = 0.0, le = 24.0)
    Daily_Unlocks : int = Field(..., ge = 0)
    Study_Hours : float = Field(..., ge = 0.0, le = 24.0)
    Physical_Activity_Hours : float = Field(..., ge = 0.0, le = 24.0)
    Sleep_Hours_Per_Night : float = Field(..., ge = 0.0, le = 24.0)

@app.get("/")
def hello():
    return {"Hello world"}

top_countries = ['Other','India','USA','Canada','Australia','UK','Germany','Mexico','Turkey','France']

#response of the server
class PredictionResponse(BaseModel):
    mental_health_score: float


@app.post("/predict", response_model=PredictionResponse)
def predictMentalScore(data: InputData):

    grouped_coutries = data.country if data.country in top_countries else 'Other'

    input_dataframe = pd.DataFrame([{
        "Age": data.age,
        "Gender": data.gender,
        "Stress_Level": data.stress_level,
        "Country": data.country,
        "Academic_Level": data.Academic_Level,
        "Most_Used_Platform": data.Most_Used_Platform,
        "Purpose_Of_Use": data.Purpose_Of_Use,
        "Avg_Daily_Usage_Hours": data.Avg_Daily_Usage_Hours,
        "Daily_Unlocks": data.Daily_Unlocks,
        "Study_Hours": data.Study_Hours,
        "Physical_Activity_Hours": data.Physical_Activity_Hours,
        "Sleep_Hours_Per_Night": data.Sleep_Hours_Per_Night,
        "Grouped_Countries": grouped_coutries
    }])

    prediction = model.predict(input_dataframe)[0]
    return PredictionResponse(mental_health_score=round(float(prediction), 2))