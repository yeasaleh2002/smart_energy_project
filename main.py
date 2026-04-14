from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

app = FastAPI(title="Smart Energy API")

# CORS Setup (Frontend connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load previously saved models
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl')
feature_names = joblib.load('feature_names.pkl')

# Input schema define
class EnergyData(BaseModel):
    building_type: str
    city_zone: str
    energy_consumption_kwh: float
    temperature_celsius: float
    occupancy_level: int
    peak_hour: str
    renewable_energy_used_percent: float
    hour: int
    month: int
    day_of_week: int

@app.post("/predict")
def predict_risk(data: EnergyData):
    # Dataframe from Dict
    df = pd.DataFrame([data.dict()])
    
    # Text data encode 
    for col in ['building_type', 'city_zone', 'peak_hour']:
        df[col] = encoders[col].transform(df[col])
    
    # Ensure column order matches training data
    df = df[feature_names]
    
    # Scale data
    scaled_features = scaler.transform(df)
    
    # Predict
    prediction_encoded = model.predict(scaled_features)[0]
    
    # converted Prediction (High, Medium, Low)
    risk_label = encoders['peak_load_risk'].inverse_transform([prediction_encoded])[0]
    
    return {"peak_load_risk": risk_label}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)