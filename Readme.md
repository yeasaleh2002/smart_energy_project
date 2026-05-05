# AI Project Report: Smart Energy Consumption & Peak Load Analysis

## 1. Introduction
The Smart Energy Consumption & Peak Load Analysis project is an AI-driven solution designed to forecast the risk of peak energy loads across different urban environments and building types. By analyzing various factors such as energy consumption patterns, temperature, occupancy, and renewable energy usage, the project aims to classify the load risk into discrete levels (Low, Medium, High). The solution encompasses an end-to-end machine learning pipeline, from data preprocessing and model training to serving predictions via a RESTful API backend.

## 2. Motivation
As urbanization accelerates and weather patterns become increasingly unpredictable, maintaining the stability of power grids has become a critical challenge. Spikes in energy consumption can lead to grid overloads, blackouts, and increased operational costs. The motivation behind this project is to leverage artificial intelligence to anticipate these peak load risks before they occur, allowing for proactive grid management, demand response strategies, and better integration of renewable energy sources.

## 3. Objective
The primary objectives of this project are:
- To analyze a synthetic or real-world dataset containing energy consumption metrics, environmental data, and building characteristics.
- To perform feature engineering and exploratory data analysis to extract meaningful temporal and categorical features.
- To train a robust machine learning classification model capable of predicting the `peak_load_risk`.
- To deploy the trained AI model using a fast and modern API framework (FastAPI), enabling client applications (e.g., frontend dashboards) to request real-time risk predictions.

## 4. Significance
This project holds significant value for several stakeholders:
- **Grid Operators & Utility Companies:** Provides early warnings for potential grid stress, allowing for dynamic pricing or load shedding.
- **Building Managers:** Helps in optimizing energy usage during high-risk periods to reduce costs and environmental impact.
- **City Planners:** Offers insights into how different city zones and building types (Residential, Commercial, Industrial) contribute to energy demand, aiding in sustainable urban development.

## 5. System Architecture
The system follows a standard machine learning production architecture, split into two main components:
1. **Model Training Pipeline (`train_model.py`):** 
   - Ingests data from the CSV file.
   - Preprocesses data (handling datetime extraction, label encoding, and standard scaling).
   - Trains a `RandomForestClassifier`.
   - Serializes and saves the trained model, scaler, encoders, and feature names as `.pkl` files using `joblib`.
2. **Inference API (`main.py`):**
   - Built with FastAPI.
   - Loads the serialized artifacts (model, scaler, encoders) into memory on startup.
   - Exposes a `/predict` POST endpoint that accepts JSON payloads of new energy data, processes it using the loaded pipeline, and returns the predicted `peak_load_risk`.
   - Includes CORS middleware to allow seamless integration with frontend web applications.

## 6. Exploratory Data Analysis (EDA)
The dataset (`smart_energy_consumption_peak_load_analysis.csv`) contains rich contextual information for each recorded instance. 
- **Features Include:** `building_type` (Industrial, Residential, Commercial), `city_zone` (North, South, East, West, Central), `energy_consumption_kwh`, `temperature_celsius`, `occupancy_level`, `peak_hour` (Yes/No), and `renewable_energy_used_percent`.
- **Temporal Extraction:** The `date_time` column is parsed to extract `hour`, `month`, and `day_of_week`, capturing seasonal and daily consumption rhythms.
- **Categorical Encoding:** Text-based features like building types and city zones are converted into numerical representations using `LabelEncoder`.
- **Target Variable:** `peak_load_risk` serves as the target class, indicating the level of risk to the energy grid.

## 7. Results and Discussion
The core AI model is built using an ensemble learning approach via the `RandomForestClassifier` (with `n_estimators=100`). Random Forest was chosen for its robustness against overfitting and its ability to handle complex, non-linear relationships between variables like temperature, time of day, and energy consumption. 
During the training phase, the data was split into an 80/20 train-test ratio. Features were normalized using a `StandardScaler` to ensure that variables with larger numeric ranges (like energy consumption in kWh) do not disproportionately influence the model compared to smaller scale variables (like temperature). The model successfully generalizes the patterns to predict risk levels on unseen test data, and the pipeline correctly packages these transformations for real-time inference in the FastAPI layer.

## 8. Conclusion
The Smart Energy project successfully demonstrates how predictive analytics can be applied to energy grid management. By unifying data preprocessing, machine learning, and modern backend web frameworks, the project delivers a scalable, production-ready solution. Future iterations could improve the system by incorporating real-time weather API data, transitioning to deep learning models for time-series forecasting, or adding a rich frontend dashboard for visual analytics.

---

## Appendix: Full Project Python Code

### `train_model.py`
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Dataset load...")
df = pd.read_csv('smart_energy_consumption_peak_load_analysis.csv')

# 1. Feature Engineering (Date to Hour, Month)
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['month'] = df['date_time'].dt.month
df['day_of_week'] = df['date_time'].dt.dayofweek
df = df.drop(['record_id', 'date_time'], axis=1)

# 2. Categorical Data Encoding
print("Converting Text data to number...")
encoders = {}
cat_cols = ['building_type', 'city_zone', 'peak_hour', 'peak_load_risk']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop('peak_load_risk', axis=1)
y = df['peak_load_risk']

# 3. Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Model Training
print("AI Model Training...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model Accuracy: {model.score(X_test, y_test)*100:.2f}%")

# 5. Saving everything for the Backend API
joblib.dump(model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("Success! Saved all files (.pkl files).")
```

### `main.py`
```python
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
```
