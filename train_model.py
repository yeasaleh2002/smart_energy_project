import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Dataset load kora hocche...")
df = pd.read_csv('smart_energy_consumption_peak_load_analysis.csv')

# 1. Feature Engineering (Date theke Hour, Month ber kora)
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['month'] = df['date_time'].dt.month
df['day_of_week'] = df['date_time'].dt.dayofweek
df = df.drop(['record_id', 'date_time'], axis=1)

# 2. Categorical Data Encoding
print("Text data ke number e convert kora hocche...")
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
print("AI Model Train kora hocche...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model Accuracy: {model.score(X_test, y_test)*100:.2f}%")

# 5. Saving everything for the Backend API
joblib.dump(model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("Success! Shob file save kora hoyeche (.pkl files).")