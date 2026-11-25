# ml/train_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

os.makedirs("data", exist_ok=True)
os.makedirs("ml", exist_ok=True)

np.random.seed(42)
n_samples = 2000

data = pd.DataFrame({
    "distance_nm": np.random.uniform(10, 500, n_samples),
    "speed_knots": np.random.uniform(10, 22, n_samples),
    "vessels_at_anchor": np.random.randint(0, 45, n_samples),
    "berth_occupancy": np.random.uniform(30, 100, n_samples),
    "wind_speed_knots": np.random.uniform(2, 35, n_samples),
})

base_transit = data["distance_nm"] / data["speed_knots"]
congestion_delay = (data["vessels_at_anchor"] * 1.4) * (data["berth_occupancy"] / 60.0)
weather_delay = np.where(
    data["wind_speed_knots"] > 20,
    (data["wind_speed_knots"] - 20) * 0.5,
    0,
)

data["total_eta_hours"] = (
    base_transit + congestion_delay + weather_delay + np.random.normal(0, 1.5, n_samples)
)

data.to_csv("data/port_traffic_data.csv", index=False)

X = data.drop(columns=["total_eta_hours"])
y = data["total_eta_hours"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)
model.fit(X_train, y_train)

print(f"Model R² Score: {model.score(X_test, y_test):.3f}")
print("Saved dataset to data/port_traffic_data.csv")
joblib.dump(model, "ml/eta_model.joblib")