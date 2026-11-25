import joblib
import pandas as pd


class PortPredictor:
    def __init__(self, model_path="ml/eta_model.joblib"):
        self.model = joblib.load(model_path)

    def predict(self, features: dict) -> dict:
        df = pd.DataFrame([features])
        predicted_eta = float(self.model.predict(df)[0])

        base_hours = features["distance_nm"] / features["speed_knots"]
        wait_hours = max(0.0, predicted_eta - base_hours)

        congestion_score = min(
            100,
            int(
                (features["berth_occupancy"] * 0.6)
                + (features["vessels_at_anchor"] * 1.2)
            ),
        )
        status = (
            "High"
            if congestion_score > 75
            else "Moderate" if congestion_score > 45 else "Low"
        )

        return {
            "predicted_eta_hours": round(predicted_eta, 2),
            "base_transit_hours": round(base_hours, 2),
            "estimated_wait_hours": round(wait_hours, 2),
            "congestion_score": congestion_score,
            "congestion_level": status,
        }