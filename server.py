# server.py
import csv
import io
from typing import List
import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ml.predictor import PortPredictor

app = FastAPI(title="Port Congestion & ETA Predictor")
predictor = PortPredictor()


class VesselInput(BaseModel):
    distance_nm: float
    speed_knots: float
    vessels_at_anchor: int
    berth_occupancy: float
    wind_speed_knots: float


@app.post("/api/predict")
def get_prediction(data: VesselInput):
    return predictor.predict(data.model_dump())


@app.post("/api/predict-batch")
def get_batch_predictions(vessels: List[VesselInput]):
    return [predictor.predict(v.model_dump()) for v in vessels]
@app.post("/api/predict-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    content = await file.read()
    try:
        text = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}")

    required_cols = {
        "distance_nm",
        "speed_knots",
        "vessels_at_anchor",
        "berth_occupancy",
        "wind_speed_knots",
    }
    if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
        raise HTTPException(
            status_code=400,
            detail=f"CSV must include columns: {', '.join(sorted(required_cols))}",
        )

    results = []
    for idx, row in enumerate(reader, start=1):
        try:
            payload = {
                "distance_nm": float(row["distance_nm"]),
                "speed_knots": float(row["speed_knots"]),
                "vessels_at_anchor": int(float(row["vessels_at_anchor"])),
                "berth_occupancy": float(row["berth_occupancy"]),
                "wind_speed_knots": float(row["wind_speed_knots"]),
            }
            pred = predictor.predict(payload)
            label = (row.get("vessel_name") or row.get("port_name") or f"CSV Vessel #{idx}").strip()
            results.append({"label": label, "input": payload, "prediction": pred})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Row {idx} invalid data: {exc}")

    return results

@app.get("/api/live-weather")
async def get_live_marine_weather(lat: float, lon: float):
    weather_url = "https://api.open-meteo.com/v1/forecast"
    marine_url = "https://marine-api.open-meteo.com/v1/marine"

    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "current": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "kn",
    }
    marine_params = {
        "latitude": lat,
        "longitude": lon,
        "current": "wave_height,swell_wave_height",
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            w_res, m_res = await client.get(weather_url, params=weather_params), await client.get(marine_url, params=marine_params)
            w_res.raise_for_status()
            m_res.raise_for_status()

        w_current = w_res.json().get("current", {})
        m_current = m_res.json().get("current", {})

        return {
            "wind_speed_knots": round(float(w_current.get("wind_speed_10m", 12.0)), 1),
            "wind_direction_deg": w_current.get("wind_direction_10m"),
            "wave_height_m": m_current.get("wave_height"),
            "swell_wave_height_m": m_current.get("swell_wave_height"),
            "observed_at": w_current.get("time"),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Live weather lookup failed: {exc}")


@app.get("/")
def serve_dashboard():
    return FileResponse("frontend/index.html")