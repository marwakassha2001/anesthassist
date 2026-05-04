"""
General Anesthesia Decision Support API
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from typing import Optional

app = FastAPI(
    title="General Anesthesia Decision Support",
    description="Predicts whether a patient needs general anesthesia for dental treatment.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

BASE   = os.path.dirname(__file__)
model  = joblib.load(os.path.join(BASE, "model.pkl"))
scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))

SCALED_COLS   = ["icdas_max", "nombre_de_dents_severes", "asa"]
FEATURE_ORDER = [
    "age", "asa", "icdas_max", "nombre_de_dents_severes",
    "calm", "agitated", "short_attention", "anxious",
    "extremely_agitated", "uncooperative",
    "special_needs", "previous_general_anesthesia",
    "presence_of_infection", "previous_failed_dental_attempts",
]

MEMORY_PATH = os.path.join(BASE, "patients.json")

def load_memory() -> dict:
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r") as f:
            return json.load(f)
    return {}

def save_memory(data: dict):
    with open(MEMORY_PATH, "w") as f:
        json.dump(data, f, indent=2)

class PatientInput(BaseModel):
    patient_id: str                      = Field(..., description="Unique patient identifier")
    patient_name: Optional[str]          = Field(None)
    age: float                           = Field(..., ge=0, le=120)
    asa: int                             = Field(..., ge=1, le=4)
    icdas_max: int                       = Field(..., ge=0, le=6)
    nombre_de_dents_severes: int         = Field(..., ge=0)
    calm: int                            = Field(..., ge=0, le=1)
    agitated: int                        = Field(..., ge=0, le=1)
    short_attention: int                 = Field(..., ge=0, le=1)
    anxious: int                         = Field(..., ge=0, le=1)
    extremely_agitated: int              = Field(..., ge=0, le=1)
    uncooperative: int                   = Field(..., ge=0, le=1)
    special_needs: int                   = Field(..., ge=0, le=1)
    previous_general_anesthesia: int     = Field(..., ge=0, le=1)
    presence_of_infection: int           = Field(..., ge=0, le=1)
    previous_failed_dental_attempts: int = Field(..., ge=0, le=1)

class PredictionResult(BaseModel):
    patient_id: str
    patient_name: Optional[str]
    decision: int
    label: str
    probability_ga: float
    probability_no_ga: float
    timestamp: str
    visit_count: int

class PatientSummary(BaseModel):
    patient_id: str
    patient_name: Optional[str]
    visit_count: int
    last_visit: str
    last_decision: int
    last_label: str

@app.post("/predict", response_model=PredictionResult)
def predict(patient: PatientInput):
    data = pd.DataFrame([patient.dict()])[FEATURE_ORDER]
    data[SCALED_COLS] = scaler.transform(data[SCALED_COLS])

    decision = int(model.predict(data)[0])
    proba    = model.predict_proba(data)[0]
    label    = "General Anesthesia Required" if decision == 1 else "General Anesthesia NOT Required"
    now      = datetime.now().isoformat(timespec="seconds")

    memory = load_memory()
    pid = patient.patient_id.strip().upper()

    if pid not in memory:
        memory[pid] = {"patient_name": patient.patient_name, "visits": []}
    if patient.patient_name:
        memory[pid]["patient_name"] = patient.patient_name

    memory[pid]["visits"].append({
        "timestamp": now,
        "inputs": {k: getattr(patient, k) for k in FEATURE_ORDER},
        "decision": decision,
        "label": label,
        "probability_ga": round(float(proba[1]), 4),
        "probability_no_ga": round(float(proba[0]), 4),
    })
    save_memory(memory)

    return PredictionResult(
        patient_id=pid,
        patient_name=memory[pid]["patient_name"],
        decision=decision,
        label=label,
        probability_ga=round(float(proba[1]), 4),
        probability_no_ga=round(float(proba[0]), 4),
        timestamp=now,
        visit_count=len(memory[pid]["visits"]),
    )

@app.get("/patients", response_model=list[PatientSummary])
def list_patients():
    memory = load_memory()
    result = []
    for pid, data in memory.items():
        visits = data["visits"]
        if not visits: continue
        last = visits[-1]
        result.append(PatientSummary(
            patient_id=pid,
            patient_name=data.get("patient_name"),
            visit_count=len(visits),
            last_visit=last["timestamp"],
            last_decision=last["decision"],
            last_label=last["label"],
        ))
    result.sort(key=lambda x: x.last_visit, reverse=True)
    return result

@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    memory = load_memory()
    pid = patient_id.strip().upper()
    if pid not in memory:
        raise HTTPException(status_code=404, detail=f"Patient '{pid}' not found.")
    return {"patient_id": pid, **memory[pid]}

@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: str):
    memory = load_memory()
    pid = patient_id.strip().upper()
    if pid not in memory:
        raise HTTPException(status_code=404, detail=f"Patient '{pid}' not found.")
    del memory[pid]
    save_memory(memory)
    return {"deleted": pid}

@app.get("/health")
def health():
    return {"status": "ok", "model": "LogisticRegression", "version": "3.0.0"}
