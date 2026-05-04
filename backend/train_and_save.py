"""
train_and_save.py
Run this to retrain the model on your machine.
Usage: python3 train_and_save.py --data "final data ag  3 .xlsx"
"""
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
args = parser.parse_args()

df = pd.read_excel(args.data, header=None)
df.columns = df.iloc[1]
df = df[2:].reset_index(drop=True)
df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")

asa_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
df["asa_classification"] = df["asa_classification"].astype(str).str.strip()
df["asa"] = df["asa_classification"].map(asa_map)

def age_from_bands(row):
    for col, mid in {"1-3": 2, "4-6": 5, "7-9": 8, "10-12": 11, ">12": 14}.items():
        try:
            if float(row[col]) == 1:
                return mid
        except: pass
    return np.nan

df["age"] = df.apply(age_from_bands, axis=1)

for col in df.columns:
    if col not in ["file_number", "asa_classification"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df_clean = df.drop(columns=["file_number", "asa_classification"])

FEATURE_ORDER = [
    "age", "asa", "icdas_max", "nombre_de_dents_severes",
    "calm", "agitated", "short_attention", "anxious",
    "extremely_agitated", "uncooperative",
    "special_needs", "previous_general_anesthesia",
    "presence_of_infection", "previous_failed_dental_attempts",
]
SCALED_COLS = ["icdas_max", "nombre_de_dents_severes", "asa"]

X = df_clean[FEATURE_ORDER]
y = df_clean["general_anesthesia_decision"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = X_train.copy(), X_test.copy()

scaler = StandardScaler()
X_train[SCALED_COLS] = scaler.fit_transform(X_train[SCALED_COLS])
X_test[SCALED_COLS]  = scaler.transform(X_test[SCALED_COLS])

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))
print("Accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 4))

joblib.dump(model,  "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Saved model.pkl and scaler.pkl")