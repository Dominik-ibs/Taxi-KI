import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

SCALER_DIR = Path("scalers")

def load_distance_scaler() -> MinMaxScaler:
    SCALER_FILE = SCALER_DIR / "minmax_distance_scaler.pkl"

    if not SCALER_FILE.exists():
        create_distance_scaler()

    return joblib.load(SCALER_FILE)

def load_duration_scaler() -> MinMaxScaler:
    SCALER_FILE = SCALER_DIR / "minmax_duration_scaler.pkl"

    if not SCALER_FILE.exists():
        create_duration_scaler()

    return joblib.load(SCALER_FILE)

def create_distance_scaler():
    SCALER_DIR.mkdir(parents=True, exist_ok=True)

    SCALER_FILE = SCALER_DIR / "minmax_distance_scaler.pkl"

    SCALER_COLUMNS = ["trip_distance"]

    FIXED_MIN_VALUES = {
        "trip_distance": 0,
    }

    FIXED_MAX_VALUES = {
        "trip_distance": 100,
    }

    if os.path.exists(SCALER_FILE):
        print("Scaler existiert.")
    else:
        scaler = MinMaxScaler()

        fixed_scaler_data = pd.DataFrame([
            FIXED_MIN_VALUES,
            FIXED_MAX_VALUES
        ])

        scaler.fit(fixed_scaler_data[SCALER_COLUMNS])

        joblib.dump(scaler, SCALER_FILE)
        print("Scaler mit festen Min-/Max-Werten erstellt und gespeichert.")

def create_duration_scaler():
    SCALER_DIR.mkdir(parents=True, exist_ok=True)

    SCALER_FILE = SCALER_DIR / "minmax_duration_scaler.pkl"

    SCALER_COLUMNS = ["Duration"]

    FIXED_MIN_VALUES = {
        "Duration": 10
    }

    FIXED_MAX_VALUES = {
        "Duration": 10800
    }

    if os.path.exists(SCALER_FILE):
        print("Scaler existiert.")
    else:
        scaler = MinMaxScaler()

        fixed_scaler_data = pd.DataFrame([
            FIXED_MIN_VALUES,
            FIXED_MAX_VALUES
        ])

        scaler.fit(fixed_scaler_data[SCALER_COLUMNS])

        joblib.dump(scaler, SCALER_FILE)
        print("Scaler mit festen Min-/Max-Werten erstellt und gespeichert.")

        
