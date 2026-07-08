import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def create_scaler():
    SCALER_DIR = Path("scalers")
    SCALER_DIR.mkdir(parents=True, exist_ok=True)

    SCALER_FILE = SCALER_DIR / "minmax_scaler.pkl"

    SCALER_COLUMNS = ["trip_distance", "Duration"]

    FIXED_MIN_VALUES = {
        "trip_distance": 0,
        "Duration": 10
    }

    FIXED_MAX_VALUES = {
        "trip_distance": 100,
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

def create_encoder():
    ENCODER_COLUMNS = ["day_of_week", "month"]
    ENCODER_DIR = Path("encoders")
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)

    ENCODER_FILE = ENCODER_DIR / "onehot_encoder.pkl"

    if ENCODER_FILE.exists():
        print("Encoder existiert.")
    else:
        encoder = OneHotEncoder(
            categories=[
                [0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            ],
            sparse_output=False,
            drop="first",
            handle_unknown="ignore"
        )

        encoder_fit_data = pd.DataFrame({
            "day_of_week": [0, 1, 2, 3, 4, 5, 6] * 12,
            "month": sorted(list(range(12)) * 7)
        })

        encoder.fit(encoder_fit_data[ENCODER_COLUMNS])

        joblib.dump(encoder, ENCODER_FILE)
        print("Encoder erstellt und gespeichert.")
