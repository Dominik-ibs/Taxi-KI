import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

def load_encoder() -> OneHotEncoder:
    ENCODER_DIR = Path("encoders")

    ENCODER_FILE = ENCODER_DIR / "onehot_encoder.pkl"

    if not ENCODER_FILE.exists():
        create_encoder()

    return joblib.load(ENCODER_FILE)

def create_encoder():
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

        encoder.fit(encoder_fit_data[get_encoder_cols()])

        joblib.dump(encoder, ENCODER_FILE)
        print("Encoder erstellt und gespeichert.")

def get_encoder_cols():
    return ["day_of_week", "month"]