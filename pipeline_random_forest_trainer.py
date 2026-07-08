import joblib
import os
import pandas as pd
import sqlite3

from sklearn.ensemble import RandomForestRegressor

SOURCE_DB_PATH = "database/clean_train.sqlite"
MODEL_PATH = "model/randomForestModel"

QUERY = f"""
SELECT *
FROM tripdata
"""

source_conn = sqlite3.connect(SOURCE_DB_PATH)

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    verbose=1
)

features = pd.read_sql_query(QUERY, source_conn)

targets = features["Duration"]
features = features.drop(columns=["Duration"])

print(features.describe())
print(targets.describe())

print("Training started...")

model.fit(features, targets)

print("Training finished")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

joblib.dump(model, MODEL_PATH)

print(f"Model saved: {MODEL_PATH}")