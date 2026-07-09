import joblib
import os
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestRegressor

SOURCE_DB_PATH = "database/clean_train.sqlite"
MODEL_PATH = "model/randomForestModel"
SAMPLE_HASH = 2654435761

# 67 Million rows incur a RAM usage of around 70 GB
# Use this divider to reduce the RAM usage by randomly sampling
SAMPLE_DIVIDER = 5

QUERY = f"""
SELECT *
FROM tripdata
WHERE ABS(rowid * {SAMPLE_HASH}) % {SAMPLE_DIVIDER} = 0
"""

source_conn = sqlite3.connect(SOURCE_DB_PATH)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=69,
    verbose=1
)

features = pd.read_sql_query(QUERY, source_conn)

targets = features["Duration"]
features = features.drop(columns=["Duration"])

print(features.describe())
print(targets.describe())

print("Starting training...")

model.fit(features, targets)

print("Training finished")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"Model Saved: {MODEL_PATH}")