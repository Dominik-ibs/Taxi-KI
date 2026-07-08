import joblib
import os
import pandas as pd
import sqlite3
from sklearn.ensemble import HistGradientBoostingRegressor

SOURCE_DB_PATH = "database/clean_train.sqlite"
MODEL_PATH = "model/hgbrModel"
SAMPLE_HASH = 2654435761

# 67 Million rows incur a RAM usage of around 70 GB
# Use this divider to reduce the RAM usage by randomly sampling
SAMPLE_DIVIDER = 5

QUERY = f"SELECT * FROM tripdata WHERE ABS(rowid * {SAMPLE_HASH}) % {SAMPLE_DIVIDER} = 0"

source_conn = sqlite3.connect(SOURCE_DB_PATH)

model = HistGradientBoostingRegressor(
    loss="squared_error",
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=69,
    verbose=1
)

features = pd.read_sql_query(QUERY, source_conn)
targets = features["Duration"]
features = features.drop(columns=["Duration"])

print(features.describe())
print(targets.describe())

model.fit(features, targets)
 
print(f"Training finished")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
 
print(f"Model Saved: {MODEL_PATH}")