import joblib
import os
import pandas as pd
import sqlite3
from sklearn.ensemble import HistGradientBoostingRegressor
from pipeline_data_cleaning import clean_dataframe

SOURCE_DB_PATH = "database/cleaned.sqlite"
MODEL_PATH = "model/hgbrModel"

TABLE = "tripdata"
CHUNKSIZE = 1_000_000

source_conn = sqlite3.connect(SOURCE_DB_PATH)

QUERY = "SELECT * FROM tripdata"

model = HistGradientBoostingRegressor(
    loss="squared_error",
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=69
)

for chunk_nr, features in enumerate(pd.read_sql_query(QUERY, source_conn, chunksize=CHUNKSIZE)):
    print(f"cleaning chunk {chunk_nr}")
    
    features = clean_dataframe(features)

    print(f"Training on chunk {chunk_nr}")

    targets = features["Duration"]
    features = features.drop(columns=["Duration"])

    model.fit(features, targets)
 
print(f"Training finished")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
 
print(f"Model Saved: {MODEL_PATH}")