import sqlite3
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TEST_DB_PATH = "database/cleaned_test.sqlite"
TABLE = "tripdata"
TARGET = "Duration"

MODEL_PATH = "model/trip_duration_model.pkl"


# ==========================
# Modell laden
# ==========================

model = joblib.load(MODEL_PATH)


# ==========================
# Testdaten laden
# ==========================

print("Starte laden der Testdaten...")

conn = sqlite3.connect(TEST_DB_PATH)

df_test = pd.read_sql_query(
    f"SELECT * FROM {TABLE}",
    conn
)

conn.close()

print(f"Testdaten geladen: {len(df_test):,} Zeilen.")

# ==========================
# Features und Zielwert trennen
# ==========================

X_test = df_test.drop(columns=[TARGET])
y_test = df_test[TARGET]


# ==========================
# Vorhersagen
# ==========================

y_pred = model.predict(X_test)


# ==========================
# Bewertung
# ==========================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


print("===== Modellbewertung =====")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")