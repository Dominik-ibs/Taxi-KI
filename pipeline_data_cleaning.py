import pandas as pd
import sqlite3
import os

# ==========================
# Konfiguration
# ==========================

SOURCE_DB_PATH = "database/train.sqlite" # Muss später parametrisiert werden
OUTPUT_DB_PATH = "database/cleaned.sqlite"

TABLE = "tripdata"
CHUNKSIZE = 1_000_000

# ==========================
# Zu behaltende Spalten
# ==========================

COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "pulocationid",
    "dolocationid",
]

WHERE = "trip_distance BETWEEN 0.1 and 40 and pulocationid BETWEEN 1 and 263 and dolocationid BETWEEN 1 and 263"

# ==========================
# Konvertieriung der Datetime-Einträge
# ==========================

DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

PARSE_DATE_DICT = {
    "tpep_pickup_datetime":DATE_FORMAT,
    "tpep_dropoff_datetime":DATE_FORMAT
}

# ==========================
# Datenbanken vorbereiten
# ==========================

if os.path.exists(OUTPUT_DB_PATH):
    os.remove(OUTPUT_DB_PATH)

source_conn = sqlite3.connect(SOURCE_DB_PATH)

target_conn = sqlite3.connect(OUTPUT_DB_PATH)
target_conn.execute("PRAGMA journal_mode=WAL")
target_conn.execute("PRAGMA synchronous=OFF")
target_conn.execute("PRAGMA cache_size=100000")
target_conn.execute("PRAGMA temp_store=MEMORY")

rows_before = 0
rows_after = 0
mode = "replace"

# ==========================
# Chunkweise Verarbeitung
# ==========================

query = f"SELECT {", ".join(COLUMNS)} FROM {TABLE} WHERE {WHERE}"

for chunk_nr, df in enumerate(
    pd.read_sql_query(query, source_conn, parse_dates=PARSE_DATE_DICT, chunksize=CHUNKSIZE)
):
    print(f"Verarbeite Chunk {chunk_nr:,}")

    rows_before += len(df)

    # =====================================================
    # 1. Zeitspalten filtern
    # =====================================================

    # innerhalb von 2019
    start_2019 = pd.Timestamp("2019-01-01")
    end_2019 = pd.Timestamp("2020-01-01")

    df = df[
        (df["tpep_pickup_datetime"] >= start_2019)
        & (df["tpep_pickup_datetime"] < end_2019)
        & (df["tpep_dropoff_datetime"] >= start_2019)
        & (df["tpep_dropoff_datetime"] < end_2019)
    ]

    # =====================================================
    # 2. Trip distance prüfen
    # =====================================================

    df = df[
        df["trip_distance"].notna()
    ]

    # =====================================================
    # 3. Feature Engineering
    # =====================================================

    df["Duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds()

    # 10 Sekunden bis 3 Stunden
    df = df[
        (df["Duration"] > 10)
        & (df["Duration"] <= 10800)
    ]

    # Durchschnittsgeschwindigkeit mph
    df["Avg_Speed"] = (
        df["trip_distance"]
        / (df["Duration"] / 3600)
    )

    df = df[
        (df["AVG_Speed"] >= 0.3)
        & (df["AVG_Speed"] <= 60)
    ]

    rows_after += len(df)

    # =====================================================
    # 4. Datum aufspalten
    # =====================================================    

    df["hour_of_day"] = df.tpep_pickup_datetime.dt.hour
    df["day_of_week"] = (df.tpep_pickup_datetime.dt.dayofweek + 1) % 7
    df["month"] = df.tpep_pickup_datetime.dt.month

    # =====================================================
    # 5. Datumszeilen entfernen
    # =====================================================

    df = df.drop(columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "Avg_Speed"], axis=1)

    # =====================================================
    # 6. Daten speichern
    # =====================================================

    df.to_sql(
        TABLE,
        target_conn,
        if_exists=mode,
        index=False
    )

    mode = "append"

    print(
        f"Chunk {chunk_nr:,}: "
        f"{len(df):,} Zeilen gespeichert"
    )

# ==========================
# Abschluss
# ==========================

source_conn.close()
target_conn.close()

removed = rows_before - rows_after

print("\n===== Statistik =====")
print(f"Ursprüngliche Zeilen: {rows_before:,}")
print(f"Verbleibende Zeilen:  {rows_after:,}")
print(f"Entfernte Zeilen:     {removed:,}")
print(f"Entfernungsrate:      {removed / rows_before:.2%}")