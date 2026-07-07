import pandas as pd
import sqlite3
import os

# ==========================
# Konfiguration
# ==========================

SOURCE_DB_PATH = "database/raw/tripdata.sqlite"
OUTPUT_DB_PATH = "database/cleaned.sqlite"

TABLE = "tripdata"
CHUNKSIZE = 1_000_000

# ==========================
# Zu entfernende Spalten
# ==========================

DROP_COLUMNS = [
    "VendorID",
    "Passenger_count",
    "RatecodeID",
    "Store_and_fwd_flag",
    "Payment_type",
    "Fare_amount",
    "Extra",
    "Mta_tax",
    "Tip_amount",
    "Tolls_amount",
    "Improvement_surcharge",
    "Total_amount",
    "Congestion_surcharge",
]

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

query = f"SELECT * FROM {TABLE}"

for chunk_nr, df in enumerate(
    pd.read_sql_query(query, source_conn, chunksize=CHUNKSIZE)
):
    print(f"Verarbeite Chunk {chunk_nr:,}")

    rows_before += len(df)

    # =====================================================
    # 1. Entfernte Spalten löschen
    # =====================================================

    existing_drop_columns = [
        col for col in DROP_COLUMNS
        if col in df.columns
    ]

    df.drop(columns=existing_drop_columns, inplace=True)

    # =====================================================
    # 2. Zeitspalten konvertieren
    # =====================================================

    df["Pickup_time"] = pd.to_datetime(
        df["Pickup_time"],
        errors="coerce"
    )

    df["Dropoff_time"] = pd.to_datetime(
        df["Dropoff_time"],
        errors="coerce"
    )

    # keine NULL-Werte
    df = df.dropna(
        subset=[
            "Pickup_time",
            "Dropoff_time"
        ]
    )

    # innerhalb von 2019
    start_2019 = pd.Timestamp("2019-01-01")
    end_2019 = pd.Timestamp("2020-01-01")

    df = df[
        (df["Pickup_time"] >= start_2019)
        & (df["Pickup_time"] < end_2019)
        & (df["Dropoff_time"] >= start_2019)
        & (df["Dropoff_time"] < end_2019)
    ]

    # =====================================================
    # 3. Trip distance prüfen
    # =====================================================

    df["Trip_distance"] = pd.to_numeric(
        df["Trip_distance"],
        errors="coerce"
    )

    df = df[
        df["Trip_distance"].notna()
    ]

    df = df[
        (df["Trip_distance"] >= 0.1)
        & (df["Trip_distance"] <= 40)
    ]

    # =====================================================
    # 4. PU / DO Location prüfen
    # =====================================================

    df["PULocationID"] = pd.to_numeric(
        df["PULocationID"],
        errors="coerce"
    )

    df["DOLocationID"] = pd.to_numeric(
        df["DOLocationID"],
        errors="coerce"
    )

    df = df[
        df["PULocationID"].between(1, 263)
        & df["DOLocationID"].between(1, 263)
    ]

    # =====================================================
    # 5. Feature Engineering
    # =====================================================

    df["Duration"] = (
        df["Dropoff_time"] - df["Pickup_time"]
    ).dt.total_seconds()

    # 10 Sekunden bis 3 Stunden
    df = df[
        (df["Duration"] > 10)
        & (df["Duration"] <= 10800)
    ]

    # Durchschnittsgeschwindigkeit mph
    df["Speed"] = (
        df["Trip_distance"]
        / (df["Duration"] / 3600)
    )

    df = df[
        (df["Speed"] >= 0.3)
        & (df["Speed"] <= 60)
    ]

    rows_after += len(df)

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