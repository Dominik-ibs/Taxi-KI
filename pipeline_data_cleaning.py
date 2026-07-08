import taxi_encoder
import numpy as np
import pandas as pd
import sqlite3
import os
import taxi_scaler, taxi_encoder

# ==========================
# Konfiguration
# ==========================

SOURCE_TRAIN_DB = "train.sqlite" # Muss später parametrisiert werden
SOURCE_TEST_DB = "test.sqlite" # Muss später parametrisiert werden

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
# Chunkweise Verarbeitung
# ==========================

query = f"SELECT {", ".join(COLUMNS)} FROM {TABLE} WHERE {WHERE}"

for db in [SOURCE_TRAIN_DB, SOURCE_TEST_DB]:
    rows_before = 0
    rows_after = 0
    mode = "replace"

    source_db_path = f"database/{db}"
    target_db_path = f"database/clean_{db}"

    if os.path.exists(target_db_path):
        os.remove(target_db_path)

    source_conn = sqlite3.connect(source_db_path)

    target_conn = sqlite3.connect(target_db_path)
    target_conn.execute("PRAGMA journal_mode=WAL")
    target_conn.execute("PRAGMA synchronous=OFF")
    target_conn.execute("PRAGMA cache_size=100000")
    target_conn.execute("PRAGMA temp_store=MEMORY")

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
            (df["Avg_Speed"] >= 0.3)
            & (df["Avg_Speed"] <= 60)
        ]

        rows_after += len(df)

        # =====================================================
        # 4. Datum aufspalten
        # =====================================================    

        df["time_of_day"] = df.tpep_pickup_datetime.dt.hour * 3600 + df.tpep_pickup_datetime.dt.minute * 60 + df.tpep_pickup_datetime.dt.second
        df["day_of_week"] = df.tpep_pickup_datetime.dt.dayofweek % 7
        df["month"] = df.tpep_pickup_datetime.dt.month % 12

        # =====================================================
        # 5. Datumszeilen entfernen
        # =====================================================

        df = df.drop(columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "Avg_Speed"])

        # =====================================================
        # 6. Daten Skalieren
        # =====================================================

        distance_scaler = taxi_scaler.load_distance_scaler()
        duration_scaler = taxi_scaler.load_duration_scaler()

        df[["trip_distance"]] = distance_scaler.transform(df[["trip_distance"]])
        df[["Duration"]] = duration_scaler.transform(df[["Duration"]])

        df["time_of_day"] = np.sin(2 * np.pi * df["time_of_day"] / 86400)

        # =====================================================
        # 7. One-Hot Encoding für Wochentage und Monate
        # =====================================================

        encoder = taxi_encoder.load_encoder()
        ENCODER_COLUMNS = taxi_encoder.get_encoder_cols()

        encoded_values = encoder.transform(df[ENCODER_COLUMNS])

        encoded_df = pd.DataFrame(
            encoded_values,
            columns=encoder.get_feature_names_out(ENCODER_COLUMNS),
            index=df.index
        )

        df = pd.concat(
            [
                df.drop(columns=ENCODER_COLUMNS),
                encoded_df
            ],
            axis=1
        )

        # =====================================================
        # 8. Daten speichern
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