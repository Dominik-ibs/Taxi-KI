import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split 
import os

chunksize = 1_000_000
SOURCE_DB_DIR = "database/raw"
TRAIN_DB_PATH = "database/train.sqlite"
TEST_DB_PATH = "database/test.sqlite"

TABLE = 'tripdata'

train_row_count = 0
test_row_count = 0

mode = 'w'

if os.path.exists(TRAIN_DB_PATH):
    os.remove(TRAIN_DB_PATH)

if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)

train_conn = sqlite3.connect(TRAIN_DB_PATH)
train_conn.execute('PRAGMA journal_mode=WAL')
train_conn.execute('PRAGMA synchronous=OFF')
train_conn.execute('PRAGMA cache_size=100000')
train_conn.execute('PRAGMA temp_store=MEMORY')

test_conn = sqlite3.connect(TEST_DB_PATH)
test_conn.execute('PRAGMA journal_mode=WAL')
test_conn.execute('PRAGMA synchronous=OFF')
test_conn.execute('PRAGMA cache_size=100000')
test_conn.execute('PRAGMA temp_store=MEMORY')

for filename in os.listdir(SOURCE_DB_DIR):
    if not os.path.isfile(os.path.join(SOURCE_DB_DIR, filename)):
        continue
    if not filename.endswith(".sqlite"):
        continue
    conn = sqlite3.connect(os.path.join(SOURCE_DB_DIR, filename))

    for chunk in pd.read_sql_query("SELECT * FROM tripdata", conn, chunksize=chunksize):
        train_set, test_set = train_test_split(chunk, test_size=0.2, random_state=69)

        train_set.to_sql(TABLE, train_conn, if_exists='append', index=False)
        test_set.to_sql(TABLE, test_conn, if_exists='append', index=False)

        train_row_count += len(train_set)
        test_row_count += len(test_set)

        mode = 'a'

    conn.close()

train_conn.close()
test_conn.close()

print(f'Train rows: {train_row_count}')
print(f'Test rows: {test_row_count}')