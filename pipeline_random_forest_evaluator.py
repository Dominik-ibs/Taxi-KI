import joblib
import sqlite3
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import taxi_scaler

SOURCE_DB_PATH = "database/clean_test.sqlite"
MODEL_PATH = "model/randomForestModel"

model = joblib.load(MODEL_PATH)

source_conn = sqlite3.connect(SOURCE_DB_PATH)

x_features = pd.read_sql_query("SELECT * FROM tripdata", source_conn)
y_target = x_features[["Duration"]]
x_features = x_features.drop(columns=["Duration"])

y_prediction = pd.DataFrame(model.predict(x_features), columns=["Duration"])

duration_scaler = taxi_scaler.load_duration_scaler()

y_target[["Duration"]] = duration_scaler.inverse_transform(y_target[["Duration"]])
y_prediction[["Duration"]] = duration_scaler.inverse_transform(y_prediction[["Duration"]])

mae = mean_absolute_error(y_target, y_prediction)
rmse = root_mean_squared_error(y_target, y_prediction)
r2 = r2_score(y_target, y_prediction)

print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")