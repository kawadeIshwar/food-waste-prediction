"""
Train and evaluate the Food Waste Prediction model.
Run: python train_model.py
Dataset: data/Food_Waste_Daily_Records.csv
Note: the supplied records are simulated academic demonstration data.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Food_Waste_Daily_Records.csv"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Attendance", "Exam_Day", "Holiday_or_Low_Service", "Menu_Type",
    "Weather", "Meals_Prepared", "Food_Prepared_kg", "Day"
]
TARGET = "Food_Wasted_kg"
CATEGORICAL = ["Exam_Day", "Holiday_or_Low_Service", "Menu_Type", "Weather", "Day"]
NUMERICAL = [c for c in FEATURES if c not in CATEGORICAL]

def main():
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numerical", "passthrough", NUMERICAL)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=250, random_state=42, min_samples_leaf=2
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "algorithm": "Random Forest Regressor",
        "records": int(len(df)),
        "training_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "MAE_kg": round(float(mean_absolute_error(y_test, predictions)), 3),
        "RMSE_kg": round(float(mean_squared_error(y_test, predictions) ** 0.5), 3),
        "R2_score": round(float(r2_score(y_test, predictions)), 3),
        "data_note": "Simulated academic demonstration data; not real canteen measurements."
    }
    joblib.dump(model, MODEL_DIR / "food_waste_random_forest.joblib")
    (OUT_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    test = pd.DataFrame({
        "Actual_Waste_kg": y_test.to_numpy(),
        "Predicted_Waste_kg": predictions.round(2),
    })
    test["Absolute_Error_kg"] = (test["Actual_Waste_kg"] - test["Predicted_Waste_kg"]).abs().round(2)
    test.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    all_predictions = df.copy()
    all_predictions["Model_Predicted_Waste_kg"] = model.predict(X).round(2)
    all_predictions.to_csv(OUT_DIR / "records_with_model_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model to: {MODEL_DIR / 'food_waste_random_forest.joblib'}")

if __name__ == "__main__":
    main()
