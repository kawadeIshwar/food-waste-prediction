"""
Interactive dashboard for Food Waste Prediction.
Run: streamlit run app.py
"""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Food_Waste_Daily_Records.csv"
MODEL_PATH = ROOT / "models" / "food_waste_random_forest.joblib"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
GRAPH_DIR = ROOT / "graphs"

FEATURES = ["Attendance", "Exam_Day", "Holiday_or_Low_Service", "Menu_Type",
            "Weather", "Meals_Prepared", "Food_Prepared_kg", "Day"]
CATEGORICAL = ["Exam_Day", "Holiday_or_Low_Service", "Menu_Type", "Weather", "Day"]
NUMERICAL = [c for c in FEATURES if c not in CATEGORICAL]

@st.cache_resource
def get_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    pre = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numerical", "passthrough", NUMERICAL)
    ])
    model = Pipeline([
        ("preprocessor", pre),
        ("regressor", RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2))
    ])
    model.fit(df[FEATURES], df["Food_Wasted_kg"])
    return model

st.set_page_config(page_title="College Canteen Food Waste Prediction", page_icon="🍽️", layout="wide")
st.title("🍽️ Food Waste Prediction for a College Canteen")
st.caption("Machine Learning project • Random Forest Regression")
st.warning("Dataset notice: the included 120 daily records are simulated sample data for academic demonstration, not actual measurements from a specific college canteen.")

df = pd.read_csv(DATA_PATH)
model = get_model()

tab1, tab2, tab3 = st.tabs(["Dashboard", "Predict Food Waste", "Dataset & Model"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Daily records", f"{len(df):,}")
    c2.metric("Average daily waste", f"{df['Food_Wasted_kg'].mean():.2f} kg")
    c3.metric("Average waste rate", f"{df['Waste_Percentage'].mean():.2f}%")
    st.subheader("Project graphs")
    graph_files = [
        ("01_daily_observed_vs_predicted.png", "Observed vs predicted waste"),
        ("02_actual_vs_predicted.png", "Actual vs predicted (test set)"),
        ("03_waste_distribution.png", "Distribution of daily waste"),
        ("04_average_waste_by_menu.png", "Average waste by menu type"),
        ("05_average_waste_by_day.png", "Average waste by day of week"),
        ("06_attendance_vs_waste.png", "Attendance vs waste"),
        ("07_daily_waste_percentage.png", "Daily waste percentage"),
        ("08_monthly_total_waste.png", "Monthly total waste"),
    ]
    for i in range(0, len(graph_files), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(graph_files):
                fname, caption = graph_files[i+j]
                path = GRAPH_DIR / fname
                if path.exists():
                    col.image(str(path), caption=caption, use_container_width=True)

with tab2:
    st.write("Enter the canteen's planned daily information. The model estimates food waste in kilograms.")
    left, right = st.columns(2)
    with left:
        attendance = st.number_input("Expected student attendance", min_value=1, max_value=5000, value=300, step=10)
        meals_prepared = st.number_input("Meals to prepare", min_value=1, max_value=6000, value=330, step=10)
        prepared_kg = st.number_input("Total food prepared (kg)", min_value=1.0, max_value=3000.0, value=120.0, step=5.0)
        day = st.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    with right:
        menu = st.selectbox("Menu type", ["North Indian", "South Indian", "Chinese", "Snacks", "Special Meal"])
        weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy"])
        exam = st.selectbox("Is it an exam day?", ["No", "Yes"])
        holiday = st.selectbox("Holiday or low-service day?", ["No", "Yes"])
    if st.button("Predict food waste", type="primary"):
        sample = pd.DataFrame([{
            "Attendance": attendance, "Exam_Day": exam,
            "Holiday_or_Low_Service": holiday, "Menu_Type": menu,
            "Weather": weather, "Meals_Prepared": meals_prepared,
            "Food_Prepared_kg": prepared_kg, "Day": day
        }])[FEATURES]
        estimate = max(0.0, float(model.predict(sample)[0]))
        rate = min(100.0, 100 * estimate / prepared_kg) if prepared_kg else 0.0
        st.success(f"Estimated food waste: **{estimate:.2f} kg**")
        st.metric("Estimated waste percentage", f"{rate:.1f}%")
        st.info("Use this estimate as a planning aid only. Validate the model with real local canteen data before operational use.")

with tab3:
    st.subheader("Sample data")
    st.dataframe(df.head(30), use_container_width=True)
    st.download_button("Download all daily records (CSV)", data=DATA_PATH.read_bytes(),
                       file_name="Food_Waste_Daily_Records.csv", mime="text/csv")
    st.subheader("Model evaluation")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.json(metrics)
    else:
        st.write("Run `python train_model.py` to generate model metrics.")
    st.write("**Features:** attendance, exam-day status, holiday/low-service status, menu type, weather, meals prepared, total food prepared, and weekday.")
    st.write("**Target:** food wasted in kilograms.")
    st.write("**Algorithm:** Random Forest Regressor.")
