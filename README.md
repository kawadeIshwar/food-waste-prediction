# Food Waste Prediction for a College Canteen

A beginner-friendly machine-learning project using a Random Forest Regressor to estimate daily food waste from canteen planning information.

## Project contents
- `app.py` — interactive Streamlit dashboard with eight graphs and a prediction form.
- `train_model.py` — trains and evaluates the Random Forest model.
- `data/Food_Waste_Daily_Records.csv` — 120 daily records.
- `data/Food_Waste_Prediction_Records_and_Graphs.xlsx` — Excel workbook with detailed records and summary sheets.
- `graphs/` — eight PNG graphs used in the dashboard.
- `outputs/` — report-ready graphs PDF, Word report content, and model outputs created by training.
- `models/` — trained model is saved here after running the training script.

## Important dataset note
The included dataset is **simulated sample data** created for academic demonstration. It is not collected from a real college canteen. Do not describe its results as real-world measurements. Replace it with verified records if your assignment requires field data.

## Run on Windows
1. Install Python 3.10 or newer.
2. Extract this ZIP to a folder.
3. Open Command Prompt or PowerShell in that folder.
4. Create a virtual environment (recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
5. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
6. Train the model and create evaluation output files:
   ```powershell
   python train_model.py
   ```
7. Launch the dashboard:
   ```powershell
   streamlit run app.py
   ```
8. A browser tab should open. Use the **Predict Food Waste** tab to enter attendance and meal-planning details.

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Methodology
1. Read daily records from CSV.
2. One-hot encode categorical variables.
3. Train a Random Forest Regressor on an 80% training split.
4. Evaluate on the 20% test split using MAE, RMSE and R².
5. Use the trained pipeline to estimate waste for new planning inputs.

## Evaluation metrics
- MAE: mean absolute prediction error in kilograms.
- RMSE: root mean squared error in kilograms.
- R²: model fit compared with a mean-value baseline.

The scores are for a simulated dataset and should not be treated as evidence of expected real-world accuracy.
