import os
import json
import joblib
import pandas as pd
from utils.finance_utils import load_quartiles,EXPENSE_COLS


QUARTILES,SRC = load_quartiles()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR,"models","savings_model.joblib")

saved = joblib.load(MODEL_PATH)

pipeline = saved["pipeline"]
feature_cols = saved["feature_cols"]
target_cols = saved["target_cols"]


async def forecast_savings(json_path:str,desired_savings_percentage:float):

    with open(json_path) as f:
        data = json.load(f)

    X = pd.DataFrame([data])

    for col in feature_cols:
        if col not in X.columns:
            X[col]=0

    X_pred = X[feature_cols]

    preds = pipeline.predict(X_pred)

    return {
        "predicted_savings":preds.tolist()
    }