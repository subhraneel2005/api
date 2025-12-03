import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model2", "cricket_model.pkl")
columns_path = os.path.join(BASE_DIR, "model2", "ohe_columns.pkl")

# Load model + one-hot column names
model = joblib.load(model_path)
model_columns = joblib.load(columns_path)

feature_order = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']


def preprocess(input_data):
    """
    Convert input dict → DataFrame → One-hot encoded row with all model columns.
    Missing dummies = 0.
    """

    df = pd.DataFrame([input_data])

    # One-hot encode like training
    df_encoded = pd.get_dummies(df)

    # Add missing columns
    missing_cols = set(model_columns) - set(df_encoded.columns)
    for col in missing_cols:
        df_encoded[col] = 0

    # Order columns
    df_encoded = df_encoded[model_columns]

    return df_encoded


def predict(input_data):
    df = preprocess(input_data)
    pred = model.predict(df)[0]

    # pred = 1 → team1 wins
    # pred = 0 → team2 wins
    return input_data["team1"] if pred == 1 else input_data["team2"]
