# src/model_training.py
import os
from .pipeline_components import load_data, train_model, evaluate_model, save_model, log_mlflow_run, extract_data, preprocess_data
from sklearn.datasets import fetch_california_housing
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CSV = os.path.join(ROOT, "data", "raw_data.csv")
MODEL_OUT = os.path.join(ROOT, "models", "model.joblib")

def train_and_log_model(data_csv=DATA_CSV, model_out=MODEL_OUT):
    # Step 1: Extract (DVC)
    data_csv = extract_data(data_csv)
    
    # Step 2: Load
    X_train, X_test, y_train, y_test = load_data(data_csv)
    
    # Step 2b: Preprocess
    X_train, X_test = preprocess_data(X_train, X_test)
    
    # Step 3: Train
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate
    mse = evaluate_model(model, X_test, y_test)
    print(f"Test MSE: {mse:.4f}")
    
    # Save model locally
    save_model(model, model_out)
    
    # Log MLflow
    log_mlflow_run(model, mse, X_example=pd.DataFrame(X_train))


if __name__ == "__main__":
    # Download dataset if CSV does not exist
    if not os.path.exists(DATA_CSV):
        df = fetch_california_housing(as_frame=True).frame
        os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)
        df["MEDV"] = df["MedHouseVal"]  # create 'MEDV' column for compatibility
        df.drop(columns=["MedHouseVal"], inplace=True)
        df.to_csv(DATA_CSV, index=False)
        print(f"Downloaded California housing dataset to {DATA_CSV}")

    train_and_log_model()
