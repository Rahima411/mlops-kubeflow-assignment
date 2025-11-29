# src/pipeline_components.py
import mlflow
import mlflow.sklearn
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd
from sklearn import set_config
import subprocess
from sklearn.preprocessing import StandardScaler

# Optional: Configure sklearn to output pandas DataFrames in pipelines
set_config(transform_output="pandas")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Setup MLflow backend (SQLite)
MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MLFLOW_ARTIFACT_ROOT = os.path.join(ROOT, "mlruns")
os.makedirs(MLFLOW_ARTIFACT_ROOT, exist_ok=True)


def extract_data(dvc_path="data/raw_data.csv"):
    """Pull dataset from DVC remote"""
    subprocess.run(["dvc", "pull", dvc_path], check=True)
    return dvc_path

def load_data(csv_path: str):
    """Load dataset from CSV"""
    df = pd.read_csv(csv_path)
    if "MEDV" not in df.columns:
        raise ValueError("Target column 'MEDV' not found in dataset")
    X = df.drop(columns=["MEDV"])
    y = df["MEDV"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_model(X_train, y_train):
    """Train Linear Regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return MSE"""
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse

def save_model(model, output_path: str):
    """Save trained model locally"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")

def log_mlflow_run(model, mse, X_example=None, model_name="LinearRegression"):
    """Log params, metrics, and model artifact to MLflow with signature & input example"""
    mlflow.sklearn.autolog(log_input_examples=True)
    with mlflow.start_run():
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("mse", mse)
        if X_example is not None:
            # Log model with example to infer signature
            mlflow.sklearn.log_model(
                model, 
                artifact_path="model", 
                input_example=X_example.iloc[:5]
            )
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")
    print(f"Run logged in MLflow with MSE={mse:.4f}")
