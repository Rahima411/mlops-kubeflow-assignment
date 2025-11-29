# pipeline.py
from src.model_training import train_and_log_model

if __name__ == "__main__":
    print("Starting MLflow pipeline...")
    train_and_log_model()
    print("Pipeline completed.")
