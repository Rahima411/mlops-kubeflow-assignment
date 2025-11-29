# MLOps Kubeflow Assignment

## Project Overview
This project demonstrates a complete **Machine Learning Operations (MLOps) pipeline** for a simple regression problem using the **California Housing dataset**.  
The pipeline integrates **data versioning, model training, evaluation, and MLflow logging** using industry-standard tools including **DVC, Kubeflow Pipelines (KFP), Kubernetes (Minikube), and Jenkins/GitHub Actions**.  

The ML model trained is a **Linear Regression** model to predict housing prices (`MEDV`), with the entire workflow automated and reproducible.

---

## Project Structure

```

mlops-kubeflow-assignment/
│
├─ data/                   # Raw and processed datasets
│  └─ raw_data.csv
│
├─ src/                    # Python scripts
│  ├─ __init__.py
│  ├─ pipeline_components.py  # Kubeflow pipeline component functions
│  └─ model_training.py       # Main training and MLflow logging script
│
├─ components/             # Compiled Kubeflow component YAMLs
│
├─ models/                 # Saved model artifacts
│  └─ model.joblib
│
├─ pipeline.py             # Main Kubeflow pipeline definition
├─ pipeline.yaml           # Compiled pipeline YAML
├─ requirements.txt        # Python dependencies
├─ Dockerfile              # Custom component image (if needed)
├─ Jenkinsfile             # CI/CD pipeline definition (Jenkins or GitHub Actions)
├─ .dvc/                   # DVC meta-files and config
├─ mlflow.db               # SQLite MLflow backend
└─ README.md

````

---

## Setup Instructions

### 1. Clone the Repository
```
git clone https://github.com/<your-username>/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment
```

### 2. Create Python Virtual Environment

```
python -m venv venv
.\venv\Scripts\Activate.ps1  # PowerShell
# or
source venv/bin/activate      # Linux/macOS
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Initialize DVC and Pull Dataset

```
dvc init
dvc remote add -d myremote <remote-storage-path>
dvc pull
```

### 5. Setup Minikube and Kubeflow Pipelines

```
minikube start
# Follow Kubeflow Pipelines standalone installation guide
# Access KFP dashboard: http://localhost:8080/pipeline
```

---

## Pipeline Walkthrough

The Kubeflow pipeline consists of **four core components**:

1. **Data Extraction**

   * Fetches versioned dataset from DVC remote storage.
   * Input: Dataset path
   * Output: Local dataset CSV

2. **Data Preprocessing**

   * Cleans, scales, and splits the dataset into training and test sets.
   * Input: Dataset CSV
   * Output: `X_train`, `X_test`, `y_train`, `y_test`

3. **Model Training**

   * Trains a Linear Regression model on the training data.
   * Input: Preprocessed training data
   * Output: Saved model artifact (`model.joblib`)

4. **Model Evaluation**

   * Evaluates the model on test data and logs metrics to MLflow.
   * Input: Trained model, test data
   * Output: MSE metric and MLflow logged run

### Run the Pipeline Locally

```
python pipeline.py
```

* This will run the full pipeline: extract → preprocess → train → evaluate → log.


---

## Continuous Integration

* **Jenkins/GitHub Actions** pipeline automates:

  1. Environment setup (install dependencies)
  2. Pipeline compilation (`pipeline.yaml`)
  3. Unit testing for components

* **Jenkinsfile** / GitHub Workflow stages:

  * Checkout code
  * Install Python dependencies
  * Compile Kubeflow pipeline
  * Trigger pipeline run

---

## MLflow Tracking

* **Backend**: SQLite (`mlflow.db`)
* **Artifacts**: Saved models (`models/model.joblib`)
* **Logged Parameters & Metrics**:

  * Model type: Linear Regression
  * Test MSE
  * Input example & model signature

Access MLflow UI locally:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

* Open: `http://localhost:5000`

---

## Data Versioning with DVC

* Dataset tracked with DVC: `data/raw_data.csv`
* Commands:

```
dvc status
dvc push
```

* Ensures reproducibility and consistent dataset versioning.

---

## Dependencies

`requirements.txt` includes:

```
scikit-learn
pandas
numpy
joblib
dvc
mlflow
kfp
```

---

## Notes

* Dataset: **California Housing** (`MEDV` column for regression target)
* Model: **Linear Regression**
* Fully modular pipeline with reusable Kubeflow components
* Supports local execution, DVC versioning, and CI/CD automation
* Ready for deployment on **Minikube + Kubeflow Pipelines**

---
