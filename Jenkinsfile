pipeline {
  agent any
  stages {
    stage('Checkout') { steps { checkout scm } }
    stage('Install dependencies') { steps { sh 'pip install -r requirements.txt' } }
    stage('DVC Pull') { steps { sh 'dvc pull' } }
    stage('Run MLflow pipeline') { steps { sh 'python pipeline.py' } }
  }
}
