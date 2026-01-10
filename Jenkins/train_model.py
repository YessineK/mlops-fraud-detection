import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from notebooks.preprocessing_fraud_class import PreprocessingFraud

# Configuration MLflow et DagsHub
DAGSHUB_USER = os.getenv('DAGSHUB_USER', 'karrayyessine1')
DAGSHUB_REPO = os.getenv('DAGSHUB_REPO', 'mlops-fraud-detection')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN', os.getenv('MLFLOW_TRACKING_PASSWORD', ''))

# Configurer les credentials
os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USER
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

# Set tracking URI
MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"✅ MLflow configured: {MLFLOW_TRACKING_URI}")
print(f"✅ User: {DAGSHUB_USER}")

# Set experiment
mlflow.set_experiment("continuous_training_pipeline")
