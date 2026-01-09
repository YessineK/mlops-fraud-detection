import os
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
import dagshub
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from dotenv import load_dotenv

# --- Configuration ---
# Calculate paths relative to this script
# this file is at Projects/MLOps/Jenkins/train_model.py
# Root is at Projects/MLOps/ (2 levels up)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

# DagsHub & MLflow Config from .env or Environment Variables
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USER', 'karrayyessine1')
DAGSHUB_REPO = os.getenv('DAGSHUB_REPO', 'mlops-fraud-detection')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")

# Initialize DagsHub and MLflow
try:
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
except Exception as e:
    print(f"Warning: Could not check DagsHub init (might be already initialized or env vars missing): {e}")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("continuous_training_pipeline")

# --- Functions ---

def load_data(filepath):
    """Charge les données preprocessées."""
    print(f"Chargement des données depuis {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier de données non trouvé: {filepath}")
        
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba)
    }

def train_and_track():
    """Fonction principale d'entraînement."""
    
    # Chemin vers les données relative to project root
    DATA_PATH = BASE_DIR / "notebooks" / "processors" / "preprocessed_data.pkl"
    
    # 1. Chargement
    try:
        data = load_data(DATA_PATH)
    except FileNotFoundError:
         print(f"Error: Data file not found at {DATA_PATH}")
         return

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    print(f"Data loaded. Train shape: {X_train.shape}")

    # 2. Définition des modèles Baseline (identique au notebook)
    baseline_models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0)
    }

    # 3. Entraînement et Tracking
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"🚀 Starting Continuous Training for {len(baseline_models)} models...")

    for name, model in baseline_models.items():
        print(f"Training {name}...")
        
        with mlflow.start_run(run_name=f"{name}_CT_{run_timestamp}"):
            # Log Params
            mlflow.log_params(model.get_params())
            mlflow.log_param('model_name', name)
            mlflow.log_param('stage', 'continuous_training')
            mlflow.log_param('data_source', DATA_PATH)
            
            # Train
            start_time = datetime.now()
            model.fit(X_train, y_train)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            metrics = calculate_metrics(y_test, y_pred, y_proba)
            
            # Log Metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_metric('training_time_seconds', duration)
            
            print(f"  --> {name} finished. ROC-AUC: {metrics['roc_auc']:.4f} ({duration:.1f}s)")
            
            # Log Model
            mlflow.sklearn.log_model(model, "model")

    print("\n✅ Continuous Training Pipeline Completed.")

if __name__ == "__main__":
    train_and_track()
