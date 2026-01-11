import os
import sys
import pandas as pd
import numpy as np
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import io
from datetime import datetime

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing_fraud_class import PreprocessingFraud

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')
MODEL_REGISTRY_NAME = os.getenv('MODEL_REGISTRY_NAME', 'fraud_detection_best_model')
MODEL_STAGE = 'Production'

# Set MLflow tracking URI
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD:
        os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent transactions using MLflow model registry",
    version="1.0.0"
)

# Global variables
model = None
preprocessor = None
model_metadata = {}

class InferencePreprocessor(PreprocessingFraud):
    """
    Subclass of PreprocessingFraud adapted for inference.
    """
    def __init__(self):
        super().__init__()
        # Load processors immediately
        self.load_processors()
        
    def preprocess_inference(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess new data for inference.
        """
        # 1. Copy input
        self.df_clean = df_input.copy()
        
        # 2. Basic Cleaning (Date conversion if present)
        if 'trans_date_trans_time' in self.df_clean.columns:
            self.df_clean['trans_date_trans_time'] = pd.to_datetime(self.df_clean['trans_date_trans_time'])
        if 'dob' in self.df_clean.columns:
            self.df_clean['dob'] = pd.to_datetime(self.df_clean['dob'])
            
        # 3. Feature Engineering
        # Temporal features
        if 'trans_date_trans_time' in self.df_clean.columns:
            self.df_clean['trans_hour'] = self.df_clean['trans_date_trans_time'].dt.hour
            self.df_clean['trans_day'] = self.df_clean['trans_date_trans_time'].dt.day
            self.df_clean['trans_month'] = self.df_clean['trans_date_trans_time'].dt.month
            self.df_clean['trans_year'] = self.df_clean['trans_date_trans_time'].dt.year
            self.df_clean['trans_dayofweek'] = self.df_clean['trans_date_trans_time'].dt.dayofweek
            self.df_clean['is_weekend'] = (self.df_clean['trans_dayofweek'] >= 5).astype(int)
            
            # Period
            self.df_clean['day_period'] = self.df_clean['trans_hour'].apply(self._get_period)
            
        # Age
        if 'trans_date_trans_time' in self.df_clean.columns and 'dob' in self.df_clean.columns:
             self.df_clean['age'] = (self.df_clean['trans_date_trans_time'] - self.df_clean['dob']).dt.days / 365.25
             
        # Distance
        if all(col in self.df_clean.columns for col in ['lat', 'long', 'merch_lat', 'merch_long']):
            self.df_clean['distance_km'] = self._haversine_distance(
                self.df_clean['lat'], self.df_clean['long'],
                self.df_clean['merch_lat'], self.df_clean['merch_long']
            )
            
        # Amount Category
        if 'amt' in self.df_clean.columns:
            self.df_clean['amt_category'] = pd.cut(
                self.df_clean['amt'],
                bins=[0, 50, 100, 200, float('inf')],
                labels=['faible', 'moyen', 'élevé', 'très_élevé']
            )
            
        # 4. Encoding
        for col in self.categorical_features:
            if col in self.df_clean.columns:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    known_classes = set(le.classes_)
                    self.df_clean[col] = self.df_clean[col].astype(str).apply(lambda x: x if x in known_classes else le.classes_[0])
                    self.df_clean[col] = le.transform(self.df_clean[col])
        
        # 5. Selection and Scaling
        numeric_cols = self.feature_names.get('numerical_features', [])
        
        # Ensure all expected columns exist
        for col in numeric_cols:
            if col not in self.df_clean.columns:
                self.df_clean[col] = 0
                
        # Scale
        if self.scaler:
            self.df_clean[numeric_cols] = self.scaler.transform(self.df_clean[numeric_cols])
            
        # Return only the features expected by the model
        all_features = self.feature_names.get('all_features', [])

        if not all_features:
            available_cols = [c for c in self.df_clean.columns if c not in ['trans_date_trans_time', 'dob']]
            return self.df_clean[available_cols]
            
        # Ensure all features exist
        missing_features = []
        for col in all_features:
            if col not in self.df_clean.columns:
                missing_features.append(col)
                self.df_clean[col] = 0

        if missing_features:
            print(f"⚠️ Missing features filled with 0: {missing_features[:5]}...")
                
        return self.df_clean[all_features]

@app.on_event("startup")
async def startup_event():
    global model, preprocessor, model_metadata
    
    # Initialize Preprocessor
    try:
        preprocessor = InferencePreprocessor()
        print("✅ Preprocessor initialized and processors loaded.")
    except Exception as e:
        print(f"❌ Error initializing preprocessor: {e}")
        
    # Load Model from Local Registry
    try:
        from pathlib import Path
        import json
        import pickle
        
        # ✅ NOUVEAU PATH - Chercher dans processors/
        model_path = Path(__file__).parent / "processors" / "Best_Fraud_RandomForest_model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        print(f"🔄 Loading model from: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model loaded successfully!")
        
        model_metadata = {
            "name": "Best_Fraud_RandomForest",
            "uri": str(model_path),
            "loaded_at": datetime.now().isoformat(),
            "source": "local_processors",
        }
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️ Running without model (predictions will fail).")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fraud Detection API. Use /docs for API documentation."}

@app.get("/health")
def health_check():
    status = "ok" if model is not None and preprocessor is not None else "degraded"
    return {"status": status, "model_loaded": model is not None, "preprocessor_loaded": preprocessor is not None}

@app.get("/features")
def get_features():
    if preprocessor and preprocessor.feature_names:
        return {
            "numerical": preprocessor.feature_names.get('numerical_features', []),
            "categorical": preprocessor.feature_names.get('categorical_features', []),
            "all_expected": preprocessor.feature_names.get('all_features', [])
        }
    return {"error": "Preprocessor not initialized or features not loaded"}

@app.get("/model-info")
def get_model_info():
    return model_metadata

# Pydantic model for input validation and default values
class TransactionInput(BaseModel):
    trans_date_trans_time: str = "2025-01-01 12:00:00"
    amt: float = 100.0
    lat: float = 40.7128
    long: float = -74.0060
    merch_lat: float = 40.7200
    merch_long: float = -74.0100
    category: str = "grocery_pos"
    gender: str = "M"
    state: str = "NY"
    zip: int = 10001
    city_pop: int = 50000
    job: str = "Developer"
    dob: str = "1990-01-01"

@app.post("/predict")
async def predict(request: Union[TransactionInput, List[TransactionInput]]):
    """
    Predict fraud for a single transaction or a batch of transactions.
    """
    if not model or not preprocessor:
        raise HTTPException(status_code=503, detail="Model or preprocessor not available")
    
    try:
        # Handle different input types
        if isinstance(request, list):
            df_input = pd.DataFrame([item.dict() for item in request])
        else:
            df_input = pd.DataFrame([request.dict()])
            
        # Preprocess
        X = preprocessor.preprocess_inference(df_input)
        
        # Predict
        predictions = model.predict(X)
        
        return {
            "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            "count": len(predictions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predictCSV")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file, get predictions, and download the result CSV.
    """
    if not model or not preprocessor:
        raise HTTPException(status_code=503, detail="Model or preprocessor not available")
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
        
    try:
        # Read CSV
        contents = await file.read()
        df_input = pd.read_csv(io.BytesIO(contents))
        
        # Preprocess
        X = preprocessor.preprocess_inference(df_input)
        
        # Predict
        predictions = model.predict(X)
        
        # Add predictions to result
        df_result = df_input.copy()
        df_result['predict'] = predictions
        
        # Save to buffer
        output = io.StringIO()
        df_result.to_csv(output, index=False)
        output.seek(0)
        
        # Return file
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=prediction_{file.filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
