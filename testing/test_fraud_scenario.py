import sys
import os
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

# Add backend/src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/src')))

# Import InferencePreprocessor from api (assuming it can be imported without side effects)
# If api.py has side effects, we might need to refactor or copy the class.
# Trying import first.
try:
    from api import InferencePreprocessor
except ImportError:
    # Fallback if running from root
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
    from backend.src.api import InferencePreprocessor

def load_model():
    """Load the model from the local registry."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_registry_dir = Path(base_dir) / 'notebooks' / 'model_registry'
    model_name = "Best_Fraud_LightGBM"
    model_path = model_registry_dir / model_name / "production.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def test_specific_fraud_case():
    """
    Test a specific fraud case provided by the user.
    Expected prediction: Class 1 (Fraud).
    """
    # 1. Load Model
    model = load_model()
    print("✅ Model loaded.")
    
    # 2. Initialize Preprocessor
    preprocessor = InferencePreprocessor()
    print("✅ Preprocessor initialized.")
    
    # 3. Define Input Data
    data = {
        'trans_date_trans_time': "2019-01-18 23:20:16",
        "category": "shopping_net",
        "amt": 1334.07,
        "gender": 'F',
        "zip": 29438,
        "lat": 32.5486,
        "long": -80.307,
        "dob": "1997-07-05",
        "merch_lat": 31.615611,
        "merch_long": -79.702908
    }
    
    df_input = pd.DataFrame([data])
    
    # 4. Preprocess
    X = preprocessor.preprocess_inference(df_input)
    print("✅ Data preprocessed.")
    
    # 5. Predict
    prediction = model.predict(X)
    print(f"✅ Prediction: {prediction}")
    
    # 6. Assert
    # Assuming prediction is an array or list
    pred_value = prediction[0] if isinstance(prediction, (np.ndarray, list)) else prediction
    
    assert pred_value == 1, f"Expected prediction 1 (Fraud), but got {pred_value}"
    print("✅ Test Passed: Fraud detected correctly.")

if __name__ == "__main__":
    test_specific_fraud_case()
