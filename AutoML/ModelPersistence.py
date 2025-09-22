# ModelPersistence.py

import joblib
import os

# ==========================
# Save the trained model
# ==========================
def save_model(model, path="best_model.pkl"):
    """
    Save a trained model to disk using joblib.
    
    Args:
        model: trained ML model
        path (str): file path to save model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if "/" in path else None
    joblib.dump(model, path)
    print(f"[INFO] Model saved successfully at {path}")


# ==========================
# Load a saved model
# ==========================
def load_model(path="best_model.pkl"):
    """
    Load a trained model from disk.
    
    Args:
        path (str): file path to the saved model
    
    Returns:
        model: loaded ML model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] No model found at {path}")
    model = joblib.load(path)
    print(f"[INFO] Model loaded successfully from {path}")
    return model
