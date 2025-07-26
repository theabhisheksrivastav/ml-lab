import os
import joblib

def save_model(model, path):
    """Save the model to a .joblib file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved to: {path}")

def load_model(path):
    """Load a model from a .joblib file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No model found at: {path}")
    print(f"📦 Loading model from: {path}")
    return joblib.load(path)
