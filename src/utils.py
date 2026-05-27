import joblib
from pathlib import Path


def load_model(model_path):
    """
    Load a saved joblib/pkl model.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)