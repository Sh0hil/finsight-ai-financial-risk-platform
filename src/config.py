from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

CREDIT_MODEL_PATH = MODEL_DIR / "credit_risk_model.pkl"
FRAUD_MODEL_PATH = MODEL_DIR / "fraud_detection_model.pkl"

SEGMENT_MODEL_PATH = MODEL_DIR / "customer_segmentation_model.pkl"
SEGMENT_SCALER_PATH = MODEL_DIR / "customer_segmentation_scaler.pkl"
SEGMENT_PCA_PATH = MODEL_DIR / "customer_segmentation_pca.pkl"

CUSTOMER_SEGMENTS_PATH = DATA_DIR / "processed" / "customer_segments.csv"