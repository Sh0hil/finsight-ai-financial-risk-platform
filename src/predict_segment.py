import pandas as pd

from src.config import (
    SEGMENT_MODEL_PATH,
    SEGMENT_SCALER_PATH,
    SEGMENT_PCA_PATH
)
from src.utils import load_model
from src.recommendations import get_segment_recommendation


segment_model = load_model(SEGMENT_MODEL_PATH)
segment_scaler = load_model(SEGMENT_SCALER_PATH)
segment_pca = load_model(SEGMENT_PCA_PATH)


def get_segment_name(segment_id: int):
    segment_mapping = {
        0: "Normal Low Activity Customers",
        1: "High Value Active Customers",
        2: "High Amount Transaction Customers",
        3: "Suspicious High Risk Customers",
        4: "Balance Draining Customers"
    }

    return segment_mapping.get(segment_id, "Unknown Segment")


def predict_customer_segment(input_data: dict):
    input_df = pd.DataFrame([input_data])

    scaled_data = segment_scaler.transform(input_df)
    segment_id = segment_model.predict(scaled_data)[0]
    segment_name = get_segment_name(int(segment_id))

    pca_values = segment_pca.transform(scaled_data)
    segment_recommendation = get_segment_recommendation(segment_name)

    return {
        "segment_id": int(segment_id),
        "segment_name": segment_name,
        "pca_1": round(float(pca_values[0][0]), 4),
        "pca_2": round(float(pca_values[0][1]), 4),
        **segment_recommendation
    }