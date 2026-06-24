import joblib
import pandas as pd
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load artifacts
kmeans = joblib.load(os.path.join(MODEL_DIR, "customer_segmentation_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "customer_segmentation_scaler.pkl"))
pca = joblib.load(os.path.join(MODEL_DIR, "customer_segmentation_pca.pkl"))

def predict_customer_segment(data: dict):
    """
    Predicts the customer segment based on transaction behavior.
    """
    # 🚨 CRITICAL FIX: Remove target leakage columns if they are passed in the payload
    data.pop("total_fraud_transactions", None)
    data.pop("fraud_ratio", None)
    
    # Convert exactly 14 features to DataFrame
    df = pd.DataFrame([data])
    
    # Scale data
    X_scaled = scaler.transform(df)
    
    # Apply PCA (for dashboard visualization purposes)
    X_pca = pca.transform(X_scaled)
    pca_1, pca_2 = X_pca[0][0], X_pca[0][1]
    
    # Predict Segment
    segment_id = kmeans.predict(X_scaled)[0]
    
    # Define Segment Names (Matching your notebook logic)
    segment_mapping = {
        0: "Normal Low Activity Customers",
        1: "High Value Active Customers",
        2: "High Amount Transaction Customers",
        3: "Suspicious High Risk Customers",
        4: "Balance Draining Customers"
    }
    
    segment_name = segment_mapping.get(segment_id, "Unknown Segment")
    
    # Determine Business Action
    if segment_id == 3:
        action = "Monitor account closely and apply stricter transaction checks."
        priority = "Critical"
    elif segment_id == 4:
        action = "Review account for potential takeover or rapid drain."
        priority = "High"
    elif segment_id == 1:
        action = "Engage with premium services and retention offers."
        priority = "Medium"
    else:
        action = "Standard monitoring."
        priority = "Low"

    return {
        "segment_id": int(segment_id),
        "segment_name": segment_name,
        "pca_1": round(float(pca_1), 4),
        "pca_2": round(float(pca_2), 4),
        "segment_action": action,
        "business_priority": priority
    }