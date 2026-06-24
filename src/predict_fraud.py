import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load the newly saved Random Forest model
fraud_model = joblib.load(os.path.join(MODEL_DIR, "fraud_detection_model.pkl"))

def predict_fraud(data: dict):
    """
    Predicts fraud probability for a single transaction.
    """
    df = pd.DataFrame([data])
    
    # 🚨 CRITICAL FIX: Dynamic Feature Engineering (7 features -> 11 features)
    df['balance_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['amount_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 0.01) 
    df['is_balance_drained'] = (df['amount'] == df['oldbalanceOrg']).astype(int)
    
    # Predict using the 11-feature dataframe
    probability = fraud_model.predict_proba(df)[:, 1][0]
    prediction = fraud_model.predict(df)[0]
    
    # Risk Level & Business Logic
    if probability >= 0.8:
        risk_level = "Critical Risk"
        decision = "Block Transaction"
        action = "Block transaction immediately and freeze account temporarily."
    elif probability >= 0.5:
        risk_level = "High Risk"
        decision = "Manual Verification"
        action = "Hold transaction and trigger identity verification or OTP."
    elif probability >= 0.3:
        risk_level = "Medium Risk"
        decision = "Step-up Authentication"
        action = "Require additional authentication method."
    else:
        risk_level = "Low Risk"
        decision = "Allow Transaction"
        action = "Process transaction normally."

    return {
        "prediction": int(prediction),
        "fraud_probability": round(float(probability), 4),
        "risk_level": risk_level,
        "decision": decision,
        "recommended_action": action
    }