import pandas as pd

from src.config import FRAUD_MODEL_PATH
from src.utils import load_model
from src.recommendations import get_fraud_recommendation


fraud_model = load_model(FRAUD_MODEL_PATH)


def create_fraud_features(input_data: dict):
    """
    Create fraud features from raw transaction input.
    """

    data = input_data.copy()

    data["balance_diff_orig"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
    data["balance_diff_dest"] = data["newbalanceDest"] - data["oldbalanceDest"]
    data["amount_balance_ratio"] = data["amount"] / (data["oldbalanceOrg"] + 1)
    data["is_balance_drained"] = int(data["newbalanceOrig"] == 0)

    return data


def predict_fraud(input_data: dict):
    """
    Predict whether a transaction is fraudulent.
    """

    processed_data = create_fraud_features(input_data)

    input_df = pd.DataFrame([processed_data])

    prediction = fraud_model.predict(input_df)[0]
    probability = fraud_model.predict_proba(input_df)[:, 1][0]

    recommendation = get_fraud_recommendation(probability)

    return {
        "prediction": int(prediction),
        **recommendation
    }