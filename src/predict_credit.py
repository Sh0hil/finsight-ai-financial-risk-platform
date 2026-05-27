import pandas as pd

from src.config import CREDIT_MODEL_PATH
from src.utils import load_model
from src.recommendations import get_credit_recommendation


credit_model = load_model(CREDIT_MODEL_PATH)


def predict_credit_risk(input_data: dict):
    """
    Predict credit default risk from applicant input data.
    """

    input_df = pd.DataFrame([input_data])

    prediction = credit_model.predict(input_df)[0]
    probability = credit_model.predict_proba(input_df)[:, 1][0]

    recommendation = get_credit_recommendation(probability)

    return {
        "prediction": int(prediction),
        **recommendation
    }