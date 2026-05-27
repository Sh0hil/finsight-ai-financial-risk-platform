import pandas as pd

from src.predict_credit import predict_credit_risk
from src.predict_fraud import predict_fraud


def batch_predict_credit(file):
    """
    Batch prediction for credit risk.
    Input: uploaded CSV file
    Output: dataframe with predictions
    """

    df = pd.read_csv(file)

    results = []

    for _, row in df.iterrows():
        input_data = row.to_dict()
        prediction = predict_credit_risk(input_data)
        results.append(prediction)

    results_df = pd.DataFrame(results)

    final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    return final_df


def batch_predict_fraud(file):
    """
    Batch prediction for fraud detection.
    Input: uploaded CSV file
    Output: dataframe with predictions
    """

    df = pd.read_csv(file)

    results = []

    for _, row in df.iterrows():
        input_data = row.to_dict()
        prediction = predict_fraud(input_data)
        results.append(prediction)

    results_df = pd.DataFrame(results)

    final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    return final_df