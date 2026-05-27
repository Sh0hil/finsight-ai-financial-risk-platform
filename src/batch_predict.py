import pandas as pd
import numpy as np

from src.config import CREDIT_MODEL_PATH, FRAUD_MODEL_PATH
from src.utils import load_model
from src.recommendations import get_credit_recommendation, get_fraud_recommendation


credit_model = load_model(CREDIT_MODEL_PATH)
fraud_model = load_model(FRAUD_MODEL_PATH)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            median_value = df[col].median()
            if pd.isna(median_value):
                median_value = 0
            df[col] = df[col].fillna(median_value)

    return df


def prepare_credit_batch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])

    if "CREDIT_INCOME_RATIO" not in df.columns:
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

    if "ANNUITY_INCOME_RATIO" not in df.columns:
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    if "CREDIT_TERM" not in df.columns:
        df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    if "DAYS_EMPLOYED_RATIO" not in df.columns:
        df["DAYS_EMPLOYED_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]

    df = clean_dataframe(df)

    return df


def prepare_fraud_batch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols = ["isFraud", "isFlaggedFraud", "nameOrig", "nameDest"]

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "balance_diff_orig" not in df.columns:
        df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

    if "balance_diff_dest" not in df.columns:
        df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    if "amount_balance_ratio" not in df.columns:
        df["amount_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    if "is_balance_drained" not in df.columns:
        df["is_balance_drained"] = (df["newbalanceOrig"] == 0).astype(int)

    df = clean_dataframe(df)

    return df


def batch_predict_credit(df: pd.DataFrame) -> pd.DataFrame:
    input_df = prepare_credit_batch(df)

    predictions = credit_model.predict(input_df)
    probabilities = credit_model.predict_proba(input_df)[:, 1]

    results = []

    for pred, prob in zip(predictions, probabilities):
        recommendation = get_credit_recommendation(prob)

        results.append({
            "prediction": int(pred),
            **recommendation
        })

    results_df = pd.DataFrame(results)

    final_df = pd.concat(
        [input_df.reset_index(drop=True), results_df],
        axis=1
    )

    final_df = final_df.replace([np.inf, -np.inf], np.nan)
    final_df = final_df.where(pd.notnull(final_df), None)

    return final_df


def batch_predict_fraud(df: pd.DataFrame) -> pd.DataFrame:
    input_df = prepare_fraud_batch(df)

    predictions = fraud_model.predict(input_df)
    probabilities = fraud_model.predict_proba(input_df)[:, 1]

    results = []

    for pred, prob in zip(predictions, probabilities):
        recommendation = get_fraud_recommendation(prob)

        results.append({
            "prediction": int(pred),
            **recommendation
        })

    results_df = pd.DataFrame(results)

    final_df = pd.concat(
        [input_df.reset_index(drop=True), results_df],
        axis=1
    )

    final_df = final_df.replace([np.inf, -np.inf], np.nan)
    final_df = final_df.where(pd.notnull(final_df), None)

    return final_df