def create_fraud_features(df):
    df = df.copy()

    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["is_balance_drained"] = (df["newbalanceOrig"] == 0).astype(int)

    return df