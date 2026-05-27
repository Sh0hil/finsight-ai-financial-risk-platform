import streamlit as st
import requests
import pandas as pd
import numpy as np
import os


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")



def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def safe_divide(numerator, denominator, default=0.0):
    """Avoid ZeroDivisionError, inf, and NaN values."""
    try:
        if denominator in [0, 0.0, None] or pd.isna(denominator):
            return default
        value = numerator / denominator
        if pd.isna(value) or np.isinf(value):
            return default
        return float(value)
    except Exception:
        return default


def clean_payload(payload: dict) -> dict:
    """
    Clean row dictionary before sending to FastAPI.
    JSON does not support NaN, inf, or numpy-specific numeric types.
    """
    cleaned = {}

    for key, value in payload.items():
        if pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, np.integer):
            cleaned[key] = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
            cleaned[key] = None if (np.isnan(value) or np.isinf(value)) else value
        elif isinstance(value, float):
            cleaned[key] = None if (np.isnan(value) or np.isinf(value)) else value
        else:
            cleaned[key] = value

    return cleaned


def post_to_api(endpoint: str, payload: dict):
    """Send a safe request to FastAPI and return response/error."""
    try:
        payload = clean_payload(payload)
        response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=30)
        return response
    except requests.exceptions.ConnectionError:
        st.error("FastAPI server is not running. Start it with: python -m uvicorn app.main:app --reload")
        return None
    except requests.exceptions.InvalidJSONError as e:
        st.error(f"Invalid JSON payload. Check NaN/inf values. Details: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None


def prepare_credit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean batch credit-risk CSV and create missing engineered features."""
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop target column if user uploads training/processed data.
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])

    # Create engineered features if missing.
    if "CREDIT_INCOME_RATIO" not in df.columns and {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = df.apply(
            lambda r: safe_divide(r["AMT_CREDIT"], r["AMT_INCOME_TOTAL"]), axis=1
        )

    if "ANNUITY_INCOME_RATIO" not in df.columns and {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_RATIO"] = df.apply(
            lambda r: safe_divide(r["AMT_ANNUITY"], r["AMT_INCOME_TOTAL"]), axis=1
        )

    if "CREDIT_TERM" not in df.columns and {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        df["CREDIT_TERM"] = df.apply(
            lambda r: safe_divide(r["AMT_ANNUITY"], r["AMT_CREDIT"]), axis=1
        )

    if "DAYS_EMPLOYED_RATIO" not in df.columns and {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
        df["DAYS_EMPLOYED_RATIO"] = df.apply(
            lambda r: safe_divide(r["DAYS_EMPLOYED"], r["DAYS_BIRTH"]), axis=1
        )

    # Fill known categorical fields.
    categorical_defaults = {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "OCCUPATION_TYPE": "Unknown",
    }

    for col, default in categorical_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    # Fill numeric columns with median, then 0 if whole column is missing.
    for col in df.select_dtypes(include=[np.number]).columns:
        median_value = df[col].median()
        if pd.isna(median_value):
            median_value = 0
        df[col] = df[col].fillna(median_value)

    # Fill remaining object columns.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")

    return df


def prepare_fraud_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean batch fraud CSV before sending rows to FastAPI."""
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop target/ID columns if user uploads original PaySim data.
    columns_to_drop = [col for col in ["isFraud", "isFlaggedFraud", "nameOrig", "nameDest"] if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    if "type" in df.columns:
        df["type"] = df["type"].fillna("PAYMENT")

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(0)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")

    return df


def show_api_error(response):
    """Display helpful FastAPI validation errors."""
    if response is None:
        return
    try:
        st.error(f"API request failed. Status code: {response.status_code}")
        st.code(response.text)
    except Exception:
        st.error("API request failed, and error details could not be displayed.")



# Streamlit UI

st.set_page_config(
    page_title="FinSight AI",
    page_icon="💳",
    layout="wide"
)

st.title("FinSight AI: Financial Risk Intelligence Platform")
st.write("Credit Risk Prediction, Fraud Detection, and Customer Segmentation using Machine Learning")

page = st.sidebar.selectbox(
    "Select Module",
    [
        "Project Overview",
        "Credit Risk Prediction",
        "Fraud Detection",
        "Customer Segmentation",
        "Batch Prediction"
    ]
)


if page == "Project Overview":
    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Credit Risk Model", "Active")

    with col2:
        st.metric("Fraud Detection Model", "Active")

    with col3:
        st.metric("Segmentation Model", "Active")

    st.subheader("About this Project")
    st.write(
        """
        FinSight AI is an end-to-end financial risk intelligence platform.
        It predicts loan default risk, detects fraudulent transactions,
        segments customers based on behavior, and provides business recommendations.
        """
    )


elif page == "Credit Risk Prediction":
    st.header("Credit Risk Prediction")

    with st.form("credit_form"):
        col1, col2 = st.columns(2)

        with col1:
            NAME_CONTRACT_TYPE = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
            CODE_GENDER = st.selectbox("Gender", ["M", "F"])
            FLAG_OWN_CAR = st.selectbox("Own Car", ["Y", "N"])
            FLAG_OWN_REALTY = st.selectbox("Own Realty", ["Y", "N"])
            CNT_CHILDREN = st.number_input("Children Count", min_value=0, max_value=10, value=1)
            AMT_INCOME_TOTAL = st.number_input("Income Total", min_value=1.0, value=157500.0)
            AMT_CREDIT = st.number_input("Credit Amount", min_value=1.0, value=770292.0)
            AMT_ANNUITY = st.number_input("Annuity Amount", min_value=1.0, value=30676.5)
            AMT_GOODS_PRICE = st.number_input("Goods Price", min_value=1.0, value=688500.0)
            NAME_INCOME_TYPE = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant"])

        with col2:
            NAME_EDUCATION_TYPE = st.selectbox(
                "Education Type",
                ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"]
            )
            NAME_FAMILY_STATUS = st.selectbox(
                "Family Status",
                ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
            )
            NAME_HOUSING_TYPE = st.selectbox(
                "Housing Type",
                ["House / apartment", "Rented apartment", "With parents", "Municipal apartment"]
            )
            DAYS_BIRTH = st.number_input("Days Birth", value=-13506)
            DAYS_EMPLOYED = st.number_input("Days Employed", value=-105)
            OCCUPATION_TYPE = st.selectbox(
                "Occupation Type",
                ["Laborers", "Core staff", "Managers", "Sales staff", "Drivers", "Accountants", "Unknown"]
            )
            CNT_FAM_MEMBERS = st.number_input("Family Members", min_value=1.0, value=3.0)

        submitted = st.form_submit_button("Predict Credit Risk")

    if submitted:
        payload = {
            "NAME_CONTRACT_TYPE": NAME_CONTRACT_TYPE,
            "CODE_GENDER": CODE_GENDER,
            "FLAG_OWN_CAR": FLAG_OWN_CAR,
            "FLAG_OWN_REALTY": FLAG_OWN_REALTY,
            "CNT_CHILDREN": CNT_CHILDREN,
            "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
            "AMT_CREDIT": AMT_CREDIT,
            "AMT_ANNUITY": AMT_ANNUITY,
            "AMT_GOODS_PRICE": AMT_GOODS_PRICE,
            "NAME_INCOME_TYPE": NAME_INCOME_TYPE,
            "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
            "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS,
            "NAME_HOUSING_TYPE": NAME_HOUSING_TYPE,
            "DAYS_BIRTH": DAYS_BIRTH,
            "DAYS_EMPLOYED": DAYS_EMPLOYED,
            "OCCUPATION_TYPE": OCCUPATION_TYPE,
            "CNT_FAM_MEMBERS": CNT_FAM_MEMBERS,
            "CREDIT_INCOME_RATIO": safe_divide(AMT_CREDIT, AMT_INCOME_TOTAL),
            "ANNUITY_INCOME_RATIO": safe_divide(AMT_ANNUITY, AMT_INCOME_TOTAL),
            "CREDIT_TERM": safe_divide(AMT_ANNUITY, AMT_CREDIT),
            "DAYS_EMPLOYED_RATIO": safe_divide(DAYS_EMPLOYED, DAYS_BIRTH),
        }

        response = post_to_api("/predict-credit-risk", payload)

        if response is not None and response.status_code == 200:
            result = response.json()
            st.success("Prediction completed!")
            st.metric("Default Probability", result.get("default_probability"))
            st.metric("Risk Level", result.get("risk_level"))
            st.write("Decision:", result.get("decision"))
            st.write("Recommendation:", result.get("recommendation"))
        else:
            show_api_error(response)


elif page == "Fraud Detection":
    st.header("Fraud Detection")

    with st.form("fraud_form"):
        step = st.number_input("Step", min_value=0, value=1)
        transaction_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Amount", min_value=0.0, value=181.0)
        oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, value=181.0)
        newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, value=0.0)
        oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=0.0)
        newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=0.0)

        submitted = st.form_submit_button("Predict Fraud")

    if submitted:
        payload = {
            "step": step,
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
        }

        response = post_to_api("/predict-fraud", payload)

        if response is not None and response.status_code == 200:
            result = response.json()
            st.success("Fraud prediction completed!")
            st.metric("Fraud Probability", result.get("fraud_probability"))
            st.metric("Risk Level", result.get("risk_level"))
            st.write("Decision:", result.get("decision"))
            st.write("Recommended Action:", result.get("recommended_action"))
        else:
            show_api_error(response)


elif page == "Customer Segmentation":
    st.header("Customer Segmentation")

    with st.form("segment_form"):
        col1, col2 = st.columns(2)

        with col1:
            total_transactions = st.number_input("Total Transactions", min_value=0, value=5)
            total_amount = st.number_input("Total Amount", min_value=0.0, value=250000.0)
            avg_amount = st.number_input("Average Amount", min_value=0.0, value=50000.0)
            max_amount = st.number_input("Max Amount", min_value=0.0, value=120000.0)
            min_amount = st.number_input("Min Amount", min_value=0.0, value=10000.0)
            total_fraud_transactions = st.number_input("Total Fraud Transactions", min_value=0, value=1)
            fraud_ratio = st.number_input("Fraud Ratio", min_value=0.0, max_value=1.0, value=0.2)

        with col2:
            avg_old_balance = st.number_input("Average Old Balance", min_value=0.0, value=100000.0)
            avg_new_balance = st.number_input("Average New Balance", min_value=0.0, value=50000.0)
            max_old_balance = st.number_input("Max Old Balance", min_value=0.0, value=200000.0)
            max_new_balance = st.number_input("Max New Balance", min_value=0.0, value=100000.0)
            CASH_IN = st.number_input("CASH_IN Count", min_value=0, value=0)
            CASH_OUT = st.number_input("CASH_OUT Count", min_value=0, value=2)
            DEBIT = st.number_input("DEBIT Count", min_value=0, value=0)
            PAYMENT = st.number_input("PAYMENT Count", min_value=0, value=1)
            TRANSFER = st.number_input("TRANSFER Count", min_value=0, value=2)

        submitted = st.form_submit_button("Predict Segment")

    if submitted:
        payload = {
            "total_transactions": total_transactions,
            "total_amount": total_amount,
            "avg_amount": avg_amount,
            "max_amount": max_amount,
            "min_amount": min_amount,
            "total_fraud_transactions": total_fraud_transactions,
            "fraud_ratio": fraud_ratio,
            "avg_old_balance": avg_old_balance,
            "avg_new_balance": avg_new_balance,
            "max_old_balance": max_old_balance,
            "max_new_balance": max_new_balance,
            "CASH_IN": CASH_IN,
            "CASH_OUT": CASH_OUT,
            "DEBIT": DEBIT,
            "PAYMENT": PAYMENT,
            "TRANSFER": TRANSFER,
        }

        response = post_to_api("/predict-segment", payload)

        if response is not None and response.status_code == 200:
            result = response.json()
            st.success("Customer segment prediction completed!")
            st.metric("Segment ID", result.get("segment_id"))
            st.metric("Segment Name", result.get("segment_name"))
            st.write("Business Priority:", result.get("business_priority"))
            st.write("Recommended Action:", result.get("segment_action"))
        else:
            show_api_error(response)


elif page == "Batch Prediction":
    st.header("Batch Prediction")

    prediction_type = st.selectbox(
        "Select Prediction Type",
        ["Credit Risk", "Fraud Detection"]
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Unable to read CSV file: {e}")
            st.stop()

        df = df.replace([np.inf, -np.inf], np.nan)

        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        st.info(
            "This batch prediction now uses the FastAPI batch endpoint. "
            "The CSV is uploaded once to the backend, and the model predicts multiple rows together."
        )

        max_rows = st.number_input(
            "Number of rows to predict",
            min_value=1,
            max_value=len(df),
            value=min(100, len(df)),
            step=50
        )

        if st.button("Run Batch Prediction"):
            if prediction_type == "Credit Risk":
                endpoint = "/batch-predict-credit-risk"
                output_filename = "credit_batch_prediction_results.csv"
            else:
                endpoint = "/batch-predict-fraud"
                output_filename = "fraud_batch_prediction_results.csv"

            # Send the uploaded dataframe as one CSV file to FastAPI batch endpoint.
            # FastAPI will handle feature engineering, cleaning, and prediction.
            csv_bytes = df.to_csv(index=False).encode("utf-8")

            files = {
                "file": ("batch_input.csv", csv_bytes, "text/csv")
            }

            params = {
                "max_rows": int(max_rows)
            }

            with st.spinner("Running batch prediction through FastAPI..."):
                try:
                    response = requests.post(
                        f"{API_URL}{endpoint}",
                        files=files,
                        params=params,
                        timeout=120
                    )
                except requests.exceptions.ConnectionError:
                    st.error(
                        "FastAPI server is not running. Start it with: "
                        "python -m uvicorn app.main:app --reload"
                    )
                    st.stop()
                except requests.exceptions.RequestException as e:
                    st.error(f"Batch prediction request failed: {e}")
                    st.stop()

            if response.status_code == 200:
                result = response.json()

                if result.get("status") == "success":
                    st.success(
                        f"Batch prediction completed successfully! "
                        f"Rows processed: {result.get('rows_processed')}"
                    )

                    results_df = pd.DataFrame(result.get("results", []))

                    st.subheader("Prediction Results")
                    st.dataframe(results_df)

                    csv_data = convert_df_to_csv(results_df)

                    st.download_button(
                        label="Download Prediction Results",
                        data=csv_data,
                        file_name=output_filename,
                        mime="text/csv"
                    )
                else:
                    st.error("Batch prediction failed.")
                    st.code(result)
            else:
                st.error(f"API request failed. Status code: {response.status_code}")
                st.code(response.text)
