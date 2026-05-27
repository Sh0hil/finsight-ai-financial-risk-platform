import numpy as np
import pandas as pd
import shap

from src.config import CREDIT_MODEL_PATH, FRAUD_MODEL_PATH
from src.utils import load_model
from src.predict_fraud import create_fraud_features


credit_model = load_model(CREDIT_MODEL_PATH)
fraud_model = load_model(FRAUD_MODEL_PATH)


def _get_pipeline_parts(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    return preprocessor, model


def _to_dense(data):
    if hasattr(data, "toarray"):
        return data.toarray()
    return data


def _get_feature_names(preprocessor, n_features):
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return np.array([f"feature_{i}" for i in range(n_features)])


def _extract_class_1_shap_values(shap_values):
    """
    Handles different SHAP output formats safely.
    Returns 1D SHAP values for class 1.
    """

    shap_values = np.array(shap_values)

    print("DEBUG SHAP shape:", shap_values.shape)

    # Case 1: shape = (samples, features)
    if shap_values.ndim == 2:
        return shap_values[0]

    # Case 2: shape = (samples, features, classes)
    if shap_values.ndim == 3:
        return shap_values[0, :, 1]

    # Case 3: shape = (classes, samples, features)
    if shap_values.ndim == 3 and shap_values.shape[0] == 2:
        return shap_values[1, 0, :]

    # Case 4: already 1D
    if shap_values.ndim == 1:
        return shap_values

    raise ValueError(f"Unsupported SHAP value shape: {shap_values.shape}")


def _create_explanation(feature_names, shap_values_single, top_n, risk_text):
    shap_values_single = np.ravel(shap_values_single)

    top_indices = np.argsort(np.abs(shap_values_single))[-top_n:][::-1]

    explanations = []

    for idx in top_indices:
        feature_name = str(feature_names[idx])
        shap_value = float(np.ravel(shap_values_single[idx])[0])

        impact = risk_text if shap_value > 0 else risk_text.replace("increases", "decreases")

        explanations.append({
            "feature": feature_name,
            "shap_value": round(shap_value, 4),
            "impact": impact
        })

    return explanations


def explain_credit_prediction(input_data: dict, top_n: int = 5):
    input_df = pd.DataFrame([input_data])

    preprocessor, model = _get_pipeline_parts(credit_model)

    X_transformed = preprocessor.transform(input_df)
    X_transformed = _to_dense(X_transformed)

    feature_names = _get_feature_names(preprocessor, X_transformed.shape[1])

    model_name = model.__class__.__name__.lower()

    if "randomforest" in model_name or "gradientboosting" in model_name or "xgb" in model_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

    elif "logisticregression" in model_name:
        explainer = shap.LinearExplainer(model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    else:
        explainer = shap.Explainer(model, X_transformed)
        shap_values = explainer(X_transformed).values

    shap_values_single = _extract_class_1_shap_values(shap_values)

    explanations = _create_explanation(
        feature_names=feature_names,
        shap_values_single=shap_values_single,
        top_n=top_n,
        risk_text="increases risk"
    )

    return {
        "model_type": model.__class__.__name__,
        "top_risk_factors": explanations
    }


def explain_fraud_prediction(input_data: dict, top_n: int = 5):
    processed_data = create_fraud_features(input_data)
    input_df = pd.DataFrame([processed_data])

    preprocessor, model = _get_pipeline_parts(fraud_model)

    X_transformed = preprocessor.transform(input_df)
    X_transformed = _to_dense(X_transformed)

    feature_names = _get_feature_names(preprocessor, X_transformed.shape[1])

    model_name = model.__class__.__name__.lower()

    if "randomforest" in model_name or "gradientboosting" in model_name or "xgb" in model_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

    elif "logisticregression" in model_name:
        explainer = shap.LinearExplainer(model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    else:
        explainer = shap.Explainer(model, X_transformed)
        shap_values = explainer(X_transformed).values

    shap_values_single = _extract_class_1_shap_values(shap_values)

    explanations = _create_explanation(
        feature_names=feature_names,
        shap_values_single=shap_values_single,
        top_n=top_n,
        risk_text="increases fraud risk"
    )

    return {
        "model_type": model.__class__.__name__,
        "top_fraud_factors": explanations
    }