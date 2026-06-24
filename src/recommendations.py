def get_credit_risk_level(default_probability):
    if default_probability >= 0.70:
        return "High Risk"
    elif default_probability >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_credit_recommendation(default_probability):
    risk_level = get_credit_risk_level(default_probability)

    if risk_level == "High Risk":
        decision = "Reject / Manual Review Required"
        recommendation = "Reject loan application or request collateral/guarantor."
    elif risk_level == "Medium Risk":
        decision = "Manual Review"
        recommendation = "Approve only after manual verification, income proof check, and possible higher interest rate."
    else:
        decision = "Approve"
        recommendation = "Approve loan application with standard terms."

    return {
        "default_probability": round(float(default_probability), 4),
        "risk_level": risk_level,
        "decision": decision,
        "recommendation": recommendation
    }


def get_fraud_risk_level(fraud_probability):
    if fraud_probability >= 0.80:
        return "Critical Risk"
    elif fraud_probability >= 0.50:
        return "High Risk"
    elif fraud_probability >= 0.30:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_fraud_recommendation(fraud_probability):
    risk_level = get_fraud_risk_level(fraud_probability)

    if risk_level == "Critical Risk":
        decision = "Block Transaction"
        action = "Block transaction immediately and freeze account temporarily."
    elif risk_level == "High Risk":
        decision = "Manual Verification"
        action = "Hold transaction and trigger identity verification or OTP."
    elif risk_level == "Medium Risk":
        decision = "Step-up Authentication"
        action = "Allow transaction after additional authentication."
    else:
        decision = "Allow Transaction"
        action = "Allow transaction normally."

    return {
        "fraud_probability": round(float(fraud_probability), 4),
        "risk_level": risk_level,
        "decision": decision,
        "recommended_action": action
    }


def get_segment_recommendation(segment_name):
    segment_name = str(segment_name).lower()

    if "high value" in segment_name:
        return {
            "segment_action": "Offer premium products, credit card upgrade, or loyalty benefits.",
            "business_priority": "High"
        }

    elif "suspicious" in segment_name or "high risk" in segment_name:
        return {
            "segment_action": "Monitor account closely and apply stricter transaction checks.",
            "business_priority": "Critical"
        }

    elif "balance draining" in segment_name:
        return {
            "segment_action": "Trigger account safety alert and review abnormal fund movement.",
            "business_priority": "High"
        }

    elif "low activity" in segment_name:
        return {
            "segment_action": "Send engagement campaign or personalized financial offers.",
            "business_priority": "Medium"
        }

    else:
        return {
            "segment_action": "Continue normal customer monitoring.",
            "business_priority": "Normal"
        }

# --- THIS IS THE MISSING FUNCTION YOU NEEDED ---
def get_final_risk_summary(default_probability, fraud_probability, segment_name):
    """
    Aggregates all risk module recommendations into a single comprehensive summary.
    """
    segment_data = get_segment_recommendation(segment_name)
    
    return {
        "credit_risk": get_credit_recommendation(default_probability),
        "fraud_risk": get_fraud_recommendation(fraud_probability),
        "customer_segment": {
            "segment_name": segment_name,
            "segment_action": segment_data.get("segment_action"),
            "business_priority": segment_data.get("business_priority")
        }
    }