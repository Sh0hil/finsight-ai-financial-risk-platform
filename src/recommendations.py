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
        recommendation = "Reject loan application or request collateral/guarantor."
        decision = "Reject / Manual Review Required"
    elif risk_level == "Medium Risk":
        recommendation = "Approve only after manual verification, income proof check, and possible higher interest rate."
        decision = "Manual Review"
    else:
        recommendation = "Approve loan application with standard terms."
        decision = "Approve"

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
        action = "Block transaction immediately and freeze account temporarily."
        decision = "Block Transaction"
    elif risk_level == "High Risk":
        action = "Hold transaction and trigger identity verification or OTP."
        decision = "Manual Verification"
    elif risk_level == "Medium Risk":
        action = "Allow transaction after additional authentication."
        decision = "Step-up Authentication"
    else:
        action = "Allow transaction normally."
        decision = "Allow Transaction"

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