from pydantic import BaseModel


class CreditRiskInput(BaseModel):
    NAME_CONTRACT_TYPE: str
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    OCCUPATION_TYPE: str
    CNT_FAM_MEMBERS: float
    CREDIT_INCOME_RATIO: float
    ANNUITY_INCOME_RATIO: float
    CREDIT_TERM: float
    DAYS_EMPLOYED_RATIO: float


class FraudInput(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


class SegmentInput(BaseModel):
    total_transactions: int
    total_amount: float
    avg_amount: float
    max_amount: float
    min_amount: float
    #total_fraud_transactions: int
    #fraud_ratio: float
    avg_old_balance: float
    avg_new_balance: float
    max_old_balance: float
    max_new_balance: float
    CASH_IN: int
    CASH_OUT: int
    DEBIT: int
    PAYMENT: int
    TRANSFER: int