from io import StringIO

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schema import CreditRiskInput, FraudInput, SegmentInput
from src.predict_credit import predict_credit_risk
from src.predict_fraud import predict_fraud
from src.predict_segment import predict_customer_segment
from src.batch_predict import batch_predict_credit, batch_predict_fraud


app = FastAPI(
    title="FinSight AI API",
    description="End-to-End Financial Risk Intelligence Platform using Machine Learning",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to FinSight AI API",
        "status": "running",
        "available_endpoints": [
            "/health",
            "/predict-credit-risk",
            "/predict-fraud",
            "/predict-segment",
            "/batch-predict-credit-risk",
            "/batch-predict-fraud"
        ]
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "FinSight AI API"
    }


@app.post("/predict-credit-risk")
def credit_risk_prediction(input_data: CreditRiskInput):
    data = input_data.model_dump()
    return predict_credit_risk(data)


@app.post("/predict-fraud")
def fraud_prediction(input_data: FraudInput):
    data = input_data.model_dump()
    return predict_fraud(data)


@app.post("/predict-segment")
def segment_prediction(input_data: SegmentInput):
    data = input_data.model_dump()
    return predict_customer_segment(data)


@app.post("/batch-predict-credit-risk")
async def batch_credit_prediction(
    file: UploadFile = File(...),
    max_rows: int = 100
):
    try:
        contents = await file.read()
        decoded = contents.decode("utf-8")

        df = pd.read_csv(StringIO(decoded))
        df = df.head(max_rows)

        result_df = batch_predict_credit(df)

        return {
            "status": "success",
            "prediction_type": "credit_risk",
            "rows_processed": len(result_df),
            "results": result_df.to_dict(orient="records")
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "prediction_type": "credit_risk",
                "message": str(e)
            }
        )


@app.post("/batch-predict-fraud")
async def batch_fraud_prediction(
    file: UploadFile = File(...),
    max_rows: int = 100
):
    try:
        contents = await file.read()
        decoded = contents.decode("utf-8")

        df = pd.read_csv(StringIO(decoded))
        df = df.head(max_rows)

        result_df = batch_predict_fraud(df)

        return {
            "status": "success",
            "prediction_type": "fraud_detection",
            "rows_processed": len(result_df),
            "results": result_df.to_dict(orient="records")
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "prediction_type": "fraud_detection",
                "message": str(e)
            }
        )