from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load model
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


class LoanApplication(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


@app.get("/")
def home():
    return {"message": "Loan Approval Prediction API is running!"}


@app.get("/model-status")
def model_status():
    return {
        "model": "Loaded successfully",
        "scaler": "Loaded successfully"
    }


@app.post("/predict")
def predict_loan(data: LoanApplication):

    # Convert categorical values to numbers
    education = 1 if data.education == "Graduate" else 0
    self_employed = 1 if data.self_employed == "Yes" else 0

    # Create input DataFrame in exact training order
    input_data = pd.DataFrame([[
        data.no_of_dependents,
        education,
        self_employed,
        data.income_annum,
        data.loan_amount,
        data.loan_term,
        data.cibil_score,
        data.residential_assets_value,
        data.commercial_assets_value,
        data.luxury_assets_value,
        data.bank_asset_value
    ]], columns=[
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ])

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_data)[0]

    if prediction == 0:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    return {
        "prediction": int(prediction),
        "result": result
    }