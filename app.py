import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)

# Load Model and Scaler
model = pickle.load(open("models/loan_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Title
st.title("🏦 Loan Approval Prediction System")
st.write("Enter the applicant details below to predict whether the loan will be approved.")
st.header("Applicant Details")

dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

income = st.number_input(
    "Annual Income",
    min_value=0,
    value=500000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=1000000
)

loan_term = st.number_input(
    "Loan Term",
    min_value=1,
    value=20
)

cibil_score = st.slider(
    "CIBIL Score",
    300,
    900,
    750
)

residential_assets = st.number_input(
    "Residential Assets Value",
    min_value=0,
    value=500000
)

commercial_assets = st.number_input(
    "Commercial Assets Value",
    min_value=0,
    value=300000
)

luxury_assets = st.number_input(
    "Luxury Assets Value",
    min_value=0,
    value=200000
)

bank_assets = st.number_input(
    "Bank Assets Value",
    min_value=0,
    value=400000
)
if st.button("Predict Loan Status"):

    # Convert text values to numbers
    education_value = 0 if education == "Graduate" else 1
    self_employed_value = 1 if self_employed == "Yes" else 0

    # Create input data
    input_data = [[
        dependents,
        education_value,
        self_employed_value,
        income,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets,
        commercial_assets,
        luxury_assets,
        bank_assets
    ]]

    # Scale the input
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 0:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")