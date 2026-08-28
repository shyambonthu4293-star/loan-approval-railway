import streamlit as st
import pandas as pd
import pickle

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="LoanPredictor",
    page_icon="🏦",
    layout="wide"
)

# -------------------------------------------------
# LOAD MODEL AND SCALER
# -------------------------------------------------
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Main title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #666;
    margin-bottom: 35px;
}

/* Section headings */
.section-title {
    font-size: 24px;
    font-weight: 650;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Cards */
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Predict button */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 10px;
    font-size: 19px;
    font-weight: 600;
}

/* Result */
.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 25px;
    font-weight: 700;
    margin-top: 25px;
}

/* Footer */
.footer {
    text-align: center;
    color: #777;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    '<div class="title">🏦 LoanPredictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Smart Loan Approval Prediction System</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# APPLICANT INFORMATION
# -------------------------------------------------
st.markdown(
    '<div class="section-title">👤 Applicant Information</div>',
    unsafe_allow_html=True
)

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10,
            value=0
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

    with col2:
        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )

        cibil_score = st.slider(
            "CIBIL Score",
            min_value=300,
            max_value=900,
            value=750
        )

# -------------------------------------------------
# FINANCIAL INFORMATION
# -------------------------------------------------
st.markdown(
    '<div class="section-title">💰 Financial Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    income = st.number_input(
        "Annual Income",
        min_value=0,
        value=500000,
        step=10000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=1000000,
        step=10000
    )

    loan_term = st.number_input(
        "Loan Term (Years)",
        min_value=1,
        value=20
    )

with col2:

    residential_assets = st.number_input(
        "Residential Assets Value",
        min_value=0,
        value=500000,
        step=10000
    )

    commercial_assets = st.number_input(
        "Commercial Assets Value",
        min_value=0,
        value=300000,
        step=10000
    )

    luxury_assets = st.number_input(
        "Luxury Assets Value",
        min_value=0,
        value=200000,
        step=10000
    )

bank_assets = st.number_input(
    "Bank Assets Value",
    min_value=0,
    value=400000,
    step=10000
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
st.markdown("---")

if st.button("🔍 Predict Loan Status"):

    # Convert categorical values
    education_value = 0 if education == "Graduate" else 1
    self_employed_value = 1 if self_employed == "Yes" else 0

    # Create input
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

    # Scale input
    input_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 0:

        st.markdown(
            """
            <div class="result-box">
                ✅ LOAN APPROVED
                <br>
                <span style="font-size:16px;font-weight:400;">
                Congratulations! The applicant is eligible for the loan.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="result-box">
                ❌ LOAN REJECTED
                <br>
                <span style="font-size:16px;font-weight:400;">
                Based on the provided information, the applicant is not eligible.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    """
    <div class="footer">
        LoanPredictor • Machine Learning Based Loan Eligibility System
    </div>
    """,
    unsafe_allow_html=True
)
