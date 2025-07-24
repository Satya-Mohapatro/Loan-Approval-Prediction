import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.pkl')

st.title("🏦 Loan Approval Prediction App")

st.markdown("""
This app predicts whether a loan will be **Approved** or **Not Approved** based on applicant details using a tuned Decision Tree model.
""")

st.sidebar.header("Input Applicant Details")

def user_input_features():
    Gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    Married = st.sidebar.selectbox("Married", ['Yes', 'No'])
    Dependents = st.sidebar.selectbox("Dependents", ['0', '1', '2', '3+'])
    Education = st.sidebar.selectbox("Education", ['Graduate', 'Not Graduate'])
    Self_Employed = st.sidebar.selectbox("Self Employed", ['Yes', 'No'])
    ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
    CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", min_value=0, value=0)
    LoanAmount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0, value=150)
    Loan_Amount_Term = st.sidebar.selectbox("Loan Amount Term", [360, 120, 180, 240, 300, 480, 60, 84, 12])
    Credit_History = st.sidebar.selectbox("Credit History (1=Clear, 0=Default)", [1, 0])
    Property_Area = st.sidebar.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

    data = {
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area
    }
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("Applicant Input Details")
st.write(input_df)

if st.button("Predict Loan Approval"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Loan Approved with Confidence: {prediction_proba:.2f}")
    else:
        st.error(f"❌ Loan Not Approved with Confidence: {prediction_proba:.2f}")
