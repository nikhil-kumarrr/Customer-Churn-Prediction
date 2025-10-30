# 🎯 Customer Churn Prediction App - Professional Version
import streamlit as st
import pickle
import numpy as np

# --- Load Trained Model ---
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# --- Page Configuration ---
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Prediction System")
st.write(
    """
    This app predicts whether a **customer is likely to churn** based on their details.
    Fill out the information below and click **Predict** to see the result.
    """
)

st.divider()

# --- User Input Section ---
st.subheader("🔍 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

st.divider()

# --- Preprocessing for Prediction ---
# Note: You’ll need to modify encoding/scaling as per your training dataset
gender_val = 1 if gender == "Male" else 0
internet_service_val = {"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service]
contract_val = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
payment_val = {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}[payment_method]

features = np.array([[age, gender_val, tenure, internet_service_val, monthly_charges, total_charges, contract_val, payment_val]])

# --- Prediction Button ---
if st.button("🚀 Predict"):
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.error("🚨 The customer is **likely to churn.** Take retention action soon!", icon="⚠️")
    else:
        st.success("✅ The customer is **not likely to churn.**", icon="✅")

# --- Footer ---
st.markdown(
    """
    <hr>
    <center>Developed with ❤️ by <b>Nikhil Kumar</b> | Machine Learning Engineer</center>
    """,
    unsafe_allow_html=True
)
