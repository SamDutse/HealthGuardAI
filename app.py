import os
import streamlit as st
import numpy as np
import joblib

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🏥",
    layout="wide"
)

# -------------------------
# Custom Styling
# -------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #0e4d92;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #08376b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
model = joblib.load("healthguard_model.pkl")

# -------------------------
# Header
# -------------------------
st.title("🏥 HealthGuard AI")
st.markdown("### Community Maternal High-Risk Pregnancy Predictor")
st.markdown(
    "This AI-powered tool predicts the likelihood of a **high-risk pregnancy** "
    "based on key maternal health indicators."
)

st.divider()

# -------------------------
# Input Section (2 Columns)
# -------------------------
st.subheader("🧾 Enter Patient Information")

left_col, right_col = st.columns(2)

with left_col:
    age = st.number_input("Age", 16, 50, 25)
    parity = st.number_input("Parity (Previous Births)", 0, 10, 1)
    anc_visits = st.number_input("Number of ANC Visits", 0, 10, 4)
    malaria = st.selectbox("Malaria During Pregnancy?", ["No", "Yes"])
    hemoglobin = st.slider("Hemoglobin Level (g/dL)", 7.0, 14.0, 11.0)

with right_col:
    nutrition_score = st.slider("Nutrition Score (1 = Poor, 5 = Excellent)", 1, 5, 3)
    distance_km = st.slider("Distance to Health Facility (km)", 1.0, 30.0, 5.0)
    prev_complication = st.selectbox("Previous Pregnancy Complication?", ["No", "Yes"])
    education_level = st.selectbox("Education Level", ["None", "Primary", "Secondary+"])

st.divider()

# -------------------------
# Preprocessing
# -------------------------
malaria = 1 if malaria == "Yes" else 0
prev_complication = 1 if prev_complication == "Yes" else 0

education_mapping = {
    "None": 0,
    "Primary": 1,
    "Secondary+": 2
}

education_level = education_mapping[education_level]

input_data = np.array([[age, parity, anc_visits, malaria,
                        hemoglobin, nutrition_score,
                        distance_km, prev_complication,
                        education_level]])

# -------------------------
# Prediction Button
# -------------------------
predict = st.button("🔍 Predict Risk Level")

# -------------------------
# Prediction Result (Immediately Visible)
# -------------------------
if predict:
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.divider()
    st.subheader("📊 Prediction Result")

    st.metric("High-Risk Probability", f"{probability:.2f}%")

    if prediction == 1:
        st.error("⚠ HIGH RISK Pregnancy Detected")

        st.markdown("""
        **Recommended Actions**
        - Immediate referral to skilled obstetric care  
        - Increase ANC monitoring frequency  
        - Manage anemia and malaria risk  
        - Close follow-up by community health worker  
        """)

    else:
        st.success("✅ NOT HIGH RISK")

        st.markdown("""
        **Recommended Actions**
        - Continue routine ANC schedule  
        - Maintain good nutrition  
        - Monitor for emerging symptoms  
        """)

    st.caption("HealthGuard AI supports — not replaces — clinical decision-making.")
