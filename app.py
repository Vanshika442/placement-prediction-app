import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("logistic_pipe_model.joblib")

# Set page config
st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎓 Campus Placement Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Enter student details to predict whether they will be placed or not</h4><br>", unsafe_allow_html=True)

# Create columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ['M', 'F'])
    ssc_p = st.number_input("10th Percentage", min_value=0.0, max_value=100.0)
    ssc_b = st.selectbox("SSC Board", ['Central', 'Others'])
    hsc_p = st.number_input("12th Percentage", min_value=0.0, max_value=100.0)
    hsc_b = st.selectbox("HSC Board", ['Central', 'Others'])
    
with col2:
    hsc_s = st.selectbox("12th Stream", ['Commerce', 'Science', 'Arts'])
    degree_p = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0)
    degree_t = st.selectbox("Degree Stream", ['Sci&Tech', 'Comm&Mgmt', 'Others'])
    workex = st.selectbox("Work Experience", ['Yes', 'No'])
    specialisation = st.selectbox("MBA Specialisation", ['Mkt&HR', 'Mkt&Fin'])
    mba_p = st.number_input("MBA Percentage", min_value=0.0, max_value=100.0)

# Predict button
if st.button("🎯 Predict Placement"):
    # Create dataframe
    input_df = pd.DataFrame([[
        gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s,
        degree_p, degree_t, workex, specialisation, mba_p
    ]], columns=[
        'Gender', '10th %', 'SSC Board', '12th %', 'HSC Board',
        '12th Stream', 'Degree %', 'Degree stream', 'Work exp',
        'specialisation', 'Mba %'
    ])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Show result
    st.subheader("📢 Prediction Result:")
    if prediction == 1:
        st.success("🎉 Congratulations! The student is likely to be **Placed**.")
    else:
        st.error("❌ The student is **Not Placed** based on the input data.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Made with ❤️ by Vanshika Mahant</div>", unsafe_allow_html=True)

