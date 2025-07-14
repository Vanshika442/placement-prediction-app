import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model
model = joblib.load("logistic_pipe_model.joblib")

st.title("Campus Placement Prediction")
st.write("Enter student academic and personal details to predict placement status.")

# Input fields
gender = st.selectbox("Gender", ['M', 'F'])
ssc_p = st.number_input("10th Percentage", min_value=0.0, max_value=100.0)
ssc_b = st.selectbox("SSC Board", ['Central', 'Others'])

hsc_p = st.number_input("12th Percentage", min_value=0.0, max_value=100.0)
hsc_b = st.selectbox("HSC Board", ['Central', 'Others'])
hsc_s = st.selectbox("12th Stream", ['Commerce', 'Science', 'Arts'])

degree_p = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0)
degree_t = st.selectbox("Degree Stream", ['Sci&Tech', 'Comm&Mgmt', 'Others'])

workex = st.selectbox("Work Experience", ['Yes', 'No'])
specialisation = st.selectbox("MBA Specialisation", ['Mkt&HR', 'Mkt&Fin'])
mba_p = st.number_input("MBA Percentage", min_value=0.0, max_value=100.0)

# On predict
if st.button("Predict Placement"):
    input_df = pd.DataFrame([[
        gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s,
        degree_p, degree_t, workex, specialisation, mba_p
    ]], columns=[
        'Gender', '10th %', 'SSC Board', '12th %', 'HSC Board',
        '12th Stream', 'Degree %', 'Degree stream', 'Work exp',
        'specialisation', 'Mba %'
    ])

    pred = model.predict(input_df)[0]
    result = "🎉 Placed!" if pred == 1 else "❌ Not Placed"

    st.subheader("Prediction:")
    st.success(result)
