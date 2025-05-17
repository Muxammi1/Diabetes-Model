
import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('diabetes_knn_model.pkl')
scaler = joblib.load('scaler.pkl')  # if you used a scaler

st.title("Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
features_scaled = scaler.transform(features)

if st.button("Predict"):
    prediction = model.predict(features_scaled)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.success(f"The model predicts: {result}")
