import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load the necessary files ---
try:
    # Adjust filenames if they are different
    model = joblib.load('../model/YALA5.pkl') 
    scaler = joblib.load('../model/scaler.pkl')
    pca = joblib.load('../model/pca.pkl')
except FileNotFoundError as e:
    st.error(f"Error: The required file '{e.filename}' was not found.")
    st.stop()

# --- Define the full list of expected columns after one-hot encoding ---
# This is the definitive list of columns your model was trained on.
expected_columns = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca',
                    'sex_Female', 'sex_Male',
                    'cp_asymptomatic', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
                    'fbs_False', 'fbs_True',
                    'restecg_lv hypertrophy', 'restecg_normal', 'restecg_st-t abnormality',
                    'exang_False', 'exang_True',
                    'slope_downsloping', 'slope_flat', 'slope_upsloping',
                    'thal_fixed defect', 'thal_normal', 'thal_reversable defect']

# Streamlit App
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Fill out the form below to predict your risk of heart disease.")

with st.expander("ℹ️ What do the inputs mean?"):
    st.markdown("""
    - **age**: Age in years  
    - **sex**: Biological sex (1 = male, 0 = female)  
    - **cp**: Chest pain type (1 = typical angina, 2 = atypical angina, etc.)
    - **trestbps**: Resting blood pressure (in mm Hg)
    - **chol**: Serum cholesterol (in mg/dl)
    - **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    - **restecg**: Resting electrocardiographic results
    - **thalach**: Maximum heart rate achieved
    - **exang**: Exercise induced angina (1 = yes; 0 = no)
    - **oldpeak**: ST depression induced by exercise
    - **slope**: Slope of the peak exercise ST segment
    - **ca**: Number of major vessels (0–3)
    - **thal**: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
    """)

# 🧾 Input Form
with st.form("heart_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
    restecg = st.selectbox("Resting ECG Result", ['normal', 'st-t abnormality', 'lv hypertrophy'])
    thalach = st.number_input("Max Heart Rate Achieved", value=150)
    exang = st.selectbox("Exercise Induced Angina", ["False", "True"])
    oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0, format="%.1f")
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ['upsloping', 'flat', 'downsloping'])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0.0, 1.0, 2.0, 3.0])
    thal = st.selectbox("Thalassemia", ['normal', 'fixed defect', 'reversable defect'])

    submitted = st.form_submit_button("Predict")

# 🧠 Prediction Logic
if submitted:
    # Create a DataFrame from the raw user inputs
    input_df = pd.DataFrame([[age, trestbps, chol, thalach, oldpeak, ca, sex, cp, fbs, restecg, exang, slope, thal]],
                            columns=['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'])

    # 1. One-hot encode the categorical features
    # : The one-hot encoding on the Streamlit inputs must be robust
    input_encoded = pd.get_dummies(input_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'])

    # 2. Align the columns with the expected feature names from training
    input_aligned = input_encoded.reindex(columns=expected_columns, fill_value=0)
    
    # 3. Scale and apply PCA to the aligned data
    input_scaled = scaler.transform(input_aligned)
    

    # 4. Make the prediction
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"🎯 Prediction: Heart Disease Type {prediction}")
    st.markdown("""
    **Note:**  
    - 0 = No disease  
    - 1–4 = Different heart disease levels/types (as per dataset)
    """)

# Footer
st.markdown("---")
st.caption("Developed as part of the AI/ML Summer Course Graduation Project")