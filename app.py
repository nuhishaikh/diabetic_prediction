import streamlit as st
import pandas as pd
import numpy as np
# from pickle import load # <-- REMOVE pickle
import joblib # <-- USE joblib instead
import os 

# --- Model Loading and Prediction Logic ---

# 1. Load the Saved Model
MODEL_FILE = 'finalized_model.sav'

if not os.path.exists(MODEL_FILE):
    st.error(f"Error: '{MODEL_FILE}' not found. Please ensure the trained model file is committed to the repository.")
    st.stop()

@st.cache_resource
def load_model():
    """Loads the model from disk using joblib for sklearn compatibility."""
    # Load model using joblib.load
    loaded_model = joblib.load(MODEL_FILE) 
    return loaded_model

loaded_model = load_model()

# 2. Configure the Streamlit App
st.set_page_config(page_title="Diabetic Prediction App", layout="wide")
st.title('Diabetes Prediction 🩺')
st.markdown("""
    This app uses a **Logistic Regression** model to predict whether a patient is diabetic based on input features.
""")

# 3. Create Input Fields for User Data
st.sidebar.header('Patient Input Features')

# Define the input features and their typical ranges
feature_info = {
    'Pregnancies': {'min': 0, 'max': 17, 'step': 1, 'default': 3},
    'Glucose': {'min': 0, 'max': 200, 'step': 1, 'default': 120},
    'BloodPressure': {'min': 0, 'max': 122, 'step': 1, 'default': 70},
    'SkinThickness': {'min': 0, 'max': 99, 'step': 1, 'default': 20},
    'Insulin': {'min': 0, 'max': 846, 'step': 10, 'default': 80},
    'BMI': {'min': 0.0, 'max': 67.1, 'step': 0.1, 'default': 32.0},
    'DiabetesPedigreeFunction': {'min': 0.078, 'max': 2.42, 'step': 0.001, 'default': 0.47},
    'Age': {'min': 21, 'max': 81, 'step': 1, 'default': 33},
}

def get_user_input():
    data = {}
    for feature, info in feature_info.items():
        if feature in ['BMI', 'DiabetesPedigreeFunction']:
            data[feature] = st.sidebar.slider(feature, float(info['min']), float(info['max']), float(info['default']), float(info['step']))
        else:
            data[feature] = st.sidebar.slider(feature, int(info['min']), int(info['max']), int(info['default']), int(info['step']))
    return pd.DataFrame(data, index=[0])

# Get and display user input
df_input = get_user_input()
st.subheader('User Input Data')
st.dataframe(df_input)

# 4. Perform Prediction
if st.sidebar.button('Predict Diabetes'):
    prediction = loaded_model.predict(df_input.values)
    prediction_proba = loaded_model.predict_proba(df_input.values)

    st.subheader('Prediction Result')
    
    if prediction[0] == 1:
        st.error('Prediction: **Diabetic** 🚨')
    else:
        st.success('Prediction: **Non-Diabetic** ✅')
        
    st.markdown("---")
    st.subheader('Confidence')
    st.write(f"Probability of being Non-Diabetic (Class 0): **{prediction_proba[0][0]:.2f}**")
    st.write(f"Probability of being Diabetic (Class 1): **{prediction_proba[0][1]:.2f}**")
