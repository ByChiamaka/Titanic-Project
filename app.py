import streamlit as st
import joblib
import numpy as np
import os

# Page Config
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

# Load Model
model_path = os.path.join('model', 'titanic_survival_model.pkl')

@st.cache_resource
def load_model():
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'titanic_survival_model.pkl' is in the 'model' folder.")
        return None

model = load_model()

# Title and Header
st.title("🚢 Titanic Survival Prediction System")
st.write("Enter passenger details to predict if they would have survived the disaster.")
st.write("---")

# Input Form
with st.form("survival_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], format_func=lambda x: f"{x}st Class" if x==1 else (f"{x}nd Class" if x==2 else f"{x}rd Class"))
        sex = st.selectbox("Sex", ["Male", "Female"])
        age = st.number_input("Age", min_value=1, max_value=100, value=25)
    
    with col2:
        fare = st.number_input("Ticket Fare ($)", min_value=0.0, value=32.0)
        embarked = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
        
    submitted = st.form_submit_button("Predict Survival")

# Prediction Logic
if submitted and model:
    # 1. Encode Inputs (Must match training encoding exactly!)
    sex_encoded = 0 if sex == "Male" else 1
    
    embarked_map = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}
    embarked_encoded = embarked_map[embarked]
    
    # 2. Create Input Array
    # Order: ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
    input_data = np.array([[pclass, sex_encoded, age, fare, embarked_encoded]])
    
    # 3. Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # 4. Display Result
    st.write("---")
    if prediction == 1:
        st.success(f"**Prediction: SURVIVED** (Confidence: {probability:.2%})")
        st.balloons()
    else:
        st.error(f"**Prediction: DID NOT SURVIVE** (Confidence: {1-probability:.2%})")