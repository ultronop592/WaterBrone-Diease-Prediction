import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np
import pandas as pd

# Must be the first Streamlit command
st.set_page_config(page_title="Waterborne Disease Predictor", layout="centered")


@st.cache_resource
def load_all():
    import os
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    model = load_model(os.path.join(script_dir, 'best_multimodal_model.h5'))
    tokenizer = joblib.load(os.path.join(script_dir, 'tokenizer.joblib'))
    scaler = joblib.load(os.path.join(script_dir, 'scaler.joblib'))
    label_encoder = joblib.load(os.path.join(script_dir, 'label_encoder.joblib'))
    return model, tokenizer, scaler, label_encoder


model, tokenizer, scaler, label_encoder = load_all()


st.title("💧 AI-Powered Waterborne Disease Predictor")
st.markdown("### Predict possible diseases from **symptoms** and **lab results** using Deep Learning.")


symptoms = st.text_area("🩺 Enter your symptoms:",
                        placeholder="Example: Fever, vomiting, and diarrhea for two days")

#
st.markdown("#### 🧪 Enter Lab Results and Patient Info")

lab_values = []
col1, col2 = st.columns(2)

with col1:
    lab_values.append(st.number_input("Sodium (mmol/L)", 0.0, 200.0, 135.0))
    lab_values.append(st.number_input("Potassium (mmol/L)", 0.0, 10.0, 4.1))
    lab_values.append(st.number_input("Chloride (mmol/L)", 0.0, 150.0, 98.0))
    lab_values.append(st.number_input("WBC Count (×10⁹/L)", 0.0, 100.0, 8.0))
    lab_values.append(st.number_input("Hemoglobin (g/dL)", 0.0, 20.0, 13.0))
    lab_values.append(st.number_input("Platelets (×10⁹/L)", 0.0, 1000.0, 250.0))
    lab_values.append(st.number_input("Urea (mg/dL)", 0.0, 100.0, 13.5))

with col2:
    lab_values.append(st.number_input("Creatinine (mg/dL)", 0.0, 5.0, 0.8))
    lab_values.append(st.number_input("Bilirubin (mg/dL)", 0.0, 5.0, 0.7))
    lab_values.append(st.number_input("ALT (U/L)", 0.0, 200.0, 25.0))
    lab_values.append(st.number_input("AST (U/L)", 0.0, 200.0, 28.0))
    lab_values.append(st.number_input("Age (years)", 0, 120, 35))
    lab_values.append(st.number_input("Hygiene Score (1-10)", 1, 10, 5))

# Categorical features
gender = st.selectbox("Gender", ["Male", "Female"])
water_source = st.selectbox("Water Source", ["Tap", "Well", "River", "Bottled"])

if st.button("🔍 Predict Disease"):
    if symptoms.strip() == "":
        st.warning("Please enter your symptoms before predicting.")
    else:
        # Preprocess text
        seq = tokenizer.texts_to_sequences([symptoms])
        seq_pad = pad_sequences(seq, maxlen=60, padding='post')

        # Scale only the 13 numerical features
        lab_values_array = np.array(lab_values).reshape(1, -1)
        lab_scaled = scaler.transform(lab_values_array)
        
        # One-hot encode categorical features (matching training)
        gender_male = 1 if gender == "Male" else 0
        water_bottled = 1 if water_source == "Bottled" else 0
        water_river = 1 if water_source == "River" else 0
        water_well = 1 if water_source == "Well" else 0
        
        # Combine scaled numerical + categorical features (13 + 4 = 17)
        cat_features = np.array([[gender_male, water_bottled, water_river, water_well]])
        final_features = np.hstack([lab_scaled, cat_features])

        # Predict - model expects list of inputs [text, tabular]
        pred = model.predict([seq_pad, final_features])
        disease = label_encoder.inverse_transform([np.argmax(pred)])

        st.success(f"🧬 **Predicted Disease:** {disease[0]}")
