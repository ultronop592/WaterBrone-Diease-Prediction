import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Must be the first Streamlit command
st.set_page_config(
    page_title="AI Disease Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)


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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #764ba2;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>💧 AI-Powered Waterborne Disease Predictor</h1>
    <p style="font-size: 18px; margin-top: 10px;">
        Advanced Deep Learning for Early Disease Detection
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/water.png", width=80)
    st.title("📊 About")
    
    st.markdown("""
    ### How it works:
    1. 🩺 Enter patient symptoms
    2. 🧪 Input laboratory results
    3. 🤖 AI analyzes the data
    4. 🎯 Get instant prediction
    
    ### Model Performance:
    - **Accuracy:** 97.83%
    - **Diseases Detected:** 9 types
    - **Technology:** BiLSTM Neural Network
    
    ### Detected Diseases:
    - Cholera
    - Typhoid
    - Hepatitis A
    - Giardiasis
    - Dysentery
    - E. Coli Infection
    - Cryptosporidiosis
    - Shigellosis
    - Healthy (No Disease)
    """)
    
    st.divider()
    st.markdown("### ⚠️ Important Notice")
    st.warning("""
    This is a **screening tool** only. 
    Always consult healthcare professionals 
    for final diagnosis and treatment.
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["🔬 Disease Prediction", "📈 Analytics", "ℹ️ Information"])

with tab1:
    st.markdown("### 🩺 Patient Symptoms")
    symptoms = st.text_area(
        "Describe the symptoms in detail:",
        placeholder="Example: Severe watery diarrhea for 2 days, vomiting, dehydration, weakness",
        height=100,
        help="Include duration, severity, and any specific symptoms"
    )

    
    st.markdown("### 🧪 Laboratory Results")
    
    # Create expandable sections for organized input
    with st.expander("⚡ Electrolytes & Blood Chemistry", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            sodium = st.number_input("Sodium (mmol/L)", 0.0, 200.0, 135.0,
                                    help="Normal: 135-145 mmol/L")
            potassium = st.number_input("Potassium (mmol/L)", 0.0, 10.0, 4.1,
                                       help="Normal: 3.5-5.0 mmol/L")
        with col2:
            chloride = st.number_input("Chloride (mmol/L)", 0.0, 150.0, 98.0,
                                      help="Normal: 96-106 mmol/L")
            wbc = st.number_input("WBC Count (×10⁹/L)", 0.0, 100.0, 8.0,
                                 help="Normal: 4.5-11.0 ×10⁹/L")
        with col3:
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 0.0, 20.0, 13.0,
                                        help="Normal: 12-16 g/dL")
            platelets = st.number_input("Platelets (×10⁹/L)", 0.0, 1000.0, 
                                       250.0, help="Normal: 150-400 ×10⁹/L")
    
    with st.expander("🔬 Kidney & Liver Function", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            urea = st.number_input("Urea (mg/dL)", 0.0, 100.0, 13.5,
                                  help="Normal: 7-20 mg/dL")
            creatinine = st.number_input("Creatinine (mg/dL)", 0.0, 5.0, 0.8,
                                        help="Normal: 0.6-1.2 mg/dL")
        with col2:
            bilirubin = st.number_input("Bilirubin (mg/dL)", 0.0, 5.0, 0.7,
                                       help="Normal: 0.1-1.2 mg/dL")
            alt = st.number_input("ALT (U/L)", 0.0, 200.0, 25.0,
                                 help="Normal: 7-55 U/L")
        with col3:
            ast = st.number_input("AST (U/L)", 0.0, 200.0, 28.0,
                                 help="Normal: 8-48 U/L")
    
    with st.expander("👤 Patient Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", 0, 120, 35)
            gender = st.selectbox("Gender", ["Male", "Female"],
                                 help="Patient's biological gender")
        with col2:
            hygiene = st.slider("Hygiene Score", 1, 10, 5,
                               help="1=Poor, 10=Excellent")
            water_source = st.selectbox("Water Source", 
                                       ["Tap", "Well", "River", "Bottled"],
                                       help="Primary source of drinking water")
    
    # Collect all lab values in order
    lab_values = [sodium, potassium, chloride, wbc, hemoglobin, platelets,
                  urea, creatinine, bilirubin, alt, ast, age, hygiene]
    
    st.divider()
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 Analyze & Predict Disease", 
                                   use_container_width=True)
    
    if predict_button:
        if symptoms.strip() == "":
            st.error("⚠️ Please enter patient symptoms before predicting.")
        else:
            with st.spinner("🔄 Analyzing patient data..."):
                # Preprocess text
                seq = tokenizer.texts_to_sequences([symptoms])
                seq_pad = pad_sequences(seq, maxlen=60, padding='post')

                # Scale only the 13 numerical features
                lab_values_array = np.array(lab_values).reshape(1, -1)
                lab_scaled = scaler.transform(lab_values_array)
                
                # One-hot encode categorical features
                gender_male = 1 if gender == "Male" else 0
                water_bottled = 1 if water_source == "Bottled" else 0
                water_river = 1 if water_source == "River" else 0
                water_well = 1 if water_source == "Well" else 0
                
                # Combine features
                cat_features = np.array([[gender_male, water_bottled, 
                                        water_river, water_well]])
                final_features = np.hstack([lab_scaled, cat_features])

                # Predict
                pred = model.predict([seq_pad, final_features], verbose=0)
                disease_idx = np.argmax(pred)
                disease = label_encoder.inverse_transform([disease_idx])[0]
                confidence = pred[0][disease_idx] * 100
                
            # Display results with animation
            st.success("✅ Analysis Complete!")
            
            # Create result card
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 2rem; border-radius: 15px; color: white; 
                        text-align: center; margin: 2rem 0;">
                <h2 style="margin: 0;">Predicted Disease</h2>
                <h1 style="font-size: 3rem; margin: 1rem 0;">{disease}</h1>
                <p style="font-size: 1.5rem; margin: 0;">
                    Confidence: {confidence:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show all probabilities
            st.markdown("### 📊 Detailed Probability Analysis")
            prob_data = []
            for i, prob in enumerate(pred[0]):
                disease_name = label_encoder.inverse_transform([i])[0]
                prob_data.append({
                    "Disease": disease_name,
                    "Probability": f"{prob*100:.2f}%",
                    "Score": prob
                })
            
            prob_df = pd.DataFrame(prob_data)
            prob_df = prob_df.sort_values("Score", ascending=False)
            
            # Display as styled table
            for idx, row in prob_df.iterrows():
                col1, col2, col3 = st.columns([3, 1, 2])
                with col1:
                    st.write(f"**{row['Disease']}**")
                with col2:
                    st.write(row['Probability'])
                with col3:
                    st.progress(row['Score'])
            
            # Medical disclaimer
            st.divider()
            st.info("""
            **⚕️ Medical Disclaimer:**  
            This prediction is generated by an AI model and should be used as a 
            screening tool only. Always consult qualified healthcare professionals 
            for proper diagnosis and treatment.
            """)

with tab2:
    st.markdown("### 📈 Model Performance Analytics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", "97.83%", "↑ 2.5%")
    with col2:
        st.metric("Diseases Detected", "9", "")
    with col3:
        st.metric("Avg Confidence", "96.2%", "↑ 1.2%")
    
    st.markdown("---")
    st.markdown("""
    ### 🎯 Model Architecture
    
    Our system uses a **Multimodal Bidirectional LSTM** neural network that:
    - Processes patient symptoms using Natural Language Processing
    - Analyzes 17 clinical and demographic features
    - Achieves 97.83% validation accuracy
    - Provides real-time predictions in under 1 second
    
    **Training Data:** 10,000 patient records  
    **Validation Method:** 80-20 train-test split with stratification  
    **Technology Stack:** TensorFlow, Keras, Scikit-learn
    """)

with tab3:
    st.markdown("### ℹ️ About Waterborne Diseases")
    
    disease_info = {
        "Cholera": {
            "description": "Acute diarrheal infection caused by Vibrio cholerae",
            "symptoms": "Severe watery diarrhea, dehydration, vomiting",
            "severity": "High - Can be fatal if untreated"
        },
        "Typhoid": {
            "description": "Bacterial infection caused by Salmonella typhi",
            "symptoms": "Sustained fever, headache, abdominal pain, weakness",
            "severity": "Moderate to High"
        },
        "Dysentery": {
            "description": "Inflammatory disease of the intestine",
            "symptoms": "Bloody diarrhea, abdominal cramps, fever",
            "severity": "Moderate"
        },
        "Hepatitis A": {
            "description": "Viral liver infection",
            "symptoms": "Jaundice, fatigue, abdominal pain, nausea",
            "severity": "Moderate"
        },
        "Giardiasis": {
            "description": "Parasitic infection of the small intestine",
            "symptoms": "Diarrhea, gas, greasy stools, stomach cramps",
            "severity": "Low to Moderate"
        },
        "E. Coli Infection": {
            "description": "Bacterial infection caused by Escherichia coli",
            "symptoms": "Severe stomach cramps, diarrhea (often bloody), vomiting",
            "severity": "Moderate to High"
        },
        "Cryptosporidiosis": {
            "description": "Parasitic disease caused by Cryptosporidium",
            "symptoms": "Watery diarrhea, stomach cramps, nausea, dehydration",
            "severity": "Moderate"
        },
        "Shigellosis": {
            "description": "Bacterial infection caused by Shigella bacteria",
            "symptoms": "Diarrhea (often bloody), fever, stomach cramps",
            "severity": "Moderate"
        },
        "Healthy": {
            "description": "No waterborne disease detected",
            "symptoms": "Normal health indicators, no concerning symptoms",
            "severity": "None - Healthy status"
        }
    }
    
    for disease, info in disease_info.items():
        with st.expander(f"🦠 {disease}"):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Common Symptoms:** {info['symptoms']}")
            st.markdown(f"**Severity:** {info['severity']}")
    
    st.markdown("---")
    st.markdown("""
    ### 🚨 When to Seek Immediate Medical Attention
    
    Seek emergency care if experiencing:
    - Severe dehydration (excessive thirst, little/no urination)
    - High fever (>103°F / 39.4°C)
    - Bloody stools or vomit
    - Severe abdominal pain
    - Confusion or altered mental state
    - Signs of shock (rapid pulse, cold skin, dizziness)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Developed For better Tretment  | © 2025</p>
    <p style="font-size: 0.9rem;">
        <strong>Disclaimer:</strong> This tool is for educational and screening 
        purposes only. Always consult healthcare professionals.
    </p>
</div>
""", unsafe_allow_html=True)
