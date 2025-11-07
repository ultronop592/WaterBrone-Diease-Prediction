# 💧 Waterborne Disease Prediction using Deep Learning

## 🧠 Overview
This project uses **Deep Learning (Bi-directional LSTM)** to predict **waterborne diseases** based on a patient's **medical report and lab test results**.  
The model analyzes clinical indicators such as **WBC count, sodium levels, bilirubin, hemoglobin, and other lab parameters**, then predicts the likelihood of diseases like:

- Cholera  
- Typhoid  
- Hepatitis A  
- Giardiasis  
- Dysentery  
- E. Coli Infection  
- Cryptosporidiosis  
- Shigellosis  
- (and Healthy cases)

The goal is to build an **AI-powered diagnostic support system** that can assist healthcare providers and patients in early disease detection.

---

## 🚀 Features
✅ Predicts 8+ common waterborne diseases  
✅ Uses realistic, human-like lab data  
✅ Built with Bi-directional LSTM deep learning model  
✅ Trained on **10,000+ synthetic but realistic patient samples**  
✅ Accepts simple text/lab report inputs  
✅ Ready for integration with **Streamlit UI** or **Flask API**  
✅ Realistic lab result simulation (WBC, sodium, bilirubin, etc.)

---

## 🧩 Dataset
The dataset consists of **10,000 records** generated using synthetic medical data that mimics real-world lab test reports.  
Each record includes:
- Patient lab parameters (WBC, RBC, Sodium, etc.)
- Short textual symptoms
- Target disease label

👉 You can retrain or extend the dataset for additional diseases.

---

## 🧱 Model Architecture
- **Model Type:** Bi-directional LSTM  
- **Framework:** TensorFlow / Keras  
- **Input:** Text + numerical lab values  
- **Output:** Predicted disease class (one of 8 diseases or healthy)

### 🔧 Layers:
1. Embedding Layer  
2. Bidirectional LSTM Layer  
3. Dense Layers with ReLU  
4. Output Layer with Softmax Activation  

---

## 🧪 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/Waterborne-Disease-Prediction.git
cd Waterborne-Disease-Prediction
