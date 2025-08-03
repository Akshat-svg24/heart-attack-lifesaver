import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Heart Attack Risk Predictor")

# Input fields
age = st.slider("Age", 20, 100, 50)
sex = st.radio("Sex", ["Male", "Female"])
cp = st.slider("Chest Pain Type (0–3)", 0, 3, 1)
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.slider("Rest ECG (0–2)", 0, 2, 1)
thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.slider("ST depression", 0.0, 6.0, 1.0)
slope = st.slider("Slope of peak exercise ST segment", 0, 2, 1)
ca = st.slider("Major Vessels Colored", 0, 3, 0)
thal = st.slider("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible)", 1, 3, 2)

# Encode inputs
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Prepare input
user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Scale
scaled_input = scaler.transform(user_input)

# Predict
if st.button("Predict"):
    prob = model.predict_proba(scaled_input)[0][1]
    st.write(f"**Risk of Heart Attack: {round(prob * 100, 2)}%**")
    if prob > 0.75:
        st.error("⚠️ High Risk! Please consult a doctor immediately.")
    elif prob > 0.4:
        st.warning("⚠️ Moderate Risk. Consider a health check-up.")
    else:
        st.success("✅ Low Risk. Keep maintaining a healthy lifestyle.")
