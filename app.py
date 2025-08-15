import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# 1. Load Model & Scaler
# =========================
@st.cache_resource
def load_model():
    with open("lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)

    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        scaler = None

    return model, scaler

model, scaler = load_model()

# =========================
# 2. Feature Names
# =========================
FEATURE_NAMES = [
    "Age",
    "Ever_Married",
    "Gender",
    "Graduated",
    "Profession",
    "Work_Experience",
    "Spending_Score",
    "Family_Size",
    "Var_1"
]

# =========================
# 3. UI
# =========================
st.title("Customer Segmentation Prediction")
st.write("Masukkan data pelanggan untuk memprediksi segmen (0, 1, 2, 3).")

Age = st.number_input("Age", min_value=0, max_value=100, value=30)
Ever_Married = st.selectbox("Ever Married", ["Yes", "No"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Graduated = st.selectbox("Graduated", ["Yes", "No"])
Profession = st.selectbox("Profession", [
    "Artist", "Doctor", "Engineer", "Entertainment", "Executive", "Healthcare",
    "Homemaker", "Lawyer", "Marketing"
])
Work_Experience = st.number_input("Work Experience (years)", min_value=0, max_value=50, value=5)
Spending_Score = st.number_input("Spending Score (0-100)", min_value=0, max_value=100, value=50)
Family_Size = st.number_input("Family Size", min_value=1, max_value=20, value=3)
Var_1 = st.selectbox("Var_1", ["Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Cat_6"])

# =========================
# 4. Encode Input
# =========================
def encode_input():
    data = {
        "Age": Age,
        "Ever_Married": 1 if Ever_Married == "Yes" else 0,
        "Gender": 1 if Gender == "Male" else 0,
        "Graduated": 1 if Graduated == "Yes" else 0,
        "Profession": [
            "Artist", "Doctor", "Engineer", "Entertainment", "Executive", "Healthcare",
            "Homemaker", "Lawyer", "Marketing"
        ].index(Profession),
        "Work_Experience": Work_Experience,
        "Spending_Score": Spending_Score,
        "Family_Size": Family_Size,
        "Var_1": ["Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Cat_6"].index(Var_1)
    }
    df = pd.DataFrame([data], columns=FEATURE_NAMES)

    # scaling jika ada
    if scaler is not None:
        df = pd.DataFrame(scaler.transform(df), columns=FEATURE_NAMES)

    return df

# =========================
# 5. Predict
# =========================
if st.button("Prediksi"):
    input_df = encode_input()
    prediction_proba = model.predict(input_df)
    prediction_class = np.argmax(prediction_proba, axis=1)[0]

    st.subheader("Hasil Prediksi")
    st.write(f"Segmentasi Pelanggan: **{prediction_class}**")
    st.write("Probabilitas tiap kelas:")
    st.write(prediction_proba)
