import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# 1. Load Pipeline
# =========================
with open("pipeline.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

# =========================
# 2. UI
# =========================
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation Prediction")
st.markdown("Masukkan data pelanggan untuk memprediksi segmen (0, 1, 2, 3).")

# =========================
# 3. Input User dengan layout kolom
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", min_value=0, max_value=100, value=30)
    Ever_Married = st.selectbox("Ever Married", ["Yes", "No"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Graduated = st.selectbox("Graduated", ["Yes", "No"])

with col2:
    Profession = st.selectbox("Profession", [
        "Artist", "Doctor", "Engineer", "Entertainment", "Executive", "Healthcare",
        "Homemaker", "Lawyer", "Marketing"
    ])
    Work_Experience = st.number_input("Work Experience (years)", min_value=0, max_value=50, value=5)
    Spending_Score = st.number_input("Spending Score (0-100)", min_value=0, max_value=100, value=50)

with col3:
    Family_Size = st.number_input("Family Size", min_value=1, max_value=20, value=3)
    Var_1 = st.selectbox("Var_1", ["Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Cat_6"])

# =========================
# 4. Buat DataFrame input
# =========================
input_df = pd.DataFrame([{
    "Age": Age,
    "Ever_Married": Ever_Married,
    "Gender": Gender,
    "Graduated": Graduated,
    "Profession": Profession,
    "Work_Experience": Work_Experience,
    "Spending_Score": Spending_Score,
    "Family_Size": Family_Size,
    "Var_1": Var_1
}])

# =========================
# 5. Prediksi & tampilkan hasil
# =========================
if st.button("Prediksi"):
    prediction_proba = model_pipeline.predict_proba(input_df)
    prediction_class = np.argmax(prediction_proba, axis=1)[0]

    st.subheader("Hasil Prediksi Segmentasi")
    st.markdown(f"**Segmentasi Pelanggan:** {prediction_class}")

    st.subheader("Probabilitas Tiap Kelas")
    proba_df = pd.DataFrame(
        prediction_proba,
        columns=[f"Segment {i}" for i in range(prediction_proba.shape[1])]
    )
    st.dataframe(proba_df.style.format("{:.2f}"))
