import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("LGBMClassifier.pkl")  # Pastikan file ada di root repo
    scaler = None  # Jika ada scaler, load di sini
    return model, scaler

model, scaler = load_model()

# =========================
# UI
# =========================
st.set_page_config(page_title="Prediksi Segmentasi", layout="centered")
st.title("🧩 Aplikasi Prediksi Segmentasi (LightGBM)")
st.write("Masukkan data pelanggan untuk mendapatkan prediksi segmen.")

# =========================
# Input Form
# =========================
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        umur = st.number_input("Umur", min_value=0, max_value=100, value=30)
        pendapatan = st.number_input("Pendapatan Tahunan (juta)", min_value=0, value=50)
        skor_belanja = st.number_input("Skor Belanja (0-100)", min_value=0, max_value=100, value=50)

    with col2:
        lama_langganan = st.number_input("Lama Berlangganan (tahun)", min_value=0, value=5)
        jumlah_transaksi = st.number_input("Jumlah Transaksi / Tahun", min_value=0, value=20)

    submit = st.form_submit_button("Prediksi")

# =========================
# Prediction Logic
# =========================
if submit:
    # Dataframe dari input
    input_data = pd.DataFrame([[
        umur,
        pendapatan,
        skor_belanja,
        lama_langganan,
        jumlah_transaksi
    ]], columns=["umur", "pendapatan", "skor_belanja", "lama_langganan", "jumlah_transaksi"])

    # Scaling jika ada
    if scaler is not None:
        input_scaled = scaler.transform(input_data)
    else:
        input_scaled = input_data

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    st.success(f"Prediksi Segmentasi: **{prediction}**")
    st.balloons()
