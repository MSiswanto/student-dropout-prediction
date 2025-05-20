import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("dropout_model.pkl")

# Judul Aplikasi
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", page_icon="🎓")
st.title("🎓 Prediksi Dropout Mahasiswa")
st.markdown("""
Aplikasi ini memprediksi apakah seorang mahasiswa berisiko Dropout, Masih Enrolled, atau Graduate.
Silakan masukkan data di bawah ini untuk melihat prediksi:
""")

# Sidebar untuk input
st.sidebar.header("Input Data Mahasiswa")

umur = st.sidebar.slider("Umur saat mendaftar", 17, 50, 20)
nilai_masuk = st.sidebar.slider("Nilai masuk", 0.00, 200.00, 120.0)
rata_rata_smt1 = st.sidebar.slider("Rata-rata nilai semester 1", 0.00, 20.00, 12.0)
rata_rata_smt2 = st.sidebar.slider("Rata-rata nilai semester 2", 0.00, 20.00, 12.0)
jumlah_mk_tdk_lulus = st.sidebar.slider("Jumlah MK tidak lulus", 0, 20, 3)

# Prediksi
if st.button("🔍 Prediksi"):
    input_data = np.array([[umur, nilai_masuk, rata_rata_smt1, rata_rata_smt2, jumlah_mk_tdk_lulus]])
    prediction = model.predict(input_data)[0]
    label = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

    st.success(f"📢 Hasil Prediksi: **{label[prediction]}**")

    if prediction == 0:
        st.warning("⚠️ Mahasiswa ini berisiko tinggi untuk Dropout. Pertimbangkan untuk memberi bimbingan khusus.")
