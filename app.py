import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("dropout_model.pkl")

st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="centered")
st.title("🎓 Prediksi Risiko Mahasiswa Dropout")
st.write("Masukkan data mahasiswa untuk memprediksi apakah mereka berisiko Dropout.")

# Input form
with st.form("prediction_form"):
    age = st.slider("Umur saat pendaftaran", 17, 70, 20)
    prev_grade = st.slider("Nilai kualifikasi sebelumnya", 0.0, 200.0, 120.0)
    admission_grade = st.slider("Nilai masuk", 0.0, 200.0, 140.0)
    displaced = st.selectbox("Terdampak (Displaced)?", [0, 1])
    special_needs = st.selectbox("Kebutuhan khusus?", [0, 1])
    debtor = st.selectbox("Menunggak biaya?", [0, 1])
    tuition_paid = st.selectbox("Biaya kuliah lunas?", [0, 1])
    gender = st.selectbox("Gender", [0, 1])  # 0 = Perempuan, 1 = Laki-laki
    scholarship = st.selectbox("Mendapat beasiswa?", [0, 1])
    grade_1 = st.slider("Rata-rata nilai semester 1", 0.0, 20.0, 10.0)
    grade_2 = st.slider("Rata-rata nilai semester 2", 0.0, 20.0, 10.0)
    approved_1 = st.slider("Jumlah mata kuliah lulus sem 1", 0, 10, 5)
    approved_2 = st.slider("Jumlah mata kuliah lulus sem 2", 0, 10, 5)

    submitted = st.form_submit_button("🔍 Prediksi Dropout")

# Prediksi
if submitted:
    input_data = np.array([[age, prev_grade, admission_grade, displaced,
                            special_needs, debtor, tuition_paid, gender,
                            scholarship, grade_1, grade_2, approved_1, approved_2]])
    
    try:
        prediction = model.predict(input_data)
        label = prediction[0]
        status = {
            "Dropout": "❌ Berisiko Dropout",
            "Graduate": "🎓 Lulus",
            "Enrolled": "✅ Masih Terdaftar"
        }.get(label, label)
        st.success(f"Hasil Prediksi: **{status}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

