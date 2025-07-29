
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Judul aplikasi
st.title('Customer Churn Predictor')
st.text('Aplikasi ini memprediksi apakah pelanggan akan churn atau tidak.')

# Sidebar untuk input fitur
st.sidebar.header("Silakan masukkan fitur pengguna:")

def user_input_features():
    Tenure = st.sidebar.slider('Tenure (bulan)', 0, 60, 12)
    WarehouseToHome = st.sidebar.slider('Warehouse to Home (km)', 0, 100, 20)
    HourSpendOnApp = st.sidebar.slider('Jam di aplikasi per hari', 0, 10, 3)
    NumberOfDeviceRegistered = st.sidebar.slider('Jumlah perangkat terdaftar', 1, 10, 2)
    SatisfactionScore = st.sidebar.selectbox('Skor Kepuasan (1–5)', [1, 2, 3, 4, 5])
    Gender = st.sidebar.selectbox('Jenis Kelamin', ['Male', 'Female'])
    CityTier = st.sidebar.selectbox('Tingkat Kota (1–3)', [1, 2, 3])
    PreferredPaymentMode = st.sidebar.selectbox('Metode Pembayaran', ['Credit Card', 'Debit Card', 'UPI', 'Cash on Delivery', 'E wallet'])
    PreferredOrderCat = st.sidebar.selectbox('Kategori Pesanan', ['Mobile Phone', 'Laptop & Accessory', 'Fashion', 'Others'])
    MaritalStatus = st.sidebar.selectbox('Status Pernikahan', ['Single', 'Married', 'Divorced'])

    data = {
        'Tenure': Tenure,
        'WarehouseToHome': WarehouseToHome,
        'HourSpendOnApp': HourSpendOnApp,
        'NumberOfDeviceRegistered': NumberOfDeviceRegistered,
        'SatisfactionScore': SatisfactionScore,
        'Gender': Gender,
        'CityTier': CityTier,
        'PreferredPaymentMode': PreferredPaymentMode,
        'PreferredOrderCat': PreferredOrderCat,
        'MaritalStatus': MaritalStatus
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Load model dan prediksi
try:
    with open('churn_model.pkl', 'rb') as file:
        model = pickle.load(file)

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Hasil Prediksi')
    st.write('Churn' if prediction[0] == 1 else 'Tidak Churn')
    st.write(f"Probabilitas Churn: {prediction_proba[0][1]:.2f}")
except FileNotFoundError:
    st.error("Model belum ditemukan. Pastikan 'churn_model.pkl' ada di direktori yang sama.")
