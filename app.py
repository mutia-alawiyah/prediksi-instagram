import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load model dan kolom yang sudah disimpan dari Colab
model = joblib.load('model_xgboost.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title('📊 Prediksi Performa Konten Instagram')

# Form Input (Tanpa Bulan)
hari_pilih = st.selectbox(
    'Hari Publish', ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
)
jenis_pilih = st.selectbox(
    'Jenis Konten', ['reels', 'carousel', 'single image', 'unknown']
)
tema_pilih = st.selectbox(
    'Tema Konten', ['entertainment', 'informasi', 'pov', 'edukasi', 'unknown']
)
kolaborasi_pilih = st.number_input('Jumlah Kolaborasi', min_value=0, value=0)

if st.button('🚀 Prediksi Sekarang', type='primary'):
  # Siapkan dataframe input (Bulan dihapus)
  input_data = pd.DataFrame(
      {
          'Jenis Konten': [jenis_pilih],
          'tema konten': [tema_pilih],
          'kolaborasi': [kolaborasi_pilih],
          'Hari_Publish': [hari_pilih],
      }
  )

  # One-hot encoding dan sesuaikan dengan kolom saat training di Colab
  input_encoded = pd.get_dummies(input_data)
  input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

  # Prediksi
  prediksi = model.predict(input_encoded)[0]
  probabilitas = model.predict_proba(input_encoded)[0]

  if prediksi == 1:
    st.success(
        f'🌟 **High Performer!** (Keyakinan: {probabilitas[1]*100:.2f}%)'
    )
  else:
    st.warning(f'⚠️ **Standar** (Keyakinan: {probabilitas[0]*100:.2f}%)')
