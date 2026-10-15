import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load model dan kolom yang sudah disimpan dari Colab
model = joblib.load('model_xgboost.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title('📊 Prediksi Performa Konten Instagram')
st.markdown(
    'Aplikasi ini memprediksi apakah konten Anda berpotensi menjadi **High'
    ' Performer** berdasarkan parameter pra-publikasi.'
)

st.divider()

# --- BAGIAN 1: FORM INPUT ---
st.subheader('📝 Masukkan Parameter Konten')

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

# --- BAGIAN 2: TOMBOL PREDIKSI & PENJELASAN DINAMIS ---
if st.button('🚀 Prediksi Sekarang', type='primary', use_container_width=True):
  # Siapkan dataframe input
  input_data = pd.DataFrame(
      {
          'Jenis Konten': [jenis_pilih],
          'tema konten': [tema_pilih],
          'kolaborasi': [kolaborasi_pilih],
          'Hari_Publish': [hari_pilih],
      }
  )

  # One-hot encoding dan sesuaikan dengan kolom saat training
  input_encoded = pd.get_dummies(input_data)
  input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

  # Proses Prediksi
  prediksi = model.predict(input_encoded)[0]
  probabilitas = model.predict_proba(input_encoded)[0]

  st.markdown('---')
  st.subheader('🎯 Hasil Analisis & Prediksi')

  # Menampilkan Kotak Hasil Status
  if prediksi == 1:
    st.success(
        f'🌟 **Prediksi: HIGH PERFORMER!**\n\n'
        f'Tingkat keyakinan model: **{probabilitas[1]*100:.2f}%**'
    )
  else:
    st.warning(
        f'⚠️ **Prediksi: Standar / Performa Biasa**\n\n'
        f'Tingkat keyakinan model: **{probabilitas[0]*100:.2f}%**'
    )

  # Menampilkan Penjelasan Rinci di Bawahnya saat Tombol Ditekan
  st.markdown('### 💡 Penjelasan Mengapa Hasil Ini Diprediksi:')
  st.markdown(
      f'- **Kombinasi Input Anda:** Anda memilih untuk mempublish konten'
      f' bertipe **{jenis_pilih.upper()}** dengan tema **{tema_pilih.upper()}**'
      f' pada hari **{hari_pilih}** dengan jumlah kolaborator sebanyak'
      f' **{kolaborasi_pilih} orang**.\n- **Logika Model:** Berdasarkan'
      f' pembelajaran pola historis dari data Anda, model mencocokkan'
      f' karakteristik kombinasi tersebut terhadap rasio *Extended Engagement'
      f' Rate* masa lalu.\n- **Kesimpulan:** {"Kombinasi ini memiliki daya tarik dan pola interaksi yang kuat berdasarkan rekam jejak data sebelumnya, sehingga dinilai berpotensi menghasilkan *engagement* di atas rata-rata." if prediksi == 1 else "Kombinasi ini memiliki karakteristik yang mirip dengan konten-konten masa lalu yang perolehan interaksinya cenderung moderat atau di bawah nilai median."}'
  )
