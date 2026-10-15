import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Konfigurasi Tema Grafik
sns.set_theme(style='whitegrid')

# Load model dan kolom yang sudah disimpan dari Colab
model = joblib.load('model_xgboost.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title('📊 Prediksi Performa Konten Instagram')
st.markdown(
    'Aplikasi ini memprediksi apakah konten Anda berpotensi menjadi **High'
    ' Performer** berdasarkan parameter pra-publikasi.'
)

st.divider()

# --- BAGIAN 1: FORM INPUT & PREDIKSI ---
st.subheader('📝 Masukkan Parameter Konten')

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

  # One-hot encoding dan sesuaikan dengan kolom saat training di Colab
  input_encoded = pd.get_dummies(input_data)
  input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

  # Prediksi
  prediksi = model.predict(input_encoded)[0]
  probabilitas = model.predict_proba(input_encoded)[0]

  st.markdown('---')
  st.subheader('🎯 Hasil Prediksi')

  if prediksi == 1:
    st.success(
        f'🌟 **High Performer!**\n\nKonten dengan kombinasi ini diprediksi'
        f' memiliki *Extended Engagement Rate* di atas rata-rata. Tingkat'
        f' keyakinan model: **{probabilitas[1]*100:.2f}%**'
    )
  else:
    st.warning(
        f'⚠️ **Standar**\n\nKonten dengan kombinasi ini diprediksi memiliki'
        f' performa standar. Tingkat keyakinan model:'
        f' **{probabilitas[0]*100:.2f}%**'
    )

# --- BAGIAN 2: DIAGRAM & PENJELASAN DI BAWAH ---
st.divider()
st.subheader('📈 Analisis & Tingkat Kepentingan Fitur (Feature Importance)')
st.markdown(
    'Diagram di bawah menunjukkan seberapa besar pengaruh masing-masing'
    ' variabel terhadap keputusan model dalam menentukan performa konten.'
)

# Mengambil nilai feature importance dari model XGBoost
if hasattr(model, 'feature_importances_'):
  importances = pd.Series(model.feature_importances_, index=model_columns).sort_values(
      ascending=True
  )

  # Membuat Plot dengan Matplotlib & Seaborn
  fig, ax = plt.subplots(figsize=(8, max(4, len(model_columns) * 0.4)))
  importances.plot(kind='barh', color='darkorange', ax=ax)
  ax.set_title(
      'Bobot Pengaruh Fitur terhadap Performa Konten',
      fontsize=12,
      fontweight='bold',
  )
  ax.set_xlabel('Nilai Kepentingan (Importance Score)')
  ax.set_ylabel('Atribut Fitur')
  plt.tight_layout()

  # Menampilkan grafik ke Streamlit
  st.pyplot(fig)

  # Penjelasan Analitis Otomatis
  st.markdown('### 💡 Penjelasan & Wawasan (*Insights*):')
  st.markdown(
      '- **Faktor Dominan:** Fitur yang memiliki batang paling panjang pada'
      ' diagram di atas adalah variabel yang paling diandalkan oleh model'
      ' untuk menilai apakah sebuah konten sukses atau tidak.\n- **Strategi'
      ' Konten:** Anda dapat memprioritaskan jenis konten, tema, atau hari'
      ' publish yang memiliki bobot pengaruh tertinggi berdasarkan pola data'
      ' historis Anda sebelumnya.\n- **Kolaborasi & Waktu:** Jumlah'
      ' kolaborator dan pemilihan hari turut memberikan kontribusi spesifik'
      ' dalam mendongkrak interaksi audiens.'
  )
else:
  st.info('Model yang digunakan tidak mendukung fitur *Feature Importance*.')
