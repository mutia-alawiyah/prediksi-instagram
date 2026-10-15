import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Konfigurasi Tema Visualisasi
sns.set_theme(style='whitegrid')

# Load model, kolom, dan dataset asli untuk referensi diagram
model = joblib.load('model_xgboost.pkl')
model_columns = joblib.load('model_columns.pkl')


# Load data asli untuk analisis/diagram pendukung
@st.cache_data
def load_original_data():
  df = pd.read_excel('coba.xlsx', sheet_name='Sheet2')
  df['Tanggal'] = pd.to_datetime(df['Tanggal'])
  df['Hari_dalam_Minggu'] = df['Tanggal'].dt.day_name()
  day_mapping = {
      'Monday': 'Senin',
      'Tuesday': 'Selasa',
      'Wednesday': 'Rabu',
      'Thursday': 'Kamis',
      'Friday': 'Jumat',
      'Saturday': 'Sabtu',
      'Sunday': 'Minggu',
  }
  df['Hari_Publish'] = df['Hari_dalam_Minggu'].map(day_mapping)
  df['Jenis Konten'] = (
      df['Jenis Konten']
      .replace(r'^\s*$', np.nan, regex=True)
      .fillna('unknown')
      .astype(str)
      .str.lower()
      .str.strip()
  )
  df['tema konten'] = (
      df['tema konten']
      .replace(r'^\s*$', np.nan, regex=True)
      .fillna('unknown')
      .astype(str)
      .str.lower()
      .str.strip()
  )
  df['kolaborasi'] = pd.to_numeric(df['kolaborasi'], errors='coerce').fillna(0)

  # Hitung Total Extended Engagement (Jumlah Absolut)
  df['Total_Engagement_Extended'] = (
      df['Like']
      + df['Comment']
      + df['Share']
      + df['Save']
      + df['Profile Visits']
      + df['Website Taps']
  )
  return df


df_data = load_original_data()

st.title('📊 Prediksi Performa & Analisis Extended Engagement')
st.markdown(
    'Aplikasi ini memprediksi apakah konten Anda berpotensi menjadi **High'
    ' Performer** serta menampilkan diagram tren berdasarkan **Total Extended'
    ' Engagement** (Like + Comment + Share + Save + Visits + Taps).'
)

st.divider()

# --- BAGIAN 1: FORM INPUT ---
st.subheader('📝 Masukkan Parameter Konten')

col1, col2 = st.columns(2)
with col1:
  hari_pilih = st.selectbox(
      'Hari Publish', ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
  )
  jenis_pilih = st.selectbox(
      'Jenis Konten', ['reels', 'carousel', 'single image', 'unknown']
  )
with col2:
  tema_pilih = st.selectbox(
      'Tema Konten', ['entertainment', 'informasi', 'pov', 'edukasi', 'unknown']
  )
  kolaborasi_pilih = st.number_input('Jumlah Kolaborasi', min_value=0, value=0)

# --- BAGIAN 2: TOMBOL PREDIKSI, PENJELASAN, & DIAGRAM ---
if st.button('🚀 Prediksi & Analisis Sekarang', type='primary', use_container_width=True):
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
  st.subheader('🎯 Hasil Prediksi & Penjelasan')

  if prediksi == 1:
    st.success(
        f'🌟 **Prediksi: HIGH PERFORMER!** (Keyakinan: {probabilitas[1]*100:.2f}%)'
    )
  else:
    st.warning(
        f'⚠️ **Prediksi: Standar / Performa Biasa** (Keyakinan:'
        f' {probabilitas[0]*100:.2f}%)'
    )

  st.markdown(
      f'**💡 Analisis Kombinasi:** Anda memilih konten **{jenis_pilih.upper()}**'
      f' bertema **{tema_pilih.upper()}** yang dipublish pada hari'
      f' **{hari_pilih}** dengan **{kolaborasi_pilih} kolaborator**. Model'
      f' menilai potensi kesuksesan kombinasi ini berdasarkan rekam jejak'
      f' interaksi absolut.'
  )

  # --- BAGIAN 3: DIAGRAM VISUALISASI BERDASARKAN EXTENDED ENGAGEMENT ---
  st.divider()
  st.subheader('📈 Diagram Rata-rata Total Extended Engagement')

  fig, axes = plt.subplots(2, 2, figsize=(12, 10))

  # 1. Grafik Berdasarkan Hari Publish
  day_order = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
  hari_agg = (
      df_data.groupby('Hari_Publish')['Total_Engagement_Extended']
      .mean()
      .reindex(day_order)
      .fillna(0)
  )
  sns.barplot(
      x=hari_agg.index,
      y=hari_agg.values,
      ax=axes[0, 0],
      palette='Blues_d',
      order=day_order,
  )
  axes[0, 0].set_title(
      'Rata-rata Extended Engagement per Hari Publish', fontweight='bold'
  )
  axes[0, 0].set_ylabel('Rata-rata Total Interaksi')
  axes[0, 0].tick_params(axis='x', rotation=30)

  # 2. Grafik Berdasarkan Jenis Konten
  jenis_agg = df_data.groupby('Jenis Konten')['Total_Engagement_Extended'].mean()
  sns.barplot(
      x=jenis_agg.index, y=jenis_agg.values, ax=axes[0, 1], palette='Purples_d'
  )
  axes[0, 1].set_title(
      'Rata-rata Extended Engagement berdasarkan Jenis Konten', fontweight='bold'
  )
  axes[0, 1].set_ylabel('Rata-rata Total Interaksi')

  # 3. Grafik Berdasarkan Tema Konten
  tema_agg = df_data.groupby('tema konten')['Total_Engagement_Extended'].mean()
  sns.barplot(
      x=tema_agg.index, y=tema_agg.values, ax=axes[1, 0], palette='Greens_d'
  )
  axes[1, 0].set_title(
      'Rata-rata Extended Engagement berdasarkan Tema Konten', fontweight='bold'
  )
  axes[1, 0].set_ylabel('Rata-rata Total Interaksi')
  axes[1, 0].tick_params(axis='x', rotation=25)

  # 4. Grafik Berdasarkan Jumlah Kolaborasi
  kolab_agg = df_data.groupby('kolaborasi')['Total_Engagement_Extended'].mean()
  sns.barplot(
      x=kolab_agg.index, y=kolab_agg.values, ax=axes[1, 1], palette='Oranges_d'
  )
  axes[1, 1].set_title(
      'Rata-rata Extended Engagement berdasarkan Jumlah Kolaborasi',
      fontweight='bold',
  )
  axes[1, 1].set_xlabel('Jumlah Kolaborator')
  axes[1, 1].set_ylabel('Rata-rata Total Interaksi')

  plt.tight_layout()
  st.pyplot(fig)
