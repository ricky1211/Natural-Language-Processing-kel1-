import streamlit as st
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Konfigurasi Halaman
st.set_page_config(page_title="Indonesian Text Preprocessor", layout="wide")

# Inisialisasi Sastrawi (Caching agar tidak loading terus menerus)
@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = load_stemmer()

# Daftar Stopwords Sederhana
stopwords_id = {"untuk", "dan", "di", "ke", "ini", "itu", "yang", "dengan", "adalah", "oleh"}

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Tokenisasi & Pembersihan Karakter
    tokens = re.findall(r'[a-z]+', text)
    # Stopword & Stemming
    cleaned = [stemmer.stem(w) for w in tokens if w not in stopwords_id]
    return " ".join(cleaned)

# --- Antarmuka Streamlit ---
st.title("🇮🇩 Indonesian Text Preprocessing App")
st.write("Unggah dataset Anda (CSV/Excel) untuk melakukan stemming dan pembersihan teks otomatis.")

uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Membaca file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Preview Data Asli")
    st.write(df.head())

    # Pilih Kolom
    column_to_process = st.selectbox("Pilih kolom teks yang akan diproses:", df.columns)

    if st.button("Mulai Proses Preprocessing"):
        with st.spinner('Sedang memproses teks... Mohon tunggu...'):
            df['teks_bersih'] = df[column_to_process].apply(preprocess_text)
            
        st.success("Selesai!")
        st.subheader("Hasil Preprocessing")
        st.write(df[[column_to_process, 'teks_bersih']].head())

        # Tombol Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Hasil Sebagai CSV",
            data=csv,
            file_name="hasil_preprocessing.csv",
            mime="text/csv",
        )