import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 1. INISIALISASI ---
# Inisialisasi Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Daftar Stopwords (Contoh sederhana, bisa ditambah atau load dari file)
stopwords_id = {"untuk", "dan", "di", "ke", "ini", "itu", "yang", "dan", "untuk", "dengan"}

# --- 2. FUNGSI PREPROCESSING ---
def preprocess_text(text):
    """
    Melakukan pembersihan teks: Case folding, Tokenization, 
    Stopword Removal, Normalization, dan Stemming.
    """
    if not isinstance(text, str):
        return ""
        
    # Case Folding
    text = text.lower()

    # Tokenization & Normalisasi (Menghapus karakter non-alfabet)
    tokens = re.findall(r'[a-z]+', text)

    # Stopword Removal & Stemming
    cleaned_tokens = []
    for word in tokens:
        if word not in stopwords_id:
            stemmed_word = stemmer.stem(word)
            if stemmed_word:
                cleaned_tokens.append(stemmed_word)

    return " ".join(cleaned_tokens)

# --- 3. FUNGSI UNTUK LOAD DATASET ---
def load_dataset(file_path):
    """Memuat dataset dari file CSV atau Excel."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV atau Excel.")

# --- 4. EKSEKUSI UTAMA ---
if __name__ == "__main__":
    # PILIHAN A: Gunakan Data Dummy (untuk testing)
    data = {
        "teks_dokumen": [
            "Pemerintah menetapkan kebijakan baru untuk meningkatkan pelayanan publik.",
            "Dokumen peraturan daerah ini bertujuan mengatur tata kelola administrasi pemerintahan.",
            "Laporan kinerja instansi pemerintah menunjukkan peningkatan efisiensi anggaran.",
        ]
    }
    df = pd.DataFrame(data)

    # PILIHAN B: Gunakan Dataset Sendiri (Hapus tanda pagar di bawah jika file sudah siap)
    # df = load_dataset("nama_file_anda.csv")

    print("--- Data Awal ---")
    print(df.head())

    # Proses Preprocessing
    print("\nSedang memproses teks... (Mohon tunggu, stemming memerlukan waktu)")
    df['teks_bersih'] = df['teks_dokumen'].apply(preprocess_text)

    print("\n--- Hasil Preprocessing ---")
    print(df[['teks_dokumen', 'teks_bersih']])

    # Simpan Hasil ke CSV Baru
    # df.to_csv("hasil_preprocessing.csv", index=False)