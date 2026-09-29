!pip install pyspellchecker openpyxl -q

import os
import re
import requests
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spellchecker import SpellChecker

# Download Resource NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Proteksi jika data_json dari Sel 1 belum terdefinisi di memori
if 'data_json' not in locals() and 'data_json' not in globals():
    print("--> [INFO] Mengambil data dari API Open Library...")
    url = "https://openlibrary.org/search.json?q=computer+science&limit=200"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        data_json = response.json().get("docs", [])
    except Exception as e:
        print(f"[Error] Gagal koneksi API: {e}")
        data_json = []

# Inisialisasi Komponen NLP
stop_words_eng = set(stopwords.words('english'))
domain_stopwords = {
    'book', 'books', 'volume', 'edition', 'published', 'press', 'library',
    'science', 'computer', 'introduction', 'guide', 'handbook', 'manual', 'using'
}
all_stopwords = stop_words_eng.union(domain_stopwords)

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
spell = SpellChecker()

print("=" * 80)
print(" 2. MEMPROSES DATA BERSIH (NLP PIPELINE + PEMBENARAN KATA / SPELL CORRECTION)")
print("=" * 80)

clean_records = []
spelling_pairs = []

for idx, book in enumerate(data_json, start=1):
    title_raw = book.get("title", "")
    author_raw = book.get("author_name", ["Unknown"])[0] if book.get("author_name") else "Unknown"
    year_raw = book.get("first_publish_year", "N/A")
    subjects_raw = ", ".join(book.get("subject", [])[:5]) if book.get("subject") else ""

    combined_text = f"{title_raw} {subjects_raw}"

    # Preprocessing
    lower_text = combined_text.lower()
    clean_text = re.sub(r'[^a-z\s]', ' ', lower_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    tokens = clean_text.split()
    filtered_tokens = [w for w in tokens if w not in all_stopwords and len(w) > 2]

    # Pembenaran Kata (Spelling Correction)
    corrected_tokens = []
    for word in filtered_tokens:
        corrected = spell.correction(word)
        final_word = corrected if corrected else word
        corrected_tokens.append(final_word)

        status = "Dikoreksi" if word != final_word else "Sesuai"
        spelling_pairs.append({
            "Kata Asli (Token)": word,
            "Hasil Pembenaran Kata": final_word,
            "Status Koreksi": status
        })

    # Stemming & Lemmatization
    stemmed_tokens = [stemmer.stem(w) for w in corrected_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in corrected_tokens]
    final_clean_text = " ".join(lemmatized_tokens)

    clean_records.append({
        "No": idx,
        "Judul Asli Buku": title_raw,
        "Penulis (Author)": author_raw,
        "Tahun Terbit": year_raw,
        "Case Folding": lower_text,
        "Regex Cleaning": clean_text,
        "Tokens": str(tokens),
        "Stopwords Removal": str(filtered_tokens),
        "Spelling Correction": str(corrected_tokens),
        "Stemming": str(stemmed_tokens),
        "Lemmatization": str(lemmatized_tokens),
        "Teks Bersih Final": final_clean_text
    })

df_bersih = pd.DataFrame(clean_records)
df_spelling = pd.DataFrame(spelling_pairs).drop_duplicates().reset_index(drop=True)
df_koreksi_saja = df_spelling[df_spelling["Status Koreksi"] == "Dikoreksi"].reset_index(drop=True)

print("\n--- TABEL PEMBENARAN KATA (SAMPEL KATA YANG DIKOREKSI) ---")
print(df_koreksi_saja.head(10))
print(f"\n--> Total kata unik diproses: {len(df_spelling)} | Dikoreksi: {len(df_koreksi_saja)} kata\n")

print("=" * 80)
print(" 3. MEMPROSES EKSTRAKSI FITUR TEKS (UNIGRAM, BIGRAM & TF-IDF)")
print("=" * 80)

corpus = [doc for doc in df_bersih["Teks Bersih Final"].tolist() if doc.strip() != ""]

# Unigram
vec_unigram = CountVectorizer(ngram_range=(1, 1))
bag_unigram = vec_unigram.fit_transform(corpus)
sum_words = bag_unigram.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vec_unigram.vocabulary_.items()]
df_unigram = pd.DataFrame(sorted(words_freq, key=lambda x: x[1], reverse=True), columns=["Kata (Unigram)", "Frekuensi"])

# Bigram
vec_bigram = CountVectorizer(ngram_range=(2, 2))
bag_bigram = vec_bigram.fit_transform(corpus)
sum_bigrams = bag_bigram.sum(axis=0)
bigrams_freq = [(word, sum_bigrams[0, idx]) for word, idx in vec_bigram.vocabulary_.items()]
df_bigram = pd.DataFrame(sorted(bigrams_freq, key=lambda x: x[1], reverse=True), columns=["Pasangan Kata (Bigram)", "Frekuensi"])

# TF-IDF Vectorizer
tfidf_vec = TfidfVectorizer(ngram_range=(1, 1))
tfidf_mat = tfidf_vec.fit_transform(corpus)
tfidf_scores = tfidf_mat.mean(axis=0).A1
df_tfidf = pd.DataFrame({
    "Kata": tfidf_vec.get_feature_names_out(),
    "Rata-Rata Bobot TF-IDF": tfidf_scores
}).sort_values(by="Rata-Rata Bobot TF-IDF", ascending=False)

print(f"--> Ekstraksi Selesai: Unigram ({len(df_unigram)}), Bigram ({len(df_bigram)}), TF-IDF ({len(df_tfidf)}).\n")
