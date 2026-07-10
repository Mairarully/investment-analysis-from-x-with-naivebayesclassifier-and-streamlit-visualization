import ast
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# =========================
# KONFIGURASI
# =========================
INPUT_FILE    = "data/data_genz_labeled_full.csv"
TRAIN_FILE    = "data/train_70.csv"
TEST_FILE     = "data/test_30.csv"
VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"

VALID_LABELS  = ["positive", "negative", "netral"]
RANDOM_STATE  = 42

# =========================
# 1. LOAD & VALIDASI DATA
# =========================
print("=" * 60)
print("FEATURE EXTRACTION + SPLIT 70:30")
print("=" * 60)

print("\n1. Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"   Total rows loaded : {len(df)}")

# Filter label valid
df = df[df["sentiment_label"].isin(VALID_LABELS)].copy()
print(f"   Setelah filter    : {len(df)}")

print("\n   Distribusi label:")
for label, cnt in df["sentiment_label"].value_counts().items():
    print(f"   {label:<10} {cnt:>5} ({cnt/len(df)*100:.1f}%)")

# =========================
# 2. SPLIT 70:30 STRATIFIED
# =========================
print("\n2. Split 70:30 (stratified)...")
train_df, test_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["sentiment_label"]
)

print(f"   Train : {len(train_df)} rows")
print(f"   Test  : {len(test_df)} rows")

print("\n   Distribusi TRAIN:")
for label, cnt in train_df["sentiment_label"].value_counts().items():
    print(f"   {label:<10} {cnt:>5} ({cnt/len(train_df)*100:.1f}%)")

print("\n   Distribusi TEST:")
for label, cnt in test_df["sentiment_label"].value_counts().items():
    print(f"   {label:<10} {cnt:>5} ({cnt/len(test_df)*100:.1f}%)")

# =========================
# 3. KONVERSI TOKEN → TEXT
# =========================
print("\n3. Konversi tokens ke string...")

def tokens_to_text(tokens_raw):
    try:
        return " ".join(ast.literal_eval(tokens_raw))
    except Exception:
        return ""

X_train_text = train_df["tokens_stem"].apply(tokens_to_text)
X_test_text  = test_df["tokens_stem"].apply(tokens_to_text)
y_train      = train_df["sentiment_label"]
y_test       = test_df["sentiment_label"]

print(f"   Contoh teks train[0]: {X_train_text.iloc[0][:60]}...")

# =========================
# 4. TF-IDF (FIT HANYA DI TRAIN)
# =========================
print("\n4. TF-IDF Vectorization...")
print("   Fit di train, transform di train & test (no leakage)")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),   # unigram + bigram
    min_df=2,             # abaikan kata yang muncul <2x
    max_df=0.95,          # abaikan kata yang muncul di >95% dokumen
    sublinear_tf=True,    # log scaling TF (lebih baik untuk teks pendek)
)

X_train = vectorizer.fit_transform(X_train_text)   # FIT + TRANSFORM
X_test  = vectorizer.transform(X_test_text)        # TRANSFORM ONLY

print(f"   Jumlah fitur      : {X_train.shape[1]}")
print(f"   Shape X_train     : {X_train.shape}")
print(f"   Shape X_test      : {X_test.shape}")

# =========================
# 5. SIMPAN HASIL
# =========================
print("\n5. Menyimpan hasil...")

# Simpan split CSV (dengan semua kolom asli, untuk keperluan analisis)
train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)
print(f"   Train CSV -> {TRAIN_FILE}")
print(f"   Test CSV  -> {TEST_FILE}")

# Simpan vectorizer (dipakai saat prediction & pipeline)
import os
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, VECTORIZER_FILE)
print(f"   Vectorizer -> {VECTORIZER_FILE}")

print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)
print("\nFile yang dihasilkan:")
print(f"  {TRAIN_FILE}         <- data train (70%)")
print(f"  {TEST_FILE}          <- data test (30%)")
print(f"  {VECTORIZER_FILE}    <- TF-IDF vectorizer (untuk train.py)")
print()
print("Langkah berikutnya: jalankan train.py")
print("  train.py akan load train_70.csv + vectorizer yang sudah disimpan")
print("=" * 60)