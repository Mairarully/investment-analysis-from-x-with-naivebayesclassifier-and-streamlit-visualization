import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# KONFIGURASI
# =========================
INPUT_FILE   = "data/data_genz_labeled_full.csv"
TRAIN_FILE   = "data/train_70.csv"
TEST_FILE    = "data/test_30.csv"

VALID_LABELS = ["positive", "negative", "netral"]
RANDOM_STATE = 42

# =========================
# 1. LOAD & VALIDASI DATA
# =========================
print("=" * 60)
print("SPLIT DATA 70:30")
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
# 3. SIMPAN HASIL
# =========================
print("\n3. Menyimpan hasil split...")
train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)
print(f"   Train CSV -> {TRAIN_FILE}")
print(f"   Test CSV  -> {TEST_FILE}")

print("\n" + "=" * 60)
print("SELESAI — Split data berhasil!")
print("=" * 60)
print("\nLangkah berikutnya: jalankan feature_extraction.py")
print("=" * 60)