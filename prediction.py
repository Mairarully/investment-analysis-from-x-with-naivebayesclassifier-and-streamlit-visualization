import ast
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("output_prediction", exist_ok=True)
OUT = "output_prediction"

MODEL_FILE  = "models/naive_bayes_3class.pkl"
INPUT_FILE  = "data/data_genz_labeled_full.csv"
OUTPUT_FILE = "data/all_data_predicted.csv"

HIGH_LIT = ["diversifikasi","analisis","fundamental","teknikal","valuasi",
            "portofolio","risk management","yield","likuiditas","inflasi",
            "volatilitas","alokasi aset"]
MED_LIT  = ["belajar","pemula","mulai investasi","newbie","nabung",
             "reksadana","deposito","obligasi","sbn","edukasi"]

def detect_literacy(text):
    text = str(text).lower()
    if any(k in text for k in HIGH_LIT):   return "high"
    elif any(k in text for k in MED_LIT):  return "medium"
    return "low"

print("=" * 60)
print("PREDICTION — Prediksi Seluruh Data")
print("=" * 60)

print("\nLoading model...")
saved      = joblib.load(MODEL_FILE)
model      = saved["model"]
vectorizer = saved["vectorizer"]
classes    = saved.get("classes", model.classes_.tolist())
print(f"  Kelas: {classes}")

print("\nLoading data...")
df = pd.read_csv(INPUT_FILE)
print(f"  Total rows: {len(df)}")

print("\nPrediksi sentimen...")
X_text = df["tokens_stem"].apply(lambda x: " ".join(ast.literal_eval(x)))
X      = vectorizer.transform(X_text)

df["sentiment_predicted"] = model.predict(X)
proba                     = model.predict_proba(X)
df["confidence"]          = proba.max(axis=1)
for i, cls in enumerate(classes):
    df[f"prob_{cls}"] = proba[:, i]

print("Deteksi literasi...")
df["literacy_level"] = df["cleaned_text"].apply(detect_literacy)

# ===================== TAMPILKAN DI TERMINAL =====================
print("\n" + "=" * 60)
print("RINGKASAN HASIL PREDIKSI")
print("=" * 60)

print("\nSentimen (predicted):")
for l, c in df["sentiment_predicted"].value_counts().items():
    bar = "█" * int(c/len(df)*40)
    print(f"  {l:<10} {c:>5} ({c/len(df)*100:.1f}%)  {bar}")

print("\nLiteracy level:")
for l, c in df["literacy_level"].value_counts().items():
    print(f"  {l:<10} {c:>5} ({c/len(df)*100:.1f}%)")

print(f"\nRata-rata confidence : {df['confidence'].mean():.3f}")
print(f"Confidence < 0.5     : {(df['confidence'] < 0.5).sum()} rows")

# ===================== SIMPAN CSV =====================
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved CSV: {OUTPUT_FILE}")

# ===================== SIMPAN RINGKASAN TEKS =====================
with open(f"{OUT}/prediction_summary.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("RINGKASAN PREDIKSI\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total data: {len(df)}\n\n")
    f.write("Sentimen (predicted):\n")
    for l, c in df["sentiment_predicted"].value_counts().items():
        f.write(f"  {l:<10} {c:>5} ({c/len(df)*100:.1f}%)\n")
    f.write("\nLiteracy level:\n")
    for l, c in df["literacy_level"].value_counts().items():
        f.write(f"  {l:<10} {c:>5} ({c/len(df)*100:.1f}%)\n")
    f.write(f"\nRata-rata confidence: {df['confidence'].mean():.3f}\n")
print(f"Saved: {OUT}/prediction_summary.txt")

# ===================== VISUALISASI SENTIMEN =====================
sent_counts = df["sentiment_predicted"].value_counts()
colors_map  = {"positive": "#2ecc71", "netral": "#3498db", "negative": "#e74c3c"}
bar_colors  = [colors_map.get(s, "#95a5a6") for s in sent_counts.index]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(sent_counts.index, sent_counts.values, color=bar_colors, edgecolor="black", width=0.5)
ax.set_title("Distribusi Sentimen (Predicted)", fontsize=13, fontweight="bold")
ax.set_xlabel("Sentimen", fontweight="bold")
ax.set_ylabel("Jumlah", fontweight="bold")
for bar, v in zip(bars, sent_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 3, str(v),
            ha="center", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/sentimen_distribution.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {OUT}/sentimen_distribution.png")

# ===================== VISUALISASI LITERASI =====================
lit_counts = df["literacy_level"].value_counts().reindex(["high","medium","low"]).dropna()
colors_lit = ["#2ecc71", "#f39c12", "#e74c3c"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(lit_counts.index, lit_counts.values, color=colors_lit, edgecolor="black", width=0.5)
ax.set_title("Distribusi Tingkat Literasi Keuangan", fontsize=13, fontweight="bold")
ax.set_xlabel("Literasi", fontweight="bold")
ax.set_ylabel("Jumlah", fontweight="bold")
for bar, v in zip(bars, lit_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 3, str(v),
            ha="center", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/literacy_distribution.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {OUT}/literacy_distribution.png")

# ===================== CONFIDENCE DISTRIBUTION =====================
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["confidence"], bins=20, color="#667eea", edgecolor="black", alpha=0.8)
ax.axvline(df["confidence"].mean(), color="red", linestyle="--",
           label=f"Mean: {df['confidence'].mean():.2f}")
ax.set_xlabel("Confidence Score", fontweight="bold")
ax.set_ylabel("Frekuensi", fontweight="bold")
ax.set_title("Distribusi Confidence Score", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/confidence_distribution.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {OUT}/confidence_distribution.png")

print("\n" + "=" * 60)
print(f"Semua output disimpan di: {OUT}/")
print("  - prediction_summary.txt")
print("  - sentimen_distribution.png")
print("  - literacy_distribution.png")
print("  - confidence_distribution.png")
print(f"  - {OUTPUT_FILE}  (CSV lengkap)")
print("=" * 60)