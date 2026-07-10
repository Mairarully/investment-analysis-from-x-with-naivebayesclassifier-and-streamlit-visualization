# -*- coding: utf-8 -*-
"""
EVALUATE MODEL — Load model dari train.py, evaluasi di test set
================================================================
CATATAN:
- File ini TIDAK melakukan training ulang
- Model di-load dari models/naive_bayes_3class.pkl (hasil train.py)
- Pastikan train.py sudah dijalankan sebelum file ini

Urutan yang benar:
1. feature_extraction.py  → split data + TF-IDF vectorizer
2. train.py               → training + simpan model
3. evaluate_model.py      → evaluasi model (file ini)
4. prediction.py          → prediksi seluruh data
5. analysis.py            → visualisasi & laporan
"""

import ast
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, confusion_matrix
)

os.makedirs("output_train", exist_ok=True)

TEST_FILE  = "data/test_30.csv"
MODEL_FILE = "models/naive_bayes_3class.pkl"
OUT        = "output_train"

print("=" * 60)
print("EVALUASI MODEL — Naive Bayes 3 Kelas")
print("=" * 60)

def tokens_to_text(t):
    try: return " ".join(ast.literal_eval(t))
    except: return ""

# ===================== LOAD MODEL =====================
print("\n1. Loading model dari train.py...")
saved      = joblib.load(MODEL_FILE)
model      = saved["model"]
vectorizer = saved["vectorizer"]
best_alpha = saved.get("alpha", "?")
print(f"   Model loaded: {MODEL_FILE}")
print(f"   Alpha        : {best_alpha}")
print(f"   Kelas        : {model.classes_.tolist()}")
print(f"   Oversampling : {saved.get('oversampling', 'tidak ada info')}")

# ===================== LOAD TEST DATA =====================
print("\n2. Loading test data...")
df_test     = pd.read_csv(TEST_FILE)
X_test_text = df_test["tokens_stem"].apply(tokens_to_text)
y_test      = df_test["sentiment_label"]
print(f"   Test rows: {len(df_test)}")

print("\n   Distribusi TEST:")
for l, c in y_test.value_counts().items():
    print(f"   {l:<10} {c:>5} ({c/len(y_test)*100:.1f}%)")

# ===================== VECTORIZE =====================
print("\n3. Vectorizing test data...")
X_test = vectorizer.transform(X_test_text)
print(f"   Shape: {X_test.shape}")

# ===================== EVALUASI =====================
print("\n4. Evaluasi di test set...")
y_pred      = model.predict(X_test)
acc         = accuracy_score(y_test, y_pred)
f1_macro    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
labels      = model.classes_.tolist()
report_str  = classification_report(y_test, y_pred, labels=labels, zero_division=0)
report_dict = classification_report(y_test, y_pred, labels=labels, zero_division=0, output_dict=True)
cm          = confusion_matrix(y_test, y_pred, labels=labels)
cm_df       = pd.DataFrame(cm, index=labels, columns=labels)

print(f"\n   Akurasi    : {acc:.4f} ({acc*100:.2f}%)")
print(f"   F1 Macro   : {f1_macro:.4f}  ← patokan utama")
print(f"   F1 Weighted: {f1_weighted:.4f}")
print()
print(report_str)
print("Confusion Matrix:")
print(cm_df.to_string())

print()
for lb in labels:
    tp  = cm_df.loc[lb, lb]
    tot = cm_df.loc[lb].sum()
    print(f"  Recall {lb:<10}: {tp/tot if tot else 0:.2f}  ({tp}/{tot})")

# ===================== SIMPAN METRIK TEKS =====================
with open(f"{OUT}/evaluation_metrics.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("EVALUASI MODEL — Naive Bayes (load dari train.py)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model File       : {MODEL_FILE}\n")
    f.write(f"Alpha            : {best_alpha}\n")
    f.write(f"Oversampling     : {saved.get('oversampling', '-')}\n\n")
    f.write(f"Akurasi          : {acc:.4f} ({acc*100:.2f}%)\n")
    f.write(f"F1 Macro         : {f1_macro:.4f}\n")
    f.write(f"F1 Weighted      : {f1_weighted:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report_str + "\n")
    f.write("Confusion Matrix:\n")
    f.write(cm_df.to_string())
print(f"\nSaved: {OUT}/evaluation_metrics.txt")

# ===================== PLOT: CONFUSION MATRIX =====================
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues",
            linewidths=2, linecolor="white", ax=ax,
            annot_kws={"fontsize": 14, "weight": "bold"})
ax.set_title(f"Confusion Matrix (alpha={best_alpha}, oversampling)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
ax.set_ylabel("Actual",    fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix_eval.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {OUT}/confusion_matrix_eval.png")

# ===================== PLOT: F1 PER KELAS =====================
f1_vals   = [report_dict.get(c, {}).get("f1-score",  0) for c in labels]
prec_vals = [report_dict.get(c, {}).get("precision", 0) for c in labels]
rec_vals  = [report_dict.get(c, {}).get("recall",    0) for c in labels]
x = range(len(labels)); w = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar([i-w for i in x], prec_vals, width=w, label="Precision", color="#3498db", edgecolor="black")
b2 = ax.bar([i   for i in x], rec_vals,  width=w, label="Recall",    color="#2ecc71", edgecolor="black")
b3 = ax.bar([i+w for i in x], f1_vals,   width=w, label="F1-Score",  color="#e67e22", edgecolor="black")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.15); ax.set_ylabel("Score", fontweight="bold")
ax.set_title("Precision / Recall / F1 per Kelas", fontsize=12, fontweight="bold")
ax.legend(); ax.axhline(0.6, color="gray", linestyle="--", alpha=0.5)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.02, f"{h:.2f}",
                ha="center", fontsize=8, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/f1_per_kelas_eval.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {OUT}/f1_per_kelas_eval.png")

print("\n" + "=" * 60)
print(f"Output disimpan di: {OUT}/")
print("  - evaluation_metrics.txt")
print("  - confusion_matrix_eval.png")
print("  - f1_per_kelas_eval.png")
print(f"\nAkurasi: {acc:.2%}  |  F1 Macro: {f1_macro:.4f}  |  Alpha: {best_alpha}")
print("=" * 60)