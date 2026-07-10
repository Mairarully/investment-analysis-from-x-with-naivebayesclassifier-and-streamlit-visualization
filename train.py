import ast
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, confusion_matrix
)

os.makedirs("models", exist_ok=True)
os.makedirs("output_train", exist_ok=True)

TRAIN_FILE      = "data/train_70.csv"
TEST_FILE       = "data/test_30.csv"
VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"
MODEL_FILE      = "models/naive_bayes_3class.pkl"
OUT             = "output_train"

print("=" * 65)
print("TRAINING v2 — Naive Bayes + Random Oversampling")
print("=" * 65)

def tokens_to_text(t):
    try: return " ".join(ast.literal_eval(t))
    except: return ""

# ===================== LOAD DATA =====================
print("\n1. Loading data...")
df_train = pd.read_csv(TRAIN_FILE)
df_test  = pd.read_csv(TEST_FILE)
print(f"   Train: {len(df_train)} | Test: {len(df_test)}")

print("\n   Distribusi TRAIN (sebelum oversampling):")
for l, c in df_train["sentiment_label"].value_counts().items():
    print(f"   {l:<10} {c:>5} ({c/len(df_train)*100:.1f}%)")

# ===================== RANDOM OVERSAMPLING =====================
print("\n2. Random Oversampling kelas minoritas...")

# Target: samakan semua kelas ke jumlah kelas mayoritas
label_counts = df_train["sentiment_label"].value_counts()
min_count = label_counts.min()  # 143
max_count = min_count * 2       # 286 — target oversampling

oversampled_dfs = []
for label, count in label_counts.items():
    df_class = df_train[df_train["sentiment_label"] == label]
    if count < max_count:
        # Oversample: ambil random dengan replacement
        df_over = df_class.sample(max_count, replace=True, random_state=42)
        oversampled_dfs.append(df_over)
        print(f"   {label:<10}: {count} → {max_count} (+{max_count-count} samples)")
    else:
        oversampled_dfs.append(df_class)
        print(f"   {label:<10}: {count} (tidak diubah, kelas mayoritas)")

df_train_os = pd.concat(oversampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\n   Total setelah oversampling: {len(df_train_os)}")

# ===================== VECTORIZE =====================
print("\n3. Loading TF-IDF vectorizer...")
vectorizer = joblib.load(VECTORIZER_FILE)

X_train_text = df_train_os["tokens_stem"].apply(tokens_to_text)
X_test_text  = df_test["tokens_stem"].apply(tokens_to_text)
y_train      = df_train_os["sentiment_label"]
y_test       = df_test["sentiment_label"]

X_train = vectorizer.transform(X_train_text)
X_test  = vectorizer.transform(X_test_text)
print(f"   Fitur: {X_train.shape[1]}")

# ===================== ALPHA TUNING =====================
print("\n4. Mencari alpha terbaik...")
best_alpha   = 0.1
best_f1      = 0
best_model   = None
alpha_results = []

for alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    m = MultinomialNB(alpha=alpha)
    m.fit(X_train, y_train)
    y_p = m.predict(X_test)
    f1  = f1_score(y_test, y_p, average="macro", zero_division=0)
    acc = accuracy_score(y_test, y_p)
    alpha_results.append({"alpha": alpha, "f1_macro": f1, "accuracy": acc})
    print(f"   alpha={alpha:<5} → F1 Macro={f1:.4f}  Accuracy={acc:.4f}")
    if f1 > best_f1:
        best_f1    = f1
        best_alpha = alpha
        best_model = m

print(f"\n   Best alpha: {best_alpha} (F1 Macro={best_f1:.4f})")

# ===================== EVALUASI FINAL =====================
print("\n5. Evaluasi model terbaik...")
model       = best_model
y_pred      = model.predict(X_test)
acc         = accuracy_score(y_test, y_pred)
f1_macro    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
report_str  = classification_report(y_test, y_pred, labels=model.classes_, zero_division=0)
report_dict = classification_report(y_test, y_pred, labels=model.classes_, zero_division=0, output_dict=True)
labels      = model.classes_.tolist()
cm          = confusion_matrix(y_test, y_pred, labels=labels)
cm_df       = pd.DataFrame(cm, index=labels, columns=labels)

# Tampilkan di terminal
print(f"\n   Akurasi    : {acc:.4f} ({acc*100:.2f}%)")
print(f"   F1 Macro   : {f1_macro:.4f}  ← patokan utama")
print(f"   F1 Weighted: {f1_weighted:.4f}")
print()
print(report_str)
print("Confusion Matrix:")
print(cm_df.to_string())

# Recall per kelas
print()
for lb in labels:
    tp  = cm_df.loc[lb, lb]
    tot = cm_df.loc[lb].sum()
    print(f"  Recall {lb:<10}: {tp/tot if tot else 0:.2f}  ({tp}/{tot})")

# ===================== SIMPAN METRIK TEKS =====================
with open(f"{OUT}/training_metrics.txt", "w", encoding="utf-8") as f:
    f.write("=" * 65 + "\n")
    f.write("TRAINING v2 — Naive Bayes + Random Oversampling\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Best Alpha       : {best_alpha}\n")
    f.write(f"Akurasi          : {acc:.4f} ({acc*100:.2f}%)\n")
    f.write(f"F1 Macro         : {f1_macro:.4f}\n")
    f.write(f"F1 Weighted      : {f1_weighted:.4f}\n\n")
    f.write("Distribusi sebelum oversampling:\n")
    for l, c in label_counts.items():
        f.write(f"  {l:<10}: {c}\n")
    f.write(f"\nDistribusi setelah oversampling:\n")
    for l in label_counts.index:
        f.write(f"  {l:<10}: {max_count}\n")
    f.write("\nClassification Report:\n")
    f.write(report_str + "\n")
    f.write("Confusion Matrix:\n")
    f.write(cm_df.to_string())
print(f"\nSaved: {OUT}/training_metrics.txt")

# ===================== PLOT: CONFUSION MATRIX =====================
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues",
            linewidths=2, linecolor="white", ax=ax,
            annot_kws={"fontsize": 14, "weight": "bold"})
ax.set_title(f"Confusion Matrix (alpha={best_alpha}, OS)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
ax.set_ylabel("Actual",    fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix_train.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {OUT}/confusion_matrix_train.png")

# ===================== PLOT: ALPHA TUNING =====================
df_alpha = pd.DataFrame(alpha_results)
fig, ax  = plt.subplots(figsize=(8, 4))
ax.plot(df_alpha["alpha"].astype(str), df_alpha["f1_macro"],
        marker="o", color="#667eea", linewidth=2, markersize=8, label="F1 Macro")
ax.plot(df_alpha["alpha"].astype(str), df_alpha["accuracy"],
        marker="s", color="#e74c3c", linewidth=2, markersize=8, linestyle="--", label="Accuracy")
best_idx = df_alpha["f1_macro"].idxmax()
ax.scatter(str(best_alpha), best_f1, color="gold", s=200, zorder=5,
           edgecolors="black", linewidths=2, label=f"Best: α={best_alpha}")
ax.set_xlabel("Alpha", fontweight="bold")
ax.set_ylabel("Score", fontweight="bold")
ax.set_title("Alpha Tuning — F1 Macro vs Accuracy", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{OUT}/alpha_tuning.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {OUT}/alpha_tuning.png")

# ===================== PLOT: F1 PER KELAS =====================
f1_vals  = [report_dict.get(c, {}).get("f1-score",  0) for c in labels]
prec_vals= [report_dict.get(c, {}).get("precision", 0) for c in labels]
rec_vals = [report_dict.get(c, {}).get("recall",    0) for c in labels]
x = range(len(labels)); w = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar([i-w for i in x], prec_vals, width=w, label="Precision", color="#3498db", edgecolor="black")
b2 = ax.bar([i   for i in x], rec_vals,  width=w, label="Recall",    color="#2ecc71", edgecolor="black")
b3 = ax.bar([i+w for i in x], f1_vals,   width=w, label="F1-Score",  color="#e67e22", edgecolor="black")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.15); ax.set_ylabel("Score", fontweight="bold")
ax.set_title("Precision / Recall / F1 per Kelas (dengan Oversampling)", fontsize=12, fontweight="bold")
ax.legend(); ax.axhline(0.6, color="gray", linestyle="--", alpha=0.5)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.02, f"{h:.2f}",
                ha="center", fontsize=8, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/f1_per_kelas_train.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved: {OUT}/f1_per_kelas_train.png")

# ===================== SIMPAN MODEL =====================
print("\n6. Menyimpan model...")
joblib.dump({
    "model":         model,
    "vectorizer":    vectorizer,
    "feature_names": vectorizer.get_feature_names_out(),
    "classes":       model.classes_.tolist(),
    "alpha":         best_alpha,
    "accuracy":      acc,
    "f1_macro":      f1_macro,
    "oversampling":  "random",
}, MODEL_FILE)
print(f"   Saved: {MODEL_FILE}")

print("\n" + "=" * 65)
print(f"Output disimpan di: {OUT}/")
print("  - training_metrics.txt")
print("  - confusion_matrix_train.png")
print("  - alpha_tuning.png")
print("  - f1_per_kelas_train.png")
print(f"\nAkurasi: {acc:.2%}  |  F1 Macro: {f1_macro:.4f}  |  Best alpha: {best_alpha}")
print("=" * 65)