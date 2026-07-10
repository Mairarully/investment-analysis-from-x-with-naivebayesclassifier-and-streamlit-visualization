import ast
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score
)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =========================
# KONFIGURASI
# =========================
TEST_FILE      = "data/test_30.csv"
PREDICTED_FILE = "data/all_data_predicted.csv"
MODEL_FILE     = "models/naive_bayes_3class.pkl"
OUTPUT_DIR     = "final_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ANALISIS VISUAL — MINAT INVESTASI GEN Z")
print("=" * 70)

# =========================
# 1. EVALUASI MODEL
# =========================
print("\nSTEP 1: Evaluasi model...")

test_df    = pd.read_csv(TEST_FILE)
saved      = joblib.load(MODEL_FILE)
model      = saved["model"]
vectorizer = saved["vectorizer"]

X_test     = test_df["tokens_stem"].apply(lambda x: " ".join(ast.literal_eval(x)))
X_test_vec = vectorizer.transform(X_test)
y_true     = test_df["sentiment_label"]
y_pred     = model.predict(X_test_vec)

accuracy    = accuracy_score(y_true, y_pred)
f1_macro    = f1_score(y_true, y_pred, average="macro", zero_division=0)
report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

print(f"  Akurasi   : {accuracy:.2%}")
print(f"  F1 Macro  : {f1_macro:.4f}")
print(f"  Precision : {report_dict['weighted avg']['precision']:.3f}")
print(f"  Recall    : {report_dict['weighted avg']['recall']:.3f}")
print(f"  F1 Weighted: {report_dict['weighted avg']['f1-score']:.3f}")

labels = ["positive", "negative", "netral"]
cm     = confusion_matrix(y_true, y_pred, labels=labels)
cm_df  = pd.DataFrame(cm, index=labels, columns=labels)

# =========================
# 2. LOAD DATA PREDIKSI
# =========================
print("\nSTEP 2: Loading predicted data...")

df = pd.read_csv(PREDICTED_FILE)
print(f"  Total data: {len(df)}")

df_valid = df[
    (df["investment_type"] != "unknown") &
    (df["investment_reason"] != "unknown")
].copy()
print(f"  Data valid: {len(df_valid)} ({len(df_valid)/len(df)*100:.1f}%)")

# =========================
# 3. OVERVIEW (4 PANEL)
# =========================
print("\nSTEP 3: Overview visualization...")

fig = plt.figure(figsize=(16, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# Panel 1 — Pie Chart minat investasi
ax1 = fig.add_subplot(gs[0, 0])
inv_pct = len(df_valid) / len(df) * 100
sizes   = [len(df_valid), len(df) - len(df_valid)]
ax1.pie(sizes, explode=(0.05, 0),
        labels=['Bahas Investasi', 'Tidak Bahas'],
        colors=['#2ecc71', '#95a5a6'],
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'})
ax1.set_title(f'Tingkat Minat Investasi Gen Z\n({inv_pct:.1f}% membahas investasi)',
              fontsize=14, fontweight='bold', pad=20)

# Panel 2 — Jenis investasi
ax2 = fig.add_subplot(gs[0, 1])
inv_counts = df_valid["investment_type"].value_counts().sort_values()
ax2.barh(inv_counts.index, inv_counts.values, color='skyblue', edgecolor='navy', linewidth=1.5)
ax2.set_xlabel('Jumlah Mentions', fontsize=11, fontweight='bold')
ax2.set_title('Jenis Investasi Paling Diminati', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3)
for i, v in enumerate(inv_counts.values):
    ax2.text(v + 1, i, str(v), va='center', fontweight='bold', fontsize=9)

# Panel 3 — Alasan investasi
ax3 = fig.add_subplot(gs[1, 0])
reason_counts = df_valid["investment_reason"].value_counts().sort_values()
ax3.barh(reason_counts.index, reason_counts.values, color='salmon', edgecolor='darkred', linewidth=1.5)
ax3.set_xlabel('Jumlah Mentions', fontsize=11, fontweight='bold')
ax3.set_title('Alasan/Motivasi Berinvestasi', fontsize=14, fontweight='bold', pad=15)
ax3.grid(axis='x', alpha=0.3)
for i, v in enumerate(reason_counts.values):
    ax3.text(v + 1, i, str(v), va='center', fontweight='bold', fontsize=9)

# Panel 4 — Sentiment
ax4 = fig.add_subplot(gs[1, 1])
sent_counts = df_valid["sentiment_predicted"].value_counts()
colors_sent = {'positive': '#2ecc71', 'netral': '#3498db', 'negative': '#e74c3c'}
bar_colors  = [colors_sent.get(s, '#95a5a6') for s in sent_counts.index]
ax4.bar(sent_counts.index, sent_counts.values, color=bar_colors, edgecolor='black', linewidth=1.5, width=0.5)
ax4.set_xlabel('Sentiment', fontsize=11, fontweight='bold')
ax4.set_ylabel('Jumlah', fontsize=11, fontweight='bold')
ax4.set_title('Sentiment Keseluruhan', fontsize=14, fontweight='bold', pad=15)
ax4.grid(axis='y', alpha=0.3)
for i, v in enumerate(sent_counts.values):
    ax4.text(i, v + 2, str(v), ha='center', fontweight='bold', fontsize=12)

plt.suptitle('ANALISIS MINAT INVESTASI GENERASI Z', fontsize=18, fontweight='bold', y=0.98)
plt.savefig(f"{OUTPUT_DIR}/overview.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: overview.png")

# =========================
# 4. CROSS-ANALYSIS HEATMAP
# =========================
print("STEP 4: Cross-analysis heatmap...")

cross_tab = pd.crosstab(df_valid["investment_type"], df_valid["investment_reason"])
fig, ax   = plt.subplots(figsize=(16, 10))
sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=1, linecolor='white', cbar_kws={'label': 'Count'},
            ax=ax, annot_kws={'fontsize': 10, 'weight': 'bold'})
ax.set_title('Jenis Investasi x Alasan/Motivasi', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Alasan Investasi', fontsize=12, fontweight='bold')
ax.set_ylabel('Jenis Investasi', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/cross_analysis_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: cross_analysis_heatmap.png")

# =========================
# 5. SENTIMENT HEATMAP
# =========================
print("STEP 5: Sentiment heatmap...")

sent_inv = pd.crosstab(
    df_valid["investment_type"],
    df_valid["sentiment_predicted"],
    normalize='index'
) * 100

fig, ax = plt.subplots(figsize=(10, max(8, len(sent_inv) * 0.4)))
sns.heatmap(sent_inv, annot=True, fmt=".1f", cmap="RdYlGn",
            linewidths=2, linecolor='white', cbar_kws={'label': '%'},
            vmin=0, vmax=100, ax=ax, annot_kws={'fontsize': 11, 'weight': 'bold'})
ax.set_title('Sentiment per Jenis Investasi (%)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Sentiment', fontsize=12, fontweight='bold')
ax.set_ylabel('Jenis Investasi', fontsize=12, fontweight='bold')
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sentiment_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: sentiment_heatmap.png")

# =========================
# 6. HEATMAP LITERASI
# =========================
print("STEP 6: Heatmap literasi...")

if "literacy_level" in df_valid.columns:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(20, 5))

    # Literasi × Jenis Investasi (%)
    lit_inv = pd.crosstab(
        df_valid["literacy_level"],
        df_valid["investment_type"],
        normalize='index'
    ) * 100
    # Urutkan level literasi
    order = [l for l in ["high", "medium", "low"] if l in lit_inv.index]
    lit_inv = lit_inv.reindex(order)

    sns.heatmap(lit_inv, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=1, linecolor='white', cbar_kws={'label': '%'},
                vmin=0, vmax=100, ax=ax_l, annot_kws={'fontsize': 8})
    ax_l.set_title('Literasi × Jenis Investasi (%)', fontsize=13, fontweight='bold', pad=15)
    ax_l.set_xlabel('Jenis Investasi', fontsize=10, fontweight='bold')
    ax_l.set_ylabel('Tingkat Literasi', fontsize=10, fontweight='bold')
    ax_l.set_xticklabels(ax_l.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax_l.set_yticklabels(ax_l.get_yticklabels(), rotation=0)

    # Literasi × Alasan Investasi (%)
    lit_reason = pd.crosstab(
        df_valid["literacy_level"],
        df_valid["investment_reason"],
        normalize='index'
    ) * 100
    lit_reason = lit_reason.reindex(order)

    sns.heatmap(lit_reason, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=1, linecolor='white', cbar_kws={'label': '%'},
                vmin=0, vmax=100, ax=ax_r, annot_kws={'fontsize': 8})
    ax_r.set_title('Literasi × Alasan Investasi (%)', fontsize=13, fontweight='bold', pad=15)
    ax_r.set_xlabel('Alasan Investasi', fontsize=10, fontweight='bold')
    ax_r.set_ylabel('Tingkat Literasi', fontsize=10, fontweight='bold')
    ax_r.set_xticklabels(ax_r.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax_r.set_yticklabels(ax_r.get_yticklabels(), rotation=0)

    plt.suptitle('Literasi Keuangan Gen Z', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/heatmap_literasi.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: heatmap_literasi.png")
else:
    print("  SKIP: kolom literacy_level tidak ditemukan (jalankan prediction.py dulu)")

# =========================
# 7. CONFUSION MATRIX
# =========================
print("STEP 7: Confusion matrix...")

cm_total = cm.sum(axis=1, keepdims=True)
cm_pct   = cm / cm_total * 100  # persentase per baris (actual)

# Buat label gabungan: jumlah\n(persen%)
annot_labels = np.array([
    [f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(cm.shape[1])]
    for i in range(cm.shape[0])
])

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm_pct, annot=annot_labels, fmt="", cmap="Blues",
            linewidths=2, linecolor='white',
            vmin=0, vmax=100,
            xticklabels=labels, yticklabels=labels,
            ax=ax, annot_kws={'fontsize': 13, 'weight': 'bold'})
ax.set_title('Confusion Matrix — Sentimen Classification\n(jumlah & % dari actual)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Actual Label',    fontsize=12, fontweight='bold', labelpad=10)
ax.tick_params(axis='both', labelsize=11)

# Garis diagonal highlight
for i in range(len(labels)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='#e74c3c', lw=3))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: confusion_matrix.png")

# =========================
# 8. LAPORAN TEKS
# =========================
print("STEP 8: Text report...")

top_combo       = cross_tab.stack().idxmax()
top_combo_count = int(cross_tab.stack().max())
top_inv         = inv_counts.index[-1]
top_reason      = reason_counts.index[-1]
top_sent        = sent_counts.index[0]

lines = [
    "=" * 70,
    "LAPORAN ANALISIS MINAT INVESTASI GEN Z",
    "SENTIMENT ANALYSIS BERBASIS NAIVE BAYES + TF-IDF",
    "=" * 70,
    f"\nTanggal Analisis : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Total Data       : {len(df)} tweet",
    f"Data Valid       : {len(df_valid)} tweet ({len(df_valid)/len(df)*100:.1f}% membahas investasi)",
    f"Data Tidak Valid : {len(df)-len(df_valid)} tweet ({(len(df)-len(df_valid))/len(df)*100:.1f}%)",

    "\n" + "=" * 70,
    "A. EVALUASI MODEL NAIVE BAYES",
    "=" * 70,
    f"Akurasi          : {accuracy:.4f} ({accuracy*100:.2f}%)",
    f"F1 Macro         : {f1_macro:.4f}  ← patokan utama (imbalanced data)",
    f"F1 Weighted      : {report_dict['weighted avg']['f1-score']:.4f}",
    f"Precision (Avg)  : {report_dict['weighted avg']['precision']:.4f}",
    f"Recall (Avg)     : {report_dict['weighted avg']['recall']:.4f}",

    "\n" + "-" * 70,
    "METRICS PER KELAS",
    "-" * 70,
    f"{'Kelas':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}",
    "-" * 70,
]
for label in ["positive", "negative", "netral"]:
    if label in report_dict:
        r = report_dict[label]
        lines.append(f"{label:<12} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {int(r['support']):>10}")

lines += [
    "-" * 70,
    "\nCONFUSION MATRIX (Actual vs Predicted):",
    "         " + "  ".join(f"{l:>10}" for l in labels),
    "-" * 55,
]
for i, actual in enumerate(labels):
    row = "  ".join(f"{cm[i,j]:>10}" for j in range(len(labels)))
    pct_correct = cm[i,i] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
    lines.append(f"{actual:<9}{row}   ({pct_correct:.1f}% benar)")

lines += [
    "\n" + "=" * 70,
    "B. DISTRIBUSI SENTIMEN",
    "=" * 70,
]
for sent, cnt in sent_counts.items():
    pct = cnt / len(df_valid) * 100
    bar = "█" * int(pct / 2)
    lines.append(f"  {sent:<10} : {cnt:>4} tweet ({pct:>5.1f}%)  {bar}")

lines += [
    "\n" + "=" * 70,
    "C. DISTRIBUSI JENIS INVESTASI",
    "=" * 70,
]
for inv, cnt in inv_counts.sort_values(ascending=False).items():
    pct = cnt / len(df_valid) * 100
    bar = "█" * int(pct / 2)
    lines.append(f"  {inv:<15} : {cnt:>4} tweet ({pct:>5.1f}%)  {bar}")

lines += [
    "\n" + "=" * 70,
    "D. DISTRIBUSI ALASAN/MOTIVASI INVESTASI",
    "=" * 70,
]
for rsn, cnt in reason_counts.sort_values(ascending=False).items():
    pct = cnt / len(df_valid) * 100
    bar = "█" * int(pct / 2)
    lines.append(f"  {rsn:<20} : {cnt:>4} tweet ({pct:>5.1f}%)  {bar}")

lines += [
    "\n" + "=" * 70,
    "E. DISTRIBUSI LITERASI KEUANGAN",
    "=" * 70,
]
if "literacy_level" in df.columns:
    lit_dist = df["literacy_level"].value_counts()
    for level in ["high", "medium", "low"]:
        if level in lit_dist:
            cnt = lit_dist[level]
            pct = cnt / len(df) * 100
            bar = "█" * int(pct / 2)
            lines.append(f"  {level:<10} : {cnt:>4} tweet ({pct:>5.1f}%)  {bar}")

lines += [
    "\n" + "=" * 70,
    "F. CROSS-ANALYSIS: INVESTASI × SENTIMEN (%)",
    "=" * 70,
]
sent_inv_pct = pd.crosstab(df_valid["investment_type"], df_valid["sentiment_predicted"], normalize="index") * 100
lines.append(f"\n{'Investasi':<15}" + "".join(f"{c:>12}" for c in sent_inv_pct.columns))
lines.append("-" * (15 + 12 * len(sent_inv_pct.columns)))
for inv in sent_inv_pct.index:
    row = "".join(f"{sent_inv_pct.loc[inv, c]:>11.1f}%" for c in sent_inv_pct.columns)
    lines.append(f"{inv:<15}{row}")

lines += [
    "\n" + "=" * 70,
    "G. INSIGHT UTAMA",
    "=" * 70,
    f"\n1. Jenis Investasi Terpopuler : {top_inv.upper()} ({inv_counts.iloc[-1]} mentions)",
    f"2. Alasan Utama               : {top_reason.upper()} ({reason_counts.iloc[-1]} mentions)",
    f"3. Sentiment Dominan          : {top_sent.upper()} ({sent_counts.iloc[0]} mentions, {sent_counts.iloc[0]/len(df_valid)*100:.1f}%)",
    f"4. Kombinasi Terkuat          : {top_combo[0].upper()} × {top_combo[1].upper()} ({top_combo_count} mentions)",
]

# =========================
# INTERPRETASI PER LITERASI
# =========================
if "literacy_level" in df_valid.columns:
    lines += [
        "\n" + "=" * 70,
        "H. INTERPRETASI PER TINGKAT LITERASI",
        "=" * 70,
    ]
    for lit_level in ["high", "medium", "low"]:
        df_lit = df_valid[df_valid["literacy_level"] == lit_level]
        if len(df_lit) == 0:
            continue
        inv_dist_lit = df_lit["investment_type"].value_counts()
        total_lit    = len(df_lit)

        inv_parts = []
        for inv, cnt in inv_dist_lit.items():
            pct      = cnt / total_lit * 100
            df_inv   = df_lit[df_lit["investment_type"] == inv]
            rsn_dist = df_inv["investment_reason"].value_counts()
            top_rsn  = rsn_dist.index[0] if len(rsn_dist) > 0 else "N/A"
            top_rsn_pct = rsn_dist.iloc[0] / len(df_inv) * 100 if len(rsn_dist) > 0 else 0
            inv_parts.append((inv, cnt, pct, top_rsn, top_rsn_pct))

        # Narasi singkat
        top = inv_parts[0]
        kalimat = (
            f"Gen Z dengan tingkat literasi keuangan {lit_level.upper()} "
            f"({total_lit} tweet) cenderung memilih investasi {top[0].upper()} "
            f"sebanyak {top[1]} tweet ({top[2]:.1f}%) dengan motivasi utama "
            f"{top[3].upper()} ({top[4]:.1f}%)"
        )
        if len(inv_parts) > 1:
            sisanya = []
            for inv, cnt, pct, rsn, rsn_pct in inv_parts[1:3]:
                sisanya.append(f"{inv.upper()} {cnt} tweet ({pct:.1f}%, motivasi: {rsn.upper()} {rsn_pct:.1f}%)")
            kalimat += ", sedangkan yang lainnya memilih " + " dan ".join(sisanya)
        kalimat += "."

        lines += [
            f"\n{'─' * 70}",
            f"[{lit_level.upper()}] — {total_lit} tweet",
            f"{'─' * 70}",
            f"Narasi: {kalimat}",
            "\nJenis Investasi:",
        ]
        for inv, cnt, pct, rsn, rsn_pct in inv_parts:
            bar = "█" * int(pct / 4)
            lines.append(f"  {inv:<15} : {cnt:>3} tweet ({pct:>5.1f}%)  {bar}")

        lines.append("\nDetail Alasan per Investasi (Top 3):")
        for inv in inv_dist_lit.index[:3]:
            df_inv   = df_lit[df_lit["investment_type"] == inv]
            rsn_dist = df_inv["investment_reason"].value_counts()
            lines.append(f"\n  {inv.upper()} ({len(df_inv)} tweet):")
            for reason, cnt in list(rsn_dist.items())[:4]:
                pct_r = cnt / len(df_inv) * 100
                lines.append(f"    → {reason:<22} : {cnt:>3} tweet ({pct_r:.1f}%)")

        # Distribusi sentimen per literasi
        sent_lit = df_lit["sentiment_predicted"].value_counts() if "sentiment_predicted" in df_lit.columns else pd.Series()
        if len(sent_lit) > 0:
            lines.append("\nSentimen:")
            for s, cnt in sent_lit.items():
                pct_s = cnt / total_lit * 100
                lines.append(f"  {s:<10} : {cnt:>3} tweet ({pct_s:.1f}%)")

    # Cetak ringkasan ke terminal
    print("\n" + "=" * 70)
    print("INTERPRETASI PER TINGKAT LITERASI")
    print("=" * 70)
    for lit_level in ["high", "medium", "low"]:
        df_lit = df_valid[df_valid["literacy_level"] == lit_level]
        if len(df_lit) == 0:
            continue
        inv_dist_lit = df_lit["investment_type"].value_counts()
        total_lit    = len(df_lit)
        inv_parts    = []
        for inv, cnt in inv_dist_lit.items():
            pct         = cnt / total_lit * 100
            df_inv      = df_lit[df_lit["investment_type"] == inv]
            rsn_dist    = df_inv["investment_reason"].value_counts()
            top_rsn     = rsn_dist.index[0] if len(rsn_dist) > 0 else "N/A"
            top_rsn_pct = rsn_dist.iloc[0] / len(df_inv) * 100 if len(rsn_dist) > 0 else 0
            inv_parts.append((inv, cnt, pct, top_rsn, top_rsn_pct))
        top = inv_parts[0]
        kalimat = (
            f"  [{lit_level.upper()}] ({total_lit} tweet): "
            f"{top[0].upper()} {top[1]} tweet ({top[2]:.1f}%), motivasi: {top[3].upper()} ({top[4]:.1f}%)"
        )
        if len(inv_parts) > 1:
            sisanya = [f"{x[0].upper()} {x[1]} tweet ({x[2]:.1f}%)" for x in inv_parts[1:3]]
            kalimat += " | lainnya: " + ", ".join(sisanya)
        print(kalimat)

with open(f"{OUTPUT_DIR}/LAPORAN_LENGKAP.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"  Saved: LAPORAN_LENGKAP.txt")

# =========================
# SUMMARY
# =========================
print("\n" + "=" * 70)
print("ANALISIS SELESAI!")
print("=" * 70)
print(f"\nOutput folder: {OUTPUT_DIR}/")
print("  1. overview.png")
print("  2. cross_analysis_heatmap.png")
print("  3. sentiment_heatmap.png")
print("  4. heatmap_literasi.png")
print("  5. confusion_matrix.png")
print("  6. LAPORAN_LENGKAP.txt")
print(f"\nAkurasi   : {accuracy:.2%}")
print(f"F1 Macro  : {f1_macro:.4f}")
print(f"Top Inv   : {top_inv.upper()}")
print(f"Top Alasan: {top_reason.upper()}")
print("=" * 70)