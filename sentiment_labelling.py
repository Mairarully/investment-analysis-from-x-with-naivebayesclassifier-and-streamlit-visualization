import pandas as pd
import ast
import re

# =========================
# KONFIGURASI
# =========================
INPUT_FILE  = "data/data_genz_only_tokenized_stopword_removed_stemmed.csv"
OUTPUT_FILE = "data/data_genz_labeled_full.csv"

# =========================
# KAMUS SENTIMEN BERBOBOT
# Dikurasi berdasarkan inspeksi langsung data Gen Z investasi
# =========================

POSITIVE_LEXICON = {
    # Profit / Return
    "cuan":      3,   # slang Gen Z, hampir selalu positif
    "profit":    2,
    "untung":    1,   # sering faktual ('keuntungan'), bobot rendah
    "gain":      2,
    "bullish":   3,   # sinyal kuat market positif
    "rebound":   2,
    "rally":     2,

    # Evaluasi positif (ada di data, jarang dinegasi)
    "bagus":     2,
    "aman":      2,
    "stabil":    2,
    "baik":      1,
    "potensi":   2,
    "prospek":   2,
    "solid":     2,
    "worth":     2,
    "mantap":    2,
    "keren":     1,

    # Emosi / sikap positif
    "optimis":   2,
    "semangat":  1,
    "sabar":     1,
    "disiplin":  2,
    "konsisten": 2,
    "percaya":   1,
    "yakin":     1,
    "senang":    2,

    # Aksi investasi positif
    "nabung":    1,
    "dividen":   1,
    "yield":     1,
    "hold":      1,
    "gas":       1,
}

NEGATIVE_LEXICON = {
    # Kerugian finansial — sinyal kuat
    "rugi":        3,
    "loss":        2,
    "minus":       2,   # 14x di data, hampir selalu negatif
    "nyangkut":    3,   # spesifik, selalu negatif
    "boncos":      3,   # slang, selalu negatif
    "anjlok":      2,
    "jeblok":      2,
    "ambruk":      2,
    "crash":       2,
    "bearish":     2,
    "rekt":        3,   # crypto slang
    "merah":       1,   # ambigu tapi sering negatif

    # Penipuan / risiko
    "scam":        3,
    "manipulasi":  2,
    "tipu":        2,
    "rawan":       1,
    "bahaya":      1,
    "waspada":     1,

    # Kegagalan
    "gagal":       2,
    "bangkrut":    3,
    "hancur":      2,
    "kolaps":      2,
    "likuid":      2,   # konteks: dicairkan paksa
    "hilang":      1,
    "ilang":       1,
    "jelek":       2,
    "buruk":       2,
    "parah":       1,
    "terjun":      1,

    # Ekspresi negatif Gen Z / umum
    "taek":        2,
    "bullshit":    2,
    "becus":       2,   # dari 'gak becus'
    "redflag":     2,

    # Emosi negatif
    "takut":       1,
    "khawatir":    1,
    "sedih":       2,
    "panik":       2,
    "panic":       2,
    "ragu":        1,
    "galau":       1,
    "stress":      1,
    "nyesel":      2,
}

# Pattern negasi — dideteksi dari cleaned_text
# karena stopword removal sudah menghapus kata ini dari tokens_stem
NEGATION_PATTERN = re.compile(
    r'(tidak|gak|ga\b|nggak|bukan|jangan|tanpa|belum|anti)\s+(\w+)',
    re.IGNORECASE
)

# =========================
# INVESTMENT TYPE & REASON KEYWORDS
# (Diperluas dari versi sebelumnya berdasarkan data aktual)
# =========================

INVESTMENT_KEYWORDS = {
    "saham": [
        "saham", "stock", "ihsg", "idx", "emiten", "dividen", "bluechip", "lq45",
        "bbri", "bbca", "tlkm", "goto", "unvr", "asii", "bmri", "tbk", "msci"
    ],
    "crypto": [
        "crypto", "bitcoin", "btc", "ethereum", "eth", "nft", "blockchain",
        "doge", "dogecoin", "altcoin", "shib", "bnb", "usdt", "binance", "web3",
        "kripto", "airdrop"
    ],
    "reksadana": [
        "reksadana", "reksa", "dana", "rd", "bibit", "ajaib", "bareksa",
        "tanamduit", "mutual", "fund", "rdpu"
    ],
    "emas": [
        "emas", "gold", "antam", "logam", "mulia", "pegadaian"
    ],
    "deposito": [
        "deposito", "depo", "tabungan", "sbn", "obligasi", "sbr", "sr"
    ],
    "p2p": [
        "p2p", "peer", "lending", "pinjaman", "modalku", "investree", "koinworks"
    ],
    "properti": [
        "properti", "rumah", "tanah", "kos", "kosan", "real", "estate", "apartemen"
    ],
}

REASON_KEYWORDS = {
    "profit_cepat": [
        "cuan", "profit", "cepat", "short", "scalping", "trading",
        "swing", "day", "moon", "rocket", "trade", "forex"
    ],
    "long_term": [
        "jangka", "panjang", "pensiun", "masa", "depan", "long", "hold",
        "hodl", "nabung", "saving", "retirement"
    ],
    "passive_income": [
        "passive", "income", "dividen", "cashflow", "yield", "bunga", "hasil", "autopilot"
    ],
    "trend_fomo": [
        "viral", "hype", "trending", "ramai", "fomo", "ikut", "boom"
    ],
    "belajar": [
        "belajar", "newbie", "pemula", "coba", "mulai", "awal",
        "latihan", "edukasi", "ilmu", "wawasan", "ajar"
    ],
    "diversifikasi": [
        "diversifikasi", "spread", "portofolio", "porto", "risk", "management"
    ],
    "rekomendasi": [
        "rekomendasi", "saran", "influencer", "youtuber", "guru", "mentor", "expert"
    ],
}


# =========================
# FUNGSI LABELLING
# =========================

def label_sentiment(tokens_raw, cleaned_text=""):
    """
    Klasifikasikan sentimen dari tokens_stem + cleaned_text.

    Alur:
    1. Parse token_list dari tokens_stem
    2. Scan cleaned_text untuk kata yang dinegasi (pola: negasi + kata)
    3. Scoring: kata dalam negated_words dibalik polaritasnya
    4. Label: positive/negative jika net score >= 1, else netral
    """
    try:
        token_list = ast.literal_eval(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
    except Exception:
        return "netral"

    if not token_list:
        return "netral"

    # Deteksi kata yang dinegasi dari cleaned_text
    negated_words = set()
    if cleaned_text:
        for match in NEGATION_PATTERN.finditer(str(cleaned_text)):
            negated_words.add(match.group(2).lower())

    # Scoring
    pos_score = neg_score = 0
    for token in token_list:
        pos_w = POSITIVE_LEXICON.get(token, 0)
        neg_w = NEGATIVE_LEXICON.get(token, 0)

        if token in negated_words:
            # Kata dinegasi → balik polaritas
            if pos_w > 0:
                neg_score += pos_w
            elif neg_w > 0:
                pos_score += neg_w
        else:
            pos_score += pos_w
            neg_score += neg_w

    net = pos_score - neg_score

    if net >= 1:
        return "positive"
    elif net <= -1:
        return "negative"
    return "netral"


def detect_investment_type(text, tokens_raw):
    text_lower = str(text).lower() if isinstance(text, str) else ""
    try:
        tl = ast.literal_eval(tokens_raw) if isinstance(tokens_raw, str) else []
    except Exception:
        tl = []
    detected = []
    for inv_type, keywords in INVESTMENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower or kw in tl:
                detected.append(inv_type)
                break
    if not detected:
        return "unknown"
    return detected[0] if len(detected) == 1 else "|".join(detected)


def detect_reason(text, tokens_raw):
    text_lower = str(text).lower() if isinstance(text, str) else ""
    try:
        tl = ast.literal_eval(tokens_raw) if isinstance(tokens_raw, str) else []
    except Exception:
        tl = []
    detected = []
    for reason, keywords in REASON_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower or kw in tl:
                detected.append(reason)
                break
    if not detected:
        return "unknown"
    return detected[0] if len(detected) == 1 else "|".join(detected)


# =========================
# MAIN PROCESS
# =========================

print("=" * 65)
print("SENTIMENT LABELLING v3 - Data-Driven + Negasi dari cleaned_text")
print("=" * 65)

print("\nLoading data...")
df = pd.read_csv(INPUT_FILE)
print(f"  Total rows: {len(df)}")

for col in ["cleaned_text", "tokens_stem"]:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan!")

print("\nLabelling...")
print("  1/3  Sentiment...")
df["sentiment_label"] = df.apply(
    lambda r: label_sentiment(r["tokens_stem"], r.get("cleaned_text", "")), axis=1
)

print("  2/3  Investment type...")
df["investment_type"] = df.apply(
    lambda r: detect_investment_type(r["cleaned_text"], r["tokens_stem"]), axis=1
)

print("  3/3  Investment reason...")
df["investment_reason"] = df.apply(
    lambda r: detect_reason(r["cleaned_text"], r["tokens_stem"]), axis=1
)

# =========================
# STATISTIK
# =========================
print("\n" + "=" * 65)
print("DISTRIBUSI HASIL LABELLING")
print("=" * 65)

total = len(df)
sent_counts = df["sentiment_label"].value_counts()

print("\n1. SENTIMEN:")
for label, cnt in sent_counts.items():
    bar = "=" * int(cnt / total * 40)
    print(f"   {label:<10} {cnt:>5} ({cnt/total*100:5.1f}%)  {bar}")

inv_counts = df["investment_type"].value_counts()
print(f"\n2. JENIS INVESTASI (top 10):")
for label, cnt in inv_counts.head(10).items():
    print(f"   {label:<35} {cnt:>5} ({cnt/total*100:5.1f}%)")

reason_counts = df["investment_reason"].value_counts()
print(f"\n3. ALASAN INVESTASI (top 10):")
for label, cnt in reason_counts.head(10).items():
    print(f"   {label:<35} {cnt:>5} ({cnt/total*100:5.1f}%)")

# =========================
# DIAGNOSTIK SAMPEL
# =========================
print("\n" + "=" * 65)
print("DIAGNOSTIK SAMPEL (5 per kelas)")
print("=" * 65)
for kelas in ["positive", "negative", "netral"]:
    subset = df[df["sentiment_label"] == kelas]
    print(f"\n[{kelas.upper()}] — {len(subset)} data")
    samples = subset["cleaned_text"].dropna().sample(min(5, len(subset)), random_state=42)
    for s in samples:
        print(f"   * {str(s)[:85]}")

# =========================
# STATUS & SARAN
# =========================
netral_pct = sent_counts.get("netral", 0) / total * 100
pos_pct    = sent_counts.get("positive", 0) / total * 100
neg_pct    = sent_counts.get("negative", 0) / total * 100

print("\n" + "=" * 65)
print("STATUS")
print("=" * 65)
print(f"   Netral   : {netral_pct:.1f}%")
print(f"   Positive : {pos_pct:.1f}%")
print(f"   Negative : {neg_pct:.1f}%")

if netral_pct > 85:
    print("\n[!] Netral masih sangat dominan. Pertimbangkan manual labelling atau IndoBERT.")
elif netral_pct > 70:
    print("\n[OK] Distribusi membaik. Lanjut ke train.py dengan SMOTE untuk tangani imbalance.")
else:
    print("\n[BAGUS] Distribusi cukup seimbang. Coba train tanpa SMOTE dulu.")

# =========================
# SAVE
# =========================
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved -> {OUTPUT_FILE}")
print(f"Total: {total} rows")
print("=" * 65)