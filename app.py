import ast
import io
import re
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score
)
from sklearn.model_selection import train_test_split
from datetime import datetime

st.set_page_config(
    page_title="Analisis Investasi Gen Z",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f8f9fa;}

    .main-title {
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        font-size: 1.05rem;
        line-height: 1.8;
    }

    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #667eea;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(102,126,234,0.08);
    }
    .metric-label { color: #888; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    .metric-value { font-size: 2rem; font-weight: 800; color: #667eea; }
    .metric-sub   { color: #764ba2; font-size: 0.82rem; margin-top: 0.3rem; }

    .step-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; padding: 3px 12px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 700; margin-right: 8px;
    }

    .lit-badge-high   { background: #00b894; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }
    .lit-badge-medium { background: #f39c12; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }
    .lit-badge-low    { background: #e74c3c; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }

    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: bold; color: #667eea; }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
    }

    .stButton>button:hover {
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: #f0f0f8; border-radius: 12px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #888; font-weight: 500; padding: 8px 20px; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea20, #764ba220) !important; color: #667eea !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ===================== KAMUS SENTIMEN v3 =====================
POSITIVE_LEXICON = {
    'cuan':3,'profit':2,'untung':1,'gain':2,'bullish':3,'rebound':2,'rally':2,
    'bagus':2,'aman':2,'stabil':2,'baik':1,'potensi':2,'prospek':2,'solid':2,'worth':2,
    'optimis':2,'semangat':1,'sabar':1,'disiplin':2,'konsisten':2,'percaya':1,'yakin':1,'senang':2,
    'nabung':1,'dividen':1,'yield':1,'hold':1,'keren':1,'mantap':2,'gas':1,
}
NEGATIVE_LEXICON = {
    'rugi':3,'loss':2,'minus':2,'nyangkut':3,'boncos':3,'anjlok':2,'jeblok':2,'ambruk':2,
    'merah':1,'crash':2,'bearish':2,'rekt':3,'scam':3,'manipulasi':2,'tipu':2,
    'bahaya':1,'waspada':1,'gagal':2,'bangkrut':3,'hancur':2,'kolaps':2,'ilang':1,
    'jelek':2,'buruk':2,'parah':1,'takut':1,'khawatir':1,'sedih':2,'panik':2,
    'stress':1,'nyesel':2,'redflag':2,'likuid':2,
}
NEGATION_PATTERN = re.compile(r'(tidak|gak|ga\b|nggak|bukan|jangan|tanpa|belum|anti)\s+(\w+)', re.IGNORECASE)
INVESTMENT_KEYWORDS = {
    "saham":     ["saham","stock","ihsg","idx","emiten","dividen","bluechip","lq45","bbri","bbca","tlkm","goto","bmri","tbk","msci"],
    "crypto":    ["crypto","bitcoin","btc","ethereum","eth","nft","blockchain","doge","altcoin","bnb","usdt","binance","kripto"],
    "reksadana": ["reksadana","reksa","dana","rd","bibit","ajaib","bareksa","tanamduit","mutual","fund","rdpu"],
    "emas":      ["emas","gold","antam","logam","mulia","pegadaian"],
    "deposito":  ["deposito","depo","tabungan","sbn","obligasi"],
    "p2p":       ["p2p","peer","lending","pinjaman","modalku","investree","koinworks"],
    "properti":  ["properti","rumah","tanah","kos","kosan","real","estate","apartemen"],
}
REASON_KEYWORDS = {
    "profit_cepat":   ["cuan","profit","cepat","short","scalping","trading","swing","day","moon","rocket","trade","forex"],
    "long_term":      ["jangka","panjang","pensiun","masa","depan","long","hold","hodl","nabung","saving","retirement"],
    "passive_income": ["passive","income","dividen","cashflow","yield","bunga","hasil","autopilot"],
    "trend_fomo":     ["viral","hype","trending","ramai","fomo","ikut","boom"],
    "belajar":        ["belajar","newbie","pemula","coba","mulai","awal","latihan","edukasi","ilmu","wawasan"],
    "diversifikasi":  ["diversifikasi","spread","portofolio","porto","risk","management"],
    "rekomendasi":    ["rekomendasi","saran","influencer","youtuber","guru","mentor","expert"],
}
HIGH_LIT = ["diversifikasi","analisis","fundamental","teknikal","valuasi","portofolio",
            "risk management","yield","likuiditas","inflasi","volatilitas","alokasi aset"]
MED_LIT  = ["belajar","pemula","mulai investasi","newbie","nabung","reksadana","deposito","obligasi","sbn","edukasi"]

def label_sentiment(tokens_raw, cleaned_text=""):
    try: token_list = ast.literal_eval(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
    except: return "netral"
    negated_words = {m.group(2).lower() for m in NEGATION_PATTERN.finditer(str(cleaned_text))}
    pos_score = neg_score = 0
    for token in token_list:
        pw = POSITIVE_LEXICON.get(token, 0)
        nw = NEGATIVE_LEXICON.get(token, 0)
        if token in negated_words:
            if pw > 0: neg_score += pw
            elif nw > 0: pos_score += nw
        else:
            pos_score += pw; neg_score += nw
    net = pos_score - neg_score
    if net >= 1: return "positive"
    elif net <= -1: return "negative"
    return "netral"

def detect_investment_type(text, tokens_raw):
    tl = []
    try: tl = ast.literal_eval(tokens_raw) if isinstance(tokens_raw, str) else []
    except: pass
    text_lower = str(text).lower()
    detected = [inv for inv, kws in INVESTMENT_KEYWORDS.items() if any(k in text_lower or k in tl for k in kws)]
    if not detected: return "unknown"
    return detected[0] if len(detected) == 1 else "|".join(detected)

def detect_reason(text, tokens_raw):
    tl = []
    try: tl = ast.literal_eval(tokens_raw) if isinstance(tokens_raw, str) else []
    except: pass
    text_lower = str(text).lower()
    detected = [r for r, kws in REASON_KEYWORDS.items() if any(k in text_lower or k in tl for k in kws)]
    if not detected: return "unknown"
    return detected[0] if len(detected) == 1 else "|".join(detected)

def detect_literacy(text):
    text = str(text).lower()
    if any(k in text for k in HIGH_LIT): return "high"
    elif any(k in text for k in MED_LIT): return "medium"
    return "low"

def tokens_to_text(t):
    try: return " ".join(ast.literal_eval(t))
    except: return ""

def dark_heatmap(data, fmt, cmap, title, xlabel, ylabel, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(max(10, len(data.columns)*1.2), max(5, len(data)*0.5)), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    kw = {"fontsize": 8, "color": "white"} if fmt == ".1f" else {"fontsize": 9}
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, linewidths=1, linecolor="#0f0f1a",
                vmin=vmin, vmax=vmax, ax=ax, annot_kws=kw)
    ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=12)
    ax.set_xlabel(xlabel, color="#f7971e", fontsize=10)
    ax.set_ylabel(ylabel, color="#f7971e", fontsize=10)
    ax.tick_params(colors="white")
    plt.xticks(rotation=45, ha="right", fontsize=8, color="white")
    plt.yticks(rotation=0, fontsize=8, color="white")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="#1a1a2e")
    buf.seek(0)
    plt.close()
    return buf

# ===================== HEADER =====================
st.markdown('<div class="main-title">🔬 ANALISIS INVESTASI GEN Z</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload CSV → Labeling → Split → Train → Evaluasi → Analisis</div>', unsafe_allow_html=True)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("## 🗂️ File Config")
    MODEL_FILE = st.text_input("Model PKL", "models/naive_bayes_3class.pkl")
    TEST_FILE  = st.text_input("Test CSV (fixed)", "data/test_30.csv")
    for fpath, label in [(MODEL_FILE, "Model PKL"), (TEST_FILE, "Test CSV")]:
        exists = os.path.exists(fpath)
        icon   = "✅" if exists else "❌"
        cls    = "status-ok" if exists else "status-err"
        st.markdown(f'<span class="{cls}">{icon} {label}</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    **Model:** Multinomial Naive Bayes  
    **Fitur:** TF-IDF (1-2 gram, 5000 fitur)  
    **Oversampling:** Random Oversampling  
    **Sentimen:** Lexicon v3 + Negasi
    """)
    st.markdown("---")
    st.markdown("### 📋 Tahapan Analisis")
    steps = ["Upload & Preview", "Labelling Sentimen", "Load Model (.pkl)",
             "Evaluasi (test_30.csv)", "Prediksi & Analisis"]
    for i, s in enumerate(steps, 1):
        st.markdown(f'<span class="step-badge">{i}</span>{s}', unsafe_allow_html=True)
        st.markdown("")

# ===================== TABS =====================
tab_home, tab_run, tab_result = st.tabs(["🏠 Home", "🚀 Jalankan Analisis", "📊 Hasil Analisis"])

with tab_home:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="section-header">Tentang Analisis</div>', unsafe_allow_html=True)
        st.markdown("""
        Analisis lengkap dari **CSV mentah** hingga **analisis minat investasi Gen Z**.

        **Algoritma & Teknik:**
        - Sentimen: Lexicon v3 berbobot + deteksi negasi dari cleaned_text
        - Model: Naive Bayes Multinomial (alpha auto-tuned)
        - Oversampling: Random oversampling kelas minoritas ke jumlah mayoritas
        - Fitur: TF-IDF unigram + bigram, 5000 fitur, sublinear TF
        - Evaluasi utama: F1 Macro (lebih jujur untuk data imbalance)
        """)
    with c2:
        st.info("""
        ### 📋 Format CSV yang Dibutuhkan

        Kolom wajib:
        - `cleaned_text` — teks sudah dibersihkan
        - `tokens_stem` — list token (format Python list)

        Contoh `tokens_stem`:
        ```
        ['beli', 'saham', 'untung', 'cuan']
        ```

        Kolom opsional tapi berguna:
        - `text` — teks asli
        - `genz_label` — label Gen Z
        """)

with tab_run:
    st.markdown('<div class="section-header">Upload & Jalankan Analisis</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (hasil preprocessing)", type="csv")

    if uploaded is None:
        st.info("Silakan upload file CSV untuk memulai.")
    else:
        df_raw = pd.read_csv(uploaded)
        st.success(f"File berhasil diupload: **{len(df_raw)} baris, {len(df_raw.columns)} kolom**")

        missing = [c for c in ["cleaned_text","tokens_stem"] if c not in df_raw.columns]
        if missing:
            st.error(f"Kolom tidak ditemukan: {missing}")
        else:
            with st.expander("Preview data (10 baris pertama)"):
                st.dataframe(df_raw.head(10))

            st.markdown("")
            if st.button("🚀 JALANKAN ANALISIS LENGKAP", use_container_width=True, type="primary"):
                prog   = st.progress(0)
                status = st.empty()

                try:
                    # STEP 1: LABELLING
                    with st.status("1️⃣ Labelling sentimen, investasi, alasan, literasi...", expanded=False):
                        df_raw["sentiment_label"]   = df_raw.apply(lambda r: label_sentiment(r["tokens_stem"], r.get("cleaned_text","")), axis=1)
                        df_raw["investment_type"]   = df_raw.apply(lambda r: detect_investment_type(r["cleaned_text"], r["tokens_stem"]), axis=1)
                        df_raw["investment_reason"] = df_raw.apply(lambda r: detect_reason(r["cleaned_text"], r["tokens_stem"]), axis=1)
                        df_raw["literacy_level"]    = df_raw["cleaned_text"].apply(detect_literacy)
                        st.write("Sentimen:", df_raw["sentiment_label"].value_counts().to_dict())
                    prog.progress(20)

                    # STEP 2: LOAD MODEL
                    with st.status("2️⃣ Load model & vectorizer dari .pkl...", expanded=False):
                        if not os.path.exists(MODEL_FILE):
                            st.error(f"Model tidak ditemukan: {MODEL_FILE}"); st.stop()
                        saved      = joblib.load(MODEL_FILE)
                        model      = saved["model"]
                        vectorizer = saved["vectorizer"]
                        best_alpha = getattr(model, "alpha", "N/A")
                        st.write(f"✅ Model loaded — alpha: **{best_alpha}**")
                    prog.progress(40)

                    # STEP 3: EVALUASI (test_30.csv fixed)
                    with st.status("3️⃣ Evaluasi model dengan test set fixed...", expanded=False):
                        if not os.path.exists(TEST_FILE):
                            st.error(f"Test file tidak ditemukan: {TEST_FILE}"); st.stop()
                        test_df    = pd.read_csv(TEST_FILE)
                        X_test_vec = vectorizer.transform(test_df["tokens_stem"].apply(tokens_to_text))
                        y_true     = test_df["sentiment_label"]
                        y_pred_ev  = model.predict(X_test_vec)

                        accuracy    = accuracy_score(y_true, y_pred_ev)
                        f1_macro    = f1_score(y_true, y_pred_ev, average="macro",    zero_division=0)
                        f1_weighted = f1_score(y_true, y_pred_ev, average="weighted", zero_division=0)
                        report_str  = classification_report(y_true, y_pred_ev, zero_division=0)
                        report_dict = classification_report(y_true, y_pred_ev, output_dict=True, zero_division=0)
                        labels_cm   = ["positive","negative","netral"]
                        cm          = confusion_matrix(y_true, y_pred_ev, labels=labels_cm)
                        cm_df       = pd.DataFrame(cm, index=labels_cm, columns=labels_cm)

                        fig, ax = plt.subplots(figsize=(7, 5))
                        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues",
                                    linewidths=2, linecolor="white", ax=ax,
                                    annot_kws={"fontsize": 13, "weight": "bold"})
                        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
                        ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("Actual", fontsize=11)
                        plt.tight_layout()
                        buf_cm = io.BytesIO()
                        plt.savefig(buf_cm, format="png", dpi=180, bbox_inches="tight")
                        buf_cm.seek(0); plt.close()
                        st.write(f"✅ Akurasi: **{accuracy:.2%}** | F1 Macro: **{f1_macro:.4f}**")
                    prog.progress(60)

                    # STEP 4: PREDIKSI + ANALISIS
                    with st.status("4️⃣ Prediksi seluruh data upload & analisis...", expanded=False):
                        X_all = vectorizer.transform(df_raw["tokens_stem"].apply(tokens_to_text))
                        df_raw["sentiment_predicted"] = model.predict(X_all)
                        df_raw["confidence"] = model.predict_proba(X_all).max(axis=1)

                        df_valid = df_raw[(df_raw["investment_type"] != "unknown") & (df_raw["investment_reason"] != "unknown")].copy()
                        st.write(f"Data valid untuk analisis: **{len(df_valid)}** baris")

                        buf_cross = buf_sent = buf_lit = None
                        if len(df_valid) > 0:
                            cross_tab = pd.crosstab(df_valid["investment_type"], df_valid["investment_reason"])
                            buf_cross = dark_heatmap(cross_tab, "d", "YlOrRd", "Jenis Investasi × Alasan", "Alasan Investasi", "Jenis Investasi")
                            sent_inv  = pd.crosstab(df_valid["investment_type"], df_valid["sentiment_predicted"], normalize="index") * 100
                            buf_sent  = dark_heatmap(sent_inv, ".1f", "RdYlGn", "Sentiment per Jenis Investasi (%)", "Sentiment", "Jenis Investasi", 0, 100)

                        if "literacy_level" in df_valid.columns and len(df_valid) > 0:
                            order   = [l for l in ["high","medium","low"] if l in df_valid["literacy_level"].values]
                            lit_inv = pd.crosstab(df_valid["literacy_level"], df_valid["investment_type"], normalize="index") * 100
                            lit_rsn = pd.crosstab(df_valid["literacy_level"], df_valid["investment_reason"], normalize="index") * 100
                            lit_inv = lit_inv.reindex([o for o in order if o in lit_inv.index])
                            lit_rsn = lit_rsn.reindex([o for o in order if o in lit_rsn.index])
                            fig2, (al, ar) = plt.subplots(1, 2, figsize=(20, 5))
                            for axx, data, title in [(al, lit_inv, "Literasi × Jenis Investasi (%)"), (ar, lit_rsn, "Literasi × Alasan Investasi (%)")]:
                                sns.heatmap(data, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=1, vmin=0, vmax=100, ax=axx, annot_kws={"fontsize": 8})
                                axx.set_title(title, fontsize=12, fontweight="bold")
                                axx.set_xticklabels(axx.get_xticklabels(), rotation=45, ha="right", fontsize=8)
                                axx.set_yticklabels(axx.get_yticklabels(), rotation=0)
                            plt.tight_layout()
                            buf_lit = io.BytesIO()
                            plt.savefig(buf_lit, format="png", dpi=180, bbox_inches="tight")
                            buf_lit.seek(0); plt.close()
                        else:
                            lit_inv = lit_rsn = None; buf_lit = None; order = []

                        insights = []
                        for lit_level in (order if order else []):
                            df_lit = df_valid[df_valid["literacy_level"] == lit_level]
                            if len(df_lit) == 0: continue
                            inv_dist  = df_lit["investment_type"].value_counts()
                            total     = len(df_lit)
                            inv_parts = []
                            for inv, cnt in inv_dist.items():
                                pct         = cnt / total * 100
                                df_inv      = df_lit[df_lit["investment_type"] == inv]
                                rsn_dist    = df_inv["investment_reason"].value_counts()
                                top_rsn     = rsn_dist.index[0] if len(rsn_dist) > 0 else "N/A"
                                top_rsn_pct = rsn_dist.iloc[0] / len(df_inv) * 100 if len(rsn_dist) > 0 else 0
                                inv_parts.append((inv, cnt, pct, top_rsn, top_rsn_pct))
                            top     = inv_parts[0]
                            kalimat = (f"Gen Z dengan tingkat literasi keuangan <b>{lit_level.upper()}</b> ({total} tweet) cenderung memilih investasi <b>{top[0].upper()}</b> sebanyak {top[1]} tweet ({top[2]:.1f}%) dengan motivasi utama <b>{top[3].upper()}</b> ({top[4]:.1f}%)")
                            if len(inv_parts) > 1:
                                sisanya = [f"<b>{inv.upper()}</b> {cnt} tweet ({pct:.1f}%, motivasi: <b>{rsn.upper()}</b> {rsn_pct:.1f}%)" for inv, cnt, pct, rsn, rsn_pct in inv_parts[1:3]]
                                kalimat += ", sedangkan yang lainnya memilih " + " dan ".join(sisanya)
                            kalimat += "."
                            insights.append("📌 " + kalimat)
                        st.write("Analisis selesai.")
                    prog.progress(100)

                    st.session_state.update({
                        "analysis": True,
                        "accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted,
                        "best_alpha": best_alpha,
                        "report_str": report_str, "report_dict": report_dict,
                        "cm_df": cm_df, "buf_cm": buf_cm,
                        "df_raw": df_raw, "df_valid": df_valid,
                        "sent_counts": df_valid["sentiment_predicted"].value_counts() if len(df_valid) > 0 else pd.Series(),
                        "inv_counts":  df_valid["investment_type"].value_counts()     if len(df_valid) > 0 else pd.Series(),
                        "literacy_dist": df_raw["literacy_level"].value_counts(),
                        "order": order, "insights": insights,
                        "buf_cross": buf_cross, "buf_sent": buf_sent, "buf_lit": buf_lit,
                    })
                    st.balloons()
                    st.success("🎉 Proses Analisis selesai! Lihat hasil di tab **Hasil Analisis**.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback; st.code(traceback.format_exc())

with tab_result:
    if "analysis" not in st.session_state:
        st.info("💡 Jalankan analisis terlebih dahulu di tab **Jalankan Analisis**.")
    else:
        acc         = st.session_state["accuracy"]
        f1_macro    = st.session_state["f1_macro"]
        f1_weighted = st.session_state["f1_weighted"]
        best_alpha  = st.session_state["best_alpha"]
        report_str  = st.session_state["report_str"]
        report_dict = st.session_state["report_dict"]
        cm_df       = st.session_state["cm_df"]
        buf_cm      = st.session_state["buf_cm"]
        df_raw      = st.session_state["df_raw"]
        df_valid    = st.session_state["df_valid"]
        literacy_dist = st.session_state["literacy_dist"]
        order       = st.session_state["order"]
        insights    = st.session_state["insights"]
        buf_cross   = st.session_state["buf_cross"]
        buf_sent    = st.session_state["buf_sent"]
        buf_lit     = st.session_state["buf_lit"]

        # METRIK
        st.markdown('<div class="section-header">🎯 Evaluasi Model</div>', unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        for col, (label, val, sub) in zip([c1,c2,c3,c4,c5], [
            ("Akurasi",    f"{acc:.2%}",      "Overall"),
            ("F1 Macro",   f"{f1_macro:.4f}", "Patokan Utama"),
            ("F1 Weighted",f"{f1_weighted:.4f}","Weighted"),
            ("Precision",  f"{report_dict['weighted avg']['precision']:.2%}","Weighted"),
            ("Best Alpha", f"{best_alpha}",   "Auto-tuned"),
        ]):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                            f'<div class="metric-value">{val}</div>'
                            f'<div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

        st.markdown("")
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown("#### 📋 Classification Report")
            st.code(report_str, language=None)
        with col2:
            st.markdown("#### 🔢 Confusion Matrix")
            buf_cm.seek(0)
            st.image(buf_cm, use_container_width=True)

        # F1 CHART
        st.markdown('<div class="section-header">📊 Precision / Recall / F1 per Kelas</div>', unsafe_allow_html=True)
        labels_list = ["positive","negative","netral"]
        fig_f1 = go.Figure()
        for metric, color in [("Precision","#667eea"),("Recall","#764ba2"),("F1-Score","#a29bfe")]:
            key = metric.lower().replace("-","")
            vals = [report_dict.get(l,{}).get(key if key!="f1score" else "f1-score",0) for l in labels_list]
            fig_f1.add_trace(go.Bar(name=metric, x=labels_list, y=vals, marker_color=color,
                                    text=[f"{v:.2f}" for v in vals], textposition="outside"))
        fig_f1.update_layout(barmode="group", plot_bgcolor="white", paper_bgcolor="white",
                             font=dict(color="#333"), yaxis=dict(range=[0,1.2], gridcolor="#eee"),
                             legend=dict(bgcolor="white"), height=330, margin=dict(t=20,b=20))
        st.plotly_chart(fig_f1, use_container_width=True)

        st.markdown("---")

        # SENTIMEN + LITERASI
        st.markdown('<div class="section-header">📊 Distribusi Sentimen & Literasi</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            sent_counts = st.session_state["sent_counts"]
            if len(sent_counts) > 0:
                colors_map = {"positive":"#00b894","netral":"#667eea","negative":"#e74c3c"}
                fig_s = go.Figure(go.Bar(
                    x=sent_counts.index.tolist(), y=sent_counts.values.tolist(),
                    marker_color=[colors_map.get(s,"#95a5a6") for s in sent_counts.index],
                    text=sent_counts.values, textposition="outside"
                ))
                fig_s.update_layout(title="Sentimen (Predicted)", plot_bgcolor="white", paper_bgcolor="white",
                                    font=dict(color="#333"), yaxis=dict(gridcolor="#eee"),
                                    height=300, margin=dict(t=40,b=20))
                st.plotly_chart(fig_s, use_container_width=True)
        with c2:
            if len(literacy_dist) > 0:
                colors_lit = {"high":"#00b894","medium":"#f39c12","low":"#e74c3c"}
                fig_l = go.Figure(go.Bar(
                    x=literacy_dist.index.tolist(), y=literacy_dist.values.tolist(),
                    marker_color=[colors_lit.get(l,"#95a5a6") for l in literacy_dist.index],
                    text=literacy_dist.values, textposition="outside"
                ))
                fig_l.update_layout(title="Tingkat Literasi Keuangan", plot_bgcolor="white", paper_bgcolor="white",
                                    font=dict(color="#333"), yaxis=dict(gridcolor="#eee"),
                                    height=300, margin=dict(t=40,b=20))
                st.plotly_chart(fig_l, use_container_width=True)

        st.markdown("---")

        # DETAIL PER LITERASI
        if order:
            st.markdown('<div class="section-header">🔍 Analisis Detail per Tingkat Literasi</div>', unsafe_allow_html=True)
            for lit_level in order:
                df_lit = df_valid[df_valid["literacy_level"] == lit_level]
                if len(df_lit) == 0: continue
                badge_html = f'<span class="lit-badge-{lit_level}">{lit_level.upper()}</span>'
                st.markdown(f"#### 📌 Gen Z Literasi {badge_html} — {len(df_lit)} tweet", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    inv_pct = (df_lit["investment_type"].value_counts() / len(df_lit) * 100).round(1)
                    fig_inv = px.bar(x=inv_pct.values, y=inv_pct.index, orientation="h",
                                    color=inv_pct.values, color_continuous_scale="Purples_r",
                                    labels={"x":"(%)","y":"Jenis Investasi"})
                    fig_inv.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                         font=dict(color="#333"), showlegend=False, height=280,
                                         margin=dict(t=10,b=10), coloraxis_showscale=False)
                    fig_inv.update_traces(text=[f"{v:.1f}%" for v in inv_pct.values], textposition="outside")
                    st.plotly_chart(fig_inv, use_container_width=True)
                with col2:
                    st.markdown("**Detail Alasan per Investasi (Top 3):**")
                    inv_dist = df_lit["investment_type"].value_counts()
                    for inv in inv_dist.index[:3]:
                        df_inv   = df_lit[df_lit["investment_type"] == inv]
                        rsn_dist = df_inv["investment_reason"].value_counts()
                        rsn_pct  = (rsn_dist / len(df_inv) * 100).round(1)
                        st.markdown(f"**{inv.upper()}** ({len(df_inv)} tweet)")
                        for reason, pct in list(rsn_pct.items())[:3]:
                            st.markdown(f"&nbsp;&nbsp;→ `{reason}`: **{pct:.1f}%**")
                st.markdown("---")

        # INSIGHT
        if insights:
            st.markdown('<div class="section-header">💡 Ringkasan Insight (Narasi Otomatis)</div>', unsafe_allow_html=True)
            for ins in insights:
                st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

        # HEATMAPS
        st.markdown('<div class="section-header">🔥 Visualisasi Cross-Analysis</div>', unsafe_allow_html=True)
        if buf_cross:
            st.markdown("**Jenis Investasi × Alasan/Motivasi:**")
            buf_cross.seek(0); st.image(buf_cross, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            if buf_sent:
                st.markdown("**Sentiment per Jenis Investasi (%):**")
                buf_sent.seek(0); st.image(buf_sent, use_container_width=True)
        with c2:
            if buf_lit:
                st.markdown("**Literasi Keuangan Gen Z:**")
                buf_lit.seek(0); st.image(buf_lit, use_container_width=True)

        st.markdown("---")

        # DOWNLOAD
        st.markdown('<div class="section-header">📥 Download Hasil</div>', unsafe_allow_html=True)
        d1,d2,d3,d4,d5 = st.columns(5)
        with d1:
            csv_buf = io.StringIO()
            df_raw.to_csv(csv_buf, index=False)
            st.download_button("📥 CSV Lengkap", csv_buf.getvalue(), "hasil_analisis.csv", "text/csv", use_container_width=True)
        with d2:
            buf_cm.seek(0)
            st.download_button("🔢 Confusion Matrix", buf_cm, "confusion_matrix.png", "image/png", use_container_width=True)
        with d3:
            if buf_cross:
                buf_cross.seek(0)
                st.download_button("🔥 Cross Heatmap", buf_cross, "cross_heatmap.png", "image/png", use_container_width=True)
        with d4:
            if buf_sent:
                buf_sent.seek(0)
                st.download_button("😊 Sentiment Map", buf_sent, "sentiment_heatmap.png", "image/png", use_container_width=True)
        with d5:
            if buf_lit:
                buf_lit.seek(0)
                st.download_button("📚 Literasi Map", buf_lit, "literasi_heatmap.png", "image/png", use_container_width=True)

# FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#444; padding:1.5rem 0; font-size:0.85rem; letter-spacing:1px;'>
    ANALISIS MINAT INVESTASI GEN Z &nbsp;·&nbsp; Skripsi 2026 &nbsp;·&nbsp; Naive Bayes + NLP + Oversampling
</div>
""", unsafe_allow_html=True)