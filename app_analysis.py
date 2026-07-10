
import ast
import io
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score
)
from datetime import datetime

st.set_page_config(
    page_title="Analisis Investasi Gen Z",
    page_icon="📊",
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

    .lit-badge-high   { background: #00b894; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }
    .lit-badge-medium { background: #f39c12; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }
    .lit-badge-low    { background: #e74c3c; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }

    .status-ok  { color: #00b894; font-weight: 700; }
    .status-err { color: #e74c3c; font-weight: 700; }

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

# ===================== HEADER =====================
st.markdown('<div class="main-title">💎 ANALISIS MINAT INVESTASI GEN Z</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Naive Bayes · TF-IDF · Twitter/X Dataset · Skripsi 2026</div>', unsafe_allow_html=True)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("## 🗂️ File Config")
    TEST_FILE      = st.text_input("Test CSV",      "data/test_30.csv")
    PREDICTED_FILE = st.text_input("Predicted CSV", "data/all_data_predicted.csv")
    MODEL_FILE     = st.text_input("Model PKL",     "models/naive_bayes_3class.pkl")
    OUTPUT_DIR     = "final_results"

    st.markdown("---")
    st.markdown("### ℹ️ Sistem")
    st.markdown("""
    **Framework:** CRISP-DM  
    **Model:** Multinomial Naive Bayes  
    **Fitur:** TF-IDF (1-2 gram)  
    **Oversampling:** Random OS  
    **Data:** Twitter/X Gen Z
    """)
    st.markdown("---")
    st.markdown("### ✅ Fitur")
    for f in ["Confusion Matrix","Precision/Recall/F1","Analisis Literasi",
              "Heatmap Cross-Analysis","Narasi Otomatis","Export PNG/TXT/XLSX"]:
        st.markdown(f"◆ {f}")

# ===================== TABS =====================
tab_home, tab_run, tab_result = st.tabs(["🏠 Home", "🚀 Jalankan Analisis", "📊 Hasil Lengkap"])

# ---- TAB HOME ----
with tab_home:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="section-header">Tentang Sistem</div>', unsafe_allow_html=True)
        st.markdown("""
        Sistem ini menganalisis **minat investasi Generasi Z** di Twitter/X menggunakan pendekatan 
        Natural Language Processing dan Machine Learning.

        **Pipeline Analisis:**
        - **Preprocessing** → Cleaning, tokenisasi, stemming
        - **Labelling** → Sentimen (lexicon v3), jenis investasi, alasan, literasi
        - **Modeling** → Naive Bayes Multinomial + TF-IDF + Random Oversampling
        - **Evaluasi** → Confusion Matrix, F1 Macro, Precision, Recall
        - **Analisis** → Cross-analysis literasi × investasi × alasan
        """)
    with c2:
        st.info("""
        ### 📋 Cara Pakai

        1. Pastikan file berikut tersedia:
           - `data/test_30.csv`
           - `data/all_data_predicted.csv`
           - `models/naive_bayes_3class.pkl`

        2. Klik tab **Jalankan Analisis**

        3. Tekan tombol **JALANKAN**

        4. Lihat hasil di tab **Hasil Lengkap**

        5. Download output (PNG, TXT, XLSX)
        """)

# ---- TAB JALANKAN ----
with tab_run:
    st.markdown('<div class="section-header">Jalankan Analisis Lengkap</div>', unsafe_allow_html=True)

    # Cek ketersediaan file
    files_ok = True
    for fpath, label in [(TEST_FILE, "Test CSV"), (PREDICTED_FILE, "Predicted CSV"), (MODEL_FILE, "Model PKL")]:
        exists = os.path.exists(fpath)
        icon   = "✅" if exists else "❌"
        color  = "status-ok" if exists else "status-err"
        st.markdown(f'<span class="{color}">{icon} {label}: <code>{fpath}</code></span>', unsafe_allow_html=True)
        if not exists: files_ok = False

    if not files_ok:
        st.error("Beberapa file tidak ditemukan. Pastikan pipeline sudah dijalankan terlebih dahulu.")

    st.markdown("")
    run_btn = st.button("🚀 JALANKAN ANALISIS LENGKAP", use_container_width=True, type="primary", disabled=not files_ok)

    if run_btn:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        prog = st.progress(0)
        status = st.empty()

        try:
            # STEP 1 — Load
            status.markdown("**🔄 Step 1/6:** Loading data & model...")
            test_df = pd.read_csv(TEST_FILE)
            df      = pd.read_csv(PREDICTED_FILE)
            saved   = joblib.load(MODEL_FILE)
            model   = saved["model"]
            vectorizer = saved["vectorizer"]
            prog.progress(15)

            # STEP 2 — Evaluasi
            status.markdown("**🔄 Step 2/6:** Evaluasi model...")
            X_test     = test_df["tokens_stem"].apply(lambda x: " ".join(ast.literal_eval(x)))
            X_test_vec = vectorizer.transform(X_test)
            y_true     = test_df["sentiment_label"]
            y_pred     = model.predict(X_test_vec)

            accuracy    = accuracy_score(y_true, y_pred)
            f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            report_str  = classification_report(y_true, y_pred, zero_division=0)
            report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            labels_cm   = ["positive", "negative", "netral"]
            cm          = confusion_matrix(y_true, y_pred, labels=labels_cm)
            cm_df       = pd.DataFrame(cm, index=labels_cm, columns=labels_cm)
            prog.progress(30)

            # STEP 3 — Confusion Matrix
            status.markdown("**🔄 Step 3/6:** Membuat confusion matrix...")
            fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
            ax.set_facecolor("white")
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues",
                        linewidths=2, linecolor="white", ax=ax,
                        annot_kws={"fontsize": 14, "weight": "bold", "color": "white"})
            ax.set_title("Confusion Matrix — Sentiment Classification",
                         fontsize=13, fontweight="bold", pad=12, color="white")
            ax.set_xlabel("Predicted", fontsize=11, color="#667eea")
            ax.set_ylabel("Actual",    fontsize=11, color="#667eea")
            ax.tick_params(colors="white")
            plt.tight_layout()
            buf_cm = io.BytesIO()
            plt.savefig(buf_cm, format="png", dpi=200, bbox_inches="tight", facecolor="white")
            buf_cm.seek(0)
            plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=200, bbox_inches="tight", facecolor="white")
            plt.close()
            prog.progress(45)

            # STEP 4 — Literasi
            status.markdown("**🔄 Step 4/6:** Analisis literasi...")
            df_valid = df[(df["investment_type"] != "unknown") & (df["investment_reason"] != "unknown")].copy()
            literacy_dist = df["literacy_level"].value_counts() if "literacy_level" in df.columns else pd.Series()

            order   = [l for l in ["high","medium","low"] if "literacy_level" in df_valid.columns and l in df_valid["literacy_level"].values]
            lit_inv = lit_reason = None
            if "literacy_level" in df_valid.columns and len(order) > 0:
                lit_inv    = pd.crosstab(df_valid["literacy_level"], df_valid["investment_type"],  normalize="index") * 100
                lit_reason = pd.crosstab(df_valid["literacy_level"], df_valid["investment_reason"], normalize="index") * 100
                lit_inv    = lit_inv.reindex([o for o in order if o in lit_inv.index])
                lit_reason = lit_reason.reindex([o for o in order if o in lit_reason.index])
            prog.progress(60)

            # STEP 5 — Heatmap Literasi
            status.markdown("**🔄 Step 5/6:** Membuat heatmap...")
            if lit_inv is not None:
                fig2, (al, ar) = plt.subplots(1, 2, figsize=(20, 5), facecolor="white")
                for axx in [al, ar]: axx.set_facecolor("white")
                sns.heatmap(lit_inv, annot=True, fmt=".1f", cmap="YlOrRd",
                            linewidths=1, vmin=0, vmax=100, ax=al, annot_kws={"fontsize": 8})
                al.set_title("Literasi × Jenis Investasi (%)", fontsize=12, fontweight="bold", color="white")
                al.set_xticklabels(al.get_xticklabels(), rotation=45, ha="right", fontsize=8, color="white")
                al.set_yticklabels(al.get_yticklabels(), color="white")
                al.set_xlabel("Jenis Investasi", color="#667eea")
                al.set_ylabel("Tingkat Literasi", color="#667eea")

                sns.heatmap(lit_reason, annot=True, fmt=".1f", cmap="YlOrRd",
                            linewidths=1, vmin=0, vmax=100, ax=ar, annot_kws={"fontsize": 8})
                ar.set_title("Literasi × Alasan Investasi (%)", fontsize=12, fontweight="bold", color="white")
                ar.set_xticklabels(ar.get_xticklabels(), rotation=45, ha="right", fontsize=8, color="white")
                ar.set_yticklabels(ar.get_yticklabels(), color="white")
                ar.set_xlabel("Alasan Investasi", color="#667eea")
                ar.set_ylabel("Tingkat Literasi", color="#667eea")

                fig2.patch.set_facecolor("white")
                plt.suptitle("Literasi Keuangan Gen Z", fontsize=14, fontweight="bold", color="white", y=1.02)
                plt.tight_layout()
                buf_lit = io.BytesIO()
                plt.savefig(buf_lit, format="png", dpi=200, bbox_inches="tight", facecolor="white")
                buf_lit.seek(0)
                plt.savefig(f"{OUTPUT_DIR}/heatmap_literasi.png", dpi=200, bbox_inches="tight", facecolor="white")
                plt.close()
            else:
                buf_lit = None
            prog.progress(80)

            # STEP 6 — Insight & Laporan
            status.markdown("**🔄 Step 6/6:** Menyusun insight & laporan...")
            insights = []
            if lit_inv is not None:
                for lit_level in order:
                    df_lit = df_valid[df_valid["literacy_level"] == lit_level]
                    if len(df_lit) == 0: continue
                    inv_dist = df_lit["investment_type"].value_counts()
                    top_inv  = inv_dist.index[0]
                    top_inv_pct = inv_dist.iloc[0] / len(df_lit) * 100
                    df_top   = df_lit[df_lit["investment_type"] == top_inv]
                    rsn_dist = df_top["investment_reason"].value_counts()
                    top_rsn  = rsn_dist.index[0] if len(rsn_dist) > 0 else "N/A"
                    top_rsn_pct = rsn_dist.iloc[0] / len(df_top) * 100 if len(rsn_dist) > 0 else 0
                    sec_txt  = f", diikuti {inv_dist.index[1].upper()} ({inv_dist.iloc[1]/len(df_lit)*100:.1f}%)" if len(inv_dist) > 1 else ""
                    insights.append(
                        f"📌 <b>Gen Z Literasi {lit_level.upper()}</b> ({len(df_lit)} tweet): "
                        f"Cenderung membahas investasi <b>{top_inv.upper()}</b> ({top_inv_pct:.1f}%) "
                        f"dengan motivasi utama <b>{top_rsn.upper()}</b> ({top_rsn_pct:.1f}%){sec_txt}."
                    )

            # Simpan laporan txt
            report_lines = [
                "=" * 65,
                "LAPORAN ANALISIS MINAT INVESTASI GEN Z",
                "=" * 65,
                f"\nTanggal     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Total Data  : {len(df)}",
                f"Data Valid  : {len(df_valid)} ({len(df_valid)/len(df)*100:.1f}%)",
                "\n" + "=" * 65,
                "EVALUASI MODEL",
                "=" * 65,
                f"Akurasi          : {accuracy:.4f} ({accuracy*100:.2f}%)",
                f"F1 Macro         : {f1_macro:.4f}",
                f"F1 Weighted      : {f1_weighted:.4f}",
                "\nClassification Report:",
                report_str,
                "Confusion Matrix:",
                cm_df.to_string(),
            ]
            with open(f"{OUTPUT_DIR}/LAPORAN_ANALISIS.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))

            # Simpan Excel
            try:
                with pd.ExcelWriter(f"{OUTPUT_DIR}/hasil_analisis.xlsx", engine="openpyxl") as writer:
                    cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
                    if lit_inv is not None:
                        lit_inv.to_excel(writer, sheet_name="Literasi_x_Investasi")
                        lit_reason.to_excel(writer, sheet_name="Literasi_x_Alasan")
                    if len(literacy_dist) > 0:
                        literacy_dist.to_frame().to_excel(writer, sheet_name="Distribusi_Literasi")
            except Exception:
                pass

            # Simpan ke session
            st.session_state.update({
                "analysis_done": True,
                "accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted,
                "report_str": report_str, "report_dict": report_dict,
                "cm_df": cm_df, "buf_cm": buf_cm, "buf_lit": buf_lit,
                "df": df, "df_valid": df_valid, "literacy_dist": literacy_dist,
                "lit_inv": lit_inv, "lit_reason": lit_reason,
                "insights": insights,
            })

            prog.progress(100)
            status.markdown("**✅ ANALISIS SELESAI!**")
            st.balloons()
            st.success("Analisis berhasil! Lihat hasil di tab **Hasil Lengkap**.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ---- TAB HASIL ----
with tab_result:
    if "analysis_done" not in st.session_state:
        st.info("💡 Jalankan analisis terlebih dahulu di tab **Jalankan Analisis**.")
    else:
        acc         = st.session_state["accuracy"]
        f1_macro    = st.session_state["f1_macro"]
        f1_weighted = st.session_state["f1_weighted"]
        report_str  = st.session_state["report_str"]
        report_dict = st.session_state["report_dict"]
        cm_df       = st.session_state["cm_df"]
        buf_cm      = st.session_state["buf_cm"]
        buf_lit     = st.session_state["buf_lit"]
        df          = st.session_state["df"]
        df_valid    = st.session_state["df_valid"]
        literacy_dist = st.session_state["literacy_dist"]
        lit_inv     = st.session_state["lit_inv"]
        lit_reason  = st.session_state["lit_reason"]
        insights    = st.session_state["insights"]

        # ---- METRIK ----
        st.markdown('<div class="section-header">🎯 Evaluasi Model Naive Bayes</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            ("Akurasi", f"{acc:.2%}", "Overall"),
            ("F1 Macro", f"{f1_macro:.4f}", "Patokan Utama"),
            ("Precision", f"{report_dict['weighted avg']['precision']:.2%}", "Weighted"),
            ("Recall",    f"{report_dict['weighted avg']['recall']:.2%}",    "Weighted"),
        ]
        for col, (label, val, sub) in zip([c1,c2,c3,c4], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        # ---- CLASSIFICATION REPORT + CM ----
        st.markdown("")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 📋 Classification Report")
            st.code(report_str, language=None)
        with col2:
            st.markdown("#### 🔢 Confusion Matrix")
            st.image(buf_cm, use_container_width=True)

        # ---- F1 PER KELAS CHART ----
        st.markdown('<div class="section-header">📊 Precision / Recall / F1 per Kelas</div>', unsafe_allow_html=True)
        labels_list = ["positive","negative","netral"]
        f1_data = {
            "Kelas":     labels_list,
            "Precision": [report_dict.get(l,{}).get("precision",0) for l in labels_list],
            "Recall":    [report_dict.get(l,{}).get("recall",0)    for l in labels_list],
            "F1-Score":  [report_dict.get(l,{}).get("f1-score",0)  for l in labels_list],
        }
        fig_f1 = go.Figure()
        colors = ["#667eea","#764ba2","#a29bfe"]
        for i, metric in enumerate(["Precision","Recall","F1-Score"]):
            fig_f1.add_trace(go.Bar(
                name=metric, x=f1_data["Kelas"], y=f1_data[metric],
                marker_color=colors[i], text=[f"{v:.2f}" for v in f1_data[metric]],
                textposition="outside"
            ))
        fig_f1.update_layout(
            barmode="group", plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="#333"), yaxis=dict(range=[0,1.15], gridcolor="#eee"),
            legend=dict(bgcolor="white", ),
            height=350, margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_f1, use_container_width=True)

        st.markdown("---")

        # ---- DISTRIBUSI LITERASI ----
        st.markdown('<div class="section-header">📚 Distribusi Literasi Keuangan</div>', unsafe_allow_html=True)
        if len(literacy_dist) > 0:
            cols = st.columns(len(literacy_dist))
            badge_cls = {"high":"lit-badge-high","medium":"lit-badge-medium","low":"lit-badge-low"}
            for i, (level, count) in enumerate(literacy_dist.items()):
                with cols[i]:
                    pct = count / len(df) * 100
                    badge = badge_cls.get(level, "")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label"><span class="{badge}">{level.upper()}</span></div>
                        <div class="metric-value">{count}</div>
                        <div class="metric-sub">{pct:.1f}% dari total</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ---- DETAIL PER LITERASI ----
        st.markdown('<div class="section-header">🔍 Analisis Detail per Tingkat Literasi</div>', unsafe_allow_html=True)
        order = [l for l in ["high","medium","low"] if "literacy_level" in df_valid.columns and l in df_valid["literacy_level"].values]
        for lit_level in order:
            df_lit   = df_valid[df_valid["literacy_level"] == lit_level]
            if len(df_lit) == 0: continue
            badge_html = f'<span class="lit-badge-{lit_level}">{lit_level.upper()}</span>'
            st.markdown(f"#### 📌 Gen Z Literasi {badge_html} — {len(df_lit)} tweet", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                inv_dist = df_lit["investment_type"].value_counts()
                inv_pct  = (inv_dist / len(df_lit) * 100).round(1)
                fig_inv  = px.bar(x=inv_pct.values, y=inv_pct.index, orientation="h",
                                  labels={"x":"Persentase (%)","y":"Jenis Investasi"},
                                  color=inv_pct.values, color_continuous_scale="Blues_r")
                fig_inv.update_layout(
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(color="#333"), showlegend=False, height=280,
                    margin=dict(t=10, b=10), coloraxis_showscale=False
                )
                fig_inv.update_traces(text=[f"{v:.1f}%" for v in inv_pct.values], textposition="outside")
                st.plotly_chart(fig_inv, use_container_width=True)

            with col2:
                st.markdown("**Detail Alasan per Investasi (Top 3):**")
                for inv in inv_dist.index[:3]:
                    df_inv    = df_lit[df_lit["investment_type"] == inv]
                    rsn_dist  = df_inv["investment_reason"].value_counts()
                    rsn_pct   = (rsn_dist / len(df_inv) * 100).round(1)
                    st.markdown(f"**{inv.upper()}** ({len(df_inv)} tweet)")
                    for reason, pct in list(rsn_pct.items())[:3]:
                        st.markdown(f"&nbsp;&nbsp;→ `{reason}`: **{pct:.1f}%**")
            st.markdown("---")

        # ---- INSIGHT OTOMATIS ----
        st.markdown('<div class="section-header">💡 Ringkasan Insight (Narasi Otomatis)</div>', unsafe_allow_html=True)
        for ins in insights:
            st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

        # ---- HEATMAP LITERASI ----
        if buf_lit:
            st.markdown('<div class="section-header">🔥 Heatmap Literasi Keuangan</div>', unsafe_allow_html=True)
            st.image(buf_lit, use_container_width=True)

        st.markdown("---")

        # ---- DOWNLOAD ----
        st.markdown('<div class="section-header">📥 Download Hasil</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            try:
                with open(f"{OUTPUT_DIR}/LAPORAN_ANALISIS.txt","r",encoding="utf-8") as f:
                    st.download_button("📄 Laporan TXT", f.read(), "LAPORAN_ANALISIS.txt", "text/plain", use_container_width=True)
            except: pass
        with d2:
            buf_cm.seek(0)
            st.download_button("🔢 Confusion Matrix", buf_cm, "confusion_matrix.png", "image/png", use_container_width=True)
        with d3:
            if buf_lit:
                buf_lit.seek(0)
                st.download_button("🔥 Heatmap Literasi", buf_lit, "heatmap_literasi.png", "image/png", use_container_width=True)
        with d4:
            try:
                with open(f"{OUTPUT_DIR}/hasil_analisis.xlsx","rb") as f:
                    st.download_button("📊 Excel", f.read(), "hasil_analisis.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                       use_container_width=True)
            except: pass

# ===================== FOOTER =====================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#444; padding:1.5rem 0; font-size:0.85rem; letter-spacing:1px;'>
    SISTEM ANALISIS MINAT INVESTASI GEN Z &nbsp;·&nbsp; Skripsi 2026 &nbsp;·&nbsp; Naive Bayes + NLP
</div>
""", unsafe_allow_html=True)