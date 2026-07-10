import pandas as pd

# =========================
# KONFIGURASI FILE
# =========================
INPUT_FILE = "data/clean_data_mentah_genz_scored.csv"
OUTPUT_FILE = "data/data_genz_only.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE, encoding="utf-8")

if "genz_score" not in df.columns:
    raise ValueError("Kolom 'genz_score' tidak ditemukan")

# =========================
# FILTER GEN Z
# =========================
df_genz = df[df["genz_score"] > 0].copy()

# =========================
# STATISTIK (OPSIONAL TAPI BAGUS)
# =========================
total_awal = len(df)
total_genz = len(df_genz)
total_non_genz = total_awal - total_genz

print("📊 Statistik Filtering Gen Z")
print(f"Total data awal     : {total_awal}")
print(f"Total Gen Z         : {total_genz}")
print(f"Total Non Gen Z     : {total_non_genz}")

# =========================
# SIMPAN FILE BARU
# =========================
df_genz.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Data Gen Z saja disimpan ke: {OUTPUT_FILE}")
