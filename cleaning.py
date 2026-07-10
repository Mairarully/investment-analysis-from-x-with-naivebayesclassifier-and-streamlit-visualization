import pandas as pd
import re
import string
from pathlib import Path

class TwitterCleaner:
    def __init__(self):
        """Inisialisasi cleaner dengan pattern regex"""
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        self.rt_pattern = re.compile(r'\bRT\b', re.IGNORECASE)
        self.number_pattern = re.compile(r'\d+')
        self.extra_whitespace = re.compile(r'\s+')
        self.repeated_char = re.compile(r'(.)\1{2,}')
        
        # Emoji pattern - comprehensive
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols
            u"\U0001FA00-\U0001FAFF"  # Extended Symbols (termasuk 🫠)
            u"\U00002600-\U000026FF"  # Miscellaneous Symbols
            u"\U00002190-\U000021FF"  # Arrows
            "]+", 
            flags=re.UNICODE
        )
    
    def clean_text(self, text, 
                   remove_urls=True,
                   remove_mentions=True, 
                   remove_hashtags=False,
                   keep_hashtag_text=True,
                   remove_rt=True,
                   remove_emoji=True,
                   remove_numbers=False,
                   remove_punctuation=True,
                   normalize_repeated_chars=True,
                   min_length=3):
        """Membersihkan teks tweet"""
        
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # 1. Hapus URL
        if remove_urls:
            text = self.url_pattern.sub('', text)
        
        # 2. Hapus RT marker
        if remove_rt:
            text = self.rt_pattern.sub('', text)
        
        # 3. Hapus mention
        if remove_mentions:
            text = self.mention_pattern.sub('', text)
        
        # 4. Handle hashtag
        if remove_hashtags:
            text = self.hashtag_pattern.sub('', text)
        elif keep_hashtag_text:
            text = self.hashtag_pattern.sub(r'\1', text)
        
        # 5. Hapus emoji
        if remove_emoji:
            text = self.emoji_pattern.sub('', text)
        
        # 6. Normalisasi karakter berulang
        if normalize_repeated_chars:
            text = self.repeated_char.sub(r'\1\1', text)
        
        # 7. Case folding (lowercase)
        text = text.lower()
        
        # 8. Hapus angka (opsional)
        if remove_numbers:
            text = self.number_pattern.sub('', text)
        
        # 9. Hapus tanda baca
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 10. Hapus whitespace berlebih
        text = self.extra_whitespace.sub(' ', text).strip()
        
        # 11. Filter berdasarkan panjang minimum
        if len(text) < min_length:
            return ""
        
        return text


def clean_single_file(input_file, 
                     output_file=None,
                     text_column='text',
                     **cleaning_params):
    """
    Fungsi untuk membersihkan SATU file
    
    Parameters:
    -----------
    input_file : str
        Path ke file CSV input
    output_file : str, optional
        Path ke file CSV output
    text_column : str
        Nama kolom yang berisi teks tweet
    **cleaning_params : dict
        Parameter untuk cleaning
    
    Returns:
    --------
    pd.DataFrame : DataFrame yang sudah dibersihkan
    dict : Statistik cleaning
    """
    
    print(f"\n{'='*70}")
    print(f"📂 Memproses: {input_file}")
    print('='*70)
    
    # Baca file CSV dengan encoding yang benar
    try:
        # Coba baca dengan UTF-8 terlebih dahulu
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        # Jika gagal, coba dengan latin1
        print("⚠️  UTF-8 gagal, mencoba dengan latin1...")
        df = pd.read_csv(input_file, encoding='latin1')
    except Exception as e:
        print(f"❌ Error membaca file: {e}")
        return None, None
    
    # Statistik awal
    total_rows = len(df)
    print(f"📊 Total data awal: {total_rows}")
    
    # Cek apakah kolom text ada
    if text_column not in df.columns:
        print(f"❌ Kolom '{text_column}' tidak ditemukan!")
        print(f"   Kolom yang tersedia: {list(df.columns)}")
        return None, None
    
    # Inisialisasi cleaner
    cleaner = TwitterCleaner()
    
    # Buat kolom baru untuk teks yang sudah dibersihkan
    print("🧹 Membersihkan teks...")
    df['cleaned_text'] = df[text_column].apply(
        lambda x: cleaner.clean_text(x, **cleaning_params)
    )
    
    # Hapus baris dengan teks kosong setelah cleaning
    df_clean = df[df['cleaned_text'].str.len() > 0].copy()
    
    # Hapus duplikat berdasarkan cleaned_text
    print("🔍 Menghapus duplikat...")
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['cleaned_text'], keep='first')
    after_dedup = len(df_clean)
    
    # Statistik akhir
    rows_after_cleaning = len(df_clean)
    removed_empty = total_rows - len(df[df['cleaned_text'].str.len() > 0])
    removed_duplicates = before_dedup - after_dedup
    
    stats = {
        'total_awal': total_rows,
        'terhapus_kosong': removed_empty,
        'terhapus_duplikat': removed_duplicates,
        'total_bersih': rows_after_cleaning,
        'persentase_tersisa': f"{(rows_after_cleaning/total_rows)*100:.2f}%"
    }
    
    # Generate output filename jika tidak diberikan
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"clean_{input_path.name}"
    
    # Simpan hasil
    try:
        df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ File bersih disimpan: {output_file}")
    except Exception as e:
        print(f"❌ Error menyimpan file: {e}")
        return df_clean, stats
    
    # Print statistik
    print(f"\n📈 Statistik Cleaning:")
    print(f"   Total awal:          {stats['total_awal']}")
    print(f"   Terhapus (kosong):   {stats['terhapus_kosong']}")
    print(f"   Terhapus (duplikat): {stats['terhapus_duplikat']}")
    print(f"   Total bersih:        {stats['total_bersih']}")
    print(f"   Persentase tersisa:  {stats['persentase_tersisa']}")
    
    # Preview hasil
    print(f"\n👀 Preview 5 data pertama:")
    print(df_clean[['text', 'cleaned_text']].head())
    
    return df_clean, stats


# ==============================================================================
# CONTOH PENGGUNAAN
# ==============================================================================

if __name__ == "__main__":
    
    # Proses SATU file
    df_clean, stats = clean_single_file(
        input_file="data_mentah.csv",           # <- GANTI dengan nama file Anda
        output_file="clean_data_mentah.csv",    # <- GANTI dengan nama output
        text_column="text",                # <- Nama kolom teks (biasanya 'text')
        
        # Parameter cleaning (sesuaikan jika perlu)
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=False,
        keep_hashtag_text=True,
        remove_rt=True,
        remove_emoji=True,
        remove_numbers=False,       # Pertahankan angka (penting untuk saham!)
        remove_punctuation=True,
        normalize_repeated_chars=True,
        min_length=3
    )
    
    if df_clean is not None:
        print(f"\n✅ SELESAI! Data bersih: {len(df_clean)} baris")
    else:
        print(f"\n❌ GAGAL! Periksa error di atas")