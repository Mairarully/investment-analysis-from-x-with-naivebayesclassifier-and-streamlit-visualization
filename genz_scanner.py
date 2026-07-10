import pandas as pd
import re
from pathlib import Path

# =========================
# UTILS: AUTO VERSIONING
# =========================
def generate_safe_filename(input_file, suffix, output_dir=None, ext=".csv"):
    """
    Membuat nama file output yang aman (tidak overwrite)
    """
    input_path = Path(input_file)
    base_name = input_path.stem
    output_dir = Path(output_dir) if output_dir else input_path.parent
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{base_name}_{suffix}{ext}"

    counter = 1
    while output_file.exists():
        counter += 1
        output_file = output_dir / f"{base_name}_{suffix}_v{counter}{ext}"

    return output_file


# =========================
# 1. KAMUS & POLA (DIPERLUAS)
# =========================

GENZ_SLANG = {
    # Kata ganti & basic
    "gue","gua","lu","lo","gw","elo","sih","dong","kok","emg","emang","cuy","bruh",
    "kuy","bro","sis","beb","sayang","say","guys","dude",
    
    # Ekspresif emotional
    "anjir","anjay","anjrit","bjir","anjrot","anying","wkwk","wkwkwk","lol","lmao",
    "ngakak","bengek","buset","parah","gokil","ngeri","mantap","bjirr","ngab",
    "asw","astaga","astaghfirullah","aduh","waduh","ampun","innalillahi","awokwok",
    "haha","hehe","xixi","huhu","hiks","uwu","owo",
    
    # Slang modern lifestyle
    "gas","gaskeun","pol","pols","spill","tea","relate","relatable","flexing","flex",
    "healing","selfhealing","fomo","mager","rebahan","santuy","santai","chill","vibing",
    "overthinking","overthink","ghosting","zonk","bucin","baper","gabut","males",
    "redflag","greenflag","yellowflag","lowkey","highkey","slay","slaying","serve",
    "valid","vibes","vibe","salty","savage","rizz","rizzler","auto","skip","bestie",
    "ngegas","ngeship","gaspol","ngehe","kesel","sebel","bete","badmood","mood",
    
    # Adverb & filler Gen Z  
    "jujurly","literally","basically","obviously","definitely","probably","honestly",
    "fr","frfr","ong","nocap","periodt","purr","ate","snatched","bussin","sheesh",
    
    # Twitter/sosmed culture
    "fess","menfess","oomf","moots","mutual","mutuals","rt","qrt","fav","unfollow",
    "softblock","hardsell","softsell","timeline","tl","tweet","twt","dm","dtf",
    "fb","ig","tiktok","fyp","fy","viral","trending","update","story","stories",
    
    # Singkatan super umum
    "yg","ya","utk","ga","gak","gk","nggak","dg","dgn","sm","sama","kalo","klo",
    "bgt","bat","baget","jd","jadi","pdhl","padahal","udh","udah","dah","sdh","sudah",
    "tdk","tidak","blm","belum","blom","aja","aj","doang","gitu","gt","gtt","gmn",
    "gimana","gmana","knp","kenapa","knapa","kok","trus","terus","trs","tapi","tp",
    "jg","juga","lg","lagi","lgi","sm","ama","dr","dari","ke","krn","karena","krna",
    "nih","ni","tuh","tu","kek","kyk","kayak","kyak","bkn","bukan","mkn","makan",
    "mnm","minum","org","orang","tmn","teman","temen","tau","tahu","tw","ngomel","curhat",
    
    # Extra common words
    "emosi","ngamuk","marah","seneng","senang","sedih","nangis","sensi","triggered",
    "cringe","awkward","random","chaos","chaotic","messy","drama","tea","ghibah",
    "ngerasa","berasa","rasanya","soalnya","makanya","padahal","sumpah","serius",
    "beneran","banget","pokoknya","intinya","sebenernya","sebenarnya","gimana"
}

GENZ_CONTEXT = {
    # Finance Gen Z
    "cuan","loss","nyangkut","serok","profit","rugi","untung","trading","trader",
    "crypto","nft","bitcoin","ethereum","reksadana","bibit","ajaib","bareksa",
    "saham","portfolio","investasi","dividen","modal","nabung","menabung","dca",
    
    # Kuliah/Karir/Hustle
    "kuliah","kampus","mahasiswa","mahasiswi","maba","mabaru","semester","skripsi",
    "dospem","dosbing","ujian","uts","uas","kuis","tugas","deadline","sidang",
    "magang","internship","intern","fresh graduate","freshgrad","fg","jobseeker",
    "loker","lowongan","karir","resign","hiring","interview","cv","hrd",
    
    # Kehidupan Gen Z
    "anak kos","kosan","kost","ngekos","kontrakan","boarding","ngampus",
    "organisasi","ormawa","bem","ukm","kepanitiaan","volunteer",
    
    # Hobi & Entertainment Gen Z
    "nonton","streaming","netflix","spotify","playlist","series","drakor","anime",
    "webtoon","manhwa","manga","game","gaming","gamers","main","mabar",
    "concert","konser","festival","event","fanbase","fandom","idol","bias",
    
    # Lifestyle Gen Z
    "thrift","thrifting","secondhand","preloved","vintage","cafe","coffee",
    "aesthetic","outfit","ootd","skincare","makeup","glow","glowup"
}

# Slang saham Gen Z (bonus scoring)
STOCK_GENZ_SLANG = {
    "hodl","hold","averaging","cutloss","tp","profit","realized","unrealized",
    "breakout","support","resisten","teknical","fundamental","chart","candle",
    "momentum","bullish","bearish","sideways","rocket","moon","lambo","rekt",
    "bag","bagholder","dip","buy the dip","btd","fomo buying","panic sell",
    "diamond hands","paper hands","apes","together","strong","gorilla",
    "porto","pf","watchlist","entry","exit","newbie","pemula"
}

FORMAL_WORDS = {
    "idx","ihsg","perseroan","emiten","laporan","keuangan",
    "pemerintah","menteri","regulasi"
}

GENZ_PHRASES = [
    r"anak kos",
    r"fresh graduate",
    r"anak kuliah",
    r"mahasiswa aktif",
    r"lagi kuliah",
    r"lg magang",
    r"nyari kerja",
    r"ngerjain tugas",
    r"bikin skripsi",
    r"baru lulus",
    r"baru masuk kerja",
    r"first job",
    r"gaji pertama",
    r"anak muda",
    r"generasi muda",
    r"anak gen z",
    r"gen z",
    r"zoomer"
]

GENZ_EMOJIS = r"[😭💀🤡🔥💯🥺✨💅🤌👁️👄🫶❤️💔🥰😩🤩😎🤪🙃😤😳🫠]"

# =========================
# 2. FUNGSI SCORING (IMPROVED)
# =========================
def detect_genz_score(text):
    if not isinstance(text, str):
        return 0

    text = text.lower()
    score = 0

    # Frasa kuat
    for p in GENZ_PHRASES:
        if re.search(p, text):
            score += 2

    # Emoji Gen Z
    if re.search(GENZ_EMOJIS, text):
        score += 1

    words = re.findall(r"\b\w+\b", text)

    # Slang
    for w in words:
        if w in GENZ_SLANG:
            score += 1

    # Konteks Gen Z (investasi / kuliah)
    for w in words:
        if w in GENZ_CONTEXT:
            score += 1

    # Penalti bahasa formal (DIKURANGI: -2 → -1)
    for w in words:
        if w in FORMAL_WORDS:
            score -= 1

    return score


# =========================
# 3. MAIN
# =========================
if __name__ == "__main__":

    input_file = "clean_data_mentah.csv"
    
    # Auto-versioning output
    output_file = generate_safe_filename(
        input_file, 
        suffix="genz_scored",
        output_dir="data"
    )

    df = pd.read_csv(input_file)

    if "cleaned_text" not in df.columns:
        raise ValueError("Kolom 'cleaned_text' tidak ditemukan")

    print("🔍 Scanning Gen Z (rule-based scoring - IMPROVED)...")

    df["genz_score"] = df["cleaned_text"].apply(detect_genz_score)

    # THRESHOLD DITURUNKAN: >= 2 → >= 1
    df["genz_label"] = df["genz_score"].apply(
        lambda x: "gen_z" if x >= 1 else "non_gen_z"
    )

    print("\n📊 Distribusi skor:")
    print(df["genz_score"].value_counts().sort_index())
    
    print("\n📊 Distribusi label:")
    print(df["genz_label"].value_counts())

    df.to_csv(output_file, index=False)
    print(f"\n✅ Selesai → {output_file}")