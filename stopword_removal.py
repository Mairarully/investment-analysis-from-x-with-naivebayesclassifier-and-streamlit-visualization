# -*- coding: utf-8 -*-
import pandas as pd
from nltk.corpus import stopwords
import nltk
from utils_filename import generate_safe_filename

nltk.download("stopwords")

stop_words = set(stopwords.words("indonesian"))

input_file = "data/data_genz_only_tokenized.csv"

df = pd.read_csv(input_file)

df["tokens_clean"] = df["tokens"].apply(
    lambda x: [t for t in eval(x) if t not in stop_words and len(t) > 2]
)

output_file = generate_safe_filename(
    input_file=input_file,
    suffix="stopword_removed",
    output_dir="data"
)

df.to_csv(output_file, index=False)
print(f"✅ Stopword removal selesai → {output_file}")
