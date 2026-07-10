# -*- coding: utf-8 -*-
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from utils_filename import generate_safe_filename

nltk.download("punkt")

input_file = "data/data_genz_only.csv"

df = pd.read_csv(input_file, encoding="utf-8")

df["tokens"] = df["cleaned_text"].astype(str).apply(word_tokenize)

output_file = generate_safe_filename(
    input_file=input_file,
    suffix="tokenized",
    output_dir="data"
)

df.to_csv(output_file, index=False)
print(f"✅ Tokenizing selesai → {output_file}")
