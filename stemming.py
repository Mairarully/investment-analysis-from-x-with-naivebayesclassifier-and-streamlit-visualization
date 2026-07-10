# -*- coding: utf-8 -*-
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from utils_filename import generate_safe_filename

factory = StemmerFactory()
stemmer = factory.create_stemmer()

input_file = "data/data_genz_only_tokenized_stopword_removed.csv"

df = pd.read_csv(input_file)

df["tokens_stem"] = df["tokens_clean"].apply(
    lambda x: [stemmer.stem(t) for t in eval(x)]
)

output_file = generate_safe_filename(
    input_file=input_file,
    suffix="stemmed",
    output_dir="data"
)

df.to_csv(output_file, index=False)
print(f"✅ Stemming selesai → {output_file}")
