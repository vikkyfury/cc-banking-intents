import pandas as pd
import re, textwrap
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

BASE = Path(__file__).resolve().parents[1]
P = BASE / "data" / "processed" / "banking_calls.parquet"

df = pd.read_parquet(P)
print("Rows:", len(df))
print(df[["n_turns","n_customer_turns"]].describe())

# How many have explicit customer_text?
has_cust = (df["customer_text"].str.len().fillna(0) > 0)
print("Has customer_text:", has_cust.mean())

# Length stats (chars)
for col in ["customer_text","full_text"]:
    L = df[col].fillna("").str.len()
    print(f"{col} mean={L.mean():.1f}, median={L.median():.0f}, p90={L.quantile(0.9):.0f}")

# Top unigrams/bigrams to sniff noise
def top_terms(series, n=30, ngram=(1,2), min_df=5, stop_words="english"):
    vec = CountVectorizer(ngram_range=ngram, min_df=min_df, stop_words=stop_words, max_features=5000)
    X = vec.fit_transform(series.values)
    counts = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    tops = sorted(zip(vocab, counts), key=lambda x: x[1], reverse=True)[:n]
    return tops

text = df["customer_text"].where(has_cust, df["full_text"])
tops = top_terms(text, n=50)
print("\nTop tokens/phrases:")
for t,c in tops:
    print(f"{t:30s} {int(c)}")

# Dump a small sample CSV for manual spot-check
out = BASE / "data" / "processed" / "banking_spotcheck_200.csv"
text.sample(200, random_state=42).to_csv(out, index=False, header=["text"])
print("\nWrote spot-check sample ->", out)
