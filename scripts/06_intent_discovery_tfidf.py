# cc-banking-intents/scripts/06_intent_discovery_tfidf.py

import re, json, textwrap
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------
# Paths
# -----------------------------
BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
SRC = PROC / "banking_calls_refined.parquet"
OUT_DIR = PROC
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Heuristics for picking "customer-like" lines
# -----------------------------
FIRST_PERSON = re.compile(r"\b(i|i'm|i’ve|i’d|i’ll|my|me|mine|can’t|couldn’t|don’t|won’t)\b", re.I)
BANK_TERMS = re.compile(
    r"\b(account|balance|statement|card|credit|debit|charge|dispute|fraud|pin|atm|"
    r"transfer|wire|ach|zelle|routing|checking|savings|overdraft|fee|interest|loan|mortgage|"
    r"password|login|online banking|mobile app|bill pay|direct deposit)\b", re.I
)

def split_sentences(s: str) -> List[str]:
    if not s:
        return []
    # Simple sentence split (., ?, !) and newlines
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", s)
    out = []
    for p in parts:
        p = p.strip()
        if 5 <= len(p) <= 400:
            out.append(p)
    return out

def pick_customer_like(full_text: str, customer_text: str) -> List[str]:
    # If we already have customer_text, use it
    if customer_text and customer_text.strip():
        return split_sentences(customer_text)
    sents = split_sentences(full_text)
    kept = []
    for s in sents:
        if FIRST_PERSON.search(s) or BANK_TERMS.search(s) or s.endswith("?"):
            kept.append(s)
    # Keep at most 20 short/medium sentences per call
    if len(kept) > 20:
        kept = sorted(kept, key=len)[:20]
    return kept

def build_corpus(df: pd.DataFrame) -> List[str]:
    corpus = []
    for _, r in df.iterrows():
        cust = (r.get("customer_text") or "").strip()
        full = (r.get("full_text") or "").strip()
        lines = pick_customer_like(full, cust)
        corpus.extend(lines)
    # Deduplicate near-identical lines; also drop very short 1–2 word lines
    corpus = list(dict.fromkeys([c for c in corpus if len(c.split()) >= 3]))
    return corpus

def choose_k(X, k_min=10, k_max=28) -> Tuple[int, float]:
    best_k, best_s = None, -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        # If everything collapsed to one label, skip
        if len(set(labels)) < 2:
            continue
        # Silhouette on a sample for speed
        try:
            s = silhouette_score(X, labels, sample_size=min(5000, X.shape[0]), random_state=42)
        except Exception:
            # Fallback to full silhouette if sampling not supported
            s = silhouette_score(X, labels)
        if s > best_s:
            best_s, best_k = s, k
    return best_k, best_s

def top_terms_per_cluster(tfidf: TfidfVectorizer, X, labels, topn=12):
    terms = tfidf.get_feature_names_out()
    out = []
    for c in sorted(set(labels)):
        idx = (labels == c)
        centroid = X[idx].mean(axis=0)
        if hasattr(centroid, "A1"):
            centroid = centroid.A1
        top_idx = np.argsort(centroid)[-topn:][::-1]
        out.append([terms[i] for i in top_idx])
    return out

def main():
    df = pd.read_parquet(SRC)
    print("Loaded refined rows:", len(df))

    corpus = build_corpus(df)
    print("Candidate customer utterances:", len(corpus))
    if len(corpus) < 50:
        print("Too few utterances—consider broadening filters or adding ZIPs.")
        return

    # -----------------------------
    # Vectorize
    # -----------------------------
    custom_stops = list(ENGLISH_STOP_WORDS.union({"uh", "um", "yeah", "yes", "okay", "ok", "right", "thank", "thanks"}))
    # If memory is tight, tweak: min_df=10, max_features=30000
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.6,
        stop_words=custom_stops,   # <-- list (valid)
        max_features=50000
    )
    X = tfidf.fit_transform(corpus)
    print("TF-IDF shape:", X.shape)

    # -----------------------------
    # Choose K and cluster
    # -----------------------------
    k, sil = choose_k(X, k_min=10, k_max=28)
    if not k:
        k = 18
    print(f"Chosen k={k} (silhouette≈{sil:.3f})")

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    # -----------------------------
    # Inspect clusters
    # -----------------------------
    cluster_terms = top_terms_per_cluster(tfidf, X, labels, topn=12)

    rows = []
    for cid in range(k):
        members = [corpus[i] for i in range(len(corpus)) if labels[i] == cid]
        preview = members[:10]
        rows.append({
            "cluster_id": cid,
            "size": len(members),
            "top_terms": ", ".join(cluster_terms[cid]),
            "examples": " | ".join(preview)
        })

    df_out = pd.DataFrame(rows).sort_values("size", ascending=False)
    out_csv = OUT_DIR / "intent_clusters_tfidf.csv"
    df_out.to_csv(out_csv, index=False)
    print("Wrote clusters ->", out_csv)

    # -----------------------------
    # Draft intent names (auto-suggest)
    # -----------------------------
    def suggest_intent(terms: List[str]) -> str:
        t = " ".join(terms[:5])
        rules = [
            (r"lost|stolen.*card|freeze|lock", "card_lost_or_stolen"),
            (r"charge|dispute|fraud|unauthor", "card_charge_dispute_or_fraud"),
            (r"balance|available|limit", "balance_or_credit_limit"),
            (r"statement|document|monthly", "request_statement_or_document"),
            (r"transfer|wire|ach|zelle", "money_transfer_wire_ach_zelle"),
            (r"password|login|reset|locked", "online_banking_login_reset"),
            (r"overdraft|fee|charge", "fees_or_overdraft"),
            (r"mortgage|loan|interest|refinance", "loan_or_mortgage_info"),
            (r"open.*account|new account", "open_new_account"),
            (r"close.*account", "close_account"),
            (r"pin|atm", "card_pin_or_atm_issue"),
            (r"direct deposit|payroll", "direct_deposit_setup_or_issue"),
            (r"bill pay|autopay|payment", "bill_pay_or_autopay_issue"),
            (r"address|update.*info|change.*phone", "profile_or_contact_update"),
        ]
        for pat, name in rules:
            if re.search(pat, t, re.I):
                return name
        return terms[0:2] and "_".join(terms[0:2]) or "misc"

    intents = []
    for cid, terms in enumerate(cluster_terms):
        intents.append({"cluster_id": cid, "suggested_intent": suggest_intent(terms), "top_terms": terms})

    out_json = OUT_DIR / "intent_clusters_tfidf.json"
    with open(out_json, "w") as f:
        json.dump(intents, f, indent=2)
    print("Wrote intent suggestions ->", out_json)

if __name__ == "__main__":
    main()
