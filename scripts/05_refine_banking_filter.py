import re, io, json, zipfile, logging, hashlib
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, List, Tuple
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import pandas as pd
import orjson

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
PROC = DATA_DIR / "processed"
LOGS = BASE / "logs"
PROC.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

# Source we already built
SRC = PROC / "banking_calls.parquet"
DF = pd.read_parquet(SRC)

# Positive banking signals (stricter)
POS = re.compile(
    r"\b(bank|banking|debit|credit|card|chargeback|dispute|fraud|"
    r"balance|statement|account|checking|savings|routing|"
    r"wire|zelle|ach|overdraft|atm|pin|mortgage|loan|"
    r"direct deposit|autopay|interest|fee|late fee|bill pay|"
    r"password|passcode|online banking|mobile banking)\b",
    re.I
)

# Negative domain terms (non-banking services)
NEG = re.compile(
    r"\b(router|modem|internet|cable|tv service|technician|installation|"
    r"outage|wifi|wi-fi|bandwidth|plumbing|hvac|gas line|appliance|"
    r"water heater|faucet|drain|leak|maintenance|electricity|utility)\b",
    re.I
)

# Heuristic split for single-blob transcripts with prefixes
LINE_SPLIT = re.compile(r"(?:^|\n)\s*(agent|rep|representative|advisor|associate|operator|support|specialist|staff|csr|customer|user|caller|client|member)\s*[:\-]\s*", re.I)

def improve_customer_text(txt: str) -> Tuple[str, str]:
    """
    If prefixes like 'Agent:'/'Customer:' are present, split and bucket.
    Return (customer_text, agent_text).
    """
    if not txt or not isinstance(txt, str):
        return "", ""

    parts = LINE_SPLIT.split(txt)
    # parts like: [pre, role1, text1, role2, text2, ...]
    if len(parts) < 3:
        return "", ""  # no prefixes found; we'll keep original later

    cust_lines, agent_lines = [], []
    role = None
    buf = ""

    # Rebuild role->text mapping
    it = iter(parts)
    pre = next(it, "")
    while True:
        role = next(it, None)
        seg  = next(it, None)
        if role is None or seg is None:
            break
        r = role.strip().lower()
        if re.search(r"customer|user|caller|client|member", r):
            cust_lines.append(seg.strip())
        else:
            agent_lines.append(seg.strip())

    return ("\n".join(cust_lines).strip(), "\n".join(agent_lines).strip())

def is_banking_text(s: str) -> bool:
    return bool(POS.search(s)) and not bool(NEG.search(s))

def main():
    rows = []
    kept = 0

    for i, r in tqdm(DF.iterrows(), total=len(DF), desc="refine"):
        cust = (r.get("customer_text") or "").strip()
        full = (r.get("full_text") or "").strip()

        # Try to improve extraction if no customer_text but full has prefixes
        if not cust and full:
            c2, a2 = improve_customer_text(full)
            if c2:
                cust = c2

        hay = cust if cust else full
        if not hay:
            continue

        if not is_banking_text(hay):
            continue

        rows.append({
            "source_zip": r["source_zip"],
            "file_name": r["file_name"],
            "customer_text": cust if cust else None,
            "full_text": full if full else None,
        })
        kept += 1

    df2 = pd.DataFrame(rows)
    out = PROC / "banking_calls_refined.parquet"
    df2.to_parquet(out, index=False)
    print(f"Refined kept: {len(df2)} / {len(DF)}")
    print("Saved ->", out)

    # quick length sanity
    lens = df2["customer_text"].fillna(df2["full_text"]).str.len()
    print("Length median:", lens.median(), "p90:", lens.quantile(0.9))

if __name__ == "__main__":
    main()
