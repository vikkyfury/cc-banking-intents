import re, io, json, zipfile, logging
from pathlib import Path
from typing import Dict, Any, Iterable, Optional
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import pandas as pd
import orjson

# --- resolve project dirs relative to this file ---
BASE_DIR = Path(__file__).resolve().parents[1]           # cc-banking-intents/
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = BASE_DIR / "logs"

# ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- logging ---
LOG_FILE = LOG_DIR / "zip_load.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# --- config ---
REPO_ID = "AIxBlock/92k-real-world-call-center-scripts-english"
ZIP_FILES = [
    "customer_service_general_inbound.zip",
    # Uncomment more if needed:
    # "home_service_inbound&telecom _outbound.zip",
    # "automotive_and_healthcare_insurance_inbound.zip",
    # "auto_insurance_customer_service_inbound.zip",
]

BANKING_PAT = re.compile(
    r"\b(bank|banking|account|balance|statement|transfer|wire|zelle|ach|"
    r"routing|checking|savings|deposit|overdraft|card|credit|debit|chargeback|"
    r"fraud|dispute|pin|atm|mortgage|loan|password|passcode|online banking)\b",
    flags=re.I
)

def read_json_safe(raw: bytes) -> Optional[Dict[str, Any]]:
    try:
        return orjson.loads(raw)
    except Exception:
        try:
            return json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            return None

def extract_plain_text(example: Dict[str, Any]) -> str:
    for k in ["transcript", "text"]:
        v = example.get(k)
        if isinstance(v, str):
            return v

    for k in ["dialog", "dialogue", "turns", "utterances"]:
        v = example.get(k)
        if isinstance(v, list):
            parts = []
            for item in v:
                if isinstance(item, dict):
                    for tk in ("text", "utterance", "content"):
                        tv = item.get(tk)
                        if isinstance(tv, str):
                            parts.append(tv)
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                return "\n".join(parts)

    strings = []
    for k, v in example.items():
        if isinstance(v, str) and 5 <= len(v) <= 20000:
            strings.append(v)
    return "\n".join(strings)

def iter_zip_json_records(zip_path: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            try:
                with zf.open(name) as f:
                    raw = f.read()
                obj = read_json_safe(raw)
                if not obj:
                    logging.warning(f"SKIP malformed JSON: {name}")
                    continue
                yield obj
            except Exception as e:
                logging.warning(f"SKIP error reading {name}: {e}")
                continue

def main():
    rows = []
    total_skipped = 0

    for zname in ZIP_FILES:
        print(f"Downloading {zname} ...")
        local = hf_hub_download(REPO_ID, filename=zname, repo_type="dataset")
        zpath = Path(local)
        print(f"Scanning {zpath.name} ...")

        for rec in tqdm(iter_zip_json_records(zpath), desc=f"scan {zpath.name}"):
            domain = rec.get("domain") or rec.get("industry") or rec.get("category")
            topic  = rec.get("topic") or rec.get("subtopic")

            text = extract_plain_text(rec)
            if not text:
                total_skipped += 1
                continue

            if (
                BANKING_PAT.search(text)
                or (domain and re.search(r"bank|finance|credit", str(domain), re.I))
                or (topic and re.search(r"bank|finance|credit", str(topic), re.I))
            ):
                rows.append({
                    "source_zip": zpath.name,
                    "domain": domain,
                    "topic": topic,
                    "text": text[:50000]
                })

            if len(rows) >= 1000:
                break

        if len(rows) >= 1000:
            break

    if not rows:
        print("No matches yet — uncomment more ZIPS in ZIP_FILES or widen keywords.")
    else:
        out_path = PROCESSED_DIR / "banking_dev_sample.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"Saved {len(rows)} rows -> {out_path}")
        print("\nTop domains:")
        print(pd.Series([r.get("domain") for r in rows]).value_counts(dropna=False).head(10))
        print("\nTop topics:")
        print(pd.Series([r.get("topic") for r in rows]).value_counts(dropna=False).head(10))

if __name__ == "__main__":
    main()
