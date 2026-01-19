import re, io, json, zipfile, logging, hashlib, textwrap
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, List, Tuple
import yaml
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import pandas as pd
import orjson

# --- resolve project dirs ---
BASE_DIR = Path(__file__).resolve().parents[1]           # cc-banking-intents/
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "configs"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- logging ---
LOG_FILE = LOG_DIR / "build_banking_subset.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# --- config ---
REPO_ID = "AIxBlock/92k-real-world-call-center-scripts-english"

# Start with a reasonable spread; you can add more as needed.
ZIP_FILES = [
    "customer_service_general_inbound.zip",
    "home_service_inbound.zip",
    "home_ervice_inbound&telecom _outbound.zip",   # (sic) filename includes a space & ampersand
]

# conservative banking keyword list (for any text, esp. customer turns)
BANKING_PAT = re.compile(
    r"\b(bank|banking|account|balance|statement|transfer|wire|zelle|ach|"
    r"routing|checking|savings|deposit|overdraft|card|credit|debit|chargeback|"
    r"fraud|dispute|pin|atm|mortgage|loan|password|passcode|online banking|"
    r"payment|autopay|direct deposit|interest|fee|late fee|billing)\b",
    flags=re.I
)

# --- helper: robust JSON parsing ---
def read_json_safe(raw: bytes) -> Optional[Dict[str, Any]]:
    try:
        return orjson.loads(raw)
    except Exception:
        try:
            return json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            return None

# --- load speaker aliases ---
def load_aliases() -> Tuple[List[str], List[str]]:
    path = CONFIG_DIR / "speaker_aliases.yaml"
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    cust = [s.lower() for s in y.get("customer", [])]
    agent = [s.lower() for s in y.get("agent", [])]
    return cust, agent

CUSTOMER_ALIASES, AGENT_ALIASES = load_aliases()

def label_is_customer(val: Optional[str]) -> Optional[bool]:
    if not val:
        return None
    v = str(val).strip().lower()
    # raw exact match
    if v in CUSTOMER_ALIASES: return True
    if v in AGENT_ALIASES: return False
    # fuzzy contains
    if any(a in v for a in CUSTOMER_ALIASES): return True
    if any(a in v for a in AGENT_ALIASES): return False
    # common roles
    if v in {"customer", "user", "caller", "client", "member"}: return True
    if v in {"agent", "rep", "csr", "advisor", "associate", "operator"}: return False
    return None

# --- normalize whitespace ---
WS_PAT = re.compile(r"[ \t\u00A0]+")
def clean_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)           # collapse excessive newlines
    s = WS_PAT.sub(" ", s)
    return s.strip()

# --- extract turns & customer-only text ---
def extract_turns(example: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str, str]:
    """
    Returns (turns, customer_text, agent_text)
    Each turn: {"speaker": "...", "text": "..."}
    """
    turns: List[Dict[str, str]] = []

    # Patterns seen in these corpora:
    for k in ["turns", "dialog", "dialogue", "utterances"]:
        v = example.get(k)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    spk = item.get("speaker") or item.get("role") or item.get("spk")
                    txt = item.get("text") or item.get("utterance") or item.get("content")
                    if isinstance(txt, str) and txt.strip():
                        turns.append({"speaker": str(spk) if spk is not None else "", "text": clean_text(txt)})
                elif isinstance(item, str) and item.strip():
                    # speaker unknown
                    turns.append({"speaker": "", "text": clean_text(item)})
            break

    # Single big transcript fallback
    if not turns:
        for k in ["transcript", "text"]:
            v = example.get(k)
            if isinstance(v, str) and v.strip():
                # naive split by lines; no speaker info
                chunks = [clean_text(x) for x in v.split("\n") if clean_text(x)]
                turns = [{"speaker": "", "text": c} for c in chunks[:5000]]
                break

    # Split into customer/agent buckets when we can infer speaker
    cust_lines, agent_lines = [], []
    for t in turns:
        is_c = label_is_customer(t.get("speaker"))
        if is_c is True:
            cust_lines.append(t["text"])
        elif is_c is False:
            agent_lines.append(t["text"])
        else:
            # unknown speaker; skip for “customer-only” set but keep turn in full record
            pass

    customer_text = clean_text("\n".join(cust_lines))
    agent_text = clean_text("\n".join(agent_lines))
    return turns, customer_text, agent_text

def iter_zip_json_records(zip_path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
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
                yield name, obj
            except Exception as e:
                logging.warning(f"SKIP error reading {name}: {e}")
                continue

def main():
    rows = []
    seen_hashes = set()
    total_checked, total_kept = 0, 0

    for zname in ZIP_FILES:
        print(f"Downloading {zname} ...")
        local = hf_hub_download(REPO_ID, filename=zname, repo_type="dataset")
        zpath = Path(local)
        print(f"Scanning {zpath.name} ...")

        for fname, rec in tqdm(iter_zip_json_records(zpath), desc=f"scan {zpath.name}"):
            total_checked += 1

            # optional metadata
            domain = rec.get("domain") or rec.get("industry") or rec.get("category")
            topic  = rec.get("topic") or rec.get("subtopic")

            turns, customer_text, agent_text = extract_turns(rec)
            full_text = clean_text("\n".join(t["text"] for t in turns))

            if not full_text:
                continue

            # Filtering: prefer customer text for banking detection; fallback to full text
            hay = customer_text if customer_text else full_text
            if not BANKING_PAT.search(hay):
                continue

            # Basic dedupe by hash of customer text (or full text if empty)
            basis = customer_text if customer_text else full_text
            h = hashlib.sha256(basis.encode("utf-8", errors="ignore")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            rows.append({
                "source_zip": zpath.name,
                "file_name": fname,
                "domain": domain,
                "topic": topic,
                "n_turns": len(turns),
                "n_customer_turns": sum(1 for t in turns if label_is_customer(t.get("speaker")) is True),
                "customer_text": customer_text,
                "agent_text": agent_text,
                "full_text": full_text,
                "hash": h
            })
            total_kept += 1

    if not rows:
        print("No banking rows found. Consider adding more ZIPs or broadening keywords.")
        return

    df = pd.DataFrame(rows)

    # Save full subset
    out_full = PROCESSED_DIR / "banking_calls.parquet"
    df.to_parquet(out_full, index=False)

    # Make a quick 1k dev sample (or all if fewer)
    sample_n = min(1000, len(df))
    df.sample(sample_n, random_state=42).to_parquet(PROCESSED_DIR / "banking_calls_sample_1k.parquet", index=False)

    print(f"Checked: {total_checked:,} | Kept: {total_kept:,}")
    print(f"Saved full subset -> {out_full} ({len(df)} rows)")
    print("Saved fast dev slice ->", PROCESSED_DIR / "banking_calls_sample_1k.parquet")

    # Quick peek
    print("\nTop source_zip:")
    print(df["source_zip"].value_counts().head(10))
    print("\nExample row (truncated):")
    ex = df.iloc[0]
    print("domain:", ex["domain"], "topic:", ex["topic"], "n_turns:", ex["n_turns"])
    print(textwrap.shorten(ex["customer_text"] or ex["full_text"], width=300, placeholder=" ..."))

if __name__ == "__main__":
    main()
