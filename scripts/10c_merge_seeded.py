from pathlib import Path
import re, json, csv
import pandas as pd
from collections import defaultdict, Counter

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
TRAIN = PROC / "training"
TRAIN.mkdir(parents=True, exist_ok=True)

SEED_PATH = TRAIN / "seed_harvest.jsonl"  # <- inside data/processed/training/
# If you ran 10b in a different place, point SEED_PATH to it:
# SEED_PATH = Path("<ABSOLUTE_PATH_TO>/fdf6a531-d69c-4ad0-be9f-472a91f6a201.jsonl")

IN_CSV  = TRAIN / "utterances_answerable.csv"     # from Step 10
IN_JSON = TRAIN / "utterances_answerable.json"
OUT_CSV = TRAIN / "utterances_answerable.merged.csv"
OUT_JSON = TRAIN / "utterances_answerable.merged.json"

# --- Tight, intent-specific relevance filters to trim false positives ---
STRICT = {
    "card_lost_or_stolen": r"\b(lost|stolen|freeze|locked|lock|block)\b.*\b(card)\b|\b(card)\b.*\b(lost|stolen|freeze|locked|lock|block)\b",
    "card_charge_dispute_or_fraud": r"\b(dispute|charge ?back|unauthori[sz]ed|fraud)\b",
    "balance_or_credit_limit": r"\b(balance|available\s+balance|credit\s+limit|limit\s+increase)\b",
    "request_statement_or_document": r"\b(statement|monthly\s+statement|pdf|document|download|mail\s+(me|it))\b",
    "money_transfer_wire_ach_zelle": r"\b(wire|ach|zelle|transfer|send\s+money)\b",
    "online_banking_login_reset": r"\b(password|passcode|login|log[\s-]?in|locked|reset|unlock)\b",
    "fees_or_overdraft": r"\b(fee|overdraft|nsf|insufficient\s+funds|maintenance\s+fee|charge\s+fee)\b",
    "loan_or_mortgage_info": r"\b(loan|mortgage|refinance|refi|rate|interest\s+rate|apr|pre[-\s]?approval)\b",
    "open_new_account": r"\b(open|opening)\s+(a\s+)?(checking|savings|account)\b",
    "close_account": r"\b(close|closing|terminate|cancel)\s+(the\s+)?(account)\b",
    "card_pin_or_atm_issue": r"\b(pin|atm|cash\s+machine|withdraw(al)?)\b",
    "direct_deposit_setup_or_issue": r"\b(direct\s+deposit|payroll|routing\s+number)\b",
    "bill_pay_or_autopay_issue": r"\b(bill\s*pay|auto\s*pay|autopay|payment|pay\s+bill|schedule\s+payment)\b",
}

COMPILED = {k: re.compile(v, re.I) for k, v in STRICT.items()}

# Normalize placeholders to slot-like braces
REDACS = re.compile(r"\[(PERSON_NAME|LOCATION|PHONE_NUMBER|EMAIL_ADDRESS|MONEY_AMOUNT|DATE|TIME|OCCUPATION)\]", re.I)
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = REDACS.sub(lambda m: "{" + m.group(1).lower() + "}", s)
    s = re.sub(r"\s+", " ", s)
    return s

def pass_strict(intent: str, text: str) -> bool:
    pat = COMPILED.get(intent)
    if not pat:
        return True
    return bool(pat.search(text))

def main():
    # Load existing answerable utterances (if present)
    if IN_CSV.exists():
        base_df = pd.read_csv(IN_CSV)
    else:
        base_df = pd.DataFrame(columns=["intent_id", "intent_name", "utterance"])

    # Load seeds
    harvested = []
    if SEED_PATH.exists():
        with open(SEED_PATH, "r") as f:
            for line in f:
                rec = json.loads(line)
                harvested.append(rec)
    else:
        print("No seed file found:", SEED_PATH)
        return

    # Normalize and filter seeds
    cleaned = defaultdict(list)
    for rec in harvested:
        iid = rec.get("intent_id")
        utt = normalize_text(rec.get("utterance", ""))
        if not iid or len(utt.split()) < 3:
            continue
        if pass_strict(iid, utt):
            cleaned[iid].append(utt)

    # Deduplicate per intent (case-insensitive)
    for iid in list(cleaned.keys()):
        seen = set()
        unique = []
        for u in cleaned[iid]:
            key = u.lower()
            if key not in seen:
                seen.add(key)
                unique.append(u)
        cleaned[iid] = unique

    # Build combined set (existing + seeds), with balancing
    combined = defaultdict(set)
    intent_names = {r.intent_id: r.intent_name for r in base_df.itertuples(index=False)} if len(base_df) else {}

    for r in base_df.itertuples(index=False):
        combined[r.intent_id].add(normalize_text(r.utterance))

    # target counts per intent
    MIN_PER_INTENT = 50
    MAX_PER_INTENT = 150

    # Pull in seeds up to cap
    for iid, utterances in cleaned.items():
        current = len(combined[iid])
        room = max(0, MAX_PER_INTENT - current)
        if room > 0:
            # prefer more diverse mix: keep questions, commands, statements
            qs = [u for u in utterances if u.endswith("?")]
            st = [u for u in utterances if not u.endswith("?")]
            take = []

            # ensure at least ~30% questions if available
            q_take = min(len(qs), max(0, int(0.3 * MAX_PER_INTENT) - sum(u.endswith("?") for u in combined[iid])))
            take.extend(qs[:q_take])
            # fill rest with statements
            rest = room - len(take)
            if rest > 0:
                take.extend(st[:rest])

            for u in take:
                combined[iid].add(u)

    # Flatten to rows
    rows = []
    for iid, utts in combined.items():
        if not utts:
            continue
        name = intent_names.get(iid, iid)
        # trim to min/max
        keep = list(utts)[:MAX_PER_INTENT]
        # if below minimum and we have more seeds, top up
        if len(keep) < MIN_PER_INTENT and iid in cleaned:
            extra = [u for u in cleaned[iid] if u not in keep]
            keep.extend(extra[:(MIN_PER_INTENT - len(keep))])
        for u in keep:
            rows.append({"intent_id": iid, "intent_name": name, "utterance": u})

    out_df = pd.DataFrame(rows).sort_values(["intent_id", "utterance"])
    out_df.to_csv(OUT_CSV, index=False)

    # Write JSON pack grouped by intent
    pack = []
    for iid, g in out_df.groupby("intent_id"):
        name = g["intent_name"].iloc[0]
        samples = g["utterance"].tolist()
        pack.append({"intent_id": iid, "name": name, "samples": samples})
    with open(OUT_JSON, "w") as f:
        json.dump(pack, f, indent=2)

    # Print per-intent counts
    counts = out_df.groupby("intent_id")["utterance"].count().sort_values(ascending=False)
    print("Saved:", OUT_CSV)
    print("Saved:", OUT_JSON)
    print("\nPer-intent counts after merge:")
    print(counts.to_string())

if __name__ == "__main__":
    main()
