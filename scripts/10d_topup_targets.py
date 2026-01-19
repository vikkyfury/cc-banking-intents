# cc-banking-intents/scripts/10d_topup_targets.py

from pathlib import Path
import re, json
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
SRC  = PROC / "banking_calls_refined.parquet"           # from step 5
TRAIN_DIR = PROC / "training"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

MERGED_CSV  = TRAIN_DIR / "utterances_answerable.merged.csv"
MERGED_JSON = TRAIN_DIR / "utterances_answerable.merged.json"

TARGETS = {
    # raise these to at least 50 examples
    "billing_zip_verification": 50,
    "branch_or_service_coverage_by_location": 50,
    "profile_or_contact_update": 50,
    "direct_deposit_setup_or_issue": 50,
}

# Intent-specific strict patterns (customer-like phrasings favored)
PATS = {
    "billing_zip_verification": re.compile(
        r"\b(billing\s+zip|postal\s+code|zip\s+code\s+(?:on|for)\s+(?:the\s+)?card|verify\s+(?:my\s+)?zip)\b",
        re.I),
    "branch_or_service_coverage_by_location": re.compile(
        r"\b(branch(?:es)?|bank\s+near\s+me|open\s+hours|hours\s+today|do\s+you\s+service\s+(?:my\s+)?area|"
        r"licensed\s+in\s+{location}|in\s+{location}\b)", re.I),
    "profile_or_contact_update": re.compile(
        r"\b(update|change|edit)\s+(?:my\s+)?(address|email|phone|contact\s+info)\b|\bhow\s+do\s+i\s+update\b",
        re.I),
    "direct_deposit_setup_or_issue": re.compile(
        r"\b(direct\s+deposit|payroll\s+deposit|routing\s+number|account\s+number\s+for\s+deposit|set\s+up\s+direct\s+deposit)\b",
        re.I),
}

# Normalize placeholders to slot-like braces
REDACS = re.compile(r"\[(PERSON_NAME|LOCATION|PHONE_NUMBER|EMAIL_ADDRESS|MONEY_AMOUNT|DATE|TIME|OCCUPATION)\]", re.I)
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = REDACS.sub(lambda m: "{" + m.group(1).lower() + "}", s)
    s = re.sub(r"\s+", " ", s)
    return s

def sentence_split(s: str):
    for part in re.split(r"(?<=[\.\?\!])\s+|\n+", s or ""):
        part = part.strip()
        if 5 <= len(part) <= 300:
            yield part

def main():
    if not MERGED_CSV.exists():
        print("Merged training CSV not found:", MERGED_CSV)
        return

    base_df = pd.read_csv(MERGED_CSV)
    # Current counts
    counts = base_df.groupby("intent_id")["utterance"].count().to_dict()

    # Which intents actually need topup
    need = {iid: tgt for iid, tgt in TARGETS.items() if counts.get(iid, 0) < tgt}
    if not need:
        print("All target intents already meet minimums. Nothing to do.")
        return

    df = pd.read_parquet(SRC)

    # Build a set of existing utterances (casefolded) per intent
    existing = {}
    for iid, g in base_df.groupby("intent_id"):
        existing[iid] = set(u.strip().lower() for u in g["utterance"].astype(str))

    # Harvest more lines
    adds = []
    for _, row in df.iterrows():
        # Prefer customer_text if present; else fallback to full_text
        text = (row.get("customer_text") or row.get("full_text") or "")
        for sent in sentence_split(text):
            norm = normalize_text(sent)
            low = norm.lower()
            # Try each target
            for iid, min_needed in need.items():
                if len([a for a in adds if a["intent_id"] == iid]) + counts.get(iid, 0) >= min_needed:
                    continue  # already satisfied
                pat = PATS[iid]
                if pat.search(norm):
                    if low not in existing.get(iid, set()):
                        # Keep some diversity: avoid purely agent-like prompts
                        if not norm.lower().startswith(("i'll ", "let me ", "i can ", "we can ")):
                            adds.append({"intent_id": iid, "intent_name": iid, "utterance": norm})
                            existing.setdefault(iid, set()).add(low)

    if not adds:
        print("No additional lines matched strict patterns. You can relax patterns in PATS.")
        return

    # Append and save
    out_df = pd.concat([base_df, pd.DataFrame(adds)], ignore_index=True)
    # Re-trim to max 150 per intent, keep first occurrences
    MAX_PER_INTENT = 150
    out_df["key"] = out_df["intent_id"].astype(str) + "||" + out_df["utterance"].astype(str).str.lower().str.strip()
    out_df = out_df.drop_duplicates("key")

    # enforce cap
    out_df["row_ix"] = range(len(out_df))
    capped = []
    for iid, g in out_df.sort_values("row_ix").groupby("intent_id", sort=False):
        capped.append(g.head(MAX_PER_INTENT))
    out_df = pd.concat(capped).drop(columns=["key","row_ix"])

    out_df.to_csv(MERGED_CSV, index=False)

    # Update JSON pack
    pack = []
    for iid, g in out_df.groupby("intent_id"):
        name = g["intent_name"].iloc[0]
        samples = g["utterance"].tolist()
        pack.append({"intent_id": iid, "name": name, "samples": samples})
    with open(MERGED_JSON, "w") as f:
        json.dump(pack, f, indent=2)

    # Print before/after for targets
    after_counts = out_df.groupby("intent_id")["utterance"].count()
    print("Top-up complete. Counts (targets):")
    for iid, tgt in TARGETS.items():
        print(f"- {iid}: {counts.get(iid,0)} -> {int(after_counts.get(iid,0))} (target {tgt})")

if __name__ == "__main__":
    main()
