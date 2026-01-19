from pathlib import Path
import pandas as pd
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
TRAIN_DIR = PROC / "training"

MERGED = TRAIN_DIR / "utterances_answerable.merged.csv"

# ✅ Whitelist of answerable intents to KEEP
WHITELIST = {
    "balance_or_credit_limit",
    "bill_pay_or_autopay_issue",
    "billing_zip_verification",
    "branch_or_service_coverage_by_location",
    "card_charge_dispute_or_fraud",
    "card_pin_or_atm_issue",
    "direct_deposit_setup_or_issue",
    "fees_or_overdraft",
    "loan_or_mortgage_info",
    "money_transfer_wire_ach_zelle",
    "online_banking_login_reset",
    "profile_or_contact_update",
    "request_statement_or_document",
}

MIN_PER_INTENT = 50   # ensure coverage
MAX_PER_INTENT = 150  # keep balance

def main():
    if not MERGED.exists():
        print("Missing:", MERGED)
        sys.exit(1)

    df = pd.read_csv(MERGED)
    before = len(df)

    # Filter to whitelist
    df = df[df["intent_id"].isin(WHITELIST)].copy()
    # De-dup
    df["key"] = df["intent_id"].astype(str) + "||" + df["utterance"].astype(str).str.lower().str.strip()
    df = df.drop_duplicates("key").drop(columns=["key"])

    # Enforce caps
    capped = []
    for iid, g in df.groupby("intent_id"):
        g = g.head(MAX_PER_INTENT)
        capped.append(g)
    df = pd.concat(capped).reset_index(drop=True)

    # Verify mins (warn if any fall short)
    counts = df.groupby("intent_id")["utterance"].count().to_dict()
    low = {iid:c for iid,c in counts.items() if c < MIN_PER_INTENT}
    if low:
        print("⚠️ These intents are below the minimum:", low)
        print("Run 10d_topup_targets.py to raise them, then re-run this script.")
    else:
        print("All intents meet the minimum of", MIN_PER_INTENT)

    df.to_csv(MERGED, index=False)
    print(f"Saved filtered training set -> {MERGED} | rows {before} -> {len(df)}")

    # Rebuild splits
    ret = subprocess.call([sys.executable, str(BASE / "scripts" / "11_build_eval_sets.py")])
    if ret != 0:
        print("Failed to rebuild eval sets.")
        sys.exit(ret)
    print("Rebuilt train/dev sets.")

if __name__ == "__main__":
    main()
