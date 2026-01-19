from pathlib import Path
import re, json
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
SRC = PROC / "banking_calls_refined.parquet"
OUT = PROC / "training" / "seed_harvest.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

SEEDS = {
  "card_lost_or_stolen": r"\b(lost|stolen)\s+card|\bfreeze\b|\block\b",
  "card_charge_dispute_or_fraud": r"\b(dispute|chargeback|unauthori[sz]ed|fraud)\b",
  "balance_or_credit_limit": r"\b(balance|available balance|credit limit)\b",
  "request_statement_or_document": r"\b(statement|monthly statement|pdf|document)\b",
  "money_transfer_wire_ach_zelle": r"\b(wire|ach|zelle|transfer)\b",
  "online_banking_login_reset": r"\b(password|passcode|login|locked|reset)\b",
  "fees_or_overdraft": r"\b(fee|overdraft|nsf)\b",
  "loan_or_mortgage_info": r"\b(loan|mortgage|refinance|rate)\b",
  "open_new_account": r"\b(open|new)\s+(account)\b",
  "close_account": r"\b(close|closing)\s+(account)\b",
  "card_pin_or_atm_issue": r"\b(pin|atm)\b",
  "direct_deposit_setup_or_issue": r"\b(direct deposit|payroll)\b",
  "bill_pay_or_autopay_issue": r"\b(bill\s?pay|autopay|auto pay|payment)\b"
}
COMPILED = {k: re.compile(v, re.I) for k,v in SEEDS.items()}

def pick_lines(text: str):
    lines = re.split(r"(?<=[\.\?\!])\s+|\n+", text or "")
    return [l.strip() for l in lines if 5 <= len(l.strip()) <= 300]

def main():
    df = pd.read_parquet(SRC)
    out = []
    for _, r in df.iterrows():
        src = r.get("customer_text") or r.get("full_text") or ""
        for line in pick_lines(src):
            for intent, pat in COMPILED.items():
                if pat.search(line):
                    out.append({"intent_id": intent, "utterance": line[:500]})
                    break  # one intent per line
    # dedupe
    seen = set()
    uniq = []
    for x in out:
        key = (x["intent_id"], x["utterance"].lower())
        if key in seen: continue
        seen.add(key)
        uniq.append(x)

    with open(OUT, "w") as f:
        for x in uniq:
            f.write(json.dumps(x) + "\n")
    print("Wrote:", OUT, "| rows:", len(uniq))

if __name__ == "__main__":
    main()
