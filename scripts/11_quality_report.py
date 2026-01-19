from pathlib import Path
import pandas as pd
import re

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
TRAIN_DIR = PROC / "training"

TRAIN = TRAIN_DIR / "intent_train.csv"
DEV   = TRAIN_DIR / "intent_dev.csv"

def main():
    tr = pd.read_csv(TRAIN)
    dv = pd.read_csv(DEV)

    # Basic counts
    print("Train rows:", len(tr), "Dev rows:", len(dv))
    print("\nPer-intent counts (train):")
    print(tr.groupby("intent_id").size().sort_values(ascending=False).to_string())
    print("\nPer-intent counts (dev):")
    print(dv.groupby("intent_id").size().sort_values(ascending=False).to_string())

    # Question/statement mix
    for df, name in [(tr,"train"),(dv,"dev")]:
        df["is_q"] = df["utterance"].astype(str).str.strip().str.endswith("?")
        mix = df.groupby("intent_id")["is_q"].mean().round(3)
        print(f"\nQuestion ratio by intent ({name}):")
        print(mix.to_string())

    # Placeholder check: ensure {} style, no [] leftovers
    square = (tr["utterance"].str.contains(r"\[.+\]", regex=True).sum()
              + dv["utterance"].str.contains(r"\[.+\]", regex=True).sum())
    braces = (tr["utterance"].str.contains(r"\{(person_name|location|phone_number|email_address|money_amount|date|time|occupation)\}", case=False).sum()
              + dv["utterance"].str.contains(r"\{(person_name|location|phone_number|email_address|money_amount|date|time|occupation)\}", case=False).sum())
    print(f"\nPlaceholders — square brackets left: {square}, brace placeholders found: {braces}")

    # Leakage: identical utterance appearing in both train and dev for same intent
    tr["key"] = tr["intent_id"].astype(str) + "||" + tr["utterance"].astype(str).str.lower().str.strip()
    dv["key"] = dv["intent_id"].astype(str) + "||" + dv["utterance"].astype(str).str.lower().str.strip()
    leaked = set(tr["key"]).intersection(set(dv["key"]))
    print("\nLeakage (exact duplicates across splits):", len(leaked))
    if leaked:
        print("Sample leaks:")
        for k in list(leaked)[:10]:
            print(" -", k)

if __name__ == "__main__":
    main()
