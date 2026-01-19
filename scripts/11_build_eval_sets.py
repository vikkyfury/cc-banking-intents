from pathlib import Path
import pandas as pd
import json
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
TRAIN_DIR = PROC / "training"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

MERGED = TRAIN_DIR / "utterances_answerable.merged.csv"
GOLD = PROC / "gold_answers_todo.csv"

OUT_INTENT_TRAIN = TRAIN_DIR / "intent_train.csv"
OUT_INTENT_DEV   = TRAIN_DIR / "intent_dev.csv"
OUT_FAQ_EVAL     = TRAIN_DIR / "faq_eval.jsonl"

def main():
    df = pd.read_csv(MERGED)  # columns: intent_id, intent_name, utterance
    # drop empties
    df = df.dropna(subset=["intent_id","utterance"])
    df["intent_id"] = df["intent_id"].astype(str).str.strip()
    df["utterance"] = df["utterance"].astype(str).str.strip()
    df = df[df["utterance"].str.len() > 2]

    # stratified split
    train_df, dev_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["intent_id"]
    )
    train_df.to_csv(OUT_INTENT_TRAIN, index=False)
    dev_df.to_csv(OUT_INTENT_DEV, index=False)

    # FAQ eval pack from gold answers
    gold = pd.read_csv(GOLD)
    gold = gold.fillna("")
    keep = gold[gold["gold_answer"].str.strip() != ""]
    with open(OUT_FAQ_EVAL, "w") as f:
        for r in keep.itertuples(index=False):
            rec = {
                "intent_id": r.intent_id,
                "question": str(r.sample_question).strip(),
                "gold_answer": str(r.gold_answer).strip(),
                "source_refs": str(r.source_refs).strip(),
                "policy_notes": str(r.policy_notes).strip()
            }
            f.write(json.dumps(rec) + "\n")

    print("Wrote:", OUT_INTENT_TRAIN)
    print("Wrote:", OUT_INTENT_DEV)
    print("Wrote:", OUT_FAQ_EVAL)
    print("Per-intent counts (train):")
    print(train_df.groupby("intent_id").size().sort_values(ascending=False).to_string())
    print("\nPer-intent counts (dev):")
    print(dev_df.groupby("intent_id").size().sort_values(ascending=False).to_string())

if __name__ == "__main__":
    main()
