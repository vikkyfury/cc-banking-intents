from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"

CATALOG_JSONL = PROC / "intent_catalog.jsonl"
OUT_GOLD = PROC / "gold_answers_todo.csv"
OUT_HANDOFF = PROC / "handoff_intents.csv"

def main():
    intents = []
    with open(CATALOG_JSONL, "r") as f:
        for line in f:
            intents.append(json.loads(line))

    # Expand examples to a few sample questions per intent
    gold_rows = []
    handoff_rows = []

    for it in intents:
        exs = it.get("examples", [])[:3] or [""]
        for ex in exs:
            row = {
                "intent_id": it["intent_id"],
                "intent_name": it["intent_name"],
                "cluster_id": it["cluster_id"],
                "sample_question": ex,
            }
            if it.get("answerable", True):
                row.update({
                    "gold_answer": "",
                    "source_refs": "",      # paste KB links/IDs here later
                    "policy_notes": "",     # auth/PII guardrails if any
                })
                gold_rows.append(row)
            else:
                row.update({
                    "handoff_reason": it.get("handoff_reason", ""),
                    "handoff_destination": "human_agent",  # or queue name
                })
                handoff_rows.append(row)

    if gold_rows:
        pd.DataFrame(gold_rows).to_csv(OUT_GOLD, index=False)
        print("Wrote gold-answer scaffold ->", OUT_GOLD)
    if handoff_rows:
        pd.DataFrame(handoff_rows).to_csv(OUT_HANDOFF, index=False)
        print("Wrote handoff list ->", OUT_HANDOFF)

    print("Next: Fill gold_answer & source_refs for answerable intents.")

if __name__ == "__main__":
    main()
