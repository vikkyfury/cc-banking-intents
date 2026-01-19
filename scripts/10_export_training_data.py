from pathlib import Path
import json, csv, re
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
CATALOG = PROC / "intent_catalog.jsonl"         # from step 7 (after overrides)
GOLD = PROC / "gold_answers_todo.csv"           # answerable only
OUT_DIR = PROC / "training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimal cleanup for examples
REDACS = re.compile(r"\[(PERSON_NAME|LOCATION|PHONE_NUMBER|EMAIL_ADDRESS|MONEY_AMOUNT|DATE|TIME|OCCUPATION)\]", re.I)

def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = REDACS.sub(lambda m: "{" + m.group(1).lower() + "}", s)  # turn placeholders into slot-like braces
    s = re.sub(r"\s+", " ", s)
    return s

def main():
    # Load catalog
    intents = [json.loads(l) for l in open(CATALOG, "r")]
    # restrict to answerable intents (we’ll still export a separate file listing handoff)
    answerable = [i for i in intents if i.get("answerable", True)]
    handoff =   [i for i in intents if not i.get("answerable", True)]

    # CSV (generic) — utterances per intent
    rows = []
    for it in answerable:
        exs = it.get("examples", [])[:25] or []
        for ex in exs:
            rows.append({
                "intent_id": it["intent_id"],
                "intent_name": it["intent_name"],
                "utterance": clean_text(ex)
            })
    out_csv = OUT_DIR / "utterances_answerable.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # JSON (Lex-friendly-ish): {name, samples:[]}
    pack = []
    for it in answerable:
        exs = [clean_text(x) for x in it.get("examples", [])[:25] if clean_text(x)]
        pack.append({"intent_id": it["intent_id"], "name": it["intent_name"], "samples": exs})
    out_json = OUT_DIR / "utterances_answerable.json"
    json.dump(pack, open(out_json, "w"), indent=2)

    # Handoff/non-answerable list (for routing rules)
    out_handoff = OUT_DIR / "handoff_intents.json"
    json.dump(
        [{"intent_id": i["intent_id"], "name": i["intent_name"], "reason": i.get("handoff_reason","")} for i in handoff],
        open(out_handoff, "w"),
        indent=2
    )

    print("Wrote:", out_csv)
    print("Wrote:", out_json)
    print("Wrote:", out_handoff)

if __name__ == "__main__":
    main()
