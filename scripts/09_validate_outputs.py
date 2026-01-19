from pathlib import Path
import pandas as pd, json, sys

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"

gold = PROC / "gold_answers_todo.csv"
handoff = PROC / "handoff_intents.csv"
catalog = PROC / "intent_catalog.jsonl"

gold_df = pd.read_csv(gold)
handoff_df = pd.read_csv(handoff)
cat = [json.loads(l) for l in open(catalog)]
ans = {c["intent_id"] for c in cat if c.get("answerable", True)}
non = {c["intent_id"] for c in cat if not c.get("answerable", True)}

ok = True

if {"gold_answer","source_refs","policy_notes"}.issubset(gold_df.columns):
    if not set(gold_df["intent_id"]).issubset(ans):
        print("❌ gold_answers_todo.csv includes non-answerable intents")
        ok = False
else:
    print("❌ gold_answers_todo.csv missing gold-answer columns")
    ok = False

if {"handoff_reason","handoff_destination"}.issubset(handoff_df.columns):
    if not set(handoff_df["intent_id"]).issubset(non):
        print("❌ handoff_intents.csv includes answerable intents")
        ok = False
else:
    print("❌ handoff_intents.csv missing handoff columns")
    ok = False

print("✅ intent count:", len(cat), "| answerable:", len(ans), "| non-answerable:", len(non))
sys.exit(0 if ok else 1)
