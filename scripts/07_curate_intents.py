from pathlib import Path
import pandas as pd
import json, re, itertools, yaml

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
CONF = BASE / "configs"
OUT = PROC
CONF.mkdir(parents=True, exist_ok=True)

CLUSTERS_CSV = PROC / "intent_clusters_tfidf.csv"
CLUSTERS_JSON = PROC / "intent_clusters_tfidf.json"

# Where you can override names/flags later
OVERRIDE_YAML = CONF / "intent_mapping_overrides.yaml"  # you will edit this (starts empty)

# Draft the first time here; you can diff/edit later
DRAFT_YAML = CONF / "intent_mapping_draft.yaml"
CATALOG_JSONL = PROC / "intent_catalog.jsonl"

# Some normalization helpers for nicer, stable intent ids
def slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    if not name:
        name = "misc"
    return name

def main():
    df = pd.read_csv(CLUSTERS_CSV)
    with open(CLUSTERS_JSON, "r") as f:
        intents = {int(x["cluster_id"]): x for x in json.load(f)}

    # Build draft rows
    rows = []
    for _, r in df.iterrows():
        cid = int(r["cluster_id"])
        size = int(r["size"])
        top_terms = [t.strip() for t in str(r["top_terms"]).split(",")]
        examples = [e.strip() for e in str(r["examples"]).split(" | ") if e.strip()]
        suggested = intents[cid]["suggested_intent"]
        intent_id = f"{slugify(suggested)}__c{cid}"

        rows.append({
            "cluster_id": cid,
            "intent_id": intent_id,
            "suggested_intent": suggested,
            "size": size,
            "top_terms": top_terms,
            "examples": examples[:15],  # keep it readable
            "answerable": True,         # default; you can flip to False in overrides
            "handoff_reason": ""
        })

    # Write a human-readable draft YAML (cluster -> suggested intent)
    draft = {
        "generated_from": str(CLUSTERS_CSV.name),
        "notes": "Edit `configs/intent_mapping_overrides.yaml` to rename intents or mark answerable=False for handoff.",
        "clusters": [
            {
                "cluster_id": r["cluster_id"],
                "intent_id": r["intent_id"],
                "suggested_intent": r["suggested_intent"],
                "size": r["size"],
                "top_terms": r["top_terms"],
                "examples": r["examples"],
                "answerable": r["answerable"],
                "handoff_reason": r["handoff_reason"],
            }
            for r in sorted(rows, key=lambda x: -x["size"])
        ]
    }
    with open(DRAFT_YAML, "w") as f:
        yaml.safe_dump(draft, f, sort_keys=False, width=120)

    # Merge overrides if present
    if OVERRIDE_YAML.exists():
        with open(OVERRIDE_YAML, "r") as f:
            overrides = yaml.safe_load(f) or {}
        override_map = {}
        for item in overrides.get("clusters", []):
            override_map[int(item["cluster_id"])] = item
        # apply
        for r in rows:
            ov = override_map.get(r["cluster_id"])
            if ov:
                if "intent_id" in ov and ov["intent_id"]:
                    r["intent_id"] = slugify(ov["intent_id"])
                if "suggested_intent" in ov and ov["suggested_intent"]:
                    r["suggested_intent"] = ov["suggested_intent"]
                if "answerable" in ov:
                    r["answerable"] = bool(ov["answerable"])
                if "handoff_reason" in ov:
                    r["handoff_reason"] = ov["handoff_reason"]

    # Emit a compact catalog for downstream steps
    with open(CATALOG_JSONL, "w") as f:
        for r in sorted(rows, key=lambda x: -x["size"]):
            rec = {
                "intent_id": r["intent_id"],
                "cluster_id": r["cluster_id"],
                "intent_name": r["suggested_intent"],
                "size": r["size"],
                "answerable": r["answerable"],
                "handoff_reason": r["handoff_reason"],
                "top_terms": r["top_terms"],
                "examples": r["examples"],
            }
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote draft -> {DRAFT_YAML}")
    print(f"Wrote intent catalog -> {CATALOG_JSONL}")
    print("Optional: create/edit overrides in", OVERRIDE_YAML)

if __name__ == "__main__":
    main()
