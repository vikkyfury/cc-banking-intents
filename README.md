# CC Banking Intents

This submodule builds a banking-focused intent dataset from the **AIxBlock/92k-real-world-call-center-scripts-english** corpus. It filters banking calls, discovers intents, curates labels, and exports training/eval sets.

## What You Get

- A cleaned banking-only subset (Parquet)
- Discovered intent clusters and a curated intent catalog
- Gold-answer scaffolds and handoff lists
- Training datasets and eval splits for intent classification / FAQ evaluation

## Repo Layout

```
cc-banking-intents/
├─ configs/
│  ├─ banking_keywords.yaml
│  ├─ speaker_aliases.yaml
│  ├─ intent_mapping_draft.yaml
│  └─ intent_mapping_overrides.yaml
├─ scripts/                   # 01..11 pipeline scripts
└─ notebooks/
```

## Data & Artifacts (Not in Git)

Generated data lives under `data/` (especially `data/processed/`) and is ignored by Git. Run the scripts to recreate locally.

## Prereqs

- Python 3.10+
- `pip install -r requirements.txt` (if you add one) or install: `datasets`, `huggingface_hub`, `pandas`, `numpy`, `scikit-learn`, `orjson`, `pyyaml`, `tqdm`

## Pipeline (Recommended Order)

1) **Inspect schema (optional)**

```bash
python3 scripts/01_sniff_schema.py
```

2) **Quick sample scan (optional)**

```bash
python3 scripts/02_safe_zip_loader.py
```

3) **Build banking subset**

```bash
python3 scripts/03_build_banking_subset.py
```

Outputs: `data/processed/banking_calls.parquet`

4) **EDA + QC**

```bash
python3 scripts/04_eda_and_qc.py
```

5) **Refine banking filter**

```bash
python3 scripts/05_refine_banking_filter.py
```

Outputs: `data/processed/banking_calls_refined.parquet`

6) **Intent discovery (TF‑IDF + clustering)**

```bash
python3 scripts/06_intent_discovery_tfidf.py
```

Outputs: `data/processed/intent_clusters_tfidf.csv/json`

7) **Curate intents**

```bash
python3 scripts/07_curate_intents.py
```

Outputs:
- `configs/intent_mapping_draft.yaml`
- `configs/intent_mapping_overrides.yaml` (edit this)
- `data/processed/intent_catalog.jsonl`

8) **Build gold scaffold**

```bash
python3 scripts/08_build_gold_scaffold.py
```

Outputs:
- `data/processed/gold_answers_todo.csv`
- `data/processed/handoff_intents.csv`

9) **Validate outputs**

```bash
python3 scripts/09_validate_outputs.py
```

10) **Export training data**

```bash
python3 scripts/10_export_training_data.py
```

Outputs:
- `data/processed/training/utterances_answerable.csv`
- `data/processed/training/utterances_answerable.json`
- `data/processed/training/handoff_intents.json`

11) **Optional seed harvest + merge**

```bash
python3 scripts/10b_seed_harvest.py
python3 scripts/10c_merge_seeded.py
python3 scripts/10d_topup_targets.py
```

12) **Build eval sets + reports**

```bash
python3 scripts/11_build_eval_sets.py
python3 scripts/11_quality_report.py
python3 scripts/11_baseline_intent.py
```

13) **Whitelist answerable intents (optional)**

```bash
python3 scripts/11c_enforce_answerable_whitelist.py
```

## Notes

- Keyword filtering and speaker parsing live in `configs/`.
- `intent_mapping_overrides.yaml` is where you rename intents or mark handoff intents.
- If you add more ZIPs, update `ZIP_FILES` in `scripts/03_build_banking_subset.py`.
