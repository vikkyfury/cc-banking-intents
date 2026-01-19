from datasets import load_dataset
from collections import Counter
import re

# 1) Load a *small* slice to inspect columns & values
#    We avoid full download here (fast check only).
ds = load_dataset("AIxBlock/92k-real-world-call-center-scripts-english", split="train[:500]")

print("Num rows in sample:", len(ds))
print("\n--- Example keys in first row ---")
print(ds[0].keys())

# 2) Peek at a couple rows to see likely field names
for i in range(3):
    print(f"\n--- Row {i} ---")
    for k, v in ds[i].items():
        # Truncate long fields
        s = str(v)
        if len(s) > 250:
            s = s[:250] + " ..."
        print(f"{k}: {s}")

# 3) Gauge likely metadata fields
#    We try common possibilities mentioned on the dataset card: domain, topic, accent.
candidates = ["domain", "topic", "industry", "category", "accent", "language"]
present = [c for c in candidates if c in ds.column_names]
print("\nLikely metadata columns found:", present)

# 4) If a 'domain' or 'topic' column exists, print its top values
for col in ["domain", "topic", "industry", "category"]:
    if col in ds.column_names:
        ctr = Counter(ds[col])
        print(f"\nTop values in `{col}`:")
        for val, n in ctr.most_common(15):
            print(f"  {val}: {n}")

# 5) If there is a 'transcript' or 'dialog' style field, preview first lines
text_fields_guess = [c for c in ds.column_names if re.search(r"text|transcript|dialog|utterance", c, re.I)]
print("\nPossible text fields:", text_fields_guess)
if text_fields_guess:
    tf = text_fields_guess[0]
    print(f"\nPreview text field `{tf}` (first item):")
    print(str(ds[0][tf])[:1000])
