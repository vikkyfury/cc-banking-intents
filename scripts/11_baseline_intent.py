from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
TRAIN_DIR = PROC / "training"

TRAIN = TRAIN_DIR / "intent_train.csv"
DEV   = TRAIN_DIR / "intent_dev.csv"

def main():
    tr = pd.read_csv(TRAIN)
    dv = pd.read_csv(DEV)

    # Simple baseline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)),
        ("clf", LinearSVC(random_state=42))
    ])
    pipe.fit(tr["utterance"], tr["intent_id"])
    preds = pipe.predict(dv["utterance"])

    print("Accuracy:", round(accuracy_score(dv["intent_id"], preds), 4))
    print("\nPer-class report:")
    print(classification_report(dv["intent_id"], preds, digits=3))

if __name__ == "__main__":
    main()
