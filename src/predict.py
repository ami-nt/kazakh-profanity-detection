"""Predict with the saved model (no retraining).

    python -m src.predict "Ойын керемет, бірақ қатып қалады"
    python -m src.predict --explain "..."        # show every step from raw text to label
    python -m src.predict --file comments.txt    # one comment per line, CSV output
    python -m src.predict                        # type comments, empty line to quit

In Python:
    from src.predict import load_model, predict
    predict(load_model(), ["comment one", "comment two"])
"""
import argparse
import csv
import sys
from pathlib import Path

import joblib

from .utils import load_config, path

MISSING_MODEL = ("Model file not found at models/model.joblib. Pull it from the repository "
                 "(see README, section 5.3) or train it: python -m src.train")


def load_model(model_path: str | Path | None = None):
    cfg = load_config()
    file = Path(model_path) if model_path else path(cfg, "model_path")
    if not file.exists():
        raise FileNotFoundError(MISSING_MODEL)
    return joblib.load(file)


def predict(model, texts: list[str]) -> list[dict]:
    probs = model.predict_proba(texts)
    return [{"text": t, "label": int(p >= model.threshold), "probability": round(float(p), 4),
             "threshold": round(model.threshold, 4)} for t, p in zip(texts, probs)]


def show(result: dict) -> None:
    verdict = "1 = profanity" if result["label"] else "0 = no profanity"
    print(f"{verdict}   P(profane) = {result['probability']:.3f}   (threshold {result['threshold']:.3f})")


def show_explanation(e: dict) -> None:
    print(f"1. raw text           : {e['raw']}")
    print(f"2. after PII scrub    : {e['after_pii_scrub']}")
    print(f"3. normalized         : {e['normalized']}")
    if len(e["windows"]) > 1:
        print(f"4. windows            : {len(e['windows'])} (scoring the highest one)")
        print(f"   scored window      : {e['scored_window']}")
    else:
        print("4. windows            : 1 (whole comment)")
    print(f"5. character n-grams  : {e['distinct_ngrams_in_window']} distinct, "
          f"{e['ngrams_known_from_training']} known from training (others are ignored)")
    print("6. largest contributions (tf-idf x weight):")
    for c in e["top_contributions"]:
        print(f"     {c['ngram']!r:12} tfidf {c['tfidf']:.3f} x weight {c['weight']:+.3f} = {c['contribution']:+.3f}")
    print(f"7. score = sum of contributions + intercept ({e['intercept']:+.3f}) = {e['score']:+.3f}")
    print(f"8. probability = sigmoid(score) = {e['probability']:.3f}")
    print(f"9. label = 1 if probability >= {e['threshold']:.3f}  ->  {e['label']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="comment(s) to classify")
    ap.add_argument("--file", type=Path, help="text file, one comment per line")
    ap.add_argument("--explain", action="store_true", help="show every step for each comment")
    ap.add_argument("--model", type=Path, help="path to model.joblib (default from config.yaml)")
    args = ap.parse_args()
    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        sys.exit(str(e))

    if args.file:
        texts = [line.rstrip("\n") for line in args.file.open(encoding="utf-8") if line.strip()]
        w = csv.DictWriter(sys.stdout, fieldnames=["text", "label", "probability", "threshold"])
        w.writeheader()
        w.writerows(predict(model, texts))
        return
    texts = [" ".join(args.text)] if args.text else []
    if not texts:
        print("Type a comment and press Enter (empty line to quit).")
        while True:
            try:
                t = input("> ").strip()
            except EOFError:
                break
            if not t:
                break
            show(predict(model, [t])[0])
            if args.explain:
                show_explanation(model.explain(t))
        return
    for t in texts:
        show(predict(model, [t])[0])
        if args.explain:
            show_explanation(model.explain(t))


if __name__ == "__main__":
    main()
