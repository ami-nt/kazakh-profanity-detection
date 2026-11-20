"""Evaluate the saved model on the test split. Run this only once.

    python -m src.evaluate            # stops if test results already exist
    python -m src.evaluate --force    # only to reproduce the same numbers

Writes test_metrics.json, confusion_matrix.png, test_slices.csv,
test_predictions.csv and test_errors.csv to results/.
"""
import argparse
import re
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from .plots import confusion_matrix as plot_confusion
from .utils import ROOT, load_config, path, save_json

LAT = re.compile(r"[a-z]{2,}", re.I)
CYR = re.compile(r"[а-яёәғқңөұүһі]", re.I)


def script_of(text: str) -> str:
    lat, cyr = bool(LAT.search(text)), bool(CYR.search(text))
    return "mixed" if lat and cyr else ("latin" if lat else "cyrillic")


def scores(y, pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "positives": int(np.sum(y)), "precision": float(p), "recall": float(r),
            "f1": float(f), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None}


def slice_table(df: pd.DataFrame) -> pd.DataFrame:
    """The same metrics for parts of the test set (domain, stars, script, length, ...)."""
    rows = []
    groups = {
        "domain": df.domain,
        "stars": df.stars.map(lambda s: "1 star" if s == 1 else ("5 stars" if s == 5 else "2-4 stars")),
        "script": df.script,
        "length": np.where(df.n_words > 30, "over 30 words", "30 words or fewer"),
        "pool": df.pool,
        "ambiguous": np.where(df.ambiguous == 1, "marked ambiguous", "clear"),
    }
    for name, key in groups.items():
        for value, sub in df.groupby(key):
            rows.append({"slice": name, "value": value, **scores(sub.label.to_numpy(), sub.pred.to_numpy())})
    pos = df[df.label == 1]
    for value, sub in pos.groupby("prof_lang"):        # only positives here, so only recall is meaningful
        s = scores(sub.label.to_numpy(), sub.pred.to_numpy())
        rows.append({"slice": "profanity language (positives)", "value": value, **s})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite existing test results")
    args = ap.parse_args()
    cfg = load_config()
    res = path(cfg, "results_dir")
    metrics_file = res / "test_metrics.json"
    if metrics_file.exists() and not args.force:
        sys.exit(f"{metrics_file.relative_to(ROOT)} already exists: the test set has been evaluated. "
                 "Settings must not be tuned on it. Use --force only to reproduce the same run.")
    model = joblib.load(path(cfg, "model_path"))
    test = pd.read_csv(path(cfg, "splits_dir") / "test.csv", dtype={"uid": str, "text": str},
                       keep_default_na=False)
    y = test.label.astype(int).to_numpy()
    prob = model.predict_proba(test.text)
    pred = (prob >= model.threshold).astype(int)
    test = test.assign(prob=prob, pred=pred, script=test.text.map(script_of),
                       n_words=test.text.str.split().str.len(),
                       ambiguous=pd.to_numeric(test.ambiguous, errors="coerce").fillna(0).astype(int),
                       stars=pd.to_numeric(test.stars, errors="coerce").astype(int))
    test["outcome"] = np.select([(y == 1) & (pred == 1), (y == 0) & (pred == 1), (y == 1) & (pred == 0)],
                                ["TP", "FP", "FN"], "TN")

    overall = scores(y, pred)
    # The test set is 44% profane. Real reviews have far less profanity, so this shows
    # the precision we would expect at lower shares, with the same recall and FP rate.
    tpr, fpr = overall["recall"], overall["false_positive_rate"] or 0.0
    at_prevalence = {f"{pi:.1%}": round(tpr * pi / (tpr * pi + fpr * (1 - pi)), 4) if tpr * pi + fpr * (1 - pi) else None
                     for pi in (0.005, 0.01, 0.02, 0.05)}
    metrics = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model.describe(),
        "threshold": model.threshold,
        "test_examples": overall["n"], "test_positives": overall["positives"],
        "test_negatives": overall["n"] - overall["positives"],
        "profanity_class": {k: overall[k] for k in ["precision", "recall", "f1"]},
        "confusion_matrix": {"rows_true_cols_pred": [[overall["tn"], overall["fp"]], [overall["fn"], overall["tp"]]],
                             "tn": overall["tn"], "fp": overall["fp"], "fn": overall["fn"], "tp": overall["tp"]},
        "accuracy": float(accuracy_score(y, pred)),
        "false_positive_rate": overall["false_positive_rate"],
        "average_precision": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "expected_precision_at_prevalence": at_prevalence,
        "note": f"Test positives are enriched ({overall['positives'] / max(overall['n'], 1):.0%} of the test set). "
                "Real reviews contain far fewer profane comments; expected_precision_at_prevalence applies "
                "the measured recall and false-positive rate to lower prevalences.",
    }
    res.mkdir(parents=True, exist_ok=True)
    save_json(metrics, metrics_file)
    plot_confusion(metrics["confusion_matrix"]["rows_true_cols_pred"], res / "confusion_matrix.png",
                   f"Test set confusion matrix (n = {overall['n']})")
    slice_table(test).to_csv(res / "test_slices.csv", index=False)
    cols = ["uid", "text", "label", "prob", "pred", "outcome", "domain", "stars", "pool",
            "prof_lang", "ambiguous", "script", "n_words", "note"]
    test[[c for c in cols if c in test.columns]].to_csv(res / "test_predictions.csv", index=False)
    errors = test[test.outcome.isin(["FP", "FN"])].copy()
    errors["top_ngrams"] = errors.text.map(
        lambda t: "; ".join(f"{c['ngram']!r}:{c['contribution']:+.2f}" for c in model.explain(t, 6)["top_contributions"]))
    errors = errors.sort_values(["outcome", "prob"], ascending=[True, False])
    errors[[c for c in cols + ["top_ngrams"] if c in errors.columns]].to_csv(res / "test_errors.csv", index=False)
    print(f"test n={overall['n']} (positives {overall['positives']}) | threshold {model.threshold:.3f}")
    print(f"profanity class: precision {overall['precision']:.3f} recall {overall['recall']:.3f} "
          f"F1 {overall['f1']:.3f} | TN {overall['tn']} FP {overall['fp']} FN {overall['fn']} TP {overall['tp']}")
    print(f"saved results/test_metrics.json, confusion_matrix.png, test_slices.csv, "
          f"test_predictions.csv, test_errors.csv ({len(errors)} errors)")


if __name__ == "__main__":
    main()
