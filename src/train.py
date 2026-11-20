"""Train the model and choose its settings on the validation split.

    python -m src.train

Tries every setting in config.yaml, picks the one with the best validation F1,
checks once whether synthetic rows help, and saves the model with its threshold
to models/model.joblib. The test split is never read here.
"""
import hashlib
import itertools
import platform
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

from .model import (ProfanityClassifier, Settings, build_classifier, build_vectorizer, prepare,
                    sigmoid, window_scores)
from .plots import pr_curve
from .utils import ROOT, load_config, path, save_json, set_seed


def best_threshold(y, p) -> tuple[float, float, float, float]:
    """Threshold with the best F1. Returns (threshold, f1, precision, recall)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i]), float(prec[i]), float(rec[i])


def threshold_for_recall(y, p, target: float) -> dict:
    """Highest threshold that still reaches the target recall."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    if len(ok) == 0:
        return {}
    i = ok[-1]
    return {"threshold": float(thr[i]), "precision": float(prec[i]), "recall": float(rec[i])}


def load_split(cfg, name: str) -> pd.DataFrame:
    return pd.read_csv(path(cfg, "splits_dir") / f"{name}.csv", dtype={"uid": str, "text": str},
                       keep_default_na=False)


def training_frame(cfg, use_synthetic: bool) -> pd.DataFrame:
    train = load_split(cfg, "train")
    if use_synthetic:
        synth = load_split(cfg, "train_synthetic")
        if len(synth):
            train = pd.concat([train, synth[["uid", "text", "label"]]], ignore_index=True)
    return train


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    mcfg = cfg["model"]
    fixed, grid = mcfg["fixed"], mcfg["grid"]
    val = load_split(cfg, "val")
    y_val = val.label.astype(int).to_numpy()
    use_syn_default = mcfg["use_synthetic"][0]
    train = training_frame(cfg, use_syn_default)
    y_tr = train.label.astype(int).to_numpy()
    print(f"train {len(train)} rows ({int(y_tr.sum())} positive, synthetic={use_syn_default}) | "
          f"val {len(val)} rows ({int(y_val.sum())} positive)")

    # 1. Grid search. The TF-IDF is built once per text setting, then reused for every C.
    rows, t0 = [], time.time()
    text_keys = list(itertools.product(grid["analyzer"], grid["ngram_range"], grid["fold"]))
    for analyzer, ngram, fold in text_keys:
        base = Settings(analyzer=analyzer, ngram_range=tuple(ngram), fold=fold, **fixed)
        vec = build_vectorizer(base)
        X_tr = vec.fit_transform(prepare(train.text, fold))
        val_norm = prepare(val.text, fold)
        for C, cw in itertools.product(grid["C"], grid["class_weight"]):
            s = replace(base, C=C, class_weight=cw, use_synthetic=use_syn_default)
            clf = build_classifier(s, seed).fit(X_tr, y_tr)
            for window in mcfg["windows"]:
                p = sigmoid(window_scores(vec, clf, val_norm, window))
                thr, f1, pr, rc = best_threshold(y_val, p)
                rows.append({**asdict(replace(s, window=window)), "vocab": len(vec.vocabulary_),
                             "val_f1": f1, "val_precision": pr, "val_recall": rc, "threshold": thr,
                             "val_ap": float(average_precision_score(y_val, p)),
                             "val_f1_at_0.5": float(f1_score(y_val, (p >= 0.5).astype(int)))})
        print(f"  {analyzer:7s} {tuple(ngram)} fold={fold!s:5s} done ({time.time() - t0:.0f}s)")

    # 2. Best validation F1 wins. Ties: higher average precision, then the smaller vocabulary.
    search = pd.DataFrame(rows)
    search["ngram_range"] = search.ngram_range.map(lambda r: f"{r[0]}-{r[1]}")
    search = search.sort_values(["val_f1", "val_ap", "vocab"], ascending=[False, False, True])
    best = search.iloc[0]
    chosen = Settings(analyzer=best.analyzer, ngram_range=tuple(int(x) for x in best.ngram_range.split("-")),
                      fold=bool(best.fold), C=float(best.C),
                      class_weight=None if pd.isna(best.class_weight) else best.class_weight,
                      window=None if pd.isna(best.window) else int(best.window),
                      use_synthetic=use_syn_default, **fixed)

    # 3. Train the chosen setting with and without synthetic rows and keep the better one.
    ablation = {}
    for use_syn in mcfg["use_synthetic"]:
        tr = training_frame(cfg, use_syn)
        m = ProfanityClassifier(replace(chosen, use_synthetic=use_syn), seed=seed).fit(
            tr.text, tr.label.astype(int))
        p = m.predict_proba(val.text)
        thr, f1, pr, rc = best_threshold(y_val, p)
        ablation[use_syn] = {"model": m, "p": p, "threshold": thr, "f1": f1, "precision": pr,
                             "recall": rc, "ap": float(average_precision_score(y_val, p)),
                             "train_rows": len(tr)}
    keep = max(ablation, key=lambda k: (ablation[k]["f1"], ablation[k]["ap"], k == use_syn_default))
    final = ablation[keep]

    # 4. Save the model with its validation threshold, and the search results.
    model = final["model"]
    model.threshold = final["threshold"]
    model.metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_rows": final["train_rows"], "val_rows": len(val),
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "val_f1": final["f1"], "val_precision": final["precision"], "val_recall": final["recall"],
        "val_average_precision": final["ap"],
    }
    out = path(cfg, "model_path")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out, compress=3)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.parent / "SHA256SUMS").write_text(f"{digest}  {out.name}\n", encoding="utf-8")

    res = path(cfg, "results_dir")
    res.mkdir(parents=True, exist_ok=True)
    search.to_csv(res / "val_search.csv", index=False)
    selection = {
        "chosen_settings": asdict(model.settings),
        "threshold": model.threshold,
        "validation_at_threshold": {k: final[k] for k in ["f1", "precision", "recall", "ap"]},
        "alternative_threshold_recall_target": threshold_for_recall(y_val, final["p"], mcfg["recall_target"]),
        "synthetic_ablation": {("with" if k else "without") + "_synthetic":
                               {m: v[m] for m in ["f1", "precision", "recall", "ap", "threshold", "train_rows"]}
                               for k, v in ablation.items()},
        "settings_tried": int(len(search)),
        "top5": search.head(5).drop(columns=["max_iter", "min_df", "sublinear_tf"]).to_dict("records"),
        "model_file": str(out.relative_to(ROOT)),
        **model.metadata,
    }
    save_json(selection, res / "val_selection.json")
    prec, rec, thrs = precision_recall_curve(y_val, final["p"])
    i = int(np.argmin(np.abs(thrs - model.threshold)))
    pr_curve(rec, prec, rec[i], prec[i], model.threshold, res / "val_pr_curve.png",
             "Validation precision-recall curve")
    print(f"\nchosen: {asdict(model.settings)}")
    print(f"threshold {model.threshold:.3f} | val F1 {final['f1']:.3f} P {final['precision']:.3f} "
          f"R {final['recall']:.3f} AP {final['ap']:.3f} | synthetic used: {keep}")
    print(f"saved {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB) and results/val_*.")


if __name__ == "__main__":
    main()
