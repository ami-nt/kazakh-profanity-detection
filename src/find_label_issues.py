"""Find possible label mistakes in the TRAINING split (never val or test).

    python -m src.find_label_issues

5-fold cross-validation inside the training split: every comment gets a
probability from a model that did not see it. Rows where the probability
strongly disagrees with the label go to results/train_label_issues.csv.
Check them in the workbook, fix real mistakes, then rerun make_dataset and train.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .model import ProfanityClassifier, Settings
from .utils import ROOT, load_config, path


def main() -> None:
    cfg = load_config()
    train = pd.read_csv(path(cfg, "splits_dir") / "train.csv", dtype={"uid": str, "text": str},
                        keep_default_na=False)
    sel_file = path(cfg, "results_dir") / "val_selection.json"
    if sel_file.exists():
        chosen = json.load(open(sel_file, encoding="utf-8"))["chosen_settings"]
        chosen["ngram_range"] = tuple(chosen["ngram_range"])
        settings = Settings(**chosen)
    else:
        settings = Settings()
    y = train.label.astype(int).to_numpy()
    oof = np.zeros(len(train))
    # GroupKFold keeps near-duplicates in the same fold.
    for tr_idx, te_idx in GroupKFold(n_splits=5).split(train, y, groups=train.group_id):
        m = ProfanityClassifier(settings, seed=cfg["seed"]).fit(train.text.iloc[tr_idx], y[tr_idx])
        oof[te_idx] = m.predict_proba(train.text.iloc[te_idx])
    train["oof_prob"] = oof
    flagged = train[((y == 1) & (oof < 0.2)) | ((y == 0) & (oof > 0.8))].copy()
    flagged["disagreement"] = np.abs(flagged.label - flagged.oof_prob)
    flagged = flagged.sort_values("disagreement", ascending=False)
    out = path(cfg, "results_dir") / "train_label_issues.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    flagged[["uid", "text", "label", "oof_prob", "pool", "note"]].to_csv(out, index=False)
    print(f"{len(flagged)} of {len(train)} training rows flagged -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
