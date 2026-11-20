"""Turn the labeled workbook into the final dataset and the train/val/test split.

    python -m src.make_dataset                       # reads labeling/labeling_sheet.xlsx
    python -m src.make_dataset --labels my_export.csv
    python -m src.make_dataset --resplit             # makes a new split: do not use after training

The split is saved in data/splits/split_assignment.csv and reused on every later
run, so fixing a label never moves a comment to another split.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from . import lexicon, pii
from .augment import make_synthetic
from .labeling_sheet import read_workbook
from .normalize import dedup_key
from .utils import ROOT, load_config, path, save_json

KEEP = ["uid", "text", "label", "ambiguous", "prof_lang", "domain", "stars", "pool", "note"]


def validate(df: pd.DataFrame, allow_unlabeled: bool) -> tuple[pd.DataFrame, dict]:
    info = {"rows_in_sheet": len(df)}
    drop = df.not_kazakh.fillna(0).astype(int) == 1
    info["dropped_not_kazakh"] = int(drop.sum())
    df = df[~drop].copy()
    bad = ~df.label.isin([0, 1])
    if bad.any():
        if not allow_unlabeled:
            sys.exit(f"{int(bad.sum())} rows have no valid label (0/1), e.g. uids "
                     f"{df.loc[bad, 'uid'].head(10).tolist()}. Fill them in and run again.")
        print(f"WARNING: dropping {int(bad.sum())} unlabeled rows (--allow-unlabeled)")
    info["dropped_unlabeled"] = int(bad.sum())
    df = df[~bad].copy()
    df["label"] = df.label.astype(int)
    df["ambiguous"] = df.ambiguous.fillna(0).astype(int)
    df["stars"] = df.stars.astype(int)
    df["text"] = df.text.map(pii.scrub)
    # If the annotator left prof_lang blank, take it from the obscene words,
    # then from the insult words, otherwise "unknown".
    auto = df.text.map(lambda t: lexicon.profanity_language(lexicon.match(t))
                       or lexicon.insult_language(t) or "unknown")
    need = (df.label == 1) & ~df.prof_lang.isin(["kk", "ru", "en", "mix"])
    df.loc[need, "prof_lang"] = auto[need]
    df.loc[df.label == 0, "prof_lang"] = ""
    info["prof_lang_filled_automatically"] = int(need.sum())
    return df[KEEP], info


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.assign(_key=df.text.map(dedup_key)).sort_values("uid")
    conflicts = df.groupby("_key").label.nunique()
    n_conflicts = int((conflicts > 1).sum())
    if n_conflicts:
        keys = conflicts[conflicts > 1].index
        print(f"WARNING: {n_conflicts} duplicate texts have different labels; keeping label 1 "
              f"for them. uids: {df[df._key.isin(keys)].uid.head(10).tolist()}")
        df.loc[df._key.isin(keys), "label"] = 1
    before = len(df)
    df = df[df._key.str.len() > 0].drop_duplicates("_key", keep="first")
    return df.drop(columns="_key"), {"exact_duplicates_removed": before - len(df),
                                     "duplicate_label_conflicts": n_conflicts}


def near_duplicate_groups(texts: list[str], threshold: float, min_len_ratio: float = 0.8,
                          chunk: int = 500) -> np.ndarray:
    """Group near-duplicate comments (union-find). Returns a group root index per text.

    Two texts are near-duplicates if the cosine similarity of their character
    3-5-grams is at least `threshold` and their lengths are similar. The length
    check stops a one-word comment ("котак") from joining every longer comment
    that contains the same word.
    """
    keys = [dedup_key(t) for t in texts]
    lengths = np.array([max(len(k), 1) for k in keys])
    vec = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**20,
                            alternate_sign=False, norm="l2")
    X = vec.transform(keys).tocsr()
    parent = np.arange(len(texts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, X.shape[0], chunk):
        sims = (X[start:start + chunk] @ X.T).tocoo()
        for i, j, s in zip(sims.row + start, sims.col, sims.data):
            if i < j and s >= threshold and \
                    min(lengths[i], lengths[j]) / max(lengths[i], lengths[j]) >= min_len_ratio:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(len(texts))])


def assign_splits(df: pd.DataFrame, val_size: float, test_size: float, seed: int) -> pd.Series:
    """Split whole groups, shuffled separately inside each (label, pool) combination."""
    rng = np.random.default_rng(seed)
    g = df.groupby("group_id").agg(label=("label", "max"), pool=("pool", "first"), n=("uid", "size"))
    split = {}
    for _, stratum in g.groupby(["label", "pool"]):
        order = stratum.index.to_numpy()[rng.permutation(len(stratum))]
        total, filled = stratum.n.sum(), 0
        for gid in order:
            share = filled / total
            split[gid] = "test" if share < test_size else ("val" if share < test_size + val_size else "train")
            filled += stratum.n[gid]
    return df.group_id.map(split)


def stats(df: pd.DataFrame, synth: pd.DataFrame, info: dict, cfg: dict) -> dict:
    out = dict(info)
    out["seed"] = cfg["seed"]
    out["near_dup_threshold"] = cfg["split"]["near_dup_threshold"]
    out["total_examples"] = int(len(df))
    out["class_counts"] = {str(k): int(v) for k, v in df.label.value_counts().sort_index().items()}
    out["near_duplicate_groups_with_2plus"] = int((df.group_id.value_counts() > 1).sum())
    out["largest_group"] = int(df.group_id.value_counts().max())
    out["splits"] = {
        s: {"n": int((df.split == s).sum()),
            "label_1": int(((df.split == s) & (df.label == 1)).sum()),
            "label_0": int(((df.split == s) & (df.label == 0)).sum())}
        for s in ["train", "val", "test"]}
    out["by_pool"] = df.groupby("pool").label.agg(["size", "sum"]).rename(
        columns={"size": "n", "sum": "label_1"}).astype(int).to_dict("index")
    out["by_domain"] = df.groupby("domain").label.agg(["size", "sum"]).rename(
        columns={"size": "n", "sum": "label_1"}).astype(int).to_dict("index")
    out["prof_lang_of_positives"] = df[df.label == 1].prof_lang.value_counts().to_dict()
    out["ambiguous"] = int(df.ambiguous.sum())
    out["synthetic_train_examples"] = {
        f"{k[0]}_label{k[1]}": int(v) for k, v in synth.groupby(["aug_type", "label"]).size().items()
    } if len(synth) else {}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", help="labeled workbook (.xlsx) or CSV export of its Label sheet")
    ap.add_argument("--resplit", action="store_true", help="ignore the saved split assignment")
    ap.add_argument("--allow-unlabeled", action="store_true", help="drop blank labels instead of failing")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    labels_file = args.labels or path(cfg, "labeling_xlsx")
    df, info = validate(read_workbook(labels_file), args.allow_unlabeled)
    df, dup_info = drop_exact_duplicates(df)
    info.update(dup_info)
    df = df.sort_values("uid").reset_index(drop=True)
    roots = near_duplicate_groups(df.text.tolist(), cfg["split"]["near_dup_threshold"])
    df["group_id"] = df.uid.to_numpy()[roots]          # a group is named after its smallest uid

    splits_dir = path(cfg, "splits_dir")
    splits_dir.mkdir(parents=True, exist_ok=True)
    frozen_file = splits_dir / "split_assignment.csv"
    new_split = assign_splits(df, cfg["split"]["val_size"], cfg["split"]["test_size"], seed)
    if frozen_file.exists() and not args.resplit:
        frozen = pd.read_csv(frozen_file, dtype=str).set_index("uid").split
        df["split"] = df.uid.map(frozen)
        # A new row goes to the split of its near-duplicate group, if the group already has one.
        by_group = df.dropna(subset=["split"]).groupby("group_id").split.first()
        df["split"] = df.split.fillna(df.group_id.map(by_group)).fillna(new_split)
        print(f"reused frozen split ({frozen_file.relative_to(ROOT)})")
    else:
        df["split"] = new_split
        print("created a new split")
    assert df.groupby("group_id").split.nunique().max() == 1, "near-duplicates split apart"
    df[["uid", "group_id", "split"]].to_csv(frozen_file, index=False)

    train = df[df.split == "train"]
    synth = make_synthetic(train, cfg, seed)          # synthetic copies come from training rows only
    for s in ["train", "val", "test"]:
        df[df.split == s].to_csv(splits_dir / f"{s}.csv", index=False)
    synth.to_csv(splits_dir / "train_synthetic.csv", index=False)
    df.to_csv(path(cfg, "labeled_csv"), index=False)
    st = stats(df, synth, info, cfg)
    save_json(st, ROOT / "data" / "dataset_stats.json")
    print(f"examples {st['total_examples']} | classes {st['class_counts']} | splits "
          + ", ".join(f"{k}: {v['n']} ({v['label_1']} pos)" for k, v in st["splits"].items())
          + f" | synthetic {len(synth)}")


if __name__ == "__main__":
    main()
