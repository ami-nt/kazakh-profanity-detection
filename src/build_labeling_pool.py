"""Choose the comments to label and write the labeling workbook.

    python -m src.build_labeling_pool                       # will not overwrite a labeled sheet
    python -m src.build_labeling_pool --out labeling/rebuilt.xlsx

Pools: candidate (a seed word matched), lookalike (innocent words that look like
swear words), insult (мал, ақымақ, ...), one_star (angry reviews) and random.
Sampling uses the seed from config.yaml, so the same pool comes out every time.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import lexicon, pii
from .labeling_sheet import write_workbook
from .normalize import dedup_key, normalize
from .utils import load_config, path, read_kazsandra

LOOKALIKE_RE = re.compile(
    r"\bкөтер|\bкотер|\bамин|\bамал|\bаман|\bамбар|\bсигнал|\bсигар|\bкот\b|\bкотик|\bкотта|\bбокс"
    r"|\bсук\b|\bжала|\bхулиган|\bхуже|\bскин|\bсекір|\bсикыр|\bсиқыр|\bаманат|\bжапа|\bкөтен|\bебі")

POOL_ORDER = ["candidate", "lookalike", "insult", "one_star", "random"]


def sample_by_domain(df: pd.DataFrame, n: int, shares: dict, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for dom, share in shares.items():
        sub = df[df.domain == dom]
        k = min(len(sub), int(round(n * share)))
        parts.append(sub.sample(k, random_state=rng.integers(1 << 31)))
    return pd.concat(parts)


def build_pool(cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    df = read_kazsandra(path(cfg, "raw_dir"))
    n_raw = len(df)
    df["text"] = df["text"].map(pii.scrub)
    df["key"] = df["text"].map(dedup_key)
    df = df[df.key.str.len() > 0].drop_duplicates("key", keep="first").copy()
    print(f"{n_raw} reviews -> {len(df)} after scrubbing and exact dedup")

    seeds = df["text"].map(lexicon.match)
    df["hits"] = seeds.map(lambda ss: ", ".join(s.name for s in ss))
    df["suggested"] = seeds.map(lexicon.suggest)
    df["prof_lang"] = seeds.map(lexicon.profanity_language)
    df["pool"] = ""
    df.loc[df.suggested != "0", "pool"] = "candidate"

    free = df.pool == ""
    norm = df["text"].map(normalize)
    look = df[free & norm.str.contains(LOOKALIKE_RE)]
    df.loc[look.sample(min(len(look), cfg["pool"]["n_lookalike"]),
                       random_state=rng.integers(1 << 31)).index, "pool"] = "lookalike"
    free = df.pool == ""
    ins = df[free & norm.str.contains(lexicon.INSULT_PATTERN)]
    df.loc[ins.sample(min(len(ins), cfg["pool"]["n_insult"]),
                      random_state=rng.integers(1 << 31)).index, "pool"] = "insult"
    free = df.pool == ""
    one = sample_by_domain(df[free & (df.stars == 1)], cfg["pool"]["n_one_star"],
                           cfg["pool"]["one_star_domain_share"], rng)
    df.loc[one.index, "pool"] = "one_star"
    free = df.pool == ""
    rnd = sample_by_domain(df[free], cfg["pool"]["n_random"], cfg["pool"]["random_domain_share"], rng)
    df.loc[rnd.index, "pool"] = "random"

    pool = df[df.pool != ""].copy()
    pool["maybe_name"] = pool["text"].map(lambda t: "1" if pii.maybe_name(t) else "")
    # Pre-fill: strong candidates 1, weak candidates blank (the annotator decides), others 0.
    pool["label"] = np.where(pool.suggested == "?", "", np.where(pool.pool == "candidate", "1", "0"))
    pool["prof_lang"] = np.where(pool.pool == "candidate", pool.prof_lang, "")
    for col in ["ambiguous", "not_kazakh", "note"]:
        pool[col] = ""
    # Hardest rows first: weak candidates, then strong candidates, then the other pools.
    pool["_pool_rank"] = pool.pool.map({p: i for i, p in enumerate(POOL_ORDER)})
    pool["_weak_first"] = (pool.suggested != "?").astype(int)
    pool["_shuffle"] = rng.random(len(pool))
    pool = pool.sort_values(["_pool_rank", "_weak_first", "_shuffle"])
    cols = ["uid", "pool", "domain", "stars", "text", "hits", "suggested", "label",
            "ambiguous", "not_kazakh", "prof_lang", "maybe_name", "note"]
    return pool[cols].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="workbook path (default: labeling/labeling_sheet.xlsx)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing workbook")
    args = ap.parse_args()
    cfg = load_config()
    xlsx = path(cfg, "labeling_xlsx") if not args.out else Path(args.out).resolve()
    if xlsx.exists() and not args.force:
        raise SystemExit(f"{xlsx} exists and may contain labels. Use --out NEW.xlsx, or --force.")
    pool = build_pool(cfg)
    out_csv = path(cfg, "pool_csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out_csv, index=False, encoding="utf-8")
    print(pool.pool.value_counts().reindex(POOL_ORDER).to_string())
    print("suggested:", pool.suggested.value_counts().to_dict())
    print("saved", out_csv)
    write_workbook(pool, xlsx)
    print("saved", xlsx)


if __name__ == "__main__":
    main()
