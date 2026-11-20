"""Small helpers: config, paths, seed, reading KazSAnDRA."""
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(cfg: dict, key: str) -> Path:
    """Full path for a key in the `paths` section of config.yaml."""
    return ROOT / cfg["paths"][key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_kazsandra(raw_dir: Path) -> pd.DataFrame:
    """All KazSAnDRA reviews with columns uid, text, stars (1-5), domain."""
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {raw_dir}. Run: python -m src.download_data")
    df = pd.concat([pd.read_csv(f, dtype={"custom_id": str, "text": str}) for f in files],
                   ignore_index=True)
    df = df.rename(columns={"custom_id": "uid"})
    df["stars"] = df["label"].astype(int) + 1          # KazSAnDRA stores 1-5 stars as 0-4
    df["text"] = df["text"].fillna("").astype(str)
    return df[["uid", "text", "stars", "domain"]].sort_values("uid").reset_index(drop=True)
