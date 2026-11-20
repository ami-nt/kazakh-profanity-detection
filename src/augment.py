"""Synthetic copies of training comments (train split only, marked is_synthetic=1).

Real data has few masked or Latin-script swear words, so three kinds of copies are made:
  masked           1-2 letters of a profane word become "*" (label 1 rows only)
  russian_letters  Kazakh letters written as Russian ones (қ -> к)
  latin            Latin spelling (қ -> q, ш -> sh)
The last two are made in equal numbers for both labels, so the script alone does not
reveal the label. Copies keep the group_id of their original comment.
"""
import numpy as np
import pandas as pd

from .lexicon import profane_spans
from .normalize import CYR_RE, KAZAKH_TO_RUSSIAN, normalize

LATIN_BASE = {
    "а": "a", "ә": "a", "б": "b", "в": "v", "г": "g", "ғ": "g", "д": "d", "е": "e", "ё": "yo",
    "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "ң": "n", "о": "o",
    "ө": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "u", "ү": "u", "ф": "f",
    "х": "h", "һ": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "", "ы": "y", "і": "i",
    "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
LATIN_VARIANTS = {"ж": ["zh", "j"], "қ": ["q", "k"], "ғ": ["g", "gh"], "ң": ["n", "ng"]}
KK_LETTERS = set("әғқңөұүһі")


def to_latin(text: str, rng: np.random.Generator) -> str:
    table = dict(LATIN_BASE)
    for ch, options in LATIN_VARIANTS.items():
        table[ch] = options[rng.integers(len(options))]
    return "".join(table.get(ch, ch) for ch in text)


def mask_word(norm_text: str, rng: np.random.Generator) -> str | None:
    """Replace 1-2 inner letters of one profane word with "*". None if there is no such word."""
    spans = [s for s in profane_spans(norm_text) if s[1] - s[0] >= 3]
    if not spans:
        return None
    start, end = spans[rng.integers(len(spans))]
    word = list(norm_text[start:end])
    inner = [i for i in range(1, len(word) - 1) if word[i].isalpha()] or [len(word) - 1]
    for i in rng.choice(inner, size=min(len(inner), int(rng.integers(1, 3))), replace=False):
        word[i] = "*"
    return norm_text[:start] + "".join(word) + norm_text[end:]


def make_synthetic(train: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    acfg = cfg["augment"]
    rows = []

    def add(parent, text, kind):
        rows.append({
            "uid": f"syn_{kind}_{parent.uid}", "parent_uid": parent.uid, "text": text,
            "label": int(parent.label), "group_id": parent.group_id, "domain": parent.domain,
            "stars": parent.stars, "pool": parent.pool, "aug_type": kind, "is_synthetic": 1,
        })

    pos = train[train.label == 1]
    n_mask = int(len(pos) * acfg["max_fraction_of_positives"])
    for parent in pos.sample(frac=1.0, random_state=seed).itertuples():
        if sum(r["aug_type"] == "masked" for r in rows) >= n_mask:
            break
        masked = mask_word(normalize(parent.text), rng)
        if masked:
            add(parent, masked, "masked")

    for label in (0, 1):
        has_kk = train[(train.label == label) & train.text.map(lambda t: bool(KK_LETTERS & set(t.lower())))]
        k = min(len(has_kk), acfg["n_russian_letters_per_class"])
        for parent in has_kk.sample(k, random_state=seed + 10 + label).itertuples():
            add(parent, normalize(parent.text).translate(KAZAKH_TO_RUSSIAN), "russian_letters")
        cyr = train[(train.label == label) & train.text.map(lambda t: bool(CYR_RE.search(t.lower())))]
        k = min(len(cyr), acfg["n_latin_per_class"])
        for parent in cyr.sample(k, random_state=seed + 20 + label).itertuples():
            add(parent, to_latin(normalize(parent.text), rng), "latin")
    return pd.DataFrame(rows)
