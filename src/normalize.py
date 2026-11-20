"""Text cleanup rules used everywhere: labeling, training and prediction.

Nothing here is learned from data, so using it on every split is safe.
"""
import html
import re
import unicodedata

# Kazakh letters and the Russian letters people often type instead (used when fold=True).
KAZAKH_TO_RUSSIAN = str.maketrans({
    "қ": "к", "ғ": "г", "ү": "у", "ұ": "у", "і": "и",
    "ө": "о", "ң": "н", "ә": "а", "һ": "х", "ё": "е",
})

# Latin letters that look like Cyrillic ones. Only fixed inside words that mix both
# scripts ("kотақ"), so real Latin-script Kazakh ("qotaq", "jaqsy") is not changed.
LATIN_HOMOGLYPHS = str.maketrans({
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р",
    "x": "х", "y": "у", "k": "к", "i": "і", "m": "м",
})

PLACEHOLDER_RE = re.compile(r"<(URL|EMAIL|USER|PHONE|NUMBER|NAME)>", re.IGNORECASE)
INVISIBLE_RE = re.compile(r"[\u00ad\u200b-\u200f\u2060\ufeff]")    # soft hyphen, zero-width spaces, BOM
CYR = "а-яёәғқңөұүһі"
CYR_RE = re.compile(f"[{CYR}]")
LAT_RE = re.compile(r"[a-z]")
LETTER = r"[^\W\d_]"
MASK_RUN_RE = re.compile(r"[*#]+")
SEP_PUNCT_RE = re.compile(rf"\b(?:{LETTER}[.\-]){{2,}}{LETTER}\b")   # қ.о.т.а.қ
SEP_SPACE_RE = re.compile(rf"\b(?:{LETTER} ){{3,}}{LETTER}\b")       # б л я т ь
REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")                         # 3+ same letters
NON_TOKEN_RE = re.compile(r"[^\w*]+|_+")                            # punctuation, emoji, underscores
TOKEN_RE = re.compile(r"\S+")


def _fix_token(tok: str) -> str:
    """Fix one word: Latin look-alike letters, 0/@/3 used as letters, mask symbols."""
    has_cyr = bool(CYR_RE.search(tok))
    if has_cyr and LAT_RE.search(tok):
        tok = tok.translate(LATIN_HOMOGLYPHS)
    if re.search(LETTER, tok):
        o, a = ("о", "а") if has_cyr else ("o", "a")
        tok = re.sub(rf"(?<={LETTER})0(?={LETTER})", o, tok)
        tok = re.sub(rf"(?<={LETTER})@(?={LETTER})", a, tok)
        if has_cyr:
            tok = re.sub(rf"(?<={LETTER})3(?={LETTER})", "з", tok)
    # A run of * or # becomes one "*" if it touches a letter (қ**тақ), otherwise a space (5*****).
    parts = MASK_RUN_RE.split(tok)
    if len(parts) > 1:
        out = parts[0]
        for nxt in parts[1:]:
            touches = (out and re.search(rf"{LETTER}$", out)) or (nxt and re.match(LETTER, nxt))
            out += ("*" if touches else " ") + nxt
        tok = out
    return tok


def _normalize_piece(text: str) -> str:
    text = SEP_PUNCT_RE.sub(lambda m: re.sub(r"[.\-]", "", m.group()), text)
    text = SEP_SPACE_RE.sub(lambda m: m.group().replace(" ", ""), text)
    text = " ".join(_fix_token(t) for t in TOKEN_RE.findall(text))
    text = NON_TOKEN_RE.sub(" ", text)
    text = REPEAT_RE.sub(r"\1", text)
    return text


def normalize(text: str, fold: bool = False) -> str:
    """Return the cleaned, lowercase form of a comment.

    With fold=True Kazakh letters are also written as Russian ones,
    so "қотақ" and "котак" become the same text.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = INVISIBLE_RE.sub("", text).lower()
    pieces, last = [], 0
    for m in PLACEHOLDER_RE.finditer(text):           # keep placeholders like <url> untouched
        pieces.append(_normalize_piece(text[last:m.start()]))
        pieces.append(f" <{m.group(1).lower()}> ")
        last = m.end()
    pieces.append(_normalize_piece(text[last:]))
    text = " ".join("".join(pieces).split())
    if fold:
        text = text.translate(KAZAKH_TO_RUSSIAN)
    return text


def dedup_key(text: str) -> str:
    """Text used to find exact duplicates: folded, without digits and masks."""
    t = normalize(text, fold=True)
    t = re.sub(r"[\d*]+", " ", t)
    return " ".join(t.split())
