"""Word patterns used to FIND comments for labeling. They are not model features.

The model learns only from character n-grams. These patterns were used to build
the labeling pool, to suggest labels, to fill prof_lang and to find the profane
word when making synthetic copies. Every label was checked by a person.

view "base": normalized text; view "fold": normalized text with қ -> к, ң -> н, ...
tier "strong": suggested label 1; tier "weak": the annotator decides.
"""
import re
from dataclasses import dataclass

from .normalize import KAZAKH_TO_RUSSIAN, normalize


@dataclass(frozen=True)
class Seed:
    name: str
    lang: str      # kk / ru / en / mask
    tier: str      # strong / weak
    view: str      # base / fold
    pattern: str


SEEDS = [
    # --- Kazakh roots -------------------------------------------------------
    Seed("kk_qotaq", "kk", "strong", "fold", r"кота[кг]"),
    Seed("kk_qotaq_lat", "kk", "strong", "base", r"\b[qk]ota[qkg]"),
    Seed("kk_sik", "kk", "strong", "base",
         r"\bс[іи][кг](?:ей|ем|ті|ти|кен|іп|ип|пе|тір|тир|ер|у|іс|ис|іл|ил|ін|ин|еді|етін)"),
    Seed("kk_sik_short", "kk", "strong", "base", r"\bсг(?:ин|ейн|еин|ем|іп|ип|иін|ііін|еймін)"),
    Seed("kk_sik_y", "kk", "weak", "base", r"\bсы[кг](?:ей|ем|ти|кен|ип|пе|тир|ер|ин)"),
    Seed("kk_sik_lat", "kk", "strong", "base", r"\bs[iy][kg](?:ey|em|ti|ken|ip|pe|tir|in)|\bsgin"),
    Seed("kk_am", "kk", "strong", "fold",
         r"\bам(?:ы|ын|ына|ынды|ыны|ынын|ыннан|ынан|дары|дар)\b|\bамш[ыи][кг]"),
    Seed("kk_am_lat", "kk", "strong", "base", r"\bamsh[yi][kq]"),
    Seed("kk_am_lat_weak", "kk", "weak", "base", r"\bam(?:y|yn|yng|yndy|yndi|yna)\b"),
    Seed("kk_am_alone", "kk", "weak", "base", r"\bам\b"),
    Seed("kk_kot", "kk", "strong", "base", r"\bкөт(?:і|ің|іңе|іне|ін|іңді|ке|тен|ақ)?\b"),
    Seed("kk_kot_ru_letters", "kk", "weak", "base", r"\bкот(?:ин|ине|ини|ыне|ын|ке)\b"),
    Seed("kk_boq", "kk", "strong", "base", r"\bбоқ\b|\bбоғ(?:ы|ын|ың|ыңды)\b|\bboq\b"),
    Seed("kk_boq_forms", "kk", "weak", "base",
         r"\bбоқ[а-яәғқңөұүһі]+|\bбок\b|\bбок(?:та|тай|тык|ты)\w*|\bbok\b"),
    Seed("kk_qanshyq", "kk", "weak", "fold", r"\bканш[ыи][кг]|\b[kq]ansh[yi][kq]"),
    Seed("kk_zhalap", "kk", "weak", "base", r"\bжалап(?:тар|сың|қа|ты|тың)?\b|\b(?:zh|j)alap\b"),
    Seed("kk_zhalep", "kk", "strong", "fold", r"\bжалеп|\b(?:zh|j)alep"),
    Seed("kk_saiqal", "kk", "weak", "fold", r"\bса[йи]кал"),
    Seed("kk_curse", "kk", "weak", "fold", r"\b(?:шеше|аке|ата|ене|апа)н(?:ди|ды|нин|нын)\b"),
    # --- Russian mat and vulgar words code-switched into Kazakh -------------
    Seed("ru_dalbayob", "ru", "strong", "base", r"\bд[ао]лб[ао]|\bd[ao]lb[ao]"),
    Seed("ru_xuy", "ru", "strong", "base",
         r"\bх[уy][йяеёюи]|\bхул[иеья]\b|\bнах[уy](?:[йияюё]|\b)|\bнахю|\bпох[уy][йияюё]|\bху[йи]н"
         r"|\bхер(?:ня|ни|ов|ова|н|а|у)?\b|\bнахер|\bпохер"),
    Seed("ru_xuy_lat", "ru", "strong", "base", r"\b(?:na|po)?h[uy][iy]\b|\b(?:na|po)?xu[iy]"),
    Seed("ru_pizd", "ru", "strong", "base",
         r"п[иі][зс]д|\bпздц|\bпзд\b|p[iy]zd|\bpzdc|\bp[iy]sdec"),
    Seed("ru_eb", "ru", "strong", "base",
         r"\b(?:за|по|на|вы|у|от|раз|съ|при|до|про|долбо|и)?[её]б(?:а|у|ну|ан|ат|ал|ыв|л|ут|ёт|ет|ош|ись|уч|ля|ли|ем)"
         r"|\bза[йи]б[аи]|\bй[оё]б|\bи[её]бан"),
    Seed("ru_eb_lat", "ru", "strong", "base",
         r"\b(?:za|po|na|vy|ras|raz)?(?:y|j)?[ej]ba[ln]|\bza[ei]bal|\bzaebis|\by[oi]b[a-z]"),
    Seed("ru_blya", "ru", "strong", "base", r"\bбл[яь]|\bбл[әэ]т|\bблят|\bbl[yj]a|\bblyat"),
    Seed("ru_suka", "ru", "strong", "base", r"\bсук(?:а|и|е|у|ин|ам|ой)\b|\bсуч(?:ка|ки|ара)|\bsuk[ai]\b"),
    Seed("ru_other", "ru", "strong", "base",
         r"\bмуд[ао]к|\bмудил|\bг[ао]нд[ао]н|\bшлюх|\bг[ао]вн|\bдерьм|\bжоп[аеуы]|\bпид[ао]?р|\bпидр"
         r"|\bзалуп|\bдроч|\bчмо\b|\bманда\b|\bуеб|\bуёб"
         r"|\bg[ao]vn|\bpid[ao]r|\bmudak|\bg[ao]nd[ao]n|\b(?:zh|j)op[aeuy]\b|\bshluh"),
    # --- English -------------------------------------------------------------
    Seed("en_profanity", "en", "strong", "base",
         r"\bf+u+c*k|\bfck|\bfuk|\bsh[i1]t\b|\bbitch|\bwtf\b|\bmotherf|\bdick\b|\bpussy|\basshole"),
    # --- Masked words (any language) ------------------------------------------
    Seed("masked", "mask", "weak", "base", r"[^\W\d_]\*|\*[^\W\d_]"),
]

INSULT_PATTERN = re.compile(
    r"\b(?:мал|малдар|малсың|малсыңдар|ақымақ|акымак|есек|ешак|ешек|топас|жынды|дурак|дура|тупой|тупая"
    r"|идиот|оңбаған|онбаган|малғұн|малгун|сволоч\w*|урод\w*|дебил\w*|кретин\w*|лох|қатын|катын"
    r"|шірік|сасық|иттің баласы|ит)\b")

# Insults, soft swearing and curses (guidelines 1.1), by language. Only used to fill a
# blank prof_lang. INSULT_PATTERN above is kept as it was, because it selected the pool.
# Long words may take suffixes (ақымақсың); short ones (мал, ит) must match exactly,
# so мала and итер do not count.
INSULT_WORDS = {
    "kk": re.compile(
        r"\b(?:ақымақ|акымак|есек|ешак|ешек|топас|жынды|оңбаған|онбаган|малғұн|малгун"
        r"|қатын|катын|шірік|сасық|сасык)\w*"
        r"|\b(?:мал|малдар|малсың|малсыңдар|ит)\b"
        r"|иттің баласы|әкеңнің басы|атаңның басы|құрып кет|қарғыс атсын"),
    "ru": re.compile(
        r"\b(?:дурак|тупо|идиот|лох|блин|капец|хрен|фиг|ч[её]рт|сволоч|урод|дебил|кретин)\w*"
        r"|\bдура\b"),
}

_COMPILED = [(s, re.compile(s.pattern)) for s in SEEDS]


def match(text: str) -> list[Seed]:
    """Seeds that match a raw comment."""
    base = normalize(text)
    fold = normalize(text, fold=True)
    return [s for s, rx in _COMPILED if rx.search(base if s.view == "base" else fold)]


def suggest(seeds: list[Seed]) -> str:
    """Suggested label for the annotator: '1', '?' or '0'."""
    if any(s.tier == "strong" for s in seeds):
        return "1"
    if seeds:
        return "?"
    return "0"


def profanity_language(seeds: list[Seed]) -> str:
    """kk / ru / en / mix from the matched seeds; '' if only masks matched."""
    langs = {s.lang for s in seeds if s.lang != "mask"}
    if not langs:
        return ""
    return langs.pop() if len(langs) == 1 else "mix"


def profane_spans(norm_text: str) -> list[tuple[int, int]]:
    """(start, end) of every word in normalized text that contains a seed match.

    Folding changes one letter into one letter, so positions are the same in both views.
    """
    fold = norm_text.translate(KAZAKH_TO_RUSSIAN)
    spans = []
    for s, rx in _COMPILED:
        if s.lang == "mask":
            continue
        for m in rx.finditer(norm_text if s.view == "base" else fold):
            start = norm_text.rfind(" ", 0, m.start()) + 1
            end = norm_text.find(" ", m.end())
            spans.append((start, len(norm_text) if end == -1 else end))
    return sorted(set(spans))


def is_insult(text: str) -> bool:
    return bool(INSULT_PATTERN.search(normalize(text)))


def insult_language(text: str) -> str:
    """kk / ru / mix from insult, soft swearing and curse words; '' if none."""
    norm = normalize(text)
    langs = [lang for lang, rx in INSULT_WORDS.items() if rx.search(norm)]
    if not langs:
        return ""
    return langs[0] if len(langs) == 1 else "mix"
