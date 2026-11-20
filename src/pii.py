"""Remove personal information.

Links, e-mails, @handles, phone numbers and long numbers (IIN, card numbers) are
replaced automatically. Names cannot be found reliably by rules (many Kazakh names
are also normal words), so `maybe_name` only marks rows for a person to check.
"""
import re

from .normalize import normalize

URL_RE = re.compile(r"(?:https?://|www\.)[^\s,;!?)]+|\b[\w-]+\.(?:kz|com|ru|org|net|me|io)(?:/[^\s,;!?)]*)?\b",
                    re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
HANDLE_RE = re.compile(r"(?<![\w@])@[\w.]{2,}")
PHONE_RE = re.compile(r"(?:\+7|\b8)[\s\-()]*7\d{2}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}\b")
LONG_NUMBER_RE = re.compile(r"\b\d(?:[\s\-]?\d){8,}\b")      # 9 or more digits

# Common first names that are not also everyday words. Only a hint for the annotator.
FIRST_NAMES = """
айгерим айдана айжан аружан асель асел алия акбота акерке аяулым балжан гулназ
гулмира дарига динара жанар жансая зарина индира камила ляззат мадина назерке
назгул нурай перизат сабина томирис улжан эльмира айсулу аида алина амина анель
арайлым ботагоз дильназ жанель камшат карина меруерт нурсулу толганай улпан
абай адиль айбек айдос айбар алихан алибек асхат бауыржан бейбарыс бекжан бекзат
данияр димаш ерасыл ербол ерлан ержан жандос жасулан мухтар нурлан нурсултан
олжас рауан руслан санжар талгат темирлан тимур шерхан илияс абылай арсен
бакытжан даурен ернар жанибек марат нуржан самат аслан айдар алишер диас ринат
ерболат бекарыс александр алексей андрей дмитрий сергей иван максим артем анна
мария елена ольга наталья екатерина анастасия юлия татьяна ирина светлана
""".split()
NAME_RE = re.compile(r"\b(?:" + "|".join(sorted(set(FIRST_NAMES), key=len, reverse=True)) + r")\b")
SURNAME_RE = re.compile(r"\b[А-ЯӘҒҚҢӨҰҮҺІ][а-яәғқңөұүһі]{2,}(?:ов|ев|ова|ева|ұлы|улы|қызы|кызы)\b")


def scrub(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = EMAIL_RE.sub("<EMAIL>", text)          # before links, because "a@b.kz" contains "b.kz"
    text = URL_RE.sub("<URL>", text)
    text = HANDLE_RE.sub("<USER>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = LONG_NUMBER_RE.sub("<NUMBER>", text)
    return text


def maybe_name(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(NAME_RE.search(normalize(text, fold=True)) or SURNAME_RE.search(text))
