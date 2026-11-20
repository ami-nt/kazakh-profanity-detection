"""Write the labeling workbook and read it back.

The "Label" sheet has one row per comment. The annotator edits only the yellow
columns and replaces private names in `text` with <NAME>.
"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.comments import Comment
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

FONT = "Arial"
YELLOW = PatternFill("solid", fgColor="FFF7CC")
HEADER_FILL = PatternFill("solid", fgColor="1F3A5F")
GREY = PatternFill("solid", fgColor="F2F2F2")
THIN = Side(style="thin", color="BFBFBF")

# (column name, width, editable by the annotator, tooltip)
COLUMNS = [
    ("uid", 15, False, "KazSAnDRA review id. Do not edit."),
    ("text", 90, False, "The comment (contact details already replaced). If it names a private "
                        "person, replace the name with <NAME>. Otherwise do not edit."),
    ("label", 8, True, "1 = contains profanity, 0 = no profanity. Required for every row."),
    ("ambiguous", 11, True, "Put 1 if you were unsure after reading the rules. Keep your best label."),
    ("prof_lang", 11, True, "Only for label 1: language of the profane word(s): kk, ru, en or mix."),
    ("not_kazakh", 11, True, "Put 1 if the comment has no Kazakh at all (pure Russian/English). "
                             "These rows are dropped."),
    ("note", 28, True, "Optional free text: why you chose a label in a hard case."),
    ("suggested", 11, False, "Word-list suggestion: 1 = strong match, ? = weak match (decide), 0 = none."),
    ("hits", 30, False, "Which seed patterns matched (see src/lexicon.py)."),
    ("pool", 11, False, "Why this row was selected: candidate, lookalike, insult, one_star, random."),
    ("domain", 11, False, "KazSAnDRA domain: appstore, market, mapping, bookstore."),
    ("stars", 7, False, "Review rating 1-5."),
    ("maybe_name", 12, False, "1 = may name a person. Check the text and replace private names with <NAME>."),
]
EDITABLE = [c for c, _, e, _ in COLUMNS if e]


def _clean(v):
    if isinstance(v, str):
        return ILLEGAL_CHARACTERS_RE.sub(" ", v)
    return v


def _guide_sheet(wb: Workbook, n_rows: int) -> None:
    ws = wb.active
    ws.title = "Guide"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 95
    rows = [
        ("Kazakh profanity labeling sheet", None),
        ("", None),
        ("What to do", "Go to the Label sheet. For every row read `text` and set `label` "
                       "(1 = profanity, 0 = none). Edit only the yellow columns."),
        ("Pre-filled labels", "Strong word-list matches are pre-filled with 1, non-candidates with 0, "
                              "weak matches are blank. Pre-fills are only suggestions: read every row "
                              "and correct them."),
        ("Order", "Rows are sorted hardest-first: weak candidates, strong candidates, then the "
                  "negative pools. Use the filter arrows to work pool by pool."),
        ("Progress", "The Progress sheet counts labeled rows and positives as you go."),
        ("Personal data", "Phone numbers, e-mails, links and @handles are already replaced. If a "
                          "row names a private person (maybe_name = 1 is a hint), replace the name "
                          "in `text` with <NAME>. Public figures and brands can stay."),
        ("Full rules", "docs/labeling_guidelines.md in the repository. Short version below."),
        ("", None),
        ("Label 1 (profanity)", "Kazakh obscene words and their forms in any spelling or script "
                                "(Cyrillic, Russian letters, Latin): қ*тақ, с*гейін, ам*, амш*қ, жәлеп. "
                                "Vulgar б*қ and к*т. Russian mat and vulgar words inside a Kazakh "
                                "comment: х*й, п*здец, бл*, *бал, с*ка, говно, жопа. English f*ck, sh*t."),
        ("", "Masked words that are still recognisable (қ*ақбас, п*здец). Profanity in a positive, "
             "joking, quoted or self-directed comment still counts. One word is enough."),
        ("", "Since guidelines 1.1 also: insults used to insult (мал, ақымақ, есек, жынды, топас, "
             "дурак, тупой, иттің баласы), euphemistic swearing (блин, капец, хрен, фиг, чёрт) and "
             "curses (атаңның басы, әкеңнің басы, құрып кет, қарғыс атсын)."),
        ("Label 0 (no profanity)", "Harsh negative reviews without any of the words above. Literal "
                                   "meanings of insult words (мал азығы, сасық иіс, шірік алма). "
                                   "ойбай. Words that name swearing: боқтық сөз, мат."),
        ("", "Innocent look-alikes and literal meanings: адам, Амина, амин, көтеру, сигнал; жалап = "
             "'licking'; қаншық = female dog in a pet review; бок = Russian 'side'. "
             "Unrecognisable masks (****), star ratings (5*****), emoji only."),
        ("Unsure?", "Choose the more likely label and put 1 in `ambiguous`; a short `note` helps "
                    "the error analysis later."),
        ("", None),
        ("Example rows (format only, not part of the data)", None),
    ]
    for r, (a, b) in enumerate(rows, start=1):
        ws.cell(r, 1, a).font = Font(name=FONT, bold=True, size=14 if r == 1 else 10)
        if b:
            c = ws.cell(r, 2, b)
            c.font = Font(name=FONT, size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(r, 1).alignment = Alignment(wrap_text=True, vertical="top")
    start = len(rows) + 1
    ex_cols = ["text", "label", "ambiguous", "prof_lang", "note"]
    examples = [
        ["Ойын керемет бл* 5 жұлдыз", 1, "", "ru", "positive review, still profane"],
        ["Сапасы өте нашар, ақшам желге кетті", 0, "", "", "angry but no vulgar words"],
        ["Мысық тағамды жалап жейді", 0, 1, "", "жалап = licking here"],
    ]
    sub = ws.cell(start, 1, "  ".join(ex_cols))
    sub.font = Font(name=FONT, size=9, italic=True, color="666666")
    for i, row in enumerate(examples, start=start + 1):
        ws.cell(i, 1, f"example {i - start}").font = Font(name=FONT, size=10, italic=True)
        ws.cell(i, 2, " | ".join(f"{k}: {v}" for k, v in zip(ex_cols, row) if v != "")).font = \
            Font(name=FONT, size=10)
    note_row = start + len(examples) + 2
    ws.cell(note_row, 1, "Rows to label").font = Font(name=FONT, bold=True)
    ws.cell(note_row, 2, n_rows).font = Font(name=FONT)


def _progress_sheet(wb: Workbook, n: int) -> None:
    ws = wb.create_sheet("Progress")
    ws.sheet_view.showGridLines = False
    last = n + 1
    lab, amb, notkk, pool = (f"Label!$C$2:$C${last}", f"Label!$D$2:$D${last}",
                             f"Label!$F$2:$F${last}", f"Label!$J$2:$J${last}")
    bold = Font(name=FONT, bold=True)
    plain = Font(name=FONT)
    ws["A1"] = "Labeling progress"
    ws["A1"].font = Font(name=FONT, bold=True, size=14)
    ws["A2"] = "Pre-filled labels are suggestions. Read every row; track your place by row number."
    ws["A2"].font = Font(name=FONT, italic=True, size=9, color="666666")
    items = [
        ("Rows in sheet", f"=COUNTA(Label!$A$2:$A${last})", "0"),
        ("Label cells filled (pre-fills count too)", f"=COUNTIF({lab},1)+COUNTIF({lab},0)", "0"),
        ("Blank label cells (must decide)", "=B3-B4", "0"),
        ("Share filled", "=IF(B3=0,0,B4/B3)", "0.0%"),
        ("Label 1 (profanity)", f"=COUNTIF({lab},1)", "0"),
        ("Label 0", f"=COUNTIF({lab},0)", "0"),
        ("Marked ambiguous", f"=COUNTIF({amb},1)", "0"),
        ("Marked not Kazakh (dropped)", f"=COUNTIF({notkk},1)", "0"),
    ]
    for i, (name, formula, fmt) in enumerate(items, start=3):
        ws.cell(i, 1, name).font = plain
        c = ws.cell(i, 2, formula)
        c.font = plain
        c.number_format = fmt
    hdr = 13
    for j, h in enumerate(["Pool", "Rows", "Labeled", "Label 1", "Label 0", "Blank"], start=1):
        c = ws.cell(hdr, j, h)
        c.font = Font(name=FONT, bold=True, color="FFFFFF")
        c.fill = HEADER_FILL
    pools = ["candidate", "lookalike", "insult", "one_star", "random"]
    for i, p in enumerate(pools, start=hdr + 1):
        ws.cell(i, 1, p).font = plain
        ws.cell(i, 2, f"=COUNTIF({pool},$A{i})").font = plain
        ws.cell(i, 4, f"=COUNTIFS({pool},$A{i},{lab},1)").font = plain
        ws.cell(i, 5, f"=COUNTIFS({pool},$A{i},{lab},0)").font = plain
        ws.cell(i, 3, f"=D{i}+E{i}").font = plain
        ws.cell(i, 6, f"=B{i}-C{i}").font = plain
    tot = hdr + len(pools) + 1
    ws.cell(tot, 1, "Total").font = bold
    for j, col in enumerate("BCDEF", start=2):
        ws.cell(tot, j, f"=SUM({col}{hdr + 1}:{col}{tot - 1})").font = bold
    ws.cell(tot + 2, 1, "Profane rows outside the candidate pool").font = plain
    ws.cell(tot + 2, 2, f"=SUM(D{hdr + 2}:D{tot - 1})").font = plain
    ws.cell(tot + 3, 1, "(these are misses of the word list; they show its recall)").font = \
        Font(name=FONT, italic=True, size=9, color="666666")
    ws.column_dimensions["A"].width = 40
    for col in "BCDEF":
        ws.column_dimensions[col].width = 12


def write_workbook(pool: pd.DataFrame, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    _guide_sheet(wb, len(pool))
    ws = wb.create_sheet("Label")
    names = [c for c, _, _, _ in COLUMNS]
    for j, (name, width, editable, tip) in enumerate(COLUMNS, start=1):
        c = ws.cell(1, j, name)
        c.font = Font(name=FONT, bold=True, color="FFFFFF")
        c.fill = HEADER_FILL
        c.alignment = Alignment(vertical="center")
        c.comment = Comment(tip, "labeling sheet")
        ws.column_dimensions[c.column_letter].width = width
    body = Font(name=FONT, size=10)
    wrap = Alignment(wrap_text=True, vertical="top")
    top = Alignment(vertical="top")
    for i, rec in enumerate(pool[names].itertuples(index=False), start=2):
        for j, (name, value) in enumerate(zip(names, rec), start=1):
            if name in ("label", "ambiguous", "not_kazakh", "maybe_name", "stars") and value not in ("", None):
                value = int(value)
            elif value is None or (isinstance(value, float) and pd.isna(value)):
                value = ""
            c = ws.cell(i, j, _clean(value) if value != "" else None)
            if isinstance(value, str) and value.startswith("="):
                c.data_type = "s"          # a review like "=)" must not become a formula
            c.font = body
            c.alignment = wrap if name == "text" else top
            c.border = Border(bottom=THIN)
            if name in EDITABLE:
                c.fill = YELLOW
    last = len(pool) + 1
    col = {n: ws.cell(1, j).column_letter for j, n in enumerate(names, start=1)}
    validations = [
        ("label", '"0,1"', "Use 1 (profanity) or 0 (no profanity)."),
        ("ambiguous", '"1"', "Use 1 or leave blank."),
        ("not_kazakh", '"1"', "Use 1 or leave blank."),
        ("prof_lang", '"kk,ru,en,mix"', "Use kk, ru, en or mix."),
    ]
    for name, options, msg in validations:
        dv = DataValidation(type="list", formula1=options, allow_blank=True,
                            showErrorMessage=True, errorTitle="Not allowed", error=msg)
        ws.add_data_validation(dv)
        dv.add(f"{col[name]}2:{col[name]}{last}")
    # Color a blank label orange, and the text light blue if it may contain a name.
    ws.conditional_formatting.add(f"{col['label']}2:{col['label']}{last}", FormulaRule(
        formula=[f'LEN({col["label"]}2)=0'], fill=PatternFill("solid", fgColor="F8CBAD")))
    ws.conditional_formatting.add(f"{col['text']}2:{col['text']}{last}", FormulaRule(
        formula=[f'{col["maybe_name"]}2=1'], fill=PatternFill("solid", fgColor="DDEBF7")))
    for name in ("uid", "suggested", "hits", "pool", "domain", "stars", "maybe_name"):
        for i in range(2, last + 1):
            ws[f"{col[name]}{i}"].fill = GREY
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{ws.cell(1, len(names)).column_letter}{last}"
    _progress_sheet(wb, len(pool))
    wb.active = wb.sheetnames.index("Label")
    wb.save(file)


def read_workbook(file: Path) -> pd.DataFrame:
    """Read the Label sheet (xlsx) or a CSV export of it. Extra columns are ignored."""
    file = Path(file)
    if file.suffix.lower() == ".csv":
        df = pd.read_csv(file, dtype=str, keep_default_na=False)
    else:
        df = pd.read_excel(file, sheet_name="Label", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for c in ("label", "ambiguous", "not_kazakh", "stars"):
        df[c] = pd.to_numeric(df[c].str.strip(), errors="coerce")
    df["prof_lang"] = df["prof_lang"].str.strip().str.lower()
    return df
