"""Fill the number tables in README.md from data/ and results/.

    python -m src.report

Only the text between <!-- BEGIN name --> and <!-- END name --> is replaced,
so no number in the README is copied by hand.
"""
import json
import re

import pandas as pd

from .lexicon import profane_spans
from .normalize import normalize
from .utils import ROOT, load_config, path


def mask_for_display(text: str) -> str:
    """Normalized text with one letter of each profane word replaced by "*"."""
    norm = normalize(text)
    chars = list(norm)
    for start, end in profane_spans(norm):
        for i in range(start + 1, end):
            if chars[i].isalpha():
                chars[i] = "*"
                break
    return "".join(chars)


def pct(x) -> str:
    return f"{x:.1%}"


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def dataset_block(st: dict) -> str:
    sp = st["splits"]
    total = st["total_examples"]
    pos = st["class_counts"].get("1", 0)
    rows = [[s, sp[s]["n"], sp[s]["label_1"], sp[s]["label_0"], pct(sp[s]["label_1"] / max(sp[s]["n"], 1))]
            for s in ["train", "val", "test"]]
    rows.append(["**total**", total, pos, st["class_counts"].get("0", 0), pct(pos / max(total, 1))])
    syn = st.get("synthetic_train_examples", {})
    syn_rows = [[k.rsplit("_label", 1)[0], k.rsplit("_label", 1)[1], v] for k, v in sorted(syn.items())]
    pool_rows = [[p, v["n"], v["label_1"]] for p, v in st["by_pool"].items()]
    dom_rows = [[d, v["n"], v["label_1"]] for d, v in st["by_domain"].items()]
    lang = ", ".join(f"{k}: {v}" for k, v in st["prof_lang_of_positives"].items())
    return "\n\n".join([
        f"Real, human-labeled examples: **{total}** (seed {st['seed']}). "
        f"Rows dropped: {st.get('dropped_not_kazakh', 0)} not Kazakh, {st.get('dropped_unlabeled', 0)} unlabeled; "
        f"{st['exact_duplicates_removed']} exact duplicates removed; {st['near_duplicate_groups_with_2plus']} "
        f"near-duplicate groups (largest {st['largest_group']}) kept inside one split each. "
        f"Marked ambiguous: {st['ambiguous']}.",
        table(["split", "examples", "label 1", "label 0", "share of 1"], rows),
        "By selection pool (why a comment was chosen for labeling):",
        table(["pool", "examples", "label 1"], pool_rows),
        "By KazSAnDRA domain:",
        table(["domain", "examples", "label 1"], dom_rows),
        f"Language of the profane words in positives: {lang}.",
        "Synthetic examples (training split only, flagged `is_synthetic=1` in "
        "`data/splits/train_synthetic.csv`):",
        table(["type", "label", "count"], syn_rows) if syn_rows else "_none_",
    ])


def validation_block(sel: dict) -> str:
    s = sel["chosen_settings"]
    v = sel["validation_at_threshold"]
    alt = sel.get("alternative_threshold_recall_target") or {}
    abl = sel["synthetic_ablation"]
    rows = [[k.replace("_", " "), f"{x['f1']:.3f}", f"{x['precision']:.3f}", f"{x['recall']:.3f}", f"{x['ap']:.3f}"]
            for k, x in abl.items()]
    text = [
        f"{sel['settings_tried']} settings compared on the validation split. Chosen: analyzer "
        f"`{s['analyzer']}`, n-grams {s['ngram_range'][0]}-{s['ngram_range'][1]}, letter folding "
        f"`{s['fold']}`, C = {s['C']}, class weights `{s['class_weight']}`, window "
        f"`{s['window']}` words, synthetic rows used: `{s['use_synthetic']}`.",
        f"Decision threshold (best validation F1): **{sel['threshold']:.3f}**. Validation at that threshold: "
        f"F1 {v['f1']:.3f}, precision {v['precision']:.3f}, recall {v['recall']:.3f}, "
        f"average precision {v['ap']:.3f}.",
    ]
    if alt:
        text.append(f"Alternative operating point for a human-review queue (recall >= target): threshold "
                    f"{alt['threshold']:.3f} gives precision {alt['precision']:.3f}, recall {alt['recall']:.3f}.")
    text.append("Synthetic-data ablation (same settings):")
    text.append(table(["training data", "val F1", "precision", "recall", "avg precision"], rows))
    return "\n\n".join(text)


def test_block(m: dict) -> str:
    c = m["confusion_matrix"]
    pc = m["profanity_class"]
    prev = ", ".join(f"{k}: {v:.2f}" for k, v in m["expected_precision_at_prevalence"].items() if v is not None)
    return "\n\n".join([
        f"Evaluated once on {m['evaluated_at'][:10]}. Test examples: **{m['test_examples']}** "
        f"({m['test_positives']} with profanity, {m['test_negatives']} without). Threshold {m['threshold']:.3f}.",
        table(["metric (profanity class)", "value"],
              [["precision", f"{pc['precision']:.3f}"], ["recall", f"{pc['recall']:.3f}"],
               ["F1", f"{pc['f1']:.3f}"], ["false-positive rate", f"{m['false_positive_rate']:.3f}"],
               ["average precision", f"{m['average_precision']:.3f}"], ["accuracy", f"{m['accuracy']:.3f}"]]),
        table(["", "predicted 0", "predicted 1"],
              [["**true 0**", f"TN = {c['tn']}", f"FP = {c['fp']}"],
               ["**true 1**", f"FN = {c['fn']}", f"TP = {c['tp']}"]]),
        "![Confusion matrix](results/confusion_matrix.png)",
        f"Expected precision if profanity were rarer than in the test set (same recall and "
        f"false-positive rate): {prev}.",
    ])


def slices_block(df: pd.DataFrame) -> str:
    keep = df[df.slice.isin(["domain", "stars", "script", "length", "profanity language (positives)"])]
    rows = []
    for r in keep.itertuples():
        fpr = "" if pd.isna(r.false_positive_rate) else f"{r.false_positive_rate:.3f}"
        prec = f"{r.precision:.3f}" if (r.tp + r.fp) else "-"
        rec = f"{r.recall:.3f}" if r.positives else "-"
        rows.append([r.slice, r.value, r.n, r.positives, prec, rec, fpr])
    return table(["slice", "value", "n", "label 1", "precision", "recall", "FP rate"], rows)


def samples_block(pred: pd.DataFrame, seed: int) -> str:
    parts = []
    for outcome, k in [("TP", 3), ("TN", 3), ("FP", 4), ("FN", 4)]:
        sub = pred[pred.outcome == outcome]
        if len(sub):
            parts.append(sub.sample(min(k, len(sub)), random_state=seed))
    if not parts:
        return "_No predictions yet._"
    rows = [[r.outcome, mask_for_display(r.text)[:120], r.label, r.pred, f"{r.prob:.3f}"]
            for r in pd.concat(parts).itertuples()]
    return table(["outcome", "comment (normalized, profanity masked)", "true", "predicted", "P(profane)"], rows)


def replace_block(readme: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- BEGIN {name} -->\n).*?(\n<!-- END {name} -->)", re.S)
    if not pattern.search(readme):
        raise ValueError(f"marker {name} missing in README.md")
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), readme)


def main() -> None:
    cfg = load_config()
    res = path(cfg, "results_dir")
    readme_file = ROOT / "README.md"
    readme = readme_file.read_text(encoding="utf-8")
    missing = "_Not available yet: run the steps above first._"
    st_file = ROOT / "data" / "dataset_stats.json"
    readme = replace_block(readme, "dataset", dataset_block(json.loads(st_file.read_text(encoding="utf-8")))
                           if st_file.exists() else missing)
    sel_file = res / "val_selection.json"
    readme = replace_block(readme, "validation", validation_block(json.loads(sel_file.read_text(encoding="utf-8")))
                           if sel_file.exists() else missing)
    m_file = res / "test_metrics.json"
    readme = replace_block(readme, "test", test_block(json.loads(m_file.read_text(encoding="utf-8")))
                           if m_file.exists() else missing)
    sl_file = res / "test_slices.csv"
    readme = replace_block(readme, "slices", slices_block(pd.read_csv(sl_file)) if sl_file.exists() else missing)
    pr_file = res / "test_predictions.csv"
    readme = replace_block(readme, "samples", samples_block(pd.read_csv(pr_file, keep_default_na=False), cfg["seed"])
                           if pr_file.exists() else missing)
    readme_file.write_text(readme, encoding="utf-8")
    print("README.md tables updated")


if __name__ == "__main__":
    main()
