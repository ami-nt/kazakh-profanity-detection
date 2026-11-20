# Data card

**Content warning:** contains obscene language. That is the subject of the task.

## Files

| File | What it is |
|---|---|
| `labeled.csv` | Final dataset: one row per real comment, with label, flags, near-duplicate group and split |
| `splits/train.csv`, `val.csv`, `test.csv` | The same rows, split 70/15/15 (seed 42) |
| `splits/train_synthetic.csv` | Synthetic training copies, flagged `is_synthetic = 1`, with `parent_uid` and `aug_type` |
| `splits/split_assignment.csv` | Frozen `uid -> group_id, split` mapping, reused on every rebuild |
| `dataset_stats.json` | Counts: classes, splits, pools, domains, duplicates, synthetic examples |
| `raw/` | KazSAnDRA CSVs, not committed. `python -m src.download_data` fetches them |

## Columns of `labeled.csv`

| Column | Meaning |
|---|---|
| `uid` | KazSAnDRA `custom_id` of the review |
| `text` | The comment, with contact details replaced by `<PHONE>`, `<EMAIL>`, `<URL>`, `<USER>`, `<NUMBER>`, and private names by `<NAME>` |
| `label` | 1 = contains profanity, 0 = does not (see `docs/labeling_guidelines.md`) |
| `ambiguous` | 1 = the annotator was unsure and kept the more likely label |
| `prof_lang` | For label 1: language of the profane word(s): `kk`, `ru`, `en`, `mix` (or `unknown`) |
| `domain` | KazSAnDRA domain: `appstore`, `market`, `mapping`, `bookstore` |
| `stars` | The review's rating, 1–5 |
| `pool` | Why the comment was selected: `candidate`, `lookalike`, `insult`, `one_star`, `random` |
| `note` | The annotator's comment, if any |
| `group_id` | Near-duplicate group, named after its smallest uid |
| `split` | `train`, `val` or `test` |

## Source and license

The data is derived from KazSAnDRA (Yeshpanov & Varol, 2024), CC BY 4.0, from
<https://github.com/IS2AI/KazSAnDRA> at commit `d769cf7f7bd62b59f88915508d4bc062310b54d6`. The
files used are `06_sc_train_ib`, `09_sc_valid` and `10_sc_test`, which together contain every
review with all five ratings. KazSAnDRA's own train/valid/test split is not used. This project
makes its own split. Changes made here: the labels, contact-detail scrubbing, the selection of
comments, and the splits.
