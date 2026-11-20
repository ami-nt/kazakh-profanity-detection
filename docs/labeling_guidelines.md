# Labeling guidelines: Kazakh profanity

Version 1.1 (11 November 2026). Obscene words are shown with one letter masked (`қ*тақ`).
The patterns in `src/lexicon.py` spell them out in full.

Change in 1.1: insults, euphemistic swearing and curses (groups F–H) now count as `1`. In 1.0
they were `0`.

## Task

**Input:** one review or comment written in Kazakh. It may be mixed with Russian or English.
**Output:** `1` if the comment contains profanity, `0` if it does not.

In this project, **profanity** means obscene words (groups A–E) and also insulting words,
euphemistic swearing and curses (groups F–H).

Profanity is a property of **words**, not of tone. A furious review that uses none of these words
is `0`. A five-star review containing one of them is `1`.

## What counts as profanity (label 1)

| Group | Covers | Examples |
|---|---|---|
| A. Kazakh obscene words | Sexual and genital words and verbs, with all their forms and compounds | қ\*тақ, қ\*тақбас, қ\*тағым; ам\*, ам\*ың, амш\*қ; с\*гу, с\*гейін, с\*ктір, сг\*н |
| B. Vulgar body and excretion words | | б\*қ, б\*ғы; к\*т, к\*тің |
| C. Sexual slurs | | ж\*леп, жалап (meaning "whore"), с\*йқал, қ\*ншық (said about a person) |
| D. Russian mat and strong vulgar words inside a Kazakh comment | Includes Kazakh spellings | х\*й, нах\*й, х\*йня; п\*здец, п\*здес; за\*бал, з\*йбал, \*баный; бл\*, бл\*ть, бл\*т; с\*ка; м\*дак; г\*ндон; шл\*ха; п\*дор; г\*вно; ж\*па; д\*рьмо; х\*р, х\*рня; д\*лбаёб, д\*лбан |
| E. English profanity inside a Kazakh comment | | f\*ck, sh\*t, b\*tch, wtf |
| F. Insults (only when used to insult; see rule 6) | Aimed at a person, company, app or thing | мал, ақымақ, есек/ешак, топас, жынды, оңбаған, малғұн, иттің баласы, қатын, шірік, сасық; дурак, тупой, идиот, дебил, урод, сволочь, лох |
| G. Euphemistic swearing | Softened stand-ins for mat | блин, капец, хрен, фиг, чёрт |
| H. Curses | Even with no obscene word | "әкеңнің басы", "атаңның басы", құрып кет, қарғыс атсын |

### Mixed-language rule (group D and E)

Kazakh online writing switches between Kazakh and Russian constantly. Russian roots take Kazakh
suffixes (за\*балсыңдар, нах\*йға), so there is no consistent line between "Kazakh" and "Russian"
profanity. For moderation both have the same effect, so **any profanity inside a Kazakh comment
gets label 1**.

The `prof_lang` column records which language the profane word(s) came from: `kk`, `ru`, `en` or
`mix`. That keeps a strict Kazakh-only label available if it is ever needed:
`label_strict = label == 1 and prof_lang in {kk, mix}`.

If `prof_lang` is left blank on a label-1 row, `make_dataset` fills it from the word lists:
first the obscene seeds, then the insult, euphemism and curse words (`lexicon.insult_language`:
мал, ақымақ → `kk`; тупой, блин → `ru`; both → `mix`). Rows whose only profane word is masked
(бл\*, с\*ки) get `unknown`.

**Out of scope:** comments with no Kazakh at all, i.e. pure Russian or pure English. Mark them
`not_kazakh = 1` and they are dropped from the dataset.

## How to apply the rules

1. **One word is enough.** Position, quoting, reported speech ("ол маған қ\*тақ деді"), jokes,
   self-directed use ("men q\*nshyqpyn") and profanity used to intensify praise all count.
2. **Every spelling counts** if a Kazakh speaker would recognise the word. That includes:
   - Russian letters for Kazakh ones (котак, сигейн);
   - Latin script (qotaq, kotak, sgin, blyat);
   - mixed scripts;
   - digits used as letters (к0тақ);
   - repeated letters (бл\*яяя);
   - letters split up (қ.о.т.а.қ);
   - words glued together (қ\*тақбастар, шешеңдіс\*гейін);
   - dropped letters (қо\*ақбас);
   - typos.
3. **Masked words:**
   - Label `1` if the word is still recognisable from the letters left and the context
     (қ\*ақбас, п\*здец, с\*\*а, н\*ху).
   - Label `0` if nothing identifiable remains: `****` on its own, star ratings like `5*****`,
     sizes like `45*45`.
4. **Words that only look similar** get `0`:
   - адам, тамақ, аман, амал;
   - the name Амина, and амин ("amen");
   - көтеру ("to lift");
   - сигнал, сигарет, скин;
   - коттедж, бокс.
5. **Literal meanings** get `0`:
   - жалап = "licking" (from жалау);
   - қаншық = a female dog, e.g. in a pet-food review;
   - бок = Russian "side";
   - кот = Russian "cat";
   - көтен = a baby's bottom in a diaper review.
6. **Insult words (group F) count only when they insult.** The same word with its ordinary
   meaning gets `0`:
   - мал = livestock (мал азығы, мал шаруашылығы); "мал сияқты курьер" is `1`;
   - есек/ешак = a real donkey;
   - сасық = a product that smells ("иісі сасық"); "сасық дүкен" as an insult is `1`;
   - шірік = rotten food ("алма шірік келді");
   - қатын = "wife" in neutral use;
   - жынды used as praise ("жынды ойын" = crazy good game);
   - блин = a pancake.

   If you cannot tell whether a word insults, choose the more likely label and put
   `ambiguous = 1`.

## What does not count (label 0)

- **Harsh negative reviews** without words from groups A–H: "Сапасы өте нашар, ақшам желге кетті".
- **Literal meanings of insult words** (rule 6).
- **Plain interjections:** ойбай.
- **Words that name swearing:** боқтық сөз, боқтау, балағат, мат, жаман сөз.
- **Neutral sexual or medical words:** секс, жыныс, порно, and the masked "с\*\*с" meaning секс.
- **Emoji or gestures alone** (🖕). We label words.

## Fixed decisions for borderline words

These are fixed so the same word always gets the same label:

| Word | Label | Reason |
|---|---|---|
| б\*қ in any use, even literal ("б\*қтың исі") | 1 | The word itself is vulgar |
| к\*т | 1 | Vulgar body word |
| с\*йқал | 1 | Sexual slur |
| жезөкше | 0 | Standard word for "prostitute", not vulgar register |
| д\*лбан, д\*лбаёб | 1 | Kazakh spelling of Russian mat |
| х\*р, х\*рня | 1 | Vulgar |
| хрен | 1 | Euphemistic swearing (group G, since 1.1) |
| ойбай | 0 | Plain interjection, often positive |
| далбаса, далбасалап | 0 | Means "rubbish, nonsense" ("Далбаса ойын"), not a personal insult. Not the Russian д\*лбаёб, which is 1 |
| ч\*о | 1 | Vulgar slang |
| секс | 0 | Neutral word |
| ам\* on its own | 1 | Almost always the vulgar word. Label 0 only for a clearly different use |

## Unsure cases

Choose the more likely label, put `1` in `ambiguous` and write a few words in `note`. These rows
stay in the dataset and are reported separately in the error analysis.

## Personal information

Phone numbers, e-mails, links, @handles and long numbers (IIN, card) are replaced automatically
with `<PHONE>`, `<EMAIL>`, `<URL>`, `<USER>` and `<NUMBER>`. Rules cannot catch person names, so if
a comment names a private person, replace the name in `text` with `<NAME>`. The `maybe_name`
column is only a hint. Public figures and brand names can stay.

## Labeled examples

| Comment | Label | Why |
|---|---|---|
| Ойын керемет бл\* 5 жұлдыз | 1 | Positive review, but it contains profanity (group D) |
| Сапасы өте нашар, ақшам желге кетті | 0 | Very negative, no vulgar word |
| Котак шыга береди | 1 | қ\*тақ in Russian letters (rule 2) |
| Маған еөшірімейдіғо қ\*ақбас | 1 | Masked but recognisable (rule 3) |
| Мал сияқты курьер, ақымақ | 1 | Insults (group F) |
| Мал азығы сапалы, тез жеткізді | 0 | мал = livestock (rule 6) |
| Мысық тағамды жалап жейді | 0 | жалап = licking (rule 5) |
| Керемет программа, ұнады 5\*\*\*\*\* | 0 | Star rating, not a mask |
| Боқтық сөз жазсам да өшіреді | 0 | Names swearing; uses no swear word |

## Process (for the README)

1. **Candidates.** The seed patterns in `src/lexicon.py` select candidate comments with high
   recall.
2. **Extra blocks.** Look-alikes, insult-word matches and angry 1-star reviews are added, plus a
   random sample with a fixed domain mix. The insult block was built as hard negatives under
   version 1.0. Under 1.1 it holds both labels (insulting use = 1, literal use = 0).
3. **Pre-fills.** The sheet arrives pre-filled: strong matches 1, weak matches blank, everything
   else 0. These are suggestions only.
4. **Manual check.** The author (a native speaker) reads every row and sets the final label.
   Positives found among the non-candidates show how much the word list misses.
5. **Second pass.** Every `ambiguous` row is re-read after the first pass.
6. **Optional agreement check.** A second annotator labels 200 random rows and Cohen's κ is
   reported.
7. **Model-assisted review, train only.** After the split, out-of-fold predictions flag possible
   label errors in the training split only. Validation and test labels are frozen before any model
   is trained, so the test set stays independent of the model.
