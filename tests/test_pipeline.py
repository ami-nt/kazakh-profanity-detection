"""Unit tests. Run from the repository root:  python -m unittest discover tests"""
import unittest

import numpy as np
import pandas as pd

from src import lexicon, pii
from src.augment import mask_word, to_latin
from src.make_dataset import assign_splits, near_duplicate_groups
from src.model import ProfanityClassifier, Settings, make_windows
from src.normalize import normalize


class TestNormalize(unittest.TestCase):
    def test_cases(self):
        cases = {
            "ҚОТАҚ!!!": "қотақ",
            "қ.о.т.а.қ бас": "қотақ бас",
            "б л я т ь": "блять",
            "а я в шоке": "а я в шоке",          # three single letters are left alone
            "kотақ": "котақ",                    # Latin k inside a Cyrillic word
            "сiгейiн": "сігейін",                # Latin i used for Kazakh і
            "к0тақ": "котақ",                    # digit used as a letter
            "қ*тақ": "қ*тақ",                    # mask kept when it touches a letter
            "5*****": "5",                       # star rating is not a mask
            "Кееереееметтт 👍👍": "керемет",
            "&lt;жағдай&gt;": "жағдай",
            "kaspi_kz": "kaspi kz",
            "жақсы.ойын": "жақсы ойын",
            "qotaq jaqsy": "qotaq jaqsy",        # real Latin-script Kazakh untouched
        }
        for raw, expected in cases.items():
            self.assertEqual(normalize(raw), expected, raw)

    def test_fold_and_idempotent(self):
        self.assertEqual(normalize("Қотақ", fold=True), "котак")
        for t in ["Сапасы өте жақсы!!", "қ*тақбас ойын", "<URL> жақсы"]:
            self.assertEqual(normalize(normalize(t)), normalize(t))


class TestPII(unittest.TestCase):
    def test_scrub(self):
        out = pii.scrub("Хабарлас +7 701 123 45 67, a@b.kz, https://x.kz/a, @user_1, ИИН 990101300123")
        for ph in ["<PHONE>", "<EMAIL>", "<URL>", "<USER>", "<NUMBER>"]:
            self.assertIn(ph, out)
        self.assertNotIn("701", out)
        self.assertEqual(pii.scrub("Бағасы 2 000 тг, 45*45 см"), "Бағасы 2 000 тг, 45*45 см")


class TestLexicon(unittest.TestCase):
    def test_hits_and_misses(self):
        for t in ["қотақбас ойын", "котак", "шешеңнің амы", "сгин", "пиздец", "Kotak bas"]:
            self.assertEqual(lexicon.suggest(lexicon.match(t)), "1", t)
        for t in ["адам жақсы", "Амина рахмет", "аллаға шүкір амин", "сигнал жоқ", "көтеру",
                  "сапасы нашар", "5***** керемет", "ебін тауып"]:
            self.assertEqual(lexicon.suggest(lexicon.match(t)), "0", t)

    def test_insult_language(self):
        self.assertEqual(lexicon.insult_language("Бұл ақымақ ойын"), "kk")
        self.assertEqual(lexicon.insult_language("Мал ойын ойнамаңдар"), "kk")
        self.assertEqual(lexicon.insult_language("Артём сен тупойсың"), "ru")
        self.assertEqual(lexicon.insult_language("Блин маған керек"), "ru")
        self.assertEqual(lexicon.insult_language("Мал катын, тупой"), "mix")
        self.assertEqual(lexicon.insult_language("малина тәтті, итерді"), "")


class TestAugment(unittest.TestCase):
    def test_mask_and_latin(self):
        rng = np.random.default_rng(0)
        masked = mask_word("ойын қотақбас екен", rng)
        self.assertIn("*", masked)
        self.assertTrue(masked.startswith("ойын қ"))
        self.assertIsNone(mask_word("сапасы жақсы", rng))
        self.assertEqual(to_latin("шешесі", rng), "sheshesi")


class TestModel(unittest.TestCase):
    def test_fit_predict_explain(self):
        pos = ["қотақбас ойын", "котак бас", "блять тоқтап қалады", "пиздец ойын", "сгин мал"] * 4
        neg = ["өте жақсы ойын", "сапасы нашар", "рахмет керемет", "ақшам кетті", "жеткізу ұзақ"] * 4
        m = ProfanityClassifier(Settings(ngram_range=(2, 4), min_df=1, C=4.0)).fit(pos + neg, [1] * 20 + [0] * 20)
        p = m.predict_proba(["қотақбас екен", "жақсы екен"])
        self.assertGreater(p[0], p[1])
        e = m.explain("қотақбас екен")
        self.assertAlmostEqual(e["probability"], float(p[0]), places=6)
        self.assertEqual(set(m.predict(["жақсы"])), {0})

    def test_windows(self):
        words = " ".join(f"w{i}" for i in range(70))
        w = make_windows(words, 30)
        self.assertEqual(w[0].split()[0], "w0")
        self.assertEqual(w[-1].split()[-1], "w69")
        self.assertEqual(make_windows("қысқа пікір", 30), ["қысқа пікір"])


class TestSplit(unittest.TestCase):
    def test_groups_stay_together(self):
        texts = ["өте керемет ойын екен маған ұнады", "өте керемет ойын екен маған ұнады!!",
                 "жеткізу өте ұзақ болды", "котак", "котак бас ойын екен ғой"]
        roots = near_duplicate_groups(texts, 0.8)
        self.assertEqual(roots[0], roots[1])          # near-duplicates grouped
        self.assertNotEqual(roots[3], roots[4])       # one shared word is not enough
        df = pd.DataFrame({"uid": [f"u{i}" for i in range(200)], "label": [i % 3 == 0 for i in range(200)],
                           "pool": "random"})
        df["label"] = df.label.astype(int)
        df["group_id"] = df.uid.str[:-1]               # groups of up to 10
        split = assign_splits(df, 0.15, 0.15, seed=42)
        self.assertEqual(df.assign(s=split).groupby("group_id").s.nunique().max(), 1)
        self.assertTrue(set(split) <= {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
