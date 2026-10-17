"""Tests for ml/src/evaluate.py and the TensorFlow-free parts of ml/src/train.py.
Run: python -m unittest discover -s ml/tests"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR / "src"))
import evaluate  # noqa: E402
import train  # noqa: E402

DIGEST = "05f384600b3bb5beeac3a70f3114f57f4e8adc83"


def grey(value=150.0, size=64) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.float32)


def result(class_id, pred, conf=0.9, lighting="good", kind="field"):
    return {"class_id": class_id, "pred": pred, "conf": conf,
            "lighting": lighting, "kind": kind}


class LightingTest(unittest.TestCase):
    def test_good_is_unchanged(self):
        image = grey()
        np.testing.assert_array_equal(evaluate.apply_lighting(image, "good", DIGEST), image)

    def test_same_digest_same_image(self):
        image = np.random.default_rng(1).uniform(0, 255, (64, 64, 3)).astype(np.float32)
        for lighting in evaluate.LIGHTING:
            a = evaluate.apply_lighting(image, lighting, DIGEST)
            b = evaluate.apply_lighting(image, lighting, DIGEST)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.shape, image.shape)
            self.assertGreaterEqual(a.min(), 0)
            self.assertLessEqual(a.max(), 255)

    def test_shadow_darkens_part_of_the_image(self):
        out = evaluate.apply_lighting(grey(), "shadow", DIGEST)
        shaded = np.isclose(out[..., 0], 150 * evaluate.SHADOW_FACTOR).mean()
        self.assertTrue(evaluate.SHADOW_COVER[0] - 0.05 <= shaded <= evaluate.SHADOW_COVER[1] + 0.05)
        self.assertTrue(np.isclose(out[..., 0], 150).any(), "the rest stays lit")

    def test_shadow_differs_per_image(self):
        a = evaluate.apply_lighting(grey(), "shadow", DIGEST)
        b = evaluate.apply_lighting(grey(), "shadow", "ffffffff" + DIGEST[8:])
        self.assertFalse(np.array_equal(a, b))

    def test_harsh_sun_washes_out_highlights(self):
        out = evaluate.apply_lighting(grey(200), "harsh_sun", DIGEST)
        self.assertTrue((out == 255).all())
        self.assertGreater(evaluate.apply_lighting(grey(50), "harsh_sun", DIGEST).mean(), 50)

    def test_low_light_is_dark(self):
        out = evaluate.apply_lighting(grey(), "low_light", DIGEST)
        self.assertAlmostEqual(out.mean(), 150 * evaluate.LOW_LIGHT_FACTOR, delta=2)
        self.assertGreater(out.std(), 0, "has noise")


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.results = [
            result("a", "a"), result("a", "b", conf=0.95), result("a", "a"),
            result("b", "b", kind="lab"), result("b", "a", conf=0.5, kind="lab"),
            result("a", "b", lighting="shadow"),
        ]

    def test_accuracy_filters(self):
        self.assertEqual(evaluate.accuracy(self.results, lighting="good"), (60.0, 5))
        self.assertEqual(evaluate.accuracy(self.results, lighting="good", kind="lab"), (50.0, 2))
        self.assertEqual(evaluate.accuracy(self.results, lighting="harsh_sun"), (None, 0))

    def test_per_class_average_weights_classes_equally(self):
        # a: 2 of 3, b: 1 of 2 -> (66.7 + 50) / 2
        avg = evaluate.per_class_average(self.results, ["a", "b", "c"], lighting="good")
        self.assertAlmostEqual(avg, (200 / 3 + 50) / 2)

    def test_confident_wrong_ignores_unsure_answers(self):
        # wrong with 0.95 counts, wrong with 0.5 is below NOT_SURE_BELOW
        self.assertEqual(evaluate.confident_wrong(self.results, lighting="good"), 20.0)

    def test_confusion_matrix(self):
        matrix = evaluate.confusion(self.results, ["a", "b"], lighting="good")
        self.assertEqual(matrix, [[2, 1], [1, 1]])
        self.assertEqual(evaluate.top_confusions(matrix, ["a", "b"]),
                         [(1, "b", "a"), (1, "a", "b")])

    def test_report_has_every_section(self):
        test = [result(c, c, lighting=lt, kind=k)
                for c in ("a", "b") for lt in evaluate.LIGHTING for k in ("field", "lab")]
        test.append(result("a", "b", lighting="shadow"))
        test.append(result("b", "a", kind="lab"))
        real = [result("a", "a", lighting="shadow")]
        text = evaluate.render_report({"model": "test"}, ["a", "b"], [result("a", "a")],
                                      test, real, "Report title")
        for heading in ("# Report title", "## Summary", "## Test accuracy by lighting",
                        "## Real field test photos", "## Accuracy per class",
                        "## Confusion matrix", "Most common mix-ups"):
            self.assertIn(heading, text)
        self.assertIn("| shadow | 66.7% | 100.0% |", text)
        self.assertIn("- shadow: 66.7% (-33.3 points)", text)

    def test_field_test_rows_need_class_and_lighting_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ("a/shadow/1.jpg", "a/sunny/2.jpg", "zzz/good/3.jpg", "a/4.jpg",
                        "a/good/notes.txt"):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(rel.encode())
            rows = evaluate.field_test_rows(root, ["a"])
            self.assertEqual([(r["class_id"], r["lighting"]) for r in rows], [("a", "shadow")])


class TrainHelpersTest(unittest.TestCase):
    def test_class_weights_balance_the_loss(self):
        weights = train.class_weights([0, 0, 0, 1], 3)
        self.assertAlmostEqual(weights[0] * 3, weights[1] * 1)
        self.assertEqual(weights[2], 0.0)

    def test_split_rows_never_mixes_splits(self):
        rows = [{"split": s, "class_id": c, "sha1": f"{i:040x}"}
                for i, (s, c) in enumerate([("train", "a"), ("test", "a"), ("val", "a"),
                                            ("train", "a"), ("train", "paused")])]
        picked = train.split_rows(rows, "train", ["a"])
        self.assertEqual(len(picked), 2)
        self.assertTrue(all(r["split"] == "train" for r in picked))
        self.assertEqual(len(train.split_rows(rows, "train", ["a"], limit=1)), 1)


if __name__ == "__main__":
    unittest.main()
