"""Tests for the TensorFlow-free parts of ml/src/export_tflite.py.
Run: python -m unittest discover -s ml/tests"""
import sys
import unittest
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR / "src"))
import export_tflite  # noqa: E402


class ExportReportTest(unittest.TestCase):
    def test_passing_export(self):
        text = export_tflite.render_export_report(
            "aug_clahe", {"image_size": 224, "classes": ["a", "b"], "preprocess": "clahe"},
            8_400_000, 210, 210, {1: 0.045, 4: 0.02})
        self.assertIn("| Size | 8.4 MB | under 20 MB | yes |", text)
        self.assertIn("100.0% of 210 val photos", text)
        self.assertIn("| Time per image, this computer, 1 thread | 45 ms |", text)
        self.assertIn("4 threads | 20 ms", text)
        self.assertIn("`clahe`", text)
        self.assertIn("budget Android phone", text)

    def test_failing_checks_are_marked(self):
        text = export_tflite.render_export_report("m", {}, 25_000_000, 190, 210, {})
        self.assertIn("| Size | 25.0 MB | under 20 MB | **no** |", text)
        self.assertIn("| 99% or more | **no** |", text)


if __name__ == "__main__":
    unittest.main()
