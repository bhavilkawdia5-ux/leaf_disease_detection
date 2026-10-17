"""Tests for ml/src/dataset.py. Run: python -m unittest discover -s ml/tests"""
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR / "src"))
import dataset  # noqa: E402

CLASSES = ML_DIR / "classes.yaml"
_colour = 0


def png_bytes() -> bytes:
    """A small PNG that differs from every other one made in this run."""
    global _colour
    _colour += 1
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (_colour % 256, _colour // 256, 7)).save(buf, "PNG")
    return buf.getvalue()


def write(path: Path, data: bytes = None) -> bytes:
    data = png_bytes() if data is None else data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


class DatasetTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name) / "data"
        self.raw = self.data / "raw"
        self.sources = Path(tmp.name) / "sources.yaml"
        self.sources.write_text(
            "sources:\n"
            "  pv:\n"
            "    kind: lab\n"
            "    require_dir: color\n"
            "    exclude: [augment]\n"
            "    map:\n"
            "      Tomato___healthy: tomato__healthy\n"
            "      Tomato___Early_blight: tomato__early_blight\n"
            "      Tomato___Leaf_Mold: skip\n"
            "  own:\n"
            "    kind: field\n"
            "    map: auto\n",
            encoding="utf-8")

    def run_build(self, dry_run=False):
        return dataset.build(self.data, CLASSES, self.sources, dry_run)

    def processed_files(self):
        return sorted(p.relative_to(self.data / "processed").as_posix()
                      for p in (self.data / "processed").rglob("*.png"))

    def test_maps_folders_and_writes_train_val(self):
        for i in range(20):
            write(self.raw / "pv/plantvillage dataset/color/Tomato___healthy" / f"{i}.png")
        write(self.raw / "own/onion__purple_blotch/noon/a.png")

        result = self.run_build()

        self.assertEqual(result.errors, [])
        classes = sorted({r["class_id"] for r in result.rows})
        self.assertEqual(classes, ["onion__purple_blotch", "tomato__healthy"])
        files = self.processed_files()
        self.assertEqual(len(files), 21)
        for f in files:
            split, class_id, name = f.split("/")
            self.assertIn(split, dataset.SPLITS)
            self.assertTrue(name.startswith(("pv__", "own__")))
        with open(self.data / "processed/manifest.csv", encoding="utf-8") as fh:
            self.assertEqual(len(list(csv.DictReader(fh))), 21)

    def test_require_dir_exclude_and_skip(self):
        write(self.raw / "pv/grayscale/Tomato___healthy/a.png")
        write(self.raw / "pv/color_augmented/color/Tomato___healthy/b.png")
        write(self.raw / "pv/color/Tomato___Leaf_Mold/c.png")

        result = self.run_build()

        self.assertEqual(result.rows, [])
        self.assertEqual(dict(result.unmapped), {})

    def test_unmapped_folder_is_reported(self):
        write(self.raw / "pv/color/Tomato___Target_Spot/a.png")
        write(self.raw / "pv/color/Tomato___Target_Spot/b.png")

        result = self.run_build()

        self.assertEqual(dict(result.unmapped), {("pv", "Tomato___Target_Spot"): 2})

    def test_duplicates_dropped(self):
        data = write(self.raw / "pv/color/Tomato___healthy/a.png")
        write(self.raw / "own/tomato__healthy/a_copy.png", data)

        result = self.run_build()

        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.duplicates, 1)

    def test_same_image_under_two_classes_is_left_out(self):
        data = write(self.raw / "pv/color/Tomato___healthy/a.png")
        write(self.raw / "pv/color/Tomato___Early_blight/a.png", data)

        result = self.run_build()

        self.assertEqual(result.rows, [])
        self.assertEqual(result.conflicts, 2)
        self.assertEqual(self.processed_files(), [])

    def test_field_test_never_used_and_never_changed(self):
        field_img = self.data / "field_test/tomato__healthy/shadow/f.png"
        data = write(field_img)
        write(self.raw / "own/tomato__healthy/f_again.png", data)
        write(self.raw / "own/tomato__healthy/other.png")
        before = hashlib.sha1(field_img.read_bytes()).hexdigest()
        mtime = field_img.stat().st_mtime_ns

        result = self.run_build()

        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.field_test_overlap, 1)
        self.assertEqual(result.field_test["tomato__healthy"], 1)
        self.assertNotIn(dataset.sha1(data), {r["sha1"] for r in result.rows})
        self.assertEqual(hashlib.sha1(field_img.read_bytes()).hexdigest(), before)
        self.assertEqual(field_img.stat().st_mtime_ns, mtime)

    def test_reads_inside_zip(self):
        zpath = self.raw / "pv/plantvillage.zip"
        zpath.parent.mkdir(parents=True)
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("plantvillage dataset/color/Tomato___healthy/a.png", png_bytes())
            zf.writestr("plantvillage dataset/color/Tomato___healthy/readme.txt", "x")

        result = self.run_build()

        self.assertEqual([r["class_id"] for r in result.rows], ["tomato__healthy"])
        self.assertIn("plantvillage.zip!", result.rows[0]["origin"])

    @unittest.skipUnless(__import__("importlib").util.find_spec("pyarrow"), "needs pyarrow")
    def test_reads_hugging_face_parquet(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.sources.write_text(
            "sources:\n"
            "  cold:\n"
            "    kind: field\n"
            "    map:\n"
            "      Iris yellow virus: onion__iris_yellow_spot_virus\n"
            "      healthy: onion__healthy\n",
            encoding="utf-8")
        table = pa.table({
            "image": [{"bytes": png_bytes(), "path": "a.png"},
                      {"bytes": png_bytes(), "path": "b.png"}],
            "label": [0, 1],
        })
        features = {"info": {"features": {"label": {"names": ["Iris yellow virus", "healthy"]}}}}
        table = table.replace_schema_metadata({"huggingface": json.dumps(features)})
        path = self.raw / "cold/raw/train-00000-of-00001.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(table, path)

        result = self.run_build()

        self.assertEqual(sorted(r["class_id"] for r in result.rows),
                         ["onion__healthy", "onion__iris_yellow_spot_virus"])

    def test_unreadable_image_skipped(self):
        write(self.raw / "own/tomato__healthy/broken.png", b"not an image")

        result = self.run_build()

        self.assertEqual(result.rows, [])
        self.assertEqual(len(result.unreadable), 1)

    def test_unknown_class_in_sources_is_an_error(self):
        self.sources.write_text(
            "sources:\n  pv:\n    kind: lab\n    map:\n      Downy: onion__downy_mildew\n",
            encoding="utf-8")
        write(self.raw / "pv/Downy/a.png")

        result = self.run_build()

        self.assertEqual(len(result.errors), 1)
        self.assertIn("onion__downy_mildew", result.errors[0])
        self.assertFalse((self.data / "processed").exists())

    def test_dry_run_writes_nothing(self):
        write(self.raw / "own/tomato__healthy/a.png")

        result = self.run_build(dry_run=True)

        self.assertEqual(len(result.rows), 1)
        self.assertFalse((self.data / "processed").exists())

    def test_rerun_rebuilds_and_split_is_stable(self):
        for i in range(30):
            write(self.raw / "own/tomato__healthy" / f"{i}.png")
        self.run_build()
        first = self.processed_files()
        (self.raw / "own/tomato__healthy/0.png").unlink()
        self.run_build()
        second = self.processed_files()

        self.assertEqual(len(second), len(first) - 1)
        self.assertTrue(set(second) <= set(first))

    def test_split_is_about_70_15_15(self):
        digests = [hashlib.sha1(str(i).encode()).hexdigest() for i in range(10000)]
        splits = [dataset.split_for(d) for d in digests]
        for split, share in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
            self.assertAlmostEqual(splits.count(split) / len(digests), share, delta=0.015)

    def test_paused_class_is_left_out(self):
        write(self.raw / "own/not_a_leaf/a.png")
        write(self.raw / "own/tomato__healthy/b.png")

        result = self.run_build()

        self.assertEqual([r["class_id"] for r in result.rows], ["tomato__healthy"])
        self.assertEqual(dict(result.paused), {"not_a_leaf": 1})
        self.assertIn("not_a_leaf", result.paused_ids)
        self.assertNotIn("not_a_leaf", result.class_ids)

    def test_missing_kind_is_an_error(self):
        self.sources.write_text("sources:\n  own:\n    map: auto\n", encoding="utf-8")

        result = self.run_build()

        self.assertEqual(len(result.errors), 1)
        self.assertIn("kind", result.errors[0])

    def test_train_cap_drops_lab_photos_first(self):
        for i in range(40):
            write(self.raw / "pv/color/Tomato___healthy" / f"lab{i}.png")
            write(self.raw / "own/tomato__healthy" / f"field{i}.png")
        old_cap = dataset.MAX_TRAIN_PER_CLASS
        dataset.MAX_TRAIN_PER_CLASS = 30
        self.addCleanup(setattr, dataset, "MAX_TRAIN_PER_CLASS", old_cap)

        result = self.run_build()

        train = [r for r in result.rows if r["split"] == "train"]
        self.assertEqual(len(train), 30)
        field_rows = [r for r in result.rows if r["kind"] == "field"]
        # Every field photo that hashed into train is kept; only lab photos are cut.
        self.assertTrue(all(r in train for r in field_rows if r["split"] == "train"))
        self.assertEqual(result.capped["tomato__healthy"],
                         sum(dataset.split_for(dataset.sha1(p.read_bytes())) == "train"
                             for p in self.raw.rglob("*.png")) - 30)
        on_disk = {p.name for p in (self.data / "processed/train/tomato__healthy").iterdir()}
        self.assertEqual(on_disk, {Path(r["file"]).name for r in train})
        # Val and test are never capped.
        self.assertGreater(sum(r["split"] != "train" for r in result.rows), 10)

    def test_manifest_records_kind(self):
        write(self.raw / "pv/color/Tomato___healthy/a.png")
        write(self.raw / "own/tomato__early_blight/b.png")
        self.run_build()

        with open(self.data / "processed/manifest.csv", encoding="utf-8") as fh:
            kinds = {r["source"]: r["kind"] for r in csv.DictReader(fh)}
        self.assertEqual(kinds, {"pv": "lab", "own": "field"})

    def test_preview_sheet_per_class(self):
        for i in range(3):
            write(self.raw / "own/tomato__healthy" / f"{i}.png")
        result = self.run_build()

        out = dataset.write_previews(self.data, result.rows, result.class_ids)

        self.assertEqual([p.name for p in out.iterdir()], ["tomato__healthy.jpg"])
        with Image.open(out / "tomato__healthy.jpg") as im:
            self.assertEqual(im.size, (5 * dataset.PREVIEW_TILE, 5 * dataset.PREVIEW_TILE))

    def test_report_prints_counts_and_notes(self):
        write(self.raw / "own/tomato__healthy/a.png")
        write(self.raw / "pv/color/Tomato___Target_Spot/a.png")
        result = self.run_build()

        out = io.StringIO()
        with redirect_stdout(out):
            dataset.print_report(result)
        text = out.getvalue()

        self.assertIn("tomato__healthy", text)
        self.assertIn("not_a_leaf", text)
        self.assertIn("Unmapped folder pv/'Tomato___Target_Spot'", text)
        self.assertIn("Under 300 images", text)

    def test_field_test_counted_by_lighting(self):
        write(self.data / "field_test/tomato__healthy/shadow/a.png")
        write(self.data / "field_test/tomato__healthy/good/b.png")
        write(self.data / "field_test/tomato__healthy/c.png")
        write(self.data / "field_test/tomato__healthy/Shade/d.png")
        result = self.run_build()

        self.assertEqual(result.field_test["tomato__healthy"], 4)
        self.assertEqual(dict(result.field_lighting),
                         {"shadow": 1, "good": 1, "untagged": 1, "Shade": 1})
        out = io.StringIO()
        with redirect_stdout(out):
            dataset.print_report(result)
        text = out.getvalue()
        self.assertIn("Field test by lighting: good 1, shadow 1, Shade 1, untagged 1", text)
        self.assertIn("No field test photos in: harsh_sun, low_light", text)
        self.assertIn("found: Shade, untagged", text)


class RealConfigTest(unittest.TestCase):
    def test_repo_sources_yaml_matches_classes_yaml(self):
        ids = dataset.load_class_ids(CLASSES)
        _, errors = dataset.load_sources(ML_DIR / "sources.yaml", ids)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
