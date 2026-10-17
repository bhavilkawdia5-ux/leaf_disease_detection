"""Test a trained model per lighting condition and write ml/reports/baseline.md.

Phase 3, see docs/SYSTEM_DESIGN.md.

    models/baseline.keras, labels.txt, train_info.json   from train.py
    data/processed/manifest.csv     val and test images (from dataset.py)
    data/field_test/<class id>/<lighting>/...  real photos with a lighting tag
    reports/baseline.md             the report (--report to change)

Public photos have no lighting tags, so each test image is checked four times
(decided 2026-10-17, docs/datasets.md): as it is ("good") and with a synthetic
shadow, harsh sun or low light, as in the Phase 0 practice run. The effects are
seeded from each image's hash, so every run sees exactly the same images.
Accuracy is reported for field and lab photos apart; only the field number
counts as real-world accuracy. Photos in data/field_test, if any, are reported
by their real lighting folder.

The test split and data/field_test are only ever measured here, never trained
or tuned on.

Usage:
    python ml/src/evaluate.py                       # model in ml/models
    python ml/src/evaluate.py --data /content/data  # in Colab
    python ml/src/evaluate.py --limit 20            # quick check
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR / "src"))
import dataset  # noqa: E402
import train  # noqa: E402

LIGHTING = dataset.LIGHTING     # good, shadow, harsh_sun, low_light
KINDS = dataset.KINDS           # lab, field
SHADOW_FACTOR = 0.4             # Phase 0 practice run: shadowed half x 0.4
SHADOW_COVER = (0.3, 0.6)       # share of the image in shadow, picked per image
HARSH_GAIN = 1.6                # harsh sun: brighter, so highlights wash out at 255
HARSH_OFFSET = 40
LOW_LIGHT_FACTOR = 0.35         # Phase 0 practice run: dark = pixels x 0.35
LOW_LIGHT_NOISE = 6.0           # camera noise in the dark, std dev in 0-255 units
NOT_SURE_BELOW = 0.70           # Phase 0 practice value; Phase 4 picks the real one
TARGET_DROP = 10                # SYSTEM_DESIGN metrics: within 10 points of good light
TARGET_CONFIDENT_WRONG = 5      # SYSTEM_DESIGN metrics: under 5% wrong and confident
EVAL_BATCH = 64


def rng_for(digest: str) -> np.random.Generator:
    return np.random.default_rng(int(digest[:8], 16))


def shadow(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A hard-edged shadow: one side of a random straight line, x SHADOW_FACTOR."""
    h, w = image.shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    along = xx * np.cos(angle) + yy * np.sin(angle)
    cover = rng.uniform(*SHADOW_COVER)
    mask = along < np.quantile(along, cover)
    out = image.copy()
    out[mask] *= SHADOW_FACTOR
    return out


def harsh_sun(image: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    return np.clip(image * HARSH_GAIN + HARSH_OFFSET, 0, 255)


def low_light(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, LOW_LIGHT_NOISE, image.shape)
    return np.clip(image * LOW_LIGHT_FACTOR + noise, 0, 255).astype(np.float32)


def apply_lighting(image: np.ndarray, lighting: str, digest: str) -> np.ndarray:
    """The image as seen under one lighting condition; same digest, same result."""
    image = image.astype(np.float32)
    if lighting == "good":
        return image
    effect = {"shadow": shadow, "harsh_sun": harsh_sun, "low_light": low_light}[lighting]
    return effect(image, rng_for(digest))


def percent(hits: int, n: int):
    return 100.0 * hits / n if n else None


def accuracy(results: list, **match) -> tuple:
    """(accuracy in %, image count) over results whose fields equal `match`."""
    rows = [r for r in results if all(r[k] == v for k, v in match.items())]
    return percent(sum(r["pred"] == r["class_id"] for r in rows), len(rows)), len(rows)


def per_class_average(results: list, class_ids: list, **match):
    """Mean of the per-class accuracies, so small classes count as much as big ones."""
    scores = [accuracy(results, class_id=cid, **match)[0] for cid in class_ids]
    scores = [s for s in scores if s is not None]
    return sum(scores) / len(scores) if scores else None


def confident_wrong(results: list, **match):
    """% of images answered wrongly with confidence at or above NOT_SURE_BELOW."""
    rows = [r for r in results if all(r[k] == v for k, v in match.items())]
    bad = sum(r["pred"] != r["class_id"] and r["conf"] >= NOT_SURE_BELOW for r in rows)
    return percent(bad, len(rows))


def confusion(results: list, class_ids: list, **match) -> list:
    """matrix[true][predicted] counts, in class_ids order."""
    index = {cid: i for i, cid in enumerate(class_ids)}
    matrix = [[0] * len(class_ids) for _ in class_ids]
    for r in results:
        if all(r[k] == v for k, v in match.items()):
            matrix[index[r["class_id"]]][index[r["pred"]]] += 1
    return matrix


def top_confusions(matrix: list, class_ids: list, count: int = 5) -> list:
    pairs = [(matrix[t][p], class_ids[t], class_ids[p])
             for t in range(len(class_ids)) for p in range(len(class_ids))
             if t != p and matrix[t][p]]
    return sorted(pairs, reverse=True)[:count]


def fmt(value, signed=False) -> str:
    if value is None:
        return "—"
    return f"{value:+.1f}" if signed else f"{value:.1f}%"


def field_test_rows(field_dir: Path, class_ids: list) -> list:
    """Real photos in data/field_test/<class id>/<lighting>/, tagged by folder."""
    rows = []
    if not field_dir.is_dir():
        return rows
    for path in sorted(p for p in field_dir.rglob("*") if p.is_file()):
        parts = path.relative_to(field_dir).parts
        if path.suffix.lower() not in dataset.IMAGE_EXTS or len(parts) < 3:
            continue
        if parts[0] in class_ids and parts[1] in LIGHTING:
            rows.append({"class_id": parts[0], "lighting": parts[1], "path": path,
                         "sha1": dataset.sha1(path.read_bytes())})
    return rows


def predict(model, items: list, class_ids: list) -> list:
    """items: dicts with path, class_id, sha1 and the lightings to test.
    Returns one result per (item, lighting) with pred and conf."""
    results = []
    for start in range(0, len(items), EVAL_BATCH):
        chunk = items[start:start + EVAL_BATCH]
        batch, meta = [], []
        for item in chunk:
            image = train.load_image(str(item["path"])).numpy()
            for lighting in item["lightings"]:
                batch.append(apply_lighting(image, lighting, item["sha1"]))
                meta.append((item, lighting))
        probs = model.predict_on_batch(np.stack(batch))
        for (item, lighting), p in zip(meta, np.asarray(probs)):
            top = int(np.argmax(p))
            results.append({**item["tags"], "class_id": item["class_id"], "lighting": lighting,
                            "pred": class_ids[top], "conf": float(p[top])})
        print(f"  {min(start + EVAL_BATCH, len(items))}/{len(items)} images", end="\r")
    print()
    return results


def render_report(info: dict, class_ids: list, val: list, test: list, real: list,
                  title: str) -> str:
    out = [f"# {title}", ""]
    out += [f"Written by `ml/src/evaluate.py` on {date.today().isoformat()}."
            f" Model: {info.get('model', '?')}, {info.get('image_size', '?')}x"
            f"{info.get('image_size', '?')}, trained by `ml/src/train.py` on"
            f" {info.get('train_images', '?')} images ({info.get('epochs_head', '?')}"
            f" epochs last layer, {info.get('epochs_fine_run', '?')} fine-tuning,"
            f" {info.get('minutes', '?')} minutes; best epoch {info.get('best_epoch', '?')})."
            " No lighting augmentation in training.", ""]
    if info.get("limit_per_class"):
        out += [f"**Quick check only:** at most {info['limit_per_class']} images per class.", ""]

    val_acc, val_n = accuracy(val)
    out += ["## Summary", ""]
    good_field = accuracy(test, lighting="good", kind="field")[0]
    out += [f"- Validation accuracy (good light, {val_n} images): {fmt(val_acc)}",
            f"- Test accuracy, field photos, good light: {fmt(good_field)}"]
    for lighting in LIGHTING[1:]:
        acc = accuracy(test, lighting=lighting, kind="field")[0]
        drop = acc - good_field if acc is not None and good_field is not None else None
        out.append(f"- {lighting}: {fmt(acc)} ({fmt(drop, signed=True)} points)")
    out += ["", f"Target: shadow and harsh sun within {TARGET_DROP} points of good light;"
            f" under {TARGET_CONFIDENT_WRONG}% wrong answers given with {NOT_SURE_BELOW:.0%}"
            " or more confidence (SYSTEM_DESIGN, Testing and metrics).", ""]

    out += ["## Test accuracy by lighting (synthetic)", "",
            "Each test image is checked unchanged (good) and with a synthetic effect:"
            f" shadow = one side of a straight line x {SHADOW_FACTOR}"
            f" ({SHADOW_COVER[0]:.0%} to {SHADOW_COVER[1]:.0%} of the image);"
            f" harsh_sun = x {HARSH_GAIN} + {HARSH_OFFSET}, clipped at 255;"
            f" low_light = x {LOW_LIGHT_FACTOR} plus noise (std {LOW_LIGHT_NOISE:g}).", "",
            "| Lighting | Field | Lab | All | Field, per-class average | Field drop vs good"
            " | Field wrong and confident |",
            "|---|---|---|---|---|---|---|"]
    for lighting in LIGHTING:
        f_acc, f_n = accuracy(test, lighting=lighting, kind="field")
        l_acc, l_n = accuracy(test, lighting=lighting, kind="lab")
        a_acc, _ = accuracy(test, lighting=lighting)
        drop = f_acc - good_field if f_acc is not None and good_field is not None else None
        out.append(f"| {lighting} | {fmt(f_acc)} | {fmt(l_acc)} | {fmt(a_acc)}"
                   f" | {fmt(per_class_average(test, class_ids, lighting=lighting, kind='field'))}"
                   f" | {'—' if lighting == 'good' else fmt(drop, signed=True)}"
                   f" | {fmt(confident_wrong(test, lighting=lighting, kind='field'))} |")
    f_n = accuracy(test, lighting="good", kind="field")[1]
    l_n = accuracy(test, lighting="good", kind="lab")[1]
    out += ["", f"{f_n} field and {l_n} lab test images per lighting condition.", ""]

    if real:
        out += ["## Real field test photos (data/field_test)", "",
                "| Lighting | Accuracy | Images | Wrong and confident |", "|---|---|---|---|"]
        for lighting in LIGHTING:
            acc, n = accuracy(real, lighting=lighting)
            out.append(f"| {lighting} | {fmt(acc)} | {n}"
                       f" | {fmt(confident_wrong(real, lighting=lighting))} |")
        out.append("")

    out += ["## Accuracy per class (test, good light)", "",
            "| Class | Field | Field images | Lab | Lab images | Shadow (field) |"
            " Harsh sun (field) | Low light (field) |",
            "|---|---|---|---|---|---|---|---|"]
    for cid in class_ids:
        f_acc, f_n = accuracy(test, lighting="good", kind="field", class_id=cid)
        l_acc, l_n = accuracy(test, lighting="good", kind="lab", class_id=cid)
        others = [fmt(accuracy(test, lighting=lt, kind="field", class_id=cid)[0])
                  for lt in LIGHTING[1:]]
        out.append(f"| {cid} | {fmt(f_acc)} | {f_n} | {fmt(l_acc)} | {l_n} | "
                   + " | ".join(others) + " |")
    out.append("")

    matrix = confusion(test, class_ids, lighting="good")
    out += ["## Confusion matrix (test, good light, field and lab)", "",
            "Rows are the true class, columns the model's answer, by number:", ""]
    out += [f"{i + 1}. {cid}" for i, cid in enumerate(class_ids)]
    out += ["", "| true \\ answer | " + " | ".join(str(i + 1) for i in range(len(class_ids)))
            + " |", "|---" * (len(class_ids) + 1) + "|"]
    for i, row in enumerate(matrix):
        cells = [f"**{n}**" if i == j else (str(n) if n else ".") for j, n in enumerate(row)]
        out.append(f"| {i + 1} | " + " | ".join(cells) + " |")
    out.append("")
    pairs = top_confusions(matrix, class_ids)
    if pairs:
        out += ["Most common mix-ups:", ""]
        out += [f"- {t} taken for {p}: {n}" for n, t, p in pairs]
        out.append("")
    return "\n".join(out)


def evaluate(data_dir: Path, model_dir: Path, report: Path, limit: int = 0,
             title: str = "Phase 3 baseline") -> str:
    import keras
    labels_path = model_dir / "labels.txt"
    model_path = model_dir / "baseline.keras"
    if not model_path.is_file():
        sys.exit(f"{model_path} not found: run python ml/src/train.py first")
    class_ids = labels_path.read_text(encoding="utf-8").split()
    info_path = model_dir / "train_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    model = keras.models.load_model(model_path)

    rows = train.read_manifest(data_dir)

    def items(split, lightings):
        return [{"path": data_dir / r["file"], "class_id": r["class_id"], "sha1": r["sha1"],
                 "lightings": lightings, "tags": {"kind": r["kind"]}}
                for r in train.split_rows(rows, split, class_ids, limit)]

    print("Validation split, good light")
    val = predict(model, items("val", ("good",)), class_ids)
    print(f"Test split, {len(LIGHTING)} lighting conditions")
    test = predict(model, items("test", LIGHTING), class_ids)
    real_items = [{"path": r["path"], "class_id": r["class_id"], "sha1": r["sha1"],
                   "lightings": ("good",), "tags": {"kind": "field", "tag": r["lighting"]}}
                  for r in field_test_rows(data_dir / "field_test", class_ids)]
    real = []
    if real_items:
        print(f"data/field_test, {len(real_items)} photos")
        real = predict(model, real_items, class_ids)
        # The photo is already in its real lighting: report it under its folder tag.
        for r in real:
            r["lighting"] = r.pop("tag")

    text = render_report(info, class_ids, val, test, real, title)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    print(f"Wrote {report}")
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=ML_DIR / "data",
                        help="folder holding processed/ and field_test/")
    parser.add_argument("--models", type=Path, default=ML_DIR / "models")
    parser.add_argument("--report", type=Path, default=ML_DIR / "reports" / "baseline.md")
    parser.add_argument("--limit", type=int, default=0,
                        help="use at most this many images per class (quick check)")
    args = parser.parse_args(argv)
    evaluate(args.data, args.models, args.report, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
