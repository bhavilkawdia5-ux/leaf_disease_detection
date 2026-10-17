"""Test a trained model per lighting condition and write a report to ml/reports/.

Phases 3 and 4, see docs/SYSTEM_DESIGN.md.

    <models>/model.keras (or baseline.keras), labels.txt, train_info.json   from train.py
    data/processed/manifest.csv     val and test images (from dataset.py)
    data/field_test/<class id>/<lighting>/...  real photos with a lighting tag
    reports/baseline.md             the report (--report to change)
    reports/<report name>.json      the main numbers, read by --compare

The photo goes through the same lighting fix the model was trained with
(train_info.json "preprocess"), after the synthetic lighting effect.

Phase 4: the "not sure" threshold is chosen from the val split only (all four
lighting conditions): the lowest confidence at which answers given at or above
it are wrong for under TARGET_CONFIDENT_WRONG % of val images. --baseline adds
a comparison with reports/baseline.md, and --compare puts several runs side by
side and picks the best by val accuracy averaged over the lighting conditions,
so the test split never chooses the model.

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
    python ml/src/evaluate.py --models ml/models/aug_clahe --report ml/reports/phase4_aug_clahe.md \\
        --title "Phase 4: augment + CLAHE" --baseline ml/reports/baseline.md
    python ml/src/evaluate.py --compare ml/reports/phase4_*.json \\
        --baseline ml/reports/baseline.md --report ml/reports/phase4_comparison.md
"""
import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR / "src"))
import dataset  # noqa: E402
import preprocess  # noqa: E402
import train  # noqa: E402

LIGHTING = dataset.LIGHTING     # good, shadow, harsh_sun, low_light
KINDS = dataset.KINDS           # lab, field
SHADOW_FACTOR = 0.4             # Phase 0 practice run: shadowed half x 0.4
SHADOW_COVER = (0.3, 0.6)       # share of the image in shadow, picked per image
HARSH_GAIN = 1.6                # harsh sun: brighter, so highlights wash out at 255
HARSH_OFFSET = 40
LOW_LIGHT_FACTOR = 0.35         # Phase 0 practice run: dark = pixels x 0.35
LOW_LIGHT_NOISE = 6.0           # camera noise in the dark, std dev in 0-255 units
NOT_SURE_BELOW = 0.70           # Phase 0 practice value, kept so runs compare with the baseline
TARGET_DROP = 10                # SYSTEM_DESIGN metrics: within 10 points of good light
TARGET_CONFIDENT_WRONG = 5      # SYSTEM_DESIGN metrics: under 5% wrong and confident
TARGET_NOT_SURE = 20            # SYSTEM_DESIGN metrics: "not sure" rate under 20%
THRESHOLDS = [round(0.30 + 0.01 * i, 2) for i in range(70)]   # 0.30 to 0.99, tried in order
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


def confident_wrong(results: list, threshold: float = NOT_SURE_BELOW, **match):
    """% of images answered wrongly with confidence at or above `threshold`."""
    rows = [r for r in results if all(r[k] == v for k, v in match.items())]
    bad = sum(r["pred"] != r["class_id"] and r["conf"] >= threshold for r in rows)
    return percent(bad, len(rows))


def not_sure_rate(results: list, threshold: float, **match):
    """% of images the app would answer "not sure" (confidence below `threshold`)."""
    rows = [r for r in results if all(r[k] == v for k, v in match.items())]
    return percent(sum(r["conf"] < threshold for r in rows), len(rows))


def answered_accuracy(results: list, threshold: float, **match):
    """Accuracy over the images the app answers (confidence at or above `threshold`)."""
    rows = [r for r in results if all(r[k] == v for k, v in match.items())
            and r["conf"] >= threshold]
    return percent(sum(r["pred"] == r["class_id"] for r in rows), len(rows))


def choose_threshold(val: list):
    """The lowest threshold in THRESHOLDS that keeps wrong-and-confident answers on
    the val split under TARGET_CONFIDENT_WRONG %. Returns (threshold, met target);
    if none does, the highest threshold and False."""
    for threshold in THRESHOLDS:
        wrong = confident_wrong(val, threshold)
        if wrong is not None and wrong < TARGET_CONFIDENT_WRONG:
            return threshold, True
    return THRESHOLDS[-1], False


def mean_lighting_accuracy(results: list, **match):
    """Accuracy averaged over the lighting conditions present (each counts equally)."""
    scores = [accuracy(results, lighting=lt, **match)[0] for lt in LIGHTING]
    scores = [s for s in scores if s is not None]
    return sum(scores) / len(scores) if scores else None


def read_baseline(path: Path) -> dict:
    """Field accuracy per lighting from the "Test accuracy by lighting" table of a
    report written by this script, e.g. reports/baseline.md: {lighting: %}."""
    found = {}
    if not path or not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0] in LIGHTING and cells[0] not in found:
            match = re.fullmatch(r"(-?[\d.]+)%", cells[1])
            if match:
                found[cells[0]] = float(match.group(1))
    return found


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


def predict(model, items: list, class_ids: list, mode: str = "none") -> list:
    """items: dicts with path, class_id, sha1 and the lightings to test. Each photo
    gets the lighting effect, then the lighting fix `mode` (preprocess.py).
    Returns one result per (item, lighting) with pred and conf."""
    results = []
    for start in range(0, len(items), EVAL_BATCH):
        chunk = items[start:start + EVAL_BATCH]
        batch, meta = [], []
        for item in chunk:
            image = train.load_image(str(item["path"])).numpy()
            for lighting in item["lightings"]:
                batch.append(preprocess.apply(apply_lighting(image, lighting, item["sha1"]), mode))
                meta.append((item, lighting))
        probs = model.predict_on_batch(np.stack(batch))
        for (item, lighting), p in zip(meta, np.asarray(probs)):
            top = int(np.argmax(p))
            results.append({**item["tags"], "class_id": item["class_id"], "lighting": lighting,
                            "pred": class_ids[top], "conf": float(p[top])})
        print(f"  {min(start + EVAL_BATCH, len(items))}/{len(items)} images", end="\r")
    print()
    return results


def training_note(info: dict) -> str:
    if info.get("augment"):
        note = "Random lighting augmentation in training (`ml/src/augment.py`)"
    else:
        note = "No lighting augmentation in training"
    mode = info.get("preprocess", "none")
    if mode != "none":
        note += f"; every photo goes through `{mode}` first (`ml/src/preprocess.py`)"
    return note + "."


def render_report(info: dict, class_ids: list, val: list, test: list, real: list,
                  title: str, threshold: float = None, threshold_met: bool = True,
                  baseline: dict = None) -> str:
    out = [f"# {title}", ""]
    out += [f"Written by `ml/src/evaluate.py` on {date.today().isoformat()}."
            f" Model: {info.get('model', '?')}, {info.get('image_size', '?')}x"
            f"{info.get('image_size', '?')}, trained by `ml/src/train.py` on"
            f" {info.get('train_images', '?')} images ({info.get('epochs_head', '?')}"
            f" epochs last layer, {info.get('epochs_fine_run', '?')} fine-tuning,"
            f" {info.get('minutes', '?')} minutes; best epoch {info.get('best_epoch', '?')})."
            f" {training_note(info)}", ""]
    if info.get("limit_per_class"):
        out += [f"**Quick check only:** at most {info['limit_per_class']} images per class.", ""]

    val_acc, val_n = accuracy(val, lighting="good")
    out += ["## Summary", ""]
    good_field = accuracy(test, lighting="good", kind="field")[0]
    out += [f"- Validation accuracy (good light, {val_n} images): {fmt(val_acc)}"]
    if any(r["lighting"] != "good" for r in val):
        out.append("- Validation accuracy, averaged over the 4 lighting conditions:"
                   f" {fmt(mean_lighting_accuracy(val))} (picks the best Phase 4 model)")
    out += [f"- Test accuracy, field photos, good light: {fmt(good_field)}"]
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

    if baseline:
        out += ["## Compared with the Phase 3 baseline", "",
                "Field test photos, accuracy per lighting condition (baseline from"
                " `ml/reports/baseline.md`).", "",
                "| Lighting | Baseline | This model | Change | Drop vs good, baseline"
                " | Drop vs good, this model |", "|---|---|---|---|---|---|"]
        for lighting in LIGHTING:
            old = baseline.get(lighting)
            new = accuracy(test, lighting=lighting, kind="field")[0]
            change = new - old if new is not None and old is not None else None
            old_drop = old - baseline["good"] if old is not None and "good" in baseline else None
            new_drop = new - good_field if new is not None and good_field is not None else None
            dash = lighting == "good"
            out.append(f"| {lighting} | {fmt(old)} | {fmt(new)} | {fmt(change, signed=True)}"
                       f" | {'—' if dash else fmt(old_drop, signed=True)}"
                       f" | {'—' if dash else fmt(new_drop, signed=True)} |")
        out.append("")

    if threshold is not None:
        out += ["## \"Not sure\" threshold", "",
                f"Chosen from the val split only, all four lighting conditions ({len(val)}"
                " checks): the lowest confidence at which wrong answers given at or above it"
                f" stay under {TARGET_CONFIDENT_WRONG}% of val images.", ""]
        if threshold_met:
            out.append(f"**Threshold: {threshold:.2f}.** Below it the app says \"not sure\".")
        else:
            out.append(f"**No threshold up to {threshold:.2f} meets the target on val;"
                       f" {threshold:.2f} is used.** The model needs to improve first.")
        out += ["", "| Split | Lighting | Not sure | Accuracy when answered"
                " | Wrong and confident |", "|---|---|---|---|---|"]
        rows = [("val", val, lt, {}) for lt in LIGHTING]
        rows += [("test, field", test, lt, {"kind": "field"}) for lt in LIGHTING]
        for name, results, lighting, extra in rows:
            out.append(f"| {name} | {lighting}"
                       f" | {fmt(not_sure_rate(results, threshold, lighting=lighting, **extra))}"
                       f" | {fmt(answered_accuracy(results, threshold, lighting=lighting, **extra))}"
                       f" | {fmt(confident_wrong(results, threshold, lighting=lighting, **extra))} |")
        out += ["", f"Targets: \"not sure\" under {TARGET_NOT_SURE}% in field testing, wrong and"
                f" confident under {TARGET_CONFIDENT_WRONG}%.", ""]

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




def find_model(model_dir: Path) -> Path:
    """model.keras from train.py, or baseline.keras from the Phase 3 run."""
    for name in ("model.keras", "baseline.keras"):
        if (model_dir / name).is_file():
            return model_dir / name
    sys.exit(f"No model.keras in {model_dir}: run python ml/src/train.py first")


def summary(name: str, title: str, model_dir: Path, info: dict, val: list, test: list,
            threshold: float, threshold_met: bool) -> dict:
    """The main numbers of one run, saved next to its report for --compare."""
    return {
        "name": name,
        "title": title,
        "models": str(model_dir),
        "augment": bool(info.get("augment")),
        "preprocess": info.get("preprocess", "none"),
        "val_good": accuracy(val, lighting="good")[0],
        "val_mean_lighting": mean_lighting_accuracy(val),
        "test_field": {lt: accuracy(test, lighting=lt, kind="field")[0] for lt in LIGHTING},
        "threshold": threshold,
        "threshold_met": threshold_met,
        "test_field_not_sure": {lt: not_sure_rate(test, threshold, lighting=lt, kind="field")
                                for lt in LIGHTING},
        "test_field_confident_wrong": {
            lt: confident_wrong(test, threshold, lighting=lt, kind="field") for lt in LIGHTING},
    }


def evaluate(data_dir: Path, model_dir: Path, report: Path, limit: int = 0,
             title: str = "Phase 3 baseline", baseline_path: Path = None) -> str:
    import keras
    model_path = find_model(model_dir)
    class_ids = (model_dir / "labels.txt").read_text(encoding="utf-8").split()
    info_path = model_dir / "train_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    mode = info.get("preprocess", "none")
    model = keras.models.load_model(model_path)

    rows = train.read_manifest(data_dir)

    def items(split, lightings):
        return [{"path": data_dir / r["file"], "class_id": r["class_id"], "sha1": r["sha1"],
                 "lightings": lightings, "tags": {"kind": r["kind"]}}
                for r in train.split_rows(rows, split, class_ids, limit)]

    print(f"Validation split, {len(LIGHTING)} lighting conditions, preprocess {mode}")
    val = predict(model, items("val", LIGHTING), class_ids, mode)
    print(f"Test split, {len(LIGHTING)} lighting conditions")
    test = predict(model, items("test", LIGHTING), class_ids, mode)
    real_items = [{"path": r["path"], "class_id": r["class_id"], "sha1": r["sha1"],
                   "lightings": ("good",), "tags": {"kind": "field", "tag": r["lighting"]}}
                  for r in field_test_rows(data_dir / "field_test", class_ids)]
    real = []
    if real_items:
        print(f"data/field_test, {len(real_items)} photos")
        real = predict(model, real_items, class_ids, mode)
        # The photo is already in its real lighting: report it under its folder tag.
        for r in real:
            r["lighting"] = r.pop("tag")

    threshold, met = choose_threshold(val)
    baseline = read_baseline(baseline_path) if baseline_path else {}
    if baseline_path and not baseline:
        print(f"No baseline numbers found in {baseline_path}; comparison left out")
    text = render_report(info, class_ids, val, test, real, title, threshold, met, baseline)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    numbers = summary(report.stem, title, model_dir, info, val, test, threshold, met)
    report.with_suffix(".json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    print(f"Wrote {report} and {report.with_suffix('.json').name};"
          f" not-sure threshold {threshold:.2f}")
    return text


def render_comparison(runs: list, baseline: dict) -> tuple:
    """Markdown table of several runs (summary() dicts) and the best run's name,
    picked by val accuracy averaged over the lighting conditions."""
    ranked = [r for r in runs if r.get("val_mean_lighting") is not None]
    best = max(ranked, key=lambda r: r["val_mean_lighting"])["name"] if ranked else None
    out = ["# Phase 4 comparison", "",
           f"Written by `ml/src/evaluate.py --compare` on {date.today().isoformat()}."
           " The best run is picked by **val** accuracy averaged over the four lighting"
           " conditions; the test columns are only reported, never used to choose.", "",
           "Test columns are field photos with synthetic lighting effects. The training"
           " augmentation is similar in kind, so these gains are likely larger than on real"
           " shadowed photos.", "",
           "| Run | Augment | Preprocess | Val, 4-lighting average | "
           + " | ".join(f"Test {lt}" for lt in LIGHTING) + " | Not-sure threshold |",
           "|---" * (5 + len(LIGHTING)) + "|"]
    if baseline:
        out.append("| Phase 3 baseline | no | none | — | "
                   + " | ".join(fmt(baseline.get(lt)) for lt in LIGHTING) + " | — |")
    for r in runs:
        mark = " **(best)**" if r["name"] == best else ""
        threshold = f"{r['threshold']:.2f}" + ("" if r.get("threshold_met", True) else " (target missed)")
        out.append(f"| {r['name']}{mark} | {'yes' if r['augment'] else 'no'} | {r['preprocess']}"
                   f" | {fmt(r['val_mean_lighting'])} | "
                   + " | ".join(fmt(r["test_field"].get(lt)) for lt in LIGHTING)
                   + f" | {threshold} |")
    out.append("")
    if baseline and best:
        top = next(r for r in runs if r["name"] == best)
        out += [f"## Best run ({best}) against the baseline", "",
                "| Lighting | Baseline | Best run | Change |", "|---|---|---|---|"]
        for lt in LIGHTING:
            old, new = baseline.get(lt), top["test_field"].get(lt)
            change = new - old if old is not None and new is not None else None
            out.append(f"| {lt} | {fmt(old)} | {fmt(new)} | {fmt(change, signed=True)} |")
        out.append("")
    return "\n".join(out), best


def compare(paths: list, baseline_path: Path, report: Path) -> str:
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    runs = [r for r in runs if "val_mean_lighting" in r]
    if not runs:
        sys.exit("No run summaries to compare: run evaluate.py on each model first")
    text, best = render_comparison(runs, read_baseline(baseline_path) if baseline_path else {})
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    top = next((r for r in runs if r["name"] == best), {})
    report.with_suffix(".json").write_text(
        json.dumps({"best": best, "models": top.get("models"), "threshold": top.get("threshold")},
                   indent=2), encoding="utf-8")
    print(f"Wrote {report}; best run by val: {best}")
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=ML_DIR / "data",
                        help="folder holding processed/ and field_test/")
    parser.add_argument("--models", type=Path, default=ML_DIR / "models")
    parser.add_argument("--report", type=Path, default=None,
                        help="default reports/baseline.md, or reports/phase4_comparison.md"
                             " with --compare")
    parser.add_argument("--title", default="Phase 3 baseline")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="report to compare with, e.g. ml/reports/baseline.md")
    parser.add_argument("--compare", type=Path, nargs="+", default=None,
                        help="run summaries (.json next to each report) to put side by side")
    parser.add_argument("--limit", type=int, default=0,
                        help="use at most this many images per class (quick check)")
    args = parser.parse_args(argv)
    reports = ML_DIR / "reports"
    if args.compare:
        compare(args.compare, args.baseline, args.report or reports / "phase4_comparison.md")
    else:
        evaluate(args.data, args.models, args.report or reports / "baseline.md", args.limit,
                 args.title, args.baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
