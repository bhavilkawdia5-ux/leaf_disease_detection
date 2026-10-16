"""Merge raw leaf datasets into data/processed with the KrishiCare class names.

Phase 2, see docs/SYSTEM_DESIGN.md.

    data/raw/<source>/...                 downloads and own photos (never changed)
    data/field_test/<class id>/<lighting>/...  held-out field photos (never
                                          changed, never trained on); lighting is
                                          good, shadow, harsh_sun or low_light
    data/processed/{train,val}/<class id>/  rebuilt by this script

Each raw image gets its class from ml/sources.yaml. Folders, .zip files and
Hugging Face .parquet files are all read in place, so raw data can stay zipped.
Exact duplicates are dropped, and any raw image that is also in field_test is
left out. The 85/15 train/val split comes from each image's hash, so an image
keeps its split when more data is added later.

Usage:
    python ml/src/dataset.py                    # data in ml/data
    python ml/src/dataset.py --data /content/drive/MyDrive/leafsight/data
    python ml/src/dataset.py --dry-run          # count only, write nothing
"""
import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

import yaml
from PIL import Image

ML_DIR = Path(__file__).resolve().parents[1]

SPLITS = ("train", "val")
VAL_PERCENT = 15        # SYSTEM_DESIGN Phase 2: split train/val 85/15
MIN_PER_CLASS = 300     # SYSTEM_DESIGN Phase 2: aim for 300+ images per class
MIN_FIELD_TEST = 50     # SYSTEM_DESIGN Phase 2: field test of 50+ per class
MAX_IMBALANCE = 3.0     # SYSTEM_DESIGN Phase 2: counts balanced within roughly 3x
LIGHTING = ("good", "shadow", "harsh_sun", "low_light")  # SYSTEM_DESIGN Phase 3 tags
UNTAGGED = "untagged"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FORMAT_EXTS = {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp", "WEBP": ".webp"}
SKIP = "skip"


def norm(name) -> str:
    """Folder-name key: case, spaces and underscores ignored."""
    return re.sub(r"[\s_]+", " ", str(name)).strip().lower()


@dataclass
class RawImage:
    origin: str                 # where it came from, for reports and the manifest
    dirs: tuple                 # folder names from the source root down to the image
    read: Callable[[], bytes]


@dataclass
class Result:
    class_ids: list
    rows: list = field(default_factory=list)            # manifest rows
    field_test: Counter = field(default_factory=Counter)
    field_test_unknown: Counter = field(default_factory=Counter)
    field_lighting: Counter = field(default_factory=Counter)  # lighting folder -> images
    unmapped: Counter = field(default_factory=Counter)  # (source, folder) -> images
    missing_sources: list = field(default_factory=list)
    unknown_sources: list = field(default_factory=list)
    unreadable: list = field(default_factory=list)
    duplicates: int = 0
    field_test_overlap: int = 0
    conflicts: int = 0
    errors: list = field(default_factory=list)


def load_class_ids(path: Path) -> list:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    ids = [c["id"] for crop in data["crops"].values() for c in crop["classes"]]
    return ids + [c["id"] for c in data.get("confusers") or []]


def load_sources(path: Path, class_ids: list) -> tuple:
    """Return ({source: config with a normalised map}, errors)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = set(class_ids)
    sources, errors = {}, []
    for name, cfg in (data.get("sources") or {}).items():
        cfg = cfg or {}
        raw_map = cfg.get("map") or {}
        if raw_map == "auto":
            raw_map = {cid: cid for cid in class_ids}
        mapping = {}
        for folder, target in raw_map.items():
            if target != SKIP and target not in known:
                errors.append(f"{name}: folder '{folder}' maps to unknown class '{target}'")
            key = norm(folder)
            if key in mapping and mapping[key] != target:
                errors.append(f"{name}: folder '{folder}' is listed twice with different classes")
            mapping[key] = target
        sources[name] = {
            "map": mapping,
            "require_dir": norm(cfg["require_dir"]) if cfg.get("require_dir") else None,
            "exclude": [norm(x) for x in cfg.get("exclude") or []],
        }
    return sources, errors


def image_ext(data: bytes):
    """File extension for a readable image, or None if it is broken."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
    except Exception:
        return None
    return FORMAT_EXTS.get(fmt)


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def split_for(digest: str) -> str:
    return "val" if int(digest[:8], 16) % 100 < VAL_PERCENT else "train"


def iter_images(source: str, root: Path):
    """Yield every candidate image under root, looking inside .zip and .parquet."""
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            yield RawImage(f"{source}/{rel.as_posix()}", rel.parts[:-1], path.read_bytes)
        elif suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                for member in sorted(zf.namelist()):
                    inner = PurePosixPath(member)
                    if member.endswith("/") or inner.suffix.lower() not in IMAGE_EXTS:
                        continue
                    yield RawImage(
                        f"{source}/{rel.as_posix()}!{member}",
                        rel.parts[:-1] + (path.stem,) + inner.parts[:-1],
                        lambda zf=zf, m=member: zf.read(m),
                    )
        elif suffix == ".parquet":
            yield from iter_parquet(source, root, path)


def iter_parquet(source: str, root: Path, path: Path):
    """Hugging Face image datasets: an `image` column ({bytes, path}) and a
    `label` column whose names are stored in the file's metadata."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit(f"{path}: reading .parquet needs pyarrow (pip install pyarrow)")
    pf = pq.ParquetFile(path)
    names = []
    meta = (pf.schema_arrow.metadata or {}).get(b"huggingface")
    if meta:
        label = json.loads(meta).get("info", {}).get("features", {}).get("label", {})
        names = label.get("names") or []
    rel = path.relative_to(root)
    row = 0
    for batch in pf.iter_batches(columns=["image", "label"]):
        for image, label in zip(batch.column("image").to_pylist(),
                                batch.column("label").to_pylist()):
            label_name = names[label] if isinstance(label, int) and label < len(names) else label
            data = (image or {}).get("bytes") or b""
            yield RawImage(
                f"{source}/{rel.as_posix()}#{row}",
                rel.parts[:-1] + (str(label_name),),
                lambda d=data: d,
            )
            row += 1


def classify(dirs: tuple, mapping: dict):
    """Class of the nearest listed folder above the image, and that folder."""
    for folder in reversed(dirs):
        if norm(folder) in mapping:
            return mapping[norm(folder)], folder
    return None, dirs[-1] if dirs else "(top level)"


def scan_field_test(field_dir: Path, class_ids: list, result: Result) -> set:
    """Count field test images per class; return their hashes. Read only."""
    hashes = set()
    if not field_dir.is_dir():
        return hashes
    for path in sorted(p for p in field_dir.rglob("*") if p.is_file()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        hashes.add(sha1(path.read_bytes()))
        parts = path.relative_to(field_dir).parts
        class_id = parts[0] if len(parts) > 1 else "(top level)"
        if class_id in class_ids:
            result.field_test[class_id] += 1
            result.field_lighting[parts[1] if len(parts) > 2 else UNTAGGED] += 1
        else:
            result.field_test_unknown[class_id] += 1
    return hashes


def build(data_dir: Path, classes_path: Path, sources_path: Path, dry_run=False) -> Result:
    class_ids = load_class_ids(classes_path)
    sources, errors = load_sources(sources_path, class_ids)
    result = Result(class_ids, errors=errors)
    if errors:
        return result

    raw_dir, processed = data_dir / "raw", data_dir / "processed"
    if raw_dir.is_dir():
        result.unknown_sources = sorted(
            p.name for p in raw_dir.iterdir() if p.is_dir() and p.name not in sources)
    field_hashes = scan_field_test(data_dir / "field_test", class_ids, result)

    if not dry_run:
        for split in SPLITS:
            shutil.rmtree(processed / split, ignore_errors=True)

    kept = {}           # hash -> manifest row
    conflicted = set()
    for name, cfg in sources.items():
        root = raw_dir / name
        if not root.is_dir():
            result.missing_sources.append(name)
            continue
        for img in iter_images(name, root):
            folders = {norm(d) for d in img.dirs}
            if cfg["require_dir"] and cfg["require_dir"] not in folders:
                continue
            if any(word in norm(img.origin) for word in cfg["exclude"]):
                continue
            class_id, folder = classify(img.dirs, cfg["map"])
            if class_id is None:
                result.unmapped[(name, folder)] += 1
                continue
            if class_id == SKIP:
                continue

            data = img.read()
            ext = image_ext(data)
            if ext is None:
                result.unreadable.append(img.origin)
                continue
            digest = sha1(data)
            if digest in field_hashes:
                result.field_test_overlap += 1
                continue
            if digest in conflicted:
                result.conflicts += 1
                continue
            if digest in kept:
                if kept[digest]["class_id"] == class_id:
                    result.duplicates += 1
                else:
                    # Same photo under two labels: trust neither.
                    old = kept.pop(digest)
                    conflicted.add(digest)
                    result.conflicts += 2
                    if not dry_run:
                        (data_dir / old["file"]).unlink(missing_ok=True)
                continue

            split = split_for(digest)
            out = processed / split / class_id / f"{name}__{digest[:16]}{ext}"
            if not dry_run:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
            kept[digest] = {
                "split": split, "class_id": class_id,
                "file": out.relative_to(data_dir).as_posix(),
                "source": name, "origin": img.origin, "sha1": digest,
            }

    result.rows = list(kept.values())
    if not dry_run and result.rows:
        processed.mkdir(parents=True, exist_ok=True)
        with open(processed / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.rows[0]))
            writer.writeheader()
            writer.writerows(sorted(result.rows, key=lambda r: r["file"]))
    return result


def print_report(result: Result) -> None:
    if result.errors:
        print("Errors in ml/sources.yaml (nothing was written):")
        for e in result.errors:
            print(f"  - {e}")
        return

    counts = defaultdict(Counter)
    by_source = defaultdict(Counter)
    for row in result.rows:
        counts[row["class_id"]][row["split"]] += 1
        by_source[row["class_id"]][row["source"]] += 1

    width = max(len(c) for c in result.class_ids)
    print(f"{'class':<{width}}  {'train':>6}  {'val':>5}  {'total':>6}  {'field':>5}  sources")
    for cid in result.class_ids:
        c = counts[cid]
        srcs = ", ".join(f"{s} {n}" for s, n in sorted(by_source[cid].items()))
        print(f"{cid:<{width}}  {c['train']:>6}  {c['val']:>5}  {sum(c.values()):>6}"
              f"  {result.field_test[cid]:>5}  {srcs}")
    train = sum(c["train"] for c in counts.values())
    val = sum(c["val"] for c in counts.values())
    print(f"{'TOTAL':<{width}}  {train:>6}  {val:>5}  {train + val:>6}"
          f"  {sum(result.field_test.values()):>5}")

    notes = []
    totals = {cid: sum(counts[cid].values()) for cid in result.class_ids}
    empty = [cid for cid, n in totals.items() if n == 0]
    if empty:
        notes.append(f"No images yet ({len(empty)}): {', '.join(empty)}")
    low = [f"{cid} ({n})" for cid, n in totals.items() if 0 < n < MIN_PER_CLASS]
    if low:
        notes.append(f"Under {MIN_PER_CLASS} images: {', '.join(low)}")
    filled = [n for n in totals.values() if n]
    if filled and max(filled) > MAX_IMBALANCE * min(filled):
        notes.append(f"Not balanced within {MAX_IMBALANCE:g}x: largest class {max(filled)},"
                     f" smallest non-empty class {min(filled)}")
    short = [f"{cid} ({result.field_test[cid]})" for cid in result.class_ids
             if result.field_test[cid] < MIN_FIELD_TEST]
    if short:
        notes.append(f"Field test under {MIN_FIELD_TEST} photos: {len(short)} classes")
    if result.field_lighting:
        tags = [t for t in LIGHTING if result.field_lighting[t]]
        others = sorted(t for t in result.field_lighting if t not in LIGHTING)
        notes.append("Field test by lighting: " + ", ".join(
            f"{t} {result.field_lighting[t]}" for t in tags + others))
        missing = [t for t in LIGHTING if not result.field_lighting[t]]
        if missing:
            notes.append(f"No field test photos in: {', '.join(missing)}")
        if others:
            notes.append(f"Field test photos need a lighting folder ({', '.join(LIGHTING)}),"
                         f" found: {', '.join(others)}")
    for (source, folder), n in sorted(result.unmapped.items()):
        notes.append(f"Unmapped folder {source}/'{folder}' ({n} images):"
                     f" add it to ml/sources.yaml as a class id or skip")
    for name, n in sorted(result.field_test_unknown.items()):
        notes.append(f"field_test/{name} is not a class id ({n} images, not counted)")
    if result.unknown_sources:
        notes.append(f"Folders in data/raw not in ml/sources.yaml (ignored):"
                     f" {', '.join(result.unknown_sources)}")
    if result.missing_sources:
        notes.append(f"Not downloaded yet: {', '.join(result.missing_sources)}")
    if result.duplicates:
        notes.append(f"Dropped {result.duplicates} exact duplicates")
    if result.field_test_overlap:
        notes.append(f"Left out {result.field_test_overlap} raw images that are in field_test")
    if result.conflicts:
        notes.append(f"Left out {result.conflicts} images found under two different classes")
    if result.unreadable:
        notes.append(f"Skipped {len(result.unreadable)} unreadable images,"
                     f" e.g. {result.unreadable[0]}")
    if notes:
        print("\nNotes:")
        for note in notes:
            print(f"  - {note}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=ML_DIR / "data",
                        help="folder holding raw/, field_test/ and processed/")
    parser.add_argument("--classes", type=Path, default=ML_DIR / "classes.yaml")
    parser.add_argument("--sources", type=Path, default=ML_DIR / "sources.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="count and check only, do not write data/processed")
    args = parser.parse_args(argv)

    result = build(args.data, args.classes, args.sources, args.dry_run)
    print_report(result)
    if result.rows and not args.dry_run:
        print(f"\nWrote {len(result.rows)} images to {args.data / 'processed'}")
    return 1 if result.errors else 0


if __name__ == "__main__":
    sys.exit(main())
