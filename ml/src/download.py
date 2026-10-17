"""Download the public datasets into data/raw/<source>/ for ml/src/dataset.py.

Phase 2, see docs/SYSTEM_DESIGN.md and docs/datasets.md (links and licences).

    plantvillage     GitHub mirror, only the raw/color folders sources.yaml uses
    plantdoc         GitHub archive (.zip, read in place by dataset.py)
    cold_onion       Hugging Face parquet, raw split only (not the augmented one)
    ngld_grape       Mendeley 8nnd2ypcv3 v5
    mendeley_tomato  Mendeley jttrv2w27r v2
    tom2024          Mendeley 3d4yg89rtr v1, the 3 onion_diseases folders sources.yaml lists

A finished source gets an empty DONE_MARKER file and is skipped next time, so
a stopped run can be started again. Mendeley files already on disk are not
downloaded twice; other sources start over.

Usage:
    python ml/src/download.py                       # into ml/data/raw
    python ml/src/download.py --data /content/data  # into /content/data/raw
    python ml/src/download.py --only ngld_grape tom2024
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ML_DIR = Path(__file__).resolve().parents[1]

PLANTVILLAGE_GIT = "https://github.com/spMohanty/PlantVillage-Dataset.git"
PLANTDOC_ZIP = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
COLD_PARQUET = ("https://huggingface.co/datasets/Project-AgML/"
                "COLD_onion_leaf_disease_classification/resolve/main/raw/train-00000-of-00001.parquet")
MENDELEY_API = "https://data.mendeley.com/public-api/datasets"
MENDELEY_FILE = "https://data.mendeley.com/public-files/datasets/{dataset}/files/{file}/file_downloaded"
TOM2024_DIR = "CATEGORY A/CATA-English/onion_diseases"
# source: (Mendeley dataset id, version, folder paths to keep, empty for all)
MENDELEY = {
    "ngld_grape": ("8nnd2ypcv3", 5, ()),
    "mendeley_tomato": ("jttrv2w27r", 2, ()),
    "tom2024": ("3d4yg89rtr", 1, tuple(f"{TOM2024_DIR}/{name}"
                                       for name in ("Alternaria_D", "Healthy_leaf", "Virosis-D"))),
}
WORKERS = 8         # parallel Mendeley downloads
RETRIES = 4
TIMEOUT = 120       # seconds per request
DONE_MARKER = ".done"
USER_AGENT = "Mozilla/5.0 (KrishiCare dataset download)"  # Mendeley returns 403 to Python's default


def fetch(url: str) -> bytes:
    for attempt in range(RETRIES):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as r:
                return r.read()
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


def save(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_bytes(fetch(url))
    tmp.rename(out)


def plantvillage(out: Path, sources_path: Path) -> None:
    """Sparse clone of only the colour folders mapped to a class."""
    mapping = yaml.safe_load(sources_path.read_text(encoding="utf-8"))["sources"]["plantvillage"]["map"]
    folders = [f"raw/color/{name}" for name, target in mapping.items() if target != "skip"]
    run = lambda *args: subprocess.run(["git", *args], check=True)
    run("clone", "--depth", "1", "--filter=blob:none", "--sparse", PLANTVILLAGE_GIT, str(out))
    run("-C", str(out), "sparse-checkout", "set", *folders)
    shutil.rmtree(out / ".git")


def plantdoc(out: Path, _) -> None:
    save(PLANTDOC_ZIP, out / "PlantDoc-Dataset-master.zip")


def cold_onion(out: Path, _) -> None:
    save(COLD_PARQUET, out / "train-00000-of-00001.parquet")


def mendeley(source: str):
    dataset, version, keep = MENDELEY[source]

    def download(out: Path, _) -> None:
        folders = json.loads(fetch(f"{MENDELEY_API}/{dataset}/folders/{version}"))
        by_id = {f["id"]: f for f in folders}

        def path(folder_id: str) -> str:
            parts = []
            while folder_id in by_id:
                parts.append(by_id[folder_id]["name"])
                folder_id = by_id[folder_id].get("parent_id")
            return "/".join(reversed(parts))

        jobs = []
        for folder in folders:
            folder_path = path(folder["id"])
            if keep and not any((folder_path + "/").startswith(k + "/") for k in keep):
                continue
            files = json.loads(fetch(f"{MENDELEY_API}/{dataset}/files"
                                     f"?folder_id={folder['id']}&version={version}"))
            for f in files:
                target = out / folder_path / f["filename"]
                if not target.exists():
                    jobs.append((MENDELEY_FILE.format(dataset=dataset, file=f["id"]), target))
        print(f"  {len(jobs)} files to download")
        with ThreadPoolExecutor(WORKERS) as pool:
            for i, _ in enumerate(pool.map(lambda job: save(*job), jobs), 1):
                if i % 500 == 0:
                    print(f"  {i}/{len(jobs)}")

    return download


DOWNLOADERS = {
    "plantvillage": plantvillage,
    "plantdoc": plantdoc,
    "cold_onion": cold_onion,
    **{name: mendeley(name) for name in MENDELEY},
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=ML_DIR / "data",
                        help="folder holding raw/ (created if missing)")
    parser.add_argument("--sources", type=Path, default=ML_DIR / "sources.yaml")
    parser.add_argument("--only", nargs="+", choices=list(DOWNLOADERS), metavar="SOURCE",
                        help=f"download only these: {', '.join(DOWNLOADERS)}")
    args = parser.parse_args(argv)

    failed = []
    for name in args.only or DOWNLOADERS:
        out = args.data / "raw" / name
        resumable = name in MENDELEY
        if (out / DONE_MARKER).exists():
            print(f"{name}: already in {out}, skipped")
            continue
        if not resumable:
            shutil.rmtree(out, ignore_errors=True)   # a stopped earlier run
        print(f"{name}: downloading into {out}")
        try:
            DOWNLOADERS[name](out, args.sources)
        except Exception as e:
            print(f"{name}: FAILED ({e}); run again to retry")
            failed.append(name)
            continue
        count = sum(1 for p in out.rglob("*") if p.is_file() and p.name != DONE_MARKER)
        (out / DONE_MARKER).touch()
        print(f"{name}: done, {count} files")
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
