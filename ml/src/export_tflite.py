"""Export a trained model to TensorFlow Lite (float16) and check size, answers and speed.

Phase 4, see docs/SYSTEM_DESIGN.md ("Export the best model to TensorFlow Lite and
check its size and speed").

    <models>/model.keras, labels.txt, train_info.json   from train.py
    <models>/model.tflite           float16 weights, input 1 x 224 x 224 x 3 RGB 0-255
    reports/export.md               the checks (--report to change)

Checks:
- size under MAX_MB (SYSTEM_DESIGN metrics: under 20 MB)
- the .tflite gives the same top answer as the Keras model on val photos
  (good light, after the model's lighting fix), so float16 lost nothing
- time per image on this computer's CPU, 1 and 4 threads. This is NOT the
  budget-phone number SYSTEM_DESIGN asks for (under 1 second); measure that on
  a phone or the emulator before Phase 4 is done.

The lighting fix (train_info.json "preprocess") is not inside the .tflite: the
app must run it before the model (LightingCorrector.kt, Phase 6).

Usage:
    python ml/src/export_tflite.py --models ml/models/aug_clahe
    python ml/src/export_tflite.py --models /content/models/aug_clahe --data /content/data
"""
import argparse
import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR / "src"))
import evaluate  # noqa: E402
import preprocess  # noqa: E402
import train  # noqa: E402

MAX_MB = 20             # SYSTEM_DESIGN metrics: model under 20 MB
MAX_SECONDS = 1.0       # SYSTEM_DESIGN metrics: diagnosis under 1 second (budget phone)
MIN_AGREEMENT = 99.0    # % of check photos where .tflite and Keras give the same answer
CHECK_PER_CLASS = 15    # val photos per class for the agreement check
TIMING_RUNS = 50        # timed runs after WARMUP_RUNS
WARMUP_RUNS = 5


def inference_model(model):
    """The same model without train.py's "augment" layer (random flip, rotation,
    zoom). It does nothing at inference, but its random ops are not in TFLite."""
    import keras
    inputs = keras.Input(shape=model.input_shape[1:], name="image")
    x = inputs
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer) or layer.name == "augment":
            continue
        x = layer(x, training=False)
    return keras.Model(inputs, x)


def convert(model_path: Path, out_path: Path) -> int:
    """Keras -> SavedModel -> .tflite with float16 weights. Returns the file size."""
    import keras
    import tensorflow as tf
    model = inference_model(keras.models.load_model(model_path))
    with tempfile.TemporaryDirectory() as tmp:
        model.export(tmp, format="tf_saved_model", verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        out_path.write_bytes(converter.convert())
    return out_path.stat().st_size


def interpreter(path: Path, threads: int):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(path), num_threads=threads)
    it.allocate_tensors()
    return it


def run(it, image: np.ndarray) -> np.ndarray:
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(inp["index"], image[None].astype(inp["dtype"]))
    it.invoke()
    return it.get_tensor(out["index"])[0]


def seconds_per_image(it, image: np.ndarray) -> float:
    for _ in range(WARMUP_RUNS):
        run(it, image)
    started = time.perf_counter()
    for _ in range(TIMING_RUNS):
        run(it, image)
    return (time.perf_counter() - started) / TIMING_RUNS


def render_export_report(name: str, info: dict, size_bytes: int, agree: int, checked: int,
                         timing: dict) -> str:
    mb = size_bytes / 1e6
    agreement = 100.0 * agree / checked if checked else None
    ok = lambda passed: "yes" if passed else "**no**"  # noqa: E731
    out = [f"# TensorFlow Lite export: {name}", "",
           f"Written by `ml/src/export_tflite.py` on {date.today().isoformat()}."
           f" Float16 weights, input 1 x {info.get('image_size', 224)} x"
           f" {info.get('image_size', 224)} x 3 RGB, 0-255; output {len(info.get('classes', []))}"
           " class probabilities in `labels.txt` order.", "",
           f"Lighting fix before the model: `{info.get('preprocess', 'none')}`. It is not inside"
           " the .tflite, so the app must run it first (Phase 6).", "",
           "| Check | Result | Target | Passed |", "|---|---|---|---|",
           f"| Size | {mb:.1f} MB | under {MAX_MB} MB | {ok(mb < MAX_MB)} |",
           f"| Same answer as Keras | {evaluate.fmt(agreement)} of {checked} val photos"
           f" | {MIN_AGREEMENT:g}% or more | {ok(agreement is not None and agreement >= MIN_AGREEMENT)} |"]
    for threads, secs in timing.items():
        out.append(f"| Time per image, this computer, {threads} thread{'s' if threads > 1 else ''}"
                   f" | {secs * 1000:.0f} ms | — | — |")
    out += ["", f"**Still to do:** time it on a budget Android phone or the emulator; the"
            f" target is under {MAX_SECONDS:g} second per image. A computer CPU is usually much"
            " faster than a budget phone, so the numbers above only show it is not far off.", ""]
    return "\n".join(out)


def export(model_dir: Path, data_dir: Path, report: Path) -> str:
    model_path = evaluate.find_model(model_dir)
    info_path = model_dir / "train_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    class_ids = (model_dir / "labels.txt").read_text(encoding="utf-8").split()
    mode = info.get("preprocess", "none")
    out_path = model_dir / "model.tflite"

    print(f"Converting {model_path}")
    size = convert(model_path, out_path)
    print(f"Wrote {out_path}, {size / 1e6:.1f} MB")

    import keras
    model = keras.models.load_model(model_path)
    rows = train.split_rows(train.read_manifest(data_dir), "val", class_ids, CHECK_PER_CLASS)
    images = [preprocess.apply(train.load_image(str(data_dir / r["file"])).numpy(), mode)
              for r in rows]
    if not images:
        sys.exit("No val photos found for the agreement check: run dataset.py first")
    keras_top = np.argmax(model.predict(np.stack(images), verbose=0), axis=1)
    it = interpreter(out_path, threads=1)
    lite_top = [int(np.argmax(run(it, image))) for image in images]
    agree = int(sum(int(a) == b for a, b in zip(keras_top, lite_top)))
    print(f"Same answer on {agree} of {len(images)} val photos")

    timing = {threads: seconds_per_image(interpreter(out_path, threads), images[0])
              for threads in (1, 4)}
    text = render_export_report(model_dir.name, info, size, agree, len(images), timing)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    print(f"Wrote {report}")
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", type=Path, required=True,
                        help="folder from train.py holding model.keras")
    parser.add_argument("--data", type=Path, default=ML_DIR / "data",
                        help="folder holding processed/ (val photos for the checks)")
    parser.add_argument("--report", type=Path, default=ML_DIR / "reports" / "export.md")
    args = parser.parse_args(argv)
    export(args.models, args.data, args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
