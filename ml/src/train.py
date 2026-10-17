"""Fine-tune MobileNetV3-Large on data/processed (Phase 3 baseline, Phase 4 robust models).

Phases 3 and 4, see docs/SYSTEM_DESIGN.md.

    data/processed/manifest.csv     which image is in train / val (from dataset.py)
    <out>/model.keras               best model by val accuracy
    <out>/labels.txt                class ids, one per line, in model output order
    <out>/train_info.json           settings and accuracy per epoch, read by evaluate.py

Two stages, as in the Phase 0 practice run: train only the new last layer with
the pretrained model frozen, then unfreeze its top layers with a small learning
rate. Only the val split picks the best epoch; the test split is never read here.

Without --augment there is no lighting augmentation: the Phase 3 baseline had to
show how much shadow and harsh sun hurt. --augment (Phase 4) adds the random
lighting effects in augment.py, and --preprocess runs a lighting fix from
preprocess.py on every photo first. Rare classes get a bigger class weight,
because train counts are 23x apart (docs/datasets.md).

Usage:
    python ml/src/train.py                          # data in ml/data
    python ml/src/train.py --data /content/data     # in Colab
    python ml/src/train.py --augment --preprocess clahe --out ml/models/aug_clahe
    python ml/src/train.py --limit 20 --epochs-head 1 --epochs-fine 1   # quick check
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR / "src"))
import augment  # noqa: E402
import dataset  # noqa: E402
import preprocess  # noqa: E402

IMG = 224               # SYSTEM_DESIGN Phase 3: 224x224
BATCH = 32
SEED = 42
EPOCHS_HEAD = 5         # frozen pretrained model, new last layer only
EPOCHS_FINE = 20        # top layers unfrozen; stops early if val stops improving.
                        # The Phase 3 baseline ran 10 and was still improving at the end.
FINE_TUNE_LAYERS = 60   # last layers of MobileNetV3-Large (188 layers) to unfreeze
LR_HEAD = 1e-3          # Phase 0 practice run
LR_FINE = 5e-5          # small, so fine-tuning nudges the pretrained weights
DROPOUT = 0.2           # Phase 0 practice run
PATIENCE = 4            # epochs without a better val accuracy before stopping (baseline: 3)
SHUFFLE = 2048


def read_manifest(data_dir: Path) -> list:
    path = data_dir / "processed" / "manifest.csv"
    if not path.is_file():
        sys.exit(f"{path} not found: run python ml/src/dataset.py first")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_rows(rows: list, split: str, class_ids: list, limit: int = 0) -> list:
    """Rows of one split for the active classes, at most `limit` per class."""
    out, seen = [], Counter()
    for row in sorted(rows, key=lambda r: r["sha1"]):
        if row["split"] != split or row["class_id"] not in class_ids:
            continue
        if limit and seen[row["class_id"]] >= limit:
            continue
        seen[row["class_id"]] += 1
        out.append(row)
    return out


def class_weights(labels: list, num_classes: int) -> dict:
    """Weight each class by total / (classes x count), so each class adds up to the
    same share of the loss. Classes with no images get weight 0."""
    counts = Counter(labels)
    total = len(labels)
    return {i: total / (num_classes * counts[i]) if counts[i] else 0.0
            for i in range(num_classes)}


def load_image(path):
    """Decode any image file to an IMG x IMG float32 RGB tensor, values 0-255."""
    import tensorflow as tf
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMG, IMG), antialias=True)
    return tf.clip_by_value(image, 0.0, 255.0)


def augment_and_fix(image: np.ndarray, seed: int, mode: str) -> np.ndarray:
    """Random lighting effects, then the lighting fix: what one training photo
    looks like in one epoch."""
    out = augment.random_augment(image, np.random.default_rng(int(seed)))
    return preprocess.apply(out, mode)


def make_dataset(data_dir: Path, rows: list, class_ids: list, training: bool,
                 use_augment: bool = False, mode: str = "none"):
    import tensorflow as tf
    paths = [str(data_dir / r["file"]) for r in rows]
    labels = [class_ids.index(r["class_id"]) for r in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    random_each_epoch = training and use_augment

    def load(path):
        image = load_image(path)
        if mode != "none" and not random_each_epoch:
            image = tf.numpy_function(lambda x: preprocess.apply(x, mode), [image], tf.float32)
            image.set_shape((IMG, IMG, 3))
        return tf.cast(tf.round(image), tf.uint8)

    # Keep the resized images in memory (about 150 KB each) so each image file
    # is decoded only once.
    ds = ds.map(lambda p, y: (load(p), y), num_parallel_calls=tf.data.AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(SHUFFLE, seed=SEED, reshuffle_each_iteration=True)
    if random_each_epoch:
        def random_lighting(x):
            seed = tf.random.uniform([], 0, 2 ** 31 - 1, dtype=tf.int64)
            out = tf.numpy_function(lambda img, s: augment_and_fix(img, s, mode),
                                    [x, seed], tf.float32)
            out.set_shape((IMG, IMG, 3))
            return out
        ds = ds.map(lambda x, y: (random_lighting(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).map(lambda x, y: (tf.cast(x, tf.float32), y))
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(num_classes: int):
    import keras
    # Geometry only; lighting augmentation is Phase 4.
    augment = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ], name="augment")
    # include_preprocessing: the model takes 0-255 pixels, like the app will send.
    base = keras.applications.MobileNetV3Large(
        input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet",
        pooling="avg", include_preprocessing=True)
    base.trainable = False
    inputs = keras.Input(shape=(IMG, IMG, 3), name="image")
    x = augment(inputs)
    x = base(x, training=False)     # BatchNorm stays in inference mode, also when unfrozen
    x = keras.layers.Dropout(DROPOUT)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs), base


def train(data_dir: Path, out_dir: Path, classes_path: Path, epochs_head: int,
          epochs_fine: int, limit: int = 0, use_augment: bool = False,
          mode: str = "none") -> dict:
    import keras
    keras.utils.set_random_seed(SEED)

    class_ids, _ = dataset.load_classes(classes_path)
    rows = read_manifest(data_dir)
    train_rows = split_rows(rows, "train", class_ids, limit)
    val_rows = split_rows(rows, "val", class_ids, limit)
    if not train_rows or not val_rows:
        sys.exit("No train or val images in the manifest: run dataset.py first")
    labels = [class_ids.index(r["class_id"]) for r in train_rows]
    weights = class_weights(labels, len(class_ids))
    print(f"{len(train_rows)} train and {len(val_rows)} val images, {len(class_ids)} classes,"
          f" lighting augmentation {'on' if use_augment else 'off'}, preprocess {mode}")

    train_ds = make_dataset(data_dir, train_rows, class_ids, True, use_augment, mode)
    val_ds = make_dataset(data_dir, val_rows, class_ids, False, use_augment, mode)
    model, base = build_model(len(class_ids))

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.keras"
    (out_dir / "labels.txt").write_text("\n".join(class_ids) + "\n", encoding="utf-8")
    best = keras.callbacks.ModelCheckpoint(
        model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True)

    started = time.time()
    print(f"\nStage 1: last layer only, {epochs_head} epochs")
    model.compile(optimizer=keras.optimizers.Adam(LR_HEAD),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    head = model.fit(train_ds, validation_data=val_ds, epochs=epochs_head,
                     class_weight=weights, callbacks=[best], verbose=2)

    print(f"\nStage 2: fine-tune the last {FINE_TUNE_LAYERS} layers, up to {epochs_fine} epochs")
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(LR_FINE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Carry stage 1's best score over, so a worse stage 2 never overwrites it.
    best.best = max(head.history["val_accuracy"])
    fine = model.fit(train_ds, validation_data=val_ds,
                     epochs=epochs_head + epochs_fine, initial_epoch=epochs_head,
                     class_weight=weights, callbacks=[best, stop], verbose=2)

    history = {k: head.history[k] + fine.history.get(k, []) for k in head.history}
    best_epoch = max(range(len(history["val_accuracy"])),
                     key=lambda i: history["val_accuracy"][i])
    info = {
        "date": date.today().isoformat(),
        "model": "MobileNetV3Large (ImageNet), include_preprocessing",
        "image_size": IMG,
        "augment": use_augment,
        "preprocess": mode,
        "classes": class_ids,
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "limit_per_class": limit,
        "epochs_head": epochs_head,
        "epochs_fine_run": len(fine.history.get("loss", [])),
        "fine_tune_layers": FINE_TUNE_LAYERS,
        "lr_head": LR_HEAD,
        "lr_fine": LR_FINE,
        "batch": BATCH,
        "seed": SEED,
        "class_weights": {class_ids[i]: round(w, 3) for i, w in weights.items()},
        "best_epoch": best_epoch + 1,
        "best_val_accuracy": history["val_accuracy"][best_epoch],
        "minutes": round((time.time() - started) / 60, 1),
        "history": history,
    }
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nBest val accuracy {info['best_val_accuracy']:.1%} at epoch {info['best_epoch']},"
          f" {info['minutes']} minutes. Saved {model_path}")
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=ML_DIR / "data",
                        help="folder holding processed/ (from dataset.py)")
    parser.add_argument("--out", type=Path, default=ML_DIR / "models")
    parser.add_argument("--classes", type=Path, default=ML_DIR / "classes.yaml")
    parser.add_argument("--epochs-head", type=int, default=EPOCHS_HEAD)
    parser.add_argument("--epochs-fine", type=int, default=EPOCHS_FINE)
    parser.add_argument("--limit", type=int, default=0,
                        help="use at most this many images per class (quick check)")
    parser.add_argument("--augment", action="store_true",
                        help="random lighting effects on training photos (augment.py)")
    parser.add_argument("--preprocess", choices=preprocess.MODES, default="none",
                        help="lighting fix run on every photo first (preprocess.py)")
    args = parser.parse_args(argv)
    train(args.data, args.out, args.classes, args.epochs_head, args.epochs_fine, args.limit,
          args.augment, args.preprocess)
    return 0


if __name__ == "__main__":
    sys.exit(main())
