"""Fine-tune MobileNetV3-Large on data/processed (the Phase 3 baseline model).

Phase 3, see docs/SYSTEM_DESIGN.md.

    data/processed/manifest.csv     which image is in train / val (from dataset.py)
    models/baseline.keras           best model by val accuracy
    models/labels.txt               class ids, one per line, in model output order
    models/train_info.json          settings and accuracy per epoch, read by evaluate.py

Two stages, as in the Phase 0 practice run: train only the new last layer with
the pretrained model frozen, then unfreeze its top layers with a small learning
rate. Only the val split picks the best epoch; the test split is never read here.

No lighting augmentation on purpose: this baseline has to show how much shadow
and harsh sun hurt, so Phase 4 can show the fix. Rare classes get a bigger class
weight, because train counts are 23x apart (docs/datasets.md).

Usage:
    python ml/src/train.py                          # data in ml/data
    python ml/src/train.py --data /content/data     # in Colab
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

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR / "src"))
import dataset  # noqa: E402

IMG = 224               # SYSTEM_DESIGN Phase 3: 224x224
BATCH = 32
SEED = 42
EPOCHS_HEAD = 5         # frozen pretrained model, new last layer only
EPOCHS_FINE = 10        # top layers unfrozen; stops early if val stops improving
FINE_TUNE_LAYERS = 60   # last layers of MobileNetV3-Large (188 layers) to unfreeze
LR_HEAD = 1e-3          # Phase 0 practice run
LR_FINE = 5e-5          # small, so fine-tuning nudges the pretrained weights
DROPOUT = 0.2           # Phase 0 practice run
PATIENCE = 3            # epochs without a better val accuracy before stopping
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


def make_dataset(data_dir: Path, rows: list, class_ids: list, training: bool):
    import tensorflow as tf
    paths = [str(data_dir / r["file"]) for r in rows]
    labels = [class_ids.index(r["class_id"]) for r in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # Keep the resized images in memory (about 150 KB each) so each image file
    # is decoded only once.
    ds = ds.map(lambda p, y: (tf.cast(tf.round(load_image(p)), tf.uint8), y),
                num_parallel_calls=tf.data.AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(SHUFFLE, seed=SEED, reshuffle_each_iteration=True)
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
          epochs_fine: int, limit: int = 0) -> dict:
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
    print(f"{len(train_rows)} train and {len(val_rows)} val images, {len(class_ids)} classes")

    train_ds = make_dataset(data_dir, train_rows, class_ids, training=True)
    val_ds = make_dataset(data_dir, val_rows, class_ids, training=False)
    model, base = build_model(len(class_ids))

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "baseline.keras"
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
    args = parser.parse_args(argv)
    train(args.data, args.out, args.classes, args.epochs_head, args.epochs_fine, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
