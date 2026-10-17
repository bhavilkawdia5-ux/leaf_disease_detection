"""Lighting fixes run on every photo before the model sees it (Phase 4).

Phase 4, see docs/SYSTEM_DESIGN.md ("Try preprocessing: grey-world colour balance
and CLAHE").

train.py --preprocess picks one mode; it is saved in train_info.json, and
evaluate.py and export_tflite.py apply the same mode. The fix is not inside the
model, so the app must run the same step first (LightingCorrector.kt, Phase 6).

    none        the photo as it is
    grey_world  scale R, G and B so their averages are equal (removes the colour of the light)
    clahe       CLAHE on the lightness channel only (evens out dark and bright areas)
"""
import numpy as np

MODES = ("none", "grey_world", "clahe")
GREY_WORLD_MAX_GAIN = 2.0   # stops a nearly one-colour photo from being pushed too far
CLAHE_CLIP = 2.0            # OpenCV's usual clip limit
CLAHE_TILES = 8             # 8 x 8 tiles, OpenCV's usual grid

_clahe = None


def grey_world(image: np.ndarray) -> np.ndarray:
    means = image.reshape(-1, 3).mean(axis=0)
    gains = means.mean() / np.maximum(means, 1e-3)
    gains = np.clip(gains, 1 / GREY_WORLD_MAX_GAIN, GREY_WORLD_MAX_GAIN)
    return np.clip(image * gains.astype(np.float32), 0, 255)


def clahe(image: np.ndarray) -> np.ndarray:
    import cv2
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILES, CLAHE_TILES))
    lab = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab[..., 0] = _clahe.apply(np.ascontiguousarray(lab[..., 0]))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply(image: np.ndarray, mode: str) -> np.ndarray:
    """The image (H x W x 3, 0-255) after the chosen fix, as float32 0-255."""
    if mode not in MODES:
        raise ValueError(f"preprocess mode must be one of {', '.join(MODES)}, got {mode!r}")
    if mode == "grey_world":
        image = grey_world(image.astype(np.float32))
    elif mode == "clahe":
        image = clahe(image)
    return np.asarray(image, dtype=np.float32)
