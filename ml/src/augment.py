"""Random lighting changes for training photos (Phase 4 lighting robustness).

Phase 4, see docs/SYSTEM_DESIGN.md.

train.py --augment runs random_augment() on every training photo, freshly each
epoch, so the model learns to ignore shadows, glare, exposure, colour of the light
and blur. Plain numpy on a 0-255 float RGB image; no TensorFlow, so it can be
unit-tested on a laptop.

The ranges below are wider and softer than the fixed effects evaluate.py uses on
the test split (hard-edged shadow x 0.4, x 1.6 + 40 harsh sun, x 0.35 low light),
so a better test score cannot come from learning to undo one exact effect. They
still overlap in kind (a dark photo is a dark photo), so gains on the synthetic
test overstate real-world gains until real shadowed photos are tested.

Not done yet: cluttered backgrounds (needs leaf cut-outs we do not have).
"""
import numpy as np

# Each effect runs with its own chance, in this order. Values are starting
# guesses for Phase 4, not tuned on the test split.
COLOUR_P = 0.5
COLOUR_GAIN = (0.85, 1.15)      # per channel: warm evening light to cool shade
EXPOSURE_P = 0.6
EXPOSURE_GAIN = (0.25, 1.8)     # picked log-uniform: very dark to overexposed
EXPOSURE_OFFSET = (-10, 45)     # added after the gain; high values wash out like harsh sun
GAMMA = (0.7, 1.4)
CONTRAST = (0.7, 1.3)
SHADOW_P = 0.5
SHADOW_DARK = (0.25, 0.8)       # shadowed pixels x this
SHADOW_COVER = (0.2, 0.7)       # share of the image behind a straight shadow edge
SHADOW_BLOB = (0.15, 0.5)       # radius of a round shadow (leaf above), share of the side
SHADOW_SOFT = (0.0, 0.08)       # edge width, share of the image side; 0 is a hard edge
GLARE_P = 0.3
GLARE_SPOTS = (1, 3)
GLARE_RADIUS = (0.04, 0.2)      # share of the shorter side
GLARE_STRENGTH = (0.5, 1.0)     # 1 = pure white at the centre
BLUR_P = 0.2
BLUR_SIGMA = (0.5, 2.0)         # pixels at 224x224: slight shake to clearly soft
NOISE_P = 0.4
NOISE_STD = (2.0, 10.0)         # camera noise, 0-255 units


def colour_shift(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return image * rng.uniform(*COLOUR_GAIN, size=3).astype(np.float32)


def exposure(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gamma, contrast around the mean, then gain and offset: dark to washed out."""
    out = 255.0 * (np.clip(image, 0, 255) / 255.0) ** rng.uniform(*GAMMA)
    mean = out.mean()
    out = mean + (out - mean) * rng.uniform(*CONTRAST)
    gain = np.exp(rng.uniform(np.log(EXPOSURE_GAIN[0]), np.log(EXPOSURE_GAIN[1])))
    return out * gain + rng.uniform(*EXPOSURE_OFFSET)


def shadow_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """0 to 1, how much each pixel is in shadow: a straight edge or a round blob,
    with a hard or soft edge."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if rng.random() < 0.5:
        angle = rng.uniform(0, 2 * np.pi)
        along = xx * np.cos(angle) + yy * np.sin(angle)
        dist = along - np.quantile(along, rng.uniform(*SHADOW_COVER))
    else:
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        ry, rx = rng.uniform(*SHADOW_BLOB) * h, rng.uniform(*SHADOW_BLOB) * w
        r = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
        dist = (r - 1) * min(ry, rx)
    soft = max(rng.uniform(*SHADOW_SOFT) * max(h, w), 0.5)
    return 1.0 / (1.0 + np.exp(np.clip(dist / soft, -50, 50)))


def shadow(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    factor = 1.0 - (1.0 - rng.uniform(*SHADOW_DARK)) * shadow_mask(h, w, rng)
    return image * factor[..., None]


def glare(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A few bright spots that fade to white at the centre, like sun on a wet leaf."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = image
    for _ in range(rng.integers(GLARE_SPOTS[0], GLARE_SPOTS[1] + 1)):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        r = rng.uniform(*GLARE_RADIUS) * min(h, w)
        alpha = rng.uniform(*GLARE_STRENGTH) * np.exp(-2.0 * ((yy - cy) ** 2 + (xx - cx) ** 2) / r ** 2)
        out = out * (1 - alpha[..., None]) + 255.0 * alpha[..., None]
    return out


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(2 * sigma)))
    kernel = np.exp(-np.arange(-radius, radius + 1) ** 2 / (2 * sigma ** 2))
    kernel = (kernel / kernel.sum()).astype(np.float32)
    h, w = image.shape[:2]
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    rows = sum(k * padded[i:i + h] for i, k in enumerate(kernel))
    return sum(k * rows[:, i:i + w] for i, k in enumerate(kernel))


def blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return gaussian_blur(image, rng.uniform(*BLUR_SIGMA))


def noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return image + rng.normal(0, rng.uniform(*NOISE_STD), image.shape).astype(np.float32)


EFFECTS = (  # (chance, effect), applied in this order
    (COLOUR_P, colour_shift),
    (EXPOSURE_P, exposure),
    (SHADOW_P, shadow),
    (GLARE_P, glare),
    (BLUR_P, blur),
    (NOISE_P, noise),
)


def random_augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The image (H x W x 3, 0-255) with random lighting effects, as float32 0-255."""
    out = image.astype(np.float32)
    for chance, effect in EFFECTS:
        if rng.random() < chance:
            out = effect(out, rng)
    return np.clip(out, 0, 255).astype(np.float32)
