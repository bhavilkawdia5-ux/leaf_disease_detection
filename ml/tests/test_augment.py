"""Tests for ml/src/augment.py, ml/src/preprocess.py and train.augment_and_fix.
Run: python -m unittest discover -s ml/tests"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR / "src"))
import augment  # noqa: E402
import preprocess  # noqa: E402
import train  # noqa: E402

HAS_CV2 = importlib.util.find_spec("cv2") is not None


def grey(value=150.0, size=64) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.float32)


def photo(seed=0, size=64) -> np.ndarray:
    return np.random.default_rng(seed).uniform(20, 235, (size, size, 3)).astype(np.float32)


def rng(seed=0) -> np.random.Generator:
    return np.random.default_rng(seed)


class AugmentTest(unittest.TestCase):
    def test_output_shape_type_and_range(self):
        for seed in range(30):
            out = augment.random_augment(photo(seed), rng(seed))
            self.assertEqual(out.shape, (64, 64, 3))
            self.assertEqual(out.dtype, np.float32)
            self.assertGreaterEqual(out.min(), 0)
            self.assertLessEqual(out.max(), 255)

    def test_same_seed_same_image_and_seeds_differ(self):
        a = augment.random_augment(photo(), rng(7))
        np.testing.assert_array_equal(a, augment.random_augment(photo(), rng(7)))
        outs = [augment.random_augment(photo(), rng(s)) for s in range(10)]
        self.assertGreater(len({o.tobytes() for o in outs}), 5)

    def test_uint8_input_works(self):
        out = augment.random_augment(photo().astype(np.uint8), rng())
        self.assertEqual(out.dtype, np.float32)

    def test_shadow_darkens_only_part_of_the_image(self):
        for seed in range(10):
            out = augment.shadow(grey(), rng(seed))
            self.assertLess(out.min(), 150 * augment.SHADOW_DARK[1] + 1)
            self.assertTrue(np.isclose(out, 150, atol=1).any(), "some of it stays lit")
            self.assertLessEqual(out.max(), 150.001)

    def test_shadow_mask_is_between_0_and_1(self):
        mask = augment.shadow_mask(40, 50, rng(3))
        self.assertEqual(mask.shape, (40, 50))
        self.assertGreaterEqual(mask.min(), 0)
        self.assertLessEqual(mask.max(), 1)

    def test_glare_brightens_towards_white(self):
        out = augment.glare(grey(100), rng(1))
        self.assertGreater(out.max(), 100 + 0.4 * 155)
        self.assertLessEqual(out.max(), 255.001)
        self.assertGreaterEqual(out.min(), 99.999, "glare never darkens")

    def test_exposure_covers_dark_and_bright(self):
        means = [augment.exposure(grey(), rng(s)).mean() for s in range(200)]
        self.assertLess(min(means), 150 * 0.5, "some very dark photos")
        self.assertGreater(max(means), 200, "some overexposed photos")

    def test_blur_keeps_flat_areas_and_softens_edges(self):
        np.testing.assert_allclose(augment.gaussian_blur(grey(), 1.5), grey(), atol=1e-3)
        edge = grey(0)
        edge[:, 32:] = 255
        out = augment.gaussian_blur(edge, 1.5)
        self.assertTrue(0 < out[10, 31, 0] < 255 and 0 < out[10, 32, 0] < 255)
        self.assertEqual(out.shape, edge.shape)

    def test_colour_shift_stays_in_range(self):
        out = augment.colour_shift(grey(), rng())
        ratio = out.reshape(-1, 3)[0] / 150
        self.assertTrue(((ratio >= augment.COLOUR_GAIN[0]) & (ratio <= augment.COLOUR_GAIN[1])).all())

    def test_noise_changes_pixels(self):
        self.assertGreater(augment.noise(grey(), rng()).std(), 1)


class PreprocessTest(unittest.TestCase):
    def test_none_returns_the_image_as_float(self):
        image = photo().astype(np.uint8)
        out = preprocess.apply(image, "none")
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out, image)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            preprocess.apply(photo(), "magic")

    def test_grey_world_removes_a_colour_cast(self):
        image = photo() * np.array([1.0, 0.9, 0.6], dtype=np.float32)   # warm light
        out = preprocess.apply(image, "grey_world")
        means = out.reshape(-1, 3).mean(axis=0)
        self.assertLess(means.max() - means.min(), 3)
        self.assertLessEqual(out.max(), 255)

    def test_grey_world_gain_is_capped(self):
        image = grey(100)
        image[..., 2] = 1   # almost no blue
        out = preprocess.grey_world(image)
        self.assertLessEqual(out[..., 2].max(), 1 * preprocess.GREY_WORLD_MAX_GAIN + 1e-3)

    @unittest.skipUnless(HAS_CV2, "opencv not installed")
    def test_clahe_lifts_a_dark_half(self):
        image = photo(size=96) * 0.8
        image[:, :48] *= 0.25
        out = preprocess.apply(image, "clahe")
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(out[:, :48].mean(), image[:, :48].mean())


class AugmentAndFixTest(unittest.TestCase):
    def test_same_seed_same_result(self):
        a = train.augment_and_fix(photo().astype(np.uint8), 5, "grey_world")
        b = train.augment_and_fix(photo().astype(np.uint8), 5, "grey_world")
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual(a.shape, (64, 64, 3))


if __name__ == "__main__":
    unittest.main()
