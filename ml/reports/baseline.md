# Phase 3 baseline

Written by `ml/src/evaluate.py` on 2026-10-17. Model: MobileNetV3Large (ImageNet), include_preprocessing, 224x224, trained by `ml/src/train.py` on 12110 images (5 epochs last layer, 10 fine-tuning, 10.0 minutes; best epoch 15). No lighting augmentation in training.

## Summary

- Validation accuracy (good light, 3118 images): 94.3%
- Test accuracy, field photos, good light: 85.7%
- shadow: 78.4% (-7.3 points)
- harsh_sun: 79.1% (-6.6 points)
- low_light: 57.4% (-28.3 points)

Target: shadow and harsh sun within 10 points of good light; under 5% wrong answers given with 70% or more confidence (SYSTEM_DESIGN, Testing and metrics).

## Test accuracy by lighting (synthetic)

Each test image is checked unchanged (good) and with a synthetic effect: shadow = one side of a straight line x 0.4 (30% to 60% of the image); harsh_sun = x 1.6 + 40, clipped at 255; low_light = x 0.35 plus noise (std 6).

| Lighting | Field | Lab | All | Field, per-class average | Field drop vs good | Field wrong and confident |
|---|---|---|---|---|---|---|
| good | 85.7% | 98.2% | 94.1% | 80.3% | — | 7.4% |
| shadow | 78.4% | 90.7% | 86.6% | 73.6% | -7.3 | 10.2% |
| harsh_sun | 79.1% | 94.2% | 89.2% | 72.7% | -6.6 | 11.0% |
| low_light | 57.4% | 69.8% | 65.7% | 51.6% | -28.3 | 24.6% |

1067 field and 2160 lab test images per lighting condition.

## Accuracy per class (test, good light)

| Class | Field | Field images | Lab | Lab images | Shadow (field) | Harsh sun (field) | Low light (field) |
|---|---|---|---|---|---|---|---|
| tomato__healthy | 83.6% | 73 | 100.0% | 237 | 71.2% | 60.3% | 67.1% |
| tomato__early_blight | 64.4% | 87 | 93.9% | 165 | 52.9% | 52.9% | 17.2% |
| tomato__late_blight | 70.0% | 30 | 99.0% | 305 | 90.0% | 73.3% | 96.7% |
| tomato__leaf_curl_virus | 53.5% | 43 | 97.6% | 801 | 48.8% | 39.5% | 53.5% |
| tomato__bacterial_spot | 23.5% | 34 | 97.8% | 322 | 23.5% | 8.8% | 20.6% |
| tomato__septoria_leaf_spot | 66.7% | 21 | 100.0% | 273 | 47.6% | 66.7% | 9.5% |
| onion__healthy | 87.2% | 180 | — | 0 | 83.3% | 78.9% | 98.9% |
| onion__purple_blotch | 100.0% | 74 | — | 0 | 94.6% | 98.6% | 81.1% |
| onion__iris_yellow_spot_virus | 95.8% | 48 | — | 0 | 91.7% | 93.8% | 56.2% |
| grape__healthy | 97.9% | 141 | 100.0% | 57 | 95.0% | 97.2% | 94.3% |
| grape__downy_mildew | 99.3% | 144 | — | 0 | 99.3% | 97.2% | 39.6% |
| grape__powdery_mildew | 96.4% | 56 | — | 0 | 69.6% | 75.0% | 16.1% |
| grape__bacterial_leaf_spot | 100.0% | 18 | — | 0 | 100.0% | 88.9% | 61.1% |
| nutrient_deficiency | 85.6% | 118 | — | 0 | 62.7% | 87.3% | 10.2% |

## Confusion matrix (test, good light, field and lab)

Rows are the true class, columns the model's answer, by number:

1. tomato__healthy
2. tomato__early_blight
3. tomato__late_blight
4. tomato__leaf_curl_virus
5. tomato__bacterial_spot
6. tomato__septoria_leaf_spot
7. onion__healthy
8. onion__purple_blotch
9. onion__iris_yellow_spot_virus
10. grape__healthy
11. grape__downy_mildew
12. grape__powdery_mildew
13. grape__bacterial_leaf_spot
14. nutrient_deficiency

| true \ answer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **298** | 5 | . | 2 | 3 | . | . | . | . | 1 | . | . | . | 1 |
| 2 | 4 | **211** | 10 | 3 | 4 | 6 | . | . | . | . | . | . | . | 14 |
| 3 | 1 | 10 | **323** | . | . | . | . | . | . | . | . | . | . | 1 |
| 4 | 4 | 13 | 5 | **805** | 10 | 6 | . | . | . | . | . | . | . | 1 |
| 5 | 5 | 14 | 4 | 1 | **323** | 7 | . | . | . | . | . | . | . | 2 |
| 6 | . | 4 | . | . | 3 | **287** | . | . | . | . | . | . | . | . |
| 7 | . | . | . | . | . | . | **157** | 5 | 18 | . | . | . | . | . |
| 8 | . | . | . | . | . | . | . | **74** | . | . | . | . | . | . |
| 9 | . | . | . | . | . | . | 2 | . | **46** | . | . | . | . | . |
| 10 | 2 | . | . | . | 1 | . | . | . | . | **195** | . | . | . | . |
| 11 | . | . | . | . | . | . | . | . | . | . | **143** | 1 | . | . |
| 12 | . | . | . | . | . | . | . | . | . | . | 1 | **54** | 1 | . |
| 13 | . | . | . | . | . | . | . | . | . | . | . | . | **18** | . |
| 14 | 2 | 12 | . | 1 | 1 | . | . | . | . | . | . | 1 | . | **101** |

Most common mix-ups:

- onion__healthy taken for onion__iris_yellow_spot_virus: 18
- tomato__early_blight taken for nutrient_deficiency: 14
- tomato__bacterial_spot taken for tomato__early_blight: 14
- tomato__leaf_curl_virus taken for tomato__early_blight: 13
- nutrient_deficiency taken for tomato__early_blight: 12
