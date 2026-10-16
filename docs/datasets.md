# Public datasets

Raw downloads live in Google Drive, not Git: `MyDrive/leafsight/data/raw/<dataset>/`,
kept zipped and untouched, with a `SOURCE.txt` (link + licence + authors).
Check each licence on its page before use and credit the authors in the app.

| Folder in raw/ | Dataset | Crops / classes | Notes |
|---|---|---|---|
| `plantvillage/` | [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (Kaggle); downloaded from the authors' [GitHub mirror](https://github.com/spMohanty/PlantVillage-Dataset) (no Kaggle login needed) | Tomato (incl. early/late blight, leaf curl, bacterial spot, septoria), grape | Lab photos, plain background. Use the `color` folder. Avoid the "Augmented" copies. Licence not stated in the repo; check the Kaggle page before release. |
| `plantdoc/` | [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) (GitHub) | Tomato and others | Real-world photos. CC BY 4.0. |
| `ngld_grape/` | [Niphad Grape Leaf Disease Dataset](https://data.mendeley.com/datasets/8nnd2ypcv3) (Mendeley) | Grape: downy mildew, powdery mildew, bacterial leaf spot, healthy | Page says 2,726 images from Nashik farms, 2023–2025; the v5 public API gave 2,473 (1 unreadable). CC BY 4.0. |
| `cold_onion/` | [COLD onion dataset](https://huggingface.co/datasets/Project-AgML/COLD_onion_leaf_disease_classification) (Hugging Face; Mendeley DOI 10.17632/7nxxn4gj5s) | Onion: purple blotch, Stemphylium/Colletotrichum blight (one class), Iris yellow spot virus, healthy | 815 raw images, only 18 of them purple blotch. Use raw, not the augmented set. CC BY 4.0. |
| `mendeley_tomato/` | [Tomato Leaf Disease Classification v2](https://data.mendeley.com/datasets/jttrv2w27r) (Mendeley) | Tomato: early/late blight, leaf curl, bacterial spot, healthy, magnesium deficiency (→ `nutrient_deficiency`), leaf miner (skipped) | 2,995 phone photos outdoors in Bangladesh, March 2025, on a white sheet. CC BY 4.0. |
| — | [AgroVision GrapeLeaf](https://ieee-dataport.org/documents/agrovision-grapeleaf-pathology-dataset) (IEEE DataPort) | Grape, 7 classes incl. anthracnose | 247 images from Satara. May need institutional access; not downloaded. |

## Gaps against the draft `ml/classes.yaml` (raise in Phase 1 talks)

- Grape anthracnose and rust: almost no public images, so own photos only.
- Onion downy mildew and thrips damage: no public images found.
- Onion Iris yellow spot virus is in COLD but not in the draft list.
- COLD merges Stemphylium blight and anthracnose (twister) into one class.

## Counts after merging (2026-10-16)

`python ml/src/dataset.py` on the five public sets, no own photos yet: 19,959
images (17,029 train, 2,930 val), 192 exact duplicates dropped, 2 images left
out for appearing under two classes.

| Class | Train | Val | Total | Sources |
|---|---|---|---|---|
| tomato__healthy | 1720 | 308 | 2028 | mendeley_tomato 380, plantdoc 63, plantvillage 1585 |
| tomato__early_blight | 1405 | 207 | 1612 | mendeley_tomato 524, plantdoc 88, plantvillage 1000 |
| tomato__late_blight | 1798 | 326 | 2124 | mendeley_tomato 112, plantdoc 111, plantvillage 1901 |
| tomato__leaf_curl_virus | 4885 | 781 | 5666 | mendeley_tomato 234, plantdoc 75, plantvillage 5357 |
| tomato__bacterial_spot | 1990 | 363 | 2353 | mendeley_tomato 117, plantdoc 109, plantvillage 2127 |
| tomato__septoria_leaf_spot | 1621 | 299 | 1920 | plantdoc 149, plantvillage 1771 |
| onion__healthy | 353 | 73 | 426 | cold_onion 426 |
| onion__purple_blotch | 16 | 2 | 18 | cold_onion 18 |
| onion__iris_yellow_spot_virus | 244 | 35 | 279 | cold_onion 279 |
| grape__healthy | 1106 | 217 | 1323 | ngld_grape 831, plantdoc 69, plantvillage 423 |
| grape__downy_mildew | 818 | 148 | 966 | ngld_grape 966 |
| grape__powdery_mildew | 350 | 50 | 400 | ngld_grape 400 |
| grape__bacterial_leaf_spot | 82 | 18 | 100 | ngld_grape 100 |
| nutrient_deficiency | 641 | 103 | 744 | mendeley_tomato 744 |

No images for 7 classes: onion__stemphylium_blight, onion__anthracnose_twister,
onion__thrips_damage, grape__anthracnose, healthy_in_shadow, dust_or_water,
not_a_leaf. Not yet balanced within 3x (5,666 vs 18), and the field test set is
empty. `nutrient_deficiency` is tomato only so far; add onion and grape photos.

## Searched for the gaps (2026-10-16), not used

- **ICAR-DOGR onion set** (Pune): 5,330 images, 9 classes incl. twister,
  thrips, anthracnose, Stemphylium and purple blotch, used in
  [arXiv 2509.05341](https://arxiv.org/html/2509.05341). Not public; ask the
  authors (IIT Bombay) or ICAR-DOGR. Best single fix for the onion gaps.
- **COLD on Mendeley** ([DOI 10.17632/7nxxn4gj5s.3](http://doi.org/10.17632/7nxxn4gj5s.3)):
  its 4,502 images are the augmented copies of the 815 raw ones we already have.
- **"Burmese grape" set** ([Mendeley k6gy38xv89](https://data.mendeley.com/datasets/k6gy38xv89/1)):
  Burmese grape is latkan (*Baccaurea ramiflora*), not grapevine. Do not use.
- **AgroVision GrapeLeaf** (row above): only open source with grape
  anthracnose; needs an IEEE DataPort login.

## Field test layout

Put each held-out photo in `data/field_test/<class id>/<lighting>/`, where
lighting is `good`, `shadow`, `harsh_sun` or `low_light`. Phase 3 reports
accuracy per lighting tag, and `dataset.py` counts photos per tag and flags
untagged ones.
