# Public datasets

Raw downloads are not in Git. `python ml/src/download.py` fetches them into
`ml/data/raw/<dataset>/` (git-ignored), untouched; the links below are the sources.
Check each licence on its page before use and credit the authors in the app.

| Folder in raw/ | Dataset | Crops / classes | Notes |
|---|---|---|---|
| `plantvillage/` | [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (Kaggle); downloaded from the authors' [GitHub mirror](https://github.com/spMohanty/PlantVillage-Dataset) (no Kaggle login needed) | Tomato (incl. early/late blight, leaf curl, bacterial spot, septoria), grape | Lab photos, plain background. Use the `color` folder. Avoid the "Augmented" copies. Licence not stated in the repo; check the Kaggle page before release. |
| `plantdoc/` | [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) (GitHub) | Tomato and others | Real-world photos. CC BY 4.0. |
| `ngld_grape/` | [Niphad Grape Leaf Disease Dataset](https://data.mendeley.com/datasets/8nnd2ypcv3) (Mendeley) | Grape: downy mildew, powdery mildew, bacterial leaf spot, healthy | Page says 2,726 images from Nashik farms, 2023–2025; the v5 public API gave 2,473 (1 unreadable). CC BY 4.0. |
| `cold_onion/` | [COLD onion dataset](https://huggingface.co/datasets/Project-AgML/COLD_onion_leaf_disease_classification) (Hugging Face; Mendeley DOI 10.17632/7nxxn4gj5s) | Onion: purple blotch, Stemphylium/Colletotrichum blight (one class), Iris yellow spot virus, healthy | 815 raw images, only 18 of them purple blotch. Use raw, not the augmented set. CC BY 4.0. |
| `mendeley_tomato/` | [Tomato Leaf Disease Classification v2](https://data.mendeley.com/datasets/jttrv2w27r) (Mendeley) | Tomato: early/late blight, leaf curl, bacterial spot, healthy, magnesium deficiency (→ `nutrient_deficiency`), leaf miner (skipped) | 2,995 phone photos outdoors in Bangladesh, March 2025, on a white sheet. CC BY 4.0. |
| `tom2024/` | [TOM2024](https://data.mendeley.com/datasets/3d4yg89rtr/1) (Mendeley; [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11871477/)) | Onion: Alternaria (→ `onion__purple_blotch`), healthy; virosis skipped | Field photos from Burkina Faso, cropped and checked by experts. Only `CATEGORY A/CATA-English/onion_diseases` downloaded (1,397 images, small crops about 150–280 px). CC BY 4.0. |
| — | [AgroVision GrapeLeaf](https://ieee-dataport.org/documents/agrovision-grapeleaf-pathology-dataset) (IEEE DataPort) | Grape, 7 classes incl. anthracnose | 247 images from Satara. May need institutional access; not downloaded. |

## Gaps against the draft `ml/classes.yaml` (raise in Phase 1 talks)

- Grape anthracnose and rust: almost no public images, so own photos only.
- Onion downy mildew and thrips damage: no public images found.
- Onion Iris yellow spot virus is in COLD but not in the draft list.
- COLD merges Stemphylium blight and anthracnose (twister) into one class.

## Counts after merging (2026-10-17)

`python ml/src/dataset.py --preview` on the six public sets, downloaded with
`ml/src/download.py`; 70/15/15 split, train capped at 1,500 per class. 18,455
images (12,110 train, 3,118 val, 3,227 test); 2,675 lab train photos dropped by
the cap, 216 exact duplicates dropped, 2 images left out for appearing under two
classes. "Test (field)" is the part of the test split that is real-world photos.

| Class | Train | Val | Test | Test (field) | Total | Sources |
|---|---|---|---|---|---|---|
| tomato__healthy | 1410 | 308 | 310 | 73 | 2028 | mendeley_tomato 380, plantdoc 63, plantvillage 1585 |
| tomato__early_blight | 1153 | 207 | 252 | 87 | 1612 | mendeley_tomato 524, plantdoc 88, plantvillage 1000 |
| tomato__late_blight | 1463 | 326 | 335 | 30 | 2124 | mendeley_tomato 112, plantdoc 111, plantvillage 1901 |
| tomato__leaf_curl_virus | 1500 | 781 | 844 | 43 | 3125 | mendeley_tomato 234, plantdoc 75, plantvillage 2816 |
| tomato__bacterial_spot | 1500 | 363 | 356 | 34 | 2219 | mendeley_tomato 117, plantdoc 109, plantvillage 1993 |
| tomato__septoria_leaf_spot | 1327 | 299 | 294 | 21 | 1920 | plantdoc 149, plantvillage 1771 |
| onion__healthy | 733 | 178 | 180 | 180 | 1091 | cold_onion 426, tom2024 665 |
| onion__purple_blotch | 365 | 85 | 74 | 74 | 524 | cold_onion 18, tom2024 506 |
| onion__iris_yellow_spot_virus | 196 | 35 | 48 | 48 | 279 | cold_onion 279 |
| grape__healthy | 908 | 217 | 198 | 141 | 1323 | ngld_grape 831, plantdoc 69, plantvillage 423 |
| grape__downy_mildew | 674 | 148 | 144 | 144 | 966 | ngld_grape 966 |
| grape__powdery_mildew | 294 | 50 | 56 | 56 | 400 | ngld_grape 400 |
| grape__bacterial_leaf_spot | 64 | 18 | 18 | 18 | 100 | ngld_grape 100 |
| nutrient_deficiency | 523 | 103 | 118 | 118 | 744 | mendeley_tomato 744 |

Train is 23.4x apart (tomato leaf curl 1,500 vs grape bacterial leaf spot 64),
so Phase 3 uses class weights (point 4 of "Workflow without field visits").
Under 300 images: onion__iris_yellow_spot_virus (279), grape__bacterial_leaf_spot
(100). Tomato test sets are mostly lab photos (21 to 87 field photos per class),
so the tomato field-accuracy numbers in Phase 3 will be rough.

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

## Workflow without field visits (decided 2026-10-17)

Farmer visits and own photos are not possible for now, so Phase 2 uses public
data only:

1. **Paused classes.** Classes with no public images are marked
   `active: false` in `ml/classes.yaml` and left out: onion Stemphylium blight,
   onion anthracnose (twister), onion thrips damage, grape anthracnose,
   healthy_in_shadow, dust_or_water, not_a_leaf. 14 classes stay active.
2. **Train / val / test = 70 / 15 / 15**, from each image's hash. The test
   split replaces the field test set: never trained or tuned on.
3. **Lab or field.** Each source in `ml/sources.yaml` has `kind: lab`
   (PlantVillage) or `kind: field` (all others). Phase 3 reports test accuracy
   for each kind; only the field number counts as real-world accuracy.
4. **Train cap of 1,500 per class**, dropping lab photos first. Val and test
   are not capped. Any imbalance left over (over 3x) is handled with class
   weights in training, an accepted exception to the Phase 2 "within 3x" check.
5. **Label check by eye.** `--preview` writes one sheet of 25 sample photos per
   class to `data/processed/preview/`. This replaces the expert check until an
   expert is available. Done 2026-10-17: all 14 sheets checked, labels look
   right.
6. **Lighting.** Public photos have no lighting tags, so Phase 3 measures the
   lighting drop with synthetic dark, shadow and harsh-sun versions of the
   test split, as in the Phase 0 practice run.

`data/field_test/` still works if real field photos become available later.

**Running it in Colab:** open `ml/notebooks/phase2_dataset.ipynb`. It downloads
the six public sets with `ml/src/download.py` (about 30 minutes; Mendeley is
slow), runs `dataset.py --preview` and shows the sample sheets. It does not use
Google Drive (decided 2026-10-17), so each new Colab session downloads again;
Phase 3 training does the same. AgroVision GrapeLeaf needs a login, so
`download.py` leaves it out.

## Field test layout

Put each held-out photo in `data/field_test/<class id>/<lighting>/`, where
lighting is `good`, `shadow`, `harsh_sun` or `low_light`. Phase 3 reports
accuracy per lighting tag, and `dataset.py` counts photos per tag and flags
untagged ones.

## Onion top-up (2026-10-17)

TOM2024 added 506 purple blotch and 665 healthy onion images (23 exact
duplicates dropped), so `onion__purple_blotch` goes from 18 to about 524.
Its `Virosis-D` folder (203) is skipped: most photos show twisted, curled
leaves, which could be anthracnose-twister rather than a virus. An expert
should check it before it is mapped to any class.

Still no public images for onion Stemphylium blight (alone), anthracnose
(twister) or thrips damage. Candidates that need a login to download:

- [onion_disease_detection](https://universe.roboflow.com/aditya-rias5/onion_disease_detection)
  (Roboflow): 617 images, purple blotch and Stemphylium by stage, object
  detection boxes (crop each box to a leaf image).
- [Onion Leaf Disease](https://universe.roboflow.com/school-0xa1n/onion-leaf-disease-eh38e)
  (Roboflow): 504 images, purple blotch, Stemphylium and others, CC BY 4.0.
