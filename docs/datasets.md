# Public datasets

Raw downloads live in Google Drive, not Git: `MyDrive/leafsight/data/raw/<dataset>/`,
kept zipped and untouched, with a `SOURCE.txt` (link + licence + authors).
Check each licence on its page before use and credit the authors in the app.

| Folder in raw/ | Dataset | Crops / classes | Notes |
|---|---|---|---|
| `plantvillage/` | [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (Kaggle) | Tomato (incl. early/late blight, leaf curl, bacterial spot, septoria), grape | Lab photos, plain background. Use the `color` folder. Avoid the "Augmented" copies. |
| `plantdoc/` | [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) (GitHub) | Tomato and others | Real-world photos. |
| `ngld_grape/` | [Niphad Grape Leaf Disease Dataset](https://data.mendeley.com/datasets/8nnd2ypcv3) (Mendeley) | Grape: downy mildew, powdery mildew, bacterial leaf spot, healthy | 2,726 images from Nashik farms, 2023–2025. |
| `cold_onion/` | [COLD onion dataset](https://huggingface.co/datasets/Project-AgML/COLD_onion_leaf_disease_classification) (Hugging Face; Mendeley DOI 10.17632/7nxxn4gj5s) | Onion: purple blotch, Stemphylium/Colletotrichum blight (one class), Iris yellow spot virus, healthy | 815 raw images. Use raw, not the augmented set. |
| — | [AgroVision GrapeLeaf](https://ieee-dataport.org/documents/agrovision-grapeleaf-pathology-dataset) (IEEE DataPort) | Grape, 7 classes incl. anthracnose | 247 images from Satara. May need institutional access; not downloaded. |

## Gaps against the draft `ml/classes.yaml` (raise in Phase 1 talks)

- Grape anthracnose and rust: almost no public images, so own photos only.
- Onion downy mildew and thrips damage: no public images found.
- Onion Iris yellow spot virus is in COLD but not in the draft list.
- COLD merges Stemphylium blight and anthracnose (twister) into one class.
