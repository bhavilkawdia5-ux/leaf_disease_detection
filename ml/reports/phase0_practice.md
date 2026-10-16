# Phase 0 practice run (2026-10-16)

Notebook: `ml/notebooks/phase0_practice.ipynb` (Colab, T4 GPU).
Practice only, not the LeafSight model.

- Data: PlantVillage via tensorflow_datasets, 38 classes, 85/15 train/val split
- Model: MobileNetV3Small (ImageNet), 3 epochs frozen + 2 epochs fine-tuning (last 30 layers)
- Lighting tests are synthetic, applied to the validation split

| Lighting | Accuracy | Drop vs good light |
|---|---|---|
| Good light | 95.1% | — |
| Dark (pixels × 0.35) | 84.1% | −11.0 pts |
| Half shadow (left half × 0.4) | 85.6% | −9.5 pts |

TFLite (float16) size: not recorded.

**Takeaway:** even on clean lab photos, darkness and shadow cost about 10 points.
Design target: shadow and harsh-sun accuracy within 10 points of good light on the
real field test set (see docs/SYSTEM_DESIGN.md, Testing and metrics).
