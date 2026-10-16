# LeafSight System Design

> Exported from the "LeafSight System Design" doc (2026-10-16) and adapted for a
> **native Android (Kotlin + Jetpack Compose)** app instead of Flutter. Product,
> pipeline, thresholds, phases and metrics are unchanged; only the app stack,
> file names and paths differ. If the original doc changes, update this file.

## How to use this doc with Claude Code

Build LeafSight one phase at a time: give Claude Code this doc, point it at the
current phase, and stop when that phase's "Done when" checks pass.

1. This doc lives in the repo as `docs/SYSTEM_DESIGN.md`.
2. `CLAUDE.md` in the repo root says:
   - "Read docs/SYSTEM_DESIGN.md before any task."
   - "Work only on the phase I name. Do not start the next phase."
   - "Follow the Coding rules section."
   - "Current phase: Phase N" (update this line as you progress).
3. Start each phase with the ready-made prompt given in that phase's section.
4. Before moving on, tick every "Done when" box yourself. Do not trust "it
   works" without running it on your phone.
5. Commit to Git at the end of each phase with a message like
   `phase-3: baseline model`.

**Golden rule:** always measure on real field photos, never only on lab images.
That is what proves the lighting USP.

## Product summary and scope

LeafSight is an offline Android app that diagnoses crop leaf diseases from a
photo and stays reliable in shadows, harsh sun and low light.

**USP:** "Reliable diagnosis in any lighting, tailored to your farm, your crop
and your phone."

**Users:** small farmers in Maharashtra using budget Android phones, often with
weak or no internet, reading Marathi, Hindi or English.

**Version 1 includes:**

- Guided camera with live checks for light, glare, shadow, blur and leaf size
- One-time camera colour calibration with white paper
- Lighting correction before diagnosis
- On-device disease model for 2 to 3 crops, with a "not sure" answer
- Farm profile (crop, stage, organic or chemical preference) that shapes advice
- Offline advice library, scan history and "Was this right?" feedback
- Tips in English, Hindi and Marathi, with optional voice

**Not in version 1:** cloud sync, weather alerts, expert dashboard, outbreak
maps, AI explanation layer, iOS release.

**Open questions to decide in Phase 1:**

- Which 2 to 3 crops? (Suggested: tomato, onion, grapes or pomegranate)
- Which 4 to 6 diseases per crop?
- Final app name (LeafSight is a placeholder)

**Decided in Phase 1 (2026-10-16):**

- App name: **KrishiCare**
- Crops: tomato, onion, grapes. Diseases: see `ml/classes.yaml`.
- The class list is **provisional**: it was checked against ICAR and NHB
  sources (`docs/farmer_notes.md`) because farmers could not be reached. Confirm
  it with farmers or a KVK officer during Phase 2 field visits.

## Architecture overview

Version 1 runs fully on the phone; cloud and training feed it later.

| Phone, offline (v1) — build first (Phases 5–6) | Cloud (Phase 8) | Training (`ml/` folder) |
|---|---|---|
| Guided camera — live checks, tips | API and sync — FastAPI, with consent | Datasets — public + field photos |
| Lighting fix — calibration, shadow fix | Feedback store — expert-confirmed photos | Augment and train — shadows, glare, blur |
| Disease model — TFLite, not-sure rule | Model updates — latest .tflite version | Field test check — accuracy per lighting |
| Advice and history — SQLite (Room), farm profile | AI explainer — optional, Claude or OpenAI | Export TFLite — under 20 MB |

The phone does all diagnosis offline. When online (Phase 8), it syncs scans and
feedback to the cloud with the user's consent; confirmed photos feed training,
and each new model returns to phones as an over-the-air update.

## On-device diagnosis pipeline

Every photo is checked and corrected before the model sees it. All steps run on
the phone; steps 2–4 are the lighting USP.

1. **Guided camera** — leaf outline, live tips
2. **Quality check** — light, glare, shadow, blur (bad photo → retake with a tip)
3. **Calibration** — white-paper colour gains
4. **Lighting fix** — CLAHE, shadow lift
5. **Leaf crop** — remove background
6. **Disease model** — TFLite, top 3 classes
7. **Confidence check** — above threshold?
8. **Result and advice** (crop, stage, preference) — or **Not sure** (retake or ask KVK) when confidence is low

Quality checks run on a 120x160 copy of the camera frame at least 4 times per
second; the rest runs once, after capture.

**Starting thresholds** (from the prototype; retune on your own photos in
Phase 5 and keep them in `Config.kt`):

| Check | Measure | Warn (yellow) | Bad (red) |
|---|---|---|---|
| No leaf | Share of frame that is leaf-green | — | under 3% |
| Too far | Share of frame that is leaf | under 16% | under 10% |
| Too dark | Median leaf brightness (0–255) | under 68 | under 45 |
| Too bright | Median leaf brightness (0–255) | over 208 | over 228 |
| Glare | Share of near-white pixels around the leaf | over 1.2% | over 3.5% |
| Shadow | Share of leaf blocks under half the bright-area brightness | over 7% | over 15% |
| Blur | Laplacian variance, brightness-normalised | under 140 | under 60 |

**Rules:**

- Auto-capture only when every check is green for 1 second.
- Glare, blur, too far or no leaf at capture: return "not sure" with a retake
  tip; these cannot be corrected.
- Shadow or brightness problems: correct, diagnose, and lower confidence.
- Below the confidence threshold chosen in Phase 4: show "not sure" with the
  top 3 guesses and a retake or KVK suggestion.

## Tech stack and project structure

The app is native Android (Kotlin + Jetpack Compose) with an on-device
TensorFlow Lite (LiteRT) model; training happens separately in Python.

| Part | Choice | Notes |
|---|---|---|
| Mobile app | Kotlin, Jetpack Compose, Material 3, Android Studio | minSdk 24 |
| Camera | CameraX (`Preview` + `ImageAnalysis`) | `ImageAnalysis` stream for live checks |
| On-device AI | LiteRT (`com.google.ai.edge.litert`, TFLite runtime) | Model file in `app/src/main/assets/models/` |
| Image processing | Android `Bitmap` + plain Kotlin pixel loops (OpenCV only if CLAHE needs it) | Quality checks and lighting correction |
| Motion | `SensorManager` (accelerometer) | Shake detection |
| Local database | Room (SQLite) | Profile, scans, feedback |
| Voice | Android `TextToSpeech` | Spoken tips |
| Photo picker | Photo Picker (`ActivityResultContracts.PickVisualMedia`) | Fallback when camera stream fails |
| Background work | Kotlin coroutines (`Dispatchers.Default`) / CameraX analyzer executor | Replaces Flutter isolates |
| Navigation / state | Navigation Compose, `ViewModel` + `StateFlow` | |
| Model training | Python, TensorFlow or PyTorch, Google Colab or PyCharm | Export to `.tflite` |
| Backend (Phase 8) | FastAPI or Firebase | Sync, feedback, model updates |

**Repository layout** (`pkg` = `app/src/main/java/com/example/leaf_disease_detection_project/`):

```
leaf_disease_detection/
  CLAUDE.md
  docs/SYSTEM_DESIGN.md
  app/                          Android app (Kotlin + Compose)
    src/main/
      java/.../                 (pkg)
        MainActivity.kt
        Config.kt               all thresholds, with source comments
        ui/
          screens/              home, camera, result, history, settings, calibration
          components/           status pill, leaf guide, chips
          theme/
        services/
          QualityChecker.kt
          LightingCorrector.kt
          Classifier.kt
          AdviceEngine.kt
        data/                   Room: AppDatabase, entities, DAOs
        model/                  Scan, FarmProfile, QualityReport, Diagnosis, Advice
      assets/
        models/                 model.tflite, labels.txt
        advice/                 advice.json
      res/
        values/strings.xml      English
        values-hi/strings.xml   Hindi
        values-mr/strings.xml   Marathi
    src/test/                   JUnit unit tests
      resources/fixtures/       good, shadow, dark, glare, blur, far, healthy
    src/androidTest/            instrumented tests
  ml/                           Python training project
    data/                       raw, processed, field_test (never trained on)
    notebooks/
    src/                        dataset.py, augment.py, train.py, evaluate.py, export_tflite.py
    reports/                    accuracy tables per experiment
  backend/                      Phase 8 only
```

## Data model

All data lives in SQLite (via Room) on the phone in version 1; nothing leaves
the device.

| Table | Key fields | Purpose |
|---|---|---|
| `farm_profile` | name, language (en/hi/mr), voice_on, auto_capture, treatment_pref (organic/chemical), calib_gain_r/g/b, calib_date | One row per user |
| `plot` | id, crop, variety, sowing_date, stage, irrigation | A user can have several plots |
| `scan` | id, plot_id, created_at, photo_path, corrected_path, quality_json, label, confidence, top3_json, status (diagnosed/unsure) | One row per photo |
| `feedback` | scan_id, answer (right/wrong/unsure), note, created_at | Training signal for later |
| `treatment_log` | id, plot_id, scan_id, product, date, notes | Spray history |

**In-memory objects (Kotlin data classes in `model/`):**

- `QualityReport`: coverage, medianBrightness, glareFraction, shadowFraction,
  sharpness, level per check (0 ok, 1 warn, 2 bad), tipKey
- `Diagnosis`: label, confidence, top3, isUnsure, reason
- `Advice`: steps list, filtered by crop, stage and treatment preference

**Advice library** (`app/src/main/assets/advice/advice.json`): one entry per
crop and disease with symptoms, organic steps, chemical guidance ("ask KVK for
product and dose"), prevention and stage notes, in all three languages. Content
comes from ICAR and state agricultural university guidelines and is checked by
an expert before release.

## Phase-wise build plan

Ten phases take LeafSight from zero to a field-tested app in about 6 to 9
months part-time. Each phase has a goal, tasks, a prompt to give Claude Code,
and checks to tick before moving on.

### Phase 0: Setup and basics (2 to 4 weeks)

**Goal:** a working toolchain and enough ML and Android/Compose basics to follow
the rest.

- Install Android Studio (with SDK), Python 3, Git
- Create the repo with the layout above, `CLAUDE.md` and this doc
- Learn: transfer learning (fast.ai or TensorFlow tutorials), Jetpack Compose
  and state (`ViewModel`, `StateFlow`)

**Claude Code prompt:** "Set up the repo using the layout in SYSTEM_DESIGN.md.
Keep the Android app in app/ and add an empty Python project in ml/ with a
requirements.txt. Do not write features yet."

**Done when:**

- [ ] Default Compose app runs on your phone
- [ ] A sample image classifier trained in Colab
- [ ] Repo pushed to GitHub

### Phase 1: Define scope (1 week)

**Goal:** fixed crops, disease classes and version 1 features.

- Choose 2 to 3 crops and 4 to 6 diseases each, plus healthy
- Talk to 5 to 10 farmers or a KVK officer
- Fill in the open questions in Product summary

**Claude Code prompt:** "Create ml/classes.yaml listing crops and classes from
SYSTEM_DESIGN.md, including confuser classes: healthy_in_shadow,
nutrient_deficiency, dust_or_water, not_a_leaf."

**Done when:**

- [ ] `classes.yaml` agreed and committed
- [ ] Notes from farmer conversations saved in `docs/`

### Phase 2: Data collection (3 to 6 weeks)

**Goal:** a labeled dataset with real field conditions and a separate field
test set.

- Download PlantVillage, PlantDoc and crop-specific Kaggle sets
- Photograph your own leaves: several phones, morning, noon, evening, shade,
  half-shadow, harsh sun
- Aim for 300 or more images per class to start
- Hold out a field test set of at least 50 real photos per class, including
  shadowed ones

**Claude Code prompt:** "Write ml/src/dataset.py to merge the datasets into
data/processed with the class names in classes.yaml, split train/val 85/15, and
keep data/field_test untouched. Print image counts per class and per split."

**Done when:**

- [ ] Counts per class printed and balanced within roughly 3x
- [ ] Field test set never used for training
- [ ] Uncertain labels checked by an expert

### Phase 3: Baseline model (2 to 3 weeks)

**Goal:** a first model and proof that lighting hurts accuracy.

- Fine-tune MobileNetV3 or EfficientNet-Lite at 224x224
- Evaluate on validation set and field test set, split by lighting condition

**Claude Code prompt:** "Write ml/src/train.py to fine-tune MobileNetV3 on
data/processed and ml/src/evaluate.py to report accuracy and a confusion matrix
on the field test set, split by lighting tag (good, shadow, harsh_sun,
low_light). Save results to reports/baseline.md."

**Done when:**

- [ ] `reports/baseline.md` shows accuracy per lighting condition
- [ ] The drop under shadow and harsh sun is measured

### Phase 4: Lighting robustness (3 to 4 weeks)

**Goal:** clearly better accuracy on difficult lighting than the baseline.

- Augment: random shadows, brightness and contrast, colour shift, glare spots,
  blur, cluttered backgrounds
- Try preprocessing: grey-world colour balance and CLAHE
- Add the "not sure" rule: confidence below a threshold set from validation data
- Export the best model to TensorFlow Lite and check its size and speed

**Claude Code prompt:** "Add ml/src/augment.py with a random hard-shadow
augmentation and glare spots, retrain, and compare against reports/baseline.md
per lighting condition. Then add export_tflite.py with float16 quantization."

**Done when:**

- [ ] Shadow and harsh-sun accuracy improved, with numbers in `reports/`
- [ ] Model under 20 MB and under 1 second per image on a budget phone
- [ ] "Not sure" threshold chosen and recorded

### Phase 5: Guided camera and quality checks (2 to 3 weeks)

**Goal:** bad photos get stopped with a clear tip before diagnosis.

- Prototype the checks in Python first and tune thresholds on your photos
- Port to `QualityChecker.kt` and build the camera screen

**Claude Code prompt:** "Implement services/QualityChecker.kt following the
On-device pipeline section, with JUnit tests using sample images in
app/src/test/resources/fixtures. Then build ui/screens/CameraScreen.kt with
CameraX, the leaf outline, status colour, tip text and auto-capture."

**Done when:**

- [ ] Checks run at least 4 times per second without camera lag
- [ ] Shadow, dark, glare, blur and too-far photos are flagged in real field tests
- [ ] Unit tests pass

### Phase 6: Complete offline app (4 to 6 weeks)

**Goal:** version 1 works end to end with no internet.

- Classifier service with LiteRT
- Lighting corrector and calibration screen
- Home, result, history and settings screens
- Room database and advice engine
- English, Hindi and Marathi strings, voice tips

**Claude Code prompt:** "Build Classifier.kt and LightingCorrector.kt, then the
result screen showing original and corrected photos, label, confidence, top 3
and advice from AdviceEngine.kt. Store every scan in Room."

**Done when:**

- [ ] Full flow works in airplane mode on a budget Android phone
- [ ] Hindi and Marathi checked by native speakers

### Phase 7: Field testing (3 to 4 weeks)

**Goal:** real-world accuracy and usability numbers.

- Test with 10 to 20 farmers on their own fields and phones
- Watch without helping; note confusion points
- Collect photos (with permission) and feedback; have an expert verify

**Done when:**

- [ ] Real-world accuracy, "not sure" rate and scan time measured
- [ ] List of usability fixes written

### Phase 8: Improve and add backend (4 to 6 weeks)

**Goal:** version 2, better on accuracy and usability.

- Retrain with field photos, re-test on the field test set
- Fix usability issues
- Backend: consent-based sync, feedback upload, over-the-air model updates
- Optional: expert review page, AI explanation layer via Claude or OpenAI API

**Claude Code prompt:** "Create backend/ with FastAPI endpoints: POST /sync,
POST /feedback, GET /model/latest. The app must keep working fully offline if
the backend is unreachable."

**Done when:**

- [ ] Version 2 beats version 1 on the field test set
- [ ] App updates its model without a Play Store update

### Phase 9: Launch and grow (ongoing)

- Play Store closed testing, then open release
- Partner with a KVK, FPO, NGO or agriculture college
- Add features by farmer demand: weather alerts, follow-up reminders, more
  crops, outbreak alerts
- Retrain regularly with confirmed photos

## Coding rules for Claude Code

These rules keep the app offline, fast and honest; Claude Code should follow
them in every phase.

1. **Offline first.** Every version 1 feature must work in airplane mode.
   Network code is optional and never blocks the user.
2. **One phase at a time.** Do not build features from later phases.
3. **Small, testable services.** Logic lives in `services/` with unit tests;
   composables only display state and call ViewModels/services.
4. **Never guess confidently.** If quality is bad or confidence is below the
   threshold, return "not sure" with a retake tip.
5. **No hard-coded thresholds in code.** Keep them in one `Config.kt` with
   comments saying where each value came from.
6. **Performance budget.** Quality checks on a downscaled frame (about 120x160)
   off the main thread (CameraX analyzer executor / `Dispatchers.Default`), at
   least 4 times per second; diagnosis under 1 second on a budget phone.
7. **All user text in string resources.** `res/values{,-hi,-mr}/strings.xml`.
   No English strings inside composables. Short, plain sentences.
8. **Privacy.** Photos and farm data stay on the phone unless the user turns on
   sync.
9. **Advice safety.** Never invent pesticide names or doses. Chemical advice
   says to ask the KVK or agriculture officer.
10. **ML experiments.** Every experiment writes a report to `ml/reports/` with
    accuracy per lighting condition. The field test set is never used for
    training or tuning.
11. **Commits.** One commit per finished task, message starting with the phase,
    e.g. `phase-5: glare check`.
12. **Explain before big changes.** For any change touching more than 3 files,
    describe the plan first and wait for approval.

## Testing, metrics and glossary

Success is measured on real field photos split by lighting condition; the
targets below are starting goals to revise after Phase 3.

| Metric | Where measured | Starting target |
|---|---|---|
| Accuracy, good light | Field test set | 85% or more |
| Accuracy, shadow and harsh sun | Field test set | Within 10 points of good light |
| Wrong answers given confidently | Field test set | Under 5% |
| "Not sure" rate | Field testing | Under 20% |
| Quality check catches bad photos | Labeled bad photos | 90% or more |
| Diagnosis time | Budget Android phone | Under 1 second |
| Model size | .tflite file | Under 20 MB |
| Full scan time | Farmer field tests | Under 30 seconds |

**Tests to keep:**

- JUnit tests for each service with fixture images in
  `app/src/test/resources/fixtures/` (good, shadow, dark, glare, blur, far,
  healthy)
- Python evaluation script run after every training experiment
- Manual airplane-mode test before every release

**Glossary:**

- **Transfer learning:** starting from a model pretrained on general images and
  fine-tuning it on leaf photos.
- **Augmentation:** randomly changing training images (shadows, brightness,
  blur) so the model learns to ignore those changes.
- **Field test set:** real farm photos kept aside and never used for training,
  used only to measure honest accuracy.
- **Confuser class:** a class for things that look like disease but are not,
  such as a healthy leaf in shadow.
- **CLAHE:** a contrast method that evens out lighting in local areas of an
  image.
- **Laplacian variance:** a sharpness score; low values mean a blurry photo.
- **TensorFlow Lite (LiteRT):** a format for running small models on phones.
- **KVK:** Krishi Vigyan Kendra, the local agricultural science centre.
