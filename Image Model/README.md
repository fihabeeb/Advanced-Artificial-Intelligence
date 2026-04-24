# Image Model — Produce Quality Classifier

A multi-output CNN that classifies fruit and vegetable images across three simultaneous attributes: **condition** (Healthy / Rotten), **colour** (red, orange, yellow, green, brown, purple, white), and **size** (small, medium, large). The trained model is used by the BRFN quality scan feature.

---

## Files

### `furtherClassifyHealthyImgs.py`
**Label generation script** — the first step in the pipeline.

Walks a dataset directory whose folders are named `<fruit>__<condition>` (e.g. `Apple__Healthy`). For each image it automatically infers two labels that aren't encoded in the folder name:

- **Colour** — uses K-means clustering on foreground pixels, picking the most saturated cluster rather than the largest. This prevents white background bleed or specular highlights from overriding the fruit's actual colour. Falls back to HSV-range voting if there are too few foreground pixels to cluster.
- **Size** — counts foreground (non-background) pixels and compares against configurable `SMALL_THRESHOLD` / `LARGE_THRESHOLD` constants.

Foreground is isolated by estimating the background colour from the four image corners and masking out pixels within a tolerance of that colour.

Outputs:
- `labels.csv` — one row per image with columns `image_path`, `fruit`, `condition`, `colour`, `size`.
- `label_verification_grid.png` — a thumbnail grid of randomly sampled images with their inferred labels overlaid, for visual checking.
- `labeling.log` — DEBUG-level log of every image processed.

### `imageModelPreprocessing.py`
**Preprocessing script** — the second step in the pipeline.

Reads `labels.csv`, loads every image with OpenCV, resizes to `224×224`, and normalises pixel values to `[0, 1]`. Label-encodes the three target columns and performs an 80/10/10 train/val/test split. Images that fail to load are skipped and their paths written to `failed_images.log`.

Outputs (saved as `.npy` arrays and a `.pkl` file):
- `X_train.npy`, `X_val.npy`, `X_test.npy` — image arrays
- `y_cond_*.npy`, `y_col_*.npy`, `y_size_*.npy` — encoded label arrays for each split
- `encoders.pkl` — the three fitted `LabelEncoder` objects (required at inference time)

The notebook ran on **29,277 images** producing an 80/10/10 split (23,421 / 2,928 / 2,928).

### `fruitsModelCNN.py`
**Model training script** — the third step in the pipeline.

Builds a shared-backbone CNN in Keras with three independent output heads:

| Layer block | Details |
|-------------|---------|
| Data augmentation | Random horizontal flip, ±10% rotation, ±10% zoom |
| Conv block ×4 | Conv2D (32→64→128→256 filters), BatchNorm, MaxPool |
| Shared head | GlobalAveragePooling → Dense(512, ReLU) → Dropout(0.4) → Dense(256, ReLU) |
| Condition head | Dense(128) → Dropout(0.3) → Softmax(2) |
| Colour head | Dense(128) → Dropout(0.3) → Softmax(n_colours) |
| Size head | Dense(128) → Dropout(0.3) → Softmax(3) |

Compiled with Adam (LR=0.001) and `sparse_categorical_crossentropy` per head. The condition head is weighted 2× more heavily than colour and size (2.0 vs 0.5) because condition is the most important label for produce grading.

Training callbacks: `EarlyStopping(patience=4)`, `ModelCheckpoint` (saves `best_model.keras`), `ReduceLROnPlateau(patience=2, factor=0.5)`.

Outputs:
- `best_model.keras` — best checkpoint by validation loss (used in production)
- `fruit_model.keras` — final model at end of training

### `fruits_pipeline.ipynb`
**Interactive notebook** — consolidates all steps above into a single runnable document with embedded outputs.

| Step | Content |
|------|---------|
| 1 | Config — all paths and hyperparameters in one cell |
| 2 | Label generation (same logic as `furtherClassifyHealthyImgs.py`) |
| 3 | Size-threshold check — computes 25th/75th percentile of foreground pixel areas and recommends `SMALL_THRESHOLD` / `LARGE_THRESHOLD` values |
| 4 | Preprocessing (same logic as `imageModelPreprocessing.py`) |
| 5 | CNN training (same logic as `fruitsModelCNN.py`) |
| 6 | Evaluation — classification reports and confusion matrix heatmaps for all three heads on the held-out test set |

The notebook was run on 29,277 images. The dataset folder structure expected is `<DATASET_PATH>/<FruitName>__<Condition>/image.jpg`.

---

## Pipeline Order

```
furtherClassifyHealthyImgs.py   →  labels.csv
        ↓
imageModelPreprocessing.py      →  X_*.npy, y_*.npy, encoders.pkl
        ↓
fruitsModelCNN.py               →  best_model.keras
```

Or run everything in one go via `fruits_pipeline.ipynb`.

---

## Output Files Used in Production

| File | Used by |
|------|---------|
| `best_model.keras` | Loaded by `ml-service/predictor.py` as the active quality model |
| `encoders.pkl` | Loaded alongside the model to decode predicted class indices back to label strings |
