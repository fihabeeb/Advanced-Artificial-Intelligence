# XAI — Explainable AI & Admin Insights

This folder contains the `insights` Django app, which provides admin-facing tools for inspecting and understanding both AI models. It covers two aspects of Explainable AI (XAI): **Grad-CAM** visual explanations for the image classifier, and an **admin dashboard** for testing and managing both models.

---

## Files

### `gradcam.py`
**Grad-CAM implementation** — generates visual heatmaps that show which regions of an image most influenced the model's condition prediction (Healthy vs Rotten).

#### How Grad-CAM works here
Grad-CAM (Gradient-weighted Class Activation Mapping) computes the gradient of the target class score with respect to the feature maps of the last convolutional layer (`conv2d_3`). These gradients are globally average-pooled to produce per-channel importance weights, which are then used to weight and sum the feature maps into a single heatmap. Regions with high activation contributed most to the prediction.

The implementation targets the **condition head** (`model.output[0]`) specifically, since condition (Healthy/Rotten) is the most important attribute for quality grading.

**Functions:**

| Function | Purpose |
|----------|---------|
| `generate_gradcam(model, img_array, class_index)` | Runs the gradient tape through the condition head and returns a normalised 2D heatmap array |
| `colorize_heatmap(heatmap)` | Applies the JET colormap (blue=low, red=high activation) and returns an RGB array |
| `overlay_heatmap(heatmap, image)` | Resizes the heatmap to match the original image and blends them (60% original, 40% heatmap) |
| `to_base64(img_array)` | Converts a NumPy image array to a base64-encoded PNG string for embedding in HTML |

---

### `views.py`
**Django views for the admin insights section.**

| View | URL | Access | Purpose |
|------|-----|--------|---------|
| `insights_index` | `/` | Staff | Landing page for the insights section |
| `classification_insights` | `classification/` | Staff | Upload an image, get quality prediction + Grad-CAM heatmaps |
| `recommendation_insights` | `recommendations/` | Staff | Enter a customer ID, see their purchase history and live recommendations |
| `upload_model` | `models/upload/` | Staff only (`@staff_member_required`) | Upload a new `.keras` model file to replace the active model in the ml-service |

**`classification_insights`** — POSTs the uploaded image to the ml-service via `ml/predictor.py` and passes the result to the template, including the three base64-encoded Grad-CAM images (original, heatmap, overlay).

**`recommendation_insights`** — looks up the customer's paid order history and calls `LSTMServiceSigmoid.get_recommendations()` to generate live recommendations, showing both the input purchase sequence and the ranked output.

**`upload_model`** — forwards the uploaded file(s) to `ml-service/models/upload` via HTTP POST. Also fetches and displays the version history from `ml-service/models/versions`. See Task 3 for the full upload flow.

---

### `urls.py`
URL routes for the insights app:

| URL | View |
|-----|------|
| `` (root) | `insights_index` |
| `recommendations/` | `recommendation_insights` |
| `classification/` | `classification_insights` |
| `models/upload/` | `upload_model` |

---

### `admin.py`
Registers the insights section in the Django admin using a dummy model (`InsightsDummy`) and a custom `InsightsAdmin` class. The `get_urls()` override injects the four insight pages directly into the admin URL namespace so they appear under the standard admin interface. Add/change/delete permissions are all disabled — the dummy model is never edited, it only exists to create the admin section entry point.

---

### `templates/admin/insights/`

| Template | Rendered by |
|----------|------------|
| `index.html` | `insights_index` — landing page |
| `classification.html` | `classification_insights` — image upload form + results with Grad-CAM images |
| `recommendation.html` | `recommendation_insights` — customer lookup form + purchase history + recommendation results |
| `upload_model.html` | `upload_model` — model file upload form + version history table |

**`classification.html`** displays:
- The overall quality score and grade
- Per-attribute breakdown (condition, colour, size scores)
- Predicted labels and confidence percentages
- Three side-by-side images: original, Grad-CAM heatmap (JET colormap), and blended overlay

---

### `__init__.py` / `apps.py`
Standard Django app boilerplate — no custom logic.

---

## Grad-CAM in the Prediction Flow

```
Admin uploads image
        ↓
classification_insights (views.py)
        ↓
ml/predictor.py  →  POST /predict/quality  →  ml-service/main.py
                                                      ↓
                                             ml-service/predictor.py  (CNN inference)
                                                      ↓
                                             gradcam.py: generate_gradcam()
                                                      ↓
                                             heatmap + overlay encoded as base64
        ↓
classification.html renders three images inline
```

The Grad-CAM is computed inside the ml-service container (where the model is loaded) and returned alongside the prediction scores. The Django app receives it as base64 strings and embeds them directly in the HTML — no separate image files are saved.
