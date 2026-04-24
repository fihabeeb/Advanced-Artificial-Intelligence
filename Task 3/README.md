# Task 3 — AI Model Management & User Interaction Tracking

This task covers two related features built on top of the BRFN Django application:

1. **Adding new AI models** to the running system without restarting any services.
2. **Recording user interactions** with both AI models (quality scanner and recommendation engine).

---

## System Overview

The application uses two AI models served by a separate FastAPI microservice (`ml-service`):

| Model | Type | Purpose |
|-------|------|---------|
| Quality Predictor | Keras image classifier | Scores produce images (condition, colour, size) with Grad-CAM heatmaps |
| Recommendation Engine | LSTM (sigmoid) | Recommends products to customers based on purchase history |

The Django application communicates with the ml-service over HTTP. Admins can upload replacement model files through the Django admin panel; the ml-service hot-reloads them immediately with no restart required.

---

## Files

### ML Service (FastAPI)

#### `main.py`
**Origin:** `DESD_BRFN/ml-service/main.py`

The FastAPI application that serves both AI models. Key endpoints:

- `GET /health` — reports whether each model is loaded.
- `POST /predict/quality` — accepts an image upload, returns a quality score, grade, per-attribute breakdown, labels, and Grad-CAM heatmap images encoded as base64.
- `POST /predict/recommendations` — accepts a user ID and purchase history, returns ranked product recommendations.
- `POST /models/upload` — accepts a new `.keras` model file (and optional `.pkl` mappings file), saves it to a timestamped version directory under `ml/versions/<model_type>/<timestamp>/`, copies it to the active path, and hot-reloads the model.
- `GET /models/versions` — lists all previously uploaded versions for each model type.

On startup, both models are loaded from their configured file paths via the `lifespan` context manager.

#### `predictor.py`
**Origin:** `DESD_BRFN/ml-service/predictor.py`

Handles loading and running the quality prediction model inside the ml-service container. Reads the model from `ML_MODEL_PATH` (defaults to `ml/best_model.keras`) and encoders from `ML_ENCODER_PATH` (defaults to `ml/encoders.pkl`). The `predict()` function preprocesses an uploaded image, runs inference, and returns scores, grades, and labels. Also exposes `get_model()` and `get_encoders()` used by the Grad-CAM generator.

---

### Django Admin — Model Upload UI

#### `views.py`
**Origin:** `DESD_BRFN/insights/views.py`

Django views for the admin insights section. The key view for this task is `upload_model()`:

- Decorated with `@staff_member_required` so only admins can access it.
- On POST, reads the uploaded model file(s) from `request.FILES` and forwards them to `ml-service/models/upload` via an HTTP POST.
- On GET (or after upload), fetches the current version list from `ml-service/models/versions` and passes it to the template for display.
- Also contains `classification_insights()` (run a quality scan in the admin) and `recommendation_insights()` (test recommendations for a specific customer).

#### `urls.py`
**Origin:** `DESD_BRFN/insights/urls.py`

URL routes for the insights admin section:

| URL | View | Name |
|-----|------|------|
| `/` | `insights_index` | `insights_index` |
| `recommendations/` | `recommendation_insights` | `insights_recommendations` |
| `classification/` | `classification_insights` | `insights_classification` |
| `models/upload/` | `upload_model` | `insights_upload_model` |

#### `upload_model.html`
**Origin:** `DESD_BRFN/insights/templates/admin/insights/upload_model.html`

Django admin template for the model upload page. Provides:

- A dropdown to select model type (`quality` or `recommendation`).
- A file picker for the `.keras` model file (required).
- A file picker for the `.pkl` mappings/encoders file (required for recommendation, optional for quality).
- An "Upload & Activate" submit button.
- A table showing all previously uploaded versions retrieved from the ml-service.

---

### User Interaction Tracking

#### `models.py`
**Origin:** `DESD_BRFN/interactions/models.py`

Defines the `UserInteraction` Django model. Each record captures a single user action with fields:

| Field | Description |
|-------|-------------|
| `user` | FK to `RegularUser` (nullable for anonymous sessions) |
| `session_key` | Session key for anonymous users |
| `interaction_type` | One of the interaction type constants below |
| `product` | FK to `Product` (nullable) |
| `metadata` | JSON field for event-specific data |
| `timestamp` | Auto-set on creation, indexed |

**Interaction types:**

| Constant | Value | Triggered when |
|----------|-------|----------------|
| `PRODUCT_VIEWED` | `product_viewed` | User opens a product detail page |
| `ADDED_TO_CART` | `added_to_cart` | User adds a product to their cart |
| `PURCHASED` | `purchased` | (defined, not yet wired) |
| `RECOMMENDATION_CLICKED` | `recommendation_clicked` | User clicks a recommended product card |
| `QUALITY_SCAN` | `quality_scan` | Producer runs an AI quality scan on an image |
| `RECOMMENDATION_SERVED` | `recommendation_served` | Recommendations are shown to a logged-in customer |

#### `utils.py`
**Origin:** `DESD_BRFN/interactions/utils.py`

Contains the `log_interaction(request, interaction_type, product=None, metadata=None)` helper. It is fire-and-forget — exceptions are caught and logged as warnings so a tracking failure never breaks the calling view. Called directly in views after AI model interactions occur.

#### `interactions/views.py`
**Origin:** `DESD_BRFN/interactions/views.py`

Contains two views:

- `recommendation_click(request, product_id)` — logs a `recommendation_clicked` interaction for the given product then redirects the user to that product's detail page. Recommendation cards in the product listing link through this view so every click is recorded.
- `export_csv(request)` — staff-only endpoint that exports all `UserInteraction` records as a CSV file. Supports optional `?type=`, `?from=`, and `?to=` query parameters for filtering. The exported data is intended for offline model retraining.

#### `interactions/urls.py`
**Origin:** `DESD_BRFN/interactions/urls.py`

URL routes for the interactions app (mounted at `i/` under `mainApp`):

| URL | View | Name |
|-----|------|------|
| `export/` | `export_csv` | `export_csv` |
| `recommendation-click/<product_id>/` | `recommendation_click` | `recommendation_click` |

---

## Data Flow

### Quality Scan (Producer)
1. Producer uploads an image on the quality scan page.
2. `producers/views.py:quality_scan_view` calls `ml/predictor.py:predict()`.
3. `ml/predictor.py` POSTs the image to `ml-service/predict/quality`.
4. The ml-service runs the Keras model and returns scores, grade, labels, and Grad-CAM images.
5. `quality_scan_view` calls `log_interaction(..., QUALITY_SCAN, metadata={score, grade, breakdown, labels})`.
6. The result is returned as JSON to the producer's browser.

### Recommendations (Customer)
1. A logged-in customer visits the product listing page.
2. `products/views.py:product_list` fetches their order history and calls `LSTMServiceSigmoid.get_recommendations()`.
3. `LSTMServiceSigmoid` POSTs to `ml-service/predict/recommendations`.
4. The ml-service runs the LSTM model and returns ranked product IDs.
5. `product_list` calls `log_interaction(..., RECOMMENDATION_SERVED, metadata={product_ids, count})`.
6. Recommendation cards are rendered in the page, each linking to `i/recommendation-click/<id>/`.
7. When the customer clicks a card, `recommendation_click` logs `recommendation_clicked` and redirects to the product.

### Uploading a New Model (Admin)
1. Admin navigates to `insights/models/upload/` in the Django admin.
2. Selects model type, uploads `.keras` (and optionally `.pkl`) file.
3. `insights/views.py:upload_model` forwards the files to `ml-service/models/upload`.
4. The ml-service saves the files to `ml/versions/<type>/<timestamp>/`, copies them to the active paths, and hot-reloads the model in memory.
5. All subsequent predictions use the new model immediately — no restart required.
