# Recommendation Model — LSTM Next-Product Predictor

A bidirectional LSTM that predicts which product a customer is most likely to buy next, based on their chronological purchase history. The trained model powers the "Personalised for You" section on the BRFN product listing page.

---

## Files

### `model.py`
**Full training pipeline** — data extraction, model architecture, training, and inference.

#### Data extraction (`get_user_sequences`)
Queries paid `OrderItem` records from the Django database, ordered chronologically by payment date. Builds a per-user list of purchased product IDs, repeating each ID by the quantity purchased. Returns `{user_id: [product_id, ...]}`.

#### Training data construction (`build_training_data`)
Creates sliding window sequences from each user's purchase history. For a `sequence_length` of 3 and history `[A, B, C, D, E]`, this produces:
- Input `[A, B, C]` → Target `D`
- Input `[B, C, D]` → Target `E`

Users with fewer items than `sequence_length` are excluded.

#### Product encoding (`encode_products`)
Maps product IDs to sequential integer indices (1-indexed, reserving 0 as the padding token). Saves `product_to_idx` and `idx_to_product` mappings for use at inference time.

#### Model architecture (`build_improved_lstm_model`)

| Layer | Details |
|-------|---------|
| Embedding | `vocab_size × 64`, mask_zero=True (ignores padding) |
| BiLSTM ×1 | 128 units, return_sequences=True, dropout=0.2 |
| BatchNorm + Dropout(0.3) | |
| BiLSTM ×2 | 64 units, return_sequences=False, dropout=0.2 |
| BatchNorm + Dropout(0.3) | |
| Dense(128, ReLU) + BatchNorm + Dropout(0.3) | |
| Dense(64, ReLU) + Dropout(0.2) | |
| Dense(num_products+1, Softmax) | Output over all products |

Compiled with Adam (LR=0.0005) and `sparse_categorical_crossentropy`.

Training callbacks: `EarlyStopping(patience=10)`, `ReduceLROnPlateau(patience=5, factor=0.5)`, `ModelCheckpoint` (saves best by val_accuracy).

#### Inference (`recommend_next_products`)
Takes the last `sequence_length` items from a user's purchase history, encodes them, runs a forward pass, and returns the top-k product IDs with their softmax probabilities. Unknown product IDs are mapped to the padding token.

Outputs saved after training:
- `ml/recommendation/final/recommendation_model.keras` — trained model
- `ml/recommendation/product_mappings.pkl` — `product_to_idx`, `idx_to_product`, `sequence_length`, `num_products`

---

### `service.py`
**Django service class** — wraps the trained model for use inside the Django application.

`RecommendationService` loads the model and mappings on instantiation (from disk or Django cache, with a 1-hour TTL). Exposes two methods:

- `get_recommendations(user_id, purchase_history, top_k=5)` — takes a list of product IDs, runs inference, fetches the corresponding `Product` objects from the database (filtering for `availability='available'`), and returns a list of `{'product': Product, 'score': float, 'confidence': str}` dicts.
- `get_recommendations_async(...)` — same but serialises the result to a JSON-safe list for AJAX responses.

Errors are caught and logged; the service returns an empty list rather than raising if the model is unavailable.

> **Note:** This is an earlier version of the recommendation service that runs the model in-process within Django. The production system replaced this with `ml/recommendation/sigmoid_service.py`, which delegates inference to the separate `ml-service` FastAPI container over HTTP.

---

## How It Relates to the Wider System

```
Database (paid orders)
        ↓
model.py: get_user_sequences()
        ↓
model.py: build_training_data()  →  sliding window sequences
        ↓
model.py: train_model()          →  recommendation_model.keras + product_mappings.pkl
        ↓
service.py: RecommendationService →  product recommendations for a given user
```

In production the trained `.keras` and `.pkl` files are uploaded to the running `ml-service` via the admin upload page (see Task 3), which hot-reloads them without a restart.
