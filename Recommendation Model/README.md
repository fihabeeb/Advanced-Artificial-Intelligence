# Recommendation Model

LSTM + Attention model for product recommendation, with recency boosting.

## Model Architecture

```
Input:
  - product_ids: (batch, 15 orders, 5 items) - product sequences
  - temporal: (batch, 15 orders, 5) - temporal features (day/month encoding)
  - user_cluster: (batch, 1) - user cluster embedding

Architecture:
  1. Embed product IDs (vocab_size=200, embedding_dim=32)
  2. Concatenate with temporal features
  3. LSTM layer (64 units)
  4. Dot-product attention over sequence
  5. User cluster embedding (8 dims)
  6. Dense layers: 64 -> 32 -> num_classes
  7. Output: logits + attention weights
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| MAX_ORDER_HISTORY | 15 |
| MAX_ITEMS_PER_ORDER | 5 |
| VOCAB_SIZE | 200 |
| NUM_CLASSES | 160 |
| LSTM_UNITS | 64 |
| DROPOUT | 0.2 |
| BATCH_SIZE | 64 |
| EPOCHS | 30 |
| LEARNING_RATE | 1e-3 |

## Files

- `LSTMAttention.py` - Model training script
- `LSTMAttentionService.py` - Inference service with XAI features
- `LSTM(Baseline)/` - Baseline LSTM for comparison

## Usage

### Training
```python
from LSTMAttention import train
model = train()
```

### Inference
```python
from LSTMAttentionService import LSTMAttentionService

service = LSTMAttentionService.get_instance()
service.load_model()

# Get recommendations with explanations
result = service.get_predictions_with_explanation(user_id=123, top_k=5)
# Returns: recommendations, attention_weights, order_details, num_orders

# Get attention weights for XAI
attn = service.get_attention_weights(user_id=123, purchase_history_with_timestamps)
```

## Features

- **Dot-product attention** - focuses on relevant past orders
- **Temporal encoding** - day of week + month (cyclic sin/cos)
- **User clustering** - K-means for personalization
- **Recency boosting** - exponential decay (0.7^n) for recent purchases
- **Gradient-based saliency** - XAI for model explanations
- **Popular fallback** - recommendations for cold-start users

## Evaluation Metrics

- Hit Rate @1, @3, @5, @10
- Top-K Accuracy