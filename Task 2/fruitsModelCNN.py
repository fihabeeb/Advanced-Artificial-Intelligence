import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle

# ---- LOAD DATA ----
print("Loading data...")
X_train = np.load("X_train.npy")
X_val   = np.load("X_val.npy")

y_cond_train  = np.load("y_cond_train.npy")
y_cond_val    = np.load("y_cond_val.npy")

y_col_train   = np.load("y_col_train.npy")
y_col_val     = np.load("y_col_val.npy")

y_size_train  = np.load("y_size_train.npy")
y_size_val    = np.load("y_size_val.npy")

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Number of classes per output
n_condition = len(encoders["condition"].classes_)  # 2
n_colour    = len(encoders["colour"].classes_)     # 6
n_size      = len(encoders["size"].classes_)       # 3

print(f"Condition classes: {n_condition}")
print(f"Colour classes:    {n_colour}")
print(f"Size classes:      {n_size}")

# ---- BUILD CNN ----
inputs = layers.Input(shape=(128, 128, 3))

# --- Shared CNN Backbone ---
x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Conv2D(256, (3,3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.4)(x)
shared = layers.Dense(256, activation="relu")(x)

# --- Output Heads ---
# Condition head (Healthy / Rotten)
cond_out = layers.Dense(128, activation="relu")(shared)
cond_out = layers.Dropout(0.3)(cond_out)
cond_out = layers.Dense(n_condition, activation="softmax", name="condition")(cond_out)

# Colour head
col_out = layers.Dense(128, activation="relu")(shared)
col_out = layers.Dropout(0.3)(col_out)
col_out = layers.Dense(n_colour, activation="softmax", name="colour")(col_out)

# Size head
size_out = layers.Dense(128, activation="relu")(shared)
size_out = layers.Dropout(0.3)(size_out)
size_out = layers.Dense(n_size, activation="softmax", name="size")(size_out)

# ---- COMPILE MODEL ----
model = models.Model(inputs=inputs, outputs=[cond_out, col_out, size_out])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        "condition": "sparse_categorical_crossentropy",
        "colour":    "sparse_categorical_crossentropy",
        "size":      "sparse_categorical_crossentropy"
    },
    metrics={
        "condition": "accuracy",
        "colour":    "accuracy",
        "size":      "accuracy"
    }
)

model.summary()

# ---- TRAIN ----
print("\nStarting training...")

history = model.fit(
    X_train,
    {"condition": y_cond_train, "colour": y_col_train, "size": y_size_train},
    validation_data=(
        X_val,
        {"condition": y_cond_val, "colour": y_col_val, "size": y_size_val}
    ),
    epochs=20,
    batch_size=32,
    callbacks=[
        # Stop early if validation loss stops improving
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        # Save the best model automatically
        tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
        # Reduce learning rate if stuck
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1)
    ]
)

# ---- SAVE ----
model.save("fruit_model.keras")
print("\nModel saved to fruit_model.keras")