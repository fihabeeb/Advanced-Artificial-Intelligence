import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
from config import IMG_SIZE, EPOCHS, BATCH_SIZE, LR

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
# Augmentation is applied only during training (Keras behaviour for these layers).
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="augmentation")

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = augment(inputs)

# --- Shared CNN Backbone ---
x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2, 2)(x)

x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Conv2D(256, (3,3), activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2,2)(x)

x = layers.GlobalAveragePooling2D()(x)
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss={
        "condition": "sparse_categorical_crossentropy",
        "colour":    "sparse_categorical_crossentropy",
        "size":      "sparse_categorical_crossentropy"
    },
    loss_weights={
        "condition": 2.0,
        "colour":    0.5,
        "size":      0.5,
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
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
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