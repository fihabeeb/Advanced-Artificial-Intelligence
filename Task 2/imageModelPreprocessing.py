import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# ---- SETTINGS ----
CSV_PATH  = "labels.csv"
IMG_SIZE  = (128, 128)
TEST_SIZE = 0.1
VAL_SIZE  = 0.1

# ---- LOAD CSV ----
df = pd.read_csv(CSV_PATH)
print(f"Total images: {len(df)}\n")

# ---- LOAD & PREPROCESS IMAGES ----
images  = []
failed  = []
total   = len(df)

print("Loading images...")
for i, img_path in enumerate(df["image_path"]):

    # Progress update every 500 images
    if i % 500 == 0:
        print(f"  {i}/{total} images loaded...")

    img = cv2.imread(img_path)

    if img is None:
        print(f"  [WARNING] Could not read: {img_path}")
        failed.append(i)
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    images.append(img)

# Drop rows where image failed to load
if failed:
    print(f"\n[WARNING] {len(failed)} images failed to load and were skipped.")
    df = df.drop(index=failed).reset_index(drop=True)

images = np.array(images, dtype=np.float32)
print(f"\nDone! Images array shape: {images.shape}")
print(f"Estimated RAM usage: {images.nbytes / 1e9:.2f} GB")

# ---- ENCODE LABELS ----
encoders = {}
labels   = {}

for col in ["condition", "colour", "size"]:
    le = LabelEncoder()
    labels[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"{col} classes: {list(le.classes_)}")

# ---- SPLIT DATA ----
print("\nSplitting data...")
X_temp, X_test, \
y_cond_temp,  y_cond_test, \
y_col_temp,   y_col_test, \
y_size_temp,  y_size_test = train_test_split(
    images,
    labels["condition"],
    labels["colour"],
    labels["size"],
    test_size=TEST_SIZE,
    random_state=42
)

X_train, X_val, \
y_cond_train, y_cond_val, \
y_col_train,  y_col_val, \
y_size_train, y_size_val = train_test_split(
    X_temp,
    y_cond_temp,
    y_col_temp,
    y_size_temp,
    test_size=VAL_SIZE / (1 - TEST_SIZE),
    random_state=42
)

print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples:       {len(X_test)}")

# ---- SAVE ----
print("\nSaving files...")
np.save("X_train.npy", X_train)
np.save("X_val.npy",   X_val)
np.save("X_test.npy",  X_test)

np.save("y_cond_train.npy", y_cond_train)
np.save("y_cond_val.npy",   y_cond_val)
np.save("y_cond_test.npy",  y_cond_test)

np.save("y_col_train.npy",  y_col_train)
np.save("y_col_val.npy",    y_col_val)
np.save("y_col_test.npy",   y_col_test)

np.save("y_size_train.npy", y_size_train)
np.save("y_size_val.npy",   y_size_val)
np.save("y_size_test.npy",  y_size_test)

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("\nAll done! Preprocessing complete.")
