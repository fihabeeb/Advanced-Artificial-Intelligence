import cv2
import os
import numpy as np
import pandas as pd
import random

# ---- SETTINGS ----
DATASET_PATH   = r"C:\Users\fihab\Documents\Fruit And Vegetable Diseases Dataset"
OUTPUT_CSV     = "labels.csv"
SAMPLE_SIZE    = 16
GRID_COLS      = 4
THUMB_SIZE     = (200, 200)

SMALL_THRESHOLD  = 8000
LARGE_THRESHOLD  = 30000

# ---- HELPERS ----

def get_dominant_colour(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colours = {
        "red":    [((0,50,50),  (10,255,255)), ((170,50,50), (180,255,255))],
        "yellow": [((20,50,50), (35,255,255))],
        "green":  [((35,50,50), (85,255,255))],
        "orange": [((10,50,50), (20,255,255))],
        "brown":  [((10,20,20), (20,100,100))],
        "purple": [((130,50,50),(155,255,255))],
    }
    best_colour, best_count = "unknown", 0
    for colour_name, ranges in colours.items():
        mask = None
        for lower, upper in ranges:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        count = cv2.countNonZero(mask)
        if count > best_count:
            best_count, best_colour = count, colour_name
    return best_colour

def get_size(image):
    h, w = image.shape[:2]
    area = h * w

    if area < 50176:      # bottom 25%
        return "small"
    elif area > 185760:   # top 25%
        return "large"
    else:
        return "medium"


def draw_label(image, fruit, condition, colour, size):
    img = cv2.resize(image, THUMB_SIZE)
    cv2.rectangle(img, (0, THUMB_SIZE[1] - 65), (THUMB_SIZE[0], THUMB_SIZE[1]), (0, 0, 0), -1)
    cond_colour = (0, 200, 0) if condition.lower() == "healthy" else (0, 0, 220)
    cv2.putText(img, f"{fruit}",            (5, THUMB_SIZE[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.putText(img, f"{condition}",        (5, THUMB_SIZE[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cond_colour,   1)
    cv2.putText(img, f"col: {colour}",      (5, THUMB_SIZE[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (255,255,255), 1)
    cv2.putText(img, f"size: {size}",       (5, THUMB_SIZE[1] - 5),  cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (255,255,255), 1)
    return img


def build_grid(thumbnails):
    rows = []
    for i in range(0, len(thumbnails), GRID_COLS):
        row_imgs = thumbnails[i:i + GRID_COLS]
        while len(row_imgs) < GRID_COLS:
            row_imgs.append(np.zeros((THUMB_SIZE[1], THUMB_SIZE[0], 3), dtype=np.uint8))
        rows.append(np.hstack(row_imgs))
    return np.vstack(rows)


# ---- MAIN LOOP ----

records = []

for folder_name in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder_name)

    # Skip anything that isn't a folder
    if not os.path.isdir(folder_path):
        continue

    # Skip folders that don't follow the "Fruit__Condition" format
    if "__" not in folder_name:
        print(f"Skipping unrecognised folder: {folder_name}")
        continue

    parts     = folder_name.split("__")
    fruit     = parts[0]   # e.g. "Apple"
    condition = parts[1]   # e.g. "Healthy"

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder_path, filename)
        image    = cv2.imread(img_path)
        if image is None:
            print(f"Could not read: {img_path}")
            continue

        colour = get_dominant_colour(image)
        size   = get_size(image)

        records.append({
            "image_path": img_path,
            "fruit":      fruit,
            "condition":  condition,
            "colour":     colour,
            "size":       size,
            "_image":     image
        })
        print(f"[{fruit} | {condition}] {filename} → colour: {colour}, size: {size}")

# ---- SAVE CSV ----
df_csv = pd.DataFrame(records).drop(columns=["_image"])
df_csv.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone! Labels saved to {OUTPUT_CSV}")

# ---- VISUAL VERIFICATION ----
sample     = random.sample(records, min(SAMPLE_SIZE, len(records)))
thumbnails = [draw_label(r["_image"], r["fruit"], r["condition"], r["colour"], r["size"]) for r in sample]
grid       = build_grid(thumbnails)

print("\nShowing sample preview — press any key to close.")
cv2.imshow("Label Verification (press any key to close)", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()