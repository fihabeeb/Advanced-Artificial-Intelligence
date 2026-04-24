import cv2
import os
import numpy as np
import pandas as pd
import random
import logging
from sklearn.cluster import KMeans
from config import (
    DATASET_PATH, OUTPUT_CSV,
    SAMPLE_SIZE, GRID_COLS, THUMB_SIZE,
    SMALL_THRESHOLD, LARGE_THRESHOLD,
)

# ---- LOGGING ----
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),           # WARNING+ to console
        logging.FileHandler("labeling.log", encoding="utf-8"),  # DEBUG+ to file
    ],
)
logging.getLogger().handlers[0].setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ---- HELPERS ----

def get_foreground_mask(image, corner_size=10, bg_tol=30):
    """Return a binary mask (255 = foreground) by excluding pixels close to the background.

    Background colour is estimated from the four corners, which are almost always
    solid background in fruit/vegetable dataset images.
    """
    h, w = image.shape[:2]
    corners = np.vstack([
        image[:corner_size,  :corner_size ].reshape(-1, 3),
        image[:corner_size,  w-corner_size:].reshape(-1, 3),
        image[h-corner_size:, :corner_size ].reshape(-1, 3),
        image[h-corner_size:, w-corner_size:].reshape(-1, 3),
    ])
    bg_color = np.median(corners, axis=0)

    # Per-pixel max channel distance from the estimated background colour.
    diff = np.abs(image.astype(np.int32) - bg_color.astype(np.int32))
    dist = np.max(diff, axis=2).astype(np.uint8)
    _, mask = cv2.threshold(dist, bg_tol, 255, cv2.THRESH_BINARY)

    # Remove speckle noise and fill small holes inside the fruit.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


_COLOUR_RANGES = {
    "red":    [((0, 70, 50),    (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
    # Orange ends at H=17 so banana yellow (H≈18-28) is not pulled into orange.
    "orange": [((10, 100, 100), (17, 255, 255))],
    # Yellow ends at H=29 so yellow-green grapes are not caught as yellow.
    "yellow": [((18, 60, 100),  (29, 255, 255))],
    # Green starts at H=30 to capture yellow-green grapes and limes.
    "green":  [((30, 40, 40),   (85, 255, 255))],
    "brown":  [((5, 30, 20),    (22, 150, 150))],
    "purple": [((125, 40, 40),  (155, 255, 255))],
    "white":  [((0, 0, 180),    (180, 40, 255))],
}


def bgr_to_colour_name(bgr_pixel):
    """Map a single BGR pixel value to a colour name via the shared HSV ranges."""
    pixel = np.uint8([[bgr_pixel]])
    h, s, v = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    for name, ranges in _COLOUR_RANGES.items():
        for (hl, sl, vl), (hu, su, vu) in ranges:
            if hl <= h <= hu and sl <= s <= su and vl <= v <= vu:
                return name
    return "unknown"


def _hsv_range_vote(image, fg_mask):
    """Fallback: count foreground pixels per HSV range and return the winner."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    best_colour, best_count = "unknown", 0
    for colour_name, ranges in _COLOUR_RANGES.items():
        mask = None
        for lower, upper in ranges:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        count = cv2.countNonZero(cv2.bitwise_and(mask, fg_mask))
        if count > best_count:
            best_count, best_colour = count, colour_name
    return best_colour


def get_dominant_colour(image, n_clusters=3, min_cluster_frac=0.10):
    """Return the dominant foreground colour using k-means on foreground pixels.

    Selects the most saturated cluster that covers at least min_cluster_frac of
    foreground pixels rather than the largest cluster. This prevents white
    background bleed or specular highlights — which have near-zero saturation —
    from overriding the fruit's actual colour. Falls back to HSV-range voting
    when there are too few foreground pixels to cluster.
    """
    fg_mask = get_foreground_mask(image)
    pixels = image[fg_mask == 255].reshape(-1, 3).astype(np.float32)

    if len(pixels) < n_clusters:
        return _hsv_range_vote(image, fg_mask)

    km = KMeans(n_clusters=n_clusters, n_init=3, random_state=0)
    km.fit(pixels)
    counts = np.bincount(km.labels_)

    # Convert all centroids to HSV to read their saturation.
    centroids_bgr = km.cluster_centers_.astype(np.uint8)
    centroids_hsv = cv2.cvtColor(
        centroids_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)

    # Rank clusters by saturation (descending). Prefer the most saturated
    # cluster that is large enough — this is the fruit's actual colour.
    min_count = max(1, int(min_cluster_frac * len(pixels)))
    for idx in np.argsort(centroids_hsv[:, 1])[::-1]:
        if counts[idx] >= min_count:
            name = bgr_to_colour_name(centroids_bgr[idx])
            if name != "unknown":
                return name

    # All coloured clusters were too small or unknown — fall back to voting.
    return _hsv_range_vote(image, fg_mask)


def get_size(image):
    # Use foreground pixel area rather than raw image dimensions so that
    # the threshold is independent of how large the source photos happen to be.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    area = cv2.countNonZero(thresh)
    if area < SMALL_THRESHOLD:
        return "small"
    elif area > LARGE_THRESHOLD:
        return "large"
    return "medium"


def draw_label(image, fruit, condition, colour, size):
    img = cv2.resize(image, THUMB_SIZE)
    cv2.rectangle(img, (0, THUMB_SIZE[1] - 65), (THUMB_SIZE[0], THUMB_SIZE[1]), (0, 0, 0), -1)
    cond_colour = (0, 200, 0) if condition.lower() == "healthy" else (0, 0, 220)
    cv2.putText(img, f"{fruit}",        (5, THUMB_SIZE[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(img, f"{condition}",    (5, THUMB_SIZE[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cond_colour,     1)
    cv2.putText(img, f"col: {colour}", (5, THUMB_SIZE[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (255, 255, 255), 1)
    cv2.putText(img, f"size: {size}",  (5, THUMB_SIZE[1] - 5),  cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (255, 255, 255), 1)
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

records      = []
images_cache = []   # parallel to records — keeps raw images out of the dict
skipped      = 0
total_seen   = 0

for folder_name in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder_name)

    if not os.path.isdir(folder_path):
        continue

    if "__" not in folder_name:
        log.warning(f"Skipping unrecognised folder: {folder_name}")
        continue

    parts     = folder_name.split("__")
    fruit     = parts[0]
    condition = parts[1]

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        total_seen += 1
        img_path = os.path.join(folder_path, filename)
        image = cv2.imread(img_path)

        if image is None:
            log.warning(f"Could not read: {img_path}")
            skipped += 1
            continue

        colour = get_dominant_colour(image)
        size   = get_size(image)

        records.append({
            "image_path": img_path,
            "fruit":      fruit,
            "condition":  condition,
            "colour":     colour,
            "size":       size,
        })
        images_cache.append(image)
        log.debug(f"[{fruit} | {condition}] {filename} → colour: {colour}, size: {size}")

        if total_seen % 500 == 0:
            print(f"  Processed {total_seen} images...")

print(f"\nDone. Processed {len(records)} images, skipped {skipped}.")

# ---- SAVE CSV ----
pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
print(f"Labels saved to {OUTPUT_CSV}")

# ---- VISUAL VERIFICATION ----
sample_indices = random.sample(range(len(records)), min(SAMPLE_SIZE, len(records)))
thumbnails = [
    draw_label(
        images_cache[i],
        records[i]["fruit"],
        records[i]["condition"],
        records[i]["colour"],
        records[i]["size"],
    )
    for i in sample_indices
]
grid = build_grid(thumbnails)

cv2.imwrite("label_verification_grid.png", grid)
print("Sample preview saved to label_verification_grid.png")
