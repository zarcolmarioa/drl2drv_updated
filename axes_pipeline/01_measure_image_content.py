"""
measure_image_content.py

Measurement-only pass over Places365 categories.
For every image, runs ADE20K semantic segmentation and records
how much of the frame is occupied by broad content buckets
(person, animal, vehicle, object, structure, nature). Makes NO
filtering decisions — a separate analysis script reads this
output and decides thresholds.

Purpose: the substantive analysis needs images where people,
animals, and objects do not dominate the frame (derealization
is about how the *environment* looks). This script produces the
per-image content profile used for that filtering, and also a
semantic distribution of scene content.

Category source — two options (set CATEGORY_SOURCE below):
  'csv'  : load all 365 categories from the standard Places365
           categories file (categories_places365.csv). Each line
           has the format:  /letter/category_name  index
           The letter prefix is stripped automatically.
           Handles sub-categories with slashes (e.g.
           balcony/interior → root/b/balcony/interior).
  'yaml' : load from a hand-curated YAML file (original behaviour,
           used for the everyday-only subset).

What it saves per image (one row):
  - path, category, filename
  - width, height
  - foreground_fraction          (person+animal+vehicle+object)
  - frac_person, frac_animal, frac_vehicle, frac_object,
    frac_structure, frac_nature  (broad-bucket pixel fractions)
  - top5_classes                 (JSON: dominant ADE20K classes)
  - class_fractions              (JSON: full sparse per-class
                                  fractions, non-zero only)

Output layout (under OUTPUT_ROOT):
  per_category/<category>.csv    (one CSV per category; natural
                                  resume boundary)
  _run_meta.json                 (model, settings, timing)

Resume: on restart, any image already present in a category's
CSV is skipped. Safe to stop and restart at any time.

Model: switch MODEL_KEY between 'segformer' and 'mask2former'.
Both output 150-class ADE20K labels, so the bucketing is
identical; only speed/accuracy differ.

Usage:
    Edit the SETTINGS section, then:
        python measure_image_content.py

Dependencies:
    pip install torch transformers pillow numpy pandas pyyaml tqdm
"""

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch
from PIL import Image
from tqdm import tqdm


# ============================================================
# SETTINGS
# ============================================================

# --- Inputs ---
IMAGES_ROOT     = "/home/mario/Documents/datasets/data_large"

# Category source — choose one:
#   'csv'  : all 365 Places365 categories from the standard file.
#            Each line format: /letter/category_name  index
#   'yaml' : hand-curated everyday subset (original behaviour).
CATEGORY_SOURCE = "csv"

# Path to the standard Places365 categories file (used when
# CATEGORY_SOURCE = 'csv'). Each line: /letter/category_name  index
CATEGORIES_CSV  = "categories_places365.csv"

# Path to hand-curated YAML (used when CATEGORY_SOURCE = 'yaml').
CATEGORIES_YAML = "places_categories_everyday.yaml"

# --- Output ---
OUTPUT_ROOT     = "data/places_content_analysis"

# --- Model selection ---
# 'segformer'   : lighter/faster (~10-20 img/s), good for huge runs
# 'mask2former' : heavier/slower (~3-8 img/s), higher accuracy
# Both produce 150-class ADE20K semantic labels.
MODEL_KEY       = "segformer"
MODEL_IDS = {
    "segformer":   "nvidia/segformer-b3-finetuned-ade-512-512",
    "mask2former": "facebook/mask2former-swin-large-ade-semantic",
}

# --- Test mode ---
# When True, process only TEST_N_PER_CLASS images per category.
# No separate output path is used: rows are written to the same
# per-category CSVs, and a normal (full) run later will simply
# resume and process the remaining images (the test images are
# skipped because they're already recorded).
TEST_MODE         = False
TEST_N_PER_CLASS  = 10

# --- Image formats ---
IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".webp"}

# --- Inference batch size (images per forward pass) ---
# SegFormer handles larger batches; Mask2Former is heavier.
BATCH_SIZE = {"segformer": 16, "mask2former": 4}

# --- Number of dominant classes to record per image ---
TOPK_CLASSES = 5

# --- How often (in images) to flush a category's rows to disk ---
SAVE_EVERY = 200

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# ADE20K 150-class -> broad bucket mapping
# ============================================================
# ADE20K class indices are 0..149 (the order used by the HF
# models below). The names are the standard ADE20K-150 labels.
# We map each class to one of six broad buckets:
#   person, animal, vehicle, object, structure, nature
# foreground = person + animal + vehicle + object.
#
# This mapping is explicit and editable: adjust which classes
# count as foreground by moving names between buckets. Any class
# not listed defaults to 'object' (conservative: counts toward
# foreground), and a warning is printed so you can re-bucket it.

ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road",
    "bed", "windowpane", "grass", "cabinet", "sidewalk", "person",
    "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea",
    "mirror", "rug", "field", "armchair", "seat", "fence", "desk",
    "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers",
    "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river",
    "bridge", "bookcase", "blind", "coffee table", "toilet", "flower",
    "book", "hill", "bench", "countertop", "stove", "palm",
    "kitchen island", "computer", "swivel chair", "boat", "bar",
    "arcade machine", "hovel", "bus", "towel", "light", "truck",
    "tower", "chandelier", "awning", "streetlight", "booth",
    "television receiver", "airplane", "dirt track", "apparel",
    "pole", "land", "bannister", "escalator", "ottoman", "bottle",
    "buffet", "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything",
    "swimming pool", "stool", "barrel", "basket", "waterfall",
    "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal",
    "bicycle", "lake", "dishwasher", "screen", "blanket",
    "sculpture", "hood", "sconce", "vase", "traffic light", "tray",
    "ashcan", "fan", "pier", "crt screen", "plate", "monitor",
    "bulletin board", "shower", "radiator", "glass", "clock", "flag",
]

# Buckets keyed by ADE20K class name.
BUCKET_MAP = {
    # --- person ---
    "person": "person",
    # --- animal ---
    "animal": "animal",
    # --- vehicle ---
    "car": "vehicle", "boat": "vehicle", "bus": "vehicle",
    "truck": "vehicle", "airplane": "vehicle", "van": "vehicle",
    "ship": "vehicle", "minibike": "vehicle", "bicycle": "vehicle",
    # --- nature (stuff: natural scene surfaces) ---
    "sky": "nature", "tree": "nature", "grass": "nature",
    "earth": "nature", "mountain": "nature", "plant": "nature",
    "water": "nature", "sea": "nature", "field": "nature",
    "rock": "nature", "sand": "nature", "river": "nature",
    "flower": "nature", "hill": "nature", "palm": "nature",
    "land": "nature", "waterfall": "nature", "lake": "nature",
    "dirt track": "nature",
    # --- structure (stuff: built scene surfaces) ---
    "wall": "structure", "building": "structure", "floor": "structure",
    "ceiling": "structure", "road": "structure", "windowpane": "structure",
    "sidewalk": "structure", "door": "structure", "house": "structure",
    "fence": "structure", "railing": "structure", "skyscraper": "structure",
    "path": "structure", "stairs": "structure", "runway": "structure",
    "stairway": "structure", "bridge": "structure", "column": "structure",
    "hovel": "structure", "tower": "structure", "awning": "structure",
    "pole": "structure", "bannister": "structure", "escalator": "structure",
    "stage": "structure", "canopy": "structure", "swimming pool": "structure",
    "tent": "structure", "step": "structure", "pier": "structure",
    "curtain": "structure", "blind": "structure", "screen door": "structure",
    "grandstand": "structure", "booth": "structure", "fountain": "structure",
    "shower": "structure",
    # everything else (furniture, appliances, small objects, etc.)
    # defaults to "object" via DEFAULT_BUCKET below.
}
DEFAULT_BUCKET = "object"
FOREGROUND_BUCKETS = {"person", "animal", "vehicle", "object"}
ALL_BUCKETS = ["person", "animal", "vehicle", "object",
               "structure", "nature"]


def build_index_to_bucket():
    """Map each ADE20K class index (0..149) to a broad bucket.
    Warn about any class that falls through to the default.
    """
    idx_to_bucket = {}
    fell_through = []
    for idx, name in enumerate(ADE20K_CLASSES):
        if name in BUCKET_MAP:
            idx_to_bucket[idx] = BUCKET_MAP[name]
        else:
            idx_to_bucket[idx] = DEFAULT_BUCKET
            fell_through.append(name)
    return idx_to_bucket, fell_through


# ============================================================
# Category / image discovery
# ============================================================

def load_categories_from_csv(csv_path):
    """Load all Places365 category names from the standard
    categories file. Each line has the format:
        /letter/category_name  index
    e.g.  /a/airfield 0
          /b/balcony/interior 5

    The letter prefix is stripped so the returned name can be
    resolved by resolve_category_folder, which adds it back from
    the first character of the name:
        airfield          -> root/a/airfield
        balcony/interior  -> root/b/balcony/interior
    """
    categories = []
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Strip trailing integer index (if present)
            parts = line.rsplit(None, 1)
            path = parts[0].strip()          # e.g.  /a/airfield
            # Remove leading slash, then split off the letter prefix
            segments = path.lstrip("/").split("/", 1)
            # segments[0] = single letter; segments[1] = category name
            if len(segments) == 2:
                categories.append(segments[1])
            elif len(segments) == 1:
                # Unexpected format — keep as-is
                categories.append(segments[0])
    return categories


def load_categories(yaml_path):
    """Load category names from a hand-curated YAML file
    (original behaviour for the everyday subset)."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return [c.strip() for c in data.get("categories", []) if c and c.strip()]


def resolve_category_folder(images_root, category):
    """Resolve a category name to its folder on disk.

    The large Places365 dataset nests every category under a
    first-letter directory, e.g.:
        <root>/w/waterfall
        <root>/l/lake/natural        (nested sub-category)
        <root>/b/balcony/interior

    The category name from the YAML may itself be nested with a
    slash (e.g. "lake/natural"). The first letter is taken from
    the FIRST path segment of the category.

    We try, in order:
      1. <root>/<first_letter>/<category>           (large layout)
      2. <root>/<first_letter>/<category_underscored>
      3. <root>/<category>                          (flat layout,
      4. <root>/<category_underscored>               for the old
                                                      small subset)
    so the script works on both the large dataset and the earlier
    flat subset without changes.
    """
    root = Path(images_root)
    first_segment = category.split("/", 1)[0]
    letter = first_segment[0].lower() if first_segment else ""
    underscored = category.replace("/", "_")
    candidates = [
        root / letter / category,
        root / letter / underscored,
        root / category,
        root / underscored,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def list_images(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# ============================================================
# Model loading
# ============================================================

def load_segmenter(model_key, device):
    """Load the chosen ADE20K semantic segmentation model and
    its image processor. Returns (model, processor, kind) where
    kind is 'segformer' or 'mask2former' (they need slightly
    different post-processing calls).
    """
    model_id = MODEL_IDS[model_key]
    if model_key == "segformer":
        from transformers import (
            SegformerForSemanticSegmentation, SegformerImageProcessor,
        )
        model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
        proc = SegformerImageProcessor.from_pretrained(model_id)
    elif model_key == "mask2former":
        from transformers import (
            Mask2FormerForUniversalSegmentation, AutoImageProcessor,
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(device)
        proc = AutoImageProcessor.from_pretrained(model_id)
    else:
        raise ValueError(f"Unknown MODEL_KEY: {model_key}")
    model.eval()
    return model, proc, model_key


def segment_batch(pil_images, model, processor, kind, device):
    """Run segmentation on a batch of PIL images. Returns a list
    of 2D numpy arrays (H x W) of ADE20K class indices, one per
    image, each resized to its original resolution.
    """
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = [(im.height, im.width) for im in pil_images]
    if kind == "segformer":
        # post_process_semantic_segmentation handles upsampling
        # to the target sizes.
        maps = processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )
    else:  # mask2former
        maps = processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )
    return [m.cpu().numpy().astype(np.int16) for m in maps]


# ============================================================
# Per-image content measurement
# ============================================================

def measure_content(seg_map, idx_to_bucket):
    """Given a (H x W) ADE20K class-index map, compute:
      - per-bucket pixel fraction
      - foreground fraction
      - top-K dominant ADE20K classes with fractions
      - full sparse per-class fractions
    """
    total = seg_map.size
    # Per-class pixel counts.
    classes, counts = np.unique(seg_map, return_counts=True)
    class_fractions = {}
    bucket_fractions = {b: 0.0 for b in ALL_BUCKETS}
    for cls_idx, cnt in zip(classes, counts):
        frac = float(cnt) / total
        cls_idx = int(cls_idx)
        if 0 <= cls_idx < len(ADE20K_CLASSES):
            cls_name = ADE20K_CLASSES[cls_idx]
        else:
            cls_name = f"idx_{cls_idx}"
        class_fractions[cls_name] = frac
        bucket = idx_to_bucket.get(cls_idx, DEFAULT_BUCKET)
        bucket_fractions[bucket] += frac

    foreground = sum(bucket_fractions[b] for b in FOREGROUND_BUCKETS)

    # Top-K dominant classes.
    top = sorted(class_fractions.items(), key=lambda kv: -kv[1])[:TOPK_CLASSES]
    top5 = [{"class": k, "frac": round(v, 4)} for k, v in top]

    return {
        "foreground_fraction": round(foreground, 4),
        "frac_person":    round(bucket_fractions["person"], 4),
        "frac_animal":    round(bucket_fractions["animal"], 4),
        "frac_vehicle":   round(bucket_fractions["vehicle"], 4),
        "frac_object":    round(bucket_fractions["object"], 4),
        "frac_structure": round(bucket_fractions["structure"], 4),
        "frac_nature":    round(bucket_fractions["nature"], 4),
        "top5_classes":   json.dumps(top5),
        "class_fractions": json.dumps(
            {k: round(v, 4) for k, v in class_fractions.items() if v >= 0.005}
        ),
    }


# ============================================================
# Resume helper
# ============================================================

def load_done_filenames(category_csv):
    """Return the set of filenames already recorded in a
    category's CSV (for resume)."""
    if not os.path.exists(category_csv):
        return set()
    try:
        df = pd.read_csv(category_csv, usecols=["filename"], dtype=str)
        return set(df["filename"].tolist())
    except Exception:
        return set()


def append_rows(category_csv, rows):
    """Append rows to a category CSV, writing a header only if
    the file does not yet exist."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(category_csv)
    df.to_csv(category_csv, mode="a", header=header, index=False,
              encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main():
    per_cat_dir = os.path.join(OUTPUT_ROOT, "per_category")
    os.makedirs(per_cat_dir, exist_ok=True)

    print("Image content measurement (ADE20K segmentation)")
    print("=" * 70)
    print(f"Device:        {DEVICE}")
    print(f"Model:         {MODEL_KEY} ({MODEL_IDS[MODEL_KEY]})")
    print(f"Images root:   {IMAGES_ROOT}")
    print(f"Category src:  {CATEGORY_SOURCE}")
    if CATEGORY_SOURCE == "csv":
        print(f"Categories:    {CATEGORIES_CSV}")
    else:
        print(f"Categories:    {CATEGORIES_YAML}")
    print(f"Output:        {OUTPUT_ROOT}")
    print(f"Test mode:     {TEST_MODE}"
          + (f" ({TEST_N_PER_CLASS}/class)" if TEST_MODE else ""))
    print("=" * 70 + "\n")

    # --- Bucket mapping ---
    idx_to_bucket, fell_through = build_index_to_bucket()
    if fell_through:
        print(f"Note: {len(fell_through)} ADE20K classes default to "
              f"'{DEFAULT_BUCKET}' (counted as foreground):")
        print(f"  {', '.join(fell_through)}\n")

    # --- Categories ---
    if CATEGORY_SOURCE == "csv":
        categories = load_categories_from_csv(CATEGORIES_CSV)
        print(f"{len(categories)} Places365 categories (from CSV).\n")
    else:
        categories = load_categories(CATEGORIES_YAML)
        print(f"{len(categories)} everyday categories (from YAML).\n")

    # --- Model ---
    print("Loading segmentation model...")
    model, processor, kind = load_segmenter(MODEL_KEY, DEVICE)
    batch_size = BATCH_SIZE[MODEL_KEY]
    print(f"  Loaded. Batch size {batch_size}.\n")

    # --- Process each category ---
    total_done = 0
    total_time = 0.0
    missing_cats = []

    for cat in categories:
        folder = resolve_category_folder(IMAGES_ROOT, cat)
        if folder is None:
            missing_cats.append(cat)
            print(f"[{cat}] no folder on disk — skipping.")
            continue

        category_csv = os.path.join(
            per_cat_dir, cat.replace("/", "_") + ".csv"
        )
        done = load_done_filenames(category_csv)
        images = list_images(folder)

        if TEST_MODE:
            images = images[:TEST_N_PER_CLASS]

        todo = [p for p in images if p.name not in done]
        if not todo:
            print(f"[{cat}] all {len(images)} target images already done.")
            continue

        print(f"[{cat}] {len(todo)} to process "
              f"({len(done)} already done, {len(images)} target).")

        buffer = []
        t0 = time.time()
        for i in tqdm(range(0, len(todo), batch_size), desc=f"  {cat}"):
            batch_paths = todo[i:i + batch_size]
            pil_images, valid_paths = [], []
            for p in batch_paths:
                try:
                    with Image.open(p) as im:
                        pil_images.append(im.convert("RGB").copy())
                    valid_paths.append(p)
                except Exception:
                    # Unreadable image — record a row with NaNs so it
                    # isn't retried forever.
                    buffer.append({
                        "path": str(p), "category": cat, "filename": p.name,
                        "width": -1, "height": -1,
                        "foreground_fraction": np.nan,
                        "frac_person": np.nan, "frac_animal": np.nan,
                        "frac_vehicle": np.nan, "frac_object": np.nan,
                        "frac_structure": np.nan, "frac_nature": np.nan,
                        "top5_classes": "[]", "class_fractions": "{}",
                    })
            if pil_images:
                seg_maps = segment_batch(pil_images, model, processor,
                                         kind, DEVICE)
                for p, im, smap in zip(valid_paths, pil_images, seg_maps):
                    content = measure_content(smap, idx_to_bucket)
                    row = {
                        "path": str(p), "category": cat, "filename": p.name,
                        "width": im.width, "height": im.height,
                    }
                    row.update(content)
                    buffer.append(row)

            if len(buffer) >= SAVE_EVERY:
                append_rows(category_csv, buffer)
                buffer = []

        # Flush remaining.
        append_rows(category_csv, buffer)
        dt = time.time() - t0
        total_done += len(todo)
        total_time += dt
        rate = len(todo) / dt if dt > 0 else 0.0
        print(f"  done {len(todo)} in {dt:.1f}s ({rate:.1f} img/s)")

    # --- Run meta + timing projection ---
    overall_rate = total_done / total_time if total_time > 0 else 0.0
    meta = {
        "model_key": MODEL_KEY,
        "model_id": MODEL_IDS[MODEL_KEY],
        "category_source": CATEGORY_SOURCE,
        "category_file": CATEGORIES_CSV if CATEGORY_SOURCE == "csv" else CATEGORIES_YAML,
        "test_mode": TEST_MODE,
        "test_n_per_class": TEST_N_PER_CLASS if TEST_MODE else None,
        "images_processed_this_run": total_done,
        "seconds_this_run": round(total_time, 1),
        "images_per_second": round(overall_rate, 2),
        "n_categories": len(categories),
        "missing_categories": missing_cats,
    }
    with open(os.path.join(OUTPUT_ROOT, "_run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Run complete. Processed {total_done} images this run "
          f"at {overall_rate:.1f} img/s.")
    if TEST_MODE and overall_rate > 0:
        # Project full-run time assuming ~5000 images/category.
        est_total = len(categories) * 5000
        est_hours = est_total / overall_rate / 3600
        print(f"\nProjection for full run (~{est_total:,} images at "
              f"this rate): ~{est_hours:.1f} hours of compute "
              f"(plus disk I/O).")
        print("Switch TEST_MODE=False and re-run to process all images;\n"
              "the test images already recorded will be skipped.")
    print(f"\nOutputs in: {OUTPUT_ROOT}/per_category/")


if __name__ == "__main__":
    main()
