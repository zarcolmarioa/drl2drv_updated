"""
02_select_images.py  (pipeline step 02)

Builds the list of images the rest of the pipeline uses.

Reads the per-category content CSVs written by
01_measure_image_content.py (unchanged from the old pipeline; its
outputs are reused, not re-run), applies the content filter and the
category exclusions, checks that every kept file is still on disk, and
writes image_index.csv. Every later step reads this file and keys its
own outputs by `image_id`, so all outputs stay aligned by construction.

What changed from the old Script A (new_encode_and_save.py, step 1)
------------------------------------------------------------------
  * Selection is now its own step. Encoding (03) and features (04) read
    the same index instead of each re-deriving it.
  * Each image gets a stable `image_id` (row number in a deterministic
    order: category order from the Places365 file, then filename) and
    an `image_key` ("<category>/<filename>"). The old pipeline resumed
    by counting rows, which would silently misalign rows if the
    selection ever changed (review item A7).
  * Unreadable images are dropped here, with a count, instead of
    becoming blank placeholder embeddings later (review item A8). The
    content step already decoded every image: rows with width -1 (or a
    missing content profile) were unreadable.
  * Optional re-bucketing of ADE20K classes (review item C1): furniture
    and other indoor objects count as "object" (foreground) in the
    content step, which biases indoor categories toward empty rooms.
    REBUCKET_TO_STRUCTURE moves listed classes from "object" to
    "structure" using the saved per-class fractions, WITHOUT re-running
    segmentation. Empty by default, which reproduces the old selection
    exactly. Use it as a sensitivity analysis, in a separate OUTPUT_DIR.
  * Aspect ratio is recorded for every image (width / height), because
    CLIP and the feature step both look at the central square.
  * Optional comparison with the old image_index.csv, so you can
    confirm the new selection reproduces the old one.

The filter rule (unchanged): an image is kept if ALL hold
    frac_person          <= MAX_FRAC_PERSON
    frac_animal          <= MAX_FRAC_ANIMAL
    frac_object          <= MAX_FRAC_OBJECT     (if the column exists)
    foreground_fraction  <= MAX_FOREGROUND
and its category is not in EXCLUDE_CATEGORIES.

Inputs
------
  CONTENT_CSV_DIR   per_category/<category>.csv from step 01
  CATEGORIES_CSV    categories_places365.csv (/letter/name index)

Outputs (under OUTPUT_DIR)
--------------------------
  image_index.csv         one row per kept image:
                            image_id, image_key, category, filename,
                            path, width, height, aspect,
                            frac_person, frac_animal, frac_vehicle,
                            frac_object, frac_structure, frac_nature,
                            foreground_fraction
  selection_summary.csv   per category: total, unreadable, duplicates,
                          passed filter, missing on disk, kept, status
  old_vs_new.csv          (only if OLD_IMAGE_INDEX is set and the sets
                          differ) the paths in one selection only
  selection_report.txt    human-readable summary

Usage
-----
    Edit SETTINGS, then:
        conda activate drl2drv
        python 02_select_images.py

Runs in minutes (reading ~365 CSVs); the optional header check reads
every file header and takes longer.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

# --- Inputs ---
# Per-category CSVs from step 01 (the old measure_image_content.py).
CONTENT_CSV_DIR     = "data/places_content_analysis/per_category"

# Standard Places365 category file. Each line: /letter/name  index
CATEGORIES_CSV      = "categories_places365.csv"

# --- Output ---
OUTPUT_DIR          = "data/selection"

# --- Content filter (same values as the old Script A) ---
MAX_FRAC_PERSON     = 0.10
MAX_FRAC_ANIMAL     = 0.10
MAX_FRAC_OBJECT     = 0.30
MAX_FOREGROUND      = 0.40      # person + animal + vehicle + object

# --- Categories excluded entirely (same list as the old Script A) ---
EXCLUDE_CATEGORIES  = [
    # person-dominated
    "nursing_home", "martial_arts_gym", "boxing_ring", "art_school",
    "beer_hall", "pub/indoor", "dressing_room", "orchestra_pit",
    "discotheque", "stage/outdoor", "army_base", "hospital_room",
    "kindergarden_classroom", "legislative_chamber",
    "biology_laboratory",
    # sports fields
    "baseball_field", "football_field", "soccer_field",
    # object-dominated indoor
    "ball_pit", "pizzeria", "bakery/shop", "supermarket", "bookstore",
    "archive", "toyshop",
    # borderline
    "candy_store", "delicatessen", "beer_garden", "market/outdoor",
    "market/indoor", "general_store/indoor", "butchers_shop",
    "amusement_arcade",
]

# --- Optional re-bucketing (review item C1) ---
#
# ADE20K class names to move from the "object" bucket (which counts as
# foreground) to "structure" (which does not), recomputed from the
# per-class fractions saved by step 01. Leave EMPTY to reproduce the
# old selection exactly.
#
# The saved per-class fractions only include classes covering at least
# 0.5% of the image, so re-bucketing is slightly conservative: tiny
# furniture fragments stay in "object".
#
# A reasonable furniture set for a sensitivity analysis (uncomment and
# use a different OUTPUT_DIR, e.g. data/selection_rebucketed):
REBUCKET_TO_STRUCTURE = [
    # "bed", "table", "chair", "sofa", "armchair", "desk", "cabinet",
    # "shelf", "wardrobe", "bookcase", "chest of drawers", "counter",
    # "countertop", "kitchen island", "coffee table", "swivel chair",
    # "seat", "bench", "bathtub", "sink", "toilet", "stove",
    # "refrigerator", "oven", "dishwasher", "washer", "fireplace",
    # "radiator", "rug", "cushion", "pillow", "lamp", "light",
    # "chandelier", "sconce", "mirror", "painting", "poster",
]

# --- File check ---
#
# "exists"  check that each kept file is on disk (fast)
# "header"  also open each file and read its size from the header,
#           and flag files whose size no longer matches the content CSV
#           (slower: reads every header; catches replaced files)
# "none"    trust the content CSVs
FILE_CHECK          = "exists"
FILE_CHECK_THREADS  = 16

# --- Optional comparison with the old selection ---
# Path to the old image_index.csv (from new_encode_and_save.py), or
# None to skip. The comparison is by file path.
OLD_IMAGE_INDEX     = "data/encoded_all_categories_un2clip_v2_full/image_index.csv"

# --- Test mode ---
# Keep at most this many images per category (random, fixed seed), for
# a quick end-to-end test of steps 03-06. None = keep all (full run).
# Use a separate OUTPUT_DIR for test selections.
TEST_N_PER_CATEGORY = None
RANDOM_SEED         = 42


# ============================================================
# Columns
# ============================================================

FRACTION_COLS = ["frac_person", "frac_animal", "frac_vehicle",
                 "frac_object", "frac_structure", "frac_nature",
                 "foreground_fraction"]
READ_COLS = ["path", "category", "filename", "width", "height"] \
    + FRACTION_COLS


# ============================================================
# Categories
# ============================================================

def load_categories(csv_path):
    """Category names in the order of the Places365 file, without the
    leading /letter/ prefix (e.g. "balcony/interior")."""
    cats = []
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path = line.rsplit(None, 1)[0].strip()
            seg = path.lstrip("/").split("/", 1)
            cats.append(seg[1] if len(seg) == 2 else seg[0])
    return cats


# ============================================================
# Re-bucketing
# ============================================================

def rebucket(df, classes):
    """Move the listed ADE20K classes from object to structure.

    Uses the `class_fractions` JSON column (classes >= 0.5% only).
    Returns df with frac_object, frac_structure and foreground_fraction
    recomputed, and the amount moved in `frac_rebucketed`.
    """
    moved = np.zeros(len(df))
    wanted = set(classes)
    for i, js in enumerate(df["class_fractions"].fillna("{}")):
        try:
            d = json.loads(js)
        except (TypeError, ValueError):
            continue
        moved[i] = sum(v for k, v in d.items() if k in wanted)
    df = df.copy()
    df["frac_rebucketed"] = moved
    df["frac_object"] = (df["frac_object"] - moved).clip(lower=0)
    df["frac_structure"] = df["frac_structure"] + moved
    df["foreground_fraction"] = (df["foreground_fraction"]
                                 - moved).clip(lower=0)
    return df


# ============================================================
# Selection
# ============================================================

def select_category(cat, rng):
    """Filter one category. Returns (kept DataFrame or None, summary)."""
    row = {"category": cat, "total": 0, "unreadable": 0,
           "duplicates": 0, "passed_filter": 0, "missing_on_disk": 0,
           "size_mismatch": 0, "kept": 0, "status": ""}

    if cat in EXCLUDE_CATEGORIES:
        row["status"] = "excluded"
        return None, row

    csv_path = os.path.join(CONTENT_CSV_DIR,
                            cat.replace("/", "_") + ".csv")
    if not os.path.exists(csv_path):
        row["status"] = "no_csv"
        return None, row

    cols = READ_COLS + (["class_fractions"] if REBUCKET_TO_STRUCTURE
                        else [])
    df = pd.read_csv(csv_path, usecols=lambda c: c in cols)
    row["total"] = len(df)

    # Step 01 appends rows across resumed runs; drop repeats.
    n0 = len(df)
    df = df.drop_duplicates("filename", keep="first")
    row["duplicates"] = n0 - len(df)

    # Unreadable in step 01: width -1 or no content profile.
    bad = (df["width"] <= 0) | df["foreground_fraction"].isna()
    row["unreadable"] = int(bad.sum())
    df = df[~bad]

    if REBUCKET_TO_STRUCTURE:
        df = rebucket(df, REBUCKET_TO_STRUCTURE)

    mask = ((df["frac_person"] <= MAX_FRAC_PERSON)
            & (df["frac_animal"] <= MAX_FRAC_ANIMAL)
            & (df["foreground_fraction"] <= MAX_FOREGROUND))
    if "frac_object" in df.columns and MAX_FRAC_OBJECT is not None:
        mask &= df["frac_object"] <= MAX_FRAC_OBJECT
    df = df[mask]
    row["passed_filter"] = len(df)

    if TEST_N_PER_CATEGORY is not None and len(df) > TEST_N_PER_CATEGORY:
        df = df.sample(n=TEST_N_PER_CATEGORY, random_state=rng)

    df = df.sort_values("filename").copy()
    df["category"] = cat
    row["kept"] = len(df)
    row["status"] = "ok"
    return df, row


def check_files(df):
    """Existence (and optionally header size) check, in threads."""
    paths = df["path"].tolist()

    def one(i):
        p = paths[i]
        if not os.path.isfile(p):
            return i, False, False
        if FILE_CHECK != "header":
            return i, True, True
        try:
            with Image.open(p) as im:
                w, h = im.size
            same = (w == df["width"].iat[i]) and (h == df["height"].iat[i])
            return i, True, same
        except Exception:
            return i, False, False

    exists = np.ones(len(paths), dtype=bool)
    same = np.ones(len(paths), dtype=bool)
    with ThreadPoolExecutor(max_workers=FILE_CHECK_THREADS) as ex:
        for i, e, s in ex.map(one, range(len(paths)), chunksize=256):
            exists[i], same[i] = e, s
    return exists, same


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("02_select_images.py")
    print("=" * 70)
    print(f"Content CSVs : {CONTENT_CSV_DIR}")
    print(f"Output       : {OUTPUT_DIR}")
    print(f"Filter       : person<={MAX_FRAC_PERSON} animal<="
          f"{MAX_FRAC_ANIMAL} object<={MAX_FRAC_OBJECT} "
          f"foreground<={MAX_FOREGROUND}")
    print(f"Re-bucketed  : {len(REBUCKET_TO_STRUCTURE)} classes")
    print(f"File check   : {FILE_CHECK}")
    print(f"Test mode    : {TEST_N_PER_CATEGORY}")

    cats = load_categories(CATEGORIES_CSV)
    print(f"\n{len(cats)} categories in {CATEGORIES_CSV}")
    unknown_excl = sorted(set(EXCLUDE_CATEGORIES) - set(cats))
    if unknown_excl:
        print(f"  WARNING: excluded names not in the category file: "
              f"{unknown_excl}")

    rng = np.random.RandomState(RANDOM_SEED)
    kept, summary = [], []
    for cat in cats:
        df, row = select_category(cat, rng)
        summary.append(row)
        if df is not None and len(df):
            kept.append(df)
    if not kept:
        raise RuntimeError("No images selected. Check CONTENT_CSV_DIR.")
    idx = pd.concat(kept, ignore_index=True)
    summ = pd.DataFrame(summary)
    print(f"Passed the filter: {len(idx):,} images")

    # --- File check ---
    if FILE_CHECK in ("exists", "header"):
        print(f"Checking files ({FILE_CHECK})...")
        exists, same = check_files(idx)
        miss = ~exists
        mism = exists & ~same
        for cat, g in idx[miss].groupby("category"):
            summ.loc[summ.category == cat, "missing_on_disk"] = len(g)
        for cat, g in idx[mism].groupby("category"):
            summ.loc[summ.category == cat, "size_mismatch"] = len(g)
        if miss.any():
            idx[miss][["category", "path"]].to_csv(
                os.path.join(OUTPUT_DIR, "missing_files.csv"), index=False)
        if mism.any():
            idx[mism][["category", "path", "width", "height"]].to_csv(
                os.path.join(OUTPUT_DIR, "size_mismatch.csv"), index=False)
        idx = idx[exists & same].reset_index(drop=True)
        kept_per_cat = idx.groupby("category").size()
        summ["kept"] = summ["category"].map(kept_per_cat).fillna(0) \
                                      .astype(int)
        print(f"  missing: {int(miss.sum())}, size mismatch: "
              f"{int(mism.sum())}")

    # --- Stable identifiers and aspect ratio ---
    idx.insert(0, "image_id", np.arange(len(idx), dtype=np.int64))
    idx.insert(1, "image_key", idx["category"] + "/" + idx["filename"])
    if idx["image_key"].duplicated().any():
        raise RuntimeError("Duplicate image_key values; check the "
                           "content CSVs.")
    idx["aspect"] = idx["width"] / idx["height"]

    cols = (["image_id", "image_key", "category", "filename", "path",
             "width", "height", "aspect"] + FRACTION_COLS
            + (["frac_rebucketed"] if REBUCKET_TO_STRUCTURE else []))
    idx = idx[cols]
    idx.to_csv(os.path.join(OUTPUT_DIR, "image_index.csv"), index=False)
    summ.to_csv(os.path.join(OUTPUT_DIR, "selection_summary.csv"),
                index=False)

    # --- Comparison with the old selection ---
    cmp_lines = []
    if OLD_IMAGE_INDEX and os.path.exists(OLD_IMAGE_INDEX):
        old = set(pd.read_csv(OLD_IMAGE_INDEX, usecols=["path"],
                              dtype=str)["path"])
        new = set(idx["path"])
        only_old, only_new = old - new, new - old
        cmp_lines = [
            f"  old selection : {len(old):,}",
            f"  new selection : {len(new):,}",
            f"  in both       : {len(old & new):,}",
            f"  only old      : {len(only_old):,}",
            f"  only new      : {len(only_new):,}",
            "  -> identical sets" if not only_old and not only_new
            else "  -> sets differ; see old_vs_new.csv",
        ]
        if only_old or only_new:
            pd.DataFrame(
                [("only_old", p) for p in sorted(only_old)]
                + [("only_new", p) for p in sorted(only_new)],
                columns=["where", "path"]).to_csv(
                os.path.join(OUTPUT_DIR, "old_vs_new.csv"), index=False)
    elif OLD_IMAGE_INDEX:
        cmp_lines = [f"  (old index not found: {OLD_IMAGE_INDEX})"]

    # --- Report ---
    a = idx["aspect"]
    lines = [
        "02_select_images.py -- summary",
        "=" * 60,
        f"Date            : {time.strftime('%Y-%m-%d %H:%M')}",
        f"Content CSVs    : {CONTENT_CSV_DIR}",
        f"Filter          : person<={MAX_FRAC_PERSON} animal<="
        f"{MAX_FRAC_ANIMAL} object<={MAX_FRAC_OBJECT} "
        f"foreground<={MAX_FOREGROUND}",
        f"Re-bucketed     : {REBUCKET_TO_STRUCTURE or 'none'}",
        f"File check      : {FILE_CHECK}",
        f"Test mode       : {TEST_N_PER_CATEGORY}",
        "",
        "Categories",
        f"  in file       : {len(cats)}",
        f"  excluded      : {int((summ.status == 'excluded').sum())}",
        f"  no CSV        : {int((summ.status == 'no_csv').sum())}",
        f"  used          : {idx['category'].nunique()}",
        "",
        "Images",
        f"  in content CSVs  : {int(summ.total.sum()):,}",
        f"  duplicate rows   : {int(summ.duplicates.sum()):,}",
        f"  unreadable       : {int(summ.unreadable.sum()):,}",
        f"  passed filter    : {int(summ.passed_filter.sum()):,}",
        f"  missing on disk  : {int(summ.missing_on_disk.sum()):,}",
        f"  size mismatch    : {int(summ.size_mismatch.sum()):,}",
        f"  KEPT             : {len(idx):,}",
        "",
        "Aspect ratio (width / height) of kept images",
        f"  square (0.99-1.01) : {((a > 0.99) & (a < 1.01)).mean():.1%}",
        f"  landscape (>1.01)  : {(a >= 1.01).mean():.1%}",
        f"  portrait (<0.99)   : {(a <= 0.99).mean():.1%}",
        f"  median             : {a.median():.3f}",
        f"  5th-95th pct       : {a.quantile(.05):.3f} - "
        f"{a.quantile(.95):.3f}",
        "  (CLIP and the feature step use the central square; non-square",
        "   images lose their edges, which is expected.)",
    ]
    if cmp_lines:
        lines += ["", "Comparison with the old selection"] + cmp_lines
    lines += ["", f"Elapsed: {time.time() - t0:.0f}s"]
    report = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "selection_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)


if __name__ == "__main__":
    main()
