"""
04_compute_features.py  (pipeline step 04)

Computes low-level visual features for every image in image_index.csv
(step 02), in one pass over the images, keyed by image_id.

Three feature sets
------------------
  CROP (primary)     all 18 candidate features, measured on EXACTLY the
                     region CLIP sees (the central square, side = the
                     shorter edge, with CLIP's own rounding; see
                     clip_box), resized in one step to FEATURE_SIZE x
                     FEATURE_SIZE (256). This matches the encoder
                     (step 03) and matches the experiment's stimuli once
                     those are built from the central square too.

  STRETCH (legacy)   the 11 original features, measured as the old
                     pipeline did: the WHOLE image squashed to 256 x 256
                     (aspect ratio distorted). Only for comparing with
                     the old results (review item A3). Not for new
                     analyses.

  CROP-336 (check)   the 18 features on the same central square at
                     336 x 336 (CLIP's own resolution), for a stratified
                     subsample only. Used in step 09 to show the choice
                     of 256 does not change the conclusions.

Which of the 18 crop features are used in the analyses is decided in
step 04b (04b_select_features.py), from the features' correlations with
EACH OTHER only, never with the axes.

The 18 candidate features (CROP set)
------------------------------------
Brightness (grey = OpenCV RGB->grey, Rec. 601 weights, 0-255)
  lum_mean          mean grey level
  lum_p50           median grey level
  val_mean          mean HSV value (brightest channel per pixel)
Contrast and brightness distribution
  lum_std           SD of grey (RMS / global contrast)
  lum_p05           5th percentile of grey (dark end; lifted by haze)
  lum_p95           95th percentile of grey (bright end)
  lum_skew          skewness of grey
  lum_entropy       Shannon entropy of the 256-bin grey histogram (bits)
Amount of colour
  sat_mean          mean HSV saturation (0-255)          [stimulus feature]
  sat_std           SD of HSV saturation
  colorfulness      Hasler & Suesstrunk (2003) colourfulness
  chroma_mean       mean CIELAB chroma C*ab = sqrt(a*^2 + b*^2)
Colour direction
  lab_a_mean        mean CIELAB a* (+ red, - green)
  lab_b_mean        mean CIELAB b* (+ yellow / warm, - blue / cool)
Fine detail and spatial structure
  contrast_local    SD of the Laplacian of grey           [stimulus feature]
  edge_density      share of Canny edge pixels (thresholds 100 / 200)
  spectral_slope    alpha of the radially averaged amplitude spectrum,
                    A(f) ~ 1 / f^alpha, fitted over SPECTRUM_FMIN to
                    SPECTRUM_FMAX cycles per image; larger = fewer fine
                    details relative to coarse structure (blur, haze)
Haze
  dark_channel_mean mean dark channel (He, Sun & Tang, 2009): per-pixel
                    minimum over R, G, B, then a local minimum filter;
                    0-1; rises with haze

lum_p05 is also a stimulus feature (the first stimulus version).

Removed from the old list (review item A4): contrast_global (identical
to lum_std) and sharpness (the square of contrast_local).

CIELAB uses OpenCV's float conversion, which applies the sRGB transfer
curve and D65 white (checked: mid-grey 0.5 -> L* 53.4).

Checks
------
  1. REGION      On N_CHECK_IMAGES non-square images: our region resized
                 to 336 (bicubic) is compared pixel by pixel with CLIP's
                 own view (resize the short side to 336, centre-crop).
                 The mean absolute difference must be below
                 CROP_CHECK_MAX_MAD grey levels; it is ~0 when the
                 region is exactly CLIP's. The stretched image is shown
                 for contrast.
  2. LEGACY      If the old files are given, the STRETCH features of a
                 few images are compared with the old
                 classical_features.csv. They should match (same code),
                 confirming the legacy set reproduces the old values.
  3. ALIGNMENT   At the end, FINAL_RECHECK_N random images are recomputed
                 and compared with the stored rows.

Inputs
------
  INDEX_CSV           data/selection/image_index.csv (step 02)
  OLD_IMAGE_INDEX     old image_index.csv       (optional, check 2)
  OLD_FEATURES_CSV    old classical_features.csv (optional, check 2)

Outputs (under OUTPUT_DIR)
--------------------------
  features_crop.npy / .csv          (N, 18)  primary set
  features_stretch.npy / .csv       (N, 11)  legacy set
  subsample_ids.npy                 stratified subsample (image_ids),
                                    SUBSAMPLE_PER_CATEGORY per category;
                                    also used by steps 04b and 09
  features_crop336_subsample.npy / .csv   (n_sub, 18)
  status.npy                        (N,) 0 not done, 1 ok, -1 unreadable
  feature_definitions.json          names, sets, definitions, settings
  failed_images.csv                 unreadable images (if any)
  run_meta.json, features_report.txt

The .csv files have an image_id column and are written at the end; the
.npy files are the working arrays (row i = image_id i).

Resume
------
Run the script again; it continues where it stopped, and refuses to
resume if the index or the feature settings changed.

Usage
-----
    Edit SETTINGS, then:
        conda activate drl2drv
        python 04_compute_features.py

CPU only. Decoding the images dominates the run time, so N_WORKERS
matters most; expect a run of several hours on the full corpus. Try
TEST_LIMIT = 2000 with a separate OUTPUT_DIR first.
"""

import hashlib
import json
import multiprocessing as mp
import os
import time
import warnings

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# SETTINGS
# ============================================================

# --- Inputs / output ---
INDEX_CSV           = "data/selection/image_index.csv"
OUTPUT_DIR          = "data/features"

# --- Dry run ---
# Process only the first TEST_LIMIT images (None = all). Use a separate
# OUTPUT_DIR for dry runs.
TEST_LIMIT          = None

# --- Geometry ---
# FEATURE_SIZE   side of the square the features are measured on. 256
#                keeps the corpus in the same units as the stimulus
#                measurements (512 px stimuli measured at 256).
# CHECK_SIZE     side of the extra subsample set (CLIP's resolution).
FEATURE_SIZE        = 256
CHECK_SIZE          = 336

# --- Subsample for steps 04b and 09 ---
# Stratified: this many random images per category (all of them if the
# category has fewer). ~33,000 images for 332 categories at 100.
SUBSAMPLE_PER_CATEGORY = 100

# --- Feature parameters ---
# Canny thresholds, as in the old pipeline.
CANNY_LOW           = 100
CANNY_HIGH          = 200

# Spectral slope: fit range in cycles per image. The lowest frequencies
# are dominated by the window and the image layout, the highest by
# resampling and JPEG noise (Nyquist is 128 at 256 px).
SPECTRUM_FMIN       = 8
SPECTRUM_FMAX       = 64

# Dark channel: side of the minimum filter at 256 px (He et al. used
# 15 px on ~600 px images). Scaled with image size for other sizes.
DARK_PATCH_AT_256   = 7

# --- Parallelism ---
# N_WORKERS    processes computing features. Set to the number of CPU
#              cores minus one or two.
# CHUNK        images per task sent to a worker.
N_WORKERS           = 24
CHUNK               = 64
FLUSH_EVERY_TASKS   = 100

# --- Checks ---
N_CHECK_IMAGES      = 16
CROP_CHECK_MAX_MAD  = 0.5      # grey levels (0-255); expected ~0
FINAL_RECHECK_N     = 32

# Old pipeline files for check 2 (None to skip). Row i of the old
# features file belongs to row i of the old index.
OLD_IMAGE_INDEX     = "data/encoded_all_categories_un2clip_v2_full/image_index.csv"
OLD_FEATURES_CSV    = "data/encoded_all_categories_un2clip_v2_full/classical_features.csv"
LEGACY_CHECK_TOL    = 1e-3     # relative

RANDOM_SEED         = 42


# ============================================================
# Feature names
# ============================================================

LEGACY = ["lum_mean", "lum_std", "lum_p05", "lum_p50", "lum_p95",
          "sat_mean", "sat_std", "val_mean", "colorfulness",
          "contrast_local", "edge_density"]
EXTRA = ["lum_skew", "lum_entropy", "chroma_mean", "lab_a_mean",
         "lab_b_mean", "spectral_slope", "dark_channel_mean"]
CROP = LEGACY + EXTRA

DEFINITIONS = {
    "lum_mean": "mean grey level (OpenCV Rec.601 grey, 0-255)",
    "lum_p50": "median grey level",
    "val_mean": "mean HSV value (max of R,G,B per pixel)",
    "lum_std": "SD of grey (RMS / global contrast; = old contrast_global)",
    "lum_p05": "5th percentile of grey",
    "lum_p95": "95th percentile of grey",
    "lum_skew": "skewness of grey",
    "lum_entropy": "Shannon entropy of the 256-bin grey histogram (bits)",
    "sat_mean": "mean HSV saturation (OpenCV, 0-255)",
    "sat_std": "SD of HSV saturation",
    "colorfulness": "Hasler & Suesstrunk (2003) colourfulness",
    "chroma_mean": "mean CIELAB chroma C*ab (sRGB, D65)",
    "lab_a_mean": "mean CIELAB a* (+red / -green)",
    "lab_b_mean": "mean CIELAB b* (+yellow / -blue)",
    "contrast_local": "SD of the Laplacian of grey (old sharpness = its "
                      "square)",
    "edge_density": "share of Canny edge pixels (thresholds 100/200)",
    "spectral_slope": "alpha of radially averaged amplitude spectrum "
                      "A(f) ~ 1/f^alpha (Hann window)",
    "dark_channel_mean": "mean dark channel (He et al., 2009), 0-1",
}


# ============================================================
# Geometry
# ============================================================

def clip_box(w, h, n=336):
    """The region of the ORIGINAL image that CLIP keeps, in original
    pixel coordinates (floats).

    CLIP resizes the short side to n (the long side truncated to an
    integer, as torchvision does) and then centre-crops n x n, with the
    crop offset rounded in the RESIZED image. Mapping that crop back
    gives a box that is generally not on whole original pixels. Cropping
    whole original pixels instead (the first version of this script)
    shifted the region by up to about one pixel, which is invisible but
    produced mean differences of 4-5 grey levels on detailed images and
    failed check 1. Resizing this exact box reproduces CLIP's view to
    within rounding (check 1)."""
    if w <= h:
        nw, nh = n, int(n * h / w)
    else:
        nw, nh = int(n * w / h), n
    left = int(round((nw - n) / 2.0))
    top = int(round((nh - n) / 2.0))
    sx, sy = nw / w, nh / h
    return (left / sx, top / sy, (left + n) / sx, (top + n) / sy)


def central_square(im, size, resample=Image.BILINEAR):
    """CLIP's region of a PIL image, resized to size x size in one step
    (Pillow's resize with a source box, antialiased)."""
    return im.resize((size, size), resample, box=clip_box(*im.size))


def clip_view(im, n=336):
    """CLIP's own geometry on a PIL image: resize the short side to n
    (bicubic; long side truncated as torchvision does), centre-crop."""
    w, h = im.size
    if w <= h:
        nw, nh = n, int(n * h / w)
    else:
        nw, nh = int(n * w / h), n
    r = im.resize((nw, nh), Image.BICUBIC)
    left = int(round((nw - n) / 2.0))
    top = int(round((nh - n) / 2.0))
    return r.crop((left, top, left + n, top + n))


# ============================================================
# Features
# ============================================================

_RADIAL = {}


def _radial_bins(n):
    """Integer radial frequency of every FFT coefficient (cached)."""
    if n not in _RADIAL:
        y, x = np.indices((n, n))
        r = np.hypot(x - n // 2, y - n // 2)
        _RADIAL[n] = np.round(r).astype(np.int64).ravel()
    return _RADIAL[n]


def spectral_slope(gray):
    n = gray.shape[0]
    g = gray - gray.mean()
    w = np.outer(np.hanning(n), np.hanning(n))
    amp = np.abs(np.fft.fftshift(np.fft.fft2(g * w))).ravel()
    ri = _radial_bins(n)
    sums = np.bincount(ri, amp)
    counts = np.bincount(ri)
    f = np.arange(len(sums))
    sel = (f >= SPECTRUM_FMIN) & (f <= SPECTRUM_FMAX) & (counts > 0)
    a = sums[sel] / counts[sel]
    if not np.all(a > 0):
        return np.nan
    slope = np.polyfit(np.log(f[sel]), np.log(a), 1)[0]
    return float(-slope)


def dark_patch(n):
    p = int(round(DARK_PATCH_AT_256 * n / 256.0))
    return max(3, p | 1)                          # odd, at least 3


def legacy_features(rgb_u8):
    """The 11 original features, with the old code's exact operations
    (new_encode_and_save.py, extract_features)."""
    rgb = rgb_u8.astype(np.float32)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    gray_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    sigma = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mu = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny(gray_u8, CANNY_LOW, CANNY_HIGH)
    vals = {
        "lum_mean": gray.mean(), "lum_std": gray.std(),
        "lum_p05": np.percentile(gray, 5),
        "lum_p50": np.percentile(gray, 50),
        "lum_p95": np.percentile(gray, 95),
        "sat_mean": S.mean(), "sat_std": S.std(), "val_mean": V.mean(),
        "colorfulness": sigma + 0.3 * mu,
        "contrast_local": lap.std(),
        "edge_density": edges.mean() / 255.0,
    }
    return [float(vals[k]) for k in LEGACY], gray_u8, gray


def all_features(rgb_u8):
    """The 18 CROP features on a square uint8 RGB array."""
    leg, gray_u8, gray = legacy_features(rgb_u8)
    n = rgb_u8.shape[0]

    sd = gray.std()
    skew = float((((gray - gray.mean()) / sd) ** 3).mean()) if sd > 0 \
        else 0.0
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    entropy = float(-(p * np.log2(p)).sum())

    lab = cv2.cvtColor(rgb_u8.astype(np.float32) / 255.0,
                       cv2.COLOR_RGB2Lab)
    a, b = lab[:, :, 1], lab[:, :, 2]
    chroma = float(np.sqrt(a ** 2 + b ** 2).mean())

    dc = rgb_u8.min(axis=2)
    k = dark_patch(n)
    dc = cv2.erode(dc, np.ones((k, k), np.uint8))
    dark = float(dc.mean() / 255.0)

    extra = {"lum_skew": skew, "lum_entropy": entropy,
             "chroma_mean": chroma, "lab_a_mean": float(a.mean()),
             "lab_b_mean": float(b.mean()),
             "spectral_slope": spectral_slope(gray),
             "dark_channel_mean": dark}
    return leg + [extra[k] for k in EXTRA]


def image_features(path, in_subsample):
    """(crop[18], stretch[11], crop336[18] or None) for one image.
    Raises on unreadable files."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        crop = central_square(im, FEATURE_SIZE)
        stretch = im.resize((FEATURE_SIZE, FEATURE_SIZE), Image.BILINEAR)
        c336 = central_square(im, CHECK_SIZE) if in_subsample else None
    f_crop = all_features(np.asarray(crop, dtype=np.uint8))
    f_str = legacy_features(np.asarray(stretch, dtype=np.uint8))[0]
    f_336 = (all_features(np.asarray(c336, dtype=np.uint8))
             if c336 is not None else None)
    return f_crop, f_str, f_336


# ============================================================
# Worker
# ============================================================

def _init_worker():
    cv2.setNumThreads(1)
    warnings.filterwarnings("ignore")


def work(task):
    """task = (ids, paths, subsample_flags). Never raises."""
    ids, paths, flags = task
    n = len(ids)
    crop = np.full((n, len(CROP)), np.nan, np.float32)
    stre = np.full((n, len(LEGACY)), np.nan, np.float32)
    sub = {}
    ok = np.zeros(n, dtype=bool)
    for j in range(n):
        try:
            fc, fs, f3 = image_features(paths[j], flags[j])
            crop[j], stre[j] = fc, fs
            if f3 is not None:
                sub[int(ids[j])] = np.asarray(f3, np.float32)
            ok[j] = True
        except Exception:
            pass
    return ids, crop, stre, sub, ok


# ============================================================
# Subsample, fingerprint, arrays
# ============================================================

def make_subsample(idx):
    rng = np.random.default_rng(RANDOM_SEED)
    out = []
    for _, g in idx.groupby("category", sort=True):
        ids = g["image_id"].to_numpy()
        k = min(SUBSAMPLE_PER_CATEGORY, len(ids))
        out.append(np.sort(rng.choice(ids, size=k, replace=False)))
    return np.sort(np.concatenate(out))


def fingerprint(idx):
    h = hashlib.sha1()
    for k, p in zip(idx["image_key"], idx["path"]):
        h.update(f"{k}\t{p}\n".encode("utf-8"))
    return h.hexdigest()


def settings_dict():
    return {"feature_size": FEATURE_SIZE, "check_size": CHECK_SIZE,
            "subsample_per_category": SUBSAMPLE_PER_CATEGORY,
            "canny": [CANNY_LOW, CANNY_HIGH],
            "spectrum_range": [SPECTRUM_FMIN, SPECTRUM_FMAX],
            "dark_patch_at_256": DARK_PATCH_AT_256,
            "random_seed": RANDOM_SEED,
            "crop_features": CROP, "stretch_features": LEGACY}


def save_status(status):
    tmp = os.path.join(OUTPUT_DIR, "status.tmp.npy")
    np.save(tmp, status)
    os.replace(tmp, os.path.join(OUTPUT_DIR, "status.npy"))


def open_arrays(n, n_sub, meta):
    P = lambda f: os.path.join(OUTPUT_DIR, f)
    fmt = np.lib.format
    if os.path.exists(P("run_meta.json")) and os.path.exists(P("status.npy")):
        with open(P("run_meta.json")) as f:
            old = json.load(f)
        diffs = [k for k in ("index_fingerprint", "n_images", "settings")
                 if old.get(k) != meta.get(k)]
        if diffs:
            raise RuntimeError(
                f"Existing run in {OUTPUT_DIR} has different {diffs}. "
                "Refusing to resume; use a new OUTPUT_DIR or delete it.")
        return (fmt.open_memmap(P("features_crop.npy"), mode="r+"),
                fmt.open_memmap(P("features_stretch.npy"), mode="r+"),
                fmt.open_memmap(P("features_crop336_subsample.npy"),
                                mode="r+"),
                np.load(P("status.npy")), True)
    crop = fmt.open_memmap(P("features_crop.npy"), mode="w+",
                           dtype=np.float32, shape=(n, len(CROP)))
    stre = fmt.open_memmap(P("features_stretch.npy"), mode="w+",
                           dtype=np.float32, shape=(n, len(LEGACY)))
    sub = fmt.open_memmap(P("features_crop336_subsample.npy"), mode="w+",
                          dtype=np.float32, shape=(n_sub, len(CROP)))
    crop[:] = np.nan
    stre[:] = np.nan
    sub[:] = np.nan
    status = np.zeros(n, dtype=np.int8)
    save_status(status)
    return crop, stre, sub, status, False


# ============================================================
# Checks
# ============================================================

def check_region(idx):
    """Check 1: the central square matches CLIP's view."""
    mads_crop, mads_str, used = [], [], 0
    for p, a in zip(idx["path"], idx["aspect"]):
        if used >= N_CHECK_IMAGES:
            break
        if abs(a - 1) < 0.05:
            continue            # square images cannot tell the two apart
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                ref = np.asarray(clip_view(im), np.float32)
                ours = np.asarray(central_square(im, 336, Image.BICUBIC),
                                  np.float32)
                strt = np.asarray(im.resize((336, 336), Image.BICUBIC),
                                  np.float32)
        except Exception:
            continue
        mads_crop.append(np.abs(ref - ours).mean())
        mads_str.append(np.abs(ref - strt).mean())
        used += 1
    if not used:
        print("  check 1 (region): no non-square images among the first "
              "rows; skipped")
        return {"region_check": "skipped"}
    worst = float(np.max(mads_crop))
    ok = worst <= CROP_CHECK_MAX_MAD
    print(f"  check 1 (region vs CLIP view, {used} non-square images): "
          f"mean abs diff central square max {worst:.2f}, stretched "
          f"image mean {np.mean(mads_str):.2f} grey levels -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("Central square does not match CLIP's view.")
    return {"region_check_max_mad": worst,
            "region_check_stretch_mean_mad": float(np.mean(mads_str))}


def check_legacy(idx):
    """Check 2: STRETCH features reproduce the old classical_features."""
    if not (OLD_IMAGE_INDEX and OLD_FEATURES_CSV
            and os.path.exists(OLD_IMAGE_INDEX)
            and os.path.exists(OLD_FEATURES_CSV)):
        print("  check 2 (legacy reproduction): old files not found; "
              "skipped")
        return {"legacy_check": "skipped"}
    old_paths = pd.read_csv(OLD_IMAGE_INDEX, usecols=["path"],
                            dtype=str)["path"]
    row_of = dict(zip(old_paths, range(len(old_paths))))
    rows, paths = [], []
    for p in idx["path"]:
        if p in row_of:
            rows.append(row_of[p])
            paths.append(p)
        if len(rows) >= N_CHECK_IMAGES:
            break
    if not rows:
        print("  check 2 (legacy reproduction): no overlapping images; "
              "skipped")
        return {"legacy_check": "skipped"}
    want = set(rows)
    old = pd.read_csv(OLD_FEATURES_CSV,
                      skiprows=lambda i: i > 0 and (i - 1) not in want)
    old.index = sorted(want)
    worst = 0.0
    for r, p in zip(rows, paths):
        _, fs, _ = image_features(p, False)
        o = old.loc[r, LEGACY].to_numpy(np.float64)
        rel = np.abs(np.asarray(fs) - o) / np.maximum(np.abs(o), 1e-6)
        worst = max(worst, float(rel.max()))
    ok = worst <= LEGACY_CHECK_TOL
    print(f"  check 2 (legacy reproduction, {len(rows)} images): max "
          f"relative diff {worst:.2e} -> {'PASS' if ok else 'WARN'}")
    if not ok:
        warnings.warn("Legacy features differ from the old file; library "
                      "versions may differ. Inspect before comparing.")
    return {"legacy_check_max_rel_diff": worst}


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.setNumThreads(1)
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

    print("=" * 70)
    print("04_compute_features.py")
    print("=" * 70)
    idx = pd.read_csv(INDEX_CSV)
    if TEST_LIMIT is not None:
        idx = idx.iloc[:TEST_LIMIT].copy()
    if not (idx["image_id"].to_numpy() == np.arange(len(idx))).all():
        raise RuntimeError("image_id must be 0..N-1 in row order.")
    n = len(idx)
    print(f"Index   : {INDEX_CSV}  ({n:,} images"
          + (", TEST_LIMIT)" if TEST_LIMIT else ")"))
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Workers : {N_WORKERS}")

    sub_ids = make_subsample(idx)
    sub_row = {int(i): r for r, i in enumerate(sub_ids)}
    in_sub = np.zeros(n, dtype=bool)
    in_sub[sub_ids] = True
    print(f"Subsample (crop at {CHECK_SIZE}): {len(sub_ids):,} images")

    print("\nStart-up checks")
    checks = {}
    checks.update(check_region(idx))
    checks.update(check_legacy(idx))

    meta = {"script": "04_compute_features.py", "index_csv": INDEX_CSV,
            "index_fingerprint": fingerprint(idx), "n_images": n,
            "settings": settings_dict(), "checks": checks,
            "opencv": cv2.__version__}
    crop, stre, sub, status, resumed = open_arrays(n, len(sub_ids), meta)
    if resumed:
        with open(os.path.join(OUTPUT_DIR, "run_meta.json")) as f:
            meta["started"] = json.load(f).get("started")
        print(f"\nResuming: {int((status != 0).sum()):,} of {n:,} done.")
    else:
        meta["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    np.save(os.path.join(OUTPUT_DIR, "subsample_ids.npy"), sub_ids)
    with open(os.path.join(OUTPUT_DIR, "feature_definitions.json"),
              "w") as f:
        json.dump({"crop": CROP, "stretch": LEGACY,
                   "definitions": DEFINITIONS,
                   "settings": settings_dict()}, f, indent=2)

    # --- Compute ---
    todo = np.where(status == 0)[0]
    paths = idx["path"].to_numpy()
    tasks = [(todo[i:i + CHUNK], paths[todo[i:i + CHUNK]],
              in_sub[todo[i:i + CHUNK]])
             for i in range(0, len(todo), CHUNK)]
    print(f"\nTo process: {len(todo):,} images in {len(tasks):,} tasks")
    t_run = time.time()
    if tasks:
        with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
            done_tasks = 0
            try:
                for ids, c, s, sb, ok in tqdm(
                        pool.imap_unordered(work, tasks),
                        total=len(tasks), desc="features", unit="task"):
                    crop[ids] = c
                    stre[ids] = s
                    for i, v in sb.items():
                        sub[sub_row[i]] = v
                    status[ids] = np.where(ok, 1, -1).astype(np.int8)
                    done_tasks += 1
                    if done_tasks % FLUSH_EVERY_TASKS == 0:
                        crop.flush(); stre.flush(); sub.flush()
                        save_status(status)
            finally:
                crop.flush(); stre.flush(); sub.flush()
                save_status(status)
    t_run = time.time() - t_run
    rate = len(todo) / t_run if t_run > 0 and len(todo) else 0.0

    # --- Check 3: alignment ---
    rng = np.random.default_rng(RANDOM_SEED)
    ok_rows = np.where(status == 1)[0]
    pick = rng.choice(ok_rows, size=min(FINAL_RECHECK_N, len(ok_rows)),
                      replace=False)
    ok3 = True
    for i in pick:
        fc, fs, f3 = image_features(paths[i], bool(in_sub[i]))
        same = (np.allclose(fc, crop[i], rtol=1e-5, atol=1e-6,
                            equal_nan=True)
                and np.allclose(fs, stre[i], rtol=1e-5, atol=1e-6,
                                equal_nan=True))
        if f3 is not None:
            same &= np.allclose(f3, sub[sub_row[int(i)]], rtol=1e-5,
                                atol=1e-6, equal_nan=True)
        ok3 &= bool(same)
    print(f"\n  check 3 (alignment): {len(pick)} images recomputed -> "
          f"{'PASS' if ok3 else 'FAIL'}")

    # --- CSV outputs ---
    done_all = not (status == 0).any()
    if done_all:
        print("Writing CSV files...")
        ids = idx["image_id"].to_numpy()
        pd.DataFrame(np.asarray(crop), columns=CROP).assign(
            image_id=ids)[["image_id"] + CROP].to_csv(
            os.path.join(OUTPUT_DIR, "features_crop.csv"), index=False,
            float_format="%.6g")
        pd.DataFrame(np.asarray(stre), columns=LEGACY).assign(
            image_id=ids)[["image_id"] + LEGACY].to_csv(
            os.path.join(OUTPUT_DIR, "features_stretch.csv"), index=False,
            float_format="%.6g")
        pd.DataFrame(np.asarray(sub), columns=CROP).assign(
            image_id=sub_ids)[["image_id"] + CROP].to_csv(
            os.path.join(OUTPUT_DIR, "features_crop336_subsample.csv"),
            index=False, float_format="%.6g")
    failed = np.where(status == -1)[0]
    if len(failed):
        idx.iloc[failed][["image_id", "path"]].to_csv(
            os.path.join(OUTPUT_DIR, "failed_images.csv"), index=False)

    # --- Report ---
    ok_mask = status == 1
    C = np.asarray(crop)[ok_mask]

    def stats_table(M, names):
        rows = []
        for j, nm in enumerate(names):
            v = M[:, j]
            rows.append({"feature": nm,
                         "mean": np.nanmean(v), "sd": np.nanstd(v),
                         "p01": np.nanpercentile(v, 1),
                         "p99": np.nanpercentile(v, 99),
                         "n_nan": int(np.isnan(v).sum())})
        return pd.DataFrame(rows)

    meta.update({"finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "n_ok": int(ok_mask.sum()), "n_failed": int(len(failed)),
                 "n_not_done": int((status == 0).sum()),
                 "images_per_second_this_run": round(rate, 1),
                 "alignment_check_pass": ok3})
    with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    lines = [
        "04_compute_features.py -- summary",
        "=" * 60,
        f"Started / finished : {meta['started']} / {meta['finished']}",
        f"Index              : {INDEX_CSV}  ({n:,} images)",
        f"Feature size       : {FEATURE_SIZE} (central square); legacy "
        "set on the stretched whole image",
        f"Subsample at {CHECK_SIZE}  : {len(sub_ids):,} images",
        "",
        "Checks",
        ("  1 region vs CLIP view : skipped"
         if checks.get("region_check") == "skipped" else
         f"  1 region vs CLIP view : PASS (max mean abs diff "
         f"{checks['region_check_max_mad']:.2f}; stretched image "
         f"{checks['region_check_stretch_mean_mad']:.2f})"),
        ("  2 legacy reproduction : skipped"
         if checks.get("legacy_check") == "skipped" else
         f"  2 legacy reproduction : max rel diff "
         f"{checks['legacy_check_max_rel_diff']:.2e}"),
        f"  3 alignment           : {'PASS' if ok3 else 'FAIL'}",
        "",
        "Images",
        f"  ok         : {meta['n_ok']:,}",
        f"  unreadable : {meta['n_failed']:,}",
        f"  not done   : {meta['n_not_done']:,}",
        f"  this run   : {len(todo):,} in {t_run / 3600:.2f} h "
        f"({rate:.1f} img/s)",
        "",
        "CROP features (all ok images)",
        stats_table(C, CROP).round(4).to_string(index=False)
        if len(C) else "  (none)",
        "",
        f"Total time: {(time.time() - t0) / 3600:.2f} h",
    ]
    if not done_all:
        lines.insert(-1, "NOTE: not finished; run again to continue. "
                         "CSV files are written when all images are done.")
    report = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "features_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    if not ok3:
        raise RuntimeError("Alignment check failed.")


if __name__ == "__main__":
    main()
