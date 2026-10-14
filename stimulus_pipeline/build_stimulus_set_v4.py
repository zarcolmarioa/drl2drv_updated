"""
build_stimulus_set_v4.py  (stimulus pipeline, v4 axes, step 3)

Takes the hand-screened scene lists, checks that the set can support
every trial type the experiment needs, selects the final subset, and
writes the stimulus images and the manifest.

Where this sits
---------------
  step 1  calibrate_levels_v4   the five colour and five fog rungs
  step 2  select_scenes_v4      shortlist -> you screen by hand
  step 3  this one              the screened set is sound -> stimuli
  step 4  analyse_pairs_v4      pair inventory and thresholds
  step 5  build_trials_v4       the trial list

You screen on CONTENT -- framing, legibility, whether the place is
ordinary. This screens on FEATURE COVERAGE. Neither does the other's
job, which is why they are separate.

TWO STIMULUS SETS
-----------------
FOG_FAMILY picks which fog this build uses, and the output directory is
named after it, so the two sets never collide:

  "haze"     the v2 fog: I' = (1-h)I + hA
  "haze_eq"  the same blend, then a linear-light gain restoring the
             original mean grey

Run the script twice, once per family, to get both. The scene list,
the trial-type thresholds and the selection are identical; only the
rendering differs. CHECK 3 verifies that assumption on this set rather
than taking it on trust.

TWO PASSES
----------
RENDER_MODE = "preview" measures everything, runs every check, selects
the set, and writes ONE CONTACT SHEET PER SCENE showing its whole
colour x fog grid -- but no stimulus images. That is the cheap pass:
you look at the sheets, drop whatever fails by eye, and only then

RENDER_MODE = "full" writes the stimulus images and the manifest.

The preview sheets come with a `screening_list.csv` of their own, so
make_screening_page.py can be pointed at them and the second cull works
exactly like the first.

What it checks
--------------
  CHECK 1  the copied feature code still reproduces data/features.
           Fatal: every number below depends on it.

  CHECK 2  the ENVELOPE, on the COMBINED cells. Step 1 could only check
           one manipulation at a time; a cell applies both, and fog
           lifting the black point while chroma drains colour is a
           different question from either alone. Any cell taking a
           feature the design does not intend to move outside the
           corpus 1-99% range is listed.

  CHECK 3  the two fog families remove the same amount of fine detail
           at matched rungs -- the assumption that lets ONE scene list
           serve BOTH stimulus sets.

           Expressed against the scene's change at the DEEPEST rung,
           not against the per-rung change. Step 2 divided by the
           per-rung change, which at the lightest rung is small enough
           that a trivial difference reads as a huge ratio; it warned
           at 1.117 and 0.840 on two samples of the same corpus, which
           is the signature of an unstable denominator rather than a
           real difference. This version is reported per rung so you
           can see where any difference actually lives.

  CHECK 4  TRIAL-TYPE AFFORDANCE. The experiment needs four kinds of
           pair and not every scene affords all four:

             sat_only   two cells in the same fog row. Chroma scaling
                        works on a* and b* at constant L*, so the fog
                        feature is unchanged along a row and only
                        colour differs.

             fog_only   two cells in different fog rows whose MEASURED
                        saturations coincide. Needed because fog drains
                        colour as a side effect, so an unmatched
                        comparison confounds the two. Which saturations
                        coincide is scene-dependent, so these are found
                        by search, not by a rule.

             together   foggier AND less saturated: the way the world
                        works. The largest expected effect.

             conflict   foggier AND MORE saturated. The cues disagree,
                        so the participant has to reveal which one they
                        follow. The only type that can show the
                        crossover. Scarce by construction.

  CHECK 5  MAGNITUDE COVERAGE. The dose-response analysis splits trials
           by difference size, so the set must produce small, medium
           and large differences rather than clustering at one value.

Inputs
------
  SCREENING_LISTS   one or more CSVs with `keep` filled in (step 2 plus
                    make_screening_page.py). Rows with keep == 1 are
                    pooled and duplicates dropped, so several screening
                    rounds combine. Their measured columns are IGNORED:
                    different rounds may have used different settings,
                    so everything is recomputed here.
  LEVELS_CSV        data/levels_v4/levels.csv            (step 1)
  STATS_CSV         data/levels_v4/corpus_feature_stats_v4.csv (step 1)
  INDEX_CSV         data/selection/image_index.csv       (step 02)
  FEATURES_NPY      data/features/features_crop.npy      (step 04)
  CATEGORY_CSV      everyday_categories.csv, for the `group` column

Outputs (under OUTPUT_DIR, which is named after the fog family)
--------------------------------------------------------------
  stimulus_set.csv        the selected scenes, with per-type counts
  stimulus_manifest.csv   one row per IMAGE: scene x cell, measured
                          features, filename. This is what step 4 and
                          step 5 read.
  grid_cells.csv          every measured cell, selected or not
  pair_affordance.csv     per-scene counts of each trial type
  pairs.csv               every usable pair, with its type and deltas
  dropped_scenes.csv      what was dropped and why
  envelope_violations.csv cells taking an unintended feature outside
                          the corpus range
  fog_equivalence.csv     CHECK 3, per rung
  balance.csv             category / group / indoor-outdoor of the set
  screening_list.csv      worksheet for the second hand-screening pass
  sheets/<scene>.jpg      one colour x fog contact sheet per scene
  stimuli/<name>.png      the stimulus images   (RENDER_MODE "full")
  affordance.png, coverage.png, balance.png
  build_report.txt

NOT WRITTEN HERE: the nine-axis profile of the final set. v2 encoded
every stimulus inline, which needs the un2CLIP checkpoint and a GPU.
It is cheaper and clearer to score the axes once, on the final set,
after the second cull -- so that is its own small step rather than
something every build pays for.

Usage
-----
    conda activate drl2drv
    python build_stimulus_set_v4.py            # preview pass
    # look at sheets/, cull with make_screening_page.py
    # add the culled list to SCREENING_LISTS, set RENDER_MODE = "full"
    python build_stimulus_set_v4.py
    # then again with FOG_FAMILY = "haze_eq"

CPU only. No encoder, no GPU.
"""

import os
import time
import itertools
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# OpenCV's own thread pool must be off BEFORE the first cv2 call and
# before any fork. multiprocessing forks on Linux, and a fork inherits
# whatever locks were held at the moment it happened: once OpenCV has
# started its workers in the parent, a child calling into cv2 waits
# forever on a mutex whose owning thread does not exist in the child.
# It deadlocks silently -- every worker at 0% CPU, no error, no
# progress. Zero means OpenCV runs in the calling thread, which is what
# a pool of single-purpose workers wants anyway: the parallelism is
# across scenes, not inside one image.
cv2.setNumThreads(0)


# ============================================================
# SETTINGS
# ============================================================

# --- Input ---
SCREENING_LISTS = [
    "data/select_scenes_v4_ec/screening_list_done.csv",
]

INDEX_CSV    = "data/selection/image_index.csv"
FEATURES_NPY = "data/features/features_crop.npy"
CATEGORY_CSV = "everyday_categories.csv"
LEVELS_CSV   = "data/levels_v4_ec/levels.csv"
STATS_CSV    = "data/levels_v4_ec/corpus_feature_stats_v4.csv"

# STATS_POPULATION
#   Which row block of corpus_feature_stats_v4.csv the envelope is read
#   from. "stimulus_path" is the right one: those 20,000 images were
#   measured the way a STIMULUS is measured (original -> 512 -> 256),
#   which is exactly how the cells below are measured, whereas
#   "everyday_colour" was measured on the corpus path (original ->
#   256). Comparing a stimulus against a corpus-path reference is the
#   mismatch that cost a round of step 1 and a bug in step 2.
#
#   For the envelope features it happens to make almost no difference
#   -- the path mainly moves contrast_local, which is excluded here
#   because it is what the manipulation is FOR. The cost of being
#   right is nothing, so be right.
#
#   "everyday_colour" is the alternative if you want the larger sample
#   (209,119 against 20,000) and tighter percentile estimates.
STATS_POPULATION = "stimulus_path"

# --- Which fog, and where it goes ---
#
# FOG_FAMILY / FOG_DIMENSION
#   "haze"    + "fog"      the v2 fog
#   "haze_eq" + "fog_eq"   brightness-corrected fog
#   Both are calibrated in step 1; the dimension names the block of
#   levels.csv to read the parameters from.
FOG_FAMILY    = "haze"
FOG_DIMENSION = "fog"

# OUTPUT_ROOT
#   The family name is appended, so the two sets land in
#   data/stimulus_set_v4_haze/ and data/stimulus_set_v4_haze_eq/ and
#   can never overwrite one another.
OUTPUT_ROOT = "data/stimulus_set_v4_ec"

# ON_EXISTING
#   "new"   write to OUTPUT_DIR_2, _3, ... if the directory already
#           holds a build. Nothing is ever overwritten; the script
#           prints where it actually wrote.
#   "error" stop instead.
ON_EXISTING = "new"

# --- The ladder (level NAMES from step 1, not parameters) ---
#
# Keep these matched between the two families so the two stimulus sets
# are comparable rung for rung.
COLOUR_LEVELS = ["p25", "p10", "p05", "p01"]
FOG_LEVELS    = ["p35", "p25", "p15", "p10", "p05"]

# INCLUDE_UNMANIPULATED_COLOUR
#   Adds the k = 1 column, the scene's own colour. Needed: the fog-only
#   and conflict pairs live largely in it, and it is the reference the
#   manifest's deltas are taken from.
INCLUDE_UNMANIPULATED_COLOUR = True

# --- Geometry (must match step 04, step 1 and step 2) ---
IMAGE_SIZE   = 512     # manipulation and output size
FEATURE_SIZE = 256     # measurement size
CLIP_PX      = 336

# --- Manipulation parameters (as 09c / steps 1 and 2) ---
MANIP_ORDER     = ("haze", "chroma")
IDENTITY_SHORT_CIRCUIT = True
HAZE_AIRLIGHT_TOP = 1.0
GAMUT_ITERS       = 16
GAMUT_CMAX_SEARCH = 160.0
GAMUT_MARGIN      = 0.995

# --- Features ---
FEAT_SAT = "sat_mean"
FEAT_FOG = "contrast_local"

# FOG_SIGN
#   Multiplier on the raw fog-feature difference so that POSITIVE
#   always means "image A is foggier". -1 for contrast_local, which
#   fog LOWERS. Applied once, in classify_pairs.
#
#   Getting this backwards would swap `together` and `conflict`, since
#   that distinction turns on whether d_fog and d_sat share a sign.
FOG_SIGN = -1.0

# --- Trial-type thresholds (v2's, in the features' own units) ---
#
# MIN_DSAT_SAT_ONLY
#   How far apart two same-fog cells must be in sat_mean to count as a
#   saturation trial. Below this the two images look alike.
MIN_DSAT_SAT_ONLY = 17.0 # 12.0

# PAIR_SAT_TOL / MIN_DFOG_FOG_ONLY
#   A fog-only pair: sat_mean within PAIR_SAT_TOL, contrast_local at
#   least MIN_DFOG_FOG_ONLY apart.
PAIR_SAT_TOL      = 3.0
MIN_DFOG_FOG_ONLY = 4.2

# MIN_DSAT_MIXED / MIN_DFOG_MIXED
#   A mixed pair must differ by at least this much on BOTH features,
#   or it is really a single-feature trial with noise. The saturation
#   threshold is lower than MIN_DSAT_SAT_ONLY because conflict pairs
#   cannot achieve large positive d_sat -- fog drains colour, so a
#   foggier image is rarely much MORE saturated.
MIN_DSAT_MIXED = 8.0
MIN_DFOG_MIXED = 3.0

# MIN_PAIRS_PER_TYPE
#   A scene must afford at least this many usable pairs of EVERY type
#   or it is dropped. Low on purpose: trials are spread across scenes,
#   so a scene needs only a handful.
MIN_PAIRS_PER_TYPE = 3

# --- Magnitude coverage ---
N_MAGNITUDE_BINS = 3
MIN_PAIRS_PER_BIN = 40

# --- Final selection ---
#
# N_SELECT
#   How many scenes to carry forward. 0 keeps every scene that passes,
#   which is what you want on a screened set of this size: the hand
#   screening has already done the choosing, and the second visual pass
#   on the grid sheets will take a few more.
N_SELECT = 0

# BALANCE_BY
#   Which label the selection spreads across when N_SELECT trims the
#   set. "indoor" uses INDOOR_CATEGORIES below, "group" uses the
#   `group` column of CATEGORY_CSV, None takes whichever scenes afford
#   the most conflict pairs. Ignored when N_SELECT is 0, though the
#   balance is REPORTED either way.
BALANCE_BY = "indoor"

# SELECTION_RULE
#   How scenes are ranked within a balancing stratum. "conflict"
#   prefers the scenes affording the most conflict pairs, which is the
#   scarce type. "random" ignores affordance once a scene has passed.
SELECTION_RULE = "conflict"

RANDOM_SEED = 42

# --- Indoor / outdoor ---
#
# INDOOR_CATEGORIES / AMBIGUOUS_CATEGORIES
#   A JUDGEMENT, not data: `everyday_categories.csv` has no such
#   column, and Places365 marks only 14 of the 82 in the name itself.
#   It is written out here so it can be checked, argued with and cited,
#   and it lands in the manifest so indoor/outdoor can be entered as a
#   covariate after the fact.
#
#   READ THIS LIST BEFORE THE FINAL BUILD. Ambiguous categories are
#   counted separately rather than forced to a side; a station platform
#   may be underground or open to the sky, and a porch is neither.
INDOOR_CATEGORIES = {
    # academic
    "auditorium", "cafeteria", "classroom", "computer_room",
    "conference_room", "dining_hall", "dorm_room", "gymnasium/indoor",
    "lecture_room", "library/indoor", "locker_room", "office",
    "office_cubicles",
    # domestic
    "balcony/interior", "bathroom", "bedroom", "closet", "corridor",
    "dining_room", "garage/indoor", "home_office", "kitchen",
    "living_room", "pantry", "shower", "staircase", "utility_room",
    # retail and food
    "bar", "clothing_store", "coffee_shop", "department_store",
    "drugstore", "fastfood_restaurant", "food_court", "gift_shop",
    "ice_cream_parlor", "laundromat", "pharmacy", "restaurant",
    "shoe_shop", "shopping_mall/indoor", "sushi_bar",
    # street and transit
    "parking_garage/indoor",
    "bus_interior", "bus_station/indoor", "car_interior",
    "elevator/door", "elevator_lobby", "entrance_hall",
    "escalator/indoor", "lobby", "reception", "train_interior",
    "waiting_room",
}
AMBIGUOUS_CATEGORIES = {
    "porch",                      # covered, but open to the outside
    "subway_station/platform",    # underground or open-air
    "train_station/platform",
}

# --- Envelope (CHECK 2) ---
#
# ENVELOPE_FEATURES
#   Features the design does NOT intend to move. sat_mean and
#   contrast_local are excluded because they are what the manipulation
#   is FOR, and lum_std / lum_entropy because they are near-duplicates
#   of contrast_local on this material -- flagging them would be
#   flagging the manipulation working.
ENVELOPE_FEATURES = ["lum_mean", "lum_p05", "lum_p50", "lum_p95",
                     "val_mean", "lum_skew", "spectral_slope"]
ENVELOPE_LO, ENVELOPE_HI = "p01", "p99"

# ENVELOPE_MAX_CELL_FRAC
#   Fraction of cells allowed outside the corpus range before the check
#   reports a problem rather than a note. Not zero: the deepest rungs
#   are meant to be extreme, and v2 shipped with 95% of its h = 0.49
#   images above the corpus p99 for lum_p05. The point is to SEE it.
ENVELOPE_MAX_CELL_FRAC = 0.25

# --- Fog-family equivalence (CHECK 3) ---
FOG_EQUIV_N   = 24      # scenes sampled
FOG_EQUIV_TOL = 0.05    # of the deepest-rung change

# --- Output images ---
#
# RENDER_MODE
#   "preview"  checks, selection and one grid sheet per scene
#   "full"     also the stimulus PNGs and the manifest
RENDER_MODE = "full"

STIMULUS_FORMAT  = "png"    # lossless; the stimuli are the experiment
SHEET_TILE_PX    = 240      # per cell in the grid sheets
SHEET_QUALITY    = 92

# --- Checks and parallelism ---
FEATURE_CHECK_N    = 8
FEATURE_CHECK_RTOL = 1e-4
N_WORKERS = 8
CHUNK     = 2


# ============================================================
# Feature code  (verbatim from 04_compute_features.py / steps 1, 2)
# ============================================================

LEGACY = ["lum_mean", "lum_std", "lum_p05", "lum_p50", "lum_p95",
          "sat_mean", "sat_std", "val_mean", "colorfulness",
          "contrast_local", "edge_density"]
EXTRA = ["lum_skew", "lum_entropy", "chroma_mean", "lab_a_mean",
         "lab_b_mean", "spectral_slope", "dark_channel_mean"]
CROP = LEGACY + EXTRA

CANNY_LOW, CANNY_HIGH = 100, 200
SPECTRUM_FMIN, SPECTRUM_FMAX = 8, 64
DARK_PATCH_AT_256 = 7

_RADIAL = {}


def _radial_bins(n):
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
    return float(-np.polyfit(np.log(f[sel]), np.log(a), 1)[0])


def dark_patch(n):
    p = int(round(DARK_PATCH_AT_256 * n / 256.0))
    return max(3, p | 1)


def legacy_features(rgb_u8):
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
    leg, gray_u8, gray = legacy_features(rgb_u8)
    n = rgb_u8.shape[0]
    sd = gray.std()
    skew = float((((gray - gray.mean()) / sd) ** 3).mean()) if sd > 0 else 0.0
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    entropy = float(-(p * np.log2(p)).sum())
    lab = cv2.cvtColor(rgb_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
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


def measure(rgb_u8, only=None):
    """Features of a square uint8 image, resized to FEATURE_SIZE first.

    The legacy-only fast path returns bit-identical values while
    skipping the FFT, the histogram, the CIELAB conversion and the dark
    channel.
    """
    if rgb_u8.shape[0] != FEATURE_SIZE:
        rgb_u8 = np.asarray(Image.fromarray(rgb_u8).resize(
            (FEATURE_SIZE, FEATURE_SIZE), Image.BILINEAR), dtype=np.uint8)
    if only is not None and all(f in LEGACY for f in only):
        vals = dict(zip(LEGACY, legacy_features(rgb_u8)[0]))
        return {f: vals[f] for f in only}
    vals = dict(zip(CROP, all_features(rgb_u8)))
    return vals if only is None else {f: vals[f] for f in only}


# ============================================================
# Geometry  (verbatim from 04_compute_features.py)
# ============================================================

def clip_box(w, h, n=CLIP_PX):
    """The region of the ORIGINAL image that CLIP keeps."""
    if w <= h:
        nw, nh = n, int(n * h / w)
    else:
        nw, nh = int(n * w / h), n
    left = int(round((nw - n) / 2.0))
    top = int(round((nh - n) / 2.0))
    sx, sy = nw / w, nh / h
    return (left / sx, top / sy, (left + n) / sx, (top + n) / sy)


def central_square(im, size, resample=Image.BILINEAR):
    """CLIP's region of a PIL image, resized in one step."""
    return im.resize((size, size), resample, box=clip_box(*im.size))


# ============================================================
# Manipulations  (verbatim from 09c / steps 1, 2)
# ============================================================

_XN, _ZN = 0.950456, 1.088754
_M_XYZ = np.array([[0.412453, 0.357580, 0.180423],
                   [0.212671, 0.715160, 0.072169],
                   [0.019334, 0.119193, 0.950227]])
_M_RGB = np.linalg.inv(_M_XYZ).astype(np.float32)


def to_u8(x):
    # np.round matters: without it this truncates, which darkens every
    # manipulated pixel by half a level.
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def srgb_to_lin(u8):
    x = u8.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb_f(x):
    x = np.maximum(x, 0.0)
    y = np.where(x <= 0.0031308, 12.92 * x,
                 1.055 * x ** (1 / 2.4) - 0.055)
    return (y * 255.0).astype(np.float32)


def mean_grey(u8):
    return float(cv2.cvtColor(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR),
                              cv2.COLOR_BGR2GRAY).mean())


def _f(img):
    return img.astype(np.float32)


def lab_of(u8):
    return cv2.cvtColor(u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def rgb_of_f(lab):
    """CIELAB -> sRGB on the 0-255 scale, UNCLIPPED."""
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0

    def finv(t):
        return np.where(t > 6.0 / 29.0, t ** 3,
                        3.0 * (6.0 / 29.0) ** 2 * (t - 4.0 / 29.0))
    xyz = np.stack([_XN * finv(fx), finv(fy), _ZN * finv(fz)], axis=-1)
    lin = xyz.astype(np.float32) @ _M_RGB.T
    y = np.where(lin <= 0.0031308, 12.92 * lin,
                 1.055 * np.maximum(lin, 0.0031308) ** (1 / 2.4) - 0.055)
    return (y * 255.0).astype(np.float32)


def _lab_to_lin(L, a, b):
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0

    def finv(t):
        return np.where(t > 6.0 / 29.0, t ** 3,
                        3.0 * (6.0 / 29.0) ** 2 * (t - 4.0 / 29.0))
    xyz = np.stack([_XN * finv(fx), finv(fy), _ZN * finv(fz)], axis=-1)
    return xyz.astype(np.float32) @ _M_RGB.T


def cmax_of(L, h, iters=GAMUT_ITERS):
    lo = np.zeros(L.shape, np.float32)
    hi = np.full(L.shape, GAMUT_CMAX_SEARCH, np.float32)
    ch, sh = np.cos(h).astype(np.float32), np.sin(h).astype(np.float32)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lin = _lab_to_lin(L, mid * ch, mid * sh)
        ok = ((lin >= 0.0) & (lin <= 1.0)).all(axis=-1)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return lo * GAMUT_MARGIN


def m_chroma(img, k, ctx):
    lab = lab_of(img)
    if k <= 1.0:
        lab[:, :, 1:] *= k
        return rgb_of_f(lab)
    a, b = lab[:, :, 1], lab[:, :, 2]
    C = np.sqrt(a * a + b * b)
    H = np.maximum(ctx["cmax"] - C, 0.0)
    C2 = C + H * (1.0 - np.exp(-(k - 1.0) * C / np.maximum(H, 1e-6)))
    s = np.where(C > 1e-6, C2 / np.maximum(C, 1e-6), 1.0)
    lab[:, :, 1] *= s
    lab[:, :, 2] *= s
    return rgb_of_f(lab)


def m_haze(img, h, ctx):
    A = np.asarray(ctx["airlight"], np.float32)[None, None, :]
    return (1.0 - h) * _f(img) + h * A


def m_haze_eq(img, h, ctx):
    """Haze, then a linear-light gain restoring the level-0 mean grey."""
    hz = to_u8(m_haze(img, h, ctx))
    target = ctx["lum_mean_512"]
    lin = srgb_to_lin(hz)
    lo, hi = -4.0, 4.0
    out = _f(hz)
    for _ in range(14):
        ev = 0.5 * (lo + hi)
        out = lin_to_srgb_f(lin * 2.0 ** ev)
        if mean_grey(to_u8(out)) < target:
            lo = ev
        else:
            hi = ev
    return out


FUNCS = {"chroma": m_chroma, "haze": m_haze, "haze_eq": m_haze_eq}
IDENTITY_PARAM = {"chroma": 1.0, "haze": 0.0, "haze_eq": 0.0}


def apply(fam, img, p, ctx):
    if fam is None:
        return img
    if IDENTITY_SHORT_CIRCUIT and p == IDENTITY_PARAM.get(fam):
        return img
    return to_u8(FUNCS[fam](img, p, ctx))


def airlight_of(sq):
    """Mean RGB of the brightest HAZE_AIRLIGHT_TOP % of the pixels."""
    g = cv2.cvtColor(cv2.cvtColor(sq, cv2.COLOR_RGB2BGR),
                     cv2.COLOR_BGR2GRAY)
    thr = np.percentile(g, 100 - HAZE_AIRLIGHT_TOP)
    return sq[g >= thr].reshape(-1, 3).mean(0)


def ctx_of(sq):
    return {"airlight": airlight_of(sq), "lum_mean_512": mean_grey(sq)}


def render_cell(sq, ctx, k, h, fam):
    """One cell of the stimulus grid, in MANIP_ORDER."""
    img = sq
    for f in MANIP_ORDER:
        img = apply(f if f == "chroma" else fam, img,
                    k if f == "chroma" else h, ctx)
    return img


# ============================================================
# Small helpers
# ============================================================

def label_tag(name):
    """A level label safe for a filename: 'k=1' -> 'k1'."""
    return str(name).replace("=", "").replace("/", "_").replace(" ", "")


def indoor_of(category):
    if category in INDOOR_CATEGORIES:
        return "indoor"
    if category in AMBIGUOUS_CATEGORIES:
        return "ambiguous"
    return "outdoor"


def resolve_output_dir():
    """OUTPUT_ROOT plus the fog family, never overwriting a build."""
    base = f"{OUTPUT_ROOT}_{FOG_FAMILY}"

    def occupied(d):
        return os.path.isdir(d) and any(
            os.path.isfile(os.path.join(d, f))
            for f in ("stimulus_manifest.csv", "grid_cells.csv",
                      "build_report.txt"))

    if not occupied(base):
        return base
    if ON_EXISTING == "error":
        raise FileExistsError(
            f"{base} already holds a build. Move or rename it, or set "
            "ON_EXISTING = 'new'.")
    n = 2
    while occupied(f"{base}_{n}"):
        n += 1
    print(f"  NOTE: {base} already holds a build; writing to "
          f"{base}_{n} instead.")
    return f"{base}_{n}"


# ============================================================
# Loading
# ============================================================

def load_levels(report):
    """The ladder: (label, parameter) per colour and fog level."""
    if not os.path.isfile(LEVELS_CSV):
        raise FileNotFoundError(f"{LEVELS_CSV} not found. Run step 1.")
    L = pd.read_csv(LEVELS_CSV)

    def pick(dim, names):
        out = []
        for nm in names:
            m = L[(L.dimension == dim) & (L.level == nm)]
            if not len(m):
                raise ValueError(
                    f"{LEVELS_CSV} has no {dim} level {nm!r}. The fog "
                    f"family is {FOG_FAMILY!r}, which reads the "
                    f"{FOG_DIMENSION!r} block -- is step 1's "
                    "INCLUDE_HAZE_EQ on?")
            r = m.iloc[0]
            if not np.isfinite(r.param):
                raise ValueError(f"{dim} level {nm} has no parameter.")
            if r.status != "ok":
                warnings.warn(f"{dim} level {nm} is {r.status!r}.")
            out.append((nm, float(r.param), float(r.target_change)))
        return out

    colour = pick("colour", COLOUR_LEVELS)
    if INCLUDE_UNMANIPULATED_COLOUR:
        colour = [("k=1", 1.0, 0.0)] + colour
    fog = pick(FOG_DIMENSION, FOG_LEVELS)

    print(f"  ladder: {len(colour)} colour x {len(fog)} fog "
          f"({FOG_FAMILY})")
    print("    colour " + ", ".join(f"{n}={p:.4f}" for n, p, _ in colour))
    print("    fog    " + ", ".join(f"{n}={p:.4f}" for n, p, _ in fog))
    report.append(("ladder",
                   f"{len(colour)} x {len(fog)}, {FOG_FAMILY}"))
    return colour, fog


def load_envelope(report):
    """Corpus 1-99% range per feature, for CHECK 2."""
    if not os.path.isfile(STATS_CSV):
        print(f"  {STATS_CSV} not found; the envelope check is skipped.")
        report.append(("check 2 envelope", "skipped (no stats)"))
        return None
    S = pd.read_csv(STATS_CSV)
    if "population" in S.columns:
        sub = S[S.population == STATS_POPULATION]
        if not len(sub):
            avail = sorted(S.population.unique())
            raise ValueError(
                f"{STATS_CSV} has no population {STATS_POPULATION!r}. "
                f"Available: {avail}")
        S = sub
    S = S.set_index("feature")
    for c in (ENVELOPE_LO, ENVELOPE_HI, "sd"):
        if c not in S.columns:
            raise ValueError(f"{STATS_CSV} has no {c!r} column.")
    print(f"  envelope from the {STATS_POPULATION!r} population, "
          f"{ENVELOPE_LO}-{ENVELOPE_HI}")
    return S


def load_screened(report):
    """Pool the kept rows from every screening list.

    Only the identity columns are taken. The measured columns are
    deliberately ignored: different screening rounds may have used
    different settings, so those numbers are not comparable across
    files. Everything is recomputed below under the current ladder.
    """
    print("\n" + "=" * 60)
    print("LOADING SCREENED SCENES")
    print("=" * 60)

    frames = []
    for path in SCREENING_LISTS:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Screening list not found: {path}")
        d = pd.read_csv(path)
        for c in ("keep", "path", "category"):
            if c not in d.columns:
                raise ValueError(f"{path} has no {c!r} column.")
        kept = d[pd.to_numeric(d["keep"], errors="coerce") == 1].copy()
        print(f"  {os.path.basename(path):34s} "
              f"{len(kept):4d} kept of {len(d)}")
        keep_cols = [c for c in ("scene_id", "path", "category")
                     if c in kept.columns]
        kept = kept[keep_cols]
        kept["source_list"] = os.path.basename(path)
        frames.append(kept)

    scenes = pd.concat(frames, ignore_index=True)
    before = len(scenes)
    scenes = scenes.drop_duplicates(subset="path").reset_index(drop=True)
    if before != len(scenes):
        print(f"  {before - len(scenes)} scene(s) kept in more than one "
              "round; counted once.")

    if "scene_id" not in scenes.columns:
        scenes["scene_id"] = [
            r.category.replace("/", "_") + "_"
            + os.path.splitext(os.path.basename(r.path))[0]
            for r in scenes.itertuples()]

    missing = [p for p in scenes.path if not os.path.isfile(p)]
    if missing:
        print(f"  WARNING: {len(missing)} file(s) not on disk; dropped. "
              f"First: {missing[0]}")
        scenes = scenes[scenes.path.apply(os.path.isfile)]
        scenes = scenes.reset_index(drop=True)
    if not len(scenes):
        raise RuntimeError(
            "No screened scenes survived. Check that `keep` is 1 for "
            "some rows and that the `path` column points at files that "
            "exist from the directory this script runs in.")

    # group, for reporting and optional balancing
    if os.path.isfile(CATEGORY_CSV):
        cat = pd.read_csv(CATEGORY_CSV)
        if {"category", "group"} <= set(cat.columns):
            scenes = scenes.merge(cat[["category", "group"]],
                                  on="category", how="left")
    if "group" not in scenes.columns:
        scenes["group"] = "unknown"
    scenes["group"] = scenes["group"].fillna("unknown")
    scenes["indoor"] = scenes.category.map(indoor_of)

    print(f"\n  {len(scenes)} unique scenes, "
          f"{scenes.category.nunique()} categories")
    print("  by group : " + ", ".join(
        f"{g} {n}" for g, n in scenes.group.value_counts().items()))
    print("  in/out   : " + ", ".join(
        f"{g} {n}" for g, n in scenes.indoor.value_counts().items()))
    worst = scenes.category.value_counts()
    print(f"  most from one category: {worst.iloc[0]} "
          f"({worst.index[0]})")
    report.append(("scenes screened", str(len(scenes))))
    report.append(("categories", str(scenes.category.nunique())))
    return scenes


# ============================================================
# CHECK 1: the copied feature code
# ============================================================

def check_feature_code(rng, report):
    """CHECK 1: the copied feature code reproduces data/features.

    Everything downstream -- the levels, the pair classification, the
    envelope -- is expressed in these units, so a drift between this
    copy and the corpus features would be silent and total.
    """
    if not (os.path.isfile(INDEX_CSV) and os.path.isfile(FEATURES_NPY)):
        print("  check 1: data/features not available; SKIPPED.")
        report.append(("check 1 feature code", "skipped"))
        return
    idx = pd.read_csv(INDEX_CSV)
    F = np.load(FEATURES_NPY, mmap_mode="r")
    pick = rng.choice(len(idx), size=FEATURE_CHECK_N * 3, replace=False)
    worst, used = 0.0, 0
    for i in pick:
        if used >= FEATURE_CHECK_N:
            break
        try:
            with Image.open(idx.path.iat[int(i)]) as im:
                sq = np.asarray(central_square(im.convert("RGB"),
                                               FEATURE_SIZE), np.uint8)
        except Exception:
            continue
        got = np.asarray(all_features(sq), np.float64)
        ref = np.asarray(F[int(i)], np.float64)
        m = np.isfinite(ref) & (np.abs(ref) > 1e-9)
        worst = max(worst, float(np.abs(got[m] - ref[m]).max()
                                 / np.abs(ref[m]).max()))
        used += 1
    ok = used > 0 and worst <= FEATURE_CHECK_RTOL
    print(f"  check 1 (feature code vs data/features, {used} images): "
          f"max relative difference {worst:.2e} -> "
          f"{'PASS' if ok else 'FAIL'}")
    report.append(("check 1 feature code",
                   f"{'PASS' if ok else 'FAIL'} ({worst:.1e})"))
    if not ok:
        raise RuntimeError("The copied feature code does not reproduce "
                           "data/features.")


# ============================================================
# Measure the grid
# ============================================================

W = None


def _init(payload):
    global W
    W = payload
    cv2.setNumThreads(0)   # see the note beside the imports
    warnings.filterwarnings("ignore")


def _grid_task(row):
    """Render and measure every cell of one scene's grid.

    All features, not just the two the pairs use: the envelope check
    needs the luminance set, and the manifest carries the rest.
    """
    sid, path = row
    colour, fog, fam = W["colour"], W["fog"], W["fam"]
    try:
        with Image.open(path) as im:
            sq = np.asarray(central_square(im.convert("RGB"), IMAGE_SIZE),
                            np.uint8)
    except Exception as exc:
        return sid, None, f"{type(exc).__name__}: {exc}"
    ctx = ctx_of(sq)
    base = measure(sq)
    cells = []
    for cn, k, _ in colour:
        for fn, h, _ in fog:
            v = measure(render_cell(sq, ctx, k, h, fam))
            cells.append({"scene_id": sid, "colour_level": cn,
                          "fog_level": fn, "chroma": k, "haze": h, **v})
    return sid, (base, cells), None


def measure_grid(scenes, colour, fog, report):
    """Build every scene's grid and measure every cell."""
    print("\n" + "=" * 60)
    print("MEASURING THE GRID")
    print("=" * 60)
    n_cells = len(colour) * len(fog)
    print(f"  {len(scenes)} scenes x {n_cells} cells = "
          f"{len(scenes) * n_cells:,} measurements")

    payload = {"colour": colour, "fog": fog, "fam": FOG_FAMILY}
    rows = list(zip(scenes.scene_id, scenes.path))
    t0 = time.time()
    results = []
    if N_WORKERS > 1 and len(rows) > 1:
        with Pool(N_WORKERS, initializer=_init,
                  initargs=(payload,)) as pool:
            for k, out in enumerate(
                    pool.imap_unordered(_grid_task, rows, CHUNK), 1):
                results.append(out)
                print(f"    {k}/{len(rows)}", end="\r")
    else:
        _init(payload)
        for k, r in enumerate(rows, 1):
            results.append(_grid_task(r))
            print(f"    {k}/{len(rows)}", end="\r")
    print(f"\n  elapsed {time.time() - t0:.0f}s")

    meta = scenes.set_index("scene_id")
    cells, bases, bad = [], [], []
    for sid, got, err in results:
        if got is None:
            bad.append((sid, err))
            continue
        base, cl = got
        bases.append({"scene_id": sid, **base})
        cells.extend(cl)
    if bad:
        print(f"  WARNING: {len(bad)} scene(s) could not be read; "
              f"first: {bad[0][0]} ({bad[0][1]})")
    if not cells:
        raise RuntimeError("No scenes could be rendered.")

    grid = pd.DataFrame(cells)
    for c in ("category", "group", "indoor"):
        grid[c] = grid.scene_id.map(meta[c])
    grid["path"] = grid.scene_id.map(meta["path"])
    base_df = pd.DataFrame(bases).set_index("scene_id")

    print(f"  {grid.scene_id.nunique()} scenes measured, "
          f"{len(grid):,} cells")
    report.append(("cells measured", f"{len(grid):,}"))
    return grid, base_df


# ============================================================
# CHECK 2: the envelope, on the combined cells
# ============================================================

def check_envelope(grid, env, report):
    """CHECK 2: does any CELL take a feature the design does not intend
    to move outside the corpus 1-99% range?

    Step 1 could only ask this of one manipulation at a time. A cell
    applies both, and the two do not simply add: fog lifts the black
    point while chroma drains colour, and whether the combination lands
    somewhere the corpus never goes is a question only the real grid
    can answer.

    This is reported per level, not just counted, because the answer
    the build needs is "from which rung does it start" rather than
    "how many".
    """
    print("\n" + "=" * 60)
    print("ENVELOPE  (CHECK 2)")
    print("=" * 60)
    if env is None:
        print("  skipped: no corpus stats.")
        return pd.DataFrame(), True

    feats = [f for f in ENVELOPE_FEATURES if f in grid.columns
             and f in env.index]
    rows = []
    for f in feats:
        lo = float(env.at[f, ENVELOPE_LO])
        hi = float(env.at[f, ENVELOPE_HI])
        sd = float(env.at[f, "sd"]) or 1.0
        v = grid[f].to_numpy(float)
        out = (v < lo) | (v > hi)
        if not out.any():
            continue
        sub = grid.loc[out, ["scene_id", "colour_level", "fog_level"]]
        excess = np.where(v[out] > hi, v[out] - hi, v[out] - lo) / sd
        rows.append(pd.DataFrame({
            "feature": f,
            "scene_id": sub.scene_id.to_numpy(),
            "colour_level": sub.colour_level.to_numpy(),
            "fog_level": sub.fog_level.to_numpy(),
            "value": v[out],
            "corpus_lo": lo, "corpus_hi": hi,
            "outside_by_sd": excess,
        }))

    viol = (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["feature", "scene_id",
                                       "colour_level", "fog_level",
                                       "value", "corpus_lo",
                                       "corpus_hi", "outside_by_sd"]))
    frac = len(viol) / float(len(grid) * max(len(feats), 1))
    print(f"  {len(feats)} features checked over {len(grid):,} cells")
    if not len(viol):
        print("  no cell leaves the corpus range on any of them.")
    else:
        print(f"  {len(viol):,} (feature, cell) pairs outside "
              f"({frac:.1%} of those checked)\n")
        by_f = (viol.groupby("feature")
                .agg(cells=("value", "size"),
                     worst_sd=("outside_by_sd",
                               lambda s: s.abs().max()))
                .sort_values("cells", ascending=False))
        print("  by feature:")
        print(by_f.round(2).to_string())
        # which rung it starts at -- the number the build acts on
        print("\n  share of cells outside, by fog rung:")
        tab = (viol.groupby(["feature", "fog_level"]).size()
               .unstack(fill_value=0))
        per_rung = grid.groupby("fog_level").size()
        print((tab / per_rung).reindex(columns=per_rung.index)
              .fillna(0).round(2).to_string())

    ok = frac <= ENVELOPE_MAX_CELL_FRAC
    print(f"\n  RESULT: {'PASS' if ok else 'NOTE'} "
          f"({frac:.1%} of checks outside, "
          f"threshold {ENVELOPE_MAX_CELL_FRAC:.0%})")
    if not ok:
        print("  The deepest rungs are meant to be extreme, so this is "
              "a flag to look at, not a failure. v2 shipped with 95% "
              "of its h = 0.49 images above the corpus p99 for "
              "lum_p05. Drop the deepest rung if you want the set to "
              "stay inside.")
    report.append(("check 2 envelope",
                   f"{'PASS' if ok else 'NOTE'} ({frac:.1%} outside)"))
    return viol, ok


# ============================================================
# CHECK 3: do the two fog families match at matched rungs?
# ============================================================

def _equiv_task(row):
    """Both fog families at every rung, for one scene."""
    sid, path = row
    fog_a, fog_b = W["fog_a"], W["fog_b"]
    try:
        with Image.open(path) as im:
            sq = np.asarray(central_square(im.convert("RGB"), IMAGE_SIZE),
                            np.uint8)
    except Exception:
        return None
    ctx = ctx_of(sq)
    base = measure(sq, [FEAT_FOG])[FEAT_FOG]
    out = []
    for (nm, pa, _), (_, pb, _) in zip(fog_a, fog_b):
        va = measure(render_cell(sq, ctx, 1.0, pa, "haze"),
                     [FEAT_FOG])[FEAT_FOG]
        vb = measure(render_cell(sq, ctx, 1.0, pb, "haze_eq"),
                     [FEAT_FOG])[FEAT_FOG]
        out.append({"scene_id": sid, "level": nm,
                    "d_haze": base - va, "d_haze_eq": base - vb})
    return out


def check_fog_equivalence(scenes, report):
    """CHECK 3: at a matched rung, do haze and haze_eq remove the same
    amount of fine detail?

    This is the assumption that lets ONE scene list serve BOTH stimulus
    sets, so it is verified rather than asserted.

    SCALED BY THE DEEPEST RUNG'S CHANGE, not by the rung's own. Step 2
    divided by the per-rung change; at the lightest rung that is small,
    so a trivial absolute difference reads as a huge ratio, and the
    check warned at 1.117 and 0.840 on two samples of the same corpus
    -- a spread far larger than its own tolerance, which is what an
    unstable denominator looks like rather than a real difference. The
    deepest rung's change is the largest, most stable quantity in the
    comparison and is the same for every rung of a scene, so the
    numbers below are comparable down the ladder.
    """
    print("\n" + "=" * 60)
    print("FOG FAMILIES  (CHECK 3)")
    print("=" * 60)
    L = pd.read_csv(LEVELS_CSV)
    have = set(L.dimension.unique())
    if not {"fog", "fog_eq"} <= have:
        print("  only one fog family is calibrated; skipped.")
        report.append(("check 3 fog families", "skipped"))
        return pd.DataFrame(), True

    def pick(dim):
        out = []
        for nm in FOG_LEVELS:
            m = L[(L.dimension == dim) & (L.level == nm)]
            if not len(m):
                return None
            out.append((nm, float(m.iloc[0].param), 0.0))
        return out

    fog_a, fog_b = pick("fog"), pick("fog_eq")
    if fog_a is None or fog_b is None:
        print("  the two families do not share these rungs; skipped.")
        report.append(("check 3 fog families", "skipped"))
        return pd.DataFrame(), True

    sub = scenes.head(FOG_EQUIV_N)
    payload = {"fog_a": fog_a, "fog_b": fog_b}
    rows = list(zip(sub.scene_id, sub.path))
    recs = []
    if N_WORKERS > 1 and len(rows) > 1:
        with Pool(N_WORKERS, initializer=_init,
                  initargs=(payload,)) as pool:
            for out in pool.imap_unordered(_equiv_task, rows, CHUNK):
                if out:
                    recs.extend(out)
    else:
        _init(payload)
        for r in rows:
            out = _equiv_task(r)
            if out:
                recs.extend(out)

    if not recs:
        print("  no scene could be measured; skipped.")
        report.append(("check 3 fog families", "skipped"))
        return pd.DataFrame(), True

    d = pd.DataFrame(recs)
    # the deepest rung's change, per scene: the stable denominator
    deep = (d[d.level == FOG_LEVELS[-1]]
            .set_index("scene_id")["d_haze"].abs())
    d["scale"] = d.scene_id.map(deep)
    d = d[d.scale > 1e-9]
    d["rel"] = (d.d_haze_eq - d.d_haze) / d.scale

    tab = (d.groupby("level")
           .agg(n=("rel", "size"),
                median_rel=("rel", "median"),
                p90_abs_rel=("rel", lambda s: s.abs().quantile(0.90)),
                worst_abs_rel=("rel", lambda s: s.abs().max()),
                mean_d_haze=("d_haze", "mean"),
                mean_d_haze_eq=("d_haze_eq", "mean"))
           .reindex(FOG_LEVELS))
    print(f"  {d.scene_id.nunique()} scenes, each rung of both families")
    print("  rel = (haze_eq change - haze change) / this scene's change "
          "at the deepest rung\n")
    print(tab.round(4).to_string())

    worst = float(tab.worst_abs_rel.max())
    ok = worst <= FOG_EQUIV_TOL
    print(f"\n  RESULT: {'PASS' if ok else 'NOTE'} "
          f"(worst {worst:.3f}, tolerance {FOG_EQUIV_TOL})")
    if not ok:
        print("  The two families do not remove quite the same amount "
              "at every rung. Read the per-rung medians above: a small "
              "median with a large worst case is a few scenes, not a "
              "systematic offset, and one scene list still serves both "
              "sets. A large median at a given rung is systematic and "
              "means that rung is not matched between the two.")
    report.append(("check 3 fog families",
                   f"{'PASS' if ok else 'NOTE'} ({worst:.3f})"))
    return tab.reset_index(), ok


# ============================================================
# CHECK 4: trial-type affordance
# ============================================================

TYPES = ["sat_only", "fog_only", "together", "conflict"]


def classify_pairs(g):
    """Classify every ordered within-scene pair of cells.

    Differences come from MEASURED features, not from the parameters,
    because the mapping from parameter to feature is scene-dependent --
    fog drains colour by different amounts in different scenes.
    """
    g = g.reset_index(drop=True)
    sat = g[FEAT_SAT].to_numpy(float)
    fog = g[FEAT_FOG].to_numpy(float)
    fl = g["fog_level"].to_numpy()
    cl = g["colour_level"].to_numpy()
    out = []
    for i, j in itertools.permutations(range(len(g)), 2):
        d_sat = sat[i] - sat[j]
        # FOG_SIGN keeps "positive means A is foggier" true whichever
        # feature names the fog axis. The test below depends on it.
        d_fog = FOG_SIGN * (fog[i] - fog[j])

        if fl[i] == fl[j]:
            # same fog row: d_fog is ~0 by construction, because chroma
            # scaling leaves L* alone
            t = "sat_only" if abs(d_sat) >= MIN_DSAT_SAT_ONLY else None
        elif (abs(d_sat) <= PAIR_SAT_TOL
              and abs(d_fog) >= MIN_DFOG_FOG_ONLY):
            t = "fog_only"
        elif (abs(d_fog) >= MIN_DFOG_MIXED
              and abs(d_sat) >= MIN_DSAT_MIXED):
            # A is the foggier image when d_fog > 0. If it is ALSO more
            # saturated (d_sat > 0) the cues disagree: conflict.
            t = "conflict" if (d_fog > 0) == (d_sat > 0) else "together"
        else:
            t = None

        if t:
            out.append({"i": i, "j": j, "type": t,
                        "colour_a": cl[i], "fog_a": fl[i],
                        "colour_b": cl[j], "fog_b": fl[j],
                        "d_sat": d_sat, "d_fog": d_fog})
    return pd.DataFrame(out)


def check_affordance(grid, out_dir, report):
    """CHECK 4: count usable pairs of each type, per scene.

    Conflict pairs are the binding constraint in practice: they need an
    image that is foggier AND more saturated, which the physics
    resists.
    """
    print("\n" + "=" * 60)
    print("TRIAL-TYPE AFFORDANCE  (CHECK 4)")
    print("=" * 60)

    rows, all_pairs = [], []
    for sid, g in grid.groupby("scene_id"):
        pairs = classify_pairs(g)
        counts = {t: int((pairs.type == t).sum()) if len(pairs) else 0
                  for t in TYPES}
        rows.append({"scene_id": sid,
                     "category": g.category.iloc[0],
                     "group": g.group.iloc[0],
                     "indoor": g.indoor.iloc[0],
                     "path": g.path.iloc[0],
                     **counts,
                     "sat_range": float(g[FEAT_SAT].max()
                                        - g[FEAT_SAT].min()),
                     "fog_range": float(g[FEAT_FOG].max()
                                        - g[FEAT_FOG].min())})
        if len(pairs):
            pairs = pairs.copy()
            pairs["scene_id"] = sid
            all_pairs.append(pairs)

    aff = pd.DataFrame(rows)
    aff["passed"] = (aff[TYPES] >= MIN_PAIRS_PER_TYPE).all(axis=1)
    pairs = (pd.concat(all_pairs, ignore_index=True)
             if all_pairs else pd.DataFrame(columns=["type"]))

    print("  usable pairs per scene:")
    print(aff[TYPES].describe().loc[
        ["min", "25%", "50%", "75%", "max"]].round(1).to_string())
    print()
    for t in TYPES:
        n_fail = int((aff[t] < MIN_PAIRS_PER_TYPE).sum())
        print(f"    {t:10s} below {MIN_PAIRS_PER_TYPE}: "
              f"{n_fail}/{len(aff)} scenes")

    n_ok = int(aff.passed.sum())
    print(f"\n  scenes affording all four types: {n_ok}/{len(aff)}")
    if n_ok < len(aff):
        dropped = aff[~aff.passed].copy()
        dropped["why"] = [
            "too few: " + ", ".join(t for t in TYPES
                                    if r[t] < MIN_PAIRS_PER_TYPE)
            for _, r in dropped.iterrows()]
        dropped.to_csv(os.path.join(out_dir, "dropped_scenes.csv"),
                       index=False)
        print("  dropped: " + ", ".join(
            f"{r.scene_id} ({r.category})"
            for r in dropped.head(8).itertuples()))
        if len(dropped) > 8:
            print(f"           ... and {len(dropped) - 8} more; see "
                  "dropped_scenes.csv")

    need = N_SELECT if N_SELECT else 1
    ok = n_ok >= need
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  Fewer than {need} scenes support every trial type. "
              "Loosen the MIN_D* thresholds, or screen more scenes.")
    report.append(("scenes affording all types", f"{n_ok}/{len(aff)}"))
    report.append(("check 4 affordance", "PASS" if ok else "FAIL"))
    return aff, pairs, ok


# ============================================================
# CHECK 5: magnitude coverage
# ============================================================

def check_coverage(pairs, aff, out_dir, report):
    """CHECK 5: does the set produce small, medium AND large
    differences?

    The dose-response analysis splits trials by difference size and
    asks whether people choose the manipulated image more often when
    the difference is larger. A rising curve is much stronger evidence
    than a single proportion, because a response bias would give a flat
    line -- but only if every bin is populated.
    """
    print("\n" + "=" * 60)
    print("MAGNITUDE COVERAGE  (CHECK 5)")
    print("=" * 60)
    if not len(pairs):
        print("  no usable pairs; skipped.")
        report.append(("check 5 coverage", "skipped"))
        return pd.DataFrame(), False

    keep = set(aff.loc[aff.passed, "scene_id"])
    p = pairs[pairs.scene_id.isin(keep)].copy()

    rows = []
    for t, feat in [("sat_only", "d_sat"), ("fog_only", "d_fog"),
                    ("together", "d_fog"), ("conflict", "d_fog")]:
        sub = p[p.type == t]
        if not len(sub):
            rows.append({"type": t, "bin": 0, "lo": np.nan,
                         "hi": np.nan, "n_pairs": 0, "ok": False})
            continue
        mag = sub[feat].abs()
        edges = np.quantile(mag, np.linspace(0, 1, N_MAGNITUDE_BINS + 1))
        for b in range(N_MAGNITUDE_BINS):
            n = int(((mag >= edges[b]) & (mag <= edges[b + 1])).sum())
            rows.append({"type": t, "bin": b + 1,
                         "lo": round(float(edges[b]), 2),
                         "hi": round(float(edges[b + 1]), 2),
                         "n_pairs": n,
                         "ok": n >= MIN_PAIRS_PER_BIN})

    cov = pd.DataFrame(rows)
    print(cov.to_string(index=False))
    ok = bool(cov.ok.all())
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(>= {MIN_PAIRS_PER_BIN} pairs per bin)")
    if not ok:
        print("  Some bins are thin, so the dose-response curve will "
              "be noisy there. Adding scenes helps more than loosening "
              "thresholds.")
    report.append(("check 5 coverage", "PASS" if ok else "FAIL"))
    _plot_coverage(p, aff, out_dir)
    return cov, ok


def _plot_coverage(pairs, aff, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for t in TYPES:
        sub = pairs[pairs.type == t]
        if len(sub):
            axes[0].scatter(sub.d_sat, sub.d_fog, s=4, alpha=0.25,
                            label=t)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].axvline(0, color="k", lw=0.6)
    axes[0].set_xlabel("d sat_mean  (A - B)")
    axes[0].set_ylabel(f"d {FEAT_FOG}  (A - B, signed so + is foggier)")
    axes[0].set_title("Usable pairs by trial type\n"
                      "(the four types occupy different regions)")
    axes[0].legend(fontsize=8, markerscale=3)
    axes[0].grid(alpha=0.3)

    ok = aff[aff.passed]
    order = ok.sort_values("conflict").reset_index(drop=True)
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    for t in TYPES:
        axes[1].bar(x, order[t], bottom=bottom, label=t)
        bottom += order[t].to_numpy()
    axes[1].set_xlabel("scenes (sorted by conflict-pair count)")
    axes[1].set_ylabel("usable pairs")
    axes[1].set_title("Affordance per scene\n"
                      "(conflict pairs are the scarce type)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "coverage.png"), dpi=150)
    plt.close(fig)


# ============================================================
# Final selection
# ============================================================

def select_scenes(aff, rng, report):
    """Pick the final set.

    With N_SELECT = 0 every scene that passed is kept: the hand
    screening has already chosen, and the second visual pass on the
    grid sheets will take a few more.

    When N_SELECT trims, scenes are ranked within each stratum of
    BALANCE_BY and taken round-robin, so the set does not end up mostly
    outdoor. Interiors carry more cultural marking and screen out more
    often, and derealization is typically reported indoors, so an
    unbalanced draw drifts away from the phenomenon.
    """
    print("\n" + "=" * 60)
    print("FINAL SELECTION")
    print("=" * 60)

    ok = aff[aff.passed].copy()
    if not N_SELECT or N_SELECT >= len(ok):
        final = ok.reset_index(drop=True)
        print(f"  keeping all {len(final)} scenes that passed")
    else:
        if SELECTION_RULE == "conflict":
            ok = ok.sort_values("conflict", ascending=False)
        elif SELECTION_RULE == "random":
            ok = ok.sample(frac=1.0, random_state=RANDOM_SEED)
        else:
            raise ValueError("SELECTION_RULE must be 'conflict' or "
                             f"'random', got {SELECTION_RULE!r}")
        if BALANCE_BY:
            if BALANCE_BY not in ok.columns:
                raise ValueError(
                    f"BALANCE_BY = {BALANCE_BY!r} is not a column; "
                    "use 'indoor', 'group' or None.")
            # round-robin: the 1st of each stratum, then the 2nd, ...
            ok["rank_in_stratum"] = ok.groupby(BALANCE_BY).cumcount()
            ok = ok.sort_values(["rank_in_stratum"], kind="stable")
        final = ok.head(N_SELECT).reset_index(drop=True)
        print(f"  {len(final)} of {int(aff.passed.sum())} passing "
              f"scenes, balanced by {BALANCE_BY!r}")

    print(f"  {final.category.nunique()} categories, "
          f"most from one: {final.category.value_counts().iloc[0]}")
    print("  by group : " + ", ".join(
        f"{g} {n}" for g, n in final.group.value_counts().items()))
    print("  in/out   : " + ", ".join(
        f"{g} {n}" for g, n in final.indoor.value_counts().items()))
    print("\n  usable pairs per scene in the final set:")
    print(final[TYPES].describe().loc[
        ["min", "50%", "max"]].round(1).to_string())

    report.append(("final scenes", str(len(final))))
    report.append(("final categories", str(final.category.nunique())))
    ind = final.indoor.value_counts()
    report.append(("final indoor/outdoor",
                   f"{ind.get('indoor', 0)}/{ind.get('outdoor', 0)}"
                   f" (+{ind.get('ambiguous', 0)} ambiguous)"))
    return final


def write_balance(final, out_dir):
    """The balance tables, as a file and a figure."""
    frames = []
    for col in ("indoor", "group", "category"):
        t = final[col].value_counts().rename_axis("value").reset_index(
            name="n_scenes")
        t.insert(0, "dimension", col)
        frames.append(t)
    bal = pd.concat(frames, ignore_index=True)
    bal.to_csv(os.path.join(out_dir, "balance.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, col, title in zip(
            axes, ["indoor", "group", "category"],
            ["Indoor / outdoor", "Group", "Category"]):
        v = final[col].value_counts()
        if col == "category":
            v = v.head(20)
        ax.barh(v.index[::-1].astype(str), v.to_numpy()[::-1])
        ax.set_title(title + ("  (top 20)" if col == "category" else ""))
        ax.grid(alpha=0.3, axis="x")
        ax.tick_params(labelsize=8)
    fig.suptitle(f"Balance of the final set  "
                 f"({len(final)} scenes, {FOG_FAMILY})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "balance.png"), dpi=150)
    plt.close(fig)
    return bal


# ============================================================
# Grid sheets  (the second screening pass looks at these)
# ============================================================

def _sheet_task(row):
    """One contact sheet: the whole colour x fog grid for one scene."""
    sid, path, out_path = row
    colour, fog, fam = W["colour"], W["fog"], W["fam"]
    try:
        with Image.open(path) as im:
            sq = np.asarray(central_square(im.convert("RGB"), IMAGE_SIZE),
                            np.uint8)
    except Exception as exc:
        return sid, f"{type(exc).__name__}: {exc}"
    ctx = ctx_of(sq)

    tile = SHEET_TILE_PX
    # the title gets a band of its own: scene ids are long enough to
    # run into the first column's label otherwise
    pad, title_h, head_h, left = 3, 20, 18, 54
    top = title_h + head_h
    W_px = left + len(colour) * (tile + pad) + pad
    H_px = top + len(fog) * (tile + pad) + pad
    sheet = Image.new("RGB", (W_px, H_px), (28, 28, 30))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), f"{sid}   [{FOG_FAMILY}]  colour across, "
                      "fog down", fill=(230, 230, 235))

    for ci, (cn, k, _) in enumerate(colour):
        x = left + ci * (tile + pad) + pad
        draw.text((x + 2, title_h + 4), str(cn), fill=(190, 190, 198))
        for fi, (fn, h, _) in enumerate(fog):
            y = top + fi * (tile + pad) + pad
            if ci == 0:
                draw.text((6, y + tile // 2 - 4), str(fn),
                          fill=(190, 190, 198))
            img = Image.fromarray(render_cell(sq, ctx, k, h, fam))
            sheet.paste(img.resize((tile, tile), Image.LANCZOS), (x, y))

    sheet.save(out_path, quality=SHEET_QUALITY)
    return sid, None


def write_sheets(final, colour, fog, out_dir, report):
    """A contact sheet per scene, plus a worksheet for the page.

    This is what the second cull looks at. One sheet per scene rather
    than 25 separate images: at this stage the question is whether the
    scene survives the manipulation -- blown highlights, posterising,
    the scene going unrecognisable at the deep end -- which the whole
    ladder answers at a glance and single images do not.
    """
    print("\n" + "=" * 60)
    print("GRID SHEETS")
    print("=" * 60)
    sheet_dir = os.path.join(out_dir, "sheets")
    os.makedirs(sheet_dir, exist_ok=True)

    rows = [(r.scene_id, r.path,
             os.path.join(sheet_dir, f"{r.scene_id}.jpg"))
            for r in final.itertuples()]
    payload = {"colour": colour, "fog": fog, "fam": FOG_FAMILY}
    t0 = time.time()
    bad = []
    if N_WORKERS > 1 and len(rows) > 1:
        with Pool(N_WORKERS, initializer=_init,
                  initargs=(payload,)) as pool:
            for k, (sid, err) in enumerate(
                    pool.imap_unordered(_sheet_task, rows, CHUNK), 1):
                if err:
                    bad.append((sid, err))
                print(f"    {k}/{len(rows)}", end="\r")
    else:
        _init(payload)
        for k, r in enumerate(rows, 1):
            sid, err = _sheet_task(r)
            if err:
                bad.append((sid, err))
            print(f"    {k}/{len(rows)}", end="\r")
    print(f"\n  {len(rows) - len(bad)} sheets in {sheet_dir} "
          f"({time.time() - t0:.0f}s)")
    if bad:
        print(f"  WARNING: {len(bad)} failed; first {bad[0]}")

    # worksheet, shaped so make_screening_page.py can read it directly
    ws = final[["scene_id", "category", "group", "indoor", "path"]].copy()
    ws.insert(0, "n", np.arange(1, len(ws) + 1))
    for t in TYPES:
        ws[t] = final[t].to_numpy()
    ws["crop"] = ["sheets/" + s + ".jpg" for s in ws.scene_id]
    ws["keep"] = ""
    ws["reason"] = ""
    ws.to_csv(os.path.join(out_dir, "screening_list.csv"), index=False)
    print(f"  worksheet: {os.path.join(out_dir, 'screening_list.csv')}")
    report.append(("grid sheets", str(len(rows) - len(bad))))


# ============================================================
# Write the stimuli
# ============================================================

def _stimulus_task(row):
    """Render and save every cell of one scene."""
    sid, path, img_dir = row
    colour, fog, fam = W["colour"], W["fog"], W["fam"]
    try:
        with Image.open(path) as im:
            sq = np.asarray(central_square(im.convert("RGB"), IMAGE_SIZE),
                            np.uint8)
    except Exception as exc:
        return sid, None, f"{type(exc).__name__}: {exc}"
    ctx = ctx_of(sq)
    out = []
    for cn, k, _ in colour:
        for fn, h, _ in fog:
            name = (f"{sid}__c{label_tag(cn)}__f{label_tag(fn)}"
                    f".{STIMULUS_FORMAT}")
            img = Image.fromarray(render_cell(sq, ctx, k, h, fam))
            img.save(os.path.join(img_dir, name))
            out.append({"scene_id": sid, "colour_level": cn,
                        "fog_level": fn, "filename": name})
    return sid, out, None


def write_stimuli(final, grid, colour, fog, out_dir, report):
    """Render and save every stimulus, and build the manifest.

    The manifest is what step 4 and step 5 read: one row per image,
    with the measured features that are the regression predictors.
    """
    print("\n" + "=" * 60)
    print("WRITING STIMULI")
    print("=" * 60)
    img_dir = os.path.join(out_dir, "stimuli")
    os.makedirs(img_dir, exist_ok=True)

    rows = [(r.scene_id, r.path, img_dir) for r in final.itertuples()]
    payload = {"colour": colour, "fog": fog, "fam": FOG_FAMILY}
    n_cells = len(colour) * len(fog)
    print(f"  {len(rows)} scenes x {n_cells} cells = "
          f"{len(rows) * n_cells:,} images")

    t0 = time.time()
    made, bad = [], []
    if N_WORKERS > 1 and len(rows) > 1:
        with Pool(N_WORKERS, initializer=_init,
                  initargs=(payload,)) as pool:
            for k, (sid, out, err) in enumerate(
                    pool.imap_unordered(_stimulus_task, rows, CHUNK), 1):
                if err:
                    bad.append((sid, err))
                else:
                    made.extend(out)
                print(f"    {k}/{len(rows)}", end="\r")
    else:
        _init(payload)
        for k, r in enumerate(rows, 1):
            sid, out, err = _stimulus_task(r)
            if err:
                bad.append((sid, err))
            else:
                made.extend(out)
            print(f"    {k}/{len(rows)}", end="\r")
    print(f"\n  elapsed {time.time() - t0:.0f}s")
    if bad:
        print(f"  WARNING: {len(bad)} scene(s) failed; first {bad[0]}")

    files = pd.DataFrame(made)
    meta = final.set_index("scene_id")
    manifest = grid.merge(files,
                          on=["scene_id", "colour_level", "fog_level"],
                          how="inner")
    manifest["fog_family"] = FOG_FAMILY
    manifest["source_path"] = manifest.scene_id.map(meta["path"])

    lead = ["scene_id", "category", "group", "indoor", "fog_family",
            "colour_level", "fog_level", "chroma", "haze", "filename",
            "source_path"]
    manifest = manifest[lead + [c for c in manifest.columns
                                if c not in lead and c != "path"]]
    manifest.to_csv(os.path.join(out_dir, "stimulus_manifest.csv"),
                    index=False)
    print(f"  {len(manifest):,} stimuli in {img_dir}")
    print(f"  manifest: {os.path.join(out_dir, 'stimulus_manifest.csv')}")
    report.append(("stimuli written", f"{len(manifest):,}"))
    return manifest


# ============================================================
# Report
# ============================================================

def write_report(out_dir, report, scenes, final, colour, fog,
                 env_viol, equiv, cov, t0, overall):
    path = os.path.join(out_dir, "build_report.txt")
    with open(path, "w") as f:
        def W_(s=""):
            f.write(s + "\n")

        W_("STIMULUS SET BUILD REPORT  (v4 axes, step 3)")
        W_("=" * 70)
        W_(f"Fog family    : {FOG_FAMILY}  (levels from "
           f"'{FOG_DIMENSION}')")
        W_(f"Render mode   : {RENDER_MODE}")
        W_(f"Screening     : {', '.join(os.path.basename(p) for p in SCREENING_LISTS)}")
        W_(f"Levels from   : {LEVELS_CSV}")
        W_("Ladder        : colour " +
           ", ".join(f"{n}={p:.4f}" for n, p, _ in colour))
        W_("                fog    " +
           ", ".join(f"{n}={p:.4f}" for n, p, _ in fog))
        W_(f"Output        : {out_dir}")
        W_(f"Elapsed       : {time.time() - t0:.1f}s")
        W_()
        for k, v in report:
            W_(f"  {k:32s} {v}")

        W_()
        W_("1. BALANCE OF THE FINAL SET")
        W_("-" * 70)
        W_("  The indoor/outdoor label is a judgement written into the")
        W_("  settings of this script, not a column of the corpus. It is")
        W_("  in the manifest so it can be used as a covariate.")
        W_()
        for col in ("indoor", "group"):
            v = final[col].value_counts()
            W_(f"  by {col}: " + ", ".join(f"{a} {b}"
                                           for a, b in v.items()))
        v = final.category.value_counts()
        W_(f"  categories: {len(v)}, most from one: {v.iloc[0]} "
           f"({v.index[0]})")
        W_("  " + ", ".join(f"{a} {b}" for a, b in v.head(12).items()))

        W_()
        W_("2. ENVELOPE  (CHECK 2)")
        W_("-" * 70)
        W_("  Cells taking a feature the design does not intend to move")
        W_("  outside the corpus 1-99% range. Step 1 could only ask this")
        W_("  of one manipulation at a time; these are the real cells,")
        W_("  with both applied.")
        W_()
        if env_viol is None or not len(env_viol):
            W_("  No cell leaves the corpus range. ")
        else:
            by_f = (env_viol.groupby("feature")
                    .agg(cells=("value", "size"),
                         worst_sd=("outside_by_sd",
                                   lambda s: s.abs().max()))
                    .sort_values("cells", ascending=False))
            W_(by_f.round(2).to_string())
            W_()
            W_("  Deepest rungs are MEANT to be extreme. v2 shipped with")
            W_("  95% of its h = 0.49 images above the corpus p99 for")
            W_("  lum_p05, so this is context for the write-up rather")
            W_("  than a fault -- unless you want the set to stay inside,")
            W_("  in which case drop the rung this starts at.")

        W_()
        W_("3. FOG FAMILIES  (CHECK 3)")
        W_("-" * 70)
        if equiv is None or not len(equiv):
            W_("  Skipped: only one family is calibrated.")
        else:
            W_("  rel = (haze_eq change - haze change) / this scene's")
            W_("  change at the deepest rung. Scaled this way, and not")
            W_("  by the rung's own change, because that denominator is")
            W_("  small at the light end and made step 2's version swing")
            W_("  between 0.84 and 1.12 on two samples of one corpus.")
            W_()
            W_(equiv.round(4).to_string(index=False))
            W_()
            W_("  A small median with a large worst case is a few")
            W_("  scenes, not an offset: one scene list still serves")
            W_("  both sets. A large median at a rung means that rung is")
            W_("  genuinely not matched between the families.")

        W_()
        W_("4. MAGNITUDE COVERAGE  (CHECK 5)")
        W_("-" * 70)
        if cov is None or not len(cov):
            W_("  Skipped.")
        else:
            W_(cov.to_string(index=False))

        W_()
        W_("5. WHAT TO DO NEXT")
        W_("-" * 70)
        if RENDER_MODE == "preview":
            W_("  This was the PREVIEW pass: no stimulus images were")
            W_("  written. Look at sheets/ -- one colour x fog grid per")
            W_("  scene. Point make_screening_page.py at this directory")
            W_("  (its screening_list.csv is already here) and cull on")
            W_("  how each scene survives the manipulation.")
            W_()
            W_("  Then put the saved list in SCREENING_LISTS, set")
            W_("  RENDER_MODE = 'full', and run again. Run once more")
            W_("  with FOG_FAMILY = 'haze_eq' for the second set.")
        else:
            W_("  stimulus_manifest.csv is what step 4 (analyse_pairs)")
            W_("  and step 5 (build_trials) read.")
            W_()
            W_("  Not done here: the nine-axis profile of the final set.")
            W_("  It needs the un2CLIP encoder, so it is its own step,")
            W_("  run once on the final set rather than on every build.")

        W_()
        W_("OVERALL: " + ("PASS" if overall else
                          "CHECK -- see the failing test above"))
    return path


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    rng = np.random.default_rng(RANDOM_SEED)

    if FOG_FAMILY not in ("haze", "haze_eq"):
        raise ValueError("FOG_FAMILY must be 'haze' or 'haze_eq', "
                         f"got {FOG_FAMILY!r}")
    if RENDER_MODE not in ("preview", "full"):
        raise ValueError("RENDER_MODE must be 'preview' or 'full', "
                         f"got {RENDER_MODE!r}")

    print("=" * 60)
    print("BUILD STIMULUS SET  (v4, step 3)")
    print("=" * 60)
    out_dir = resolve_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    print(f"  fog family : {FOG_FAMILY}")
    print(f"  mode       : {RENDER_MODE}")
    print(f"  output     : {out_dir}")

    report = []
    colour, fog = load_levels(report)
    env = load_envelope(report)
    check_feature_code(rng, report)

    scenes = load_screened(report)
    grid, base = measure_grid(scenes, colour, fog, report)
    grid.to_csv(os.path.join(out_dir, "grid_cells.csv"), index=False)

    env_viol, _ = check_envelope(grid, env, report)
    if len(env_viol):
        env_viol.to_csv(
            os.path.join(out_dir, "envelope_violations.csv"), index=False)

    equiv, _ = check_fog_equivalence(scenes, report)
    if len(equiv):
        equiv.to_csv(os.path.join(out_dir, "fog_equivalence.csv"),
                     index=False)

    aff, pairs, ok_aff = check_affordance(grid, out_dir, report)
    aff.to_csv(os.path.join(out_dir, "pair_affordance.csv"), index=False)
    if len(pairs):
        pairs.to_csv(os.path.join(out_dir, "pairs.csv"), index=False)
    if not ok_aff:
        print("\nAffordance check failed; stopping before selection.")
        write_report(out_dir, report, scenes, aff.head(0), colour, fog,
                     env_viol, equiv, None, t0, False)
        return

    cov, ok_cov = check_coverage(pairs, aff, out_dir, report)
    final = select_scenes(aff, rng, report)
    final.to_csv(os.path.join(out_dir, "stimulus_set.csv"), index=False)
    write_balance(final, out_dir)

    write_sheets(final, colour, fog, out_dir, report)

    if RENDER_MODE == "full":
        sel = grid[grid.scene_id.isin(set(final.scene_id))]
        write_stimuli(final, sel, colour, fog, out_dir, report)
    else:
        print("\n  RENDER_MODE is 'preview': no stimulus images "
              "written.")
        report.append(("stimuli written", "none (preview)"))

    path = write_report(out_dir, report, scenes, final, colour, fog,
                        env_viol, equiv, cov, t0, ok_aff and ok_cov)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in report:
        print(f"  {k:32s} {v}")
    print(f"\n  Report : {path}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    if RENDER_MODE == "preview":
        print("\n  NEXT: look at the sheets, cull with "
              "make_screening_page.py,")
        print("        then re-run with RENDER_MODE = 'full'.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
