"""
build_stimulus_set_v5.py  (v5 stimulus pipeline: the final stimulus set)

Builds the v5 stimulus set, in the form the v5 trial-design scripts read.
It plays the role v4's build_stimulus_set_v4_corrected.py played for v4.
No existing script or data file is modified.

The set
-------
47 scenes (the v4 set, all kept) x 49 versions = 2,303 images, 512 x 512
px PNG. Each scene is a 7 x 7 grid around its unedited photograph:

  contrast  c-3 ... c0 ... c+3   below: v4's brightness-equated haze
                                 (haze_eq); above: the same blend run
                                 backwards on RGB (dehaze_rgb, v4's
                                 m_contrast_up); at every contrast level
                                 a chroma factor restores the scene's
                                 sat_mean ("A4_haze_cc_rgb")
  colour    s-3 ... s0 ... s+3   v4's CIELAB chroma edit, both directions

Levels are equal steps in corpus SD (definition (b) of the test script),
solved per scene at +-1/3, +-2/3, +-1 SD of contrast_local and of
sat_mean, measured on the stimulus path (photograph -> 512 -> 256). The
colour edit follows the contrast edit (v4's MANIP_ORDER). There is no
clipping budget; clipping is measured and reported only. Why these
choices: DECISION_NOTE in SETTINGS.

How it is made
--------------
The levels and images are made by the code of test_edit_levels_v5.py,
copied verbatim (each block is marked), with the settings of the test
run the method was chosen on (4aa278d54d, check T0). Each scene's levels
are solved again from scratch, every image is rendered and saved, and:

  T1  every image must be BYTE-IDENTICAL to the same cell of that test
      run (images/A4_haze_cc_rgb/b_sd/), so the visual check and the
      CLIP results of the test run carry over unchanged;
  T2  every level parameter must equal the test run's.

The level solver is left exactly as in the test run, including its known
imprecision: the last colour correction of a contrast level is not
followed by a contrast re-solve, so some contrast levels miss their
target by a few hundredths of an SD. Check T3 reports every miss.

Every measurement is taken from the PNG as saved (check M2), as the
trial-design scripts will re-measure it.

Outputs (OUTPUT_ROOT/<settings hash>/, never overwritten)
---------------------------------------------------------
  stimuli/<scene>__c<k>__s<k>.png  the images (k = n3 ... 0 ... p3)
  stimulus_manifest.csv   one row per image: scene, levels, file name,
                          Places365 file ID, edit parameters, targets,
                          changes in SD, z_sat / z_con, distance from
                          typical (corpus and original; combined and per
                          feature), the 18 corpus features, 3 texture
                          measures, clipping, artefacts, envelope,
                          png_sha256; use = "pair" for all 49 cells
  cell_plan.csv           the 49 cells
  sources.csv             the 47 source photographs: Places365 file ID,
                          corpus image_key, size, SHA-256 (the photographs
                          themselves are not included)
  level_deviations.csv    every scene's 12 levels: target, achieved,
                          parameter, colour-correction factor
  cell_deviations.csv     each cell's distance from its nominal levels
  level_summary.csv       per level: measured change, cross-talk,
                          clipping, artefacts, distance from typical
  reach.csv, reach_grid.csv        the per-scene search the levels use
  reference_stimulus_path.csv      the 20,000-image corpus reference
  envelope_violations.csv, identity.csv, unedited.csv
  report.txt              start here; plain-language summary at the end
  checks.csv, settings.json, run_meta.json
  README.md, CITATION.cff (placeholders to complete)
  provenance/build_stimulus_set_v5.py   a copy of this script
  checksums.sha256, FROZEN.json         the freeze (relative paths only)

Usage
-----
    conda activate drl2drv
    python build_stimulus_set_v5.py

Run from the project folder; every path is relative to it. About 5-10
minutes on 20 workers if the test run's reference cache is present
(data/edit_levels_v5_cache), 10-15 minutes more without it; ~1 GB.
"""

import hashlib
import json
import os
import platform
import shutil
import time
import warnings
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageEnhance, ImageStat


# ============================================================
# SETTINGS
# ============================================================
_G0 = set(globals())

# ------------------------------------------------------------
# 1. THE FIXED CHOICES (decided on test run 4aa278d54d)
# ------------------------------------------------------------
# DECISION_NOTE is copied into settings.json, the README and the report.
DECISION_NOTE = (
    "Chosen after test_edit_levels_v5.py, run 4aa278d54d (committed as "
    "v5/stimulus_levels_data/4aa278d54d/), and a visual check of its "
    "viewer. Method A4_haze_cc_rgb: contrast down with v4's brightness-"
    "equated haze (haze_eq), contrast up with the same blend run "
    "backwards on RGB (v4's m_contrast_up, 'dehaze_rgb'), and at every "
    "contrast level a chroma factor that restores the scene's mean "
    "saturation, so the contrast levels leave saturation unchanged. "
    "Colour: v4's CIELAB chroma edit in both directions. Levels: "
    "definition (b), equal steps in corpus SD solved per scene, at "
    "+-1/3, +-2/3 and +-1 SD on both features, so the unedited "
    "original sits in the middle of both ladders and 'edited' points "
    "both ways. No clipping budget: the upward contrast levels clip "
    "(reported, not used to cap or exclude), and the +3 images were "
    "judged acceptable. All 47 v4 scenes, none set aside, for variety. "
    "No grid block in v5.")

APPROACH   = "A4_haze_cc_rgb"
DEFINITION = "b"                 # equal steps in corpus SD, per scene
B_SPAN_SD  = 1.0                 # levels at +-1/3, 2/3, 3/3 of this
LEVEL_FRACTIONS = [1 / 3, 2 / 3, 1.0]

# The approach exactly as the test script defined it (definitions are not
# needed here).
APPROACHES = {
    "A4_haze_cc_rgb": dict(contrast_down="haze_eq", contrast_up="dehaze_rgb",
                           colour="chroma", compensate=True),
}

# The 47 scenes: the v4 stimulus set, all kept.
N_SCENES_EXPECTED = 47
SCENES = [
    "alley_00001206", "apartment_building_outdoor_00004677",
    "auditorium_00001394", "balcony_interior_00004440",
    "bathroom_00001909", "bedroom_00001289", "boardwalk_00001503",
    "cafeteria_00002762", "campus_00003823", "campus_00004529",
    "computer_room_00001489", "conference_room_00002730",
    "courtyard_00004070", "crosswalk_00001791", "dining_hall_00001273",
    "dining_hall_00001301", "dining_room_00003076", "downtown_00001598",
    "downtown_00001745", "downtown_00004891", "driveway_00001180",
    "driveway_00003009", "elevator_lobby_00003261",
    "elevator_lobby_00003556", "gymnasium_indoor_00003333",
    "home_office_00004226", "japanese_garden_00000532",
    "japanese_garden_00001073", "japanese_garden_00001226",
    "laundromat_00004208", "living_room_00002633", "lobby_00001554",
    "park_00001014", "parking_garage_outdoor_00003825",
    "picnic_area_00000702", "picnic_area_00004871", "promenade_00002520",
    "residential_neighborhood_00001583", "restaurant_00000867",
    "shopping_mall_indoor_00003139", "shower_00000779",
    "subway_station_platform_00003408", "subway_station_platform_00004871",
    "train_interior_00004637", "train_station_platform_00003977",
    "waiting_room_00002953", "yard_00000709",
]

# ------------------------------------------------------------
# 2. INPUTS (read, never written; paths relative to the project folder)
# ------------------------------------------------------------
# The v4 manifest gives each scene's category, group, indoor/outdoor and
# source photograph; the v4 stimuli folder holds v4's unedited image of
# each scene (check O1).
V4_MANIFEST_CSV = "data/stimulus_set_v4_ext_haze_eq/stimulus_manifest.csv"
V4_STIMULI_DIR  = "data/stimulus_set_v4_ext_haze_eq/stimuli"
V4_STATS_CSV    = "data/levels_v4_ext/corpus_feature_stats_v4.csv"
INDEX_CSV    = "data/selection/image_index.csv"          # step 02
FEATURES_NPY = "data/features/features_crop.npy"         # step 04
FEATURE_DEFS = "data/features/feature_definitions.json"  # step 04
FLAGS_CSV    = "data/monochrome/image_flags.csv"         # step 04a
DUP_CSV      = "data/checks_v4/duplicate_groups.csv"     # 09a
CATEGORY_CSV = "everyday_categories.csv"                 # as v4 step 1

# TEST_RUN_DIR
#   The test run the method was chosen on. Its images of this method
#   (images/A4_haze_cc_rgb/b_sd/) must come out byte-identical (check
#   T1), its parameters equal (T2), and its settings equal to the copied
#   settings below (T0). None, or a folder without the images, skips
#   what cannot be checked (reported as SKIP, never silently).
TEST_RUN_DIR = "data/edit_levels_v5/4aa278d54d"
TEST_RUN_ID  = "4aa278d54d"

# PLACES365_ROOT
#   The folder the Places365 file IDs are relative to. None = the common
#   folder of the 47 source photographs (Places365's own layout below it,
#   e.g. a/alley/00001206.jpg). Only the relative ID is written anywhere.
PLACES365_ROOT = None

# ------------------------------------------------------------
# 3. OUTPUT
# ------------------------------------------------------------
# A new folder OUTPUT_ROOT/<settings hash>; a folder already holding
# files is never touched (_2, _3, ... instead).
OUTPUT_ROOT = "data/stimulus_set_v5"
STIMULI_SUBDIR = "stimuli"
SET_NAME = "drl2drv v5 stimulus set"

# The 20,000-image stimulus-path reference is read from the test run's
# cache when it is there (same code, same images: same key), else
# re-measured and cached in CACHE_DIR. Nothing is written to the test
# run's cache.
USE_CACHE = True
CACHE_DIR = "data/stimulus_set_v5_cache"
CACHE_READ_DIRS = ["data/edit_levels_v5_cache"]

# Tolerances of the checks
LEVEL_CHECK_TOL_SD = 0.01    # T3: a level within this of its target
REMEASURE_TOL      = 1e-9    # M2: features from the PNG vs in memory

N_WORKERS = 20
CHUNK     = 1

# ------------------------------------------------------------
# 4. COPIED FROM test_edit_levels_v5.py, run 4aa278d54d. DO NOT EDIT.
# ------------------------------------------------------------
# Everything below fixes how the images are made and measured. The
# values are the test run's (check T0 compares them with its
# settings.json); changing any of them changes the images, and T1 then
# fails. Comments are shortened; the test script explains each.

# geometry and v4's manipulations
IMAGE_SIZE        = 512        # edits applied, and images shown, at this size
FEATURE_SIZE      = 256        # features measured at this size
CLIP_PX           = 336        # CLIP's resolution, for clip_box
HAZE_AIRLIGHT_TOP = 1.0        # % brightest pixels whose mean is the airlight
GAMUT_ITERS       = 16
GAMUT_CMAX_SEARCH = 160.0
GAMUT_MARGIN      = 0.995
MANIP_ORDER       = ("haze", "chroma")   # contrast edit first, then colour
IDENTITY_SHORT_CIRCUIT = True  # the "no change" setting returns the image
FEAT_SAT = "sat_mean"
FEAT_CON = "contrast_local"

# corpus reference (as v4 step 1)
RANDOM_SEED        = 42
MIN_SIDE           = 512
EXCLUDE_GREYSCALE  = True
EXCLUDE_TONED      = True
EXCLUDE_DUPLICATES = True
RESTRICT_TO_KEEP_CATEGORIES = True
STATS_POPULATION   = "everyday"
MIN_POPULATION     = 1000
N_REFERENCE_IMAGES = 20_000
REFERENCE_SEED     = 20250101
FEATURE_CHECK_N    = 8
FEATURE_CHECK_RTOL = 1e-4
REF_PERCENTILES = list(range(1, 100))
REF_CHECK_RTOL = 2e-5
SD_SAT_V5    = 39.182
SD_FOG_V5    = 9.348
SD_CHECK_TOL = 0.0006

# clipping: measured and reported; with B_USE_CLIP_BUDGET False it never
# stops a level
CLIP_MAX_NEW       = 0.05
CLIP_DEPTH         = 4.0
CLIP_REPORT_DEPTHS = (0.5, 2.0, 4.0, 8.0)
GAMUT_MAP_MIN_DC   = 3.0
B_USE_CLIP_BUDGET  = False

# the edits' parameters (every family is kept, as in the test script,
# so its code is copied whole; only haze_eq, dehaze_rgb and chroma are
# used)
CLARITY_SIGMA_FRAC = 0.03
SHARPEN_SIGMA      = 3.0
PYR_LEVELS         = 3
GRID_N           = 24
HAZE_MAX         = 0.95
CHROMA_MIN       = 0.0
CHROMA_MAX       = 2.0
CHROMA_MAX_REACH = 4.0
DEHAZE_MIN       = -2.0
CLARITY_MAX      = 4.0
SHARPEN_MAX      = 6.0
PYR_GAIN_MAX     = 4.0
PIL_CONTRAST_MAX = 2.5
PIL_COLOR_MAX    = 3.0
TONE_MAX         = 4.0

# the colour correction of each contrast level, and the level solver
COMP_K_RANGE = (0.25, 3.0)
COMP_ITERS   = 20
COMP_ROUNDS  = 2      # contrast and correction re-solved alternately; the
                      # last correction is not followed by a contrast
                      # re-solve, which is what leaves the level misses
                      # reported by T3 (kept: images byte-identical)
REACH_BISECT_ITERS = 12
SOLVE_ITERS      = 24
SOLVE_TOL_SD     = 0.002
PLATEAU_REL      = 0.05
REACH_ARTEFACT_LIMITS = {"halo_share": None, "noise_ratio": None,
                         "band_new_empty": None}

# artefact measures (numbers only)
HALO_CANNY     = (100, 200)
HALO_BAND_PX   = 8
HALO_WINDOW_PX = 9
HALO_TOL_L     = 3.0
NOISE_FLAT_PERCENTILE = 40
NOISE_SMOOTH_SIGMA    = 2.0
NOISE_HP_SIGMA        = 1.0
BAND_RANGE_PCT = (1, 99)

# envelope (as v4) and the descriptive measures
ENVELOPE_FEATURES = ["lum_mean", "lum_p05", "lum_p50", "lum_p95",
                     "val_mean", "lum_skew", "spectral_slope"]
ENVELOPE_LO, ENVELOPE_HI = "p01", "p99"
EXTRA_TEXTURE = ["fine_texture", "noise_sigma", "edge_overshoot"]
DESCRIPTIVE   = ["lum_std", "spectral_slope", "lum_mean", "lum_p05",
                 "lum_p95"] + EXTRA_TEXTURE

# Used by the copied ladders_b only for the test script's ratio
# definition, which is not built here.
R_FEATURES = []

# The names above, compared with the test run by check T0.
COPIED_SETTINGS = [
    "IMAGE_SIZE", "FEATURE_SIZE", "CLIP_PX", "HAZE_AIRLIGHT_TOP",
    "GAMUT_ITERS", "GAMUT_CMAX_SEARCH", "GAMUT_MARGIN", "MANIP_ORDER",
    "IDENTITY_SHORT_CIRCUIT", "FEAT_SAT", "FEAT_CON", "RANDOM_SEED",
    "MIN_SIDE", "EXCLUDE_GREYSCALE", "EXCLUDE_TONED", "EXCLUDE_DUPLICATES",
    "RESTRICT_TO_KEEP_CATEGORIES", "STATS_POPULATION", "MIN_POPULATION",
    "N_REFERENCE_IMAGES", "REFERENCE_SEED", "FEATURE_CHECK_N",
    "FEATURE_CHECK_RTOL", "REF_PERCENTILES", "REF_CHECK_RTOL",
    "SD_SAT_V5", "SD_FOG_V5", "SD_CHECK_TOL", "CLIP_MAX_NEW", "CLIP_DEPTH",
    "CLIP_REPORT_DEPTHS", "GAMUT_MAP_MIN_DC", "B_USE_CLIP_BUDGET",
    "CLARITY_SIGMA_FRAC", "SHARPEN_SIGMA", "PYR_LEVELS", "GRID_N",
    "HAZE_MAX", "CHROMA_MIN", "CHROMA_MAX", "CHROMA_MAX_REACH",
    "DEHAZE_MIN", "CLARITY_MAX", "SHARPEN_MAX", "PYR_GAIN_MAX",
    "PIL_CONTRAST_MAX", "PIL_COLOR_MAX", "TONE_MAX", "COMP_K_RANGE",
    "COMP_ITERS", "COMP_ROUNDS", "REACH_BISECT_ITERS", "SOLVE_ITERS",
    "SOLVE_TOL_SD", "PLATEAU_REL", "REACH_ARTEFACT_LIMITS", "HALO_CANNY",
    "HALO_BAND_PX", "HALO_WINDOW_PX", "HALO_TOL_L",
    "NOISE_FLAT_PERCENTILE", "NOISE_SMOOTH_SIGMA", "NOISE_HP_SIGMA",
    "BAND_RANGE_PCT", "ENVELOPE_FEATURES", "ENVELOPE_LO", "ENVELOPE_HI",
    "B_SPAN_SD", "LEVEL_FRACTIONS",
]

_SETTING_NAMES = sorted(k for k in globals()
                        if k not in _G0 and not k.startswith("_"))
# Operational settings that change no result; left out of the hash.
_NOT_HASHED = {"N_WORKERS", "CHUNK", "USE_CACHE", "CACHE_DIR",
               "CACHE_READ_DIRS", "TEST_RUN_DIR"}


# ################################################################
# VERBATIM from build_stimulus_set_v4_corrected.py (lines 740-1176), as in test_edit_levels_v5.py: feature code, geometry, conversions, v4 manipulations
# ################################################################


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


def srgb_to_lin_f(x):
    """sRGB on the 0-255 scale -> linear light, for UNCLIPPED input.

    srgb_to_lin takes uint8, so it cannot be used on a gain that has
    pushed pixels past 0 or 255. Below the sRGB knee the transfer is
    linear, and that branch extends to negative values unchanged; above
    1 the power branch extends smoothly. Both extensions are the
    standard ones, and they matter only for pixels about to be pulled
    back inside.
    """
    y = x.astype(np.float32) / 255.0
    # np.where evaluates BOTH branches, so the power has to be given a
    # non-negative base even where its result is about to be discarded;
    # otherwise every negative pixel raises an invalid-value warning.
    hi = np.maximum(y + 0.055, 0.0) / 1.055
    return np.where(y <= 0.04045, y / 12.92,
                    hi ** 2.4).astype(np.float32)


def lin_to_srgb_f(x):
    """Linear light -> sRGB on the 0-255 scale, unclipped: values above
    1 in linear light map above 255, and NEGATIVE values map below 0.

    The branch test is on the SIGNED value, exactly as in srgb_to_lin_f,
    which makes the two exact inverses over the whole line:
    srgb_to_lin_f sends every negative through its linear branch
    (y <= 0.04045 is true for all y < 0), so the inverse of that branch
    -- 12.92 * x -- has to cover every negative too. Branching on |x|
    instead is a legitimate convention but is NOT the inverse of the
    function above: it sends anything past about -10.3 levels through
    the power branch, so a pixel a gain drove to -41.9 is reported as
    -29.6.

    THIS USED TO BEGIN `x = np.maximum(x, 0.0)`, and that one line made
    the dark end of the contrast-gain families impossible to MEASURE.
    Every manipulation ending here returned a non-negative array, so
    nothing downstream could find a pixel below zero -- not because a
    gain spares the shadows, but because the clamp happened one step
    before anything looked. A diagnostic that cannot return a non-zero
    value is not evidence.

    THE RENDERED IMAGE IS UNCHANGED. to_u8 clips to [0, 255], so a
    pixel clamped to 0 here was clipped to 0 there; CHECK 8 reproducing
    the earlier build bit-for-bit is the proof of that, and is the
    reason this edit could be made at all after the set was built.
    """
    # np.where evaluates both branches, so the power needs a
    # non-negative base even where its result is discarded.
    hi = np.maximum(x, 0.0)
    y = np.where(x <= 0.0031308, 12.92 * x,
                 1.055 * hi ** (1 / 2.4) - 0.055)
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


def cmax_of_img(img):
    """Per-pixel in-gamut chroma ceiling of an image, as m_chroma needs
    it for k > 1: read the lightness and hue off THIS image and ask
    cmax_of for the largest chroma that still lands inside sRGB.

    Verbatim from step 1. The old build had no level above k = 1, so
    ctx never carried a "cmax" and nothing noticed it was missing.
    """
    lab = lab_of(img)
    return cmax_of(lab[:, :, 0],
                   np.arctan2(lab[:, :, 2], lab[:, :, 1]))


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


def m_contrast_gain_srgb(img, g, ctx):
    """Contrast increase as a gain about the scene's mean grey, applied
    to sRGB VALUES. The same family of operation as Pillow's
    ImageEnhance.Contrast with a factor above 1:

        I' = m + g * (I - m),   m = the image's mean grey

    THE DEFAULT for the contrast-increase column. m_contrast_gain does
    the same thing in linear light, and the difference is where the
    crushed range falls:

      space    scene mean 28   mean 100   mean 195   (sRGB level below
      linear             1.8       18.0       44.8    which a pixel is
      sRGB               3.4       12.3       23.9    pushed negative)

    The linear threshold climbs steeply with scene brightness, because
    the gamma curve is flat near black and steep near white, so on a
    bright scene a linear gain crushes a wide band of real shadow
    detail. The two cross over near a scene mean of 60, and the
    stimulus scenes span roughly 27 to 165.

    The mean is restored afterwards as the other families do, which is
    nearly a no-op here -- a gain about the mean preserves it except
    for what clipping takes -- but it is the reason this is not
    literally ImageEnhance.Contrast(g). Step 1 measured the gap: within
    one level on all five scenes tested, 0% of pixels past one level.

    Verbatim from step 1.
    """
    m = mean_grey(img)
    out = m + g * (_f(img) - m)
    target = ctx["lum_mean_512"]
    lin = srgb_to_lin_f(out)
    lo, hi = -4.0, 4.0
    for _ in range(14):
        ev = 0.5 * (lo + hi)
        out = lin_to_srgb_f(lin * 2.0 ** ev)
        if mean_grey(to_u8(out)) < target:
            lo = ev
        else:
            hi = ev
    return out


def m_contrast_gain(img, g, ctx):
    """The same gain in LINEAR LIGHT, with the mean restored.

    Kept so the choice of space stays visible and reversible, not
    because it is used. Step 1 compared the two at a matched clipping
    budget: in linear light no percentile target could be solved at all
    (gain 1.0146, +0.111 SD, 83% of its clipping at the dark end),
    where sRGB reached +0.22 SD with less cross-talk onto the colour
    feature. Verbatim from step 1.
    """
    lin = srgb_to_lin_f(_f(img))
    m = float(lin.mean())
    out = lin_to_srgb_f(m + g * (lin - m))
    target = ctx["lum_mean_512"]
    lo, hi = -4.0, 4.0
    base = srgb_to_lin_f(out)
    for _ in range(14):
        ev = 0.5 * (lo + hi)
        out = lin_to_srgb_f(base * 2.0 ** ev)
        if mean_grey(to_u8(out)) < target:
            lo = ev
        else:
            hi = ev
    return out


FUNCS = {"chroma": m_chroma, "haze": m_haze, "haze_eq": m_haze_eq,
         "contrast_gain": m_contrast_gain,
         "contrast_gain_srgb": m_contrast_gain_srgb}
IDENTITY_PARAM = {"chroma": 1.0, "haze": 0.0, "haze_eq": 0.0,
                  "contrast_gain": 1.0, "contrast_gain_srgb": 1.0}


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
    """One cell of the stimulus grid, in MANIP_ORDER.

    The in-gamut chroma ceiling for k > 1 is read off the image AS IT
    STANDS, after the haze step, not off the unmodified square. Haze
    moves both lightness and hue, and the ceiling is a function of
    both, so a ceiling taken from the square belongs to a different
    image: at the deep fog rungs it would let chroma run past the
    gamut, and the soft-saturating blend in m_chroma would stop
    protecting anything.

    For k <= 1 nothing is computed -- m_chroma's k <= 1 branch never
    touches ctx["cmax"] -- so the 25 cells this build shares with the
    old one go down exactly the old path. That is what makes CHECK 8 a
    meaningful test rather than a tautology.
    """
    img = sq
    for f in MANIP_ORDER:
        if f == "chroma":
            ctx_c = dict(ctx, cmax=cmax_of_img(img)) if k > 1.0 else ctx
            img = apply("chroma", img, k, ctx_c)
        else:
            img = apply(fam, img, h, ctx)
    return img


# ################################################################
# VERBATIM from 09c_manipulation_sweep.py (lines 318-329, 436-446, 529-585), as in test_edit_levels_v5.py: 04c texture measures
# ################################################################
BIL_SIGMA_COLOR = 15.0
BIL_SIGMA_SPACE_AT_256 = 3.0
NOISE_FLAT_PERCENTILE = 40
NOISE_SMOOTH_SIGMA_AT_256 = 1.0
NOISE_CLIP = (3, 252)
NOISE_MIN_PIXELS = 500
OVERSHOOT_CANNY = (100, 200)
OVERSHOOT_T_NEAR_AT_256 = 3
OVERSHOOT_T_OUT_AT_256 = 6
OVERSHOOT_MIN_CONTRAST = 20.0
OVERSHOOT_MAX_EDGES = 3000
OVERSHOOT_MIN_EDGES = 50


def grey(rgb_u8):
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    g8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return g8, g8.astype(np.float32)


def sobel(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return gx, gy, np.sqrt(gx * gx + gy * gy)



def fine_texture(g, s):
    bil = cv2.bilateralFilter(g, -1, BIL_SIGMA_COLOR,
                              BIL_SIGMA_SPACE_AT_256 * s)
    return float((g - bil).std())


_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def noise_sigma(g, s):
    c = np.abs(cv2.filter2D(g, cv2.CV_32F, _IMMERKAER))[1:-1, 1:-1]
    _, _, ms = sobel(cv2.GaussianBlur(g, (0, 0),
                                      NOISE_SMOOTH_SIGMA_AT_256 * s))
    ms = ms[1:-1, 1:-1]
    gi = g[1:-1, 1:-1]
    valid = (gi > NOISE_CLIP[0]) & (gi < NOISE_CLIP[1])
    if valid.sum() < NOISE_MIN_PIXELS:
        return np.nan
    thr = np.percentile(ms[valid], NOISE_FLAT_PERCENTILE)
    sel = valid & (ms <= thr)
    if sel.sum() < NOISE_MIN_PIXELS:
        return np.nan
    return float(np.sqrt(np.pi / 2.0) * c[sel].mean() / 6.0)


def edge_overshoot(g8, g, gx, gy, mag, s):
    t_near = max(1, int(round(OVERSHOOT_T_NEAR_AT_256 * s)))
    t_out = max(t_near + 1, int(round(OVERSHOOT_T_OUT_AT_256 * s)))
    edges = cv2.Canny(g8, OVERSHOOT_CANNY[0], OVERSHOOT_CANNY[1]) > 0
    m = t_out + 2
    edges[:m, :] = edges[-m:, :] = False
    edges[:, :m] = edges[:, -m:] = False
    edges &= mag > 0
    ys, xs = np.nonzero(edges)
    if len(ys) > OVERSHOOT_MAX_EDGES:
        k = np.linspace(0, len(ys) - 1, OVERSHOOT_MAX_EDGES).astype(int)
        ys, xs = ys[k], xs[k]
    if len(ys) < OVERSHOOT_MIN_EDGES:
        return np.nan
    nx = (gx[ys, xs] / mag[ys, xs])[:, None]
    ny = (gy[ys, xs] / mag[ys, xs])[:, None]
    t = np.arange(1, t_out + 1, dtype=np.float32)[None, :]
    X, Y = xs[:, None].astype(np.float32), ys[:, None].astype(np.float32)
    bright = cv2.remap(g, (X + t * nx).astype(np.float32),
                       (Y + t * ny).astype(np.float32), cv2.INTER_LINEAR)
    dark = cv2.remap(g, (X - t * nx).astype(np.float32),
                     (Y - t * ny).astype(np.float32), cv2.INTER_LINEAR)
    pb = np.median(bright[:, t_near:], axis=1)
    pd_ = np.median(dark[:, t_near:], axis=1)
    over = np.maximum(bright[:, :t_near].max(axis=1) - pb, 0.0)
    under = np.maximum(pd_ - dark[:, :t_near].min(axis=1), 0.0)
    contrast = pb - pd_
    ok = contrast >= OVERSHOOT_MIN_CONTRAST
    if ok.sum() < OVERSHOOT_MIN_EDGES:
        return np.nan
    return float(np.minimum((over + under)[ok] / contrast[ok], 1.0).mean())



# ################################################################
# VERBATIM from calibrate_levels_v4_corrected.py (lines 1344-1388), as in test_edit_levels_v5.py: m_contrast_up (dehaze_rgb)
# ################################################################
def m_contrast_up(img, h, ctx):
    """Contrast INCREASE: the equal-luminance haze run backwards.

    h <= 0, so the blend moves AWAY from the airlight,

        I' = (1 - h) * I + h * A      with h < 0,

    which is (1+|h|)*I - |h|*A: deviations from the airlight are
    amplified. The same linear-light gain then restores the level-0
    mean grey, so this is the exact inverse of m_haze_eq and the
    feature moves back along the same path.

    GRID BLOCKS ONLY. The pair grid never uses it, so it does not need
    to stay byte-identical with 09c, which is why it can differ from
    m_haze_eq in the one way that matters here:

    m_haze_eq quantises between the blend and the gain (`to_u8`). That
    is harmless when the blend moves pixels INWARD, toward the
    airlight. Run backwards the blend pushes them out past 0 and 255,
    and an intermediate to_u8 would clip them there -- before the gain
    could pull them back, and where apply() can no longer see it. So
    the intermediate stays in float, which keeps the clipping visible
    to apply() and lets the gain recover the part of it that is
    recoverable.

    Because the airlight is the BRIGHT end of the image, amplifying
    away from it darkens shadows faster than it brightens highlights,
    so the black end is where clipping appears first. CHECK 9 reports
    it against CLIP_MAX_NEW_UP, and CONTRAST_UP_FAMILY is the
    setting to change if it is too large.
    """
    A = np.asarray(ctx["airlight"], np.float32)[None, None, :]
    blend = (1.0 - h) * _f(img) + h * A
    target = ctx["lum_mean_512"]
    lin = srgb_to_lin_f(blend)
    lo, hi = -4.0, 4.0
    out = blend
    for _ in range(14):
        ev = 0.5 * (lo + hi)
        out = lin_to_srgb_f(lin * 2.0 ** ev)
        if mean_grey(to_u8(out)) < target:
            lo = ev
        else:
            hi = ev
    return out


# ################################################################
# VERBATIM from test_edit_levels_v5.py: the edits (all families, as the test run had them), steps, clipping records
# ################################################################


# ============================================================
# New edits (this script; not part of v4)
# ============================================================
#
# Every edit takes a uint8 RGB image and returns FLOAT sRGB values on the
# 0-255 scale, unclipped, as the v4 manipulations do, so clipping can be
# measured before to_u8 hides it. The lightness-only edits also return a
# mask of the pixels their gamut handling had to change.

def _lin(a, b, n):
    """v4's lin(): n + 1 evenly spaced points from a to b."""
    return np.linspace(a, b, n + 1)


def _lab_l_edit(img, L2):
    """Replace the CIELAB L* of `img` by L2 at fixed a*, b*.

    L2 is clamped to [0, 100]. A pixel that then falls outside sRGB by
    more than half a level has its chroma reduced to the in-gamut ceiling
    at its new lightness and its own hue (cmax_of, v4's gamut search), as
    09c's temperature edit does. Returns (float sRGB, mask of pixels whose
    L* left [0, 100] or whose chroma was cut by more than
    GAMUT_MAP_MIN_DC).
    """
    lab = lab_of(img)
    L2 = np.asarray(L2, np.float32)
    out_l = (L2 < -1e-3) | (L2 > 100.0 + 1e-3)
    lab[:, :, 0] = np.clip(L2, 0.0, 100.0)
    x = rgb_of_f(lab)
    oog = ((x < -0.5) | (x > 255.5)).any(axis=2)
    cut = np.zeros(oog.shape, bool)
    if oog.any():
        a = lab[:, :, 1][oog]
        b = lab[:, :, 2][oog]
        C = np.sqrt(a * a + b * b)
        cm = cmax_of(lab[:, :, 0][oog], np.arctan2(b, a))
        s = np.where(C > cm, cm / np.maximum(C, 1e-6), 1.0).astype(np.float32)
        lab[:, :, 1][oog] = a * s
        lab[:, :, 2][oog] = b * s
        cut[oog] = (C - np.minimum(C, cm)) > GAMUT_MAP_MIN_DC
        x = rgb_of_f(lab)
    return x, (cut | out_l)


def _restore_mean_grey(x, ctx):
    """haze_eq's brightness restoration on an UNCLIPPED float image: a
    linear-light gain, found by 14-step bisection, that brings the mean
    grey back to the original's. As in m_contrast_up, the input stays in
    float so pixels pushed past the range remain visible to the clipping
    count."""
    target = ctx["lum_mean_512"]
    lin = srgb_to_lin_f(x)
    lo, hi = -4.0, 4.0
    out = x
    for _ in range(14):
        ev = 0.5 * (lo + hi)
        out = lin_to_srgb_f(lin * 2.0 ** ev)
        if mean_grey(to_u8(out)) < target:
            lo = ev
        else:
            hi = ev
    return out


def airlight_L(ctx):
    """CIELAB L* of the scene's airlight colour."""
    a = np.asarray(ctx["airlight"], np.float32).reshape(1, 1, 3)
    return float(lab_of(a)[0, 0, 0])


def m_dehaze_L(img, h, ctx):
    """The inverse haze blend on L* only: L' = (1 - h) L + h L_A, h <= 0,
    which amplifies lightness away from the airlight's; a*, b* unchanged
    (gamut-mapped where needed); then haze_eq's brightness restoration."""
    L = lab_of(img)[:, :, 0]
    L2 = (1.0 - h) * L + h * airlight_L(ctx)
    x, m = _lab_l_edit(img, L2)
    return _restore_mean_grey(x, ctx), m


def _usm_L(img, a, sigma):
    L = lab_of(img)[:, :, 0]
    L2 = L + a * (L - cv2.GaussianBlur(L, (0, 0), sigma))
    return _lab_l_edit(img, L2)


def m_clarity_L(img, a, ctx):
    """Large-radius unsharp mask on L*: L' = L + a (L - G_sigma * L),
    sigma = CLARITY_SIGMA_FRAC x the image size."""
    return _usm_L(img, a, CLARITY_SIGMA_FRAC * img.shape[0])


def m_sharpen_L(img, a, ctx):
    """Small-radius unsharp mask on L*, sigma = SHARPEN_SIGMA (09c's
    sharpen used the same radius on each RGB channel)."""
    return _usm_L(img, a, SHARPEN_SIGMA)


def _pyr_base(L, n):
    """The Laplacian pyramid's base: L reduced n times and expanded back.
    The pyramid is linear, so scaling the n finest bands by g is exactly
    base + g (L - base)."""
    sizes, g = [], L
    for _ in range(n):
        sizes.append((g.shape[1], g.shape[0]))
        g = cv2.pyrDown(g)
    for s in reversed(sizes):
        g = cv2.pyrUp(g, dstsize=s)
    return g


def m_pyramid_L(img, g, ctx):
    """Fine-detail gain on L*: the finest PYR_LEVELS Laplacian bands
    scaled by g (g < 1 removes detail, g > 1 adds it)."""
    L = lab_of(img)[:, :, 0]
    B = _pyr_base(L, PYR_LEVELS)
    return _lab_l_edit(img, B + g * (L - B))


_Y_W = _M_XYZ[1].astype(np.float32)      # luminance weights of the sRGB
                                         # matrix lab_of / rgb_of_f use


def _tone_curve(t, k, m):
    """09c's m_contrast curve on a 0-1 scale: slope k at m; for k <= 1 a
    linear compression toward m, for k > 1 an S-curve with 0 and 1
    fixed."""
    if k <= 1.0:
        return m + k * (t - m)
    lo = m * np.power(np.clip(t / m, 0, None), k)
    hi = 1.0 - (1.0 - m) * np.power(np.clip((1.0 - t) / (1.0 - m), 0, None),
                                    k)
    return np.where(t < m, lo, hi)


def m_tone_L(img, k, ctx):
    """Global contrast on CIELAB L* only, one parameter both ways: the
    tone curve of 09c's m_contrast (_tone_curve: slope k at the image's
    mean L*, black and white held fixed) applied to L*, with a* and b*
    unchanged.

      k <= 1   linear compression toward the mean L*: lowers contrast,
               moves every pixel inward, cannot clip
      k > 1    an S-curve: raises midtone contrast; the extremes are
               compressed instead of clipped

    A colour keeps its CIELAB chroma when its lightness changes, so it
    keeps its colourfulness (no glowing dark reds, unlike tone_Y). HSV
    saturation still moves somewhat (lightening a dark colour at fixed
    a*, b* lowers it), and is reported as cross-talk. A pixel pushed
    outside sRGB is gamut-mapped at its new lightness and hue, and
    counted, by _lab_l_edit."""
    L = lab_of(img)[:, :, 0]
    t = L / 100.0
    m = float(np.clip(t.mean(), 0.02, 0.98))
    return _lab_l_edit(img, 100.0 * _tone_curve(t, k, m))


def m_tone_Y(img, k, ctx):
    """Global contrast on luminance, one parameter both ways, keeping each
    pixel's RGB proportions (its chromaticity).

    The tone curve of 09c's m_contrast (slope k at the mean, black and
    white fixed) is applied to the lightness of each pixel's luminance,
    L*(Y) / 100, about its image mean; the pixel's linear-light RGB is
    then scaled by Y' / Y.

      k <= 1   linear compression toward the mean: lowers contrast and
               moves every pixel inward, so it cannot clip
      k > 1    an S-curve: raises midtone contrast with black and white
               held fixed; only a bright, saturated pixel whose largest
               channel would pass white can clip, and is counted

    Why luminance and RGB proportions, not CIELAB L* at fixed a*, b*:
    the colour feature is HSV saturation, which depends on a colour's
    lightness at fixed a*, b* (brightening a dark colour lowers it), so
    an L*-only curve moved sat_mean by 0.3-0.4 SD per SD of contrast in a
    test, and gamut-mapped more pixels; scaling linear RGB keeps the RGB
    ratios that HSV saturation is computed from (about 0.05 in the same
    test)."""
    lin = srgb_to_lin(img).astype(np.float32)
    Y = lin @ _Y_W
    L = np.where(Y > 216.0 / 24389.0, 116.0 * np.cbrt(Y) - 16.0,
                 Y * 24389.0 / 27.0) / 100.0
    m = float(np.clip(L.mean(), 0.02, 0.98))
    L2 = 100.0 * _tone_curve(L, k, m)
    f = (L2 + 16.0) / 116.0
    Y2 = np.where(L2 > 8.0, f ** 3, L2 * 27.0 / 24389.0)
    s = np.where(Y > 1e-6, Y2 / np.maximum(Y, 1e-6), 1.0).astype(np.float32)
    return lin_to_srgb_f(lin * s[:, :, None])


def m_pil_contrast(img, f, ctx):
    """The float model of Pillow's ImageEnhance.Contrast, used only to
    measure clipping (Pillow clips internally): a blend with a solid grey
    at the rounded mean of the image's "L" conversion."""
    mean = int(ImageStat.Stat(Image.fromarray(img).convert("L")).mean[0]
               + 0.5)
    return mean + f * (img.astype(np.float32) - mean)


def m_pil_color(img, f, ctx):
    """The float model of Pillow's ImageEnhance.Color: a blend with the
    image's own greyscale ("L" converted back to RGB)."""
    G = np.asarray(Image.fromarray(img).convert("L").convert("RGB"),
                   np.float32)
    return G + f * (img.astype(np.float32) - G)


def r_pil_contrast(img, f):
    return np.asarray(ImageEnhance.Contrast(Image.fromarray(img)).enhance(f),
                      np.uint8)


def r_pil_color(img, f):
    return np.asarray(ImageEnhance.Color(Image.fromarray(img)).enhance(f),
                      np.uint8)


# ------------------------------------------------------------
# The family registry
# ------------------------------------------------------------
#   fn        the edit (float output; may also return a gamut mask)
#   feature   the feature it is calibrated on
#   identity  the "no change" parameter
#   dirs      directions it can go
#   render    Pillow families: the image is Pillow's own output; the float
#             model only counts clipping
#   source    where the code comes from

FAMILIES = {
    "haze_eq":      dict(fn=m_haze_eq, feature=FEAT_CON, identity=0.0,
                         dirs=("down",), render=None, mask=False,
                         source="v4 (verbatim)"),
    "dehaze_rgb":   dict(fn=m_contrast_up, feature=FEAT_CON, identity=0.0,
                         dirs=("up",), render=None, mask=False,
                         source="v4 m_contrast_up (verbatim)"),
    "dehaze_L":     dict(fn=m_dehaze_L, feature=FEAT_CON, identity=0.0,
                         dirs=("up",), render=None, mask=True, source="new"),
    "clarity_L":    dict(fn=m_clarity_L, feature=FEAT_CON, identity=0.0,
                         dirs=("up",), render=None, mask=True, source="new"),
    "sharpen_L":    dict(fn=m_sharpen_L, feature=FEAT_CON, identity=0.0,
                         dirs=("up",), render=None, mask=True, source="new"),
    "pyramid_L":    dict(fn=m_pyramid_L, feature=FEAT_CON, identity=1.0,
                         dirs=("down", "up"), render=None, mask=True,
                         source="new"),
    "tone_L":       dict(fn=m_tone_L, feature=FEAT_CON, identity=1.0,
                         dirs=("down", "up"), render=None, mask=True,
                         source="new (09c m_contrast curve, on L*)"),
    "tone_Y":       dict(fn=m_tone_Y, feature=FEAT_CON, identity=1.0,
                         dirs=("down", "up"), render=None, mask=False,
                         source="new (09c m_contrast curve, on luminance)"),
    "pil_contrast": dict(fn=m_pil_contrast, feature=FEAT_CON, identity=1.0,
                         dirs=("down", "up"), render=r_pil_contrast,
                         mask=False, source="Pillow ImageEnhance.Contrast"),
    "chroma":       dict(fn=m_chroma, feature=FEAT_SAT, identity=1.0,
                         dirs=("down", "up"), render=None, mask=False,
                         source="v4 (verbatim)"),
    "pil_color":    dict(fn=m_pil_color, feature=FEAT_SAT, identity=1.0,
                         dirs=("down", "up"), render=r_pil_color,
                         mask=False, source="Pillow ImageEnhance.Color"),
}

CLIP_DEPTHS = tuple(sorted(set(CLIP_REPORT_DEPTHS) | {CLIP_DEPTH}))


def other_feature(feat):
    return FEAT_CON if feat == FEAT_SAT else FEAT_SAT


def grid_of(fam, direction, purpose="a"):
    """The parameter grid of one family and direction, from the identity
    outward. purpose "a" is the definition-(a) calibration grid (v4's for
    haze_eq and chroma); "reach" is the per-scene search of (b)."""
    up = direction == "up"
    if fam == "haze_eq":
        return _lin(0.0, HAZE_MAX, GRID_N)
    if fam == "chroma":
        if not up:
            return _lin(1.0, CHROMA_MIN, GRID_N)
        return _lin(1.0, CHROMA_MAX if purpose == "a" else CHROMA_MAX_REACH,
                    GRID_N)
    if fam in ("dehaze_rgb", "dehaze_L"):
        # fine near the original, coarse far out (the range is wide)
        return np.r_[0.0, -np.geomspace(0.01, -DEHAZE_MIN, GRID_N)]
    if fam == "clarity_L":
        return np.r_[0.0, np.geomspace(0.02, CLARITY_MAX, GRID_N)]
    if fam == "sharpen_L":
        return np.r_[0.0, np.geomspace(0.02, SHARPEN_MAX, GRID_N)]
    top = {"pyramid_L": PYR_GAIN_MAX, "pil_contrast": PIL_CONTRAST_MAX,
           "pil_color": PIL_COLOR_MAX, "tone_Y": TONE_MAX,
           "tone_L": TONE_MAX}[fam]
    if not up:
        return _lin(1.0, 0.0, GRID_N)
    first = 0.01 if fam == "pil_contrast" else 0.02
    return np.r_[1.0, 1.0 + np.geomspace(first, top - 1.0, GRID_N)]


def fam_call(fam, img, p, ctx):
    out = FAMILIES[fam]["fn"](img, p, ctx)
    if FAMILIES[fam]["mask"]:
        return out
    return out, None


def apply_step(fam, img, p, ctx0):
    """One edit on one image: (uint8 image, clip record).

    The clip record holds, per depth in CLIP_DEPTHS, the pixels with a
    channel that was inside (0, 255) in THIS STEP'S INPUT and was pushed
    more than that many levels past the range (v4's apply(), at several
    depths), and the gamut mask of the lightness-only edits (pixels with
    no channel at the rail). For chroma above 1 the in-gamut ceiling is
    read off the image as it stands, as v4's render_cell does.
    """
    F = FAMILIES[fam]
    if IDENTITY_SHORT_CIRCUIT and p == F["identity"]:
        return img, None
    ctx = ctx0
    if fam == "chroma" and p > 1.0:
        if ctx0.get("square") is img:
            # the unmodified square: its ceiling is computed once and kept
            # on the context (the same value v4's ctx_of(r, need_cmax)
            # caches on its calibration rows)
            if "cmax" not in ctx0:
                ctx0["cmax"] = cmax_of_img(img)
            ctx = ctx0
        else:
            ctx = dict(ctx0, cmax=cmax_of_img(img))
    x, gm = fam_call(fam, img, p, ctx)
    inside = (img > 0) & (img < 255)
    over = {d: (((x < -d) | (x > 255.0 + d)) & inside).any(axis=2)
            for d in CLIP_DEPTHS}
    if gm is not None:
        gm = gm & inside.all(axis=2)
    u8 = F["render"](img, p) if F["render"] else to_u8(x)
    return u8, (over, gm)


def merge_clip(a, b):
    if a is None:
        return b
    if b is None:
        return a
    over = {d: a[0][d] | b[0][d] for d in CLIP_DEPTHS}
    if a[1] is None:
        gm = b[1]
    elif b[1] is None:
        gm = a[1]
    else:
        gm = a[1] | b[1]
    return over, gm


def clip_shares(rec):
    """{depth: share of newly clipped pixels} plus the gamut share. The
    budgeted number is the share at CLIP_DEPTH, gamut mask included."""
    out = {f"clip_{d:g}lv": 0.0 for d in CLIP_DEPTHS}
    out["gamut_mapped"] = 0.0
    if rec is None:
        return out
    over, gm = rec
    for d in CLIP_DEPTHS:
        m = over[d] if gm is None else (over[d] | gm)
        out[f"clip_{d:g}lv"] = float(m.mean())
    if gm is not None:
        out["gamut_mapped"] = float(gm.mean())
    return out


def clip_budget_share(rec):
    return clip_shares(rec)[f"clip_{CLIP_DEPTH:g}lv"]


def render_steps(sq, ctx0, steps, cache=None):
    """Apply a sequence of (family, parameter) steps to the square.
    Returns (uint8 image, merged clip record). `cache` (a dict) memoises
    every prefix, so cells that share their contrast step render it
    once."""
    img, rec = sq, None
    for i in range(len(steps)):
        key = tuple(steps[:i + 1])
        if cache is not None and key in cache:
            img, rec = cache[key]
            continue
        fam, p = steps[i]
        img, r = apply_step(fam, img, float(p), ctx0)
        rec = merge_clip(rec, r)
        if cache is not None:
            cache[key] = (img, rec)
    return img, rec


# ################################################################
# From test_edit_levels_v5.py, adapted: run state, checks, cache (read-only use of the test run's cache), pool
# ################################################################


# ============================================================
# Run state, checks, output folder, cache
# ============================================================

RUN = {"out": None, "base": None, "checks": [], "warnings": [],
       "t0": time.time(), "timings": {}, "meta": {}}
CHECKS = RUN["checks"]


def B(name):
    """A path in the run folder."""
    return os.path.join(RUN["base"], name)


def settings_dict():
    out = {}
    for k in _SETTING_NAMES:
        v = globals()[k]
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = repr(v)
    return out


def settings_hash():
    s = {k: v for k, v in settings_dict().items() if k not in _NOT_HASHED}
    s["_script"] = "build_stimulus_set_v5.py"
    return hashlib.sha1(json.dumps(s, sort_keys=True).encode()).hexdigest()[:10]


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_checks():
    if RUN["base"] is None:
        return
    pd.DataFrame(CHECKS, columns=["code", "status", "detail"]).to_csv(
        B("checks.csv"), index=False)


def write_meta(status, message=None):
    if RUN["base"] is None:
        return
    try:
        import scipy
        sv = scipy.__version__
    except ImportError:
        sv = None
    meta = {"script": "build_stimulus_set_v5.py", "status": status,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_min": round((time.time() - RUN["t0"]) / 60, 2),
            "settings_hash": RUN["meta"].get("settings_hash"),
            "versions": {"python": platform.python_version(),
                         "numpy": np.__version__, "pandas": pd.__version__,
                         "opencv": cv2.__version__, "pillow": PIL.__version__,
                         "scipy": sv, "platform": platform.platform()},
            "timings_min": RUN["timings"],
            "n_checks": len(CHECKS),
            "checks_not_passed": [c for c in CHECKS
                                  if c["status"] not in ("PASS", "NOTE")],
            "warnings": RUN["warnings"], **RUN["meta"]}
    if message:
        meta["stop_message"] = message
    with open(B("run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)


def stop(msg):
    """Stop the run, leaving checks.csv, run_meta.json and STOPPED.txt."""
    print(f"\nSTOPPED: {msg}")
    if RUN["base"] is not None:
        write_checks()
        write_meta("stopped", msg)
        with open(B("STOPPED.txt"), "w") as f:
            f.write(msg + "\n\nChecks so far:\n")
            for c in CHECKS:
                f.write(f"  {c['code']:<5} {c['status']:<5} {c['detail']}\n")
    raise SystemExit(msg)


def check(code, ok, detail, critical=True):
    """Record a check. A failed critical check stops the run; a failed
    non-critical one is a WARN. (Signature as score_stimuli_on_axes_v4.py,
    whose copied functions call it.)"""
    status = "PASS" if ok else ("FAIL" if critical else "WARN")
    CHECKS.append({"code": code, "status": status, "detail": detail})
    print(f"  [{status}] {code} {detail}")
    if not ok and critical:
        stop(f"Check {code} failed: {detail}")
    if not ok:
        RUN["warnings"].append(f"{code}: {detail}")
    return ok


def note(code, detail):
    """A reported number that is not a pass/fail test."""
    CHECKS.append({"code": code, "status": "NOTE", "detail": detail})
    print(f"  [NOTE] {code} {detail}")


def skip(code, detail):
    CHECKS.append({"code": code, "status": "SKIP", "detail": detail})
    RUN["warnings"].append(f"{code} skipped: {detail}")
    print(f"  [SKIP] {code} {detail}")


def occupied(d):
    return os.path.isdir(d) and any(not f.startswith(".")
                                    for f in os.listdir(d))


def resolve_run_dir(h):
    base = os.path.join(OUTPUT_ROOT, h)
    if not occupied(base):
        return base
    n = 2
    while occupied(f"{base}_{n}"):
        n += 1
    print(f"  NOTE: {base} already holds a run; writing to {base}_{n}.")
    return f"{base}_{n}"


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        print("\n" + "=" * 70 + f"\n{self.name}\n" + "=" * 70)
        return self

    def __exit__(self, *a):
        RUN["timings"][self.name] = round((time.time() - self.t) / 60, 2)
        print(f"  ({self.name}: {(time.time() - self.t) / 60:.1f} min)")


def _src(*fns):
    import inspect
    h = hashlib.sha1()
    for f in fns:
        h.update(inspect.getsource(f).encode("utf-8"))
    return h.hexdigest()[:16]


def cache_get(name, key):
    """Read a cached array set from CACHE_DIR or, failing that, from one of
    CACHE_READ_DIRS (the test run's cache: read, never written)."""
    if not USE_CACHE:
        return None
    for d in [CACHE_DIR] + list(CACHE_READ_DIRS):
        p = os.path.join(d, f"{name}_{key}.npz")
        if not os.path.isfile(p):
            continue
        try:
            RUN["meta"][f"cache_{name}"] = p
            return dict(np.load(p, allow_pickle=False))
        except Exception as exc:
            print(f"  cache {p} unreadable ({exc}); trying the next.")
    return None


def cache_put(name, key, **arrays):
    if not USE_CACHE:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, f"{name}_{key}.npz")
    tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, p)


def ix(i):
    """Level index as a filename tag: -3 -> n3, 0 -> 0, 2 -> p2."""
    return "0" if i == 0 else ("n" if i < 0 else "p") + str(abs(int(i)))


def lab(prefix, i):
    """Level index as a readable label: c-3, c0, c+2."""
    return f"{prefix}0" if i == 0 else f"{prefix}{int(i):+d}"


def stim_name(sid, cx, sx):
    """The stimulus file name: <scene>__cp3__sn1.png (p = plus, n = minus;
    no '+' in a file name, which a web address can turn into a space)."""
    return f"{sid}__c{ix(cx)}__s{ix(sx)}.png"


def test_file(sid, cx, sx):
    """The same image in the test run (its own naming)."""
    return os.path.join(TEST_RUN_DIR, "images", APPROACH, "b_sd", sid,
                        f"c{ix(cx)}_s{ix(sx)}.png")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def run_pool(fn, tasks, payload, label):
    """fn over tasks in N_WORKERS processes, each holding `payload`;
    results returned in task order. N_WORKERS = 1 runs in this process,
    which is the way to get a readable traceback."""
    t0 = time.time()
    out = [None] * len(tasks)
    if N_WORKERS > 1 and len(tasks) > 1:
        with Pool(N_WORKERS, initializer=_init,
                  initargs=(payload,)) as pool:
            for k, (i, r) in enumerate(pool.imap_unordered(
                    _indexed, list(enumerate([(fn, t) for t in tasks])),
                    CHUNK), 1):
                out[i] = r
                print(f"    {label}: {k}/{len(tasks)} "
                      f"({time.time() - t0:.0f}s)", end="\r")
    else:
        _init(payload)
        for k, t in enumerate(tasks, 1):
            out[k - 1] = fn(t)
            print(f"    {label}: {k}/{len(tasks)} "
                  f"({time.time() - t0:.0f}s)", end="\r")
    print()
    return out


W = None


def _init(payload):
    global W
    W = payload
    cv2.setNumThreads(1)
    warnings.filterwarnings("ignore")


def _indexed(arg):
    i, (fn, t) = arg
    return i, fn(t)


def load_square(path):
    """The 512 px central square of a photograph, as v4 renders it."""
    with Image.open(path) as im:
        return np.asarray(central_square(im.convert("RGB"), IMAGE_SIZE),
                          np.uint8)


# ################################################################
# VERBATIM from test_edit_levels_v5.py: corpus, stimulus-path reference, its checks
# ################################################################
def load_categories(n, idx):
    if not (CATEGORY_CSV and os.path.isfile(CATEGORY_CSV)):
        if RESTRICT_TO_KEEP_CATEGORIES or STATS_POPULATION == "everyday":
            stop(f"{CATEGORY_CSV} is required by "
                 "RESTRICT_TO_KEEP_CATEGORIES / STATS_POPULATION.")
        return np.ones(n, bool), 0
    cats = pd.read_csv(CATEGORY_CSV)
    keep = set(cats.loc[cats["keep"].astype(int) == 1, "category"])
    mask = idx.category.isin(keep).to_numpy()
    print(f"  scene categories: {len(keep)} with keep=1, covering "
          f"{int(mask.sum()):,} images")
    return mask, len(keep)


def load_corpus():
    """The index, the stored features (memory-mapped) and the eligibility
    mask of v4 step 1."""
    for p in (INDEX_CSV, FEATURES_NPY, FEATURE_DEFS, FLAGS_CSV):
        if not os.path.isfile(p):
            stop(f"Input not found: {p}")
    idx = pd.read_csv(INDEX_CSV,
                      usecols=["image_id", "image_key", "category",
                               "path", "width", "height"])
    n = len(idx)
    if not (idx.image_id.to_numpy() == np.arange(n)).all():
        stop("image_id must be 0..N-1 in row order.")
    with open(FEATURE_DEFS) as f:
        defs = json.load(f)
    if defs["crop"] != CROP:
        stop(f"The feature list of {FEATURE_DEFS} differs from this "
             f"script's: {defs['crop']} vs {CROP}")
    if defs["settings"]["feature_size"] != FEATURE_SIZE:
        stop("FEATURE_SIZE differs from step 04's.")
    F = np.load(FEATURES_NPY, mmap_mode="r")
    if F.shape != (n, len(CROP)):
        stop(f"features_crop.npy is {F.shape}, expected {(n, len(CROP))}.")
    flags = pd.read_csv(FLAGS_CSV).set_index("image_id").reindex(
        np.arange(n))
    colour = ~flags["is_greyscale"].astype(bool).to_numpy()
    toned = (flags["is_toned"].astype(bool).to_numpy()
             if "is_toned" in flags.columns else np.zeros(n, bool))
    dup = np.zeros(n, bool)
    if EXCLUDE_DUPLICATES and DUP_CSV and os.path.isfile(DUP_CSV):
        dup[pd.read_csv(DUP_CSV)["image_id"].astype(int).to_numpy()] = True
    elif EXCLUDE_DUPLICATES:
        RUN["warnings"].append(f"{DUP_CSV} not found; duplicates kept")
    side = np.minimum(idx.width.to_numpy(), idx.height.to_numpy())
    keep_cat, _ = load_categories(n, idx)
    ok = np.isfinite(np.asarray(F[:, [CROP.index(FEAT_SAT),
                                      CROP.index(FEAT_CON)]], np.float64)
                     ).all(1)
    eligible = ok & (side >= MIN_SIDE) & ~dup
    if EXCLUDE_GREYSCALE:
        eligible &= colour
    if EXCLUDE_TONED:
        eligible &= ~toned
    if RESTRICT_TO_KEEP_CATEGORIES:
        eligible &= keep_cat
    print(f"  {n:,} corpus images; eligible {int(eligible.sum()):,}")
    RUN["meta"]["eligible_images"] = int(eligible.sum())
    return idx, F, eligible


def check_feature_code(idx, F, rng):
    """CHECK F1: the copied feature code reproduces data/features."""
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
    check("F1", used > 0 and worst <= FEATURE_CHECK_RTOL,
          f"copied feature code vs data/features on {used} images: max "
          f"relative difference {worst:.1e} (tol {FEATURE_CHECK_RTOL:g})")


def _ref_task(args):
    i, path = args
    try:
        sq = load_square(path)
    except Exception:
        return None
    v = measure(sq)
    return np.asarray([v[f] for f in CROP], np.float64)


def measure_reference(idx, eligible):
    """Every integer percentile of every feature over v4's 20,000-image
    reference, measured through the stimulus path (original -> 512 ->
    256). Returns a DataFrame indexed by feature: mean, sd, p01 ... p99."""
    rng = np.random.default_rng(REFERENCE_SEED)
    pool = np.where(eligible)[0]
    n = min(N_REFERENCE_IMAGES, len(pool))
    if n < MIN_POPULATION:
        stop(f"only {len(pool)} eligible images; cannot build the reference.")
    ids = np.sort(rng.choice(pool, size=n, replace=False))
    key = hashlib.sha1(json.dumps({
        "ids": hashlib.sha1(np.asarray(ids, np.int64).tobytes()).hexdigest(),
        "image_size": IMAGE_SIZE, "feature_size": FEATURE_SIZE,
        "clip_px": CLIP_PX, "percentiles": REF_PERCENTILES,
        "code": _src(measure, legacy_features, all_features, central_square,
                     clip_box, load_square)}, sort_keys=True)
        .encode()).hexdigest()[:16]
    blob = cache_get("reference", key)
    if blob is not None:
        V = blob["V"]
        print(f"  reference: cache hit ({len(V):,} images)")
    else:
        tasks = [(int(i), idx.path.iat[int(i)]) for i in ids]
        print(f"  re-measuring {len(tasks):,} photographs on the stimulus "
              "path")
        got = run_pool(_ref_task, tasks, None, "reference")
        V = np.asarray([g for g in got if g is not None], np.float64)
        V = V[np.isfinite(V).all(1)]
        cache_put("reference", key, V=V)
    if len(V) < MIN_POPULATION:
        stop(f"only {len(V)} reference images could be measured.")
    cols = {"feature": CROP, "n_images": len(V), "mean": V.mean(0),
            "sd": V.std(0, ddof=1)}
    pct = np.percentile(V, REF_PERCENTILES, axis=0)
    cols.update({f"p{q:02d}": pct[i] for i, q in enumerate(REF_PERCENTILES)})
    st = pd.DataFrame(cols).set_index("feature")
    RUN["meta"]["reference_images"] = int(len(V))
    return st


def check_reference(st):
    """R1: the re-measured reference against v4's stored one; R2: its SDs
    against v5's predictor divisors."""
    if not os.path.isfile(V4_STATS_CSV):
        skip("R1", f"{V4_STATS_CSV} not found")
    else:
        S = pd.read_csv(V4_STATS_CSV)
        S = S[S.population == "stimulus_path"].set_index("feature")
        cols = [c for c in S.columns if c.startswith("p") and c in st.columns]
        worst, where = 0.0, ""
        for f in CROP:
            if f not in S.index:
                continue
            for c in cols + ["sd", "mean"]:
                a, b = float(st.at[f, c]), float(S.at[f, c])
                d = abs(a - b) / max(abs(b), 1e-6)
                if abs(a - b) > 1e-9 and d > worst:
                    worst, where = d, f"{f} {c}: {a:.6g} vs {b:.6g}"
        check("R1", worst <= REF_CHECK_RTOL,
              f"re-measured stimulus-path reference vs {V4_STATS_CSV}, "
              f"{len(cols)} percentiles + mean + sd of {len(CROP)} features:"
              f" worst relative difference {worst:.1e}"
              + (f" ({where})" if where else ""), critical=False)
    ds = abs(st.at[FEAT_SAT, "sd"] - SD_SAT_V5)
    dc = abs(st.at[FEAT_CON, "sd"] - SD_FOG_V5)
    check("R2", ds <= SD_CHECK_TOL and dc <= SD_CHECK_TOL,
          f"stimulus-path SD: sat_mean {st.at[FEAT_SAT, 'sd']:.4f} (v5 "
          f"{SD_SAT_V5}), contrast_local {st.at[FEAT_CON, 'sd']:.4f} (v5 "
          f"{SD_FOG_V5})", critical=False)


# ################################################################
# VERBATIM from test_edit_levels_v5.py: chroma_on (the colour-correction search)
# ################################################################
def chroma_on(img, k, cmax):
    """apply_step("chroma", ...) with the in-gamut ceiling supplied, so a
    search over k computes it once per image instead of once per k."""
    if IDENTITY_SHORT_CIRCUIT and k == 1.0:
        return img, None
    x = m_chroma(img, k, {"cmax": cmax})
    inside = (img > 0) & (img < 255)
    over = {d: (((x < -d) | (x > 255.0 + d)) & inside).any(axis=2)
            for d in CLIP_DEPTHS}
    return to_u8(x), (over, None)


# ################################################################
# VERBATIM from test_edit_levels_v5.py: measurement of one image (features, texture, artefacts)
# ################################################################


# ============================================================
# Measurement of one image: features, texture, artefacts
# ============================================================

def measure_extra(rgb_u8):
    """04c's fine_texture, noise_sigma and edge_overshoot, measured as the
    corpus was (resized to FEATURE_SIZE; 09c's code at scale s = 1)."""
    if rgb_u8.shape[0] != FEATURE_SIZE:
        rgb_u8 = np.asarray(Image.fromarray(rgb_u8).resize(
            (FEATURE_SIZE, FEATURE_SIZE), Image.BILINEAR), dtype=np.uint8)
    s = rgb_u8.shape[0] / 256.0
    g8, g = grey(rgb_u8)
    gx, gy, mag = sobel(g)
    return {"fine_texture": fine_texture(g, s),
            "noise_sigma": noise_sigma(g, s),
            "edge_overshoot": edge_overshoot(g8, g, gx, gy, mag, s)}


def _hp_energy(L, flat):
    r = L - cv2.GaussianBlur(L, (0, 0), NOISE_HP_SIGMA)
    return float(np.mean(r[flat] ** 2)) if flat.any() else np.nan


def _banding(g8):
    lo, hi = np.percentile(g8, BAND_RANGE_PCT)
    h = np.bincount(g8.ravel(), minlength=256).astype(np.float64)
    seg = h[int(np.floor(lo)):int(np.ceil(hi)) + 1]
    empty = int((seg == 0).sum())
    if len(seg) < 3 or seg.mean() <= 0:
        return empty, 0.0
    comb = float(np.mean(np.abs(seg[1:-1] - 0.5 * (seg[:-2] + seg[2:])))
                 / seg.mean())
    return empty, comb


def scene_aux(sq):
    """What the artefact measures need from the original, once per scene."""
    L0 = lab_of(sq)[:, :, 0]
    g8, g = grey(sq)
    edges = cv2.Canny(g8, HALO_CANNY[0], HALO_CANNY[1]) > 0
    k = 2 * HALO_BAND_PX + 1
    band = cv2.dilate(edges.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    w = np.ones((HALO_WINDOW_PX, HALO_WINDOW_PX), np.uint8)
    _, _, ms = sobel(cv2.GaussianBlur(g, (0, 0), NOISE_SMOOTH_SIGMA))
    valid = (g > NOISE_CLIP[0]) & (g < NOISE_CLIP[1])
    flat = np.zeros(g.shape, bool)
    if valid.sum() >= NOISE_MIN_PIXELS:
        flat = valid & (ms <= np.percentile(ms[valid], NOISE_FLAT_PERCENTILE))
    empty0, comb0 = _banding(g8)
    return {"L0": L0, "bin0": np.clip(np.round(L0), 0, 100).astype(np.int64),
            "band": band, "lmin": cv2.erode(L0, w), "lmax": cv2.dilate(L0, w),
            "flat": flat, "E0": _hp_energy(L0, flat),
            "empty0": empty0, "comb0": comb0}


def artefacts(aux, u8):
    """Halo share, noise ratio and banding of an edited 512 px image
    against its original (see the settings for the definitions)."""
    L1 = lab_of(u8)[:, :, 0]
    s = np.bincount(aux["bin0"].ravel(), L1.ravel().astype(np.float64), 101)
    c = np.bincount(aux["bin0"].ravel(), minlength=101)
    have = c > 0
    xs = np.arange(101)
    T = np.interp(xs, xs[have], s[have] / c[have])
    T = np.maximum.accumulate(T)
    hi = np.interp(aux["lmax"], xs, T)
    lo = np.interp(aux["lmin"], xs, T)
    band = aux["band"]
    over = band & ((L1 > hi + HALO_TOL_L) | (L1 < lo - HALO_TOL_L))
    halo = float(over.sum() / band.sum()) if band.any() else np.nan
    E1 = _hp_energy(L1, aux["flat"])
    noise = E1 / aux["E0"] if aux["E0"] and aux["E0"] > 0 else np.nan
    g8, _ = grey(u8)
    empty, comb = _banding(g8)
    return {"halo_share": halo, "noise_ratio": noise,
            "band_new_empty": empty - aux["empty0"],
            "band_comb_diff": comb - aux["comb0"]}


def measure_image(u8, aux):
    """Everything measured on one rendered image."""
    v = measure(u8)
    v.update(measure_extra(u8))
    v.update(artefacts(aux, u8))
    return v


# ################################################################
# VERBATIM from test_edit_levels_v5.py: reach search, level solver, ladders of definition (b)
# ################################################################
CONTRAST_IDX = [-3, -2, -1, 0, 1, 2, 3]


def _artefact_limit_hit(a):
    for k, lim in REACH_ARTEFACT_LIMITS.items():
        if lim is not None and np.isfinite(a.get(k, np.nan)) and a[k] > lim:
            return k
    return None


def _reach_task(task):
    sid, path = task
    sq = load_square(path)
    ctx0 = ctx_of(sq)
    ctx0["square"] = sq           # lets apply_step keep sq's gamut ceiling
    aux = scene_aux(sq)
    base = measure(sq)
    sd = W["sd"]
    grid_rows, reach_rows, tables = [], [], {}
    for fam, direction in W["dims"]:
        feat = FAMILIES[fam]["feature"]
        cross = other_feature(feat)
        s = -1.0 if direction == "down" else 1.0
        G = grid_of(fam, direction, "reach")
        ch, cr, cl = [], [], []
        for gi, p in enumerate(G):
            u8, rec = apply_step(fam, sq, float(p), ctx0)
            v = measure(u8, [feat, cross])
            a = artefacts(aux, u8)
            cs = clip_shares(rec)
            ch.append((v[feat] - base[feat]) / sd[feat])
            cr.append((v[cross] - base[cross]) / sd[cross])
            cl.append(cs[f"clip_{CLIP_DEPTH:g}lv"])
            grid_rows.append({"scene_id": sid, "family": fam,
                              "direction": direction, "grid_index": gi,
                              "param": float(p), "change_sd": ch[-1],
                              "cross_sd": cr[-1], **cs, **a,
                              "artefact_limit_hit": _artefact_limit_hit(a)})
        # where does it stop?
        j_ok, reason, max_gain = len(G) - 1, "end of grid", 0.0
        for j in range(1, len(G)):
            if B_USE_CLIP_BUDGET and cl[j] > CLIP_MAX_NEW:
                j_ok, reason = j - 1, "clipping"
                break
            hit = grid_rows[-len(G) + j]["artefact_limit_hit"]
            if hit:
                j_ok, reason = j - 1, f"artefact ({hit})"
                break
            gain = s * ch[j] - s * ch[j - 1]
            if gain < -1e-9:
                j_ok, reason = j - 1, "feature peaks"
                break
            # the first step is left out of the largest gain: Pillow's
            # edit jumps at the original (check P2), and that jump made
            # every later step look like a plateau
            if (feat == FEAT_SAT and direction == "up" and j > 2
                    and gain < PLATEAU_REL * max_gain):
                j_ok, reason = j - 1, "gamut plateau"
                break
            if j > 1:
                max_gain = max(max_gain, gain)
        params = [float(p) for p in G[:j_ok + 1]]
        changes = list(ch[:j_ok + 1])
        # a clipping limit is refined between the two grid points
        if reason == "clipping":
            lo, hi = float(G[j_ok]), float(G[j_ok + 1])
            c_lo = ch[j_ok]
            for _ in range(REACH_BISECT_ITERS):
                mid = 0.5 * (lo + hi)
                u8, rec = apply_step(fam, sq, mid, ctx0)
                if clip_budget_share(rec) <= CLIP_MAX_NEW:
                    lo = mid
                    c_lo = (measure(u8, [feat])[feat] - base[feat]) / sd[feat]
                else:
                    hi = mid
            if lo != params[-1] and s * c_lo >= s * changes[-1]:
                params.append(lo)
                changes.append(c_lo)
        tables[(fam, direction)] = (np.array(params), np.array(changes))
        reach_rows.append({"scene_id": sid, "family": fam,
                           "direction": direction,
                           "feature": feat, "reach_sd": abs(changes[-1]),
                           "reach_param": params[-1], "limit": reason,
                           "grid_points_ok": j_ok + 1})
    return {"scene_id": sid, "grid": grid_rows, "reach": reach_rows,
            "tables": tables}


def _f_sd(sq, ctx0, steps, feat, base_val, sd):
    u8, _ = render_steps(sq, ctx0, steps)
    return (measure(u8, [feat])[feat] - base_val) / sd


def solve_target(sq, ctx0, make_steps, feat, base_val, sd, target,
                 lo, hi, f_lo=None, f_hi=None):
    """Bisection on the parameter p of make_steps(p) until the measured
    change of `feat` (corpus SD) is within SOLVE_TOL_SD of `target`.
    Assumes the change is monotone between lo and hi and that hi is past
    the target (otherwise returns hi, capped). Returns (p, achieved,
    capped)."""
    s = 1.0 if target >= 0 else -1.0
    if f_hi is None:
        f_hi = _f_sd(sq, ctx0, make_steps(hi), feat, base_val, sd)
    if s * f_hi < s * target - SOLVE_TOL_SD:
        return hi, f_hi, True
    if f_lo is None:
        f_lo = _f_sd(sq, ctx0, make_steps(lo), feat, base_val, sd)
    best = (hi, f_hi) if abs(f_hi - target) < abs(f_lo - target) \
        else (lo, f_lo)
    for _ in range(SOLVE_ITERS):
        if abs(best[1] - target) <= SOLVE_TOL_SD:
            break
        mid = 0.5 * (lo + hi)
        fm = _f_sd(sq, ctx0, make_steps(mid), feat, base_val, sd)
        if abs(fm - target) < abs(best[1] - target):
            best = (mid, fm)
        if s * fm < s * target:
            lo = mid
        else:
            hi = mid
    return best[0], best[1], False


def solve_from_table(sq, ctx0, fam, table, feat, base_val, sd, target,
                     prefix=(), suffix=()):
    """Definition (b): the parameter of `fam` that moves the scene's
    feature by `target` SD, bracketed by the reach table of pass 1. A
    target beyond the reach gives the reach parameter, capped."""
    params, changes = table
    s = 1.0 if target >= 0 else -1.0

    def mk(p):
        return tuple(prefix) + ((fam, float(p)),) + tuple(suffix)
    if not prefix and not suffix:
        hit = np.where(s * changes >= s * target)[0]
        if not len(hit):
            return float(params[-1]), float(changes[-1]), True
        j = int(hit[0])
        if j == 0:
            return float(params[0]), float(changes[0]), False
        return solve_target(sq, ctx0, mk, feat, base_val, sd, target,
                            float(params[j - 1]), float(params[j]),
                            float(changes[j - 1]), float(changes[j]))
    return solve_target(sq, ctx0, mk, feat, base_val, sd, target,
                        float(params[0]), float(params[-1]))


def solve_comp(sq, ctx0, contrast_steps, base_sat, sd_sat):
    """The chroma factor that brings the scene's sat_mean back to the
    original's after `contrast_steps`. Returns (k, residual SD, capped)."""
    img_c, _ = render_steps(sq, ctx0, contrast_steps)
    cm = cmax_of_img(img_c)

    def f(k):
        u8, _ = chroma_on(img_c, float(k), cm)
        return (measure(u8, [FEAT_SAT])[FEAT_SAT] - base_sat) / sd_sat
    lo, hi = COMP_K_RANGE
    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0:
        return lo, f_lo, True
    if f_hi < 0:
        return hi, f_hi, True
    best = (1.0, f(1.0))
    for _ in range(COMP_ITERS):
        if abs(best[1]) <= SOLVE_TOL_SD:
            break
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < abs(best[1]):
            best = (mid, fm)
        if fm < 0:
            lo = mid
        else:
            hi = mid
    return best[0], best[1], False


def _level(i, name, steps, target, capped, status, fam=None, param=None,
           comp_k=np.nan, comp_capped=False, comp_resid=np.nan):
    return {"index": i, "name": name, "steps": tuple(steps),
            "target_sd": target, "capped": bool(capped or comp_capped),
            "status": status, "family": fam, "param": param,
            "comp_k": comp_k, "comp_capped": bool(comp_capped),
            "comp_residual_sd": comp_resid}


def ladders_b(approach, sq, ctx0, base, tables, mode="b"):
    """Definition (b): per-scene parameters for +-1/3, 2/3, 3/3 of the
    rendered span. mode "r", definition (r): the features in R_FEATURES
    step by equal RATIOS instead (see the settings), solved the same way
    on the change in corpus SD each ratio means for this scene."""
    a = APPROACHES[approach]
    sd = W["sd"]
    out = {}
    for feature, fd, fu in (("contrast", a["contrast_down"], a["contrast_up"]),
                            ("colour", a["colour"], a["colour"])):
        span = W["span"][(approach, feature)]
        by_ratio = mode == "r" and feature in R_FEATURES
        lad = []
        for i in CONTRAST_IDX:
            if i == 0:
                lad.append(_level(0, "orig", (), 0.0, False, "original"))
                continue
            fam, direction = (fd, "down") if i < 0 else (fu, "up")
            feat = FAMILIES[fam]["feature"]
            frac = LEVEL_FRACTIONS[abs(i) - 1]
            if by_ratio:
                ratio = W["ratio"][feat] ** (frac if i < 0 else -frac)
                target = base[feat] * (ratio - 1.0) / sd[feat]
                name = f"x{ratio:.3f}"
            else:
                target = np.sign(i) * frac * span
                name = f"{target:+.2f}SD"
            tab = tables[(fam, direction)]
            p, ach, cap = solve_from_table(sq, ctx0, fam, tab, feat,
                                           base[feat], sd[feat], target)
            steps = [(fam, p)]
            kc, kcap, kres = np.nan, False, np.nan
            if feature == "contrast" and a["compensate"]:
                for rnd in range(COMP_ROUNDS):
                    kc, kres, kcap = solve_comp(sq, ctx0, [(fam, p)],
                                                base[FEAT_SAT], sd[FEAT_SAT])
                    if rnd < COMP_ROUNDS - 1 and not cap:
                        p, ach, cap = solve_from_table(
                            sq, ctx0, fam, tab, feat, base[feat], sd[feat],
                            target, suffix=(("chroma", kc),))
                steps.append(("chroma", kc))
            status = (f"capped at reach ({abs(ach):.3f} SD)" if cap
                      else "ok")
            lad.append(_level(i, name, steps, target, cap,
                              status, fam, p, kc, kcap, kres))
        out[feature] = lad
    return out


# ################################################################
# VERBATIM from test_edit_levels_v5.py: derived measures (changes, z, distance from typical, envelope)
# ################################################################
def add_derived(cells, bases, st):
    """Changes from the scene's original, distances from typical, and the
    envelope, for every rendered image."""
    b = bases.set_index("scene_id")
    for feat, tag in ((FEAT_SAT, "sat"), (FEAT_CON, "con")):
        sd, med = float(st.at[feat, "sd"]), float(st.at[feat, "p50"])
        cells[f"d_{tag}"] = cells[feat] - cells.scene_id.map(b[feat])
        cells[f"d_{tag}_sd"] = cells[f"d_{tag}"] / sd
        cells[f"z_{tag}"] = (cells[feat] - med) / sd
    for f in DESCRIPTIVE:
        if f in cells.columns and f in st.index:
            cells[f"d_{f}_sd"] = ((cells[f] - cells.scene_id.map(b[f]))
                                  / float(st.at[f, "sd"]))
        elif f in cells.columns:
            cells[f"d_{f}"] = cells[f] - cells.scene_id.map(b[f])
    cells["dist_corpus"] = np.hypot(cells.z_sat, cells.z_con)
    cells["dist_corpus_sat"] = cells.z_sat.abs()
    cells["dist_corpus_con"] = cells.z_con.abs()
    cells["dist_orig"] = np.hypot(cells.d_sat_sd, cells.d_con_sd)
    cells["dist_orig_sat"] = cells.d_sat_sd.abs()
    cells["dist_orig_con"] = cells.d_con_sd.abs()
    out_n = np.zeros(len(cells), int)
    out_f = [[] for _ in range(len(cells))]
    for f in ENVELOPE_FEATURES:
        lo, hi = float(st.at[f, ENVELOPE_LO]), float(st.at[f, ENVELOPE_HI])
        v = cells[f].to_numpy(float)
        o = (v < lo) | (v > hi)
        out_n += o
        for i in np.where(o)[0]:
            out_f[i].append(f)
    cells["envelope_n"] = out_n
    cells["envelope_features"] = [";".join(x) for x in out_f]
    return cells


def envelope_violations(cells, st):
    rows = []
    for f in ENVELOPE_FEATURES:
        lo, hi = float(st.at[f, ENVELOPE_LO]), float(st.at[f, ENVELOPE_HI])
        sd = float(st.at[f, "sd"])
        v = cells[f].to_numpy(float)
        o = (v < lo) | (v > hi)
        if not o.any():
            continue
        sub = cells.loc[o, ["approach", "definition", "scene_id", "cx", "sx",
                            "filename"]].copy()
        sub["feature"] = f
        sub["value"] = v[o]
        sub["outside_by_sd"] = np.where(v[o] > hi, v[o] - hi, v[o] - lo) / sd
        rows.append(sub)
    return (pd.concat(rows, ignore_index=True) if rows else
            pd.DataFrame(columns=["approach", "definition", "scene_id", "cx",
                                  "sx", "filename", "feature", "value",
                                  "outside_by_sd"]))


# ################################################################
# NEW: scenes, sources, one scene's images, the manifest
# ################################################################


# ============================================================
# The scenes and their source photographs
# ============================================================

def load_scenes(idx):
    """The 47 scenes from the v4 manifest: category, group, indoor, the
    source photograph, v4's unedited image, and the Places365 file ID."""
    if not os.path.isfile(V4_MANIFEST_CSV):
        stop(f"{V4_MANIFEST_CSV} not found.")
    man = pd.read_csv(V4_MANIFEST_CSV)
    for c in ("scene_id", "category", "source_path", "kind", "chroma",
              "haze", "filename"):
        if c not in man.columns:
            stop(f"{V4_MANIFEST_CSV} has no {c!r} column.")
    agg = {"category": ("category", "first"),
           "source_path": ("source_path", "first"),
           "n_paths": ("source_path", "nunique")}
    for c in ("group", "indoor"):
        if c in man.columns:
            agg[c] = (c, "first")
    sc = man.groupby("scene_id", sort=True).agg(**agg).reset_index()
    for c in ("group", "indoor"):
        if c not in sc.columns:
            sc[c] = ""
    got, want = set(sc.scene_id), set(SCENES)
    missing = [p for p in sc.source_path if not os.path.isfile(p)]
    check("S1", got == want and len(SCENES) == N_SCENES_EXPECTED
          and (sc.n_paths == 1).all() and not missing,
          f"{len(want)} scenes in SCENES, {len(got)} in the v4 manifest, "
          f"{len(got & want)} in both; one source photograph each, "
          f"{len(sc) - len(missing)} on disk"
          + (f"; only in SCENES: {sorted(want - got)[:3]}" if want - got
             else "")
          + (f"; only in the manifest: {sorted(got - want)[:3]}"
             if got - want else ""))
    sc = sc[sc.scene_id.isin(want)].reset_index(drop=True)
    # v4's unedited image of each scene (chroma 1, haze 0, a grid cell)
    orig = man[(man.kind == "grid") & np.isclose(man.chroma, 1.0)
               & np.isclose(man.haze, 0.0)]
    if "contrast_gain" in orig.columns:
        orig = orig[np.isclose(orig.contrast_gain.fillna(1.0), 1.0)]
    v4o = orig.groupby("scene_id").filename.first()
    sc["v4_unedited_file"] = sc.scene_id.map(v4o).fillna("")
    # Places365 file IDs, relative to the dataset's image folder
    paths = [os.path.normpath(p) for p in sc.source_path]
    root = (os.path.normpath(PLACES365_ROOT) if PLACES365_ROOT else
            os.path.commonpath([os.path.dirname(p) for p in paths]))
    sc["places365_file"] = [os.path.relpath(p, root).replace(os.sep, "/")
                            for p in paths]
    RUN["meta"]["places365_root_name"] = os.path.basename(root)
    key = dict(zip(idx.path.map(os.path.normpath), idx.image_key))
    wh = dict(zip(idx.path.map(os.path.normpath),
                  zip(idx.width, idx.height)))
    sc["image_key"] = [key.get(p, "") for p in paths]
    sc["source_width"] = [wh.get(p, (np.nan, np.nan))[0] for p in paths]
    sc["source_height"] = [wh.get(p, (np.nan, np.nan))[1] for p in paths]
    sc["source_sha256"] = [sha256_file(p) if os.path.isfile(p) else ""
                           for p in sc.source_path]
    sc["source_bytes"] = [os.path.getsize(p) if os.path.isfile(p) else -1
                          for p in sc.source_path]
    return sc


# ============================================================
# One scene: levels, images, measurements
# ============================================================

def dims_needed():
    a = APPROACHES[APPROACH]
    return sorted({(a["contrast_down"], "down"), (a["contrast_up"], "up"),
                   (a["colour"], "down"), (a["colour"], "up")})


def _build_task(task):
    """Solve the 12 levels of one scene (the test script's ladders_b,
    unchanged), render the 49 cells exactly as the test script rendered
    them, save each PNG, and measure it FROM THE SAVED FILE."""
    sid, path, tables, v4_file = task
    out_dir = W["base_dir"]
    sd = W["sd"]
    sq = load_square(path)
    ctx0 = ctx_of(sq)
    ctx0["square"] = sq           # lets apply_step keep sq's gamut ceiling
    aux = scene_aux(sq)
    base = measure_image(sq, aux)
    res = {"scene_id": sid, "base": {"scene_id": sid, **base}, "cells": [],
           "ladders": [], "identity": [], "unedited": {}}

    lad = ladders_b(APPROACH, sq, ctx0, base, tables, DEFINITION)
    for feature in ("contrast", "colour"):
        for lv in lad[feature]:
            res["ladders"].append({
                "scene_id": sid, "feature": feature, "index": lv["index"],
                "name": lv["name"], "family": lv["family"],
                "param": lv["param"], "comp_k": lv["comp_k"],
                "target_sd": lv["target_sd"], "capped": lv["capped"],
                "comp_capped": lv["comp_capped"],
                "comp_residual_sd": lv["comp_residual_sd"],
                "status": lv["status"], "steps": json.dumps(lv["steps"])})

    cache = {}
    for cl in lad["contrast"]:
        img_c, rec_c = render_steps(sq, ctx0, list(cl["steps"]), cache)
        for sl in lad["colour"]:
            u8, rec = img_c, rec_c
            for fam, p in sl["steps"]:
                u8, r = apply_step(fam, u8, float(p), ctx0)
                rec = merge_clip(rec, r)
            name = stim_name(sid, cl["index"], sl["index"])
            rel = f"{STIMULI_SUBDIR}/{name}"
            dst = os.path.join(out_dir, STIMULI_SUBDIR, name)
            Image.fromarray(u8).save(dst)
            # measured from the file as saved, as the trial scripts will
            with Image.open(dst) as im:
                mode, size = im.mode, im.size
                f8 = np.asarray(im.convert("RGB"), np.uint8)
            v = measure_image(f8, aux)
            v.update(clip_shares(rec))
            tf = test_file(sid, cl["index"], sl["index"])
            sha = sha256_file(dst)
            ti = (sha == sha256_file(tf)) if (TEST_RUN_DIR and
                                               os.path.isfile(tf)) else None
            res["cells"].append({
                "scene_id": sid, "approach": APPROACH,
                "definition": DEFINITION_NAME,
                "cx": cl["index"], "sx": sl["index"],
                "contrast_family": cl["family"] or "",
                "contrast_param": cl["param"], "comp_k": cl["comp_k"],
                "comp_residual_sd": cl["comp_residual_sd"],
                "colour_family": sl["family"] or "",
                "colour_k": sl["param"],
                "contrast_target_sd": cl["target_sd"],
                "colour_target_sd": sl["target_sd"],
                "contrast_capped": cl["capped"],
                "colour_capped": sl["capped"],
                "capped": bool(cl["capped"] or sl["capped"]),
                "filename": name, "relpath": rel,
                "png_mode": mode, "png_width": size[0],
                "png_height": size[1],
                "pixels_equal_in_memory": bool(np.array_equal(f8, u8)),
                "png_sha256": sha, "test_run_byte_identical": ti, **v})

    # identity: each family used, at its "no change" setting, without the
    # short cut (as the test script's check I1)
    for fam in sorted({f for f, _ in dims_needed()}):
        F = FAMILIES[fam]
        p = F["identity"]
        ctx = dict(ctx0, cmax=cmax_of_img(sq)) if fam == "chroma" else ctx0
        x, _ = fam_call(fam, sq, p, ctx)
        u8 = F["render"](sq, p) if F["render"] else to_u8(x)
        vv = measure(u8, [FEAT_SAT, FEAT_CON])
        dd = np.abs(u8.astype(np.int16) - sq.astype(np.int16))
        res["identity"].append({
            "scene_id": sid, "family": fam, "param": p,
            "max_abs_level": int(dd.max()),
            "share_pixels_changed": float((dd > 0).any(axis=2).mean()),
            "d_sat_sd": (vv[FEAT_SAT] - base[FEAT_SAT]) / sd[FEAT_SAT],
            "d_con_sd": (vv[FEAT_CON] - base[FEAT_CON]) / sd[FEAT_CON]})

    # the unedited cell against the square and against v4's unedited PNG
    dst0 = os.path.join(out_dir, STIMULI_SUBDIR, stim_name(sid, 0, 0))
    with Image.open(dst0) as im:
        o8 = np.asarray(im.convert("RGB"), np.uint8)
    u = {"equals_square": bool(np.array_equal(o8, sq)),
         "v4_file": v4_file, "v4_pixels_identical": None,
         "v4_bytes_identical": None}
    fp = os.path.join(V4_STIMULI_DIR, v4_file) if v4_file else ""
    if fp and os.path.isfile(fp):
        with Image.open(fp) as im:
            ref = np.asarray(im.convert("RGB"), np.uint8)
        u["v4_pixels_identical"] = bool(np.array_equal(ref, o8))
        u["v4_bytes_identical"] = sha256_file(fp) == sha256_file(dst0)
    res["unedited"] = u
    return res


DEFINITION_NAME = "b_sd"


def cell_plan():
    """The 49 cells every scene has. Parameters are per scene, so they are
    in the manifest, not here."""
    a = APPROACHES[APPROACH]
    rows = []
    for cx in CONTRAST_IDX:
        for sx in CONTRAST_IDX:
            fc = (a["contrast_down"] if cx < 0 else
                  a["contrast_up"] if cx > 0 else "none")
            rows.append({
                "contrast_level": lab("c", cx), "colour_level": lab("s", sx),
                "contrast_step": cx, "colour_step": sx,
                "is_original": cx == 0 and sx == 0,
                "use": "pair", "kind": "cell",
                "contrast_family": fc,
                "colour_correction": ("chroma factor restoring the scene's "
                                      "sat_mean" if cx else "none"),
                "colour_family": a["colour"] if sx else "none",
                "contrast_target_sd": np.sign(cx) * (
                    LEVEL_FRACTIONS[abs(cx) - 1] * B_SPAN_SD if cx else 0.0),
                "colour_target_sd": np.sign(sx) * (
                    LEVEL_FRACTIONS[abs(sx) - 1] * B_SPAN_SD if sx else 0.0),
                "filename_pattern": f"<scene_id>__c{ix(cx)}__s{ix(sx)}.png",
                "note": "parameters differ per scene; see "
                        "stimulus_manifest.csv"})
    return pd.DataFrame(rows)


def build_manifest(cells, scenes):
    m = cells.copy()
    sc = scenes.set_index("scene_id")
    for c in ("category", "group", "indoor", "places365_file"):
        m[c] = m.scene_id.map(sc[c])
    for c in ("contrast_family", "colour_family"):
        m[c] = m[c].replace("", np.nan).fillna("none")
    m["use"] = "pair"
    m["kind"] = "cell"
    m["contrast_level"] = [lab("c", v) for v in m.cx]
    m["colour_level"] = [lab("s", v) for v in m.sx]
    m["contrast_step"] = m.cx.astype(int)
    m["colour_step"] = m.sx.astype(int)
    m["is_original"] = (m.cx == 0) & (m.sx == 0)
    lead = ["scene_id", "category", "group", "indoor", "approach",
            "definition", "use", "kind", "contrast_level", "colour_level",
            "contrast_step", "colour_step", "is_original", "filename",
            "places365_file", "contrast_family", "contrast_param", "comp_k",
            "comp_residual_sd", "colour_family", "colour_k",
            "contrast_target_sd", "colour_target_sd", "d_con_sd", "d_sat_sd",
            "contrast_dev_sd", "colour_dev_sd", "capped", "contrast_capped",
            "colour_capped", "z_sat", "z_con", "dist_corpus",
            "dist_corpus_sat", "dist_corpus_con", "dist_orig",
            "dist_orig_sat", "dist_orig_con"]
    lead += CROP + EXTRA_TEXTURE
    drop = {"cx", "sx", "relpath", "png_mode", "png_width", "png_height",
            "pixels_equal_in_memory", "test_run_byte_identical"}
    rest = [c for c in m.columns if c not in lead and c not in drop]
    m = m[lead + rest]
    return (m.sort_values(["scene_id", "contrast_step", "colour_step"],
                          kind="stable").reset_index(drop=True))


# ============================================================
# Checks
# ============================================================

def _norm(v):
    """A setting as JSON would store it (tuples become lists)."""
    return json.loads(json.dumps(v, default=_json_default))


def check_test_settings():
    """T0: the copied settings equal the test run's."""
    p = os.path.join(TEST_RUN_DIR or "", "settings.json")
    if not (TEST_RUN_DIR and os.path.isfile(p)):
        skip("T0", f"{p} not found; the copied settings are not compared")
        return
    with open(p) as f:
        ts = json.load(f)["settings"]
    diff = []
    for k in COPIED_SETTINGS:
        if k not in ts:
            diff.append(f"{k} (not in the test run)")
        elif _norm(globals()[k]) != _norm(ts[k]):
            diff.append(f"{k}: {globals()[k]!r} vs {ts[k]!r}")
    ta = ts.get("APPROACHES", {}).get(APPROACH)
    mine = dict(APPROACHES[APPROACH])
    if ta is None:
        diff.append(f"APPROACHES[{APPROACH}] not in the test run")
    else:
        ta = {k: v for k, v in ta.items() if k != "definitions"}
        if _norm(ta) != _norm(mine):
            diff.append(f"APPROACHES[{APPROACH}]: {mine} vs {ta}")
    check("T0", not diff,
          f"{len(COPIED_SETTINGS)} copied settings and the approach vs "
          f"{p}: " + ("all equal" if not diff else
                      f"{len(diff)} differ: " + "; ".join(diff[:5])))


def checks_after_render(cells, lad, ident, uned):
    n_exp = len(SCENES) * len(CONTRAST_IDX) ** 2
    # --- files ---
    sdir = B(STIMULI_SUBDIR)
    on_disk = set(os.listdir(sdir))
    listed = set(cells.filename)
    check("E1", len(cells) == n_exp and cells.filename.is_unique
          and on_disk == listed,
          f"{len(cells):,} images listed (expected {n_exp:,}), names "
          f"unique, {len(on_disk):,} files in {STIMULI_SUBDIR}/, "
          f"{len(listed & on_disk):,} listed and present"
          + (f"; not on disk: {sorted(listed - on_disk)[:2]}"
             if listed - on_disk else "")
          + (f"; not listed: {sorted(on_disk - listed)[:2]}"
             if on_disk - listed else ""))
    ok = ((cells.png_width == IMAGE_SIZE) & (cells.png_height == IMAGE_SIZE)
          & (cells.png_mode == "RGB"))
    check("Z1", bool(ok.all()),
          f"{int(ok.sum()):,} of {len(cells):,} PNGs are {IMAGE_SIZE} x "
          f"{IMAGE_SIZE} RGB")
    check("M2", bool(cells.pixels_equal_in_memory.all()),
          f"every PNG read back equals the image measured and saved "
          f"({int(cells.pixels_equal_in_memory.sum()):,} of {len(cells):,});"
          " every measurement in the manifest is taken from the saved file")
    # --- byte identity with the test run ---
    ti = cells.test_run_byte_identical
    have = ti.notna()
    if not have.any():
        skip("T1", f"no test-run images under {TEST_RUN_DIR}/images/"
             f"{APPROACH}/b_sd/; byte identity not checked")
    else:
        bad = cells[have & (ti != True)]           # noqa: E712
        check("T1", have.all() and bad.empty,
              f"{int((ti == True).sum()):,} of {len(cells):,} images "  # noqa
              f"byte-identical to test run {TEST_RUN_ID} "
              f"({int(have.sum()):,} compared)"
              + (f"; first difference: {bad.filename.iat[0]}"
                 if len(bad) else ""))
    # --- parameters against the test run ---
    p = os.path.join(TEST_RUN_DIR or "", "ladders.csv")
    if TEST_RUN_DIR and os.path.isfile(p):
        t = pd.read_csv(p)
        t = t[(t.approach == APPROACH) & (t.definition == DEFINITION_NAME)]
        mg = lad.merge(t, on=["scene_id", "feature", "index"],
                       suffixes=("", "_t"))
        worst = 0.0
        for c in ("param", "comp_k"):
            a, b = mg[c].to_numpy(float), mg[c + "_t"].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            if both.any():
                worst = max(worst, float(np.max(
                    np.abs(a[both] - b[both])
                    / np.maximum(np.abs(b[both]), 1e-9))))
            if (np.isfinite(a) != np.isfinite(b)).any():
                worst = np.inf
        check("T2", len(mg) == len(lad) and worst <= 1e-5,
              f"level parameters and colour-correction factors vs {p}: "
              f"{len(mg):,} of {len(lad):,} levels matched, worst relative "
              f"difference {worst:.1e} (the test run stored 6 significant "
              "digits)", critical=False)
    else:
        skip("T2", f"{p} not found")
    # --- identity ---
    worst_lv = int(ident.max_abs_level.max())
    worst_sd = float(np.nanmax(np.abs(ident[["d_sat_sd", "d_con_sd"]]
                                      .to_numpy())))
    check("I1", worst_lv <= 1 and worst_sd <= 0.005,
          "each edit at its no-change setting, without the short cut: max "
          f"pixel change {worst_lv} level, max change of sat_mean / "
          f"contrast_local {worst_sd:.4f} SD. The level-0 images use the "
          "short cut, so they are the photograph itself (O1)",
          critical=False)
    eq = uned.equals_square
    check("O1", bool(eq.all()),
          f"the unedited cell (c0, s0) is the scene's {IMAGE_SIZE} px "
          f"square, pixel for pixel, in {int(eq.sum())} of {len(uned)} "
          "scenes")
    vb = uned.v4_bytes_identical
    if vb.notna().any():
        check("O2", bool((vb == True).all()),                 # noqa: E712
              f"the unedited cell is byte-identical to v4's unedited image "
              f"(chroma 1, haze 0) in {int((vb == True).sum())} of "  # noqa
              f"{len(uned)} scenes", critical=False)
    else:
        skip("O2", f"v4's unedited images not found in {V4_STIMULI_DIR}")
    # --- every level reached ---
    capped = lad[lad.capped]
    check("L0", capped.empty,
          f"levels reached: {len(lad) - len(capped)} of {len(lad)} "
          f"({len(SCENES)} scenes x 2 features x 7 levels), "
          f"{len(capped)} capped"
          + (f", e.g. {capped.scene_id.iat[0]} {capped.feature.iat[0]} "
             f"{capped.name.iat[0]}" if len(capped) else ""))
    # --- monotone ladders ---
    bad = []
    for sid, g in cells.groupby("scene_id"):
        c = g[g.sx == 0].sort_values("cx").d_con_sd.to_numpy()
        s = g[g.cx == 0].sort_values("sx").d_sat_sd.to_numpy()
        if not (np.diff(c) > 0).all():
            bad.append(f"{sid} contrast")
        if not (np.diff(s) > 0).all():
            bad.append(f"{sid} colour")
    check("M1", not bad,
          f"both ladders through the original strictly in order in "
          f"{len(SCENES) - len({b.split()[0] for b in bad})} of "
          f"{len(SCENES)} scenes" + (f"; not: {bad[:4]}" if bad else ""))
    rows_bad = 0
    for (sid, sx), g in cells.groupby(["scene_id", "sx"]):
        rows_bad += not (np.diff(g.sort_values("cx").d_con_sd) > 0).all()
    cols_bad = 0
    for (sid, cx), g in cells.groupby(["scene_id", "cx"]):
        cols_bad += not (np.diff(g.sort_values("sx").d_sat_sd) > 0).all()
    note("M1g", f"across the whole 7 x 7 grid: contrast in order along "
         f"{len(SCENES) * 7 - rows_bad} of {len(SCENES) * 7} colour "
         f"columns, colour in order along {len(SCENES) * 7 - cols_bad} of "
         f"{len(SCENES) * 7} contrast rows")
    # --- targets ---
    uc = lad[(lad["index"] != 0) & ~lad.capped].copy()
    uc["err"] = (uc.achieved_sd - uc.target_sd).abs()
    off = uc[uc.err > LEVEL_CHECK_TOL_SD]
    w = uc.groupby("feature").err.max()
    check("T3", off.empty,
          f"{len(uc) - len(off)} of {len(uc)} levels within "
          f"{LEVEL_CHECK_TOL_SD:g} SD of their target; largest miss: "
          f"contrast {w.get('contrast', np.nan):.4f} SD, colour "
          f"{w.get('colour', np.nan):.4f} SD; {len(off)} misses, all "
          f"{'contrast' if (off.feature == 'contrast').all() else 'mixed'}"
          " (the colour correction is the last step of a contrast level, "
          "and contrast is not re-solved after it; see level_deviations"
          ".csv)", critical=False)
    cx = cells[(cells.cx != 0) & (cells.sx != 0)]
    note("T4", f"off the two ladders (both features edited, {len(cx):,} "
         "cells) each feature lands away from its level's nominal value, "
         "because "
         "the colour level is solved on the original and applied after the "
         "contrast edit: contrast |dev| median "
         f"{cx.contrast_dev_sd.abs().median():.3f}, max "
         f"{cx.contrast_dev_sd.abs().max():.3f} SD; colour |dev| median "
         f"{cx.colour_dev_sd.abs().median():.3f}, max "
         f"{cx.colour_dev_sd.abs().max():.3f} SD (cell_deviations.csv). The "
         "trial scripts use each image's measured values, not the nominal "
         "levels.")
    con_only = cells[(cells.sx == 0) & (cells.cx != 0)]
    col_only = cells[(cells.cx == 0) & (cells.sx != 0)]
    note("X1", "cross-talk on the ladders: |change of sat_mean| at the "
         f"contrast levels median {con_only.d_sat_sd.abs().median():.3f}, "
         f"max {con_only.d_sat_sd.abs().max():.3f} SD; |change of "
         f"contrast_local| at the colour levels median "
         f"{col_only.d_con_sd.abs().median():.3f}, max "
         f"{col_only.d_con_sd.abs().max():.3f} SD")


def relative_path_scan(root):
    """A0: no absolute path of this machine in any text file written."""
    needles = {os.path.abspath(os.getcwd()), os.path.expanduser("~")}
    needles = {n for n in needles if n and n != os.sep}
    hits = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith((".csv", ".json", ".txt", ".md", ".cff",
                               ".sha256", ".py")):
                continue
            p = os.path.join(dp, f)
            with open(p, encoding="utf-8", errors="replace") as fh:
                t = fh.read()
            for n in needles:
                if n in t:
                    hits.append(os.path.relpath(p, root))
                    break
    return hits


# ============================================================
# Report, README, citation, freeze
# ============================================================

def _wrap(s, width=72, indent="  "):
    import textwrap
    return textwrap.fill(s, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def _tbl(df, fmt="{:.3f}"):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if not np.isfinite(v)
                            else fmt.format(v))
    return "\n".join("    " + ln for ln in d.to_string(index=False)
                     .splitlines())


def level_summary(man):
    """Per contrast level (colour at the original) and per colour level
    (contrast at the original): the measured changes, clipping and
    artefacts, median and worst over the scenes."""
    rows = []
    for feat, step, other in (("contrast", "contrast_step", "colour_step"),
                              ("colour", "colour_step", "contrast_step")):
        g = man[man[other] == 0]
        for k, x in g.groupby(step):
            own = x.d_con_sd if feat == "contrast" else x.d_sat_sd
            cross = x.d_sat_sd if feat == "contrast" else x.d_con_sd
            rows.append({
                "ladder": feat, "level": (lab("c", k) if feat == "contrast"
                                          else lab("s", k)),
                "own_median_sd": own.median(), "own_min_sd": own.min(),
                "own_max_sd": own.max(),
                "cross_median_sd": cross.median(),
                "clip_median": x[f"clip_{CLIP_DEPTH:g}lv"].median(),
                "clip_max": x[f"clip_{CLIP_DEPTH:g}lv"].max(),
                "halo_max": x.halo_share.max(),
                "noise_ratio_max": x.noise_ratio.max(),
                "dist_orig_median": x.dist_orig.median(),
                "dist_corpus_median": x.dist_corpus.median()})
    return pd.DataFrame(rows)


def write_report(T):
    man, st = T["manifest"], T["st"]
    L = []
    w = L.append
    w(f"{SET_NAME.upper()}: BUILD REPORT")
    w("=" * 72)
    w(f"run folder      {T['out_rel']}")
    w(f"settings hash   {T['hash']}")
    w(f"created         {time.strftime('%Y-%m-%d %H:%M')}")
    w(f"method          {APPROACH}, definition (b) equal SD steps, span "
      f"+-{B_SPAN_SD:g} SD")
    w(f"scenes          {man.scene_id.nunique()}, {len(CONTRAST_IDX)} x "
      f"{len(CONTRAST_IDX)} cells each, {len(man):,} images")
    w(f"test run        {TEST_RUN_ID}")
    w("")
    w("WHY THIS SET")
    w(_wrap(DECISION_NOTE))
    w("")
    w("1. CHECKS")
    w("-" * 72)
    for c in CHECKS:
        w(f"  {c['code']:<5} {c['status']:<5} {c['detail']}")
    w("  A0 (no absolute paths in any text file) runs after this report "
      "is written;\n  FROZEN.json records it.")
    w("")
    w("2. THE REFERENCE (stimulus path: photograph -> 512 px -> 256 px)")
    w("-" * 72)
    for f in (FEAT_SAT, FEAT_CON):
        w(f"  {f:<15} median {st.at[f, 'p50']:.3f}  SD {st.at[f, 'sd']:.3f}"
          f"  p01 {st.at[f, 'p01']:.2f}  p99 {st.at[f, 'p99']:.2f}")
    w("  z_sat, z_con = (feature - corpus median) / corpus SD.")
    w("")
    w(f"3. THE LADDERS AS RENDERED (median and range over the "
      f"{man.scene_id.nunique()} scenes)")
    w("-" * 72)
    w(_tbl(T["levels"]))
    w("  own = the ladder's feature, change from the original in corpus "
      "SD; cross =\n  the other feature; clip = share of pixels newly "
      f"pushed > {CLIP_DEPTH:g} levels past 0 or 255;\n  halo, noise: see "
      "the test script; dist_orig / dist_corpus: distance from the\n  "
      "scene's original / the corpus median, sqrt(z_sat^2 + z_con^2).")
    w("")
    w("4. HOW EXACTLY THE LEVELS ARE HIT")
    w("-" * 72)
    dv = T["devs"]
    for feat in ("contrast", "colour"):
        x = dv[dv.feature == feat]
        w(f"  {feat:<9} levels: largest miss {x.abs_dev_sd.max():.4f} SD, "
          f"median {x.abs_dev_sd.median():.4f} SD, "
          f"{int((x.abs_dev_sd > LEVEL_CHECK_TOL_SD).sum())} of {len(x)} "
          f"beyond {LEVEL_CHECK_TOL_SD:g} SD")
    off = man[(man.contrast_step != 0) & (man.colour_step != 0)]
    w(f"  off the ladders ({len(off):,} cells with both features edited):"
      f" contrast within\n  {off.contrast_dev_sd.abs().max():.3f} SD of its"
      f" level (median {off.contrast_dev_sd.abs().median():.3f}), colour "
      f"within {off.colour_dev_sd.abs().max():.3f} SD\n  (median "
      f"{off.colour_dev_sd.abs().median():.3f}). Per cell: "
      "cell_deviations.csv.")
    w("")
    w("5. FILES")
    w("-" * 72)
    w(f"  {len(man):,} PNGs in {STIMULI_SUBDIR}/ "
      f"({T['stim_bytes'] / 1e9:.2f} GB); every file is listed in "
      "checksums.sha256.")
    for k, v in RUN["timings"].items():
        w(f"  {k:<45} {v:.1f} min")
    w("")
    w("6. PLAIN-LANGUAGE SUMMARY")
    w("-" * 72)
    nf = [c for c in CHECKS if c["status"] not in ("PASS", "NOTE")]
    s = []
    s.append(
        f"This folder is the v5 stimulus set: {man.scene_id.nunique()} "
        f"scenes, each in {len(CONTRAST_IDX) ** 2} versions, "
        f"{len(man):,} images of {IMAGE_SIZE} x {IMAGE_SIZE} px. Every "
        "scene's unedited photograph is the middle cell, with three "
        "contrast levels below and three above it (about 0.33, 0.67 and "
        "1 corpus SD) crossed with three colour levels below and three "
        "above.")
    t1 = [c for c in CHECKS if c["code"] == "T1"]
    if t1 and t1[0]["status"] == "PASS":
        s.append(f"Every image is byte-identical to the image of the same "
                 f"cell in test run {TEST_RUN_ID}, so the visual check and "
                 "the CLIP results of that run apply to this set "
                 "unchanged.")
    elif t1:
        s.append("The byte comparison with the test run did not pass "
                 "(check T1): read section 1.")
    xc = dv[dv.feature == "contrast"].abs_dev_sd
    s.append(
        "On the two ladders through the original, every colour level is "
        f"within {dv[dv.feature == 'colour'].abs_dev_sd.max():.3f} SD of "
        "its target and every contrast level within "
        f"{xc.max():.3f} SD ({int((xc > LEVEL_CHECK_TOL_SD).sum())} of "
        f"{len(xc)} contrast levels miss by more than "
        f"{LEVEL_CHECK_TOL_SD:g} SD). Where both features are edited, the "
        "colour edit's effect depends on the contrast edit before it, so "
        "those cells sit up to "
        f"{off.colour_dev_sd.abs().max():.2f} SD from the nominal colour "
        "level. The trial-design scripts use each image's measured values.")
    s.append(
        "The upward contrast levels clip: in the typical scene "
        + ", ".join(f"{r.level} {r.clip_median:.0%}" for r in
                    T["levels"][(T["levels"].ladder == "contrast")
                                & T["levels"].level.str.contains(r"\+")]
                    .itertuples())
        + " of pixels (worst scene "
        f"{T['levels'][T['levels'].ladder == 'contrast'].clip_max.max():.0%}"
        "). This was accepted: it is reported, never used to cap a level "
        "or drop an image.")
    s.append("Checks not passed: " + (", ".join(
        f"{c['code']} ({c['status']})" for c in nf) if nf else "none")
        + ". The source photographs are not included; sources.csv gives "
        "each one's Places365 file ID and SHA-256, so the set can be "
        "rebuilt from the dataset with the copy of the script in "
        "provenance/.")
    for p in s:
        w(_wrap(p))
        w("")
    with open(B("report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


README_TEMPLATE = """# {name}

Stimulus set for the v5 image-choice experiment of the drl2drv project:
{n_scenes} scenes x {n_cells} versions (one unedited) = {n_images:,} images,
{size} x {size} px PNG.

Built by `build_stimulus_set_v5.py` (a copy is in `provenance/`), settings
hash `{hash}`, on {date}. Every path in this folder is relative to it.

## The design

Each scene is shown in a 7 x 7 grid of versions:

* **Contrast** (`contrast_level` c-3 ... c0 ... c+3, `contrast_step` -3 ... 3).
  Below the original: v4's brightness-equated haze (`haze_eq`). Above it:
  the same blend run backwards on RGB (`dehaze_rgb`, v4's `m_contrast_up`).
  At every contrast level a chroma factor (`comp_k`) restores the scene's
  mean saturation, so contrast levels leave `sat_mean` unchanged.
* **Colour** (`colour_level` s-3 ... s0 ... s+3, `colour_step` -3 ... 3):
  v4's CIELAB chroma edit, scaling a*, b* below the original and blending
  toward each pixel's in-gamut ceiling above it (`colour_k`).
* **Levels**: equal steps in corpus SD, solved separately for every scene,
  at -1, -2/3, -1/3, 0, +1/3, +2/3, +1 SD of `contrast_local` (SD
  {sd_con:.3f}) and of `sat_mean` (SD {sd_sat:.3f}), measured on the
  stimulus path (photograph -> 512 px -> 256 px). The colour edit is
  applied after the contrast edit.
* The cell c0/s0 (`is_original`) is the unedited photograph (its central
  512 px square).

Why this design: {why}

## Precision

On the two ladders through the original, colour levels are within
{dev_col:.3f} SD of their targets and contrast levels within {dev_con:.3f} SD
({n_off} of {n_lev} contrast levels miss by more than {tol:g} SD). Where
both features are edited, the colour edit's effect depends on the
contrast edit before it: those cells sit up to {dev_grid:.2f} SD from the
nominal colour level (`colour_dev_sd`). Use each image's measured values
(`d_con_sd`, `d_sat_sd`, the feature columns), not the nominal levels.

The upward contrast levels clip (pixels pushed past 0 or 255): median
over scenes {clip_txt}. This is reported (`clip_*` columns), not used to
cap levels or drop images.

## Files

| file | content |
|---|---|
| `stimuli/` | the {n_images:,} images, `<scene_id>__c<k>__s<k>.png` (k = n3 ... 0 ... p3; p = plus, n = minus) |
| `stimulus_manifest.csv` | one row per image: scene, levels, file name, edit parameters, targets, every measurement |
| `cell_plan.csv` | the 49 cells of every scene |
| `sources.csv` | the {n_scenes} source photographs: Places365 file ID, SHA-256, size |
| `level_deviations.csv` | each scene's 12 levels: target, achieved, parameter |
| `cell_deviations.csv` | each cell's distance from its nominal levels, summarised over scenes |
| `level_summary.csv` | per level: measured change, cross-talk, clipping, artefacts, distance from typical |
| `reach.csv`, `reach_grid.csv` | the per-scene parameter search the levels were solved from |
| `reference_stimulus_path.csv` | the 20,000-image corpus reference (mean, SD, percentiles) |
| `envelope_violations.csv` | images with a feature outside the corpus 1-99% range (reported) |
| `identity.csv`, `unedited.csv` | the no-change checks |
| `report.txt`, `checks.csv` | the build report and every check |
| `settings.json`, `run_meta.json` | settings, versions, timings |
| `provenance/build_stimulus_set_v5.py` | the script that built this folder |
| `CITATION.cff` | how to cite (placeholders to complete) |
| `checksums.sha256`, `FROZEN.json` | integrity: `sha256sum -c checksums.sha256` |

Main manifest columns: `scene_id`, `category`, `group`, `indoor`,
`contrast_level`, `colour_level`, `contrast_step`, `colour_step`,
`is_original`, `use` (all `pair`), `filename`, `places365_file`;
`contrast_param` (haze h, negative = dehaze), `comp_k`, `colour_k`;
`contrast_target_sd`, `colour_target_sd`, `d_con_sd`, `d_sat_sd`,
`contrast_dev_sd`, `colour_dev_sd`; `z_sat`, `z_con`; `dist_corpus`,
`dist_orig` (combined) and `_sat`, `_con` (per feature); the 18 corpus
features and `fine_texture`, `noise_sigma`, `edge_overshoot`; clipping
(`clip_0.5lv` ... `clip_8lv`), `halo_share`, `noise_ratio`,
`band_new_empty`; `envelope_n`, `envelope_features`; `png_sha256`.

## Sources and licence

The images are derived from photographs of the Places365 dataset
(Zhou, Lapedriza, Khosla, Oliva & Torralba, 2018, "Places: A 10 million
image database for scene recognition", IEEE TPAMI 40(6), 1452-1464).
The Places365 documentation states that the copyright of all images
belongs to the image owners.

**Licence of this set: to be confirmed.** The source photographs are
not included. `sources.csv` lists each one's Places365 file ID (relative
to the dataset's image folder) and SHA-256, so the set can be rebuilt
exactly: obtain Places365, check the hashes, and run the script in
`provenance/` (it also needs the corpus files listed in its settings).

## Citation

See `CITATION.cff` (authors, version and DOI to be completed).
"""


CITATION_TEMPLATE = """cff-version: 1.2.0
message: "If you use this stimulus set, please cite it as below."
type: dataset
title: "{name}"
version: "{hash}"
date-released: "{date}"
authors:
  - family-names: "TO BE COMPLETED"
    given-names: "TO BE COMPLETED"
# doi: "10.5281/zenodo.TO-BE-ASSIGNED"
# license: TO BE CONFIRMED (derived from Places365 photographs; see README.md)
abstract: >-
  {n_images} images ({n_scenes} scenes x 49 versions, 512 x 512 px) in which
  contrast and colour are each moved in three equal corpus-SD steps below
  and above the unedited photograph, for the v5 image-choice experiment.
keywords:
  - stimulus set
  - image contrast
  - colour saturation
references:
  - type: article
    title: "Places: A 10 million image database for scene recognition"
    authors:
      - family-names: Zhou
        given-names: Bolei
      - family-names: Lapedriza
        given-names: Agata
      - family-names: Khosla
        given-names: Aditya
      - family-names: Oliva
        given-names: Aude
      - family-names: Torralba
        given-names: Antonio
    journal: "IEEE Transactions on Pattern Analysis and Machine Intelligence"
    volume: 40
    issue: 6
    start: 1452
    end: 1464
    year: 2018
"""


def write_readme_and_citation(T):
    man = T["manifest"]
    dv = T["devs"]
    off = man[(man.contrast_step != 0) & (man.colour_step != 0)]
    lv = T["levels"]
    up = lv[(lv.ladder == "contrast") & lv.level.str.contains(r"\+")]
    xc = dv[dv.feature == "contrast"].abs_dev_sd
    kw = dict(name=SET_NAME, n_scenes=man.scene_id.nunique(),
              n_cells=len(CONTRAST_IDX) ** 2, n_images=len(man),
              size=IMAGE_SIZE, hash=T["hash"],
              date=time.strftime("%Y-%m-%d"),
              sd_con=float(T["st"].at[FEAT_CON, "sd"]),
              sd_sat=float(T["st"].at[FEAT_SAT, "sd"]),
              why=DECISION_NOTE,
              dev_col=float(dv[dv.feature == "colour"].abs_dev_sd.max()),
              dev_con=float(xc.max()),
              n_off=int((xc > LEVEL_CHECK_TOL_SD).sum()), n_lev=len(xc),
              tol=LEVEL_CHECK_TOL_SD,
              dev_grid=float(off.colour_dev_sd.abs().max()),
              clip_txt=", ".join(f"{r.level} {r.clip_median:.0%}"
                                 for r in up.itertuples()))
    with open(B("README.md"), "w") as f:
        f.write(README_TEMPLATE.format(**kw))
    with open(B("CITATION.cff"), "w") as f:
        f.write(CITATION_TEMPLATE.format(**kw))


FROZEN_UNHASHED = ["checksums.sha256", "FROZEN.json"]


def freeze(T):
    """checksums.sha256 over every file, verified by reading it back, and
    FROZEN.json. Every path written is relative to the run folder."""
    root = RUN["base"]
    files = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            rel = os.path.relpath(os.path.join(dp, f), root).replace(
                os.sep, "/")
            if rel not in FROZEN_UNHASHED:
                files.append(rel)
    files.sort()
    digests = {rel: sha256_file(os.path.join(root, rel)) for rel in files}
    with open(B("checksums.sha256"), "w") as f:
        for rel in files:
            f.write(f"{digests[rel]}  {rel}\n")
    # read back and verify
    bad = []
    with open(B("checksums.sha256")) as f:
        for ln in f:
            h, rel = ln.rstrip("\n").split("  ", 1)
            if sha256_file(os.path.join(root, rel)) != h:
                bad.append(rel)
    stim = sorted(r for r in files if r.startswith(STIMULI_SUBDIR + "/"))
    stim_digest = hashlib.sha256("".join(
        f"{digests[r]}  {r}\n" for r in stim).encode()).hexdigest()
    fails = [c for c in CHECKS if c["status"] in ("FAIL", "WARN")]
    frozen = {
        "frozen": time.strftime("%Y-%m-%d %H:%M:%S"),
        "set": SET_NAME,
        "settings_hash": T["hash"],
        "method": {"approach": APPROACH, "definition": "b (equal SD steps, "
                   "per scene)", "span_sd": B_SPAN_SD,
                   "level_fractions": LEVEL_FRACTIONS},
        "n_scenes": int(T["manifest"].scene_id.nunique()),
        "n_images": int(len(T["manifest"])),
        "files_hashed": len(files),
        "checksums_sha256": sha256_file(B("checksums.sha256")),
        "stimuli_sha256": stim_digest,
        "stimuli_sha256_definition": (
            "SHA-256 of the checksums.sha256 lines of the files in "
            f"{STIMULI_SUBDIR}/, sorted by path: one number for the images"),
        "manifest_sha256": digests["stimulus_manifest.csv"],
        "script": {"name": "build_stimulus_set_v5.py",
                   "copy": "provenance/build_stimulus_set_v5.py",
                   "sha256": digests[
                       "provenance/build_stimulus_set_v5.py"]},
        "test_run": {"id": TEST_RUN_ID,
                     "byte_identical_images": T["n_byte_identical"],
                     "compared": T["n_compared"]},
        "sources": {"dataset": "Places365", "file": "sources.csv",
                    "included": False, "licence": "to be confirmed"},
        "checks": {"pass": sum(c["status"] == "PASS" for c in CHECKS),
                   "note": sum(c["status"] == "NOTE" for c in CHECKS),
                   "skip": sum(c["status"] == "SKIP" for c in CHECKS),
                   "warn": sum(c["status"] == "WARN" for c in CHECKS),
                   "fail": sum(c["status"] == "FAIL" for c in CHECKS),
                   "not_passed": [f"{c['code']} {c['status']}"
                                  for c in fails]},
        "checksums_verified": not bad,
        "paths": "every path is relative to this folder",
        "unhashed": FROZEN_UNHASHED,
        "use": ("copy this folder whole; verify with `sha256sum -c "
                "checksums.sha256` before use"),
    }
    with open(B("FROZEN.json"), "w") as f:
        json.dump(frozen, f, indent=2, default=_json_default)
    return bad, len(files)


# ============================================================
# Main
# ============================================================

def main():
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    cv2.setNumThreads(1)
    print("=" * 70 + "\nbuild_stimulus_set_v5.py\n" + "=" * 70)
    h = settings_hash()
    out = resolve_run_dir(h)
    os.makedirs(os.path.join(out, STIMULI_SUBDIR), exist_ok=True)
    RUN["base"] = RUN["out"] = out
    RUN["meta"]["settings_hash"] = h
    RUN["meta"]["run_folder"] = out
    with open(B("settings.json"), "w") as f:
        json.dump({"settings_hash": h, "hashed_settings_exclude":
                   sorted(_NOT_HASHED), "settings": settings_dict()}, f,
                  indent=2, default=_json_default)
    print(f"Output: {out}")
    write_meta("running")

    with Timer("Start-up checks and the reference"):
        check_test_settings()
        idx, F, eligible = load_corpus()
        rng = np.random.default_rng(RANDOM_SEED)
        check_feature_code(idx, F, rng)
        st = measure_reference(idx, eligible)
        st.reset_index().to_csv(B("reference_stimulus_path.csv"),
                                index=False, float_format="%.8g")
        check_reference(st)
        scenes = load_scenes(idx)
        del F
    sd = {FEAT_SAT: float(st.at[FEAT_SAT, "sd"]),
          FEAT_CON: float(st.at[FEAT_CON, "sd"])}

    with Timer("Reach of each edit on every scene"):
        tasks = [(r.scene_id, r.source_path) for r in scenes.itertuples()]
        res = run_pool(_reach_task, tasks, {"dims": dims_needed(), "sd": sd},
                       "reach")
        tables = {r["scene_id"]: r["tables"] for r in res}
        pd.DataFrame([x for r in res for x in r["reach"]]).to_csv(
            B("reach.csv"), index=False, float_format="%.8g")
        pd.DataFrame([x for r in res for x in r["grid"]]).to_csv(
            B("reach_grid.csv"), index=False, float_format="%.8g")
        del res

    with Timer("Levels, images and measurements"):
        payload = {"sd": sd, "base_dir": out,
                   "span": {(APPROACH, "contrast"): B_SPAN_SD,
                            (APPROACH, "colour"): B_SPAN_SD}}
        tasks = [(r.scene_id, r.source_path, tables[r.scene_id],
                  r.v4_unedited_file) for r in scenes.itertuples()]
        res = run_pool(_build_task, tasks, payload, "scenes")
        cells = pd.DataFrame([x for r in res for x in r["cells"]])
        lad = pd.DataFrame([x for r in res for x in r["ladders"]])
        bases = pd.DataFrame([r["base"] for r in res])
        ident = pd.DataFrame([x for r in res for x in r["identity"]])
        uned = pd.DataFrame([{"scene_id": r["scene_id"], **r["unedited"]}
                             for r in res])
        del res

    with Timer("Tables and checks"):
        cells = add_derived(cells, bases, st)
        cells["contrast_dev_sd"] = cells.d_con_sd - cells.contrast_target_sd
        cells["colour_dev_sd"] = cells.d_sat_sd - cells.colour_target_sd
        # achieved change of every level, read off the ladders' own cells
        con = cells[cells.sx == 0].set_index(["scene_id", "cx"]).d_con_sd
        col = cells[cells.cx == 0].set_index(["scene_id", "sx"]).d_sat_sd
        lad["achieved_sd"] = [
            (con if f == "contrast" else col).get((sid, i), np.nan)
            for sid, f, i in zip(lad.scene_id, lad.feature, lad["index"])]
        lad["abs_dev_sd"] = (lad.achieved_sd - lad.target_sd).abs()
        lad.insert(2, "level", [lab("c" if f == "contrast" else "s", i)
                                for f, i in zip(lad.feature, lad["index"])])
        man = build_manifest(cells, scenes)
        man.to_csv(B("stimulus_manifest.csv"), index=False,
                   float_format="%.8g")
        cell_plan().to_csv(B("cell_plan.csv"), index=False)
        scenes[["scene_id", "category", "group", "indoor", "places365_file",
                "image_key", "source_width", "source_height",
                "source_bytes", "source_sha256"]].to_csv(
            B("sources.csv"), index=False)
        lad.to_csv(B("level_deviations.csv"), index=False,
                   float_format="%.8g")
        cd = (man.groupby(["contrast_level", "colour_level", "contrast_step",
                           "colour_step"])
              .agg(contrast_dev_median=("contrast_dev_sd", "median"),
                   contrast_dev_absmax=("contrast_dev_sd",
                                        lambda s: s.abs().max()),
                   colour_dev_median=("colour_dev_sd", "median"),
                   colour_dev_absmax=("colour_dev_sd",
                                      lambda s: s.abs().max()))
              .reset_index().sort_values(["contrast_step", "colour_step"]))
        cd.to_csv(B("cell_deviations.csv"), index=False, float_format="%.6g")
        ident.to_csv(B("identity.csv"), index=False, float_format="%.6g")
        uned.to_csv(B("unedited.csv"), index=False)
        ev = envelope_violations(cells, st)
        ev.to_csv(B("envelope_violations.csv"), index=False,
                  float_format="%.6g")
        note("V1", f"envelope (as v4, reported): {ev.filename.nunique()} "
             f"images in {ev.scene_id.nunique()} scenes have a feature "
             "outside the corpus 1-99% range"
             + (" (" + ", ".join(f"{k} {v}" for k, v in
                                 ev.feature.value_counts().items()) + ")"
                if len(ev) else ""))
        checks_after_render(cells, lad, ident, uned)

    with Timer("Report, README and freeze"):
        os.makedirs(B("provenance"), exist_ok=True)
        shutil.copyfile(os.path.abspath(__file__),
                        B("provenance/build_stimulus_set_v5.py"))
        ti = cells.test_run_byte_identical
        T = {"manifest": man, "st": st, "hash": h,
             "out_rel": out.replace(os.sep, "/"),
             "levels": level_summary(man),
             "devs": lad[lad["index"] != 0],
             "stim_bytes": sum(os.path.getsize(os.path.join(
                 out, STIMULI_SUBDIR, f)) for f in man.filename),
             "n_byte_identical": int((ti == True).sum()),      # noqa: E712
             "n_compared": int(ti.notna().sum())}
        T["levels"].to_csv(B("level_summary.csv"), index=False,
                           float_format="%.6g")
        write_readme_and_citation(T)
        write_report(T)
        write_checks()
        write_meta("finished")
        hits = relative_path_scan(out)
        check("A0", not hits,
              "no absolute path of this machine in any text file of the "
              "folder" + (f"; found in {hits[:4]}" if hits else ""))
        write_checks()
        write_meta("finished")
        bad, nfiles = freeze(T)
        hits = relative_path_scan(out)
        if bad or hits:
            stop(f"freeze failed: {len(bad)} checksum mismatches, absolute "
                 f"paths in {hits[:3]}")
        print(f"  {nfiles:,} files hashed; checksums verified")

    print(f"\nDone in {(time.time() - RUN['t0']) / 60:.1f} min.  Report: "
          f"{B('report.txt')}")


if __name__ == "__main__":
    main()
