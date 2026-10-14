"""
select_scenes_v4.py  (stimulus pipeline for the v4 axes, step 2)

Chooses which photographs become the experiment's scenes, and produces a
shortlist for you to screen by hand.

Two kinds of screening, and only one of them can be automated
-------------------------------------------------------------
This script screens on MEASUREMENT: is the scene neutral to start with,
can the manipulations actually move it, does it afford the trial types
the experiment needs. It cannot tell an ordinary bedroom from an
architectural showpiece, cannot see a stray face or a sign in a language
the participant will read, and cannot judge whether a scene stays
legible at the size it will be shown. Those are what the hand-screening
step is for, and this script's real output is the worksheet for it.

Where this sits
---------------
  step 1 (calibrate_levels)   the levels of the grid          [done]
  step 2 (this one)           which scenes -> you screen by hand
  step 3 (build_stimulus_set) the screened set -> the stimuli

What it does
------------
  1. Filters the corpus to a candidate pool, from stored tables only --
     no images are opened, so this part is fast over 1.1M rows:
       * an everyday category (keep = 1 in CATEGORY_CSV). Derealization
         is about FAMILIAR surroundings becoming strange; a volcano has
         no familiar baseline to depart from.
       * colour, not a duplicate, not toned, short side >= MIN_SIDE.
       * NEUTRAL on both target axes -- a scene that already scores
         high on fog carries that lean into every cell it lands in.
       * MID-RANGE on both manipulated features. An already-grey scene
         cannot be made greyer; an already-flat one cannot be made
         much flatter.
       * no people. Step 02's filter admits up to 10% person pixels,
         which is fine for a corpus and too loose for a stimulus: a
         face is where the eye goes first, and gives the participant
         something to respond to other than the scene.
       * capped per category, so the result is not about bedrooms.
  2. Measures REACHABILITY per candidate, by building the actual grid
     and measuring every cell. Three questions:
       * does the colour ladder move THIS scene far enough?
       * does the fog ladder?
       * does the scene afford a SATURATION-MATCHED fog pair -- two
         cells with near-identical saturation but very different
         contrast? Fog drains colour as a side effect, so an unmatched
         foggy-vs-clear comparison confounds the two. Only matched
         pairs make the fog contrast interpretable, and a scene that
         affords none cannot contribute a fog trial.
  3. Writes the shortlist, the screening worksheet and contact sheets.

Reachability, and why it is a fraction rather than a threshold
---------------------------------------------------------------
Step 1 established that the change a level produces is PROPORTIONAL to
the scene's own starting value. The previous pipeline used absolute
thresholds (MIN_SAT_RANGE = 25.0, MIN_FOG_RANGE = 6.6), which are
thresholds in the feature's units and have to be re-derived whenever the
grid changes. Here a scene passes when its measured change reaches
REACH_FRACTION of the level's target -- a rule in the same units as the
calibration, which needs no re-derivation.

CPU only
--------
The v4 pipeline already scored every corpus image on every axis
(step 06), so the untouched scenes' axis positions are a lookup. The
axis profile ACROSS the grid still needs the encoder, but step 3 encodes
every stimulus anyway, so that check lives there instead. This script
opens no model and needs no GPU.

Both fog families at once
-------------------------
Plain haze and haze_eq are calibrated on the same feature, so at a given
rung they remove the same amount of fine detail -- which means they give
the same reachability verdict and the same scene list. CHECK 4 verifies
that on a subsample rather than assuming it. One scene list therefore
serves both stimulus sets.

Inputs
------
  INDEX_CSV       data/selection/image_index.csv          (step 02)
  FEATURES_NPY    data/features/features_crop.npy         (step 04)
  FEATURE_DEFS    data/features/feature_definitions.json  (step 04)
  FLAGS_CSV       data/monochrome/image_flags.csv         (step 04a)
  DUP_CSV         data/checks_v4/duplicate_groups.csv     (09a)
  TONED_CSV       data/checks_v4/toned_images.csv         (09a)
  CATEGORY_CSV    everyday_categories.csv
  SCORES_DIR      data/scores_v4/                         (step 06)
  LEVELS_CSV      data/levels_v4/levels.csv               (step 1)
  REF_STATS_CSV   data/levels_v4/corpus_feature_stats_v4.csv (step 1)

Outputs (under OUTPUT_DIR)
--------------------------
  candidates.csv        the pool surviving the corpus filters
  band_counts.csv       how many candidates survive at several band
                        settings, so the bands can be chosen from data
  reachability.csv      per candidate: measured ranges, matched-pair
                        gap, pass/fail and why
  grid_measurements.csv every cell of every candidate, measured
  shortlist.csv         THE SHORTLIST, balanced across categories
  screening_list.csv    the worksheet: fill in `keep` and `reason`
  crops/<scene_id>.jpg  each shortlisted scene as it will be cropped
  contact_<n>.png       contact sheets of those crops, for screening
  grid_sheets/<id>.png  the full grid for a subset, to spot-check
  selection.png         where the candidates sit in the corpus
  reachability.png      the reachability distributions
  select_report.txt     summary and the checks

Usage
-----
    conda activate drl2drv
    python select_scenes_v4.py
    # then open the contact sheets, fill in `keep` in
    # screening_list.csv, and run step 3.

CPU only. The expensive part is N_CANDIDATES x 26 renders and
measurements; a few minutes with N_WORKERS set to the number of cores.
"""

import itertools
import json
import os
import time
import warnings
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

# --- Inputs ---
INDEX_CSV       = "data/selection/image_index.csv"
FEATURES_NPY    = "data/features/features_crop.npy"
FEATURE_DEFS    = "data/features/feature_definitions.json"
FLAGS_CSV       = "data/monochrome/image_flags.csv"
DUP_CSV         = "data/checks_v4/duplicate_groups.csv"
TONED_CSV       = "data/checks_v4/toned_images.csv"
CATEGORY_CSV    = "everyday_categories.csv"
SCORES_DIR      = "data/scores_v4"
LEVELS_CSV      = "data/levels_v4_ec/levels.csv"
REF_STATS_CSV   = "data/levels_v4_ec/corpus_feature_stats_v4.csv"

# BAND_POPULATION
#   Which block of corpus_feature_stats_v4.csv the FEATURE BANDS are
#   read from.
#
#   This must be the CORPUS measurement path, not the stimulus path,
#   because the band is applied to each image's stored feature value --
#   and those were measured by step 04 as original -> 256. Step 1's
#   targets come from the stimulus path (original -> 512 -> 256) because
#   they are compared against measured stimuli. Two different questions,
#   two different populations; mixing them is exactly the mistake that
#   put step 1's fog ladder 13% out.
#
#     band on a stored feature      -> everyday_colour  (corpus path)
#     target vs a measured stimulus -> stimulus_path    (step 1)
BAND_POPULATION = "everyday_colour"

# --- Output ---
OUTPUT_DIR      = "data/select_scenes_v4_ec"

# --- The axes ---
#
# Matched exactly against score_catalog.csv; the run stops if either
# matches zero rows or more than one.
AXIS_FOG        = "v4::A07 fog, blurry, hazy"
AXIS_COLOUR     = "v4::A13 dull, less vivid, colorless"

# --- The ladder (names of levels in LEVELS_CSV) ---
#
# The same rungs for both fog families, so the two stimulus sets differ
# only in the manipulation. Step 1's levels.csv holds all ten rungs for
# each, so these can be changed without re-running it.
FOG_LEVELS      = ["p35", "p25", "p15", "p10", "p05"]
COLOUR_LEVELS   = ["p25", "p10", "p05", "p01"]      # k = 1 is the first
                                                    # column, added here
# FOG_FAMILY
#   Which fog this script measures with. Both give the same verdict
#   (CHECK 4 confirms it), so this only decides what the grid sheets
#   show.
FOG_FAMILY      = "haze"            # "haze" or "haze_eq"
FOG_DIMENSION   = "fog"             # "fog" pairs with "haze",
                                    # "fog_eq" with "haze_eq"

# --- Corpus filters ---
#
# AXIS_PCTL_LOW / HIGH
#   Band, in percentiles of the corpus axis distribution, that a scene's
#   UNTOUCHED score must fall inside on BOTH axes. 35-65 keeps scenes
#   near the median on both fog and colour. The report prints candidate
#   counts at several bands, so this can be set from the data rather
#   than guessed.
AXIS_PCTL_LOW   = 35.0
AXIS_PCTL_HIGH  = 65.0

# SAT_PCTL / FOG_PCTL
#   Bands on the two manipulated features, so the grid has somewhere to
#   move them. Separate bands because the two manipulations behave
#   differently at the extremes.
SAT_PCTL_LOW,  SAT_PCTL_HIGH  = 30.0, 75.0
FOG_PCTL_LOW,  FOG_PCTL_HIGH  = 30.0, 75.0

# BAND_SWEEP
#   Extra band settings to count candidates at, for the report only.
#   Nothing is selected from these.
BAND_SWEEP      = [(25.0, 75.0), (30.0, 70.0), (35.0, 65.0),
                   (40.0, 60.0), (45.0, 55.0)]

# --- Eligibility ---
MIN_SIDE        = 512
EXCLUDE_GREYSCALE  = True
EXCLUDE_TONED      = True
EXCLUDE_DUPLICATES = True

# EXCLUDE_PEOPLE / MAX_PERSON_FRAC
#   image_index.csv carries frac_person from step 02. Step 02 admitted
#   up to 0.10; stimuli should have none.
EXCLUDE_PEOPLE  = True
MAX_PERSON_FRAC = 0.0

# MAX_PER_CATEGORY
#   Candidates any one category may contribute. Large enough that each
#   category offers several to choose between during hand screening,
#   small enough that no category dominates.
MAX_PER_CATEGORY = 500 # 25

# N_CANDIDATES
#   Pool entering the reachability measurement. Several times N_FINAL,
#   so hand screening can reject freely.
N_CANDIDATES    = 2000 # 600

# N_FINAL
#   Size of the shortlist written to the worksheet. NOT the stimulus
#   set: step 3 selects ~30 from whatever you keep.
N_FINAL         = 1800 # 400

# --- Reachability ---
#
# REACH_FRACTION
#   A scene passes when its own measured change at the DEEPEST level of
#   a dimension reaches this fraction of that level's target change.
#   Replaces the previous pipeline's absolute MIN_SAT_RANGE and
#   MIN_FOG_RANGE with a rule in the units the calibration used.
REACH_FRACTION  = 0.6

# PAIR_SAT_TOL / MIN_FOG_GAP
#   A saturation-matched fog pair: two cells whose sat_mean differs by
#   less than PAIR_SAT_TOL and whose contrast_local differs by at least
#   MIN_FOG_GAP. Values from analyse_pairs_loc_con.py, which step 1's
#   grid check confirmed still produce ample pairs on the new scale.
PAIR_SAT_TOL    = 3.0
MIN_FOG_GAP     = 4.2

# --- Geometry (must match step 04, step 1 and step 3) ---
IMAGE_SIZE      = 512
FEATURE_SIZE    = 256
CLIP_PX         = 336

# --- Manipulation parameters (as 09c / step 1) ---
MANIP_ORDER     = ("haze", "chroma")
IDENTITY_SHORT_CIRCUIT = True
HAZE_AIRLIGHT_TOP = 1.0
GAMUT_ITERS     = 16
GAMUT_CMAX_SEARCH = 160.0
GAMUT_MARGIN    = 0.995

# --- Screening output ---
#
# CROP_QUALITY
#   The crops are for looking at, not for measuring, so JPEG is fine.
CROP_SIZE       = 512
CROP_QUALITY    = 92

# CONTACT_PER_SHEET / CONTACT_PX
#   Contact sheets for the first screening pass. These show the PLAIN
#   crop, because content judgements -- framing, legibility, people,
#   text -- are made on the unmanipulated scene.
CONTACT_PER_SHEET = 30
CONTACT_PX      = 220

# N_GRID_SHEETS
#   Shortlisted scenes also rendered as the full colour x fog grid, to
#   spot-check that the manipulation behaves. None = all of them.
N_GRID_SHEETS   = 40

# --- Checks ---
FEATURE_CHECK_N    = 8
FEATURE_CHECK_RTOL = 1e-4
FOG_EQUIV_N        = 24     # scenes used by CHECK 4
FOG_EQUIV_TOL      = 0.02   # relative

# --- Parallelism ---
N_WORKERS       = 20
CHUNK           = 4
RANDOM_SEED     = 123


# ============================================================
# Feature code  (verbatim from 04_compute_features.py / step 1)
# ============================================================

LEGACY = ["lum_mean", "lum_std", "lum_p05", "lum_p50", "lum_p95",
          "sat_mean", "sat_std", "val_mean", "colorfulness",
          "contrast_local", "edge_density"]
EXTRA = ["lum_skew", "lum_entropy", "chroma_mean", "lab_a_mean",
         "lab_b_mean", "spectral_slope", "dark_channel_mean"]
CROP = LEGACY + EXTRA

FEAT_COLOUR = "sat_mean"
FEAT_FOG    = "contrast_local"

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

    The fast path for legacy-only requests returns bit-identical values
    while skipping the FFT, the histogram, the CIELAB conversion and the
    dark channel. Both features this script uses are in that set.
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
# Manipulations  (verbatim from 09c / step 1)
# ============================================================

_XN, _ZN = 0.950456, 1.088754
_M_XYZ = np.array([[0.412453, 0.357580, 0.180423],
                   [0.212671, 0.715160, 0.072169],
                   [0.019334, 0.119193, 0.950227]])
_M_RGB = np.linalg.inv(_M_XYZ).astype(np.float32)


def to_u8(x):
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
# Loading
# ============================================================

def scene_id_of(category, path):
    return (category.replace("/", "_") + "_"
            + os.path.splitext(os.path.basename(path))[0])


def load_levels(report):
    """The ladder: (label, parameter) for the colour and fog levels
    named in the settings, plus each level's target change."""
    if not os.path.isfile(LEVELS_CSV):
        raise FileNotFoundError(
            f"{LEVELS_CSV} not found. Run step 1 first.")
    L = pd.read_csv(LEVELS_CSV)

    def pick(dim, names):
        out = []
        for nm in names:
            m = L[(L.dimension == dim) & (L.level == nm)]
            if not len(m):
                raise ValueError(f"{LEVELS_CSV} has no {dim} level {nm!r}")
            r = m.iloc[0]
            if not np.isfinite(r.param):
                raise ValueError(f"{dim} level {nm} has no parameter.")
            if r.status != "ok":
                warnings.warn(f"{dim} level {nm} is {r.status!r}.")
            out.append((nm, float(r.param), float(r.target_change)))
        return out

    colour = [("k=1", 1.0, 0.0)] + pick("colour", COLOUR_LEVELS)
    fog = pick(FOG_DIMENSION, FOG_LEVELS)
    print(f"  ladder: {len(colour)} colour x {len(fog)} fog "
          f"({FOG_FAMILY})")
    print("    colour " + ", ".join(f"{n}={p:.3f}" for n, p, _ in colour))
    print("    fog    " + ", ".join(f"{n}={p:.3f}" for n, p, _ in fog))
    report.append(("ladder", f"{len(colour)} x {len(fog)}, {FOG_FAMILY}"))
    return colour, fog


def load_scores(n, report):
    """The two axis columns of scores_z.npy, read in chunks."""
    cat_path = os.path.join(SCORES_DIR, "score_catalog.csv")
    z_path = os.path.join(SCORES_DIR, "scores_z.npy")
    for p in (cat_path, z_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p} not found (step 06).")
    cat = pd.read_csv(cat_path)

    cols = {}
    for label, key in (("fog", AXIS_FOG), ("colour", AXIS_COLOUR)):
        hit = cat[cat.key == key]
        if len(hit) != 1:
            raise ValueError(
                f"Axis key {key!r} matched {len(hit)} rows of "
                f"{cat_path}; expected exactly 1.\n"
                "Check the key against score_catalog.csv.")
        cols[label] = int(hit.col.iloc[0])
    print(f"  axes: fog -> column {cols['fog']}, "
          f"colour -> column {cols['colour']}")

    Z = np.load(z_path, mmap_mode="r")
    if Z.shape[0] != n:
        raise ValueError(f"scores_z.npy has {Z.shape[0]} rows, the index "
                         f"has {n}.")
    out = {}
    for label, j in cols.items():
        v = np.empty(n, np.float32)
        for a in range(0, n, 200_000):
            b = min(a + 200_000, n)
            v[a:b] = np.asarray(Z[a:b, j], np.float32)
        out[label] = v
    report.append(("axis columns",
                   f"fog {cols['fog']}, colour {cols['colour']}"))
    return out


def load_reference(report):
    """The percentile table step 1 calibrated against."""
    if not os.path.isfile(REF_STATS_CSV):
        raise FileNotFoundError(f"{REF_STATS_CSV} not found (step 1).")
    S = pd.read_csv(REF_STATS_CSV)
    if "population" in S.columns:
        S = S[S.population == BAND_POPULATION]
        if not len(S):
            raise ValueError(
                f"{REF_STATS_CSV} has no population "
                f"{BAND_POPULATION!r}. Available: "
                f"{sorted(pd.read_csv(REF_STATS_CSV).population.unique())}")
    print(f"  band percentiles: {BAND_POPULATION} "
          f"(n = {int(S.n_images.iloc[0]):,})")
    report.append(("band population", BAND_POPULATION))
    return S.set_index("feature")


def load_corpus(report):
    """The index, the stored features and every eligibility mask."""
    print("\n" + "=" * 60)
    print("CORPUS")
    print("=" * 60)
    for p in (INDEX_CSV, FEATURES_NPY, FEATURE_DEFS, FLAGS_CSV,
              CATEGORY_CSV):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    want = ["image_id", "image_key", "category", "path", "width",
            "height"]
    head = pd.read_csv(INDEX_CSV, nrows=1)
    if "frac_person" in head.columns:
        want.append("frac_person")
    idx = pd.read_csv(INDEX_CSV, usecols=want)
    n = len(idx)
    if not (idx.image_id.to_numpy() == np.arange(n)).all():
        raise ValueError("image_id must be 0..N-1 in row order.")

    with open(FEATURE_DEFS) as f:
        defs = json.load(f)
    if defs["crop"] != CROP:
        raise ValueError("The feature list does not match step 04's.")
    if defs["settings"]["feature_size"] != FEATURE_SIZE:
        raise ValueError("FEATURE_SIZE differs from step 04's.")

    F = np.load(FEATURES_NPY, mmap_mode="r")
    if F.shape != (n, len(CROP)):
        raise ValueError(f"features_crop.npy is {F.shape}.")

    flags = pd.read_csv(FLAGS_CSV).set_index("image_id").reindex(
        np.arange(n))
    colour = ~flags["is_greyscale"].astype(bool).to_numpy()
    toned = (flags["is_toned"].astype(bool).to_numpy()
             if "is_toned" in flags.columns else np.zeros(n, bool))
    if EXCLUDE_TONED and os.path.isfile(TONED_CSV):
        t = pd.read_csv(TONED_CSV)
        if "image_id" in t.columns:
            toned = toned.copy()
            toned[t.image_id.astype(int).to_numpy()] = True

    dup = np.zeros(n, bool)
    if EXCLUDE_DUPLICATES and os.path.isfile(DUP_CSV):
        dup[pd.read_csv(DUP_CSV)["image_id"].astype(int).to_numpy()] = True
    elif EXCLUDE_DUPLICATES:
        warnings.warn(f"{DUP_CSV} not found; duplicates not excluded.")

    cats = pd.read_csv(CATEGORY_CSV)
    keep_names = set(cats.loc[cats["keep"].astype(int) == 1, "category"])
    missing = sorted(keep_names - set(idx.category.unique()))
    if missing:
        warnings.warn(f"{len(missing)} keep=1 categories are absent from "
                      f"the index, e.g. {missing[:3]}")
    keep_cat = idx.category.isin(keep_names).to_numpy()

    side = np.minimum(idx.width.to_numpy(), idx.height.to_numpy())
    sat = np.asarray(F[:, CROP.index(FEAT_COLOUR)], np.float64)
    fog = np.asarray(F[:, CROP.index(FEAT_FOG)], np.float64)

    print(f"  {n:,} images; {int(colour.sum()):,} colour; "
          f"{len(keep_names)} keep=1 categories covering "
          f"{int(keep_cat.sum()):,}")
    report += [("corpus images", f"{n:,}"),
               ("keep=1 categories", str(len(keep_names)))]
    return idx, F, dict(colour=colour, toned=toned, dup=dup,
                        keep_cat=keep_cat, side=side, sat=sat, fog=fog)


# ============================================================
# Checks
# ============================================================

def check_feature_code(idx, F, rng, report):
    """CHECK 1: the copied feature code reproduces data/features."""
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


def check_fog_equivalence(cand, colour, report):
    """CHECK 4: the two fog families give the same reachability.

    They are calibrated on the same feature, so at a given rung they
    should remove the same amount of fine detail -- which is what makes
    one scene list serve both stimulus sets. Verified rather than
    assumed, on a subsample.
    """
    L = pd.read_csv(LEVELS_CSV)
    pairs = []
    for nm in FOG_LEVELS:
        a = L[(L.dimension == "fog") & (L.level == nm)]
        b = L[(L.dimension == "fog_eq") & (L.level == nm)]
        if len(a) and len(b):
            pairs.append((nm, float(a.param.iloc[0]),
                          float(b.param.iloc[0])))
    if not pairs:
        print("  check 4 (fog families): fog_eq not calibrated; skipped")
        report.append(("check 4 fog families", "skipped"))
        return

    sub = cand.head(FOG_EQUIV_N)
    worst, used = 0.0, 0
    for _, r in sub.iterrows():
        try:
            with Image.open(r["path"]) as im:
                sq = np.asarray(central_square(im.convert("RGB"),
                                               IMAGE_SIZE), np.uint8)
        except Exception:
            continue
        ctx = ctx_of(sq)
        base = measure(sq, [FEAT_FOG])[FEAT_FOG]
        for nm, pa, pb in pairs:
            va = measure(render_cell(sq, ctx, 1.0, pa, "haze"),
                         [FEAT_FOG])[FEAT_FOG]
            vb = measure(render_cell(sq, ctx, 1.0, pb, "haze_eq"),
                         [FEAT_FOG])[FEAT_FOG]
            da, db = base - va, base - vb
            if abs(da) > 1e-6:
                worst = max(worst, abs(db - da) / abs(da))
        used += 1
    ok = used > 0 and worst <= FOG_EQUIV_TOL
    print(f"  check 4 (the two fog families remove the same detail, "
          f"{used} scenes): worst relative difference {worst:.3f} -> "
          f"{'PASS' if ok else 'WARN'}")
    report.append(("check 4 fog families",
                   f"{'PASS' if ok else 'WARN'} ({worst:.3f})"))
    if not ok:
        warnings.warn("The two fog families do not remove the same "
                      "amount of detail at matched rungs, so one scene "
                      "list may not serve both stimulus sets. Select "
                      "separately per family, or re-check step 1.")


# ============================================================
# Candidate pool
# ============================================================

def band_counts(idx, M, scores, ref, report):
    """How many images survive at each band in BAND_SWEEP.

    Reported so the bands are chosen from the data rather than
    inherited. Nothing is selected here.
    """
    base = M["colour"] & M["keep_cat"] & (M["side"] >= MIN_SIDE) \
        & ~M["dup"] & ~M["toned"]
    if EXCLUDE_PEOPLE and "frac_person" in idx.columns:
        base &= idx.frac_person.fillna(1.0).to_numpy() <= MAX_PERSON_FRAC
    rows = []
    for lo, hi in BAND_SWEEP:
        m = base.copy()
        for v in (scores["fog"], scores["colour"]):
            a, b = np.percentile(v, lo), np.percentile(v, hi)
            m &= (v >= a) & (v <= b)
        n_axis = int(m.sum())
        for feat, key in ((FEAT_COLOUR, "sat"), (FEAT_FOG, "fog")):
            a = float(ref.at[feat, f"p{int(round(lo)):02d}"]) \
                if f"p{int(round(lo)):02d}" in ref.columns else np.nan
            b = float(ref.at[feat, f"p{int(round(hi)):02d}"]) \
                if f"p{int(round(hi)):02d}" in ref.columns else np.nan
            if np.isfinite(a) and np.isfinite(b):
                m &= (M[key] >= a) & (M[key] <= b)
        rows.append({"pctl_low": lo, "pctl_high": hi,
                     "after_axes": n_axis, "after_features": int(m.sum())})
    bc = pd.DataFrame(rows)
    print("\n  candidates surviving at each band (both axes and both "
          "features):")
    print("  " + bc.to_string(index=False).replace("\n", "\n  "))
    bc.to_csv(os.path.join(OUTPUT_DIR, "band_counts.csv"), index=False)
    return bc


def select_candidates(idx, M, scores, ref, rng, report):
    """Filter the corpus down to N_CANDIDATES, from stored tables."""
    print("\n" + "=" * 60)
    print("CANDIDATE POOL")
    print("=" * 60)
    n = len(idx)
    m = np.ones(n, bool)

    def step(mask, label):
        nonlocal m
        m &= mask
        print(f"  {label:<34} {int(m.sum()):>10,}")

    step(M["keep_cat"], "everyday category")
    step(M["colour"], "colour")
    step(~M["dup"], "not a duplicate")
    step(~M["toned"], "not toned")
    step(M["side"] >= MIN_SIDE, f"short side >= {MIN_SIDE}")
    if EXCLUDE_PEOPLE:
        if "frac_person" in idx.columns:
            step(idx.frac_person.fillna(1.0).to_numpy() <= MAX_PERSON_FRAC,
                 f"frac_person <= {MAX_PERSON_FRAC}")
        else:
            print("  people filter SKIPPED: no frac_person column")

    # --- neutral on both axes ---
    lims = {}
    for label in ("fog", "colour"):
        v = scores[label]
        lo = float(np.percentile(v, AXIS_PCTL_LOW))
        hi = float(np.percentile(v, AXIS_PCTL_HIGH))
        lims[f"axis_{label}"] = (lo, hi)
        step((v >= lo) & (v <= hi), f"axis {label} in "
             f"{AXIS_PCTL_LOW:.0f}-{AXIS_PCTL_HIGH:.0f}%")

    # --- mid-range on both manipulated features ---
    for key, feat, lo_p, hi_p in (("sat", FEAT_COLOUR, SAT_PCTL_LOW,
                                   SAT_PCTL_HIGH),
                                  ("fog", FEAT_FOG, FOG_PCTL_LOW,
                                   FOG_PCTL_HIGH)):
        lo = float(ref.at[feat, f"p{int(round(lo_p)):02d}"])
        hi = float(ref.at[feat, f"p{int(round(hi_p)):02d}"])
        lims[feat] = (lo, hi)
        step((M[key] >= lo) & (M[key] <= hi),
             f"{feat} in {lo_p:.0f}-{hi_p:.0f}% ({lo:.1f}-{hi:.1f})")

    if not m.any():
        raise RuntimeError(
            "No images survive the filters. The counts above show which "
            "one emptied the pool; widen that band, or check that the "
            "band population matches how the features were measured.")

    df = idx[m].copy()
    df["axis_fog"] = scores["fog"][m]
    df["axis_colour"] = scores["colour"][m]
    df[FEAT_COLOUR] = M["sat"][m]
    df[FEAT_FOG] = M["fog"][m]

    # --- files present ---
    df = df[df["path"].apply(os.path.isfile)]
    print(f"  {'on disk':<34} {len(df):>10,}")
    if df.empty:
        raise RuntimeError("No surviving files found on disk. Check that "
                           "the paths resolve from this directory.")

    # --- per-category cap, then the draw ---
    df = df.sample(frac=1.0, random_state=RANDOM_SEED)
    df = df.groupby("category", group_keys=False).head(MAX_PER_CATEGORY)
    print(f"  {'per-category cap':<34} {len(df):>10,} "
          f"({df.category.nunique()} categories)")
    if len(df) < N_CANDIDATES:
        warnings.warn(f"only {len(df)} images survive the filters but "
                      f"N_CANDIDATES = {N_CANDIDATES}. Widen the bands "
                      "or raise MAX_PER_CATEGORY.")
    cand = df.sample(min(N_CANDIDATES, len(df)),
                     random_state=RANDOM_SEED).reset_index(drop=True)
    cand["scene_id"] = [scene_id_of(c, p)
                        for c, p in zip(cand.category, cand.path)]
    cand.to_csv(os.path.join(OUTPUT_DIR, "candidates.csv"), index=False)
    print(f"\n  candidates: {len(cand)} in "
          f"{cand.category.nunique()} categories")
    report.append(("candidates", str(len(cand))))
    report.append(("candidate categories", str(cand.category.nunique())))
    return cand, lims


# ============================================================
# Reachability  (workers)
# ============================================================

W = None


def _init(payload):
    global W
    W = payload
    cv2.setNumThreads(1)
    warnings.filterwarnings("ignore")


def _reach_task(row):
    """Build the grid for one candidate and measure every cell."""
    sid, path = row
    colour, fog, fam = W["colour"], W["fog"], W["fam"]
    try:
        with Image.open(path) as im:
            sq = np.asarray(central_square(im.convert("RGB"), IMAGE_SIZE),
                            np.uint8)
    except Exception as exc:
        return sid, None, str(exc)
    ctx = ctx_of(sq)
    base = measure(sq, [FEAT_COLOUR, FEAT_FOG])
    cells = []
    for cn, k, _ in colour:
        for fn, h, _ in fog:
            v = measure(render_cell(sq, ctx, k, h, fam),
                        [FEAT_COLOUR, FEAT_FOG])
            cells.append({"scene_id": sid, "colour_level": cn,
                          "fog_level": fn, "chroma": k, "haze": h,
                          "sat": v[FEAT_COLOUR], "fog": v[FEAT_FOG]})
    return sid, (base, cells), None


def measure_reachability(cand, colour, fog, report):
    """Build every candidate's grid and ask what it can reach."""
    print("\n" + "=" * 60)
    print("REACHABILITY")
    print("=" * 60)
    n_cells = len(colour) * len(fog)
    print(f"  {len(cand)} candidates x {n_cells} cells = "
          f"{len(cand) * n_cells:,} renders")

    payload = {"colour": colour, "fog": fog, "fam": FOG_FAMILY}
    rows = list(zip(cand.scene_id, cand.path))
    t0 = time.time()
    results, bad = [], 0
    if N_WORKERS > 1:
        with Pool(N_WORKERS, initializer=_init,
                  initargs=(payload,)) as pool:
            for k, out in enumerate(pool.imap_unordered(_reach_task, rows,
                                                        CHUNK), 1):
                results.append(out)
                if k % 50 == 0:
                    print(f"    {k}/{len(rows)}", end="\r")
    else:
        _init(payload)
        for k, r in enumerate(rows, 1):
            results.append(_reach_task(r))
            print(f"    {k}/{len(rows)}", end="\r")
    print(f"\n  elapsed {time.time() - t0:.0f}s")

    # deepest level of each dimension, for the reach fractions
    col_deep, fog_deep = colour[-1], fog[-1]
    meta = cand.set_index("scene_id")

    out, grid_rows = [], []
    for sid, got, err in results:
        if got is None:
            bad += 1
            continue
        base, cells = got
        grid_rows.extend(cells)
        g = pd.DataFrame(cells)

        sat_range = float(g.sat.max() - g.sat.min())
        fog_range = float(g.fog.max() - g.fog.min())

        # reach: the scene's own change at the deepest level, against
        # that level's target
        d_col = float(g[(g.colour_level == col_deep[0])
                        & (g.fog_level == fog[0][0])].sat.iloc[0]
                      - base[FEAT_COLOUR])
        d_fog = float(g[(g.colour_level == "k=1")
                        & (g.fog_level == fog_deep[0])].fog.iloc[0]
                      - base[FEAT_FOG])
        reach_col = abs(d_col) / abs(col_deep[2]) if col_deep[2] else np.nan
        reach_fog = abs(d_fog) / abs(fog_deep[2]) if fog_deep[2] else np.nan

        # saturation-matched fog pair
        best_gap = 0.0
        rec = g.to_dict("records")
        for i, j in itertools.combinations(range(len(rec)), 2):
            a, b = rec[i], rec[j]
            if abs(a["sat"] - b["sat"]) > PAIR_SAT_TOL:
                continue
            best_gap = max(best_gap, abs(a["fog"] - b["fog"]))

        out.append({
            "scene_id": sid,
            "category": meta.at[sid, "category"],
            "path": meta.at[sid, "path"],
            "base_sat": base[FEAT_COLOUR], "base_fog": base[FEAT_FOG],
            "sat_range": sat_range, "fog_range": fog_range,
            "reach_colour": reach_col, "reach_fog": reach_fog,
            "best_matched_fog_gap": best_gap,
            "pass_colour": reach_col >= REACH_FRACTION,
            "pass_fog": reach_fog >= REACH_FRACTION,
            "pass_pair": best_gap >= MIN_FOG_GAP,
        })

    reach = pd.DataFrame(out)
    reach["passed"] = (reach.pass_colour & reach.pass_fog
                       & reach.pass_pair)
    reach.to_csv(os.path.join(OUTPUT_DIR, "reachability.csv"),
                 index=False, float_format="%.6g")
    pd.DataFrame(grid_rows).to_csv(
        os.path.join(OUTPUT_DIR, "grid_measurements.csv"),
        index=False, float_format="%.6g")

    if bad:
        warnings.warn(f"{bad} candidates could not be read.")
    print(f"\n  reach colour : median {reach.reach_colour.median():.2f}, "
          f"{int(reach.pass_colour.sum())}/{len(reach)} pass "
          f"(>= {REACH_FRACTION})")
    print(f"  reach fog    : median {reach.reach_fog.median():.2f}, "
          f"{int(reach.pass_fog.sum())}/{len(reach)} pass")
    print(f"  matched gap  : median "
          f"{reach.best_matched_fog_gap.median():.1f}, "
          f"{int(reach.pass_pair.sum())}/{len(reach)} pass "
          f"(>= {MIN_FOG_GAP})")
    n_ok = int(reach.passed.sum())
    ok = n_ok >= N_FINAL
    print(f"\n  usable scenes: {n_ok}/{len(reach)} -> "
          f"{'PASS' if ok else 'CHECK'} (want {N_FINAL})")
    if not ok:
        print("  Too few. Widen the bands, lower REACH_FRACTION, or "
              "raise N_CANDIDATES.")
    report.append(("usable scenes", f"{n_ok}/{len(reach)}"))
    report.append(("reachability", "PASS" if ok else "CHECK"))
    return reach, ok


# ============================================================
# Shortlist
# ============================================================

def shortlist(reach, report):
    """The scenes that go to hand screening, balanced by category.

    Round-robin across categories -- the 1st of each, then the 2nd of
    each -- so the list is spread rather than filled from whichever
    categories happen to sort first. Within a category the order is
    random: the matched-pair gap varies little across the pool, so
    ordering by it would look principled while being arbitrary.
    """
    print("\n" + "=" * 60)
    print("SHORTLIST")
    print("=" * 60)
    ok = reach[reach.passed].sample(frac=1.0, random_state=RANDOM_SEED)
    ok = ok.assign(rank_in_cat=ok.groupby("category").cumcount())
    ok = ok.sort_values(["rank_in_cat"], kind="stable")
    final = ok.head(N_FINAL).reset_index(drop=True)
    final.to_csv(os.path.join(OUTPUT_DIR, "shortlist.csv"), index=False,
                 float_format="%.6g")
    print(f"  {len(final)} scenes, {final.category.nunique()} categories "
          f"(max {final.category.value_counts().max()} per category)")
    report.append(("shortlist", str(len(final))))
    report.append(("shortlist categories", str(final.category.nunique())))
    return final


# ============================================================
# Screening material
# ============================================================

def write_crops(final, report):
    """Each shortlisted scene as it will be cropped, plus contact
    sheets. Content judgements are made on the plain crop, so these are
    unmanipulated."""
    print("\n" + "=" * 60)
    print("SCREENING MATERIAL")
    print("=" * 60)
    d = os.path.join(OUTPUT_DIR, "crops")
    os.makedirs(d, exist_ok=True)
    thumbs, ok_rows = [], []
    for _, r in final.iterrows():
        try:
            with Image.open(r["path"]) as im:
                sq = central_square(im.convert("RGB"), CROP_SIZE)
        except Exception:
            continue
        sq.save(os.path.join(d, f"{r['scene_id']}.jpg"),
                quality=CROP_QUALITY)
        thumbs.append((r["scene_id"], r["category"],
                       sq.resize((CONTACT_PX, CONTACT_PX),
                                 Image.LANCZOS)))
        ok_rows.append(r["scene_id"])
    print(f"  {len(thumbs)} crops written to {d}")

    n_sheets = 0
    for s in range(0, len(thumbs), CONTACT_PER_SHEET):
        chunk = thumbs[s:s + CONTACT_PER_SHEET]
        nc = 6
        nr = int(np.ceil(len(chunk) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.6 * nr),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for i, (sid, cat, im) in enumerate(chunk):
            ax = axes[i // nc][i % nc]
            ax.imshow(im)
            ax.set_title(f"{s + i + 1}. {cat}\n{sid[-28:]}", fontsize=6)
        fig.suptitle(f"scenes {s + 1}-{s + len(chunk)} of {len(thumbs)}"
                     "   (content screening: framing, legibility, "
                     "people, text)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(OUTPUT_DIR, f"contact_{n_sheets + 1}.png"),
                    dpi=120)
        plt.close(fig)
        n_sheets += 1
    print(f"  {n_sheets} contact sheets")
    report.append(("crops", str(len(thumbs))))
    return set(ok_rows)


def write_grid_sheets(final, colour, fog, report):
    """The full grid for a subset, to spot-check that the manipulation
    behaves on real scenes."""
    d = os.path.join(OUTPUT_DIR, "grid_sheets")
    os.makedirs(d, exist_ok=True)
    n = len(final) if N_GRID_SHEETS is None else min(int(N_GRID_SHEETS),
                                                     len(final))
    done = 0
    for _, r in final.head(n).iterrows():
        try:
            with Image.open(r["path"]) as im:
                sq = np.asarray(central_square(im.convert("RGB"),
                                               IMAGE_SIZE), np.uint8)
        except Exception:
            continue
        ctx = ctx_of(sq)
        nr, nc = len(colour), len(fog)
        fig, axes = plt.subplots(nr, nc, figsize=(1.9 * nc, 2.0 * nr),
                                 squeeze=False)
        for i, (cn, k, _) in enumerate(colour):
            for j, (fn, h, _) in enumerate(fog):
                ax = axes[i][j]
                ax.imshow(render_cell(sq, ctx, k, h, FOG_FAMILY))
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"{fn}\nh={h:.3f}", fontsize=7)
                if j == 0:
                    ax.set_ylabel(f"{cn}\nk={k:.3f}", fontsize=7)
        fig.suptitle(f"{r['category']} — {r['scene_id']}  "
                     f"(colour rows x {FOG_FAMILY} columns)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(d, f"{r['scene_id']}.png"), dpi=110)
        plt.close(fig)
        done += 1
    print(f"  {done} grid sheets in {d}")
    report.append(("grid sheets", str(done)))


def write_worksheet(final, rendered, report):
    """The worksheet to fill in by hand.

    `keep` and `reason` are empty for you to complete. Step 3 reads the
    rows with keep == 1. Keeping `reason` turns the manual step into a
    documented screening procedure rather than an undocumented one.
    """
    cols = ["scene_id", "category", "path", "base_sat", "base_fog",
            "reach_colour", "reach_fog", "best_matched_fog_gap"]
    w = final[[c for c in cols if c in final.columns]].copy()
    w.insert(0, "n", np.arange(1, len(w) + 1))
    w["crop"] = ["crops/" + s + ".jpg" for s in w.scene_id]
    w["rendered"] = [s in rendered for s in w.scene_id]
    w["keep"] = ""
    w["reason"] = ""
    path = os.path.join(OUTPUT_DIR, "screening_list.csv")
    w.to_csv(path, index=False, float_format="%.6g")
    print(f"  worksheet: {path}")
    report.append(("worksheet rows", str(len(w))))
    return path


# ============================================================
# Figures
# ============================================================

def plot_selection(scores, M, cand, lims):
    """Where the candidates sit relative to the eligible corpus."""
    elig = M["colour"] & M["keep_cat"] & (M["side"] >= MIN_SIDE) \
        & ~M["dup"] & ~M["toned"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pairs = [("axis_colour", "axis_fog", scores["colour"], scores["fog"],
              "colour axis (z)", "fog axis (z)", "axis_colour",
              "axis_fog"),
             (FEAT_COLOUR, FEAT_FOG, M["sat"], M["fog"],
              "sat_mean", "contrast_local", FEAT_COLOUR, FEAT_FOG)]
    for ax, (cx, cy, vx, vy, lx, ly, kx, ky) in zip(axes, pairs):
        s = np.random.default_rng(0).choice(np.where(elig)[0],
                                            size=min(40000,
                                                     int(elig.sum())),
                                            replace=False)
        ax.scatter(vx[s], vy[s], s=2, alpha=0.08, color="0.6",
                   label="eligible corpus")
        ax.scatter(cand[cx], cand[cy], s=8, alpha=0.7, color="C3",
                   label="candidates")
        for key, axis in ((kx, "x"), (ky, "y")):
            if key in lims:
                lo, hi = lims[key]
                (ax.axvline if axis == "x" else ax.axhline)(
                    lo, color="k", lw=0.7, ls=":")
                (ax.axvline if axis == "x" else ax.axhline)(
                    hi, color="k", lw=0.7, ls=":")
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.legend(fontsize=8, markerscale=3)
        ax.grid(alpha=0.3)
    axes[0].set_title("Axis position: candidates are near the median on "
                      "both")
    axes[1].set_title("Feature position: room to move in both "
                      "directions")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "selection.png"), dpi=150)
    plt.close(fig)


def plot_reachability(reach):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, thr, lab in (
            (axes[0], "reach_colour", REACH_FRACTION,
             "colour reach (measured / target)"),
            (axes[1], "reach_fog", REACH_FRACTION,
             "fog reach (measured / target)"),
            (axes[2], "best_matched_fog_gap", MIN_FOG_GAP,
             "best saturation-matched fog gap")):
        ax.hist(reach[col].dropna(), bins=40, color="C0", alpha=0.85)
        ax.axvline(thr, color="C3", lw=1.5, ls="--",
                   label=f"threshold {thr}")
        ax.set_xlabel(lab)
        ax.set_ylabel("scenes")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Can the grid move each candidate far enough?")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "reachability.png"), dpi=150)
    plt.close(fig)


# ============================================================
# Report
# ============================================================

def write_report(cand, reach, final, bands, lims, report, elapsed):
    L = []
    W_ = L.append
    W_("SCENE SELECTION REPORT  (stimulus pipeline, v4 axes, step 2)")
    W_("=" * 70)
    W_(f"Ladder        : {len(COLOUR_LEVELS) + 1} colour x "
       f"{len(FOG_LEVELS)} fog, {FOG_FAMILY}")
    W_(f"Levels from   : {LEVELS_CSV}")
    W_(f"Band stats    : {BAND_POPULATION} in {REF_STATS_CSV}")
    W_("                (corpus path, matching the stored features the "
       "bands are applied to;")
    W_("                 step 1's targets use the stimulus path, which "
       "is the right")
    W_("                 reference for comparing against a measured "
       "stimulus)")
    W_(f"Axis band     : {AXIS_PCTL_LOW:.0f}-{AXIS_PCTL_HIGH:.0f}%")
    W_(f"Feature bands : sat {SAT_PCTL_LOW:.0f}-{SAT_PCTL_HIGH:.0f}%, "
       f"fog {FOG_PCTL_LOW:.0f}-{FOG_PCTL_HIGH:.0f}%")
    W_(f"Elapsed       : {elapsed:.0f}s")
    W_("")
    for k, v in report:
        W_(f"  {k:32s} {v}")

    W_("\n1. BAND SWEEP")
    W_("-" * 70)
    W_("  How many images survive at each band setting. The bands decide "
       "how neutral a")
    W_("  scene has to be to start with; too narrow and the pool is thin, "
       "too wide and")
    W_("  scenes arrive already foggy or already drab. Chosen band is "
       f"{AXIS_PCTL_LOW:.0f}-{AXIS_PCTL_HIGH:.0f}%.")
    W_("  " + bands.to_string(index=False).replace("\n", "\n  "))

    W_("\n2. WHAT THE BANDS CUT ON")
    W_("-" * 70)
    for k, (lo, hi) in lims.items():
        W_(f"      {k:<14} {lo:10.4f} .. {hi:10.4f}")

    W_("\n3. REACHABILITY")
    W_("-" * 70)
    W_("  reach = the scene's own measured change at the deepest level, "
       "divided by that")
    W_("  level's target change. 1.0 means the scene moves exactly as "
       "far as intended;")
    W_(f"  a scene passes at {REACH_FRACTION}. The change is "
       "proportional to the scene's")
    W_("  starting value, so a scene that starts near the corpus median "
       "lands near 1.")
    cols = ["reach_colour", "reach_fog", "best_matched_fog_gap",
            "sat_range", "fog_range"]
    W_("  " + reach[cols].describe().loc[
        ["min", "25%", "50%", "75%", "max"]].round(2)
        .to_string().replace("\n", "\n  "))
    W_("")
    for c, thr in (("pass_colour", REACH_FRACTION),
                   ("pass_fog", REACH_FRACTION),
                   ("pass_pair", MIN_FOG_GAP)):
        W_(f"      {c:<14} {int(reach[c].sum()):>5} / {len(reach)} "
           f"(threshold {thr})")
    fails = reach[~reach.passed]
    if len(fails):
        why = []
        for c in ("pass_colour", "pass_fog", "pass_pair"):
            why.append(f"{c.replace('pass_', '')} "
                       f"{int((~fails[c]).sum())}")
        W_(f"      dropped {len(fails)}: " + ", ".join(why))

    W_("\n4. THE SHORTLIST")
    W_("-" * 70)
    W_(f"  {len(final)} scenes across {final.category.nunique()} "
       "categories.")
    vc = final.category.value_counts()
    W_(f"  per category: min {vc.min()}, median {int(vc.median())}, "
       f"max {vc.max()}")
    W_("")
    W_("  NEXT: open the contact sheets, then fill in `keep` (1 to keep, "
       "0 to drop) and")
    W_("  `reason` in screening_list.csv. Reject freely -- the "
       "reachability check is a")
    W_("  guard against unusable scenes, not a selection mechanism, so "
       "rejecting on")
    W_("  content costs nothing in feature coverage. Step 3 selects the "
       "final ~30 from")
    W_("  whatever you keep, and re-checks reachability on them.")
    W_("")
    W_("  Judge on CONTENT: framing, legibility at the size shown, "
       "visible people or")
    W_("  faces, readable text, and whether the place is ordinary rather "
       "than a showpiece.")

    path = os.path.join(OUTPUT_DIR, "select_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    return path


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.setNumThreads(1)
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    rng = np.random.default_rng(RANDOM_SEED)
    report = []

    print("=" * 60)
    print("SELECT SCENES  (v4 stimulus pipeline, step 2)")
    print("=" * 60)
    print(f"  Index  : {INDEX_CSV}")
    print(f"  Scores : {SCORES_DIR}")
    print(f"  Levels : {LEVELS_CSV}")
    print(f"  Output : {OUTPUT_DIR}")

    idx, F, M = load_corpus(report)
    colour, fog = load_levels(report)
    scores = load_scores(len(idx), report)
    ref = load_reference(report)

    print("\n" + "=" * 60)
    print("START-UP CHECKS")
    print("=" * 60)
    check_feature_code(idx, F, rng, report)

    bands = band_counts(idx, M, scores, ref, report)
    cand, lims = select_candidates(idx, M, scores, ref, rng, report)
    check_fog_equivalence(cand, colour, report)

    reach, _ = measure_reachability(cand, colour, fog, report)
    final = shortlist(reach, report)

    rendered = write_crops(final, report)
    write_grid_sheets(final, colour, fog, report)
    write_worksheet(final, rendered, report)

    plot_selection(scores, M, cand, lims)
    plot_reachability(reach)

    path = write_report(cand, reach, final, bands, lims, report,
                        time.time() - t0)
    print(f"\n  Report : {path}")
    print(f"  Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
