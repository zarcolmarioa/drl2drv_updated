"""
09c_manipulation_sweep.py  (pipeline step 09, part c: manipulation sweep)

The corpus analysis (07, 09a, 09b) shows which image features go with
which axes. Correlation is not intervention (review G2): this script
manipulates each feature in real photographs, re-encodes them with
un2CLIP, and measures how far every axis moves, how specifically, and
what else the manipulation changed. It replaces the earlier sweep scripts
(Scripts C-E: v2 axes, stretched images, hand-picked levels).

Version 2 (5 Oct)
-----------------
  * The six manipulations the first run capped at once by clipping are
    rebuilt so that they cannot clip: chroma up (gamut-limited boost),
    exposure up (highlight shoulder), contrast up (S-curve), temperature
    (gains <= 1, original lightness restored, gamut mapping) and hue
    compression (gamut mapping). Their other directions are unchanged.
  * Levels stop where the calibrated feature's median change peaks
    (tunnel blur p99 and band blur 2x were placed beyond the peak).
  * New: effects beyond the quality control, and beyond quality, mood and
    threat (report sections 3 and 4; effects.csv).
  * The calibration cache is reused: only the rebuilt manipulations are
    recomputed (FAMILY_VERSION is part of the cache key). Copy the first
    run's calibration_cache.npz into OUTPUT_DIR (data/sweep_v2) first.
  * Note: contrast up works on each channel, so strong levels also raise
    saturation and posterise (see the side effects); contrast down, also
    per channel, lowers saturation in the same way.

Images
------
N_IMAGES colour photographs, one per category (categories drawn at
random), with every selected feature defined, short side >= MIN_SIDE, and
no copy in 09a's duplicate groups. Each is CLIP's central square
(the region 04 and 06 describe) resized to IMAGE_SIZE (512) px; all
manipulations are applied at that size.

Measurement (the same pipeline for every variant, including level 0)
--------------------------------------------------------------------
  features  the 512 px image resized to 256 px, then step 04's 18
            features and step 04c's 10 features (their code, copied
            verbatim and checked against the stored values: check 2).
  CLIP      the 512 px image with the official preprocessing (bicubic
            resize to 336), un2CLIP with its own projection (as step 03;
            check 1 re-encodes the original files and compares them with
            the stored embeddings), then every axis vector of step 05.
Effects are in units of the axis's corpus SD (06's z scale), relative to
the same image's level 0.

Manipulations (family, parameter, feature it is calibrated on)
--------------------------------------------------------------
  Both directions (10 levels + level 0):
    chroma        CIELAB chroma x k at constant L*     sat_mean
                  and hue; up: gamut-limited boost
                  towards each pixel's gamut boundary
    exposure      linear-light gain 2^ev; up: with a   lum_mean
                  highlight shoulder (white stays white)
    contrast      each channel around the mean grey:   lum_std
                  down linear x k; up S-curve with
                  slope k at the mean, ends fixed
    gamma         each channel 255 (x/255)^g           lum_skew
    temperature   linear R x 2^t, B x 2^-t divided by  cast_b
                  the larger gain, original L* back,
                  chroma gamut-mapped
  One direction (5 levels + level 0):
    lift          x' = l*255 + (1-l) x (black lift)    lum_p05   up
    posterise     n levels per channel (bin centres)   lum_entropy down
    haze          x' = (1-h) x + h A, A = the image's  dark_channel_mean up
                  own airlight (mean colour of its
                  brightest 1%)
    haze_eq       haze, then linear-light gain back to the level-0 mean
                  grey (brightness-equated; the SAME parameters as haze,
                  not calibrated separately)
    blur          Gaussian, sigma px at 512            contrast_local down
    sharpen       unsharp mask (sigma 3 at 512)        contrast_local up
    smooth        bilateral filter x3 (loss of fine    fine_texture   down
                  detail), sigma_color
    grain         Gaussian luminance noise, sigma      noise_sigma    up
    hue_compress  hues pulled to the mean hue,         hue_spread     down
                  chroma gamut-mapped at the new hue
    vignette      tunnel darkening, 1 - v r^2          periph_lum     down
    tunnel_blur   peripheral blur, sigma               periph_blur    up
    band_blur     miniature (band) blur, sigma         miniature_index up
Glow is not included (removed by decision). periph_blur and
miniature_index are unreliable measures in natural photographs (04c,
F1); here they only set the levels, which are also reported as
parameters.

Levels (review 0.2)
-------------------
Set on the measured feature, not on the parameter. For an increase, the
target changes are the corpus's (colour images, 256 px) p75 - p50,
p90 - p50 and p99 - p50 ("natural range"), and 2D and 4D with
D = p99 - p50 ("extended"); for a decrease the same with p25, p10, p01.
CALIBRATION: every parameter on a grid of GRID_N values is applied to
all sweep images; the level's parameter is where the MEDIAN change over
the images reaches the target (interpolated on the grid). A level is
CAPPED where the mean share of NEWLY CLIPPED pixels reaches CLIP_MAX_NEW
(interpolated between grid points), or at the end of the grid if the
target is out of reach; levels beyond a capped one are dropped.
Newly clipped = a channel that was inside the range (0 < value < 255)
pushed outside it (blown highlights, crushed shadows, out-of-gamut
colours); channels already at 0 or 255 in the photograph do not count.
Grain and sharpening clip by nature at the extremes (noise tails, edge
halos): for them (CLIP_EXEMPT) clipping is recorded but does not cap.
A level is also capped where the median change of the calibrated feature
PEAKS on the grid ("capped (feature peaks)"): some measures fall again at
extreme strengths (the regional blur measures once the edges vanish), and
a stronger parameter there would not be a stronger level.
Everything is recorded in levels.csv and calibration_grid.csv.

Statistics
----------
For each variant level and axis: the mean change over images (corpus SD
units), a 95% interval from resampling images (N_BOOT), and the share of
images moving in the mean's direction. For the v3b axes also:
  unique   the change of the axis beyond what the other seven v3b axes'
           changes predict, using the corpus relation between the axes
           (within-category regression of each axis on the other seven,
           colour images; the manipulation analogue of 09a's C13).
  share    d_axis^2 / sum of d^2 over the eight axes: 1 = only this axis
           moved, 0.125 = all moved equally.
  per SD   d divided by the median change of the calibrated feature in
           corpus SD units: effect per unit of measured change (effects.csv).
  beyond   the change beyond what the change of the quality control
           ("beyond q") or of quality, mood and threat ("beyond qmt")
           predicts, using the corpus relation between each axis and those
           controls (within-category regression, colour images). Separates
           "looks derealised" from "looks like a bad / sad / scary photo",
           under the assumption that the manipulation moves both in the
           ratio natural variation does.
The quality control ("a bad photo" vs "a good photo"; + = worse) is the
out-of-distribution check: a manipulation that mainly makes the image
look like a bad photo is reported as such.
Side effects: the median change of every feature (corpus SD units).

Checks (the run stops if one fails)
-----------------------------------
  1. ENCODER    the sweep images' original files, preprocessed as step 03
                does, reproduce 06's stored embeddings (cosine >=
                EMB_CHECK_MIN_COS). Also reported: cosine between level 0
                (via the 512 px square) and the stored embedding.
  2. FEATURES   the copied feature code on the original files (central
                square at 256, as 04 / 04c) reproduces the stored
                features (relative tolerance FEATURE_CHECK_RTOL).
  3. PREPROCESS the resize to 336 done in the workers equals the official
                preprocessing on the same 512 px image (difference 0).
  4. AXES       the stored axis vectors are unit length.
  5. LAB        the unclipped Lab -> sRGB conversion used by the colour
                manipulations agrees with OpenCV's inside the gamut
                (round trip of the sweep images, max |diff| < 1 grey
                level).

Inputs
------
  INDEX_CSV, FLAGS_CSV, FEATURES_DIR, EXTRA_DIR, SELECTION_JSON,
  AXES_DIR (axis_catalog.csv, axis_vectors.npy), SCORES_DIR
  (score_catalog.csv, scores_z.npy), EMB_PATH, DUP_CSV (09a's
  duplicate_groups.csv), UN2CLIP_CKPT_PATH (as step 03).

Outputs (under OUTPUT_DIR)
--------------------------
  sweep_report.txt       summary (start here)
  sweep_images.csv       the images
  calibration_grid.csv   every grid parameter: median change of the
                         calibrated feature, mean newly clipped share
  levels.csv             every level: target, parameter, status, achieved
                         median change
  variants.csv           one row per image x level: parameter, clipped
                         share, all 28 features
  embeddings.npy         (variants, 768) un2CLIP embeddings, rows as in
                         variants.csv (rescoring needs no re-encoding)
  variant_scores.npy     (variants, axes) raw scores on every step-05 axis
  score_keys.csv         the columns of variant_scores.npy
  effects.csv            level x axis: mean d, interval, same-sign share;
                         unique d, beyond q / qmt and share for the v3b
                         axes
  side_effects.csv       level x feature: median change (corpus SD units)
  contact_<family>.png   CONTACT_N images at every level
  run_meta.json

Usage
-----
    conda activate drl2drv
    python 09c_manipulation_sweep.py

Dry run first: N_IMAGES = 12 and a separate OUTPUT_DIR (about 5-10
minutes). Look at levels.csv: which levels are capped by clipping, and
calibration_grid.csv: how fast clipping grows. Brightening, contrast,
chroma and temperature increases push pixels out of range early in 8-bit
images; if the 1% rule caps them below the natural range, decide on
CLIP_MAX_NEW before the full run.
Full run (150 images, up to 111 variants each): calibration and variants
are CPU work in N_WORKERS processes (roughly 15-40 minutes with 8
workers); encoding runs on the GPU alongside. Memory about 3-5 GB.
Calibration results are cached (calibration_cache.npz): with the same
images, a rerun re-uses every grid point already evaluated, so changing
CLIP_MAX_NEW, the percentiles or a grid only costs what is new
(REUSE_CALIBRATION = False recomputes everything).
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

# --- Inputs ---
INDEX_CSV       = "data/selection/image_index.csv"
FLAGS_CSV       = "data/monochrome/image_flags.csv"
FEATURES_DIR    = "data/features"
EXTRA_DIR       = "data/features_extra"            # 04c (version 2)
SELECTION_JSON  = "data/feature_selection_v3/selected_features.json"
AXES_DIR        = "data/axes"
SCORES_DIR      = "data/scores"
EMB_PATH        = "data/embeddings/embeddings_768.npy"
DUP_CSV         = "data/checks_v3/duplicate_groups.csv"   # 09a (C12)
OUTPUT_DIR      = "data/sweep_v2"   # copy data/sweep/calibration_cache.npz
                                    # here to reuse the first calibration

# --- Encoder (as step 03) ---
CLIP_MODEL_NAME   = "openai/clip-vit-large-patch14-336"
UN2CLIP_CKPT_PATH = "./pretrained_models/openai_vit_l_14_336.ckpt"
CLIP_PX           = 336
BATCH_SIZE        = 64
ALLOW_TF32        = False

# --- Images ---
N_IMAGES        = 150
IMAGE_SIZE      = 512       # manipulations are applied at this size
FEATURE_SIZE    = 256       # features measured at this size (as 04)
MIN_SIDE        = 512       # short side of the original, px

# --- Manipulations and levels ---
MANIPULATIONS   = None      # None = all; or a list of family names
UP_PERCENTILES  = [75, 90, 99]
DOWN_PERCENTILES = [25, 10, 1]
EXTENDED        = [2, 4]    # multiples of D = p99 - p50 (p01 - p50)
GRID_N          = 24        # grid points per direction (plus identity)
CLIP_MAX_NEW    = 0.01      # mean share of newly clipped pixels allowed
CLIP_EXEMPT     = ["grain", "sharpen"]   # clipping recorded, not capped
                            # (noise tails and edge halos clip by nature)
HAZE_AIRLIGHT_TOP = 1.0     # % brightest pixels whose mean is the airlight
GAMUT_ITERS     = 16        # bisection steps for the gamut boundary
GAMUT_CMAX_SEARCH = 160.0   # upper bound of the search (CIELAB chroma)
GAMUT_MARGIN    = 0.995     # stay this fraction inside the boundary

# --- Axes reported ---
PRIMARY_SET     = "v3b"
COMPARE_SET     = "v2"
QUALITY_KEY     = "controls::quality"     # + = "a bad photo"
# "beyond" adjustments: each v3b axis's change minus what these control
# axes' changes predict (corpus relation, within categories)
ADJUST          = {"q": ["controls::quality"],
                   "qmt": ["controls::quality", "controls::mood",
                           "controls::threat"]}

# --- Statistics ---
N_BOOT          = 2000
CI_LEVEL        = 0.95
SIDE_MIN        = 0.5       # side effects reported from |median| >= this SD

# --- Checks ---
EMB_CHECK_MIN_COS  = 0.99999
FEATURE_CHECK_N    = 8
FEATURE_CHECK_RTOL = 1e-4

# --- Run ---
N_WORKERS       = 8
REUSE_CALIBRATION = True
CONTACT_N       = 4
CONTACT_PX      = 150
RANDOM_SEED     = 42


# ============================================================
# Step 04 / 04c feature code (copied verbatim; check 2)
# ============================================================

CANNY_LOW, CANNY_HIGH = 100, 200
SPECTRUM_FMIN, SPECTRUM_FMAX = 8, 64
DARK_PATCH_AT_256 = 7
LEGACY = ["lum_mean", "lum_std", "lum_p05", "lum_p50", "lum_p95",
          "sat_mean", "sat_std", "val_mean", "colorfulness",
          "contrast_local", "edge_density"]
EXTRA = ["lum_skew", "lum_entropy", "chroma_mean", "lab_a_mean",
         "lab_b_mean", "spectral_slope", "dark_channel_mean"]
CROP = LEGACY + EXTRA

REBLUR_SIGMA_AT_256 = 1.0
BLUR_DETECT_SIGMA_AT_256 = 1.5
BLUR_CANNY = (10, 25)
BLUR_GRAD_MIN = 1.5
BLUR_SIGMA_CAP = 8.0
MIN_EDGES_IMAGE_AT_256 = 100
MIN_EDGES_REGION_AT_256 = 30
PERIPH_COL_OUTER = 0.15
PERIPH_COL_INNER = 0.30
CHROMA_MIN = 5.0
CHROMATIC_MIN_FRACTION = 0.05
CAST_TOP_PERCENT = 5.0
CAST_CLIP_RGB = 250
CAST_MIN_NONCLIPPED = 0.10
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
NEW = ["edge_blur", "periph_blur", "miniature_index", "periph_lum",
       "hue_spread", "cast_a", "cast_b", "fine_texture", "noise_sigma",
       "edge_overshoot"]
ALL_FEATURES = CROP + NEW


def clip_box(w, h, n=336):
    if w <= h:
        nw, nh = n, int(n * h / w)
    else:
        nw, nh = int(n * w / h), n
    left = int(round((nw - n) / 2.0))
    top = int(round((nh - n) / 2.0))
    sx, sy = nw / w, nh / h
    return (left / sx, top / sy, (left + n) / sx, (top + n) / sy)


def central_square(im, size, resample=Image.BILINEAR):
    return im.resize((size, size), resample, box=clip_box(*im.size))


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
    slope = np.polyfit(np.log(f[sel]), np.log(a), 1)[0]
    return float(-slope)


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


def grey(rgb_u8):
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    g8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return g8, g8.astype(np.float32)


def sobel(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return gx, gy, np.sqrt(gx * gx + gy * gy)


def edge_blur_map(g8, g, mag, s):
    sig0 = REBLUR_SIGMA_AT_256 * s
    sigd = BLUR_DETECT_SIGMA_AT_256 * s
    _, _, mag2 = sobel(cv2.GaussianBlur(g, (0, 0), sig0))
    gs = cv2.GaussianBlur(g, (0, 0), sigd)
    _, _, mags = sobel(gs)
    edges = cv2.Canny(np.clip(np.round(gs), 0, 255).astype(np.uint8),
                      BLUR_CANNY[0], BLUR_CANNY[1], L2gradient=True) > 0
    m = int(np.ceil(3 * max(sig0, sigd))) + 1
    edges[:m, :] = edges[-m:, :] = False
    edges[:, :m] = edges[:, -m:] = False
    edges &= mags >= BLUR_GRAD_MIN
    ys, xs = np.nonzero(edges)
    R = mag[ys, xs] / np.maximum(mag2[ys, xs], 1e-6)
    sig = np.full(R.shape, BLUR_SIGMA_CAP, np.float64)
    ok = R > 1.0 + 1e-3
    sig[ok] = np.minimum(sig0 / np.sqrt(R[ok] ** 2 - 1.0) / s,
                         BLUR_SIGMA_CAP)
    return ys, xs, sig


def blur_features(ys, xs, sig, n, s):
    min_img = MIN_EDGES_IMAGE_AT_256 * s
    min_reg = MIN_EDGES_REGION_AT_256 * s
    edge_blur = float(np.median(sig)) if len(sig) >= min_img else np.nan
    row = np.minimum((ys * 3) // n, 2)
    col = np.minimum((xs * 3) // n, 2)

    def cell(r, c):
        m = (row == r) & (col == c)
        return float(np.median(sig[m])) if m.sum() >= min_reg else np.nan

    def pair(a, b):
        v = [x for x in (a, b) if np.isfinite(x)]
        return float(np.mean(v)) if v else np.nan

    C = cell(1, 1)
    LR = pair(cell(1, 0), cell(1, 2))
    TB = pair(cell(0, 1), cell(2, 1))
    return edge_blur, LR - C, TB - LR


def periph_lum(g):
    n = g.shape[1]
    c0 = int(round(PERIPH_COL_OUTER * n))
    c1 = n - c0
    m0 = int(round((0.5 - PERIPH_COL_INNER / 2) * n))
    m1 = int(round((0.5 + PERIPH_COL_INNER / 2) * n))
    h = g.shape[0]
    r = []
    for k in range(3):
        rows = slice(k * h // 3, (k + 1) * h // 3)
        outer = np.concatenate([g[rows, :c0].ravel(),
                                g[rows, c1:].ravel()]).mean()
        central = g[rows, m0:m1].mean()
        r.append(np.log((outer + 1.0) / (central + 1.0)))
    return float(np.mean(r))


def colour_features(rgb_u8):
    lab = cv2.cvtColor(rgb_u8.astype(np.float32) / 255.0,
                       cv2.COLOR_RGB2Lab)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    C = np.sqrt(a * a + b * b)
    chrom = C >= CHROMA_MIN
    cfrac = float(chrom.mean())
    if cfrac >= CHROMATIC_MIN_FRACTION:
        hue = 1.0 - float(np.hypot(a[chrom].sum(), b[chrom].sum())
                          / C[chrom].sum())
    else:
        hue = np.nan
    nonclip = rgb_u8.max(axis=2) < CAST_CLIP_RGB
    if nonclip.mean() >= CAST_MIN_NONCLIPPED:
        thr = np.percentile(L[nonclip], 100.0 - CAST_TOP_PERCENT)
        top = nonclip & (L >= thr)
        ca, cb = float(a[top].mean()), float(b[top].mean())
    else:
        ca = cb = np.nan
    return hue, ca, cb, cfrac


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


def new_features(rgb_u8, only=None):
    """04c's 10 features (NEW), or only those named in `only`."""
    n = rgb_u8.shape[0]
    s = n / 256.0
    want = set(NEW if only is None else only)
    g8, g = grey(rgb_u8)
    v = {}
    if want & {"edge_blur", "periph_blur", "miniature_index",
               "edge_overshoot"}:
        gx, gy, mag = sobel(g)
        if want & {"edge_blur", "periph_blur", "miniature_index"}:
            ys, xs, sig = edge_blur_map(g8, g, mag, s)
            (v["edge_blur"], v["periph_blur"],
             v["miniature_index"]) = blur_features(ys, xs, sig, n, s)
        if "edge_overshoot" in want:
            v["edge_overshoot"] = edge_overshoot(g8, g, gx, gy, mag, s)
    if "periph_lum" in want:
        v["periph_lum"] = periph_lum(g)
    if want & {"hue_spread", "cast_a", "cast_b"}:
        v["hue_spread"], v["cast_a"], v["cast_b"], _ = colour_features(rgb_u8)
    if "fine_texture" in want:
        v["fine_texture"] = fine_texture(g, s)
    if "noise_sigma" in want:
        v["noise_sigma"] = noise_sigma(g, s)
    return v


def measure(rgb_u8, only=None):
    """Features of a square uint8 image (any size >= FEATURE_SIZE): the
    image is resized to FEATURE_SIZE first, as the corpus was measured.
    Returns {name: value} for ALL_FEATURES or the names in `only`."""
    if rgb_u8.shape[0] != FEATURE_SIZE:
        rgb_u8 = np.asarray(Image.fromarray(rgb_u8).resize(
            (FEATURE_SIZE, FEATURE_SIZE), Image.BILINEAR), dtype=np.uint8)
    want = ALL_FEATURES if only is None else list(only)
    out = {}
    if any(f in CROP for f in want):
        out.update(zip(CROP, all_features(rgb_u8)))
    nw = [f for f in want if f in NEW]
    if nw:
        out.update(new_features(rgb_u8, nw))
    return {f: out[f] for f in want}


# ============================================================
# Manipulations (applied at IMAGE_SIZE)
# ============================================================
# Every manipulation takes a uint8 RGB image and returns FLOAT values on
# the 0-255 scale, NOT clipped: apply() then records the share of pixels
# pushed outside the displayable range (any channel < 0 or > 255, e.g.
# blown highlights or out-of-gamut colours) and converts to uint8. That
# share is the clipping the level rule caps (CLIP_MAX_NEW).

def to_u8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def srgb_to_lin(u8):
    x = u8.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb_f(x):
    """Linear light -> sRGB on the 0-255 scale, unclipped (values above 1
    in linear light map above 255)."""
    x = np.maximum(x, 0.0)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return (y * 255.0).astype(np.float32)


def lab_of(u8):
    return cv2.cvtColor(u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


_XN, _ZN = 0.950456, 1.088754                  # OpenCV's D65 white
_M_XYZ = np.array([[0.412453, 0.357580, 0.180423],  # OpenCV's sRGB matrix
                   [0.212671, 0.715160, 0.072169],
                   [0.019334, 0.119193, 0.950227]])
_M_RGB = np.linalg.inv(_M_XYZ).astype(np.float32)


def rgb_of_f(lab):
    """CIELAB -> sRGB on the 0-255 scale, UNCLIPPED: out-of-gamut colours
    fall outside 0-255. (OpenCV's float Lab2RGB clips silently, which
    would hide gamut clipping; this uses OpenCV's constants and agrees
    with it inside the gamut, see check 5.)"""
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
    """CIELAB -> linear sRGB (OpenCV's constants), no clipping."""
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0

    def finv(t):
        return np.where(t > 6.0 / 29.0, t ** 3,
                        3.0 * (6.0 / 29.0) ** 2 * (t - 4.0 / 29.0))
    xyz = np.stack([_XN * finv(fx), finv(fy), _ZN * finv(fz)], axis=-1)
    return xyz.astype(np.float32) @ _M_RGB.T


def cmax_of(L, h, iters=GAMUT_ITERS):
    """Largest CIELAB chroma inside the sRGB gamut at lightness L and hue
    angle h (radians), per pixel, by bisection. The gamut at fixed L and
    hue is a segment from the grey axis outwards, so every chroma below
    this value is displayable."""
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


def lab_from_lch(L, C, h):
    return np.stack([L, C * np.cos(h), C * np.sin(h)], axis=-1)


def mean_grey(u8):
    return float(cv2.cvtColor(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR),
                              cv2.COLOR_BGR2GRAY).mean())


def _radius(n, corner_one=False):
    y, x = np.indices((n, n), dtype=np.float32)
    r = np.hypot(x + 0.5 - n / 2.0, y + 0.5 - n / 2.0) / (n / 2.0)
    return r / np.sqrt(2.0) if corner_one else r


def _f(img):
    return img.astype(np.float32)


def m_chroma(img, k, ctx):
    """Chroma x k at constant lightness and hue. k <= 1: plain scaling.
    k > 1 (version 2): gamut-limited boost. Each pixel moves towards its
    own gamut boundary Cmax (at its lightness and hue):
        C' = C + H (1 - exp(-(k - 1) C / H)),  H = Cmax - C,
    which equals k C for small boosts, never reaches the boundary, and
    leaves pixels already at the boundary unchanged (vibrance-like)."""
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


def m_exposure(img, ev, ctx):
    """Linear-light gain 2^ev. ev <= 0: plain gain. ev > 0 (version 2):
    gain with a highlight shoulder, y = G x / (1 + (G - 1) x): slope G in
    the shadows, white stays white, nothing is pushed past white."""
    x = srgb_to_lin(img)
    G = 2.0 ** ev
    if ev <= 0:
        return lin_to_srgb_f(x * G)
    return lin_to_srgb_f(G * x / (1.0 + (G - 1.0) * x))


def m_contrast(img, k, ctx):
    """Each channel around the image's mean grey m. k <= 1: linear,
    m + k (x - m). k > 1 (version 2): an S-curve with slope k at m and
    fixed end points (0 and 255 stay put, nothing leaves the range):
        y = m (x/m)^k                         below m,
        y = 1 - (1 - m) ((1 - x)/(1 - m))^k   above m   (on 0-1)."""
    m = mean_grey(img)
    if k <= 1.0:
        return m + k * (_f(img) - m)
    t = _f(img) / 255.0
    mm = float(np.clip(m / 255.0, 0.02, 0.98))
    lo = mm * np.power(np.clip(t / mm, 0, None), k)
    hi = 1.0 - (1.0 - mm) * np.power(np.clip((1.0 - t) / (1.0 - mm), 0, None), k)
    return 255.0 * np.where(t < mm, lo, hi)


def m_gamma(img, g, ctx):
    """Each channel: 255 (x / 255)^g (never leaves the range)."""
    return 255.0 * (_f(img) / 255.0) ** g


def m_temperature(img, t, ctx):
    """White-balance shift (version 2). Linear-light gains R x 2^t,
    B x 2^-t, divided by the larger gain so that no channel is pushed past
    white; then each pixel gets its original CIELAB lightness back (so the
    shift changes colour, not brightness), and chroma is limited to the
    gamut at that lightness and hue (gamut mapping instead of clipping)."""
    gains = np.array([2.0 ** t, 1.0, 2.0 ** (-t)], np.float32)
    gains /= gains.max()
    x = srgb_to_lin(img) * gains[None, None, :]
    shifted = lin_to_srgb_f(x) / 255.0
    lab2 = cv2.cvtColor(np.clip(shifted, 0, 1).astype(np.float32),
                        cv2.COLOR_RGB2Lab)
    L = lab_of(img)[:, :, 0]
    a, b = lab2[:, :, 1], lab2[:, :, 2]
    C, h = np.sqrt(a * a + b * b), np.arctan2(b, a)
    C = np.minimum(C, cmax_of(L, h))
    return rgb_of_f(lab_from_lch(L, C, h))


def m_lift(img, l, ctx):
    return 255.0 * l + (1.0 - l) * _f(img)


def m_posterise(img, n, ctx):
    """n levels per channel, each bin replaced by its centre."""
    n = int(round(n))
    if n >= 256:
        return _f(img)
    q = np.floor(_f(img) * n / 256.0)
    return (q + 0.5) * 256.0 / n - 0.5


def m_haze(img, h, ctx):
    A = np.asarray(ctx["airlight"], np.float32)[None, None, :]
    return (1.0 - h) * _f(img) + h * A


def m_haze_eq(img, h, ctx):
    """Haze, then a linear-light gain that restores the level-0 mean grey
    (bisection on the gain)."""
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


def m_blur(img, s, ctx):
    return _f(img) if s <= 0 else cv2.GaussianBlur(_f(img), (0, 0), s)


def m_sharpen(img, a, ctx):
    f = _f(img)
    return f + a * (f - cv2.GaussianBlur(f, (0, 0), 3.0))


def m_smooth(img, sc, ctx):
    if sc <= 0:
        return _f(img)
    out = img
    for _ in range(3):
        out = cv2.bilateralFilter(out, -1, float(sc), 6.0)
    return _f(out)


def m_grain(img, s, ctx):
    noise = np.random.default_rng(ctx["seed"]).normal(0, 1, img.shape[:2])
    return _f(img) + s * noise[:, :, None].astype(np.float32)


def m_hue_compress(img, v, ctx):
    lab = lab_of(img)
    a, b = lab[:, :, 1], lab[:, :, 2]
    C = np.sqrt(a * a + b * b)
    chrom = C >= CHROMA_MIN
    if not chrom.any() or v <= 0:
        return _f(img)
    h0 = np.arctan2(b[chrom].sum(), a[chrom].sum())
    h = np.arctan2(b, a)
    d = np.angle(np.exp(1j * (h0 - h)))
    h2 = h + v * d
    # version 2: chroma limited to the gamut at the new hue (a colour
    # rotated to another hue at the same lightness and chroma may not
    # exist in sRGB)
    C2 = np.minimum(C, cmax_of(lab[:, :, 0], h2))
    return rgb_of_f(lab_from_lch(lab[:, :, 0], C2, h2))


def m_vignette(img, v, ctx):
    w = np.clip(1.0 - v * _radius(img.shape[0], True) ** 2, 0, 1)
    return _f(img) * w[:, :, None]


def m_tunnel_blur(img, s, ctx):
    f = _f(img)
    if s <= 0:
        return f
    w = np.clip((_radius(img.shape[0]) - 0.4) / 0.5, 0, 1)[:, :, None]
    return (1 - w) * f + w * cv2.GaussianBlur(f, (0, 0), s)


def m_band_blur(img, s, ctx):
    f = _f(img)
    if s <= 0:
        return f
    n = img.shape[0]
    y = np.abs(np.arange(n, dtype=np.float32) + 0.5 - n / 2.0) / (n / 2.0)
    w = np.clip((y - 1.0 / 3.0) / 0.3, 0, 1)[:, None, None]
    return (1 - w) * f + w * cv2.GaussianBlur(f, (0, 0), s)


def geo(lo, hi, n, zero=None):
    """Grid from the identity outwards: `zero` (if given) then a geometric
    sequence from lo to hi."""
    g = list(np.geomspace(lo, hi, n))
    return np.array(([zero] if zero is not None else []) + g)


def lin(a, b, n):
    return np.linspace(a, b, n + 1)


# family: function, calibrated feature, {direction: grid from the
# identity (grid[0]) outwards}, discrete parameter?
FAMILIES = {
    "chroma":      (m_chroma, "sat_mean",
                    {"up": np.r_[1.0, geo(1.05, 20.0, GRID_N)],
                     "down": lin(1.0, 0.0, GRID_N)}, False),
    "exposure":    (m_exposure, "lum_mean",
                    {"up": lin(0.0, 6.0, GRID_N), "down": lin(0.0, -4.0, GRID_N)},
                    False),
    "contrast":    (m_contrast, "lum_std",
                    {"up": np.r_[1.0, geo(1.05, 8.0, GRID_N)],
                     "down": lin(1.0, 0.0, GRID_N)}, False),
    "gamma":       (m_gamma, "lum_skew",
                    {"up": np.r_[1.0, geo(1.05, 4.0, GRID_N)],
                     "down": np.r_[1.0, geo(0.95, 0.25, GRID_N)]}, False),
    "temperature": (m_temperature, "cast_b",
                    {"up": lin(0.0, 2.5, GRID_N), "down": lin(0.0, -2.5, GRID_N)},
                    False),
    "lift":        (m_lift, "lum_p05", {"up": lin(0.0, 0.6, GRID_N)}, False),
    "posterise":   (m_posterise, "lum_entropy",
                    {"down": np.unique(np.round(np.geomspace(
                        256, 2, GRID_N + 1)))[::-1]}, True),
    "haze":        (m_haze, "dark_channel_mean",
                    {"up": lin(0.0, 0.95, GRID_N)}, False),
    "blur":        (m_blur, "contrast_local",
                    {"down": geo(0.25, 16.0, GRID_N, zero=0.0)}, False),
    "sharpen":     (m_sharpen, "contrast_local",
                    {"up": geo(0.05, 8.0, GRID_N, zero=0.0)}, False),
    "smooth":      (m_smooth, "fine_texture",
                    {"down": geo(2.0, 150.0, GRID_N, zero=0.0)}, False),
    "grain":       (m_grain, "noise_sigma",
                    {"up": geo(0.5, 60.0, GRID_N, zero=0.0)}, False),
    "hue_compress": (m_hue_compress, "hue_spread",
                     {"down": lin(0.0, 1.0, GRID_N)}, False),
    "vignette":    (m_vignette, "periph_lum", {"down": lin(0.0, 1.0, GRID_N)},
                    False),
    "tunnel_blur": (m_tunnel_blur, "periph_blur",
                    {"up": geo(0.25, 24.0, GRID_N, zero=0.0)}, False),
    "band_blur":   (m_band_blur, "miniature_index",
                    {"up": geo(0.25, 24.0, GRID_N, zero=0.0)}, False),
}
# haze_eq uses haze's calibrated parameters (a matched control)
TIED = {"haze_eq": (m_haze_eq, "haze")}
# Version of each (family, direction) whose manipulation changed since the
# first run; part of the calibration-cache key, so only these are
# recomputed. Version 2 (5 Oct): rebuilt so that they cannot clip.
FAMILY_VERSION = {("chroma", "up"): 2, ("exposure", "up"): 2,
                  ("contrast", "up"): 2, ("temperature", "up"): 2,
                  ("temperature", "down"): 2, ("hue_compress", "down"): 2}
FUNCS = {k: v[0] for k, v in FAMILIES.items()}
FUNCS.update({k: v[0] for k, v in TIED.items()})


def apply(fam, img, p, ctx):
    """(uint8 image, share of NEWLY clipped pixels): pixels with a channel
    that was inside the range (0 < value < 255) and was pushed outside it.
    Channels already at 0 or 255 in the photograph (a blown sky) are not
    counted: they carry no information to lose."""
    if fam is None:
        return img, 0.0
    x = FUNCS[fam](img, p, ctx)
    inside = (img > 0) & (img < 255)
    out = (((x < -0.5) | (x > 255.5)) & inside).any(axis=2).mean()
    return to_u8(x), float(out)


# ============================================================
# Worker processes (images shared by fork)
# ============================================================

W_IMGS, W_CTX, W_BASE = None, None, None


def calib_task(task):
    """(family, direction, grid index, parameter) -> per-image change of
    the calibrated feature and clipped share."""
    fam, dirn, gi, p = task
    feat = FAMILIES[fam][1]
    d, cl = [], []
    for i, img in enumerate(W_IMGS):
        out, c = apply(fam, img, p, W_CTX[i])
        v = measure(out, [feat])[feat]
        d.append(v - W_BASE[i][feat])
        cl.append(c)
    return fam, dirn, gi, np.array(d), np.array(cl)


def to_clip_px(u8):
    """The official preprocessing's resize (torchvision Resize(bicubic) on
    a PIL image = PIL bicubic resize; CenterCrop is a no-op here)."""
    return np.asarray(Image.fromarray(u8).resize((CLIP_PX, CLIP_PX),
                                                 Image.BICUBIC), np.uint8)


def variant_task(args):
    """All variants of one image: features, clipped share, CLIP-sized
    pixels."""
    i, levels = args
    img = W_IMGS[i]
    feats, clips, px = [], [], []
    for fam, p in levels:
        out, c = apply(fam, img, p, W_CTX[i])
        m = measure(out)
        feats.append([m[f] for f in ALL_FEATURES])
        clips.append(c)
        px.append(to_clip_px(out))
    return i, np.array(feats), np.array(clips), np.stack(px)


# ============================================================
# Encoder (as step 03)
# ============================================================

def load_un2clip_state_dict(path):
    import torch
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        raw = raw["model"]
    return {(k[len("visual."):] if k.startswith("visual.") else k): v
            for k, v in raw.items()}


def remap_openai_to_hf(sd):
    d_model = sd["class_embedding"].shape[0]
    hf, unused = {}, []

    def block(rest, v):
        n, sub = rest.split(".", 1)
        p = f"encoder.layers.{n}."
        if sub in ("attn.in_proj_weight", "attn.in_proj_bias"):
            q, k, vv = v.split(d_model, dim=0)
            kind = "weight" if sub.endswith("weight") else "bias"
            return {p + f"self_attn.q_proj.{kind}": q,
                    p + f"self_attn.k_proj.{kind}": k,
                    p + f"self_attn.v_proj.{kind}": vv}
        for a, b in (("attn.out_proj.", "self_attn.out_proj."),
                     ("ln_1.", "layer_norm1."), ("ln_2.", "layer_norm2."),
                     ("mlp.c_fc.", "mlp.fc1."), ("mlp.c_proj.", "mlp.fc2.")):
            if sub.startswith(a):
                return {p + b + sub[len(a):]: v}
        return None

    for k, v in sd.items():
        if k == "proj":
            continue
        if k.startswith("transformer.resblocks."):
            r = block(k[len("transformer.resblocks."):], v)
        elif k.startswith("resblocks."):
            r = block(k[len("resblocks."):], v)
        elif k == "class_embedding":
            r = {"embeddings.class_embedding": v}
        elif k == "positional_embedding":
            r = {"embeddings.position_embedding.weight": v}
        elif k == "conv1.weight":
            r = {"embeddings.patch_embedding.weight": v}
        elif k.startswith("ln_pre."):
            r = {"pre_layrnorm." + k[len("ln_pre."):]: v}
        elif k.startswith("ln_post."):
            r = {"post_layernorm." + k[len("ln_post."):]: v}
        else:
            r = None
        if r is None:
            unused.append(k)
        else:
            hf.update(r)
    if "proj" not in sd:
        raise KeyError("un2CLIP checkpoint has no 'proj' key.")
    return hf, sd["proj"], unused


def load_encoder():
    """Returns (encode_u8, encode_pil, preprocess, info).
    encode_u8: (B, CLIP_PX, CLIP_PX, 3) uint8 already resized -> (B, 768)
    encode_pil: list of PIL images -> (B, 768), official preprocessing."""
    import torch
    from torchvision.transforms import (CenterCrop, Compose,
                                        InterpolationMode, Normalize,
                                        Resize, ToTensor)
    from transformers import CLIPImageProcessor, CLIPModel
    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = ALLOW_TF32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.text_model = None
    sd = load_un2clip_state_dict(UN2CLIP_CKPT_PATH)
    hf_sd, proj, unused = remap_openai_to_hf(sd)
    missing, unexpected = model.vision_model.load_state_dict(hf_sd, strict=False)
    buffers = {n for n, _ in model.vision_model.named_buffers()}
    missing_w = [k for k in missing if k not in buffers]
    if missing_w or unexpected or unused:
        raise SystemExit(f"Encoder: weight loading incomplete ({missing_w[:3]}, "
                         f"{unexpected[:3]}, {unused[:3]}).")
    with torch.no_grad():
        model.visual_projection.weight.copy_(proj.T)
    if not torch.equal(model.visual_projection.weight, proj.T.to(
            model.visual_projection.weight.dtype)):
        raise SystemExit("Encoder: un2CLIP projection not applied.")
    model = model.to(device).eval()
    n_px = proc.crop_size["height"]
    if n_px != CLIP_PX:
        raise SystemExit(f"CLIP_PX = {CLIP_PX} but the model uses {n_px}.")
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
    pre = Compose([Resize(n_px, interpolation=InterpolationMode.BICUBIC),
                   CenterCrop(n_px), lambda im: im.convert("RGB"), ToTensor(),
                   Normalize(tuple(proc.image_mean), tuple(proc.image_std))])

    @torch.inference_mode()
    def run(x):
        pooled = model.vision_model(pixel_values=x.to(device)).pooler_output
        e = model.visual_projection(pooled)
        e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return e.float().cpu().numpy()

    def u8_to_tensor(arr):
        x = torch.from_numpy(np.array(arr, copy=True)).permute(0, 3, 1, 2)
        return (x.float().div(255.0) - mean) / std

    def encode_u8(arr):
        return np.concatenate([run(u8_to_tensor(arr[i:i + BATCH_SIZE]))
                               for i in range(0, len(arr), BATCH_SIZE)])

    def encode_pil(ims):
        return np.concatenate([run(torch.stack([pre(im) for im in
                                                ims[i:i + BATCH_SIZE]]))
                               for i in range(0, len(ims), BATCH_SIZE)])

    def pre_diff(u8_512):
        """Check 3: worker path vs official preprocessing, max |diff|."""
        a = u8_to_tensor(to_clip_px(u8_512)[None])[0]
        b = pre(Image.fromarray(u8_512))
        return float((a - b).abs().max())

    return encode_u8, encode_pil, pre_diff, {"device": device,
                                             "tf32": ALLOW_TF32}


# ============================================================
# Helpers
# ============================================================

def P(*n):
    return os.path.join(OUTPUT_DIR, *n)


def short(a):
    return a.split(" / ")[0]


def ci(a, level=CI_LEVEL, axis=0):
    lo = (1 - level) / 2 * 100
    return (np.nanpercentile(a, lo, axis=axis),
            np.nanpercentile(a, 100 - lo, axis=axis))


# ============================================================
# Main
# ============================================================

def main():
    global W_IMGS, W_CTX, W_BASE
    # OpenCV single-threaded BEFORE any OpenCV work: the workers are forked
    # from this process, and an OpenCV thread pool inherited through fork
    # can deadlock them (parallelism comes from the N_WORKERS processes).
    cv2.setNumThreads(1)
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    checks = {}
    print("=" * 70)
    print("09c_manipulation_sweep.py")
    print("=" * 70)

    fams = list(FAMILIES) + list(TIED)
    if MANIPULATIONS is not None:
        bad = [m for m in MANIPULATIONS if m not in fams]
        if bad:
            raise SystemExit(f"Unknown manipulations: {bad}")
        fams = [m for m in fams if m in MANIPULATIONS]
        for t, (_, src) in TIED.items():
            if t in fams and src not in fams:
                raise SystemExit(f"{t} needs {src} in MANIPULATIONS.")
    cal_fams = [f for f in fams if f in FAMILIES]

    # --- corpus: index, flags, features ---
    idx = pd.read_csv(INDEX_CSV, usecols=["image_id", "category", "path"])
    N = len(idx)
    fl = pd.read_csv(FLAGS_CSV, usecols=["image_id", "is_greyscale"])
    colour = ~fl.set_index("image_id").reindex(np.arange(N))["is_greyscale"] \
        .astype(bool).to_numpy()
    defs = json.load(open(os.path.join(FEATURES_DIR, "feature_definitions.json")))
    if defs["crop"] != CROP:
        raise SystemExit("features_crop columns differ from step 04's list.")
    Fc = np.load(os.path.join(FEATURES_DIR, "features_crop.npy"))
    xdefs = json.load(open(os.path.join(EXTRA_DIR, "extra_definitions.json")))
    xcols = xdefs["crop"]
    Ex = np.load(os.path.join(EXTRA_DIR, "extra_crop.npy"))
    if Fc.shape[0] != N or Ex.shape[0] != N:
        raise SystemExit("Feature files do not match the image index.")
    corpus = {f: Fc[:, CROP.index(f)] for f in CROP}
    corpus.update({f: Ex[:, xcols.index(f)] for f in NEW})
    sel = json.load(open(SELECTION_JSON))
    feats_sel = sel["sets"][sel.get("primary", "colour_only")]
    stats = {}
    for f in ALL_FEATURES:
        x = corpus[f][colour]
        x = x[np.isfinite(x)]
        q = np.percentile(x, [1, 10, 25, 50, 75, 90, 99])
        stats[f] = {"q": dict(zip([1, 10, 25, 50, 75, 90, 99], q)),
                    "sd": float(x.std())}
    pd.DataFrame([{"feature": f, "sd": s["sd"],
                   **{f"p{k:02d}": v for k, v in s["q"].items()}}
                  for f, s in stats.items()]).to_csv(
        P("corpus_feature_stats.csv"), index=False, float_format="%.5g")

    # --- choose the images ---
    side = Ex[:, xcols.index("native_side")] if "native_side" in xcols \
        else np.full(N, np.inf)
    ok = colour & (side >= MIN_SIDE)
    for f in feats_sel + [FAMILIES[m][1] for m in cal_fams]:
        ok &= np.isfinite(corpus[f])
    dup_ids = set()
    if DUP_CSV and os.path.exists(DUP_CSV):
        dup_ids = set(pd.read_csv(DUP_CSV)["image_id"].astype(int))
    else:
        print(f"  WARNING: {DUP_CSV} not found; duplicates not excluded.")
    ok &= ~np.isin(np.arange(N), list(dup_ids))
    Emb = np.load(EMB_PATH, mmap_mode="r")
    cats = idx["category"].to_numpy()
    cand = pd.Series(np.where(ok)[0]).groupby(cats[ok])
    cat_list = list(cand.groups)
    order = rng.permutation(len(cat_list))
    chosen = []
    pools = {c: list(rng.permutation(cand.get_group(c).to_numpy()))
             for c in cat_list}
    while len(chosen) < N_IMAGES and any(pools.values()):
        for k in order:
            c = cat_list[k]
            while pools[c]:
                i = int(pools[c].pop())
                if np.isfinite(np.asarray(Emb[i])).all() and \
                        np.abs(np.asarray(Emb[i])).sum() > 0:
                    chosen.append(i)
                    break
            if len(chosen) >= N_IMAGES:
                break
    ids = np.array(chosen)
    nI = len(ids)
    print(f"Images: {nI} ({len(set(cats[ids]))} categories; {ok.sum():,} "
          f"eligible; {len(dup_ids):,} duplicate-group images excluded)")
    if nI < N_IMAGES:
        print(f"  WARNING: only {nI} eligible images found.")

    imgs, ctx, originals = [], [], []
    for n_, i in enumerate(ids):
        with Image.open(idx["path"].iat[i]) as im:
            im = im.convert("RGB")
            originals.append(im.copy())
            sq = np.asarray(central_square(im, IMAGE_SIZE), np.uint8)
        imgs.append(sq)
        g = cv2.cvtColor(cv2.cvtColor(sq, cv2.COLOR_RGB2BGR),
                         cv2.COLOR_BGR2GRAY)
        thr = np.percentile(g, 100 - HAZE_AIRLIGHT_TOP)
        air = sq[g >= thr].reshape(-1, 3).mean(0)
        lab0 = lab_of(sq)
        cm = cmax_of(lab0[:, :, 0], np.arctan2(lab0[:, :, 2], lab0[:, :, 1]))
        ctx.append({"airlight": air.tolist(), "cmax": cm,
                    "lum_mean_512": mean_grey(sq),
                    "seed": RANDOM_SEED + 7919 * n_})
    pd.DataFrame({"image_id": ids, "category": cats[ids],
                  "path": idx["path"].to_numpy()[ids],
                  "airlight": [c["airlight"] for c in ctx]}) \
        .to_csv(P("sweep_images.csv"), index=False)

    # --- check 2: the copied feature code ---
    print("Check 2: feature code against the stored features...")
    worst = 0.0
    for n_ in range(min(FEATURE_CHECK_N, nI)):
        i = ids[n_]
        sq = np.asarray(central_square(originals[n_], FEATURE_SIZE), np.uint8)
        m = measure(sq)
        for f in ALL_FEATURES:
            a, b = m[f], corpus[f][i]
            if np.isnan(a) and np.isnan(b):
                continue
            rel = abs(a - b) / max(abs(b), 1e-6)
            worst = max(worst, rel if np.isfinite(rel) else np.inf)
    checks["C2_features_max_rel_diff"] = worst
    print(f"  max relative difference {worst:.2e} -> "
          f"{'PASS' if worst <= FEATURE_CHECK_RTOL else 'FAIL'}")
    if worst > FEATURE_CHECK_RTOL:
        raise SystemExit("Check 2 failed: the feature code differs from 04/04c.")

    # --- check 5: the unclipped Lab -> sRGB conversion ---
    d5 = max(float(np.abs(rgb_of_f(lab_of(im)) - cv2.cvtColor(
        lab_of(im), cv2.COLOR_Lab2RGB) * 255.0).max()) for im in imgs[:8])
    checks["C5_lab_max_diff"] = d5
    print(f"Check 5: Lab -> sRGB vs OpenCV, max |diff| {d5:.3f} -> "
          f"{'PASS' if d5 < 1.0 else 'FAIL'}")
    if d5 >= 1.0:
        raise SystemExit("Check 5 failed.")

    # --- level 0 features ---
    base = [measure(im) for im in imgs]
    W_IMGS, W_CTX, W_BASE = imgs, ctx, base
    # One pool for the whole run, forked BEFORE PyTorch / CUDA are loaded
    # (forking a process that holds a CUDA context or OpenMP threads can
    # hang the workers). The workers never import torch.
    pool = mp.get_context("fork").Pool(N_WORKERS)
    try:
        run_sweep(pool, locals())
    finally:
        pool.terminate()
        pool.join()


def run_sweep(pool, env):
    """Everything from the calibration on (split from main() so that the
    worker pool is created first)."""
    globals_needed = ("t0", "rng", "checks", "fams", "cal_fams", "idx", "N",
                      "colour", "corpus", "feats_sel", "stats", "ids", "nI",
                      "imgs", "ctx", "cats", "Emb")
    (t0, rng, checks, fams, cal_fams, idx, N, colour, corpus, feats_sel,
     stats, ids, nI, imgs, ctx, cats, Emb) = (env[k] for k in globals_needed)
    originals = env["originals"]

    # --- targets per family and direction ---
    def targets(feat, dirn):
        q = stats[feat]["q"]
        if dirn == "up":
            nat = [(f"p{p}", q[p] - q[50]) for p in UP_PERCENTILES]
            D = q[99] - q[50]
        else:
            nat = [(f"p{p:02d}", q[p] - q[50]) for p in DOWN_PERCENTILES]
            D = q[1] - q[50]
        return nat + [(f"{k}x", k * D) for k in EXTENDED]

    # --- calibration: grid evaluations (cached), then the levels ---
    # The cache holds every grid point already evaluated on these images
    # (keyed by family, direction and parameter), so changing
    # CLIP_MAX_NEW, the percentiles or a grid re-derives the levels
    # without re-running what is already known.
    img_hash = hashlib.sha1(ids.tobytes()).hexdigest()
    base_hash = hashlib.sha1(json.dumps(
        {"images": img_hash, "size": IMAGE_SIZE, "fsize": FEATURE_SIZE,
         "airlight": HAZE_AIRLIGHT_TOP, "seed": RANDOM_SEED}).encode()).hexdigest()
    cache_path = P("calibration_cache.npz")
    cache = {}
    if REUSE_CALIBRATION and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        if str(z["base_hash"]) == base_hash:
            cache = {k: z[k] for k in z.files if k != "base_hash"}
        else:
            print("  calibration cache is for other images or settings: "
                  "not used")

    def ckey(fam, dirn, p):
        v = FAMILY_VERSION.get((fam, dirn), 1)
        return f"{fam}|{dirn}|{p:.10g}" + (f"|v{v}" if v > 1 else "")

    tasks = []
    for fam in cal_fams:
        for dirn, grid in FAMILIES[fam][2].items():
            for gi, p in enumerate(grid[1:], start=1):
                if ckey(fam, dirn, float(p)) not in cache:
                    tasks.append((fam, dirn, gi, float(p)))
    n_all = sum(len(g) - 1 for f in cal_fams for g in FAMILIES[f][2].values())
    print(f"Calibration: {n_all} grid points ({n_all - len(tasks)} cached) x "
          f"{nI} images ({N_WORKERS} workers)...")
    t1 = time.time()
    if tasks:
        for k, (fam, dirn, gi, d, cl) in enumerate(
                pool.imap_unordered(calib_task, tasks)):
            p_ = FAMILIES[fam][2][dirn][gi]
            cache[ckey(fam, dirn, float(p_))] = np.vstack([d, cl])
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(tasks)} ({(time.time() - t1) / 60:.1f} "
                      "min)")
        np.savez(cache_path, base_hash=np.array(base_hash), **cache)
    res = {}
    for fam in cal_fams:
        for dirn, grid in FAMILIES[fam][2].items():
            for gi, p in enumerate(grid[1:], start=1):
                d, cl = cache[ckey(fam, dirn, float(p))]
                res[(fam, dirn, gi)] = (d, cl)
    lev_path = P("levels.csv")
    grid_rows, lev_rows = [], []
    for fam in cal_fams:
        fn, feat, grids, discrete = FAMILIES[fam]
        for dirn, grid in grids.items():
            sgn = 1.0 if dirn == "up" else -1.0
            med = np.r_[0.0, [np.nanmedian(res[(fam, dirn, gi)][0])
                              for gi in range(1, len(grid))]]
            clp = np.r_[0.0, [np.mean(res[(fam, dirn, gi)][1])
                              for gi in range(1, len(grid))]]
            for gi in range(len(grid)):
                grid_rows.append({"family": fam, "direction": dirn,
                                  "param": grid[gi], "median_change": med[gi],
                                  "clipped_new": clp[gi]})
            # positions on the grid as fractional indices: u_clip is
            # where the clipped share reaches CLIP_MAX_NEW (interpolated)
            last = len(grid) - 1
            over = np.where(clp > CLIP_MAX_NEW)[0] \
                if fam not in CLIP_EXEMPT else np.array([], int)
            if len(over):
                j = over[0]
                u_clip = (j - 1) + (CLIP_MAX_NEW - clp[j - 1]) / \
                    max(clp[j] - clp[j - 1], 1e-12)
            else:
                u_clip = float(last)

            def at(u, down=False):
                if discrete:
                    return float(grid[int(np.floor(u)) if down
                                      else int(np.round(u))])
                k0 = min(int(np.floor(u)), last - 1)
                return float(grid[k0] + (u - k0) * (grid[k0 + 1] - grid[k0]))

            # the calibrated feature must still be rising: levels are only
            # placed up to the grid point where its median change peaks
            # (some measures fall again at extreme strengths, e.g. the
            # regional blur measures once the edges disappear)
            peak = int(np.nanargmax(np.where(np.isfinite(med), sgn * med,
                                             -np.inf)))
            u_lim = min(u_clip, float(peak))
            capped_prev, prev_p = False, None
            wrong = not np.nanmax(sgn * med[1:]) > 0
            for lab, T in targets(feat, dirn):
                row = {"family": fam, "direction": dirn, "level": lab,
                       "feature": feat, "target_change": T,
                       "target_sd": T / stats[feat]["sd"]}
                if wrong:
                    lev_rows.append({**row, "param": np.nan,
                                     "status": "dropped (the feature "
                                               "moves the other way)"})
                    continue
                if capped_prev:
                    lev_rows.append({**row, "param": np.nan,
                                     "status": "dropped"})
                    continue
                hit = np.where(sgn * med >= sgn * T)[0]
                hit = hit[(hit > 0) & (hit <= peak)]
                u_t = np.inf
                if len(hit):
                    j = hit[0]
                    a_, b_ = sgn * med[j - 1], sgn * med[j]
                    u_t = (j - 1) + ((sgn * T - a_) / (b_ - a_) if b_ > a_
                                     else 1.0)
                if u_t <= u_lim:
                    pv = at(u_t)
                    if discrete and np.round(u_t) > u_lim:
                        pv = at(u_t, down=True)
                    status = "ok"
                else:
                    capped_prev = True
                    pv = at(u_lim, down=True)
                    if len(over) and u_clip <= peak:
                        status = "capped (clipping)"
                    elif peak < last:
                        status = "capped (feature peaks)"
                    else:
                        status = "capped (out of range)"
                    if np.isclose(pv, grid[0]):
                        lev_rows.append({**row, "param": np.nan,
                                         "status": "dropped (clips at "
                                                   "once)"})
                        continue
                if prev_p is not None and np.isclose(pv, prev_p):
                    lev_rows.append({**row, "param": np.nan,
                                     "status": "dropped (same as the "
                                               "previous level)"})
                    continue
                prev_p = pv
                lev_rows.append({**row, "param": pv, "status": status})
    pd.DataFrame(grid_rows).to_csv(P("calibration_grid.csv"), index=False,
                                   float_format="%.6g")
    levels = pd.DataFrame(lev_rows)
    print(f"  done in {(time.time() - t1) / 60:.1f} min")
    # tied families (haze_eq)
    levels = levels[~levels["family"].isin(list(TIED))]
    for t, (_, src) in TIED.items():
        if t in fams:
            x = levels[levels["family"] == src].copy()
            x["family"] = t
            levels = pd.concat([levels, x], ignore_index=True)
    levels = levels.reset_index(drop=True)
    use = levels[levels["param"].notna()].reset_index(drop=True)
    print(f"Levels: {len(use)} generated "
          f"({(levels['status'] != 'ok').sum()} capped or dropped)")

    # --- encoder and checks 1, 3 ---
    print("Loading the encoder...")
    encode_u8, encode_pil, pre_diff, enc_info = load_encoder()
    print("Check 1: re-encoding the original files...")
    e_orig = encode_pil(originals)
    stored = np.asarray(Emb[ids], dtype=np.float64)
    cos1 = (e_orig * stored).sum(1) / np.linalg.norm(stored, axis=1)
    checks["C1_encoder_min_cos"] = float(cos1.min())
    print(f"  min cosine with the stored embeddings {cos1.min():.6f} -> "
          f"{'PASS' if cos1.min() >= EMB_CHECK_MIN_COS else 'FAIL'}")
    if cos1.min() < EMB_CHECK_MIN_COS:
        raise SystemExit("Check 1 failed: the encoder does not reproduce step 03.")
    d3 = max(pre_diff(imgs[k]) for k in range(min(4, nI)))
    checks["C3_preprocess_max_diff"] = d3
    print(f"Check 3: worker resize vs official preprocessing, max |diff| "
          f"{d3:.2e} -> {'PASS' if d3 == 0 else 'FAIL'}")
    if d3 != 0:
        raise SystemExit("Check 3 failed.")
    del originals

    # --- axes ---
    acat = pd.read_csv(os.path.join(AXES_DIR, "axis_catalog.csv"))
    AV = np.load(os.path.join(AXES_DIR, "axis_vectors.npy")).astype(np.float64)
    nrm = np.linalg.norm(AV, axis=1)
    checks["C4_axes_max_norm_error"] = float(np.abs(nrm - 1).max())
    if np.abs(nrm - 1).max() > 1e-4:
        raise SystemExit("Check 4 failed: axis vectors are not unit length.")
    acat = acat[acat["row"] >= 0].sort_values("row")     # -1: no vector
    if not (acat["row"].to_numpy() == np.arange(len(AV))).all():
        raise SystemExit("axis_catalog rows do not match axis_vectors.npy.")
    keys = list(acat["key"])
    sc = pd.read_csv(os.path.join(SCORES_DIR, "score_catalog.csv"))
    sd_raw = dict(zip(sc["key"], sc["sd_raw"]))
    key_sd = np.array([sd_raw.get(k, np.nan) for k in keys])
    prim = [k for k in keys if k.startswith(f"{PRIMARY_SET}::") and
            bool(acat.set_index("key").at[k, "included"])]
    comp = [k for k in keys if k.startswith(f"{COMPARE_SET}::") and
            bool(acat.set_index("key").at[k, "included"])]
    cents = [k.replace(f"{PRIMARY_SET}::", f"{PRIMARY_SET}@centroid::")
             for k in prim]
    cents = [k for k in cents if k in keys]
    ctrls = [k for k in keys if k.startswith("controls::")]
    rep_keys = prim + cents + ctrls + comp
    kix = {k: i for i, k in enumerate(keys)}

    # --- corpus relation between the primary axes and the controls
    # (for "unique" and "beyond") ---
    print("Corpus relation between the primary axes and the controls "
          "(within categories)...")
    adj_keys = list(dict.fromkeys(k for v in ADJUST.values() for k in v))
    miss = [k for k in adj_keys if k not in kix]
    if miss:
        raise SystemExit(f"ADJUST: {miss} not among the axes.")
    rel_keys = prim + adj_keys
    Z = np.load(os.path.join(SCORES_DIR, "scores_z.npy"), mmap_mode="r")
    zcols = [int(sc.set_index("key").at[k, "col"]) for k in rel_keys]
    codes = pd.Categorical(idx["category"]).codes.astype(int)
    Cn = codes.max() + 1
    cnt = np.zeros(Cn)
    sums = np.zeros((Cn, len(rel_keys)))
    T = np.zeros((len(rel_keys), len(rel_keys)))
    for a0 in range(0, N, 200_000):
        b0 = min(a0 + 200_000, N)
        X = np.asarray(Z[a0:b0], np.float64)[:, zcols]
        m = colour[a0:b0] & np.isfinite(X).all(1)
        X, cd = X[m], codes[a0:b0][m]
        cnt += np.bincount(cd, minlength=Cn)
        np.add.at(sums, cd, X)
        T += X.T @ X
    okc = cnt > 0
    Sw = T - np.einsum("c,ci,cj->ij", 1 / cnt[okc], sums[okc], sums[okc])
    Bu = np.zeros((len(prim), len(prim)))       # row a: coefficients on others
    for a in range(len(prim)):
        o = [j for j in range(len(prim)) if j != a]
        Bu[a, o] = np.linalg.solve(Sw[np.ix_(o, o)], Sw[o, a])
    # beyond: row a = coefficients of axis a on the adjustment controls
    Bc = {}
    for nm, ks in ADJUST.items():
        g = [rel_keys.index(k) for k in ks]
        Bc[nm] = np.linalg.solve(Sw[np.ix_(g, g)], Sw[np.ix_(g, range(len(prim)))]).T
    corr_q = {k: Sw[rel_keys.index(k), :len(prim)] / np.sqrt(
        Sw[rel_keys.index(k), rel_keys.index(k)] * np.diag(Sw)[:len(prim)])
        for k in adj_keys}

    # --- variants: generate, measure, encode ---
    var_list = [(None, 0.0, "level0", "", "")] + \
        [(r.family, float(r.param), r.level, r.direction, r.status)
         for r in use.itertuples()]
    nV = len(var_list)
    print(f"Variants: {nV} per image x {nI} images = {nV * nI:,}...")
    feats_all = np.full((nI, nV, len(ALL_FEATURES)), np.nan)
    clips_all = np.zeros((nI, nV))
    E = np.zeros((nI, nV, 768), np.float32)
    t1 = time.time()
    lv = [(f, p) for f, p, *_ in var_list]
    for k, (i, fe, cl, px) in enumerate(
            pool.imap(variant_task, [(i, lv) for i in range(nI)])):
        feats_all[i], clips_all[i] = fe, cl
        E[i] = encode_u8(px)
        if (k + 1) % 10 == 0 or k + 1 == nI:
            el = (time.time() - t1) / 60
            print(f"  {k + 1}/{nI} images ({el:.1f} min; about "
                  f"{el / (k + 1) * (nI - k - 1):.1f} min left)")
    base_cos = np.array([float(E[i, 0] @ Emb[ids[i]]) /
                         np.linalg.norm(Emb[ids[i]]) for i in range(nI)])

    # --- scores and effects ---
    S = np.einsum("ivd,kd->ivk", E.astype(np.float64), AV)       # raw
    Dz = (S - S[:, :1, :]) / key_sd[None, None, :]              # corpus SD
    Df = feats_all - feats_all[:, :1, :]
    fsd = np.array([stats[f]["sd"] for f in ALL_FEATURES])
    W = np.stack([np.bincount(rng.integers(0, nI, nI), minlength=nI)
                  for _ in range(N_BOOT)]).astype(float) / nI
    pi = [kix[k] for k in prim]
    eff_rows, side_rows = [], []
    lev_ach = {}
    for v, (fam, p, lab, dirn, status) in enumerate(var_list):
        if fam is None:
            continue
        X = Dz[:, v, :]
        mean = X.mean(0)
        bs = W @ np.nan_to_num(X)
        lo, hi = ci(bs)
        U = X[:, pi] - X[:, pi] @ Bu.T                           # unique
        um = U.mean(0)
        ulo, uhi = ci(W @ U)
        bey = {}
        for nm, ks in ADJUST.items():
            Bm = X[:, pi] - X[:, [kix[k] for k in ks]] @ Bc[nm].T
            bl, bh = ci(W @ Bm)
            bey[nm] = (Bm.mean(0), bl, bh)
        dm = mean[pi]
        share = dm ** 2 / max((dm ** 2).sum(), 1e-12)
        tgt = FAMILIES[TIED[fam][1]][1] if fam in TIED else FAMILIES[fam][1]
        tj = ALL_FEATURES.index(tgt)
        mc = float(np.nanmedian(Df[:, v, tj]))
        mc_sd = mc / stats[tgt]["sd"]
        lev_ach[(fam, dirn, lab)] = (mc, float(clips_all[:, v].mean()))
        for k in rep_keys:
            j = kix[k]
            row = {"family": fam, "direction": dirn, "level": lab, "param": p,
                   "status": status, "axis": k, "d": mean[j], "lo": lo[j],
                   "hi": hi[j],
                   "same_sign": float((np.sign(X[:, j]) == np.sign(mean[j])).mean()),
                   "feature_change_sd": mc_sd,
                   "d_per_sd": mean[j] / mc_sd if abs(mc_sd) >= 0.05
                   else np.nan}
            if k in prim:
                a = prim.index(k)
                row.update({"unique": um[a], "unique_lo": ulo[a],
                            "unique_hi": uhi[a], "share": share[a]})
                for nm, (bm_, bl_, bh_) in bey.items():
                    row.update({f"beyond_{nm}": bm_[a],
                                f"beyond_{nm}_lo": bl_[a],
                                f"beyond_{nm}_hi": bh_[a]})
            eff_rows.append(row)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(Df[:, v, :], axis=0) / fsd
        for f, x in zip(ALL_FEATURES, med):
            side_rows.append({"family": fam, "direction": dirn, "level": lab,
                              "feature": f, "median_change_sd": x,
                              "calibrated": f == tgt, "selected": f in feats_sel})
    eff = pd.DataFrame(eff_rows)
    side = pd.DataFrame(side_rows)
    eff.to_csv(P("effects.csv"), index=False, float_format="%.4f")
    side.to_csv(P("side_effects.csv"), index=False, float_format="%.4f")
    levels["achieved_change"] = [lev_ach.get((r.family, r.direction, r.level),
                                             (np.nan, np.nan))[0]
                                 for r in levels.itertuples()]
    levels["clipped_new"] = [lev_ach.get((r.family, r.direction, r.level),
                                         (np.nan, np.nan))[1]
                             for r in levels.itertuples()]
    levels.to_csv(lev_path, index=False, float_format="%.6g")

    # variants table and arrays
    vr = []
    for i in range(nI):
        for v, (fam, p, lab, dirn, status) in enumerate(var_list):
            vr.append({"row": i * nV + v, "image_id": ids[i],
                       "family": fam or "none", "direction": dirn,
                       "level": lab, "param": p, "clipped_new": clips_all[i, v],
                       **dict(zip(ALL_FEATURES, feats_all[i, v]))})
    pd.DataFrame(vr).to_csv(P("variants.csv"), index=False, float_format="%.6g")
    np.save(P("embeddings.npy"), E.reshape(nI * nV, 768))
    np.save(P("variant_scores.npy"), S.reshape(nI * nV, -1).astype(np.float32))
    pd.DataFrame({"col": range(len(keys)), "key": keys, "sd_raw": key_sd}) \
        .to_csv(P("score_keys.csv"), index=False)

    # --- contact sheets ---
    print("Contact sheets...")
    for fam in fams:
        lv_f = [(0.0, "level 0")] + [
            (r.param, f"{r.direction} {r.level}") for r in
            use[use["family"] == fam].itertuples()]
        nr, nc = min(CONTACT_N, nI), len(lv_f)
        fig, axs = plt.subplots(nr, nc, figsize=(1.25 * nc, 1.35 * nr),
                                squeeze=False)
        for a_ in axs.ravel():
            a_.axis("off")
        for r in range(nr):
            for c, (p, lab) in enumerate(lv_f):
                im = imgs[r] if c == 0 else apply(fam, imgs[r], p, ctx[r])[0]
                axs[r, c].imshow(np.asarray(Image.fromarray(im).resize(
                    (CONTACT_PX, CONTACT_PX), Image.BILINEAR)))
                if r == 0:
                    axs[r, c].set_title(f"{lab}\n{p:.3g}", fontsize=5)
        fig.suptitle(f"{fam} (calibrated on "
                     f"{FAMILIES[TIED[fam][1]][1] if fam in TIED else FAMILIES[fam][1]})",
                     fontsize=7, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(P(f"contact_{fam}.png"), dpi=150)
        plt.close(fig)

    # ========================================================
    # Report
    # ========================================================
    L = []
    W_ = L.append
    pl = [short(k.split("::")[1])[:6] for k in prim]
    W_("09c MANIPULATION SWEEP")
    W_("=" * 78)
    W_(f"Images   : {nI} colour photographs, {len(set(cats[ids]))} categories; "
       f"CLIP's central square at {IMAGE_SIZE} px; features at {FEATURE_SIZE} px")
    W_(f"Variants : {nV} per image ({len(use)} levels + level 0), "
       f"{nV * nI:,} in total")
    W_(f"Checks   : encoder min cosine {checks['C1_encoder_min_cos']:.6f}; "
       f"feature code max rel. diff {checks['C2_features_max_rel_diff']:.1e}; "
       f"preprocessing diff {checks['C3_preprocess_max_diff']:.0e}; Lab "
       f"{checks['C5_lab_max_diff']:.3f}; axes unit")
    W_(f"Level 0 vs stored embedding (512 px route): median cosine "
       f"{np.median(base_cos):.4f}, min {base_cos.min():.4f}")
    W_("Effects  : d = mean change over images in corpus SD units of the axis "
       "(06's z),")
    W_("           relative to the same image's level 0; interval from "
       f"{N_BOOT} image resamples.")
    W_("")

    W_("1. LEVELS (target = change of the calibrated feature; achieved = "
       "median over images)")
    W_("-" * 78)
    W_(f"  {'family':<13}{'dir':<5}{'level':<6}{'feature':<18}{'target':>9}"
       f"{'achieved':>10}{'param':>10}{'clip':>7}  status")
    for r in levels.itertuples():
        W_(f"  {r.family:<13}{r.direction:<5}{r.level:<6}{r.feature[:17]:<18}"
           f"{r.target_change:>9.3g}{r.achieved_change:>10.3g}"
           f"{r.param:>10.4g}{r.clipped_new:>7.3f}  {r.status}")
    W_("")

    def natural_top(fam, dirn):
        x = use[(use["family"] == fam) & (use["direction"] == dirn)]
        nat = x[x["level"].str.startswith("p")]
        return (nat.iloc[-1]["level"] if len(nat) else
                (x.iloc[0]["level"] if len(x) else None))

    def ext_top(fam, dirn):
        x = use[(use["family"] == fam) & (use["direction"] == dirn)]
        return x.iloc[-1]["level"] if len(x) else None

    fd = [(f, d) for f in fams for d in (FAMILIES[f][2] if f in FAMILIES
                                         else FAMILIES[TIED[f][1]][2])]

    def table(title, pick):
        W_(title)
        W_("-" * 78)
        W_(f"  {'manipulation':<22}{'lvl':<5}" + "".join(f"{x:>7}" for x in pl) +
           f"{'qual':>7}  top: unique / beyond q / share")
        for fam, dirn in fd:
            lab = pick(fam, dirn)
            if lab is None:
                continue
            x = eff[(eff["family"] == fam) & (eff["direction"] == dirn) &
                    (eff["level"] == lab)].set_index("axis")
            if not len(x):
                continue
            ds = [x.at[k, "d"] for k in prim]
            q = x.at[QUALITY_KEY, "d"] if QUALITY_KEY in x.index else np.nan
            top = prim[int(np.argmax(np.abs(ds)))]
            t_ = x.loc[top]
            flag = "*" if (t_['lo'] > 0 or t_['hi'] < 0) else " "
            W_(f"  {fam + ' ' + dirn:<22}{lab:<5}" +
               "".join(f"{v:>+7.2f}" for v in ds) + f"{q:>+7.2f}  "
               f"{short(top.split('::')[1])[:10]}{flag} {t_['unique']:+.2f} / "
               f"{t_['beyond_q']:+.2f} / {t_['share']:.2f}")
        W_("")

    table("2. EFFECTS AT THE TOP OF THE NATURAL RANGE (p99 or p01, or the "
          "highest level reached)", natural_top)
    W_("   qual = quality control (+ = more like 'a bad photo'); * = interval "
       "excludes 0;")
    W_("   unique = change beyond what the other axes predict; share = "
       "d^2 / sum d^2;")
    W_("   beyond q = change beyond what the quality control's change "
       "predicts (section 3).")
    W_("")

    def beyond_table(title, nm, pick):
        W_(title)
        W_("-" * 78)
        W_(f"  {'manipulation':<22}{'lvl':<5}" + "".join(f"{x:>7}" for x in pl)
           + "   largest")
        for fam, dirn in fd:
            lab = pick(fam, dirn)
            if lab is None:
                continue
            x = eff[(eff["family"] == fam) & (eff["direction"] == dirn) &
                    (eff["level"] == lab)].set_index("axis")
            if not len(x):
                continue
            bs_ = [x.at[k, f"beyond_{nm}"] for k in prim]
            j = int(np.argmax(np.abs(bs_)))
            r_ = x.loc[prim[j]]
            flag = "*" if (r_[f"beyond_{nm}_lo"] > 0 or
                           r_[f"beyond_{nm}_hi"] < 0) else " "
            W_(f"  {fam + ' ' + dirn:<22}{lab:<5}" +
               "".join(f"{v:>+7.2f}" for v in bs_) +
               f"   {short(prim[j].split('::')[1])[:10]}{flag} "
               f"(raw {r_['d']:+.2f})")
        W_("")

    beyond_table("3. BEYOND THE QUALITY CONTROL, TOP OF THE NATURAL RANGE "
                 "(d minus what the change in 'a bad photo' predicts)", "q",
                 natural_top)
    W_("   Prediction from the corpus (within categories, colour images): "
       "within r with quality")
    W_("   " + ", ".join(f"{short(k.split('::')[1])[:8]} {corr_q[QUALITY_KEY][i]:+.2f}"
                         for i, k in enumerate(prim)) + ".")
    W_("   It assumes the manipulation moves an axis and the quality control "
       "in the same ratio")
    W_("   as natural variation does; descriptive, not a causal decomposition.")
    W_("")
    beyond_table("4. BEYOND QUALITY, MOOD AND THREAT, TOP OF THE NATURAL RANGE",
                 "qmt", natural_top)
    table("5. EFFECTS AT THE STRONGEST LEVEL (4x, or the highest reached)",
          ext_top)

    W_("6. DOSE-RESPONSE OF THE MOST AFFECTED AXIS (d at each level)")
    W_("-" * 78)
    for fam, dirn in fd:
        x = eff[(eff["family"] == fam) & (eff["direction"] == dirn) &
                eff["axis"].isin(prim)]
        if not len(x):
            continue
        top = x.loc[x["d"].abs().idxmax(), "axis"]
        y = x[x["axis"] == top]
        W_(f"  {fam + ' ' + dirn:<22}{short(top.split('::')[1])[:12]:<13}" +
           "  ".join(f"{r.level} {r.d:+.2f}" for r in y.itertuples()))
    W_("")

    W_(f"7. SIDE EFFECTS AT THE TOP OF THE NATURAL RANGE (features other than "
       f"the calibrated one with |median change| >= {SIDE_MIN} corpus SD; "
       "selected features marked *)")
    W_("-" * 78)
    for fam, dirn in fd:
        lab = natural_top(fam, dirn)
        x = side[(side["family"] == fam) & (side["direction"] == dirn) &
                 (side["level"] == lab) & ~side["calibrated"]]
        x = x[x["median_change_sd"].abs() >= SIDE_MIN].copy()
        x = x.reindex(x["median_change_sd"].abs().sort_values(
            ascending=False).index)
        W_(f"  {fam + ' ' + dirn:<22}" + (", ".join(
            f"{r.feature}{'*' if r.selected else ''} {r.median_change_sd:+.1f}"
            for r in x.itertuples()) or "none"))
    W_("")

    def other_table(title, ks):
        lab_ = [short(k.split("::")[1])[:6] for k in ks]
        W_(title)
        W_("-" * 78)
        W_(f"  {'manipulation':<22}{'lvl':<5}" + "".join(f"{x:>7}" for x in lab_))
        for fam, dirn in fd:
            lab = natural_top(fam, dirn)
            x = eff[(eff["family"] == fam) & (eff["direction"] == dirn) &
                    (eff["level"] == lab)].set_index("axis")
            if not len(x):
                continue
            W_(f"  {fam + ' ' + dirn:<22}{lab:<5}" + "".join(
                f"{x.at[k, 'd']:>+7.2f}" for k in ks))
        W_("")

    other_table("8. CONTROL AXES AT THE TOP OF THE NATURAL RANGE", ctrls)
    other_table(f"9. POSITIVE POLES ({PRIMARY_SET}@centroid) AT THE TOP OF THE "
                "NATURAL RANGE", cents)
    other_table(f"10. {COMPARE_SET} AXES AT THE TOP OF THE NATURAL RANGE "
                "(continuity with the earlier sweeps)", comp)
    W_(f"Elapsed: {(time.time() - t0) / 60:.1f} min")
    open(P("sweep_report.txt"), "w").write("\n".join(L) + "\n")

    json.dump({"script": "09c_manipulation_sweep.py", "version": 2,
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "calibration_base_hash": base_hash, "n_images": int(nI),
               "n_variants_per_image": nV, "families": fams,
               "checks": checks, "encoder": enc_info,
               "settings": {"image_size": IMAGE_SIZE,
                            "feature_size": FEATURE_SIZE,
                            "min_side": MIN_SIDE, "grid_n": GRID_N,
                            "clip_max_new": CLIP_MAX_NEW,
                            "up": UP_PERCENTILES, "down": DOWN_PERCENTILES,
                            "extended": EXTENDED, "n_boot": N_BOOT,
                            "adjust": ADJUST,
                            "family_version": {f"{a}|{b}": v for (a, b), v
                                               in FAMILY_VERSION.items()},
                            "seed": RANDOM_SEED},
               "runtime_min": (time.time() - t0) / 60},
              open(P("run_meta.json"), "w"), indent=2)
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Report: "
          f"{P('sweep_report.txt')}")


if __name__ == "__main__":
    main()
