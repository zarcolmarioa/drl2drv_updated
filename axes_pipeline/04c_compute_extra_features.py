"""
04c_compute_extra_features.py  (pipeline step 04c)

Measures the NEW candidate features (the manipulations being considered
for the experiment) for every image in image_index.csv, on exactly the
same region as step 04: CLIP's central square, resized in one step to
256 x 256. Step 04 and its outputs are not touched; this step writes its
own folder, keyed by the same image_id.

Version 2 (after the first full run)
------------------------------------
  * periph_blur and miniature_index redefined on a 3 x 3 grid so each
    compares regions at the SAME distance from the centre. In version 1
    they overlapped: the miniature (band) blur moved periph_blur more
    than its own measure, and radial blur moved miniature_index.
  * glow_halo removed: in natural photos it mostly measured how large
    the bright regions are (rho with lum_p95 +0.52) and the glow
    manipulation moved it by only 0.2 SD. Glow stays a sweep
    manipulation, without a corpus measure.
  * check 2 adds a SEPARATION test for the two blur-region features.
  * extra_definitions.json records which features are corpus candidates
    for step 04b and which are measured for the sweep only (see below).
Use a new OUTPUT_DIR; version 1's folder is kept for reference.

Why a separate step
-------------------
Step 04 is run and checked. Rerunning it would change nothing for its 18
features. The new features are added next to it, and step 04b is rerun
later with both files as candidates (a new selection version; the
current 10-feature selection is kept).

The new features (10)
---------------------
Every spatial parameter is set for 256 px and scaled with the image
size, so the same feature can be measured at 336 and 512 px.

Blur at edges (shared machinery)
  A content-independent blur estimate at edge pixels (Zhuo & Sim, 2011,
  "Defocus map estimation from a single image"): the image is blurred
  again with a known Gaussian (REBLUR_SIGMA); at an edge the ratio R of
  the gradient before and after gives the edge's own blur,
  sigma = sigma0 / sqrt(R^2 - 1). Unlike contrast_local, it does not
  confuse smooth content (sky, walls) with blur: it looks only at edges.
  Reported in pixels at 256. Known limit: very strong blur (above about
  4 px at 256) wipes out edges altogether, so the estimate saturates;
  this matters for heavily defocused backgrounds and tilt-shift effects
  stronger than that.
  edge_blur        median edge blur over the whole image
                   (the measure for uniform blur, the reference arm)
  The two regional blur features use a 3 x 3 grid of equal cells:

        +-----+-----+-----+
        |     |  T  |     |     C     centre cell
        +-----+-----+-----+     L, R  left and right cells of the
        |  L  |  C  |  R  |           middle row
        +-----+-----+-----+     T, B  top and bottom cells of the
        |     |  B  |     |           middle column
        +-----+-----+-----+

  T, B, L and R lie at the same distance from the centre (the grid is
  symmetric under a 90-degree turn), so blur that grows with distance
  from the centre affects them equally.
  periph_blur      mean blur of L and R minus C; positive = blurrier at
                   the sides. A horizontal band blur (miniature) leaves
                   the whole middle row sharp, so it does not move this
                                                  [tunnel vision, blur]
  miniature_index  mean blur of T and B minus mean of L and R; positive
                   = blurrier above and below than at the sides. Radial
                   (tunnel) blur moves T, B, L and R alike, so it
                   cancels                          [miniature effect]
  A cell needs MIN_EDGES_REGION_AT_256 edge pixels; if one cell of a
  pair is missing the other is used alone; a missing pair, or a
  missing C, gives NaN.

Light
  periph_lum       log ratio of mean grey in the outer columns (outer
                   PERIPH_COL_OUTER of the width on each side) to the
                   central columns, in each third of the height, then
                   averaged over the thirds. Comparing left/right edges
                   with the centre AT THE SAME HEIGHT removes the sky /
                   ground difference between top and bottom that a
                   radial ring would pick up. Negative = darker edges
                                                  [tunnel vision, dark]

Colour
  hue_spread       1 - |sum of (a*, b*)| / sum of chroma, over pixels
                   with chroma >= CHROMA_MIN (CIELAB). Chroma-weighted
                   circular spread of hue: 0 = one hue, 1 = hues all
                   around the circle. NaN when fewer than
                   CHROMATIC_MIN_FRACTION of the pixels are chromatic
                   (e.g. greyscale photographs)          [hue variety]
  cast_a, cast_b   mean a* and b* of the brightest 5% of non-clipped
                   pixels (white-patch assumption: bright surfaces show
                   the illuminant). A colour cast / white balance shift
                   moves these; the existing lab_a_mean / lab_b_mean
                   also move with the objects' own colours
                                             [colour temperature]

Fine structure
  fine_texture     SD of (grey - bilateral-filtered grey): the
                   low-amplitude texture that edge-preserving smoothing
                   removes, while strong edges are kept
                                           [loss of fine detail]
  noise_sigma      Immerkaer (1996) fast noise estimate (the 3 x 3
                   Laplacian-difference mask), computed only in the
                   flattest NOISE_FLAT_PERCENTILE % of pixels (gradient
                   of the lightly smoothed image) and away from clipped
                   values, so texture is not counted as noise; grey
                   levels                             [grain / noise]
  edge_overshoot   mean over strong edges of (overshoot on the bright
                   side + undershoot on the dark side) / edge contrast,
                   from the grey profile sampled along the gradient
                   direction; the halo that sharpening (unsharp masking)
                   leaves                            [hyper-sharpness]

Auxiliary columns (not candidates; they explain the NaNs and the scale)
  chromatic_fraction   share of pixels with chroma >= CHROMA_MIN
  n_edges_blur         edge pixels used by the blur estimate (per 256 px)
  native_side          shorter side of the ORIGINAL image in pixels

Roles (recorded in extra_definitions.json for step 04b)
-------------------------------------------------------
  corpus candidates  periph_blur, miniature_index, periph_lum,
                     hue_spread, cast_a, cast_b: nearly independent of
                     step 04's features in version 1 (within-category
                     |rho| <= 0.58, most <= 0.25).
  sweep only         edge_blur, fine_texture, noise_sigma,
                     edge_overshoot: in natural photos they belong to
                     the fine-detail family (within-category |rho|
                     0.60-0.75 with contrast_local / edge_density), which
                     contrast_local already represents in the corpus
                     model. They are still measured, for the sweep's
                     calibration and for descriptive reporting.
  The split uses only the features' relations with each other (the
  version 1 preview), never the axes.

Sizes
-----
  256   all images (primary, as step 04)
  336   step 04's subsample (subsample_ids.npy): CLIP's resolution, for
        the 256-vs-336 stability used by 04b
  512   the same subsample: the stimulus resolution. Grain, fine texture
        and sharpening live at the finest scales, which downsampling to
        256 or 336 partly removes; 256 vs 512 shows how much a feature
        depends on the measurement scale. native_side says how many
        subsample images had to be enlarged to reach 512.

Checks
------
  1. SAME AS 04   The index is the one step 04 used (fingerprint in its
                  run_meta.json), step 04's feature settings equal the
                  copies here, and for ALIGN_N random images the 18
                  step-04 features recomputed here (the functions are
                  copied verbatim from 04) match 04's stored values.
                  This confirms the region, the rows and the library
                  versions all agree.
  2. VALIDITY     On VALIDITY_N images (one per random category), each
                  new feature's own manipulation is applied at three
                  strengths (e.g. added grain for noise_sigma, a radial
                  darkening for periph_lum). PASS if the feature moves
                  in the expected direction, more with each strength,
                  and in at least VALIDITY_MIN_SHARE of the images at
                  the strongest level. Images the manipulation hardly
                  changes (mean change below VALIDITY_MIN_TOUCH grey
                  levels at the strongest level) are left out for it.
                  SEPARATION: the miniature (band) blur must not move
                  periph_blur, and the tunnel (radial) blur must not
                  move miniature_index, by more than SEPARATION_MAX
                  times the change in the manipulation's own feature
                  (both in SDs over the check images). The CROSS-TALK table also shows
                  what every manipulation does to every other feature
                  (new and step 04), in SDs of that feature over the
                  check images, e.g. whether grain raises
                  contrast_local. These manipulations are simple
                  reference versions for testing the MEASURES; they are
                  not the stimulus manipulations (those belong to the
                  sweep).
  3. ALIGNMENT    At the end, FINAL_RECHECK_N random images are
                  recomputed and compared with the stored rows.

Inputs
------
  INDEX_CSV       data/selection/image_index.csv (step 02)
  FEATURES_DIR    step 04's folder: run_meta.json,
                  feature_definitions.json, features_crop.npy,
                  status.npy, subsample_ids.npy

Outputs (under OUTPUT_DIR)
--------------------------
  extra_crop.npy / .csv                (N, 13) 10 features + 3 aux
  extra_crop336_subsample.npy / .csv   (n_sub, 13)
  extra_crop512_subsample.npy / .csv   (n_sub, 13)
  subsample_ids.npy                    copy of step 04's
  status.npy                           (N,) 0 not done, 1 ok, -1 unreadable
  extra_definitions.json               names, roles, definitions, settings
  validity_summary.csv                 check 2, one row per manipulation
                                       (and per separation test)
  validity_crosstalk.csv               check 2, manipulation x feature
  failed_images.csv                    unreadable images (if any)
  run_meta.json, extra_report.txt      summary (start here)

The .csv files have an image_id column and are written at the end; the
.npy files are the working arrays (row i = image_id i). A NaN means the
feature is undefined for that image (reason in the aux columns).

Resume
------
Run the script again; it continues where it stopped, and refuses to
resume if the index or the settings changed.

Usage
-----
    Edit SETTINGS, then:
        conda activate drl2drv
        python 04c_compute_extra_features.py

CPU only. Try TEST_LIMIT = 2000 with a separate OUTPUT_DIR first. Per
image the new features cost about as much as step 04's (~17 ms at 256
in testing), so expect a run of similar length. Check 2 adds about a
minute at start-up.
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
FEATURES_DIR        = "data/features"          # step 04's output
OUTPUT_DIR          = "data/features_extra"   # version 1: _extra

# --- Dry run ---
# Process only the first TEST_LIMIT images (None = all). Use a separate
# OUTPUT_DIR for dry runs.
TEST_LIMIT          = None

# --- Geometry ---
FEATURE_SIZE        = 256            # all images (as step 04)
SUBSAMPLE_SIZES     = [336, 512]     # step 04's subsample only

# --- Blur at edges (edge_blur, periph_blur, miniature_index) ---
# REBLUR_SIGMA_AT_256  the known extra blur (px at 256).
# BLUR_DETECT_SIGMA_AT_256  edges are FOUND on a smoothed copy (px at
#                      256) so blurred edges are not lost: on the
#                      original, a blurred edge's gradient is so much
#                      weaker than a sharp one's that it drops out, and
#                      the estimate is left with the sharp edges. The
#                      blur itself is estimated on the original.
# BLUR_CANNY           Canny thresholds on the smoothed copy.
# BLUR_GRAD_MIN        minimum gradient of the smoothed copy at the edge,
#                      grey levels / px.
# BLUR_SIGMA_CAP       largest blur reported (px at 256); edges whose
#                      gradient does not drop on re-blurring get the cap.
# MIN_EDGES_*_AT_256   minimum edge pixels for a value (else NaN).
REBLUR_SIGMA_AT_256 = 1.0
BLUR_DETECT_SIGMA_AT_256 = 1.5
BLUR_CANNY          = (10, 25)
BLUR_GRAD_MIN       = 1.5
BLUR_SIGMA_CAP      = 8.0
MIN_EDGES_IMAGE_AT_256  = 100
MIN_EDGES_REGION_AT_256 = 30       # per grid cell (1/9 of the image)

# --- periph_lum ---
PERIPH_COL_OUTER    = 0.15           # outer share of the width, each side
PERIPH_COL_INNER    = 0.30           # central share of the width

# --- hue_spread ---
CHROMA_MIN          = 5.0            # CIELAB chroma counted as chromatic
CHROMATIC_MIN_FRACTION = 0.05

# --- cast_a, cast_b ---
CAST_TOP_PERCENT    = 5.0            # brightest share of pixels used
CAST_CLIP_RGB       = 250            # pixels with any channel >= this
                                     # are clipped (a*, b* meaningless)
CAST_MIN_NONCLIPPED = 0.10           # share of pixels needed

# --- fine_texture ---
BIL_SIGMA_COLOR     = 15.0           # grey levels
BIL_SIGMA_SPACE_AT_256 = 3.0         # px

# --- noise_sigma ---
NOISE_FLAT_PERCENTILE = 40           # flattest share of pixels used
NOISE_SMOOTH_SIGMA_AT_256 = 1.0      # smoothing before the flatness test
NOISE_CLIP          = (3, 252)       # grey values excluded (clipped)
NOISE_MIN_PIXELS    = 500

# --- edge_overshoot ---
OVERSHOOT_CANNY     = (100, 200)     # strong edges (as 04's edge_density)
OVERSHOOT_T_NEAR_AT_256 = 3          # samples 1..T_NEAR: near the edge
OVERSHOOT_T_OUT_AT_256 = 6           # samples T_NEAR+1..T_OUT: plateau
OVERSHOOT_MIN_CONTRAST = 20.0        # grey levels between the plateaus
OVERSHOOT_MAX_EDGES = 3000           # evenly spaced subset per image
OVERSHOOT_MIN_EDGES = 50

# --- Parallelism ---
N_WORKERS           = 20
CHUNK               = 64
FLUSH_EVERY_TASKS   = 100

# --- Checks ---
ALIGN_N             = 32             # check 1
ALIGN_RTOL          = 1e-4
VALIDITY_N          = 40             # check 2
VALIDITY_MIN_SHARE  = 0.80
VALIDITY_MIN_TOUCH  = 1.0            # grey levels, mean abs change
SEPARATION_MAX      = 0.25           # separation test, see Checks
VALIDITY_FAIL_STOPS = True           # stop if a feature fails check 2
FINAL_RECHECK_N     = 32             # check 3

RANDOM_SEED         = 42


# ============================================================
# Feature names
# ============================================================

NEW = ["edge_blur", "periph_blur", "miniature_index", "periph_lum",
       "hue_spread", "cast_a", "cast_b", "fine_texture", "noise_sigma",
       "edge_overshoot"]
AUX = ["chromatic_fraction", "n_edges_blur", "native_side"]
CORPUS_CANDIDATES = ["periph_blur", "miniature_index", "periph_lum",
                     "hue_spread", "cast_a", "cast_b"]
SWEEP_ONLY = ["edge_blur", "fine_texture", "noise_sigma", "edge_overshoot"]
COLS = NEW + AUX

DEFINITIONS = {
    "edge_blur": "median edge blur (re-blur gradient ratio; px at 256)",
    "periph_blur": "edge blur, mean of left/right grid cells minus the "
                   "centre cell (px at 256)",
    "miniature_index": "edge blur, mean of top/bottom grid cells minus "
                       "mean of left/right cells (px at 256)",
    "periph_lum": "log(mean grey outer columns / central columns), "
                  "averaged over height thirds",
    "hue_spread": "1 - |sum(a*,b*)| / sum(chroma) over chromatic pixels",
    "cast_a": "mean a* of the brightest non-clipped pixels",
    "cast_b": "mean b* of the brightest non-clipped pixels",
    "fine_texture": "SD of grey minus bilateral-filtered grey",
    "noise_sigma": "Immerkaer noise estimate in flat pixels (grey levels)",
    "edge_overshoot": "mean (overshoot + undershoot) / contrast at strong "
                      "edges",
    "chromatic_fraction": "share of pixels with chroma >= CHROMA_MIN",
    "n_edges_blur": "edge pixels used by the blur estimate, per 256 px",
    "native_side": "shorter side of the original image (px)",
}
PHENOMENON = {
    "edge_blur": "uniform blur (reference)",
    "periph_blur": "tunnel vision (blur)",
    "miniature_index": "miniature / tilt-shift",
    "periph_lum": "tunnel vision (darkening)",
    "hue_spread": "hue variety",
    "cast_a": "colour temperature / tint",
    "cast_b": "colour temperature",
    "fine_texture": "loss of fine detail",
    "noise_sigma": "grain / noise",
    "edge_overshoot": "hyper-sharpness",
}


# ============================================================
# Copied verbatim from step 04 (check 1 compares with 04's values)
# ============================================================

CANNY_LOW           = 100
CANNY_HIGH          = 200
SPECTRUM_FMIN       = 8
SPECTRUM_FMAX       = 64
DARK_PATCH_AT_256   = 7

LEGACY = ["lum_mean", "lum_std", "lum_p05", "lum_p50", "lum_p95",
          "sat_mean", "sat_std", "val_mean", "colorfulness",
          "contrast_local", "edge_density"]
EXTRA = ["lum_skew", "lum_entropy", "chroma_mean", "lab_a_mean",
         "lab_b_mean", "spectral_slope", "dark_channel_mean"]
CROP = LEGACY + EXTRA


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


def fingerprint(idx):
    h = hashlib.sha1()
    for k, p in zip(idx["image_key"], idx["path"]):
        h.update(f"{k}\t{p}\n".encode("utf-8"))
    return h.hexdigest()


# ============================================================
# New features
# ============================================================

def grey(rgb_u8):
    """Grey exactly as step 04 (RGB -> BGR -> grey, Rec. 601)."""
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    g8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return g8, g8.astype(np.float32)


def sobel(g):
    """Gradient in grey levels per pixel (Sobel / 8)."""
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return gx, gy, np.sqrt(gx * gx + gy * gy)


def edge_blur_map(g8, g, mag, s):
    """Blur (px at 256) at edge pixels: returns ys, xs, sigma."""
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
    """edge_blur, periph_blur, miniature_index from the edge blur map,
    the last two on a 3 x 3 grid (see the header)."""
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


def extra_features(rgb_u8, native_side=np.nan):
    """The 13 columns (COLS) on a square uint8 RGB array."""
    n = rgb_u8.shape[0]
    s = n / 256.0
    g8, g = grey(rgb_u8)
    gx, gy, mag = sobel(g)
    ys, xs, sig = edge_blur_map(g8, g, mag, s)
    eb, pb, mi = blur_features(ys, xs, sig, n, s)
    hue, ca, cb, cfrac = colour_features(rgb_u8)
    v = {"edge_blur": eb, "periph_blur": pb, "miniature_index": mi,
         "periph_lum": periph_lum(g),
         "hue_spread": hue, "cast_a": ca, "cast_b": cb,
         "fine_texture": fine_texture(g, s),
         "noise_sigma": noise_sigma(g, s),
         "edge_overshoot": edge_overshoot(g8, g, gx, gy, mag, s),
         "chromatic_fraction": cfrac,
         "n_edges_blur": len(sig) / s, "native_side": native_side}
    return [float(v[k]) for k in COLS]


def image_extra(path, in_subsample):
    """(f256, {size: f} for the subsample sizes or {}). Raises on
    unreadable files."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        side = float(min(im.size))
        c = central_square(im, FEATURE_SIZE)
        subs = {n: central_square(im, n) for n in SUBSAMPLE_SIZES} \
            if in_subsample else {}
    f = extra_features(np.asarray(c, dtype=np.uint8), side)
    fs = {n: extra_features(np.asarray(v, dtype=np.uint8), side)
          for n, v in subs.items()}
    return f, fs


# ============================================================
# Reference manipulations for check 2 (not the stimulus versions)
# ============================================================

def _radius(n, corner_one=False):
    y, x = np.indices((n, n), dtype=np.float32)
    r = np.hypot(x + 0.5 - n / 2.0, y + 0.5 - n / 2.0) / (n / 2.0)
    return r / np.sqrt(2.0) if corner_one else r


def _lab(img):
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def _rgb(lab):
    out = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def m_grain(img, v, seed):
    noise = np.random.default_rng(seed).normal(0, v, img.shape[:2])
    return np.clip(img + noise[:, :, None], 0, 255).astype(np.uint8)


def m_blur(img, v, seed):
    return cv2.GaussianBlur(img, (0, 0), v)


def m_unsharp(img, v, seed):
    f = img.astype(np.float32)
    return np.clip(f + v * (f - cv2.GaussianBlur(f, (0, 0), 1.5)),
                   0, 255).astype(np.uint8)


def m_smooth(img, v, seed):
    out = img
    for _ in range(int(v)):
        out = cv2.bilateralFilter(out, -1, 25, 3)
    return out


def m_vignette(img, v, seed):
    w = np.clip(1.0 - v * _radius(img.shape[0], True) ** 2, 0, 1)
    return np.clip(img * w[:, :, None], 0, 255).astype(np.uint8)


def m_periph_blur(img, v, seed):
    f = img.astype(np.float32)
    w = v * np.clip((_radius(img.shape[0]) - 0.4) / 0.5, 0, 1)
    return np.clip((1 - w[:, :, None]) * f + w[:, :, None]
                   * cv2.GaussianBlur(f, (0, 0), 3.0), 0, 255
                   ).astype(np.uint8)


def m_band_blur(img, v, seed):
    n = img.shape[0]
    f = img.astype(np.float32)
    y = np.abs(np.arange(n, dtype=np.float32) + 0.5 - n / 2.0) / (n / 2.0)
    w = np.clip((y - 1.0 / 3.0) / 0.3, 0, 1)[:, None, None]   # middle
    #                                                  third kept sharp
    return np.clip((1 - w) * f + w * cv2.GaussianBlur(f, (0, 0), v),
                   0, 255).astype(np.uint8)


def m_hue_compress(img, v, seed):
    lab = _lab(img)
    a, b = lab[:, :, 1], lab[:, :, 2]
    C = np.sqrt(a * a + b * b)
    chrom = C >= CHROMA_MIN
    if not chrom.any():
        return img
    h0 = np.arctan2(b[chrom].sum(), a[chrom].sum())
    h = np.arctan2(b, a)
    d = np.angle(np.exp(1j * (h0 - h)))
    h2 = h + v * d
    lab[:, :, 1], lab[:, :, 2] = C * np.cos(h2), C * np.sin(h2)
    return _rgb(lab)


def m_warm(img, v, seed):
    lab = _lab(img)
    lab[:, :, 2] += v
    return _rgb(lab)


# (name, function, target feature, expected sign, three strengths)
VALIDITY = [
    ("grain", m_grain, "noise_sigma", +1, [4.0, 8.0, 16.0]),
    ("blur", m_blur, "edge_blur", +1, [0.75, 1.5, 3.0]),
    ("unsharp", m_unsharp, "edge_overshoot", +1, [0.5, 1.0, 2.0]),
    ("smooth", m_smooth, "fine_texture", -1, [1, 2, 4]),
    ("vignette", m_vignette, "periph_lum", -1, [0.2, 0.4, 0.6]),
    ("periph_blur", m_periph_blur, "periph_blur", +1, [0.33, 0.67, 1.0]),
    ("band_blur", m_band_blur, "miniature_index", +1, [1.0, 2.0, 3.0]),
    ("hue_compress", m_hue_compress, "hue_spread", -1, [0.33, 0.67, 1.0]),
    ("warm", m_warm, "cast_b", +1, [5.0, 10.0, 20.0]),
]

# Separation: (manipulation, feature it must NOT move much)
SEPARATION = [("band_blur", "periph_blur"),
              ("periph_blur", "miniature_index")]


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
    out = np.full((n, len(COLS)), np.nan, np.float32)
    sub = {}
    ok = np.zeros(n, dtype=bool)
    for j in range(n):
        try:
            f, fs = image_extra(paths[j], flags[j])
            out[j] = f
            if fs:
                sub[int(ids[j])] = {k: np.asarray(v, np.float32)
                                    for k, v in fs.items()}
            ok[j] = True
        except Exception:
            pass
    return ids, out, sub, ok


# ============================================================
# Arrays and resume
# ============================================================

def settings_dict():
    keys = ["FEATURE_SIZE", "SUBSAMPLE_SIZES", "REBLUR_SIGMA_AT_256",
            "BLUR_DETECT_SIGMA_AT_256", "BLUR_CANNY", "BLUR_GRAD_MIN",
            "BLUR_SIGMA_CAP",
            "MIN_EDGES_IMAGE_AT_256", "MIN_EDGES_REGION_AT_256",
            "PERIPH_COL_OUTER", "PERIPH_COL_INNER",
            "CHROMA_MIN", "CHROMATIC_MIN_FRACTION",
            "CAST_TOP_PERCENT", "CAST_CLIP_RGB", "CAST_MIN_NONCLIPPED",
            "BIL_SIGMA_COLOR", "BIL_SIGMA_SPACE_AT_256",
            "NOISE_FLAT_PERCENTILE", "NOISE_SMOOTH_SIGMA_AT_256",
            "NOISE_CLIP", "NOISE_MIN_PIXELS", "OVERSHOOT_CANNY",
            "OVERSHOOT_T_NEAR_AT_256", "OVERSHOOT_T_OUT_AT_256",
            "OVERSHOOT_MIN_CONTRAST", "OVERSHOOT_MAX_EDGES",
            "OVERSHOOT_MIN_EDGES"]
    g = globals()
    d = {k.lower(): (list(g[k]) if isinstance(g[k], tuple) else g[k])
         for k in keys}
    d["columns"] = COLS
    d["version"] = 2
    return d


def save_status(status):
    tmp = os.path.join(OUTPUT_DIR, "status.tmp.npy")
    np.save(tmp, status)
    os.replace(tmp, os.path.join(OUTPUT_DIR, "status.npy"))


def open_arrays(n, n_sub, meta):
    P = lambda f: os.path.join(OUTPUT_DIR, f)
    fmt = np.lib.format
    names = {size: f"extra_crop{size}_subsample.npy"
             for size in SUBSAMPLE_SIZES}
    if os.path.exists(P("run_meta.json")) and os.path.exists(P("status.npy")):
        with open(P("run_meta.json")) as f:
            old = json.load(f)
        diffs = [k for k in ("index_fingerprint", "n_images", "settings")
                 if old.get(k) != meta.get(k)]
        if diffs:
            raise RuntimeError(
                f"Existing run in {OUTPUT_DIR} has different {diffs}. "
                "Refusing to resume; use a new OUTPUT_DIR or delete it.")
        return (fmt.open_memmap(P("extra_crop.npy"), mode="r+"),
                {k: fmt.open_memmap(P(v), mode="r+")
                 for k, v in names.items()},
                np.load(P("status.npy")), True)
    main = fmt.open_memmap(P("extra_crop.npy"), mode="w+",
                           dtype=np.float32, shape=(n, len(COLS)))
    main[:] = np.nan
    subs = {}
    for k, v in names.items():
        subs[k] = fmt.open_memmap(P(v), mode="w+", dtype=np.float32,
                                  shape=(n_sub, len(COLS)))
        subs[k][:] = np.nan
    status = np.zeros(n, dtype=np.int8)
    save_status(status)
    return main, subs, status, False


# ============================================================
# Checks
# ============================================================

def check_same_as_04(idx_full, n):
    """Check 1. Returns (step-04 subsample ids, 04 features memmap,
    check results)."""
    P4 = lambda f: os.path.join(FEATURES_DIR, f)
    for f in ("run_meta.json", "feature_definitions.json",
              "features_crop.npy", "status.npy", "subsample_ids.npy"):
        if not os.path.exists(P4(f)):
            raise RuntimeError(f"Step 04 file missing: {P4(f)}")
    meta4 = json.load(open(P4("run_meta.json")))
    defs4 = json.load(open(P4("feature_definitions.json")))
    n4 = int(meta4["n_images"])
    if n > n4:
        raise RuntimeError(f"Step 04 covered {n4:,} images; this run "
                           f"needs {n:,}.")
    if fingerprint(idx_full.iloc[:n4]) != meta4["index_fingerprint"]:
        raise RuntimeError("INDEX_CSV is not the index step 04 used.")
    s4 = defs4["settings"]
    mine = {"feature_size": FEATURE_SIZE, "canny": [CANNY_LOW, CANNY_HIGH],
            "spectrum_range": [SPECTRUM_FMIN, SPECTRUM_FMAX],
            "dark_patch_at_256": DARK_PATCH_AT_256, "crop_features": CROP}
    bad = [k for k, v in mine.items() if s4.get(k) != v]
    if bad:
        raise RuntimeError(f"Step 04 settings differ from the copies "
                           f"here: {bad}")
    if 336 in SUBSAMPLE_SIZES and s4.get("check_size") != 336:
        raise RuntimeError("Step 04's subsample size is not 336.")

    F4 = np.load(P4("features_crop.npy"), mmap_mode="r")
    st4 = np.load(P4("status.npy"))
    sub_ids = np.load(P4("subsample_ids.npy"))
    sub_ids = sub_ids[sub_ids < n]

    rng = np.random.default_rng(RANDOM_SEED)
    ok_rows = np.where(st4[:n] == 1)[0]
    pick = rng.choice(ok_rows, size=min(ALIGN_N, len(ok_rows)),
                      replace=False)
    worst, used = 0.0, 0
    paths = idx_full["path"].to_numpy()
    for i in pick:
        with Image.open(paths[i]) as im:
            c = central_square(im.convert("RGB"), FEATURE_SIZE)
        f = np.asarray(all_features(np.asarray(c, dtype=np.uint8)))
        ref = np.asarray(F4[i], np.float64)
        rel = np.abs(f - ref) / np.maximum(np.abs(ref), 1e-3)
        rel = rel[np.isfinite(rel)]
        worst = max(worst, float(rel.max()) if len(rel) else 0.0)
        used += 1
    ok = worst <= ALIGN_RTOL
    print(f"  check 1 (same as 04): index and settings match; {used} "
          f"images, 18 features recomputed, max rel diff {worst:.1e} -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("Recomputed step-04 features do not match the "
                           "stored values (region, rows or library "
                           "versions differ).")
    return sub_ids, F4, {"same_as_04_max_rel_diff": worst,
                         "same_as_04_n": used}


def check_validity(idx):
    """Check 2: each feature against its own reference manipulation."""
    rng = np.random.default_rng(RANDOM_SEED + 1)
    cats = idx["category"].unique()
    cats = rng.permutation(cats)
    imgs = []
    for c in cats:
        if len(imgs) >= VALIDITY_N:
            break
        p = rng.choice(idx.loc[idx["category"] == c, "path"].to_numpy())
        try:
            with Image.open(p) as im:
                imgs.append(np.asarray(central_square(im.convert("RGB"),
                                                      FEATURE_SIZE),
                                       dtype=np.uint8))
        except Exception:
            continue
    names = COLS[:len(NEW)] + CROP

    def feats(img):
        return np.asarray(extra_features(img)[:len(NEW)]
                          + all_features(img), np.float64)

    base = np.stack([feats(im) for im in imgs])
    sd = np.nanstd(base, axis=0)
    rows, cross = [], []
    for name, fn, target, sign, levels in VALIDITY:
        j = names.index(target)
        vals, touch = [], []
        for v in levels:
            out = [fn(im, v, RANDOM_SEED + k) for k, im in enumerate(imgs)]
            vals.append(np.stack([feats(o) for o in out]))
            touch = [float(np.abs(o.astype(np.float32) - im).mean())
                     for o, im in zip(out, imgs)]     # strongest level
        vals = np.stack(vals)                         # (L, n, F)
        d = vals - base[None]                         # change
        touched = np.asarray(touch) >= VALIDITY_MIN_TOUCH
        good = np.isfinite(d[:, :, j]).all(axis=0) & touched
        dj = sign * d[:, good, j]                     # expected > 0
        med = np.median(dj, axis=1) if good.any() else np.full(3, np.nan)
        mono = bool(good.any() and np.all(med > 0)
                    and np.all(np.diff(med) > 0))
        share = float((dj[-1] > 0).mean()) if good.any() else np.nan
        passed = mono and share >= VALIDITY_MIN_SHARE
        rows.append({"manipulation": name, "target": target,
                     "expected": "+" if sign > 0 else "-",
                     "levels": levels, "n_images": int(good.sum()),
                     "median_change_by_level":
                         [round(float(x) * sign, 4) for x in med],
                     "share_expected_at_max": round(share, 3),
                     "monotone": mono, "passed": passed})
        z = (np.nanmedian(d[-1][touched], axis=0) if touched.any()
             else np.full(len(names), np.nan)) / np.where(sd > 0, sd, np.nan)
        cross.append(pd.Series(z, index=names, name=name))
    crosstalk = pd.DataFrame(cross)
    own = {m: t for m, _, t, _, _ in VALIDITY}
    for mname, feat in SEPARATION:
        tz = abs(crosstalk.loc[mname, own[mname]])
        oz = abs(crosstalk.loc[mname, feat])
        ok = bool(np.isfinite(tz) and np.isfinite(oz)
                  and oz <= SEPARATION_MAX * tz)
        rows.append({"manipulation": mname, "target": f"not {feat}",
                     "expected": "0", "levels": "strongest",
                     "n_images": np.nan,
                     "median_change_by_level":
                         f"{feat} {oz:.2f} SD vs {own[mname]} {tz:.2f} SD",
                     "share_expected_at_max": np.nan,
                     "monotone": np.nan, "passed": ok})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUTPUT_DIR, "validity_summary.csv"),
                   index=False)
    crosstalk.to_csv(os.path.join(OUTPUT_DIR, "validity_crosstalk.csv"),
                     float_format="%.3f")
    n_pass = int(summary["passed"].sum())
    print(f"  check 2 (validity, {len(imgs)} images): {n_pass} of "
          f"{len(summary)} tests pass")
    for r in summary.itertuples():
        if r.expected == "0":
            print(f"    {r.manipulation:<13} -> {r.target:<20} "
                  f"{'PASS' if r.passed else 'FAIL'}  "
                  f"{r.median_change_by_level} (max ratio {SEPARATION_MAX})")
            continue
        print(f"    {r.manipulation:<13} -> {r.target:<20} "
              f"{'PASS' if r.passed else 'FAIL'}  median change "
              f"{r.median_change_by_level}  share {r.share_expected_at_max}"
              f"  (n={r.n_images:.0f})")
    if VALIDITY_FAIL_STOPS and n_pass < len(summary):
        raise RuntimeError("A test failed check 2 (see "
                           "validity_summary.csv). Set VALIDITY_FAIL_STOPS "
                           "= False to continue anyway.")
    return summary, crosstalk, len(imgs)


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.setNumThreads(1)
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

    print("=" * 70)
    print("04c_compute_extra_features.py")
    print("=" * 70)
    idx_full = pd.read_csv(INDEX_CSV)
    idx = idx_full.iloc[:TEST_LIMIT].copy() if TEST_LIMIT else idx_full
    if not (idx["image_id"].to_numpy() == np.arange(len(idx))).all():
        raise RuntimeError("image_id must be 0..N-1 in row order.")
    n = len(idx)
    print(f"Index   : {INDEX_CSV}  ({n:,} images"
          + (", TEST_LIMIT)" if TEST_LIMIT else ")"))
    print(f"Step 04 : {FEATURES_DIR}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Workers : {N_WORKERS}")

    print("\nStart-up checks")
    sub_ids, F4, checks = check_same_as_04(idx_full, n)
    sub_row = {int(i): r for r, i in enumerate(sub_ids)}
    in_sub = np.zeros(n, dtype=bool)
    in_sub[sub_ids] = True
    print(f"  subsample (sizes {SUBSAMPLE_SIZES}): {len(sub_ids):,} images "
          "(step 04's)")
    vsum, vcross, n_valid = check_validity(idx)

    meta = {"script": "04c_compute_extra_features.py",
            "index_csv": INDEX_CSV, "features_dir": FEATURES_DIR,
            "index_fingerprint": fingerprint(idx), "n_images": n,
            "settings": settings_dict(), "checks": checks,
            "opencv": cv2.__version__}
    main_arr, subs, status, resumed = open_arrays(n, len(sub_ids), meta)
    if resumed:
        with open(os.path.join(OUTPUT_DIR, "run_meta.json")) as f:
            meta["started"] = json.load(f).get("started")
        print(f"\nResuming: {int((status != 0).sum()):,} of {n:,} done.")
    else:
        meta["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    np.save(os.path.join(OUTPUT_DIR, "subsample_ids.npy"), sub_ids)
    with open(os.path.join(OUTPUT_DIR, "extra_definitions.json"), "w") as f:
        json.dump({"crop": COLS, "features": NEW, "auxiliary": AUX,
                   "corpus_candidates": CORPUS_CANDIDATES,
                   "sweep_only": SWEEP_ONLY, "version": 2,
                   "phenomenon": PHENOMENON, "definitions": DEFINITIONS,
                   "settings": settings_dict()}, f, indent=2)

    # --- Compute ---
    todo = np.where(status == 0)[0]
    paths = idx["path"].to_numpy()
    tasks = [(todo[i:i + CHUNK], paths[todo[i:i + CHUNK]],
              in_sub[todo[i:i + CHUNK]])
             for i in range(0, len(todo), CHUNK)]
    print(f"\nTo process: {len(todo):,} images in {len(tasks):,} tasks")
    t_run = time.time()

    def flush():
        main_arr.flush()
        for a in subs.values():
            a.flush()
        save_status(status)

    if tasks:
        with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
            done_tasks = 0
            try:
                for ids, out, sb, ok in tqdm(
                        pool.imap_unordered(work, tasks),
                        total=len(tasks), desc="features", unit="task"):
                    main_arr[ids] = out
                    for i, d in sb.items():
                        for size, v in d.items():
                            subs[size][sub_row[i]] = v
                    status[ids] = np.where(ok, 1, -1).astype(np.int8)
                    done_tasks += 1
                    if done_tasks % FLUSH_EVERY_TASKS == 0:
                        flush()
            finally:
                flush()
    t_run = time.time() - t_run
    rate = len(todo) / t_run if t_run > 0 and len(todo) else 0.0

    # --- Check 3: alignment ---
    rng = np.random.default_rng(RANDOM_SEED)
    ok_rows = np.where(status == 1)[0]
    pick = rng.choice(ok_rows, size=min(FINAL_RECHECK_N, len(ok_rows)),
                      replace=False)
    ok3 = True
    for i in pick:
        f, fs = image_extra(paths[i], bool(in_sub[i]))
        same = np.allclose(f, main_arr[i], rtol=1e-5, atol=1e-6,
                           equal_nan=True)
        for size, v in fs.items():
            same &= np.allclose(v, subs[size][sub_row[int(i)]], rtol=1e-5,
                                atol=1e-6, equal_nan=True)
        ok3 &= bool(same)
    print(f"\n  check 3 (alignment): {len(pick)} images recomputed -> "
          f"{'PASS' if ok3 else 'FAIL'}")

    # --- CSV outputs ---
    done_all = not (status == 0).any()
    if done_all:
        print("Writing CSV files...")
        pd.DataFrame(np.asarray(main_arr), columns=COLS).assign(
            image_id=idx["image_id"].to_numpy())[["image_id"] + COLS].to_csv(
            os.path.join(OUTPUT_DIR, "extra_crop.csv"), index=False,
            float_format="%.6g")
        for size, a in subs.items():
            pd.DataFrame(np.asarray(a), columns=COLS).assign(
                image_id=sub_ids)[["image_id"] + COLS].to_csv(
                os.path.join(OUTPUT_DIR, f"extra_crop{size}_subsample.csv"),
                index=False, float_format="%.6g")
    failed = np.where(status == -1)[0]
    if len(failed):
        idx.iloc[failed][["image_id", "path"]].to_csv(
            os.path.join(OUTPUT_DIR, "failed_images.csv"), index=False)

    # --- Summaries for the report ---
    ok_mask = status == 1
    X = np.asarray(main_arr)[ok_mask]
    stats = []
    for j, nm in enumerate(COLS):
        v = X[:, j]
        fin = np.isfinite(v)
        stats.append({"feature": nm,
                      "mean": np.nanmean(v) if fin.any() else np.nan,
                      "sd": np.nanstd(v) if fin.any() else np.nan,
                      "p01": np.nanpercentile(v, 1) if fin.any() else np.nan,
                      "p99": np.nanpercentile(v, 99) if fin.any() else np.nan,
                      "pct_nan": 100.0 * (~fin).mean() if len(v) else np.nan})
    stats = pd.DataFrame(stats)

    # stability across sizes (subsample, finite pairs)
    sub_ok = ok_mask[sub_ids]
    base = np.asarray(main_arr)[sub_ids]
    stab = {}
    for size, a in subs.items():
        A = np.asarray(a)
        for j, nm in enumerate(NEW):
            x, y = base[sub_ok, j], A[sub_ok, j]
            m = np.isfinite(x) & np.isfinite(y)
            stab[(nm, size)] = (pd.Series(x[m]).rank().corr(
                pd.Series(y[m]).rank()) if m.sum() > 10 else np.nan)
    native = base[sub_ok, COLS.index("native_side")]
    enlarged = {size: float((native < size).mean()) if len(native) else
                np.nan for size in SUBSAMPLE_SIZES}

    # preview: within-category Spearman with step 04's features
    cats = pd.Categorical(idx["category"].to_numpy()[sub_ids[sub_ok]]).codes
    F4s = np.asarray(F4[sub_ids[sub_ok]], np.float64)
    Xs = base[sub_ok][:, :len(NEW)].astype(np.float64)

    def centred_ranks(M):
        D = pd.DataFrame(M)
        D = D - D.groupby(cats).transform("mean")
        return D.rank()

    closest = {}
    if len(Xs) > 20:
        R4, Rn = centred_ranks(F4s), centred_ranks(Xs)
        for j, nm in enumerate(NEW):
            r = [(CROP[k], Rn[j].corr(R4[k])) for k in range(len(CROP))]
            r = [t for t in r if np.isfinite(t[1])]
            closest[nm] = sorted(r, key=lambda t: -abs(t[1]))[:3]

    meta.update({"finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "n_ok": int(ok_mask.sum()), "n_failed": int(len(failed)),
                 "n_not_done": int((status == 0).sum()),
                 "images_per_second_this_run": round(rate, 1),
                 "validity_pass": int(vsum["passed"].sum()),
                 "alignment_check_pass": ok3})
    with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- Report ---
    L = ["04c_compute_extra_features.py -- summary", "=" * 60,
         f"Started / finished : {meta['started']} / {meta['finished']}",
         f"Index              : {INDEX_CSV}  ({n:,} images)",
         f"Feature size       : {FEATURE_SIZE} (central square, as 04)",
         f"Subsample          : {len(sub_ids):,} images at "
         f"{SUBSAMPLE_SIZES} (step 04's)",
         "", "Checks",
         f"  1 same as 04 : PASS (max rel diff "
         f"{checks['same_as_04_max_rel_diff']:.1e}, "
         f"{checks['same_as_04_n']} images)",
         f"  2 validity   : {int(vsum['passed'].sum())} of {len(vsum)} tests pass "
         f"({n_valid} images)",
         f"  3 alignment  : {'PASS' if ok3 else 'FAIL'}",
         "", "Check 2: each feature against its own manipulation "
         "(median change at the three strengths, sign = expected)"]
    for r in vsum.itertuples():
        if r.expected == "0":
            L.append(f"  {r.manipulation:<13} -> {r.target:<20} "
                     f"{'PASS' if r.passed else 'FAIL'}  "
                     f"{r.median_change_by_level} (separation, max ratio "
                     f"{SEPARATION_MAX})")
            continue
        L.append(f"  {r.manipulation:<13} -> {r.target:<20} "
                 f"{'PASS' if r.passed else 'FAIL'}  "
                 f"{r.median_change_by_level}  share at max "
                 f"{r.share_expected_at_max}  (n={r.n_images:.0f})")
    L += ["", "Check 2 cross-talk: other features moved by >= 0.5 SD at "
          "the strongest level (median change / SD over the check images)"]
    for name, row in vcross.iterrows():
        tgt = vsum[vsum["expected"] != "0"].set_index(
            "manipulation").loc[name, "target"]
        others = row.drop(tgt).dropna()
        others = others[others.abs() >= 0.5].sort_values(key=abs,
                                                         ascending=False)
        txt = ", ".join(f"{k} {v:+.1f}" for k, v in others.items()) \
            or "none"
        L.append(f"  {name:<13} target {tgt} {row[tgt]:+.1f}; others: "
                 f"{txt}")
    L += ["", "Images", f"  ok         : {meta['n_ok']:,}",
          f"  unreadable : {meta['n_failed']:,}",
          f"  not done   : {meta['n_not_done']:,}",
          f"  this run   : {len(todo):,} in {t_run / 3600:.2f} h "
          f"({rate:.1f} img/s)",
          "", "Features at 256 (all ok images; pct_nan = undefined)",
          stats.round(4).to_string(index=False) if len(X) else "  (none)",
          "", "Stability: Spearman between sizes on the subsample "
          "(finite pairs)"]
    hdr = "  " + f"{'feature':<16}" + "".join(f"  256 vs {s:<4}"
                                              for s in SUBSAMPLE_SIZES)
    L.append(hdr)
    for nm in NEW:
        L.append("  " + f"{nm:<16}" + "".join(
            f"  {stab[(nm, s)]:>+10.3f}" for s in SUBSAMPLE_SIZES))
    L.append("  subsample images enlarged to reach the size: "
             + ", ".join(f"{s}: {100 * v:.1f}%" for s, v in
                         enlarged.items()))
    L += ["", "Preview (step 04b decides): the 3 step-04 features most "
          "correlated within category (Spearman, subsample)"]
    for nm in NEW:
        if nm in closest:
            role = "corpus" if nm in CORPUS_CANDIDATES else "sweep only"
            L.append(f"  {nm:<16} [{role:<10}] " + ", ".join(
                f"{k} {v:+.2f}" for k, v in closest[nm]))
    L += ["", f"Total time: {(time.time() - t0) / 3600:.2f} h"]
    if not done_all:
        L.insert(-1, "NOTE: not finished; run again to continue. CSV files "
                     "are written when all images are done.")
    report = "\n".join(L)
    with open(os.path.join(OUTPUT_DIR, "extra_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    if not ok3:
        raise RuntimeError("Alignment check failed.")


if __name__ == "__main__":
    main()
