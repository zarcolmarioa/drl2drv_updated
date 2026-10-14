"""
07_category_confounds_v4.py  (pipeline step 07, v4 run)

v4 run (v4_protocol.md 1.17): the same script as 07_category_confounds.py
in its current configuration (the v3 run's third run, data/confounds_v3:
12 features, feature_selection_v3), with
  * the scores of 06_project_axes_v4.py (data/scores_v4) and the axes of
    05_build_axes_v4.py (data/axes_v4); output in data/confounds_v4;
  * v4 among the main sets. Every v2/v3 result equals the v3 run's: the
    category resamples do not depend on the number of axes, and each axis
    is fitted on its own (the same bootstrap draws, the same estimates).
    CHECK R1: the v2/v3 rows of within_between.csv are compared with
    V3_RESULTS (data/confounds_v3); the report says PASS or lists the
    largest differences;
  * R2b: the v4@nocontrol axes (darkness and enclosure directions
    removed) analysed like the main sets, and reported next to v4 as a
    sensitivity check, never as the result;
  * CONTROL SHARE (decision 13): for every main axis, the share of its
    within-category variance explained by the quality, mood and threat
    control axes together (R squared, with an interval from the same
    category resamples), and its within-category correlation with each
    control. Reported, not used to drop an axis;
  * figures for the v4 axes only (v2/v3 figures are in the v3 run's
    folder); within-category figures rank the categories on the v4 axis
    itself;
  * not rerun (unchanged, in the v3 run's folder): the legacy block and
    bridge, the feature-set bridge, the 04c feature figures.
  Correction (6 Oct): the first v4 run used the settings of the v3 run's
  SECOND run (16 features, feature_selection_v2), including four 04c
  features dropped after their visual check (periph_blur,
  miniature_index, periph_lum, cast_a). That run (data/confounds_v4) is
  superseded by this one.
  Overlap between axes (decision 13: partialled links for pairs with
  r >= 0.5) is computed in step 09, not here.

The central question: WITHIN the same kind of scene, do images that score
higher on a derealization axis differ in their low-level features?

Scene categories differ both in their features and in their axis scores
(beaches are bright and colourful; jail cells are dim and grey). A
correlation over all images mixes two questions:
  between categories  are grey kinds of place more lifeless?
  within categories   among jail cells, is the greyer one more lifeless?
The article's claim rests on the second. This step separates the two.

Version 2 (28 Sep): features from steps 04 and 04c
--------------------------------------------------
  * Features: the selection of 04b version 2 (16 features: version 1's
    10 plus periph_blur, miniature_index, periph_lum, hue_spread, cast_a,
    cast_b from step 04c). The selection file says which step each
    feature comes from; they are read from both folders.
  * Images with an undefined feature are left out of that image set
    (complete cases). hue_spread is undefined for greyscale photographs,
    so the all_images set now loses most of them; the version-1 run
    (10 features) remains the all_images reference. Counts in the report.
  * FEATURE-SET BRIDGE: every axis x feature result of the 10 shared
    features, with 10 features (the version-1 run, PREVIOUS_RESULTS)
    and with 16. within_rho changes only through the images left out;
    within_beta also changes because the new features are held
    constant.
  * FEATURE FIGURES (the visual check that decides whether a new
    feature is kept): for every selected 04c feature, the images at the
    top and bottom of its range, within categories and over the whole
    corpus, shown as the region the feature was measured on (CLIP's
    central square at 256 px) with the regions the feature compares
    drawn on it.
  * Output goes to a new folder; version 1's results are kept.

Decisions (27 Sep; see pipeline_review.md, section 0.4)
-------------------------------------------------------
  Axes      the five main versions (v2, v2_merged, v3a, v3b, v3b_neg) and
            the six control axes; v3b is primary. Caption variants,
            baselines and sub-axes are tested in 09.
  Features  the ones chosen by 04b (CLIP's crop, 256 px); version 2:
            16, see above.
  Images    colour_only (primary) and all_images; categories with at
            least MIN_IMAGES_PER_CATEGORY images.
  Model     WITHIN (fixed effects): every variable minus its category
              mean, then an ordinary regression. Purely within-category
              by construction (review B2).
            BETWEEN: the same regression on the category means (one
              category = one observation). Together with WITHIN this is
              the within/between (Mundlak) decomposition.
            CONSISTENCY: the same statistics inside each category
              separately: share of categories with the same sign, and
              the spread.
            MIXED MODEL on a subsample (colour_only, v2 and v3b), for
              continuity with the old Script B: the old random-intercept
              model and the Mundlak random-intercept model.
  Uncertainty  95% intervals from resampling whole categories
            (N_BOOT times); category-clustered standard errors as a
            cross-check. No p-value verdicts (review B7, B10).

Statistics (per image set, axis and feature)
-------------------------------------------
  eta2              share of the axis's variance between categories
  within_rho        Spearman correlation of the category-centred axis
                    score and category-centred feature (review B4: both
                    centred). Ranks are taken once on the full image set;
                    the bootstrap resamples categories with those ranks.
  within_beta       standardised coefficient of the feature in the joint
                    within model (all selected features together)
  between_rho       Spearman correlation across category means
  between_beta      standardised coefficient in the joint model on the
                    category means
  consistency       per category: share with the same sign as the pooled
                    estimate, and the median and quartiles, for the
                    marginal r and the joint beta
  vif               within-category VIF of each feature (from 04b's rule)

Legacy block (old Script B methods, for comparison only)
--------------------------------------------------------
Reproduces the old analyses on the NEW scores, v2 axes, all images,
categories with >= 30 images, with the old 11 features measured the old
way (STRETCH) and the same 11 measured on CLIP's crop:
  A2  pooled Spearman (feature vs axis score)
  A4  eta2
  A5  mixed model: axis ~ features + (1 | category), features and score
      standardised over all images, REML, lbfgs (review B2, B7, B8)
  A6  "spearman_centered": axis score centred by category, feature NOT
      centred (review B4)
It then builds a BRIDGE table that changes one thing at a time:
  old published number (OLD_RESULTS_DIR)
  -> old method, new encoding, old features  (differs by encoding only)
  -> old method, crop features               (effect of review A3)
  -> new method, v2 axes                     (effect of the method)
  -> new method, v3b axes                    (effect of the anchors)
The old verdict labels need LEACE and are left to step 08.
Everything in the legacy files is labelled "legacy".

Figures (visual check; the article versions are made in step 10)
-----------------------------------------------------------------
For every axis of FIG_SETS (v3b and v2):
  <axis>_global.png  the old style: top and bottom FIG_TOPK images of the
                     whole corpus (all images).
  <axis>_within.png  categories ranked by their mean score on the v3b
                     axis (colour_only): the 2 highest, the 2 nearest the
                     median and the 2 lowest; in each, the FIG_PER_CAT
                     highest and lowest images. The v2 figure uses the
                     categories of the matching v3b axis (staged and
                     simulation use constructed's).
Every image shows CLIP's crop box, its image_id, category, file name,
z-score and key features.
For every selected 04c feature (FEATURE_FIGURES):
  features/<feature>_within.png  categories ranked by their mean value
                     of the feature (colour_only): the 2 highest, the 2
                     nearest the median and the 2 lowest; in each, the
                     FIG_PER_CAT highest and lowest images.
  features/<feature>_global.png  top and bottom FIG_TOPK images of the
                     colour images.
  Each cell shows the measured region itself (CLIP's central square,
  resized to 256 px as in 04/04c, then enlarged for display) with the
  compared regions drawn: the 3 x 3 grid cells for periph_blur (C vs
  L, R) and miniature_index (T, B vs L, R), the outer and central
  columns for periph_lum. The value and its within-category rank are in
  the title. Categories chosen are in feature_figure_categories.csv. SAFEGUARD: each shown image is rescored from
its embedding while plotting; if it differs from the stored score by
more than FIG_SCORE_TOL, the run stops (figure_check.csv). Images are
shown as stored on disk, without EXIF rotation, as CLIP read them.

Inputs
------
  INDEX_CSV       data/selection/image_index.csv        (02)
  EMB_PATH        data/embeddings/embeddings_768.npy    (03, figure check)
  FEATURES_DIR    data/features/                        (04)
  EXTRA_DIR       data/features_extra/                  (04c)
  FLAGS_CSV       data/monochrome/image_flags.csv       (04a)
  SELECTION_JSON  data/feature_selection_v2/selected_features.json (04b)
  PREVIOUS_RESULTS  version 1's within_between.csv (feature-set bridge;
                  None to skip)
  AXES_DIR        data/axes/                            (05, vectors)
  SCORES_DIR      data/scores/                          (06)
  OLD_RESULTS_DIR results/methodological_analysis_un2clip_v2_full
                  (old Script B output; None to skip the published column)

Outputs (under OUTPUT_DIR)
--------------------------
  confound_report.txt           summary (start here)
  eta_squared.csv
  within_between.csv            one row per image set x axis x feature:
                                every statistic above with its interval
  consistency.csv               per-category distributions
  category_means_axes_<set>.csv, category_means_features_<set>.csv
  per_category_<set>.csv        per-category r and beta (long format)
  mixed_subsample.csv           mixed-model check
  legacy_results.csv            old methods on new data
  bridge.csv                    the bridge table (all v2 axes x features)
  feature_set_bridge.csv        10 vs 16 features, every shared result
  figure_categories.csv         the categories chosen for the figures
  figure_check.csv              stored vs recomputed score of every image
  figures/<set>/<axis>_global.png, <axis>_within.png
  figures/features/<feature>_within.png, <feature>_global.png
  feature_figure_categories.csv
  run_meta.json

Usage
-----
    conda activate drl2drv
    python 07_category_confounds_v4.py

Needs statsmodels for the mixed models (pip install statsmodels); without
it those parts are skipped with a note. The legacy mixed model on all
1.1 million images is the slowest part (set LEGACY_MIXED_MAX_PER_CAT to
subsample it if needed).
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

# --- Inputs ---
INDEX_CSV       = "data/selection/image_index.csv"
EMB_PATH        = "data/embeddings/embeddings_768.npy"
FEATURES_DIR    = "data/features"
EXTRA_DIR       = "data/features_extra"        # 04c (version 2)
FLAGS_CSV       = "data/monochrome/image_flags.csv"
SELECTION_JSON  = "data/feature_selection_v3/selected_features.json"   # 12 features (current)
PREVIOUS_RESULTS = None        # feature-set bridge: done in the v3 run
AXES_DIR        = "data/axes_v4"          # 05_build_axes_v4.py
SCORES_DIR      = "data/scores_v4"        # 06_project_axes_v4.py
OLD_RESULTS_DIR = "results/methodological_analysis_un2clip_v2_full"

# --- Output ---
OUTPUT_DIR      = "data/confounds_v4_12f"  # new folder (v3 run: data/confounds_v3;
                                           # data/confounds_v4 = superseded 16-feature run)
V3_RESULTS      = "data/confounds_v3/within_between.csv"   # check R1; None to skip
R1_TOL          = 1.0001e-4    # the v3 run's CSV is rounded to 4 decimals

# --- Scope ---
V4_SET          = "v4"
R2B_SETS        = ["v4@nocontrol"]         # R2b sensitivity version
MAIN_SETS       = ["v2", "v2_merged", "v3a", "v3b", "v3b_neg", V4_SET]
PRIMARY_SET     = "v3b"
CONTROL_SHARE   = ["quality", "mood", "threat"]   # decision 13
IMAGE_SETS      = ["colour_only", "all_images"]     # first = primary
MIN_IMAGES_PER_CATEGORY = 30

# --- Uncertainty ---
N_BOOT          = 1000
CI_LEVEL        = 0.95
RANDOM_SEED     = 42

# --- Mixed-model check (subsample, colour_only) ---
RUN_MIXED_SUBSAMPLE = True
MIXED_SETS      = ["v2", "v3b", V4_SET]
MIXED_PER_CAT   = 200

# --- Legacy block ---
RUN_LEGACY      = False     # unchanged; in the v3 run's folder
LEGACY_MIN_IMAGES_PER_CATEGORY = 30
LEGACY_MIXED_MAX_PER_CAT = None     # None = all images, as the old run
# v2 axis -> v3b axis for the bridge (others keep their name)
V2_TO_V3 = {"world-as-staged / scripted": "constructed / artificial world",
            "simulation / digital world": "constructed / artificial world"}
# pairs shown in the report (all pairs are in the CSV files)
KEY_PAIRS = [("lifeless / deadened", "sat_mean"),
             ("lifeless / deadened", "lum_mean"),
             ("veiled / foggy", "contrast_local"),
             ("veiled / foggy", "lum_mean"),
             ("veiled / foggy", "sat_mean")]
# v4 key pairs, shown next to the v3b pair (axis id, v3b axis, feature)
KEY_PAIRS_V4 = [("A14", "lifeless / deadened", "sat_mean"),
                ("A13", "lifeless / deadened", "sat_mean"),
                ("A14", "lifeless / deadened", "lum_mean"),
                ("A13", "lifeless / deadened", "lum_mean"),
                ("A07", "veiled / foggy", "contrast_local"),
                ("A07", "veiled / foggy", "lum_mean"),
                ("A07", "veiled / foggy", "sat_mean")]

# --- Figures ---
RUN_FIGURES     = True
FIG_SETS        = [V4_SET]       # v3b and v2 figures: in the v3 run's folder
FIG_OWN_RANKING = [V4_SET]       # sets whose within figures rank categories on the axis itself
FIG_TOPK        = 10                # global figure: top and bottom K
FIG_PER_CAT     = 3                 # within figure: top and bottom per category
FIG_MIN_CAT_IMAGES = 300            # categories eligible for the figures
FIG_FEATURES    = ["sat_mean", "lum_mean", "contrast_local"]
FIG_THUMB_PX    = 320
FIG_SCORE_TOL   = 1e-4
# --- Feature figures (visual check of the 04c features) ---
RUN_FEATURE_FIGURES = False     # done in the v3 run
FEATURE_FIGURES = None          # None = every selected 04c feature
FEATURE_FIG_PX  = 256           # measured size (as 04 / 04c)
CLIP_N_PX       = 336
BOX_COLOUR      = "#00e5ff"
INK             = "#1f1f1e"


# ============================================================
# Helpers
# ============================================================

def P(*name):
    return os.path.join(OUTPUT_DIR, *name)


def short(axis):
    return axis.split(" / ")[0]


def safe(name):
    """File-name form of an axis name, e.g. 'veiled / foggy' -> 'veiled_foggy'."""
    out = "".join(c if c.isalnum() else "_" for c in name)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def clip_box(w, h, n=CLIP_N_PX):
    """CLIP's crop in original pixel coordinates (same as step 04)."""
    if w <= h:
        nw, nh = n, int(n * h / w)
    else:
        nw, nh = int(n * w / h), n
    left = int(round((nw - n) / 2.0))
    top = int(round((nh - n) / 2.0))
    sx, sy = nw / w, nh / h
    return (left / sx, top / sy, (left + n) / sx, (top + n) / sy)


def ci(a, level=CI_LEVEL, axis=0):
    lo = (1 - level) / 2 * 100
    return (np.nanpercentile(a, lo, axis=axis),
            np.nanpercentile(a, 100 - lo, axis=axis))


def rank_cols(X):
    out = np.empty(X.shape, dtype=np.float64)
    for j in range(X.shape[1]):
        out[:, j] = rankdata(X[:, j])
    return out


def cat_stats(V, codes, n_cat):
    """Per category: count, mean (C x p), centred cross-products
    (C x p x p)."""
    p = V.shape[1]
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(n_cat + 1))
    n = np.zeros(n_cat)
    m = np.zeros((n_cat, p))
    M = np.zeros((n_cat, p, p))
    for c in range(n_cat):
        rows = order[bounds[c]:bounds[c + 1]]
        X = np.asarray(V[rows], dtype=np.float64)
        n[c] = len(rows)
        if len(rows):
            m[c] = X.mean(0)
            Xc = X - m[c]
            M[c] = Xc.T @ Xc
    return n, m, M


def pooled(n, m, M, w, within):
    """Total centred cross-product matrix of the categories weighted by
    w (bootstrap multiplicities). within=True: every variable centred by
    its own category mean (the between part drops out)."""
    S = np.tensordot(w, M, axes=1)
    if within:
        return S, float((w * n).sum())
    N = float((w * n).sum())
    mu = (w * n) @ m / N
    D = m - mu
    S = S + np.einsum("c,ci,cj->ij", w * n, D, D)
    return S, N


def corr_from(S, F, A):
    d = np.sqrt(np.diag(S))
    return S[np.ix_(F, A)] / np.outer(d[F], d[A])


def joint_from(S, F, A):
    """Standardised coefficients of every axis on all features."""
    Sxx = S[np.ix_(F, F)]
    B = np.linalg.solve(Sxx, S[np.ix_(F, A)])
    sx = np.sqrt(np.diag(Sxx))
    sy = np.sqrt(np.diag(S)[A])
    return B, B * sx[:, None] / sy[None, :]


def between_from(mc, w, F, A):
    """Joint standardised coefficients and correlations across category
    means, each category one observation (multiplicity w)."""
    W = w.sum()
    mu = w @ mc / W
    D = mc - mu
    S = (D * w[:, None]).T @ D
    _, Bs = joint_from(S, F, A)
    return Bs, corr_from(S, F, A)


def vif(S, F):
    d = np.sqrt(np.diag(S)[F])
    R = S[np.ix_(F, F)] / np.outer(d, d)
    return np.diag(np.linalg.inv(R))


def cluster_se(M_w, Sxx, Braw, F, A, sx, sy):
    """Category-clustered SEs of the standardised within coefficients."""
    inv = np.linalg.inv(Sxx)
    G = M_w[:, F][:, :, A] - np.einsum("cfk,ka->cfa", M_w[:, F][:, :, F], Braw)
    C = G.shape[0]
    se = np.zeros(Braw.shape)
    for a in range(len(A)):
        meat = G[:, :, a].T @ G[:, :, a] * C / (C - 1)
        V = inv @ meat @ inv
        se[:, a] = np.sqrt(np.diag(V)) * sx / sy[a]
    return se


def eta_squared(n, m, M, idx):
    grand = n @ m[:, idx] / n.sum()
    ssb = n @ (m[:, idx] - grand) ** 2
    ssw = np.array([M[:, j, j].sum() for j in idx])
    return ssb / (ssb + ssw)


# ============================================================
# Loading
# ============================================================

def load_inputs():
    idx = pd.read_csv(INDEX_CSV, usecols=["image_id", "category", "path",
                                          "filename"])
    n = len(idx)
    if not (idx["image_id"].to_numpy() == np.arange(n)).all():
        raise SystemExit("image_id must be 0..N-1 in row order.")
    flags = pd.read_csv(FLAGS_CSV, usecols=["image_id", "is_greyscale"])
    grey = flags.set_index("image_id").reindex(np.arange(n))["is_greyscale"]
    if grey.isna().any():
        raise SystemExit("image_flags.csv does not cover every image.")
    colour = ~grey.astype(bool).to_numpy()

    defs = json.load(open(os.path.join(FEATURES_DIR, "feature_definitions.json")))
    sel = json.load(open(SELECTION_JSON))
    feats = sel["sets"][sel.get("primary", "colour_only")]
    sources = sel.get("feature_sources", {})
    Fc = np.load(os.path.join(FEATURES_DIR, "features_crop.npy"), mmap_mode="r")
    Fs = np.load(os.path.join(FEATURES_DIR, "features_stretch.npy"), mmap_mode="r")
    crop_names, stretch_names = defs["crop"], defs["stretch"]
    if Fc.shape[0] != n or Fs.shape[0] != n:
        raise SystemExit("Feature arrays do not match the image index.")
    feats_new = [f for f in feats if sources.get(f, "04") == "04c"]
    E, extra_names = None, []
    if feats_new:
        xdefs = json.load(open(os.path.join(EXTRA_DIR,
                                            "extra_definitions.json")))
        m4 = os.path.join(FEATURES_DIR, "run_meta.json")
        m4c = os.path.join(EXTRA_DIR, "run_meta.json")
        if os.path.exists(m4) and os.path.exists(m4c):
            a, b = json.load(open(m4)), json.load(open(m4c))
            if a.get("index_fingerprint") != b.get("index_fingerprint"):
                raise SystemExit("Steps 04 and 04c used different indexes.")
        E = np.load(os.path.join(EXTRA_DIR, "extra_crop.npy"), mmap_mode="r")
        extra_names = xdefs["crop"]
        if E.shape[0] != n:
            raise SystemExit("04c features do not match the image index.")
    for f in feats + FIG_FEATURES:
        where = extra_names if sources.get(f, "04") == "04c" else crop_names
        if f not in where:
            raise SystemExit(f"Feature {f} not found in its step's file.")

    def get_features(names):
        """(N, k) float32 array of the named features, from 04 or 04c."""
        out = np.empty((n, len(names)), np.float32)
        for j, f in enumerate(names):
            if sources.get(f, "04") == "04c":
                out[:, j] = E[:, extra_names.index(f)]
            else:
                out[:, j] = Fc[:, crop_names.index(f)]
        return out

    sc = pd.read_csv(os.path.join(SCORES_DIR, "score_catalog.csv"))
    Z = np.load(os.path.join(SCORES_DIR, "scores_z.npy"), mmap_mode="r")
    R = np.load(os.path.join(SCORES_DIR, "scores_raw.npy"), mmap_mode="r")
    if Z.shape[0] != n:
        raise SystemExit("Scores do not match the image index.")
    return (idx, colour, feats, feats_new, get_features, Fc, Fs, crop_names,
            stretch_names, sc, Z, R)


# ============================================================
# New methods, one image set
# ============================================================

def control_share(S, A, ctl):
    """Within-category R squared of every axis on the control axes ctl
    (indices into S), from the pooled within cross-products S."""
    out = np.full(len(A), np.nan)
    if not ctl:
        return out
    Scc = S[np.ix_(ctl, ctl)]
    for j, a in enumerate(A):
        if a in ctl:
            continue
        s = S[ctl, a]
        out[j] = float(s @ np.linalg.solve(Scc, s) / S[a, a])
    return out


def analyse(label, keep, codes_all, cat_names, Fsel, Zsel, feats, axes_df, rng):
    """Returns (rows, cons_rows, percat_df, eta_rows, cat_means, cs_rows)."""
    codes = codes_all[keep]
    n_cat_all = len(cat_names)
    counts = np.bincount(codes, minlength=n_cat_all)
    ok_cat = counts >= MIN_IMAGES_PER_CATEGORY
    rows_ok = ok_cat[codes]
    codes = codes[rows_ok]
    remap = -np.ones(n_cat_all, int)
    remap[np.where(ok_cat)[0]] = np.arange(ok_cat.sum())
    codes = remap[codes]
    names = np.array(cat_names)[ok_cat]
    C = len(names)

    V = np.hstack([Fsel[keep][rows_ok], Zsel[keep][rows_ok]])
    good = np.isfinite(V).all(1)
    V, codes = V[good], codes[good]
    nf, na = len(feats), axes_df.shape[0]
    F, A = np.arange(nf), np.arange(nf, nf + na)
    n_undef = int((~good).sum())
    print(f"  [{label}] {len(V):,} images, {C} categories "
          f"({n_undef:,} left out: an undefined value)")

    # raw values: per-category stats (means, within cross-products)
    n, m, M = cat_stats(V, codes, C)
    # within-centred ranks (review B4)
    Vw = V - m[codes]
    Rw = rank_cols(Vw)
    del Vw
    _, mr, Mr = cat_stats(Rw, codes, C)
    del Rw

    w1 = np.ones(C)
    S_w, N = pooled(n, m, M, w1, within=True)
    Sr, _ = pooled(n, mr, Mr, w1, within=False)
    rho_w = corr_from(Sr, F, A)
    Braw, Bw = joint_from(S_w, F, A)
    Bb, rho_b_lin = between_from(m, w1, F, A)
    # Spearman across category means
    mrank = rank_cols(m)
    _, rho_b = between_from(mrank, w1, F, A)
    vifs = vif(S_w, F)
    sx = np.sqrt(np.diag(S_w)[F])
    sy = np.sqrt(np.diag(S_w)[A])
    se_cl = cluster_se(M, S_w[np.ix_(F, F)], Braw, F, A, sx, sy)
    eta = eta_squared(n, m, M, list(A))
    eta_f = eta_squared(n, m, M, list(F))
    ctl = [nf + j for j, ax in enumerate(axes_df.itertuples())
           if ax.kind == "control" and ax.axis in CONTROL_SHARE]
    cshare = control_share(S_w, A, ctl)
    cs_boot = np.zeros((N_BOOT, na))

    # bootstrap over categories
    bt = {k: np.zeros((N_BOOT, nf, na)) for k in
          ("rho_w", "beta_w", "beta_b", "rho_b")}
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, C, C), minlength=C).astype(float)
        Sb, _ = pooled(n, m, M, w, within=True)
        Srb, _ = pooled(n, mr, Mr, w, within=False)
        bt["rho_w"][b] = corr_from(Srb, F, A)
        bt["beta_w"][b] = joint_from(Sb, F, A)[1]
        bt["beta_b"][b] = between_from(m, w, F, A)[0]
        bt["rho_b"][b] = between_from(mrank, w, F, A)[1]
        cs_boot[b] = control_share(Sb, A, ctl)
    cis = {k: ci(v) for k, v in bt.items()}
    with warnings.catch_warnings():     # control axes themselves: all NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        cs_lo, cs_hi = ci(cs_boot)
    dA = np.sqrt(np.diag(S_w))
    cs_rows = []
    for j, ax in enumerate(axes_df.itertuples()):
        if ax.kind == "control":
            continue
        row = {"image_set": label, "set": ax.set, "kind": ax.kind, "axis": ax.axis,
               "control_share": cshare[j], "control_share_lo": cs_lo[j],
               "control_share_hi": cs_hi[j]}
        for c in ctl:
            cname = axes_df["axis"].iloc[c - nf]
            row[f"r_{cname}"] = float(S_w[A[j], c] / (dA[A[j]] * dA[c]))
        cs_rows.append(row)

    # consistency across categories
    pc_rows = []
    r_c = np.full((C, nf, na), np.nan)
    b_c = np.full((C, nf, na), np.nan)
    for c in range(C):
        Mc = M[c]
        d = np.sqrt(np.diag(Mc))
        if (d[F] == 0).any() or (d[A] == 0).any() or n[c] < MIN_IMAGES_PER_CATEGORY:
            continue
        r_c[c] = Mc[np.ix_(F, A)] / np.outer(d[F], d[A])
        try:
            b_c[c] = joint_from(Mc, F, A)[1]
        except np.linalg.LinAlgError:
            pass

    rows, cons = [], []
    for j, ax in enumerate(axes_df.itertuples()):
        for i, f in enumerate(feats):
            rc, bc = r_c[:, i, j], b_c[:, i, j]
            same_r = np.nanmean(np.sign(rc) == np.sign(rho_w[i, j]))
            same_b = np.nanmean(np.sign(bc) == np.sign(Bw[i, j]))
            rows.append({
                "image_set": label, "set": ax.set, "kind": ax.kind,
                "axis": ax.axis, "feature": f, "n_images": int(N),
                "n_categories": C,
                "within_rho": rho_w[i, j],
                "within_rho_lo": cis["rho_w"][0][i, j],
                "within_rho_hi": cis["rho_w"][1][i, j],
                "within_beta": Bw[i, j],
                "within_beta_lo": cis["beta_w"][0][i, j],
                "within_beta_hi": cis["beta_w"][1][i, j],
                "within_beta_se_cluster": se_cl[i, j],
                "between_rho": rho_b[i, j],
                "between_rho_lo": cis["rho_b"][0][i, j],
                "between_rho_hi": cis["rho_b"][1][i, j],
                "between_beta": Bb[i, j],
                "between_beta_lo": cis["beta_b"][0][i, j],
                "between_beta_hi": cis["beta_b"][1][i, j],
                "consistency_r_same_sign": same_r,
                "consistency_beta_same_sign": same_b,
                "vif_within": vifs[i],
            })
            cons.append({
                "image_set": label, "set": ax.set, "axis": ax.axis,
                "feature": f, "r_median": np.nanmedian(rc),
                "r_q25": np.nanpercentile(rc, 25),
                "r_q75": np.nanpercentile(rc, 75), "r_same_sign": same_r,
                "beta_median": np.nanmedian(bc),
                "beta_q25": np.nanpercentile(bc, 25),
                "beta_q75": np.nanpercentile(bc, 75), "beta_same_sign": same_b})
    # per-category long table (primary set axes only, to keep it small)
    prim = [j for j, s in enumerate(axes_df["set"]) if s == PRIMARY_SET]
    for c in range(C):
        for j in prim:
            for i, f in enumerate(feats):
                pc_rows.append({"category": names[c], "n": int(n[c]),
                                "axis": axes_df["axis"].iloc[j], "feature": f,
                                "r": r_c[c, i, j], "beta": b_c[c, i, j]})
    eta_rows = [{"image_set": label, "set": ax.set, "kind": ax.kind,
                 "axis": ax.axis, "eta2": eta[j]}
                for j, ax in enumerate(axes_df.itertuples())]
    eta_rows += [{"image_set": label, "set": "feature", "kind": "feature",
                  "axis": f, "eta2": eta_f[i]} for i, f in enumerate(feats)]
    means_ax = pd.DataFrame(m[:, A], columns=axes_df["key"])
    means_ax.insert(0, "n_images", n.astype(int))
    means_ax.insert(0, "category", names)
    means_f = pd.DataFrame(m[:, F], columns=feats)
    means_f.insert(0, "n_images", n.astype(int))
    means_f.insert(0, "category", names)
    for rw in rows:
        rw["n_left_out_undefined"] = n_undef
    return rows, cons, pd.DataFrame(pc_rows), eta_rows, (means_ax, means_f), cs_rows


# ============================================================
# Mixed models
# ============================================================

def mixed_subsample(keep, codes, cat_names, Fsel, Zsel, feats, axes_df, rng):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  statsmodels not installed: mixed-model check skipped.")
        return pd.DataFrame()
    ids = np.where(keep)[0]
    pick = []
    for c in np.unique(codes[ids]):
        cand = ids[codes[ids] == c]
        pick.append(rng.choice(cand, min(MIXED_PER_CAT, len(cand)), replace=False))
    pick = np.sort(np.concatenate(pick))
    X = np.asarray(Fsel[pick], dtype=np.float64)
    ok = np.isfinite(X).all(1) & np.isfinite(np.asarray(Zsel[pick])).all(1)
    pick, X = pick[ok], X[ok]
    cat = np.array(cat_names)[codes[pick]]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(len(feats))])
    df["cat"] = cat
    zf = (df.iloc[:, :len(feats)] - df.iloc[:, :len(feats)].mean()) / \
        df.iloc[:, :len(feats)].std()
    wf = df.groupby("cat")[list(df.columns[:len(feats)])].transform("mean")
    within = (df.iloc[:, :len(feats)] - wf)
    within = within / within.std()
    between = (wf - wf.mean()) / wf.std()
    out = []
    print(f"  mixed-model check: {len(pick):,} images "
          f"({MIXED_PER_CAT} per category), {axes_df.shape[0]} axes")
    for j, ax in enumerate(axes_df.itertuples()):
        y = np.asarray(Zsel[pick, j], dtype=np.float64)
        d1 = zf.copy()
        d1["y"] = (y - y.mean()) / y.std()
        d1["cat"] = cat
        d2 = pd.concat([within.add_prefix("w_"), between.add_prefix("b_")], axis=1)
        yw = y - pd.Series(y).groupby(cat).transform("mean").to_numpy()
        d2["y"] = (y - y.mean()) / yw.std()
        d2["cat"] = cat
        for label, d, rhs in (
                ("old_random_intercept", d1, " + ".join(zf.columns)),
                ("mundlak_random_intercept", d2,
                 " + ".join(list(d2.columns[:2 * len(feats)])))):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = smf.mixedlm(f"y ~ {rhs}", d, groups=d["cat"]).fit(
                        reml=True, method="lbfgs", maxiter=200)
                conv = bool(res.converged)
                for i, f in enumerate(feats):
                    key = f"f{i}" if label.startswith("old") else f"w_f{i}"
                    out.append({"model": label, "set": ax.set, "axis": ax.axis,
                                "feature": f, "beta": res.params.get(key, np.nan),
                                "se": res.bse.get(key, np.nan),
                                "converged": conv})
            except Exception as e:
                out.append({"model": label, "set": ax.set, "axis": ax.axis,
                            "feature": "", "beta": np.nan, "se": np.nan,
                            "converged": False, "error": str(e)[:200]})
    return pd.DataFrame(out)


# ============================================================
# Legacy (old Script B) methods
# ============================================================

def legacy_block(codes, cat_names, Fs, Fc, stretch_names, crop_names, Z, sc):
    v2 = sc[(sc["set"] == "v2")].sort_values("col")
    cols = v2["col"].to_numpy()
    counts = np.bincount(codes, minlength=len(cat_names))
    keep = counts[codes] >= LEGACY_MIN_IMAGES_PER_CATEGORY
    rows = []
    fsets = {"stretch": (Fs, stretch_names),
             "crop": (Fc, crop_names)}
    try:
        import statsmodels.formula.api as smf
        have_sm = True
    except ImportError:
        have_sm = False
        print("  statsmodels not installed: legacy mixed model skipped.")
    rng = np.random.default_rng(RANDOM_SEED + 1)
    Y = np.asarray(Z[:, cols], dtype=np.float64)
    for fs_label, (Farr, names) in fsets.items():
        use = [f for f in stretch_names if f in names]
        fi = [names.index(f) for f in use]
        X = np.asarray(Farr[:, fi], dtype=np.float64)
        ok = keep & np.isfinite(X).all(1) & np.isfinite(Y).all(1)
        Xo, Yo, co = X[ok], Y[ok], codes[ok]
        print(f"  legacy [{fs_label}] {ok.sum():,} images, {len(use)} features")
        # A2 pooled Spearman
        rX, rY = rank_cols(Xo), rank_cols(Yo)
        rX -= rX.mean(0)
        rY -= rY.mean(0)
        a2 = (rX.T @ rY) / np.outer(np.linalg.norm(rX, axis=0),
                                   np.linalg.norm(rY, axis=0))
        # A6: axis centred by category, feature not centred
        C = len(cat_names)
        sums = np.zeros((C, Yo.shape[1]))
        np.add.at(sums, co, Yo)
        cnt = np.bincount(co, minlength=C)[:, None]
        Yc = Yo - (sums / np.maximum(cnt, 1))[co]
        rYc = rank_cols(Yc)
        rYc -= rYc.mean(0)
        a6 = (rX.T @ rYc) / np.outer(np.linalg.norm(rX, axis=0),
                                    np.linalg.norm(rYc, axis=0))
        del rX, rY, rYc, Yc
        # A4 eta2 (feature-set independent; computed once)
        # A5 mixed model
        me = np.full((len(use), len(cols)), np.nan)
        conv = [None] * len(cols)
        if have_sm:
            sel = np.arange(len(Xo))
            if LEGACY_MIXED_MAX_PER_CAT:
                sel = np.concatenate([
                    rng.choice(np.where(co == c)[0],
                               min(LEGACY_MIXED_MAX_PER_CAT, (co == c).sum()),
                               replace=False)
                    for c in np.unique(co)])
            Xs = Xo[sel]
            Xs = (Xs - Xs.mean(0)) / Xs.std(0, ddof=1)
            base = pd.DataFrame(Xs, columns=[f"f{i}" for i in range(len(use))])
            base["cat"] = co[sel]
            rhs = " + ".join(base.columns[:len(use)])
            for j in range(len(cols)):
                y = Yo[sel, j]
                d = base.copy()
                d["y"] = (y - y.mean()) / y.std(ddof=1)
                t = time.time()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = smf.mixedlm(f"y ~ {rhs}", d, groups=d["cat"]).fit(
                            reml=True, method="lbfgs", maxiter=200)
                    me[:, j] = [res.params[f"f{i}"] for i in range(len(use))]
                    conv[j] = bool(res.converged)
                except Exception as e:
                    print(f"    mixed model failed for {v2['axis'].iloc[j]}: {e}")
                print(f"    A5 {short(v2['axis'].iloc[j]):<24} "
                      f"{time.time() - t:5.0f}s", flush=True)
        for j, ax in enumerate(v2["axis"]):
            for i, f in enumerate(use):
                rows.append({"features": fs_label, "axis": ax, "feature": f,
                             "legacy_A2_spearman": a2[i, j],
                             "legacy_A6_spearman_centered": a6[i, j],
                             "legacy_A5_mixed_beta": me[i, j],
                             "legacy_A5_converged": conv[j]})
    leg = pd.DataFrame(rows)
    # A4 eta2 on all images (categories >= min)
    Yk, ck = Y[keep], codes[keep]
    eta = {}
    for j, ax in enumerate(v2["axis"]):
        y = Yk[:, j]
        g = pd.Series(y).groupby(ck)
        ssb = (g.count() * (g.mean() - y.mean()) ** 2).sum()
        eta[ax] = ssb / ((y - y.mean()) ** 2).sum()
    leg["legacy_A4_eta2"] = leg["axis"].map(eta)
    return leg


def read_old(path, cols):
    f = os.path.join(OLD_RESULTS_DIR, path) if OLD_RESULTS_DIR else None
    if not f or not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return d[[c for c in cols if c in d.columns]]


def build_bridge(leg, wb):
    """One row per v2 axis x legacy feature."""
    st = leg[leg["features"] == "stretch"].set_index(["axis", "feature"])
    cr = leg[leg["features"] == "crop"].set_index(["axis", "feature"])
    br = st[["legacy_A2_spearman", "legacy_A6_spearman_centered",
             "legacy_A5_mixed_beta", "legacy_A4_eta2"]].add_suffix("_stretch")
    br = br.join(cr[["legacy_A2_spearman", "legacy_A6_spearman_centered",
                     "legacy_A5_mixed_beta"]].add_suffix("_crop"))
    old2 = read_old("feature_axis_correlation_before_after.csv",
                    ["feature", "axis", "spearman_before"])
    old6 = read_old("feature_axis_correlation_centered.csv",
                    ["feature", "axis", "spearman_centered"])
    old5 = read_old("mixed_effects_results.csv", ["feature", "axis", "beta"])
    old4 = read_old("axis_variance_decomposition.csv", ["axis", "eta_squared"])
    for d, col, new in ((old2, "spearman_before", "old_A2_spearman"),
                        (old6, "spearman_centered", "old_A6_spearman_centered"),
                        (old5, "beta", "old_A5_mixed_beta")):
        if d is not None:
            br = br.join(d.set_index(["axis", "feature"])[col].rename(new))
    br = br.reset_index()
    if old4 is not None:
        br = br.merge(old4.rename(columns={"eta_squared": "old_A4_eta2"}),
                      on="axis", how="left")
    a = wb[wb["image_set"] == "all_images"]
    new_v2 = a[a["set"] == "v2"].set_index(["axis", "feature"])
    new_v3 = a[a["set"] == PRIMARY_SET].set_index(["axis", "feature"])
    br["new_within_rho_v2"] = [new_v2["within_rho"].get((r.axis, r.feature), np.nan)
                               for r in br.itertuples()]
    br["new_within_beta_v2"] = [new_v2["within_beta"].get((r.axis, r.feature), np.nan)
                                for r in br.itertuples()]
    br["v3b_axis"] = [V2_TO_V3.get(x, x) for x in br["axis"]]
    br["new_within_rho_v3b"] = [new_v3["within_rho"].get((r.v3b_axis, r.feature), np.nan)
                                for r in br.itertuples()]
    br["new_within_beta_v3b"] = [new_v3["within_beta"].get((r.v3b_axis, r.feature), np.nan)
                                 for r in br.itertuples()]
    return br


# ============================================================
# Figures
# ============================================================

def load_crop(path, size=FEATURE_FIG_PX):
    """The region the features were measured on: CLIP's central square,
    resized in one step to size x size (as steps 04 and 04c)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        return im.resize((size, size), Image.BILINEAR,
                         box=clip_box(*im.size))


def draw_regions(axp, feature, n):
    """Draw the regions a 04c feature compares."""
    kw = dict(color=BOX_COLOUR, lw=0.9)
    t = n / 3.0
    if feature in ("periph_blur", "miniature_index"):
        for k in (1, 2):
            axp.axhline(k * t - 0.5, **kw)
            axp.axvline(k * t - 0.5, **kw)
        labels = ({"C": (1, 1), "L": (1, 0), "R": (1, 2)}
                  if feature == "periph_blur" else
                  {"T": (0, 1), "B": (2, 1), "L": (1, 0), "R": (1, 2)})
        for lab, (rr, cc) in labels.items():
            axp.text((cc + 0.5) * t, (rr + 0.5) * t, lab, color=BOX_COLOUR,
                     fontsize=9, ha="center", va="center", weight="bold",
                     alpha=0.8)
    elif feature == "periph_lum":
        for x in (0.15, 0.85):
            axp.axvline(x * n - 0.5, **kw)
        for x in (0.35, 0.65):
            axp.axvline(x * n - 0.5, color=BOX_COLOUR, lw=0.9, ls="--")
        for k in (1, 2):
            axp.axhline(k * t - 0.5, color=BOX_COLOUR, lw=0.5, ls=":")


def feature_figures(idx, colour, codes, cat_names, Fvals, feats_new):
    """Visual check of the 04c features (see the header)."""
    out = P("figures", "features")
    os.makedirs(out, exist_ok=True)
    counts_col = np.bincount(codes[colour], minlength=len(cat_names))
    elig = counts_col >= FIG_MIN_CAT_IMAGES
    chosen = []
    for f in feats_new:
        v = Fvals[f]
        fin = colour & np.isfinite(v)
        s = np.bincount(codes[fin], weights=v[fin], minlength=len(cat_names))
        cnt = np.bincount(codes[fin], minlength=len(cat_names))
        mean = np.where(elig & (cnt > 0), s / np.maximum(cnt, 1), np.nan)
        valid = np.isfinite(mean)
        if valid.sum() < 6:
            print(f"    feature figures: {f}: too few eligible categories")
            continue
        order = np.argsort(np.where(valid, mean, np.inf))[:valid.sum()]
        med = np.nanmedian(mean[valid])
        mid = np.argsort(np.abs(np.where(valid, mean, np.inf) - med))[:2]
        cats = [("high", order[-1]), ("high", order[-2]), ("mid", mid[0]),
                ("mid", mid[1]), ("low", order[1]), ("low", order[0])]
        # --- within categories ---
        ncol = 2 * FIG_PER_CAT
        fig, axs = plt.subplots(len(cats), ncol,
                                figsize=(2.3 * ncol, 2.65 * len(cats)))
        for ri, (lvl, c) in enumerate(cats):
            chosen.append({"feature": f, "level": lvl,
                           "category": cat_names[c], "mean": mean[c],
                           "n_colour_only": int(counts_col[c])})
            ids = np.where((codes == c) & fin)[0]
            o = ids[np.argsort(v[ids])]
            hi, lo = o[::-1][:FIG_PER_CAT], o[:FIG_PER_CAT]
            for k, iid in enumerate(list(hi) + list(lo)):
                axp = axs[ri, k]
                rank = int((v[ids] > v[iid]).sum()) + 1
                try:
                    axp.imshow(load_crop(idx["path"].iat[iid]))
                    draw_regions(axp, f, FEATURE_FIG_PX)
                except Exception as e:
                    axp.text(0.5, 0.5, f"cannot open\n{e}", ha="center",
                             va="center", fontsize=6, color="red",
                             transform=axp.transAxes)
                axp.set_xticks([])
                axp.set_yticks([])
                axp.set_title(f"#{iid} {str(idx['filename'].iat[iid])[:18]}\n"
                              f"{f} {v[iid]:+.3g}  (rank {rank}/{len(ids)})",
                              fontsize=6, color=INK, loc="left")
            axs[ri, 0].set_ylabel(f"{lvl}\n{cat_names[c][:20]}\n"
                                  f"mean {mean[c]:+.3g}", fontsize=7,
                                  color=INK)
        fig.suptitle(f"{f} · within category: {FIG_PER_CAT} highest (left) "
                     f"and lowest (right); the measured region (CLIP's "
                     f"central square, {FEATURE_FIG_PX} px) with the "
                     f"compared regions", fontsize=9, color=INK, x=0.01,
                     ha="left")
        fig.tight_layout()
        fig.savefig(P("figures", "features", f"{safe(f)}_within.png"),
                    dpi=150)
        plt.close(fig)
        # --- global (colour images) ---
        ok = fin
        top = np.argsort(np.where(ok, v, -np.inf))[::-1][:FIG_TOPK]
        bot = np.argsort(np.where(ok, v, np.inf))[:FIG_TOPK]
        fig, axs = plt.subplots(2, FIG_TOPK, figsize=(2.1 * FIG_TOPK, 5.0))
        for row, ids in ((0, top), (1, bot)):
            for k, iid in enumerate(ids):
                axp = axs[row, k]
                try:
                    axp.imshow(load_crop(idx["path"].iat[iid]))
                    draw_regions(axp, f, FEATURE_FIG_PX)
                except Exception as e:
                    axp.text(0.5, 0.5, f"cannot open\n{e}", ha="center",
                             va="center", fontsize=6, color="red",
                             transform=axp.transAxes)
                axp.set_xticks([])
                axp.set_yticks([])
                axp.set_title(f"#{iid} {idx['category'].iat[iid][:16]}\n"
                              f"{v[iid]:+.3g}", fontsize=6, color=INK,
                              loc="left")
        axs[0, 0].set_ylabel("highest", fontsize=9, color=INK)
        axs[1, 0].set_ylabel("lowest", fontsize=9, color=INK)
        fig.suptitle(f"{f} · top and bottom {FIG_TOPK} of the colour images "
                     f"(measured region, {FEATURE_FIG_PX} px)", fontsize=10,
                     color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(P("figures", "features", f"{safe(f)}_global.png"),
                    dpi=130)
        plt.close(fig)
        print(f"    feature figures: {f}")
    pd.DataFrame(chosen).to_csv(P("feature_figure_categories.csv"),
                                index=False)
    return pd.DataFrame(chosen)


def feature_set_bridge(wb):
    """Shared features: the 10-feature run (PREVIOUS_RESULTS) vs this
    run."""
    if not PREVIOUS_RESULTS or not os.path.exists(PREVIOUS_RESULTS):
        return pd.DataFrame()
    old = pd.read_csv(PREVIOUS_RESULTS)
    keys = ["image_set", "set", "axis", "feature"]
    cols = ["within_rho", "within_beta", "between_beta", "n_images"]
    m = old[keys + cols].merge(wb[keys + cols], on=keys,
                               suffixes=("_prev", "_now"))
    for c in ("within_rho", "within_beta", "between_beta"):
        m[f"{c}_change"] = m[f"{c}_now"] - m[f"{c}_prev"]
    return m


def load_thumb(path):
    with Image.open(path) as im:
        w, h = im.size
        im = im.convert("RGB")
        box = clip_box(w, h)
        s = FIG_THUMB_PX / max(w, h)
        th = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return th, [v * s for v in box]


def draw_cell(axp, iid, idx, zval, feats_row, title_extra=""):
    path = idx["path"].iat[iid]
    try:
        th, box = load_thumb(path)
        axp.imshow(th)
        x0, y0, x1, y1 = box
        axp.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                edgecolor="black", lw=2.6))
        axp.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                edgecolor=BOX_COLOUR, lw=1.2))
    except Exception as e:
        axp.text(0.5, 0.5, f"cannot open\n{e}", ha="center", va="center",
                 fontsize=6, color="red", transform=axp.transAxes)
    axp.set_xticks([])
    axp.set_yticks([])
    fv = "  ".join(f"{k[:3]} {v:.3g}" for k, v in feats_row.items())
    axp.set_title(f"#{iid} {idx['category'].iat[iid][:18]}\n"
                  f"{str(idx['filename'].iat[iid])[:22]}\n"
                  f"z {zval:+.2f}{title_extra}\n{fv}", fontsize=5.5,
                  color=INK, loc="left")


def figures(idx, colour, codes, cat_names, sc, getz, Rr, Fc, crop_names):
    os.makedirs(P("figures"), exist_ok=True)
    E = np.load(EMB_PATH, mmap_mode="r")
    Avec = np.load(os.path.join(AXES_DIR, "axis_vectors.npy")).astype(np.float64)
    fcol = [crop_names.index(f) for f in FIG_FEATURES]
    checks, chosen = [], []

    def check(iid, col):
        rec = float(np.asarray(E[iid], dtype=np.float64) @ Avec[col])
        sto = float(Rr[iid, col])
        checks.append({"image_id": iid, "col": col, "stored": sto,
                       "recomputed": rec, "diff": abs(rec - sto)})
        if not np.isfinite(sto) or abs(rec - sto) > FIG_SCORE_TOL:
            pd.DataFrame(checks).to_csv(P("figure_check.csv"), index=False)
            raise SystemExit(f"FIGURE CHECK FAILED: image {iid}, column {col}: "
                             f"stored {sto} vs recomputed {rec}.")

    def frow(iid):
        return dict(zip(FIG_FEATURES, np.asarray(Fc[iid, fcol], dtype=float)))

    # categories from the v3b ranking (colour_only)
    counts_col = np.bincount(codes[colour], minlength=len(cat_names))
    elig = counts_col >= FIG_MIN_CAT_IMAGES
    v3 = sc[sc["set"].isin([PRIMARY_SET] + FIG_OWN_RANKING)]
    cat_choice = {}
    for r in v3.itertuples():
        z = getz(r.col)
        fin = colour & np.isfinite(z)
        s = np.bincount(codes[fin], weights=z[fin], minlength=len(cat_names))
        cnt = np.bincount(codes[fin], minlength=len(cat_names))
        mean = np.where(elig, s / np.maximum(cnt, 1), np.nan)
        valid = elig & np.isfinite(mean)
        order = np.argsort(np.where(valid, mean, np.inf))
        order = order[:valid.sum()]                        # ascending
        med = np.nanmedian(mean[valid])
        mid = np.argsort(np.abs(np.where(valid, mean, np.inf) - med))[:2]
        pick = [("high", order[-1]), ("high", order[-2]), ("mid", mid[0]),
                ("mid", mid[1]), ("low", order[1]), ("low", order[0])]
        cat_choice[r.axis] = [(lvl, c, mean[c]) for lvl, c in pick]
        for lvl, c, mu in cat_choice[r.axis]:
            chosen.append({"v3b_axis": r.axis, "level": lvl,
                           "category": cat_names[c], "mean_z": mu,
                           "n_colour_only": int(counts_col[c])})

    for fset in FIG_SETS:
        os.makedirs(P("figures", fset), exist_ok=True)
        for r in sc[sc["set"] == fset].itertuples():
            z = getz(r.col)
            # --- global (all images) ---
            ok = np.isfinite(z)
            order = np.argsort(np.where(ok, z, -np.inf))
            top = order[::-1][:FIG_TOPK]
            bot = np.argsort(np.where(ok, z, np.inf))[:FIG_TOPK]
            fig, axs = plt.subplots(2, FIG_TOPK, figsize=(1.9 * FIG_TOPK, 5.2))
            for k, iid in enumerate(top):
                check(int(iid), r.col)
                draw_cell(axs[0, k], int(iid), idx, z[iid], frow(iid))
            for k, iid in enumerate(bot):
                check(int(iid), r.col)
                draw_cell(axs[1, k], int(iid), idx, z[iid], frow(iid))
            axs[0, 0].set_ylabel("highest", fontsize=9, color=INK)
            axs[1, 0].set_ylabel("lowest", fontsize=9, color=INK)
            fig.suptitle(f"{fset} · {r.axis} · top and bottom {FIG_TOPK} of all "
                         f"images (box = CLIP's crop)", fontsize=10, color=INK,
                         x=0.01, ha="left")
            fig.tight_layout()
            fig.savefig(P("figures", fset, f"{safe(r.axis)}_global.png"),
                        dpi=130)
            plt.close(fig)
            # --- within categories (colour_only) ---
            v3_axis = V2_TO_V3.get(r.axis, r.axis)
            if v3_axis not in cat_choice:
                continue
            cats = cat_choice[v3_axis]
            ncol = 2 * FIG_PER_CAT
            fig, axs = plt.subplots(len(cats), ncol,
                                    figsize=(1.9 * ncol, 2.35 * len(cats)))
            for ri, (lvl, c, mu) in enumerate(cats):
                ids = np.where((codes == c) & colour & ok)[0]
                o = ids[np.argsort(z[ids])]
                hi, lo = o[::-1][:FIG_PER_CAT], o[:FIG_PER_CAT]
                for k, iid in enumerate(list(hi) + list(lo)):
                    check(int(iid), r.col)
                    rank = int((z[ids] > z[iid]).sum()) + 1
                    draw_cell(axs[ri, k], int(iid), idx, z[iid], frow(iid),
                              f"  (rank {rank}/{len(ids)})")
                axs[ri, 0].set_ylabel(f"{lvl}\n{cat_names[c][:20]}\n"
                                      f"mean z {mu:+.2f}", fontsize=7, color=INK)
            rank_on = ("this axis" if fset in FIG_OWN_RANKING
                       else f"v3b {short(v3_axis)}")
            fig.suptitle(f"{fset} · {r.axis} · within category: "
                         f"{FIG_PER_CAT} highest (left) and lowest (right); "
                         f"categories ranked on {rank_on} (colour_only)",
                         fontsize=9, color=INK, x=0.01, ha="left")
            fig.tight_layout()
            fig.savefig(P("figures", fset, f"{safe(r.axis)}_within.png"),
                        dpi=130)
            plt.close(fig)
            print(f"    figures: {fset} {short(r.axis)}")
    pd.DataFrame(checks).to_csv(P("figure_check.csv"), index=False)
    pd.DataFrame(chosen).to_csv(P("figure_categories.csv"), index=False)
    return pd.DataFrame(checks), pd.DataFrame(chosen)


# ============================================================
# Main
# ============================================================

def fmt_ci(v, lo, hi):
    return f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    print("=" * 70)
    print("07_category_confounds_v4.py")
    print("=" * 70)
    (idx, colour, feats, feats_new, get_features, Fc, Fs, crop_names,
     stretch_names, sc, Z, Rr) = load_inputs()
    n = len(idx)
    cat_names = sorted(idx["category"].unique())
    codes = pd.Categorical(idx["category"], categories=cat_names).codes.astype(int)

    axes_df = sc[sc["set"].isin(MAIN_SETS + R2B_SETS) | (sc["kind"] == "control")] \
        .sort_values("col").reset_index(drop=True)
    print(f"Images {n:,}; features {len(feats)}; axes {len(axes_df)}")
    Fsel = get_features(feats)
    if feats_new:
        print(f"Step 04c features: {', '.join(feats_new)}")
    Zsel = np.asarray(Z[:, axes_df["col"].to_numpy()], dtype=np.float32)

    # --- new methods ---
    wb_rows, cons_rows, eta_rows, pcs, cs_all = [], [], [], {}, []
    masks = {"colour_only": colour, "all_images": np.ones(n, bool)}
    for label in IMAGE_SETS:
        r, c_, pc, e, (mA, mF), cs_ = analyse(label, masks[label], codes, cat_names,
                                              Fsel, Zsel, feats, axes_df, rng)
        cs_all += cs_
        wb_rows += r
        cons_rows += c_
        eta_rows += e
        pc.to_csv(P(f"per_category_{label}.csv"), index=False, float_format="%.4f")
        mA.to_csv(P(f"category_means_axes_{label}.csv"), index=False,
                  float_format="%.5g")
        mF.to_csv(P(f"category_means_features_{label}.csv"), index=False,
                  float_format="%.5g")
    wb = pd.DataFrame(wb_rows)
    wb.to_csv(P("within_between.csv"), index=False, float_format="%.4f")

    # --- check R1: v2/v3 results equal the v3 run's ---
    r1 = None
    if V3_RESULTS and os.path.exists(V3_RESULTS):
        old = pd.read_csv(V3_RESULTS)
        keys = ["image_set", "set", "axis", "feature"]
        cols = ["within_rho", "within_beta", "between_beta", "within_beta_lo",
                "within_beta_hi", "n_images"]
        cols = [c for c in cols if c in old.columns]
        m_ = old[keys + cols].merge(wb[keys + cols], on=keys, suffixes=("_v3", "_now"))
        diffs = {c: float((m_[f"{c}_v3"] - m_[f"{c}_now"]).abs().max()) for c in cols}
        worst = max(diffs.values()) if diffs else np.nan
        # the v3 run's CSV is rounded to 4 decimals
        ok = len(m_) > 0 and all(v <= R1_TOL for c, v in diffs.items()
                                 if c != "n_images") and diffs.get("n_images", 0) == 0
        r1 = {"n_old": len(old), "n_matched": len(m_), "max_diff": diffs, "pass": ok}
        print(f"  R1 v3 run: {len(m_)} of {len(old)} rows matched; max |diff| "
              f"{worst:.1e} -> {'PASS' if ok else 'WARN'}")
    else:
        print("  R1 v3 run: skipped (V3_RESULTS not found)")
    pd.DataFrame(cons_rows).to_csv(P("consistency.csv"), index=False,
                                   float_format="%.4f")
    cs_df = pd.DataFrame(cs_all)
    cs_df.to_csv(P("control_share.csv"), index=False, float_format="%.4f")
    eta_df = pd.DataFrame(eta_rows)
    eta_df.to_csv(P("eta_squared.csv"), index=False, float_format="%.4f")
    fsb = feature_set_bridge(wb)
    if len(fsb):
        fsb.to_csv(P("feature_set_bridge.csv"), index=False,
                   float_format="%.4f")

    # --- mixed-model check ---
    mix = pd.DataFrame()
    if RUN_MIXED_SUBSAMPLE:
        sub_axes = axes_df[axes_df["set"].isin(MIXED_SETS)]
        cols = [list(axes_df.index).index(i) for i in sub_axes.index]
        mix = mixed_subsample(colour, codes, cat_names, Fsel, Zsel[:, cols],
                              feats, sub_axes.reset_index(drop=True), rng)
        mix.to_csv(P("mixed_subsample.csv"), index=False, float_format="%.4f")

    # --- legacy ---
    br = pd.DataFrame()
    if RUN_LEGACY:
        print("Legacy block (old Script B methods)...")
        leg = legacy_block(codes, cat_names, Fs, Fc, stretch_names, crop_names,
                           Z, sc)
        leg.to_csv(P("legacy_results.csv"), index=False, float_format="%.4f")
        br = build_bridge(leg, wb)
        br.to_csv(P("bridge.csv"), index=False, float_format="%.4f")

    # --- figures ---
    fchk, fcat = pd.DataFrame(), pd.DataFrame()
    if RUN_FIGURES:
        print("Figures...")
        colpos = {c: j for j, c in enumerate(axes_df["col"])}

        def getz(col):
            if col in colpos:
                return Zsel[:, colpos[col]].astype(np.float64)
            return np.asarray(Z[:, col], dtype=np.float64)
        fchk, fcat = figures(idx, colour, codes, cat_names, sc, getz, Rr, Fc,
                             crop_names)
    ffig = pd.DataFrame()
    if RUN_FEATURE_FIGURES and feats_new:
        print("Feature figures (visual check of the 04c features)...")
        show = FEATURE_FIGURES or feats_new
        Fvals = {f: Fsel[:, feats.index(f)].astype(np.float64)
                 if f in feats else get_features([f])[:, 0].astype(np.float64)
                 for f in show}
        ffig = feature_figures(idx, colour, codes, cat_names, Fvals, show)

    # --- report ---
    prim = IMAGE_SETS[0]
    L = ["07_category_confounds_v4.py -- summary", "=" * 78,
         f"Date      : {time.strftime('%Y-%m-%d %H:%M')}",
         f"Features  : {', '.join(feats)}",
         f"            ({len(feats_new)} from step 04c: "
         f"{', '.join(feats_new) or 'none'})",
         f"Selection : {SELECTION_JSON}",
         f"Axes      : {len(axes_df)} ({', '.join(MAIN_SETS + R2B_SETS)} + "
         f"controls); primary {PRIMARY_SET}",
         f"Scores    : {SCORES_DIR}",
         ("CHECK R1  : skipped (V3_RESULTS not found)" if r1 is None else
          f"CHECK R1  : v2/v3 results vs {V3_RESULTS}: {r1['n_matched']} of "
          f"{r1['n_old']} rows matched; largest |diff| "
          + ", ".join(f"{k} {v:.1e}" for k, v in r1["max_diff"].items())
          + f" -> {'PASS (CSV rounded to 4 decimals)' if r1['pass'] else 'WARN: differs from the v3 run'}"),
         f"Images    : {', '.join(IMAGE_SETS)} (primary {prim}); categories "
         f">= {MIN_IMAGES_PER_CATEGORY} images",
         f"Intervals : {int(CI_LEVEL * 100)}% from {N_BOOT} resamples of whole "
         "categories", "",
         "HOW TO READ",
         "  within rho   Spearman of category-centred score and feature "
         "(one feature at a time)",
         "  within beta  the feature's standardised coefficient with all "
         f"{len(feats)} features together, within category",
         "  between beta the same across category means (one category = "
         "one observation)",
         "  same sign    share of categories whose own within-category "
         "coefficient has the pooled sign", ""]
    L.append("IMAGES LEFT OUT (an undefined feature value):")
    for label in IMAGE_SETS:
        q = wb[wb["image_set"] == label]
        if len(q):
            L.append(f"  {label:<12} {int(q['n_left_out_undefined'].iloc[0]):,}"
                     f" left out; {int(q['n_images'].iloc[0]):,} used")
    if "hue_spread" in feats:
        L.append("  hue_spread is undefined for greyscale photographs, so "
                 "all_images is close to colour_only here;")
        L.append("  the version-1 run (10 features) is the all_images "
                 "reference.")
    L.append("")
    for label in IMAGE_SETS:
        e = eta_df[(eta_df["image_set"] == label) & (eta_df["set"] == PRIMARY_SET)]
        L.append(f"ETA2 ({label}, {PRIMARY_SET}): share of each axis's variance "
                 "between categories")
        L += [f"  {r.axis[:42]:<42} {r.eta2:.3f}" for r in e.itertuples()]
    L.append("")

    d = wb[(wb["image_set"] == prim) & (wb["set"] == PRIMARY_SET)]
    L.append(f"PRIMARY RESULTS ({PRIMARY_SET}, {prim})")
    for ax in d["axis"].unique():
        L.append(f"  {ax}")
        L.append(f"    {'feature':<15} {'within rho':<24} {'within beta':<24} "
                 f"{'between beta':<24} same sign (r / beta)")
        for r in d[d["axis"] == ax].itertuples():
            L.append(f"    {r.feature:<15} "
                     f"{fmt_ci(r.within_rho, r.within_rho_lo, r.within_rho_hi):<24} "
                     f"{fmt_ci(r.within_beta, r.within_beta_lo, r.within_beta_hi):<24} "
                     f"{fmt_ci(r.between_beta, r.between_beta_lo, r.between_beta_hi):<24} "
                     f"{r.consistency_r_same_sign:.0%} / "
                     f"{r.consistency_beta_same_sign:.0%}")
    L.append("")
    if feats_new:
        L.append(f"NEW FEATURES ({PRIMARY_SET}, {prim}): within beta; "
                 "* = interval excludes 0")
        q = d[d["feature"].isin(feats_new)]
        hdr = f"  {'axis':<26}" + "".join(f"{f[:13]:>15}" for f in feats_new)
        L.append(hdr)
        for ax in q["axis"].unique():
            row = f"  {short(ax)[:26]:<26}"
            for f in feats_new:
                t = q[(q["axis"] == ax) & (q["feature"] == f)]
                if not len(t):
                    row += f"{'':>15}"
                    continue
                t = t.iloc[0]
                star = "*" if (t.within_beta_lo > 0 or t.within_beta_hi < 0) \
                    else " "
                row += f"{t.within_beta:>+14.3f}{star}"
            L.append(row)
        L.append("")
    L.append(f"KEY PAIRS ACROSS VERSIONS ({prim}; within rho / within beta)")
    for ax, f in KEY_PAIRS:
        parts = []
        for s in MAIN_SETS:
            a = V2_TO_V3.get(ax, ax) if s not in ("v2",) else ax
            q = wb[(wb["image_set"] == prim) & (wb["set"] == s) &
                   (wb["axis"] == a) & (wb["feature"] == f)]
            if len(q):
                parts.append(f"{s} {q.within_rho.iloc[0]:+.2f}/"
                             f"{q.within_beta.iloc[0]:+.2f}")
        L.append(f"  {short(ax)} x {f}: " + "  ".join(parts))
    L.append("")
    L.append(f"CONTROLS ({prim}; within rho for the first 4 features)")
    c = wb[(wb["image_set"] == prim) & (wb["kind"] == "control")]
    for ax in c["axis"].unique():
        q = c[c["axis"] == ax].head(4)
        L.append(f"  {ax:<10} " + "  ".join(f"{r.feature} {r.within_rho:+.2f}"
                                             for r in q.itertuples()))
    L.append("")
    L.append("IMAGE SETS: largest |difference| colour_only vs all_images "
             f"({PRIMARY_SET}, within rho)")
    if len(IMAGE_SETS) > 1:
        a = wb[wb["set"] == PRIMARY_SET].pivot_table(
            index=["axis", "feature"], columns="image_set", values="within_rho")
        dd = (a[IMAGE_SETS[0]] - a[IMAGE_SETS[1]]).abs().sort_values(ascending=False)
        L += [f"  {i[0][:30]:<30} {i[1]:<15} {v:.3f}" for i, v in dd.head(5).items()]
    L.append("")
    if len(mix):
        m_ = mix.dropna(subset=["beta"])
        L.append(f"MIXED-MODEL CHECK (colour_only subsample, {MIXED_PER_CAT} per "
                 f"category): converged {int(m_.drop_duplicates(['model', 'set', 'axis'])['converged'].sum())}"
                 f" of {m_.drop_duplicates(['model', 'set', 'axis']).shape[0]} fits")
        w = wb[(wb["image_set"] == "colour_only")].set_index(["set", "axis", "feature"])
        for model in ("old_random_intercept", "mundlak_random_intercept"):
            q = m_[m_["model"] == model]
            diffs = [abs(r.beta - w["within_beta"].get((r.set, r.axis, r.feature), np.nan))
                     for r in q.itertuples()]
            L.append(f"  {model:<26} median |beta - within beta| "
                     f"{np.nanmedian(diffs):.3f}")
        L.append("")
    # ---------------- v4 ----------------
    q4 = wb[(wb["image_set"] == prim) & (wb["set"] == V4_SET)]
    if len(q4):
        L += ["=" * 78, f"v4 AXES ({prim}; v4_protocol.md 1.17)", "=" * 78]
        for label in IMAGE_SETS:
            e = eta_df[(eta_df["image_set"] == label) & (eta_df["set"] == V4_SET)]
            L.append(f"ETA2 ({label}, {V4_SET})")
            L += [f"  {r.axis[:42]:<42} {r.eta2:.3f}" for r in e.itertuples()]
        L.append("")
        L.append(f"RESULTS ({V4_SET}, {prim})")
        for ax in q4["axis"].unique():
            L.append(f"  {ax}")
            L.append(f"    {'feature':<15} {'within rho':<24} {'within beta':<24} "
                     f"{'between beta':<24} same sign (r / beta)")
            for r in q4[q4["axis"] == ax].itertuples():
                L.append(f"    {r.feature:<15} "
                         f"{fmt_ci(r.within_rho, r.within_rho_lo, r.within_rho_hi):<24} "
                         f"{fmt_ci(r.within_beta, r.within_beta_lo, r.within_beta_hi):<24} "
                         f"{fmt_ci(r.between_beta, r.between_beta_lo, r.between_beta_hi):<24} "
                         f"{r.consistency_r_same_sign:.0%} / "
                         f"{r.consistency_beta_same_sign:.0%}")
        L.append("")
        L.append(f"KEY PAIRS: v4 next to v3b ({prim}; within rho / within beta "
                 "[interval])")
        for aid, v3ax, f in KEY_PAIRS_V4:
            a4 = [a for a in q4["axis"].unique() if a.startswith(aid + " ")]
            t4 = q4[(q4["axis"].isin(a4)) & (q4["feature"] == f)]
            t3 = wb[(wb["image_set"] == prim) & (wb["set"] == PRIMARY_SET) &
                    (wb["axis"] == v3ax) & (wb["feature"] == f)]
            if len(t4) and len(t3):
                a_, b_ = t4.iloc[0], t3.iloc[0]
                L.append(f"  {aid} x {f:<15} v4 {a_.within_rho:+.2f} / "
                         f"{fmt_ci(a_.within_beta, a_.within_beta_lo, a_.within_beta_hi)}"
                         f"   v3b {short(v3ax)} {b_.within_rho:+.2f} / "
                         f"{fmt_ci(b_.within_beta, b_.within_beta_lo, b_.within_beta_hi)}")
        L.append("")
        # R2b
        qn = wb[(wb["image_set"] == prim) & (wb["set"].isin(R2B_SETS))]
        if len(qn):
            mrg = q4.merge(qn, on=["axis", "feature"], suffixes=("", "_nc"))
            L.append(f"R2b SENSITIVITY ({prim}): within beta of the v4 axis and of "
                     "its version with the darkness and enclosure directions removed")
            L.append("  links whose interval excludes 0 in either version; "
                     "* = the interval excludes 0")
            sig = lambda lo, hi: "*" if (lo > 0 or hi < 0) else " "
            for r in mrg.itertuples():
                s1 = sig(r.within_beta_lo, r.within_beta_hi)
                s2 = sig(r.within_beta_lo_nc, r.within_beta_hi_nc)
                if s1 == "*" or s2 == "*":
                    L.append(f"    {r.axis[:30]:<30} {r.feature:<15} "
                             f"{r.within_beta:+.3f}{s1}  ->  "
                             f"{r.within_beta_nc:+.3f}{s2}")
            d_ = (mrg["within_beta"] - mrg["within_beta_nc"]).abs()
            L.append(f"  largest |change| in within beta: {d_.max():.3f} "
                     f"({mrg.loc[d_.idxmax(), 'axis'][:30]} x "
                     f"{mrg.loc[d_.idxmax(), 'feature']})")
            L.append("")
        # control share
        if len(cs_df):
            L.append(f"CONTROL SHARE (decision 13; {prim}): share of each axis's "
                     f"within-category variance explained by "
                     f"{', '.join(CONTROL_SHARE)} together, and the within "
                     "correlation with each. Reported, not used to drop an axis.")
            rc = [c for c in cs_df.columns if c.startswith("r_")]
            for s_ in (V4_SET, PRIMARY_SET):
                for r in cs_df[(cs_df["image_set"] == prim) & (cs_df["set"] == s_)].itertuples():
                    L.append(f"  {s_:<4} {r.axis[:34]:<34} "
                             f"{r.control_share:.2f} [{r.control_share_lo:.2f}, "
                             f"{r.control_share_hi:.2f}]   "
                             + "  ".join(f"{c[2:]} {getattr(r, c):+.2f}" for c in rc))
            L.append("")
        # images: colour_only vs all_images for v4
        if len(IMAGE_SETS) > 1:
            a = wb[wb["set"] == V4_SET].pivot_table(
                index=["axis", "feature"], columns="image_set", values="within_rho")
            dd = (a[IMAGE_SETS[0]] - a[IMAGE_SETS[1]]).abs().sort_values(ascending=False)
            L.append(f"IMAGE SETS ({V4_SET}): largest |difference| colour_only vs "
                     "all_images, within rho")
            L += [f"  {i[0][:30]:<30} {i[1]:<15} {v:.3f}" for i, v in dd.head(5).items()]
            L.append("")

    if len(br):
        L.append("BRIDGE (v2 axes, all images; full table in bridge.csv)")
        L.append("  columns: old published | old method new data (old "
                 "features) | old method, crop features | new within rho "
                 "(v2) | new within rho (v3b)")
        for ax, f in KEY_PAIRS:
            q = br[(br["axis"] == ax) & (br["feature"] == f)]
            if not len(q):
                continue
            r = q.iloc[0]
            g = lambda k: r[k] if k in r and pd.notna(r[k]) else np.nan
            L.append(f"  {short(ax)} x {f}")
            L.append(f"    raw Spearman (A2)     {g('old_A2_spearman'):+.3f} | "
                     f"{g('legacy_A2_spearman_stretch'):+.3f} | "
                     f"{g('legacy_A2_spearman_crop'):+.3f}")
            L.append(f"    centred (A6)          {g('old_A6_spearman_centered'):+.3f} | "
                     f"{g('legacy_A6_spearman_centered_stretch'):+.3f} | "
                     f"{g('legacy_A6_spearman_centered_crop'):+.3f} | "
                     f"{g('new_within_rho_v2'):+.3f} | {g('new_within_rho_v3b'):+.3f}")
            L.append(f"    mixed beta (A5)       {g('old_A5_mixed_beta'):+.3f} | "
                     f"{g('legacy_A5_mixed_beta_stretch'):+.3f} | "
                     f"{g('legacy_A5_mixed_beta_crop'):+.3f} | within beta v2 "
                     f"{g('new_within_beta_v2'):+.3f} | v3b "
                     f"{g('new_within_beta_v3b'):+.3f}")
        if "old_A2_spearman" in br:
            dmax = (br["old_A2_spearman"] - br["legacy_A2_spearman_stretch"]).abs().max()
            L.append(f"  check: largest |old published - old method on new data| "
                     f"(A2) = {dmax:.4f} (encoding change only; expected small)")
        else:
            L.append("  old published results not found in OLD_RESULTS_DIR")
        L.append("")
    if len(fsb):
        q = fsb[(fsb["image_set"] == prim) & (fsb["set"] == PRIMARY_SET)]
        L.append(f"FEATURE-SET BRIDGE ({PRIMARY_SET}, {prim}): the shared "
                 f"features with {len(feats)} vs the version-1 run "
                 "(feature_set_bridge.csv)")
        L.append(f"  images: {int(q['n_images_prev'].iloc[0]):,} -> "
                 f"{int(q['n_images_now'].iloc[0]):,}" if len(q) else "")
        L.append(f"  largest |change|: within rho "
                 f"{q['within_rho_change'].abs().max():.3f}, within beta "
                 f"{q['within_beta_change'].abs().max():.3f}")
        L.append("  largest within-beta changes (before -> after):")
        top = q.reindex(q["within_beta_change"].abs()
                        .sort_values(ascending=False).index).head(8)
        for rr in top.itertuples():
            L.append(f"    {short(rr.axis)[:22]:<22} x {rr.feature:<15} "
                     f"{rr.within_beta_prev:+.3f} -> {rr.within_beta_now:+.3f}")
        L.append("")
    elif PREVIOUS_RESULTS:
        L.append(f"FEATURE-SET BRIDGE: {PREVIOUS_RESULTS} not found; skipped")
        L.append("")
    if len(ffig):
        L.append(f"FEATURE FIGURES: {ffig['feature'].nunique()} features in "
                 f"{P('figures', 'features')}; categories in "
                 "feature_figure_categories.csv")
        L.append("")
    if len(fchk):
        L.append(f"FIGURES: {fchk['image_id'].nunique()} images shown; stored vs "
                 f"recomputed score max |diff| {fchk['diff'].max():.1e} -> PASS")
        L.append(f"  in {P('figures')}; categories in figure_categories.csv")
    L.append(f"\nElapsed: {(time.time() - t0) / 60:.1f} min")
    report = "\n".join(L)
    with open(P("confound_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    json.dump({"script": "07_category_confounds_v4.py",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "features": feats, "features_04c": feats_new,
               "selection_json": SELECTION_JSON, "extra_dir": EXTRA_DIR,
               "previous_results": PREVIOUS_RESULTS, "main_sets": MAIN_SETS,
               "r2b_sets": R2B_SETS, "control_share": CONTROL_SHARE,
               "scores_dir": SCORES_DIR, "axes_dir": AXES_DIR,
               "v3_results": V3_RESULTS, "check_r1": r1,
               "image_sets": IMAGE_SETS, "n_boot": N_BOOT,
               "min_images_per_category": MIN_IMAGES_PER_CATEGORY,
               "seed": RANDOM_SEED, "old_results_dir": OLD_RESULTS_DIR},
              open(P("run_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
