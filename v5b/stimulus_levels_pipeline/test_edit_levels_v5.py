"""
test_edit_levels_v5.py  (v5 stimulus pipeline: edits above AND below the
                         original, tested before the pipeline is rebuilt)

Renders, measures, displays and CLIP-scores candidate edit sets for the 47
v4 scenes. Nothing here is a stimulus set: it is the evidence for choosing
one. No existing script or data file is modified.

Why this exists
---------------
In the v4 stimuli every contrast edit is a haze (haze_eq) and the unhazed
original never appears in a pair; saturation has four levels below the
original and one above. So "the less edited image" is almost always the
crisper, more colourful one, and a participant who picks "the normal-
looking image" for the ordinary episode, or "the odd-looking image" for
derealization, produces the predicted difference without using memory.
With three levels above AND three below the original on each feature,
"edited-looking" points both ways and that strategy becomes detectable.
This script asks, for several ways of editing upward, whether such a
ladder can be built, how it measures, how it looks at 512 px, and how
CLIP reads it.

What it does
------------
  0. Start-up checks: the copied feature code against data/features
     (fatal), the 20,000-image stimulus-path reference re-measured against
     the stored one, the stimulus-path SDs against v5's 39.182 / 9.348.
  1. DEFINITION (a), corpus percentiles, median-calibrated as in v4: every
     family x direction is applied on a parameter grid to v4's 150-image
     calibration sample, and each level is the parameter at which the
     MEDIAN image's measured feature reaches the named corpus percentile
     (p15/p25/p35 below, p65/p75/p85 above). The haze_eq and chroma levels
     v4 already has must come out identical (a check).
  2. DEFINITION (b), equal steps in corpus SD, per scene: each family's
     REACH on each scene (how far the feature moves before the clipping
     budget, if B_USE_CLIP_BUDGET, the colour-gamut plateau, a reversal or
     the end of the grid), the symmetric span min(up, down) per scene, the
     span 90% of scenes reach, the median scene's span and the fixed
     +-0.3/0.6/0.9 SD steps. The rendered span is B_SPAN_SD (the same for
     every approach and feature), or, if that is None, the span 90% of
     scenes reach. Levels at 1/3, 2/3 and 3/3 of the rendered span are solved
     PER SCENE, so every scene hits exactly +-k SD unless it cannot, in
     which case the level is rendered at the scene's reach and labelled
     CAPPED.
  3. DEFINITION (c), Kusztor-style fixed Pillow factors, for the Pillow
     approach only.
     DEFINITION (r), equal RATIO steps: like (b), but the features in
     R_FEATURES (contrast) are multiplied by q^(1/3), q^(2/3), q below the
     original and divided by them above, so a step up is as large as a
     ratio as the step down; q is the corpus-median image's -R_SPAN_SD
     ratio (0.653 at 1 SD). The other feature keeps (b)'s SD steps.
  4. Every scene x approach x definition rendered as the full 7 x 7 grid
     (contrast edit first, then colour, as v4's MANIP_ORDER), every cell
     saved as PNG and measured: the 18 corpus features, three 04c
     texture measures, the change from the scene's original in corpus SD,
     cross-talk, newly clipped pixels, the envelope, and three artefact
     measures (halo, noise amplification, banding).
  5. Normality check (distance from typical, per image and per pair, and
     its correlation with the v5 direction terms), compared with the v4
     design rendered by the same code.
  Scenes listed in EXCLUDE_SCENES are rendered and shown but SET ASIDE:
     left out of every pooled number (spans, summaries, normality,
     scene-fit counts, CLIP statistics) and reported on their own.
  6. Scene fit: per scene x approach x definition, whether all seven
     levels are reached on each feature, which are capped, and which
     flags occur.
  7. Contact sheets, and a self-contained HTML viewer (pairs at 512 CSS
     px, the experiment's own surround), plus a small viewer pack to copy
     to other devices.
  8. CLIP scoring with the v4 encoder and axes (switch: RUN_CLIP), the
     pre-stated questions in both directions, and the editedness
     measures, including one NEW, EXPLORATORY caption pair.

CLIP is not a human observer. Its results say how the edits move the
text-defined concepts, not what participants will see.

The edits
---------
  haze_eq        v4's brightness-equated haze (verbatim). Down only.
  dehaze_rgb     the inverse of the haze blend, run backwards on RGB, with
                 haze_eq's brightness restoration: v4's m_contrast_up,
                 verbatim. Up only. v4 found it clipped at once.
  dehaze_L       the same inverse blend on CIELAB L* only, about the
                 airlight's L*, then haze_eq's linear-light brightness
                 restoration. Up only.
  clarity_L      unsharp mask on L*, sigma = CLARITY_SIGMA_FRAC x 512.
  sharpen_L      unsharp mask on L*, sigma = SHARPEN_SIGMA (09c's 3 px).
  pyramid_L      the finest PYR_LEVELS Laplacian-pyramid bands of L*
                 scaled by one gain g (g < 1 down, g > 1 up).
  tone_L         a global tone curve on CIELAB L* (09c's contrast curve),
                 a*, b* unchanged: k < 1 compresses toward the mean L*,
                 k > 1 is an S-curve with black and white fixed. Used by
                 B3_tonecurve.
  tone_Y         the same curve on luminance with RGB proportions kept.
                 Kept for comparison, used by no approach: it made dark
                 saturated colours glow when contrast was reduced.
  pil_contrast   Pillow ImageEnhance.Contrast (Kusztor et al. 2025).
  chroma         v4's CIELAB chroma edit (verbatim): k <= 1 scales a*, b*;
                 k > 1 blends toward each pixel's in-gamut ceiling.
  pil_color      Pillow ImageEnhance.Color (Kusztor et al. 2025).
  The approaches pairing them are in APPROACHES. A4_haze_cc restores the
  saturation each contrast level moves with a per-level chroma factor;
  A4_haze_cc_rgb does the same with dehaze_rgb going up, the exact mirror
  of haze_eq.

Lightness-only edits change L* at fixed a*, b*; a pixel pushed outside
sRGB has its chroma reduced to the gamut boundary at its new lightness
and hue (cmax_of, as 09c's temperature edit does), and such pixels count
toward the clipping budget (where one applies).

Units
-----
Every change is in corpus SD of the feature, measured on the stimulus
path (original -> 512 -> 256), relative to the same scene's original.
The pair predictors use v5's fixed divisors (pred_sat = -d_sat / 39.182,
pred_fog = d_fog / 9.348, d_fog = -(contrast_local_A - contrast_local_B)).

Conventions
-----------
  * Feature code, geometry and the v4 manipulations are copied VERBATIM
    from build_stimulus_set_v4_corrected.py (the script that rendered the
    stimuli); m_contrast_up and interp_param from
    calibrate_levels_v4_corrected.py; the three texture measures from
    09c_manipulation_sweep.py; the encoder, the axis loading and the
    corpus checks from score_stimuli_on_axes_v4.py; the text encoder from
    05_build_axes_v4.py. Each block is marked where it starts.
  * Paths are relative to the project folder; run from there.
  * Output goes to OUTPUT_ROOT/<settings hash>/ and never overwrites a run
    (a folder already holding files gets _2, _3, ...).
  * Fixed seeds throughout.

Outputs (under the run folder)
------------------------------
  report.txt            start here; plain-language summary at the end
  checks.csv            every check: code, status, detail
  settings.json         every setting, and the hash that names the folder
  run_meta.json         versions, timings, status
  levels_a.csv          definition (a): one row per family x direction x
                        percentile: parameter, status, achieved change
  calibration_grid.csv  definition (a): every grid point on the sample
  comp_a.csv            definition (a): the A4 compensation factors
  reach.csv             definition (b): per scene x family x direction
  reach_grid.csv        definition (b): every grid point of the search,
                        with clipping and the artefact measures, so any
                        artefact threshold can be applied afterwards
  spans.csv             definition (b): per scene x approach x feature
  span_summary.csv      median-scene span, 90% span, rendered span, and the
                        share of scenes reaching +-0.3/0.6/0.9 SD
  ladders.csv           every level of every set: target, parameter(s),
                        achieved change, CAPPED
  cells.csv             one row per rendered image (all measurements)
  crosstalk.csv, artefacts.csv, envelope_violations.csv
  normality_images.csv, normality_pairs.csv.gz, normality_summary.csv,
  straddle_counts.csv
  scene_fit.csv, scene_fit_summary.csv
  reproduction_v4.csv   the v4 grid re-rendered against the v4 PNGs
  identity.csv, pillow_model.csv
  viewer.html           the viewer; links images/ by relative path
  viewer_pack/          a small copy of the viewer for other devices
  images/               EVERY image, kept apart from the tables:
    <approach>/<definition>/<scene_id>/c<k>_s<k>.png   (k = n3..0..p3)
    sheets/<approach>/<definition>/<scene_id>.png      contact sheets
  clip/                 CLIP outputs (embeddings, scores, effects,
                        questions, editedness, checks of the encoder)

Usage
-----
    conda activate drl2drv
    python test_edit_levels_v5.py

Rendering is CPU work in N_WORKERS processes. CLIP needs the un2CLIP
checkpoint, data/embeddings, data/axes_v4 and data/scores_v4, and is fast
on a CUDA GPU (8-10 GB) and slow on a CPU. To score on another machine,
run with RUN_CLIP = False here and then, there, with CLIP_ONLY_FROM set to
this run's folder: only the CLIP part runs, into that folder's clip/
(clip_2/, ... if it exists).

Rough run time and size (scaled from a test run; 20 workers)
------------------------------------------------------------
  rendering part   about 15-30 min the first time: the 20,000-image
                   reference ~2-5 min, the definition-(a) grids ~5-10 min
                   (both cached for later runs), the reach search ~2 min,
                   the 47 x 735 images ~8-15 min. One core: several hours.
  CLIP part        about 9,700 images (the cross cells of 15 sets, plus
                   v4's cross cells): ~5-10 min on a CUDA GPU with 8-10 GB
                   (ViT-L/14-336 in fp32, batch 32), including the corpus
                   reproduction checks; ~2-4 h on a CPU. RAM ~4 GB (the
                   1.1M x 768 embedding file is memory-mapped).
  disk             ~34,500 PNGs at 512 px, roughly 8-12 GB, plus sheets.
  Fifth run (A4_haze_cc, A4_haze_cc_rgb, B1_pyramid; (b) and (r)):
                   6 sets, ~13,800 PNGs, ~4,600 images for CLIP; about
                   10 min of rendering on 20 workers.
  Fourth run (all eight approaches, definition (b) only):
                   8 sets x 49 = 392 images per scene, ~18,400 PNGs (4-6
                   GB); about 5,300 images for CLIP. The reach search runs
                   every family's full grid (no budget stops it early).
"""

import hashlib
import io
import json
import os
import platform
import shutil

import time
import warnings
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageDraw, ImageEnhance, ImageStat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================
_G0 = set(globals())

# --- Inputs (all read, never written) --------------------------------
#
# The scenes are the 47 of the v4 stimulus set, taken from its manifest
# (scene_id, category, source_path), so the scene list cannot drift from
# the set the experiment uses.
V4_MANIFEST_CSV  = "data/stimulus_set_v4_ext_haze_eq/stimulus_manifest.csv"
V4_STIMULI_DIR   = "data/stimulus_set_v4_ext_haze_eq/stimuli"
V4_LEVELS_CSV    = "data/levels_v4_ext/levels.csv"
V4_STATS_CSV     = "data/levels_v4_ext/corpus_feature_stats_v4.csv"
V4_PER_IMAGE_CSV = "data/levels_v4_ext/per_image_levels.csv"
# v4's committed CLIP scores of its own stimuli; the v4 cross cells are
# re-scored here and must agree (check CL1). None to skip.
V4_AXES_EFFECTS_CSV = "data/stimulus_axes_v4/cell_effects.csv"

INDEX_CSV    = "data/selection/image_index.csv"          # step 02
FEATURES_NPY = "data/features/features_crop.npy"         # step 04
FEATURE_DEFS = "data/features/feature_definitions.json"  # step 04
FLAGS_CSV    = "data/monochrome/image_flags.csv"         # step 04a
DUP_CSV      = "data/checks_v4/duplicate_groups.csv"     # 09a
CATEGORY_CSV = "everyday_categories.csv"                 # as v4 step 1

N_SCENES_EXPECTED = 47

# --- Output ----------------------------------------------------------
# The run folder is OUTPUT_ROOT/<hash of the settings>; a folder that
# already holds files is never touched (_2, _3, ... instead).
OUTPUT_ROOT = "data/edit_levels_v5"

# The two slow, unchanging stages (the 20,000-image reference and the
# definition-(a) calibration grids) are cached here, keyed on a
# fingerprint of everything that can change them, including the source
# of the functions involved. Outside the run folder on purpose. Delete
# the folder to force a full recompute.
USE_CACHE = True
CACHE_DIR = "data/edit_levels_v5_cache"

# --- Geometry and the v4 manipulations (must match v4; do not edit) --
IMAGE_SIZE        = 512        # edits applied, and images shown, at this size
FEATURE_SIZE      = 256        # features measured at this size
CLIP_PX           = 336        # CLIP's resolution, for clip_box
HAZE_AIRLIGHT_TOP = 1.0        # % brightest pixels whose mean is the airlight
GAMUT_ITERS       = 16
GAMUT_CMAX_SEARCH = 160.0
GAMUT_MARGIN      = 0.995
MANIP_ORDER       = ("haze", "chroma")   # contrast edit first, then colour
IDENTITY_SHORT_CIRCUIT = True  # the "no change" setting returns the image
                               # itself, as v4 (identity.csv measures what
                               # each edit would do without it)
FEAT_SAT = "sat_mean"
FEAT_CON = "contrast_local"

# --- Corpus reference and calibration sample (as v4 step 1) ----------
N_CALIB_IMAGES     = 150
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

# REF_PERCENTILES
#   Every integer percentile is written for the re-measured stimulus-path
#   reference, so p85 (needed now) and anything asked later are there.
REF_PERCENTILES = list(range(1, 100))

# REF_CHECK_RTOL
#   The re-measured reference must reproduce v4's stored stimulus-path
#   percentiles (same seed, same images, same code). Stored with 6
#   significant digits, so the tolerance is relative.
REF_CHECK_RTOL = 2e-5

# The predictor divisors of v5 (corpus SD on the stimulus path). The
# re-measured SDs must agree to SD_CHECK_TOL.
SD_SAT_V5    = 39.182
SD_FOG_V5    = 9.348
SD_CHECK_TOL = 0.0006

# --- Definition (a): corpus percentiles --------------------------------
# Below the median, then above it, nearest first. p15/p25/p35 mirror
# p85/p75/p65 in percentile points (not in SD: see the report).
PCT_DOWN = [35, 25, 15]
PCT_UP   = [65, 75, 85]

# --- Clipping (one rule, both directions) ------------------------------
# A level (definition a) or a reach (definition b) stops where the share
# of NEWLY clipped pixels exceeds CLIP_MAX_NEW: a pixel with a channel
# that was inside (0, 255) and is pushed more than CLIP_DEPTH levels past
# the range, or (lightness-only edits) whose chroma had to be cut by more
# than GAMUT_MAP_MIN_DC to stay inside sRGB, or whose L* left [0, 100].
# The shares at the other CLIP_REPORT_DEPTHS are reported, not used.
#
# LOOSENED for the second run (first run: 0.01, 0.5, 0.5). The half-level
# rule was v4's rule for its DOWNWARD levels; haze moves pixels inward
# and never trips it, while any contrast increase pushes the extremes
# outward and trips it at once (first run: every upward edit stopped at
# +0.03 to +0.27 SD, invisible at 512 px). 4 levels is v4's rule for its
# upward column; a chroma cut of 3 CIELAB units is about one just-
# noticeable difference. Still one rule for both directions; the
# downward edits do not clip, so their levels do not move (check C2).
#
# Third run: CLIP_MAX_NEW 0.03 -> 0.05, a middle step. In the second run,
# 3% -> 5% lifted the median scene's S-curve reach from 0.96 to 1.30 SD
# but the value 90% of scenes reach only from 0.35 to 0.45 SD: a few
# scenes were the limit (see EXCLUDE_SCENES). 10% would have reached
# 0.84 SD for every scene at Kusztor-like clipping.
CLIP_MAX_NEW       = 0.05
CLIP_DEPTH         = 4.0
CLIP_REPORT_DEPTHS = (0.5, 2.0, 4.0, 8.0)
GAMUT_MAP_MIN_DC   = 3.0      # CIELAB chroma units

# --- The edits ---------------------------------------------------------
CLARITY_SIGMA_FRAC = 0.03     # clarity radius: 3% of 512 = 15.4 px
SHARPEN_SIGMA      = 3.0      # fine sharpening radius, px at 512 (09c)
PYR_LEVELS         = 3        # pyramid bands scaled: about 1, 2, 4 px at
                              # 512 (contrast_local sees roughly 2-4 px)

# Parameter grids, from the identity outward (GRID_N steps each). The
# haze_eq and chroma grids are v4's, so definition (a) reproduces v4.
GRID_N           = 24
HAZE_MAX         = 0.95
CHROMA_MIN       = 0.0
CHROMA_MAX       = 2.0        # chroma above 1, definition (a) (v4's grid)
CHROMA_MAX_REACH = 4.0        # chroma above 1, reach search (b): wider, so
                              # the gamut plateau is found, not the grid end
DEHAZE_MIN       = -2.0       # most negative haze for the inverse blend
                              # (deviations from the airlight amplified up
                              # to x3). Fifth run: was -0.50 (x1.5), which
                              # stopped the upward dehaze of about 40 of
                              # 47 scenes at the end of the range, short
                              # of what definition (r) asks going up. Now
                              # the reach ends only where contrast_local
                              # stops rising ("feature peaks").
CLARITY_MAX      = 4.0        # unsharp-mask amount
SHARPEN_MAX      = 6.0
PYR_GAIN_MAX     = 4.0
PIL_CONTRAST_MAX = 2.5
PIL_COLOR_MAX    = 3.0
TONE_MAX         = 4.0        # tone-curve slope at the mean (tone_L, tone_Y)

# --- Approaches ----------------------------------------------------------
#   contrast_down / contrast_up   the edit below / above the original
#   colour                        the saturation edit (both directions)
#   compensate                    restore sat_mean at every contrast
#                                 level with a chroma factor
#   definitions                   which level definitions apply
APPROACHES = {
    "A1_dehaze_rgb": dict(contrast_down="haze_eq", contrast_up="dehaze_rgb",
                          colour="chroma", compensate=False,
                          definitions=["a", "b"]),
    "A1_dehaze_L":   dict(contrast_down="haze_eq", contrast_up="dehaze_L",
                          colour="chroma", compensate=False,
                          definitions=["a", "b"]),
    "A2_clarity":    dict(contrast_down="haze_eq", contrast_up="clarity_L",
                          colour="chroma", compensate=False,
                          definitions=["a", "b"]),
    "A3_sharpen":    dict(contrast_down="haze_eq", contrast_up="sharpen_L",
                          colour="chroma", compensate=False,
                          definitions=["a", "b"]),
    # haze_eq with its saturation loss restored, mirrored by the
    # lightness-only inverse blend (its saturation change restored too)
    "A4_haze_cc":    dict(contrast_down="haze_eq", contrast_up="dehaze_L",
                          colour="chroma", compensate=True,
                          definitions=["a", "b", "r"]),
    # the exact mirror of haze_eq: v4's inverse blend on RGB (dehaze_rgb)
    # going up, saturation restored in both directions
    "A4_haze_cc_rgb": dict(contrast_down="haze_eq", contrast_up="dehaze_rgb",
                           colour="chroma", compensate=True,
                           definitions=["a", "b", "r"]),
    "B1_pyramid":    dict(contrast_down="pyramid_L", contrast_up="pyramid_L",
                          colour="chroma", compensate=False,
                          definitions=["a", "b", "r"]),
    "B2_pillow":     dict(contrast_down="pil_contrast",
                          contrast_up="pil_contrast", colour="pil_color",
                          compensate=False, definitions=["a", "b", "c"]),
}
# B3: the tone curve on CIELAB L* (tone_L). The second run used tone_Y
# (luminance, RGB proportions kept), which made dark reds and blues glow
# when contrast was reduced (CIELAB chroma +24% / +36% at p15) and
# clipped bright coloured highlights going up; tone_Y stays in the
# registry for comparison but no approach uses it.
APPROACHES["B3_tonecurve"] = dict(contrast_down="tone_L", contrast_up="tone_L",
                                  colour="chroma", compensate=False,
                                  definitions=["a", "b", "r"])
DEFINITION_NAMES = {"a": "a_pct", "b": "b_sd", "c": "c_kusztor",
                    "r": "r_ratio"}

# RUN_APPROACHES
#   Which approaches this run renders (None = all). The second run keeps
#   the new tone curve and the three comparisons; the inverse-haze and
#   clarity approaches stopped at about +0.03 to +0.11 SD in the first
#   run. Approaches left out are dropped from APPROACHES below.
#   Fourth run: every approach again (None). With no clipping budget on
#   definition (b) (B_USE_CLIP_BUDGET below), the upward edits that
#   stopped at once in the first run can now reach the span, so all of
#   them are worth a look.
#   Fifth run: the two preferred approaches and the exact-mirror A4.
RUN_APPROACHES = ["A4_haze_cc", "A4_haze_cc_rgb", "B1_pyramid"]

# RUN_DEFINITIONS
#   Which level definitions this run renders (None = every one an
#   approach lists). Fourth run: (b) only. Definition (a)'s calibration
#   still runs for haze_eq and chroma, cached, because check C2 (v4's
#   levels reproduced) needs it; nothing is rendered with it.
#   Fifth run: (b) and (r), side by side.
RUN_DEFINITIONS = ["b", "r"]

# EXCLUDE_SCENES
#   Scenes SET ASIDE. They are still rendered, measured, saved and shown
#   in viewer.html (marked "set aside"), and listed in report section
#   8b, but they are left out of everything pooled over scenes: the
#   rendered span of definition (b) (so they are solved toward the span
#   the others reach, and capped where they cannot), the ladder, cross-
#   talk, artefact and envelope summaries, the scene-fit summary, the
#   normality check (the v4 baseline too), the CLIP statistics, and the
#   viewer pack. Definition (a) is calibrated on 150 corpus photographs,
#   not on these scenes, so it is unaffected. [] sets nothing aside.
#
#   Third run: six scenes, the weakest S-curve reach under the 5%
#   clipping budget. Fourth run: none. Without a clipping budget on
#   definition (b) the span is fixed (B_SPAN_SD) and no scene limits it.
EXCLUDE_SCENES = []

# The A4 compensation factor: chroma k searched in this range. Above
# k = 1 the soft-saturating blend can stop short of full restoration;
# what is left is reported (comp_residual_sd), never hidden.
COMP_K_RANGE = (0.25, 3.0)
COMP_ITERS   = 20
COMP_ROUNDS  = 2              # definition (b): contrast and compensation
                              # are re-solved alternately this many times

# --- Definition (b): equal SD steps, per scene ---------------------------
# B_SPAN_SD
#   The span rendered by definition (b), corpus SD, the same for every
#   approach and both features: levels at +-1/3, 2/3, 3/3 of it, solved
#   per scene. None = the earlier rule: the span 90% of scenes reach
#   (SPAN_QUANTILE), different for every approach and feature.
#   Fourth run: 1.0, about as far as v4's p15 haze level and roughly the
#   corpus p15 / p85 for a normal distribution, so the steps are about
#   0.33 SD apart, like the fixed +-0.3 / 0.6 / 0.9 steps.
B_SPAN_SD        = 1.0
# B_USE_CLIP_BUDGET
#   Whether the clipping budget (CLIP_MAX_NEW) stops the reach of
#   definition (b). False = it does not: the reach stops only at the end
#   of the parameter grid, a reversal of the feature, or (colour going
#   up) the gamut plateau, so a level is CAPPED only where the edit
#   itself cannot go further. Clipping is still measured on every image
#   and reported (artefacts.csv, cells.csv, the viewer), never hidden.
#   Fourth run: False.
B_USE_CLIP_BUDGET = False
# --- Definition (r): equal steps as RATIOS ------------------------------
# For the features in R_FEATURES, every scene's value is multiplied by
# q^(1/3), q^(2/3), q below the original and divided by the same factors
# above it: equal steps in log units, so a step up is as large AS A RATIO
# as the step down (perceived contrast follows ratios, not differences).
# q is the ratio of the corpus-median image's level at -R_SPAN_SD:
#   q = 1 - R_SPAN_SD x SD / corpus median   (contrast_local, 1 SD: 0.653)
# so the median image goes from -1 SD to about +1.53 SD. A scene with
# more contrast than the median moves further in SD both ways, one with
# less moves less. Features not in R_FEATURES keep (b)'s equal SD steps:
# colour would need about +2.4 SD of saturation above the original to
# mirror -1 SD as a ratio, beyond the screen's colour range.
# The normality check measures distance in corpus SD, so by design (r)
# looks asymmetric there; that is the intended trade.
R_FEATURES = ["contrast"]
R_SPAN_SD  = 1.0
SPAN_QUANTILE    = 0.10       # B_SPAN_SD = None: the rendered span is the
                              # value 90% of scenes reach (10th percentile)
LEVEL_FRACTIONS  = [1 / 3, 2 / 3, 1.0]
FIXED_STEPS_SD   = [0.3, 0.6, 0.9]   # reported as numbers only
REACH_BISECT_ITERS = 12       # refine a clipping limit between grid points
SOLVE_ITERS      = 24         # bisection on the parameter per level
SOLVE_TOL_SD     = 0.002      # |achieved - target| that ends the search
# LEVEL_CHECK_TOL_SD
#   Check L1: an uncapped level of (b) should land within this of its
#   target. Looser than SOLVE_TOL_SD because the measured change is a step
#   function of the parameter (8-bit output; haze_eq rounds between its
#   blend and its 14-step brightness search; Pillow truncates).
LEVEL_CHECK_TOL_SD = 0.01
# PLATEAU_REL
#   The gamut limit of the colour edits above the original: the search
#   stops where one grid step adds less than this share of the largest
#   step gain so far (the soft blend approaches the gamut ceiling and
#   never crosses it). Contrast edits and edits below the original stop
#   only at the clipping budget, a reversal, or the end of the grid.
PLATEAU_REL      = 0.05
# REACH_ARTEFACT_LIMITS
#   Optional artefact limits on the reach. None = off (the default: no
#   thresholds were agreed). Every grid point's artefact values are in
#   reach_grid.csv, so a limit can be applied afterwards without
#   re-running. Keys: halo_share, noise_ratio, band_new_empty.
REACH_ARTEFACT_LIMITS = {"halo_share": None, "noise_ratio": None,
                         "band_new_empty": None}

# --- Definition (c): Kusztor-style fixed Pillow factors ------------------
# Seven factors, most reduced first; 1.0 is the original. Applied to
# ImageEnhance.Contrast and ImageEnhance.Color alike.
KUSZTOR_FACTORS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

# --- Artefact measures (numbers only; no thresholds) ---------------------
# Halo: pixels near strong edges whose L* lies outside the ORIGINAL's
# local min/max (HALO_WINDOW_PX square) by more than HALO_TOL_L, after
# that min/max is passed through the image's own global tone curve. The
# tone curve is the mean edited L* per original L* bin, made monotone, so
# a purely global edit (haze, a gain) scores ~0 and only LOCAL overshoot
# (an unsharp mask's halo) counts.
HALO_CANNY     = (100, 200)
HALO_BAND_PX   = 8
HALO_WINDOW_PX = 9
HALO_TOL_L     = 3.0
# Noise amplification: high-pass energy of L* (L* minus a sigma
# NOISE_HP_SIGMA blur) in the original's flattest NOISE_FLAT_PERCENTILE %
# of pixels (gradient of a sigma NOISE_SMOOTH_SIGMA blur, grey 3-252, the
# rule of 04c's noise_sigma at 512 px), edited / original.
NOISE_FLAT_PERCENTILE = 40
NOISE_SMOOTH_SIGMA    = 2.0
NOISE_HP_SIGMA        = 1.0
# Banding: on the 512 px grey image, empty histogram bins between the
# BAND_RANGE_PCT percentiles, and a comb index (mean |h_i - (h_i-1 +
# h_i+1) / 2| / mean h), each edited minus original.
BAND_RANGE_PCT = (1, 99)

# --- Envelope (as v4) ----------------------------------------------------
ENVELOPE_FEATURES = ["lum_mean", "lum_p05", "lum_p50", "lum_p95",
                     "val_mean", "lum_skew", "spectral_slope"]
ENVELOPE_LO, ENVELOPE_HI = "p01", "p99"

# Extra descriptive measures reported because contrast_local sees only a
# narrow band of detail (04c's measures at 256, copied from 09c).
EXTRA_TEXTURE = ["fine_texture", "noise_sigma", "edge_overshoot"]
DESCRIPTIVE   = ["lum_std", "spectral_slope", "lum_mean", "lum_p05",
                 "lum_p95"] + EXTRA_TEXTURE

# --- Pair rules (v5: analyse_pair_capacity_v5.py) ------------------------
V4_MIN_DSAT_SAT_ONLY = 17.0
V4_PAIR_SAT_TOL      = 3.0
V4_MIN_DFOG_FOG_ONLY = 4.2
V4_MIN_DSAT_MIXED    = 8.0
V4_MIN_DFOG_MIXED    = 3.0
V4_BALANCE_LO, V4_BALANCE_HI = 0.6, 1.7
SAT_ONLY_MIN_DSAT    = 31.1
SAT_ONLY_CON_TOL     = 0.72
FOG_ONLY_SAT_TOL     = 3.0
BALANCED_CONFLICT_ONLY = True
FOG_SIGN             = -1.0
# The v4 design as the baseline of the normality check, with v5's
# exclusions (no h = 0 row in pairs; p65 dropped).
V4_DROP_COLOUR = ["p65"]
V4_DROP_FOG    = ["h=0"]
# A pair "straddles" the original on a feature when its two images lie on
# opposite sides by level; the measured version also needs both changes
# beyond this many SD.
STRADDLE_TOL_SD = 0.02

# --- Scene fit -------------------------------------------------------------
# A step between adjacent levels smaller than this (corpus SD) counts as
# collapsed: v4's MIN_COLOUR_STEP of 1.0 sat_mean unit, in SD.
MIN_STEP_SD = 1.0 / SD_SAT_V5

# --- Display -------------------------------------------------------------
SHEET_TILE_PX  = 200          # contact-sheet cell (the grid block's range)
VIEWER_GRID_PX = 200          # grid view cell, CSS px
VIEWER_BG      = "#7f7f7f"    # the experiment's surround
VIEWER_GAP_PX  = 32           # the experiment's gap between the pair
VIEWER_PACK         = True
VIEWER_PACK_SCENES  = None    # None: VIEWER_PACK_N scenes spread across
                              # baseline brightness; or a list of scene_ids
VIEWER_PACK_N       = 10
VIEWER_PACK_CELLS   = "cross" # "cross" (the 13 cells the pairs use) or "all"
VIEWER_PACK_ZIP     = True

# --- CLIP ----------------------------------------------------------------
RUN_CLIP       = True
CLIP_ONLY_FROM = None         # an earlier run folder: score it, render nothing
CLIP_CELLS     = "cross"      # "cross" (13 per set) or "all" (49 per set)
EMB_DIR    = "data/embeddings"
AXES_DIR   = "data/axes_v4"
SCORES_DIR = "data/scores_v4"
CLIP_MODEL_NAME   = "openai/clip-vit-large-patch14-336"
UN2CLIP_CKPT_PATH = "./pretrained_models/openai_vit_l_14_336.ckpt"
OPENAI_MODEL_NAME = "ViT-L/14@336px"
BATCH_SIZE        = 32
ALLOW_TF32        = False
HASH_CHECKPOINT   = True
V4_SET          = "v4"
NOCONTROL_SET   = "v4@nocontrol"
CONTROLS_SET    = "controls"
V4_IDS          = ["A01", "A02", "A03", "A04", "A05", "A07", "A11",
                   "A13", "A14"]
QUALITY_KEY     = "controls::quality"           # + = "a bad photo"
REPORT_CONTROLS = ["controls::quality", "controls::weather",
                   "controls::greyscale"]
CTRL_TEMPLATE   = "a scene that looks {}"
CONTROLS = {
    "enclosure": [("enclosed", "open"), ("cramped", "spacious"),
                  ("like a closed room", "like a wide open landscape"),
                  ("like an indoor room", "like the outdoors")],
    "darkness":  [("dark", "bright"), ("dimly lit", "brightly lit"),
                  ("like nighttime", "like daytime"), ("shadowy", "sunlit")],
}
SD_SAMPLE       = 200_000
# NEW, EXPLORATORY: one caption pair, encoded with step 05's text encoder,
# scored like a control direction. Not part of v4; labelled as such.
NEW_CAPTION_PAIR = ("an edited photo", "an unedited photo")
TEXT_CHECK_N     = 16          # stored captions re-encoded (check T1)
TEXT_MIN_COS     = 0.9999
EDITEDNESS_KEYS  = ["ctrl_quality", "A03", "A04", "new_edited"]
EQUIV_MARGIN_SD  = 0.20        # "about the same", descriptive
N_BOOT      = 2000
CI_LEVEL    = 0.95
N_CHECK_IMAGES        = 16
SAME_PIXELS_MIN_COS   = 0.9999
OWN_PREPROC_MIN_COS   = 0.99999
CORPUS_RANDOM_N       = 32
CORPUS_EMB_MIN_COS    = 0.99999
CORPUS_POOLED_MIN_COS = 0.9999
Z_STORED_TOL          = 1e-4
Z_REENCODED_TOL       = 1e-3
AXIS_NORM_TOL         = 1e-4
NOCONTROL_MIN_COS     = 0.9999
STIM_RECHECK_N        = 8
STIM_RECHECK_MIN_COS  = 0.99999
V4_CLIP_TOL           = 1e-3   # CL1: v4 cross cells re-scored vs committed

# --- Parallelism -----------------------------------------------------------
N_WORKERS = 20
CHUNK     = 1

if RUN_APPROACHES is not None:
    _bad = [a for a in RUN_APPROACHES if a not in APPROACHES]
    if _bad:
        raise SystemExit(f"RUN_APPROACHES names unknown approaches: {_bad}")
    APPROACHES = {k: v for k, v in APPROACHES.items() if k in RUN_APPROACHES}
if RUN_DEFINITIONS is not None:
    _bad = [d for d in RUN_DEFINITIONS if d not in DEFINITION_NAMES]
    if _bad:
        raise SystemExit(f"RUN_DEFINITIONS names unknown definitions: {_bad}")
    APPROACHES = {k: dict(v, definitions=[d for d in v["definitions"]
                                          if d in RUN_DEFINITIONS])
                  for k, v in APPROACHES.items()}
    APPROACHES = {k: v for k, v in APPROACHES.items() if v["definitions"]}

_SETTING_NAMES = sorted(k for k in globals()
                        if k not in _G0 and not k.startswith("_"))
# Operational settings that do not change any result; left out of the hash.
_NOT_HASHED = {"N_WORKERS", "CHUNK", "USE_CACHE", "CACHE_DIR",
               "CLIP_ONLY_FROM", "RUN_CLIP", "VIEWER_PACK",
               "VIEWER_PACK_SCENES", "VIEWER_PACK_N", "VIEWER_PACK_CELLS",
               "VIEWER_PACK_ZIP", "HASH_CHECKPOINT"}


# ################################################################
# VERBATIM from build_stimulus_set_v4_corrected.py (lines 740-1176): feature code, geometry, conversions, v4 manipulations, apply, render_cell
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
# VERBATIM from 09c_manipulation_sweep.py (lines 318-329, 436-446, 529-585): 04c texture measures
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
# VERBATIM from calibrate_levels_v4_corrected.py (lines 1344-1388): m_contrast_up (dehaze_rgb)
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
# VERBATIM from calibrate_levels_v4_corrected.py (lines 2264-2297): interp_param
# ################################################################
def interp_param(grid_d, target, cap=None):
    """The parameter at which the median change reaches `target`.

    The grid starts at the identity (change 0) and moves away from it,
    so the median change is monotone in the grid index; the first
    crossing is taken and interpolated linearly between the two
    neighbouring grid points.

    Returns (parameter, status).
    """
    g = grid_d.sort_values("grid_index")
    x = g.param.to_numpy()
    y = g.median_change.to_numpy()
    clip = g.mean_clipped_new.to_numpy()

    # Where clipping would exceed the cap, the grid is cut short.
    over = np.where(clip > (CLIP_MAX_NEW if cap is None else cap))[0]
    cut = over[0] if len(over) else len(x)
    x, y = x[:cut], y[:cut]
    if len(x) < 2:
        return np.nan, "capped (clipping)"

    s = np.sign(target) if target != 0 else 1.0
    hit = np.where(s * y >= s * target)[0]
    if not len(hit):
        status = ("capped (clipping)" if cut < len(g)
                  else "capped (out of range)")
        return float(x[-1]), status
    j = int(hit[0])
    if j == 0:
        return float(x[0]), "ok"
    y0, y1 = y[j - 1], y[j]
    t = 0.0 if y1 == y0 else (target - y0) / (y1 - y0)
    return float(x[j - 1] + t * (x[j] - x[j - 1])), "ok"


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


# ============================================================
# Run state, checks, output folder, cache
# ============================================================

RUN = {"out": None, "base": None, "checks": [], "warnings": [],
       "t0": time.time(), "timings": {}, "meta": {}}
CHECKS = RUN["checks"]


def P(name):
    """A path in the current output folder (the clip/ folder while CLIP
    runs, because the verbatim CLIP code writes through P)."""
    return os.path.join(RUN["out"], name)


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
    s["_script"] = "test_edit_levels_v5.py"
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
    meta = {"script": "test_edit_levels_v5.py", "status": status,
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
    if not USE_CACHE:
        return None
    p = os.path.join(CACHE_DIR, f"{name}_{key}.npz")
    if not os.path.isfile(p):
        return None
    try:
        return dict(np.load(p, allow_pickle=False))
    except Exception as exc:
        print(f"  cache {p} unreadable ({exc}); recomputing.")
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


def cell_file(approach, d, sid, cx, sx):
    """Relative path (from the run folder) of one cell's PNG."""
    return (f"images/{approach}/{DEFINITION_NAMES[d]}/{sid}/"
            f"c{ix(cx)}_s{ix(sx)}.png")


def sheet_file(approach, d, sid):
    return f"images/sheets/{approach}/{DEFINITION_NAMES[d]}/{sid}.png"


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


# ============================================================
# Corpus, stimulus-path reference, calibration sample
# From calibrate_levels_v4_corrected.py (load_categories, _percentiles,
# load_corpus, _ref_task, measure_reference, check_feature_code,
# draw_sample), adapted: report lists replaced by checks, every integer
# percentile written, the cache keyed in this script's own folder.
# The eligibility rule and the random draws are unchanged, so the
# reference and the calibration sample are v4's.
# ============================================================

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


def draw_sample(idx, eligible, rng):
    """v4's calibration sample: N_CALIB_IMAGES eligible images drawn with
    the step-1 generator AFTER check_feature_code has used it, as step 1
    does."""
    pool = np.where(eligible)[0]
    if len(pool) < N_CALIB_IMAGES:
        stop(f"only {len(pool)} eligible images.")
    ids = np.sort(rng.choice(pool, size=N_CALIB_IMAGES, replace=False))
    rows, bad = [], 0
    for i in ids:
        try:
            sq = load_square(idx.path.iat[int(i)])
        except Exception:
            bad += 1
            continue
        rows.append({"image_id": int(i), "square": sq,
                     "airlight": airlight_of(sq),
                     "lum_mean_512": mean_grey(sq)})
    if bad:
        RUN["warnings"].append(f"{bad} sampled images unreadable")
    for r in rows:
        r["base"] = measure(r["square"])
    if os.path.isfile(V4_PER_IMAGE_CSV):
        v4ids = set(pd.read_csv(V4_PER_IMAGE_CSV,
                                usecols=["image_id"]).image_id.astype(int))
        mine = {r["image_id"] for r in rows}
        check("C1", mine == v4ids,
              f"calibration sample: {len(mine)} images, "
              f"{len(mine & v4ids)} shared with v4's per_image_levels.csv "
              f"({len(v4ids)})", critical=False)
    else:
        skip("C1", f"{V4_PER_IMAGE_CSV} not found")
    return rows


# ============================================================
# Definition (a): corpus percentiles, calibrated on the median image
# ============================================================

def dims_for_a():
    """Every (family, direction) definition (a) needs, plus chroma both
    ways (which the reproduction check C2 needs anyway)."""
    dims = {("chroma", "down"), ("chroma", "up"), ("haze_eq", "down")}
    for a in APPROACHES.values():
        if "a" not in a["definitions"]:
            continue
        dims.add((a["contrast_down"], "down"))
        dims.add((a["contrast_up"], "up"))
        dims.add((a["colour"], "down"))
        dims.add((a["colour"], "up"))
    return sorted(dims)


def _calib_task(args):
    fam, direction, gi, p = args
    feat = FAMILIES[fam]["feature"]
    cross = other_feature(feat)
    d, c, k = [], [], []
    for r in W["rows"]:
        out, rec = apply_step(fam, r["square"], p, r)
        v = measure(out, [feat, cross])
        d.append(v[feat] - r["base"][feat])
        c.append(v[cross] - r["base"][cross])
        k.append(clip_budget_share(rec))
    return np.array(d), np.array(c), np.array(k)


def _dim_key(fam, direction, rows):
    payload = json.dumps({
        "fam": fam, "dir": direction,
        "grid": [round(float(p), 12) for p in grid_of(fam, direction, "a")],
        "ids": [int(r["image_id"]) for r in rows],
        "clip": [CLIP_DEPTH, list(CLIP_DEPTHS), GAMUT_MAP_MIN_DC],
        "sizes": [IMAGE_SIZE, FEATURE_SIZE, CLIP_PX],
        "edits": [CLARITY_SIGMA_FRAC, SHARPEN_SIGMA, PYR_LEVELS,
                  GAMUT_ITERS, GAMUT_CMAX_SEARCH, GAMUT_MARGIN,
                  IDENTITY_SHORT_CIRCUIT],
        "code": _src(FAMILIES[fam]["fn"], apply_step, measure,
                     legacy_features, _lab_l_edit, _restore_mean_grey,
                     m_chroma, m_haze_eq, m_haze, rgb_of_f, lab_of,
                     cmax_of, cmax_of_img, lin_to_srgb_f, srgb_to_lin_f,
                     srgb_to_lin, mean_grey, to_u8)},
        sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def run_calibration_grid(rows):
    dims = dims_for_a()
    results, fresh, keys = {}, [], {}
    for fam, direction in dims:
        keys[(fam, direction)] = _dim_key(fam, direction, rows)
        blob = cache_get(f"grid_{fam}_{direction}", keys[(fam, direction)])
        if blob is None:
            fresh.append((fam, direction))
            continue
        for gi in range(len(blob["param"])):
            results[(fam, direction, gi)] = (float(blob["param"][gi]),
                                             blob["d"][gi], blob["c"][gi],
                                             blob["k"][gi])
    print(f"  {len(dims)} family x direction grids; "
          f"{len(dims) - len(fresh)} from the cache, {len(fresh)} computed")
    tasks = [(fam, direction, gi, float(p)) for fam, direction in fresh
             for gi, p in enumerate(grid_of(fam, direction, "a"))]
    if tasks:
        got = run_pool(_calib_task, tasks, {"rows": rows},
                       f"grid ({len(rows)} images)")
        for t, (d, c, k) in zip(tasks, got):
            results[(t[0], t[1], t[2])] = (t[3], d, c, k)
        for fam, direction in fresh:
            g = sorted(k for k in results if k[:2] == (fam, direction))
            cache_put(f"grid_{fam}_{direction}", keys[(fam, direction)],
                      param=np.array([results[k][0] for k in g]),
                      d=np.array([results[k][1] for k in g]),
                      c=np.array([results[k][2] for k in g]),
                      k=np.array([results[k][3] for k in g]))
    out = []
    for (fam, direction, gi), (p, d, c, k) in results.items():
        out.append({"family": fam, "direction": direction,
                    "dimension": f"{fam}:{direction}", "grid_index": gi,
                    "param": p, "feature": FAMILIES[fam]["feature"],
                    "median_change": float(np.median(d)),
                    "q25_change": float(np.percentile(d, 25)),
                    "q75_change": float(np.percentile(d, 75)),
                    "median_cross_change": float(np.median(c)),
                    "mean_clipped_new": float(np.mean(k)),
                    "max_clipped_new": float(np.max(k))})
    return (pd.DataFrame(out).sort_values(["dimension", "grid_index"])
            .reset_index(drop=True))


def solve_levels_a(grid, rows, st):
    """One row per family x direction x percentile: the parameter at which
    the median image reaches the target (v4's interp_param), its status,
    and what it achieves when applied to the sample again."""
    rowsout = []
    for (fam, direction), g in grid.groupby(["family", "direction"]):
        g = g.sort_values("grid_index")
        feat = FAMILIES[fam]["feature"]
        p50, sd = float(st.at[feat, "p50"]), float(st.at[feat, "sd"])
        # the feature must move away from the original monotonically; a
        # grid that reverses is cut at its peak (as 09c does)
        y = g.median_change.to_numpy()
        s = -1.0 if direction == "down" else 1.0
        rev = np.where(np.diff(s * y) < -1e-9)[0]
        peak_cut = len(rev) > 0
        gg = g.iloc[: rev[0] + 1] if peak_cut else g
        for q in (PCT_DOWN if direction == "down" else PCT_UP):
            target = float(st.at[feat, f"p{q:02d}"]) - p50
            param, status = interp_param(gg, target, CLIP_MAX_NEW)
            if peak_cut and status == "capped (out of range)":
                status = "capped (feature peaks)"
            if not np.isfinite(param):
                param = FAMILIES[fam]["identity"]
                status = "capped (no step within the clipping budget)"
            rowsout.append({"family": fam, "direction": direction,
                            "feature": feat, "level": f"p{q:02d}",
                            "percentile": q, "target_change": target,
                            "target_sd": target / sd, "param": param,
                            "status": status})
    tasks = [(r["family"], r["direction"], i, r["param"])
             for i, r in enumerate(rowsout)]
    got = run_pool(_calib_task, tasks, {"rows": rows}, "levels (a)")
    for r, (d, c, k) in zip(rowsout, got):
        feat = r["feature"]
        sd = float(st.at[feat, "sd"])
        sdc = float(st.at[other_feature(feat), "sd"])
        t = r["target_change"]
        r.update({"achieved_change": float(np.median(d)),
                  "achieved_sd": float(np.median(d)) / sd,
                  "achieved_q25_sd": float(np.percentile(d, 25)) / sd,
                  "achieved_q75_sd": float(np.percentile(d, 75)) / sd,
                  "share_reaching_target": float((d <= t).mean() if t < 0
                                                 else (d >= t).mean()),
                  "cross_change_sd": float(np.median(c)) / sdc,
                  "mean_clipped_new": float(np.mean(k)),
                  "capped": r["status"] != "ok"})
    return pd.DataFrame(rowsout)


def check_levels_vs_v4(levels):
    """C2: haze_eq and chroma levels that v4 also calibrated must come out
    with v4's parameters (same sample, same reference, same grid)."""
    if not os.path.isfile(V4_LEVELS_CSV):
        skip("C2", f"{V4_LEVELS_CSV} not found")
        return
    L = pd.read_csv(V4_LEVELS_CSV)
    pairs = []
    for r in levels.itertuples():
        dim = {("haze_eq", "down"): "fog_eq", ("chroma", "down"): "colour",
               ("chroma", "up"): "colour_up"}.get((r.family, r.direction))
        if dim is None:
            continue
        m = L[(L.dimension == dim) & (L.level == r.level)]
        if not len(m):
            continue
        pairs.append((f"{dim} {r.level}", float(r.param),
                      float(m.param.iloc[0])))
    if not pairs:
        skip("C2", "no level in common with v4")
        return
    worst = max(abs(a - b) / max(abs(b), 1e-6) for _, a, b in pairs)
    check("C2", worst <= 1e-5,
          f"{len(pairs)} levels shared with {V4_LEVELS_CSV} "
          f"({', '.join(f'{n}={a:.4f}' for n, a, _ in pairs)}): worst "
          f"relative difference {worst:.1e}", critical=False)


# ------------------------------------------------------------
# The A4 compensation factor
# ------------------------------------------------------------

COMP_GRID = np.r_[np.linspace(COMP_K_RANGE[0], 1.0, 13)[:-1],
                  np.linspace(1.0, COMP_K_RANGE[1], 33)]


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


def _comp_a_task(args):
    """For one sample image: sat_mean change after (contrast step at each
    level) + chroma k, over COMP_GRID."""
    i = args
    r = W["rows"][i]
    out = []
    for fam, p in W["levels"]:
        img, _ = apply_step(fam, r["square"], p, r)
        cm = cmax_of_img(img)
        row = []
        for k in COMP_GRID:
            u8, _ = chroma_on(img, float(k), cm)
            row.append(measure(u8, [FEAT_SAT])[FEAT_SAT] - r["base"][FEAT_SAT])
        out.append(row)
    return np.array(out)


def calibrate_comp_a(levels, rows, st):
    """Definition (a) compensation: for every contrast level of every
    compensated approach, the chroma factor at which the MEDIAN image's
    sat_mean change is zero (interpolated on COMP_GRID), then re-applied
    to measure what is left."""
    need = []
    for name, a in APPROACHES.items():
        if not (a["compensate"] and "a" in a["definitions"]):
            continue
        for direction, fam in (("down", a["contrast_down"]),
                               ("up", a["contrast_up"])):
            sub = levels[(levels.family == fam)
                         & (levels.direction == direction)]
            for r in sub.itertuples():
                if (fam, float(r.param)) not in need:
                    need.append((fam, float(r.param)))
    if not need:
        return pd.DataFrame()
    got = run_pool(_comp_a_task, list(range(len(rows))),
                   {"rows": rows, "levels": need}, "compensation (a)")
    D = np.stack(got)                      # image x level x k
    med = np.median(D, axis=0)
    sd = float(st.at[FEAT_SAT, "sd"])
    out = []
    for j, (fam, p) in enumerate(need):
        y = med[j]
        status = "ok"
        if y.min() > 0 or y.max() < 0:
            k = float(COMP_GRID[int(np.argmin(np.abs(y)))])
            status = "capped (out of range)"
        else:
            k = float(np.interp(0.0, y, COMP_GRID)) if np.all(np.diff(y) > 0) \
                else float(COMP_GRID[int(np.argmin(np.abs(y)))])
        kk = int(np.argmin(np.abs(COMP_GRID - k)))
        out.append({"family": fam, "param": p, "comp_k": k,
                    "status": status,
                    "residual_sd_at_grid": float(med[j, kk]) / sd,
                    "uncompensated_change_sd":
                        float(med[j, int(np.argmin(np.abs(COMP_GRID - 1.0)))])
                        / sd})
    return pd.DataFrame(out)


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


# ============================================================
# The scenes
# ============================================================

def load_scenes():
    """The 47 scenes of the v4 stimulus set, from its manifest, with the
    manifest's grid cells (for the reproduction check)."""
    if not os.path.isfile(V4_MANIFEST_CSV):
        stop(f"{V4_MANIFEST_CSV} not found.")
    man = pd.read_csv(V4_MANIFEST_CSV)
    for c in ("scene_id", "category", "source_path", "kind", "colour_level",
              "fog_level", "chroma", "haze", "filename"):
        if c not in man.columns:
            stop(f"{V4_MANIFEST_CSV} has no {c!r} column.")
    sc = (man.groupby("scene_id", sort=True)
          .agg(category=("category", "first"),
               source_path=("source_path", "first"),
               n_paths=("source_path", "nunique"))
          .reset_index())
    missing = [p for p in sc.source_path if not os.path.isfile(p)]
    check("S1", len(sc) == N_SCENES_EXPECTED and not missing
          and (sc.n_paths == 1).all(),
          f"{len(sc)} scenes in the v4 manifest (expected "
          f"{N_SCENES_EXPECTED}), one source photograph each, "
          f"{len(sc) - len(missing)} on disk"
          + (f"; missing e.g. {missing[0]}" if missing else ""))
    grid = man[man.kind == "grid"].copy()
    bad = [s for s in EXCLUDE_SCENES if s not in set(sc.scene_id)]
    if bad:
        stop(f"EXCLUDE_SCENES names scenes not in the v4 manifest: {bad}")
    sc["set_aside"] = sc.scene_id.isin(EXCLUDE_SCENES)
    if EXCLUDE_SCENES:
        note("S2", f"{len(EXCLUDE_SCENES)} scene(s) set aside (rendered and "
             f"shown, left out of every pooled number): "
             + ", ".join(EXCLUDE_SCENES))
    return sc.drop(columns="n_paths"), grid


# ============================================================
# Definition (b), pass 1: the reach of every family on every scene
# ============================================================

def dims_for_b():
    dims = set()
    for a in APPROACHES.values():
        if not {"b", "r"} & set(a["definitions"]):
            continue
        dims |= {(a["contrast_down"], "down"), (a["contrast_up"], "up"),
                 (a["colour"], "down"), (a["colour"], "up")}
    return sorted(dims)


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


def spans_from_reach(reach):
    """Per scene x approach x feature: up reach, down reach, symmetric span;
    and the summary the rendered levels of (b) are set from."""
    R = reach.set_index(["scene_id", "family", "direction"])
    rows = []
    for name, a in APPROACHES.items():
        if not {"b", "r"} & set(a["definitions"]):
            continue
        for feature, fd, fu in (("contrast", a["contrast_down"],
                                 a["contrast_up"]),
                                ("colour", a["colour"], a["colour"])):
            for sid in sorted(reach.scene_id.unique()):
                dn = R.loc[(sid, fd, "down")]
                up = R.loc[(sid, fu, "up")]
                rows.append({"scene_id": sid, "approach": name,
                             "feature": feature, "family_down": fd,
                             "family_up": fu,
                             "reach_down_sd": dn.reach_sd,
                             "limit_down": dn.limit,
                             "reach_up_sd": up.reach_sd,
                             "limit_up": up.limit,
                             "span_sd": min(dn.reach_sd, up.reach_sd)})
    spans = pd.DataFrame(rows)
    spans["set_aside"] = spans.scene_id.isin(EXCLUDE_SCENES)
    summ_rows = []
    for (name, feature), g in spans[~spans.set_aside].groupby(
            ["approach", "feature"], sort=False):
        r = {"approach": name, "feature": feature, "n_scenes": len(g),
             "span_median_scene": float(g.span_sd.median()),
             "span_rendered_q": float(g.span_sd.quantile(SPAN_QUANTILE)),
             "span_rendered": (float(B_SPAN_SD) if B_SPAN_SD is not None
                               else float(g.span_sd.quantile(SPAN_QUANTILE))),
             "share_reach_rendered": (float((g.span_sd >= B_SPAN_SD
                                             - SOLVE_TOL_SD).mean())
                                      if B_SPAN_SD is not None else np.nan),
             "span_min": float(g.span_sd.min()),
             "span_max": float(g.span_sd.max()),
             "reach_up_median": float(g.reach_up_sd.median()),
             "reach_down_median": float(g.reach_down_sd.median()),
             "binding_side": ("up" if (g.reach_up_sd < g.reach_down_sd).mean()
                              > 0.5 else "down"),
             "limit_up_most_common": g.limit_up.mode().iat[0],
             "limit_down_most_common": g.limit_down.mode().iat[0]}
        for step in FIXED_STEPS_SD:
            r[f"share_reach_pm{step:g}"] = float((g.span_sd >= step).mean())
        summ_rows.append(r)
    return spans, pd.DataFrame(summ_rows)


# ============================================================
# Pass 2: the ladders of every set, the cells, the sheets
# ============================================================

CONTRAST_IDX = [-3, -2, -1, 0, 1, 2, 3]


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


def ladders_a(approach):
    """Definition (a): the same parameters for every scene."""
    a = APPROACHES[approach]
    LA, CA = W["levels_a"], W["comp_a"]
    names_dn = [f"p{q:02d}" for q in PCT_DOWN][::-1]       # p15, p25, p35
    names_up = [f"p{q:02d}" for q in PCT_UP]               # p65, p75, p85
    out = {}
    for feature, fd, fu in (("contrast", a["contrast_down"], a["contrast_up"]),
                            ("colour", a["colour"], a["colour"])):
        lad = []
        for i in CONTRAST_IDX:
            if i == 0:
                lad.append(_level(0, "orig", (), 0.0, False, "original"))
                continue
            fam, direction = (fd, "down") if i < 0 else (fu, "up")
            name = names_dn[i + 3] if i < 0 else names_up[i - 1]
            p, status, tsd = LA[(fam, direction, name)]
            steps = [(fam, p)]
            kc, kcap = np.nan, False
            if feature == "contrast" and a["compensate"]:
                kc, kstat = CA[(fam, p)]
                kcap = kstat != "ok"
                steps.append(("chroma", kc))
            lad.append(_level(i, name, steps, tsd, status != "ok", status,
                              fam, p, kc, kcap))
        out[feature] = lad
    return out


def ladders_c(approach):
    """Definition (c): fixed Pillow factors (Kusztor-style)."""
    a = APPROACHES[approach]
    out = {}
    fs = sorted(KUSZTOR_FACTORS)
    for feature, fam in (("contrast", a["contrast_down"]),
                         ("colour", a["colour"])):
        lad = []
        for i, f in zip(CONTRAST_IDX, fs):
            if f == 1.0:
                lad.append(_level(i, "x1.0", (), np.nan, False, "original"))
            else:
                lad.append(_level(i, f"x{f:.1f}", [(fam, f)], np.nan, False,
                                  "fixed factor", fam, f))
        out[feature] = lad
    return out


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


def _sheet(sid, approach, d, lad, tiles, labels, path):
    """7 x 7 contact sheet at SHEET_TILE_PX per cell: contrast down the
    rows (most increased at the top), colour across (most vivid left);
    the cross through the original is framed. Values are drawn in a label
    band, never on the picture."""
    t, pad, lab_h, head, left = SHEET_TILE_PX, 4, 30, 40, 70
    n = len(CONTRAST_IDX)
    Wd = left + n * (t + pad) + pad
    Ht = head + n * (t + lab_h + pad) + pad
    sheet = Image.new("RGB", (Wd, Ht), (127, 127, 127))
    dr = ImageDraw.Draw(sheet)
    dr.text((6, 4), f"{sid}   {approach}   definition "
                    f"{DEFINITION_NAMES[d]}   rows: contrast (top = most "
                    "increased); columns: colour (left = most vivid)",
            fill=(10, 10, 10))
    cols = lad["colour"][::-1]
    rows_ = lad["contrast"][::-1]
    for ci, cl in enumerate(cols):
        x = left + ci * (t + pad) + pad
        dr.text((x + 2, 22), f"colour {cl['name']}" + (" CAPPED" if
                                                      cl["capped"] else ""),
                fill=(150, 0, 0) if cl["capped"] else (10, 10, 10))
    for ri, rl in enumerate(rows_):
        y = head + ri * (t + lab_h + pad) + pad
        dr.text((4, y + t // 2), f"{rl['name']}", fill=(10, 10, 10))
        if rl["capped"]:
            dr.text((4, y + t // 2 + 12), "CAPPED", fill=(150, 0, 0))
        for ci, cl in enumerate(cols):
            x = left + ci * (t + pad) + pad
            key = (rl["index"], cl["index"])
            im = Image.fromarray(tiles[key]).resize((t, t), Image.LANCZOS)
            sheet.paste(im, (x, y))
            if rl["index"] == 0 or cl["index"] == 0:
                dr.rectangle([x - 2, y - 2, x + t + 1, y + t + 1],
                             outline=(255, 255, 255), width=2)
            txt, capped = labels[key]
            dr.text((x + 2, y + t + 2), txt,
                    fill=(150, 0, 0) if capped else (10, 10, 10))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)


def _render_task(task):
    sid, path, tables, v4cells = task
    base_dir = W["base_dir"]
    sd = W["sd"]
    sq = load_square(path)
    ctx0 = ctx_of(sq)
    ctx0["square"] = sq           # lets apply_step keep sq's gamut ceiling
    aux = scene_aux(sq)
    base = measure_image(sq, aux)
    out = {"scene_id": sid, "base": {"scene_id": sid, **base}, "cells": [],
           "ladders": [], "repro": [], "identity": [], "pillow": [],
           "v4cells": []}

    # ---- the ladders of every set ----
    sets = {}
    for name, a in APPROACHES.items():
        for d in a["definitions"]:
            if d == "a":
                sets[(name, d)] = ladders_a(name)
            elif d in ("b", "r"):
                sets[(name, d)] = ladders_b(name, sq, ctx0, base, tables, d)
            elif d == "c":
                sets[(name, d)] = ladders_c(name)
    for (name, d), lad in sets.items():
        for feature in ("contrast", "colour"):
            for lv in lad[feature]:
                out["ladders"].append({
                    "scene_id": sid, "approach": name,
                    "definition": DEFINITION_NAMES[d], "feature": feature,
                    "index": lv["index"], "name": lv["name"],
                    "family": lv["family"], "param": lv["param"],
                    "comp_k": lv["comp_k"], "target_sd": lv["target_sd"],
                    "capped": lv["capped"], "comp_capped": lv["comp_capped"],
                    "comp_residual_sd": lv["comp_residual_sd"],
                    "status": lv["status"],
                    "steps": json.dumps(lv["steps"])})

    # ---- render, measure, save every cell ----
    done = {}            # full steps -> (measurements, relative file)
    for (name, d), lad in sets.items():
        cache, tiles, labels = {}, {}, {}
        for cl in lad["contrast"]:
            img_c, rec_c = render_steps(sq, ctx0, list(cl["steps"]), cache)
            for sl in lad["colour"]:
                rel = cell_file(name, d, sid, cl["index"], sl["index"])
                dst = os.path.join(base_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                key = cl["steps"] + sl["steps"]
                if key in done:
                    # the same edit already rendered for another set:
                    # copy its file and its measurements
                    v, src = done[key]
                    shutil.copyfile(os.path.join(base_dir, src), dst)
                    with Image.open(dst) as im:
                        u8 = np.asarray(im.convert("RGB"), np.uint8)
                else:
                    u8, rec = img_c, rec_c
                    for fam, p in sl["steps"]:
                        u8, r = apply_step(fam, u8, float(p), ctx0)
                        rec = merge_clip(rec, r)
                    v = measure_image(u8, aux)
                    v.update(clip_shares(rec))
                    Image.fromarray(u8).save(dst)
                    done[key] = (v, rel)
                dsat = (v[FEAT_SAT] - base[FEAT_SAT]) / sd[FEAT_SAT]
                dcon = (v[FEAT_CON] - base[FEAT_CON]) / sd[FEAT_CON]
                cap = cl["capped"] or sl["capped"]
                tiles[(cl["index"], sl["index"])] = u8
                labels[(cl["index"], sl["index"])] = (
                    f"S{dsat:+.2f} C{dcon:+.2f}" + (" CAPPED" if cap else ""),
                    cap)
                out["cells"].append({
                    "scene_id": sid, "approach": name,
                    "definition": DEFINITION_NAMES[d],
                    "cx": cl["index"], "sx": sl["index"],
                    "contrast_name": cl["name"], "colour_name": sl["name"],
                    "contrast_family": cl["family"],
                    "contrast_param": cl["param"], "comp_k": cl["comp_k"],
                    "colour_family": sl["family"],
                    "colour_param": sl["param"],
                    "contrast_target_sd": cl["target_sd"],
                    "colour_target_sd": sl["target_sd"],
                    "contrast_capped": cl["capped"],
                    "colour_capped": sl["capped"], "capped": cap,
                    "contrast_status": cl["status"],
                    "colour_status": sl["status"],
                    "filename": rel, **v})
        _sheet(sid, name, d, lad, tiles, labels,
               os.path.join(base_dir, sheet_file(name, d, sid)))

    # ---- identity: every edit at its "no change" setting, no short cut --
    for fam, F in FAMILIES.items():
        p = F["identity"]
        ctx = dict(ctx0, cmax=cmax_of_img(sq)) if fam == "chroma" else ctx0
        x, _ = fam_call(fam, sq, p, ctx)
        u8 = F["render"](sq, p) if F["render"] else to_u8(x)
        v = measure(u8, [FEAT_SAT, FEAT_CON])
        dd = np.abs(u8.astype(np.int16) - sq.astype(np.int16))
        out["identity"].append({
            "scene_id": sid, "family": fam, "param": p,
            "max_abs_level": int(dd.max()),
            "share_pixels_changed": float((dd > 0).any(axis=2).mean()),
            "d_sat_sd": (v[FEAT_SAT] - base[FEAT_SAT]) / sd[FEAT_SAT],
            "d_con_sd": (v[FEAT_CON] - base[FEAT_CON]) / sd[FEAT_CON]})

    # ---- Pillow against its float model ----
    # Pillow truncates where the float model rounds; at a factor a hair
    # from 1 that is a shift of about half a level on every pixel, i.e.
    # Pillow's output jumps at the original. mean_shift_vs_original
    # records it.
    for fam in ("pil_contrast", "pil_color"):
        for p in (0.6, 1.4, 1.0 - 1e-6, 1.0 + 1e-6):
            x = FAMILIES[fam]["fn"](sq, p, ctx0)
            r = FAMILIES[fam]["render"](sq, p)
            dd = np.abs(r.astype(np.int16) - to_u8(x).astype(np.int16))
            out["pillow"].append({
                "scene_id": sid, "family": fam, "param": p,
                "max_abs_level": int(dd.max()),
                "share_differing": float((dd > 0).any(axis=2).mean()),
                "mean_shift_vs_original": float(
                    (r.astype(np.float64) - sq).mean())})

    # ---- v4 reproduction, and the v4 grid measured as a baseline ----
    for c in v4cells:
        k, h = float(c["chroma"]), float(c["haze"])
        img_v4 = render_cell(sq, ctx0, k, h, "haze_eq")        # verbatim v4
        steps = [(f, p) for f, p in (("haze_eq", h), ("chroma", k))
                 if p != FAMILIES[f]["identity"]]
        img_me, rec = render_steps(sq, ctx0, steps)
        rec_row = {"scene_id": sid, "colour_level": c["colour_level"],
                   "fog_level": c["fog_level"], "chroma": k, "haze": h,
                   "filename": c["filename"],
                   "generic_equals_v4_renderer":
                       bool(np.array_equal(img_v4, img_me))}
        fp = os.path.join(V4_STIMULI_DIR, c["filename"])
        if os.path.isfile(fp):
            with Image.open(fp) as im:
                ref = np.asarray(im.convert("RGB"), np.uint8)
            dd = np.abs(ref.astype(np.int16) - img_v4.astype(np.int16))
            buf = io.BytesIO()
            Image.fromarray(img_v4).save(buf, format="PNG")
            with open(fp, "rb") as f:
                fb = f.read()
            rec_row.update({
                "pixels_identical": bool(dd.max() == 0),
                "max_abs_level": int(dd.max()),
                "share_differing": float((dd > 0).any(axis=2).mean()),
                "bytes_identical": hashlib.md5(buf.getvalue()).hexdigest()
                == hashlib.md5(fb).hexdigest()})
        else:
            rec_row.update({"pixels_identical": None, "max_abs_level": None,
                            "share_differing": None, "bytes_identical": None})
        out["repro"].append(rec_row)
        v = measure_image(img_v4, aux)
        v.update(clip_shares(rec))
        out["v4cells"].append({"scene_id": sid,
                               "colour_level": c["colour_level"],
                               "fog_level": c["fog_level"],
                               "chroma": k, "haze": h,
                               "filename": os.path.join(V4_STIMULI_DIR,
                                                        c["filename"]),
                               **v})
    return out


# ============================================================
# Aggregation: changes, cross-talk, artefacts, envelope
# ============================================================

V4_FOG_IDX = {"h=0": 0, "p35": -1, "p25": -2, "p15": -3, "p10": -4,
              "p05": -5}
V4_COL_IDX = {"p75": 2, "p65": 1, "k=1": 0, "p25": -1, "p10": -2,
              "p05": -3, "p01": -4}
ARTEFACT_COLS = ["halo_share", "noise_ratio", "band_new_empty",
                 "band_comb_diff"]


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


def cross_cells(cells):
    """The two ladders through the original: contrast levels at the
    original colour (sx = 0) and colour levels at the original contrast."""
    c = cells[(cells.sx == 0)].assign(feature="contrast", index=lambda d: d.cx)
    s = cells[(cells.cx == 0)].assign(feature="colour", index=lambda d: d.sx)
    return pd.concat([c, s], ignore_index=True)


def ladder_achieved(ladders, cells):
    X = cross_cells(cells)
    X["achieved_sd"] = np.where(X.feature == "contrast", X.d_con_sd,
                                X.d_sat_sd)
    X["cross_sd"] = np.where(X.feature == "contrast", X.d_sat_sd,
                             X.d_con_sd)
    m = X[["scene_id", "approach", "definition", "feature", "index",
           "achieved_sd", "cross_sd"]]
    return ladders.merge(m, on=["scene_id", "approach", "definition",
                                "feature", "index"], how="left")


def crosstalk_table(lad):
    rows = []
    g = lad[lad["index"] != 0].groupby(["approach", "definition", "feature",
                                        "index", "name"], sort=False)
    for (a, d, f, i, n), x in g:
        own, cr = x.achieved_sd.to_numpy(), x.cross_sd.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(cr / own)
        ok = np.isfinite(ratio)
        rows.append({"approach": a, "definition": d, "feature": f,
                     "index": i, "level": n, "n_scenes": len(x),
                     "own_median_sd": float(np.median(own)),
                     "own_min_sd": float(own.min()),
                     "own_max_sd": float(own.max()),
                     "cross_median_sd": float(np.median(cr)),
                     "ratio_of_medians": float(abs(np.median(cr)
                                                   / np.median(own)))
                     if np.median(own) != 0 else np.nan,
                     "ratio_median": float(np.median(ratio[ok]))
                     if ok.any() else np.nan,
                     "ratio_worst_scene": float(np.max(ratio[ok]))
                     if ok.any() else np.nan,
                     "n_capped": int(x.capped.sum())})
    return pd.DataFrame(rows)


def artefact_table(cells, v4cells):
    """Per set and level (cross cells), and over all 49 cells: medians and
    maxima of the artefact and clipping measures. The v4 grid, measured
    by the same code, is the comparison."""
    cols = (ARTEFACT_COLS + [f"clip_{d:g}lv" for d in CLIP_DEPTHS]
            + ["gamut_mapped"]
            + [c for c in cells.columns if c.startswith("d_")
               and c not in ("d_sat", "d_con", "d_sat_sd", "d_con_sd")])
    rows = []
    X = cross_cells(cells)
    for (a, d, f, i), x in X[X["index"] != 0].groupby(
            ["approach", "definition", "feature", "index"], sort=False):
        r = {"approach": a, "definition": d, "group": f"{f} {i:+d}"}
        for c in cols:
            if c in x.columns:
                r[f"{c}_median"] = float(x[c].median())
                r[f"{c}_max"] = float(x[c].max())
        rows.append(r)
    for (a, d), x in cells.groupby(["approach", "definition"], sort=False):
        r = {"approach": a, "definition": d, "group": "all 49 cells"}
        for c in cols:
            if c in x.columns:
                r[f"{c}_median"] = float(x[c].median())
                r[f"{c}_max"] = float(x[c].max())
        rows.append(r)
    if len(v4cells):
        r = {"approach": "V4_ref", "definition": "v4", "group": "all 42 cells"}
        for c in cols:
            if c in v4cells.columns:
                r[f"{c}_median"] = float(v4cells[c].median())
                r[f"{c}_max"] = float(v4cells[c].max())
        rows.append(r)
    return pd.DataFrame(rows)


# ============================================================
# Normality check
# ============================================================

def classify_pairs(sat_a, con_a, sat_b, con_b, same_row):
    """v4's classify_pair (analyse_pair_capacity_v5.classify_pair_v4),
    vectorised, then v5's filters. Returns type labels ('' = no trial)."""
    d_sat = sat_a - sat_b
    d_fog = FOG_SIGN * (con_a - con_b)
    a_s, a_f = np.abs(d_sat), np.abs(d_fog)
    t = np.full(len(d_sat), "", dtype=object)
    sat_only = same_row & (a_s >= V4_MIN_DSAT_SAT_ONLY)
    fog_only = ~same_row & (a_s <= V4_PAIR_SAT_TOL) & (a_f >= V4_MIN_DFOG_FOG_ONLY)
    mixed = (~same_row & ~fog_only & (a_f >= V4_MIN_DFOG_MIXED)
             & (a_s >= V4_MIN_DSAT_MIXED))
    t[sat_only] = "sat_only"
    t[fog_only] = "fog_only"
    t[mixed & ((d_fog > 0) == (d_sat > 0))] = "conflict"
    t[mixed & ((d_fog > 0) != (d_sat > 0))] = "together"
    # v5's filters
    t[(t == "sat_only") & ((a_s < SAT_ONLY_MIN_DSAT)
                           | (np.abs(con_a - con_b) > SAT_ONLY_CON_TOL))] = ""
    t[(t == "fog_only") & (a_s > FOG_ONLY_SAT_TOL)] = ""
    if BALANCED_CONFLICT_ONLY:
        with np.errstate(divide="ignore", invalid="ignore"):
            bal = (a_s / SD_SAT_V5) / (a_f / SD_FOG_V5)
        t[(t == "conflict") & ~((bal >= V4_BALANCE_LO)
                                & (bal <= V4_BALANCE_HI))] = ""
    return t, d_sat, d_fog


def scene_pairs(x):
    """Every ordered pair of distinct images of one scene in one set."""
    n = len(x)
    ia, ib = np.where(~np.eye(n, dtype=bool))
    A = {c: x[c].to_numpy()[ia] for c in x.columns}
    Bv = {c: x[c].to_numpy()[ib] for c in x.columns}
    t, d_sat, d_fog = classify_pairs(A[FEAT_SAT], A[FEAT_CON],
                                     Bv[FEAT_SAT], Bv[FEAT_CON],
                                     A["cx"] == Bv["cx"])
    out = {"cx_a": A["cx"], "sx_a": A["sx"], "cx_b": Bv["cx"],
           "sx_b": Bv["sx"], "type": t, "d_sat": d_sat, "d_fog": d_fog,
           "pred_sat": -d_sat / SD_SAT_V5, "pred_fog": d_fog / SD_FOG_V5,
           "d_con_sd_a": A["d_con_sd"], "d_con_sd_b": Bv["d_con_sd"],
           "d_sat_sd_a": A["d_sat_sd"], "d_sat_sd_b": Bv["d_sat_sd"]}
    for ref in ("corpus", "orig"):
        for part in ("", "_sat", "_con"):
            c = f"dist_{ref}{part}"
            out[f"N_{ref}{part}"] = Bv[c] - A[c]
    return pd.DataFrame(out)


def _corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan
    p = float(np.corrcoef(a, b)[0, 1])
    s = float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])
    return p, s


def normality(cells, v4cells):
    """Per set: every ordered within-scene pair, typed by v5's rules; the
    correlation of the 'closer to normal' term with each direction term,
    over all typed pairs, per type, and over all ordered pairs; and the
    pairs that straddle the original."""
    keep = ["scene_id", "cx", "sx", FEAT_SAT, FEAT_CON, "d_sat_sd",
            "d_con_sd", "dist_corpus", "dist_corpus_sat", "dist_corpus_con",
            "dist_orig", "dist_orig_sat", "dist_orig_con"]
    groups = [((a, d), x[keep]) for (a, d), x in
              cells.groupby(["approach", "definition"], sort=False)]
    if len(v4cells):
        v = v4cells.copy()
        v = v[~v.colour_level.isin(V4_DROP_COLOUR)
              & ~v.fog_level.isin(V4_DROP_FOG)]
        v["cx"] = v.fog_level.map(V4_FOG_IDX)
        v["sx"] = v.colour_level.map(V4_COL_IDX)
        groups.append((("V4_ref", "v4 (v5 pair rules, no h=0, no p65)"),
                       v[keep]))
    pairs, summ, strad = [], [], []
    for (a, d), x in groups:
        P_ = pd.concat([scene_pairs(s.drop(columns="scene_id")
                                    .reset_index(drop=True))
                        .assign(scene_id=sid)
                        for sid, s in x.groupby("scene_id")],
                       ignore_index=True)
        P_["approach"], P_["definition"] = a, d
        typed = P_[P_.type != ""]
        pairs.append(typed)
        subsets = [("all typed pairs", typed), ("all ordered pairs", P_)]
        subsets += [(t, typed[typed.type == t]) for t in
                    ("sat_only", "fog_only", "together", "conflict")]
        for sub, y in subsets:
            for ref in ("corpus", "orig"):
                for part, pred in (("", "pred_sat"), ("", "pred_fog"),
                                   ("_sat", "pred_sat"), ("_con", "pred_fog")):
                    p, s = _corr(y[f"N_{ref}{part}"].to_numpy(float),
                                 y[pred].to_numpy(float))
                    summ.append({"approach": a, "definition": d,
                                 "subset": sub, "n_pairs": len(y),
                                 "reference": ref,
                                 "normality_term": "combined" if not part
                                 else part[1:],
                                 "predictor": pred, "pearson": p,
                                 "spearman": s})
        u = typed[(typed.cx_a * 10 + typed.sx_a) < (typed.cx_b * 10
                                                    + typed.sx_b)]
        for t, y in [("all typed", u)] + [(t, u[u.type == t]) for t in
                                          ("sat_only", "fog_only", "together",
                                           "conflict")]:
            strad.append({
                "approach": a, "definition": d, "type": t,
                "n_unordered_pairs": len(y),
                "straddle_contrast_by_level": int((y.cx_a * y.cx_b < 0).sum()),
                "straddle_colour_by_level": int((y.sx_a * y.sx_b < 0).sum()),
                "straddle_contrast_measured": int(
                    ((y.d_con_sd_a * y.d_con_sd_b < 0)
                     & (y.d_con_sd_a.abs() > STRADDLE_TOL_SD)
                     & (y.d_con_sd_b.abs() > STRADDLE_TOL_SD)).sum()),
                "straddle_colour_measured": int(
                    ((y.d_sat_sd_a * y.d_sat_sd_b < 0)
                     & (y.d_sat_sd_a.abs() > STRADDLE_TOL_SD)
                     & (y.d_sat_sd_b.abs() > STRADDLE_TOL_SD)).sum()),
                "pairs_with_the_original": int(
                    (((y.cx_a == 0) & (y.sx_a == 0))
                     | ((y.cx_b == 0) & (y.sx_b == 0))).sum())})
    return (pd.concat(pairs, ignore_index=True), pd.DataFrame(summ),
            pd.DataFrame(strad))


# ============================================================
# Scene fit
# ============================================================

def scene_fit(cells, v4max):
    rows = []
    for (a, d, sid), x in cells.groupby(["approach", "definition",
                                          "scene_id"], sort=False):
        r = {"approach": a, "definition": d, "scene_id": sid}
        fits = True
        for feature, idx, own in (("contrast", "cx", "d_con_sd"),
                                  ("colour", "sx", "d_sat_sd")):
            other = "sx" if idx == "cx" else "cx"
            lad = x[x[other] == 0].sort_values(idx)
            capcol = f"{feature}_capped"
            names = lad.contrast_name if feature == "contrast" \
                else lad.colour_name
            capped = [n for n, c in zip(names, lad[capcol]) if c]
            v = lad[own].to_numpy(float)
            steps = np.diff(v)
            mono = bool((steps >= MIN_STEP_SD).all())
            r[f"{feature}_all7_reached"] = not capped
            r[f"{feature}_capped_levels"] = ";".join(capped)
            r[f"{feature}_monotone"] = mono
            r[f"{feature}_smallest_step_sd"] = float(steps.min())
            r[f"{feature}_span_down_sd"] = float(v[0])
            r[f"{feature}_span_up_sd"] = float(v[-1])
            fits &= (not capped) and mono
        clip = x[f"clip_{CLIP_DEPTH:g}lv"]
        r["cells_over_clip_budget"] = int((clip > CLIP_MAX_NEW).sum())
        r["cells_outside_envelope"] = int((x.envelope_n > 0).sum())
        r["envelope_features"] = ";".join(sorted(
            {f for s in x.envelope_features for f in s.split(";") if f}))
        for c in ("halo_share", "noise_ratio", "band_new_empty"):
            r[f"{c}_max"] = float(x[c].max())
            r[f"{c}_above_v4_max"] = (bool(x[c].max() > v4max[c])
                                      if c in v4max else None)
        if B_USE_CLIP_BUDGET or d not in (DEFINITION_NAMES["b"],
                                          DEFINITION_NAMES["r"]):
            fits &= r["cells_over_clip_budget"] == 0
        r["fits"] = fits
        r["fits_and_inside_envelope"] = fits and r["cells_outside_envelope"] == 0
        rows.append(r)
    fit = pd.DataFrame(rows)
    summ = []
    for (a, d), x in fit.groupby(["approach", "definition"], sort=False):
        summ.append({
            "approach": a, "definition": d, "n_scenes": len(x),
            "fit": int(x.fits.sum()),
            "fit_and_inside_envelope": int(x.fits_and_inside_envelope.sum()),
            "contrast_all7": int(x.contrast_all7_reached.sum()),
            "colour_all7": int(x.colour_all7_reached.sum()),
            "contrast_monotone": int(x.contrast_monotone.sum()),
            "colour_monotone": int(x.colour_monotone.sum()),
            "any_cell_over_clip_budget": int((x.cells_over_clip_budget > 0).sum()),
            "any_cell_outside_envelope": int((x.cells_outside_envelope > 0).sum()),
            "halo_above_v4_max": int(x.halo_share_above_v4_max.fillna(False).sum()),
            "noise_above_v4_max": int(x.noise_ratio_above_v4_max.fillna(False).sum()),
            "banding_above_v4_max":
                int(x.band_new_empty_above_v4_max.fillna(False).sum())})
    return fit, pd.DataFrame(summ)


# ============================================================
# The viewer
# ============================================================

VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Edit levels v5</title>
<style>
  :root { --bg: __BG__; --ink: #111; --muted: #333; --panel: #e9e9e9;
          --capped: #9b0000; --gap: __GAP__px; --px: __PX__px; }
  html, body { margin: 0; background: var(--bg); color: var(--ink);
    font-family: -apple-system, "Segoe UI", "Hiragino Sans", "Noto Sans JP",
    Roboto, Helvetica, Arial, sans-serif; font-size: 14px; }
  header { position: sticky; top: 0; z-index: 2; background: var(--panel);
    padding: 8px 12px; display: flex; flex-wrap: wrap; gap: 8px 14px;
    align-items: center; border-bottom: 1px solid #999; }
  header label { display: inline-flex; gap: 4px; align-items: center; }
  select, button { font: inherit; }
  #info { font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 12px; padding: 6px 12px; background: #d8d8d8; }
  #info .warn { color: var(--capped); font-weight: 600; }
  main { padding: 16px 12px 40px; }
  .pair { display: flex; gap: var(--gap); justify-content: center;
    min-width: calc(2 * var(--px) + var(--gap)); }
  .panel { width: var(--px); flex: none; }
  .panel img, .row img { width: var(--px); height: var(--px); display: block;
    object-fit: fill; -webkit-user-drag: none; user-select: none; }
  .cap { font-size: 12px; line-height: 1.35; padding: 4px 0;
    font-family: ui-monospace, Menlo, Consolas, monospace; min-height: 34px; }
  .cap.capped { color: var(--capped); font-weight: 600; }
  .hidecaps .cap { visibility: hidden; }
  .row { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 8px; }
  .row .panel { width: var(--px); }
  table.grid { border-collapse: separate; border-spacing: 4px; margin: auto; }
  table.grid td { vertical-align: top; }
  table.grid img { display: block; object-fit: fill; }
  table.grid .cross img { outline: 2px solid #fff; }
  table.grid .hd { font-size: 12px; text-align: center; }
  table.grid .rh { font-size: 12px; white-space: nowrap; padding-right: 6px; }
  .missing { background: #6f6f6f; }
  #title { text-align: center; margin: 0 0 12px; font-size: 15px; }
  .note { max-width: 980px; margin: 18px auto 0; font-size: 12px;
    color: var(--muted); }
</style>
</head>
<body>
<header>
  <label>Mode <select id="mode">
    <option value="pairs">Pairs at 512 px</option>
    <option value="ladder">Ladder at 512 px</option>
    <option value="grid">7 x 7 grid</option>
    <option value="compare">Approaches side by side</option>
  </select></label>
  <label>Scene <select id="scene"></select></label>
  <label>Approach <select id="approach"></select></label>
  <label>Definition <select id="def"></select></label>
  <label>Feature <select id="feature">
    <option value="contrast">contrast</option>
    <option value="colour">colour</option></select></label>
  <label id="pairbox">Pair <select id="pair"></select></label>
  <label id="cellbox">Cell px <select id="cellpx">
    <option>100</option><option>150</option><option selected>200</option>
    <option>240</option></select></label>
  <button id="swap" title="s">Swap sides</button>
  <label><input type="checkbox" id="caps" checked> Captions (c)</label>
</header>
<div id="info"></div>
<main id="main"></main>
<div class="note">Keys: left/right = previous/next pair or level, up/down =
previous/next scene, s = swap sides, c = captions. Images are drawn at their
CSS size with no scaling (as in the experiment: cut off, never shrunk), on the
experiment's grey surround. Captions give the measured change from the
scene's original in corpus SD: S = sat_mean, C = contrast_local. CAPPED = the
level could not reach its target and is shown at its cap.</div>
<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
const IDX = [-3, -2, -1, 0, 1, 2, 3];
const PAIRS = [[-3,-2],[-2,-1],[-1,0],[0,1],[1,2],[2,3],[-1,1],[-2,2],[-3,3]];
let swapped = false;

function ix(i) { return i === 0 ? "0" : (i < 0 ? "n" : "p") + Math.abs(i); }
function file(a, d, s, cx, sx) {
  return `images/${a}/${d}/${s}/c${ix(cx)}_s${ix(sx)}.png`;
}
function setKey() { return $("approach").value + "|" + $("def").value; }
function cell(set, s, cx, sx) {
  const c = set.cells[s]; if (!c) return null;
  return c[(cx + 3) * 7 + (sx + 3)];
}
function has(cx, sx) { return DATA.meta.cells === "all" || cx === 0 || sx === 0; }
function caption(set, s, cx, sx) {
  const v = cell(set, s, cx, sx); if (!v) return ["", false];
  const nm = `contrast ${set.contrast_names[cx + 3]} / colour ${set.colour_names[sx + 3]}`;
  const t = `${nm}\nS ${v[0] >= 0 ? "+" : ""}${v[0].toFixed(2)} SD   C ${v[1] >= 0 ? "+" : ""}${v[1].toFixed(2)} SD${v[2] ? "   CAPPED" : ""}`;
  return [t, !!v[2]];
}
function panel(a, d, set, s, cx, sx, px) {
  const div = document.createElement("div"); div.className = "panel";
  if (px) div.style.width = px + "px";
  const img = document.createElement("img");
  img.src = file(a, d, s, cx, sx); img.alt = "";
  img.width = px || DATA.meta.image_px; img.height = px || DATA.meta.image_px;
  if (px) { img.style.width = px + "px"; img.style.height = px + "px"; }
  const cap = document.createElement("div");
  const [t, c] = caption(set, s, cx, sx);
  cap.className = "cap" + (c ? " capped" : ""); cap.textContent = t;
  cap.style.whiteSpace = "pre-wrap";
  div.append(img, cap); return div;
}
function fillSelect(sel, items, keep) {
  const old = sel.value; sel.innerHTML = "";
  for (const [v, t] of items) { const o = document.createElement("option");
    o.value = v; o.textContent = t; sel.append(o); }
  if (keep && items.some(([v]) => v === old)) sel.value = old;
}
function info() {
  const need = 2 * DATA.meta.image_px + DATA.meta.gap_px + 64;
  const w = window.innerWidth, h = window.innerHeight;
  const warn = w < need ? ` <span class="warn">window narrower than ${need} px: the pair is cut off at the right</span>` : "";
  $("info").innerHTML = `devicePixelRatio ${window.devicePixelRatio} (an image of 512 CSS px covers ${Math.round(512 * window.devicePixelRatio)} device px) | window ${w} x ${h} | screen ${screen.width} x ${screen.height} | run ${DATA.meta.hash}${DATA.meta.pack ? " (viewer pack)" : ""}${warn}`;
}
function render() {
  info();
  const a = $("approach").value, d = $("def").value, s = $("scene").value;
  const set = DATA.sets[setKey()];
  const f = $("feature").value, mode = $("mode").value;
  const main = $("main"); main.innerHTML = "";
  document.body.classList.toggle("hidecaps", !$("caps").checked);
  $("pairbox").style.display = mode === "pairs" ? "" : "none";
  $("cellbox").style.display = (mode === "grid" || mode === "compare") ? "" : "none";
  $("swap").style.display = mode === "pairs" ? "" : "none";
  const at = (i) => f === "contrast" ? [i, 0] : [0, i];
  const h = document.createElement("div"); h.id = "title";
  if (mode === "pairs") {
    let [i, j] = PAIRS[+$("pair").value];
    if (swapped) [i, j] = [j, i];
    h.textContent = `${s} | ${a} | ${d} | ${f}: level ${i >= 0 ? "+" : ""}${i} vs ${j >= 0 ? "+" : ""}${j}`;
    const p = document.createElement("div"); p.className = "pair";
    p.append(panel(a, d, set, s, ...at(i)), panel(a, d, set, s, ...at(j)));
    main.append(h, p);
  } else if (mode === "ladder") {
    h.textContent = `${s} | ${a} | ${d} | ${f} ladder, left = most reduced`;
    const r = document.createElement("div"); r.className = "row";
    for (const i of IDX) r.append(panel(a, d, set, s, ...at(i)));
    main.append(h, r);
  } else if (mode === "grid") {
    const px = +$("cellpx").value;
    h.textContent = `${s} | ${a} | ${d} | rows: contrast (top = most increased), columns: colour (left = most vivid)`;
    const t = document.createElement("table"); t.className = "grid";
    const hr = t.insertRow(); hr.insertCell();
    for (const sx of [...IDX].reverse()) { const c = hr.insertCell();
      c.className = "hd"; c.textContent = "colour " + set.colour_names[sx + 3]; }
    for (const cx of [...IDX].reverse()) {
      const r = t.insertRow(); const rh = r.insertCell(); rh.className = "rh";
      rh.textContent = "contrast " + set.contrast_names[cx + 3];
      for (const sx of [...IDX].reverse()) {
        const c = r.insertCell();
        if (cx === 0 || sx === 0) c.className = "cross";
        if (has(cx, sx)) c.append(panel(a, d, set, s, cx, sx, px));
        else { const m = document.createElement("div"); m.className = "missing";
          m.style.width = m.style.height = px + "px"; c.append(m); }
      }
    }
    main.append(h, t);
  } else {
    const px = +$("cellpx").value;
    h.textContent = `${s} | definition ${d} | ${f} ladders of every approach, left = most reduced`;
    const t = document.createElement("table"); t.className = "grid";
    for (const ap of DATA.approaches) {
      const k = ap.name + "|" + d; if (!DATA.sets[k]) continue;
      const r = t.insertRow(); const rh = r.insertCell(); rh.className = "rh";
      rh.textContent = ap.name;
      for (const i of IDX) r.insertCell().append(
        panel(ap.name, d, DATA.sets[k], s, ...at(i), px));
    }
    main.append(h, t);
  }
}
function syncDefs() {
  const ap = DATA.approaches.find((x) => x.name === $("approach").value);
  fillSelect($("def"), ap.defs.map((x) => [x, x]), true);
}
function init() {
  fillSelect($("scene"), DATA.scenes.map((x) =>
    [x, x + ((DATA.meta.aside || []).includes(x) ? "  (set aside)" : "")]));
  fillSelect($("approach"), DATA.approaches.map((x) => [x.name, x.name]));
  syncDefs();
  fillSelect($("pair"), PAIRS.map((p, k) => [String(k),
    `${p[0] >= 0 ? "+" : ""}${p[0]} vs ${p[1] >= 0 ? "+" : ""}${p[1]}` +
    (p[0] * p[1] < 0 ? "  (straddles the original)" : "")]));
  $("pair").value = "2";
  for (const id of ["mode", "scene", "def", "feature", "pair", "cellpx", "caps"])
    $(id).addEventListener("change", () => { swapped = false; render(); });
  $("approach").addEventListener("change", () => { syncDefs(); render(); });
  $("swap").addEventListener("click", () => { swapped = !swapped; render(); });
  window.addEventListener("resize", info);
  window.addEventListener("keydown", (e) => {
    const step = (sel, k) => { const n = sel.options.length;
      sel.selectedIndex = (sel.selectedIndex + k + n) % n; };
    if (e.key === "ArrowRight") step($("pair"), 1);
    else if (e.key === "ArrowLeft") step($("pair"), -1);
    else if (e.key === "ArrowDown") step($("scene"), 1);
    else if (e.key === "ArrowUp") step($("scene"), -1);
    else if (e.key === "s") swapped = !swapped;
    else if (e.key === "c") $("caps").checked = !$("caps").checked;
    else return;
    e.preventDefault(); render();
  });
  render();
}
init();
</script>
</body>
</html>
"""


def viewer_data(cells, scenes, cells_mode, pack):
    sets = {}
    for (a, d), x in cells.groupby(["approach", "definition"], sort=False):
        x = x[x.scene_id.isin(scenes)]
        cn = (x[x.sx == 0].drop_duplicates("cx").sort_values("cx")
              .contrast_name.tolist())
        sn = (x[x.cx == 0].drop_duplicates("sx").sort_values("sx")
              .colour_name.tolist())
        per = {}
        for sid, y in x.groupby("scene_id"):
            y = y.set_index(["cx", "sx"])
            arr = []
            for cx in CONTRAST_IDX:
                for sx in CONTRAST_IDX:
                    r = y.loc[(cx, sx)]
                    arr.append([round(float(r.d_sat_sd), 3),
                                round(float(r.d_con_sd), 3),
                                int(bool(r.capped))])
            per[sid] = arr
        sets[f"{a}|{d}"] = {"contrast_names": cn, "colour_names": sn,
                            "cells": per}
    appr = []
    for a in APPROACHES:
        ds = [DEFINITION_NAMES[d] for d in APPROACHES[a]["definitions"]]
        appr.append({"name": a, "defs": ds})
    return {"meta": {"hash": RUN["meta"].get("settings_hash", ""),
                     "image_px": IMAGE_SIZE, "gap_px": VIEWER_GAP_PX,
                     "cells": cells_mode, "pack": pack,
                     "aside": [s for s in scenes if s in EXCLUDE_SCENES]},
            "approaches": appr, "scenes": list(scenes), "sets": sets}


def write_viewer(path, data):
    html = (VIEWER_HTML.replace("__BG__", VIEWER_BG)
            .replace("__GAP__", str(VIEWER_GAP_PX))
            .replace("__PX__", str(IMAGE_SIZE))
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def pack_scenes(bases):
    if VIEWER_PACK_SCENES:
        bad = [s for s in VIEWER_PACK_SCENES if s not in set(bases.scene_id)]
        if bad:
            stop(f"VIEWER_PACK_SCENES not among the scenes: {bad}")
        return list(VIEWER_PACK_SCENES)
    b = (bases[~bases.scene_id.isin(EXCLUDE_SCENES)]
         .sort_values("lum_mean").reset_index(drop=True))
    n = min(VIEWER_PACK_N, len(b))
    return list(b.scene_id.iloc[np.unique(np.linspace(0, len(b) - 1, n)
                                          .astype(int))])


def write_viewers(cells, bases):
    scenes = sorted(cells.scene_id.unique())
    write_viewer(B("viewer.html"), viewer_data(cells, scenes, "all", False))
    if not VIEWER_PACK:
        return None
    ps = pack_scenes(bases)
    root = B("viewer_pack")
    sub = cells[cells.scene_id.isin(ps)]
    if VIEWER_PACK_CELLS == "cross":
        sub_files = sub[(sub.cx == 0) | (sub.sx == 0)]
    else:
        sub_files = sub
    for rel in sub_files.filename:
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(B(rel), dst)
    write_viewer(os.path.join(root, "viewer.html"),
                 viewer_data(cells, ps, VIEWER_PACK_CELLS, True))
    size = sum(os.path.getsize(os.path.join(root, r))
               for r in sub_files.filename) / 1e6
    zpath = None
    if VIEWER_PACK_ZIP:
        zpath = B("viewer_pack.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for dp, _, fs in os.walk(root):
                for f in fs:
                    p = os.path.join(dp, f)
                    z.write(p, os.path.relpath(p, os.path.dirname(root)))
    return {"scenes": ps, "n_images": len(sub_files), "mb": size,
            "zip": zpath}


# ################################################################
# VERBATIM from score_stimuli_on_axes_v4.py (lines 429-505): statistics helpers
# ################################################################
def sha256(path, chunk=1 << 24):
    """From 03_encode_images.py, sha256()."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fingerprint(idx):
    """From 03_encode_images.py, fingerprint() (also 06's S1)."""
    h = hashlib.sha1()
    for k, p in zip(idx["image_key"], idx["path"]):
        h.update(f"{k}\t{p}\n".encode("utf-8"))
    return h.hexdigest()


def unit(v):
    """From 09c_rescore_sweep_v4.py, unit()."""
    return v / np.linalg.norm(v)


def ci(a, level=CI_LEVEL, axis=0):
    """From 09c_rescore_sweep_v4.py, ci(): percentile interval."""
    lo = (1 - level) / 2 * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanpercentile(a, lo, axis=axis),
                np.nanpercentile(a, 100 - lo, axis=axis))


def boot_weights(n, rng):
    """Resampling weights, one row per resample: the multinomial
    weights of 09c_rescore_sweep_v4.py, main() ("effects")."""
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n)
                     for _ in range(N_BOOT)]).astype(float) / n


def summ(X, W):
    """Mean over the first axis (scenes), interval from the resampling
    weights W, and the share of scenes moving in the mean's direction
    (09c's same_sign)."""
    m = X.mean(0)
    lo, hi = ci(W @ X)
    ss = (np.sign(X) == np.sign(m)).mean(0)
    return m, lo, hi, ss


def classify(lo, hi, sign=1):
    """'shown' / 'not shown' / 'reversed' for an interval whose
    intended side of 0 is `sign`."""
    if sign < 0:
        lo, hi = -hi, -lo
    if lo > 0:
        return "shown"
    if hi < 0:
        return "reversed"
    return "not shown"


def verdict(classes):
    if any(c == "reversed" for c in classes):
        return "FAIL"
    if all(c == "shown" for c in classes):
        return "PASS"
    return "PARTIAL"


def fci(m, lo, hi):
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def mark(lo, hi):
    return "*" if (lo > 0 or hi < 0) else " "


# ################################################################
# VERBATIM from score_stimuli_on_axes_v4.py (lines 584-843): the un2CLIP encoder
# ################################################################
def load_un2clip_state_dict(path):
    """From 03_encode_images.py, load_un2clip_state_dict()."""
    import torch
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        raw = raw["model"]
    out = {}
    for k, v in raw.items():
        if k.startswith("visual."):
            k = k[len("visual."):]
        out[k] = v
    return out


def remap_openai_to_hf(sd):
    """From 03_encode_images.py, remap_openai_to_hf() (verbatim)."""
    d_model = sd["class_embedding"].shape[0]
    hf, unused = {}, []

    def block(rest, v):
        n, sub = rest.split(".", 1)
        p = f"encoder.layers.{n}."
        if sub in ("attn.in_proj_weight", "attn.in_proj_bias"):
            q, k, vv = v.split(d_model, dim=0)      # rows ordered q, k, v
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


def load_model(device):
    """From 03_encode_images.py, load_model(). The failures that raise
    there are recorded as checks E1 / E2 here, and stop the run."""
    import torch
    from transformers import CLIPImageProcessor, CLIPModel
    print(f"  base model : {CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)
    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.text_model = None                 # the text tower is not needed

    print(f"  un2CLIP    : {UN2CLIP_CKPT_PATH}")
    if not os.path.exists(UN2CLIP_CKPT_PATH):
        stop(f"un2CLIP checkpoint not found: {UN2CLIP_CKPT_PATH}")
    sd = load_un2clip_state_dict(UN2CLIP_CKPT_PATH)
    hf_sd, proj, unused = remap_openai_to_hf(sd)

    # --- Check 1 (03): every weight loaded ---
    missing, unexpected = model.vision_model.load_state_dict(hf_sd,
                                                             strict=False)
    buffers = {n for n, _ in model.vision_model.named_buffers()}
    missing_w = [k for k in missing if k not in buffers]
    ok1 = not missing_w and not unexpected and not unused
    check("E1", ok1, f"weights: {len(hf_sd)} tensors loaded; missing "
          f"{len(missing_w)}, unexpected {len(unexpected)}, unused "
          f"checkpoint keys {len(unused)}"
          + ("" if ok1 else f" ({missing_w[:3]}, {list(unexpected)[:3]}, "
             f"{unused[:3]})"))

    # --- The fix of 03: use un2CLIP's projection ---
    with torch.no_grad():
        if tuple(proj.T.shape) != tuple(model.visual_projection.weight.shape):
            stop(f"proj shape {tuple(proj.shape)} does not fit "
                 f"visual_projection "
                 f"{tuple(model.visual_projection.weight.shape)}")
        std_w = model.visual_projection.weight.detach().clone()
        model.visual_projection.weight.copy_(proj.T)

    # --- Check 2 (03): projection in place ---
    w = model.visual_projection.weight.detach()
    ok2 = torch.equal(w, proj.T.to(w.dtype))
    changed = (proj.T.float() - std_w.float()).abs().max().item()
    check("E2", ok2, f"un2CLIP projection applied (differs from the "
          f"standard one by up to {changed:.2e})")

    model = model.to(device).eval()
    info = {"weights_loaded": len(hf_sd), "proj_applied": bool(ok2),
            "proj_max_diff_vs_standard": changed}
    return model, processor, sd, info


def embed(model, pixels):
    """From 03_encode_images.py, embed(): (pooled_1024,
    embedding_768_normalised). The @torch.inference_mode() decorator is
    a context manager here, so torch is imported lazily."""
    import torch
    with torch.inference_mode():
        pooled = model.vision_model(pixel_values=pixels).pooler_output
        emb = model.visual_projection(pooled)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return pooled, emb


def _to_rgb(im):
    """From 03_encode_images.py, _to_rgb()."""
    return im.convert("RGB")


class OpenAIPreprocess:
    """From 03_encode_images.py, OpenAIPreprocess: exactly
    clip._transform(n_px): Resize (bicubic, on the PIL image) ->
    CenterCrop -> RGB -> ToTensor -> Normalize."""

    def __init__(self, n_px, mean, std):
        from torchvision.transforms import (CenterCrop, Compose,
                                            InterpolationMode,
                                            Normalize, Resize, ToTensor)
        self.t = Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            _to_rgb,
            ToTensor(),
            Normalize(tuple(mean), tuple(std)),
        ])

    def __call__(self, im):
        return self.t(im)


class Un2CLIPEncoder:
    """The encoder of step 03, with its start-up checks E1-E3.

    Images are opened as 03's ImageSet.__getitem__ opens them --
    `with Image.open(path) as im: preprocess(im)` -- with no conversion
    before the resize; RGB conversion happens after the centre crop,
    inside the preprocessing, as in 03. That is the route for the corpus
    images (check C2) and for the stimuli alike.
    """

    def __init__(self):
        import torch
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = ALLOW_TF32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  device     : {self.device}")
        self.model, self.processor, self.sd, self.info = \
            load_model(self.device)
        n_px = self.processor.crop_size["height"]
        self.pre = OpenAIPreprocess(n_px, self.processor.image_mean,
                                    self.processor.image_std)
        self.d_pool = self.model.visual_projection.weight.shape[1]
        self.d_emb = self.model.visual_projection.weight.shape[0]
        self.info.update({
            "device": self.device, "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "preprocessing": {"mode": "openai", "size": n_px,
                              "mean": list(self.processor.image_mean),
                              "std": list(self.processor.image_std)},
            "dims": {"pooled": int(self.d_pool),
                     "embedding": int(self.d_emb)}})

    def encode_paths(self, paths, batch_size=None, what="images"):
        """(pooled float32, embedding float32) for a list of files."""
        torch = self.torch
        bs = batch_size or BATCH_SIZE
        n = len(paths)
        Pz = np.zeros((n, self.d_pool), np.float32)
        Ez = np.zeros((n, self.d_emb), np.float32)
        t1 = time.time()
        for a in range(0, n, bs):
            b = min(a + bs, n)
            px = []
            for p in paths[a:b]:
                try:
                    with Image.open(p) as im:
                        px.append(self.pre(im))
                except Exception as exc:
                    stop(f"Cannot read {p}: {type(exc).__name__}: {exc}")
            x = torch.stack(px).to(self.device)
            p_, e_ = embed(self.model, x)
            Pz[a:b] = p_.float().cpu().numpy()
            Ez[a:b] = e_.float().cpu().numpy()
            if n > bs:
                print(f"    {what}: {b}/{n} ({time.time() - t1:.0f}s)",
                      end="\r")
        if n > bs:
            print()
        return Pz, Ez

    def official_check(self, paths):
        """From 03_encode_images.py, official_check(): embeddings against
        the official un2CLIP loader (clip.load + visual.load_state_dict),
        on the same pixels (E3a) and each with its own preprocessing
        (E3b). Here on stimulus images rather than the first index rows."""
        torch = self.torch
        try:
            import clip
        except ImportError:
            skip("E3", "official loader: the OpenAI 'clip' package is not "
                 "installed (E1-E2 still guarantee the projection)")
            return {"official_check": "skipped"}
        ref, ref_pre = clip.load(OPENAI_MODEL_NAME, device="cpu", jit=False)
        ref.visual.load_state_dict(self.sd)          # strict, as un2CLIP does
        ref = ref.float().to(self.device).eval()
        imgs = []
        for p in paths[:N_CHECK_IMAGES]:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB").copy())

        def cos(a, b):
            a = a.float() / a.float().norm(dim=-1, keepdim=True)
            b = b.float() / b.float().norm(dim=-1, keepdim=True)
            return (a * b).sum(-1).cpu().numpy()

        with torch.inference_mode():
            px_ref = torch.stack([ref_pre(im) for im in imgs]).to(self.device)
            e_ref = ref.encode_image(px_ref)
            _, e_same = embed(self.model, px_ref)
            px_own = torch.stack([self.pre(im) for im in imgs]).to(self.device)
            _, e_own = embed(self.model, px_own)
        c_same, c_own = cos(e_ref, e_same), cos(e_ref, e_own)
        del ref
        if self.device == "cuda":
            torch.cuda.empty_cache()
        check("E3", c_same.min() >= SAME_PIXELS_MIN_COS,
              f"official loader, same pixels: min cos {c_same.min():.6f} "
              f"({len(imgs)} stimuli)")
        check("E3b", c_own.min() >= OWN_PREPROC_MIN_COS,
              f"official vs 'openai' preprocessing: min cos "
              f"{c_own.min():.6f}", critical=False)
        return {"official_check": "run",
                "same_pixels_min_cos": float(c_same.min()),
                "own_preproc_min_cos": float(c_own.min())}




# ################################################################
# VERBATIM from score_stimuli_on_axes_v4.py (lines 1090-1282): axes, control directions, corpus reproduction
# ################################################################
def load_axes():
    print("\nAxes")
    acat = pd.read_csv(os.path.join(AXES_DIR, "axis_catalog.csv"))
    AV = np.load(os.path.join(AXES_DIR, "axis_vectors.npy")).astype(np.float64)
    # From 09c_rescore_sweep_v4.py, main() ("axes"): rows >= 0 have vectors.
    acat = acat[acat["row"] >= 0].sort_values("row")
    rows_ok = (acat["row"].to_numpy() == np.arange(len(AV))).all()
    nerr = float(np.abs(np.linalg.norm(AV, axis=1) - 1).max())
    kv = {k: AV[i] for i, k in enumerate(acat["key"])}
    inc = dict(zip(acat["key"], acat["included"].astype(str).str.lower()
                   .isin(["true", "1"])))
    v4 = acat[(acat["set"] == V4_SET)]
    v4 = v4[v4["key"].map(inc)]
    ids = {a.split()[0]: k for a, k in zip(v4["axis"], v4["key"])}
    problems = []
    if not rows_ok:
        problems.append("axis_catalog rows do not match axis_vectors.npy")
    if nerr > AXIS_NORM_TOL:
        problems.append(f"axis vectors not unit length ({nerr:.1e})")
    if sorted(ids) != sorted(V4_IDS):
        problems.append(f"v4 axes found {sorted(ids)}, expected {V4_IDS}")
    v4keys = [ids[i] for i in V4_IDS if i in ids]
    nckeys = [f"{NOCONTROL_SET}::{k.split('::', 1)[1]}" for k in v4keys]
    miss_nc = [k for k in nckeys if k not in kv]
    if miss_nc:
        problems.append(f"no stored nocontrol vector for {miss_nc[:3]}")
    ctrl = acat[(acat["set"] == CONTROLS_SET)]
    ctrlkeys = [k for k in ctrl["key"] if inc[k]]
    miss_c = [k for k in REPORT_CONTROLS + [QUALITY_KEY] if k not in ctrlkeys]
    if miss_c:
        problems.append(f"control axes {miss_c} not found; available "
                        f"{ctrlkeys}")
    sc = pd.read_csv(os.path.join(SCORES_DIR, "score_catalog.csv"))
    scd = sc.set_index("key")
    allkeys = v4keys + nckeys + ctrlkeys
    miss_sd = [k for k in allkeys if k not in scd.index
               or not np.isfinite(scd.at[k, "sd_raw"])]
    if miss_sd:
        problems.append(f"no corpus SD in score_catalog.csv for "
                        f"{miss_sd[:3]}")
    check("A1", not problems,
          f"{len(AV)} vectors, unit length (max error {nerr:.1e}); "
          f"{len(v4keys)} v4 axes, {len(nckeys)} nocontrol, "
          f"{len(ctrlkeys)} controls; corpus mean and SD for each"
          if not problems else "; ".join(problems))

    # --- A2: control directions, rebuilt as 05_v4 / 09c_rescore_v4 (R4) ---
    caps = pd.read_csv(os.path.join(AXES_DIR, "captions.csv"))
    C = np.load(os.path.join(AXES_DIR, "caption_embeddings.npy")).astype(
        np.float64)
    cidx = dict(zip(caps["caption"], caps["row"]))
    need = [CTRL_TEMPLATE.format(x) for prs in CONTROLS.values()
            for p in prs for x in p]
    m = [c for c in need if c not in cidx]
    if m:
        stop(f"Control captions not in captions.csv: {m}")
    ctrl_vec = {}
    for cn, prs in CONTROLS.items():
        ctrl_vec[cn] = unit(np.mean([C[cidx[CTRL_TEMPLATE.format(a)]]
                                     - C[cidx[CTRL_TEMPLATE.format(b)]]
                                     for a, b in prs], axis=0))
    Q = np.linalg.qr(np.stack(list(ctrl_vec.values()), axis=1))[0]
    cos4 = [float(unit(kv[k] - Q @ (Q.T @ kv[k])) @ kv[n])
            for k, n in zip(v4keys, nckeys)]
    check("A2", min(cos4) >= NOCONTROL_MIN_COS,
          f"enclosure and darkness directions rebuilt from the caption "
          f"embeddings; nocontrol vectors reproduced, min cos "
          f"{min(cos4):.6f}")

    # --- the keys scored, in one table ---
    rows = []
    for k in v4keys:
        rows.append(("v4", k.split("::", 1)[1].split()[0], k))
    for k in nckeys:
        rows.append(("v4@nocontrol",
                     "nc_" + k.split("::", 1)[1].split()[0], k))
    for k in ctrlkeys:
        rows.append(("controls", "ctrl_" + k.split("::", 1)[1].split()[0]
                     .replace("/", "_"), k))
    keys = pd.DataFrame(rows, columns=["set", "label", "key"])
    for c in ("col", "mean_raw", "sd_raw", "mean_raw_colour",
              "sd_raw_colour"):
        keys[c] = [scd.at[k, c] if c in scd.columns else np.nan
                   for k in keys.key]
    keys["sd_source"] = "score_catalog.csv"
    dirs = pd.DataFrame([("direction", f"dir_{cn}", f"direction::{cn}")
                         for cn in ctrl_vec], columns=["set", "label", "key"])
    for cn in ctrl_vec:
        kv[f"direction::{cn}"] = ctrl_vec[cn]
    keys = pd.concat([keys, dirs], ignore_index=True)
    keys["V"] = [kv[k] for k in keys.key]
    if keys.label.duplicated().any():
        stop(f"Duplicate key labels: {list(keys.label[keys.label.duplicated()])}")
    RUN["meta"]["axis_vectors_sha256"] = sha256(
        os.path.join(AXES_DIR, "axis_vectors.npy"))
    return keys


def direction_sd(keys, Emb, status, rng):
    """Corpus mean and SD of the control directions from SD_SAMPLE
    stored embeddings: 09c_rescore_sweep_v4.py, main() ("corpus SD of the
    control directions"). Also the sampled / stored SD ratio of A07 (R5
    there), as a sanity figure."""
    pool_ = np.where(status == 1)[0]
    smp = np.sort(rng.choice(pool_, min(SD_SAMPLE, len(pool_)),
                             replace=False))
    Xs = np.asarray(Emb[smp], np.float64)
    Xs = Xs[np.isfinite(Xs).all(1) & (np.abs(Xs).sum(1) > 0)]
    for i in np.where(keys.set == "direction")[0]:
        s = Xs @ keys.V.iat[i]
        keys.at[i, "mean_raw"] = float(s.mean())
        keys.at[i, "sd_raw"] = float(s.std())
        keys.at[i, "sd_source"] = f"sample of {len(Xs):,} embeddings"
    j = int(np.where(keys.label == "A07")[0][0])
    r5 = float((Xs @ keys.V.iat[j]).std() / keys.sd_raw.iat[j])
    return len(Xs), r5


# ============================================================
# 5. Corpus reproduction (checks C1-C3)
# ============================================================

def corpus_checks(enc, idx, scenes, keys, rng):
    print("\nCorpus reproduction")
    with open(os.path.join(EMB_DIR, "run_meta.json")) as f:
        emb_meta = json.load(f)
    fp = fingerprint(idx)
    Emb = np.load(os.path.join(EMB_DIR, "embeddings_768.npy"), mmap_mode="r")
    Pool = np.load(os.path.join(EMB_DIR, "pooled_1024.npy"), mmap_mode="r")
    status = np.load(os.path.join(EMB_DIR, "status.npy"))
    ids = np.load(os.path.join(EMB_DIR, "image_ids.npy"))
    n = len(idx)
    ok1 = (emb_meta.get("index_fingerprint") == fp and Emb.shape[0] == n
           and Pool.shape[0] == n and len(status) == n
           and (ids == np.arange(n)).all())
    check("C1", ok1, f"image_index.csv is the index step 03 encoded "
          f"(fingerprint {fp[:12]}...; {n:,} rows aligned with the "
          "embeddings)")
    src = scenes.image_id.to_numpy()
    if (status[src] != 1).any():
        stop(f"Source photographs not encoded in step 03: "
             f"{list(scenes.scene_id[status[src] != 1][:3])}")
    pool_ = np.setdiff1d(np.where(status == 1)[0], src)
    rnd = rng.choice(pool_, min(CORPUS_RANDOM_N, len(pool_)), replace=False)
    rows = np.concatenate([src, rnd]).astype(np.int64)
    role = ["scene source"] * len(src) + ["random"] * len(rnd)
    paths = idx.path.to_numpy()[rows].tolist()
    Pn, En = enc.encode_paths(paths, what="corpus images")
    o = np.argsort(rows)                      # memmap reads in row order
    inv = np.empty_like(o)
    inv[o] = np.arange(len(o))
    E_st = np.asarray(Emb[rows[o]], np.float64)[inv]
    P_st = np.asarray(Pool[rows[o]], np.float64)[inv]
    cos_e = (En * E_st).sum(1) / np.linalg.norm(E_st, axis=1)
    cos_p = (Pn * P_st).sum(1) / (np.linalg.norm(Pn, axis=1)
                                  * np.linalg.norm(P_st, axis=1))
    check("C2", cos_e.min() >= CORPUS_EMB_MIN_COS
          and cos_p.min() >= CORPUS_POOLED_MIN_COS,
          f"{len(rows)} corpus images re-encoded ({len(src)} scene sources "
          f"+ {len(rnd)} random), opened as 03 opens them: min cos with "
          f"stored embeddings {cos_e.min():.7f}, with stored pooled vectors "
          f"{cos_p.min():.6f}")

    sc_keys = keys[keys.col.notna() & (keys.set != "direction")]
    cols = sc_keys.col.astype(int).to_numpy()
    A = np.stack(sc_keys.V.to_list())
    mu = sc_keys.mean_raw.to_numpy(float)
    sd = sc_keys.sd_raw.to_numpy(float)
    Z = np.load(os.path.join(SCORES_DIR, "scores_z.npy"), mmap_mode="r")
    if Z.shape[0] != n:
        stop("scores_z.npy rows do not match the image index.")
    Zst = np.asarray(Z[rows[o]], np.float64)[inv][:, cols]
    z_st = (E_st @ A.T - mu) / sd
    z_re = (En.astype(np.float64) @ A.T - mu) / sd
    d_st = np.abs(z_st - Zst).max(1)
    d_re = np.abs(z_re - Zst).max(1)
    check("C3", d_st.max() <= Z_STORED_TOL and d_re.max() <= Z_REENCODED_TOL,
          f"z-scores on {len(cols)} axes vs scores_z.npy: max |diff| "
          f"{d_st.max():.1e} from the stored embeddings (tol "
          f"{Z_STORED_TOL:g}), {d_re.max():.1e} from the re-encoded ones "
          f"(tol {Z_REENCODED_TOL:g})")
    pd.DataFrame({"image_id": rows, "role": role, "path": paths,
                  "cos_embedding": cos_e, "cos_pooled": cos_p,
                  "max_abs_dz_stored_emb": d_st,
                  "max_abs_dz_reencoded": d_re}).to_csv(
        P("corpus_check.csv"), index=False, float_format="%.8g")
    stored_src = E_st[:len(src)]
    return Emb, status, stored_src, {
        "n": int(len(rows)), "min_cos_emb": float(cos_e.min()),
        "min_cos_pooled": float(cos_p.min()),
        "max_dz_stored": float(d_st.max()),
        "max_dz_reencoded": float(d_re.max())}



# ============================================================
# CLIP: text encoder for the new caption pair
# From 05_build_axes_v4.py, load_hf() and encode_hf(), with torch
# imported lazily (the rest of this script needs no torch).
# ============================================================

def encode_text(captions, device):
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)
    model.vision_model = None           # text tower only
    model.visual_projection = None
    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
    model = model.to(device).eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(captions), 64):
            b = tok(captions[i:i + 64], padding="max_length",
                    max_length=77, truncation=False,
                    return_tensors="pt").to(device)
            pooled = model.text_model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"]).pooler_output
            e = model.text_projection(pooled)
            e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            out.append(e.float().cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


def new_direction(device):
    """The NEW, EXPLORATORY direction: unit(e(pos) - e(neg)) for
    NEW_CAPTION_PAIR, encoded as step 05 encodes captions. Check T1
    first re-encodes stored captions and compares them with
    caption_embeddings.npy."""
    caps = pd.read_csv(os.path.join(AXES_DIR, "captions.csv"))
    C = np.load(os.path.join(AXES_DIR, "caption_embeddings.npy"))
    caps = caps.sort_values("row").reset_index(drop=True)
    pick = np.unique(np.linspace(0, len(caps) - 1, TEXT_CHECK_N).astype(int))
    extra = [i for i, c in enumerate(caps.caption) if "photo" in str(c)][:4]
    pick = sorted(set(pick) | set(extra))
    texts = [str(caps.caption.iat[i]) for i in pick] + list(NEW_CAPTION_PAIR)
    E = encode_text(texts, device)
    stored = C[caps.row.to_numpy()[pick]].astype(np.float64)
    cos = (E[:len(pick)] * stored).sum(1) / np.linalg.norm(stored, axis=1)
    check("T1", cos.min() >= TEXT_MIN_COS,
          f"text encoder: {len(pick)} stored captions re-encoded, min cos "
          f"with caption_embeddings.npy {cos.min():.7f}")
    v = E[-2].astype(np.float64) - E[-1].astype(np.float64)
    return unit(v)


# ============================================================
# CLIP: what is encoded, scored and asked
# ============================================================

def match_scene_ids(scenes):
    """From score_stimuli_on_axes_v4.py, match_scenes(), on the scene
    table: each scene's corpus row by path, then category + filename."""
    idx = pd.read_csv(INDEX_CSV, usecols=lambda c: c in
                      ("image_id", "image_key", "category", "path"))
    by_path, by_cf = {}, {}
    for i, p, c in zip(idx.image_id.to_numpy(), idx.path.astype(str),
                       idx.category.astype(str)):
        by_path.setdefault(os.path.normpath(p), []).append(int(i))
        by_cf.setdefault((c, os.path.basename(p)), []).append(int(i))
    ids, how, bad = [], [], []
    for r in scenes.itertuples():
        h = by_path.get(os.path.normpath(r.source_path), [])
        m = "path"
        if not h:
            h = by_cf.get((r.category, os.path.basename(r.source_path)), [])
            m = "category+filename"
        if len(h) != 1:
            bad.append(r.scene_id)
        ids.append(h[0] if len(h) == 1 else -1)
        how.append(m)
    sc = scenes.copy()
    sc["image_id"], sc["match"] = ids, how
    shared = pd.Series(ids)[lambda s: s >= 0].duplicated().sum()
    check("M3", not bad and not shared,
          f"{len(sc)} scenes each matching one corpus row"
          if not bad and not shared else
          f"{len(bad)} scenes without exactly one corpus row ({bad[:3]}); "
          f"{int(shared)} rows shared")
    return sc, idx


def clip_items(cells, img_root, v4grid):
    """The images to encode: the cross (or all) cells of every set, and
    the v4 cross cells read from the v4 stimulus folder."""
    x = cells if CLIP_CELLS == "all" else cells[(cells.cx == 0)
                                                | (cells.sx == 0)]
    items = pd.DataFrame({"set": x.approach + "|" + x.definition,
                          "approach": x.approach, "definition": x.definition,
                          "scene_id": x.scene_id, "cx": x.cx, "sx": x.sx,
                          "path": [os.path.join(img_root, f)
                                   for f in x.filename],
                          "colour_level": "", "fog_level": ""})
    if v4grid is not None and len(v4grid):
        g = v4grid[(v4grid.colour_level == "k=1") | (v4grid.fog_level == "h=0")]
        v = pd.DataFrame({"set": "V4_ref|v4", "approach": "V4_ref",
                          "definition": "v4", "scene_id": g.scene_id,
                          "cx": g.fog_level.map(V4_FOG_IDX),
                          "sx": g.colour_level.map(V4_COL_IDX),
                          "path": [os.path.join(V4_STIMULI_DIR, f)
                                   for f in g.filename],
                          "colour_level": g.colour_level,
                          "fog_level": g.fog_level})
        items = pd.concat([items, v], ignore_index=True)
    return items.reset_index(drop=True)


def clip_effects(items, raw, Ez, keys, Wb):
    """Per set and cell: change from the scene's original (corpus SD of
    each key), mean over scenes with a scene-resampling interval; and the
    embedding distance 1 - cos from the original (raw units)."""
    lab = list(keys.label)
    sd = keys.sd_raw.to_numpy(float)
    rows, store = [], {}
    sids = sorted(items.scene_id.unique())
    si = {s: i for i, s in enumerate(sids)}
    for setk, g in items.groupby("set", sort=False):
        cellsk = sorted({(int(a), int(b)) for a, b in zip(g.cx, g.sx)})
        R = np.full((len(sids), len(cellsk), len(lab)), np.nan)
        E = np.full((len(sids), len(cellsk), Ez.shape[1]), np.nan)
        ck = {c: i for i, c in enumerate(cellsk)}
        for r in g.itertuples():
            R[si[r.scene_id], ck[(int(r.cx), int(r.sx))]] = raw[r.Index]
            E[si[r.scene_id], ck[(int(r.cx), int(r.sx))]] = Ez[r.Index]
        ok = ~np.isnan(R).any(axis=(1, 2))
        R, E = R[ok], E[ok]
        ref = ck[(0, 0)]
        D = (R - R[:, ref:ref + 1]) / sd
        dist = 1.0 - (E * E[:, ref:ref + 1]).sum(-1)
        Wk = Wb[:, ok]
        Wk = Wk / Wk.sum(1, keepdims=True)
        store[setk] = (cellsk, D, dist, Wk)
        for c, (cx, sx) in enumerate(cellsk):
            if (cx, sx) == (0, 0):
                continue
            m, lo, hi, ss = summ(D[:, c, :], Wk)
            for j, l in enumerate(lab):
                rows.append({"set": setk, "cx": cx, "sx": sx,
                             "quantity": l, "key": keys.key.iat[j],
                             "key_set": keys.set.iat[j], "d": m[j],
                             "lo": lo[j], "hi": hi[j], "same_sign": ss[j]})
            m, lo, hi, ss = summ(dist[:, c:c + 1], Wk)
            rows.append({"set": setk, "cx": cx, "sx": sx,
                         "quantity": "emb_dist", "key": "1 - cos",
                         "key_set": "direction-free", "d": m[0],
                         "lo": lo[0], "hi": hi[0], "same_sign": ss[0]})
    return pd.DataFrame(rows), store


def clip_questions(store, keys):
    """The pre-stated questions, both directions, per set; on the v4 axes
    and on v4@nocontrol."""
    lab = list(keys.label)
    qrows = []
    for setk, (cellsk, D, dist, Wk) in store.items():
        if setk.startswith("V4_ref"):
            continue
        ck = {c: i for i, c in enumerate(cellsk)}
        for kset, pre in (("v4", ""), ("v4@nocontrol", "nc_")):
            J07, J13 = lab.index(pre + "A07"), lab.index(pre + "A13")

            def lv(feature, i, j):
                c = ck[(i, 0)] if feature == "contrast" else ck[(0, i)]
                return D[:, c, j]

            def add(q, level, tested, x, sign=1):
                m, lo, hi, ss = summ(x[:, None], Wk)
                qrows.append({"set": setk, "key_set": kset, "question": q,
                              "level": level, "tested": tested,
                              "mean": m[0], "lo": lo[0], "hi": hi[0],
                              "same_sign": ss[0],
                              "class": classify(lo[0], hi[0], sign)})
            for feature, Jown, Jother, qd, qo, qs in (
                    ("contrast", J07, J13, "Q1 contrast direction",
                     "Q1o contrast order", "Q3 contrast specificity"),
                    ("colour", J13, J07, "Q2 colour direction",
                     "Q2o colour order", "Q3 colour specificity")):
                own = "A07" if feature == "contrast" else "A13"
                oth = "A13" if feature == "contrast" else "A07"
                for i in (-3, -2, -1, 1, 2, 3):
                    s = 1 if i < 0 else -1
                    add(qd, f"{i:+d}", f"{'+' if s > 0 else '-'}d({own})",
                        s * lv(feature, i, Jown))
                    add(qs, f"{i:+d}",
                        f"{'+' if s > 0 else '-'}(d({own}) - d({oth}))",
                        s * (lv(feature, i, Jown) - lv(feature, i, Jother)))
                for i in (-3, -2, -1, 0, 1, 2):
                    a_ = lv(feature, i, Jown) if i != 0 else 0 * D[:, 0, Jown]
                    b_ = lv(feature, i + 1, Jown) if i + 1 != 0 \
                        else 0 * D[:, 0, Jown]
                    add(qo, f"{i:+d} -> {i + 1:+d}",
                        f"d({own})[{i:+d}] - d({own})[{i + 1:+d}]", a_ - b_)
    Q = pd.DataFrame(qrows)
    V = (Q.groupby(["set", "key_set", "question"], sort=False)["class"]
         .apply(lambda c: verdict(list(c))).rename("verdict").reset_index())
    return Q, V


def clip_editedness(store, keys):
    """Levels at the same distance below and above the original: do they
    move the editedness measures by about the same amount?"""
    lab = list(keys.label)
    rows = []
    for setk, (cellsk, D, dist, Wk) in store.items():
        if setk.startswith("V4_ref"):
            continue
        ck = {c: i for i, c in enumerate(cellsk)}
        for feature in ("contrast", "colour"):
            for j in (1, 2, 3):
                cd = ck[(-j, 0)] if feature == "contrast" else ck[(0, -j)]
                cu = ck[(j, 0)] if feature == "contrast" else ck[(0, j)]
                qs = []
                for k in EDITEDNESS_KEYS:
                    if k in lab:
                        x = D[:, :, lab.index(k)]
                        qs.append((k, x[:, cd], x[:, cu], True))
                        qs.append((f"|{k}|", np.abs(x[:, cd]),
                                   np.abs(x[:, cu]), True))
                qs.append(("emb_dist", dist[:, cd], dist[:, cu], False))
                for name, dn, up, sd_units in qs:
                    m, lo, hi, _ = summ((dn - up)[:, None], Wk)
                    md = float(np.mean(dn))
                    mu = float(np.mean(up))
                    if sd_units and lo[0] >= -EQUIV_MARGIN_SD \
                            and hi[0] <= EQUIV_MARGIN_SD:
                        v = f"symmetric (within +-{EQUIV_MARGIN_SD:g} SD)"
                    elif lo[0] > 0:
                        v = "down moves it more"
                    elif hi[0] < 0:
                        v = "up moves it more"
                    else:
                        v = ("inconclusive" if sd_units
                             else "no difference shown")
                    rows.append({"set": setk, "feature": feature,
                                 "distance": j, "quantity": name,
                                 "units": "corpus SD" if sd_units
                                 else "1 - cos",
                                 "mean_down": md, "mean_up": mu,
                                 "down_minus_up": m[0], "lo": lo[0],
                                 "hi": hi[0],
                                 "difference_shown": bool(lo[0] > 0
                                                          or hi[0] < 0),
                                 "verdict": v,
                                 "exploratory": name.strip("|")
                                 == "new_edited"})
    return pd.DataFrame(rows)


def check_v4_clip(store, keys, items):
    """CL1: the v4 cross cells, re-scored here, against the committed v4
    cell effects."""
    if not (V4_AXES_EFFECTS_CSV and os.path.isfile(V4_AXES_EFFECTS_CSV)):
        skip("CL1", f"{V4_AXES_EFFECTS_CSV} not found")
        return
    if "V4_ref|v4" not in store:
        skip("CL1", "no v4 cells encoded")
        return
    ref = pd.read_csv(V4_AXES_EFFECTS_CSV)
    ref = ref[(ref.scale == "sd_raw") & (ref.kind == "grid")]
    cellsk, D, _, _ = store["V4_ref|v4"]
    lab = list(keys.label)
    inv_f = {v: k for k, v in V4_FOG_IDX.items()}
    inv_c = {v: k for k, v in V4_COL_IDX.items()}
    worst, n = 0.0, 0
    for c, (cx, sx) in enumerate(cellsk):
        if (cx, sx) == (0, 0):
            continue
        name = f"{inv_c[sx]} {inv_f[cx]}"
        for q in ("A07", "A13", "ctrl_quality"):
            m = ref[(ref.cell == name) & (ref.quantity == q)]
            if not len(m) or q not in lab:
                continue
            worst = max(worst, abs(float(D[:, c, lab.index(q)].mean())
                                   - float(m.d.iloc[0])))
            n += 1
    check("CL1", n > 0 and worst <= V4_CLIP_TOL,
          f"v4 cross cells re-scored here vs {V4_AXES_EFFECTS_CSV}: "
          f"{n} cell x axis means, worst difference {worst:.1e} SD "
          f"(tol {V4_CLIP_TOL:g})", critical=False)


def clip_figure(eff, path):
    """Mean change on fog (A07) along the contrast ladder and on dull
    (A13) along the colour ladder, per approach, one panel pair per
    definition."""
    defs = [d for d in DEFINITION_NAMES.values()
            if eff.set.str.endswith("|" + d).any()]
    if not defs:
        return
    fig, axs = plt.subplots(len(defs), 2, figsize=(11, 3.4 * len(defs)),
                            squeeze=False)
    names = list(APPROACHES)
    colours = plt.get_cmap("tab10")
    for r, d in enumerate(defs):
        for c, (feature, q) in enumerate((("contrast", "A07"),
                                          ("colour", "A13"))):
            ax = axs[r, c]
            for setk in sorted(eff.set.unique()):
                if not setk.endswith("|" + d):
                    continue
                x = eff[(eff.set == setk) & (eff.quantity == q)]
                x = x[x.sx == 0] if feature == "contrast" else x[x.cx == 0]
                idx = x.cx if feature == "contrast" else x.sx
                o = np.argsort(idx.to_numpy())
                xs = np.r_[idx.to_numpy()[o], 0]
                ys = np.r_[x.d.to_numpy()[o], 0.0]
                oo = np.argsort(xs)
                a = setk.split("|")[0]
                ax.plot(xs[oo], ys[oo], marker="o", lw=1.4, label=a,
                        color=colours(names.index(a) % 10))
            ax.axhline(0, color="#999", lw=0.8)
            ax.set_title(f"{d}: {q} along the {feature} ladder", fontsize=10)
            ax.set_xlabel("level (0 = original)")
            ax.set_ylabel("change, corpus SD")
            ax.spines[["top", "right"]].set_visible(False)
    for r in range(len(defs)):
        axs[r, 1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_clip(img_root, cells, scenes, v4grid, aside=None):
    """The whole CLIP part. Writes into RUN["out"] (the clip folder).

    `cells` holds the pooled scenes only. The v4 cross cells are encoded
    for EVERY scene: check CL1 compares them with v4's committed means,
    which are over all 47 scenes; every other statistic of V4_ref uses
    the pooled scenes only (`aside` = the set-aside scene ids)."""
    aside = set(EXCLUDE_SCENES if aside is None else aside)
    scenes = scenes[~scenes.scene_id.isin(aside)]
    rng_corpus = np.random.default_rng([RANDOM_SEED, 1])
    rng_boot = np.random.default_rng([RANDOM_SEED, 2])
    rng_dir = np.random.default_rng([RANDOM_SEED, 4])
    rng_main = np.random.default_rng([RANDOM_SEED, 5])
    sc, idx = match_scene_ids(scenes)
    keys = load_axes()
    print("\n  encoder")
    enc = Un2CLIPEncoder()
    nd = new_direction(enc.device)
    items = clip_items(cells, img_root, v4grid)
    enc.info.update(enc.official_check(list(items.path)))
    if HASH_CHECKPOINT:
        enc.info["un2clip_ckpt_sha256"] = sha256(UN2CLIP_CKPT_PATH)
    RUN["meta"]["encoder"] = enc.info
    Emb, status, _, cinfo = corpus_checks(enc, idx, sc, keys, rng_corpus)
    RUN["meta"]["corpus_check"] = cinfo
    n_smp, r5 = direction_sd(keys, Emb, status, rng_dir)
    RUN["meta"]["direction_sd_sample"] = {"n": n_smp, "A07_ratio": r5}
    # the new direction, scored like the control directions
    pool_ = np.where(status == 1)[0]
    smp = np.sort(np.random.default_rng([RANDOM_SEED, 4]).choice(
        pool_, min(SD_SAMPLE, len(pool_)), replace=False))
    Xs = np.asarray(Emb[smp], np.float64)
    Xs = Xs[np.isfinite(Xs).all(1) & (np.abs(Xs).sum(1) > 0)]
    s_ = Xs @ nd
    keys = pd.concat([keys, pd.DataFrame([{
        "set": "exploratory (new)", "label": "new_edited",
        "key": f"new::{NEW_CAPTION_PAIR[0]} vs {NEW_CAPTION_PAIR[1]}",
        "mean_raw": float(s_.mean()), "sd_raw": float(s_.std()),
        "sd_source": f"sample of {len(Xs):,} embeddings", "V": nd}])],
        ignore_index=True)

    print(f"\n  encoding {len(items):,} images")
    Pz, Ez = enc.encode_paths(list(items.path), what="images")
    nrm = np.linalg.norm(Ez, axis=1)
    check("S2c", np.isfinite(Ez).all() and np.abs(nrm - 1).max() < 1e-4,
          f"{len(Ez)} embeddings finite, norm {nrm.min():.6f} - "
          f"{nrm.max():.6f}")
    pick = rng_main.choice(len(items), min(STIM_RECHECK_N, len(items)),
                           replace=False)
    cs = []
    for i in pick:
        _, e1 = enc.encode_paths([items.path.iat[int(i)]], batch_size=1)
        cs.append(float(e1[0] @ Ez[int(i)]))
    check("S3c", min(cs) >= STIM_RECHECK_MIN_COS,
          f"{len(cs)} images re-encoded one at a time: min cos {min(cs):.7f}")
    np.save(P("embeddings_768.npy"), Ez.astype(np.float32))
    np.save(P("pooled_1024.npy"), Pz.astype(np.float32))
    A = np.stack(keys.V.to_list())
    raw = Ez.astype(np.float64) @ A.T
    z = (raw - keys.mean_raw.to_numpy(float)) / keys.sd_raw.to_numpy(float)
    isc = pd.concat([items.drop(columns="path").assign(
        file=[os.path.relpath(p, img_root) if p.startswith(img_root) else p
              for p in items.path]),
        pd.DataFrame(raw, columns=[f"raw_{l}" for l in keys.label]),
        pd.DataFrame(z, columns=[f"z_{l}" for l in keys.label])], axis=1)
    isc.to_csv(P("image_scores.csv"), index=False, float_format="%.6g")
    keys.drop(columns="V").to_csv(P("score_keys.csv"), index=False,
                                  float_format="%.8g")
    # V4_ref of the set-aside scenes: CL1 only
    pooled = ~(items.scene_id.isin(aside)).to_numpy()
    items_all, raw_all, Ez_all = items, raw, Ez
    items = items[pooled].reset_index(drop=True)
    raw, Ez = raw[pooled], Ez[pooled]
    sids = sorted(items.scene_id.unique())
    Wb = boot_weights(len(sids), rng_boot)
    eff, store = clip_effects(items, raw, Ez, keys, Wb)
    eff.to_csv(P("effects.csv"), index=False, float_format="%.5g")
    Q, V = clip_questions(store, keys)
    Q.to_csv(P("question_levels.csv"), index=False, float_format="%.5g")
    V.to_csv(P("question_verdicts.csv"), index=False)
    E = clip_editedness(store, keys)
    E.to_csv(P("editedness.csv"), index=False, float_format="%.5g")
    v4i = np.where((items_all.approach == "V4_ref").to_numpy())[0]
    if len(v4i):
        it4 = items_all.iloc[v4i].reset_index(drop=True)
        _, store4 = clip_effects(
            it4, raw_all[v4i], Ez_all[v4i], keys,
            boot_weights(it4.scene_id.nunique(),
                         np.random.default_rng([RANDOM_SEED, 6])))
    else:
        store4 = {}
    check_v4_clip(store4, keys, items_all)
    clip_figure(eff, P("ladders_A07_A13.png"))
    return {"eff": eff, "Q": Q, "V": V, "E": E, "n_images": len(items)}


# ============================================================
# Report
# ============================================================

def _tbl(df, cols=None, fmt="{:+.2f}", maxrows=200):
    if df is None or not len(df):
        return "    (none)\n"
    d = df[cols] if cols else df
    d = d.head(maxrows)
    s = d.to_string(index=False, float_format=lambda v: fmt.format(v))
    return "\n".join("    " + ln for ln in s.splitlines()) + "\n"


def write_report(T):
    """T: every table the run produced. Sections follow the task; the
    plain-language summary is the last section."""
    L = []
    w = L.append
    w("TEST OF EDIT LEVELS ABOVE AND BELOW THE ORIGINAL (v5)")
    w("=" * 72)
    w(f"run folder      {RUN['base']}")
    w(f"settings hash   {RUN['meta'].get('settings_hash')}")
    w(f"created         {time.strftime('%Y-%m-%d %H:%M')}")
    w(f"scenes          {T['n_scenes']} pooled"
      + (f" + {len(EXCLUDE_SCENES)} set aside (section 8b)"
         if EXCLUDE_SCENES else ""))
    w(f"approaches      {', '.join(APPROACHES)}")
    _dd = {"a": "a = corpus percentiles (median image, as v4)",
           "b": "b = equal SD steps, per scene",
           "c": "c = fixed Pillow factors (B2 only)",
           "r": "r = equal RATIO steps on " + "/".join(R_FEATURES)
                + ", equal SD steps otherwise"}
    _run = sorted({d for a in APPROACHES.values() for d in a["definitions"]})
    w("definitions     " + "; ".join(_dd[d] for d in _run))
    w("")
    w("CLIP is not a human observer, and none of these numbers is a "
      "threshold: the\nvisual check of the grids and pairs decides. "
      "Artefact numbers are reported,\nnot judged.")
    w("")

    w("1. CHECKS")
    w("-" * 72)
    for c in CHECKS:
        w(f"  {c['code']:<5} {c['status']:<5} {c['detail']}")
    w("")

    w("2. REFERENCE")
    w("-" * 72)
    st = T["st"]
    w(f"  stimulus-path reference: {int(st.n_images.iloc[0]):,} images")
    for f in (FEAT_SAT, FEAT_CON):
        w(f"  {f:<15} median {st.at[f, 'p50']:.3f}  SD {st.at[f, 'sd']:.3f}"
          f"  p15 {st.at[f, 'p15']:.2f}  p85 {st.at[f, 'p85']:.2f}")
    w("  In SD the percentile targets are not symmetric about the median: "
      "the corpus\n  distributions are skewed (see target_sd in levels_a.csv).")
    w("")

    w("3. DEFINITION (a): CORPUS PERCENTILES (median image, one parameter "
      "for all scenes)")
    w("-" * 72)
    la = T["levels_a"].copy()
    la["capped"] = la.status.where(la.status != "ok", "")
    w(_tbl(la, ["family", "direction", "level", "target_sd", "param",
                "achieved_sd", "share_reaching_target", "cross_change_sd",
                "mean_clipped_new", "capped"], "{:.3f}"))
    if len(T["comp_a"]):
        w("  A4 compensation (chroma factor restoring the median image's "
          "sat_mean):")
        w(_tbl(T["comp_a"], fmt="{:.3f}"))

    w("4. DEFINITION (b): REACH AND SPANS (per scene)")
    w("-" * 72)
    if B_USE_CLIP_BUDGET:
        w(f"  Reach = how far each scene's feature moves before more than "
          f"{CLIP_MAX_NEW:.0%} of pixels\n  are newly clipped (beyond "
          f"{CLIP_DEPTH:g} level, or gamut-mapped), the colour edit's gamut "
          f"plateau,\n  or a reversal.")
    else:
        w("  Reach = how far each scene's feature moves before the end of "
          "the parameter grid,\n  a reversal, or (colour going up) the "
          "gamut plateau. NO CLIPPING BUDGET: clipping\n  is measured and "
          "reported (sections 6 and 8), it does not stop the reach.")
    if B_SPAN_SD is not None:
        w(f"  Span = min(up, down). Rendered span = {B_SPAN_SD:g} SD for "
          f"every approach and feature\n  (B_SPAN_SD); levels at 1/3, 2/3, "
          f"3/3 of it, solved per scene. share_reach_rendered =\n  share "
          f"of scenes reaching it (the rest are CAPPED at their reach).")
    else:
        w(f"  Span = min(up, down). Rendered span = the value "
          f"{1 - SPAN_QUANTILE:.0%} of scenes\n  reach; levels at 1/3, 2/3, "
          "3/3 of it, solved per scene.")
    w("  Fixed steps: share of scenes whose span reaches +-0.3 / 0.6 / "
      "0.9 SD.")
    if any("r" in a["definitions"] for a in APPROACHES.values()):
        w(f"  Definition (r) uses the same reach: {'/'.join(R_FEATURES)} "
          f"steps by equal ratios (check R0 gives\n  the ratio), so its "
          f"targets in SD differ per scene and its level names are the "
          f"ratios\n  (x0.653 = 65.3% of the original's value).")
    w(_tbl(T["span_summary"], ["approach", "feature", "span_median_scene",
                               "span_rendered_q", "span_rendered",
                               "share_reach_rendered", "span_min",
                               "reach_up_median", "reach_down_median",
                               "binding_side", "limit_up_most_common"]
           + [f"share_reach_pm{s:g}" for s in FIXED_STEPS_SD], "{:.2f}"))
    lim = (T["reach"].groupby(["family", "direction", "limit"]).size()
           .rename("scenes").reset_index())
    w("  What stops each family:")
    w(_tbl(lim, fmt="{:.0f}"))

    w("5. LADDERS AS RENDERED (own feature, measured; median over scenes)")
    w("-" * 72)
    ct = T["crosstalk"]
    w(_tbl(ct, ["approach", "definition", "feature", "level",
                "own_median_sd", "own_min_sd", "own_max_sd",
                "cross_median_sd", "ratio_of_medians", "ratio_worst_scene",
                "n_capped"], "{:+.2f}"))
    w("  cross_median_sd is the cross-talk: how far the OTHER feature moves "
      "(corpus SD);\n  ratio_of_medians is v4's |cross / own|.")
    w("")

    w("6. CLIPPING AND ARTEFACTS (numbers only)")
    w("-" * 72)
    a = T["artefacts"]
    cols = ["approach", "definition", "group", "halo_share_max",
            "noise_ratio_max", "band_new_empty_max",
            f"clip_{CLIP_DEPTH:g}lv_max", "gamut_mapped_max"]
    w(_tbl(a[a.group.str.startswith("all")], [c for c in cols if c in a],
           "{:.3f}"))
    w("  Per level: artefacts.csv. Halo = share of near-edge pixels beyond "
      "the original's\n  local range after the global tone curve; noise = "
      "flat-region high-pass energy,\n  edited / original; banding = new "
      "empty grey levels. V4_ref is the v4 grid,\n  measured by the same "
      "code.")
    w("")
    ev = T["envelope"]
    w("7. ENVELOPE (features outside the corpus 1-99% range)")
    w("-" * 72)
    if len(ev):
        e = (ev.groupby(["approach", "definition", "feature"]).size()
             .rename("cells").reset_index())
        w(_tbl(e, fmt="{:.0f}"))
    else:
        w("    no image leaves the range on any envelope feature\n")

    w("8. SCENE FIT")
    w("-" * 72)
    w("  fit = all seven levels reached on both features, own feature "
      "monotone with no\n  collapsed step"
      + (", no cell over the clipping budget." if B_USE_CLIP_BUDGET else
         " (definition (b) has no clipping budget this run:\n  "
         "cells_over_clip_budget is reported, not counted).")
      + " The envelope and the artefacts\n  (compared with the v4 grid's "
      "maxima) are reported beside it, not counted.")
    w(_tbl(T["fit_summary"], fmt="{:.0f}"))

    if EXCLUDE_SCENES:
        A = T["aside"]
        w("8b. SCENES SET ASIDE (EXCLUDE_SCENES)")
        w("-" * 72)
        w("  Rendered and shown (viewer: 'set aside'), left out of every "
          "number above and\n  below. Their own reach, and the levels "
          "capped when solved toward the pooled span:")
        sp = A["spans"]
        if len(sp):
            piv = sp.pivot_table(index="scene_id", columns=["approach",
                                                            "feature"],
                                 values="span_sd", sort=False)
            piv.columns = [f"{a} {f}" for a, f in piv.columns]
            w("  span (min of up and down reach), corpus SD:")
            w(_tbl(piv.reset_index(), fmt="{:.2f}"))
        fb = A["fit"]
        fb = fb[fb.definition == DEFINITION_NAMES["b"]]
        if len(fb):
            w("  definition (b): levels capped (contrast; colour):")
            w(_tbl(fb[["scene_id", "approach", "contrast_capped_levels",
                       "colour_capped_levels"]].fillna(""), fmt="{}"))
    w("9. NORMALITY CHECK")
    w("-" * 72)
    w("  Distance from typical: sqrt(z_sat^2 + z_con^2), z from the corpus "
      "median (corpus SD),\n  or from the scene's own original. Pair term "
      "= distance(B) - distance(A): positive\n  = A is the more typical "
      "image. Direction terms as v5 (positive = A is the predicted,\n  "
      "less saturated or lower-contrast image). A correlation far from 0 "
      "means 'pick the\n  more (or less) normal-looking image' would mimic "
      "the predicted weights.")
    ns = T["norm_summary"]
    key = ns[(ns.subset == "all typed pairs")
             & (ns.normality_term == "combined")]
    piv = key.pivot_table(index=["approach", "definition"],
                          columns=["reference", "predictor"],
                          values="pearson", sort=False)
    piv.columns = [f"r({r},{p})" for r, p in piv.columns]
    w(_tbl(piv.reset_index(), fmt="{:+.2f}"))
    w("  Straddling pairs (unordered, typed by v5's rules):")
    sc_ = T["straddle"]
    w(_tbl(sc_[sc_.type == "all typed"].drop(columns="type"), fmt="{:.0f}"))

    if T.get("clip"):
        C = T["clip"]
        w("10. CLIP (un2CLIP, v4 axes; CLIP is not a human observer)")
        w("-" * 72)
        V = C["V"].copy()
        short = {"Q1 contrast direction": "Q1 con dir",
                 "Q1o contrast order": "Q1o con order",
                 "Q3 contrast specificity": "Q3 con spec",
                 "Q2 colour direction": "Q2 col dir",
                 "Q2o colour order": "Q2o col order",
                 "Q3 colour specificity": "Q3 col spec"}
        V["q"] = V.question.map(short)
        w("  Q1 contrast down raises fog (A07), up lowers it; Q2 colour down "
          "raises dull (A13),\n  up lowers it; Q1o/Q2o every adjacent step "
          "in order; Q3 contrast edits move A07\n  more than A13, colour "
          "edits A13 more than A07 (both directions).")
        for kset in ("v4", "v4@nocontrol"):
            piv = V[V.key_set == kset].pivot_table(
                index="set", columns="q", values="verdict", aggfunc="first",
                sort=False)[list(short.values())]
            w(f"  [{kset}]")
            w(_tbl(piv.reset_index(), fmt="{}"))
        w("  shown / not shown / reversed per level: question_levels.csv. "
          "PASS = every level\n  shown; PARTIAL = none reversed; FAIL = any "
          "reversed.")
        w("")
        E = C["E"].copy()
        code = {"symmetric": "S", "down": "D", "up": "U", "inconclusive": "I",
                "no difference": "-"}
        E["c"] = [next(v for k, v in code.items() if t.startswith(k))
                  for t in E.verdict]
        e2 = (E.groupby(["set", "quantity"], sort=False)["c"]
              .apply(lambda v: "".join(v)).unstack("quantity"))
        w("  Editedness: levels at the same distance below and above the "
          "original, one letter\n  per feature x distance (contrast 1, 2, 3, "
          f"then colour 1, 2, 3): S = the same within\n  +-{EQUIV_MARGIN_SD:g}"
          " SD, D = the level below moves it more, U = the level above moves "
          "it more,\n  I = inconclusive, - = no difference shown (emb_dist, "
          "1 - cos, has no margin).\n  new_edited is NEW and EXPLORATORY, "
          "not a v4 direction. Details: clip/editedness.csv.")
        w(_tbl(e2.reset_index(), fmt="{}"))
    else:
        w("10. CLIP: not run (RUN_CLIP = False). Run with CLIP_ONLY_FROM = "
          "this folder.")
        w("")

    w("11. FILES AND TIMES")
    w("-" * 72)
    w(f"  images: {T['n_images']:,} PNGs under images/ "
      f"({T['images_gb']:.1f} GB), contact sheets under images/sheets/")
    if T.get("pack"):
        p = T["pack"]
        w(f"  viewer pack: {len(p['scenes'])} scenes, {p['n_images']:,} "
          f"images, {p['mb']:.0f} MB"
          + (f", zipped as {os.path.basename(p['zip'])}" if p["zip"] else ""))
    for k, v in RUN["timings"].items():
        w(f"  {k:<40} {v:6.1f} min")
    w("")

    w("12. PLAIN-LANGUAGE SUMMARY")
    w("-" * 72)
    for para in plain_summary(T):
        for ln in _wrap(para):
            w(ln)
        w("")
    with open(B("report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


def _wrap(s, n=72):
    import textwrap
    return textwrap.wrap(s, n, initial_indent="  ", subsequent_indent="  ")


def plain_summary(T):
    out = []
    bad = [c for c in CHECKS if c["status"] in ("FAIL", "WARN")]
    out.append(
        "Checks: " + (f"{len(bad)} check(s) did not pass: "
                      + ", ".join(f"{c['code']} ({c['status']})" for c in bad)
                      + ". Read section 1 before anything else."
                      if bad else "every check passed or was a note."))
    ss = T["span_summary"]
    con = ss[ss.feature == "contrast"].sort_values("span_median_scene",
                                                   ascending=False)
    col = ss[ss.feature == "colour"]
    if len(con) and B_SPAN_SD is not None:
        out.append(
            f"Definition (b) is rendered at the same +-{B_SPAN_SD:g} SD for "
            "every approach and feature"
            + ("" if B_USE_CLIP_BUDGET else ", with no clipping budget "
               "(clipping is reported, not limited)")
            + ". Share of scenes reaching it on contrast: "
            + ", ".join(f"{r.approach} {r.share_reach_rendered:.0%}"
                        for r in con.itertuples())
            + "; on colour: "
            + ", ".join(f"{r.approach} {r.share_reach_rendered:.0%}"
                        for r in col.itertuples())
            + ". Scenes below 100% are CAPPED at their reach on those "
            "levels. For comparison, the contrast span 90% of scenes reach "
            "is "
            + ", ".join(f"{r.approach} {r.span_rendered_q:.2f}"
                        for r in con.itertuples()) + " SD.")
    elif len(con):
        best = con.iloc[0]
        out.append(
            "How far can each feature go both ways? With equal steps in "
            "corpus SD, the contrast span that 90% of scenes reach is "
            + ", ".join(f"{r.approach} {r.span_rendered_q:.2f}"
                        for r in con.itertuples())
            + " SD. The widest is " + f"{best.approach}. "
            + "In every approach the binding side is mostly "
            + ("the upward edit" if (con.binding_side == "up").mean() > 0.5
               else "the downward edit")
            + ". The colour span is "
            + ", ".join(f"{APPROACHES[r.approach]['colour']} "
                        f"{r.span_rendered_q:.2f}"
                        for r in col.drop_duplicates("span_rendered_q")
                        .itertuples())
            + " SD (chroma is shared by every approach but B2).")
    la = T["levels_a"]
    cap = la[la.status != "ok"]
    if any("a" in a["definitions"] for a in APPROACHES.values()):
        out.append(
            "With corpus percentiles (one parameter for every scene, set on "
            "the median image), "
            + (f"{len(cap)} of {len(la)} levels could not reach their "
               "target and are rendered at their cap: "
               + ", ".join(f"{r.family} {r.level}"
                           for r in cap.itertuples())
               + "." if len(cap) else "every level reached its target."))
    fs = T["fit_summary"]
    out.append(
        "Scene fit (all seven levels reached on both features, monotone"
        + (", within the clipping budget" if B_USE_CLIP_BUDGET else "")
        + f"), out of {T['n_scenes']}: "
        + "; ".join(f"{r.approach} {r.definition} {r.fit}"
                    for r in fs.itertuples()) + ".")
    ct = T["crosstalk"]
    ctc = ct[ct.feature == "contrast"]
    if len(ctc):
        g = (ctc.assign(r=ctc.cross_median_sd.abs())
             .groupby(["approach", "definition"]).r.max())
        out.append(
            "Cross-talk: the largest median change of sat_mean caused by a "
            "contrast level is "
            + ", ".join(f"{a} {d} {v:.2f}" for (a, d), v in g.items())
            + " SD. A4 restores sat_mean by construction; what is left is in "
            "ladders.csv (comp_residual_sd).")
    ns = T["norm_summary"]
    k = ns[(ns.subset == "all typed pairs") & (ns.normality_term == "combined")
           & (ns.reference == "orig")]
    if len(k):
        v4 = k[k.approach == "V4_ref"]
        rest = k[k.approach != "V4_ref"]
        txt = ("Normality: in the v4 design the 'closer to the original' "
               "term correlates "
               + ", ".join(f"{r.pearson:+.2f} with {r.predictor}"
                           for r in v4.itertuples())
               + " over its typed pairs. ")
        if len(rest):
            mx = (rest.assign(a=rest.pearson.abs())
                  .groupby(["approach", "definition"]).a.max())
            txt += ("In the candidate sets the largest absolute correlation "
                    "is " + ", ".join(f"{a} {d} {v:.2f}"
                                      for (a, d), v in mx.items())
                    + ". Near zero means the strategy is no longer confounded "
                    "with the predictors.")
        out.append(txt)
    if T.get("clip"):
        V = T["clip"]["V"]
        v4v = V[V.key_set == "v4"]
        g = v4v.groupby("set").verdict.apply(
            lambda v: f"{(v == 'PASS').sum()}/{len(v)}")
        out.append(
            "CLIP (pre-stated questions on the v4 axes, both directions; "
            "PASS = every level shown): " + ", ".join(f"{s} {x}"
                                                       for s, x in g.items())
            + ". Editedness by level distance is in clip/editedness.csv; "
            "the 'edited photo' direction is new and exploratory. CLIP is "
            "not a human observer.")
    out.append(
        "Next step: open viewer.html (or viewer_pack/viewer.html on other "
        "devices) and judge, at 512 px, whether each step is visible, "
        "whether the levels above the original look as edited as those "
        "below, and whether any method leaves visible artefacts. The numbers "
        "here say which candidates are worth that look.")
    return out


# ============================================================
# Main
# ============================================================

def checks_after_render(cells, lad, repro, ident, pil):
    # --- reproduction of the v4 stimuli ---
    if len(repro):
        g = bool(repro.generic_equals_v4_renderer.all())
        check("RG", g, f"this script's renderer equals v4's render_cell on "
              f"all {len(repro):,} v4 grid cells", critical=False)
        have = repro[repro.pixels_identical.notna()]
        if len(have):
            npx = int(have.pixels_identical.astype(bool).sum())
            nby = int(have.bytes_identical.astype(bool).sum())
            check("RP1", npx == len(have),
                  f"v4 haze_eq x chroma cells re-rendered at v4's parameters:"
                  f" {npx}/{len(have)} pixel-identical to the v4 PNGs"
                  + ("" if npx == len(have) else
                     f"; worst {int(have.max_abs_level.max())} level(s)"),
                  critical=False)
            if nby == len(have):
                check("RP2", True, f"{nby}/{len(have)} PNG files "
                      "byte-identical as well")
            else:
                note("RP2", f"{nby}/{len(have)} PNG files byte-identical. "
                     "Where the pixels are identical a byte difference is the "
                     "PNG encoder (Pillow / zlib version), not the image.")
        else:
            skip("RP1", f"no v4 PNGs found under {V4_STIMULI_DIR}")
    # --- identity ---
    g = (ident.groupby("family").agg(max_abs=("max_abs_level", "max"),
                                     d_sat=("d_sat_sd",
                                            lambda s: s.abs().max()),
                                     d_con=("d_con_sd",
                                            lambda s: s.abs().max())))
    ok = bool(((g.max_abs <= 1) & (g.d_sat < 0.01) & (g.d_con < 0.01)).all())
    check("I1", ok, "every edit at its no-change setting, without the "
          "short cut (max pixel change; max |change| of sat_mean, "
          "contrast_local in SD): " + "; ".join(
              f"{f} {r.max_abs} lv, {r.d_sat:.3f}, {r.d_con:.3f}"
              for f, r in g.iterrows())
          + ". Rendered images use the short cut, so the original is exact.",
          critical=False)
    # --- Pillow model ---
    check("P1", int(pil.max_abs_level.max()) <= 1,
          f"Pillow output vs its float model (used to count clipping): "
          f"max difference {int(pil.max_abs_level.max())} level(s)",
          critical=False)
    near = pil[(pil.param - 1.0).abs() < 1e-3]
    note("P2", "Pillow's blend truncates instead of rounding: at a factor "
         "within 1e-6 of 1 the output already shifts by "
         f"{near.mean_shift_vs_original.mean():+.2f} levels on average "
         "(every pixel, both directions), so a Pillow edit jumps at the "
         "original and levels very close to it cannot be hit exactly")
    # --- definition (b): levels on target ---
    b = lad[lad.definition.isin([DEFINITION_NAMES["b"], DEFINITION_NAMES["r"]])
            & (lad["index"] != 0)]
    if len(b):
        ok_ = b[~b.capped]
        err = (ok_.achieved_sd - ok_.target_sd).abs()
        off = ok_.assign(err=err)[err > LEVEL_CHECK_TOL_SD]
        worst = (ok_.assign(err=err).groupby("approach").err.max()
                 .sort_values(ascending=False))
        check("L1", not len(off),
              f"definitions (b), (r): {len(ok_) - len(off):,} of {len(ok_):,} "
              f"uncapped levels within {LEVEL_CHECK_TOL_SD:g} SD of their "
              f"target on the rendered image; worst by approach: "
              + ", ".join(f"{a} {v:.3f}" for a, v in worst.items())
              + f"; {int(b.capped.sum())} levels capped", critical=False)
    # --- monotone ladders ---
    note("M1", "ladders with a collapsed or reversed step are listed in "
         "scene_fit.csv (contrast_monotone, colour_monotone)")


def main_clip_only():
    src = CLIP_ONLY_FROM
    if not os.path.isfile(os.path.join(src, "cells.csv")):
        raise SystemExit(f"{src} holds no cells.csv; not a run folder.")
    out = os.path.join(src, "clip")
    n = 2
    while occupied(out):
        out = os.path.join(src, f"clip_{n}")
        n += 1
    os.makedirs(out, exist_ok=True)
    RUN["base"] = RUN["out"] = out
    with open(os.path.join(src, "settings.json")) as f:
        RUN["meta"]["settings_hash"] = json.load(f).get("settings_hash")
    print(f"CLIP only, for {src} -> {out}")
    cells = pd.read_csv(os.path.join(src, "cells.csv"), low_memory=False)
    scenes, v4grid = load_scenes()
    if "set_aside" in cells.columns:      # the run's own list, not today's
        aside = set(cells.scene_id[cells.set_aside.astype(bool)])
        cells = cells[~cells.set_aside.astype(bool)]
    else:
        aside = set()
    with Timer("CLIP"):
        res = run_clip(src, cells, scenes, v4grid, aside)
    V = res["V"]
    with open(P("clip_report.txt"), "w") as f:
        f.write("CLIP SCORING (CLIP_ONLY_FROM run)\n\nChecks:\n")
        for c in CHECKS:
            f.write(f"  {c['code']:<5} {c['status']:<5} {c['detail']}\n")
        f.write("\nVerdicts (PASS = every level shown):\n")
        f.write(V.to_string(index=False) + "\n")
        f.write("\nCLIP is not a human observer.\n")
    write_checks()
    write_meta("finished")


def main():
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    cv2.setNumThreads(1)
    print("=" * 70 + "\ntest_edit_levels_v5.py\n" + "=" * 70)
    if CLIP_ONLY_FROM:
        return main_clip_only()
    h = settings_hash()
    out = resolve_run_dir(h)
    os.makedirs(out, exist_ok=True)
    RUN["base"] = RUN["out"] = out
    RUN["meta"]["settings_hash"] = h
    with open(B("settings.json"), "w") as f:
        json.dump({"settings_hash": h, "hashed_settings_exclude":
                   sorted(_NOT_HASHED), "settings": settings_dict()}, f,
                  indent=2, default=_json_default)
    print(f"Output: {out}")
    write_meta("running")

    with Timer("Start-up checks and the reference"):
        idx, F, eligible = load_corpus()
        rng = np.random.default_rng(RANDOM_SEED)
        check_feature_code(idx, F, rng)          # uses rng first, as v4
        st = measure_reference(idx, eligible)
        st.reset_index().to_csv(B("reference_stimulus_path.csv"),
                                index=False, float_format="%.6g")
        check_reference(st)
        scenes, v4grid = load_scenes()
    sd = {FEAT_SAT: float(st.at[FEAT_SAT, "sd"]),
          FEAT_CON: float(st.at[FEAT_CON, "sd"])}

    with Timer("Definition (a): calibration on the median image"):
        rows = draw_sample(idx, eligible, rng)
        lines = []
        for feat in (FEAT_SAT, FEAT_CON):
            v = np.array([r["base"][feat] for r in rows])
            se = 1.2533 * sd[feat] / np.sqrt(len(v))
            lines.append(f"{feat} {np.median(v):.2f} vs reference "
                         f"{st.at[feat, 'p50']:.2f} "
                         f"({(np.median(v) - st.at[feat, 'p50']) / se:+.1f} SE)")
        note("C0", "calibration sample median vs reference median: "
             + "; ".join(lines))
        grid = run_calibration_grid(rows)
        grid.to_csv(B("calibration_grid.csv"), index=False,
                    float_format="%.6g")
        levels = solve_levels_a(grid, rows, st)
        levels.to_csv(B("levels_a.csv"), index=False, float_format="%.6g")
        check_levels_vs_v4(levels)
        comp_a = calibrate_comp_a(levels, rows, st)
        if len(comp_a):
            comp_a.to_csv(B("comp_a.csv"), index=False, float_format="%.6g")
        del rows

    with Timer("Definition (b): reach on every scene"):
        tasks = [(r.scene_id, r.source_path) for r in scenes.itertuples()]
        res = run_pool(_reach_task, tasks, {"dims": dims_for_b(), "sd": sd},
                       "reach")
        reach = pd.DataFrame([x for r in res for x in r["reach"]])
        pd.DataFrame([x for r in res for x in r["grid"]]).to_csv(
            B("reach_grid.csv"), index=False, float_format="%.6g")
        reach.to_csv(B("reach.csv"), index=False, float_format="%.6g")
        tables = {r["scene_id"]: r["tables"] for r in res}
        spans, span_summary = spans_from_reach(reach)
        spans.to_csv(B("spans.csv"), index=False, float_format="%.5g")
        span_summary.to_csv(B("span_summary.csv"), index=False,
                            float_format="%.5g")
        print(span_summary[["approach", "feature", "span_median_scene",
                            "span_rendered_q", "span_rendered"]]
              .to_string(index=False))

    with Timer("Rendering, measuring and saving every cell"):
        LA = {(r.family, r.direction, r.level):
              (float(r.param), r.status, float(r.target_sd))
              for r in levels.itertuples()}
        CA = {(r.family, float(r.param)): (float(r.comp_k), r.status)
              for r in comp_a.itertuples()} if len(comp_a) else {}
        span = {(r.approach, r.feature): float(r.span_rendered)
                for r in span_summary.itertuples()}
        ratio = {}
        for feat in (FEAT_SAT, FEAT_CON):
            if ("contrast" if feat == FEAT_CON else "colour") in R_FEATURES:
                ratio[feat] = 1.0 - R_SPAN_SD * sd[feat] / float(
                    st.at[feat, "p50"])
        if any("r" in a["definitions"] for a in APPROACHES.values()):
            note("R0", "definition (r): ratio per full step "
                 + ", ".join(f"{f} q = {q:.4f} (up: x{1 / q:.4f})"
                             for f, q in ratio.items())
                 + "; the corpus-median image moves "
                 + ", ".join(f"{f} {-R_SPAN_SD:+.2f} / "
                             f"{(1 / q - 1) * float(st.at[f, 'p50']) / sd[f]:+.2f} SD"
                             for f, q in ratio.items()))
        payload = {"levels_a": LA, "comp_a": CA, "span": span, "sd": sd,
                   "ratio": ratio, "base_dir": out}
        g4 = {sid: g.to_dict("records") for sid, g in v4grid.groupby(
            "scene_id")}
        tasks = [(r.scene_id, r.source_path, tables.get(r.scene_id, {}),
                  g4.get(r.scene_id, [])) for r in scenes.itertuples()]
        res = run_pool(_render_task, tasks, payload, "scenes")
        cells = pd.DataFrame([x for r in res for x in r["cells"]])
        lad = pd.DataFrame([x for r in res for x in r["ladders"]])
        bases = pd.DataFrame([r["base"] for r in res])
        repro = pd.DataFrame([x for r in res for x in r["repro"]])
        ident = pd.DataFrame([x for r in res for x in r["identity"]])
        pil = pd.DataFrame([x for r in res for x in r["pillow"]])
        v4cells = pd.DataFrame([x for r in res for x in r["v4cells"]])

    with Timer("Measurements, normality, scene fit"):
        cells = add_derived(cells, bases, st)
        if len(v4cells):
            v4cells = add_derived(v4cells.assign(
                approach="V4_ref", definition="v4", cx=0, sx=0), bases, st)
        lad = ladder_achieved(lad, cells)
        # set-aside scenes: flagged everywhere, left out of pooled numbers
        for d_ in (cells, lad, bases, v4cells):
            if len(d_):
                d_["set_aside"] = d_.scene_id.isin(EXCLUDE_SCENES)
        cin = cells[~cells.set_aside]
        lin = lad[~lad.set_aside]
        v4in = v4cells[~v4cells.set_aside] if len(v4cells) else v4cells
        cells.to_csv(B("cells.csv"), index=False, float_format="%.6g")
        bases.to_csv(B("originals.csv"), index=False, float_format="%.6g")
        lad.to_csv(B("ladders.csv"), index=False, float_format="%.6g")
        repro.to_csv(B("reproduction_v4.csv"), index=False)
        ident.to_csv(B("identity.csv"), index=False, float_format="%.6g")
        pil.to_csv(B("pillow_model.csv"), index=False)
        if len(v4cells):
            v4cells.to_csv(B("v4_grid_measured.csv"), index=False,
                           float_format="%.6g")
        ct = crosstalk_table(lin)
        ct.to_csv(B("crosstalk.csv"), index=False, float_format="%.5g")
        art = artefact_table(cin, v4in)
        art.to_csv(B("artefacts.csv"), index=False, float_format="%.5g")
        ev = envelope_violations(cells, st)
        ev["set_aside"] = ev.scene_id.isin(EXCLUDE_SCENES)
        ev.to_csv(B("envelope_violations.csv"), index=False,
                  float_format="%.5g")
        ev = ev[~ev.set_aside]
        pairs, nsum, strad = normality(cin, v4in)
        pairs.to_csv(B("normality_pairs.csv.gz"), index=False,
                     float_format="%.5g", compression="gzip")
        nsum.to_csv(B("normality_summary.csv"), index=False,
                    float_format="%.5g")
        strad.to_csv(B("straddle_counts.csv"), index=False)
        cells[["approach", "definition", "scene_id", "set_aside", "cx",
               "sx", "contrast_name", "colour_name", "z_sat", "z_con",
               "dist_corpus", "dist_corpus_sat", "dist_corpus_con",
               "dist_orig", "dist_orig_sat", "dist_orig_con"]].to_csv(
            B("normality_images.csv"), index=False, float_format="%.5g")
        v4max = ({c: float(v4in[c].max()) for c in
                  ("halo_share", "noise_ratio", "band_new_empty")}
                 if len(v4in) else {})
        fit, _ = scene_fit(cells, v4max)
        fit["set_aside"] = fit.scene_id.isin(EXCLUDE_SCENES)
        _, fit_summary = scene_fit(cin, v4max)
        fit.to_csv(B("scene_fit.csv"), index=False, float_format="%.4g")
        fit_summary.to_csv(B("scene_fit_summary.csv"), index=False)
        checks_after_render(cells, lad, repro, ident, pil)

    with Timer("Viewer"):
        pack = write_viewers(cells, bases)

    clip_res = None
    if RUN_CLIP:
        with Timer("CLIP"):
            RUN["out"] = os.path.join(out, "clip")
            os.makedirs(RUN["out"], exist_ok=True)
            clip_res = run_clip(out, cin, scenes, v4grid)
            RUN["out"] = out

    nbytes = sum(os.path.getsize(os.path.join(dp, f))
                 for dp, _, fs in os.walk(B("images")) for f in fs)
    T = {"n_scenes": int((~scenes.set_aside).sum()),
         "aside": {"spans": spans[spans.set_aside], "fit": fit[fit.set_aside],
                   "reach": reach[reach.scene_id.isin(EXCLUDE_SCENES)]},
         "st": st, "levels_a": levels,
         "comp_a": comp_a, "span_summary": span_summary,
         "reach": reach[~reach.scene_id.isin(EXCLUDE_SCENES)],
         "crosstalk": ct, "artefacts": art, "envelope": ev,
         "fit_summary": fit_summary, "norm_summary": nsum,
         "straddle": strad, "clip": clip_res, "pack": pack,
         "n_images": int(cells.filename.nunique()),
         "images_gb": nbytes / 1e9}
    write_report(T)
    write_checks()
    write_meta("finished")
    print(f"\nDone in {(time.time() - RUN['t0']) / 60:.1f} min.  Report: "
          f"{B('report.txt')}")


if __name__ == "__main__":
    main()
