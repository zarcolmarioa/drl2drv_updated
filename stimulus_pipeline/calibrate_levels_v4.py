"""
calibrate_levels_v4.py  (stimulus pipeline for the v4 axes, step 1)

Chooses the five haze strengths and the five chroma factors of the
stimulus grid, by calibrating each manipulation against the corpus
distribution of the feature the analysis uses.

Why calibrate at all
--------------------
The old stimulus set used parameters picked by hand (haze 0.15 ... 0.60,
chroma 1.0 ... 0.2). The same parameter does different things to
different photographs, and the numbers mean nothing to a reader. Here a
level is defined by HOW MUCH THE MEASURED FEATURE CHANGES:

    "p10" = the change that takes a typical scene (the corpus median)
            down to where the lowest 10% of real photographs sit.

The parameter that achieves it is found by search. The manipulation is
unchanged; only the five values of h and k are chosen differently.

    dimension   manipulation      measured feature   direction
    colour      chroma, k <= 1    sat_mean           falls
    fog         haze,  h > 0      contrast_local     falls

Both features are measured exactly as the corpus was measured (step 04:
CLIP's central square, resized to 256 px), so a target expressed in
corpus percentiles is meaningful for the stimuli.

What this script does
---------------------
  1. Recomputes the corpus percentiles of every feature over two
     populations -- all colour images, and colour images in the keep = 1
     scene categories -- at every percentile the ladders may ask for.
     STATS_POPULATION picks which defines the targets; the default is
     the everyday-scene population, because that is where the stimuli
     come from and the change a level produces is proportional to the
     scene's baseline.
  2. Draws a fixed random sample of eligible corpus images
     (N_CALIB_IMAGES, RANDOM_SEED), applies each manipulation on a
     GRID_N-point parameter grid, and measures the feature each time.
  3. For every target, keeps the parameter at which the MEDIAN change
     over the sample reaches the target, interpolating between grid
     points. A target is capped if the grid ends first, or if the mean
     share of newly clipped pixels would exceed CLIP_MAX_NEW.
  4. Reports, for each candidate ladder, the parameters, the achieved
     median change, the scatter across images, and the cross-talk (how
     far the other dimension's feature moves).

It does NOT choose the final ladder, and it does not choose between the
two fog families either. Both plain haze and brightness-equalised haze
are calibrated on the same feature and carried forward as equals, so
steps 2 and 3 can build two stimulus sets and the choice can be made by
looking at finished stimuli rather than by re-running the pipeline.
Nothing here encodes images or touches the axes.

Where this sits
---------------
  step 1 (this one)         the levels of the grid
  step 2 (select_scenes)    which scenes, screened by hand
  step 3 (build_stimulus)   the stimuli themselves, scored on the axes

Manipulations
-------------
Copied VERBATIM from 09c_manipulation_sweep.py (the v4 pipeline's
manipulation sweep), together with the feature code of step 04. Two
start-up checks verify the copies:

  CHECK 1  the copied feature code reproduces data/features for a
           sample of corpus images (as 09c's check 2 did)
  CHECK 2  the recomputed percentiles match the v4 run's
           corpus_feature_stats.csv where that file has them

and three more verify the calibration itself:

  CHECK 3  the feature changes monotonically along each parameter grid
  CHECK 4  the chroma levels reproduce the v4 run's levels.csv (same
           rule, different image sample, so a tolerance is used and a
           mismatch is a warning, not an error)
  CHECK 5  what the first point of each grid does to the photograph.
           Both should be the identity, and with to_u8 rounding
           correctly both are, to within 0.01 of a feature unit. The
           check is kept because it is what caught the opposite: an
           earlier copy of to_u8 truncated instead of rounding, which
           darkened every manipulated pixel by half a level and showed
           up here as a +0.5 shift in sat_mean at k = 1.
  CHECK 6  the ENVELOPE. Does any level take a feature the design does
           not intend to move outside the corpus 1-99% range? In
           practice this is about brightness: haze adds a bright veil
           and lifts the black point about 1.7x harder than it lowers
           local contrast, at every rung. m_haze_eq is calibrated
           alongside it on the same feature so the two can be compared.
  CHECK 7  the GRID, once per fog family. Each ladder is built as a
           colour x fog grid, every cell measured, and the trials it
           affords counted under the previous pipeline's own thresholds
           -- saturation-matched fog pairs, contrast-matched colour
           pairs, conflict pairs -- against what the measured v2
           stimulus set afforded.
  CHECK 8  the SAMPLE against the reference it is calibrated to. Every
           parameter is set by the sample's median baseline and every
           target by the reference median, so a gap between them moves
           the whole ladder by that proportion. This is the check that
           catches a measurement-path mismatch; without it, one went
           unnoticed for three runs.

Only check 1 is fatal. Checks 2 and 4 compare against a previous run and
warn; checks 3, 5, 6 and 7 are reported for you to act on.

  haze   I' = (1 - h) * I + h * A,  A = the mean RGB of the brightest
         1% of the pixels of the UNMODIFIED central square. Applied
         first, because the airlight comes from the unmodified image.
  chroma CIELAB chroma x k at constant L* and hue (float Lab, with the
         unclipped Lab -> sRGB path). k <= 1 here.

Inputs
------
  INDEX_CSV        data/selection/image_index.csv          (step 02)
  FEATURES_NPY     data/features/features_crop.npy         (step 04)
  FEATURE_DEFS     data/features/feature_definitions.json  (step 04)
  FLAGS_CSV        data/monochrome/image_flags.csv         (step 04a)
  DUP_CSV          data/checks_v4/duplicate_groups.csv     (09a)
  CATEGORY_CSV     everyday_categories.csv                 (scene keep list)
  REF_STATS_CSV    data/sweep_v2/corpus_feature_stats.csv  (optional, check 2)
  REF_LEVELS_CSV   data/sweep_v2/levels.csv                (optional, check 4)

Outputs (under OUTPUT_DIR)
--------------------------
  corpus_feature_stats_v4.csv   percentiles of every feature, one block
                                per population (`population` column:
                                all_colour and everyday_colour)
  calibration_grid.csv          every parameter tried: median change of
                                the calibrated feature, its quartiles,
                                the cross-talk feature, clipped share
  levels.csv                    one row per candidate level: target in
                                feature units and in corpus SD, the
                                parameter, status, achieved median
                                change, scatter, cross-talk
  per_image_levels.csv          each sampled image at each chosen level:
                                its own achieved change (the scatter
                                behind the median)
  level_features.csv            ALL 18 features at each chosen level,
                                so the side effects nobody calibrated
                                on (brightness, colorfulness,
                                chroma_mean) are on record
  calibration.png               feature change against parameter, with
                                the targets marked
  scatter.png                   per-image achieved change at each level
  examples/ladder_fog.png       the fog ladder rendered on real scenes
  examples/ladder_fog_eq.png    the haze_eq ladder, likewise
  examples/ladder_colour.png    the colour ladder, likewise
  examples/compare_fog.png      plain haze against haze_eq at MATCHED
                                contrast targets -- the sheet that
                                decides between the two manipulations
  examples/grid_<id>.png        the full colour x fog grid for a few
                                scenes, plus every cell at full size
                                under examples/cells_<id>/
  examples/grid_eq_<id>.png     the same scenes built with haze_eq
  envelope_violations.csv       levels taking an unintended feature
                                outside the corpus range (check 6)
  grid_cells.csv                every cell of the chosen grid, measured
  grid_pairs.csv                the trials that grid affords (check 7)
  levels_report.txt             summary and the checks

The numbers cannot say whether a level looks like a photograph taken in
fog; that is a judgement for the eye, which is what the examples are
for. Look at them before fixing the ladder.

Usage
-----
    conda activate drl2drv
    python calibrate_levels_v4.py

CPU only, no GPU, no encoder, no axes. The work is
N_CALIB_IMAGES x (GRID_N + 1) x 2 renders for the grid plus
N_CALIB_IMAGES x 12 for the chosen levels: about 9,300 at the defaults,
a few minutes with N_WORKERS set to the number of cores. Set
N_WORKERS = 1 to run in one process, which is the way to get a readable
traceback.
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

# --- Inputs (the v4 pipeline; read, never written) ---
INDEX_CSV       = "data/selection/image_index.csv"
FEATURES_NPY    = "data/features/features_crop.npy"
FEATURE_DEFS    = "data/features/feature_definitions.json"
FLAGS_CSV       = "data/monochrome/image_flags.csv"
DUP_CSV         = "data/checks_v4/duplicate_groups.csv"

# CATEGORY_CSV
#   The scene-category keep list (category, group, tier, keep), the same
#   file select_scenes uses. 82 of its categories have keep = 1.
CATEGORY_CSV    = "everyday_categories.csv"

# Cross-checks; set to None to skip.
REF_STATS_CSV   = "data/sweep_v2/corpus_feature_stats.csv"
REF_LEVELS_CSV  = "data/sweep_v2/levels.csv"

# --- Output ---
OUTPUT_DIR      = "data/levels_v4_ec"

# --- The two dimensions ---
#
# Each entry: the manipulation family, the feature its levels are
# defined on, the direction of the change, and the parameter grid
# (from the identity outwards, as in 09c).
#
# CROSS_FEATURE is the other dimension's feature, measured at every
# parameter so the cross-talk is on record: haze also lowers sat_mean,
# and chroma may move contrast_local slightly because grey is computed
# from gamma-encoded RGB rather than from L*.
FEAT_COLOUR     = "sat_mean"
FEAT_FOG        = "contrast_local"

# GRID_N
#   Parameter grid points per dimension, as in 09c. The level's
#   parameter is interpolated between grid points, so this sets the
#   resolution of the search, not of the levels.
GRID_N          = 24

# HAZE_MAX / CHROMA_MIN
#   Ends of the parameter grids. 09c used haze 0 ... 0.95 and chroma
#   1 ... 0 (0 = fully grey).
HAZE_MAX        = 0.95
CHROMA_MIN      = 0.0

# --- Targets ---
#
# A target is a CHANGE relative to the corpus median of the feature,
# taken from the corpus percentiles of colour images: "p10" means
# p10 - p50. Both dimensions fall, so the targets are negative.
#
# More targets are calibrated than a ladder needs, so that the ladder
# can be chosen from the report without re-running. The mild end is
# sampled finely because that is where the usable ladder turns out to
# live: the haze manipulation's brightness side effect caps the deep
# end (see the envelope check), and the colour dimension's own range
# shrinks as haze grows, so a five-rung ladder may have to fit
# entirely inside p45 ... p15.
TARGET_PERCENTILES = [45, 40, 35, 30, 25, 20, 15, 10, 5, 1]

# INCLUDE_HAZE_EQ
#   Also calibrate m_haze_eq -- haze followed by a linear-light gain
#   back to the original mean grey -- on the SAME feature as plain
#   haze, so the two can be compared rung for rung. Report-only: it
#   never enters a ladder or an example image. The question it answers
#   is whether brightness-equalised haze stays inside the corpus
#   envelope where plain haze does not; two effects pull opposite ways
#   (equalising the mean pulls the black point down, but haze_eq needs
#   a larger h for the same contrast, which pushes it back up) and
#   only measurement settles it.
INCLUDE_HAZE_EQ = True

# CANDIDATE_LADDERS
#   The ladders the report compares. The colour dimension's first level
#   is the unmanipulated colour (k = 1), so it needs four targets; the
#   fog dimension needs five, because every cell carries some haze.
#   (No cell of the grid is the untouched photograph.)
#
#   What separates them, once the numbers are in:
#     - the RANGE in corpus SD, which is the largest difference the
#       dimension can show within one scene, and which should be
#       comparable between the two dimensions so that neither is the
#       easier one to see for a purely physical reason;
#     - the worst CROSS-TALK ratio, since haze also moves sat_mean and
#       the mildest haze rungs move it more than they move
#       contrast_local. The report prints both for every ladder.
#   fog_v2 and colour_v2 reproduce the ranges of the previous stimulus
#   set (haze 0.15 ... 0.60, chroma 1.0 ... 0.2), which has been piloted
#   and whose top rung the eye accepted. They are the benchmark the
#   others are judged against.
#   fog_v2 reproduces the previous set's haze ladder: its 0.15, 0.26,
#   0.37, 0.49, 0.60 interpolate to about p35, p25, p15, p10, p05 in
#   this calibration. colour_v2 likewise matches chroma 1.0 ... 0.2
#   (v2's deepest is -1.33 SD, colour_v2's is -1.46 SD).
CANDIDATE_LADDERS = {
    "fog_v2":      ("fog", [35, 25, 15, 10, 5]),
    "fog_mild":    ("fog", [45, 40, 35, 30, 25]),
    "fog_wide":    ("fog", [40, 25, 15, 10, 5]),
    "fog_deep":    ("fog", [40, 25, 10, 5, 1]),
    "fogeq_v2":    ("fog_eq", [35, 25, 15, 10, 5]),
    "fogeq_wide":  ("fog_eq", [45, 35, 25, 15, 5]),
    "fogeq_deep":  ("fog_eq", [40, 25, 10, 5, 1]),
    "colour_v2":   ("colour", [25, 10, 5, 1]),
    "colour_mild": ("colour", [40, 25, 15, 10]),
}

# --- Calibration sample ---
#
# N_CALIB_IMAGES
#   Images the median is taken over. 09c used 150 (one per category);
#   here they are drawn at random from the eligible pool, as the old
#   pipeline's sweeps did.
N_CALIB_IMAGES  = 150
RANDOM_SEED     = 42

# --- Eligibility of a calibration image (as 09c, plus toning) ---
MIN_SIDE        = 512       # short side of the original, px
EXCLUDE_GREYSCALE = True
EXCLUDE_TONED   = True      # is_toned in image_flags.csv
EXCLUDE_DUPLICATES = True   # every image_id in DUP_CSV, representatives too

# RESTRICT_TO_KEEP_CATEGORIES
#   Draw the calibration sample only from categories with keep = 1 in
#   CATEGORY_CSV -- the everyday environments the scenes will come from.
#
#   This matters because the change a level produces is PROPORTIONAL to
#   the scene's baseline (step 1 established that the per-scene spread is
#   a pure scale effect). Calibrating on the whole corpus sets the
#   parameters by the median of a population that includes mountains,
#   oceans, deserts and icebergs -- none of which can ever be a stimulus.
#   The first run did exactly that: two of its three example scenes were
#   `hangar/indoor` and `iceberg`, both keep = 0.
RESTRICT_TO_KEEP_CATEGORIES = True

# STATS_POPULATION
#   Which population's percentiles define the TARGETS, the SD used
#   throughout, and the envelope.
#
#     "everyday"  colour images in keep = 1 categories. "p05" then means
#                 "as low-contrast as the bottom 5% of everyday-scene
#                 photographs", which is the population the experiment
#                 is about -- derealization is familiar surroundings
#                 becoming strange.
#     "all"       colour images anywhere in the corpus; what the first
#                 run used.
#
#   Both are always computed and both are written to
#   corpus_feature_stats_v4.csv with a `population` column, so the cost
#   of the choice is visible in the report either way.
STATS_POPULATION = "everyday"

# MIN_POPULATION
#   Fewest images a population may have before its percentiles are
#   refused. A guard against a mistyped category name silently reducing
#   the keep=1 population to a handful, which would move every target.
MIN_POPULATION  = 1000

# --- Which measurement path the reference is on ---
#
# REFERENCE_MODE
#   The targets compare a manipulated stimulus against a distribution of
#   real photographs. Both sides have to be measured the same way, and
#   they were not:
#
#     step 04 measured the corpus  original -> 256   (one resize)
#     a stimulus is measured       original -> 512 -> 256  (two)
#
#   Every resize softens fine detail, so the two-step path reads
#   contrast_local about 13% lower. Saturation is unaffected (colour
#   averages out; detail does not). Calibrating a stimulus-path
#   measurement against a corpus-path target therefore made every fog
#   level about 13% stronger than its label.
#
#     "stimulus"  (default) re-measure N_REFERENCE_IMAGES eligible
#                 photographs through the STIMULUS path and take the
#                 percentiles from those. Costs one pass over those
#                 images; puts the targets, the SDs and the envelope on
#                 the same scale as everything downstream -- the
#                 manifest, the pair analysis, the z scores.
#     "corpus"    read the percentiles out of features_crop.npy, as the
#                 earlier runs did. Kept so the difference can be
#                 reproduced.
#
#   Either way the report prints the measured per-feature ratio between
#   the two paths, so the size of the effect is on record.
REFERENCE_MODE  = "stimulus"

# N_REFERENCE_IMAGES
#   Photographs re-measured through the stimulus path. The percentiles
#   wanted go down to the 1st, so this needs to be large: 20,000 puts
#   about 200 images in the bottom percentile.
N_REFERENCE_IMAGES = 20_000
REFERENCE_SEED  = 20250101      # separate from RANDOM_SEED, so changing
                                # this never changes the calibration draw

# --- Geometry (must match step 04 and 09c) ---
IMAGE_SIZE      = 512       # manipulations applied at this size
FEATURE_SIZE    = 256       # features measured at this size
CLIP_PX         = 336       # CLIP's own resolution, for clip_box

# --- Manipulation parameters (as 09c) ---
HAZE_AIRLIGHT_TOP = 1.0     # % brightest pixels whose mean is the airlight
GAMUT_ITERS     = 16
GAMUT_CMAX_SEARCH = 160.0
GAMUT_MARGIN    = 0.995

# CLIP_MAX_NEW
#   A level is capped where the mean share of NEWLY clipped pixels
#   reaches this. Neither manipulation should ever clip (haze blends
#   two valid colours; chroma below 1 shrinks towards grey), so this is
#   a safeguard and a recorded number.
CLIP_MAX_NEW    = 0.01

# --- Example stimuli ---
#
# The numbers say how far each level moves the measured feature. They do
# not say whether the result looks like a photograph somebody took in
# fog, which is a judgement only the eye makes. These settings render
# the chosen ladder on real scenes so it can be looked at.
SAVE_EXAMPLES   = True

# EXAMPLE_FOG / EXAMPLE_COLOUR
#   The ladder to render, by level name. The colour ladder gets k = 1
#   as its first column automatically, so it is named with four levels.
#   The default is fog_v2 + colour_v2, the previous set's ladder: it is
#   the one that has been piloted, and the grid check below reports what
#   it affords against what the measured v2 set afforded. Change these
#   once the envelope and grid sections say which ladder to use.
EXAMPLE_FOG     = ["p35", "p25", "p15", "p10", "p05"]
EXAMPLE_COLOUR  = ["p25", "p10", "p05", "p01"]

# EXAMPLE_FOG_EQ
#   The haze_eq ladder to render alongside. Both manipulations are
#   calibrated on contrast_local, so a rung name means the SAME contrast
#   reduction either way -- which is what makes the comparison sheet
#   readable: each column holds two images that differ only in how the
#   brightness was handled.
#
#   The default reaches p05 because that is the deepest rung haze_eq
#   keeps inside the envelope, where plain haze stops at p15.
EXAMPLE_FOG_EQ  = ["p45", "p35", "p25", "p15", "p05"]

# EXAMPLE_N
#   Scenes per contact sheet, drawn at evenly spaced quantiles of the
#   BASELINE of the relevant feature -- not at random. The change a
#   level produces is proportional to the scene's baseline, so this
#   deliberately shows the weakest scene as well as the strongest.
EXAMPLE_N       = 6

# EXAMPLE_GRID_SCENES
#   Scenes rendered as the full colour x fog grid, which is the layout
#   the experiment actually draws its pairs from.
EXAMPLE_GRID_SCENES = 3

# EXAMPLE_COMPARE_SCENES
#   Scenes on compare_fog.png, which puts plain haze and haze_eq on
#   matched contrast rungs. Two rows per scene, so keep this small.
EXAMPLE_COMPARE_SCENES = 3

# EXAMPLE_FULL_SIZE
#   Also write every cell of those grids as its own IMAGE_SIZE png, for
#   looking at closely rather than as a thumbnail.
EXAMPLE_FULL_SIZE = True

# MANIP_ORDER
#   The order the two manipulations are composed in for a combined
#   cell. This does NOT affect the calibration, which applies each
#   manipulation on its own, but it changes what the off-diagonal cells
#   look like, so step 3 must use the same order.
#
#   HAZE THEN CHROMA, matching build_stimulus_set_loc_con.py. Besides
#   comparability with the existing set, this is the better order: the
#   chroma step also drains the veil, so k -> 0 gives a neutral image at
#   any haze level. The reverse order leaves the airlight's own colour
#   untouched by the chroma step, which puts a saturation FLOOR of
#   roughly h x sat(airlight) under every cell -- eating into the colour
#   dimension's range exactly in the hazy columns where the measured
#   v2 grid shows it is already scarcest (0.84 SD of sat_mean available
#   at h = 0.15, falling to 0.28 SD at h = 0.60).
#
#   The airlight is taken from the UNMODIFIED square either way, so the
#   veil is the same across a whole row of the grid.
MANIP_ORDER     = ("haze", "chroma")

# IDENTITY_SHORT_CIRCUIT
#   Leave the image untouched at k = 1 and h = 0 rather than putting it
#   through the manipulation, as the v2 builder did
#   (`if scale == 1.0: return pil_img.copy()`). Check 5 measures what
#   this costs, and with to_u8 rounding correctly the answer is almost
#   nothing: m_haze at h = 0 is exact, and m_chroma at k = 1 moves
#   sat_mean by under 0.01 units. So this setting is a matter of
#   matching v2 rather than of accuracy either way.
IDENTITY_SHORT_CIRCUIT = True

# --- Envelope check (check 6) ---
#
# ENVELOPE_FEATURES
#   Features that should stay inside the corpus 1-99% range at every
#   level. These are the ones the design does NOT intend to move, so an
#   excursion here is an unintended artefact rather than the
#   manipulation working.
#
#   The colour and contrast families are deliberately left out: pushing
#   sat_mean below the corpus p01 is the colour manipulation's whole
#   purpose, and in the measured v2 set 46% of stimuli sit outside the
#   envelope on sat_mean for exactly that reason. Checking them would
#   flag the design rather than its side effects.
#
#   lum_std and lum_entropy are left out for the same reason, one step
#   removed: global contrast is near-identical in meaning to the local
#   contrast being calibrated, and entropy is a near-monotone function
#   of it -- the histogram narrows because the contrast fell, which is
#   the manipulation working, not an artefact. (An earlier version kept
#   lum_entropy while dropping lum_std, which was inconsistent; entropy
#   was then the only thing standing between haze_eq and a clean sheet
#   at p01, so the inconsistency mattered.)
#
#   What remains is the luminance LEVEL and distribution plus the
#   spatial statistics -- and the luminance level is where plain haze
#   leaves the corpus: in the v2 stimuli, 95% of images at h = 0.49 and
#   100% at h = 0.60 sit above the corpus p99 for lum_p05.
#
#   In practice lum_p05, the black point, is the feature that binds.
ENVELOPE_FEATURES = ["lum_mean", "lum_p05", "lum_p50", "lum_p95",
                     "val_mean", "lum_skew", "spectral_slope"]
ENVELOPE_LO, ENVELOPE_HI = "p01", "p99"

# --- Grid and pair inventory (check 7) ---
#
# GRID_CHECK
#   Render the chosen 5 x 5 ladder on the calibration sample, measure
#   every cell, and enumerate the pairs it affords using the v2
#   thresholds below. This is the question the levels exist to serve:
#   can the grid supply saturation-matched fog pairs, contrast-matched
#   colour pairs, and enough conflict pairs? Reported against what the
#   measured v2 set afforded, as a benchmark.
#
#   These scenes are drawn at random from the eligible pool, so the
#   counts are indicative: step 2 applies percentile bands which will
#   shift them.
GRID_CHECK      = True

# v2's trial-type thresholds, in the features' own units
# (analyse_pairs_loc_con.py).
PAIR_SAT_TOL        = 3.0     # "saturation held constant"
MIN_DSAT_SAT_ONLY   = 12.0
MIN_DFOG_FOG_ONLY   = 4.2
MIN_DSAT_MIXED      = 8.0
MIN_DFOG_MIXED      = 3.0
BALANCE_LO, BALANCE_HI = 0.6, 1.7
FOG_SIGN            = -1.0    # haze LOWERS contrast_local

# What the measured v2 stimulus set afforded, for comparison.
V2_PAIR_COUNTS = {"sat_only": 1118, "fog_only": 1428,
                  "together": 5720, "conflict": 1200}
V2_SCENES = 30

# MIN_PAIRS_PER_TYPE_GRID
#   Every scene must afford at least this many pairs of every type,
#   as build_stimulus_set_loc_con.py required (MIN_PAIRS_PER_TYPE = 3).
MIN_PAIRS_PER_TYPE_GRID = 3

# EXTRA_LEVEL_FEATURES
#   Measure all 18 features (not just the two being calibrated) at each
#   chosen level, and write level_features.csv. This is what puts the
#   brightness confound of haze, and its effect on colorfulness and
#   chroma_mean, on record with a number rather than as an expectation.
EXTRA_LEVEL_FEATURES = True

# --- Checks ---
FEATURE_CHECK_N    = 8      # corpus images re-measured (check 1)
FEATURE_CHECK_RTOL = 1e-4

# STATS_CHECK_TOL_SD
#   Check 2 compares each recomputed percentile with the v4 run's, in
#   units of that feature's corpus SD rather than as a relative
#   difference: several features have a percentile sitting at or near
#   zero (lum_skew, lab_a_mean), where a relative difference is
#   meaningless however close the two numbers are.
STATS_CHECK_TOL_SD = 0.05

# SAMPLE_CHECK_MAX_SE
#   How far the calibration sample's median baseline may sit from the
#   reference median, in standard errors of the median, before check 8
#   warns. 3 allows ordinary sampling noise and catches a systematic
#   offset like the measurement-path mismatch (which showed as 3.5 SE).
SAMPLE_CHECK_MAX_SE = 3.0
LEVEL_CHECK_RTOL   = 0.10   # chroma parameters vs the v4 run's levels.csv

# --- Parallelism ---
N_WORKERS       = 20
CHUNK           = 1


# ============================================================
# Feature code  (verbatim from 04_compute_features.py / 09c)
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
    """Features of a square uint8 image, resized to FEATURE_SIZE first,
    as the corpus was measured.

    `only` restricts the result. When every requested feature belongs to
    the legacy set, only legacy_features is run: it returns bit-identical
    values for those features (all_features passes them through
    unchanged) while skipping the FFT, the histogram, the CIELAB
    conversion and the dark channel. Both of the features this script
    calibrates on are in that set, so the grid runs about three times
    faster. The full set is still computed wherever it is needed -- for
    the baseline of every image, and for check 1.
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
    """The region of the ORIGINAL image that CLIP keeps, in original
    pixel coordinates (floats)."""
    if w <= h:
        nw, nh = n, int(n * h / w)
    else:
        nw, nh = int(n * w / h), n
    left = int(round((nw - n) / 2.0))
    top = int(round((nh - n) / 2.0))
    sx, sy = nw / w, nh / h
    return (left / sx, top / sy, (left + n) / sx, (top + n) / sy)


def central_square(im, size, resample=Image.BILINEAR):
    """CLIP's region of a PIL image, resized to size x size in one
    step."""
    return im.resize((size, size), resample, box=clip_box(*im.size))


# ============================================================
# Manipulations  (verbatim from 09c_manipulation_sweep.py)
# ============================================================

_XN, _ZN = 0.950456, 1.088754                       # OpenCV's D65 white
_M_XYZ = np.array([[0.412453, 0.357580, 0.180423],  # OpenCV's sRGB matrix
                   [0.212671, 0.715160, 0.072169],
                   [0.019334, 0.119193, 0.950227]])
_M_RGB = np.linalg.inv(_M_XYZ).astype(np.float32)


def to_u8(x):
    # np.round matters: without it this truncates, which darkens every
    # manipulated pixel by half a level on average. Check 1 would not
    # catch it, because that check measures unmanipulated images.
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


def mean_grey(u8):
    return float(cv2.cvtColor(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR),
                              cv2.COLOR_BGR2GRAY).mean())


def _f(img):
    return img.astype(np.float32)


def lab_of(u8):
    return cv2.cvtColor(u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def rgb_of_f(lab):
    """CIELAB -> sRGB on the 0-255 scale, UNCLIPPED, so gamut clipping
    is visible to the caller instead of being hidden by OpenCV."""
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
    """Largest CIELAB chroma inside the sRGB gamut at lightness L and
    hue angle h, per pixel, by bisection. Only needed for k > 1, which
    this script never uses; kept so the copy stays verbatim."""
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
    """Chroma x k at constant lightness and hue."""
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


FUNCS = {"chroma": m_chroma, "haze": m_haze, "haze_eq": m_haze_eq}


IDENTITY_PARAM = {"chroma": 1.0, "haze": 0.0, "haze_eq": 0.0}


def apply(fam, img, p, ctx):
    """(uint8 image, share of NEWLY clipped pixels): pixels with a
    channel that was inside the range and was pushed outside it."""
    if fam is None:
        return img, 0.0
    if IDENTITY_SHORT_CIRCUIT and p == IDENTITY_PARAM.get(fam):
        return img, 0.0
    x = FUNCS[fam](img, p, ctx)
    inside = (img > 0) & (img < 255)
    out = (((x < -0.5) | (x > 255.5)) & inside).any(axis=2).mean()
    return to_u8(x), float(out)


def ctx_of(r):
    """The per-image context the manipulations need: the airlight for
    haze, and the original mean grey for haze_eq to restore."""
    return {"airlight": r["airlight"], "lum_mean_512": r["lum_mean_512"]}


def airlight_of(sq):
    """The mean RGB of the brightest HAZE_AIRLIGHT_TOP % of the pixels
    of the unmodified square (09c)."""
    g = cv2.cvtColor(cv2.cvtColor(sq, cv2.COLOR_RGB2BGR),
                     cv2.COLOR_BGR2GRAY)
    thr = np.percentile(g, 100 - HAZE_AIRLIGHT_TOP)
    return sq[g >= thr].reshape(-1, 3).mean(0)


# ============================================================
# The parameter grids and the dimensions
# ============================================================

def lin(a, b, n):
    return np.linspace(a, b, n + 1)


DIMENSIONS = {
    # name:   (family, calibrated feature, cross-talk feature, grid)
    "fog":    ("haze", FEAT_FOG, FEAT_COLOUR, lin(0.0, HAZE_MAX, GRID_N)),
    "colour": ("chroma", FEAT_COLOUR, FEAT_FOG, lin(1.0, CHROMA_MIN, GRID_N)),
}

# fog_eq is calibrated on the same feature as fog, so the two compare
# rung for rung: the same contrast reduction, a different treatment of
# brightness. BOTH are first-class -- the pipeline carries two fog
# families all the way to two stimulus sets, so the choice between them
# can be made by looking at finished stimuli rather than by re-running
# everything. Nothing here decides which one the experiment uses.
if INCLUDE_HAZE_EQ:
    DIMENSIONS["fog_eq"] = ("haze_eq", FEAT_FOG, FEAT_COLOUR,
                            lin(0.0, HAZE_MAX, GRID_N))
REPORT_ONLY = set()

# FOG_VARIANTS
#   The fog families built as full stimulus grids, each paired with the
#   one colour ladder. (dimension, manipulation, ladder setting name).
FOG_VARIANTS = [("fog", "haze", "EXAMPLE_FOG")] + (
    [("fog_eq", "haze_eq", "EXAMPLE_FOG_EQ")] if INCLUDE_HAZE_EQ else [])


# ============================================================
# Corpus percentiles
# ============================================================

def load_categories(n, idx, report):
    """Boolean mask: is this image in a keep = 1 scene category?

    Categories are matched on the Places365 slash form, which both
    image_index.csv and everyday_categories.csv use, so no name
    conversion is needed. Returns (mask, n_keep_categories); an
    all-true mask if the file is missing.
    """
    if not (CATEGORY_CSV and os.path.isfile(CATEGORY_CSV)):
        if RESTRICT_TO_KEEP_CATEGORIES or STATS_POPULATION == "everyday":
            raise FileNotFoundError(
                f"{CATEGORY_CSV} is required by "
                "RESTRICT_TO_KEEP_CATEGORIES / STATS_POPULATION.")
        return np.ones(n, bool), 0
    cats = pd.read_csv(CATEGORY_CSV)
    keep = set(cats.loc[cats["keep"].astype(int) == 1, "category"])
    in_index = set(idx.category.unique())
    missing = sorted(keep - in_index)
    if missing:
        warnings.warn(f"{len(missing)} keep=1 categories are not in the "
                      f"image index, e.g. {missing[:3]}")
    mask = idx.category.isin(keep).to_numpy()
    print(f"  scene categories: {len(keep)} with keep=1, covering "
          f"{int(mask.sum()):,} images")
    report.append(("keep=1 categories", f"{len(keep)}"))
    return mask, len(keep)


def _percentiles(F, mask, n, want, label):
    """Percentiles, mean and SD of every feature over the images `mask`
    selects, read in chunks so the feature array is never fully loaded.
    """
    vals = []
    for a in range(0, n, 200_000):
        b = min(a + 200_000, n)
        X = np.asarray(F[a:b], np.float64)
        m = mask[a:b] & np.isfinite(X).all(1)
        if m.any():
            vals.append(X[m])
    V = np.concatenate(vals) if vals else np.zeros((0, len(CROP)))
    if len(V) < MIN_POPULATION:
        raise RuntimeError(f"only {len(V)} images in population "
                           f"'{label}'; cannot take percentiles.")
    out = pd.DataFrame({"population": label, "feature": CROP,
                        "n_images": len(V), "mean": V.mean(0),
                        "sd": V.std(0, ddof=1)})
    for q in sorted(want):
        out[f"p{q:02d}"] = np.percentile(V, q, axis=0)
    return out


def load_corpus(report):
    """The eligible pool and the corpus percentiles of every feature.

    Percentiles are computed over TWO populations -- all colour images,
    and colour images in keep = 1 scene categories -- and
    STATS_POPULATION picks which one defines the targets. Both are
    written out.

    Eligibility (09c's rule plus toning): a colour photograph, not in a
    duplicate group, not toned, short side >= MIN_SIDE, all features
    finite, and -- when RESTRICT_TO_KEEP_CATEGORIES -- in a keep = 1
    category.
    """
    print("\n" + "=" * 60)
    print("CORPUS")
    print("=" * 60)
    for p in (INDEX_CSV, FEATURES_NPY, FEATURE_DEFS, FLAGS_CSV):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    idx = pd.read_csv(INDEX_CSV,
                      usecols=["image_id", "image_key", "category",
                               "path", "width", "height"])
    n = len(idx)
    if not (idx.image_id.to_numpy() == np.arange(n)).all():
        raise ValueError("image_id must be 0..N-1 in row order.")

    with open(FEATURE_DEFS) as f:
        defs = json.load(f)
    names = defs["crop"]
    if names != CROP:
        raise ValueError(
            "The feature list in this script does not match "
            f"{FEATURE_DEFS}:\n  script: {CROP}\n  file  : {names}")
    if defs["settings"]["feature_size"] != FEATURE_SIZE:
        raise ValueError("FEATURE_SIZE differs from step 04's.")

    F = np.load(FEATURES_NPY, mmap_mode="r")
    if F.shape != (n, len(CROP)):
        raise ValueError(f"features_crop.npy is {F.shape}, expected "
                         f"{(n, len(CROP))}.")

    flags = pd.read_csv(FLAGS_CSV).set_index("image_id").reindex(
        np.arange(n))
    colour = ~flags["is_greyscale"].astype(bool).to_numpy()
    toned = (flags["is_toned"].astype(bool).to_numpy()
             if "is_toned" in flags.columns else np.zeros(n, bool))

    dup = np.zeros(n, bool)
    if EXCLUDE_DUPLICATES and DUP_CSV and os.path.isfile(DUP_CSV):
        dup_ids = pd.read_csv(DUP_CSV)["image_id"].astype(int).to_numpy()
        dup[dup_ids] = True
    elif EXCLUDE_DUPLICATES:
        warnings.warn(f"{DUP_CSV} not found; duplicates not excluded.")

    side = np.minimum(idx.width.to_numpy(), idx.height.to_numpy())

    keep_cat, n_keep = load_categories(n, idx, report)

    # --- Percentiles over both populations, in chunks ---
    # every target the ladders may ask for, plus the reference points
    # the report and the envelope check use
    want = sorted(set(TARGET_PERCENTILES) | {1, 50, 75, 90, 99})
    tables = {
        "all_colour": _percentiles(F, colour, n, want, "all_colour"),
        "everyday_colour": _percentiles(F, colour & keep_cat, n, want,
                                        "everyday_colour"),
    }
    chosen = ("everyday_colour" if STATS_POPULATION == "everyday"
              else "all_colour")
    if STATS_POPULATION not in ("everyday", "all"):
        raise ValueError(f"STATS_POPULATION must be 'everyday' or 'all', "
                         f"not {STATS_POPULATION!r}")
    stats = tables[chosen].drop(columns=["population"]).reset_index(drop=True)
    print(f"  {n:,} images; {int(colour.sum()):,} colour")
    for k, t in tables.items():
        mark = "  <- targets" if k == chosen else ""
        print(f"    percentiles over {k:<16} "
              f"{int(t.n_images.iloc[0]):>10,}{mark}")
    report.append(("target population",
                   f"{chosen} ({int(stats.n_images.iloc[0]):,} images)"))

    ok = np.isfinite(np.asarray(F[:, [CROP.index(FEAT_COLOUR),
                                      CROP.index(FEAT_FOG)]], np.float64)
                     ).all(1)
    eligible = ok & (side >= MIN_SIDE) & ~dup
    if EXCLUDE_GREYSCALE:
        eligible &= colour
    if EXCLUDE_TONED:
        eligible &= ~toned
    if RESTRICT_TO_KEEP_CATEGORIES:
        eligible &= keep_cat
    print(f"  eligible for calibration: {int(eligible.sum()):,} "
          f"(colour {int(colour.sum()):,}, >= {MIN_SIDE}px "
          f"{int((side >= MIN_SIDE).sum()):,}, duplicates "
          f"{int(dup.sum()):,}, toned {int(toned.sum()):,}"
          + (f", keep=1 categories {int(keep_cat.sum()):,}"
             if RESTRICT_TO_KEEP_CATEGORIES else "") + ")")
    report += [("corpus images", f"{n:,}"),
               ("colour images", f"{int(colour.sum()):,}"),
               ("eligible for calibration", f"{int(eligible.sum()):,}")]
    return idx, F, stats, tables, eligible


def _ref_task(args):
    """Measure one photograph through the STIMULUS path: central square
    at IMAGE_SIZE, then down to FEATURE_SIZE, exactly as a rendered
    stimulus is measured."""
    i, path = args
    try:
        with Image.open(path) as im:
            sq = np.asarray(central_square(im.convert("RGB"), IMAGE_SIZE),
                            np.uint8)
    except Exception:
        return i, None
    v = measure(sq)
    return i, np.asarray([v[f] for f in CROP], np.float64)


def measure_reference(idx, F, eligible, report):
    """Percentiles of every feature, measured through the stimulus path.

    Returns (stats, ratios). `ratios` is the per-feature median of
    stimulus-path / corpus-path on the SAME photographs, which is the
    diagnostic: it says how much each feature shifts between the two
    measurement paths, and it is free because the corpus values are
    already stored.
    """
    print("\n" + "=" * 60)
    print("REFERENCE DISTRIBUTION (stimulus measurement path)")
    print("=" * 60)
    rng = np.random.default_rng(REFERENCE_SEED)
    pool = np.where(eligible)[0]
    n = min(N_REFERENCE_IMAGES, len(pool))
    if n < MIN_POPULATION:
        raise RuntimeError(f"only {len(pool)} eligible images; cannot "
                           "build a reference distribution.")
    ids = np.sort(rng.choice(pool, size=n, replace=False))
    tasks = [(int(i), idx.path.iat[int(i)]) for i in ids]
    print(f"  re-measuring {n:,} eligible photographs "
          f"(original -> {IMAGE_SIZE} -> {FEATURE_SIZE})")

    t0 = time.time()
    got = {}
    if N_WORKERS > 1:
        with Pool(N_WORKERS, initializer=_init, initargs=(None,)) as pool_:
            for k, (i, v) in enumerate(pool_.imap_unordered(_ref_task, tasks,
                                                            64), 1):
                if v is not None:
                    got[i] = v
                if k % 2000 == 0:
                    print(f"    {k:,}/{n:,}", end="\r")
    else:
        _init(None)
        for k, t in enumerate(tasks, 1):
            i, v = _ref_task(t)
            if v is not None:
                got[i] = v
    print(f"\n  {len(got):,} measured in {time.time() - t0:.0f}s")
    if len(got) < MIN_POPULATION:
        raise RuntimeError(f"only {len(got)} images could be measured.")

    keep_ids = np.array(sorted(got))
    V = np.asarray([got[i] for i in keep_ids], np.float64)
    V = V[np.isfinite(V).all(1)]

    want = sorted(set(TARGET_PERCENTILES) | {1, 50, 75, 90, 99})
    stats = pd.DataFrame({"population": "stimulus_path", "feature": CROP,
                          "n_images": len(V), "mean": V.mean(0),
                          "sd": V.std(0, ddof=1)})
    for q in want:
        stats[f"p{q:02d}"] = np.percentile(V, q, axis=0)

    # --- the ratio between the two measurement paths ---
    stored = np.asarray(F[keep_ids], np.float64)
    stim = np.asarray([got[i] for i in keep_ids], np.float64)
    ok = np.isfinite(stored) & np.isfinite(stim) & (np.abs(stored) > 1e-6)
    ratios = np.array([np.median(stim[ok[:, j], j] / stored[ok[:, j], j])
                       if ok[:, j].any() else np.nan
                       for j in range(len(CROP))])
    print("  stimulus path / corpus path, per feature "
          "(1.00 = the two agree):")
    order = np.argsort(np.abs(ratios - 1))[::-1]
    for j in order[:6]:
        print(f"    {CROP[j]:<18} {ratios[j]:.4f}")
    report.append(("reference images", f"{len(V):,}"))
    report.append(("worst path ratio",
                   f"{CROP[order[0]]} {ratios[order[0]]:.4f}"))
    return stats, pd.Series(ratios, index=CROP)


def check_stats(stats, report):
    """CHECK 2: the recomputed percentiles match the v4 run's."""
    if not (REF_STATS_CSV and os.path.isfile(REF_STATS_CSV)):
        print("  check 2 (percentiles vs the v4 run): file not found; "
              "skipped")
        report.append(("check 2 percentiles", "skipped"))
        return
    ref = pd.read_csv(REF_STATS_CSV).set_index("feature")
    cols = [c for c in ["p01", "p10", "p25", "p50", "p75", "p90", "p99"]
            if c in ref.columns]
    worst, where = 0.0, ""
    for f in CROP:
        if f not in ref.index:
            continue
        sd = max(abs(float(ref.at[f, "sd"])), 1e-9)
        for c in cols:
            a = float(stats.loc[stats.feature == f, c].iloc[0])
            b = float(ref.at[f, c])
            d = abs(a - b) / sd           # in corpus SD of that feature
            if d > worst:
                worst, where = d, f"{f} {c}: {a:.4g} vs {b:.4g}"
    ok = worst <= STATS_CHECK_TOL_SD
    print(f"  check 2 (percentiles vs the v4 run): worst difference "
          f"{worst:.3f} SD ({where}) -> {'PASS' if ok else 'WARN'}")
    report.append(("check 2 percentiles",
                   f"{'PASS' if ok else 'WARN'} ({worst:.3f} SD)"))
    if not ok:
        warnings.warn("Recomputed percentiles differ from the v4 run's; "
                      "check the colour mask and the feature columns.")


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
        raise RuntimeError(
            "The copied feature code does not reproduce data/features. "
            "The copies in this script have drifted from step 04, or "
            "the images on disk have changed.")


# ============================================================
# The calibration sample
# ============================================================

def draw_sample(idx, eligible, rng, report):
    """N_CALIB_IMAGES eligible images, drawn at random, loaded as
    512 px central squares with their airlight."""
    pool = np.where(eligible)[0]
    if len(pool) < N_CALIB_IMAGES:
        raise RuntimeError(f"only {len(pool)} eligible images.")
    ids = np.sort(rng.choice(pool, size=N_CALIB_IMAGES, replace=False))

    rows, bad = [], 0
    for i in ids:
        try:
            with Image.open(idx.path.iat[int(i)]) as im:
                sq = np.asarray(central_square(im.convert("RGB"),
                                               IMAGE_SIZE), np.uint8)
        except Exception:
            bad += 1
            continue
        rows.append({"image_id": int(i),
                     "category": idx.category.iat[int(i)],
                     "path": idx.path.iat[int(i)],
                     "square": sq,
                     "airlight": airlight_of(sq),
                     # m_haze_eq restores this mean grey after hazing
                     "lum_mean_512": mean_grey(sq)})
    if bad:
        warnings.warn(f"{bad} sampled images could not be read.")
    print(f"  calibration sample: {len(rows)} images, "
          f"{len({r['category'] for r in rows})} categories")
    report.append(("calibration images", str(len(rows))))
    return rows


# ============================================================
# Applying the grid  (workers)
# ============================================================

W_ROWS = None


def _init(rows):
    global W_ROWS
    W_ROWS = rows
    cv2.setNumThreads(1)
    warnings.filterwarnings("ignore")


def _one_param(dim, p):
    """One parameter applied to every image held by this worker.

    Returns the per-image change of the calibrated feature, of the
    cross-talk feature, and the share of newly clipped pixels.
    """
    fam, feat, cross, _ = DIMENSIONS[dim]
    d_feat, d_cross, clip = [], [], []
    for r in W_ROWS:
        ctx = ctx_of(r)
        out, c = apply(fam, r["square"], p, ctx)
        v = measure(out, [feat, cross])
        d_feat.append(v[feat] - r["base"][feat])
        d_cross.append(v[cross] - r["base"][cross])
        clip.append(c)
    return np.array(d_feat), np.array(d_cross), np.array(clip)


def _task(args):
    """Grid task: (dimension, grid index, parameter)."""
    dim, gi, p = args
    return (dim, gi, p) + _one_param(dim, p)


def _task_level(args):
    """Level task: (dimension, level name, parameter).

    Returns the same three arrays as a grid task, plus -- when
    EXTRA_LEVEL_FEATURES is on -- the full 18-feature matrix of the
    manipulated images, so that the side effects nobody calibrated on
    (brightness, colorfulness, chroma_mean) are measured rather than
    assumed. Only 12 parameters go through this path, so measuring
    everything here costs little.
    """
    dim, level, p = args
    out = _one_param(dim, p)
    if not EXTRA_LEVEL_FEATURES:
        return (dim, level, p) + out + (None,)
    fam = DIMENSIONS[dim][0]
    F = []
    for r in W_ROWS:
        img, _ = apply(fam, r["square"], p, ctx_of(r))
        v = measure(img)
        F.append([v[f] for f in CROP])
    return (dim, level, p) + out + (np.asarray(F, np.float64),)


def run_tasks(fn, tasks, rows, label):
    """Run `fn` over `tasks` in a pool of N_WORKERS, each worker holding
    the loaded calibration images. Falls back to one process when
    N_WORKERS is 1, which is also the way to get a readable traceback.
    """
    print(f"  {len(tasks)} parameters x {len(rows)} images = "
          f"{len(tasks) * len(rows):,} renders ({label})")
    t0 = time.time()
    out = []
    if N_WORKERS > 1:
        with Pool(N_WORKERS, initializer=_init, initargs=(rows,)) as pool:
            for k, r in enumerate(pool.imap_unordered(fn, tasks, CHUNK), 1):
                out.append(r)
                print(f"    {k}/{len(tasks)}", end="\r")
    else:
        _init(rows)
        for k, t in enumerate(tasks, 1):
            out.append(fn(t))
            print(f"    {k}/{len(tasks)}", end="\r")
    print(f"\n  elapsed {time.time() - t0:.0f}s")
    return out


def run_grid(rows, report):
    """Every parameter of both grids applied to every sampled image."""
    print("\n" + "=" * 60)
    print("APPLYING THE PARAMETER GRIDS")
    print("=" * 60)
    for r in rows:
        r["base"] = measure(r["square"])

    tasks = [(dim, gi, float(p))
             for dim, (_, _, _, grid) in DIMENSIONS.items()
             for gi, p in enumerate(grid)]
    results = run_tasks(_task, tasks, rows, "grid")

    grid_rows, per_image = [], {}
    for dim, gi, p, d_feat, d_cross, clip in results:
        fam, feat, cross, _ = DIMENSIONS[dim]
        per_image[(dim, gi)] = d_feat
        grid_rows.append({
            "dimension": dim, "family": fam, "grid_index": gi,
            "param": p, "feature": feat, "cross_feature": cross,
            "median_change": float(np.median(d_feat)),
            "q25_change": float(np.percentile(d_feat, 25)),
            "q75_change": float(np.percentile(d_feat, 75)),
            "median_cross_change": float(np.median(d_cross)),
            "mean_clipped_new": float(np.mean(clip)),
        })
    grid = (pd.DataFrame(grid_rows)
            .sort_values(["dimension", "grid_index"])
            .reset_index(drop=True))
    return grid, per_image


# ============================================================
# Choosing the levels
# ============================================================

def interp_param(grid_d, target):
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
    over = np.where(clip > CLIP_MAX_NEW)[0]
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


def choose_levels(grid, stats, rows, report):
    """One row per (dimension, target percentile): the target in feature
    units and in corpus SD, the parameter, its status, and what it
    actually achieves on the sample.

    The parameter comes from the grid by interpolation; it is then
    applied to every sampled image again, so that the achieved change is
    measured rather than interpolated. Note that the parameter is set
    where the MEDIAN change reaches the target, so about half the sample
    reaches it by construction (`share_reaching_target` near 0.50); a
    clearly smaller share means the level was capped.
    """
    print("\n" + "=" * 60)
    print("LEVELS")
    print("=" * 60)
    st = stats.set_index("feature")

    # --- the parameter of every level, from the grid ---
    rowsout = []
    for dim, (fam, feat, cross, _) in DIMENSIONS.items():
        g = grid[grid.dimension == dim]
        p50, sd = float(st.at[feat, "p50"]), float(st.at[feat, "sd"])
        for q in TARGET_PERCENTILES:
            target = float(st.at[feat, f"p{q:02d}"]) - p50
            param, status = interp_param(g, target)
            rowsout.append({"dimension": dim, "family": fam,
                            "feature": feat, "level": f"p{q:02d}",
                            "percentile": q, "target_change": target,
                            "target_sd": target / sd, "param": param,
                            "status": status})

    # --- apply each level to the whole sample, in the pool ---
    tasks = [(r["dimension"], r["level"], float(r["param"]))
             for r in rowsout if np.isfinite(r["param"])]
    results = run_tasks(_task_level, tasks, rows, "levels")
    got = {(dim, lv): (d, c, k) for dim, lv, _, d, c, k, _F in results}
    full = {(dim, lv): F for dim, lv, _, _d, _c, _k, F in results
            if F is not None}

    per_img = []
    for r in rowsout:
        key = (r["dimension"], r["level"])
        if key not in got:
            continue
        dim, feat = r["dimension"], r["feature"]
        sd = float(st.at[feat, "sd"])
        cross = DIMENSIONS[dim][2]
        d, c, k = got[key]
        target = r["target_change"]
        r.update({
            "achieved_change": float(np.median(d)),
            "achieved_sd": float(np.median(d) / sd),
            "achieved_q25": float(np.percentile(d, 25)),
            "achieved_q75": float(np.percentile(d, 75)),
            "share_reaching_target": float((d <= target).mean() if target < 0
                                           else (d >= target).mean()),
            "cross_change_median": float(np.median(c)),
            "cross_change_sd": float(np.median(c) / float(st.at[cross, "sd"])),
            "mean_clipped_new": float(np.mean(k)),
        })
        per_img.append(pd.DataFrame({
            "image_id": [x["image_id"] for x in rows],
            "category": [x["category"] for x in rows],
            "dimension": dim, "level": r["level"], "param": r["param"],
            "change": d, "cross_change": c,
            "base": [x["base"][feat] for x in rows], "clipped_new": k}))

    for r in rowsout:
        print(f"  {r['dimension']:<7} {r['level']}  target "
              f"{r['target_change']:+8.3f} ({r['target_sd']:+.2f} SD)  "
              f"param {r['param']:7.4f}  {r['status']:<22} achieved "
              f"{r.get('achieved_change', float('nan')):+8.3f}")

    levels = pd.DataFrame(rowsout)
    per_image = (pd.concat(per_img, ignore_index=True) if per_img
                 else pd.DataFrame())
    n_capped = int((levels.status != "ok").sum())
    report.append(("levels calibrated", f"{len(levels)} ({n_capped} capped)"))

    # --- every feature at every chosen level, not only the two ---
    side = []
    if full:
        base = np.asarray([[r["base"][f] for f in CROP] for r in rows],
                          np.float64)
        sd = stats.set_index("feature")["sd"]
        for (dim, lv), F in sorted(full.items()):
            d = np.median(F - base, axis=0)
            side.append(pd.DataFrame({
                "dimension": dim, "level": lv, "feature": CROP,
                "base_median": np.median(base, axis=0),
                "achieved_median": np.median(F, axis=0),
                "median_change": d,
                "change_sd": d / sd.reindex(CROP).to_numpy(),
            }))
    side_effects = (pd.concat(side, ignore_index=True) if side
                    else pd.DataFrame())
    return levels, per_image, side_effects


def check_monotonic(grid, report):
    """CHECK 3: the calibrated feature changes monotonically along each
    parameter grid."""
    bad = []
    for dim in DIMENSIONS:
        g = grid[grid.dimension == dim].sort_values("grid_index")
        d = np.diff(g.median_change.to_numpy())
        if not (np.all(d <= 1e-9) or np.all(d >= -1e-9)):
            bad.append(dim)
    ok = not bad
    print(f"  check 3 (monotonic grids): "
          f"{'PASS' if ok else 'FAIL ' + str(bad)}")
    report.append(("check 3 monotonic grids",
                   "PASS" if ok else f"FAIL: {bad}"))
    return ok


def check_identity(rows, stats, report):
    """CHECK 5: what the first point of each grid does to the image.

    Both grids start at the parameter that should leave the photograph
    alone: haze 0 and chroma 1. Haze is exactly the identity, because it
    blends the image with the airlight in floating point and h = 0 drops
    the airlight term. Chroma is NOT: it converts to CIELAB and back in
    float32 and rounds to 8 bits, so k = 1 returns the image shifted by
    at most one level per channel.

    The size of that shift is reported here because it decides one thing
    in the stimulus builder: whether the k = 1 column of the grid is the
    untouched photograph or the photograph put through m_chroma. Either
    is defensible, but every cell of a row must be treated the same way,
    and the number belongs on record.
    """
    st = stats.set_index("feature")
    lines = []
    for dim, (fam, feat, cross, _g) in DIMENSIONS.items():
        p = IDENTITY_PARAM[fam]
        d_own, d_cross = [], []
        for r in rows:
            # deliberately bypasses IDENTITY_SHORT_CIRCUIT: the point is
            # to measure what the short circuit avoids, so the number is
            # on record whichever way the setting is left
            x = to_u8(FUNCS[fam](r["square"], p, ctx_of(r)))
            v = measure(x, [feat, cross])
            d_own.append(v[feat] - r["base"][feat])
            d_cross.append(v[cross] - r["base"][cross])
        mo, mc = float(np.median(d_own)), float(np.median(d_cross))
        sd = float(st.at[feat, "sd"])
        sd_c = float(st.at[cross, "sd"])
        lines.append(f"{dim} ({fam} = {p:g}): {feat} "
                     f"{mo:+.3f} ({mo / sd:+.3f} SD), "
                     f"{cross} {mc:+.3f} ({mc / sd_c:+.3f} SD)")
    print("  check 5 (identity point of each grid):")
    for ln in lines:
        print(f"      {ln}")
    report.append(("check 5 identity points", "; ".join(lines)))
    return lines


def check_sample(rows, stats, report):
    """CHECK 8: is the calibration sample typical of the reference?

    Every parameter is set by the sample's median baseline, and every
    target by the reference median, so a gap between them puts the whole
    ladder off by that proportion. This is the check that would have
    caught the measurement-path mismatch on the first run instead of the
    third: the sample's contrast_local median came out 13% below the
    reference, because the two were measured along different paths.

    Reported in standard errors of the median, so sampling noise and a
    systematic offset can be told apart.
    """
    print("\n" + "=" * 60)
    print("SAMPLE VS REFERENCE")
    print("=" * 60)
    st = stats.set_index("feature")
    lines, worst = [], 0.0
    for feat in (FEAT_COLOUR, FEAT_FOG):
        v = np.array([r["base"][feat] for r in rows], float)
        med = float(np.median(v))
        ref = float(st.at[feat, "p50"])
        sd = float(st.at[feat, "sd"])
        se = 1.2533 * sd / np.sqrt(len(v))      # SE of a median
        z = (med - ref) / se if se > 0 else np.nan
        lines.append(f"{feat}: sample {med:.2f} vs reference {ref:.2f} "
                     f"({med - ref:+.2f}, {z:+.1f} SE, "
                     f"{(med / ref - 1) * 100:+.1f}%)")
        worst = max(worst, abs(z))
    ok = worst <= SAMPLE_CHECK_MAX_SE
    for ln in lines:
        print(f"      {ln}")
    print(f"  check 8 (sample vs reference): worst {worst:.1f} SE -> "
          f"{'PASS' if ok else 'WARN'}")
    report.append(("check 8 sample vs reference",
                   f"{'PASS' if ok else 'WARN'} ({worst:.1f} SE)"))
    if not ok:
        warnings.warn(
            "The calibration sample's median baseline differs from the "
            "reference by more than sampling noise explains. Every level "
            "is off by that proportion. Check that the sample and the "
            "reference are measured along the same path, and that the "
            "eligibility rule matches the reference population.")
    return lines


def check_envelope(side_effects, stats, report):
    """CHECK 6: does any level take a feature the design does NOT
    intend to move outside the corpus 1-99% range?

    The levels are calibrated on one feature each, and that feature is
    expected to leave the corpus range -- that is the manipulation
    working. The question here is about everything else, and in
    practice about the luminance distribution: haze adds a bright veil,
    which lifts the black point far harder than it lowers local
    contrast. In the measured v2 stimulus set, every image at h = 0.60
    sat above the corpus 99th percentile for lum_p05.

    Returns a table of violations, and the deepest level of each
    dimension that stays inside.
    """
    print("\n" + "=" * 60)
    print("ENVELOPE CHECK")
    print("=" * 60)
    if not len(side_effects):
        print("  no side-effect table (EXTRA_LEVEL_FEATURES is off); "
              "skipped")
        report.append(("check 6 envelope", "skipped"))
        return pd.DataFrame(), {}

    st = stats.set_index("feature")
    rows = []
    for r in side_effects.itertuples():
        if r.feature not in ENVELOPE_FEATURES:
            continue
        lo = float(st.at[r.feature, ENVELOPE_LO])
        hi = float(st.at[r.feature, ENVELOPE_HI])
        v = float(r.achieved_median)
        if v < lo or v > hi:
            rows.append({"dimension": r.dimension, "level": r.level,
                         "feature": r.feature, "achieved": v,
                         "corpus_lo": lo, "corpus_hi": hi,
                         "outside_by_sd":
                             (v - hi if v > hi else v - lo)
                             / float(st.at[r.feature, "sd"])})
    viol = pd.DataFrame(rows)

    # The deepest level of each dimension with no violation. Levels are
    # ordered by TARGET_PERCENTILES, deepest last.
    order = [f"p{q:02d}" for q in TARGET_PERCENTILES]
    deepest = {}
    for dim in DIMENSIONS:
        bad = set(viol[viol.dimension == dim].level) if len(viol) else set()
        ok = [lv for lv in order if lv not in bad]
        deepest[dim] = ok[-1] if ok else None

    if len(viol):
        print(f"  {len(viol)} violations over "
              f"{viol.level.nunique()} levels:")
        for dim, g in viol.groupby("dimension"):
            feats = ", ".join(sorted(set(g.feature)))
            print(f"    {dim:<7} {sorted(set(g.level))}  ({feats})")
    else:
        print("  no level takes an unintended feature outside the "
              "corpus range")
    print("  deepest level staying inside: "
          + ", ".join(f"{d} {lv}" for d, lv in deepest.items()))
    report.append(("check 6 envelope",
                   f"{len(viol)} violations; deepest inside: "
                   + ", ".join(f"{d} {lv}" for d, lv in deepest.items())))
    return viol, deepest


def check_reference_levels(levels, report):
    """CHECK 4: the chroma levels reproduce the v4 run's levels.csv.

    The v4 run calibrated chroma on sat_mean with the same rule, on a
    different sample (150 images, one per category), so the parameters
    should be close but need not be equal.
    """
    if not (REF_LEVELS_CSV and os.path.isfile(REF_LEVELS_CSV)):
        print("  check 4 (chroma levels vs the v4 run): file not found; "
              "skipped")
        report.append(("check 4 chroma levels", "skipped"))
        return
    if STATS_POPULATION != "all":
        print("  check 4 (chroma levels vs the v4 run): the v4 run's "
              "targets came from ALL colour\n      images, these from "
              f"{STATS_POPULATION}; the parameters are not expected to "
              "match. Reported, not judged.")
    ref = pd.read_csv(REF_LEVELS_CSV)
    ref = ref[(ref.family == "chroma") & (ref.direction == "down")
              & (ref.status == "ok")]
    comp, worst = [], 0.0
    for r in ref.itertuples():
        m = levels[(levels.dimension == "colour") & (levels.level == r.level)]
        if not len(m) or not np.isfinite(m.param.iloc[0]):
            continue
        ours = float(m.param.iloc[0])
        rel = abs(ours - r.param) / max(abs(r.param), 1e-9)
        worst = max(worst, rel)
        comp.append(f"{r.level}: {ours:.4f} vs {r.param:.4f}")
    # With a different target population the parameters SHOULD differ,
    # so the comparison is reported rather than judged.
    judged = STATS_POPULATION == "all"
    ok = bool(comp) and worst <= LEVEL_CHECK_RTOL
    verdict = (("PASS" if ok else "WARN") if judged else "reported")
    print("  check 4 (chroma levels vs the v4 run): " + "; ".join(comp)
          + f"  -> worst {worst:.1%} {verdict}")
    report.append(("check 4 chroma levels", f"{verdict} ({worst:.1%})"))
    if judged and not ok and comp:
        warnings.warn("The chroma levels differ from the v4 run's by "
                      "more than the tolerance; the samples differ, but "
                      "check the manipulation copy.")


# ============================================================
# Figures
# ============================================================

def plot_calibration(grid, levels, stats):
    st = stats.set_index("feature")
    fig, axes = plt.subplots(1, len(DIMENSIONS),
                             figsize=(7 * len(DIMENSIONS), 5.5))
    for ax, (dim, (fam, feat, cross, _)) in zip(np.atleast_1d(axes),
                                                DIMENSIONS.items()):
        g = grid[grid.dimension == dim].sort_values("param")
        sd = float(st.at[feat, "sd"])
        ax.plot(g.param, g.median_change / sd, color="C0", marker="o",
                ms=3, label=f"{feat} (calibrated)")
        ax.fill_between(g.param, g.q25_change / sd, g.q75_change / sd,
                        color="C0", alpha=0.2, label="interquartile range")
        sd_c = float(st.at[cross, "sd"])
        ax.plot(g.param, g.median_cross_change / sd_c, color="C1",
                marker="s", ms=3, label=f"{cross} (cross-talk)")
        for r in levels[levels.dimension == dim].itertuples():
            if not np.isfinite(r.param):
                continue
            ax.axvline(r.param, color="k", lw=0.6, ls=":")
            ax.annotate(r.level, (r.param, r.target_sd), fontsize=8,
                        rotation=90, va="bottom", ha="right")
            ax.plot([r.param], [r.target_sd], "k*", ms=8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(f"{fam} parameter")
        ax.set_ylabel("median change (corpus SD)")
        ax.set_title(f"{dim}: {fam} calibrated on {feat}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "calibration.png"), dpi=150)
    plt.close(fig)


def plot_scatter(per_image, levels, stats):
    if not len(per_image):
        return
    st = stats.set_index("feature")
    dims = list(DIMENSIONS)
    fig, axes = plt.subplots(1, len(dims), figsize=(7 * len(dims), 5))
    for ax, dim in zip(np.atleast_1d(axes), dims):
        feat = DIMENSIONS[dim][1]
        sd = float(st.at[feat, "sd"])
        d = per_image[per_image.dimension == dim]
        order = [f"p{q:02d}" for q in TARGET_PERCENTILES
                 if f"p{q:02d}" in set(d.level)]
        data = [d[d.level == lv].change.to_numpy() / sd for lv in order]
        # labels are set afterwards: the boxplot keyword for them was
        # renamed in matplotlib 3.9, this works either side of that
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(np.arange(1, len(order) + 1))
        ax.set_xticklabels(order)
        for i, lv in enumerate(order, 1):
            t = levels[(levels.dimension == dim) & (levels.level == lv)]
            if len(t):
                ax.plot([i], [t.target_sd.iloc[0]], "r*", ms=10)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(f"change in {feat} (corpus SD)")
        ax.set_xlabel("level")
        ax.set_title(f"{dim}: per-image change (red star = target)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "scatter.png"), dpi=150)
    plt.close(fig)


# ============================================================
# The grid and what it affords  (check 7)
# ============================================================

def _task_cell(args):
    """Measure one cell of the stimulus grid on every worker image."""
    ci, cname, k, fi, fname, h, fam = args
    out = []
    for r in W_ROWS:
        v = measure(render_cell(r, k, h, fam), [FEAT_COLOUR, FEAT_FOG])
        out.append((r["image_id"], v[FEAT_COLOUR], v[FEAT_FOG]))
    return ci, cname, k, fi, fname, h, out


def classify_pair(a, b, sat_range, fog_range):
    """v2's classification, from analyse_pairs_loc_con.py.

    Differences come from MEASURED features, never from the parameters,
    because the mapping from parameter to feature is scene-dependent.
    FOG_SIGN keeps "positive means foggier" true for contrast_local,
    which haze lowers.
    """
    d_sat = a["sat"] - b["sat"]
    d_fog = FOG_SIGN * (a["fog"] - b["fog"])
    if a["haze"] == b["haze"]:
        if abs(d_sat) < MIN_DSAT_SAT_ONLY:
            return None
        t = "sat_only"
    elif abs(d_sat) <= PAIR_SAT_TOL and abs(d_fog) >= MIN_DFOG_FOG_ONLY:
        t = "fog_only"
    elif abs(d_fog) >= MIN_DFOG_MIXED and abs(d_sat) >= MIN_DSAT_MIXED:
        t = "conflict" if ((d_fog > 0) == (d_sat > 0)) else "together"
    else:
        return None
    z_sat = abs(d_sat) / sat_range if sat_range > 0 else np.nan
    z_fog = abs(d_fog) / fog_range if fog_range > 0 else np.nan
    return {"type": t, "d_sat": d_sat, "d_fog": d_fog,
            "z_sat": z_sat, "z_fog": z_fog,
            "balance": (z_sat / z_fog) if (z_fog and z_fog > 0) else np.nan}


def check_grid(rows, levels, stats, report, dim="fog", fam="haze",
               ladder=None):
    """CHECK 7: build one fog family's ladder as a grid, measure every
    cell, and count the trials it affords.

    This is the question the levels exist to serve. A ladder can hit
    every target and still fail here -- in particular the colour
    dimension's usable range shrinks as fog grows, because fog has
    already drained the colour before chroma touches it, so a ladder
    that starts its fog high leaves little room for colour trials.

    Run once per fog family, so both candidate stimulus sets are costed
    before either is built.
    """
    print("\n" + "=" * 60)
    print(f"GRID AND PAIR INVENTORY  ({fam})")
    print("=" * 60)
    fog = ladder_params(levels, dim, ladder or EXAMPLE_FOG)
    col = [("k=1", 1.0)] + ladder_params(levels, "colour", EXAMPLE_COLOUR)
    if not fog or len(col) < 2:
        print("  ladders could not be resolved; skipped")
        report.append((f"check 7 grid ({fam})", "skipped"))
        return pd.DataFrame(), pd.DataFrame()

    tasks = [(ci, cn, k, fi, fn, h, fam)
             for ci, (cn, k) in enumerate(col)
             for fi, (fn, h) in enumerate(fog)]
    res = run_tasks(_task_cell, tasks, rows, f"grid cells ({fam})")

    cells = []
    for ci, cn, k, fi, fn, h, vals in res:
        for iid, sat, cfog in vals:
            cells.append({"image_id": iid, "family": fam,
                          "colour_level": cn,
                          "fog_level": fn, "chroma": k, "haze": h,
                          "sat": sat, "fog": cfog})
    cells = pd.DataFrame(cells)

    # --- colour range available within each fog column ---
    st = stats.set_index("feature")
    sd_s = float(st.at[FEAT_COLOUR, "sd"])
    sd_f = float(st.at[FEAT_FOG, "sd"])
    colrange = (cells.groupby(["image_id", "fog_level"]).sat
                .agg(lambda s: s.max() - s.min())
                .groupby("fog_level").median())
    print("\n  sat_mean range available within each fog column "
          "(median over scenes):")
    for fn, _ in fog:
        v = float(colrange.get(fn, np.nan))
        print(f"    {fn}  {v:6.1f}  ({v / sd_s:.2f} SD)")

    # --- pairs, by v2's rules ---
    recs = []
    for iid, g in cells.groupby("image_id"):
        g = g.to_dict("records")
        sr = max(x["sat"] for x in g) - min(x["sat"] for x in g)
        fr = max(x["fog"] for x in g) - min(x["fog"] for x in g)
        for i, j in itertools.permutations(range(len(g)), 2):
            p = classify_pair(g[i], g[j], sr, fr)
            if p:
                p["image_id"] = iid
                recs.append(p)
    pairs = pd.DataFrame(recs)
    if not len(pairs):
        print("\n  the grid affords NO usable pairs at all.")
        report.append((f"check 7 grid ({fam})", "FAIL: no usable pairs"))
        return cells, pairs

    pairs["balanced"] = pairs.balance.between(BALANCE_LO, BALANCE_HI)
    n_scenes = cells.image_id.nunique()
    counts = pairs.type.value_counts()
    per_scene = (pairs.groupby(["image_id", "type"]).size()
                 .unstack().reindex(columns=list(V2_PAIR_COUNTS))
                 .fillna(0).astype(int))

    print(f"\n  {n_scenes} scenes x {len(col)} x {len(fog)} cells "
          f"-> {len(pairs):,} usable pairs")
    print(f"  {'type':<10} {'per scene':>10} {'v2 per scene':>13} "
          f"{'min':>5}")
    ok = True
    for t in V2_PAIR_COUNTS:
        n = int(counts.get(t, 0))
        mine = n / max(n_scenes, 1)
        theirs = V2_PAIR_COUNTS[t] / V2_SCENES
        mn = int(per_scene[t].min()) if t in per_scene else 0
        print(f"  {t:<10} {mine:10.1f} {theirs:13.1f} {mn:5d}")
        if mn < MIN_PAIRS_PER_TYPE_GRID:
            ok = False

    for t, feat in (("fog_only", "d_fog"), ("sat_only", "d_sat")):
        s = pairs.loc[pairs.type == t, feat].abs()
        sd = sd_f if t == "fog_only" else sd_s
        if len(s):
            print(f"  {t}: median {s.median():.1f} ({s.median() / sd:.2f} "
                  f"SD), max {s.max():.1f} ({s.max() / sd:.2f} SD)")
    nb = int(pairs.balanced[pairs.type == "conflict"].sum())
    print(f"  balanced conflict pairs ({BALANCE_LO}-{BALANCE_HI}): {nb:,}")

    print(f"\n  RESULT: {'PASS' if ok else 'CHECK'} "
          f"(every scene affords >= {MIN_PAIRS_PER_TYPE_GRID} of each "
          "type)")
    report.append((f"check 7 grid ({fam})",
                   ("PASS" if ok else "CHECK") + "; "
                   + ", ".join(f"{t} {counts.get(t, 0) / n_scenes:.1f}/scene"
                               for t in V2_PAIR_COUNTS)))
    pairs["family"] = fam
    return cells, pairs


# ============================================================
# Example stimuli
# ============================================================

def render_cell(r, k, h, fog_family="haze"):
    """One cell of the stimulus grid: chroma factor k and haze h applied
    to one scene in MANIP_ORDER.

    `fog_family` selects which haze to use -- "haze" or "haze_eq" --
    so the two can be rendered side by side on matched rungs.

    Composition order is MANIP_ORDER, and IDENTITY_SHORT_CIRCUIT decides
    whether k = 1 and h = 0 pass through the manipulation at all; both
    match the v2 builder by default. The airlight comes from the
    unmodified square, so the veil is the same across a row.
    """
    img = r["square"]
    ctx = ctx_of(r)
    for fam in MANIP_ORDER:
        if fam == "chroma":
            img, _ = apply("chroma", img, k, ctx)
        else:
            img, _ = apply(fog_family, img, h, ctx)
    return img


def ladder_params(levels, dim, names):
    """[(label, parameter)] for the named levels of one dimension."""
    out = []
    for nm in names:
        m = levels[(levels.dimension == dim) & (levels.level == nm)]
        if len(m) and np.isfinite(m.param.iloc[0]):
            out.append((nm, float(m.param.iloc[0])))
    return out


def pick_scenes(rows, feature, n):
    """n scenes at evenly spaced quantiles of their BASELINE value of
    `feature`, weakest first.

    Deliberately not a random draw: the change a level produces is
    proportional to the baseline, so a ladder that works on the median
    scene may do almost nothing to the scene at the bottom. These are
    the scenes to look at.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i]["base"][feature])
    if n >= len(order):
        return order
    q = np.linspace(0, len(order) - 1, n).round().astype(int)
    return [order[i] for i in q]


def _sheet(cells, row_labels, col_labels, title, path, thumb=256):
    """A labelled contact sheet; cells[r][c] is a uint8 image."""
    nr, nc = len(cells), len(cells[0])
    fig, axes = plt.subplots(nr, nc, figsize=(1.7 * nc + 1.2, 1.7 * nr + 0.9),
                             squeeze=False)
    for i in range(nr):
        for j in range(nc):
            ax = axes[i][j]
            im = cells[i][j]
            if im.shape[0] != thumb:
                im = np.asarray(Image.fromarray(im).resize(
                    (thumb, thumb), Image.BILINEAR), np.uint8)
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_examples(rows, levels, report):
    """Render the chosen ladders on real scenes.

    Three things are written:
      ladder_fog.png      the fog ladder at full colour, one row per
                          scene, scenes ordered by baseline contrast
      ladder_colour.png   the colour ladder with no haze, scenes
                          ordered by baseline saturation
      grid_<id>.png       the full colour x fog grid for a few scenes,
                          which is the space the experiment draws pairs
                          from, plus every cell at full size
    """
    if not SAVE_EXAMPLES:
        return
    print("\n" + "=" * 60)
    print("EXAMPLE STIMULI")
    print("=" * 60)
    out = os.path.join(OUTPUT_DIR, "examples")
    os.makedirs(out, exist_ok=True)

    fog = ladder_params(levels, "fog", EXAMPLE_FOG)
    col = [("k=1", 1.0)] + ladder_params(levels, "colour", EXAMPLE_COLOUR)
    if not fog or len(col) < 2:
        warnings.warn("example ladders could not be resolved; skipped.")
        return

    def label(i):
        r = rows[i]
        return (f"{r['category']} #{r['image_id']}\n"
                f"sat {r['base'][FEAT_COLOUR]:.0f} "
                f"con {r['base'][FEAT_FOG]:.0f}")

    # --- fog ladder, at full colour ---
    pick = pick_scenes(rows, FEAT_FOG, EXAMPLE_N)
    cells = [[render_cell(rows[i], 1.0, h) for _, h in fog] for i in pick]
    _sheet(cells, [label(i) for i in pick],
           [f"{nm}\nh={h:.3f}" for nm, h in fog],
           "fog ladder (chroma k = 1); scenes ordered by baseline "
           "contrast_local, weakest first",
           os.path.join(out, "ladder_fog.png"))

    # --- colour ladder, with no haze ---
    pick = pick_scenes(rows, FEAT_COLOUR, EXAMPLE_N)
    cells = [[render_cell(rows[i], k, 0.0) for _, k in col] for i in pick]
    _sheet(cells, [label(i) for i in pick],
           [f"{nm}\nk={k:.3f}" for nm, k in col],
           "colour ladder (haze h = 0); scenes ordered by baseline "
           "sat_mean, weakest first",
           os.path.join(out, "ladder_colour.png"))

    # --- the haze_eq ladder, and the comparison that decides between
    #     the two manipulations ---
    fog_eq = (ladder_params(levels, "fog_eq", EXAMPLE_FOG_EQ)
              if "fog_eq" in DIMENSIONS else [])
    if fog_eq:
        pick = pick_scenes(rows, FEAT_FOG, EXAMPLE_N)
        cells = [[render_cell(rows[i], 1.0, h, "haze_eq")
                  for _, h in fog_eq] for i in pick]
        _sheet(cells, [label(i) for i in pick],
               [f"{nm}\nh={h:.3f}" for nm, h in fog_eq],
               "haze_eq ladder (chroma k = 1); brightness restored to "
               "the original mean after hazing",
               os.path.join(out, "ladder_fog_eq.png"))

        # Rung for rung, both manipulations hit the SAME contrast_local
        # target, so a column differs only in how brightness was
        # handled. Two rows per scene: plain haze above, haze_eq below.
        pick = pick_scenes(rows, FEAT_FOG, EXAMPLE_COMPARE_SCENES)
        fog_m = ladder_params(levels, "fog", EXAMPLE_FOG_EQ)
        cells, labels = [], []
        for i in pick:
            cells.append([render_cell(rows[i], 1.0, h) for _, h in fog_m])
            labels.append(f"#{rows[i]['image_id']}\nhaze")
            cells.append([render_cell(rows[i], 1.0, h, "haze_eq")
                          for _, h in fog_eq])
            labels.append(f"#{rows[i]['image_id']}\nhaze_eq")
        _sheet(cells, labels, [nm for nm, _ in fog_eq],
               "haze (upper row) vs haze_eq (lower row) at matched "
               "contrast_local targets\n"
               "same column = same contrast reduction, different "
               "brightness handling",
               os.path.join(out, "compare_fog.png"))

    # --- the full grid, for a few scenes, in both manipulations ---
    mid = pick_scenes(rows, FEAT_FOG, EXAMPLE_GRID_SCENES + 2)[1:-1] \
        if EXAMPLE_GRID_SCENES + 2 <= len(rows) else range(
            min(EXAMPLE_GRID_SCENES, len(rows)))
    variants = [("", "haze", fog)]
    if fog_eq:
        variants.append(("eq_", "haze_eq", fog_eq))
    n_full = 0
    for i in list(mid)[:EXAMPLE_GRID_SCENES]:
        r = rows[i]
        for tag, fam, ladder in variants:
            cells = [[render_cell(r, k, h, fam) for _, h in ladder]
                     for _, k in col]
            _sheet(cells, [f"{nm}\nk={k:.3f}" for nm, k in col],
                   [f"{nm}\nh={h:.3f}" for nm, h in ladder],
                   f"{r['category']} #{r['image_id']}  --  colour (rows) "
                   f"x {fam} (columns), order "
                   f"{' then '.join(MANIP_ORDER)}",
                   os.path.join(out,
                                f"grid_{tag}{r['image_id']:06d}.png"))
            if EXAMPLE_FULL_SIZE:
                d = os.path.join(out, f"cells_{tag}{r['image_id']:06d}")
                os.makedirs(d, exist_ok=True)
                for ci, (cn, k) in enumerate(col):
                    for fi, (fn, h) in enumerate(ladder):
                        Image.fromarray(cells[ci][fi]).save(os.path.join(
                            d, f"c{ci}_{cn}_f{fi}_{fn}.png"))
                        n_full += 1
                Image.fromarray(r["square"]).save(
                    os.path.join(d, "original.png"))

    print(f"  {out}: {2 + (2 if fog_eq else 0)} ladder sheets, "
          f"{EXAMPLE_GRID_SCENES * len(variants)} grid sheets"
          + (f", {n_full} full-size cells" if EXAMPLE_FULL_SIZE else ""))
    if fog_eq:
        print("  compare_fog.png is the one to look at: it is the only "
              "thing left\n  between the numbers and the choice of "
              "manipulation.")
    report.append(("example stimuli",
                   f"{len(col)} x {len(fog)} grid, order "
                   f"{'+'.join(MANIP_ORDER)}"
                   + (f"; haze_eq rendered on {EXAMPLE_FOG_EQ}"
                      if fog_eq else "")))


# ============================================================
# Report
# ============================================================

def write_report(levels, grid, stats, stats_tables, report,
                 identity_lines, sample_lines, ratios, side_effects,
                 viol, deepest, elapsed):
    st = stats.set_index("feature")
    L = []
    W = L.append
    W("LEVEL CALIBRATION REPORT  (stimulus pipeline, v4 axes)")
    W("=" * 70)
    W(f"Calibration sample : {N_CALIB_IMAGES} eligible corpus images, "
      f"seed {RANDOM_SEED}")
    W(f"Geometry           : central square, manipulated at "
      f"{IMAGE_SIZE} px, measured at {FEATURE_SIZE} px")
    W(f"Grids              : haze 0 .. {HAZE_MAX} and chroma 1 .. "
      f"{CHROMA_MIN}, {GRID_N} steps ({GRID_N + 1} points)")
    W(f"Clipping cap       : {CLIP_MAX_NEW}")
    W(f"Elapsed            : {elapsed:.0f}s")
    W("")
    for k, v in report:
        W(f"  {k:32s} {v}")

    W("\n1. CORPUS PERCENTILES OF THE TWO FEATURES")
    W("-" * 70)
    W(f"  Targets, SDs and the envelope all come from the "
      f"{'everyday-scene' if STATS_POPULATION == 'everyday' else 'all-colour'}"
      f" population.")
    W("  everyday_colour = colour images in the keep=1 categories of "
      f"{CATEGORY_CSV},")
    W("  which is where the scenes come from; all_colour is the whole "
      "corpus, which")
    W("  includes mountains, oceans and icebergs that can never be "
      "stimuli.")
    cols = ["p01", "p05", "p10", "p15", "p25", "p50", "p75", "p90", "p99"]
    for name, t in stats_tables.items():
        mark = "   <- used" if ((name == "everyday_colour")
                                == (STATS_POPULATION == "everyday")) else ""
        W(f"\n  {name} (n = {int(t.n_images.iloc[0]):,}){mark}")
        sub = t[t.feature.isin([FEAT_COLOUR, FEAT_FOG])]
        W("  " + sub[["feature", "sd"] + cols].round(3)
          .to_string(index=False).replace("\n", "\n  "))

    if len(stats_tables) == 2:
        W("\n  What the choice of population costs, on the two "
          "calibrated features:")
        a = stats_tables["all_colour"].set_index("feature")
        b = stats_tables["everyday_colour"].set_index("feature")
        for f in (FEAT_COLOUR, FEAT_FOG):
            sd = float(b.at[f, "sd"])
            diffs = ", ".join(
                f"{c} {float(b.at[f, c]) - float(a.at[f, c]):+.2f}"
                for c in ["p50", "p25", "p10", "p05", "p01"])
            W(f"      {f:<16} everyday minus all: {diffs}   "
              f"(SD {sd:.2f})")

    W("\n2. CALIBRATED LEVELS")
    W("-" * 70)
    W("  target_change is relative to the corpus median; achieved is the "
      "median over the")
    W("  sample; cross = median change of the OTHER dimension's feature "
      "at the same")
    W("  parameter. share_reaching_target sits near 0.50 by "
      "construction, because the")
    W("  parameter is placed where the median change reaches the target; "
      "a clearly")
    W("  smaller share means the level was capped.")
    show = ["dimension", "level", "target_change", "target_sd", "param",
            "status", "achieved_change", "achieved_sd",
            "share_reaching_target", "cross_change_median",
            "mean_clipped_new"]
    show = [c for c in show if c in levels.columns]
    W("  " + levels[show].round(4).to_string(index=False)
      .replace("\n", "\n  "))

    W("\n3. CANDIDATE LADDERS")
    W("-" * 70)
    W("  Each ladder is a choice of five levels per dimension. The "
      "colour dimension's")
    W("  first level is k = 1 (no chroma change), so it needs four "
      "targets; every cell")
    W("  of the grid carries some haze, so no cell is the untouched "
      "photograph.")
    W("  RANGE is the largest within-scene difference the ladder can "
      "show, in corpus SD;")
    W("  the two dimensions' ranges should be comparable, or one is "
      "the easier to see")
    W("  for a purely physical reason. CROSS is the worst |cross/own| "
      "ratio over the")
    W("  ladder's rungs: above 1 the manipulation moves the OTHER "
      "dimension's feature")
    W("  more than its own.")
    for name, (dim, qs) in CANDIDATE_LADDERS.items():
        parts, ok, sds, cross_ratios = [], True, [], []
        for q in qs:
            r = levels[(levels.dimension == dim)
                       & (levels.percentile == q)]
            if not len(r) or not np.isfinite(r.param.iloc[0]):
                parts.append(f"p{q:02d}=?")
                ok = False
                continue
            r = r.iloc[0]
            parts.append(f"p{q:02d}={r.param:.3f}")
            ok &= (r.status == "ok")
            own = r.get("achieved_sd", np.nan)
            crs = r.get("cross_change_sd", np.nan)
            sds.append(own)
            if np.isfinite(own) and own != 0:
                cross_ratios.append(abs(crs / own))
        # the ladder's own values are negative; the colour ladder also
        # holds k = 1, which is 0 SD, so its range runs from 0
        first = "k=1, " if dim == "colour" else ""
        if sds:
            top, bottom = max(sds), min(sds)
            span = -bottom if dim == "colour" else top - bottom
        else:
            span = float("nan")
        W(f"  {name:<13} {dim:<7} {first}" + ", ".join(parts))
        W(f"  {'':<13} {'':<7} range {span:.2f} SD   worst cross "
          f"{max(cross_ratios) if cross_ratios else float('nan'):.2f}"
          + ("" if ok else "   <-- contains a capped level"))

    W("\n4. CROSS-TALK BETWEEN THE TWO DIMENSIONS")
    W("-" * 70)
    W("  Each level's effect on the OTHER dimension's feature, in that "
      "feature's")
    W("  corpus SD. Haze necessarily desaturates, so the fog levels move "
      "sat_mean;")
    W("  that is a property of the manipulation, not an error, but it is "
      "what the")
    W("  conflict trials are built to separate, so the size of it "
      "matters.")
    for dim, (fam, feat, cross, _) in DIMENSIONS.items():
        d = levels[levels.dimension == dim]
        if "cross_change_sd" not in d.columns:
            continue
        W(f"  {dim} ({fam}) -> {cross}:")
        for r in d.itertuples():
            own = getattr(r, "achieved_sd", float("nan"))
            crs = getattr(r, "cross_change_sd", float("nan"))
            ratio = (abs(crs / own) if np.isfinite(own) and own != 0
                     else float("nan"))
            W(f"      {r.level}  own {own:+.2f} SD   cross {crs:+.2f} SD"
              f"   |cross/own| {ratio:.2f}")

    W("\n5. WHAT THE EXTREME LEVELS LOOK LIKE IN OLD UNITS")
    W("-" * 70)
    W("  The previous stimulus set used haze 0.15 .. 0.60 and chroma "
      "1.0 .. 0.2,")
    W("  chosen by hand. For comparison, in this calibration:")
    for dim, old in (("fog", [0.15, 0.30, 0.45, 0.60]),
                     ("colour", [0.8, 0.6, 0.4, 0.2])):
        g = grid[grid.dimension == dim].sort_values("param")
        feat = DIMENSIONS[dim][1]
        sd = float(st.at[feat, "sd"])
        # sorted by param ascending, so np.interp's requirement on the
        # x values holds for both dimensions; the y values need not be
        # increasing
        vals = np.interp(old, g.param.to_numpy(),
                         g.median_change.to_numpy())
        W(f"  {dim:<7} " + ", ".join(
            f"{p:g} -> {v / sd:+.2f} SD" for p, v in zip(old, vals)))

    if len(side_effects):
        W("\n6. EVERYTHING ELSE THE MANIPULATIONS MOVE")
        W("-" * 70)
        W("  The levels are calibrated on one feature each, but they "
          "change others too.")
        W("  Median change at each level, in that feature's corpus SD; "
          "features that move")
        W("  by less than 0.10 SD at every level are left out. Haze "
          "raises brightness --")
        W("  that confound is accepted by design, and this is its "
          "size.")
        for dim in DIMENSIONS:
            d = side_effects[side_effects.dimension == dim]
            if not len(d):
                continue
            piv = d.pivot(index="feature", columns="level",
                          values="change_sd")
            order = [f"p{q:02d}" for q in TARGET_PERCENTILES
                     if f"p{q:02d}" in piv.columns]
            piv = piv[order]
            piv = piv[piv.abs().max(axis=1) >= 0.10]
            piv = piv.reindex(piv.abs().max(axis=1)
                              .sort_values(ascending=False).index)
            W(f"  {dim} ({DIMENSIONS[dim][0]}):")
            W("  " + piv.round(2).to_string().replace("\n", "\n  "))

    W("\n7. ENVELOPE: WHERE THE LADDER HAS TO STOP")
    W("-" * 70)
    W("  Each level's effect on the features the design does NOT intend "
      "to move, against")
    W("  the corpus 1-99% range. The calibrated features are left out: "
      "taking sat_mean")
    W("  below the corpus p01 is the colour manipulation's purpose, and "
      "46% of the")
    W("  measured v2 stimuli sit outside on that feature for exactly "
      "that reason.")
    W("")
    W("  For reference, the previous stimulus set at its own top haze "
      "level (0.60):")
    W("    lum_p05 median 156 against a corpus p99 of 114 -- 100% of "
      "those images")
    W("    outside; at h = 0.49, 95%. So an excursion here is not fatal, "
      "it is what")
    W("    the piloted set already did. It is reported so the choice is "
      "deliberate.")
    if len(viol):
        W(f"\n  {len(viol)} violations:")
        v = viol[["dimension", "level", "feature", "achieved",
                  "corpus_lo", "corpus_hi", "outside_by_sd"]]
        W("  " + v.round(3).to_string(index=False).replace("\n", "\n  "))
    else:
        W("\n  No level takes an unintended feature outside the corpus "
          "range.")
    if deepest:
        W("\n  Deepest level of each dimension that stays inside:")
        for d, lv in deepest.items():
            tag = "  (report-only)" if d in REPORT_ONLY else ""
            W(f"      {d:<7} {lv}{tag}")
        if "fog" in deepest and "fog_eq" in deepest:
            W("")
            W("  fog vs fog_eq is the comparison this run exists to "
              "make: both are")
            W("  calibrated on contrast_local, so the same rung means "
              "the same contrast")
            W("  reduction. If fog_eq reaches deeper before leaving the "
              "envelope, that is")
            W("  the case for switching; if not, keep plain haze and "
              "stop where fog does.")

    W("\n8. MEASUREMENT PATH AND THE CALIBRATION SAMPLE")
    W("-" * 70)
    W("  A stimulus is measured original -> 512 -> 256; step 04 measured "
      "the corpus")
    W("  original -> 256. Every resize softens fine detail, so the two "
      "paths do not")
    W("  agree on detail measures, and calibrating one against the other "
      "puts the fog")
    W("  ladder off by that ratio. REFERENCE_MODE = "
      f"{REFERENCE_MODE!r}.")
    if ratios is not None:
        W("")
        W("  stimulus path / corpus path, measured on the same "
          "photographs:")
        for f in sorted(ratios.index, key=lambda f: -abs(ratios[f] - 1)):
            if abs(ratios[f] - 1) >= 0.01:
                W(f"      {f:<18} {ratios[f]:.4f}")
        W("      (features agreeing within 1% are left out)")
    if sample_lines:
        W("")
        W("  The calibration sample against the reference it is "
          "calibrated to. Every")
        W("  parameter is set by the sample median and every target by "
          "the reference")
        W("  median, so a gap here moves the whole ladder:")
        for ln in sample_lines:
            W(f"      {ln}")

    W("\n9. IDENTITY POINTS")
    W("-" * 70)
    W("  What the first point of each grid does to the photograph, "
      "measured by applying")
    W("  the manipulation directly rather than through "
      "IDENTITY_SHORT_CIRCUIT, so the")
    W("  number is on record whichever way that setting is left. Both "
      "should be the")
    W("  identity and both are, to within 0.01 of a feature unit.")
    W("")
    W("  Kept because it caught the opposite: an earlier copy of to_u8 "
      "truncated where")
    W("  09c rounds, darkening every manipulated pixel by half a level, "
      "which appeared")
    W("  here as a +0.5 shift in sat_mean at k = 1. Check 1 cannot see "
      "that bug, since")
    W("  it measures unmanipulated images.")
    for ln in identity_lines:
        W(f"      {ln}")

    path = os.path.join(OUTPUT_DIR, "levels_report.txt")
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
    print("CALIBRATE LEVELS  (v4 stimulus pipeline, step 1)")
    print("=" * 60)
    print(f"  Index  : {INDEX_CSV}")
    print(f"  Output : {OUTPUT_DIR}")

    idx, F, stats, stats_tables, eligible = load_corpus(report)

    # The targets must be on the same measurement path as the stimuli.
    ratios = None
    if REFERENCE_MODE == "stimulus":
        ref, ratios = measure_reference(idx, F, eligible, report)
        stats_tables["stimulus_path"] = ref
        stats = ref.drop(columns=["population"]).reset_index(drop=True)
    elif REFERENCE_MODE != "corpus":
        raise ValueError("REFERENCE_MODE must be 'stimulus' or 'corpus'.")
    report.append(("reference mode", REFERENCE_MODE))

    pd.concat(stats_tables.values(), ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "corpus_feature_stats_v4.csv"),
                 index=False, float_format="%.6g")

    print("\n" + "=" * 60)
    print("START-UP CHECKS")
    print("=" * 60)
    check_feature_code(idx, F, rng, report)
    # always the all-colour table: the reference file from the v4 run
    # was computed over all colour images, so that is the like-for-like
    # comparison whatever STATS_POPULATION is set to
    check_stats(stats_tables["all_colour"], report)

    rows = draw_sample(idx, eligible, rng, report)
    grid, _ = run_grid(rows, report)
    grid.to_csv(os.path.join(OUTPUT_DIR, "calibration_grid.csv"),
                index=False, float_format="%.6g")
    check_monotonic(grid, report)
    sample_lines = check_sample(rows, stats, report)
    identity_lines = check_identity(rows, stats, report)

    levels, per_image, side_effects = choose_levels(grid, stats, rows,
                                                    report)
    levels.to_csv(os.path.join(OUTPUT_DIR, "levels.csv"), index=False,
                  float_format="%.6g")
    if len(per_image):
        per_image.to_csv(os.path.join(OUTPUT_DIR, "per_image_levels.csv"),
                         index=False, float_format="%.6g")
    if len(side_effects):
        side_effects.to_csv(os.path.join(OUTPUT_DIR, "level_features.csv"),
                            index=False, float_format="%.6g")
    check_reference_levels(levels, report)
    viol, deepest = check_envelope(side_effects, stats, report)
    if len(viol):
        viol.to_csv(os.path.join(OUTPUT_DIR, "envelope_violations.csv"),
                    index=False, float_format="%.6g")

    if GRID_CHECK:
        all_cells, all_pairs = [], []
        for dim, fam, setting in FOG_VARIANTS:
            ladder = globals()[setting]
            c, pr = check_grid(rows, levels, stats, report, dim, fam,
                               ladder)
            if len(c):
                all_cells.append(c)
            if len(pr):
                all_pairs.append(pr)
        if all_cells:
            pd.concat(all_cells, ignore_index=True).to_csv(
                os.path.join(OUTPUT_DIR, "grid_cells.csv"),
                index=False, float_format="%.6g")
        if all_pairs:
            pd.concat(all_pairs, ignore_index=True).to_csv(
                os.path.join(OUTPUT_DIR, "grid_pairs.csv"),
                index=False, float_format="%.6g")

    plot_calibration(grid, levels, stats)
    plot_scatter(per_image, levels, stats)
    save_examples(rows, levels, report)

    path = write_report(levels, grid, stats, stats_tables, report,
                        identity_lines, sample_lines, ratios, side_effects,
                        viol, deepest, time.time() - t0)
    print(f"\n  Report : {path}")
    print(f"  Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
