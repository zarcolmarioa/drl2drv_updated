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
                (contrast reduction by a haze model: the image is
                 blended toward its own airlight, which lowers local
                 contrast. It models the FOG COMPLAINT -- what people
                 describe as foggy, hazy or blurry -- and is not a
                 claim that the experience is literally fog.)

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
                                all_colour, everyday_colour and
                                stimulus_path)

        WHICH BLOCK TO READ, because it matters and the two differ:
          stimulus_path    measured the way a STIMULUS is measured
                           (original -> IMAGE_SIZE -> FEATURE_SIZE).
                           Its `sd` is the fixed divisor the registered
                           analysis normalises differences by, so the
                           weights do not depend on the grid. For the
                           two stimulus features it reads
                           sat_mean 39.18 and contrast_local 9.35.
          everyday_colour  measured the way the STORED corpus features
                           were. This is the block select_scenes_v4's
                           bands belong to, because they are applied to
                           those stored features. Its contrast_local sd
                           is 10.39 -- about 10% larger, because the
                           extra resize on the stimulus path smooths
                           exactly the fine detail contrast_local
                           measures. Using the wrong one rescales every
                           weight.
  calibration_grid.csv          every parameter tried: median change of
                                the calibrated feature, its quartiles,
                                the cross-talk feature, clipped share
  levels.csv                    one row per candidate level: target in
                                feature units and in corpus SD, the
                                parameter, status, achieved median
                                change, scatter, cross-talk.
                                Now also carries the levels ABOVE the
                                original: dimension `colour_up` (vivid
                                colour, for the pair grid and the grid
                                blocks) and `contrast_up` (more
                                contrast than the original, GRID BLOCKS
                                ONLY -- see GRID_ONLY_DIMENSIONS)
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

import hashlib
import inspect
import itertools
import json
import os
import textwrap
import time
import warnings
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageStat

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

# PREV_LEVELS_CSV
#   The levels.csv of the run the current stimulus set was built from.
#   CHECK 10 compares every level that already existed against it: a
#   single changed parameter means the shared machinery moved and every
#   downstream file is stale. None to skip.
PREV_LEVELS_CSV = "data/levels_v4_ec/levels.csv"

# --- Cache -------------------------------------------------------
#
# USE_CACHE / CACHE_DIR
#   The two slowest stages are recomputed identically on every run: the
#   20,000-image stimulus-path reference (which decodes from disk, so
#   it is the slowest per image), and the per-dimension grids for the
#   dimensions that have not changed -- fog, colour and fog_eq have
#   been recomputed on every run since the ladders were settled, and
#   CHECK 10 then confirms they come out identical.
#
#   Each cached block is keyed on a FINGERPRINT of everything that can
#   change it: the manipulation's own source, the feature code, the
#   geometry, the grid, and the image ids it was measured on. Any
#   mismatch recomputes. That matters more than the speed: reusing a
#   stale block is the same class of error as reading a stale output
#   directory, which cost this pipeline a fog ladder that was wrong by
#   19%. The report lists which blocks were reused and which were
#   recomputed, so a run is never silent about it.
#
#   The cache lives OUTSIDE OUTPUT_DIR on purpose: outputs go to a new
#   folder every run, and a cache that moved with them would never hit.
#   Delete the folder to force a full recompute.
USE_CACHE = True
CACHE_DIR = "data/levels_cache"
REF_LEVELS_CSV  = "data/sweep_v2/levels.csv"

# --- Output ---
OUTPUT_DIR      = "data/levels_v4_ext"

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

# CHROMA_MAX
#   Top of the vivid colour grid. m_chroma above 1 saturates smoothly
#   toward each pixel's in-gamut chroma ceiling rather than clipping,
#   so a grid that overshoots costs resolution, not validity -- which
#   is why this is set wide and the calibration is left to find the
#   parameter.
#
#   2.0 rather than something tighter because the targets are a long
#   way up: the corpus median of sat_mean is 66.7 and p75 is 94.1, so
#   p75 asks for +27.5 saturation units and p90 asks for +59.3.
#
#   The number comes from a test on one photograph, scaled. That image
#   starts at sat_mean 14, far BELOW the corpus median, so its absolute
#   gains understate what a typical scene gets: it gained +4.3 at
#   k = 1.35 and +11.0 at k = 2.0, a factor of 1.78 on its own chroma,
#   which on a median scene at 66.7 would be about +52 -- past p75 and
#   near p90. So 2.0 should be wide enough, and the calibration (150
#   images, aiming at the median) is what settles it. Clipping was 0.00
#   at every k tested, because chroma above 1 approaches each pixel's
#   in-gamut ceiling asymptotically rather than crossing it; the cost
#   of a deep target is a level that is approached and never met, which
#   CHECK 9 reports as `capped`. p90 is the one to expect trouble from,
#   and EXAMPLE_COLOUR_UP is where to take a shallower pair if so.
CHROMA_MAX      = 2.0

# HAZE_MIN
#   Bottom of the contrast-increase grid: the equal-luminance haze run
#   BACKWARDS, away from the airlight. Negative. GRID BLOCKS ONLY --
#   the pair grid never uses these levels, so they need not match any
#   earlier run.
HAZE_MIN        = -0.35

# CONTRAST_UP_FAMILY
#   Which manipulation the contrast-increase dimension uses.
#
#     "contrast_gain_srgb"  a gain about the scene's mean grey on
#                           sRGB VALUES -- the same family as Pillow's
#                           ImageEnhance.Contrast. THE DEFAULT.
#     "contrast_gain"       the same gain in linear light. Kept for the
#                           side-by-side comparison, not as the default:
#                           on a bright scene it crushes a much wider
#                           band of shadow detail (see
#                           m_contrast_gain_srgb for the table).
#     "contrast_up"         the backward equal-luminance haze. The
#                           first attempt. It is the exact inverse of
#                           the fog manipulation, which is attractive,
#                           but it amplifies away from the airlight --
#                           the BRIGHT end -- so it drives the shadows
#                           out first and failed every level.
#
#   MEASURED, 150 images, run of 2026-10-16. Each space was taken to
#   its own cap and the two were then compared at the SAME achieved
#   contrast (+0.217 SD), which is the only comparison that means
#   anything:
#
#                               sRGB      linear
#     pixels clipped >4 levels  0.042     0.025
#     sat_std                  +0.39 SD  +0.74 SD
#     sat_mean (cross-talk)    +0.19 SD  +0.21 SD
#     lum_p05                  -0.33 SD  -0.41 SD
#     cross/own on sat_mean     0.85      0.98
#
#   Linear light clips about 1.7x less for the same contrast; sRGB
#   moves the COLOUR axis about half as much. sRGB wins, because
#   clipping is an artefact and cross-talk is a threat to the design:
#   this pipeline exists to keep the fog axis and the colour axis
#   separable, and a column whose cross/own ratio is 0.98 moves the
#   other axis as much as its own. The Pillow equivalence is worth
#   something in a preregistration as well -- "contrast factor 1.09"
#   is an operation a reader already knows.
#
#   The order of resort, if the gain sheet shows crushed shadows: take
#   the strongest gain the sheet does hold and label the column by that
#   gain, rather than dropping to a percentile that was never the
#   criterion; drop the column only if no gain is both clean and
#   visibly clearer than the original. Deliberately a setting rather
#   than an automatic switch: changing the manipulation changes what
#   the level means, and that belongs in the settings block where it is
#   on record, not in hidden control flow.
CONTRAST_UP_FAMILY = "contrast_gain_srgb"

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

# TARGET_PERCENTILES_UP
#   Targets for the dimensions that move a feature ABOVE the corpus
#   median: the vivid colour levels, and the contrast-increase levels
#   the grid blocks need.
#
#   Kept separate from TARGET_PERCENTILES rather than merged into it,
#   because interp_param assumes the grid STARTS at the identity and
#   moves one way, so that the median change is monotone in the grid
#   index and the first crossing is the answer. A single grid spanning
#   both sides of the identity breaks that: for any target on the upper
#   side the first grid point already satisfies it, and the function
#   returns the end of the grid instead of interpolating. Each
#   direction therefore gets its own dimension and its own grid.
TARGET_PERCENTILES_UP = [55, 60, 65, 70, 75, 90]

# STATS_PERCENTILES
#   The percentiles written for every feature in every population
#   block of corpus_feature_stats_v4.csv. It has to cover every ladder
#   target plus everything another script looks up.
#
#   55, 60, 65 and 70 are here for select_scenes_v4's band sweep, which
#   reads p{low} and p{high} for each row of its BAND_SWEEP. The high
#   side of those pairs was missing, and that script skipped the
#   feature filter silently when a column was absent -- so four of the
#   five rows of its table reported the axis-only count twice. The
#   selected scenes were unaffected (the run's own feature bands, 30
#   and 75, both existed), but the table was wrong.
STATS_PERCENTILES = sorted(set(TARGET_PERCENTILES)
                           | set(TARGET_PERCENTILES_UP)
                           | {1, 50, 75, 90, 99})

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

# CLIP_MAX_NEW_UP
#   The clipping cap for the dimensions that move a feature ABOVE the
#   original. Separate from CLIP_MAX_NEW because the two directions are
#   not comparable: blending toward the airlight moves pixels INWARD
#   and almost never clips, so 0.01 is easy there, whereas amplifying
#   away from it pushes them out past 0 and 255 and clips at once. A
#   test on one bright, detailed photograph clipped 1.3% of pixels at
#   the shallowest useful backward step and 2.8% at a p75 contrast
#   increase -- both already past 0.01.
#
#   The first run set this equal to CLIP_MAX_NEW (0.01) and the
#   backward blend failed every level: it clipped 2.2% of pixels at the
#   shallowest grid step, for a gain of 0.08 SD. Raised here rather
#   than in CLIP_MAX_NEW, because the downward levels are calibrated
#   and must not move (CHECK 10).
#
#   0.03 was the second guess, and the run of 2026-10-16 showed what it
#   buys: a gain of 1.058 for +0.150 SD, which is 63% of p60 and 41%
#   of p65. That column is barely distinguishable from the original on
#   the contact sheet, and a column nobody can tell from the original
#   yields choices that are noise -- which is the one thing this column
#   must not be. So the cap is not the criterion; being visibly clearer
#   than the original is. 0.06 is set wide enough to let p60 through
#   (p60 needs about 5% at this depth, p65 about 8%), and the gain is
#   then chosen from GAIN_SHEET_GAINS on pictures of the actual
#   stimulus scenes. Whatever gain is chosen goes in the documentation
#   with the sheet that justified it.
CLIP_MAX_NEW_UP = 0.06

# CLIP_DEPTH_UP
#   How far past 0 or 255 a channel must be pushed before it counts as
#   clipped, for the UP dimensions only. See apply() for why the
#   half-level rule does not fit a contrast increase: it counts a
#   channel at 1 going to 0.
#
#   4 levels is about 1.6% of the range -- a loss of real shadow or
#   highlight detail rather than a rounding of near-black to black.
#   CHECK 9 reports the share at several depths so the sensitivity to
#   this choice is visible rather than assumed.
CLIP_DEPTH_UP = 4.0

# CLIP_REPORT_DEPTHS
#   The depths CHECK 9 reports, so the effect of CLIP_DEPTH_UP can be
#   read off instead of argued about.
CLIP_REPORT_DEPTHS = [0.5, 2.0, 4.0, 8.0]

# COMPARE_CONTRAST_SPACES
#   Calibrate the contrast increase in BOTH spaces and report them side
#   by side: contrast_up on sRGB values (the default) and
#   contrast_up_lin in linear light. One extra dimension, about a
#   minute of run time, and it settles on the real sample a question
#   that cannot be settled by argument -- the two spaces cross over
#   near a scene mean of 60 and these scenes span 28 to 195. Set False
#   once the space is chosen.
COMPARE_CONTRAST_SPACES = True

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

# EXAMPLE_COLOUR_UP
#   The vivid colour levels. p75 MIRRORS THE FIRST STEP DOWN (p25):
#   both are 25 percentile points from the median. p65 is an
#   intermediate step, not a mirror of anything.
#
#   The mirror of the second step down (p10) would be p90, and p90 is
#   NOT REACHABLE: the first run put it at the gamut limit, where the
#   median scene gained +34.8 of the +60.6 required and only 2 of 150
#   scenes reached the target. Widening CHROMA_MAX would not rescue it
#   -- chroma above 1 approaches each pixel's in-gamut ceiling
#   asymptotically, and the grid is already flattening. So the upward
#   ladder is shorter than the downward one, and asymmetric by
#   necessity rather than by choice.
EXAMPLE_COLOUR_UP = ["p65", "p75"]

# NOTE FOR STEP 7, recorded here so it is not lost: the fog columns
# move colour as well -- haze drains saturation going down, and the
# contrast gain raises it going up -- so a fog screen is not a pure
# contrast screen in either direction. Now that chroma above 1 works,
# each fog column could have its mean saturation pulled back to the
# original's with a per-column chroma factor, making the fog screen
# colour-neutral. Not implemented; the cross-talk is reported for both
# directions in section 4 instead.

# GRID_CONTRAST_LEVELS
#   Contrast ABOVE the original, for the grid blocks only.
#
#   ONE column. The block is descriptive: one column is enough to show
#   whether people place "normal" above the original, and anyone beyond
#   it is reported as a count at the edge rather than resolved.
#
#   p65 was the first target, because it mirrors the first step down
#   the fog ladder (p50 to p35). It is not reachable: at the depth-4
#   clipping cap the gain method stops well short of it, and raising
#   the cap far enough costs about 8% of pixels pushed more than four
#   levels past the range. p60 is the target now, and the final column
#   is NOT named by this percentile in the outputs -- it is labelled by
#   the gain and the SD it achieves, because a label that names a
#   percentile the column does not reach is the kind of thing that ends
#   up in a paper unchallenged.
GRID_CONTRAST_LEVELS = ["p60"]

# EXAMPLE_N
#   Scenes per contact sheet, drawn at evenly spaced quantiles of the
#   BASELINE of the relevant feature -- not at random. The change a
#   level produces is proportional to the scene's baseline, so this
#   deliberately shows the weakest scene as well as the strongest.
EXAMPLE_N       = 6

# CONTRAST_SHEET_SCENES / CONTRAST_SHEET_DARK_SCENES
#   The contrast contact sheet: how many scenes spread across baseline
#   contrast, and how many of the DARKEST scenes to add on top.
#
#   The dark ones were added on the expectation that the gain would
#   crush shadows. It does not -- see GAIN_SHEET_WORST. They are kept
#   anyway, because this sheet is about the SEAM at the original and a
#   seam should be checked at both ends of the brightness range. The
#   bright-end cost is gain_sweep.png's job.
CONTRAST_SHEET_SCENES      = 5
CONTRAST_SHEET_DARK_SCENES = 3

# SCENES_CSV
#   The scenes the stimulus set is actually built on. The contrast-up
#   column is only ever rendered for THESE photographs, and they were
#   selected inside the middle bands of contrast and colour, so the
#   calibration sample -- drawn at random from the whole eligible pool
#   -- contains darker and more extreme images than any stimulus and
#   overstates what the gain costs.
#
#   Either shape is accepted: the stimulus manifest (one row per
#   rendered image, columns scene_id / category / source_path) or a
#   screening worksheet (one row per candidate, columns scene_id /
#   category / path / keep, filtered to keep = 1). Rows are collapsed
#   to one per scene. If the file is not found the gain sheet is
#   skipped with a message and the rest of the run is unaffected --
#   this diagnostic decides one column, it is not a dependency of the
#   ladders.
SCENES_CSV = "data/stimulus_set_v4_ec_haze_eq_2/stimulus_manifest.csv"

# GAIN_SHEET_GAINS
#   The gain sweep that decides the contrast-up column.
#
#   1.058 is where the old 0.03 cap stopped, and it is on the sheet as
#   the baseline the others have to beat. 1.093 is p60. 1.110 is one
#   step past it, so the sheet shows whether p60 sits near a cliff or
#   in the middle of a safe range -- a value that only just holds is a
#   worse choice than a slightly smaller one that clearly holds.
#   1.0875 is the gain the column SHIPS at -- the strongest the cap
#   allows -- and it is on the sheet because a column nobody has looked
#   at is not a column anyone has checked. It was missing from the
#   first sweep, which bracketed it at 1.075 and 1.093 without ever
#   rendering it.
GAIN_SHEET_GAINS = [1.058, 1.075, 1.0875, 1.093, 1.110]

# GAIN_SHEET_WORST / GAIN_SHEET_DARK / GAIN_SHEET_PICK_GAIN
#   Which scenes go on the sheet. EVERY stimulus scene is measured --
#   188 renders, seconds -- and the sheet shows the WORST ones.
#
#   The first version showed the five darkest scenes, on the reasoning
#   that a gain about the mean pulls shadows toward 0. The measurement
#   refuted it: on the real stimulus scenes nothing at all went below
#   0, at any gain, at any depth, and all the clipping was at the
#   BRIGHT end. Darkness does not predict it -- the two darkest scenes
#   (night streets, mean grey 27 and 53) were the safest at 0.4%,
#   while a dining hall with a wall of windows lost 5.1%. What binds is
#   a large area already close to white, which a dark scene can lack
#   entirely.
#
#   So the ranking is now the thing being measured rather than a proxy
#   for it, and the five worst are chosen at GAIN_SHEET_PICK_GAIN --
#   the gain actually being considered, since the ordering of scenes
#   can change with gain. One dark scene is kept for contrast: the
#   shadow question is answered and the sheet should keep showing that
#   it stays answered.
GAIN_SHEET_WORST     = 5
GAIN_SHEET_DARK      = 1
GAIN_SHEET_PICK_GAIN = 1.0875

# GAIN_SHEET_CELL_IN / GAIN_SHEET_DPI
#   The gain sheet is rendered at FULL resolution: one image pixel per
#   output pixel, so the bright regions can be zoomed into. At 130 dpi
#   and 1.7 in per cell the first version drew each 512 px square into
#   221 px, which resampled away the very gradients the sheet exists to
#   judge -- every cell looked like the original. Every other contact
#   sheet in this script is a thumbnail on purpose, because it is read
#   as a row; this one is read by zooming.
GAIN_SHEET_CELL_IN = IMAGE_SIZE / 200.0
GAIN_SHEET_DPI     = 200

# VISIBLE_DEEP_SHARE / FLATTEN_MARGIN
#   Where a bright-end excursion stops being specular sparkle and
#   starts being a flat patch.
#
#   The statistic is the share of pixels pushed more than 8 levels past
#   255 -- the deep share, which cannot be rounding. 2% of a 512 px
#   square is about a 72 x 72 block: big enough to read as a flat area
#   rather than a highlight. Photographs have blown highlights already,
#   so a small share is normal and is not what this is looking for.
#
#   FLATTEN_MARGIN is the second, stricter reading, applied at BOTH
#   ends: `flattened` is the share of pixels at least this many levels
#   below white that render to exactly 255, and `crushed` the share at
#   least this many levels above black that render to exactly 0. Those
#   are detail destroyed, rather than near-white pixels merging into
#   white and near-black into black, and they separate a window that
#   was already blown from a sky gradient being flattened -- two cases
#   the raw share scores identically.
#
#   `crushed` is expected to be near zero: a pixel can only go negative
#   if it starts below m(g-1)/g, which is 2 to 14 across these scenes,
#   so almost nothing 15 levels up can reach 0. That is an argument,
#   and arguments about the dark end have been wrong twice here, so it
#   is measured.
VISIBLE_DEEP_SHARE = 0.02
FLATTEN_MARGIN     = 15

# CONTRAST_UP_GAIN_EXPECTED
#   The gain the contrast-up column is expected to land on, recorded so
#   that a run which silently lands somewhere else says so.
#
#   It is NOT hard-coded into the manipulation: the solver still picks
#   the strongest grid point inside CLIP_MAX_NEW_UP, and this is what
#   that came out as on the run of 2026-10-16 -- grid point 6 of 25,
#   for +0.2233 SD. That is 94% of the p60 step (+0.2372 SD) and the
#   column is labelled by its gain and its SD everywhere, never as p60,
#   because it does not reach p60. If the sample, the cap or the grid
#   changes, the solver will move and the check will report it rather
#   than letting a stale number into the documentation.
CONTRAST_UP_GAIN_EXPECTED = 1.0875

# PIL_CHECK_SCENES
#   CHECK 11 compares the contrast-up manipulation against Pillow's
#   ImageEnhance.Contrast on this many stimulus scenes. The point is
#   the preregistration: "a contrast gain about the scene mean, as in
#   Pillow's ImageEnhance.Contrast at factor 1.0875" is a claim a
#   reader can check in two lines, and it should be true.
PIL_CHECK_SCENES = 5

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

# COLOUR_RELATED_FEATURES
#   Measured and reported for the colour dimensions, never counted as
#   envelope violations. They track the same construct as sat_mean, so
#   leaving the corpus range is the manipulation working -- but chroma
#   above 1 pushes them UP, which nothing else in this pipeline does,
#   and that is worth seeing rather than assuming.
COLOUR_RELATED_FEATURES = ["colorfulness", "chroma_mean", "sat_std"]

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
N_WORKERS       = 24
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


def srgb_to_lin_f(x):
    """sRGB on the 0-255 scale -> linear light, for UNCLIPPED input.

    srgb_to_lin takes uint8, so it cannot be used on a blend that has
    pushed pixels past 0 or 255. Below the sRGB knee the transfer is
    linear, and that branch extends to negative values unchanged; above
    1 the power branch extends smoothly. Both extensions are the
    standard ones, and they matter only for pixels the gain is about to
    pull back inside.
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

    The branch test is on the SIGNED value, exactly as in
    srgb_to_lin_f, which makes the two exact inverses over the whole
    line: srgb_to_lin_f sends every negative through its linear branch
    (y <= 0.04045 is true for all y < 0), so the inverse of that branch
    -- 12.92 * x -- has to cover every negative too.

    The first version of this fix branched on |x| instead, the odd
    extension. That is a legitimate convention but it is NOT the
    inverse of the function above: it sent anything past about -10.3
    levels through the power branch, so a pixel the gain drove to -41.9
    was reported as -29.6. The four depths this script reports (0.5, 2,
    4 and 8 levels) all sit inside the region where the two agreed, so
    the counts published from it were right -- but a deeper threshold
    would have been quietly wrong, which is the same shape of fault as
    the clamp it replaced.

    This used to begin `x = np.maximum(x, 0.0)`, and that one line made
    the dark end of the contrast-gain families IMPOSSIBLE to measure.
    Every manipulation that ends here returned a non-negative array, so
    apply() and _ends() found nothing below 0 -- not because the gain
    spares the shadows, but because the clamp happened one step before
    anything looked. A diagnostic that cannot return a non-zero value
    is not evidence.

    The RENDERED IMAGE IS UNCHANGED: to_u8 clips to [0, 255], so a
    pixel that was clamped to 0 here is clipped to 0 there. Only what
    the diagnostics can see changes, and the fog_eq ladder recomputing
    bit-for-bit under CHECK 10 is the proof of that.

    DIVERGENCE, to be closed at step 9: the verbatim copies of this
    function in 09c_manipulation_sweep, build_stimulus_set_v4 and
    select_scenes_v4 still carry the clamp. They produce identical
    images, so no stimulus and no calibrated level depends on which
    version is used -- but the copies are no longer byte-identical and
    the same edit belongs in all three.
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


def cmax_of_img(img):
    """Per-pixel in-gamut chroma ceiling of an image, as m_chroma needs
    it for k > 1: read the lightness and hue off THIS image and ask
    cmax_of for the largest chroma that still lands inside sRGB."""
    lab = lab_of(img)
    return cmax_of(lab[:, :, 0],
                   np.arctan2(lab[:, :, 2], lab[:, :, 1]))


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


def m_contrast_gain(img, g, ctx):
    """The fallback contrast increase: a gain about the scene's own mean
    grey, in linear light, with the mean restored afterwards.

    Used instead of m_contrast_up when the backward blend clips beyond
    CLIP_MAX_NEW. It amplifies about the MEAN rather than about the
    airlight, so it loads the two tails evenly instead of driving the
    shadows first, and it clips later for the same change in
    contrast_local. The cost is that it is no longer the inverse of the
    haze model, so the grid-block columns would no longer sit on the
    same path as the pair grid's -- which is why it is the fallback and
    not the default.
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
    detail. The two cross over near a scene mean of 60. The stimulus
    scenes span roughly 28 to 195, so neither space wins everywhere,
    and sRGB is harsher on HIGHLIGHTS in exchange (on one bright test
    photograph with a blown sky it clipped more overall at matched
    contrast). Both are therefore calibrated and compared on the real
    sample -- see COMPARE_CONTRAST_SPACES.

    The mean is restored afterwards as the other families do, which is
    nearly a no-op here: a gain about the mean preserves it except for
    what clipping takes.
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


FUNCS = {"chroma": m_chroma, "haze": m_haze, "haze_eq": m_haze_eq,
         "contrast_up": m_contrast_up, "contrast_gain": m_contrast_gain,
         "contrast_gain_srgb": m_contrast_gain_srgb}


IDENTITY_PARAM = {"chroma": 1.0, "haze": 0.0, "haze_eq": 0.0,
                  "contrast_up": 0.0, "contrast_gain": 1.0,
                  "contrast_gain_srgb": 1.0}


def apply(fam, img, p, ctx, depth=0.5):
    """(uint8 image, share of NEWLY clipped pixels): pixels with a
    channel that was inside the range and was pushed outside it by more
    than `depth` levels.

    `depth` exists because the default half-level rule does not fit the
    UP direction. It was written for manipulations that should never
    clip at all -- blending toward the airlight moves pixels inward --
    so any crossing at all is a fault worth catching. A contrast
    INCREASE must push the extreme pixels out; that is what increasing
    contrast is. At the gains in question the rule counts a channel at
    value 1 going to 0, which no one can see.

    So the up dimensions measure how FAR a pixel is pushed, not whether
    it crossed: CLIP_DEPTH_UP levels past the range, which is a loss of
    real shadow or highlight detail rather than a rounding of near-black
    to black. The down dimensions keep the half-level rule unchanged,
    so their calibrated levels cannot move (CHECK 10).
    """
    if fam is None:
        return img, 0.0
    if IDENTITY_SHORT_CIRCUIT and p == IDENTITY_PARAM.get(fam):
        return img, 0.0
    x = FUNCS[fam](img, p, ctx)
    inside = (img > 0) & (img < 255)
    out = (((x < -depth) | (x > 255.0 + depth)) & inside).any(axis=2).mean()
    return to_u8(x), float(out)


# ============================================================
# Cache fingerprints
# ============================================================

def _src_hash(*fns):
    """A hash of the SOURCE of the given functions, so a cached block is
    invalidated by any edit to the code that produced it."""
    h = hashlib.sha1()
    for f in fns:
        h.update(inspect.getsource(f).encode("utf-8"))
    return h.hexdigest()[:16]


# What each manipulation family actually depends on. Listed per family
# rather than hashing the whole module, so that adding a manipulation
# does not invalidate the grids of the others -- which is most of what
# a run changes while a pipeline is being tuned.
def _manip_deps(fam):
    common = (to_u8, _f, measure)
    table = {
        "haze": (m_haze,),
        "haze_eq": (m_haze_eq, m_haze, srgb_to_lin, lin_to_srgb_f,
                    mean_grey),
        "chroma": (m_chroma, lab_of, rgb_of_f, _lab_to_lin, cmax_of,
                   cmax_of_img, ctx_of),
        "contrast_up": (m_contrast_up, srgb_to_lin_f, lin_to_srgb_f,
                        mean_grey),
        "contrast_gain": (m_contrast_gain, srgb_to_lin_f,
                          lin_to_srgb_f, mean_grey),
        "contrast_gain_srgb": (m_contrast_gain_srgb, srgb_to_lin_f,
                               lin_to_srgb_f, mean_grey),
    }
    return table[fam] + common


def grid_fingerprint(dim, rows):
    """Everything that can change one dimension's grid result."""
    fam, feat, cross, grid = DIMENSIONS[dim]
    payload = json.dumps({
        "dimension": dim, "family": fam,
        "feature": feat, "cross_feature": cross,
        "grid": [round(float(p), 12) for p in grid],
        "image_size": IMAGE_SIZE, "feature_size": FEATURE_SIZE,
        "clip_px": CLIP_PX,
        "airlight_top": HAZE_AIRLIGHT_TOP,
        "gamut": [GAMUT_ITERS, GAMUT_CMAX_SEARCH, GAMUT_MARGIN],
        "identity_short_circuit": IDENTITY_SHORT_CIRCUIT,
        "clip_depth": clip_depth(dim),
        "image_ids": [int(r["image_id"]) for r in rows],
        "code": _src_hash(*_manip_deps(fam), apply, _one_param,
                          central_square, clip_box),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def reference_fingerprint(ids):
    """Everything that can change the stimulus-path reference."""
    payload = json.dumps({
        "seed": REFERENCE_SEED, "n": int(N_REFERENCE_IMAGES),
        "image_size": IMAGE_SIZE, "feature_size": FEATURE_SIZE,
        "clip_px": CLIP_PX,
        "percentiles": list(STATS_PERCENTILES),
        "features": list(CROP),
        "ids": hashlib.sha1(np.asarray(ids, np.int64).tobytes())
                      .hexdigest(),
        "code": _src_hash(measure, central_square, clip_box,
                          legacy_features, all_features),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def cache_path(kind, name, key):
    return os.path.join(CACHE_DIR, f"{kind}_{name}_{key}.npz")


def cache_get(kind, name, key):
    if not USE_CACHE:
        return None
    p = cache_path(kind, name, key)
    if not os.path.isfile(p):
        return None
    try:
        return dict(np.load(p, allow_pickle=False))
    except Exception as exc:
        warnings.warn(f"cache {p} unreadable ({exc}); recomputing.")
        return None


def cache_put(kind, name, key, **arrays):
    if not USE_CACHE:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = cache_path(kind, name, key)
    tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, p)


def ctx_of(r, need_cmax=False):
    """The per-image context the manipulations need: the airlight for
    haze, the original mean grey for haze_eq to restore, and -- only
    when asked -- the per-pixel in-gamut chroma ceiling that m_chroma
    needs for k > 1.

    The ceiling is off by default because computing it is a 16-step
    bisection over every pixel, and nothing below k = 1 wants it. It is
    cached on the row, so a worker pays for it once.

    The cached value is read off the UNMODIFIED square. That is correct
    wherever chroma is the only manipulation, which covers the grid and
    the level measurements. It is NOT correct inside render_cell, where
    MANIP_ORDER applies haze first: haze moves both lightness and hue,
    so it moves the ceiling. render_cell recomputes it on the hazed
    image. 09c took the ceiling from the original square unconditionally,
    which is only right when chroma runs first or k <= 1.
    """
    ctx = {"airlight": r["airlight"], "lum_mean_512": r["lum_mean_512"]}
    if need_cmax:
        if "cmax" not in r:
            r["cmax"] = cmax_of_img(r["square"])
        ctx["cmax"] = r["cmax"]
    return ctx


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
# --- Dimensions that move a feature ABOVE the original ---
#
# Each gets its own grid, starting at its identity and moving one way,
# for the reason given at TARGET_PERCENTILES_UP. `colour_up` feeds the
# pair grid as well as the grid blocks; `contrast_up` is grid-block
# only, and CONTRAST_UP_FAMILY (in SETTINGS) chooses which
# manipulation it uses when the backward blend clips too much.
DIMENSIONS["colour_up"] = ("chroma", FEAT_COLOUR, FEAT_FOG,
                           lin(1.0, CHROMA_MAX, GRID_N))
DIMENSIONS["contrast_up"] = (
    CONTRAST_UP_FAMILY, FEAT_FOG, FEAT_COLOUR,
    lin(0.0, HAZE_MIN, GRID_N) if CONTRAST_UP_FAMILY == "contrast_up"
    else lin(1.0, 1.0 - HAZE_MIN, GRID_N))

# The same contrast increase in the other space, for comparison only.
# Never feeds a grid block; it exists so one run can say which space
# costs less on the real sample.
if COMPARE_CONTRAST_SPACES:
    DIMENSIONS["contrast_up_lin"] = (
        "contrast_gain", FEAT_FOG, FEAT_COLOUR,
        lin(1.0, 1.0 - HAZE_MIN, GRID_N))

# UP_DIMENSIONS
#   Which dimensions take TARGET_PERCENTILES_UP instead of
#   TARGET_PERCENTILES, their own clipping cap, and the depth-aware
#   clipping rule. Read by choose_levels, clip_cap and clip_depth.
UP_DIMENSIONS = {"colour_up", "contrast_up"} | (
    {"contrast_up_lin"} if COMPARE_CONTRAST_SPACES else set())

# GRID_ONLY_DIMENSIONS
#   Dimensions whose levels exist for the grid blocks and never enter a
#   pair. Marked so step 3 can tag the rendered images by use.
GRID_ONLY_DIMENSIONS = {"contrast_up"} | (
    {"contrast_up_lin"} if COMPARE_CONTRAST_SPACES else set())

# contrast_up_lin never builds anything; it is measured so the space
# can be chosen from the sample rather than from one photograph.
REPORT_ONLY = ({"contrast_up_lin"} if COMPARE_CONTRAST_SPACES
               else set())

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
    want = STATS_PERCENTILES
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

    # --- cached? the slowest stage in the script, and it depends on
    # nothing that has changed since the ladders were settled ---
    key = reference_fingerprint(ids)
    blob = cache_get("reference", "stimulus_path", key)
    if blob is not None:
        stats = pd.DataFrame({
            "population": "stimulus_path", "feature": CROP,
            "n_images": int(blob["n_images"][0]),
            "mean": blob["mean"], "sd": blob["sd"]})
        for j, q in enumerate(STATS_PERCENTILES):
            stats[f"p{q:02d}"] = blob["pcts"][j]
        ratios = pd.Series(blob["ratios"], index=CROP)
        print(f"  cache HIT: {int(blob['n_images'][0]):,} images, "
              f"key {key}")
        report.append(("reference images",
                       f"{int(blob['n_images'][0]):,} (cached)"))
        order = np.argsort(np.abs(ratios.to_numpy() - 1))[::-1]
        report.append(("worst path ratio",
                       f"{CROP[order[0]]} {ratios.iloc[order[0]]:.4f}"))
        return stats, ratios

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

    stats = pd.DataFrame({"population": "stimulus_path", "feature": CROP,
                          "n_images": len(V), "mean": V.mean(0),
                          "sd": V.std(0, ddof=1)})
    for q in STATS_PERCENTILES:
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
    cache_put("reference", "stimulus_path", key,
              n_images=np.array([len(V)], np.int64),
              mean=V.mean(0), sd=V.std(0, ddof=1),
              pcts=np.asarray([np.percentile(V, q, axis=0)
                               for q in STATS_PERCENTILES], np.float64),
              ratios=np.asarray(ratios, np.float64))
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
    need = (fam == "chroma" and p > 1.0)
    depth = clip_depth(dim)
    for r in W_ROWS:
        ctx = ctx_of(r, need_cmax=need)
        out, c = apply(fam, r["square"], p, ctx, depth)
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
    need = (fam == "chroma" and p > 1.0)
    F = []
    for r in W_ROWS:
        img, _ = apply(fam, r["square"], p, ctx_of(r, need_cmax=need))
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

    # --- which dimensions can be read from the cache? ---
    keys = {dim: grid_fingerprint(dim, rows) for dim in DIMENSIONS}
    results, reused, fresh = [], [], []
    for dim in DIMENSIONS:
        blob = cache_get("grid", dim, keys[dim])
        if blob is None:
            fresh.append(dim)
            continue
        for gi in range(blob["param"].shape[0]):
            results.append((dim, int(blob["grid_index"][gi]),
                            float(blob["param"][gi]),
                            blob["d_feat"][gi], blob["d_cross"][gi],
                            blob["clip"][gi]))
        reused.append(dim)
    if reused:
        print(f"  cache HIT  : {', '.join(reused)}")
    if fresh:
        print(f"  cache MISS : {', '.join(fresh)}")
    report.append(("grid cache",
                   f"{len(reused)} reused, {len(fresh)} recomputed"))

    tasks = [(dim, gi, float(p))
             for dim in fresh
             for gi, p in enumerate(DIMENSIONS[dim][3])]
    got = run_tasks(_task, tasks, rows, "grid") if tasks else []
    results.extend(got)

    # --- store what was just computed ---
    for dim in fresh:
        rs = sorted((r for r in got if r[0] == dim), key=lambda r: r[1])
        if not rs:
            continue
        cache_put("grid", dim, keys[dim],
                  grid_index=np.array([r[1] for r in rs], np.int64),
                  param=np.array([r[2] for r in rs], np.float64),
                  d_feat=np.asarray([r[3] for r in rs], np.float64),
                  d_cross=np.asarray([r[4] for r in rs], np.float64),
                  clip=np.asarray([r[5] for r in rs], np.float64))

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

def levels_of(dim):
    """A dimension's level names, shallowest first.

    Every place that orders levels has to ask this rather than assume
    TARGET_PERCENTILES. The first run assumed it in two places, and
    both failed silently on the new dimensions: the side-effects table
    in the report came out as an empty frame for colour_up, and the
    "deepest level inside the envelope" line printed p01 for two
    dimensions that have no p01. Nothing crashed; the new ladders were
    simply absent from the report.
    """
    qs = (TARGET_PERCENTILES_UP if dim in UP_DIMENSIONS
          else TARGET_PERCENTILES)
    return [f"p{q:02d}" for q in qs]


def clip_cap(dim):
    """The clipping cap for a dimension: the upward dimensions get their
    own, for the reason given at CLIP_MAX_NEW_UP."""
    return CLIP_MAX_NEW_UP if dim in UP_DIMENSIONS else CLIP_MAX_NEW


def clip_depth(dim):
    """How far past the range a pixel must go to count as clipped. Half
    a level for the downward dimensions, which must not move; several
    levels for the upward ones -- see apply()."""
    return CLIP_DEPTH_UP if dim in UP_DIMENSIONS else 0.5


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
        # Dimensions that move a feature above the median take the
        # upward target list; see TARGET_PERCENTILES_UP.
        for q in (TARGET_PERCENTILES_UP if dim in UP_DIMENSIONS
                  else TARGET_PERCENTILES):
            target = float(st.at[feat, f"p{q:02d}"]) - p50
            param, status = interp_param(g, target, clip_cap(dim))
            rowsout.append({"dimension": dim, "family": fam,
                            "feature": feat, "level": f"p{q:02d}",
                            "percentile": q, "target_change": target,
                            "target_sd": target / sd, "param": param,
                            "status": status,
                            # step 3 tags each rendered image by use;
                            # carried here so it reads the answer
                            # instead of re-deriving it
                            "grid_only": dim in GRID_ONLY_DIMENSIONS})

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


def _task_depth(args):
    """(dimension, level, parameter) -> the mean share of pixels pushed
    past the range, at EVERY depth in CLIP_REPORT_DEPTHS.

    One render per image, then the thresholds are read off the same
    float array. The first version of this check rendered once per
    depth, in the main process rather than the pool -- four times the
    work, none of it parallel, and about 24 minutes added to an
    11-minute run. The depths are thresholds on one result, not four
    different results.
    """
    dim, level, p = args
    fam = DIMENSIONS[dim][0]
    need = (fam == "chroma" and p > 1.0)
    acc = {d: [] for d in CLIP_REPORT_DEPTHS}
    lo_s, hi_s = [], []
    for r in W_ROWS:
        img = r["square"]
        if IDENTITY_SHORT_CIRCUIT and p == IDENTITY_PARAM.get(fam):
            for d in CLIP_REPORT_DEPTHS:
                acc[d].append(0.0)
            lo_s.append(0.0)
            hi_s.append(0.0)
            continue
        x = FUNCS[fam](img, p, ctx_of(r, need_cmax=need))
        inside = (img > 0) & (img < 255)
        for d in CLIP_REPORT_DEPTHS:
            acc[d].append(float((((x < -d) | (x > 255.0 + d)) & inside)
                                .any(axis=2).mean()))
        a, b = _ends(img, x, CLIP_DEPTH_UP)
        lo_s.append(a)
        hi_s.append(b)
    return (dim, level, p,
            {d: float(np.mean(v)) for d, v in acc.items()},
            float(np.mean(lo_s)), float(np.mean(hi_s)))


def check_new_levels(levels, rows, report):
    """CHECK 9: the levels above the original behave.

    Three things have to hold, and each is a way the new ladders could
    fail quietly:

      direction   every `*_up` level must RAISE its own feature. A
                  negative achieved change means the grid was built the
                  wrong way round, or the manipulation does not do what
                  its name says.
      reach       the target has to be reachable. chroma above 1
                  saturates toward each pixel's in-gamut ceiling, so a
                  deep target can be approached and never met; the
                  status column says `capped` when that happens.
      clipping    chroma above 1 and the backward blend are the two
                  places in this pipeline where clipping is a real
                  risk, so the share is reported per level against
                  CLIP_MAX_NEW rather than left to the envelope check.
    """
    print("\n" + "=" * 60)
    print("LEVELS ABOVE THE ORIGINAL  (check 9)")
    print("=" * 60)
    up = levels[levels.dimension.isin(UP_DIMENSIONS)].copy()
    if not len(up):
        print("  no upward dimensions; skipped")
        report.append(("check 9 new levels", "skipped"))
        return True

    cols = ["dimension", "family", "feature", "level", "percentile",
            "target_change", "param", "status", "achieved_change",
            "mean_clipped_new"]
    cols = [c for c in cols if c in up.columns]
    print(up[cols].to_string(index=False))

    bad_dir = up[up.achieved_change <= 0] if "achieved_change" in up else up[:0]
    capped = up[up.status != "ok"] if "status" in up else up[:0]
    over = (up[up.mean_clipped_new > CLIP_MAX_NEW_UP]
            if "mean_clipped_new" in up else up[:0])

    ok = not len(bad_dir)
    print()
    if len(bad_dir):
        print("  FAIL: these levels do not raise their own feature:")
        for _, r in bad_dir.iterrows():
            print(f"    {r.dimension} {r.level}: "
                  f"{r.achieved_change:+.3f}")
    else:
        print("  direction: every upward level raises its own feature")
    if len(capped):
        print(f"  NOTE: {len(capped)} level(s) did not reach the target: "
              + ", ".join(f"{r.dimension} {r.level} ({r.status})"
                          for _, r in capped.iterrows()))
        print("    For colour this is the in-gamut ceiling, not a bug. "
              "Choose a")
        print("    shallower percentile in EXAMPLE_COLOUR_UP if the gap "
              "matters.")
    if len(over):
        print(f"  CLIPPING above CLIP_MAX_NEW_UP "
              f"({CLIP_MAX_NEW_UP:.3f}) at {len(over)} level(s):")
        for _, r in over.iterrows():
            print(f"    {r.dimension} {r.level}: "
                  f"{r.mean_clipped_new:.4f}")
        print("    For contrast_up the order of resort is "
              "CONTRAST_UP_FAMILY, then")
        print("    GRID_CONTRAST_LEVELS; for colour_up, a "
              "shallower EXAMPLE_COLOUR_UP.")
    else:
        print(f"  clipping: every upward level inside "
              f"{CLIP_MAX_NEW_UP:.3f}")

    # --- how much of the clipping is shallow? ---
    #
    # The number above uses CLIP_DEPTH_UP. The same levels measured at
    # other depths say how much of it is near-black rounding and how
    # much is lost detail, so the choice of depth is visible rather
    # than taken on trust.
    tasks = [(dim, lv, p) for dim in sorted(UP_DIMENSIONS)
             for lv, p in ladder_params(levels, dim, levels_of(dim))]
    drows = []
    if tasks:
        for dim, lv, p, shares, lo, hi in run_tasks(
                _task_depth, tasks, rows, "clipping by depth"):
            rec = {"dimension": dim, "level": lv, "param": p}
            rec.update({f">{d:g}lv": shares[d]
                        for d in CLIP_REPORT_DEPTHS})
            # Which END is binding. The pooled share cannot say, and
            # the two ends mean different things: shadows going to
            # black is the risk the gain runs, highlights going to
            # white would mean the mean-restoring step overshoots.
            rec[f"below0@{CLIP_DEPTH_UP:g}"] = lo
            rec[f"above255@{CLIP_DEPTH_UP:g}"] = hi
            drows.append(rec)
    if drows:
        depth_tbl = pd.DataFrame(drows)
        print("\n  clipping by depth (share of pixels pushed more than "
              "N levels past the range):")
        print("  " + depth_tbl.round(4).to_string(index=False)
              .replace("\n", "\n  "))
        print(f"    CLIP_DEPTH_UP = {CLIP_DEPTH_UP:g}, cap "
              f"{CLIP_MAX_NEW_UP:.3f}")
        depth_tbl.to_csv(
            os.path.join(OUTPUT_DIR, "clip_by_depth.csv"),
            index=False, float_format="%.6g")
        if COMPARE_CONTRAST_SPACES:
            print("\n  the two contrast spaces, at matched target:")
            a = depth_tbl[depth_tbl.dimension == "contrast_up"]
            b = depth_tbl[depth_tbl.dimension == "contrast_up_lin"]
            col = f">{CLIP_DEPTH_UP:g}lv"
            for lv in levels_of("contrast_up"):
                ra = a[a.level == lv]
                rb = b[b.level == lv]
                if len(ra) and len(rb):
                    print(f"    {lv}: sRGB {float(ra[col].iloc[0]):.4f}"
                          f"   linear {float(rb[col].iloc[0]):.4f}")
            print("    Lower is better. Whichever wins on the sample is "
                  "the space to keep;")
            print("    set CONTRAST_UP_FAMILY and COMPARE_CONTRAST_SPACES "
                  "= False.")

    # --- did the solver land where the documentation says? ---
    #
    # The gain is not hard-coded; it is whatever the cap allows. This
    # compares that to the value on record, so a run that drifts says
    # so instead of leaving a stale number in the write-up.
    chosen = ladder_params(levels, "contrast_up", GRID_CONTRAST_LEVELS)
    if chosen:
        gnow = chosen[0][1]
        same = abs(gnow - CONTRAST_UP_GAIN_EXPECTED) < 1e-6
        m = levels[(levels.dimension == "contrast_up")
                   & (levels.level == GRID_CONTRAST_LEVELS[0])]
        sdv = float(m.achieved_sd.iloc[0]) if len(m) else float("nan")
        print(f"\n  contrast-up column: gain {gnow:.4f}, +{sdv:.4f} SD")
        if same:
            print(f"    matches CONTRAST_UP_GAIN_EXPECTED "
                  f"({CONTRAST_UP_GAIN_EXPECTED}) -> PASS")
        else:
            print(f"    DIFFERS from CONTRAST_UP_GAIN_EXPECTED "
                  f"({CONTRAST_UP_GAIN_EXPECTED}).")
            print("    The cap, the grid or the sample has moved. "
                  "Update the setting and the")
            print("    documentation together, or find out why.")
        report.append(("contrast-up column",
                       f"gain {gnow:.4f}, +{sdv:.4f} SD"
                       + ("" if same else "  <- NOT the recorded gain")))

    report.append(("check 9 new levels",
                   ("PASS" if ok else "FAIL")
                   + f" ({len(capped)} capped, {len(over)} over the "
                     f"clipping cap at depth {CLIP_DEPTH_UP:g})"))
    return ok


def check_levels_unchanged(levels, report):
    """CHECK 10: the levels that already existed are unchanged.

    Adding percentiles, dimensions and manipulations must not move a
    single parameter of a level the current stimulus set was built
    from. If one moves, something in the shared machinery changed and
    every file downstream of it is stale -- which is exactly the
    failure this pipeline has hit before, so it is checked rather than
    assumed.

    Compared on the parameter, to the tolerance the grid itself can
    resolve. Levels that are new in this run are listed, not compared.
    """
    print("\n" + "=" * 60)
    print("EXISTING LEVELS UNCHANGED  (check 10)")
    print("=" * 60)
    if not (PREV_LEVELS_CSV and os.path.isfile(PREV_LEVELS_CSV)):
        print(f"  {PREV_LEVELS_CSV} not found; skipped")
        report.append(("check 10 existing levels", "skipped"))
        return True
    prev = pd.read_csv(PREV_LEVELS_CSV)
    key = ["dimension", "level"]
    a = prev.set_index(key)["param"]
    b = levels.set_index(key)["param"]
    shared = a.index.intersection(b.index)
    added = b.index.difference(a.index)
    dropped = a.index.difference(b.index)

    # RELATIVE tolerance, because levels.csv is written with
    # float_format="%.6g" -- six significant digits. A parameter near
    # 0.7 is therefore stored to within 5e-7 of its true value, so an
    # absolute tolerance below that can never pass: the first run
    # "failed" at a worst delta of 4.9e-07, which is exactly that
    # rounding half-spacing and not a changed level. 1e-5 relative sits
    # far inside what the grid can resolve and far outside the storage
    # rounding.
    rel = (b.loc[shared] - a.loc[shared]).abs() / a.loc[shared].abs()\
        .clip(lower=1e-12)
    diff = rel
    worst = float(diff.max()) if len(diff) else 0.0
    tol = 1e-5
    bad = diff[diff > tol]
    print(f"  {len(shared)} level(s) compared against "
          f"{PREV_LEVELS_CSV}")
    print(f"  worst RELATIVE parameter difference: {worst:.3e} "
          f"(tolerance {tol:.0e})")
    if len(added):
        print(f"  new in this run ({len(added)}): "
              + ", ".join(f"{d}/{l}" for d, l in added))
    if len(dropped):
        print(f"  WARNING: {len(dropped)} level(s) present before and "
              "missing now: "
              + ", ".join(f"{d}/{l}" for d, l in dropped))
    ok = not len(bad) and not len(dropped)
    if len(bad):
        print("  FAIL: these existing levels moved:")
        for (d, l), v in bad.items():
            print(f"    {d}/{l}: {a.loc[(d, l)]:.6f} -> "
                  f"{b.loc[(d, l)]:.6f}  (delta {v:.3e})")
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    report.append(("check 10 existing levels",
                   ("PASS" if ok else "FAIL")
                   + f" (worst relative delta {worst:.1e}, "
                     f"{len(added)} new)"))
    return ok


CAPPED_STATUS_COL = "status"


def check_envelope(side_effects, stats, report, levels_tbl):
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
        return pd.DataFrame(), {}, pd.DataFrame()

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

    # --- colour-related features, reported rather than counted ---
    #
    # ENVELOPE_FEATURES deliberately leaves these out: for the colour
    # dimensions they measure the same construct as the calibrated
    # feature, so taking them outside the corpus range is the
    # manipulation working, not a fault -- exactly as sat_mean itself
    # is excluded. But chroma ABOVE 1 is the one place in this pipeline
    # that pushes them UP, and whether the vivid levels leave the
    # corpus extremes is a question worth an answer rather than an
    # assumption. Reported here, never added to the violation count.
    colour_rows = []
    for r in side_effects.itertuples():
        if r.feature not in COLOUR_RELATED_FEATURES:
            continue
        if r.dimension not in ("colour", "colour_up"):
            continue
        lo = float(st.at[r.feature, ENVELOPE_LO])
        hi = float(st.at[r.feature, ENVELOPE_HI])
        v = float(r.achieved_median)
        colour_rows.append({
            "dimension": r.dimension, "level": r.level,
            "feature": r.feature, "achieved": v,
            "corpus_lo": lo, "corpus_hi": hi,
            "outside": "above" if v > hi else
                       ("below" if v < lo else ""),
            "outside_by_sd": ((v - hi if v > hi else v - lo)
                              / float(st.at[r.feature, "sd"]))
            if (v > hi or v < lo) else 0.0})
    colour_env = pd.DataFrame(colour_rows)
    if len(colour_env):
        out = colour_env[colour_env.outside != ""]
        print("\n  colour-related features (reported, not counted as "
              "violations):")
        if len(out):
            print(f"    {len(out)} of {len(colour_env)} level-feature "
                  "pairs sit outside the corpus "
                  f"{ENVELOPE_LO}-{ENVELOPE_HI} range:")
            print("    " + out.round(3).to_string(index=False)
                  .replace("\n", "\n    "))
        else:
            print(f"    all {len(colour_env)} inside the corpus "
                  f"{ENVELOPE_LO}-{ENVELOPE_HI} range")
        report.append(("colour-related outside corpus range",
                       f"{len(out)} of {len(colour_env)}"))

    # The deepest level of each dimension with no violation, ordered by
    # that dimension's OWN levels (see levels_of). Levels the run never
    # calibrated are not candidates: a capped level has no parameter, so
    # calling it "inside the envelope" would be meaningless.
    # Only levels that were actually calibrated are candidates. A
    # capped level has no parameter of its own -- interp_param hands
    # back the last surviving grid point -- so calling it "inside the
    # envelope" would describe a level the run never built. The first
    # run reported contrast_up p90 on exactly that basis, while every
    # contrast_up level shared one fallback parameter.
    have = set(zip(side_effects.dimension, side_effects.level))
    if CAPPED_STATUS_COL in levels_tbl.columns:
        ok_lv = levels_tbl[levels_tbl[CAPPED_STATUS_COL] == "ok"]
        have &= set(zip(ok_lv.dimension, ok_lv.level))
    deepest = {}
    for dim in DIMENSIONS:
        bad = set(viol[viol.dimension == dim].level) if len(viol) else set()
        ok = [lv for lv in levels_of(dim)
              if lv not in bad and (dim, lv) in have]
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
    # Named for what it actually computes. Capped levels are excluded
    # (they are not levels anyone can use), so for a dimension whose
    # deeper levels are capped this is NOT an envelope limit: it is the
    # deepest SOLVED level, and the levels past it were left out for
    # being capped, not for leaving the envelope. Read as an envelope
    # limit it says the opposite of the truth -- contrast_up p65 sits
    # inside the envelope and still does not appear here.
    print("  deepest SOLVED level inside the envelope "
          "(capped levels excluded, so this is not an envelope "
          "limit): "
          + ", ".join(f"{d} {lv}" for d, lv in deepest.items()))
    report.append(("check 6 envelope",
                   f"{len(viol)} violations; deepest solved level "
                   f"inside (capped excluded): "
                   + ", ".join(f"{d} {lv}" for d, lv in deepest.items())))
    return viol, deepest, colour_env


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
    # Wrapped rather than one long row: six dimensions in a single row
    # is 42 inches wide, which is unreadable at any sensible zoom.
    ncol = min(3, len(dims))
    nrow = int(np.ceil(len(dims) / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(7 * ncol, 5 * nrow),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(dims):]:
        ax.axis("off")
    for ax, dim in zip(flat, dims):
        feat = DIMENSIONS[dim][1]
        sd = float(st.at[feat, "sd"])
        d = per_image[per_image.dimension == dim]
        # levels_of, not TARGET_PERCENTILES: the upward dimensions
        # use p55..p90, so ordering by the downward list left their
        # panels empty. Same bug as the report's side-effects pivot and
        # the envelope's deepest-inside line.
        order = [lv for lv in levels_of(dim) if lv in set(d.level)]
        if not order:
            # Say so on the panel rather than drawing a blank box: an
            # empty panel looks like a plotting fault, and the first
            # run's empty colour_up panels were read as exactly that.
            ax.text(0.5, 0.5, f"{dim}\nno levels solved",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
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
    # Vivid levels first, then the original, then the reductions, so
    # the row reads left to right as more colour -> less colour. The
    # contact sheets that decide whether the vivid end looks artificial
    # are step 3's job, but they cost nothing here and the question is
    # worth seeing early.
    col = (list(reversed(ladder_params(levels, "colour_up",
                                       EXAMPLE_COLOUR_UP)))
           + [("k=1", 1.0)]
           + ladder_params(levels, "colour", EXAMPLE_COLOUR))
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
            ctx_c = ctx
            if k > 1.0:
                # Haze has already run, and it moves both lightness and
                # hue, so the in-gamut chroma ceiling has to be read off
                # the image AS IT STANDS. Taking it from the unmodified
                # square -- which is what 09c did -- would use a ceiling
                # that belongs to a different image.
                ctx_c = dict(ctx, cmax=cmax_of_img(img))
            img, _ = apply("chroma", img, k, ctx_c)
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


def contrast_ladder_sheet(rows, levels, out):
    """A contact sheet of the FULL contrast dimension, one row per
    scene: the contrast-increase column, the original, then the hazed
    levels, left to right.

    The point is to read the row as ONE continuous dimension. The
    column above the original is made a different way from the ones
    below it -- a gain about the scene's own mean grey, where the hazed
    columns blend toward the airlight -- so the join at the original is
    where a seam would show, and this is the sheet that would show it.

    Dark scenes are forced into the sample so that the join is shown
    across the brightness range, not because the shadows are where the
    gain fails. They are not: measured on every stimulus scene, nothing
    went below 0 at any gain. The bright end is what binds, and the
    sheet that judges it is gain_sweep.png, which is selected on the
    measurement and rendered at full resolution. This sheet is for the
    seam at the original.
    """
    up = ladder_params(levels, "contrast_up", GRID_CONTRAST_LEVELS)
    if not up:
        print("  contrast ladder: no contrast_up level calibrated; "
              "skipped")
        return
    fog_dim, fog_fam, ladder_name = FOG_VARIANTS[-1]
    down = ladder_params(levels, fog_dim, globals()[ladder_name])
    if not down:
        print("  contrast ladder: no fog ladder resolved; skipped")
        return

    cols = ([(f"+{nm}", CONTRAST_UP_FAMILY, p)
             for nm, p in reversed(up)]
            + [("original", None, None)]
            + [(nm, fog_fam, p) for nm, p in down])

    pick = pick_scenes(rows, FEAT_FOG, CONTRAST_SHEET_SCENES)
    dark = sorted(range(len(rows)), key=lambda i: rows[i]["base"]["lum_mean"])
    for i in dark[:CONTRAST_SHEET_DARK_SCENES]:
        if i not in pick:
            pick.append(i)

    def label(i):
        r = rows[i]
        return (f"{r['category']} #{r['image_id']}\n"
                f"lum {r['base']['lum_mean']:.0f} "
                f"con {r['base'][FEAT_FOG]:.0f}")

    cells = [[(rows[i]["square"] if fam is None
               else apply(fam, rows[i]["square"], p, ctx_of(rows[i]))[0])
              for _, fam, p in cols] for i in pick]

    # The up column is labelled by the gain it uses and the contrast it
    # achieves, NOT by the percentile it was aimed at: it does not
    # reach that percentile, and a label naming one it never reached is
    # how an overstated number gets into a paper unchallenged. The
    # hazed columns keep their percentile names, which they do reach.
    ach = {}
    for nm, _f, p in cols:
        if p is not None and nm.startswith("+"):
            m = levels[(levels.dimension == "contrast_up")
                       & (levels.level == nm[1:])]
            if len(m):
                ach[nm] = float(m.achieved_sd.iloc[0])
    col_labels = []
    for nm, _f, p in cols:
        if p is None:
            col_labels.append(nm)
        elif nm in ach:
            col_labels.append(f"gain {p:.3f}\n+{ach[nm]:.2f} SD")
        else:
            col_labels.append(f"{nm}\n{p:.3f}")
    _sheet(cells, [label(i) for i in pick], col_labels,
           "contrast dimension end to end. LEFT of 'original' is a gain "
           "about the scene mean; RIGHT of it is haze. The join at "
           "'original' is where a seam would show, and scenes are "
           "spread across the brightness range so the join can be "
           "judged at both ends of it. For the bright-end cost of the "
           "gain column see gain_sweep.png.",
           os.path.join(out, "ladder_contrast_full.png"))
    print(f"  contrast ladder: {len(pick)} scenes x {len(cols)} columns "
          "-> examples/ladder_contrast_full.png")


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


def _sheet(cells, row_labels, col_labels, title, path, thumb=256,
           captions=None, cell_in=1.7, dpi=130):
    """A labelled contact sheet; cells[r][c] is a uint8 image.

    `captions`, if given, is the same shape as `cells` and is drawn
    under each cell -- the numbers that go with the picture, next to
    the picture, so the sheet can be read on its own months later.

    The title is WRAPPED to the figure width. It used to be one long
    line, and matplotlib does not wrap suptitle: the sheet that was
    supposed to explain itself had both its ends cut off, which is the
    one failure a self-describing output must not have.
    """
    nr, nc = len(cells), len(cells[0])
    # cell_in * dpi is the cell's size in output pixels. Pass
    # thumb = cell_in * dpi to get one image pixel per output pixel,
    # which is what a sheet meant for zooming needs.
    fw = cell_in * nc + 1.2
    scale = cell_in / 1.7
    cap_h = (0.62 if captions else 0.0)
    wrapped = textwrap.fill(" ".join(title.split()),
                            width=max(40, int(fw * 11 / max(scale, 1))))
    nlines = wrapped.count("\n") + 1
    title_h = (0.21 * nlines + 0.10) * max(scale, 1.0)
    fh = 1.7 * nr + 0.9 + cap_h * nr + title_h
    fig, axes = plt.subplots(nr, nc, figsize=(fw, fh), squeeze=False)
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
                ax.set_title(col_labels[j],
                             fontsize=min(14, 8 * scale))
            if j == 0:
                ax.set_ylabel(row_labels[i],
                              fontsize=min(12, 7 * scale))
            if captions:
                ax.set_xlabel(captions[i][j],
                              fontsize=min(11, 6 * scale))
    fig.suptitle(wrapped, fontsize=min(16, 10 * scale),
                 y=1 - 0.08 / fh, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - title_h / fh))
    # tight_layout decides the REAL axes size, and it is always smaller
    # than cell_in once the labels and the padding are taken out of it.
    # Asking for 512 px cells at 200 dpi and 2.56 in got 314 px -- the
    # sheet was still being downsampled, quietly, which is the one
    # fault full resolution exists to prevent. So measure the axes that
    # tight_layout actually produced and raise the dpi until they reach
    # `thumb` pixels, with a ceiling so a large sheet cannot run away.
    frac = axes[0][0].get_position().width
    if frac > 0:
        need = thumb / (frac * fw)
        if need > dpi:
            dpi = min(need, 14000.0 / max(fw, fh))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def load_stimulus_scenes():
    """The photographs the stimulus set is built on, as 512 px central
    squares, measured.

    Returns [] -- not an exception -- when SCENES_CSV is missing or
    unreadable. The gain sheet decides one column; the ladders do not
    depend on it, and a calibration run that aborts because a
    diagnostic's input moved would be the wrong trade.
    """
    if not (SCENES_CSV and os.path.isfile(SCENES_CSV)):
        print(f"  stimulus scenes: {SCENES_CSV} not found; "
              "gain sheet skipped")
        print("    point SCENES_CSV at the stimulus manifest or the "
              "screened worksheet.")
        return []
    t = pd.read_csv(SCENES_CSV)
    pcol = next((c for c in ("source_path", "path", "file", "filename")
                 if c in t.columns), None)
    if pcol is None:
        warnings.warn(f"{SCENES_CSV} has no path column; "
                      "gain sheet skipped.")
        return []
    if "keep" in t.columns:
        t = t[pd.to_numeric(t["keep"], errors="coerce").fillna(0) > 0]
    key = "scene_id" if "scene_id" in t.columns else pcol
    t = t.drop_duplicates(subset=[key])

    rows, bad = [], 0
    for _, r in t.iterrows():
        try:
            with Image.open(r[pcol]) as im:
                sq = np.asarray(central_square(im.convert("RGB"),
                                               IMAGE_SIZE), np.uint8)
        except Exception:
            bad += 1
            continue
        rows.append({
            "scene_id": str(r[key]),
            "category": str(r["category"]) if "category" in t.columns
                        else "",
            "path": r[pcol],
            "square": sq,
            "airlight": airlight_of(sq),
            "lum_mean_512": mean_grey(sq),
            "base": measure(sq)})
    if bad:
        warnings.warn(f"{bad} stimulus scenes could not be read.")
    print(f"  stimulus scenes: {len(rows)} loaded from {SCENES_CSV}")
    return rows


def _ends(img, x, depth):
    """(share pushed below 0, share pushed above 255) by more than
    `depth` levels, counting only channels that started strictly
    inside the range -- the same exclusion apply() makes, for the same
    reason: a channel already at 0 was not pushed anywhere.

    Split because the two ends fail differently and are fixed
    differently, and because which one binds is NOT obvious in advance.
    A gain about the mean pulls dark pixels down, so crushed shadows
    are the expected risk -- but the mean-restoring step that follows
    pushes the whole frame back up, so a scene with a bright source in
    a dark room can clip at the TOP and not at the bottom at all. A
    test on exactly that scene did: 1.9% above 255, nothing below 0.
    The pooled share cannot tell those apart, which is the whole reason
    for reporting them separately rather than assuming the answer.
    """
    inside = (img > 0) & (img < 255)
    lo = ((x < -depth) & inside).any(axis=2).mean()
    hi = ((x > 255.0 + depth) & inside).any(axis=2).mean()
    return float(lo), float(hi)


def _gain_stats(r, g, sd):
    """Everything one (scene, gain) says about the contrast-up column.

    Returns the record and the rendered image, so the caller can build
    the sheet from the same render the numbers came from.
    """
    sq = r["square"]
    x = FUNCS[CONTRAST_UP_FAMILY](sq, g, ctx_of(r))
    img = to_u8(x)
    d = measure(img)[FEAT_FOG] - r["base"][FEAT_FOG]

    # --- can anything reach the dark end at all? ---
    #
    # The gain is I' = m + g(I - m) about the scene's mean grey, so a
    # channel lands below 0 only if it starts below m(g-1)/g. The
    # mean-restoring step that follows can only push values UP -- it
    # exists to put back the mean that clipping at 255 took away -- so
    # this threshold is an upper bound on what can go negative, and a
    # share of zero here is a reason rather than a coincidence.
    m = mean_grey(sq)
    thr = m * (g - 1.0) / g
    inside = (sq > 0) & (sq < 255)
    rec = {"scene_id": r["scene_id"], "category": r["category"],
           "lum_mean_512": m, "gain": g,
           "d_contrast_local": d, "d_sd": d / sd,
           "dark_threshold": thr,
           "share_dark_enough": float(((sq < thr) & inside)
                                      .any(axis=2).mean()),
           # detail destroyed, at both ends: a channel at least
           # FLATTEN_MARGIN levels inside the range that renders to
           # exactly 255 or exactly 0
           "flattened": float((((sq <= 255 - FLATTEN_MARGIN)
                                & (img == 255)).any(axis=2).mean())),
           "crushed": float((((sq >= FLATTEN_MARGIN)
                              & (img == 0)).any(axis=2).mean()))}
    for dep in CLIP_REPORT_DEPTHS:
        a, b = _ends(sq, x, dep)
        rec[f"below0_{dep:g}lv"] = a
        rec[f"above255_{dep:g}lv"] = b
    return rec, img


def check_pil_equivalence(scenes, report):
    """CHECK 11: is the contrast-up column Pillow's
    ImageEnhance.Contrast?

    Pillow builds a flat image at the mean of the L channel and blends:

        mean       = int(Stat(img.convert("L")).mean[0] + 0.5)
        enhance(f) = mean + f * (img - mean)

    which is the gain step of m_contrast_gain_srgb, with two knowable
    differences:

      the mean    Pillow ROUNDS it to an integer and takes it from
                  PIL's own L conversion; this script keeps it as a
                  float from cv2's. Both are Rec. 601, so the two means
                  differ by less than a level, and a difference d in
                  the mean moves every pixel by d(1 - g) -- at g =
                  1.0875 that is 0.09 of a level per level of mean, so
                  under 0.05 levels. Invisible, but not exactly zero.

      the restore this script then rescales exposure to put back the
                  mean grey that clipping at 255 removes. Pillow does
                  not. That step is OURS and it is what makes the two
                  differ by more than rounding.

    So the check reports BOTH comparisons. The claim the documentation
    is allowed to make is the first one: the gain is Pillow's
    operation. The second number is the size of what this script adds
    on top, and saying it is zero when it is not would be the kind of
    claim a reader checks in two lines and catches.
    """
    if not scenes:
        report.append(("check 11 pillow", "skipped (no scenes)"))
        return
    g = CONTRAST_UP_GAIN_EXPECTED
    rows = []
    for r in scenes[:PIL_CHECK_SCENES]:
        sq = r["square"]
        pil = np.asarray(ImageEnhance.Contrast(Image.fromarray(sq))
                         .enhance(g), np.uint8)
        m = mean_grey(sq)
        gain_only = to_u8(m + g * (_f(sq) - m))
        full = to_u8(FUNCS["contrast_gain_srgb"](sq, g, ctx_of(r)))
        pil_m = ImageStat.Stat(Image.fromarray(sq).convert("L")).mean[0]
        rows.append({
            "scene_id": r["scene_id"],
            "mean_cv2": m, "mean_pil": int(pil_m + 0.5),
            "gain_max": int(np.abs(gain_only.astype(np.int16)
                                   - pil.astype(np.int16)).max()),
            "gain_over1": float((np.abs(gain_only.astype(np.int16)
                                        - pil.astype(np.int16)) > 1)
                                .mean()),
            "full_max": int(np.abs(full.astype(np.int16)
                                   - pil.astype(np.int16)).max()),
            "full_over1": float((np.abs(full.astype(np.int16)
                                        - pil.astype(np.int16)) > 1)
                                .mean())}) 
    t = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print(f"CHECK 11: ImageEnhance.Contrast(factor={g})")
    print("=" * 60)
    print("  " + t.round(4).to_string(index=False).replace("\n", "\n  "))
    ok = bool((t.gain_max <= 1).all())
    print(f"\n  gain step vs Pillow      : max |diff| "
          f"{int(t.gain_max.max())} level(s) -> "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"  full family vs Pillow    : max |diff| "
          f"{int(t.full_max.max())} level(s), "
          f"{t.full_over1.max() * 100:.1f}% of pixels past 1 level")
    print("  The second line is the mean-restoring step, which Pillow "
          "has no equivalent of.")
    report.append(("check 11 pillow",
                   f"{'PASS' if ok else 'FAIL'} (gain step within "
                   f"{int(t.gain_max.max())} level; full family "
                   f"{int(t.full_max.max())})"))
    t.to_csv(os.path.join(OUTPUT_DIR, "pil_equivalence.csv"),
             index=False, float_format="%.6g")


def gain_diagnostic_sheet(stats, report):
    """The sweep that decides the contrast-up column.

    EVERY stimulus scene is measured at every gain; the sheet then
    shows the WORST ones, at full resolution. See GAIN_SHEET_WORST for
    why the selection is no longer the darkest scenes.

    The question this sheet answers is not "does it clip", which the
    numbers already answer. It is whether the column is visibly clearer
    than the original while both ends of the range still hold, because
    a column nobody can distinguish from the original turns its trials
    into noise -- which is the one way this column cannot be allowed to
    fail. That is a judgement about a picture, so the script lays the
    picture out and reports the numbers beside it rather than choosing.
    """
    scenes = load_stimulus_scenes()
    if not scenes:
        report.append(("gain sheet", "skipped (no stimulus scenes)"))
        return pd.DataFrame()
    sd = float(stats.set_index("feature").at[FEAT_FOG, "sd"])
    check_pil_equivalence(scenes, report)

    # --- every scene, every gain ---
    recs = []
    t0 = time.time()
    for n, r in enumerate(scenes, 1):
        for g in GAIN_SHEET_GAINS:
            recs.append(_gain_stats(r, g, sd)[0])
        print(f"    {n}/{len(scenes)} scenes", end="\r")
    tbl = pd.DataFrame(recs)
    tbl.to_csv(os.path.join(OUTPUT_DIR, "gain_sweep.csv"),
               index=False, float_format="%.6g")
    print(f"\n  gain sweep: {len(scenes)} scenes x "
          f"{len(GAIN_SHEET_GAINS)} gains in {time.time() - t0:.0f}s")

    deep = f"above255_{max(CLIP_REPORT_DEPTHS):g}lv"
    at = tbl[np.isclose(tbl.gain, GAIN_SHEET_PICK_GAIN)]
    if not len(at):
        at = tbl[tbl.gain == tbl.gain.max()]

    # --- the distribution, which is what the five-scene sheet could
    #     not show ---
    print("\n  bright-end clipping across ALL stimulus scenes "
          "(share of pixels pushed past 255):")
    rows = []
    for g in GAIN_SHEET_GAINS:
        t = tbl[np.isclose(tbl.gain, g)]
        row = {"gain": g, "+SD median": t.d_sd.median()}
        for dep in CLIP_REPORT_DEPTHS:
            row[f">{dep:g}lv med"] = t[f"above255_{dep:g}lv"].median()
        row[f">{max(CLIP_REPORT_DEPTHS):g}lv p90"] = \
            t[deep].quantile(0.90)
        row[f">{max(CLIP_REPORT_DEPTHS):g}lv max"] = t[deep].max()
        row["over line"] = int((t[deep] > VISIBLE_DEEP_SHARE).sum())
        rows.append(row)
    dist = pd.DataFrame(rows)
    print("  " + dist.round(4).to_string(index=False)
          .replace("\n", "\n  "))

    # --- the dark end: why 'below 0' is zero ---
    zero = all((tbl[f"below0_{d:g}lv"] == 0).all()
               for d in CLIP_REPORT_DEPTHS)
    print("\n  dark end. A channel can only go below 0 if it starts "
          "below m(g-1)/g,")
    print("  with m the scene's mean grey; the mean-restoring step "
          "only pushes values up.")
    for g in GAIN_SHEET_GAINS:
        t = tbl[np.isclose(tbl.gain, g)]
        print(f"    gain {g:.3f}: threshold {t.dark_threshold.min():.1f}"
              f" .. {t.dark_threshold.max():.1f} "
              f"(median {t.dark_threshold.median():.1f}); "
              f"pixels that dark: median "
              f"{t.share_dark_enough.median() * 100:.3f}%, max "
              f"{t.share_dark_enough.max() * 100:.3f}%")
    print(f"    every below-0 share in the table is zero: {zero}")
    report.append(("gain sheet dark end",
                   "all below-0 shares zero" if zero
                   else "SOME BELOW-0 CLIPPING"))

    # --- which scenes are over the line ---
    over = at[at[deep] > VISIBLE_DEEP_SHARE].sort_values(
        deep, ascending=False)
    print(f"\n  deep bright-end share at gain "
          f"{GAIN_SHEET_PICK_GAIN:.4f}: the "
          f"{len(over)} scenes above {VISIBLE_DEEP_SHARE * 100:.0f}%, "
          "worst first.")
    print("  REPORTED, not a threshold: it counts how blown an "
          "already-blown region gets,")
    print("  and it barely tracks real loss. flattened/crushed are "
          "the criterion -- pixels")
    print(f"  at least {FLATTEN_MARGIN} levels inside the range that "
          "render to 255 or to 0.")
    if len(over):
        for _, r in over.iterrows():
            print(f"    {r.scene_id:<34} deep {r[deep] * 100:5.2f}%   "
                  f"flattened {r.flattened * 100:5.2f}%   "
                  f"crushed {r.crushed * 100:5.2f}%")
    else:
        print("    none")
    print(f"\n  real detail loss at gain {GAIN_SHEET_PICK_GAIN:.4f}, "
          "over ALL scenes:")
    for nm in ("flattened", "crushed"):
        print(f"    {nm:<10} non-zero on {int((at[nm] > 0).sum())} of "
              f"{len(at)} scenes; median {at[nm].median() * 100:.2f}%, "
              f"worst {at[nm].max() * 100:.2f}% "
              f"({at.loc[at[nm].idxmax(), 'scene_id']})")
    report.append(("gain sheet worst loss",
                   f"flattened {at.flattened.max() * 100:.2f}%, "
                   f"crushed {at.crushed.max() * 100:.2f}%"))
    over[["scene_id", "category", "lum_mean_512", "gain", "d_sd",
          deep, "flattened", "crushed"]].to_csv(
        os.path.join(OUTPUT_DIR, "gain_over_line.csv"),
        index=False, float_format="%.6g")
    report.append(("gain sheet over line",
                   f"{len(over)} of {len(scenes)} scenes at "
                   f"gain {GAIN_SHEET_PICK_GAIN:.3f}"))

    # --- the sheet: the worst, plus one dark scene for contrast ---
    rank = at.sort_values(deep, ascending=False).scene_id.tolist()
    chosen = rank[:GAIN_SHEET_WORST]
    dark = sorted(scenes, key=lambda r: r["lum_mean_512"])
    for r in dark:
        if len(chosen) >= GAIN_SHEET_WORST + GAIN_SHEET_DARK:
            break
        if r["scene_id"] not in chosen:
            chosen.append(r["scene_id"])
    by_id = {r["scene_id"]: r for r in scenes}
    pick = [by_id[s] for s in chosen if s in by_id]

    cells, caps = [], []
    for r in pick:
        row_img = [r["square"]]
        row_cap = [f"baseline\n{FEAT_FOG} {r['base'][FEAT_FOG]:.1f}"]
        for g in GAIN_SHEET_GAINS:
            rec, img = _gain_stats(r, g, sd)
            row_img.append(img)
            mark = " <- SHIPS" if abs(
                g - CONTRAST_UP_GAIN_EXPECTED) < 1e-9 else ""
            row_cap.append(
                f"+{rec['d_sd']:.2f} SD{mark}\n"
                f"hi {rec[f'above255_{CLIP_DEPTH_UP:g}lv'] * 100:.1f}%"
                f"  lo {rec[f'below0_{CLIP_DEPTH_UP:g}lv'] * 100:.1f}%"
                f"  deep {rec[deep] * 100:.1f}%\n"
                f"flattened {rec['flattened'] * 100:.2f}%  "
                f"crushed {rec['crushed'] * 100:.2f}%")
        cells.append(row_img)
        caps.append(row_cap)

    med = tbl.groupby("gain").d_sd.median()
    col_labels = ["original"] + [
        f"gain {g:.3f}\n+{med.get(g, float('nan')):.2f} SD median"
        for g in GAIN_SHEET_GAINS]
    row_labels = [f"{r['scene_id']}\nlum {r['lum_mean_512']:.0f}"
                  for r in pick]
    out = os.path.join(OUTPUT_DIR, "examples")
    os.makedirs(out, exist_ok=True)
    _sheet(cells, row_labels, col_labels,
           f"contrast gain ({CONTRAST_UP_FAMILY}): the "
           f"{GAIN_SHEET_WORST} stimulus scenes that clip MOST at the "
           f"bright end at gain {GAIN_SHEET_PICK_GAIN:.4f}, plus the "
           f"darkest scene. Full resolution -- zoom into the bright "
           f"regions. Under each cell: contrast gained; hi/lo = pixels "
           f"past 255 / past 0 by more than {CLIP_DEPTH_UP:g} levels; "
           f"deep = past 255 by more than "
           f"{max(CLIP_REPORT_DEPTHS):g}; flattened/crushed = pixels "
           f"at least {FLATTEN_MARGIN} levels inside the range that "
           f"became 255 or 0, which is the detail actually lost.",
           os.path.join(out, "gain_sweep.png"), thumb=IMAGE_SIZE,
           captions=caps, cell_in=GAIN_SHEET_CELL_IN,
           dpi=GAIN_SHEET_DPI)
    print(f"\n  sheet: {len(pick)} scenes x {len(GAIN_SHEET_GAINS) + 1}"
          f" columns at {IMAGE_SIZE} px -> examples/gain_sweep.png")
    report.append(("gain sheet",
                   f"{len(scenes)} scenes measured, "
                   f"{len(pick)} on the sheet"))
    return tbl


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
    # Vivid levels first, then the original, then the reductions, so
    # the row reads left to right as more colour -> less colour. The
    # contact sheets that decide whether the vivid end looks artificial
    # are step 3's job, but they cost nothing here and the question is
    # worth seeing early.
    col = (list(reversed(ladder_params(levels, "colour_up",
                                       EXAMPLE_COLOUR_UP)))
           + [("k=1", 1.0)]
           + ladder_params(levels, "colour", EXAMPLE_COLOUR))
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

    # --- the contrast dimension end to end ---
    contrast_ladder_sheet(rows, levels, out)

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
    W(f"Grids (up)         : chroma 1 .. {CHROMA_MAX}, "
      f"{CONTRAST_UP_FAMILY} 1 .. {1.0 - HAZE_MIN:g}")
    # Both caps, and the depth the up one is read at. One number here
    # used to stand for two different rules, which is exactly how a
    # reader ends up attributing the down ladder's 0.01 to a column
    # that was never held to it.
    W(f"Clipping cap       : {CLIP_MAX_NEW} down (half-level rule), "
      f"{CLIP_MAX_NEW_UP} up (at depth {CLIP_DEPTH_UP:g} levels)")
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
            order = [lv for lv in levels_of(dim)
                     if lv in piv.columns]
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
        W("\n  Deepest SOLVED level of each dimension that stays "
          "inside the envelope.")
        W("  Capped levels are excluded, because a capped level is not "
          "one anyone can")
        W("  use. So for a dimension whose deeper levels are capped "
          "this is NOT where the")
        W("  envelope stops: the levels past it were dropped for being "
          "capped, and may")
        W("  well be inside. The violations listed above are the "
          "envelope's own answer.")
        for d, lv in deepest.items():
            tag = "  (report-only)" if d in REPORT_ONLY else ""
            shown = lv if lv is not None else "none tested"
            W(f"      {d:<12} {shown}{tag}")
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
    ok_new = check_new_levels(levels, rows, report)
    ok_same = check_levels_unchanged(levels, report)
    viol, deepest, colour_env = check_envelope(side_effects, stats,
                                               report, levels)
    if len(colour_env):
        colour_env.to_csv(
            os.path.join(OUTPUT_DIR, "colour_related_envelope.csv"),
            index=False, float_format="%.6g")
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
    if SAVE_EXAMPLES:
        gain_diagnostic_sheet(stats, report)
    save_examples(rows, levels, report)

    path = write_report(levels, grid, stats, stats_tables, report,
                        identity_lines, sample_lines, ratios, side_effects,
                        viol, deepest, time.time() - t0)
    print(f"\n  Report : {path}")
    print(f"  Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
