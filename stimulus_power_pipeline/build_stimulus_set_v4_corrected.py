"""
build_stimulus_set_v4_corrected.py  (stimulus pipeline, v4 axes, step 3)

Takes the hand-screened scene lists, checks that the set can support
every trial type the experiment needs, selects the final subset, and
writes the stimulus images and the manifest.

Where this sits
---------------
  step 1  calibrate_levels_v4_corrected   the ladders, both directions
  step 2  select_scenes_v4_corrected      shortlist -> you screen by hand
  step 3  this one                        the screened set -> stimuli
  step 4  analyse_pairs_v4                pair inventory and thresholds
  step 5  build_trials_v4                 the trial list

WHAT CHANGED FROM build_stimulus_set_v4.py
------------------------------------------
The old build rendered 5 colour x 5 fog = 25 cells per scene, every one
of them manipulated on both axes. The grid blocks need three things it
did not have, so the cell set grew to 45:

  1. TWO VIVID COLOUR LEVELS, p65 and p75 from step 1's `colour_up`
     ladder. They go in at every fog rung, not only at h = 0, because
     the pair grid uses them too.

  2. THE UNMANIPULATED FOG ROW, h = 0. The colour grid screen holds
     haze at zero and varies colour across it, so those seven cells
     have to exist as files. The old build had no h = 0 cell at all.

  3. THE CONTRAST-INCREASE COLUMN, at three gains. Step 1 could not
     calibrate it -- every percentile target clipped past the budget --
     so it is parameterised here by GAIN rather than by percentile, and
     three gains are rendered so step 7 can choose from pictures at the
     experiment's real cell size. PROVISIONAL: see section 6 of
     v4_step1_settings.md.

Four faults in the old script are fixed, in rising order of how much
they would have cost:

  a. m_chroma for k > 1 reads ctx["cmax"], the per-pixel in-gamut
     chroma ceiling, and ctx_of never supplied it. The old build never
     noticed because it had no level above k = 1. It is computed here
     AFTER the haze step, because haze moves lightness and hue and so
     moves the ceiling -- taking it from the unmodified square, which
     is what 09c did, uses a ceiling belonging to a different image.

  b. lin_to_srgb_f began `x = np.maximum(x, 0.0)`, which makes
     negative values structurally impossible and so makes the dark end
     of a contrast gain impossible to MEASURE. The rendered image is
     unchanged -- to_u8 clips anyway -- but the diagnostics were blind.
     Replaced with the sign-preserving version from step 1.

  c. load_envelope read the stats file's percentiles without requiring
     a `population` column, so a file that lacked one would be read as
     though every row were the population asked for. That is the exact
     fault that cost step 2 a round. It now raises.

  d. The affordance and coverage checks were handed every cell. With
     an h = 0 row and a contrast column in the grid they would now
     silently enter the pair inventory and step 4 would stop matching.
     They are handed the PAIR cells only.

WHAT EACH IMAGE IS FOR
----------------------
Every row of the manifest carries a `use` column, which is what step 4,
step 5 and step 7 filter on:

  pair   the pair grid only: 7 colour x 5 haze rungs, minus the five
         the fog grid screen also shows                        30 cells
  both   the k = 1 column at the five haze rungs: the fog grid screen
         shows these AND the pair trials use them                5 cells
  grid   the grid screens only: the h = 0 row (7 colour cells) and the
         three contrast-increase cells                          10 cells
                                                         total 45 cells

The h = 0 row is deliberately NOT in the pair set. Pairs in the old
build were always two manipulated images, and keeping that true is what
lets step 4 reproduce the old pair table on the old levels.

You screen on CONTENT -- framing, legibility, whether the place is
ordinary. This screens on FEATURE COVERAGE. Neither does the other's
job, which is why they are separate.

TWO STIMULUS SETS
-----------------
FOG_FAMILY picks which fog this build uses, and the output directory is
named after it, so the two sets never collide:

  "haze"     the v2 fog: I' = (1-h)I + hA
  "haze_eq"  the same blend, then a linear-light gain restoring the
             original mean grey   <- THE DEFAULT, and the family the
             final stimulus set uses

The default is now haze_eq, because that is what the shipped set is
built from; "haze" is left as a one-line alternative in the settings.
Run the script twice, once per family, if you want both. The scene
list, the trial-type thresholds and the selection are identical; only
the rendering differs. CHECK 3 verifies that assumption on this set
rather than taking it on trust.

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

On the 47-scene set the second cull has already happened, so the
default is "full" and this is a single pass. Note that CHECK 8 has
nothing to compare in the preview pass -- it reads the PNGs off disk,
and the preview pass does not write any.

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

  CHECK 6  THE COLOUR LADDER IS MONOTONE IN EVERY SCENE, and no step
           collapses. The ladder is calibrated on the corpus median, so
           it is a statement about the typical image, not a guarantee
           about any particular one; a scene whose p65 and p75 cells
           measure the same saturation gives a pair trial with no
           stimulus difference. Checked at every fog rung, because
           haze drains colour and leaves less room above it.

  CHECK 7  CONTRAST-UP DAMAGE, per scene and per gain. `flattened` is
           the share of pixels at least FLATTEN_MARGIN levels below
           white in the original that render as exactly 255, `crushed`
           the same at the black end. These are REPORTED NUMBERS, not a
           threshold: step 7 chooses the gain and the grid scenes
           together, from them and from a mock-up.

  CHECK 8  REPRODUCTION. Every cell this build shares with an earlier
           one is compared to it: first the ladder parameters, then the
           PNG bytes, and where the bytes differ, the largest pixel
           difference and the share of pixels affected. Checking the
           parameters first is what makes a failure readable -- it
           separates "the ladder moved" from "the renderer changed",
           which a bare hash cannot.

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
                          features, `use`, filename. This is what
                          step 4, step 5 and step 7 read.
  cell_plan.csv           the 45 cells, what renders each and what it
                          is for. The same table every pass uses.
  colour_ladder.csv       CHECK 6, per scene x fog rung
  contrast_up_damage.csv  CHECK 7, per scene x gain
  reproduction.csv        CHECK 8, per shared image
  sheets_vivid/<scene>.jpg    full-resolution colour ladder
  sheets_contrast/<scene>.jpg full-resolution contrast-up gains
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
    python build_stimulus_set_v4_corrected.py

RENDER_MODE is "full" and the screening list is the finished one, so
this is a single pass: the 47 scenes are already hand-screened and the
second cull has already happened. Set RENDER_MODE = "preview" if you
want the cheap pass (every check, every sheet, no stimulus images).

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
    "data/select_scenes_v4_ext/screening_list_done.csv",
]

INDEX_CSV    = "data/selection/image_index.csv"
FEATURES_NPY = "data/features/features_crop.npy"
CATEGORY_CSV = "everyday_categories.csv"
LEVELS_CSV   = "data/levels_v4_ext/levels.csv"
STATS_CSV    = "data/levels_v4_ext/corpus_feature_stats_v4.csv"

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
#   The default is the equal-luminance family, which is what the final
#   stimulus set is built from. The plain blend is the alternative:
#       FOG_FAMILY, FOG_DIMENSION = "haze", "fog"
FOG_FAMILY    = "haze_eq"
FOG_DIMENSION = "fog_eq"

# OUTPUT_ROOT
#   The family name is APPENDED, so this writes to
#   data/stimulus_set_v4_ext_haze_eq/ and a "haze" run would write to
#   data/stimulus_set_v4_ext_haze/. The two can never overwrite one
#   another, and neither can touch the earlier _ec builds.
OUTPUT_ROOT = "data/stimulus_set_v4_ext"

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
#
# COLOUR_LEVELS / COLOUR_UP_LEVELS
#   The two halves of the colour axis, read from DIFFERENT blocks of
#   levels.csv: COLOUR_LEVELS from `colour` (chroma below 1) and
#   COLOUR_UP_LEVELS from `colour_up` (chroma above 1). Step 1 keeps
#   them apart because its solver assumes the ladder starts at the
#   identity and moves one way; the names repeat across the two blocks
#   (there is a `colour` p25 and a `colour_up` p65) so the dimension,
#   not the name, is what identifies a rung.
#
#   Seven columns in all: p65, p75, k=1, p25, p10, p05, p01. The grid
#   screen shows two above the original and four below, which is the
#   layout step 7 builds; the pair grid uses all seven at every rung.
#
#   p90 is deliberately absent. Step 1 could not reach it -- the
#   corpus p90 of sat_mean needs a chroma multiplier past the gamut
#   ceiling, so it came back "capped (out of range)" having achieved
#   +0.89 SD against a +1.55 SD target. A level that does not reach
#   its target is not a calibrated level, and ALLOW_CAPPED_LEVELS
#   below is what stops it being rendered as though it were.
COLOUR_LEVELS    = ["p25", "p10", "p05", "p01"]
COLOUR_UP_LEVELS = ["p65", "p75"]
FOG_LEVELS       = ["p35", "p25", "p15", "p10", "p05"]

# INCLUDE_UNMANIPULATED_COLOUR
#   Adds the k = 1 column, the scene's own colour. Needed: the fog-only
#   and conflict pairs live largely in it, and it is the reference the
#   manifest's deltas are taken from.
INCLUDE_UNMANIPULATED_COLOUR = True

# INCLUDE_UNMANIPULATED_FOG
#   Adds the h = 0 ROW, the scene's own contrast. New in this build.
#   The colour grid screen holds haze at zero and varies colour across
#   it, so those seven cells have to exist as files -- the old build
#   had no h = 0 cell at all and the grid block had nothing to show as
#   the original.
#
#   These cells are marked `use = "grid"`, NOT pair: see the setting
#   below.
INCLUDE_UNMANIPULATED_FOG = True

# PAIR_INCLUDES_UNMANIPULATED_FOG
#   Whether the h = 0 row may also be used for PAIR trials.
#
#   False, which is the old build's behaviour made explicit: every pair
#   there was two manipulated images, because the h = 0 cells did not
#   exist. Keeping that true is what lets step 4 reproduce the old pair
#   table on the levels the two builds share -- turn this on and every
#   scene gains a block of pairs against its own untouched photograph,
#   the pair inventory changes, and the check at step 4 stops meaning
#   anything.
#
#   There is a real argument for True: a pair against the unmanipulated
#   image is the cleanest possible comparison, and 7 more cells per
#   scene would enter the inventory. It is a DESIGN change, not a
#   setting to flip quietly -- if you turn it on, step 4 has to be
#   re-run on both settings and the difference reported.
PAIR_INCLUDES_UNMANIPULATED_FOG = False

# The labels the identity cells carry. They have to be stable, because
# they end up in filenames and in the manifest, and step 7 matches on
# them. "k=1" is the old build's label and is kept so the 25 shared
# cells keep their old filenames; "h=0" is new.
COLOUR_IDENTITY_LABEL = "k=1"
FOG_IDENTITY_LABEL    = "h=0"

# ALLOW_CAPPED_LEVELS
#   Whether a level step 1 recorded as anything other than "ok" may be
#   rendered. False, and it RAISES rather than warning.
#
#   The old script warned. That warning could never be seen: the bottom
#   of this file sets warnings.filterwarnings("ignore",
#   category=UserWarning) before calling main, so every
#   warnings.warn in the script is discarded. Anything that must reach
#   you has to print or raise.
#
#   The level this exists for is colour_up p90. Step 1 wrote it a
#   parameter -- the gamut ceiling, 2.0 -- and marked it "capped (out
#   of range)", because the multiplier needed to reach the corpus p90
#   of sat_mean is past that ceiling; it achieved +0.89 SD against a
#   +1.55 SD target. A parameter plus a status is a trap: isfinite()
#   passes, so a ladder naming p90 would render happily and the column
#   would sit at whatever the cap left rather than at a calibrated
#   step, with nothing in the output saying so.
ALLOW_CAPPED_LEVELS = False

# --- The contrast-increase column (grid blocks only) ---
#
# This column is the one part of the design that is NOT calibrated to a
# percentile, and the reason is worth keeping in the script. Step 1
# tried: every target from p55 up came back "capped (clipping)", with
# no parameter written, because the budget it was allowed to spend
# (CLIP_MAX_NEW_UP = 6% of pixels moved at least 4 levels outside the
# range) is exhausted before the feature has moved far. So there is no
# param column to read, and the column is parameterised by GAIN here.
#
# CONTRAST_UP_FAMILY
#   "contrast_gain_srgb"  I' = m + g(I - m) on sRGB VALUES, about the
#                         scene's mean grey, then a mean-restoring
#                         exposure. The same family of operation as
#                         PIL.ImageEnhance.Contrast at factor g, which
#                         step 1 verified to within one level on five
#                         real scenes -- though it is not identical to
#                         it, because of the exposure step.
#   "contrast_gain"       the same gain in linear light. Rejected: at
#                         the clipping budget it solved no level at all
#                         (gain 1.0146, +0.111 SD, 83% of its clipping
#                         at the dark end), where sRGB delivers twice
#                         the contrast with better cross-talk.
CONTRAST_UP_FAMILY = "contrast_gain_srgb"

# CONTRAST_UP_GAINS
#   Three gains, rendered for every scene, so step 7 can choose the
#   gain and the grid scenes TOGETHER from a mock-up at the
#   experiment's real cell size (240 px, not 512).
#
#     1.0875  +0.22 SD, 94% of what a p60 step would be. The value
#             step 1 settled on: the strongest gain that holds the
#             clipping budget across the sample.
#     1.1100  +0.27 SD. 22 of the 47 scenes take no measured damage at
#             all; at a 0.25% tolerance, 28 do.
#     1.1500  UNMEASURED -- step 1's sweep stopped at 1.11, so this one
#             is rendered blind and judged on the pictures. Extrapolating
#             the sweep, expect roughly 8% of pixels past four levels
#             and `flattened` above 3% on the worst scenes.
#
#   All three are PROVISIONAL. If the column is not distinguishable
#   from the original at 240 px it comes out altogether: an
#   indistinguishable column does not reduce the ceiling effect, it
#   splits "my normal is at least this clear" between two columns that
#   look the same, which is worse than a clean pile-up at the original.
CONTRAST_UP_GAINS = [1.0875, 1.110, 1.150]

# CONTRAST_UP_AT
#   Where in the grid the column sits: the scene's own colour and its
#   own contrast as the starting point, so the only thing that differs
#   from the grid's original cell is the gain.
CONTRAST_UP_AT_COLOUR = COLOUR_IDENTITY_LABEL
CONTRAST_UP_AT_FOG    = FOG_IDENTITY_LABEL

# FLATTEN_MARGIN / CLIP_REPORT_DEPTHS
#   How CHECK 7 counts damage, both verbatim from step 1 so the numbers
#   are comparable with gain_sweep.csv.
#
#   `flattened`  share of pixels at least FLATTEN_MARGIN levels BELOW
#                white in the original that render as exactly 255. The
#                margin is what separates real loss from a pixel that
#                was already at 252 and would have rounded up anyway.
#   `crushed`    the same at the black end: at least FLATTEN_MARGIN
#                levels above black, rendering as exactly 0.
#   The depths are the softer reading: how far outside [0, 255] the
#   unclipped value went, before to_u8 pulled it back.
FLATTEN_MARGIN     = 15
CLIP_REPORT_DEPTHS = (0.5, 2.0, 4.0, 8.0)

# --- Geometry (must match step 04, step 1 and step 2) ---
IMAGE_SIZE   = 512     # manipulation and output size
FEATURE_SIZE = 256     # measurement size
CLIP_PX      = 336

# --- Manipulation parameters (as 09c / steps 1 and 2) ---
# MANIP_ORDER
#   Fog first, then colour. The order matters twice over: the airlight
#   comes from the unmodified square so the veil is constant across a
#   row, and the in-gamut chroma ceiling for k > 1 is read off the
#   image AS IT STANDS AFTER THE HAZE, because haze moves lightness and
#   hue and so moves the ceiling.
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

# --- Colour ladder monotonicity (CHECK 6) ---
#
# MIN_COLOUR_STEP
#   The smallest sat_mean step between ADJACENT colour levels that
#   still counts as a step, in the feature's own units. Below this the
#   two cells are the same picture as far as a participant is
#   concerned, and a pair built from them is a trial with no stimulus.
#
#   1.0 is deliberately loose. The point is to find the scenes where a
#   step has COLLAPSED -- gamut clipping at the vivid end, or haze
#   having already taken the colour out at the deep fog rungs -- not to
#   police the spacing, which is step 1's job and is set on the corpus
#   median rather than per scene. For scale: the calibrated gap between
#   p65 and p75 is 12.9 sat_mean units on the median image, and
#   MIN_DSAT_SAT_ONLY (what a pair trial needs) is 17.0.
MIN_COLOUR_STEP = 1.0

# COLOUR_MONOTONE_MAX_BAD
#   Share of (scene, fog rung) ladders allowed to be non-monotone or to
#   collapse a step before the check reports rather than passes. Not
#   zero: a ladder at the deepest fog rung of a nearly colourless scene
#   has almost nothing left to scale, and that is the manipulation
#   working, not a fault.
COLOUR_MONOTONE_MAX_BAD = 0.05

# --- Reproduction of the earlier build (CHECK 8) ---
#
# COMPARE_DIR
#   An earlier build to reproduce. Every cell the two share -- same
#   scene, same colour level, same fog level -- is compared. Set to
#   None to skip.
#
#   The default points at the build the experiment currently uses. The
#   25 cells per scene it has are a subset of this build's 45, so the
#   comparison covers 47 x 25 = 1,175 images and says nothing about the
#   20 that are new. That is the point: the new cells cannot be
#   reproduced, and the old ones must be.
COMPARE_DIR = "data/stimulus_set_v4_ec_haze_eq_2"

# COMPARE_PARAM_TOL
#   How close a ladder parameter has to be to the earlier build's to
#   count as the same rung. Tight: these are the same numbers from the
#   same calibration, so anything above float noise means the ladder
#   moved, and then a byte difference is EXPECTED rather than a fault.
#   Reporting the parameters first is what makes the check readable.
COMPARE_PARAM_TOL = 1e-9

# --- Output images ---
#
# RENDER_MODE
#   "preview"  checks, selection and one grid sheet per scene
#   "full"     also the stimulus PNGs and the manifest
RENDER_MODE = "full"

STIMULUS_FORMAT  = "png"    # lossless; the stimuli are the experiment

# SHEET_TILE_PX
#   240, which is CONFIG.grid.max_cell_px in the experiment: the whole
#   grid sheet is then a fair preview of what a participant sees. Do
#   NOT read artefacts off these -- see the two settings below.
SHEET_TILE_PX    = 240
SHEET_QUALITY    = 92

# VIVID_SHEET_* / CONTRAST_SHEET_*
#   Two FULL-RESOLUTION diagnostics, one per new block of cells, for
#   the question the 240 px sheets cannot answer: does the edit leave
#   visible damage when you zoom in.
#
#   This is a lesson from step 1, where a contact sheet rendered at
#   221 px made a real contrast change look like no change at all, and
#   then a second attempt at nominally 512 px came out at 314 px
#   because tight_layout shrank the axes. Both are built here with PIL
#   at exactly IMAGE_SIZE per cell, where there is no layout engine to
#   get in the way and the number is simply the number.
#
#   N = 0 means every scene. The files are large (a 7-cell vivid sheet
#   is about 3,600 px wide), which is the point.
VIVID_SHEET_N     = 8       # scenes, spread across the brightness range
VIVID_SHEET_PX    = IMAGE_SIZE
CONTRAST_SHEET_N  = 8
CONTRAST_SHEET_PX = IMAGE_SIZE

# --- Checks and parallelism ---
FEATURE_CHECK_N    = 8
FEATURE_CHECK_RTOL = 1e-4
N_WORKERS = 20              # as the step 1 run
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


def render_spec(sq, ctx, cell, fam):
    """One cell from the cell plan, whichever block it belongs to.

    The single entry point every pass goes through -- measuring, the
    contact sheets, the stimulus writer -- so the three can never drift
    apart on how a cell is made.
    """
    if cell["kind"] == "contrast_up":
        return to_u8(FUNCS[CONTRAST_UP_FAMILY](sq, cell["gain"], ctx))
    return render_cell(sq, ctx, cell["chroma"], cell["haze"], fam)


# ============================================================
# Small helpers
# ============================================================

def label_tag(name):
    """A level label safe for a filename: 'k=1' -> 'k1', 'h=0' -> 'h0'."""
    return str(name).replace("=", "").replace("/", "_").replace(" ", "")


def gain_tag(g):
    """A gain as a filename-safe integer: 1.0875 -> 'g10875'.

    Four decimal places as a fixed-width integer rather than the
    decimal itself, so the names sort, carry no second dot before the
    extension, and are unambiguous about which gain produced which
    file.
    """
    return f"g{int(round(g * 10000)):05d}"


def cell_filename(sid, cell):
    """The stimulus filename for one cell of one scene.

    The two chroma-x-haze tags come first and are UNCHANGED from the
    old build, so every cell the two builds share also shares its
    filename and CHECK 8 can compare them by name. The contrast cells
    are new, carry a third tag, and cannot collide with anything.
    """
    name = (f"{sid}__c{label_tag(cell['colour_level'])}"
            f"__f{label_tag(cell['fog_level'])}")
    if cell["kind"] == "contrast_up":
        name += f"__{gain_tag(cell['gain'])}"
    return f"{name}.{STIMULUS_FORMAT}"


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
    """The ladder: (label, parameter, target change) per colour and fog
    level, in GRID ORDER -- most vivid to least, clearest to foggiest.

    The order is not cosmetic. CHECK 6 asks whether sat_mean is
    monotone across the colour levels, which is only a question if the
    levels arrive in the order the design intends; and the contact
    sheets are read left to right.
    """
    if not os.path.isfile(LEVELS_CSV):
        raise FileNotFoundError(f"{LEVELS_CSV} not found. Run step 1.")
    L = pd.read_csv(LEVELS_CSV)

    def pick(dim, names):
        out = []
        for nm in names:
            m = L[(L.dimension == dim) & (L.level == nm)]
            if not len(m):
                have = sorted(L[L.dimension == dim].level.unique())
                raise RuntimeError(
                    f"{LEVELS_CSV} has no {dim!r} level {nm!r}.\n"
                    f"  present in that block: "
                    f"{', '.join(have) if have else '(no such block)'}\n"
                    f"  the fog family is {FOG_FAMILY!r}, which reads "
                    f"the {FOG_DIMENSION!r} block -- is step 1's "
                    "INCLUDE_HAZE_EQ on, and did it write colour_up?")
            r = m.iloc[0]
            if not np.isfinite(r.param):
                raise RuntimeError(
                    f"{dim!r} level {nm!r} has no parameter: step 1 "
                    f"recorded it as {r.status!r}. A level step 1 could "
                    "not solve cannot be rendered. Drop it from the "
                    "ladder, or re-run step 1 with a larger budget.")
            if r.status != "ok" and not ALLOW_CAPPED_LEVELS:
                got = (f", achieving {r.achieved_change:+.3f} "
                       f"({r.achieved_sd:+.3f} SD) against a target of "
                       f"{r.target_change:+.3f} "
                       f"({r.target_sd:+.3f} SD)"
                       if np.isfinite(r.get("achieved_change", np.nan))
                       else "")
                raise RuntimeError(
                    f"{dim!r} level {nm!r} is {r.status!r}{got}.\n"
                    "  Step 1 wrote a parameter for it but could not "
                    "reach the target, so the level\n  is not "
                    "calibrated: its spacing against its neighbours is "
                    "whatever the cap left,\n  not the percentile step "
                    "the design asked for. Drop it from the ladder, or "
                    "set\n  ALLOW_CAPPED_LEVELS = True and say so in "
                    "the documentation.")
            out.append((nm, float(r.param), float(r.target_change)))
        return out

    # vivid first, then the original, then the drained levels: the
    # order the grid screen shows them in
    colour = pick("colour_up", COLOUR_UP_LEVELS)[::-1]
    if INCLUDE_UNMANIPULATED_COLOUR:
        colour = colour + [(COLOUR_IDENTITY_LABEL, 1.0, 0.0)]
    colour = colour + pick("colour", COLOUR_LEVELS)

    fog = pick(FOG_DIMENSION, FOG_LEVELS)
    if INCLUDE_UNMANIPULATED_FOG:
        fog = [(FOG_IDENTITY_LABEL, 0.0, 0.0)] + fog

    print(f"  ladder: {len(colour)} colour x {len(fog)} fog "
          f"({FOG_FAMILY})")
    print("    colour " + ", ".join(f"{n}={p:.4f}" for n, p, _ in colour))
    print("    fog    " + ", ".join(f"{n}={p:.4f}" for n, p, _ in fog))
    print("    contrast " + ", ".join(f"{g:.4f}"
                                      for g in CONTRAST_UP_GAINS)
          + f"  ({CONTRAST_UP_FAMILY}, grid only)")
    report.append(("ladder",
                   f"{len(colour)} x {len(fog)} + "
                   f"{len(CONTRAST_UP_GAINS)} contrast, {FOG_FAMILY}"))
    return colour, fog


def build_cells(colour, fog, out_dir, report):
    """The cell plan: every image one scene produces, what renders it,
    and what it is for.

    ONE table, written to disk and used by the measuring pass, the
    contact sheets and the stimulus writer alike. The old script
    rebuilt the colour-x-fog loop in each of the three, which was
    harmless while every cell was the same kind of thing and is not
    once there are three kinds.

    `use` is the column step 4, step 5 and step 7 filter on:

      pair   the pair grid only
      both   in the pair grid AND on a grid screen
      grid   a grid screen only -- never a pair trial

    The two grid screens are a cross through the plan: the fog screen
    holds colour at the identity and walks down the fog rungs, the
    colour screen holds fog at the identity and walks across the
    colour levels. So a cell is on a grid screen exactly when one of
    its two levels is the identity. The pair grid is everything with
    REAL haze, at every colour level -- the h = 0 row is left out of
    it on purpose, because pairs in the old build were always two
    manipulated images and keeping that true is what lets step 4
    reproduce the old pair table.
    """
    cells = []
    for cn, k, _ in colour:
        for fn, h, _ in fog:
            in_pair = (fn != FOG_IDENTITY_LABEL
                       or PAIR_INCLUDES_UNMANIPULATED_FOG)
            in_grid = (cn == COLOUR_IDENTITY_LABEL
                       or fn == FOG_IDENTITY_LABEL)
            cells.append({
                "kind": "grid", "colour_level": cn, "fog_level": fn,
                "chroma": k, "haze": h, "gain": 1.0,
                "contrast_level": "",
                "use": ("both" if (in_pair and in_grid)
                        else "pair" if in_pair else "grid"),
            })
    for g in CONTRAST_UP_GAINS:
        cells.append({
            "kind": "contrast_up",
            "colour_level": CONTRAST_UP_AT_COLOUR,
            "fog_level": CONTRAST_UP_AT_FOG,
            "chroma": 1.0, "haze": 0.0, "gain": float(g),
            "contrast_level": gain_tag(g),
            "use": "grid",
        })

    # a cell is identified by (colour_level, fog_level, contrast_level)
    # everywhere downstream, so those three must be unique
    key = [(c["colour_level"], c["fog_level"], c["contrast_level"])
           for c in cells]
    if len(set(key)) != len(key):
        dup = [k for k in set(key) if key.count(k) > 1]
        raise RuntimeError(
            "The cell plan has duplicate cells, so the manifest could "
            f"not be keyed: {dup}. The contrast column sits at "
            f"({CONTRAST_UP_AT_COLOUR!r}, {CONTRAST_UP_AT_FOG!r}) and "
            "is distinguished from the grid cell there by its "
            "contrast_level -- check gain_tag.")

    plan = pd.DataFrame(cells)
    plan.to_csv(os.path.join(out_dir, "cell_plan.csv"), index=False)
    n = plan.use.value_counts()
    print(f"  cell plan: {len(plan)} cells per scene  "
          f"(pair {n.get('pair', 0)}, both {n.get('both', 0)}, "
          f"grid {n.get('grid', 0)})")
    report.append(("cells per scene",
                   f"{len(plan)} (pair {n.get('pair', 0)}, both "
                   f"{n.get('both', 0)}, grid {n.get('grid', 0)})"))
    return cells


def load_envelope(report):
    """Corpus 1-99% range per feature, for CHECK 2."""
    if not os.path.isfile(STATS_CSV):
        print(f"  {STATS_CSV} not found; the envelope check is skipped.")
        report.append(("check 2 envelope", "skipped (no stats)"))
        return None
    S = pd.read_csv(STATS_CSV)

    # The guard this replaced was `if "population" in S.columns:`, which
    # silently accepted a stats file without that column and then read
    # EVERY row as though it were the population asked for. On a file
    # holding three populations stacked, set_index("feature") keeps
    # whichever row happens to come last per feature -- a reference
    # that belongs to no population in particular. That is the fault
    # that cost step 2 a round and published a sweep table describing a
    # threshold the selection never used. It raises now.
    if "population" not in S.columns:
        raise RuntimeError(
            f"{STATS_CSV} has no 'population' column, so the envelope "
            f"cannot be read from the {STATS_POPULATION!r} population.\n"
            "  Re-run step 1 (calibrate_levels_v4_corrected.py): it "
            "writes one row block per\n  population, labelled. An "
            "unlabelled stats file was the step 2 fault -- without "
            "the\n  column, every row reads as the population asked "
            "for and the envelope is wrong\n  by a silent amount.")
    sub = S[S.population == STATS_POPULATION]
    if not len(sub):
        avail = sorted(S.population.unique())
        raise RuntimeError(
            f"{STATS_CSV} has no population {STATS_POPULATION!r}. "
            f"Available: {avail}")
    S = sub.set_index("feature")
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


def _damage(sq, x, img):
    """What a manipulation cost, for one rendered cell.

    `flattened` and `crushed` are the strict reading: a pixel at least
    FLATTEN_MARGIN levels inside the range in the original that comes
    out exactly at the rail. The margin is what separates real loss
    from a pixel that was already at 252 and would have rounded up
    anyway.

    The depths are the softer reading: how far outside [0, 255] the
    UNCLIPPED value went before to_u8 pulled it back. A pixel 0.4
    levels past white loses nothing a viewer could see; one 8 levels
    past it has had real detail taken. `inside` restricts both ends to
    pixels that were not already at the rail in the original, because
    a pixel that starts at 255 cannot be flattened by anything.
    """
    inside = (sq > 0) & (sq < 255)
    rec = {
        "flattened": float((((sq <= 255 - FLATTEN_MARGIN)
                             & (img == 255)).any(axis=2).mean())),
        "crushed": float((((sq >= FLATTEN_MARGIN)
                           & (img == 0)).any(axis=2).mean())),
    }
    for dep in CLIP_REPORT_DEPTHS:
        rec[f"below0_{dep:g}lv"] = float(
            ((x < -dep) & inside).any(axis=2).mean())
        rec[f"above255_{dep:g}lv"] = float(
            ((x > 255.0 + dep) & inside).any(axis=2).mean())
    return rec


def _grid_task(row):
    """Render and measure every cell of one scene, over the cell plan.

    All features, not just the two the pairs use: the envelope check
    needs the luminance set, and the manifest carries the rest.

    The contrast cells are measured TWICE over, in a sense: once for
    their features like any other cell, and once for what the gain
    cost, which is the unclipped array `x` against the original. That
    second measurement is why the renderer is inlined for them here
    rather than going through render_spec -- render_spec returns the
    clipped uint8 image, and by then the evidence is gone.
    """
    sid, path = row
    cells, fam = W["cells"], W["fam"]
    try:
        with Image.open(path) as im:
            sq = np.asarray(central_square(im.convert("RGB"), IMAGE_SIZE),
                            np.uint8)
    except Exception as exc:
        return sid, None, f"{type(exc).__name__}: {exc}"
    ctx = ctx_of(sq)
    base = measure(sq)
    out = []
    for cell in cells:
        if cell["kind"] == "contrast_up":
            x = FUNCS[CONTRAST_UP_FAMILY](sq, cell["gain"], ctx)
            img = to_u8(x)
            dmg = _damage(sq, x, img)
        else:
            img = render_spec(sq, ctx, cell, fam)
            dmg = {}
        v = measure(img)
        out.append({"scene_id": sid,
                    "colour_level": cell["colour_level"],
                    "fog_level": cell["fog_level"],
                    "contrast_level": cell["contrast_level"],
                    "use": cell["use"], "kind": cell["kind"],
                    "chroma": cell["chroma"], "haze": cell["haze"],
                    "contrast_gain": cell["gain"],
                    "filename": cell_filename(sid, cell),
                    **v, **dmg})
    return sid, (base, out), None


def measure_grid(scenes, cells, report):
    """Build every scene's grid and measure every cell."""
    print("\n" + "=" * 60)
    print("MEASURING THE GRID")
    print("=" * 60)
    n_cells = len(cells)
    print(f"  {len(scenes)} scenes x {n_cells} cells = "
          f"{len(scenes) * n_cells:,} measurements")

    payload = {"cells": cells, "fam": FOG_FAMILY}
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
        sub = grid.loc[out, ["scene_id", "colour_level", "fog_level",
                             "contrast_level", "use"]]
        excess = np.where(v[out] > hi, v[out] - hi, v[out] - lo) / sd
        rows.append(pd.DataFrame({
            "feature": f,
            "scene_id": sub.scene_id.to_numpy(),
            "colour_level": sub.colour_level.to_numpy(),
            "fog_level": sub.fog_level.to_numpy(),
            "contrast_level": sub.contrast_level.to_numpy(),
            "use": sub.use.to_numpy(),
            "value": v[out],
            "corpus_lo": lo, "corpus_hi": hi,
            "outside_by_sd": excess,
        }))

    viol = (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["feature", "scene_id",
                                       "colour_level", "fog_level",
                                       "contrast_level", "use",
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
        # which block of cells it is: a pair cell outside the corpus
        # range is a stimulus the regression will use, a grid-only cell
        # outside it is a picture on a descriptive screen
        print("\n  by what the cell is for:")
        print((viol.use.value_counts().rename("cells")
               .to_frame().to_string()))
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


def pair_cells(grid):
    """The cells a pair trial may be built from.

    FAULT FIXED HERE. The old script handed every cell to the pair
    classifier, which was harmless while every cell was a manipulated
    chroma-x-haze cell. It is not harmless now: the h = 0 row and the
    three contrast cells would enter the pair inventory, and then

      - every scene would gain a block of pairs against its own
        unmanipulated image, which the design does not use;
      - the contrast cells, whose fog feature moves UP, would be
        classified as `conflict` and `together` pairs by a rule written
        for a feature that only moves down;
      - step 4 would stop reproducing the old pair table, and the
        reason would be invisible.

    So the pair checks see `use` in (pair, both) and nothing else. The
    grid screens are descriptive and are analysed separately.
    """
    return grid[grid.use.isin(("pair", "both"))].copy()


def check_affordance(grid, out_dir, report):
    """CHECK 4: count usable pairs of each type, per scene.

    Conflict pairs are the binding constraint in practice: they need an
    image that is foggier AND more saturated, which the physics
    resists.

    Run on the PAIR cells only -- see pair_cells.
    """
    print("\n" + "=" * 60)
    print("TRIAL-TYPE AFFORDANCE  (CHECK 4)")
    print("=" * 60)

    grid = pair_cells(grid)
    print(f"  {len(grid):,} pair cells "
          f"({grid.scene_id.nunique()} scenes x "
          f"{len(grid) // max(grid.scene_id.nunique(), 1)}); the "
          "h = 0 row and the\n  contrast column are grid-only and are "
          "excluded.")

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
# CHECK 6: the colour ladder, in every scene
# ============================================================

def check_colour_ladder(grid, colour, out_dir, report):
    """CHECK 6: is sat_mean monotone across the colour levels in every
    scene, at every fog rung, and does any step collapse?

    Step 1 calibrates the ladder on the corpus MEDIAN. That makes it a
    statement about the typical image and not a promise about any
    particular one, and two things can break it on a particular scene:

      the vivid end  chroma above 1 soft-saturates toward the per-pixel
                     gamut ceiling, so a scene already near the ceiling
                     has little room and p65 and p75 can land on top of
                     one another;

      the deep fog   haze drains colour as a side effect, so at the
                     foggiest rung there may be almost no chroma left
                     to scale and the whole ladder compresses.

    A collapsed step is not a cosmetic problem. A pair built from two
    cells that measure the same saturation is a trial with no stimulus
    difference, which the analysis will score as a coin flip and treat
    as evidence.

    Reported at every rung rather than at h = 0 only, because the
    second failure mode lives entirely at the deep rungs and would be
    invisible in a check run on the unmanipulated row.
    """
    print("\n" + "=" * 60)
    print("COLOUR LADDER  (CHECK 6)")
    print("=" * 60)

    order = [cn for cn, _, _ in colour]
    rank = {cn: i for i, cn in enumerate(order)}
    g = grid[grid.kind == "grid"].copy()
    g["rank"] = g.colour_level.map(rank)

    rows = []
    for (sid, fl), sub_ in g.groupby(["scene_id", "fog_level"]):
        s = sub_.sort_values("rank")
        v = s[FEAT_SAT].to_numpy(float)
        if len(v) < 2:
            continue
        # order is vivid -> drained, so a sound ladder DECREASES
        d = np.diff(v)
        rows.append({
            "scene_id": sid, "fog_level": fl,
            "sat_top": v[0], "sat_bottom": v[-1],
            "span": v[0] - v[-1],
            "min_step": float(-d.max()),      # the weakest step
            "max_step": float(-d.min()),
            "monotone": bool((d < 0).all()),
            "collapsed": bool((-d < MIN_COLOUR_STEP).any()),
            "weakest_between": order[int(np.argmax(d))] + " -> "
                               + order[int(np.argmax(d)) + 1],
        })

    lad = pd.DataFrame(rows)
    lad["ok"] = lad.monotone & ~lad.collapsed
    lad.to_csv(os.path.join(out_dir, "colour_ladder.csv"), index=False)

    bad = lad[~lad.ok]
    frac = len(bad) / float(max(len(lad), 1))
    print(f"  {len(lad)} ladders checked "
          f"({grid.scene_id.nunique()} scenes x "
          f"{lad.fog_level.nunique()} fog rungs), "
          f"{len(order)} colour levels each")
    print(f"  order: {' > '.join(order)}  (sat_mean should fall "
          "left to right)")
    print()
    print("  weakest step in the ladder, by fog rung:")
    print(lad.groupby("fog_level")
          .agg(n=("min_step", "size"),
               worst=("min_step", "min"),
               median=("min_step", "median"),
               not_monotone=("monotone", lambda s: int((~s).sum())),
               collapsed=("collapsed", "sum"))
          .round(2).to_string())

    if len(bad):
        print(f"\n  {len(bad)} ladder(s) non-monotone or collapsed "
              f"({frac:.1%}):")
        print(bad[["scene_id", "fog_level", "min_step",
                   "weakest_between", "monotone"]]
              .head(12).round(2).to_string(index=False))
        if len(bad) > 12:
            print(f"    ... and {len(bad) - 12} more; see "
                  "colour_ladder.csv")

    ok = frac <= COLOUR_MONOTONE_MAX_BAD
    print(f"\n  RESULT: {'PASS' if ok else 'NOTE'} "
          f"({frac:.1%} of ladders affected, threshold "
          f"{COLOUR_MONOTONE_MAX_BAD:.0%}, step floor "
          f"{MIN_COLOUR_STEP})")
    if not ok:
        print("  Read WHERE before acting. Collapses confined to the "
              "deepest fog rung are the manipulation working -- haze "
              "has taken the colour already -- and step 4 will not "
              "build a usable pair from them anyway, because "
              f"MIN_DSAT_SAT_ONLY is {MIN_DSAT_SAT_ONLY}. Collapses at "
              "h = 0, or between p65 and p75, are the gamut ceiling "
              "and mean the vivid end is too close to it on this "
              "material.")
    report.append(("check 6 colour ladder",
                   f"{'PASS' if ok else 'NOTE'} "
                   f"({len(bad)}/{len(lad)} affected)"))
    return lad, ok


# ============================================================
# CHECK 7: what the contrast column costs
# ============================================================

def check_contrast_damage(grid, out_dir, report):
    """CHECK 7: per scene and per gain, what the contrast gain costs.

    REPORTED NUMBERS, NOT A THRESHOLD. There is no gain at which this
    check fails, because the decision it feeds is not a pass/fail one:
    step 7 chooses the gain and the 24 grid scenes together, from these
    figures and from a mock-up of a real grid screen at 240 px. Writing
    a threshold here would be inventing a criterion the design does not
    have.

    What the figures were at step 1, on the same 47 scenes, so a shift
    here is visible:

      gain    scenes with no measured damage    worst flattened
      1.0580                47 / 47                    0.00%
      1.0750                45 / 47                    0.17%
      1.0875                40 / 47                    0.75%
      1.1100                22 / 47                    2.48%

    The 1.110 row is the one that matters for step 7: 22 undamaged
    scenes is fewer than the 24 the two grid blocks need, so a
    zero-damage rule does not fit and the selection needs a stated
    tolerance instead -- at 0.25% there are 28 candidates, spanning the
    full brightness range of the set rather than the narrow 81-148 band
    the strict rule leaves. 1.150 was never swept and has no row.
    """
    print("\n" + "=" * 60)
    print("CONTRAST-UP DAMAGE  (CHECK 7)")
    print("=" * 60)

    c = grid[grid.kind == "contrast_up"].copy()
    if not len(c):
        print("  no contrast cells in the plan; skipped.")
        report.append(("check 7 contrast damage", "skipped"))
        return pd.DataFrame()

    base = (grid[(grid.kind == "grid")
                 & (grid.colour_level == COLOUR_IDENTITY_LABEL)
                 & (grid.fog_level == FOG_IDENTITY_LABEL)]
            .set_index("scene_id"))
    c["base_fog"] = c.scene_id.map(base[FEAT_FOG])
    c["base_lum"] = c.scene_id.map(base["lum_mean"])
    c["d_fog"] = c[FEAT_FOG] - c.base_fog
    c["damage"] = c.flattened + c.crushed

    dep = [f"{s}_{d:g}lv" for d in CLIP_REPORT_DEPTHS
           for s in ("below0", "above255")]
    keep = (["scene_id", "category", "contrast_gain", "contrast_level",
             "base_lum", "base_fog", FEAT_FOG, "d_fog",
             "flattened", "crushed", "damage"] + dep)
    out = c[[k for k in keep if k in c.columns]].sort_values(
        ["contrast_gain", "damage"], ascending=[True, False])
    out.to_csv(os.path.join(out_dir, "contrast_up_damage.csv"),
               index=False)

    sd = None
    if os.path.isfile(STATS_CSV):
        try:
            S = pd.read_csv(STATS_CSV)
            S = S[S.population == STATS_POPULATION].set_index("feature")
            sd = float(S.at[FEAT_FOG, "sd"])
        except Exception:
            sd = None

    print(f"  {c.scene_id.nunique()} scenes x "
          f"{c.contrast_gain.nunique()} gains")
    print(f"  the contrast-up column is `use = grid`: it never enters "
          "a pair trial.\n")
    tab = (c.groupby("contrast_gain")
           .agg(clean=("damage", lambda s: int((s <= 0).sum())),
                n=("damage", "size"),
                worst_flat=("flattened", "max"),
                worst_crush=("crushed", "max"),
                median_d_fog=("d_fog", "median"))
           .reset_index())
    tab["clean"] = (tab.clean.astype(str) + " / "
                    + tab.n.astype(str))
    # NOT `for c in ...`: `c` is the contrast-cell frame in this scope
    for col in ("worst_flat", "worst_crush"):
        tab[col] = (tab[col] * 100).map("{:.2f}%".format)
    if sd:
        tab["median_+SD"] = (tab.median_d_fog / sd).map("{:+.3f}".format)
    tab["median_d_fog"] = tab.median_d_fog.map("{:+.3f}".format)
    print(tab.drop(columns=["n"]).to_string(index=False))

    for g in sorted(c.contrast_gain.unique()):
        # only scenes that actually took damage: nlargest(3) would
        # otherwise pad the list with clean scenes under the heading
        # "worst", which reads as though they were damaged too
        s = c[(c.contrast_gain == g) & (c.damage > 0)].nlargest(
            3, "damage")
        if not len(s):
            print(f"\n  gain {g:.4f}: no scene takes measured damage.")
            continue
        print(f"\n  worst scenes at gain {g:.4f} "
              f"({int((c[c.contrast_gain == g].damage > 0).sum())} "
              "of "
              f"{int((c.contrast_gain == g).sum())} damaged):")
        for r in s.itertuples():
            print(f"    {r.scene_id:38s} flattened "
                  f"{r.flattened * 100:5.2f}%  crushed "
                  f"{r.crushed * 100:5.2f}%  (mean grey "
                  f"{r.base_lum:.0f})")

    print("\n  How many scenes are available at each gain, for a "
          "damage tolerance:")
    for tol in (0.0, 0.0025, 0.005, 0.01, 0.02):
        avail = {g: int((c[c.contrast_gain == g].damage <= tol).sum())
                 for g in sorted(c.contrast_gain.unique())}
        mark = ""
        if min(avail.values()) >= 24:
            mark = "   <- 24 available at every gain"
        print(f"    tolerance {tol * 100:5.2f}%  "
              + "  ".join(f"{g:.4f}: {n:2d}" for g, n in avail.items())
              + mark)
    print("\n  24 is what two grid blocks of 12 distinct scenes need. "
          "Step 7 picks\n  the gain and the scenes together; nothing "
          "is decided here.")

    report.append(("check 7 contrast damage",
                   "reported (no threshold)"))
    return out


# ============================================================
# CHECK 8: does this build reproduce the last one?
# ============================================================

def check_reproduction(manifest, out_dir, report):
    """CHECK 8: every cell this build shares with COMPARE_DIR, compared.

    Two comparisons, in this order, because the order is what makes a
    failure readable:

      1. THE LADDER PARAMETERS. The earlier manifest carries the chroma
         and haze that produced each of its images. If those differ
         from this build's, the images SHOULD differ, and nothing about
         the renderer has been tested. A bare hash comparison cannot
         tell the two apart, and would report a moved ladder as a
         broken renderer.

      2. THE PNG BYTES, then, where they differ, the largest pixel
         difference and the share of pixels affected. A difference of
         one level on a handful of pixels is a rounding boundary; a
         difference of thirty on most of the frame is a different
         image.

    What this can and cannot cover: the earlier build has 25 cells per
    scene and this one has 45, so the comparison covers the 25 and says
    nothing about the 20 that are new. The new cells cannot be
    reproduced -- there is nothing to reproduce them against -- and the
    old ones must be, because the experiment has already been built
    around them.
    """
    print("\n" + "=" * 60)
    print("REPRODUCTION  (CHECK 8)")
    print("=" * 60)
    if not COMPARE_DIR:
        print("  COMPARE_DIR is None; skipped.")
        report.append(("check 8 reproduction", "skipped (not asked)"))
        return pd.DataFrame(), True

    ref_man = os.path.join(COMPARE_DIR, "stimulus_manifest.csv")
    ref_img = os.path.join(COMPARE_DIR, "stimuli")
    if not os.path.isfile(ref_man):
        print(f"  {ref_man} not found; skipped.")
        report.append(("check 8 reproduction", "skipped (no manifest)"))
        return pd.DataFrame(), True

    R = pd.read_csv(ref_man)
    print(f"  reference: {COMPARE_DIR}")
    print(f"    {len(R):,} images, {R.scene_id.nunique()} scenes, "
          f"family {R.fog_family.iloc[0] if 'fog_family' in R else '?'}")
    if "fog_family" in R.columns and R.fog_family.iloc[0] != FOG_FAMILY:
        print(f"    NOTE: the reference is the "
              f"{R.fog_family.iloc[0]!r} family and this build is "
              f"{FOG_FAMILY!r}.")
        print("    Nothing can match across families; this is the "
              "wrong reference for this run.")

    key = ["scene_id", "colour_level", "fog_level"]
    mine = manifest[manifest.kind == "grid"].copy()
    both = mine.merge(R[key + ["chroma", "haze", "filename"]],
                      on=key, how="inner", suffixes=("", "_ref"))
    print(f"  {len(both):,} cells in common "
          f"({both.scene_id.nunique()} scenes x "
          f"{len(both) // max(both.scene_id.nunique(), 1)} cells)")
    if not len(both):
        print("  nothing in common; skipped.")
        report.append(("check 8 reproduction", "skipped (no overlap)"))
        return pd.DataFrame(), True

    # ---- 1. the ladder parameters ----
    both["d_chroma"] = (both.chroma - both.chroma_ref).abs()
    both["d_haze"] = (both.haze - both.haze_ref).abs()
    worst_p = float(max(both.d_chroma.max(), both.d_haze.max()))
    same_par = worst_p <= COMPARE_PARAM_TOL
    print(f"\n  1. ladder parameters: worst difference {worst_p:.3e} "
          f"-> {'SAME' if same_par else 'MOVED'}")
    if not same_par:
        moved = (both[both.d_chroma + both.d_haze > COMPARE_PARAM_TOL]
                 .groupby(["colour_level", "fog_level"])
                 .agg(chroma=("chroma", "first"),
                      chroma_ref=("chroma_ref", "first"),
                      haze=("haze", "first"),
                      haze_ref=("haze_ref", "first")))
        print(moved.round(6).to_string())
        print("\n  The ladder moved, so the images are EXPECTED to "
              "differ and the byte\n  comparison below tests nothing "
              "about the renderer. Either step 1 was\n  re-run with "
              "different settings, or COMPARE_DIR is the wrong build.")

    # ---- 2. the bytes ----
    if not os.path.isdir(ref_img):
        print(f"\n  2. {ref_img} not found; bytes not compared.")
        print("     The parameter comparison above still stands, and "
              "is the half that\n     says whether this is even the "
              "right reference.")
        report.append(("check 8 reproduction",
                       f"params {'SAME' if same_par else 'MOVED'}, "
                       "no images to compare"))
        # a `status` column either way: write_report groups on it, and
        # an early return without one is how this function would crash
        # on exactly the run where something was already wrong
        both = both.assign(status="not compared",
                           max_abs_level=np.nan,
                           share_differing=np.nan)
        return both, same_par

    import hashlib
    rows = []
    for r in both.itertuples():
        a = os.path.join(out_dir, "stimuli", r.filename)
        b = os.path.join(ref_img, r.filename_ref)
        rec = {"scene_id": r.scene_id, "colour_level": r.colour_level,
               "fog_level": r.fog_level, "filename": r.filename,
               "d_chroma": r.d_chroma, "d_haze": r.d_haze}
        if not (os.path.isfile(a) and os.path.isfile(b)):
            rec.update(status="missing", max_abs_level=np.nan,
                       share_differing=np.nan)
            rows.append(rec)
            continue
        ha = hashlib.md5(open(a, "rb").read()).hexdigest()
        hb = hashlib.md5(open(b, "rb").read()).hexdigest()
        if ha == hb:
            rec.update(status="identical", max_abs_level=0.0,
                       share_differing=0.0)
        else:
            with Image.open(a) as ia, Image.open(b) as ib:
                A = np.asarray(ia.convert("RGB"), np.int16)
                B = np.asarray(ib.convert("RGB"), np.int16)
            if A.shape != B.shape:
                rec.update(status="shape", max_abs_level=np.nan,
                           share_differing=np.nan)
            else:
                d = np.abs(A - B)
                rec.update(status="differs",
                           max_abs_level=float(d.max()),
                           share_differing=float(
                               (d > 0).any(axis=2).mean()))
        rows.append(rec)

    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(out_dir, "reproduction.csv"), index=False)
    n = rep.status.value_counts()
    print(f"\n  2. PNG bytes, {len(rep):,} shared images:")
    for k_, v_ in n.items():
        print(f"       {k_:10s} {v_:,}")

    diff = rep[rep.status == "differs"]
    if len(diff):
        print(f"\n     worst pixel difference "
              f"{diff.max_abs_level.max():.0f} level(s); "
              f"median share of pixels differing "
              f"{diff.share_differing.median():.1%}")
        print(diff.nlargest(6, "max_abs_level")[
            ["filename", "max_abs_level", "share_differing"]]
            .to_string(index=False))

    # The verdict has to carry BOTH halves. An earlier version of this
    # function reported only the bytes, and on a test where the ladder
    # had moved it printed "PASS -- every shared image byte-identical"
    # directly under "the ladder moved, so this tests nothing". A check
    # that can say PASS beside its own statement that it is meaningless
    # is worse than no check: it is the line someone quotes six months
    # later.
    same_px = bool(len(rep)) and bool((rep.status == "identical").all())
    if not same_par:
        verdict, ok = "INCONCLUSIVE", False
    elif same_px:
        verdict, ok = "PASS", True
    else:
        verdict, ok = "FAIL", False
    print(f"\n  RESULT: {verdict}")
    if verdict == "PASS":
        print("  Same ladder, and every shared image byte-identical: "
              "the twenty new\n  cells per scene are ADDITIONS to the "
              "existing set, not a new set. Trial\n  lists and data "
              "already collected on the shared cells stay valid.")
    elif verdict == "INCONCLUSIVE":
        print("  The ladder is not the one that built COMPARE_DIR, so "
              "nothing here tests\n  the renderer, whatever the byte "
              "counts above say. Point COMPARE_DIR at the\n  build "
              "this ladder produced, or accept that the set is new and "
              "say so in the\n  documentation -- in which case the "
              "trial lists have to be rebuilt too.")
    if verdict == "FAIL":
        print("  The ladder is unchanged, so this is the RENDERER. The "
              "edits in this\n  build that could do it: the cmax fix "
              "(k > 1 only -- it cannot touch a\n  shared cell), and "
              "lin_to_srgb_f losing its clamp (to_u8 clips to the\n"
              "  same bytes either way, so it should not). A one-level "
              "difference on a\n  few pixels is a rounding boundary; "
              "anything larger is a real change and\n  the set should "
              "not be shipped until it is explained.")
    report.append(("check 8 reproduction",
                   f"{verdict} ({int(n.get('identical', 0))}/{len(rep)}"
                   " identical, params "
                   + ("SAME" if same_par else "MOVED") + ")"))
    return rep, ok


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

def _row_sheet(sq, ctx, specs, tile, title, labels, fam):
    """A one-row contact sheet at EXACTLY `tile` pixels per cell.

    Built with PIL rather than matplotlib on purpose. Step 1 lost two
    rounds to this: a sheet asked for at 512 px came out at 314 because
    tight_layout shrank the axes to fit the labels, and a real contrast
    change read as no change at all. There is no layout engine here, so
    the number is the number.
    """
    pad, title_h, head_h = 4, 22, 18
    top = title_h + head_h
    sheet = Image.new("RGB",
                      (len(specs) * (tile + pad) + pad,
                       top + tile + pad), (28, 28, 30))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), title, fill=(230, 230, 235))
    for i, (spec, lab) in enumerate(zip(specs, labels)):
        x = i * (tile + pad) + pad
        draw.text((x + 2, title_h + 3), lab, fill=(190, 190, 198))
        img = Image.fromarray(render_spec(sq, ctx, spec, fam))
        if img.size != (tile, tile):
            img = img.resize((tile, tile), Image.LANCZOS)
        sheet.paste(img, (x, top))
    return sheet


def _sheet_task(row):
    """Three contact sheets for one scene.

    grid     the whole colour x fog plan at SHEET_TILE_PX (240, the
             experiment's maximum cell size), so the sheet is a fair
             preview of what a participant will see.
    vivid    the colour ladder at FULL resolution, for the question the
             240 px sheet cannot answer: do the vivid levels look
             artificial when you zoom in.
    contrast the contrast-up gains at full resolution, same reason.

    The last two are written only for the scenes named in the payload;
    they are large files.
    """
    sid, path, out_path, vivid_path, contrast_path = row
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
    draw.text((6, 5), f"{sid}   [{FOG_FAMILY}]  colour across "
                      f"(vivid left), fog down (h = 0 top)",
              fill=(230, 230, 235))

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

    if vivid_path:
        specs = [{"kind": "grid", "chroma": k, "haze": 0.0}
                 for _, k, _ in colour]
        labs = [f"{cn}  k={k:.3f}" for cn, k, _ in colour]
        _row_sheet(sq, ctx, specs, VIVID_SHEET_PX,
                   f"{sid}   colour ladder at h = 0, full resolution "
                   f"({VIVID_SHEET_PX} px per cell)", labs, fam
                   ).save(vivid_path, quality=SHEET_QUALITY)

    if contrast_path:
        specs = ([{"kind": "grid", "chroma": 1.0, "haze": 0.0}]
                 + [{"kind": "contrast_up", "gain": g}
                    for g in CONTRAST_UP_GAINS])
        labs = (["original"]
                + [f"gain {g:.4f}" for g in CONTRAST_UP_GAINS])
        _row_sheet(sq, ctx, specs, CONTRAST_SHEET_PX,
                   f"{sid}   contrast-up gains, full resolution "
                   f"({CONTRAST_SHEET_PX} px per cell) -- "
                   "PROVISIONAL COLUMN", labs, fam
                   ).save(contrast_path, quality=SHEET_QUALITY)

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
    vivid_dir = os.path.join(out_dir, "sheets_vivid")
    con_dir = os.path.join(out_dir, "sheets_contrast")
    for d in (sheet_dir, vivid_dir, con_dir):
        os.makedirs(d, exist_ok=True)

    # which scenes get the full-resolution sheets: spread across the
    # brightness range rather than taken off the top of the list,
    # because what the contrast gain costs depends on how bright the
    # scene is and the first N of an alphabetical list is not a sample
    f = final.reset_index(drop=True)
    if "base_lum" in f.columns and f.base_lum.notna().any():
        f = f.sort_values("base_lum").reset_index(drop=True)
    pick_v = set(f.scene_id if not VIVID_SHEET_N else f.scene_id.iloc[
        np.unique(np.linspace(0, len(f) - 1,
                              min(VIVID_SHEET_N, len(f))).astype(int))])
    pick_c = set(f.scene_id if not CONTRAST_SHEET_N else f.scene_id.iloc[
        np.unique(np.linspace(0, len(f) - 1,
                              min(CONTRAST_SHEET_N, len(f))).astype(int))])

    rows = [(r.scene_id, r.path,
             os.path.join(sheet_dir, f"{r.scene_id}.jpg"),
             os.path.join(vivid_dir, f"{r.scene_id}.jpg")
             if r.scene_id in pick_v else None,
             os.path.join(con_dir, f"{r.scene_id}.jpg")
             if r.scene_id in pick_c else None)
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
          f"at {SHEET_TILE_PX} px per cell ({time.time() - t0:.0f}s)")
    print(f"  {len(pick_v)} full-resolution colour ladders in "
          f"{vivid_dir}")
    print(f"  {len(pick_c)} full-resolution contrast rows in {con_dir}")
    print("  Judge ARTEFACTS on the full-resolution sheets and "
          "VISIBILITY on the\n  240 px grid sheets -- 240 is the "
          "experiment's maximum cell size, so a\n  difference you "
          "cannot see there is a difference a participant cannot use.")
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
    """Render and save every cell of one scene, over the cell plan.

    The filename comes from cell_filename, the same function the
    measuring pass used, so the manifest's `filename` column and the
    file on disk cannot disagree.
    """
    sid, path, img_dir = row
    cells, fam = W["cells"], W["fam"]
    try:
        with Image.open(path) as im:
            sq = np.asarray(central_square(im.convert("RGB"), IMAGE_SIZE),
                            np.uint8)
    except Exception as exc:
        return sid, None, f"{type(exc).__name__}: {exc}"
    ctx = ctx_of(sq)
    out = []
    for cell in cells:
        name = cell_filename(sid, cell)
        Image.fromarray(render_spec(sq, ctx, cell, fam)).save(
            os.path.join(img_dir, name))
        out.append({"scene_id": sid,
                    "colour_level": cell["colour_level"],
                    "fog_level": cell["fog_level"],
                    "contrast_level": cell["contrast_level"],
                    "filename": name})
    return sid, out, None


def write_stimuli(final, grid, cells, out_dir, report):
    """Render and save every stimulus, and build the manifest.

    The manifest is what step 4, step 5 and step 7 read: one row per
    image, with the measured features that are the regression
    predictors and the `use` column that says which of the three may
    have it.
    """
    print("\n" + "=" * 60)
    print("WRITING STIMULI")
    print("=" * 60)
    img_dir = os.path.join(out_dir, "stimuli")
    os.makedirs(img_dir, exist_ok=True)

    rows = [(r.scene_id, r.path, img_dir) for r in final.itertuples()]
    payload = {"cells": cells, "fam": FOG_FAMILY}
    n_cells = len(cells)
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
    # the cell key is three columns now: the contrast cells share their
    # colour and fog levels with the grid's original cell and are told
    # apart by contrast_level
    key = ["scene_id", "colour_level", "fog_level", "contrast_level"]
    grid = grid.copy()
    for d in (grid, files):
        d["contrast_level"] = d["contrast_level"].fillna("")
    manifest = grid.drop(columns=["filename"]).merge(
        files, on=key, how="inner")
    if len(manifest) != len(files):
        raise RuntimeError(
            f"the manifest join lost rows: {len(files):,} files "
            f"written, {len(manifest):,} joined. The cell key is "
            f"{key} and must be unique in both tables.")
    manifest["fog_family"] = FOG_FAMILY
    manifest["source_path"] = manifest.scene_id.map(meta["path"])

    lead = ["scene_id", "category", "group", "indoor", "fog_family",
            "use", "kind", "colour_level", "fog_level",
            "contrast_level", "chroma", "haze", "contrast_gain",
            "filename", "source_path"]
    manifest = manifest[[c for c in lead if c in manifest.columns]
                        + [c for c in manifest.columns
                           if c not in lead and c != "path"]]
    # CANONICAL ROW ORDER. The measuring and writing passes both use
    # imap_unordered, so without this the manifest's scene order
    # follows WORKER COMPLETION ORDER and differs from run to run.
    # Nothing about the images depends on it -- but step 4's pair
    # table inherits the order, and step 5 shuffles that table to
    # sample trials, so a re-run with identical settings and the same
    # seed could produce a DIFFERENT TRIAL LIST. That is what made the
    # shipped trial list impossible to reproduce.
    order = {(c["colour_level"], c["fog_level"], c["contrast_level"]): i
             for i, c in enumerate(cells)}
    manifest["_cell"] = [
        order.get((r.colour_level, r.fog_level,
                   r.contrast_level if isinstance(r.contrast_level, str)
                   else ""), 1 << 20)
        for r in manifest.itertuples()]
    manifest = (manifest.sort_values(["scene_id", "_cell"],
                                     kind="stable")
                .drop(columns=["_cell"]).reset_index(drop=True))
    manifest.to_csv(os.path.join(out_dir, "stimulus_manifest.csv"),
                    index=False)
    print(f"  {len(manifest):,} stimuli in {img_dir}")
    print("  by use: " + ", ".join(
        f"{u} {n:,}" for u, n in manifest.use.value_counts().items()))
    print(f"  manifest: {os.path.join(out_dir, 'stimulus_manifest.csv')}")
    report.append(("stimuli written", f"{len(manifest):,}"))
    nu = manifest.use.value_counts()
    report.append(("manifest by use",
                   f"pair {nu.get('pair', 0):,}, both "
                   f"{nu.get('both', 0):,}, grid {nu.get('grid', 0):,}"))
    return manifest


# ============================================================
# Report
# ============================================================

def write_report(out_dir, report, scenes, final, colour, fog, cells,
                 env_viol, equiv, cov, lad, dmg, repro, t0, overall):
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
        W_("                contr. " +
           ", ".join(f"{g:.4f}" for g in CONTRAST_UP_GAINS)
           + f"   ({CONTRAST_UP_FAMILY}, grid only, PROVISIONAL)")
        W_(f"Cells/scene   : {len(cells)}")
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
        W_("2. WHAT EACH IMAGE IS FOR")
        W_("-" * 70)
        W_("  Every manifest row carries a `use` column. Step 4 and")
        W_("  step 5 must filter on it; step 7 reads the grid rows.")
        W_()
        plan = pd.DataFrame(cells)
        for u, note in (
                ("pair", "pair trials only"),
                ("both", "pair trials AND the fog grid screen"),
                ("grid", "the grid screens only -- never a pair")):
            s = plan[plan.use == u]
            W_(f"  {u:5s} {len(s):3d} cells/scene   {note}")
        W_()
        W_("  The two grid screens are a cross through the plan: the")
        W_("  fog screen holds colour at the original and walks down")
        W_("  the fog rungs, the colour screen holds fog at the")
        W_("  original and walks across the colour levels. So a cell")
        W_("  is on a grid screen exactly when one of its two levels")
        W_("  is the original.")
        W_()
        W_("  The h = 0 row is NOT in the pair set. Pairs in the")
        W_("  earlier build were always two manipulated images, and")
        W_("  keeping that true is what lets step 4 reproduce the old")
        W_("  pair table on the old levels.")
        W_()
        W_("  POSITION CONFOUND, for the write-up. The original sits")
        W_("  in a different column on the two grid screens: second")
        W_("  from the left on the fog screen (one contrast-up column")
        W_("  above it, five haze columns below), third on the colour")
        W_("  screen (two vivid columns above, four drained below).")
        W_("  So comparing WHICH COLUMN people choose across the two")
        W_("  features is partly a position effect. The shift within")
        W_("  each feature is unaffected, and that is what the")
        W_("  analysis uses. A symmetric six-column layout would fix")
        W_("  it at the cost of one fog rung and one vivid level on")
        W_("  the grid screens -- both would stay in the pair grid.")
        W_("  Step 7 may switch.")
        W_()
        W_("  Measure the normal-to-episode shift in FEATURE UNITS")
        W_("  (SD), not in grid steps: the contrast-up step is")
        W_("  +0.22 SD and the haze steps are -0.13 SD and larger, so")
        W_("  a step is not a constant amount of anything.")

        W_()
        W_("3. ENVELOPE  (CHECK 2)")
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
        W_("4. FOG FAMILIES  (CHECK 3)")
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
        W_("5. MAGNITUDE COVERAGE  (CHECK 5)")
        W_("-" * 70)
        W_("  Pair cells only: the h = 0 row and the contrast column")
        W_("  are excluded, so these counts are comparable with the")
        W_("  earlier build's on the levels the two share.")
        W_()
        if cov is None or not len(cov):
            W_("  Skipped.")
        else:
            W_(cov.to_string(index=False))

        W_()
        W_("6. COLOUR LADDER  (CHECK 6)")
        W_("-" * 70)
        W_("  Is sat_mean monotone across the colour levels in every")
        W_("  scene, at every fog rung, and does any step collapse?")
        W_("  The ladder is calibrated on the corpus MEDIAN, so it is")
        W_("  a statement about the typical image; two things break it")
        W_("  on a particular scene -- the gamut ceiling at the vivid")
        W_("  end, and haze having already drained the colour at the")
        W_("  deep rungs. A collapsed step gives a pair trial with no")
        W_("  stimulus difference, which the analysis would score as a")
        W_("  coin flip and treat as evidence.")
        W_()
        if lad is None or not len(lad):
            W_("  Skipped.")
        else:
            W_(lad.groupby("fog_level")
               .agg(n=("min_step", "size"),
                    weakest_step=("min_step", "min"),
                    median_step=("min_step", "median"),
                    not_monotone=("monotone",
                                  lambda s: int((~s).sum())),
                    collapsed=("collapsed", "sum"))
               .round(2).to_string())
            bad = lad[~lad.ok]
            W_()
            W_(f"  {len(bad)} of {len(lad)} ladders affected "
               f"(step floor {MIN_COLOUR_STEP} sat_mean units).")
            if len(bad):
                W_("  Collapses confined to the deepest fog rung are")
                W_("  the manipulation working and step 4 would not")
                W_("  build a usable pair from them anyway "
                   f"(MIN_DSAT_SAT_ONLY = {MIN_DSAT_SAT_ONLY}).")
                W_("  Collapses at h = 0, or between the two vivid")
                W_("  levels, are the gamut ceiling.")

        W_()
        W_("7. CONTRAST-UP DAMAGE  (CHECK 7)")
        W_("-" * 70)
        W_("  REPORTED NUMBERS, NOT A THRESHOLD. The column is")
        W_("  PROVISIONAL and grid-only; step 7 chooses the gain and")
        W_("  the 24 grid scenes together, from these figures and from")
        W_("  a mock-up of a real grid screen at the experiment's")
        W_("  maximum cell size (240 px, not 512).")
        W_()
        W_("  `flattened` = share of pixels at least "
           f"{FLATTEN_MARGIN} levels below")
        W_("  white in the original that render as exactly 255;")
        W_("  `crushed` = the same at the black end.")
        W_()
        if dmg is None or not len(dmg):
            W_("  Skipped.")
        else:
            t = (dmg.groupby("contrast_gain")
                 .agg(scenes=("damage", "size"),
                      clean=("damage", lambda s: int((s <= 0).sum())),
                      worst_flattened=("flattened", "max"),
                      worst_crushed=("crushed", "max"),
                      median_d_fog=("d_fog", "median")))
            W_(t.round(5).to_string())
            W_()
            W_("  Availability for the 24 grid scenes, at a tolerance")
            W_("  on flattened + crushed:")
            for tol in (0.0, 0.0025, 0.005, 0.01, 0.02):
                line = "  ".join(
                    f"{g:.4f}: {int((dmg[dmg.contrast_gain == g].damage <= tol).sum()):2d}"
                    for g in sorted(dmg.contrast_gain.unique()))
                W_(f"    {tol * 100:5.2f}%   {line}")
            W_()
            W_("  Step 1's sweep stopped at 1.110, so the highest gain")
            W_("  here is UNMEASURED and is rendered blind for the")
            W_("  mock-up. A zero-damage rule does not fit at 1.110 --")
            W_("  22 of 47 scenes, against the 24 two blocks need --")
            W_("  so the selection needs a stated tolerance.")

        W_()
        W_("8. REPRODUCTION  (CHECK 8)")
        W_("-" * 70)
        W_(f"  Reference: {COMPARE_DIR}")
        W_("  The ladder parameters are compared first, then the PNG")
        W_("  bytes. The order is what makes a failure readable: a")
        W_("  moved ladder SHOULD give different images, and a bare")
        W_("  hash cannot tell that from a broken renderer.")
        W_()
        if repro is None or not len(repro):
            W_("  Skipped or nothing in common.")
        else:
            moved = float(max(repro.d_chroma.max(),
                              repro.d_haze.max()))
            W_(f"  ladder parameters: worst difference {moved:.3e} -> "
               + ("SAME" if moved <= COMPARE_PARAM_TOL else "MOVED"))
            W_()
            W_(repro.status.value_counts().rename("images")
               .to_frame().to_string())
            if moved > COMPARE_PARAM_TOL:
                W_()
                W_("  THE LADDER MOVED, so the byte counts above test")
                W_("  nothing about the renderer and the images are")
                W_("  expected to differ. Either COMPARE_DIR is the")
                W_("  wrong build, or step 1 was re-run with different")
                W_("  settings and this set is a NEW set -- in which")
                W_("  case the trial lists have to be rebuilt too.")
            d = repro[repro.status == "differs"]
            if len(d):
                W_()
                W_(f"  worst pixel difference "
                   f"{d.max_abs_level.max():.0f} level(s) on "
                   f"{d.share_differing.median():.1%} of pixels "
                   "(median)")
                W_(d.nlargest(6, "max_abs_level")[
                    ["filename", "max_abs_level", "share_differing"]]
                   .to_string(index=False))
                W_()
                W_("  The edits in this build that could do it: the")
                W_("  cmax fix, which applies only at k > 1 and so")
                W_("  cannot touch a shared cell; and lin_to_srgb_f")
                W_("  losing its clamp, which to_u8 should make")
                W_("  invisible. A one-level difference on a few")
                W_("  pixels is a rounding boundary; anything larger")
                W_("  needs explaining before the set is used.")
            elif moved <= COMPARE_PARAM_TOL:
                W_()
                W_("  Same ladder, and every shared image is")
                W_("  byte-identical: the twenty new cells per scene")
                W_("  are ADDITIONS to the existing set rather than a")
                W_("  new set. Trial lists and data already collected")
                W_("  on the shared cells stay valid.")

        W_()
        W_("9. WHAT TO DO NEXT")
        W_("-" * 70)
        if RENDER_MODE == "preview":
            W_("  This was the PREVIEW pass: no stimulus images were")
            W_("  written, so CHECK 8 had nothing to compare. Look at")
            W_("  sheets/ at 240 px for what a participant will see,")
            W_("  and sheets_vivid/ and sheets_contrast/ at full")
            W_("  resolution for artefacts.")
            W_()
            W_("  Then set RENDER_MODE = 'full' and run again.")
        else:
            W_("  stimulus_manifest.csv is what step 4 (analyse_pairs)")
            W_("  and step 5 (build_trials) read, FILTERED ON `use`:")
            W_("  both of those want use in (pair, both), and step 7")
            W_("  wants use in (grid, both).")
            W_()
            W_("  Look at two things by eye before step 4:")
            W_("    sheets_vivid/     do p65 and p75 look artificial?")
            W_("    sheets_contrast/  is the gain visible at all, and")
            W_("                      what does it cost in the sky?")
            W_()
            W_("  Open questions this build does not settle:")
            W_("    - the contrast-up column is provisional. Step 7")
            W_("      decides it from a mock-up at 240 px. If it is")
            W_("      not distinguishable from the original there it")
            W_("      comes out: an invisible column does not reduce")
            W_("      the ceiling effect, it splits the people at the")
            W_("      ceiling between two identical-looking columns.")
            W_("    - the grid-block analysis should report how many")
            W_("      people choose the CLEAREST fog column in the")
            W_("      normal block and treat their shift as a lower")
            W_("      bound, per person. The censoring bias is")
            W_("      conservative -- it can only shrink the shift --")
            W_("      but it should be declared rather than absorbed.")
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
    cells = build_cells(colour, fog, out_dir, report)
    env = load_envelope(report)
    check_feature_code(rng, report)

    scenes = load_screened(report)
    grid, base = measure_grid(scenes, cells, report)
    grid.to_csv(os.path.join(out_dir, "grid_cells.csv"), index=False)

    env_viol, _ = check_envelope(grid, env, report)
    if len(env_viol):
        env_viol.to_csv(
            os.path.join(out_dir, "envelope_violations.csv"), index=False)

    equiv, _ = check_fog_equivalence(scenes, report)
    if len(equiv):
        equiv.to_csv(os.path.join(out_dir, "fog_equivalence.csv"),
                     index=False)

    lad, _ = check_colour_ladder(grid, colour, out_dir, report)
    dmg = check_contrast_damage(grid, out_dir, report)

    aff, pairs, ok_aff = check_affordance(grid, out_dir, report)
    aff.to_csv(os.path.join(out_dir, "pair_affordance.csv"), index=False)
    if len(pairs):
        pairs.to_csv(os.path.join(out_dir, "pairs.csv"), index=False)
    if not ok_aff:
        print("\nAffordance check failed; stopping before selection.")
        write_report(out_dir, report, scenes, aff.head(0), colour, fog,
                     cells, env_viol, equiv, None, lad, dmg, None, t0,
                     False)
        return

    cov, ok_cov = check_coverage(pairs, aff, out_dir, report)
    final = select_scenes(aff, rng, report)
    # the unmanipulated mean grey, so the full-resolution sheets can be
    # spread across the brightness range instead of taken off the top
    # of an alphabetical list -- what a contrast gain costs depends on
    # how bright the scene is
    if "lum_mean" in base.columns:
        final["base_lum"] = final.scene_id.map(base["lum_mean"])
    final.to_csv(os.path.join(out_dir, "stimulus_set.csv"), index=False)
    write_balance(final, out_dir)

    write_sheets(final, colour, fog, out_dir, report)

    manifest, repro = None, None
    if RENDER_MODE == "full":
        sel = grid[grid.scene_id.isin(set(final.scene_id))]
        manifest = write_stimuli(final, sel, cells, out_dir, report)
        repro, ok_rep = check_reproduction(manifest, out_dir, report)
    else:
        print("\n  RENDER_MODE is 'preview': no stimulus images "
              "written, so CHECK 8 has nothing to compare.")
        report.append(("stimuli written", "none (preview)"))
        report.append(("check 8 reproduction", "skipped (preview)"))
        ok_rep = True

    path = write_report(out_dir, report, scenes, final, colour, fog,
                        cells, env_viol, equiv, cov, lad, dmg, repro,
                        t0, ok_aff and ok_cov and ok_rep)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in report:
        print(f"  {k:32s} {v}")
    print(f"\n  Report : {path}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    if RENDER_MODE == "preview":
        print("\n  NEXT: look at the sheets, then re-run with "
              "RENDER_MODE = 'full'.")
    else:
        print("\n  NEXT, by eye:")
        print(f"    {os.path.join(out_dir, 'sheets_vivid')}     "
              "do p65 and p75 look artificial?")
        print(f"    {os.path.join(out_dir, 'sheets_contrast')}  "
              "is the gain visible, and what does it cost?")
        print("  Then step 4, filtering the manifest on `use`.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
