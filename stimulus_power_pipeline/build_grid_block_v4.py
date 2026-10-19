"""
build_grid_block_v4.py   (step 7b)

WHERE THIS SITS
---------------
  step 1   calibrate_levels_v4        the level parameters
  step 2   select_scenes_v4           the scene pool
  step 3   build_stimulus_set_v4      the images and the manifest   <- read here
  step 4   analyse_pairs_v4           the pair inventory
  step 5   build_trials_v4            the pair trial list
  step 6   choose_colour_condition_v4 the shipping draw
  step 7a  freeze_trial_list_v4       the frozen pair package
  step 7b  build_grid_block_v4        THIS SCRIPT
  step 8   simulate_power_v4          design analysis for the pair trials
  step 9   documentation

Nothing earlier is touched. The frozen pair package is not read, not
written, and not re-checksummed; its images are a disjoint set of cells
from the ones this script ships (see WHICH CELLS, below).

WHAT THE GRID BLOCK IS
----------------------
Two short blocks of four screens each. Every screen shows one feature at
every level at once:

    columns   the levels of one feature, in display order
    rows      three different scenes at the same levels, so that no
              single photograph drives the choice

The participant picks the COLUMN that looks most like the world looked.
It is DESCRIPTIVE ONLY: it asks WHERE on the manipulated range an
experience sits, and the registered tests in step 8 use the pair trials
alone. Nothing here is a hypothesis test.

    EPISODE BLOCK   "which column looks most like your surroundings
                    looked during the episode", with a
                    "none of these" button and a free-text box when
                    `none` is chosen.

    NORMAL BLOCK    "which column looks most like your surroundings
                    normally look", at the END of the session, with NO
                    `none` button and no free text.

The quantity of interest is the NORMAL-TO-EPISODE SHIFT, and it is
measured in FEATURE UNITS (corpus SD of sat_mean and contrast_local),
never in grid steps. Step 3's report gives the reason: the contrast-up
step is +0.22 SD while the haze steps are -0.13 SD and larger, so one
step is not a constant amount of anything.

WHY TWO SCENE SETS
------------------
The shift is a within-person comparison of two blocks, so a photograph
may not appear in both: recognising it from the earlier block would
anchor the second answer. 24 scenes are therefore chosen as 12 MATCHED
PAIRS, one member of each pair going to set A and the other to set B,
in the SAME SLOT (same screen, same row). The two sets are then swapped
between participants, so neither set is systematically the episode set.

Matching by pairs rather than by set means is deliberate. Matched set
means can hide a set that is bimodal; a pair bound at the slot level
makes the two blocks comparable row by row, and the within-pair
distance is reportable per pair rather than as one aggregate.

WHICH CELLS
-----------
Step 3 already renders everything this script needs, and `build_cells`
there marks each one:

    use = "pair"   pair trials only            30 cells/scene
    use = "both"   pair trials AND a grid screen 5 cells/scene
    use = "grid"   grid screens only           10 cells/scene

A cell is on a grid screen exactly when one of its two levels is the
original, so the two screens are a cross through the plan:

    fog screen      colour at the original, walking down the fog rungs
    colour screen   fog at the original, walking across the colour levels

This script reads `use in ("grid", "both")` and ships copies. It
renders nothing. The manipulation code, the feature code and the
filename helpers are IMPORTED from step 3's script rather than copied,
so the two can never drift; STEP3_SCRIPT below is the path.

    fog columns      h=0 (the original), p35, p25, p15, p10, p05
    colour columns   p75, k=1 (the original), p25, p10, p05, p01

Six columns on each screen, so the whole block computes ONE cell size
from the widest screen in the file. Two cells step 3 rendered are
deliberately not shown -- the contrast-up column and the colour
screen's p65 -- and UNUSED_CELL_NOTES records why; unused_cells.md
carries it out of this script and into the documentation.

THE CONTRAST-UP COLUMN, AND WHY THE FOG AXIS IS CENSORED
--------------------------------------------------------
The fog screen would ideally carry a column ABOVE the original, so
that a participant whose normal vision is clearer than the photograph
has somewhere to put it. Step 3 rendered three candidate gains
(1.0875, 1.110, 1.150) for that decision and called all three
PROVISIONAL, because such a column is only worth having if it is
distinguishable from the original AT THE SIZE THE PARTICIPANT SEES IT.

None of them is. Judged against step 3's sheets_contrast/ diagnostic,
drawn at the full 512 px, no gain was distinguishable from the
original -- and the grid shows cells at 149-239 px, so a difference
invisible at 512 px cannot appear there. The column is DROPPED.

The reason it is so weak is worth keeping: the gain holds the mean
fixed and moves only the histogram tails (about +7 on lum_p95 and -6
on lum_p05 at 1.0875), so the change lives in a small fraction of the
frame and downscaling averages it away. Haze moves the same
statistics further AND lays a coherent veil over every pixel, which is
what survives a thumbnail. Equal SD on contrast_local is not equal
visibility in the two directions, and the damage budget never bound:
1.0875 does no measured damage at all on 40 of 47 scenes.

THE CONSEQUENCE, which the analysis must carry. The fog screen now
stops at the original, so a participant whose vision is clearer than
the photograph answers on the end column. In the NORMAL-APPEARANCE
block that is the common case rather than the edge case, because most
people's normal vision is not foggy. Every fog answer on the original
column is a LOWER BOUND on that person's normal, and the
normal-to-episode shift computed from it is a lower bound on the
shift. The bias is conservative -- it can only shrink the shift -- but
it is declared rather than absorbed, and the COUNT of such answers in
each block is reported alongside. report_censoring() computes this
from the columns rather than from this paragraph, so it cannot drift
if the columns change.

POSITION CONFOUND, for the write-up
-----------------------------------
The two screens have the same NUMBER of columns but the original is
not in the same place on them: 1st of 6 on the fog screen, 2nd of 6 on
the colour screen, because fog has nothing above the original and
colour has p75. So comparing WHICH COLUMN people choose across the two
features is partly a position effect. The shift within each feature is
unaffected, and that is the quantity the analysis uses.
SYMMETRIC_COLUMNS would equalise the position, and it is off: with no
contrast-up column it would have to strip p75 from the colour screen
too, costing the one above-normal column that is actually visible.

THE TWO MODES
-------------
MODE = "mockup"    Selects the scenes, runs every check, writes the
                   scene table, the matched pairs, the headroom costs
                   and the layout arithmetic, and composes the mock-up
                   screens. Ships nothing. Run this FIRST.

MODE = "package"   Everything above, plus the uploadable package:
                   the image copies, the trial files, the preload list
                   and the interface spec. Requires CONTRAST_UP_GAIN to
                   have been decided (a value or an explicit None).

INPUTS
------
  STIMULUS_ROOT/stimulus_manifest.csv   one row per image, with `use`
  STIMULUS_ROOT/cell_plan.csv           the authoritative cell plan
  STIMULUS_ROOT/contrast_up_damage.csv  per scene x gain, flattened/crushed
  STIMULUS_ROOT/colour_ladder.csv       per scene x fog rung, sat span
  STIMULUS_ROOT/stimuli/*.png           the images themselves
  TRIAL_LIST_CSV                        the frozen pair list, for the
                                        reuse report only (optional)
  STEP3_SCRIPT                          imported, not copied

OUTPUTS
-------
  OUTPUT/grid_report.txt         what was run and what came out
  OUTPUT/scene_table.csv         every scene: baselines, spans, damage
  OUTPUT/pairs.csv               the 12 matched pairs and their distances
  OUTPUT/headroom_cost.csv       matching cost at each gain x tolerance
  OUTPUT/cell_sizes.csv          the layout arithmetic, every case
  OUTPUT/screens.csv             block x screen x row x column, resolved
  OUTPUT/pair_reuse.csv          how often each grid scene was a pair trial
  OUTPUT/unused_cells.csv        cells rendered and not shown
  OUTPUT/unused_cells.md         the same, with the reason, for the
                                 pipeline documentation
  OUTPUT/selection.png           the pool, the pairs and the two sets
  OUTPUT/headroom.png            matching cost against headroom tolerance
  OUTPUT/mockup/*.png            composed screens at the real cell size
  OUTPUT/package/                the uploadable folder (MODE = "package")

CHECKS
------
  CHECK 1  INPUTS AGREE. The manifest and the cell plan describe the
           same cells, every grid cell exists for every scene, and the
           file named in the manifest is on disk.
  CHECK 2  COLUMNS EXIST. Every column of both screens resolves to a
           file for every selected scene.
  CHECK 3  MONOTONE. contrast_local falls across the fog columns and
           sat_mean falls across the colour columns, in every selected
           row, by at least MONOTONIC_FLOOR. A row that fails would
           show its columns out of order.
  CHECK 4  SETS DISJOINT. No photograph is in both sets, every slot is
           filled, and each set is used once per block.
  CHECK 5  MATCHING. The within-pair distances and the set-mean
           differences, against MATCH_REPORT_TOL.
  CHECK 6  COPIES FAITHFUL (MODE = "package"). Every shipped image is
           re-measured and compared with the manifest's recorded
           features, and byte-compared with its source when shipped
           untouched. This is what makes "the shipped image is the
           measured image" a verified statement rather than an
           assumption.
  CHECK 7  LAYOUT. The computed cell size is at least MIN_CELL_PX in
           every window, and the number of columns is the same on both
           screens unless SYMMETRIC_COLUMNS says otherwise.

USAGE
-----
    conda activate drl2drv
    python build_grid_block_v4.py          # MODE = "mockup" first
    # look at OUTPUT/mockup/, set CONTRAST_UP_GAIN, then
    python build_grid_block_v4.py          # MODE = "package"
"""

import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

# --- Mode ---
#
# MODE
#   "mockup"   select, check, report, compose the mock-up screens.
#              Ships nothing. RUN THIS FIRST: the contrast-up gain is
#              decided from its output.
#   "package"  the above plus the uploadable package. Requires
#              CONTRAST_UP_GAIN to be a decided value or an explicit
#              None, and refuses to run while it is "undecided".
MODE = "package" # "mockup"

# --- Input ---
#
# STIMULUS_ROOT
#   Step 3's output directory for the fog family in use. The build
#   report names the directory it actually wrote; a re-run of step 3
#   lands in _2, _3, so check rather than assume.
STIMULUS_ROOT = "data/stimulus_set_v4_ext_haze_eq"

MANIFEST_CSV  = os.path.join(STIMULUS_ROOT, "stimulus_manifest.csv")
CELL_PLAN_CSV = os.path.join(STIMULUS_ROOT, "cell_plan.csv")
DAMAGE_CSV    = os.path.join(STIMULUS_ROOT, "contrast_up_damage.csv")
LADDER_CSV    = os.path.join(STIMULUS_ROOT, "colour_ladder.csv")
STIMULI_DIR   = os.path.join(STIMULUS_ROOT, "stimuli")

# TRIAL_LIST_CSV
#   The frozen pair list, read ONLY to report how often each grid scene
#   was already seen in the pair trials. Set to None to skip. The file
#   is never written and its checksums are not touched.
TRIAL_LIST_CSV = "data/frozen_trial_list_v4/trial_list.csv"

# STEP3_SCRIPT
#   Step 3's script, IMPORTED for its feature code and its filename
#   helpers. Importing rather than copying is what stops the two
#   drifting: if step 3's `measure` or `cell_filename` ever changes,
#   this script changes with it, and CHECK 6 fails loudly instead of
#   comparing against a stale copy. The script has an
#   `if __name__ == "__main__"` guard, so importing it runs nothing.
STEP3_SCRIPT = "build_stimulus_set_v4_corrected.py"

# --- Output ---
OUTPUT_ROOT = "data/grid_block_v4"

# ON_EXISTING
#   "new"    write to OUTPUT_ROOT_2, _3, ... if the directory already
#            holds a run. Nothing is ever overwritten; the script
#            prints where it actually wrote.
#   "error"  stop instead.
ON_EXISTING = "new"

# ============================================================
# The screens
# ============================================================

# SCREENS
#   The four screens of EACH block, in presentation order, as
#   (feature, column order). "forward" puts the clearest / most vivid
#   column on the LEFT; "reversed" flips it.
#
#   Each feature is asked twice, with different scenes and the column
#   order reversed, so that (a) the two answers can be compared for
#   stability and (b) answering by position -- always the leftmost or
#   always the rightmost column -- is detectable.
SCREENS = [
    ("fog",    "forward"),
    ("colour", "forward"),
    ("fog",    "reversed"),
    ("colour", "reversed"),
]

# ROWS_PER_SCREEN
#   Scenes per screen. 3 is what the layout arithmetic below supports
#   at seven columns: it gives about 149 px per cell on a 1366-wide
#   window and about 234 px on a 1920-wide one.
ROWS_PER_SCREEN = 3

# BLOCKS
#   (name, has_none_button, has_free_text). The episode block offers
#   "none of these" and a free-text box when it is chosen; the normal
#   block offers neither, because "my surroundings normally look like
#   none of these" is not an answer the shift can use and an empty
#   normal answer loses the whole pair.
BLOCKS = [
    ("episode", True,  True),
    ("normal",  False, False),
]

# FOG_COLUMNS / COLOUR_COLUMNS
#   The columns of each screen in DISPLAY ORDER for "forward": most
#   clear / most vivid first. "UP" is the placeholder for the
#   contrast-up column and is dropped automatically when
#   CONTRAST_UP_GAIN is None.
#
#   These are level LABELS from step 3's ladder, not parameters. The
#   parameters are read from the cell plan, so a label that step 3 did
#   not render fails CHECK 2 rather than silently rendering nothing.
FOG_COLUMNS    = ["UP", "h=0", "p35", "p25", "p15", "p10", "p05"]
COLOUR_COLUMNS = ["p75", "k=1", "p25", "p10", "p05", "p01"]

# p65 IS DELIBERATELY ABSENT from COLOUR_COLUMNS, and it is the only
# rung step 3 rendered that the grid does not show. Two reasons, and
# they are different reasons:
#
#   SIX COLUMNS ON BOTH SCREENS. With the contrast-up column dropped
#   the fog screen has six, and a colour screen of seven would make
#   the two screens compute different cell sizes -- or, under the
#   experiment's "widest screen in the file" rule, make the fog screen
#   smaller than it needs to be. Equal counts keep one size for the
#   whole block.
#
#   EQUAL FIRST STEPS. p75 is 25 percentile points above the original
#   and p25 is 25 below, so with p65 gone the first column on each
#   side of the original is the same distance away in the units the
#   ladder was calibrated in. With p65 in, the first step up was 10
#   points and the first step down 25.
#
# What this does NOT do is put the original in the same POSITION on
# both screens: it is 1st of 6 on the fog screen and 2nd of 6 on the
# colour screen, because the fog screen has nothing above the
# original and the colour screen has p75. So "which column did you
# pick" is still not comparable across the two features. The analysis
# is in feature units within each feature, where that does not enter.
# SYMMETRIC_COLUMNS below is the setting that would equalise the
# position, and it is off: see its comment for what it would cost.
#
# p65 stays in the PAIR trials at every fog rung. Only the grid's
# p65 x h=0 cell goes unused, and UNUSED_CELL_NOTES records that.

# SYMMETRIC_COLUMNS
#   True levels the two screens so the original sits in the SAME
#   position on both -- the position confound in the header. The count
#   of columns above the original and the count below it are each
#   levelled down to the smaller of the two screens', and the columns
#   dropped are the OUTERMOST on each side. What that costs depends on
#   whether the contrast-up column is kept:
#
#     with an up column   fog loses p05, colour loses p75; both
#                         screens become six columns with the original
#                         second. This is the layout step 3's build
#                         report describes.
#     without one         fog loses p05, colour loses p75 AND p65;
#                         both become five columns with the original
#                         leftmost, and the colour screen then has no
#                         column above the original at all.
#
#   Everything dropped stays in the pair grid either way.
#
#   False: the asymmetry is kept and reported. The analysis is in
#   feature units within each feature, where the confound does not
#   enter, and dropping p05 would shorten the fog range exactly where
#   a severe episode is most likely to sit. Note the second case
#   above: with no up column, symmetry costs the colour screen its
#   whole above-normal side, which is a worse trade than the confound.
#
#   False: the asymmetry is kept and reported. The analysis is in
#   feature units within each feature, where the confound does not
#   enter, and dropping p05 would shorten the fog range exactly where
#   a severe episode is most likely to sit.
SYMMETRIC_COLUMNS = False

# CONTRAST_UP_GAIN
#   Which contrast-up column the fog screen carries.
#
#   "undecided"  MODE = "mockup" treats this as "show me all of them,
#                one per screen"; MODE = "package" refuses to run.
#   "g10875"     gain 1.0875, +0.22 SD. 40 of 47 scenes take no
#                measured damage at all; 46 at a 0.50% tolerance.
#   "g11100"     gain 1.110, +0.27 SD. 22 of 47 clean, 34 at 0.50%.
#                Below the 24 scenes two sets need at a zero-damage
#                rule, so this gain forces a stated tolerance.
#   "g11500"     gain 1.150. UNMEASURED by step 1's sweep, rendered
#                blind. 1 of 47 clean.
#   None         the column is DROPPED. The fog screen becomes six
#                columns and the original sits leftmost on it.
#
#   SETTLED: None. The column was judged against step 3's
#   sheets_contrast/ diagnostic, which is drawn at the FULL 512 px,
#   and no gain was distinguishable from the original there. A
#   difference invisible at 512 px cannot appear at 149, so the
#   mock-up could only confirm it. The damage numbers never bore on
#   this: 1.0875 does no measured damage at all on 40 of 47 scenes
#   and is still invisible.
#
#   Why it was so weak, for the write-up: the gain holds the mean
#   fixed and moves only the tails -- about +7 on lum_p95 and -6 on
#   lum_p05 at 1.0875 -- so the change lives in a small fraction of
#   the frame and downscaling averages it away. Haze moves the same
#   statistics further AND lays a coherent veil over every pixel,
#   which is what survives a thumbnail. Equal SD on contrast_local is
#   not equal visibility in the two directions.
CONTRAST_UP_GAIN = None

# MOCKUP_GAINS
#   Which gains get a mock-up screen of their own, one gain per screen.
#   All three at once would be nine columns and 112 px per cell on a
#   1366-wide window, which is not the layout anyone will see, so
#   judging a gain from a nine-column sheet would be judging it at the
#   wrong size.
#   Now EMPTY: the gain question is settled, so the mock-up composes
#   only the screens that will ship. Put the three tags back to
#   reopen it.
MOCKUP_GAINS = []

# MOCKUP_INCLUDE_NO_UP
#   Also compose the fog screen WITHOUT an up column, so the
#   comparison is against the layout that drops it rather than against
#   nothing.
MOCKUP_INCLUDE_NO_UP = True

# MOCKUP_DETAIL_N
#   Scenes in the up-column detail figure: the original beside all
#   three gains, at the computed cell size, for scenes spread across
#   the damage range. This is the figure that answers "is it
#   distinguishable at all"; the screen mock-ups answer "is it
#   distinguishable in context".
MOCKUP_DETAIL_N = 6

# UNUSED_CELL_NOTES
#   Step 3 renders more grid cells than the grid shows, and every one
#   it does not show needs a recorded reason -- otherwise the stimulus
#   set carries images that no document accounts for, and the next
#   person to read it cannot tell a deliberate omission from a bug.
#
#   Keyed by the label unused_cells() derives: "contrast_up" for any
#   contrast-up cell, and "<colour level>@<fog level>" otherwise. A
#   cell that goes unused with no note here is a WARNING, not a
#   silence.
UNUSED_CELL_NOTES = {
    "contrast_up":
        "Rendered as three candidate gains (1.0875, 1.110, 1.150) for "
        "a column ABOVE the original on the fog screen, and dropped. "
        "No gain was distinguishable from the original on step 3's "
        "sheets_contrast/ diagnostic, which is drawn at the full "
        "512 px; the grid shows cells at 149-239 px, so a difference "
        "invisible at 512 px cannot appear there. The damage figures "
        "were never the binding constraint: 1.0875 does no measured "
        "damage at all on 40 of 47 scenes and is still invisible. The "
        "gain holds the mean fixed and moves only the histogram "
        "tails, which downscaling averages away, whereas haze lays a "
        "coherent veil over every pixel.",
    "p65@h=0":
        "Rendered as the first colour rung above the original and "
        "dropped from the GRID only -- p65 remains in the pair trials "
        "at every fog rung. Two reasons: it leaves six columns on "
        "both grid screens, so the whole block computes one cell "
        "size; and it makes the first step on each side of the "
        "original the same size, since p75 is 25 percentile points "
        "above the original and p25 is 25 below.",
}


# ============================================================
# Layout  (from the experiment's own code -- keep in step)
# ============================================================
#
# These reproduce config.js `grid` and grid.js's cell-size arithmetic.
# If the experiment's CSS changes, change them here and re-run the
# mock-up: a mock-up at the wrong cell size is worse than none, because
# it invites a decision about visibility from a picture nobody will see.
#
#   byWidth  = (innerWidth - PAGE_SIDE_PX - (cols-1)*COL_GAP_PX)/cols
#              - CELL_CHROME_PX
#   byHeight = (innerHeight - CHROME_PX - (rows-1)*ROW_GAP_PX)/rows
#   cell     = floor(min(byWidth, byHeight, MAX_CELL_PX)), at least
#              MIN_CELL_PX
MAX_CELL_PX     = 240
MIN_CELL_PX     = 100
ROW_GAP_PX      = 6
COL_GAP_PX      = 22
CHROME_PX       = 190   # instructions, buttons, page padding
PAGE_SIDE_PX    = 48    # left + right page padding
CELL_CHROME_PX  = 14    # 2 x (3 px border + 4 px padding) per column
IMAGE_BORDER_PX = 1     # the border drawn around each image
PAGE_BG         = "#7f7f7f"
CELL_BORDER_RGB = (0, 0, 0)

# WINDOWS
#   (label, innerWidth, innerHeight) to compute the mock-up for.
#
#   The inner height is NOT the screen height: the browser's own
#   chrome takes 100-130 px and the OS taskbar can take 40 more. At
#   1366x768 a realistic window is about 650 px of inner height, and
#   the HEIGHT binds -- 149 px per cell, not the 155 px the width
#   would allow. At 1920x1080 the WIDTH binds at seven columns, so the
#   exact inner height barely matters there; 920 is a window with
#   tabs, bookmarks and a taskbar.
WINDOWS = [
    ("1366x768",  1366, 650),
    ("1920x1080", 1920, 920),
]

# MOCKUP_LABELS
#   Draw the column label and the measured feature value above each
#   column in the mock-up. For YOUR reading only -- the experiment
#   shows no labels, and the mock-up marks the label strip clearly so
#   it is never mistaken for part of the screen.
MOCKUP_LABELS = True

# ============================================================
# Scene selection
# ============================================================

# N_SETS / SET_SIZE
#   Two sets of 12: SCREENS x ROWS_PER_SCREEN = 4 x 3. Checked against
#   each other at startup rather than trusted.
N_SETS   = 2
SET_SIZE = 12

# MATCH_FEATURES
#   What the two sets are matched on, as (manifest column, SD). The
#   baseline cell -- colour at the original, fog at the original -- is
#   what a scene looks like before anything is done to it, and it is
#   what determines where its whole ladder sits.
#
#   The SDs are the CORPUS SDs the registered analysis uses, so a
#   within-pair distance here is in the same units as an effect in
#   step 8. Matching in raw units would weight sat_mean four times as
#   heavily as contrast_local for no reason other than its scale.
SD_SAT = 39.182
SD_FOG = 9.348
MATCH_FEATURES = [("base_sat", SD_SAT), ("base_fog", SD_FOG)]

# MATCH_INCLUDE_SPANS
#   Also match on the row RANGE of each feature (how far the columns
#   travel on that scene), not just the baseline.
#
#   False by default, which is what was agreed: the sets are matched on
#   baseline contrast and colour. The spans are reported per pair
#   either way, so if the within-pair span differences turn out large
#   this can be turned on and the cost read off the report.
MATCH_INCLUDE_SPANS = False

# HEADROOM_GAIN / HEADROOM_TOLERANCE
#   The SECONDARY criterion: prefer scenes that take little damage
#   from the contrast-up gain, measured as flattened + crushed from
#   step 3's CHECK 7.
#
#   HEADROOM_GAIN   which gain's damage to score on. None, or a
#                   CONTRAST_UP_GAIN of None, switches the criterion
#                   off entirely -- there is no headroom to care about
#                   once the column is dropped.
#   HEADROOM_TOLERANCE  the share of pixels that may be flattened or
#                   crushed. 0.0 is a zero-damage rule; step 3's
#                   report gives the eligible counts at 0, 0.25%,
#                   0.5%, 1% and 2%.
HEADROOM_GAIN      = None     # None = follow CONTRAST_UP_GAIN
HEADROOM_TOLERANCE = 0.005

# MATCH_SLACK
#   How much total within-pair distance (summed over the 12 pairs, in
#   SD units) the headroom preference may cost before it is abandoned.
#
#   This is what makes the two criteria LEXICOGRAPHIC rather than a
#   weighted blend. Stage 1 finds the best matching with headroom
#   ignored. Stage 2 finds the best matching among headroom-eligible
#   scenes. Stage 2 is taken only if it costs no more than MATCH_SLACK;
#   otherwise stage 1 is taken and the report says the headroom
#   preference was dropped and by how much it missed.
#
#   0.15 SD over 12 pairs is 0.0125 SD per pair: a quarter of the
#   typical within-pair distance, and far below anything the design is
#   sensitive to.
MATCH_SLACK = 0.15

# MATCH_RESTARTS
#   Random restarts of the local search. The seed pass is greedy and
#   the improvement pass is swap-and-replace until no single move
#   helps, which can stop short of the best arrangement; restarts from
#   shuffled seeds and keeping the best is cheap at this size (47
#   scenes, 12 pairs) and makes the result stable against the seed.
MATCH_RESTARTS = 40

# MATCH_REPORT_TOL
#   CHECK 5 warns if the two sets' MEAN differs by more than this on
#   any matched feature, in SD units. Not a hard failure: the pairs are
#   what the design rests on and the means follow from them.
MATCH_REPORT_TOL = 0.05

# BALANCE_GROUPS
#   Prefer three different scene groups on each screen, and a similar
#   indoor/outdoor split between the two sets, among arrangements that
#   are equally good on the matching.
BALANCE_GROUPS = True

# EXCLUDE_SCENES
#   scene_ids never to use in the grid block. Empty: a grid scene
#   being one a participant also saw in the pair trials is the v2
#   behaviour and is deliberate -- it is what lets a grid placement be
#   compared with the same person's pair choices. The reuse counts are
#   reported per scene so an unusually heavily used scene can be
#   excluded here afterwards.
EXCLUDE_SCENES = []

# FORCE_PAIRS
#   Pairs to use regardless of the matching, as (scene_id, scene_id).
#   For overriding the automatic choice after looking at the preview,
#   without editing the code. Forced pairs count towards the 12.
FORCE_PAIRS = []

RANDOM_SEED = 42

# ============================================================
# The set swap
# ============================================================

# SWAP_RULE
#   How a participant is assigned which set goes to which block.
#
#   "random_recorded"  the client draws it with probability 0.5 at
#                      session start and records the draw. No server
#                      state, which this study does not have. The
#                      counterbalance is then binomial rather than
#                      exact: at 60 participants the expected
#                      imbalance is about 4 either way, which the
#                      report states.
#   "parity"           alternate by an incrementing participant
#                      counter. Exact, but it needs a server-side
#                      counter, so it is only available if one exists.
SWAP_RULE = "random_recorded"

# ============================================================
# Shipping
# ============================================================

# SHIP_LONG_EDGE
#   0    copy step 3's 512 px PNG untouched. The shipped file is then
#        BYTE-IDENTICAL to the measured one, which CHECK 6 verifies by
#        hash, and provenance needs no argument.
#   256  downscale to the measurement size with the same resize
#        `measure` applies, which makes the shipped file exactly the
#        array the features were computed on, and cuts the package by
#        roughly four fifths. The cost: the browser then scales
#        256 -> 149 rather than 512 -> 149, and local contrast is a
#        fine-detail measure, so the two paths do not land in the same
#        place. Smaller package, weaker provenance.
#
#   0 is the default. The report prints both sizes so the trade is a
#   number rather than a guess.
SHIP_LONG_EDGE = 0

# SHIP_FORMAT
#   "png" only. A JPEG grid block would be smaller again and is not
#   offered: the features were measured on lossless pixels, and a
#   quality setting is one more thing that would have to be defended.
SHIP_FORMAT = "png"

# PACKAGE_PREVIEW
#   Also write a composed preview of every screen in the package, at
#   the smaller window's cell size, for the record. The experiment
#   does NOT use these.
PACKAGE_PREVIEW = True

# ============================================================
# Checks
# ============================================================

# MONOTONIC_FLOOR
#   CHECK 3: the smallest step, in raw feature units, between adjacent
#   columns of a selected row. Step 3's CHECK 6 used 1.0 sat_mean
#   units for the colour ladder and found 0 of 282 ladders affected;
#   the same floor is used here for both features so that a row whose
#   columns are visually indistinguishable cannot be selected.
MONOTONIC_FLOOR = 1.0

# REMEASURE_ALL / REMEASURE_RTOL
#   CHECK 6: re-measure the shipped images and compare with the
#   manifest. True measures every shipped file (168 images, a few
#   seconds); an integer measures that many at random.
#
#   At SHIP_LONG_EDGE = 0 the comparison should be EXACT, because the
#   file is the same file. At 256 it should also be exact, because the
#   resize is the one `measure` would have applied anyway. A non-zero
#   difference means one of those two statements is wrong.
REMEASURE_ALL  = True
REMEASURE_RTOL = 1e-6

FIG_DPI = 150


# ============================================================
# Step 3, imported
# ============================================================

def load_step3():
    """Import step 3's script as a module.

    Everything this script needs from step 3 -- the feature code, the
    filename helpers, the measurement size -- comes from here rather
    than from a copy, so there is exactly one definition of each in the
    pipeline. Step 3 guards its entry point, so importing it runs
    nothing and reads no data.
    """
    if not os.path.isfile(STEP3_SCRIPT):
        raise FileNotFoundError(
            f"STEP3_SCRIPT not found: {STEP3_SCRIPT}\n"
            "  This script imports step 3's feature code and filename\n"
            "  helpers rather than copying them, so it needs the file.\n"
            "  Point STEP3_SCRIPT at build_stimulus_set_v4_corrected.py.")
    spec = importlib.util.spec_from_file_location("step3_build",
                                                  STEP3_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step3_build"] = mod
    spec.loader.exec_module(mod)

    need = ["measure", "cell_filename", "label_tag", "gain_tag",
            "FEATURE_SIZE", "IMAGE_SIZE", "STIMULUS_FORMAT"]
    missing = [n for n in need if not hasattr(mod, n)]
    if missing:
        raise RuntimeError(
            f"{STEP3_SCRIPT} has no {', '.join(missing)}. This is not "
            "the step 3 script this step was written against -- check "
            "that it is build_stimulus_set_v4_corrected.py and not an "
            "earlier version.")
    return mod


S3 = None   # set in main(); the imported step 3 module


# ============================================================
# Small helpers
# ============================================================

def resolve_output_dir():
    """OUTPUT_ROOT, never overwriting an earlier run."""
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
        return OUTPUT_ROOT
    if ON_EXISTING == "error":
        raise RuntimeError(
            f"{OUTPUT_ROOT} already exists and ON_EXISTING is "
            "'error'. Move it, or set ON_EXISTING = 'new'.")
    n = 2
    while os.path.exists(f"{OUTPUT_ROOT}_{n}"):
        n += 1
    out = f"{OUTPUT_ROOT}_{n}"
    os.makedirs(out)
    return out


def banner(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def fmt_pct(x):
    return f"{100.0 * x:.3f}%"


CELL_PLAN = None    # set by load_inputs(); the authoritative plan
L_LADDER = None     # set by load_inputs(); step 3's colour ladder


# ============================================================
# The experiment's layout arithmetic
# ============================================================

def cell_px(cols, rows, inner_w, inner_h):
    """The cell size the experiment will compute, verbatim from
    grid.js.

    Returned with the two candidate sizes and which one bound, because
    which one binds is the thing that decides whether adding or
    dropping a column changes anything: when the HEIGHT binds, a
    narrower screen costs nothing, and when the WIDTH binds, every
    extra column is paid for in pixels.
    """
    by_width = ((inner_w - PAGE_SIDE_PX - (cols - 1) * COL_GAP_PX)
                / float(cols) - CELL_CHROME_PX)
    by_height = ((inner_h - CHROME_PX - (rows - 1) * ROW_GAP_PX)
                 / float(rows))
    raw = min(by_width, by_height, float(MAX_CELL_PX))
    px = max(MIN_CELL_PX, int(np.floor(raw)))
    if by_width <= by_height and by_width <= MAX_CELL_PX:
        binds = "width"
    elif by_height < by_width and by_height <= MAX_CELL_PX:
        binds = "height"
    else:
        binds = f"cap {MAX_CELL_PX}"
    return px, by_width, by_height, binds


def file_cell_px(inner_w, inner_h, this_cols, other_cols):
    """The cell size the experiment computes, for a file whose two
    screens have `this_cols` and `other_cols` columns.

    ONE SIZE FOR THE WHOLE BLOCK, taken from the WIDEST screen in the
    file. That is the experiment's rule, and it matters as soon as the
    screens differ: sizing each screen by its own column count would
    make a six-column screen bigger than a seven-column one, which the
    experiment will not do. An earlier version of this script did
    exactly that, and it agreed with the experiment only by accident,
    while every screen happened to have the same count.
    """
    return cell_px(max(this_cols, other_cols), ROWS_PER_SCREEN,
                   inner_w, inner_h)


def layout_table(col_counts, realised, report):
    """CHECK 7: the cell size in every window, for every column count
    this run will show, with the one the experiment will actually
    compute marked."""
    banner("LAYOUT  (CHECK 7)")
    rows = []
    for label, iw, ih in WINDOWS:
        for cols in sorted(set(col_counts)):
            px, bw, bh, binds = cell_px(cols, ROWS_PER_SCREEN, iw, ih)
            rows.append({"window": label, "inner_w": iw, "inner_h": ih,
                         "cols": cols, "rows": ROWS_PER_SCREEN,
                         "by_width": round(bw, 1),
                         "by_height": round(bh, 1),
                         "binds": binds, "cell_px": px,
                         "realised": cols == realised,
                         "at_floor": px <= MIN_CELL_PX})
    T = pd.DataFrame(rows)
    print(T.to_string(index=False))

    floored = T[T.at_floor]
    if len(floored):
        print(f"\n  WARNING: {len(floored)} case(s) hit the "
              f"{MIN_CELL_PX} px floor, which means the layout no "
              "longer fits and the")
        print("  experiment will overflow rather than shrink further. "
              "Drop a column or a row.")
        report.append(("check 7 layout",
                       f"FAIL ({len(floored)} at the floor)"))
        ok = False
    else:
        print(f"\n  cell size {T.cell_px.min()}-{T.cell_px.max()} px "
              f"across {len(WINDOWS)} window(s) and "
              f"{T.cols.nunique()} column count(s)")
        report.append(("check 7 layout",
                       f"PASS ({T.cell_px.min()}-{T.cell_px.max()} px)"))
        ok = True

    print(f"\n  THE BLOCK SHIPS {realised} COLUMNS on its widest "
          "screen, so that row is the")
    print("  size every screen is drawn at -- one size for the whole "
          "block, which is what")
    print("  the experiment computes.")
    R = T[T.realised]
    for r in R.itertuples():
        print(f"    {r.window:<11s} {r.cell_px} px  ({r.binds} binds)")
    report.append(("realised cell size",
                   ", ".join(f"{r.window} {r.cell_px} px"
                             for r in R.itertuples())
                   + f" at {realised} columns"))
    for label, iw, ih in WINDOWS:
        sub = T[T.window == label]
        b = sorted(sub.binds.unique())
        print(f"  {label}: binds on {', '.join(b)}")
    print("\n  Where the HEIGHT binds, dropping the contrast-up column "
          "buys no extra pixels;")
    print("  where the WIDTH binds, it buys some. Both happen here, "
          "which is why the")
    print("  mock-up is composed for both windows.")
    print("\n  NOTE, worth settling against the stylesheet: byWidth "
          f"subtracts {CELL_CHROME_PX} px of")
    print("  border and padding per COLUMN and byHeight subtracts "
          "none per ROW. If the")
    print("  cells carry that border and padding vertically too, each "
          f"row is {CELL_CHROME_PX} px")
    print(f"  taller than the formula reserves -- "
          f"{ROWS_PER_SCREEN * CELL_CHROME_PX} px over "
          f"{ROWS_PER_SCREEN} rows, which would overflow the space "
          "CHROME_PX")
    print("  leaves. The mock-up is drawn the way the formula reads, "
          "so if the real page")
    print("  scrolls, this is the line to come back to.")
    return T, ok


def settings_check():
    """Every setting that can contradict another, checked before any
    work is done.

    The alternative is a failure three minutes in, with half a package
    written, from a typo in a level label.
    """
    banner("SETTINGS")
    bad = []

    if MODE not in ("mockup", "package"):
        bad.append(f"MODE is {MODE!r}; expected 'mockup' or 'package'.")

    if MODE == "package" and CONTRAST_UP_GAIN == "undecided":
        bad.append(
            "MODE is 'package' but CONTRAST_UP_GAIN is still "
            "'undecided'.\n"
            "    The contrast-up column is provisional and the mock-up "
            "is what settles it.\n"
            "    Run MODE = 'mockup', look at OUTPUT/mockup/, then set "
            "CONTRAST_UP_GAIN to\n"
            "    a gain tag or to None (dropping the column) and run "
            "again.")

    if CONTRAST_UP_GAIN not in ("undecided", None) and \
            CONTRAST_UP_GAIN not in MOCKUP_GAINS:
        print(f"  NOTE: CONTRAST_UP_GAIN {CONTRAST_UP_GAIN!r} is not in "
              "MOCKUP_GAINS, so no")
        print("  mock-up was composed for it. That is allowed, but the "
              "gain was then chosen")
        print("  from something other than this script's mock-up.")

    if len(SCREENS) * ROWS_PER_SCREEN != SET_SIZE:
        bad.append(
            f"SET_SIZE is {SET_SIZE} but the screen plan needs "
            f"{len(SCREENS)} x {ROWS_PER_SCREEN} = "
            f"{len(SCREENS) * ROWS_PER_SCREEN} scenes per set.")

    if N_SETS != len(BLOCKS):
        bad.append(
            f"N_SETS is {N_SETS} and there are {len(BLOCKS)} blocks. "
            "One set per block, swapped between participants, is the "
            "design; any other count has no defined swap.")

    feats = [f for f, _ in SCREENS]
    for f in set(feats):
        if feats.count(f) != 2:
            bad.append(
                f"feature {f!r} appears {feats.count(f)} time(s) in "
                "SCREENS; the design asks each feature twice, once in "
                "each column order.")
    for f in set(feats):
        orders = sorted(o for g, o in SCREENS if g == f)
        if orders != ["forward", "reversed"]:
            bad.append(
                f"feature {f!r} has column orders {orders}; expected "
                "one 'forward' and one 'reversed'.")

    if SET_SIZE % 2:
        bad.append(f"SET_SIZE {SET_SIZE} is odd, so it cannot be split "
                   "evenly between the two features.")

    if SWAP_RULE not in ("random_recorded", "parity"):
        bad.append(f"SWAP_RULE is {SWAP_RULE!r}; expected "
                   "'random_recorded' or 'parity'.")

    if SHIP_FORMAT != "png":
        bad.append(f"SHIP_FORMAT is {SHIP_FORMAT!r}; only 'png' is "
                   "supported (see the setting's comment).")

    if SHIP_LONG_EDGE not in (0, 256):
        bad.append(f"SHIP_LONG_EDGE is {SHIP_LONG_EDGE}; expected 0 "
                   "(copy untouched) or 256 (the measurement size).")

    if FORCE_PAIRS:
        flat = [s for p in FORCE_PAIRS for s in p]
        if len(set(flat)) != len(flat):
            bad.append("FORCE_PAIRS uses the same scene twice.")
        if len(FORCE_PAIRS) > SET_SIZE:
            bad.append(f"FORCE_PAIRS has {len(FORCE_PAIRS)} pairs but "
                       f"only {SET_SIZE} are used.")
        clash = set(flat) & set(EXCLUDE_SCENES)
        if clash:
            bad.append("FORCE_PAIRS and EXCLUDE_SCENES both name "
                       f"{sorted(clash)}.")

    if not (0.0 <= HEADROOM_TOLERANCE <= 1.0):
        bad.append(f"HEADROOM_TOLERANCE is {HEADROOM_TOLERANCE}; it is "
                   "a share of pixels, so it belongs in [0, 1].")

    if MATCH_SLACK < 0:
        bad.append(f"MATCH_SLACK is {MATCH_SLACK}; it is a distance "
                   "budget and cannot be negative.")

    if bad:
        raise RuntimeError(
            "The settings contradict each other:\n  - "
            + "\n  - ".join(bad))

    print(f"  mode            {MODE}")
    print("  blocks          "
          + ", ".join(f"{n} ({'none' if hn else 'no none'})"
                      for n, hn, _ in BLOCKS))
    print(f"  screens/block   {len(SCREENS)} x {ROWS_PER_SCREEN} rows")
    print(f"  sets            {N_SETS} x {SET_SIZE} scenes")
    print(f"  contrast-up     {CONTRAST_UP_GAIN!r}")
    print(f"  swap            {SWAP_RULE}")
    print("  ship            "
          + ("512 px copies, untouched" if SHIP_LONG_EDGE == 0
             else f"{SHIP_LONG_EDGE} px, re-encoded"))


# ============================================================
# Inputs  (CHECK 1)
# ============================================================

def load_inputs(report):
    """The manifest, the cell plan, the damage table and the colour
    ladder, cross-checked against each other.

    The cell plan is the authoritative description of what step 3
    rendered -- it is the one table step 3's measuring pass, contact
    sheets and stimulus writer all go through -- so it is read rather
    than the loop being rebuilt here. The manifest is then checked
    against it instead of being trusted.
    """
    global CELL_PLAN, L_LADDER
    banner("INPUTS  (CHECK 1)")

    for path, what in ((MANIFEST_CSV, "the manifest"),
                       (CELL_PLAN_CSV, "the cell plan"),
                       (DAMAGE_CSV, "the contrast-up damage table"),
                       (LADDER_CSV, "the colour ladder")):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found ({what}). Step 3 writes all four "
                f"into its output directory; STIMULUS_ROOT is "
                f"{STIMULUS_ROOT!r} -- check the directory step 3's "
                "build report says it actually wrote, since a re-run "
                "lands in _2, _3.")
    if not os.path.isdir(STIMULI_DIR):
        raise FileNotFoundError(
            f"{STIMULI_DIR} not found. Step 3 must have been run with "
            "RENDER_MODE = 'full'; a 'preview' run measures everything "
            "but writes no images.")

    M = pd.read_csv(MANIFEST_CSV)
    P = pd.read_csv(CELL_PLAN_CSV)
    D = pd.read_csv(DAMAGE_CSV)
    L = pd.read_csv(LADDER_CSV)
    CELL_PLAN = P
    L_LADDER = L

    for col in ("scene_id", "use", "kind", "colour_level", "fog_level",
                "contrast_level", "chroma", "haze", "filename",
                "sat_mean", "contrast_local", "group", "indoor",
                "category"):
        if col not in M.columns:
            raise RuntimeError(
                f"{MANIFEST_CSV} has no {col!r} column. This is not a "
                "manifest from the step 3 version this step was "
                "written against.")

    n_scenes = M.scene_id.nunique()
    print(f"  manifest    {len(M):,} images, {n_scenes} scenes")
    print(f"  cell plan   {len(P)} cells/scene")
    print("  by use      " + ", ".join(
        f"{u} {n:,}" for u, n in M.use.value_counts().items()))

    # ---- the manifest and the cell plan describe the same cells ----
    key = ["kind", "colour_level", "fog_level", "contrast_level"]
    mk = (M[key].fillna("").drop_duplicates()
          .sort_values(key).reset_index(drop=True))
    pk = (P[key].fillna("").drop_duplicates()
          .sort_values(key).reset_index(drop=True))
    if not mk.equals(pk):
        only_m = set(map(tuple, mk.values)) - set(map(tuple, pk.values))
        only_p = set(map(tuple, pk.values)) - set(map(tuple, mk.values))
        raise RuntimeError(
            "The manifest and the cell plan do not describe the same "
            "cells.\n"
            f"  in the manifest only: {sorted(only_m)}\n"
            f"  in the plan only:     {sorted(only_p)}\n"
            "  They come from one run of step 3, so this means the two "
            "files are from\n  different runs -- check STIMULUS_ROOT.")
    print(f"  manifest and cell plan agree on all {len(pk)} cells")

    # ---- every cell exists for every scene ----
    per = M.groupby("scene_id").size()
    if per.nunique() != 1 or per.iloc[0] != len(P):
        odd = per[per != len(P)]
        raise RuntimeError(
            f"{len(odd)} scene(s) do not have all {len(P)} cells: "
            f"{dict(odd.head())}. An incomplete scene cannot be put on "
            "a grid screen, because one of its columns would be "
            "missing.")
    print(f"  every scene has all {len(P)} cells")

    # ---- the grid-usable cells ----
    G = M[M.use.isin(("grid", "both"))].copy()
    n_grid = len(P[P.use.isin(("grid", "both"))])
    print(f"  grid-usable {len(G):,} images "
          f"({n_grid} cells/scene x {n_scenes} scenes)")
    if len(G) != n_grid * n_scenes:
        raise RuntimeError(
            f"{len(G)} grid-usable images but {n_grid} x {n_scenes} = "
            f"{n_grid * n_scenes} expected.")

    # ---- the files are on disk ----
    missing = [f for f in G.filename if not os.path.isfile(
        os.path.join(STIMULI_DIR, f))]
    if missing:
        raise RuntimeError(
            f"{len(missing)} grid image(s) named in the manifest are "
            f"not in {STIMULI_DIR}. First: {missing[0]}")
    print(f"  all {len(G):,} grid images are on disk")

    # ---- the damage table ----
    for col in ("scene_id", "contrast_level", "flattened", "crushed",
                "damage", "d_fog"):
        if col not in D.columns:
            raise RuntimeError(f"{DAMAGE_CSV} has no {col!r} column.")
    gains = sorted(D.contrast_level.unique())
    print(f"  damage      {len(D)} rows, gains {', '.join(gains)}")
    if D.scene_id.nunique() != n_scenes:
        print(f"  WARNING: the damage table covers "
              f"{D.scene_id.nunique()} scenes, not {n_scenes}.")

    report.append(("stimulus set", STIMULUS_ROOT))
    report.append(("scenes available", str(n_scenes)))
    report.append(("grid-usable images", f"{len(G):,}"))
    report.append(("check 1 inputs", "PASS"))
    return M, G, P, D, L


# ============================================================
# The columns of each screen
# ============================================================

def effective_up():
    """The contrast-up column to build with.

    While CONTRAST_UP_GAIN is "undecided" the selection, the scene
    table and the monotonicity check are all built with NO up column.
    That is the conservative reading: the selection must not depend on
    a gain nobody has chosen yet. The mock-up then composes a screen
    for each candidate gain, and the selection is recomputed on the
    package run once the gain is set -- which can move it, because the
    headroom criterion only exists once there is a gain to have
    headroom for. The report says which of the two a run was.
    """
    return None if CONTRAST_UP_GAIN == "undecided" else CONTRAST_UP_GAIN


def _untrimmed_labels(feature, up_tag):
    """One screen's columns before any symmetry trim, with the index
    of the original."""
    col_orig = getattr(S3, "COLOUR_IDENTITY_LABEL", "k=1")
    fog_orig = getattr(S3, "FOG_IDENTITY_LABEL", "h=0")
    if feature == "fog":
        labels = list(FOG_COLUMNS)
        if up_tag is None and "UP" in labels:
            labels.remove("UP")
        original = fog_orig
    elif feature == "colour":
        labels = list(COLOUR_COLUMNS)
        original = col_orig
    else:
        raise RuntimeError(f"unknown feature {feature!r}")
    if original not in labels:
        raise RuntimeError(
            f"the {feature} screen's columns {labels} do not include "
            f"the original ({original!r}); without it there is nothing "
            "for the shift to be measured from.")
    return labels, labels.index(original)


def symmetry_trim(up_tag):
    """How many columns to drop from each end of each screen so that
    the original sits at the SAME POSITION on both.

    The position of the original is one plus the number of columns
    above it, so symmetry needs the two screens to agree on the count
    ABOVE the original -- and, for the screens to be the same width,
    on the count BELOW it too. Both counts are therefore levelled down
    to the smaller of the two, and the columns dropped are the
    OUTERMOST on each side: the extremes are the ones a participant is
    least likely to need, and they stay in the pair grid regardless.

    An earlier version dropped one column from the end of each screen,
    which trimmed both screens without equalising anything and left
    the original at position 2 on one and 3 on the other. The check at
    the end of describe_columns is what caught it, and it is kept for
    that reason.
    """
    above, below = {}, {}
    for feature in ("fog", "colour"):
        labels, zero = _untrimmed_labels(feature, up_tag)
        above[feature] = zero
        below[feature] = len(labels) - 1 - zero
    ta, tb = min(above.values()), min(below.values())
    return {f: (above[f] - ta, below[f] - tb)
            for f in ("fog", "colour")}


def resolve_columns(feature, report=None, up="setting"):
    """Which cell each column of one screen shows, in display order
    for "forward".

    `up` overrides the contrast-up column for this call, which is how
    the mock-up composes one screen per candidate gain without
    disturbing the run's own settings.

    A column is a (colour level, fog level, contrast level) triple, and
    the triple is looked up in the cell plan, so a label step 3 never
    rendered is an error here rather than a missing file later.

    `level_index` is signed and counted FROM THE ORIGINAL:

        fog     -1 contrast-up, 0 the original, +1..+5 the fog rungs
        colour  -2 p75, -1 p65, 0 the original, +1..+4 the drained

    Negative is "more than normal", positive is "less than normal",
    which is the direction a derealization report runs. The analysis
    uses the measured feature value rather than the index; the index is
    for the data file, where a signed step is easier to read than a
    position that moves when the column order reverses.
    """
    col_orig = getattr(S3, "COLOUR_IDENTITY_LABEL", "k=1")
    fog_orig = getattr(S3, "FOG_IDENTITY_LABEL", "h=0")
    up_tag = effective_up() if up == "setting" else up

    labels, _ = _untrimmed_labels(feature, up_tag)
    if SYMMETRIC_COLUMNS:
        da, db = symmetry_trim(up_tag)[feature]
        labels = labels[da:len(labels) - db] if db else labels[da:]
    original = fog_orig if feature == "fog" else col_orig
    zero = labels.index(original)

    cols = []
    for pos, lab in enumerate(labels):
        if lab == "UP":
            tag = up_tag
            cell = {"kind": "contrast_up", "colour_level": col_orig,
                    "fog_level": fog_orig, "contrast_level": tag}
        elif feature == "fog":
            cell = {"kind": "grid", "colour_level": col_orig,
                    "fog_level": lab, "contrast_level": None}
        else:
            cell = {"kind": "grid", "colour_level": lab,
                    "fog_level": fog_orig, "contrast_level": None}

        sel = CELL_PLAN[
            (CELL_PLAN.kind == cell["kind"])
            & (CELL_PLAN.colour_level == cell["colour_level"])
            & (CELL_PLAN.fog_level == cell["fog_level"])
            & (CELL_PLAN.contrast_level.fillna("")
               == (cell["contrast_level"] or ""))]
        if not len(sel):
            raise RuntimeError(
                f"the {feature} screen asks for column {lab!r}, which "
                "is not a cell step 3 rendered.\n"
                f"  looked for kind={cell['kind']!r}, "
                f"colour={cell['colour_level']!r}, "
                f"fog={cell['fog_level']!r}, "
                f"contrast={cell['contrast_level']!r}\n"
                "  Check FOG_COLUMNS / COLOUR_COLUMNS against the cell "
                "plan's level labels.")
        r = sel.iloc[0]
        if r.use not in ("grid", "both"):
            raise RuntimeError(
                f"column {lab!r} of the {feature} screen is a cell "
                f"step 3 marked use={r.use!r}, not a grid cell. Only "
                "cells on the grid cross may go on a grid screen.")
        cols.append({
            "label": lab if lab != "UP" else f"up {up_tag}",
            "column_label": lab,
            "kind": cell["kind"],
            "colour_level": cell["colour_level"],
            "fog_level": cell["fog_level"],
            "contrast_level": cell["contrast_level"],
            "chroma": float(r.chroma), "haze": float(r.haze),
            "gain": float(r.gain),
            "level_index": pos - zero,
            "forward_position": pos + 1,
            "is_original": lab == original,
        })
    if report is not None:
        report.append((f"{feature} columns",
                       f"{len(cols)}: " + ", ".join(
                           c["label"] for c in cols)))
    return cols


def columns_for_other(feature):
    """The other screen's columns, which the cell size depends on."""
    return resolve_columns("colour" if feature == "fog" else "fog")


def feature_column(feature):
    """Which measured feature the screen's columns move."""
    return {"fog": "contrast_local", "colour": "sat_mean"}[feature]


def describe_columns(report):
    """Print both screens' columns and say where the original sits."""
    banner("THE COLUMNS")
    out = {}
    for feature in ("fog", "colour"):
        cols = resolve_columns(feature, report)
        out[feature] = cols
        feat = feature_column(feature)
        print(f"\n  {feature} screen ({feat}), forward order:")
        for c in cols:
            mark = "  <- the original" if c["is_original"] else ""
            print(f"    {c['forward_position']}. {c['label']:<12s} "
                  f"index {c['level_index']:+d}   "
                  f"chroma {c['chroma']:.4f}  haze {c['haze']:.4f}  "
                  f"gain {c['gain']:.4f}{mark}")

    pos = {f: [c["forward_position"] for c in cols if c["is_original"]][0]
           for f, cols in out.items()}
    ncol = {f: len(cols) for f, cols in out.items()}
    print(f"\n  the original sits at position {pos['fog']} of "
          f"{ncol['fog']} on the fog screen and {pos['colour']} of "
          f"{ncol['colour']} on the colour screen.")
    if SYMMETRIC_COLUMNS and (pos["fog"] != pos["colour"]
                              or ncol["fog"] != ncol["colour"]):
        raise RuntimeError(
            "SYMMETRIC_COLUMNS is True but the two screens came out "
            f"as {ncol['fog']} columns with the original at "
            f"{pos['fog']} and {ncol['colour']} columns with it at "
            f"{pos['colour']}. The whole point of the setting is that "
            "those agree, so this is a bug in symmetry_trim rather "
            "than a configuration to work around.")
    if pos["fog"] != pos["colour"]:
        print("  THE POSITION CONFOUND: comparing WHICH COLUMN people "
              "pick across the two")
        print("  features is therefore partly a position effect. The "
              "shift within each")
        print("  feature is unaffected, and that is what the analysis "
              "uses. SYMMETRIC_COLUMNS")
        print("  = True removes it at the cost of the deepest fog rung "
              "and the most vivid")
        print("  colour level, on the grid screens only.")
        report.append(("position of the original",
                       f"fog {pos['fog']}/{ncol['fog']}, colour "
                       f"{pos['colour']}/{ncol['colour']} "
                       "(asymmetric, reported)"))
    else:
        report.append(("position of the original",
                       f"{pos['fog']} on both (symmetric)"))
    return out


# ============================================================
# The scene table  (CHECK 2, CHECK 3)
# ============================================================

def cell_row(G, scene_id, col):
    """The manifest row for one column of one scene."""
    sel = G[(G.scene_id == scene_id)
            & (G.kind == col["kind"])
            & (G.colour_level == col["colour_level"])
            & (G.fog_level == col["fog_level"])
            & (G.contrast_level.fillna("")
               == (col["contrast_level"] or ""))]
    if len(sel) != 1:
        raise RuntimeError(
            f"{len(sel)} manifest rows for {scene_id} column "
            f"{col['label']!r}; expected exactly one. A cell is keyed "
            "by (colour level, fog level, contrast level), so this "
            "means the manifest is not keyed as step 3 intended.")
    return sel.iloc[0]


def check_against_ladder(S, L, columns, report):
    """Part of CHECK 2: my colour span, computed by resolving the
    columns myself, against step 3's own colour_ladder.csv.

    Step 3's CHECK 6 measured the same quantity independently -- the
    saturation of the most vivid column minus the least, at the h = 0
    row -- so the two should agree EXACTLY when the colour screen
    carries the full ladder. If they do not, my column resolution has
    picked different cells from the ones step 3 calls the ladder, and
    everything downstream is built on the wrong cells.

    Skipped when SYMMETRIC_COLUMNS has dropped a column, because then
    the two are deliberately measuring different things.
    """
    if SYMMETRIC_COLUMNS:
        print("  SYMMETRIC_COLUMNS has dropped a colour column, so the "
              "span is deliberately")
        print("  not step 3's ladder span; the cross-check is skipped.")
        report.append(("colour span vs step 3's ladder",
                       "skipped (symmetric columns)"))
        return True
    fog_orig = getattr(S3, "FOG_IDENTITY_LABEL", "h=0")
    sub_l = L[L.fog_level == fog_orig].set_index("scene_id")
    common = [s for s in S.index if s in sub_l.index]
    if not common:
        print("  colour_ladder.csv has no h = 0 row; cross-check "
              "skipped.")
        report.append(("colour span vs step 3's ladder",
                       "skipped (no h=0 row)"))
        return True
    d = (S.loc[common, "colour_span"].to_numpy(float)
         - sub_l.loc[common, "span"].to_numpy(float))
    worst = float(np.abs(d).max())
    print(f"  colour span vs step 3's colour_ladder.csv: worst "
          f"difference {worst:.3e} over {len(common)} scenes")
    if worst > 1e-6:
        off = pd.Series(np.abs(d), index=common).sort_values(
            ascending=False).head(5)
        print("  FAIL: the columns resolved here are not the ladder "
              "step 3 measured.")
        print("  " + off.to_string().replace("\n", "\n  "))
        report.append(("colour span vs step 3's ladder",
                       f"FAIL (worst {worst:.1e})"))
        return False
    report.append(("colour span vs step 3's ladder",
                   f"PASS (exact over {len(common)} scenes)"))
    return True


def scene_table(G, D, columns, report):
    """One row per scene: the baselines the sets are matched on, the
    range each screen's columns travel on that scene, and the damage
    each contrast-up gain does to it.

    The SPAN is what makes the columns look different from each other.
    A scene with a narrow span shows six columns that all look much the
    same, which asks the participant to make a distinction the stimulus
    does not offer -- so the span is what the rows are ranked on when
    scenes are handed to screens.
    """
    banner("THE SCENES  (CHECK 2)")

    rows = []
    for sid, g in G.groupby("scene_id"):
        meta = g.iloc[0]
        rec = {"scene_id": sid, "category": meta.category,
               "group": meta.group, "indoor": meta.indoor}
        for feature, cols in columns.items():
            feat = feature_column(feature)
            vals, files = [], []
            for c in cols:
                r = cell_row(G, sid, c)
                vals.append(float(r[feat]))
                files.append(r.filename)
            rec[f"{feature}_span"] = max(vals) - min(vals)
            rec[f"{feature}_span_sd"] = (max(vals) - min(vals)) / (
                SD_FOG if feature == "fog" else SD_SAT)
            rec[f"{feature}_files"] = files
            rec[f"{feature}_values"] = vals
        orig_c = [c for c in columns["colour"] if c["is_original"]][0]
        orig_f = [c for c in columns["fog"] if c["is_original"]][0]
        rec["base_sat"] = float(cell_row(G, sid, orig_c).sat_mean)
        rec["base_fog"] = float(cell_row(G, sid, orig_f).contrast_local)
        rows.append(rec)

    S = pd.DataFrame(rows).set_index("scene_id").sort_index()
    print(f"  {len(S)} scenes, every column of both screens resolved "
          "to a file")
    report.append(("check 2 columns exist", f"PASS ({len(S)} scenes)"))
    check_against_ladder(S, L_LADDER, columns, report)

    # ---- the damage columns ----
    dmg = D.pivot_table(index="scene_id", columns="contrast_level",
                        values="damage")
    dfog = D.pivot_table(index="scene_id", columns="contrast_level",
                         values="d_fog")
    S = S.join(dmg.add_prefix("damage_")).join(dfog.add_prefix("dfog_"))

    show = ["base_sat", "base_fog", "colour_span", "fog_span"]
    print()
    print("  " + S[show].describe().round(2).to_string()
          .replace("\n", "\n  "))

    print("\n  span in corpus SD units: colour "
          f"{S.colour_span_sd.min():.2f}-{S.colour_span_sd.max():.2f}, "
          f"fog {S.fog_span_sd.min():.2f}-{S.fog_span_sd.max():.2f}")
    report.append(("colour span (SD)",
                   f"{S.colour_span_sd.min():.2f}-"
                   f"{S.colour_span_sd.max():.2f}"))
    report.append(("fog span (SD)",
                   f"{S.fog_span_sd.min():.2f}-"
                   f"{S.fog_span_sd.max():.2f}"))
    return S


def check_monotone(S, columns, report):
    """CHECK 3: every column step, in every scene, in the direction the
    screen intends and by at least MONOTONIC_FLOOR.

    A row whose columns are out of order would show the participant a
    ladder that does not climb, and a step below the floor is a pair of
    columns that look the same -- either way the choice is not the one
    the screen is asking for. Checked on every scene, not only the
    selected ones, because the selection is what this feeds.
    """
    banner("MONOTONE COLUMNS  (CHECK 3)")

    bad, weak = [], []
    rows = []
    for feature, cols in columns.items():
        feat = feature_column(feature)
        for sid in S.index:
            vals = S.loc[sid, f"{feature}_values"]
            steps = np.diff(vals)              # forward order
            # forward order runs from most to least, so every step
            # must be negative
            rows.append({"scene_id": sid, "feature": feature,
                         "weakest_step": float(-steps.max()),
                         "strongest_step": float(-steps.min()),
                         "monotone": bool((steps < 0).all()),
                         "above_floor":
                             bool((-steps >= MONOTONIC_FLOOR).all())})
            if not (steps < 0).all():
                where = [cols[i]["label"] + " -> " + cols[i + 1]["label"]
                         for i in np.where(steps >= 0)[0]]
                bad.append((sid, feature, where))
            elif not (-steps >= MONOTONIC_FLOOR).all():
                i = int(np.argmax(steps))
                weak.append((sid, feature,
                             f"{cols[i]['label']} -> "
                             f"{cols[i + 1]['label']}",
                             float(-steps[i])))
        sub = pd.DataFrame([r for r in rows if r["feature"] == feature])
        print(f"  {feature:<7s} ({feat}): weakest step "
              f"{sub.weakest_step.min():.2f}, median "
              f"{sub.weakest_step.median():.2f}, "
              f"{int(sub.monotone.sum())}/{len(sub)} monotone, "
              f"{int(sub.above_floor.sum())}/{len(sub)} above the "
              f"{MONOTONIC_FLOOR} floor")

    T = pd.DataFrame(rows)
    if bad:
        print(f"\n  NOT MONOTONE in {len(bad)} scene x feature(s):")
        for sid, feature, where in bad[:10]:
            print(f"    {sid} ({feature}): {', '.join(where)}")
        print("  These scenes are excluded from the selection.")
        report.append(("check 3 monotone", f"FAIL ({len(bad)})"))
    elif weak:
        print(f"\n  monotone everywhere, but {len(weak)} scene x "
              f"feature(s) have a step below {MONOTONIC_FLOOR}:")
        for sid, feature, where, v in weak[:10]:
            print(f"    {sid} ({feature}): {where} = {v:.2f}")
        print("  These are excluded from the selection for that "
              "feature.")
        report.append(("check 3 monotone",
                       f"PASS, {len(weak)} below the floor"))
    else:
        print(f"\n  PASS: every column step in every scene is in the "
              f"intended direction and at least {MONOTONIC_FLOOR}.")
        report.append(("check 3 monotone", "PASS"))

    # a scene is usable for a feature only if that feature's ladder is
    # monotone and above the floor on it
    usable = {}
    for feature in columns:
        sub = T[T.feature == feature].set_index("scene_id")
        usable[feature] = set(
            sub[(sub.monotone) & (sub.above_floor)].index)
        n_out = len(S) - len(usable[feature])
        if n_out:
            print(f"  {feature}: {n_out} scene(s) not usable")
    return T, usable


# ============================================================
# Matched pairs  (CHECK 4, CHECK 5)
# ============================================================

def match_space(S, ids):
    """The coordinates the matching works in: each matched feature
    divided by its corpus SD, so one unit of distance is one SD of the
    feature the registered analysis measures."""
    feats = list(MATCH_FEATURES)
    if MATCH_INCLUDE_SPANS:
        feats = feats + [("fog_span", SD_FOG), ("colour_span", SD_SAT)]
    z = np.column_stack([S.loc[ids, f].to_numpy(float) / sd
                         for f, sd in feats])
    return z, [f for f, _ in feats]


def pair_distances(z):
    d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d


def _greedy_pairs(Dm, allowed, npairs, forced, rng, shuffle=False):
    """A seed arrangement: take the closest available pair, repeat.

    Forced pairs are placed first and their members removed, so a
    forced pair is honoured whatever it costs.
    """
    used = set()
    pairs = []
    for i, j in forced:
        pairs.append((i, j))
        used.add(i)
        used.add(j)
    cand = [(Dm[i, j], i, j)
            for i in allowed for j in allowed
            if i < j and i not in used and j not in used]
    if shuffle:
        # jitter the order so restarts explore, while still being
        # mostly greedy
        cand = [(d * (1.0 + 0.35 * rng.random()), i, j)
                for d, i, j in cand]
    cand.sort()
    for _, i, j in cand:
        if i in used or j in used:
            continue
        pairs.append((i, j))
        used.add(i)
        used.add(j)
        if len(pairs) == npairs:
            break
    return pairs if len(pairs) == npairs else None


def _improve(Dm, pairs, allowed, forced_set, max_moves=10000):
    """Local search: swap members between two pairs, or replace a
    member with an unused scene, while either helps.

    Forced pairs are never touched.
    """
    pairs = [list(p) for p in pairs]
    free = sorted(set(allowed) - {x for p in pairs for x in p})
    movable = [k for k, p in enumerate(pairs)
               if tuple(sorted(p)) not in forced_set]

    for _ in range(max_moves):
        best = None
        for a, b in itertools.combinations(movable, 2):
            i, j = pairs[a]
            k, l = pairs[b]
            cur = Dm[i, j] + Dm[k, l]
            for alt in ((i, k, j, l), (i, l, j, k)):
                new = Dm[alt[0], alt[1]] + Dm[alt[2], alt[3]]
                if new < cur - 1e-12 and (best is None
                                          or new - cur < best[0]):
                    best = (new - cur, "swap", a, b, alt)
        for a in movable:
            i, j = pairs[a]
            for f in free:
                for drop, keep in ((i, j), (j, i)):
                    new = Dm[f, keep]
                    if new < Dm[i, j] - 1e-12 and (
                            best is None or new - Dm[i, j] < best[0]):
                        best = (new - Dm[i, j], "repl", a, f, drop, keep)
        if best is None:
            break
        if best[1] == "swap":
            _, _, a, b, alt = best
            pairs[a] = [alt[0], alt[1]]
            pairs[b] = [alt[2], alt[3]]
        else:
            _, _, a, f, drop, keep = best
            pairs[a] = [f, keep]
            free.remove(f)
            free.append(drop)
    return [tuple(sorted(p)) for p in pairs]


def _cost(Dm, pairs):
    return float(sum(Dm[i, j] for i, j in pairs))


def best_matching(Dm, allowed, npairs, forced, rng):
    """The best arrangement this search finds, over MATCH_RESTARTS
    restarts."""
    forced_set = {tuple(sorted(p)) for p in forced}
    best, best_c = None, np.inf
    for r in range(max(1, MATCH_RESTARTS)):
        seed = _greedy_pairs(Dm, allowed, npairs, forced, rng,
                             shuffle=(r > 0))
        if seed is None:
            return None, np.inf
        p = _improve(Dm, seed, allowed, forced_set)
        c = _cost(Dm, p)
        if c < best_c - 1e-12:
            best, best_c = p, c
    return best, best_c


def match_pairs(S, usable, report, root):
    """Twelve matched pairs, chosen on the baselines first and on
    contrast headroom second.

    The two criteria are LEXICOGRAPHIC, not blended:

      stage 1  the best matching over every eligible scene, with
               headroom ignored. This is the benchmark.
      stage 2  the best matching over the scenes that meet the
               headroom tolerance. Taken only if it costs no more than
               MATCH_SLACK more than stage 1.

    The matching is what the normal-to-episode comparison rests on, so
    it wins; the headroom only breaks what would otherwise be a near
    tie. The cost of every tolerance is tabulated either way, so the
    choice is visible rather than buried in a weight.
    """
    banner("MATCHED PAIRS  (CHECK 5)")

    both = usable["fog"] & usable["colour"]
    only_one = (usable["fog"] ^ usable["colour"])
    pool = sorted(set(S.index) - set(EXCLUDE_SCENES))
    pool = [s for s in pool if s in both]
    if only_one:
        print(f"  {len(only_one)} scene(s) are usable for only one "
              "feature and are left out:")
        print("    " + ", ".join(sorted(only_one)[:6]))
        print("    A pair is assigned to a feature AFTER the matching, "
              "so a scene that can")
        print("    only take one role would constrain the matching "
              "and the assignment at once.")
    if EXCLUDE_SCENES:
        print(f"  {len(EXCLUDE_SCENES)} scene(s) excluded by setting")
    print(f"  pool: {len(pool)} scenes, need "
          f"{N_SETS * SET_SIZE} as {SET_SIZE} pairs")
    if len(pool) < N_SETS * SET_SIZE:
        raise RuntimeError(
            f"only {len(pool)} usable scenes but "
            f"{N_SETS * SET_SIZE} are needed. Lower MONOTONIC_FLOOR, "
            "drop a column, or shrink SET_SIZE -- the design cannot be "
            "built from this pool.")

    idx = {s: k for k, s in enumerate(pool)}
    z, feat_names = match_space(S, pool)
    Dm = pair_distances(z)
    rng = np.random.default_rng(RANDOM_SEED)

    forced = []
    for a, b in FORCE_PAIRS:
        for s in (a, b):
            if s not in idx:
                raise RuntimeError(
                    f"FORCE_PAIRS names {s!r}, which is not in the "
                    "usable pool (excluded, or it failed CHECK 3).")
        forced.append((idx[a], idx[b]))
    if forced:
        print(f"  {len(forced)} pair(s) forced by setting")

    all_i = list(range(len(pool)))
    print(f"\n  matching on {', '.join(feat_names)}, in corpus SD "
          f"units, over {MATCH_RESTARTS} restarts")

    # ---- stage 1: matching only ----
    p1, c1 = best_matching(Dm, all_i, SET_SIZE, forced, rng)
    if p1 is None:
        raise RuntimeError("no arrangement of the pool gives "
                           f"{SET_SIZE} disjoint pairs.")
    w1 = max(Dm[i, j] for i, j in p1)
    print(f"\n  stage 1 (headroom ignored): total "
          f"{c1:.4f} SD over {SET_SIZE} pairs, "
          f"mean {c1 / SET_SIZE:.4f}, worst pair {w1:.4f}")

    # ---- the headroom table, every gain x tolerance ----
    gain = HEADROOM_GAIN or (
        None if CONTRAST_UP_GAIN in (None, "undecided")
        else CONTRAST_UP_GAIN)
    tol_grid = [0.0, 0.0025, 0.005, 0.01, 0.02]
    if HEADROOM_TOLERANCE not in tol_grid:
        tol_grid = sorted(tol_grid + [HEADROOM_TOLERANCE])
    gains_present = [c.replace("damage_", "") for c in S.columns
                     if c.startswith("damage_")]
    hrows = []
    print("\n  HEADROOM COST  (matching total against the "
          "zero-headroom benchmark)")
    print(f"  {'gain':<9s} {'tol':>7s} {'eligible':>9s} "
          f"{'total SD':>9s} {'extra':>8s} {'worst':>7s}")
    for g in sorted(gains_present):
        for tol in tol_grid:
            ok = set(S.index[S["damage_" + g] <= tol]) & set(pool)
            elig = [idx[s] for s in pool if s in ok]
            row = {"gain": g, "tolerance": tol, "eligible": len(elig)}
            if len(elig) < N_SETS * SET_SIZE or any(
                    i not in elig or j not in elig for i, j in forced):
                row.update({"total_sd": np.nan, "extra_sd": np.nan,
                            "worst_pair_sd": np.nan,
                            "note": "too few eligible scenes"})
                print(f"  {g:<9s} {fmt_pct(tol):>7s} {len(elig):>9d} "
                      f"{'--':>9s} {'--':>8s} {'--':>7s}")
            else:
                p, c = best_matching(Dm, elig, SET_SIZE, forced, rng)
                w = max(Dm[i, j] for i, j in p)
                row.update({"total_sd": round(c, 5),
                            "extra_sd": round(c - c1, 5),
                            "worst_pair_sd": round(w, 5), "note": ""})
                print(f"  {g:<9s} {fmt_pct(tol):>7s} {len(elig):>9d} "
                      f"{c:>9.4f} {c - c1:>+8.4f} {w:>7.4f}")
            hrows.append(row)
    H = pd.DataFrame(hrows)
    H.to_csv(os.path.join(root, "headroom_cost.csv"), index=False)

    # ---- stage 2: the headroom preference, if it is cheap enough ----
    chosen, cost, note = p1, c1, "matching only (no headroom criterion)"
    if gain is None:
        print("\n  no contrast-up column is in play, so there is no "
              "headroom to prefer:")
        print("  the matching-only arrangement is taken.")
    else:
        ok = set(S.index[S["damage_" + gain] <= HEADROOM_TOLERANCE])
        elig = [idx[s] for s in pool if s in ok]
        print(f"\n  stage 2 (gain {gain}, tolerance "
              f"{fmt_pct(HEADROOM_TOLERANCE)}): "
              f"{len(elig)} of {len(pool)} scenes eligible")
        if len(elig) < N_SETS * SET_SIZE:
            print(f"  only {len(elig)} eligible, {N_SETS * SET_SIZE} "
                  "needed -- the headroom preference is DROPPED and "
                  "stage 1 is taken.")
            note = (f"headroom dropped: only {len(elig)} scenes meet "
                    f"{gain} at {fmt_pct(HEADROOM_TOLERANCE)}")
        else:
            p2, c2 = best_matching(Dm, elig, SET_SIZE, forced, rng)
            print(f"  total {c2:.4f} SD, which is {c2 - c1:+.4f} "
                  f"against stage 1 (budget {MATCH_SLACK:+.4f})")
            if c2 - c1 <= MATCH_SLACK:
                chosen, cost = p2, c2
                note = (f"headroom honoured: {gain} at "
                        f"{fmt_pct(HEADROOM_TOLERANCE)}, cost "
                        f"{c2 - c1:+.4f} SD")
                print("  within budget: the headroom-eligible "
                      "arrangement is taken.")
            else:
                note = (f"headroom dropped: {gain} at "
                        f"{fmt_pct(HEADROOM_TOLERANCE)} would cost "
                        f"{c2 - c1:+.4f} SD, over the "
                        f"{MATCH_SLACK:+.4f} budget")
                print("  over budget: the headroom preference is "
                      "DROPPED and stage 1 is taken.")
                print("  Matching is the primary criterion, so a "
                      "headroom gain that costs")
                print("  comparability is not taken.")

    print(f"\n  TAKEN: {note}")
    report.append(("matching", note))
    report.append(("matching total",
                   f"{cost:.4f} SD over {SET_SIZE} pairs "
                   f"(mean {cost / SET_SIZE:.4f})"))
    report.append(("matched on",
                   ", ".join(feat_names) + " (corpus SD units)"))

    pairs = [(pool[i], pool[j]) for i, j in chosen]
    return pairs, Dm, idx, H, cost, feat_names


# ============================================================
# Slots and sets  (CHECK 4, CHECK 5)
# ============================================================

def _zs(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


def assign_slots(pairs, S, Dm, idx, report):
    """Hand each pair a slot -- which feature, which screen, which row
    -- and decide which member goes to set A.

    THE PAIR IS BOUND TO THE SLOT. Both members sit on the same screen
    in the same row of their own set, so the two blocks are comparable
    row by row rather than only on average. That is the whole reason
    for matching pairs instead of matching set means.

    Which feature a pair takes is decided by its SPAN: the fog screen's
    columns differ in local contrast, so it gets the pairs whose
    contrast range is widest, and the colour screen gets the pairs
    whose saturation range is widest. A scene with a narrow range shows
    columns that look alike, which asks for a distinction the stimulus
    does not offer.
    """
    banner("SLOTS AND SETS  (CHECK 4)")
    rng = np.random.default_rng(RANDOM_SEED + 1)

    P = pd.DataFrame([{"pair": k, "a": a, "b": b,
                       "fog_span_sd": (S.loc[a, "fog_span_sd"]
                                       + S.loc[b, "fog_span_sd"]) / 2,
                       "colour_span_sd": (S.loc[a, "colour_span_sd"]
                                          + S.loc[b, "colour_span_sd"])
                                         / 2,
                       "distance_sd": Dm[idx[a], idx[b]]}
                      for k, (a, b) in enumerate(pairs)])

    # ---- which feature ----
    P["score"] = _zs(P.fog_span_sd) - _zs(P.colour_span_sd)
    order = P.sort_values("score", ascending=False).pair.tolist()
    n_each = SET_SIZE // 2
    role = {p: "fog" for p in order[:n_each]}
    role.update({p: "colour" for p in order[n_each:]})
    P["feature"] = P.pair.map(role)
    print(f"  {n_each} pairs to the fog screens, {n_each} to the "
          "colour screens, by span:")
    for feature in ("fog", "colour"):
        sub = P[P.feature == feature]
        print(f"    {feature:<7s} fog span "
              f"{sub.fog_span_sd.mean():.2f} SD, colour span "
              f"{sub.colour_span_sd.mean():.2f} SD")

    # ---- which screen, and which row ----
    #
    # Each feature's pairs split into its two screens. The split is
    # chosen so the two screens carry a similar range -- otherwise the
    # forward and reversed asking of the same feature would differ in
    # how easy it is, and the stability comparison between them would
    # confound order with difficulty.
    assign = {}
    for feature in ("fog", "colour"):
        sub = P[P.feature == feature]
        span_col = f"{feature}_span_sd"
        ids = sub.pair.tolist()
        spans = dict(zip(sub.pair, sub[span_col]))
        groups = {p: {S.loc[sub.set_index('pair').loc[p, 'a'], "group"],
                      S.loc[sub.set_index('pair').loc[p, 'b'], "group"]}
                  for p in ids}
        best, best_key = None, None
        for combo in itertools.combinations(ids, ROWS_PER_SCREEN):
            rest = tuple(p for p in ids if p not in combo)
            if combo[0] != ids[0]:
                continue          # each split once, not twice
            d = abs(np.mean([spans[p] for p in combo])
                    - np.mean([spans[p] for p in rest]))
            div = 0
            if BALANCE_GROUPS:
                for side in (combo, rest):
                    seen = set()
                    for p in side:
                        seen |= groups[p]
                    div += (ROWS_PER_SCREEN - min(len(seen),
                                                  ROWS_PER_SCREEN))
            key = (div, d)
            if best_key is None or key < best_key:
                best, best_key = (combo, rest), key
        trios = list(best)
        rng.shuffle(trios)        # no systematic link to column order
        screens = [(i, f, o) for i, (f, o) in enumerate(SCREENS, 1)
                   if f == feature]
        for (sidx, f, o), trio in zip(screens, trios):
            trio = sorted(trio, key=lambda p: -spans[p])
            for row, p in enumerate(trio, 1):
                assign[p] = {"screen": sidx, "column_order": o,
                             "row": row}
        print(f"    {feature}: screens "
              + " | ".join(
                  f"{sidx} ({o}) spans "
                  + ", ".join(f"{spans[p]:.2f}" for p in
                              sorted([q for q in assign
                                      if assign[q]['screen'] == sidx],
                                     key=lambda q: assign[q]['row']))
                  for sidx, f2, o in screens))

    for c in ("screen", "column_order", "row"):
        P[c] = P.pair.map(lambda p: assign[p][c])

    # ---- which member goes to set A ----
    #
    # A seeded flip would do, but a greedy balancing pass costs
    # nothing and keeps the two sets' MEANS close as well as their
    # pairs, so the set-level summary in the write-up does not need a
    # caveat. Indoor/outdoor is the tie-break.
    run = {f: 0.0 for f, _ in MATCH_FEATURES}
    run_indoor = 0
    rows = []
    for p in rng.permutation(P.pair.to_numpy()):
        r = P[P.pair == p].iloc[0]
        best, best_key = None, None
        for first, second in ((r.a, r.b), (r.b, r.a)):
            d = sum(abs(run[f] + (S.loc[first, f] - S.loc[second, f])
                        / sd) for f, sd in MATCH_FEATURES)
            ind = run_indoor + (
                (1 if S.loc[first, "indoor"] == "indoor" else 0)
                - (1 if S.loc[second, "indoor"] == "indoor" else 0))
            key = (d, abs(ind))
            if best_key is None or key < best_key:
                best, best_key = (first, second), key
        a, b = best
        for f, sd in MATCH_FEATURES:
            run[f] += (S.loc[a, f] - S.loc[b, f]) / sd
        run_indoor += ((1 if S.loc[a, "indoor"] == "indoor" else 0)
                       - (1 if S.loc[b, "indoor"] == "indoor" else 0))
        rows.append({"pair": int(p), "set_A": a, "set_B": b})
    A = pd.DataFrame(rows).set_index("pair")
    P = P.set_index("pair").join(A).reset_index()
    P["set_A_group"] = P.set_A.map(S.group)
    P["set_B_group"] = P.set_B.map(S.group)
    for f, sd in MATCH_FEATURES:
        P[f"set_A_{f}"] = P.set_A.map(S[f])
        P[f"set_B_{f}"] = P.set_B.map(S[f])

    # ---- CHECK 4 ----
    a_set, b_set = set(P.set_A), set(P.set_B)
    problems = []
    if a_set & b_set:
        problems.append(f"{sorted(a_set & b_set)} is in both sets")
    if len(a_set) != SET_SIZE or len(b_set) != SET_SIZE:
        problems.append(f"sets have {len(a_set)} and {len(b_set)} "
                        f"scenes; {SET_SIZE} expected")
    slots = sorted((r.screen, r.row) for r in P.itertuples())
    want = sorted((i, row) for i, _ in enumerate(SCREENS, 1)
                  for row in range(1, ROWS_PER_SCREEN + 1))
    if slots != want:
        problems.append("the slots are not exactly one pair per "
                        f"(screen, row): {slots}")
    if problems:
        raise RuntimeError("CHECK 4 failed:\n  - "
                           + "\n  - ".join(problems))
    print(f"\n  CHECK 4 PASS: the two sets are disjoint, "
          f"{SET_SIZE} scenes each, one pair per slot.")
    print("  No photograph is in both blocks for any participant, "
          "whichever way the sets swap.")
    report.append(("check 4 sets disjoint", "PASS"))

    # ---- CHECK 5 ----
    print("\n  CHECK 5  within-pair distances and set means")
    print(f"    worst pair  {P.distance_sd.max():.4f} SD")
    print(f"    median      {P.distance_sd.median():.4f} SD")
    worst = []
    for f, sd in MATCH_FEATURES:
        da = P[f"set_A_{f}"].mean()
        db = P[f"set_B_{f}"].mean()
        diff = abs(da - db) / sd
        worst.append(diff)
        flag = "  OVER TOL" if diff > MATCH_REPORT_TOL else ""
        print(f"    {f:<9s} set A {da:8.3f}   set B {db:8.3f}   "
              f"diff {diff:.4f} SD{flag}")
    ia = (P.set_A.map(S.indoor) == "indoor").sum()
    ib = (P.set_B.map(S.indoor) == "indoor").sum()
    print(f"    indoor      set A {ia}/{SET_SIZE}   set B "
          f"{ib}/{SET_SIZE}")
    if max(worst) > MATCH_REPORT_TOL:
        print(f"    WARNING: a set mean differs by more than "
              f"{MATCH_REPORT_TOL} SD. The pairs are what the design "
              "rests on,")
        print("    so this is reported rather than fixed, but it is "
              "worth a look at pairs.csv.")
        report.append(("check 5 matching",
                       f"PASS with a warning (worst set-mean diff "
                       f"{max(worst):.4f} SD)"))
    else:
        report.append(("check 5 matching",
                       f"PASS (worst pair {P.distance_sd.max():.4f} SD, "
                       f"worst set-mean diff {max(worst):.4f} SD)"))
    report.append(("within-pair distance",
                   f"median {P.distance_sd.median():.4f} SD, worst "
                   f"{P.distance_sd.max():.4f} SD"))
    return P


# ============================================================
# The screens, resolved
# ============================================================

def build_screens(P, S, G, columns, report):
    """One row per (set, screen, row, column): the scene, the cell, the
    file, the measured feature value, and the signed level index.

    This is keyed on the SET, not the block. Which set is the episode
    block and which is the normal block is a per-participant draw, so
    binding a file to a block here would be wrong.
    """
    banner("THE SCREENS")
    rows = []
    for r in P.itertuples():
        feature = r.feature
        cols = columns[feature]
        feat = feature_column(feature)
        order = (cols if r.column_order == "forward" else cols[::-1])
        for which, scene in (("A", r.set_A), ("B", r.set_B)):
            base = S.loc[scene, "base_fog" if feature == "fog"
                         else "base_sat"]
            for disp, c in enumerate(order, 1):
                m = cell_row(G, scene, c)
                val = float(m[feat])
                rows.append({
                    "set": which, "screen": int(r.screen),
                    "feature": feature, "column_order": r.column_order,
                    "row": int(r.row), "display_column": disp,
                    "column_label": c["column_label"],
                    "level_index": c["level_index"],
                    "is_original": c["is_original"],
                    "pair": int(r.pair), "scene_id": scene,
                    "category": S.loc[scene, "category"],
                    "group": S.loc[scene, "group"],
                    "indoor": S.loc[scene, "indoor"],
                    "kind": c["kind"],
                    "colour_level": c["colour_level"],
                    "fog_level": c["fog_level"],
                    "contrast_level": c["contrast_level"] or "",
                    "chroma": c["chroma"], "haze": c["haze"],
                    "gain": c["gain"],
                    "filename": m.filename,
                    "sat_mean": float(m.sat_mean),
                    "contrast_local": float(m.contrast_local),
                    feat + "_from_original": val - float(base),
                    "sd_from_original": (val - float(base)) / (
                        SD_FOG if feature == "fog" else SD_SAT),
                })
    X = pd.DataFrame(rows)
    n_img = X.filename.nunique()
    print(f"  {len(X)} cells over {N_SETS} sets x {len(SCREENS)} "
          f"screens x {ROWS_PER_SCREEN} rows")
    print(f"  {n_img} distinct images to ship")
    for feature in ("fog", "colour"):
        sub = X[X.feature == feature]
        print(f"  {feature:<7s} range from the original: "
              f"{sub.sd_from_original.min():+.2f} to "
              f"{sub.sd_from_original.max():+.2f} SD")
    report.append(("cells on screens", str(len(X))))
    report.append(("images to ship", str(n_img)))
    for feature in ("fog", "colour"):
        sub = X[X.feature == feature]
        report.append((f"{feature} range on the screens",
                       f"{sub.sd_from_original.min():+.2f} to "
                       f"{sub.sd_from_original.max():+.2f} SD"))
    return X


def unused_cells(M, columns, root, report):
    """Which rendered grid cells this block does not show, and why.

    Written as a file of its own as well as into the report, because
    step 9's documentation needs a source for it rather than someone's
    memory of a conversation. A cell with no recorded reason is a
    warning: an unexplained image in the stimulus set is exactly the
    thing that later looks like a mistake.
    """
    banner("RENDERED BUT NOT USED")

    used = set()
    for feature, cols in columns.items():
        for c in cols:
            used.add((c["kind"], c["colour_level"], c["fog_level"],
                      c["contrast_level"] or ""))

    G = M[M.use.isin(("grid", "both"))]
    key = ["kind", "colour_level", "fog_level", "contrast_level"]
    cells = (G[key].fillna("").drop_duplicates()
             .sort_values(key).reset_index(drop=True))

    rows, unexplained = [], []
    for c in cells.itertuples(index=False):
        t = (c.kind, c.colour_level, c.fog_level, c.contrast_level)
        if t in used:
            continue
        label = ("contrast_up" if c.kind == "contrast_up"
                 else f"{c.colour_level}@{c.fog_level}")
        n = len(G[(G.kind == c.kind)
                  & (G.colour_level == c.colour_level)
                  & (G.fog_level == c.fog_level)
                  & (G.contrast_level.fillna("") == c.contrast_level)])
        note = UNUSED_CELL_NOTES.get(label, "")
        if not note:
            unexplained.append(label)
        rows.append({"label": label, "kind": c.kind,
                     "colour_level": c.colour_level,
                     "fog_level": c.fog_level,
                     "contrast_level": c.contrast_level,
                     "images": n, "reason": note})

    U = pd.DataFrame(rows)
    if not len(U):
        print("  every rendered grid cell is on a screen.")
        report.append(("rendered but not used", "none"))
        return U

    by = U.groupby("label", as_index=False).agg(
        images=("images", "sum"), cells=("label", "size"))
    print(f"  {int(U.images.sum())} image(s) in {len(U)} cell(s) are "
          "rendered and not shown:")
    for r in by.itertuples():
        print(f"    {r.label:<14s} {r.images:4d} images "
              f"({r.cells} cell{'s' if r.cells > 1 else ''})")
    if unexplained:
        print(f"\n  WARNING: no recorded reason for "
              f"{', '.join(sorted(set(unexplained)))}.")
        print("  Add it to UNUSED_CELL_NOTES. An image in the "
              "stimulus set that no document")
        print("  accounts for is indistinguishable from a mistake "
              "later.")
        report.append(("rendered but not used",
                       f"{int(U.images.sum())} images, "
                       f"{len(set(unexplained))} UNEXPLAINED"))
    else:
        print("\n  every one has a recorded reason.")
        report.append(("rendered but not used",
                       f"{int(U.images.sum())} images, all explained"))

    U.to_csv(os.path.join(root, "unused_cells.csv"), index=False)
    path = os.path.join(root, "unused_cells.md")
    with open(path, "w") as f:
        W = f.write
        W("# Rendered but not used: the grid block\n\n")
        W(f"From `{STIMULUS_ROOT}`, built by step 3. Step 7b shows a "
          "subset of the cells step 3\nrenders; this is the rest, "
          "with the reason each one is not shown. **Nothing was "
          "deleted\nor re-rendered** -- the images are still in the "
          "stimulus set and still carry their\nmeasured features.\n\n")
        for r in by.itertuples():
            W(f"## `{r.label}` -- {r.images} images\n\n")
            W(UNUSED_CELL_NOTES.get(r.label,
                                    "**No reason recorded.**") + "\n\n")
        W("## What the grid does show\n\n")
        for feature, cols in columns.items():
            W(f"- **{feature}**: "
              + ", ".join(f"`{c['column_label']}`" for c in cols)
              + f" ({len(cols)} columns)\n")
        W("\nThe pair trials are unaffected by all of this: every "
          "cell named above is\n`use = \"grid\"` in the manifest and "
          "was never in the pair inventory.\n")
    print(f"  {path}")
    return U


def censoring(columns):
    """Which end of each feature's axis has no column beyond it.

    A participant whose experience lies past the end of a screen has
    to answer on the end column, so that answer is a BOUND, not a
    location. With the contrast-up column dropped the fog screen stops
    at the original, which means every fog answer on the original is
    censored from above -- and in the normal-appearance block that is
    the common case rather than the edge case, because most people's
    normal vision is not foggy.
    """
    out = {}
    for feature, cols in columns.items():
        idx = [c["level_index"] for c in cols]
        out[feature] = {
            "above": not any(i < 0 for i in idx),
            "below": not any(i > 0 for i in idx),
            "top": [c["column_label"] for c in cols
                    if c["level_index"] == min(idx)][0],
            "bottom": [c["column_label"] for c in cols
                       if c["level_index"] == max(idx)][0]}
    return out


def report_censoring(columns, report):
    banner("CENSORING")
    cen = censoring(columns)
    # "Censored" here means censored AT THE ORIGINAL -- no column on
    # one side of the scene's own appearance -- which is the case that
    # bites, because the original is where the normal-appearance block
    # puts most people. Every ladder is also censored at its extreme
    # ends (an episode foggier than p05 lands on p05); that is
    # ordinary and is noted separately below.
    for feature, c in cen.items():
        ends = []
        if c["above"]:
            ends.append(f"ABOVE the original (nothing clearer / more "
                        f"vivid than {c['top']!r})")
        if c["below"]:
            ends.append(f"below the original (nothing beyond "
                        f"{c['bottom']!r})")
        print(f"  {feature:<7s} " + ("censored " + " and ".join(ends)
                                     if ends else
                                     "has columns on BOTH sides of "
                                     "the original"))
    print("\n  Separately, and ordinarily: every ladder stops "
          "somewhere, so an experience")
    print("  beyond the last column lands on it. The fog screen ends "
          f"at {cen['fog']['bottom']!r} and the")
    print(f"  colour screen at {cen['colour']['bottom']!r}. That "
          "affects the most severe episodes; the")
    print("  censoring AT THE ORIGINAL above affects the NORMAL "
          "block, which is a different")
    print("  and larger problem.")
    if cen["fog"]["above"]:
        print("\n  THE FOG AXIS IS CENSORED AT THE ORIGINAL. The fog "
              "screen has no column")
        print("  clearer than the photograph as taken, so a "
              "participant whose vision is")
        print("  clearer than that answers on the original column.")
        print("  In the NORMAL-APPEARANCE block this is the common "
              "case, not the edge case:")
        print("  most people's normal vision is not foggy. Every fog "
              "answer on the original")
        print("  column is therefore a LOWER BOUND on that person's "
              "normal, and the")
        print("  normal-to-episode shift computed from it is a lower "
              "bound on the shift.")
        print("  The bias is conservative -- it can only shrink the "
              "shift -- but it must be")
        print("  declared, and the COUNT of such answers in each "
              "block reported alongside.")
    report.append(("fog axis",
                   "censored at the original (no column above); fog "
                   "answers there are lower bounds, count per block "
                   "to be reported"
                   if cen["fog"]["above"] else "open above"))
    report.append(("colour axis",
                   "censored above" if cen["colour"]["above"]
                   else f"open above ({cen['colour']['top']})"))
    return cen


def pair_reuse(X, report, root):
    """How often each grid scene was already shown in a pair trial.

    v2 drew grid scenes from the pair pool deliberately: it is what
    lets a grid placement be compared with the same person's pair
    choices. The cost is familiarity, and a scene that happens to
    appear in many pair trials carries more of it, so the counts are
    reported per scene rather than assumed even.
    """
    if not TRIAL_LIST_CSV or not os.path.isfile(TRIAL_LIST_CSV):
        print("\n  no trial list given; the reuse report is skipped.")
        report.append(("pair reuse", "not checked (no trial list)"))
        return None
    banner("REUSE FROM THE PAIR TRIALS")
    T = pd.read_csv(TRIAL_LIST_CSV)
    cols = [c for c in T.columns if "scene" in c.lower()]
    if not cols:
        print(f"  {TRIAL_LIST_CSV} has no scene column; skipped.")
        report.append(("pair reuse", "not checked (no scene column)"))
        return None
    seen = pd.concat([T[c] for c in cols]).value_counts()
    scenes = sorted(X.scene_id.unique())
    R = pd.DataFrame({"scene_id": scenes})
    R["set"] = R.scene_id.map(
        X.drop_duplicates("scene_id").set_index("scene_id")["set"])
    R["feature"] = R.scene_id.map(
        X.drop_duplicates("scene_id").set_index("scene_id").feature)
    R["pair_trials"] = R.scene_id.map(seen).fillna(0).astype(int)
    R = R.sort_values("pair_trials", ascending=False)
    R.to_csv(os.path.join(root, "pair_reuse.csv"), index=False)
    print(f"  trial list: {len(T)} rows, scene columns "
          f"{', '.join(cols)}")
    print(f"  grid scenes appear in {R.pair_trials.min()}-"
          f"{R.pair_trials.max()} pair trials "
          f"(median {int(R.pair_trials.median())})")
    print("  most reused: " + ", ".join(
        f"{r.scene_id} ({r.pair_trials})" for r in R.head(4).itertuples()))
    never = R[R.pair_trials == 0]
    if len(never):
        print(f"  {len(never)} grid scene(s) never appear in a pair "
              "trial, so for those the grid")
        print("  placement cannot be compared with the same person's "
              "pair choices.")
    for which in ("A", "B"):
        sub = R[R["set"] == which]
        print(f"  set {which}: median {int(sub.pair_trials.median())}, "
              f"total {int(sub.pair_trials.sum())}")
    report.append(("pair reuse",
                   f"{R.pair_trials.min()}-{R.pair_trials.max()} trials "
                   f"per grid scene (median "
                   f"{int(R.pair_trials.median())})"))
    return R


# ============================================================
# Composition
# ============================================================

SERIES_COLOURS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
SERIES_MARKERS = ["o", "s", "^", "D"]
SERIES_STYLES = ["-", "--", "-.", ":"]

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]


def _font(size):
    for p in _FONT_PATHS:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _load_cell(filename, px):
    """One cell at the display size, scaled the way a browser would.

    LANCZOS rather than BILINEAR on purpose: it is the closer stand-in
    for what a browser does when it scales an image down, and the
    question the mock-up answers is a question about what the
    participant sees, not about what was measured.
    """
    with Image.open(os.path.join(STIMULI_DIR, filename)) as im:
        return im.convert("RGB").resize((px, px), Image.LANCZOS)


def compose_screen(cells, px, labels, title, subtitle,
                   inner_w=None, inner_h=None):
    """One grid screen as the participant will see it.

    THE CANVAS IS THE WINDOW. When `inner_w` and `inner_h` are given,
    the lower part of the image is exactly the browser's inner area,
    with the instruction and button space the layout reserves
    (CHROME_PX) blocked out and the grid placed in what is left. An
    earlier version drew only the strip the cells occupy, which made
    every mock-up look comfortably full whatever the window -- and the
    slack, or the lack of it, is half of what there is to see.

    The annotation bar and the label strip are drawn ABOVE that area
    and marked, so nothing a reader is told is ever inside the picture
    of the screen.

    `cells` is a list of rows, each a list of (filename, label) in
    DISPLAY order.
    """
    rows = len(cells)
    cols = len(cells[0])
    # A COLUMN takes px plus its CSS border and padding; a ROW takes px
    # alone. That asymmetry is not an oversight here -- it is what the
    # experiment's own arithmetic does: byWidth subtracts
    # CELL_CHROME_PX per column and byHeight subtracts nothing per row.
    # Drawing it any other way would make the mock-up disagree with the
    # page it is a mock-up of. See the note in layout_table.
    cell_total = px + CELL_CHROME_PX
    grid_w = cols * cell_total + (cols - 1) * COL_GAP_PX
    grid_h = rows * px + (rows - 1) * ROW_GAP_PX

    win_w = inner_w if inner_w else grid_w + PAGE_SIDE_PX
    win_h = inner_h if inner_h else grid_h + CHROME_PX
    head_h, lab_h = 54, (34 if labels else 0)

    img = Image.new("RGB", (win_w, head_h + lab_h + win_h), PAGE_BG)
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, win_w, head_h], fill=(26, 26, 26))
    d.text((12, 7), title, font=_font(17), fill=(255, 255, 255))
    d.text((12, 31), subtitle, font=_font(13), fill=(185, 185, 185))

    y_win = head_h + lab_h
    if labels:
        d.rectangle([0, head_h, win_w, y_win], fill=(56, 56, 56))
        d.text((6, head_h + 2),
               "the strip below is an annotation -- it is NOT on the "
               "screen", font=_font(10), fill=(148, 148, 148))

    # the instruction and button space the layout reserves
    instr_h = int(round(CHROME_PX * 0.6))
    d.rectangle([0, y_win, win_w, y_win + instr_h],
                fill=(112, 112, 112))
    d.text((PAGE_SIDE_PX // 2, y_win + instr_h // 2 - 7),
           "[ instructions and the question go here ]",
           font=_font(13), fill=(214, 214, 214))
    btn_y = y_win + win_h - (CHROME_PX - instr_h)
    d.rectangle([0, btn_y, win_w, head_h + lab_h + win_h],
                fill=(112, 112, 112))
    d.text((PAGE_SIDE_PX // 2, btn_y + 10),
           "[ 'none of these' / continue ]", font=_font(13),
           fill=(214, 214, 214))

    # the grid, centred in what is left
    x0 = max(PAGE_SIDE_PX // 2, (win_w - grid_w) // 2)
    space = btn_y - (y_win + instr_h)
    y0 = y_win + instr_h + max(0, (space - grid_h) // 2)

    if labels:
        f = _font(12)
        for c in range(cols):
            x = x0 + c * (cell_total + COL_GAP_PX)
            d.text((x + 2, head_h + 16), labels[c], font=f,
                   fill=(236, 236, 236))

    pad = CELL_CHROME_PX // 2
    for r, row in enumerate(cells):
        y = y0 + r * (px + ROW_GAP_PX)
        for c, (fn, _) in enumerate(row):
            x = x0 + c * (cell_total + COL_GAP_PX)
            d.rectangle([x, y - pad, x + cell_total - 1, y + px - 1 + pad],
                        fill=CELL_BORDER_RGB)
            img.paste(_load_cell(fn, px), (x + pad, y))

    if grid_w > win_w - PAGE_SIDE_PX or grid_h > space:
        d.text((12, y_win + instr_h + 2),
               "OVERFLOW: the grid does not fit this window",
               font=_font(14), fill=(255, 90, 90))
    return img


def mockup(X, S, G, report, root):
    """A screen per window per candidate gain, at the cell size the
    experiment will compute.

    This is the figure the contrast-up decision is made from. The
    number that matters is in the subtitle: at seven columns the cell
    is about 149 px on a 1366-wide window, not the 240 px maximum, and
    240 px is the size step 3's contact sheets were drawn at. A gain
    that is visible on a contact sheet and invisible here is not
    visible.

    One gain per screen. All three at once would be nine columns and
    about 112 px a cell, which is not a layout anyone will see, so a
    judgement made from it would be a judgement at the wrong size.
    """
    banner("MOCK-UP")
    mdir = os.path.join(root, "mockup")
    os.makedirs(mdir, exist_ok=True)

    variants = []
    if MODE == "mockup":
        for g in MOCKUP_GAINS:
            variants.append(("fog", g))
        if MOCKUP_INCLUDE_NO_UP:
            variants.append(("fog", None))
    else:
        variants.append(("fog", effective_up()))
    variants.append(("colour", None))

    written = []
    for label, iw, ih in WINDOWS:
        for feature, up in variants:
            cols = resolve_columns(feature, up=up)
            other = len(columns_for_other(feature))
            px, bw, bh, binds = file_cell_px(iw, ih, len(cols), other)
            # the forward screen of this feature, set A
            sub = X[(X.feature == feature) & (X["set"] == "A")
                    & (X.column_order == "forward")]
            if not len(sub):
                continue
            screen = int(sub.screen.iloc[0])
            scenes = (sub.drop_duplicates("row")
                      .sort_values("row").scene_id.tolist())
            feat = feature_column(feature)
            cells, lab = [], None
            for sid in scenes:
                row = []
                for c in cols:
                    m = cell_row(G, sid, c)
                    row.append((m.filename, c["label"]))
                cells.append(row)
            base_col = [c for c in cols if c["is_original"]][0]
            lab = []
            for c in cols:
                vals = [float(cell_row(G, sid, c)[feat])
                        for sid in scenes]
                b = [float(cell_row(G, sid, base_col)[feat])
                     for sid in scenes]
                sd = SD_FOG if feature == "fog" else SD_SAT
                mean_sd = np.mean([(v - bb) / sd
                                   for v, bb in zip(vals, b)])
                lab.append(f"{c['label']}  {mean_sd:+.2f}SD")

            tag = (up if up else "no-up") if feature == "fog" else "all"
            name = f"mockup_{label}_{feature}_{tag}.png"
            title = (f"{feature} screen, forward  --  {len(cols)} "
                     f"columns x {ROWS_PER_SCREEN} rows")
            subtitle = (
                f"{label} window (inner {iw}x{ih})   cell "
                f"{px} px   byWidth {bw:.0f}, byHeight {bh:.0f}, "
                f"binds on {binds}   set A, screen {screen}")
            img = compose_screen(cells, px,
                                 lab if MOCKUP_LABELS else None,
                                 title, subtitle,
                                 inner_w=iw, inner_h=ih)
            img.save(os.path.join(mdir, name))
            written.append({"file": name, "window": label,
                            "feature": feature, "up": tag,
                            "cols": len(cols), "cell_px": px,
                            "binds": binds})
            print(f"  {name:<44s} {len(cols)} cols, {px} px/cell "
                  f"({binds} binds)")

    # ---- the up-column detail ----
    if MOCKUP_GAINS and MODE == "mockup":
        detail = up_detail(S, G, mdir, report)
    else:
        detail = None

    W = pd.DataFrame(written)
    W.to_csv(os.path.join(root, "mockup_index.csv"), index=False)
    report.append(("mock-up screens", str(len(written))))
    print(f"\n  {len(written)} screen(s) in {mdir}")
    print("  WHAT TO LOOK FOR: is the up column distinguishable from "
          "the column next to")
    print("  it, at this size, on these scenes? If it is not, set "
          "CONTRAST_UP_GAIN = None.")
    print("  An invisible column does not reduce the ceiling effect; "
          "it splits the people")
    print("  at the ceiling over two columns that look the same.")
    return W, detail


def up_detail(S, G, mdir, report):
    """The original beside all three gains, at the real cell size, for
    scenes spread across the damage range.

    The screen mock-ups answer "is it distinguishable in context". This
    answers the prior question -- "is it distinguishable at all" -- and
    it picks the scenes rather than taking whichever ones the selection
    happened to choose, so the answer is not luck of the draw. The
    scenes are ordered by how much damage the strongest gain does to
    them, because that is where the gain both shows most and costs
    most.
    """
    dcol = "damage_" + MOCKUP_GAINS[-1]
    if dcol not in S.columns:
        return None
    order = S.sort_values(dcol, ascending=False).index.tolist()
    n = min(MOCKUP_DETAIL_N, len(order))
    take = [order[int(round(k * (len(order) - 1) / max(n - 1, 1)))]
            for k in range(n)]

    cols = [{"kind": "grid",
             "colour_level": getattr(S3, "COLOUR_IDENTITY_LABEL", "k=1"),
             "fog_level": getattr(S3, "FOG_IDENTITY_LABEL", "h=0"),
             "contrast_level": None, "label": "original"}]
    for g in MOCKUP_GAINS:
        cols.append({"kind": "contrast_up",
                     "colour_level": getattr(
                         S3, "COLOUR_IDENTITY_LABEL", "k=1"),
                     "fog_level": getattr(
                         S3, "FOG_IDENTITY_LABEL", "h=0"),
                     "contrast_level": g, "label": g})

    out = []
    for label, iw, ih in WINDOWS:
        # the size a FOG screen cell takes in this window, which is
        # the size the column will actually be seen at
        ncol = len(resolve_columns("fog", up=MOCKUP_GAINS[0]))
        px, _, _, _ = file_cell_px(iw, ih, ncol,
                                   len(resolve_columns("colour")))
        cells = []
        for sid in take:
            cells.append([(cell_row(G, sid, c).filename, c["label"])
                          for c in cols])
        lab = []
        for c in cols:
            ds = []
            for sid in take:
                v = float(cell_row(G, sid, c).contrast_local)
                b = float(S.loc[sid, "base_fog"])
                ds.append((v - b) / SD_FOG)
            dmg = (0.0 if c["contrast_level"] is None
                   else float(S.loc[take, "damage_"
                                    + c["contrast_level"]].mean()))
            lab.append(f"{c['label']}  {np.mean(ds):+.2f}SD  "
                       f"dmg {fmt_pct(dmg)}")
        name = f"up_column_detail_{label}.png"
        # NOT drawn in a window: this is a detail strip, not a
        # screen, and dressing it as one would invite reading the
        # layout off it.
        img = compose_screen(
            cells, px, lab,
            "contrast-up detail: the original beside every candidate "
            "gain",
            f"{label} window, {px} px per cell (the fog screen's own "
            f"cell size at {ncol} columns); {n} scenes, most damaged "
            "at the top")
        img.save(os.path.join(mdir, name))
        out.append(name)
        print(f"  {name:<44s} {len(cols)} cols, {px} px/cell")
    report.append(("up-column detail", ", ".join(out)))
    return out


# ============================================================
# Figures
# ============================================================

def plot_selection(S, P, root):
    """The pool, the pairs and the two sets, in the space the matching
    works in."""
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.scatter(S.base_sat / SD_SAT, S.base_fog / SD_FOG, s=26,
               facecolor="none", edgecolor="#9a9a9a", linewidth=0.9,
               label=f"pool ({len(S)})", zorder=2)
    for r in P.itertuples():
        xa, ya = S.loc[r.set_A, "base_sat"] / SD_SAT, \
            S.loc[r.set_A, "base_fog"] / SD_FOG
        xb, yb = S.loc[r.set_B, "base_sat"] / SD_SAT, \
            S.loc[r.set_B, "base_fog"] / SD_FOG
        ax.plot([xa, xb], [ya, yb], color="#555555", linewidth=0.8,
                zorder=3)
    for k, (which, col, mk) in enumerate((("A", SERIES_COLOURS[0], "o"),
                                          ("B", SERIES_COLOURS[1], "s"))):
        for feature, fill in (("fog", True), ("colour", False)):
            sub = P[P.feature == feature]
            ids = sub[f"set_{which}"]
            ax.scatter(S.loc[ids, "base_sat"] / SD_SAT,
                       S.loc[ids, "base_fog"] / SD_FOG,
                       s=70, marker=mk, zorder=4,
                       color=col if fill else "white",
                       edgecolor=col, linewidth=1.6,
                       label=(f"set {which}, {feature} screens"))
    ax.set_xlabel("baseline sat_mean  (corpus SD)")
    ax.set_ylabel("baseline contrast_local  (corpus SD)")
    ax.set_title("Grid scenes: 12 matched pairs, split into two sets\n"
                 "a line joins the two members of a pair; filled = fog "
                 "screens, open = colour screens", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(root, "selection.png"), dpi=FIG_DPI)
    plt.close(fig)


def plot_headroom(H, root):
    """What the headroom preference costs the matching, per gain."""
    if H is None or not len(H):
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    gains = sorted(H.gain.unique())
    if len(gains) > len(SERIES_COLOURS):
        raise RuntimeError(
            f"{len(gains)} gains but only {len(SERIES_COLOURS)} "
            "colours are defined. Add to SERIES_COLOURS rather than "
            "letting matplotlib invent one.")
    for i, g in enumerate(gains):
        sub = H[H.gain == g].sort_values("tolerance")
        ax.plot(100 * sub.tolerance, sub.extra_sd,
                color=SERIES_COLOURS[i], marker=SERIES_MARKERS[i],
                linestyle=SERIES_STYLES[i], label=g)
        ax2.plot(100 * sub.tolerance, sub.eligible,
                 color=SERIES_COLOURS[i], marker=SERIES_MARKERS[i],
                 linestyle=SERIES_STYLES[i], label=g)
    ax.axhline(MATCH_SLACK, color="#444444", linewidth=1.0,
               linestyle=":")
    ax.annotate(f"MATCH_SLACK = {MATCH_SLACK}", (0.02, MATCH_SLACK),
                xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("headroom tolerance: flattened + crushed (% of pixels)")
    ax.set_ylabel("extra matching distance (SD, total over the pairs)")
    ax.set_title("cost of the headroom preference", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax2.axhline(N_SETS * SET_SIZE, color="#444444", linewidth=1.0,
                linestyle=":")
    ax2.annotate(f"{N_SETS * SET_SIZE} needed",
                 (0.02, N_SETS * SET_SIZE), xytext=(4, 4),
                 textcoords="offset points", fontsize=8)
    ax2.set_xlabel("headroom tolerance (% of pixels)")
    ax2.set_ylabel("scenes eligible")
    ax2.set_title("how many scenes clear it", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "headroom.png"), dpi=FIG_DPI)
    plt.close(fig)


def plot_ladders(X, root):
    """Where each selected row's columns actually sit, in SD from that
    scene's own original -- the quantity the analysis uses."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax, feature in zip(axes, ("fog", "colour")):
        sub = X[(X.feature == feature)
                & (X.column_order == "forward")]
        for i, (which, col) in enumerate((("A", SERIES_COLOURS[0]),
                                          ("B", SERIES_COLOURS[1]))):
            s2 = sub[sub["set"] == which]
            for sid, g in s2.groupby("scene_id"):
                g = g.sort_values("level_index")
                ax.plot(g.level_index, g.sd_from_original,
                        color=col, alpha=0.55, linewidth=1.1,
                        marker=SERIES_MARKERS[i], markersize=3.5,
                        linestyle=SERIES_STYLES[i])
            ax.plot([], [], color=col, marker=SERIES_MARKERS[i],
                    linestyle=SERIES_STYLES[i], label=f"set {which}")
        ax.axhline(0, color="#444444", linewidth=0.9)
        ax.axvline(0, color="#444444", linewidth=0.9, linestyle=":")
        ax.set_xlabel("level index  (0 = the scene's own original)")
        ax.set_ylabel(f"{feature_column(feature)}  "
                      "(SD from the original)")
        ax.set_title(f"{feature} screen: every selected row",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("The columns in feature units. One step is not a "
                 "constant amount of anything,\nwhich is why the "
                 "normal-to-episode shift is measured here and not in "
                 "steps.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(root, "ladders.png"), dpi=FIG_DPI)
    plt.close(fig)


# ============================================================
# The package  (CHECK 6)
# ============================================================

PACKAGE_VERSION = "v4-grid-1"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ship_images(X, root, report):
    """Copy every image a screen needs into the package.

    THE ORIGINAL FILENAME IS KEPT. v2 renamed each file by its
    position, because the experiment there built the grid from the
    names; here the layout is in grid_trials.json, so renaming would
    only break the one thing worth having -- a shipped file whose name
    says which cell of which scene it is, and which can therefore be
    hashed against the stimulus set.

    At SHIP_LONG_EDGE = 0 the file is copied byte for byte. At 256 it
    is re-encoded at the measurement size, which is a smaller package
    and a weaker claim; CHECK 6 tells the two apart.
    """
    banner("SHIPPING")
    pdir = os.path.join(root, "package")
    sdir = os.path.join(pdir, "stimuli")
    os.makedirs(sdir, exist_ok=True)

    files = sorted(X.filename.unique())
    rows = []
    t0 = time.time()
    for fn in files:
        src = os.path.join(STIMULI_DIR, fn)
        dst = os.path.join(sdir, fn)
        if SHIP_LONG_EDGE == 0:
            shutil.copy2(src, dst)
            how = "copied"
        else:
            with Image.open(src) as im:
                im.convert("RGB").resize(
                    (SHIP_LONG_EDGE, SHIP_LONG_EDGE),
                    Image.BILINEAR).save(dst, optimize=True)
            how = f"resized to {SHIP_LONG_EDGE}"
        with Image.open(dst) as im:
            w, h = im.size
        rows.append({"filename": fn, "width": w, "height": h,
                     "bytes": os.path.getsize(dst),
                     "source_bytes": os.path.getsize(src),
                     "how": how, "sha256": _sha256(dst)})
    F = pd.DataFrame(rows)
    mb = F.bytes.sum() / 1e6
    src_mb = F.source_bytes.sum() / 1e6
    print(f"  {len(F)} images {how} in {time.time() - t0:.1f}s")
    print(f"  package {mb:.1f} MB   (the 512 px originals would be "
          f"{src_mb:.1f} MB)")
    if SHIP_LONG_EDGE == 0:
        print(f"  SHIP_LONG_EDGE = 256 would make this roughly "
              f"{mb / 4.5:.1f} MB, at the cost")
        print("  described in the setting: the browser would scale "
              "256 -> cell rather than")
        print("  512 -> cell, and local contrast is a fine-detail "
              "measure.")
    report.append(("images shipped",
                   f"{len(F)} at {F.width.iloc[0]} px, {mb:.1f} MB"))
    return F, pdir, sdir


def check_copies(X, F, sdir, report):
    """CHECK 6: the shipped image is the measured image.

    Two statements, checked separately:

      the FEATURES. Every shipped file is re-measured with step 3's own
      `measure` and compared with the manifest's recorded values. This
      is the statement the analysis depends on -- the sat_mean and
      contrast_local attached to a column in grid_trials.csv are the
      features of the file the participant will be shown.

      the BYTES. At SHIP_LONG_EDGE = 0 the file is also hashed against
      its source. A features match with a bytes mismatch would mean
      two different files that happen to measure the same, which is
      not something to discover later.
    """
    banner("COPIES FAITHFUL  (CHECK 6)")
    man = X.drop_duplicates("filename").set_index("filename")
    files = list(F.filename)
    if REMEASURE_ALL is not True:
        n = int(REMEASURE_ALL)
        rng = np.random.default_rng(RANDOM_SEED + 2)
        files = list(rng.choice(files, size=min(n, len(files)),
                                replace=False))
    print(f"  re-measuring {len(files)} of {len(F)} shipped image(s) "
          "with step 3's own code")

    rows = []
    t0 = time.time()
    for fn in files:
        with Image.open(os.path.join(sdir, fn)) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        got = S3.measure(arr, only=["sat_mean", "contrast_local"])
        rec = man.loc[fn]
        rows.append({
            "filename": fn,
            "d_sat": float(got["sat_mean"]) - float(rec.sat_mean),
            "d_fog": (float(got["contrast_local"])
                      - float(rec.contrast_local)),
            "rel_sat": abs(float(got["sat_mean"])
                           - float(rec.sat_mean))
            / max(abs(float(rec.sat_mean)), 1e-9),
            "rel_fog": abs(float(got["contrast_local"])
                           - float(rec.contrast_local))
            / max(abs(float(rec.contrast_local)), 1e-9)})
    R = pd.DataFrame(rows)
    worst = max(R.rel_sat.max(), R.rel_fog.max())
    print(f"  worst relative difference {worst:.3e} "
          f"(tolerance {REMEASURE_RTOL:.0e})  in "
          f"{time.time() - t0:.1f}s")
    print(f"  worst absolute: sat_mean {R.d_sat.abs().max():.3e}, "
          f"contrast_local {R.d_fog.abs().max():.3e}")

    ok = worst <= REMEASURE_RTOL
    if not ok:
        bad = R[(R.rel_sat > REMEASURE_RTOL)
                | (R.rel_fog > REMEASURE_RTOL)]
        print(f"\n  FAIL: {len(bad)} image(s) do not measure as the "
              "manifest says.")
        print("  " + bad.head(6).to_string(index=False)
              .replace("\n", "\n  "))
        if SHIP_LONG_EDGE == 0:
            print("\n  At SHIP_LONG_EDGE = 0 this should be exactly "
                  "zero: the shipped file IS the")
            print("  measured file. A difference means the manifest "
                  "and the images in")
            print("  STIMULUS_ROOT/stimuli are from different runs of "
                  "step 3.")
        else:
            print(f"\n  At SHIP_LONG_EDGE = {SHIP_LONG_EDGE} this "
                  "should also be zero, because the")
            print("  resize is the one `measure` applies anyway. A "
                  "difference means it is not.")

    hash_note = "not applicable (images were re-encoded)"
    if SHIP_LONG_EDGE == 0:
        bad_h = []
        for r in F.itertuples():
            if _sha256(os.path.join(STIMULI_DIR, r.filename)) != r.sha256:
                bad_h.append(r.filename)
        if bad_h:
            ok = False
            print(f"\n  FAIL: {len(bad_h)} shipped file(s) differ from "
                  f"their source. First: {bad_h[0]}")
            hash_note = f"FAIL ({len(bad_h)} differ)"
        else:
            print(f"  all {len(F)} shipped files are byte-identical to "
                  "their source in the stimulus set")
            hash_note = f"PASS ({len(F)} byte-identical)"

    report.append(("check 6 features",
                   ("PASS" if worst <= REMEASURE_RTOL else "FAIL")
                   + f" (worst relative {worst:.1e})"))
    report.append(("check 6 bytes", hash_note))
    return R, ok


def write_package(X, P, F, pdir, S, report, cen):
    """The trial files, the preload list and the interface spec."""
    banner("THE PACKAGE")

    sets = {}
    for which in ("A", "B"):
        screens = []
        for sidx, (feature, order) in enumerate(SCREENS, 1):
            sub = X[(X["set"] == which) & (X.screen == sidx)]
            if not len(sub):
                continue
            first = sub[sub.row == sub.row.min()].sort_values(
                "display_column")
            cols = [{"display_column": int(c.display_column),
                     "column_label": c.column_label,
                     "level_index": int(c.level_index),
                     "is_original": bool(c.is_original),
                     "chroma": float(c.chroma), "haze": float(c.haze),
                     "gain": float(c.gain)}
                    for c in first.itertuples()]
            rows = []
            for row in sorted(sub.row.unique()):
                rr = sub[sub.row == row].sort_values("display_column")
                rows.append({
                    "row": int(row),
                    "scene_id": rr.scene_id.iloc[0],
                    "category": rr.category.iloc[0],
                    "images": ["stimuli/" + f for f in rr.filename],
                    "sat_mean": [round(float(v), 4)
                                 for v in rr.sat_mean],
                    "contrast_local": [round(float(v), 4)
                                       for v in rr.contrast_local],
                    "sd_from_original": [round(float(v), 5)
                                         for v in rr.sd_from_original]})
            screens.append({
                "screen": sidx, "feature": feature,
                "feature_measure": feature_column(feature),
                "column_order": order, "columns": cols, "rows": rows})
        sets[which] = screens

    doc = {
        "version": PACKAGE_VERSION,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stimulus_set": STIMULUS_ROOT,
        "descriptive_only": True,
        "blocks": [{"name": n, "none_button": bool(hn),
                    "free_text_on_none": bool(ft),
                    "when": ("after the pair trials" if n == "episode"
                             else "at the end of the session")}
                   for n, hn, ft in BLOCKS],
        "set_assignment": {
            "rule": SWAP_RULE,
            "options": [{"code": 0, "episode": "A", "normal": "B"},
                        {"code": 1, "episode": "B", "normal": "A"}]},
        "layout": {"max_cell_px": MAX_CELL_PX,
                   "min_cell_px": MIN_CELL_PX,
                   "row_gap_px": ROW_GAP_PX, "col_gap_px": COL_GAP_PX,
                   "chrome_px": CHROME_PX,
                   "page_side_px": PAGE_SIDE_PX,
                   "cell_chrome_px": CELL_CHROME_PX,
                   "rows_per_screen": ROWS_PER_SCREEN},
        "feature_sd": {"sat_mean": SD_SAT, "contrast_local": SD_FOG},
        "sets": sets,
    }
    with open(os.path.join(pdir, "grid_trials.json"), "w") as f:
        json.dump(doc, f, indent=1)

    X.to_csv(os.path.join(pdir, "grid_trials.csv"), index=False)
    F.drop(columns=["source_bytes"]).to_csv(
        os.path.join(pdir, "manifest.csv"), index=False)
    with open(os.path.join(pdir, "preload.json"), "w") as f:
        json.dump({"images": ["stimuli/" + fn
                              for fn in sorted(F.filename)]}, f,
                  indent=1)

    n_img = len(F)
    mb = F.bytes.sum() / 1e6
    with open(os.path.join(pdir, "README.txt"), "w") as f:
        W = f.write
        W("GRID BLOCK PACKAGE  (v4, step 7b)\n")
        W("=" * 62 + "\n\n")
        W(f"  version      {PACKAGE_VERSION}\n")
        W(f"  built        {doc['built']}\n")
        W(f"  from         {STIMULUS_ROOT}\n")
        W(f"  blocks       {len(BLOCKS)} "
          f"({', '.join(n for n, _, _ in BLOCKS)})\n")
        W(f"  screens      {len(SCREENS)} per block, "
          f"{ROWS_PER_SCREEN} scenes each\n")
        W(f"  scene sets   {N_SETS} x {SET_SIZE}, swapped between "
          "participants\n")
        W(f"  images       {n_img}, "
          f"{F.width.iloc[0]}x{F.height.iloc[0]}, {mb:.1f} MB\n\n")
        W("DESCRIPTIVE ONLY\n")
        W("  The grid block asks WHERE on the manipulated range an\n")
        W("  experience sits. It is not a test of the hypothesis: the\n")
        W("  registered analyses use the pair trials alone.\n\n")
        W("THE TWO BLOCKS\n")
        for n, hn, ft in BLOCKS:
            W(f"  {n:<9s} {'with' if hn else 'no'} 'none of these'"
              f"{', free text when none' if ft else ''}\n")
        W("\n  A participant sees ONE set in the episode block and the\n")
        W("  OTHER in the normal block, so no photograph appears in\n")
        W("  both. Which way round is the set assignment, below.\n\n")
        W("FILES\n")
        W("  grid_trials.json  the layout. `sets` has one entry per\n")
        W("                    scene set; each screen lists `columns`\n")
        W("                    left to right and `rows`, whose\n")
        W("                    `images` are in the same order as the\n")
        W("                    columns.\n")
        W("  grid_trials.csv   one row per cell, with the measured\n")
        W("                    features and the distance from that\n")
        W("                    scene's own original, in corpus SD.\n")
        W("  manifest.csv      every shipped image, its size and its\n")
        W("                    SHA-256.\n")
        W("  preload.json      the image paths, for fetching up front.\n")
        W("  interface_spec.md what the experiment must render and\n")
        W("                    record.\n\n")
        W("CENSORING, for whoever analyses this\n")
        if cen["fog"]["above"]:
            W("  The fog screen has NO column clearer than the\n")
            W("  photograph as taken. A participant whose vision is\n")
            W("  clearer answers on the original column, which in the\n")
            W("  normal block is the common case. Those answers are\n")
            W("  LOWER BOUNDS, and so is any shift computed from\n")
            W("  them. Report the count of fog answers on the\n")
            W("  original column in each block.\n\n")
        else:
            W("  Both screens carry a column beyond the original on\n")
            W("  each side, so neither axis is censored at it.\n\n")
        W("DISPLAY\n")
        W(f"  Cells are capped at {MAX_CELL_PX} px and the layout\n")
        W("  computes the real size from the window; at seven\n")
        W("  columns and three rows that is about 149 px on a\n")
        W("  1366-wide window and about 234 px on a 1920-wide one.\n")
        W("  Local contrast is fine detail, so do not shrink further:\n")
        W(f"  {MIN_CELL_PX} px is the floor, and hitting it means the\n")
        W("  layout no longer fits.\n")
    write_interface_spec(pdir, doc, X, P, S, cen)

    print(f"  grid_trials.json   {len(sets)} sets, "
          f"{sum(len(v) for v in sets.values())} screens")
    print(f"  grid_trials.csv    {len(X)} cells")
    print(f"  manifest.csv       {n_img} images")
    print(f"  preload.json       {n_img} paths, {mb:.1f} MB")
    print("  README.txt, interface_spec.md")
    report.append(("package", pdir))
    return doc


def package_previews(X, G, root):
    """Every screen of both sets, composed at the smaller window's cell
    size, for the record.

    The experiment does NOT use these. They are here so that what was
    shipped can be looked at months later without rebuilding anything,
    and so that a reversed screen can be checked against its forward
    twin by eye rather than by reading a CSV.
    """
    pdir = os.path.join(root, "package", "previews")
    os.makedirs(pdir, exist_ok=True)
    label, iw, ih = WINDOWS[0]
    n = 0
    for which in ("A", "B"):
        for sidx, (feature, order) in enumerate(SCREENS, 1):
            sub = X[(X["set"] == which) & (X.screen == sidx)]
            if not len(sub):
                continue
            ncol = int(sub.display_column.max())
            px, _, _, binds = file_cell_px(
                iw, ih, ncol, int(X.display_column.max()))
            cells, labs = [], None
            for row in sorted(sub.row.unique()):
                rr = sub[sub.row == row].sort_values("display_column")
                cells.append([(f, "") for f in rr.filename])
                if labs is None:
                    labs = [f"{c.column_label} {c.level_index:+d}"
                            for c in rr.itertuples()]
            img = compose_screen(
                cells, px, labs if MOCKUP_LABELS else None,
                f"set {which}, screen {sidx}: {feature} {order}",
                f"as shipped -- {label} window, {px} px per cell, "
                f"{int(ncol)} columns ({binds} binds)",
                inner_w=iw, inner_h=ih)
            img.save(os.path.join(
                pdir, f"set{which}_s{sidx}_{feature}_{order}.png"))
            n += 1
    print(f"  previews/          {n} screens at {label}")
    return n


def write_interface_spec(pdir, doc, X, P, S, cen):
    """What the experiment must render, and what it must record.

    The script writes the grid files; the experiment records the
    answers. This is the contract between the two, and it is written
    here because this is where the layout and the level indices are
    decided.
    """
    path = os.path.join(pdir, "interface_spec.md")
    with open(path, "w") as f:
        W = f.write
        W("# Grid block: interface spec\n\n")
        W(f"Package `{PACKAGE_VERSION}`, built {doc['built']} from "
          f"`{STIMULUS_ROOT}`.\n\n")
        W("The grid block is **descriptive only**. Nothing recorded "
          "here enters a registered\ntest; the registered analyses use "
          "the pair trials alone.\n\n")

        W("## The two blocks\n\n")
        W("| block | when | question | 'none of these' | free text |\n")
        W("|---|---|---|---|---|\n")
        for b in doc["blocks"]:
            q = ("which column looks most like your surroundings "
                 "looked during the episode"
                 if b["name"] == "episode" else
                 "which column looks most like your surroundings "
                 "normally look")
            W(f"| {b['name']} | {b['when']} | {q} | "
              f"{'yes' if b['none_button'] else 'no'} | "
              f"{'yes, when none is chosen' if b['free_text_on_none'] else 'no'} |\n")
        W("\nThe normal block has no `none` button on purpose: "
          "\"my surroundings normally look\nlike none of these\" is not "
          "an answer the normal-to-episode shift can use, and an\n"
          "empty normal answer loses the whole pair.\n\n")

        W("## The set swap\n\n")
        W(f"`set_assignment.rule` is `{SWAP_RULE}`.\n\n")
        if SWAP_RULE == "random_recorded":
            W("At session start the client draws `code` from "
              "{0, 1} with probability 0.5 each,\nrecords it, and uses "
              "it for the whole session:\n\n")
            W("```\ncode 0 -> episode block shows set A, normal block "
              "shows set B\ncode 1 -> episode block shows set B, normal "
              "block shows set A\n```\n\n")
            W("No server state is needed, which this study does not "
              "have. The counterbalance is\ntherefore **binomial, not "
              "exact**: at 60 participants the split is 30 +/- 4 about "
              "two\nthirds of the time. The code must be in the data "
              "file for every participant, or\nthe imbalance cannot be "
              "checked or adjusted for.\n\n")
        else:
            W("The server assigns `code` by alternating an "
              "incrementing participant counter, so\nthe "
              "counterbalance is exact. This requires a counter that "
              "survives between\nsessions; if there is none, use "
              "`random_recorded` instead.\n\n")

        W("## Rendering a screen\n\n")
        W("For the chosen set, take `sets[<set>]` and present its "
          "screens in `screen` order.\nFor each screen:\n\n")
        W("- `columns` is the layout **left to right as displayed**. "
          "`column_order` says whether\n  that is the forward or the "
          "reversed arrangement; the order in the file is already\n  "
          "the display order, so do not reverse it again.\n")
        W("- each entry of `rows` has `images` in the **same order as "
          "`columns`**. Build the grid\n  from those files; nothing is "
          "pre-composited.\n")
        W(f"- cell size: `floor(min(byWidth, byHeight, "
          f"{MAX_CELL_PX}))`, at least {MIN_CELL_PX}, with\n")
        W(f"  `byWidth = (innerWidth - {PAGE_SIDE_PX} - (cols-1)*"
          f"{COL_GAP_PX})/cols - {CELL_CHROME_PX}` and\n")
        W(f"  `byHeight = (innerHeight - {CHROME_PX} - (rows-1)*"
          f"{ROW_GAP_PX})/rows`.\n")
        W("- the participant picks a **column**, not a cell. A click "
          "anywhere in a column\n  selects that column.\n")
        W(f"- the cell size is ONE size for the whole block, computed "
          f"from the WIDEST screen\n  in the file "
          f"({int(X.display_column.max())} columns here). Do not size "
          "each screen by its own column\n  count.\n\n")

        W("## What to record, per screen\n\n")
        W("| field | what |\n|---|---|\n")
        for k, v in [
            ("block", "`episode` or `normal`"),
            ("set_code", "the `code` drawn for this participant"),
            ("scene_set", "`A` or `B`, the set shown on this screen"),
            ("screen", "the screen number from the file"),
            ("feature", "`fog` or `colour`"),
            ("column_order", "`forward` or `reversed`, as shipped"),
            ("scene_ids", "the three scenes, in row order"),
            ("chosen_display_column",
             "1-based position of the chosen column AS DISPLAYED"),
            ("chosen_level_index",
             "the `level_index` of that column: 0 is the scene's own "
             "original, negative is more than normal, positive is "
             "less"),
            ("chose_none",
             "true when 'none of these' was used (episode block only)"),
            ("free_text", "the typed answer, when `none` was chosen"),
            ("rt_ms", "from screen onset to the confirming click"),
            ("cell_px",
             "the cell size the layout computed, which varies by "
             "window and bears on whether the end columns were "
             "distinguishable"),
            ("inner_w, inner_h", "the window at the time of the screen"),
        ]:
            W(f"| `{k}` | {v} |\n")
        W("\nRecord **both** `chosen_display_column` and "
          "`chosen_level_index`. The position moves\nwhen the column "
          "order reverses and the index does not, so storing only one "
          "of them\nmakes the reversed screens unreadable without the "
          "package in hand.\n\n")
        W("Do **not** copy the level parameters, the chroma, the haze "
          "or the measured features\ninto the data file. They are in "
          "`grid_trials.csv`, keyed by set, screen, row and\ncolumn, "
          "and a second copy is a second thing that can disagree.\n\n")

        W("## Analysis note\n\n")
        W("The normal-to-episode shift is computed in **feature "
          "units**, not in grid steps:\nthe contrast-up step is about "
          "+0.22 SD and the fog steps are -0.13 SD and larger,\nso a "
          "step is not a constant amount of anything. "
          "`grid_trials.csv` carries\n`sd_from_original` per cell for "
          "exactly this.\n\n")
        pos = {}
        for feature in ("fog", "colour"):
            sub = X[(X.feature == feature)
                    & (X.column_order == "forward")
                    & X.is_original]
            if len(sub):
                pos[feature] = int(sub.display_column.iloc[0])
        if len(set(pos.values())) > 1:
            W(f"The original sits at position {pos['fog']} on the fog "
              f"screen and {pos['colour']} on the colour\nscreen, so "
              "comparing *which column* people choose across the two "
              "features is partly\na position effect. The shift within "
              "each feature is unaffected.\n\n")
        if cen["fog"]["above"]:
            W("### The fog axis is censored at the original\n\n")
            W("The fog screen has **no column clearer than the "
              "photograph as taken**. A\nparticipant whose vision is "
              "clearer than that has to answer on the original\n"
              "column, and in the normal-appearance block that is the "
              "common case rather than\nthe edge case -- most "
              "people's normal vision is not foggy.\n\n")
            W("So a fog answer on the original column is a **lower "
              "bound** on that person's\nnormal, and the "
              "normal-to-episode shift computed from it is a lower "
              "bound on the\nshift. The bias is conservative; it can "
              "only shrink the shift. It is declared\nrather than "
              "absorbed, and **the count of fog answers on the "
              "original column in\neach block is reported alongside "
              "the shift**.\n\n")
            W("This is why `chosen_level_index` must be recorded: the "
              "censored answers are\nexactly `level_index == 0` on a "
              "`fog` screen, and they cannot be identified from\nthe "
              "display column alone once the column order reverses.\n")
        else:
            W("Both screens carry a column beyond the original on "
              "each side, so neither axis\nis censored at it.\n")


# ============================================================
# The report
# ============================================================

def write_report(root, report, elapsed, unused=None, cen=None):
    """The file that survives. Result files stay on the machine that
    ran them, so what was run and what came out has to be here."""
    path = os.path.join(root, "grid_report.txt")
    with open(path, "w") as f:
        W = f.write
        W("GRID BLOCK REPORT  (v4, step 7b)\n")
        W("=" * 70 + "\n")
        W(f"Mode          : {MODE}\n")
        W(f"Stimulus set  : {STIMULUS_ROOT}\n")
        W(f"Step 3 script : {STEP3_SCRIPT} (imported, not copied)\n")
        W(f"Output        : {root}\n")
        W(f"Seed          : {RANDOM_SEED}\n")
        W(f"Elapsed       : {elapsed:.1f}s\n\n")
        for k, v in report:
            W(f"  {k:<32s} {v}\n")

        W("\n1. WHAT THIS BLOCK IS FOR\n")
        W("-" * 70 + "\n")
        W("  Two blocks of four screens. Each screen shows one\n")
        W("  feature at every level at once and asks which column\n")
        W("  looks most like the world looked -- during the episode\n")
        W("  in one block, normally in the other.\n\n")
        W("  DESCRIPTIVE ONLY. It locates an experience on the\n")
        W("  manipulated range; it does not test the hypothesis. The\n")
        W("  registered analyses in step 8 use the pair trials alone.\n")
        W("  The quantity of interest is the normal-to-episode shift,\n")
        W("  in FEATURE UNITS rather than grid steps: the contrast-up\n")
        W("  step is about +0.22 SD and the fog steps are -0.13 SD and\n")
        W("  larger, so one step is not a constant amount of anything.\n")

        W("\n2. WHY TWO MATCHED SCENE SETS\n")
        W("-" * 70 + "\n")
        W("  The shift is a within-person comparison of two blocks, so\n")
        W("  a photograph may not appear in both: recognising it would\n")
        W("  anchor the second answer. 24 scenes are chosen as 12\n")
        W("  MATCHED PAIRS, one member to each set, in the SAME SLOT.\n\n")
        W("  Pairs rather than set means: matched means can hide a set\n")
        W("  that is bimodal, and a pair bound to a slot makes the two\n")
        W("  blocks comparable row by row. The per-pair distances are\n")
        W("  in pairs.csv.\n")

        W("\n3. NOTHING WAS RENDERED\n")
        W("-" * 70 + "\n")
        W("  Step 3 already renders every cell a grid screen needs and\n")
        W("  marks it use = 'grid' or 'both'. This step reads those\n")
        W("  rows and ships copies. In particular the two columns\n")
        W("  above the original on the colour screen (p65, p75 at\n")
        W("  h = 0) exist for every scene already, so no add-on\n")
        W("  rendering pass was needed and step 3 was not re-run.\n\n")
        W("  The feature code and the filename helpers are IMPORTED\n")
        W("  from step 3's script. There is one definition of each in\n")
        W("  the pipeline, so the two cannot drift.\n\n")
        W("  The frozen pair package was not read, written or\n")
        W("  re-checksummed. Its images are a disjoint set of cells.\n")

        W("\n4. THE CONTRAST-UP COLUMN\n")
        W("-" * 70 + "\n")
        if CONTRAST_UP_GAIN == "undecided":
            W("  UNDECIDED. This run composed a mock-up screen for\n")
            W("  each candidate gain at the cell size the experiment\n")
            W("  will compute, and selected the scenes WITHOUT an up\n")
            W("  column -- the conservative reading, since the\n")
            W("  selection must not depend on a gain nobody has\n")
            W("  chosen. Set CONTRAST_UP_GAIN and re-run as\n")
            W("  MODE = 'package'; the selection is recomputed then,\n")
            W("  and it can move, because the headroom criterion only\n")
            W("  exists once there is a gain to have headroom for.\n")
        elif CONTRAST_UP_GAIN is None:
            W("  DROPPED. No gain was distinguishable from the\n")
            W("  original on step 3's sheets_contrast/ diagnostic,\n")
            W("  which is drawn at the FULL 512 px. The grid shows\n")
            W("  cells at 149-239 px, so a difference invisible at\n")
            W("  512 px cannot appear there, and the mock-up could\n")
            W("  only confirm it.\n\n")
            W("  The damage figures were never the binding\n")
            W("  constraint: gain 1.0875 does no measured damage at\n")
            W("  all on 40 of 47 scenes and is still invisible. The\n")
            W("  gain holds the mean fixed and moves only the\n")
            W("  histogram tails -- about +7 on lum_p95 and -6 on\n")
            W("  lum_p05 -- so the change lives in a small fraction\n")
            W("  of the frame and downscaling averages it away.\n")
            W("  Haze moves the same statistics further AND lays a\n")
            W("  coherent veil over every pixel, which is what\n")
            W("  survives a thumbnail.\n")
        else:
            W(f"  {CONTRAST_UP_GAIN}, chosen from the mock-up. All\n")
            W("  three gains were provisional in step 3 and the\n")
            W("  question the damage numbers cannot answer is whether\n")
            W("  the column is visible at the real cell size.\n")

        W("\n5. RENDERED BUT NOT USED\n")
        W("-" * 70 + "\n")
        if unused is None or not len(unused):
            W("  Every grid cell step 3 rendered is on a screen.\n")
        else:
            W("  Step 3 renders more grid cells than the grid shows.\n")
            W("  Nothing was deleted or re-rendered; the images are\n")
            W("  still in the stimulus set with their measured\n")
            W("  features. unused_cells.md carries this into the\n")
            W("  pipeline documentation.\n\n")
            by = unused.groupby("label", as_index=False).agg(
                images=("images", "sum"))
            for r in by.itertuples():
                W(f"  {r.label} -- {r.images} images\n")
                note = UNUSED_CELL_NOTES.get(r.label,
                                             "NO REASON RECORDED.")
                line = ""
                for word in note.split():
                    if len(line) + len(word) > 64:
                        W("    " + line + "\n")
                        line = ""
                    line += (" " if line else "") + word
                if line:
                    W("    " + line + "\n")
                W("\n")
            W("  The pair trials are unaffected: every cell above is\n")
            W("  use = 'grid' in the manifest and was never in the\n")
            W("  pair inventory.\n")

        W("\n6. CENSORING\n")
        W("-" * 70 + "\n")
        if cen and cen["fog"]["above"]:
            W("  THE FOG AXIS IS CENSORED AT THE ORIGINAL. The fog\n")
            W("  screen has no column clearer than the photograph as\n")
            W("  taken, so a participant whose vision is clearer than\n")
            W("  that answers on the original column.\n\n")
            W("  In the NORMAL-APPEARANCE block this is the common\n")
            W("  case, not the edge case: most people's normal vision\n")
            W("  is not foggy. Every fog answer on the original\n")
            W("  column is a LOWER BOUND on that person's normal, and\n")
            W("  the normal-to-episode shift computed from it is a\n")
            W("  lower bound on the shift.\n\n")
            W("  The bias is conservative -- it can only shrink the\n")
            W("  shift -- but it is declared rather than absorbed.\n")
            W("  REPORT THE COUNT of fog answers on the original\n")
            W("  column in EACH BLOCK, alongside the shift.\n\n")
        else:
            W("  The fog screen carries a column above the original,\n")
            W("  so the fog axis is not censored there.\n\n")
        if cen and not cen["colour"]["above"]:
            W(f"  The colour screen keeps "
              f"{cen['colour']['top']} above the original, so the\n")
            W("  colour axis is open on that side.\n")

        W("\n7. WHAT THE EXPERIMENT MUST DO\n")
        W("-" * 70 + "\n")
        if MODE == "package":
            W("  package/interface_spec.md is the contract: what to\n")
            W("  render, how the cell size is computed, how the set\n")
            W("  swap is assigned, and what to record per screen.\n\n")
            W("  Two fields are easy to leave out and both are needed:\n")
            W("  the set assignment code, without which the\n")
            W("  counterbalance cannot be checked, and the realised\n")
            W("  cell size, which varies by window and bears on\n")
            W("  whether the end columns were distinguishable at all.\n")
        else:
            W("  Not written in this mode. Run MODE = 'package' once\n")
            W("  the contrast-up gain is settled.\n")

        W("\n8. WHAT TO LOOK AT\n")
        W("-" * 70 + "\n")
        W("  mockup/            the screens at the real cell size.\n")
        W("                     The decision is made here.\n")
        W("  selection.png      the pool, the pairs and the two sets.\n")
        W("  headroom.png       what the headroom preference costs.\n")
        W("  ladders.png        the columns in feature units.\n")
        W("  pairs.csv          per-pair distance and slot.\n")
        W("  headroom_cost.csv  matching cost at every gain x\n")
        W("                     tolerance.\n")
        W("  cell_sizes.csv     the layout arithmetic.\n")

        W("\nOVERALL: " + ("PASS" if all(
            "FAIL" not in str(v) for _, v in report) else "FAIL") + "\n")
    print(f"\n  report: {path}")


# ============================================================
# main
# ============================================================

def main():
    global S3
    t0 = time.time()
    report = []

    print("=" * 60)
    print("BUILD GRID BLOCK  (v4, step 7b)")
    print("=" * 60)

    settings_check()
    S3 = load_step3()
    print(f"  step 3 imported from {STEP3_SCRIPT} "
          f"(measurement size {S3.FEATURE_SIZE} px)")

    root = resolve_output_dir()
    print(f"  output: {root}")

    M, G, P_plan, D, L = load_inputs(report)
    columns = describe_columns(report)

    col_counts = [len(columns["fog"]), len(columns["colour"])]
    realised = max(col_counts)
    if MODE == "mockup":
        for g in MOCKUP_GAINS:
            col_counts.append(len(resolve_columns("fog", up=g)))
        if MOCKUP_INCLUDE_NO_UP:
            col_counts.append(len(resolve_columns("fog", up=None)))
    CS, layout_ok = layout_table(col_counts, realised, report)
    CS.to_csv(os.path.join(root, "cell_sizes.csv"), index=False)

    S = scene_table(G, D, columns, report)
    mono, usable = check_monotone(S, columns, report)
    mono.to_csv(os.path.join(root, "monotone.csv"), index=False)
    S.drop(columns=[c for c in S.columns
                    if c.endswith(("_files", "_values"))]).to_csv(
        os.path.join(root, "scene_table.csv"))

    pairs, Dm, idx, H, cost, feat_names = match_pairs(
        S, usable, report, root)
    PP = assign_slots(pairs, S, Dm, idx, report)
    PP.to_csv(os.path.join(root, "pairs.csv"), index=False)

    X = build_screens(PP, S, G, columns, report)
    X.to_csv(os.path.join(root, "screens.csv"), index=False)
    UN = unused_cells(M, columns, root, report)
    CEN = report_censoring(columns, report)
    pair_reuse(X, report, root)

    banner("FIGURES")
    plot_selection(S, PP, root)
    plot_headroom(H, root)
    plot_ladders(X, root)
    print("  selection.png, headroom.png, ladders.png")

    mockup(X, S, G, report, root)

    if MODE == "package":
        F, pdir, sdir = ship_images(X, root, report)
        _, copies_ok = check_copies(X, F, sdir, report)
        write_package(X, PP, F, pdir, S, report, CEN)
        if PACKAGE_PREVIEW:
            package_previews(X, G, root)
        if not copies_ok:
            print("\n  CHECK 6 FAILED. The package was written so the "
                  "difference can be looked at,")
            print("  but do not upload it until the check passes.")
    else:
        print("\n  MODE = 'mockup': nothing was shipped.")
        print("  Look at mockup/, set CONTRAST_UP_GAIN, then run "
              "MODE = 'package'.")

    write_report(root, report, time.time() - t0, UN, CEN)

    bad = [k for k, v in report if "FAIL" in str(v)]
    print("\n" + "=" * 60)
    if bad:
        print("FINISHED WITH FAILURES: " + ", ".join(bad))
    else:
        print(f"DONE in {time.time() - t0:.1f}s -- every check passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
