"""
build_trials_v4.py  (stimulus pipeline, v4 axes, step 5)

Samples the trials a participant actually sees, orders the session, and
writes a self-contained package ready to upload.

Where this sits
---------------
  step 3  build_stimulus_set_v4  the stimuli and the manifest
  step 4  analyse_pairs_v4       every usable pair -> all_pairs.csv
  step 5  this one               the trial list the experiment runs
  step 6  simulate_power_v4      is that trial list powered?

SELF-CONTAINED and lightweight: it reads all_pairs.csv and the rendered
images, nothing else. Step 4 has already classified every pair and
flagged the balanced conflict ones.

SEVEN TRIAL LISTS, ONE INVOCATION
---------------------------------
VERSIONS is a LIST: one entry per trial list. Each gets its own list
and its own package, built with identical settings and a shared
RANDOM_SEED, so two versions differ only in what their entry varies --
the step-4 variant (which colour levels exist), the colour cut, the
bin mode, and which random streams it uses.

Four of them are the design versions the preregistration compares:

  current        original levels, current sampling   (the baseline)
  extra_levels   vivid levels,    current sampling
  larger_diffs   original levels, cut sampling
  both           vivid levels,    cut sampling

Two more vary only the cut, so step 6 can read the detection gained
against the range lost off a curve rather than from one value. The
seventh is a reproduction control, not a candidate: it rebuilds the
trial list the earlier run shipped, to show that the conflict trials
differing in the others are the balance normalisation and not a drift
in the sampler.

ONE RANDOM STREAM PER DRAW
--------------------------
Every draw has its OWN generator, seeded from (RANDOM_SEED, the name
of the draw). This is not tidiness; it is what makes the comparison
between versions mean anything.

With a single generator, the draws run in sequence off one stream, so
how many numbers the colour draw consumes decides where the fog draw
starts. The colour cut changes the number of usable magnitude bins,
which changes the number of draws, which moves the stream -- and the
fog, together and conflict trials then differ between two versions
whose only declared difference is a threshold on COLOUR pairs. Measured
on `current` against `larger_diffs`, which share a ladder and differ
only in the cut: 1 of 60 fog_only trials shared, 1 of 30 together, 0 of
30 conflict. Every trial was valid; they were simply a different random
draw.

Two consequences, both bad. The fog-only condition stops being a
control -- its detection moves between versions for a reason that has
nothing to do with the design. And colour information leaks in from
types the cut does not touch: `both` gained 16.4 units of summed
squared colour predictor from its `together` trials alone, about a
sixth of its total gain over `current`.

Under per-type streams, two versions built from the SAME step-4 variant
have IDENTICAL fog_only, together and conflict trials, and the only
difference between them is the colour cut. check_streams verifies this
and fails if it is ever untrue.

What this does NOT promise: versions built from DIFFERENT variants
still differ in every type, and that is the design rather than a leak.
The vivid colour levels add new images, hence new pairs of every type
-- a fog pair at the p75 colour level did not exist on the original
ladder -- so `current` and `extra_levels` draw from genuinely different
fog pools.

The legacy single-stream behaviour is kept as a per-version setting,
used by the reproduction control: the shipped list was built that way,
so matching it requires it.

TWENTY SEEDS PER VERSION
------------------------
One draw is one sample. Two versions can differ because one of them is
a better design, or because one of them got a luckier draw of 180
trials out of a pool of thousands, and a single build cannot tell those
apart.

So each version is built at N_SEEDS seeds. The FIRST is RANDOM_SEED and
is the real one: it gets the figure, the report and the package, and it
is the list that would ship. The rest are replicates, written to
<root>/seeds/<label>/seed_<n>/trial_list.csv, with no package and no
figure. Step 6 simulates every seed and reports each version's mean
detection with its between-seed spread, so a difference between
versions can be read against the spread of a single version.

THE SHIPPED SEED IS FIXED IN ADVANCE and is not chosen on the results.
Picking the seed that simulates best would be picking the draw that
flatters the design, which is the same error as choosing an analysis
after seeing the data.

THE TRIAL COUNTS, AND WHY
-------------------------
40 saturation-only, 40 fog-only, 40 mixed (half together, half
conflict) = 120 analysed trials, as in v2.

40 is not arbitrary. This is a design in which EACH PARTICIPANT IS A
DATA POINT: the analysis estimates a per-person weight on each cue and
then asks how those weights differ between people -- whether someone
who describes their experience as foggy weighs the cues differently
from someone who describes it as lifeless. So the count is set by the
precision of the INDIVIDUAL estimate, not the group one. A proportion
from 40 trials has a standard error near sqrt(0.25/40) = 0.079, which
sits just below the expected spread between people. Below about 30 the
within-person noise starts to dominate and individual differences
become unreadable -- and individual differences are the whole reason a
small sample is worth running.

If piloting shows the session is too long, cut all three together, and
cut N_MIXED LAST: the conflict trials inside it carry the most
information, being the only ones that can show the crossover.

WHAT IS AND IS NOT ANALYSED
---------------------------
  analysed      the four real types only, 120 trials
  consistency   repeats of pairs already shown, sides swapped. EXCLUDED
                from the analysis on purpose: they are an INDEPENDENT
                quality check, and the agreement rate is what tells you
                whether a flat weight means "does not track these
                features" or "was not attending". A check that is also
                part of the analysis it is checking is no check at all.
  catch         an attention check. In the current design
                ("no_images") NO IMAGES ARE SHOWN: the screen carries
                only an instruction naming one arrow key, and takes a
                single press. The row still carries an image pair, so
                the alternative design ("images") can be chosen
                instead without new stimuli -- see CATCH_MODE. The
                mode is written into every catch row and the
                experiment must stop if its own setting disagrees.
                Exclusion is at 3 or more failures of 10.
  practice      unanalysed by definition

Practice and catch pairs are drawn from pairs the analysed set never
uses, so no pair is ever seen twice except a consistency repeat -- and
there the repeat comes SECOND, so the analysed presentation is the
uncontaminated one.

PRACTICE IS DRAWN BEFORE CATCH, so that changing the number of catch
trials cannot move the warm-up. See build_main_trials.

WHAT CHANGING N_CATCH DOES CHANGE: the catch trials themselves, the
presentation order (the session shuffle acts on a frame of a different
length) and therefore every trial number, including the values in
`repeat_of`. The 180 analysed trials and the 15 consistency repeats are
the same pairs either way, which is what lets a step-6 result survive a
change to the attention check.

Inputs
------
  VARIANTS      all_pairs.csv per step-4 variant, plus where that
                set's rendered images live (step 3)
  VERSIONS      (label, variant, colour cut, bin mode) per trial list
  REPRODUCE_FROM  the shipped trial list the control must reproduce

Outputs (per set, under OUTPUT_ROOT/<label>/)
---------------------------------------------
  trial_list.csv      the full session, in presentation order
  trial_design.png    the trials in difference space, and bin counts
  trials_report.txt   counts and checks
  package/            self-contained, ready to upload:
    trial_list.csv      relative paths
    trial_list.json     the same trials for the experiment to load
    stimuli/            ONLY the images this list uses
    manifest.csv        every shipped image with its features
    preload.json        the paths, for fetching up front
    README.txt          how to read it all

Outputs (once, under OUTPUT_ROOT/)
----------------------------------
  trials_overview.txt   all versions side by side, and the checks
  comparison.csv        one row per quantity, one column per version
  bins_by_version.csv   trial counts per bin and type, per version
  stream_check.csv      which types two versions share, pair by pair
                        -- the check that the streams are separate
  reproduction.csv      the control against the shipped list
  seeds_by_version.csv  one row per version per seed: r, VIF, spread
  seeds/<label>/seed_<n>/trial_list.csv
                        the replicate draws, for step 6

Usage
-----
    conda activate drl2drv
    python build_trials_v4.py
"""

import os
import io
import json
import time
import hashlib
import warnings
import contextlib

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cv2.setNumThreads(0)


# ============================================================
# SETTINGS
# ============================================================

# --- Input / output ---
#
# VARIANTS
#   Where each pair table and its stimuli live. One entry per step-4
#   variant. The stimuli directory is the set's own rendered images --
#   crossing these would build a trial list whose predictors describe
#   different pictures from the ones shown, which the packaging guard
#   further down exists to catch.
VARIANTS = {
    "original": ("data/pair_analysis_v4_ext/original/all_pairs.csv",
                 "data/stimulus_set_v4_ext_haze_eq/stimuli"),
    "extended": ("data/pair_analysis_v4_ext/extended/all_pairs.csv",
                 "data/stimulus_set_v4_ext_haze_eq/stimuli"),
    # built by re-running step 4 with BALANCE_NORM = "scene_range";
    # used only by the reproduction version below
    "legacy":   ("data/pair_analysis_v4_ext_legacy/original/all_pairs.csv",
                 "data/stimulus_set_v4_ext_haze_eq/stimuli"),
}

# VERSIONS
#   (label, variant, colour cut, bin mode) per trial list. Everything
#   else -- counts, caps, seed -- is identical across versions, so a
#   difference between two of them is the thing being varied and
#   nothing else.
#
#   THE FOUR DESIGN VERSIONS vary two things independently:
#
#     current        original levels, current sampling   (the baseline)
#     extra_levels   vivid levels,    current sampling
#     larger_diffs   original levels, cut sampling
#     both           vivid levels,    cut sampling
#
#   THE TWO SENSITIVITY VERSIONS vary only the cut, so step 6 can
#   measure what detection is gained against what range is lost, and
#   the cut can be chosen from that rather than guessed here.
#
#   THE REPRODUCTION VERSION is a one-off control, not a candidate. It
#   reads the legacy pair table (scene-range balance), uses the legacy
#   rank binning AND the legacy single random stream, which together
#   should reproduce the trial list the earlier run shipped, trial for
#   trial. See REPRODUCE_FROM.
VERSIONS = [
    # label            variant     colour cut   bin mode    streams
    ("current",        "original",       None,  "fixed",    "per_type"),
    ("extra_levels",   "extended",       None,  "fixed",    "per_type"),
    ("larger_diffs",   "original",       31.1,  "fixed",    "per_type"),
    ("both",           "extended",       31.1,  "fixed",    "per_type"),
    ("sens_cut_25_5",  "extended",       25.5,  "fixed",    "per_type"),
    ("sens_cut_35_3",  "extended",       35.3,  "fixed",    "per_type"),
    ("reproduction",   "legacy",         None,  "quantile", "single"),
]

# STREAMS — the fifth field of a VERSIONS entry
#   "per_type"  one generator per draw, seeded from (seed, draw name).
#               Use this for everything new. Two versions built from the
#               same variant then have IDENTICAL fog_only, together and
#               conflict trials, so the colour cut is the only
#               difference between them -- see the header.
#   "single"    the legacy behaviour: one generator for the whole
#               build, so every draw depends on how many numbers the
#               draws before it consumed. Kept ONLY so the reproduction
#               control can match the shipped list, which was built
#               this way.
#
#   The stream names are listed in STREAM_NAMES below. Adding a draw
#   means adding a name, not reusing one: two draws sharing a name are
#   back to the coupling this exists to remove.

# COLOUR_CUT — the third field of a VERSIONS entry
#   A FIXED RAW THRESHOLD on |d_sat|, in sat_mean units, below which a
#   colour-only pair is not eligible. Fixed rather than a percentile of
#   each pool, so the cut sits at the same place in every version and
#   two versions differ by the levels available rather than by where
#   the knife fell.
#
#   The pool costs, measured on the extended set (6,282 colour-only
#   pairs spanning 17-109 units):
#
#     cut    pairs kept   range kept   min per scene
#     none        6,282      17-109              88
#     25.5        4,712      26-109              46
#     31.1        3,770      31-109              28
#     35.3        3,142      35-109              22
#
#   Every cut keeps all 47 scenes; what is lost is the bottom of the
#   dose-response curve, which is where a flat line would be most
#   diagnostic of a response bias. That trade is what step 6 measures.
#   None means no cut.

# --- Magnitude bins ---
#
# BIN_MODE — the fourth field of a VERSIONS entry
#   "fixed"     bins are value ranges in corpus-SD units, shared by
#               every version, so bin 2 means the same difference
#               everywhere and the versions can be compared.
#   "quantile"  the legacy behaviour: rank-based bins holding exactly
#               equal numbers of pairs, recomputed per pool. Used ONLY
#               by the reproduction version.
#
#   The two cannot be mixed in one comparison, and they do not agree:
#   rank binning splits ties and near-edge pairs differently from a
#   value threshold, so the reproduction version has to use the legacy
#   mode to match trial for trial.

# BIN_EDGES_SD
#   Inner edges per type, in corpus-SD units of that type's driver
#   (|d_sat| for sat_only, |d_fog| for the rest). n edges give n+1
#   bins.
#
#   THE FOG-DRIVEN TYPES use the CURRENT version's own tertiles, so
#   "fixed" and "quantile" agree on them to within tie-breaking, and
#   their pools are never cut.
#
#   COLOUR USES FIVE BINS, NOT THREE, and the reason is worth keeping.
#   The natural choice -- the current version's tertiles, 0.579 and
#   0.794 SD -- does not survive the cut. The extended colour pool
#   reaches 2.77 SD where the current one stops at 1.72, so the current
#   top tertile would hold everything from 0.79 SD upward; and a cut at
#   31.1 units (0.794 SD) empties bin 1 and leaves FOUR pairs in bin 2.
#   All sixty colour trials would come from one bin, and the colour
#   dose-response curve -- the thing the cut is meant to sharpen --
#   would be gone.
#
#   Five fixed edges spanning the union of both pools leave three
#   usable bins at every cut:
#
#   The top edge is 1.80 SD, above the ORIGINAL pool's maximum of
#   1.72. That bin is therefore empty on the original levels and is
#   dropped there by the rule below, which is the intended behaviour:
#   the edges are shared so the bins mean the same thing, and a pool
#   that does not reach a bin simply does not use it. Pairs, and the
#   distinct scenes that can supply them under the cap:
#
#     pool / cut      bin1      bin2      bin3      bin4      bin5   usable
#     original      680/47    660/47    510/47    180/34      0/0        4
#     extended     1074/47   1406/47   1680/47   1798/47    324/36       5
#     ext / 25.5      0/ 0    916/47   1680/47   1798/47    324/36       4
#     ext / 31.1      0/ 0      0/ 0   1646/47   1798/47    324/36       3
#     ext / 35.3      0/ 0      0/ 0   1022/47   1798/47    324/36       3
BIN_EDGES_SD = {
    "sat_only": (0.58, 0.79, 1.10, 1.80),   # 5 bins
    "fog_only": (0.5894, 0.8259),           # 3 bins, current tertiles
    "together": (0.5517, 0.8362),           # 3 bins, current tertiles
    "conflict": (0.4506, 0.5714),           # 3 bins, current tertiles
}

# EMPTY_BIN_RULE
#   What to do when a cut empties a bin. "spread" divides the type's
#   trials evenly over the bins that still hold at least
#   MIN_PAIRS_PER_BIN pairs, and the report states which bins were
#   dropped and how the trials were redistributed. "error" stops
#   instead.
#
#   Spreading rather than erroring is deliberate: emptying the low bins
#   is exactly what a cut is FOR, so it is an expected outcome to be
#   recorded, not a fault. What must not happen silently is a bin
#   quietly receiving fewer trials than its share.
EMPTY_BIN_RULE = "spread"

# MIN_PAIRS_PER_BIN
#   A bin holding fewer pairs than this is treated as empty. Four pairs
#   in a bin cannot supply twelve trials, and under a scene cap they
#   would come from at most four scenes.
#
#   PAIRS ARE NOT THE BINDING CONSTRAINT; SCENES ARE. A bin can hold
#   hundreds of pairs drawn from seven scenes, and at
#   MAX_PAIRS_PER_SCENE_PER_TYPE = 2 those seven scenes can supply
#   fourteen trials however many pairs there are. assign_bins
#   therefore also drops a bin whose distinct scenes cannot cover its
#   share, and iterates -- dropping one bin raises every other bin's
#   share, which can put a second bin out of reach.
#
#   This was found by the sampler failing loudly on the `current`
#   version, which is the behaviour wanted: it drew 57 of 60 and
#   stopped, rather than returning a short list that looked complete.
MIN_PAIRS_PER_BIN = 20

# --- Reproduction of the shipped trial list ---
#
# REPRODUCE_FROM
#   The trial list the "reproduction" version must reproduce. It was
#   built from the scene-range balance definition and the legacy rank
#   bins, which is why that version carries both.
#
#   CONFLICT TRIALS ARE THE POINT OF THIS CHECK. Changing the balance
#   normalisation changes which conflict pairs are eligible, so the
#   four design versions above deliberately do NOT reproduce the
#   shipped list. This version isolates that: if it matches trial for
#   trial, the difference in the others is the normalisation and
#   nothing else has drifted.
#
#   Set to None to skip.
REPRODUCE_FROM = "data/trials_v4_ec/haze_eq/trial_list.csv"
REPRODUCE_LABEL = "reproduction"

# REPRODUCE_POOL_ORDER
#   The earlier run's all_pairs.csv, used ONLY to restore the row
#   order its trial list was drawn from.
#
#   Why this is needed. The pair SET is identical -- step 4 verified
#   15,004 pairs, none missing on either side -- but the shipped list
#   was drawn from a table in a different ORDER, and the sampler
#   shuffles the pool, so order is part of what picks the trials. The
#   order was never a stable property: step 3 measured scenes with
#   imap_unordered, so its manifest followed worker completion order
#   and the pair table inherited it. The shipped trial list was
#   therefore not reproducible even from a re-run of the ORIGINAL
#   scripts with the same seed.
#
#   That is fixed going forward -- steps 3, 4 and 5 now each impose a
#   canonical order, and two runs of step 5 produce byte-identical
#   trial lists. But the fix cannot retroactively make the shipped
#   list reproducible, so the one remaining way to check that the
#   SAMPLER is unchanged is to feed it the old order back.
#
#   Set to None to skip the ordering, in which case the reproduction
#   version will differ from the shipped list for this reason alone.
REPRODUCE_POOL_ORDER = "data/pair_analysis_v4_ec/haze_eq/all_pairs.csv"

OUTPUT_ROOT = "data/trials_v4_ext"

# ON_EXISTING
#   "new" writes to OUTPUT_ROOT_2, _3, ... rather than overwriting.
#   A trial list and its package must stay in step, so a half-replaced
#   directory is worse than a duplicate one.
ON_EXISTING = "new"

# --- Trial counts ---
#
# N_SAT / N_FOG / N_MIXED
#   Real trials of each kind. Equal counts by design. See the header
#   for why 40.
N_SAT   = 60 # 40
N_FOG   = 60 # 40
N_MIXED = 60 # 40

# MIXED_CONFLICT_FRAC
#   Share of the mixed trials that are conflict rather than together.
#   Half and half: `together` gives the largest effect and acts as the
#   floor check, `conflict` gives the crossover.
MIXED_CONFLICT_FRAC = 0.5

# N_CONSISTENCY
#   Pairs repeated later in the session with the sides swapped, so the
#   repeat is not recognisable. Gives a per-participant agreement rate,
#   which is what lets you tell "does not track these features" from
#   "was not attending". Essential when each participant is a data
#   point.
N_CONSISTENCY = 15

# N_CATCH
#   Trials whose instruction is replaced by something task-irrelevant,
#   so the response tests whether the participant is reading at all.
#
#   REGISTERED RULE: exclude a participant who fails THREE OR MORE.
#   The count is set by what that rule can actually detect, which is
#   worth stating because the first value chosen here could not detect
#   much.
#
#   A lapse on a catch trial is a coin flip, so it fails half the time.
#   Share of participants excluded by a three-failure rule:
#
#     share of trials lapsed    6 catch    10 catch
#       0.02 (a misclick)        0.0002      0.0009
#       0.10                     0.002       0.012
#       0.20                     0.016       0.070
#       0.40                     0.099       0.322
#       0.60                     0.256       0.617
#       1.00 (not reading)       0.656       0.945
#
#   At 6 trials the rule removed a participant who was not reading at
#   all only two times in three. At 10 it removes them 95 times in a
#   hundred, while a careful participant with a 2% misclick rate is
#   removed about once in a thousand.
#
#   WHAT IT STILL CANNOT DO, and this is the reason the analysis plan
#   does not rest on it: a participant lapsing on a FIFTH of trials is
#   caught only 7% of the time even at 10. Catch trials remove gross
#   inattention; moderate inattention is left to the pre-registered
#   sensitivity analysis with a per-person lapse parameter, because it
#   attenuates every fitted weight by roughly (1 - lapse rate) and so
#   mimics a group difference in cue weighting.
#
#   Suzuki et al. used two per block of 32 and excluded participants
#   failing four of eight.
N_CATCH = 10

# N_PRACTICE
#   Unanalysed trials at the start, to settle the participant into the
#   task before anything is measured.
N_PRACTICE = 6

# PRACTICE_TYPES / PRACTICE_MAG_QUANTILE
#   WHICH pairs the practice trials are drawn from, and this changed.
#
#   They used to come from the `together` pool alone, because those
#   have the most obvious difference. The cost was not obvious: every
#   practice trial then showed one image that was BOTH hazier and less
#   colourful, so the whole warm-up taught that the two cues move
#   together and that the more heavily edited image is the one to pick.
#   A participant who generalised that would carry it into the
#   conflict trials, which are precisely the trials where the cues do
#   NOT move together and where the choice is supposed to reveal which
#   cue they follow.
#
#   Practice is now drawn from a MIX of types at small-to-medium
#   differences: `together` keeps a recognisable example in, and
#   `sat_only` and `fog_only` show each cue moving on its own.
#   `conflict` is deliberately excluded -- a warm-up should not
#   rehearse the trials the hypothesis turns on.
#
#   PRACTICE_MAG_QUANTILE caps the difference at this quantile of each
#   pool, so practice is representative rather than a run of the
#   easiest pairs in the set. Still above threshold: the pools have
#   already been filtered by step 4's minimum differences.
PRACTICE_TYPES = ("sat_only", "fog_only", "together")
PRACTICE_MAG_QUANTILE = 0.60

# --- Sampling constraints ---
#
# N_MAGNITUDE_BINS
#   Used only when BIN_MODE is "quantile" (the reproduction version).
#   Under "fixed", the bin count comes from BIN_EDGES_SD, which is per
#   type -- five bins for colour, three for the fog-driven types.
N_MAGNITUDE_BINS = 3

# BALANCED_CONFLICT_ONLY
#   Draw conflict trials only from pairs step 4 flagged as balanced.
#
#   The flag now means balanced IN CORPUS-SD UNITS -- the units the
#   registered model fits -- so a balance of 1 means a participant with
#   equal weights in that model is at 50/50. Step 4 normalised by each
#   scene's own range until 2026-10-17; the figure quoted here before,
#   that the fog cue runs about 1.65x stronger than the colour cue, was
#   an artefact of that choice. In corpus-SD units it is 1.43x on the
#   original ladder and 1.03x on the extended one -- close to even.
#
#   The filter is therefore kept for PRECISION rather than to correct a
#   large bias: a conflict trial where the two cues are equally strong
#   is the one that actually forces a participant to reveal which cue
#   they follow. Turning this off exists only so the effect of the
#   constraint can be measured.
BALANCED_CONFLICT_ONLY = True

# MAX_PAIRS_PER_SCENE_PER_TYPE
#   Cap on how many trials of one type any single scene contributes, so
#   no photograph dominates.
#
#   Counted ACROSS the whole draw for that type, not within each
#   magnitude bin -- see stratified_sample for why that distinction
#   matters and what v2 did instead.
#
#   1 is deliberate and stricter than v2's value. Step 4 checked it:
#   with 47 scenes and 40 trials per type, enough scenes can supply one
#   pair of every type, so a cap of 1 costs nothing. It matters because
#   two scenes (courtyard and the train platform) have fog that barely
#   drains colour and so hold an outsized share of the conflict pool --
#   at a cap of 1 they contribute one conflict trial each, like
#   everyone else.
#
#   The ceiling is one trial per scene per type, so 4 analysed trials
#   from any one scene out of 120. Raise it if a pool is too thin to
#   fill a type, which the draw will say plainly.
#
#   None computes a looser cap from the scene count, as v2 did.
MAX_PAIRS_PER_SCENE_PER_TYPE = 2

# --- Presentation ---
#
# CONSISTENCY_MIN_GAP
#   Minimum trials between a pair and its repeat, so the repeat is not
#   recognised as one.
CONSISTENCY_MIN_GAP = 25

# CATCH_MODE
#   WHAT A CATCH TRIAL ACTUALLY IS. The two modes differ in what the
#   participant sees and in what counts as a failure, and they are not
#   equally good checks.
#
#   "no_images"  THE CURRENT DESIGN. No images. The screen shows only
#                an instruction naming one arrow key, UP or DOWN, and
#                takes a single press. The sequence of directions lives
#                in the experiment's own config (balanced, half up and
#                half down), not here.
#
#                A participant who presses the task keys without
#                reading fails EVERY catch trial. One who presses at
#                random among the four live keys passes one in four.
#
#   "images"     The pair is shown as on a normal trial, with the
#                question replaced by an instruction naming a SIDE to
#                click. A participant who presses left or right without
#                reading therefore passes HALF of them by chance, which
#                makes this the weaker check -- roughly 2.5x less
#                sensitive to moderate inattention. Exclusion at three
#                or more failures of ten:
#
#                  share of trials lapsed   no_images   images
#                         0.10                 0.034     0.012
#                         0.20                 0.180     0.070
#                         0.40                 0.617     0.322
#                    presses left/right        1.000     0.945
#
#                assuming that in no-image mode a lapsed press lands on
#                a random one of the four live keys (so it fails three
#                times in four) and in image mode on left or right (so
#                it fails once in two).
#
#   THE MODE IS WRITTEN INTO THE TRIAL LIST, on every catch row, as
#   `catch_mode`. The experiment carries its own setting and must stop
#   with an error if the two disagree -- which is how the mismatch this
#   setting exists to fix went unnoticed: the list used to carry an
#   instruction and an expected response for a catch trial the
#   experiment never ran.
#
#   DECIDE THE MODE BEFORE THE FREEZE. Catch rows keep their image pair
#   in both modes, so switching needs no new images -- but the mode is
#   in the file, so a switch changes the trial list, its checksum and
#   its preload list, and needs a re-freeze. The power results still
#   hold across a switch, because the 180 analysed trials are
#   untouched by it.
CATCH_MODE = "no_images"

# MAIN_INSTRUCTION
#   The question asked on every real trial. Kept here so the wording
#   lives with the trial list rather than only in the experiment
#   software.
#
#   Catch rows get NO instruction -- the column is left empty on them,
#   and the experiment supplies that text from its own files in either
#   mode.
MAIN_INSTRUCTION = ("Which of these looks more like how the world "
                    "looked to you?")

# --- Packaging ---
PACKAGE = True

# SHIP_FORMAT / SHIP_KWARGS
#   Format for the packaged images.
#
#   "png"  lossless, zero drift, nothing to justify.
#   "jpeg" roughly 3-5x smaller. ALWAYS pass subsampling=0 (4:4:4):
#          JPEG's default halves the resolution of the colour channels,
#          and saturation is a dependent variable here, so that is
#          exactly the wrong economy.
#
#   Do NOT resize. Features were measured on a 256 downsample of the
#   512 image, and that downsampling smooths fine detail. A file that
#   is already smaller skips the smoothing, and contrast_local lands
#   about 9 units away.
SHIP_FORMAT = "png"
SHIP_KWARGS = dict(format="PNG")
# For JPEG instead:
# SHIP_FORMAT = "jpeg"
# SHIP_KWARGS = dict(format="JPEG", quality=95, subsampling=0)

# REMEASURE_AFTER_CONVERT
#   Recompute sat_mean and contrast_local on the CONVERTED files and
#   overwrite d_sat, d_fog, z_sat and z_fog in the packaged trial list,
#   so the predictors describe the images participants actually see
#   rather than their lossless ancestors. Pointless for PNG (nothing
#   changes), worth it for any lossy format -- and it doubles as a
#   check that STIMULI_SRC really is the right folder.
REMEASURE_AFTER_CONVERT = True

# REMEASURE_MAX_SHIFT
#   Guard on the re-measurement. Conversion shifts these features by a
#   fraction of a unit. A shift beyond this means the stimuli directory
#   is not the one the pair table was built from -- the two fog
#   families being the obvious way to get that wrong -- and the script
#   stops rather than silently overwriting the predictors with
#   unrelated values.
REMEASURE_MAX_SHIFT = 0.6

# FEATURE_TARGET_SIZE
#   Only used by the re-measurement. A MEASUREMENT CONVENTION inherited
#   from step 04, not a CLIP requirement. Do not change it.
FEATURE_TARGET_SIZE = (256, 256)

# FOG_SIGN
#   Multiplier on the raw fog-feature difference so a POSITIVE value
#   always means "image A is foggier". Used ONLY by the re-measurement:
#   the sampling reads d_fog straight from all_pairs.csv, where step 4
#   has already applied the sign.
FOG_SIGN = -1.0

# JSON_COLUMNS
#   Columns written to trial_list.json, the version a web experiment
#   loads. `instruction` is deliberately omitted: the question is the
#   same on every real trial, and the catch instruction belongs in the
#   experiment code rather than repeated on every row. The CSV keeps it
#   as the authoritative record.
#
#   NaN becomes null in JSON. `catch_mode` is set only on catch trials
#   and `repeat_of` only on consistency trials, so those are null on
#   most rows -- treat null as "not applicable".
#
#   THERE IS NO `correct_response`. There used to be, set to "right" on
#   every catch row, describing a catch trial the experiment does not
#   run: the real one shows no images and takes an arrow key named by
#   the experiment's own config. A column the experiment ignores is how
#   that mismatch survived, so the expected response now lives only
#   where the instruction does -- in the experiment.
#   THE ORIENTED PREDICTORS, `pred_sat` and `pred_fog`, are the ones
#   the registered model fits. They are in CORPUS-SD UNITS and are
#   signed so that POSITIVE MEANS IMAGE A IS THE PREDICTED CHOICE --
#   the less colourful image, the foggier image. A coefficient is then
#   "per SD of the feature", and a positive coefficient means the
#   participant chose as predicted.
#
#   `pred_sat` is the negated `d_sat`; `pred_fog` equals `d_fog`
#   divided by its SD, with no sign change, because step 4 applied
#   FOG_SIGN when it built the table. Negating both would flip the fog
#   coefficient.
#
#   The raw signed differences stay beside them, and the
#   per-scene-range versions (`pred_sat_range`, `pred_fog_range`) are
#   carried in the CSV for anyone who wants scene-relative units.
JSON_COLUMNS = [
    "trial", "trial_type", "analysed", "scene_id", "category",
    "left_path", "right_path", "a_is_left",
    "pred_sat", "pred_fog",
    "d_sat", "d_fog", "z_sat", "z_fog", "mag_bin",
    "catch_mode", "repeat_of",
]

CATCH_MODES = ("no_images", "images")

# RANDOM_SEED
#   The seed of the list that would SHIP. Fixed in advance and not
#   chosen on any result -- see the header.
RANDOM_SEED = 42

# N_SEEDS / SEED_SPACING
#   How many draws of each version to build. The first is RANDOM_SEED
#   and is the real one: figure, report and package. The rest are
#   replicates for step 6, written under <root>/seeds/, so that a
#   difference between two versions can be read against the spread of
#   one version across draws.
#
#   20 is enough to see the spread without being the dominant cost: the
#   replicates skip packaging, which is the slow part.
#
#   Set N_SEEDS = 1 to build only the shipping list.
N_SEEDS = 20
SEED_SPACING = 1000

# SEED_SWEEP_SKIP
#   Versions built at the shipping seed only. The reproduction control
#   is checked against one fixed shipped list, so replicate draws of it
#   would answer nothing.
SEED_SWEEP_SKIP = ("reproduction",)

# STREAM_NAMES
#   Every named draw, for the record and for the report. One generator
#   each under "per_type".
#
#   The four type names are also the keys the sampler looks up, so they
#   must match REAL_TYPES. The rest are the draws that come after:
#
#     catch, practice      which unused pairs become catch and practice
#     consistency          which trials are repeated
#     sides_*              left/right assignment, per group
#     main_order           the shuffle before the consistency draw
#     body_order           the shuffle that interleaves the types
#     spacing              the search that separates each repeat from
#                          its original
STREAM_NAMES = ("sat_only", "fog_only", "together", "conflict",
                "catch", "practice", "consistency",
                "sides_main", "sides_catch", "sides_practice",
                "main_order", "body_order", "spacing")

REAL_TYPES = ["sat_only", "fog_only", "together", "conflict"]


def seed_list():
    """The seeds, shipping seed first."""
    return [RANDOM_SEED + SEED_SPACING * i for i in range(N_SEEDS)]


# ============================================================
# Small helpers
# ============================================================

def check_settings():
    """Settings that would otherwise fail late or silently."""
    if CATCH_MODE not in CATCH_MODES:
        raise ValueError(
            f"CATCH_MODE must be one of {CATCH_MODES}, got "
            f"{CATCH_MODE!r}. The value is written into every catch "
            "row and the\nexperiment compares it against its own "
            "setting, so a typo here would stop\nthe experiment on "
            "load rather than quietly mis-running it -- but it is "
            "cheaper\nto catch now.")


def resolve_output_root():
    """OUTPUT_ROOT, never overwriting an existing build."""
    def occupied(d):
        if not os.path.isdir(d):
            return False
        return any(os.path.isfile(os.path.join(d, lab,
                                               "trial_list.csv"))
                   for lab, *_ in VERSIONS)

    if not occupied(OUTPUT_ROOT):
        return OUTPUT_ROOT
    if ON_EXISTING == "error":
        raise FileExistsError(
            f"{OUTPUT_ROOT} already holds trial lists. Move it, or set "
            "ON_EXISTING = 'new'.")
    n = 2
    while occupied(f"{OUTPUT_ROOT}_{n}"):
        n += 1
    print(f"  NOTE: {OUTPUT_ROOT} already holds trial lists; writing "
          f"to {OUTPUT_ROOT}_{n} instead.")
    return f"{OUTPUT_ROOT}_{n}"


class Streams:
    """The random generators for one build.

    Under "per_type" each named draw gets its own generator, seeded
    from (seed, name), so no draw can be moved by a change to another
    one. Under "single" every call returns the same generator, which is
    the legacy behaviour the reproduction control needs.

    The name is hashed with SHA-256 rather than Python's hash(), which
    is salted per process and would make a build unreproducible between
    runs -- the exact fault this script has already been bitten by once
    (step 3's imap_unordered manifest order).
    """

    def __init__(self, seed, mode):
        if mode not in ("per_type", "single"):
            raise ValueError(
                f"streams must be 'per_type' or 'single', got {mode!r}")
        self.seed = int(seed)
        self.mode = mode
        self._single = np.random.default_rng(self.seed)
        self._made = {}
        self._used = []

    def __call__(self, name):
        if name not in STREAM_NAMES:
            raise KeyError(
                f"{name!r} is not in STREAM_NAMES. Add it there rather "
                "than reusing another draw's stream.")
        if name not in self._used:
            self._used.append(name)
        if self.mode == "single":
            return self._single
        if name not in self._made:
            tag = int.from_bytes(
                hashlib.sha256(name.encode()).digest()[:8], "big")
            self._made[name] = np.random.default_rng([self.seed, tag])
        return self._made[name]

    def shuffle_seed(self, name, offset):
        """A `random_state` for a pandas shuffle.

        Under "single" this is seed + offset, which is what the shipped
        list used. Under "per_type" it comes from that draw's own
        generator instead, so the two shuffles are independent of each
        other and of the seed arithmetic.
        """
        if self.mode == "single":
            return self.seed + offset
        return int(self(name).integers(1 << 30))

    def describe(self):
        if self.mode == "single":
            return (f"single (legacy): one generator, seed {self.seed}; "
                    "every draw depends on the ones before it")
        return (f"per_type: {len(self._used)} generators, each seeded "
                f"from (seed {self.seed}, draw name)")


def pair_key(a, b):
    """Identify a pair by its IMAGES, ignoring which is A and which B.

    Pairs are enumerated in both directions by step 4, so (a, b) and
    (b, a) are two rows describing the SAME two pictures. Everything
    that asks "has the participant seen these two images together
    already?" must use this key.

    v2 compared the ordered tuple in one place and the sorted one in
    another. The mismatch let a pair drawn as a real trial come back as
    practice in the opposite order -- the exact double-exposure the
    separate practice pool exists to prevent -- and left the repeat
    ordering unable to find its original. Rare, silent, and avoided
    entirely by having one key.
    """
    return (a, b) if str(a) <= str(b) else (b, a)


def keys_of(df):
    return [pair_key(a, b) for a, b in zip(df.file_a, df.file_b)]


# ============================================================
# Sampling
# ============================================================

SD_OF = {"d_sat": 39.182, "d_fog": 9.348}


def assign_bins(pool, mag_col, mode, edges, report_to=None,
                label="", n_trials=None, max_per_scene=None):
    """Add a `mag_bin` column, and say which bins are usable.

    Returns (pool, usable bins). Two modes:

      "fixed"     value ranges in corpus-SD units, shared across
                  versions, so bin 2 means the same difference
                  everywhere.
      "quantile"  the legacy rank-based bins, equal counts per bin,
                  recomputed per pool.

    A bin holding fewer than MIN_PAIRS_PER_BIN pairs is reported as
    unusable rather than silently under-filled. Emptying the low bins
    is what a colour cut is FOR, so it is an expected outcome to
    record, not a fault -- but a bin quietly receiving fewer trials
    than its share would be one.
    """
    pool = pool.copy()
    mag = pool[mag_col].abs()
    if mode == "quantile":
        pool["mag_bin"] = pd.qcut(
            mag.rank(method="first"), N_MAGNITUDE_BINS,
            labels=range(1, N_MAGNITUDE_BINS + 1)).astype(int)
        n_bins = N_MAGNITUDE_BINS
    elif mode == "fixed":
        if not edges:
            raise ValueError(
                f"BIN_EDGES_SD has no entry for {label!r}; fixed "
                "binning needs one edge tuple per type.")
        sd = SD_OF[mag_col]
        pool["mag_bin"] = np.digitize(mag / sd, edges) + 1
        n_bins = len(edges) + 1
    else:
        raise ValueError(
            f"BIN_MODE must be 'fixed' or 'quantile', got {mode!r}")

    counts = {b: int((pool.mag_bin == b).sum())
              for b in range(1, n_bins + 1)}
    scenes = {b: int(pool.loc[pool.mag_bin == b, "scene_id"].nunique())
              for b in range(1, n_bins + 1)}
    usable = [b for b, c in counts.items() if c >= MIN_PAIRS_PER_BIN]
    thin = [b for b, c in counts.items() if 0 < c < MIN_PAIRS_PER_BIN]
    empty = [b for b, c in counts.items() if c == 0]

    # Now the constraint that actually binds: a bin must have enough
    # DISTINCT SCENES to supply its share under the per-scene cap.
    # Iterated, because dropping one bin raises every other bin's
    # share and can put a second out of reach.
    starved = []
    if n_trials and max_per_scene:
        while usable:
            share = int(np.ceil(n_trials / len(usable)))
            short = [b for b in usable
                     if scenes[b] * max_per_scene < share]
            if not short:
                break
            starved += short
            usable = [b for b in usable if b not in short]

    if report_to is not None and starved:
        note = (f"bins {sorted(starved)} dropped: too few scenes to "
                f"supply their share at a cap of {max_per_scene} "
                "(scenes per bin: "
                + ", ".join(f"{b}:{scenes[b]}" for b in sorted(starved))
                + ")")
        print(f"    {note}")
        report_to.append((f"{label} bins starved", note))
    if report_to is not None and (thin or empty):
        if EMPTY_BIN_RULE == "error":
            raise RuntimeError(
                f"[{label}] bins {empty + thin} hold fewer than "
                f"{MIN_PAIRS_PER_BIN} pairs ({counts}). Set "
                "EMPTY_BIN_RULE = 'spread' to divide the trials over "
                "the remaining bins, or widen the cut.")
        note = (f"bins {empty} empty, {thin} thin (<"
                f"{MIN_PAIRS_PER_BIN}); {len(usable)} usable, trials "
                "spread evenly over those")
        print(f"    {note}")
        report_to.append((f"{label} bins dropped", note))
    return pool, usable, counts


def stratified_sample(pool, n, rng, mag_col, n_bins,
                      max_per_scene=None, mode="quantile", edges=None,
                      report_to=None, label="", preserve_order=False):
    """Draw `n` pairs spread across magnitude bins and scenes.

    Random sampling would clump: most pairs sit at mid-range
    magnitudes, so a uniform draw leaves the extreme bins thin and the
    dose-response curve unsupported at its ends. Binning first, then
    drawing equally from each bin, fixes that.

    THE SCENE CAP IS COUNTED ACROSS THE WHOLE DRAW, not within each
    bin. Counting per bin -- which is what v2 did -- quietly multiplies
    the cap by the number of bins: a "cap of 1" became one trial per
    scene per bin, so three. That is exactly the concentration the cap
    exists to prevent, and on this set it let one scene supply seven of
    the analysed trials.

    Bins are visited in a random order so that no bin systematically
    gets first pick of the scenes.
    """
    if not len(pool):
        raise ValueError("Empty pool; cannot sample.")

    pool, usable, counts = assign_bins(pool, mag_col, mode, edges,
                                       report_to, label, n,
                                       max_per_scene)
    if not usable:
        raise RuntimeError(
            f"[{label}] no bin holds {MIN_PAIRS_PER_BIN} pairs "
            f"({counts}). The cut is too deep for this pool.")

    # Trials are divided over the USABLE bins, so a cut that empties
    # the low end costs range rather than trials.
    nb = len(usable)
    per_bin = [n // nb] * nb
    for k in range(n % nb):
        per_bin[k] += 1
    want_of = dict(zip(usable, per_bin))

    order = list(usable)
    rng.shuffle(order)

    out, taken, used = [], set(), {}
    for b in order:
        want = want_of[b]
        sub = pool[pool.mag_bin == b]
        if not len(sub):
            print(f"    WARNING: bin {b} is empty")
            continue
        # Sort before shuffling, so the draw depends on the pool's
        # CONTENTS and the seed, never on the order the rows arrived
        # in. Without this, a step-3 re-run that happened to finish
        # its scenes in a different order would produce a different
        # trial list from identical settings and the same seed.
        #
        # preserve_order skips it, for the reproduction control alone:
        # that version has had the earlier run's row order restored on
        # purpose, and sorting would throw it away again.
        if not preserve_order:
            sub = sub.sort_values(["scene_id", "file_a", "file_b"],
                                  kind="stable")
        sub = sub.sample(frac=1.0, random_state=rng.integers(1 << 30))
        # Greedy pass: take a pair only if its pictures have not been
        # used already (in either direction) and its scene still has
        # room under the cap.
        rows = []
        for pos, (sid, fa, fb) in enumerate(
                zip(sub.scene_id, sub.file_a, sub.file_b)):
            if len(rows) >= want:
                break
            k = pair_key(fa, fb)
            if k in taken:
                continue
            if (max_per_scene is not None
                    and used.get(sid, 0) >= max_per_scene):
                continue
            rows.append(pos)
            taken.add(k)
            used[sid] = used.get(sid, 0) + 1
        if len(rows) < want:
            print(f"    WARNING: bin {b} wanted {want}, could place "
                  f"{len(rows)} under a cap of {max_per_scene} per "
                  "scene")
        out.append(sub.iloc[rows])

    got = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return got.sort_values("mag_bin").reset_index(drop=True)


def apply_colour_cut(pairs, cut, report):
    """Drop colour-only pairs below a fixed |d_sat| threshold.

    Only `sat_only` is touched. The mixed types already require a fog
    difference, and cutting them would change what `together` and
    `conflict` mean rather than how large they are.
    """
    if not cut:
        report.append(("colour cut", "none"))
        return pairs
    m = pairs.type == "sat_only"
    keep = m & (pairs.d_sat.abs() >= cut)
    before = int(m.sum())
    out = pairs[(~m) | keep].reset_index(drop=True)
    after = int((out.type == "sat_only").sum())
    lo = pairs.loc[m, "d_sat"].abs().min()
    print(f"  colour cut at |d_sat| >= {cut} "
          f"({cut / SD_OF['d_sat']:.3f} SD): "
          f"{after:,} of {before:,} colour-only pairs kept")
    print(f"    range dropped: {lo:.1f}-{cut} sat_mean units")
    report.append(("colour cut",
                   f"|d_sat| >= {cut} ({cut / SD_OF['d_sat']:.3f} SD)"))
    report.append(("colour pairs after cut",
                   f"{after:,} of {before:,} "
                   f"({after / max(before, 1):.0%})"))
    report.append(("colour range dropped",
                   f"{lo:.1f}-{cut} sat_mean units"))
    return out


def build_main_trials(pairs, streams, report, colour_cut=None,
                      bin_mode="fixed", preserve_order=False):
    """Sample the real trials of each type, plus practice and catch.

    Practice and catch pairs are drawn SEPARATELY and never appear
    among the analysed trials: any pair seen twice gives the second
    presentation the advantage of prior exposure.
    """
    print("\n" + "=" * 60)
    print("SAMPLING TRIALS")
    print("=" * 60)

    pairs = apply_colour_cut(pairs, colour_cut, report)
    print(f"  bin mode: {bin_mode}")
    report.append(("bin mode", bin_mode))

    n_scenes = pairs.scene_id.nunique()
    cap = MAX_PAIRS_PER_SCENE_PER_TYPE
    if cap is None:
        # A little headroom over perfect balance, so a scene short of
        # one type does not block the draw.
        cap = int(np.ceil(max(N_SAT, N_FOG, N_MIXED) / n_scenes)) + 2
    print(f"  {n_scenes} scenes, cap {cap} trial(s) per scene per type")

    n_conflict = int(round(N_MIXED * MIXED_CONFLICT_FRAC))
    n_together = N_MIXED - n_conflict

    conflict_pool = pairs[pairs.type == "conflict"]
    if BALANCED_CONFLICT_ONLY:
        if "balanced" not in pairs.columns:
            raise KeyError(
                "all_pairs.csv has no `balanced` column; re-run step 4 "
                "(analyse_pairs_v4.py).")
        conflict_pool = conflict_pool[conflict_pool.balanced]

    specs = [
        ("sat_only", N_SAT, "d_sat", pairs[pairs.type == "sat_only"]),
        ("fog_only", N_FOG, "d_fog", pairs[pairs.type == "fog_only"]),
        ("together", n_together, "d_fog",
         pairs[pairs.type == "together"]),
        ("conflict", n_conflict, "d_fog", conflict_pool),
    ]

    chunks = []
    for name, want, mag_col, pool in specs:
        print(f"\n  {name}: want {want}, pool {len(pool):,}")
        if len(pool) < want:
            raise RuntimeError(
                f"Pool for {name} has only {len(pool)} pairs but "
                f"{want} trials were requested. Loosen the thresholds "
                "in step 4, lower the trial count, or raise "
                "MAX_PAIRS_PER_SCENE_PER_TYPE.")
        got = stratified_sample(pool, want, streams(name), mag_col,
                                N_MAGNITUDE_BINS, cap,
                                mode=bin_mode,
                                edges=BIN_EDGES_SD.get(name),
                                report_to=report, label=name,
                                preserve_order=preserve_order)
        if len(got) < want:
            raise RuntimeError(
                f"Drew only {len(got)} of {want} {name} trials at a "
                f"cap of {cap} per scene. Raise "
                "MAX_PAIRS_PER_SCENE_PER_TYPE or lower the count.")
        got = got.assign(trial_type=name, mag_driver=mag_col)
        print(f"    drew {len(got)}, {got.scene_id.nunique()} scenes, "
              f"bins {sorted(got.mag_bin.value_counts().to_dict().items())}")
        chunks.append(got)
        report.append((f"{name} trials", str(len(got))))

    main = pd.concat(chunks, ignore_index=True)

    # --- Practice and catch, from pairs NOT used as real trials ---
    #
    # CATCH uses `together`: the difference is obvious there, which is
    # what makes a replaced instruction stand out as a test of
    # attention rather than blend into a hard discrimination.
    #
    # PRACTICE does not, and that was a change. Drawn from `together`
    # alone, every warm-up trial showed one image that was both hazier
    # AND less colourful, teaching that the two cues move together and
    # that the more heavily edited image is the one to pick. Carried
    # into the conflict trials -- where the cues disagree, and where
    # the choice is the measurement -- that is a trained bias on
    # exactly the trials the hypothesis turns on.
    #
    # PRACTICE IS DRAWN FIRST, AND THAT ORDER IS DELIBERATE. Both
    # draws exclude pairs already taken, so whichever goes second
    # depends on how many trials the first one took. Raising N_CATCH
    # would therefore have moved the practice trials -- four fewer
    # rows in every practice pool, and a different pick from each --
    # even though nothing about the warm-up had changed.
    #
    # Practice is the one to hold still: it is what a participant
    # learns before anything is measured, so it should not shift
    # because an attention check was lengthened. Catch is drawn from
    # thousands of obvious pairs and does not care which ones it gets.
    used = set(keys_of(main))

    def unused(pool):
        if not len(pool):
            return pool
        return pool[~pd.Series(keys_of(pool),
                               index=pool.index).isin(used)]

    # practice: round-robin over the types, smallest-to-medium
    # differences only, one scene used at most once.
    #
    # One draw per slot off the practice stream, in a fixed type order,
    # so a change to the colour pool changes the colour practice trial
    # and not the others. The scene exclusion still couples them -- a
    # different colour pick excludes a different scene -- which is why
    # practice is reported as expected-to-differ across LADDERS, though
    # it is now invariant to N_CATCH.
    want = [PRACTICE_TYPES[i % len(PRACTICE_TYPES)]
            for i in range(N_PRACTICE)]
    prac_rows, prac_scenes = [], set()
    for t in want:
        drv = "d_sat" if t == "sat_only" else "d_fog"
        pool = unused(pairs[pairs.type == t])
        if not len(pool):
            continue
        thr = pool[drv].abs().quantile(PRACTICE_MAG_QUANTILE)
        sub_ = pool[(pool[drv].abs() <= thr)
                    & (~pool.scene_id.isin(prac_scenes))]
        if not len(sub_):
            sub_ = pool[~pool.scene_id.isin(prac_scenes)]
        if not len(sub_):
            continue
        row = sub_.sample(
            1, random_state=streams("practice").integers(1 << 30))
        prac_rows.append(row.assign(trial_type="practice",
                                    mag_driver=drv, mag_bin=0))
        prac_scenes.add(row.scene_id.iloc[0])
        used |= set(keys_of(row))
    if len(prac_rows) < N_PRACTICE:
        raise RuntimeError(
            f"Could place only {len(prac_rows)} of {N_PRACTICE} "
            f"practice trials from {PRACTICE_TYPES}. Lower "
            "N_PRACTICE or widen PRACTICE_MAG_QUANTILE.")
    prac = pd.concat(prac_rows, ignore_index=True)

    # catch, from `together` pairs neither the analysed trials nor
    # practice has taken.
    catch_pool = unused(pairs[pairs.type == "together"])
    if len(catch_pool) < N_CATCH:
        raise RuntimeError(
            f"Only {len(catch_pool)} unused `together` pairs for "
            f"{N_CATCH} catch trials.")
    catch = (catch_pool.sample(N_CATCH,
                               random_state=streams("catch")
                               .integers(1 << 30))
             .assign(trial_type="catch", mag_driver="d_fog",
                     mag_bin=0))
    used |= set(keys_of(catch))

    mix = prac.type.value_counts().to_dict()
    print(f"\n  practice: {len(prac)} from "
          f"{', '.join(f'{v} {k}' for k, v in sorted(mix.items()))}, "
          f"at or below the {PRACTICE_MAG_QUANTILE:.0%} magnitude "
          "quantile")
    print(f"  catch   : {len(catch)} from unused `together` pairs, "
          "drawn after practice")
    print("  neither overlaps the analysed trials, in either "
          "direction")
    report.append(("practice trials",
                   f"{len(prac)} ("
                   + ", ".join(f"{v} {k}" for k, v in sorted(mix.items()))
                   + ")"))
    report.append(("catch trials", str(len(catch))))
    return main, prac, catch


# ============================================================
# Assemble the session
# ============================================================

def _balanced_sides(labels, rng):
    """Exactly half of each group gets True, assigned at random.

    Independent coin flips give a binomial spread -- with 40 trials per
    type the realised split can easily land at 0.40 or 0.53. That
    matters because the model has no intercept: the design is
    symmetric, so there is no baseline preference to absorb. A
    participant with a side preference plus unequal side assignment
    creates exactly such a constant, and with nowhere to go it leaks
    into whichever weight correlates with the imbalance.

    Simulated on a participant who tracks NOTHING but prefers one side,
    the spurious weight was:

        side preference     realised 0.46     forced 50/50
        mild  (0.5)             0.059            0.016
        moderate (1.0)          0.136            0.004
        strong (2.0)            0.214            0.006

    Not fatal against a minimum detectable weight near 0.86, but free
    to remove, and it pushes one weight rather than both -- unhelpful
    in a design whose headline result is b_sat against b_fog.

    Odd group sizes get the extra trial assigned at random, so no group
    is systematically biased.
    """
    out = np.zeros(len(labels), dtype=bool)
    for lab in pd.unique(labels):
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        if len(idx) % 2 and rng.random() < 0.5:
            half += 1
        out[idx[:half]] = True
    return out


def _assign_sides(df, labels, rng):
    """left/right assignment. `swap` means image B is shown on the left.

    The stored d_sat and d_fog always refer to A minus B, so the
    analysis is unaffected; left_file / right_file are what the
    software displays.
    """
    df = df.copy()
    swap = ~_balanced_sides(labels, rng)
    df["left_file"] = np.where(swap, df.file_b, df.file_a)
    df["right_file"] = np.where(swap, df.file_a, df.file_b)
    df["a_is_left"] = ~swap
    return df


def assemble(main, prac, catch, streams, report):
    """Add practice, consistency and catch trials, randomise sides, and
    order the session.

    Types are interleaved rather than blocked, so a participant does
    not meet a run of trials in which only the colour changes and work
    out the manipulation.
    """
    print("\n" + "=" * 60)
    print("ASSEMBLING THE SESSION")
    print("=" * 60)

    main = main.sample(
        frac=1.0,
        random_state=streams.shuffle_seed("main_order", 0)
    ).reset_index(drop=True)
    # Balanced WITHIN each trial type -- see _balanced_sides.
    main = _assign_sides(main, main.trial_type.to_numpy(),
                         streams("sides_main"))

    # --- Consistency repeats ---
    # Drawn from across the types, repeated with the sides swapped so
    # the repeat is not recognisable as one.
    #
    # The stream is its own, so the POSITIONS drawn are the same across
    # versions -- but the rows at those positions are not, because the
    # colour trials differ. Consistency trials are not analysed, so
    # this is reported rather than checked.
    idx = streams("consistency").choice(len(main), N_CONSISTENCY,
                                       replace=False)
    rep = main.iloc[idx].copy()
    rep["left_file"], rep["right_file"] = (rep.right_file.to_numpy(),
                                           rep.left_file.to_numpy())
    rep["a_is_left"] = ~rep.a_is_left
    rep["trial_type"] = "consistency"
    # repeat_of is NOT set here. `idx` indexes the shuffled frame, and
    # the rows are about to be reordered twice and renumbered, so a
    # value stored now would point at nothing. Resolved at the end.

    catch = _assign_sides(catch, np.full(len(catch), "catch"),
                          streams("sides_catch"))
    # The mode the experiment must be running, written on every catch
    # row so a mismatch can be caught on load rather than discovered
    # in the data.
    catch["catch_mode"] = CATCH_MODE
    prac = _assign_sides(prac, np.full(len(prac), "practice"),
                         streams("sides_practice"))

    # --- Order: practice first, then everything interleaved ---
    body = pd.concat([main, rep, catch], ignore_index=True)
    body = body.sample(
        frac=1.0, random_state=streams.shuffle_seed("body_order", 1)
    ).reset_index(drop=True)

    # Space each repeat from its original, AND put the repeat second.
    # If the repeat came first, the later presentation would be the
    # analysed one -- so the analysed trial would be the contaminated
    # one, which is backwards.
    body = _space_repeats(body, streams("spacing"))
    body = _order_repeats_last(body)

    session = pd.concat([prac, body], ignore_index=True)
    session.insert(0, "trial", range(1, len(session) + 1))
    # The real question on the real trials; empty on catch rows, where
    # the experiment supplies its own text in either mode.
    session["instruction"] = np.where(
        session.trial_type == "catch", "", MAIN_INSTRUCTION)
    # `analysed` marks the trials entering the regression and the
    # proportion test: the four real types only. See the header for why
    # consistency trials are excluded.
    session["analysed"] = session.trial_type.isin(REAL_TYPES)
    session = _resolve_repeat_of(session)

    print(f"  {len(session)} trials total")
    print(session.trial_type.value_counts().to_string())
    report.append(("total trials", str(len(session))))
    report.append(("analysed trials", str(int(session.analysed.sum()))))
    report.append(("consistency (checked, not analysed)",
                   str(int((session.trial_type == "consistency").sum()))))
    return session


def _resolve_repeat_of(session):
    """Point each consistency trial at the trial it repeats.

    Must run AFTER `trial` is assigned. Storing a positional index
    before the reordering steps would leave every value pointing at an
    unrelated trial -- worse than an empty column, because the
    consistency rate is computed by pairing each repeat with its
    original and asking whether the same image was chosen. Pairing
    against unrelated trials returns eleven coin flips and an agreement
    rate near 50% for every participant, however attentive.

    Matching is by the unordered image pair. Practice and catch draw
    from pairs the analysed set never uses, so the only other trial
    sharing a repeat's pictures is its original.
    """
    session = session.reset_index(drop=True)
    key = keys_of(session)
    session["repeat_of"] = pd.NA

    for i, r in session.iterrows():
        if r.trial_type != "consistency":
            continue
        hits = [j for j, k in enumerate(key)
                if k == key[i]
                and session.loc[j, "trial_type"] != "consistency"]
        if len(hits) != 1:
            raise RuntimeError(
                f"Consistency trial {r.trial} matches {len(hits)} "
                "non-consistency trials by image pair; expected 1.")
        session.loc[i, "repeat_of"] = int(session.loc[hits[0], "trial"])

    # --- Assertions: both would have caught the bug described above ---
    con = session[session.trial_type == "consistency"]
    for _, r in con.iterrows():
        tgt = session[session.trial == r.repeat_of]
        if len(tgt) != 1:
            raise RuntimeError(
                f"repeat_of {r.repeat_of} does not resolve to a trial.")
        tgt = tgt.iloc[0]
        if pair_key(tgt.file_a, tgt.file_b) != pair_key(r.file_a,
                                                        r.file_b):
            raise RuntimeError(
                f"Trial {r.trial} points at trial {r.repeat_of}, which "
                "shows a different image pair.")
        if tgt.trial >= r.trial:
            raise RuntimeError(
                f"Trial {r.trial} points at trial {r.repeat_of}, which "
                "is not earlier. The repeat must follow its original, "
                "so the analysed trial is the uncontaminated one.")
    print(f"    repeat_of resolved and verified for {len(con)} "
          "consistency trials")
    return session


def _order_repeats_last(body):
    """Ensure each consistency repeat follows its original.

    Swaps the two rows' POSITIONS where necessary, leaving every other
    trial where it is, so the spacing achieved by _space_repeats
    survives.
    """
    body = body.reset_index(drop=True)
    pos = {}
    for i, k in enumerate(keys_of(body)):
        pos.setdefault(k, []).append(i)

    order = list(range(len(body)))
    n_swapped = 0
    for idxs in pos.values():
        if len(idxs) != 2:
            continue
        a, b = sorted(idxs)
        if body.loc[a, "trial_type"] == "consistency":
            order[a], order[b] = order[b], order[a]
            n_swapped += 1
    if n_swapped:
        print(f"    reordered {n_swapped} repeat(s) to follow their "
              "originals")
    return body.iloc[order].reset_index(drop=True)


def _space_repeats(body, rng, max_tries=400):
    """Nudge consistency repeats away from their originals.

    A repeat sitting a few trials after its original would be
    recognised, and the agreement rate would measure memory rather than
    consistency.
    """
    body = body.reset_index(drop=True)
    bad = []
    for _ in range(max_tries):
        pos = {}
        for i, k in enumerate(keys_of(body)):
            pos.setdefault(k, []).append(i)
        bad = [v for v in pos.values()
               if len(v) > 1
               and (max(v) - min(v)) < CONSISTENCY_MIN_GAP]
        if not bad:
            return body
        # Move the later member of the closest offending pair.
        v = min(bad, key=lambda x: max(x) - min(x))
        i = max(v)
        j = int(rng.integers(0, len(body)))
        order = list(range(len(body)))
        order[i], order[j] = order[j], order[i]
        body = body.iloc[order].reset_index(drop=True)
    print(f"    NOTE: could not space every repeat by "
          f"{CONSISTENCY_MIN_GAP} trials; {len(bad)} remain close.")
    return body


# ============================================================
# Checks
# ============================================================

def check_session(session, label, report):
    """Sanity checks on the finished list."""
    print("\n" + "=" * 60)
    print(f"CHECKS  [{label}]")
    print("=" * 60)
    an = session[session.analysed]

    # Collinearity of the two predictors across the analysed trials.
    # This is the number deciding whether the two weights are
    # separately estimable.
    r = float(np.corrcoef(an.d_sat, an.d_fog)[0, 1])
    vif = 1.0 / max(1e-12, 1 - r ** 2)
    print(f"  r(d_sat, d_fog) over analysed trials : {r:+.3f}")
    print(f"  VIF                                  : {vif:.2f}")
    if abs(r) > 0.6:
        print("    HIGH. The two weights will be hard to separate. "
              "Check the trial-type proportions.")
    report.append(("r(d_sat, d_fog)", f"{r:+.3f}"))
    report.append(("VIF", f"{vif:.2f}"))

    # Side balance per type. Forced to 0.50 within each type, so
    # anything else means the assignment did not take.
    side = an.groupby("trial_type").a_is_left.agg(["mean", "size"])
    print("\n  proportion with image A on the left, by type:")
    print(side.round(3).to_string())
    worst = float((side["mean"] - 0.5).abs().max())
    if worst > 0.03:
        print(f"    WARNING: worst deviation from 0.50 is {worst:.3f}; "
              "side balancing may not have applied.")
    report.append(("side balance (worst dev)", f"{worst:.3f}"))

    # Scene spread.
    per_scene = an.scene_id.value_counts()
    print(f"\n  analysed trials per scene: min {per_scene.min()}, "
          f"median {int(per_scene.median())}, max {per_scene.max()} "
          f"over {an.scene_id.nunique()} scenes")
    report.append(("scenes used", str(an.scene_id.nunique())))
    report.append(("max trials from one scene", str(per_scene.max())))

    # No pair may be seen twice except a consistency repeat.
    k = keys_of(session)
    dup = pd.Series(k).value_counts()
    multi = dup[dup > 1]
    n_con = int((session.trial_type == "consistency").sum())
    if len(multi) != n_con or (multi > 2).any():
        raise RuntimeError(
            f"{len(multi)} image pairs appear more than once but there "
            f"are {n_con} consistency trials. Every repeat must be a "
            "consistency trial and no pair may appear three times.")
    print(f"\n  repeated image pairs: {len(multi)}, all consistency "
          "repeats")

    # Consistency repeats: gap, and whether the sides were swapped. A
    # swapped repeat tests that the choice survives a change of
    # position as well as a gap in time.
    con = session[session.trial_type == "consistency"]
    if len(con):
        gaps, swaps = [], []
        for _, r2 in con.iterrows():
            o = session[session.trial == r2.repeat_of].iloc[0]
            gaps.append(int(r2.trial) - int(o.trial))
            swaps.append(bool(o.left_file == r2.right_file))
        print(f"  consistency repeats: min gap {min(gaps)}, median "
              f"{int(np.median(gaps))}, sides swapped "
              f"{sum(swaps)}/{len(swaps)}")
        report.append(("consistency min gap", str(min(gaps))))
        if sum(swaps) != len(swaps):
            print("    WARNING: not every repeat has its sides swapped.")

    # --- The catch trial says what it is ---
    #
    # The list used to carry an instruction and an expected response
    # for a catch trial the experiment does not run. These checks make
    # the declaration explicit and keep it consistent, so a mismatch
    # between the list and the experiment is caught on load.
    cat = session[session.trial_type == "catch"]
    if "catch_mode" not in session.columns:
        raise RuntimeError(
            "No `catch_mode` column. Every catch row must declare "
            "which kind of catch\ntrial the experiment is expected to "
            "run.")
    modes = set(cat.catch_mode.dropna())
    if modes != {CATCH_MODE}:
        raise RuntimeError(
            f"Catch rows declare {sorted(modes)}; expected exactly "
            f"{{{CATCH_MODE!r}}}. One list,\none mode.")
    stray = session[(session.trial_type != "catch")
                    & session.catch_mode.notna()]
    if len(stray):
        raise RuntimeError(
            f"{len(stray)} non-catch row(s) carry a catch_mode. The "
            "column marks catch\ntrials and nothing else.")
    if "correct_response" in session.columns:
        raise RuntimeError(
            "A `correct_response` column is present. The expected "
            "response belongs in\nthe experiment, with the "
            "instruction text; a column the experiment ignores\nis "
            "how the previous mismatch survived.")
    blank = cat.instruction.fillna("").astype(str).str.strip()
    if (blank != "").any():
        raise RuntimeError(
            "Catch rows carry instruction text. The experiment "
            "supplies it in either\nmode, so the column must be empty "
            "on those rows.")
    if (an.instruction.fillna("").astype(str).str.strip() == "").any():
        raise RuntimeError(
            "An analysed trial has no instruction. MAIN_INSTRUCTION is "
            "the record of\nthe question actually asked.")
    print(f"\n  catch trials: {len(cat)}, mode {CATCH_MODE!r}, "
          "no instruction and no expected\n    response in the list "
          "(the experiment owns both)")
    report.append(("catch mode", CATCH_MODE))

    print("\n  trials per magnitude bin, by type:")
    print(pd.crosstab(an.trial_type, an.mag_bin).to_string())
    return r, vif


def make_figure(session, label, out_dir):
    an = session[session.analysed]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for t, g in an.groupby("trial_type"):
        axes[0].scatter(g.d_sat, g.d_fog, s=28, alpha=0.75, label=t)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].axvline(0, color="k", lw=0.6)
    axes[0].set_xlabel("d sat_mean  (A - B)")
    axes[0].set_ylabel("d contrast_local  (A - B, + means A is foggier)")
    axes[0].set_title(f"The trials, in difference space  [{label}]\n"
                      "(spread, not clumped, is what makes the two "
                      "weights separable)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    ct = pd.crosstab(an.trial_type, an.mag_bin)
    ct.plot(kind="bar", stacked=True, ax=axes[1], rot=0)
    axes[1].set_xlabel("trial type")
    axes[1].set_ylabel("trials")
    axes[1].set_title("Magnitude bins per type\n"
                      "(the dose-response analysis needs all three)")
    axes[1].legend(title="bin", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trial_design.png"), dpi=150)
    plt.close(fig)


# ============================================================
# Packaging
# ============================================================

def _features_of(path, target_size):
    """sat_mean and contrast_local of a file, measured exactly as step
    04 measured the corpus: resize to target_size, then compute.

    contrast_local is the standard deviation of the Laplacian -- the
    same computation the pipeline uses throughout, so the values are
    comparable with the manifest.
    """
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB").resize(target_size,
                                                  Image.BILINEAR),
                         dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    S = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)[:, :, 1]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(S.mean()), float(cv2.Laplacian(g, cv2.CV_32F).std())


def package(session, label, stimuli_src, out_dir, report):
    """Write a self-contained folder ready to upload.

    Only the images the trial list references are copied -- about 200
    of the 1,175 rendered, which is the difference between a package
    that preloads and one that does not.
    """
    print("\n" + "=" * 60)
    print(f"PACKAGING  [{label}]")
    print("=" * 60)
    pkg_dir = os.path.join(out_dir, "package")
    img_dir = os.path.join(pkg_dir, "stimuli")
    os.makedirs(img_dir, exist_ok=True)

    used = sorted(set(session.file_a) | set(session.file_b))
    print(f"  {len(used)} distinct images from {stimuli_src}")
    print(f"  format: {SHIP_FORMAT} {SHIP_KWARGS}")

    ext = ".png" if SHIP_FORMAT.lower() == "png" else ".jpg"
    rename, rows, total = {}, [], 0
    for n, fn in enumerate(used, 1):
        src = os.path.join(stimuli_src, fn)
        if not os.path.isfile(src):
            raise FileNotFoundError(
                f"Stimulus not found: {src}\nIs STIMULI_SRC for "
                f"{label!r} the folder this pair table was built from?")
        out_name = os.path.splitext(fn)[0] + ext
        dst = os.path.join(img_dir, out_name)
        with Image.open(src) as im:
            im.convert("RGB").save(dst, **SHIP_KWARGS)
        rename[fn] = out_name
        n_bytes = os.path.getsize(dst)
        total += n_bytes
        rec = {"filename": out_name, "source": fn,
               "kB": round(n_bytes / 1024, 1)}
        if REMEASURE_AFTER_CONVERT:
            sat, fog = _features_of(dst, FEATURE_TARGET_SIZE)
            rec["sat_mean"], rec["contrast_local"] = sat, fog
        rows.append(rec)
        if n % 50 == 0:
            print(f"    {n}/{len(used)}", end="\r")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(pkg_dir, "manifest.csv"), index=False)
    mb = total / 1024 / 1024
    print(f"\n  wrote {len(used)} images, {mb:.1f} MB "
          f"({total / len(used) / 1024:.0f} kB each)")

    out = session.copy()
    out["left_path"] = "stimuli/" + out.left_file.map(rename)
    out["right_path"] = "stimuli/" + out.right_file.map(rename)

    # --- Re-measure the predictors on the converted files ---
    if REMEASURE_AFTER_CONVERT:
        m = manifest.set_index("source")
        sat_a = m.loc[out.file_a, "sat_mean"].to_numpy()
        sat_b = m.loc[out.file_b, "sat_mean"].to_numpy()
        fog_a = m.loc[out.file_a, "contrast_local"].to_numpy()
        fog_b = m.loc[out.file_b, "contrast_local"].to_numpy()

        old_sat, old_fog = out.d_sat.to_numpy(), out.d_fog.to_numpy()
        new_sat = sat_a - sat_b
        new_fog = FOG_SIGN * (fog_a - fog_b)

        # z_* are the normalised versions: each difference divided by
        # that scene's own range. Rescale rather than recompute, since
        # the ranges are unchanged by conversion.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_s = np.where(np.abs(old_sat) > 1e-9,
                               new_sat / old_sat, 1.0)
            ratio_f = np.where(np.abs(old_fog) > 1e-9,
                               new_fog / old_fog, 1.0)
        out["z_sat"] = out.z_sat.to_numpy() * np.abs(ratio_s)
        out["z_fog"] = out.z_fog.to_numpy() * np.abs(ratio_f)
        if {"pred_sat_range", "pred_fog_range"} <= set(out.columns):
            out["pred_sat_range"] = (out.pred_sat_range.to_numpy()
                                     * np.abs(ratio_s))
            out["pred_fog_range"] = (out.pred_fog_range.to_numpy()
                                     * np.abs(ratio_f))
        out["d_sat"], out["d_fog"] = new_sat, new_fog
        # The oriented predictors are derived, so they are recomputed
        # from the re-measured differences rather than scaled. Scaling
        # a signed quantity by |ratio| would be wrong wherever the
        # re-measurement crossed zero.
        out["pred_sat"] = -new_sat / SD_OF["d_sat"]
        out["pred_fog"] = new_fog / SD_OF["d_fog"]

        d_s = float(np.abs(new_sat - old_sat).max())
        d_f = float(np.abs(new_fog - old_fog).max())
        print(f"  re-measured: worst |d_sat| shift {d_s:.3f}, "
              f"worst |d_fog| shift {d_f:.3f}")
        report.append(("remeasure d_sat shift", f"{d_s:.3f}"))

        # A shift of tens of units means the files are not the ones the
        # pair table describes -- the two fog families being the
        # obvious way to get that wrong -- and re-measuring would
        # silently overwrite the predictors with unrelated values.
        if d_s > REMEASURE_MAX_SHIFT or d_f > REMEASURE_MAX_SHIFT:
            raise RuntimeError(
                f"[{label}] Re-measured features shifted by {d_s:.1f} "
                f"(sat) and {d_f:.1f} (fog), far beyond what conversion "
                f"can cause (limit {REMEASURE_MAX_SHIFT}).\n"
                f"STIMULI_SRC = {stimuli_src!r} probably does not hold "
                "the images this trial list was built from -- check "
                "that the fog family matches the pair table.")

    cols = ["trial", "trial_type", "analysed", "instruction",
            "scene_id", "category", "left_path", "right_path",
            "a_is_left", "pred_sat", "pred_fog",
            "pred_sat_range", "pred_fog_range",
            "d_sat", "d_fog", "z_sat", "z_fog",
            "mag_bin", "catch_mode", "repeat_of"]
    out[[c for c in cols if c in out.columns]].to_csv(
        os.path.join(pkg_dir, "trial_list.csv"), index=False)

    missing = [c for c in JSON_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(
            f"JSON_COLUMNS names columns that do not exist: {missing}")
    out[JSON_COLUMNS].to_json(
        os.path.join(pkg_dir, "trial_list.json"),
        orient="records", indent=1)
    print(f"  trial_list.json: {len(JSON_COLUMNS)} columns, "
          f"{len(out)} records")

    # --- preload ---
    #
    # In "no_images" mode a catch trial shows no pictures, so the
    # images its row happens to carry are never displayed and must not
    # be fetched. They still SHIP, in stimuli/, so a later switch to
    # image mode needs no new images -- but a preload list that names
    # files the experiment is told not to fetch is a list that cannot
    # be followed literally, and the experiment is specified to fetch
    # exactly this list.
    #
    # Dropped by IMAGE, not by row: an image a catch row uses may also
    # appear in an analysed trial, and that one is still needed.
    shown = session[session.trial_type != "catch"]
    keep = set(shown.file_a) | set(shown.file_b)
    if CATCH_MODE == "no_images":
        names = sorted(rename[f] for f in rename if f in keep)
        n_drop = len(rename) - len(names)
    else:
        names = sorted(rename.values())
        n_drop = 0
    with open(os.path.join(pkg_dir, "preload.json"), "w") as f:
        json.dump(["stimuli/" + n for n in names], f, indent=1)
    print(f"  preload.json: {len(names)} images"
          + (f"; {n_drop} catch-only image(s) shipped but not "
             "preloaded (no_images mode)" if n_drop else ""))
    report.append(("preload images", f"{len(names)} of {len(rename)}"))

    n_an = int(session.analysed.sum())
    with open(os.path.join(pkg_dir, "README.txt"), "w") as f:
        f.write(f"EXPERIMENT PACKAGE  ({label})\n")
        f.write("=" * 56 + "\n\n")
        f.write(f"  trials        {len(session)} ({n_an} analysed)\n")
        f.write(f"  images        {len(used)}\n")
        f.write(f"  format        {SHIP_FORMAT} {SHIP_KWARGS}\n")
        f.write(f"  total size    {mb:.1f} MB\n")
        f.write(f"  re-measured   {REMEASURE_AFTER_CONVERT}\n")
        f.write(f"  stimuli from  {stimuli_src}\n\n")
        f.write("FILES\n")
        f.write("  trial_list.csv   one row per trial, in order\n")
        f.write("  trial_list.json  the same trials as an array of\n")
        f.write("                   objects, for the experiment to\n")
        f.write("                   load. No `instruction` column --\n")
        f.write("                   keep the question text in code.\n")
        f.write("                   null means 'not applicable'.\n")
        f.write("  stimuli/         the images, referenced by\n")
        f.write("                   left_path and right_path\n")
        f.write("  manifest.csv     every image with its measured\n")
        f.write("                   sat_mean and contrast_local\n")
        f.write("  preload.json     the image paths to fetch before\n")
        f.write("                   the first trial. FETCH EXACTLY\n")
        f.write("                   THIS LIST. In 'no_images' catch\n")
        f.write("                   mode it omits images used only by\n")
        f.write("                   catch rows, which ship in stimuli/\n")
        f.write("                   but are never shown.\n")
        f.write("                   before the first trial\n\n")
        f.write("TRIAL LIST\n")
        f.write("  trial        presentation order\n")
        f.write("  trial_type   sat_only / fog_only / together /\n")
        f.write("               conflict / consistency / catch /\n")
        f.write("               practice\n")
        f.write("  analysed     True ONLY for the four real types.\n")
        f.write("               Practice, catch and consistency are\n")
        f.write("               quality checks, excluded from every\n")
        f.write("               analysis.\n")
        f.write("  instruction  the question to show. Catch trials\n")
        f.write("               carry a different one.\n")
        f.write("  left_path    image for the left position\n")
        f.write("  right_path   image for the right position\n")
        f.write("  a_is_left    whether the left image is image A.\n")
        f.write("               The analysis needs it to map a side\n")
        f.write("               to a choice, because d_sat and d_fog\n")
        f.write("               are always A minus B.\n")
        f.write("  pred_sat     THE REGISTERED MODEL'S COLOUR\n")
        f.write("  pred_fog     AND FOG PREDICTORS. See below.\n")
        f.write("  d_sat,d_fog  raw signed differences, in feature\n")
        f.write("               units, always A minus B\n")
        f.write("  z_sat,z_fog  the same differences normalised, as\n")
        f.write("               step 4 computed them. Reported for\n")
        f.write("               reference; the model fits pred_sat\n")
        f.write("               and pred_fog.\n")
        f.write("  pred_sat_range, pred_fog_range\n")
        f.write("               the predictors divided by each\n")
        f.write("               scene's own feature range instead\n")
        f.write("  mag_bin      magnitude bin, for the\n")
        f.write("               dose-response analysis\n")
        f.write("  catch_mode   catch trials only: which kind of\n")
        f.write("               catch trial this list expects, either\n")
        f.write("               'no_images' (an arrow key named by an\n")
        f.write("               instruction, no pictures shown) or\n")
        f.write("               'images' (the pair shown, a side to\n")
        f.write("               click). THE EXPERIMENT MUST CHECK\n")
        f.write("               THIS AGAINST ITS OWN SETTING AND STOP\n")
        f.write("               IF THEY DISAGREE. There is no\n")
        f.write("               correct_response column: the expected\n")
        f.write("               response lives in the experiment, with\n")
        f.write("               the instruction text.\n")
        f.write("  repeat_of    consistency trials only: the trial\n")
        f.write("               this one repeats\n\n")
        f.write("\nTHE PREDICTORS\n")
        f.write("  pred_sat and pred_fog are what the registered\n")
        f.write("  model fits. Two things about them matter:\n\n")
        f.write("  UNITS. Both are in CORPUS-SD UNITS -- the raw\n")
        f.write(f"  difference over that feature's corpus SD (sat_mean\n")
        f.write(f"  {SD_OF['d_sat']}, contrast_local {SD_OF['d_fog']}, "
                "from the 20,000-image\n")
        f.write("  stimulus-path reference). A coefficient is then\n")
        f.write("  'per SD of the feature', and the two are on one\n")
        f.write("  scale and directly comparable.\n\n")
        f.write("  SIGN. POSITIVE MEANS IMAGE A IS THE PREDICTED\n")
        f.write("  CHOICE -- A is the less colourful image\n")
        f.write("  (pred_sat) or the foggier one (pred_fog). So a\n")
        f.write("  POSITIVE coefficient means the participant chose\n")
        f.write("  as predicted. Note this is NOT simply d/SD:\n")
        f.write("  pred_sat is the NEGATED d_sat, because d_sat is\n")
        f.write("  positive when A is MORE colourful. pred_fog is\n")
        f.write("  d_fog unchanged, because step 4 already applied\n")
        f.write("  the sign there. Negating both would flip the fog\n")
        f.write("  coefficient and nothing would announce it.\n\n")
        f.write("RECORDING RESPONSES\n")
        f.write("  Record which SIDE was clicked (left/right) and the\n")
        f.write("  reaction time. Combine with a_is_left to recover\n")
        f.write("  which image was chosen.\n")

    print(f"  package written to {pkg_dir}")
    report.append(("package images", str(len(used))))
    report.append(("package size", f"{mb:.1f} MB"))
    return out


# ============================================================
# One set
# ============================================================

def restore_pool_order(pairs, ref_csv):
    """Put the pool back into the row order an earlier run used.

    The sampler shuffles the pool, so its order helps decide which
    trials are drawn. Restoring the old order is the only way left to
    test whether the SAMPLER changed, now that the order itself is no
    longer reproducible -- see REPRODUCE_POOL_ORDER.

    Pairs the reference does not contain go last, in canonical order,
    so a mismatch degrades gracefully instead of dropping rows.
    """
    if not os.path.isfile(ref_csv):
        print(f"  NOTE: {ref_csv} not found; pool order not restored, "
              "so this\n        version will not match the shipped "
              "list.")
        return pairs
    ref = pd.read_csv(ref_csv, usecols=["scene_id", "file_a", "file_b"])
    rank = {(a, b, c): i for i, (a, b, c) in enumerate(
        zip(ref.scene_id, ref.file_a, ref.file_b))}
    key = [rank.get((a, b, c), 1 << 30) for a, b, c in
           zip(pairs.scene_id, pairs.file_a, pairs.file_b)]
    out = pairs.assign(_rank=key).sort_values(
        ["_rank", "scene_id", "file_a", "file_b"],
        kind="stable").drop(columns=["_rank"]).reset_index(drop=True)
    n_un = sum(1 for k in key if k == 1 << 30)
    print(f"  pool order restored from {ref_csv}")
    print(f"    {len(pairs) - n_un:,} of {len(pairs):,} pairs matched "
          "the reference order"
          + (f"; {n_un:,} placed last" if n_un else ""))
    return out


_POOL_CACHE = {}


def load_pool(variant, pairs_csv, restore_order):
    """Read a pair table once and reuse it.

    The seed sweep builds every version at N_SEEDS seeds, and reading
    the same 15,000-row table 20 times over is pure cost. Nothing
    writes into the frame -- the cut, the binning and the draw all copy
    -- so one read can serve every build.
    """
    key = (variant, bool(restore_order))
    if key not in _POOL_CACHE:
        pairs = pd.read_csv(pairs_csv)
        if restore_order:
            pairs = restore_pool_order(pairs, REPRODUCE_POOL_ORDER)
        _POOL_CACHE[key] = pairs
    return _POOL_CACHE[key]


def build_one(label, variant, colour_cut, bin_mode, streams_mode, root,
              seed=None, full=True):
    """Build one trial list.

    `full` distinguishes the SHIPPING draw from a replicate. The
    shipping draw (seed RANDOM_SEED) gets the figure, the report and
    the package and lands in <root>/<label>/. A replicate gets only its
    trial_list.csv, under <root>/seeds/<label>/seed_<n>/, which is all
    step 6 needs to measure the spread between draws.
    """
    seed = RANDOM_SEED if seed is None else int(seed)
    print("\n" + "#" * 60)
    print(f"# {label}   [{variant}, cut {colour_cut}, {bin_mode} bins, "
          f"{streams_mode} streams, seed {seed}]")
    print("#" * 60)
    if variant not in VARIANTS:
        raise KeyError(
            f"VERSIONS names variant {variant!r}, which is not in "
            f"VARIANTS ({sorted(VARIANTS)}).")
    pairs_csv, stimuli_src = VARIANTS[variant]
    if not os.path.isfile(pairs_csv):
        raise FileNotFoundError(
            f"Pair table not found: {pairs_csv}\nRun step 4 "
            "(analyse_pairs_v4_corrected.py) first. The 'legacy' "
            "variant needs a run with\nBALANCE_NORM = 'scene_range'.")
    restore = bool(label == REPRODUCE_LABEL and REPRODUCE_POOL_ORDER)
    pairs = load_pool(variant, pairs_csv, restore)
    for c in ("scene_id", "type", "file_a", "file_b", "d_sat", "d_fog"):
        if c not in pairs.columns:
            raise ValueError(f"{pairs_csv} has no {c!r} column.")
    for c in ("pred_sat", "pred_fog"):
        if c not in pairs.columns:
            raise ValueError(
                f"{pairs_csv} has no {c!r} column. Re-run step 4: the "
                "oriented predictors are written there, in corpus-SD "
                "units, so that one name means one thing in both "
                "files.")

    out_dir = (os.path.join(root, label) if full
               else os.path.join(root, "seeds", label, f"seed_{seed}"))
    os.makedirs(out_dir, exist_ok=True)

    # ONE GENERATOR PER DRAW, seeded from (seed, draw name) -- so that
    # changing the colour pool cannot move the fog, together or
    # conflict draws. See the header, and Streams. The reproduction
    # control passes "single" instead, because the shipped list it has
    # to match was built off one stream.
    streams = Streams(seed, streams_mode)
    report = []

    main, prac, catch = build_main_trials(
        pairs, streams, report, colour_cut, bin_mode,
        preserve_order=restore)
    session = assemble(main, prac, catch, streams, report)
    r, vif = check_session(session, label, report)
    session.to_csv(os.path.join(out_dir, "trial_list.csv"), index=False)

    if not full:
        # A replicate draw: the list itself is the whole output.
        return {"label": label, "variant": variant,
                "colour_cut": colour_cut, "bin_mode": bin_mode,
                "streams": streams_mode, "seed": seed, "full": False,
                "session": session, "packaged": None, "report": report,
                "r": r, "vif": vif, "out_dir": out_dir, "path": None}

    make_figure(session, label, out_dir)

    packaged = None
    if PACKAGE:
        packaged = package(session, label, stimuli_src, out_dir, report)

    path = os.path.join(out_dir, "trials_report.txt")
    with open(path, "w") as f:
        f.write(f"TRIAL LIST REPORT  (v4 axes, step 5) -- {label}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Pairs from  : {pairs_csv}\n")
        f.write(f"Stimuli from: {stimuli_src}\n")
        f.write(f"Variant     : {variant}\n")
        f.write(f"Colour cut  : "
                + (f"|d_sat| >= {colour_cut} sat_mean units "
                   f"({colour_cut / SD_OF['d_sat']:.3f} SD)"
                   if colour_cut else "none") + "\n")
        f.write(f"Bins        : {bin_mode}"
                + ("" if bin_mode == "quantile"
                   else "  " + "; ".join(
                       f"{k} {len(v) + 1}" for k, v in
                       BIN_EDGES_SD.items())) + "\n")
        f.write(f"Counts      : {N_SAT} sat_only, {N_FOG} fog_only, "
                f"{N_MIXED} mixed "
                f"({MIXED_CONFLICT_FRAC:.0%} conflict)\n")
        f.write(f"Extras      : {N_CONSISTENCY} consistency, "
                f"{N_CATCH} catch, {N_PRACTICE} practice\n")
        f.write(f"Catch       : mode {CATCH_MODE!r}; exclude a "
                "participant failing 3 or more\n")
        f.write(f"Conflict    : "
                f"{'balanced subset only' if BALANCED_CONFLICT_ONLY else 'all'}\n")
        f.write(f"Scene cap   : {MAX_PAIRS_PER_SCENE_PER_TYPE} per "
                "type\n")
        f.write(f"Streams     : {streams.describe()}\n")
        f.write(f"Seed        : {seed}"
                + (f"  (+ {N_SEEDS - 1} replicate draw(s) under "
                   "seeds/, for step 6)"
                   if N_SEEDS > 1 and label not in SEED_SWEEP_SKIP
                   else "") + "\n\n")
        for k, v in report:
            f.write(f"  {k:36s} {v}\n")
        f.write("\nTRIALS BY TYPE\n")
        f.write(session.trial_type.value_counts().to_string() + "\n")
        f.write("\nMAGNITUDE BINS (analysed trials)\n")
        an = session[session.analysed]
        f.write(pd.crosstab(an.trial_type, an.mag_bin).to_string())
        f.write("\n\n  Bins are ")
        if bin_mode == "fixed":
            f.write("FIXED value ranges in corpus-SD units,\n")
            f.write("  shared by every version, so bin 2 means the "
                    "same\n  difference everywhere:\n")
            for k, v in BIN_EDGES_SD.items():
                f.write(f"    {k:9s} edges {v}\n")
            f.write("  A bin holding fewer than "
                    f"{MIN_PAIRS_PER_BIN} pairs is treated as\n"
                    "  empty and its trials are spread evenly over "
                    "the bins that\n  remain "
                    f"(EMPTY_BIN_RULE = {EMPTY_BIN_RULE!r}). Any bin "
                    "dropped this way is\n  listed above.\n")
        else:
            f.write(f"LEGACY rank-based quantiles, "
                    f"{N_MAGNITUDE_BINS} per type,\n"
                    "  recomputed per pool. Used only by the "
                    "reproduction version;\n  these bins are NOT "
                    "comparable with the fixed ones.\n")
        f.write("\n\nWHAT TO CHECK BY EYE\n")
        f.write("  r(d_sat, d_fog) should be small: it is what decides\n")
        f.write("  whether the two weights can be told apart. Above\n")
        f.write("  about 0.6 they cannot.\n")
    return {"label": label, "variant": variant,
            "colour_cut": colour_cut, "bin_mode": bin_mode,
            "streams": streams_mode, "seed": seed, "full": True,
            "session": session, "packaged": packaged,
            "report": report, "r": r, "vif": vif, "out_dir": out_dir,
            "path": path}


# ============================================================
# Reproduction, and the versions side by side
# ============================================================

def write_overview(results, comp, repro, ok_repro, streams_tab,
                   ok_streams, seed_sum, root, t0):
    """One report across all versions, beside the per-version ones."""
    path = os.path.join(root, "trials_overview.txt")
    with open(path, "w") as f:
        def W(s=""):
            f.write(s + "\n")

        W("TRIAL LISTS, ALL VERSIONS  (v4 axes, step 5)")
        W("=" * 70)
        W(f"Counts   : {N_SAT} sat_only, {N_FOG} fog_only, "
          f"{N_MIXED} mixed ({MIXED_CONFLICT_FRAC:.0%} conflict)")
        W(f"Extras   : {N_CONSISTENCY} consistency, {N_CATCH} catch, "
          f"{N_PRACTICE} practice")
        W(f"Catch    : mode {CATCH_MODE!r}; exclusion at 3 or more "
          "failures of " + str(N_CATCH))
        W(f"Seed     : {RANDOM_SEED} for the shipping list, "
          f"{N_SEEDS} draws in all")
        W(f"Streams  : one generator per draw, seeded from (seed, draw "
          "name)")
        W(f"Elapsed  : {time.time() - t0:.1f}s")
        W()
        W("WHAT EACH VERSION VARIES")
        W("-" * 70)
        W("  Everything except the variant, the colour cut, the bin")
        W("  mode and the stream setting is held identical, so a")
        W("  difference between two versions is the thing being varied")
        W("  and nothing else. Only the control varies the streams.")
        W()
        W("    current        original levels, no cut   (the baseline)")
        W("    extra_levels   vivid levels,    no cut")
        W("    larger_diffs   original levels, cut")
        W("    both           vivid levels,    cut")
        W("    sens_cut_*     the cut varied, vivid levels")
        W("    reproduction   a control, not a candidate")
        W()
        W("0a. STREAM SEPARATION  (the check on the comparison itself)")
        W("-" * 70)
        W("  The colour cut filters `sat_only` and nothing else, so two")
        W("  versions built from the same ladder should have the SAME")
        W("  fog_only, together and conflict trials.")
        W()
        W("  Off a single random stream they did not. The cut changes")
        W("  how many usable magnitude bins the colour draw has, which")
        W("  changes how many numbers it consumes, which moves the")
        W("  stream for every draw after it. `current` against")
        W("  `larger_diffs` -- same ladder, the cut the only declared")
        W("  difference -- shared 1 of 60 fog_only trials, 1 of 30")
        W("  together and 0 of 30 conflict.")
        W()
        W("  Two things were wrong with that. The fog condition could")
        W("  not be read as a control, because its detection moved")
        W("  between versions for no design reason. And colour")
        W("  information leaked in from types the cut never touches:")
        W("  `both` gained 16.4 units of summed squared colour")
        W("  predictor from its `together` trials alone, about a sixth")
        W("  of its total gain over `current`.")
        W()
        W("  Each draw now has its own generator, seeded from (seed,")
        W("  draw name). What that buys is checked here and nowhere")
        W("  else, so this verdict matters:")
        W()
        if streams_tab is None:
            W("  Skipped (fewer than two per-type versions).")
        else:
            jt = streams_tab[streams_tab.judged]
            W(jt[["version_a", "version_b", "type", "trials_a",
                  "shared", "verdict"]].to_string(index=False))
            W()
            W(f"  RESULT: {'PASS' if ok_streams else 'FAIL'}")
            W()
            W("  NOT judged, and why:")
            W("    sat_only            the cut is meant to change it.")
            W("    different ladders   the vivid levels add images,")
            W("                        hence new pairs of EVERY type.")
            W("                        A fog pair at the p75 colour")
            W("                        level did not exist on the")
            W("                        original ladder, so `current`")
            W("                        and `extra_levels` draw from")
            W("                        genuinely different fog pools.")
            W("                        That is the design, not a leak.")
            W("    practice,           drawn after the analysed types")
            W("    catch,              and coupled to them through the")
            W("    consistency         pairs already used, or through")
            W("                        which trials exist to repeat.")
            W("                        None of the three is analysed.")
            W("    reproduction        on the legacy single stream by")
            W("                        definition.")
            W()
            W("  Two rows in the comparison table read on this:")
            W("    colour info outside sat_only   equal within a")
            W("                                   ladder once the")
            W("                                   streams are separate")
            W("                                   -- it was not before")
            W("    fog info outside sat_only      the clean fog")
            W("                                   control, exactly")
            W("                                   equal within a ladder")
            W()
            W("  `fog info (SD^2)`, the total, is NOT exactly equal,")
            W("  and that is not a leak. A colour-only pair sits in one")
            W("  fog row, but changing chroma moves contrast_local a")
            W("  little, so every colour trial carries a residual fog")
            W("  difference -- 0.005 of 67 on the extended ladder. The")
            W("  cut changes which colour trials are drawn, so the")
            W("  total moves in the fourth significant figure while the")
            W("  fog, together and conflict trials are identical.")
            W()
            W("  Full table, every version pair and type:")
            W("  stream_check.csv")
        W()
        W("0b. TWENTY DRAWS PER VERSION")
        W("-" * 70)
        W("  A version can look better because it is a better design")
        W("  or because it drew well: 180 trials out of a pool of")
        W("  thousands is one sample. Each version is therefore built")
        W(f"  at {N_SEEDS} seeds. The first, {RANDOM_SEED}, is the")
        W("  shipping list -- it gets the figure, report and package.")
        W("  The rest are replicates under seeds/, read by step 6,")
        W("  which reports each version's mean detection with the")
        W("  spread of its own draws beside it.")
        W()
        W("  THE SHIPPING SEED IS FIXED IN ADVANCE and is not chosen")
        W("  on any result. Picking the seed that simulates best would")
        W("  be choosing the draw that flatters the design.")
        W()
        W("  Information per draw, mean and between-seed SD. The SD is")
        W("  the yardstick: a difference between two versions smaller")
        W("  than this is draw luck.")
        W()
        if seed_sum is None or not len(seed_sum):
            W("  Skipped (N_SEEDS = 1).")
        else:
            W(seed_sum.to_string())
        W()
        W("0. REPRODUCTION OF THE SHIPPED TRIAL LIST")
        W("-" * 70)
        W("  The four design versions deliberately do NOT reproduce")
        W("  it: the balance normalisation changed in step 4, so a")
        W("  different set of conflict pairs is eligible. The control")
        W("  version isolates that -- it reads the legacy pair table")
        W("  and uses the legacy rank bins, so if it matches, the")
        W("  conflict trials that differ elsewhere differ for that")
        W("  reason and the sampler has not drifted.")
        W()
        W("  The verdict is on the FOUR ANALYSED TYPES. practice,")
        W("  catch and consistency are expected to differ: practice")
        W("  is now drawn from a mix of types rather than `together`")
        W("  alone, and that shifts the random stream for the draws")
        W("  that follow it. None of the three is analysed.")
        W()
        W("  The control also has the earlier run's POOL ORDER")
        W("  restored. The pair set was already identical, but the")
        W("  shipped list was drawn from a table in a different row")
        W("  order, and that order was never reproducible -- step 3")
        W("  measured scenes with imap_unordered, so its manifest")
        W("  followed worker completion order. Steps 3, 4 and 5 each")
        W("  impose a canonical order now, and two runs of step 5")
        W("  give byte-identical trial lists; but the shipped list")
        W("  itself can only be matched by feeding the old order")
        W("  back, which is what REPRODUCE_POOL_ORDER does.")
        W()
        if repro is None:
            W("  Skipped (no reference list).")
        else:
            W(pd.DataFrame(repro).T.to_string())
            W()
            W(f"  RESULT: {'PASS' if ok_repro else 'FAIL'}")
        W()
        W("1. THE VERSIONS SIDE BY SIDE")
        W("-" * 70)
        W(comp.to_string())
        W()
        W("  `sat_only bins used` is the row to watch. A cut empties")
        W("  the low magnitude bins, and if it empties all but one")
        W("  the colour dose-response curve has nothing left to rest")
        W("  on -- which is the cost the cut is traded against.")
        W()
        W("2. WHAT STEP 6 DECIDES")
        W("-" * 70)
        W("  The cut is not chosen here. Step 6 simulates each")
        W("  version and measures how much better a colour-following")
        W("  participant is detected, against how much of the")
        W("  dose-response range was given up to get it. The")
        W("  sensitivity versions exist so that trade can be read off")
        W("  a curve rather than guessed from one value.")
        W()
        W("  Per-version detail is in each version's own")
        W("  trials_report.txt; trial counts per bin and type for")
        W("  every version are in bins_by_version.csv; the per-seed")
        W("  draws are in seeds_by_version.csv and under seeds/.")
    return path


def check_reproduction(results, root):
    """The control version must reproduce the shipped trial list.

    The four design versions deliberately do NOT: changing the balance
    normalisation changes which conflict pairs are eligible, so their
    conflict trials differ by design. This version isolates that. It
    reads the legacy pair table (scene-range balance) and uses the
    legacy rank bins, so if it matches trial for trial, the difference
    in the others is the normalisation and nothing else has drifted in
    the sampler.

    Compared on the set of (scene_id, file_a, file_b) per type, not on
    trial order: order depends on the repeat-spacing search, which is
    not what this is testing.
    """
    print("\n" + "=" * 60)
    print("REPRODUCTION OF THE SHIPPED TRIAL LIST")
    print("=" * 60)
    if not REPRODUCE_FROM:
        print("  REPRODUCE_FROM is None; skipped.")
        return None, True
    if not os.path.isfile(REPRODUCE_FROM):
        print(f"  {REPRODUCE_FROM} not found; skipped.")
        return None, True
    hit = [r for r in results if r["label"] == REPRODUCE_LABEL]
    if not hit:
        print(f"  no version labelled {REPRODUCE_LABEL!r}; skipped.")
        return None, True

    old = pd.read_csv(REPRODUCE_FROM)
    new = hit[0]["session"]
    print(f"  reference: {REPRODUCE_FROM}")
    print(f"    {len(old)} trials   vs   {len(new)} in "
          f"{REPRODUCE_LABEL!r}")

    need = {"trial_type", "file_a", "file_b", "scene_id"}
    if not need <= set(old.columns):
        print(f"  the reference lacks {sorted(need - set(old.columns))}"
              "; comparing counts only.")
        same = len(old) == len(new)
        print(f"  RESULT: {'PASS' if same else 'FAIL'}")
        return None, same

    # THE VERDICT IS ON THE FOUR ANALYSED TYPES. practice, catch and
    # consistency are expected to differ and are reported, not judged:
    # practice is drawn differently on purpose (a mix of types rather
    # than `together` alone), and that change moves the random stream,
    # which moves catch and the consistency repeats drawn after it.
    # None of the three is analysed.
    rows, ok = {}, True
    for t in sorted(set(old.trial_type) | set(new.trial_type)):
        a = set(keys_of(old[old.trial_type == t]))
        b = set(keys_of(new[new.trial_type == t]))
        same = a == b
        judged = t in REAL_TYPES
        rows[t] = {"reference": len(a), "this run": len(b),
                   "shared": len(a & b),
                   "verdict": ("same" if same
                               else "differs" if not judged
                               else "DIFFERS"),
                   "counts?": "yes" if judged else "not analysed"}
        if judged and not same:
            ok = False
    tab = pd.DataFrame(rows).T
    print()
    print(tab.to_string())

    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}  "
          "(on the four analysed types)")
    if ok:
        print("  All four analysed types reproduce the shipped list "
              "exactly, pair for\n  pair. The sampler is unchanged, "
              "so the conflict trials that differ in\n  the other "
              "versions differ because of the balance normalisation "
              "and for\n  no other reason.")
        extra = [t for t, r in rows.items()
                 if t not in REAL_TYPES and r["verdict"] != "same"]
        if extra:
            print(f"  {extra} differ as expected: practice is drawn "
                  "from a mix of types\n  now rather than `together` "
                  "alone, and that shifts the random stream for\n  "
                  "the catch and consistency draws that follow. None "
                  "of the three is analysed.")
    else:
        diff = [t for t in REAL_TYPES
                if rows.get(t, {}).get("verdict") == "DIFFERS"]
        print(f"  {diff} differ, and should not. This version reads "
              "the LEGACY pair table,\n  uses the legacy rank bins, "
              "and has the earlier run's pool order\n  restored, so "
              "a difference here is the SAMPLER. Check that\n  "
              "data/pair_analysis_v4_ext_legacy/ was built with "
              "BALANCE_NORM =\n  'scene_range', that "
              "REPRODUCE_POOL_ORDER points at the earlier "
              f"all_pairs.csv,\n  and that the seed is "
              f"{RANDOM_SEED}.")
    pd.DataFrame(rows).T.to_csv(
        os.path.join(root, "reproduction.csv"))
    return rows, ok


def check_streams(results, root):
    """Versions sharing a ladder must share every type the cut leaves
    alone.

    THIS IS THE CHECK ON THE FIX. The colour cut filters `sat_only` and
    nothing else, so `fog_only`, `together` and `conflict` should be
    the same trials in two versions built from the same step-4 variant.
    Off a single random stream they were not: the cut changed how many
    numbers the colour draw consumed, and every draw after it moved.
    `current` against `larger_diffs` shared 1 of 60 fog_only trials, 1
    of 30 together and 0 of 30 conflict.

    That made the fog condition useless as a control -- its detection
    differed between versions for no design reason -- and leaked colour
    information in from types the cut does not touch.

    WHAT IS JUDGED: pairs of versions with the SAME variant, both on
    per-type streams. Those must match exactly on the three untouched
    types.

    WHAT IS REPORTED BUT NOT JUDGED:
      - `sat_only`, which the cut is supposed to change.
      - pairs with DIFFERENT variants. The vivid levels add images,
        hence new pairs of every type, so a different fog draw there is
        the design and not a leak.
      - the reproduction control, which is on the legacy single stream
        by definition.

    Compared on sets of (scene_id, file_a, file_b) per type, not on
    presentation order: the order depends on the repeat-spacing search
    and on which trials the colour pool supplied, so it differs even
    when the trials are identical.
    """
    print("\n" + "=" * 60)
    print("STREAM SEPARATION")
    print("=" * 60)
    cand = [r for r in results
            if r["label"] != REPRODUCE_LABEL and r["streams"] == "per_type"]
    if len(cand) < 2:
        print("  fewer than two per-type versions; nothing to compare.")
        return None, True

    untouched = ["fog_only", "together", "conflict"]
    rows, ok = [], True
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            same_variant = a["variant"] == b["variant"]
            sa, sb = a["session"], b["session"]
            for t in REAL_TYPES:
                ka = set(keys_of(sa[sa.trial_type == t]))
                kb = set(keys_of(sb[sb.trial_type == t]))
                judged = same_variant and t in untouched
                identical = ka == kb
                rows.append({
                    "version_a": a["label"], "version_b": b["label"],
                    "same_ladder": same_variant, "type": t,
                    "trials_a": len(ka), "trials_b": len(kb),
                    "shared": len(ka & kb),
                    "judged": judged,
                    "verdict": ("same" if identical
                                else "DIFFERS" if judged
                                else "differs (expected)"),
                })
                if judged and not identical:
                    ok = False
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(root, "stream_check.csv"), index=False)

    jt = tab[tab.judged]
    print(f"  {len(jt)} comparisons judged "
          f"({jt.version_a.nunique()} ladder-sharing version pairs x "
          f"{len(untouched)} untouched types)")
    if len(jt):
        print(jt[["version_a", "version_b", "type", "shared",
                  "trials_a", "verdict"]].to_string(index=False))
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("  Every pair of versions sharing a ladder has identical "
              "fog_only, together\n  and conflict trials, so the "
              "colour cut is the only difference between\n  them and "
              "the fog condition is a usable control.")
    else:
        bad = jt[jt.verdict == "DIFFERS"]
        print("  These should be identical and are not, so a draw is "
              "still sharing a\n  stream with another. Check that "
              "every call into Streams uses its own\n  name from "
              "STREAM_NAMES:")
        print(bad[["version_a", "version_b", "type",
                   "shared", "trials_a"]].to_string(index=False))

    cross = tab[(~tab.same_ladder) & (tab.type != "sat_only")]
    if len(cross):
        print("\n  Across ladders, reported only. The vivid levels add "
              "new pairs of every\n  type, so these differ by design:")
        print(cross.groupby("type")[["shared", "trials_a"]]
              .mean().round(1).to_string())
    return tab, ok


def info_split(session):
    """Colour and fog information in a list, by trial type.

    The sum of the squared oriented predictor, in corpus-SD units --
    the quantity that sets how precisely a weight can be estimated, and
    so the one to compare designs on.

    Split by type because that is what exposed the stream leak: `both`
    gained 16.4 units of COLOUR information from its `together` trials,
    which the colour cut never touches. A version's colour information
    should grow in `sat_only` and nowhere else.
    """
    an = session[session.analysed]
    out = {}
    for name, col in (("sat", "pred_sat"), ("fog", "pred_fog")):
        out[f"info_{name}"] = float((an[col] ** 2).sum())
        for t in REAL_TYPES:
            g = an[an.trial_type == t]
            out[f"info_{name}_{t}"] = float((g[col] ** 2).sum())
    out["info_sat_outside_sat_only"] = (
        out["info_sat"] - out["info_sat_sat_only"])
    # The FOG information outside the colour-only trials. This is the
    # clean control quantity, and it is exactly equal between versions
    # sharing a ladder.
    #
    # The total is not, quite, and the reason is worth knowing before
    # it looks like a leak: a colour-only pair sits in one fog row, but
    # changing chroma moves contrast_local a little, so each colour
    # trial carries a residual fog difference. It is tiny -- 0.005 of
    # 67 on the extended ladder -- but it is not zero, and the colour
    # cut changes which colour trials are drawn, so the TOTAL fog
    # information moves in the fourth significant figure while the
    # fog, together and conflict trials are bit-identical.
    out["info_fog_outside_sat_only"] = (
        out["info_fog"] - out["info_fog_sat_only"])
    out["info_ratio_sat_fog"] = (out["info_sat"]
                                 / max(1e-12, out["info_fog"]))
    return out


def run_seed_sweep(results, root):
    """Build every version at the replicate seeds, and tabulate.

    One draw is one sample: a version can look better because it is a
    better design or because it drew well. The sweep separates those by
    building the same version at N_SEEDS seeds, so step 6 can report
    each version's mean detection with the spread of its own draws
    beside it.

    The replicates are quiet and cheap -- no figure, no package, no
    report -- and their output is one trial_list.csv each.
    """
    seeds = seed_list()
    rows = [dict(version=r["label"], seed=r["seed"], shipping=True,
                 variant=r["variant"],
                 colour_cut=r["colour_cut"] or 0.0,
                 r_sat_fog=r["r"], vif=r["vif"],
                 scenes=int(r["session"].scene_id.nunique()),
                 **info_split(r["session"]))
            for r in results]
    if len(seeds) < 2:
        print("\n  N_SEEDS = 1; no replicate draws.")
        return pd.DataFrame(rows)

    todo = [(lab, var, cut, mode, st, s)
            for lab, var, cut, mode, st in VERSIONS
            if lab not in SEED_SWEEP_SKIP
            for s in seeds[1:]]
    print("\n" + "=" * 60)
    print("REPLICATE DRAWS")
    print("=" * 60)
    print(f"  {len(todo)} builds: "
          f"{len({t[0] for t in todo})} versions x {len(seeds) - 1} "
          "further seeds")
    print(f"  skipped: {', '.join(SEED_SWEEP_SKIP) or 'none'}")
    print(f"  into   : {os.path.join(root, 'seeds')}")
    t0 = time.time()
    for n, (lab, var, cut, mode, st, s) in enumerate(todo, 1):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rep = build_one(lab, var, cut, mode, st, root,
                                seed=s, full=False)
        except Exception:
            # The replicate builds are silent, so hand back the log of
            # the one that failed rather than an exception with no
            # context.
            print(buf.getvalue()[-2000:])
            raise
        rows.append(dict(version=lab, seed=s, shipping=False,
                         variant=var, colour_cut=cut or 0.0,
                         r_sat_fog=rep["r"], vif=rep["vif"],
                         scenes=int(rep["session"].scene_id.nunique()),
                         **info_split(rep["session"])))
        if n % 20 == 0 or n == len(todo):
            print(f"    {n}/{len(todo)}  ({time.time() - t0:.0f}s)")
    tab = pd.DataFrame(rows).sort_values(["version", "seed"])
    tab.to_csv(os.path.join(root, "seeds_by_version.csv"), index=False)
    return tab


def summarise_seeds(seed_tab, root):
    """Mean and between-seed spread per version."""
    cols = ["info_sat", "info_fog", "info_ratio_sat_fog",
            "info_sat_outside_sat_only", "info_fog_outside_sat_only",
            "r_sat_fog"]
    g = seed_tab.groupby("version")
    out = pd.concat([g[cols].mean().round(3).add_suffix("_mean"),
                     g[cols].std(ddof=1).round(3).add_suffix("_sd"),
                     g.size().rename("seeds")], axis=1)
    order = [lab for lab, *_ in VERSIONS if lab in out.index]
    out = out.loc[order]
    out.to_csv(os.path.join(root, "seeds_summary.csv"))
    return out


def build_comparison(results, root):
    """One row per quantity, one column per version."""
    rows = {}

    def put(k, label, v):
        rows.setdefault(k, {})[label] = v

    for r in results:
        L = r["label"]
        s = r["session"]
        an = s[s.analysed]
        put("variant", L, r["variant"])
        put("colour cut", L, r["colour_cut"] or "none")
        put("bins", L, r["bin_mode"])
        put("streams", L, r["streams"])
        put("trials presented", L, len(s))
        put("trials analysed", L, len(an))
        for t in REAL_TYPES:
            put(f"{t} trials", L, int((an.trial_type == t).sum()))
        put("scenes used", L, s.scene_id.nunique())
        put("r(d_sat, d_fog)", L, round(r["r"], 3))
        put("VIF", L, round(r["vif"], 2))
        so = an[an.trial_type == "sat_only"]
        if len(so):
            put("sat_only median |d_sat|", L,
                round(float(so.d_sat.abs().median()), 1))
            put("sat_only min |d_sat|", L,
                round(float(so.d_sat.abs().min()), 1))
            put("sat_only max |d_sat|", L,
                round(float(so.d_sat.abs().max()), 1))
            put("sat_only bins used", L,
                so.mag_bin.nunique())
        # The information in the list: the sum of the squared oriented
        # predictor, in corpus-SD units. This is what step 6 turns into
        # detection, so it is the quantity the versions differ on.
        #
        # `colour info outside sat_only` is the leak monitor. The cut
        # touches sat_only alone, so that row should be the same in two
        # versions sharing a ladder. It was not, before the streams
        # were separated -- see check_streams.
        inf = info_split(s)
        put("colour info (SD^2)", L, round(inf["info_sat"], 1))
        put("fog info (SD^2)", L, round(inf["info_fog"], 1))
        put("colour/fog info", L, round(inf["info_ratio_sat_fog"], 3))
        put("colour info in sat_only", L,
            round(inf["info_sat_sat_only"], 1))
        put("colour info outside sat_only", L,
            round(inf["info_sat_outside_sat_only"], 1))
        put("fog info outside sat_only", L,
            round(inf["info_fog_outside_sat_only"], 3))
    comp = pd.DataFrame(rows).T
    comp.index.name = "quantity"
    comp.to_csv(os.path.join(root, "comparison.csv"))
    return comp


def write_bin_table(results, root):
    """Trial counts per bin and type, for every version."""
    frames = []
    for r in results:
        s = r["session"]
        an = s[s.analysed]
        t = pd.crosstab(an.trial_type, an.mag_bin)
        t.insert(0, "version", r["label"])
        frames.append(t.reset_index())
    out = pd.concat(frames, ignore_index=True).fillna(0)
    for c in out.columns:
        if c not in ("version", "trial_type"):
            out[c] = out[c].astype(int)
    out.to_csv(os.path.join(root, "bins_by_version.csv"), index=False)
    return out


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("BUILD TRIALS  (v4, step 5)")
    print("=" * 60)
    check_settings()
    root = resolve_output_root()
    os.makedirs(root, exist_ok=True)
    n_an = N_SAT + N_FOG + N_MIXED
    print(f"  versions: {', '.join(l for l, *_ in VERSIONS)}")
    print(f"  analysed: {n_an} trials "
          f"({N_SAT}/{N_FOG}/{N_MIXED})")
    print(f"  plus    : {N_CONSISTENCY} consistency, {N_CATCH} catch, "
          f"{N_PRACTICE} practice = "
          f"{n_an + N_CONSISTENCY + N_CATCH + N_PRACTICE} presented")
    print(f"  output  : {root}")
    print(f"  seeds   : {N_SEEDS} draw(s) per version, shipping "
          f"seed {RANDOM_SEED}")
    print("  everything but the variant, the colour cut and the bin "
          "mode is held\n  identical across versions, so a difference "
          "between two of them is the\n  thing being varied and "
          "nothing else. Each draw has its own random\n  stream, "
          "seeded from (seed, draw name), so the colour cut cannot "
          "move\n  the fog, together or conflict trials -- which is "
          "what check_streams\n  verifies.")

    results = [build_one(label, variant, cut, mode, st, root)
               for label, variant, cut, mode, st in VERSIONS]

    repro, ok_repro = check_reproduction(results, root)
    streams_tab, ok_streams = check_streams(results, root)
    comp = build_comparison(results, root)
    bins = write_bin_table(results, root)
    seed_tab = run_seed_sweep(results, root)
    seed_sum = (summarise_seeds(seed_tab, root)
                if len(seed_tab) > len(results) else None)

    print("\n" + "=" * 60)
    print("THE VERSIONS SIDE BY SIDE")
    print("=" * 60)
    print(comp.to_string())
    print("\n" + "=" * 60)
    print("TRIALS PER BIN AND TYPE")
    print("=" * 60)
    print(bins.to_string(index=False))

    if seed_sum is not None:
        print("\n" + "=" * 60)
        print(f"ACROSS {N_SEEDS} DRAWS PER VERSION")
        print("=" * 60)
        print(seed_sum.to_string())
        print("\n  The _sd columns are the yardstick: a difference "
              "between two versions\n  smaller than the spread of one "
              "version's own draws is draw luck.")

    write_overview(results, comp, repro, ok_repro, streams_tab,
                   ok_streams, seed_sum, root, t0)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        s = r["session"]
        print(f"  {r['label']:14s} {len(s):3d} trials "
              f"({int(s.analysed.sum())} analysed)   "
              f"r(d_sat,d_fog) {r['r']:+.3f}   -> {r['out_dir']}")
    print()
    print(f"  {'reproduction':14s} "
          + ("skipped" if repro is None
             else "PASS" if ok_repro else "FAIL"))
    print(f"  {'streams':14s} "
          + ("skipped" if streams_tab is None
             else "PASS" if ok_streams else "FAIL"))
    print(f"\n  Elapsed: {time.time() - t0:.1f}s")
    print("\n  NEXT: step 6 (choose_colour_condition_v4) measures what "
          "each cut buys\n        in detection against the range it "
          f"costs, over all {N_SEEDS} draws, and\n        the cut is "
          "chosen there.")
    if streams_tab is not None and not ok_streams:
        raise RuntimeError(
            "STREAM SEPARATION FAILED. Two versions sharing a ladder "
            "have different\nfog_only, together or conflict trials, so "
            "a comparison between them would\nnot isolate the colour "
            "cut. Every report above was written; see\n"
            + os.path.join(root, "stream_check.csv")
            + " for which pairs differ. Do not\nchoose a version from "
            "these numbers until it passes.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
