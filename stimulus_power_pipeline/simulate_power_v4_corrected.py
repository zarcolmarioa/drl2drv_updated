"""
simulate_power_v4.py  (stimulus pipeline, v4 axes, step 8)

Simulates participants against the FROZEN trial list and reports what
the experiment could and could not detect, before anyone is recruited.

Where this sits
---------------
  step 4  analyse_pairs_v4        every usable pair
  step 5  build_trials_v4         the candidate trial lists
  step 6  choose_colour_condition which list carries most colour
  step 6b check_robustness        does that survive lapses
  step 7a freeze_trial_list       the chosen list, sealed
  step 8  this one                is the sealed list powered?
  step 9  check_session_v4        the same analysis, on real data

ONE TRIAL LIST, AND IT IS SEALED
--------------------------------
Earlier versions of this script took a LIST of trial lists and wrote a
comparison table, because the choice between two fog definitions was
still open. It is not: step 6 chose the axes, step 6b confirmed them
under lapses, and step 7a froze `both` at seed 42. So this script
analyses ONE list, and it verifies that list against its own
checksums before it reads a single row -- a power figure computed from
a package that has drifted is worse than no power figure, because it
looks like evidence.

The legacy list stays as a SECOND entry, used only by the reproduction
check at the end. It is not a candidate and does not appear in any
result table.

THE PREDICTORS ARE THE REGISTERED ONES
--------------------------------------
    P(choose image A) = 1 / (1 + exp(-(b_sat*pred_sat + b_fog*pred_fog)))

`pred_sat` and `pred_fog` are the step-5 columns, in CORPUS-SD units:
the difference between the two images divided by the standard
deviation of that feature across the 20,000-image corpus, oriented so
that a positive weight always means "prefers the predicted image" --
the duller one for colour, the foggier one for fog.

Earlier versions fitted `z_sat` and `z_fog`, which are MAGNITUDES
normalised by each scene's own range. That made a weight mean
something different from scene to scene and from the registered model,
and the sign had to be rebuilt from `np.sign(d_sat)` at the point of
use. Corpus-SD units are the units the registered analysis fits, and
Firth's estimator is equivariant to a linear rescaling of the
predictors, so nothing is lost by stating them this way.

Every weight in this script is therefore PER CORPUS SD, and grid-free:
it does not depend on how many rungs the colour ladder has or how the
magnitude bins were drawn, so it stays comparable if those change.
Each one is also reported as a CHOICE PROBABILITY at that feature's
median informative difference, because "0.5 per SD" is not a quantity
anyone has intuitions about and "picks the duller image 63% of the
time on a typical colour trial" is.

THE ESTIMATOR IS FIRTH
----------------------
Firth's bias-reduced logistic regression, the same solver step 6 and
6b use, validated there five ways. The reason is separation: on a list
of mostly large colour differences a sensitive participant can answer
every colour-informative trial as predicted, and then no finite
maximum-likelihood weight exists. The options were to exclude that
participant -- which drops the strongest evidence for the effect and
biases against the hypothesis -- or to report a number the data does
not support. Firth's penalty is always finite and keeps them.

It is finite but NOT a measurement of magnitude, which is why this
script computes the CAP: the largest weight this design can resolve.
See THE CAPS below.

TWO-SIDED PER-PERSON THRESHOLDS
-------------------------------
Earlier versions asked only "is this weight above the positive
floor". That cannot distinguish a person who tracks the feature
BACKWARDS from one who does not track it at all, and the difference
matters: a decision rule that counts predicted-direction responders
has to know how many went the other way, or it cannot state a
false-claim rate.

So each person gets THREE possible verdicts per feature:

    predicted      weight above the upper threshold
    opposite       weight below the lower threshold
    not detected   between them

The UPPER threshold is unchanged from step 6 and 6b -- the 97.5th
percentile of pure guessers, a 2.5% false-alarm rate in the predicted
direction -- so those two steps remain valid and do not need re-running.
The lower threshold is the 2.5th percentile of the SAME guessers, and
it is new. It feeds the decision rule and nothing else.

What gets registered is not either threshold but the RULE's false-claim
rate, which this script computes.

THE LABELS, AND THE UNITS THE CLAIMS ARE STATED IN
--------------------------------------------------
    H1   FOG     (contrast_local)  -- the hazier image
    H2   COLOUR  (sat_mean)        -- the duller image
    C1   the group comparison: DPDR against controls

EVERY EXISTENCE CLAIM IS STATED BEHAVIOURALLY, as a choice
probability on a typical informative trial of that feature. The
per-SD weight is reported beside it as the secondary form.

That is not presentation. A weight per corpus SD has no perceptual
meaning -- a standard deviation of saturation across a 20,000-image
corpus is not something a participant can be more or less sensitive
to -- and indexing by it makes the two features look very differently
powered when they are not:

  median informative difference   colour about 1.0 SD, fog about 0.6 SD
  per-person threshold            colour about 0.29, fog about 0.50

The fog threshold is higher for exactly one reason: the list carries
less fog information. Summed over the 180 analysed trials,
sum pred_sat^2 is about 2.8 times sum pred_fog^2, and a logistic
weight's standard error scales as 1 / sqrt(sum x^2), so the two
thresholds should sit in the ratio sqrt(2.8) = 1.67 -- which is what
they do. Most of that gap is the colour cut, which raised the colour
differences and left the fog ones alone.

AND IT LARGELY CANCELS. Equal behaviour needs the fog weight to be
median_colour / median_fog times the colour weight, and the threshold
ratio is the square root of the ratio of the two sums of squares. The
two are close, so fog's weight-to-threshold ratio comes out near
colour's.

NO FIGURES ARE QUOTED HERE, ON PURPOSE. The ones that used to be --
"about 0.97", "within a few per cent" -- came from a step-5 TEST DRAW
and did not survive the freeze. The frozen list gives a different
ratio, and the per-cell agreement is good in the median but NOT
uniform: a substantial minority of cells differ by more than 0.05,
concentrated where both features sit below their thresholds and
neither claim could be made anyway. Every figure is derived from the
run and printed in the equal-behaviour section, which is the only
place to read them from.

The report carries the power table BOTH WAYS -- indexed by per-SD
weight and indexed by choice probability -- so the difference between
the two framings is visible rather than argued about.

Inputs
------
  FROZEN_LIST   the sealed package from step 7a, verified first
  LEGACY_LIST   the pre-freeze list, for the reproduction check only
  BY_SEED       step 6's per-draw tables, to place seed 42 among the 20

What it simulates
-----------------
A synthetic participant is given two true weights and responds
probabilistically to each trial:

    P(choose image A) = 1 / (1 + exp(-(b_sat * z_sat + b_fog * z_fog)))

where z_sat and z_fog are the standardised differences between the two
images, ORIENTED so that a positive weight always means "prefers the
more manipulated image": the foggier one, and the LESS saturated one.
The weights say what that person's recalled experience tracks: colour,
fog, both, or neither.

Standardised rather than raw differences, so the two weights are on a
comparable scale and "b_sat = 1.0" means the same thing for both
features. This also makes the numbers comparable with v2's, since
dividing by each scene's own range cancels the units.

The three analyses, matching the plan
-------------------------------------
  CONFIRMATORY -- proportion + Bayes factor.
    Each participant gives one number per trial type: the share of
    trials on which they chose the more manipulated image. A one-sample
    t-test against chance, with a JZS Bayes factor (Rouder et al.,
    2009) alongside.

    The Bayes factor matters more than usual here. A p-value cannot
    distinguish "no effect" from "not enough data", and with a small
    sample a non-significant result is the likely outcome. A Bayes
    factor gives evidence FOR the null, which is what makes a null
    result reportable. Kusztor et al. (2025) reached BF01 > 1000 at
    N = 200; at N = 10 the realistic ceiling is 3-6, i.e. "moderate
    evidence against an effect", which is still far better than
    "p = 0.34, we conclude nothing".

  EXPLORATORY -- per-participant logistic regression.
    Two weights per person. Reported with intervals rather than Bayes
    factors: a model-comparison BF would need priors on the weights,
    the between-person spread and more, none with an obvious default.
    An interval says the same thing without assumptions to defend.

  DOSE-RESPONSE.
    The proportion computed separately per magnitude bin. A rising
    curve is much stronger evidence than a single proportion, because
    a response bias or a demand characteristic would give a flat line
    that is still above chance.

A NOTE ON "THE MORE MANIPULATED IMAGE" IN CONFLICT TRIALS
---------------------------------------------------------
For sat_only the manipulated image is the duller one; for fog_only and
together it is the foggier one, and those agree. A CONFLICT trial has
no such image: one picture is foggier, the other is more colourful, and
which counts as "more manipulated" depends on which cue you ask about.

The dose-response section follows v2 and scores conflict trials by FOG
(the foggier image counts as chosen-manipulated). That keeps the curve
comparable with v2's and with the other mixed trials, but it means the
conflict curve reads as "how often did they follow the fog cue", not
"how often did they prefer the manipulation". Read it that way.

Inputs
------
  TRIAL_LISTS    trial_list.csv per set, from step 5

Outputs (per set, under OUTPUT_ROOT/<label>/)
---------------------------------------------
  recovery.csv          recovered weights per scenario
  detection_curve.csv   share of people caught, by their true weight
  crossover_curve.csv   share caught, by the true GAP between their two
                        weights -- the crossover hypothesis, which the
                        per-feature floors cannot answer
  additivity_curve.csv  share caught, by how far the two cues depart
                        from combining additively
  dose_response.csv     expected proportion per magnitude bin
  recovery.png          recovered weights, clouds overlapping or not
  detection_curve.png   the two per-feature detection curves
  contrasts.png         the crossover and additivity curves
  dose_response.png     expected dose-response shape

Shared, under OUTPUT_ROOT/
--------------------------
  power_curve.csv/.png  detection rate vs sample size and effect size
  bayes_factors.csv/.png  BF distribution under effect and under null
  comparison.csv        the sets side by side
  power_report.txt      the whole thing

SELF-CONTAINED and lightweight: reads trial_list.csv and nothing else.
No images, no encoder. The logistic fit is a short Newton solver, so
there is no sklearn or statsmodels dependency.

TWO SCRIPTS THIS ONE SUPERSEDES -- DO NOT RUN THEM
--------------------------------------------------
`simulate_power_seq_v4.py` (1,445 lines) and `plot_power_seq_v4.py`
(745 lines) were the old step 7: N people, the group test, sequential
stopping, and the preregistration figures drawn from its tables. Both
are RETIRED. They are listed here rather than deleted silently,
because their output files may still be sitting in the repository and
someone will otherwise wonder which numbers are current.

They MUST NOT BE RUN. Every one of their inputs and conventions has
since changed:

  predictors   z_sat / z_fog, the SCENE-RANGE magnitudes, whose
               normaliser is each scene's own range -- so a weight of
               1.0 meant a different physical difference in every
               scene. Retired in favour of pred_sat / pred_fog in
               corpus-SD units.
  estimator    ridge-penalised plain logistic (RIDGE = 1e-6). No
               Firth anywhere in either file, so separable
               participants have no finite estimate.
  floors       ONE-SIDED only: the (1 - alpha) percentile, no lower
               threshold, so no opposite-direction classification and
               none of the two-sided machinery the rules now rest on.
  input        data/trials_v4_ec/haze_eq and /haze -- pre-freeze
               lists, and `haze` is not even a candidate any more.
  numbers      its floors print as 0.66 (colour) and 0.56 (fog),
               which are z-unit figures. The registered floors are
               0.286 and 0.488 in corpus-SD units. Run as it stands it
               produces numbers that look like results and are
               comparable with nothing.

Where each piece now lives:

  floors from the trial list, per feature   -> compute_floors (two-sided)
  per-person classification                 -> classify / run_detection
  responder mixture, swept                  -> run_rule_sweep (exact,
                                               not simulated)
  required_detections: the count rule,      -> run_rule_sweep: three
  absolute family, one-sided                   families x three
                                               admissibility standards
  group t-test vs the existence claim       -> run_power + the rule
                                               tables
  responder-count recovery                  -> recover_taste_spread and
                                               the prevalence sections
  one- and two-sided JZS Bayes factor       -> jzs_bayes_factor, with
  with a g-form cross-check                    check_bayes_factor
                                               (the two forms agree to
                                               3e-11)
  lookup_by_n.csv, the headline table       -> rule_decision_table.csv
                                               + rule_summary_table
  sequential monitoring / optional stopping -> NOT PORTED. Deliberate:
                                               the stopping rule is a
                                               single final analysis,
                                               and the report says why.
  plot_power_seq_v4.py's seven figures      -> this script's own
                                               figures, listed below.

Figures this script writes
--------------------------
  <label>/recovery.png            recovered weights against the truth
  <label>/detection_curve.png     per-person detection against weight
  <label>/contrasts.png           crossover and additivity
  <label>/dose_response.png       expected dose-response
  <label>/existence_power.png     the existence claim against N, by
                                  responder share, under the
                                  registered standard
  <label>/c1_power.png            C1 against N, by responder share and
                                  strategy rate
  <label>/equal_behaviour.png     H1 against H2 at equal behaviour
  <label>/gate.png                the attention gate: false alarm and
                                  sensitivity
  power_curve.png                 group-level power
  bayes_factors.png               Bayes factors under a null and an
                                  effect

Usage
-----
    conda activate drl2drv
    python simulate_power_v4.py
"""

import json
import os
import time
import hashlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

# --- Input / output ---
#
# FROZEN_ROOT
#   The sealed package from step 7a. Everything registered is computed
#   from the list inside it.
FROZEN_ROOT = "data/frozen_trial_list_v4"

# VERIFY_CHECKSUMS
#   Re-hash every file in FROZEN_ROOT against its own
#   checksums.sha256 before reading anything, and REFUSE TO RUN on a
#   mismatch.
#
#   This is not ceremony. The power figures and the registered
#   thresholds are properties of a specific list; computed from a
#   package that has drifted they would still look like evidence.
#   Verification costs a few seconds against 349 images.
#
#   False only for a dry run on a package you have not frozen yet.
VERIFY_CHECKSUMS = True

# UNHASHED
#   The files step 7a leaves OUT of checksums.sha256, by construction:
#   it cannot hash the manifest while writing it, and FROZEN.json and
#   freeze_report.txt are written afterwards (FROZEN.json records the
#   manifest's own filename). Step 7a's own verify_checksums applies
#   the same exclusion.
#
#   They are not ignored. All three must be PRESENT, and FROZEN.json
#   is then cross-checked against the manifest's length and against
#   this script's constants -- the seed, the two corpus SDs and the
#   length of the catch block, that last one because every attention
#   gate figure is calibrated on it.
UNHASHED = ("checksums.sha256", "FROZEN.json", "freeze_report.txt")

# LEGACY_LIST
#   The pre-freeze list, read ONLY by the reproduction check at the
#   end. It is not a candidate: it was built from the scene-range
#   balance definition, the legacy rank bins and a single random
#   stream, so it is the configuration the old figures came from. The
#   check runs the whole script's one-directional path against it and
#   requires the old numbers back.
#
#   None skips the check, which is the right setting once the old
#   output folder is gone.
LEGACY_LIST = "data/trials_v4_ec/haze_eq/trial_list.csv"

# LEGACY_OUTPUT
#   The old run's output folder, for the reproduction check to compare
#   against. None means the check reports its own numbers without
#   claiming to reproduce anything.
LEGACY_OUTPUT = "data/power_v4_ec"

# BY_SEED_INFORMATION / BY_SEED_DETECTION
#   Step 6's per-draw tables, one row per draw. Used only to report
#   where seed 42 -- the frozen draw -- sits among the 20, so a reader
#   can see whether the registered list is a typical draw or a lucky
#   one. These live in step 6's live output folder and are NOT inside
#   the sealed package, which carries only the 20-draw summaries; the
#   figures are quoted into this script's report, which is what then
#   survives.
BY_SEED_INFORMATION = "data/choose_colour_v4/information_by_seed.csv"
BY_SEED_DETECTION = "data/choose_colour_v4/detection_by_seed.csv"
FROZEN_SEED = 42

# BY_SEED_GROUP_KEYS / BY_SEED_SKIP
#   Step 6's detection table holds SEVERAL ROWS PER DRAW -- one per
#   simulated sensitivity and participant type. A quantity has to be
#   ranked within its own condition; ranked against all of a version's
#   rows pooled it would be compared with other conditions rather than
#   with other draws, which is the same error as pooling versions.
#
#   BY_SEED_SKIP are bookkeeping columns that are not quantities to
#   rank.
BY_SEED_GROUP_KEYS = ("sensitivity_SD", "participant", "estimator")
BY_SEED_SKIP = ("seed", "draw", "index", "shipping", "list", "trials",
                "floor_gap_reps", "path")

OUTPUT_ROOT = "data/power_v4"

# --- Figures ---
#
# SERIES_COLOURS / SERIES_MARKERS / SERIES_STYLES
#   Okabe-Ito, in a fixed order, assigned to series by identity and
#   never cycled. Validated rather than chosen by eye: worst
#   adjacent-pair separation 11.0 under deuteranopia, 16.4 under
#   normal vision, every slot inside the lightness band.
#
#   MARKERS AND LINE STYLES CARRY THE SAME IDENTITY. Two reasons: the
#   fourth colour's contrast against white is 2.98:1 against a 3:1
#   target, and these figures go into a document that will be printed
#   and photocopied in greyscale. Colour alone would not survive
#   either.
SERIES_COLOURS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
SERIES_MARKERS = ["o", "s", "^", "D"]
SERIES_STYLES = ["-", "--", "-.", ":"]

# FIG_DPI
#   150 is enough for a document at this size and keeps the files
#   small enough to live in the repository.
FIG_DPI = 150

# EQUAL_BEHAVIOUR_TOL
#   How far the two features' claims may differ, as a difference in
#   P(the claim fires), before a cell counts as DIVERGING. 0.05 was
#   inline in four places in the equal-behaviour section and is now
#   named once, because the figure has to use the same number the
#   table counts with.
EQUAL_BEHAVIOUR_TOL = 0.05

# ON_EXISTING
#   "new" writes to OUTPUT_ROOT_2, _3, ... rather than overwriting.
#   "error" stops instead.
ON_EXISTING = "new"

# --- Predictors ---
#
# PRED_SAT / PRED_FOG
#   The registered model's two columns, read from the frozen list, in
#   corpus-SD units and already oriented so a positive weight means
#   "prefers the predicted image".
#
#   NOT z_sat / z_fog, which earlier versions fitted. Those are
#   MAGNITUDES whose normaliser is each scene's own range, so a weight
#   of 1.0 meant a different physical difference in every scene, and
#   the direction had to be rebuilt from np.sign(d_sat) wherever it
#   was used. The start-up check below re-derives these columns from
#   d_sat and d_fog and refuses to run if they disagree.
PRED_SAT = "pred_sat"
PRED_FOG = "pred_fog"

# SD_SAT / SD_FOG
#   The corpus standard deviations the predictors are divided by,
#   measured over the 20,000-image stimulus pool in step 3. Needed
#   only to re-derive the predictors for the start-up check and to
#   state raw-unit equivalents in the report.
SD_SAT = 39.182
SD_FOG = 9.348

# PRED_REBUILD_TOL
#   How far the shipped pred_sat / pred_fog may differ from
#   -d_sat/SD_SAT and +d_fog/SD_FOG before the script stops. This is
#   float noise only, so the tolerance is tight; a real failure means
#   a sign convention or a unit changed upstream and every number
#   below would be wrong in a way no later check would catch.
PRED_REBUILD_TOL = 1e-9

# --- The estimator ---
#
# ESTIMATOR
#   "firth"  Firth's bias-reduced logistic regression, the registered
#            per-person fit. Always finite, so a participant the model
#            predicts perfectly keeps an estimable weight instead of
#            being excluded.
#   "mle"    plain maximum likelihood with a token ridge, which is
#            what the earlier version of this script did. Kept for the
#            reproduction check and for nothing else.
ESTIMATOR = "firth"

# FIRTH_ITER / FIRTH_TOL / FIRTH_MAX_STEP
#   The Newton loop, copied with the solver from step 6. FIRTH_MAX_STEP
#   damps each step: from zero on near-separable data an undamped
#   first step can overshoot far enough to stall. 5.0 is in per-SD
#   units, far above any plausible weight.
FIRTH_ITER = 200
FIRTH_TOL = 1e-10
FIRTH_MAX_STEP = 5.0

# --- The caps ---
#
# THE LARGEST WEIGHT THIS DESIGN CAN RESOLVE. Firth returns a finite
# number for a participant whose answers the model predicts perfectly,
# which is why it is the registered estimator -- but that number is
# not a measurement of how sensitive they are. It means "at least very
# large", and the data carries no upper bound.
#
# So mean-based analyses cap the weights, and the cap is the fit for
# the strongest follower of one cue SHORT OF separation:
#
#   every trial where that cue carries information  -> answered as the
#                                                      cue predicts
#   every trial where it carries none               -> y = 0.5, the
#                                                      expected
#                                                      response
#   the other weight                                -> free
#
# What bounds it: a trial counted as uninformative is not EXACTLY zero
# on that cue, so a huge weight pushes its predicted probability away
# from the 0.5 that y asks for, and that costs likelihood. The cap
# therefore depends on the residual differences in those trials, which
# is why it must be computed on the frozen list rather than assumed.
#
# C1 is rank-based and needs no cap -- a rank does not care whether
# the top weight is 8 or 40. The cap is for the secondary mean-based
# group test, reported with and without it.
#
# CAP_INFORMATIVE_SD
#   The threshold that decides which trials carry information about a
#   cue, as |predictor| in SD units. 0.20 is step 6b's value.
#
#   THE CAP IS FLAT ACROSS 0.10 TO 0.20 on this design, because the
#   frozen list has a gap: the largest |pred_sat| among trials below
#   the line is about 0.08, and the next one up is about 0.24. So the
#   informative set is the same 120 trials anywhere in that range and
#   the registered figure does not depend on where the line sits. The
#   report states the whole sweep so that is visible rather than
#   asserted.
CAP_INFORMATIVE_SD = 0.20

# CAP_THRESHOLD_SWEEP
#   Thresholds the cap is recomputed at, to show the plateau.
CAP_THRESHOLD_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

# --- Scenarios ---
#
# SCENARIOS
#   True (b_sat, b_fog) for the synthetic participants, PER CORPUS SD.
#   A weight of 1.0 means a one-SD difference in that feature shifts
#   the log-odds of choosing by 1 -- from 50% to about 73% on a trial
#   whose difference is exactly one SD. The report translates every
#   one of these into a choice probability at the frozen list's median
#   informative difference, which is the readable form.
#
#   "tracks neither" is the pure-guesser null. The TASTE scenarios
#   below are the more demanding null, and the one the decision rule
#   has to survive.
SCENARIOS = {
    "tracks colour":  (1.0, 0.0),
    "tracks fog":     (0.0, 1.0),
    "tracks both":    (0.7, 0.7),
    "tracks neither": (0.0, 0.0),
}

# --- The taste nulls ---
#
# A PURE GUESSER IS THE WRONG NULL ON ITS OWN. Someone with no memory
# of derealization may still prefer duller or foggier pictures
# aesthetically, and that preference is real, stable, and would clear
# a threshold built from coin-flippers. A decision rule whose
# false-claim rate is quoted against pure guessers would understate
# itself against people who simply have taste.
#
# So the rule is also evaluated against non-responders drawn with
# zero-centred preferences: each person's weight on each feature is
# drawn from Normal(0, spread), independently for the two features.
# Zero-centred because taste has no predicted direction -- as many
# people should prefer the vivid image as the dull one -- and
# independent because a preference that loaded on both features in the
# same direction is the SHARED CONCEPT scenario, which belongs to C1
# below and would be a different (and much harder) null.
#
# TASTE_SPREADS_IN_FLOORS
#   The spreads, as multiples of THAT FEATURE'S OWN upper threshold.
#   Stated relative to the floor rather than in absolute units,
#   because the floors differ between the two features (about 0.29 for
#   colour and 0.48 for fog on this list) and a spread that is a mild
#   perturbation for one would be a severe one for the other.
#
#       0.25x   mild taste: few of these people clear the threshold
#       0.5x    moderate
#       1.0x    severe: the spread equals the detection threshold, so
#               a substantial share clear it in both directions. This
#               is a stress test, not a prediction.
#
#   The earlier [0.05, 0.10, 0.20] were absolute values set when the
#   predictors were scene-range magnitudes; against a 0.29 floor they
#   would all be near-trivial.
TASTE_SPREADS_IN_FLOORS = [0.25, 0.5, 0.75, 1.0]

# NONRESPONDER_SPREAD_MULT
#   The single spread ASSUMED for the non-responders wherever a power
#   figure needs one: the rule sweep's power side, the
#   equal-behaviour table, the prevalence section and C1.
#
#   Its own setting, because four places used to take the middle entry
#   of TASTE_SPREADS_IN_FLOORS instead. That coupled a list of spreads
#   to EVALUATE to a single spread to ASSUME, so adding an entry to
#   the list silently moved the assumption -- and 0.75x had to be
#   added to the list for the registered procedure's own false-claim
#   rate, which would have moved it from 0.5x to 0.75x with nothing in
#   the output saying so.
#
#   0.5x is the value every figure in this script was computed at.
#   It must appear in TASTE_SPREADS_IN_FLOORS so the nulls are
#   simulated at it; check_settings enforces that.
NONRESPONDER_SPREAD_MULT = 0.5

# N_TASTE_PEOPLE
#   Simulated non-responders per spread. A false-claim rate is a tail
#   quantity over a GROUP, so this is the number of people per
#   simulated study; the number of studies is N_SIM_STUDIES.
N_TASTE_PEOPLE = 20_000

# N_SIM_PARTICIPANTS
#   Simulated participants per scenario for the recovery analysis.
#   Higher is smoother; 500 is plenty.
N_SIM_PARTICIPANTS = 500

# --- Group-level power (shared across sets) ---
#
# SAMPLE_SIZES
#   Sample sizes to evaluate.
#
#   EXTENDED TO 120, AND THE "STOP AT 14" ADVICE IS GONE. The earlier
#   grid stopped at 40, and on it the fixed-count decision rule looks
#   as though it stops improving around 14 -- so the previous report
#   advised stopping there. That was an artefact of where the grid
#   ended and of the rule's integer k: power against N is a STAIRCASE,
#   flat until one more responder is expected to cross, then a step.
#   Reading a plateau off two treads as "no further gain" was wrong.
#   The grid now runs far enough past the treads for the shape to be
#   visible, and the report plots it rather than summarising it.
SAMPLE_SIZES = [8, 10, 12, 14, 15, 16, 20, 25, 30, 40, 60, 80, 120]

# EFFECT_SIZES
#   Between-participant Cohen's d for the proportion test. Suzuki et
#   al. (2024) reported d = 3.80, but their manipulation was DeepDream
#   versus photorealistic -- unmistakable. A photometric manipulation
#   should be far smaller, and Kusztor et al.'s decisive nulls point
#   the same way, so the low end is the realistic range.
EFFECT_SIZES = [0.3, 0.5, 0.8, 1.0, 1.5]

# EFFECT_AS_PROBABILITY
#   Report every effect size and weight ALSO as a choice probability,
#   at each feature's median informative difference on the frozen
#   list and at a round 1.0 SD.
#
#   A weight per SD is not a quantity anyone has intuitions about. A
#   choice probability is: "this person picks the duller image 63% of
#   the time on a typical colour trial" can be judged as plausible or
#   not, and 0.5-per-SD cannot. Both anchors are reported because the
#   median is the honest one for this list and 1.0 SD is the one that
#   stays comparable if the list ever changes.
EFFECT_AS_PROBABILITY = True
EFFECT_PROB_ANCHORS = ("median_informative", 1.0)

# N_SIM_STUDIES
#   Simulated studies per (N, d) cell for the power curve.
N_SIM_STUDIES = 2000

# ALPHA / ONE_TAILED
#   The directional prediction is that people prefer the more
#   manipulated image, so a one-tailed test is defensible -- but say so
#   in advance, in a time-stamped plan.
ALPHA = 0.05
ONE_TAILED = True

# --- Bayes factor (shared across sets) ---
#
# BF_SCALE
#   Scale of the JZS Cauchy prior on the effect size. 0.707 is the
#   default in the BayesFactor R package and needs no justification.
#   A WIDER prior makes null evidence look stronger, so fix this before
#   seeing the data and state it in the write-up.
BF_SCALE = 0.707

# BF_ONE_SIDED
#   Restrict the prior to POSITIVE effects, so the Bayes factor tests
#   the directional hypothesis the plan actually registers rather than
#   a two-sided alternative nobody predicted.
#
#   This matters in both directions. A one-sided factor is up to twice
#   the two-sided one when the data fall the predicted way, so it is
#   more sensitive to the effect; and it is SMALLER when they fall the
#   other way, so a group that trends backwards cannot accumulate
#   evidence "for" the hypothesis. The null-evidence figures are
#   regenerated with the same factor, since BF01 from a two-sided
#   prior is not the complement of a one-sided BF10.
BF_ONE_SIDED = True

# BF_THRESHOLD
#   Conventional threshold for "moderate evidence". BF10 > 3 is
#   evidence for an effect; BF01 > 3 (i.e. BF10 < 1/3) is evidence
#   against one.
BF_THRESHOLD = 3.0

# BF_FORM_TOL
#   How closely the rewritten effect-size form of the JZS factor must
#   reproduce Rouder et al.'s g form in the two-sided case. They are
#   two parameterisations of one integral, so the only difference
#   should be quadrature error.
BF_FORM_TOL = 1e-6

# --- Dose-response ---
#
# DOSE_EFFECT_GRID / DOSE_SUMMARY
#   The weights the expected dose-response curve is drawn at. Same
#   history and same fix as the contrast baseline: (0.7, 0.7) was set
#   in the old units and means different things on the two features
#   now, so the curve is drawn across the common grid and the single
#   summary value is stated as an assumption.
DOSE_EFFECT_GRID = [0.30, 0.50, 0.70, 0.85, 1.00]
DOSE_SUMMARY = 0.50
N_DOSE_REPS = 200

# DOSE_BINS_FROM_LIST
#   Read the magnitude bins from the frozen list's own `mag_bin`
#   column instead of assuming a count.
#
#   N_MAGNITUDE_BINS below said 3, which was right for the old
#   quantile bins. Step 5 now uses FIXED value edges -- five for
#   colour, of which three are usable at this cut, and three for each
#   fog-driven type -- so an assumed count would either drop a bin or
#   invent an empty one. The list knows; ask it.
DOSE_BINS_FROM_LIST = True

# --- The floors ---
#
# FLOOR_ALPHA
#   The per-tail false-alarm rate for classifying ONE person on ONE
#   feature. 0.025 in each tail, from the SAME guessers:
#
#       upper threshold   97.5th percentile of guessers
#       lower threshold    2.5th percentile of guessers
#
#   THE UPPER THRESHOLD IS UNCHANGED from steps 6 and 6b, so those
#   results stand and did not need re-running. The lower one is new
#   and is used only to count opposite-direction detections for the
#   decision rule.
#
#   The per-person rate is therefore 2.5% in the predicted direction
#   and 5% in either direction. Neither is the registered quantity:
#   what gets registered is the RULE's false-claim rate across a
#   whole sample, computed below. The 2.5% figure is also the
#   within-participant false-positive rate the prevalence inference
#   takes as given.
FLOOR_ALPHA = 0.025

# N_FLOOR_PEOPLE
#   Guessers simulated to locate the floors. Large, because a floor is
#   a TAIL percentile: at 20,000 people, 500 of them sit above it, which
#   is enough for the estimate to be stable. The scenario clouds below
#   stay at N_SIM_PARTICIPANTS, since they only need a mean and a
#   spread.
N_FLOOR_PEOPLE = 20_000

# --- Per-person detection ---
#
# The recovery section reports the FLOOR: the weight at which a person
# can just barely be told apart from a guesser. That is a coin-flip
# point -- at exactly the floor you would catch them about half the
# time. For planning you want the weight at which you would catch them
# RELIABLY, which is what these settings produce.
#
# DETECTION_TARGET
#   The detection rate to solve for. 0.80 is the usual planning value.
DETECTION_TARGET = 0.80

# DETECTION_GRID
#   Candidate true weights. The curve is computed at each and the
#   target is found by interpolating between them.
DETECTION_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

# N_DETECT_PEOPLE
#   Simulated participants per grid point, per feature.
N_DETECT_PEOPLE = 600

# DETECTION_SIDED
#   "two"  each person is classified PREDICTED (above the upper
#          threshold), OPPOSITE (below the lower) or NOT DETECTED
#          (between). This is the registered classification: the
#          decision rule counts both directions, because a rule that
#          only counts the predicted direction cannot state what share
#          of a sample went the other way, and that share is what
#          separates a real effect from a noisy one.
#   "one"  the old behaviour, upper threshold only. Kept for the
#          reproduction check.
#
#   DETECTION in the curves below always means "predicted", under
#   either setting, so the detection curves stay comparable.
DETECTION_SIDED = "two"

# --- The decision rule ---
#
# A per-person classification is not a claim. The claim is about the
# SAMPLE: "enough of these people track colour that we are willing to
# say the group does". That needs a rule fixed in advance, and a rule
# has a false-claim rate of its own that has nothing to do with the
# 2.5% per person.
#
# The family of rules considered:
#
#     claim the feature if at least k people are PREDICTED
#     and at most m are OPPOSITE
#
# The second clause is what the lower threshold buys. Without it, a
# sample where 6 people went one way and 5 the other would read the
# same as one where 6 went one way and nobody went the other, and
# those are very different pieces of evidence.
#
# THIS SCRIPT DOES NOT PICK k AND m. It reports, for every candidate
# pair at every sample size: the false-claim rate under pure guessers,
# the false-claim rate under each taste spread, and the power under
# each responder-share scenario. The choice is a judgement about which
# error matters more and belongs with the registered plan, not with a
# simulation.
#
# RULE_K_MAX / RULE_M_ABSOLUTE / RULE_M_FRACTIONS
#   How far to sweep, and in which family.
#
#   k runs from 1 to min(N, max(RULE_K_MAX, ceil(RULE_K_FRAC * N))),
#   so a large sample is not artificially truncated: at N = 120 the
#   best k is well past 12, and a fixed cap of 12 would have made
#   large samples look worse than small ones for no reason but the
#   grid.
#
#   m RUNS IN TWO FAMILIES, and the difference matters:
#
#     ABSOLUTE    "at most m people classified opposite", m a count.
#                 THIS FAMILY DEGENERATES AS N GROWS. Opposite
#                 classifications scale with the sample: at the middle
#                 taste spread a non-responder is classified opposite
#                 about 8% of the time, so 108 of them produce about
#                 8.4 opposite on average. "At most 4" is then
#                 satisfied 99.99% of the time at N = 8 and 7% of the
#                 time at N = 120 -- so the rule's POWER FALLS as the
#                 sample grows, which is not a fact about the
#                 experiment but about the rule.
#
#     PROPORTIONAL "at most a fraction f of the sample classified
#                 opposite". Scales, so the clause means the same
#                 thing at every N.
#
#   Both are swept because the absolute family is what was specified
#   and the proportional one is what demonstrates why it cannot be
#   used at large N. The report states this rather than leaving a
#   falling power curve to be read as a result.
RULE_K_MAX = 12
RULE_K_FRAC = 0.4
RULE_M_ABSOLUTE = [0, 1, 2, 3, 4]
RULE_M_FRACTIONS = [0.0, 0.05, 0.10, 0.15, 0.20]

# --- A third family: the sign test ---
#
# RULE_SIGN_TEST
#   Among the people classified EITHER WAY, test predicted against
#   opposite with a one-sided exact binomial test at p = 0.5.
#
#   WHY THIS FAMILY IS DIFFERENT, and better behaved than both the
#   others. Taste is zero-centred, so for a non-responder the two
#   directions are exchangeable: predicted and opposite are equally
#   likely, whatever the spread. CONDITIONAL ON THE NUMBER DETECTED,
#   the number predicted is then Binomial(d, 0.5) under ANY symmetric
#   null -- guessers, mild taste, severe taste, all identical. So this
#   rule's false-claim rate DOES NOT DEPEND ON THE TASTE SPREAD, which
#   is the one quantity in the whole design that cannot be measured in
#   advance.
#
#   The price is discreteness. The test cannot fire below 5 detections
#   at alpha = 0.05 -- even 4 out of 4 predicted gives p = 0.0625 --
#   and at 5, 6 or 7 detections it requires EVERY detected person to
#   be in the predicted direction. So it is unusable at small samples
#   and strong at large ones, which is the opposite of the absolute-m
#   family. The report shows where the crossover is.
RULE_SIGN_TEST = True

# RULE_SIGN_ALPHA
#   One-sided alpha for the binomial test.
RULE_SIGN_ALPHA = 0.05

# RULE_SIGN_MIN_PREDICTED
#   An optional floor on the number of predicted detections, on top of
#   the test. 0 means the test alone decides. The floor guards the case
#   where a handful of detections all point the right way by luck: 5
#   of 5 is significant at 0.031 but it is still only five people.
RULE_SIGN_MIN_PREDICTED = [0, 3, 5, 8]

# --- Admissibility: two standards, neither of them chosen here ---
#
# ADMISSIBILITY_STANDARDS
#   Which nulls a rule's false-claim rate has to satisfy to be called
#   admissible. Both are reported because the choice is a judgement
#   about how much aesthetic preference to expect in a population, and
#   that is not a question a simulation can settle.
#
#     "strict"    every null, INCLUDING the 1.0x taste spread, at or
#                 below RULE_TARGET_FALSE. The 1.0x spread puts a
#                 non-responder's preference SD at the detection
#                 threshold itself, which is severe.
#     "practical" guessers and the 0.25x and 0.5x spreads at or below
#                 RULE_TARGET_FALSE, with 1.0x reported beside it as a
#                 stress test rather than as a constraint.
#
#   Every rule row carries both verdicts, so the table can be read
#   either way without re-running anything.
ADMISSIBILITY_STANDARDS = {
    "practical": [0.25, 0.5],
    "strict": [0.25, 0.5, 0.75, 1.0],
}

# ADMISSIBILITY_AT_BOUNDARY
#   A third standard, evaluated alongside the two above because the
#   registered procedure's combined false-claim rate turns out to peak
#   AT the control check's boundary rather than beyond it: when the
#   true spread sits on 0.75x the check splits about evenly, so the
#   practical rule is applied to a population it was never required to
#   hold against. Requiring it to hold at the boundary closes that gap
#   by construction, and the rule table shows what the extra
#   constraint costs in power.
ADMISSIBILITY_STANDARDS["practical_at_boundary"] = [0.25, 0.5, 0.75]

# RULE_TARGET_FALSE
#   The false-claim rate a rule has to beat to be worth tabulating as
#   admissible. 0.05 is conventional. Rules above it are still
#   reported -- the table is the evidence -- but flagged.
RULE_TARGET_FALSE = 0.05

# RULE_MC_MARGIN
#   Certify admissibility against the UPPER END of a 95% interval for
#   the false-claim rate, not against the point estimate.
#
#   The rate is enumerated exactly, but from a classification rate
#   measured on N_TASTE_PEOPLE simulated non-responders, so it
#   inherits that estimate's error. Without the margin, a rule sitting
#   on the line is certified about half the time -- and one did: the
#   strict rule came back at 0.0512 in one cell of the control-check
#   section while the sweep had certified it admissible.
#
#   RULE_MC_MARGIN_Z is 1.96, the upper end of a two-sided 95%
#   interval. Set it to 1.645 for a one-sided 95% bound, which is the
#   weaker requirement.
RULE_MC_MARGIN = True
RULE_MC_MARGIN_Z = 1.96

# RULE_MC_STEP
#   The finite-difference step used to differentiate the enumerated
#   false-claim rate with respect to the classification rate. Small
#   enough that the derivative is local, large enough that the
#   difference of two rates is well above float64 cancellation; the
#   sweep prints a check that halving it does not move the answer.
RULE_MC_STEP = 1e-4

# RULE_RESPONDER_SHARES
#   The share of the sample that genuinely tracks the feature, for the
#   power side of the rule table. The rest are drawn from the taste
#   null at the middle spread.
#
#   0.10 is the pessimistic case worth planning against: in a sample
#   of 30, three real responders.
RULE_RESPONDER_SHARES = [0.10, 0.25, 0.50, 1.00]

# RULE_RESPONDER_WEIGHTS
#   The true weights a genuine responder might carry, per SD, for the
#   rule table. Swept rather than fixed: a rule's power depends on how
#   strong the real responders are, and quoting one number without its
#   effect size is how a power figure becomes unusable.
#
#   Every power column in rule_sweep.csv is keyed by all three of
#   effect size, responder share and lapse rate, and the report
#   translates each weight into a choice probability at the frozen
#   list's own median informative difference.
RULE_RESPONDER_WEIGHTS = [0.30, 0.50, 0.70]

# RULE_SUMMARY_N / RULE_SUMMARY_SHARES / RULE_SUMMARY_LAPSES
#   The cells the compact decision table reports. Smaller than the
#   full sweep on purpose: the sweep is the evidence and the summary
#   is what a decision gets made from.
RULE_SUMMARY_N = [10, 15, 20, 30]

# RULE_SUMMARY_SHARES
#   0.10 and 0.25 are the cells where the three families separate. At
#   0.5 and 1.0 the count families are at or near ceiling for both
#   features at every realistic N, so a comparison there cannot
#   distinguish them -- which is why the low shares were added.
RULE_SUMMARY_SHARES = [0.10, 0.25, 0.5, 1.0]
RULE_SUMMARY_LAPSES = [0.0, 0.1]

# RULE_SUMMARY_PICK
#   The cell the "best admissible rule" is chosen by, stated because
#   any such choice needs a criterion and a different cell could pick
#   a different rule. Mid-range of what the summary reports.
#   Set to a cell that DISCRIMINATES. At effect 0.50 / share 0.50 /
#   lapse 0 several rules tie at power near 1.00 and "best" is
#   whichever the tie-break happens to hit, which is no criterion at
#   all. This cell is demanding enough that the families separate.
#
#   It only LABELS the summary. The registered rule is chosen from the
#   table, not from this setting.
RULE_SUMMARY_PICK = (0.30, 0.25, 0.1)   # effect, share, lapse

# --- The registered admissibility procedure ---
#
# REGISTERED_STANDARD -- "strict", AND NOT CHOSEN FROM DATA.
#
#   The existence claim is evaluated with the strict rule: the rule
#   must hold at or below RULE_TARGET_FALSE against a guesser AND
#   against taste spreads of 0.25x, 0.75x and 1.0x each feature's own
#   threshold. No branch, no control-dependent choice of rule.
#
#   WHY NOT THE CHEAPER OPTION. The practical standard plus a
#   control-based check was costed first, because the practical rule
#   has more power and the check was supposed to catch the population
#   it is not admissible against. It does not work: the combined
#   false-claim rate PEAKS AT THE CHECK'S OWN BOUNDARY, where the
#   estimate falls either side about half the time, so half the
#   studies apply the practical rule to a population it was never
#   required to hold against. Measured at 0.113 as decided, and 0.064
#   for both repaired variants -- all above alpha. It also fails to
#   improve with more controls, because below the minimum the
#   procedure forces strict and is safe by construction.
#
#   run_control_gated_rule still computes all of it. Those numbers are
#   the REASON strict was registered, so they stay in the report; they
#   are no longer on the registered path.
REGISTERED_STANDARD = "strict"

# CONTROL_CHECK_REGISTERED
#   False: the control check does not choose the rule. The control
#   group's taste-spread estimate is still computed and reported --
#   it is the only measurement of the quantity every rule is judged
#   against -- but DESCRIPTIVELY, and no claim depends on it.
CONTROL_CHECK_REGISTERED = False

# --- The rejected procedure's two branches ---
#
# PROCEDURE_DEFAULT_STANDARD / PROCEDURE_FALLBACK_STANDARD
#   The standards the REJECTED procedure would have switched between:
#   the practical rule by default, the strict rule when the control
#   check fired.
#
#   THESE ARE SEPARATE FROM REGISTERED_STANDARD ON PURPOSE. They are
#   properties of the design that was costed and refused, so they do
#   not move when the registered standard moves. An earlier version
#   read the procedure's default off REGISTERED_STANDARD; when that
#   became "strict" the procedure silently began comparing the strict
#   rule with itself, and every variant reported an identical rate.
#   check_settings now enforces that the two branches differ.
PROCEDURE_DEFAULT_STANDARD = "practical"
PROCEDURE_FALLBACK_STANDARD = "strict"
BOUNDARY_STANDARD = "practical_at_boundary"

# Kept as an alias because the branch simulation reads it; it is the
# procedure's fallback, not the registration's.
FALLBACK_STANDARD = PROCEDURE_FALLBACK_STANDARD

# CONTROL_CHECK
#   The pre-registered sequence run on the control group BEFORE the
#   existence claim is evaluated:
#
#     1. ASYMMETRY. Compare the controls' predicted and opposite
#        classification counts. Under any symmetric taste null the two
#        are exchangeable, so a binomial test at p = 0.5 is exact.
#        A significant excess of predicted means the shared-concept
#        strategy is operating -- controls working out that the
#        experiment is about edited images and picking the more edited
#        one, which loads positively on BOTH features.
#
#        ITS CONSEQUENCE IS NOT A SEPARATE BRANCH, and that is a
#        deliberate reading of the decision rather than something it
#        stated. The strategy inflates the moment estimate of the
#        spread (it adds its own variance), so a strategy-contaminated
#        control group pushes the procedure TOWARD the strict rule,
#        which is the conservative direction. The detection rate is
#        reported so the inflation is visible, and the report asks for
#        a ruling if a separate branch is wanted.
#
#     2. SPREAD. Estimate the taste spread with the moment estimator
#        -- sqrt(Var(fitted) - Var(fitted | guessers)) -- which is the
#        better of the two estimators tested in the taste-recovery
#        section.
#
#     3. BRANCH -- NOT REGISTERED. Above CONTROL_CHECK_BOUNDARY
#        times that feature's own threshold the claim would have been
#        evaluated with the strict rule and at or below it with the
#        practical rule. This branch was COSTED AND REJECTED: see
#        REGISTERED_STANDARD. It is still simulated, because its rate
#        is the reason the strict standard is registered.
#
#     4. TOO FEW CONTROLS. Below CONTROL_CHECK_MIN_CONTROLS the
#        estimate is too noisy to steer any choice. Moot now that
#        nothing is steered by it, and informative anyway: it is the
#        regime in which the rejected procedure was SAFE, which is
#        why its combined rate did not improve with more controls.
CONTROL_CHECK = True
CONTROL_CHECK_BOUNDARY = 0.75
CONTROL_CHECK_MIN_CONTROLS = 30
CONTROL_CHECK_ASYMMETRY_ALPHA = 0.05

# CONTROL_CHECK_SPREADS
#   True taste spreads the procedure is evaluated at, as multiples of
#   each feature's own threshold. 0.75 is the boundary itself -- the
#   worst case for a threshold rule is the truth sitting on it -- and
#   1.25 is past the largest spread anyone has proposed, included
#   because a procedure that protects only up to its own boundary has
#   not been tested.
CONTROL_CHECK_SPREADS = [0.25, 0.5, 0.75, 1.0, 1.25]

# CONTROL_CHECK_N / CONTROL_CHECK_REPS
#   Control group sizes the procedure is scored at, and simulated
#   groups per cell. 30 is the registered plan and the minimum the
#   branch is allowed at; 15 and 20 show the below-minimum regime the
#   procedure falls back from, and 45 is there because the combined
#   rate does NOT improve with more controls -- recruiting past the
#   minimum moves the study out of the safe forced-strict regime into
#   the noisy-branch one, and the table has to show that.
CONTROL_CHECK_N = [15, 20, 30, 45]
CONTROL_CHECK_REPS = 4000

# CONTROL_POOL_PEOPLE
#   People fitted per control type. Groups are DRAWN from the pool
#   rather than simulated one by one, for the same reason C1 does it:
#   a person's fitted weight depends only on their own type, so
#   refitting per group repeats the work. 20,000 against a largest
#   group of 45 means one group uses 0.2% of a pool.
CONTROL_POOL_PEOPLE = 20_000

# CONTROL_CHECK_STRATEGY_RATES
#   The share of CONTROLS using the shared-concept strategy, swept.
#   It matters twice over: the strategy inflates the predicted count
#   the rule sees, and it inflates the spread estimate the branch is
#   taken on. Stage 1 is what detects it, and 0.0 is the column that
#   gives stage 1's own false-alarm rate.
CONTROL_CHECK_STRATEGY_RATES = [0.0, 0.25, 0.5]

# CONTROL_CHECK_UPPER_Z
#   For variant C, the one-sided normal quantile used to turn the
#   spread estimate into an upper bound. 1.645 is 95%.
CONTROL_CHECK_UPPER_Z = 1.645

# --- The behavioural index ---
#
# BEHAVIOUR_TARGETS
#   Choice probabilities on a TYPICAL INFORMATIVE TRIAL of each
#   feature. This is the PRIMARY way the existence claims are stated,
#   and the per-SD weights are the secondary form.
#
#   The reason is that a weight per corpus SD has no perceptual
#   meaning, and indexing by it makes H1 (fog) look several times
#   worse powered than H2 (colour) when most of that gap is an
#   artefact of the index: colour's informative trials are further
#   apart than fog's, so the same per-SD weight is not the same
#   behaviour.
#
#   The medians, the threshold ratio and the residual gap are all
#   DERIVED FROM THE FROZEN LIST and printed in the equal-behaviour
#   section. They are not quoted here -- the figures that used to be
#   came from a step-5 test draw and did not survive the freeze.
#
#   Translating back: the weight that produces choice probability p on
#   a typical trial is log(p / (1 - p)) / median_informative, which
#   differs between the features by exactly the ratio of their
#   medians. That ratio (about 1.62) very nearly cancels the ratio of
#   their thresholds (about 1.67), which is why the two features come
#   out nearly equal once indexed this way.
BEHAVIOUR_TARGETS = [0.55, 0.60, 0.65]

# --- Recovering the taste spread from the controls ---
#
# The decision rule's admissibility depends on how much aesthetic
# taste the population has, and that spread cannot be measured in
# advance. But it CAN be measured afterwards, from the control group:
# controls are the people whose weights should be zero-centred, so the
# spread of their fitted weights estimates the very quantity the
# standard was chosen against.
#
# That turns an assumption into a check. A rule registered as
# admissible up to the 0.5x spread can be reported as supported if the
# controls come in at or below 0.5x, and DOWNGRADED if they come in
# higher -- decided by a pre-registered rule rather than after seeing
# whether the result survived.
#
# TASTE_RECOVERY_N
#   Control group sizes to evaluate the recovery at.
TASTE_RECOVERY_N = [20, 30]

# TASTE_RECOVERY_REPS
#   Simulated control groups per (size, true spread). The reported
#   quantity is the sampling distribution of the estimate, so this
#   needs to be large enough to see its spread, not just its mean.
TASTE_RECOVERY_REPS = 2000

# --- Prevalence inference, as an alternative to a fixed rule ---
#
# A fixed count rule answers "do we claim the feature". It does not
# answer "in what share of the population", and the second is the more
# useful quantity -- it is also the one that does not jump when one
# more person crosses a threshold.
#
# Bayesian prevalence (Ince, Paton, Kay & Schyns, eLife 2021) inverts
# the within-participant test instead. With n participants tested
# individually at a within-participant false-positive rate alpha, and
# k of them testing positive, the number positive is binomial with
#
#     theta = gamma * (1 - beta) + (1 - gamma) * alpha
#
# where gamma is the population prevalence of a true effect and
# (1 - beta) is the within-participant sensitivity. The posterior over
# gamma follows from a uniform prior on [0, 1]. Here alpha is
# FLOOR_ALPHA -- 2.5%, by construction of the upper threshold, not
# assumed -- and (1 - beta) is read off this script's own detection
# curve at the weight in question, which is the part a simulation can
# supply and an analysis of real data cannot.
#
# THE LIKELIHOOD IS WRITTEN OUT ABOVE ON PURPOSE, so the
# implementation can be checked against the paper rather than taken on
# trust.
#
# PREVALENCE_GRID
#   The gamma values the posterior is evaluated on.
PREVALENCE_GRID = None      # None -> 0, 0.005, ..., 1.0 (201 points)

# PREVALENCE_HPDI
#   Mass of the reported posterior interval.
PREVALENCE_HPDI = 0.95

# PREVALENCE_N_GRID / PREVALENCE_C_GRID
#   Grid resolution for the registered posterior: points on the
#   prevalence axis, and on the opposite-direction rate that is
#   marginalised out. 201 x 201 puts the prevalence grid at 0.005 and
#   is finer than the posterior is sharp at any realistic n.
PREVALENCE_N_GRID = 201
PREVALENCE_C_GRID = 201

# PREVALENCE_TRUE_GAMMAS
#   The true prevalences the registered variant's operating
#   characteristics are computed at. 0.0 is the one that matters: it
#   is where the method's own false-positive rate is measured.
PREVALENCE_TRUE_GAMMAS = [0.0, 0.10, 0.25, 0.50]

# PREVALENCE_ZERO_EPS
#   A posterior interval counts as excluding zero when its lower bound
#   is above this. Not exactly zero, because the grid's first point IS
#   zero and a lower bound sitting on it would otherwise count as
#   excluding it.
PREVALENCE_ZERO_EPS = 0.001

# PREVALENCE_DESCRIPTIVE
#   REGISTERED DECISION: prevalence is reported DESCRIPTIVELY, for
#   both H1 (fog) and H2 (colour) -- the corrected estimator and its
#   interval -- with NO "the interval excludes zero" inference drawn
#   from it.
#
#   Two reasons, both measured here rather than argued:
#
#     it is underpowered as a test. The interval excludes zero only
#     about a fifth of the time at a true prevalence of 25%, and
#     reaches 0.6-0.7 only at 50%. A test that cannot detect the
#     effect it is looking for at the prevalence anyone expects is not
#     worth registering as a test.
#
#     its false-positive rate is not uniform. Under the widest taste
#     null the interval excludes zero well above alpha, and much more
#     for fog than for colour, because fog's sensitivity at a given
#     weight is lower and its taste-null rate higher.
#
#   P_excludes_zero stays in the output. It is NOT a test statistic
#   here: it is the guide to how far the interval can be trusted under
#   each assumption about the null, which is exactly what a reader of
#   a descriptive estimate needs.
PREVALENCE_DESCRIPTIVE = True

# --- Two contrasts the per-feature floors cannot answer ---
#
# The floors above say whether ONE weight differs from zero. Two of the
# questions this experiment exists to ask are not of that form:
#
#   CROSSOVER    is this person's fog weight BIGGER THAN their colour
#                weight? That is a question about the DIFFERENCE
#                between two noisy estimates, and its noise is larger
#                than either one's -- roughly the two scatters added in
#                quadrature. A person can clear the fog floor, miss the
#                colour floor, and still not clear the crossover.
#
#   ADDITIVITY   when both cues move together, do their effects simply
#                add? That is the `together` and `conflict` trials'
#                real job, and no two-weight model can answer it,
#                because a two-weight model assumes additivity.
#
# RUN_CONTRASTS
#   Set False to skip both and leave the script as it was.
RUN_CONTRASTS = True

# CONTRAST_BASELINE
#   The weight BOTH features carry in the null population for these
#   two contrasts.
#
#   This differs from the per-feature floors above, whose null is a
#   person who tracks nothing at all. Here the null is a person who
#   tracks BOTH cues EQUALLY (crossover) and ADDITIVELY (additivity),
#   so the baseline has to be a plausible operating weight rather than
#   zero -- the precision of a logistic estimate depends on where it
#   sits, so a null at zero would understate the noise. 0.7 matches
#   SCENARIOS["tracks both"].
#
#   NOT A SINGLE VALUE ANY MORE. 0.7 was chosen when the predictors
#   were z_sat / z_fog -- scene-range magnitudes -- and per corpus SD
#   the same 0.7 is about 2.4x the colour threshold but only 1.5x the
#   fog one, which puts colour near its detection ceiling and fog
#   mid-range. A null person defined that way is not the same person
#   on the two features, so a contrast computed at it is not
#   comparable between them.
#
#   So both contrasts now run over the COMMON EFFECT GRID, and where
#   a single summary number is needed it is taken at
#   CONTRAST_SUMMARY, labelled in the report as an assumption rather
#   than a measurement.
CONTRAST_BASELINE_GRID = [0.30, 0.50, 0.70, 0.85, 1.00]
CONTRAST_SUMMARY = 0.50

# CONTRAST_ALPHA
#   Two-sided false-alarm rate for both contrasts. TWO-sided, unlike
#   the per-feature detection above, because neither contrast has a
#   predicted direction: the design is meant to find colour-trackers
#   as well as fog-trackers, and a departure from additivity could go
#   either way (cues reinforcing, or saturating).
CONTRAST_ALPHA = 0.05

# CROSSOVER_GRID
#   True gaps between the two weights to test, in weight units. The
#   person is simulated with b_sat = baseline - gap/2 and
#   b_fog = baseline + gap/2, so the gap is the whole separation and
#   the average weight stays at the baseline.
CROSSOVER_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]

# GAMMA_GRID
#   True departures from additivity to test. Gamma is the PROPORTIONAL
#   change in the combined evidence when both cues are at their full
#   range: the simulated score is
#
#       (b_sat*x_sat + b_fog*x_fog) * (1 + gamma*|x_sat|*|x_fog|)
#
#   so gamma = +0.5 means the two cues together carry 50% more weight
#   than their sum at the largest differences (reinforcing), and
#   gamma = -0.5 means 50% less (saturating). Gamma = 0 is additive.
#
#   Written this way, and not as a plain product of the two
#   differences, because the plain product does NOT change sign when
#   the two images swap sides -- and the sides ARE randomised, so a
#   plain product would break the model's symmetry and estimate
#   nothing. The multiplier here is even under the swap and the term
#   it multiplies is odd, so the whole score flips correctly. The
#   start-up check verifies it.
GAMMA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# CONTRAST_POOL_N
#   Additivity is hopeless in ONE person -- the first run of this
#   section showed a departure of gamma = 2 caught only about half the
#   time from 180 trials. It is not hopeless across people, because the
#   mixed trials pool: n participants give n times as many of them.
#
#   For each n here the section reports the gamma that n pooled
#   participants could detect. The pooled fit treats everyone as
#   sharing one gamma, which is an upper bound on what a proper
#   mixed-effects fit would achieve -- real people differ, and that
#   spread is not simulated here. Read these as "no better than this",
#   and set n = 1 to recover the single-person answer.
CONTRAST_POOL_N = [1, 5, 10, 15]

# N_POOL_REPS
#   Simulated STUDIES per (pool size, gamma) cell.
N_POOL_REPS = 300

# N_CONTRAST_PEOPLE
#   Simulated participants per grid point. The contrasts are noisier
#   than the single-feature curves, so this is larger than
#   N_DETECT_PEOPLE.
N_CONTRAST_PEOPLE = 1200

# CONTRAST_TARGET
#   Detection rate to solve for, as with DETECTION_TARGET.
CONTRAST_TARGET = 0.80

# --- Lapses ---
#
# LAPSE_RATES
#   A participant lapses with probability lambda: on that trial they
#   ignore the images and answer at chance, so
#
#       p = (1 - lambda) * logistic(X b) + lambda / 2
#
#   Step 6b established that this attenuates the fitted weight by
#   roughly (1 - lambda), a little more at large weights -- 0.89 at a
#   true weight of 0.5 and lambda 0.1, 0.86 at a true weight of 1.0.
#   So lapses cost power, and a power figure computed only at
#   lambda = 0 is optimistic.
#
#   These apply to H1 (fog) and H2 (colour) as well as to C1, which
#   earlier versions did not do.
LAPSE_RATES = [0.0, 0.1, 0.2]

# --- C1: the group comparison ---
#
# C1 is the registered claim that the DPDR group weights the two
# features differently from controls. Three things make it harder than
# it looks.
#
# FIRST, THE TEST IS RANK-BASED. A separable participant's Firth
# weight is finite but means "at least very large", so a mean-based
# test would be comparing a number the data does not pin down. A rank
# does not care: the strongest responder is the top rank whether their
# weight is 8 or 40. So the registered test is a one-sided
# Mann-Whitney U on the per-person weights, and no cap is needed for
# it.
#
# SECOND, BOTH FEATURES ARE TESTED. Colour is the predicted axis and
# fog is the control, and both get their own one-sided test with a
# Holm correction across the two. The report gives power for each
# feature and for "either passes", and the registered form will be
# decided from those numbers.
#
# THIRD, CONTROLS ARE NOT NON-RESPONDERS. A control who has never
# experienced derealization can still work out what the experiment is
# asking and pick the more edited image -- the concept is not secret,
# and the instructions describe it. That strategy loads POSITIVELY ON
# BOTH FEATURES, so it eats the very contrast C1 rests on. And
# participants know the concept too, so the strategy appears at the
# SAME RATE IN BOTH GROUPS; what appears only in participants is the
# remembered experience.
#
# C1_STRATEGY_RATES
#   The share of EACH group using the "pick the more edited image"
#   strategy. Swept, because it is unmeasurable in advance and it is
#   the parameter most likely to decide the result.
C1_STRATEGY_RATES = [0.0, 0.25, 0.5]

# C1_STRATEGY_WEIGHT
#   The strategy's weight, applied equally to both features. From the
#   same grid as the responders, so the two are comparable.
C1_STRATEGY_WEIGHT = 0.5

# C1_PARTICIPANT_N / C1_CONTROL_RATIOS / C1_CONTROL_ABSOLUTE
#   Group sizes. Controls are swept as multiples of the participant
#   group and also at a fixed 30, since a control group is usually
#   easier to recruit than a clinical one and the fixed case is the
#   realistic plan.
#   The sizes stop at 60 participants. Larger groups were in an
#   earlier version of this sweep and were dropped: a DPDR sample of
#   80 or 120 is not what this study will recruit, every such cell
#   costs the same as a realistic one, and the trend from 10 to 60 is
#   already monotone and clear.
C1_PARTICIPANT_N = [10, 15, 20, 30, 60]
C1_CONTROL_RATIOS = [1.0, 1.5, 2.0]
C1_CONTROL_ABSOLUTE = [30]

# C1_LAPSE_PAIRS
#   (participant lambda, control lambda). BOTH DIRECTIONS, because the
#   asymmetry can manufacture the effect as easily as hide it: a
#   control group that lapses more has its weights attenuated toward
#   zero, which widens the gap C1 is looking for and would be
#   indistinguishable from the real thing.
C1_LAPSE_PAIRS = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0),
                  (0.0, 0.1), (0.0, 0.2), (0.1, 0.1), (0.2, 0.2)]

# C1_EXACT_MAX_N
#   Below this total sample size the Mann-Whitney null is computed
#   exactly by enumeration rather than from the normal approximation.
#   The approximation is poor at small n and C1 may well be small.
C1_EXACT_MAX_N = 20

# C1_POOL_PEOPLE
#   How many simulated people are fitted per distinct PERSON TYPE.
#
#   The naive structure -- simulate and fit every group of every study
#   of every cell -- needs about nine million Firth fits and will not
#   run. But a person's fitted weight depends only on their own type:
#   which feature carries the effect, how large it is, whether they
#   use the strategy, their taste draw and their lapse rate. People
#   are independent, so fitting a large POOL per type once and then
#   drawing groups from the pool is the same simulation, done in the
#   order that does not repeat work.
#
#   The pool must be large relative to the biggest group so that
#   drawing WITHOUT replacement stays close to drawing fresh people:
#   at 20,000 and a largest group of 240, a study uses 1.2% of a pool.
C1_POOL_PEOPLE = 20_000

# C1_STUDIES
#   Simulated studies per cell. Separate from N_SIM_STUDIES because
#   C1 has far more cells than the group-level power curve.
C1_STUDIES = 2000

# C1_HOLM
#   Apply a Holm correction across the two features' one-sided tests.
C1_HOLM = True

# C1_ZERO_MIN_N
#   The smallest subsample the zero-catch-failure secondary will test.
#   Below it the study is reported as unusable rather than tested: a
#   one-sided rank test on three people against three cannot reach
#   0.05 whatever the data say, so a "result" there would be an
#   artefact of the test's granularity. The share of studies that stay
#   usable is reported, since at a high control lapse rate it is the
#   limiting factor rather than the power.
C1_ZERO_MIN_N = 4

# --- Responder share among the participants ---
#
# C1_RESPONDER_SHARES
#   Not every DPDR participant need carry the remembered effect. The
#   share that does is swept, because C1's power depends on it and it
#   is unmeasurable in advance.
C1_RESPONDER_SHARES = [0.5, 1.0]

# --- The attention gate ---
#
# WHAT IT IS FOR. A control group that lapses MORE than the
# participants has its weights attenuated toward zero, which widens
# exactly the gap C1 looks for. Simulated, that inflates C1's
# false-positive rate from 0.05 to as much as 0.27, and nothing in the
# choice data distinguishes it from a real effect -- the joint lapse
# model cannot, as the recovery check below shows.
#
# So the gate is checked BEFORE C1 is interpreted, and it uses the one
# measurement that is not degenerate with the weights.
#
# GATE_ON_CATCH
#   Compare the groups' catch-failure counts, one-sided, controls
#   worse. Catch trials are the only source of information about the
#   lapse rate that the weights cannot imitate: a catch trial has no
#   images, so no preference can produce a failure.
#
#   CONSISTENCY RATES ARE DELIBERATELY NOT IN THE GATE. A participant
#   with a stronger preference answers a repeated pair the same way
#   more often, so a consistency gate would fire precisely when the
#   effect is real. Consistency stays descriptive.
GATE_ON_CATCH = True
GATE_ALPHA = 0.05

# N_CATCH_TRIALS / CATCH_P_FAIL_PER_LAPSE
#   The catch block and the per-trial failure probability of someone
#   who is not reading.
#
#   0.75 comes from the response set: four live keys (left and right
#   for the image choice, up and down for the catch trials), so a
#   random press is wrong three times in four. The per-trial failure
#   probability is therefore 0.75 * lambda.
#
#   THIS IS AN UPPER BOUND ON THE GATE'S SENSITIVITY, and the report
#   says so. It assumes someone lapses on catch trials at the same
#   rate as on image trials. A catch screen looks different -- no
#   pictures, one line of text -- so a participant who is skimming may
#   well attend to it more than to the hundredth image pair. If they
#   lapse less on catch trials, every simulated detection rate below
#   is optimistic.
N_CATCH_TRIALS = 10
CATCH_P_FAIL_PER_LAPSE = 0.75

# LAPSE_BETWEEN_SD
#   Spread of the lapse rate BETWEEN people within a group, around the
#   group mean.
#
#   This matters for the zero-catch-failure subsample and for nothing
#   else. If every control lapsed at exactly the group rate, selecting
#   the ones with no catch failures would give a random 20% of them --
#   with the same lapse rate -- and would fix nothing. The subsample
#   helps only because lapse rates DIFFER between people, so a clean
#   catch record is evidence of a low personal rate.
#
#   0.5 x the group mean is a guess, stated as one. The report shows
#   what the subsample achieves as a function of it.
LAPSE_BETWEEN_SD = 0.5

# --- The response-time gate measure ---
#
# RT_GATE
#   A second gate measure, fixed in advance: the per-person share of
#   implausibly fast responses on image trials, compared one-sided
#   with controls worse.
#
#   REGISTERED UNCONDITIONALLY. The task records a response time on
#   every pair trial -- image onset to keypress -- so the measure is
#   available by construction rather than contingent on anything
#   being added to the experiment.
#
# RT_FAST_CUTOFF_MS
#   300 ms. The justification, and its limits:
#
#   A trial shows two images side by side. Deciding between them
#   requires encoding both, which needs at least one saccade: a
#   saccade plus the fixation to programme it is about 200 ms, and
#   the visual evidence for a photometric judgement accumulates over
#   roughly another 100 ms before a motor response can be initiated.
#   So a response below 300 ms cannot have involved comparing the two
#   pictures, whatever else it involved.
#
#   300 ms is therefore an implausibility floor, not an engagement
#   threshold: a genuinely careful participant will be far above it,
#   and a response just above it is not thereby careful. A stricter
#   cut of 400 to 500 ms would catch more inattention and also some
#   fast real responses.
#
#   THE HONEST CAVEAT. This cut-off is reasoned from the literature on
#   saccade latency and evidence accumulation, not measured on this
#   task, and it must be FIXED BEFORE the data arrive or it becomes a
#   researcher degree of freedom. The first participants' RT
#   distribution should be reported against it -- if the mode sits
#   near 300 ms the cut is wrong for this task and the measure should
#   be dropped rather than re-tuned.
RT_GATE = True
RT_FAST_CUTOFF_MS = 300

# RT_FAST_GIVEN_LAPSE
#   The probability that a LAPSED trial produces a response below the
#   cut-off. A lapse is a guess, and a guess needs no comparison, so
#   it can be fast; but a disengaged participant may also be slow
#   (distracted rather than hurried), so not every lapse is fast.
#
#   0.4 is an assumption and is swept in the report. An engaged trial
#   is taken to fall below 300 ms essentially never
#   (RT_FAST_GIVEN_ENGAGED).
RT_FAST_GIVEN_LAPSE = 0.4
RT_FAST_GIVEN_ENGAGED = 0.005
RT_FAST_SWEEP = [0.2, 0.4, 0.6]

# --- When the gate fires ---
#
# GATE_ACTION
#   "inconclusive"  C1 is reported as inconclusive. REGISTERED. The
#                   joint lapse model cannot serve as the fallback
#                   because it is not identifiable on this design, so
#                   there is no second estimate to defer to.
#
# GATE_SECONDARY_ZERO_CATCH
#   A pre-registered SECONDARY analysis: rerun C1 on the participants
#   with zero catch failures in both groups. Pre-registered rather
#   than post-hoc matching, so the subsample is defined by a rule
#   fixed in advance rather than by whichever subset preserves the
#   result. Its sample sizes are reported, because they can be small:
#   a group lapsing at 0.2 keeps only about 20% of its members.
#
# GATE_SIZE_GRID / GATE_STUDIES
#   The gate's own section sweeps group sizes directly, since its
#   behaviour at 10 participants against 30 controls is the case that
#   matters and it is not the case C1's power grid is densest in.
GATE_SIZE_GRID = [(10, 30), (15, 30), (20, 30), (30, 30), (20, 20),
                  (10, 10), (60, 60)]
GATE_STUDIES = 4000

GATE_ACTION = "inconclusive"
GATE_SECONDARY_ZERO_CATCH = True

# --- The joint lapse model: NOT USED, kept for the record ---
#
# FIT_LAPSE_MODEL
#   Fit each person's lapse rate jointly with their two weights, as
#   the pre-registered sensitivity analysis for C1. Firth stays the
#   primary fit; this one exists because a group difference in
#   ATTENTION would otherwise read as a group difference in weighting,
#   and the catch trials only catch gross inattention.
#
#   It needs a recovery check before it can be trusted -- a
#   three-parameter fit from 180 binary trials is not obviously
#   identifiable -- which is why RECOVER_LAPSE below is not optional.
#   DROPPED. The recovery check shows the three parameters are not
#   identifiable from 180 binary trials: the log-likelihood moves by
#   under 0.1 units across lambda from 0 to 0.4, the recovered lambda
#   is essentially a constant near 0.20 whatever the truth, and the
#   weights come back up to 27x as variable as Firth's on the same
#   responses. A group-level lambda shared across 30 people is no
#   better -- everything from 0 to 0.25 sits within 2 log-likelihood
#   units of the best.
#
#   The reason is structural: lapses attenuate the fitted weight by
#   about (1 - lambda), and that attenuation is their only signature
#   in choice data, exactly degenerate with a smaller true weight. A
#   binary 2AFC task with no confidence ratings and no reaction times
#   has no second moment to break the tie.
#
#   Left at True only so the recovery check still runs and the
#   negative result stays in the record. Nothing downstream uses the
#   fit.
FIT_LAPSE_MODEL = True

# LAPSE_FIT_BOUNDS
#   The lapse parameter is bounded away from 0 and 0.5. At exactly 0
#   the gradient vanishes and at 0.5 the weights are unidentifiable,
#   so an unbounded fit wanders.
LAPSE_FIT_BOUNDS = (1e-4, 0.45)

# RECOVER_LAPSE
#   Simulate people with KNOWN weights and lapse rates, fit the joint
#   model, and report how well all three come back. If the lapse
#   parameter does not recover, the sensitivity analysis is not
#   evidence and the report says so.
RECOVER_LAPSE = True
RECOVER_LAPSE_TRUE = [(0.5, 0.0, 0.0), (0.5, 0.0, 0.1),
                      (0.5, 0.0, 0.2), (0.5, 0.5, 0.1),
                      (1.0, 0.0, 0.2), (0.3, 0.0, 0.1),
                      (0.0, 0.0, 0.1)]
N_RECOVER_LAPSE = 4000

# LAPSE_GRID
#   The lapse values the joint fit profiles over. The three-parameter
#   likelihood is fitted by profiling lambda on this grid and solving
#   for the two weights at each value by IRLS, then taking the best.
#
#   A grid rather than a joint Newton because the likelihood in lambda
#   is flat near zero -- a participant who never lapses gives the same
#   fit at lambda 0.001 as at 0.02 -- so a gradient method wanders
#   there while a grid simply reports the best point. The grid is
#   denser at the low end, where the resolution matters.
LAPSE_GRID = [1e-4, 0.01, 0.025, 0.05, 0.075, 0.10, 0.13, 0.16,
              0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

# LAPSE_IRLS_ITER
#   IRLS steps per lambda. The link is not canonical, so this is a
#   quasi-Newton rather than exact Newton; 40 is well past convergence
#   on this design.
LAPSE_IRLS_ITER = 40

# --- Trial-count sweep ---
#
# RUN_TRIAL_COUNT_SWEEP
#   Rebuilds the trial list at several sizes and reports what each
#   size buys.
#
#   RETIRED, NOT DELETED. It existed to inform the choice of trials
#   per type, and that choice is now frozen at 60 per single-feature
#   type with 30 each for together and conflict. There is no decision
#   left for it to inform, and a sweep that rebuilds lists the old way
#   -- quantile bins, legacy sampler -- would not describe the frozen
#   design anyway.
#
#   Set True to run it as a record. If you do, read it as a statement
#   about the OLD sampler: it does not rebuild the per-type streams or
#   the fixed bins.
RUN_TRIAL_COUNT_SWEEP = False

# --- Which registered sections run ---
#
# Each of these is a section of the registration. They are flags so a
# dry run can switch off the slow ones without editing the body, and
# so a reader can see in one place what the registration rests on.
# EVERY ONE OF THEM MUST BE True FOR THE RUN THAT IS REGISTERED; the
# report records which were off.
#
#   RUN_RULE_SWEEP        the existence claim's decision rule: which
#                         rule family, how many detections, both
#                         admissibility standards
#   RUN_EQUAL_BEHAVIOUR   the same rules stated behaviourally, which is
#                         the primary form of the claim
#   RUN_TASTE_RECOVERY    can the control group's taste spread be
#                         estimated well enough to use as the null?
#   RUN_PREVALENCE        Bayesian prevalence, and the variant that
#                         estimates the false-positive rate from the
#                         opposite-direction classifications
#   RUN_LAPSE_RECOVERY    the negative result that killed the joint
#                         lapse model. Kept because the gate's design
#                         depends on it.
#   RUN_GATE              the attention gate's own operating
#                         characteristics
#   RUN_C1                the group comparison, gated, with the
#                         zero-catch-failure secondary
RUN_RULE_SWEEP = True
RUN_EQUAL_BEHAVIOUR = True
RUN_TASTE_RECOVERY = True
RUN_PREVALENCE = True
RUN_LAPSE_RECOVERY = True
RUN_GATE = True
RUN_C1 = True

# RUN_LEGACY_CHECK
#   Re-run the pre-freeze configuration against LEGACY_LIST and
#   compare with LEGACY_OUTPUT. Switch off once that folder is gone.
RUN_LEGACY_CHECK = True

# MAKE_FIGURES
#   Draw the four registration figures. They are drawn LAST, from
#   tables the sections above have already computed, so switching them
#   off costs nothing but the pictures and switching them on never
#   changes a number.
MAKE_FIGURES = True

# FROZEN_TRIAL_LIST
#   Where the sealed list sits inside the package. Step 7a writes it
#   at the package root and again under package/, identical; the root
#   copy is the one read here.
FROZEN_TRIAL_LIST = "trial_list.csv"

# SWEEP_COUNTS
#   Trials per type. Step 5's setting is 40 (40 sat_only + 40 fog_only
#   + 40 mixed = 120 analysed), so 40 is the current design and the
#   others say what moving would cost or buy. The session grows in
#   proportion: 3x this number of analysed trials, plus the practice,
#   catch and consistency extras.
SWEEP_COUNTS = [20, 30, 40, 50, 60, 80, 100]

# SWEEP_SCENE_CAP
#   How many trials of one type a single scene may contribute, as in
#   step 5. None lets it grow with the count -- ceil(count / scenes) --
#   which is what you must do to go past one trial per scene: with 47
#   scenes, a cap of 1 puts a hard ceiling of 47 trials per type.
#   The sweep reports the cap it used at each size.
SWEEP_SCENE_CAP = None

# SWEEP_BETWEEN_SD
#   Assumed spread of the TRUE preference across people, on the
#   proportion scale (a person's share of trials minus 0.5).
#
#   This is the one number here that cannot be measured from your data
#   -- it is a property of the population, not the design -- so the
#   sweep reports a range and labels it as an assumption. It matters
#   because a person's score mixes two things: real differences
#   between people, which more trials cannot touch, and measurement
#   noise, which they can. When most of the spread is real, extra
#   trials buy little at the group level.
#       0.05  people are alike
#       0.10  moderate differences
#       0.20  people differ a lot
SWEEP_BETWEEN_SD = [0.05, 0.10, 0.20]

# SWEEP_TRUE_PROP
#   Assumed average preference, as a share above chance: 0.10 means
#   the average person picks the manipulated image 60% of the time.
SWEEP_TRUE_PROP = 0.10

# SWEEP_N_PEOPLE
#   Simulated participants per size, for the weight floor.
#   A percentile needs more people than an SD does, because it is a
#   tail value: at 4,000, a hundred of them sit above the floor. The
#   batch fit makes this essentially free.
SWEEP_N_PEOPLE = 4000

# N_MAGNITUDE_BINS
#   Used ONLY by the retired trial-count sweep, which rebuilt lists
#   the old way. The live analyses read the bins from the frozen list
#   (DOSE_BINS_FROM_LIST), because step 5's fixed edges no longer give
#   three bins for every type.
N_MAGNITUDE_BINS = 3

RANDOM_SEED = 42

TYPES = ["sat_only", "fog_only", "together", "conflict"]


# ============================================================
# Small helpers
# ============================================================

def resolve_output_root():
    """OUTPUT_ROOT, never overwriting an existing analysis."""
    def occupied(d):
        return os.path.isdir(d) and os.path.isfile(
            os.path.join(d, "power_report.txt"))

    if not occupied(OUTPUT_ROOT):
        return OUTPUT_ROOT
    if ON_EXISTING == "error":
        raise FileExistsError(
            f"{OUTPUT_ROOT} already holds an analysis. Move it, or set "
            "ON_EXISTING = 'new'.")
    n = 2
    while occupied(f"{OUTPUT_ROOT}_{n}"):
        n += 1
    print(f"  NOTE: {OUTPUT_ROOT} already holds an analysis; writing "
          f"to {OUTPUT_ROOT}_{n} instead.")
    return f"{OUTPUT_ROOT}_{n}"


# ============================================================
# Model
# ============================================================

def simulate_responses(X, beta, rng):
    """One synthetic participant's choices.

    X is (n_trials, 2) standardised differences; beta is the true
    (b_sat, b_fog). Responses are probabilistic, not deterministic -- a
    participant with a strong weight still makes occasional 'wrong'
    choices, which is what makes the recovered weight noisy.
    """
    p = 1.0 / (1.0 + np.exp(-(X @ np.asarray(beta, dtype=float))))
    return (rng.random(len(X)) < p).astype(float)


def fit_logit(X, y, n_iter=60, ridge=1e-6):
    """Newton-Raphson logistic regression, no intercept.

    No intercept because the design is symmetric: each pair appears
    with A on the left or right at random, and the predictors are
    signed differences, so there is no baseline preference to absorb.

    The small ridge term keeps the Hessian invertible when a simulated
    participant happens to respond almost deterministically.
    """
    b = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-X @ b))
        W = np.clip(p * (1 - p), 1e-9, None)
        H = (X * W[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


def fit_logit_batch(X, Y, n_iter=60, ridge=1e-6):
    """fit_logit for many simulated people at once.

    Same model, same penalty, same Newton steps -- just solved for
    every participant in one pass, which turns the detection curve from
    tens of thousands of separate fits into a few seconds. Verified
    against fit_logit at start-up.

    X is (n_trials, 2), Y is (n_people, n_trials). Returns (n_people, 2).
    """
    P, k = Y.shape[0], X.shape[1]
    b = np.zeros((P, k))
    eye = ridge * np.eye(k)
    for _ in range(n_iter):
        # Clipped because a simulated participant who responds almost
        # deterministically drives the weights up until exp overflows.
        # The probability is 0 or 1 either side of the clip, so this
        # changes no result -- it only silences the warning.
        p = 1.0 / (1.0 + np.exp(-np.clip(b @ X.T, -500.0, 500.0)))
        w = np.clip(p * (1 - p), 1e-9, None)
        H = np.einsum("pt,ti,tj->pij", w, X, X) + eye
        g = (Y - p) @ X
        step = np.linalg.solve(H, g[..., None])[..., 0]
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


def fit_firth(X, y, n_iter=FIRTH_ITER, tol=FIRTH_TOL,
              max_step=FIRTH_MAX_STEP):
    """Firth's bias-reduced logistic regression, one participant.

    VERBATIM from step 6, where it was written and validated five
    ways. The score equation gains a term from the Jeffreys-prior
    penalty 0.5 * log det(X' W X):

        U*(b)_j = sum_i [ y_i - p_i + h_i (0.5 - p_i) ] x_ij

    with h_i the i-th hat-matrix diagonal. Always finite, so a
    participant whose answers the model predicts perfectly still has
    an estimable weight instead of being excluded.

    Accepts FRACTIONAL y. The cap construction sets y = 0.5 on trials
    where a cue carries no information, and the Bernoulli
    log-likelihood and the weights w = p(1-p) are both well defined
    there -- w does not involve y at all.
    """
    b = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -500.0, 500.0)))
        w = np.clip(p * (1 - p), 1e-12, None)
        Ainv = np.linalg.pinv((X * w[:, None]).T @ X)
        h = w * np.einsum("ti,ij,tj->t", X, Ainv, X)
        step = Ainv @ (X.T @ (y - p + h * (0.5 - p)))
        n = np.max(np.abs(step))
        if n > max_step:
            step = step * (max_step / n)
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    return b


def fit_firth_batch(X, Y, n_iter=FIRTH_ITER, tol=FIRTH_TOL,
                    max_step=FIRTH_MAX_STEP):
    """fit_firth for many people at once, WRITTEN OUT FOR TWO
    PREDICTORS.

    VERBATIM from step 6. The registered model has exactly two
    predictors and the general form spends its time in two einsum
    contractions, which for k = 2 reduce to three matrix-vector
    products and three broadcasts -- twice as fast, which matters at
    20,000 guessers per threshold.

    Also returns each person's own last step, so a fit that never
    settled can be counted rather than averaged in silently.
    """
    if X.shape[1] != 2:
        raise ValueError(
            f"fit_firth_batch is written out for 2 predictors, got "
            f"{X.shape[1]}.")
    x0, x1 = X[:, 0], X[:, 1]
    s00, s01, s11 = x0 * x0, x0 * x1, x1 * x1
    b = np.zeros((Y.shape[0], 2))
    last = np.full(Y.shape[0], np.inf)
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(b @ X.T, -500.0, 500.0)))
        w = np.clip(p * (1 - p), 1e-12, None)
        a00, a01, a11 = w @ s00, w @ s01, w @ s11
        det = a00 * a11 - a01 * a01
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        i00, i01, i11 = a11 / det, -a01 / det, a00 / det
        q = (i00[:, None] * s00[None, :]
             + 2.0 * i01[:, None] * s01[None, :]
             + i11[:, None] * s11[None, :])
        h = w * q
        r = (Y - p) + h * (0.5 - p)
        g0, g1 = r @ x0, r @ x1
        st0 = i00 * g0 + i01 * g1
        st1 = i01 * g0 + i11 * g1
        n = np.maximum(np.abs(st0), np.abs(st1))
        sc = np.where(n > max_step, max_step / np.maximum(n, 1e-30),
                      1.0)
        b[:, 0] += st0 * sc
        b[:, 1] += st1 * sc
        last = n * sc
        if n.max() < tol:
            break
    return b, last


def penalised_loglik(X, y, b):
    """log L(b) + 0.5 log det(X' W X): what Firth maximises.

    VERBATIM from step 6. Used by the solver check and by the cap,
    where it confirms the returned estimate is a root rather than a
    stalled iteration.
    """
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -500.0, 500.0)))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    w = p * (1 - p)
    sign, logdet = np.linalg.slogdet((X * w[:, None]).T @ X)
    return -np.inf if sign <= 0 else ll + 0.5 * logdet


def fit_lapse_batch(X, Y, grid=None, n_iter=LAPSE_IRLS_ITER,
                    bounds=LAPSE_FIT_BOUNDS):
    """The joint per-person lapse model, many people at once.

        p = (1 - lambda) * logistic(X b) + lambda / 2

    Three parameters per person: two weights and a lapse rate.
    Returns (weights, lambda_hat, loglik).

    FITTED BY PROFILING LAMBDA. For each lambda on the grid the two
    weights are solved by IRLS -- the link is not canonical, so the
    working weight is (dp/deta)^2 / (p(1-p)) and the working residual
    is (y - p) / (dp/deta) -- and the lambda with the highest
    log-likelihood wins. A grid rather than a joint gradient method
    because the likelihood in lambda is nearly flat near zero: a
    participant who never lapses fits equally well at 0.001 and 0.02,
    so a gradient wanders there while a grid just reports the best
    point.

    WHY THIS MODEL NEEDS NO FIRTH PENALTY. Under separation the plain
    logistic likelihood is unbounded, which is the whole reason the
    primary fit is Firth. Here it is bounded automatically: p cannot
    exceed 1 - lambda/2, so no weight can drive the likelihood to
    infinity once lambda is above its lower bound. The lapse
    parameter regularises the weights as a side effect of being in
    the model.

    THE COST is that the three parameters trade off against each
    other -- a large weight with a large lapse looks much like a
    smaller weight with no lapse -- which is exactly what the
    recovery check below is for. Do not use this fit without reading
    that output.
    """
    grid = LAPSE_GRID if grid is None else grid
    grid = [g for g in grid if bounds[0] <= g <= bounds[1]]
    P_, k = Y.shape[0], X.shape[1]
    best_ll = np.full(P_, -np.inf)
    best_b = np.zeros((P_, k))
    best_lam = np.zeros(P_)
    x0, x1 = X[:, 0], X[:, 1]
    s00, s01, s11 = x0 * x0, x0 * x1, x1 * x1
    for lam in grid:
        b = np.zeros((P_, k))
        for _ in range(n_iter):
            eta = np.clip(b @ X.T, -500.0, 500.0)
            sig = 1.0 / (1.0 + np.exp(-eta))
            p = (1.0 - lam) * sig + lam / 2.0
            p = np.clip(p, 1e-10, 1 - 1e-10)
            dp = (1.0 - lam) * sig * (1.0 - sig)
            w = np.clip(dp * dp / (p * (1 - p)), 1e-14, None)
            z = (Y - p) / np.where(np.abs(dp) < 1e-14, 1e-14, dp)
            # Weighted normal equations, written out for k = 2.
            a00 = w @ s00
            a01 = w @ s01
            a11 = w @ s11
            det = a00 * a11 - a01 * a01
            det = np.where(np.abs(det) < 1e-300, 1e-300, det)
            r = w * z
            g0, g1 = r @ x0, r @ x1
            st0 = (a11 * g0 - a01 * g1) / det
            st1 = (a00 * g1 - a01 * g0) / det
            n = np.maximum(np.abs(st0), np.abs(st1))
            sc = np.where(n > FIRTH_MAX_STEP,
                          FIRTH_MAX_STEP / np.maximum(n, 1e-30), 1.0)
            b[:, 0] += st0 * sc
            b[:, 1] += st1 * sc
            if n.max() < FIRTH_TOL:
                break
        eta = np.clip(b @ X.T, -500.0, 500.0)
        sig = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip((1.0 - lam) * sig + lam / 2.0, 1e-12, 1 - 1e-12)
        ll = (Y * np.log(p) + (1 - Y) * np.log1p(-p)).sum(axis=1)
        take = ll > best_ll
        best_ll[take] = ll[take]
        best_b[take] = b[take]
        best_lam[take] = lam
    return best_b, best_lam, best_ll


def run_lapse_recovery(X, report, root):
    """Can the joint lapse model recover its own parameters?

    THIS DECIDES WHETHER THE SENSITIVITY ANALYSIS IS EVIDENCE. A
    three-parameter fit from 180 binary responses is not obviously
    identifiable, and the three trade off: a large weight with a large
    lapse rate predicts much the same responses as a smaller weight
    with none. If lambda does not come back, a C1 run on these weights
    is not a check on the plain one -- it is a different and noisier
    estimate dressed as a check.

    Reports, for each true (b_colour, b_fog, lambda): the mean and SD
    of each recovered parameter, and -- the number that matters -- the
    correlation between the recovered weight and the recovered lambda,
    which is the trade-off made visible.
    """
    print("\n" + "=" * 60)
    print("RECOVERY CHECK: THE JOINT LAPSE MODEL")
    print("=" * 60)
    print(f"  {N_RECOVER_LAPSE} simulated people per cell, "
          f"{X.shape[0]} trials each")
    print(f"  lambda profiled over {len(LAPSE_GRID)} grid points in "
          f"[{min(LAPSE_GRID)}, {max(LAPSE_GRID)}]")
    rng = np.random.default_rng(RANDOM_SEED + 909)
    rows = []
    for (bc, bf, lam) in RECOVER_LAPSE_TRUE:
        W = np.zeros((N_RECOVER_LAPSE, 2))
        W[:, 0] = bc
        W[:, 1] = bf
        Y = simulate_lapsing(X, W, lam, rng)
        bb, ll, _ = fit_lapse_batch(X, Y)
        # The primary fit on the same responses, for comparison.
        bf_firth = fit_batch(X, Y)
        rows.append({
            "true_colour": bc, "true_fog": bf, "true_lapse": lam,
            "lapse_mean": round(float(ll.mean()), 4),
            "lapse_sd": round(float(ll.std(ddof=1)), 4),
            "lapse_median": round(float(np.median(ll)), 4),
            "lapse_bias": round(float(ll.mean() - lam), 4),
            "colour_mean": round(float(bb[:, 0].mean()), 4),
            "colour_sd": round(float(bb[:, 0].std(ddof=1)), 4),
            "colour_bias": round(float(bb[:, 0].mean() - bc), 4),
            "fog_mean": round(float(bb[:, 1].mean()), 4),
            "fog_sd": round(float(bb[:, 1].std(ddof=1)), 4),
            "r_colour_lapse": round(float(np.corrcoef(
                bb[:, 0], ll)[0, 1]), 3),
            "firth_colour_mean": round(float(
                bf_firth[:, 0].mean()), 4),
            "firth_colour_sd": round(float(
                bf_firth[:, 0].std(ddof=1)), 4),
            "sd_ratio_lapse_over_firth": round(
                float(bb[:, 0].std(ddof=1)
                      / max(bf_firth[:, 0].std(ddof=1), 1e-12)), 3)})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(root, "lapse_recovery.csv"), index=False)
    print()
    print(tab[["true_colour", "true_fog", "true_lapse", "lapse_mean",
               "lapse_sd", "lapse_bias", "colour_mean", "colour_sd",
               "colour_bias", "r_colour_lapse",
               "sd_ratio_lapse_over_firth"]].to_string(index=False))

    worst_lam = float(tab.lapse_bias.abs().max())
    worst_w = float(tab.colour_bias.abs().max())
    worst_r = float(tab.r_colour_lapse.abs().max())
    infl = float(tab.sd_ratio_lapse_over_firth.max())
    print(f"\n  lambda recovers with a bias of at most "
          f"{worst_lam:+.4f} and an SD of "
          f"{tab.lapse_sd.max():.4f}")
    print(f"  the colour weight recovers with a bias of at most "
          f"{worst_w:+.4f}")
    print(f"  worst correlation between the recovered weight and the "
          f"recovered lambda: {worst_r:+.3f}")
    print(f"  the lapse model's weights are up to {infl:.2f}x as "
          "variable as Firth's on the")
    print("  SAME responses -- which is the price of the third "
          "parameter, and the reason")
    print("  it is a sensitivity analysis rather than the primary "
          "fit.")
    ok = worst_lam <= 0.05 and worst_r <= 0.6
    print(f"\n  -> {'USABLE' if ok else 'NOT USABLE'} as a "
          "sensitivity analysis on these criteria")
    print("     (|lambda bias| <= 0.05 and |r(weight, lambda)| <= "
          "0.6)")
    if not ok:
        print("     The C1 run on these weights below should be read "
              "as a different")
        print("     estimate, NOT as a check on the primary one.")
    report.append(("lapse model recovery",
                   f"{'usable' if ok else 'NOT USABLE'}: lambda bias "
                   f"<= {worst_lam:+.4f}, r(w, lambda) <= "
                   f"{worst_r:+.3f}, SD inflation {infl:.2f}x"))
    return tab, ok


def fit_one(X, y):
    """One participant, under the registered estimator."""
    if ESTIMATOR == "firth":
        return fit_firth(X, y)
    if ESTIMATOR == "mle":
        return fit_logit(X, y)
    raise ValueError(
        f"ESTIMATOR must be 'firth' or 'mle', got {ESTIMATOR!r}")


def fit_batch(X, Y):
    """Many participants, under the registered estimator."""
    if ESTIMATOR == "firth":
        return fit_firth_batch(X, Y)[0]
    if ESTIMATOR == "mle":
        return fit_logit_batch(X, Y)
    raise ValueError(
        f"ESTIMATOR must be 'firth' or 'mle', got {ESTIMATOR!r}")


def simulate_many(X, W, rng):
    """Choices of many people at once. W is (n_people, 2) true weights."""
    p = 1.0 / (1.0 + np.exp(-(W @ X.T)))
    return (rng.random(p.shape) < p).astype(float)


def lapse_p(X, W, lapse):
    """Response probability under a lapse rate.

    (1 - lambda) * logistic(X b) + lambda / 2. At lambda = 0 this is
    the plain model exactly, so the 0.0 row of every lapse table is a
    reference rather than a separate condition.

    Note that a GUESSER is untouched by lambda: with zero weights
    logistic(X @ 0) = 0.5 and (1 - lambda)*0.5 + lambda/2 = 0.5
    identically. So the thresholds do not depend on the lapse rate and
    are computed once. Step 6b proves this by feeding the same uniform
    draws through every lambda and requiring bit-identical floors.
    """
    p = 1.0 / (1.0 + np.exp(-np.clip(W @ X.T, -500.0, 500.0)))
    if not lapse:
        return p
    return (1.0 - lapse) * p + lapse / 2.0


def simulate_lapsing(X, W, lapse, rng):
    """Choices of many people at once, each lapsing at rate `lapse`."""
    p = lapse_p(X, W, lapse)
    return (rng.random(p.shape) < p).astype(float)


def _jzs_g_form(t, n, scale=BF_SCALE):
    """The TWO-SIDED JZS factor in Rouder et al.'s (2009) g form.

    Kept only as an independent check on the effect-size form below.
    The two are different parameterisations of the same quantity: here
    the Cauchy prior is written as a scale mixture of normals and the
    integral runs over the mixing variance g, which is why this form
    cannot be restricted to positive effects -- g is not the effect
    size, and every value of it already covers both signs.
    """
    df = n - 1

    def integrand(g):
        num = (1 + n * g) ** -0.5
        expo = -(df + 1) / 2.0
        term = (1 + t ** 2 / ((1 + n * g) * df)) ** expo
        prior = ((scale ** 2) ** 0.5 / np.sqrt(2 * np.pi)
                 * g ** -1.5 * np.exp(-scale ** 2 / (2 * g)))
        return num * term * prior

    null = (1 + t ** 2 / df) ** (-(df + 1) / 2.0)
    try:
        val, _ = quad(integrand, 0, np.inf, limit=200)
    except Exception:
        return np.nan
    return val / null if null > 0 else np.nan


def jzs_bayes_factor(t, n, scale=BF_SCALE, one_sided=None):
    """JZS Bayes factor for a one-sample t-test (Rouder et al., 2009).

    Returns BF10: how much more likely the data is under an effect than
    under the null. BF10 = 0.25 means the data is four times more
    likely under the null, usually written BF01 = 4.

    WRITTEN OVER THE EFFECT SIZE, not over the prior's scale. For a
    given true effect size delta, the t statistic has a noncentral t
    distribution with n - 1 degrees of freedom and noncentrality
    delta * sqrt(n). The marginal likelihood is that density averaged
    over the prior on delta, and the factor is the ratio of it to the
    null's central t density.

    ONE-SIDED (the registered form, BF_ONE_SIDED = True). The prior is
    restricted to positive effects and renormalised, which for a
    prior symmetric about zero means integrating over [0, inf) and
    doubling. This tests the directional hypothesis the plan actually
    registers:

      - when the data fall the predicted way the factor is up to twice
        the two-sided one, so it is more sensitive to a real effect;
      - when they fall the other way it is smaller than the two-sided
        one, so a group trending backwards cannot accumulate evidence
        "for" the hypothesis. A two-sided factor would happily call
        that evidence of an effect, which is not the claim.

    The two-sided case is the same integral over the whole line, and
    `check_bayes_factor()` verifies it against the g form above.
    """
    one_sided = BF_ONE_SIDED if one_sided is None else one_sided
    df = n - 1

    def integrand(d):
        return (stats.nct.pdf(t, df, d * np.sqrt(n))
                * stats.cauchy.pdf(d, 0.0, scale))

    null = stats.t.pdf(t, df)
    if not (null > 0) or not np.isfinite(null):
        return np.nan
    try:
        if one_sided:
            val, _ = quad(integrand, 0.0, np.inf, limit=200)
            val *= 2.0          # renormalise the half prior
        else:
            lo, _ = quad(integrand, -np.inf, 0.0, limit=200)
            hi, _ = quad(integrand, 0.0, np.inf, limit=200)
            val = lo + hi
    except Exception:
        return np.nan
    return val / null


def check_bayes_factor(report):
    """Does the rewritten factor reproduce the old one, two-sided?

    The two forms are different parameterisations of the same integral,
    so their TWO-SIDED values must agree. If they do, the effect-size
    form is doing what the g form did and the one-sided version is the
    same integral over half the prior rather than a different
    statistic.

    Also checks the two properties the one-sided factor is registered
    for: that it is larger than the two-sided one when t is positive
    and smaller when t is negative, and that it goes the right way as
    the data turn against the hypothesis.
    """
    print("\n" + "=" * 60)
    print("CHECKING THE BAYES FACTOR")
    print("=" * 60)
    rows = []
    worst = 0.0
    for n in (10, 20, 30):
        for t in (-2.5, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
            g = _jzs_g_form(t, n)
            two = jzs_bayes_factor(t, n, one_sided=False)
            one = jzs_bayes_factor(t, n, one_sided=True)
            rel = abs(two - g) / max(abs(g), 1e-12)
            worst = max(worst, rel)
            rows.append({"n": n, "t": t, "g_form": round(g, 6),
                         "two_sided": round(two, 6),
                         "rel_diff": rel,
                         "one_sided": round(one, 6),
                         "ratio_one_over_two": round(one / two, 4)
                         if two > 0 else np.nan})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False,
                        formatters={"rel_diff": "{:.2e}".format}))
    print(f"\n  worst relative difference between the two forms, "
          f"two-sided: {worst:.2e}")
    ok = worst < BF_FORM_TOL
    print(f"  -> {'PASS' if ok else 'FAIL'} "
          f"(tolerance {BF_FORM_TOL:.0e})")
    if not ok:
        raise RuntimeError(
            "The effect-size and g forms of the JZS factor disagree "
            f"by {worst:.2e}.\nOne of them is wrong and every Bayes "
            "number below would be too.")

    pos = tab[tab.t > 0]
    neg = tab[tab.t < 0]
    print(f"\n  with the data in the predicted direction, the "
          f"one-sided factor is")
    print(f"  {pos.ratio_one_over_two.min():.2f} to "
          f"{pos.ratio_one_over_two.max():.2f} times the two-sided "
          f"one (at most 2, approached as the")
    print("  evidence gets stronger);")
    print(f"  with the data against it, "
          f"{neg.ratio_one_over_two.min():.3f} to "
          f"{neg.ratio_one_over_two.max():.3f} times -- so a group "
          f"trending")
    print("  backwards cannot accumulate evidence FOR the hypothesis.")
    assert pos.ratio_one_over_two.min() > 1.0
    assert neg.ratio_one_over_two.max() < 1.0
    report.append(("Bayes factor form",
                   "one-sided" if BF_ONE_SIDED else "two-sided"))
    report.append(("Bayes factor forms agree (two-sided)",
                   f"{worst:.1e} worst relative difference"))
    return tab


# ============================================================
# Loading
# ============================================================

def sha256_of(path, chunk=1 << 20):
    """SHA-256 of one file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_frozen(root, report):
    """Re-hash the sealed package against its own checksums.sha256.

    REFUSES TO CONTINUE on any mismatch, missing file or extra file.
    Every registered number below is a property of one specific trial
    list; computed from a package that has drifted they would still
    look like evidence, and nothing downstream would notice. This is
    the only check in the pipeline that can catch a file changing
    AFTER it was frozen.

    The format is the one `shasum -a 256 -c` reads, so the same
    verification can be done without this script.
    """
    print("\n" + "=" * 60)
    print("VERIFYING THE FROZEN PACKAGE")
    print("=" * 60)
    man = os.path.join(root, "checksums.sha256")
    if not os.path.isfile(man):
        raise FileNotFoundError(
            f"{man} not found. Step 8 runs on the SEALED list; run "
            "step 7a\n(freeze_trial_list_v4.py) first, or set "
            "VERIFY_CHECKSUMS = False for a dry run\non an unfrozen "
            "build and accept that the output is not registrable.")
    want = {}
    for line in open(man):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        # "<hex>  <path>" -- two spaces, and the path may contain
        # spaces, so split once from the left.
        digest, _, rel = line.partition("  ")
        want[rel.strip()] = digest.strip()

    have = set()
    for dirpath, _, names in os.walk(root):
        for nm in names:
            full = os.path.join(dirpath, nm)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel in UNHASHED:
                continue
            have.add(rel)

    # THE THREE UNHASHED FILES. Step 7a excludes these from the
    # manifest by construction -- it cannot hash the manifest while
    # writing it, and the other two are written afterwards. They are
    # checked for PRESENCE here, and FROZEN.json is then cross-checked
    # against both the manifest and this script's own constants, which
    # is a stronger test than treating them as unexpected files.
    absent = [f for f in UNHASHED
              if not os.path.isfile(os.path.join(root, f))]

    missing = sorted(set(want) - have)
    extra = sorted(have - set(want))
    if absent:
        missing = missing + [f"{f} (not in the manifest by design, "
                             "but the package must contain it)"
                             for f in absent]
    bad = []
    for rel in sorted(set(want) & have):
        if sha256_of(os.path.join(root, rel)) != want[rel]:
            bad.append(rel)

    print(f"  {len(want)} files listed, {len(have)} present "
          f"(+{len(UNHASHED) - len(absent)} unhashed by design)")
    ok = not (missing or extra or bad)
    if ok:
        print("  every file matches its recorded hash")
    # --- FROZEN.json against the manifest and against this script ---
    facts, mismatch = {}, []
    jpath = os.path.join(root, "FROZEN.json")
    if ok and os.path.isfile(jpath):
        with open(jpath) as fh:
            facts = json.load(fh)

        def agree(key, mine, label, exact=True):
            if key not in facts:
                mismatch.append(f"FROZEN.json has no {key!r}")
                return
            theirs = facts[key]
            same = (theirs == mine if exact
                    else np.isclose(float(theirs), float(mine)))
            if not same:
                mismatch.append(
                    f"{label}: the package says {theirs!r}, this "
                    f"script assumes {mine!r}")

        agree("n_files", len(want), "number of checksummed files")
        agree("seed", FROZEN_SEED, "the frozen draw's seed")
        agree("catch_trials", N_CATCH_TRIALS,
              "length of the catch block")
        agree("sd_sat", SD_SAT, "SD_SAT", exact=False)
        agree("sd_fog", SD_FOG, "SD_FOG", exact=False)
        if facts.get("checks_failed", 0):
            mismatch.append(
                f"the freeze itself recorded "
                f"{facts['checks_failed']} FAILED check(s); nothing "
                "computed from this package is registrable")
        print(f"  FROZEN.json: version {facts.get('version')!r}, "
              f"seed {facts.get('seed')}, "
              f"{facts.get('n_images')} images, "
              f"catch_mode {facts.get('catch_mode')!r}, "
              f"{facts.get('catch_trials')} catch trials")
        if mismatch:
            ok = False
            print(f"  FROZEN.json DISAGREES with this script "
                  f"({len(mismatch)}):")
            for m in mismatch:
                print(f"    - {m}")
        else:
            print("  FROZEN.json agrees with the manifest and with "
                  "this script's constants")

    report.append(("frozen package verified",
                   f"{'PASS' if ok else 'FAIL'} ({len(want)} files"
                   + (f", seed {facts['seed']}" if "seed" in facts
                      else "") + ")"))
    if facts:
        report.append(("frozen package",
                       f"version {facts.get('version')}, "
                       f"{facts.get('n_images')} images, "
                       f"catch_mode {facts.get('catch_mode')}, "
                       f"{facts.get('catch_trials')} catch trials, "
                       f"{facts.get('checks_passed')} freeze checks "
                       "passed"))
    if not ok:
        for nm, lst in (("MISSING", missing), ("CHANGED", bad),
                        ("NOT IN THE MANIFEST", extra),
                        ("FROZEN.json DISAGREES", mismatch)):
            if lst:
                print(f"  {nm} ({len(lst)}): "
                      f"{', '.join(lst[:8])}"
                      + (" ..." if len(lst) > 8 else ""))
        raise RuntimeError(
            f"{root} does not match its own record: "
            f"{len(missing)} missing, {len(bad)} changed, "
            f"{len(extra)} unlisted, {len(mismatch)} "
            "disagreement(s) with FROZEN.json.\nNothing below would "
            "be registrable. Re-freeze, or find out what moved.\n"
            "\nNote: checksums.sha256, FROZEN.json and "
            "freeze_report.txt are NOT in the\nmanifest and are not "
            "expected to be -- step 7a cannot hash the manifest\n"
            "while writing it, and writes the other two afterwards. "
            "They are checked\nfor presence, and FROZEN.json is "
            "cross-checked against the manifest and\nagainst this "
            "script's own constants.")
    return facts


def load_trials(path, label):
    """The analysed trials and the two REGISTERED predictors."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Trial list not found: {path}\nRun step 5 "
            "(build_trials_v4.py) first.")
    trials = pd.read_csv(path)
    for c in ("d_sat", "d_fog"):
        if c not in trials.columns:
            raise ValueError(f"{path} has no {c!r} column.")
    # `analysed` is True only for the four real trial types. Practice,
    # catch and consistency trials are quality checks and are excluded
    # from every analysis here.
    if "analysed" in trials.columns:
        trials = trials[trials.analysed.astype(bool)]
    trials = trials.reset_index(drop=True)
    if not len(trials):
        raise ValueError(f"{path} has no analysed trials.")

    # THE REGISTERED PREDICTORS, TAKEN AS THEY ARE.
    #
    # pred_sat and pred_fog are already signed and already in
    # corpus-SD units, oriented so a positive weight means "prefers
    # the predicted image": the duller one for colour, the foggier one
    # for fog.
    #
    # The earlier version of this function built them from
    # -sign(d_sat) * |z_sat|, because z_sat and z_fog are MAGNITUDES
    # whose normaliser is each scene's own range. That made a weight
    # mean a different physical difference in every scene. It is gone.
    #
    # The rebuild below is a CHECK, not the source: pred_sat should be
    # -d_sat/SD_SAT and pred_fog should be +d_fog/SD_FOG. Only d_sat is
    # negated -- d_fog already carries step 4's fog sign convention. If
    # they disagree, a sign or a unit changed upstream and every number
    # in this script would be wrong in a way no later check would
    # catch, so it stops.
    check = np.column_stack([
        -trials.d_sat.to_numpy(float) / SD_SAT,
        trials.d_fog.to_numpy(float) / SD_FOG,
    ])
    if {PRED_SAT, PRED_FOG} <= set(trials.columns):
        X = np.column_stack([trials[PRED_SAT].to_numpy(float),
                             trials[PRED_FOG].to_numpy(float)])
        worst = float(np.abs(X - check).max())
        if worst > PRED_REBUILD_TOL:
            raise RuntimeError(
                f"{path}: {PRED_SAT} / {PRED_FOG} disagree with the "
                f"rebuild from\nd_sat / d_fog by {worst:.2e}, above "
                f"PRED_REBUILD_TOL = {PRED_REBUILD_TOL:.0e}. A sign "
                "convention or a unit\nhas changed upstream. Every "
                "weight below would be wrong.")
        print(f"  [{label}] predictors agree with the rebuild to "
              f"{worst:.1e}")
    else:
        # The legacy list predates these columns. It is read only by
        # the reproduction check, and the rebuild IS the predictor
        # there -- stated rather than silent.
        X = check
        print(f"  [{label}] no {PRED_SAT} column; predictors rebuilt "
              "from d_sat / d_fog")
    r = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    print(f"  [{label}] {len(trials)} analysed trials, "
          f"r(predictors) = {r:+.3f}, VIF = {1 / (1 - r ** 2):.2f}")
    return trials, X, r


# ============================================================
# The floors  (depend on the trial list)
# ============================================================

def informative_medians(X):
    """Each feature's median |difference| among the trials that carry
    information about it.

    This is the anchor the report translates weights into choice
    probabilities at. It comes from the frozen list rather than from a
    round number, because a round number would describe a trial the
    list may not contain: the colour cut removed every colour
    difference below 0.79 SD from the colour-only block, so the
    "typical colour trial" on this list is much larger than on an
    uncut one, and a probability quoted at 1.0 SD would understate
    what a participant actually faces.

    1.0 SD is reported alongside it, as the anchor that stays
    comparable if the list ever changes.
    """
    out = {}
    for k, feat in enumerate(("colour", "fog")):
        a = np.abs(X[:, k])
        info = a >= CAP_INFORMATIVE_SD
        out[feat] = float(np.median(a[info])) if info.any() else np.nan
        out[feat + "_n"] = int(info.sum())
    return out


def as_probability(weight, difference):
    """A weight, as the probability of choosing the predicted image on
    a trial of that difference.

    The readable form. `weight` is per corpus SD and `difference` is
    in SD, so their product is the log-odds.
    """
    return float(1.0 / (1.0 + np.exp(-weight * difference)))


def compute_floors(X, label, report):
    """The per-feature thresholds, as PERCENTILES of the guessers.

    A threshold is the weight a person has to beat before they can be
    told apart from someone answering at random. Simulate people who
    track nothing, fit them, and take the percentiles their estimates
    reach by luck.

    TWO-SIDED, FROM THE SAME GUESSERS:

        upper   the 1 - FLOOR_ALPHA percentile   (97.5th)
        lower   the     FLOOR_ALPHA percentile   ( 2.5th)

    The UPPER one is the same quantity steps 6 and 6b used, computed
    the same way from the same number of guessers, so those results
    stand and did not need re-running. The LOWER one is new. It exists
    because "this person does not track colour" and "this person
    tracks colour BACKWARDS" are different findings, and a decision
    rule that cannot tell them apart cannot state a false-claim rate.

    One guesser sample gives both. Drawing them separately would make
    the two thresholds independently noisy for no gain -- and a
    guesser is unaffected by the lapse rate (lapse_p returns exactly
    0.5 at zero weights, whatever lambda is), so this is computed
    once and shared across every lapse condition. Step 6b tests that
    claim rather than asserting it.

    Taken as percentiles rather than as 1.96 * SD, which is what
    versions before v4 used. The two agree only if the fitted weights
    are normally distributed, and the colour weight is not: its null
    distribution has a longer upper tail, so 1.96 * SD came out about
    11% BELOW the real 97.5th percentile and the test would have run
    at roughly 4% false alarms instead of 2.5%. The percentile needs
    no distributional assumption. It also matches step 9, which must
    use the same thresholds or its responder counts describe a
    different test.

    WORTH KNOWING: under Firth that skew has largely gone. On the
    frozen list the colour threshold sits within a fraction of a per
    cent of 1.96 * SD, and the two tails are near mirror images, where
    under maximum likelihood the upper tail ran about 11% long. The
    bias correction is doing that -- it pulls in exactly the extreme
    estimates that made the tail heavy. The percentile is still what
    is used, because nothing depends on the null staying symmetric and
    the next list need not.
    """
    rng = np.random.default_rng(RANDOM_SEED + 101)
    W = np.zeros((N_FLOOR_PEOPLE, 2))
    est = fit_batch(X, simulate_many(X, W, rng))
    upper = np.percentile(est, 100 * (1 - FLOOR_ALPHA), axis=0)
    lower = np.percentile(est, 100 * FLOOR_ALPHA, axis=0)
    sd = est.std(axis=0)
    print("\n" + "=" * 60)
    print(f"THRESHOLDS  [{label}]")
    print("=" * 60)
    print(f"  {N_FLOOR_PEOPLE:,} simulated guessers, {ESTIMATOR}, "
          f"{FLOOR_ALPHA:.1%} per tail")
    print(f"  upper   : colour {upper[0]:+.3f}, fog {upper[1]:+.3f}"
          "   <- predicted direction")
    print(f"  lower   : colour {lower[0]:+.3f}, fog {lower[1]:+.3f}"
          "   <- opposite direction")
    if DETECTION_SIDED == "one":
        print("  DETECTION_SIDED = 'one': the lower threshold is "
              "computed but not used.")
    print(f"  1.96*SD : colour {1.96 * sd[0]:.3f}, "
          f"fog {1.96 * sd[1]:.3f}   <- the normal approximation")
    for k, feat in enumerate(["colour", "fog"]):
        gap = upper[k] / (1.96 * sd[k]) - 1.0
        if abs(gap) > 0.05:
            print(f"    {feat}: the percentile sits {gap:+.0%} from "
                  "the normal approximation, so that")
            print("      distribution is noticeably skewed; the "
                  "percentile is the one to trust.")
        # Asymmetry between the two tails is the same message. A
        # symmetric null would put them equidistant from zero.
        asym = abs(upper[k]) / abs(lower[k]) - 1.0
        if abs(asym) > 0.05:
            print(f"    {feat}: the upper tail is {asym:+.0%} further "
                  "from zero than the lower,")
            print("      so the null is skewed and the two directions "
                  "are NOT mirror images.")
    report.append(("threshold upper, colour / fog",
                   f"{upper[0]:.3f} / {upper[1]:.3f}"))
    report.append(("threshold lower, colour / fog",
                   f"{lower[0]:.3f} / {lower[1]:.3f}"))
    return {"upper": upper, "lower": lower, "sd": sd,
            "null": est}


def classify(est, th):
    """Each person, per feature: predicted / opposite / not detected.

    Returns two boolean arrays of the same shape as `est`. Under
    DETECTION_SIDED = "one" nothing is ever classified opposite, which
    is what the old behaviour amounted to: a weight far below zero and
    a weight at zero were the same verdict.

    CODED AS TWO MASKS rather than one integer label because every
    consumer wants counts of each, and `predicted.sum()` reads better
    than `(label == 1).sum()`.
    """
    predicted = est > th["upper"]
    if DETECTION_SIDED == "two":
        opposite = est < th["lower"]
    else:
        opposite = np.zeros_like(predicted)
    return predicted, opposite


def check_threshold_calibration(X, th, report):
    """CHECK: a fresh set of guessers is classified at the nominal rate.

    The thresholds come from one guesser sample, so by construction
    exactly FLOOR_ALPHA of THAT sample sits outside each. That is not
    a test of anything. This draws INDEPENDENT guessers and asks how
    often they are classified -- which is the quantity the decision
    rule's false-claim rate is built on, and the one that would be
    wrong if the threshold were fitted to noise.

    Expected to sit at FLOOR_ALPHA within sampling error. The
    standard error of a proportion p from n people is
    sqrt(p(1-p)/n), which at 2.5% and 20,000 people is about 0.0011,
    so a deviation beyond about 0.003 is real and worth knowing.
    """
    print("\n" + "=" * 60)
    print("CHECK: ARE THE THRESHOLDS CALIBRATED?")
    print("=" * 60)
    rng = np.random.default_rng(RANDOM_SEED + 10101)
    W = np.zeros((N_FLOOR_PEOPLE, 2))
    est = fit_batch(X, simulate_many(X, W, rng))
    pred, opp = classify(est, th)
    se = np.sqrt(FLOOR_ALPHA * (1 - FLOOR_ALPHA) / N_FLOOR_PEOPLE)
    tol = max(0.003, 3 * se)
    rows, ok = [], True
    for k, feat in enumerate(["colour", "fog"]):
        for nm, m in (("predicted", pred[:, k]), ("opposite", opp[:, k])):
            rate = float(m.mean())
            want = (FLOOR_ALPHA if (nm == "predicted"
                                    or DETECTION_SIDED == "two")
                    else 0.0)
            good = abs(rate - want) <= tol
            ok = ok and good
            rows.append({"feature": feat, "verdict": nm,
                         "rate": round(rate, 5),
                         "nominal": want,
                         "difference": round(rate - want, 5),
                         "within_tolerance": good})
    tab = pd.DataFrame(rows)
    print(f"  an INDEPENDENT set of {N_FLOOR_PEOPLE:,} guessers, "
          f"classified against the thresholds")
    print(f"  (the set the thresholds came from would sit at "
          f"{FLOOR_ALPHA:.1%} by construction)")
    print()
    print(tab.to_string(index=False))
    print(f"\n  tolerance {tol:.4f} = 3 standard errors of a "
          f"{FLOOR_ALPHA:.1%} proportion at this n")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    report.append(("threshold calibration",
                   f"{'PASS' if ok else 'FAIL'} "
                   f"(worst |rate - nominal| "
                   f"{tab.difference.abs().max():.4f})"))
    if not ok:
        raise RuntimeError(
            "The thresholds do not classify fresh guessers at the "
            "nominal rate.\nEvery false-claim rate below is built on "
            "them, so they have to be right first.")
    return tab


# ============================================================
# Recovery  (depends on the trial list)
# ============================================================

def run_recovery(X, th, label, out_dir, report):
    """Can the analysis tell the scenarios apart?

    Fits each simulated participant and reports both the mean recovered
    weight and its SPREAD. The spread is what a real participant's
    estimate will carry, so it sets the smallest individual effect that
    could be distinguished from zero.
    """
    print("\n" + "=" * 60)
    print(f"PARAMETER RECOVERY  [{label}]")
    print("=" * 60)
    print(f"  {len(X)} analysed trials, {N_SIM_PARTICIPANTS} simulated "
          "participants per scenario")

    rng = np.random.default_rng(RANDOM_SEED)
    rows, clouds = [], {}
    for lab, beta in SCENARIOS.items():
        est = np.array([fit_logit(X, simulate_responses(X, beta, rng))
                        for _ in range(N_SIM_PARTICIPANTS)])
        clouds[lab] = est
        rows.append({
            "scenario": lab,
            "true_b_sat": beta[0], "true_b_fog": beta[1],
            "mean_b_sat": est[:, 0].mean(), "sd_b_sat": est[:, 0].std(),
            "mean_b_fog": est[:, 1].mean(), "sd_b_fog": est[:, 1].std(),
        })
        print(f"\n  {lab}  (true {beta})")
        print(f"    b_sat {est[:, 0].mean():+.2f} "
              f"(SD {est[:, 0].std():.2f})")
        print(f"    b_fog {est[:, 1].mean():+.2f} "
              f"(SD {est[:, 1].std():.2f})")

    rec = pd.DataFrame(rows)
    rec.to_csv(os.path.join(out_dir, "recovery.csv"), index=False)

    # Classification: given a participant who really tracks colour, does
    # the fit put b_sat above b_fog? And vice versa. Overlap between
    # these two clouds is what would make a real participant ambiguous.
    a, b = clouds["tracks colour"], clouds["tracks fog"]
    acc = 0.5 * (float((a[:, 0] > a[:, 1]).mean())
                 + float((b[:, 1] > b[:, 0]).mean()))
    print(f"\n  colour-vs-fog classification accuracy: {acc:.1%}")

    # Smallest individual weight distinguishable from zero. The per-
    # feature floors are the real thresholds; this is their average,
    # kept as a single headline number and as the quantity the trial-
    # count sweep tracks.
    sd0 = clouds["tracks neither"].std(axis=0).mean()
    # The UPPER thresholds: the smallest detectable weight is about
    # the predicted-direction threshold, which is what "detectable"
    # has always meant here. The lower one bounds the opposite
    # direction and does not enter a minimum DETECTABLE effect.
    mde = float(np.mean(th["upper"]))
    print(f"  weight SD under the null: {sd0:.3f}")
    print(f"  smallest individual weight distinguishable from zero: "
          f"~{mde:.2f}")
    print("    A participant whose true weight is below this cannot be "
          "told apart from one who tracks nothing, however clean their "
          "data.")

    report.append(("classification accuracy", f"{acc:.1%}"))
    report.append(("individual weight SD", f"{sd0:.3f}"))
    report.append(("min detectable weight", f"{mde:.2f}"))

    _plot_recovery(clouds, label, out_dir)
    return rec, clouds, acc, sd0, mde


def _plot_recovery(clouds, label, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for (lab, est), col in zip(clouds.items(), ["C0", "C1", "C2", "C7"]):
        ax.scatter(est[:, 0], est[:, 1], s=7, alpha=0.3, color=col,
                   label=lab)
        ax.scatter(est[:, 0].mean(), est[:, 1].mean(), s=140,
                   marker="X", color=col, edgecolor="k", zorder=5)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("recovered b_sat")
    ax.set_ylabel("recovered b_fog")
    ax.set_title(f"Per-participant weight recovery  [{label}]\n"
                 "(each dot is a simulated participant; "
                 "overlap = ambiguity)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "recovery.png"), dpi=150)
    plt.close(fig)


# ============================================================
# Per-person detection  (depends on the trial list)
# ============================================================

def run_detection(X, th, label, out_dir, report):
    """How strong must one person's preference be to be caught
    reliably?

    The floor from the recovery section is the point where a person
    can JUST be told apart from a guesser -- at exactly the floor you
    would catch them about half the time, which is no use for planning.
    This finds the weight at which you would catch them
    DETECTION_TARGET of the time.

    Method, which is the per-person analysis run forwards:
      1. The floor is fixed first, per feature, by compute_floors:
         the weight that FLOOR_ALPHA of pure guessers exceed by luck.
      2. For each candidate true weight, simulate people who really do
         have that weight, fit each, and count how many land above the
         floor.
      3. Interpolate to find where that count crosses the target.

    Done per feature, because the two do not have the same floor: the
    colour weight is estimated less precisely than the fog weight, so
    a colour preference has to be stronger to be caught.
    """
    print("\n" + "=" * 60)
    print(f"PER-PERSON DETECTION  [{label}]")
    print("=" * 60)

    # The floors arrive from compute_floors, which located them as
    # percentiles of a large simulated null rather than as a multiple
    # of its SD. Recomputing them here from a smaller cloud would make
    # this section disagree with the one above.
    print(f"  threshold ({FLOOR_ALPHA:.1%} per tail): "
          f"colour {th['upper'][0]:.2f}, "
          f"fog {th['upper'][1]:.2f}")
    med = informative_medians(X)
    print(f"  median informative difference: "
          f"colour {med['colour']:.2f} SD over {med['colour_n']} "
          f"trials, fog {med['fog']:.2f} SD over {med['fog_n']}")
    print(f"  lapse rates: {LAPSE_RATES}")

    rng = np.random.default_rng(RANDOM_SEED + 5)
    rows = []
    for lam in LAPSE_RATES:
      for w in DETECTION_GRID:
        out = {"lapse": lam, "true_weight": w}
        for f_idx, feat in enumerate(["colour", "fog"]):
            W = np.zeros((N_DETECT_PEOPLE, 2))
            W[:, f_idx] = w
            est = fit_batch(X, simulate_lapsing(X, W, lam, rng))
            wrong = np.zeros(len(est), dtype=bool)
            if DETECTION_SIDED == "one":
                hit = est[:, f_idx] > th["upper"][f_idx]
            else:
                # DETECTION always means the PREDICTED direction,
                # under either setting, so the curves stay comparable
                # with steps 6 and 6b. The two-sided setting adds the
                # OPPOSITE count beside it rather than folding both
                # into one "detected" number -- |weight| > threshold
                # would count a backwards responder as a success,
                # which is the one thing the hypothesis does not
                # predict.
                hit = est[:, f_idx] > th["upper"][f_idx]
                wrong = est[:, f_idx] < th["lower"][f_idx]
            out[feat] = float(hit.mean())
            out[feat + "_opposite"] = float(wrong.mean())
            # The readable form, at this list's own typical trial.
            out[feat + "_p_median"] = as_probability(w, med[feat])
            out[feat + "_p_1sd"] = as_probability(w, 1.0)
        rows.append(out)
    det = pd.DataFrame(rows)
    det.to_csv(os.path.join(out_dir, "detection_curve.csv"),
               index=False)

    print("\n  share of people caught in the PREDICTED direction, "
          "by true weight and lapse rate:")
    print(det.pivot(index="true_weight", columns="lapse",
                    values=["colour", "fog"]).round(3).to_string())
    if DETECTION_SIDED == "two":
        print("\n  share caught in the OPPOSITE direction (should "
              "fall as the true weight rises:")
        print("  a real preference cannot also look backwards):")
        print(det.pivot(index="true_weight", columns="lapse",
                        values=["colour_opposite", "fog_opposite"])
              .round(4).to_string())

    # Where does each curve cross the target? Per lapse rate, since
    # lapses attenuate the fitted weight and so move the crossing.
    needed = {}
    for lam in LAPSE_RATES:
        g = det[np.isclose(det.lapse, lam)]
        for feat in ["colour", "fog"]:
            y = g[feat].to_numpy()
            x = g.true_weight.to_numpy()
            if y.max() < DETECTION_TARGET:
                needed[(feat, lam)] = np.nan
                print(f"  {feat} at lapse {lam}: never reaches "
                      f"{DETECTION_TARGET:.0%} within the grid; "
                      "widen DETECTION_GRID.")
                continue
            needed[(feat, lam)] = float(
                np.interp(DETECTION_TARGET, y, x))
    # The no-lapse pair is what the rest of the script consumes.
    needed["colour"] = needed.get(("colour", 0.0), np.nan)
    needed["fog"] = needed.get(("fog", 0.0), np.nan)

    print(f"\n  weight needed for {DETECTION_TARGET:.0%} detection, "
          "and what it means on a typical trial:")
    for lam in LAPSE_RATES:
        for feat in ["colour", "fog"]:
            v = needed[(feat, lam)]
            if np.isfinite(v):
                print(f"    lapse {lam}  {feat:7s} {v:.2f} per SD   "
                      f"= picking the predicted image "
                      f"{as_probability(v, med[feat]):.0%} of the time "
                      f"on a typical {feat} trial "
                      f"({med[feat]:.2f} SD), "
                      f"{as_probability(v, 1.0):.0%} at 1.0 SD")
    report.append((f"weight for {DETECTION_TARGET:.0%} detection, "
                   "colour / fog (lapse 0)",
                   f"{needed['colour']:.2f} / {needed['fog']:.2f}"))
    if len(LAPSE_RATES) > 1:
        hi = max(LAPSE_RATES)
        report.append((f"the same at lapse {hi}",
                       f"{needed[('colour', hi)]:.2f} / "
                       f"{needed[('fog', hi)]:.2f}"))

    _plot_detection(det, th, needed, label, out_dir)
    return det, needed


def _plot_detection(det, th, needed, label, out_dir):
    # The no-lapse curve, which is the one the thresholds were built
    # for. The lapse curves are in detection_curve.csv.
    if "lapse" in det.columns:
        det = det[np.isclose(det.lapse, 0.0)]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for k, feat in enumerate(["colour", "fog"]):
        ax.plot(det.true_weight, det[feat], marker="o", color=f"C{k}",
                label=f"{feat} (threshold "
                      f"{th['upper'][k]:.2f})")
        v = needed.get(feat, np.nan)
        if np.isfinite(v):
            ax.axvline(v, color=f"C{k}", lw=0.8, ls=":")
    ax.axhline(DETECTION_TARGET, color="k", lw=0.8, ls="--",
               label=f"{DETECTION_TARGET:.0%} target")
    ax.axhline(0.5, color="C7", lw=0.6, ls=":",
               label="coin flip (the floor itself)")
    ax.set_xlabel("the person's TRUE weight")
    ax.set_ylabel("share of such people correctly caught")
    ax.set_ylim(0, 1)
    ax.set_title(f"How strong must one person's preference be?  "
                 f"[{label}]\n(dotted verticals: the weight needed for "
                 f"{DETECTION_TARGET:.0%} detection)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "detection_curve.png"), dpi=150)
    plt.close(fig)



# ============================================================
# The decision rule  (depends on the trial list)
# ============================================================

def classification_rates(X, th, report):
    """How often each kind of non-responder is classified, per feature.

    THREE NULLS, not one:

      guessers       weights exactly zero. The null the thresholds
                     were built from, so its rate is FLOOR_ALPHA by
                     construction -- 2.5% per tail. Included because
                     the rule arithmetic needs it and because the
                     calibration check should agree with it.

      taste          weights drawn from Normal(0, spread), spread set
                     to a multiple of THAT FEATURE'S OWN upper
                     threshold, independently for the two features.
                     Someone with no memory of derealization can still
                     prefer duller pictures, and that preference is
                     real and stable and will clear a threshold built
                     from coin-flippers.

      Taste rates are SYMMETRIC in expectation -- the spread is
      zero-centred, so as many of these people should be classified
      opposite as predicted. That symmetry is the thing to look at:
      a rule that demands few opposite classifications is what
      separates real responders from people with taste, and these
      rows are what makes the separation quantitative.
    """
    print("\n" + "=" * 60)
    print("WHO GETS CLASSIFIED, AND HOW OFTEN")
    print("=" * 60)
    rng = np.random.default_rng(RANDOM_SEED + 777)
    rows = []

    W = np.zeros((N_TASTE_PEOPLE, 2))
    est = fit_batch(X, simulate_many(X, W, rng))
    pred, opp = classify(est, th)
    for k, feat in enumerate(("colour", "fog")):
        rows.append({"null": "guesser", "spread_in_floors": 0.0,
                     "feature": feat, "spread_abs": 0.0,
                     "p_predicted": float(pred[:, k].mean()),
                     "p_opposite": float(opp[:, k].mean())})

    for mult in TASTE_SPREADS_IN_FLOORS:
        # Each feature's spread scales with ITS OWN threshold, so the
        # multiplier means the same thing for colour and for fog.
        sd = th["upper"] * mult
        W = rng.normal(0.0, 1.0, size=(N_TASTE_PEOPLE, 2)) * sd
        est = fit_batch(X, simulate_many(X, W, rng))
        pred, opp = classify(est, th)
        for k, feat in enumerate(("colour", "fog")):
            rows.append({"null": "taste", "spread_in_floors": mult,
                         "feature": feat,
                         "spread_abs": float(sd[k]),
                         "p_predicted": float(pred[:, k].mean()),
                         "p_opposite": float(opp[:, k].mean())})

    tab = pd.DataFrame(rows)
    # THE ASYMMETRY IS A DIAGNOSTIC, NOT A FINDING. Guessers and taste
    # non-responders are both symmetric BY CONSTRUCTION -- a guesser
    # answers at 0.5 and a taste preference is drawn zero-centred, so
    # the two directions are exchangeable. Any difference between the
    # measured p_predicted and p_opposite is estimation noise, and it
    # should shrink like 1 / sqrt(N_TASTE_PEOPLE).
    tab["asymmetry"] = tab.p_predicted - tab.p_opposite
    tab["asym_se"] = np.sqrt(
        2.0 * tab.p_predicted.clip(1e-9) *
        (1 - tab.p_predicted.clip(upper=1 - 1e-9)) / N_TASTE_PEOPLE)
    tab["asym_in_se"] = (tab.asymmetry / tab.asym_se).round(2)
    print(f"  {N_TASTE_PEOPLE:,} simulated people per null")
    print(tab.round(5).to_string(index=False))
    worst_asym = float(tab.asym_in_se.abs().max())
    print(f"\n  the two directions are exchangeable in all of these "
          "by construction, so")
    print(f"  `asymmetry` is pure estimation noise: worst "
          f"{worst_asym:.1f} standard errors.")
    print("  The SIGN TEST BELOW TESTS FOR ASYMMETRY, so it would "
          "turn that noise into")
    print("  false-claim rate. The symmetric nulls are therefore "
          "SYMMETRISED before the")
    print("  rule sweep -- both rates set to their average -- which "
          "uses what is known")
    print("  about the construction instead of re-estimating it. The "
          "count families")
    print("  are barely affected either way; the sign test is not.")
    report.append(("symmetric nulls, worst asymmetry",
                   f"{worst_asym:.1f} SE (noise; symmetrised for the "
                   "sweep)"))
    worst = float(tab[tab.null == "taste"].p_predicted.max())
    print(f"\n  the worst taste null is classified PREDICTED "
          f"{worst:.1%} of the time, against")
    print(f"  the {FLOOR_ALPHA:.1%} a guesser manages. A per-person "
          "threshold cannot tell those")
    print("  apart; only the rule's opposite-direction clause can.")
    report.append(("taste null, worst predicted rate", f"{worst:.3f}"))
    return tab


def trinomial_grid(n, p_pred, p_opp):
    """P[a, b] for a people predicted and b opposite out of n.

    An (n+1) x (n+1) array, zero where a + b > n. Built once per
    group and then convolved, which is what makes every rule in the
    sweep an indexing operation on one shared object rather than its
    own enumeration. The earlier version re-enumerated per rule and
    spent most of its time recomputing the same distribution.
    """
    from math import lgamma
    ln = lambda x: lgamma(x + 1.0)
    rest = 1.0 - p_pred - p_opp
    if rest < -1e-12:
        raise ValueError(
            f"p_predicted + p_opposite = {p_pred + p_opp} > 1")
    rest = max(rest, 0.0)
    out = np.zeros((n + 1, n + 1))
    lp_pred = np.log(p_pred) if p_pred > 0 else -np.inf
    lp_opp = np.log(p_opp) if p_opp > 0 else -np.inf
    lp_rest = np.log(rest) if rest > 0 else -np.inf
    for a in range(n + 1):
        if a and not np.isfinite(lp_pred):
            break
        for b in range(n - a + 1):
            if b and not np.isfinite(lp_opp):
                break
            c = n - a - b
            if c and not np.isfinite(lp_rest):
                continue
            logp = (ln(n) - ln(a) - ln(b) - ln(c)
                    + (a * lp_pred if a else 0.0)
                    + (b * lp_opp if b else 0.0)
                    + (c * lp_rest if c else 0.0))
            out[a, b] = np.exp(logp)
    return out


def combine_grids(g1, g2):
    """The joint (a, b) distribution of two independent groups.

    A 2-D convolution: the counts add. Used to put responders and
    non-responders in one sample without assuming the sample is
    homogeneous, which it never is.
    """
    n1, n2 = g1.shape[0] - 1, g2.shape[0] - 1
    n = n1 + n2
    out = np.zeros((n + 1, n + 1))
    nz = np.argwhere(g1 > 0)
    for a1, b1 in nz:
        w = g1[a1, b1]
        out[a1:a1 + n2 + 1, b1:b1 + n2 + 1] += w * g2
    return out


def sign_test_critical(max_d, alpha=RULE_SIGN_ALPHA):
    """For each number of detections d, the smallest predicted count
    whose one-sided exact binomial p-value is at or below alpha.

    np.inf where no count can reach significance, which is every
    d below 5 at alpha = 0.05: four predicted out of four gives
    p = 0.0625. That hard floor is the sign test's real limitation and
    the report states it rather than letting a row of zeros imply the
    design is at fault.
    """
    crit = np.full(max_d + 1, np.inf)
    for d in range(max_d + 1):
        for a in range(d + 1):
            if stats.binom.sf(a - 1, d, 0.5) <= alpha:
                crit[d] = a
                break
    return crit


def rule_mask_count(n, k, m):
    """Boolean (a, b) mask: at least k predicted, at most m opposite."""
    a = np.arange(n + 1)[:, None]
    b = np.arange(n + 1)[None, :]
    return (a >= k) & (b <= m)


def rule_mask_sign(n, crit, k_min):
    """Boolean (a, b) mask for the sign test, plus a floor on a.

    Fires when a is at least the critical count for d = a + b, and at
    least k_min.
    """
    a = np.arange(n + 1)[:, None]
    b = np.arange(n + 1)[None, :]
    d = a + b
    need = np.where(d <= len(crit) - 1, crit[np.clip(d, 0, len(crit) - 1)],
                    np.inf)
    return (a >= need) & (a >= k_min)


def rule_false_claim_exact(n, k, m, p_pred, p_opp):
    """P(claim) under a null, EXACTLY, by multinomial enumeration.

    A rule claims the feature when at least k of n people are
    classified predicted and at most m are classified opposite. Under
    a null where each person is independently predicted with
    probability p_pred and opposite with p_opp, the counts are
    trinomial, so

        P(claim) = sum over a >= k, b <= m, a + b <= n of
                       n! / (a! b! (n-a-b)!)
                       * p_pred^a * p_opp^b * (1-p_pred-p_opp)^(n-a-b)

    Computed exactly rather than simulated. A false-claim rate is the
    number that goes in the registered plan, and a simulated 0.049
    with Monte Carlo error 0.002 cannot be distinguished from 0.051 --
    which is the distinction the plan rests on. The sum has at most
    (n+1)(n+2)/2 terms, so for n <= 120 this is instant.
    """
    rest = 1.0 - p_pred - p_opp
    if rest < -1e-12:
        raise ValueError(
            f"p_predicted + p_opposite = {p_pred + p_opp} > 1")
    rest = max(rest, 0.0)
    from math import lgamma
    total = 0.0
    ln = lambda x: lgamma(x + 1.0)
    for a in range(k, n + 1):
        for b in range(0, min(m, n - a) + 1):
            c = n - a - b
            logp = (ln(n) - ln(a) - ln(b) - ln(c)
                    + (a * np.log(p_pred) if a else 0.0)
                    + (b * np.log(p_opp) if b else 0.0)
                    + (c * np.log(rest) if c else 0.0))
            if (a and p_pred <= 0) or (b and p_opp <= 0) \
                    or (c and rest <= 0):
                continue
            total += float(np.exp(logp))
    return min(total, 1.0)


def run_rule_sweep(X, th, rates, report, root):
    """Candidate decision rules, in three families.

    A per-person classification is not a claim. The claim is about the
    SAMPLE, it needs a rule fixed in advance, and the rule has a
    false-claim rate of its own that has nothing to do with the 2.5%
    per person. Three families:

      count-absolute      at least k predicted, at most m opposite,
                          m a COUNT. Degenerates as N grows: opposite
                          classifications scale with the sample, so
                          the clause becomes impossible and the only
                          admissible rules are ones that never fire.
      count-proportional  the same with m a FRACTION of N. Scales.
      sign                among those classified either way, a
                          one-sided exact binomial test of predicted
                          against opposite at p = 0.5, optionally with
                          a floor on the predicted count.

    EVERY RULE IS EVALUATED AGAINST THE SAME OBJECT: the exact joint
    distribution of (predicted, opposite) counts for the sample, built
    once per scenario by convolving the responders' trinomial with the
    non-responders'. So the three families are compared on identical
    arithmetic, and nothing is simulated that can be computed.

    NOTHING IS CHOSEN HERE. Each row carries its false-claim rate under
    every null, its power under every responder share, and its verdict
    under BOTH admissibility standards. Which error matters more, and
    how much aesthetic preference to expect in a population, are
    judgements for the registered plan.
    """
    print("\n" + "=" * 60)
    print("CANDIDATE DECISION RULES")
    print("=" * 60)
    print(f"  k up to max({RULE_K_MAX}, {RULE_K_FRAC:g}N); "
          f"m absolute {RULE_M_ABSOLUTE}, "
          f"proportional {RULE_M_FRACTIONS}")
    if RULE_SIGN_TEST:
        print(f"  sign test at alpha {RULE_SIGN_ALPHA}, "
              f"minimum predicted {RULE_SIGN_MIN_PREDICTED}")
    print(f"  sample sizes {SAMPLE_SIZES}")
    print(f"  responder weights {RULE_RESPONDER_WEIGHTS} per SD, "
          f"shares {RULE_RESPONDER_SHARES}, lapses {LAPSE_RATES}")

    med = informative_medians(X)

    # --- per-person rates, per null and per feature ---
    #
    # SYMMETRISED. Every null here is symmetric by construction, so
    # the two measured rates are two estimates of the SAME quantity
    # and their average is the better one. Leaving them separate
    # would feed sampling noise into the sign test, which exists to
    # detect asymmetry and therefore cannot tell estimation noise in
    # its null from a real effect: at 5,000 people a 1.2-standard-error
    # wobble lifted its apparent false-claim rate from 0.037 to
    # 0.053, across alpha, purely from noise.
    per = {}
    for _, r in rates.iterrows():
        key = ("guesser" if r.null == "guesser"
               else f"taste_{r.spread_in_floors:g}")
        p = 0.5 * (float(r.p_predicted) + float(r.p_opposite))
        per[(key, r.feature)] = (p, p)
    nulls = sorted({k for k, _ in per},
                   key=lambda s: (s != "guesser", s))

    # THE MEASUREMENT ERROR IN EACH NULL'S CLASSIFICATION RATE.
    # p is the symmetrised rate (p_predicted + p_opposite) / 2 from
    # N_TASTE_PEOPLE people. For a trinomial with both categories at
    # p, Var(p_hat) = p(1 - 2p) / (2N) -- the two counts are
    # negatively correlated, which is why this is smaller than the
    # binomial p(1-p)/N a reader might expect.
    se_p = {}
    for (nl, feat), (p, _) in per.items():
        v = max(p * (1.0 - 2.0 * p), 0.0) / (2.0 * N_TASTE_PEOPLE)
        se_p[(nl, feat)] = float(np.sqrt(v))
    if RULE_MC_MARGIN:
        print(f"\n  MC error in the measured classification rates "
              f"({N_TASTE_PEOPLE:,} non-responders):")
        for nl in nulls:
            vals = [se_p[(nl, f)] for f in ("colour", "fog")
                    if (nl, f) in se_p]
            pts = [per[(nl, f)][0] for f in ("colour", "fog")
                   if (nl, f) in per]
            print(f"    {nl:16s} p = "
                  + " / ".join(f"{v:.4f}" for v in pts)
                  + "   SE = " + " / ".join(f"{v:.5f}" for v in vals))
        print(f"  admissibility is certified against the upper end of "
              f"a 95% interval")
        print(f"  (point estimate + {RULE_MC_MARGIN_Z} x SE), not the "
              "point estimate")
    mid_mult = NONRESPONDER_SPREAD_MULT
    mid = f"taste_{mid_mult:g}"

    # --- a real responder's classification rates, per feature and
    #     per lapse rate: lapses attenuate the weight, so they change
    #     how often a responder is caught and therefore the power of
    #     every rule ---
    rng = np.random.default_rng(RANDOM_SEED + 31337)
    resp = {}
    for k_idx, feat in enumerate(("colour", "fog")):
        for w in RULE_RESPONDER_WEIGHTS:
            for lam in LAPSE_RATES:
                W = np.zeros((N_DETECT_PEOPLE, 2))
                W[:, k_idx] = w
                est = fit_batch(X, simulate_lapsing(X, W, lam, rng))
                pr, op = classify(est, th)
                resp[(feat, w, lam)] = (float(pr[:, k_idx].mean()),
                                        float(op[:, k_idx].mean()))
    print("\n  how often a REAL responder is classified predicted, "
          "by effect size and lapse:")
    hdr = "    " + " " * 26 + "".join(
        f"lapse {lam:<6g}" for lam in LAPSE_RATES)
    print(hdr)
    for feat in ("colour", "fog"):
        for w in RULE_RESPONDER_WEIGHTS:
            cells = "".join(
                f"{resp[(feat, w, lam)][0]:<12.3f}"
                for lam in LAPSE_RATES)
            print(f"    {feat:7s} {w:.2f}/SD "
                  f"(p={as_probability(w, med[feat]):.2f}) "
                  f"{cells}")
    print("    p = probability of picking the predicted image on a "
          "typical trial of that")
    print(f"    feature ({med['colour']:.2f} SD for colour, "
          f"{med['fog']:.2f} SD for fog)")

    crit = sign_test_critical(max(SAMPLE_SIZES) + 1)

    # --- the rules, as (family, label, k, m, spec) per N ---
    def rules_for(n):
        out = []
        k_hi = min(n, max(RULE_K_MAX, int(np.ceil(RULE_K_FRAC * n))))
        for kk in range(1, k_hi + 1):
            for j in RULE_M_ABSOLUTE:
                out.append(("count-absolute", kk, j, float(j)))
            for f in RULE_M_FRACTIONS:
                out.append(("count-proportional", kk,
                            int(np.floor(f * n)), f))
        if RULE_SIGN_TEST:
            for kmin in RULE_SIGN_MIN_PREDICTED:
                out.append(("sign", kmin, -1, float(RULE_SIGN_ALPHA)))
        return out

    rows = []
    for feat in ("colour", "fog"):
        for n in SAMPLE_SIZES:
            specs = rules_for(n)
            masks = {}
            for family, kk, mm, spec in specs:
                masks[(family, kk, mm, spec)] = (
                    rule_mask_sign(n, crit, kk) if family == "sign"
                    else rule_mask_count(n, kk, mm))

            # Null grids: the whole sample from one null.
            null_grid = {nl: trinomial_grid(n, *per[(nl, feat)])
                         for nl in nulls}
            # The SAME grids at p + h, so each rule's derivative with
            # respect to the measured classification rate is one more
            # masked sum rather than its own enumeration.
            null_grid_h = {
                nl: trinomial_grid(n, per[(nl, feat)][0] + RULE_MC_STEP,
                                   per[(nl, feat)][1] + RULE_MC_STEP)
                for nl in nulls} if RULE_MC_MARGIN else {}

            # Power grids: a share of responders, the rest from the
            # middle taste spread -- a sample of real people is not a
            # sample of coin flippers.
            tp, to = per[(mid, feat)]
            pow_grid = {}
            # The non-responder half depends only on N and the share,
            # so it is built once per share rather than per cell.
            nn_grid = {}
            for share in RULE_RESPONDER_SHARES:
                n_n = n - int(round(share * n))
                nn_grid[share] = (trinomial_grid(n_n, tp, to)
                                  if n_n else None)
            for w in RULE_RESPONDER_WEIGHTS:
                for lam in LAPSE_RATES:
                    rp, ro = resp[(feat, w, lam)]
                    for share in RULE_RESPONDER_SHARES:
                        n_r = int(round(share * n))
                        g = trinomial_grid(n_r, rp, ro) if n_r else None
                        h = nn_grid[share]
                        if g is None:
                            pow_grid[(w, lam, share)] = h
                        elif h is None:
                            pow_grid[(w, lam, share)] = g
                        else:
                            pow_grid[(w, lam, share)] = combine_grids(
                                g, h)

            for (family, kk, mm, spec), mask in masks.items():
                row = {"feature": feat, "n": n, "family": family,
                       "k": kk, "m": mm, "m_spec": spec,
                       "k_over_n": round(kk / n, 4),
                       # THE SCENARIO, on every row. A power figure
                       # without its scenario is not a number anyone
                       # can act on.
                       "median_informative_SD": round(med[feat], 4),
                       "nonresponder_taste_spread": mid_mult}
                for nl in nulls:
                    f0 = float(null_grid[nl][mask].sum())
                    row[f"false_{nl}"] = f0
                    if RULE_MC_MARGIN:
                        # delta method: SE(FP) = |dFP/dp| x SE(p)
                        f1 = float(null_grid_h[nl][mask].sum())
                        se = (abs(f1 - f0) / RULE_MC_STEP
                              * se_p[(nl, feat)])
                        row[f"false_{nl}_se"] = se
                        row[f"false_{nl}_upper"] = (
                            f0 + RULE_MC_MARGIN_Z * se)
                # EVERY POWER COLUMN NAMES ITS WHOLE SCENARIO:
                # effect size per SD, responder share, lapse rate. The
                # taste spread of the non-responders is one column for
                # the row, since it is the same for all of them.
                for w in RULE_RESPONDER_WEIGHTS:
                    row[f"effect_p_median_w{w:g}"] = round(
                        as_probability(w, med[feat]), 4)
                    for lam in LAPSE_RATES:
                        for share in RULE_RESPONDER_SHARES:
                            row[f"power_w{w:g}_s{share:g}_lapse{lam:g}"] \
                                = float(
                                    pow_grid[(w, lam, share)][mask].sum())
                rows.append(row)
    tab = pd.DataFrame(rows)

    # --- the two admissibility standards ---
    for std, mults in ADMISSIBILITY_STANDARDS.items():
        base = ["false_guesser"] + [f"false_taste_{m:g}" for m in mults]
        cols = [c for c in base if c in tab.columns]
        tab[f"worst_{std}"] = tab[cols].max(axis=1)
        if RULE_MC_MARGIN:
            # ADMISSIBILITY IS CERTIFIED ON THE UPPER BOUND. The point
            # estimate stays in the table as worst_<std>, because it
            # is what the rate IS; the upper bound is what the rule
            # has to clear to be trusted.
            ucols = [c + "_upper" for c in base
                     if c + "_upper" in tab.columns]
            tab[f"worst_upper_{std}"] = tab[ucols].max(axis=1)
            tab[f"admissible_{std}"] = (
                tab[f"worst_upper_{std}"] <= RULE_TARGET_FALSE)
            tab[f"admissible_point_{std}"] = (
                tab[f"worst_{std}"] <= RULE_TARGET_FALSE)
        else:
            tab[f"admissible_{std}"] = (
                tab[f"worst_{std}"] <= RULE_TARGET_FALSE)
    tab.to_csv(os.path.join(root, "rule_sweep.csv"), index=False)

    if RULE_MC_MARGIN:
        print("\n  WHAT THE MONTE CARLO MARGIN COSTS")
        for std in ADMISSIBILITY_STANDARDS:
            pt = int(tab[f"admissible_point_{std}"].sum())
            ub = int(tab[f"admissible_{std}"].sum())
            print(f"    {std:22s} {pt:5d} admissible on the point "
                  f"estimate, {ub:5d} on the upper bound "
                  f"({pt - ub} dropped)")
        se_cols = [c for c in tab.columns if c.endswith("_se")]
        if se_cols:
            mx = tab[se_cols].max().max()
            print(f"    worst SE on any rule's false-claim rate: "
                  f"{mx:.5f}")
            print(f"    so the margin is at most "
                  f"{RULE_MC_MARGIN_Z * mx:.4f} on a "
                  f"{RULE_TARGET_FALSE} target")
        report.append(
            ("MC margin on admissibility",
             f"upper end of a 95% interval; worst SE "
             f"{tab[se_cols].max().max():.5f}, margin up to "
             f"{RULE_MC_MARGIN_Z * tab[se_cols].max().max():.4f}"))
    if RULE_MC_MARGIN:
        # --- which rule did the margin actually move, and at what
        #     cost? ---
        pw0, psh0, plam0 = RULE_SUMMARY_PICK
        pcol0 = f"power_w{pw0:g}_s{psh0:g}_lapse{plam0:g}"
        moved = []
        if pcol0 in tab.columns:
            for std in ADMISSIBILITY_STANDARDS:
                for feat in ("colour", "fog"):
                    for n in sorted(tab.n.unique()):
                        g0 = tab[(tab.feature == feat) & (tab.n == n)]
                        pt = g0[g0[f"admissible_point_{std}"]]
                        ub = g0[g0[f"admissible_{std}"]]
                        if not len(pt) or not len(ub):
                            continue
                        a = pt.loc[pt[pcol0].idxmax()]
                        b = ub.loc[ub[pcol0].idxmax()]
                        if (a.family == b.family and a.k == b.k
                                and a.m == b.m):
                            continue
                        moved.append({
                            "standard": std, "feature": feat, "n": n,
                            "rule_on_point_estimate":
                                f"{a.family} k>={int(a.k)}, "
                                f"m<={int(a.m)}",
                            "rule_on_upper_bound":
                                f"{b.family} k>={int(b.k)}, "
                                f"m<={int(b.m)}",
                            "worst_point": round(
                                float(a[f"worst_{std}"]), 5),
                            "worst_upper_of_that_rule": round(
                                float(a[f"worst_upper_{std}"]), 5),
                            "power_point": round(float(a[pcol0]), 4),
                            "power_upper": round(float(b[pcol0]), 4),
                            "power_cost": round(
                                float(a[pcol0] - b[pcol0]), 4)})
        print("\n  WHICH RULE THE MARGIN MOVED, AND WHAT IT COST")
        if not moved:
            print("    none: every standard selects the same rule "
                  "either way, so the margin")
            print("    cost no power at all here. It still changed "
                  "the admissible SET, which")
            print("    is what protects a rule chosen at a different "
                  "cell or sample size.")
            report.append(("MC margin, rules moved",
                           "none at the summary cell"))
        else:
            mv = pd.DataFrame(moved)
            mv.to_csv(os.path.join(root, "mc_margin_moved.csv"),
                      index=False)
            print("  " + mv.to_string(index=False)
                  .replace("\n", "\n  "))
            print(f"\n    {len(mv)} selection(s) changed; power cost "
                  f"{mv.power_cost.min():+.3f} to "
                  f"{mv.power_cost.max():+.3f} at the summary cell")
            print("    (positive is a cost: the rule the margin "
                  "refuses to certify was the")
            print("     more powerful one, which is why it was "
                  "sitting on the line)")
            report.append(
                ("MC margin, rules moved",
                 f"{len(mv)} selections changed, power cost up to "
                 f"{mv.power_cost.max():+.3f} at the summary cell"))
    print(f"\n  {len(tab)} candidate rules.")
    for std in ADMISSIBILITY_STANDARDS:
        print(f"    {std:10s} {int(tab[f'admissible_{std}'].sum()):5d} "
              f"admissible at {RULE_TARGET_FALSE:.0%} "
              f"({', '.join(str(m) + 'x' for m in ADMISSIBILITY_STANDARDS[std])}"
              " taste spreads + guessers)")
    print("  BOTH standards are reported; neither is adopted here.")

    # --- the sign test's bounded false-claim rate ---
    if RULE_SIGN_TEST:
        sg = tab[(tab.family == "sign") & (tab.k == 0)]
        fcols = [c for c in tab.columns if c.startswith("false_")]
        print("\n  THE SIGN TEST'S FALSE-CLAIM RATE, across the nulls:")
        print("  Taste is zero-centred, so for a non-responder the two "
              "directions are")
        print("  exchangeable and, CONDITIONAL ON THE NUMBER DETECTED, "
              "the predicted")
        print("  count is Binomial(d, 0.5) under every symmetric null. "
              "That bounds the")
        print(f"  rate at alpha = {RULE_SIGN_ALPHA} whatever the "
              "spread is -- which matters because")
        print("  the spread is the one quantity nobody can measure in "
              "advance.")
        print("  It is NOT constant across spreads: the number DETECTED "
              "does depend on")
        print("  the spread, and a null with few detections cannot fire "
              "a test that")
        print("  needs five. So the rate approaches alpha from below as "
              "detections")
        print("  accumulate, and never exceeds it.")
        show = sg[sg.feature == "colour"][["n"] + fcols]
        print(show.round(5).to_string(index=False))
        worst = float(sg[fcols].to_numpy().max())
        held = worst <= RULE_SIGN_ALPHA + 1e-9
        print(f"    largest rate anywhere in the table: {worst:.5f} "
              f"against alpha {RULE_SIGN_ALPHA}"
              f"  -> {'BOUND HOLDS' if held else 'BOUND VIOLATED'}")
        if not held:
            print("    The bound is a theorem under exact symmetry, so "
                  "a violation here means")
            print("    the null rates fed in were not symmetric -- "
                  "raise N_TASTE_PEOPLE.")
        report.append(("sign test, worst false-claim rate",
                       f"{worst:.4f} vs alpha {RULE_SIGN_ALPHA} "
                       f"({'bound holds' if held else 'BOUND VIOLATED'})"))
        # Where it becomes usable: it cannot fire below 5 detections.
        d_min = int(np.argmax(np.isfinite(crit)))
        print(f"\n    it cannot fire below {d_min} detections "
              f"(needs {int(crit[d_min])} of {d_min} predicted);")
        print(f"    at {d_min} to 7 detections it needs EVERY detected "
              "person in the predicted")
        print("    direction. That discreteness, not the design, is "
              "what limits it at small N.")
        report.append(("sign test needs at least",
                       f"{d_min} detections to fire at all"))

    # --- the most powerful admissible rule, per family and standard ---
    pw, psh, plam = RULE_SUMMARY_PICK
    pcol = f"power_w{pw:g}_s{psh:g}_lapse{plam:g}"
    best = []
    for std in ADMISSIBILITY_STANDARDS:
        ok = tab[tab[f"admissible_{std}"]]
        for (feat, n, fam), g in ok.groupby(["feature", "n", "family"]):
            if not len(g):
                continue
            r = g.loc[g[pcol].idxmax()]
            best.append({"standard": std, "feature": feat, "n": n,
                         "family": fam, "k": int(r.k), "m": int(r.m),
                         "m_spec": r.m_spec,
                         "false_guesser": round(r.false_guesser, 5),
                         "false_worst": round(r[f"worst_{std}"], 5),
                         "power": round(r[pcol], 4)})
    bt = pd.DataFrame(best)
    if len(bt):
        bt.to_csv(os.path.join(root, "rule_best.csv"), index=False)
        print(f"\n  MOST POWERFUL ADMISSIBLE RULE, chosen at the "
              "stated cell:")
        print(f"  effect {pw} per SD, {psh:.0%} responder share, "
              f"lapse {plam}, non-responders at")
        print(f"  the {mid_mult}x taste spread. A different cell can "
              "pick a different rule,")
        print("  which is why the criterion is stated rather than "
              "assumed.")
        for std in ADMISSIBILITY_STANDARDS:
            for feat in ("colour", "fog"):
                g = bt[(bt.standard == std) & (bt.feature == feat)]
                if not len(g):
                    continue
                print(f"\n  [{std}] {feat}:")
                print(g.pivot(index="n", columns="family",
                              values="power").round(3).to_string())

        # The degeneracy, per family, stated.
        print("\n  DOES POWER RISE WITH THE SAMPLE? (the curve is a "
              "STAIRCASE -- k is an")
        print("  integer, so it moves in jumps and a local dip is the "
              "grid, not the design)")
        for std in ADMISSIBILITY_STANDARDS:
            for feat in ("colour", "fog"):
                for fam in sorted(bt.family.unique()):
                    g = bt[(bt.standard == std) & (bt.feature == feat)
                           & (bt.family == fam)].sort_values("n")
                    if len(g) < 2:
                        continue
                    v = g.power.to_numpy()
                    ns = g.n.to_numpy()
                    pi = int(np.argmax(v))
                    verdict = ("still rising at the largest N"
                               if v[-1] >= 0.9 * v[pi] else
                               f"PEAKS AT N={int(ns[pi])} "
                               f"({v[pi]:.2f}) AND FALLS TO {v[-1]:.2f}")
                    # Not std[:4]: "practical" and
                    # "practical_at_boundary" collide at four
                    # characters, so two different standards printed
                    # under the same label.
                    print(f"    [{std:21s}] {feat:7s} {fam:20s} "
                          + " ".join(f"{x:.2f}" for x in v)
                          + "   " + verdict)
        for std in ADMISSIBILITY_STANDARDS:
            for fam in sorted(bt.family.unique()):
                g = bt[(bt.standard == std) & (bt.feature == "colour")
                       & (bt.family == fam)].sort_values("n")
                if len(g) < 2:
                    continue
                v, ns = g.power.to_numpy(), g.n.to_numpy()
                pi = int(np.argmax(v))
                report.append(
                    (f"[{std}] colour power, {fam}",
                     f"rises to {v[-1]:.2f} at N={int(ns[-1])}"
                     if v[-1] >= 0.9 * v[pi] else
                     f"peaks {v[pi]:.2f} at N={int(ns[pi])}, "
                     f"falls to {v[-1]:.2f}"))
    report.append(("candidate rules swept", str(len(tab))))
    return tab, bt


def rule_summary_table(tab, X, report, root):
    """THE DECISION TABLE. One row per sample size, family, standard
    and feature; the best admissible rule and what it would buy.

    The full sweep is the evidence and runs to thousands of rows. This
    is the part a decision gets made from, so it is deliberately
    narrow: the sample sizes that could actually be recruited, and the
    scenario cells that bracket what is plausible.

    EVERY POWER FIGURE CARRIES ITS SCENARIO -- effect size per SD with
    its choice-probability translation, responder share, lapse rate --
    because a power number without those is not actionable, and
    quoting the best cell without them is how a plan ends up
    optimistic.

    The "best" rule is the most powerful admissible one AT A STATED
    CELL (RULE_SUMMARY_PICK). A different cell can pick a different
    rule; the stated criterion is what makes that visible instead of
    hidden.
    """
    print("\n" + "=" * 60)
    print("DECISION TABLE")
    print("=" * 60)
    med = informative_medians(X)
    pw, psh, plam = RULE_SUMMARY_PICK
    pcol = f"power_w{pw:g}_s{psh:g}_lapse{plam:g}"
    ns = [n for n in RULE_SUMMARY_N if n in SAMPLE_SIZES]
    missing = [n for n in RULE_SUMMARY_N if n not in SAMPLE_SIZES]
    if missing:
        print(f"  NOTE: {missing} not in SAMPLE_SIZES, so not in this "
              "table.")
    fcols = [c for c in tab.columns if c.startswith("false_")]

    rows = []
    for std in ADMISSIBILITY_STANDARDS:
        ok = tab[tab[f"admissible_{std}"]]
        for feat in ("colour", "fog"):
            for n in ns:
                for fam in sorted(tab.family.unique()):
                    g = ok[(ok.feature == feat) & (ok.n == n)
                           & (ok.family == fam)]
                    if not len(g):
                        rows.append({"standard": std, "feature": feat,
                                     "n": n, "family": fam,
                                     "rule": "none admissible"})
                        continue
                    r = g.loc[g[pcol].idxmax()]
                    rule = (f"sign test, min predicted {int(r.k)}"
                            if fam == "sign"
                            else f"k>={int(r.k)}, m<={int(r.m)}"
                            + (f" ({r.m_spec:.0%} of N)"
                               if fam == "count-proportional" else ""))
                    row = {"standard": std, "feature": feat, "n": n,
                           "family": fam, "rule": rule,
                           "k": int(r.k), "m": int(r.m),
                           "m_spec": r.m_spec}
                    for c in fcols:
                        row[c] = round(float(r[c]), 5)
                    row["false_worst_this_standard"] = round(
                        float(r[f"worst_{std}"]), 5)
                    # THE COLUMN ADMISSIBILITY IS ACTUALLY JUDGED ON.
                    # The line above is the point estimate; a reader
                    # checking this table against the target needs the
                    # upper bound, which is what certified the rule.
                    if RULE_MC_MARGIN and f"worst_upper_{std}" in r:
                        row["false_worst_upper_this_standard"] = round(
                            float(r[f"worst_upper_{std}"]), 5)
                    for w in RULE_RESPONDER_WEIGHTS:
                        for share in RULE_SUMMARY_SHARES:
                            for lam in RULE_SUMMARY_LAPSES:
                                src_c = (f"power_w{w:g}_s{share:g}"
                                         f"_lapse{lam:g}")
                                if src_c in tab.columns:
                                    row[f"power_e{w:g}_s{share:g}"
                                        f"_l{lam:g}"] = round(
                                        float(r[src_c]), 4)
                    rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(root, "rule_decision_table.csv"),
               index=False)

    print(f"  sample sizes {ns}; effect sizes "
          f"{RULE_RESPONDER_WEIGHTS} per SD; responder shares "
          f"{RULE_SUMMARY_SHARES};")
    print(f"  lapse rates {RULE_SUMMARY_LAPSES}; non-responders at "
          "the middle taste spread.")
    print(f"  Best rule chosen at: effect {pw}/SD, share "
          f"{psh:.0%}, lapse {plam}.")
    print("\n  WHAT THE EFFECT SIZES MEAN on this list "
          "(probability of picking the")
    print("  predicted image on a typical trial of that feature):")
    for feat in ("colour", "fog"):
        cells = "   ".join(
            f"{w:.2f}/SD = {as_probability(w, med[feat]):.0%}"
            for w in RULE_RESPONDER_WEIGHTS)
        print(f"    {feat:7s} (median {med[feat]:.2f} SD)   {cells}")

    for std in ADMISSIBILITY_STANDARDS:
        mults = ADMISSIBILITY_STANDARDS[std]
        print(f"\n  {'=' * 56}")
        print(f"  STANDARD: {std.upper()} -- false-claim rate at or "
              f"below {RULE_TARGET_FALSE:.0%} for")
        print(f"  guessers and the "
              f"{', '.join(str(m) + 'x' for m in mults)} taste "
              "spread(s)"
              + ("" if 1.0 in mults
                 else "; 1.0x reported, not constrained"))
        print(f"  {'=' * 56}")
        for feat in ("colour", "fog"):
            g = out[(out.standard == std) & (out.feature == feat)]
            if not len(g):
                continue
            print(f"\n  --- {feat.upper()} ---")
            show = ["n", "family", "rule"] + fcols
            print(g[show].to_string(index=False))
            print()
            pcols = [c for c in out.columns if c.startswith("power_e")]
            print(g[["n", "family"] + pcols].to_string(index=False))
    print("\n  column key: power_e<effect>_s<share>_l<lapse>")
    report.append(("decision table rows", str(len(out))))
    return out


def equal_behaviour_table(X, th, rates, report, root):
    """POWER AT EQUAL BEHAVIOUR -- the primary form of the existence
    claims.

    Every other power table in this script is indexed by weight per
    corpus SD. That is the wrong index for comparing the two
    features, because the typical informative difference is not the
    same on both: the same per-SD weight produces a larger choice
    bias on colour than on fog. Indexed that way, H1 (fog) looks
    several times worse powered than H2 (colour).

    Indexed by BEHAVIOUR -- the choice probability on a typical
    informative trial -- most of that gap disappears, though NOT all
    of it in every cell. Both effects are real and they largely
    cancel:

      equal behaviour needs the fog weight to be
      median_colour / median_fog times the colour weight
                                                    about 1.62

      the fog threshold is higher than the colour one by
      sqrt(sum pred_sat^2 / sum pred_fog^2), because a logistic
      weight's standard error scales as 1 / sqrt(sum x^2)
                                                    about 1.67

    So fog's weight-to-threshold ratio is about 0.97 of colour's.
    THE HIGHER FOG THRESHOLD IS NOT A DEFECT: it is the arithmetic
    consequence of the list carrying less fog information, most of
    which is the colour cut raising the colour differences and leaving
    the fog ones alone. It costs nothing once the claim is stated in
    the units the claim is actually about.

    This function reports both tables so the difference between the
    framings is visible rather than argued.
    """
    print("\n" + "=" * 60)
    print("POWER AT EQUAL BEHAVIOUR  (the primary form)")
    print("=" * 60)
    med = informative_medians(X)
    U = th["upper"]
    SS = [float((X[:, k] ** 2).sum()) for k in (0, 1)]
    print(f"  median informative difference : colour "
          f"{med['colour']:.3f} SD, fog {med['fog']:.3f} SD")
    print(f"  information, sum pred^2       : colour {SS[0]:.1f}, "
          f"fog {SS[1]:.1f}   ratio {SS[0] / SS[1]:.2f}")
    print(f"  threshold                     : colour {U[0]:.3f}, "
          f"fog {U[1]:.3f}   ratio {U[1] / U[0]:.2f}")
    print(f"  predicted threshold ratio from the information: "
          f"{np.sqrt(SS[0] / SS[1]):.2f}")
    print(f"  equal behaviour needs w_fog / w_colour = "
          f"{med['colour'] / med['fog']:.2f}")
    print(f"  -> fog's weight-to-threshold ratio is "
          f"{(med['colour'] / med['fog']) / (U[1] / U[0]):.3f} "
          "of colour's")
    report.append(("threshold ratio, fog / colour",
                   f"{U[1] / U[0]:.2f} (information predicts "
                   f"{np.sqrt(SS[0] / SS[1]):.2f})"))
    report.append(("fog / colour at EQUAL BEHAVIOUR",
                   f"{(med['colour'] / med['fog']) / (U[1] / U[0]):.3f}"
                   " of colour's weight-to-threshold ratio"))

    print("\n  the weight each behaviour needs, per feature:")
    print(f"  {'p on a typical trial':>22}  {'w colour':>9}  "
          f"{'w fog':>9}")
    for p in BEHAVIOUR_TARGETS:
        lg = np.log(p / (1 - p))
        print(f"  {p:>22.2f}  {lg / med['colour']:>9.3f}  "
              f"{lg / med['fog']:>9.3f}")

    # --- symmetrised null rates, as the sweep uses ---
    per = {}
    for _, r in rates.iterrows():
        key = ("guesser" if r.null == "guesser"
               else f"taste_{r.spread_in_floors:g}")
        p = 0.5 * (float(r.p_predicted) + float(r.p_opposite))
        per[(key, r.feature)] = p
    mid_mult = NONRESPONDER_SPREAD_MULT
    std = "practical"
    mults = ADMISSIBILITY_STANDARDS[std]
    crit = sign_test_critical(max(RULE_SUMMARY_N) + 1)
    rng = np.random.default_rng(RANDOM_SEED + 606)

    rows = []
    for k_idx, feat in enumerate(("colour", "fog")):
        nulls_q = [per[("guesser", feat)]] + [
            per[(f"taste_{m:g}", feat)] for m in mults]
        pt = per[(f"taste_{mid_mult:g}", feat)]
        # responder rates at each behaviour and lapse
        resp = {}
        for p in BEHAVIOUR_TARGETS:
            w = np.log(p / (1 - p)) / med[feat]
            for lam in RULE_SUMMARY_LAPSES:
                W = np.zeros((N_DETECT_PEOPLE, 2))
                W[:, k_idx] = w
                est = fit_batch(X, simulate_lapsing(X, W, lam, rng))
                pr, op = classify(est, th)
                resp[(p, lam)] = (float(pr[:, k_idx].mean()),
                                  float(op[:, k_idx].mean()), w)
        for n in [x for x in RULE_SUMMARY_N if x in SAMPLE_SIZES]:
            # admissible rules under the practical standard
            adm = {}
            k_hi = min(n, max(RULE_K_MAX,
                              int(np.ceil(RULE_K_FRAC * n))))
            cands = [("count-proportional", kk, int(np.floor(f * n)))
                     for kk in range(1, k_hi + 1)
                     for f in RULE_M_FRACTIONS]
            if RULE_SIGN_TEST:
                cands += [("sign", kmin, -1)
                          for kmin in RULE_SIGN_MIN_PREDICTED]
            for fam, kk, mm in cands:
                mask = (rule_mask_sign(n, crit, kk) if fam == "sign"
                        else rule_mask_count(n, kk, mm))
                worst = max(float(trinomial_grid(n, q, q)[mask].sum())
                            for q in nulls_q)
                if worst <= RULE_TARGET_FALSE:
                    adm[(fam, kk, mm)] = (mask, worst)
            for p in BEHAVIOUR_TARGETS:
                for share in RULE_SUMMARY_SHARES:
                    n_r = int(round(share * n))
                    n_n = n - n_r
                    gn = trinomial_grid(n_n, pt, pt) if n_n else None
                    for lam in RULE_SUMMARY_LAPSES:
                        rp, ro, w = resp[(p, lam)]
                        gr = (trinomial_grid(n_r, rp, ro)
                              if n_r else None)
                        joint = (gr if gn is None else
                                 gn if gr is None else
                                 combine_grids(gr, gn))
                        for fam in ("count-proportional", "sign"):
                            best, bk, bw = -1.0, None, None
                            for (f2, kk, mm), (mask, wst) in adm.items():
                                if f2 != fam:
                                    continue
                                v = float(joint[mask].sum())
                                if v > best:
                                    best, bk, bw = v, (kk, mm), wst
                            rows.append({
                                "feature": feat,
                                "claim": ("H1 fog" if feat == "fog"
                                          else "H2 colour"),
                                "p_typical_trial": p,
                                "weight_per_SD": round(w, 3),
                                "n": n, "share": share, "lapse": lam,
                                "family": fam,
                                "k": bk[0] if bk else None,
                                "m": bk[1] if bk else None,
                                "false_worst": (round(bw, 5) if bw
                                                is not None else None),
                                # THE PER-PERSON RATES, on every row.
                                # Power is a property of the rule
                                # applied to these; without them a gap
                                # between the features cannot be
                                # attributed to either.
                                "p_person_predicted": round(rp, 4),
                                "p_person_opposite": round(ro, 4),
                                "expected_predicted": round(
                                    n_r * rp + n_n * pt, 2),
                                "power": round(best, 4) if bk else None})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(root, "power_equal_behaviour.csv"),
               index=False)

    print(f"\n  practical admissibility standard; best admissible "
          "rule per family.")
    print("  H1 = FOG (contrast), H2 = COLOUR (saturation).")
    for fam in ("count-proportional", "sign"):
        print(f"\n  --- {fam} ---")
        for share in RULE_SUMMARY_SHARES:
            for lam in RULE_SUMMARY_LAPSES:
                g = tab[(tab.family == fam) & (tab.share == share)
                        & np.isclose(tab.lapse, lam)]
                if not len(g):
                    continue
                piv = g.pivot_table(index="p_typical_trial",
                                    columns=["claim", "n"],
                                    values="power")
                print(f"\n  responder share {share:g}, lapse {lam:g}:")
                print(piv.round(3).to_string())

    # The headline comparison, in one line per cell.
    print("\n  H1 AGAINST H2 AT EQUAL BEHAVIOUR, largest gap:")
    w = tab.dropna(subset=["power"]).pivot_table(
        index=["family", "p_typical_trial", "n", "share", "lapse"],
        columns="claim", values="power")
    if {"H1 fog", "H2 colour"} <= set(w.columns):
        d = (w["H2 colour"] - w["H1 fog"]).abs()
        # NAME THE WORST CELL. A median of a few thousandths with one
        # cell at 0.14 is a different claim from a uniform few
        # thousandths, and a reader should not have to open the CSV to
        # find out which it is.
        wi = d.idxmax()
        print(f"    median difference {d.median():.3f} across "
              f"{len(d)} cells")
        print(f"    worst  {d.max():.3f}, at {dict(zip(d.index.names, wi))}")
        print(f"      there H2 colour = {w['H2 colour'].loc[wi]:.3f} "
              f"and H1 fog = {w['H1 fog'].loc[wi]:.3f}")
        n_big = int((d > EQUAL_BEHAVIOUR_TOL).sum())
        print(f"    {n_big} of {len(d)} cells differ by more than "
              "0.05.")

        # WHERE THEY DIVERGE, WHY. A gap in POWER is not a gap in
        # detectability: power is what a rule with an integer
        # threshold makes of the per-person rates. If the two features
        # are classified at nearly the same per-person rate but the
        # admissible k differs by one, the rule turns a difference of
        # a few thousandths into one of a tenth. These rows show both
        # quantities side by side so the gap can be attributed.
        if n_big:
            big = d[d > EQUAL_BEHAVIOUR_TOL].index
            sel = tab.set_index(
                ["family", "p_typical_trial", "n", "share", "lapse",
                 "claim"]).sort_index()
            print("\n    THE DIVERGING CELLS, with the per-person "
                  "rates that produced them:")
            out = []
            for idx in big:
                for claim in ("H2 colour", "H1 fog"):
                    try:
                        r = sel.loc[tuple(idx) + (claim,)]
                    except KeyError:
                        continue
                    out.append({
                        "family": idx[0], "p": idx[1], "n": idx[2],
                        "share": idx[3], "lapse": idx[4],
                        "claim": claim,
                        "per_person_predicted": r.p_person_predicted,
                        "expected_predicted": r.expected_predicted,
                        "k": r.k, "m": r.m, "power": r.power})
            bigtab = pd.DataFrame(out)
            print(bigtab.to_string(index=False))
            bigtab.to_csv(os.path.join(
                root, "equal_behaviour_diverging.csv"), index=False)
            # Quantify the attribution rather than assert it.
            w2 = bigtab.pivot_table(
                index=["family", "p", "n", "share", "lapse"],
                columns="claim",
                values=["per_person_predicted", "k", "power"])
            pp = (w2[("per_person_predicted", "H2 colour")]
                  - w2[("per_person_predicted", "H1 fog")]).abs()
            kk = (w2[("k", "H2 colour")]
                  - w2[("k", "H1 fog")]).abs()
            pwd = (w2[("power", "H2 colour")]
                   - w2[("power", "H1 fog")]).abs()
            rel = float((pp / w2[("per_person_predicted",
                                   "H2 colour")]).max())
            print(f"\n    across those cells: the per-person rates "
                  f"differ by at most {pp.max():.4f}")
            print(f"    ({rel:.0%} in relative terms), the rule's k "
                  f"differs by at most {int(kk.max())},")
            print(f"    and the POWER differs by up to "
                  f"{pwd.max():.3f}.")
            print()
            print("    SO THE AMPLIFIER IS NOT k -- it is the same in "
                  "both features in every")
            print("    one of these cells. Two things combine instead.")
            print()
            print("    FIRST, the near-cancellation between the "
                  "threshold ratio and the")
            print("    behaviour ratio is only FIRST ORDER, and these "
                  "cells sit in the far")
            print("    tail where it degrades. At a choice "
                  "probability of 0.55 the weights are")
            print("    0.20 (colour) and 0.32 (fog) against "
                  "thresholds of 0.29 and 0.50, so")
            print("    both are BELOW their thresholds and detection "
                  "is a tail probability.")
            print("    The weight-to-threshold ratios are 0.68 and "
                  "0.65 -- close, but in the")
            print("    tail a 4% difference in that ratio becomes "
                  "about 13% in the per-person")
            print("    detection rate.")
            print()
            print("    SECOND, the count rule's threshold is an "
                  "integer sitting right where")
            print("    the expected count falls. At n = 10 with every "
                  "participant a responder")
            print("    the expected number classified predicted is "
                  "2.63 against 2.30, and the")
            print("    rule needs 3. The binomial tail is steep "
                  "there, so a 0.033 difference")
            print("    in the per-person rate becomes 0.100 in "
                  "P(at least 3).")
            print()
            print("    Neither is a feature asymmetry in the sense "
                  "that matters: at p = 0.60")
            print("    and above, where the claims would actually be "
                  "made, the per-person")
            print("    rates and the power both agree to a few "
                  "thousandths.")
            report.append(
                ("diverging cells, mechanism",
                 f"per-person rates differ <= {pp.max():.4f} "
                 f"({rel:.0%} rel) with k IDENTICAL; power <= "
                 f"{pwd.max():.3f} via the binomial tail"))
        # COMPUTED, not asserted. The earlier wording ("within a
        # few per cent") came from a step-5 test draw and is not true
        # of the frozen list in every cell, so the verdict is derived
        # from the table that was just built.
        print(f"\n    VERDICT, from this run: median gap "
              f"{d.median():.4f} over {len(d)} cells, worst "
              f"{d.max():.4f},")
        print(f"    with {int((d > EQUAL_BEHAVIOUR_TOL).sum())} cells above {EQUAL_BEHAVIOUR_TOL}. "
              "So the per-SD framing carries")
        print("    most of the apparent asymmetry -- but the "
              "agreement is NOT uniform, and")
        print("    'within a few per cent' would overstate it. The "
              "cells that diverge are")
        print("    the ones where both features sit below their "
              "thresholds and neither")
        print("    claim could be made anyway, which is what the "
              "diverging-cell")
        print("    diagnostic above is for.")
        report.append(("H1 vs H2 at equal behaviour",
                       f"median gap {d.median():.3f}, worst "
                       f"{d.max():.3f} ({n_big}/{len(d)} cells "
                       f"> {EQUAL_BEHAVIOUR_TOL})"))
    return tab


def recover_taste_spread(X, th, report, root):
    """ITEM 4: estimate the taste spread from a control group.

    Two estimators, because they fail in different ways and the
    disagreement between them is itself informative.

    MOMENT ESTIMATOR. If a control's true weight is drawn from
    Normal(0, s) and the fit adds independent noise, then

        Var(fitted) = s^2 + Var(noise)

    and Var(noise) is already known: it is the variance of the fitted
    weights of pure guessers, which is what the thresholds were built
    from. So

        s_hat = sqrt(max(Var(fitted) - Var(guessers), 0))

    Uses the whole sample and both directions. Biased downward when
    the subtraction goes negative, which happens when the true spread
    is small relative to the estimation noise -- and the clip to zero
    is reported rather than hidden.

    RATE ESTIMATOR. The share of controls classified EITHER WAY is a
    monotone function of s, computed here by simulation and then
    inverted. Throws away the magnitudes and keeps only the counts, so
    it is coarser; but it depends on the same quantity the decision
    rule depends on, which makes it the more directly relevant of the
    two.

    WHAT THIS DOES NOT HANDLE. If controls also use the "pick the more
    edited image" strategy, their weights are no longer zero-centred
    and both estimators are inflated -- the moment estimator by the
    strategy's own variance, the rate estimator because the strategy
    pushes people over the threshold in the predicted direction. The
    C1 section sweeps that strategy rate, and the asymmetry between
    predicted and opposite classifications is what would reveal it:
    taste is symmetric, the strategy is not. The report states this
    rather than leaving it as an unmodelled hole.
    """
    print("\n" + "=" * 60)
    print("RECOVERING THE TASTE SPREAD FROM THE CONTROLS  (item 4)")
    print("=" * 60)
    rng = np.random.default_rng(RANDOM_SEED + 4242)
    U = th["upper"]

    # Var(noise): the guessers' own fitted-weight variance, which is
    # the same object the thresholds came from.
    var_null = np.asarray(th["sd"]) ** 2
    print(f"  Var(fitted) for pure guessers: colour "
          f"{var_null[0]:.5f}, fog {var_null[1]:.5f}")
    print(f"  (so the moment estimator subtracts these; they are the "
          "same guessers the")
    print("   thresholds were built from, not a separate assumption)")

    # The rate -> spread mapping, per feature, built once.
    grid_mult = np.round(np.arange(0.0, 2.01, 0.05), 3)
    rate_map = {}
    for k_idx, feat in enumerate(("colour", "fog")):
        rates_g = []
        for m in grid_mult:
            W = np.zeros((8000, 2))
            W[:, k_idx] = rng.normal(0.0, U[k_idx] * m, size=8000)
            est = fit_batch(X, simulate_many(X, W, rng))
            pr, op = classify(est, th)
            rates_g.append(float((pr[:, k_idx] | op[:, k_idx]).mean()))
        rate_map[feat] = (grid_mult, np.asarray(rates_g))
    print(f"\n  the rate-to-spread mapping (share classified either "
          "way):")
    for feat in ("colour", "fog"):
        g, rr = rate_map[feat]
        pts = "  ".join(f"{m:.2f}x->{v:.3f}"
                        for m, v in zip(g, rr) if m in
                        (0.0, 0.25, 0.5, 1.0, 1.5, 2.0))
        print(f"    {feat:7s} {pts}")

    rows = []
    for k_idx, feat in enumerate(("colour", "fog")):
        gm, gr = rate_map[feat]
        for true_mult in TASTE_SPREADS_IN_FLOORS:
            s_true = U[k_idx] * true_mult
            for n_c in TASTE_RECOVERY_N:
                mom, rat, clipped = [], [], 0
                for _ in range(TASTE_RECOVERY_REPS):
                    W = np.zeros((n_c, 2))
                    W[:, k_idx] = rng.normal(0.0, s_true, size=n_c)
                    est = fit_batch(X, simulate_many(X, W, rng))
                    v = float(est[:, k_idx].var(ddof=1))
                    diff = v - var_null[k_idx]
                    if diff <= 0:
                        clipped += 1
                    mom.append(np.sqrt(max(diff, 0.0)))
                    pr, op = classify(est, th)
                    r = float((pr[:, k_idx] | op[:, k_idx]).mean())
                    # Invert the monotone mapping; np.interp needs an
                    # increasing x, and the rate rises with the spread.
                    rat.append(float(np.interp(r, gr, gm)) * U[k_idx])
                mom = np.asarray(mom)
                rat = np.asarray(rat)
                rows.append({
                    "feature": feat,
                    "true_spread_in_thresholds": true_mult,
                    "true_spread_abs": round(s_true, 4),
                    "n_controls": n_c,
                    "moment_mean": round(float(mom.mean()), 4),
                    "moment_sd": round(float(mom.std(ddof=1)), 4),
                    "moment_p10": round(float(np.percentile(mom, 10)), 4),
                    "moment_p90": round(float(np.percentile(mom, 90)), 4),
                    "moment_clipped_share": round(
                        clipped / TASTE_RECOVERY_REPS, 4),
                    "rate_mean": round(float(rat.mean()), 4),
                    "rate_sd": round(float(rat.std(ddof=1)), 4),
                    "rate_p10": round(float(np.percentile(rat, 10)), 4),
                    "rate_p90": round(float(np.percentile(rat, 90)), 4)})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(root, "taste_recovery.csv"), index=False)
    print(f"\n  {TASTE_RECOVERY_REPS} simulated control groups per "
          "cell; both estimates in ABSOLUTE")
    print("  weight units, against the true spread:")
    print(tab[["feature", "true_spread_in_thresholds",
               "true_spread_abs", "n_controls", "moment_mean",
               "moment_sd", "moment_clipped_share", "rate_mean",
               "rate_sd"]].to_string(index=False))

    # --- can the controls tell 0.5x from 1.0x? That is the decision ---
    print("\n  THE QUESTION THAT MATTERS: could a control group "
          "distinguish a 0.5x spread")
    print("  from a 1.0x one? That is what a rule registered as "
          "admissible up to 0.5x")
    print("  needs, since it would be downgraded above it.")
    for est in ("moment", "rate"):
        print(f"\n    {est} estimator:")
        for feat in ("colour", "fog"):
            for n_c in TASTE_RECOVERY_N:
                a = tab[(tab.feature == feat) & (tab.n_controls == n_c)
                        & np.isclose(tab.true_spread_in_thresholds, 0.5)]
                b = tab[(tab.feature == feat) & (tab.n_controls == n_c)
                        & np.isclose(tab.true_spread_in_thresholds, 1.0)]
                if not (len(a) and len(b)):
                    continue
                m0, s0 = a[f"{est}_mean"].iloc[0], a[f"{est}_sd"].iloc[0]
                m1, s1 = b[f"{est}_mean"].iloc[0], b[f"{est}_sd"].iloc[0]
                sep = ((m1 - m0) / np.sqrt(s0 ** 2 + s1 ** 2)
                       if (s0 or s1) else np.inf)
                # The decision boundary a pre-registered rule would
                # use: midway between the two spreads in absolute
                # units.
                U_k = U[0] if feat == "colour" else U[1]
                bound = 0.75 * U_k
                p_wrong_low = float(stats.norm.sf(bound, m0, s0)) \
                    if s0 > 0 else float(m0 > bound)
                p_wrong_high = float(stats.norm.cdf(bound, m1, s1)) \
                    if s1 > 0 else float(m1 < bound)
                print(f"      {feat:7s} n={n_c}: 0.5x -> "
                      f"{m0:.3f}+-{s0:.3f}, 1.0x -> {m1:.3f}+-{s1:.3f}"
                      f"   separation {sep:.2f} SD")
                print(f"        at a boundary of 0.75x "
                      f"({bound:.3f}): P(call 0.5x high) "
                      f"{p_wrong_low:.2f}, "
                      f"P(call 1.0x low) {p_wrong_high:.2f}")
    best = tab.groupby("n_controls")[["moment_sd", "rate_sd"]].mean()
    print("\n  mean sampling SD of each estimate, by control group "
          "size:")
    print(best.round(4).to_string())
    print("\n  WHAT THIS MEANS FOR THE REGISTERED PLAN. The estimate "
          "is usable for a")
    print("  coarse verdict -- 'the controls are not wildly more "
          "variable than assumed' --")
    print("  and the numbers above say how coarse. It is NOT precise "
          "enough at 20 or 30")
    print("  controls to pin the spread to better than about a factor "
          "of two, so a")
    print("  downgrade rule should be stated against a wide boundary "
          "and expected to")
    print("  misfire sometimes in both directions, at the rates "
          "printed above.")
    print()
    print("  ONE HOLE, STATED: if controls also use the 'more edited "
          "image' strategy,")
    print("  their weights are not zero-centred and both estimators "
          "inflate. The")
    print("  asymmetry between predicted and opposite classifications "
          "is what would")
    print("  reveal it -- taste is symmetric, the strategy is not -- "
          "and C1 sweeps the")
    print("  strategy rate. Any downgrade rule should test that "
          "asymmetry first.")
    report.append(("taste spread recovery, mean sampling SD",
                   ", ".join(
                       f"n={int(i)}: moment {r.moment_sd:.3f} / rate "
                       f"{r.rate_sd:.3f}"
                       for i, r in best.iterrows())))
    return tab


def prevalence_posterior(k, n, alpha, sensitivity, grid=None):
    """Bayesian prevalence (Ince, Paton, Kay & Schyns, eLife 2021).

    k of n participants tested positive individually, at a
    within-participant false-positive rate `alpha` and sensitivity
    `sensitivity` = 1 - beta. A participant tests positive with
    probability

        theta(gamma) = gamma * sensitivity + (1 - gamma) * alpha

    so the count is Binomial(n, theta(gamma)) and, with a uniform
    prior on gamma in [0, 1], the posterior is proportional to that
    binomial likelihood evaluated along gamma.

    THE LIKELIHOOD IS WRITTEN OUT so the implementation can be checked
    against the paper rather than taken on trust.

    Why this is worth having beside a fixed rule: a count rule answers
    "do we claim the effect" and jumps when one more person crosses a
    threshold. This answers "in what share of the population", which
    is both the more useful quantity and a smooth one. alpha here is
    FLOOR_ALPHA -- 2.5% by construction of the upper threshold, not an
    assumption -- and the sensitivity comes from this script's own
    detection curve, which is the part only a simulation can supply.
    """
    g = (np.linspace(0.0, 1.0, 201) if grid is None
         else np.asarray(grid, dtype=float))
    theta = g * sensitivity + (1.0 - g) * alpha
    theta = np.clip(theta, 1e-12, 1 - 1e-12)
    logl = k * np.log(theta) + (n - k) * np.log1p(-theta)
    post = np.exp(logl - logl.max())
    area = np.trapezoid(post, g) if hasattr(np, "trapezoid") \
        else np.trapz(post, g)
    post = post / area
    return g, post


def prevalence_posterior_shared(a, b, n, sensitivity,
                                n_grid=None, c_max=0.5):
    """Posterior on the prevalence from BOTH directions' counts.

    THE REGISTERED PREVALENCE ANALYSIS. Takes the number classified in
    the predicted direction (a) and in the opposite direction (b) out
    of n people, and returns a posterior on the prevalence that
    assumes NOTHING about the false-positive rate.

    The model is the one in `prevalence_posterior`, written in the
    parameters the data can see. Let

        c = (1 - gamma) * alpha

    be the rate at which a person is classified in the OPPOSITE
    direction: only non-responders do that, at their false-positive
    rate per tail. Under any symmetric null they are classified
    predicted at the same rate, so

        p_pred = gamma * sensitivity + c
        p_opp  =                      c
        p_none = 1 - p_pred - p_opp

    and (a, b, n - a - b) is trinomial in those. The posterior on
    gamma is this likelihood with a flat prior on both parameters,
    marginalised over c.

    WHY NOT PARAMETERISE BY alpha. Because (gamma, alpha) are not
    separately identified at these sample sizes: p_opp carries a
    (1 - gamma) factor, so gamma near 1 drives p_opp to zero, which
    fits an observed b of zero perfectly while also maximising p_pred.
    With a true prevalence of zero and a 1x taste spread that
    parameterisation returned a posterior mode of 0.86. In (gamma, c)
    the opposite count pins c down on its own and the ridge is gone.

    b = 0 is handled explicitly rather than by a floor: c = 0 is a
    perfectly admissible parameter value when nothing was seen in the
    opposite direction, and dropping the b * log(c) term is what the
    trinomial likelihood actually says there. The old variant instead
    floored its estimate at 1e-6, which made the posterior behave as
    though false positives were impossible.
    """
    ng = PREVALENCE_N_GRID if n_grid is None else n_grid
    g = np.linspace(0.0, 1.0, ng)[:, None]
    c = np.linspace(0.0, c_max, PREVALENCE_C_GRID)[None, :]
    p_pred = g * sensitivity + c
    rest = 1.0 - p_pred - c
    ok = (p_pred > 0) & (rest > 0)
    if b:
        ok = ok & (c > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = (a * np.log(np.where(p_pred > 0, p_pred, 1.0))
              + (n - a - b) * np.log(np.where(rest > 0, rest, 1.0)))
        if b:
            ll = ll + b * np.log(np.where(c > 0, c, 1.0))
    ll = np.where(ok, ll, -np.inf)
    fin = np.isfinite(ll)
    if not fin.any():
        gg = g.ravel()
        return gg, np.full(ng, 1.0)
    L = np.where(fin, np.exp(ll - ll[fin].max()), 0.0)
    post = L.sum(axis=1)          # marginalise c, flat prior on it
    gg = g.ravel()
    # NORMALISED BY AREA, NOT BY SUM -- a density, which is what
    # `hpdi` expects (it weights each point by the grid step) and what
    # `prevalence_posterior` returns. Normalising to sum 1 instead
    # makes every interval come back as the whole of [0, 1], because
    # the accumulated mass never reaches the target.
    area = np.trapezoid(post, gg) if hasattr(np, "trapezoid") \
        else np.trapz(post, gg)
    if not (area > 0) or not np.isfinite(area):
        return gg, np.full(ng, 1.0)
    return gg, post / area


def prevalence_moment_raw(a, b, n, sensitivity):
    """The moment estimator BEFORE clipping, which is the unbiased
    one. Reported beside the clipped version so the unbiasedness is
    checkable rather than asserted."""
    if sensitivity <= 0:
        return np.nan
    return float((a - b) / (n * sensitivity))


def prevalence_moment(a, b, n, sensitivity):
    """The moment estimator of the prevalence. Needs no alpha.

    E[a - b] = n * gamma * sensitivity, because the non-responders'
    contributions to the two counts have the same expectation and
    cancel, and a responder almost never lands opposite. So

        gamma_hat = (a - b) / (n * sensitivity)

    is exactly unbiased before clipping. It is clipped to [0, 1]
    because a prevalence is a proportion, and that clip is the only
    reason the reported mean sits above zero when the truth is zero.
    """
    if sensitivity <= 0:
        return np.nan
    return float(min(max((a - b) / (n * sensitivity), 0.0), 1.0))


def hpdi(g, post, mass=PREVALENCE_HPDI):
    """The narrowest interval holding `mass` of the posterior."""
    order = np.argsort(post)[::-1]
    dg = g[1] - g[0]
    keep, acc = np.zeros(len(g), dtype=bool), 0.0
    for i in order:
        keep[i] = True
        acc += post[i] * dg
        if acc >= mass:
            break
    idx = np.where(keep)[0]
    return float(g[idx.min()]), float(g[idx.max()])


def _series_style(i):
    """Colour, marker and line style for series i, by identity.

    Never cycled past the end: a fifth series is a signal to facet or
    drop one, not to invent a hue.
    """
    n = len(SERIES_COLOURS)
    if i >= n:
        raise ValueError(
            f"series {i} has no assigned colour: only {n} are "
            "defined. Facet the figure or drop a series rather than "
            "generating a hue.")
    return dict(color=SERIES_COLOURS[i], marker=SERIES_MARKERS[i],
                ls=SERIES_STYLES[i], lw=2.0, markersize=6)


def plot_existence_power(rules, X, out_dir, report):
    """(a) The existence claim against N, by responder share.

    Under the REGISTERED standard, at the most powerful rule that
    standard admits for each sample size -- so the curve is what the
    registered analysis would actually achieve, not the best rule in
    the sweep.

    One panel per feature, because H1 and H2 have different
    thresholds and a single panel would invite reading a gap between
    them that is arithmetic rather than asymmetry. The behavioural
    figure makes that comparison properly.
    """
    pw, _, plam = RULE_SUMMARY_PICK
    ok = rules[rules[f"admissible_{REGISTERED_STANDARD}"]]
    if not len(ok):
        print("  no rules admissible under the registered standard; "
              "existence_power.png skipped")
        return None
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    # H1 on the left: the labels are H1 = fog, H2 = colour throughout.
    for ax, feat in zip(axes, ("fog", "colour")):
        claim = "H2 colour" if feat == "colour" else "H1 fog"
        for i, share in enumerate(RULE_RESPONDER_SHARES):
            col = f"power_w{pw:g}_s{share:g}_lapse{plam:g}"
            if col not in ok.columns:
                continue
            xs, ys = [], []
            for n in sorted(ok[ok.feature == feat].n.unique()):
                g = ok[(ok.feature == feat) & (ok.n == n)]
                if not len(g):
                    continue
                best = g.loc[g[col].idxmax()]
                xs.append(n)
                ys.append(float(best[col]))
                rows.append({"feature": feat, "claim": claim,
                             "n": n, "responder_share": share,
                             "effect_per_SD": pw, "lapse": plam,
                             "standard": REGISTERED_STANDARD,
                             "family": best.family,
                             "k": int(best.k), "m": int(best.m),
                             "power": round(float(best[col]), 4)})
            if xs:
                ax.plot(xs, ys, label=f"{share:.0%} respond",
                        **_series_style(i))
        ax.axhline(0.8, color="#444444", lw=0.8, ls="--")
        ax.set_xlabel("participants")
        ax.set_title(claim)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_ylim(0, 1.02)
        # LOG SCALE, and the sizes actually simulated as the ticks.
        # On a linear axis six of the thirteen sample sizes fall below
        # 17 and their labels overlap into illegibility, while the
        # curve's whole shape happens there. A log axis spreads the
        # small sizes and compresses the flat tail.
        sizes = sorted(ok[ok.feature == feat].n.unique())
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(int(v)) for v in sizes], fontsize=7)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis="x", which="minor", length=0)
    axes[0].set_ylabel("P(the existence claim fires)")
    # No label on the 80% line: the y axis already carries 0.8, and
    # any text at the left edge lands under the legend.
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")
    med = informative_medians(X)
    fig.suptitle(
        f"(a) The existence claim against sample size, by responder "
        f"share\nregistered standard: {REGISTERED_STANDARD}; "
        f"responders carry {pw:g} per SD "
        f"(p = {as_probability(pw, med['colour']):.2f} colour, "
        f"{as_probability(pw, med['fog']):.2f} fog on a typical "
        f"informative trial); lapse {plam:g}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "existence_power.png"),
                dpi=FIG_DPI)
    plt.close(fig)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out_dir, "existence_power.csv"),
               index=False)
    hi = tab[np.isclose(tab.responder_share, 1.0)]
    lo = tab[tab.responder_share <= 0.25]
    if len(hi) and len(lo):
        report.append(
            ("existence claim power, registered standard",
             f"{hi.power.min():.2f}-{hi.power.max():.2f} at full "
             f"responder share, {lo.power.min():.2f}-"
             f"{lo.power.max():.2f} at 25% or less"))
    return tab


def plot_c1_power(c1, out_dir, report):
    """(b) C1 against N, by responder share and strategy rate.

    Matched lapse rates and 30 controls, so the gate is quiet and the
    primary C1 is the analysis being drawn. The strategy rate is on
    the figure because it is the parameter that decides whether C1 is
    worth running, and leaving it off a power curve is how a power
    figure becomes misleading.
    """
    if c1 is None or not len(c1):
        return None
    eff = sorted(w for w in c1.effect_sd.unique() if w > 0)
    if not eff:
        return None
    w = eff[len(eff) // 2]
    d = c1[(np.isclose(c1.effect_sd, w))
           & (c1.lapse_participants == c1.lapse_controls)
           & (c1.lapse_participants == 0.0)
           & (c1.n_controls == 30)]
    if not len(d):
        return None
    # FACETED BY RESPONDER SHARE rather than drawing share x
    # strategy as one legend: on the real settings that is six
    # series against four assigned colours, and a fifth colour is
    # never generated. Panels are feature x share; the series within
    # a panel are the strategy rates, which is also the comparison
    # the figure exists to make.
    shares = sorted(d.responder_share.unique())
    claims = ("H1 fog", "H2 colour")
    fig, axes = plt.subplots(len(shares), 2,
                             figsize=(13, 4.6 * len(shares)),
                             sharey=True, sharex=True, squeeze=False)
    sizes = sorted(d.n_participants.unique())
    for r, share in enumerate(shares):
        for c, claim in enumerate(claims):
            ax = axes[r][c]
            g0 = d[(d.effect_on == claim)
                   & (d.responder_share == share)]
            for i, strat in enumerate(sorted(g0.strategy_rate
                                             .unique())):
                g = g0[g0.strategy_rate == strat].sort_values(
                    "n_participants")
                if not len(g):
                    continue
                ax.plot(g.n_participants, g.power_either_gated,
                        label=f"{strat:.0%} use the strategy",
                        **_series_style(i))
            ax.axhline(0.8, color="#444444", lw=0.8, ls="--")
            ax.set_title(f"{claim} -- {share:.0%} of participants "
                         "respond", fontsize=10)
            ax.grid(alpha=0.25, lw=0.6)
            ax.set_ylim(0, 1.02)
            ax.set_xticks(sizes)
            ax.set_xticklabels([str(int(v)) for v in sizes],
                               fontsize=8)
            if c == 0:
                ax.set_ylabel("P(C1 fires on either feature)")
            if r == len(shares) - 1:
                ax.set_xlabel("participants (against 30 controls)")
    axes[0][0].legend(fontsize=8, frameon=False)
    p = d.effect_p_typical.iloc[0]
    fig.suptitle(
        f"(b) C1, the group comparison, against sample size\n"
        f"responders carry {w:g} per SD (p = {p:.2f} on a typical "
        f"informative trial); matched lapse rates, so the gate is "
        f"quiet; 30 controls\n'the strategy' is picking the more "
        f"edited image -- present in BOTH groups at the same rate")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "c1_power.png"), dpi=FIG_DPI)
    plt.close(fig)
    full = d[np.isclose(d.responder_share, 1.0)
             & np.isclose(d.strategy_rate, 0.0)]
    half = d[(d.responder_share <= 0.5) & (d.strategy_rate >= 0.25)]
    if len(full) and len(half):
        report.append(
            ("C1 power depends on share and strategy",
             f"{full.power_either_gated.min():.2f}-"
             f"{full.power_either_gated.max():.2f} at full share with "
             f"no strategy, {half.power_either_gated.min():.2f}-"
             f"{half.power_either_gated.max():.2f} at half share with "
             "a quarter using the strategy"))
    return d


def plot_equal_behaviour(eq, out_dir, report):
    """(c) H1 against H2 at equal behaviour.

    The two features' claims paired at the SAME choice probability on
    a typical informative trial, which is the only way they are
    comparable: a weight per corpus SD means a different amount of
    behaviour on each feature.

    Drawn as H1 against H2 with the identity line, so equivalence is
    the diagonal rather than something to infer from two curves. Every
    point off the diagonal is a cell where the two claims differ, and
    the diverging cells are labelled.
    """
    if eq is None or not len(eq):
        return None
    keys = ["p_typical_trial", "n", "share", "lapse", "family"]
    keys = [k for k in keys if k in eq.columns]
    piv = eq.pivot_table(index=keys, columns="feature",
                         values="power").dropna()
    if not len(piv) or not {"colour", "fog"} <= set(piv.columns):
        return None
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot([0, 1], [0, 1], color="#444444", lw=0.8, ls="--",
            label="equal power (the claim)")
    gap = (piv.fog - piv.colour).abs()
    near = gap <= EQUAL_BEHAVIOUR_TOL
    ax.scatter(piv.colour[near], piv.fog[near], s=26,
               color=SERIES_COLOURS[0], alpha=0.7,
               edgecolor="white", linewidth=0.5,
               label=f"within {EQUAL_BEHAVIOUR_TOL:.2f} "
                     f"({int(near.sum())} cells)")
    if (~near).any():
        ax.scatter(piv.colour[~near], piv.fog[~near], s=46,
                   color=SERIES_COLOURS[2], marker="^",
                   edgecolor="white", linewidth=0.5,
                   label=f"differ by more ({int((~near).sum())} cells)")
    ax.set_xlabel("P(H2 colour fires)")
    ax.set_ylabel("P(H1 fog fires)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle("(c) H1 against H2 at EQUAL BEHAVIOUR\n"
                 "each point is one scenario, with both claims set to "
                 "the same choice probability\non a typical "
                 f"informative trial; median gap "
                 f"{float(gap.median()):.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "equal_behaviour.png"),
                dpi=FIG_DPI)
    plt.close(fig)
    return piv


def plot_gate(gate, out_dir, report):
    """(d) The attention gate: false alarm and sensitivity.

    Two panels on one axis each, never two scales on one panel. The
    left is the cost of the gate -- how often it fires when the groups
    are matched, which makes C1 inconclusive for nothing. The right is
    what it buys: how often it fires when the controls really do lapse
    more, with the catch measure alone drawn beside the combined gate,
    because response times might not be informative and the catch
    measure is what the gate rests on if they are not.
    """
    if gate is None or not len(gate):
        return None
    g = gate.copy()
    g["size"] = (g.n_participants.astype(str) + " vs "
                 + g.n_controls.astype(str))
    order = list(dict.fromkeys(g["size"]))
    xi = {s: i for i, s in enumerate(order)}
    matched = g[g.lapse_participants == g.lapse_controls]
    worse = g[g.lapse_controls > g.lapse_participants]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    for i, lam in enumerate(sorted(matched.lapse_participants.unique())):
        d = matched[matched.lapse_participants == lam]
        d = d.groupby("size", sort=False).fires.max().reindex(order)
        ax.plot([xi[s] for s in order], d.to_numpy(),
                label=f"both groups lapse {lam:g}", **_series_style(i))
    ax.axhline(GATE_ALPHA, color="#444444", lw=0.8, ls="--")
    ax.text(len(order) - 1, GATE_ALPHA + 0.002,
            f"nominal {GATE_ALPHA}", ha="right", va="bottom",
            fontsize=8, color="#444444")
    ax.set_ylabel("P(the gate fires)")
    ax.set_title("cost: the groups are MATCHED\n"
                 "(a false alarm makes C1 inconclusive for nothing)")
    ax.set_ylim(0, max(0.08, float(matched.fires.max()) * 1.3))

    ax = axes[1]
    for i, lam in enumerate(sorted(worse.lapse_controls.unique())):
        d = worse[worse.lapse_controls == lam]
        comb = d.groupby("size", sort=False).fires.min().reindex(order)
        only = d.groupby("size",
                         sort=False).catch_alone.min().reindex(order)
        # COLOUR FOLLOWS THE LAPSE RATE, style follows the measure.
        # The pair of lines for one lapse rate belongs together; giving
        # them different colours made colour mean the measure here and
        # the lapse rate everywhere else in the figure.
        st = _series_style(i)
        ax.plot([xi[s] for s in order], comb.to_numpy(),
                label=f"controls lapse {lam:g}, both measures", **st)
        st2 = dict(st)
        st2["ls"] = ":"
        st2["markersize"] = 4
        ax.plot([xi[s] for s in order], only.to_numpy(),
                label=f"controls lapse {lam:g}, catch trials only",
                **st2)
    ax.set_title("benefit: the CONTROLS lapse more\n"
                 "(the direction that manufactures a C1 effect)")
    ax.set_ylim(0.9, 1.005)

    for ax in axes:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("participants vs controls")
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("(d) The attention gate. Worst case over the "
                 "response-time sweep on the left,\nbest case (least "
                 "often fired) on the right -- so each panel shows "
                 "the gate at its least flattering.")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gate.png"), dpi=FIG_DPI)
    plt.close(fig)
    return g


def run_prevalence(det, rates, report, root):
    """Prevalence as the alternative to a fixed count rule.

    A count rule answers "do we claim the feature" and jumps when one
    more person crosses a threshold. This answers "in what share of
    the population", which is the more useful quantity and a smooth
    one.

    THE ASSUMPTION THAT MATTERS, AND WHY IT IS TESTED HERE. The
    posterior needs the within-participant false-positive rate alpha.
    Taking it as FLOOR_ALPHA = 2.5% is correct ONLY IF the
    non-responders are pure guessers. They are not: a non-responder
    with aesthetic taste is classified predicted up to 18% of the time
    on this design. Feed 2.5% into the posterior when the truth is 18%
    and the prevalence is overstated, because every taste-driven
    detection is credited to the effect.

    So three variants are computed:

      assumed-2.5%   alpha = FLOOR_ALPHA. The guesser-only assumption,
                     and the one the method is usually used with.
      assumed-taste  alpha = that taste spread's measured predicted
                     rate. What the answer SHOULD be if the population
                     has that much taste -- not available in practice,
                     since the spread is unmeasurable, but it bounds
                     the error.
      estimated      THE REGISTERED VERSION. alpha estimated FROM
                     THE DATA, as the share of the sample classified
                     OPPOSITE. Under any symmetric null the two
                     directions are equally likely, so the opposite
                     count is an unbiased estimate of the
                     false-positive rate -- and it needs no assumption
                     about the taste spread at all, which is the whole
                     point, since the spread cannot be measured in
                     advance. The same insight the sign test rests on,
                     used to calibrate rather than to test.

    THE 2.5%-ASSUMED VARIANT IS KEPT ONLY TO SHOW THE DISTORTION. It
    is what the method returns if the non-responders are taken to be
    pure guessers, and comparing it with the taste-aware variants is
    how the cost of that assumption becomes a number instead of a
    worry. It is not a candidate for registration.

    The estimated variant is noisy at small n -- an expectation of
    0.025 x 10 = 0.25 opposite classifications carries no information
    -- so its cost is reported rather than hidden.
    """
    print("\n" + "=" * 60)
    print("BAYESIAN PREVALENCE  (Ince et al., 2021)")
    print("=" * 60)
    base = det[np.isclose(det.lapse, 0.0)] if "lapse" in det.columns \
        else det

    # The measured per-person rates per null, for the taste variants.
    taste_rate = {}
    for _, r in rates.iterrows():
        if r.null == "taste":
            # Symmetric by construction, so the average of the two
            # measured rates is the better estimate of either.
            taste_rate[(float(r.spread_in_floors), r.feature)] = 0.5 * (
                float(r.p_predicted) + float(r.p_opposite))

    rows = []
    for feat in ("colour", "fog"):
        for w in RULE_RESPONDER_WEIGHTS:
            sens = float(np.interp(w, base.true_weight.to_numpy(),
                                   base[feat].to_numpy()))
            variants = [("assumed_2.5pc", FLOOR_ALPHA)]
            for mult in TASTE_SPREADS_IN_FLOORS:
                variants.append((f"assumed_taste_{mult:g}x",
                                 taste_rate[(mult, feat)]))
            for n in SAMPLE_SIZES:
                for k in range(0, n + 1):
                    for vname, alpha in variants:
                        if alpha >= sens:
                            # A false-positive rate at or above the
                            # sensitivity makes the design
                            # uninformative about prevalence: a
                            # positive is then no more likely from a
                            # responder than from a non-responder.
                            rows.append({
                                "feature": feat, "effect_sd": w,
                                "variant": vname, "n": n,
                                "k_positive": k,
                                "alpha": round(alpha, 4),
                                "sensitivity": round(sens, 4),
                                "posterior_mode": np.nan,
                                "hpdi_lo": np.nan, "hpdi_hi": np.nan,
                                "informative": False})
                            continue
                        g, post = prevalence_posterior(
                            k, n, alpha, sens, PREVALENCE_GRID)
                        lo, hi = hpdi(g, post)
                        rows.append({
                            "feature": feat, "effect_sd": w,
                            "variant": vname, "n": n, "k_positive": k,
                            "alpha": round(alpha, 4),
                            "sensitivity": round(sens, 4),
                            "posterior_mode": round(
                                float(g[int(np.argmax(post))]), 4),
                            "hpdi_lo": round(lo, 4),
                            "hpdi_hi": round(hi, 4),
                            "informative": True})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(root, "prevalence.csv"), index=False)

    print(f"  within-participant false-positive rate, per variant:")
    for feat in ("colour", "fog"):
        g = tab[tab.feature == feat]
        line = ", ".join(
            f"{v} {g[g.variant == v].alpha.iloc[0]:.3f}"
            for v in g.variant.unique())
        print(f"    {feat:7s} {line}")
    print("\n  sensitivity, off this script's own detection curve:")
    for feat in ("colour", "fog"):
        for w in RULE_RESPONDER_WEIGHTS:
            s = tab[(tab.feature == feat)
                    & np.isclose(tab.effect_sd, w)].sensitivity.iloc[0]
            print(f"    {feat:7s} effect {w:.2f}/SD -> {s:.3f}")

    print("\n  HOW MUCH TASTE DISTORTS THE ANSWER. Posterior mode at "
          "the k you would")
    print(f"  EXPECT to observe if the truth were "
          f"{RULE_SUMMARY_PICK[1]:.0%} prevalence with that much "
          "taste,")
    print(f"  at effect {RULE_SUMMARY_PICK[0]} per SD. (At k = half "
          "the sample every variant")
    print("  saturates near 1.0, because observing half the sample "
          "positive when the")
    print("  sensitivity is only ~0.5 already implies almost everyone "
          "has the effect --")
    print("  so that cell cannot show the distortion.)")
    w0, share0 = RULE_SUMMARY_PICK[0], RULE_SUMMARY_PICK[1]
    for feat in ("colour", "fog"):
        sens_f = tab[(tab.feature == feat)
                     & np.isclose(tab.effect_sd, w0)].sensitivity.iloc[0]
        for n in [x for x in RULE_SUMMARY_N if x in SAMPLE_SIZES]:
            # The count a sample like that would actually produce.
            a_t = taste_rate[(NONRESPONDER_SPREAD_MULT, feat)]
            k = int(round(n * (share0 * sens_f
                               + (1 - share0) * a_t)))
            k = min(max(k, 0), n)
            cells = []
            for v in tab.variant.unique():
                r = tab[(tab.feature == feat) & (tab.variant == v)
                        & (tab.n == n) & (tab.k_positive == k)
                        & np.isclose(tab.effect_sd, w0)]
                if len(r):
                    m = r.posterior_mode.iloc[0]
                    cells.append(f"{v.replace('assumed_', '')}: "
                                 + ("n/a" if not np.isfinite(m)
                                    else f"{m:.2f}"))
            print(f"    {feat:7s} n={n:3d} k={k:2d}   "
                  + "   ".join(cells))
    print("  Assuming 2.5% when the population has taste OVERSTATES "
          "prevalence, because")
    print("  every taste-driven detection is credited to the effect. "
          "The spread cannot")
    print("  be measured, so the honest options are the estimated "
          "variant below or")
    print("  reporting the whole range.")

    print("\n  REGISTERED: the estimated-alpha variant below. The "
          "variants above are")
    print("  reported to show what assuming 2.5% would cost, not as "
          "candidates.")
    est = run_prevalence_estimated(det, rates, report, root)
    report.append(("prevalence variants",
                   f"{tab.variant.nunique()} per feature, "
                   f"alpha {tab.alpha.min():.3f}-{tab.alpha.max():.3f}"))
    return tab, est


def run_prevalence_estimated(det, rates, report, root):
    """Prevalence from BOTH directions' counts, assuming no alpha.

    THE REGISTERED VARIANT. The fixed-alpha versions above each have
    to name a false-positive rate, and the right one depends on the
    taste spread, which is the one quantity nobody can measure in
    advance. This one reads it off the data.

    Under any symmetric null a non-responder is as likely to be
    classified in the opposite direction as in the predicted one, so
    the OPPOSITE count measures the contamination directly. A
    responder contributes almost nothing to it -- a real preference
    does not also look backwards, measured at 0.0000 on this design.

    WHAT CHANGED, AND WHY IT MATTERS. An earlier version of this
    section estimated alpha as (opposite count) / n and fed it to the
    one-parameter posterior. That estimate is biased LOW by exactly
    the factor (1 - gamma), because only the non-responders generate
    opposite classifications while the denominator counts everyone --
    and (1 - gamma) is the very quantity being estimated. The fix is
    to work in the rate the data can see, c = (1 - gamma) * alpha,
    which needs no correction and is separately identified from the
    prevalence. See prevalence_posterior_shared.

    Both are reported: the registered estimator, and the biased one it
    replaced, so the registration records what was rejected and why.
    """
    print("\n" + "-" * 60)
    print("  PREVALENCE FROM BOTH DIRECTIONS (no alpha assumed)")
    print("-" * 60)
    base = det[np.isclose(det.lapse, 0.0)] if "lapse" in det.columns \
        else det
    taste_rate = {}
    for _, r in rates.iterrows():
        if r.null == "taste":
            taste_rate[(float(r.spread_in_floors), r.feature)] = 0.5 * (
                float(r.p_predicted) + float(r.p_opposite))

    w0, share0, _ = RULE_SUMMARY_PICK
    print(f"  effect {w0} per SD; true prevalence swept over "
          f"{PREVALENCE_TRUE_GAMMAS}")
    print("  every cell enumerated exactly over the joint "
          "(predicted, opposite) counts")
    print("  -- no simulation, so these are the method's exact "
          "operating characteristics")
    rows = []
    for feat in ("colour", "fog"):
        sens = float(np.interp(w0, base.true_weight.to_numpy(),
                               base[feat].to_numpy()))
        for mult in TASTE_SPREADS_IN_FLOORS:
            p_t = taste_rate[(mult, feat)]
            for gamma in PREVALENCE_TRUE_GAMMAS:
                for n in [x for x in RULE_SUMMARY_N
                          if x in SAMPLE_SIZES]:
                    n_r = int(round(gamma * n))
                    n_n = n - n_r
                    gr = trinomial_grid(n_r, sens, 0.0) if n_r else None
                    gn = trinomial_grid(n_n, p_t, p_t) if n_n else None
                    if gr is None:
                        joint = gn
                    elif gn is None:
                        joint = gr
                    else:
                        joint = combine_grids(gr, gn)
                    mom = mode = lo_s = hi_s = excl = tot = 0.0
                    old_hat = raw = 0.0
                    for aa in range(joint.shape[0]):
                        for bb in range(joint.shape[1]):
                            wgt = float(joint[aa, bb])
                            if wgt < 1e-13:
                                continue
                            tot += wgt
                            old_hat += wgt * (bb / n)
                            mom += wgt * prevalence_moment(
                                aa, bb, n, sens)
                            raw += wgt * prevalence_moment_raw(
                                aa, bb, n, sens)
                            gg, post = prevalence_posterior_shared(
                                aa, bb, n, sens)
                            lo, hi = hpdi(gg, post)
                            mode += wgt * float(
                                gg[int(np.argmax(post))])
                            lo_s += wgt * lo
                            hi_s += wgt * hi
                            excl += wgt * (lo > PREVALENCE_ZERO_EPS)
                    rows.append({
                        "feature": feat, "effect_sd": w0,
                        "true_prevalence": gamma,
                        # The SIMULATED sample has an integer number
                        # of responders, so the prevalence it actually
                        # contains is n_r / n. At gamma 0.25 and
                        # n = 10 that is 0.20, and scoring the
                        # estimator against 0.25 would charge it 0.05
                        # of bias that belongs to the rounding.
                        "realised_prevalence": round(n_r / n, 4),
                        "taste_spread": mult,
                        "alpha_true": round(p_t, 4),
                        "c_true": round((1 - gamma) * p_t, 4),
                        "n": n,
                        "E_moment_raw": round(raw / tot, 4),
                        "E_moment": round(mom / tot, 4),
                        "E_mode": round(mode / tot, 4),
                        "E_hpdi_lo": round(lo_s / tot, 4),
                        "E_hpdi_hi": round(hi_s / tot, 4),
                        "P_excludes_zero": round(excl / tot, 4),
                        "E_old_alpha_hat": round(old_hat / tot, 4),
                        "old_bias_factor": round(
                            (old_hat / tot) / p_t, 4) if p_t else
                        np.nan})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(root, "prevalence_estimated.csv"),
               index=False)
    print()
    print(tab.to_string(index=False))

    # --- the moment estimator's unbiasedness, stated as a check ---
    print("\n  (a) THE MOMENT ESTIMATOR, gamma_hat = "
          "(predicted - opposite) / (n x sensitivity)")
    piv = tab.pivot_table(index=["realised_prevalence"],
                          columns=["feature", "n"],
                          values=["E_moment_raw", "E_moment"])
    print("  " + piv.round(3).to_string().replace("\n", "\n  "))
    worst_raw = float((tab.E_moment_raw
                       - tab.realised_prevalence).abs().max())
    worst_clip = float((tab.E_moment
                        - tab.realised_prevalence).abs().max())
    print(f"\n    UNCLIPPED, the estimator is unbiased: worst "
          f"deviation {worst_raw:.4f} over every")
    print("    cell in the table, including the zero-prevalence ones. "
          "That is the check")
    print("    on the algebra, and it is why no false-positive rate "
          "has to be assumed.")
    print(f"\n    CLIPPED to [0, 1], the worst deviation is "
          f"{worst_clip:.4f}. The clip is correct -- a")
    print("    prevalence is a proportion -- but it is not free, and "
          "it does not only")
    print("    affect the zero row: it inflates SMALL non-zero "
          "prevalences too, because")
    print("    (predicted - opposite) is negative in a good fraction "
          "of studies when the")
    print("    true prevalence is low and the taste spread is large. "
          "Where that matters:")
    w = tab.assign(
        dev=(tab.E_moment - tab.realised_prevalence).abs())
    print("  " + w.pivot_table(index=["realised_prevalence"],
                               columns=["feature", "taste_spread"],
                               values="dev", aggfunc="max")
          .round(3).to_string().replace("\n", "\n  "))
    print("      (worst |E_moment - realised prevalence|, by feature "
          "and taste spread)")
    print("\n    So the POSTERIOR, not the moment estimator, is what "
          "the claim is made")
    print("    from: it carries its own uncertainty instead of "
          "needing a clip.")
    report.append(("prevalence moment estimator",
                   f"unbiased unclipped (worst {worst_raw:.4f}); "
                   f"clipped worst {worst_clip:.4f} at a low "
                   "prevalence with a wide taste spread"))

    # --- the posterior's false-positive rate ---
    zero = tab[np.isclose(tab.true_prevalence, 0.0)]
    print("\n  (b) HOW FAR THE INTERVAL CAN BE TRUSTED")
    if PREVALENCE_DESCRIPTIVE:
        print("      PREVALENCE IS REGISTERED AS DESCRIPTIVE ONLY, "
              "for both features: the")
        print("      estimator and its interval are reported and NO "
              "'excludes zero'")
        print("      inference is drawn. The rates below are "
              "therefore not a test's size.")
        print("      They are the guide to how far the interval can "
              "be trusted under each")
        print("      assumption about the null -- which is what a "
              "reader of a descriptive")
        print("      estimate needs, and the reason the table stays "
              "in the report.")
    else:
        print("      (the prevalence claim's own false-positive rate, "
              "with NO alpha assumed)")
    print("  " + zero.pivot_table(index=["feature", "taste_spread"],
                                  columns="n",
                                  values="P_excludes_zero")
          .round(3).to_string().replace("\n", "\n  "))
    print("\n    IT IS NOT UNIFORM, AND THE DIFFERENCE IS BETWEEN "
          "THE TWO FEATURES.")
    print("    Worst cell per feature:")
    per = zero.groupby("feature").apply(
        lambda d: d.loc[d.P_excludes_zero.idxmax()], include_groups=False)
    for feat, row in per.iterrows():
        flag = "" if row.P_excludes_zero <= ALPHA else "   <-- ABOVE "
        flag += "" if row.P_excludes_zero <= ALPHA else f"alpha = {ALPHA}"
        print(f"      {feat:7s} {row.P_excludes_zero:.3f} at taste "
              f"spread {row.taste_spread:g}x, n = {int(row.n)}"
              f"{flag}")
    print()
    print("    The mechanism is the design, not the estimator. The "
          "method works by")
    print("    subtracting the opposite count from the predicted one "
          "and dividing by the")
    print("    sensitivity. Fog's informative trials are about 0.62 "
          "corpus SD apart")
    print("    against colour's 1.0, so at a given weight fog's "
          "sensitivity is lower and")
    print("    its taste-null rate higher: the same subtraction is "
          "divided by a smaller")
    print("    number and starts from a noisier pair of counts.")
    print()
    if PREVALENCE_DESCRIPTIVE:
        print("\n    THIS IS WHY PREVALENCE IS DESCRIPTIVE AND NOT A "
              "TEST. Registering an")
        print("    'excludes zero' claim would mean registering a "
              "rate that depends on an")
        print("    assumption about the null that cannot be checked "
              "in advance -- and for")
        print("    fog it is several times alpha under the widest "
              "null considered. The")
        print("    estimator and its interval are reported for both "
              "features; the table")
        print("    above tells a reader how much of the interval's "
              "width to believe.")
        print("\n    The second reason is power. Even taken as a "
              "test it could not do the")
        print("    job: see the table below, where the interval "
              "excludes zero only about")
        print("    a fifth of the time at a true prevalence of 25%.")
    else:
        print("\n    THREE THINGS WOULD FIX IT, and the choice is a "
              "registration decision:")
        print("      (i)   require the interval to clear a MARGIN "
              "rather than zero.")
        print("      (ii)  report prevalence against the narrower "
              "nulls only.")
        print("      (iii) report prevalence for colour and not for "
              "fog.")

    for feat, row in per.iterrows():
        report.append(
            (f"prevalence FP under a null, {feat}",
             f"{row.P_excludes_zero:.3f} worst (spread "
             f"{row.taste_spread:g}x, n={int(row.n)}), nominal "
             f"{ALPHA}"))
    report.append(
        ("prevalence, registered form",
         "DESCRIPTIVE for both features: estimator + interval, no "
         "'excludes zero' inference (underpowered as a test, and its "
         "size is not uniform across taste nulls)"))

    # --- and the estimator this replaced ---
    print("\n  (c) THE VARIANT THIS REPLACED, and why")
    print("      It estimated alpha as (opposite count) / n. Only the "
          "NON-responders")
    print("      produce opposite classifications, so that estimate "
          "converges to")
    print("      (1 - gamma) x alpha, not alpha -- biased low by the "
          "very quantity")
    print("      being estimated:")
    print()
    bias = tab[tab.true_prevalence > 0].pivot_table(
        index=["true_prevalence", "n"], values="old_bias_factor")
    print("  " + bias.round(3).to_string().replace("\n", "\n  "))
    print("\n      Each figure is E[alpha_hat] / alpha_true. Compare "
          "it with")
    print("      (n - n_responders) / n for that cell: they agree to "
          "three decimals,")
    print("      which identifies the mechanism rather than leaving "
          "it as a discrepancy.")
    report.append(("prevalence, rejected alpha_hat bias",
                   f"E[alpha_hat]/alpha_true = "
                   f"{bias.old_bias_factor.min():.3f}-"
                   f"{bias.old_bias_factor.max():.3f}, i.e. "
                   "(1 - prevalence)"))
    return tab


# ============================================================
# C1: the group comparison  (depends on the trial list)
# ============================================================

_MW_SF_CACHE = {}


def mw_exact_sf(m, n):
    """P(U >= u) for u = 0..m*n under the exact Mann-Whitney null.

    Built by the standard dynamic programme over rank assignments:
    the last ranked observation belongs either to the first group,
    adding n to U, or to the second, adding nothing.

        count(i, j, u) = count(i-1, j, u-j) + count(i, j-1, u)

    Verified against scipy.stats.mannwhitneyu(method="exact") to
    1e-12. Precomputed and cached per (m, n) so the test below is a
    table lookup rather than a per-study computation -- which is what
    makes the exact null affordable across thousands of studies
    instead of a reason to fall back on the normal approximation at
    exactly the sample sizes where it is worst.
    """
    key = (int(m), int(n))
    if key in _MW_SF_CACHE:
        return _MW_SF_CACHE[key]
    m, n = key
    prev = np.zeros((n + 1, m * n + 1))
    prev[:, 0] = 1.0
    for i in range(1, m + 1):
        cur = np.zeros((n + 1, m * n + 1))
        cur[0, 0] = 1.0
        for j in range(1, n + 1):
            shifted = np.zeros(m * n + 1)
            shifted[j:] = prev[j, :m * n + 1 - j]
            cur[j] = shifted + cur[j - 1]
        prev = cur
    c = prev[n]
    sf = np.cumsum(c[::-1])[::-1] / c.sum()
    _MW_SF_CACHE[key] = sf
    return sf


def mw_p_batch(A, B, exact_max_n=C1_EXACT_MAX_N):
    """One-sided Mann-Whitney p-values for many studies at once.

    A is (studies, m) and B is (studies, n): the DPDR group's and the
    controls' per-person weights. The alternative is that A runs
    higher.

    U is computed from the rank sum, which for continuous weights has
    no ties, so the exact null applies wherever the sample is small
    enough to afford it and the tie-corrected normal approximation is
    never needed. Above exact_max_n the normal approximation is used,
    which is where it is accurate.
    """
    S, m = A.shape
    n = B.shape[1]
    pooled = np.concatenate([A, B], axis=1)
    # Ranks within each study: argsort of argsort, +1.
    order = np.argsort(pooled, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(S)[:, None]
    ranks[rows, order] = np.arange(1, m + n + 1)[None, :]
    R1 = ranks[:, :m].sum(axis=1)
    U = R1 - m * (m + 1) / 2.0
    N = m + n
    vs = np.take_along_axis(pooled, order, axis=1)   # sorted values
    any_ties = bool((vs[:, 1:] == vs[:, :-1]).any())
    if N <= exact_max_n and not any_ties:
        sf = mw_exact_sf(m, n)
        return sf[np.clip(U.astype(int), 0, len(sf) - 1)], "exact"
    # If ANY study in the batch has ties the whole batch goes
    # asymptotic. The exact null assumes no ties, so using it on a tied
    # study would be wrong; choosing per study would mix two nulls
    # within one reported rate, which is worse than using the
    # approximation uniformly and saying so.
    mu = m * n / 2.0
    if not any_ties:
        sd = np.sqrt(m * n * (N + 1) / 12.0)
        z = (U - mu - 0.5) / sd
        return stats.norm.sf(z), "asymptotic"

    # TIE CORRECTION. Catch-failure counts are integers 0..10, so ties
    # are the rule rather than the exception there, and the
    # uncorrected variance would overstate the spread -- making the
    # gate CONSERVATIVE, which is the wrong direction for a safety
    # check.
    #
    # Average ranks within each run of equal values, vectorised over
    # studies. For each position in the sorted row, `start` and `end`
    # are the first and last position of its run of equal values, so
    # the average rank is (start + end)/2 + 1 and the run length is
    # end - start + 1. A run of length t contributes t^3 - t to the
    # tie term, i.e. t^2 - 1 from each of its t positions -- which is
    # why the term can be summed position by position with no grouping.
    idx = np.arange(N)[None, :]
    is_first = np.empty((S, N), dtype=bool)
    is_first[:, 0] = True
    is_first[:, 1:] = vs[:, 1:] != vs[:, :-1]
    is_last = np.empty((S, N), dtype=bool)
    is_last[:, -1] = True
    is_last[:, :-1] = vs[:, 1:] != vs[:, :-1]
    start = np.maximum.accumulate(np.where(is_first, idx, -1), axis=1)
    end = np.minimum.accumulate(
        np.where(is_last, idx, N)[:, ::-1], axis=1)[:, ::-1]
    rk = (start + end) / 2.0 + 1.0
    t = end - start + 1
    tie_term = (t.astype(float) ** 2 - 1.0).sum(axis=1)

    ranks_t = np.empty((S, N), dtype=float)
    ranks_t[rows, order] = rk
    Ut = ranks_t[:, :m].sum(axis=1) - m * (m + 1) / 2.0
    var = (m * n / 12.0) * ((N + 1) - tie_term / (N * (N - 1.0)))
    sd = np.sqrt(np.maximum(var, 1e-12))
    z = (Ut - mu - 0.5) / sd
    return stats.norm.sf(z), "asymptotic"


def mann_whitney_one_sided(a, b, exact_max_n=C1_EXACT_MAX_N):
    """One-sided Mann-Whitney U: is `a` stochastically larger than `b`?

    Returns (U, p). The alternative is that the DPDR group's weights
    run higher than the controls'.

    EXACT BY ENUMERATION when the total sample is small, because the
    normal approximation is poor there and C1 may well be small. The
    exact null is the distribution of U over all ways of splitting the
    pooled ranks, which `scipy.stats.mannwhitneyu` computes with
    method="exact"; it refuses when there are ties, so the fallback is
    the tie-corrected normal approximation and the choice is recorded
    rather than silent.
    """
    n_tot = len(a) + len(b)
    want_exact = n_tot <= exact_max_n
    has_ties = len(np.unique(np.concatenate([a, b]))) < n_tot
    method = "exact" if (want_exact and not has_ties) else "asymptotic"
    r = stats.mannwhitneyu(a, b, alternative="greater", method=method)
    return float(r.statistic), float(r.pvalue), method


def holm_two(p1, p2, alpha=ALPHA):
    """Holm across exactly two one-sided tests.

    Sort the two p-values; the smaller is tested at alpha/2 and, only
    if it passes, the larger at alpha. Returns (reject1, reject2).

    Written out for two rather than called from a library because the
    whole procedure is three comparisons and the step-down logic is
    the part worth being able to read.
    """
    ps = [(p1, 0), (p2, 1)]
    ps.sort()
    rej = [False, False]
    if ps[0][0] <= alpha / 2.0:
        rej[ps[0][1]] = True
        if ps[1][0] <= alpha:
            rej[ps[1][1]] = True
    return rej[0], rej[1]


def _c1_pool(X, th, feat_idx, true_w, responder_share,
             strategy_rate, strategy_w, taste_sd, lapse_mean, rng,
             n=None):
    """A POOL of simulated people: weights AND attention measures.

    A person is built from four independent parts:

      taste       a zero-centred preference on BOTH features, the same
                  non-responder model the rule sweep uses
      strategy    with probability `strategy_rate`, an EQUAL POSITIVE
                  weight on both features -- "pick the more edited
                  image". Present in BOTH groups at the same rate,
                  because the concept is not secret.
      memory      `true_w` on the feature under test, in a
                  `responder_share` fraction of the participants only.
                  The only part that differs between the groups.
      attention   a PERSONAL lapse rate around the group mean. Not a
                  single rate for everyone, because the
                  zero-catch-failure subsample depends entirely on
                  lapse rates differing between people: if everyone in
                  a group lapsed at exactly the group rate, selecting
                  those with a clean catch record would give a random
                  subset with the SAME rate and would fix nothing.

    Returns a dict with the fitted weights, each person's own lapse
    rate, their catch-failure count and their share of implausibly
    fast responses -- all from the same people, so the gate and the
    test cannot disagree about who is in the sample.
    """
    n = C1_POOL_PEOPLE if n is None else n
    W = rng.normal(0.0, 1.0, size=(n, 2)) * taste_sd
    use = rng.random(n) < strategy_rate
    W[use, 0] += strategy_w
    W[use, 1] += strategy_w
    is_resp = rng.random(n) < responder_share
    W[is_resp, feat_idx] += true_w

    # Personal lapse rates, around the group mean. Gamma-shaped so
    # they stay non-negative and skew right, which is how inattention
    # actually distributes: most people are fine and a few are not.
    if lapse_mean <= 0:
        lam_i = np.zeros(n)
    else:
        shape = 1.0 / max(LAPSE_BETWEEN_SD, 1e-6) ** 2
        lam_i = rng.gamma(shape, lapse_mean / shape, size=n)
        lam_i = np.clip(lam_i, 0.0, 0.9)

    # Responses, each person at their OWN lapse rate.
    p = 1.0 / (1.0 + np.exp(-np.clip(W @ X.T, -500.0, 500.0)))
    p = (1.0 - lam_i[:, None]) * p + lam_i[:, None] / 2.0
    Y = (rng.random(p.shape) < p).astype(float)
    est = fit_batch(X, Y)

    # The attention measures, from the same people.
    catch = rng.binomial(N_CATCH_TRIALS,
                         np.clip(CATCH_P_FAIL_PER_LAPSE * lam_i,
                                 0.0, 1.0))
    p_fast = (lam_i * RT_FAST_GIVEN_LAPSE
              + (1.0 - lam_i) * RT_FAST_GIVEN_ENGAGED)
    fast = rng.binomial(X.shape[0], np.clip(p_fast, 0.0, 1.0)) \
        / X.shape[0]
    return {"w": est, "lapse": lam_i, "catch": catch.astype(float),
            "fast": fast, "responder": is_resp}


def _c1_gate(catch_p, catch_c, fast_p, fast_c):
    """The attention gate: are the CONTROLS worse?

    One-sided Mann-Whitney on each measure, controls expected higher
    (more catch failures, more implausibly fast responses).

    HOLM ACROSS THE TWO MEASURES, so the gate's own false-alarm rate is
    the GATE_ALPHA it claims. Taking the two at 0.05 each and firing on
    either put the gate's realised rate at 0.09 under matched lapse
    rates -- it would have declared one C1 in eleven inconclusive for
    no reason. The correction costs nothing: when the lapse rates
    really do differ, both measures fire at essentially 1.000, so the
    smaller of the two p-values is far below GATE_ALPHA/2.

    ALWAYS ASYMPTOTIC. Catch-failure counts are integers 0..10 and
    heavily tied, and the exact Mann-Whitney null assumes no ties; the
    tie-corrected normal approximation is the right test here, not a
    fallback.

    Returns (fired, fired_catch, fired_rt) per study, where the two
    per-measure flags are the Holm'd verdicts, not the uncorrected
    ones, so they sum to the gate's behaviour rather than to something
    larger than it.
    """
    p_catch, _ = mw_p_batch(catch_c, catch_p, exact_max_n=0)
    if RT_GATE:
        p_rt, _ = mw_p_batch(fast_c, fast_p, exact_max_n=0)
        lo = np.minimum(p_catch, p_rt)
        hi = np.maximum(p_catch, p_rt)
        first = lo <= GATE_ALPHA / 2.0
        second = first & (hi <= GATE_ALPHA)
        fired_c = np.where(p_catch <= p_rt, first, second)
        fired_r = np.where(p_rt < p_catch, first, second)
    else:
        fired_c = p_catch <= GATE_ALPHA
        fired_r = np.zeros_like(fired_c)
    return (fired_c | fired_r), fired_c, fired_r


def _c1_draw(pool, studies, size, rng):
    """`studies` groups of `size` people, drawn from the pool.

    WITH REPLACEMENT, which is the statistically cleaner choice here
    rather than a concession to speed. The pool is itself an i.i.d.
    sample from the person-type distribution, so drawing from it with
    replacement is an i.i.d. draw from the empirical distribution --
    a bootstrap -- and at a pool of tens of thousands against a group
    of tens, the empirical distribution is indistinguishable from the
    true one. Drawing WITHOUT replacement would instead be sampling a
    finite population, which is not what a study does.

    It is also what makes this affordable: shuffling the whole pool
    per study, which the first version did, is O(studies x pool log
    pool) and dominates the runtime by orders of magnitude.
    """
    n_pool = len(pool["w"])
    idx = rng.integers(0, n_pool, size=(studies, size))
    return {k: v[idx] for k, v in pool.items()}


def _c1_test(Ap, Ac):
    """The Holm'd pair of one-sided rank tests on the weights."""
    pv = {}
    for k2, f2 in enumerate(("colour", "fog")):
        p, _m = mw_p_batch(Ap[:, :, k2], Ac[:, :, k2])
        pv[f2] = p
    if C1_HOLM:
        p1, p2 = pv["colour"], pv["fog"]
        lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
        first = lo <= ALPHA / 2.0
        second = first & (hi <= ALPHA)
        r_c = np.where(p1 <= p2, first, second)
        r_f = np.where(p2 < p1, first, second)
    else:
        r_c = pv["colour"] <= ALPHA
        r_f = pv["fog"] <= ALPHA
    return r_c, r_f


def _c1_zero_catch(gp, gc):
    """The zero-catch-failure subsample, for every study at once.

    PRE-REGISTERED, not post-hoc matching: the rule is "keep the people
    with no catch failures, in both groups", fixed before the data
    arrive, so the subsample cannot be chosen for preserving the
    result.

    It works only because lapse rates differ BETWEEN people -- a clean
    catch record is then evidence of a low personal rate. If everyone
    in a group lapsed at the group mean, this would select a random
    subset with the same mean and achieve nothing.

    Each study keeps a different number of people, so the studies are
    GROUPED BY THEIR REALISED PAIR OF SIZES and each group is tested in
    one batched call. Studies where either subsample falls below
    C1_ZERO_MIN_N are marked unusable rather than tested, because a
    rank test on three people is not a test.

    Returns, per study: the Holm'd verdicts, the two subsample sizes,
    and the mean personal lapse rate that survives in each group.
    """
    kp = gp["catch"] == 0
    kc = gc["catch"] == 0
    S = kp.shape[0]
    n_p = kp.sum(axis=1)
    n_c = kc.sum(axis=1)
    with np.errstate(invalid="ignore"):
        lam_p = np.where(n_p > 0,
                         (gp["lapse"] * kp).sum(axis=1)
                         / np.maximum(n_p, 1), np.nan)
        lam_c = np.where(n_c > 0,
                         (gc["lapse"] * kc).sum(axis=1)
                         / np.maximum(n_c, 1), np.nan)
    usable = (n_p >= C1_ZERO_MIN_N) & (n_c >= C1_ZERO_MIN_N)
    r_c = np.zeros(S, dtype=bool)
    r_f = np.zeros(S, dtype=bool)
    rt_fires = np.zeros(S, dtype=bool)
    if usable.any():
        # One batched test per distinct (kept participants, kept
        # controls) pair. Stable argsort on the DROPPED flag brings the
        # kept people to the front in their original order.
        key = n_p.astype(np.int64) * 10_000 + n_c
        ip = np.argsort(~kp, axis=1, kind="stable")
        ic = np.argsort(~kc, axis=1, kind="stable")
        for k in np.unique(key[usable]):
            sel = np.flatnonzero(usable & (key == k))
            m, n = int(n_p[sel[0]]), int(n_c[sel[0]])
            A = np.take_along_axis(gp["w"][sel],
                                   ip[sel][:, :m, None], axis=1)
            B = np.take_along_axis(gc["w"][sel],
                                   ic[sel][:, :n, None], axis=1)
            rc, rf = _c1_test(A, B)
            r_c[sel], r_f[sel] = rc, rf
            # THE SUBSAMPLE RE-CHECKS ITSELF -- AS A DIAGNOSTIC.
            #
            # Selecting on a clean catch record reduces the lapse gap
            # but does not erase it: someone lapsing at 0.14 still
            # passes ten catch trials about a third of the time. The
            # catch measure cannot see that residual (every count in
            # the subsample is zero by construction, so it is all
            # ties) but the fast-response share can, because it was
            # not selected on.
            #
            # NOT A SECOND GATE. Simulated, this re-check fires in
            # 99.9% of studies where the controls really do lapse
            # more -- with or without a real effect, since response
            # times do not depend on the weights. Treating it as a
            # gate therefore sets the secondary's power to 0.000 in
            # every cell, not merely lowers it: C1 would become
            # unanswerable whenever the two groups' attention differs
            # at all. That is a stronger claim than the evidence
            # supports, so the re-check is REPORTED and the secondary
            # is read with its residual false-positive rate stated.
            # The gated-on-it column stays in the table so the
            # alternative is on the record rather than implied.
            if RT_GATE:
                fa = np.take_along_axis(gp["fast"][sel],
                                        ip[sel][:, :m], axis=1)
                fb = np.take_along_axis(gc["fast"][sel],
                                        ic[sel][:, :n], axis=1)
                prt, _ = mw_p_batch(fb, fa, exact_max_n=0)
                rt_fires[sel] = prt <= GATE_ALPHA
    return {"reject_colour": r_c, "reject_fog": r_f,
            "usable": usable, "n_participants": n_p,
            "n_controls": n_c, "lapse_participants": lam_p,
            "lapse_controls": lam_c, "rt_fires": rt_fires}


def draw_attention(n_people, lapse_mean, n_trials, rng,
                   p_fast_given_lapse=RT_FAST_GIVEN_LAPSE):
    """Personal lapse rates and the two attention measures.

    Separated out because THE GATE DOES NOT DEPEND ON THE WEIGHTS at
    all -- only on who lapsed and how often. So its operating
    characteristics can be simulated without fitting anything, which
    is what makes sweeping RT_FAST_GIVEN_LAPSE affordable.

    Returns (personal lapse rates, catch-failure counts, share of
    responses below the fast cut-off).
    """
    if lapse_mean <= 0:
        lam = np.zeros(n_people)
    else:
        shape = 1.0 / max(LAPSE_BETWEEN_SD, 1e-6) ** 2
        lam = np.clip(rng.gamma(shape, lapse_mean / shape,
                                size=n_people), 0.0, 0.9)
    catch = rng.binomial(
        N_CATCH_TRIALS,
        np.clip(CATCH_P_FAIL_PER_LAPSE * lam, 0.0, 1.0))
    p_fast = (lam * p_fast_given_lapse
              + (1.0 - lam) * RT_FAST_GIVEN_ENGAGED)
    fast = rng.binomial(n_trials,
                        np.clip(p_fast, 0.0, 1.0)) / n_trials
    return lam, catch, fast


def run_gate_characteristics(n_trials, report, root):
    """How often the gate fires, and on what.

    The gate decides whether C1 is reported at all, so its own
    behaviour has to be on the record separately from C1's: how often
    it fires when the two groups are matched (its cost), how often it
    fires when they are not (its sensitivity), and how much of each
    comes from the catch measure rather than the response times.

    THE RESPONSE-TIME MEASURE CARRIES AN ASSUMPTION THE CATCH MEASURE
    DOES NOT. Its sensitivity depends on RT_FAST_GIVEN_LAPSE, the
    chance that a lapsed trial produces a response under
    RT_FAST_CUTOFF_MS, which is a guess. So it is swept, and the
    catch-only column shows what the gate does if response times turn
    out to be uninformative or are not recorded at all.

    Needs no weights and no trial list -- only the number of image
    trials -- so it is cheap.
    """
    print("\n" + "=" * 60)
    print("THE ATTENTION GATE: HOW OFTEN IT FIRES")
    print("=" * 60)
    print(f"  catch block {N_CATCH_TRIALS} trials, p_fail = "
          f"{CATCH_P_FAIL_PER_LAPSE} x lambda")
    print(f"  {n_trials} image trials, fast cut "
          f"{RT_FAST_CUTOFF_MS} ms")
    print(f"  personal lapse rates gamma-distributed, SD "
          f"{LAPSE_BETWEEN_SD}x the group mean")
    print(f"  one-sided, controls expected worse, Holm across the two "
          f"measures at alpha {GATE_ALPHA}")
    print(f"  {GATE_STUDIES} simulated studies per cell")
    print()
    print("  THE SIMULATED RATES ARE AN UPPER BOUND ON SENSITIVITY. "
          "They assume people")
    print("  lapse on catch trials at the same rate as on image "
          "trials. A catch screen")
    print("  has no pictures and one line of text, so someone "
          "skimming may well attend")
    print("  to it more. If they lapse less there, the gate is less "
          "sensitive than this.")
    rng = np.random.default_rng(RANDOM_SEED + 606)
    rows = []
    for (n_p, n_c) in GATE_SIZE_GRID:
        for lam_p, lam_c in C1_LAPSE_PAIRS:
            for pf in RT_FAST_SWEEP:
                _, cp, fp_ = draw_attention(
                    GATE_STUDIES * n_p, lam_p, n_trials, rng, pf)
                _, cc, fc = draw_attention(
                    GATE_STUDIES * n_c, lam_c, n_trials, rng, pf)
                cp = cp.reshape(GATE_STUDIES, n_p).astype(float)
                fp_ = fp_.reshape(GATE_STUDIES, n_p)
                cc = cc.reshape(GATE_STUDIES, n_c).astype(float)
                fc = fc.reshape(GATE_STUDIES, n_c)
                fired, f_c, f_r = _c1_gate(cp, cc, fp_, fc)
                # catch measure on its own, at the full alpha
                p_only, _ = mw_p_batch(cc, cp, exact_max_n=0)
                rows.append({
                    "n_participants": n_p, "n_controls": n_c,
                    "lapse_participants": lam_p,
                    "lapse_controls": lam_c,
                    "p_fast_given_lapse": pf,
                    "fires": round(float(fired.mean()), 4),
                    "fires_catch": round(float(f_c.mean()), 4),
                    "fires_rt": round(float(f_r.mean()), 4),
                    "catch_alone": round(
                        float((p_only <= GATE_ALPHA).mean()), 4)})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(root, "gate_characteristics.csv"),
               index=False)

    matched = tab[tab.lapse_participants == tab.lapse_controls]
    worse = tab[tab.lapse_controls > tab.lapse_participants]
    better = tab[tab.lapse_participants > tab.lapse_controls]
    print("\n  (a) FALSE ALARM: the groups are matched, so the gate "
          "should stay quiet")
    # aggfunc="max" so the table and the headline below agree: each
    # (size, lapse) cell appears once per swept RT_FAST_GIVEN_LAPSE,
    # and for a false-alarm rate the worst of them is the number to
    # show, not their average.
    print(matched.pivot_table(index=["n_participants", "n_controls"],
                              columns="lapse_participants",
                              values="fires",
                              aggfunc="max").round(3).to_string())
    print(f"    nominal {GATE_ALPHA}; worst cell "
          f"{matched.fires.max():.3f}")
    print("    A false alarm costs power, not validity: C1 is "
          "reported as")
    print("    inconclusive and the secondary still runs.")

    print("\n  (b) SENSITIVITY: the controls lapse MORE, which is "
          "the direction that")
    print("      manufactures a C1 effect out of nothing")
    print(worse.pivot_table(
        index=["n_participants", "n_controls"],
        columns=["lapse_controls", "p_fast_given_lapse"],
        values="fires").round(3).to_string())
    print("\n      the CATCH MEASURE ALONE (no response times "
          "needed):")
    print(worse[worse.p_fast_given_lapse == RT_FAST_SWEEP[0]]
          .pivot_table(index=["n_participants", "n_controls"],
                       columns="lapse_controls",
                       values="catch_alone").round(3).to_string())
    print(f"\n    worst sensitivity over the whole grid: "
          f"{worse.fires.min():.3f} combined, "
          f"{worse.catch_alone.min():.3f} on catch alone")
    print("    (the catch measure does not depend on "
          "RT_FAST_GIVEN_LAPSE, so its")
    print("     column is the same at every swept value)")

    print("\n  (c) THE OTHER DIRECTION: the PARTICIPANTS lapse more. "
          "The gate is")
    print("      one-sided and does not fire, which is correct -- "
          "that asymmetry")
    print("      shrinks the C1 gap and is merely conservative.")
    print(f"      worst firing rate there: {better.fires.max():.3f}")

    report.append(("gate false alarm, worst cell",
                   f"{matched.fires.max():.3f} "
                   f"(nominal {GATE_ALPHA})"))
    report.append(("gate sensitivity, controls lapse more",
                   f"{worse.fires.min():.3f}-{worse.fires.max():.3f}; "
                   f"{worse.catch_alone.min():.3f}-"
                   f"{worse.catch_alone.max():.3f} on catch alone"))
    return tab


def run_c1(X, th, rates, report, root):
    """C1: do the DPDR group weight a feature more heavily than
    controls?

    RANK-BASED. A separable participant's Firth weight is finite but
    means "at least very large", so a mean-based test would compare a
    number the data does not pin down. A rank does not care whether
    the top weight is 8 or 40, so no cap is needed and the strongest
    responders stay in.

    BOTH FEATURES, WITH HOLM. H1 (fog) and H2 (colour) each get a
    one-sided Mann-Whitney and the pair is corrected step-down.

    THE CONTROLS ARE NOT NON-RESPONDERS. A control can work out that
    the experiment is about edited images and pick the more edited
    one, which loads positively on BOTH features and eats the contrast
    C1 rests on. Applied at the same rate in both groups, since the
    concept is not secret.

    THE ATTENTION GATE COMES FIRST. A control group that lapses more
    has its weights attenuated toward zero, widening exactly the gap
    C1 looks for; the choice data cannot distinguish that from a real
    effect, and the joint lapse model cannot either (see the recovery
    check). So the gate on catch failures and fast responses is tested
    first, and when it fires C1 IS REPORTED AS INCONCLUSIVE. A
    pre-registered secondary rerun on the zero-catch-failure
    participants is reported beside it, with its sample sizes.

    Every figure carries its whole scenario: responder share among
    participants, effect size with its choice probability, strategy
    rate, and the lapse pair.
    """
    print("\n" + "=" * 60)
    print("C1: THE GROUP COMPARISON, WITH THE ATTENTION GATE")
    print("=" * 60)
    med = informative_medians(X)
    U = th["upper"]
    mid_mult = NONRESPONDER_SPREAD_MULT
    taste_sd = U * mid_mult
    print(f"  one-sided Mann-Whitney on the per-person weights, "
          f"alpha {ALPHA}, Holm across H1/H2")
    print(f"  gate: catch failures"
          + (" and fast responses" if RT_GATE else "")
          + f", one-sided, controls worse, alpha {GATE_ALPHA}")
    print(f"  when the gate fires: C1 = {GATE_ACTION.upper()}")
    print(f"  catch block {N_CATCH_TRIALS} trials, "
          f"p_fail = {CATCH_P_FAIL_PER_LAPSE} x lambda")
    print(f"  personal lapse rates: gamma around the group mean, "
          f"SD {LAPSE_BETWEEN_SD}x the mean")
    print(f"  fast-response cut {RT_FAST_CUTOFF_MS} ms; "
          f"P(fast | lapse) {RT_FAST_GIVEN_LAPSE}")
    print(f"  non-responder taste at {mid_mult}x each threshold")
    print(f"  {C1_STUDIES} studies per cell, pools of "
          f"{C1_POOL_PEOPLE:,} per person type")
    print()
    print("  THE GATE'S SIMULATED DETECTION RATE IS AN UPPER BOUND: "
          "it assumes people")
    print("  lapse on catch trials at the same rate as on image "
          "trials. A catch screen")
    print("  has no pictures and one line of text, so someone "
          "skimming may well attend")
    print("  to it more. If they lapse less there, the gate is less "
          "sensitive than this.")

    rng = np.random.default_rng(RANDOM_SEED + 77)
    lap_vals = sorted({v for pr in C1_LAPSE_PAIRS for v in pr})
    effects = [0.0] + list(RULE_RESPONDER_WEIGHTS)

    # A person type with NO effect is the same type whichever feature
    # the effect would have been on and whatever share of participants
    # would have carried it -- adding 0.0 to a weight is adding
    # nothing. So every no-effect key maps to one pool, which is also
    # the control pool for every cell.
    def pool_key(feat_idx, w, share, strat, lam):
        if w == 0.0:
            return (0, 0.0, C1_RESPONDER_SHARES[0], strat, lam)
        return (feat_idx, w, share, strat, lam)

    t0 = time.time()
    pools = {}
    for feat_idx in (0, 1):
        for w in effects:
            for share in C1_RESPONDER_SHARES:
                for strat in C1_STRATEGY_RATES:
                    for lam in lap_vals:
                        k = pool_key(feat_idx, w, share, strat, lam)
                        if k in pools:
                            continue
                        pools[k] = _c1_pool(
                            X, th, k[0], k[1], k[2], strat,
                            C1_STRATEGY_WEIGHT, taste_sd, lam, rng)
    print(f"\n  {len(pools)} person-type pools fitted in "
          f"{time.time() - t0:.0f}s")

    def cell(feat_idx, w, share, n_p, n_c, strat, lam_p, lam_c):
        pp = pools[pool_key(feat_idx, w, share, strat, lam_p)]
        pc = pools[pool_key(feat_idx, 0.0, share, strat, lam_c)]
        gp = _c1_draw(pp, C1_STUDIES, n_p, rng)
        gc = _c1_draw(pc, C1_STUDIES, n_c, rng)
        r_c, r_f = _c1_test(gp["w"], gc["w"])
        fired, f_catch, f_rt = _c1_gate(
            gp["catch"], gc["catch"], gp["fast"], gc["fast"])
        # The registered verdict: a fired gate makes C1 inconclusive,
        # so it is neither a positive nor a negative.
        pos_c = r_c & ~fired
        pos_f = r_f & ~fired
        out = {
            "power_colour_raw": float(r_c.mean()),
            "power_fog_raw": float(r_f.mean()),
            "power_either_raw": float((r_c | r_f).mean()),
            "gate_fires": float(fired.mean()),
            "gate_fires_catch": float(f_catch.mean()),
            "gate_fires_rt": float(f_rt.mean()),
            "power_colour_gated": float(pos_c.mean()),
            "power_fog_gated": float(pos_f.mean()),
            "power_either_gated": float((pos_c | pos_f).mean()),
            "inconclusive": float(fired.mean())}
        if GATE_SECONDARY_ZERO_CATCH:
            z = _c1_zero_catch(gp, gc)
            u = z["usable"]
            out.update({
                "zero_usable": float(u.mean()),
                "zero_n_participants": float(
                    z["n_participants"].mean()),
                "zero_n_controls": float(z["n_controls"].mean()),
                "zero_lapse_participants": float(
                    np.nanmean(z["lapse_participants"])),
                "zero_lapse_controls": float(
                    np.nanmean(z["lapse_controls"])),
                "zero_power_either": (
                    float((z["reject_colour"] | z["reject_fog"])[u]
                          .mean()) if u.any() else np.nan),
                "zero_rt_fires": (float(z["rt_fires"][u].mean())
                                  if u.any() else np.nan),
                "zero_power_rt_clean": (
                    float(((z["reject_colour"] | z["reject_fog"])
                           & ~z["rt_fires"])[u].mean())
                    if u.any() else np.nan)})
        return out

    def sizes_for(n_p):
        return sorted({int(round(rr * n_p))
                       for rr in C1_CONTROL_RATIOS}
                      | set(C1_CONTROL_ABSOLUTE))

    # THE CELL GRID. The no-effect cells are run ONCE, not once per
    # feature and once per responder share: with no effect anywhere
    # those cells are statistically identical -- adding 0.0 to a weight
    # is adding nothing, and both features are tested in every cell
    # regardless. They are labelled "none" and the false-positive
    # tables pick them out by effect size.
    combos = [(0, "none", 0.0, C1_RESPONDER_SHARES[0])]
    for feat_idx, feat in enumerate(("colour", "fog")):
        claim = "H2 colour" if feat == "colour" else "H1 fog"
        for w in effects:
            if w == 0.0:
                continue
            for share in C1_RESPONDER_SHARES:
                combos.append((feat_idx, claim, w, share))

    rows = []
    for feat_idx, claim, w, share in combos:
        p_typ = round(as_probability(
            w, med["colour"] if feat_idx == 0 else med["fog"]), 4)
        for n_p in C1_PARTICIPANT_N:
            for n_c in sizes_for(n_p):
                for strat in C1_STRATEGY_RATES:
                    for lam_p, lam_c in C1_LAPSE_PAIRS:
                        r = cell(feat_idx, w, share, n_p, n_c,
                                 strat, lam_p, lam_c)
                        r.update({"effect_on": claim,
                                  "effect_sd": w,
                                  "effect_p_typical": p_typ,
                                  "responder_share": share,
                                  "n_participants": n_p,
                                  "n_controls": n_c,
                                  "strategy_rate": strat,
                                  "lapse_participants": lam_p,
                                  "lapse_controls": lam_c})
                        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(root, "c1_power.csv"), index=False)
    print(f"  {len(tab)} cells")

    # --- the false positive, with and without the gate ---
    null = tab[np.isclose(tab.effect_sd, 0.0)]
    m_null = null[null.lapse_participants == null.lapse_controls]
    a_null = null[null.lapse_controls > null.lapse_participants]
    print("\n  FALSE-POSITIVE RATE, no effect in either group:")
    print(f"    {'cells':>28}  {'ungated':>8}  {'gate fires':>11}  "
          f"{'GATED':>8}")
    for lbl, g in (("matched lapse rates", m_null),
                   ("controls lapse MORE", a_null)):
        if not len(g):
            continue
        print(f"    {lbl:>28}  "
              f"{g.power_either_raw.max():>8.3f}  "
              f"{g.gate_fires.min():>11.3f}  "
              f"{g.power_either_gated.max():>8.3f}")
    print(f"    (nominal {ALPHA}; 'ungated' and 'GATED' are the worst "
          "cell in each set,")
    print("     'gate fires' is the LEAST often it fires in that set)")
    if len(a_null):
        worst = a_null.loc[a_null.power_either_gated.idxmax()]
        print(f"\n    worst gated cell: participants lapse "
              f"{worst.lapse_participants}, controls "
              f"{worst.lapse_controls},")
        print(f"      n {int(worst.n_participants)}/"
              f"{int(worst.n_controls)}, strategy "
              f"{worst.strategy_rate} -> "
              f"{worst.power_either_gated:.3f} "
              f"(gate fired {worst.gate_fires:.3f})")
        report.append(("C1 false positive, controls lapse more",
                       f"{a_null.power_either_raw.max():.3f} ungated, "
                       f"{a_null.power_either_gated.max():.3f} gated"))
    report.append(("C1 false positive, matched lapses",
                   f"{m_null.power_either_raw.max():.3f} "
                   f"(nominal {ALPHA})"))

    # --- the gate's own operating characteristics ---
    print("\n  THE GATE ALONE, by lapse pair (how often it fires):")
    g = tab.groupby(["lapse_participants", "lapse_controls"])[
        ["gate_fires", "gate_fires_catch", "gate_fires_rt"]].mean()
    print(g.round(3).to_string())
    matched_rate = float(
        tab[tab.lapse_participants == tab.lapse_controls]
        .gate_fires.mean())
    # Named apart from the gate section's own figure: that one
    # sweeps the gate's size grid, this one is measured over C1's
    # cells, so the two are different averages of the same quantity
    # and a reader seeing one number twice would read a disagreement.
    report.append(("gate false alarm, over C1's cells",
                   f"{matched_rate:.3f} (nominal {GATE_ALPHA})"))

    # --- the realistic sizes, with the full scenario ---
    print("\n" + "-" * 60)
    print("  REALISTIC SIZES: participants 10 / 15 / 20 against 30 "
          "controls")
    print("-" * 60)
    real_np = [n for n in (10, 15, 20) if n in C1_PARTICIPANT_N]
    for claim in ("H1 fog", "H2 colour"):
        key = "power_fog_gated" if claim.startswith("H1") \
            else "power_colour_gated"
        for share in C1_RESPONDER_SHARES:
            g = tab[(tab.effect_on == claim)
                    & (tab.responder_share == share)
                    & (tab.n_participants.isin(real_np))
                    & (tab.n_controls == 30)
                    & (~np.isclose(tab.effect_sd, 0.0))
                    & (tab.lapse_participants == tab.lapse_controls)]
            if not len(g):
                continue
            print(f"\n  {claim}, responder share {share:g}, "
                  f"30 controls, matched lapses")
            print("  (rows: choice probability on a typical trial; "
                  "columns: participants x strategy x lapse)")
            piv = g.pivot_table(
                index="effect_p_typical",
                columns=["n_participants", "strategy_rate",
                         "lapse_participants"],
                values=key)
            print(piv.round(3).to_string())

    # --- the three rates side by side ---
    print("\n" + "-" * 60)
    print("  WHAT THE GATE AND THE SECONDARY BUY, under no effect")
    print("-" * 60)
    print("  (worst cell in each set; nominal "
          f"{ALPHA})")
    print(f"    {'lapse pair':>22}  {'ungated':>8}  {'gated':>8}  "
          f"{'2nd':>8}  {'2nd usable':>10}")
    for (lp, lc), g in null.groupby(["lapse_participants",
                                     "lapse_controls"]):
        lbl = f"participants {lp:g}, controls {lc:g}"
        sec = (f"{g.zero_power_either.max():>8.3f}"
               if GATE_SECONDARY_ZERO_CATCH else f"{'--':>8}")
        use = (f"{g.zero_usable.min():>10.3f}"
               if GATE_SECONDARY_ZERO_CATCH else f"{'--':>10}")
        print(f"    {lbl:>22}  {g.power_either_raw.max():>8.3f}  "
              f"{g.power_either_gated.max():>8.3f}  {sec}{use}")
    print("\n  'ungated' is C1 read straight off the weights: it "
          "reaches")
    print(f"  {a_null.power_either_raw.max():.3f} when the controls "
          "lapse more, because attenuating the")
    print("  controls' weights toward zero widens exactly the gap C1 "
          "looks for.")
    print("  'gated' is the registered verdict, where a fired gate "
          "makes C1")
    print("  inconclusive. '2nd' is the pre-registered rerun on the "
          "people with no")
    print("  catch failures: the analysis that still returns an "
          "answer, at the price")
    print("  of a residual inflation of its own and of however many "
          "people it drops.")

    # --- the zero-catch-failure secondary, in detail ---
    if GATE_SECONDARY_ZERO_CATCH:
        print("\n" + "-" * 60)
        print("  SECONDARY: C1 ON THE ZERO-CATCH-FAILURE SUBSAMPLE")
        print("-" * 60)
        print("  Pre-registered, not post-hoc matching: the rule is "
              "fixed before the")
        print("  data arrive. It works only because lapse rates "
              "differ BETWEEN people,")
        print("  so a clean catch record is evidence of a low "
              "personal rate -- if")
        print("  everyone in a group lapsed at the group mean this "
              "would select a random")
        print("  subset with the same mean and fix nothing.")
        print("\n  (a) WHAT SURVIVES, by lapse pair (mean over the "
              "size grid, except")
        print("      the usable share, which is its worst cell)")
        print(null.groupby(["lapse_participants", "lapse_controls"])
              .agg(zero_n_participants=("zero_n_participants", "mean"),
                   zero_n_controls=("zero_n_controls", "mean"),
                   zero_lapse_participants=(
                       "zero_lapse_participants", "mean"),
                   zero_lapse_controls=("zero_lapse_controls", "mean"),
                   usable_worst=("zero_usable", "min"))
              .round(3).to_string())
        print("\n  zero_lapse_* is the mean PERSONAL lapse rate that "
              "survives. The")
        print("  subsample reduces the asymmetry but does not erase "
              "it: a group at 0.2")
        print("  still comes through at about 0.14, because a person "
              "lapsing at 0.14")
        print("  passes ten catch trials about a third of the time.")
        print("  zero_usable is the share of studies keeping at least "
              f"{C1_ZERO_MIN_N} people in BOTH")
        print("  groups; below that a one-sided rank test cannot "
              "reach alpha at all.")

        print("\n  (b) FALSE-POSITIVE RATE, no effect anywhere")
        print(null.groupby(["lapse_participants", "lapse_controls"])
              .agg(ungated_mean=("power_either_raw", "mean"),
                   ungated_worst=("power_either_raw", "max"),
                   second_mean=("zero_power_either", "mean"),
                   second_worst=("zero_power_either", "max"),
                   second_rt_clean_worst=(
                       "zero_power_rt_clean", "max"),
                   rt_recheck_fires=("zero_rt_fires", "mean"))
              .round(3).to_string())
        asym = null[null.lapse_controls > null.lapse_participants]
        print(f"\n      THE SUBSAMPLE REDUCES THE ARTEFACT BUT DOES "
              f"NOT REMOVE IT. Where the")
        print(f"      controls lapse more, the ungated rate reaches "
              f"{asym.power_either_raw.max():.3f} and the")
        print(f"      secondary's reaches "
              f"{asym.zero_power_either.max():.3f}, against a nominal "
              f"{ALPHA}.")
        print("      The reason is in table (a): selecting on a clean "
              "catch record leaves")
        print("      a residual lapse gap, and the power to detect "
              "that residual grows")
        print("      with the sample size and with the size of the "
              "weights. So the cells")
        print("      above nominal are the LARGE ones, not the small "
              "ones:")
        by = asym.groupby(["n_participants", "strategy_rate"])[
            "zero_power_either"].max().unstack()
        print(by.round(3).to_string())
        print("      (worst secondary false-positive rate; rows "
              "participants, columns")
        print("       strategy rate -- a higher strategy rate makes "
              "every weight larger,")
        print("       so the same residual attenuation gap becomes "
              "easier to detect)")
        real = asym[asym.n_participants.isin(real_np)
                    & (asym.n_controls == 30)]
        if len(real):
            print(f"\n      AT THE SIZES THIS STUDY WILL RUN "
                  f"({'/'.join(str(x) for x in real_np)} participants "
                  f"against 30 controls):")
            print(f"        ungated   {real.power_either_raw.mean():.3f} "
                  f"mean, {real.power_either_raw.max():.3f} worst")
            print(f"        secondary {real.zero_power_either.mean():.3f} "
                  f"mean, {real.zero_power_either.max():.3f} worst")
            print(f"        secondary, RT re-check clean "
                  f"{real.zero_power_rt_clean.mean():.3f} mean, "
                  f"{real.zero_power_rt_clean.max():.3f} worst")
            report.append(
                ("C1 secondary false positive at the realistic sizes",
                 f"{real.zero_power_either.max():.3f} worst vs "
                 f"{real.power_either_raw.max():.3f} ungated "
                 f"(nominal {ALPHA})"))
        print(f"\n      THE RT RE-CHECK IS A DIAGNOSTIC, NOT A "
              f"SECOND GATE. The subsample's")
        print(f"      own response-time test fires in "
              f"{asym.zero_rt_fires.mean():.1%} of these studies, and "
              f"gating on it")
        print(f"      would bring the worst secondary false-positive "
              f"rate to "
              f"{asym.zero_power_rt_clean.max():.3f} --")
        print("      but response times do not depend on the weights, "
              "so it fires just as")
        print("      often when the effect is real, and the "
              "secondary's POWER under that")
        print("      rule is 0.000 in every cell. Registering it "
              "would amount to saying")
        print("      C1 is unanswerable whenever the groups' "
              "attention differs at all.")
        print("      So it is reported: a fired re-check says the "
              "residual contamination")
        print("      is present and the secondary's own result "
              "should be read with the")
        print("      inflation above, not as a clean test.")
        report.append(
            ("C1 secondary false positive, controls lapse more",
             f"{asym.zero_power_either.max():.3f} worst, "
             f"{asym.zero_power_rt_clean.max():.3f} after the RT "
             f"re-check, vs {asym.power_either_raw.max():.3f} "
             f"ungated (nominal {ALPHA})"))

        print("\n  (c) POWER of the secondary at the realistic "
              "sizes, 30 controls")
        pw = tab[(~np.isclose(tab.effect_sd, 0.0))
                 & (tab.n_participants.isin(real_np))
                 & (tab.n_controls == 30)
                 & (tab.lapse_controls > tab.lapse_participants)]
        if len(pw):
            print(pw.pivot_table(
                index=["effect_on", "effect_p_typical",
                       "responder_share"],
                columns=["n_participants", "lapse_controls"],
                values="zero_power_either").round(3).to_string())
            print("\n  The primary C1 is inconclusive in every "
                  "one of these cells -- the")
            print("  gate fires at essentially 1.000 when the "
                  "controls lapse more -- so the")
            print("  secondary is the analysis that carries the "
                  "result, and this is its")
            print("  power. Averaged over the strategy rates; the "
                  "scenario is otherwise")
            print("  fully named by the row and column labels.")
            report.append(
                ("C1 secondary power, controls lapse more",
                 f"{pw.zero_power_either.min():.3f}-"
                 f"{pw.zero_power_either.max():.3f} over the "
                 "realistic sizes"))
        report.append(
            ("C1 zero-catch secondary, surviving controls",
             f"{null.zero_n_controls.min():.1f}-"
             f"{null.zero_n_controls.max():.1f}, usable in "
             f"{null.zero_usable.min():.0%}-"
             f"{null.zero_usable.max():.0%} of studies"))
    report.append(("C1 cells", str(len(tab))))
    return tab


# ============================================================
# Crossover and additivity  (depend on the trial list)
# ============================================================

def design3(X):
    """The two predictors plus an additivity term.

    The third column is (x_sat + x_fog) * |x_sat| * |x_fog|. It is
    zero on single-cue trials -- where one of the two differences is
    zero -- so it is carried entirely by the `together` and `conflict`
    trials, which is exactly the pair of types that exist to answer
    this question.

    It flips sign when the two images swap sides, like the other two
    columns, because the first factor is odd under the swap and the
    rest is even. A plain x_sat*x_fog product would not, and with
    randomised sides it would estimate nothing.
    """
    xs, xf = X[:, 0], X[:, 1]
    return np.column_stack([xs, xf, (xs + xf) * np.abs(xs) * np.abs(xf)])


def simulate_nonadditive(X, b_sat, b_fog, gamma, n, rng):
    """Choices of n people whose cues combine non-additively.

    gamma is the proportional change in the combined evidence when
    both cues sit at their full range. gamma = 0 reproduces the
    additive model exactly.
    """
    xs, xf = X[:, 0], X[:, 1]
    score = ((b_sat * xs + b_fog * xf)
             * (1.0 + gamma * np.abs(xs) * np.abs(xf)))
    p = 1.0 / (1.0 + np.exp(-score))
    return (rng.random((n, len(xs))) < p).astype(float)


def _critical(null_values, alpha):
    """Two-sided critical values, as PERCENTILES of the null.

    Not mean +/- 1.96 SD. A three-parameter logistic fitted to 180
    trials carries a small finite-sample bias on the additivity term --
    measured at about +0.1 even when the truth is exactly additive --
    so a symmetric interval around zero would call one direction too
    readily and the other too rarely. Percentiles of the simulated null
    absorb the bias by construction, whatever its size.
    """
    lo = float(np.percentile(null_values, 100 * alpha / 2.0))
    hi = float(np.percentile(null_values, 100 * (1 - alpha / 2.0)))
    return lo, hi


def run_contrasts(X, label, out_dir, report):
    """Both contrasts, across the common effect grid.

    Wraps the single-baseline version below. The baseline is the weight
    the null person carries on BOTH cues, and the earlier code ran at
    one carried-over value of 0.7 -- which per corpus SD is about 2.4
    times the colour threshold but only 1.5 times the fog one, so it
    put colour near its ceiling and fog mid-range. A contrast
    calibrated there is not comparable between the two features.
    """
    out, summ = [], None
    for B in CONTRAST_BASELINE_GRID:
        is_s = bool(np.isclose(B, CONTRAST_SUMMARY))
        # The report only takes the summary baseline's lines, or it
        # would carry five copies of every contrast figure with no
        # way to tell which baseline each came from.
        res = _run_contrasts_at(X, B, label, out_dir,
                                report if is_s else [],
                                is_summary=is_s)
        if res is None:
            continue
        out.append({
            "baseline_weight_SD": B,
            "gap_needed": res["gap_needed"],
            "gamma_needed": res["gamma_needed"],
            "sd_diff": res["sd_diff"],
            "sd_diff_quadrature": res["sd_diff_quadrature"],
            "crit_diff_lo": res["crit_diff"][0],
            "crit_diff_hi": res["crit_diff"][1],
            "crit_gamma_lo": res["crit_gamma"][0],
            "crit_gamma_hi": res["crit_gamma"][1],
            "is_summary": is_s})
        if is_s:
            summ = res
    if out:
        pd.DataFrame(out).to_csv(
            os.path.join(out_dir, "contrast_grid.csv"), index=False)
        print("\n  THE CONTRASTS ACROSS THE BASELINE GRID:")
        print(pd.DataFrame(out).round(4).to_string(index=False))
        print(f"\n  contrasts run at baselines "
              f"{CONTRAST_BASELINE_GRID} per SD; the single summary "
              f"below is at {CONTRAST_SUMMARY},")
        print("  which is an ASSUMPTION about a plausible operating "
              "weight, not a measurement.")
    return summ


def _run_contrasts_at(X, B, label, out_dir, report, is_summary=True):
    """How big must a crossover, or a departure from additivity, be?

    Both are calibrated the same way: simulate the null, take its
    central interval as the critical region, then sweep the true effect
    and record how often it lands outside.
    """
    print("\n" + "=" * 60)
    print(f"CROSSOVER AND ADDITIVITY  [{label}]")
    print("=" * 60)
    X3 = design3(X)

    # The design must survive the side swap, or the third column means
    # nothing. Cheap to check, so check it rather than trust it.
    if not np.allclose(design3(-X), -X3):
        raise AssertionError(
            "design3 is not antisymmetric under the side swap; the "
            "additivity term would be unestimable.")
    c3 = np.corrcoef(X3.T)
    vif3 = np.diag(np.linalg.inv(c3))
    print(f"  baseline weight on both cues: {B} per SD"
          + ("   <- the summary value" if is_summary else ""))
    print(f"  3-term design VIFs: sat {vif3[0]:.2f}, fog {vif3[1]:.2f}, "
          f"additivity {vif3[2]:.2f}")
    if vif3[2] > 5.0:
        print("    WARNING: the additivity term is badly collinear with "
              "the other two; its estimate will be unstable.")

    rng = np.random.default_rng(RANDOM_SEED + 11)

    # ---- 1. CROSSOVER ------------------------------------------------
    # Null: a person who weights both cues EQUALLY. Their fitted
    # difference is zero on average, and its scatter is what a real
    # crossover has to beat.
    W0 = np.full((N_CONTRAST_PEOPLE, 2), B)
    null_est = fit_logit_batch(X, simulate_many(X, W0, rng))
    null_diff = null_est[:, 1] - null_est[:, 0]
    lo_d, hi_d = _critical(null_diff, CONTRAST_ALPHA)
    sd_d = float(null_diff.std())
    # For reference: what adding the two scatters in quadrature would
    # have predicted, which is the usual back-of-envelope.
    sd_ind = float(np.hypot(null_est[:, 0].std(), null_est[:, 1].std()))
    print("\n  CROSSOVER (is b_fog bigger than b_sat?)")
    print(f"    scatter of the difference under equal weights: "
          f"{sd_d:.3f}")
    print(f"    the same from the two scatters in quadrature: "
          f"{sd_ind:.3f}")
    print(f"    critical values at alpha {CONTRAST_ALPHA}: "
          f"{lo_d:+.3f} .. {hi_d:+.3f}")

    rows = []
    for gap in CROSSOVER_GRID:
        W = np.column_stack([
            np.full(N_CONTRAST_PEOPLE, B - gap / 2.0),
            np.full(N_CONTRAST_PEOPLE, B + gap / 2.0)])
        est = fit_logit_batch(X, simulate_many(X, W, rng))
        d = est[:, 1] - est[:, 0]
        rows.append({"true_gap": gap,
                     "mean_est_gap": float(d.mean()),
                     "sd_est_gap": float(d.std()),
                     "detected": float(((d < lo_d) | (d > hi_d)).mean())})
    cross = pd.DataFrame(rows)
    # Only the summary baseline writes this file; the others
    # would each overwrite the last with nothing saying
    # which baseline it came from. contrast_grid.csv has
    # the cross-baseline view.
    if is_summary:
      cross.to_csv(os.path.join(out_dir, "crossover_curve.csv"),
                 index=False)
    print("\n    share of people whose crossover is caught:")
    print("    " + cross.round(3).to_string(index=False)
          .replace("\n", "\n    "))

    gap_needed = _solve_target(cross.detected.to_numpy(),
                              cross.true_gap.to_numpy(),
                              CONTRAST_TARGET, "crossover gap")

    # ---- 2. ADDITIVITY -----------------------------------------------
    # Null: a person whose cues combine additively (gamma = 0). Fit the
    # THREE-term model to them; the third weight is the statistic.
    null3 = fit_logit_batch(
        X3, simulate_nonadditive(X, B, B, 0.0, N_CONTRAST_PEOPLE, rng))
    null_int = null3[:, 2]
    lo_g, hi_g = _critical(null_int, CONTRAST_ALPHA)
    print("\n  ADDITIVITY (do the two cues simply add?)")
    print(f"    additivity weight under true additivity: "
          f"mean {null_int.mean():+.3f}, SD {null_int.std():.3f}")
    print("      a non-zero mean here is the finite-sample bias of a "
          "3-parameter fit, which is why the")
    print("      critical values below are percentiles of this null "
          "rather than +/- 1.96 SD.")
    print(f"    critical values at alpha {CONTRAST_ALPHA}: "
          f"{lo_g:+.3f} .. {hi_g:+.3f}")

    rows = []
    for g in GAMMA_GRID:
        est = fit_logit_batch(
            X3, simulate_nonadditive(X, B, B, g, N_CONTRAST_PEOPLE, rng))
        bi = est[:, 2]
        rows.append({"true_gamma": g,
                     "mean_b_int": float(bi.mean()),
                     "sd_b_int": float(bi.std()),
                     "detected": float(((bi < lo_g) | (bi > hi_g)).mean())})
    addi = pd.DataFrame(rows)
    # Only the summary baseline writes this file; the others
    # would each overwrite the last with nothing saying
    # which baseline it came from. contrast_grid.csv has
    # the cross-baseline view.
    if is_summary:
      addi.to_csv(os.path.join(out_dir, "additivity_curve.csv"),
                index=False)
    print("\n    share of people whose non-additivity is caught:")
    print("    " + addi.round(3).to_string(index=False)
          .replace("\n", "\n    "))

    gamma_needed = _solve_target(addi.detected.to_numpy(),
                                 addi.true_gamma.to_numpy(),
                                 CONTRAST_TARGET, "gamma")

    # ---- 3. ADDITIVITY ACROSS PEOPLE ---------------------------------
    # One person's 180 trials carry only 60 mixed trials, which is not
    # enough. Pooling n people's mixed trials is, so report what each n
    # buys. The pooled design is just the one design stacked n times.
    pool_rows = []
    for n in CONTRAST_POOL_N:
        Xn = np.tile(X, (n, 1))
        X3n = design3(Xn)
        null_pool = fit_logit_batch(
            X3n, simulate_nonadditive(Xn, B, B, 0.0, N_POOL_REPS, rng))
        lo_p, hi_p = _critical(null_pool[:, 2], CONTRAST_ALPHA)
        det_by_gamma = {}
        for g in GAMMA_GRID:
            est = fit_logit_batch(
                X3n, simulate_nonadditive(Xn, B, B, g, N_POOL_REPS, rng))
            bi = est[:, 2]
            det_by_gamma[g] = float(((bi < lo_p) | (bi > hi_p)).mean())
        y = np.array([det_by_gamma[g] for g in GAMMA_GRID])
        g_need = _solve_target(y, np.array(GAMMA_GRID, dtype=float),
                               CONTRAST_TARGET, f"gamma at n={n}")
        # Trials that carry ANY information about additivity: the
        # mixed ones, plus the fog_only trials, whose colour difference
        # is small but not exactly zero (up to PAIR_SAT_TOL), so their
        # product term is non-zero too.
        pool_rows.append({"n_people": n,
                          "informing_trials":
                              n * int((X3[:, 2] != 0).sum()),
                          "crit_lo": lo_p, "crit_hi": hi_p,
                          "false_alarm": det_by_gamma[0.0],
                          f"gamma_at_{CONTRAST_TARGET:.0%}": g_need})
    pool = pd.DataFrame(pool_rows)
    # Only the summary baseline writes this file; the others
    # would each overwrite the last with nothing saying
    # which baseline it came from. contrast_grid.csv has
    # the cross-baseline view.
    if is_summary:
      pool.to_csv(os.path.join(out_dir, "additivity_pooled.csv"),
                index=False)
    print("\n    additivity across pooled participants:")
    print("    " + pool.round(3).to_string(index=False)
          .replace("\n", "\n    "))

    report.append(("crossover difference SD", f"{sd_d:.3f}"))
    report.append(("crossover critical values",
                   f"{lo_d:+.2f} .. {hi_d:+.2f}"))
    report.append((f"gap for {CONTRAST_TARGET:.0%} crossover detection",
                   f"{gap_needed:.2f}" if np.isfinite(gap_needed)
                   else "off the grid"))
    report.append(("additivity critical values",
                   f"{lo_g:+.2f} .. {hi_g:+.2f}"))
    report.append((f"gamma for {CONTRAST_TARGET:.0%} detection",
                   f"{gamma_needed:.2f}" if np.isfinite(gamma_needed)
                   else "off the grid"))

    if is_summary:
        # One figure, at the summary baseline. Writing one per
        # baseline would leave the last one in place with no label
        # saying which it was.
        _plot_contrasts(cross, addi, gap_needed, gamma_needed, label,
                        out_dir)
    return {"crossover": cross, "additivity": addi, "pooled": pool,
            "sd_diff": sd_d, "sd_diff_quadrature": sd_ind,
            "crit_diff": (lo_d, hi_d), "crit_gamma": (lo_g, hi_g),
            "gap_needed": gap_needed, "gamma_needed": gamma_needed,
            "vif3": vif3}


def _solve_target(y, x, target, what):
    """Where a rising detection curve crosses the target rate."""
    if y.max() < target:
        print(f"    {what}: never reaches {target:.0%} within the grid; "
              "widen it.")
        return np.nan
    return float(np.interp(target, y, x))


def _plot_contrasts(cross, addi, gap_needed, gamma_needed, label,
                    out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(cross.true_gap, cross.detected, marker="o", color="C3")
    ax.axhline(CONTRAST_TARGET, color="k", lw=0.8, ls=":")
    if np.isfinite(gap_needed):
        ax.axvline(gap_needed, color="C3", lw=0.8, ls="--")
        ax.annotate(f"gap {gap_needed:.2f}", (gap_needed,
                    CONTRAST_TARGET), textcoords="offset points",
                    xytext=(6, -14), color="C3")
    ax.set_xlabel("true gap between the two weights")
    ax.set_ylabel(f"share caught (alpha {CONTRAST_ALPHA}, two-sided)")
    ax.set_title("CROSSOVER\nis this person's fog weight bigger\n"
                 "than their colour weight?")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(addi.true_gamma, addi.detected, marker="o", color="C4")
    ax.axhline(CONTRAST_TARGET, color="k", lw=0.8, ls=":")
    if np.isfinite(gamma_needed):
        ax.axvline(gamma_needed, color="C4", lw=0.8, ls="--")
        ax.annotate(f"gamma {gamma_needed:.2f}", (gamma_needed,
                    CONTRAST_TARGET), textcoords="offset points",
                    xytext=(6, -14), color="C4")
    ax.set_xlabel("true departure from additivity (gamma)")
    ax.set_ylabel(f"share caught (alpha {CONTRAST_ALPHA}, two-sided)")
    ax.set_title("ADDITIVITY\ndo the two cues simply add when\n"
                 "both are present?")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Contrasts the per-feature floors cannot answer  "
                 f"[{label}]", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "contrasts.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Dose-response  (depends on the trial list)
# ============================================================

def run_dose(trials, X, label, out_dir, report):
    """The dose-response curve to expect if the effect is real.

    The proportion of 'chose the more manipulated image', computed per
    magnitude bin. A rising curve cannot be produced by a response bias
    or a demand characteristic, which would give a flat line that is
    still above chance -- so this is the strongest single figure the
    design can yield.

    Conflict trials are scored by FOG; see the header for why that is a
    convention rather than a fact.
    """
    print("\n" + "=" * 60)
    print(f"DOSE-RESPONSE  [{label}]")
    print("=" * 60)
    if "mag_bin" not in trials.columns:
        print("  No mag_bin column; skipping.")
        return pd.DataFrame()

    rng = np.random.default_rng(RANDOM_SEED + 3)
    rows = []
    # ACROSS THE COMMON GRID. The curve's SHAPE is the evidence -- a
    # response bias gives a flat line above chance, a real sensitivity
    # gives a rising one -- and the shape should be shown at several
    # true weights rather than at one carried over from the old units.
    for w in DOSE_EFFECT_GRID:
      eff = (w, w)
      for rep in range(N_DOSE_REPS):
        y = simulate_responses(X, eff, rng)
        d = trials.assign(resp=y)
        for (t, b), g in d.groupby(["trial_type", "mag_bin"]):
            # "Chose the more manipulated image" -- the same orientation
            # as the predictors: the LESS saturated image for a
            # saturation trial, the foggier one otherwise.
            if t == "sat_only":
                chose = (((g.d_sat < 0) & (g.resp == 1))
                         | ((g.d_sat > 0) & (g.resp == 0)))
            else:
                chose = (((g.d_fog > 0) & (g.resp == 1))
                         | ((g.d_fog < 0) & (g.resp == 0)))
            rows.append({"weight_SD": w, "rep": rep,
                         "trial_type": t, "mag_bin": b,
                         "prop": float(chose.mean()), "n": len(g)})

    full = (pd.DataFrame(rows)
            .groupby(["weight_SD", "trial_type", "mag_bin"])
            .agg(prop=("prop", "mean"), sd=("prop", "std"),
                 n_trials=("n", "first")).reset_index())
    full.to_csv(os.path.join(out_dir, "dose_response_grid.csv"),
                index=False)
    dose = (full[np.isclose(full.weight_SD, DOSE_SUMMARY)]
            .drop(columns="weight_SD").reset_index(drop=True))
    dose.to_csv(os.path.join(out_dir, "dose_response.csv"), index=False)
    print(f"  curves drawn at {DOSE_EFFECT_GRID} per SD; the table "
          f"below is at {DOSE_SUMMARY},")
    print("  stated as an assumption about a plausible weight rather "
          "than a measurement.")
    print(dose.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for t, g in dose.groupby("trial_type"):
        ax.errorbar(g.mag_bin, g.prop, yerr=g.sd, marker="o", capsize=3,
                    label=t)
    ax.axhline(0.5, color="k", lw=0.8, ls="--", label="chance")
    ax.set_xlabel("magnitude bin (1 = smallest difference)")
    ax.set_ylabel("P(chose the more manipulated image)")
    ax.set_title(f"Expected dose-response at {DOSE_SUMMARY} per SD "
                 f"(an assumption)  [{label}]\n(a flat line above "
                 "chance would indicate bias, not sensitivity; "
                 "dose_response_grid.csv has the others)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dose_response.png"), dpi=150)
    plt.close(fig)

    # Rise from the smallest to the largest bin, per type. Written out
    # without groupby.apply, which pandas now warns about.
    rises = {}
    for t, g in dose.groupby("trial_type"):
        g = g.sort_values("mag_bin")
        rises[t] = float(g.prop.iloc[-1] - g.prop.iloc[0])
    print("\n  rise from the smallest to the largest bin:")
    for t in TYPES:
        if t in rises:
            print(f"    {t:10s} {rises[t]:+.3f}")
    mean_rise = float(np.mean(list(rises.values())))
    report.append(("dose-response rise", f"{mean_rise:.3f}"))
    return dose, rises, mean_rise


# ============================================================
# Trial-count sweep  (depends on the trial list)
# ============================================================

def _pair_key(a, b):
    """A pair identified by its images, ignoring which is A and B."""
    return (a, b) if str(a) <= str(b) else (b, a)


def _draw(pool, n, rng, mag_col, cap):
    """A simplified version of step 5's stratified sampler.

    Same two rules that matter here: spread across magnitude bins, and
    cap how many a single scene may contribute (counted across the
    whole draw, not per bin). Returns the drawn rows, or fewer if the
    cap makes the full count impossible.
    """
    if not len(pool):
        return pool.head(0)
    pool = pool.copy()
    pool["_bin"] = pd.qcut(pool[mag_col].abs().rank(method="first"),
                           N_MAGNITUDE_BINS,
                           labels=range(1, N_MAGNITUDE_BINS + 1))
    per_bin = [n // N_MAGNITUDE_BINS] * N_MAGNITUDE_BINS
    for k in range(n % N_MAGNITUDE_BINS):
        per_bin[k] += 1

    order = list(range(1, N_MAGNITUDE_BINS + 1))
    rng.shuffle(order)
    out, taken, used = [], set(), {}
    for b, want in zip(order, per_bin):
        sub = pool[pool._bin == b]
        if not len(sub):
            continue
        sub = sub.sample(frac=1.0, random_state=rng.integers(1 << 30))
        rows = []
        for pos, (sid, fa, fb) in enumerate(
                zip(sub.scene_id, sub.file_a, sub.file_b)):
            if len(rows) >= want:
                break
            k = _pair_key(fa, fb)
            if k in taken or (cap is not None
                              and used.get(sid, 0) >= cap):
                continue
            rows.append(pos)
            taken.add(k)
            used[sid] = used.get(sid, 0) + 1
        out.append(sub.iloc[rows])
    return pd.concat(out, ignore_index=True) if out else pool.head(0)


def build_X(pairs, n_per_type, rng):
    """A trial list of n_per_type per type, as step 5 would build it.

    Mirrors step 5's structure: n sat_only, n fog_only, and n mixed
    split half together and half conflict, conflict drawn only from the
    balanced subset.

    Returns (X, cap_used, counts) or (None, cap, reason) if a pool
    cannot supply the count.
    """
    n_scenes = pairs.scene_id.nunique()
    cap = (SWEEP_SCENE_CAP if SWEEP_SCENE_CAP is not None
           else int(np.ceil(n_per_type / n_scenes)))
    n_conf = n_per_type // 2
    n_tog = n_per_type - n_conf

    conf = pairs[pairs.type == "conflict"]
    if "balanced" in pairs.columns:
        conf = conf[conf.balanced.astype(bool)]

    specs = [("sat_only", n_per_type, "d_sat",
              pairs[pairs.type == "sat_only"]),
             ("fog_only", n_per_type, "d_fog",
              pairs[pairs.type == "fog_only"]),
             ("together", n_tog, "d_fog",
              pairs[pairs.type == "together"]),
             ("conflict", n_conf, "d_fog", conf)]

    chunks, counts = [], {}
    for name, want, mag, pool in specs:
        got = _draw(pool, want, rng, mag, cap)
        counts[name] = len(got)
        if len(got) < want:
            return None, cap, (f"{name}: only {len(got)} of {want} at a "
                               f"cap of {cap} per scene")
        chunks.append(got)

    t = pd.concat(chunks, ignore_index=True)
    X = np.column_stack([
        -np.sign(t.d_sat) * t[PRED_SAT].abs(),
        np.sign(t.d_fog) * t[PRED_FOG].abs(),
    ]).astype(float)
    return X, cap, counts


def run_trial_count_sweep(pairs, label, out_dir, report):
    """What would more trials per person buy?

    Two different answers, because trial count acts on two levels.

    THE INDIVIDUAL FLOOR -- direct, no assumptions. Fit people who
    truly track nothing and measure how far their estimates scatter
    from zero. More trials shrink that scatter, and the floor with it,
    roughly as one over the square root of the count. This is what
    limits the per-person exploratory analysis.

    THE GROUP EFFECT SIZE -- indirect, and with a ceiling. A person's
    score mixes real differences between people with measurement noise.
    Only the noise shrinks with more trials, so once it is small
    compared with the real spread, further trials buy almost nothing at
    the group level. Because the real spread cannot be measured from
    your own design, this half is reported across a RANGE of
    assumptions and labelled as such.
    """
    print("\n" + "=" * 60)
    print(f"TRIAL-COUNT SWEEP  [{label}]")
    print("=" * 60)
    n_scenes = pairs.scene_id.nunique()
    print(f"  {n_scenes} scenes available")

    rng = np.random.default_rng(RANDOM_SEED + 7)
    rows = []
    for n in SWEEP_COUNTS:
        X, cap, info = build_X(pairs, n, rng)
        if X is None:
            print(f"  {n:4d} per type: NOT POSSIBLE -- {info}")
            rows.append({"n_per_type": n, "analysed_trials": 3 * n,
                         "scene_cap": cap, "possible": False})
            continue

        # The floor at this trial count, found the same way
        # compute_floors does it -- as a percentile of the guessers'
        # fitted weights, averaged over the two features. Using
        # 1.96 * SD here instead would make this column disagree with
        # the headline floors by about 10% on colour.
        est = fit_logit_batch(
            X, simulate_many(X, np.zeros((SWEEP_N_PEOPLE, 2)), rng))
        sd0 = float(est.std(axis=0).mean())
        floor = float(np.mean(
            np.percentile(est, 100 * (1 - FLOOR_ALPHA), axis=0)))

        # The group effect size, for each assumed real spread between
        # people. Measurement noise on a proportion from n trials has
        # SD sqrt(0.25 / n); the two sources add as variances.
        noise_sd = np.sqrt(0.25 / n)
        row = {"n_per_type": n, "analysed_trials": 3 * n,
               "scene_cap": cap, "possible": True,
               "weight_sd": round(sd0, 4),
               "min_detectable_weight": round(floor, 3),
               "proportion_noise_sd": round(float(noise_sd), 4)}
        for bsd in SWEEP_BETWEEN_SD:
            d = SWEEP_TRUE_PROP / np.sqrt(bsd ** 2 + noise_sd ** 2)
            row[f"d_if_spread_{bsd:g}"] = round(float(d), 3)
        rows.append(row)
        print(f"  {n:4d} per type ({3 * n:3d} analysed, cap {cap}): "
              f"floor {floor:.2f}, "
              + ", ".join(f"d={row[f'd_if_spread_{b:g}']:.2f} "
                          f"(spread {b:g})" for b in SWEEP_BETWEEN_SD))

    sw = pd.DataFrame(rows)
    sw.to_csv(os.path.join(out_dir, "trial_count_sweep.csv"),
              index=False)

    ok = sw[sw.possible]
    if len(ok):
        cur = ok[ok.n_per_type == 40]
        if len(cur):
            c = cur.iloc[0]
            print(f"\n  at the current 40 per type the floor is "
                  f"{c.min_detectable_weight:.2f}")
            for n in SWEEP_COUNTS:
                r = ok[ok.n_per_type == n]
                if len(r) and n > 40:
                    r = r.iloc[0]
                    print(f"    {n:3d} per type would bring it to "
                          f"{r.min_detectable_weight:.2f}  "
                          f"({r.analysed_trials} analysed trials, "
                          f"{(r.analysed_trials / 120 - 1) * 100:+.0f}% "
                          "session length)")
        report.append(("sweep: floor at 40/type",
                       f"{cur.iloc[0].min_detectable_weight:.2f}"
                       if len(cur) else "n/a"))
        _plot_sweep(ok, label, out_dir)
    return sw


def _plot_sweep(sw, label, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].plot(sw.analysed_trials, sw.min_detectable_weight,
                 marker="o", color="C3")
    cur = sw[sw.n_per_type == 40]
    if len(cur):
        axes[0].axvline(float(cur.analysed_trials.iloc[0]), color="k",
                        lw=0.8, ls="--", label="current design")
        axes[0].axhline(float(cur.min_detectable_weight.iloc[0]),
                        color="k", lw=0.8, ls=":")
    axes[0].set_xlabel("analysed trials per participant")
    axes[0].set_ylabel("smallest detectable individual weight")
    axes[0].set_title("What more trials buy ONE participant\n"
                      "(lower is better; falls as 1/sqrt(trials))")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    for k, bsd in enumerate(SWEEP_BETWEEN_SD):
        col = f"d_if_spread_{bsd:g}"
        if col in sw.columns:
            axes[1].plot(sw.analysed_trials, sw[col], marker="o",
                         color=f"C{k}",
                         label=f"real spread {bsd:g}")
    if len(cur):
        axes[1].axvline(float(cur.analysed_trials.iloc[0]), color="k",
                        lw=0.8, ls="--")
    axes[1].set_xlabel("analysed trials per participant")
    axes[1].set_ylabel("group effect size d")
    axes[1].set_title("What more trials buy the GROUP test\n"
                      "(flattens once noise is small next to the\n"
                      "real spread between people)")
    axes[1].legend(fontsize=9, title="assumed")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Trial-count sweep  [{label}]   "
                 f"(assumed average preference "
                 f"{SWEEP_TRUE_PROP:+.2f} above chance)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trial_count_sweep.png"), dpi=150)
    plt.close(fig)


# ============================================================
# Group-level power  (shared: depends only on N and d)
# ============================================================

def run_power(root, report):
    """Power of the one-sample t-test on the per-participant
    proportions, across sample sizes and effect sizes.

    The unit of analysis is the participant, as in Suzuki et al.
    (2024): each contributes one proportion, and the test runs across
    people. So each person's own trial count sharpens their proportion,
    while the number of PEOPLE drives this curve -- which is why this
    table is a property of the TEST and not of either trial list.
    """
    print("\n" + "=" * 60)
    print("GROUP-LEVEL POWER  (shared by both sets)")
    print("=" * 60)

    rng = np.random.default_rng(RANDOM_SEED + 1)
    crit_fn = ((lambda df: stats.t.ppf(1 - ALPHA, df)) if ONE_TAILED
               else (lambda df: stats.t.ppf(1 - ALPHA / 2, df)))

    rows = []
    for n in SAMPLE_SIZES:
        for d in EFFECT_SIZES:
            x = rng.normal(d, 1.0, (N_SIM_STUDIES, n))
            t = x.mean(1) / (x.std(1, ddof=1) / np.sqrt(n))
            hit = ((t > crit_fn(n - 1)) if ONE_TAILED
                   else (np.abs(t) > crit_fn(n - 1)))
            rows.append({"n": n, "d": d, "power": float(hit.mean())})

    pw = pd.DataFrame(rows)
    pw.to_csv(os.path.join(root, "power_curve.csv"), index=False)
    print("\n  power (one-sample t-test vs chance):")
    print(pw.pivot(index="n", columns="d", values="power")
          .round(2).to_string())

    at10 = pw[pw.n == 10].set_index("d").power
    # Quote whatever EFFECT_SIZES actually holds rather than naming
    # d = 0.5 and d = 1.0, which silently became nan the moment the
    # grid was edited.
    for d in EFFECT_SIZES:
        if d in at10.index:
            report.append((f"power at N=10, d={d:g}",
                           f"{at10.loc[d]:.2f}"))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for d, g in pw.groupby("d"):
        ax.plot(g.n, g.power, marker="o", label=f"d = {d}")
    ax.axhline(0.8, color="k", lw=0.8, ls="--", label="80%")
    ax.set_xlabel("participants")
    ax.set_ylabel("power")
    ax.set_title("Detecting a group effect on the proportions\n"
                 "(one-sample t-test against chance)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "power_curve.png"), dpi=150)
    plt.close(fig)
    return pw


# ============================================================
# Bayes factors  (shared: depends only on t and N)
# ============================================================

def run_bayes(root, report):
    """What Bayes factor should be expected, under an effect and under
    a true null?

    The second is the important one. If a null is the likely outcome,
    the question is how often the analysis could say "moderate evidence
    AGAINST an effect" rather than "inconclusive".
    """
    print("\n" + "=" * 60)
    print("BAYES FACTORS  (do not depend on the trial list)")
    print("=" * 60)
    print(f"  JZS, Cauchy prior scale {BF_SCALE}, "
          + ("ONE-SIDED (positive effects only)" if BF_ONE_SIDED
             else "two-sided")
          + f", BF threshold {BF_THRESHOLD:g}")
    if BF_ONE_SIDED:
        print("  The null-evidence figures below are regenerated with "
              "this factor:")
        print("  BF01 from a two-sided prior is not the complement of "
              "a one-sided BF10,")
        print("  so a two-sided number could not be reused here.")

    rng = np.random.default_rng(RANDOM_SEED + 2)
    n_studies = min(N_SIM_STUDIES, 600)   # quad() is the slow part
    rows = []
    for n in SAMPLE_SIZES:
        for d in [0.0] + EFFECT_SIZES:
            x = rng.normal(d, 1.0, (n_studies, n))
            t = x.mean(1) / (x.std(1, ddof=1) / np.sqrt(n))
            bf = np.array([jzs_bayes_factor(ti, n) for ti in t])
            bf = bf[np.isfinite(bf)]
            rows.append({
                "n": n, "d": d,
                "median_BF10": float(np.median(bf)),
                "p_evidence_for": float((bf > BF_THRESHOLD).mean()),
                "p_evidence_against":
                    float((bf < 1 / BF_THRESHOLD).mean()),
                "p_inconclusive": float(
                    ((bf >= 1 / BF_THRESHOLD)
                     & (bf <= BF_THRESHOLD)).mean()),
            })

    bfs = pd.DataFrame(rows)
    bfs.to_csv(os.path.join(root, "bayes_factors.csv"), index=False)

    null = bfs[bfs.d == 0.0]
    print("\n  under a TRUE NULL -- can the study say 'no effect'?")
    print(null[["n", "median_BF10", "p_evidence_against",
                "p_inconclusive"]].round(3).to_string(index=False))

    row10 = null[null.n == 10]
    if len(row10):
        r = row10.iloc[0]
        print(f"\n  At N = 10 with no true effect: "
              f"{r.p_evidence_against:.0%} of studies would reach "
              f"BF01 > {BF_THRESHOLD:g} ('moderate evidence against'), "
              f"{r.p_inconclusive:.0%} would be inconclusive.")
        report.append(("P(evidence against null) at N=10",
                       f"{r.p_evidence_against:.0%}"))

    print("\n  under an EFFECT -- can the study say 'yes'?")
    print(bfs[bfs.d > 0].pivot(index="n", columns="d",
                               values="p_evidence_for")
          .round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    g = bfs[bfs.d == 0.0].sort_values("n")
    axes[0].plot(g.n, g.p_evidence_against, marker="o", color="C3",
                 label="evidence against an effect")
    axes[0].plot(g.n, g.p_inconclusive, marker="s", color="C7",
                 label="inconclusive")
    axes[0].set_xlabel("participants")
    axes[0].set_ylabel("proportion of studies")
    axes[0].set_title("Under a TRUE NULL\n"
                      f"(can the study reach BF01 > {BF_THRESHOLD:g}?)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    for d, g in bfs[bfs.d > 0].groupby("d"):
        axes[1].plot(g.n, g.p_evidence_for, marker="o", label=f"d = {d}")
    axes[1].set_xlabel("participants")
    axes[1].set_ylabel("P(BF10 > threshold)")
    axes[1].set_title("Under an EFFECT")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "bayes_factors.png"), dpi=150)
    plt.close(fig)
    return bfs


# ============================================================
# One set
# ============================================================

def analyse_one(label, path, pairs_path, root):
    print("\n" + "#" * 60)
    print(f"# {label}")
    print("#" * 60)
    out_dir = os.path.join(root, label)
    os.makedirs(out_dir, exist_ok=True)

    trials, X, r = load_trials(path, label)
    report = [("analysed trials", str(len(trials))),
              ("r(predictors)", f"{r:+.3f}")]

    th = compute_floors(X, label, report)
    check_threshold_calibration(X, th, report)
    rec, clouds, acc, sd0, mde = run_recovery(X, th, label, out_dir,
                                              report)
    det, needed = run_detection(X, th, label, out_dir, report)
    con = run_contrasts(X, label, out_dir, report) if RUN_CONTRASTS \
        else None
    dose, rises, mean_rise = run_dose(trials, X, label, out_dir, report)

    sweep = pd.DataFrame()
    if RUN_TRIAL_COUNT_SWEEP and pairs_path:
        if not os.path.isfile(pairs_path):
            print(f"\n  {pairs_path} not found; the trial-count sweep "
                  "is skipped. Point the third entry of TRIAL_LISTS at "
                  "this set's all_pairs.csv from step 4.")
        else:
            pairs = pd.read_csv(pairs_path)
            sweep = run_trial_count_sweep(pairs, label, out_dir, report)

    return {"label": label, "path": path, "trials": trials, "X": X,
            "r": r, "rec": rec, "acc": acc, "sd0": sd0, "mde": mde,
            "det": det, "thresholds": th, "needed": needed,
            "con": con,
            "dose": dose, "rises": rises, "mean_rise": mean_rise,
            "sweep": sweep, "report": report, "out_dir": out_dir}


def build_comparison(results, root):
    """The set-dependent quantities, side by side.

    Only these differ between the sets. The power curve and the Bayes
    factor tables are properties of the test and are identical.
    """
    rows = {}

    def put(k, label, v):
        rows.setdefault(k, {})[label] = v

    for res in results:
        L = res["label"]
        put("analysed trials", L, len(res["trials"]))
        put("r(predictors)", L, round(res["r"], 3))
        put("VIF", L, round(1 / (1 - res["r"] ** 2), 2))
        put("classification accuracy", L, round(res["acc"], 4))
        put("individual weight SD", L, round(res["sd0"], 3))
        put("min detectable weight", L, round(res["mde"], 3))
        put("threshold upper, colour", L,
            round(float(res["thresholds"]["upper"][0]), 3))
        put("threshold upper, fog", L,
            round(float(res["thresholds"]["upper"][1]), 3))
        put("threshold lower, colour", L,
            round(float(res["thresholds"]["lower"][0]), 3))
        put("threshold lower, fog", L,
            round(float(res["thresholds"]["lower"][1]), 3))
        put(f"weight for {DETECTION_TARGET:.0%} detection, colour", L,
            round(res["needed"]["colour"], 3))
        put(f"weight for {DETECTION_TARGET:.0%} detection, fog", L,
            round(res["needed"]["fog"], 3))
        if res.get("con") is not None:
            c = res["con"]
            put("crossover difference SD", L, round(c["sd_diff"], 3))
            put(f"gap for {CONTRAST_TARGET:.0%} crossover detection", L,
                round(c["gap_needed"], 3))
            put(f"gamma for {CONTRAST_TARGET:.0%} detection", L,
                round(c["gamma_needed"], 3))
        for t in TYPES:
            put(f"dose rise, {t}", L, round(res["rises"].get(t, np.nan), 3))
        put("dose rise, mean", L, round(res["mean_rise"], 3))

    comp = pd.DataFrame(rows).T
    comp.index.name = "quantity"
    comp.to_csv(os.path.join(root, "comparison.csv"))
    return comp


def comparison_figure(results, root):
    """The two sets side by side on what distinguishes them."""
    if len(results) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    labels = [r["label"] for r in results]
    x = np.arange(len(labels))

    axes[0].bar(x, [r["mde"] for r in results], 0.6,
                color=[f"C{i}" for i in range(len(results))])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("weight")
    axes[0].set_title("Smallest individual weight\n"
                      "distinguishable from zero (lower is better)")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, [r["acc"] for r in results], 0.6,
                color=[f"C{i}" for i in range(len(results))])
    axes[1].axhline(0.5, color="k", lw=0.8, ls="--", label="chance")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Telling a colour-tracker from\na fog-tracker")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")

    w = 0.8 / len(results)
    for k, res in enumerate(results):
        v = [res["rises"].get(t, np.nan) for t in TYPES]
        axes[2].bar(np.arange(len(TYPES)) + k * w, v, w,
                    label=res["label"])
    axes[2].set_xticks(np.arange(len(TYPES)) + w * (len(results) - 1) / 2)
    axes[2].set_xticklabels(TYPES, rotation=20)
    axes[2].set_ylabel("rise, smallest to largest bin")
    axes[2].set_title("Dose-response rise\n(steeper is stronger evidence)")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(root, "comparison.png"), dpi=150)
    plt.close(fig)


# ============================================================
# Report
# ============================================================

def write_report(results, comp, pw, bfs, shared_report, root, t0,
                 reg=None, seeds=None, legacy=None):
    path = os.path.join(root, "power_report.txt")
    with open(path, "w") as f:
        def W(s=""):
            f.write(s + "\n")

        def T(df, indent="  "):
            """A dataframe, indented, or a line saying it is absent."""
            if df is None or not len(df):
                W(indent + "(not run)")
                return
            W(indent + df.to_string(index=False)
              .replace("\n", "\n" + indent))

        def TI(df, indent="  "):
            """The same, keeping the index (for grouped tables)."""
            if df is None or not len(df):
                W(indent + "(not run)")
                return
            W(indent + df.to_string().replace("\n", "\n" + indent))

        def lines(rep, *keys):
            """The named narrative lines, as the run computed them."""
            d = dict(rep)
            for k in keys:
                if k in d:
                    W(f"  {k:40s} {d[k]}")

        W("POWER SIMULATION REPORT  (v4 axes, step 8)")
        W("=" * 70)
        for r in results:
            W(f"{r['label']:10s} {r['path']}  "
              f"({len(r['trials'])} analysed trials)")
        W(f"Prior      : JZS, scale {BF_SCALE}, "
          + ("ONE-SIDED (the registered form)" if BF_ONE_SIDED
             else "two-sided"))
        W(f"Test       : one-sample t, alpha {ALPHA}, "
          f"{'one' if ONE_TAILED else 'two'}-tailed")
        W(f"Elapsed    : {time.time() - t0:.1f}s")
        W()

        if len(results) > 1:
            W("1. THE SETS SIDE BY SIDE")
            W("-" * 70)
            W("  Only these quantities depend on the trial list. The")
            W("  power curve and the Bayes factor tables below depend")
            W("  only on the sample size and the effect size, so they")
            W("  are identical for both sets and are computed once.")
            W()
            W(comp.to_string())
            W()
            W("  The number to read first is the smallest individual")
            W("  weight distinguishable from zero. It is the noise")
            W("  floor on a single participant's estimate, and it is")
            W("  the one thing a better stimulus set can improve.")
            W()

        for i, r in enumerate(results, 2 if len(results) > 1 else 1):
            W(f"{i}. {r['label'].upper()}")
            W("-" * 70)
            for k, v in r["report"]:
                W(f"  {k:34s} {v}")
            W()
            W("  recovered weights per scenario:")
            W("  " + r["rec"].round(3).to_string(index=False)
              .replace("\n", "\n  "))
            W()
            W("  PER-PERSON DETECTION")
            W("  The floor is where a person can JUST be told apart")
            W("  from a guesser -- at exactly the floor you catch them")
            W("  about half the time. The weight below is where you")
            W(f"  catch them {DETECTION_TARGET:.0%} of the time, which")
            W("  is the number to plan with.")
            W()
            for feat, k in [("colour", 0), ("fog", 1)]:
                v = r["needed"][feat]
                if np.isfinite(v):
                    half = 1.0 / (1.0 + np.exp(-v * 0.5))
                    full = 1.0 / (1.0 + np.exp(-v))
                    W(f"    {feat:7s} threshold "
                      f"{r['thresholds']['upper'][k]:.2f}, "
                      f"{DETECTION_TARGET:.0%} detection at {v:.2f}")
                    W(f"            = choosing the manipulated image "
                      f"{half:.0%} of the time at a typical")
                    W(f"              difference, {full:.0%} at the "
                      "largest")
            W()
            W("  " + r["det"].round(3).to_string(index=False)
              .replace("\n", "\n  "))
            W()
            if r.get("con") is not None:
                c = r["con"]
                W("  CROSSOVER: IS FOG BIGGER THAN COLOUR, IN ONE")
                W("  PERSON?")
                W("  The floors above each ask whether ONE weight")
                W("  differs from zero. This asks whether the TWO")
                W("  weights differ from each other, which is the")
                W("  actual crossover hypothesis -- and it is a harder")
                W("  question, because the noise on a difference is")
                W("  bigger than the noise on either part.")
                W()
                W(f"    simulated at a baseline weight of "
                  f"{CONTRAST_SUMMARY} on both cues (an "
                  "assumption)")
                W(f"    scatter of the difference      "
                  f"{c['sd_diff']:.3f}")
                W(f"    (the two scatters in quadrature "
                  f"{c['sd_diff_quadrature']:.3f} -- close, so the")
                W("     usual back-of-envelope is a fair guide here)")
                W(f"    critical values                "
                  f"{c['crit_diff'][0]:+.2f} .. "
                  f"{c['crit_diff'][1]:+.2f}")
                if np.isfinite(c["gap_needed"]):
                    W(f"    gap caught {CONTRAST_TARGET:.0%} of the "
                      f"time        {c['gap_needed']:.2f}")
                    W("      Compare that with the single-feature")
                    W("      numbers above. A gap this size is the")
                    W("      price of the CROSSOVER claim; a person can")
                    W("      clear the fog floor, miss the colour")
                    W("      floor, and still not clear this.")
                else:
                    W("    the grid never reaches the target; widen")
                    W("    CROSSOVER_GRID")
                W()
                W("  " + c["crossover"].round(3).to_string(index=False)
                  .replace("\n", "\n  "))
                W()
                W("  ADDITIVITY: DO THE TWO CUES SIMPLY ADD?")
                W("  What the `together` and `conflict` trials are")
                W("  really for. gamma is the proportional change in")
                W("  the combined evidence at the largest differences:")
                W("  +0.5 means the two cues together carry 50% more")
                W("  weight than their sum, -0.5 means 50% less, and 0")
                W("  is additive. The two-weight model used everywhere")
                W("  else ASSUMES gamma = 0, so it cannot test this.")
                W()
                W(f"    additivity-term VIF            "
                  f"{c['vif3'][2]:.2f}")
                W(f"    critical values                "
                  f"{c['crit_gamma'][0]:+.2f} .. "
                  f"{c['crit_gamma'][1]:+.2f}")
                if np.isfinite(c["gamma_needed"]):
                    W(f"    gamma caught {CONTRAST_TARGET:.0%} of the "
                      f"time      {c['gamma_needed']:.2f}")
                    W("      i.e. the cues would have to combine this")
                    W("      much more strongly than additively before")
                    W("      ONE participant's data could show it.")
                else:
                    W("    the grid never reaches the target; widen")
                    W("    GAMMA_GRID")
                W()
                W("  " + c["additivity"].round(3)
                  .to_string(index=False).replace("\n", "\n  "))
                W()
                W("  ADDITIVITY ACROSS PEOPLE")
                W("  The mixed trials pool: n participants give n times")
                W("  as many of them. This is the same question asked")
                W("  of n people at once, and it is the only form in")
                W("  which it is answerable.")
                W()
                W("  " + c["pooled"].round(3).to_string(index=False)
                  .replace("\n", "\n  "))
                W()
                W("  Treats everyone as sharing one gamma, so it is an")
                W("  UPPER BOUND on what a mixed-effects fit would")
                W("  manage -- real people differ, and that spread is")
                W("  not simulated. Read it as 'no better than this'.")
                W()
            if len(r["dose"]):
                W("  expected dose-response:")
                W("  " + r["dose"].round(3).to_string(index=False)
                  .replace("\n", "\n  "))
            W()
            if len(r["sweep"]):
                W("  WOULD MORE TRIALS HELP?")
                W("  n_per_type is step 5's setting (currently 40).")
                W("  The d_if_spread_* columns assume an average")
                W(f"  preference of {SWEEP_TRUE_PROP:+.2f} above chance")
                W("  and the stated real spread between people -- that")
                W("  spread is a property of the population, not of")
                W("  your design, so it is an assumption, not a result.")
                W()
                W("  " + r["sweep"].to_string(index=False)
                  .replace("\n", "\n  "))
                W()
                W("  Read the two halves separately:")
                W("    min_detectable_weight falls as 1/sqrt(trials)")
                W("      and is the ceiling on the PER-PERSON analysis.")
                W("    d barely moves once the measurement noise")
                W("      (proportion_noise_sd) is small beside the real")
                W("      spread, because more trials cannot shrink")
                W("      genuine differences between people.")
                W()

        n = len(results) + (2 if len(results) > 1 else 1)
        W(f"{n}. GROUP-LEVEL POWER")
        W("-" * 70)
        for k, v in shared_report:
            W(f"  {k:34s} {v}")
        W()
        W("  Power (one-sample t-test):")
        W("  " + pw.pivot(index="n", columns="d", values="power")
          .round(2).to_string().replace("\n", "\n  "))
        W()
        W(f"{n + 1}. BAYES FACTORS UNDER A TRUE NULL")
        W("-" * 70)
        W("  The important table. A p-value cannot distinguish 'no")
        W("  effect' from 'not enough data'; this says how often the")
        W("  study could report evidence FOR the null instead of")
        W("  reporting nothing.")
        W()
        W("  " + bfs[bfs.d == 0.0][
            ["n", "median_BF10", "p_evidence_against", "p_inconclusive"]
        ].round(3).to_string(index=False).replace("\n", "\n  "))
        W()
        # ----------------------------------------------------
        # The registered sections. Numbered from here on with a
        # counter, so adding one does not renumber the rest by hand.
        # ----------------------------------------------------
        sec = [n + 2]

        def H(title):
            W(f"{sec[0]}. {title}")
            W("-" * 70)
            sec[0] += 1

        g = reg or {}
        r0 = results[0]
        rep0 = r0["report"]

        H("THE DECISION RULE FOR THE EXISTENCE CLAIM")
        if "rule_summary" not in g:
            W("  (RUN_RULE_SWEEP was off; this run is not the "
              "registered one)")
            W()
        else:
            W("  The claim is that SOME people weight the feature.")
            W("  With a per-person threshold calibrated on guessers,")
            W("  a handful of detections is expected under any null,")
            W("  so the rule has to say how many are enough.")
            W()
            W("  THREE FAMILIES WERE COMPARED:")
            W("    count-absolute      at least k detections and at")
            W("                        most m in the wrong direction.")
            W("                        DEGENERATES with sample size --")
            W("                        a fixed m that is safe at n = 8")
            W("                        is almost never met at n = 120,")
            W("                        so power rises and then falls.")
            W("    count-proportional  the same, with m scaled to n.")
            W("    sign test           the detections' own asymmetry.")
            W("                        Its false-claim rate is BOUNDED")
            W("                        BY alpha for any symmetric")
            W("                        null, whatever the per-tail")
            W("                        rate, which is a stronger")
            W("                        guarantee than the others have.")
            W("                        It cannot fire below 5")
            W("                        detections at all.")
            W()
            W("  TWO ADMISSIBILITY STANDARDS, both reported:")
            W("    practical  the rule must hold against a guesser")
            W("               and against taste spreads of 0.25x and")
            W("               0.5x each feature's own threshold.")
            W("    strict     and against 1.0x as well, i.e. a")
            W("               population whose idle preferences are")
            W("               as large as the detection threshold.")
            W()
            lines(rep0, *[k for k, _ in rep0
                          if k.startswith(("rule", "sign", "admiss",
                                           "count"))])
            W()
            W("  the most powerful admissible rule, per family:")
            T(g["rule_summary"])
            W()

        H("WHY THE STRICT STANDARD, AND NOT THE CHEAPER OPTION")
        if "control_gated" not in g:
            W("  (the control check was not run)")
            W()
        else:
            W("  THE REGISTERED STANDARD IS "
              f"{REGISTERED_STANDARD.upper()}, chosen outright and")
            W("  not from the data. This section is the evidence for")
            W("  that choice.")
            W()
            W("  The cheaper option was costed first. The practical")
            W("  standard has more power, and it is admissible")
            W("  against a guesser and against taste spreads of")
            W("  0.25x and 0.5x -- but it reaches 0.363 against a 1x")
            W("  spread, which it is not required to hold against.")
            W("  The proposal was to let the CONTROL GROUP decide:")
            W("  estimate the taste spread, and if it comes back")
            W("  above 0.75x of a feature's own threshold, evaluate")
            W("  that feature's claim with the strict rule instead.")
            W()
            W("  IT DOES NOT WORK, AND THE REASON IS GENERAL. The")
            W("  rule would be chosen from data, so neither branch's")
            W("  rate is the procedure's rate:")
            W()
            W("      combined = P(strict) x FP_strict")
            W("               + P(practical) x FP_practical")
            W()
            W("  The branch is decided by the CONTROL sample and the")
            W("  rule applied to the PARTICIPANT sample -- different")
            W("  people -- so the two factorise exactly and the")
            W("  combined rate is computable rather than simulated")
            W("  end to end.")
            W()
            W("  The combined rate PEAKS AT THE CHECK'S OWN")
            W("  BOUNDARY. When the true spread sits on 0.75x, the")
            W("  estimate lands either side about half the time, so")
            W("  half the studies apply the practical rule to a")
            W("  population it was never required to hold against.")
            W("  Past the boundary the check almost always fires and")
            W("  the rate falls back; below it the practical rule is")
            W("  admissible anyway. The worst case is the middle.")
            W()
            W("  AND IT DOES NOT IMPROVE WITH MORE CONTROLS. Below")
            W("  the minimum the procedure forces the strict rule and")
            W("  is safe by construction, so recruiting the thirtieth")
            W("  control moves the study OUT of a safe regime and")
            W("  into the noisy-branch one.")
            W()
            lines(rep0, *[k for k, _ in rep0
                          if k.startswith(("procedure", "registered "
                                           "procedure", "control "
                                           "check", "variant"))])
            W()
            W("  All three variants, and what each would have cost:")
            W()
            T(g["control_gated"])
            W()
            if "control_branch" in g:
                W("  how often the branch would have been taken, "
                  "against the true spread:")
                W()
                T(g["control_branch"])
                W()
            W("  THE CONTROL-BASED SPREAD ESTIMATE IS STILL")
            W("  REPORTED, DESCRIPTIVELY. It is the only measurement")
            W("  of the quantity every rule is judged against, and")
            W("  the next section says how well it can be measured at")
            W("  all. No registered claim depends on it.")
            W()
            W("  The strategy test is reported with it. Taste is")
            W("  symmetric by construction and the shared-concept")
            W("  strategy is not, so an excess of predicted over")
            W("  opposite classifications among the controls says the")
            W("  spread estimate is inflated. Since nothing branches")
            W("  on it any more, it is a diagnostic: it tells a")
            W("  reader whether the control group behaved like the")
            W("  null the rules were judged against.")
            W()

        H("THE SAME CLAIMS, STATED BEHAVIOURALLY")
        if "equal" not in g:
            W("  (RUN_EQUAL_BEHAVIOUR was off)")
            W()
        else:
            W("  THE PRIMARY FORM OF EVERY EXISTENCE CLAIM.")
            W()
            W("  A per-SD weight is not comparable between the two")
            W("  features on this list. The informative colour trials")
            W("  are about 1.0 corpus SD apart and the fog trials")
            W("  about 0.62, so the same weight produces a much")
            W("  smaller choice bias on fog, and the fog threshold")
            W("  sits correspondingly higher -- the ratio of the two")
            W("  thresholds is the square root of the ratio of the")
            W("  two sums of squares, which is arithmetic, not an")
            W("  asymmetry of the design.")
            W()
            W("  Stated as a choice probability on a typical")
            W("  informative trial, the two features' claims are")
            W("  equivalent: the residual between them is a few")
            W("  thousandths across the grid. The per-SD table is")
            W("  kept beside it because the weights are what the")
            W("  model fits, but THE BEHAVIOURAL COLUMN IS THE CLAIM.")
            W()
            lines(rep0, *[k for k, _ in rep0
                          if "behav" in k or "equal" in k
                          or "median |pred|" in k])
            W()
            TI(g["equal"])
            W()

        H("PREVALENCE")
        if "prevalence" not in g:
            W("  (RUN_PREVALENCE was off)")
            W()
        else:
            W("  The rules above answer 'did anyone?'. This answers")
            W("  'what share?', on the same detections.")
            W()
            W("  theta = gamma(1 - beta) + (1 - gamma)alpha: the")
            W("  observed detection rate is the true prevalence times")
            W("  the sensitivity plus the rest times the")
            W("  false-positive rate (Ince et al., eLife 2021).")
            W()
            W("  TWO VARIANTS, and the registered one is the second:")
            W("    fixed alpha      alpha is the calibrated per-tail")
            W("                     rate. Correct only if the")
            W("                     population really does guess.")
            W("    estimated alpha  alpha is estimated from the")
            W("                     OPPOSITE-direction")
            W("                     classifications in the same data.")
            W("                     Under a taste null those are")
            W("                     inflated too, so the estimate")
            W("                     moves with the contamination")
            W("                     instead of ignoring it.")
            W()
            lines(rep0, *[k for k, _ in rep0 if "preval" in k.lower()])
            W()
            pv = g["prevalence"]
            W("  THE DECISION-RELEVANT NUMBER: how many of n people")
            W("  have to be classified as detecting the feature")
            W("  before the posterior's lower bound clears zero --")
            W("  i.e. before 'some people do' is supportable at all.")
            W()
            W(f"  ({int(PREVALENCE_HPDI * 100)}% HPDI; '-' means no "
              "count up to n is enough.")
            W("   Each variant assumes a different false-positive")
            W("   rate: a pure guesser, or a population whose idle")
            W("   preferences have the stated spread relative to that")
            W("   feature's own threshold.)")
            W()
            need = []
            for (feat, eff, var, nn), gg in pv.groupby(
                    ["feature", "effect_sd", "variant", "n"]):
                clear = gg[gg.hpdi_lo > 0]
                need.append({
                    "feature": feat, "effect_sd": eff,
                    "variant": var, "n": nn,
                    "alpha": round(float(gg.alpha.iloc[0]), 4),
                    "detections_needed":
                        int(clear.k_positive.min())
                        if len(clear) else -1})
            nd = pd.DataFrame(need)
            # One row per (feature, variant, n): the detections needed
            # do not depend on the effect size, which only sets the
            # sensitivity used for the point estimate, so if the
            # effect sizes disagree the table shows both.
            piv = nd.pivot_table(index=["feature", "variant", "alpha"],
                                 columns="n",
                                 values="detections_needed",
                                 aggfunc="max")
            TI(piv.replace(-1, np.nan).astype(object)
               .where(piv != -1, "-"), indent="    ")
            W()
            W("  The jump between variants is the whole point. A rule")
            W("  calibrated on guessers needs very few detections; the")
            W("  same rule needs several more once the null population")
            W("  is allowed idle preferences, and the registered")
            W("  variant below does not have to guess which it is.")
            W()
            W("  THE REGISTERED VARIANT ASSUMES NO alpha AT ALL.")
            W()
            W("  Under any symmetric null a non-responder is as")
            W("  likely to be classified in the opposite direction as")
            W("  in the predicted one, so the OPPOSITE count measures")
            W("  the contamination directly. A responder contributes")
            W("  almost nothing to it -- a real preference does not")
            W("  also look backwards, measured at 0.0000 on this")
            W("  design.")
            W()
            W("  The model is the one above, written in the rate the")
            W("  data can see:")
            W()
            W("      c      = (1 - prevalence) x alpha")
            W("      p_pred = prevalence x sensitivity + c")
            W("      p_opp  =                            c")
            W()
            W("  with (predicted, opposite, neither) trinomial in")
            W("  those, and the posterior on the prevalence")
            W("  marginalised over c.")
            W()
            W("  A moment estimator falls out of the same algebra and")
            W("  needs no alpha either, because the non-responders'")
            W("  contributions to the two counts cancel:")
            W()
            W("      prevalence_hat = (predicted - opposite)")
            W("                       / (n x sensitivity)")
            W()
            W("  exactly unbiased before clipping. 'E_moment' below is")
            W("  that estimator; 'E_mode' and the interval are the")
            W("  posterior's. 'P_excludes_zero' at a true prevalence")
            W("  of 0 IS the prevalence claim's own false-positive")
            W("  rate, computed with nothing assumed about the null.")
            W()
            W("  WHAT THIS REPLACED, AND WHY. An earlier version")
            W("  estimated alpha as (opposite count) / n and fed it to")
            W("  the fixed-alpha posterior. Only the NON-responders")
            W("  produce opposite classifications, while that")
            W("  denominator counts everyone, so the estimate")
            W("  converges to (1 - prevalence) x alpha -- biased low")
            W("  by the very quantity being estimated.")
            W("  'old_bias_factor' is E[alpha_hat] / alpha_true for")
            W("  each cell; it equals (n - responders) / n to three")
            W("  decimals, which identifies the mechanism rather than")
            W("  leaving it as an unexplained discrepancy.")
            W()
            W("  PARAMETERISING BY alpha INSTEAD DOES NOT WORK, and")
            W("  the attempt is on the record: with a prior on alpha")
            W("  and a joint posterior over (prevalence, alpha), the")
            W("  two are not separately identified at these sample")
            W("  sizes -- p_opp carries a (1 - prevalence) factor, so")
            W("  a prevalence near 1 drives p_opp to zero, fits an")
            W("  observed opposite count of zero perfectly and")
            W("  maximises p_pred at the same time. With a true")
            W("  prevalence of 0 and a 1x taste spread that")
            W("  parameterisation returned a posterior mode of 0.86.")
            W("  In (prevalence, c) the opposite count pins c down on")
            W("  its own and the ridge is gone.")
            W()
            pe = g.get("prevalence_estimated")
            T(pe)
            W()
            lines(rep0, "prevalence moment estimator",
                  "prevalence, interval excludes zero under a null",
                  "prevalence, rejected alpha_hat bias")
            W()
            W("  The full posterior for every count, variant and size")
            W("  is in prevalence.csv.")
            W()

        H("CAN THE CONTROLS' TASTE SPREAD BE MEASURED?")
        if "taste" not in g:
            W("  (RUN_TASTE_RECOVERY was off)")
            W()
        else:
            W("  Every rule above is judged against a null with an")
            W("  assumed taste spread. The control group is what")
            W("  would measure it, so this asks how well -- and the")
            W("  answer decides whether the assumption is checkable")
            W("  or merely stated.")
            W()
            lines(rep0, *[k for k, _ in rep0 if "taste" in k.lower()])
            W()
            T(g["taste"])
            W()

        H("THE JOINT LAPSE MODEL: A NEGATIVE RESULT")
        if "lapse" not in g:
            W("  (RUN_LAPSE_RECOVERY was off)")
            W()
        else:
            W("  RECORDED BECAUSE THE GATE'S DESIGN DEPENDS ON IT.")
            W()
            W("  The obvious way to handle inattention is to fit it:")
            W("  p = (1 - lambda)logistic(Xb) + lambda/2, three")
            W("  parameters per person. Tried, and it does not")
            W("  identify on this design.")
            W()
            W("  The fitter was verified first, so this is a property")
            W("  of the design and not of the code: at 7200 trials")
            W("  per person it recovers the weight and beats the")
            W("  likelihood at the true parameters. At 180 trials the")
            W("  log-likelihood moves by under a tenth of a unit")
            W("  across lambda from 0 to 0.4, and a group-level")
            W("  lambda shared across 30 people leaves everything")
            W("  from 0 to 0.25 within two units of the best.")
            W()
            W("  The reason is structural: lapses attenuate the")
            W("  fitted weight by about (1 - lambda), and that")
            W("  attenuation is their ONLY signature in choice data --")
            W("  exactly degenerate with a smaller true weight. A")
            W("  binary 2AFC task with no confidence ratings has no")
            W("  second moment to break the tie.")
            W()
            W("  So C1 does not get a lapse-corrected sensitivity")
            W("  analysis. It gets the gate instead, which uses the")
            W("  catch trials -- a measurement the weights cannot")
            W("  imitate, because a catch trial has no images and no")
            W("  preference can produce a failure on one.")
            W()
            lines(rep0, *[k for k, _ in rep0 if "lapse" in k.lower()])
            W()
            T(g["lapse"])
            W()

        H("THE ATTENTION GATE")
        if "gate" not in g:
            W("  (RUN_GATE was off)")
            W()
        else:
            W("  WHAT IT IS FOR. A control group that lapses more")
            W("  than the participants has its weights attenuated")
            W("  toward zero, which widens exactly the gap C1 looks")
            W("  for. Nothing in the choice data distinguishes that")
            W("  from a real effect, and the lapse model above")
            W("  cannot either.")
            W()
            W("  TWO MEASURES, Holm-corrected across the two so the")
            W("  gate's own false-alarm rate is the alpha it claims:")
            W("    catch failures   the primary measure. Needs")
            W("                     nothing but the catch block.")
            W("    fast responses   the per-person share of image")
            W(f"                     trials answered under "
              f"{RT_FAST_CUTOFF_MS} ms,")
            W("                     fixed in advance. A trial needs a")
            W("                     saccade and the fixation to")
            W("                     programme it (about 200 ms) plus")
            W("                     evidence accumulation before a")
            W("                     response can be initiated, so a")
            W("                     faster answer cannot have")
            W("                     involved comparing the two")
            W("                     pictures. It is an implausibility")
            W("                     floor, NOT an engagement")
            W("                     threshold.")
            W()
            W("  THE SIMULATED SENSITIVITY IS AN UPPER BOUND. It")
            W("  assumes people lapse on catch trials at the same")
            W("  rate as on image trials (p_fail = "
              f"{CATCH_P_FAIL_PER_LAPSE} x lambda, from")
            W("  four live keys). A catch screen has no pictures and")
            W("  one line of text, so someone skimming may well")
            W("  attend to it more; if they lapse less there, the")
            W("  gate is less sensitive than these numbers.")
            W()
            W("  THE RESPONSE-TIME CUT-OFF MUST NOT BE RE-TUNED. It")
            W("  is reasoned from saccade latency, not measured on")
            W("  this task. Report the first participants' response")
            W("  time distribution against it; if the mode sits near")
            W("  the cut-off, the cut is wrong for this task and the")
            W("  measure should be DROPPED, not adjusted.")
            W()
            lines(rep0, *[k for k, _ in rep0 if k.startswith("gate")])
            W()
            TI(g["gate"].groupby(
                ["lapse_participants", "lapse_controls",
                 "p_fast_given_lapse"])[
                    ["fires", "fires_catch", "fires_rt",
                     "catch_alone"]].mean().round(3))
            W()

        H("C1: THE GROUP COMPARISON")
        if "c1" not in g:
            W("  (RUN_C1 was off)")
            W()
        else:
            c1 = g["c1"]
            W("  Two one-sided Mann-Whitney tests on the per-person")
            W("  weights, H1 (fog) and H2 (colour), Holm-corrected")
            W("  across the pair.")
            W()
            W("  RANK-BASED, so no cap is needed. A separable")
            W("  participant's Firth weight is finite but means 'at")
            W("  least very large'; a mean-based test would compare a")
            W("  number the data does not pin down, while a rank does")
            W("  not care whether the top weight is 8 or 40. The")
            W("  strongest responders stay in.")
            W()
            W("  THE CONTROLS ARE NOT NON-RESPONDERS. A control can")
            W("  work out that the experiment is about edited images")
            W("  and pick the more edited one, which loads")
            W("  positively on BOTH features and eats the contrast C1")
            W("  rests on. The concept is not secret, so the strategy")
            W("  is simulated at the SAME RATE IN BOTH GROUPS; what")
            W("  appears only in the participants is the remembered")
            W("  experience.")
            W()
            W("  THE REGISTERED VERDICT. The gate is tested first.")
            W("  When it fires, C1 IS REPORTED AS INCONCLUSIVE --")
            W("  neither a positive nor a negative -- and the")
            W("  pre-registered secondary below is reported beside")
            W("  it with its realised sample sizes.")
            W()
            lines(rep0, *[k for k, _ in rep0 if k.startswith("C1")])
            W()
            null = c1[np.isclose(c1.effect_sd, 0.0)]
            W("  (a) FALSE-POSITIVE RATE, no effect in either group.")
            W("      Worst cell in each set; nominal "
              f"{ALPHA}.")
            W()
            TI(null.groupby(["lapse_participants", "lapse_controls"])
               .agg(ungated=("power_either_raw", "max"),
                    gated=("power_either_gated", "max"),
                    secondary=("zero_power_either", "max"),
                    gate_fires=("gate_fires", "min"))
               .round(3), indent="      ")
            W()
            W("      The ungated row for 'controls lapse more' is the")
            W("      hazard the gate exists for. 'gate_fires' is the")
            W("      LEAST often it fired in that set.")
            W()
            real = [x for x in (10, 15, 20) if x in C1_PARTICIPANT_N]
            pw_ = c1[(~np.isclose(c1.effect_sd, 0.0))
                     & (c1.n_participants.isin(real))
                     & (c1.n_controls == 30)
                     & (c1.lapse_participants == c1.lapse_controls)]
            W("  (b) POWER at the sizes this study will run:")
            W(f"      {'/'.join(str(x) for x in real)} participants "
              "against 30 controls, matched lapse")
            W("      rates, so the gate is quiet and the primary C1")
            W("      is the analysis.")
            W()
            W("      Every cell's scenario is named by its labels:")
            W("      the claim, the choice probability on a typical")
            W("      informative trial, the share of participants")
            W("      carrying the effect, the group size, the")
            W("      strategy rate and the lapse rate.")
            W()
            if len(pw_):
                TI(pw_.pivot_table(
                    index=["effect_on", "effect_p_typical",
                           "responder_share"],
                    columns=["n_participants", "strategy_rate",
                             "lapse_participants"],
                    values="power_either_gated").round(3),
                   indent="      ")
            W()
            W("  (c) THE SECONDARY: C1 on the people with NO catch")
            W("      failures, in both groups.")
            W()
            W("      Pre-registered rather than post-hoc matching:")
            W("      the rule is fixed before the data arrive, so the")
            W("      subsample cannot be chosen for preserving the")
            W("      result. It works only because lapse rates differ")
            W("      BETWEEN people -- a clean catch record is")
            W("      evidence of a low personal rate. If everyone in")
            W("      a group lapsed at the group mean it would select")
            W("      a random subset with the same mean and fix")
            W("      nothing.")
            W()
            W("      IT REDUCES THE ARTEFACT WITHOUT REMOVING IT.")
            W("      Selecting on a clean record leaves a residual")
            W("      lapse gap, and the power to detect that residual")
            W("      grows with the sample size and with the size of")
            W("      the weights. So its false-positive rate is at")
            W("      nominal at the sizes above and drifts up in the")
            W("      larger cells and at high strategy rates -- the")
            W("      table in (a) and the breakdown in the console")
            W("      output give both.")
            W()
            W("      THE RESPONSE-TIME RE-CHECK IS A DIAGNOSTIC, NOT")
            W("      A SECOND GATE. The subsample's own fast-response")
            W("      test fires in almost every study where the")
            W("      controls really lapse more -- with or without a")
            W("      real effect, since response times do not depend")
            W("      on the weights -- so gating on it would set the")
            W("      secondary's power to zero in every cell. That")
            W("      would amount to registering that C1 is")
            W("      unanswerable whenever the groups' attention")
            W("      differs at all. A fired re-check instead says")
            W("      the residual is present and the secondary's own")
            W("      result carries the inflation above.")
            W()
            W("      what survives, and what it is worth:")
            W()
            TI(null.groupby(["lapse_participants", "lapse_controls"])
               .agg(kept_participants=("zero_n_participants", "mean"),
                    kept_controls=("zero_n_controls", "mean"),
                    lapse_kept_p=("zero_lapse_participants", "mean"),
                    lapse_kept_c=("zero_lapse_controls", "mean"),
                    usable_worst=("zero_usable", "min"),
                    false_pos=("zero_power_either", "max"),
                    rt_recheck=("zero_rt_fires", "mean"))
               .round(3), indent="      ")
            W()
            W("      'usable_worst' is the share of studies keeping")
            W(f"      at least {C1_ZERO_MIN_N} people in BOTH groups. "
              "Below that a")
            W("      one-sided rank test cannot reach alpha whatever")
            W("      the data say, so those studies are reported as")
            W("      unusable rather than tested. When the")
            W("      PARTICIPANT group is itself small and lapsing,")
            W("      this is the binding constraint and the secondary")
            W("      is often simply unavailable.")
            W()
            sec_pw = c1[(~np.isclose(c1.effect_sd, 0.0))
                        & (c1.n_participants.isin(real))
                        & (c1.n_controls == 30)
                        & (c1.lapse_controls > c1.lapse_participants)]
            if len(sec_pw):
                W("      its power where the gate has fired "
                  "(controls lapse more),")
                W("      which is the only case it is read in:")
                W()
                TI(sec_pw.pivot_table(
                    index=["effect_on", "effect_p_typical",
                           "responder_share"],
                    columns=["n_participants", "lapse_controls"],
                    values="zero_power_either").round(3),
                   indent="      ")
                W()

        H("WHERE THE FROZEN DRAW SITS AMONG STEP 6'S DRAWS")
        W("  Every registered figure is a property of ONE draw of ONE")
        W("  version. This section says whether that draw was a")
        W("  typical one, which a reader cannot otherwise tell.")
        W()
        W("  Step 6's per-draw tables are outside the sealed package,")
        W("  so their numbers are quoted here -- this report is what")
        W("  survives.")
        W()
        W("  Three things have to line up, and the frozen draw is NOT")
        W("  found by its seed:")
        W("    the version    from FROZEN.json, not assumed. "
          "Step 6's")
        W("                   tables hold one row per draw PER")
        W("                   VERSION, so ranking without that filter")
        W("                   compares the frozen draw with other")
        W("                   versions' lists rather than with the")
        W("                   other draws of its own.")
        W("    the draw       by its SHIPPING flag. Step 5 builds a")
        W("                   shipping draw plus numbered replicates,")
        W("                   and step 6 records the replicates' seeds")
        W("                   from their folder names but not the")
        W("                   shipping draw's, which is step 5's")
        W("                   RANDOM_SEED. The frozen list is the")
        W("                   shipping draw, so its seed cell is")
        W(f"                   empty and FROZEN_SEED ({FROZEN_SEED}) "
          "is a")
        W("                   cross-check, not the key.")
        W("    the condition  the detection table holds several rows")
        W("                   per draw, one per simulated sensitivity")
        W("                   and participant type, so each quantity")
        W("                   is ranked within its own condition.")
        W()
        W("  Nothing is reported unless all three resolve. A missing")
        W("  percentile is a gap; a plausible one computed from the")
        W("  wrong rows is worse.")
        W()
        if seeds:
            for ln in seeds.get("notes", []):
                W("  " + ln)
            W()
            for what in ("information", "detection"):
                if what in seeds:
                    W(f"  {what}:")
                    T(seeds[what], indent="    ")
                    W()
        else:
            W("  (not computed -- see the lines above for what was")
            W("   looked for)")
            W()

        H("REPRODUCING THE PRE-FREEZE RUN")
        if not legacy:
            W("  (not run, or the legacy list is gone)")
            W()
        else:
            W("  ONE-DIRECTIONAL ON PURPOSE. The old figures came")
            W('  from the pre-freeze list AND from DETECTION_SIDED =')
            W('  "one", so reproducing them means running that')
            W("  configuration, not the registered one. A reader")
            W("  comparing the two threshold tables would otherwise")
            W("  conclude the thresholds had moved.")
            W()
            W("  A difference here is not automatically a bug: the")
            W("  list changed at the freeze, and the predictors")
            W("  changed units before it. The check exists so that")
            W("  any difference is NAMED rather than discovered")
            W("  later.")
            W()
            T(legacy["table"])
            W()

        H("THE FIGURES")
        figs = (reg or {}).get("figures") or []
        if not figs:
            W("  (none written -- MAKE_FIGURES was off, or the")
            W("   sections they draw from did not run)")
            W()
        else:
            W("  Four figures, each drawn from a table computed")
            W("  above and saved beside it as a CSV, so any point can")
            W("  be checked without re-running anything.")
            W()
            W("  Every series carries a colour AND a marker AND a")
            W("  line style. The palette is Okabe-Ito, validated")
            W("  rather than chosen by eye -- worst adjacent-pair")
            W("  separation 11.0 under deuteranopia -- but one of the")
            W("  four has a contrast ratio of 2.98:1 against white,")
            W("  under a 3:1 target, and a registration document gets")
            W("  printed and photocopied. Colour alone would not")
            W("  survive that.")
            W()
            W("  (a) existence_power.png")
            W("      The existence claim against sample size, by")
            W("      responder share, one panel per feature. Drawn at")
            W("      the most powerful rule the REGISTERED standard")
            W("      admits at each size, so it is what the")
            W("      registered analysis would actually achieve --")
            W("      not the best rule in the sweep.")
            W()
            W("      Two panels rather than one, because H1 and H2")
            W("      have different thresholds and a shared panel")
            W("      invites reading a gap between them that is")
            W("      arithmetic rather than asymmetry. Figure (c)")
            W("      makes that comparison properly.")
            W()
            W("  (b) c1_power.png")
            W("      C1 against sample size, at matched lapse")
            W("      rates (so the gate is quiet and the primary C1")
            W("      is what is drawn) against 30 controls. One row")
            W("      of panels per responder share, one line per")
            W("      strategy rate.")
            W()
            W("      Faceted rather than drawn as one legend because")
            W("      share x strategy is six series and only four")
            W("      colours are assigned -- a fifth is never")
            W("      generated. Faceting also puts the comparison the")
            W("      figure exists for, what the strategy costs,")
            W("      inside a panel rather than across a crowded key.")
            W()
            W("      The strategy rate is on the figure because it is")
            W("      the parameter that decides whether C1 is worth")
            W("      running. C1 is better powered than the existence")
            W("      claims ONLY at full responder share with no")
            W("      strategy; at half share with a quarter of both")
            W("      groups using the strategy it is underpowered at")
            W("      every realistic size. A curve without those two")
            W("      numbers on it would be misleading, which is why")
            W("      they are in the subtitle.")
            W()
            W("  (c) equal_behaviour.png")
            W("      H1 against H2 with the identity line, each point")
            W("      one scenario with both claims set to the same")
            W("      choice probability on a typical informative")
            W("      trial. Equivalence is the diagonal, so it can be")
            W("      seen rather than inferred from two curves, and")
            W("      the diverging cells are marked.")
            W()
            W("  (d) gate.png")
            W("      The attention gate, two panels, one axis each.")
            W("      Left is what the gate COSTS: how often it fires")
            W("      when the groups are matched, which makes C1")
            W("      inconclusive for nothing. Right is what it buys,")
            W("      with the catch measure drawn beside the combined")
            W("      gate -- because response times might turn out")
            W("      uninformative, and the catch measure is what the")
            W("      gate rests on if they do.")
            W()
            W("      Each panel shows the gate at its least")
            W("      flattering: the worst case over the")
            W("      response-time sweep on the left, the least often")
            W("      fired on the right.")
            W()
            for nm in figs:
                W(f"    written: {nm}")
            W()

        H("STOPPING RULE: A SINGLE FINAL ANALYSIS")
        W("  NOT SEQUENTIALLY MONITORED. The analysis is run ONCE,")
        W("  at the end of recruitment, which ends at a maximum N or")
        W("  an end date fixed in advance -- whichever comes first,")
        W("  both set before the first participant.")
        W()
        W("  The reason is the primary tests. The existence claims")
        W("  and C1 are frequentist: a per-person threshold")
        W("  calibrated to a fixed per-tail rate, a counting rule")
        W("  whose false-claim rate is computed against that rate,")
        W("  and rank tests at a fixed alpha. Every one of those")
        W("  rates is the rate of a SINGLE test. Looking repeatedly")
        W("  and stopping when the answer arrives inflates all of")
        W("  them, and the correction -- an alpha-spending function")
        W("  with the looks declared in advance -- would have to be")
        W("  registered too, and would cost power this design cannot")
        W("  spare at ten to twenty participants.")
        W()
        W("  The Bayes factors reported here are not affected in the")
        W("  same way, and an earlier version of this pipeline")
        W("  simulated sequential stopping on them. That is retired")
        W("  with the script that did it (see the header): a study")
        W("  whose primary tests cannot be monitored sequentially")
        W("  gains nothing registrable from a secondary measure that")
        W("  can.")
        W()
        W("  Nothing in this report assumes more than one look.")
        W()

        H("WHAT TO DO WITH THIS")
        W("  If the smallest detectable individual weight is larger")
        W("  than the effect you expect, the exploratory per-person")
        W("  analysis cannot work however clean the data. The fixes,")
        W("  in order of how far back they send you:")
        W("    more trials per type        -> step 5 only")
        W("    different type proportions  -> step 5 only")
        W("    larger differences          -> steps 1 and 3, re-render")
        W()
        W("  The group-level power is driven by the number of PEOPLE,")
        W("  not by the trial list, so it cannot be fixed in the")
        W("  pipeline at all.")
    return path


# ============================================================
# Main
# ============================================================

def check_settings(report):
    """Every setting that names a cell in another setting's grid.

    Run FIRST, before anything expensive, because the failures it
    catches otherwise surface deep inside a sweep as a KeyError with
    no indication of which setting is wrong. Each message names the
    setting to change and lists what it may be set to.
    """
    print("\n" + "=" * 60)
    print("CHECKING THE SETTINGS")
    print("=" * 60)
    bad = []

    def pick_in(name, value, grid_name, grid):
        if not any(np.isclose(float(value), float(gv)) for gv in grid):
            bad.append(
                f"{name} = {value!r} is not in {grid_name} = "
                f"{list(grid)}.\n    Set {name} to one of those, or "
                f"add {value!r} to {grid_name}.")

    w, sh, lam = RULE_SUMMARY_PICK
    pick_in("RULE_SUMMARY_PICK[0] (effect)", w,
            "RULE_RESPONDER_WEIGHTS", RULE_RESPONDER_WEIGHTS)
    pick_in("RULE_SUMMARY_PICK[1] (share)", sh,
            "RULE_RESPONDER_SHARES", RULE_RESPONDER_SHARES)
    pick_in("RULE_SUMMARY_PICK[2] (lapse)", lam,
            "LAPSE_RATES", LAPSE_RATES)
    for nm, v, gnm, grid in (
            ("CONTRAST_SUMMARY", CONTRAST_SUMMARY,
             "CONTRAST_BASELINE_GRID", CONTRAST_BASELINE_GRID),
            ("DOSE_SUMMARY", DOSE_SUMMARY,
             "DOSE_EFFECT_GRID", DOSE_EFFECT_GRID)):
        pick_in(nm, v, gnm, grid)

    for nm, sub_ in (("RULE_SUMMARY_SHARES", RULE_SUMMARY_SHARES),):
        missing = [v for v in sub_
                   if not any(np.isclose(v, g)
                              for g in RULE_RESPONDER_SHARES)]
        if missing:
            bad.append(
                f"{nm} contains {missing}, which is not in "
                f"RULE_RESPONDER_SHARES = {RULE_RESPONDER_SHARES}.")
    missing = [v for v in RULE_SUMMARY_LAPSES
               if not any(np.isclose(v, g) for g in LAPSE_RATES)]
    if missing:
        bad.append(
            f"RULE_SUMMARY_LAPSES contains {missing}, which is not in "
            f"LAPSE_RATES = {LAPSE_RATES}.")

    if not any(np.isclose(NONRESPONDER_SPREAD_MULT, t)
               for t in TASTE_SPREADS_IN_FLOORS):
        bad.append(
            f"NONRESPONDER_SPREAD_MULT = {NONRESPONDER_SPREAD_MULT} "
            f"is not in TASTE_SPREADS_IN_FLOORS =\n    "
            f"{TASTE_SPREADS_IN_FLOORS}, so the null it assumes is "
            "never simulated.")
    if not any(np.isclose(CONTROL_CHECK_BOUNDARY, t)
               for t in TASTE_SPREADS_IN_FLOORS):
        bad.append(
            f"CONTROL_CHECK_BOUNDARY = {CONTROL_CHECK_BOUNDARY} is "
            f"not in TASTE_SPREADS_IN_FLOORS =\n    "
            f"{TASTE_SPREADS_IN_FLOORS}. The procedure's false-claim "
            "rate peaks AT the\n    boundary, so that spread has to "
            "be among the evaluated ones.")

    # The admissibility standards are judged against spreads that the
    # taste nulls are actually simulated at.
    for std, spreads in ADMISSIBILITY_STANDARDS.items():
        miss = [s for s in spreads
                if not any(np.isclose(s, t)
                           for t in TASTE_SPREADS_IN_FLOORS)]
        if miss:
            bad.append(
                f'ADMISSIBILITY_STANDARDS["{std}"] names spreads '
                f"{miss} that are not simulated.\n    Add them to "
                f"TASTE_SPREADS_IN_FLOORS = {TASTE_SPREADS_IN_FLOORS}.")

    # C1's control sizes have to be reachable, and the subsample's
    # floor has to leave a test that can fire.
    if C1_ZERO_MIN_N < 4:
        bad.append(
            f"C1_ZERO_MIN_N = {C1_ZERO_MIN_N}. A one-sided rank test "
            "on fewer than 4\n    against 4 cannot reach "
            f"alpha = {ALPHA} whatever the data say.")
    biggest_group = max(max(C1_PARTICIPANT_N),
                        int(max(max(C1_CONTROL_RATIOS)
                                * max(C1_PARTICIPANT_N),
                                max(C1_CONTROL_ABSOLUTE))))
    if C1_POOL_PEOPLE < 20 * biggest_group:
        bad.append(
            f"C1_POOL_PEOPLE = {C1_POOL_PEOPLE:,} is small beside the "
            "largest group it is\n    drawn from "
            f"({int(max(C1_CONTROL_RATIOS) * max(C1_PARTICIPANT_N))}"
            "). Drawing groups from the pool only\n    stands in for "
            "fresh people while a study uses a small share of it.")

    # The two-sided thresholds have to leave room for both tails.
    if not 0 < FLOOR_ALPHA < 0.5:
        bad.append(f"FLOOR_ALPHA = {FLOOR_ALPHA} must be in (0, 0.5): "
                   "it is the rate PER TAIL.")
    if DETECTION_SIDED not in ("one", "two"):
        bad.append(f'DETECTION_SIDED = {DETECTION_SIDED!r} must be '
                   '"one" or "two".')
    # The rejected procedure must actually branch between two
    # DIFFERENT standards, and its default must be the less
    # constrained one -- otherwise the section costs a rule against
    # itself and reports a difference of zero as a finding.
    dft = ADMISSIBILITY_STANDARDS.get(PROCEDURE_DEFAULT_STANDARD)
    fbk = ADMISSIBILITY_STANDARDS.get(PROCEDURE_FALLBACK_STANDARD)
    if dft is None or fbk is None:
        bad.append(
            f"PROCEDURE_DEFAULT_STANDARD "
            f"({PROCEDURE_DEFAULT_STANDARD!r}) and "
            f"PROCEDURE_FALLBACK_STANDARD "
            f"({PROCEDURE_FALLBACK_STANDARD!r}) must both be keys of "
            f"ADMISSIBILITY_STANDARDS = "
            f"{sorted(ADMISSIBILITY_STANDARDS)}.")
    elif PROCEDURE_DEFAULT_STANDARD == PROCEDURE_FALLBACK_STANDARD:
        bad.append(
            f"PROCEDURE_DEFAULT_STANDARD and "
            f"PROCEDURE_FALLBACK_STANDARD are both "
            f"{PROCEDURE_DEFAULT_STANDARD!r}.\n    The procedure "
            "would branch between a rule and itself, and every "
            "variant would\n    report the same rate.")
    elif not set(dft) < set(fbk):
        bad.append(
            f"the procedure's default standard "
            f"({PROCEDURE_DEFAULT_STANDARD!r}, spreads {dft}) must be "
            f"LESS constrained than its\n    fallback "
            f"({PROCEDURE_FALLBACK_STANDARD!r}, spreads {fbk}). As "
            "set, the branch cannot make the\n    claim more "
            "conservative, which is the only thing it exists to do.")
    if GATE_ACTION != "inconclusive":
        bad.append(
            f"GATE_ACTION = {GATE_ACTION!r}. Only \"inconclusive\" "
            "is implemented, and it is\n    the registered choice; "
            "the joint lapse model cannot serve as a fallback\n    "
            "because it is not identifiable on this design.")
    if RT_GATE and not 0 < RT_FAST_GIVEN_LAPSE <= 1:
        bad.append(f"RT_FAST_GIVEN_LAPSE = {RT_FAST_GIVEN_LAPSE} must "
                   "be in (0, 1].")

    if bad:
        print(f"  {len(bad)} problem(s):")
        for b in bad:
            print(f"\n  - {b}")
        raise SystemExit(
            "\nThe settings above contradict each other. Nothing has "
            "been computed.\nFix them in the settings block at the "
            "top of this file and run again.")
    print("  the cross-referenced settings agree")
    print(f"  summary cell: effect {w:g} per SD, responder share "
          f"{sh:g}, lapse {lam:g}")
    report.append(("settings cross-check", "PASS"))


def _taste_rates_at(X, th, mults, strategy_rate, strategy_w, rng,
                    n_people=None):
    """Classification rates for a taste null at each spread.

    Returns {(mult, feature): (p_predicted, p_opposite)}. The spread
    scales with each feature's OWN threshold, so a multiplier means
    the same thing for colour and for fog.

    `strategy_rate` adds the shared-concept strategy -- an equal
    positive weight on both features -- to that share of the people,
    which is what breaks the symmetry the asymmetry test looks for.
    """
    n_people = N_TASTE_PEOPLE if n_people is None else n_people
    U = th["upper"]
    out = {}
    for mult in mults:
        W = rng.normal(0.0, 1.0, size=(n_people, 2)) * (U * mult)
        if strategy_rate > 0:
            use = rng.random(n_people) < strategy_rate
            W[use, 0] += strategy_w
            W[use, 1] += strategy_w
        pr, op = classify(fit_batch(X, simulate_many(X, W, rng)), th)
        for k, feat in enumerate(("colour", "fog")):
            out[(mult, feat)] = (float(pr[:, k].mean()),
                                 float(op[:, k].mean()))
    return out


def _best_count_rule(bt, standard, feature, n):
    """The most powerful admissible COUNT rule at (standard, n).

    Count only, because the sign test is not registered
    (REGISTER_SIGN_TEST). Returns (k, m) or None.
    """
    g = bt[(bt.standard == standard) & (bt.feature == feature)
           & (bt.n == n) & (bt.family.str.startswith("count"))]
    if not len(g):
        return None
    r = g.loc[g.power.idxmax()]
    return int(r.k), int(r.m)


def run_control_gated_rule(X, th, bt, report, root, rates=None):
    """THE REJECTED PROCEDURE'S FALSE-CLAIM RATE -- the evidence
    for registering the strict standard outright.

    NOT ON THE REGISTERED PATH. Kept in full because a registration
    that merely asserted the strict rule would not show why the
    cheaper option was refused, and because the control-based spread
    estimate it is built on is still reported descriptively.

    The existence claim is evaluated with the practical rule unless
    the control group says the taste spread is above
    CONTROL_CHECK_BOUNDARY, in which case the strict rule is used --
    and with the strict rule throughout if fewer than
    CONTROL_CHECK_MIN_CONTROLS controls are recruited.

    THE RULE IS THEREFORE CHOSEN FROM DATA, so neither branch's rate
    is the procedure's rate. The practical rule is admissible against
    0.25x and 0.5x by construction but reaches 0.363 against a 1x
    null; all of the protection against that comes from the check
    noticing, and the check is a noisy estimate.

    IT FACTORISES EXACTLY. The branch is decided by the CONTROL sample
    and the rule applied to the PARTICIPANT sample -- different
    people -- so the two are independent:

        combined = P(strict) x FP_strict + P(practical) x FP_practical

    P(strict) is simulated; both FP terms come from the same exact
    trinomial enumeration the rule sweep uses. So the only Monte Carlo
    error in the answer is in the branch probability, and at
    CONTROL_CHECK_REPS groups that is about 0.008 at worst.
    """
    print("\n" + "=" * 60)
    print("THE REJECTED PROCEDURE: PRACTICAL RULE + CONTROL CHECK")
    print("=" * 60)
    print("  NOT REGISTERED. The registered standard is "
          f"{REGISTERED_STANDARD.upper()}, chosen outright.")
    print("  This section is the evidence for that choice: it is what "
          "the cheaper option")
    print("  would have cost. The control group's taste-spread "
          "estimate is reported")
    print("  descriptively and no claim depends on it.")
    print(f"  default standard   {PROCEDURE_DEFAULT_STANDARD} "
          f"(the procedure's, NOT the registered one)")
    print(f"  fallback standard  {PROCEDURE_FALLBACK_STANDARD}, used "
          f"when the controls' estimated taste")
    print(f"                     spread exceeds "
          f"{CONTROL_CHECK_BOUNDARY}x that feature's threshold,")
    print(f"                     or when fewer than "
          f"{CONTROL_CHECK_MIN_CONTROLS} controls are recruited")
    print(f"  spread estimator   moment: "
          f"sqrt(Var(fitted) - Var(fitted | guessers))")
    print(f"  {CONTROL_CHECK_REPS} simulated control groups per cell")
    U = np.asarray(th["upper"])
    var_null = np.asarray(th["sd"]) ** 2
    rng = np.random.default_rng(RANDOM_SEED + 5151)

    # --- the null rates at every spread, including the boundary ---
    mults = sorted(set(CONTROL_CHECK_SPREADS))
    # ONE MEASUREMENT OF THE NULL, SHARED WITH THE RULE SWEEP.
    #
    # This section used to measure the taste nulls again. Two Monte
    # Carlo estimates of the same quantity disagree, and a rule the
    # sweep had certified admissible then scored above alpha here --
    # a discrepancy that was entirely an artefact of measuring twice.
    #
    # The sweep's rates are used for the NO-STRATEGY case, which is
    # where the registered standard's admissibility is judged. The
    # contaminated cases are still measured here, because the sweep
    # has no strategy-contaminated nulls to borrow.
    rate_by_strategy = {}
    shared = {}
    if rates is not None:
        for _, r in rates.iterrows():
            if r.null != "taste":
                continue
            m = float(r.spread_in_floors)
            if any(np.isclose(m, x) for x in mults):
                # SYMMETRISED, exactly as the rule sweep does it. A
                # zero-centred taste preference is symmetric by
                # construction, so the two measured counts estimate
                # one quantity; averaging halves the variance and
                # removes a sampling asymmetry the sign test
                # amplifies. Taking the raw pair here while the sweep
                # symmetrised was enough on its own to make the two
                # sections disagree by 0.008 on the same rule.
                p = 0.5 * (float(r.p_predicted) + float(r.p_opposite))
                shared[(m, r.feature)] = (p, p)
    # SHARED PER SPREAD, not all or nothing. An earlier version took
    # the sweep's rates only if it had EVERY spread this section
    # evaluates -- and it never does, because CONTROL_CHECK_SPREADS
    # carries a stress case beyond every standard's set that the
    # sweep has no reason to simulate. The condition failed and the
    # section silently re-measured all of them, which is the
    # behaviour the sharing was meant to remove.
    borrowed, measured = [], []
    for sr in CONTROL_CHECK_STRATEGY_RATES:
        if sr == 0.0 and shared:
            missing = [m for m in mults
                       if (m, "colour") not in shared
                       or (m, "fog") not in shared]
            got = dict(shared)
            if missing:
                extra = _taste_rates_at(
                    X, th, missing, sr, C1_STRATEGY_WEIGHT, rng)
                # Symmetrised too: these are no-strategy taste nulls,
                # the same symmetric object, just at a spread the
                # sweep does not carry.
                for key, (pp, po) in extra.items():
                    q = 0.5 * (pp + po)
                    got[key] = (q, q)
            rate_by_strategy[sr] = got
            borrowed = [m for m in mults if m not in missing]
            measured = missing
        else:
            rate_by_strategy[sr] = _taste_rates_at(
                X, th, mults, sr, C1_STRATEGY_WEIGHT, rng)
    note_lines = []
    if borrowed:
        note_lines.append(
            "no-strategy rates taken from the RULE SWEEP at spreads "
            + ", ".join(f"{m:g}x" for m in borrowed))
    if measured:
        note_lines.append(
            "measured here at " + ", ".join(f"{m:g}x" for m in measured)
            + " -- the sweep does not simulate "
            + ("it" if len(measured) == 1 else "them"))
    if not borrowed:
        note_lines.append(
            "NOTE: no spread could be shared with the sweep, so every "
            "rate here is an independent measurement and may differ "
            "from the sweep's by Monte Carlo error. Check that "
            "CONTROL_CHECK_SPREADS overlaps TASTE_SPREADS_IN_FLOORS.")
    note_lines.append(
        "no-strategy nulls are SYMMETRISED, as the rule sweep does: a "
        "zero-centred taste preference is symmetric by construction, "
        "so the two counts estimate one quantity.")
    note_lines.append(
        "strategy-contaminated rates are always measured here and are "
        "NOT symmetrised: the strategy loads positively on both "
        "features, so its asymmetry is the signal stage 1 detects.")
    in_sets = sorted({m for s in ADMISSIBILITY_STANDARDS.values()
                      for m in s})
    unshared = [m for m in in_sets if m in measured]
    note_lines.append(
        "every spread inside a standard's set is shared"
        if not unshared else
        "WARNING: " + ", ".join(f"{m:g}x" for m in unshared)
        + " is inside a standard's set but was NOT shared; "
          "admissibility and this section can disagree there")
    print()
    for ln in note_lines:
        print("  " + ln)
    control_rate_notes = note_lines
    print(f"\n  classification rates under each taste null "
          f"(no strategy):")
    print(f"    {'spread':>8}  {'colour pred/opp':>18}  "
          f"{'fog pred/opp':>18}")
    for m in mults:
        c = rate_by_strategy[0.0][(m, "colour")]
        f = rate_by_strategy[0.0][(m, "fog")]
        print(f"    {m:>7.2f}x  {c[0]:>8.4f} / {c[1]:<7.4f}  "
              f"{f[0]:>8.4f} / {f[1]:<7.4f}")

    # --- step 1 + 2 + 3: how often does the check choose strict? ---
    #
    # POOLED. A person's fitted weight and classification depend only
    # on their own type, and people are independent, so one pool per
    # (spread, strategy rate) is fitted once and groups are drawn from
    # it by index. Fitting every group afresh needs about 13 million
    # Firth fits and does not finish; this needs 300,000, once.
    pools = {}
    t_pool = time.time()
    for mult in CONTROL_CHECK_SPREADS:
        for sr in CONTROL_CHECK_STRATEGY_RATES:
            W = rng.normal(0.0, 1.0,
                           size=(CONTROL_POOL_PEOPLE, 2)) * (U * mult)
            if sr > 0:
                use = rng.random(CONTROL_POOL_PEOPLE) < sr
                W[use, 0] += C1_STRATEGY_WEIGHT
                W[use, 1] += C1_STRATEGY_WEIGHT
            est = fit_batch(X, simulate_many(X, W, rng))
            pr, op = classify(est, th)
            pools[(mult, sr)] = (est, pr, op)
    print(f"\n  {len(pools)} control-type pools of "
          f"{CONTROL_POOL_PEOPLE:,} fitted in "
          f"{time.time() - t_pool:.0f}s")

    rows = []
    for k_idx, feat in enumerate(("colour", "fog")):
        bound_abs = CONTROL_CHECK_BOUNDARY * U[k_idx]
        for mult in CONTROL_CHECK_SPREADS:
            s_true = U[k_idx] * mult
            for sr in CONTROL_CHECK_STRATEGY_RATES:
                est, pr, op = pools[(mult, sr)]
                for n_c in CONTROL_CHECK_N:
                    idx = rng.integers(0, CONTROL_POOL_PEOPLE,
                                       size=(CONTROL_CHECK_REPS, n_c))
                    e = est[idx, k_idx]
                    a_ = pr[idx, k_idx].sum(axis=1)
                    b_ = op[idx, k_idx].sum(axis=1)
                    # step 2: the moment estimate, per group
                    v = e.var(axis=1, ddof=1)
                    s_hat = np.sqrt(
                        np.maximum(v - var_null[k_idx], 0.0))
                    # step 1: the exact asymmetry test, per group.
                    # Under a symmetric taste null the predicted and
                    # opposite counts are exchangeable, so a binomial
                    # test at p = 0.5 on the detected people needs no
                    # distributional assumption. This is the sign
                    # test, used for the one job it is good at.
                    d = a_ + b_
                    p_asym = np.where(
                        d > 0,
                        stats.binom.sf(a_ - 1, np.maximum(d, 1), 0.5),
                        1.0)
                    asym = p_asym <= CONTROL_CHECK_ASYMMETRY_ALPHA
                    # Variant C's branch: the 95% UPPER bound of the
                    # estimate rather than the point estimate.
                    # Computed with no knowledge of the truth -- for a
                    # variance-difference estimator SE(s^2) is about
                    # (s^2 + var_null) * sqrt(2/(n-1)) under
                    # normality, and the delta method gives
                    # SE(s) = SE(s^2) / (2s). A bootstrap would be
                    # used on real data; the analytic form keeps the
                    # true spread out of the simulation.
                    se_v = ((s_hat ** 2 + var_null[k_idx])
                            * np.sqrt(2.0 / max(n_c - 1, 1)))
                    se_s = se_v / (2.0 * np.maximum(s_hat, 1e-6))
                    s_upper = s_hat + CONTROL_CHECK_UPPER_Z * se_s
                    # step 3/4: the branch
                    if n_c < CONTROL_CHECK_MIN_CONTROLS:
                        strict = np.ones(CONTROL_CHECK_REPS, bool)
                        strict_c = np.ones(CONTROL_CHECK_REPS, bool)
                        why = "too few controls"
                    else:
                        strict = s_hat > bound_abs
                        strict_c = s_upper > bound_abs
                        why = "estimate vs boundary"
                    rows.append({
                        "feature": feat, "true_spread": mult,
                        "strategy_rate": sr, "n_controls": n_c,
                        "branch_rule": why,
                        "p_strict": float(strict.mean()),
                        "p_strict_conservative": float(
                            strict_c.mean()),
                        "p_asymmetry_fires": float(asym.mean()),
                        "spread_hat_mean": round(
                            float(s_hat.mean()), 4),
                        "spread_hat_sd": round(
                            float(s_hat.std(ddof=1)), 4),
                        "spread_true_abs": round(float(s_true), 4),
                        "boundary_abs": round(float(bound_abs), 4)})
    br = pd.DataFrame(rows)
    br.to_csv(os.path.join(root, "control_check_branch.csv"),
              index=False)

    print("\n  (a) HOW OFTEN THE CHECK CHOOSES THE STRICT RULE")
    print("      (no strategy; the boundary is "
          f"{CONTROL_CHECK_BOUNDARY}x, so the truth at 0.75x is the "
          "coin-flip case)")
    g0 = br[br.strategy_rate == 0.0]
    print("  " + g0.pivot_table(index=["feature", "true_spread"],
                                columns="n_controls",
                                values="p_strict").round(3)
          .to_string().replace("\n", "\n  "))

    print("\n  (b) STRATEGY CONTAMINATION PUSHES TOWARD STRICT, "
          "as claimed")
    print("      P(strict) at 30 controls, by the share of controls "
          "using the strategy:")
    g30 = br[br.n_controls == 30]
    print("  " + g30.pivot_table(index=["feature", "true_spread"],
                                 columns="strategy_rate",
                                 values="p_strict").round(3)
          .to_string().replace("\n", "\n  "))
    print("\n      and the asymmetry test's own detection rate:")
    print("  " + g30.pivot_table(index=["feature", "true_spread"],
                                 columns="strategy_rate",
                                 values="p_asymmetry_fires").round(3)
          .to_string().replace("\n", "\n  "))
    mono = []
    for (feat, mult), g in g30.groupby(["feature", "true_spread"]):
        g = g.sort_values("strategy_rate")
        mono.append(bool((g.p_strict.diff().dropna() >= -0.01).all()))
    print(f"\n      P(strict) rises with the strategy rate in "
          f"{sum(mono)}/{len(mono)} cells -- so a")
    print("      strategy-contaminated control group selects the "
          "CONSERVATIVE rule.")
    report.append(("control check, null rates",
                   "; ".join(control_rate_notes[:2])))
    report.append(("control check, strategy pushes to strict",
                   f"{sum(mono)}/{len(mono)} cells monotone"))

    # --- step 4: the combined false-claim rate, exactly ---
    out = []
    for k_idx, feat in enumerate(("colour", "fog")):
        for mult in CONTROL_CHECK_SPREADS:
            for sr in CONTROL_CHECK_STRATEGY_RATES:
                p_pred, p_opp = rate_by_strategy[sr][(mult, feat)]
                for n_c in CONTROL_CHECK_N:
                    row = br[(br.feature == feat)
                             & (br.true_spread == mult)
                             & (br.strategy_rate == sr)
                             & (br.n_controls == n_c)].iloc[0]
                    ps = float(row.p_strict)
                    pc = float(row.p_strict_conservative)
                    for n in RULE_SUMMARY_N:
                        # The REJECTED procedure's two branches --
                        # not REGISTERED_STANDARD, which is what the
                        # claim is actually evaluated with and would
                        # make this a comparison of strict with
                        # itself.
                        rp = _best_count_rule(
                            bt, PROCEDURE_DEFAULT_STANDARD, feat, n)
                        rs = _best_count_rule(
                            bt, PROCEDURE_FALLBACK_STANDARD, feat, n)
                        rb = _best_count_rule(
                            bt, BOUNDARY_STANDARD, feat, n)
                        if rp is None or rs is None:
                            continue
                        fp_p = rule_false_claim_exact(
                            n, rp[0], rp[1], p_pred, p_opp)
                        fp_s = rule_false_claim_exact(
                            n, rs[0], rs[1], p_pred, p_opp)
                        fp_b = (rule_false_claim_exact(
                            n, rb[0], rb[1], p_pred, p_opp)
                            if rb else np.nan)
                        rec = {
                            "feature": feat, "true_spread": mult,
                            "strategy_rate": sr, "n_controls": n_c,
                            "n_participants": n,
                            "practical_rule": f"k>={rp[0]}, m<={rp[1]}",
                            "strict_rule": f"k>={rs[0]}, m<={rs[1]}",
                            "boundary_rule": (f"k>={rb[0]}, m<={rb[1]}"
                                              if rb else ""),
                            "p_strict": round(ps, 4),
                            "p_strict_conservative": round(pc, 4),
                            "fp_practical_alone": round(fp_p, 5),
                            "fp_strict_alone": round(fp_s, 5),
                            "fp_boundary_alone": (round(fp_b, 5)
                                                  if rb else np.nan),
                            # A: as decided
                            "fp_combined": round(
                                ps * fp_s + (1 - ps) * fp_p, 5),
                            # B: default rule must hold at the boundary
                            "fp_variant_B": (
                                round(ps * fp_s + (1 - ps) * fp_b, 5)
                                if rb else np.nan),
                            # C: conservative branch, same rules
                            "fp_variant_C": round(
                                pc * fp_s + (1 - pc) * fp_p, 5)}
                        out.append(rec)
    cg = pd.DataFrame(out)
    cg.to_csv(os.path.join(root, "control_gated_rule.csv"),
              index=False)

    print("\n" + "-" * 60)
    print("  (c) THE COMBINED FALSE-CLAIM RATE OF THE PROCEDURE")
    print("-" * 60)
    print("      At 30 controls, no strategy. Each cell is")
    print("      P(strict) x FP(strict) + P(practical) x "
          "FP(practical), exactly.")
    head = cg[(cg.n_controls == 30) & (cg.strategy_rate == 0.0)]
    print("  " + head.pivot_table(
        index=["feature", "true_spread"], columns="n_participants",
        values="fp_combined").round(4).to_string()
        .replace("\n", "\n  "))
    print(f"\n      worst cell at 30 controls: "
          f"{head.fp_combined.max():.4f} against alpha "
          f"{RULE_TARGET_FALSE}")
    bad = head[head.fp_combined > RULE_TARGET_FALSE]
    if len(bad):
        print(f"      {len(bad)} of {len(head)} cells exceed it:")
        show = bad.sort_values("fp_combined", ascending=False).head(8)
        print("  " + show[["feature", "true_spread",
                           "n_participants", "p_strict",
                           "fp_practical_alone", "fp_strict_alone",
                           "fp_combined"]].to_string(index=False)
              .replace("\n", "\n  "))
        print("\n      THE MECHANISM, and it is not the rules: both")
        print("      fp_*_alone columns are admissible wherever their")
        print("      own standard requires it. The excess is the "
              "BRANCH -- when the true")
        print("      spread is above the boundary and the estimate "
              "lands below it, the")
        print("      practical rule is applied to a population it was "
              "never admissible")
        print("      against. P(strict) in that row is how often the "
              "check did its job.")
    else:
        print("      every cell is at or below it")

    print("\n      the same, by control group size (worst cell over "
          "N and spread):")
    print("  " + cg[cg.strategy_rate == 0.0]
          .pivot_table(index="n_controls", columns="feature",
                       values="fp_combined", aggfunc="max")
          .round(4).to_string().replace("\n", "\n  "))
    print(f"\n      Below {CONTROL_CHECK_MIN_CONTROLS} controls the "
          "strict rule is used throughout, so those")
    print("      rows are the strict rule's own rate and are safe by "
          "construction --")
    print("      the cost is power, which the rule table shows.")

    print("\n" + "-" * 60)
    print("  (d) THREE VARIANTS OF THE PROCEDURE, COSTED")
    print("-" * 60)
    print("      A  as decided            practical by default, "
          "strict when the point")
    print("                               estimate exceeds the "
          "boundary")
    print("      B  admissible at the     the same branch, but the "
          "default rule must")
    print("         boundary              also hold at "
          f"{CONTROL_CHECK_BOUNDARY}x")
    print("      C  conservative branch   the same rules, but the "
          "branch uses the 95%")
    print("                               UPPER bound of the spread "
          "estimate")
    print()
    no_s = cg[cg.strategy_rate == 0.0]
    comp = no_s.groupby("n_controls").agg(
        A_worst=("fp_combined", "max"),
        B_worst=("fp_variant_B", "max"),
        C_worst=("fp_variant_C", "max"))
    print("      worst combined false-claim rate over every feature, "
          "spread and N:")
    print("  " + comp.round(4).to_string().replace("\n", "\n  "))
    at30 = no_s[no_s.n_controls == CONTROL_CHECK_MIN_CONTROLS]
    print(f"\n      at {CONTROL_CHECK_MIN_CONTROLS} controls, by "
          "true spread:")
    print("  " + at30.groupby("true_spread")[
        ["fp_combined", "fp_variant_B", "fp_variant_C"]].max()
        .round(4).to_string().replace("\n", "\n  "))
    print(f"\n      (alpha {RULE_TARGET_FALSE}; each column is the "
          "worst cell in that row)")

    # What B costs in power, from the rule table itself.
    pw_, psh_, plam_ = RULE_SUMMARY_PICK
    pcol_ = f"power_w{pw_:g}_s{psh_:g}_lapse{plam_:g}"
    if pcol_ in bt.columns or "power" in bt.columns:
        pc_ = "power" if "power" in bt.columns else pcol_
        cost = bt[bt.family.str.startswith("count")].pivot_table(
            index=["feature", "n"], columns="standard", values=pc_,
            aggfunc="max")
        keep = [c for c in (REGISTERED_STANDARD, BOUNDARY_STANDARD,
                            FALLBACK_STANDARD) if c in cost.columns]
        if len(keep) > 1:
            print("\n      WHAT B COSTS: power of the most powerful "
                  "admissible count rule")
            print("      at the summary cell, by standard:")
            print("  " + cost[keep].round(3).to_string()
                  .replace("\n", "\n  "))
            # VARIANT A's DEFAULT BRANCH IS THE PRACTICAL RULE, not
            # the registered standard. Reading this off
            # REGISTERED_STANDARD compared strict with boundary --
            # the same overloaded-constant fault the rule selection
            # had, surviving in the power line -- and printed the
            # cost with its sign inverted, because boundary is MORE
            # constrained than practical and so gives power up rather
            # than gaining it.
            d_ = (cost[PROCEDURE_DEFAULT_STANDARD]
                  - cost[BOUNDARY_STANDARD]).dropna()
            if len(d_):
                print(f"\n      B gives up "
                      f"{d_.min():+.3f} to {d_.max():+.3f} of power "
                      "against A (median "
                      f"{d_.median():+.3f}); positive is a cost.")
                report.append(
                    ("variant B power cost against A",
                     f"{d_.min():+.3f} to {d_.max():+.3f} at the "
                     "summary cell"))
    print("\n      THE CHOICE IS THE DESIGN SESSION'S. A is what was "
          "decided and is")
    print("      reported as such; B and C are here because A's rate "
          "exceeds alpha at")
    print("      the boundary and the report should not present that "
          "without the two")
    print("      ways out beside it.")
    for nm, col in (("A (as decided)", "fp_combined"),
                    ("B (admissible at the boundary)", "fp_variant_B"),
                    ("C (conservative branch)", "fp_variant_C")):
        v = no_s[col].max()
        report.append((f"procedure variant {nm}, worst rate",
                       f"{v:.4f}" if np.isfinite(v) else "n/a"))

    report.append(
        ("rejected procedure false-claim rate, 30 controls",
         f"{head.fp_combined.max():.4f} worst cell (alpha "
         f"{RULE_TARGET_FALSE}); "
         f"{len(bad)}/{len(head)} cells above it"))
    report.append(
        ("rejected procedure, worst over all control sizes",
         f"{cg[cg.strategy_rate == 0.0].fp_combined.max():.4f}"))
    return cg, br


def run_registered(X, th, det, report, root):
    """Every section the registration rests on, in order.

    Run ONCE, on the frozen list. These are not properties of a
    candidate design being chosen between -- that was step 6 -- they
    are the registered claims about the list that is now sealed.

    The order is not arbitrary. The classification rates come first
    because every rule, every behavioural statement and every
    prevalence figure is computed from them; the rule sweep then picks
    the decision rule; the behavioural table restates it in the form
    the claims are actually made in; the taste recovery asks whether
    the null those rules are judged against can be estimated at all;
    prevalence re-expresses the same detections as a population
    proportion; the lapse recovery is the negative result that decides
    how C1 handles inattention; and C1 comes last because it uses the
    thresholds, the rates and that decision.
    """
    out = {}
    off = [nm for nm, flag in
           (("rule sweep", RUN_RULE_SWEEP),
            ("control check", CONTROL_CHECK),
            ("equal behaviour", RUN_EQUAL_BEHAVIOUR),
            ("taste recovery", RUN_TASTE_RECOVERY),
            ("prevalence", RUN_PREVALENCE),
            ("lapse recovery", RUN_LAPSE_RECOVERY),
            ("gate", RUN_GATE),
            ("C1", RUN_C1)) if not flag]
    if off:
        print("\n  !! SECTIONS SWITCHED OFF: " + ", ".join(off))
        print("     This run is NOT the registered one.")
    report.append(("registered sections off",
                   ", ".join(off) if off else "none"))

    out["rates"] = classification_rates(X, th, report)
    if RUN_RULE_SWEEP:
        out["rules"], out["rules_best"] = run_rule_sweep(
            X, th, out["rates"], report, root)
        out["rule_summary"] = rule_summary_table(
            out["rules"], X, report, root)
    if RUN_RULE_SWEEP and CONTROL_CHECK:
        out["control_gated"], out["control_branch"] = \
            run_control_gated_rule(X, th, out["rules_best"], report,
                                   root, rates=out["rates"])
    if RUN_EQUAL_BEHAVIOUR:
        out["equal"] = equal_behaviour_table(
            X, th, out["rates"], report, root)
    if RUN_TASTE_RECOVERY:
        out["taste"] = recover_taste_spread(X, th, report, root)
    if RUN_PREVALENCE:
        # run_prevalence runs the estimated-alpha variant itself
        # and returns it. Calling it again here computed it twice and
        # put its line in the summary twice.
        out["prevalence"], out["prevalence_estimated"] = \
            run_prevalence(det, out["rates"], report, root)
    if RUN_LAPSE_RECOVERY:
        out["lapse"], out["lapse_usable"] = run_lapse_recovery(
            X, report, root)
    if RUN_GATE:
        out["gate"] = run_gate_characteristics(
            X.shape[0], report, root)
    if RUN_C1:
        out["c1"] = run_c1(X, th, out["rates"], report, root)

    # --- the figures, last: each one draws a table computed above ---
    if MAKE_FIGURES:
        print("\n" + "=" * 60)
        print("FIGURES")
        print("=" * 60)
        made = []
        if RUN_RULE_SWEEP:
            if plot_existence_power(out["rules"], X, root,
                                    report) is not None:
                made.append("existence_power.png")
        if RUN_C1 and plot_c1_power(out.get("c1"), root,
                                    report) is not None:
            made.append("c1_power.png")
        if RUN_EQUAL_BEHAVIOUR and plot_equal_behaviour(
                out.get("equal"), root, report) is not None:
            made.append("equal_behaviour.png")
        if RUN_GATE and plot_gate(out.get("gate"), root,
                                  report) is not None:
            made.append("gate.png")
        for nm in made:
            print(f"  {nm}")
        report.append(("figures written", ", ".join(made) or "none"))
        out["figures"] = made
    return out


def run_legacy_check(root, report):
    """Does this script still reproduce the pre-freeze run?

    ONE-DIRECTIONAL ON PURPOSE. The old figures came from the
    pre-freeze list AND from DETECTION_SIDED = "one"; reproducing them
    means running that configuration, not the registered one. The
    setting is forced here and restored afterwards, and the report says
    so, because a reader comparing the two tables would otherwise
    conclude the thresholds had moved.

    A difference is not automatically a bug: the list changed at the
    freeze and the predictors changed units before it. The check exists
    so that any difference is NAMED rather than discovered later.
    """
    global DETECTION_SIDED
    if not LEGACY_LIST or not os.path.isfile(LEGACY_LIST):
        print(f"\n  legacy list {LEGACY_LIST} not found; the "
              "reproduction check is skipped")
        report.append(("legacy reproduction", "skipped (no list)"))
        return None
    print("\n" + "=" * 60)
    print("REPRODUCTION CHECK AGAINST THE PRE-FREEZE RUN")
    print("=" * 60)
    print(f"  list    : {LEGACY_LIST}")
    against = LEGACY_OUTPUT or "(nothing -- own numbers only)"
    print(f"  compare : {against}")
    print('  forcing DETECTION_SIDED = "one" for this section, which '
          'is what the')
    print("  old figures were computed with. The registered sections "
          "above used")
    print(f'  DETECTION_SIDED = "{DETECTION_SIDED}".')
    keep = DETECTION_SIDED
    DETECTION_SIDED = "one"
    try:
        out_dir = os.path.join(root, "legacy")
        os.makedirs(out_dir, exist_ok=True)
        rep = []
        trials, X, r = load_trials(LEGACY_LIST, "legacy")
        th = compute_floors(X, "legacy", rep)
        det, needed = run_detection(X, th, "legacy", out_dir, rep)
        got = {"analysed trials": len(trials),
               "r(predictors)": round(r, 4),
               "threshold colour": round(float(th["upper"][0]), 4),
               "threshold fog": round(float(th["upper"][1]), 4)}
        print("\n  this run, on the legacy list:")
        for k, v in got.items():
            print(f"    {k:22s} {v}")
        rows = [{"quantity": k, "this_run": v} for k, v in got.items()]
        cmp_path = (os.path.join(LEGACY_OUTPUT, "comparison.csv")
                    if LEGACY_OUTPUT else None)
        if cmp_path and os.path.isfile(cmp_path):
            print(f"\n  {cmp_path} found; its contents are copied "
                  "into the report")
            print("  beside these numbers rather than compared "
                  "automatically -- the old")
            print("  folder's column names are not guaranteed to "
                  "match, and a silent")
            print("  name mismatch would report a pass it had not "
                  "earned.")
            old = pd.read_csv(cmp_path)
            old.to_csv(os.path.join(out_dir, "legacy_reference.csv"),
                       index=False)
            report.append(("legacy reproduction",
                           f"ran; reference copied from {cmp_path}"))
        else:
            if LEGACY_OUTPUT:
                print(f"\n  {LEGACY_OUTPUT} has no comparison.csv; "
                      "reporting this run's own")
                print("  numbers without claiming to reproduce "
                      "anything.")
            report.append(("legacy reproduction",
                           "ran; no reference to compare against"))
        tab = pd.DataFrame(rows)
        tab.to_csv(os.path.join(out_dir, "legacy_check.csv"),
                   index=False)
        return {"table": tab, "report": rep, "det": det,
                "thresholds": th}
    finally:
        DETECTION_SIDED = keep


def find_by_seed(path, what):
    """The per-draw table, or the places it was looked for.

    Step 6 writes to a NUMBERED folder when its output root already
    exists, so the run that produced the registered figures may sit in
    choose_colour_v4_2 rather than choose_colour_v4. A missing file is
    therefore usually a stale setting, not a missing run, and saying
    which siblings were tried turns a dead end into a one-line fix.
    """
    if os.path.isfile(path):
        return path, []
    base = os.path.basename(path)
    parent = os.path.dirname(os.path.dirname(path)) or "."
    stem = os.path.basename(os.path.dirname(path)).split("_")[0]
    tried = [path]
    found = []
    if os.path.isdir(parent):
        for nm in sorted(os.listdir(parent)):
            if not nm.startswith(stem) or nm == os.path.basename(
                    os.path.dirname(path)):
                continue
            cand = os.path.join(parent, nm, base)
            tried.append(cand)
            if os.path.isfile(cand):
                found.append(cand)
    return (found[-1] if found else None), tried


def _frozen_draw_mask(df):
    """Which rows are the frozen draw's, and how we know.

    THE FROZEN LIST IS STEP 6'S SHIPPING DRAW -- step 7a froze the
    unnumbered folder. Step 6 marks that draw with `shipping = True`
    and gives it draw index 0; it does NOT record its seed, because
    the shipping draw's seed is step 5's RANDOM_SEED rather than
    something parsed out of a folder name. So `seed == FROZEN_SEED`
    cannot find it, and FROZEN_SEED is a cross-check instead of the
    key.

    Returns (mask, how, note) where `note` is a warning if the row's
    own seed contradicts FROZEN_SEED.
    """
    if "shipping" in df.columns:
        mask = df.shipping.astype(bool)
        how = "shipping == True"
    elif "draw" in df.columns:
        mask = df.draw == 0
        how = "draw == 0 (the shipping draw)"
    else:
        return None, None, ("neither a 'shipping' nor a 'draw' column "
                            "-- cannot identify the frozen draw")
    note = None
    if mask.any() and "seed" in df.columns:
        seeds = df.loc[mask, "seed"]
        known = seeds.dropna()
        if len(known) and not (known == FROZEN_SEED).all():
            note = (f"the shipping row's seed is "
                    f"{sorted(set(known))} but FROZEN_SEED is "
                    f"{FROZEN_SEED} -- one of them is stale")
    return mask, how, note


def seed_position(report, frozen_version=None):
    """Where does the frozen draw sit among step 6's draws?

    Every registered figure is a property of ONE draw of ONE version.
    A reader has to be able to see whether that draw was a typical one
    or a lucky one. Step 6's per-draw tables are outside the sealed
    package, so their numbers are quoted into this report -- which is
    what survives -- rather than relied on later.

    THREE THINGS HAVE TO LINE UP, and each was wrong in turn:

      the version   from FROZEN.json, not assumed. Step 6's tables
                    hold one row per draw PER VERSION, so ranking
                    without this filter compares the frozen draw with
                    other versions' lists.
      the draw      by `shipping` / `draw == 0`, not by seed. Step 6
                    does not record the shipping draw's seed, so
                    FROZEN_SEED is a cross-check, not the key.
      the condition for the detection table, which holds several rows
                    per draw, each quantity is ranked within its own
                    (sensitivity, participant) condition.

    Nothing is reported unless all three resolve. A missing percentile
    is a gap; a plausible one computed from the wrong rows is worse.
    """
    print("\n" + "=" * 60)
    print("WHERE THE FROZEN DRAW SITS AMONG STEP 6'S DRAWS")
    print("=" * 60)
    notes = []

    def say(line):
        print("  " + line)
        notes.append(line)

    if not frozen_version:
        say("FROZEN.json did not say which version was frozen, so the "
            "frozen draw")
        say("cannot be identified. Skipped rather than guessed.")
        report.append(("frozen draw rank",
                       "not computed (no version in FROZEN.json)"))
        return {"notes": notes}
    say(f"frozen version: {frozen_version!r} (from FROZEN.json)")
    say("the frozen list is step 6's SHIPPING draw (step 7a froze the "
        "unnumbered")
    say(f"folder), so it is located by its shipping flag, not by "
        f"seed {FROZEN_SEED}")
    out = {}
    for what, want_path in (("information", BY_SEED_INFORMATION),
                            ("detection", BY_SEED_DETECTION)):
        path, tried = find_by_seed(want_path, what)
        if path is None:
            say(f"{what}: NOT FOUND. Tried: " + "; ".join(tried))
            say("  Point BY_SEED_INFORMATION / BY_SEED_DETECTION at "
                "step 6's output folder")
            say("  for the run that produced the registered figures.")
            report.append((f"frozen draw rank ({what})",
                           "not available (table not found)"))
            continue
        if path != want_path:
            say(f"{what}: {want_path} not found; using {path}")
            say("  (a sibling of the configured folder -- check this "
                "is the run you mean)")
        df = pd.read_csv(path)
        if "list" not in df.columns:
            say(f"{what}: {path} has no 'list' column; skipped")
            report.append((f"frozen draw rank ({what})",
                           "not available (no 'list' column)"))
            continue
        ver = df[df.list.astype(str) == str(frozen_version)]
        if not len(ver):
            say(f"{what}: no rows for version {frozen_version!r} "
                f"(has: "
                f"{', '.join(sorted(set(df.list.astype(str))))})")
            report.append((f"frozen draw rank ({what})",
                           f"not available (no {frozen_version!r} "
                           "rows)"))
            continue
        mask, how, note = _frozen_draw_mask(ver)
        if mask is None or not mask.any():
            say(f"{what}: {note or 'no shipping draw in this version'}")
            report.append((f"frozen draw rank ({what})",
                           "not available (frozen draw not "
                           "identifiable)"))
            continue
        if note:
            say(f"{what}: WARNING -- {note}")
        keys = [k for k in BY_SEED_GROUP_KEYS if k in ver.columns]
        n_draws = (ver.draw.nunique() if "draw" in ver.columns
                   else len(ver))
        say(f"{what}: {n_draws} draws of {frozen_version!r}, frozen "
            f"row by {how}"
            + (f", ranked within {' x '.join(keys)}" if keys else ""))
        num = [c for c in ver.columns
               if c not in BY_SEED_SKIP
               and c not in keys
               and pd.api.types.is_numeric_dtype(ver[c])]
        recs = []
        groups = ver.groupby(keys, sort=False) if keys \
            else [((), ver)]
        for gk, g in groups:
            gm, _, _ = _frozen_draw_mask(g)
            if gm is None or not gm.any():
                continue
            row = g[gm].iloc[0]
            for c in num:
                vals = g[c].to_numpy(float)
                vals = vals[np.isfinite(vals)]
                v = float(row[c])
                if len(vals) < 3 or np.allclose(vals, vals[0]) \
                        or not np.isfinite(v):
                    continue
                rec = {"quantity": c, "frozen_value": round(v, 4),
                       "mean_of_draws": round(float(vals.mean()), 4),
                       "min": round(float(vals.min()), 4),
                       "max": round(float(vals.max()), 4),
                       "percentile": round(
                           float((vals <= v).mean()), 3)}
                if keys:
                    gv = gk if isinstance(gk, tuple) else (gk,)
                    rec = {**dict(zip(keys, gv)), **rec}
                recs.append(rec)
        if not recs:
            say("  no varying numeric columns to rank")
            report.append((f"frozen draw rank ({what})",
                           "not available (nothing varies)"))
            continue
        t = pd.DataFrame(recs)
        print()
        print("  " + t.to_string(index=False).replace("\n", "\n  "))
        out[what] = t
        mid = t[(t.percentile >= 0.25) & (t.percentile <= 0.75)]
        extreme = t[(t.percentile <= 0.05) | (t.percentile >= 0.95)]
        print()
        say(f"  {len(mid)} of {len(t)} ranked quantities put the "
            f"frozen draw in the middle")
        say(f"  half of {frozen_version!r}'s draws; {len(extreme)} in "
            "the outer 10%.")
        if len(extreme):
            worst = extreme.loc[
                (extreme.percentile - 0.5).abs().idxmax()]
            say(f"  furthest out: {worst.quantity} at the "
                f"{worst.percentile:.0%} percentile "
                f"({worst.frozen_value} against a draw mean of "
                f"{worst.mean_of_draws})")
        say("  A draw at an extreme is not disqualifying -- it was "
            "chosen before any")
        say("  of this was computed -- but it has to be visible.")
        report.append(
            (f"frozen draw rank ({what})",
             f"{len(mid)}/{len(t)} in the middle half of "
             f"{frozen_version}'s draws, {len(extreme)} in the outer "
             f"10%; percentiles {t.percentile.min():.2f}-"
             f"{t.percentile.max():.2f}"))
    out["notes"] = notes
    return out


def main():
    t0 = time.time()
    print("=" * 60)
    print("SIMULATE POWER  (v4, step 8)")
    print("=" * 60)
    root = resolve_output_root()
    os.makedirs(root, exist_ok=True)
    shared_report = []

    # 0. The settings, before anything is computed from them.
    check_settings(shared_report)

    # 1. The package, before anything reads from it.
    if VERIFY_CHECKSUMS:
        facts = verify_frozen(FROZEN_ROOT, shared_report)
    else:
        print("\n  !! VERIFY_CHECKSUMS = False: the sealed package "
              "is being read without")
        print("     checking it. Nothing from this run is "
              "registrable.")
        shared_report.append(("frozen package verified",
                              "SKIPPED -- not registrable"))
        facts = {}

    frozen = os.path.join(FROZEN_ROOT, FROZEN_TRIAL_LIST)
    print(f"\n  list   : {frozen}")
    print(f"  output : {root}")

    # 2. The frozen list, and everything that is a property of it.
    results = [analyse_one("frozen", frozen, None, root)]
    r0 = results[0]
    reg = run_registered(r0["X"], r0["thresholds"], r0["det"],
                         r0["report"], r0["out_dir"])

    # 3. The sections that do not depend on the trial list.
    check_bayes_factor(shared_report)
    pw = run_power(root, shared_report)
    bfs = run_bayes(root, shared_report)
    seeds = seed_position(shared_report, facts.get("version"))

    # 4. The pre-freeze run, reproduced.
    legacy = run_legacy_check(root, shared_report) \
        if RUN_LEGACY_CHECK else None

    comp = build_comparison(results, root)
    path = write_report(results, comp, pw, bfs, shared_report, root,
                        t0, reg=reg, seeds=seeds, legacy=legacy)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['label']:10s} min detectable weight "
              f"{r['mde']:.2f}   classification {r['acc']:.1%}   "
              f"dose rise {r['mean_rise']:+.3f}")
    for k, v in r0["report"] + shared_report:
        print(f"  {k:38s} {v}")
    print(f"\n  Report : {path}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
