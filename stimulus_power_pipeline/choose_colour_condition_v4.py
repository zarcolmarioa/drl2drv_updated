"""
choose_colour_condition_v4.py  (stimulus pipeline, v4 axes, step 6)

Compares the candidate trial lists on ONE question: how often is a
participant with a FIXED sensitivity caught?

Where this sits
---------------
  step 3  build_stimulus_set_v4_corrected  the stimuli + manifest
  step 4  analyse_pairs_v4_corrected       the pair inventory
  step 5  build_trials_v4_corrected        the candidate trial lists
  step 6  this one                         which candidate to ship
  step 7  freeze the list, build the grid blocks
  step 8  simulate_power_v4                is the frozen list powered?

THIS SCRIPT CHOOSES NOTHING. It measures and stops. The version is
chosen by a person, from the table it prints.

WHY IT IS SEPARATE FROM STEP 8
------------------------------
Step 8 asks "is the chosen list powered?" -- a question about studies,
sample sizes and Bayes factors. This asks "which list detects a given
person best?", which needs only the PER-PERSON part of that machinery:
fit one participant, compare the fitted weight against a floor. Those
functions are copied here verbatim (see the note above the model
section), so this script does not wait on step 8's changes and step 8
does not wait on this.

EVERY VERSION IS MEASURED OVER MANY DRAWS
-----------------------------------------
Step 5 builds each version at N_SEEDS seeds: one shipping draw and the
rest replicates. This script measures EVERY draw and reports each
version's mean with the spread between its own draws beside it.

That spread is the yardstick. A trial list is 180 trials sampled from a
pool of thousands, so two versions can differ because one is a better
design or because one got a luckier sample, and a single draw cannot
tell those apart. On the first run of this script -- one draw per
version -- `current` and `both` differed by 2.3 points on the fog
control, against a Monte Carlo error of 1.1: apparently real, and
entirely a different draw of fog trials.

Nothing here picks a seed. The shipping draw is step 5's RANDOM_SEED,
fixed in advance, and it is reported as one draw among the rest.

WHAT PAIRS WITH WHAT, AND WHY THE SEEDING IS BY DRAW INDEX
----------------------------------------------------------
The simulated people are seeded from the DRAW INDEX, never from the
version. Two consequences, both wanted:

  Within one index, every version faces the SAME simulated people --
  the same true weights and the same random numbers. A difference
  between versions at that index is then the trial lists, with the
  participant noise common to both and largely cancelling.

  Across indices, the people are independent. So averaging over draws
  averages over independent participant noise as well as over designs,
  and the spread between draws carries both -- which is what the
  difference columns use for their uncertainty.

Differences from the baseline are therefore PAIRED by draw index: the
difference is computed at each index and then averaged, and its
standard error is the spread of those paired differences. That is a
stronger comparison than two independent means, and it needs no
assumption about the Monte Carlo error being the only noise.

THE TWO RULES THAT MAKE THE COMPARISON FAIR
-------------------------------------------
1. SENSITIVITY IS FIXED IN UNITS THAT DO NOT DEPEND ON THE TRIAL
   LIST. If sensitivity were fixed in units each list defines for
   itself, a list would be rewarded for having a small denominator
   rather than for detecting better.

   The fit uses `pred_sat` and `pred_fog` -- the registered model's
   predictors, in CORPUS-SD UNITS -- so the grid is in per-SD terms
   throughout and nothing is converted. The corpus SDs are two fixed
   constants from step 1, identical for every list, so the rule holds.

   A NOTE ON THIS RULE, because it was written when it mattered more,
   and on why the units changed. Step 4 used to normalise by each
   SCENE's own feature range, which varies by scene and by ladder;
   fixing sensitivity in those units would genuinely have confounded
   the comparison, and this script therefore worked in RAW feature
   units to stay clear of it. Step 4 now normalises by the corpus SD,
   so raw and per-SD differ by a fixed scale factor and the raw
   detour buys nothing -- while costing something, since the registered
   model and step 8 fit the SD columns and this script fitting raw
   ones left two scales in play for no reason. Firth is equivariant
   under a linear change of predictor scale, verified in CHECK 1e, so
   the switch changes no estimate.

2. EACH DRAW'S FLOOR COMES FROM RANDOM CHOOSERS ON THAT DRAW. The
   floor is the weight that FLOOR_ALPHA of pure guessers exceed by
   luck, and it depends on the trial list's own geometry. Using one
   list's floor on another would hand the comparison to whichever list
   happened to have the looser threshold. Floors are computed per DRAW,
   not per version, for the same reason.

THE CHECK THIS SCRIPT RUNS BEFORE IT MEASURES ANYTHING
------------------------------------------------------
CHECK 4 verifies the property the whole comparison rests on: two
versions built from the same colour ladder must have IDENTICAL
fog_only, together and conflict trials, because the colour cut filters
colour-only pairs and nothing else.

That was once false. Step 5 drew every trial type off one random
stream, so the colour cut -- by changing how many numbers the colour
draw consumed -- moved the fog, together and conflict draws too.
`current` and `larger_diffs`, which differ only in the cut, shared 1 of
60 fog_only trials, 1 of 30 together and 0 of 30 conflict. The fog
condition was not a control, and colour information leaked in from
types the cut never touches. Step 5 now gives each draw its own
stream; this check confirms it in the files themselves rather than
trusting that it was done.

It is checked HERE as well as in step 5 because step 6 is where the
numbers are read, and a measurement is only as good as the invariant
it assumes.

WHAT IT REPORTS
---------------
  1. INFORMATION AND FLOORS per version, averaged over draws, with the
     SHIPPED list included so a change in the floor can be traced to
     the thing that caused it.
  2. DETECTION at each sensitivity, for three participant types and
     both features, as a mean over draws with the between-draw spread.
  3. FALSE DETECTIONS, from an INDEPENDENT set of random choosers --
     not the ones that set the floor, which would be circular.
  4. DIFFERENCES FROM THE BASELINE, paired by draw index, each with the
     standard error of the paired difference.
  4b. THE FOG CONTROL, pulled out on its own: within a ladder the fog
     trials are identical, so a fog-only participant must be detected
     equally well whatever the colour cut. If that moves, something is
     wrong with the sampling and not with the design.
  5. RANGES AND TRADE-OFFS: the spread of differences each list uses,
     and how heavily edited its images are, since a list that detects
     better by using larger edits is buying detection with
     conspicuousness.
  6. CONFLICT BALANCE, as a distribution per version.

PLAIN HAZE IS OPTIONAL
----------------------
OPTIONAL_ROOTS names step-5 output roots built from the plain-haze
stimulus set. They are reported in a section of their own, never mixed
into the main table, because haze and haze_eq differ in brightness as
well as in contrast -- see the caveat printed with that section.
Anything absent is skipped with a note, so the script runs whether or
not that branch has been built.

To build it:
    step 3   FOG_FAMILY = "haze",  FOG_DIMENSION = "fog"
    step 4   MANIFEST -> that manifest, OUTPUT_ROOT -> ..._haze
    step 5   VARIANTS -> those pair tables, OUTPUT_ROOT -> ..._haze

Inputs
------
  TRIALS_ROOT     step 5's output root: <label>/trial_list.csv for the
                  shipping draw, seeds/<label>/seed_*/trial_list.csv
                  for the replicates
  VERSIONS        which labels under it to compare
  REFERENCE_LIST  the shipped list, for tracing the floors
  OPTIONAL_ROOTS  plain-haze candidates, if they exist

Outputs (under OUTPUT_ROOT)
---------------------------
  information.csv          per version: information, floors, spread
  information_by_seed.csv  the same, one row per draw
  detection.csv            version x sensitivity x type x feature,
                           mean over draws with the spread
  detection_by_seed.csv    the same, one row per draw
  differences.csv          paired against the baseline, with the SE
  fog_control.csv          section 4b on its own
  stream_check.csv         CHECK 4, pair by pair
  ranges.csv               difference ranges and edit-size trade-offs
  balance.csv              conflict balance per version
  false_alarms.csv         the independent check on the floors
  detection.png, information.png, differences.png
  choose_report.txt

Usage
-----
    conda activate drl2drv
    python choose_colour_condition_v4.py

CPU only. About 2.5s per draw, so a few minutes for a full sweep.
Set MAX_SEEDS to a small number for a quick trial run.
"""

import os
import re
import glob
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

# --- Input / output ---
#
# TRIALS_ROOT
#   Step 5's output root. The shipping draw of each version is
#   <root>/<label>/trial_list.csv and its replicates are
#   <root>/seeds/<label>/seed_<n>/trial_list.csv.
#
#   CHECK THIS AFTER EVERY STEP-5 RUN. Step 5 writes to a new numbered
#   directory rather than overwriting, so this will be _2, _3 and so on
#   as runs accumulate. Pointing at a stale root is the easiest way to
#   measure the wrong thing, and nothing downstream would notice.
TRIALS_ROOT = "data/trials_v4_ext"

# VERSIONS
#   Which labels under TRIALS_ROOT to compare. The four design versions
#   first, then the sensitivity versions, then the reproduction
#   control.
#
#   `reproduction` is NOT a candidate. It is step 5's control, which
#   rebuilds the shipped list, and it is here so that the shipped
#   list's numbers appear in the same table as everything else. It has
#   one draw only -- step 5 skips it in the seed sweep, since it is
#   checked against one fixed shipped list.
VERSIONS = [
    "current",
    "extra_levels",
    "larger_diffs",
    "both",
    "sens_cut_25_5",
    "sens_cut_35_3",
    "reproduction",
]

# LADDER_OF
#   Which colour ladder each version was built from -- step 5's
#   `variant`. CHECK 4 uses it: versions sharing a ladder must have
#   identical fog_only, together and conflict trials, because the
#   colour cut touches colour-only pairs and nothing else.
#
#   Versions on DIFFERENT ladders are expected to differ in every type
#   and are not judged. The vivid levels add images, hence new pairs of
#   every type: a fog pair at the p75 colour level did not exist on the
#   original ladder. That is the design, not a leak -- and it is why
#   the `current` against `both` comparison is read against the
#   between-draw spread rather than taken at face value.
#
#   A label left out of this map is not checked, and the report says so.
LADDER_OF = {
    "current":       "original",
    "larger_diffs":  "original",
    "extra_levels":  "extended",
    "both":          "extended",
    "sens_cut_25_5": "extended",
    "sens_cut_35_3": "extended",
    # `reproduction` is deliberately absent: it is built from the
    # legacy pair table AND the legacy single random stream, so it
    # cannot share trials with anything here.
}

# REFERENCE_LIST
#   The list the experiment currently ships, included in the
#   information table so that a change in the floors can be traced to
#   its cause rather than guessed at. One draw, by nature. Set to None
#   to leave it out.
REFERENCE_LIST = ("shipped", "data/trials_v4_ec/haze_eq/trial_list.csv")

# OPTIONAL_ROOTS
#   (suffix, step-5 root, labels) for the plain-haze branch. Absent
#   directories are skipped with a note, so this can stay populated
#   whether or not the branch exists. Reported separately -- see PLAIN
#   HAZE IS OPTIONAL in the header. Replicate draws are picked up the
#   same way if that root has them.
OPTIONAL_ROOTS = [
    ("haze", "data/trials_v4_ext_haze", ["current", "both"]),
]

# BASELINE
#   The version every difference is taken against. "current" is the
#   like-for-like comparison: original levels, no cut.
#
#   Note it is NOT the shipped list. The two differ in their magnitude
#   bins, and that difference alone moves the colour information by
#   about 38% -- see section 1 of the report. If you want "what we
#   would have had", that is the `reproduction` row.
BASELINE = "current"

# --- The estimator ---
#
# ESTIMATOR
#   "firth"  Firth's bias-reduced logistic regression, the registered
#            per-person fit.
#   "mle"    the plain maximum-likelihood fit with a token ridge, which
#            is what this script used before. Kept so the two can be
#            compared, not as an option for a reported result.
#
#   WHY THE ESTIMATOR CHANGED. The fit used to carry a ridge of 1e-6,
#   nominally to keep it finite when a participant's answers are
#   perfectly predicted by the model -- separation, where the
#   maximum-likelihood estimate does not exist. Measured on this
#   design, that ridge is 3.6e-11 of the diagonal it is added to,
#   because the predictors were in raw feature units and X'WX runs to
#   tens of thousands. It did nothing: refitting at 1e-3 moved the
#   weights by less than 1e-5.
#
#   So the fits were maximum-likelihood in practice and unprotected in
#   principle. The alternative of excluding separable fits was
#   considered and rejected: a participant who answers every colour
#   trial as the hypothesis predicts is the strongest evidence the
#   design can produce, and dropping them would bias against the
#   hypothesis. Firth is always finite, differs from the MLE by about
#   2% where there is no separation, and removes most of the
#   small-sample bias -- which matters because the registered control
#   comparison reads weight MAGNITUDES between groups, not just signs.
#
#   THE FLOORS ARE COMPUTED WITH THE SAME ESTIMATOR. A floor from one
#   estimator applied to weights from another is not a threshold, it is
#   an offset. Firth shrinks the guessers' weights and the responders'
#   weights alike, so the two effects largely cancel in a detection
#   rate -- "largely" being why this is re-run rather than argued.
ESTIMATOR = "firth"

# MLE_COMPARE
#   Also fit the floors and one detection row with the MLE, and report
#   how far the two estimators differ. For the record of a changed
#   estimator, not for a result. Costs one extra floor per draw.
MLE_COMPARE = True

# FIRTH_ITER / FIRTH_TOL / FIRTH_MAX_STEP
#   The Newton loop. FIRTH_MAX_STEP damps each step: starting from zero
#   on near-separable data, an undamped first step can overshoot far
#   enough to stall. 5.0 is in per-SD units, so it is large compared
#   with any plausible weight and only ever bites on the pathological
#   cases it exists for.
FIRTH_ITER = 200
FIRTH_TOL = 1e-10
FIRTH_MAX_STEP = 5.0

# RIDGE
#   Used only by the "mle" estimator and the comparison. See above for
#   why it is inert at this scale.
RIDGE = 1e-6

OUTPUT_ROOT = "data/choose_colour_v4"

# ON_EXISTING
#   "new" writes to OUTPUT_ROOT_2, _3, ... rather than overwriting.
ON_EXISTING = "new"

# MAX_SEEDS
#   Cap on draws per version, counting the shipping draw first. None
#   uses every draw step 5 built. Set it to 2 or 3 for a quick trial
#   run; leave it None for the numbers that get reported.
#
#   The report states how many draws each version actually used, so a
#   capped run cannot be mistaken for a full one.
MAX_SEEDS = None

# --- The predictors ---
#
# SD_SAT / SD_FOG
#   The corpus SDs, from the `stimulus_path` population of step 1's
#   corpus_feature_stats_v4.csv (n = 20,000). The same constants step 4
#   and the registered model use.
SD_SAT = 39.182
SD_FOG = 9.348

# ORIENTATION, and why only one of the two is negated.
#   A predictor is positive when image A is the one the hypothesis says
#   will be chosen: the LESS colourful image, the FOGGIER image.
#
#   d_sat is raw (sat_a - sat_b), so it is positive when A is MORE
#   colourful -- the opposite of the prediction -- and is negated.
#   d_fog already carries step 4's FOG_SIGN and is positive when A is
#   the foggier image, so it is used as it stands. Negating both would
#   flip the fog coefficient and nothing in the output would say so.
#
#   This is the same construction step 5 writes into `pred_sat` and
#   `pred_fog`. Those columns are used when present and rebuilt from
#   d_sat / d_fog when not, and CHECK 2 verifies the two agree.

# --- The sensitivity grid ---
#
# SENSITIVITY_SD
#   True weights for the simulated participants, written in per-SD
#   terms, and applied directly: the predictors are the registered
#   model's `pred_sat` and `pred_fog`, already in corpus-SD units, so
#   there is nothing to convert. Rule 1 is satisfied because the two
#   SDs are fixed constants, identical for every list.
#
#   Chosen from the baseline list, where detection runs 26% at 0.30 and
#   99% at 1.00. Below 0.30 everything is at the floor and above 1.00
#   everything is at the ceiling, and neither end separates the
#   candidates. The informative middle is 0.40 to 0.70.
SENSITIVITY_SD = [0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00]

# PARTICIPANT_TYPES
#   (colour weight, fog weight) as MULTIPLES of the sensitivity, so
#   each type is simulated across the whole grid.
#
#   `equal` is the realistic case and `colour_only` is the one the
#   re-run exists to improve. `fog_only` is the control: the colour cut
#   must not cost fog detection, and section 4b is where that shows.
PARTICIPANT_TYPES = {
    "equal":       (1.0, 1.0),
    "colour_only": (1.0, 0.0),
    "fog_only":    (0.0, 1.0),
}

# --- Floors and simulation sizes ---
#
# FLOOR_ALPHA
#   False alarms per person per feature. The floor is the weight that
#   this share of pure guessers exceeds by luck. 0.025 one-sided, as in
#   step 8, so the two agree about what "detected" means.
FLOOR_ALPHA = 0.025

# N_FLOOR_PEOPLE
#   Guessers used to LOCATE each floor. Large, because the floor is a
#   far percentile and a noisy floor would move every detection rate
#   computed against it.
N_FLOOR_PEOPLE = 20_000

# N_FALSE_PEOPLE
#   A SECOND, INDEPENDENT set of guessers, used only to check that the
#   floor delivers the false-alarm rate it promises. Reusing the people
#   who set the floor would be circular -- they hit it 2.5% of the time
#   by construction, so the check would pass whatever the floor was.
N_FALSE_PEOPLE = 20_000

# N_DETECT_PEOPLE
#   People per (draw, sensitivity, type) cell. 4,000 gives a Monte
#   Carlo error of about 0.8 points on a rate near 50% for ONE draw.
#   Over 20 draws the mean is far tighter than that, and the
#   uncertainty that matters is no longer Monte Carlo but the spread
#   between draws -- which is why the difference columns use the
#   paired standard error instead.
N_DETECT_PEOPLE = 4_000

# FLOOR_REPLICATES
#   Extra, independent sets of guessers used to locate each list's
#   floor A SECOND, THIRD... time, on the SHIPPING draw only. The
#   detection rates always use the first one; these exist to say how
#   much the floor itself moves with the guessers, which is the
#   question the colour/fog floor comparison turns on.
#
#   The floor is a 97.5th percentile, so it is noisier than a mean, and
#   the gap between the colour and fog floors -- the thing that flipped
#   between the shipped list and this baseline -- is a difference of two
#   such percentiles. Reporting it without its own error would invite
#   reading a flip that was luck.
#
#   1 turns this off.
FLOOR_REPLICATES = 5

RANDOM_SEED = 42

# Seed offsets, kept apart so the three populations are independent.
#
# THE SIMULATION SEEDS DEPEND ON THE DRAW INDEX, NOT ON THE VERSION --
# see the header. Every version faces the same people at a given
# index, which pairs the comparison; the people differ between
# indices, which is what makes averaging over draws worth anything.
SEED_FLOOR, SEED_FALSE, SEED_DETECT = 101, 202, 303

# --- Checks ---
#
# FIT_CHECK_TOL
#   CHECK 1 fits one simulated participant with the batch solver and
#   with the single-participant solver and requires them to agree. They
#   are the same model written twice, and a drift between them would
#   silently change every number here.
FIT_CHECK_TOL = 1e-6

# GRAD_CHECK_TOL
#   CHECK 1b differentiates log L + 0.5 log det(X'WX) numerically at
#   the returned estimate and requires the gradient to vanish. The
#   tolerance is loose because the derivative is a finite difference of
#   a log-likelihood near its maximum, where the curvature makes the
#   second-order error the limit; 1e-4 is far below anything a wrong
#   penalty would produce (an unpenalised fit misses by order 1).
GRAD_CHECK_TOL = 1e-4

# EQUIVAR_CHECK_TOL
#   CHECK 1e fits in raw units, rescales, and requires the per-SD fit
#   back. Firth is equivariant under a linear change of predictor
#   scale, so this is exact up to arithmetic.
EQUIVAR_CHECK_TOL = 1e-6

# PRED_CHECK_TOL
#   CHECK 2 rebuilds the oriented predictors from d_sat / d_fog and
#   requires them to match step 5's `pred_sat` / `pred_fog` columns.
#   This is what catches a sign convention or a unit changing upstream.
PRED_CHECK_TOL = 1e-9

# STREAM_CHECK_FATAL
#   CHECK 4 compares versions that share a ladder. True stops the run
#   when they differ on a type the colour cut does not touch.
#
#   Fatal by default, and that is the right default: if it fails, the
#   comparison in every table below is between two different random
#   draws of fog trials as well as between two designs, and no amount
#   of care reading the numbers recovers from that.
STREAM_CHECK_FATAL = True

TYPES = ["sat_only", "fog_only", "together", "conflict"]

# The types a colour cut must leave untouched.
UNTOUCHED_TYPES = ["fog_only", "together", "conflict"]


# ============================================================
# Helpers
# ============================================================

def resolve_output_root():
    """OUTPUT_ROOT, never overwriting an existing analysis."""
    def occupied(d):
        return os.path.isdir(d) and os.path.isfile(
            os.path.join(d, "choose_report.txt"))

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


def draw_paths(root, label):
    """Every draw of one version: the shipping one first.

    Returns [(seed, path)]. The shipping draw is <root>/<label>/ and
    carries step 5's RANDOM_SEED; the replicates are under
    <root>/seeds/<label>/seed_<n>/ and are sorted by seed so the DRAW
    INDEX is a stable property of the files rather than of the order
    the filesystem happened to list them in.
    """
    out = []
    ship = os.path.join(root, label, "trial_list.csv")
    if os.path.isfile(ship):
        out.append((None, ship))
    pat = os.path.join(root, "seeds", label, "seed_*", "trial_list.csv")
    reps = []
    for p in glob.glob(pat):
        m = re.search(r"seed_(\d+)", p)
        if m:
            reps.append((int(m.group(1)), p))
    out += sorted(reps)
    if MAX_SEEDS is not None:
        out = out[:MAX_SEEDS]
    return out


# ============================================================
# Model  (verbatim from simulate_power_v4.py)
# ============================================================
#
# Copied rather than imported, which is this pipeline's convention: a
# script is standalone and runs from its own settings block. The cost
# is that an edit has to be made in both places; the benefit is that
# neither script can be broken by a change made for the other's sake.
# CHECK 1 verifies the two solvers below still agree with each other.

def simulate_responses(X, beta, rng):
    """One synthetic participant's choices.

    X is (n_trials, 2) signed differences; beta is the true
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
    every participant in one pass. X is (n_trials, 2), Y is
    (n_people, n_trials). Returns (n_people, 2).
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


def simulate_many(X, W, rng):
    """Choices of many people at once. W is (n_people, 2) true weights."""
    p = 1.0 / (1.0 + np.exp(-(W @ X.T)))
    return (rng.random(p.shape) < p).astype(float)


# --- Firth's bias-reduced fit: the registered estimator -------------
#
# Written here and copied verbatim into step 6b and step 8, which
# inverts this pipeline's usual direction (step 8 is normally where the
# shared model functions live). It is here because this is where the
# estimator had to change first, and the note in each copy says so.

def fit_firth(X, y, n_iter=FIRTH_ITER, tol=FIRTH_TOL,
              max_step=FIRTH_MAX_STEP):
    """Firth's bias-reduced logistic regression, one participant.

    The score equation gains a term from the Jeffreys-prior penalty
    0.5 * log det(X' W X):

        U*(b)_j = sum_i [ y_i - p_i + h_i (0.5 - p_i) ] x_ij

    where h_i is the i-th diagonal of the hat matrix
    H = W^(1/2) X (X'WX)^(-1) X' W^(1/2).

    ALWAYS FINITE. Under separation the maximum-likelihood estimate
    runs off to infinity and whatever a Newton loop returns is set by
    its iteration cap; this returns a root. Verified: on perfectly
    separable data the estimate is identical at 200, 1,000 and 4,000
    iterations, with the penalised gradient at 7e-8.

    FINITE IS NOT MODEST, and that is worth knowing before relying on
    it. On this design a participant who answers every one of the 120
    colour-informative trials as predicted fits at about 9 per SD, and
    one who answers every trial as predicted at about 41 -- finite,
    estimable, and still two orders of magnitude above a typical 0.5.
    Step 6b measures how far anyone comes: the closest was 0.868 of
    colour-informative trials, against the 1.000 needed, so this does
    not arise here. If it ever did, one such participant would still
    dominate a group mean.
    """
    b = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -500.0, 500.0)))
        w = np.clip(p * (1 - p), 1e-12, None)
        A = (X * w[:, None]).T @ X
        Ainv = np.linalg.pinv(A)
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
    """fit_firth for many simulated people at once.

    The same estimator, WRITTEN OUT FOR TWO PREDICTORS. The registered
    model has exactly two, and the general form spends its time in two
    einsum contractions -- X'WX and the hat diagonal -- which for k = 2
    reduce to three matrix-vector products and three broadcasts. That
    is twice as fast, which matters at 20,000 guessers per floor per
    draw and 121 draws: it is the difference between a half-hour run
    and an hour.

    The cost is that this one is specific to k = 2, which the assertion
    below enforces rather than leaving to be discovered. `fit_firth` is
    the general, readable form and CHECK 1a holds the two to agreement
    on the real design, so the fast path is verified rather than
    trusted -- the same arrangement as fit_logit and fit_logit_batch.

    Also returns each person's own last step, so a fit that never
    settled can be counted rather than averaged in silently.
    """
    if X.shape[1] != 2:
        raise ValueError(
            f"fit_firth_batch is written out for 2 predictors, got "
            f"{X.shape[1]}. Use fit_firth, or restore the general "
            "einsum form.")
    x0, x1 = X[:, 0], X[:, 1]
    s00, s01, s11 = x0 * x0, x0 * x1, x1 * x1
    P = Y.shape[0]
    b = np.zeros((P, 2))
    last = np.full(P, np.inf)
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(b @ X.T, -500.0, 500.0)))
        w = np.clip(p * (1 - p), 1e-12, None)
        # X' W X, one matrix-vector product per distinct entry
        a00, a01, a11 = w @ s00, w @ s01, w @ s11
        det = a00 * a11 - a01 * a01
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        i00, i01, i11 = a11 / det, -a01 / det, a00 / det
        # the hat diagonal: h_t = w_t * x_t' (X'WX)^-1 x_t
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

    Used only by CHECK 1b, which differentiates it numerically at the
    returned estimate. That check is what distinguishes an
    implementation of Firth from something Firth-shaped.
    """
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -500.0, 500.0)))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    w = p * (1 - p)
    sign, logdet = np.linalg.slogdet((X * w[:, None]).T @ X)
    return -np.inf if sign <= 0 else ll + 0.5 * logdet


def fit_batch(X, Y):
    """The estimator ESTIMATOR names. Returns (weights, last step)."""
    if ESTIMATOR == "firth":
        return fit_firth_batch(X, Y)
    if ESTIMATOR == "mle":
        b = fit_logit_batch(X, Y, ridge=RIDGE)
        return b, np.zeros(len(b))
    raise ValueError(
        f"ESTIMATOR must be 'firth' or 'mle', got {ESTIMATOR!r}")


# ============================================================
# Loading
# ============================================================

def load_draw(label, seed, index, path):
    """One trial list, its RAW-unit predictors, and its descriptives.

    The predictors are built in RAW units -- log-odds per raw unit of
    sat_mean or contrast_local -- because that is what rule 1 requires
    the sensitivity grid to be expressed in. The corpus-SD version is
    kept alongside for the information figures, where a quantity on one
    scale for both features is what makes them comparable.
    """
    t = pd.read_csv(path)
    for c in ("d_sat", "d_fog"):
        if c not in t.columns:
            raise ValueError(f"{path} has no {c!r} column.")
    if "analysed" in t.columns:
        t = t[t.analysed.astype(bool)]
    t = t.reset_index(drop=True)
    if not len(t):
        raise ValueError(f"{path} has no analysed trials.")

    # THE REGISTERED MODEL'S PREDICTORS, in corpus-SD units, taken
    # from the columns step 5 wrote rather than rebuilt here -- so this
    # script fits what the registered model fits and step 8 fits.
    #
    # `X_check` rebuilds the same quantity from the raw differences, and
    # CHECK 2 requires the two to agree. That is what catches a sign
    # convention or a unit changing upstream, which would otherwise
    # show only as a detection curve quietly running the wrong way.
    X_check = np.column_stack([-t.d_sat.to_numpy(float) / SD_SAT,
                               t.d_fog.to_numpy(float) / SD_FOG])
    if {"pred_sat", "pred_fog"} <= set(t.columns):
        X = np.column_stack([t.pred_sat.to_numpy(float),
                             t.pred_fog.to_numpy(float)])
        from_cols = True
    else:
        X = X_check
        from_cols = False
    r = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    return {"label": label, "seed": seed, "index": index, "path": path,
            "shipping": index == 0, "trials": t,
            "X": X, "X_check": X_check, "from_cols": from_cols,
            "r": r}


def load_version(root, label, report, optional=False):
    """Every draw of one version."""
    paths = draw_paths(root, label)
    if not paths:
        if optional:
            print(f"  {label:14s} not built; skipped")
            return []
        raise FileNotFoundError(
            f"No trial list for {label!r} under {root}.\nExpected "
            f"{os.path.join(root, label, 'trial_list.csv')}\nRun step 5 "
            "(build_trials_v4_corrected.py) first, and check that "
            "TRIALS_ROOT points at\nits output -- step 5 writes to a "
            "new numbered directory on each run.")
    draws = [load_draw(label, s, i, p)
             for i, (s, p) in enumerate(paths)]
    n = len(draws)
    rs = [d["r"] for d in draws]
    print(f"  {label:14s} {n:2d} draw(s), {len(draws[0]['trials']):3d} "
          f"trials   r(predictors) {np.mean(rs):+.3f}"
          + (f" +/- {np.std(rs, ddof=1):.3f}" if n > 1 else ""))
    report.append((f"draws {label}", str(n)))
    return draws


def load_single(label, path, report):
    """A one-draw list given by an explicit path (the shipped list)."""
    if not os.path.isfile(path):
        print(f"  {label:14s} not found; skipped  ({path})")
        return []
    d = load_draw(label, None, 0, path)
    print(f"  {label:14s}  1 draw , {len(d['trials']):3d} trials   "
          f"r(predictors) {d['r']:+.3f}")
    report.append((f"draws {label}", "1"))
    return [d]


# ============================================================
# CHECK 1, CHECK 2, CHECK 4
# ============================================================

def check_solvers(draws, report):
    """CHECK 1: five checks on the estimator, run on the real design.

    Every number in this script comes from the batch solver, so the
    batch solver has to be the estimator it claims to be. These run on
    the registered model as it will actually be fitted: no intercept,
    `pred_sat` and `pred_fog` in corpus-SD units, on this list's own
    180 analysed trials.

      1a  batch against single-participant, same answers
      1b  the penalised objective's gradient is zero at the estimate.
          THE CHECK THAT MATTERS: it distinguishes Firth from
          something Firth-shaped, and needs no reference
          implementation
      1c  against the plain MLE where there is no separation -- a
          small shrinkage, not a shift
      1d  finite and stable on deliberately separable data, where the
          MLE is neither
      1e  equivariant to the predictor scale, which is what makes
          fitting in SD units the same estimator as fitting in raw
          units
    """
    s = draws[0]
    X = s["X"]
    rng = np.random.default_rng(RANDOM_SEED)
    beta = np.array([0.5, 0.5])          # per SD, directly
    Y = np.array([simulate_responses(X, beta, rng) for _ in range(64)])
    verdicts = []

    # 1a
    one = np.array([fit_firth(X, y) for y in Y])
    many, _ = fit_firth_batch(X, Y)
    w = float(np.abs(one - many).max())
    ok = w <= FIT_CHECK_TOL
    verdicts.append(ok)
    print(f"  1a batch vs single, 64 fits        worst {w:.2e}   "
          f"{'PASS' if ok else 'FAIL'}")
    report.append(("check 1a batch vs single",
                   f"{'PASS' if ok else 'FAIL'} ({w:.1e})"))

    # 1b -- the gradient of log L + 0.5 log det I at the estimate
    worst_g, h = 0.0, 1e-5
    for y, b in zip(Y[:16], one[:16]):
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            worst_g = max(worst_g, abs(
                (penalised_loglik(X, y, b + e)
                 - penalised_loglik(X, y, b - e)) / (2 * h)))
    ok = worst_g <= GRAD_CHECK_TOL
    verdicts.append(ok)
    print(f"  1b penalised gradient = 0          worst {worst_g:.2e}"
          f"   {'PASS' if ok else 'FAIL'}")
    report.append(("check 1b Firth objective",
                   f"{'PASS' if ok else 'FAIL'} ({worst_g:.1e})"))

    # 1c -- against the MLE, away from separation
    mle = np.array([fit_logit(X, y, ridge=RIDGE) for y in Y])
    shrink = float(np.median(np.abs(one) / np.maximum(np.abs(mle),
                                                      1e-12)))
    shift = float(np.abs(one - mle).max())
    print(f"  1c vs MLE, no separation           median "
          f"|b_firth|/|b_mle| {shrink:.4f}, worst shift {shift:.4f}")
    report.append(("check 1c Firth vs MLE",
                   f"shrink {shrink:.4f}, worst shift {shift:.4f}"))

    # 1d -- separable data: finite, and the same at any iteration cap
    y_sep = (X[:, 0] + X[:, 1] > 0).astype(float)
    b1 = fit_firth(X, y_sep, n_iter=FIRTH_ITER)
    b2 = fit_firth(X, y_sep, n_iter=FIRTH_ITER * 10)
    b_mle = fit_logit(X, y_sep, ridge=RIDGE)
    stable = float(np.abs(b1 - b2).max())
    ok = np.all(np.isfinite(b1)) and stable <= FIT_CHECK_TOL
    verdicts.append(ok)
    print(f"  1d separable data                  Firth "
          f"[{b1[0]:.3f} {b1[1]:.3f}], same at {FIRTH_ITER * 10} "
          f"iterations (diff {stable:.1e})   {'PASS' if ok else 'FAIL'}")
    print(f"       the MLE has no finite solution here: "
          f"[{b_mle[0]:.1f} {b_mle[1]:.1f}], which is where its "
          "iteration cap left it.")
    print("       Firth is FINITE, not modest -- see fit_firth.")
    print("       Step 6b measures how close anyone comes; on this "
          "design, not close.")
    report.append(("check 1d separable",
                   f"{'PASS' if ok else 'FAIL'} "
                   f"(Firth {b1[0]:.2f}/{b1[1]:.2f} finite)"))

    # 1e -- equivariance: fit raw, rescale, get the SD-unit fit
    D = np.array([SD_SAT, SD_FOG])
    raw = np.array([fit_firth(X * D, y) for y in Y[:16]])
    w = float(np.abs(raw * D - one[:16]).max())
    ok = w <= EQUIVAR_CHECK_TOL
    verdicts.append(ok)
    print(f"  1e equivariant to units            worst {w:.2e}   "
          f"{'PASS' if ok else 'FAIL'}")
    report.append(("check 1e equivariance",
                   f"{'PASS' if ok else 'FAIL'} ({w:.1e})"))

    if not all(verdicts):
        raise RuntimeError(
            "A solver check failed. Every detection rate below comes "
            "from this\nestimator, so nothing after this point would "
            "mean anything.")


def check_predictors(draws, report):
    """CHECK 2: the rebuilt predictors match step 5's columns.

    Step 5 writes `pred_sat` and `pred_fog` in corpus-SD units; this
    script rebuilds the same quantity from the raw differences. They
    must agree, and if they stop agreeing the cause is a sign
    convention or a unit changing upstream -- which would otherwise
    show up only as a detection curve that quietly ran the wrong way.
    """
    worst, checked = 0.0, 0
    for s in draws:
        t = s["trials"]
        if not s["from_cols"]:
            continue
        worst = max(worst, float(np.abs(s["X"] - s["X_check"]).max()))
        checked += 1
    if not checked:
        print("  check 2 (predictors): no list carries pred_sat / "
              "pred_fog; SKIPPED.")
        report.append(("check 2 predictors", "skipped"))
        return
    ok = worst <= PRED_CHECK_TOL
    print(f"  check 2 (rebuilt vs step 5's pred_*, {checked} draws): "
          f"worst difference {worst:.2e} -> {'PASS' if ok else 'FAIL'}")
    report.append(("check 2 predictors",
                   f"{'PASS' if ok else 'FAIL'} ({worst:.1e})"))
    if not ok:
        raise RuntimeError(
            "The predictors rebuilt here do not match step 5's "
            "pred_sat / pred_fog. Check the orientation note in the "
            "settings and SD_SAT / SD_FOG against step 4's.")


def _pair_keys(t):
    """A trial's two images, ignoring which is A and which is B."""
    return {tuple(sorted((str(a), str(b))))
            for a, b in zip(t.file_a, t.file_b)}


def check_streams(by_version, report, root):
    """CHECK 4: versions sharing a ladder share the untouched types.

    THE COMPARISON IN EVERY TABLE BELOW DEPENDS ON THIS. The colour cut
    filters colour-only pairs and nothing else, so two versions built
    from the same ladder must have the same fog_only, together and
    conflict trials -- at every draw index, since step 5 seeds each
    draw's streams from (seed, draw name).

    It was once false, and not visibly so. Step 5 drew every type off
    one random stream, so the cut changed how many numbers the colour
    draw consumed and moved every draw after it: `current` against
    `larger_diffs` shared 1 of 60 fog_only trials, 1 of 30 together and
    0 of 30 conflict. The fog condition was not a control, and colour
    information leaked in from types the cut never touches -- `both`
    gained 16.4 units of summed squared colour predictor from its
    `together` trials alone.

    Compared on sets of unordered image pairs, per type, per draw
    index. Versions on different ladders are reported and not judged;
    `sat_only` is reported and not judged, since changing it is the
    point.
    """
    print("\n" + "=" * 60)
    print("CHECK 4: STREAM SEPARATION")
    print("=" * 60)
    labs = [l for l in by_version
            if l in LADDER_OF and by_version[l]]
    skipped = [l for l in by_version if l not in LADDER_OF]
    if skipped:
        print(f"  not in LADDER_OF, so not checked: "
              f"{', '.join(skipped)}")
    rows, ok = [], True
    for i, a in enumerate(labs):
        for b in labs[i + 1:]:
            same = LADDER_OF[a] == LADDER_OF[b]
            da = {d["index"]: d for d in by_version[a]}
            db = {d["index"]: d for d in by_version[b]}
            shared_idx = sorted(set(da) & set(db))
            for t in TYPES:
                judged = same and t in UNTOUCHED_TYPES
                n_same = n_cmp = 0
                shared_tr = []
                for ix in shared_idx:
                    ta, tb = da[ix]["trials"], db[ix]["trials"]
                    ka = _pair_keys(ta[ta.trial_type == t])
                    kb = _pair_keys(tb[tb.trial_type == t])
                    n_cmp += 1
                    n_same += int(ka == kb)
                    shared_tr.append(len(ka & kb))
                    if judged and ka != kb:
                        ok = False
                rows.append({
                    "version_a": a, "version_b": b,
                    "same_ladder": same, "type": t,
                    "draws_compared": n_cmp,
                    "draws_identical": n_same,
                    "mean_trials_shared": (round(float(np.mean(
                        shared_tr)), 1) if shared_tr else np.nan),
                    "judged": judged,
                    "verdict": ("same" if n_same == n_cmp
                                else "DIFFERS" if judged
                                else "differs (expected)"),
                })
    if not rows:
        # Fewer than two versions share a ladder -- a one-version run,
        # or a VERSIONS list with nothing in common. Nothing to judge,
        # and an empty frame has no columns to index.
        print("  no two versions share a ladder; nothing to judge.")
        report.append(("check 4 streams", "nothing to judge"))
        return None, True
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(root, "stream_check.csv"), index=False)
    jt = tab[tab.judged]
    if not len(jt):
        print("  no two versions share a ladder; nothing to judge.")
        report.append(("check 4 streams", "nothing to judge"))
        return tab, True
    print(f"  {len(jt)} comparisons judged, over "
          f"{int(jt.draws_compared.max())} draw(s) each")
    print(jt[["version_a", "version_b", "type", "draws_compared",
              "draws_identical", "verdict"]].to_string(index=False))
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    report.append(("check 4 streams", "PASS" if ok else "FAIL"))
    if ok:
        print("  Within a ladder, the fog_only, together and conflict "
              "trials are the same\n  trials at every draw. So a "
              "difference between two such versions is the\n  colour "
              "cut, and the fog rows in section 4b are a real control.")
    else:
        bad = jt[jt.verdict == "DIFFERS"]
        print("  These must be identical and are not:")
        print(bad[["version_a", "version_b", "type", "draws_compared",
                   "draws_identical",
                   "mean_trials_shared"]].to_string(index=False))
        print("\n  The lists were built with a shared random stream, so "
              "the colour cut moved\n  the fog, together and conflict "
              "draws as well. Re-run step 5 with\n  per-type streams "
              "before reading anything below.")
        if STREAM_CHECK_FATAL:
            raise RuntimeError(
                "CHECK 4 FAILED: versions sharing a ladder have "
                "different fog_only,\ntogether or conflict trials. See "
                + os.path.join(root, "stream_check.csv")
                + ".\nEvery comparison below would confound the colour "
                "cut with a different\nrandom draw of fog trials. Set "
                "STREAM_CHECK_FATAL = False only to\ninspect the "
                "damage.")
    return tab, ok


# ============================================================
# The per-draw measurement
# ============================================================

def measure_draw(d):
    """Floors, false alarms and detection for one draw.

    One pass per draw, so each trial list is touched once. Everything
    is seeded from the DRAW INDEX rather than the version, so that
    versions are paired within an index and independent across indices
    -- see the header.
    """
    X = d["X"]
    ix = d["index"]

    # --- the floor: rule 2, this draw's own guessers ---
    #
    # THE SAME ESTIMATOR AS THE WEIGHTS IT JUDGES. Firth shrinks the
    # guessers' fitted weights as well as the responders', so a floor
    # from the MLE applied to Firth weights would not be a threshold.
    rng = np.random.default_rng([RANDOM_SEED, SEED_FLOOR, ix])
    guess = simulate_many(X, np.zeros((N_FLOOR_PEOPLE, 2)), rng)
    est, _ = fit_batch(X, guess)
    floor = np.percentile(est, 100 * (1 - FLOOR_ALPHA), axis=0)
    d["floor"] = floor

    # The other estimator's floor on the SAME guessers, for the record
    # of a changed estimator. Not used for anything.
    alt = {}
    if MLE_COMPARE:
        other = (fit_logit_batch(X, guess, ridge=RIDGE)
                 if ESTIMATOR == "firth" else
                 fit_firth_batch(X, guess)[0])
        fl2 = np.percentile(other, 100 * (1 - FLOOR_ALPHA), axis=0)
        alt = {"floor_colour_SD_other": float(fl2[0]),
               "floor_fog_SD_other": float(fl2[1]),
               "floor_colour_shift": float(floor[0] - fl2[0]),
               "floor_fog_shift": float(floor[1] - fl2[1])}

    # The floor's OWN error, from further independent guesser sets.
    # Shipping draws only -- across draws the spread already mixes the
    # design sample in, and this is the part that is pure guesser luck.
    # Detection always uses the first floor above, never these.
    gap_reps = [float(floor[0] - floor[1])]
    if d["shipping"] and FLOOR_REPLICATES > 1:
        for rep in range(1, FLOOR_REPLICATES):
            r2 = np.random.default_rng(
                [RANDOM_SEED, SEED_FLOOR, ix, rep])
            e2, _ = fit_batch(
                X, simulate_many(X, np.zeros((N_FLOOR_PEOPLE, 2)), r2))
            f2 = np.percentile(e2, 100 * (1 - FLOOR_ALPHA), axis=0)
            gap_reps.append(float(f2[0] - f2[1]))

    # --- information, in corpus-SD units, overall and per type ---
    t = d["trials"]
    ss = (X ** 2).sum(axis=0)
    info = {"SS_colour": float(ss[0]), "SS_fog": float(ss[1]),
            "info_ratio": float(ss[0] / max(1e-12, ss[1]))}
    if "trial_type" in t.columns:
        for ty in TYPES:
            m = (t.trial_type == ty).to_numpy()
            info[f"SS_colour_{ty}"] = float((X[m, 0] ** 2).sum())
            info[f"SS_fog_{ty}"] = float((X[m, 1] ** 2).sum())
        info["SS_colour_outside_sat_only"] = (
            info["SS_colour"] - info["SS_colour_sat_only"])
        info["SS_fog_outside_sat_only"] = (
            info["SS_fog"] - info["SS_fog_sat_only"])

    # --- false alarms: a SECOND, independent set of guessers ---
    rng = np.random.default_rng([RANDOM_SEED, SEED_FALSE, ix])
    est, _ = fit_batch(
        X, simulate_many(X, np.zeros((N_FALSE_PEOPLE, 2)), rng))
    hit = est > floor
    false_rates = {"false_colour": float(hit[:, 0].mean()),
                   "false_fog": float(hit[:, 1].mean()),
                   "either": float(hit.any(axis=1).mean())}

    # --- detection, over the grid and the participant types ---
    det = []
    for si, sens in enumerate(SENSITIVITY_SD):
        for ti, (tname, (ms, mf)) in enumerate(
                PARTICIPANT_TYPES.items()):
            rng = np.random.default_rng(
                [RANDOM_SEED, SEED_DETECT, ix, si, ti])
            # Per SD, applied directly: the predictors are already
            # in those units, so there is nothing to convert.
            W = np.zeros((N_DETECT_PEOPLE, 2))
            W[:, 0] = sens * ms
            W[:, 1] = sens * mf
            est, _ = fit_batch(X, simulate_many(X, W, rng))
            hit = est > floor
            det.append({"sensitivity_SD": sens, "participant": tname,
                        "detect_colour": float(hit[:, 0].mean()),
                        "detect_fog": float(hit[:, 1].mean())})

    so = t[t.trial_type == "sat_only"] if "trial_type" in t else t
    row = {"list": d["label"], "seed": d["seed"], "draw": ix,
           "shipping": d["shipping"], "trials": len(t), "r": d["r"],
           "estimator": ESTIMATOR,
           "floor_colour_SD": float(floor[0]),
           "floor_fog_SD": float(floor[1]),
           "floor_gap_SD": float(floor[0] - floor[1]),
           "floor_gap_mc_sd": (float(np.std(gap_reps, ddof=1))
                               if len(gap_reps) > 1 else np.nan),
           "floor_gap_reps": len(gap_reps),
           "sat_only_median_abs_d": (float(so.d_sat.abs().median())
                                     if len(so) else np.nan),
           **info, **false_rates, **alt}
    det = pd.DataFrame(det).assign(list=d["label"], seed=d["seed"],
                                   draw=ix)
    return row, det


def measure_all(draws, label=""):
    """measure_draw over every draw, with progress."""
    print(f"\n  measuring {len(draws)} draw(s){label}: "
          f"{N_FLOOR_PEOPLE:,} guessers for each floor, "
          f"{N_FALSE_PEOPLE:,} fresh\n  for the check, "
          f"{N_DETECT_PEOPLE:,} per detection cell "
          f"({len(SENSITIVITY_SD)} x {len(PARTICIPANT_TYPES)} cells)")
    t0 = time.time()
    rows, dets = [], []
    for n, d in enumerate(draws, 1):
        row, det = measure_draw(d)
        rows.append(row)
        dets.append(det)
        if n % 10 == 0 or n == len(draws):
            print(f"    {n}/{len(draws)}  ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows), pd.concat(dets, ignore_index=True)


# ============================================================
# Aggregation over draws
# ============================================================

def mean_sd(g, col):
    """Mean, between-draw SD, and the SE of the mean."""
    v = g[col].to_numpy(float)
    n = len(v)
    sd = float(np.std(v, ddof=1)) if n > 1 else np.nan
    return (float(v.mean()), sd,
            (sd / np.sqrt(n)) if n > 1 else np.nan, n)


def summarise_information(by_seed):
    """1. Information and floors per version, over draws.

    The SD columns are between-draw spread: how much this quantity
    moves from one sample of 180 trials to the next, with the design
    held fixed. A difference between two versions smaller than this is
    not a property of the designs.
    """
    print("\n" + "=" * 60)
    print("1. INFORMATION AND FLOORS")
    print("=" * 60)
    rows = []
    for lab, g in by_seed.groupby("list", sort=False):
        r = {"list": lab, "draws": len(g)}
        for col, short in (("SS_colour", "SS_colour"),
                           ("SS_fog", "SS_fog"),
                           ("info_ratio", "info_ratio"),
                           ("floor_colour_SD", "floor_colour_SD"),
                           ("floor_fog_SD", "floor_fog_SD")):
            m, sd, se, _ = mean_sd(g, col)
            r[short] = round(m, 3 if "floor" in col or "ratio" in col
                             else 1)
            r[short + "_sd"] = (round(sd, 3) if np.isfinite(sd)
                                else np.nan)
        # WHICH FLOOR IS LOWER, and whether that is more than luck.
        # The gap is colour floor minus fog floor, per draw, so its
        # spread over draws is the error on the comparison. For a
        # one-draw list the error comes from FLOOR_REPLICATES instead:
        # independent guesser sets on the same list.
        gm, gsd, gse, gn = mean_sd(g, "floor_gap_SD")
        mc = float(g.floor_gap_mc_sd.dropna().mean()) if len(
            g.floor_gap_mc_sd.dropna()) else np.nan
        err = gse if np.isfinite(gse) else mc
        r["floor_gap_SD"] = round(gm, 3)
        r["floor_gap_err"] = (round(float(err), 3)
                              if np.isfinite(err) else np.nan)
        r["floor_gap_err_from"] = ("draws" if np.isfinite(gse)
                                   else "guesser sets")
        lower = "colour" if gm < 0 else "fog"
        r["lower_floor"] = (
            lower if (np.isfinite(err) and abs(gm) > 2 * err)
            else f"{lower}?" if np.isfinite(err) else lower)
        if "SS_colour_outside_sat_only" in g.columns:
            r["SS_colour_in_sat_only"] = round(
                float(g.SS_colour_sat_only.mean()), 1)
            r["SS_colour_outside"] = round(
                float(g.SS_colour_outside_sat_only.mean()), 1)
            r["SS_fog_outside"] = round(
                float(g.SS_fog_outside_sat_only.mean()), 3)
        r["sat_only_median_abs_d"] = round(
            float(g.sat_only_median_abs_d.mean()), 1)
        rows.append(r)
    info = pd.DataFrame(rows)
    show = ["list", "draws", "SS_colour", "SS_colour_sd", "SS_fog",
            "SS_fog_sd", "info_ratio", "floor_colour_SD",
            "floor_fog_SD", "floor_gap_SD", "floor_gap_err",
            "lower_floor"]
    print(info[show].to_string(index=False))
    print("\n  info_ratio is colour information over fog information,")
    print("  both as sums of squared predictors in corpus-SD units. A")
    print("  ratio above 1 means the list carries MORE colour")
    print("  information than fog information.")
    print("  The _sd columns are the spread BETWEEN DRAWS of one")
    print("  version, which is the yardstick for every difference.")
    print("\n  floor_gap_SD is the colour floor minus the fog floor:")
    print("  negative means a colour-following participant is easier")
    print("  to catch than a fog-following one on that list. Its error")
    print("  is the spread over draws, or -- for a one-draw list --")
    print(f"  over {FLOOR_REPLICATES} independent guesser sets on the "
          "same list.")
    print("  `lower_floor` carries a ? when the gap is inside 2 of")
    print("  those errors, i.e. when the ordering is not established.")
    if "SS_colour_outside" in info.columns:
        print("\n  Where the colour information sits, and the fog")
        print("  information that the cut must not touch:")
        print(info[["list", "SS_colour_in_sat_only",
                    "SS_colour_outside",
                    "SS_fog_outside"]].to_string(index=False))
        print("\n  Within a ladder the last two columns are equal -- "
              "the cut\n  changes colour-only trials and nothing else. "
              "They were NOT\n  equal before step 5 separated the "
              "random streams, and that\n  inequality was the leak "
              "CHECK 4 now guards against.")
    return info


def summarise_detection(det_by_seed):
    """2. Detection per version, over draws.

    Three spreads are reported, and they answer different questions:

      sd_between  how much the rate moves from draw to draw. This
                  carries BOTH the design sample and the simulated
                  participants, and it is the honest spread of "what
                  this version gives you".
      sd_mc       the part that is simulated-participant noise alone,
                  sqrt(p(1-p)/N) -- known, not estimated.
      sd_design   what is left after removing it, in quadrature. This
                  is the draw-to-draw variation of the DESIGN, and it
                  is the quantity that does not shrink by simulating
                  more people.
    """
    print("\n" + "=" * 60)
    print("2. DETECTION")
    print("=" * 60)
    rows = []
    for (lab, sens, part), g in det_by_seed.groupby(
            ["list", "sensitivity_SD", "participant"], sort=False):
        for feat in ("detect_colour", "detect_fog"):
            m, sd, se, n = mean_sd(g, feat)
            mc = np.sqrt(max(m * (1 - m), 0.0) / N_DETECT_PEOPLE)
            design = (np.sqrt(max(sd ** 2 - mc ** 2, 0.0))
                      if np.isfinite(sd) else np.nan)
            rows.append({
                "list": lab, "sensitivity_SD": sens,
                "participant": part,
                "feature": feat.replace("detect_", ""),
                "detect": round(m, 4), "sd_between": round(sd, 4),
                "se_mean": round(se, 4), "sd_mc": round(mc, 4),
                "sd_design": round(design, 4), "draws": n,
            })
    det = pd.DataFrame(rows)
    for t in PARTICIPANT_TYPES:
        feat = "colour" if t != "fog_only" else "fog"
        sub = det[(det.participant == t) & (det.feature == feat)]
        print(f"\n  {t}  ({feat} detection, mean over draws):")
        print(sub.pivot(index="sensitivity_SD", columns="list",
                        values="detect").round(3).to_string())
        print(f"  between-draw SD, same cells:")
        print(sub.pivot(index="sensitivity_SD", columns="list",
                        values="sd_between").round(3).to_string())
    return det


def summarise_false_alarms(by_seed, report):
    """CHECK 3: does each floor deliver the false-alarm rate it
    promises, on people it has never seen?

    An INDEPENDENT set of guessers, seeded apart from the ones that
    located the floor. Checking against the floor-setting people would
    be circular: they exceed it FLOOR_ALPHA of the time by
    construction, whatever the floor is, so the check could not fail.
    """
    print("\n" + "=" * 60)
    print("3. FALSE ALARMS  (independent guessers)")
    print("=" * 60)
    rows = []
    for lab, g in by_seed.groupby("list", sort=False):
        rows.append({
            "list": lab, "draws": len(g),
            "false_colour": round(float(g.false_colour.mean()), 4),
            "false_fog": round(float(g.false_fog.mean()), 4),
            "either": round(float(g.either.mean()), 4),
            "worst_draw_colour": round(float(g.false_colour.max()), 4),
            "worst_draw_fog": round(float(g.false_fog.max()), 4)})
    fa = pd.DataFrame(rows)
    print(fa.to_string(index=False))
    # The mean over draws is over independent guesser sets, so its
    # error falls with the number of draws as well as with N.
    n_tot = int(by_seed.groupby("list").size().min()) * N_FALSE_PEOPLE
    se = np.sqrt(FLOOR_ALPHA * (1 - FLOOR_ALPHA) / n_tot)
    worst = float(np.abs(
        fa[["false_colour", "false_fog"]].to_numpy() - FLOOR_ALPHA
    ).max())
    ok = worst <= 4 * se
    print(f"\n  expected {FLOOR_ALPHA:.3f} per feature; Monte Carlo "
          f"error of a version's mean\n  {se:.5f} "
          f"(at least {n_tot:,} guessers per version)")
    print(f"  worst deviation {worst:.4f} ({worst / se:.1f} MC errors) "
          f"-> {'PASS' if ok else 'CHECK'}")
    print("  `either` is the chance a guesser is called a responder on "
          "at least one\n  feature -- roughly twice the per-feature "
          "rate, and the number that\n  matters if the analysis tests "
          "both.")
    report.append(("check 3 false alarms",
                   f"{'PASS' if ok else 'CHECK'} (worst {worst:.4f})"))
    return fa


# ============================================================
# 4. Differences from the baseline, paired by draw
# ============================================================

def differences(det_by_seed, report):
    """4. Each version against the baseline, paired by draw index.

    The difference is taken WITHIN each draw index and then averaged,
    and its standard error is the spread of those paired differences.
    Pairing is what makes this worth doing: at a given index both
    versions face the same simulated people, so the participant noise
    is common to the two rates and cancels in their difference.

    This replaces the single-draw Monte Carlo error the first version
    of this script reported. That error described only the simulated
    people and said nothing about the design sample -- which is the
    larger of the two, and the one that made a 2.3-point gap on the
    fog control look real when it was a different draw of fog trials.

    `resolvable` is |mean difference| > 2 x SE, so it now answers "is
    this bigger than the variation between draws of the same design?"
    rather than "is this bigger than the simulation noise?".
    """
    print("\n" + "=" * 60)
    print(f"4. DIFFERENCE FROM {BASELINE!r}  (paired by draw)")
    print("=" * 60)
    base = det_by_seed[det_by_seed.list == BASELINE]
    if not len(base):
        print(f"  {BASELINE!r} is not among the lists; skipped.")
        return pd.DataFrame()
    key = ["sensitivity_SD", "participant", "draw"]
    rows = []
    for lab, g in det_by_seed.groupby("list", sort=False):
        if lab == BASELINE:
            continue
        m = g.merge(base, on=key, suffixes=("", "_b"))
        if not len(m):
            continue
        for feat in ("detect_colour", "detect_fog"):
            m = m.assign(_d=m[feat] - m[feat + "_b"])
            for (sens, part), gg in m.groupby(
                    ["sensitivity_SD", "participant"], sort=False):
                v = gg["_d"].to_numpy(float)
                n = len(v)
                sd = float(np.std(v, ddof=1)) if n > 1 else np.nan
                se = sd / np.sqrt(n) if n > 1 else np.nan
                p1 = float(gg[feat].mean())
                p0 = float(gg[feat + "_b"].mean())
                # With one draw there is nothing to pair, so fall back
                # to the Monte Carlo error and say so in the column.
                mc = np.sqrt(p1 * (1 - p1) / N_DETECT_PEOPLE
                             + p0 * (1 - p0) / N_DETECT_PEOPLE)
                use = se if np.isfinite(se) else mc
                ship = gg[gg.draw == 0]
                rows.append({
                    "list": lab, "sensitivity_SD": sens,
                    "participant": part,
                    "feature": feat.replace("detect_", ""),
                    "baseline": round(p0, 4), "this_list": round(p1, 4),
                    "difference": round(float(v.mean()), 4),
                    "paired_sd": (round(sd, 4) if np.isfinite(sd)
                                  else np.nan),
                    "se": round(float(use), 4),
                    "se_from": "paired draws" if np.isfinite(se)
                               else "monte carlo (1 draw)",
                    "draws": n,
                    "shipping_draw_difference": (
                        round(float(ship["_d"].iloc[0]), 4)
                        if len(ship) else np.nan),
                    "mc_error_one_draw": round(float(mc), 4),
                    "resolvable": bool(abs(v.mean()) > 2 * use),
                })
    diff = pd.DataFrame(rows)
    if len(diff):
        show = diff[(diff.participant == "colour_only")
                    & (diff.feature == "colour")]
        print("  colour detection for a colour-only participant, "
              "difference from baseline:\n")
        print(show.pivot(index="sensitivity_SD", columns="list",
                         values="difference").round(3).to_string())
        print("\n  standard error of that difference (paired over "
              "draws):\n")
        print(show.pivot(index="sensitivity_SD", columns="list",
                         values="se").round(3).to_string())
        print(f"\n  typical SE {show.se.median():.4f}, so differences "
              f"under about {2 * show.se.median():.3f} are\n  within "
              "the variation between draws of one design.")
        n_un = int((~show.resolvable).sum())
        print(f"  {len(show) - n_un} of {len(show)} cells resolvable at "
              "2 SE.")
    return diff


def fog_control(diff, report):
    """4b. The fog control, on its own.

    Within a ladder, the fog_only, together and conflict trials are
    identical (CHECK 4), so a participant who follows fog alone must be
    detected equally well whichever colour cut is applied. The fit uses
    all 180 trials, so the colour trials do enter it -- a small
    difference is expected -- but a LARGE one would mean the colour cut
    is costing fog detection, which is the one thing it must not do.

    This is reported separately because on the first run it was not a
    control at all: the versions had different fog trials, and the
    2.3-point gap between `current` and `both` could not be attributed.
    """
    print("\n" + "=" * 60)
    print("4b. THE FOG CONTROL")
    print("=" * 60)
    if not len(diff):
        print("  no differences computed; skipped.")
        return pd.DataFrame()
    base_ladder = LADDER_OF.get(BASELINE)
    fc = diff[(diff.participant == "fog_only")
              & (diff.feature == "fog")].copy()
    fc["same_ladder_as_baseline"] = [
        LADDER_OF.get(l) == base_ladder for l in fc.list]
    same = fc[fc.same_ladder_as_baseline]
    print(f"  fog detection for a fog-only participant, difference "
          f"from {BASELINE!r}:\n")
    print(fc.pivot(index="sensitivity_SD", columns="list",
                   values="difference").round(3).to_string())
    if len(same):
        worst = float(same.difference.abs().max())
        med_se = float(same.se.median())
        ok = bool((same.difference.abs() <= 2 * same.se).all())
        print(f"\n  Versions on the SAME ladder as {BASELINE!r} "
              f"({', '.join(sorted(set(same.list)))}):")
        print(f"    worst |difference| {worst:.4f}, typical SE "
              f"{med_se:.4f}  -> {'PASS' if ok else 'CHECK'}")
        print("    These share their fog, together and conflict trials "
              "with the\n    baseline, so this is the clean control: "
              "the colour cut should\n    not move fog detection, and "
              "any residual comes from the colour\n    trials still "
              "entering the same 180-trial fit.")
        report.append(("4b fog control (same ladder)",
                       f"{'PASS' if ok else 'CHECK'} "
                       f"(worst {worst:.4f}, SE {med_se:.4f})"))
    other = fc[~fc.same_ladder_as_baseline]
    if len(other):
        print(f"\n  Versions on a DIFFERENT ladder "
              f"({', '.join(sorted(set(other.list)))}) are NOT a "
              "control:\n    the vivid levels add new pairs of every "
              "type, so their fog trials\n    are a different draw by "
              "design. Read these against the between-draw\n    spread "
              "in section 2, not as a cost of the cut.")
    return fc


# ============================================================
# 5. Ranges and trade-offs
# ============================================================

def ranges_and_tradeoffs(by_version, report):
    """5. What each version spends to get its detection.

    A list can raise colour detection in two ways: by spreading its
    trials over a wider range of differences, or by using larger edits.
    The second is not free -- larger colour edits also look more
    edited, and an image a participant can see has been manipulated is
    a different stimulus from one they cannot. These columns are what
    make that cost visible next to the gain.

    Averaged over draws, since both quantities are properties of the
    sample as much as of the design.
    """
    print("\n" + "=" * 60)
    print("5. RANGES AND TRADE-OFFS")
    print("=" * 60)
    rows = []
    for lab, draws in by_version.items():
        if not draws:
            continue
        acc = []
        for d in draws:
            t = d["trials"]
            so = (t[t.trial_type == "sat_only"]
                  if "trial_type" in t else t)
            r = {}
            for nm, col, sub in (("colour", "d_sat", so),
                                 ("fog", "d_fog", t)):
                v = sub[col].abs()
                r[f"{nm}_min"] = float(v.min())
                r[f"{nm}_median"] = float(v.median())
                r[f"{nm}_max"] = float(v.max())
                r[f"{nm}_span_SD"] = float(
                    (v.max() - v.min())
                    / (SD_SAT if nm == "colour" else SD_FOG))
            if {"chroma_a", "chroma_b"} <= set(t.columns):
                ed = pd.concat([(t.chroma_a - 1).abs(),
                                (t.chroma_b - 1).abs()])
                r["mean_chroma_edit"] = float(ed.mean())
                r["max_chroma_edit"] = float(ed.max())
            if "mag_bin" in t.columns and len(so):
                r["sat_bins_used"] = float(so.mag_bin.nunique())
            acc.append(r)
        a = pd.DataFrame(acc)
        out = {"list": lab, "draws": len(a)}
        for c in a.columns:
            out[c] = round(float(a[c].mean()),
                           3 if "chroma" in c or "span" in c else 1)
        rows.append(out)
    rng = pd.DataFrame(rows)
    print(rng.to_string(index=False))
    print("\n  mean_chroma_edit is the average distance of a shown "
          "image from the")
    print("  unmanipulated original (|k - 1|). A list that detects "
          "better AND sits")
    print("  higher here is buying detection with conspicuousness, "
          "which the")
    print("  grid-screen inspection at step 7 should take into "
          "account.")
    return rng


# ============================================================
# 6. Conflict balance
# ============================================================

def balance_distribution(by_version, report):
    """6. The balance of each version's conflict trials.

    Conflict trials are only interpretable when the two cues are
    comparably strong, so the distribution of the ratio -- not just its
    median -- is what says whether a version's conflict trials do their
    job.

    THE SHIPPING DRAW'S distribution is reported, with the mean and
    spread of the median and of share_below_1 over all draws beside it.
    On the first run three versions reported identical distributions
    here, which looked like a property of the design and was an
    artefact of the shared random stream: their conflict trials were
    the same 30 trials by accident. Within a ladder they are now
    identical BY CONSTRUCTION, which is the point, and across ladders
    they differ because the pools differ.
    """
    print("\n" + "=" * 60)
    print("6. CONFLICT BALANCE")
    print("=" * 60)
    rows = []
    for lab, draws in by_version.items():
        if not draws:
            continue
        ship = draws[0]["trials"]
        if ("balance" not in ship.columns
                or "trial_type" not in ship.columns):
            continue
        c = ship[ship.trial_type == "conflict"].balance.dropna()
        if not len(c):
            continue
        q = c.quantile([0, .1, .25, .5, .75, .9, 1]).round(3)
        meds, shares = [], []
        for d in draws:
            t = d["trials"]
            cc = t[t.trial_type == "conflict"].balance.dropna()
            if len(cc):
                meds.append(float(cc.median()))
                shares.append(float((cc < 1).mean()))
        rows.append({
            "list": lab, "draws": len(meds), "n_conflict": len(c),
            "min": q.iloc[0], "p10": q.iloc[1], "p25": q.iloc[2],
            "median": q.iloc[3], "p75": q.iloc[4], "p90": q.iloc[5],
            "max": q.iloc[6],
            "share_below_1": round(float((c < 1).mean()), 3),
            "median_over_draws": round(float(np.mean(meds)), 3),
            "median_sd": (round(float(np.std(meds, ddof=1)), 3)
                          if len(meds) > 1 else np.nan),
            "share_below_1_over_draws": round(float(np.mean(shares)), 3),
        })
    bal = pd.DataFrame(rows)
    if len(bal):
        print(bal.to_string(index=False))
        print("\n  Quantiles are the SHIPPING draw; the last three "
              "columns average over\n  every draw. A balance of 1 "
              "means a participant with equal weights in\n  the "
              "registered model is at 50/50 on that trial. "
              "`share_below_1` is the\n  share of conflict trials "
              "where the fog cue is the stronger of the two.")
    else:
        print("  no list carries a `balance` column; skipped.")
    return bal


# ============================================================
# Figures
# ============================================================

def make_figures(det, info, diff, out_dir):
    labs = list(dict.fromkeys(det.list))

    # 1. detection curves, with the between-draw spread as a band
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (tname, feat) in zip(axes, [
            ("colour_only", "colour"), ("equal", "colour"),
            ("fog_only", "fog")]):
        sub = det[(det.participant == tname) & (det.feature == feat)]
        for lab in labs:
            g = sub[sub.list == lab].sort_values("sensitivity_SD")
            if not len(g):
                continue
            ax.plot(g.sensitivity_SD, g.detect, marker="o", ms=4,
                    label=lab, lw=1.6,
                    ls="--" if lab in ("reproduction", "shipped")
                    else "-")
            sd = g.sd_between.fillna(0).to_numpy()
            ax.fill_between(g.sensitivity_SD, g.detect - sd,
                            g.detect + sd, alpha=0.12, lw=0)
        ax.set_title(f"{tname}\n{feat} detection")
        ax.set_xlabel("true sensitivity (per SD)")
        ax.grid(alpha=0.3)
        ax.axhline(0.8, color="k", lw=0.6, ls=":")
    axes[0].set_ylabel("share of people detected")
    axes[0].legend(fontsize=7)
    fig.suptitle("Detection by trial list  (mean over draws, band = "
                 "between-draw SD; dotted line = 80%)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "detection.png"), dpi=150)
    plt.close(fig)

    # 2. information, with the between-draw spread as an error bar
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(info))
    ax.bar(x - 0.2, info.SS_colour, 0.4, label="colour",
           yerr=info.SS_colour_sd.fillna(0), capsize=3)
    ax.bar(x + 0.2, info.SS_fog, 0.4, label="fog",
           yerr=info.SS_fog_sd.fillna(0), capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(info.list, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("sum of squared predictors (corpus-SD units)")
    ax.set_title("Information per trial list\n"
                 "what the logistic fit has to work with "
                 "(error bar = between-draw SD)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "information.png"), dpi=150)
    plt.close(fig)

    # 3. the differences that decide, with their paired SE
    if not len(diff):
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (part, feat, title) in zip(axes, [
            ("colour_only", "colour",
             "colour detection gained\n(colour-only participant)"),
            ("fog_only", "fog",
             "fog detection, the control\n(fog-only participant)")]):
        sub = diff[(diff.participant == part) & (diff.feature == feat)]
        for lab in dict.fromkeys(sub.list):
            g = sub[sub.list == lab].sort_values("sensitivity_SD")
            ax.errorbar(g.sensitivity_SD, g.difference,
                        yerr=2 * g.se.fillna(0), marker="o", ms=4,
                        lw=1.5, capsize=3, label=lab)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("true sensitivity (per SD)")
        ax.set_ylabel(f"difference from {BASELINE}")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Difference from the baseline, paired by draw "
                 "(error bars = 2 SE of the paired difference)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "differences.png"), dpi=150)
    plt.close(fig)


# ============================================================
# Report
# ============================================================

def write_report(out_dir, report, info, det, diff, fc, rng_tab, bal,
                 fa, streams, opt_info, opt_det, n_draws, t0):
    path = os.path.join(out_dir, "choose_report.txt")
    with open(path, "w") as f:
        def W(s=""):
            f.write(s + "\n")

        W("CHOOSE THE COLOUR CONDITION  (v4 axes, step 6)")
        W("=" * 70)
        W(f"Trial lists: {TRIALS_ROOT}")
        W(f"Baseline   : {BASELINE}")
        W(f"Draws      : {n_draws}"
          + ("" if MAX_SEEDS is None
             else f"   (MAX_SEEDS = {MAX_SEEDS}: A CAPPED RUN)"))
        W(f"Sensitivity: {SENSITIVITY_SD} (per SD), applied directly "
          "to pred_sat / pred_fog")
        W(f"Estimator  : {ESTIMATOR}"
          + ("  (Firth bias-reduced; floors computed with the same "
             "estimator)" if ESTIMATOR == "firth" else
             "  (plain MLE, ridge " + str(RIDGE) + ")"))
        W(f"SDs        : sat_mean {SD_SAT}, contrast_local {SD_FOG} "
          "(step 1, stimulus_path, n = 20,000)")
        W(f"People     : {N_DETECT_PEOPLE:,} per cell per draw, "
          f"{N_FLOOR_PEOPLE:,} for each floor,")
        W(f"             {N_FALSE_PEOPLE:,} independent for the check")
        W(f"Floor      : {FLOOR_ALPHA:.1%} false alarms per person "
          "per feature, one-sided")
        W(f"Seed       : {RANDOM_SEED}, offsets "
          f"{SEED_FLOOR}/{SEED_FALSE}/{SEED_DETECT}, indexed by DRAW "
          "not by version")
        W(f"Elapsed    : {time.time() - t0:.1f}s")
        W()
        for k, v in report:
            W(f"  {k:32s} {v}")

        W()
        W("THIS SCRIPT CHOOSES NOTHING")
        W("-" * 70)
        W("  It measures and stops. The version is chosen by a person")
        W("  from the tables below.")

        W()
        W("WHY EVERY VERSION IS MEASURED OVER MANY DRAWS")
        W("-" * 70)
        W("  A trial list is 180 trials sampled from a pool of")
        W("  thousands. Two versions can differ because one is a")
        W("  better design or because one drew better, and a single")
        W("  draw cannot tell those apart. Step 5 therefore builds")
        W("  each version at several seeds, and every table here is a")
        W("  mean over draws with the between-draw spread beside it.")
        W()
        W("  THE FIRST RUN OF THIS SCRIPT USED ONE DRAW PER VERSION,")
        W("  and it misled. The fog control read 0.531 for `current`")
        W("  against 0.508 for `larger_diffs` at 0.50 SD, with a")
        W("  Monte Carlo error of 0.011 -- an apparently real cost of")
        W("  the colour cut. It was two different draws of fog trials,")
        W("  because step 5 drew every trial type off one random")
        W("  stream and the cut moved the stream. CHECK 4 below now")
        W("  verifies that cannot happen, and the difference columns")
        W("  use the spread between draws rather than the simulation")
        W("  error.")
        W()
        W("  The simulated people are seeded by DRAW INDEX, not by")
        W("  version. So within an index every version faces the same")
        W("  people, and differences are PAIRED by index -- the")
        W("  participant noise is common to both rates and cancels.")
        W("  The shipping draw is step 5's seed, fixed in advance; it")
        W("  is reported as one draw among the rest and no seed is")
        W("  chosen on any result here.")

        W()
        W("THE ESTIMATOR CHANGED, AND WHY")
        W("-" * 70)
        W("  The per-person fit is now Firth's bias-reduced logistic")
        W("  regression. It used to be plain maximum likelihood with a")
        W("  ridge of " + str(RIDGE) + ", nominally there to keep the")
        W("  fit finite under separation -- a participant whose")
        W("  answers the model predicts perfectly, where no finite")
        W("  maximum-likelihood estimate exists.")
        W()
        W("  That ridge did nothing. The predictors were in raw")
        W("  feature units, so the diagonal it was added to ran to")
        W("  tens of thousands and the ridge was 3.6e-11 of it;")
        W("  refitting at 1e-3 moved the weights by less than 1e-5.")
        W("  The fits were maximum-likelihood in practice and")
        W("  unprotected in principle.")
        W()
        W("  Excluding separable fits was considered and rejected: a")
        W("  participant who answers every colour trial as predicted")
        W("  is the strongest evidence this design can produce, and")
        W("  dropping them would bias against the hypothesis. Firth is")
        W("  always finite, differs from the MLE by about 2% away from")
        W("  separation, and removes most of the small-sample bias --")
        W("  which matters because the registered control comparison")
        W("  reads magnitudes between groups, not just signs.")
        W()
        W("  THE FLOORS USE THE SAME ESTIMATOR. Firth shrinks the")
        W("  guessers' weights as well as the responders', so the two")
        W("  effects largely cancel in a detection rate -- 'largely'")
        W("  being why the whole comparison was re-run rather than")
        W("  argued. information_by_seed.csv carries both floors per")
        W("  draw, under floor_*_SD and floor_*_SD_other.")
        W()
        W("  THE UNITS CHANGED WITH IT. The fit now uses `pred_sat`")
        W("  and `pred_fog` as step 5 writes them, in corpus-SD units,")
        W("  so this script, the registered model and step 8 are the")
        W("  same model. Firth is equivariant under a linear change of")
        W("  predictor scale (CHECK 1e), so no estimate moved because")
        W("  of the unit change.")
        W()
        W("CHECK 4: STREAM SEPARATION")
        W("-" * 70)
        W("  The colour cut filters colour-only pairs and nothing")
        W("  else, so two versions built from the same ladder must")
        W("  have IDENTICAL fog_only, together and conflict trials,")
        W("  at every draw.")
        W()
        W("  This was once false. `current` and `larger_diffs`, which")
        W("  differ only in the cut, shared 1 of 60 fog_only trials,")
        W("  1 of 30 together and 0 of 30 conflict. Two consequences:")
        W("  the fog condition was not a control, and colour")
        W("  information leaked in from types the cut never touches --")
        W("  `both` gained 16.4 units of summed squared colour")
        W("  predictor from its `together` trials alone, about a sixth")
        W("  of its total gain over `current`.")
        W()
        W("  Checked here as well as in step 5 because this is where")
        W("  the numbers are read, and a measurement is only as good")
        W("  as the invariant it assumes.")
        W()
        if streams is None or not len(streams):
            W("  Nothing to judge.")
        else:
            jt = streams[streams.judged]
            if len(jt):
                W(jt[["version_a", "version_b", "type",
                      "draws_compared", "draws_identical",
                      "verdict"]].to_string(index=False))
            W()
            W("  Versions on DIFFERENT ladders are reported in")
            W("  stream_check.csv and not judged: the vivid levels add")
            W("  images, hence new pairs of every type, so their fog")
            W("  trials are a different draw by design. That is why")
            W("  `current` against `both` is read against the")
            W("  between-draw spread and not at face value.")

        W()
        W("1. INFORMATION AND FLOORS")
        W("-" * 70)
        W("  Information is the sum of squared predictors in")
        W("  corpus-SD units -- what the logistic fit has to work")
        W("  with. The floor is the weight that "
          f"{FLOOR_ALPHA:.1%} of pure")
        W("  guessers exceed by luck ON THAT DRAW, so every draw is")
        W("  judged against its own threshold.")
        W()
        W(info.to_string(index=False))
        W()
        if "shipped" in set(info.list):
            a = info[info.list == "shipped"].iloc[0]
            b = info[info.list == BASELINE]
            W("  TRACING THE FLOORS. The shipped list and the")
            W(f"  {BASELINE!r} list use the same ladder, the same pool")
            W("  and the same 60 colour trials. They differ in their")
            W("  MAGNITUDE BINS: the shipped list used rank-based")
            W("  quantile bins, which hold equal numbers of pairs and")
            W("  so concentrate trials where the pairs are -- in the")
            W("  middle. Fixed value edges push trials into the tails,")
            W("  and information is a SUM OF SQUARES, so trials at")
            W("  large differences count disproportionately.")
            if len(b):
                bb = b.iloc[0]
                W()
                W(f"    shipped   colour information {a.SS_colour:7.1f}"
                  f"   floor {a.floor_colour_SD:.3f} SD")
                W(f"    {BASELINE:9s} colour information "
                  f"{bb.SS_colour:7.1f}   floor "
                  f"{bb.floor_colour_SD:.3f} SD")
                W(f"    change    {bb.SS_colour / a.SS_colour - 1:+.0%}"
                  "  with no change to the stimuli or the pool")
                W()
                W("  The `reproduction` row is the same ladder AND the")
                W("  same bins as the shipped list, so it should sit")
                W("  beside `shipped` rather than beside the baseline.")
                W("  That is the check that the binning is the cause.")
            W()
            W("  This is a side effect of the binning decision, not of")
            W("  anything the four versions were designed to vary. The")
            W("  baseline moved, and the preregistration should say so.")

        W()
        W("2. DETECTION")
        W("-" * 70)
        W("  Share of people caught, by their true sensitivity, as a")
        W("  mean over draws. Three spreads are in detection.csv:")
        W("    sd_between  draw to draw, carrying both the design")
        W("                sample and the simulated people")
        W("    sd_mc       the simulated-people part alone,")
        W("                sqrt(p(1-p)/N) -- known, not estimated")
        W("    sd_design   what is left, in quadrature: the design's")
        W("                own draw-to-draw variation, which does NOT")
        W("                shrink by simulating more people")
        for t in PARTICIPANT_TYPES:
            feat = "colour" if t != "fog_only" else "fog"
            sub = det[(det.participant == t) & (det.feature == feat)]
            W()
            W(f"  {t}  ({feat} detection, mean over draws)")
            W(sub.pivot(index="sensitivity_SD", columns="list",
                        values="detect").round(3).to_string())
            W(f"  between-draw SD")
            W(sub.pivot(index="sensitivity_SD", columns="list",
                        values="sd_between").round(3).to_string())

        W()
        W("3. FALSE ALARMS  (independent guessers)")
        W("-" * 70)
        W("  A SECOND set of guessers, seeded apart from the ones that")
        W("  located the floor. Checking the floor against the people")
        W("  who set it would be circular -- they exceed it "
          f"{FLOOR_ALPHA:.1%} of")
        W("  the time by construction, whatever the floor is.")
        W()
        W(fa.to_string(index=False))

        W()
        W(f"4. DIFFERENCE FROM {BASELINE!r}  (paired by draw)")
        W("-" * 70)
        W("  The difference is taken within each draw index and then")
        W("  averaged; its standard error is the spread of those")
        W("  paired differences. `resolvable` is |difference| > 2 SE,")
        W("  which now asks whether a gap is bigger than the")
        W("  variation between draws of the same design -- not merely")
        W("  bigger than the simulation noise.")
        W()
        W("  differences.csv also carries `shipping_draw_difference`")
        W("  and `mc_error_one_draw`, which are what the first run of")
        W("  this script reported, so the two can be compared.")
        if len(diff):
            show = diff[(diff.participant == "colour_only")
                        & (diff.feature == "colour")]
            W()
            W("  colour detection, colour-only participant:")
            W(show.pivot(index="sensitivity_SD", columns="list",
                         values="difference").round(3).to_string())
            W()
            W("  standard error of the paired difference:")
            W(show.pivot(index="sensitivity_SD", columns="list",
                         values="se").round(3).to_string())
            W()
            W(f"  typical SE {show.se.median():.4f}, so differences "
              f"under {2 * show.se.median():.3f} sit")
            W("  inside the variation between draws of one design.")

        W()
        W("4b. THE FOG CONTROL")
        W("-" * 70)
        W("  Within a ladder the fog, together and conflict trials are")
        W("  identical, so a fog-only participant must be detected")
        W("  equally well whatever the colour cut. The fit uses all")
        W("  180 trials, so the colour trials do enter it and a small")
        W("  residual is expected; a large one would mean the cut is")
        W("  costing fog detection, which is the one thing it must")
        W("  not do.")
        W()
        W("  Versions on a DIFFERENT ladder are not a control here --")
        W("  their fog trials are a different draw by design.")
        if len(fc):
            W()
            W(fc.pivot(index="sensitivity_SD", columns="list",
                       values="difference").round(3).to_string())
            same = fc[fc.same_ladder_as_baseline]
            if len(same):
                W()
                W(f"  same ladder as {BASELINE!r}: worst |difference| "
                  f"{same.difference.abs().max():.4f}, typical SE "
                  f"{same.se.median():.4f}")

        W()
        W("5. RANGES AND TRADE-OFFS")
        W("-" * 70)
        W(rng_tab.to_string(index=False))
        W()
        W("  A list can raise colour detection by spreading its trials")
        W("  over a wider range, or by using larger edits. The second")
        W("  is not free: larger colour edits also look more edited.")
        W("  mean_chroma_edit is the average distance of a shown image")
        W("  from the unmanipulated original. Averaged over draws.")

        W()
        W("6. CONFLICT BALANCE")
        W("-" * 70)
        if len(bal):
            W(bal.to_string(index=False))
            W()
            W("  Quantiles are the SHIPPING draw; the last three")
            W("  columns average over every draw. A balance of 1 means")
            W("  a participant with equal weights in the registered")
            W("  model is at 50/50 on that trial.")
            W()
            W("  ON THE FIRST RUN three versions reported identical")
            W("  distributions here and it looked like a property of")
            W("  the design. It was the shared random stream: their")
            W("  conflict draws had consumed the same number of random")
            W("  numbers and landed on the same 30 trials by accident.")
            W("  Within a ladder they are now identical BY")
            W("  CONSTRUCTION, which is the point; across ladders they")
            W("  differ because the pools differ.")
        else:
            W("  No list carries a `balance` column.")

        W()
        W("7. PLAIN HAZE")
        W("-" * 70)
        if opt_info is None or not len(opt_info):
            W("  Not built. These lists need the plain-haze stimulus")
            W("  set, which step 3 builds with FOG_FAMILY = 'haze';")
            W("  step 4 and step 5 then run on it unchanged. Point")
            W("  OPTIONAL_ROOTS at that step-5 root and re-run to")
            W("  include them.")
        else:
            W("  REPORTED SEPARATELY, AND NOT COMPARABLE ROW FOR ROW")
            W("  WITH THE TABLES ABOVE. haze and haze_eq differ in")
            W("  BRIGHTNESS as well as in contrast: plain haze lifts")
            W("  the whole image toward the airlight, so a participant")
            W("  responding to it is partly responding to a tone")
            W("  change that haze_eq removes. Plain haze also fails")
            W("  the pre-committed brightness gate (r = -0.968,")
            W("  VIF 16.03, against haze_eq's r = +0.345, VIF 1.21)")
            W("  and is the only family with envelope violations.")
            W()
            W("  So a detection advantage here is not evidence for")
            W("  plain haze; it may be the brightness cue doing the")
            W("  work. These rows are for reference only.")
            W()
            W(opt_info.to_string(index=False))
            if opt_det is not None and len(opt_det):
                W()
                W("  colour detection, colour-only participant:")
                W(opt_det[(opt_det.participant == "colour_only")
                          & (opt_det.feature == "colour")]
                  .pivot(index="sensitivity_SD", columns="list",
                         values="detect").round(3).to_string())

        W()
        W("8. WHAT TO DECIDE, AND WHAT THIS CANNOT TELL YOU")
        W("-" * 70)
        W("  The choice is between detection and the dose-response")
        W("  range given up to get it. This script measures the first")
        W("  and describes the second; it cannot weigh them for you.")
        W()
        W("  Four things it does NOT measure:")
        W("    - whether the larger edits look artificial, which is a")
        W("      question for the contact sheets, not a simulation;")
        W("    - how a real participant differs from a logistic model")
        W("      with a fixed weight and no lapse rate;")
        W("    - anything about the grid blocks, which use different")
        W("      cells and are step 7's business;")
        W("    - whether a cross-ladder difference is the vivid levels")
        W("      or the different mixed trials they bring with them.")
        W("      Within a ladder that is settled by CHECK 4; across")
        W("      ladders it is not, and the between-draw spread is")
        W("      what keeps it honest.")
        W()
        W("  Once you choose, step 7 freezes that list: rebuild it,")
        W("  checksum trial_list.json and the package, and record the")
        W("  checksum, settings and seed in the documentation.")
        W()
        W("  STILL OPEN, for step 8: simulate_power_v4.py has a stale")
        W("  comment near its predictor construction saying the")
        W("  differences are divided by each scene's own range. They")
        W("  are divided by the corpus SD. The code is right and the")
        W("  comment is not.")
    return path


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("CHOOSE THE COLOUR CONDITION  (v4, step 6)")
    print("=" * 60)
    root = resolve_output_root()
    os.makedirs(root, exist_ok=True)
    print(f"  trials : {TRIALS_ROOT}")
    print(f"  output : {root}")
    print(f"  grid   : {SENSITIVITY_SD} per SD, applied directly")
    print(f"  fit    : {ESTIMATOR}"
          + ("  (Firth; floors from the same estimator)"
             if ESTIMATOR == "firth" else f"  (MLE, ridge {RIDGE})"))
    print(f"  people : {N_DETECT_PEOPLE:,} per cell per draw")
    if MAX_SEEDS is not None:
        print(f"  NOTE   : MAX_SEEDS = {MAX_SEEDS}; this is a capped "
              "run, not the full sweep.")

    report = []
    print("\n" + "=" * 60)
    print("LOADING TRIAL LISTS")
    print("=" * 60)
    by_version = {lab: load_version(TRIALS_ROOT, lab, report)
                  for lab in VERSIONS}
    if REFERENCE_LIST:
        d = load_single(*REFERENCE_LIST, report=report)
        if d:
            by_version[REFERENCE_LIST[0]] = d
    draws = [d for v in by_version.values() for d in v]
    if not draws:
        raise RuntimeError("No trial lists loaded.")
    n_draws = ", ".join(f"{k} {len(v)}" for k, v in by_version.items()
                        if v)
    print(f"\n  {len(draws)} draws in all: {n_draws}")
    report.append(("draws in all", str(len(draws))))

    print()
    check_solvers(draws, report)
    check_predictors(draws, report)
    streams, _ = check_streams(by_version, report, root)

    by_seed, det_by_seed = measure_all(draws)
    info = summarise_information(by_seed)
    det = summarise_detection(det_by_seed)
    fa = summarise_false_alarms(by_seed, report)
    diff = differences(det_by_seed, report)
    fc = fog_control(diff, report)
    rng_tab = ranges_and_tradeoffs(by_version, report)
    bal = balance_distribution(by_version, report)

    # --- plain haze, optional and reported apart ---
    print("\n" + "=" * 60)
    print("7. PLAIN HAZE  (optional)")
    print("=" * 60)
    opt = {}
    for suffix, oroot, labels in OPTIONAL_ROOTS:
        if not os.path.isdir(oroot):
            print(f"  {oroot} not built; skipped")
            continue
        for lab in labels:
            got = load_version(oroot, lab, [], optional=True)
            if got:
                for d in got:
                    d["label"] = f"{lab}_{suffix}"
                opt[f"{lab}_{suffix}"] = got
    opt_info = opt_det = None
    if opt:
        print("  NOT comparable row for row with the tables above: "
              "plain haze changes\n  BRIGHTNESS as well as contrast. "
              "See section 7 of the report.")
        odraws = [d for v in opt.values() for d in v]
        o_by_seed, o_det_by_seed = measure_all(odraws, " (plain haze)")
        opt_info = summarise_information(o_by_seed)
        opt_det = summarise_detection(o_det_by_seed)
        opt_info.to_csv(os.path.join(root, "haze_information.csv"),
                        index=False)
        opt_det.to_csv(os.path.join(root, "haze_detection.csv"),
                       index=False)
    else:
        print("  none built; skipped. See section 7 of the report for "
              "how to build them.")
        report.append(("plain haze", "not built"))

    info.to_csv(os.path.join(root, "information.csv"), index=False)
    by_seed.to_csv(os.path.join(root, "information_by_seed.csv"),
                   index=False)
    det.to_csv(os.path.join(root, "detection.csv"), index=False)
    det_by_seed.to_csv(os.path.join(root, "detection_by_seed.csv"),
                       index=False)
    fa.to_csv(os.path.join(root, "false_alarms.csv"), index=False)
    if len(diff):
        diff.to_csv(os.path.join(root, "differences.csv"), index=False)
    if len(fc):
        fc.to_csv(os.path.join(root, "fog_control.csv"), index=False)
    rng_tab.to_csv(os.path.join(root, "ranges.csv"), index=False)
    if len(bal):
        bal.to_csv(os.path.join(root, "balance.csv"), index=False)
    make_figures(det, info, diff, root)

    path = write_report(root, report, info, det, diff, fc, rng_tab,
                        bal, fa, streams, opt_info, opt_det, n_draws,
                        t0)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in report:
        print(f"  {k:32s} {v}")
    print(f"\n  Report : {path}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print("\n  THIS SCRIPT CHOOSES NOTHING. Read the tables, pick a "
          "version,\n  and step 7 freezes it.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
