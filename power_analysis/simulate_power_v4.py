"""
simulate_power_v4.py  (stimulus pipeline, v4 axes, step 6)

Simulates participants against the actual trial lists and reports what
the experiment could and could not detect, BEFORE anyone is recruited.

Where this sits
---------------
  step 4  analyse_pairs_v4   every usable pair
  step 5  build_trials_v4    the trial lists
  step 6  this one           are those lists powered?
  step 7  check_session_v4   the same analysis, on real data

Kept separate from step 5 because the workflow is a loop. If the answer
here is "underpowered", you change the trial proportions in step 5 and
rebuild; if you want to ask a different design question -- more
participants, a different effect size -- you re-run this without
touching the lists.

TWO TRIAL LISTS
---------------
TRIAL_LISTS is a LIST, one entry per stimulus set, and a comparison
table is written on top. This is the last piece of evidence for the
choice between the two fogs: the brightness-corrected set has wider
magnitude ranges, and the magnitude range is what drives detectability,
so the two sets need not be equally powered.

WHAT DEPENDS ON THE TRIAL LIST, AND WHAT DOES NOT
-------------------------------------------------
Two of the four analyses depend on the trials; two do not. v2 computed
all four inside the per-list run, which made the last two look like
results about the trial list when they are not.

  DEPENDS on the trials
    recovery       fits simulated participants to the real predictors,
                   so the spread of recovered weights -- and therefore
                   the smallest detectable individual weight -- is a
                   property of THIS list
    dose-response  the expected curve across THIS list's magnitude bins

  DOES NOT depend on the trials
    power curve    a one-sample t-test on per-participant scores, as a
                   function of N and Cohen's d. Once each person's
                   score exists, the test knows nothing about how many
                   trials produced it
    Bayes factors  likewise: a function of t and N only

So the last two are computed ONCE and shared. They describe the TEST,
not the stimuli, and they are identical for both sets by construction.
This also removes most of the runtime, since the Bayes factor integrals
are the slow part.

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

Usage
-----
    conda activate drl2drv
    python simulate_power_v4.py
"""

import os
import time
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
# TRIAL_LISTS
#   (label, trial_list.csv) per stimulus set. The label names the
#   output subdirectory and the column in the comparison table.
#   The third entry is that set's all_pairs.csv from step 4. It is
#   needed ONLY by the trial-count sweep below, which has to build
#   trial lists at sizes other than the one step 5 produced. Set it to
#   None to skip the sweep for that set.
TRIAL_LISTS = [
    ("haze_eq", "data/trials_v4_ec/haze_eq/trial_list.csv",
     "data/pair_analysis_v4_ec/haze_eq/all_pairs.csv"),
    ("haze",    "data/trials_v4_ec/haze/trial_list.csv",
     "data/pair_analysis_v4_ec/haze/all_pairs.csv"),
]

OUTPUT_ROOT = "data/power_v4_ec"

# ON_EXISTING
#   "new" writes to OUTPUT_ROOT_2, _3, ... rather than overwriting.
#   "error" stops instead.
ON_EXISTING = "new"

# --- Predictors ---
#
# Standardised differences: each raw difference divided by that scene's
# own range for the feature. Using the standardised versions means
# b_sat and b_fog are on the same scale, so "tracks colour" and "tracks
# fog" are comparable claims -- and the numbers are comparable with
# v2's, since the scene-range division cancels the units.
PRED_SAT = "z_sat"
PRED_FOG = "z_fog"

# --- Scenarios ---
#
# SCENARIOS
#   True (b_sat, b_fog) for the synthetic participants, in standardised
#   units. A weight of 1.0 means a one-SD difference in that feature
#   shifts the log-odds of choosing by 1, i.e. from 50% to about 73%.
#   "tracks neither" is the null case and is the one to watch, since it
#   is the likely outcome.
SCENARIOS = {
    "tracks colour":  (1.0, 0.0),
    "tracks fog":     (0.0, 1.0),
    "tracks both":    (0.7, 0.7),
    "tracks neither": (0.0, 0.0),
}

# N_SIM_PARTICIPANTS
#   Simulated participants per scenario for the recovery analysis.
#   Higher is smoother; 500 is plenty.
N_SIM_PARTICIPANTS = 500

# --- Group-level power (shared across sets) ---
#
# SAMPLE_SIZES
#   Sample sizes to evaluate. 10 is the realistic pilot; the larger
#   values show what recruiting more would buy.
SAMPLE_SIZES = [8, 10, 15, 20, 30, 40]

# EFFECT_SIZES
#   Between-participant Cohen's d for the proportion test. Suzuki et
#   al. (2024) reported d = 3.80, but their manipulation was DeepDream
#   versus photorealistic -- unmistakable. A photometric manipulation
#   should be far smaller, and Kusztor et al.'s decisive nulls point
#   the same way, so the low end is the realistic range.
EFFECT_SIZES = [0.3, 0.5, 0.8, 1.0, 1.5]

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

# BF_THRESHOLD
#   Conventional threshold for "moderate evidence". BF10 > 3 is
#   evidence for an effect; BF01 > 3 (i.e. BF10 < 1/3) is evidence
#   against one.
BF_THRESHOLD = 3.0

# --- Dose-response ---
DOSE_EFFECT = (0.7, 0.7)   # weights used for the expected curve
N_DOSE_REPS = 200

# --- The floors ---
#
# FLOOR_ALPHA
#   One-sided false-alarm rate for calling ONE person a responder on
#   ONE feature. The floor is the weight that this share of pure
#   guessers exceeds by chance.
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
#   "one" counts a person as detected when their weight exceeds the
#   positive floor only. The hypothesis is directional -- people are
#   expected to prefer the MORE manipulated image -- so a weight far
#   below zero is not a detection. This also matches the 2.5%
#   false-alarm rate of the per-person responder test.
DETECTION_SIDED = "one"

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
CONTRAST_BASELINE = 0.7

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

# --- Trial-count sweep ---
#
# RUN_TRIAL_COUNT_SWEEP
#   Rebuilds the trial list at several sizes and reports what each size
#   buys. This is the one question the rest of the script cannot
#   answer, because it analyses a single fixed list.
RUN_TRIAL_COUNT_SWEEP = True

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
#   Must match steps 4 and 5: the sweep rebuilds trial lists the same
#   way step 5 does, spread evenly across this many bins of difference
#   size.
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


def simulate_many(X, W, rng):
    """Choices of many people at once. W is (n_people, 2) true weights."""
    p = 1.0 / (1.0 + np.exp(-(W @ X.T)))
    return (rng.random(p.shape) < p).astype(float)


def jzs_bayes_factor(t, n, scale=BF_SCALE):
    """JZS Bayes factor for a one-sample t-test (Rouder et al., 2009).

    Returns BF10: how much more likely the data is under an effect than
    under the null. BF10 = 0.25 means the data is four times more
    likely under the null, usually written BF01 = 4.

    Integrates the marginal likelihood over a Cauchy prior on the
    effect size, which is what "JZS" refers to.
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


# ============================================================
# Loading
# ============================================================

def load_trials(path, label):
    """The analysed trials and the two oriented predictors."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Trial list not found: {path}\nRun step 5 "
            "(build_trials_v4.py) first.")
    trials = pd.read_csv(path)
    for c in ("d_sat", "d_fog", PRED_SAT, PRED_FOG):
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

    # Signed standardised predictors, oriented so that a POSITIVE
    # weight always means "prefers the more manipulated image".
    #
    # For fog that is the foggier image, so the sign of d_fog is used
    # as-is. For saturation it is the LESS saturated image -- the
    # drained one, the one that scores higher on the lifeless axis --
    # so the sign of d_sat is INVERTED. Without this flip a positive
    # b_sat would mean "prefers vivid colour", which is the opposite of
    # the hypothesis and makes the dose-response curve run downward.
    #
    # z_sat and z_fog are stored as magnitudes by step 4, so the sign
    # comes from the raw differences.
    X = np.column_stack([
        -np.sign(trials.d_sat) * trials[PRED_SAT].abs(),
        np.sign(trials.d_fog) * trials[PRED_FOG].abs(),
    ]).astype(float)
    r = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    print(f"  [{label}] {len(trials)} analysed trials, "
          f"r(predictors) = {r:+.3f}, VIF = {1 / (1 - r ** 2):.2f}")
    return trials, X, r


# ============================================================
# The floors  (depend on the trial list)
# ============================================================

def compute_floors(X, label, report):
    """Per-feature floors, as PERCENTILES of the guessers' estimates.

    A floor is the weight a person has to beat before they can be told
    apart from someone responding at random. Simulate people who track
    nothing, fit them, and take the weight that FLOOR_ALPHA of them
    exceed by luck.

    Taken as a percentile of the simulated null rather than as
    1.96 * SD, which is what earlier versions used. The two agree only
    if the fitted weights are normally distributed, and the colour
    weight is not: its null distribution has a longer upper tail, so
    1.96 * SD came out about 11% BELOW the real 97.5th percentile and
    the test would have run at roughly 4% false alarms instead of 2.5%.
    The percentile needs no distributional assumption and is right
    either way. It also matches step 7, which must use the same floors
    or its responder counts describe a different test.
    """
    rng = np.random.default_rng(RANDOM_SEED + 101)
    W = np.zeros((N_FLOOR_PEOPLE, 2))
    est = fit_logit_batch(X, simulate_many(X, W, rng))
    floors = np.percentile(est, 100 * (1 - FLOOR_ALPHA), axis=0)
    sd = est.std(axis=0)
    print("\n" + "=" * 60)
    print(f"FLOORS  [{label}]")
    print("=" * 60)
    print(f"  {N_FLOOR_PEOPLE:,} simulated guessers, "
          f"{FLOOR_ALPHA:.1%} false alarms per person per feature")
    print(f"  floor   : colour {floors[0]:.3f}, fog {floors[1]:.3f}")
    print(f"  1.96*SD : colour {1.96 * sd[0]:.3f}, "
          f"fog {1.96 * sd[1]:.3f}   <- the normal approximation")
    for k, feat in enumerate(["colour", "fog"]):
        gap = floors[k] / (1.96 * sd[k]) - 1.0
        if abs(gap) > 0.05:
            print(f"    {feat}: the percentile sits {gap:+.0%} from the "
                  "normal approximation, so that")
            print("      distribution is noticeably skewed; the "
                  "percentile is the one to trust.")
    report.append(("floor, colour / fog",
                   f"{floors[0]:.3f} / {floors[1]:.3f}"))
    return floors


# ============================================================
# Recovery  (depends on the trial list)
# ============================================================

def run_recovery(X, floors, label, out_dir, report):
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
    mde = float(np.mean(floors))
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

def run_detection(X, floors, label, out_dir, report):
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
    print(f"  floor ({FLOOR_ALPHA:.1%} false alarms): "
          f"colour {floors[0]:.2f}, fog {floors[1]:.2f}")

    rng = np.random.default_rng(RANDOM_SEED + 5)
    rows = []
    for w in DETECTION_GRID:
        out = {"true_weight": w}
        for f_idx, feat in enumerate(["colour", "fog"]):
            W = np.zeros((N_DETECT_PEOPLE, 2))
            W[:, f_idx] = w
            est = fit_logit_batch(X, simulate_many(X, W, rng))
            if DETECTION_SIDED == "one":
                hit = est[:, f_idx] > floors[f_idx]
            else:
                hit = np.abs(est[:, f_idx]) > floors[f_idx]
            out[feat] = float(hit.mean())
        rows.append(out)
    det = pd.DataFrame(rows)
    det.to_csv(os.path.join(out_dir, "detection_curve.csv"),
               index=False)

    print("\n  share of people caught, by their true weight:")
    print(det.round(3).to_string(index=False))

    # Where does each curve cross the target?
    needed = {}
    for feat in ["colour", "fog"]:
        y = det[feat].to_numpy()
        x = det.true_weight.to_numpy()
        if y.max() < DETECTION_TARGET:
            needed[feat] = np.nan
            print(f"  {feat}: never reaches {DETECTION_TARGET:.0%} "
                  "within the grid; widen DETECTION_GRID.")
            continue
        # np.interp needs an increasing x, and the curve rises with w
        needed[feat] = float(np.interp(DETECTION_TARGET, y, x))

    print(f"\n  weight needed for {DETECTION_TARGET:.0%} detection:")
    for feat in ["colour", "fog"]:
        v = needed[feat]
        if np.isfinite(v):
            half = 1.0 / (1.0 + np.exp(-v * 0.5))
            full = 1.0 / (1.0 + np.exp(-v))
            print(f"    {feat:7s} {v:.2f}   = choosing the manipulated "
                  f"image {half:.0%} of the time at a typical "
                  f"difference, {full:.0%} at the largest")
    report.append((f"weight for {DETECTION_TARGET:.0%} detection, "
                   "colour / fog",
                   f"{needed['colour']:.2f} / {needed['fog']:.2f}"))

    _plot_detection(det, floors, needed, label, out_dir)
    return det, needed


def _plot_detection(det, floors, needed, label, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for k, feat in enumerate(["colour", "fog"]):
        ax.plot(det.true_weight, det[feat], marker="o", color=f"C{k}",
                label=f"{feat} (floor {floors[k]:.2f})")
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
    """How big must a crossover, or a departure from additivity, be?

    Both are calibrated the same way: simulate the null, take its
    central interval as the critical region, then sweep the true effect
    and record how often it lands outside.
    """
    print("\n" + "=" * 60)
    print(f"CROSSOVER AND ADDITIVITY  [{label}]")
    print("=" * 60)
    B = CONTRAST_BASELINE
    X3 = design3(X)

    # The design must survive the side swap, or the third column means
    # nothing. Cheap to check, so check it rather than trust it.
    if not np.allclose(design3(-X), -X3):
        raise AssertionError(
            "design3 is not antisymmetric under the side swap; the "
            "additivity term would be unestimable.")
    c3 = np.corrcoef(X3.T)
    vif3 = np.diag(np.linalg.inv(c3))
    print(f"  baseline weight on both cues: {B}")
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

    _plot_contrasts(cross, addi, gap_needed, gamma_needed, label, out_dir)
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
    for rep in range(N_DOSE_REPS):
        y = simulate_responses(X, DOSE_EFFECT, rng)
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
            rows.append({"rep": rep, "trial_type": t, "mag_bin": b,
                         "prop": float(chose.mean()), "n": len(g)})

    dose = (pd.DataFrame(rows)
            .groupby(["trial_type", "mag_bin"])
            .agg(prop=("prop", "mean"), sd=("prop", "std"),
                 n_trials=("n", "first")).reset_index())
    dose.to_csv(os.path.join(out_dir, "dose_response.csv"), index=False)
    print(f"  expected with true weights {DOSE_EFFECT}:")
    print(dose.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for t, g in dose.groupby("trial_type"):
        ax.errorbar(g.mag_bin, g.prop, yerr=g.sd, marker="o", capsize=3,
                    label=t)
    ax.axhline(0.5, color="k", lw=0.8, ls="--", label="chance")
    ax.set_xlabel("magnitude bin (1 = smallest difference)")
    ax.set_ylabel("P(chose the more manipulated image)")
    ax.set_title(f"Expected dose-response, true weights {DOSE_EFFECT} "
                 f" [{label}]\n(a flat line above chance would indicate "
                 "bias, not sensitivity)")
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
    report.append(("power at N=10, d=0.5", f"{at10.get(0.5, np.nan):.2f}"))
    report.append(("power at N=10, d=1.0", f"{at10.get(1.0, np.nan):.2f}"))

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
    print("BAYES FACTORS  (shared by both sets)")
    print("=" * 60)

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

    floors = compute_floors(X, label, report)
    rec, clouds, acc, sd0, mde = run_recovery(X, floors, label, out_dir,
                                              report)
    det, needed = run_detection(X, floors, label, out_dir, report)
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
            "det": det, "floors": floors, "needed": needed,
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
        put("floor, colour", L, round(float(res["floors"][0]), 3))
        put("floor, fog", L, round(float(res["floors"][1]), 3))
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

def write_report(results, comp, pw, bfs, shared_report, root, t0):
    path = os.path.join(root, "power_report.txt")
    with open(path, "w") as f:
        def W(s=""):
            f.write(s + "\n")

        W("POWER SIMULATION REPORT  (v4 axes, step 6)")
        W("=" * 70)
        for r in results:
            W(f"{r['label']:10s} {r['path']}  "
              f"({len(r['trials'])} analysed trials)")
        W(f"Prior      : JZS, scale {BF_SCALE}")
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
                    W(f"    {feat:7s} floor {r['floors'][k]:.2f}, "
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
                  f"{CONTRAST_BASELINE} on both cues")
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
        W(f"{n}. GROUP-LEVEL POWER  (both sets)")
        W("-" * 70)
        for k, v in shared_report:
            W(f"  {k:34s} {v}")
        W()
        W("  Power (one-sample t-test):")
        W("  " + pw.pivot(index="n", columns="d", values="power")
          .round(2).to_string().replace("\n", "\n  "))
        W()
        W(f"{n + 1}. BAYES FACTORS UNDER A TRUE NULL  (both sets)")
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
        W(f"{n + 2}. WHAT TO DO WITH THIS")
        W("-" * 70)
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

def main():
    t0 = time.time()
    print("=" * 60)
    print("SIMULATE POWER  (v4, step 6)")
    print("=" * 60)
    root = resolve_output_root()
    os.makedirs(root, exist_ok=True)
    print(f"  sets   : {', '.join(l for l, _, _ in TRIAL_LISTS)}")
    print(f"  output : {root}")

    results = [analyse_one(label, path, pairs, root)
               for label, path, pairs in TRIAL_LISTS]

    shared_report = []
    pw = run_power(root, shared_report)
    bfs = run_bayes(root, shared_report)

    comp = build_comparison(results, root)
    if len(results) > 1:
        comparison_figure(results, root)
        print("\n" + "=" * 60)
        print("COMPARISON  (only these depend on the trial list)")
        print("=" * 60)
        print(comp.to_string())

    path = write_report(results, comp, pw, bfs, shared_report, root, t0)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['label']:10s} min detectable weight "
              f"{r['mde']:.2f}   classification {r['acc']:.1%}   "
              f"dose rise {r['mean_rise']:+.3f}")
    for k, v in shared_report:
        print(f"  {k:34s} {v}")
    print(f"\n  Report : {path}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
