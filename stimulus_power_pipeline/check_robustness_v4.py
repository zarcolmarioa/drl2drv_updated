"""
check_robustness_v4.py  (stimulus pipeline, v4 axes, step 6b)

Two checks on the candidate trial lists before one of them is frozen:
does the ranking survive response lapses, and are the fitted weights
stable enough to compare between groups?

Where this sits
---------------
  step 5   build_trials_v4_corrected        the candidate trial lists
  step 6   choose_colour_condition_v4       which candidate detects best
  step 6b  this one                         does that survive lapses,
                                            and are the weights usable?
  step 7   freeze the chosen list, build the grid blocks
  step 8   simulate_power_v4                is the frozen list powered?

Step 6 answered "which list catches a participant with a given
sensitivity most often". It assumed a participant who never slips, and
it only ever asked whether a fitted weight cleared a floor -- never
whether the weight itself was any good. Both assumptions are load
bearing for different reasons, and this script tests them.

CHECK A: LAPSES
---------------
A real participant blinks, mis-clicks, and stops attending for a trial
or two. The standard way to model that is a LAPSE RATE: with
probability lambda the response is a coin flip regardless of the
images, and otherwise it follows the logistic model.

    p(choose A) = (1 - lambda) * logistic(X beta) + lambda / 2

Lapses pull every response toward 0.5, so they shrink the fitted
weights and lower detection everywhere. The question is not whether
detection falls -- it must -- but whether the ORDERING of the
candidates and the SIZE of their advantage over the baseline survive.
A design that wins only for a perfect responder is not a design worth
registering.

A property worth stating, because it is why the floors are not
recomputed per lapse rate: THE NULL IS UNCHANGED. A pure guesser
already responds at 0.5, and mixing 0.5 with 0.5 is still 0.5, so the
floor -- a percentile of guessers' fitted weights -- does not depend on
lambda. CHECK 2 verifies that empirically rather than taking it on
faith.

CHECK B: WEIGHT STABILITY
-------------------------
Detection only asks whether a fitted weight cleared a floor. The
registered control comparison asks something harder: it compares
weights BETWEEN GROUPS, so the magnitude has to mean something, not
just its sign.

The worry is specific. The cut versions show mostly large colour
differences, and a sensitive participant facing large differences
answers nearly every colour trial in the predicted direction. In the
limit the data are linearly separable, the maximum-likelihood weight is
infinite, and what the fit returns is set by the penalty rather than by
the participant. The ridge is 1e-6, which is almost no penalty at all,
so such a fit returns a very large number. A handful of those in a
group inflates its mean and its variance, and a between-group
difference computed from them is not measuring sensitivity.

So for every cell this reports:

  share_separated        fits whose own weights classify every single
                         response correctly -- the data are separable
                         and there is no finite MLE
  share_colour_perfect   people who answered EVERY colour-informative
                         trial in the predicted direction. Model-free,
                         and the direct form of "approaching perfect
                         separation"
  share_extreme          fits whose colour weight exceeds
                         EXTREME_WEIGHT_SD in per-SD units
  share_unconverged      fits whose last Newton step was still larger
                         than CONVERGE_TOL at the iteration cap
  mean / sd / p50 / p90  the fitted colour weight against its TRUE
  bias, sd_over_true     value, in per-SD units. sd_over_true is the
                         noise-to-signal ratio, and it is what a
                         between-group comparison has to overcome
  sd_trimmed             the same spread with the extreme fits removed,
                         so it is visible how much of the spread is
                         them

AND WHAT THE CHANGE OF ESTIMATOR DID. The per-person fit is now
Firth's bias-reduced logistic regression, which is also the registered
model. Every cell is fitted twice on the SAME responses -- once with
Firth, once with the plain maximum-likelihood estimator the earlier
version of this script used -- and the comparison reports, per cell,
how far the weights move, how far the floors move, and whether the
detection rates change. Floors are compared estimator against its own
floor: Firth weights judged against an MLE floor would measure an
offset rather than a difference.

That replaces the ridge diagnostic this script used to carry, which
refitted every cell at a larger ridge to show the penalty was doing
nothing. It was: in raw feature units the diagonal of X'WX ran to tens
of thousands and 1e-6 sat eleven orders of magnitude below it. So the
fits were maximum-likelihood in practice and unprotected in principle
-- a participant the model predicted perfectly had no finite estimate
at all -- which is the reason the estimator changed. Under Firth the
penalty is not inert, so the question worth reporting is no longer
"is the penalty doing anything" but "what did changing the estimator
do". This is still a MEASUREMENT: the estimator was chosen on the
argument, not on these numbers.

HOW THE SIMULATION IS PAIRED
----------------------------
One set of uniform random numbers per (draw index, sensitivity,
participant type), REUSED across every list and every lapse rate. So
the same simulated person, with the same luck, meets all four lists at
all three lapse rates. Differences between lists and between lapse
rates are then as close to a within-person comparison as a simulation
can be, and the standard error of a difference comes from the spread
over draws rather than from the simulation noise.

Nothing here chooses a seed or a version. It reports, and the decision
rule was fixed in advance: `both` ships if the ranking and the gains
hold and the weights are stable; otherwise sens_cut_25_5.

Inputs
------
  TRIALS_ROOT  step 5's output root, the same one step 6 read
  LISTS        the baseline and the candidates under test

Outputs (under OUTPUT_ROOT)
---------------------------
  detection_by_lapse.csv   list x sensitivity x type x lapse, mean over
                           draws with the between-draw spread
  differences_by_lapse.csv each candidate against the baseline, paired
                           by draw, at each lapse rate
  ranking.csv              the candidate order at every cell, and
                           whether adjacent gaps are resolvable
  weights.csv              the stability diagnostics above
  by_seed.csv              every draw, for anything that needs a
                           distribution rather than a summary
  detection_by_lapse.png, weights.png
  robustness_report.txt

Usage
-----
    conda activate drl2drv
    python check_robustness_v4.py

CPU only. About 8s per draw, so roughly ten minutes for the full
sweep. Set MAX_SEEDS = 3 for a trial run first.
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
#   Step 5's output root -- THE SAME ONE STEP 6 READ. Step 5 writes to
#   a new numbered directory on every run, so this is the setting to
#   check before anything else: measuring robustness on a different
#   build from the one step 6 ranked would compare two unrelated
#   things.
TRIALS_ROOT = "data/trials_v4_ext"

# LISTS
#   The baseline first, then the candidates under test. Only the three
#   candidates still in contention after step 6 are here --
#   sens_cut_35_3 was dominated (it bought +0.004 over `both`, inside
#   the noise, for a narrower range) and `larger_diffs` and the two
#   one-draw reference lists are not candidates.
LISTS = ["current", "extra_levels", "sens_cut_25_5", "both"]

BASELINE = "current"

# CANDIDATES
#   The three being ranked, in the order step 6 put them. CHECK A asks
#   whether this order survives every lapse rate.
CANDIDATES = ["extra_levels", "sens_cut_25_5", "both"]

OUTPUT_ROOT = "data/robustness_v4"

# ON_EXISTING
#   "new" writes to OUTPUT_ROOT_2, _3, ... rather than overwriting.
ON_EXISTING = "new"

# MAX_SEEDS
#   Draws per list, shipping draw first. None uses every draw step 5
#   built. Set 3 for a trial run; the report states what was used, so a
#   capped run cannot be mistaken for a full one.
MAX_SEEDS = None

# --- The lapse rates ---
#
# LAPSE_RATES
#   Share of trials answered at random regardless of the images:
#
#     p(choose A) = (1 - lambda) * logistic(X beta) + lambda / 2
#
#   0.0 reproduces step 6 and is the reference. 0.1 and 0.2 are the
#   rates asked for. 0.2 is severe for a 207-trial online session with
#   6 catch trials -- a participant lapsing on a fifth of trials should
#   be failing the catch trials and be excluded -- so it is a stress
#   test rather than an expectation.
LAPSE_RATES = [0.0, 0.1, 0.2]

# --- The sensitivity grid (as in step 6) ---
SENSITIVITY_SD = [0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00]

PARTICIPANT_TYPES = {
    "equal":       (1.0, 1.0),
    "colour_only": (1.0, 0.0),
    "fog_only":    (0.0, 1.0),
}

# --- The predictors ---
#
# The corpus SDs, from step 1's corpus_feature_stats_v4.csv
# (stimulus_path population, n = 20,000). The constants step 4, step 5
# and step 6 use.
SD_SAT = 39.182
SD_FOG = 9.348

# Orientation: a predictor is positive when image A is the one the
# hypothesis says will be chosen -- the LESS colourful image, the
# FOGGIER image. d_sat is negated; d_fog already carries step 4's
# FOG_SIGN. See step 6's settings for the full note.

# --- Floors and simulation sizes ---
FLOOR_ALPHA = 0.025
N_FLOOR_PEOPLE = 20_000
N_DETECT_PEOPLE = 4_000

# --- The fit, and what counts as unstable ---
#
# RIDGE
#   The penalty step 6 and step 8 use. Small enough to be almost no
#   penalty, which is the point of CHECK B.
RIDGE = 1e-6

# ESTIMATOR / COMPARE_MLE
#   The per-person fit is now Firth's bias-reduced logistic regression,
#   as is the registered model. COMPARE_MLE refits the SAME responses
#   with the plain maximum-likelihood estimator the earlier version
#   used, and reports per cell how far the weights move, how far the
#   floors move, and whether detection changes.
#
#   That comparison replaces the ridge diagnostic this script used to
#   carry. The old one refitted at a larger ridge to show the penalty
#   was doing nothing, which it was: the predictors were in raw feature
#   units, the diagonal of X'WX ran to tens of thousands, and 1e-6 was
#   3.6e-11 of it. The conclusion was that the fits were
#   maximum-likelihood in practice and unprotected in principle, which
#   is why the estimator changed. With Firth the penalty is no longer
#   inert -- that is the point of it -- so the question worth reporting
#   is no longer "is the penalty doing anything" but "what did changing
#   the estimator do".
#
#   None skips the second fit and roughly halves the runtime.
ESTIMATOR = "firth"
COMPARE_MLE = True

# FIRTH_ITER / FIRTH_TOL / FIRTH_MAX_STEP
#   The Newton loop for Firth. FIRTH_MAX_STEP damps each step: from
#   zero on near-separable data an undamped first step can overshoot
#   far enough to stall. 5.0 is in per-SD units, far above any
#   plausible weight.
FIRTH_ITER = 200
FIRTH_TOL = 1e-10
FIRTH_MAX_STEP = 5.0

# EXTREME_WEIGHT_SD
#   A fitted colour weight above this, in per-SD units, is reported as
#   extreme. The true weights run 0.30 to 1.00, so 5.0 is five to
#   sixteen times the truth: not a noisy estimate but a fit that has
#   run away.
EXTREME_WEIGHT_SD = 5.0

# CONVERGE_TOL / ITER_CAP
#   A fit is reported as unconverged if its last Newton step still
#   exceeded CONVERGE_TOL after ITER_CAP iterations. The verbatim
#   solver stops when the LARGEST step over all people is small, so one
#   straggler is otherwise invisible; the traced solver keeps a step
#   size per person.
CONVERGE_TOL = 1e-6
ITER_CAP = 60

# COLOUR_INFORMATIVE_SD
#   A trial counts as colour-informative when |pred_sat| is at least
#   this, in corpus-SD units. Used only by `share_colour_perfect`: a
#   fog-only trial carries a residual colour difference near zero, and
#   counting those would make "answered every colour trial as predicted"
#   mean "answered every trial as predicted".
COLOUR_INFORMATIVE_SD = 0.20

RANDOM_SEED = 42

# Seed offsets. The detection offset seeds the uniforms that are SHARED
# across lists and lapse rates -- see HOW THE SIMULATION IS PAIRED.
SEED_FLOOR, SEED_DETECT, SEED_LAPSE_CHECK = 101, 303, 404

# --- Checks ---
FIT_CHECK_TOL = 1e-8

# GRAD_CHECK_TOL / EQUIVAR_CHECK_TOL
#   CHECK 1b differentiates log L + 0.5 log det(X'WX) numerically at
#   the estimate and requires the gradient to vanish -- the check that
#   distinguishes Firth from something Firth-shaped. The tolerance is
#   loose because it is a finite difference of a log-likelihood near
#   its maximum; an unpenalised fit misses by order 1.
#
#   CHECK 1e fits in raw units, rescales, and requires the per-SD fit
#   back. Firth is equivariant under a linear change of predictor
#   scale, so this is exact up to arithmetic.
GRAD_CHECK_TOL = 1e-4
EQUIVAR_CHECK_TOL = 1e-6

# MATERIAL_SHIFT
#   How large a change in a detection rate counts as material when the
#   two estimators are compared. 0.01 is one percentage point, which
#   is below the resolution of any sample-size decision this feeds and
#   well above the one-person granularity of a cell. With 4,000
#   simulated people a shift of a single person is statistically
#   detectable and means nothing, which is why this threshold exists
#   alongside the paired standard error rather than instead of it.
MATERIAL_SHIFT = 0.01

# N_FLOOR_SETS
#   How many independent guesser sets CHECK 2b builds to measure the
#   floor's own Monte Carlo spread.
#
#   The earlier version of this check compared ONE extra guesser set
#   against the reference and called the difference "the floor's own
#   error". That is a single draw from the distribution of such
#   differences, not its spread, and it understated the spread by
#   about threefold: on `current` one draw gave 0.0023 where the
#   actual SD of a difference is 0.0076. A hand-picked floor of 0.02
#   was carrying the verdict.
#
#   12 sets give the set-to-set SD to about 20% relative, which is
#   enough to judge a shift against twice it. At 0.8 s per floor fit
#   this costs about ten seconds.
N_FLOOR_SETS = 12

TYPES = ["sat_only", "fog_only", "together", "conflict"]


# ============================================================
# Helpers
# ============================================================

def resolve_output_root():
    """OUTPUT_ROOT, never overwriting an existing run."""
    def occupied(d):
        return os.path.isdir(d) and os.path.isfile(
            os.path.join(d, "robustness_report.txt"))

    if not occupied(OUTPUT_ROOT):
        return OUTPUT_ROOT
    if ON_EXISTING == "error":
        raise FileExistsError(
            f"{OUTPUT_ROOT} already holds a run. Move it, or set "
            "ON_EXISTING = 'new'.")
    n = 2
    while occupied(f"{OUTPUT_ROOT}_{n}"):
        n += 1
    print(f"  NOTE: {OUTPUT_ROOT} already holds a run; writing to "
          f"{OUTPUT_ROOT}_{n} instead.")
    return f"{OUTPUT_ROOT}_{n}"


def draw_paths(root, label):
    """Every draw of one version, shipping draw first."""
    out = []
    ship = os.path.join(root, label, "trial_list.csv")
    if os.path.isfile(ship):
        out.append((None, ship))
    reps = []
    for p in glob.glob(os.path.join(root, "seeds", label, "seed_*",
                                    "trial_list.csv")):
        m = re.search(r"seed_(\d+)", p)
        if m:
            reps.append((int(m.group(1)), p))
    out += sorted(reps)
    if MAX_SEEDS is not None:
        out = out[:MAX_SEEDS]
    return out


# ============================================================
# Model
# ============================================================
#
# simulate_responses, fit_logit, fit_logit_batch and simulate_many are
# VERBATIM from simulate_power_v4.py, as step 6 keeps them: a script in
# this pipeline is standalone and runs from its own settings block.
# fit_firth and fit_firth_batch below are likewise verbatim from step
# 6, where they were written and validated.
#
# The maximum-likelihood solvers are no longer the registered
# estimator. They are kept because COMPARE_MLE refits every cell with
# them, which is how this script answers what changing the estimator
# did; fit_logit_batch itself is unused and kept only so the verbatim
# block stays verbatim. CHECK 1 holds every solver here to agreement
# with its own single-fit form.

def simulate_responses(X, beta, rng):
    """One synthetic participant's choices."""
    p = 1.0 / (1.0 + np.exp(-(X @ np.asarray(beta, dtype=float))))
    return (rng.random(len(X)) < p).astype(float)


def fit_logit(X, y, n_iter=60, ridge=1e-6):
    """Newton-Raphson logistic regression, no intercept."""
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
    """fit_logit for many simulated people at once."""
    P, k = Y.shape[0], X.shape[1]
    b = np.zeros((P, k))
    eye = ridge * np.eye(k)
    for _ in range(n_iter):
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
    """Choices of many people at once. W is (n_people, 2) weights."""
    p = 1.0 / (1.0 + np.exp(-(W @ X.T)))
    return (rng.random(p.shape) < p).astype(float)


# --- added here, not in simulate_power_v4.py ---

def fit_logit_batch_traced(X, Y, n_iter=ITER_CAP, ridge=RIDGE):
    """fit_logit_batch, plus a last step size PER PERSON.

    Identical arithmetic to the verbatim solver, with one difference
    that matters for CHECK B: the verbatim loop stops when the largest
    step over ALL people is small, which is the right stopping rule for
    speed but hides a single participant still moving. This keeps each
    person's own last step, so a fit that never settled can be counted
    instead of being averaged in silently.

    Returns (b, last_step_per_person, iterations_used).
    """
    P, k = Y.shape[0], X.shape[1]
    b = np.zeros((P, k))
    eye = ridge * np.eye(k)
    last = np.full(P, np.inf)
    used = 0
    for it in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(b @ X.T, -500.0, 500.0)))
        w = np.clip(p * (1 - p), 1e-9, None)
        H = np.einsum("pt,ti,tj->pij", w, X, X) + eye
        g = (Y - p) @ X
        step = np.linalg.solve(H, g[..., None])[..., 0]
        b = b + step
        last = np.max(np.abs(step), axis=1)
        used = it + 1
        if np.max(last) < 1e-9:
            break
    return b, last, used


def fit_firth(X, y, n_iter=FIRTH_ITER, tol=FIRTH_TOL,
              max_step=FIRTH_MAX_STEP):
    """Firth's bias-reduced logistic regression, one participant.

    VERBATIM from step 6, which is where it was written and validated.
    The score equation gains a term from the Jeffreys-prior penalty
    0.5 * log det(X' W X):

        U*(b)_j = sum_i [ y_i - p_i + h_i (0.5 - p_i) ] x_ij

    with h_i the i-th hat-matrix diagonal. Always finite, so a
    participant whose answers the model predicts perfectly still has an
    estimable weight instead of being excluded -- which is the reason
    the estimator changed. FINITE IS NOT MODEST: on this design such a
    participant fits at about 9 per SD if every colour-informative
    trial goes as predicted and about 41 if every trial does. CHECK B
    below measures how close anyone comes.
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
    20,000 guessers per floor per list per draw. CHECK 1a holds it to
    agreement with fit_firth.

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

    VERBATIM from step 6. Used only by CHECK 1b.
    """
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -500.0, 500.0)))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    w = p * (1 - p)
    sign, logdet = np.linalg.slogdet((X * w[:, None]).T @ X)
    return -np.inf if sign <= 0 else ll + 0.5 * logdet


def fit_main(X, Y):
    """The registered estimator. Returns (weights, last step)."""
    if ESTIMATOR == "firth":
        return fit_firth_batch(X, Y)
    if ESTIMATOR == "mle":
        return fit_logit_batch_traced(X, Y, ridge=RIDGE)[:2]
    raise ValueError(
        f"ESTIMATOR must be 'firth' or 'mle', got {ESTIMATOR!r}")


def fit_other(X, Y):
    """The estimator NOT in use, for the comparison."""
    if ESTIMATOR == "firth":
        return fit_logit_batch_traced(X, Y, ridge=RIDGE)[:2]
    return fit_firth_batch(X, Y)


def lapse_p(X, W, lapse):
    """Response probability under a lapse rate.

    (1 - lambda) * logistic(X beta) + lambda / 2. At lambda = 0 this is
    the step-6 model exactly, which is what makes the 0.0 row a
    reference rather than a separate condition.
    """
    p = 1.0 / (1.0 + np.exp(-(W @ X.T)))
    if not lapse:
        return p
    return (1.0 - lapse) * p + lapse / 2.0


# ============================================================
# Loading
# ============================================================

def load_draw(label, seed, index, path):
    t = pd.read_csv(path)
    for c in ("d_sat", "d_fog"):
        if c not in t.columns:
            raise ValueError(f"{path} has no {c!r} column.")
    if "analysed" in t.columns:
        t = t[t.analysed.astype(bool)]
    t = t.reset_index(drop=True)
    # THE REGISTERED MODEL'S PREDICTORS, in corpus-SD units, taken
    # from step 5's columns rather than rebuilt, so this script fits
    # what the registered model fits. `X_check` is the rebuild from the
    # raw differences and must agree.
    X_check = np.column_stack([-t.d_sat.to_numpy(float) / SD_SAT,
                               t.d_fog.to_numpy(float) / SD_FOG])
    if {"pred_sat", "pred_fog"} <= set(t.columns):
        X_sd = np.column_stack([t.pred_sat.to_numpy(float),
                                t.pred_fog.to_numpy(float)])
        if np.abs(X_sd - X_check).max() > 1e-9:
            raise RuntimeError(
                f"{path}: pred_sat / pred_fog disagree with the "
                "rebuild from d_sat / d_fog by\n"
                f"{np.abs(X_sd - X_check).max():.2e}. A sign "
                "convention or a unit has changed upstream.")
    else:
        X_sd = X_check
    return {"label": label, "seed": seed, "index": index, "path": path,
            "trials": t, "X_sd": X_sd,
            "colour_informative":
                np.abs(X_sd[:, 0]) >= COLOUR_INFORMATIVE_SD}


def load_all(root):
    """Every draw of every list, grouped by DRAW INDEX.

    Indexed by draw rather than by list because the simulation loop
    runs one draw index at a time: the whole point is that the four
    lists meet the same simulated people at that index.
    """
    print("\n" + "=" * 60)
    print("LOADING TRIAL LISTS")
    print("=" * 60)
    by_label = {}
    for lab in LISTS:
        paths = draw_paths(root, lab)
        if not paths:
            raise FileNotFoundError(
                f"No trial list for {lab!r} under {root}.\nRun step 5 "
                "first, and check TRIALS_ROOT -- step 5 writes to a "
                "new numbered directory\non each run, and this must be "
                "the build step 6 ranked.")
        by_label[lab] = [load_draw(lab, s, i, p)
                         for i, (s, p) in enumerate(paths)]
        n_tr = len(by_label[lab][0]["trials"])
        n_ci = int(by_label[lab][0]["colour_informative"].sum())
        print(f"  {lab:14s} {len(paths):2d} draw(s), {n_tr} trials, "
              f"{n_ci} colour-informative "
              f"(|pred_sat| >= {COLOUR_INFORMATIVE_SD})")

    n = min(len(v) for v in by_label.values())
    if len({len(v) for v in by_label.values()}) > 1:
        print(f"  NOTE: unequal draw counts; using the first {n} of "
              "each so every comparison is paired.")
    # Every list must have the same number of trials, or the shared
    # uniforms below would not line up.
    widths = {lab: len(v[0]["trials"]) for lab, v in by_label.items()}
    if len(set(widths.values())) > 1:
        raise ValueError(
            f"The lists have different trial counts ({widths}). The "
            "paired simulation needs them equal.")
    return by_label, n, list(widths.values())[0]


# ============================================================
# CHECK 1 and CHECK 2
# ============================================================

def check_solvers(draw, report):
    """CHECK 1: traced, batch and single solvers agree.

    Every diagnostic in CHECK B comes from the traced solver and every
    detection rate from the batch one. If they drift apart, the
    diagnostics describe fits that nobody reported.
    """
    X = draw["X_sd"]
    rng = np.random.default_rng(RANDOM_SEED)
    beta = np.array([0.5, 0.5])          # per SD, directly
    Y = np.array([simulate_responses(X, beta, rng) for _ in range(64)])
    verdicts = []

    # 1a -- batch against single
    one = np.array([fit_firth(X, y) for y in Y])
    many, _ = fit_firth_batch(X, Y)
    w = float(np.abs(one - many).max())
    ok = w <= FIT_CHECK_TOL
    verdicts.append(ok)
    print(f"  1a batch vs single, 64 fits        worst {w:.2e}   "
          f"{'PASS' if ok else 'FAIL'}")
    report.append(("check 1a batch vs single",
                   f"{'PASS' if ok else 'FAIL'} ({w:.1e})"))

    # 1b -- the penalised objective's gradient vanishes at the estimate
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

    # 1c -- the MLE's own traced solver still agrees with its batch
    # form, since the comparison below uses it
    mle_b, _, iters = fit_logit_batch_traced(X, Y, ridge=RIDGE)
    mle_one = np.array([fit_logit(X, y, ridge=RIDGE) for y in Y])
    w = float(np.abs(mle_b - mle_one).max())
    ok = w <= FIT_CHECK_TOL
    verdicts.append(ok)
    print(f"  1c MLE traced vs single            worst {w:.2e}   "
          f"{'PASS' if ok else 'FAIL'}  ({iters} iterations)")
    report.append(("check 1c MLE solvers",
                   f"{'PASS' if ok else 'FAIL'} ({w:.1e})"))

    # 1d -- separable data: finite, and the same at any iteration cap
    y_sep = (X[:, 0] + X[:, 1] > 0).astype(float)
    b1 = fit_firth(X, y_sep, n_iter=FIRTH_ITER)
    b2 = fit_firth(X, y_sep, n_iter=FIRTH_ITER * 10)
    stable = float(np.abs(b1 - b2).max())
    ok = np.all(np.isfinite(b1)) and stable <= FIT_CHECK_TOL
    verdicts.append(ok)
    print(f"  1d separable data                  Firth "
          f"[{b1[0]:.3f} {b1[1]:.3f}], unchanged at "
          f"{FIRTH_ITER * 10} iterations   {'PASS' if ok else 'FAIL'}")
    report.append(("check 1d separable",
                   f"{'PASS' if ok else 'FAIL'} "
                   f"({b1[0]:.2f}/{b1[1]:.2f} finite)"))

    # 1e -- equivariance, which is what makes SD units free
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
            "A solver check failed. Every number below comes from "
            "these estimators.")


def report_penalty_history(draw, report):
    """WHY THE ESTIMATOR CHANGED. A record, not a live diagnostic.

    The earlier version of this script fitted with a ridge of 1e-6 and
    checked, by refitting at 1e-3, whether that penalty was doing
    anything. It was not, and this function is the arithmetic behind
    that: the ridge is added to the diagonal of X'WX, the predictors
    were in RAW feature units -- a colour difference of 30 sat_mean
    units, not 0.8 SD -- so the diagonal ran to tens of thousands and
    the penalty sat eleven orders of magnitude below it.

    Which cut both ways. The fitted magnitudes were the
    maximum-likelihood ones and owed nothing to an unregistered penalty
    choice. But the penalty also gave NO protection when a fit
    approached separation: a participant whose answers the model
    predicted perfectly had no finite estimate, so the only options
    were to exclude them or to report a number the data did not
    support. Excluding them is the worse of the two -- a separable fit
    in the predicted direction is the strongest evidence for the
    effect, so dropping it biases against the hypothesis. Firth's
    penalty is always finite and keeps the participant.

    The numbers below are computed in the raw units the old fits used,
    so the record is reproducible. Nothing downstream depends on them.
    """
    X = draw["X_sd"] * np.array([SD_SAT, SD_FOG])
    diag = np.diag((X * 0.25).T @ X)
    rel = RIDGE / diag
    print("\n  why the estimator changed (the old fits, for the "
          "record):")
    print(f"    diagonal of X'WX (raw units)  colour {diag[0]:,.0f}   "
          f"fog {diag[1]:,.0f}")
    print(f"    a ridge of {RIDGE} was {rel[0]:.1e} and {rel[1]:.1e} "
          "of it")
    print("    So those fits were maximum-likelihood in practice: the "
          "penalty was not\n    holding the weights down, and gave no "
          "finite estimate at separation.\n    Firth's does, which is "
          "why it is now the registered fit.")
    report.append(("old ridge relative to X'WX",
                   f"{rel.max():.1e} (inert; see penalty_history.csv)"))
    return pd.DataFrame([{"feature": "colour", "hessian_diag": diag[0],
                          "old_ridge": RIDGE, "relative": rel[0]},
                         {"feature": "fog", "hessian_diag": diag[1],
                          "old_ridge": RIDGE, "relative": rel[1]}])


def check_floor_invariance(draw, report):
    """CHECK 2: the floor does not depend on the lapse rate.

    A pure guesser has zero weights, so logistic(X @ 0) = 0.5, and the
    lapse mixture gives (1 - lam) * 0.5 + lam / 2 = 0.5 EXACTLY at
    every lam. So the null distribution of fitted weights -- and the
    floor taken from it -- cannot depend on lam at all, and the floors
    in the measurement below are computed once per draw and shared
    across the lapse rates.

    That is an argument, and an argument is not a test. TWO tests:

    2a  PAIRED, exact. The same uniform draws at every lam. If the
        arithmetic above holds, the simulated responses are
        bit-identical across lam and so is every fitted weight, so the
        floors must agree to the last bit. Zero tolerance. This is the
        test that would catch a bug in how lam is applied -- and it is
        worth having because the SAME lapse_p generates the lapsing
        participants in CHECK A, where lam genuinely does change the
        responses.

    2b  UNPAIRED, against a measured spread. A fresh guesser set per
        lam, which is what the detection loop's floors are built from.
        Two independent sets never give the same percentile, so this
        asks only whether the lam-to-lam difference is the size of an
        ordinary sampling difference. The reference is measured, not
        chosen: N_FLOOR_SETS independent sets at lam = 0 give the
        set-to-set SD, and a difference between two independent sets
        has sqrt(2) times that.

    THE EARLIER VERSION OF THIS CHECK GOT 2b WRONG. It used a single
    extra guesser set as its reference, which is one draw from the
    distribution of differences rather than its spread, and then
    passed on a hand-picked floor of 0.02 instead. On `current` that
    understated the spread about threefold.
    """
    print("\n" + "=" * 60)
    print("CHECK 2: THE FLOORS DO NOT DEPEND ON THE LAPSE RATE")
    print("=" * 60)
    X = draw["X_sd"]
    zero = np.zeros((N_FLOOR_PEOPLE, 2))
    pct = 100 * (1 - FLOOR_ALPHA)

    def floor_from(u, lam):
        Y = (u < lapse_p(X, zero, lam)).astype(float)
        b, _ = fit_main(X, Y)
        return np.percentile(b, pct, axis=0), Y

    # --- 2a: the same uniforms at every lapse rate ---
    u = np.random.default_rng(
        [RANDOM_SEED, SEED_LAPSE_CHECK, 0]).random(
            (N_FLOOR_PEOPLE, X.shape[0]))
    rows, base_fl, base_Y, exact = [], None, None, True
    for lam in LAPSE_RATES:
        fl, Y = floor_from(u, lam)
        if base_fl is None:
            base_fl, base_Y = fl, Y
        same = bool(np.array_equal(Y, base_Y))
        dc = float(fl[0] - base_fl[0])
        df = float(fl[1] - base_fl[1])
        exact = exact and same and dc == 0.0 and df == 0.0
        rows.append({"test": "2a paired", "lapse": lam,
                     "floor_colour_SD": round(float(fl[0]), 6),
                     "floor_fog_SD": round(float(fl[1]), 6),
                     "diff_colour": dc, "diff_fog": df,
                     "responses_identical": same})
    print("\n  2a  the SAME uniform draws at every lapse rate:")
    print(pd.DataFrame(rows).to_string(index=False))
    print("      A guesser answers at 0.5 whatever lambda is, so the "
          "responses and\n      the floors must be identical to the "
          "last bit.")
    ok_a = exact
    print(f"      -> {'PASS' if ok_a else 'FAIL'}")
    report.append(("check 2a lapse cannot touch a guesser",
                   "PASS (bit-identical)" if ok_a
                   else "FAIL: lambda changed a guesser's responses "
                        "or floor"))
    if not ok_a:
        raise RuntimeError(
            "A lapse changed a guesser's responses. The same lapse_p "
            "generates the\nlapsing participants in CHECK A, so every "
            "detection rate below is suspect.")

    # --- 2b: the floor's own spread, measured ---
    ref = []
    for i in range(N_FLOOR_SETS):
        ui = np.random.default_rng(
            [RANDOM_SEED, SEED_LAPSE_CHECK, 1000 + i]).random(
                (N_FLOOR_PEOPLE, X.shape[0]))
        ref.append(floor_from(ui, 0.0)[0])
    ref = np.array(ref)
    sd_set = ref.std(ddof=1, axis=0)
    sd_diff = sd_set * np.sqrt(2.0)
    print(f"\n  2b  {N_FLOOR_SETS} independent guesser sets at lapse "
          "0, to measure the spread:")
    print(f"      colour floor  mean {ref[:, 0].mean():.4f}   "
          f"set-to-set SD {sd_set[0]:.4f}   "
          f"SD of a difference {sd_diff[0]:.4f}")
    print(f"      fog    floor  mean {ref[:, 1].mean():.4f}   "
          f"set-to-set SD {sd_set[1]:.4f}   "
          f"SD of a difference {sd_diff[1]:.4f}")

    base = ref[0]
    brows, worst_z = [], 0.0
    for lam in LAPSE_RATES:
        ul = np.random.default_rng(
            [RANDOM_SEED, SEED_LAPSE_CHECK,
             2000 + int(lam * 100)]).random(
                (N_FLOOR_PEOPLE, X.shape[0]))
        fl = floor_from(ul, lam)[0]
        d = fl - base
        z = np.abs(d) / sd_diff
        worst_z = max(worst_z, float(z.max()))
        brows.append({"test": "2b unpaired", "lapse": lam,
                      "floor_colour_SD": round(float(fl[0]), 4),
                      "floor_fog_SD": round(float(fl[1]), 4),
                      "diff_colour": round(float(d[0]), 4),
                      "diff_fog": round(float(d[1]), 4),
                      "z_colour": round(float(z[0]), 2),
                      "z_fog": round(float(z[1]), 2)})
    print("\n      a fresh guesser set per lapse rate, against the "
          "first reference set,\n      in units of the SD of such a "
          "difference:")
    print(pd.DataFrame(brows).to_string(index=False))
    ok_b = worst_z <= 2.0
    print(f"      worst shift {worst_z:.2f} SD of a sampling "
          f"difference -> {'PASS' if ok_b else 'CHECK'}")
    if not ok_b:
        print("      Beyond 2 SD. 2a passed, so lambda does not touch "
              "a guesser's\n      responses; this would be the "
              "percentile itself, and widening\n      N_FLOOR_PEOPLE "
              "is the fix rather than changing the model.")
    report.append(("check 2b floor vs lapse, unpaired",
                   f"{'PASS' if ok_b else 'CHECK'} "
                   f"(worst {worst_z:.2f} SD of a sampling "
                   f"difference)"))

    tab = pd.DataFrame(rows + brows)
    tab["note"] = ([""] * len(rows)
                   + [f"reference: {N_FLOOR_SETS} sets, SD of a "
                      f"difference {sd_diff[0]:.4f} colour / "
                      f"{sd_diff[1]:.4f} fog"] * len(brows))
    return tab


# ============================================================
# The measurement
# ============================================================

def weight_stats(b_fit, last_step, Y, X_sd, informative, true_sd,
                 X_fit, suffix=""):
    """CHECK B's diagnostics for one cell.

    `b_fit` and `X_fit` are the fit's own weights and design, which on
    the registered model are already in per-SD units -- so a weight is
    directly comparable with its true value and with the other
    feature. `X_fit` is used only for the sign of a margin, so the
    units do not enter.
    """
    b_sd = b_fit[:, 0]
    out = {}
    ex = np.abs(b_sd) > EXTREME_WEIGHT_SD
    out[f"mean{suffix}"] = float(b_sd.mean())
    out[f"sd{suffix}"] = float(b_sd.std(ddof=1))
    out[f"p50{suffix}"] = float(np.percentile(b_sd, 50))
    out[f"p90{suffix}"] = float(np.percentile(b_sd, 90))
    out[f"p99{suffix}"] = float(np.percentile(b_sd, 99))
    out[f"max{suffix}"] = float(np.abs(b_sd).max())
    out[f"share_extreme{suffix}"] = float(ex.mean())
    keep = ~ex
    out[f"sd_trimmed{suffix}"] = (float(b_sd[keep].std(ddof=1))
                                  if keep.sum() > 1 else np.nan)
    if suffix:
        return out

    # Only computed for the primary fit. share_separated and
    # share_unconverged are properties of the estimator, so they are
    # reported for the registered one; the rest describe the DATA and
    # are identical under either.
    out["true_sd"] = float(true_sd)
    out["bias"] = float(b_sd.mean() - true_sd)
    out["sd_over_true"] = (float(b_sd.std(ddof=1) / true_sd)
                           if true_sd > 0 else np.nan)
    out["share_unconverged"] = float((last_step > CONVERGE_TOL).mean())
    # Separation: this person's own fitted weights classify every
    # response correctly, so the data are linearly separable and no
    # finite maximum-likelihood estimate exists.
    margin = (2.0 * Y - 1.0) * (b_fit @ X_fit.T)
    out["share_separated"] = float((margin > 0).all(axis=1).mean())
    # Model-free version: every colour-informative trial answered in
    # the predicted direction. This is the quantity that makes the
    # colour weight unbounded on a list of large colour differences.
    if informative.any():
        sgn = np.sign(X_sd[informative, 0])
        agree = (2.0 * Y[:, informative] - 1.0) * sgn > 0
        out["share_colour_perfect"] = float(agree.all(axis=1).mean())
        # THE MARGIN, not just the endpoint. A share at exactly 100% is
        # zero in every cell, which says "no separation" without saying
        # how close anyone came. These say how close: separation needs
        # colour_correct_max to reach 1.0.
        frac = agree.mean(axis=1)
        out["n_colour_informative"] = int(informative.sum())
        out["colour_correct_mean"] = float(frac.mean())
        out["colour_correct_p99"] = float(np.percentile(frac, 99))
        out["colour_correct_max"] = float(frac.max())
    else:
        out["share_colour_perfect"] = np.nan
        out["colour_correct_max"] = np.nan
    return out


def measure(by_label, n_draws, n_trials):
    """Floors, detection and weight diagnostics, one draw index at a
    time.

    The loop order is draw -> cell -> lapse -> list, so that ONE set of
    uniforms per (draw, sensitivity, type) serves every lapse rate and
    every list. That is what makes the comparisons paired; see the
    header.
    """
    print("\n" + "=" * 60)
    print("MEASURING")
    print("=" * 60)
    n_cells = (len(SENSITIVITY_SD) * len(PARTICIPANT_TYPES)
               * len(LAPSE_RATES) * len(LISTS))
    print(f"  {n_draws} draws x {n_cells} cells "
          f"({len(SENSITIVITY_SD)} sensitivities x "
          f"{len(PARTICIPANT_TYPES)} types x {len(LAPSE_RATES)} lapse "
          f"rates x {len(LISTS)} lists)")
    print(f"  {N_DETECT_PEOPLE:,} people per cell, "
          f"{N_FLOOR_PEOPLE:,} guessers per floor per list per draw")
    print(f"  fits per draw: {n_cells}"
          + (f" x 2 ({ESTIMATOR} and the other estimator)"
             if COMPARE_MLE else ""))
    t0 = time.time()

    det_rows, w_rows, floor_rows = [], [], []
    for ix in range(n_draws):
        # --- floors, per list, per draw; shared across lapse rates ---
        floors, floors_other = {}, {}
        for lab in LISTS:
            d = by_label[lab][ix]
            X = d["X_sd"]
            rng = np.random.default_rng([RANDOM_SEED, SEED_FLOOR, ix])
            guess = simulate_many(X, np.zeros((N_FLOOR_PEOPLE, 2)),
                                  rng)
            # THE SAME ESTIMATOR AS THE WEIGHTS IT JUDGES.
            est, _ = fit_main(X, guess)
            fl = np.percentile(est, 100 * (1 - FLOOR_ALPHA), axis=0)
            floors[lab] = fl
            row = {"list": lab, "draw": ix, "seed": d["seed"],
                   "floor_colour_SD": float(fl[0]),
                   "floor_fog_SD": float(fl[1])}
            if COMPARE_MLE:
                # The other estimator on the SAME guessers, so the
                # floor shift is the estimator and not the sample.
                o, _ = fit_other(X, guess)
                fo = np.percentile(o, 100 * (1 - FLOOR_ALPHA), axis=0)
                floors_other[lab] = fo
                row.update({
                    "floor_colour_SD_other": float(fo[0]),
                    "floor_fog_SD_other": float(fo[1]),
                    "floor_colour_shift": float(fl[0] - fo[0]),
                    "floor_fog_shift": float(fl[1] - fo[1])})
            floor_rows.append(row)

        for si, sens in enumerate(SENSITIVITY_SD):
            for ti, (tname, (ms, mf)) in enumerate(
                    PARTICIPANT_TYPES.items()):
                # ONE set of uniforms for this cell, reused by every
                # lapse rate and every list.
                u = np.random.default_rng(
                    [RANDOM_SEED, SEED_DETECT, ix, si, ti]
                ).random((N_DETECT_PEOPLE, n_trials))
                # Per SD, applied directly: the predictors are
                # already in those units.
                W = np.zeros((N_DETECT_PEOPLE, 2))
                W[:, 0] = sens * ms
                W[:, 1] = sens * mf
                for lam in LAPSE_RATES:
                    for lab in LISTS:
                        d = by_label[lab][ix]
                        X = d["X_sd"]
                        p = lapse_p(X, W, lam)
                        Y = (u < p).astype(float)
                        b, last = fit_main(X, Y)
                        hit = b > floors[lab]
                        rec = {
                            "list": lab, "draw": ix, "seed": d["seed"],
                            "sensitivity_SD": sens,
                            "participant": tname, "lapse": lam,
                            "detect_colour": float(hit[:, 0].mean()),
                            "detect_fog": float(hit[:, 1].mean())}
                        if COMPARE_MLE:
                            # The other estimator on the SAME
                            # responses, judged against ITS OWN floor.
                            # Comparing one estimator's weights with
                            # another's floor would measure an offset,
                            # not a difference.
                            bo, _ = fit_other(X, Y)
                            ho = bo > floors_other[lab]
                            rec.update({
                                "detect_colour_other":
                                    float(ho[:, 0].mean()),
                                "detect_fog_other":
                                    float(ho[:, 1].mean())})
                        det_rows.append(rec)
                        if tname == "fog_only":
                            # The colour weight is null here; its
                            # spread is the floor's business, not CHECK
                            # B's.
                            continue
                        st = weight_stats(
                            b, last, Y, X,
                            d["colour_informative"], sens * ms, X)
                        if COMPARE_MLE:
                            st.update(weight_stats(
                                bo, last, Y, X,
                                d["colour_informative"], sens * ms, X,
                                suffix="_other"))
                        w_rows.append({
                            "list": lab, "draw": ix, "seed": d["seed"],
                            "sensitivity_SD": sens,
                            "participant": tname, "lapse": lam, **st})
        el = time.time() - t0
        print(f"    draw {ix + 1}/{n_draws}  ({el:.0f}s, "
              f"{el / (ix + 1):.1f}s each)")
    return (pd.DataFrame(det_rows), pd.DataFrame(w_rows),
            pd.DataFrame(floor_rows))


# ============================================================
# Aggregation
# ============================================================

def summarise_detection(det):
    """Mean over draws, with the between-draw spread."""
    print("\n" + "=" * 60)
    print("A. DETECTION AGAINST THE LAPSE RATE")
    print("=" * 60)
    rows = []
    for (lab, sens, part, lam), g in det.groupby(
            ["list", "sensitivity_SD", "participant", "lapse"],
            sort=False):
        for feat in ("detect_colour", "detect_fog"):
            v = g[feat].to_numpy(float)
            n = len(v)
            sd = float(v.std(ddof=1)) if n > 1 else np.nan
            r = {"list": lab, "sensitivity_SD": sens,
                 "participant": part, "lapse": lam,
                 "feature": feat.replace("detect_", ""),
                 "detect": round(float(v.mean()), 4),
                 "sd_between": round(sd, 4),
                 "se_mean": (round(sd / np.sqrt(n), 4)
                             if n > 1 else np.nan),
                 "draws": n}
            oc = feat + "_other"
            if oc in g.columns:
                # The same cell under the other estimator, each judged
                # against its own floor. Paired over draws, so the
                # shift's error is the spread of the per-draw
                # difference and not the two spreads added.
                o = g[oc].to_numpy(float)
                dv = v - o
                r["detect_other"] = round(float(o.mean()), 4)
                r["detect_shift"] = round(float(dv.mean()), 4)
                r["detect_shift_se"] = (
                    round(float(dv.std(ddof=1) / np.sqrt(n)), 4)
                    if n > 1 else np.nan)
                r["detect_shift_worst_draw"] = round(
                    float(np.abs(dv).max()), 4)
            rows.append(r)
    out = pd.DataFrame(rows)
    sub = out[(out.participant == "colour_only")
              & (out.feature == "colour")]
    for lam in LAPSE_RATES:
        print(f"\n  colour detection, colour-only participant, "
              f"lapse {lam}:")
        print(sub[sub.lapse == lam]
              .pivot(index="sensitivity_SD", columns="list",
                     values="detect").round(3).to_string())
    fog = out[(out.participant == "fog_only") & (out.feature == "fog")]
    print("\n  fog detection, fog-only participant (the control), "
          "by lapse:")
    print(fog.pivot_table(index="sensitivity_SD",
                          columns=["lapse", "list"],
                          values="detect").round(3).to_string())
    return out


def differences_by_lapse(det, report):
    """Each candidate against the baseline, paired by draw, per lapse.

    Paired because the same simulated people, with the same uniforms,
    met both lists at that draw and that lapse rate. The standard error
    is the spread of the paired differences over draws, which carries
    both the design sample and the residual simulation noise.
    """
    print("\n" + "=" * 60)
    print(f"A2. GAIN OVER {BASELINE!r}, BY LAPSE RATE")
    print("=" * 60)
    base = det[det.list == BASELINE]
    key = ["sensitivity_SD", "participant", "lapse", "draw"]
    rows = []
    for lab, g in det.groupby("list", sort=False):
        if lab == BASELINE:
            continue
        m = g.merge(base, on=key, suffixes=("", "_b"))
        for feat in ("detect_colour", "detect_fog"):
            m["_d"] = m[feat] - m[feat + "_b"]
            for (sens, part, lam), gg in m.groupby(
                    ["sensitivity_SD", "participant", "lapse"],
                    sort=False):
                v = gg["_d"].to_numpy(float)
                n = len(v)
                sd = float(v.std(ddof=1)) if n > 1 else np.nan
                se = sd / np.sqrt(n) if n > 1 else np.nan
                rows.append({
                    "list": lab, "sensitivity_SD": sens,
                    "participant": part, "lapse": lam,
                    "feature": feat.replace("detect_", ""),
                    "baseline": round(float(gg[feat + "_b"].mean()), 4),
                    "this_list": round(float(gg[feat].mean()), 4),
                    "difference": round(float(v.mean()), 4),
                    "se": round(se, 4) if np.isfinite(se) else np.nan,
                    "draws": n,
                    "resolvable": bool(np.isfinite(se)
                                       and abs(v.mean()) > 2 * se)})
    diff = pd.DataFrame(rows)
    show = diff[(diff.participant == "colour_only")
                & (diff.feature == "colour")]
    for lam in LAPSE_RATES:
        print(f"\n  colour gain, colour-only participant, lapse {lam}:")
        print(show[show.lapse == lam]
              .pivot(index="sensitivity_SD", columns="list",
                     values="difference").round(3).to_string())
    print("\n  Lapses pull every response toward 0.5, so detection and "
          "the gains both\n  fall. What has to survive is the ORDER of "
          "the candidates and the size\n  of the advantage -- see the "
          "ranking table.")
    return diff


def ranking(diff, report):
    """Does the candidate order survive every lapse rate?

    Checked on the colour detection of a colour-only participant, which
    is the quantity the re-run exists to improve, at every sensitivity
    and every lapse rate. Adjacent gaps are tested against the paired
    standard errors; a pair whose gap is inside 2 SE is reported as
    TIED rather than counted as an inversion, because an unresolvable
    gap is not evidence of an order either way.
    """
    print("\n" + "=" * 60)
    print("A3. IS THE RANKING PRESERVED?")
    print("=" * 60)
    print(f"  expected order (step 6, no lapse): "
          f"{' < '.join(CANDIDATES)}")
    d = diff[(diff.participant == "colour_only")
             & (diff.feature == "colour")
             & (diff.list.isin(CANDIDATES))]
    rows, inversions = [], 0
    for (sens, lam), g in d.groupby(["sensitivity_SD", "lapse"],
                                    sort=False):
        got = g.set_index("list")
        order = list(got.difference.sort_values().index)
        r = {"sensitivity_SD": sens, "lapse": lam,
             "order": " < ".join(order),
             "as_expected": order == CANDIDATES}
        for a, b in zip(CANDIDATES, CANDIDATES[1:]):
            if a not in got.index or b not in got.index:
                continue
            gap = float(got.loc[b, "difference"]
                        - got.loc[a, "difference"])
            se = float(np.hypot(got.loc[a, "se"], got.loc[b, "se"]))
            r[f"gap_{b}_minus_{a}"] = round(gap, 4)
            r[f"verdict_{b}_vs_{a}"] = (
                "ok" if gap > 2 * se else
                "tied" if abs(gap) <= 2 * se else "INVERTED")
            if gap < -2 * se:
                inversions += 1
        rows.append(r)
    tab = pd.DataFrame(rows)
    print()
    print(tab.to_string(index=False))
    ok = inversions == 0
    print(f"\n  resolvable inversions: {inversions} -> "
          f"{'PASS' if ok else 'FAIL'}")
    if ok:
        print("  The order never reverses by more than the paired "
              "error, at any lapse\n  rate. `tied` means two "
              "candidates are indistinguishable in that cell,\n  "
              "which is a finding about them and not a failure.")
    else:
        print("  The order reverses somewhere by more than the error. "
              "The provisional\n  choice rests on the order, so this "
              "has to be read before freezing.")
    report.append(("A3 ranking preserved",
                   f"{'PASS' if ok else 'FAIL'} "
                   f"({inversions} resolvable inversion(s))"))
    return tab, ok


def summarise_weights(w, report):
    """CHECK B: are the fitted colour weights stable enough to compare?

    Averaged over draws. The shares are proportions of simulated
    people, so their mean over draws is the overall rate; the WORST
    draw is reported too, because a rate that is fine on average and
    bad on one draw is still bad on the list that gets shipped.
    """
    print("\n" + "=" * 60)
    print("B. WEIGHT STABILITY")
    print("=" * 60)
    agg = {}
    for c in ("true_sd", "mean", "sd", "sd_trimmed", "p50", "p90",
              "p99", "bias", "sd_over_true", "share_extreme",
              "share_separated", "share_colour_perfect",
              "share_unconverged"):
        if c in w.columns:
            agg[c] = "mean"
    for c in ("share_extreme", "share_separated",
              "share_colour_perfect", "max"):
        if c in w.columns:
            agg[c] = "mean"
    rows = []
    for (lab, sens, part, lam), g in w.groupby(
            ["list", "sensitivity_SD", "participant", "lapse"],
            sort=False):
        r = {"list": lab, "sensitivity_SD": sens, "participant": part,
             "lapse": lam, "draws": len(g)}
        for c in ("true_sd", "mean", "sd", "sd_trimmed", "p50", "p90",
                  "p99", "max", "bias", "sd_over_true",
                  "n_colour_informative", "colour_correct_mean",
                  "colour_correct_p99", "colour_correct_max"):
            if c in g.columns:
                r[c] = round(float(g[c].mean()), 4)
        for c in ("share_extreme", "share_separated",
                  "share_colour_perfect", "share_unconverged"):
            if c in g.columns:
                r[c] = round(float(g[c].mean()), 5)
                r[c + "_worst_draw"] = round(float(g[c].max()), 5)
        if "sd_other" in g.columns:
            r["mean_other"] = round(float(g.mean_other.mean()), 4)
            r["sd_other"] = round(float(g.sd_other.mean()), 4)
            r["p90_other"] = round(float(g.p90_other.mean()), 4)
            r["share_extreme_other"] = round(
                float(g.share_extreme_other.mean()), 5)
            # Paired over draws: the same responses went through both
            # estimators, so the shift is a within-draw difference.
            dm = g["mean"].to_numpy(float) - g.mean_other.to_numpy(
                float)
            ds = g["sd"].to_numpy(float) - g.sd_other.to_numpy(float)
            r["weight_shift_mean"] = round(float(dm.mean()), 4)
            r["weight_shift_mean_worst"] = round(
                float(np.abs(dm).max()), 4)
            r["weight_shift_sd"] = round(float(ds.mean()), 4)
            # As a fraction of the weight, which is the form the other
            # session asked for: a shift of 0.01 means nothing at a
            # weight of 1 and everything at a weight of 0.02.
            base = float(g.mean_other.mean())
            r["weight_shift_pct"] = (round(100.0 * dm.mean() / base, 2)
                                     if abs(base) > 1e-12 else np.nan)
        rows.append(r)
    out = pd.DataFrame(rows)

    co = out[(out.participant == "colour_only")]
    for lam in LAPSE_RATES:
        g = co[co.lapse == lam]
        print(f"\n  colour-only participant, lapse {lam}: fitted colour "
              "weight against the truth")
        print(g[["list", "sensitivity_SD", "true_sd", "mean", "bias",
                 "sd", "sd_over_true", "p90", "p99"]]
              .to_string(index=False))

    print("\n  HOW CLOSE ANYONE CAME TO SEPARATION, colour-only.")
    print("  Separation needs colour_correct_max = 1.000: every")
    print("  colour-informative trial answered as predicted.")
    mc = [c for c in ["list", "sensitivity_SD", "lapse",
                      "n_colour_informative", "colour_correct_mean",
                      "colour_correct_p99", "colour_correct_max"]
          if c in co.columns]
    print(co[co.lapse == 0.0][mc].to_string(index=False))

    print("\n  instability, colour-only, every lapse rate:")
    cols = ["list", "sensitivity_SD", "lapse", "share_separated",
            "share_colour_perfect", "share_extreme",
            "share_extreme_worst_draw", "share_unconverged"]
    cols = [c for c in cols if c in co.columns]
    print(co[cols].to_string(index=False))

    worst_ex = float(co.share_extreme.max())
    worst_ex_draw = float(co.share_extreme_worst_draw.max())
    worst_sep = float(co.share_separated.max())
    worst_perf = float(co.share_colour_perfect.max())
    worst_unc = float(co.share_unconverged.max())
    print(f"\n  worst over all cells (colour-only):")
    print(f"    separated fits          {worst_sep:.5f}")
    print(f"    all colour trials right {worst_perf:.5f}")
    print(f"    extreme weights         {worst_ex:.5f}  "
          f"(worst single draw {worst_ex_draw:.5f})")
    print(f"    unconverged fits        {worst_unc:.5f}")
    report.append(("B worst share separated", f"{worst_sep:.5f}"))
    report.append(("B worst share extreme",
                   f"{worst_ex:.5f} (worst draw {worst_ex_draw:.5f})"))
    report.append(("B worst share unconverged", f"{worst_unc:.5f}"))
    if "colour_correct_max" in co.columns:
        cm = float(co.colour_correct_max.max())
        print(f"    closest to separation    {cm:.3f} of "
              "colour-informative trials (needs 1.000)")
        report.append(("B closest to separation",
                       f"{cm:.3f} of colour trials (needs 1.000)"))

    # THE ANSWER TO THE WORRY, and it runs the other way: the cut
    # versions carry MORE colour information, so their fitted weights
    # are TIGHTER, not looser. This is the quantity a between-group
    # comparison has to overcome, so it is worth stating on its own.
    mid = co[(co.sensitivity_SD == 0.50) & (co.lapse == 0.0)]
    if len(mid):
        print("\n  noise-to-signal in the fitted colour weight at "
              "0.50 SD, lapse 0:")
        for _, r in mid.iterrows():
            print(f"    {r['list']:14s} sd/true {r.sd_over_true:.3f}"
                  f"   sd {r['sd']:.3f}")
        b = mid[mid.list == BASELINE]
        if len(b):
            b0 = float(b.sd_over_true.iloc[0])
            best = mid.loc[mid.sd_over_true.idxmin()]
            print(f"    tightest: {best['list']} at "
                  f"{best.sd_over_true:.3f}, "
                  f"{1 - best.sd_over_true / b0:.0%} tighter than "
                  f"{BASELINE}")
            report.append(("B noise-to-signal at 0.50 SD",
                           f"{BASELINE} {b0:.3f}, best "
                           f"{best['list']} {best.sd_over_true:.3f}"))

    if "weight_shift_mean" in out.columns:
        sm = float(co.weight_shift_mean.abs().max())
        sp = float(co.weight_shift_pct.abs().max())
        ss = float(co.weight_shift_sd.abs().max())
        print(f"\n  what the change of estimator did to the weights "
              f"({ESTIMATOR} vs the other):")
        print(f"    largest shift in the mean fitted weight "
              f"{sm:.4f} SD ({sp:.1f}%)")
        print(f"    largest shift in its SD                 "
              f"{ss:.4f} SD")
        print("    Firth shrinks towards zero, so a small negative "
              "shift is expected and\n    is the bias correction "
              "working. Per cell: estimator_comparison.csv.")
        report.append(("B weight shift, estimator",
                       f"<= {sm:.4f} SD ({sp:.1f}%), "
                       f"sd <= {ss:.4f} SD"))
    return out


def compare_estimators(det_s, wt, floors, report):
    """PER CELL: how far the weights move, how far the floors move,
    whether detection changes.

    This is the table the other session asked for when it registered
    Firth. One row per (list, sensitivity, participant, lapse), with
    the weight movement from CHECK B and the detection movement from
    CHECK A side by side, plus a floor table per list. Every
    comparison is paired within a draw -- the same responses, the same
    guessers -- so a shift is a within-draw difference and its error
    is the spread of that difference over draws, not the two spreads
    added.
    """
    print("\n" + "=" * 60)
    print(f"C. {ESTIMATOR.upper()} AGAINST THE OTHER ESTIMATOR")
    print("=" * 60)
    if "detect_shift" not in det_s.columns:
        print("  COMPARE_MLE is off; nothing to compare.")
        return pd.DataFrame(), pd.DataFrame()

    # --- the floors ---
    fl_rows = []
    for lab, g in floors.groupby("list", sort=False):
        r = {"list": lab, "draws": len(g)}
        for feat in ("colour", "fog"):
            m = g[f"floor_{feat}_SD"].to_numpy(float)
            o = g[f"floor_{feat}_SD_other"].to_numpy(float)
            d = m - o
            n = len(d)
            r[f"floor_{feat}"] = round(float(m.mean()), 4)
            r[f"floor_{feat}_other"] = round(float(o.mean()), 4)
            r[f"floor_{feat}_shift"] = round(float(d.mean()), 4)
            r[f"floor_{feat}_shift_pct"] = round(
                100.0 * float(d.mean()) / float(o.mean()), 2)
            r[f"floor_{feat}_shift_se"] = (
                round(float(d.std(ddof=1) / np.sqrt(n)), 4)
                if n > 1 else np.nan)
        fl_rows.append(r)
    fl = pd.DataFrame(fl_rows)
    print("\n  THE FLOORS, per list (per SD, same guessers both ways):")
    print(fl[["list", "draws", "floor_colour", "floor_colour_other",
              "floor_colour_shift", "floor_colour_shift_pct",
              "floor_fog", "floor_fog_other", "floor_fog_shift",
              "floor_fog_shift_pct"]].to_string(index=False))
    worst_fl = float(np.abs(fl.floor_colour_shift_pct).max())
    print(f"    largest colour-floor move {worst_fl:.1f}%. A lower "
          "floor is a lower bar for\n    detection; the detection "
          "table below is what it actually bought.")
    report.append(("C floor move, colour",
                   f"<= {worst_fl:.1f}% across lists"))

    # --- the cells ---
    keep = ["list", "sensitivity_SD", "participant", "lapse",
            "feature", "detect", "detect_other", "detect_shift",
            "detect_shift_se", "detect_shift_worst_draw", "draws"]
    cells = det_s[keep].copy()
    wcols = [c for c in ("mean", "mean_other", "weight_shift_mean",
                         "weight_shift_pct", "weight_shift_mean_worst",
                         "sd", "sd_other", "weight_shift_sd",
                         "share_extreme", "share_extreme_other")
             if c in wt.columns]
    cells = cells.merge(
        wt[["list", "sensitivity_SD", "participant", "lapse"] + wcols],
        on=["list", "sensitivity_SD", "participant", "lapse"],
        how="left")

    co = cells[(cells.participant == "colour_only")
               & (cells.feature == "colour")]
    print("\n  COLOUR-ONLY PARTICIPANT, colour weight and colour "
          "detection:")
    for lam in LAPSE_RATES:
        g = co[co.lapse == lam]
        if not len(g):
            continue
        print(f"\n    lapse {lam}:")
        print(g[["list", "sensitivity_SD", "mean", "mean_other",
                 "weight_shift_mean", "weight_shift_pct", "detect",
                 "detect_other", "detect_shift", "detect_shift_se"]]
              .round(4).to_string(index=False))

    fo = cells[(cells.participant == "fog_only")
               & (cells.feature == "fog")]
    print("\n  FOG-ONLY PARTICIPANT, fog detection (the control):")
    print(fo[["list", "sensitivity_SD", "lapse", "detect",
              "detect_other", "detect_shift", "detect_shift_se"]]
          .round(4).to_string(index=False))

    # TWO COUNTS, because they answer different questions. "Beyond 2
    # SE" asks whether a shift is real; MATERIAL_SHIFT asks whether it
    # is large enough to change a power figure. With this many
    # simulated people a shift of one person in a cell is detectable
    # and meaningless, so the second count is the one that matters,
    # and the first is reported so it cannot be said to be hidden.
    ds = cells.detect_shift.abs()
    worst = float(ds.max())
    n_mat = int((ds > MATERIAL_SHIFT).sum())
    have_se = cells.detect_shift_se.notna()
    print(f"\n  largest detection shift in any cell {worst:.4f}")
    print(f"    cells moving by more than {MATERIAL_SHIFT} "
          f"(a material shift): {n_mat} of {len(cells)}")
    if have_se.any():
        se = cells.loc[have_se, "detect_shift_se"]
        big = int((ds[have_se] > 2 * se).sum())
        print(f"    cells moving by more than twice the paired error:  "
              f"{big} of {int(have_se.sum())}")
        extra = f", {big}/{int(have_se.sum())} beyond 2 SE"
    else:
        # One draw: there is no paired error, so no claim is made
        # about significance.
        print("    only one draw, so the shift has no paired error "
              "and nothing is\n    claimed about its significance.")
        extra = " (one draw: no paired error)"
    print("    Firth both lowers the floor and shrinks the weights, "
          "so the two largely\n    cancel in a detection rate. That "
          "is the result, not an accident of\n    rounding: the "
          "columns above show both moves.")
    report.append(("C detection shift, worst cell",
                   f"{worst:.4f} ({n_mat}/{len(cells)} cells "
                   f"> {MATERIAL_SHIFT}{extra})"))
    return cells, fl


# ============================================================
# Figures
# ============================================================

def make_figures(det, w, out_dir):
    labs = [l for l in LISTS]

    fig, axes = plt.subplots(1, len(LAPSE_RATES),
                             figsize=(5.2 * len(LAPSE_RATES), 4.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    sub = det[(det.participant == "colour_only")
              & (det.feature == "colour")]
    for ax, lam in zip(axes, LAPSE_RATES):
        s = sub[sub.lapse == lam]
        for lab in labs:
            g = s[s.list == lab].sort_values("sensitivity_SD")
            if not len(g):
                continue
            ax.plot(g.sensitivity_SD, g.detect, marker="o", ms=4,
                    lw=1.6, label=lab,
                    ls="--" if lab == BASELINE else "-")
            sd = g.sd_between.fillna(0).to_numpy()
            ax.fill_between(g.sensitivity_SD, g.detect - sd,
                            g.detect + sd, alpha=0.12, lw=0)
        ax.set_title(f"lapse {lam}")
        ax.set_xlabel("true sensitivity (per SD)")
        ax.axhline(0.8, color="k", lw=0.6, ls=":")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("colour detection, colour-only participant")
    axes[0].legend(fontsize=8)
    fig.suptitle("Detection against the lapse rate  (mean over draws, "
                 "band = between-draw SD)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "detection_by_lapse.png"),
                dpi=150)
    plt.close(fig)

    if not len(w):
        return
    co = w[w.participant == "colour_only"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for lab in labs:
        g = co[(co.list == lab) & (co.lapse == 0.0)].sort_values(
            "sensitivity_SD")
        if not len(g):
            continue
        axes[0].errorbar(g.true_sd, g["mean"], yerr=g["sd"], marker="o",
                         ms=4, lw=1.5, capsize=3, label=lab)
        axes[1].plot(g.sensitivity_SD, g.sd_over_true, marker="o",
                     ms=4, lw=1.5, label=lab)
        axes[2].plot(g.sensitivity_SD, g.share_extreme, marker="o",
                     ms=4, lw=1.5, label=lab)
    lim = [0, max(SENSITIVITY_SD) * 1.6]
    axes[0].plot(lim, lim, color="k", lw=0.8, ls=":")
    axes[0].set_xlabel("true colour weight (per SD)")
    axes[0].set_ylabel("fitted (error bar = SD over people)")
    axes[0].set_title("Recovery of the colour weight\n"
                      "(dotted = perfect, lapse 0)")
    axes[1].set_xlabel("true sensitivity (per SD)")
    axes[1].set_ylabel("SD of fitted / true")
    axes[1].set_title("Noise-to-signal in the fitted weight\n"
                      "what a between-group comparison must overcome")
    axes[2].set_xlabel("true sensitivity (per SD)")
    axes[2].set_ylabel(f"share with |b| > {EXTREME_WEIGHT_SD} SD")
    axes[2].set_title("Runaway fits\n(near-separation)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "weights.png"), dpi=150)
    plt.close(fig)


# ============================================================
# Report
# ============================================================

def write_report(out_dir, report, det, diff, rank, ok_rank, wt,
                 floors, lapse_tab, cells, fl_cmp, n_draws, t0):
    path = os.path.join(out_dir, "robustness_report.txt")
    with open(path, "w") as f:
        def W(s=""):
            f.write(s + "\n")

        W("ROBUSTNESS OF THE CANDIDATE TRIAL LISTS  (v4, step 6b)")
        W("=" * 70)
        W(f"Trial lists : {TRIALS_ROOT}")
        W(f"Lists       : {', '.join(LISTS)}  (baseline {BASELINE})")
        W(f"Candidates  : {' < '.join(CANDIDATES)}  (step 6 order)")
        W(f"Draws       : {n_draws} per list"
          + ("" if MAX_SEEDS is None
             else f"   (MAX_SEEDS = {MAX_SEEDS}: A CAPPED RUN)"))
        W(f"Lapse rates : {LAPSE_RATES}")
        W(f"Sensitivity : {SENSITIVITY_SD} (per SD), applied to the "
          "per-SD predictors directly")
        W(f"People      : {N_DETECT_PEOPLE:,} per cell, "
          f"{N_FLOOR_PEOPLE:,} per floor")
        W(f"Fit         : {ESTIMATOR}, no intercept, pred_sat and "
          "pred_fog in corpus-SD units"
          + ("; every cell also refitted with the other estimator "
             "on the\n              same responses"
             if COMPARE_MLE else ""))
        W(f"Extreme     : |fitted colour weight| > "
          f"{EXTREME_WEIGHT_SD} per SD")
        W(f"Seed        : {RANDOM_SEED}, offsets "
          f"{SEED_FLOOR}/{SEED_DETECT}; uniforms shared across lists "
          "and lapse rates")
        W(f"Elapsed     : {time.time() - t0:.1f}s")
        W()
        for k, v in report:
            W(f"  {k:34s} {v}")

        W()
        W("WHAT THIS ADDS TO STEP 6")
        W("-" * 70)
        W("  Step 6 asked which list catches a participant with a")
        W("  given sensitivity most often. It assumed a participant")
        W("  who never slips, and it only asked whether a fitted")
        W("  weight cleared a floor -- never whether the weight was")
        W("  any good. Both matter here for different reasons.")
        W()
        W("  LAPSES, because a design that wins only for a perfect")
        W("  responder is not worth registering.")
        W()
        W("  WEIGHT STABILITY, because the registered control")
        W("  comparison compares weights BETWEEN GROUPS. Detection")
        W("  needs only the sign; a group difference needs the")
        W("  magnitude to mean something. On a list of mostly large")
        W("  colour differences a sensitive participant can answer")
        W("  nearly every colour trial as predicted, and in the limit")
        W("  the data are separable, the maximum-likelihood weight is")
        W("  infinite, and what the fit returns is set by the penalty")
        W(f"  -- which is {RIDGE}, almost nothing. A handful of those")
        W("  fits inflates a group's mean and variance.")
        W()
        W("  THE SIMULATION IS PAIRED THREE WAYS. One set of uniform")
        W("  random numbers per (draw, sensitivity, participant type)")
        W("  serves every list and every lapse rate, so the same")
        W("  person with the same luck meets all four lists at all")
        W("  three lapse rates.")

        W()
        W("CHECK 2: THE FLOORS DO NOT DEPEND ON THE LAPSE RATE")
        W("-" * 70)
        W("  A pure guesser answers at 0.5, and mixing that with a")
        W("  coin flip is still 0.5 EXACTLY, so the null distribution")
        W("  of fitted weights cannot move with lambda. The floors")
        W("  are therefore computed once per draw and shared across")
        W("  the lapse rates -- which is an argument, so it is tested")
        W("  twice.")
        W()
        W("  2a is the exact test: the SAME uniform draws at every")
        W("  lambda, so the responses and the floors must agree to the")
        W("  last bit. This is what would catch a bug in how lambda is")
        W("  applied, and it is worth having because the same code")
        W("  generates the lapsing participants in section A.")
        W()
        W("  2b is the sampling test: a FRESH guesser set per lambda,")
        W("  as the detection loop's floors are built, judged against")
        W(f"  a MEASURED spread -- {N_FLOOR_SETS} independent sets at")
        W("  lambda 0 give the set-to-set SD, and a difference between")
        W("  two independent sets has sqrt(2) times that. An earlier")
        W("  version of this check used ONE extra set as its")
        W("  reference, which is a single draw from the distribution")
        W("  of differences rather than its spread, and understated it")
        W("  about threefold.")
        W()
        W(lapse_tab.to_string(index=False))

        W()
        W("A. DETECTION AGAINST THE LAPSE RATE")
        W("-" * 70)
        W("  Colour detection, colour-only participant, mean over")
        W("  draws. Lapses pull every response toward 0.5, so these")
        W("  fall at every lapse rate by construction.")
        sub = det[(det.participant == "colour_only")
                  & (det.feature == "colour")]
        for lam in LAPSE_RATES:
            W()
            W(f"  lapse {lam}")
            W(sub[sub.lapse == lam]
              .pivot(index="sensitivity_SD", columns="list",
                     values="detect").round(3).to_string())
        W()
        W("  THE FOG CONTROL, fog-only participant, fog detection:")
        fog = det[(det.participant == "fog_only")
                  & (det.feature == "fog")]
        for lam in LAPSE_RATES:
            W()
            W(f"  lapse {lam}")
            W(fog[fog.lapse == lam]
              .pivot(index="sensitivity_SD", columns="list",
                     values="detect").round(3).to_string())

        W()
        W(f"A2. GAIN OVER {BASELINE!r}, PAIRED BY DRAW")
        W("-" * 70)
        show = diff[(diff.participant == "colour_only")
                    & (diff.feature == "colour")]
        for lam in LAPSE_RATES:
            W()
            W(f"  lapse {lam}: difference")
            W(show[show.lapse == lam]
              .pivot(index="sensitivity_SD", columns="list",
                     values="difference").round(3).to_string())
            W(f"  lapse {lam}: standard error of that difference")
            W(show[show.lapse == lam]
              .pivot(index="sensitivity_SD", columns="list",
                     values="se").round(4).to_string())

        W()
        W("A3. IS THE RANKING PRESERVED?")
        W("-" * 70)
        W(f"  Expected order from step 6: {' < '.join(CANDIDATES)}.")
        W("  Adjacent gaps are tested against the paired standard")
        W("  errors. `tied` means two candidates cannot be separated")
        W("  in that cell, which is a finding about them rather than a")
        W("  failure; only a gap that reverses by more than 2 SE")
        W("  counts as an inversion.")
        W()
        W(rank.to_string(index=False))
        W()
        W(f"  RESULT: {'PASS' if ok_rank else 'FAIL'}")

        W()
        W("B. WEIGHT STABILITY")
        W("-" * 70)
        W("  Fitted colour weight against its true value, in per-SD")
        W("  units, for a colour-only participant. `sd_over_true` is")
        W("  the noise-to-signal ratio and is what a between-group")
        W("  comparison has to overcome.")
        co = wt[wt.participant == "colour_only"]
        for lam in LAPSE_RATES:
            W()
            W(f"  lapse {lam}")
            W(co[co.lapse == lam][
                ["list", "sensitivity_SD", "true_sd", "mean", "bias",
                 "sd", "sd_over_true", "p90", "p99"]
            ].to_string(index=False))
        W()
        W("  INSTABILITY. share_separated: the fit's own weights")
        W("  classify every response correctly, so no finite MLE")
        W("  exists. share_colour_perfect: every colour-informative")
        W("  trial answered as predicted -- the model-free form of the")
        W("  same thing. share_extreme: |weight| above")
        W(f"  {EXTREME_WEIGHT_SD} per SD. share_unconverged: the last")
        W(f"  Newton step still above {CONVERGE_TOL} at the cap.")
        cols = [c for c in
                ["list", "sensitivity_SD", "lapse", "share_separated",
                 "share_colour_perfect", "share_extreme",
                 "share_extreme_worst_draw", "share_unconverged"]
                if c in co.columns]
        W()
        W(co[cols].to_string(index=False))
        if "colour_correct_max" in co.columns:
            W()
            W("  HOW CLOSE ANYONE CAME. share_separated and")
            W("  share_colour_perfect are zero in every cell, which")
            W("  says there was no separation without saying how near")
            W("  it was. Separation needs colour_correct_max = 1.000:")
            W("  EVERY colour-informative trial answered as predicted.")
            mc = [c for c in ["list", "sensitivity_SD",
                              "n_colour_informative",
                              "colour_correct_mean",
                              "colour_correct_p99",
                              "colour_correct_max"]
                  if c in co.columns]
            W()
            W(co[co.lapse == 0.0][mc].to_string(index=False))
        mid = co[(co.sensitivity_SD == 0.50) & (co.lapse == 0.0)]
        if len(mid):
            W()
            W("  THE WORRY RUNS THE OTHER WAY. The concern was that")
            W("  mostly-large colour differences would destabilise the")
            W("  fitted weights. They tighten them: more colour")
            W("  information is a more precise estimate, and")
            W("  `sd_over_true` -- the noise-to-signal ratio a")
            W("  between-group comparison has to overcome -- is")
            W("  LOWEST on the cut versions.")
            W()
            W("    at 0.50 SD, lapse 0:")
            for _, r in mid.iterrows():
                W(f"      {r['list']:14s} sd/true {r.sd_over_true:.3f}")
        if "weight_shift_mean" in co.columns:
            W()
            W("  WHAT THE CHANGE OF ESTIMATOR DID. Every cell was")
            W(f"  fitted twice on the SAME responses, with {ESTIMATOR}")
            W("  and with the plain maximum-likelihood estimator the")
            W("  earlier version of this script used. Paired within a")
            W("  draw, so a shift is a within-draw difference.")
            W(f"    largest shift in the mean weight "
              f"{co.weight_shift_mean.abs().max():.4f} SD "
              f"({co.weight_shift_pct.abs().max():.1f}%)")
            W(f"    largest shift in its SD          "
              f"{co.weight_shift_sd.abs().max():.4f} SD")
            W("  Firth shrinks towards zero, so a small negative shift")
            W("  is the bias correction working rather than a")
            W("  discrepancy. Section C has it per cell, with the")
            W("  floors and the detection rates beside it.")
            W()
            W("  WHY THE ESTIMATOR CHANGED. The old fits used a ridge")
            W(f"  of {RIDGE} on predictors in RAW feature units, so the")
            W("  diagonal of X'WX ran to tens of thousands and the")
            W("  penalty sat eleven orders of magnitude below it --")
            W("  see penalty_history.csv. Those fits were therefore")
            W("  maximum-likelihood in practice, and gave no finite")
            W("  estimate for a participant the model predicted")
            W("  perfectly. The only options were to exclude that")
            W("  participant or to report a number the data did not")
            W("  support, and excluding is the worse of the two: a")
            W("  separable fit in the predicted direction is the")
            W("  strongest evidence for the effect, so dropping it")
            W("  biases against the hypothesis. Firth's penalty is")
            W("  always finite and keeps the participant, which is why")
            W("  it is the registered fit. The separation diagnostics")
            W("  above are still reported per cell and per draw --")
            W("  they describe the data, not the solver.")

        W()
        if len(cells):
            W()
            W(f"C. {ESTIMATOR.upper()} AGAINST THE OTHER ESTIMATOR")
            W("-" * 70)
            W("  Every cell was fitted twice on the SAME responses and")
            W("  every floor built from the SAME guessers, so each")
            W("  shift below is a within-draw difference. Each")
            W("  estimator's weights are judged against ITS OWN floor;")
            W("  one estimator's weights against the other's floor")
            W("  would measure an offset rather than a difference.")
            W()
            W("  THE FLOORS, per list, per SD:")
            W(fl_cmp[["list", "floor_colour", "floor_colour_other",
                      "floor_colour_shift", "floor_colour_shift_pct",
                      "floor_fog", "floor_fog_other",
                      "floor_fog_shift", "floor_fog_shift_pct"]]
              .to_string(index=False))
            W()
            W("  THE WEIGHTS AND THE DETECTION RATES, colour-only")
            W("  participant. Full table: estimator_comparison.csv.")
            cc = cells[(cells.participant == "colour_only")
                       & (cells.feature == "colour")]
            for lam in LAPSE_RATES:
                g = cc[cc.lapse == lam]
                if not len(g):
                    continue
                W()
                W(f"  lapse {lam}")
                W(g[["list", "sensitivity_SD", "mean", "mean_other",
                     "weight_shift_pct", "detect", "detect_other",
                     "detect_shift"]].round(4).to_string(index=False))
            dsx = cells.detect_shift.abs()
            W()
            W("  THE TWO MOVES LARGELY CANCEL. Firth shrinks the")
            W("  weights towards zero AND lowers the floor they are")
            W("  judged against, by about the same couple of per cent,")
            W("  so a detection rate barely moves:")
            W(f"    largest weight shift   "
              f"{cc.weight_shift_pct.abs().max():.1f}%")
            W(f"    largest floor shift    "
              f"{fl_cmp.floor_colour_shift_pct.abs().max():.1f}% "
              "(colour), "
              f"{fl_cmp.floor_fog_shift_pct.abs().max():.1f}% (fog)")
            W(f"    largest detection shift in any cell "
              f"{float(dsx.max()):.4f}")
            W(f"    cells moving by more than {MATERIAL_SHIFT}: "
              f"{int((dsx > MATERIAL_SHIFT).sum())} of {len(cells)}")
            W()
            W("  So the choice of estimator does not move the design")
            W("  comparison. It was not made on these numbers: Firth")
            W("  was registered because it gives a finite estimate")
            W("  for a participant the model predicts perfectly, and")
            W("  this section is the check that it changed nothing")
            W("  else.")

        W()
        W("WHAT THIS DOES AND DOES NOT SETTLE")
        W("-" * 70)
        W("  Settles: whether the candidate order and the size of the")
        W("  gains survive lapses of 0.1 and 0.2, and how often a fit")
        W("  runs away on each candidate.")
        W()
        W("  Does not settle:")
        W("    - whether a real participant is a logistic responder")
        W("      with a fixed weight plus a lapse rate at all. The")
        W("      lapse model is the standard allowance for")
        W("      inattention, not a model of a person.")
        W("    - whether the larger colour edits look edited, which")
        W("      is a question for the contact sheets at step 7.")
        W("    - the power of the control comparison, which needs the")
        W("      group structure and sample size and is step 8's.")
        W()
        W("  The decision rule was fixed before this ran: `both`")
        W("  ships if the ranking and the gains hold and the weights")
        W("  are stable; otherwise sens_cut_25_5. Nothing here chose a")
        W("  seed or a version.")
    return path


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("ROBUSTNESS CHECKS  (v4, step 6b)")
    print("=" * 60)
    root = resolve_output_root()
    os.makedirs(root, exist_ok=True)
    print(f"  trials : {TRIALS_ROOT}")
    print(f"  output : {root}")
    print(f"  lapse  : {LAPSE_RATES}")
    print(f"  lists  : {', '.join(LISTS)}  (baseline {BASELINE})")
    if MAX_SEEDS is not None:
        print(f"  NOTE   : MAX_SEEDS = {MAX_SEEDS}; a capped run.")

    report = []
    by_label, n_draws, n_trials = load_all(TRIALS_ROOT)
    report.append(("draws per list", str(n_draws)))

    print()
    check_solvers(by_label[BASELINE][0], report)
    hist_tab = report_penalty_history(by_label[BASELINE][0], report)
    lapse_tab = check_floor_invariance(by_label[BASELINE][0], report)

    det, w, floors = measure(by_label, n_draws, n_trials)
    det_s = summarise_detection(det)
    diff = differences_by_lapse(det, report)
    rank, ok_rank = ranking(diff, report)
    wt = summarise_weights(w, report)
    cells, fl_cmp = compare_estimators(det_s, wt, floors, report)

    det.to_csv(os.path.join(root, "by_seed.csv"), index=False)
    det_s.to_csv(os.path.join(root, "detection_by_lapse.csv"),
                 index=False)
    diff.to_csv(os.path.join(root, "differences_by_lapse.csv"),
                index=False)
    rank.to_csv(os.path.join(root, "ranking.csv"), index=False)
    wt.to_csv(os.path.join(root, "weights.csv"), index=False)
    w.to_csv(os.path.join(root, "weights_by_seed.csv"), index=False)
    floors.to_csv(os.path.join(root, "floors_by_seed.csv"), index=False)
    lapse_tab.to_csv(os.path.join(root, "floor_invariance.csv"),
                     index=False)
    hist_tab.to_csv(os.path.join(root, "penalty_history.csv"),
                    index=False)
    if len(cells):
        cells.to_csv(os.path.join(root, "estimator_comparison.csv"),
                     index=False)
        fl_cmp.to_csv(os.path.join(root, "floor_comparison.csv"),
                      index=False)
    make_figures(det_s, wt, root)

    path = write_report(root, report, det_s, diff, rank, ok_rank, wt,
                        floors, lapse_tab, cells, fl_cmp, n_draws, t0)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in report:
        print(f"  {k:34s} {v}")
    print(f"\n  Report : {path}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print("\n  THE DECISION RULE WAS FIXED BEFORE THIS RAN: `both` "
          "ships if the ranking\n  and the gains hold and the weights "
          "are stable; otherwise sens_cut_25_5.\n  This script reports "
          "and stops.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
