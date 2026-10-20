"""
analyse_study_v4.py

The registered analysis of the derealization image-choice study (v4),
run on one dataset, real or simulated.

Two layers
----------
  LOADER     reads the session data and produces three tidy tables:
               trials    one row per presented trial
               sessions  one row per session: session fields, exclusion
                         facts, word ratings, state items
               grid      one row per grid screen
             INPUT_FORMAT = "tidy" reads tables already in that form (what
             simulate_study_data_v4.py writes). INPUT_FORMAT = "raw", the
             experiment's own session CSVs, is finished once the experiment
             update is merged and DATA_DICTIONARY.md describes the new
             format; until then it stops with a message.
  ANALYSIS   reads only the tidy tables, the frozen trial list, the
             registration table, the grid package and the two private files.

What it does, in the registered order
-------------------------------------
  A  who is analysable: test and development sessions, the first complete
     session per ID, consent, images that did not fit, 3 or more catch
     failures; Tier D set aside and reported separately; beyond the
     maximum (the first 20 participants and 30 controls by completion
     time are analysed, the rest listed); partial exclusions per analysis
  B  per person: Firth logistic regression, no intercept,
     chose A ~ pred_sat + pred_fog over the 180 analysed trials; 95%
     profile penalised-likelihood intervals; classification against the
     registered thresholds
  C  H1 (fog) and H2 (colour): the registered (k, m) rule for the final N
  D  C1: the attention check (catch failures, fast responses; one-sided
     Mann-Whitney, Holm, alpha 0.05), then one-sided Mann-Whitney per
     feature, Holm; the zero-catch-failure test beside it
  E  secondary: one-sided t-test and one-sided JZS Bayes factor on the
     weights, with and without the cap
  F  exploratory and descriptive: prevalence under three sensitivities,
     the Suzuki-style counts, H3 (the weight angle), E1 (descriptions
     against weights), E2 (the grid shift), dose-response and additivity,
     quality and covariates
  G  outputs: report, tables, figures, run_meta.json; shareable outputs
     use pseudonyms (P01.., C01.., D01..; excluded people XP01.., XC01..),
     and the key is written to PRIVATE/, clearly named

What is copied, and why
-----------------------
The registered error rates were computed with simulate_power_v4_n10_20.py,
so every function this analysis shares with it is COPIED VERBATIM and
marked as a copy. At start-up the script checks the copies against
values the power script itself produced on a fixed dataset and, when
POWER_SCRIPT is present, against the power script run now, function by
function and source by source. Two things the power script describes but
does not compute are this script's own code and are labelled as such:
the cap (built exactly as the power script's SETTINGS describe it) and
the profile penalised-likelihood intervals.

Checking against simulated data
-------------------------------
RUN_MODE = "validate_sim" analyses every dataset written by
simulate_study_data_v4.py, compares each result with that dataset's truth
files, runs the false-claim check on the simulated no-effect batches, and
writes a validation summary. Use it after any change to this script.

Usage
-----
    python analyse_study_v4.py

No arguments. Every setting is in SETTINGS below. Paths are relative to
the project folder: scripts at its top level, data under data/.
"""

import ast
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
import time
import warnings
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.integrate import quad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker


# ============================================================
# SETTINGS
# ============================================================

# --- What to analyse ---
#
# RUN_MODE
#   "one"           analyse DATASET_DIR. For real data, this is the mode.
#   "validate_sim"  analyse every dataset under SIM_ROOT/datasets, compare
#                   each with its truth files, and run the false-claim
#                   check on SIM_ROOT/false_claim.
RUN_MODE = "validate_sim"

# DATASET_DIR
#   For RUN_MODE = "one": the folder holding tidy/ (or, for
#   INPUT_FORMAT = "raw", the folder of session CSVs). When it also holds
#   truth.json (a simulated dataset), the result is checked against it.
DATASET_DIR = "data/sim_study_v4/datasets/messy"

# INPUT_FORMAT
#   "tidy"  DATASET_DIR/tidy/{trials,sessions,grid}.csv
#   "raw"   the experiment's session CSVs, in DATASET_DIR. NOT YET
#           AVAILABLE: finished after the experiment update is merged.
INPUT_FORMAT = "tidy"

# SURVEY_FILE / PARTICIPANT_TABLE
#   The two private files. For a simulated dataset they are the fakes in
#   DATASET_DIR/private/. For real data, set the paths to the files on
#   this computer; they are read here and never copied anywhere.
#   The survey is an .xlsx with ResponseId and Q01_score .. Q29_score;
#   extra header rows (as some survey exports write) are detected and
#   dropped. The participant table is .csv or .xlsx with participant_id,
#   group (participant / control) and tier (A-D, empty for controls).
SURVEY_FILE = None              # None -> DATASET_DIR/private/survey_FAKE.xlsx
PARTICIPANT_TABLE = None        # None -> DATASET_DIR/private/participant_table_FAKE.csv

# SIM_ROOT
#   For RUN_MODE = "validate_sim": the simulation run to validate against.
SIM_ROOT = "data/sim_study_v4"

# --- Inputs (relative to the project folder) ---
FROZEN_ROOT = "data/frozen_trial_list_v4"
FROZEN_TRIAL_LIST = "trial_list.csv"
COMBINED_RULE = "data/registration_v4/combined_rule.csv"
GRID_TRIALS = "data/grid_block_v4/package/grid_trials.csv"
DETECTION_CURVE = "data/power_v4_n10_20/frozen/detection_curve.csv"

# POWER_SCRIPT
#   Optional. When present the start-up check also runs the power
#   script's own functions and compares every copy's source with it.
POWER_SCRIPT = "simulate_power_v4_n10_20.py"

# --- Output ---
#
# OUTPUT_ROOT
#   Never overwritten: a rerun writes to OUTPUT_ROOT_2, _3, ...
#   In "validate_sim" mode the root is OUTPUT_ROOT + "_validation".
OUTPUT_ROOT = "data/analysis_v4"
MAKE_FIGURES = True

# --- What is registered, stated so the run can check it ---
REGISTERED_THRESHOLDS = {"colour": (0.286, -0.290), "fog": (0.488, -0.486)}
REGISTERED_RULES = {
    "colour": {10: (4, 0), 11: (5, 2), 12: (5, 2), 13: (5, 1),
               14: (5, 1), 15: (6, 3), 16: (6, 2), 17: (6, 2),
               18: (6, 2), 19: (7, 3), 20: (7, 3)},
    "fog":    {10: (4, 0), 11: (5, 2), 12: (5, 2), 13: (5, 1),
               14: (5, 1), 15: (5, 1), 16: (5, 1), 17: (5, 1),
               18: (5, 1), 19: (5, 1), 20: (5, 1)},
}
# REGISTERED_CAP
#   The cap as it will be stated in the preregistration (3 dp). The cap
#   is recomputed on the frozen list and must round to these.
REGISTERED_CAP = {"colour": 8.378, "fog": 10.269}

# PREVALENCE_SENSITIVITIES
#   (label, assumed weight per SD); the sensitivity is read from the
#   power run's detection curve at lapse 0, by linear interpolation, as
#   the power script does. REGISTERED_SENSITIVITIES is what it must give
#   (2 dp, docs/v4_stimulus_power_explained.md section 12).
PREVALENCE_SENSITIVITIES = [("weak", 0.30), ("moderate", 0.50),
                            ("strong", 0.70)]
REGISTERED_SENSITIVITIES = {"colour": (0.50, 0.86, 0.99),
                            "fog": (0.22, 0.51, 0.79)}

N_MIN_CLAIM = 10          # below this, H1 and H2 are descriptive
# N_MAX_PARTICIPANTS / N_MAX_CONTROLS
#   The recruitment maxima. If more people than this pass every
#   exclusion, the first N_MAX by session COMPLETION time
#   (session_end_time; ties by session id) are analysed and the rest are
#   listed as beyond the maximum (tables/beyond_maximum.csv), not
#   analysed. Tier D does not count toward N and is not capped.
N_MAX_PARTICIPANTS = 20
N_MAX_CONTROLS = 30
MIN_CONTROLS_C1 = 20      # below this, C1 is descriptive
CATCH_FAIL_EXCLUDE = 3    # 3 or more of the 10 catch trials failed

# --- Exploratory settings ---
SUZUKI_FOLLOW = 39            # of 60 single-feature trials
WORD_IDS = ["foggy", "faded", "veiled", "lifeless", "unreal",
            "dreamlike", "distant", "like_a_picture", "oppressive",
            "threatening"]
WORD_BASELINE = ["unreal", "dreamlike", "distant", "like_a_picture"]
STATE_ITEMS = ["state_q02", "state_q13", "state_q19", "state_q29"]
# Both rating blocks use 0-6 (0 = not at all, 6 = exactly / completely);
# a value outside that range, or not a whole number, stops the run.
WORD_MAX = 6
STATE_ITEM_MAX = 6
CDS_ITEMS = 29
AFS_ITEMS = [2, 13, 19, 29]
# E1: (description, weight) pairs; Spearman with a percentile bootstrap.
E1_PAIRS = [("cds_q13", "w_fog", "CDS item 13 vs fog weight"),
            ("cds_q02", "w_colour", "CDS item 2 vs colour weight "
                                    "(reported, not a colour test)"),
            ("foggy_rel", "w_fog", "foggy minus baseline vs fog weight"),
            ("faded_rel", "w_colour",
             "faded minus baseline vs colour weight")]
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261021
CI_LEVEL = 0.95

# PROFILE_SEARCH_MAX
#   How far (per SD) from the estimate the profile interval is searched.
#   A bound not reached within it is reported as unbounded, which is
#   what a person answering every informative trial one way produces.
PROFILE_SEARCH_MAX = 50.0
PROFILE_XTOL = 1e-8

# LOCAL_TZ
#   Session times are recorded in UTC; the interview date is a local
#   date. The task date is the session start in this time zone.
LOCAL_TZ = "Asia/Tokyo"

TIDY_TRIALS = "trials.csv"
TIDY_SESSIONS = "sessions.csv"
TIDY_GRID = "grid.csv"
TEST_IDS = ["Z_21121989", "CHAIN_2026"]
ID_PATTERN = r"^R_[A-Za-z0-9]{15}$"

# Figures: the power script's Okabe-Ito order, checked for colour-vision
# separation (worst adjacent pair dE 11.0 under deuteranopia). Groups also
# differ by marker, so colour never carries identity alone.
GROUP_STYLE = {"participant": ("#0072B2", "o", "participants"),
               "control": ("#D55E00", "s", "controls"),
               "tier_D": ("#009E73", "^", "Tier D")}
FIG_DPI = 150

# --- The start-up identity check (see simulate_study_data_v4.py) ---
IDENTITY_SEED = 20261020
IDENTITY_W = [[0.0, 0.0], [0.5, 0.0], [0.0, 0.7], [0.5, 0.7],
              [-0.3, 0.2], [1.5, -0.4], [3.0, 3.0], [0.29, 0.49]]
IDENTITY_ABS_TOL = 1e-9
IDENTITY_REL_TOL = 1e-9

# MW_AGREE_TOL
#   The C1 p-values come from mw_p_batch (what the power script
#   simulated); the scipy-based mann_whitney_one_sided must agree to this.
MW_AGREE_TOL = 1e-9

# Agreement with a simulated dataset's truth (validate_sim, or a
# simulated DATASET_DIR): weights to this tolerance; everything else
# exactly.
TRUTH_WEIGHT_TOL = 1e-7


# ============================================================
# COPIED SETTINGS
#
# Values identical to simulate_power_v4_n10_20.py. The copied functions
# read them as globals; when POWER_SCRIPT is present the start-up check
# compares every one of them with the power script's own value.
# ============================================================

UNHASHED = ("checksums.sha256", "FROZEN.json", "freeze_report.txt")
FROZEN_SEED = 42
PRED_SAT = "pred_sat"
PRED_FOG = "pred_fog"
SD_SAT = 39.182
SD_FOG = 9.348
PRED_REBUILD_TOL = 1e-9
ESTIMATOR = "firth"
FIRTH_ITER = 200
FIRTH_TOL = 1e-10
FIRTH_MAX_STEP = 5.0
CAP_INFORMATIVE_SD = 0.20
CAP_THRESHOLD_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
ALPHA = 0.05
ONE_TAILED = True
BF_SCALE = 0.707
BF_ONE_SIDED = True
BF_THRESHOLD = 3.0
BF_FORM_TOL = 1e-6
FLOOR_ALPHA = 0.025
N_FLOOR_PEOPLE = 20_000
DETECTION_SIDED = "two"
PREVALENCE_HPDI = 0.95
PREVALENCE_N_GRID = 201
PREVALENCE_C_GRID = 201
PREVALENCE_ZERO_EPS = 0.001
C1_EXACT_MAX_N = 20
C1_HOLM = True
C1_ZERO_MIN_N = 4
GATE_ALPHA = 0.05
N_CATCH_TRIALS = 10
CATCH_P_FAIL_PER_LAPSE = 0.75
LAPSE_BETWEEN_SD = 0.5
RT_GATE = True
RT_FAST_CUTOFF_MS = 300
RT_FAST_GIVEN_LAPSE = 0.4
RT_FAST_GIVEN_ENGAGED = 0.005
RANDOM_SEED = 42

COPIED_SETTINGS = [
    "UNHASHED", "FROZEN_SEED", "PRED_SAT", "PRED_FOG", "SD_SAT", "SD_FOG",
    "PRED_REBUILD_TOL", "ESTIMATOR", "FIRTH_ITER", "FIRTH_TOL",
    "FIRTH_MAX_STEP", "CAP_INFORMATIVE_SD", "CAP_THRESHOLD_SWEEP", "ALPHA",
    "ONE_TAILED", "BF_SCALE", "BF_ONE_SIDED", "BF_THRESHOLD", "BF_FORM_TOL",
    "FLOOR_ALPHA", "N_FLOOR_PEOPLE", "DETECTION_SIDED", "PREVALENCE_HPDI",
    "PREVALENCE_N_GRID", "PREVALENCE_C_GRID", "PREVALENCE_ZERO_EPS",
    "C1_EXACT_MAX_N", "C1_HOLM", "C1_ZERO_MIN_N", "GATE_ALPHA",
    "N_CATCH_TRIALS", "CATCH_P_FAIL_PER_LAPSE", "LAPSE_BETWEEN_SD",
    "RT_GATE", "RT_FAST_CUTOFF_MS", "RT_FAST_GIVEN_LAPSE",
    "RT_FAST_GIVEN_ENGAGED", "RANDOM_SEED",
]

COPIED_FUNCTIONS = [
    "fit_logit_batch", "fit_firth", "fit_firth_batch", "fit_batch",
    "penalised_loglik", "simulate_many", "lapse_p", "simulate_lapsing",
    "_jzs_g_form", "jzs_bayes_factor", "check_bayes_factor", "sha256_of",
    "verify_frozen", "load_trials", "informative_medians", "as_probability",
    "compute_floors", "classify", "rule_mask_count",
    "prevalence_posterior_shared", "prevalence_moment_raw",
    "prevalence_moment", "hpdi", "mw_exact_sf", "mw_p_batch",
    "mann_whitney_one_sided", "holm_two", "_c1_gate", "_c1_test",
    "_c1_zero_catch", "draw_attention",
]


# ============================================================
# COPIED FUNCTIONS
#
# VERBATIM from simulate_power_v4_n10_20.py, each marked with its source
# lines. Do not edit them here: the start-up check compares each one
# with the original, and a copy that differs stops the run.
# ============================================================

# COPY of simulate_power_v4_n10_20.py, lines 1928-1954.
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


# COPY of simulate_power_v4_n10_20.py, lines 1957-1989.
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


# COPY of simulate_power_v4_n10_20.py, lines 1992-2037.
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


# COPY of simulate_power_v4_n10_20.py, lines 2238-2245.
def fit_batch(X, Y):
    """Many participants, under the registered estimator."""
    if ESTIMATOR == "firth":
        return fit_firth_batch(X, Y)[0]
    if ESTIMATOR == "mle":
        return fit_logit_batch(X, Y)
    raise ValueError(
        f"ESTIMATOR must be 'firth' or 'mle', got {ESTIMATOR!r}")


# COPY of simulate_power_v4_n10_20.py, lines 2040-2052.
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


# COPY of simulate_power_v4_n10_20.py, lines 2248-2251.
def simulate_many(X, W, rng):
    """Choices of many people at once. W is (n_people, 2) true weights."""
    p = 1.0 / (1.0 + np.exp(-(W @ X.T)))
    return (rng.random(p.shape) < p).astype(float)


# COPY of simulate_power_v4_n10_20.py, lines 2254-2270.
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


# COPY of simulate_power_v4_n10_20.py, lines 2273-2276.
def simulate_lapsing(X, W, lapse, rng):
    """Choices of many people at once, each lapsing at rate `lapse`."""
    p = lapse_p(X, W, lapse)
    return (rng.random(p.shape) < p).astype(float)


# COPY of simulate_power_v4_n10_20.py, lines 2279-2304.
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


# COPY of simulate_power_v4_n10_20.py, lines 2307-2357.
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


# COPY of simulate_power_v4_n10_20.py, lines 2360-2425.
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


# COPY of simulate_power_v4_n10_20.py, lines 2432-2441.
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


# COPY of simulate_power_v4_n10_20.py, lines 2444-2589.
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


# COPY of simulate_power_v4_n10_20.py, lines 2592-2656.
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


# COPY of simulate_power_v4_n10_20.py, lines 2663-2685.
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


# COPY of simulate_power_v4_n10_20.py, lines 2688-2695.
def as_probability(weight, difference):
    """A weight, as the probability of choosing the predicted image on
    a trial of that difference.

    The readable form. `weight` is per corpus SD and `difference` is
    in SD, so their product is the log-odds.
    """
    return float(1.0 / (1.0 + np.exp(-weight * difference)))


# COPY of simulate_power_v4_n10_20.py, lines 2698-2784.
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


# COPY of simulate_power_v4_n10_20.py, lines 2787-2804.
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


# COPY of simulate_power_v4_n10_20.py, lines 3297-3301.
def rule_mask_count(n, k, m):
    """Boolean (a, b) mask: at least k predicted, at most m opposite."""
    a = np.arange(n + 1)[:, None]
    b = np.arange(n + 1)[None, :]
    return (a >= k) & (b <= m)


# COPY of simulate_power_v4_n10_20.py, lines 4500-4572.
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


# COPY of simulate_power_v4_n10_20.py, lines 4575-4581.
def prevalence_moment_raw(a, b, n, sensitivity):
    """The moment estimator BEFORE clipping, which is the unbiased
    one. Reported beside the clipped version so the unbiasedness is
    checkable rather than asserted."""
    if sensitivity <= 0:
        return np.nan
    return float((a - b) / (n * sensitivity))


# COPY of simulate_power_v4_n10_20.py, lines 4584-4599.
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


# COPY of simulate_power_v4_n10_20.py, lines 4602-4613.
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


# COPY of simulate_power_v4_n10_20.py, line 5389.
_MW_SF_CACHE = {}


# COPY of simulate_power_v4_n10_20.py, lines 5392-5425.
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


# COPY of simulate_power_v4_n10_20.py, lines 5428-5501.
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


# COPY of simulate_power_v4_n10_20.py, lines 5504-5523.
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


# COPY of simulate_power_v4_n10_20.py, lines 5526-5543.
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


# COPY of simulate_power_v4_n10_20.py, lines 5611-5647.
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


# COPY of simulate_power_v4_n10_20.py, lines 5671-5687.
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


# COPY of simulate_power_v4_n10_20.py, lines 5690-5776.
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


# COPY of simulate_power_v4_n10_20.py, lines 5779-5804.
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


# ============================================================
# IDENTITY_REFERENCE
#
# What simulate_power_v4_n10_20.py's OWN functions produced on the fixed
# identity dataset (IDENTITY_SEED, IDENTITY_W, the frozen list), computed
# 2026-10-20 with Python 3.13.16, numpy 2.5.3, scipy 1.18.1.
# The digests of simulated choices depend on numpy's random streams; if
# a different numpy changes a stream, these differ and the run stops --
# and the power script's registered thresholds would then need checking
# on that numpy too.
# ============================================================

IDENTITY_REFERENCE = {
    '_c1_gate/catch': 'FFFFT',
    '_c1_gate/fired': 'TFTFT',
    '_c1_gate/rt': 'TFTFF',
    '_c1_test/colour': 'FTTFF',
    '_c1_test/fog': 'FFFFF',
    '_c1_zero_catch/n': [4.0, 5.0, 2.0, 4.0, 6.0, 5.0, 7.0, 6.0, 3.0, 4.0],
    '_c1_zero_catch/reject_colour': 'FFFFF',
    '_c1_zero_catch/reject_fog': 'FFFFF',
    '_c1_zero_catch/rt_fires': 'TFFFF',
    '_c1_zero_catch/usable': 'TTFFT',
    '_jzs_g_form/2.1,10': 1.4456188579096265,
    'as_probability/0.5,0.97': 0.6189278548001903,
    'classify/opposite/colour': 500.0,
    'classify/opposite/fog': 500.0,
    'classify/predicted/colour': 500.0,
    'classify/predicted/fog': 500.0,
    'compute_floors/lower/colour': -0.2896056357511166,
    'compute_floors/lower/fog': -0.48559721218587304,
    'compute_floors/null_mean/colour': -0.0008310777540807824,
    'compute_floors/null_mean/fog': 0.0007845519680728465,
    'compute_floors/sd/colour': 0.1463828989550224,
    'compute_floors/sd/fog': 0.2467603087799088,
    'compute_floors/upper/colour': 0.2863223717155787,
    'compute_floors/upper/fog': 0.4880483057825434,
    'draw_attention/catch_sha256': '7ceb8b64c86e853afce124af9a6561e645d477fb07b0880afd80d36e9f372e3b',
    'draw_attention/fast_sum': 2.2611111111111115,
    'draw_attention/lapse_sum': 5.4037404488652925,
    'fit_batch/0/colour': 0.22286672374858105,
    'fit_batch/0/fog': 0.016689801579631798,
    'fit_batch/1/colour': 0.49604541171474625,
    'fit_batch/1/fog': -0.0016736860652614355,
    'fit_batch/2/colour': -0.0200935636404387,
    'fit_batch/2/fog': 0.5870986019640844,
    'fit_batch/3/colour': 0.38740410475173026,
    'fit_batch/3/fog': 0.9306131889740917,
    'fit_batch/4/colour': -0.1576748944524751,
    'fit_batch/4/fog': -0.14769331456940193,
    'fit_batch/5/colour': 0.9667252625391509,
    'fit_batch/5/fog': -0.5063622568924957,
    'fit_batch/6/colour': 2.140861514971723,
    'fit_batch/6/fog': 1.2287162688283497,
    'fit_batch/7/colour': 0.5216060663101221,
    'fit_batch/7/fog': 0.18513815764394095,
    'fit_firth/0/colour': 0.22286672374863856,
    'fit_firth/0/fog': 0.016689801579676315,
    'fit_firth/1/colour': 0.4960454117146191,
    'fit_firth/1/fog': -0.0016736860653579092,
    'fit_firth/2/colour': -0.02009356364047454,
    'fit_firth/2/fog': 0.5870986019640364,
    'fit_firth/3/colour': 0.3874041047517441,
    'fit_firth/3/fog': 0.9306131889741246,
    'fit_firth/4/colour': -0.15767489445257168,
    'fit_firth/4/fog': -0.14769331456947749,
    'fit_firth/5/colour': 0.9667252625393563,
    'fit_firth/5/fog': -0.5063622568917688,
    'fit_firth/6/colour': 2.140861514971723,
    'fit_firth/6/fog': 1.22871626882835,
    'fit_firth/7/colour': 0.5216060663100633,
    'fit_firth/7/fog': 0.1851381576439,
    'fit_firth_batch/0/colour': 0.22286672374858105,
    'fit_firth_batch/0/fog': 0.016689801579631798,
    'fit_firth_batch/1/colour': 0.49604541171474625,
    'fit_firth_batch/1/fog': -0.0016736860652614355,
    'fit_firth_batch/2/colour': -0.0200935636404387,
    'fit_firth_batch/2/fog': 0.5870986019640844,
    'fit_firth_batch/3/colour': 0.38740410475173026,
    'fit_firth_batch/3/fog': 0.9306131889740917,
    'fit_firth_batch/4/colour': -0.1576748944524751,
    'fit_firth_batch/4/fog': -0.14769331456940193,
    'fit_firth_batch/5/colour': 0.9667252625391509,
    'fit_firth_batch/5/fog': -0.5063622568924957,
    'fit_firth_batch/6/colour': 2.140861514971723,
    'fit_firth_batch/6/fog': 1.2287162688283497,
    'fit_firth_batch/7/colour': 0.5216060663101221,
    'fit_firth_batch/7/fog': 0.18513815764394095,
    'fit_logit_batch/0/colour': 0.2271851150897008,
    'fit_logit_batch/0/fog': 0.017666778919130016,
    'fit_logit_batch/1/colour': 0.5059669742679315,
    'fit_logit_batch/1/fog': -0.00024347505040727189,
    'fit_logit_batch/2/colour': -0.01971818114167136,
    'fit_logit_batch/2/fog': 0.5964710175695054,
    'fit_logit_batch/3/colour': 0.3960258068482941,
    'fit_logit_batch/3/fog': 0.9464972354074183,
    'fit_logit_batch/4/colour': -0.16089950559945929,
    'fit_logit_batch/4/fog': -0.1505517881407587,
    'fit_logit_batch/5/colour': 0.9875132325086207,
    'fit_logit_batch/5/fog': -0.5120769121358539,
    'holm_two/0.001,0.2': 'TF',
    'holm_two/0.01,0.04': 'TT',
    'holm_two/0.03,0.04': 'FF',
    'holm_two/0.2,0.024': 'FT',
    'informative_medians/colour': 0.9713604453755071,
    'informative_medians/colour_n': 120.0,
    'informative_medians/fog': 0.6416750913202939,
    'informative_medians/fog_n': 120.0,
    'jzs_bayes_factor/-1.0,15/one': 0.14663090568264842,
    'jzs_bayes_factor/-1.0,15/two': 0.40292888263711385,
    'jzs_bayes_factor/0.5,20/one': 0.35355222878647574,
    'jzs_bayes_factor/0.5,20/two': 0.2600153765676458,
    'jzs_bayes_factor/2.1,10/one': 2.763762221888209,
    'jzs_bayes_factor/2.1,10/two': 1.4456188579096574,
    'jzs_bayes_factor/3.4,12/one': 18.04275852902397,
    'jzs_bayes_factor/3.4,12/two': 9.06818484168585,
    'lapse_p/lapse0.2/[3,17]': 0.599771613299786,
    'lapse_p/lapse0.2/sum': 736.3802812058475,
    'lapse_p/lapse0/[3,17]': 0.6247145166247325,
    'lapse_p/lapse0/sum': 740.4753515073091,
    'load_trials/X_sha256': 'ecb4c8a53546a5b10535cb19f5baac3e21dc4e3a1673c5a9c2d98131f2dd20ee',
    'mann_whitney_one_sided/exact/0': [70.0, 0.04894022386282448],
    'mann_whitney_one_sided/exact/0/method': 'exact',
    'mann_whitney_one_sided/exact/1': [54.0, 0.3392156862745098],
    'mann_whitney_one_sided/exact/1/method': 'exact',
    'mann_whitney_one_sided/ties/0': [134.0, 0.06239791329212847],
    'mann_whitney_one_sided/ties/0/method': 'asymptotic',
    'mann_whitney_one_sided/ties/1': [131.5, 0.07631931163686224],
    'mann_whitney_one_sided/ties/1/method': 'asymptotic',
    'mw_exact_sf/5,6/[20]': 0.21428571428571427,
    'mw_exact_sf/5,6/sum': 16.0,
    'mw_p_batch/asym/method': 'asymptotic',
    'mw_p_batch/asym/p': [0.002140310584960722, 0.07078690058441392, 0.24403704658262404, 0.014223516488281518, 0.08707084160752154],
    'mw_p_batch/exact/method': 'exact',
    'mw_p_batch/exact/p': [0.04894022386282448, 0.3392156862745098, 0.04124791617051679, 0.23633404778915615, 0.04894022386282448],
    'mw_p_batch/ties/method': 'asymptotic',
    'mw_p_batch/ties/p': [0.06239791329212847, 0.07631931163686224, 0.30006595575169154, 0.11095632451944115, 0.0043147765520927354],
    'penalised_loglik/0': -120.24370429500797,
    'penalised_loglik/1': -116.02897273387214,
    'penalised_loglik/2': -118.63555894522017,
    'penalised_loglik/3': -110.78872720550007,
    'penalised_loglik/4': -120.57723774196428,
    'penalised_loglik/5': -103.94020300467484,
    'penalised_loglik/6': -72.12537201026973,
    'penalised_loglik/7': -114.9877880531155,
    'prevalence/0,0,10,0.22/hpdi_hi': 0.78,
    'prevalence/0,0,10,0.22/hpdi_lo': 0.0,
    'prevalence/0,0,10,0.22/mode': 0.005,
    'prevalence/0,0,10,0.22/moment': 0.0,
    'prevalence/0,0,10,0.22/moment_raw': 0.0,
    'prevalence/0,0,10,0.22/post_sum': 201.40486603125584,
    'prevalence/3,4,15,0.51/hpdi_hi': 0.54,
    'prevalence/3,4,15,0.51/hpdi_lo': 0.0,
    'prevalence/3,4,15,0.51/mode': 0.0,
    'prevalence/3,4,15,0.51/moment': 0.0,
    'prevalence/3,4,15,0.51/moment_raw': -0.13071895424836602,
    'prevalence/3,4,15,0.51/post_sum': 201.6367367341002,
    'prevalence/5,1,12,0.86/hpdi_hi': 0.63,
    'prevalence/5,1,12,0.86/hpdi_lo': 0.0,
    'prevalence/5,1,12,0.86/mode': 0.32,
    'prevalence/5,1,12,0.86/moment': 0.38759689922480617,
    'prevalence/5,1,12,0.86/moment_raw': 0.38759689922480617,
    'prevalence/5,1,12,0.86/post_sum': 200.30119131175644,
    'prevalence/9,0,20,0.5/hpdi_hi': 1.0,
    'prevalence/9,0,20,0.5/hpdi_lo': 0.385,
    'prevalence/9,0,20,0.5/mode': 0.79,
    'prevalence/9,0,20,0.5/moment': 0.9,
    'prevalence/9,0,20,0.5/moment_raw': 0.9,
    'prevalence/9,0,20,0.5/post_sum': 200.70972969754033,
    'rule_mask_count/10,4,0/shape': 11.0,
    'rule_mask_count/10,4,0/sum': 7.0,
    'rule_mask_count/12,5,2/shape': 13.0,
    'rule_mask_count/12,5,2/sum': 24.0,
    'rule_mask_count/20,7,3/shape': 21.0,
    'rule_mask_count/20,7,3/sum': 56.0,
    'simulate_lapsing/sha256': 'f4e368fd878343f913ab2258dfcb7a0785e79e75a7216a663fa4be8764d2036f',
    'simulate_many/sha256': '653b845a5e48a86ed72be5fc4719df890f9441f05f95ec95840a2c718cbb1c24',
}


# ============================================================
# Small helpers  (this script's own code from here on)
# ============================================================

class CheckFailed(RuntimeError):
    """A check that means the run cannot be trusted. Stops the run."""


class Checks:
    """Every check the run makes, in order, with its verdict.

    `require` stops the run on failure; `note` records a result that is
    a property of the data (for example a chance false claim in one
    simulated draw) and does not stop it; `info` records a fact.
    """

    def __init__(self, prefix=""):
        self.rows, self.prefix = [], prefix

    def _add(self, name, result, stops, detail):
        name = f"{self.prefix}{name}"
        self.rows.append({"check": name, "result": result,
                          "stops_run": stops, "detail": str(detail)})
        print(f"  [{result}] {name}" + (f"  ({detail})" if detail else ""))

    def require(self, name, ok, detail=""):
        self._add(name, "PASS" if ok else "FAIL", True, detail)
        if not ok:
            raise CheckFailed(f"{self.prefix}{name}: {detail}")

    def note(self, name, ok, detail=""):
        self._add(name, "PASS" if ok else "FAIL", False, detail)

    def info(self, name, detail=""):
        self._add(name, "INFO", False, detail)


def resolve_output_root(base):
    """base, or the first free base_<n>. Never overwrites."""
    def occupied(d):
        return os.path.exists(d) and (not os.path.isdir(d)
                                      or len(os.listdir(d)) > 0)
    if not occupied(base):
        return base
    n = 2
    while occupied(f"{base}_{n}"):
        n += 1
    return f"{base}_{n}"


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if not np.isfinite(x) else x
    return x


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(obj), fh, indent=1, ensure_ascii=False)


def settings_snapshot():
    out = {}
    for k, v in list(globals().items()):
        if (k.isupper() and not k.startswith("_") and not callable(v)
                and not isinstance(v, type(sys)) and k != "IDENTITY_REFERENCE"):
            try:
                json.dumps(jsonable(v))
                out[k] = jsonable(v)
            except TypeError:
                out[k] = repr(v)
    return out


def to_bool(s):
    """A CSV boolean column, which may arrive as bool, str or NaN."""
    m = {True: True, False: False, "True": True, "False": False,
         "true": True, "false": False, "TRUE": True, "FALSE": False,
         1: True, 0: False, "1": True, "0": False}
    return s.map(lambda v: m.get(v, np.nan) if not (isinstance(v, float)
                                                    and np.isnan(v))
                 else np.nan)


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA" if x is None or np.isnan(x) else (
            "+inf" if x > 0 else "-inf")
    return f"{x:.{d}f}"


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "NA"
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


# ============================================================
# The start-up identity check
# ============================================================

def identity_values(ns, X):
    """What the copied functions produce on one fixed dataset.

    Called with this script's own copies (globals()) and, when the power
    script is present, with the power script's module namespace; each
    function then runs with its own module's settings. Returns a flat
    dict of quantity -> value, plus "_th" (compute_floors' thresholds).
    """
    g = (lambda k: ns[k]) if isinstance(ns, dict) else \
        (lambda k: getattr(ns, k))
    v = {}

    def h(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

    def flags(a):
        return "".join("T" if x else "F" for x in np.ravel(a))

    W = np.array(IDENTITY_W, dtype=float)
    v["load_trials/X_sha256"] = h(np.round(X, 12))
    Y = g("simulate_lapsing")(X, W, 0.1, np.random.default_rng(IDENTITY_SEED))
    v["simulate_lapsing/sha256"] = h(Y.astype(np.uint8))
    Y0 = g("simulate_many")(X, W, np.random.default_rng(IDENTITY_SEED + 1))
    v["simulate_many/sha256"] = h(Y0.astype(np.uint8))
    for lam in (0.0, 0.2):
        P = g("lapse_p")(X, W, lam)
        v[f"lapse_p/lapse{lam:g}/sum"] = float(P.sum())
        v[f"lapse_p/lapse{lam:g}/[3,17]"] = float(P[3, 17])

    b, last = g("fit_firth_batch")(X, Y)
    fb = g("fit_batch")(X, Y)
    for i in range(len(W)):
        bi = g("fit_firth")(X, Y[i])
        for j, f in enumerate(("colour", "fog")):
            v[f"fit_firth_batch/{i}/{f}"] = float(b[i, j])
            v[f"fit_firth/{i}/{f}"] = float(bi[j])
            v[f"fit_batch/{i}/{f}"] = float(fb[i, j])
        v[f"penalised_loglik/{i}"] = float(
            g("penalised_loglik")(X, Y[i], bi))
    lb = g("fit_logit_batch")(X, Y[:6])
    for i in range(6):
        for j, f in enumerate(("colour", "fog")):
            v[f"fit_logit_batch/{i}/{f}"] = float(lb[i, j])

    th = g("compute_floors")(X, "identity check", [])
    for j, f in enumerate(("colour", "fog")):
        v[f"compute_floors/upper/{f}"] = float(th["upper"][j])
        v[f"compute_floors/lower/{f}"] = float(th["lower"][j])
        v[f"compute_floors/sd/{f}"] = float(th["sd"][j])
        v[f"compute_floors/null_mean/{f}"] = float(th["null"][:, j].mean())
    pr, op = g("classify")(th["null"], th)
    for j, f in enumerate(("colour", "fog")):
        v[f"classify/predicted/{f}"] = float(pr[:, j].sum())
        v[f"classify/opposite/{f}"] = float(op[:, j].sum())
    for (n, k, m) in ((10, 4, 0), (12, 5, 2), (20, 7, 3)):
        mk = g("rule_mask_count")(n, k, m)
        v[f"rule_mask_count/{n},{k},{m}/sum"] = float(mk.sum())
        v[f"rule_mask_count/{n},{k},{m}/shape"] = float(mk.shape[0])
    med = g("informative_medians")(X)
    for k_ in sorted(med):
        v[f"informative_medians/{k_}"] = float(med[k_])
    v["as_probability/0.5,0.97"] = float(g("as_probability")(0.5, 0.97))

    for (t, n) in ((2.1, 10), (-1.0, 15), (0.5, 20), (3.4, 12)):
        v[f"jzs_bayes_factor/{t},{n}/one"] = float(
            g("jzs_bayes_factor")(t, n, one_sided=True))
        v[f"jzs_bayes_factor/{t},{n}/two"] = float(
            g("jzs_bayes_factor")(t, n, one_sided=False))
    v["_jzs_g_form/2.1,10"] = float(g("_jzs_g_form")(2.1, 10))

    for (a, b_, n, s) in ((5, 1, 12, 0.86), (0, 0, 10, 0.22),
                          (9, 0, 20, 0.5), (3, 4, 15, 0.51)):
        key = f"prevalence/{a},{b_},{n},{s}"
        gg, post = g("prevalence_posterior_shared")(a, b_, n, s)
        lo, hi = g("hpdi")(gg, post)
        v[key + "/post_sum"] = float(post.sum())
        v[key + "/mode"] = float(gg[int(np.argmax(post))])
        v[key + "/hpdi_lo"] = float(lo)
        v[key + "/hpdi_hi"] = float(hi)
        v[key + "/moment"] = float(g("prevalence_moment")(a, b_, n, s))
        v[key + "/moment_raw"] = float(
            g("prevalence_moment_raw")(a, b_, n, s))

    lam, catch, fast = g("draw_attention")(
        40, 0.15, X.shape[0], np.random.default_rng(IDENTITY_SEED + 2))
    v["draw_attention/lapse_sum"] = float(lam.sum())
    v["draw_attention/catch_sha256"] = h(catch.astype(np.int64))
    v["draw_attention/fast_sum"] = float(fast.sum())

    sf = g("mw_exact_sf")(5, 6)
    v["mw_exact_sf/5,6/sum"] = float(sf.sum())
    v["mw_exact_sf/5,6/[20]"] = float(sf[20])
    r4 = np.random.default_rng(IDENTITY_SEED + 3)
    A, B = r4.normal(size=(5, 8)) + 0.4, r4.normal(size=(5, 12))
    p, meth = g("mw_p_batch")(A, B)
    v["mw_p_batch/exact/p"] = [float(x) for x in p]
    v["mw_p_batch/exact/method"] = meth
    A2, B2 = r4.normal(size=(5, 10)) + 0.4, r4.normal(size=(5, 15))
    p, meth = g("mw_p_batch")(A2, B2)
    v["mw_p_batch/asym/p"] = [float(x) for x in p]
    v["mw_p_batch/asym/method"] = meth
    Ai = r4.integers(0, 3, size=(5, 10)).astype(float)
    Bi = r4.integers(0, 4, size=(5, 20)).astype(float)
    p, meth = g("mw_p_batch")(Bi, Ai, exact_max_n=0)
    v["mw_p_batch/ties/p"] = [float(x) for x in p]
    v["mw_p_batch/ties/method"] = meth
    for i in range(2):
        U, p1, m1 = g("mann_whitney_one_sided")(A[i], B[i])
        v[f"mann_whitney_one_sided/exact/{i}"] = [float(U), float(p1)]
        v[f"mann_whitney_one_sided/exact/{i}/method"] = m1
        U, p1, m1 = g("mann_whitney_one_sided")(Bi[i], Ai[i])
        v[f"mann_whitney_one_sided/ties/{i}"] = [float(U), float(p1)]
        v[f"mann_whitney_one_sided/ties/{i}/method"] = m1
    for (p1, p2) in ((0.01, 0.04), (0.03, 0.04), (0.001, 0.2),
                     (0.2, 0.024)):
        v[f"holm_two/{p1},{p2}"] = flags(g("holm_two")(p1, p2))
    Fa = r4.binomial(180, 0.01, size=(5, 10)) / 180.0
    Fb = r4.binomial(180, 0.016, size=(5, 20)) / 180.0
    fired, fc, fr = g("_c1_gate")(Ai, Bi, Fa, Fb)
    v["_c1_gate/fired"] = flags(fired)
    v["_c1_gate/catch"] = flags(fc)
    v["_c1_gate/rt"] = flags(fr)
    Ap = r4.normal(size=(5, 10, 2)) + np.array([0.9, 0.2])
    Ac = r4.normal(size=(5, 20, 2))
    rc, rf = g("_c1_test")(Ap, Ac)
    v["_c1_test/colour"] = flags(rc)
    v["_c1_test/fog"] = flags(rf)
    gp = {"w": Ap, "catch": Ai, "lapse": np.zeros_like(Ai), "fast": Fa}
    gc = {"w": Ac, "catch": Bi, "lapse": np.zeros_like(Bi), "fast": Fb}
    z = g("_c1_zero_catch")(gp, gc)
    for k_ in ("reject_colour", "reject_fog", "usable", "rt_fires"):
        v[f"_c1_zero_catch/{k_}"] = flags(z[k_])
    v["_c1_zero_catch/n"] = [float(x) for x in
                             np.concatenate([z["n_participants"],
                                             z["n_controls"]])]
    v["_th"] = th
    return v


def compare_values(mine, theirs):
    rows = []
    for k in sorted(set(mine) | set(theirs)):
        a, b = mine.get(k), theirs.get(k)
        if a is None or b is None:
            ok = False
        elif isinstance(a, str) or isinstance(b, str):
            ok = a == b
        else:
            aa = np.atleast_1d(np.asarray(a, float))
            bb = np.atleast_1d(np.asarray(b, float))
            ok = aa.shape == bb.shape and bool(np.all(
                (np.abs(aa - bb) <= IDENTITY_ABS_TOL
                 + IDENTITY_REL_TOL * np.abs(bb))
                | (np.isnan(aa) & np.isnan(bb))))
        rows.append({"quantity": k, "this_script": a, "reference": b,
                     "ok": ok})
    return rows


def function_sources(path, names):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    return {n.name: ast.dump(n) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names}


def load_power_module(path):
    spec = importlib.util.spec_from_file_location("power_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_identity_check(X, own_facts, checks, out_dir):
    print("\n" + "=" * 60)
    print("CHECKING THE COPIED FUNCTIONS")
    print("=" * 60)
    own = identity_values(globals(), X)
    th = own.pop("_th")
    rows = compare_values(own, IDENTITY_REFERENCE)
    for r in rows:
        r["against"] = "stored reference"
    bad = [r["quantity"] for r in rows if not r["ok"]]
    checks.require(
        "copied functions reproduce the stored reference values", not bad,
        f"{len(rows) - len(bad)}/{len(rows)} quantities agree"
        + (f"; differ: {', '.join(bad[:6])}" if bad else ""))
    try:
        check_bayes_factor([])
        ok, detail = True, "one- and two-sided forms agree"
    except (RuntimeError, AssertionError) as e:
        ok, detail = False, str(e).splitlines()[0]
    checks.require("JZS Bayes factor self-check (check_bayes_factor)", ok,
                   detail)
    if os.path.isfile(POWER_SCRIPT):
        pw = load_power_module(POWER_SCRIPT)
        live = identity_values(vars(pw), X)
        live.pop("_th")
        lrows = compare_values(own, live)
        for r in lrows:
            r["against"] = "power script, run now"
        rows += lrows
        bad = [r["quantity"] for r in lrows if not r["ok"]]
        checks.require(
            "copied functions agree with the power script run now", not bad,
            f"{len(lrows) - len(bad)}/{len(lrows)} quantities"
            + (f"; differ: {', '.join(bad[:6])}" if bad else ""))
        mine = function_sources(os.path.abspath(__file__), COPIED_FUNCTIONS)
        theirs = function_sources(POWER_SCRIPT, COPIED_FUNCTIONS)
        diff = [n for n in COPIED_FUNCTIONS
                if mine.get(n) is None or mine.get(n) != theirs.get(n)]
        checks.require("copied functions are verbatim (syntax tree)",
                       not diff, f"{len(COPIED_FUNCTIONS) - len(diff)}/"
                       f"{len(COPIED_FUNCTIONS)} identical"
                       + (f"; differ: {', '.join(diff)}" if diff else ""))
        sdiff = [n for n in COPIED_SETTINGS
                 if not hasattr(pw, n) or getattr(pw, n) != globals()[n]]
        checks.require("copied settings equal the power script's",
                       not sdiff, f"{len(COPIED_SETTINGS) - len(sdiff)}/"
                       f"{len(COPIED_SETTINGS)} equal"
                       + (f"; differ: {', '.join(sdiff)}" if sdiff else ""))
        their_facts = pw.verify_frozen(FROZEN_ROOT, [])
        checks.require("verify_frozen gives the power script's verdict",
                       their_facts == own_facts, "same FROZEN.json facts"
                       if their_facts == own_facts else "facts differ")
    else:
        checks.info("live comparison with the power script skipped",
                    f"{POWER_SCRIPT} not found; the stored reference "
                    "values were computed with it")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "identity_check.csv"),
                              index=False)
    return th


# ============================================================
# Inputs shared by every dataset
# ============================================================

class ListInfo:
    """The frozen list, arranged for analysis."""


def load_full_list(path, trials_an, X, checks):
    tl = pd.read_csv(path).sort_values("trial").reset_index(drop=True)
    L = ListInfo()
    L.tl = tl
    an = tl.analysed.astype(bool).to_numpy()
    L.idx_an = np.flatnonzero(an)
    L.trial_an = tl.trial.to_numpy()[L.idx_an]
    L.X_an = X
    L.n, L.n_an = len(tl), len(L.idx_an)
    L.by_trial = tl.set_index("trial")
    same = (np.array_equal(L.trial_an, trials_an.trial.to_numpy())
            and np.array_equal(tl.loc[L.idx_an, [PRED_SAT, PRED_FOG]]
                               .to_numpy(float), X))
    counts = tl.trial_type.value_counts().to_dict()
    want = {"sat_only": 60, "fog_only": 60, "together": 30,
            "conflict": 30, "consistency": 15, "catch": 10, "practice": 6}
    checks.require("frozen list: 211 trials, registered counts, analysed "
                   "rows in load_trials' order",
                   L.n == 211 and L.n_an == 180 and counts == want and same,
                   f"{L.n} trials, {L.n_an} analysed")
    return L


def load_rules(checks):
    cr = pd.read_csv(COMBINED_RULE)
    rules, bad = {}, []
    for feat, table in REGISTERED_RULES.items():
        for n, (k, m) in table.items():
            row = cr[(cr.feature == feat) & (cr.n == n)]
            if len(row) != 1:
                bad.append(f"{feat} N={n}: {len(row)} rows")
                continue
            row = row.iloc[0]
            if (int(row.k), int(row.m)) != (k, m):
                bad.append(f"{feat} N={n}: file k>={row.k}, m<={row.m}")
            if not row.false_worst_upper <= 0.05:
                bad.append(f"{feat} N={n}: false_worst_upper above 0.05")
            rules[(feat, n)] = {c: row[c] for c in row.index}
    checks.require("combined_rule.csv equals the registered rule at every "
                   "N from 10 to 20", not bad and len(cr) == 22,
                   f"{len(rules)} rules" if not bad else "; ".join(bad))
    return rules


def load_grid(checks):
    g = pd.read_csv(GRID_TRIALS)
    screens, problems = {}, []
    for (s, sc), d in g.groupby(["set", "screen"]):
        if len(d) != 18 or d.feature.nunique() != 1 \
                or d.column_order.nunique() != 1:
            problems.append(f"{s}/{sc}")
            continue
        disp = d.groupby("level_index").display_column.agg(
            lambda x: x.iloc[0] if x.nunique() == 1 else -999)
        if (disp == -999).any():
            problems.append(f"{s}/{sc} display column")
        screens[(s, int(sc))] = {
            "feature": d.feature.iloc[0],
            "column_order": d.column_order.iloc[0],
            "levels": sorted(int(x) for x in d.level_index.unique()),
            "disp": {int(k): int(x) for k, x in disp.items()},
            "sd_mean": {int(k): float(x) for k, x in d.groupby(
                "level_index").sd_from_original.mean().items()},
            "scene_ids": ";".join(d.sort_values("row").drop_duplicates(
                "row").scene_id.tolist())}
    checks.require("grid_trials.csv: 2 sets x 4 screens x 3 rows x 6 "
                   "columns, one feature per screen",
                   len(g) == 144 and not problems and len(screens) == 8,
                   "; ".join(problems))
    return screens


def load_sensitivities(checks):
    """Sensitivity per feature and assumed weight, as the power script
    reads it: the detection curve at lapse 0, linear interpolation."""
    det = pd.read_csv(DETECTION_CURVE)
    base = det[np.isclose(det.lapse, 0.0)]
    sens, got = {}, {}
    for feat in ("colour", "fog"):
        vals = []
        for label, w in PREVALENCE_SENSITIVITIES:
            s = float(np.interp(w, base.true_weight.to_numpy(),
                                base[feat].to_numpy()))
            sens[(feat, label)] = (w, s)
            vals.append(round(s, 2))
        got[feat] = tuple(vals)
    checks.require("prevalence sensitivities equal the registered ones "
                   "(2 dp)", got == REGISTERED_SENSITIVITIES,
                   f"colour {got['colour']}, fog {got['fog']}")
    return sens


def compute_caps(X, checks):
    """THE CAP, built exactly as the power script's SETTINGS describe it.

    This script's own code: the power script describes the construction
    (SETTINGS, "The caps") but never computes it. For each feature:
      every trial where |predictor| >= CAP_INFORMATIVE_SD -> answered as
                                                    that cue predicts
      every other trial                             -> y = 0.5
      the other weight                              -> free
    and the cap is that feature's Firth weight. The mirror construction
    (every informative trial answered AGAINST the cue) gives the lower
    end. The fitted point is confirmed to be a maximum of the penalised
    likelihood, and the sweep over CAP_THRESHOLD_SWEEP is reported so
    the plateau is visible.
    """
    caps, rows = {}, []
    for k, feat in enumerate(("colour", "fog")):
        for thr in CAP_THRESHOLD_SWEEP:
            a = X[:, k]
            info = np.abs(a) >= thr
            for side, y in (("upper", np.where(info, (a > 0) * 1.0, 0.5)),
                            ("lower", np.where(info, (a < 0) * 1.0, 0.5))):
                b = fit_firth(X, y)
                base = penalised_loglik(X, y, b)
                worst = max(penalised_loglik(X, y, b + d * e) - base
                            for e in np.eye(2) for d in (-1e-4, 1e-4))
                rows.append({"feature": feat, "informative_sd": thr,
                             "side": side, "n_informative": int(info.sum()),
                             "cap": float(b[k]), "other_weight": float(b[1 - k]),
                             "is_maximum": worst <= 1e-12})
        r = [x for x in rows if x["feature"] == feat
             and np.isclose(x["informative_sd"], CAP_INFORMATIVE_SD)]
        up = [x for x in r if x["side"] == "upper"][0]
        lo = [x for x in r if x["side"] == "lower"][0]
        caps[feat] = {"upper": up["cap"], "lower": lo["cap"],
                      "n_informative": up["n_informative"]}
    tab = pd.DataFrame(rows)
    got = {f: round(caps[f]["upper"], 3) for f in caps}
    sym = all(abs(caps[f]["upper"] + caps[f]["lower"]) < 1e-9 for f in caps)
    checks.require("the cap reproduces the registered values (3 dp), is "
                   "symmetric, and is a maximum",
                   got == REGISTERED_CAP and sym and tab.is_maximum.all(),
                   f"colour {caps['colour']['upper']:.5f}, "
                   f"fog {caps['fog']['upper']:.5f}")
    return caps, tab


# ============================================================
# The loader
# ============================================================

TRIAL_COLUMNS = ["session_id", "participant_id", "group", "trial",
                 "trial_type", "analysed", "key_pressed", "side_pressed",
                 "chose_A", "rt_ms", "catch_key", "catch_pass", "repeat_of"]
SESSION_COLUMNS = (
    ["session_id", "participant_id", "group", "session_start_time",
     "session_end_time", "complete", "is_test", "is_dev", "consent_given", "images_fit",
     "device_pixel_ratio", "gamma_estimate", "set_code", "interview_date",
     "has_word_block", "has_grid_episode", "has_grid_normal",
     "has_state_block"] + [f"word_{w}" for w in WORD_IDS] + STATE_ITEMS)
GRID_COLUMNS = ["session_id", "participant_id", "block", "set_code",
                "scene_set", "screen", "feature", "column_order",
                "scene_ids", "chosen_display_column", "chosen_level_index",
                "chose_none", "free_text", "rt_ms", "cell_px"]


def load_tidy(dataset_dir, checks):
    """The three tidy tables, with types fixed and columns checked."""
    d = os.path.join(dataset_dir, "tidy")
    T = pd.read_csv(os.path.join(d, TIDY_TRIALS), dtype={"session_id": str})
    S = pd.read_csv(os.path.join(d, TIDY_SESSIONS),
                    dtype={"session_id": str, "interview_date": str})
    G = pd.read_csv(os.path.join(d, TIDY_GRID), dtype={"session_id": str,
                                                       "free_text": str})
    miss = ([f"trials.{c}" for c in TRIAL_COLUMNS if c not in T.columns]
            + [f"sessions.{c}" for c in SESSION_COLUMNS if c not in S.columns]
            + [f"grid.{c}" for c in GRID_COLUMNS if c not in G.columns])
    checks.require("tidy tables have every expected column", not miss,
                   ", ".join(miss[:8]))
    for c in ("complete", "is_test", "is_dev", "consent_given", "images_fit",
              "has_word_block", "has_grid_episode", "has_grid_normal",
              "has_state_block"):
        S[c] = to_bool(S[c])
    bad = [c for c in ("complete", "is_test", "is_dev", "consent_given",
                       "images_fit") if S[c].isna().any()]
    checks.require("session flags are all True/False", not bad,
                   ", ".join(bad))
    for c in ("complete", "is_test", "is_dev", "consent_given", "images_fit",
              "has_word_block", "has_grid_episode", "has_grid_normal",
              "has_state_block"):
        S[c] = S[c].fillna(False).astype(bool)
    T["analysed"] = to_bool(T.analysed).astype(bool)
    T["catch_pass"] = to_bool(T.catch_pass)
    G["chose_none"] = to_bool(G.chose_none).fillna(False).astype(bool)
    S["interview_date"] = S.interview_date.fillna("")
    G["free_text"] = G.free_text.fillna("")
    checks.require("session ids are unique", not S.session_id.duplicated()
                   .any())
    bad = []
    for cols, top, what in (([f"word_{w}" for w in WORD_IDS], WORD_MAX,
                             "word ratings"),
                            (STATE_ITEMS, STATE_ITEM_MAX, "state items")):
        v = S[cols].apply(pd.to_numeric, errors="coerce")
        given = S[cols].notna()
        if (given & v.isna()).any().any():
            bad.append(f"{what}: a value that is not a number")
        vv = v.to_numpy(float)
        ok = np.isnan(vv) | ((vv >= 0) & (vv <= top) & (vv == np.round(vv)))
        if not ok.all():
            bad.append(f"{what}: a value outside whole numbers 0-{top}")
        S[cols] = v
    checks.require("word ratings and state items are whole numbers 0-6",
                   not bad, "; ".join(bad))
    return {"trials": T, "sessions": S, "grid": G}


def load_raw(dataset_dir, checks):
    """The experiment's own session CSVs -> the tidy tables.

    NOT YET WRITTEN, on purpose: the experiment is being updated to a new
    format (session order, group column, word ids, two grid blocks, state
    items, interview_date). This is finished against the updated
    DATA_DICTIONARY.md and real test sessions from both groups.
    """
    checks.require("raw session CSVs can be read", False,
                   "INPUT_FORMAT = 'raw' is finished after the experiment "
                   "update is merged; use INPUT_FORMAT = 'tidy'")


def load_private(dataset_dir, checks):
    """The survey and the participant table, both private."""
    sp = SURVEY_FILE or os.path.join(dataset_dir, "private",
                                     "survey_FAKE.xlsx")
    pp = PARTICIPANT_TABLE or os.path.join(dataset_dir, "private",
                                           "participant_table_FAKE.csv")
    for p, what in ((sp, "survey file"), (pp, "participant table")):
        checks.require(f"{what} found", os.path.isfile(p), p)
    sv = read_xlsx(sp)
    items = [f"Q{i:02d}_score" for i in range(1, CDS_ITEMS + 1)]
    miss = [c for c in ["ResponseId"] + items if c not in sv.columns]
    checks.require("survey has ResponseId and Q01_score .. Q29_score",
                   not miss, ", ".join(miss[:5]))
    num = sv[items].apply(pd.to_numeric, errors="coerce")
    header_like = num.isna().all(axis=1) & sv[items].notna().any(axis=1)
    n_header = int(header_like.sum())
    sv = sv[~header_like].copy()
    sv[items] = num[~header_like]
    sv["ResponseId"] = sv.ResponseId.astype(str)
    checks.require("survey ResponseIds are unique",
                   not sv.ResponseId.duplicated().any())
    if n_header:
        checks.info("survey: extra header rows dropped", str(n_header))
    sv["cds_total"] = sv[items].sum(axis=1, min_count=CDS_ITEMS)
    sv["cds_afs"] = sv[[f"Q{i:02d}_score" for i in AFS_ITEMS]].sum(
        axis=1, min_count=len(AFS_ITEMS))
    sv = sv.rename(columns={"Q13_score": "cds_q13", "Q02_score": "cds_q02"})

    pt = (pd.read_excel(pp) if pp.lower().endswith(".xlsx")
          and _has_openpyxl() else read_xlsx(pp)
          if pp.lower().endswith(".xlsx") else
          pd.read_csv(pp, dtype=str, keep_default_na=False))
    pt = pt.astype(str).replace({"nan": ""})
    miss = [c for c in ("participant_id", "group", "tier")
            if c not in pt.columns]
    checks.require("participant table has participant_id, group, tier",
                   not miss, ", ".join(miss))
    bad = pt[~pt.group.isin(["participant", "control"])
             | (pt.group.eq("participant") & ~pt.tier.isin(list("ABCD")))
             | (pt.group.eq("control") & pt.tier.ne(""))]
    checks.require("participant table: groups and tiers valid",
                   bad.empty and not pt.participant_id.duplicated().any(),
                   f"{len(bad)} invalid rows")
    return sv, pt, sp, pp


def _has_openpyxl():
    try:
        import openpyxl  # noqa: F401
        return True
    except ImportError:
        return False


def read_xlsx(path):
    """First sheet of an .xlsx as a DataFrame, first row as the header.

    Standard library only (the analysis environment promises no openpyxl).
    Handles shared strings, inline strings, numbers and booleans. The same
    reader as simulate_study_data_v4.py's.
    """
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        shared = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter(
                    "{%s}t" % ns["m"])))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        first = wb.find("m:sheets/m:sheet", ns)
        rid = first.get("{http://schemas.openxmlformats.org/officeDocument/"
                        "2006/relationships}id")
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = [r.get("Target") for r in rels
                  if r.get("Id") == rid][0].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        sh = ET.fromstring(z.read(target))
    table = []
    for row in sh.iter("{%s}row" % ns["m"]):
        rec = {}
        for c in row.findall("m:c", ns):
            ref = re.match(r"([A-Z]+)", c.get("r")).group(1)
            j = 0
            for ch in ref:
                j = j * 26 + (ord(ch) - 64)
            t = c.get("t")
            if t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter("{%s}t" % ns["m"]))
            else:
                v = c.find("m:v", ns)
                if v is None:
                    continue
                if t == "s":
                    val = shared[int(v.text)]
                elif t == "b":
                    val = v.text == "1"
                elif t in ("str", "e"):
                    val = v.text
                else:
                    f = float(v.text)
                    val = int(f) if f.is_integer() else f
            rec[j - 1] = val
        table.append(rec)
    if not table:
        return pd.DataFrame()
    width = max(max(r) + 1 for r in table if r)
    grid_ = [[r.get(j) for j in range(width)] for r in table]
    return pd.DataFrame(grid_[1:], columns=grid_[0])


# ============================================================
# A. Who is analysable
# ============================================================

STATUS_ORDER = ["test_or_dev", "incomplete", "repeat_not_used",
                "excluded_no_consent", "excluded_images_fit",
                "excluded_catch", "tier_D", "beyond_maximum", "analysable"]


def apply_exclusions(tidy, pt, L, checks):
    """One status per session, in the registered order.

      1  test or development session          excluded from everything
      2  not complete (not all 211 trials)    not a session for rule 3
      3  later sessions of the same ID        listed, never used
      4  no consent                           excluded
      5  images did not fit                   excluded
      6  3 or more of the 10 catch trials     excluded
      7  Tier D                               analysed separately
      8  beyond the maximum                   listed, not analysed: of
                                              the people left, the first
                                              20 participants and 30
                                              controls by session
                                              completion time are kept
    """
    S, T = tidy["sessions"].copy(), tidy["trials"]
    real = S[~(S.is_test | S.is_dev)]
    ptab = pt.set_index("participant_id")
    unknown = sorted(set(real.participant_id) - set(ptab.index))
    checks.require("every non-test session's ID is in the participant "
                   "table", not unknown, f"{len(unknown)} unknown")
    gdiff = real[real.group.to_numpy()
                 != ptab.group.reindex(real.participant_id).to_numpy()]
    checks.require("session group agrees with the participant table",
                   gdiff.empty, f"{len(gdiff)} sessions differ")

    per = T.groupby("session_id")
    n_rows = per.size()
    an = T[T.analysed]
    n_an = an.groupby("session_id").chose_A.apply(lambda s: s.notna().sum())
    S["n_trial_rows"] = n_rows.reindex(S.session_id).fillna(0).astype(
        int).to_numpy()
    S["n_analysed_answered"] = n_an.reindex(S.session_id).fillna(0).astype(
        int).to_numpy()
    S["is_complete"] = (S.complete & (S.n_trial_rows == L.n)
                        & (S.n_analysed_answered == L.n_an))
    cf = (T[T.trial_type == "catch"].groupby("session_id").catch_pass
          .apply(lambda s: int((s != True).sum())))
    S["n_catch_fail"] = cf.reindex(S.session_id).fillna(
        N_CATCH_TRIALS).astype(int).to_numpy()
    S["tier"] = ptab.tier.reindex(S.participant_id).fillna("").to_numpy()
    S["start"] = pd.to_datetime(S.session_start_time, utc=True)
    S["end"] = pd.to_datetime(S.session_end_time, utc=True, errors="coerce")

    status = pd.Series("", index=S.session_id.to_numpy(), dtype=object)
    test = (S.is_test | S.is_dev).to_numpy()
    status[S.session_id[test]] = "test_or_dev"
    inc = ~test & ~S.is_complete.to_numpy()
    status[S.session_id[inc]] = "incomplete"
    rest = S[~test & S.is_complete].sort_values(["start", "session_id"])
    first = rest.drop_duplicates("participant_id", keep="first")
    status[rest.session_id[~rest.session_id.isin(first.session_id)]] = \
        "repeat_not_used"
    for _, r in first.iterrows():
        if not r.consent_given:
            st = "excluded_no_consent"
        elif not r.images_fit:
            st = "excluded_images_fit"
        elif r.n_catch_fail >= CATCH_FAIL_EXCLUDE:
            st = "excluded_catch"
        elif r.tier == "D":
            st = "tier_D"
        else:
            st = "analysable"
        status[r.session_id] = st
    # 8: the maxima, by completion time (ties by session id).
    S["completion_rank"] = pd.array([pd.NA] * len(S), dtype="Int64")
    for grp, limit in (("participant", N_MAX_PARTICIPANTS),
                       ("control", N_MAX_CONTROLS)):
        sel = S[(status[S.session_id].to_numpy() == "analysable")
                & (S.group == grp).to_numpy()]
        if len(sel) > limit:
            checks.require(f"every {grp} session has a completion time "
                           "(needed to apply the maximum)",
                           sel.end.notna().all(),
                           f"{int(sel.end.isna().sum())} missing")
        sel = sel.sort_values(["end", "session_id"])
        S.loc[sel.index, "completion_rank"] = np.arange(1, len(sel) + 1)
        status[sel.session_id.iloc[limit:]] = "beyond_maximum"
    S["status"] = status[S.session_id].to_numpy()
    counts = (S.groupby(["status", "group"]).size().unstack(fill_value=0)
              .reindex(index=STATUS_ORDER, columns=["participant", "control"],
                       fill_value=0))
    return S, counts


def assign_pseudonyms(S):
    """Shareable codes for every session.

    The person's code is set by the status of their USED session:
    P01.. analysable participants, C01.. analysable controls, D01.. Tier D,
    BP01.. / BC01.. beyond the maximum,
    XP01.. / XC01.. excluded participants / controls, each numbered by
    session start time. The used session carries the bare code; any other
    session of the same person carries the code plus '-s<k>', k being its
    place in that person's sessions by time. Test and development
    sessions are TEST01, TEST02, ...
    """
    S = S.sort_values(["start", "session_id"])
    person, counters = {}, {}

    def new(pre):
        counters[pre] = counters.get(pre, 0) + 1
        return f"{pre}{counters[pre]:02d}"

    for st, grp, pre in (("analysable", "participant", "P"),
                         ("analysable", "control", "C"),
                         ("tier_D", "participant", "D"),
                         ("beyond_maximum", "participant", "BP"),
                         ("beyond_maximum", "control", "BC")):
        for pid in S[(S.status == st) & (S.group == grp)].participant_id:
            person[pid] = new(pre)
    for grp, pre in (("participant", "XP"), ("control", "XC")):
        sel = S[S.status.str.startswith("excluded") & (S.group == grp)]
        for pid in sel.participant_id:
            person[pid] = new(pre)
    used = {"analysable", "tier_D", "beyond_maximum", "excluded_no_consent",
            "excluded_images_fit", "excluded_catch"}
    code, nth = {}, {}
    for _, r in S.iterrows():
        if r.status == "test_or_dev":
            code[r.session_id] = new("TEST")
            continue
        if r.participant_id not in person:
            person[r.participant_id] = new(
                "XP" if r.group == "participant" else "XC")
        base = person[r.participant_id]
        nth[base] = nth.get(base, 0) + 1
        code[r.session_id] = (base if r.status in used
                              else f"{base}-s{nth[base]}")
    return pd.Series(code)


# ============================================================
# B. The per-person model
# ============================================================

def firth_score(X, y, b):
    """Firth's modified score at b; ~0 at a converged estimate."""
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -500.0, 500.0)))
    w = np.clip(p * (1 - p), 1e-12, None)
    Ainv = np.linalg.pinv((X * w[:, None]).T @ X)
    h = w * np.einsum("ti,ij,tj->t", X, Ainv, X)
    return X.T @ (y - p + h * (0.5 - p))


def profile_interval(X, y, b, j, level=CI_LEVEL):
    """95% profile penalised-likelihood interval for weight j.

    This script's own code (the power script has none). The profile is
    the penalised log-likelihood maximised over the other weight with
    weight j held fixed; the interval is where it lies within
    chi2_1(level)/2 of its maximum (Heinze & Schemper, 2002). A bound the
    profile never reaches within PROFILE_SEARCH_MAX of the estimate is
    returned as +/-inf, which is what an answer pattern near separation
    produces.
    """
    other = 1 - j
    crit = stats.chi2.ppf(level, 1) / 2.0
    pl_max = penalised_loglik(X, y, b)
    cache = {"u": float(b[other])}

    def prof(val):
        def neg(u):
            bb = np.empty(2)
            bb[j], bb[other] = val, u
            return -penalised_loglik(X, y, bb)
        c = cache["u"]
        lo_u, hi_u = c - 5.0, c + 5.0
        for _ in range(6):
            res = optimize.minimize_scalar(neg, bounds=(lo_u, hi_u),
                                           method="bounded",
                                           options={"xatol": 1e-10})
            if hi_u - res.x > 1e-3 and res.x - lo_u > 1e-3:
                break
            lo_u, hi_u = lo_u - 10.0, hi_u + 10.0
        cache["u"] = float(res.x)
        return -float(res.fun)

    def gap(val):
        return pl_max - prof(val) - crit

    w_ = np.clip(1 / (1 + np.exp(-X @ b)) * (1 - 1 / (1 + np.exp(-X @ b))),
                 1e-12, None)
    se = float(np.sqrt(np.linalg.pinv((X * w_[:, None]).T @ X)[j, j]))
    out = []
    for direction in (-1.0, 1.0):
        cache["u"] = float(b[other])
        step, prev, found = max(0.25 * se, 0.01), float(b[j]), None
        while abs(prev - b[j]) < PROFILE_SEARCH_MAX:
            cur = float(b[j]) + direction * min(
                abs(prev - b[j]) + step, PROFILE_SEARCH_MAX)
            if gap(cur) > 0:
                found = (prev, cur)
                break
            prev, step = cur, step * 2.0
        if found is None:
            out.append(direction * np.inf)
            continue
        cache["u"] = float(b[other])
        out.append(float(optimize.brentq(gap, *sorted(found),
                                         xtol=PROFILE_XTOL)))
    return out[0], out[1]


def fit_people(S, T, L, th):
    """Firth weights, profile intervals and classes for the analysable
    and Tier D sessions."""
    keep = S[S.status.isin(["analysable", "tier_D"])]
    an = T[T.analysed & T.session_id.isin(keep.session_id)]
    Y = an.pivot(index="session_id", columns="trial", values="chose_A")
    Y = Y.loc[keep.session_id, L.trial_an].astype(float)
    rows = []
    for sid, y in Y.iterrows():
        y = y.to_numpy()
        b = fit_firth(L.X_an, y)
        sc = float(np.abs(firth_score(L.X_an, y, b)).max())
        r = {"session_id": sid, "w_colour": float(b[0]), "w_fog": float(b[1]),
             "score_max": sc}
        for j, f in enumerate(("colour", "fog")):
            lo, hi = profile_interval(L.X_an, y, b, j)
            r[f"ci_lo_{f}"], r[f"ci_hi_{f}"] = lo, hi
        rows.append(r)
    W = pd.DataFrame(rows)
    est = W[["w_colour", "w_fog"]].to_numpy()
    pred, opp = classify(est, th)
    for j, f in enumerate(("colour", "fog")):
        W[f"class_{f}"] = np.where(pred[:, j], "predicted",
                                   np.where(opp[:, j], "opposite",
                                            "not detected"))
    return W.merge(keep[["session_id", "participant_id", "group", "tier",
                         "status", "n_catch_fail"]], on="session_id")


# ============================================================
# C. H1 (fog) and H2 (colour)
# ============================================================

def existence_claims(Pw, rules):
    """The registered count rule for the final N, per feature."""
    n = len(Pw)
    out = {}
    for feat, claim in (("fog", "H1"), ("colour", "H2")):
        a = int((Pw[f"class_{feat}"] == "predicted").sum())
        b = int((Pw[f"class_{feat}"] == "opposite").sum())
        d = {"claim": claim, "feature": feat, "n": n, "predicted": a,
             "opposite": b, "not_detected": n - a - b}
        if n < N_MIN_CLAIM:
            d["verdict"] = "descriptive"
            d["why"] = (f"N = {n} < {N_MIN_CLAIM}: no claim is made; the "
                        "counts are descriptive")
        else:
            r = rules[(feat, n)]
            k, m = int(r["k"]), int(r["m"])
            hit = bool(rule_mask_count(n, k, m)[a, b])
            d.update({"k": k, "m": m, "rule": f"k >= {k}, m <= {m}",
                      "verdict": "supported" if hit else "not supported",
                      "false_worst": float(r["false_worst"]),
                      "false_worst_upper": float(r["false_worst_upper"]),
                      "false_guesser": float(r["false_guesser"]),
                      "planning_power": float(r["planning_power"]),
                      "why": f"{a} predicted (need >= {k}), {b} opposite "
                             f"(allow <= {m})"})
        out[claim] = d
    return out


# ============================================================
# D. C1, with the attention check first
# ============================================================

def c1_analysis(Pw, Cw, checks):
    """The attention check, then C1, then the zero-catch-failure test.

    The C1 p-values are mw_p_batch's, through _c1_test -- the code the
    registered rates were simulated with. mann_whitney_one_sided (scipy)
    is run beside it and must agree. All of it on the ANALYSABLE people,
    after the >= 3 catch-failure exclusion (the registered gate
    simulation did not apply that exclusion; the report says so).
    """
    if not len(Pw) or not len(Cw):
        return {"available": False}
    cp, cc = Pw.n_catch_fail.to_numpy(float), Cw.n_catch_fail.to_numpy(float)
    fp, fc = Pw.fast_share.to_numpy(float), Cw.fast_share.to_numpy(float)
    fired, f_c, f_r = _c1_gate(cp[None], cc[None], fp[None], fc[None])
    p_catch = float(mw_p_batch(cc[None], cp[None], exact_max_n=0)[0][0])
    p_rt = float(mw_p_batch(fc[None], fp[None], exact_max_n=0)[0][0])
    Wp = Pw[["w_colour", "w_fog"]].to_numpy()
    Wc = Cw[["w_colour", "w_fog"]].to_numpy()
    rc, rf = _c1_test(Wp[None], Wc[None])
    out = {"available": True, "n_participants": len(Pw),
           "n_controls": len(Cw), "descriptive": len(Cw) < MIN_CONTROLS_C1,
           "gate": {"fired": bool(fired[0]), "fired_catch": bool(f_c[0]),
                    "fired_rt": bool(f_r[0]), "p_catch": p_catch,
                    "p_rt": p_rt,
                    "median_catch_participants": float(np.median(cp)),
                    "median_catch_controls": float(np.median(cc)),
                    "mean_fast_participants": float(fp.mean()),
                    "mean_fast_controls": float(fc.mean())}}
    pv = {}
    for k, feat, rej in ((0, "colour", rc), (1, "fog", rf)):
        p, meth = mw_p_batch(Wp[None, :, k], Wc[None, :, k])
        U, p2, m2 = mann_whitney_one_sided(Wp[:, k], Wc[:, k])
        pv[feat] = float(p[0])
        agree = abs(float(p[0]) - p2) <= MW_AGREE_TOL and meth == m2
        checks.require(f"C1 {feat}: mw_p_batch and scipy's Mann-Whitney "
                       "agree", agree, f"{float(p[0]):.10g} vs {p2:.10g} "
                       f"({meth} / {m2})")
        if out["descriptive"]:
            verdict = "descriptive"
        elif fired[0]:
            verdict = "inconclusive"
        else:
            verdict = "supported" if rej[0] else "not supported"
        out[f"C1_{feat}"] = {"U": U, "p": float(p[0]), "method": meth,
                             "holm_reject": bool(rej[0]), "verdict": verdict,
                             "median_participants": float(np.median(Wp[:, k])),
                             "median_controls": float(np.median(Wc[:, k]))}
    h = holm_two(pv["colour"], pv["fog"])
    checks.require("C1: holm_two and _c1_test give the same verdicts",
                   (bool(h[0]), bool(h[1])) == (bool(rc[0]), bool(rf[0])))

    gp = {"w": Wp[None], "catch": cp[None], "lapse": np.zeros((1, len(cp))),
          "fast": fp[None]}
    gc = {"w": Wc[None], "catch": cc[None], "lapse": np.zeros((1, len(cc))),
          "fast": fc[None]}
    z = _c1_zero_catch(gp, gc)
    zc = {"usable": bool(z["usable"][0]),
          "n_participants": int(z["n_participants"][0]),
          "n_controls": int(z["n_controls"][0]),
          "min_n": C1_ZERO_MIN_N, "rt_recheck_fires": bool(z["rt_fires"][0])}
    if zc["usable"]:
        kp, kc = cp == 0, cc == 0
        for k, feat, key in ((0, "colour", "reject_colour"),
                             (1, "fog", "reject_fog")):
            p, meth = mw_p_batch(Wp[kp][None, :, k], Wc[kc][None, :, k])
            zc[f"C1_{feat}"] = {"p": float(p[0]), "method": meth,
                                "holm_reject": bool(z[key][0]),
                                "verdict": ("supported" if z[key][0]
                                            else "not supported")}
    out["zero_catch"] = zc
    return out


# ============================================================
# E. Secondary: t-test and Bayes factor, with and without the cap
# ============================================================

def secondary_tests(Pw, caps):
    rows = []
    for feat in ("colour", "fog"):
        for capped in (False, True):
            w = Pw[f"w_{feat}"].to_numpy(float)
            n_cl = 0
            if capped:
                lo, hi = caps[feat]["lower"], caps[feat]["upper"]
                n_cl = int(((w < lo) | (w > hi)).sum())
                w = np.clip(w, lo, hi)
            n = len(w)
            r = {"feature": feat, "capped": capped, "n": n,
                 "n_clipped": n_cl,
                 "cap": caps[feat]["upper"] if capped else np.nan}
            sd = w.std(ddof=1) if n >= 2 else np.nan
            if n >= 2 and sd > 0:
                t = float(w.mean() / (sd / np.sqrt(n)))
                p = float(stats.t.sf(t, n - 1))
                p_sp = float(stats.ttest_1samp(w, 0.0,
                                               alternative="greater").pvalue)
                bf = float(jzs_bayes_factor(t, n))
                r.update({"mean": float(w.mean()), "sd": float(sd), "t": t,
                          "df": n - 1, "p_one_sided": p, "p_scipy": p_sp,
                          "BF10_one_sided": bf,
                          "BF01": 1.0 / bf if bf > 0 else np.nan,
                          "bayes_reading": (
                              "evidence for (BF10 > 3)" if bf > BF_THRESHOLD
                              else "evidence against (BF01 > 3)"
                              if bf < 1 / BF_THRESHOLD else "inconclusive")})
            rows.append(r)
    return pd.DataFrame(rows)


# ============================================================
# F. Exploratory and descriptive
# ============================================================

def prevalence_table(Pw, sens):
    """Counts, and the estimate under three assumed sensitivities.
    Descriptive: no 'excludes zero' inference is drawn."""
    n = len(Pw)
    rows = []
    for feat in ("colour", "fog"):
        a = int((Pw[f"class_{feat}"] == "predicted").sum())
        b = int((Pw[f"class_{feat}"] == "opposite").sum())
        for label, _w in PREVALENCE_SENSITIVITIES:
            w, s = sens[(feat, label)]
            r = {"feature": feat, "n": n, "predicted": a, "opposite": b,
                 "assumed_responder": f"{label} {w:.2f}/SD",
                 "sensitivity": s}
            if n:
                gg, post = prevalence_posterior_shared(a, b, n, s)
                lo, hi = hpdi(gg, post)
                r.update({"moment": prevalence_moment(a, b, n, s),
                          "moment_raw": prevalence_moment_raw(a, b, n, s),
                          "posterior_mode": float(gg[int(np.argmax(post))]),
                          "hpdi95_lo": lo, "hpdi95_hi": hi})
            rows.append(r)
    return pd.DataFrame(rows)


def predicted_choice(an, L):
    """Per analysed trial: 1 if the predicted image was chosen.

    The predicted image is the one the trial's driving feature favours
    (mag_driver: d_sat -> pred_sat, d_fog -> pred_fog). So conflict
    trials are scored by FOG, as in the power script: read the conflict
    curve as "followed the fog cue".
    """
    tl = L.by_trial
    drv = tl.mag_driver.reindex(an.trial).to_numpy()
    ps = tl[PRED_SAT].reindex(an.trial).to_numpy()
    pf = tl[PRED_FOG].reindex(an.trial).to_numpy()
    x = np.where(drv == "d_sat", ps, pf)
    y = an.chose_A.to_numpy(float)
    return np.where(x > 0, y, 1.0 - y), tl.mag_bin.reindex(an.trial).to_numpy()


def suzuki_counts(T, sids, L):
    an = T[T.analysed & T.session_id.isin(sids)].copy()
    an["pred_choice"], an["mag_bin"] = predicted_choice(an, L)
    out = []
    for typ, key in (("sat_only", "sat"), ("fog_only", "fog")):
        g = an[an.trial_type == typ].groupby("session_id").pred_choice
        out.append(pd.DataFrame({f"suzuki_{key}_predicted": g.sum(),
                                 f"suzuki_{key}_trials": g.size()}))
    s = pd.concat(out, axis=1)
    s["follows_sat"] = s.suzuki_sat_predicted >= SUZUKI_FOLLOW
    s["follows_fog"] = s.suzuki_fog_predicted >= SUZUKI_FOLLOW
    return s.reset_index()


def suzuki_tests(people):
    rows = []
    for key, label in (("sat", "colour (sat_only)"), ("fog", "fog (fog_only)")):
        for grp in ("participant", "control"):
            d = people[people.group_label == grp]
            prop = (d[f"suzuki_{key}_predicted"]
                    / d[f"suzuki_{key}_trials"]).to_numpy(float)
            r = {"trials": label, "group": grp, "n": len(prop),
                 "n_following": int(d[f"follows_{key}"].sum())}
            if len(prop) >= 2 and prop.std(ddof=1) > 0:
                res = stats.ttest_1samp(prop, 0.5, alternative="greater")
                r.update({"mean_proportion": float(prop.mean()),
                          "t": float(res.statistic),
                          "p_one_sided": float(res.pvalue)})
            rows.append(r)
    return pd.DataFrame(rows)


def h3_summary(people):
    """H3: the weight angle atan2(b_fog, b_sat), compared between people
    only, with no per-person test; plus who is detected on both."""
    rows = []
    for grp in ("participant", "control"):
        d = people[people.group_label == grp]
        a = d.angle_deg.to_numpy(float)
        r = {"group": grp, "n": len(a)}
        if len(a):
            q = np.percentile(a, [0, 25, 50, 75, 100])
            r.update({"min": q[0], "q1": q[1], "median": q[2], "q3": q[3],
                      "max": q[4]})
        r["detected_predicted_both"] = int(
            ((d.class_colour == "predicted") & (d.class_fog == "predicted"))
            .sum())
        r["detected_either_direction_both"] = int(
            ((d.class_colour != "not detected")
             & (d.class_fog != "not detected")).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def spearman_boot(x, y, seed):
    """Spearman's rho with a percentile bootstrap interval over people."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    out = {"n": n, "rho": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
           "boot_degenerate": 0}
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return out
    out["rho"] = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(BOOTSTRAP_REPS, n))
    rx = stats.rankdata(x[idx], axis=1)
    ry = stats.rankdata(y[idx], axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (rx * ry).sum(1) / den
    good = np.isfinite(r) & (den > 0)
    out["boot_degenerate"] = int((~good).sum())
    a = (1 - CI_LEVEL) / 2
    out["ci_lo"], out["ci_hi"] = (float(v) for v in
                                  np.percentile(r[good], [100 * a,
                                                          100 * (1 - a)]))
    return out


def grid_per_person(G, S_used, screens, checks):
    """E2: the normal-to-episode shift per person and feature.

    For each screen, the mean sd_from_original of the chosen column over
    its three rows; averaged over the two screens of a feature in each
    block; episode minus normal. 'None of these' screens contribute
    nothing; a person with both episode screens of a feature 'none', or a
    missing block, has no shift for that feature and is counted.
    """
    G = G[G.session_id.isin(S_used.session_id)]
    probs, rows = [], []
    set_code = S_used.set_index("session_id").set_code
    for sid in S_used.session_id:
        g = G[G.session_id == sid]
        sc = set_code.get(sid)
        ep = "A" if sc == 0 else "B"
        want = {"episode": ep, "normal": "B" if ep == "A" else "A"}
        r = {"session_id": sid, "set_code": sc,
             "cell_px": float(g.cell_px.iloc[0]) if len(g) else np.nan}
        vals = {}
        for block in ("episode", "normal"):
            b = g[g.block == block]
            r[f"grid_{block}_screens"] = len(b)
            if len(b) and set(b.scene_set) != {want[block]}:
                probs.append(f"{sid}: {block} block shows set "
                             f"{sorted(set(b.scene_set))}, set code {sc}")
            for _, x in b.iterrows():
                info = screens.get((x.scene_set, int(x.screen)))
                if info is None or info["feature"] != x.feature:
                    probs.append(f"{sid}: unknown screen {x.scene_set}/"
                                 f"{x.screen}")
                    continue
                if x.chose_none:
                    if block != "episode":
                        probs.append(f"{sid}: 'none' in the normal block")
                    continue
                lv = int(x.chosen_level_index)
                if lv not in info["levels"] or info["disp"][lv] != int(
                        x.chosen_display_column):
                    probs.append(f"{sid}: level/column mismatch")
                    continue
                vals.setdefault((block, x.feature), []).append(
                    info["sd_mean"][lv])
            for f in ("colour", "fog"):
                bf = b[b.feature == f]
                r[f"grid_{block}_{f}_at_original"] = int(
                    (~bf.chose_none & (bf.chosen_level_index == 0)).sum())
            r[f"grid_{block}_none"] = int(b.chose_none.sum())
        r["grid_free_text"] = int((g.free_text.str.len() > 0).sum())
        for f in ("colour", "fog"):
            e, n_ = vals.get(("episode", f), []), vals.get(("normal", f), [])
            r[f"grid_episode_mean_sd_{f}"] = float(np.mean(e)) if e else np.nan
            r[f"grid_normal_mean_sd_{f}"] = float(np.mean(n_)) if n_ else np.nan
            r[f"grid_shift_{f}"] = (float(np.mean(e) - np.mean(n_))
                                    if e and n_ else np.nan)
        r["grid_both_episode_fog_none"] = int(
            g[(g.block == "episode") & (g.feature == "fog")].chose_none.sum()
            == 2)
        rows.append(r)
    checks.require("grid rows agree with the grid package and the set code",
                   not probs, "; ".join(probs[:4]))
    return pd.DataFrame(rows)


def grid_summary(people):
    rows = []
    for grp in ("participant", "control"):
        d = people[people.group_label == grp]
        for f in ("colour", "fog"):
            s = d[f"grid_shift_{f}"].dropna()
            rows.append({
                "group": grp, "feature": f, "n_people": len(d),
                "n_with_shift": len(s),
                "n_missing_block": int(((d.grid_episode_screens == 0)
                                        | (d.grid_normal_screens == 0)).sum()),
                "n_both_episode_screens_none": int(
                    d.grid_both_episode_fog_none.sum()) if f == "fog" else
                int((d.grid_episode_screens.gt(0)
                     & d[f"grid_episode_mean_sd_{f}"].isna()).sum()),
                "shift_mean": float(s.mean()) if len(s) else np.nan,
                "shift_median": float(s.median()) if len(s) else np.nan,
                "shift_sd": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
                "episode_answers_at_original": int(
                    d[f"grid_episode_{f}_at_original"].sum()),
                "normal_answers_at_original": int(
                    d[f"grid_normal_{f}_at_original"].sum()),
                "episode_none": int(d.grid_episode_none.sum()) if f == "fog"
                else np.nan,
                "set_code_0": int((d.set_code == 0).sum()),
                "set_code_1": int((d.set_code == 1).sum()),
                "cell_px_median": float(d.cell_px.median())
                if d.cell_px.notna().any() else np.nan})
    return pd.DataFrame(rows)


def dose_response(T, people, L):
    """Share of predicted choices by trial type and magnitude bin, per
    group: the mean of the people's shares, and the pooled share."""
    rows = []
    for grp in ("participant", "control"):
        sids = people[people.group_label == grp].session_id
        an = T[T.analysed & T.session_id.isin(sids)].copy()
        if not len(an):
            continue
        an["pred_choice"], an["mag_bin"] = predicted_choice(an, L)
        per = an.groupby(["trial_type", "mag_bin", "session_id"]
                         ).pred_choice.mean().reset_index()
        for (typ, b), d in per.groupby(["trial_type", "mag_bin"]):
            pooled = an[(an.trial_type == typ) & (an.mag_bin == b)]
            rows.append({"group": grp, "trial_type": typ, "mag_bin": int(b),
                         "n_people": len(d),
                         "mean_share": float(d.pred_choice.mean()),
                         "sd_share": float(d.pred_choice.std(ddof=1))
                         if len(d) > 1 else np.nan,
                         "pooled_share": float(pooled.pred_choice.mean()),
                         "n_trials": len(pooled)})
    return pd.DataFrame(rows)


def quality_per_person(T, S_used, L):
    """Consistency on the 15 repeated pairs, RT, side, covariates."""
    rows = []
    tl = L.by_trial
    cons = tl[tl.trial_type == "consistency"]
    for sid, d in T[T.session_id.isin(S_used.session_id)].groupby(
            "session_id"):
        byt = d.set_index("trial")
        agree = [float(byt.chose_A.get(int(t)) == byt.chose_A.get(int(o)))
                 for t, o in cons.repeat_of.items()]
        an = d[d.analysed]
        rows.append({"session_id": sid,
                     "consistency": float(np.mean(agree)),
                     "consistency_pairs": len(agree),
                     "rt_median_ms": float(an.rt_ms.median()),
                     "rt_q1_ms": float(an.rt_ms.quantile(0.25)),
                     "rt_q3_ms": float(an.rt_ms.quantile(0.75)),
                     "fast_share": float((an.rt_ms < RT_FAST_CUTOFF_MS).mean()),
                     "side_left_share": float((an.side_pressed == "left")
                                              .mean())})
    q = pd.DataFrame(rows)
    s = S_used.set_index("session_id")
    start_local = s.start.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)\
        .dt.normalize()
    idate = pd.to_datetime(s.interview_date.replace("", np.nan),
                           errors="coerce")
    delay = (start_local - idate).dt.days
    q["delay_days"] = delay.reindex(q.session_id).to_numpy()
    for c in ["gamma_estimate", "device_pixel_ratio", "set_code"] + \
            STATE_ITEMS:
        q[c] = s[c].reindex(q.session_id).to_numpy()
    q["has_state_block"] = s.has_state_block.reindex(q.session_id).to_numpy()
    return q


def word_scores(S_used):
    s = S_used.set_index("session_id")
    words = s[[f"word_{w}" for w in WORD_IDS]].astype(float)
    has = s.has_word_block.astype(bool) & words.notna().all(axis=1)
    flat = has & (words.nunique(axis=1) == 1)
    base = words[[f"word_{w}" for w in WORD_BASELINE]].mean(axis=1)
    out = pd.DataFrame({"session_id": s.index,
                        "has_word_block": has.to_numpy(),
                        "flat_word_ratings": flat.to_numpy(),
                        "foggy_rel": (words.word_foggy - base).to_numpy(),
                        "faded_rel": (words.word_faded - base).to_numpy()})
    use = out.has_word_block & ~out.flat_word_ratings
    out.loc[~use, ["foggy_rel", "faded_rel"]] = np.nan
    return out


# ============================================================
# G. Figures
# ============================================================

def _style(ax):
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#444444", labelsize=8)


def fig_weights(people, th, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    order = ["participant", "control", "tier_D"]
    for ax, (j, f) in zip(axes, enumerate(("colour", "fog"))):
        x0 = 0
        ends = people[[f"w_{f}", f"ci_lo_{f}", f"ci_hi_{f}"]].to_numpy(float)
        fin = np.abs(ends[np.isfinite(ends)])
        lim = float(min(4.0, max(1.0, 1.15 * fin.max()))) if fin.size \
            else 4.0
        for grp in order:
            d = people[people.group_label == grp].sort_values(f"w_{f}")
            if not len(d):
                continue
            col, mk, lab = GROUP_STYLE[grp]
            xs = x0 + np.arange(len(d))
            w = d[f"w_{f}"].to_numpy()
            lo = np.clip(d[f"ci_lo_{f}"].to_numpy(), -lim, lim)
            hi = np.clip(d[f"ci_hi_{f}"].to_numpy(), -lim, lim)
            ax.vlines(xs, lo, hi, color=col, lw=1.0, alpha=0.7)
            det = (d[f"class_{f}"] != "not detected").to_numpy()
            ax.scatter(xs[det], np.clip(w[det], -lim, lim), marker=mk, s=36,
                       color=col, edgecolor="white", lw=0.8, zorder=3,
                       label=f"{lab}, detected")
            ax.scatter(xs[~det], np.clip(w[~det], -lim, lim), marker=mk, s=36,
                       facecolor="white", edgecolor=col, lw=1.2, zorder=3,
                       label=f"{lab}, not detected")
            x0 += len(d) + 3
        for v in (th["upper"][j], th["lower"][j]):
            ax.axhline(v, color="#555555", lw=0.9, ls="--")
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_ylabel(f"{f} weight per corpus SD (95% profile interval)",
                      fontsize=9, color="#333333")
        n_clip = int((np.abs(ends) > lim).sum())
        ax.set_title(f"{f}: thresholds {th['upper'][j]:+.3f} / "
                     f"{th['lower'][j]:+.3f} (dashed)"
                     + (f"; {n_clip} interval end(s) beyond +/-{lim:.1f} "
                        "drawn at the edge" if n_clip else ""),
                     fontsize=9, color="#333333")
        _style(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center",
               ncol=len(l))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_c1(people, c1, path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    panels = [("w_colour", "colour weight"), ("w_fog", "fog weight"),
              ("n_catch_fail", "catch failures (of 10)"),
              ("fast_share", "share of responses < 300 ms")]
    rng = np.random.default_rng(0)
    for ax, (col, lab) in zip(axes, panels):
        for i, grp in enumerate(("participant", "control")):
            d = people[people.group_label == grp][col].to_numpy(float)
            if not len(d):
                continue
            c, mk, gl = GROUP_STYLE[grp]
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, marker=mk,
                       s=22, color=c, alpha=0.8, edgecolor="white", lw=0.5)
            ax.hlines(np.median(d), i - 0.25, i + 0.25, color="#222222",
                      lw=1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["participants", "controls"], fontsize=8)
        ax.set_title(lab, fontsize=9, color="#333333")
        if col == "n_catch_fail":
            ax.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(integer=True))
        _style(ax)
    if c1.get("available"):
        g = c1["gate"]
        axes[2].set_xlabel(f"one-sided p {fmt_p(g['p_catch'])}", fontsize=8)
        axes[3].set_xlabel(f"one-sided p {fmt_p(g['p_rt'])}", fontsize=8)
        for ax, f in ((axes[0], "colour"), (axes[1], "fog")):
            r = c1[f"C1_{f}"]
            ax.set_xlabel(f"one-sided p {fmt_p(r['p'])}: {r['verdict']}",
                          fontsize=8)
    fig.suptitle("C1 and the attention check (bars: group medians)",
                 fontsize=10, color="#222222")
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_grid(people, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    rng = np.random.default_rng(1)
    for ax, f in zip(axes, ("colour", "fog")):
        for i, grp in enumerate(("participant", "control")):
            d = people[people.group_label == grp][f"grid_shift_{f}"].dropna()
            if not len(d):
                continue
            c, mk, gl = GROUP_STYLE[grp]
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, marker=mk,
                       s=22, color=c, alpha=0.8, edgecolor="white", lw=0.5)
            ax.hlines(d.mean(), i - 0.25, i + 0.25, color="#222222", lw=1.5)
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["participants", "controls"], fontsize=8)
        ax.set_ylabel("episode minus normal, corpus SD", fontsize=9)
        ax.set_title(f"grid shift: {f} (negative = less "
                     f"{'colourful' if f == 'colour' else 'local contrast'})",
                     fontsize=9, color="#333333")
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


# ============================================================
# Comparison with a simulated dataset's truth
# ============================================================

def compare_with_truth(dataset_dir, S, W, claims, c1, partial, people,
                       counts, checks):
    """Does this analysis find exactly the answers the simulation put in?

    Exact: every session's status, every class, every verdict, the
    exclusion and partial-exclusion counts, each grid shift. Weights to
    TRUTH_WEIGHT_TOL (the truth was fitted with fit_firth_batch, this
    analysis uses fit_firth). Design expectations: 'fixed' ones must hold;
    single-draw ones are reported.
    """
    tj = json.load(open(os.path.join(dataset_dir, "truth.json")))
    tp = pd.read_csv(os.path.join(dataset_dir, "truth_people.csv"),
                     dtype={"session_id": str})
    st = S.set_index("session_id").status
    bad = tp[tp.expected_status.to_numpy()
             != st.reindex(tp.session_id).to_numpy()]
    checks.require("truth: every session's status", bad.empty,
                   f"{len(bad)} differ")
    m = W.merge(tp[["session_id", "fitted_w_colour", "fitted_w_fog",
                    "class_colour", "class_fog"]], on="session_id",
                suffixes=("", "_truth"))
    dw = float(np.nanmax(np.abs(np.r_[m.w_colour - m.fitted_w_colour,
                                      m.w_fog - m.fitted_w_fog])))
    checks.require("truth: weights (fit_firth vs the simulation's "
                   "fit_firth_batch)", dw <= TRUTH_WEIGHT_TOL and
                   len(m) == len(W), f"max difference {dw:.1e}")
    cls = ((m.class_colour == m.class_colour_truth)
           & (m.class_fog == m.class_fog_truth)).all()
    checks.require("truth: every class", bool(cls))
    real = tj["realised"]
    diff = []
    for claim in ("H1", "H2"):
        for k in ("predicted", "opposite", "verdict"):
            if claims[claim][k] != real[claim][k]:
                diff.append(f"{claim} {k}")
    if c1.get("available"):
        if c1["gate"]["fired"] != real["gate"]["fired"]:
            diff.append("gate")
        for f in ("colour", "fog"):
            if c1[f"C1_{f}"]["verdict"] != real[f"C1_{f}"]["verdict"]:
                diff.append(f"C1 {f} verdict")
            if abs(c1[f"C1_{f}"]["p"] - real[f"C1_{f}"]["p"]) > 1e-9:
                diff.append(f"C1 {f} p")
    checks.require("truth: H1, H2, attention check and C1 as realised",
                   not diff, ", ".join(diff))
    want = tj["expected_status_counts"]
    got = {s: {g: int(counts.loc[s, g]) for g in ("participant", "control")}
           for s in want}
    others = {s: int(counts.loc[s].sum()) for s in counts.index
              if s not in want}
    checks.require("truth: exclusion counts per step and group",
                   got == want and not any(others.values()),
                   json.dumps(got))
    tpart = tj["partial_exclusions_among_analysable_participants"]
    checks.require("truth: partial exclusions", partial == tpart,
                   json.dumps(partial))
    g = people.merge(tp[["session_id", "grid_shift_colour",
                         "grid_shift_fog"]], on="session_id",
                     suffixes=("", "_truth"))
    gd = []
    for f in ("colour", "fog"):
        a, b = g[f"grid_shift_{f}"], g[f"grid_shift_{f}_truth"]
        if not ((a.isna() == b.isna()).all()
                and (np.abs(a - b).fillna(0) <= 1e-9).all()):
            gd.append(f)
    checks.require("truth: grid shift per person", not gd, ", ".join(gd))
    exp = tj["design_expectation"]
    verdicts = {"H1": claims["H1"]["verdict"], "H2": claims["H2"]["verdict"]}
    if c1.get("available"):
        for f in ("colour", "fog"):
            verdicts[f"C1_{f}"] = c1[f"C1_{f}"]["verdict"]
    rows = []
    for key, e in exp.items():
        v = verdicts.get(key)
        miss = (v == "supported" if e["verdict"] == "no claim"
                else v != e["verdict"]) if e["kind"] != "likely" else False
        rows.append({"item": key, "expected": e["verdict"],
                     "kind": e["kind"], "verdict": v, "miss": miss,
                     "p_miss": e.get("p_miss", np.nan)})
        if e["kind"] == "fixed":
            checks.require(f"design: {key} {e['verdict']}", not miss, v)
    return pd.DataFrame(rows)


# ============================================================
# The false-claim check
# ============================================================

def false_claim_check(sim_root, L, th, rules, checks, out_dir):
    """1,000 simulated no-effect studies per cell, through this
    analysis's own per-person path (fit_firth, classify, the registered
    rule), against the registered rates: H1 and H2 separately, within the
    binomial 95% range of false_worst (worst taste null) or at or below
    its upper bound for false_guesser (random choosers). The union is
    descriptive."""
    print("\n" + "=" * 60)
    print("FALSE-CLAIM CHECK")
    print("=" * 60)
    d = os.path.join(sim_root, "false_claim")
    man = pd.read_csv(os.path.join(d, "manifest.csv"))
    rows = []
    for f in sorted(x for x in os.listdir(d) if x.endswith(".npz")):
        z = np.load(os.path.join(d, f))
        checks.require(f"{f}: trial order matches the frozen list",
                       np.array_equal(z["trial"], L.trial_an))
        Y = z["Y"].astype(float)
        S_, n, _ = Y.shape
        mult = float(z["taste_mult"])
        est = np.array([fit_firth(L.X_an, y) for y in Y.reshape(-1, L.n_an)])
        pred, opp = classify(est, th)
        claims = {}
        for k, feat in enumerate(("colour", "fog")):
            a = pred[:, k].reshape(S_, n).sum(1)
            b = opp[:, k].reshape(S_, n).sum(1)
            r = rules[(feat, n)]
            claims[feat] = rule_mask_count(n, int(r["k"]), int(r["m"]))[a, b]
        for feat, claim in (("fog", "H1"), ("colour", "H2")):
            r = rules[(feat, n)]
            col = "false_guesser" if mult == 0 else "false_worst"
            p = float(r[col])
            cnt = int(claims[feat].sum())
            lo = int(stats.binom.ppf(0.025, S_, p))
            hi = int(stats.binom.ppf(0.975, S_, p))
            ok = cnt <= hi if mult == 0 else lo <= cnt <= hi
            prev = man[(man.file == f) & (man.claim == claim)].preview_claims
            rows.append({"file": f, "null": str(z["null"]), "n": n,
                         "claim": claim, "rule": f"k >= {int(r['k'])}, "
                                                 f"m <= {int(r['m'])}",
                         "registered_column": col, "registered_rate": p,
                         "claims": cnt, "studies": S_, "rate": cnt / S_,
                         "binomial_95_lo": lo if mult else 0,
                         "binomial_95_hi": hi, "within": ok,
                         "simulation_preview": int(prev.iloc[0])
                         if len(prev) else np.nan})
            checks.note(f"false claims, {claim}, {f}", ok,
                        f"{cnt}/{S_} vs {col} {p:.4g}, allowed "
                        f"{lo if mult else 0}..{hi}")
        u = int((claims["colour"] | claims["fog"]).sum())
        rows.append({"file": f, "null": str(z["null"]), "n": n,
                     "claim": "H1 or H2 (descriptive)", "claims": u,
                     "studies": S_, "rate": u / S_})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out_dir, "false_claim_check.csv"), index=False)
    per = tab[tab.claim.isin(["H1", "H2"])]
    same = (per.claims == per.simulation_preview).all()
    checks.note("false claims: this analysis's counts equal the "
                "simulation's preview counts", bool(same))
    checks.note("false claims: every cell within the registered rate's "
                "Monte Carlo range", bool(per.within.all()),
                f"{int(per.within.sum())}/{len(per)} cells")
    return tab


# ============================================================
# One dataset
# ============================================================

def analyse_dataset(dataset_dir, out, sh, checks):
    """Sections A to G on one dataset; returns a summary dict."""
    t0 = time.time()
    for sub in ("tables", "figures", "PRIVATE"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    L, th, rules, screens, sens, caps = (sh[k] for k in (
        "L", "th", "rules", "screens", "sens", "caps"))
    tidy = (load_tidy(dataset_dir, checks) if INPUT_FORMAT == "tidy"
            else load_raw(dataset_dir, checks))
    sv, pt, sp, pp = load_private(dataset_dir, checks)
    T, G = tidy["trials"], tidy["grid"]

    # --- A ---
    S, counts = apply_exclusions(tidy, pt, L, checks)
    n_p = int(counts.loc["analysable", "participant"])
    n_c = int(counts.loc["analysable", "control"])
    checks.require(f"final N within the maxima ({N_MAX_PARTICIPANTS} "
                   f"participants, {N_MAX_CONTROLS} controls)",
                   n_p <= N_MAX_PARTICIPANTS and n_c <= N_MAX_CONTROLS,
                   f"N = {n_p} and {n_c}")
    n_beyond = counts.loc["beyond_maximum"]
    if n_beyond.sum():
        checks.info("beyond the maximum, listed and not analysed",
                    f"{int(n_beyond['participant'])} participants, "
                    f"{int(n_beyond['control'])} controls")
    codes = assign_pseudonyms(S)
    S["code"] = codes.reindex(S.session_id).to_numpy()

    # --- B ---
    W = fit_people(S, T, L, th)
    nonconv = W[W.score_max > 1e-6]
    checks.require("every Firth fit converged (score ~ 0)", nonconv.empty,
                   f"{len(nonconv)} not converged")
    used = S[S.status.isin(["analysable", "tier_D"])]
    q = quality_per_person(T, used, L)
    W = W.merge(q[["session_id", "fast_share"]], on="session_id")
    W["group_label"] = np.where(W.status == "tier_D", "tier_D", W.group)
    for f in ("colour", "fog"):
        W[f"w_{f}_capped"] = np.clip(W[f"w_{f}"], caps[f]["lower"],
                                     caps[f]["upper"])
    W["angle_deg"] = np.degrees(np.arctan2(W.w_fog, W.w_colour))
    Pw = W[W.group_label == "participant"]
    Cw = W[W.group_label == "control"]

    # --- C, D, E ---
    claims = existence_claims(Pw, rules)
    c1 = c1_analysis(Pw, Cw, checks)
    sec = secondary_tests(Pw, caps)

    # --- F ---
    prev = prevalence_table(Pw, sens)
    suz = suzuki_counts(T, used.session_id, L)
    words = word_scores(used)
    grid = grid_per_person(G, used, screens, checks)
    people = (W.merge(q.drop(columns=["fast_share"]), on="session_id")
              .merge(suz, on="session_id", how="left")
              .merge(words, on="session_id", how="left")
              .merge(grid, on="session_id", how="left",
                     suffixes=("", "_grid")))
    people = people.drop(columns=[c for c in people.columns
                                  if c.endswith("_grid")])
    cds = sv.set_index("ResponseId")[["cds_q02", "cds_q13", "cds_afs",
                                      "cds_total"]]
    people = people.join(cds, on="participant_id")
    people["code"] = codes.reindex(people.session_id).to_numpy()
    gorder = {"participant": 0, "control": 1, "tier_D": 2}
    people = (people.assign(_g=people.group_label.map(gorder))
              .sort_values(["_g", "code"]).drop(columns="_g")
              .reset_index(drop=True))
    suzt = suzuki_tests(people)
    h3 = h3_summary(people)
    pp_ = people[people.group_label == "participant"]
    e1 = []
    for i, (xcol, ycol, label) in enumerate(E1_PAIRS):
        r = spearman_boot(pp_[xcol], pp_[ycol], BOOTSTRAP_SEED + i)
        r.update({"pair": label, "description": xcol, "weight": ycol})
        e1.append(r)
    e1 = pd.DataFrame(e1)[["pair", "description", "weight", "n", "rho",
                           "ci_lo", "ci_hi", "boot_degenerate"]]
    gsum = grid_summary(people)
    dose = dose_response(T, people, L)
    partial = {
        "flat_word_ratings": int(pp_.flat_word_ratings.sum()),
        "no_word_block": int((~pp_.has_word_block.astype(bool)).sum()),
        "no_grid_shift_colour": int(pp_.grid_shift_colour.isna().sum()),
        "no_grid_shift_fog": int(pp_.grid_shift_fog.isna().sum()),
        "both_episode_fog_screens_none": int(
            pp_.grid_both_episode_fog_none.sum()),
        "no_state_block": int((~pp_.has_state_block.astype(bool)).sum()),
        "no_interview_date": int(pp_.delay_days.isna().sum()),
        "no_survey_row": int(pp_.cds_total.isna().sum())}
    truth_partial = {k: v for k, v in partial.items() if k != "no_survey_row"}

    # --- G: tables (shareable: pseudonyms only) ---
    tdir = os.path.join(out, "tables")
    share_cols = (["code", "group_label", "tier", "status",
                   "w_colour", "ci_lo_colour", "ci_hi_colour", "class_colour",
                   "w_fog", "ci_lo_fog", "ci_hi_fog", "class_fog",
                   "w_colour_capped", "w_fog_capped", "angle_deg",
                   "n_catch_fail", "fast_share", "consistency",
                   "rt_median_ms", "rt_q1_ms", "rt_q3_ms", "side_left_share",
                   "suzuki_sat_predicted", "follows_sat",
                   "suzuki_fog_predicted", "follows_fog",
                   "cds_q02", "cds_q13", "cds_afs", "cds_total",
                   "has_word_block", "flat_word_ratings", "foggy_rel",
                   "faded_rel", "grid_shift_colour", "grid_shift_fog",
                   "grid_episode_screens", "grid_normal_screens",
                   "grid_both_episode_fog_none", "grid_free_text",
                   "set_code", "cell_px", "gamma_estimate",
                   "device_pixel_ratio", "delay_days", "has_state_block"]
                  + STATE_ITEMS)
    people[share_cols].rename(columns={"group_label": "group"}).to_csv(
        os.path.join(tdir, "per_person.csv"), index=False)
    gcols = ["code", "group_label", "set_code", "cell_px"] + [
        c for c in people.columns if c.startswith("grid_")]
    people[gcols].rename(columns={"group_label": "group"}).to_csv(
        os.path.join(tdir, "grid_per_person.csv"), index=False)
    gsum.to_csv(os.path.join(tdir, "grid_summary.csv"), index=False)
    qcols = ["code", "group_label", "consistency", "rt_median_ms",
             "rt_q1_ms", "rt_q3_ms", "fast_share", "side_left_share",
             "n_catch_fail", "gamma_estimate", "device_pixel_ratio",
             "cell_px", "delay_days"] + STATE_ITEMS
    people[qcols].rename(columns={"group_label": "group"}).to_csv(
        os.path.join(tdir, "quality.csv"), index=False)
    prev.to_csv(os.path.join(tdir, "prevalence.csv"), index=False)
    sec.to_csv(os.path.join(tdir, "secondary_tests.csv"), index=False)
    suzt.to_csv(os.path.join(tdir, "suzuki_tests.csv"), index=False)
    h3.to_csv(os.path.join(tdir, "h3_balance.csv"), index=False)
    e1.to_csv(os.path.join(tdir, "e1_correlations.csv"), index=False)
    dose.to_csv(os.path.join(tdir, "dose_response.csv"), index=False)
    sh["cap_table"].to_csv(os.path.join(tdir, "cap_sweep.csv"), index=False)
    ex = S[["code", "group", "tier", "status", "n_catch_fail",
            "consent_given", "images_fit", "is_test", "is_dev",
            "is_complete", "completion_rank"]].sort_values("code")
    ex.to_csv(os.path.join(tdir, "exclusions_per_session.csv"), index=False)
    bm = S[S.status == "beyond_maximum"].sort_values(
        ["group", "completion_rank"])
    bm.assign(label="beyond the maximum: not analysed",
              maximum=np.where(bm.group == "participant", N_MAX_PARTICIPANTS,
                               N_MAX_CONTROLS))[
        ["code", "group", "tier", "completion_rank", "maximum", "label",
         "n_catch_fail"]].to_csv(os.path.join(tdir, "beyond_maximum.csv"),
                                 index=False)
    counts.to_csv(os.path.join(tdir, "exclusion_counts.csv"))
    tests = test_table(claims, c1, sec, suzt)
    tests.to_csv(os.path.join(tdir, "tests.csv"), index=False)

    # --- PRIVATE ---
    key = S[["code", "participant_id", "session_id", "group", "status"]]
    key.sort_values("code").to_csv(
        os.path.join(out, "PRIVATE", "pseudonym_key_PRIVATE.csv"),
        index=False)
    ft = G[(G.free_text.str.len() > 0) & G.session_id.isin(used.session_id)]
    ft.assign(code=codes.reindex(ft.session_id).to_numpy())[
        ["code", "block", "screen", "feature", "free_text"]].to_csv(
        os.path.join(out, "PRIVATE", "grid_free_text_PRIVATE.csv"),
        index=False)

    if MAKE_FIGURES:
        fig_weights(people, th, os.path.join(out, "figures",
                                             "weights_per_person.png"))
        fig_c1(people, c1, os.path.join(out, "figures",
                                        "c1_distributions.png"))
        fig_grid(people, os.path.join(out, "figures", "grid_shifts.png"))

    truth_rows = None
    if os.path.isfile(os.path.join(dataset_dir, "truth.json")):
        print("\n  comparing with the simulated truth")
        truth_rows = compare_with_truth(dataset_dir, S, W, claims, c1,
                                        truth_partial, people, counts,
                                        checks)
        truth_rows.to_csv(os.path.join(out, "truth_comparison.csv"),
                          index=False)
    res = {"dataset": dataset_dir, "S": S, "counts": counts, "people": people,
           "claims": claims, "c1": c1, "sec": sec, "prev": prev,
           "suzt": suzt, "h3": h3, "e1": e1, "gsum": gsum, "dose": dose,
           "partial": partial, "truth": truth_rows, "survey": sp,
           "ptable": pp, "elapsed": time.time() - t0}
    write_report(out, res, sh, checks)
    return res


def test_table(claims, c1, sec, suzt):
    rows = []
    for c in ("H1", "H2"):
        d = claims[c]
        rows.append({"analysis": f"{c} ({d['feature']})", "kind": "primary",
                     "n": d["n"], "statistic": f"{d['predicted']} predicted, "
                     f"{d['opposite']} opposite", "rule": d.get("rule", ""),
                     "p": np.nan, "verdict": d["verdict"],
                     "registered_false_claim": d.get("false_worst", np.nan),
                     "registered_power": d.get("planning_power", np.nan)})
    if c1.get("available"):
        g = c1["gate"]
        rows.append({"analysis": "attention check: catch failures",
                     "kind": "gate", "n": f"{c1['n_participants']}+"
                     f"{c1['n_controls']}", "p": g["p_catch"],
                     "verdict": "fires" if g["fired_catch"] else "no"})
        rows.append({"analysis": "attention check: responses < 300 ms",
                     "kind": "gate", "n": f"{c1['n_participants']}+"
                     f"{c1['n_controls']}", "p": g["p_rt"],
                     "verdict": "fires" if g["fired_rt"] else "no"})
        for f in ("colour", "fog"):
            r = c1[f"C1_{f}"]
            rows.append({"analysis": f"C1 {f}", "kind": "primary",
                         "n": f"{c1['n_participants']}+{c1['n_controls']}",
                         "statistic": f"U = {r['U']:.1f} ({r['method']})",
                         "p": r["p"], "verdict": r["verdict"]})
        z = c1["zero_catch"]
        for f in ("colour", "fog"):
            r = z.get(f"C1_{f}")
            rows.append({"analysis": f"C1 {f}, zero catch failures",
                         "kind": "secondary (registered)",
                         "n": f"{z['n_participants']}+{z['n_controls']}",
                         "p": r["p"] if r else np.nan,
                         "verdict": r["verdict"] if r else
                         f"not usable (< {z['min_n']} per group)"})
    for _, r in sec.iterrows():
        rows.append({"analysis": f"t-test and BF, {r.feature}"
                     + (" (capped)" if r.capped else ""),
                     "kind": "secondary", "n": r.n,
                     "statistic": (f"t = {r.t:.3f}, BF10 = "
                                   f"{r.BF10_one_sided:.3g}")
                     if "t" in r and pd.notna(r.get("t")) else "",
                     "p": r.get("p_one_sided", np.nan),
                     "verdict": r.get("bayes_reading", "")})
    for _, r in suzt.iterrows():
        rows.append({"analysis": f"Suzuki-style t-test, {r.trials}, "
                     f"{r.group}", "kind": "exploratory", "n": r.n,
                     "p": r.get("p_one_sided", np.nan),
                     "statistic": f"{r.n_following} following"})
    return pd.DataFrame(rows)


# ============================================================
# The report
# ============================================================

def write_report(out, res, sh, checks):
    L_ = []
    W = L_.append
    th, caps = sh["th"], sh["caps"]
    S, counts, people = res["S"], res["counts"], res["people"]
    W("STUDY ANALYSIS  (v4)  --  analyse_study_v4.py")
    W("=" * 72)
    W(f"dataset      {res['dataset']}  ({INPUT_FORMAT} tables)")
    W(f"run          {time.strftime('%Y-%m-%d %H:%M:%S')}")
    W(f"python {platform.python_version()}, numpy {np.__version__}, "
      f"pandas {pd.__version__}")
    W("Shareable: every person is a pseudonym. The key is in PRIVATE/.")
    W("")
    W("REGISTERED QUANTITIES USED")
    W("-" * 72)
    for j, f in enumerate(("colour", "fog")):
        W(f"  thresholds {f:7s} {th['upper'][j]:+.5f} / {th['lower'][j]:+.5f}"
          f"   (registered {REGISTERED_THRESHOLDS[f][0]:+.3f} / "
          f"{REGISTERED_THRESHOLDS[f][1]:+.3f})")
    for f in ("colour", "fog"):
        W(f"  cap        {f:7s} +/-{caps[f]['upper']:.5f}   (registered "
          f"{REGISTERED_CAP[f]:.3f}; {caps[f]['n_informative']} informative "
          f"trials at |predictor| >= {CAP_INFORMATIVE_SD})")
    W("")
    W("A. WHO IS ANALYSABLE")
    W("-" * 72)
    W(f"  {'step':28s} {'participants':>13s} {'controls':>9s}")
    labels = {"test_or_dev": "test or development session",
              "incomplete": "not complete",
              "repeat_not_used": "later session (listed)",
              "excluded_no_consent": "no consent",
              "excluded_images_fit": "images did not fit",
              "excluded_catch": ">= 3 catch failures",
              "tier_D": "Tier D (separate)",
              "beyond_maximum": "beyond the maximum (listed)",
              "analysable": "ANALYSABLE"}
    for st in STATUS_ORDER:
        W(f"  {labels[st]:28s} {counts.loc[st, 'participant']:13d} "
          f"{counts.loc[st, 'control']:9d}")
    rep = S[S.status == "repeat_not_used"]
    if len(rep):
        W(f"  later sessions, not used: {', '.join(sorted(rep.code))}")
    bm = S[S.status == "beyond_maximum"]
    if len(bm):
        W(f"  beyond the maximum (the first {N_MAX_PARTICIPANTS} participants "
          f"and {N_MAX_CONTROLS} controls by completion time are analysed):")
        W(f"    {', '.join(sorted(bm.code))}  -> tables/beyond_maximum.csv")
    p = res["partial"]
    W("  partial exclusions among analysable participants:")
    for k, v in p.items():
        W(f"    {k.replace('_', ' '):34s} {v}")
    n_p, n_c = int(counts.loc["analysable", "participant"]), \
        int(counts.loc["analysable", "control"])
    W(f"  FINAL N: {n_p} participants, {n_c} controls"
      f" (+ {int(counts.loc['tier_D', 'participant'])} Tier D, reported "
      "separately)")
    W("")
    W("B. PER-PERSON WEIGHTS  (Firth, no intercept; 95% profile intervals)")
    W("-" * 72)
    W(f"  {'code':6s} {'colour':>8s} {'interval':>19s} {'class':13s} "
      f"{'fog':>8s} {'interval':>19s} {'class':13s}")
    for _, r in people.iterrows():
        W(f"  {r.code:6s} {r.w_colour:+8.3f} [{fmt(r.ci_lo_colour, 2):>7s}, "
          f"{fmt(r.ci_hi_colour, 2):>7s}] {r.class_colour:13s} "
          f"{r.w_fog:+8.3f} [{fmt(r.ci_lo_fog, 2):>7s}, "
          f"{fmt(r.ci_hi_fog, 2):>7s}] {r.class_fog:13s}")
    W("")
    W("C. H1 (FOG) AND H2 (COLOUR)  -- primary, two separate claims at 5%")
    W("-" * 72)
    for c in ("H1", "H2"):
        d = res["claims"][c]
        W(f"  {c} {d['feature']}: {d['predicted']} predicted, "
          f"{d['opposite']} opposite, {d['not_detected']} not detected "
          f"(N = {d['n']})")
        if "rule" in d:
            W(f"     rule {d['rule']}  ->  {d['verdict'].upper()}")
            W(f"     registered false-claim rate {d['false_worst']:.4f} "
              f"(upper bound {d['false_worst_upper']:.4f}); power at the "
              f"planning cell {d['planning_power']:.3f}")
        else:
            W(f"     {d['why']}")
    W("")
    W("D. C1 (PARTICIPANTS AGAINST CONTROLS)  -- primary")
    W("-" * 72)
    c1 = res["c1"]
    if not c1.get("available"):
        W("  not computable (one group is empty)")
    else:
        g = c1["gate"]
        W(f"  people excluded for >= 3 catch failures: participants "
          f"{counts.loc['excluded_catch', 'participant']}, controls "
          f"{counts.loc['excluded_catch', 'control']}")
        W("  NOTE: the registered gate and C1 simulation did not apply that "
          "exclusion;")
        W("  here both run on the analysable people, after it.")
        W(f"  attention check: catch failures, controls worse, one-sided "
          f"p {fmt_p(g['p_catch'])};")
        W(f"                   responses < 300 ms, controls worse, one-sided "
          f"p {fmt_p(g['p_rt'])}")
        W(f"                   Holm across the two at {GATE_ALPHA}: "
          + ("FIRES -> C1 is INCONCLUSIVE" if g["fired"] else "does not fire"))
        for f in ("colour", "fog"):
            r = c1[f"C1_{f}"]
            W(f"  C1 {f:6s}: medians {r['median_participants']:+.3f} vs "
              f"{r['median_controls']:+.3f}, U = {r['U']:.1f}, one-sided p "
              f"{fmt_p(r['p'])} ({r['method']}), Holm "
              f"{'rejects' if r['holm_reject'] else 'does not reject'} "
              f"->  {r['verdict'].upper()}")
        if c1["descriptive"]:
            W(f"  fewer than {MIN_CONTROLS_C1} analysable controls: C1 is "
              "DESCRIPTIVE.")
        z = c1["zero_catch"]
        W(f"  zero catch failures (registered secondary): "
          f"{z['n_participants']} participants, {z['n_controls']} controls")
        if z["usable"]:
            for f in ("colour", "fog"):
                r = z[f"C1_{f}"]
                W(f"    {f:6s}: one-sided p {fmt_p(r['p'])}, Holm "
                  f"{'rejects' if r['holm_reject'] else 'does not reject'}")
            W(f"    fast-response re-check in the subsample (diagnostic): "
              f"{'fires' if z['rt_recheck_fires'] else 'does not fire'}")
        else:
            W(f"    not usable: fewer than {z['min_n']} in a group")
    W("")
    W("E. SECONDARY: one-sided t-test and JZS Bayes factor (scale 0.707)")
    W("-" * 72)
    for _, r in res["sec"].iterrows():
        tag = (f"capped at +/-{r.cap:.3f}, {r.n_clipped} clipped"
               if r.capped else "uncapped")
        if pd.notna(r.get("t", np.nan)):
            W(f"  {r.feature:6s} {tag:28s} mean {r['mean']:+.3f}, "
              f"t({r.df}) = {r.t:.3f}, p {fmt_p(r.p_one_sided)}, "
              f"BF10 {r.BF10_one_sided:.3g} ({r.bayes_reading})")
        else:
            W(f"  {r.feature:6s} {tag:28s} not computable (N = {r.n})")
    W("")
    W("F. EXPLORATORY AND DESCRIPTIVE  (intervals, no claims)")
    W("-" * 72)
    W("  Prevalence (descriptive; no 'excludes zero' inference):")
    for _, r in res["prev"].iterrows():
        if pd.notna(r.get("moment", np.nan)):
            W(f"    {r.feature:6s} {r.predicted}+/{r.opposite}- of {r.n}, "
              f"assuming a {r.assumed_responder} responder (sensitivity "
              f"{r.sensitivity:.2f}): estimate {r.moment:.2f}, mode "
              f"{r.posterior_mode:.2f}, 95% HPDI [{r.hpdi95_lo:.2f}, "
              f"{r.hpdi95_hi:.2f}]")
    W(f"  Suzuki-style (>= {SUZUKI_FOLLOW} of 60 counts as following):")
    for _, r in res["suzt"].iterrows():
        W(f"    {r.trials:18s} {r.group:12s} {r.n_following}/{r.n} following"
          + (f", mean {r.mean_proportion:.3f}, one-sided p "
             f"{fmt_p(r.p_one_sided)}" if pd.notna(r.get("t", np.nan))
             else ""))
    W("  H3, angle atan2(b_fog, b_sat) in degrees, between people only:")
    for _, r in res["h3"].iterrows():
        if r.n:
            W(f"    {r.group:12s} n {r.n}: median {r['median']:.1f}, IQR "
              f"[{r.q1:.1f}, {r.q3:.1f}], range [{r['min']:.1f}, "
              f"{r['max']:.1f}]; detected predicted on both "
              f"{r.detected_predicted_both}")
    W("  E1, descriptions against weights (participants; Spearman, "
      f"{BOOTSTRAP_REPS} bootstrap resamples):")
    for _, r in res["e1"].iterrows():
        W(f"    {r.pair:44s} n {r.n:2d}  rho {fmt(r.rho, 2):>5s}  "
          f"[{fmt(r.ci_lo, 2)}, {fmt(r.ci_hi, 2)}]")
    W("  E2, grid shift (episode minus normal, corpus SD):")
    for _, r in res["gsum"].iterrows():
        W(f"    {r.group:12s} {r.feature:6s} n {r.n_with_shift}/"
          f"{r.n_people}: mean {fmt(r.shift_mean)}, median "
          f"{fmt(r.shift_median)}; at original episode/normal "
          f"{r.episode_answers_at_original}/{r.normal_answers_at_original}")
    W("    (fog answers at the original are censored: the axis has no "
      "clearer column)")
    W("  Dose-response and additivity: tables/dose_response.csv")
    W("  Quality and covariates: tables/quality.csv")
    W("")
    if res["truth"] is not None:
        W("SIMULATED DATA: COMPARISON WITH THE TRUTH")
        W("-" * 72)
        for _, r in res["truth"].iterrows():
            W(f"  {r['item']:10s} expected {r.expected:14s} ({r.kind}) "
              f"got {r.verdict}" + ("  MISS" if r.miss else ""))
        W("")
    W("CHECKS")
    W("-" * 72)
    for r in checks.rows:
        W(f"  [{r['result']}] {r['check']}"
          + (f"  ({r['detail']})" if r["detail"] else ""))
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L_) + "\n")


# ============================================================
# Main
# ============================================================

def shared_setup(root, checks):
    try:
        facts = verify_frozen(FROZEN_ROOT, [])
    except (RuntimeError, FileNotFoundError) as e:
        checks.require("frozen package matches checksums.sha256", False,
                       str(e).splitlines()[0])
    checks.require("frozen package matches checksums.sha256", True,
                   f"{facts.get('n_files')} files, seed {facts.get('seed')}")
    path = os.path.join(FROZEN_ROOT, FROZEN_TRIAL_LIST)
    trials_an, X, _r = load_trials(path, "frozen")
    L = load_full_list(path, trials_an, X, checks)
    th = run_identity_check(X, facts, checks, root)
    got = {f: (round(float(th["upper"][j]), 3), round(float(th["lower"][j]),
                                                      3))
           for j, f in enumerate(("colour", "fog"))}
    checks.require("recomputed thresholds equal the registered ones (3 dp)",
                   got == REGISTERED_THRESHOLDS,
                   f"colour {got['colour']}, fog {got['fog']}")
    rules = load_rules(checks)
    screens = load_grid(checks)
    sens = load_sensitivities(checks)
    caps, cap_table = compute_caps(X, checks)
    inputs = {"trial_list.csv": sha256_of(path),
              "checksums.sha256": sha256_of(os.path.join(
                  FROZEN_ROOT, "checksums.sha256")),
              "combined_rule.csv": sha256_of(COMBINED_RULE),
              "grid_trials.csv": sha256_of(GRID_TRIALS),
              "detection_curve.csv": sha256_of(DETECTION_CURVE),
              "this script": sha256_of(os.path.abspath(__file__))}
    if os.path.isfile(POWER_SCRIPT):
        inputs[POWER_SCRIPT] = sha256_of(POWER_SCRIPT)
    return {"L": L, "th": th, "rules": rules, "screens": screens,
            "sens": sens, "caps": caps, "cap_table": cap_table,
            "facts": facts, "inputs": inputs}


def dataset_inputs(dataset_dir):
    d = os.path.join(dataset_dir, "tidy")
    out = {}
    for f in (TIDY_TRIALS, TIDY_SESSIONS, TIDY_GRID):
        p = os.path.join(d, f)
        if os.path.isfile(p):
            out[f"tidy/{f}"] = sha256_of(p)
    return out


def main():
    t0 = time.time()
    base = OUTPUT_ROOT + ("_validation" if RUN_MODE == "validate_sim" else "")
    root = resolve_output_root(base)
    os.makedirs(root)
    print(f"Writing to {root}")
    checks = Checks()
    meta = {"script": os.path.basename(__file__), "run_mode": RUN_MODE,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "output": root,
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": sys.modules["scipy"].__version__,
            "settings": settings_snapshot()}
    status = "complete"
    try:
        sh = shared_setup(root, checks)
        meta["inputs_sha256"] = sh["inputs"]
        meta["thresholds"] = {"upper": sh["th"]["upper"],
                              "lower": sh["th"]["lower"]}
        meta["caps"] = sh["caps"]
        if RUN_MODE == "one":
            meta["inputs_sha256"].update(dataset_inputs(DATASET_DIR))
            res = analyse_dataset(DATASET_DIR, root, sh, checks)
            # Name and checksum only: a full path can carry a user name.
            meta["private_inputs"] = {
                k: {"file": os.path.basename(v), "sha256": sha256_of(v)}
                for k, v in (("survey", res["survey"]),
                             ("participant_table", res["ptable"]))}
        elif RUN_MODE == "validate_sim":
            status = run_validation(root, sh, checks, meta)
        else:
            checks.require("RUN_MODE is 'one' or 'validate_sim'", False,
                           RUN_MODE)
    except CheckFailed as e:
        status = f"STOPPED: {e}"
    meta["status"] = status
    meta["checks"] = checks.rows
    meta["elapsed_s"] = round(time.time() - t0, 1)
    write_json(os.path.join(root, "run_meta.json"), meta)
    if status != "complete":
        print(f"\n{status}\nSee {root}/run_meta.json. Nothing in {root} "
              "should be used.")
        sys.exit(1)
    print(f"\nDone in {time.time() - t0:.0f} s. Report: "
          f"{os.path.join(root, 'report.txt')}")


def run_validation(root, sh, checks, meta):
    """Every simulated dataset against its truth, then the false claims."""
    ddir = os.path.join(SIM_ROOT, "datasets")
    names = sorted(os.listdir(ddir))
    rows, failed = [], []
    for name in names:
        print("\n" + "=" * 60)
        print(f"DATASET {name}")
        print("=" * 60)
        ck = Checks(prefix=f"[{name}] ")
        out = os.path.join(root, "datasets", name)
        os.makedirs(out, exist_ok=True)
        try:
            res = analyse_dataset(os.path.join(ddir, name), out, sh, ck)
            tr = res["truth"]
            for _, r in tr.iterrows():
                rows.append({"dataset": name, **r.to_dict()})
            ok = True
        except CheckFailed as e:
            failed.append(f"{name}: {e}")
            ok = False
        write_json(os.path.join(out, "run_meta.json"),
                   {"dataset": name, "status": "complete" if ok
                    else "STOPPED", "checks": ck.rows})
        checks.rows += [r for r in ck.rows if r["result"] != "PASS"
                        or "truth" in r["check"] or "design" in r["check"]]
        checks.note(f"dataset {name}: analysed and every truth check passed",
                    ok)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(root, "validation_verdicts.csv"), index=False)
    is_h = tab["item"].isin(["H1", "H2"])
    for kind, label, mask in (
            ("near_certain", "C1 inconclusive when controls lapse more",
             ~is_h),
            ("no_claim", "no claim under no effect, H1/H2", is_h),
            ("no_claim", "no claim under no effect, C1", ~is_h)):
        sel = tab[(tab.kind == kind) & mask]
        miss = sel[sel.miss.astype(bool)]
        p_any = 1 - float(np.prod(1 - sel.p_miss.fillna(0).to_numpy(float)))
        checks.note(f"{label} (single draws)", miss.empty,
                    f"{len(sel) - len(miss)}/{len(sel)}"
                    + (f"; a miss among these is expected {p_any:.0%} of "
                       "the time" if kind == "no_claim" else "")
                    + ("; missed: " + ", ".join(miss.dataset + " "
                                                + miss["item"])
                       if len(miss) else ""))
    fc = false_claim_check(SIM_ROOT, sh["L"], sh["th"], sh["rules"], checks,
                           root)
    write_validation_report(root, tab, fc, checks, failed)
    meta["datasets"] = names
    meta["datasets_failed"] = failed
    return "complete" if not failed else \
        f"STOPPED: {len(failed)} dataset(s) failed: {failed[0]}"


def write_validation_report(root, tab, fc, checks, failed):
    L_ = ["VALIDATION ON SIMULATED DATA  --  analyse_study_v4.py",
          "=" * 72, f"run {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"simulation {SIM_ROOT}", ""]
    L_.append(f"datasets analysed: {tab.dataset.nunique()}; failed: "
              f"{len(failed)}")
    for f in failed:
        L_.append(f"  FAILED {f}")
    L_ += ["", "VERDICTS AGAINST THE DESIGN (realised verdicts match the "
           "simulation's own in every dataset that passed)", "-" * 72]
    for name, d in tab.groupby("dataset"):
        L_.append(f"  {name}")
        for _, r in d.iterrows():
            L_.append(f"    {r['item']:10s} {r.verdict:14s} expected "
                      f"{r.expected} ({r.kind})"
                      + ("  MISS" if r.miss else ""))
    L_ += ["", "FALSE-CLAIM CHECK (this analysis's per-person path)",
           "-" * 72]
    for _, r in fc.iterrows():
        if r.claim in ("H1", "H2"):
            L_.append(f"  {r.file:24s} {r.claim} {r.rule:15s} "
                      f"{int(r.claims):4d}/{int(r.studies)} vs "
                      f"{r.registered_column} {r.registered_rate:.4g} "
                      f"[{int(r.binomial_95_lo)}..{int(r.binomial_95_hi)}] "
                      f"{'within' if r.within else 'OUTSIDE'}")
        else:
            L_.append(f"  {r.file:24s} {r.claim:24s} {int(r.claims):4d} "
                      f"({r.rate:.3f})")
    L_ += ["", "CHECKS (shared, plus every non-PASS and every truth check)",
           "-" * 72]
    for r in checks.rows:
        L_.append(f"  [{r['result']}] {r['check']}"
                  + (f"  ({r['detail']})" if r["detail"] else ""))
    with open(os.path.join(root, "validation_report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(L_) + "\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
