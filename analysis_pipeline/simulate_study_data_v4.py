"""
simulate_study_data_v4.py

Simulated study datasets with KNOWN ANSWERS, for developing and testing
analyse_study_v4.py without anyone's real data.

What it writes
--------------
Under OUTPUT_ROOT (data/sim_study_v4, or _2, _3, ... if that folder
already holds a run):

  datasets/<name>/tidy/trials.csv     one row per presented trial
  datasets/<name>/tidy/sessions.csv   one row per session file
  datasets/<name>/tidy/grid.csv       one row per grid screen
  datasets/<name>/private/survey_FAKE.xlsx
                                      ResponseId, Q01_score .. Q29_score
  datasets/<name>/private/participant_table_FAKE.csv
                                      participant_id, group, tier
  datasets/<name>/truth.json          the scenario, the expected verdicts,
                                      and the realised answers
  datasets/<name>/truth_people.csv    every simulated session: true
                                      weights, lapse rate, status, and
                                      the answers the analysis must find
  false_claim/<null>_n<N>.npz         1,000 no-effect studies per cell,
                                      choices only, for the analysis
                                      script's false-claim check
  false_claim/manifest.csv
  datasets.csv                        one row per dataset
  report.txt                          the human-readable record
  run_meta.json                       settings, input checksums, checks

The tidy tables are the SAME tables the analysis script's loader will
produce from real session CSVs. The analysis reads only these. Raw
session CSVs in the experiment's own format come later, once the
experiment update is merged and DATA_DICTIONARY.md describes it.

The generating model is the power script's
-------------------------------------------
Every person is built as simulate_power_v4_n10_20.py's `_c1_pool` builds
one, from four independent parts:

  taste      Normal(0, mult x that feature's upper threshold), on BOTH
             features independently -- the power script's taste null
  strategy   with probability `strategy`, +C1_STRATEGY_WEIGHT on both
             features ("pick the more edited image"), the same rate in
             both groups
  memory     the effect weight, participants only, in an EXACT share of
             the analysable participants (round(share x N)), which is
             the registered planning cell's definition
  attention  a personal lapse rate around the group mean, drawn by the
             power script's own `draw_attention`, which also gives the
             catch-failure count and the share of fast responses

Choices come from the power script's `simulate_lapsing` at that personal
lapse rate. The catch failures and the fast responses are then placed on
particular trials uniformly at random, so their counts have exactly the
power script's distribution.

Recruitment works as registered: a person with 3 or more catch failures
is not analysable, and recruitment continues until the dataset has its
stated number of analysable participants and controls. The people who
were excluded are in the data, as they would be in a real study.

Everything else -- word ratings, grid choices, session fields, CDS
scores -- is well-formed and loosely related to the weights, but not a
model of anything.

Scenarios
---------
  participants   null_random, null_taste_worst, fog_only, colour_only,
                 both, everyone   x   N = 9, 10, 15, 20
                 each with controls matched to the participants' taste
                 and lapse rate, no strategy, 30 controls
  controls       no_strategy, quarter_strategy, lapse_more
                 x   19 and 30 controls, against participants "both"
                 at N = 20
  messy          one dataset with every exclusion and partial exclusion
  messy_over_max 21 participants and 31 controls who pass every
                 exclusion: the first 20 and 30 by completion time are
                 analysed, the last to complete in each group is listed as
                 beyond the maximum
  false claims   null_random and null_taste_worst at N = 10, 15, 20,
                 1,000 studies each

What is copied, and why
-----------------------
The registered error rates were computed with simulate_power_v4_n10_20.py,
so the functions this script needs from it are COPIED VERBATIM and
marked as copies. At start-up the script checks that they still give the
power script's results on a fixed simulated dataset: against reference
values stored below, and, when POWER_SCRIPT is present, against the
power script itself, function by function and source by source.

Usage
-----
    python simulate_study_data_v4.py

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
import zlib
from xml.sax.saxutils import escape as xml_escape
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# SETTINGS
# ============================================================

# --- Inputs (relative to the project folder) ---
#
# FROZEN_ROOT / FROZEN_TRIAL_LIST
#   The sealed trial-list package. Verified against its own
#   checksums.sha256 before anything is read; any mismatch stops the run.
FROZEN_ROOT = "data/frozen_trial_list_v4"
FROZEN_TRIAL_LIST = "trial_list.csv"

# COMBINED_RULE
#   The registered (k, m) rule per feature and per N from 10 to 20.
COMBINED_RULE = "data/registration_v4/combined_rule.csv"

# GRID_TRIALS
#   The grid package's per-cell table: set, screen, level, the measured
#   change from the original (sd_from_original), and the display column.
GRID_TRIALS = "data/grid_block_v4/package/grid_trials.csv"

# POWER_SCRIPT
#   The power script the copied functions come from. OPTIONAL: when it is
#   present the start-up check also runs the power script's own
#   functions and compares them, and compares the source of every copy.
#   When it is absent the check uses the reference values stored in
#   IDENTITY_REFERENCE below, which were computed with it.
POWER_SCRIPT = "simulate_power_v4_n10_20.py"

# --- Output ---
#
# OUTPUT_ROOT
#   Never overwritten: a rerun writes to OUTPUT_ROOT_2, _3, ...
OUTPUT_ROOT = "data/sim_study_v4"

# --- Randomness ---
#
# MASTER_SEED
#   Every dataset draws from its own stream, seeded from this and the
#   dataset's name, so one dataset can be regenerated, or a scenario
#   added, without moving any other dataset.
MASTER_SEED = 20261020

# --- What is registered, stated so the run can check it ---
#
# REGISTERED_THRESHOLDS
#   (upper, lower) per feature, as stated in the registration (3 dp).
#   The thresholds this script uses are RECOMPUTED with the power
#   script's compute_floors on the frozen list and must round to these.
REGISTERED_THRESHOLDS = {"colour": (0.286, -0.290), "fog": (0.488, -0.486)}

# REGISTERED_RULES
#   (k, m) per N, from docs/v4_stimulus_power_explained.md section 13.
#   combined_rule.csv must agree with this table row for row.
REGISTERED_RULES = {
    "colour": {10: (4, 0), 11: (5, 2), 12: (5, 2), 13: (5, 1),
               14: (5, 1), 15: (6, 3), 16: (6, 2), 17: (6, 2),
               18: (6, 2), 19: (7, 3), 20: (7, 3)},
    "fog":    {10: (4, 0), 11: (5, 2), 12: (5, 2), 13: (5, 1),
               14: (5, 1), 15: (5, 1), 16: (5, 1), 17: (5, 1),
               18: (5, 1), 19: (5, 1), 20: (5, 1)},
}
N_MIN_CLAIM = 10          # below this, H1 and H2 are descriptive
# N_MAX_PARTICIPANTS / N_MAX_CONTROLS
#   The recruitment maxima. Beyond them the analysis keeps the first 20
#   participants and the first 30 controls by session COMPLETION time
#   (session_end_time) and lists the rest as beyond the maximum.
N_MAX_PARTICIPANTS = 20
N_MAX_CONTROLS = 30
MIN_CONTROLS_C1 = 20      # below this, C1 is descriptive
CATCH_FAIL_EXCLUDE = 3    # 3 or more failures of the 10 is not analysable

# C1_NULL_RATE_GUIDE
#   Per-feature C1 false-claim rate used ONLY to say how often a
#   single no-effect dataset should show a chance C1 claim. The power
#   document's no-effect grid puts it between 0.017 and 0.028; 0.025 is
#   the upper end of the usual range. Nothing is decided by it.
C1_NULL_RATE_GUIDE = 0.025

# --- Scenarios ---
#
# Each scenario is a population. Weights are PER CORPUS SD.
#   taste_mult  taste spread as a multiple of each feature's own upper
#               threshold (the power script's taste null)
#   lapse       group mean lapse rate; personal rates vary around it
#   share       share of analysable participants carrying the effect
#   w_colour, w_fog   the effect, added to the responders' weights
#   strategy    share of people (BOTH groups) using the shared-idea
#               strategy, +C1_STRATEGY_WEIGHT on both features
#
# The effect scenarios are the registered planning cell: half the
# participants responding, colour 0.50/SD, fog 0.70/SD, lapse 0.1, the
# others at the 0.5x taste spread (NONRESPONDER_SPREAD_MULT). In "both"
# and "everyone" the same responders carry both effects.
PARTICIPANT_SCENARIOS = {
    "null_random":      dict(taste_mult=0.0, lapse=0.1, share=0.0,
                             w_colour=0.0, w_fog=0.0, strategy=0.0),
    "null_taste_worst": dict(taste_mult=1.0, lapse=0.0, share=0.0,
                             w_colour=0.0, w_fog=0.0, strategy=0.0),
    "fog_only":         dict(taste_mult=0.5, lapse=0.1, share=0.5,
                             w_colour=0.0, w_fog=0.7, strategy=0.0),
    "colour_only":      dict(taste_mult=0.5, lapse=0.1, share=0.5,
                             w_colour=0.5, w_fog=0.0, strategy=0.0),
    "both":             dict(taste_mult=0.5, lapse=0.1, share=0.5,
                             w_colour=0.5, w_fog=0.7, strategy=0.0),
    "everyone":         dict(taste_mult=0.5, lapse=0.1, share=1.0,
                             w_colour=0.5, w_fog=0.7, strategy=0.0),
}
PARTICIPANT_NS = [9, 10, 15, 20]

# BASELINE_CONTROLS_N
#   The control group that goes with every participant scenario. Its
#   taste spread and lapse rate MATCH the participants', so a C1 claim
#   in a no-effect dataset is a false one and the attention check has
#   no real difference to find.
BASELINE_CONTROLS_N = 30

# CONTROL_SCENARIOS
#   Set against participants "both" at N = 20. `participant_overrides`
#   changes the participants in the same dataset: the strategy appears
#   in both groups at the same rate (the registered C1 assumption), and
#   "lapse_more" is the power script's (0.0, 0.2) lapse pair.
CONTROL_SCENARIOS = {
    "no_strategy":      dict(control=dict(taste_mult=0.5, lapse=0.1,
                                          strategy=0.0),
                             participant_overrides={}),
    "quarter_strategy": dict(control=dict(taste_mult=0.5, lapse=0.1,
                                          strategy=0.25),
                             participant_overrides=dict(strategy=0.25)),
    "lapse_more":       dict(control=dict(taste_mult=0.5, lapse=0.2,
                                          strategy=0.0),
                             participant_overrides=dict(lapse=0.0)),
}
CONTROL_NS = [19, 30]
CONTROL_SCENARIO_PARTICIPANTS = ("both", 20)

# --- The false-claim batches ---
#
# FALSE_CLAIM_NULLS
#   No-effect populations, at LAPSE 0 as the registered rates were
#   computed. "random" is every weight exactly zero; "taste_worst" is
#   the 1.0x taste spread, which is the worst null for every registered
#   rule (false_worst in combined_rule.csv is its rate).
FALSE_CLAIM_NULLS = {"null_random": 0.0, "null_taste_worst": 1.0}
FALSE_CLAIM_NS = [10, 15, 20]
FALSE_CLAIM_STUDIES = 1000

# --- The messy dataset ---
#
# MESSY_CORE_PARTICIPANTS / MESSY_CONTROLS
#   Analysable participants (Tiers A-C) and controls the messy dataset
#   ends with, after every exclusion. The participants are drawn from
#   PARTICIPANT_SCENARIOS["both"].
MESSY_CORE_PARTICIPANTS = 10
MESSY_CONTROLS = 20

# MESSY_OVER_MAX
#   The over-the-maximum dataset: analysable participants and controls
#   before the cap (one more than each maximum). In each group the person
#   who completes LAST started before the group's last starter, so only
#   the completion-time rule picks the right person to set aside.
MESSY_OVER_MAX = (21, 31)

# --- Session content ---
#
# WORD_IDS
#   The ten word ids, in the registered order: targets, close words,
#   the derealization baseline, fillers.
WORD_IDS = ["foggy", "faded", "veiled", "lifeless", "unreal",
            "dreamlike", "distant", "like_a_picture", "oppressive",
            "threatening"]
WORD_MAX = 6

# STATE_ITEMS / STATE_ITEM_MAX
#   The four CDS items in state format, rated 0-6 (0 = not at all,
#   6 = completely), the same scale as the word ratings, as specified in
#   the experiment changes.
STATE_ITEMS = ["state_q02", "state_q13", "state_q19", "state_q29"]
STATE_ITEM_MAX = 6

# CDS_ITEMS / AFS_ITEMS
#   CDS-29 item scores: frequency 0-4 plus duration 1-6 when the item is
#   endorsed, so 0 or 2-10. AfS is the sum of items 2, 13, 19 and 29.
CDS_ITEMS = 29
AFS_ITEMS = [2, 13, 19, 29]
SURVEY_EXTRA_RESPONDENTS = 15   # pool respondents who never took part

# CATCH_KEYS
#   The expected key on each of the 10 catch trials, in presentation
#   order, balanced 5 and 5. A placeholder: the real sequence is in the
#   experiment's config.js and arrives in the data as catch_key.
CATCH_KEYS = ["up", "down", "down", "up", "up",
              "down", "up", "down", "down", "up"]

# RT model (ms). Engaged responses are 300 ms or slower by construction;
# the fast ones are exactly those `draw_attention` asked for.
RT_ENGAGED_MEDIAN = 1100.0
RT_ENGAGED_SIGMA = 0.55
RT_FAST_RANGE = (150, 299)

# Grid model, in LEVEL-INDEX units (positive = less colourful / foggier).
# normal block: latent level ~ Normal(mean, sd); the episode block adds
# slope x the person's true weight and more noise.
GRID_NORMAL = {"colour": (0.0, 0.7), "fog": (-0.6, 0.6)}
GRID_EPISODE_SLOPE = {"colour": 2.0, "fog": 1.6}
GRID_EPISODE_SD = 0.7
GRID_SCREEN_SD = 0.4
GRID_NONE_P = 0.04       # "none of these", episode block only
GRID_WINDOWS = [(1280, 720), (1366, 768), (1440, 900), (1536, 864),
                (1920, 1080), (2560, 1440), (1120, 680)]

# Tidy-table file names, shared with the analysis script.
TIDY_TRIALS = "trials.csv"
TIDY_SESSIONS = "sessions.csv"
TIDY_GRID = "grid.csv"
SURVEY_FILE = "survey_FAKE.xlsx"
PARTICIPANT_TABLE_FILE = "participant_table_FAKE.csv"

# Test IDs as the experiment marks them (DATA_DICTIONARY.md).
TEST_IDS = ["Z_21121989", "CHAIN_2026"]
ID_PATTERN = r"^R_[A-Za-z0-9]{15}$"

# --- The start-up identity check ---
#
# IDENTITY_SEED / IDENTITY_W
#   The fixed simulated dataset the copies are checked on: eight people
#   with these true weights on the frozen list's 180 analysed trials,
#   answering at lapse 0.1. The last two sit near separation and right
#   at the thresholds, which is where solvers disagree if they differ.
IDENTITY_SEED = 20261020
IDENTITY_W = [[0.0, 0.0], [0.5, 0.0], [0.0, 0.7], [0.5, 0.7],
              [-0.3, 0.2], [1.5, -0.4], [3.0, 3.0], [0.29, 0.49]]
# Tolerance: |a - b| <= IDENTITY_ABS_TOL + IDENTITY_REL_TOL * |b|.
# Rounding only: an algorithmic difference moves results by far more.
IDENTITY_ABS_TOL = 1e-9
IDENTITY_REL_TOL = 1e-9


# ============================================================
# COPIED SETTINGS
#
# Values identical to simulate_power_v4_n10_20.py. The copied functions
# below read them as globals. When POWER_SCRIPT is present the start-up
# check compares every one of them with the power script's own value.
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
NONRESPONDER_SPREAD_MULT = 0.5
TASTE_SPREADS_IN_FLOORS = [0.25, 0.5, 0.75, 1.0]
ALPHA = 0.05
FLOOR_ALPHA = 0.025
N_FLOOR_PEOPLE = 20_000
DETECTION_SIDED = "two"
C1_STRATEGY_WEIGHT = 0.5
C1_EXACT_MAX_N = 20
C1_HOLM = True
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
    "FIRTH_MAX_STEP", "NONRESPONDER_SPREAD_MULT", "TASTE_SPREADS_IN_FLOORS",
    "ALPHA", "FLOOR_ALPHA", "N_FLOOR_PEOPLE", "DETECTION_SIDED",
    "C1_STRATEGY_WEIGHT", "C1_EXACT_MAX_N", "C1_HOLM", "GATE_ALPHA",
    "N_CATCH_TRIALS", "CATCH_P_FAIL_PER_LAPSE", "LAPSE_BETWEEN_SD",
    "RT_GATE", "RT_FAST_CUTOFF_MS", "RT_FAST_GIVEN_LAPSE",
    "RT_FAST_GIVEN_ENGAGED", "RANDOM_SEED",
]

COPIED_FUNCTIONS = [
    "fit_logit_batch", "fit_firth", "fit_firth_batch", "fit_batch",
    "simulate_many", "lapse_p", "simulate_lapsing", "sha256_of",
    "verify_frozen", "load_trials", "compute_floors", "classify",
    "rule_mask_count", "mw_exact_sf", "mw_p_batch", "_c1_gate", "_c1_test",
    "draw_attention",
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
    'lapse_p/lapse0.2/[3,17]': 0.599771613299786,
    'lapse_p/lapse0.2/sum': 736.3802812058475,
    'lapse_p/lapse0/[3,17]': 0.6247145166247325,
    'lapse_p/lapse0/sum': 740.4753515073091,
    'load_trials/X_sha256': 'ecb4c8a53546a5b10535cb19f5baac3e21dc4e3a1673c5a9c2d98131f2dd20ee',
    'mw_exact_sf/5,6/[20]': 0.21428571428571427,
    'mw_exact_sf/5,6/sum': 16.0,
    'mw_p_batch/asym/method': 'asymptotic',
    'mw_p_batch/asym/p': [0.002140310584960722, 0.07078690058441392, 0.24403704658262404, 0.014223516488281518, 0.08707084160752154],
    'mw_p_batch/exact/method': 'exact',
    'mw_p_batch/exact/p': [0.04894022386282448, 0.3392156862745098, 0.04124791617051679, 0.23633404778915615, 0.04894022386282448],
    'mw_p_batch/ties/method': 'asymptotic',
    'mw_p_batch/ties/p': [0.06239791329212847, 0.07631931163686224, 0.30006595575169154, 0.11095632451944115, 0.0043147765520927354],
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
    a property of the simulated data rather than a fault (an unlucky
    draw), and does not stop it.
    """

    def __init__(self):
        self.rows = []

    def require(self, name, ok, detail=""):
        self.rows.append({"check": name, "result": "PASS" if ok else "FAIL",
                          "stops_run": True, "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail else ""))
        if not ok:
            raise CheckFailed(f"{name}: {detail}")

    def note(self, name, ok, detail=""):
        self.rows.append({"check": name, "result": "PASS" if ok else "FAIL",
                          "stops_run": False, "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail else ""))

    def info(self, name, detail=""):
        self.rows.append({"check": name, "result": "INFO",
                          "stops_run": False, "detail": str(detail)})
        print(f"  [INFO] {name}" + (f"  ({detail})" if detail else ""))


def resolve_output_root():
    """OUTPUT_ROOT, or the first free OUTPUT_ROOT_<n>. Never overwrites."""
    def occupied(d):
        return os.path.exists(d) and (not os.path.isdir(d)
                                      or len(os.listdir(d)) > 0)
    if not occupied(OUTPUT_ROOT):
        return OUTPUT_ROOT
    n = 2
    while occupied(f"{OUTPUT_ROOT}_{n}"):
        n += 1
    return f"{OUTPUT_ROOT}_{n}"


def dataset_rng(name):
    """An independent stream per dataset, from MASTER_SEED and the name."""
    return np.random.default_rng(
        np.random.SeedSequence([MASTER_SEED, zlib.crc32(name.encode())]))


def jsonable(x):
    """For json.dump: numpy scalars, arrays, tuples, NaN."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if not np.isfinite(x) else x
    return x


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(obj), fh, indent=1, ensure_ascii=False)


def settings_snapshot():
    """Every upper-case setting of this script, for run_meta.json."""
    out = {}
    for k, v in list(globals().items()):
        if k.isupper() and not k.startswith("_") and not callable(v) \
                and not isinstance(v, type(sys)):
            if k == "IDENTITY_REFERENCE":
                continue
            try:
                json.dumps(jsonable(v))
                out[k] = jsonable(v)
            except TypeError:
                out[k] = repr(v)
    return out


# ============================================================
# The start-up identity check
# ============================================================

def identity_values(ns, X):
    """What the copied functions produce on one fixed dataset.

    Called twice with the same code: once with this script's own copies
    (globals()) and, when the power script is present, once with the
    power script's module namespace. Each function is then run with its
    own module's settings, so a drifted setting shows up too.

    Returns a flat dict of quantity -> value (floats and hex digests),
    plus "_th", the thresholds compute_floors returned, which the run
    goes on to use.
    """
    g = (lambda k: ns[k]) if isinstance(ns, dict) else \
        (lambda k: getattr(ns, k))
    v = {}

    def h(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

    W = np.array(IDENTITY_W, dtype=float)
    v["load_trials/X_sha256"] = h(np.round(X, 12))

    Y = g("simulate_lapsing")(X, W, 0.1,
                              np.random.default_rng(IDENTITY_SEED))
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
    Fa = r4.binomial(180, 0.01, size=(5, 10)) / 180.0
    Fb = r4.binomial(180, 0.016, size=(5, 20)) / 180.0
    fired, fc, fr = g("_c1_gate")(Ai, Bi, Fa, Fb)
    v["_c1_gate/fired"] = "".join("T" if x else "F" for x in fired)
    v["_c1_gate/catch"] = "".join("T" if x else "F" for x in fc)
    v["_c1_gate/rt"] = "".join("T" if x else "F" for x in fr)
    Ap = r4.normal(size=(5, 10, 2)) + np.array([0.9, 0.2])
    Ac = r4.normal(size=(5, 20, 2))
    rc, rf = g("_c1_test")(Ap, Ac)
    v["_c1_test/colour"] = "".join("T" if x else "F" for x in rc)
    v["_c1_test/fog"] = "".join("T" if x else "F" for x in rf)
    v["_th"] = th
    return v


def compare_values(mine, theirs):
    """Rows of (quantity, mine, theirs, ok) over the union of keys."""
    rows = []
    for k in sorted(set(mine) | set(theirs)):
        a, b = mine.get(k), theirs.get(k)
        if a is None or b is None:
            ok = False
        elif isinstance(b, str) or isinstance(a, str):
            ok = a == b
        else:
            aa, bb = np.atleast_1d(np.asarray(a, float)), \
                np.atleast_1d(np.asarray(b, float))
            ok = (aa.shape == bb.shape and bool(np.all(
                np.abs(aa - bb) <= IDENTITY_ABS_TOL
                + IDENTITY_REL_TOL * np.abs(bb))))
        rows.append({"quantity": k, "this_script": a, "reference": b,
                     "ok": ok})
    return rows


def function_sources(path, names):
    """{name: ast.dump(FunctionDef)} for the named top-level functions."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = ast.dump(node)
    return out


def load_power_module(path):
    spec = importlib.util.spec_from_file_location("power_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_identity_check(X, own_facts, checks, out_dir):
    """Do the copied functions still give the power script's results?

    1. Against IDENTITY_REFERENCE, values the power script itself
       produced on this dataset. Always run.
    2. When POWER_SCRIPT is present: against the power script run now,
       the source of every copied function (as a syntax tree, so
       formatting cannot matter and any change in code or docstring
       does), every copied setting, and verify_frozen's verdict.
    """
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
        "copied functions reproduce the stored reference values",
        not bad, f"{len(rows) - len(bad)}/{len(rows)} quantities agree"
        + (f"; differ: {', '.join(bad[:6])}" if bad else ""))

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
            "copied functions agree with the power script run now",
            not bad, f"{len(lrows) - len(bad)}/{len(lrows)} quantities"
            + (f"; differ: {', '.join(bad[:6])}" if bad else ""))

        mine_src = function_sources(os.path.abspath(__file__),
                                    COPIED_FUNCTIONS)
        theirs_src = function_sources(POWER_SCRIPT, COPIED_FUNCTIONS)
        diff = [n for n in COPIED_FUNCTIONS
                if mine_src.get(n) is None
                or mine_src.get(n) != theirs_src.get(n)]
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
                       their_facts == own_facts,
                       "same FROZEN.json facts" if their_facts == own_facts
                       else "facts differ")
    else:
        checks.info("live comparison with the power script skipped",
                    f"{POWER_SCRIPT} not found; the stored reference "
                    "values were computed with it")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "identity_check.csv"),
                              index=False)
    return th


# ============================================================
# Inputs
# ============================================================

class ListInfo:
    """The frozen list, arranged for simulation."""


def load_full_list(path, trials_an, X, checks):
    """All 211 presented trials, and the analysed ones in X's order."""
    tl = pd.read_csv(path).sort_values("trial").reset_index(drop=True)
    L = ListInfo()
    L.tl = tl
    an = tl.analysed.astype(bool).to_numpy()
    L.idx_an = np.flatnonzero(an)
    L.idx_catch = np.flatnonzero(tl.trial_type.eq("catch").to_numpy())
    L.idx_ex = np.flatnonzero(~an & ~tl.trial_type.eq("catch").to_numpy())
    L.trial_an = tl.trial.to_numpy()[L.idx_an]
    L.X_an = X
    L.X_ex = tl.loc[L.idx_ex, [PRED_SAT, PRED_FOG]].to_numpy(float)
    L.a_is_left = tl.a_is_left.astype(bool).to_numpy()
    L.n = len(tl)
    L.n_an = len(L.idx_an)
    same = (np.array_equal(tl.trial.to_numpy()[L.idx_an],
                           trials_an.trial.to_numpy())
            and np.array_equal(tl.loc[L.idx_an, [PRED_SAT, PRED_FOG]]
                               .to_numpy(float), X))
    counts = tl.trial_type.value_counts().to_dict()
    want = {"sat_only": 60, "fog_only": 60, "together": 30,
            "conflict": 30, "consistency": 15, "catch": 10, "practice": 6}
    checks.require("frozen list: 211 trials, registered counts, analysed "
                   "rows in load_trials' order",
                   L.n == 211 and L.n_an == 180 and counts == want and same
                   and len(L.idx_catch) == N_CATCH_TRIALS,
                   f"{L.n} trials, {L.n_an} analysed")
    return L


def load_rules(checks):
    cr = pd.read_csv(COMBINED_RULE)
    need = {"feature", "n", "k", "m", "false_guesser", "false_worst",
            "false_worst_upper", "planning_power", "claim"}
    checks.require("combined_rule.csv has the expected columns",
                   need <= set(cr.columns),
                   f"missing {sorted(need - set(cr.columns))}"
                   if not need <= set(cr.columns) else "")
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
            if not (row.false_worst_upper <= 0.05):
                bad.append(f"{feat} N={n}: false_worst_upper "
                           f"{row.false_worst_upper:.4f}")
            rules[(feat, n)] = {c: row[c] for c in row.index}
    checks.require("combined_rule.csv equals the registered rule at every "
                   "N, false-claim upper bound <= 0.05",
                   not bad and len(cr) == 22,
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
                "row").scene_id.tolist()),
        }
    sets = {s: sorted(sc for (ss, sc) in screens if ss == s)
            for s in ("A", "B")}
    checks.require("grid_trials.csv: 2 sets x 4 screens x 3 rows x 6 "
                   "columns, one feature per screen",
                   len(g) == 144 and not problems and len(screens) == 8
                   and all(len(v) == 4 for v in sets.values()),
                   "; ".join(problems))
    return {"screens": screens, "sets": sets}


# ============================================================
# People
# ============================================================

_ID_CHARS = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                          "abcdefghijklmnopqrstuvwxyz0123456789"))


class IdMaker:
    """Unique IDs in the real pattern: R_ plus 15 letters and digits."""

    def __init__(self, rng):
        self.rng, self.used = rng, set()

    def __call__(self):
        while True:
            pid = "R_" + "".join(self.rng.choice(_ID_CHARS, 15))
            if pid not in self.used:
                self.used.add(pid)
                return pid


def make_person(rng, ids, group, scen, th, responder, attention,
                tier="", role="", pid=None):
    """One person, built as the power script's _c1_pool builds one."""
    lam, catch, fast = attention
    taste = rng.normal(0.0, 1.0, size=2) * (th["upper"] * scen["taste_mult"])
    strategy = bool(rng.random() < scen.get("strategy", 0.0))
    w = taste.copy()
    if strategy:
        w += C1_STRATEGY_WEIGHT
    effect = np.zeros(2)
    if responder:
        effect = np.array([scen.get("w_colour", 0.0), scen.get("w_fog", 0.0)])
        w += effect
    return {"participant_id": pid or ids(), "group": group, "tier": tier,
            "role": role, "w": w, "taste": taste, "strategy": strategy,
            "responder": bool(responder), "effect": effect,
            "lapse": float(lam), "catch": int(catch), "fast": float(fast)}


def draw_one_attention(rng, lapse_mean, n_trials):
    lam, catch, fast = draw_attention(1, lapse_mean, n_trials, rng)
    return float(lam[0]), int(catch[0]), float(fast[0])


def build_population(rng, ids, group, n_target, scen, th, n_trials,
                     record_recruits=True):
    """Recruit until n_target people are analysable on the catch rule.

    Exactly round(share x n_target) of the analysable participants carry
    the effect. A recruit with CATCH_FAIL_EXCLUDE or more failures is not
    analysable; with record_recruits they stay in the dataset as a
    session to be excluded (their effect status drawn at the share), and
    recruitment continues. Attention is independent of the weights, so
    this leaves the analysable people's weights exactly as specified.
    """
    share = scen.get("share", 0.0) if group == "participant" else 0.0
    n_resp = int(round(share * n_target))
    resp = np.zeros(n_target, dtype=bool)
    resp[rng.permutation(n_target)[:n_resp]] = True
    ok, excluded = [], []
    for slot in range(n_target):
        while True:
            att = draw_one_attention(rng, scen["lapse"], n_trials)
            if att[1] < CATCH_FAIL_EXCLUDE:
                break
            if record_recruits:
                excluded.append(make_person(
                    rng, ids, group, scen, th,
                    bool(rng.random() < share), att, role="recruit"))
        ok.append(make_person(rng, ids, group, scen, th, bool(resp[slot]),
                              att, role="core"))
    return ok, excluded


# ============================================================
# Sessions: trials, grid, words, state items
# ============================================================

def simulate_trials(rng, person, L, session_id):
    """All 211 presented trials of one session, in file order."""
    w = np.asarray(person["w"], float)[None, :]
    lam = person["lapse"]
    chose = np.full(L.n, np.nan)
    chose[L.idx_an] = simulate_lapsing(L.X_an, w, lam, rng)[0]
    chose[L.idx_ex] = simulate_lapsing(L.X_ex, w, lam, rng)[0]

    fast = np.zeros(L.n, dtype=bool)
    n_fast = int(round(person["fast"] * L.n_an))
    fast[L.idx_an[rng.choice(L.n_an, n_fast, replace=False)]] = True
    p_fast = lam * RT_FAST_GIVEN_LAPSE + (1 - lam) * RT_FAST_GIVEN_ENGAGED
    fast[L.idx_ex] = rng.random(len(L.idx_ex)) < p_fast

    rt = 300.0 + rng.lognormal(np.log(RT_ENGAGED_MEDIAN - 300.0),
                               RT_ENGAGED_SIGMA, L.n)
    rt[fast] = rng.integers(RT_FAST_RANGE[0], RT_FAST_RANGE[1] + 1,
                            int(fast.sum()))
    rt = np.round(rt)

    image = ~np.isnan(chose)
    side = np.full(L.n, None, dtype=object)
    picked_left = np.where(chose == 1, L.a_is_left, ~L.a_is_left)
    side[image] = np.where(picked_left[image], "left", "right")
    key = side.copy()

    catch_key = np.full(L.n, None, dtype=object)
    catch_pass = np.full(L.n, None, dtype=object)
    fails = set(rng.choice(N_CATCH_TRIALS, person["catch"],
                           replace=False).tolist())
    for j, pos in enumerate(L.idx_catch):
        want = CATCH_KEYS[j]
        catch_key[pos] = want
        if j in fails:
            key[pos] = rng.choice([k for k in ("up", "down", "left", "right")
                                   if k != want])
            catch_pass[pos] = False
        else:
            key[pos] = want
            catch_pass[pos] = True
        rt[pos] = np.round(300.0 + rng.lognormal(np.log(500.0), 0.4))

    tl = L.tl
    return pd.DataFrame({
        "session_id": session_id,
        "participant_id": person["participant_id"],
        "group": person["group"],
        "trial": tl.trial.to_numpy(),
        "trial_type": tl.trial_type.to_numpy(),
        "analysed": tl.analysed.astype(bool).to_numpy(),
        "key_pressed": key,
        "side_pressed": side,
        "chose_A": pd.Series(chose).astype("Int64").to_numpy(),
        "rt_ms": rt.astype(int),
        "catch_key": catch_key,
        "catch_pass": catch_pass,
        "repeat_of": tl.repeat_of.astype("Int64").to_numpy(),
    })


def cell_px(window):
    """The grid cell size, by the package's interface_spec.md formula."""
    w, h = window
    by_w = (w - 48 - 5 * 22) / 6 - 14
    by_h = (h - 190 - 2 * 6) / 3
    return int(max(100, np.floor(min(by_w, by_h, 240))))


FREE_TEXTS = ["(simulated) greyer than any of these",
              "(simulated) as if behind a dirty window",
              "(simulated) flatter, but not darker",
              "(simulated) hard to say"]


def simulate_grid(rng, person, session_id, set_code, window, grid,
                  missing_blocks=(), none_fog_both=False):
    """Both grid blocks; returns (rows, realised truth)."""
    ep_set, no_set = ("A", "B") if set_code == 0 else ("B", "A")
    w = {"colour": float(person["w"][0]), "fog": float(person["w"][1])}
    lat_n = {f: rng.normal(*GRID_NORMAL[f]) for f in ("colour", "fog")}
    lat_e = {f: lat_n[f] + GRID_EPISODE_SLOPE[f] * w[f]
             + rng.normal(0.0, GRID_EPISODE_SD) for f in ("colour", "fog")}
    px = cell_px(window)
    rows = []
    vals = {(b, f): [] for b in ("episode", "normal")
            for f in ("colour", "fog")}
    for block, sset, lat in (("episode", ep_set, lat_e),
                             ("normal", no_set, lat_n)):
        if block in missing_blocks:
            continue
        for pos, sc in enumerate(grid["sets"][sset], start=1):
            info = grid["screens"][(sset, sc)]
            f = info["feature"]
            none = block == "episode" and (
                (none_fog_both and f == "fog") or rng.random() < GRID_NONE_P)
            if none:
                level, disp = None, None
                text = str(rng.choice(FREE_TEXTS))
            else:
                lo, hi = info["levels"][0], info["levels"][-1]
                level = int(np.clip(np.rint(lat[f] + rng.normal(
                    0.0, GRID_SCREEN_SD)), lo, hi))
                disp = info["disp"][level]
                text = ""
                vals[(block, f)].append(info["sd_mean"][level])
            rows.append({
                "session_id": session_id,
                "participant_id": person["participant_id"],
                "group": person["group"], "block": block,
                "set_code": set_code, "scene_set": sset, "screen": sc,
                "screen_position": pos, "feature": f,
                "column_order": info["column_order"],
                "scene_ids": info["scene_ids"],
                "chosen_display_column": disp,
                "chosen_level_index": level, "chose_none": bool(none),
                "free_text": text,
                "rt_ms": int(np.round(rng.lognormal(np.log(6000), 0.5))),
                "cell_px": px, "inner_w": window[0], "inner_h": window[1]})
    truth = {}
    for f in ("colour", "fog"):
        e, n = vals[("episode", f)], vals[("normal", f)]
        truth[f"grid_episode_mean_sd_{f}"] = float(np.mean(e)) if e else np.nan
        truth[f"grid_normal_mean_sd_{f}"] = float(np.mean(n)) if n else np.nan
        truth[f"grid_shift_{f}"] = (float(np.mean(e) - np.mean(n))
                                    if e and n else np.nan)
    for block in ("episode", "normal"):
        br = [r for r in rows if r["block"] == block]
        truth[f"grid_{block}_fog_at_original"] = sum(
            1 for r in br if r["feature"] == "fog"
            and r["chosen_level_index"] == 0)
        truth[f"grid_{block}_unedited"] = sum(
            1 for r in br if r["chosen_level_index"] == 0)
        truth[f"grid_{block}_none"] = sum(1 for r in br if r["chose_none"])
        truth[f"grid_{block}_present"] = len(br) > 0
    truth["grid_both_fog_none"] = (
        sum(1 for r in rows if r["block"] == "episode"
            and r["feature"] == "fog" and r["chose_none"]) == 2)
    return rows, truth


def simulate_words(rng, person, flat=False):
    """Ten 0-6 ratings, loosely following the person's true weights."""
    if flat:
        v = int(rng.integers(1, WORD_MAX))
        return {f"word_{w}": v for w in WORD_IDS}
    base = rng.uniform(1.5, 4.5)
    wc, wf = float(person["w"][0]), float(person["w"][1])
    eff = {"foggy": 1.6 * wf / 0.7, "faded": 1.6 * wc / 0.5,
           "veiled": 0.8 * wf / 0.7, "lifeless": 0.8 * wc / 0.5,
           "oppressive": -1.0, "threatening": -1.0}
    out = {}
    for w in WORD_IDS:
        out[f"word_{w}"] = int(np.clip(np.rint(
            base + eff.get(w, 0.0) + rng.normal(0.0, 0.9)), 0, WORD_MAX))
    if len(set(out.values())) == 1:          # not flat unless asked
        out["word_oppressive"] = (out["word_oppressive"] + 1) % (WORD_MAX + 1)
    return out


def simulate_state(rng, person):
    hi = 4 if person["group"] == "participant" else 2
    return {s: int(rng.integers(0, min(hi, STATE_ITEM_MAX) + 1))
            for s in STATE_ITEMS}


def cds_item(rng, p_endorse, f_boost=0.0):
    if rng.random() >= p_endorse:
        return 0
    f = int(np.clip(1 + rng.poisson(1.0 + f_boost), 1, 4))
    d = int(rng.integers(1, 7))
    return f + d


def simulate_cds(rng, group, tier, w):
    """29 item scores consistent with the group and tier definitions."""
    afs_idx = [i - 1 for i in AFS_ITEMS]
    other = [i for i in range(CDS_ITEMS) if i not in afs_idx]
    s = np.zeros(CDS_ITEMS, dtype=int)
    if group == "control":
        for i in other:
            s[i] = cds_item(rng, 0.12)
        while s.sum() > 20:
            nz = [i for i in other if s[i] > 0]
            s[rng.choice(nz)] = 0
        return s
    if tier == "D":
        for i in other:
            s[i] = cds_item(rng, 0.97, 2.0)
        while s.sum() < 70:
            i = rng.choice(other)
            s[i] = min(10, s[i] + 2 if s[i] else 2)
        s[afs_idx[0]] = 2 if rng.random() < 0.5 else 0
        return s
    boost13 = max(float(w[1]), 0.0) * 2.0
    boost2 = max(float(w[0]), 0.0) * 2.0
    while True:
        for i in other:
            s[i] = cds_item(rng, 0.35)
        p = 0.85 if tier in ("A", "B") else 0.3
        for i in afs_idx:
            s[i] = cds_item(rng, p, boost13 if i == 12 else
                            boost2 if i == 1 else 0.0)
        afs = s[afs_idx].sum()
        if (tier in ("A", "B") and afs >= 6) or (tier == "C" and afs < 6):
            return s


def make_session(rng, person, L, grid, start, *, consent=True,
                 images_fit=True, is_test=False, is_dev=False,
                 flat_words=False, no_word_block=False, no_state=False,
                 missing_grid=(), none_fog_both=False,
                 no_interview_date=False, interview_date="",
                 duration_min=None):
    """One complete session: (sessions row, trials, grid rows, truth)."""
    ts = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    sid = f"{person['participant_id']}_{ts.replace(':', '-')}"
    dur = float(rng.uniform(28, 55))
    end = start + pd.Timedelta(minutes=dur if duration_min is None
                               else float(duration_min))
    set_code = int(rng.integers(0, 2))
    window = GRID_WINDOWS[int(rng.integers(0, len(GRID_WINDOWS)))]
    trials = simulate_trials(rng, person, L, sid)
    grows, gtruth = simulate_grid(rng, person, sid, set_code, window, grid,
                                  missing_grid, none_fog_both)
    row = {
        "session_id": sid, "participant_id": person["participant_id"],
        "group": person["group"], "session_start_time": ts,
        "session_end_time": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "complete": True, "is_test": bool(is_test), "is_dev": bool(is_dev),
        "consent_given": bool(consent), "images_fit": bool(images_fit),
        "device_pixel_ratio": float(rng.choice([1.0, 1.25, 1.5, 2.0],
                                               p=[.35, .2, .15, .3])),
        "gamma_estimate": round(float(rng.normal(2.2, 0.15)), 3),
        "set_code": set_code,
        "language": "ja" if rng.random() < 0.8 else "en",
        "interview_date": "" if no_interview_date else interview_date,
    }
    row["has_word_block"] = not no_word_block
    words = simulate_words(rng, person, flat_words)
    for k, v in words.items():
        row[k] = None if no_word_block else v
    row["has_state_block"] = not no_state
    st = simulate_state(rng, person)
    for k, v in st.items():
        row[k] = None if no_state else v
    row["has_grid_episode"] = "episode" not in missing_grid
    row["has_grid_normal"] = "normal" not in missing_grid
    catch = trials[trials.trial_type == "catch"]
    row["n_catch_fail"] = int((~catch.catch_pass.astype(bool)).sum())
    row["flat_word_ratings"] = (not no_word_block) and len(
        set(words.values())) == 1
    truth = dict(gtruth)
    truth["flat_word_ratings"] = row["flat_word_ratings"]
    return row, trials, pd.DataFrame(grows), truth


SESSION_COLUMNS = (
    ["session_id", "participant_id", "group", "session_start_time",
     "session_end_time", "complete", "is_test", "is_dev", "consent_given",
     "images_fit", "device_pixel_ratio", "gamma_estimate", "set_code",
     "language", "interview_date", "n_catch_fail", "flat_word_ratings",
     "has_word_block", "has_grid_episode", "has_grid_normal",
     "has_state_block"]
    + [f"word_{w}" for w in WORD_IDS] + STATE_ITEMS)


# ============================================================
# Datasets
# ============================================================

def session_times(rng, n):
    """n session start times over six weeks, in random order."""
    base = pd.Timestamp("2026-11-02T00:00:00")
    mins = np.sort(rng.uniform(0, 42 * 24 * 60, n))
    t = [base + pd.Timedelta(minutes=float(m)) for m in mins]
    return [t[i] for i in rng.permutation(n)]


def dataset_specs():
    specs = []
    for ps, pdef in PARTICIPANT_SCENARIOS.items():
        for n in PARTICIPANT_NS:
            cdef = dict(taste_mult=pdef["taste_mult"], lapse=pdef["lapse"],
                        strategy=pdef["strategy"])
            specs.append({
                "name": f"P-{ps}_n{n:02d}__C-matched_n{BASELINE_CONTROLS_N}",
                "kind": "participant scenario", "p_scenario": ps,
                "p_def": dict(pdef), "n_p": n,
                "c_scenario": "matched (same taste and lapse, no strategy)",
                "c_def": cdef, "n_c": BASELINE_CONTROLS_N})
    ps, n = CONTROL_SCENARIO_PARTICIPANTS
    for cs, cd in CONTROL_SCENARIOS.items():
        for nc in CONTROL_NS:
            pdef = dict(PARTICIPANT_SCENARIOS[ps])
            pdef.update(cd["participant_overrides"])
            specs.append({
                "name": f"P-{ps}_n{n:02d}__C-{cs}_n{nc}",
                "kind": "control scenario", "p_scenario": ps,
                "p_def": pdef, "n_p": n, "c_scenario": cs,
                "c_def": dict(cd["control"]), "n_c": nc})
    return specs


def design_expectation(spec, rules):
    """What the design says each verdict should be, and how firm that is.

    `kind` says how firm the expectation is:
      fixed         the design fixes it outright (descriptive below
                    10 participants or 20 controls); a miss is a bug
      near_certain  the attention check fires when controls lapse at
                    0.2 and participants at 0
      no_claim      no effect: any verdict but "supported" (a C1 that
                    the attention check makes inconclusive by chance is
                    not a claim). A single draw, so it misses at the
                    false-claim rate, `p_miss`
      likely        an effect is present; reported, not checked
    """
    pdef, np_, nc = spec["p_def"], spec["n_p"], spec["n_c"]
    exp = {}
    for feat, claim, wkey in (("fog", "H1", "w_fog"),
                              ("colour", "H2", "w_colour")):
        has = pdef.get(wkey, 0) > 0 and pdef.get("share", 0) > 0
        if np_ < N_MIN_CLAIM:
            exp[claim] = {"verdict": "descriptive", "kind": "fixed",
                          "why": f"N = {np_} < {N_MIN_CLAIM}"}
        elif np_ > N_MAX_PARTICIPANTS:
            exp[claim] = {"verdict": "stop", "kind": "fixed"}
        elif has:
            r = rules[(feat, np_)]
            exp[claim] = {"verdict": "supported", "kind": "likely",
                          "why": f"effect present; registered planning "
                                 f"power {r['planning_power']:.3f}"}
        else:
            r = rules[(feat, np_)]
            null_rate = (r["false_guesser"] if pdef["taste_mult"] == 0
                         else r["false_worst"])
            exp[claim] = {"verdict": "no claim", "kind": "no_claim",
                          "p_miss": float(null_rate),
                          "why": f"no effect; registered false-claim rate "
                                 f"{null_rate:.4g}"}
    for feat, wkey in (("colour", "w_colour"), ("fog", "w_fog")):
        key = f"C1_{feat}"
        has = pdef.get(wkey, 0) > 0 and pdef.get("share", 0) > 0
        if nc < MIN_CONTROLS_C1:
            exp[key] = {"verdict": "descriptive", "kind": "fixed",
                        "why": f"{nc} controls < {MIN_CONTROLS_C1}"}
        elif spec["c_def"]["lapse"] > pdef["lapse"]:
            exp[key] = {"verdict": "inconclusive", "kind": "near_certain",
                        "why": "controls lapse more: the attention check "
                               "should fire"}
        elif has:
            exp[key] = {"verdict": "supported", "kind": "likely",
                        "why": "effect present (power depends on the "
                               "scenario)"}
        else:
            exp[key] = {"verdict": "no claim", "kind": "no_claim",
                        "p_miss": C1_NULL_RATE_GUIDE,
                        "why": "no effect: 'not supported' or, if the "
                               "attention check fires by chance, "
                               "'inconclusive'; never 'supported'"}
    return exp


def build_dataset(spec, L, th, grid):
    """A scenario dataset: people, sessions, survey, participant table."""
    rng = dataset_rng(spec["name"])
    ids = IdMaker(rng)
    ok_p, ex_p = build_population(rng, ids, "participant", spec["n_p"],
                                  spec["p_def"], th, L.n_an)
    ok_c, ex_c = build_population(rng, ids, "control", spec["n_c"],
                                  spec["c_def"], th, L.n_an)
    for p in ok_p + ex_p:
        p["tier"] = str(rng.choice(["A", "B", "C"]))
    people = ok_p + ex_p + ok_c + ex_c
    plan = [(p, {}) for p in people]
    return assemble(rng, plan, L, grid, people, spec)


def build_messy(L, th, grid):
    """Every exclusion and partial exclusion, once, with known counts."""
    name = "messy"
    rng = dataset_rng(name)
    ids = IdMaker(rng)
    scen = PARTICIPANT_SCENARIOS["both"]
    cscen = dict(taste_mult=scen["taste_mult"], lapse=scen["lapse"],
                 strategy=0.0)
    core, _ = build_population(rng, ids, "participant",
                               MESSY_CORE_PARTICIPANTS, scen, th, L.n_an,
                               record_recruits=False)
    for p in core:
        p["tier"] = str(rng.choice(["A", "B", "C"]))
    roles = [("two_sessions", {}), ("flat_words", {"flat_words": True}),
             ("no_word_block", {"no_word_block": True}),
             ("no_grid_normal", {"missing_grid": ("normal",)}),
             ("both_fog_none", {"none_fog_both": True}),
             ("no_state_block", {"no_state": True}),
             ("no_interview_date", {"no_interview_date": True}),
             ("two_catch_failures", {})]
    plan = []
    for i, p in enumerate(core):
        role, kw = roles[i] if i < len(roles) else ("plain", {})
        p["role"] = role
        if role == "two_catch_failures":
            p["catch"] = 2
        plan.append((p, kw))

    def extra(group, role, tier="", catch=None, scen_=scen, pid=None):
        while True:
            att = draw_one_attention(rng, scen_["lapse"], L.n_an)
            if att[1] < CATCH_FAIL_EXCLUDE:
                break
        p = make_person(rng, ids, group, scen_, th, False, att, tier=tier,
                        role=role, pid=pid)
        if catch is not None:
            p["catch"] = catch
        return p

    ex = [(extra("participant", "three_catch_failures", "A", catch=3), {}),
          (extra("participant", "images_fit_false", "B"),
           {"images_fit": False}),
          (extra("participant", "no_consent", "C"), {"consent": False}),
          (extra("participant", "tier_D", "D"), {})]
    ctrl, _ = build_population(rng, ids, "control", MESSY_CONTROLS, cscen,
                               th, L.n_an, record_recruits=False)
    ex.append((extra("control", "four_catch_failures", catch=4,
                     scen_=cscen), {}))
    null = dict(taste_mult=0.0, lapse=0.1, strategy=0.0)
    tests = [(extra("participant", "test_id", scen_=null, pid=t),
              {"is_test": True}) for t in TEST_IDS]
    tests.append((extra("participant", "dev_session", scen_=null),
                  {"is_dev": True}))
    plan = plan + ex + [(p, {}) for p in ctrl] + tests
    people = [p for p, _ in plan]
    spec = {"name": name, "kind": "messy", "p_scenario": "both",
            "p_def": dict(scen), "n_p": MESSY_CORE_PARTICIPANTS,
            "c_scenario": "matched", "c_def": cscen, "n_c": MESSY_CONTROLS}
    return assemble(rng, plan, L, grid, people, spec)


def build_messy_over_max(L, th, grid):
    """One more analysable participant and control than the maxima.

    Everyone passes every exclusion. In each group the first person is
    made to start 10 (15) minutes before the group's last starter and to
    take 120 minutes, so they complete last: by start time they would be
    kept, by completion time -- the registered rule -- they are set aside.
    """
    name = "messy_over_max"
    rng = dataset_rng(name)
    ids = IdMaker(rng)
    scen = PARTICIPANT_SCENARIOS["both"]
    cscen = dict(taste_mult=scen["taste_mult"], lapse=scen["lapse"],
                 strategy=0.0)
    n_p, n_c = MESSY_OVER_MAX
    part, _ = build_population(rng, ids, "participant", n_p, scen, th,
                               L.n_an, record_recruits=False)
    for p in part:
        p["tier"] = str(rng.choice(["A", "B", "C"]))
    ctrl, _ = build_population(rng, ids, "control", n_c, cscen, th, L.n_an,
                               record_recruits=False)
    plan = []
    for people_, mins in ((part, 10), (ctrl, 15)):
        for i, p in enumerate(people_):
            kw = {}
            if i == 0:
                p["role"] = "finishes_last"
                kw = {"start_before_latest_min": mins, "duration_min": 120}
            plan.append((p, kw))
    spec = {"name": name, "kind": "messy", "p_scenario": "both",
            "p_def": dict(scen), "n_p": N_MAX_PARTICIPANTS,
            "c_scenario": "matched", "c_def": cscen, "n_c": N_MAX_CONTROLS}
    return assemble(rng, plan, L, grid, part + ctrl, spec)


def assemble(rng, plan, L, grid, people, spec):
    """Sessions for every planned person, then the private files."""
    times = session_times(rng, len(plan))
    # `start_before_latest_min`: start that session this many minutes
    # before the latest OTHER start in the same group; with duration_min
    # the person starts before the group's last starter but finishes last.
    latest = {}
    for (p, kw), t in zip(plan, times):
        if "start_before_latest_min" not in kw:
            latest[p["group"]] = max(latest.get(p["group"], t), t)
    srows, trials, grids, truth_rows = [], [], [], []
    for (p, kw), t in zip(plan, times):
        if "start_before_latest_min" in kw:
            t = latest[p["group"]] - pd.Timedelta(
                minutes=kw["start_before_latest_min"])
            kw = {k: v for k, v in kw.items()
                  if k != "start_before_latest_min"}
        # One interview per participant, 3 to 60 days before the task.
        # Controls are not interviewed; test and dev sessions have none.
        idate = ""
        if p["group"] == "participant" and not (kw.get("is_test")
                                                or kw.get("is_dev")):
            idate = (t - pd.Timedelta(days=int(rng.integers(3, 61)))
                     ).strftime("%Y-%m-%d")
        p["interview_date"] = idate
        kw = dict(kw, interview_date=idate)
        row, tr, gr, gt = make_session(rng, p, L, grid, t, **kw)
        srows.append(row)
        trials.append(tr)
        grids.append(gr)
        truth_rows.append(person_truth(p, row, gt, kw))
        if p["role"] == "two_sessions":
            # A second complete session three days later: listed, never
            # used. Written FIRST, so file order cannot pick it.
            t2 = t + pd.Timedelta(days=3)
            r2, tr2, gr2, gt2 = make_session(rng, p, L, grid, t2,
                                             interview_date=idate)
            srows.insert(0, r2)
            trials.insert(0, tr2)
            grids.insert(0, gr2)
            pt = person_truth(p, r2, gt2, {})
            pt["expected_status"] = "repeat_not_used"
            truth_rows.insert(0, pt)
    mark_beyond_maximum(srows, truth_rows)
    sessions = pd.DataFrame(srows)[SESSION_COLUMNS]
    tidy = {"sessions": sessions,
            "trials": pd.concat(trials, ignore_index=True),
            "grid": pd.concat([g for g in grids if len(g)],
                              ignore_index=True)}
    survey, ptable = private_files(rng, people)
    return {"spec": spec, "tidy": tidy, "survey": survey, "ptable": ptable,
            "truth_people": pd.DataFrame(truth_rows), "people": people}


def mark_beyond_maximum(srows, truth_rows):
    """The registered maxima, applied to the intended statuses.

    Among each group's analysable sessions, the first N_MAX_PARTICIPANTS
    (N_MAX_CONTROLS) by completion time stay analysable; the rest become
    beyond_maximum. Ties in completion time go by session_id. Tier D does
    not count toward N and is not capped.
    """
    end = {r["session_id"]: r["session_end_time"] for r in srows}
    for grp, limit in (("participant", N_MAX_PARTICIPANTS),
                       ("control", N_MAX_CONTROLS)):
        cand = [t for t in truth_rows if t["group"] == grp
                and t["expected_status"] == "analysable"]
        cand.sort(key=lambda t: (end[t["session_id"]], t["session_id"]))
        for t in cand[limit:]:
            t["expected_status"] = "beyond_maximum"


def person_truth(p, row, gtruth, kw):
    if kw.get("is_test") or kw.get("is_dev"):
        status = "test_or_dev"
    elif not kw.get("consent", True):
        status = "excluded_no_consent"
    elif not kw.get("images_fit", True):
        status = "excluded_images_fit"
    elif p["catch"] >= CATCH_FAIL_EXCLUDE:
        status = "excluded_catch"
    elif p["tier"] == "D":
        status = "tier_D"
    else:
        status = "analysable"
    out = {"session_id": row["session_id"],
           "participant_id": p["participant_id"], "group": p["group"],
           "tier": p["tier"], "role": p["role"], "expected_status": status,
           "true_w_colour": float(p["w"][0]), "true_w_fog": float(p["w"][1]),
           "taste_colour": float(p["taste"][0]),
           "taste_fog": float(p["taste"][1]),
           "effect_colour": float(p["effect"][0]),
           "effect_fog": float(p["effect"][1]),
           "strategy": p["strategy"], "responder": p["responder"],
           "lapse": p["lapse"], "catch_fail": p["catch"],
           "fast_share": p["fast"]}
    out.update(gtruth)
    return out


def private_files(rng, people):
    """The fake survey (one row per respondent) and participant table."""
    rows, seen, prow = [], set(), []
    for p in people:
        pid = p["participant_id"]
        if pid in seen or p["role"] in ("test_id", "dev_session"):
            continue
        seen.add(pid)
        s = simulate_cds(rng, p["group"], p["tier"], p["w"])
        rows.append({"ResponseId": pid, **{f"Q{i + 1:02d}_score": int(s[i])
                                           for i in range(CDS_ITEMS)}})
        prow.append({"participant_id": pid, "group": p["group"],
                     "tier": p["tier"]})
    ids = IdMaker(rng)
    ids.used |= seen
    for _ in range(SURVEY_EXTRA_RESPONDENTS):
        grp = "control" if rng.random() < 0.5 else "participant"
        tier = "" if grp == "control" else str(rng.choice(list("ABCD")))
        s = simulate_cds(rng, grp, tier, np.zeros(2))
        rows.append({"ResponseId": ids(), **{f"Q{i + 1:02d}_score": int(s[i])
                                             for i in range(CDS_ITEMS)}})
    survey = pd.DataFrame(rows)
    survey = survey.iloc[rng.permutation(len(survey))].reset_index(drop=True)
    return survey, pd.DataFrame(prow)


# ============================================================
# The answers the analysis must find, computed from the tidy tables
# ============================================================

def realised_answers(ds, L, th, rules):
    """Exclusions, weights, verdicts: the answers in these exact data.

    Computed FROM THE TIDY TABLES as written, with the copied power-script
    functions (fit_firth_batch, classify, _c1_gate, _c1_test,
    mw_p_batch), so the analysis script, reading the same tables by its
    own route, must arrive at the same answers.

    Exclusions in the registered order: test or development sessions;
    the first complete session per ID (later ones listed); no consent;
    images not fitting; 3 or more catch failures. Then Tier D is set
    aside, analysed separately.
    """
    S = ds["tidy"]["sessions"]
    T = ds["tidy"]["trials"]
    tier = ds["ptable"].set_index("participant_id").tier.fillna("")
    status = pd.Series("", index=S.session_id.to_numpy(), dtype=object)
    test = (S.is_test | S.is_dev).to_numpy()
    status[S.session_id[test]] = "test_or_dev"
    rest = S[~test & S.complete].sort_values(["session_start_time",
                                              "session_id"])
    first = rest.drop_duplicates("participant_id", keep="first")
    status[rest.session_id[~rest.session_id.isin(first.session_id)]] = \
        "repeat_not_used"
    cf = (T[T.trial_type == "catch"].groupby("session_id").catch_pass
          .apply(lambda s: int((~s.astype(bool)).sum())))
    for _, r in first.iterrows():
        sid = r.session_id
        if not r.consent_given:
            status[sid] = "excluded_no_consent"
        elif not r.images_fit:
            status[sid] = "excluded_images_fit"
        elif cf[sid] >= CATCH_FAIL_EXCLUDE:
            status[sid] = "excluded_catch"
        elif tier.get(r.participant_id, "") == "D":
            status[sid] = "tier_D"
        else:
            status[sid] = "analysable"
    # The maxima: first 20 participants / 30 controls by completion time.
    for grp, limit in (("participant", N_MAX_PARTICIPANTS),
                       ("control", N_MAX_CONTROLS)):
        sel = S[(status[S.session_id].to_numpy() == "analysable")
                & (S.group == grp).to_numpy()].sort_values(
            ["session_end_time", "session_id"])
        status[sel.session_id.iloc[limit:]] = "beyond_maximum"

    keep = status[status.isin(["analysable", "tier_D"])].index
    an = T[T.analysed & T.session_id.isin(keep)]
    Y = an.pivot(index="session_id", columns="trial", values="chose_A")
    Y = Y.loc[:, L.trial_an].astype(float)
    est, last = fit_firth_batch(L.X_an, Y.to_numpy())
    pred, opp = classify(est, th)
    fast = an.assign(f=an.rt_ms < RT_FAST_CUTOFF_MS).groupby(
        "session_id").f.mean()
    W = pd.DataFrame({"session_id": Y.index, "w_colour": est[:, 0],
                      "w_fog": est[:, 1], "last_step": last,
                      "class_colour": np.where(pred[:, 0], "predicted",
                                               np.where(opp[:, 0], "opposite",
                                                        "not detected")),
                      "class_fog": np.where(pred[:, 1], "predicted",
                                            np.where(opp[:, 1], "opposite",
                                                     "not detected"))})
    W["status"] = status[W.session_id].to_numpy()
    W["group"] = S.set_index("session_id").group[W.session_id].to_numpy()
    W["catch_fail"] = cf[W.session_id].to_numpy()
    W["fast_share"] = fast[W.session_id].to_numpy()

    P = W[(W.status == "analysable") & (W.group == "participant")]
    C = W[(W.status == "analysable") & (W.group == "control")]
    n_p, n_c = len(P), len(C)
    out = {"n_participants": n_p, "n_controls": n_c,
           "n_tier_D": int((W.status == "tier_D").sum())}
    for feat, claim in (("fog", "H1"), ("colour", "H2")):
        a = int((P[f"class_{feat}"] == "predicted").sum())
        b = int((P[f"class_{feat}"] == "opposite").sum())
        d = {"feature": feat, "predicted": a, "opposite": b, "n": n_p}
        if n_p < N_MIN_CLAIM:
            d["verdict"] = "descriptive"
        else:
            r = rules[(feat, n_p)]
            k, m = int(r["k"]), int(r["m"])
            d.update({"k": k, "m": m,
                      "verdict": "supported"
                      if rule_mask_count(n_p, k, m)[a, b] else
                      "not supported"})
        out[claim] = d
    if n_p and n_c:
        fired, f_c, f_r = _c1_gate(P.catch_fail.to_numpy(float)[None],
                                   C.catch_fail.to_numpy(float)[None],
                                   P.fast_share.to_numpy(float)[None],
                                   C.fast_share.to_numpy(float)[None])
        p_catch, _ = mw_p_batch(C.catch_fail.to_numpy(float)[None],
                                P.catch_fail.to_numpy(float)[None],
                                exact_max_n=0)
        p_rt, _ = mw_p_batch(C.fast_share.to_numpy(float)[None],
                             P.fast_share.to_numpy(float)[None],
                             exact_max_n=0)
        Wp = P[["w_colour", "w_fog"]].to_numpy()[None]
        Wc = C[["w_colour", "w_fog"]].to_numpy()[None]
        rc, rf = _c1_test(Wp, Wc)
        out["gate"] = {"fired": bool(fired[0]), "fired_catch": bool(f_c[0]),
                       "fired_rt": bool(f_r[0]), "p_catch": float(p_catch[0]),
                       "p_rt": float(p_rt[0])}
        for feat, rej, k in (("colour", rc, 0), ("fog", rf, 1)):
            p, meth = mw_p_batch(Wp[:, :, k], Wc[:, :, k])
            if n_c < MIN_CONTROLS_C1:
                verdict = "descriptive"
            elif fired[0]:
                verdict = "inconclusive"
            else:
                verdict = "supported" if rej[0] else "not supported"
            out[f"C1_{feat}"] = {"p": float(p[0]), "method": meth,
                                 "holm_reject": bool(rej[0]),
                                 "verdict": verdict}
    steps = {}
    for st in ("test_or_dev", "repeat_not_used", "excluded_no_consent",
               "excluded_images_fit", "excluded_catch", "tier_D",
               "beyond_maximum", "analysable"):
        sel = S[status[S.session_id].to_numpy() == st]
        steps[st] = {g: int((sel.group == g).sum())
                     for g in ("participant", "control")}
    out["status_counts"] = steps
    return out, status, W


# ============================================================
# Structural checks on one dataset
# ============================================================

def check_dataset(ds, L, grid, checks):
    """Is the dataset well-formed, and does it encode its own truth?"""
    name = ds["spec"]["name"]
    S, T, G = ds["tidy"]["sessions"], ds["tidy"]["trials"], ds["tidy"]["grid"]
    tp = ds["truth_people"].set_index("session_id")
    prob = []

    per = T.groupby("session_id")
    if not (per.size() == L.n).all():
        prob.append("a session without 211 trial rows")
    ref = L.tl.set_index("trial")
    tt = ref.trial_type.reindex(T.trial).to_numpy()
    if not (T.trial_type.to_numpy() == tt).all():
        prob.append("trial types differ from the frozen list")
    if not (T.analysed.to_numpy() == ref.analysed.astype(bool)
            .reindex(T.trial).to_numpy()).all():
        prob.append("analysed flags differ from the frozen list")
    img = T.trial_type != "catch"
    if T.chose_A[img].isna().any() or T.chose_A[~img].notna().any():
        prob.append("chose_A missing on an image trial or set on a catch")
    a_left = ref.a_is_left.astype(bool).reindex(T.trial).to_numpy()
    side_ok = np.where(T.chose_A.fillna(-1).to_numpy() == 1,
                       np.where(a_left, "left", "right"),
                       np.where(a_left, "right", "left"))
    if not (T.side_pressed[img].to_numpy() == side_ok[img.to_numpy()]).all():
        prob.append("side_pressed disagrees with chose_A and a_is_left")
    c = T[~img]
    if not ((c.key_pressed == c.catch_key) == c.catch_pass.astype(bool)).all():
        prob.append("catch_pass disagrees with the keys")
    cf = c.groupby("session_id").catch_pass.apply(
        lambda s: int((~s.astype(bool)).sum()))
    if not (cf.reindex(tp.index) == tp.catch_fail).all():
        prob.append("catch failures differ from the truth")
    if not (S.set_index("session_id").n_catch_fail.reindex(cf.index)
            == cf).all():
        prob.append("n_catch_fail differs from the catch rows")
    an = T[T.analysed]
    fs = (an.rt_ms < RT_FAST_CUTOFF_MS).groupby(an.session_id).mean()
    if not np.allclose(fs.reindex(tp.index), tp.fast_share, atol=1e-12):
        prob.append("fast-response share differs from the truth")
    if T.rt_ms.isna().any():
        prob.append("missing RT")

    pid_ok = S[~S.is_test].participant_id.str.match(ID_PATTERN)
    if not pid_ok.all():
        prob.append("an ID outside the R_ + 15 pattern")
    if S.session_id.duplicated().any():
        prob.append("duplicate session_id")
    words = S[[f"word_{w}" for w in WORD_IDS]]
    has = S.has_word_block.to_numpy()
    if words[~has].notna().any().any() or words[has].isna().any().any():
        prob.append("word ratings present/absent against has_word_block")
    vals = words[has].to_numpy(float)
    if ((vals < 0) | (vals > WORD_MAX) | (vals != np.round(vals))).any():
        prob.append("a word rating outside 0-6")

    for sid, gs in G.groupby("session_id"):
        srow = S.set_index("session_id").loc[sid]
        ep = "A" if srow.set_code == 0 else "B"
        for block, want_set, present in (
                ("episode", ep, srow.has_grid_episode),
                ("normal", "B" if ep == "A" else "A", srow.has_grid_normal)):
            b = gs[gs.block == block]
            if present != (len(b) == 4) or (len(b) and
                                            set(b.scene_set) != {want_set}):
                prob.append(f"{sid}: {block} block rows or set")
            for _, r in b.iterrows():
                info = grid["screens"][(r.scene_set, int(r.screen))]
                if r.chose_none:
                    if block != "episode" or pd.notna(r.chosen_level_index):
                        prob.append(f"{sid}: 'none' where not allowed")
                else:
                    lv = int(r.chosen_level_index)
                    if lv not in info["levels"] or \
                            info["disp"][lv] != int(r.chosen_display_column):
                        prob.append(f"{sid}: level/column mismatch")
    for sid in S.session_id:
        srow = S.set_index("session_id").loc[sid]
        if not srow.has_grid_episode and not srow.has_grid_normal:
            continue
        if sid not in set(G.session_id):
            prob.append(f"{sid}: grid rows missing")

    sv = ds["survey"]
    item = [f"Q{i:02d}_score" for i in range(1, CDS_ITEMS + 1)]
    afs = sv[[f"Q{i:02d}_score" for i in AFS_ITEMS]].sum(axis=1)
    tot = sv[item].sum(axis=1)
    pt = ds["ptable"].set_index("participant_id")
    if sv.ResponseId.duplicated().any():
        prob.append("duplicate ResponseId")
    for pid, r in pt.iterrows():
        if pid not in set(sv.ResponseId):
            prob.append(f"{pid} not in the survey")
            continue
        i = sv.index[sv.ResponseId == pid][0]
        if r.group == "control" and not (afs[i] == 0 and tot[i] <= 20):
            prob.append(f"{pid}: control CDS outside AfS 0, total <= 20")
        if r.tier == "D" and not (tot[i] >= 70 and afs[i] <= 2):
            prob.append(f"{pid}: Tier D CDS outside total >= 70, AfS <= 2")
    sg = S[~(S.is_test | S.is_dev)].drop_duplicates("participant_id")
    for _, r in sg.iterrows():
        if r.participant_id not in pt.index:
            prob.append(f"{r.participant_id} not in the participant table")
        elif pt.loc[r.participant_id, "group"] != r.group:
            prob.append(f"{r.participant_id}: group differs")

    checks.require(f"[{name}] well-formed and consistent with its truth",
                   not prob, "; ".join(prob[:5]))


# ============================================================
# Writing
# ============================================================

def _col_letter(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def write_xlsx(path, df, sheet="Sheet1"):
    """A minimal one-sheet .xlsx with the standard library only.

    The real survey is an .xlsx, and the analysis environment promises
    numpy, pandas, scipy and matplotlib -- not openpyxl. So the fake is
    written by hand, and the analysis has a matching reader.
    """
    def cell(ref, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, (int, float, np.integer, np.floating)) and \
                not isinstance(v, bool):
            return f'<c r="{ref}"><v>{v}</v></c>'
        return (f'<c r="{ref}" t="inlineStr"><is><t>'
                f'{xml_escape(str(v))}</t></is></c>')
    rows = []
    cols = list(df.columns)
    rows.append('<row r="1">' + "".join(
        cell(f"{_col_letter(j)}1", c) for j, c in enumerate(cols)) + "</row>")
    for i, rec in enumerate(df.itertuples(index=False), start=2):
        rows.append(f'<row r="{i}">' + "".join(
            cell(f"{_col_letter(j)}{i}", v) for j, v in enumerate(rec))
            + "</row>")
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    files = {
        "[Content_Types].xml":
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
            'content-types"><Default Extension="rels" ContentType="'
            'application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="'
            'application/vnd.openxmlformats-officedocument.spreadsheetml.'
            'worksheet+xml"/></Types>',
        "_rels/.rels":
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/'
            '2006/relationships"><Relationship Id="rId1" Type="http://schemas.'
            'openxmlformats.org/officeDocument/2006/relationships/'
            'officeDocument" Target="xl/workbook.xml"/></Relationships>',
        "xl/workbook.xml":
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets><sheet name="'
            f'{sheet}" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels":
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/'
            '2006/relationships"><Relationship Id="rId1" Type="http://schemas.'
            'openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/></Relationships>',
        "xl/worksheets/sheet1.xml":
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<worksheet xmlns="{ns}"><sheetData>{"".join(rows)}'
            f'</sheetData></worksheet>',
    }
    # A fixed timestamp inside the zip, so a rerun writes identical bytes.
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in files.items():
            info = zipfile.ZipInfo(k, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, v)


def read_xlsx(path):
    """First sheet of an .xlsx as a DataFrame, first row as the header.

    Standard library only. Handles shared strings (what Excel and most
    exporters write), inline strings, numbers and booleans. The analysis
    script carries the same reader.
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


def write_dataset(ds, root, real, status, weights, expect, checks):
    d = os.path.join(root, "datasets", ds["spec"]["name"])
    for sub in ("tidy", "private"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    ds["tidy"]["trials"].to_csv(os.path.join(d, "tidy", TIDY_TRIALS),
                                index=False)
    ds["tidy"]["sessions"].to_csv(os.path.join(d, "tidy", TIDY_SESSIONS),
                                  index=False)
    ds["tidy"]["grid"].to_csv(os.path.join(d, "tidy", TIDY_GRID), index=False)
    xp = os.path.join(d, "private", SURVEY_FILE)
    write_xlsx(xp, ds["survey"])
    back = read_xlsx(xp)
    same = back.shape == ds["survey"].shape and \
        (back.astype(str).to_numpy() == ds["survey"].astype(str)
         .to_numpy()).all()
    try:
        import openpyxl  # noqa: F401  (only to cross-check the file)
        via = pd.read_excel(xp)
        same = same and (via.astype(str).to_numpy()
                         == ds["survey"].astype(str).to_numpy()).all()
    except ImportError:
        pass
    checks.require(f"[{ds['spec']['name']}] survey .xlsx reads back "
                   "unchanged", bool(same))
    ds["ptable"].to_csv(os.path.join(d, "private", PARTICIPANT_TABLE_FILE),
                        index=False)

    tp = ds["truth_people"].copy()
    tp["realised_status"] = status.reindex(tp.session_id).to_numpy()
    w = weights.set_index("session_id")
    for c in ("w_colour", "w_fog", "class_colour", "class_fog"):
        tp[f"fitted_{c}" if c.startswith("w_") else c] = \
            w[c].reindex(tp.session_id).to_numpy()
    tp.to_csv(os.path.join(d, "truth_people.csv"), index=False)

    an = tp[tp.realised_status == "analysable"]
    anp = an[an.group == "participant"]
    partial = {
        "flat_word_ratings": int(anp.flat_word_ratings.sum()),
        "no_word_block": int((~ds["tidy"]["sessions"].set_index("session_id")
                              .has_word_block.reindex(anp.session_id))
                             .sum()),
        "no_grid_shift_colour": int(anp.grid_shift_colour.isna().sum()),
        "no_grid_shift_fog": int(anp.grid_shift_fog.isna().sum()),
        "both_episode_fog_screens_none": int(anp.grid_both_fog_none.sum()),
        "no_state_block": int((~ds["tidy"]["sessions"].set_index(
            "session_id").has_state_block.reindex(anp.session_id)).sum()),
        "no_interview_date": int((ds["tidy"]["sessions"].set_index(
            "session_id").interview_date.reindex(anp.session_id)
            .fillna("") == "").sum()),
    }
    truth = {
        "dataset": ds["spec"]["name"], "kind": ds["spec"]["kind"],
        "seed": {"MASTER_SEED": MASTER_SEED,
                 "stream": f"SeedSequence([MASTER_SEED, crc32('"
                           f"{ds['spec']['name']}')])"},
        "participants": {"scenario": ds["spec"]["p_scenario"],
                         "population": ds["spec"]["p_def"],
                         "analysable_N": ds["spec"]["n_p"]},
        "controls": {"scenario": ds["spec"]["c_scenario"],
                     "population": ds["spec"]["c_def"],
                     "analysable_N": ds["spec"]["n_c"]},
        "exclusion_order": ["test_or_dev", "repeat_not_used",
                            "excluded_no_consent", "excluded_images_fit",
                            "excluded_catch", "tier_D (analysed separately)"],
        "expected_status_counts": {
            st: {g: int(((ds["truth_people"].expected_status == st)
                         & (ds["truth_people"].group == g)).sum())
                 for g in ("participant", "control")}
            for st in sorted(ds["truth_people"].expected_status.unique())},
        "partial_exclusions_among_analysable_participants": partial,
        "design_expectation": expect,
        "realised": real,
        "thresholds_used": {"colour": [None, None], "fog": [None, None]},
        "notes": [
            "realised = what these exact data give, computed from the tidy "
            "tables with the copied power-script functions "
            "(fit_firth_batch path). The analysis must reproduce it.",
            "design_expectation = what the design says; a 'firm' "
            "'not supported' can still fail by chance at the registered "
            "false-claim rate.",
            "C1 and its attention check are run on the analysable people, "
            "after the >= 3 catch-failure exclusion.",
        ],
    }
    return d, truth


# ============================================================
# The false-claim batches
# ============================================================

def make_false_claim_batches(root, X, L, th, rules, checks):
    """1,000 no-effect studies per (null, N), for the analysis's check.

    Lapse 0, as the registered rates were computed. Choices only: the
    false-claim check needs nothing else, and full tidy tables for
    6,000 studies would be gigabytes. A PREVIEW of the claim rate is
    computed here with fit_firth_batch; the registered check is run by
    the analysis script with its own per-person path.
    """
    print("\n" + "=" * 60)
    print("FALSE-CLAIM BATCHES")
    print("=" * 60)
    d = os.path.join(root, "false_claim")
    os.makedirs(d, exist_ok=True)
    rows = []
    for null, mult in FALSE_CLAIM_NULLS.items():
        for n in FALSE_CLAIM_NS:
            name = f"{null}_n{n:02d}"
            rng = dataset_rng(f"false_claim/{name}")
            S = FALSE_CLAIM_STUDIES
            W = rng.normal(0.0, 1.0, size=(S * n, 2)) * (th["upper"] * mult)
            Y = simulate_many(X, W, rng)
            np.savez_compressed(
                os.path.join(d, f"{name}.npz"),
                Y=Y.reshape(S, n, X.shape[0]).astype(np.uint8),
                W=W.reshape(S, n, 2), trial=L.trial_an,
                null=null, taste_mult=mult, n=n, studies=S)
            est, _ = fit_firth_batch(X, Y)
            pred, opp = classify(est, th)
            claims = {}
            for k, feat in enumerate(("colour", "fog")):
                a = pred[:, k].reshape(S, n).sum(axis=1)
                b = opp[:, k].reshape(S, n).sum(axis=1)
                r = rules[(feat, n)]
                claims[feat] = rule_mask_count(n, int(r["k"]),
                                               int(r["m"]))[a, b]
            for feat, claim in (("fog", "H1"), ("colour", "H2")):
                r = rules[(feat, n)]
                p = float(r["false_guesser"] if mult == 0
                          else r["false_worst"])
                cnt = int(claims[feat].sum())
                lo = int(stats.binom.ppf(0.025, S, p))
                hi = int(stats.binom.ppf(0.975, S, p))
                ok = cnt <= hi if mult == 0 else lo <= cnt <= hi
                rows.append({"file": f"{name}.npz", "null": null, "n": n,
                             "claim": claim, "feature": feat,
                             "rule": f"k >= {int(r['k'])}, m <= {int(r['m'])}",
                             "registered_rate": p,
                             "registered_rate_column": "false_guesser"
                             if mult == 0 else "false_worst",
                             "preview_claims": cnt,
                             "preview_rate": cnt / S,
                             "binomial_95_lo": lo if mult else 0,
                             "binomial_95_hi": hi,
                             "preview_within": ok})
            union = int((claims["colour"] | claims["fog"]).sum())
            rows.append({"file": f"{name}.npz", "null": null, "n": n,
                         "claim": "H1 or H2 (descriptive)", "feature": "",
                         "rule": "", "registered_rate": np.nan,
                         "registered_rate_column": "none registered",
                         "preview_claims": union,
                         "preview_rate": union / S,
                         "binomial_95_lo": np.nan, "binomial_95_hi": np.nan,
                         "preview_within": np.nan})
            print(f"  {name}: H1 {rows[-3]['preview_claims']}, "
                  f"H2 {rows[-2]['preview_claims']}, either {union} "
                  f"of {S}")
    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(d, "manifest.csv"), index=False)
    per = man[man.registered_rate.notna()]
    checks.note("false-claim batches: preview rates within the binomial "
                "95% range of the registered rates (H1, H2 separately)",
                bool(per.preview_within.astype(bool).all()),
                f"{int(per.preview_within.astype(bool).sum())}/{len(per)} "
                "cells; the registered check runs in the analysis script")
    return man


# ============================================================
# Report
# ============================================================

def write_report(root, t0, facts, th, rules, index, man, checks, inputs):
    L_ = []
    W = L_.append
    W("SIMULATED STUDY DATA  (v4)  --  simulate_study_data_v4.py")
    W("=" * 72)
    W(f"run          {time.strftime('%Y-%m-%d %H:%M:%S')}  "
      f"({time.time() - t0:.0f} s)")
    W(f"output       {root}")
    W(f"master seed  {MASTER_SEED}")
    W(f"python {platform.python_version()}, numpy {np.__version__}, "
      f"pandas {pd.__version__}")
    W("")
    W("INPUTS")
    W("-" * 72)
    for k, v in inputs.items():
        W(f"  {k:28s} {v}")
    W(f"  frozen package: version {facts.get('version')!r}, seed "
      f"{facts.get('seed')}, {facts.get('n_files')} files verified")
    W("")
    W("THRESHOLDS (recomputed with compute_floors on the frozen list)")
    W("-" * 72)
    for j, f in enumerate(("colour", "fog")):
        W(f"  {f:7s} upper {th['upper'][j]:+.5f}  lower {th['lower'][j]:+.5f}"
          f"   registered {REGISTERED_THRESHOLDS[f][0]:+.3f} / "
          f"{REGISTERED_THRESHOLDS[f][1]:+.3f}")
    W("")
    W("THE MODEL")
    W("-" * 72)
    W("  Each person: taste ~ Normal(0, taste_mult x upper threshold) on both")
    W("  features; with probability `strategy` +0.5/SD on both; responders")
    W("  (exactly round(share x N) analysable participants) add the effect.")
    W("  Personal lapse rate, catch failures and fast-response share from the")
    W("  power script's draw_attention; choices from simulate_lapsing at that")
    W("  lapse rate. A recruit with >= 3 catch failures is excluded and")
    W("  recruitment continues, so every dataset ends at its stated N.")
    W("  Word ratings, grid choices, CDS scores and session fields are")
    W("  well-formed and loosely tied to the weights; they model nothing.")
    W("  Placeholders until the experiment update is merged: the catch key")
    W("  sequence and the survey layout (one header row). State items are")
    W(f"  rated 0-{STATE_ITEM_MAX}, as specified in the experiment changes.")
    W("")
    W("DATASETS")
    W("-" * 72)
    W("  verdicts: realised (in these exact data) | design expectation;")
    W("  '!' marks an expectation the realised verdict misses. Under no")
    W("  effect each verdict is a single draw and misses at the false-claim")
    W("  rate; the registered H1/H2 rates are tested by the false-claim")
    W("  batches, not by these datasets.")
    W("")
    for r in index.itertuples():
        W(f"  {r.dataset}")
        W(f"    sessions {r.sessions}  analysable P {r.n_participants} / "
          f"C {r.n_controls}  excluded (catch) P {r.excl_catch_p} / "
          f"C {r.excl_catch_c}  Tier D {r.n_tier_D}")
        W(f"    H1 fog    {r.H1_counts:9s} {r.H1_realised:14s}| "
          f"{r.H1_expected}{' !' if r.H1_miss else ''}")
        W(f"    H2 colour {r.H2_counts:9s} {r.H2_realised:14s}| "
          f"{r.H2_expected}{' !' if r.H2_miss else ''}")
        W(f"    gate      {r.gate:23s}")
        W(f"    C1 colour {r.C1_colour_realised:24s}| {r.C1_colour_expected}"
          f"{' !' if r.C1_colour_miss else ''}")
        W(f"    C1 fog    {r.C1_fog_realised:24s}| {r.C1_fog_expected}"
          f"{' !' if r.C1_fog_miss else ''}")
    W("")
    W("FALSE-CLAIM BATCHES (preview; the registered check is the analysis's)")
    W("-" * 72)
    for r in man.itertuples():
        if r.registered_rate_column == "none registered":
            W(f"  {r.file:24s} {r.claim:24s} {r.preview_claims:4d} "
              f"({r.preview_rate:.3f})")
        else:
            W(f"  {r.file:24s} {r.claim:2s} {r.rule:15s} "
              f"{r.preview_claims:4d}/{FALSE_CLAIM_STUDIES} "
              f"vs {r.registered_rate_column} {r.registered_rate:.4g} "
              f"[{r.binomial_95_lo:.0f}..{r.binomial_95_hi:.0f}]  "
              f"{'within' if r.preview_within else 'OUTSIDE'}")
    W("")
    W("CHECKS")
    W("-" * 72)
    for r in checks.rows:
        W(f"  [{r['result']}] {r['check']}"
          + (f"  ({r['detail']})" if r["detail"] else ""))
    with open(os.path.join(root, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L_) + "\n")


def expectation_missed(e, realised):
    """An expectation (other than 'likely') the verdict contradicts."""
    if e["kind"] == "likely":
        return False
    if e["verdict"] == "no claim":
        return realised == "supported"
    return realised != e["verdict"]


def index_row(ds, real, expect, n_sessions):
    sc = real["status_counts"]
    row = {"dataset": ds["spec"]["name"], "kind": ds["spec"]["kind"],
           "sessions": n_sessions,
           "n_participants": real["n_participants"],
           "n_controls": real["n_controls"], "n_tier_D": real["n_tier_D"],
           "excl_catch_p": sc["excluded_catch"]["participant"],
           "excl_catch_c": sc["excluded_catch"]["control"]}
    for claim in ("H1", "H2"):
        r = real[claim]
        row[f"{claim}_counts"] = f"{r['predicted']}+/{r['opposite']}-"
        row[f"{claim}_realised"] = r["verdict"]
        e = expect[claim]
        row[f"{claim}_expected"] = e["verdict"] + (
            " (likely)" if e["kind"] == "likely" else "")
        row[f"{claim}_kind"] = e["kind"]
        row[f"{claim}_p_miss"] = e.get("p_miss", np.nan)
        row[f"{claim}_miss"] = expectation_missed(e, r["verdict"])
    g = real.get("gate")
    row["gate"] = ("fired" if g and g["fired"] else "did not fire") + (
        f" (p catch {g['p_catch']:.3g}, rt {g['p_rt']:.3g})" if g else "")
    for feat in ("colour", "fog"):
        r = real.get(f"C1_{feat}", {"verdict": "n/a", "p": np.nan})
        row[f"C1_{feat}_realised"] = f"{r['verdict']} (p {r['p']:.3g})"
        row[f"C1_{feat}_verdict"] = r["verdict"]
        e = expect[f"C1_{feat}"]
        row[f"C1_{feat}_expected"] = e["verdict"] + (
            " (likely)" if e["kind"] == "likely" else "")
        row[f"C1_{feat}_kind"] = e["kind"]
        row[f"C1_{feat}_p_miss"] = e.get("p_miss", np.nan)
        row[f"C1_{feat}_miss"] = expectation_missed(e, r["verdict"])
    return row


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    root = resolve_output_root()
    os.makedirs(root)
    print(f"Writing to {root}")
    checks = Checks()
    meta = {"script": os.path.basename(__file__),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output": root, "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": sys.modules["scipy"].__version__,
            "settings": settings_snapshot()}
    try:
        try:
            facts = verify_frozen(FROZEN_ROOT, [])
        except (RuntimeError, FileNotFoundError) as e:
            checks.require("frozen package matches checksums.sha256", False,
                           str(e).splitlines()[0])
        checks.require("frozen package matches checksums.sha256", True,
                       f"{facts.get('n_files')} files, seed "
                       f"{facts.get('seed')}")
        path = os.path.join(FROZEN_ROOT, FROZEN_TRIAL_LIST)
        trials_an, X, _r = load_trials(path, "frozen")
        L = load_full_list(path, trials_an, X, checks)
        inputs = {"trial_list.csv": sha256_of(path),
                  "checksums.sha256": sha256_of(os.path.join(
                      FROZEN_ROOT, "checksums.sha256")),
                  "combined_rule.csv": sha256_of(COMBINED_RULE),
                  "grid_trials.csv": sha256_of(GRID_TRIALS),
                  "this script": sha256_of(os.path.abspath(__file__))}
        if os.path.isfile(POWER_SCRIPT):
            inputs[POWER_SCRIPT] = sha256_of(POWER_SCRIPT)
        meta["inputs_sha256"] = inputs

        th = run_identity_check(X, facts, checks, root)
        got = {f: (round(float(th["upper"][j]), 3),
                   round(float(th["lower"][j]), 3))
               for j, f in enumerate(("colour", "fog"))}
        checks.require("recomputed thresholds equal the registered ones "
                       "(3 dp)", got == REGISTERED_THRESHOLDS,
                       f"colour {got['colour']}, fog {got['fog']}")
        meta["thresholds"] = {"upper": th["upper"], "lower": th["lower"]}
        rules = load_rules(checks)
        grid = load_grid(checks)

        print("\n" + "=" * 60)
        print("DATASETS")
        print("=" * 60)
        built = [build_dataset(s, L, th, grid) for s in dataset_specs()]
        built.append(build_messy(L, th, grid))
        built.append(build_messy_over_max(L, th, grid))
        index = []
        for ds in built:
            name = ds["spec"]["name"]
            print(f"\n  {name}")
            check_dataset(ds, L, grid, checks)
            real, status, weights = realised_answers(ds, L, th, rules)
            exp_ = design_expectation(ds["spec"], rules)
            st_ok = (status.reindex(ds["truth_people"].session_id)
                     .to_numpy()
                     == ds["truth_people"].expected_status.to_numpy()).all()
            checks.require(f"[{name}] every session's realised status is "
                           "its intended one", bool(st_ok))
            nn = (real["n_participants"], real["n_controls"])
            checks.require(f"[{name}] analysable N as specified",
                           nn == (ds["spec"]["n_p"], ds["spec"]["n_c"]),
                           f"P {nn[0]}, C {nn[1]}")
            d, truth = write_dataset(ds, root, real, status, weights, exp_,
                                     checks)
            truth["thresholds_used"] = {
                "colour": [float(th["upper"][0]), float(th["lower"][0])],
                "fog": [float(th["upper"][1]), float(th["lower"][1])]}
            write_json(os.path.join(d, "truth.json"), truth)
            index.append(index_row(ds, real, exp_,
                                   len(ds["tidy"]["sessions"])))
        index = pd.DataFrame(index)
        index.to_csv(os.path.join(root, "datasets.csv"), index=False)

        print("\n" + "=" * 60)
        print("EXPECTED VERDICTS IN THE CLEAR SCENARIOS")
        print("=" * 60)
        cols = ("H1", "H2", "C1_colour", "C1_fog")

        def misses(kind):
            out, n, pr = [], 0, []
            for col in cols:
                sel = index[index[f"{col}_kind"] == kind]
                n += len(sel)
                pr += sel[f"{col}_p_miss"].fillna(0).tolist()
                out += [f"{d} {col}" for d in sel.dataset[sel[f"{col}_miss"]]]
            return out, n, pr

        m, n, _ = misses("fixed")
        checks.require("verdicts the design fixes (descriptive below 10 "
                       "participants or 20 controls)", not m,
                       f"{n} verdicts" if not m else "; ".join(m))
        m, n, _ = misses("near_certain")
        checks.note("C1 inconclusive when controls lapse more", not m,
                    f"{n} verdicts" if not m else "; ".join(m))
        for label, sub in (("H1/H2", cols[:2]), ("C1", cols[2:])):
            mm, nn, pr = [], 0, []
            for col in sub:
                sel = index[index[f"{col}_kind"] == "no_claim"]
                nn += len(sel)
                pr += sel[f"{col}_p_miss"].tolist()
                mm += [f"{d} {col}" for d in sel.dataset[sel[f"{col}_miss"]]]
            p_any = 1.0 - float(np.prod(1.0 - np.asarray(pr, float)))
            checks.note(f"no claim under no effect, {label} (single draws)",
                        not mm, f"{nn - len(mm)}/{nn} no claim; a miss "
                        f"somewhere among these is expected "
                        f"{p_any:.0%} of the time"
                        + (f"; chance claim in: {', '.join(mm)}"
                           if mm else ""))
        matched = index[index.dataset.str.contains("C-matched")
                        | index.dataset.str.contains("C-no_strategy")
                        | index.dataset.str.contains("C-quarter")]
        fired = int(matched.gate.str.startswith("fired").sum())
        checks.info("attention check firing with no lapse difference",
                    f"{fired} of {len(matched)} datasets (the registered "
                    "gate's matched false-alarm rate is about 0.04); a "
                    "firing there makes C1 inconclusive, not a claim")
        messy = [b for b in built if b["spec"]["name"] == "messy"][0]
        mexp = {"test_or_dev": {"participant": 3, "control": 0},
                "repeat_not_used": {"participant": 1, "control": 0},
                "excluded_no_consent": {"participant": 1, "control": 0},
                "excluded_images_fit": {"participant": 1, "control": 0},
                "excluded_catch": {"participant": 1, "control": 1},
                "tier_D": {"participant": 1, "control": 0},
                "analysable": {"participant": MESSY_CORE_PARTICIPANTS,
                               "control": MESSY_CONTROLS}}
        mexp["beyond_maximum"] = {"participant": 0, "control": 0}
        mreal, _, _ = realised_answers(messy, L, th, rules)
        checks.require("messy dataset: every exclusion count as designed",
                       mreal["status_counts"] == mexp,
                       json.dumps(mreal["status_counts"]))
        over = [b for b in built if b["spec"]["name"] == "messy_over_max"][0]
        oreal, ostat, _ = realised_answers(over, L, th, rules)
        oexp = {st: {"participant": 0, "control": 0} for st in mexp}
        oexp["analysable"] = {"participant": N_MAX_PARTICIPANTS,
                              "control": N_MAX_CONTROLS}
        oexp["beyond_maximum"] = {"participant": 1, "control": 1}
        late = over["truth_people"]
        late = set(late.session_id[late.role == "finishes_last"])
        cut = set(ostat[ostat == "beyond_maximum"].index)
        os_ = over["tidy"]["sessions"].set_index("session_id")
        for sid in late:
            g = os_.loc[sid, "group"]
            others = os_[(os_.group == g) & (os_.index != sid)]
            checks.require("over-maximum dataset: the last to complete "
                           f"({g}) started before the group's last starter",
                           os_.loc[sid, "session_start_time"]
                           < others.session_start_time.max())
        checks.require("over-maximum dataset: the first 20 and 30 by "
                       "completion time kept, the last to complete in each "
                       "group set aside", oreal["status_counts"] == oexp
                       and cut == late and len(cut) == 2,
                       json.dumps(oreal["status_counts"]))

        man = make_false_claim_batches(root, X, L, th, rules, checks)
        meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta["elapsed_s"] = round(time.time() - t0, 1)
        meta["datasets"] = index.dataset.tolist()
        meta["checks"] = checks.rows
        meta["status"] = "complete"
        write_json(os.path.join(root, "run_meta.json"), meta)
        write_report(root, t0, facts, th, rules, index, man, checks, inputs)
        print(f"\nDone in {time.time() - t0:.0f} s. Report: "
              f"{os.path.join(root, 'report.txt')}")
    except CheckFailed as e:
        meta["status"] = f"STOPPED: {e}"
        meta["checks"] = checks.rows
        write_json(os.path.join(root, "run_meta.json"), meta)
        print(f"\nSTOPPED. A check failed: {e}\nNothing in {root} should be "
              "used. See run_meta.json.")
        sys.exit(1)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
