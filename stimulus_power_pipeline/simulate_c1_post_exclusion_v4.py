"""
simulate_c1_post_exclusion_v4.py

C1 as the registered analysis will actually run it: the catch-trial
exclusion first, then the attention check, then the group test.

Why this script exists
----------------------
The registered C1 figures (c1_compact.csv, c1_cells.csv) come from
run_c1() in simulate_power_v4_n10_20.py. That function draws participants
and controls and runs the attention check (_c1_gate) and the group test
(_c1_test) on everyone it drew. The real analysis (analyse_study_v4.py)
first EXCLUDES anyone who failed 3 or more of the 10 catch trials and
keeps recruiting until each group reaches its target number of analysable
people; only then does it run the attention check and the test. So the
registered C1 numbers do not describe the procedure that will be run
(issue tracker item C05). This script simulates the whole procedure.

Nothing registered is changed: not a threshold, not a test, not a rule.
The script measures how the registered procedure behaves.

What it does, in order
----------------------
  0  CHECKS, each of which stops the run if it fails:
       - the registration files match the committed copies (SHA-256);
       - the power-run files it reads match the registration's record;
       - the frozen trial list matches its own checksums (verify_frozen);
       - every copied function is verbatim (syntax tree) and gives
         results identical to the original's, run now on fixed inputs;
       - every setting the copies read equals the original's;
       - the computed thresholds round to the registered ones.
  1  REPLAY. The copied run_c1 is run with the power script's own seed
     and must give back c1_power.csv: the attention columns exactly, the
     weight-dependent columns to within REPLAY_STUDY_TOL studies per cell
     (the only allowance, for floating-point differences in the Firth fit
     between machines). The person pools it builds are kept.
  2  VALIDATION. The new procedure with the exclusion SWITCHED OFF, on
     the replayed (registered) pools, with new study draws. It must
     reproduce c1_cells.csv within Monte Carlo error, by criteria stated
     in the report before the results.
  3  THE REGISTERED PROCEDURE, at the registered (rounded) thresholds, on
     new pools: per study, draw candidates in order, exclude anyone with
     3 or more catch failures, stop at the target, record how many were
     drawn and excluded; then the attention check, then C1 per feature
     with Holm, "inconclusive" if the check fired, "descriptive" below
     20 controls; and the zero-catch-failure test beside it. The same
     cells are also run with the exclusion off ON THE SAME POOLS, so the
     effect of the exclusion is read from a paired comparison.

Which thresholds where
----------------------
The registered rates were computed with the thresholds the power script
computes (unrounded). The analysis applies the rounded registered ones.
In C1 a threshold enters only through the taste spread (0.5 x the upper
threshold). Steps 1 and 2 reproduce the registered numbers, so they use
the computed thresholds; step 3 describes the procedure that will be
run, so it uses the registered ones. The report states the difference.

Inputs (relative to the project folder)
---------------------------------------
  simulate_power_v4_n10_20.py                 the original, at top level
  data/frozen_trial_list_v4/                  the sealed trial list
  data/registration_v4_2/                     combined_rule.csv,
                                              c1_cells.csv, c1_compact.csv
  data/power_v4_n10_20/frozen/c1_power.csv    for the replay

Outputs, under data/simulate_c1_post_exclusion_v4/ (never overwritten;
a rerun goes to _2, _3, ...)
---------------------------------------------------------------------
  report.txt                      the human-readable report
  run_meta.json                   settings, input checksums, check results
  tables/identity_check.csv       the copied functions against the original
  tables/replay_comparison.csv    run_c1 replayed against c1_power.csv
  tables/validation_cells.csv     step 2, per cell and measure, with z
  tables/c1_post_exclusion.csv    step 3, every cell, exclusion on
  tables/c1_exclusion_off.csv     step 3, every cell, exclusion off
  tables/c1_side_by_side.csv      registered / off / on, per cell
  tables/c1_compact_side_by_side.csv   the registered compact scenario
  tables/exclusion_counts.csv     how many are drawn and excluded
  tables/gate_summary.csv         how often the check fires
  tables/false_positive_summary.csv    no-effect rates by lapse pair
  figures/*.png
  logs/replay_stdout.txt          what run_c1 printed during the replay

Usage
-----
    python simulate_c1_post_exclusion_v4.py

No arguments. Every setting is in SETTINGS below. Python 3 with numpy,
pandas, scipy and matplotlib.
"""

import ast
import contextlib
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

# --- Inputs ---
FROZEN_ROOT = "data/frozen_trial_list_v4"
FROZEN_TRIAL_LIST = "trial_list.csv"
REGISTRATION_DIR = "data/registration_v4"
POWER_FROZEN_DIR = "data/power_v4_n10_20/frozen"
POWER_SCRIPT = "simulate_power_v4_n10_20.py"

# COMMITTED_SHA256
#   The registration files as committed in the repository
#   (data_power_v4/registration_v4/). The local copies must match
#   exactly, or the run stops.
COMMITTED_SHA256 = {
    "combined_rule.csv":
        "9e9e9a2a2b74c709792a0bc88767b20a9a0ed2f021fbf79fe419eb9382377a0a",
    "c1_cells.csv":
        "54f26c6bbe9ce3a29319b46520bbbb1e8c3c3895fdbc070fd67e6fca35ef1cb0",
    "c1_compact.csv":
        "5b812e39eaa7beb77aa6611dbb2e470631ee59162b8b8c05342e40c35c398598",
}

# POWER_RUN_SHA256
#   The power-run file the replay is compared with, as recorded in the
#   registration's run_meta.json (and identical to the committed copy).
POWER_RUN_SHA256 = {
    "c1_power.csv":
        "f7033f3160e9fe4de6f23049ff8f3d4fcae20caa030299112cce11767950e807",
}

# POWER_SCRIPT_SHA256
#   The committed simulate_power_v4_n10_20.py. The copies below are
#   checked against the file on disk; this makes sure that file is the
#   committed one.
POWER_SCRIPT_SHA256 = (
    "852f3b33eb349153fbff9f95c50e1cd3c4f84f68b204d79a23f020f471bd1730")

# --- Output ---
OUTPUT_ROOT = "data/simulate_c1_post_exclusion_v4"
MAKE_FIGURES = True
FIG_DPI = 150

# --- What is registered (stated so the run can check it; not changed) ---
REGISTERED_THRESHOLDS = {"colour": (0.286, -0.290), "fog": (0.488, -0.486)}
CATCH_FAIL_EXCLUDE = 3     # 3 or more of the 10 catch trials -> excluded
MIN_CONTROLS_C1 = 20       # below this, C1 is descriptive
N_MAX_PARTICIPANTS = 20    # registered maxima, for the report only
N_MAX_CONTROLS = 30

# --- The cells of step 3 (and of the validation) ---
#
# The cells of c1_cells.csv at 20 and 30 controls: 13 effect / feature /
# responder-share combinations x 3 participant sizes x 2 control sizes x
# 3 strategy rates x 7 lapse pairs = 1,638 cells. 40 controls is beyond
# the registered maximum of 30 and is not run.
PARTICIPANT_N = [10, 15, 20]
CONTROL_N = [20, 30]

# STUDIES_PER_CELL
#   Simulated studies per cell in steps 2 and 3. 4,000 (the replay keeps
#   the registered 2,000, which it must). At 4,000 the Monte Carlo SE of
#   a rate of 0.05 is 0.0034.
STUDIES_PER_CELL = 4000

# --- The replay and the validation ---
RUN_REPLAY = True

# REPLAY_STUDY_TOL
#   How many studies per cell a weight-dependent rate may differ by in
#   the replay. The attention columns do not depend on the weights and
#   must match exactly. A Firth weight can differ in the last bits
#   between machines, which can swap two ranks in a rare study; nothing
#   else can make the replay differ.
REPLAY_STUDY_TOL = 2

# VALIDATION CRITERIA (step 2), fixed before the run and printed in the
# report before the results. For every cell and measure,
#   z = (new - registered) / sqrt(p(1-p) (1/n_registered + 1/n_new)),
# p pooled. The pools are the registered ones (step 1), so this is
# Monte Carlo error only. Measures conditional on the zero-catch
# subsample being usable use the usable studies as n, and are skipped
# when either n is below VAL_MIN_N.
VAL_Z = 1.96
VAL_MAX_SHARE_BEYOND = 0.08    # at most 8% of |z| beyond 1.96
VAL_MAX_ABS_Z = 4.5            # none beyond 4.5
VAL_MEAN_Z = 0.10              # mean z within +/- 0.10
VAL_MIN_N = 50
VAL_MEASURES = [
    ("power_colour_raw", None), ("power_fog_raw", None),
    ("power_either_raw", None), ("gate_fires", None),
    ("gate_fires_catch", None), ("gate_fires_rt", None),
    ("power_colour_gated", None), ("power_fog_gated", None),
    ("power_either_gated", None), ("zero_usable", None),
    ("zero_power_either", "zero_usable"),
    ("zero_rt_fires", "zero_usable"),
    ("zero_power_rt_clean", "zero_usable"),
]

# --- Seeds ---
#   The replay uses the power script's own seed (RANDOM_SEED + 77,
#   inside run_c1). The others are this script's own.
SEED_VALIDATION_DRAWS = 20261101
SEED_MAIN_POOLS = 20261102
SEED_MAIN_DRAWS = 20261103

# --- Identity check inputs ---
IDENTITY_SEED = 20261104

# --- Figures: the power script's Okabe-Ito order, markers and lines ---
SERIES_COLOURS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
SERIES_MARKERS = ["o", "s", "^", "D"]
SERIES_STYLES = ["-", "--", "-.", ":"]


# ============================================================
# COPIED SETTINGS
#
# Values identical to simulate_power_v4_n10_20.py. The copied functions
# read them as globals; the start-up check compares every one with the
# power script's own value and stops if any differs.
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
NONRESPONDER_SPREAD_MULT = 0.5
RULE_RESPONDER_WEIGHTS = [0.30, 0.50, 0.70]
ALPHA = 0.05
FLOOR_ALPHA = 0.025
N_FLOOR_PEOPLE = 20_000
DETECTION_SIDED = "two"
C1_STRATEGY_RATES = [0.0, 0.25, 0.5]
C1_STRATEGY_WEIGHT = 0.5
C1_PARTICIPANT_N = [10, 15, 20]
C1_CONTROL_RATIOS = [1.0, 1.5, 2.0]
C1_CONTROL_ABSOLUTE = [20, 30, 40]
C1_LAPSE_PAIRS = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0),
                  (0.0, 0.1), (0.0, 0.2), (0.1, 0.1), (0.2, 0.2)]
C1_EXACT_MAX_N = 20
C1_POOL_PEOPLE = 20_000
C1_STUDIES = 2000
C1_HOLM = True
C1_ZERO_MIN_N = 4
C1_RESPONDER_SHARES = [0.5, 1.0]
GATE_ALPHA = 0.05
N_CATCH_TRIALS = 10
CATCH_P_FAIL_PER_LAPSE = 0.75
LAPSE_BETWEEN_SD = 0.5
RT_GATE = True
RT_FAST_CUTOFF_MS = 300
RT_FAST_GIVEN_LAPSE = 0.4
RT_FAST_GIVEN_ENGAGED = 0.005
GATE_ACTION = "inconclusive"
GATE_SECONDARY_ZERO_CATCH = True
RANDOM_SEED = 42

COPIED_SETTINGS = [
    "UNHASHED", "FROZEN_SEED", "PRED_SAT", "PRED_FOG", "SD_SAT", "SD_FOG",
    "PRED_REBUILD_TOL", "ESTIMATOR", "FIRTH_ITER", "FIRTH_TOL",
    "FIRTH_MAX_STEP", "CAP_INFORMATIVE_SD", "NONRESPONDER_SPREAD_MULT",
    "RULE_RESPONDER_WEIGHTS", "ALPHA", "FLOOR_ALPHA", "N_FLOOR_PEOPLE",
    "DETECTION_SIDED", "C1_STRATEGY_RATES", "C1_STRATEGY_WEIGHT",
    "C1_PARTICIPANT_N", "C1_CONTROL_RATIOS", "C1_CONTROL_ABSOLUTE",
    "C1_LAPSE_PAIRS", "C1_EXACT_MAX_N", "C1_POOL_PEOPLE", "C1_STUDIES",
    "C1_HOLM", "C1_ZERO_MIN_N", "C1_RESPONDER_SHARES", "GATE_ALPHA",
    "N_CATCH_TRIALS", "CATCH_P_FAIL_PER_LAPSE", "LAPSE_BETWEEN_SD",
    "RT_GATE", "RT_FAST_CUTOFF_MS", "RT_FAST_GIVEN_LAPSE",
    "RT_FAST_GIVEN_ENGAGED", "GATE_ACTION", "GATE_SECONDARY_ZERO_CATCH",
    "RANDOM_SEED",
]


# ============================================================
# COPIED FUNCTIONS
#
# VERBATIM from simulate_power_v4_n10_20.py, each marked with its
# source lines, in the original's order. Do not edit them here: the
# start-up check compares each one with the original (syntax tree
# and results), and a copy that differs stops the run.
# ============================================================

# --- COPY of simulate_power_v4_n10_20.py L1928-L1954 (fit_logit_batch), verbatim ---
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
# --- end of copy (fit_logit_batch) ---

# --- COPY of simulate_power_v4_n10_20.py L1992-L2037 (fit_firth_batch), verbatim ---
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
# --- end of copy (fit_firth_batch) ---

# --- COPY of simulate_power_v4_n10_20.py L2238-L2245 (fit_batch), verbatim ---
def fit_batch(X, Y):
    """Many participants, under the registered estimator."""
    if ESTIMATOR == "firth":
        return fit_firth_batch(X, Y)[0]
    if ESTIMATOR == "mle":
        return fit_logit_batch(X, Y)
    raise ValueError(
        f"ESTIMATOR must be 'firth' or 'mle', got {ESTIMATOR!r}")
# --- end of copy (fit_batch) ---

# --- COPY of simulate_power_v4_n10_20.py L2248-L2251 (simulate_many), verbatim ---
def simulate_many(X, W, rng):
    """Choices of many people at once. W is (n_people, 2) true weights."""
    p = 1.0 / (1.0 + np.exp(-(W @ X.T)))
    return (rng.random(p.shape) < p).astype(float)
# --- end of copy (simulate_many) ---

# --- COPY of simulate_power_v4_n10_20.py L2432-L2441 (sha256_of), verbatim ---
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
# --- end of copy (sha256_of) ---

# --- COPY of simulate_power_v4_n10_20.py L2444-L2589 (verify_frozen), verbatim ---
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
# --- end of copy (verify_frozen) ---

# --- COPY of simulate_power_v4_n10_20.py L2592-L2656 (load_trials), verbatim ---
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
# --- end of copy (load_trials) ---

# --- COPY of simulate_power_v4_n10_20.py L2663-L2685 (informative_medians), verbatim ---
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
# --- end of copy (informative_medians) ---

# --- COPY of simulate_power_v4_n10_20.py L2688-L2695 (as_probability), verbatim ---
def as_probability(weight, difference):
    """A weight, as the probability of choosing the predicted image on
    a trial of that difference.

    The readable form. `weight` is per corpus SD and `difference` is
    in SD, so their product is the log-odds.
    """
    return float(1.0 / (1.0 + np.exp(-weight * difference)))
# --- end of copy (as_probability) ---

# --- COPY of simulate_power_v4_n10_20.py L2698-L2784 (compute_floors), verbatim ---
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
# --- end of copy (compute_floors) ---

# --- COPY of simulate_power_v4_n10_20.py L2787-L2804 (classify), verbatim ---
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
# --- end of copy (classify) ---

# --- COPY of simulate_power_v4_n10_20.py L5389 (the cache mw_exact_sf uses) ---
_MW_SF_CACHE = {}

# --- COPY of simulate_power_v4_n10_20.py L5392-L5425 (mw_exact_sf), verbatim ---
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
# --- end of copy (mw_exact_sf) ---

# --- COPY of simulate_power_v4_n10_20.py L5428-L5501 (mw_p_batch), verbatim ---
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
# --- end of copy (mw_p_batch) ---

# --- COPY of simulate_power_v4_n10_20.py L5526-L5543 (holm_two), verbatim ---
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
# --- end of copy (holm_two) ---

# --- COPY of simulate_power_v4_n10_20.py L5546-L5608 (_c1_pool), verbatim ---
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
# --- end of copy (_c1_pool) ---

# --- COPY of simulate_power_v4_n10_20.py L5611-L5647 (_c1_gate), verbatim ---
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
# --- end of copy (_c1_gate) ---

# --- COPY of simulate_power_v4_n10_20.py L5650-L5668 (_c1_draw), verbatim ---
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
# --- end of copy (_c1_draw) ---

# --- COPY of simulate_power_v4_n10_20.py L5671-L5687 (_c1_test), verbatim ---
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
# --- end of copy (_c1_test) ---

# --- COPY of simulate_power_v4_n10_20.py L5690-L5776 (_c1_zero_catch), verbatim ---
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
# --- end of copy (_c1_zero_catch) ---

# --- COPY of simulate_power_v4_n10_20.py L5779-L5804 (draw_attention), verbatim ---
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
# --- end of copy (draw_attention) ---

# --- COPY of simulate_power_v4_n10_20.py L5933-L6379 (run_c1), verbatim ---
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
# --- end of copy (run_c1) ---


# ============================================================
# THIS SCRIPT'S OWN CODE (not copied)
# ============================================================

COPIED_FUNCTIONS = [
    "fit_logit_batch", "fit_firth_batch", "fit_batch", "simulate_many",
    "sha256_of", "verify_frozen", "load_trials", "informative_medians",
    "as_probability", "compute_floors", "classify", "mw_exact_sf",
    "mw_p_batch", "holm_two", "_c1_pool", "_c1_gate", "_c1_draw",
    "_c1_test", "_c1_zero_catch", "draw_attention", "run_c1",
]

KEYS = ["effect_on", "effect_sd", "responder_share", "n_participants",
        "n_controls", "strategy_rate", "lapse_participants",
        "lapse_controls"]

# Columns of run_c1's table that do not depend on any fitted weight: the
# replay must reproduce these exactly.
ATTENTION_COLS = ["gate_fires", "gate_fires_catch", "gate_fires_rt",
                  "inconclusive", "zero_usable", "zero_n_participants",
                  "zero_n_controls", "zero_lapse_participants",
                  "zero_lapse_controls", "zero_rt_fires"]
# Columns that do, with the denominator their rate is taken over.
WEIGHT_COLS = [("power_colour_raw", None), ("power_fog_raw", None),
               ("power_either_raw", None), ("power_colour_gated", None),
               ("power_fog_gated", None), ("power_either_gated", None),
               ("zero_power_either", "zero_usable"),
               ("zero_power_rt_clean", "zero_usable")]

IDENTITY_TOL = 1e-12


class CheckFailed(Exception):
    pass


class Checks:
    """Every check, in order. A failed `require` stops the run."""

    def __init__(self):
        self.rows = []

    def require(self, name, ok, detail=""):
        ok = bool(ok)
        self.rows.append({"check": name,
                          "result": "PASS" if ok else "FAIL",
                          "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail else ""))
        if not ok:
            raise CheckFailed(f"{name}: {detail}")

    def info(self, name, detail=""):
        self.rows.append({"check": name, "result": "INFO",
                          "detail": str(detail)})
        print(f"  [INFO] {name}" + (f"  ({detail})" if detail else ""))


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def resolve_output_root(base):
    """base, or base_2, base_3, ...: never an existing folder."""
    if not os.path.exists(base):
        return base
    k = 2
    while os.path.exists(f"{base}_{k}"):
        k += 1
    return f"{base}_{k}"


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    return x


def settings_snapshot():
    g = globals()
    out = {}
    for k, v in g.items():
        if k.isupper() and not callable(v) and not k.startswith("_"):
            try:
                json.dumps(jsonable(v))
                out[k] = jsonable(v)
            except TypeError:
                out[k] = repr(v)
    return out


# ------------------------------------------------------------
# Start-up: input checksums
# ------------------------------------------------------------

def check_inputs(checks, meta):
    banner("CHECKING THE INPUT FILES")
    sums = {}
    for name, want in COMMITTED_SHA256.items():
        path = os.path.join(REGISTRATION_DIR, name)
        checks.require(f"{path} exists", os.path.isfile(path))
        got = sha256_of(path)
        sums[path] = got
        checks.require(f"{name} matches the committed copy (SHA-256)",
                       got == want, got[:16] + "...")
    for name, want in POWER_RUN_SHA256.items():
        path = os.path.join(POWER_FROZEN_DIR, name)
        checks.require(f"{path} exists", os.path.isfile(path))
        got = sha256_of(path)
        sums[path] = got
        checks.require(f"{name} matches the registration's record",
                       got == want, got[:16] + "...")
    checks.require(f"{POWER_SCRIPT} exists next to this script",
                   os.path.isfile(POWER_SCRIPT))
    got = sha256_of(POWER_SCRIPT)
    sums[POWER_SCRIPT] = got
    checks.require(f"{POWER_SCRIPT} is the committed version (SHA-256)",
                   got == POWER_SCRIPT_SHA256, got[:16] + "...")
    sums[os.path.abspath(__file__)] = sha256_of(os.path.abspath(__file__))
    meta["input_sha256"] = sums


# ------------------------------------------------------------
# Start-up: the copied functions against the original
# ------------------------------------------------------------

def function_sources(path, names):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    return {n.name: ast.dump(n) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names}


def referenced_globals(path, names):
    """Names each copied function reads that it does not define itself."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = set()
    for n in tree.body:
        if not (isinstance(n, ast.FunctionDef) and n.name in names):
            continue
        local = {a.arg for a in ast.walk(n) if isinstance(a, ast.arg)}
        for x in ast.walk(n):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                local.add(x.id)
            if isinstance(x, (ast.FunctionDef,)) and x is not n:
                local.add(x.name)
        for x in ast.walk(n):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) \
                    and x.id not in local:
                out.add(x.id)
    return out


def load_power_module(path):
    spec = importlib.util.spec_from_file_location("power_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cmp(a, b):
    """Largest absolute difference between two results; inf if their
    shapes or NaN patterns differ."""
    if isinstance(a, dict):
        if set(a) != set(b):
            return np.inf
        return max((_cmp(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (tuple, list)) and not np.isscalar(a):
        if len(a) != len(b):
            return np.inf
        return max((_cmp(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, str):
        return 0.0 if a == b else np.inf
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    na, nb = np.isnan(a), np.isnan(b)
    if (na != nb).any():
        return np.inf
    if a.size == 0 or na.all():
        return 0.0
    return float(np.max(np.abs(a[~na] - b[~nb])))


def identity_check(X, th_computed, checks, tdir):
    """Each copied function against the original, run now."""
    banner("CHECKING THE COPIED FUNCTIONS AGAINST THE ORIGINAL")
    mine = function_sources(os.path.abspath(__file__), COPIED_FUNCTIONS)
    theirs = function_sources(POWER_SCRIPT, COPIED_FUNCTIONS)
    diff = [n for n in COPIED_FUNCTIONS
            if mine.get(n) is None or mine.get(n) != theirs.get(n)]
    checks.require("copied functions are verbatim (syntax tree)", not diff,
                   f"{len(COPIED_FUNCTIONS) - len(diff)}/"
                   f"{len(COPIED_FUNCTIONS)} identical"
                   + (f"; differ: {', '.join(diff)}" if diff else ""))

    pw = load_power_module(POWER_SCRIPT)
    sdiff = [n for n in COPIED_SETTINGS
             if not hasattr(pw, n) or getattr(pw, n) != globals()[n]]
    checks.require("copied settings equal the power script's", not sdiff,
                   f"{len(COPIED_SETTINGS) - len(sdiff)}/"
                   f"{len(COPIED_SETTINGS)} equal"
                   + (f"; differ: {', '.join(sdiff)}" if sdiff else ""))
    used = referenced_globals(POWER_SCRIPT, COPIED_FUNCTIONS)
    needed = sorted(n for n in used
                    if hasattr(pw, n) and not callable(getattr(pw, n))
                    and not isinstance(getattr(pw, n), type(np))
                    and n != "_MW_SF_CACHE")
    missing = [n for n in needed if n not in COPIED_SETTINGS]
    checks.require("every setting the copies read is a copied setting",
                   not missing, f"{len(needed)} read"
                   + (f"; not copied: {', '.join(missing)}"
                      if missing else ""))

    # --- the same inputs through both ---
    rows = []

    def both(label, f_mine, f_theirs):
        a, b = f_mine(), f_theirs()
        d = _cmp(a, b)
        rows.append({"quantity": label, "max_abs_diff": d,
                     "ok": bool(d <= IDENTITY_TOL)})

    rng = np.random.default_rng(IDENTITY_SEED)
    W = np.column_stack([rng.normal(0, 0.6, 400), rng.normal(0, 0.9, 400)])
    W[:5] = [[0, 0], [3.0, 3.0], [-2.0, 0.5], [0.29, 0.49], [8.0, -8.0]]
    p = 1.0 / (1.0 + np.exp(-(W @ X.T)))
    Y = (rng.random(p.shape) < p).astype(float)
    both("fit_firth_batch (estimates, last step)",
         lambda: fit_firth_batch(X, Y), lambda: pw.fit_firth_batch(X, Y))
    both("fit_batch", lambda: fit_batch(X, Y), lambda: pw.fit_batch(X, Y))
    both("fit_logit_batch", lambda: fit_logit_batch(X, Y[:50]),
         lambda: pw.fit_logit_batch(X, Y[:50]))
    both("simulate_many",
         lambda: simulate_many(X, W, np.random.default_rng(1)),
         lambda: pw.simulate_many(X, W, np.random.default_rng(1)))
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        th_theirs = pw.compute_floors(X, "frozen", [])
    both("compute_floors (upper, lower, sd, every null weight)",
         lambda: th_computed, lambda: th_theirs)
    est = fit_batch(X, Y)
    both("classify", lambda: classify(est, th_computed),
         lambda: pw.classify(est, th_theirs))
    both("informative_medians", lambda: informative_medians(X),
         lambda: pw.informative_medians(X))
    both("as_probability",
         lambda: [as_probability(w, d) for w in (0.3, 0.5, 0.7)
                  for d in (0.6, 1.0)],
         lambda: [pw.as_probability(w, d) for w in (0.3, 0.5, 0.7)
                  for d in (0.6, 1.0)])

    def pool(mod):
        return mod._c1_pool(X, th_computed, 1, 0.7, 0.5, 0.25, 0.5,
                            th_computed["upper"] * 0.5, 0.2,
                            np.random.default_rng(IDENTITY_SEED + 1),
                            n=3000)
    pm, pt = pool(sys.modules[__name__]), pool(pw)
    both("_c1_pool (weights, lapse, catch, fast, responder)",
         lambda: pm, lambda: pt)
    both("draw_attention",
         lambda: draw_attention(500, 0.15, 180,
                                np.random.default_rng(IDENTITY_SEED + 2)),
         lambda: pw.draw_attention(500, 0.15, 180,
                                   np.random.default_rng(IDENTITY_SEED + 2)))
    for size, lab in ((6, "exact null"), (25, "normal approximation")):
        gp = _c1_draw(pm, 300, size, np.random.default_rng(size))
        gc = _c1_draw(pm, 300, size + 3, np.random.default_rng(size + 1))
        both(f"_c1_draw ({size} people)", lambda: gp,
             lambda: pw._c1_draw(pm, 300, size,
                                 np.random.default_rng(size)))
        both(f"mw_p_batch ({lab})",
             lambda: mw_p_batch(gp["w"][:, :, 0], gc["w"][:, :, 0]),
             lambda: pw.mw_p_batch(gp["w"][:, :, 0], gc["w"][:, :, 0]))
        both(f"mw_p_batch (tied counts, {lab})",
             lambda: mw_p_batch(gc["catch"], gp["catch"], exact_max_n=0),
             lambda: pw.mw_p_batch(gc["catch"], gp["catch"],
                                   exact_max_n=0))
        both(f"_c1_test ({lab})", lambda: _c1_test(gp["w"], gc["w"]),
             lambda: pw._c1_test(gp["w"], gc["w"]))
        both(f"_c1_gate ({lab})",
             lambda: _c1_gate(gp["catch"], gc["catch"], gp["fast"],
                              gc["fast"]),
             lambda: pw._c1_gate(gp["catch"], gc["catch"], gp["fast"],
                                 gc["fast"]))
        both(f"_c1_zero_catch ({lab})", lambda: _c1_zero_catch(gp, gc),
             lambda: pw._c1_zero_catch(gp, gc))
    both("mw_exact_sf (7, 9)", lambda: mw_exact_sf(7, 9),
         lambda: pw.mw_exact_sf(7, 9))
    pv = np.random.default_rng(3).random((200, 2)) * 0.08
    both("holm_two", lambda: [holm_two(a, b) for a, b in pv],
         lambda: [pw.holm_two(a, b) for a, b in pv])
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        f_mine = verify_frozen(FROZEN_ROOT, [])
        f_theirs = pw.verify_frozen(FROZEN_ROOT, [])
        _, X2, r2 = pw.load_trials(
            os.path.join(FROZEN_ROOT, FROZEN_TRIAL_LIST), "frozen")
    rows.append({"quantity": "verify_frozen (FROZEN.json facts)",
                 "max_abs_diff": 0.0 if f_mine == f_theirs else np.inf,
                 "ok": f_mine == f_theirs})
    both("load_trials (predictors)", lambda: X, lambda: X2)
    rows.append({"quantity": "run_c1", "max_abs_diff": np.nan,
                 "ok": True, "note": "checked by the replay (step 1)"})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(tdir, "identity_check.csv"), index=False)
    bad = tab[~tab.ok]
    checks.require("copied functions give the original's results",
                   bad.empty, f"{len(tab) - len(bad)}/{len(tab)} identical "
                   f"to {IDENTITY_TOL:g}"
                   + (f"; differ: {', '.join(bad.quantity)}"
                      if len(bad) else ""))
    return tab


# ------------------------------------------------------------
# The cells, and the person pools
# ------------------------------------------------------------

def pool_key(feat_idx, w, share, strat, lam):
    """run_c1's own pool key: a person with no effect is one type
    whatever feature or share the effect would have had."""
    if w == 0.0:
        return (0, 0.0, C1_RESPONDER_SHARES[0], strat, lam)
    return (feat_idx, w, share, strat, lam)


def combos():
    """run_c1's 13 effect / feature / share combinations, in its order."""
    out = [(0, "none", 0.0, C1_RESPONDER_SHARES[0])]
    for feat_idx, feat in enumerate(("colour", "fog")):
        claim = "H2 colour" if feat == "colour" else "H1 fog"
        for w in RULE_RESPONDER_WEIGHTS:
            for share in C1_RESPONDER_SHARES:
                out.append((feat_idx, claim, w, share))
    return out


def build_pools(X, th, taste_sd, rng):
    """Every person-type pool, in run_c1's order and with its key."""
    lap_vals = sorted({v for pr in C1_LAPSE_PAIRS for v in pr})
    effects = [0.0] + list(RULE_RESPONDER_WEIGHTS)
    pools = {}
    for feat_idx in (0, 1):
        for w in effects:
            for share in C1_RESPONDER_SHARES:
                for strat in C1_STRATEGY_RATES:
                    for lam in lap_vals:
                        k = pool_key(feat_idx, w, share, strat, lam)
                        if k in pools:
                            continue
                        pools[k] = _c1_pool(X, th, k[0], k[1], k[2], strat,
                                            C1_STRATEGY_WEIGHT, taste_sd,
                                            lam, rng)
    return pools


def draw_group(pool, studies, size, rng, exclude):
    """`studies` analysable groups of `size` people.

    exclude False: the registered draw, _c1_draw itself.

    exclude True: the registered recruitment. Candidates are drawn from
    the pool in order; anyone with CATCH_FAIL_EXCLUDE or more catch
    failures is excluded and does not count; drawing stops at the
    `size`-th analysable person. Returns the group and, per study, how
    many were drawn and how many excluded. (Drawn people are
    independent, so this is the same as drawing the group from the
    analysable part of the pool; doing it in order is what gives the
    numbers drawn and excluded.)
    """
    if not exclude:
        g = _c1_draw(pool, studies, size, rng)
        return g, np.full(studies, size), np.zeros(studies, dtype=int)
    n_pool = len(pool["w"])
    ok = pool["catch"] < CATCH_FAIL_EXCLUDE
    p_ok = max(float(ok.mean()), 1e-3)
    width = int(np.ceil(1.3 * size / p_ok)) + 8
    cand = rng.integers(0, n_pool, size=(studies, width))
    okc = ok[cand]
    while (okc.sum(axis=1) < size).any():
        more = rng.integers(0, n_pool, size=(studies, width))
        cand = np.concatenate([cand, more], axis=1)
        okc = ok[cand]
    pos = np.argsort(~okc, axis=1, kind="stable")[:, :size]
    idx = np.take_along_axis(cand, pos, axis=1)
    drawn = pos[:, -1] + 1
    g = {k: v[idx] for k, v in pool.items()}
    return g, drawn, drawn - size


def evaluate(gp, gc, n_c):
    """One cell: the registered steps on the analysable groups.

    The verdict order is the analysis's: descriptive below
    MIN_CONTROLS_C1 controls, otherwise inconclusive if the attention
    check fired, otherwise supported or not. The rates are those of
    run_c1, plus the share of descriptive studies.
    """
    r_c, r_f = _c1_test(gp["w"], gc["w"])
    fired, f_catch, f_rt = _c1_gate(gp["catch"], gc["catch"],
                                    gp["fast"], gc["fast"])
    descriptive = n_c < MIN_CONTROLS_C1
    live = ~fired & (not descriptive)
    pos_c, pos_f = r_c & live, r_f & live
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
        "inconclusive": float((fired & (not descriptive)).mean()),
        "descriptive": float(descriptive)}
    z = _c1_zero_catch(gp, gc)
    u = z["usable"]
    either = z["reject_colour"] | z["reject_fog"]
    out.update({
        "zero_usable": float(u.mean()),
        "zero_n_participants": float(z["n_participants"].mean()),
        "zero_n_controls": float(z["n_controls"].mean()),
        "zero_lapse_participants": float(np.nanmean(
            z["lapse_participants"])),
        "zero_lapse_controls": float(np.nanmean(z["lapse_controls"])),
        "zero_power_either": (float(either[u].mean()) if u.any()
                              else np.nan),
        "zero_rt_fires": (float(z["rt_fires"][u].mean()) if u.any()
                          else np.nan),
        "zero_power_rt_clean": (float((either & ~z["rt_fires"])[u].mean())
                                if u.any() else np.nan)})
    out.update({
        "lapse_mean_participants": float(gp["lapse"].mean()),
        "lapse_mean_controls": float(gc["lapse"].mean()),
        "catch_mean_participants": float(gp["catch"].mean()),
        "catch_mean_controls": float(gc["catch"].mean()),
        "catch_max_analysable": float(max(gp["catch"].max(),
                                          gc["catch"].max())),
        "fast_mean_participants": float(gp["fast"].mean()),
        "fast_mean_controls": float(gc["fast"].mean())})
    return out


def run_grid(X, pools, rng, studies, exclude, label):
    """Every cell of step 2 / 3, one arm."""
    med = informative_medians(X)
    cells = []
    for feat_idx, claim, w, share in combos():
        for n_p in PARTICIPANT_N:
            for n_c in CONTROL_N:
                for strat in C1_STRATEGY_RATES:
                    for lam_p, lam_c in C1_LAPSE_PAIRS:
                        cells.append((feat_idx, claim, w, share, n_p, n_c,
                                      strat, lam_p, lam_c))
    rows, t0 = [], time.time()
    for i, (feat_idx, claim, w, share, n_p, n_c, strat, lam_p,
            lam_c) in enumerate(cells):
        pp = pools[pool_key(feat_idx, w, share, strat, lam_p)]
        pc = pools[pool_key(feat_idx, 0.0, share, strat, lam_c)]
        gp, dp, ep = draw_group(pp, studies, n_p, rng, exclude)
        gc, dc, ec = draw_group(pc, studies, n_c, rng, exclude)
        r = evaluate(gp, gc, n_c)
        r.update({
            "pool_q_participants": float(
                (pp["catch"] >= CATCH_FAIL_EXCLUDE).mean()),
            "pool_q_controls": float(
                (pc["catch"] >= CATCH_FAIL_EXCLUDE).mean()),
            "excluded_participants_mean": float(ep.mean()),
            "excluded_participants_p95": float(np.percentile(ep, 95)),
            "excluded_participants_max": int(ep.max()),
            "excluded_participants_any": float((ep > 0).mean()),
            "drawn_participants_mean": float(dp.mean()),
            "excluded_controls_mean": float(ec.mean()),
            "excluded_controls_p95": float(np.percentile(ec, 95)),
            "excluded_controls_max": int(ec.max()),
            "excluded_controls_any": float((ec > 0).mean()),
            "drawn_controls_mean": float(dc.mean()),
            "effect_on": claim, "effect_sd": w,
            "effect_p_typical": round(as_probability(
                w, med["colour"] if feat_idx == 0 else med["fog"]), 4),
            "responder_share": share, "n_participants": n_p,
            "n_controls": n_c, "strategy_rate": strat,
            "lapse_participants": lam_p, "lapse_controls": lam_c,
            "studies": studies, "exclusion": bool(exclude)})
        rows.append(r)
        if i in (19, 199) or (i + 1) % 400 == 0 or i + 1 == len(cells):
            el = time.time() - t0
            print(f"    [{label}] {i + 1}/{len(cells)} cells, "
                  f"{el:.0f}s, about "
                  f"{el / (i + 1) * (len(cells) - i - 1) / 60:.1f} min "
                  "left", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Step 1: the replay
# ------------------------------------------------------------

def replay(X, th_computed, checks, out_dir, tdir):
    """The copied run_c1, unchanged, with the power script's seed.

    While it runs, _c1_pool is wrapped so the pools it builds are kept:
    the wrapper calls the copy with the same arguments and the same
    random stream, so nothing about the run changes. Those pools are
    the registered ones, and step 2 uses them.
    """
    banner("STEP 1: REPLAYING run_c1 (the registered C1 simulation)")
    rdir = os.path.join(out_dir, "replay")
    os.makedirs(rdir, exist_ok=True)
    kept = {}
    original = globals()["_c1_pool"]

    def recording_pool(X_, th_, feat_idx, true_w, responder_share,
                       strategy_rate, strategy_w, taste_sd, lapse_mean,
                       rng, n=None):
        res = original(X_, th_, feat_idx, true_w, responder_share,
                       strategy_rate, strategy_w, taste_sd, lapse_mean,
                       rng, n=n)
        kept[(feat_idx, true_w, responder_share, strategy_rate,
              lapse_mean)] = res
        return res

    t0 = time.time()
    log = os.path.join(out_dir, "logs", "replay_stdout.txt")
    print(f"  running (several minutes; its own output goes to "
          f"{log})", flush=True)
    globals()["_c1_pool"] = recording_pool
    try:
        with open(log, "w") as fh, contextlib.redirect_stdout(fh):
            tab = run_c1(X, th_computed, None, [], rdir)
    finally:
        globals()["_c1_pool"] = original
    print(f"  done in {time.time() - t0:.0f}s, {len(kept)} pools kept")

    ref = pd.read_csv(os.path.join(POWER_FROZEN_DIR, "c1_power.csv"))
    rows = []
    checks.require("replay has the same cells as c1_power.csv",
                   len(tab) == len(ref)
                   and (tab[KEYS].astype(str).values
                        == ref[KEYS].astype(str).values).all(),
                   f"{len(tab)} vs {len(ref)} rows")
    for c in ATTENTION_COLS:
        d = (tab[c] - ref[c]).abs()
        both_nan = tab[c].isna() & ref[c].isna()
        worst = float(d[~both_nan].max()) if (~both_nan).any() else 0.0
        rows.append({"column": c, "kind": "attention (exact)",
                     "max_abs_diff": worst, "max_studies_diff": np.nan,
                     "cells_differing": int((d[~both_nan] > 1e-12).sum()),
                     "ok": bool(worst <= 1e-12)})
    for c, cond in WEIGHT_COLS:
        den = C1_STUDIES * (ref[cond] if cond else 1.0)
        d = (tab[c] - ref[c]).abs()
        both_nan = tab[c].isna() & ref[c].isna()
        st = (d * den)[~both_nan]
        rows.append({"column": c, "kind": "weight-dependent",
                     "max_abs_diff": float(d[~both_nan].max()),
                     "max_studies_diff": float(st.max()),
                     "cells_differing": int((st > 1e-6).sum()),
                     "ok": bool(st.max() <= REPLAY_STUDY_TOL + 1e-6)})
    cmp = pd.DataFrame(rows)
    cmp.to_csv(os.path.join(tdir, "replay_comparison.csv"), index=False)
    print(cmp.to_string(index=False))
    checks.require(
        "replay: attention columns identical to c1_power.csv",
        cmp[cmp.kind.str.startswith("attention")].ok.all())
    checks.require(
        f"replay: weight-dependent rates within {REPLAY_STUDY_TOL} "
        "studies per cell of c1_power.csv",
        cmp[cmp.kind == "weight-dependent"].ok.all(),
        f"worst {cmp.max_studies_diff.max():.0f} studies, "
        f"{int(cmp.cells_differing.sum())} cell-columns differ at all")

    # The canonical registration file is the subset at 20/30/40 controls.
    cells = pd.read_csv(os.path.join(REGISTRATION_DIR, "c1_cells.csv"))
    m = cells.merge(tab, on=KEYS, suffixes=("_reg", "_rep"))
    worst_att = max(float((m[c + "_reg"] - m[c + "_rep"]).abs().max())
                    for c in ATTENTION_COLS)
    worst_w = 0.0
    for c, cond in WEIGHT_COLS:
        den = C1_STUDIES * (m[cond + "_reg"] if cond else 1.0)
        st = ((m[c + "_reg"] - m[c + "_rep"]).abs() * den).max()
        worst_w = max(worst_w, float(np.nan_to_num(st)))
    checks.require(
        "replay reproduces c1_cells.csv (attention exact, weights within "
        f"{REPLAY_STUDY_TOL} studies)",
        len(m) == len(cells) and worst_att <= 1e-12
        and worst_w <= REPLAY_STUDY_TOL + 1e-6,
        f"{len(m)}/{len(cells)} cells matched; worst {worst_w:.0f} studies")
    # Every pool step 2 needs must have been recorded.
    need = build_pool_keys()
    checks.require("replay kept every registered pool",
                   need <= set(kept), f"{len(need & set(kept))}/{len(need)}")
    return tab, cmp, kept


def build_pool_keys():
    lap_vals = sorted({v for pr in C1_LAPSE_PAIRS for v in pr})
    keys = set()
    for feat_idx in (0, 1):
        for w in [0.0] + list(RULE_RESPONDER_WEIGHTS):
            for share in C1_RESPONDER_SHARES:
                for strat in C1_STRATEGY_RATES:
                    for lam in lap_vals:
                        keys.add(pool_key(feat_idx, w, share, strat, lam))
    return keys


# ------------------------------------------------------------
# Step 2: validation statistics
# ------------------------------------------------------------

def z_table(reg, new, n_reg, n_new):
    m = reg.merge(new, on=KEYS, suffixes=("_reg", "_new"))
    out = []
    for meas, cond in VAL_MEASURES:
        p1, p2 = m[meas + "_reg"].to_numpy(), m[meas + "_new"].to_numpy()
        n1 = n_reg * (m[cond + "_reg"].to_numpy() if cond else 1.0)
        n2 = n_new * (m[cond + "_new"].to_numpy() if cond else 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = (p1 * n1 + p2 * n2) / (n1 + n2)
            var = p * (1 - p) * (1 / n1 + 1 / n2)
            z = np.where(var > 0, (p2 - p1) / np.sqrt(var), 0.0)
        skip = (np.isnan(p1) | np.isnan(p2) | (n1 < VAL_MIN_N)
                | (n2 < VAL_MIN_N))
        d = m[KEYS].copy()
        d["measure"] = meas
        d["registered"], d["new"] = p1, p2
        d["n_registered"], d["n_new"] = n1, n2
        d["z"] = np.where(skip, np.nan, z)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def validation_verdict(zt, checks):
    z = zt.z.dropna()
    share = float((z.abs() > VAL_Z).mean())
    worst = float(z.abs().max())
    mean = float(z.mean())
    detail = (f"{len(z)} comparisons ({int(zt.z.isna().sum())} skipped); "
              f"{share:.1%} beyond {VAL_Z}; max |z| {worst:.2f}; "
              f"mean z {mean:+.3f}; SD z {float(z.std()):.3f}")
    print("  " + detail)
    checks.require(
        "validation: exclusion off reproduces c1_cells.csv within Monte "
        "Carlo error", share <= VAL_MAX_SHARE_BEYOND
        and worst <= VAL_MAX_ABS_Z and abs(mean) <= VAL_MEAN_Z, detail)
    return {"comparisons": len(z), "skipped": int(zt.z.isna().sum()),
            "share_beyond": share, "max_abs_z": worst, "mean_z": mean,
            "sd_z": float(z.std())}


# ------------------------------------------------------------
# Exclusion arithmetic, for a cross-check of the simulation
# ------------------------------------------------------------

def p_excluded_exact(lam_mean):
    """P(a person has >= CATCH_FAIL_EXCLUDE catch failures), integrating
    the binomial over the gamma distribution of personal lapse rates
    (the clip at 0.9 is applied, as in _c1_pool)."""
    if lam_mean <= 0:
        return 0.0
    shape = 1.0 / max(LAPSE_BETWEEN_SD, 1e-6) ** 2
    scale = lam_mean / shape

    def f(lam):
        q = min(CATCH_P_FAIL_PER_LAPSE * min(lam, 0.9), 1.0)
        return (stats.binom.sf(CATCH_FAIL_EXCLUDE - 1, N_CATCH_TRIALS, q)
                * stats.gamma.pdf(lam, shape, scale=scale))
    from scipy.integrate import quad
    return float(quad(f, 0, np.inf, limit=200)[0])


# ------------------------------------------------------------
# Summaries
# ------------------------------------------------------------

def lapse_kind(r):
    if r.lapse_participants == r.lapse_controls:
        return "matched"
    return ("controls lapse more" if r.lapse_controls > r.lapse_participants
            else "participants lapse more")


def side_by_side(reg, val, off, on):
    meas = ["power_colour_gated", "power_fog_gated", "power_either_gated",
            "power_either_raw", "gate_fires", "inconclusive",
            "zero_usable", "zero_power_either", "zero_n_participants",
            "zero_n_controls"]
    out = reg[KEYS + ["effect_p_typical"]].copy()
    for name, t in (("reg", reg), ("val", val), ("off", off), ("on", on)):
        sub = t[KEYS + meas].rename(columns={c: f"{c}_{name}"
                                             for c in meas})
        out = out.merge(sub, on=KEYS, how="left")
    extra = ["excluded_participants_mean", "excluded_participants_p95",
             "excluded_controls_mean", "excluded_controls_p95",
             "lapse_mean_participants", "lapse_mean_controls",
             "catch_mean_participants", "catch_mean_controls"]
    out = out.merge(on[KEYS + extra], on=KEYS, how="left")
    S = STUDIES_PER_CELL
    for c in ("power_colour_gated", "power_fog_gated",
              "power_either_gated", "gate_fires"):
        out[f"{c}_on_minus_reg"] = out[f"{c}_on"] - out[f"{c}_reg"]
        out[f"{c}_on_minus_off"] = out[f"{c}_on"] - out[f"{c}_off"]
        pon, poff = out[f"{c}_on"], out[f"{c}_off"]
        out[f"{c}_on_minus_off_se"] = np.sqrt(
            (pon * (1 - pon) + poff * (1 - poff)) / S)
    out["lapse_kind"] = out.apply(lapse_kind, axis=1)
    return out


def summaries(on, off, reg, val):
    """Exclusion counts, the check's firing rate, the false positives."""
    # --- how many are excluded ---
    rows = []
    for grp, lcol, ncol in (("participants", "lapse_participants",
                             "n_participants"),
                            ("controls", "lapse_controls", "n_controls")):
        for (lam, n), g in on.groupby([lcol, ncol]):
            q = p_excluded_exact(lam)
            # The number excluded before the n-th analysable person is
            # negative binomial: mean n q / (1 - q), variance
            # n q / (1 - q)^2, with q the pool's own exclusion share.
            qp = g[f"pool_q_{grp}"]
            exp_pool = (n * qp / (1 - qp)).mean()
            var_pool = (n * qp / (1 - qp) ** 2).mean()
            obs = g[f"excluded_{grp}_mean"].mean()
            se = np.sqrt(var_pool / (STUDIES_PER_CELL * len(g)))
            rows.append({
                "group": grp, "group_lapse": lam, "target_analysable": n,
                "p_excluded_per_person_exact": q,
                "p_excluded_per_person_pools": qp.mean(),
                "excluded_expected": n * q / (1 - q) if q < 1 else np.inf,
                "excluded_expected_pools": exp_pool,
                "excluded_mean_z_vs_pools": ((obs - exp_pool) / se
                                             if se > 0 else 0.0),
                "excluded_mean": obs,
                "excluded_p95": g[f"excluded_{grp}_p95"].mean(),
                "excluded_max": int(g[f"excluded_{grp}_max"].max()),
                "share_of_studies_with_any": g[f"excluded_{grp}_any"].mean(),
                "drawn_mean": g[f"drawn_{grp}_mean"].mean(),
                "mean_personal_lapse_analysable":
                    g[f"lapse_mean_{grp}"].mean(),
                "mean_catch_failures_analysable":
                    g[f"catch_mean_{grp}"].mean(),
                "mean_catch_failures_no_exclusion":
                    off[(off[lcol] == lam) & (off[ncol] == n)][
                        f"catch_mean_{grp}"].mean(),
                "cells": len(g)})
    excl = pd.DataFrame(rows)

    # --- how often the check fires ---
    rows = []
    for (lp, lc), _ in on.groupby(["lapse_participants", "lapse_controls"]):
        sel = lambda t: t[(t.lapse_participants == lp)
                          & (t.lapse_controls == lc)]
        r = {"lapse_participants": lp, "lapse_controls": lc,
             "lapse_kind": lapse_kind(pd.Series(
                 {"lapse_participants": lp, "lapse_controls": lc}))}
        for name, t in (("registered", reg), ("validation_off", val),
                        ("off", off), ("on", on)):
            s = sel(t)
            r[f"fires_mean_{name}"] = s.gate_fires.mean()
            r[f"fires_min_{name}"] = s.gate_fires.min()
            r[f"fires_max_{name}"] = s.gate_fires.max()
        s = sel(on)
        r["fires_catch_on"] = s.gate_fires_catch.mean()
        r["fires_rt_on"] = s.gate_fires_rt.mean()
        r["fires_catch_off"] = sel(off).gate_fires_catch.mean()
        r["fires_rt_off"] = sel(off).gate_fires_rt.mean()
        rows.append(r)
    gate = pd.DataFrame(rows)

    # --- the false positives: no effect anywhere ---
    rows = []
    null = lambda t: t[np.isclose(t.effect_sd, 0.0)]
    for (lp, lc), _ in null(on).groupby(["lapse_participants",
                                         "lapse_controls"]):
        r = {"lapse_participants": lp, "lapse_controls": lc,
             "lapse_kind": lapse_kind(pd.Series(
                 {"lapse_participants": lp, "lapse_controls": lc}))}
        for name, t in (("registered", reg), ("off", off), ("on", on)):
            s = null(t)
            s = s[(s.lapse_participants == lp) & (s.lapse_controls == lc)]
            for c in ("power_colour_gated", "power_fog_gated",
                      "power_either_gated", "power_either_raw",
                      "zero_power_either"):
                r[f"{c}_mean_{name}"] = s[c].mean()
                r[f"{c}_worst_{name}"] = s[c].max()
        rows.append(r)
    fp = pd.DataFrame(rows)
    return excl, gate, fp


def compact_side_by_side(on, off, val):
    comp = pd.read_csv(os.path.join(REGISTRATION_DIR, "c1_compact.csv"))
    comp = comp[comp.n_controls.isin(CONTROL_N)]
    meas = ["power_colour_gated", "power_fog_gated", "power_either_gated",
            "gate_fires"]
    out = comp.copy()
    for name, t in (("validation_off", val), ("off", off), ("on", on)):
        sub = t[KEYS + meas].rename(columns={c: f"{c}_{name}"
                                             for c in meas})
        out = out.merge(sub, on=KEYS, how="left")
    sub = on[KEYS + ["excluded_participants_mean",
                     "excluded_controls_mean"]]
    return out.merge(sub, on=KEYS, how="left")


# ------------------------------------------------------------
# Figures
# ------------------------------------------------------------

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)


def fig_compact(comp, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[1].sharey(axes[0])
    S = STUDIES_PER_CELL
    for ax, grid, col, title in (
            (axes[0], "colour", "power_colour_gated",
             "C1 power, effect on colour"),
            (axes[1], "fog", "power_fog_gated", "C1 power, effect on fog"),
            (axes[2], "no effect", "power_either_gated",
             "false positive (either feature), no effect")):
        g = comp[comp.grid == grid]
        for j, n_c in enumerate(CONTROL_N):
            s = g[g.n_controls == n_c].sort_values("n_participants")
            ax.plot(s.n_participants, s[col], color=SERIES_COLOURS[0],
                    marker=SERIES_MARKERS[j], ls=SERIES_STYLES[j],
                    label=f"registered, {n_c} controls")
            p = s[col + "_on"]
            ax.errorbar(s.n_participants + 0.25, p,
                        yerr=1.96 * np.sqrt(p * (1 - p) / S),
                        color=SERIES_COLOURS[2], marker=SERIES_MARKERS[j],
                        ls=SERIES_STYLES[j], capsize=2,
                        label=f"with the exclusion, {n_c} controls")
        if grid == "no effect":
            ax.axhline(ALPHA, color="#555555", lw=0.8, ls=":")
            ax.set_ylim(0, 0.1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("participants")
        ax.set_xticks(PARTICIPANT_N)
        _style(ax)
    axes[0].set_ylabel("share of studies: C1 supported")
    axes[0].legend(fontsize=7, frameon=False)
    r = comp.iloc[0]
    fig.suptitle("The registered compact scenario: share "
                 f"{r.responder_share:g}, strategy {r.strategy_rate:g}, "
                 f"lapse {r.lapse_participants:g}/{r.lapse_controls:g}. "
                 "Bars: 95% Monte Carlo interval.", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_false_positive(fp, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(fp))
    for j, (name, lab) in enumerate((("registered", "registered"),
                                     ("on", "with the exclusion"))):
        mean = fp[f"power_either_gated_mean_{name}"]
        worst = fp[f"power_either_gated_worst_{name}"]
        ax.plot(x + (j - 0.5) * 0.2, mean, ls="none",
                marker=SERIES_MARKERS[j], color=SERIES_COLOURS[2 * j],
                label=f"{lab}: mean over cells")
        ax.plot(x + (j - 0.5) * 0.2, worst, ls="none", marker="_",
                ms=14, color=SERIES_COLOURS[2 * j],
                label=f"{lab}: worst cell")
    ax.axhline(ALPHA, color="#555555", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g} / {b:g}" for a, b in
                        zip(fp.lapse_participants, fp.lapse_controls)])
    ax.set_xlabel("lapse rate: participants / controls")
    ax.set_ylabel("false positive (either feature)")
    ax.set_title("C1 with no effect anywhere, after the attention check "
                 "(18 cells per lapse pair: sizes x strategy)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_gate(gate, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(gate))
    for j, (name, lab) in enumerate((("registered", "registered"),
                                     ("on", "with the exclusion"))):
        ax.bar(x + (j - 0.5) * 0.38, gate[f"fires_mean_{name}"], 0.36,
               color=SERIES_COLOURS[2 * j], label=lab,
               hatch=None if j == 0 else "//", edgecolor="white")
    ax.axhline(GATE_ALPHA, color="#555555", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g} / {b:g}" for a, b in
                        zip(gate.lapse_participants, gate.lapse_controls)])
    ax.set_xlabel("lapse rate: participants / controls")
    ax.set_ylabel("share of studies where the check fires")
    ax.set_title("The attention check (mean over cells)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_exclusions(excl, path):
    e = excl[excl.group_lapse > 0]
    sizes = sorted(e.target_analysable.unique())
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(sizes))
    for j, lam in enumerate(sorted(e.group_lapse.unique())):
        s = (e[e.group_lapse == lam].groupby("target_analysable")
             [["excluded_mean", "excluded_p95"]].mean().reindex(sizes))
        ax.bar(x + (j - 0.5) * 0.38, s.excluded_mean, 0.36,
               color=SERIES_COLOURS[j], label=f"group lapse {lam:g}: mean")
        ax.plot(x + (j - 0.5) * 0.38, s.excluded_p95, ls="none",
                marker="_", ms=16, color="#333333",
                label="95th percentile" if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("target analysable size")
    ax.set_ylabel("people excluded per study")
    ax.set_title("Excluded for 3+ catch failures before the target is "
                 "reached", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


# ------------------------------------------------------------
# The report
# ------------------------------------------------------------

def fmt_tab(df, cols=None, rnd=3):
    d = df if cols is None else df[cols]
    return d.round(rnd).to_string(index=False)


def write_report(path, R, checks, meta):
    L = []
    W = L.append
    W("C1 AFTER THE CATCH-TRIAL EXCLUSION  (simulate_c1_post_exclusion_v4)")
    W("=" * 70)
    W(f"run          {meta['started']}  ({meta.get('elapsed_min', 0):.1f} "
      "min)")
    W(f"output       {meta['output']}")
    W(f"python       {meta['python']}, numpy {meta['numpy']}, "
      f"scipy {meta['scipy']}, pandas {meta['pandas']}")
    W("")
    W("WHAT THIS IS")
    W("-" * 70)
    W("The registered C1 numbers (c1_compact.csv, c1_cells.csv) were "
      "simulated without")
    W("the registered exclusion of people with 3 or more catch failures. "
      "This run")
    W("simulates the whole registered procedure: recruit until each group "
      "has its")
    W("target number of analysable people, then the attention check, then "
      "C1 per")
    W("feature with Holm; inconclusive if the check fired, descriptive "
      "below 20")
    W("controls; the zero-catch-failure test beside it. Nothing registered "
      "is changed.")
    W("")
    W("Thresholds. Replay and validation use the thresholds the power "
      "script computes,")
    W("as the registered rates did; the registered procedure (step 3) uses "
      "the rounded")
    W("registered ones, which the analysis applies. In C1 a threshold "
      "enters only")
    W("through the taste spread, 0.5 x the upper threshold.")
    th = R.get("th_computed")
    if th is not None:
        for k, f in enumerate(("colour", "fog")):
            reg = REGISTERED_THRESHOLDS[f]
            W(f"  {f:6s} computed upper {th['upper'][k]:+.5f} lower "
              f"{th['lower'][k]:+.5f}; registered {reg[0]:+.3f} / "
              f"{reg[1]:+.3f}; difference {reg[0] - th['upper'][k]:+.5f} "
              f"/ {reg[1] - th['lower'][k]:+.5f}")
    W("")
    W("SETTINGS THAT DEFINE THE RUN")
    W("-" * 70)
    W(f"  studies per cell        {STUDIES_PER_CELL} (steps 2 and 3); the "
      f"replay keeps {C1_STUDIES}")
    W(f"  cells                   13 effect/feature/share x participants "
      f"{PARTICIPANT_N} x controls {CONTROL_N}")
    W(f"                          x strategy {C1_STRATEGY_RATES} x 7 lapse "
      "pairs = 1,638")
    W(f"  controls                40 is not run: the registered maximum "
      f"is {N_MAX_CONTROLS}")
    W(f"  recruitment             unlimited: every study reaches its "
      "target")
    W(f"  exclusion               {CATCH_FAIL_EXCLUDE}+ of "
      f"{N_CATCH_TRIALS} catch trials failed")
    W(f"  person model            run_c1's: taste "
      f"{NONRESPONDER_SPREAD_MULT}x threshold, strategy weight "
      f"{C1_STRATEGY_WEIGHT},")
    W(f"                          personal lapse gamma (SD "
      f"{LAPSE_BETWEEN_SD}x mean), catch p_fail "
      f"{CATCH_P_FAIL_PER_LAPSE} x lapse,")
    W(f"                          P(fast | lapse) {RT_FAST_GIVEN_LAPSE}, "
      f"P(fast | engaged) {RT_FAST_GIVEN_ENGAGED}")
    W(f"  pools                   {C1_POOL_PEOPLE:,} people per type")
    W(f"  seeds                   replay {RANDOM_SEED}+77 (run_c1's own); "
      f"validation draws {SEED_VALIDATION_DRAWS};")
    W(f"                          main pools {SEED_MAIN_POOLS}; main draws "
      f"{SEED_MAIN_DRAWS} (on), {SEED_MAIN_DRAWS + 1} (off)")
    W("")
    W("CRITERIA, FIXED BEFORE THE RUN")
    W("-" * 70)
    W(f"  replay      attention columns identical to c1_power.csv; "
      "weight-dependent")
    W(f"              rates within {REPLAY_STUDY_TOL} studies per cell")
    W("  validation  exclusion off, registered pools, new draws, against "
      "c1_cells.csv")
    W("              (20 and 30 controls), per cell and measure: z = "
      "difference /")
    W("              sqrt(p(1-p)(1/n_reg + 1/n_new)). Pass if at most "
      f"{VAL_MAX_SHARE_BEYOND:.0%} of |z|")
    W(f"              exceed {VAL_Z}, none exceeds {VAL_MAX_ABS_Z}, and "
      f"the mean z is within")
    W(f"              +/-{VAL_MEAN_Z}. Conditional measures use the usable "
      f"studies as n; skipped below {VAL_MIN_N}.")
    W("")
    W("CHECKS")
    W("-" * 70)
    for r in checks.rows:
        W(f"  [{r['result']}] {r['check']}"
          + (f"  ({r['detail']})" if r["detail"] else ""))
    if R.get("failed"):
        W("")
        W(f"THE RUN STOPPED: {R['failed']}")
        W("No result below this point is valid.")
        open(path, "w").write("\n".join(L) + "\n")
        return
    if "validation" in R:
        v = R["validation"]
        W("")
        W(f"  validation: {v['comparisons']} comparisons, "
          f"{v['share_beyond']:.1%} beyond {VAL_Z}, max |z| "
          f"{v['max_abs_z']:.2f}, mean z {v['mean_z']:+.3f}, SD z "
          f"{v['sd_z']:.3f}")
    if "excl" not in R:
        open(path, "w").write("\n".join(L) + "\n")
        return

    excl, gate, fp, comp = R["excl"], R["gate"], R["fp"], R["comp"]
    W("")
    W("RESULTS")
    W("=" * 70)
    W("")
    W("1. HOW MANY ARE EXCLUDED BEFORE THE TARGET IS REACHED")
    W("-" * 70)
    W("Per study, averaged over the cells with that group lapse rate and "
      "target size.")
    W("q_exact is the per-person exclusion probability under the lapse "
      "model (binomial")
    W("catch failures integrated over the gamma of personal lapse rates); "
      "q_pools is the")
    W("share in the simulated pools. 'expected' is n q / (1 - q), the "
      "negative-binomial")
    W("mean; z compares the simulated mean with it at the pools' own q.")
    e = excl[excl.group_lapse > 0].rename(columns={
        "p_excluded_per_person_exact": "q_exact",
        "p_excluded_per_person_pools": "q_pools",
        "excluded_expected": "expected",
        "excluded_mean_z_vs_pools": "z",
        "share_of_studies_with_any": "any_excluded",
        "mean_personal_lapse_analysable": "lapse_kept",
        "mean_catch_failures_analysable": "catch_kept",
        "mean_catch_failures_no_exclusion": "catch_no_excl"})
    W(fmt_tab(e, ["group", "group_lapse", "target_analysable", "q_exact",
                  "q_pools", "expected", "excluded_mean", "z",
                  "excluded_p95", "excluded_max", "any_excluded",
                  "drawn_mean"]))
    W("")
    W("  What the analysable people look like (mean personal lapse rate "
      "and mean catch")
    W("  failures, with the exclusion and without it):")
    W(fmt_tab(e, ["group", "group_lapse", "target_analysable",
                  "lapse_kept", "catch_kept", "catch_no_excl"]))
    W("At group lapse 0 nobody fails a catch trial and nobody is "
      "excluded.")
    W("")
    W("2. HOW OFTEN THE ATTENTION CHECK FIRES (C1 inconclusive)")
    W("-" * 70)
    W("Mean over the 234 cells of each lapse pair (and the range); "
      "'off' is the same")
    W("pools as 'on' with the exclusion switched off.")
    W(fmt_tab(gate, ["lapse_participants", "lapse_controls", "lapse_kind",
                     "fires_mean_registered", "fires_mean_off",
                     "fires_mean_on", "fires_min_on", "fires_max_on",
                     "fires_catch_off", "fires_catch_on", "fires_rt_off",
                     "fires_rt_on"]))
    W("")
    m = gate[gate.lapse_kind == "matched"]
    a = gate[gate.lapse_kind == "controls lapse more"]
    W(f"  matched lapses: fires {m.fires_mean_on.mean():.3f} on average, "
      f"worst cell {m.fires_max_on.max():.3f} (registered "
      f"{m.fires_mean_registered.mean():.3f}, nominal {GATE_ALPHA})")
    W(f"  controls lapse more: fires {a.fires_min_on.min():.3f} to "
      f"{a.fires_max_on.max():.3f} (registered "
      f"{a.fires_min_registered.min():.3f} to "
      f"{a.fires_max_registered.max():.3f})")
    W("")
    W("3. FALSE POSITIVES: NO EFFECT IN EITHER GROUP")
    W("-" * 70)
    W("C1 supported (not inconclusive), per feature and for either "
      "feature; mean and")
    W("worst over the 18 cells (sizes x strategy) of each lapse pair.")
    W(fmt_tab(fp, ["lapse_participants", "lapse_controls",
                   "power_colour_gated_worst_registered",
                   "power_colour_gated_worst_on",
                   "power_fog_gated_worst_registered",
                   "power_fog_gated_worst_on",
                   "power_either_gated_mean_registered",
                   "power_either_gated_mean_on",
                   "power_either_gated_worst_registered",
                   "power_either_gated_worst_on"]))
    W("")
    W("  Ungated (the test read straight off the weights) and the "
      "zero-catch-failure")
    W("  test, worst cell:")
    W(fmt_tab(fp, ["lapse_participants", "lapse_controls",
                   "power_either_raw_worst_registered",
                   "power_either_raw_worst_on",
                   "zero_power_either_worst_registered",
                   "zero_power_either_worst_on"]))
    W("")
    W("4. C1 POWER AT THE REGISTERED COMPACT SCENARIO (c1_compact.csv)")
    W("-" * 70)
    r0 = comp.iloc[0]
    W(f"Share {r0.responder_share:g}, strategy {r0.strategy_rate:g} in both "
      f"groups, lapse {r0.lapse_participants:g}/{r0.lapse_controls:g}; "
      "effect 0.5/SD on colour or")
    W("0.7/SD on fog. 'on' is the registered procedure with the exclusion; "
      "'off' the")
    W("same pools without it. Studies per cell: "
      f"registered {C1_STUDIES}, this run {STUDIES_PER_CELL}.")
    for grid, col in (("colour", "power_colour_gated"),
                      ("fog", "power_fog_gated"),
                      ("no effect", "power_either_gated")):
        g = comp[comp.grid == grid]
        W(f"\n  {grid}: {col}")
        W(fmt_tab(g, ["n_participants", "n_controls", col,
                      col + "_validation_off", col + "_off", col + "_on",
                      "gate_fires", "gate_fires_on",
                      "excluded_participants_mean",
                      "excluded_controls_mean"]))
    W("")
    W("5. C1 POWER OVER ALL CELLS, matched lapses, 30 controls")
    W("-" * 70)
    sb = R["sbs"]
    pw_ = sb[(sb.effect_sd > 0) & (sb.n_controls == 30)
             & (sb.lapse_kind == "matched")].copy()
    pw_["power_reg"] = np.where(pw_.effect_on == "H2 colour",
                                pw_.power_colour_gated_reg,
                                pw_.power_fog_gated_reg)
    pw_["power_on"] = np.where(pw_.effect_on == "H2 colour",
                               pw_.power_colour_gated_on,
                               pw_.power_fog_gated_on)
    piv = pw_.pivot_table(index=["effect_on", "effect_sd",
                                 "responder_share", "lapse_participants"],
                          columns="n_participants",
                          values=["power_reg", "power_on"],
                          aggfunc="mean")
    W("Mean over the three strategy rates; columns: participants.")
    W(piv.round(3).to_string())
    W("")
    d = (pw_.power_on - pw_.power_reg)
    W(f"  difference on - registered over these cells: mean {d.mean():+.4f}"
      f", range {d.min():+.4f} to {d.max():+.4f}")
    W("")
    W("6. THE EFFECT OF THE EXCLUSION ITSELF (paired: same pools, on - off)")
    W("-" * 70)
    W("On minus off per cell, then the mean and range over the cells of "
      "each kind;")
    W("'mean z' is the mean of (on - off) / SE per cell (SE from both "
      "arms' Monte Carlo")
    W("error; n/a where both arms are 0 or 1 in every study).")
    sb = sb.copy()
    sb["power_tested_on_minus_off"] = np.where(
        sb.effect_on == "H2 colour", sb.power_colour_gated_on_minus_off,
        sb.power_fog_gated_on_minus_off)
    sb["power_tested_on_minus_off_se"] = np.where(
        sb.effect_on == "H2 colour", sb.power_colour_gated_on_minus_off_se,
        sb.power_fog_gated_on_minus_off_se)
    for kind in ("matched", "controls lapse more",
                 "participants lapse more"):
        s = sb[sb.lapse_kind == kind]
        for lab, sel, c in (
                ("check fires, all cells", s, "gate_fires"),
                ("false positive, no effect", s[s.effect_sd == 0],
                 "power_either_gated"),
                ("power, tested feature", s[s.effect_sd > 0],
                 "power_tested")):
            dd = sel[f"{c}_on_minus_off"]
            se_ = sel[f"{c}_on_minus_off_se"].replace(0, np.nan)
            zz = (dd / se_).mean()
            W(f"  {kind:24s} {lab:27s} mean {dd.mean():+.4f}, range "
              f"{dd.min():+.4f} to {dd.max():+.4f}, mean z "
              + ("n/a" if not np.isfinite(zz) else f"{zz:+.2f}"))
    W("")
    W("NOTES")
    W("-" * 70)
    W("- The catch failures and fast responses come from the power "
      "script's lapse model,")
    W("  not from data; the report on the gate in the power script says "
      "why its")
    W("  sensitivity is an upper bound (people may lapse less on a catch "
      "screen).")
    W("- Recruitment is unlimited here. A smaller final N is covered by "
      "the registered")
    W("  rule for that N and by 'descriptive below 20 controls'.")
    W("- Every number is a rate over simulated studies; the Monte Carlo "
      "SE of a rate p")
    W(f"  is sqrt(p(1-p)/{STUDIES_PER_CELL}) here and "
      f"sqrt(p(1-p)/{C1_STUDIES}) in the registered tables.")
    W("")
    W("FILES")
    W("-" * 70)
    for f in sorted(os.listdir(os.path.join(meta["output"], "tables"))):
        W(f"  tables/{f}")
    if os.path.isdir(os.path.join(meta["output"], "figures")):
        for f in sorted(os.listdir(os.path.join(meta["output"],
                                                "figures"))):
            W(f"  figures/{f}")
    open(path, "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    t_start = time.time()
    out = resolve_output_root(OUTPUT_ROOT)
    for sub in ("tables", "figures", "logs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    tdir, fdir = os.path.join(out, "tables"), os.path.join(out, "figures")
    meta = {"script": os.path.basename(__file__),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output": out, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": stats.__name__ and
            __import__("scipy").__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform()}
    print(f"output: {out}")
    checks, R = Checks(), {}
    try:
        check_inputs(checks, meta)
        banner("THE FROZEN TRIAL LIST")
        facts = verify_frozen(FROZEN_ROOT, [])
        trials, X, _r = load_trials(
            os.path.join(FROZEN_ROOT, FROZEN_TRIAL_LIST), "frozen")
        checks.require("frozen trial list verified; 180 analysed trials",
                       len(X) == 180, f"{len(X)} analysed trials")
        meta["frozen"] = facts
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            th_c = compute_floors(X, "frozen", [])
        R["th_computed"] = th_c
        rounded = [round(float(v), 3) for v in
                   (th_c["upper"][0], th_c["lower"][0],
                    th_c["upper"][1], th_c["lower"][1])]
        want = [REGISTERED_THRESHOLDS["colour"][0],
                REGISTERED_THRESHOLDS["colour"][1],
                REGISTERED_THRESHOLDS["fog"][0],
                REGISTERED_THRESHOLDS["fog"][1]]
        checks.require("computed thresholds round to the registered ones",
                       rounded == want, f"{rounded}")
        identity_check(X, th_c, checks, tdir)

        # --- step 1 ---
        if RUN_REPLAY:
            _, _, reg_pools = replay(X, th_c, checks, out, tdir)
        else:
            banner("STEP 1 SKIPPED (RUN_REPLAY = False): rebuilding the "
                   "registered pools")
            checks.info("replay skipped",
                        "pools rebuilt with run_c1's seed and order; their "
                        "identity is tested only by the validation")
            reg_pools = build_pools(
                X, th_c, th_c["upper"] * NONRESPONDER_SPREAD_MULT,
                np.random.default_rng(RANDOM_SEED + 77))

        # --- step 2 ---
        banner("STEP 2: VALIDATION, exclusion off, registered pools")
        val = run_grid(X, reg_pools, np.random.default_rng(
            SEED_VALIDATION_DRAWS), STUDIES_PER_CELL, False, "validation")
        val.to_csv(os.path.join(tdir, "validation_off_cells.csv"),
                   index=False)
        reg = pd.read_csv(os.path.join(REGISTRATION_DIR, "c1_cells.csv"))
        reg = reg[reg.n_controls.isin(CONTROL_N)].reset_index(drop=True)
        checks.require("validation has every registered cell at 20/30 "
                       "controls", len(reg.merge(val, on=KEYS)) == len(reg)
                       == len(val), f"{len(val)} cells")
        zt = z_table(reg, val, C1_STUDIES, STUDIES_PER_CELL)
        zt.to_csv(os.path.join(tdir, "validation_cells.csv"), index=False)
        R["validation"] = validation_verdict(zt, checks)
        del reg_pools

        # --- step 3 ---
        banner("STEP 3: THE REGISTERED PROCEDURE, with the exclusion")
        th_reg = {"upper": np.array([REGISTERED_THRESHOLDS["colour"][0],
                                     REGISTERED_THRESHOLDS["fog"][0]]),
                  "lower": np.array([REGISTERED_THRESHOLDS["colour"][1],
                                     REGISTERED_THRESHOLDS["fog"][1]])}
        t0 = time.time()
        pools = build_pools(X, th_reg,
                            th_reg["upper"] * NONRESPONDER_SPREAD_MULT,
                            np.random.default_rng(SEED_MAIN_POOLS))
        print(f"  {len(pools)} pools at the registered thresholds, "
              f"{time.time() - t0:.0f}s")
        on = run_grid(X, pools, np.random.default_rng(SEED_MAIN_DRAWS),
                      STUDIES_PER_CELL, True, "exclusion on")
        on.to_csv(os.path.join(tdir, "c1_post_exclusion.csv"), index=False)
        checks.require("with the exclusion, nobody analysable has "
                       f"{CATCH_FAIL_EXCLUDE}+ catch failures",
                       on.catch_max_analysable.max() < CATCH_FAIL_EXCLUDE,
                       f"max {on.catch_max_analysable.max():.0f}")
        off = run_grid(X, pools, np.random.default_rng(SEED_MAIN_DRAWS + 1),
                       STUDIES_PER_CELL, False, "exclusion off")
        off.to_csv(os.path.join(tdir, "c1_exclusion_off.csv"), index=False)
        checks.require("with the exclusion off, nobody is excluded",
                       off.excluded_participants_max.max() == 0
                       and off.excluded_controls_max.max() == 0)

        excl, gate, fp = summaries(on, off, reg, val)
        tol = excl[excl.group_lapse > 0]
        worst = float(tol.excluded_mean_z_vs_pools.abs().max())
        checks.require("recruitment: numbers excluded match the negative "
                       "binomial expectation from each pool's own "
                       "exclusion share", worst <= 4.5,
                       f"worst {worst:.2f} SE")
        dq = float((tol.p_excluded_per_person_pools
                    - tol.p_excluded_per_person_exact).abs().max())
        checks.info("pools' exclusion share against the exact lapse-model "
                    "probability", f"largest difference {dq:.4f} "
                    f"(pools of {C1_POOL_PEOPLE:,}: SE about 0.003)")
        sbs = side_by_side(reg, val, off, on)
        comp = compact_side_by_side(on, off, val)
        excl.to_csv(os.path.join(tdir, "exclusion_counts.csv"), index=False)
        gate.to_csv(os.path.join(tdir, "gate_summary.csv"), index=False)
        fp.to_csv(os.path.join(tdir, "false_positive_summary.csv"),
                  index=False)
        sbs.to_csv(os.path.join(tdir, "c1_side_by_side.csv"), index=False)
        comp.to_csv(os.path.join(tdir, "c1_compact_side_by_side.csv"),
                    index=False)
        R.update({"excl": excl, "gate": gate, "fp": fp, "comp": comp,
                  "sbs": sbs})
        if MAKE_FIGURES:
            fig_compact(comp, os.path.join(fdir, "c1_compact_power.png"))
            fig_false_positive(fp, os.path.join(
                fdir, "c1_false_positive_by_lapse.png"))
            fig_gate(gate, os.path.join(fdir, "attention_check_fires.png"))
            fig_exclusions(excl, os.path.join(fdir, "exclusions.png"))
    except CheckFailed as e:
        R["failed"] = str(e)
        print(f"\nSTOPPED: a check failed: {e}")
    meta["elapsed_min"] = (time.time() - t_start) / 60
    meta["checks"] = checks.rows
    meta["settings"] = settings_snapshot()
    meta["result"] = "stopped: " + R["failed"] if "failed" in R else \
        "complete"
    if "validation" in R:
        meta["validation"] = R["validation"]
    with open(os.path.join(out, "run_meta.json"), "w") as fh:
        json.dump(jsonable(meta), fh, indent=1)
    write_report(os.path.join(out, "report.txt"), R, checks, meta)
    print(f"\nreport: {os.path.join(out, 'report.txt')}  "
          f"({meta['elapsed_min']:.1f} min)")
    if "failed" in R:
        sys.exit(1)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
