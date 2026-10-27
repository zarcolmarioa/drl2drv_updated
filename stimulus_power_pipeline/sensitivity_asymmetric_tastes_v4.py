"""
sensitivity_asymmetric_tastes_v4.py

What happens to the registered false-claim rates when people WITHOUT the
remembered effect share a directional preference: a general lean toward
duller or foggier pictures that has nothing to do with an episode.

Why this script exists
----------------------
H1 (fog) and H2 (colour) were calibrated to at most 5% false claims
against random choosers and against personal tastes drawn from a normal
distribution CENTRED ON ZERO (classification_rates in
simulate_power_v4_n10_20.py). Zero-centred tastes are symmetric by
construction, and the registered rates were computed from symmetrised
per-person rates. A population that, on average, leans toward the
predicted image is outside that calibration (issue tracker items C04 and
S01). This script measures how far outside, without changing anything
registered: the thresholds, the (k, m) rule for each N and C1 are applied
unchanged.

What it simulates
-----------------
People with no memory effect, each with a personal taste on each feature
drawn from

    Normal(mu, spread x that feature's upper threshold),

mu in the predicted direction, on colour only, on fog only, or on both:

    mu      0, 0.1, 0.25, 0.5, 0.75, 1.0 x that feature's upper threshold
    spread  0.25, 0.5, 0.75, 1.0 x the upper threshold (registered spreads)
    lapse   0 and 0.1, one fixed rate for everyone, as in the rule sweep

The feature without a bias keeps the zero-centred taste. Each person
answers the 180 analysed trials, is fitted with the registered Firth
model and classified against the registered thresholds. Then, for every
N from 10 to 20, with that N's registered (k, m):

    P(H2 colour claimed), P(H1 fog claimed)   exactly, by enumeration
    P(either claim), P(both claims)           exactly, by enumerating the
                                              joint colour x fog counts

all conditional on the measured per-person rates, with the Monte Carlo
error of those rates carried through to the per-feature claim rates.

MIXTURES. A share pi of the null people biased, the rest symmetric:
because people are independent, a person's classification probabilities
are the pi-weighted average of the two, so the mixtures need no new
simulation.

THE BIAS AS BEHAVIOUR. Each mu is also stated as the probability of
choosing the predicted image on a typical informative trial of that
feature (as_probability at the frozen list's median informative
difference), and at a 1.0 SD difference.

THE C1 PART. The same bias (mu = 0.25, 0.5, 1.0 x threshold, on colour,
fog or both) in participants AND controls, no memory effect in either,
through the registered post-exclusion procedure of
simulate_c1_post_exclusion_v4.py: recruit until each group has its target
number of analysable people (3+ catch failures excluded), attention
check, C1 per feature with Holm, inconclusive if the check fired. Taste
spread 0.5x (as run_c1), strategy 0 and 0.25 in both groups, all seven
lapse pairs, 10/15/20 participants against 20/30 controls. mu = 0 is run
beside it as the reference. This shows whether C1 keeps its nominal
false-positive rate when both groups share a general preference.

Checks (each stops the run if it fails)
---------------------------------------
  - the registration files match the committed copies (SHA-256), and the
    power-run file used for standard errors matches the registration's
    record; the original power script is the committed one;
  - the frozen trial list matches its own checksums;
  - every copied function is verbatim (syntax tree) and gives the
    original's results, run now; every setting the copies read equals
    the original's;
  - this script's biased pool reduces exactly to the copied _c1_pool;
  - the exact enumeration of the joint counts gives the same per-feature
    claim rates as the copied rule_false_claim_exact;
  - REPLAY: the copied classification_rates, run with its own seed and
    the computed thresholds, reproduces every registered false_worst in
    combined_rule.csv (to within a person or two in 20,000, the only
    allowance, for floating-point differences between machines);
  - INDEPENDENT REPRODUCTION at mu = 0: this script's own code, new
    seed, computed thresholds, reproduces the registered per-person
    rates and false-claim rates within Monte Carlo error (criterion
    printed in the report before the results).

Which thresholds where: the reproduction checks use the thresholds the
power script computes, as the registered rates did; everything else uses
the rounded registered thresholds, which the analysis applies.

Inputs (relative to the project folder)
---------------------------------------
  simulate_power_v4_n10_20.py                  the original, at top level
  data/frozen_trial_list_v4/                   the sealed trial list
  data/registration_v4_2/combined_rule.csv     the registered (k, m)
  data/power_v4_n10_20/frozen/rule_sweep.csv   the registered SEs

Outputs, under data/sensitivity_asymmetric_tastes_v4/ (never
overwritten; a rerun goes to _2, _3, ...)
--------------------------------------------------------------------
  report.txt, run_meta.json
  tables/identity_check.csv          the copies against the original
  tables/validation_mu0.csv          the two reproduction checks
  tables/bias_as_probability.csv     each mu as a choice probability
  tables/per_person_rates.csv        P(predicted), P(opposite), per pool
  tables/joint_rates.csv             colour x fog classes, per pool
  tables/claim_rates.csv             P(claim) per target, mu, spread,
                                     lapse and N, beside the registered
  tables/mixture_claim_rates.csv     the same for mixtures
  tables/c1_bias_cells.csv           C1 under a shared bias, every cell
  tables/c1_bias_summary.csv         by bias and lapse pair
  figures/*.png

Usage
-----
    python sensitivity_asymmetric_tastes_v4.py

No arguments. Every setting is in SETTINGS below.
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
import scipy
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

COMMITTED_SHA256 = {
    "combined_rule.csv":
        "9e9e9a2a2b74c709792a0bc88767b20a9a0ed2f021fbf79fe419eb9382377a0a",
    "c1_cells.csv":
        "54f26c6bbe9ce3a29319b46520bbbb1e8c3c3895fdbc070fd67e6fca35ef1cb0",
    "c1_compact.csv":
        "5b812e39eaa7beb77aa6611dbb2e470631ee59162b8b8c05342e40c35c398598",
}
POWER_RUN_SHA256 = {
    "rule_sweep.csv":
        "0d5d03282c95492419f0db561c87035db4bd2d6d5185f78e53a8b9d5e3bcffd7",
}
POWER_SCRIPT_SHA256 = (
    "852f3b33eb349153fbff9f95c50e1cd3c4f84f68b204d79a23f020f471bd1730")

# --- Output ---
OUTPUT_ROOT = "data/sensitivity_asymmetric_tastes_v4"
MAKE_FIGURES = True
FIG_DPI = 150

# --- What is registered (stated so the run can check it; not changed) ---
REGISTERED_THRESHOLDS = {"colour": (0.286, -0.290), "fog": (0.488, -0.486)}
N_RANGE = list(range(10, 21))
CATCH_FAIL_EXCLUDE = 3
MIN_CONTROLS_C1 = 20

# --- The biased nulls (H1 and H2) ---
BIAS_TARGETS = ["colour", "fog", "both"]
MU_MULTS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]   # x that feature's upper
SPREADS = [0.25, 0.5, 0.75, 1.0]              # x the upper threshold
LAPSES = [0.0, 0.1]                           # fixed rate per person
PEOPLE_PER_POOL = 20_000                      # as N_TASTE_PEOPLE

# MIX_SHARES
#   The share pi of null people who carry the bias in the mixtures; the
#   rest have the zero-centred taste at the same spread.
MIX_SHARES = [0.1, 0.25, 0.5]

# --- The C1 part ---
BIAS_C1_MULTS = [0.25, 0.5, 1.0]              # mu = 0 is run as reference
BIAS_C1_TARGETS = ["colour", "fog", "both"]
BIAS_C1_STRATEGIES = [0.0, 0.25]
BIAS_C1_SPREAD = 0.5                          # = NONRESPONDER_SPREAD_MULT
BIAS_C1_PARTICIPANT_N = [10, 15, 20]
BIAS_C1_CONTROL_N = [20, 30]
BIAS_C1_STUDIES = 4000

# --- The reproduction criterion at mu = 0, fixed before the run ---
#   Every per-person rate (predicted and opposite, both features, the
#   four spreads, lapse 0) and every registered false-claim rate at
#   those spreads must lie within VAL_MAX_ABS_Z standard errors of the
#   registered value, the SE combining both runs' Monte Carlo error.
#   About 16 nearly independent comparisons: 3.3 is the two-sided
#   0.001 point, so a correct run fails by chance about 2% of the time.
VAL_MAX_ABS_Z = 3.3
# The replay may differ only by a person or two in 20,000 per rate.
REPLAY_PEOPLE_TOL = 2

# --- Seeds ---
SEED_REPRODUCTION = 20261201
SEED_POOLS = 20261202
SEED_C1_POOLS = 20261203
SEED_C1_DRAWS = 20261204
IDENTITY_SEED = 20261205

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
TASTE_SPREADS_IN_FLOORS = [0.25, 0.5, 0.75, 1.0]
NONRESPONDER_SPREAD_MULT = 0.5
N_TASTE_PEOPLE = 20_000
ALPHA = 0.05
FLOOR_ALPHA = 0.025
N_FLOOR_PEOPLE = 20_000
DETECTION_SIDED = "two"
RULE_MC_STEP = 1e-4
C1_STRATEGY_WEIGHT = 0.5
C1_LAPSE_PAIRS = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0),
                  (0.0, 0.1), (0.0, 0.2), (0.1, 0.1), (0.2, 0.2)]
C1_EXACT_MAX_N = 20
C1_POOL_PEOPLE = 20_000
C1_HOLM = True
C1_ZERO_MIN_N = 4
GATE_ALPHA = 0.05
N_CATCH_TRIALS = 10
CATCH_P_FAIL_PER_LAPSE = 0.75
LAPSE_BETWEEN_SD = 0.5
RT_GATE = True
RT_FAST_GIVEN_LAPSE = 0.4
RT_FAST_GIVEN_ENGAGED = 0.005
RANDOM_SEED = 42

COPIED_SETTINGS = [
    "UNHASHED", "FROZEN_SEED", "PRED_SAT", "PRED_FOG", "SD_SAT", "SD_FOG",
    "PRED_REBUILD_TOL", "ESTIMATOR", "FIRTH_ITER", "FIRTH_TOL",
    "FIRTH_MAX_STEP", "CAP_INFORMATIVE_SD", "TASTE_SPREADS_IN_FLOORS",
    "NONRESPONDER_SPREAD_MULT", "N_TASTE_PEOPLE", "ALPHA", "FLOOR_ALPHA",
    "N_FLOOR_PEOPLE", "DETECTION_SIDED", "RULE_MC_STEP",
    "C1_STRATEGY_WEIGHT", "C1_LAPSE_PAIRS", "C1_EXACT_MAX_N",
    "C1_POOL_PEOPLE", "C1_HOLM", "C1_ZERO_MIN_N", "GATE_ALPHA",
    "N_CATCH_TRIALS", "CATCH_P_FAIL_PER_LAPSE", "LAPSE_BETWEEN_SD",
    "RT_GATE", "RT_FAST_GIVEN_LAPSE", "RT_FAST_GIVEN_ENGAGED",
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

# --- COPY of simulate_power_v4_n10_20.py L2254-L2270 (lapse_p), verbatim ---
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
# --- end of copy (lapse_p) ---

# --- COPY of simulate_power_v4_n10_20.py L2273-L2276 (simulate_lapsing), verbatim ---
def simulate_lapsing(X, W, lapse, rng):
    """Choices of many people at once, each lapsing at rate `lapse`."""
    p = lapse_p(X, W, lapse)
    return (rng.random(p.shape) < p).astype(float)
# --- end of copy (simulate_lapsing) ---

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

# --- COPY of simulate_power_v4_n10_20.py L3128-L3221 (classification_rates), verbatim ---
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
# --- end of copy (classification_rates) ---

# --- COPY of simulate_power_v4_n10_20.py L3224-L3258 (trinomial_grid), verbatim ---
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
# --- end of copy (trinomial_grid) ---

# --- COPY of simulate_power_v4_n10_20.py L3297-L3301 (rule_mask_count), verbatim ---
def rule_mask_count(n, k, m):
    """Boolean (a, b) mask: at least k predicted, at most m opposite."""
    a = np.arange(n + 1)[:, None]
    b = np.arange(n + 1)[None, :]
    return (a >= k) & (b <= m)
# --- end of copy (rule_mask_count) ---

# --- COPY of simulate_power_v4_n10_20.py L3318-L3356 (rule_false_claim_exact), verbatim ---
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
# --- end of copy (rule_false_claim_exact) ---

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


# ============================================================
# THIS SCRIPT'S OWN CODE (not copied)
# ============================================================

COPIED_FUNCTIONS = [
    "fit_logit_batch", "fit_firth_batch", "fit_batch", "simulate_many",
    "lapse_p", "simulate_lapsing", "sha256_of", "verify_frozen",
    "load_trials", "informative_medians", "as_probability",
    "compute_floors", "classify", "classification_rates",
    "trinomial_grid", "rule_mask_count", "rule_false_claim_exact",
    "mw_exact_sf", "mw_p_batch", "holm_two", "_c1_pool", "_c1_gate",
    "_c1_draw", "_c1_test", "_c1_zero_catch", "draw_attention",
]

FEATURES = ("colour", "fog")
CLAIM = {"colour": "H2", "fog": "H1"}
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
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    return x


def settings_snapshot():
    out = {}
    for k, v in globals().items():
        if k.isupper() and not callable(v) and not k.startswith("_"):
            try:
                json.dumps(jsonable(v))
                out[k] = jsonable(v)
            except TypeError:
                out[k] = repr(v)
    return out


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
            if isinstance(x, ast.FunctionDef) and x is not n:
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
    structure, shapes or NaN patterns differ."""
    if isinstance(a, pd.DataFrame):
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            return np.inf
        return max((_cmp(a[c].to_numpy(), b[c].to_numpy())
                    for c in a.columns), default=0.0)
    if isinstance(a, dict):
        if set(a) != set(b):
            return np.inf
        return max((_cmp(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return np.inf
        return max((_cmp(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, str):
        return 0.0 if a == b else np.inf
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
        return 0.0 if np.array_equal(a.astype(str), b.astype(str)) \
            else np.inf
    a, b = a.astype(float), b.astype(float)
    if a.shape != b.shape:
        return np.inf
    na, nb = np.isnan(a), np.isnan(b)
    if (na != nb).any():
        return np.inf
    if a.size == 0 or na.all():
        return 0.0
    return float(np.max(np.abs(a[~na] - b[~nb])))


# ------------------------------------------------------------
# This script's own model pieces
# ------------------------------------------------------------

def c1_pool_biased(X, mu, strategy_rate, strategy_w, taste_sd, lapse_mean,
                   rng, n=None):
    """_c1_pool with a shared mean preference `mu` (one value per
    feature) added to every person, and no memory effect.

    THIS SCRIPT'S OWN CODE, written to consume the random stream exactly
    as _c1_pool does: taste, strategy, responder draw (nobody responds),
    personal lapse rates, responses, attention. With mu = (0, 0) it is
    _c1_pool with no effect, and with mu on one feature it is _c1_pool
    with that effect in every person; the start-up check requires both,
    bit for bit.
    """
    n = C1_POOL_PEOPLE if n is None else n
    W = rng.normal(0.0, 1.0, size=(n, 2)) * taste_sd
    use = rng.random(n) < strategy_rate
    W[use, 0] += strategy_w
    W[use, 1] += strategy_w
    is_resp = rng.random(n) < 0.0
    W[:, 0] += mu[0]
    W[:, 1] += mu[1]
    if lapse_mean <= 0:
        lam_i = np.zeros(n)
    else:
        shape = 1.0 / max(LAPSE_BETWEEN_SD, 1e-6) ** 2
        lam_i = rng.gamma(shape, lapse_mean / shape, size=n)
        lam_i = np.clip(lam_i, 0.0, 0.9)
    p = 1.0 / (1.0 + np.exp(-np.clip(W @ X.T, -500.0, 500.0)))
    p = (1.0 - lam_i[:, None]) * p + lam_i[:, None] / 2.0
    Y = (rng.random(p.shape) < p).astype(float)
    est = fit_batch(X, Y)
    catch = rng.binomial(N_CATCH_TRIALS,
                         np.clip(CATCH_P_FAIL_PER_LAPSE * lam_i, 0.0, 1.0))
    p_fast = (lam_i * RT_FAST_GIVEN_LAPSE
              + (1.0 - lam_i) * RT_FAST_GIVEN_ENGAGED)
    fast = rng.binomial(X.shape[0], np.clip(p_fast, 0.0, 1.0)) \
        / X.shape[0]
    return {"w": est, "lapse": lam_i, "catch": catch.astype(float),
            "fast": fast, "responder": is_resp}


def null_people(X, th, sd, mu, lapse, rng, n=None):
    """People with no memory effect: taste Normal(mu, sd) per feature,
    one fixed lapse rate. Returns each person's class per feature:
    0 not detected, 1 predicted, 2 opposite."""
    n = PEOPLE_PER_POOL if n is None else n
    W = rng.normal(0.0, 1.0, size=(n, 2)) * sd + mu
    est = fit_batch(X, simulate_lapsing(X, W, lapse, rng))
    pred, opp = classify(est, th)
    return np.where(pred, 1, np.where(opp, 2, 0)).astype(np.int8)


def rates_from_codes(codes):
    """Per-feature (predicted, opposite) and the 3 x 3 joint table."""
    per = {f: (float((codes[:, k] == 1).mean()),
               float((codes[:, k] == 2).mean()))
           for k, f in enumerate(FEATURES)}
    joint = np.zeros((3, 3))
    np.add.at(joint, (codes[:, 0], codes[:, 1]), 1.0)
    return per, joint / len(codes)


def joint_claims(joint, rules):
    """P(H2), P(H1), P(either), P(both) for every N in N_RANGE, exactly.

    The joint (colour predicted, colour opposite, fog predicted, fog
    opposite) count distribution of a sample, built one person at a time
    from the 3 x 3 joint classification table: each person adds one to
    the count their pair of classes falls in. The registered (k, m) for
    each feature is then a mask over it.
    """
    nmax = max(N_RANGE)
    A = np.zeros((nmax + 1,) * 4)
    A[0, 0, 0, 0] = 1.0
    out = {}

    def sl(cls):   # (axis of 'predicted', axis of 'opposite') shift
        return {0: (0, 0), 1: (1, 0), 2: (0, 1)}[cls]

    for step in range(1, nmax + 1):
        B = np.zeros_like(A)
        for cc in range(3):
            for cf in range(3):
                p = joint[cc, cf]
                if p <= 0:
                    continue
                s = sl(cc) + sl(cf)
                src = tuple(slice(0, nmax + 1 - d) for d in s)
                dst = tuple(slice(d, nmax + 1) for d in s)
                B[dst] += p * A[src]
        A = B
        if step in N_RANGE:
            masks = {}
            for f in FEATURES:
                k, m = rules[(f, step)]
                mk = np.zeros((nmax + 1, nmax + 1), dtype=bool)
                mk[:step + 1, :step + 1] = rule_mask_count(step, k, m)
                masks[f] = mk
            Mc = masks["colour"][:, :, None, None]
            Mf = masks["fog"][None, None, :, :]
            out[step] = {"H2": float((A * Mc).sum()),
                         "H1": float((A * Mf).sum()),
                         "either": float((A * (Mc | Mf)).sum()),
                         "both": float((A * (Mc & Mf)).sum())}
    return out


def claim_se(n, k, m, pp, po, n_people):
    """Monte Carlo SE of an exact claim rate, from the SE of the two
    per-person rates it is computed from (delta method; the two rates
    are multinomial, so negatively correlated)."""
    h = RULE_MC_STEP
    f0 = rule_false_claim_exact(n, k, m, pp, po)
    g1 = (rule_false_claim_exact(n, k, m, pp + h, po) - f0) / h
    g2 = (rule_false_claim_exact(n, k, m, pp, min(po + h, 1 - pp)) - f0) / h
    v = (g1 * g1 * pp * (1 - pp) + g2 * g2 * po * (1 - po)
         - 2 * g1 * g2 * pp * po) / n_people
    return float(np.sqrt(max(v, 0.0)))


def mu_vector(target, mult, U):
    mu = np.zeros(2)
    if target in ("colour", "both"):
        mu[0] = mult * U[0]
    if target in ("fog", "both"):
        mu[1] = mult * U[1]
    return mu


# ------------------------------------------------------------
# Start-up: the copied functions against the original
# ------------------------------------------------------------

def identity_check(X, th_c, checks, tdir):
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

    rows = []
    me = sys.modules[__name__]

    def both(label, f):
        d = _cmp(f(me), f(pw))
        rows.append({"quantity": label, "max_abs_diff": d,
                     "ok": bool(d <= IDENTITY_TOL)})

    rng = np.random.default_rng(IDENTITY_SEED)
    W = np.column_stack([rng.normal(0, 0.6, 400), rng.normal(0, 0.9, 400)])
    W[:5] = [[0, 0], [3.0, 3.0], [-2.0, 0.5], [0.29, 0.49], [8.0, -8.0]]
    p = 1.0 / (1.0 + np.exp(-(W @ X.T)))
    Y = (rng.random(p.shape) < p).astype(float)
    both("fit_firth_batch", lambda M: M.fit_firth_batch(X, Y))
    both("fit_batch", lambda M: M.fit_batch(X, Y))
    both("fit_logit_batch", lambda M: M.fit_logit_batch(X, Y[:50]))
    both("simulate_many",
         lambda M: M.simulate_many(X, W, np.random.default_rng(1)))
    both("lapse_p", lambda M: M.lapse_p(X, W, 0.1))
    both("simulate_lapsing",
         lambda M: M.simulate_lapsing(X, W, 0.1, np.random.default_rng(2)))
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        th_t = pw.compute_floors(X, "frozen", [])
        rates_t = pw.classification_rates(X, th_t, [])
        rates_m = classification_rates(X, th_c, [])
    rows.append({"quantity": "compute_floors (upper, lower, sd, nulls)",
                 "max_abs_diff": _cmp(th_c, th_t),
                 "ok": _cmp(th_c, th_t) <= IDENTITY_TOL})
    rows.append({"quantity": "classification_rates (every null, both "
                             "features)",
                 "max_abs_diff": _cmp(rates_m, rates_t),
                 "ok": _cmp(rates_m, rates_t) <= IDENTITY_TOL})
    est = fit_batch(X, Y)
    both("classify", lambda M: M.classify(est, th_c))
    both("informative_medians", lambda M: M.informative_medians(X))
    both("as_probability", lambda M: [M.as_probability(w, d)
                                      for w in (0.1, 0.3, 0.5)
                                      for d in (0.6, 1.0)])
    both("trinomial_grid", lambda M: M.trinomial_grid(15, 0.18, 0.02))
    both("rule_mask_count", lambda M: M.rule_mask_count(15, 6, 3))
    both("rule_false_claim_exact",
         lambda M: [M.rule_false_claim_exact(n, k, m, a, b)
                    for n, k, m in ((10, 4, 0), (15, 6, 3), (20, 5, 1))
                    for a, b in ((0.18, 0.18), (0.2, 0.02), (0.05, 0.3))])
    both("_c1_pool", lambda M: M._c1_pool(
        X, th_c, 0, 0.3, 0.5, 0.25, 0.5, th_c["upper"] * 0.5, 0.1,
        np.random.default_rng(IDENTITY_SEED + 1), n=3000))
    pm = _c1_pool(X, th_c, 0, 0.3, 0.5, 0.25, 0.5, th_c["upper"] * 0.5,
                  0.2, np.random.default_rng(IDENTITY_SEED + 2), n=3000)
    both("draw_attention", lambda M: M.draw_attention(
        500, 0.15, 180, np.random.default_rng(IDENTITY_SEED + 3)))
    for size, lab in ((6, "exact null"), (25, "normal approximation")):
        gp = _c1_draw(pm, 300, size, np.random.default_rng(size))
        gc = _c1_draw(pm, 300, size + 3, np.random.default_rng(size + 1))
        both(f"_c1_draw ({size})", lambda M: M._c1_draw(
            pm, 300, size, np.random.default_rng(size)))
        both(f"mw_p_batch ({lab})", lambda M: M.mw_p_batch(
            gp["w"][:, :, 0], gc["w"][:, :, 0]))
        both(f"mw_p_batch, ties ({lab})", lambda M: M.mw_p_batch(
            gc["catch"], gp["catch"], exact_max_n=0))
        both(f"_c1_test ({lab})", lambda M: M._c1_test(gp["w"], gc["w"]))
        both(f"_c1_gate ({lab})", lambda M: M._c1_gate(
            gp["catch"], gc["catch"], gp["fast"], gc["fast"]))
        both(f"_c1_zero_catch ({lab})", lambda M: M._c1_zero_catch(gp, gc))
    both("mw_exact_sf", lambda M: M.mw_exact_sf(7, 9))
    pv = np.random.default_rng(3).random((200, 2)) * 0.08
    both("holm_two", lambda M: [M.holm_two(a, b) for a, b in pv])
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        f_m = verify_frozen(FROZEN_ROOT, [])
        f_t = pw.verify_frozen(FROZEN_ROOT, [])
        _, X2, _ = pw.load_trials(
            os.path.join(FROZEN_ROOT, FROZEN_TRIAL_LIST), "frozen")
    rows.append({"quantity": "verify_frozen", "max_abs_diff":
                 0.0 if f_m == f_t else np.inf, "ok": f_m == f_t})
    rows.append({"quantity": "load_trials", "max_abs_diff": _cmp(X, X2),
                 "ok": _cmp(X, X2) <= IDENTITY_TOL})

    # This script's own pieces, against the copies they must reduce to.
    sd = th_c["upper"] * 0.5
    for label, mu, args in (
            ("c1_pool_biased, mu = 0  ==  _c1_pool, no effect",
             (0.0, 0.0), (0, 0.0, 0.5)),
            ("c1_pool_biased, mu on colour  ==  _c1_pool, everyone "
             "responding on colour", (0.143, 0.0), (0, 0.143, 1.0)),
            ("c1_pool_biased, mu on fog  ==  _c1_pool, everyone "
             "responding on fog", (0.0, 0.244), (1, 0.244, 1.0))):
        a = c1_pool_biased(X, np.array(mu), 0.25, 0.5, sd, 0.1,
                           np.random.default_rng(IDENTITY_SEED + 4),
                           n=3000)
        b = _c1_pool(X, th_c, args[0], args[1], args[2], 0.25, 0.5, sd,
                     0.1, np.random.default_rng(IDENTITY_SEED + 4),
                     n=3000)
        a = {k: v for k, v in a.items() if k != "responder"}
        b = {k: v for k, v in b.items() if k != "responder"}
        d = _cmp(a, b)
        rows.append({"quantity": label, "max_abs_diff": d,
                     "ok": d <= IDENTITY_TOL})
    rules = {(f, n): (int(k), int(m)) for f, n, k, m in
             ((f, n, 4 + (n > 12), (n > 15)) for f in FEATURES
              for n in N_RANGE)}
    J = np.array([[0.60, 0.10, 0.08], [0.09, 0.03, 0.01],
                  [0.05, 0.02, 0.02]])
    jc = joint_claims(J, rules)
    worst = 0.0
    for n in N_RANGE:
        for f, key, idx in (("colour", "H2", 0), ("fog", "H1", 1)):
            marg = J.sum(axis=1 - idx)
            k, m = rules[(f, n)]
            ex = rule_false_claim_exact(n, k, m, marg[1], marg[2])
            worst = max(worst, abs(ex - jc[n][key]))
    rows.append({"quantity": "joint enumeration: per-feature margins == "
                             "rule_false_claim_exact",
                 "max_abs_diff": worst, "ok": worst <= 1e-10})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(tdir, "identity_check.csv"), index=False)
    bad = tab[~tab.ok.astype(bool)]
    checks.require("copied functions give the original's results; this "
                   "script's own pieces reduce to them", bad.empty,
                   f"{len(tab) - len(bad)}/{len(tab)} identical"
                   + (f"; differ: {', '.join(bad.quantity)}"
                      if len(bad) else ""))
    return rates_m


# ------------------------------------------------------------
# Validation at mu = 0
# ------------------------------------------------------------

def symmetrised(rates):
    """The rule sweep's per-person rate per null: (pred + opp) / 2."""
    out = {}
    for _, r in rates.iterrows():
        key = ("guesser" if r.null == "guesser"
               else f"taste_{r.spread_in_floors:g}")
        out[(key, r.feature)] = 0.5 * (float(r.p_predicted)
                                       + float(r.p_opposite))
    return out


def validate(X, th_c, replay_rates, rule_tab, sweep, checks, tdir):
    banner("VALIDATION AT mu = 0")
    sym = symmetrised(replay_rates)
    rows = []
    # --- (a) the replay against combined_rule.csv and rule_sweep.csv ---
    for _, r in rule_tab.iterrows():
        f, n, k, m = r.feature, int(r.n), int(r.k), int(r.m)
        sw = sweep[(sweep.feature == f) & (sweep.n == n) & (sweep.k == k)
                   & (sweep.m == m) & (sweep.family == r.family)].iloc[0]
        for null, col_reg in (("guesser", "false_guesser"),
                              ("taste_0.25", "false_taste_0.25"),
                              ("taste_0.5", "false_taste_0.5"),
                              ("taste_0.75", "false_taste_0.75"),
                              ("taste_1", "false_taste_1")):
            p = sym[(null, f)]
            mine = rule_false_claim_exact(n, k, m, p, p)
            h = 1.0 / N_TASTE_PEOPLE
            deriv = abs(rule_false_claim_exact(n, k, m, p + h, p + h)
                        - mine) / h
            tol = deriv * REPLAY_PEOPLE_TOL / N_TASTE_PEOPLE + 1e-12
            reg = float(sw[col_reg])
            rows.append({"check": "replay", "feature": f, "n": n,
                         "rule": f"k>={k}, m<={m}", "null": null,
                         "registered": reg, "this_run": mine,
                         "diff": mine - reg, "tolerance": tol, "z": np.nan,
                         "ok": abs(mine - reg) <= tol})
        rows.append({"check": "replay vs combined_rule false_worst",
                     "feature": f, "n": n, "rule": f"k>={k}, m<={m}",
                     "null": "taste_1", "registered": float(r.false_worst),
                     "this_run": rule_false_claim_exact(
                         n, k, m, sym[("taste_1", f)],
                         sym[("taste_1", f)]),
                     "diff": np.nan, "tolerance": np.nan, "z": np.nan,
                     "ok": True})
        last = rows[-1]
        last["diff"] = last["this_run"] - last["registered"]
        tol = rows[-2]["tolerance"]
        last["tolerance"] = tol
        last["ok"] = abs(last["diff"]) <= tol
    a = pd.DataFrame(rows)
    checks.require("replay: classification_rates reproduces every "
                   "registered false-claim rate (combined_rule false_worst "
                   "and rule_sweep, all five nulls)", a.ok.all(),
                   f"{int(a.ok.sum())}/{len(a)}; worst |diff| "
                   f"{a['diff'].abs().max():.2e}")

    # --- (b) this script's own code, new seed, computed thresholds ---
    rng = np.random.default_rng(SEED_REPRODUCTION)
    rows = []
    own = {}
    for mult in [0.0] + list(TASTE_SPREADS_IN_FLOORS):
        sd = th_c["upper"] * mult
        codes = null_people(X, th_c, sd, np.zeros(2), 0.0, rng,
                            n=PEOPLE_PER_POOL)
        per, _ = rates_from_codes(codes)
        key = "guesser" if mult == 0 else f"taste_{mult:g}"
        own[key] = per
        for f in FEATURES:
            p_reg = sym[(key, f)]
            v_reg = max(p_reg * (1 - 2 * p_reg), 0) / (2 * N_TASTE_PEOPLE)
            for d, p_new in zip(("predicted", "opposite"), per[f]):
                v_new = p_new * (1 - p_new) / PEOPLE_PER_POOL
                rows.append({"check": "per-person rate", "feature": f,
                             "null": key, "direction": d,
                             "registered": p_reg, "this_run": p_new,
                             "diff": p_new - p_reg,
                             "z": (p_new - p_reg) / np.sqrt(v_reg + v_new)})
    for _, r in rule_tab.iterrows():
        f, n, k, m = r.feature, int(r.n), int(r.k), int(r.m)
        sw = sweep[(sweep.feature == f) & (sweep.n == n) & (sweep.k == k)
                   & (sweep.m == m) & (sweep.family == r.family)].iloc[0]
        for key in ["guesser"] + [f"taste_{s:g}"
                                  for s in TASTE_SPREADS_IN_FLOORS]:
            col = "false_guesser" if key == "guesser" else f"false_{key}"
            pp, po = own[key][f]
            mine = rule_false_claim_exact(n, k, m, pp, po)
            se_new = claim_se(n, k, m, pp, po, PEOPLE_PER_POOL)
            se_reg = float(sw[col + "_se"])
            reg = float(sw[col])
            se = np.sqrt(se_new ** 2 + se_reg ** 2)
            rows.append({"check": "false-claim rate", "feature": f,
                         "null": key, "n": n, "rule": f"k>={k}, m<={m}",
                         "registered": reg, "this_run": mine,
                         "diff": mine - reg,
                         "z": (mine - reg) / se if se > 0 else 0.0})
    b = pd.DataFrame(rows)
    worst = float(b.z.abs().max())
    checks.require(
        "independent reproduction at mu = 0: per-person and false-claim "
        f"rates within {VAL_MAX_ABS_Z} SE of the registered values",
        worst <= VAL_MAX_ABS_Z,
        f"{len(b)} comparisons, max |z| {worst:.2f}")
    tab = pd.concat([a, b], ignore_index=True)
    tab.to_csv(os.path.join(tdir, "validation_mu0.csv"), index=False)
    return tab, own


# ------------------------------------------------------------
# The H1 / H2 grid
# ------------------------------------------------------------

def run_rule_grid(X, th, rules, reg_false, med):
    """Every pool: per-person rates, the joint table, the claim rates."""
    U = th["upper"]
    rng = np.random.default_rng(SEED_POOLS)
    pools = []
    for spread in SPREADS:
        for lapse in LAPSES:
            pools.append(("none", 0.0, spread, lapse))
            for target in BIAS_TARGETS:
                for mult in MU_MULTS:
                    if mult > 0:
                        pools.append((target, mult, spread, lapse))
    per_rows, joint_rows, claim_rows, joints = [], [], [], {}
    t0 = time.time()
    for i, (target, mult, spread, lapse) in enumerate(pools):
        mu = mu_vector(target, mult, U)
        codes = null_people(X, th, U * spread, mu, lapse, rng)
        per, J = rates_from_codes(codes)
        joints[(target, mult, spread, lapse)] = J
        lab = {"target": target, "mu_mult": mult,
               "mu_colour": mu[0], "mu_fog": mu[1], "spread": spread,
               "lapse": lapse}
        for k, f in enumerate(FEATURES):
            pp, po = per[f]
            per_rows.append({**lab, "feature": f,
                             "feature_biased": bool(mu[k] > 0),
                             "p_predicted": pp, "p_opposite": po,
                             "se_predicted": np.sqrt(
                                 pp * (1 - pp) / PEOPLE_PER_POOL),
                             "se_opposite": np.sqrt(
                                 po * (1 - po) / PEOPLE_PER_POOL),
                             "p_choice_typical": as_probability(
                                 mu[k], med[f])})
        jr = dict(lab)
        for a_, ca in enumerate(("none", "pred", "opp")):
            for b_, cb in enumerate(("none", "pred", "opp")):
                jr[f"colour_{ca}__fog_{cb}"] = J[a_, b_]
        joint_rows.append(jr)
        claim_rows += claim_rows_for(lab, per, J, rules, reg_false)
        if i in (0, 9) or (i + 1) % 32 == 0 or i + 1 == len(pools):
            el = time.time() - t0
            print(f"    {i + 1}/{len(pools)} pools, {el:.0f}s, about "
                  f"{el / (i + 1) * (len(pools) - i - 1) / 60:.1f} min "
                  "left", flush=True)
    return (pd.DataFrame(per_rows), pd.DataFrame(joint_rows),
            pd.DataFrame(claim_rows), joints)


def claim_rows_for(lab, per, J, rules, reg_false):
    jc = joint_claims(J, rules)
    rows = []
    for n in N_RANGE:
        r = dict(lab)
        r["n"] = n
        for f in FEATURES:
            k, m = rules[(f, n)]
            pp, po = per[f]
            ex = rule_false_claim_exact(n, k, m, pp, po)
            c = CLAIM[f]
            r.update({f"{c}_rule": f"k>={k}, m<={m}",
                      f"p_{c}_claim": ex,
                      f"se_{c}_claim": claim_se(n, k, m, pp, po,
                                                PEOPLE_PER_POOL),
                      f"registered_{c}_false_worst": reg_false[(f, n)]})
            r[f"{c}_upper"] = r[f"p_{c}_claim"] + 1.96 * r[f"se_{c}_claim"]
        r.update({"p_either_claim": jc[n]["either"],
                  "p_both_claims": jc[n]["both"],
                  "enum_vs_exact_H2": jc[n]["H2"] - r["p_H2_claim"],
                  "enum_vs_exact_H1": jc[n]["H1"] - r["p_H1_claim"]})
        rows.append(r)
    return rows


def mixtures(joints, rules, reg_false):
    rows = []
    for (target, mult, spread, lapse), J in joints.items():
        if target == "none":
            continue
        J0 = joints[("none", 0.0, spread, lapse)]
        for pi in MIX_SHARES:
            Jm = pi * J + (1 - pi) * J0
            per = {"colour": (Jm[1, :].sum(), Jm[2, :].sum()),
                   "fog": (Jm[:, 1].sum(), Jm[:, 2].sum())}
            lab = {"target": target, "mu_mult": mult, "share_biased": pi,
                   "spread": spread, "lapse": lapse}
            for r in claim_rows_for(lab, per, Jm, rules, reg_false):
                for f in FEATURES:
                    r[f"p_{f}_predicted"], r[f"p_{f}_opposite"] = per[f]
                rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# The C1 part: the post-exclusion procedure under a shared bias
# (draw_group and evaluate are the same code as in
# simulate_c1_post_exclusion_v4.py)
# ------------------------------------------------------------

def draw_group(pool, studies, size, rng, exclude):
    """`studies` analysable groups of `size` people. With the exclusion,
    candidates are drawn in order and anyone with CATCH_FAIL_EXCLUDE or
    more catch failures is excluded until `size` are analysable."""
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
    """The registered steps on the analysable groups: attention check,
    C1 per feature with Holm; descriptive below MIN_CONTROLS_C1 controls,
    otherwise inconclusive if the check fired; the zero-catch test."""
    r_c, r_f = _c1_test(gp["w"], gc["w"])
    fired, f_catch, f_rt = _c1_gate(gp["catch"], gc["catch"],
                                    gp["fast"], gc["fast"])
    descriptive = n_c < MIN_CONTROLS_C1
    live = ~fired & (not descriptive)
    pos_c, pos_f = r_c & live, r_f & live
    z = _c1_zero_catch(gp, gc)
    u = z["usable"]
    either = z["reject_colour"] | z["reject_fog"]
    return {
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
        "descriptive": float(descriptive),
        "zero_usable": float(u.mean()),
        "zero_power_either": (float(either[u].mean()) if u.any()
                              else np.nan),
        "median_w_colour_participants": float(np.median(gp["w"][:, :, 0])),
        "median_w_fog_participants": float(np.median(gp["w"][:, :, 1])),
        "median_w_colour_controls": float(np.median(gc["w"][:, :, 0])),
        "median_w_fog_controls": float(np.median(gc["w"][:, :, 1]))}


def run_c1_bias(X, th, med):
    U = th["upper"]
    taste_sd = U * BIAS_C1_SPREAD
    lam_vals = sorted({v for pr in C1_LAPSE_PAIRS for v in pr})
    settings = [("none", 0.0)] + [(t, m) for t in BIAS_C1_TARGETS
                                  for m in BIAS_C1_MULTS]
    rng = np.random.default_rng(SEED_C1_POOLS)
    pools = {}
    t0 = time.time()
    for target, mult in settings:
        mu = mu_vector(target, mult, U)
        for strat in BIAS_C1_STRATEGIES:
            for lam in lam_vals:
                pools[(target, mult, strat, lam)] = c1_pool_biased(
                    X, mu, strat, C1_STRATEGY_WEIGHT, taste_sd, lam, rng)
    print(f"    {len(pools)} pools, {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(SEED_C1_DRAWS)
    rows = []
    cells = [(t, m, s, lp, lc, n_p, n_c) for t, m in settings
             for s in BIAS_C1_STRATEGIES for lp, lc in C1_LAPSE_PAIRS
             for n_p in BIAS_C1_PARTICIPANT_N for n_c in BIAS_C1_CONTROL_N]
    t0 = time.time()
    for i, (t, m, s, lp, lc, n_p, n_c) in enumerate(cells):
        gp, _, ep = draw_group(pools[(t, m, s, lp)], BIAS_C1_STUDIES, n_p,
                               rng, True)
        gc, _, ec = draw_group(pools[(t, m, s, lc)], BIAS_C1_STUDIES, n_c,
                               rng, True)
        r = evaluate(gp, gc, n_c)
        mu = mu_vector(t, m, U)
        r.update({"target": t, "mu_mult": m, "mu_colour": mu[0],
                  "mu_fog": mu[1],
                  "p_choice_typical_colour": as_probability(
                      mu[0], med["colour"]),
                  "p_choice_typical_fog": as_probability(mu[1], med["fog"]),
                  "strategy_rate": s, "lapse_participants": lp,
                  "lapse_controls": lc, "n_participants": n_p,
                  "n_controls": n_c,
                  "excluded_participants_mean": float(ep.mean()),
                  "excluded_controls_mean": float(ec.mean()),
                  "studies": BIAS_C1_STUDIES})
        rows.append(r)
        if i in (19,) or (i + 1) % 200 == 0 or i + 1 == len(cells):
            el = time.time() - t0
            print(f"    {i + 1}/{len(cells)} C1 cells, {el:.0f}s, about "
                  f"{el / (i + 1) * (len(cells) - i - 1) / 60:.1f} min "
                  "left", flush=True)
    return pd.DataFrame(rows)


def c1_summary(c1):
    def kind(r):
        if r.lapse_participants == r.lapse_controls:
            return "matched"
        return ("controls lapse more"
                if r.lapse_controls > r.lapse_participants
                else "participants lapse more")
    c1 = c1.copy()
    c1["lapse_kind"] = c1.apply(kind, axis=1)
    g = c1.groupby(["target", "mu_mult", "lapse_participants",
                    "lapse_controls", "lapse_kind"])
    out = g.agg(fp_either_mean=("power_either_gated", "mean"),
                fp_either_worst=("power_either_gated", "max"),
                fp_colour_worst=("power_colour_gated", "max"),
                fp_fog_worst=("power_fog_gated", "max"),
                ungated_either_worst=("power_either_raw", "max"),
                gate_fires_mean=("gate_fires", "mean"),
                gate_fires_min=("gate_fires", "min"),
                zero_catch_worst=("zero_power_either", "max"),
                cells=("power_either_gated", "size")).reset_index()
    ref = out[out.target == "none"].set_index(
        ["lapse_participants", "lapse_controls"])
    for c in ("fp_either_mean", "fp_either_worst", "gate_fires_mean"):
        out[c + "_minus_mu0"] = out[c] - ref[c].reindex(
            pd.MultiIndex.from_frame(out[["lapse_participants",
                                          "lapse_controls"]])).to_numpy()
    order = {t: i for i, t in enumerate(["none"] + BIAS_C1_TARGETS)}
    out["_o"] = out.target.map(order)
    return (out.sort_values(["_o", "mu_mult", "lapse_participants",
                             "lapse_controls"]).drop(columns="_o")
            .reset_index(drop=True))


# ------------------------------------------------------------
# Figures
# ------------------------------------------------------------

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)


def _mu_ticks(ax, feat, med, U):
    ax.set_xticks(MU_MULTS)
    if feat is None:
        ax.set_xticklabels([f"{m:g}" for m in MU_MULTS], fontsize=7)
    else:
        k = FEATURES.index(feat)
        ax.set_xticklabels([f"{m:g}\n{as_probability(m * U[k], med[feat]):.3f}"
                            for m in MU_MULTS], fontsize=7)


def fig_claims(claims, med, U, path, lapse=0.0):
    ns = [10, 15, 20]
    panels = [("colour", "p_H2_claim", "H2 colour claimed\n(bias on colour)",
               "colour"),
              ("fog", "p_H1_claim", "H1 fog claimed\n(bias on fog)", "fog"),
              ("both", "p_either_claim", "either claim\n(bias on both)",
               None)]
    fig, axes = plt.subplots(len(ns), 3, figsize=(12, 9), sharey="row")
    for i, n in enumerate(ns):
        for j, (target, col, title, feat) in enumerate(panels):
            ax = axes[i, j]
            c = claims[(claims.n == n) & (claims.lapse == lapse)]
            for s_i, spread in enumerate(SPREADS):
                s = c[(c.spread == spread)
                      & ((c.target == target) | (c.target == "none"))]
                s = s.sort_values("mu_mult")
                ax.plot(s.mu_mult, s[col], color=SERIES_COLOURS[s_i],
                        marker=SERIES_MARKERS[s_i], ls=SERIES_STYLES[s_i],
                        ms=4, label=f"spread {spread:g}x")
            ax.axhline(0.05, color="#555555", lw=0.8, ls=":")
            _mu_ticks(ax, feat, med, U)
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"N = {n}\nP(claim), no memory effect")
            _style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("mu, x upper threshold\n(choice prob. on a typical "
                      "trial)", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Registered rules under a shared bias, lapse {lapse:g}. "
                 "Dotted: 0.05.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_per_person(per, med, U, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, feat in zip(axes, FEATURES):
        d = per[(per.feature == feat) & (per.lapse == 0.0)
                & ((per.target == feat) | (per.target == "none"))]
        for s_i, spread in enumerate(SPREADS):
            s = d[d.spread == spread].sort_values("mu_mult")
            ax.plot(s.mu_mult, s.p_predicted, color=SERIES_COLOURS[s_i],
                    marker=SERIES_MARKERS[s_i], ls="-", ms=4,
                    label=f"spread {spread:g}x: predicted")
            ax.plot(s.mu_mult, s.p_opposite, color=SERIES_COLOURS[s_i],
                    marker=SERIES_MARKERS[s_i], ls=":", ms=4,
                    mfc="white", label=f"spread {spread:g}x: opposite")
        _mu_ticks(ax, feat, med, U)
        ax.set_title(f"{feat}: bias on {feat}, lapse 0", fontsize=10)
        ax.set_xlabel("mu, x upper threshold (choice prob. on a typical "
                      "trial)", fontsize=8)
        _style(ax)
    axes[0].set_ylabel("per-person classification probability")
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_mixtures(mix, claims, path, spread=1.0, lapse=0.0):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (feat, col) in zip(axes, (("colour", "p_H2_claim"),
                                      ("fog", "p_H1_claim"))):
        for i, mult in enumerate([m for m in MU_MULTS if m >= 0.25]):
            xs, ys = [0.0], []
            base = claims[(claims.target == "none") & (claims.spread == spread)
                          & (claims.lapse == lapse)]
            ys.append(base[col].max())
            for pi in MIX_SHARES:
                s = mix[(mix.target == feat) & (mix.mu_mult == mult)
                        & (mix.spread == spread) & (mix.lapse == lapse)
                        & (mix.share_biased == pi)]
                xs.append(pi)
                ys.append(s[col].max())
            s = claims[(claims.target == feat) & (claims.mu_mult == mult)
                       & (claims.spread == spread) & (claims.lapse == lapse)]
            xs.append(1.0)
            ys.append(s[col].max())
            ax.plot(xs, ys, color=SERIES_COLOURS[i % 4],
                    marker=SERIES_MARKERS[i % 4], ls=SERIES_STYLES[i % 4],
                    label=f"mu {mult:g}x")
        ax.axhline(0.05, color="#555555", lw=0.8, ls=":")
        ax.set_title(f"{CLAIM[feat]} {feat}: worst N in 10-20, spread "
                     f"{spread:g}x, lapse {lapse:g}", fontsize=10)
        ax.set_xlabel("share of null people carrying the bias")
        _style(ax)
    axes[0].set_ylabel("P(claim), no memory effect")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_c1(summ, path):
    pairs = (summ[summ.target == "none"]
             [["lapse_participants", "lapse_controls"]].values.tolist())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, t in zip(axes, BIAS_C1_TARGETS):
        x = np.arange(len(pairs))
        for i, m in enumerate([0.0] + BIAS_C1_MULTS):
            s = summ[(summ.target == ("none" if m == 0 else t))
                     & (summ.mu_mult == m)]
            s = s.set_index(["lapse_participants", "lapse_controls"]) \
                .reindex([tuple(p) for p in pairs])
            ax.plot(x + (i - 1.5) * 0.15, s.fp_either_worst, ls="none",
                    marker=SERIES_MARKERS[i], color=SERIES_COLOURS[i],
                    label=f"mu {m:g}x")
        ax.axhline(0.05, color="#555555", lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:g}/{b:g}" for a, b in pairs], fontsize=8)
        ax.set_xlabel("lapse: participants / controls")
        ax.set_title(f"shared bias on {t}", fontsize=10)
        _style(ax)
    axes[0].set_ylabel("C1 false positive, either feature\n(worst cell, "
                       "after the check)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("C1 with the same bias in both groups and no memory "
                 "effect, post-exclusion procedure", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


# ------------------------------------------------------------
# The report
# ------------------------------------------------------------

def write_report(path, R, checks, meta):
    L = []
    W = L.append
    W("REGISTERED RULES UNDER A SHARED DIRECTIONAL PREFERENCE")
    W("(sensitivity_asymmetric_tastes_v4)")
    W("=" * 70)
    W(f"run          {meta['started']}  ({meta.get('elapsed_min', 0):.1f} "
      "min)")
    W(f"output       {meta['output']}")
    W(f"python       {meta['python']}, numpy {meta['numpy']}, scipy "
      f"{meta['scipy']}, pandas {meta['pandas']}")
    W("")
    W("WHAT THIS IS")
    W("-" * 70)
    W("The registered 5% false-claim rate for H1 and H2 assumes that people "
      "without the")
    W("memory effect have tastes centred on zero. Here their tastes are "
      "centred on mu,")
    W("in the predicted direction, and the registered thresholds and (k, m) "
      "are applied")
    W("unchanged. Claim rates are exact given the measured per-person "
      "rates; their")
    W("Monte Carlo SE is carried through (se_* columns, *_upper = point + "
      "1.96 SE).")
    W("")
    th = R.get("th_c")
    if th is not None:
        W("Thresholds: reproduction checks use the computed ones; "
          "everything else the")
        W("registered (rounded) ones.")
        for k, f in enumerate(FEATURES):
            reg = REGISTERED_THRESHOLDS[f]
            W(f"  {f:6s} computed {th['upper'][k]:+.5f} / "
              f"{th['lower'][k]:+.5f}; registered {reg[0]:+.3f} / "
              f"{reg[1]:+.3f}")
    W("")
    W("SETTINGS")
    W("-" * 70)
    W(f"  bias targets   {BIAS_TARGETS}")
    W(f"  mu             {MU_MULTS} x that feature's upper threshold")
    W(f"  spreads        {SPREADS} x upper threshold")
    W(f"  lapse          {LAPSES} (one fixed rate per person; with a "
      "fixed rate the")
    W("                 catch exclusion is independent of the weights and "
      "cannot change")
    W("                 these rates, so it is not simulated in this part)")
    W(f"  people / pool  {PEOPLE_PER_POOL:,}")
    W(f"  N              {N_RANGE[0]}-{N_RANGE[-1]}, registered (k, m) "
      "per N from combined_rule.csv")
    W(f"  mixtures       share biased {MIX_SHARES}")
    W(f"  C1 part        mu {BIAS_C1_MULTS} (and 0) on {BIAS_C1_TARGETS}, "
      f"spread {BIAS_C1_SPREAD}x,")
    W(f"                 strategy {BIAS_C1_STRATEGIES}, 7 lapse pairs, "
      f"{BIAS_C1_PARTICIPANT_N} x {BIAS_C1_CONTROL_N},")
    W(f"                 {BIAS_C1_STUDIES} studies per cell, "
      "post-exclusion procedure")
    W(f"  seeds          reproduction {SEED_REPRODUCTION}, pools "
      f"{SEED_POOLS}, C1 pools {SEED_C1_POOLS},")
    W(f"                 C1 draws {SEED_C1_DRAWS}; replay uses the power "
      "script's own")
    W("")
    W("CRITERIA, FIXED BEFORE THE RUN")
    W("-" * 70)
    W(f"  replay         classification_rates with its own seed and the "
      "computed thresholds")
    W("                 must reproduce every registered false-claim rate "
      "(five nulls, all")
    W(f"                 22 rules) to within {REPLAY_PEOPLE_TOL} people in "
      f"{N_TASTE_PEOPLE:,} per rate.")
    W(f"  independent    this script's code, new seed, mu = 0, lapse 0: "
      "every per-person")
    W("                 rate and every registered false-claim rate within "
      f"{VAL_MAX_ABS_Z} combined SE.")
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
    if "claims" not in R:
        open(path, "w").write("\n".join(L) + "\n")
        return
    prob, per, claims, mix = R["prob"], R["per"], R["claims"], R["mix"]
    W("")
    W("RESULTS")
    W("=" * 70)
    W("")
    W("1. THE SIZE OF EACH BIAS, AS BEHAVIOUR")
    W("-" * 70)
    W("Probability that a person with exactly the mean taste picks the "
      "predicted image")
    W("on a typical informative trial of that feature (median informative "
      "difference),")
    W("and on a 1.0 SD difference.")
    W(prob.round(4).to_string(index=False))
    W("")
    W("2. PER-PERSON CLASSIFICATION (the biased feature, lapse 0)")
    W("-" * 70)
    for f in FEATURES:
        d = per[(per.feature == f) & (per.lapse == 0.0)
                & ((per.target == f) | (per.target == "none"))]
        W(f"\n  {f}: P(predicted) / P(opposite); rows mu, columns spread")
        a = d.pivot_table(index="mu_mult", columns="spread",
                          values="p_predicted")
        b = d.pivot_table(index="mu_mult", columns="spread",
                          values="p_opposite")
        both_ = a.round(4).astype(str) + " / " + b.round(4).astype(str)
        W(both_.to_string())
    W("")
    W("3. P(CLAIM) WITH NO MEMORY EFFECT, against the registered "
      "false_worst")
    W("-" * 70)
    W("The registered false_worst is the rate at spread 1.0, mu = 0. "
      "Rows: mu;")
    W("columns: spread. Worst N in 10-20 and the N where it occurs.")
    for lapse in LAPSES:
        for target, col, feat in (("colour", "p_H2_claim", "colour"),
                                  ("fog", "p_H1_claim", "fog"),
                                  ("both", "p_H2_claim", "colour"),
                                  ("both", "p_H1_claim", "fog"),
                                  ("both", "p_either_claim", None)):
            d = claims[(claims.lapse == lapse)
                       & ((claims.target == target)
                          | (claims.target == "none"))]
            idx = d.groupby(["mu_mult", "spread"])[col].idxmax()
            w = d.loc[idx]
            piv = w.pivot_table(index="mu_mult", columns="spread",
                                values=col)
            pn = w.pivot_table(index="mu_mult", columns="spread",
                               values="n")
            cell = piv.round(3).astype(str) + " (N" + \
                pn.astype(int).astype(str) + ")"
            W(f"\n  lapse {lapse:g}, bias on {target}: {col}")
            W(cell.to_string())
    W("")
    W("  At every N (lapse 0, spread 1.0, the registered worst case):")
    for target, col in (("colour", "p_H2_claim"), ("fog", "p_H1_claim"),
                        ("both", "p_either_claim")):
        d = claims[(claims.lapse == 0.0) & (claims.spread == 1.0)
                   & ((claims.target == target)
                      | (claims.target == "none"))]
        piv = d.pivot_table(index="mu_mult", columns="n", values=col)
        W(f"\n  bias on {target}: {col}")
        W(piv.round(3).to_string())
    reg = claims[(claims.target == "none") & (claims.lapse == 0.0)
                 & (claims.spread == 1.0)]
    W("")
    W("  Registered false_worst, per N (H2 colour / H1 fog):")
    W("  " + "  ".join(f"N{int(r.n)} {r.registered_H2_false_worst:.3f}/"
                       f"{r.registered_H1_false_worst:.3f}"
                       for _, r in reg.iterrows()))
    W("")
    W("4. WHERE THE 5% IS CROSSED (lapse 0)")
    W("-" * 70)
    W("The smallest mu (x threshold) at which the claim rate, at its worst "
      "N, exceeds 0.05.")
    for target, col in (("colour", "p_H2_claim"), ("fog", "p_H1_claim"),
                        ("both", "p_either_claim")):
        for spread in SPREADS:
            d = claims[(claims.lapse == 0.0) & (claims.spread == spread)
                       & ((claims.target == target)
                          | (claims.target == "none"))]
            w = d.groupby("mu_mult")[col].max()
            over = w[w > 0.05]
            W(f"  bias on {target:6s} {col:15s} spread {spread:<5g} "
              + (f"first above 0.05 at mu {over.index.min():g}x "
                 f"({w[over.index.min()]:.3f})" if len(over)
                 else "never above 0.05 on this grid"))
    W("")
    W("5. MIXTURES: a share of the null people biased (lapse 0, spread "
      "1.0)")
    W("-" * 70)
    for target, col in (("colour", "p_H2_claim"), ("fog", "p_H1_claim"),
                        ("both", "p_either_claim")):
        d = mix[(mix.target == target) & (mix.lapse == 0.0)
                & (mix.spread == 1.0)]
        piv = d.groupby(["mu_mult", "share_biased"])[col].max().unstack()
        W(f"\n  bias on {target}: {col}, worst N; rows mu, columns share "
          "biased")
        W(piv.round(3).to_string())
    if "c1sum" in R:
        s = R["c1sum"]
        W("")
        W("6. C1 WITH THE SAME BIAS IN BOTH GROUPS (no memory effect)")
        W("-" * 70)
        W("False positive = C1 supported for either feature after the "
          "attention check;")
        W("worst and mean over the 12 cells (3 x 2 sizes x 2 strategies) of "
          "each lapse pair.")
        W("mu = 0 ('none') is the reference run with this script's own "
          "pools.")
        W(s[["target", "mu_mult", "lapse_participants", "lapse_controls",
             "fp_either_mean", "fp_either_worst", "fp_colour_worst",
             "fp_fog_worst", "ungated_either_worst", "gate_fires_mean",
             "zero_catch_worst"]].round(3).to_string(index=False))
        m = s[s.lapse_kind == "matched"]
        W("")
        W(f"  matched lapses: worst {m.fp_either_worst.max():.3f}, mean "
          f"{m.fp_either_mean.mean():.3f} (nominal {ALPHA}); at mu = 0 "
          f"worst {m[m.target == 'none'].fp_either_worst.max():.3f}")
        a = s[s.lapse_kind == "controls lapse more"]
        W(f"  controls lapse more: worst after the check "
          f"{a.fp_either_worst.max():.3f}; ungated worst "
          f"{a.ungated_either_worst.max():.3f}")
        W("  Monte Carlo SE of one cell's rate near 0.05: "
          f"{np.sqrt(0.05 * 0.95 / BIAS_C1_STUDIES):.4f}")
    W("")
    W("NOTES")
    W("-" * 70)
    W("- Claim rates are exact given the per-person rates measured on "
      f"{PEOPLE_PER_POOL:,} simulated")
    W("  people per pool. 'either' and 'both' are exact given the measured "
      "3 x 3 joint")
    W("  table (the two classifications of one person are fitted jointly "
      "and are not")
    W("  independent).")
    W("- H1 and H2 are two separate registered claims, each calibrated at "
      "5%. 'Either")
    W("  claim' is not a registered quantity: even at mu = 0 it is close to "
      "the sum of")
    W("  the two rates (up to about 0.09 at spread 1.0). It is reported "
      "because a")
    W("  shared bias on both features raises both at once.")
    W("- A shared bias at mu = 1.0 puts the average null person's weight AT "
      "the detection")
    W("  threshold; the grid is a sensitivity range, not an estimate of "
      "any population.")
    W("")
    W("FILES")
    W("-" * 70)
    for sub in ("tables", "figures"):
        d = os.path.join(meta["output"], sub)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                W(f"  {sub}/{f}")
    open(path, "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    t_start = time.time()
    out = resolve_output_root(OUTPUT_ROOT)
    for sub in ("tables", "figures"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    tdir, fdir = os.path.join(out, "tables"), os.path.join(out, "figures")
    meta = {"script": os.path.basename(__file__),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "output": out,
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform()}
    print(f"output: {out}")
    checks, R = Checks(), {}
    try:
        check_inputs(checks, meta)
        banner("THE FROZEN TRIAL LIST")
        facts = verify_frozen(FROZEN_ROOT, [])
        _, X, _r = load_trials(os.path.join(FROZEN_ROOT, FROZEN_TRIAL_LIST),
                               "frozen")
        checks.require("frozen trial list verified; 180 analysed trials",
                       len(X) == 180, f"{len(X)} analysed trials")
        meta["frozen"] = facts
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            th_c = compute_floors(X, "frozen", [])
        R["th_c"] = th_c
        rounded = [round(float(th_c[s][k]), 3) for k in (0, 1)
                   for s in ("upper", "lower")]
        want = [REGISTERED_THRESHOLDS[f][j] for f in FEATURES
                for j in (0, 1)]
        checks.require("computed thresholds round to the registered ones",
                       rounded == want, f"{rounded}")
        replay_rates = identity_check(X, th_c, checks, tdir)

        rule_tab = pd.read_csv(os.path.join(REGISTRATION_DIR,
                                            "combined_rule.csv"))
        sweep = pd.read_csv(os.path.join(POWER_FROZEN_DIR,
                                         "rule_sweep.csv"))
        rules = {(r.feature, int(r.n)): (int(r.k), int(r.m))
                 for _, r in rule_tab.iterrows()}
        checks.require("combined_rule.csv has a rule for every feature and "
                       "N in 10-20", len(rules) == 2 * len(N_RANGE)
                       and all((f, n) in rules for f in FEATURES
                               for n in N_RANGE))
        reg_false = {(r.feature, int(r.n)): float(r.false_worst)
                     for _, r in rule_tab.iterrows()}
        validate(X, th_c, replay_rates, rule_tab, sweep, checks, tdir)

        th = {"upper": np.array([REGISTERED_THRESHOLDS[f][0]
                                 for f in FEATURES]),
              "lower": np.array([REGISTERED_THRESHOLDS[f][1]
                                 for f in FEATURES])}
        med = informative_medians(X)
        U = th["upper"]
        rows = []
        for target in BIAS_TARGETS:
            for mult in MU_MULTS:
                mu = mu_vector(target, mult, U)
                rows.append({
                    "target": target, "mu_mult": mult,
                    "mu_colour_per_SD": mu[0], "mu_fog_per_SD": mu[1],
                    "p_typical_colour": as_probability(mu[0],
                                                       med["colour"]),
                    "p_typical_fog": as_probability(mu[1], med["fog"]),
                    "p_1SD_colour": as_probability(mu[0], 1.0),
                    "p_1SD_fog": as_probability(mu[1], 1.0)})
        prob = pd.DataFrame(rows)
        prob.to_csv(os.path.join(tdir, "bias_as_probability.csv"),
                    index=False)
        checks.info("median informative difference (typical trial)",
                    f"colour {med['colour']:.3f} SD, fog {med['fog']:.3f} SD")

        banner("H1 AND H2 UNDER A SHARED BIAS")
        per, joint, claims, joints = run_rule_grid(X, th, rules, reg_false,
                                                   med)
        worst_enum = float(max(claims.enum_vs_exact_H2.abs().max(),
                               claims.enum_vs_exact_H1.abs().max()))
        checks.require("joint enumeration agrees with the exact "
                       "per-feature rates in every cell", worst_enum < 1e-9,
                       f"worst {worst_enum:.1e}")
        mix = mixtures(joints, rules, reg_false)
        per.to_csv(os.path.join(tdir, "per_person_rates.csv"), index=False)
        joint.to_csv(os.path.join(tdir, "joint_rates.csv"), index=False)
        claims.to_csv(os.path.join(tdir, "claim_rates.csv"), index=False)
        mix.to_csv(os.path.join(tdir, "mixture_claim_rates.csv"),
                   index=False)
        R.update({"prob": prob, "per": per, "claims": claims, "mix": mix})
        z0 = claims[(claims.target == "none") & (claims.lapse == 0.0)
                    & (claims.spread == 1.0)]
        d = max((z0.p_H2_claim - z0.registered_H2_false_worst).abs().max(),
                (z0.p_H1_claim - z0.registered_H1_false_worst).abs().max())
        checks.info("mu = 0, registered thresholds, spread 1.0: largest "
                    "difference from false_worst", f"{d:.4f} (Monte Carlo "
                    "and threshold rounding; the criterion is the "
                    "independent check above)")

        banner("C1 UNDER A SHARED BIAS (post-exclusion procedure)")
        c1 = run_c1_bias(X, th, med)
        c1.to_csv(os.path.join(tdir, "c1_bias_cells.csv"), index=False)
        c1sum = c1_summary(c1)
        c1sum.to_csv(os.path.join(tdir, "c1_bias_summary.csv"),
                     index=False)
        R["c1sum"] = c1sum
        if MAKE_FIGURES:
            for lapse in LAPSES:
                fig_claims(claims, med, U, os.path.join(
                    fdir, f"claim_rate_vs_bias_lapse{lapse:g}.png"), lapse)
            fig_per_person(per, med, U, os.path.join(
                fdir, "per_person_rates.png"))
            fig_mixtures(mix, claims, os.path.join(fdir, "mixtures.png"))
            fig_c1(c1sum, os.path.join(fdir, "c1_false_positive_bias.png"))
    except CheckFailed as e:
        R["failed"] = str(e)
        print(f"\nSTOPPED: a check failed: {e}")
    meta["elapsed_min"] = (time.time() - t_start) / 60
    meta["checks"] = checks.rows
    meta["settings"] = settings_snapshot()
    meta["result"] = ("stopped: " + R["failed"]) if "failed" in R \
        else "complete"
    with open(os.path.join(out, "run_meta.json"), "w") as fh:
        json.dump(jsonable(meta), fh, indent=1)
    write_report(os.path.join(out, "report.txt"), R, checks, meta)
    print(f"\nreport: {os.path.join(out, 'report.txt')}  "
          f"({meta['elapsed_min']:.1f} min)")
    if "failed" in R:
        sys.exit(1)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    # rule_false_claim_exact (copied) takes log(0) when a per-person rate
    # is exactly zero and then skips that term; the warning is harmless.
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            message="divide by zero encountered in log")
    main()
