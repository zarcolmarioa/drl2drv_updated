"""
sign_test_registration_v4.py   (step 8c)

WHERE THIS SITS
---------------
  step 8   simulate_power_v4_n10_20    the design analysis (untouched)
  step 8b  registration_table_v4       the count rule, selected per N
  step 8c  THIS SCRIPT                 the sign test as the PRIMARY rule
                                       for H1 and H2, with the count rule
                                       kept as the registered SECONDARY

This script RUNS NO SIMULATION. It reads the frozen outputs of step 8
and step 8b and computes everything else exactly. Every number it prints
is either a value of rule_sweep.csv / registration_table.csv or an exact
enumeration from per-person rates RECOVERED from rule_sweep.csv itself.

THE DECISION
------------
  PRIMARY    the sign test (Dixon & Mood, 1946). Among the participants
             detected in either direction, the number in the predicted
             direction is tested against Binomial(d, 0.5), one-sided,
             exact, alpha 0.05, no floor on the number predicted (the
             sweep's sign family with minimum predicted 0).
  SECONDARY  the registered count rule, unchanged: at least k predicted
             and at most m opposite, (k, m) per N from combined_rule.csv.

Under ANY symmetric (zero-centred) no-effect model the predicted count
given d detections is Binomial(d, 0.5), so the sign test's false-claim
rate cannot exceed alpha whatever the taste spread. The count rule's
rate is controlled only at the spreads it was calibrated against.

WHY NOTHING NEEDS SIMULATING
----------------------------
rule_sweep.csv is not a table of simulated study counts. Each rate in it
is an EXACT trinomial sum, built from per-person classification rates
that were themselves simulated:

  people without the effect   P(predicted) = P(opposite) = q, the mean
                              of the two measured rates over
                              N_TASTE_PEOPLE people, so q is an exact
                              multiple of 1 / (2 x N_TASTE_PEOPLE)
  responders                  P(predicted), P(opposite) measured on
                              N_DETECT_PEOPLE people, so exact
                              multiples of 1 / N_DETECT_PEOPLE

Those rates are not written to any file. They are RECOVERED here: for
each null, the single multiple of 1/40,000 that reproduces the COUNT
rules' false-claim rates in the sweep; for each responder type, the
single pair of multiples of 1/600 that reproduces the COUNT rules' power
at full response. Both recoveries use count rules only, so reproducing
the SIGN rows from them afterwards is an out-of-sample test, not a fit.
Every sign-test and registered-count-rule power and false-claim value in
the sweep is then rebuilt from them and must agree to EXACT_TOL.

Where the review's "about 0.182 per direction for colour and 0.186 for
fog" comes from: the symmetrised rates of classification_rates() in the
power script at spread 1.0 x threshold, 0.18195 and 0.185525 exactly.
The power report prints them to 4 decimals only (the assumed_taste_1x
alpha of its prevalence section); this script recovers them exactly and
checks them against those printed values.

WHAT IT PRODUCES  (OUT_ROOT)
----------------
  sign_test_table.csv         item 1: the sign test per feature and N --
                              false-claim rates under every null with SE
                              and upper bound, the worst of them, and
                              power in every cell -- with the registered
                              count rule beside it as the secondary rule
  decision_table.csv          item 2: d = 1..20 detected participants,
                              the minimum number predicted for support
                              and the exact one-sided p at that boundary
  exact_false_claim.csv       item 3: the exact trinomial rate beside the
                              sweep's, at every null's recovered q and at
                              the nominal 2.5%
  worst_case_over_q.csv       item 3: the largest rate over q in [0, 0.5]
                              per N, for the sign test, the count rule
                              (secondary) and either
  firing_probability.csv      item 4: P(at least 5 detections) per N and
                              scenario
  primary_secondary_null.csv  item 5: P(sign), P(count), P(either),
                              P(both) under every no-effect model
  recovered_rates.csv         the per-person rates the sweep was built
                              from, as recovered
  identity_check.csv          the copied functions against the original
  sign_test_registration.png  item 6: the figure
  sign_test_report.txt        item 7: the report
  run_meta.json               settings, input checksums, check results

CHECKS  (each stops the run when it fails)
------
  CHECK 1  INPUTS. Every file present, its SHA-256 recorded, and equal to
           the committed copy.
  CHECK 2  SIGN ROWS. The sign rows with minimum predicted 0 exist for
           every feature x N in 10..20, and their values equal the sign
           rows of registration_table.csv wherever both report the same
           quantity (to the table's rounding).
  CHECK 3  EXACT vs SIMULATED. The exact trinomial rate equals the
           sweep's sign rate at every N and null.
  CHECK 4  WORST CASE. The sign test's largest rate over all q is at or
           below alpha at every N.
  CHECK 5  DECISION TABLE. It reproduces scipy's one-sided binomial
           p-values, and every boundary is the smallest that works.
  Also:    the copies are verbatim and give the original's results; the
           recovered rates are unique and reproduce the sweep; the two
           registration columns for the quarter-share cell agree.

USAGE
-----
    python sign_test_registration_v4.py

No arguments. Every setting is in SETTINGS below. Paths are relative to
the project folder: scripts at its top level, data under data/.
"""

import ast
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy import stats, optimize
from scipy.special import gammaln

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

# --- Inputs ---
IN_ROOT = "data/power_v4_n10_20/frozen"
SWEEP_CSV = os.path.join(IN_ROOT, "rule_sweep.csv")
# power_report.txt sits in the PARENT of the set folder; read only for
# the 4-decimal cross-check of the recovered null rates.
POWER_REPORT = os.path.join(os.path.dirname(IN_ROOT), "power_report.txt")

REG_ROOT = "data/registration_v4"
COMBINED_CSV = os.path.join(REG_ROOT, "combined_rule.csv")
REGTABLE_CSV = os.path.join(REG_ROOT, "registration_table.csv")

# POWER_SCRIPT
#   The original, for the start-up check of the copied functions.
#   Required: the run stops with a message if it is not here.
POWER_SCRIPT = "simulate_power_v4_n10_20.py"

# COMMITTED_SHA256
#   The committed copies (repository paths data_power_v4/...). A local
#   file that differs stops the run: every number below is a property
#   of these exact files. A difference in line endings alone (a Windows
#   checkout with autocrlf) also changes the hash; re-download the file
#   as committed if that is the cause.
COMMITTED_SHA256 = {
    SWEEP_CSV:
        "0d5d03282c95492419f0db561c87035db4bd2d6d5185f78e53a8b9d5e3bcffd7",
    COMBINED_CSV:
        "9e9e9a2a2b74c709792a0bc88767b20a9a0ed2f021fbf79fe419eb9382377a0a",
    REGTABLE_CSV:
        "fffc96842d8c99d5bb67b972df96a9d9ed7ab7a4aec2e49d81215a3197edc9ad",
    POWER_REPORT:
        "4296cfb4cdc4be54c1f1a6b94c1a126d078c68fc0ab27d93e4de04d4299332b5",
    POWER_SCRIPT:
        "852f3b33eb349153fbff9f95c50e1cd3c4f84f68b204d79a23f020f471bd1730",
}

# --- Output ---
OUT_ROOT = "data/sign_test_v4"
# ON_EXISTING
#   "new"    write to OUT_ROOT_2, _3, ... rather than overwrite.
#   "error"  stop instead.
ON_EXISTING = "new"

# --- The rules ---
N_MIN = 10
N_MAX = 20
SIGN_FAMILY = "sign"
SIGN_MIN_PREDICTED = 0      # the variant registered as primary
ALPHA = 0.05                # the sign test's one-sided alpha
TARGET_FALSE = 0.05         # the 5% line in the tables and the figure
FIRE_MIN_DETECTIONS = 5     # the smallest d at which the test can fire;
                            # CHECK 5 confirms it from the decision table
DECISION_D_MAX = 20         # rows of the decision table

# --- The no-effect models, as the sweep names them ---
#   "guesser" is the random chooser (weights exactly zero); the others
#   are zero-centred tastes at that multiple of the feature's threshold.
NULLS = ["guesser", "0.25", "0.5", "0.75", "1"]
NULL_COL = {"guesser": "false_guesser", "0.25": "false_taste_0.25",
            "0.5": "false_taste_0.5", "0.75": "false_taste_0.75",
            "1": "false_taste_1"}
# NOMINAL_GUESSER_Q
#   The thresholds' nominal rate per direction. The REALISED rate of the
#   sweep's guesser sample is recovered below; both are reported.
NOMINAL_GUESSER_Q = 0.025

# --- The cells (the registration script's, read the same way) ---
PLANNING_WEIGHT = {"colour": 0.50, "fog": 0.70}
ROBUSTNESS_WEIGHT = {"colour": 0.30, "fog": 0.50}
PLANNING_SHARE = 0.50
PLANNING_LAPSE = 0.1
COST_LAPSE = 0.1
# CELLS
#   name -> (weight set, share, lapse). "quarter" and "cost_planning_qtr"
#   are the SAME cell (planning weights, share 0.25, lapse 0.1): it is
#   computed once and reported under both labels, and the two
#   registration columns that carry it are checked to agree.
CELLS = {
    "planning":          ("planning",   0.50, 0.1),
    "cost_lower_half":   ("robustness", 0.50, 0.1),
    "quarter":           ("planning",   0.25, 0.1),
    "all_respond":       ("planning",   1.00, 0.1),
    "planning_lapse0":   ("planning",   0.50, 0.0),
    "planning_lapse0.2": ("planning",   0.50, 0.2),
}
CELL_LABEL = {
    "planning": "planning cell (half respond, lapse 0.1)",
    "cost_lower_half": "cost cell: weaker pair, half respond",
    "quarter": "quarter respond = cost cell: planning weights, "
               "quarter share",
    "all_respond": "all respond",
    "planning_lapse0": "planning cell at lapse 0",
    "planning_lapse0.2": "planning cell at lapse 0.2",
}
# The registration_table.csv column for the count rule in each cell;
# None where that table has no such column (read from the sweep row of
# the registered rule instead).
REGTABLE_POWER_COL = {
    "planning": "power_planning_s0.5_l0.1",
    "cost_lower_half": "cost_lower_half",
    "quarter": "power_s0.25_l0.1",
    "all_respond": "power_s1_l0.1",
    "planning_lapse0": "power_s0.5_l0",
    "planning_lapse0.2": None,
}

# --- Recovery and exactness ---
EXACT_TOL = 1e-12        # exact quantities must agree to this
UNIQUE_GAP = 1e-9        # the runner-up candidate must miss by this
RESP_OPP_MAX = 60        # largest opposite count (of 600) searched
RECOVER_N = (10, 20)     # sizes whose count rows the recovery uses
Q_GRID = 20001           # grid on [0, 0.5] for the worst case over q

FIG_DPI = 150
SERIES_COLOURS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
SERIES_MARKERS = ["o", "s", "^", "D"]
SERIES_STYLES = ["-", "--", "-.", ":"]

FEATURES = ("fog", "colour")          # fog first: H1, then H2
CLAIM = {"fog": "H1 fog", "colour": "H2 colour"}


# ============================================================
# COPIED SETTINGS
#
# Values identical to simulate_power_v4_n10_20.py. The copied functions
# and the recovery read them; the start-up check compares each with the
# power script's own value and stops if any differs.
# ============================================================

RULE_SIGN_ALPHA = 0.05
RULE_SIGN_MIN_PREDICTED = [0, 3, 5, 8]
SAMPLE_SIZES = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                25, 30, 40, 60]
N_TASTE_PEOPLE = 20_000
N_DETECT_PEOPLE = 600
TASTE_SPREADS_IN_FLOORS = [0.25, 0.5, 0.75, 1.0]
NONRESPONDER_SPREAD_MULT = 0.5
RULE_RESPONDER_WEIGHTS = [0.30, 0.50, 0.70]
RULE_RESPONDER_SHARES = [0.10, 0.25, 0.50, 1.00]
LAPSE_RATES = [0.0, 0.1, 0.2]

COPIED_SETTINGS = [
    "RULE_SIGN_ALPHA", "RULE_SIGN_MIN_PREDICTED", "SAMPLE_SIZES",
    "N_TASTE_PEOPLE", "N_DETECT_PEOPLE", "TASTE_SPREADS_IN_FLOORS",
    "NONRESPONDER_SPREAD_MULT", "RULE_RESPONDER_WEIGHTS",
    "RULE_RESPONDER_SHARES", "LAPSE_RATES",
]

# ============================================================
# COPIED FUNCTIONS
#
# VERBATIM from simulate_power_v4_n10_20.py, each marked with its
# source lines, in the original's order. Do not edit them here: the
# start-up check compares each one with the original (syntax tree
# and results), and a copy that differs stops the run.
# ============================================================

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

# --- COPY of simulate_power_v4_n10_20.py L3261-L3275 (combine_grids), verbatim ---
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
# --- end of copy (combine_grids) ---

# --- COPY of simulate_power_v4_n10_20.py L3278-L3294 (sign_test_critical), verbatim ---
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
# --- end of copy (sign_test_critical) ---

# --- COPY of simulate_power_v4_n10_20.py L3297-L3301 (rule_mask_count), verbatim ---
def rule_mask_count(n, k, m):
    """Boolean (a, b) mask: at least k predicted, at most m opposite."""
    a = np.arange(n + 1)[:, None]
    b = np.arange(n + 1)[None, :]
    return (a >= k) & (b <= m)
# --- end of copy (rule_mask_count) ---

# --- COPY of simulate_power_v4_n10_20.py L3304-L3315 (rule_mask_sign), verbatim ---
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
# --- end of copy (rule_mask_sign) ---

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


# ============================================================
# THIS SCRIPT'S OWN CODE (not copied)
# ============================================================

COPIED_FUNCTIONS = ["sha256_of", "trinomial_grid", "combine_grids",
                    "sign_test_critical", "rule_mask_count",
                    "rule_mask_sign", "rule_false_claim_exact"]

# power_report.txt is read only for a cross-check; the run goes on
# without it.
OPTIONAL_INPUTS = {POWER_REPORT}


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
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def resolve_output_dir():
    if not os.path.exists(OUT_ROOT):
        os.makedirs(OUT_ROOT)
        return OUT_ROOT
    if ON_EXISTING == "error":
        raise RuntimeError(f"{OUT_ROOT} exists and ON_EXISTING is "
                           "'error'.")
    n = 2
    while os.path.exists(f"{OUT_ROOT}_{n}"):
        n += 1
    out = f"{OUT_ROOT}_{n}"
    os.makedirs(out)
    return out


def power_col(weight, share, lapse):
    """The sweep's power column name, as registration_table_v4.py
    spells it: power_w<weight>_s<share>_lapse<lapse>."""
    return f"power_w{weight:g}_s{share:g}_lapse{lapse:g}"


def cell_weight(cell, feat):
    which = CELLS[cell][0]
    return (PLANNING_WEIGHT if which == "planning"
            else ROBUSTNESS_WEIGHT)[feat]


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


# ============================================================
# CHECK 1: the inputs
# ============================================================

def check_inputs(checks, meta):
    banner("INPUTS  (CHECK 1)")
    if not os.path.isfile(POWER_SCRIPT):
        checks.require(
            f"{POWER_SCRIPT} is present (needed to check the copied "
            "functions)", False,
            f"not found at {os.path.abspath(POWER_SCRIPT)}. Put the "
            "committed simulate_power_v4_n10_20.py at the top level of "
            "the project folder, next to this script, or set "
            "POWER_SCRIPT in SETTINGS")
    sums = {}
    for path, want in COMMITTED_SHA256.items():
        if not os.path.isfile(path):
            if path in OPTIONAL_INPUTS:
                checks.info(f"{path} not found",
                            "optional: only the 4-decimal cross-check "
                            "of the recovered rates is skipped")
                sums[path] = None
                continue
            checks.require(f"{path} exists", False,
                           "required input missing")
        got = sha256_of(path)
        sums[path] = got
        checks.require(f"{path} is the committed copy (SHA-256)",
                       got == want,
                       f"{got[:16]}..." + ("" if got == want else
                                           f" expected {want[:16]}..."))
    sums[os.path.abspath(__file__)] = sha256_of(os.path.abspath(__file__))
    meta["input_sha256"] = sums
    checks.require("ALPHA equals the sweep's RULE_SIGN_ALPHA",
                   ALPHA == RULE_SIGN_ALPHA, f"{ALPHA} / {RULE_SIGN_ALPHA}")
    checks.require("the registered variant is a sweep variant",
                   SIGN_MIN_PREDICTED in RULE_SIGN_MIN_PREDICTED,
                   f"minimum predicted {SIGN_MIN_PREDICTED}")


# ============================================================
# The copied functions against the original
# ============================================================

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


def _diff(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    fa, fb = np.isfinite(a), np.isfinite(b)
    if (fa != fb).any() or (a[~fa] != b[~fb]).any():
        return np.inf
    return float(np.max(np.abs(a[fa] - b[fb]))) if fa.any() else 0.0


def identity_check(checks, root):
    banner("THE COPIED FUNCTIONS AGAINST THE ORIGINAL")
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
    rows = []
    me = sys.modules[__name__]

    def both(label, f):
        d = _diff(f(me), f(pw))
        rows.append({"quantity": label, "max_abs_diff": d,
                     "ok": d <= EXACT_TOL})

    both("sha256_of (this script's own file)",
         lambda M: [int(M.sha256_of(POWER_SCRIPT)[:12], 16)])
    for n, a, b in ((10, 0.18195, 0.18195), (15, 0.6, 0.01),
                    (20, 0.024325, 0.024325), (12, 0.0, 0.3)):
        both(f"trinomial_grid({n}, {a}, {b})",
             lambda M: M.trinomial_grid(n, a, b))
    g1 = trinomial_grid(5, 0.5, 0.01)
    g2 = trinomial_grid(7, 0.08, 0.08)
    both("combine_grids", lambda M: M.combine_grids(g1, g2))
    both("sign_test_critical(61)", lambda M: M.sign_test_critical(61))
    crit = sign_test_critical(61)
    for n in (10, 20):
        both(f"rule_mask_count({n}, 5, 1)",
             lambda M: M.rule_mask_count(n, 5, 1))
        both(f"rule_mask_sign({n}, crit, 0)",
             lambda M: M.rule_mask_sign(n, crit, 0))
    both("rule_false_claim_exact",
         lambda M: [M.rule_false_claim_exact(n, k, m, p, p)
                    for n, k, m in ((10, 4, 0), (15, 6, 3), (20, 5, 1))
                    for p in (0.024325, 0.18195, 0.3)])
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(root, "identity_check.csv"), index=False)
    bad = tab[~tab.ok]
    checks.require("copied functions give the original's results",
                   bad.empty, f"{len(tab) - len(bad)}/{len(tab)} "
                   f"identical to {EXACT_TOL:g}"
                   + (f"; differ: {', '.join(bad.quantity)}"
                      if len(bad) else ""))


# ============================================================
# Reading the sweep and the registration
# ============================================================

def load_tables(checks):
    banner("READING THE SWEEP AND THE REGISTRATION")
    S = pd.read_csv(SWEEP_CSV)
    C = pd.read_csv(COMBINED_CSV)
    T = pd.read_csv(REGTABLE_CSV)
    print(f"  rule_sweep.csv          {len(S):,} rows")
    print(f"  combined_rule.csv       {len(C)} rows")
    print(f"  registration_table.csv  {len(T)} rows")
    need = (["feature", "n", "family", "k", "m",
             "nonresponder_taste_spread", "worst_strict",
             "worst_upper_strict"]
            + [NULL_COL[x] + s for x in NULLS for s in ("", "_se",
                                                         "_upper")])
    for feat in FEATURES:
        for cell in CELLS:
            w = cell_weight(cell, feat)
            _, share, lapse = CELLS[cell]
            need.append(power_col(w, share, lapse))
    missing = sorted(set(c for c in need if c not in S.columns))
    checks.require("rule_sweep.csv has every column this script reads",
                   not missing, ", ".join(missing[:6]))
    spread = sorted(S.nonresponder_taste_spread.unique())
    checks.require("the sweep's non-responders sit at "
                   "NONRESPONDER_SPREAD_MULT", spread ==
                   [NONRESPONDER_SPREAD_MULT], f"{spread}")
    reg = {}
    for feat in FEATURES:
        for n in range(N_MIN, N_MAX + 1):
            c = C[(C.feature == feat) & (C.n == n)]
            if len(c) != 1:
                checks.require(f"combined_rule.csv has one rule for "
                               f"{feat} N = {n}", False, f"{len(c)} rows")
            reg[(feat, n)] = c.iloc[0]
    checks.require("combined_rule.csv covers both features x N "
                   f"{N_MIN}..{N_MAX}", True, f"{len(reg)} rules")
    return S, C, T, reg


def sweep_row(S, feat, n, family, k, m, m_spec=None):
    """One rule's row of the sweep. A proportional rule can appear under
    two m fractions that floor to the same m (at N = 10, 0.0 and 0.05
    both give m = 0); they are the same rule, so the registered row is
    picked by its own m_spec, as combined_rule.csv records it."""
    r = S[(S.feature == feat) & (S.n == n) & (S.family == family)
          & (S.k == k) & (S.m == m)]
    if m_spec is not None and len(r) > 1:
        r = r[np.isclose(r.m_spec, float(m_spec))]
    if len(r) != 1:
        raise CheckFailed(f"rule_sweep.csv: {len(r)} rows for {feat} "
                          f"N={n} {family} k={k} m={m}"
                          + (f" m_spec={m_spec}" if m_spec is not None
                             else ""))
    return r.iloc[0]


def reg_row(S, reg, feat, n):
    """The sweep row of the registered count rule at (feature, N)."""
    r = reg[(feat, n)]
    return sweep_row(S, feat, n, r.family, int(r.k), int(r.m), r.m_spec)


def sign_rows(S, checks):
    """CHECK 2 (first part): the sign rows used exist everywhere."""
    banner("THE SIGN ROWS  (CHECK 2)")
    rows, missing = {}, []
    for feat in FEATURES:
        for n in range(N_MIN, N_MAX + 1):
            r = S[(S.feature == feat) & (S.n == n)
                  & (S.family == SIGN_FAMILY)
                  & (S.k == SIGN_MIN_PREDICTED)]
            if len(r) != 1:
                missing.append(f"{feat} N{n}")
            else:
                rows[(feat, n)] = r.iloc[0]
    checks.require(f"sign rows with minimum predicted "
                   f"{SIGN_MIN_PREDICTED} exist for every feature x N "
                   f"{N_MIN}..{N_MAX}", not missing,
                   f"{len(rows)} rows" + (f"; missing {missing}"
                                          if missing else ""))
    # the sweep's own worst column is the max over the five nulls
    worst_bad = 0.0
    for r in rows.values():
        w = max(float(r[NULL_COL[x]]) for x in NULLS)
        wu = max(float(r[NULL_COL[x] + "_upper"]) for x in NULLS)
        worst_bad = max(worst_bad, abs(w - r.worst_strict),
                        abs(wu - r.worst_upper_strict))
    checks.require("the worst of the five nulls equals the sweep's "
                   "worst_strict / worst_upper_strict",
                   worst_bad <= EXACT_TOL, f"largest gap {worst_bad:.1e}")
    return rows


# ============================================================
# Exact enumeration, vectorised over candidate rates
# ============================================================

def _cells(n):
    a = np.arange(n + 1)[:, None]
    b = np.arange(n + 1)[None, :]
    ok = (a + b) <= n
    return a, b, ok


def sym_pmf_matrix(n, qs):
    """The symmetric trinomial pmf, P(pred) = P(opp) = q, for every q in
    `qs` at once: shape (len(qs), (n+1)^2), flattened over (a, b), zero
    where a + b > n. Entry: n!/(a! b! c!) q^(a+b) (1-2q)^c.

    THIS SCRIPT'S OWN CODE, for speed over many q. Every value that is
    reported is cross-checked against the copied trinomial_grid
    (CHECK 3 and the refinement of the worst case)."""
    a, b, ok = _cells(n)
    c = np.where(ok, n - a - b, 0)
    coef = np.where(ok, np.exp(gammaln(n + 1) - gammaln(a + 1)
                               - gammaln(b + 1) - gammaln(c + 1)), 0.0)
    q = np.asarray(qs, float)[:, None, None]
    out = coef[None] * q ** (a + b)[None] \
        * np.clip(1 - 2 * q, 0, None) ** c[None]
    return out.reshape(len(qs), -1)


def sym_claim_curve(n, mask, qs):
    """P(claim) for one rule's (a, b) mask, at every q in `qs`."""
    return sym_pmf_matrix(n, qs) @ mask.ravel().astype(float)


def pmf_pairs(n, rp, ro):
    """Trinomial pmf over (a, b) for many (rp, ro) pairs: shape
    (pairs, n+1, n+1). This script's own code, used only to search for
    the responder rates; the result is then verified with the copies."""
    a, b, ok = _cells(n)
    c = np.where(ok, n - a - b, 0)
    coef = np.where(ok, np.exp(gammaln(n + 1) - gammaln(a + 1)
                               - gammaln(b + 1) - gammaln(c + 1)), 0.0)
    rp = np.asarray(rp, float)[:, None, None]
    ro = np.asarray(ro, float)[:, None, None]
    rest = np.clip(1 - rp - ro, 0, None)
    return coef[None] * rp ** a[None] * ro ** b[None] * rest ** c[None]


# ============================================================
# Recovering the per-person rates the sweep was built from
# ============================================================

def count_rows_for(S, feat, n):
    return S[(S.feature == feat) & (S.n == n)
             & (S.family.isin(["count-absolute", "count-proportional"]))]


def recover_null_rates(S, checks):
    """For each feature and null: the one multiple of 1/40,000 that
    reproduces the sweep's COUNT-rule false-claim rates at N 10..20.
    (The sign rows are left out, so CHECK 3 tests the result
    out-of-sample.)"""
    banner("RECOVERING THE NO-EFFECT RATES FROM THE SWEEP")
    denom = 2 * N_TASTE_PEOPLE
    qs = np.arange(0, denom // 2 + 1) / denom
    err = {(f, nl): np.zeros(len(qs)) for f in FEATURES for nl in NULLS}
    for n in range(N_MIN, N_MAX + 1):
        pmn = sym_pmf_matrix(n, qs)
        for feat in FEATURES:
            R = count_rows_for(S, feat, n).drop_duplicates(["k", "m"])
            M = np.stack([rule_mask_count(n, int(r.k), int(r.m)).ravel()
                          for _, r in R.iterrows()], axis=1).astype(float)
            pred = pmn @ M
            for nl in NULLS:
                tgt = R[NULL_COL[nl]].to_numpy(float)[None, :]
                err[(feat, nl)] = np.maximum(
                    err[(feat, nl)], np.abs(pred - tgt).max(axis=1))
        del pmn
    out, rows = {}, []
    for feat in FEATURES:
        for nl in NULLS:
            e = err[(feat, nl)]
            i = int(np.argmin(e))
            second = float(np.partition(e, 1)[1])
            out[(feat, nl)] = qs[i]
            rows.append({"kind": "no effect", "feature": feat,
                         "null": nl, "weight_per_SD": np.nan,
                         "lapse": np.nan, "p_predicted": qs[i],
                         "p_opposite": qs[i],
                         "as_count": f"{i}/{denom}",
                         "fit_max_abs_err": float(e[i]),
                         "runner_up_err": second})
            print(f"  {feat:7s} {nl:8s} q = {qs[i]:.6f} ({i}/{denom})  "
                  f"fit {e[i]:.1e}, runner-up {second:.1e}")
    worst = max(r["fit_max_abs_err"] for r in rows)
    gap = min(r["runner_up_err"] for r in rows)
    checks.require("no-effect rates: one multiple of 1/40,000 per null "
                   "reproduces every count-rule false-claim rate",
                   worst <= EXACT_TOL and gap >= UNIQUE_GAP,
                   f"worst fit {worst:.1e}, nearest other candidate "
                   f"misses by {gap:.1e}")
    return out, rows


def recover_responder_rates(S, checks):
    """For each feature, weight and lapse: the one pair of multiples of
    1/600 that reproduces the sweep's COUNT-rule power at full
    response (every responder, so the sample is one trinomial)."""
    banner("RECOVERING THE RESPONDERS' RATES FROM THE SWEEP")
    D = N_DETECT_PEOPLE
    cp, co = np.meshgrid(np.arange(D + 1), np.arange(RESP_OPP_MAX + 1),
                         indexing="ij")
    keep = (cp + co) <= D
    cp, co = cp[keep], co[keep]
    pm = {n: pmf_pairs(n, cp / D, co / D).reshape(len(cp), -1)
          for n in RECOVER_N}
    out, rows = {}, []
    for feat in FEATURES:
        masks = {n: count_rows_for(S, feat, n).drop_duplicates(["k", "m"])
                 for n in RECOVER_N}
        M = {n: np.stack([rule_mask_count(n, int(r.k), int(r.m)).ravel()
                          for _, r in masks[n].iterrows()], axis=1)
                .astype(float) for n in RECOVER_N}
        for w in RULE_RESPONDER_WEIGHTS:
            for lam in LAPSE_RATES:
                col = power_col(w, 1.0, lam)
                err = np.zeros(len(cp))
                for n in RECOVER_N:
                    pred = pm[n] @ M[n]
                    tgt = masks[n][col].to_numpy(float)[None, :]
                    err = np.maximum(err, np.abs(pred - tgt).max(axis=1))
                i = int(np.argmin(err))
                second = float(np.partition(err, 1)[1])
                rp, ro = cp[i] / D, co[i] / D
                out[(feat, w, lam)] = (rp, ro)
                rows.append({"kind": "responder", "feature": feat,
                             "null": "", "weight_per_SD": w,
                             "lapse": lam, "p_predicted": rp,
                             "p_opposite": ro,
                             "as_count": f"{cp[i]}/{D}, {co[i]}/{D}",
                             "fit_max_abs_err": float(err[i]),
                             "runner_up_err": second})
                print(f"  {feat:7s} {w:.1f}/SD lapse {lam:.1f}  "
                      f"predicted {rp:.4f} ({cp[i]}/{D}), opposite "
                      f"{ro:.4f} ({co[i]}/{D})  fit {err[i]:.1e}")
    worst = max(r["fit_max_abs_err"] for r in rows)
    gap = min(r["runner_up_err"] for r in rows)
    checks.require("responder rates: one pair of multiples of 1/600 per "
                   "type reproduces every count-rule power at full "
                   "response", worst <= EXACT_TOL and gap >= UNIQUE_GAP,
                   f"worst fit {worst:.1e}, nearest other candidate "
                   f"misses by {gap:.1e}")
    return out, rows


def published_cross_check(q, checks):
    """The recovered spread rates against the 4-decimal values the power
    report prints (its prevalence table's assumed_taste_* alpha)."""
    if not os.path.isfile(POWER_REPORT):
        return None
    lines = open(POWER_REPORT, errors="replace").read().splitlines()
    found, feat = {}, None
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        if parts[0] in ("colour", "fog") and len(parts) > 2 and \
                parts[1].startswith("assumed_"):
            feat, parts = parts[0], parts[1:]
        elif not parts[0].startswith("assumed_"):
            continue
        if feat and parts[0].startswith("assumed_taste_"):
            sp = parts[0][len("assumed_taste_"):].rstrip("x")
            try:
                found[(feat, sp)] = float(parts[1])
            except (ValueError, IndexError):
                pass
    if not found:
        checks.info("4-decimal cross-check", "the prevalence table was "
                    "not found in power_report.txt; skipped")
        return None
    # within half a unit of the 4th decimal: a value on a rounding tie
    # (0.07785) may print either way, so no rounding is applied here
    bad = [(k, v, q[k]) for k, v in found.items()
           if k in q and abs(q[k] - v) > 0.00005 + 1e-12]
    checks.require("recovered rates agree with the 4-decimal values in "
                   "power_report.txt", not bad and len(found) == 8,
                   f"{len(found)} values compared"
                   + (f"; differ {bad}" if bad else ""))
    return found


# ============================================================
# The sample's joint (predicted, opposite) distribution
# ============================================================

def mixture_grid(n, share, resp, q_mid):
    """The sweep's own construction: int(round(share x n)) responders,
    the rest non-responders at the middle taste spread, convolved."""
    n_r = int(round(share * n))
    n_n = n - n_r
    g = trinomial_grid(n_r, *resp) if n_r else None
    h = trinomial_grid(n_n, q_mid, q_mid) if n_n else None
    if g is None:
        return h
    if h is None:
        return g
    return combine_grids(g, h)


def reproduce_sweep(S, sign, reg, qn, resp, crit, checks):
    """Every sign and registered-count value of the sweep, rebuilt from
    the recovered rates with the copied functions."""
    banner("REBUILDING THE SWEEP FROM THE RECOVERED RATES")
    worst = {"sign_false": 0.0, "count_false": 0.0, "sign_power": 0.0,
             "count_power": 0.0}
    grids = {}
    for feat in FEATURES:
        q_mid = qn[(feat, f"{NONRESPONDER_SPREAD_MULT:g}")]
        for n in range(N_MIN, N_MAX + 1):
            ms = rule_mask_sign(n, crit, SIGN_MIN_PREDICTED)
            r = reg[(feat, n)]
            mc = rule_mask_count(n, int(r.k), int(r.m))
            cr = reg_row(S, reg, feat, n)
            for nl in NULLS:
                g = trinomial_grid(n, qn[(feat, nl)], qn[(feat, nl)])
                worst["sign_false"] = max(worst["sign_false"], abs(
                    g[ms].sum() - sign[(feat, n)][NULL_COL[nl]]))
                worst["count_false"] = max(worst["count_false"], abs(
                    g[mc].sum() - cr[NULL_COL[nl]]))
                # and the exact count-rule rate by the copied enumerator
                ex = rule_false_claim_exact(n, int(r.k), int(r.m),
                                            qn[(feat, nl)], qn[(feat, nl)])
                worst["count_false"] = max(worst["count_false"],
                                           abs(ex - cr[NULL_COL[nl]]))
            for w in RULE_RESPONDER_WEIGHTS:
                for lam in LAPSE_RATES:
                    for share in RULE_RESPONDER_SHARES:
                        G = mixture_grid(n, share, resp[(feat, w, lam)],
                                         q_mid)
                        grids[(feat, n, w, share, lam)] = G
                        col = power_col(w, share, lam)
                        worst["sign_power"] = max(
                            worst["sign_power"],
                            abs(G[ms].sum() - sign[(feat, n)][col]))
                        worst["count_power"] = max(
                            worst["count_power"],
                            abs(G[mc].sum() - cr[col]))
    for k, v in worst.items():
        print(f"  {k:12s} largest difference {v:.1e}")
    checks.require(
        "the recovered rates rebuild every sign and registered-count "
        "power and false-claim value of the sweep (sign rows "
        "out-of-sample)", max(worst.values()) <= EXACT_TOL,
        ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    return grids


# ============================================================
# Item 1: the sign-test table, with the count rule beside it
# ============================================================

def item1_table(S, T, sign, reg, checks):
    banner("ITEM 1: THE SIGN TEST, WITH THE COUNT RULE AS SECONDARY")
    rows = []
    for feat in FEATURES:
        for n in range(N_MIN, N_MAX + 1):
            sr = sign[(feat, n)]
            r = reg[(feat, n)]
            cr = reg_row(S, reg, feat, n)
            tc = T[(T.feature == feat) & (T.n == n)
                   & (T.rule_kind == "count")]
            if len(tc) != 1:
                raise CheckFailed(f"registration_table.csv: {len(tc)} "
                                  f"count rows for {feat} N={n}")
            tc = tc.iloc[0]
            for role, rule, src in (("primary", "sign", sr),
                                    ("secondary", "count", cr)):
                rec = {"feature": feat, "claim": CLAIM[feat], "n": n,
                       "role": role,
                       "rule": (f"sign test, alpha {ALPHA:g}, minimum "
                                f"predicted {SIGN_MIN_PREDICTED}"
                                if rule == "sign" else
                                f"count: k >= {int(r.k)}, m <= {int(r.m)}"),
                       "source": ("rule_sweep.csv" if rule == "sign" else
                                  "registration_table.csv; per-null "
                                  "rates and lapse 0.2 from the "
                                  "rule_sweep.csv row of the registered "
                                  "rule")}
                for nl in NULLS:
                    c = NULL_COL[nl]
                    rec[c] = float(src[c])
                    rec[c + "_se"] = float(src[c + "_se"])
                    rec[c + "_upper"] = float(src[c + "_upper"])
                rec["false_worst"] = max(rec[NULL_COL[x]] for x in NULLS)
                rec["false_worst_upper"] = max(
                    rec[NULL_COL[x] + "_upper"] for x in NULLS)
                for cell in CELLS:
                    w = cell_weight(cell, feat)
                    _, share, lapse = CELLS[cell]
                    col = power_col(w, share, lapse)
                    if rule == "count" and REGTABLE_POWER_COL[cell]:
                        rec[f"power_{cell}"] = float(
                            tc[REGTABLE_POWER_COL[cell]])
                    else:
                        rec[f"power_{cell}"] = float(src[col])
                rec["power_cost_planning_qtr"] = rec["power_quarter"]
                if rule == "count":
                    rec["power_cost_planning_qtr"] = float(
                        tc["cost_planning_qtr"])
                rows.append(rec)
    A = pd.DataFrame(rows)

    # CHECK 2, second part: the sign rows against registration_table.csv
    tsg = T[T.rule_kind == "sign"]
    pairs = [("false_worst", "false_worst", 5),
             ("false_worst_upper", "false_worst_upper", 5),
             ("false_guesser", "false_guesser", 5),
             ("power_planning", "power_planning_s0.5_l0.1", 4),
             ("power_cost_lower_half", "cost_lower_half", 4),
             ("power_quarter", "power_s0.25_l0.1", 4),
             ("power_quarter", "cost_planning_qtr", 4),
             ("power_all_respond", "power_s1_l0.1", 4),
             ("power_planning_lapse0", "power_s0.5_l0", 4)]
    bad, n_cmp = [], 0
    for feat in FEATURES:
        for n in range(N_MIN, N_MAX + 1):
            t = tsg[(tsg.feature == feat) & (tsg.n == n)]
            if len(t) != 1 or int(t.iloc[0].k) != SIGN_MIN_PREDICTED:
                bad.append(f"{feat} N{n}: registration_table sign row "
                           "missing or not minimum predicted "
                           f"{SIGN_MIN_PREDICTED}")
                continue
            t = t.iloc[0]
            a = A[(A.feature == feat) & (A.n == n)
                  & (A.role == "primary")].iloc[0]
            for mine, theirs, dp in pairs:
                n_cmp += 1
                if abs(a[mine] - float(t[theirs])) > 0.5 * 10 ** -dp \
                        + 1e-12:
                    bad.append(f"{feat} N{n} {theirs}: {a[mine]:.6f} vs "
                               f"{float(t[theirs])}")
    checks.require("sign rows equal the sign rows of "
                   "registration_table.csv (to its rounding)", not bad,
                   f"{n_cmp - len(bad)}/{n_cmp} values"
                   + (f"; {bad[:3]}" if bad else ""))

    # the count rule: registration_table.csv against the sweep row
    bad, n_cmp = [], 0
    for feat in FEATURES:
        for n in range(N_MIN, N_MAX + 1):
            r = reg[(feat, n)]
            cr = reg_row(S, reg, feat, n)
            tc = T[(T.feature == feat) & (T.n == n)
                   & (T.rule_kind == "count")].iloc[0]
            for cell, col in REGTABLE_POWER_COL.items():
                if col is None:
                    continue
                n_cmp += 1
                w = cell_weight(cell, feat)
                _, share, lapse = CELLS[cell]
                if abs(float(cr[power_col(w, share, lapse)])
                       - float(tc[col])) > 0.5e-4 + 1e-12:
                    bad.append(f"{feat} N{n} {col}")
            n_cmp += 1
            if abs(float(cr.worst_upper_strict)
                   - float(tc.false_worst_upper)) > 0.5e-5 + 1e-12:
                bad.append(f"{feat} N{n} false_worst_upper")
    checks.require("count rule: registration_table.csv equals the sweep "
                   "row of the registered rule (to its rounding)", not bad,
                   f"{n_cmp - len(bad)}/{n_cmp} values"
                   + (f"; {bad[:3]}" if bad else ""))
    gap = float((T["power_s0.25_l0.1"] - T["cost_planning_qtr"]).abs().max())
    checks.require("the quarter-share cell: registration_table's "
                   "power_s0.25_l0.1 and cost_planning_qtr agree",
                   gap <= 1e-12, f"largest gap {gap:.1e} over "
                   f"{len(T)} rows")

    for feat in FEATURES:
        print(f"\n  {CLAIM[feat]}: planning-cell power, sign / count")
        s = A[(A.feature == feat)].pivot(index="n", columns="role",
                                         values="power_planning")
        print("   " + "  ".join(f"N{n}: {s.loc[n, 'primary']:.3f}/"
                                f"{s.loc[n, 'secondary']:.3f}"
                                for n in s.index))
    return A


# ============================================================
# Item 2: the decision table
# ============================================================

def item2_decision(crit, checks):
    banner("ITEM 2: THE SIGN TEST'S DECISION TABLE  (CHECK 5)")
    rows, bad = [], []
    for d in range(1, DECISION_D_MAX + 1):
        c = crit[d]
        rec = {"detected_d": d}
        if not np.isfinite(c):
            p_all = float(stats.binom.sf(d - 1, d, 0.5))
            rec.update({"min_predicted": None, "max_opposite": None,
                        "p_at_boundary": None,
                        "p_one_fewer": None,
                        "p_if_all_predicted": p_all,
                        "reads": f"no claim possible (even {d} of {d} "
                                 f"gives p = {p_all:.4f})"})
            if p_all <= ALPHA:
                bad.append(f"d={d}: {d}/{d} would reach alpha")
            try:
                sp = stats.binomtest(d, d, 0.5,
                                     alternative="greater").pvalue
                if abs(sp - p_all) > EXACT_TOL:
                    bad.append(f"d={d}: scipy {sp} vs {p_all}")
            except AttributeError:
                pass
        else:
            c = int(c)
            p = float(stats.binom.sf(c - 1, d, 0.5))
            p1 = float(stats.binom.sf(c - 2, d, 0.5)) if c >= 1 else None
            sp = float(stats.binomtest(c, d, 0.5,
                                       alternative="greater").pvalue)
            sp1 = (float(stats.binomtest(c - 1, d, 0.5,
                                         alternative="greater").pvalue)
                   if c >= 1 else None)
            reads = ("all of them" if c == d else f"{c} of {d}")
            rec.update({"min_predicted": c, "max_opposite": d - c,
                        "p_at_boundary": p, "p_one_fewer": p1,
                        "p_if_all_predicted": float(0.5 ** d),
                        "p_scipy_binomtest": sp, "reads": reads})
            if abs(sp - p) > EXACT_TOL:
                bad.append(f"d={d}: p {p} vs scipy {sp}")
            if p > ALPHA:
                bad.append(f"d={d}: boundary p {p} above alpha")
            if sp1 is not None and sp1 <= ALPHA:
                bad.append(f"d={d}: one fewer also reaches alpha")
        rows.append(rec)
    D = pd.DataFrame(rows)
    for r in D.itertuples():
        print(f"  d = {r.detected_d:2d}  {r.reads}"
              + (f"   p = {r.p_at_boundary:.4f}"
                 if r.p_at_boundary == r.p_at_boundary
                 and r.p_at_boundary is not None else ""))
    first = int(D.dropna(subset=["min_predicted"]).detected_d.min())
    checks.require("decision table reproduces scipy's one-sided binomial "
                   "p-values, and every boundary is the smallest that "
                   "works", not bad, f"d = 1..{DECISION_D_MAX}"
                   + (f"; {bad[:3]}" if bad else ""))
    checks.require("the sign test can fire from "
                   f"{FIRE_MIN_DETECTIONS} detections and not below",
                   first == FIRE_MIN_DETECTIONS, f"first at d = {first}")
    return D


# ============================================================
# Item 3: exact vs simulated, and the worst case over q
# ============================================================

def item3_exact(sign, reg, qn, crit, checks):
    banner("ITEM 3: THE EXACT FALSE-CLAIM RATE  (CHECKS 3 AND 4)")
    rows = []
    for feat in FEATURES:
        for n in range(N_MIN, N_MAX + 1):
            ms = rule_mask_sign(n, crit, SIGN_MIN_PREDICTED)
            sr = sign[(feat, n)]
            for nl in NULLS + ["nominal"]:
                q = (NOMINAL_GUESSER_Q if nl == "nominal"
                     else qn[(feat, nl)])
                ex = float(trinomial_grid(n, q, q)[ms].sum())
                rec = {"feature": feat, "n": n,
                       "null": ("guesser, nominal 2.5%" if nl == "nominal"
                                else ("guesser, realised" if nl ==
                                      "guesser" else f"taste {nl}x")),
                       "q_per_direction": q, "exact": ex}
                if nl != "nominal":
                    sim = float(sr[NULL_COL[nl]])
                    se = float(sr[NULL_COL[nl] + "_se"])
                    rec.update({"sweep": sim, "sweep_se": se,
                                "sweep_upper": float(
                                    sr[NULL_COL[nl] + "_upper"]),
                                "diff": ex - sim,
                                "z": (ex - sim) / se if se > 0 else 0.0})
                rows.append(rec)
    E = pd.DataFrame(rows)
    d = E.dropna(subset=["sweep"])
    worst = float(d["diff"].abs().max())
    zmax = float(d.z.abs().max())
    checks.require("exact trinomial rate equals the sweep's sign rate at "
                   "every N and null", worst <= EXACT_TOL,
                   f"{len(d)} values; largest |difference| {worst:.1e}, "
                   f"largest |z| against the sweep's SE {zmax:.1e}")
    return E


def worst_over_q(n, mask):
    """The largest P(claim) over q in [0, 0.5]: a grid, then a bounded
    refinement of the best grid point with the copied trinomial_grid."""
    qs = np.linspace(0.0, 0.5, Q_GRID)
    v = sym_claim_curve(n, mask, qs)
    i = int(np.argmax(v))
    lo, hi = qs[max(i - 1, 0)], qs[min(i + 1, len(qs) - 1)]
    f = lambda q: -float(trinomial_grid(n, q, q)[mask].sum())
    best_q, best = qs[i], -f(qs[i])
    if hi > lo:
        r = optimize.minimize_scalar(f, bounds=(lo, hi), method="bounded",
                                     options={"xatol": 1e-12})
        if -r.fun > best:
            best_q, best = float(r.x), float(-r.fun)
    grid_vs_copy = abs(v[i] - (-f(qs[i])))
    return best, best_q, grid_vs_copy


def item3_worst(reg, crit, checks):
    rows, gvc = [], 0.0
    for feat in FEATURES:
        for n in range(N_MIN, N_MAX + 1):
            ms = rule_mask_sign(n, crit, SIGN_MIN_PREDICTED)
            r = reg[(feat, n)]
            mc = rule_mask_count(n, int(r.k), int(r.m))
            rec = {"feature": feat, "n": n,
                   "count_rule": f"k >= {int(r.k)}, m <= {int(r.m)}"}
            for lab, m in (("sign", ms), ("count", mc),
                           ("either", ms | mc)):
                w, q, g = worst_over_q(n, m)
                gvc = max(gvc, g)
                rec[f"{lab}_worst_over_q"] = w
                rec[f"{lab}_q_at_worst"] = q
            # the bound: given d detections the test's size is at most
            # max_d P(Bin(d, .5) >= crit_d), for every q
            sizes = [float(stats.binom.sf(int(crit[d]) - 1, d, 0.5))
                     for d in range(1, n + 1) if np.isfinite(crit[d])]
            rec["sign_bound_max_size_given_d"] = max(sizes)
            rows.append(rec)
    Wq = pd.DataFrame(rows)
    checks.require("the vectorised enumeration agrees with the copied "
                   "trinomial_grid", gvc <= EXACT_TOL, f"{gvc:.1e}")
    over = Wq[Wq.sign_worst_over_q > ALPHA + 1e-12]
    checks.require(f"sign test: worst case over all q is at or below "
                   f"{ALPHA} at every N", over.empty,
                   f"largest {Wq.sign_worst_over_q.max():.4f} "
                   f"(N = {int(Wq.loc[Wq.sign_worst_over_q.idxmax(), 'n'])})"
                   + ("" if over.empty else
                      f"; above alpha at N {sorted(over.n.unique())}"))
    above = Wq[Wq.sign_worst_over_q > Wq.sign_bound_max_size_given_d
               + 1e-12]
    checks.require("sign test: worst case never exceeds the largest size "
                   "given d", above.empty)
    print("  N   sign worst (q)        count worst (q)       either")
    for feat in FEATURES:
        for r in Wq[Wq.feature == feat].itertuples():
            print(f"  {feat:6s} {r.n:2d}  {r.sign_worst_over_q:.4f} "
                  f"({r.sign_q_at_worst:.3f})   "
                  f"{r.count_worst_over_q:.4f} ({r.count_q_at_worst:.3f})"
                  f"   {r.either_worst_over_q:.4f}")
    return Wq


# ============================================================
# Item 4: how often the sign test can fire at all
# ============================================================

def item4_firing(grids, qn, crit, checks):
    banner("ITEM 4: P(AT LEAST 5 DETECTIONS)")
    rows = []
    for feat in FEATURES:
        for n in range(N_MIN, N_MAX + 1):
            ms = rule_mask_sign(n, crit, SIGN_MIN_PREDICTED)
            a = np.arange(n + 1)[:, None]
            b = np.arange(n + 1)[None, :]
            d = a + b
            scen = []
            for cell in CELLS:
                w = cell_weight(cell, feat)
                _, share, lapse = CELLS[cell]
                scen.append((cell, f"{w:g}/SD", share, lapse,
                             grids[(feat, n, w, share, lapse)]))
            for nl in ("0.5", "1"):
                q = qn[(feat, nl)]
                scen.append((f"no effect, taste {nl}x", "", np.nan,
                             np.nan, trinomial_grid(n, q, q)))
            for name, wlab, share, lapse, G in scen:
                p5 = float(G[d >= FIRE_MIN_DETECTIONS].sum())
                pw = float(G[ms].sum())
                rows.append({"feature": feat, "n": n, "scenario": name,
                             "weight": wlab, "share": share,
                             "lapse": lapse,
                             "p_at_least_5_detections": p5,
                             "expected_detections": float((G * d).sum()),
                             "sign_power": pw,
                             "sign_power_given_5_plus": (pw / p5 if p5 > 0
                                                         else np.nan)})
    F = pd.DataFrame(rows)
    for feat in FEATURES:
        s = F[(F.feature == feat) & (F.scenario == "planning")]
        print(f"  {feat:7s} planning: " + " ".join(
            f"{r.n}:{r.p_at_least_5_detections:.3f}"
            for r in s.itertuples()))
    checks.info("item 4 derivation",
                "exact, from the per-person rates recovered from the "
                "sweep (the same rates that rebuild its power); no "
                "simulation")
    return F


# ============================================================
# Item 5: primary and secondary together, with no effect
# ============================================================

def item5_union(reg, qn, crit, Wq, checks):
    banner("ITEM 5: PRIMARY AND SECONDARY TOGETHER, NO EFFECT")
    rows = []
    for feat in FEATURES:
        for n in range(N_MIN, N_MAX + 1):
            ms = rule_mask_sign(n, crit, SIGN_MIN_PREDICTED)
            r = reg[(feat, n)]
            mc = rule_mask_count(n, int(r.k), int(r.m))
            for nl in NULLS + ["nominal"]:
                q = (NOMINAL_GUESSER_Q if nl == "nominal"
                     else qn[(feat, nl)])
                g = trinomial_grid(n, q, q)
                rows.append({"feature": feat, "n": n,
                             "null": ("guesser, nominal 2.5%"
                                      if nl == "nominal" else
                                      ("guesser, realised" if nl ==
                                       "guesser" else f"taste {nl}x")),
                             "q_per_direction": q,
                             "count_rule": f"k >= {int(r.k)}, "
                                           f"m <= {int(r.m)}",
                             "p_sign": float(g[ms].sum()),
                             "p_count": float(g[mc].sum()),
                             "p_either": float(g[ms | mc].sum()),
                             "p_both": float(g[ms & mc].sum())})
            w = Wq[(Wq.feature == feat) & (Wq.n == n)].iloc[0]
            rows.append({"feature": feat, "n": n,
                         "null": "worst over q in [0, 0.5], each "
                                 "column at its own q",
                         "q_per_direction": np.nan,
                         "count_rule": f"k >= {int(r.k)}, m <= {int(r.m)}",
                         "p_sign": w.sign_worst_over_q,
                         "p_count": w.count_worst_over_q,
                         "p_either": w.either_worst_over_q,
                         "p_both": np.nan})
    U = pd.DataFrame(rows)
    cal = U[U.null.str.startswith(("guesser", "taste"))]
    bad = 0.0
    for r in cal.itertuples():
        bad = max(bad, abs(r.p_either - (r.p_sign + r.p_count - r.p_both)))
    checks.require("P(either) = P(sign) + P(count) - P(both) in every "
                   "cell", bad <= EXACT_TOL, f"{bad:.1e}")
    over = cal[cal.p_either > TARGET_FALSE]
    i = cal.p_either.idxmax()
    checks.info("P(either) within the calibrated nulls",
                f"worst {cal.loc[i, 'p_either']:.4f} ({cal.loc[i, 'feature']}"
                f", N = {cal.loc[i, 'n']}, {cal.loc[i, 'null']}); "
                f"{len(over)} cell(s) above {TARGET_FALSE}")
    return U


# ============================================================
# Item 6: the figure
# ============================================================

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)


def item6_figure(A, Wq, path):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.4), sharex=True)
    ns = list(range(N_MIN, N_MAX + 1))
    for j, feat in enumerate(FEATURES):
        P = A[(A.feature == feat) & (A.role == "primary")].sort_values("n")
        Cn = A[(A.feature == feat)
               & (A.role == "secondary")].sort_values("n")
        ax = axes[0, j]
        for i, (cell, lab) in enumerate((
                ("all_respond", "all respond"),
                ("planning", "planning: half respond"),
                ("quarter", "a quarter respond"))):
            ax.plot(P.n, P[f"power_{cell}"], color=SERIES_COLOURS[i],
                    marker=SERIES_MARKERS[i], ls=SERIES_STYLES[i], lw=1.8,
                    ms=5, label=f"sign test: {lab}")
        ax.plot(Cn.n, Cn.power_planning, color=SERIES_COLOURS[3],
                marker=SERIES_MARKERS[3], ms=3.5, ls="--", lw=1.0,
                label="count rule (secondary): planning")
        w = PLANNING_WEIGHT[feat]
        ax.set_title(f"{CLAIM[feat]}: power  ({w:g}/SD, lapse "
                     f"{PLANNING_LAPSE:g})", fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("P(claim supported)" if j == 0 else "")
        _style(ax)

        ax = axes[1, j]
        ax.errorbar(P.n, P.false_worst,
                    yerr=[np.zeros(len(P)), P.false_worst_upper
                          - P.false_worst],
                    color=SERIES_COLOURS[0], marker=SERIES_MARKERS[0],
                    ls=SERIES_STYLES[0], lw=1.8, ms=5, capsize=3,
                    label="sign test: worst simulated null "
                          "(bar to its 95% upper bound)")
        Wf = Wq[Wq.feature == feat].sort_values("n")
        ax.plot(Wf.n, Wf.sign_worst_over_q, color=SERIES_COLOURS[1],
                marker=SERIES_MARKERS[1], ls=SERIES_STYLES[1], lw=1.8,
                ms=5, label="sign test: exact worst case, any symmetric "
                            "null")
        ax.plot(Wf.n, Wf.count_worst_over_q, color=SERIES_COLOURS[3],
                marker=SERIES_MARKERS[3], ms=3.5, ls="--", lw=1.0,
                label="count rule (secondary): exact worst case, any "
                      "symmetric null")
        ax.axhline(TARGET_FALSE, color="#555555", lw=0.9, ls=":")
        ax.text(N_MAX + 0.3, TARGET_FALSE, "5%", va="center",
                fontsize=8, color="#555555")
        ax.set_title(f"{CLAIM[feat]}: false-claim rate", fontsize=10)
        ax.set_xlabel("final N (report group)")
        ax.set_ylabel("P(claim), no effect" if j == 0 else "")
        ax.set_ylim(0, max(0.16, float(Wq.count_worst_over_q.max())
                           * 1.08))
        ax.set_xticks(ns)
        _style(ax)
    fig.suptitle("Primary rule: the sign test (one-sided exact binomial, "
                 f"alpha {ALPHA:g}).  Secondary: the registered count "
                 "rule.", fontsize=10, y=0.985)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.165,
                        hspace=0.32, wspace=0.16)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=2, fontsize=8, frameon=False)
    h, lab = axes[1, 0].get_legend_handles_labels()
    # simulated first, then the two exact worst cases
    order = sorted(range(len(lab)),
                   key=lambda i: (not lab[i].startswith("sign test: worst"),
                                  lab[i].startswith("count")))
    fig.legend([h[i] for i in order], [lab[i] for i in order],
               loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=1,
               fontsize=8, frameon=False)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


# ============================================================
# Item 7: the report
# ============================================================

def write_report(path, R, checks, meta):
    L = []
    W = L.append
    W("SIGN TEST AS THE PRIMARY RULE FOR H1 AND H2  "
      "(sign_test_registration_v4)")
    W("=" * 74)
    W(f"run      {meta['started']}  ({meta.get('elapsed_s', 0):.1f} s)")
    W(f"output   {meta['output']}")
    W(f"python   {meta['python']}, numpy {meta['numpy']}, scipy "
      f"{meta['scipy']}, pandas {meta['pandas']}")
    W("")
    W("WHAT THIS IS")
    W("-" * 74)
    W("Primary rule for H1 (fog) and H2 (colour): the sign test (Dixon & "
      "Mood, 1946).")
    W("Among the participants detected in either direction, the number "
      "detected in the")
    W(f"predicted direction is tested against Binomial(d, 0.5), "
      f"one-sided, exact, alpha {ALPHA:g},")
    W("with no floor on the number predicted. Secondary, registered rule: "
      "the count rule")
    W("of combined_rule.csv, unchanged. Thresholds, scenarios and nulls "
      "are unchanged.")
    W("")
    W("INPUTS")
    W("-" * 74)
    for p, h in meta.get("input_sha256", {}).items():
        W(f"  {p}")
        W(f"      sha256 {h}" if h else "      not present (optional)")
    W("")
    W("WHAT WAS READ, DERIVED AND SIMULATED")
    W("-" * 74)
    W("  Read      rule_sweep.csv: the sign rows (minimum predicted 0) "
      "and the rows of")
    W("            the registered count rule; registration_table.csv: "
      "the count rule's")
    W("            figures; combined_rule.csv: its (k, m) per N.")
    W("  Derived   the per-person rates the sweep was built from, "
      "recovered exactly from")
    W("            its count-rule rows (no-effect rates as multiples of "
      "1/40,000;")
    W("            responders' rates as multiples of 1/600); from them, "
      "by exact")
    W("            enumeration: the exact false-claim rates and their "
      "worst case over q")
    W("            (item 3), P(at least 5 detections) (item 4), and the "
      "joint rates of")
    W("            the two rules (item 5).")
    W("  Simulated NOTHING.")
    W("")
    W("CHECKS")
    W("-" * 74)
    for r in checks.rows:
        W(f"  [{r['result']}] {r['check']}"
          + (f"  ({r['detail']})" if r["detail"] else ""))
    if R.get("failed"):
        W("")
        W(f"THE RUN STOPPED: {R['failed']}")
        W("Nothing after the failed check was computed.")
        open(path, "w").write("\n".join(L) + "\n")
        return
    rates = R["rates"]
    A, D, E, Wq, F, U = (R[k] for k in ("A", "D", "E", "Wq", "F", "U"))
    W("")
    W("THE PER-PERSON RATES, RECOVERED FROM THE SWEEP")
    W("-" * 74)
    W("No effect: P(predicted) = P(opposite) = q per direction. The "
      "review's 'about")
    W("0.182 (colour) and 0.186 (fog) at the worst spread' are the "
      "taste-1x rows:")
    W("the symmetrised rates of classification_rates() in the power "
      "script, printed")
    W("to 4 decimals in its report (prevalence section, assumed_taste_1x "
      "alpha).")
    nr = rates[rates.kind == "no effect"]
    W(nr[["feature", "null", "p_predicted", "as_count",
           "fit_max_abs_err"]].rename(columns={"p_predicted": "q"})
      .to_string(index=False))
    W("")
    W("Responders (used by item 4 and by the rebuilt power):")
    rr = rates[rates.kind == "responder"]
    W(rr[["feature", "weight_per_SD", "lapse", "p_predicted",
          "p_opposite", "as_count"]].to_string(index=False))
    W("")
    W("1. THE SIGN TEST (PRIMARY) WITH THE COUNT RULE (SECONDARY)")
    W("-" * 74)
    W("False-claim rate: the worst of the five no-effect models (random "
      "choosers and")
    W("tastes 0.25, 0.5, 0.75, 1 x threshold), point and 95% upper bound. "
      "Power at")
    W(f"lapse {PLANNING_LAPSE:g} unless stated; weights per SD: planning "
      f"colour {PLANNING_WEIGHT['colour']:g} / fog "
      f"{PLANNING_WEIGHT['fog']:g},")
    W(f"weaker pair colour {ROBUSTNESS_WEIGHT['colour']:g} / fog "
      f"{ROBUSTNESS_WEIGHT['fog']:g}. 'quarter' is also the cost cell")
    W("'planning weights at a quarter share' (one cell, two labels). "
      "Every column")
    W("is in sign_test_table.csv.")
    cols = ["n", "false_worst", "false_worst_upper", "power_planning",
            "power_cost_lower_half", "power_quarter", "power_all_respond",
            "power_planning_lapse0", "power_planning_lapse0.2"]
    short = {"false_worst": "false", "false_worst_upper": "upper",
             "power_planning": "planning", "power_cost_lower_half":
             "weaker", "power_quarter": "quarter", "power_all_respond":
             "all", "power_planning_lapse0": "plan_l0",
             "power_planning_lapse0.2": "plan_l0.2"}
    for feat in FEATURES:
        for role, lab in (("primary", "sign test (primary)"),
                          ("secondary", "count rule (secondary)")):
            s = A[(A.feature == feat) & (A.role == role)].sort_values("n")
            W(f"\n  {CLAIM[feat]}, {lab}"
              + ("" if role == "primary" else ": " + ", ".join(
                  f"N{r.n} {r.rule[7:]}" for r in s.itertuples()
                  if r.n in (N_MIN, 15, N_MAX)) + " ..."))
            W(s[cols].rename(columns=short).round(4).to_string(
                index=False))
    W("")
    W("2. THE DECISION TABLE")
    W("-" * 74)
    W("d = participants detected in either direction. 'Needs' is the "
      "smallest number")
    W("in the predicted direction that supports the claim; p is the exact "
      "one-sided")
    W("binomial p at that boundary (scipy.stats.binomtest agrees).")
    for r in D.itertuples():
        if r.min_predicted is None or r.min_predicted != r.min_predicted:
            W(f"  d = {r.detected_d:2d}   {r.reads}")
        else:
            W(f"  d = {r.detected_d:2d}   needs {int(r.min_predicted):2d} "
              f"({r.reads}; at most {int(r.max_opposite)} opposite)   "
              f"p = {r.p_at_boundary:.4f}")
    W("")
    W("3. EXACT FALSE-CLAIM RATES, AND THE WORST CASE OVER q")
    W("-" * 74)
    W("The sweep's rates are themselves exact enumerations at simulated "
      "per-person")
    W("rates, so at the same q the exact value agrees to floating-point "
      "precision")
    W("(CHECK 3) -- far inside the Monte Carlo error, which belongs to q.")
    for feat in FEATURES:
        s = E[E.feature == feat]
        W(f"\n  {CLAIM[feat]}: exact (= sweep) by N; columns are the "
          "no-effect models")
        piv = s.pivot(index="n", columns="null", values="exact")
        W(piv.round(5).to_string())
    W("")
    W("  Worst case over EVERY symmetric no-effect model (q in [0, 0.5]). "
      "For the sign")
    W("  test this is guaranteed: given d detections its size is at most "
      "alpha. The")
    W("  count rule (secondary) is calibrated only at the five models "
      "above; outside")
    W("  them it is not bounded by 5%.")
    for feat in FEATURES:
        s = Wq[Wq.feature == feat]
        W(f"\n  {CLAIM[feat]}")
        W(s[["n", "sign_worst_over_q", "sign_q_at_worst",
             "count_rule", "count_worst_over_q", "count_q_at_worst",
             "either_worst_over_q"]].round(4).to_string(index=False))
    W("")
    W("4. HOW OFTEN THE SIGN TEST CAN FIRE AT ALL")
    W("-" * 74)
    W(f"P(at least {FIRE_MIN_DETECTIONS} detections, both directions "
      "together). Exact, from the recovered")
    W("per-person rates in the sweep's own mixture of responders and "
      "taste-0.5x")
    W("non-responders; nothing simulated.")
    for feat in FEATURES:
        s = F[F.feature == feat]
        piv = s.pivot(index="n", columns="scenario",
                      values="p_at_least_5_detections")
        order = [c for c in list(CELLS) + ["no effect, taste 0.5x",
                                           "no effect, taste 1x"]
                 if c in piv.columns]
        W(f"\n  {CLAIM[feat]}")
        W(piv[order].round(3).to_string())
    W("")
    W("5. PRIMARY AND SECONDARY TOGETHER, WITH NO EFFECT")
    W("-" * 74)
    W("P(sign test supports), P(count rule supports), P(either), P(both), "
      "exactly.")
    for feat in FEATURES:
        s = U[(U.feature == feat)
              & U.null.isin(["taste 1x", "guesser, realised"])]
        piv = s.pivot(index="n", columns="null",
                      values=["p_sign", "p_count", "p_either"])
        W(f"\n  {CLAIM[feat]}  (random choosers and the widest calibrated "
          "taste; all models")
        W("  in primary_secondary_null.csv)")
        W(piv.round(4).to_string())
    W("")
    W("WHAT THE SIGN TEST GAINS AND COSTS RELATIVE TO THE COUNT RULE")
    W("-" * 74)
    for para in R["summary"]:
        W(para)
        W("")
    W("FILES")
    W("-" * 74)
    for f in sorted(os.listdir(meta["output"])):
        W(f"  {f}")
    open(path, "w").write("\n".join(L) + "\n")


def wrap(text, width=74):
    import textwrap
    return textwrap.fill(text, width)


def summary_paragraphs(A, Wq, F, U):
    P = A[A.role == "primary"].set_index(["feature", "n"])
    C = A[A.role == "secondary"].set_index(["feature", "n"])
    sw_cal = float(P.false_worst_upper.max())
    sw_any = float(Wq.sign_worst_over_q.max())
    n_any = int(Wq.loc[Wq.sign_worst_over_q.idxmax(), "n"])
    cw_cal = float(C.false_worst_upper.max())
    i = Wq.count_worst_over_q.idxmax()
    cw_any = float(Wq.loc[i, "count_worst_over_q"])
    cw_n, cw_q = int(Wq.loc[i, "n"]), float(Wq.loc[i, "count_q_at_worst"])
    out = []
    out.append(wrap(
        "GAINS. The sign test's false-claim rate does not depend on the "
        "taste spread: under any symmetric no-effect model it is at most "
        f"{sw_any:.4f} (N = {n_any}), and at most {sw_cal:.4f} (upper "
        "bound) under the five simulated models. The count rule holds "
        f"its 5% (upper bound at most {cw_cal:.4f}) only at the spreads "
        "it was calibrated against; outside them its rate reaches "
        f"{cw_any:.4f} (N = {cw_n}, q = {cw_q:.3f} per direction). The "
        "sign test is a standard, published test (Dixon & Mood, 1946); "
        "the count rule is this study's own construction, selected per N "
        "by simulation."))
    d, below = {}, {}
    for feat in FEATURES:
        diff = (P.loc[feat].power_planning - C.loc[feat].power_planning)
        d[feat] = (float(diff.min()), float(diff.max()))
        below[feat] = int((diff < 0).sum())
    n_all = N_MAX - N_MIN + 1
    if all(v == n_all for v in below.values()):
        where = "at every N in the range"
    else:
        where = ("at " + ", ".join(f"{below[f]} of {n_all} N on {f}"
                                   for f in FEATURES))
    lo_cells = []
    for cell, lab in (("cost_lower_half", "weaker pair, half respond"),
                      ("quarter", "planning weights, quarter respond")):
        for feat in FEATURES:
            diff = P.loc[feat][f"power_{cell}"] - \
                C.loc[feat][f"power_{cell}"]
            wins = [int(n) for n, v in diff.items() if v >= 0]
            if not wins:
                txt = "the count rule is more powerful at every N"
            elif wins == list(range(wins[0], N_MAX + 1)):
                txt = (f"the sign test is at least as powerful from "
                       f"N = {wins[0]}")
            else:
                txt = ("the sign test is at least as powerful at N = "
                       + ", ".join(map(str, wins)))
            lo_cells.append(f"{lab}, {feat}: {txt}")
    f10 = F[(F.scenario == "planning") & (F.n == N_MIN)]
    fire = ", ".join(f"{r.feature} {r.p_at_least_5_detections:.2f}"
                     for r in f10.itertuples())
    out.append(wrap(
        "COSTS. In the planning cell the sign test is less powerful than "
        f"the count rule {where}: the difference (sign minus count) "
        f"ranges {d['fog'][0]:+.3f} to {d['fog'][1]:+.3f} on fog and "
        f"{d['colour'][0]:+.3f} to {d['colour'][1]:+.3f} on colour. The "
        f"test cannot fire with fewer than {FIRE_MIN_DETECTIONS} "
        "detections, and with 5 to 7 it needs every detected participant "
        f"in the predicted direction; at N = {N_MIN} in the planning "
        f"cell, P(at least {FIRE_MIN_DETECTIONS} detections) is {fire}. "
        "In the two cost cells: " + "; ".join(lo_cells) + ". Its "
        "guarantee, like the count rule's calibration, holds for "
        "symmetric no-effect models only: neither rule is protected "
        "against a preference shared across people in the predicted "
        "direction."))
    cal = U[U.null.str.startswith(("guesser", "taste"))]
    j = cal.p_either.idxmax()
    anyq = U[U.null.str.startswith("worst")]
    k = anyq.p_either.idxmax()
    n_over = int((cal.p_either > TARGET_FALSE).sum())
    out.append(wrap(
        "PRIMARY AND SECONDARY TOGETHER. Each rule is calibrated on its "
        "own. With no effect, the probability that at least one of the "
        "two claims is made reaches "
        f"{cal.loc[j, 'p_either']:.4f} within the calibrated models "
        f"({cal.loc[j, 'feature']}, N = {cal.loc[j, 'n']}, "
        f"{cal.loc[j, 'null']}; {n_over} cell(s) above {TARGET_FALSE}), "
        f"and {anyq.loc[k, 'p_either']:.2f} over all symmetric models "
        f"({anyq.loc[k, 'feature']}, N = {anyq.loc[k, 'n']}, where the "
        "count rule alone reaches "
        f"{Wq.loc[(Wq.feature == anyq.loc[k, 'feature']) & (Wq.n == anyq.loc[k, 'n']), 'count_worst_over_q'].iloc[0]:.2f}). The two "
        "rules are not a family with joint control at 5%."))
    return out


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 62)
    print("SIGN TEST REGISTRATION  (v4, step 8c)")
    print("=" * 62)
    print("  reads the frozen outputs; runs no simulation")
    root = resolve_output_dir()
    print(f"  output: {root}")
    meta = {"script": os.path.basename(__file__),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output": root, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
            "simulated": "nothing"}
    checks, R = Checks(), {}
    try:
        check_inputs(checks, meta)
        identity_check(checks, root)
        S, C, T, reg = load_tables(checks)
        sign = sign_rows(S, checks)
        crit = sign_test_critical(max(SAMPLE_SIZES) + 1)
        qn, nrows = recover_null_rates(S, checks)
        published_cross_check(qn, checks)
        resp, rrows = recover_responder_rates(S, checks)
        rates = pd.DataFrame(nrows + rrows)
        rates.to_csv(os.path.join(root, "recovered_rates.csv"),
                     index=False)
        R["rates"] = rates
        meta["recovered_no_effect_rates"] = {f"{f}|{nl}": v
                                             for (f, nl), v in qn.items()}
        grids = reproduce_sweep(S, sign, reg, qn, resp, crit, checks)

        A = item1_table(S, T, sign, reg, checks)
        A.to_csv(os.path.join(root, "sign_test_table.csv"), index=False)
        D = item2_decision(crit, checks)
        D.to_csv(os.path.join(root, "decision_table.csv"), index=False)
        E = item3_exact(sign, reg, qn, crit, checks)
        E.to_csv(os.path.join(root, "exact_false_claim.csv"), index=False)
        Wq = item3_worst(reg, crit, checks)
        Wq.to_csv(os.path.join(root, "worst_case_over_q.csv"),
                  index=False)
        F = item4_firing(grids, qn, crit, checks)
        F.to_csv(os.path.join(root, "firing_probability.csv"),
                 index=False)
        U = item5_union(reg, qn, crit, Wq, checks)
        U.to_csv(os.path.join(root, "primary_secondary_null.csv"),
                 index=False)
        item6_figure(A, Wq, os.path.join(root,
                                         "sign_test_registration.png"))
        R.update({"A": A, "D": D, "E": E, "Wq": Wq, "F": F, "U": U,
                  "summary": summary_paragraphs(A, Wq, F, U)})
        banner("GAINS AND COSTS")
        for p in R["summary"]:
            print(p + "\n")
    except CheckFailed as e:
        R["failed"] = str(e)
        print(f"\nSTOPPED: a check failed.\n  {e}")
    meta["elapsed_s"] = time.time() - t0
    meta["checks"] = checks.rows
    meta["settings"] = settings_snapshot()
    meta["result"] = ("stopped: " + R["failed"]) if "failed" in R \
        else "complete"
    with open(os.path.join(root, "run_meta.json"), "w") as fh:
        json.dump(jsonable(meta), fh, indent=1)
    write_report(os.path.join(root, "sign_test_report.txt"), R, checks,
                 meta)
    print("\n" + "=" * 62)
    if "failed" in R:
        print("STOPPED WITH A FAILED CHECK -- see sign_test_report.txt")
        print("=" * 62)
        sys.exit(1)
    print(f"DONE in {meta['elapsed_s']:.1f}s -- every check passed.")
    print("=" * 62)


if __name__ == "__main__":
    main()
