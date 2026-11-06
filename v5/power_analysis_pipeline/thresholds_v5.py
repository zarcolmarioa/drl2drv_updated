"""
thresholds_v5.py  --  v5 power analysis, script 6: the per-person model and
                      its detection thresholds

WHAT THIS ANSWERS
    Each participant does two image-choice blocks. In each block, a Firth
    logistic regression without intercept gives two weights (saturation and
    contrast) from the analysed trials, and a weight is "detected" when it
    lies beyond the 2.5% tails of the weights that random choosers reach by
    luck. This script:

      1. REPRODUCES the per-set thresholds frozen in design.json (20,000
         random choosers per set, both block sizes), exactly, by rebuilding
         the inputs build_trials_v5.py used and its random stream; then
         again with an independent seed, within Monte Carlo error.
      2. COMPARES two per-person models:
           two-weight   the current model: saturation and contrast weights
           side         the same plus a side term (+1 when image A is on the
                        left, -1 when on the right), a nuisance term only:
                        its estimate is never a result
         on thresholds, precision, and detection at the planning weights
         with lapse 0.1.
      3. SIMULATES SIDE PREFERENCES: random choosers and planning-weight
         people with a fixed side bias of 0 to 1.5 logit towards either
         side, and a population whose side biases vary between people;
         for each model, the false-detection rate in each tail, the shift
         in each weight, and detection.
      4. CONFIRMS that the thresholds do not depend on the trial order
         (orders 1-4 of each set).
      5. RECOMMENDS one model by a rule fixed in SETTINGS before the run,
         and computes the detection curve for that model (the v5
         counterpart of the v4 preregistration's Figure 4).

PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES
    The planning weights (0.5 saturation, 0.7 contrast per corpus SD), the
    lapse rate (0.1), the side biases and their spread between people are
    assumptions chosen for planning. No image-choice data exist yet. The
    report labels every figure that depends on them.

THE MODELS
    For person i on trial t of a block, with x_sat, x_con the trial's
    predictors (pred_sat, pred_fog: A minus B in corpus SD units, positive
    = image A is the predicted choice) and s_t = +1 if A is on the left,
    -1 if on the right:
      two-weight   logit P(choose A) = w_sat x_sat + w_con x_con
      side         logit P(choose A) = w_sat x_sat + w_con x_con + b s_t
    b > 0 is a preference for the LEFT image. Simulated people lapse: with
    lapse rate l, P = (1 - l) logistic(.) + l / 2 (v4's lapse_p).
    Both models are fitted with Firth's bias-reduced logistic regression:
    the two-weight model with v4's registered fit_firth_batch, the side
    model with fit_firth_batch_k below (the same algorithm for any number
    of predictors, checked against v4's single-person fit_firth).

THE RULE FOR THE RECOMMENDATION (fixed in SETTINGS before the run)
    Register the side model only if BOTH hold:
      (a) the side bias matters for the two-weight model: at a side bias
          of RULE_SIDE_BIAS logit (either direction), the two-weight
          model's false-detection rate for random choosers exceeds
          RULE_TAIL_MAX in either tail, or a weight moves by more than
          RULE_WEIGHT_BIAS_MAX from its value without side bias (random
          choosers or planning-weight people), in some set and size;
      (b) the side model costs little: its detection at the planning
          weights (no side bias, lapse 0.1) is at most
          RULE_DETECTION_LOSS_MAX below the two-weight model's, in every
          set, size and feature.
    Otherwise keep the two-weight model and its frozen thresholds.

REUSED (imported, not copied; the SHA-256 of each imported file is recorded
and checked against the committed version)
    simulate_power_v4_n10_20.py
        fit_firth_batch   the registered estimator (two predictors)
        fit_firth         the single-person Firth fit, any number of
                          predictors: the reference for fit_firth_batch_k
        simulate_many     random choosers (used for the exact reproduction)
        simulate_lapsing  responses with lapses (reference for respond())
        FIRTH_ITER, FIRTH_TOL, FIRTH_MAX_STEP   the estimator's settings
    analyse_pair_capacity_v5.py
        Streams           named random generators (SHA-256 of the name);
                          build_trials_v5.py's thresholds used
                          Streams(42, "build_trials_v5", size)("thresholdsA")
    New here: fit_firth_batch_k (Firth for any number of predictors, many
    people at once), respond() (responses with a side bias), the side
    sweep, the order checks, the rule, the report and the figures.

INPUTS (relative to PROJECT_ROOT)
    data/frozen_trial_lists_v5/v5_150/ and v5_180/   the frozen packages:
        checksums.sha256, package/design.json, package/lists/<set>_o<k>.json
        (only the files read are required; images are not needed)
    simulate_power_v4_n10_20.py, analyse_pair_capacity_v5.py

OUTPUTS (one new folder per run under OUTPUT_ROOT, named after the key
settings plus a hash of all settings; never overwritten)
    report.txt                what was done, what was found, every check
    checks.csv                every check, PASS or FAIL
    thresholds.csv            thresholds per size, set, model and source
    precision_detection.csv   null spread, precision and detection at the
                              planning weights, per model
    side_sweep.csv            per side bias: tail rates, weight shifts,
                              detection, per model and person type
    order_check.csv           the order-invariance checks
    detection_curve.csv       detection against true weight (chosen model)
    detection_curve.png, side_sweep.png   (if MAKE_FIGURES)
    settings.json, run_meta.json

REQUIRES  Python 3.9+, numpy, pandas, scipy; matplotlib for the figures
RUN       python thresholds_v5.py        (about 5-10 minutes)
"""

# ============================================================
# SETTINGS
# ============================================================

# --- Where things are ---
#
# PROJECT_ROOT
#   The project folder. Every path below is relative to it. None = the
#   folder this script is in.
PROJECT_ROOT = None

# PACKAGES
#   Block size -> frozen package folder (the one holding checksums.sha256
#   and package/).
PACKAGES = {150: "data/frozen_trial_lists_v5/v5_150",
            180: "data/frozen_trial_lists_v5/v5_180"}
BLOCK_SIZES = [150, 180]
SETS = ("A", "B")
ORDERS = (1, 2, 3, 4)

# Imported scripts, and the SHA-256 of their committed versions (commit
# 644dbe6 of the repository). A different file still runs, but the check
# fails and the report says so.
V4_SIM_SCRIPT = "simulate_power_v4_n10_20.py"
CAPACITY_SCRIPT = "analyse_pair_capacity_v5.py"
EXPECTED_SHA256 = {
    V4_SIM_SCRIPT:
        "852f3b33eb349153fbff9f95c50e1cd3c4f84f68b204d79a23f020f471bd1730",
    CAPACITY_SCRIPT:
        "5a8483a2a388482295c998ffcbe91e4a0f2df70f668d472cc4718d3a138addc7",
}

OUTPUT_ROOT = "data/power_v5/thresholds_v5"
MAKE_FIGURES = True
FIG_DPI = 150

# --- Thresholds (the registered rule) ---
#
# N_THRESHOLD_PEOPLE / THRESHOLD_ALPHA
#   Random choosers per set and the share per tail: 20,000 and 2.5%, as in
#   v4 and design.json. Thresholds are percentiles of their weights.
N_THRESHOLD_PEOPLE = 20_000
THRESHOLD_ALPHA = 0.025

# REPRO_SEED / REPRO_STREAM_KEY / REPRO_STREAM_NAME
#   The stream build_trials_v5.py drew its random choosers from:
#   Streams(42, "build_trials_v5", size)("thresholds" + set), on the
#   analysed rows in item order. Reusing it must give design.json's numbers.
REPRO_SEED = 42
REPRO_STREAM_KEY = "build_trials_v5"
REPRO_STREAM_NAME = "thresholds{set}"

# REPRO_EXACT_TOL
#   The reproduction must match design.json within REPRO_EXACT_TOL (exact
#   apart from the last bits).
REPRO_EXACT_TOL = 1e-12

# QUANTILE_Z
#   A threshold from independent random choosers (another seed, or another
#   trial order) must agree with the registered one within QUANTILE_Z
#   standard errors of the difference. Each SE is estimated from the
#   choosers' own weights (distribution-free, from the order statistics
#   around the 2.5% / 97.5% points): about 0.003 for saturation and 0.005
#   for contrast per estimate, so about 0.004-0.007 for a difference of two
#   independent estimates. 4 rather than 3 because about 100 thresholds are
#   compared in one run. The largest absolute difference is reported too.
#   (A fixed tolerance of 0.01 was planned first; it is below the Monte
#   Carlo error of a contrast threshold difference and failed on noise.)
QUANTILE_Z = 4.0

# SET_THRESHOLD_TOL
#   The two matched sets' upper thresholds should agree within this
#   relative difference (build_trials_v5.py's check), for each model.
SET_THRESHOLD_TOL = 0.05

# --- People (PLANNING ASSUMPTIONS, not estimates) ---
#
# PLANNING_WEIGHTS / PLANNING_LAPSE
#   The v4 preregistration's planning weights per corpus SD (saturation,
#   contrast) and lapse rate.
PLANNING_WEIGHTS = (0.5, 0.7)
PLANNING_LAPSE = 0.1

# N_PEOPLE
#   Simulated people per cell for precision, detection, the side sweep and
#   the detection curve. At 10,000 the Monte Carlo SE of a rate is at most
#   0.005, and 0.0016 for a rate near 0.025.
N_PEOPLE = 10_000

# SIDE_BIASES / SIDE_POPULATION_SD / SIDE_LAPSE
#   Fixed side biases (logit; P(left) = 0.56, 0.62, 0.68, 0.73, 0.82 for a
#   person who otherwise does not care), each simulated towards the LEFT
#   (+) and towards the RIGHT (-), plus 0. The population row draws each
#   person's side bias from Normal(0, SIDE_POPULATION_SD). Everyone in the
#   sweep lapses at SIDE_LAPSE.
SIDE_BIASES = [0.25, 0.5, 0.75, 1.0, 1.5]
SIDE_POPULATION_SD = 0.25
SIDE_LAPSE = PLANNING_LAPSE

# --- The rule for the recommendation (see the docstring) ---
RULE_SIDE_BIAS = 0.5
RULE_TAIL_MAX = 0.035
RULE_WEIGHT_BIAS_MAX = 0.05
RULE_DETECTION_LOSS_MAX = 0.01

# --- The detection curve (chosen model only) ---
#
# CURVE_WEIGHTS / CURVE_LAPSES
#   True weight on ONE feature (the other 0), per corpus SD, and lapse
#   rates, as in the v4 preregistration's Figure 4. 0.3/0.5/0.7 are the
#   v4 robustness and planning weights.
CURVE_WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
CURVE_LAPSES = [0.0, 0.1, 0.2]

# --- Check tolerances ---
ESTIMATOR_TOL = 1e-9         # batch Firth against the reference fits
PERMUTATION_TOL = 1e-10      # one person's fit, rows permuted
MC_Z = 3.0                   # Monte Carlo checks: within MC_Z SE
UNCONVERGED_STEP = 1e-6      # a fit whose last step exceeds this did not
                             # settle (explore_block_length_v5's criterion)

# --- Seeds ---
#   This script's own streams: Streams(RANDOM_SEED, STREAM_KEY, size, set).
RANDOM_SEED = 42
STREAM_KEY = "thresholds_v5"

# ============================================================
# Code
# ============================================================

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

FEATURES = ("sat", "con")                       # column order of X
FEATURE_LABEL = {"sat": "saturation", "con": "contrast"}
MODELS = ("two-weight", "side")
MODEL_LABEL = {"two-weight": "two-weight model (current)",
               "side": "side model (two weights + side term)"}
ANALYSED_TYPES = ("sat_only", "fog_only", "together", "conflict")
TYPE_COUNT_KEY = {"sat_only": "sat_only", "fog_only": "fog_only",
                  "together": "together", "conflict": "conflict"}

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def root_path(*parts):
    root = PROJECT_ROOT or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, *parts)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

class Checks:
    """Every check, PASS or FAIL, in the order run."""

    def __init__(self):
        self.rows = []

    def add(self, section, what, ok, detail=""):
        status = "PASS" if bool(ok) else "FAIL"
        self.rows.append({"section": section, "check": what,
                          "status": status, "detail": str(detail)})
        log(f"{status}  [{section}] {what}  {detail}")
        return bool(ok)

    def frame(self):
        return pd.DataFrame(self.rows)

    @property
    def n_fail(self):
        return sum(r["status"] == "FAIL" for r in self.rows)


# ------------------------------------------------------------
# Settings, output folder
# ------------------------------------------------------------

SETTING_NAMES = [
    "PROJECT_ROOT", "PACKAGES", "BLOCK_SIZES", "SETS", "ORDERS",
    "V4_SIM_SCRIPT", "CAPACITY_SCRIPT", "EXPECTED_SHA256", "OUTPUT_ROOT",
    "MAKE_FIGURES", "FIG_DPI", "N_THRESHOLD_PEOPLE", "THRESHOLD_ALPHA",
    "REPRO_SEED", "REPRO_STREAM_KEY", "REPRO_STREAM_NAME", "REPRO_EXACT_TOL",
    "QUANTILE_Z", "SET_THRESHOLD_TOL", "PLANNING_WEIGHTS",
    "PLANNING_LAPSE", "N_PEOPLE", "SIDE_BIASES", "SIDE_POPULATION_SD",
    "SIDE_LAPSE", "RULE_SIDE_BIAS", "RULE_TAIL_MAX", "RULE_WEIGHT_BIAS_MAX",
    "RULE_DETECTION_LOSS_MAX", "CURVE_WEIGHTS", "CURVE_LAPSES",
    "ESTIMATOR_TOL", "PERMUTATION_TOL", "MC_Z", "UNCONVERGED_STEP",
    "RANDOM_SEED", "STREAM_KEY",
]


def settings_dict():
    g = globals()
    return {k: g[k] for k in SETTING_NAMES}


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def resolve_output_dir():
    s = settings_dict()
    h = hashlib.sha256(json.dumps(jsonable(s), sort_keys=True)
                       .encode()).hexdigest()[:8]
    name = (f"sizes{'-'.join(str(n) for n in BLOCK_SIZES)}"
            f"_choosers{N_THRESHOLD_PEOPLE}_people{N_PEOPLE}"
            f"_side{min(SIDE_BIASES):g}-{max(SIDE_BIASES):g}"
            f"-popsd{SIDE_POPULATION_SD:g}"
            f"_w{PLANNING_WEIGHTS[0]:g}-{PLANNING_WEIGHTS[1]:g}"
            f"_lapse{PLANNING_LAPSE:g}_seed{RANDOM_SEED}__s{h}")
    base = root_path(OUTPUT_ROOT, name)
    out, k = base, 1
    while os.path.exists(out):          # never overwrite an earlier run
        k += 1
        out = f"{base}_run{k}"
    os.makedirs(out)
    return out


# ------------------------------------------------------------
# Imports of the existing scripts (read-only)
# ------------------------------------------------------------

def load_module(path, name, ck):
    ok = os.path.exists(path)
    ck.add("inputs", f"{os.path.basename(path)} exists", ok, path)
    if not ok:
        raise SystemExit(f"missing {path}")
    digest = sha256_of(path)
    want = EXPECTED_SHA256.get(os.path.basename(path))
    ck.add("inputs", f"{os.path.basename(path)} is the committed version "
           "(SHA-256)", digest == want, digest[:16] + "...")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(mod)
    return mod, digest


# ------------------------------------------------------------
# The frozen packages
# ------------------------------------------------------------

def read_checksums(pkg_dir):
    rows = {}
    with open(os.path.join(pkg_dir, "checksums.sha256")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split(None, 1)
            rows[rel.lstrip("*").strip()] = digest
    return rows


def verify_package(size, ck, file_hashes):
    """Verify every non-image entry of checksums.sha256; the images are not
    committed and not needed. Every file this script reads must be listed."""
    pkg_dir = root_path(PACKAGES[size])
    sums = read_checksums(pkg_dir)
    images = [r for r in sums if r.startswith("package/stimuli/")]
    others = [r for r in sums if r not in images]
    bad = []
    for rel in others:
        p = os.path.join(pkg_dir, rel)
        if not os.path.exists(p) or sha256_of(p) != sums[rel]:
            bad.append(rel)
    ck.add(f"package {size}", "every non-image file matches checksums.sha256",
           not bad, f"{len(others) - len(bad)} of {len(others)} match; "
           f"{len(images)} image entries not checked (images not needed)"
           + (f"; mismatched or missing: {bad[:5]}" if bad else ""))
    needed = ["package/design.json"] + [
        f"package/lists/{b}_o{k}.json" for b in SETS for k in ORDERS]
    unlisted = [r for r in needed if r not in sums]
    ck.add(f"package {size}", "every file read here is listed in "
           "checksums.sha256", not unlisted,
           f"{len(needed)} files" + (f"; unlisted: {unlisted}"
                                     if unlisted else ""))
    for r in needed:
        file_hashes[f"{PACKAGES[size]}/{r}"] = sums.get(r)
    return pkg_dir


def load_lists(size, pkg_dir, ck):
    """Analysed rows of every list of this size. Returns design.json and
    {set: {"items": DataFrame in item order, "orders": {k: DataFrame in
    presentation order}}}."""
    with open(os.path.join(pkg_dir, "package", "design.json")) as f:
        design = json.load(f)
    counts = design["counts_per_block"]
    ck.add(f"package {size}", "design.json is for this block size",
           design["block_size"] == size and counts["analysed"] == size,
           f"block_size {design['block_size']}, analysed {counts['analysed']}")
    sd_sat, sd_fog = 39.182, 9.348
    ck.add(f"package {size}", "the predictors' definition is the expected one",
           f"{sd_sat}" in design["predictors"]
           and f"{sd_fog}" in design["predictors"], design["predictors"])
    out = {}
    for b in SETS:
        orders = {}
        for k in ORDERS:
            with open(os.path.join(pkg_dir, "package", "lists",
                                   f"{b}_o{k}.json")) as f:
                rows = pd.DataFrame(json.load(f))
            a = rows[rows["analysed"] == True].reset_index(drop=True)  # noqa
            tc = a["trial_type"].value_counts().to_dict()
            ok = (len(rows) == design["main_trials_per_block"]
                  and len(a) == counts["analysed"]
                  and set(a["trial_type"]) <= set(ANALYSED_TYPES)
                  and all(tc.get(t, 0) == counts[TYPE_COUNT_KEY[t]]
                          for t in ANALYSED_TYPES))
            ck.add(f"lists {size}", f"{b}_o{k}: {len(rows)} rows, analysed "
                   "counts by type equal design.json", ok,
                   f"{len(a)} analysed {tc}")
            rebuilt = np.max(np.abs(np.r_[
                a["pred_sat"] - (-a["d_sat"] / sd_sat),
                a["pred_fog"] - (a["d_fog"] / sd_fog)]))
            side_ok = np.all((a["left_path"] == "stimuli/" + a["file_a"])
                             == a["a_is_left"].astype(bool))
            ck.add(f"lists {size}", f"{b}_o{k}: predictors rebuild from "
                   "d_sat, d_fog and a_is_left agrees with the left image",
                   rebuilt <= 1e-9 and side_ok,
                   f"max |difference| {rebuilt:.1e}")
            orders[k] = a
        items = orders[ORDERS[0]].sort_values("item_id").reset_index(drop=True)
        same = True
        for k in ORDERS[1:]:
            o = orders[k].sort_values("item_id").reset_index(drop=True)
            same &= (list(o["item_id"]) == list(items["item_id"])
                     and np.array_equal(o["pred_sat"].to_numpy(),
                                        items["pred_sat"].to_numpy())
                     and np.array_equal(o["pred_fog"].to_numpy(),
                                        items["pred_fog"].to_numpy())
                     and np.array_equal(o["a_is_left"].to_numpy(),
                                        items["a_is_left"].to_numpy()))
        ck.add(f"lists {size}", f"set {b}: the {len(ORDERS)} orders hold the "
               "same items with identical predictors and sides", same,
               f"{len(items)} items, {items['scene_id'].nunique()} scenes")
        out[b] = {"items": items, "orders": orders}
    return design, out


def design_matrix(df, model):
    X = df[["pred_sat", "pred_fog"]].to_numpy(float)
    if model == "two-weight":
        return X
    side = np.where(df["a_is_left"].astype(bool), 1.0, -1.0)
    return np.column_stack([X, side])


# ------------------------------------------------------------
# Estimation and simulation
# ------------------------------------------------------------

def fit_firth_batch_k(X, Y, n_iter, tol, max_step):
    """Firth logistic regression without intercept, many people at once,
    ANY number of predictors k.

    The same algorithm as v4's fit_firth (score with the hat-diagonal
    correction, step capped at max_step, stop when every person's step is
    below tol) and fit_firth_batch (which is written out for k = 2 only).
    Returns the estimates (people x k) and each person's last step."""
    T, k = X.shape
    P = Y.shape[0]
    S = (X[:, :, None] * X[:, None, :]).reshape(T, k * k)   # x_t x_t'
    b = np.zeros((P, k))
    last = np.full(P, np.inf)
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(b @ X.T, -500.0, 500.0)))
        w = np.clip(p * (1 - p), 1e-12, None)
        A = (w @ S).reshape(P, k, k)                    # X' W X per person
        Ai = np.linalg.inv(A)
        q = Ai.reshape(P, k * k) @ S.T                  # x_t' A^-1 x_t
        h = w * q
        r = (Y - p) + h * (0.5 - p)
        g = r @ X
        st = np.einsum("pij,pj->pi", Ai, g)
        n = np.max(np.abs(st), axis=1)
        sc = np.where(n > max_step, max_step / np.maximum(n, 1e-30), 1.0)
        b += st * sc[:, None]
        last = n * sc
        if n.max() < tol:
            break
    return b, last


def make_fitter(v4):
    def fit(X, Y):
        if X.shape[1] == 2:
            return v4.fit_firth_batch(X, Y)        # the registered estimator
        return fit_firth_batch_k(X, Y, v4.FIRTH_ITER, v4.FIRTH_TOL,
                                 v4.FIRTH_MAX_STEP)
    return fit


def respond(X2, side, W, bias, lapse, U):
    """Choices of many people: weights W (people x 2), side bias per person
    (logit, + = left), lapse rate, and uniforms U (people x trials).
    With bias 0 this is v4's simulate_lapsing on the same uniforms."""
    eta = W @ X2.T + bias[:, None] * side[None, :]
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -500.0, 500.0)))
    if lapse:
        p = (1.0 - lapse) * p + lapse / 2.0
    return (U < p).astype(float)


def thresholds_from(est):
    return {"upper": np.percentile(est, 100 * (1 - THRESHOLD_ALPHA), axis=0),
            "lower": np.percentile(est, 100 * THRESHOLD_ALPHA, axis=0)}


def quantile_se(est, q, z=1.96):
    """SE of the q-th quantile of each column, distribution-free: half the
    distance between the order statistics that bound a 95% interval for it,
    divided by 1.96."""
    n = est.shape[0]
    half = z * np.sqrt(n * q * (1 - q))
    lo = int(np.clip(np.floor(n * q - half), 0, n - 1))
    hi = int(np.clip(np.ceil(n * q + half), 0, n - 1))
    srt = np.sort(est, axis=0)
    return (srt[hi] - srt[lo]) / (2 * z)


def threshold_se(est):
    """SE of (upper sat, upper con, lower sat, lower con)."""
    return np.r_[quantile_se(est[:, :2], 1 - THRESHOLD_ALPHA),
                 quantile_se(est[:, :2], THRESHOLD_ALPHA)]


def null_summary(est, last):
    th = thresholds_from(est[:, :2])
    return {**th, "null_sd": est[:, :2].std(axis=0),
            "null_corr": float(np.corrcoef(est[:, 0], est[:, 1])[0, 1]),
            "unconverged": float(np.mean(last > UNCONVERGED_STEP))}


# ------------------------------------------------------------
# 1. Estimator checks
# ------------------------------------------------------------

def check_estimators(v4, fit, X3, ck, rng):
    X2 = X3[:, :2]
    P = 200
    W = rng.normal(0.3, 0.6, (P, 2))
    W[:5] = [[2.5, 3.0], [-2.0, 2.0], [0.0, 0.0], [3.0, -3.0], [0.5, 0.7]]
    Y = v4.simulate_lapsing(X2, W, 0.1, rng)
    b_ref, _ = v4.fit_firth_batch(X2, Y)
    b_k, _ = fit_firth_batch_k(X2, Y, v4.FIRTH_ITER, v4.FIRTH_TOL,
                               v4.FIRTH_MAX_STEP)
    d = float(np.max(np.abs(b_ref - b_k)))
    ck.add("estimator", "fit_firth_batch_k with two predictors equals v4's "
           "fit_firth_batch", d <= ESTIMATOR_TOL,
           f"max |difference| {d:.1e} over {P} people")
    P3 = 60
    W = rng.normal(0.3, 0.6, (P3, 2))
    bias = rng.normal(0.0, 0.6, P3)
    bias[:3] = [2.0, -2.0, 0.0]
    Y = respond(X2, X3[:, 2], W, bias, 0.1, rng.random((P3, len(X3))))
    b_k, _ = fit(X3, Y)
    b_ref = np.array([v4.fit_firth(X3, Y[i]) for i in range(P3)])
    d = float(np.max(np.abs(b_ref - b_k)))
    ck.add("estimator", "fit_firth_batch_k with three predictors (side "
           "model) equals v4's single-person fit_firth", d <= ESTIMATOR_TOL,
           f"max |difference| {d:.1e} over {P3} people")
    W = rng.normal(0.3, 0.6, (50, 2))
    seed = int(rng.integers(1 << 31))
    Y1 = v4.simulate_lapsing(X2, W, 0.1, np.random.default_rng(seed))
    Y2 = respond(X2, X3[:, 2], W, np.zeros(50), 0.1,
                 np.random.default_rng(seed).random((50, len(X2))))
    ck.add("estimator", "respond() with no side bias equals v4's "
           "simulate_lapsing on the same random numbers",
           np.array_equal(Y1, Y2), "50 people, identical responses")


# ------------------------------------------------------------
# 2. Thresholds: reproduction, independent seed, side model
# ------------------------------------------------------------

def run_thresholds(v4, cap, fit, size, design, L, ck, st):
    rows, keep = [], {}
    reg = design["thresholds_by_set"]
    for b in SETS:
        items = L[b]["items"]
        X2 = design_matrix(items, "two-weight")
        X3 = design_matrix(items, "side")
        # (a) exact reproduction: build_trials_v5's inputs and stream
        rng = cap.Streams(REPRO_SEED, REPRO_STREAM_KEY, size)(
            REPRO_STREAM_NAME.format(set=b))
        Yr = v4.simulate_many(X2, np.zeros((N_THRESHOLD_PEOPLE, 2)), rng)
        est, last = v4.fit_firth_batch(X2, Yr)
        rep = null_summary(est, last)
        want = np.array([reg[b]["upper"]["sat"], reg[b]["upper"]["con"],
                         reg[b]["lower"]["sat"], reg[b]["lower"]["con"]])
        got = np.r_[rep["upper"], rep["lower"]]
        d = float(np.max(np.abs(got - want)))
        ck.add(f"thresholds {size}", f"set {b}: reproduction equals "
               "design.json (same inputs, same stream)", d <= REPRO_EXACT_TOL,
               f"max |difference| {d:.1e}")
        ck.add(f"thresholds {size}", f"set {b}: every random chooser's fit "
               "converged (reproduction)", rep["unconverged"] == 0,
               f"unconverged share {rep['unconverged']:.5f}")
        # (b) independent seed
        Yi = v4.simulate_many(X2, np.zeros((N_THRESHOLD_PEOPLE, 2)),
                              st[b]("independent"))
        est_i, last_i = v4.fit_firth_batch(X2, Yi)
        ind = null_summary(est_i, last_i)
        se_reg = threshold_se(est)
        se_d = np.sqrt(se_reg ** 2 + threshold_se(est_i) ** 2)
        diff = np.r_[ind["upper"], ind["lower"]] - want
        zmax = float(np.max(np.abs(diff) / se_d))
        ck.add(f"thresholds {size}", f"set {b}: independent seed agrees with "
               f"design.json within {QUANTILE_Z:g} SE", zmax <= QUANTILE_Z,
               f"max |difference| {np.max(np.abs(diff)):.4f}, largest "
               f"|z| {zmax:.2f} (SE of a difference {se_d.min():.4f}-"
               f"{se_d.max():.4f})")
        # (c) the side model, fitted to the SAME random choosers as the
        #     registered thresholds (a random chooser's answers do not
        #     depend on the side, so the same answers serve both models).
        #     Thresholds of the two models then differ only because the
        #     models differ, not by Monte Carlo luck: with separate
        #     choosers, the error in a detection difference between the
        #     models (about 0.009) is as large as RULE_DETECTION_LOSS_MAX.
        Ys = Yr
        est_s, last_s = fit(X3, Ys)
        sid = null_summary(est_s, last_s)
        ck.add(f"thresholds {size}", f"set {b}: every random chooser's fit "
               "converged (side model)", sid["unconverged"] == 0,
               f"unconverged share {sid['unconverged']:.5f}")
        se_of = {"design.json (registered)": se_reg,
                 "reproduced (same stream)": se_reg,
                 "independent seed": threshold_se(est_i),
                 "same choosers as design.json": threshold_se(est_s)}
        for src, model, s in (("design.json (registered)", "two-weight",
                               {"upper": want[:2], "lower": want[2:],
                                "null_sd": rep["null_sd"],
                                "null_corr": rep["null_corr"],
                                "unconverged": rep["unconverged"]}),
                              ("reproduced (same stream)", "two-weight", rep),
                              ("independent seed", "two-weight", ind),
                              ("same choosers as design.json", "side",
                               sid)):
            for j, f in enumerate(FEATURES):
                rows.append({"size": size, "set": b, "model": model,
                             "source": src, "feature": FEATURE_LABEL[f],
                             "upper": s["upper"][j], "lower": s["lower"][j],
                             "null_sd": s["null_sd"][j],
                             "null_corr": s["null_corr"],
                             "unconverged": s["unconverged"],
                             "se_upper": se_of[src][j],
                             "se_lower": se_of[src][j + 2]})
        keep[b] = {"two-weight": {"upper": want[:2], "lower": want[2:],
                                  "null_sd": rep["null_sd"],
                                  "null_corr": rep["null_corr"]},
                   "side": sid, "Yr": Yr, "est_r": est,
                   "Ys": Ys, "est_s": est_s, "se_reg": se_reg,
                   "side_corr": [float(np.corrcoef(X3[:, 2], X3[:, j])[0, 1])
                                 for j in range(2)]}
    for model in MODELS:
        up = np.array([keep[b][model]["upper"] for b in SETS])
        rel = float(np.max(np.abs(up[0] - up[1]) / np.mean(up, axis=0)))
        ck.add(f"thresholds {size}", f"{model} model: the two sets' upper "
               f"thresholds agree within {SET_THRESHOLD_TOL:.0%}",
               rel <= SET_THRESHOLD_TOL, f"largest relative difference "
               f"{rel:.1%}")
    return rows, keep


# ------------------------------------------------------------
# 3. Order invariance
# ------------------------------------------------------------

def run_order_checks(v4, fit, size, L, keep, ck, st):
    """(i) The same random choosers, each answering every ITEM the same
    way, fitted on each order's presentation sequence: per-person weights
    and thresholds must be identical (the fit does not depend on row
    order). (ii) Independent random choosers per order: thresholds within
    QUANTILE_Z SE of the registered ones (responses of random choosers
    do not depend on trial position)."""
    rows = []
    for b in SETS:
        items = L[b]["items"]
        pos = {it: i for i, it in enumerate(items["item_id"])}
        for k in ORDERS:
            od = L[b]["orders"][k]
            perm = np.array([pos[it] for it in od["item_id"]])
            for model, Y, est0 in (("two-weight", keep[b]["Yr"],
                                    keep[b]["est_r"]),
                                   ("side", keep[b]["Ys"],
                                    keep[b]["est_s"])):
                Xk = design_matrix(od, model)
                est_k, _ = fit(Xk, Y[:, perm])
                d_person = float(np.max(np.abs(est_k - est0)))
                th0 = thresholds_from(est0[:, :2])
                thk = thresholds_from(est_k[:, :2])
                d_th = float(np.max(np.abs(np.r_[thk["upper"] - th0["upper"],
                                                 thk["lower"] - th0["lower"]])))
                rows.append({"size": size, "set": b, "order": k,
                             "model": model, "check": "same answers per item",
                             "max_person_diff": d_person,
                             "max_threshold_diff": d_th})
            # independent choosers per order, two-weight model
            Xk = design_matrix(od, "two-weight")
            Yk = v4.simulate_many(Xk, np.zeros((N_THRESHOLD_PEOPLE, 2)),
                                  st[b](f"order{k}"))
            est_k, _ = v4.fit_firth_batch(Xk, Yk)
            thk = thresholds_from(est_k)
            reg = keep[b]["two-weight"]
            diff = np.r_[thk["upper"] - reg["upper"],
                         thk["lower"] - reg["lower"]]
            se_d = np.sqrt(keep[b]["se_reg"] ** 2 + threshold_se(est_k) ** 2)
            row = {"size": size, "set": b, "order": k, "model": "two-weight",
                   "check": "independent choosers", "max_person_diff": np.nan,
                   "max_threshold_diff": float(np.max(np.abs(diff))),
                   "max_abs_z": float(np.max(np.abs(diff) / se_d))}
            for j, f in enumerate(FEATURES):
                row[f"upper_{f}"] = thk["upper"][j]
                row[f"lower_{f}"] = thk["lower"][j]
            rows.append(row)
    df = pd.DataFrame(rows)
    same = df[df["check"] == "same answers per item"]
    for model in MODELS:
        s = same[same["model"] == model]
        ck.add(f"order {size}", f"{model} model: the same answers per item "
               f"give the same weight for every person in orders "
               f"{ORDERS[0]}-{ORDERS[-1]}",
               s["max_person_diff"].max() <= PERMUTATION_TOL,
               f"max |difference| {s['max_person_diff'].max():.1e} over "
               f"{N_THRESHOLD_PEOPLE:,} people x {len(s)} lists")
    ind = df[df["check"] == "independent choosers"]
    ck.add(f"order {size}", "independent random choosers on each order give "
           f"thresholds within {QUANTILE_Z:g} SE of the registered ones",
           ind["max_abs_z"].max() <= QUANTILE_Z,
           f"max |difference| {ind['max_threshold_diff'].max():.4f}, "
           f"largest |z| {ind['max_abs_z'].max():.2f}")
    return df


# ------------------------------------------------------------
# 4. Precision, detection and the side sweep
# ------------------------------------------------------------

def side_values():
    vals = [("none", 0.0)]
    for v in SIDE_BIASES:
        vals += [(f"left {v:g}", v), (f"right {v:g}", -v)]
    vals.append((f"population SD {SIDE_POPULATION_SD:g}", None))
    return vals


def run_side_sweep(fit, size, L, keep, ck, st):
    """Random choosers and planning-weight people, each with every side
    value; the same random numbers (U) for every side value and both
    models, so differences between rows are not Monte Carlo noise."""
    rows = []
    for b in SETS:
        items = L[b]["items"]
        X2 = design_matrix(items, "two-weight")
        X3 = design_matrix(items, "side")
        side = X3[:, 2]
        pop = st[b]("side_population").normal(0.0, SIDE_POPULATION_SD,
                                               N_PEOPLE)
        for person, W0 in (("random chooser", (0.0, 0.0)),
                           ("planning weights", PLANNING_WEIGHTS)):
            U = st[b](f"sweep_{person}").random((N_PEOPLE, len(items)))
            W = np.tile(np.asarray(W0, float), (N_PEOPLE, 1))
            base = {}
            for label, v in side_values():
                bias = pop if v is None else np.full(N_PEOPLE, v)
                Y = respond(X2, side, W, bias, SIDE_LAPSE, U)
                for model, X in (("two-weight", X2), ("side", X3)):
                    est, last = fit(X, Y)
                    th = keep[b][model]
                    for j, f in enumerate(FEATURES):
                        e = est[:, j]
                        r = {"size": size, "set": b, "model": model,
                             "person": person, "side_bias": label,
                             "side_value": np.nan if v is None else v,
                             "feature": FEATURE_LABEL[f],
                             "true_weight": W0[j],
                             "upper_tail": float(np.mean(e > th["upper"][j])),
                             "lower_tail": float(np.mean(e < th["lower"][j])),
                             "mean": float(e.mean()), "sd": float(e.std()),
                             "unconverged": float(np.mean(
                                 last > UNCONVERGED_STEP))}
                        if label == "none":
                            base[(model, j)] = r["mean"]
                        r["shift"] = r["mean"] - base[(model, j)]
                        rows.append(r)
                    if model == "side" and person == "planning weights":
                        rows.append({"size": size, "set": b, "model": model,
                                     "person": person, "side_bias": label,
                                     "side_value": np.nan if v is None else v,
                                     "feature": "side term (estimator check "
                                     "only)", "true_weight": np.nan,
                                     "mean": float(est[:, 2].mean()),
                                     "sd": float(est[:, 2].std())})
    df = pd.DataFrame(rows)
    ck.add(f"side sweep {size}", "every fit converged",
           df["unconverged"].fillna(0).max() == 0,
           f"largest unconverged share {df['unconverged'].max():.5f}")
    s = df[(df["feature"].str.startswith("side term"))
           & df["side_value"].notna()].groupby("side_value")["mean"].mean()
    mono = bool(np.all(np.diff(s.to_numpy()) > 0)) and abs(s.loc[0.0]) < 0.02
    ck.add(f"side sweep {size}", "the side model's side estimate rises with "
           "the simulated side bias and is near 0 without one (estimator "
           "check; never a result)", mono,
           f"mean side estimate {s.min():+.2f} at {s.index.min():+g} to "
           f"{s.max():+.2f} at {s.index.max():+g}")
    rc = df[(df["person"] == "random chooser") & (df["side_bias"] == "none")
            & ~df["feature"].str.startswith("side")]
    se = np.sqrt(THRESHOLD_ALPHA * (1 - THRESHOLD_ALPHA) / N_PEOPLE)
    worst = float(np.max(np.abs(np.r_[rc["upper_tail"], rc["lower_tail"]]
                                - THRESHOLD_ALPHA)))
    ck.add(f"side sweep {size}", "without side bias, random choosers fall "
           f"beyond each threshold {THRESHOLD_ALPHA:.1%} of the time (within "
           f"{MC_Z:g} SE + threshold error)", worst <= MC_Z * se + 0.004,
           f"largest |rate - {THRESHOLD_ALPHA}| {worst:.4f} (SE {se:.4f})")
    return df


def precision_table(sweep, keep, size):
    rows = []
    for b in SETS:
        for model in MODELS:
            pl = sweep[(sweep["size"] == size) & (sweep["set"] == b)
                       & (sweep["model"] == model)
                       & (sweep["person"] == "planning weights")
                       & (sweep["side_bias"] == "none")]
            for j, f in enumerate(FEATURES):
                r = pl[pl["feature"] == FEATURE_LABEL[f]].iloc[0]
                th = keep[b][model]
                rows.append({"size": size, "set": b, "model": model,
                             "feature": FEATURE_LABEL[f],
                             "upper": th["upper"][j], "lower": th["lower"][j],
                             "null_sd": th["null_sd"][j],
                             "null_corr": th["null_corr"],
                             "planning_weight": PLANNING_WEIGHTS[j],
                             "planning_mean": r["mean"],
                             "planning_sd": r["sd"],
                             "detection": r["upper_tail"],
                             "opposite": r["lower_tail"],
                             "side_corr_with_predictor":
                                 keep[b]["side_corr"][j]})
    return rows


# ------------------------------------------------------------
# 5. The rule
# ------------------------------------------------------------

def apply_rule(sweep, prec):
    two = sweep[(sweep["model"] == "two-weight")
                & ~sweep["feature"].str.startswith("side")]
    at = two[two["side_value"].abs().sub(RULE_SIDE_BIAS).abs() < 1e-12]
    rc = at[at["person"] == "random chooser"]
    tail = float(np.max(np.r_[rc["upper_tail"], rc["lower_tail"]]))
    shift = float(at["shift"].abs().max())
    i_t = np.argmax(np.maximum(rc["upper_tail"], rc["lower_tail"]).to_numpy())
    i_s = np.argmax(at["shift"].abs().to_numpy())
    wt, ws = rc.iloc[i_t], at.iloc[i_s]
    a = (tail > RULE_TAIL_MAX) or (shift > RULE_WEIGHT_BIAS_MAX)
    P = prec.pivot_table(index=["size", "set", "feature"], columns="model",
                         values="detection")
    loss = P["two-weight"] - P["side"]
    worst_loss = float(loss.max())
    wl = loss.idxmax()
    bpart = worst_loss <= RULE_DETECTION_LOSS_MAX
    chosen = "side" if (a and bpart) else "two-weight"
    return {"a": a, "b": bpart, "chosen": chosen,
            "tail": tail, "tail_where": f"{int(wt['size'])} {wt['set']}, "
            f"{wt['feature']}, {wt['side_bias']}",
            "shift": shift, "shift_where": f"{int(ws['size'])} {ws['set']}, "
            f"{ws['feature']}, {ws['person']}, {ws['side_bias']}",
            "loss": worst_loss, "loss_where": f"{wl[0]} {wl[1]}, {wl[2]}",
            "loss_table": loss}


# ------------------------------------------------------------
# 6. Detection curve (chosen model)
# ------------------------------------------------------------

def run_curve(fit, model, size, L, keep, st):
    rows = []
    for b in SETS:
        items = L[b]["items"]
        X2 = design_matrix(items, "two-weight")
        X = design_matrix(items, model)
        side = design_matrix(items, "side")[:, 2]
        th = keep[b][model]
        for lam in CURVE_LAPSES:
            for w in CURVE_WEIGHTS:
                r = {"size": size, "set": b, "model": model, "lapse": lam,
                     "true_weight": w}
                for j, f in enumerate(FEATURES):
                    W = np.zeros((N_PEOPLE, 2))
                    W[:, j] = w
                    U = st[b](f"curve_{lam:g}_{w:g}_{f}").random(
                        (N_PEOPLE, len(items)))
                    Y = respond(X2, side, W, np.zeros(N_PEOPLE), lam, U)
                    est, _ = fit(X, Y)
                    r[f"{f}_detected"] = float(np.mean(est[:, j]
                                                       > th["upper"][j]))
                    r[f"{f}_opposite"] = float(np.mean(est[:, j]
                                                       < th["lower"][j]))
                    r[f"{f}_mean"] = float(est[:, j].mean())
                rows.append(r)
    return rows


# ------------------------------------------------------------
# Figures
# ------------------------------------------------------------

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
             "#56B4E9"]


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)


def fig_curve(curve, model, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(BLOCK_SIZES), 2, figsize=(9, 3.6 *
                             len(BLOCK_SIZES)), sharex=True, sharey=True,
                             squeeze=False)
    for i, size in enumerate(BLOCK_SIZES):
        for j, f in enumerate(("con", "sat")):
            ax = axes[i, j]
            for c, lam in enumerate(CURVE_LAPSES):
                for b, ls in zip(SETS, ("-", "--")):
                    d = curve[(curve["size"] == size) & (curve["set"] == b)
                              & (curve["lapse"] == lam)]
                    ax.plot(d["true_weight"], d[f"{f}_detected"], ls,
                            marker="o", ms=3.5, color=OKABE_ITO[c],
                            label=f"lapse {lam:g}, set {b}")
            ax.axhline(THRESHOLD_ALPHA, color="#888888", lw=0.8, ls=":")
            ax.axvline(PLANNING_WEIGHTS[FEATURES.index(f)], color="#888888",
                       lw=0.8, ls="-.")
            ax.set_title(f"{FEATURE_LABEL[f].capitalize()}, {size} trials "
                         "per block", fontsize=10)
            ax.set_ylim(0, 1.02)
            _style(ax)
            if i == len(BLOCK_SIZES) - 1:
                ax.set_xlabel("True weight per corpus SD")
            if j == 0:
                ax.set_ylabel("Share detected\n(predicted direction)")
    axes[0, 1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Probability that a person's preference is detected "
                 f"({'two-weight' if model == 'two-weight' else 'side'} "
                 "model)\ndash-dot line: planning weight (an assumption); "
                 "dotted line: 2.5%", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_side(sweep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = sweep[sweep["side_value"].notna()
              & ~sweep["feature"].str.startswith("side")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7), sharex=True)
    lists = [(s, b) for s in BLOCK_SIZES for b in SETS]
    styles = ["-", "--", "-.", ":"]
    for j, f in enumerate(("con", "sat")):
        for i, (person, col, ylab) in enumerate((
                ("random chooser", None, "Random choosers: share beyond\n"
                 "the higher of the two thresholds' tails"),
                ("planning weights", "upper_tail",
                 "Planning-weight people:\nshare detected"))):
            ax = axes[i, j]
            for m, model in enumerate(MODELS):
                for k, (s, b) in enumerate(lists):
                    q = d[(d["model"] == model) & (d["person"] == person)
                          & (d["feature"] == FEATURE_LABEL[f])
                          & (d["size"] == s) & (d["set"] == b)
                          ].sort_values("side_value")
                    y = (np.maximum(q["upper_tail"], q["lower_tail"])
                         if col is None else q[col])
                    ax.plot(q["side_value"], y, styles[k],
                            color=OKABE_ITO[0 if m == 0 else 3], lw=1.2,
                            label=f"{'two-weight' if m == 0 else 'side'} "
                            f"model, {s} {b}")
            if i == 0:
                ax.axhline(THRESHOLD_ALPHA, color="#888888", lw=0.8, ls=":")
                ax.axhline(RULE_TAIL_MAX, color="#888888", lw=0.8, ls="--")
            ax.set_title(FEATURE_LABEL[f].capitalize(), fontsize=10)
            _style(ax)
            if i == 1:
                ax.set_xlabel("Side bias (logit; + = prefers the left "
                              "image)")
            if j == 0:
                ax.set_ylabel(ylab)
    axes[0, 1].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("What a side preference does to each model (side biases "
                 "and planning weights are assumptions)\ndotted: 2.5%; "
                 f"dashed: the rule's limit {RULE_TAIL_MAX}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

def fmt(df, cols=None, nd=3):
    d = df if cols is None else df[cols]
    return d.to_string(index=False, float_format=lambda x: f"{x:.{nd}f}")


def write_report(out, ck, thr, prec, sweep, order, curve, rule, meta):
    L = []
    w = L.append
    w("THRESHOLDS AND THE PER-PERSON MODEL  (thresholds_v5.py, script 6)")
    w("=" * 74)
    w(f"run       {meta['started']}  ({meta['minutes']:.1f} min)")
    w(f"output    {out}")
    w(f"python    {meta['python']}, numpy {meta['numpy']}, "
      f"pandas {meta['pandas']}")
    w("")
    w("PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES. The planning")
    w(f"weights ({PLANNING_WEIGHTS[0]} saturation, {PLANNING_WEIGHTS[1]} "
      f"contrast per corpus SD), the lapse rate ({PLANNING_LAPSE}), the side")
    w(f"biases ({', '.join(f'{v:g}' for v in SIDE_BIASES)} logit, both "
      f"directions) and their population SD ({SIDE_POPULATION_SD}) were")
    w("chosen for planning; no image-choice data exist. Every number below")
    w("that involves simulated people with a preference depends on them.")
    w("")
    w("1. WHAT WAS DONE")
    w("-" * 74)
    w(f"For each block size ({', '.join(map(str, BLOCK_SIZES))}) and set "
      f"({', '.join(SETS)}), on the analysed trials of the frozen lists:")
    w(f"  * thresholds from {N_THRESHOLD_PEOPLE:,} random choosers (2.5% per "
      "tail): reproduced with build_trials_v5's")
    w("    own inputs and stream, rerun with an independent seed, and "
      "computed for")
    w("    the side model (two weights + a side term, +1 when A is on the "
      "left), fitted")
    w("    to the same random choosers as the registered thresholds;")
    w(f"  * {N_PEOPLE:,} people per cell at the planning weights (lapse "
      f"{PLANNING_LAPSE}) and {N_PEOPLE:,} random")
    w("    choosers, each with every side bias, fitted with both models on "
      "the same")
    w("    random numbers;")
    w(f"  * the order checks on orders {ORDERS[0]}-{ORDERS[-1]};")
    w("  * the rule below, then the detection curve for the chosen model.")
    w("Fits: Firth logistic regression without intercept (v4's "
      "fit_firth_batch for")
    w("two predictors; fit_firth_batch_k, checked against v4's fit_firth, "
      "for three).")
    w("")
    w("2. CHECKS")
    w("-" * 74)
    for r in ck.rows:
        w(f"  {r['status']}  [{r['section']}] {r['check']}")
        if r["detail"]:
            w(f"          {r['detail']}")
    w(f"\n  {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL")
    w("")
    w("3. THRESHOLDS")
    w("-" * 74)
    w("Registered = design.json. The reproduction uses the same inputs and "
      "random")
    w("stream as build_trials_v5.py; the independent seed is this script's "
      "own. The side")
    w("model is fitted to the same random choosers as the registered "
      "thresholds, so the")
    w("two models' thresholds differ only because the models differ (a "
      "random chooser's")
    w("answers do not depend on the side).")
    for size in BLOCK_SIZES:
        t = thr[thr["size"] == size].copy()
        w(f"\n  {size} trials per block")
        w("  " + fmt(t, ["set", "model", "source", "feature", "upper",
                         "lower", "se_upper", "null_sd"], 4)
          .replace("\n", "\n  "))
    w("\n  se_upper: Monte Carlo SE of the upper threshold, from the "
      "choosers' own weights.")
    w("\n  Rounded to 3 decimals, as v4 registered them (two-weight model):")
    reg = thr[thr["source"] == "design.json (registered)"]
    for size in BLOCK_SIZES:
        for b in SETS:
            q = reg[(reg["size"] == size) & (reg["set"] == b)]
            s_ = q[q["feature"] == "saturation"].iloc[0]
            c_ = q[q["feature"] == "contrast"].iloc[0]
            w(f"    {size} set {b}: saturation {s_['upper']:+.3f} / "
              f"{s_['lower']:+.3f}   contrast {c_['upper']:+.3f} / "
              f"{c_['lower']:+.3f}")
    w("")
    w("4. THE TWO MODELS AT THE PLANNING WEIGHTS (no side bias, lapse "
      f"{PLANNING_LAPSE})")
    w("-" * 74)
    w("null_sd: SD of random choosers' weights (precision without an "
      "effect);")
    w("planning_sd: SD of the weights of people at the planning weights;")
    w("detection: share above the upper threshold; opposite: below the "
      "lower;")
    w("side_corr: correlation of the side term with that predictor in the "
      "list.")
    w("[planning weights and lapse: ASSUMPTIONS]")
    pr = prec.copy()
    for size in BLOCK_SIZES:
        w(f"\n  {size} trials per block")
        w("  " + fmt(pr[pr["size"] == size],
                     ["set", "feature", "model", "upper", "null_sd",
                      "planning_mean", "planning_sd", "detection", "opposite",
                      "side_corr_with_predictor"], 4).replace("\n", "\n  "))
    P = prec.pivot_table(index=["size", "set", "feature"], columns="model",
                         values=["detection", "planning_sd", "null_sd"])
    w("\n  Side model minus two-weight model:")
    dd = pd.DataFrame({
        "detection": P[("detection", "side")] - P[("detection",
                                                   "two-weight")],
        "planning_sd": P[("planning_sd", "side")] - P[("planning_sd",
                                                       "two-weight")],
        "null_sd": P[("null_sd", "side")] - P[("null_sd", "two-weight")],
    }).reset_index()
    w("  " + fmt(dd, nd=4).replace("\n", "\n  "))
    w("")
    w("5. WHAT SIDE BIAS DOES TO EACH MODEL")
    w("-" * 74)
    w("Each person has a fixed side bias (logit; + = prefers the left "
      "image),")
    w(f"lapse {SIDE_LAPSE}. Random choosers have no feature weights; "
      "planning-weight")
    w(f"people have {PLANNING_WEIGHTS}. upper/lower: share beyond each "
      "threshold (for")
    w("random choosers both are false detections; nominal 0.025). shift: "
      "the mean")
    w("weight minus its value without side bias, same model, same random "
      "numbers.")
    w("[side biases and planning weights: ASSUMPTIONS]")
    sw = sweep[~sweep["feature"].str.startswith("side")]
    for person in ("random chooser", "planning weights"):
        for size in BLOCK_SIZES:
            for b in SETS:
                q = sw[(sw["person"] == person) & (sw["size"] == size)
                       & (sw["set"] == b)]
                cols = (("upper_tail", "lower_tail", "shift")
                        if person == "random chooser"
                        else ("upper_tail", "shift"))
                tab = []
                for lab, _ in side_values():
                    r = {"side_bias": lab}
                    for model in MODELS:
                        for f in FEATURES:
                            x = q[(q["side_bias"] == lab)
                                  & (q["model"] == model)
                                  & (q["feature"] == FEATURE_LABEL[f])]
                            for c in cols:
                                r[f"{model[:3]}:{f}:"
                                  f"{c.replace('_tail', '')}"] = (
                                    float(x[c].iloc[0]) if len(x) else np.nan)
                    tab.append(r)
                piv = pd.DataFrame(tab).set_index("side_bias")
                who = ("random choosers" if person == "random chooser"
                       else "people at the planning weights")
                note = ("; upper = detection" if person != "random chooser"
                        else "")
                w(f"\n  {who}, {size} trials, set {b}  (two = two-weight "
                  f"model, sid = side model{note})")
                w("  " + piv.to_string(float_format=lambda x: f"{x:.3f}")
                  .replace("\n", "\n  "))
    w("")
    w("6. TRIAL ORDER")
    w("-" * 74)
    w("(i) The same random choosers answering every item the same way, "
      "fitted on")
    w("each order's presentation sequence: the largest difference in any "
      "person's")
    w("weight and in any threshold. (ii) Independent random choosers on "
      "each order:")
    w("the largest difference from the registered thresholds.")
    w("Why the thresholds cannot depend on the order: a random chooser's "
      "answers")
    w("do not depend on a trial's position, and the fit does not depend on "
      "the")
    w("order of the rows. Order could matter for real people (fatigue, "
      "practice")
    w("within a block); that is a question for script 7, not for the "
      "thresholds.")
    og = order.groupby(["size", "check", "model"]).agg(
        max_person_diff=("max_person_diff", "max"),
        max_threshold_diff=("max_threshold_diff", "max")).reset_index()
    w("  " + og.to_string(index=False, float_format=lambda x: f"{x:.2e}")
      .replace("\n", "\n  "))
    w("")
    w("7. RECOMMENDATION")
    w("-" * 74)
    w("The rule, fixed before the run: register the side model only if")
    w(f"  (a) at a side bias of {RULE_SIDE_BIAS} logit (either direction) "
      "the two-weight model's")
    w(f"      false-detection rate exceeds {RULE_TAIL_MAX} in either tail, "
      "or a weight moves")
    w(f"      by more than {RULE_WEIGHT_BIAS_MAX}; AND")
    w(f"  (b) the side model loses at most {RULE_DETECTION_LOSS_MAX} of "
      "detection at the planning")
    w("      weights in every set, size and feature.")
    w(f"  (a) worst tail rate {rule['tail']:.4f} ({rule['tail_where']}); "
      f"worst weight shift {rule['shift']:.4f}")
    w(f"      ({rule['shift_where']})  ->  (a) "
      f"{'HOLDS' if rule['a'] else 'does not hold'}")
    w(f"  (b) largest detection loss {rule['loss']:+.4f} "
      f"({rule['loss_where']})  ->  (b) "
      f"{'HOLDS' if rule['b'] else 'does not hold'}")
    pop = sweep[(sweep["model"] == "two-weight")
                & sweep["side_bias"].str.startswith("population")
                & ~sweep["feature"].str.startswith("side")]
    prc = pop[pop["person"] == "random chooser"]
    w(f"  For comparison, a population whose side biases vary between people "
      f"(SD {SIDE_POPULATION_SD} logit,")
    w(f"  an assumption): two-weight model, worst tail rate "
      f"{np.max(np.r_[prc['upper_tail'], prc['lower_tail']]):.4f}, worst "
      f"weight shift {pop['shift'].abs().max():.4f}.")
    w("")
    w("  Not part of the rule, for both models: worst false-detection rate "
      "of random")
    w("  choosers (either tail, any set and size) at each fixed side bias "
      "(either")
    w("  direction), and whether it is lopsided (one tail) or symmetric:")
    rcs = sweep[(sweep["person"] == "random chooser")
                & sweep["side_value"].notna()
                & ~sweep["feature"].str.startswith("side")]
    for model in MODELS:
        parts = []
        for v in [0.0] + list(SIDE_BIASES):
            q = rcs[(rcs["model"] == model)
                    & ((rcs["side_value"].abs() - v).abs() < 1e-12)]
            worst = float(np.max(np.r_[q["upper_tail"], q["lower_tail"]]))
            gap = float(np.max(np.abs(q["upper_tail"] - q["lower_tail"])))
            parts.append(f"{v:g}: {worst:.3f} (tails differ by up to "
                         f"{gap:.3f})")
        w(f"    {model}: " + "; ".join(parts[:3]))
        w("      " + "; ".join(parts[3:]))
    w("  A lopsided rise pushes weights in one direction (a bias); a "
      "symmetric rise")
    w("  widens the spread (more detections in both directions). Under "
      "strong side")
    w("  bias the side model's random choosers spread more, because "
      "one-sided answers")
    w("  carry less information about the features.")
    w("")
    if rule["chosen"] == "two-weight":
        w("  RECOMMENDED: the two-weight model (current), with the "
          "thresholds frozen in")
        w("  design.json. " + (
            "A side bias of the size tested does not move the two-weight "
            "model's\n  false-detection rates or weights beyond the rule's "
            "limits, because the side\n  of image A is nearly uncorrelated "
            "with the predictors in every list."
            if not rule["a"] else
            "Side bias does matter for it, but the side model costs more "
            "detection\n  than the rule allows."))
    else:
        w("  RECOMMENDED: the side model. A side preference of the size in "
          "the rule moves")
        w("  the two-weight model's weights or false-detection rates beyond "
          "the rule's")
        w("  limits, and the side model costs at most the detection shown in "
          "(b).")
        w("  Its thresholds are NOT the ones in design.json: registering it "
          "means")
        w("  replacing the frozen thresholds with these (same random "
          "choosers, side model):")
        sm = thr[thr["source"] == "same choosers as design.json"]
        for size in BLOCK_SIZES:
            for b in SETS:
                q = sm[(sm["size"] == size) & (sm["set"] == b)]
                s_ = q[q["feature"] == "saturation"].iloc[0]
                c_ = q[q["feature"] == "contrast"].iloc[0]
                w(f"    {size} set {b}: saturation {s_['upper']:+.3f} / "
                  f"{s_['lower']:+.3f}   contrast {c_['upper']:+.3f} / "
                  f"{c_['lower']:+.3f}")
        w("  The side term is a nuisance term only: its estimate is never "
          "reported as a")
        w("  result.")
    w("")
    w(f"8. DETECTION CURVE ({MODEL_LABEL[rule['chosen']]})")
    w("-" * 74)
    w("One feature at the true weight, the other 0, no side bias. "
      "detected: share")
    w("above the upper threshold; opposite: share below the lower. "
      f"{N_PEOPLE:,} people per point.")
    w("[true weights and lapse rates: ASSUMPTIONS for planning]")
    for size in BLOCK_SIZES:
        q = curve[curve["size"] == size]
        w(f"\n  {size} trials per block")
        w("  " + fmt(q, ["set", "lapse", "true_weight", "con_detected",
                         "con_opposite", "sat_detected", "sat_opposite"])
          .replace("\n", "\n  "))
    w("")
    w("9. FILES")
    w("-" * 74)
    for fn in meta["files"]:
        w(f"  {fn}")
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    ck = Checks()
    out = resolve_output_dir()
    log(f"output folder {out}")
    v4, h_v4 = load_module(root_path(V4_SIM_SCRIPT), "v4sim", ck)
    cap, h_cap = load_module(root_path(CAPACITY_SCRIPT), "capacity", ck)
    fit = make_fitter(v4)
    file_hashes = {V4_SIM_SCRIPT: h_v4, CAPACITY_SCRIPT: h_cap}

    designs, lists = {}, {}
    for size in BLOCK_SIZES:
        pkg = verify_package(size, ck, file_hashes)
        designs[size], lists[size] = load_lists(size, pkg, ck)

    s0 = BLOCK_SIZES[-1]
    check_estimators(v4, fit, design_matrix(lists[s0]["A"]["items"], "side"),
                     ck, cap.Streams(RANDOM_SEED, STREAM_KEY)("estimator"))

    thr_rows, prec_rows, sweeps, orders, keeps = [], [], [], [], {}
    for size in BLOCK_SIZES:
        st = {b: cap.Streams(RANDOM_SEED, STREAM_KEY, size, b) for b in SETS}
        log(f"thresholds, {size} trials")
        rows, keep = run_thresholds(v4, cap, fit, size, designs[size],
                                    lists[size], ck, st)
        thr_rows += rows
        keeps[size] = keep
        log(f"order checks, {size} trials")
        orders.append(run_order_checks(v4, fit, size, lists[size], keep, ck,
                                       st))
        log(f"side sweep, {size} trials")
        sw = run_side_sweep(fit, size, lists[size], keep, ck, st)
        sweeps.append(sw)
        prec_rows += precision_table(sw, keep, size)
        for b in SETS:              # free the big response matrices
            for k in ("Yr", "est_r", "Ys", "est_s"):
                keep[b].pop(k, None)

    thr = pd.DataFrame(thr_rows)
    prec = pd.DataFrame(prec_rows)
    sweep = pd.concat(sweeps, ignore_index=True)
    order = pd.concat(orders, ignore_index=True)
    rule = apply_rule(sweep, prec)
    log(f"rule: (a) {rule['a']}, (b) {rule['b']} -> {rule['chosen']}")

    curve_rows = []
    for size in BLOCK_SIZES:
        st = {b: cap.Streams(RANDOM_SEED, STREAM_KEY, size, b) for b in SETS}
        log(f"detection curve, {size} trials ({rule['chosen']} model)")
        curve_rows += run_curve(fit, rule["chosen"], size, lists[size],
                                keeps[size], st)
    curve = pd.DataFrame(curve_rows)
    z = curve[curve["true_weight"] == 0.0]
    se = np.sqrt(THRESHOLD_ALPHA * (1 - THRESHOLD_ALPHA) / N_PEOPLE)
    worst = float(np.max(np.abs(np.r_[z["sat_detected"], z["con_detected"],
                                      z["sat_opposite"], z["con_opposite"]]
                                - THRESHOLD_ALPHA)))
    ck.add("detection curve", "at true weight 0 the share beyond each "
           f"threshold is {THRESHOLD_ALPHA} within {MC_Z:g} SE + threshold "
           "error", worst <= MC_Z * se + 0.004,
           f"largest |rate - {THRESHOLD_ALPHA}| {worst:.4f} (SE {se:.4f})")
    mono = all(np.all(np.diff(g.sort_values("true_weight")[f"{f}_detected"]
                              .to_numpy()) > -MC_Z * 0.005)
               for _, g in curve.groupby(["size", "set", "lapse"])
               for f in FEATURES)
    ck.add("detection curve", "detection rises with the true weight (within "
           "Monte Carlo error)", mono, "")

    # ---- write ----
    thr.to_csv(os.path.join(out, "thresholds.csv"), index=False)
    prec.to_csv(os.path.join(out, "precision_detection.csv"), index=False)
    sweep.to_csv(os.path.join(out, "side_sweep.csv"), index=False)
    order.to_csv(os.path.join(out, "order_check.csv"), index=False)
    curve.to_csv(os.path.join(out, "detection_curve.csv"), index=False)
    if MAKE_FIGURES:
        try:
            fig_curve(curve, rule["chosen"],
                      os.path.join(out, "detection_curve.png"))
            fig_side(sweep, os.path.join(out, "side_sweep.png"))
        except ImportError:
            log("matplotlib not installed: no figures")
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(jsonable(settings_dict()), f, indent=2)
    ck.frame().to_csv(os.path.join(out, "checks.csv"), index=False)

    files = sorted(os.listdir(out)) + ["report.txt", "run_meta.json"]
    meta = {"started": started, "minutes": (time.time() - T0) / 60,
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "files": sorted(set(files))}
    write_report(out, ck, thr, prec, sweep, order, curve, rule, meta)
    with open(os.path.join(out, "run_meta.json"), "w") as f:
        json.dump(jsonable({
            **{k: v for k, v in meta.items() if k != "files"},
            "output_dir": out, "platform": platform.platform(),
            "input_sha256": file_hashes,
            "chosen_model": rule["chosen"],
            "rule": {k: v for k, v in rule.items() if k != "loss_table"},
            "checks_pass": len(ck.rows) - ck.n_fail,
            "checks_fail": ck.n_fail}), f, indent=2)
    log(f"done: {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL; "
        f"recommended model: {rule['chosen']}")
    log(f"report: {os.path.join(out, 'report.txt')}")


if __name__ == "__main__":
    main()
