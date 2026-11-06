"""
explore_block_length_v5.py  --  v5 step 2: how many trials per block?

WHAT THIS ANSWERS
    Script 1 showed that two matched sets of pairs exist at every size and
    mix tested, and p65 was dropped. This script asks what each candidate
    block length buys, by simulating participants on concrete block lists:

      * PRECISION of each weight: the spread of the Firth estimate, for a
        person at the planning weights and for a random chooser;
      * THRESHOLDS: the 2.5% and 97.5% percentiles of the weights that
        random choosers reach by luck (N_THRESHOLD_PEOPLE per list);
      * DETECTION: the share of people at the planning weights (saturation
        0.5, contrast 0.7 per corpus SD, lapse 0.1) whose weight clears the
        upper threshold;
      * PAIRED-TEST POWER: a first estimate of the power of a one-sided
        paired test on each person's weight difference between the two
        blocks, at N = 10, 15 and 20, under the v4 preregistration's
        planning scenario (primary) and the other scenarios in SETTINGS;
      * CHOICE PROBABILITY: the share of predicted-image choices the
        planning weights give on a typical informative trial of each list,
        against v4's interpretation (0.6191 and 0.6104);
      * SEPARATION of the two weights (the correlation of their estimates,
        and the number of conflict trials), and the SESSION LENGTH of the
        two pair blocks, built from the experiment's fixed durations plus
        an assumed response time.

    Contrast is the bottleneck (its trials carry about a third of the
    saturation information), so every contrast figure is reported first.

WHERE THE BLOCK LISTS COME FROM
    Not from saved example sets. For every size x mix, N_DRAWS pairs of
    matched sets are drawn with script 1's own functions, imported from
    analyse_pair_capacity_v5.py (stage1, one_draw, evaluate, ...), under
    script 1's settings for the level set named in LEVEL_SET. The draw
    seeds are script 1's, so draw 0 is the example set script 1 saved; the
    report checks that when SCRIPT1_RUN_DIR is given. Every figure is
    reported as the mean over draws with its range [min-max], so the
    spread between draws is visible. The v4 frozen list (180 analysed
    trials) is run alongside as a reference only.

REUSED (imported, not copied, so there is one definition of each)
    analyse_pair_capacity_v5.py
        load_pairs, build_pool, stage1, one_draw, evaluate,
        largest_remainder, Checks, Streams        the matched draws
    simulate_power_v4_n10_20.py
        fit_firth_batch, fit_firth                the registered estimator
        simulate_many                             random choosers
        simulate_lapsing (lapse_p)                responses with lapses:
                                                  p = (1-l) logistic + l/2
        compute_floors's rule                     thresholds = percentiles
                                                  of random choosers (its
                                                  seed is reused for the v4
                                                  reproduction check)

THE PERSON MODEL FOR THE PAIRED TEST (all settings)
    Each simulated person has a stable TASTE on each weight, the same in
    both blocks, drawn from Normal(0, TASTE_SD); TASTE_SD is v4's assumed
    non-responder spread (0.5 x the registered v4 thresholds). A share of
    people are RESPONDERS. A responder may carry a SHARED preference, the
    same in both blocks, and adds a SHIFT drawn from Normal(shift mean,
    shift SD) in the TARGET block only (derealization episode in the
    report group; the comparison group's analogue is the same arithmetic).
    A non-responder has taste only. The REFERENCE block (ordinary episode)
    carries taste plus any shared preference. The difference between the
    blocks therefore holds only the shift.

    "planning" is the v4 preregistration's planning scenario: half the
    group respond, with a derealization-specific shift of 0.5 (saturation)
    and 0.7 (contrast) per corpus SD; the rest have taste only; lapse 0.1.
    "half shift" is the same people, but half of each responder's
    preference is shared by both episodes, so only half of it separates
    the blocks. The others are sensitivity rows.
    Every person lapses at LAPSE in both blocks. Each block's weights are
    fitted separately with Firth, and the per-person difference
    (target - reference) is tested one-sided against zero across the N
    people. Which set goes with the target block alternates between people
    (the counterbalancing), and every study is split as evenly as N allows.
    Block order effects are NOT modelled.

INPUTS (run from the project folder; both imported scripts must be there)
    analyse_pair_capacity_v5.py, simulate_power_v4_n10_20.py
    the inputs script 1 reads (all_pairs.csv, optionally the manifest)
    V4_TRIAL_LIST    data/frozen_trial_list_v4/trial_list.csv
    SCRIPT1_RUN_DIR  optional, for the draw-0 reproduction check

OUTPUTS (one new folder per run, never overwritten)
    block_length_report.txt   what was done, what was found, every check
    summary_by_combination.csv   one row per size x mix (+ v4 reference)
    lists.csv                 one row per block list (draw x set)
    paired_power.csv          one row per draw x scenario x N x test
    session_length.csv        trials, breaks and minutes per size and
                              response-time assumption (pair blocks only)
    checks.csv                every check, PASS or FAIL
    paired_power_con.png, paired_power_sat.png
                              power against block size per scenario
                              (if matplotlib is installed)
    settings.json             every setting this run used

REQUIRES  numpy, pandas, scipy >= 1.9; matplotlib for the figure only
RUN       python explore_block_length_v5.py
"""

# ============================================================
# SETTINGS
# ============================================================

# --- Imported scripts (in the project folder) ---
CAPACITY_SCRIPT = "analyse_pair_capacity_v5.py"
V4_SIM_SCRIPT = "simulate_power_v4_n10_20.py"

# CAPACITY_OVERRIDES
#   Settings of script 1 to change for this run only, e.g.
#   {"MAX_PER_SCENE_PER_TYPE": 3}. Empty = script 1's settings as saved in
#   its file. Every setting script 1 runs with is written to settings.json.
CAPACITY_OVERRIDES = {}

# SCRIPT1_RUN_DIR
#   Script 1's output folder for the same settings. If its example sets are
#   there, draw 0 of every combination is checked against them. None or a
#   missing folder skips that check.
SCRIPT1_RUN_DIR = ("data/pair_capacity_v5/disjoint-pair_match-scene_bin_cap2"
                   "_satcut31.1_contol0.72_sizes120-150-180_levels-without_p65"
                   "_draws20_seed42__s4d22db73")

# --- The v4 reference ---
V4_TRIAL_LIST = "data/frozen_trial_list_v4/trial_list.csv"
INCLUDE_V4_REFERENCE = True

# V4_REGISTERED_THRESHOLDS / V4_THRESHOLD_TOL
#   The registered v4 thresholds (saturation, contrast). Recomputing them on
#   the frozen list with v4's own seed must reproduce them within the
#   tolerance: a check that the estimator and the threshold rule here are
#   the registered ones.
V4_REGISTERED_THRESHOLDS = {"upper": (0.286, 0.488),
                            "lower": (-0.290, -0.486)}
V4_THRESHOLD_TOL = 0.005

# --- What to explore ---
LEVEL_SET = "without_p65"              # a key of script 1's LEVEL_SETS
BLOCK_SIZES = [120, 150, 180]          # analysed trials per block
MIXES = ["conflict_heavy", "v4"]       # keys of script 1's TYPE_MIXES

# N_DRAWS
#   Pairs of matched sets drawn per size x mix. Every quantity is reported
#   as the mean over draws and its range, so this sets how well the spread
#   between draws is seen.
N_DRAWS = 20

# --- People ---
#
# PLANNING_WEIGHTS / LAPSE
#   Saturation and contrast weights per corpus SD, and the lapse rate (share
#   of trials answered at random), for detection and precision.
PLANNING_WEIGHTS = (0.5, 0.7)
LAPSE = 0.1

# DETECTION_LAPSES
#   Detection is reported at each of these lapse rates; LAPSE is the
#   planning one and must be among them.
DETECTION_LAPSES = [0.0, 0.1]

# N_THRESHOLD_PEOPLE / THRESHOLD_ALPHA
#   Random choosers per block list, and the share per tail. 20,000 per list,
#   as in v4: the thresholds are percentiles, so the tails need many people.
N_THRESHOLD_PEOPLE = 20_000
THRESHOLD_ALPHA = 0.025

# N_DETECTION_PEOPLE
#   People at fixed weights per list and lapse rate, for detection and
#   precision (Monte Carlo SE of a detection rate <= 0.005).
N_DETECTION_PEOPLE = 10_000

# --- The paired test ---
#
# TASTE_SD
#   Between-person SD of a stable taste on each weight (saturation,
#   contrast), the same in both blocks. v4's assumed non-responder spread:
#   NONRESPONDER_SPREAD_MULT 0.5 x the registered v4 upper thresholds
#   (0.286, 0.488).
TASTE_SD = (0.143, 0.244)

# PAIRED_SCENARIOS
#   name -> per responder: "shift" (sat, con), the target-block-only part,
#   with between-responder SD "shift_sd"; "shared" (sat, con), a preference
#   present in BOTH blocks, which cancels in the difference; and "share",
#   the share of responders. Everyone also has taste (TASTE_SD) and lapses
#   at LAPSE. Listed in report order: the "no effect" scenario checks the
#   false-positive rate (not called "null": pandas reads that word back
#   from a CSV as missing); then the headline scenarios; then sensitivity.
#
#   planning    the v4 preregistration's planning scenario: half respond,
#               shift 0.5 / 0.7 per SD between the derealization block and
#               the ordinary block, the rest taste only, lapse 0.1.
#   half shift  the same responders, but half of each one's preference is
#               shared by both episodes, so the shift between blocks is
#               0.25 / 0.35.
#   all respond ...  the sensitivity rows (the first run's scenarios, with
#               every person a responder; "all respond" was called
#               "planning" in the first run).
PAIRED_SCENARIOS = {
    "no effect":  {"shift": (0.0, 0.0), "shift_sd": (0.0, 0.0),
                   "shared": (0.0, 0.0), "share": 1.0},
    "planning":   {"shift": (0.5, 0.7), "shift_sd": (0.0, 0.0),
                   "shared": (0.0, 0.0), "share": 0.5},
    "half shift": {"shift": (0.25, 0.35), "shift_sd": (0.0, 0.0),
                   "shared": (0.25, 0.35), "share": 0.5},
    "all respond": {"shift": (0.5, 0.7), "shift_sd": (0.0, 0.0),
                    "shared": (0.0, 0.0), "share": 1.0},
    "all respond, spread 50%": {"shift": (0.5, 0.7),
                                "shift_sd": (0.25, 0.35),
                                "shared": (0.0, 0.0), "share": 1.0},
    "all respond, half shift, spread 50%": {"shift": (0.25, 0.35),
                                            "shift_sd": (0.125, 0.175),
                                            "shared": (0.0, 0.0),
                                            "share": 1.0},
}
NULL_SCENARIO = "no effect"
PRIMARY_SCENARIO = "planning"

# HEADLINE_SCENARIOS
#   Reported in every table, primary first. The rest of PAIRED_SCENARIOS
#   (apart from NULL_SCENARIO) are sensitivity rows, in sections 5 and 6.
HEADLINE_SCENARIOS = ("planning", "half shift")

PAIRED_N = [10, 15, 20]

# PAIRED_TESTS / PRIMARY_TEST / ALPHA
#   "t": one-sided paired t-test on the per-person differences.
#   "wilcoxon": one-sided Wilcoxon signed-rank (exact at these N), which does
#   not lean on the size of a few extreme differences.
#   Each weight is tested on its own at ALPHA, one-sided, uncorrected; "both"
#   is the share of studies where both weights are significant.
PAIRED_TESTS = ("t", "wilcoxon")
PRIMARY_TEST = "t"
ALPHA = 0.05

# POOL_PEOPLE / N_STUDIES
#   Per draw and scenario, POOL_PEOPLE people are simulated and fitted once
#   (both blocks); each of N_STUDIES studies then samples its N people from
#   that pool with replacement. Pool people are independent draws from the
#   person model, so this is the same as simulating each study afresh
#   (Monte Carlo SE of a power <= 0.008).
POOL_PEOPLE = 10_000
N_STUDIES = 4_000

# POWER_TARGET
#   The report lists, per mix and N, the smallest block size whose power
#   (primary scenario and test) reaches this. A reading aid, not a rule.
POWER_TARGET = 0.80

# --- What the planning weights mean (v4's interpretation) ---
#
# INFORMATIVE_SD
#   A trial is informative about a weight when |its predictor| >= this (SD
#   units): v4's CAP_INFORMATIVE_SD. The typical informative trial is the
#   median |predictor| among those trials (v4's informative_medians).
# V4_CHOICE_PROB / CHOICE_PROB_TOL
#   v4's interpretation of the planning weights: on a typical informative
#   trial of the frozen list a responder picks the predicted image with
#   probability 0.6191 (saturation 0.5) and 0.6104 (contrast 0.7), before
#   lapses (v4 design document, section 12). The v4 list must reproduce
#   these, and each new combination's mean must lie within the tolerance.
INFORMATIVE_SD = 0.20
V4_CHOICE_PROB = (0.6191, 0.6104)
CHOICE_PROB_TOL = 0.02

# --- Checks ---
SET_THRESHOLD_TOL = 0.05   # the two matched sets' thresholds, relative diff
EST_CORR_MAX = 0.30        # |corr(saturation, contrast estimates)| under null
EXTREME_WEIGHT = 5.0       # |weight| per SD counted as extreme ...
EXTREME_SHARE_MAX = 0.001  # ... and the largest share allowed
# TYPE1_FWER
#   The false-positive check: in every cell (size x mix x N x test x weight)
#   the mean rejection rate under NULL_SCENARIO must not exceed
#   ALPHA + z * SE, where SE is the between-draw standard error (draws are
#   independent; studies within a draw share a pool and are not) and z is
#   Bonferroni-corrected over all cells for this familywise rate.
TYPE1_FWER = 0.01

# --- Session length: an ESTIMATE of the two PAIR BLOCKS only ---
#
# Built from its parts. It covers the two pair blocks (their instructions,
# practice, trials and breaks) and nothing else: not consent, calibration,
# the grid blocks, word ratings or questionnaires. It is to be replaced by
# a timed pilot.
#
# Trial counts per block on top of the analysed trials. These only feed the
# estimate; script 3 sets the real counts.
PRACTICE_BY_BLOCK = (6, 2)     # first block, second block
CATCH_PER_BLOCK = 5
CONSISTENCY_PER_BLOCK = 8

# Fixed durations, copied from the experiment's config.js
# (github.com/zarcolmarioa/dpdr_experiment, branch v4-study-design, commit
# 1bbbefb, 2026-10-30), and how src/phases/pairs.js uses them:
#   image trial   fixation FIXATION_MS, then the pair until a key (self-
#                 paced, no timeout), then a blank BLANK_MS
#   practice      the first PRACTICE_FEEDBACK_TRIALS practice trials add a
#                 feedback screen of PRACTICE_FEEDBACK_MS (first block only
#                 here, FEEDBACK_IN_BLOCK)
#   catch trial   no fixation, no blank: CATCH_LOCKOUT_MS, then one key
#   breaks        a break screen before every BREAK_EVERY-th main trial
#                 (never before the first): floor((n_main - 1) / BREAK_EVERY)
#                 per block, assuming the experiment applies it per block
FIXATION_MS = 500
BLANK_MS = 250
PRACTICE_FEEDBACK_MS = 900
PRACTICE_FEEDBACK_TRIALS = 2
FEEDBACK_IN_BLOCK = (True, False)
CATCH_LOCKOUT_MS = 750
BREAK_EVERY = 41

# Assumed, not from the configuration (self-paced parts):
# RESPONSE_TIME_S   image trial, pair onset to key press, per assumption
# CATCH_RESPONSE_S  catch trial, end of lockout to key press
# SCREEN_S          each self-paced screen; "block instructions" is the
#                   per-block introduction (which episode to bring to mind)
RESPONSE_TIME_S = {"fast": 2.0, "typical": 3.0, "slow": 4.0}
CATCH_RESPONSE_S = {"fast": 1.0, "typical": 1.5, "slow": 2.5}
SCREEN_S = {"block instructions": 45, "practice intro": 20,
            "practice end": 10, "break": 30}

# V4_SESSION
#   The same model applied to the v4 session (6 practice, 205 main trials of
#   which 10 catch, 4 breaks, one block), for comparison with the v4 freeze
#   report's 14 to 18 minutes. Reported, not checked.
V4_SESSION = {"practice": 6, "main": 205, "catch": 10,
              "freeze_report_minutes": (14, 18)}

MAKE_FIGURE = True
OUTPUT_ROOT = "data/block_length_v5"
ON_EXISTING = "new"
RANDOM_SEED = 42


# ============================================================
# Imports
# ============================================================

import hashlib
import importlib.util
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

W_NAMES = ("sat", "con")


def load_module(path, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Run this script from the project folder, "
            "where the v4 and v5 scripts are.")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# Settings and output folder
# ============================================================

def settings_dict(cap):
    own = {k: v for k, v in globals().items()
           if k.isupper() and not k.startswith("_") and k != "W_NAMES"}
    script1 = {k: getattr(cap, k) for k in sorted(dir(cap))
               if k.isupper() and not k.startswith("_")}
    return {"script2": own, "script1_in_force": script1}


def check_settings(cap):
    if LEVEL_SET not in cap.LEVEL_SETS:
        raise ValueError(f"LEVEL_SET {LEVEL_SET!r} is not in script 1's "
                         f"LEVEL_SETS {list(cap.LEVEL_SETS)}")
    for m in MIXES:
        if m not in cap.TYPE_MIXES:
            raise ValueError(f"mix {m!r} is not in script 1's TYPE_MIXES")
    if LAPSE not in DETECTION_LAPSES:
        raise ValueError("LAPSE must be one of DETECTION_LAPSES")
    for s in (NULL_SCENARIO, PRIMARY_SCENARIO) + tuple(HEADLINE_SCENARIOS):
        if s not in PAIRED_SCENARIOS:
            raise ValueError(f"{s!r} is not in PAIRED_SCENARIOS")
    if HEADLINE_SCENARIOS[0] != PRIMARY_SCENARIO:
        raise ValueError("HEADLINE_SCENARIOS must start with PRIMARY_SCENARIO")
    for k, v in PAIRED_SCENARIOS.items():
        if set(v) != {"shift", "shift_sd", "shared", "share"}:
            raise ValueError(f"scenario {k!r} needs shift, shift_sd, shared "
                             "and share")
    if PRIMARY_TEST not in PAIRED_TESTS:
        raise ValueError("PRIMARY_TEST must be one of PAIRED_TESTS")
    for k in CAPACITY_OVERRIDES:
        if not hasattr(cap, k):
            raise ValueError(f"CAPACITY_OVERRIDES names {k!r}, which script "
                             "1 does not have")


def resolve_output_dir(cap):
    blob = json.dumps(settings_dict(cap), sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode()).hexdigest()[:8]
    name = (f"{LEVEL_SET}_sizes{'-'.join(map(str, BLOCK_SIZES))}"
            f"_mixes-{'-'.join(MIXES)}_draws{N_DRAWS}"
            f"_w{PLANNING_WEIGHTS[0]:g}-{PLANNING_WEIGHTS[1]:g}"
            f"_lapse{LAPSE:g}_null{N_THRESHOLD_PEOPLE}_seed{RANDOM_SEED}"
            f"__s{h}")
    base = os.path.join(OUTPUT_ROOT, name)
    if not os.path.exists(base):
        return base
    if ON_EXISTING == "error":
        raise FileExistsError(f"{base} exists")
    n = 2
    while os.path.exists(f"{base}_{n}"):
        n += 1
    print(f"  NOTE: {base} exists; writing to {base}_{n} instead.")
    return f"{base}_{n}"


# ============================================================
# Simulation pieces
# ============================================================

def fit(v4, X, Y):
    """Firth fit of many people; returns estimates and the share unconverged.
    v4.fit_firth_batch, the registered estimator."""
    b, last = v4.fit_firth_batch(X, Y)
    return b, float(np.mean(last > 1e-6))


def thresholds(v4, X, rng):
    """Two-sided thresholds from random choosers (v4's compute_floors rule:
    percentiles, not 1.96 SD), plus the null spread and the correlation of
    the two estimates."""
    est, unconv = fit(v4, X, v4.simulate_many(
        X, np.zeros((N_THRESHOLD_PEOPLE, 2)), rng))
    return {
        "upper": np.percentile(est, 100 * (1 - THRESHOLD_ALPHA), axis=0),
        "lower": np.percentile(est, 100 * THRESHOLD_ALPHA, axis=0),
        "null_sd": est.std(axis=0),
        "null_corr": float(np.corrcoef(est[:, 0], est[:, 1])[0, 1]),
        "unconverged": unconv,
        "extreme": float(np.mean(np.abs(est) > EXTREME_WEIGHT)),
    }


def detection(v4, X, th, lapse, rng):
    """People at PLANNING_WEIGHTS: share above the upper threshold, and the
    mean and SD of their estimates."""
    W = np.tile(np.asarray(PLANNING_WEIGHTS, float), (N_DETECTION_PEOPLE, 1))
    est, unconv = fit(v4, X, v4.simulate_lapsing(X, W, lapse, rng))
    return {
        "det": (est > th["upper"]).mean(axis=0),
        "mean": est.mean(axis=0), "sd": est.std(axis=0),
        "unconverged": unconv,
        "extreme": float(np.mean(np.abs(est) > EXTREME_WEIGHT)),
    }


def paired_pool(v4, XA, XB, scen, rng):
    """POOL_PEOPLE people, each fitted on both blocks.

    Returns (D, on_A, extreme share, unconverged share): D is (people, 2),
    the target-block estimate minus the reference-block estimate; on_A says
    whether set A was that person's target block (even people: A)."""
    P = POOL_PEOPLE
    taste = rng.normal(0.0, 1.0, (P, 2)) * np.asarray(TASTE_SD)
    resp = rng.random(P) < scen["share"]
    shift = (np.asarray(scen["shift"])
             + rng.normal(0.0, 1.0, (P, 2)) * np.asarray(scen["shift_sd"]))
    shift = np.where(resp[:, None], shift, 0.0)
    shared = np.where(resp[:, None], np.asarray(scen["shared"]), 0.0)
    W_ref = taste + shared
    W_tgt = W_ref + shift
    on_A = (np.arange(P) % 2) == 0
    est_t = np.empty((P, 2))
    est_r = np.empty((P, 2))
    ext, unc = [], []
    for m, Xt, Xr in ((on_A, XA, XB), (~on_A, XB, XA)):
        et, u1 = fit(v4, Xt, v4.simulate_lapsing(Xt, W_tgt[m], LAPSE, rng))
        er, u2 = fit(v4, Xr, v4.simulate_lapsing(Xr, W_ref[m], LAPSE, rng))
        est_t[m], est_r[m] = et, er
        ext += [np.mean(np.abs(et) > EXTREME_WEIGHT),
                np.mean(np.abs(er) > EXTREME_WEIGHT)]
        unc += [u1, u2]
    return est_t - est_r, on_A, float(np.mean(ext)), float(np.mean(unc))


def study_index(on_A, N, rng):
    """N_STUDIES x N indices into the pool, each study split between the two
    counterbalancing arms as evenly as N allows (an odd extra alternates
    between arms across studies). Sampled with replacement: pool people are
    independent draws from the person model."""
    a = np.flatnonzero(on_A)
    b = np.flatnonzero(~on_A)
    S = N_STUDIES
    half = N // 2
    nA = np.full(S, half)
    if N % 2:
        nA[: S // 2] += 1
    idx = np.empty((S, N), dtype=int)
    for k in np.unique(nA):
        rows = np.flatnonzero(nA == k)
        ia = a[rng.integers(0, len(a), (len(rows), k))]
        ib = b[rng.integers(0, len(b), (len(rows), N - k))]
        idx[rows] = np.concatenate([ia, ib], axis=1)
    return idx


_WILCOXON_SF = {}


def wilcoxon_sf_table(n):
    """Exact null distribution of the signed-rank statistic W+ for n
    differences with no ties or zeros: P(W+ >= w) for w = 0..n(n+1)/2,
    by counting subsets of the ranks 1..n with each sum."""
    if n not in _WILCOXON_SF:
        top = n * (n + 1) // 2
        counts = np.zeros(top + 1)
        counts[0] = 1.0
        for r in range(1, n + 1):
            counts[r:] = counts[r:] + counts[:-r].copy()
        pmf = counts / counts.sum()
        _WILCOXON_SF[n] = pmf[::-1].cumsum()[::-1]
    return _WILCOXON_SF[n]


def wilcoxon_greater(d):
    """One-sided exact Wilcoxon signed-rank p-values, one per row of d
    (H1: location > 0). Vectorised: SciPy's version is the reference and is
    checked against this at start-up. Continuous differences have no ties
    or zeros, which the exact distribution assumes."""
    n = d.shape[1]
    ranks = np.argsort(np.argsort(np.abs(d), axis=1), axis=1) + 1
    wplus = np.where(d > 0, ranks, 0).sum(axis=1)
    return wilcoxon_sf_table(n)[wplus]


def paired_tests(D, idx):
    """One-sided p-values (H1: mean difference > 0) per study and weight."""
    out = {}
    N = idx.shape[1]
    for w in range(2):
        d = D[idx, w]                                     # studies x N
        if "t" in PAIRED_TESTS:
            m = d.mean(1)
            s = d.std(1, ddof=1)
            t = m / (s / np.sqrt(N))
            out[("t", w)] = stats.t.sf(t, N - 1)
        if "wilcoxon" in PAIRED_TESTS:
            out[("wilcoxon", w)] = wilcoxon_greater(d)
    return out


# ============================================================
# Running one combination
# ============================================================

def info_of(X):
    xs, xc = X[:, 0], X[:, 1]
    cos = float((xs * xc).sum() / np.sqrt((xs ** 2).sum() * (xc ** 2).sum()))
    return float((xs ** 2).sum()), float((xc ** 2).sum()), cos


def informative_medians(X):
    """Median |predictor| among the trials informative about each weight
    (|predictor| >= INFORMATIVE_SD): v4's informative_medians, with the
    threshold as a setting. Checked against v4's function on the v4 list."""
    out = {}
    for k, w in enumerate(W_NAMES):
        a = np.abs(X[:, k])
        info = a >= INFORMATIVE_SD
        out[w] = float(np.median(a[info])) if info.any() else np.nan
        out[w + "_n"] = int(info.sum())
    return out


def choice_probability(v4, X):
    """The share of predicted-image choices the planning weights give on a
    typical informative trial: v4.as_probability(weight, median), and the
    same after lapses, (1 - LAPSE) p + LAPSE / 2."""
    med = informative_medians(X)
    out = {}
    for k, w in enumerate(W_NAMES):
        p = v4.as_probability(PLANNING_WEIGHTS[k], med[w])
        out[f"inf_median_{w}"] = med[w]
        out[f"inf_n_{w}"] = med[w + "_n"]
        out[f"p_choice_{w}"] = p
        out[f"p_choice_{w}_lapse"] = (1 - LAPSE) * p + LAPSE / 2
    return out


def analyse_list(v4, X, rng_th, rng_det, label_row):
    """Thresholds, precision, detection and choice probability for one
    block list."""
    th = thresholds(v4, X, rng_th)
    isat, icon, cos = info_of(X)
    row = dict(label_row)
    row.update({"n_analysed": len(X), "info_sat": isat, "info_con": icon,
                "cos": cos})
    row.update(choice_probability(v4, X))
    for k, w in enumerate(W_NAMES):
        row[f"thr_upper_{w}"] = th["upper"][k]
        row[f"thr_lower_{w}"] = th["lower"][k]
        row[f"null_sd_{w}"] = th["null_sd"][k]
    row["null_est_corr"] = th["null_corr"]
    unconv, extreme = [th["unconverged"]], [th["extreme"]]
    for lp in DETECTION_LAPSES:
        d = detection(v4, X, th, lp, rng_det)
        for k, w in enumerate(W_NAMES):
            row[f"det_{w}_lapse{lp:g}"] = d["det"][k]
            row[f"plan_mean_{w}_lapse{lp:g}"] = d["mean"][k]
            row[f"plan_sd_{w}_lapse{lp:g}"] = d["sd"][k]
        unconv.append(d["unconverged"])
        extreme.append(d["extreme"])
    row["unconverged_share"] = max(unconv)
    row["extreme_share"] = max(extreme)
    return th, row


def analyse_pair_of_lists(v4, XA, XB, base_row, st, paired_rows):
    for sname, scen in PAIRED_SCENARIOS.items():
        D, on_A, ext, unc = paired_pool(v4, XA, XB, scen,
                                        st(f"pool|{sname}"))
        for N in PAIRED_N:
            idx = study_index(on_A, N, st(f"studies|{sname}|{N}"))
            pv = paired_tests(D, idx)
            for test in PAIRED_TESTS:
                sig = [pv[(test, w)] < ALPHA for w in range(2)]
                row = dict(base_row)
                row.update({
                    "scenario": sname, "N": N, "test": test,
                    "power_sat": float(sig[0].mean()),
                    "power_con": float(sig[1].mean()),
                    "power_both": float((sig[0] & sig[1]).mean()),
                    "mean_diff_sat": float(D[:, 0].mean()),
                    "sd_diff_sat": float(D[:, 0].std()),
                    "mean_diff_con": float(D[:, 1].mean()),
                    "sd_diff_con": float(D[:, 1].std()),
                    "extreme_share": ext, "unconverged_share": unc})
                paired_rows.append(row)


def check_draw0(cap, sets, size, mix):
    """Draw 0 against script 1's saved example set, by pair keys."""
    if not SCRIPT1_RUN_DIR:
        return None
    ex = os.path.join(SCRIPT1_RUN_DIR, "example_sets",
                      f"{LEVEL_SET}_n{size}_{mix}")
    if not os.path.isdir(ex):
        return None
    for blk in ("A", "B"):
        saved = pd.read_csv(os.path.join(ex, f"set_{blk}.csv"))
        a = set(map(tuple, saved[["file_a", "file_b"]].to_numpy()))
        b = set(map(tuple, sets[blk][["file_a", "file_b"]].to_numpy()))
        if a != b:
            return False
    return True


# ============================================================
# Aggregation
# ============================================================

def rng_str(s, fmt="{:.3f}"):
    s = pd.Series(s).dropna()
    if not len(s):
        return "-"
    if s.min() == s.max():
        return fmt.format(s.mean())
    return (fmt.format(s.mean()) + " [" + fmt.format(s.min()) + "-"
            + fmt.format(s.max()) + "]")


def combo_key(r):
    return (r["size"], r["mix"])


def summarise(lists, paired):
    rows = []
    for (size, mix), g in lists.groupby(["size", "mix"], sort=False):
        row = {"size": size, "mix": mix, "lists": len(g),
               "draws": g.draw.nunique()}
        for c in ["info_sat", "info_con", "cos", "null_est_corr",
                  "thr_upper_sat", "thr_upper_con", "thr_lower_sat",
                  "thr_lower_con", "null_sd_sat", "null_sd_con"] + \
                [c for c in g.columns if c.startswith(("det_", "plan_"))]:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_min"] = g[c].min()
            row[f"{c}_max"] = g[c].max()
        p = paired[(paired["size"] == size) & (paired.mix == mix)]
        for (sc, N, test), h in p.groupby(["scenario", "N", "test"]):
            for w in ("con", "sat", "both"):
                k = f"power_{w}|{sc}|N{N}|{test}"
                row[k + "_mean"] = h[f"power_{w}"].mean()
                row[k + "_min"] = h[f"power_{w}"].min()
                row[k + "_max"] = h[f"power_{w}"].max()
        rows.append(row)
    return pd.DataFrame(rows)


def block_seconds(n_practice, n_main, n_catch, feedback, level):
    """Seconds for one pair block under one response-time assumption, and
    its number of breaks. n_main includes the catch trials."""
    rt = RESPONSE_TIME_S[level]
    image = (FIXATION_MS + BLANK_MS) / 1000 + rt
    n_image = n_practice + n_main - n_catch
    sec = n_image * image
    sec += n_catch * (CATCH_LOCKOUT_MS / 1000 + CATCH_RESPONSE_S[level])
    if feedback:
        sec += min(PRACTICE_FEEDBACK_TRIALS, n_practice) \
            * PRACTICE_FEEDBACK_MS / 1000
    breaks = (n_main - 1) // BREAK_EVERY if BREAK_EVERY > 0 else 0
    sec += breaks * SCREEN_S["break"]
    sec += SCREEN_S["block instructions"]
    if n_practice:
        sec += SCREEN_S["practice intro"] + SCREEN_S["practice end"]
    return sec, breaks


def session_rows():
    """Trials, breaks and minutes for the two pair blocks, per size and
    response-time assumption."""
    rows = []
    for n in BLOCK_SIZES:
        n_main = n + CONSISTENCY_PER_BLOCK + CATCH_PER_BLOCK
        row = {"analysed_per_block": n, "main_per_block": n_main,
               "practice_block_1": PRACTICE_BY_BLOCK[0],
               "practice_block_2": PRACTICE_BY_BLOCK[1],
               "trials_total": 2 * n_main + sum(PRACTICE_BY_BLOCK),
               "analysed_total": 2 * n}
        for level in RESPONSE_TIME_S:
            tot = 0.0
            for b in (0, 1):
                sec, brk = block_seconds(PRACTICE_BY_BLOCK[b], n_main,
                                         CATCH_PER_BLOCK,
                                         FEEDBACK_IN_BLOCK[b], level)
                row[f"minutes_block_{b + 1}_{level}"] = sec / 60
                row["breaks_per_block"] = brk
                tot += sec
            row[f"minutes_total_{level}"] = tot / 60
        rows.append(row)
    return pd.DataFrame(rows)


def v4_session_minutes():
    out = {}
    for level in RESPONSE_TIME_S:
        sec, brk = block_seconds(V4_SESSION["practice"], V4_SESSION["main"],
                                 V4_SESSION["catch"], True, level)
        out[level] = sec / 60
    out["breaks"] = brk
    return out


# ============================================================
# Figure
# ============================================================

def make_figure(paired, v4_paired, out_dir):
    """One PNG per weight: rows = the non-null scenarios, columns = N; power
    of the primary test against block size, one line per mix (mean over
    draws, bars = range over draws), the v4 list as a grey reference."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    # categorical slots 1-3 of the dataviz reference palette (light), which
    # validate for colour-vision deficiency as a set of three
    colours = ["#2a78d6", "#eb6834", "#1baf7a"]
    markers = ["o", "s", "^"]
    scen = [s for s in PAIRED_SCENARIOS if s != NULL_SCENARIO]
    paths = []
    for w, wname in (("con", "Contrast"), ("sat", "Saturation")):
        fig, axes = plt.subplots(len(scen), len(PAIRED_N),
                                 figsize=(3.6 * len(PAIRED_N),
                                          2.6 * len(scen) + 0.8),
                                 sharex=True, sharey=True, squeeze=False)
        for r, sc in enumerate(scen):
            p = paired[(paired.scenario == sc)
                       & (paired.test == PRIMARY_TEST)]
            for c, N in enumerate(PAIRED_N):
                ax = axes[r, c]
                for i, mix in enumerate(MIXES):
                    g = (p[(p.mix == mix) & (p.N == N)]
                         .groupby("size")[f"power_{w}"])
                    m, lo, hi = g.mean(), g.min(), g.max()
                    x = np.array(m.index, float) + (i - 1) * 1.5
                    ax.errorbar(x, m.values, yerr=[m.values - lo.values,
                                                   hi.values - m.values],
                                color=colours[i % 3], marker=markers[i % 3],
                                ms=5, lw=2, capsize=3, label=mix)
                if v4_paired is not None:
                    q = v4_paired[(v4_paired.N == N)
                                  & (v4_paired.scenario == sc)
                                  & (v4_paired.test == PRIMARY_TEST)]
                    if len(q):
                        ax.plot([180], [q[f"power_{w}"].iloc[0]],
                                marker="D", ms=5, color="#52514e", ls="none",
                                label="v4 list x2 (ref.)")
                ax.axhline(POWER_TARGET, color="#b5b3ab", lw=1, ls="--")
                ax.set_ylim(0, 1.03)
                ax.grid(axis="y", color="#e6e5e0", lw=0.8)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.set_title(f"{sc}, N = {N}", fontsize=9.5, loc="left")
                if c == 0:
                    ax.set_ylabel("power")
                if r == len(scen) - 1:
                    ax.set_xlabel("analysed trials per block")
        axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
        fig.suptitle(f"{wname}: one-sided paired {PRIMARY_TEST}-test, alpha "
                     f"{ALPHA} (mean over draws, bars = range)",
                     fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        path = os.path.join(out_dir, f"paired_power_{w}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


# ============================================================
# Report
# ============================================================

def scenario_role(name):
    if name == NULL_SCENARIO:
        return "null check"
    if name == PRIMARY_SCENARIO:
        return "primary"
    if name in HEADLINE_SCENARIOS:
        return "headline"
    return "sensitivity"


def write_report(out_dir, ck, log, lists, paired, summ, v4_rows, sess,
                 matching, t0, cap):
    L = []
    w = L.append
    w("=" * 78)
    w("v5 STEP 2: HOW MANY TRIALS PER BLOCK?")
    w("explore_block_length_v5.py")
    w("=" * 78)
    w(f"run finished  {time.strftime('%Y-%m-%d %H:%M:%S')}   elapsed "
      f"{time.time() - t0:.0f} s")
    w(f"output        {out_dir}")
    w("")
    w("1. WHAT THIS RUN DID")
    w("-" * 78)
    w(f"For each block size x type mix it drew {N_DRAWS} pairs of matched "
      "sets with script 1's")
    w("functions (level set "
      f"'{LEVEL_SET}'), and on every set (= one block list) it simulated:")
    w(f"  * {N_THRESHOLD_PEOPLE:,} random choosers -> two-sided thresholds "
      f"({THRESHOLD_ALPHA:.1%} per tail) and the")
    w("    null spread of each weight;")
    w(f"  * {N_DETECTION_PEOPLE:,} people at the planning weights "
      f"(saturation {PLANNING_WEIGHTS[0]}, contrast {PLANNING_WEIGHTS[1]} "
      "per SD)")
    w(f"    at each lapse rate {DETECTION_LAPSES} -> detection and "
      "precision;")
    w(f"and on each pair of sets, per scenario, {POOL_PEOPLE:,} people fitted "
      f"on both blocks, from which {N_STUDIES:,}")
    w(f"studies of N = {PAIRED_N} were sampled -> paired-test power. Firth "
      "throughout (v4's")
    w("fit_firth_batch). Figures are the mean over draws, with the range "
      "[min-max] over")
    w("draws (and over the two sets of each draw, for per-list figures).")
    w("")
    w("Person model for the paired test: stable taste ~ Normal(0, "
      f"{TASTE_SD}) in both blocks;")
    w("a responder adds a shared preference to both blocks and a shift to "
      "the target block")
    w(f"only; lapse {LAPSE} in both blocks; block order not modelled.")
    w(f"  {'scenario':36s} {'role':11s} {'responders':>10s}  shift "
      "(sd)                  shared")
    for k, s in PAIRED_SCENARIOS.items():
        w(f"  {k:36s} {scenario_role(k):11s} {s['share']:10.0%}  "
          f"{str(s['shift']):12s} ({s['shift_sd'][0]:g}, "
          f"{s['shift_sd'][1]:g})   {s['shared']}")
    w(f"Primary: scenario '{PRIMARY_SCENARIO}' (the v4 preregistration's "
      "planning scenario),")
    w(f"{PRIMARY_TEST}-test, alpha {ALPHA} one-sided per weight, "
      "uncorrected.")
    w("")
    w("Inputs and provenance")
    for line in log:
        w("  " + line)
    w("")

    def table(weight, wname):
        wi = W_NAMES.index(weight)
        w(f"{'size':>4s} {'mix':15s} {'info':>21s} {'threshold (upper)':>23s} "
          f"{'SD at plan.':>21s} {'detect, lapse ' + str(LAPSE):>21s}")
        for _, r in summ.iterrows():
            g = lists[(lists["size"] == r["size"]) & (lists.mix == r.mix)]
            w(f"{r['size']:4d} {r.mix:15s} "
              f"{rng_str(g[f'info_{weight}'], '{:.1f}'):>21s} "
              f"{rng_str(g[f'thr_upper_{weight}']):>23s} "
              f"{rng_str(g[f'plan_sd_{weight}_lapse{LAPSE:g}']):>21s} "
              f"{rng_str(g[f'det_{weight}_lapse{LAPSE:g}']):>21s}")
        if v4_rows is not None:
            r = v4_rows["list"]
            w(f"{180:4d} {'v4 frozen (ref)':15s} "
              f"{r[f'info_{weight}']:21.1f} {r[f'thr_upper_{weight}']:23.3f} "
              f"{r[f'plan_sd_{weight}_lapse{LAPSE:g}']:21.3f} "
              f"{r[f'det_{weight}_lapse{LAPSE:g}']:21.3f}")
        w("")
        for sc in HEADLINE_SCENARIOS:
            w(f"Paired-test power, {wname}, scenario '{sc}' "
              f"({scenario_role(sc)}), {PRIMARY_TEST}-test")
            w(f"{'size':>4s} {'mix':15s} "
              + " ".join(f"{'N = ' + str(N):>21s}" for N in PAIRED_N))
            pp = paired[(paired.scenario == sc)
                        & (paired.test == PRIMARY_TEST)]
            for _, r in summ.iterrows():
                g = pp[(pp["size"] == r["size"]) & (pp.mix == r.mix)]
                w(f"{r['size']:4d} {r.mix:15s} " + " ".join(
                    f"{rng_str(g[g.N == N][f'power_{weight}']):>21s}"
                    for N in PAIRED_N))
            if v4_rows is not None:
                q = v4_rows["paired"]
                q = q[(q.scenario == sc) & (q.test == PRIMARY_TEST)]
                w(f"{180:4d} {'v4 list x2 (ref)':15s} " + " ".join(
                    f"{q[q.N == N][f'power_{weight}'].iloc[0]:21.3f}"
                    for N in PAIRED_N))
            w("")
        _ = wi

    w("2. CONTRAST (the bottleneck)")
    w("-" * 78)
    w("Per block list: information (sum x_con^2), upper threshold, SD of the "
      "contrast")
    w("estimate for a responder at the planning weights, and the share of "
      "such people detected.")
    table("con", "contrast")
    w("3. SATURATION")
    w("-" * 78)
    table("sat", "saturation")

    w("4. WHAT THE PLANNING WEIGHTS MEAN ON THESE LISTS")
    w("-" * 78)
    w(f"Share of predicted-image choices a responder at saturation "
      f"{PLANNING_WEIGHTS[0]} / contrast {PLANNING_WEIGHTS[1]} per SD")
    w("makes on a TYPICAL INFORMATIVE trial: the median |predictor| among "
      f"trials with |predictor| >= {INFORMATIVE_SD}")
    w("(v4's definition). v4's interpretation: "
      f"{V4_CHOICE_PROB[0]:.4f} (saturation), {V4_CHOICE_PROB[1]:.4f} "
      f"(contrast), before lapses.")
    w(f"{'size':>4s} {'mix':15s} {'median |x_sat|':>20s} {'P(sat)':>20s} "
      f"{'median |x_con|':>20s} {'P(con)':>20s}")
    for _, r in summ.iterrows():
        g = lists[(lists["size"] == r["size"]) & (lists.mix == r.mix)]
        w(f"{r['size']:4d} {r.mix:15s} {rng_str(g.inf_median_sat):>20s} "
          f"{rng_str(g.p_choice_sat):>20s} {rng_str(g.inf_median_con):>20s} "
          f"{rng_str(g.p_choice_con):>20s}")
    if v4_rows is not None:
        r = v4_rows["list"]
        w(f"{180:4d} {'v4 frozen (ref)':15s} {r['inf_median_sat']:20.3f} "
          f"{r['p_choice_sat']:20.4f} {r['inf_median_con']:20.3f} "
          f"{r['p_choice_con']:20.4f}")
    g = lists
    w(f"After lapses ({LAPSE}): saturation {rng_str(g.p_choice_sat_lapse)}, "
      f"contrast {rng_str(g.p_choice_con_lapse)} over all lists.")
    w("Informative trials per list: saturation "
      f"{rng_str(g.inf_n_sat, '{:.0f}')}, contrast "
      f"{rng_str(g.inf_n_con, '{:.0f}')}.")
    w("")

    # smallest size reaching target
    w(f"5. SMALLEST BLOCK SIZE REACHING POWER {POWER_TARGET:.2f} "
      f"({PRIMARY_TEST}-test), per scenario")
    w("-" * 78)
    w("'mean' uses the mean power over draws; 'every draw' requires the "
      "worst draw to reach it;")
    w(f"'none' = no size up to {max(BLOCK_SIZES)} reaches it.")
    for sc in PAIRED_SCENARIOS:
        if sc == NULL_SCENARIO:
            continue
        pp = paired[(paired.scenario == sc) & (paired.test == PRIMARY_TEST)]
        w(f"[{sc}]  ({scenario_role(sc)})")
        for weight, wname in (("con", "contrast"), ("sat", "saturation"),
                              ("both", "both")):
            for mix in MIXES:
                parts = []
                for N in PAIRED_N:
                    g = pp[(pp.mix == mix) & (pp.N == N)].groupby("size")[
                        f"power_{weight}"]
                    m, lo = g.mean(), g.min()
                    a = [s for s in sorted(m.index) if m[s] >= POWER_TARGET]
                    b = [s for s in sorted(lo.index)
                         if lo[s] >= POWER_TARGET]
                    parts.append(f"N={N}: {a[0] if a else 'none':>4} / "
                                 f"{b[0] if b else 'none':>4}")
                w(f"  {wname:10s} {mix:15s} " + "   ".join(parts))
        w("")
    w("  (each cell: by mean / in every draw)")
    w("")

    w("6. PAIRED-TEST POWER, ALL SCENARIOS AND TESTS (mean over draws)")
    w("-" * 78)
    agg = (paired.groupby(["scenario", "test", "size", "mix", "N"],
                          sort=False)
           [["power_con", "power_sat", "power_both"]].mean().reset_index())
    for sc in PAIRED_SCENARIOS:
        for test in PAIRED_TESTS:
            w(f"[{sc}] ({scenario_role(sc)}) {test}-test        "
              "contrast / saturation / both")
            g = agg[(agg.scenario == sc) & (agg.test == test)]
            w(f"  {'size':>4s} {'mix':15s} " + " ".join(
                f"{'N=' + str(N):>20s}" for N in PAIRED_N))
            for (size, mix), h in g.groupby(["size", "mix"], sort=False):
                cells = []
                for N in PAIRED_N:
                    x = h[h.N == N].iloc[0]
                    cells.append(f"{x.power_con:.2f}/{x.power_sat:.2f}/"
                                 f"{x.power_both:.2f}")
                w(f"  {size:4d} {mix:15s} " + " ".join(f"{c:>20s}"
                                                       for c in cells))
            w("")
    dd = (paired[paired.test == PAIRED_TESTS[0]]
          .groupby(["scenario", "size", "mix"], sort=False)
          [["mean_diff_con", "sd_diff_con", "mean_diff_sat", "sd_diff_sat"]]
          .mean().reset_index())
    dd["_o"] = dd.scenario.map({k: i for i, k in
                                enumerate(PAIRED_SCENARIOS)})
    dd = dd.sort_values(["_o", "size"], kind="stable")
    w("Per-person weight difference (target - reference), mean and SD over "
      "the pool:")
    w(f"  {'scenario':36s} {'size':>4s} {'mix':15s} {'contrast':>17s} "
      f"{'saturation':>17s} {'dz con':>7s} {'dz sat':>7s}")
    for _, r in dd.iterrows():
        dzc = r.mean_diff_con / r.sd_diff_con if r.sd_diff_con else np.nan
        dzs = r.mean_diff_sat / r.sd_diff_sat if r.sd_diff_sat else np.nan
        w(f"  {r.scenario:36s} {r['size']:4d} {r.mix:15s} "
          f"{r.mean_diff_con:+7.3f} sd {r.sd_diff_con:5.3f} "
          f"{r.mean_diff_sat:+7.3f} sd {r.sd_diff_sat:5.3f} "
          f"{dzc:7.2f} {dzs:7.2f}")
    w("  (the mean difference is below the shift because lapses attenuate "
      "every fitted weight)")
    w("")

    w("7. SEPARATING THE TWO WEIGHTS")
    w("-" * 78)
    w("Conflict trials per block, and the correlation of the two estimates "
      "among random")
    w("choosers (how far one weight's error leaks into the other), and the "
      "predictor cos.")
    for _, r in summ.iterrows():
        counts = cap.largest_remainder(r["size"], cap.TYPE_MIXES[r.mix])
        g = lists[(lists["size"] == r["size"]) & (lists.mix == r.mix)]
        w(f"  {r['size']:4d} {r.mix:15s} conflict {counts['conflict']:3d}, "
          f"together {counts['together']:3d}   corr(estimates) "
          f"{rng_str(g.null_est_corr):>24s}   cos {rng_str(g.cos)}")
    if v4_rows is not None:
        r = v4_rows["list"]
        w(f"  {180:4d} {'v4 frozen (ref)':15s} conflict  30, together  30"
          f"   corr(estimates) {r['null_est_corr']:24.3f}   cos "
          f"{r['cos']:.3f}")
    w("")

    w("8. SESSION LENGTH OF THE TWO PAIR BLOCKS (AN ESTIMATE)")
    w("-" * 78)
    w("COVERS ONLY THE TWO PAIR BLOCKS: their instructions, practice, "
      "trials and breaks.")
    w("Not consent, calibration, the grid blocks, word ratings or "
      "questionnaires.")
    w("THE RESULT IS AN ESTIMATE, to be replaced by a timed pilot.")
    w("Fixed durations from the experiment's config.js (dpdr_experiment, "
      "v4-study-design, 1bbbefb):")
    w(f"  fixation {FIXATION_MS} ms + blank {BLANK_MS} ms per image trial; "
      f"catch lockout {CATCH_LOCKOUT_MS} ms; practice")
    w(f"  feedback {PRACTICE_FEEDBACK_MS} ms on the first "
      f"{PRACTICE_FEEDBACK_TRIALS} practice trials (first block); a break "
      f"every {BREAK_EVERY} main trials.")
    w("Assumed (self-paced): response time per image trial "
      + ", ".join(f"{k} {v:g} s" for k, v in RESPONSE_TIME_S.items())
      + ";")
    w("  catch response " + ", ".join(f"{k} {v:g} s" for k, v in
                                      CATCH_RESPONSE_S.items())
      + "; screens " + ", ".join(f"{k} {v:g} s" for k, v in
                                 SCREEN_S.items()) + ".")
    w(f"Per block: analysed + {CONSISTENCY_PER_BLOCK} consistency + "
      f"{CATCH_PER_BLOCK} catch, plus practice {PRACTICE_BY_BLOCK} "
      "(first, second block).")
    levels = list(RESPONSE_TIME_S)
    w(f"  {'per block':>9s} {'trials':>7s} {'breaks':>7s}   " + "   ".join(
        f"{lv + ' (b1+b2)':>22s}" for lv in levels))
    for _, r in sess.iterrows():
        cells = []
        for lv in levels:
            cells.append(f"{r[f'minutes_total_{lv}']:5.1f} min "
                         f"({r[f'minutes_block_1_{lv}']:.1f}+"
                         f"{r[f'minutes_block_2_{lv}']:.1f})")
        w(f"  {int(r.analysed_per_block):9d} {int(r.trials_total):7d} "
          f"{int(r.breaks_per_block):7d}   " + "   ".join(
              f"{c:>22s}" for c in cells))
    v4m = v4_session_minutes()
    w(f"  For comparison, the same model on the v4 session (one block, "
      f"{V4_SESSION['practice'] + V4_SESSION['main']} trials, "
      f"{v4m['breaks']} breaks): " + ", ".join(
          f"{lv} {v4m[lv]:.1f}" for lv in levels) + " min")
    w(f"  (v4's freeze report estimated {V4_SESSION['freeze_report_minutes'][0]}"
      f" to {V4_SESSION['freeze_report_minutes'][1]} min with its own "
      "per-trial assumption).")
    w("")

    if v4_rows is not None:
        w("9. THE v4 REFERENCE")
        w("-" * 78)
        r = v4_rows["list"]
        w("The frozen v4 list (180 analysed trials) as one block list. Its "
          "thresholds use v4's")
        w("own seed and reproduce the registered values (see the checks). "
          "For the paired test")
        w("the same list is used as BOTH blocks: a reference for "
          "information only, not a")
        w("valid design (it repeats every pair).")
        w(f"  thresholds  saturation {r['thr_upper_sat']:+.3f} / "
          f"{r['thr_lower_sat']:+.3f}, contrast {r['thr_upper_con']:+.3f} / "
          f"{r['thr_lower_con']:+.3f}")
        w(f"  detection at the planning weights, lapse {LAPSE}: saturation "
          f"{r[f'det_sat_lapse{LAPSE:g}']:.3f}, contrast "
          f"{r[f'det_con_lapse{LAPSE:g}']:.3f}")
        w("")

    w("10. CHECKS")
    w("-" * 78)
    cf = ck.frame()
    for row in cf.itertuples():
        w(f"  {row.status:7s} [{row.scope}] {row.check}"
          + (f"  ({row.detail})" if row.detail else ""))
    n_f = int((cf.status == "FAIL").sum())
    w("")
    w(f"  {int((cf.status == 'PASS').sum())} PASS, {n_f} FAIL, "
      f"{int((cf.status == 'SKIPPED').sum())} SKIPPED")
    w("")
    w("11. NOTES AND LIMITS")
    w("-" * 78)
    w("* Power is a first estimate under the stated person model. The shift "
      "sizes, the taste")
    w("  spread, the share of responders and the lapse rate are assumptions, "
      "not estimates;")
    w("  section 6 shows how far the answer moves when they change.")
    w("* Each weight is tested on its own at alpha, uncorrected. A corrected "
      "or joint primary")
    w("  test would need the power read from 'both' or recomputed.")
    w("* Block order is not modelled: an order effect would add to the "
      "difference in half")
    w("  the people and subtract in the other half, widening its spread.")
    w("* Taste is assumed stable across blocks. Taste that differed between "
      "blocks would")
    w("  add its own variance to every difference.")
    w("* The session length covers the two pair blocks only and is an "
      "estimate: the response")
    w("  times and screen times are assumptions, to be replaced by a timed "
      "pilot.")
    with open(os.path.join(out_dir, "block_length_report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    return n_f


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("v5 step 2: how many trials per block?")
    cap = load_module(CAPACITY_SCRIPT, "analyse_pair_capacity_v5")
    v4 = load_module(V4_SIM_SCRIPT, "simulate_power_v4_n10_20")
    for k, v in CAPACITY_OVERRIDES.items():
        setattr(cap, k, v)
    check_settings(cap)
    ck = cap.Checks()
    log = [f"script 1   {CAPACITY_SCRIPT}  sha256 "
           f"{sha256_of(CAPACITY_SCRIPT)[:16]}",
           f"v4 sim     {V4_SIM_SCRIPT}  sha256 "
           f"{sha256_of(V4_SIM_SCRIPT)[:16]}",
           f"overrides  {CAPACITY_OVERRIDES or 'none'}"]

    # --- estimator check: the batch Firth fit equals the single fit ---
    rng = np.random.default_rng(RANDOM_SEED)
    Xc = rng.normal(0, 1, (90, 2))
    Yc = v4.simulate_many(Xc, np.tile([0.5, 0.7], (20, 1)), rng)
    bb, _ = v4.fit_firth_batch(Xc, Yc)
    bs = np.array([v4.fit_firth(Xc, y) for y in Yc])
    diff = float(np.abs(bb - bs).max())
    ck.add("estimator", "batch Firth fit equals the single-person fit",
           diff < 1e-6, f"max |diff| {diff:.1e} over 20 people")

    # --- the vectorised Wilcoxon equals SciPy's exact test ---
    if "wilcoxon" in PAIRED_TESTS:
        worst = 0.0
        for n in PAIRED_N:
            d = rng.normal(0.3, 1.0, (200, n))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref = stats.wilcoxon(d, alternative="greater", axis=1,
                                     method="exact").pvalue
            worst = max(worst, float(np.abs(wilcoxon_greater(d) - ref)
                                     .max()))
        ck.add("estimator", "vectorised one-sided Wilcoxon equals SciPy's "
               "exact test", worst < 1e-10,
               f"max |diff| in p {worst:.1e}, 200 studies at each N")

    # --- pool, via script 1 ---
    s1_log = []
    p = cap.load_pairs(cap.Checks(), s1_log)
    log += s1_log
    pool = cap.build_pool(p, LEVEL_SET, cap.Checks(), [])
    pools_t = {t: pool[pool.type == t].reset_index(drop=True)
               for t in cap.TYPES}
    scenes = sorted(p.scene_id.unique())

    out_dir = resolve_output_dir(cap)
    os.makedirs(out_dir)
    print(f"  output: {out_dir}")

    # --- the v4 reference ---
    v4_rows = None
    if INCLUDE_V4_REFERENCE and os.path.isfile(V4_TRIAL_LIST):
        t = pd.read_csv(V4_TRIAL_LIST)
        a = t[t["analysed"].astype(str).str.lower().isin(["true", "1"])]
        X4 = a[["pred_sat", "pred_fog"]].to_numpy(float)
        log.append(f"v4 list    {V4_TRIAL_LIST}: {len(X4)} analysed trials")
        st = cap.Streams(RANDOM_SEED, "block_length", "v4_reference")
        th_rng = np.random.default_rng(v4.RANDOM_SEED + 101)  # v4's seed
        th4, row4 = analyse_list(v4, X4, th_rng, st("det"),
                                 {"size": 180, "mix": "v4 frozen",
                                  "draw": 0, "set": "-"})
        for kind in ("upper", "lower"):
            got = th4[kind]
            want = np.asarray(V4_REGISTERED_THRESHOLDS[kind])
            ck.add("v4 reference", f"{kind} thresholds reproduce the "
                   "registered v4 values", bool(np.all(np.abs(got - want)
                                                       <= V4_THRESHOLD_TOL)),
                   f"got {got[0]:+.3f} / {got[1]:+.3f}, registered "
                   f"{want[0]:+.3f} / {want[1]:+.3f}")
        ok_med = True
        if abs(INFORMATIVE_SD - v4.CAP_INFORMATIVE_SD) < 1e-12:
            ref = v4.informative_medians(X4)
            ok_med = (np.isclose(ref["colour"], row4["inf_median_sat"])
                      and np.isclose(ref["fog"], row4["inf_median_con"]))
            ck.add("v4 reference", "informative medians equal v4's "
                   "informative_medians", bool(ok_med),
                   f"saturation {row4['inf_median_sat']:.4f}, contrast "
                   f"{row4['inf_median_con']:.4f}")
        got = (row4["p_choice_sat"], row4["p_choice_con"])
        ck.add("v4 reference", "the planning weights reproduce v4's choice "
               "probabilities (0.6191 / 0.6104)",
               bool(abs(got[0] - V4_CHOICE_PROB[0]) <= 5e-4
                    and abs(got[1] - V4_CHOICE_PROB[1]) <= 5e-4),
               f"got {got[0]:.4f} / {got[1]:.4f}")
        p4 = []
        analyse_pair_of_lists(v4, X4, X4, {"size": 180, "mix": "v4 x2",
                                           "draw": 0}, st, p4)
        v4_rows = {"list": row4, "paired": pd.DataFrame(p4)}
        print(f"  v4 reference done ({time.time() - t0:.0f} s)")
    elif INCLUDE_V4_REFERENCE:
        ck.add("v4 reference", "thresholds reproduce the registered v4 "
               "values", "SKIPPED", f"{V4_TRIAL_LIST} not found")

    # --- the combinations ---
    list_rows, paired_rows, matching = [], [], []
    draw0 = []
    for size in BLOCK_SIZES:
        for mix in MIXES:
            t1 = time.time()
            counts = cap.largest_remainder(size, cap.TYPE_MIXES[mix])
            s1 = cap.stage1(pools_t, counts, scenes,
                            cap.MAX_PER_SCENE_PER_TYPE)
            scope = f"n={size} {mix}"
            if s1["status"] != "feasible":
                ck.add(scope, "counts feasible (script 1 stage 1)", False,
                       s1["status"])
                continue
            combo = f"{LEVEL_SET}|{size}|{mix}"   # script 1's draw seeds
            n_ok = 0
            for d in range(N_DRAWS):
                sets, how, n_allowed = cap.one_draw(pools_t, counts, s1,
                                                    scenes, combo, d)
                if sets is None:
                    matching.append((size, mix, d, False))
                    continue
                met, chk = cap.evaluate(sets, counts, s1, n_allowed)
                ok = all(v for v in chk.values() if v is not None)
                matching.append((size, mix, d, ok))
                n_ok += ok
                if d == 0:
                    r0 = check_draw0(cap, sets, size, mix)
                    if r0 is not None:
                        draw0.append(r0)
                st = cap.Streams(RANDOM_SEED, "block_length", combo, d)
                Xs, ths = {}, {}
                for blk in ("A", "B"):
                    m = sets[blk][sets[blk].role == "analysed"]
                    Xs[blk] = m[["x_sat", "x_con"]].to_numpy(float)
                    ths[blk], row = analyse_list(
                        v4, Xs[blk], st(f"null|{blk}"), st(f"det|{blk}"),
                        {"size": size, "mix": mix, "draw": d, "set": blk,
                         "built": how})
                    list_rows.append(row)
                rel = max(abs(ths["A"]["upper"][k] - ths["B"]["upper"][k])
                          / np.mean([ths["A"]["upper"][k],
                                     ths["B"]["upper"][k]])
                          for k in range(2))
                list_rows[-1]["set_threshold_rel_diff"] = rel
                list_rows[-2]["set_threshold_rel_diff"] = rel
                analyse_pair_of_lists(v4, Xs["A"], Xs["B"],
                                      {"size": size, "mix": mix, "draw": d},
                                      st, paired_rows)
            ck.add(scope, "every draw is built and passes script 1's "
                   "matching checks", n_ok == N_DRAWS,
                   f"{n_ok} of {N_DRAWS}")
            print(f"  n={size:<4d} {mix:15s} {N_DRAWS} draws "
                  f"({time.time() - t1:.0f} s)")

    lists = pd.DataFrame(list_rows)
    paired = pd.DataFrame(paired_rows)

    # --- checks on the results ---
    if draw0:
        ck.add("reproduction", "draw 0 equals script 1's saved example set "
               "(pair for pair, both sets)", all(draw0),
               f"{sum(draw0)} of {len(draw0)} combinations")
    else:
        ck.add("reproduction", "draw 0 equals script 1's saved example set",
               "SKIPPED", "SCRIPT1_RUN_DIR not found")
    # false-positive cells: mean over draws and the between-draw SE
    nul = paired[paired.scenario == NULL_SCENARIO]
    tl = pd.concat([nul.assign(w="sat", rate=nul.power_sat),
                    nul.assign(w="con", rate=nul.power_con)])
    type1 = (tl.groupby(["size", "mix", "N", "test", "w"]).rate
             .agg(["mean", "std", "count"]).reset_index())
    se_floor = np.sqrt(ALPHA * (1 - ALPHA) / (N_STUDIES * N_DRAWS))
    type1["se"] = np.maximum(type1["std"] / np.sqrt(type1["count"]),
                             se_floor)
    type1["z"] = (type1["mean"] - ALPHA) / type1["se"]
    z_crit = float(stats.norm.isf(TYPE1_FWER / max(len(type1), 1)))
    for test in PAIRED_TESTS:
        r = tl[tl.test == test].rate.mean()
        log.append(f"false-positive rate, {test}-test, all cells pooled: "
                   f"{r:.4f} (alpha {ALPHA})")
    for (size, mix), g in lists.groupby(["size", "mix"], sort=False):
        scope = f"n={size} {mix}"
        ck.add(scope, f"the two sets' upper thresholds agree within "
               f"{SET_THRESHOLD_TOL:.0%}",
               bool((g.set_threshold_rel_diff <= SET_THRESHOLD_TOL).all()),
               f"worst {g.set_threshold_rel_diff.max():.1%}")
        ck.add(scope, f"|corr(estimates)| <= {EST_CORR_MAX} under the null "
               "(weights separable)",
               bool((g.null_est_corr.abs() <= EST_CORR_MAX).all()),
               f"worst {g.null_est_corr.abs().max():.3f}")
        q = paired[(paired["size"] == size) & (paired.mix == mix)]
        ext = max(g.extreme_share.max(), q.extreme_share.max())
        ck.add(scope, f"share of |weight| > {EXTREME_WEIGHT:g} <= "
               f"{EXTREME_SHARE_MAX}", ext <= EXTREME_SHARE_MAX,
               f"worst {ext:.5f}")
        unc = max(g.unconverged_share.max(), q.unconverged_share.max())
        ck.add(scope, "every Firth fit converged", unc == 0,
               f"worst share unconverged {unc:.5f}")
        cells = type1[(type1["size"] == size) & (type1.mix == mix)]
        worst = cells.loc[cells.z.idxmax()]
        ck.add(scope, f"false-positive rate under '{NULL_SCENARIO}' <= "
               f"alpha + {z_crit:.2f} between-draw SE in every cell",
               bool((cells.z <= z_crit).all()),
               f"worst {worst['mean']:.4f} (z {worst.z:+.2f}, "
               f"{worst.test}, N={int(worst.N)}, {worst.w})")
    if v4_rows is not None:
        q = v4_rows["paired"]
        ext = max(v4_rows["list"]["extreme_share"], q.extreme_share.max())
        ck.add("v4 reference", f"share of |weight| > {EXTREME_WEIGHT:g} <= "
               f"{EXTREME_SHARE_MAX}", ext <= EXTREME_SHARE_MAX,
               f"worst {ext:.5f}")
    for (size, mix), g in lists.groupby(["size", "mix"], sort=False):
        dev = [abs(g.p_choice_sat.mean() - V4_CHOICE_PROB[0]),
               abs(g.p_choice_con.mean() - V4_CHOICE_PROB[1])]
        ck.add(f"n={size} {mix}", "planning weights give v4's choice "
               f"probabilities within {CHOICE_PROB_TOL} (typical informative "
               "trial)", max(dev) <= CHOICE_PROB_TOL + 1e-9,
               f"saturation {g.p_choice_sat.mean():.4f}, contrast "
               f"{g.p_choice_con.mean():.4f} (v4 {V4_CHOICE_PROB[0]:.4f}, "
               f"{V4_CHOICE_PROB[1]:.4f})")

    # --- outputs ---
    summ = summarise(lists, paired)
    sess = session_rows()
    lists.to_csv(os.path.join(out_dir, "lists.csv"), index=False)
    allp = paired if v4_rows is None else pd.concat(
        [paired, v4_rows["paired"]], ignore_index=True)
    allp.to_csv(os.path.join(out_dir, "paired_power.csv"), index=False)
    summ.to_csv(os.path.join(out_dir, "summary_by_combination.csv"),
                index=False)
    sess.to_csv(os.path.join(out_dir, "session_length.csv"), index=False)
    if v4_rows is not None:
        pd.DataFrame([v4_rows["list"]]).to_csv(
            os.path.join(out_dir, "v4_reference_list.csv"), index=False)
    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump(settings_dict(cap), f, indent=2, default=str)
    fig = make_figure(paired, None if v4_rows is None else
                      v4_rows["paired"], out_dir) if MAKE_FIGURE else None
    if MAKE_FIGURE and fig is None:
        log.append("figure     skipped (matplotlib not available)")
    n_fail = write_report(out_dir, ck, log, lists, paired, summ, v4_rows,
                          sess, matching, t0, cap)
    ck.frame().to_csv(os.path.join(out_dir, "checks.csv"), index=False)
    print(f"  report: {os.path.join(out_dir, 'block_length_report.txt')}")
    print(f"  {n_fail} check(s) FAIL; elapsed {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
