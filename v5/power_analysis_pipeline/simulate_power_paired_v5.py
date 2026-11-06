"""
simulate_power_paired_v5.py  --  v5 power analysis, script 7: the primary
                                 paired test (the PI's point 4)

WHAT THIS ANSWERS
    Each report-group participant does a derealization-episode block and an
    ordinary-episode block. Per block, the registered per-person model
    (script 6's choice: Firth logistic regression without intercept, with a
    saturation weight, a contrast weight and a side term) gives the weights.
    The PRIMARY test, per feature, is a one-sided test of the within-person
    difference (derealization-block weight minus ordinary-block weight)
    across the report group: H1 contrast, H2 saturation, each at alpha 0.05,
    one-sided, uncorrected. Candidates: the paired t-test and the exact
    Wilcoxon signed-rank test. This script:

      0. SANITY CHECK. Under explore_block_length_v5.py's simplified person
         model, its own functions (imported, unmodified) and this script's
         simulation give the same power on the frozen lists, within Monte
         Carlo error; and the frozen lists' power lies within the range
         explore found over its 20 draws of each size.
      1. FALSE POSITIVES with no effect: with no nuisance, with each
         nuisance alone (default value, then the ends of its range), with
         all at their defaults and all at their high ends.
      2. THREATS NO TEST CAN REMOVE, as their own rows: more lapses in the
         derealization block (+0.05, +0.10, +0.15), alone and with the
         "edited looks odd" and "memory makes it duller" biases; and a bias
         that differs between the episodes.
      3. POWER: planning, a quarter responding, all responding, each with
         its half-shift version beside it; with all nuisances at their
         defaults (the headline) and with none.
      4. THE CHOICE OF PRIMARY TEST by the rule fixed in SETTINGS.
      5. A MIXED-EFFECTS LOGISTIC MODEL over all trials and people as an
         alternative (the user's addition): does it gain enough power,
         especially under half shift, to be worth registering?
      6. THE SECONDARY sign test on detected counts within each block.
      7. PER-BLOCK ATTENTION MEASURES: catch failures and consistency
         (agreement between a trial and its repeat), per block, and how
         often they pick up a lapse difference between the episodes.
    N = 10-20 report-group participants (analysable), both block sizes.

PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES
    No image-choice data exist. The share of responders, their shift
    (0.5 saturation, 0.7 contrast per corpus SD between the two blocks;
    0.25 / 0.35 under half shift), taste spread, lapse rates, and every
    nuisance value below are assumptions for planning. The report labels
    every figure that depends on them.

THE PERSON MODEL (the truth; the fit is always the registered model)
    For person i, block b (D = derealization, O = ordinary) and choice
    trial t showing scene s(t), with x_sat, x_con the trial's predictors
    and side_t = +1 if image A is on the left, -1 if on the right:

      logit P(choose A) = sum_f (w_ibf + u_i,s(t),b,f + v_s(t),f) x_tf
                          + side_ib * side_t + g_i,pair(t)
      w_iO = taste_iO + shared_i + dull - odd_i
      w_iD = taste_iD + shared_i + dull - odd_i + responder_i * shift
             + differential bias
      P    = (1 - lapse_ibt) logistic(.) + lapse_ibt / 2

    taste_iD, taste_iO   bivariate normal, SD TASTE_SD, correlation
                         taste_corr (1 = the same in both blocks)
    u_isb                person x scene deviation in the two weights,
                         SD scene_sd_mult x TASTE_SD, correlation
                         scene_corr between blocks (the same 47 scenes
                         appear in both blocks), so trials of the same
                         scene are not independent
    v_s                  a scene effect shared by everyone (common_scene_sd)
    side_ib              Normal(side_mean, side_sd), correlation side_corr
                         between blocks; + = prefers the left image
    dull                 everyone, both blocks, towards the predicted image
                         ("memory makes it duller")
    odd_i                a share odd_share of people, odd_weight on both
                         features against the predicted (more edited) image
                         in both blocks ("edited looks odd")
    block order          the second block: lapse + order_lapse, and every
                         feature preference x order_scale
    lapse_ib             personal lapse rate, Gamma with mean LAPSE_MEAN
                         and SD LAPSE_BETWEEN_SD x mean (v4's model), the
                         same in both episodes by default; + episode_lapse
                         in the derealization block (threat rows only);
                         lapse_drift makes it rise across a block's trials
                         (centred, so the block mean is unchanged)
    Catch trials: 5 per block, each failed with probability
    CATCH_P_FAIL_PER_LAPSE x the block's lapse rate (v4's model); a person
    with 3 or more failures over the session is excluded and recruitment
    continues. Consistency repeats (12 or 15 per block, not analysed) are
    simulated with the same model: agreement = the same image chosen on the
    original and its repeat (the repeat shows it on the other side).
    pair_sd adds a person x pair preference for image A (logit), the same
    on the original and its repeat; without it (the default) people agree
    with themselves on repeats at about chance, because their only
    preferences are the small feature tastes.
    Counterbalancing: participant k takes the next slot of the frozen
    assignment table (block order, set mapping, trial order); excluded
    people use up their slots, as in the real study.

REUSED (imported, not copied; SHA-256 of each file recorded and checked)
    simulate_power_v4_n10_20.py   fit_firth_batch (two predictors),
                                  sign_test_critical (Table 4's rule),
                                  FIRTH_ITER / FIRTH_TOL / FIRTH_MAX_STEP
    thresholds_v5.py (script 6)   fit_firth_batch_k (the side model)
    explore_block_length_v5.py    paired_pool, study_index, paired_tests
                                  (the sanity check, run unmodified),
                                  wilcoxon_greater (the exact one-sided
                                  Wilcoxon, vectorised; checked against
                                  SciPy here)
    analyse_pair_capacity_v5.py   Streams (named random generators)
    From power_compare.py (adapted, not imported): the comparison of
    tests at N = 10/15/20 and the shared-bias scenario. Its approximated
    design, 5,000-chooser thresholds, own Firth fit and one-sample tests
    are replaced by the frozen lists, script 6's thresholds, the
    registered estimators and the paired difference.
    New here: the person model above, slot-based recruitment with
    exclusion, the mixed model, the per-block measures, the rule, the
    report and the figures.

THE MIXED MODEL
    One logistic model per study over all analysed trials of both blocks
    and all people, no intercept: fixed effects saturation, contrast,
    saturation x block, contrast x block (block coded +1/2 derealization,
    -1/2 ordinary) and the side term; random person slopes on all of them,
    uncorrelated. Fitted by the Laplace approximation (each person's
    random effects at their posterior mode, Newton steps) with EM updates
    of the fixed effects and variances. One-sided Wald z on the two
    block-by-feature effects, SE from the generalised-least-squares form
    [sum_i (Sigma + V_i)^-1]^-1, V_i the person's own sampling covariance;
    the t(N-1) reference is reported beside it. No R is used; the fit is
    checked by recovering known parameters.

INPUTS (relative to PROJECT_ROOT)
    data/frozen_trial_lists_v5/v5_150/, v5_180/    (checksums verified)
    THRESHOLDS_RUN_DIR (script 6's output: thresholds.csv, checks.csv,
        run_meta.json)
    EXPLORE_RESULTS_CSV (explore_block_length_v5's paired_power.csv)
    the four imported scripts above

OUTPUTS (one new folder per run under OUTPUT_ROOT, named after the key
settings plus a hash of all settings; never overwritten)
    report.txt                 what was done, what was found, every check
    checks.csv                 every check, PASS or FAIL
    scenarios.csv              every scenario and its settings
    paired_power.csv           per scenario, size, N, test, feature: rate
    secondary_sign.csv         the sign test within each block
    block_measures.csv         catch failures and consistency per block,
                               and how often each flags a lapse difference
    mixed_model.csv            the mixed model beside the paired tests
    sanity_explore.csv         the sanity check
    pool_replicates.csv        headline cells on two independent pools
    power_vs_n.png, false_positives.png, block_measures.png
    settings.json, run_meta.json

REQUIRES  Python 3.9+, numpy, pandas, scipy; matplotlib for the figures
RUN       python simulate_power_paired_v5.py
          (about 20-30 minutes without the mixed model; the mixed model
          adds about 1-2 hours: RUN_MIXED_MODEL)
"""

# ============================================================
# SETTINGS
# ============================================================

# --- Where things are ---
PROJECT_ROOT = None          # None = the folder this script is in
PACKAGES = {150: "data/frozen_trial_lists_v5/v5_150",
            180: "data/frozen_trial_lists_v5/v5_180"}
BLOCK_SIZES = [150, 180]
SETS = ("A", "B")
ORDERS = (1, 2, 3, 4)
GROUP = "participant"        # the report group's rows of assignment.csv

V4_SIM_SCRIPT = "simulate_power_v4_n10_20.py"
CAPACITY_SCRIPT = "analyse_pair_capacity_v5.py"
EXPLORE_SCRIPT = "explore_block_length_v5.py"
THRESHOLDS_SCRIPT = "thresholds_v5.py"
EXPECTED_SHA256 = {
    V4_SIM_SCRIPT:
        "852f3b33eb349153fbff9f95c50e1cd3c4f84f68b204d79a23f020f471bd1730",
    CAPACITY_SCRIPT:
        "5a8483a2a388482295c998ffcbe91e4a0f2df70f668d472cc4718d3a138addc7",
    EXPLORE_SCRIPT:
        "98d641f60252be83dcece7a480cc2dd184a495c1fa8751458f052a823389bfb6",
    THRESHOLDS_SCRIPT:
        "90457ef9c8208581f54a4e3db90cabfb1efd058bcc8f8a4ba77d79e23daa589b",
}

# THRESHOLDS_RUN_DIR / PERSON_MODEL
#   Script 6's output folder, and the per-person model registered from it
#   ("side": two weights + side term; "two-weight": the original model).
#   The thresholds are read from its thresholds.csv: for "side" the rows
#   with source "same choosers as design.json", for "two-weight" the rows
#   "design.json (registered)".
THRESHOLDS_RUN_DIR = ("data/power_v5/thresholds_v5/sizes150-180_choosers20000"
                      "_people10000_side0.25-1.5-popsd0.25_w0.5-0.7_lapse0.1"
                      "_seed42__s02be6391")
PERSON_MODEL = "side"
THRESHOLD_SOURCE = {"side": "same choosers as design.json",
                    "two-weight": "design.json (registered)"}

# EXPLORE_RESULTS_CSV / EXPLORE_MIX
#   explore_block_length_v5's paired_power.csv (the run with sizes
#   120-150-180) and the mix whose draws match the frozen lists.
EXPLORE_RESULTS_CSV = (
    "data/block_length_v5/without_p65_sizes120-150-180_mixes-conflict_heavy"
    "-v4_draws20_w0.5-0.7_lapse0.1_null20000_seed42__s6821ca69/"
    "paired_power.csv")
EXPLORE_MIX = "conflict_heavy"

OUTPUT_ROOT = "data/power_v5/paired_v5"
MAKE_FIGURES = True
FIG_DPI = 150

# --- People (PLANNING ASSUMPTIONS, not estimates) ---
#
# TASTE_SD
#   Between-person SD of personal taste on (saturation, contrast): v4's
#   non-responder spread, 0.5 x the v4 thresholds (0.286, 0.488).
TASTE_SD = (0.143, 0.244)

# LAPSE_MEAN / LAPSE_BETWEEN_SD / LAPSE_MAX
#   Personal lapse rates: Gamma with this mean and SD LAPSE_BETWEEN_SD x
#   mean, capped at LAPSE_MAX (v4's model, simulate_power_v4_n10_20).
LAPSE_MEAN = 0.1
LAPSE_BETWEEN_SD = 0.5
LAPSE_MAX = 0.9

# Catch trials and exclusion (registered rule; v4's catch model)
CATCH_PER_BLOCK = 5
CATCH_P_FAIL_PER_LAPSE = 0.75
CATCH_FAIL_EXCLUDE = 3          # 3 or more of 10 over the session
EXCLUSION = True

# EFFECTS
#   share: share of responders; shift: their derealization-only shift
#   (saturation, contrast) per corpus SD, with between-responder SD
#   shift_sd; shared: a responder preference present in BOTH blocks.
#   "half shift": half of each responder's preference is shared by both
#   episodes, so only 0.25 / 0.35 separates the blocks.
EFFECTS = {
    "no effect": {"share": 0.0, "shift": (0.0, 0.0),
                  "shift_sd": (0.0, 0.0), "shared": (0.0, 0.0)},
    "planning": {"share": 0.5, "shift": (0.5, 0.7),
                 "shift_sd": (0.0, 0.0), "shared": (0.0, 0.0)},
    "half shift": {"share": 0.5, "shift": (0.25, 0.35),
                   "shift_sd": (0.0, 0.0), "shared": (0.25, 0.35)},
    "a quarter respond": {"share": 0.25, "shift": (0.5, 0.7),
                          "shift_sd": (0.0, 0.0), "shared": (0.0, 0.0)},
    "a quarter respond, half shift": {"share": 0.25, "shift": (0.25, 0.35),
                                      "shift_sd": (0.0, 0.0),
                                      "shared": (0.25, 0.35)},
    "all respond": {"share": 1.0, "shift": (0.5, 0.7),
                    "shift_sd": (0.0, 0.0), "shared": (0.0, 0.0)},
    "all respond, half shift": {"share": 1.0, "shift": (0.25, 0.35),
                                "shift_sd": (0.0, 0.0),
                                "shared": (0.25, 0.35)},
}

# NUISANCES
#   NUISANCE_OFF: none (taste identical in both blocks, no scene effects,
#   no side bias, no shared bias, no order effect, equal lapses; personal
#   lapse rates still differ between people).
#   NUISANCE_DEFAULT: the plausible values agreed for planning.
#   The ranges swept are in SENSITIVITY_ROWS below.
NUISANCE_OFF = {
    "taste_corr": 1.0,            # taste correlation between blocks
    "scene_sd_mult": 0.0,         # person x scene SD, x TASTE_SD
    "scene_corr": 0.7,            # its correlation between blocks
    "common_scene_sd": 0.0,       # scene effect shared by everyone
    "side_mean": 0.0,             # side bias, logit (+ = left)
    "side_sd": 0.0,
    "side_corr": 0.8,             # its correlation between blocks
    "dull": (0.0, 0.0),           # "memory makes it duller", everyone
    "odd_share": 0.0,             # "edited looks odd": share of people ...
    "odd_weight": 0.5,            # ... and their weight against, per SD
    "order_lapse": 0.0,           # second block: + lapse
    "order_scale": 1.0,           # second block: x preferences
    "episode_lapse": 0.0,         # derealization block: + lapse (threat)
    "differential_bias": (0.0, 0.0),   # derealization block only (threat)
    "lapse_drift": 0.0,           # lapse rise across a block, centred
    "pair_sd": 0.0,               # person x pair preference for image A,
                                  # logit SD; the same on a pair's repeat
}
NUISANCE_DEFAULT = {**NUISANCE_OFF,
                    "taste_corr": 0.7, "scene_sd_mult": 1.0,
                    "scene_corr": 0.7, "side_sd": 0.25,
                    "dull": (0.25, 0.35), "odd_share": 0.25,
                    "order_lapse": 0.05, "order_scale": 0.9}
NUISANCE_HIGH = {**NUISANCE_OFF,
                 "taste_corr": 0.0, "scene_sd_mult": 2.0, "scene_corr": 0.0,
                 "side_mean": 0.2, "side_sd": 0.5,
                 "dull": (0.5, 0.7), "odd_share": 0.5,
                 "order_lapse": 0.10, "order_scale": 0.8}

# The single-nuisance rows used by the test-choice rule (each alone, on
# NUISANCE_OFF, at its default value).
DEFAULT_ROWS = [
    ("taste stability 0.7", {"taste_corr": 0.7}),
    ("scene dependence (SD x1, stability 0.7)",
     {"scene_sd_mult": 1.0, "scene_corr": 0.7}),
    ("side bias (SD 0.25)", {"side_sd": 0.25}),
    ("memory makes it duller (+0.25/+0.35, everyone)",
     {"dull": (0.25, 0.35)}),
    ("edited looks odd (a quarter, -0.5)", {"odd_share": 0.25}),
    ("block order (second block: lapse +0.05, preferences x0.9)",
     {"order_lapse": 0.05, "order_scale": 0.9}),
]
# Sensitivity rows (reported; not used by the rule), each alone on
# NUISANCE_OFF.
SENSITIVITY_ROWS = [
    ("taste stability 0.5", {"taste_corr": 0.5}),
    ("taste stability 0.3", {"taste_corr": 0.3}),
    ("taste stability 0", {"taste_corr": 0.0}),
    ("scene dependence SD x0.5", {"scene_sd_mult": 0.5}),
    ("scene dependence SD x2", {"scene_sd_mult": 2.0}),
    ("scene dependence, stability 0", {"scene_sd_mult": 1.0,
                                       "scene_corr": 0.0}),
    ("scene dependence, stability 1", {"scene_sd_mult": 1.0,
                                       "scene_corr": 1.0}),
    ("common scene effect SD 0.1", {"common_scene_sd": 0.1}),
    ("side bias SD 0.5", {"side_sd": 0.5}),
    ("side bias SD 0.25, mean 0.2", {"side_sd": 0.25, "side_mean": 0.2}),
    ("memory makes it duller +0.5/+0.7", {"dull": (0.5, 0.7)}),
    ("edited looks odd, half", {"odd_share": 0.5}),
    ("block order: second-block lapse +0.10", {"order_lapse": 0.10}),
    ("block order: preferences x0.8", {"order_scale": 0.8}),
    ("block order: preferences x1.2", {"order_scale": 1.2}),
    ("lapse rising across the block (+0.10)", {"lapse_drift": 0.10}),
    ("pair-specific preferences SD 1.0", {"pair_sd": 1.0}),
    ("pair-specific preferences SD 2.0", {"pair_sd": 2.0}),
]
# PAIR_SD_FOR_ROWS
#   A person's preference for one image of a specific pair, beyond the two
#   features (logit SD), the same when the pair is repeated: what makes
#   people agree with themselves on repeats. Not part of the agreed
#   defaults (0); the rows using it show what it does to the tests, to
#   power and to the consistency measure. A value for the PI to calibrate.
PAIR_SD_FOR_ROWS = 1.5
# Threats no test can remove (reported as their own rows; not in the rule).
EPISODE_LAPSES = [0.05, 0.10, 0.15]
THREAT_INTERACTION_LAPSE = 0.10
DIFFERENTIAL_BIAS = (0.10, 0.14)   # dulling 40% stronger in derealization

# --- Studies ---
N_GRID = list(range(10, 21))      # analysable report-group participants
TABLE_N = [10, 15, 20]            # the N shown in report tables
# POOL_PEOPLE / N_STUDIES / N_BATCHES
#   People per pool (one pool per scenario and size, each person fitted on
#   both blocks) and studies per cell, drawn from the pool with
#   replacement. A finite pool adds its own Monte Carlo error: about
#   sqrt(N x phi^2 / POOL_PEOPLE) on a power near 0.5 at N = 20, which is
#   0.013 for 20,000 people and 0.008 for 50,000. With 10,000 studies
#   (0.005) the total is then <= 0.01. Every rate's SE is MEASURED: the pool
#   is split into N_BATCHES disjoint sub-pools, each study draws from one,
#   and the SE is the spread of the sub-pool rates / sqrt(N_BATCHES); it
#   includes the pool's error.
POOL_PEOPLE = 50_000
N_STUDIES = 10_000
N_BATCHES = 20
TESTS = ("t", "wilcoxon")
ALPHA = 0.05

# --- The rule for choosing the primary test (fixed before the run) ---
#   A test qualifies if, in every rule row (no nuisance, each default
#   nuisance alone, all defaults together; no effect), at every N in RULE_N,
#   both sizes and both features, its false-positive rate is at most
#   ALPHA + RULE_SE_MULT x SE, SE being that cell's measured Monte Carlo SE
#   (pool and studies; never below the binomial SE of ALPHA over
#   N_STUDIES).
#   Among qualifying tests, the one with the higher mean power over
#   planning and half shift (all nuisances at defaults), RULE_N, both sizes
#   and both features wins; within RULE_TIE of each other, RULE_PREFER.
RULE_N = [10, 15, 20]
RULE_SE_MULT = 2.0
RULE_TIE = 0.02
RULE_PREFER = "wilcoxon"

# --- Sanity check against explore_block_length_v5 ---
#   Its scenarios run with its own functions on the frozen lists, and with
#   this script's simulation set to its simplified model; agreement if no
#   |z| exceeds SANITY_MAX_Z and at most SANITY_SHARE_BEYOND of |z| exceed
#   1.96. The frozen lists' power must lie within explore's range over its
#   draws, widened by SANITY_RANGE_SE standard errors.
SANITY_SCENARIOS = ["no effect", "planning", "half shift"]
SANITY_N = [10, 15, 20]
SANITY_MAX_Z = 3.5
SANITY_SHARE_BEYOND = 0.10
SANITY_RANGE_SE = 3.0

# --- Pool replicates (Monte Carlo check) ---
#   These scenarios (all nuisances at defaults) are rerun on an independent
#   pool; same criteria as the sanity check.
REPLICATE_EFFECTS = ["no effect", "planning", "half shift"]

# --- The mixed model ---
RUN_MIXED_MODEL = True
MIXED_STUDIES = 2_500
MIXED_N = [10, 15, 20]
MIXED_CELLS = [("no effect", "no nuisances"),
               ("no effect", "all nuisances (defaults)"),
               ("planning", "all nuisances (defaults)"),
               ("half shift", "all nuisances (defaults)")]
MIXED_MAX_ITER = 300
MIXED_TOL = 1e-5               # largest change of a fixed effect per step
MIXED_CHUNK = 250              # studies fitted at once (memory)
MIXED_GAIN = 0.05              # "worth registering": power gain under
MIXED_GAIN_N = 15              # half shift at this N, both sizes, both
                               # features, with false positives controlled

# --- Per-block attention measures ---
#   Per study: consistency lower in the derealization block (one-sided
#   paired t on each person's agreement difference) and more catch
#   failures there (one-sided exact sign test, ties dropped), at ALPHA.

# --- Seeds ---
RANDOM_SEED = 42
STREAM_KEY = "paired_v5"

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
import time

import numpy as np
import pandas as pd
from scipy import stats

FEATURES = ("sat", "con")
FEATURE_LABEL = {"sat": "saturation", "con": "contrast"}
ANALYSED_TYPES = ("sat_only", "fog_only", "together", "conflict")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def root_path(*parts):
    root = PROJECT_ROOT or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, *parts)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, section, what, ok, detail=""):
        status = "PASS" if bool(ok) else "FAIL"
        self.rows.append({"section": section, "check": what,
                          "status": status, "detail": str(detail)})
        log(f"{status}  [{section}] {what}  {detail}")
        return bool(ok)

    @property
    def n_fail(self):
        return sum(r["status"] == "FAIL" for r in self.rows)


SETTING_NAMES = [
    "PROJECT_ROOT", "PACKAGES", "BLOCK_SIZES", "SETS", "ORDERS", "GROUP",
    "V4_SIM_SCRIPT", "CAPACITY_SCRIPT", "EXPLORE_SCRIPT", "THRESHOLDS_SCRIPT",
    "EXPECTED_SHA256", "THRESHOLDS_RUN_DIR", "PERSON_MODEL",
    "THRESHOLD_SOURCE", "EXPLORE_RESULTS_CSV", "EXPLORE_MIX", "OUTPUT_ROOT",
    "MAKE_FIGURES", "FIG_DPI", "TASTE_SD", "LAPSE_MEAN", "LAPSE_BETWEEN_SD",
    "LAPSE_MAX", "CATCH_PER_BLOCK", "CATCH_P_FAIL_PER_LAPSE",
    "CATCH_FAIL_EXCLUDE", "EXCLUSION", "EFFECTS", "NUISANCE_OFF",
    "NUISANCE_DEFAULT", "NUISANCE_HIGH", "DEFAULT_ROWS", "SENSITIVITY_ROWS",
    "EPISODE_LAPSES", "THREAT_INTERACTION_LAPSE", "DIFFERENTIAL_BIAS",
    "PAIR_SD_FOR_ROWS",
    "N_GRID", "TABLE_N", "POOL_PEOPLE", "N_STUDIES", "N_BATCHES", "TESTS",
    "ALPHA",
    "RULE_N", "RULE_SE_MULT", "RULE_TIE", "RULE_PREFER", "SANITY_SCENARIOS",
    "SANITY_N", "SANITY_MAX_Z", "SANITY_SHARE_BEYOND", "SANITY_RANGE_SE",
    "REPLICATE_EFFECTS", "RUN_MIXED_MODEL", "MIXED_STUDIES", "MIXED_N",
    "MIXED_CELLS", "MIXED_MAX_ITER", "MIXED_TOL", "MIXED_CHUNK",
    "MIXED_GAIN", "MIXED_GAIN_N", "RANDOM_SEED", "STREAM_KEY",
]


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def settings_dict():
    g = globals()
    return {k: g[k] for k in SETTING_NAMES}


def resolve_output_dir():
    h = hashlib.sha256(json.dumps(jsonable(settings_dict()), sort_keys=True)
                       .encode()).hexdigest()[:8]
    name = (f"sizes{'-'.join(map(str, BLOCK_SIZES))}"
            f"_N{N_GRID[0]}-{N_GRID[-1]}_model-{PERSON_MODEL}"
            f"_pool{POOL_PEOPLE}_studies{N_STUDIES}"
            f"_mixed{MIXED_STUDIES if RUN_MIXED_MODEL else 0}"
            f"_seed{RANDOM_SEED}__s{h}")
    base = root_path(OUTPUT_ROOT, name)
    out, k = base, 1
    while os.path.exists(out):          # never overwrite an earlier run
        k += 1
        out = f"{base}_run{k}"
    os.makedirs(out)
    return out


def load_module(fname, ck, hashes):
    path = root_path(fname)
    ok = os.path.exists(path)
    ck.add("inputs", f"{fname} exists", ok, path)
    if not ok:
        raise SystemExit(f"missing {path}")
    digest = sha256_of(path)
    hashes[fname] = digest
    ck.add("inputs", f"{fname} is the expected version (SHA-256)",
           digest == EXPECTED_SHA256.get(fname), digest[:16] + "...")
    spec = importlib.util.spec_from_file_location(
        os.path.splitext(fname)[0] + "_imported", path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------
# Inputs: packages, thresholds
# ------------------------------------------------------------

def read_checksums(pkg_dir):
    rows = {}
    with open(os.path.join(pkg_dir, "checksums.sha256")) as f:
        for line in f:
            if line.strip():
                d, rel = line.strip().split(None, 1)
                rows[rel.lstrip("*").strip()] = d
    return rows


def load_package(size, ck, hashes):
    """Lists of this size, as arrays ready for simulation."""
    pkg = root_path(PACKAGES[size])
    sums = read_checksums(pkg)
    needed = (["package/design.json", "package/assignment.csv"]
              + [f"package/lists/{b}_o{k}.json" for b in SETS
                 for k in ORDERS])
    bad = [r for r in needed if r not in sums
           or sha256_of(os.path.join(pkg, r)) != sums[r]]
    ck.add(f"package {size}", "every file read matches checksums.sha256",
           not bad, f"{len(needed)} files" + (f"; bad: {bad}" if bad else ""))
    for r in needed:
        hashes[f"{PACKAGES[size]}/{r}"] = sums.get(r)
    with open(os.path.join(pkg, "package", "design.json")) as f:
        design = json.load(f)
    counts = design["counts_per_block"]
    lists, scenes = {}, None
    item_order = {}
    for b in SETS:
        for k in ORDERS:
            lid = f"{b}_o{k}"
            with open(os.path.join(pkg, "package", "lists",
                                   f"{lid}.json")) as f:
                L = pd.DataFrame(json.load(f))
            if scenes is None:
                scenes = sorted(L.loc[L["trial_type"] != "catch",
                                      "scene_id"].dropna().unique())
            lists[lid] = L
    sidx = {s: i for i, s in enumerate(scenes)}
    out = {}
    for lid, L in lists.items():
        b = lid[0]
        choice = L["trial_type"] != "catch"
        an = (L["analysed"] == True).to_numpy()        # noqa: E712
        ch = choice.to_numpy()
        n_an = int(an.sum())
        cons = L[L["trial_type"] == "consistency"]
        by_trial = {int(t): i for i, t in enumerate(L["trial"])}
        rep = np.array([by_trial[int(t)] for t in cons["trial"]])
        orig = np.array([by_trial[int(t)] for t in cons["repeat_of"]])
        same_a = (L.loc[cons.index, "file_a"].to_numpy()
                  == L["file_a"].to_numpy()[orig])
        if b not in item_order:
            item_order[b] = sorted(L.loc[an, "item_id"])
        pos_item = {it: i for i, it in enumerate(item_order[b])}
        X = L[["pred_sat", "pred_fog"]].to_numpy(float)
        X = np.nan_to_num(X)
        side = np.where(L["a_is_left"].fillna(False).astype(bool), 1.0, -1.0)
        ci = np.full(len(L), -1)
        ci[an] = [pos_item[it] for it in L.loc[an, "item_id"]]
        ci[cons.index.to_numpy()] = [pos_item[it] for it in
                                     cons["repeat_of_item"]]
        out[lid] = {
            "set": b, "choice_item": ci,
            "X": X, "side": side,
            "scene": np.array([sidx.get(s, 0) for s in
                               L["scene_id"].fillna(scenes[0])]),
            "choice": ch, "analysed": an,
            "pos": np.arange(len(L)) / (len(L) - 1),
            "rep": rep, "orig": orig, "same_a": same_a,
            "item_cols": np.array([pos_item[it] for it in
                                   L.loc[an, "item_id"]]),
            "n_cons": len(cons), "n_catch": int((~choice).sum()),
        }
        ok = (n_an == counts["analysed"] and len(cons) == counts["consistency"]
              and out[lid]["n_catch"] == counts["catch"]
              and bool(np.all(same_a))
              and len(L) == design["main_trials_per_block"]
              and set(L.loc[an, "trial_type"]) <= set(ANALYSED_TYPES))
        ck.add(f"package {size}", f"{lid}: analysed, consistency and catch "
               "counts equal design.json; every repeat shows the same image A",
               ok, f"{n_an} analysed, {len(cons)} repeats, "
               f"{out[lid]['n_catch']} catch")
    # item-order design matrices per set (for the mixed model)
    for b in SETS:
        L0 = out[f"{b}_o1"]
        an_rows = np.flatnonzero(L0["analysed"])
        Xi = np.empty((len(an_rows), 2))
        si = np.empty(len(an_rows))
        Xi[L0["item_cols"]] = L0["X"][an_rows]
        si[L0["item_cols"]] = L0["side"][an_rows]
        out[f"items_{b}"] = {"X": Xi, "side": si}
    asg = pd.read_csv(os.path.join(pkg, "package", "assignment.csv"))
    asg = asg[asg["group"] == GROUP].sort_values("slot").reset_index(drop=True)
    ck.add(f"package {size}", f"assignment table: 64 slots for the {GROUP} "
           "group, each with a derealization and an ordinary block",
           len(asg) == 64 and set(asg["block1_episode"]) | set(
               asg["block2_episode"]) == {"derealization", "ordinary"},
           f"{len(asg)} rows")
    rows = []
    for r in asg.itertuples():
        d_first = r.block1_episode == "derealization"
        rows.append({"der_list": r.block1_list if d_first else r.block2_list,
                     "ord_list": r.block2_list if d_first else r.block1_list,
                     "der_pos": 1 if d_first else 2})
    return {"lists": out, "slots": rows, "n_scenes": len(scenes),
            "design": design}


def load_thresholds(ck, hashes):
    d = root_path(THRESHOLDS_RUN_DIR)
    p = os.path.join(d, "thresholds.csv")
    ok = os.path.exists(p)
    ck.add("inputs", "script 6's thresholds.csv exists", ok, p)
    if not ok:
        raise SystemExit(f"missing {p}")
    hashes["thresholds.csv"] = sha256_of(p)
    c = pd.read_csv(os.path.join(d, "checks.csv"))
    ck.add("inputs", "script 6's run passed every check",
           (c["status"] == "PASS").all(),
           f"{(c['status'] == 'PASS').sum()} PASS, "
           f"{(c['status'] != 'PASS').sum()} FAIL")
    with open(os.path.join(d, "run_meta.json")) as f:
        meta = json.load(f)
    ck.add("inputs", f"script 6 recommended the model used here "
           f"({PERSON_MODEL})", meta.get("chosen_model") == PERSON_MODEL,
           f"script 6: {meta.get('chosen_model')}")
    t = pd.read_csv(p)
    t = t[(t["model"] == PERSON_MODEL)
          & (t["source"] == THRESHOLD_SOURCE[PERSON_MODEL])]
    th = {}
    for size in BLOCK_SIZES:
        for b in SETS:
            q = t[(t["size"] == size) & (t["set"] == b)].set_index("feature")
            th[(size, b)] = {
                "upper": np.array([q.loc["saturation", "upper"],
                                   q.loc["contrast", "upper"]]),
                "lower": np.array([q.loc["saturation", "lower"],
                                   q.loc["contrast", "lower"]])}
    ck.add("inputs", "thresholds found for every size and set",
           len(th) == len(BLOCK_SIZES) * len(SETS), f"{len(th)} sets")
    return th


# ------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------

def build_scenarios():
    S = []

    def add(name, effect, nuis, role):
        S.append({"name": name, "effect": effect, "nuis": dict(nuis),
                  "role": role})

    add("no nuisances", "no effect", NUISANCE_OFF, "null (rule)")
    for name, o in DEFAULT_ROWS:
        add(name, "no effect", {**NUISANCE_OFF, **o}, "null (rule)")
    add("all nuisances (defaults)", "no effect", NUISANCE_DEFAULT,
        "null (rule)")
    for name, o in SENSITIVITY_ROWS:
        add(name, "no effect", {**NUISANCE_OFF, **o}, "null (sensitivity)")
    add("all nuisances (high ends)", "no effect", NUISANCE_HIGH,
        "null (sensitivity)")
    odd = dict(DEFAULT_ROWS)["edited looks odd (a quarter, -0.5)"]
    dull = dict(DEFAULT_ROWS)["memory makes it duller (+0.25/+0.35, "
                              "everyone)"]
    for el in EPISODE_LAPSES:
        add(f"derealization lapse +{el:g}", "no effect",
            {**NUISANCE_OFF, "episode_lapse": el}, "threat")
    for el in EPISODE_LAPSES:
        add(f"derealization lapse +{el:g}, with edited looks odd",
            "no effect", {**NUISANCE_OFF, **odd, "episode_lapse": el},
            "threat")
    for el in EPISODE_LAPSES:
        add(f"derealization lapse +{el:g}, with memory makes it duller",
            "no effect", {**NUISANCE_OFF, **dull, "episode_lapse": el},
            "threat")
    add(f"all nuisances (defaults), derealization lapse "
        f"+{THREAT_INTERACTION_LAPSE:g}", "no effect",
        {**NUISANCE_DEFAULT, "episode_lapse": THREAT_INTERACTION_LAPSE},
        "threat")
    add(f"all nuisances (defaults), dulling stronger in derealization "
        f"(+{DIFFERENTIAL_BIAS[0]:g}/+{DIFFERENTIAL_BIAS[1]:g})",
        "no effect", {**NUISANCE_DEFAULT,
                      "differential_bias": DIFFERENTIAL_BIAS}, "threat")
    ps = {"pair_sd": PAIR_SD_FOR_ROWS}
    for el in EPISODE_LAPSES:
        add(f"derealization lapse +{el:g}, with pair-specific preferences "
            f"SD {PAIR_SD_FOR_ROWS:g}", "no effect",
            {**NUISANCE_OFF, **ps, "episode_lapse": el}, "threat")
    add(f"all nuisances (defaults) + pair-specific preferences SD "
        f"{PAIR_SD_FOR_ROWS:g}", "no effect", {**NUISANCE_DEFAULT, **ps},
        "null (sensitivity)")
    for eff in EFFECTS:
        if eff == "no effect":
            continue
        add("all nuisances (defaults)", eff, NUISANCE_DEFAULT, "power")
        add("no nuisances", eff, NUISANCE_OFF, "power (no nuisances)")
    for eff in ("planning", "half shift"):
        add(f"all nuisances (defaults) + pair-specific preferences SD "
            f"{PAIR_SD_FOR_ROWS:g}", eff, {**NUISANCE_DEFAULT, **ps},
            "power (sensitivity)")
    return S


# ------------------------------------------------------------
# Estimation helpers
# ------------------------------------------------------------

class Fitter:
    def __init__(self, v4, s6):
        self.v4, self.s6 = v4, s6

    def __call__(self, X, Y):
        if X.shape[1] == 2:
            return self.v4.fit_firth_batch(X, Y)[0]
        return self.s6.fit_firth_batch_k(X, Y, self.v4.FIRTH_ITER,
                                         self.v4.FIRTH_TOL,
                                         self.v4.FIRTH_MAX_STEP)[0]


def model_X(X2, side, model):
    return X2 if model == "two-weight" else np.column_stack([X2, side])


# ------------------------------------------------------------
# The person model: one pool of people, each fitted on both blocks
# ------------------------------------------------------------

def simulate_pool(pk, eff, nuis, rng, fit, P=POOL_PEOPLE, model=PERSON_MODEL,
                  lapse_fixed=False, keep_y=False):
    """P people of one scenario; person j takes slot row j mod 64."""
    lists, slots = pk["lists"], pk["slots"]
    nsc = pk["n_scenes"]
    row = np.arange(P) % len(slots)
    der_list = np.array([slots[r]["der_list"] for r in row])
    ord_list = np.array([slots[r]["ord_list"] for r in row])
    der_pos = np.array([slots[r]["der_pos"] for r in row])
    tsd = np.asarray(TASTE_SD)

    resp = rng.random(P) < eff["share"]
    shift = (np.asarray(eff["shift"]) + rng.normal(0, 1, (P, 2))
             * np.asarray(eff["shift_sd"]))
    shift = np.where(resp[:, None], shift, 0.0)
    shared = np.where(resp[:, None], np.asarray(eff["shared"]), 0.0)

    rho = nuis["taste_corr"]
    z1, z2 = rng.normal(0, 1, (P, 2)), rng.normal(0, 1, (P, 2))
    taste_O = z1 * tsd
    taste_D = (rho * z1 + np.sqrt(max(1 - rho ** 2, 0.0)) * z2) * tsd
    common = (np.asarray(nuis["dull"])
              - np.where(rng.random(P) < nuis["odd_share"],
                         nuis["odd_weight"], 0.0)[:, None])
    w = {"O": taste_O + shared + common,
         "D": taste_D + shared + common + shift
         + np.asarray(nuis["differential_bias"])}

    ssd = nuis["scene_sd_mult"] * tsd
    if nuis["scene_sd_mult"] > 0:
        rs = nuis["scene_corr"]
        a = rng.normal(0, 1, (P, nsc, 2))
        c = rng.normal(0, 1, (P, nsc, 2))
        u = {"O": a * ssd,
             "D": (rs * a + np.sqrt(max(1 - rs ** 2, 0.0)) * c) * ssd}
    else:
        u = {"O": None, "D": None}
    v = (rng.normal(0, nuis["common_scene_sd"], (nsc, 2))
         if nuis["common_scene_sd"] > 0 else None)

    rsd = nuis["side_corr"]
    y1, y2 = rng.normal(0, 1, P), rng.normal(0, 1, P)
    side = {"O": nuis["side_mean"] + nuis["side_sd"] * y1,
            "D": nuis["side_mean"] + nuis["side_sd"]
            * (rsd * y1 + np.sqrt(max(1 - rsd ** 2, 0.0)) * y2)}

    if lapse_fixed or LAPSE_MEAN <= 0:
        lam = np.full(P, LAPSE_MEAN)
    else:
        shape = 1.0 / LAPSE_BETWEEN_SD ** 2
        lam = np.clip(rng.gamma(shape, LAPSE_MEAN / shape, P), 0, LAPSE_MAX)
    second = {"D": der_pos == 2, "O": der_pos == 1}
    lam_b = {}
    for b in ("D", "O"):
        lb = lam + nuis["order_lapse"] * second[b]
        if b == "D":
            lb = lb + nuis["episode_lapse"]
        lam_b[b] = np.clip(lb, 0.0, LAPSE_MAX)
        scale = np.where(second[b], nuis["order_scale"], 1.0)[:, None]
        w[b] = w[b] * scale
        if u[b] is not None:
            u[b] = u[b] * scale[:, :, None]

    n_items = len(pk["lists"][f"items_{SETS[0]}"]["X"])
    k = 2 if model == "two-weight" else 3
    out = {"row": row, "batch": (np.arange(P) // len(slots)) % N_BATCHES,
           "der_set": np.array([s[0] for s in der_list]),
           "ord_set": np.array([s[0] for s in ord_list]),
           "der_pos": der_pos, "responder": resp,
           "lapse_D": lam_b["D"], "lapse_O": lam_b["O"]}
    for b, lst in (("D", der_list), ("O", ord_list)):
        est = np.empty((P, k))
        agree = np.empty(P)
        Yitem = (np.empty((P, n_items), dtype=np.uint8) if keep_y else None)
        for lid in np.unique(lst):
            m = np.flatnonzero(lst == lid)
            L = lists[lid]
            ch = np.flatnonzero(L["choice"])
            X = L["X"][ch]
            sc = L["scene"][ch]
            eta = w[b][m] @ X.T + side[b][m][:, None] * L["side"][ch][None, :]
            if u[b] is not None:
                eta += np.einsum("ptf,tf->pt", u[b][m][:, sc, :], X)
            if v is not None:
                eta += (v[sc] * X).sum(1)[None, :]
            if nuis.get("pair_sd", 0) > 0:
                g = rng.normal(0, nuis["pair_sd"], (len(m), n_items))
                eta += g[:, L["choice_item"][ch]]
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
            lt = lam_b[b][m][:, None] + nuis["lapse_drift"] * (
                L["pos"][ch][None, :] - 0.5)
            lt = np.clip(lt, 0.0, LAPSE_MAX)
            p = (1 - lt) * p + lt / 2
            Y = (rng.random(p.shape) < p).astype(float)
            # column positions within the choice rows
            pos_in_ch = np.full(len(L["choice"]), -1)
            pos_in_ch[ch] = np.arange(len(ch))
            an = pos_in_ch[np.flatnonzero(L["analysed"])]
            Ya = Y[:, an]
            Xa = model_X(L["X"][L["analysed"]], L["side"][L["analysed"]],
                         model)
            est[m] = fit(Xa, Ya)
            yr, yo = Y[:, pos_in_ch[L["rep"]]], Y[:, pos_in_ch[L["orig"]]]
            agree[m] = np.mean(np.where(L["same_a"][None, :], yr == yo,
                                        yr != yo), axis=1)
            if keep_y:
                Yitem[np.ix_(m, L["item_cols"])] = Ya.astype(np.uint8)
        out[f"est_{b}"] = est
        out[f"agree_{b}"] = agree
        out[f"catch_{b}"] = rng.binomial(
            CATCH_PER_BLOCK, np.clip(CATCH_P_FAIL_PER_LAPSE * lam_b[b], 0, 1))
        if keep_y:
            out[f"Y_{b}"] = Yitem
    out["excluded"] = (out["catch_D"] + out["catch_O"]) >= CATCH_FAIL_EXCLUDE
    return out


def person_thresholds(pool, size, TH):
    for b, key in (("D", "der_set"), ("O", "ord_set")):
        up = np.array([TH[(size, s)]["upper"] for s in SETS])
        lo = np.array([TH[(size, s)]["lower"] for s in SETS])
        i = (pool[key] == SETS[1]).astype(int)
        pool[f"up_{b}"], pool[f"lo_{b}"] = up[i], lo[i]


# ------------------------------------------------------------
# Studies: recruitment by slot, with exclusion
# ------------------------------------------------------------

def draw_studies(pool, N, S, rng, exclusion=EXCLUSION):
    """S studies of N analysable people. Study s draws from sub-pool
    s mod N_BATCHES only. Candidate k takes slot k (row (k-1) mod 64) and is
    drawn from that sub-pool's people of that row; anyone excluded uses up
    the slot and recruitment continues. Returns the people (S x N), the
    number excluded per study and each study's sub-pool."""
    rows, bat = pool["row"], pool["batch"]
    nrow = rows.max() + 1
    K = bat.max() + 1
    members = [[np.flatnonzero((rows == r) & (bat == k)) for r in range(nrow)]
               for k in range(K)]
    cnt = np.array([[len(m) for m in mk] for mk in members])
    M = np.zeros((K, nrow, cnt.max()), dtype=int)
    for k in range(K):
        for r in range(nrow):
            M[k, r, :cnt[k, r]] = members[k][r]
    excl = pool["excluded"] if exclusion else np.zeros(len(rows), bool)
    sb = np.arange(S) % K
    width = int(np.ceil(1.5 * N)) + 10
    while True:
        slot_row = np.arange(width) % nrow
        pick = (rng.random((S, width)) * cnt[sb[:, None], slot_row[None, :]]
                ).astype(int)
        cand = M[sb[:, None], slot_row[None, :], pick]
        ok = ~excl[cand]
        if (ok.sum(1) >= N).all():
            break
        width *= 2
    pos = np.argsort(~ok, axis=1, kind="stable")[:, :N]
    idx = np.take_along_axis(cand, pos, axis=1)
    return idx, pos[:, -1] + 1 - N, sb


def rate_se(x, sb):
    """Mean of a per-study quantity and its Monte Carlo SE from the
    sub-pool means (includes the pool's error)."""
    x = np.asarray(x, float)
    K = sb.max() + 1
    m = np.array([x[sb == k].mean() for k in range(K)])
    p, n = float(x.mean()), len(x)
    # never below the binomial SE of the studies (with p(1-p) floored at
    # 5/n, so a rate of exactly 0 or 1 does not get an SE of 0)
    floor = np.sqrt(max(p * (1 - p), 5.0 / n) / n) if set(np.unique(x)) <= \
        {0.0, 1.0} else 0.0
    return p, float(max(m.std(ddof=1) / np.sqrt(K), floor))


def t_greater(d):
    n = d.shape[1]
    s = d.std(1, ddof=1)
    t = d.mean(1) / (s / np.sqrt(n))
    return stats.t.sf(t, n - 1)


def run_tests(pool, idx, explore, crit):
    """Per study: paired tests per feature, the sign test per block, and
    the per-block attention tests."""
    res = {}
    D = pool["est_D"][idx, :2] - pool["est_O"][idx, :2]        # S x N x 2
    for j, f in enumerate(FEATURES):
        res[("t", f)] = t_greater(D[:, :, j]) <= ALPHA
        res[("wilcoxon", f)] = explore.wilcoxon_greater(D[:, :, j]) <= ALPHA
        res[("mean_diff", f)] = D[:, :, j].mean(1)
        for b in ("D", "O"):
            e = pool[f"est_{b}"][idx, j]
            a = (e > pool[f"up_{b}"][idx, j]).sum(1)
            d = a + (e < pool[f"lo_{b}"][idx, j]).sum(1)
            res[(f"sign_{b}", f)] = a >= crit[d]
    ag = pool["agree_D"][idx] - pool["agree_O"][idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        res[("consistency_flags", "-")] = np.nan_to_num(
            t_greater(-ag), nan=1.0) <= ALPHA
    cd = pool["catch_D"][idx] - pool["catch_O"][idx]
    npos, nnz = (cd > 0).sum(1), (cd != 0).sum(1)
    pc = np.where(nnz > 0, stats.binom.sf(npos - 1, np.maximum(nnz, 1), 0.5),
                  1.0)
    res[("catch_flags", "-")] = pc <= ALPHA
    return res


# ------------------------------------------------------------
# The mixed model (Laplace approximation, EM updates)
# ------------------------------------------------------------

def mixed_templates(pk, model):
    """Design rows per set mapping: derealization block on set A or B.
    Columns: sat, con, sat x block, con x block (block +1/2 derealization,
    -1/2 ordinary) [, side]."""
    out = []
    for der in SETS:
        orr = [s for s in SETS if s != der][0]
        parts = []
        for s, dcode in ((der, 0.5), (orr, -0.5)):
            it = pk["lists"][f"items_{s}"]
            Z = np.column_stack([it["X"], dcode * it["X"]])
            if model != "two-weight":
                Z = np.column_stack([Z, it["side"]])
            parts.append(Z)
        out.append(np.vstack(parts))
    return np.stack(out)                     # (2, T, k)


def fit_mixed(Zt, tsel, Y, max_iter=MIXED_MAX_ITER, tol=MIXED_TOL):
    """Many studies at once. Zt (2, T, k) design templates; tsel (S, N)
    which template each person uses; Y (S, N, T) responses.
    Returns beta (S, k), its SE (S, k), sigma^2 (S, k), iterations used,
    and whether each study converged."""
    S, N, T = Y.shape
    k = Zt.shape[2]
    St = (Zt[:, :, :, None] * Zt[:, :, None, :]).reshape(2, T, k * k)
    PN = S * N
    y = Y.reshape(PN, T).astype(float)
    tf = tsel.reshape(PN)
    grp = [np.flatnonzero(tf == j) for j in range(2)]
    b = np.zeros((PN, k))
    beta = np.zeros((S, k))
    s2 = np.full((S, k), 0.1)
    conv = np.zeros(S, bool)
    it_used = np.zeros(S, int)

    def info(bb):
        A = np.empty((PN, k * k))
        r = np.empty((PN, T))
        for j in range(2):
            g = grp[j]
            if len(g) == 0:
                continue
            eta = bb[g] @ Zt[j].T
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            A[g] = (p * (1 - p)) @ St[j]
            r[g] = y[g] - p
        grad = np.empty((PN, k))
        for j in range(2):
            g = grp[j]
            if len(g):
                grad[g] = r[g] @ Zt[j]
        return A.reshape(PN, k, k), grad

    for it in range(max_iter):
        bet_p = np.repeat(beta, N, axis=0)
        s2_p = np.repeat(s2, N, axis=0)
        for _ in range(2):                  # Newton steps to the modes
            A, g = info(b)
            g = g - (b - bet_p) / s2_p
            H = A + np.einsum("pk,kl->pkl", 1.0 / s2_p, np.eye(k))
            step = np.linalg.solve(H, g[:, :, None])[:, :, 0]
            step = np.clip(step, -2.0, 2.0)
            b = b + step
        A, _ = info(b)
        H = A + np.einsum("pk,kl->pkl", 1.0 / s2_p, np.eye(k))
        Hd = np.diagonal(np.linalg.inv(H), axis1=1, axis2=2)
        bs = b.reshape(S, N, k)
        beta_new = bs.mean(1)
        s2_new = np.maximum(((bs - beta_new[:, None]) ** 2
                             + Hd.reshape(S, N, k)).mean(1), 1e-8)
        change = np.abs(beta_new - beta).max(1)
        newly = (~conv) & (change < tol)
        it_used[~conv] = it + 1
        conv |= newly
        beta = np.where(conv[:, None] & ~newly[:, None], beta, beta_new)
        s2 = np.where(conv[:, None] & ~newly[:, None], s2, s2_new)
        if conv.all():
            break
    A, _ = info(b)
    V = np.linalg.inv(A + 1e-10 * np.eye(k))              # person sampling cov
    s2_p = np.repeat(s2, N, axis=0)
    W = np.linalg.inv(V + np.einsum("pk,kl->pkl", s2_p, np.eye(k)))
    Wsum = W.reshape(S, N, k, k).sum(1)
    cov = np.linalg.inv(Wsum)
    se = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    return beta, se, s2, it_used, conv


def check_mixed(Zt, rng, ck):
    """Recover known parameters: 40 studies of 150 people simulated from the
    mixed model itself (no lapse); the z-scores of the fixed effects must
    look standard normal."""
    k = Zt.shape[2]
    beta = np.array([0.4, 0.6, 0.3, 0.4, 0.1][:k])
    sd = np.array([0.15, 0.25, 0.2, 0.3, 0.2][:k])
    S, N = 40, 150
    tsel = rng.integers(0, 2, (S, N))
    bi = beta + rng.normal(0, 1, (S, N, k)) * sd
    Z = Zt[tsel]                                            # S,N,T,k
    p = 1 / (1 + np.exp(-np.einsum("sntk,snk->snt", Z, bi)))
    Y = (rng.random(p.shape) < p).astype(np.uint8)
    est, se, s2, its, conv = fit_mixed(Zt, tsel, Y)
    z = (est - beta) / se
    ok = (abs(z.mean()) < 0.35 and 0.7 < z.std() < 1.35 and conv.all())
    ck.add("mixed model", "recovers known fixed effects (40 studies of 150 "
           "people simulated from the model): z-scores centred, SD near 1, "
           "every fit converged", ok,
           f"mean z {z.mean():+.2f}, SD z {z.std():.2f}, iterations "
           f"{its.min()}-{its.max()}; mean estimated SD "
           f"{np.sqrt(s2).mean(0).round(2).tolist()} (true {sd.tolist()})")


# ------------------------------------------------------------
# Sanity check against explore_block_length_v5
# ------------------------------------------------------------

def run_sanity(v4, explore, pk_by_size, fit, cap, ck):
    rows = []
    ex_res = pd.read_csv(root_path(EXPLORE_RESULTS_CSV))
    for size in BLOCK_SIZES:
        pk = pk_by_size[size]
        XA = pk["lists"]["items_A"]["X"]
        XB = pk["lists"]["items_B"]["X"]
        for scen in SANITY_SCENARIOS:
            st = cap.Streams(RANDOM_SEED, STREAM_KEY, "sanity", size, scen)
            # explore's own functions, unmodified
            esc = explore.PAIRED_SCENARIOS[scen]
            Dx, on_A, _, _ = explore.paired_pool(v4, XA, XB, esc,
                                                 st("explore_pool"))
            # explore's pool size and study count are its settings
            # (module variables); they are set to this script's values
            # before the run (see run_sanity's caller), the functions are
            # unchanged
            # this script, set to explore's model
            eff = EFFECTS[scen]
            pool = simulate_pool(pk, eff, NUISANCE_OFF, st("own_pool"), fit,
                                 P=POOL_PEOPLE, model="two-weight",
                                 lapse_fixed=True)
            D = pool["est_D"][:, :2] - pool["est_O"][:, :2]
            for N in SANITY_N:
                ix = explore.study_index(on_A, N, st(f"explore_idx{N}"))
                pe = explore.paired_tests(Dx, ix)
                idx, _, sb = draw_studies(pool, N, N_STUDIES,
                                          st(f"own_idx{N}"), exclusion=False)
                for j, f in enumerate(FEATURES):
                    d = D[idx, j]
                    own = {"t": rate_se(t_greater(d) <= ALPHA, sb),
                           "wilcoxon": rate_se(
                               explore.wilcoxon_greater(d) <= ALPHA, sb)}
                    for test in TESTS:
                        rx = float(np.mean(pe[(test, j)] <= ALPHA))
                        ro, se_o = own[test]
                        # same pool size and study count on both sides, so
                        # the same Monte Carlo SE: the difference has
                        # sqrt(2) x it
                        se = np.sqrt(2) * max(se_o, 1e-4)
                        er = ex_res[(ex_res["size"] == size)
                                    & (ex_res["mix"] == EXPLORE_MIX)
                                    & (ex_res["scenario"] == scen)
                                    & (ex_res["N"] == N)
                                    & (ex_res["test"] == test)][
                            f"power_{f}"]
                        rows.append({
                            "size": size, "scenario": scen, "N": N,
                            "test": test, "feature": FEATURE_LABEL[f],
                            "explore_functions": rx, "this_script": ro,
                            "se_each": se_o,
                            "z": (ro - rx) / se,
                            "draws_min": er.min(), "draws_mean": er.mean(),
                            "draws_max": er.max(), "n_draws": len(er)})
            log(f"sanity {size} {scen} done")
    df = pd.DataFrame(rows)
    az = df["z"].abs()
    ck.add("sanity", "this script reproduces explore_block_length_v5's "
           "functions under its simplified model (power and false positives "
           "on the frozen lists)",
           az.max() <= SANITY_MAX_Z and (az > 1.96).mean()
           <= SANITY_SHARE_BEYOND,
           f"{len(df)} comparisons; largest |z| {az.max():.2f}; "
           f"{(az > 1.96).mean():.1%} beyond 1.96; mean z {df['z'].mean():+.2f}")
    se = df["se_each"].clip(lower=0.002)
    inside = ((df["explore_functions"] >= df["draws_min"]
               - SANITY_RANGE_SE * se)
              & (df["explore_functions"] <= df["draws_max"]
                 + SANITY_RANGE_SE * se))
    df["within_draw_range"] = inside
    ck.add("sanity", "the frozen lists' power lies within explore's range "
           f"over its 20 draws of the same size (+/- {SANITY_RANGE_SE:g} SE)",
           inside.all() and (df["n_draws"] > 0).all(),
           f"{int(inside.sum())} of {len(df)} cells")
    return df


# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------

def scenario_rows(pool, scen, size, explore, crit, cap, tag=""):
    rows, sign_rows, bm_rows = [], [], []
    st = cap.Streams(RANDOM_SEED, STREAM_KEY, "studies", size, scen["name"],
                     scen["effect"], tag)
    keep = ~pool["excluded"]
    for N in N_GRID:
        idx, n_excl, sb = draw_studies(pool, N, N_STUDIES, st(f"N{N}"))
        r = run_tests(pool, idx, explore, crit)
        base = {"size": size, "scenario": scen["name"],
                "effect": scen["effect"], "role": scen["role"], "N": N}
        for test in TESTS:
            for f in FEATURES:
                p, se = rate_se(r[(test, f)], sb)
                rows.append({**base, "test": test,
                             "feature": FEATURE_LABEL[f], "rate": p,
                             "se": se,
                             "mean_diff": float(np.mean(
                                 r[("mean_diff", f)])),
                             "excluded_per_study": float(np.mean(n_excl))})
        for b, lab in (("D", "derealization"), ("O", "ordinary")):
            for f in FEATURES:
                p, se = rate_se(r[(f"sign_{b}", f)], sb)
                sign_rows.append({**base, "block": lab,
                                  "feature": FEATURE_LABEL[f],
                                  "claim_rate": p, "se": se})
        pc, sc_ = rate_se(r[("consistency_flags", "-")], sb)
        pk_, sk = rate_se(r[("catch_flags", "-")], sb)
        bm_rows.append({**base, "consistency_flags": pc,
                        "consistency_se": sc_, "catch_flags": pk_,
                        "catch_se": sk})
    means = {"size": size, "scenario": scen["name"], "effect": scen["effect"],
             "role": scen["role"], "N": "pool (analysable people)",
             "excluded_share": float(pool["excluded"].mean()),
             "catch_D": float(pool["catch_D"][keep].mean()),
             "catch_O": float(pool["catch_O"][keep].mean()),
             "agree_D": float(pool["agree_D"][keep].mean()),
             "agree_O": float(pool["agree_O"][keep].mean()),
             "lapse_D": float(pool["lapse_D"][keep].mean()),
             "lapse_O": float(pool["lapse_O"][keep].mean())}
    return rows, sign_rows, bm_rows, means


def replicate_compare(a, b):
    m = a.merge(b, on=["size", "scenario", "effect", "N", "test", "feature"],
                suffixes=("_1", "_2"))
    se = np.sqrt(m["se_1"] ** 2 + m["se_2"] ** 2).clip(lower=1e-4)
    m["z"] = (m["rate_2"] - m["rate_1"]) / se
    return m


def apply_rule(pw):
    se0 = np.sqrt(ALPHA * (1 - ALPHA) / N_STUDIES)
    nul = pw[(pw["role"] == "null (rule)") & pw["N"].isin(RULE_N)].copy()
    nul["limit"] = ALPHA + RULE_SE_MULT * np.maximum(nul["se"], se0)
    pwr = pw[(pw["role"] == "power")
             & pw["effect"].isin(["planning", "half shift"])
             & pw["N"].isin(RULE_N)]
    res = {"limit": float(nul["limit"].mean()),
           "limit_range": (float(nul["limit"].min()),
                           float(nul["limit"].max())), "tests": {}}
    n_cells = len(nul) // len(TESTS)
    res["cells"] = n_cells
    p1 = float(stats.norm.sf(RULE_SE_MULT))
    res["expected_exceed_if_exact"] = n_cells * p1
    res["chance_95"] = int(stats.binom.ppf(0.95, n_cells, p1))
    for test in TESTS:
        q = nul[nul["test"] == test]
        ex = q[q["rate"] > q["limit"]]
        res["tests"][test] = {
            "qualifies": len(ex) == 0, "n_exceed": len(ex),
            "exceed_cells": [f"{r.scenario}, {r.size}, N={r.N}, "
                             f"{r.feature}: {r.rate:.4f} > {r.limit:.4f}"
                             for r in ex.itertuples()],
            "worst": float(q["rate"].max()),
            "worst_where": (lambda r: f"{r['scenario']}, {r['size']}, "
                            f"N={r['N']}, {r['feature']}")(
                q.loc[q["rate"].idxmax()]),
            "mean_null": float(q["rate"].mean()),
            "power": float(pwr[pwr["test"] == test]["rate"].mean())}
    ok = [t for t in TESTS if res["tests"][t]["qualifies"]]
    if not ok:
        res["chosen"] = None
    elif len(ok) == 1:
        res["chosen"] = ok[0]
    else:
        best = max(ok, key=lambda t: res["tests"][t]["power"])
        other = [t for t in ok if t != best][0]
        gap = res["tests"][best]["power"] - res["tests"][other]["power"]
        res["chosen"] = RULE_PREFER if gap <= RULE_TIE and \
            RULE_PREFER in ok else best
        res["gap"] = gap
    return res


# ------------------------------------------------------------
# Figures
# ------------------------------------------------------------

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
             "#56B4E9", "#000000"]


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)


def figures(pw, bm_means, bm, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    TEST_LABEL = {"t": "paired t-test", "wilcoxon": "Wilcoxon signed-rank"}
    # 1. power against N
    fig, axes = plt.subplots(len(BLOCK_SIZES), 2, figsize=(10, 4 *
                             len(BLOCK_SIZES)), sharey=True, squeeze=False)
    series = [("planning", "power", 0), ("half shift", "power", 1),
              ("planning", "power (no nuisances)", 2),
              ("half shift", "power (no nuisances)", 4),
              ("no effect", "null (rule)", 6)]
    for i, size in enumerate(BLOCK_SIZES):
        for j, f in enumerate(("con", "sat")):
            ax = axes[i, j]
            for eff, role, c in series:
                for test, ls in (("t", "-"), ("wilcoxon", "--")):
                    q = pw[(pw["size"] == size) & (pw["effect"] == eff)
                           & (pw["role"] == role) & (pw["test"] == test)
                           & (pw["feature"] == FEATURE_LABEL[f])]
                    if eff == "no effect":
                        q = q[q["scenario"] == "all nuisances (defaults)"]
                    if q.empty:
                        continue
                    lab = (f"{eff}, "
                           f"{'nuisances at defaults' if role != 'power (no nuisances)' else 'no nuisances'}"
                           f", {TEST_LABEL[test]}")
                    ax.plot(q["N"], q["rate"], ls, color=OKABE_ITO[c],
                            marker="o", ms=3, label=lab)
            ax.axhline(ALPHA, color="#888888", lw=0.8, ls=":")
            ax.set_title(f"{FEATURE_LABEL[f].capitalize()}, {size} trials per "
                         "block", fontsize=10)
            ax.set_ylim(0, 1)
            _style(ax)
            ax.set_xlabel("Report-group participants (analysable)")
            if j == 0:
                ax.set_ylabel("Probability the one-sided paired\ntest is "
                              "significant")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.suptitle("Primary test: derealization-block weight minus "
                 "ordinary-block weight\n(responder shares, shifts and "
                 "nuisances are planning assumptions)", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "power_vs_n.png"), dpi=FIG_DPI)
    plt.close(fig)

    # 2. false positives across scenarios at N = 15
    nn = 15 if 15 in N_GRID else N_GRID[len(N_GRID) // 2]
    q = pw[(pw["effect"] == "no effect") & (pw["N"] == nn)]
    names = list(dict.fromkeys(q["scenario"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(names) + 2),
                             sharey=True)
    for j, f in enumerate(("con", "sat")):
        ax = axes[j]
        for k_, (size, test) in enumerate([(s, t) for s in BLOCK_SIZES
                                           for t in TESTS]):
            r = q[(q["size"] == size) & (q["test"] == test)
                  & (q["feature"] == FEATURE_LABEL[f])].set_index(
                "scenario").reindex(names)
            y = np.arange(len(names)) + (k_ - 1.5) * 0.15
            ax.plot(r["rate"], y, "o", ms=3.5, color=OKABE_ITO[k_],
                    label=f"{size} trials, {TEST_LABEL[test]}")
        ax.axvline(ALPHA, color="#888888", lw=0.8)
        ax.set_title(f"{FEATURE_LABEL[f].capitalize()} (N = {nn})",
                     fontsize=10)
        ax.set_xlabel("Share of studies significant with no effect")
        _style(ax)
    axes[0].set_yticks(np.arange(len(names)))
    axes[0].set_yticklabels(names, fontsize=7)
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle("False positives with no effect, by nuisance (rows after "
                 "'all nuisances (high ends)' are threats no test can "
                 "remove)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "false_positives.png"), dpi=FIG_DPI)
    plt.close(fig)

    # 3. per-block measures picking up an episode lapse difference
    fig, axes = plt.subplots(2, len(BLOCK_SIZES), figsize=(10, 7.2),
                             sharey=True, sharex=True, squeeze=False)
    variants = [("", "no pair-specific preferences"),
                (f", with pair-specific preferences SD {PAIR_SD_FOR_ROWS:g}",
                 f"pair-specific preferences SD {PAIR_SD_FOR_ROWS:g}")]
    for row, (suffix, title) in enumerate(variants):
        sel = [("no nuisances", "no lapse difference", 0)] if not suffix \
            else []
        sel += [(f"derealization lapse +{el:g}{suffix}",
                 f"derealization lapse +{el:g}", c + 1)
                for c, el in enumerate(EPISODE_LAPSES)]
        for i, size in enumerate(BLOCK_SIZES):
            ax = axes[row, i]
            for name, lab, c in sel:
                r = bm[(bm["size"] == size) & (bm["scenario"] == name)
                       & (bm["effect"] == "no effect")]
                if r.empty:
                    continue
                ax.plot(r["N"], r["consistency_flags"], "-",
                        color=OKABE_ITO[c], marker="o", ms=3,
                        label=f"{lab}: consistency")
                ax.plot(r["N"], r["catch_flags"], "--", color=OKABE_ITO[c],
                        marker="s", ms=3, label=f"{lab}: catch trials")
            ax.axhline(ALPHA, color="#888888", lw=0.8, ls=":")
            ax.set_title(f"{size} trials per block, {title}", fontsize=9)
            if row == 1:
                ax.set_xlabel("Report-group participants (analysable)")
            _style(ax)
        axes[row, 0].set_ylabel("Share of studies flagging more\nlapses "
                                "in the derealization block")
        axes[row, -1].legend(fontsize=6, frameon=False)
    fig.suptitle("Do per-block consistency and catch failures pick up a "
                 "lapse difference? (lapse values are assumptions)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "block_measures.png"), dpi=FIG_DPI)
    plt.close(fig)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

def tab(df, nd=3):
    return "  " + df.to_string(index=False, float_format=lambda x:
                               f"{x:.{nd}f}").replace("\n", "\n  ")


def wide(pw, rows_filter, n_list, by=("scenario",)):
    q = pw[rows_filter & pw["N"].isin(n_list)].copy()
    q["col"] = (q["test"].map({"t": "t", "wilcoxon": "W"}) + ":"
                + q["feature"].str[:3] + ":N" + q["N"].astype(str))
    piv = q.pivot_table(index=list(by), columns="col", values="rate",
                        sort=False)
    cols = [f"{t}:{f}:N{n}" for f in ("con", "sat") for t in ("t", "W")
            for n in n_list]
    return piv[[c for c in cols if c in piv.columns]]


def write_report(out, ck, scen_df, pw, sg, bm, bmm, mx, san, rep, rule,
                 mixed_verdict, meta):
    L = []
    w = L.append
    w("THE PRIMARY PAIRED TEST  (simulate_power_paired_v5.py, script 7)")
    w("=" * 78)
    w(f"run       {meta['started']}  ({meta['minutes']:.1f} min)")
    w(f"output    {out}")
    w(f"python    {meta['python']}, numpy {meta['numpy']}, scipy "
      f"{meta['scipy']}, pandas {meta['pandas']}")
    w("")
    w("PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES. No image-choice "
      "data exist.")
    w("The responder shares and shifts, taste spread, lapse rates and every "
      "nuisance")
    w("value below were chosen for planning; every power and false-positive "
      "rate here")
    w("depends on them.")
    w("")
    w("1. WHAT WAS DONE")
    w("-" * 78)
    w(f"Per-person model: {PERSON_MODEL} model (script 6), Firth, fitted per "
      "block on the")
    w("analysed trials; thresholds from script 6 (secondary sign test only).")
    w("Primary test per feature: one-sided, derealization-block weight minus "
      "ordinary-")
    w(f"block weight, alpha {ALPHA}, uncorrected; paired t-test and exact "
      "Wilcoxon.")
    w(f"Pools of {POOL_PEOPLE:,} people per scenario and size, each fitted "
      f"on both blocks; {N_STUDIES:,}")
    w("studies per cell drawn from the pool. Every rate's Monte Carlo SE is "
      "measured from")
    w(f"{N_BATCHES} disjoint sub-pools (it includes the pool's own error): "
      f"largest power SE {pw['se'].max():.4f},")
    w(f"largest SE of a no-effect rate "
      f"{pw[pw['effect'] == 'no effect']['se'].max():.4f} (columns 'se' in "
      "the CSV files).")
    w("Recruitment: each participant takes the next slot of the frozen "
      "assignment table;")
    w(f"anyone with {CATCH_FAIL_EXCLUDE}+ catch failures of "
      f"{2 * CATCH_PER_BLOCK} is excluded and uses up the slot.")
    w(f"N = {N_GRID[0]}-{N_GRID[-1]} analysable people; tables show N = "
      f"{', '.join(map(str, TABLE_N))}, paired_power.csv has every N.")
    w("The person model is in the script's docstring. Values:")
    w(f"  taste SD {TASTE_SD}; personal lapse Gamma, mean {LAPSE_MEAN}, SD "
      f"{LAPSE_BETWEEN_SD} x mean;")
    w(f"  catch p(fail) = {CATCH_P_FAIL_PER_LAPSE} x lapse, "
      f"{CATCH_PER_BLOCK} per block.")
    w("  Nuisances at their defaults [ASSUMPTIONS]:")
    for k_, v in NUISANCE_DEFAULT.items():
        if v != NUISANCE_OFF[k_]:
            w(f"    {k_:18s} {v}")
    w("  (scene SD = scene_sd_mult x taste SD; dull and odd in per-SD "
      "weight units; side in logit)")
    w("")
    w("2. CHECKS")
    w("-" * 78)
    for r in ck.rows:
        w(f"  {r['status']}  [{r['section']}] {r['check']}")
        if r["detail"]:
            w(f"          {r['detail']}")
    w(f"\n  {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL")
    w("")
    w("3. SANITY CHECK: explore_block_length_v5's simplified model on the "
      "frozen lists")
    w("-" * 78)
    w("explore_functions: explore's paired_pool / study_index / paired_tests "
      "run")
    w(f"unmodified (its pool and study settings set to {POOL_PEOPLE:,} and "
      f"{N_STUDIES:,}); this_script: this script's simulation")
    w("set to the same model (two-weight fit, lapse 0.1 for everyone, taste "
      "the same in")
    w("both blocks, no exclusion). draws: explore's published range over its "
      "20 draws.")
    w(tab(san[["size", "scenario", "N", "test", "feature",
                "explore_functions", "this_script", "se_each", "z",
                "draws_min", "draws_max"]]))
    w("")
    w("4. FALSE POSITIVES WITH NO EFFECT")
    w("-" * 78)
    w("Share of studies where the one-sided paired test is significant. "
      "t = paired")
    w("t-test, W = Wilcoxon; con = contrast (H1), sat = saturation (H2). "
      "Monte Carlo SE about")
    w(f"{pw[pw['role'] == 'null (rule)']['se'].mean():.4f}. Rule limit per "
      f"cell {ALPHA} + {RULE_SE_MULT:g} SE: "
      f"{rule['limit_range'][0]:.4f}-{rule['limit_range'][1]:.4f}.")
    for role, title in (("null (rule)", "Rows used by the test-choice rule "
                         "(each default nuisance alone, then all)"),
                        ("null (sensitivity)", "Sensitivity rows (the ranges"
                         "; not used by the rule)"),
                        ("threat", "THREATS NO TEST CAN REMOVE (more lapses "
                         "in the derealization block,\nalone and with a "
                         "shared bias; a bias that differs between episodes)"
                         )):
        w(f"\n  {title}")
        for size in BLOCK_SIZES:
            t_ = wide(pw, (pw["role"] == role) & (pw["size"] == size),
                      TABLE_N)
            w(f"\n  {size} trials per block")
            w("  " + t_.to_string(float_format=lambda x: f"{x:.3f}")
              .replace("\n", "\n  "))
    w("\n  Why the threat rows are threats: a bias present in both blocks "
      "cancels in the")
    w("  difference only if both blocks express it equally. More lapses in "
      "the")
    w("  derealization block shrink its weights towards zero: a shared "
      "preference against")
    w("  the edited image (negative weights) then looks like a positive "
      "difference, a")
    w("  shared preference for it ('duller') like a negative one. A bias "
      "stronger in one")
    w("  episode is indistinguishable from an effect. Section 9 shows how "
      "well the")
    w("  per-block attention measures pick up a lapse difference.")
    w("")
    w("5. POWER [responder shares and shifts: ASSUMPTIONS]")
    w("-" * 78)
    for role, title in (("power", "All nuisances at their defaults (the "
                         "headline)"),
                        ("power (no nuisances)", "No nuisances"),
                        ("power (sensitivity)", "All nuisances at their "
                         "defaults plus pair-specific preferences (a "
                         "sensitivity row)")):
        w(f"\n  {title}")
        for size in BLOCK_SIZES:
            t_ = wide(pw, (pw["role"] == role) & (pw["size"] == size),
                      TABLE_N, by=("effect",))
            w(f"\n  {size} trials per block")
            w("  " + t_.to_string(float_format=lambda x: f"{x:.3f}")
              .replace("\n", "\n  "))
    w("\n  Planning and half shift at every N (all nuisances at defaults):")
    q = pw[(pw["role"] == "power") & pw["effect"].isin(["planning",
                                                        "half shift"])]
    piv = q.pivot_table(index=["size", "effect", "N"],
                        columns=["test", "feature"], values="rate")
    w("  " + piv.to_string(float_format=lambda x: f"{x:.3f}")
      .replace("\n", "\n  "))
    w("")
    w("6. THE CHOICE OF PRIMARY TEST")
    w("-" * 78)
    w("The rule, fixed before the run: a test qualifies if its false-"
      "positive rate is")
    w(f"at most {ALPHA} + {RULE_SE_MULT:g} SE (that cell's Monte Carlo SE) "
      f"in every rule row (section 4), at N = {', '.join(map(str, RULE_N))},"
      " both sizes")
    w("and both features. Among qualifying tests, the higher mean power over "
      "planning")
    w(f"and half shift (all nuisances at defaults, same N, sizes, features) "
      f"wins; within")
    w(f"{RULE_TIE} of each other, the {RULE_PREFER} test.")
    w(f"  Cells per test: {rule['cells']}. If a test's true rate were "
      f"exactly {ALPHA} everywhere, about")
    w(f"  {rule['expected_exceed_if_exact']:.1f} of them would exceed the "
      f"limit by chance alone (up to {rule['chance_95']} in 95% of runs), "
      "so a test")
    w("  that is exactly calibrated can fail this rule by chance; the cells "
      "over the limit")
    w("  are listed so this can be judged. The rule is applied as fixed; "
      "this line is")
    w("  information only.")
    for test, r in rule["tests"].items():
        w(f"  {test:9s} qualifies: {'yes' if r['qualifies'] else 'NO'}; "
          f"cells over the limit {r['n_exceed']}; worst "
          f"{r['worst']:.4f} ({r['worst_where']});")
        w(f"            mean false-positive rate {r['mean_null']:.4f}; mean "
          f"power {r['power']:.4f}")
        for c in r["exceed_cells"]:
            w(f"            over the limit: {c}")
    if rule["chosen"] is None:
        w("\n  NO TEST QUALIFIES. The rule chooses nothing; see the cells "
          "over the limit.")
    else:
        extra = (f" (power gap {rule['gap']:+.4f})" if "gap" in rule
                 else "")
        w(f"\n  RECOMMENDED PRIMARY TEST: {rule['chosen']}{extra}.")
    w("")
    w("7. THE MIXED-EFFECTS LOGISTIC MODEL (alternative; the user's addition)")
    w("-" * 78)
    if mx is None or mx.empty:
        w("  Not run (RUN_MIXED_MODEL = False).")
    else:
        w(f"  {MIXED_STUDIES:,} studies per cell; the paired tests on the "
          "SAME studies beside it.")
        w("  z: one-sided Wald z; t(N-1): the same statistic against t with "
          "N-1 df.")
        w(tab(mx, 3))
        w("")
        for line in mixed_verdict:
            w("  " + line)
    w("")
    w("8. SECONDARY: SIGN TEST ON DETECTED COUNTS WITHIN EACH BLOCK")
    w("-" * 78)
    w("Per block, each person's weight is classified with the per-set "
      "thresholds of")
    w(f"script 6 ({PERSON_MODEL} model); the claim needs d >= 5 detections "
      "and the exact one-sided")
    w("binomial (v4's Table 4 rule). Derealization block: the secondary "
      "H1/H2.")
    w("Ordinary block: the same rule, as a reference.")
    w("Note: within one block the sign test asks whether a directional "
      "preference exists;")
    w("it cannot tell a preference shared by both episodes (e.g. 'memory "
      "makes it duller',")
    w("one of the default nuisances) from a derealization effect, so with "
      "nuisances at")
    w("their defaults it claims in many studies with no derealization "
      "effect (second table).")
    w("That is why the paired test is primary.")
    for role in ("power", "null (rule)"):
        q = sg[(sg["role"] == role) & sg["N"].isin(TABLE_N)]
        if role == "null (rule)":
            q = q[q["scenario"].isin(["no nuisances",
                                      "all nuisances (defaults)"])]
        q = q.copy()
        q["col"] = (q["block"].str[:3] + ":" + q["feature"].str[:3] + ":N"
                    + q["N"].astype(str))
        piv = q.pivot_table(index=["size", "scenario", "effect"],
                            columns="col", values="claim_rate", sort=False)
        cols = [f"{b}:{f}:N{n}" for b in ("der", "ord") for f in ("con",
                                                                 "sat")
                for n in TABLE_N]
        w("")
        w("  " + piv[[c for c in cols if c in piv.columns]].to_string(
            float_format=lambda x: f"{x:.3f}").replace("\n", "\n  "))
    w("")
    w("9. PER-BLOCK ATTENTION MEASURES")
    w("-" * 78)
    w("Means over analysable people in each pool: lapse rate, catch failures "
      "(of 5),")
    w("consistency (share of repeats answered the same way), per block; and "
      "the share")
    w("excluded. Then, per study, how often each measure flags more lapses "
      "in the")
    w(f"derealization block (consistency: one-sided paired t; catch: "
      f"one-sided sign test; alpha {ALPHA}).")
    sel = bmm[bmm["effect"] == "no effect"][[
        "size", "scenario", "excluded_share", "lapse_D", "lapse_O",
        "catch_D", "catch_O", "agree_D", "agree_O"]]
    w(tab(sel, 3))
    q = bm[(bm["effect"] == "no effect") & bm["N"].isin(TABLE_N)].copy()
    piv = q.pivot_table(index=["size", "scenario"], columns="N",
                        values=["consistency_flags", "catch_flags"],
                        sort=False)
    w("")
    w("  " + piv.to_string(float_format=lambda x: f"{x:.3f}")
      .replace("\n", "\n  "))
    w("")
    w("10. MONTE CARLO CHECK: headline cells on two independent pools")
    w("-" * 78)
    w(tab(rep[rep["N"].isin(TABLE_N)][["size", "effect", "N", "test",
                                        "feature", "rate_1", "rate_2",
                                        "se_1", "z"]], 4))
    w("")
    w("11. FILES")
    w("-" * 78)
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
    hashes = {}
    out = resolve_output_dir()
    log(f"output folder {out}")
    v4 = load_module(V4_SIM_SCRIPT, ck, hashes)
    cap = load_module(CAPACITY_SCRIPT, ck, hashes)
    explore = load_module(EXPLORE_SCRIPT, ck, hashes)
    s6 = load_module(THRESHOLDS_SCRIPT, ck, hashes)
    fit = Fitter(v4, s6)
    TH = load_thresholds(ck, hashes)
    p = root_path(EXPLORE_RESULTS_CSV)
    ck.add("inputs", "explore's paired_power.csv exists", os.path.exists(p),
           p)
    hashes["explore paired_power.csv"] = sha256_of(p)
    pk = {size: load_package(size, ck, hashes) for size in BLOCK_SIZES}
    crit = v4.sign_test_critical(max(N_GRID))

    # --- tool checks ---
    rng = cap.Streams(RANDOM_SEED, STREAM_KEY)("tool_checks")
    worst = 0.0
    for n in (10, 15, 20):
        d = rng.normal(0.2, 1, (200, n))
        pv = explore.wilcoxon_greater(d)
        ref = np.array([stats.wilcoxon(r, alternative="greater",
                                       method="exact").pvalue for r in d])
        worst = max(worst, float(np.max(np.abs(pv - ref))))
    ck.add("tools", "explore's vectorised Wilcoxon equals SciPy's exact test",
           worst < 1e-12, f"max |difference| in p {worst:.1e}")
    d = rng.normal(0.2, 1, (200, 12))
    ref = stats.ttest_1samp(d, 0, axis=1, alternative="greater").pvalue
    ck.add("tools", "the paired t-test equals SciPy's", np.max(np.abs(
        t_greater(d) - ref)) < 1e-12, "")
    ck.add("tools", "sign-test critical values: none below 5 detections, "
           "5 of 5 at d = 5 (v4's Table 4 rule)",
           np.all(np.isinf(crit[:5])) and crit[5] == 5,
           f"d=5..10: {crit[5:11].tolist()}")
    X3 = model_X(pk[BLOCK_SIZES[-1]]["lists"]["items_A"]["X"],
                 pk[BLOCK_SIZES[-1]]["lists"]["items_A"]["side"], "side")
    W = rng.normal(0.3, 0.5, (30, 2))
    Y = (rng.random((30, len(X3))) < 0.5).astype(float)
    b1 = fit(X3, Y)
    b2 = np.array([v4.fit_firth(X3, Y[i]) for i in range(30)])
    ck.add("tools", "the side-model fit equals v4's single-person fit_firth",
           np.max(np.abs(b1 - b2)) < 1e-9,
           f"max |difference| {np.max(np.abs(b1 - b2)):.1e}")

    # --- sanity check ---
    log("sanity check against explore_block_length_v5")
    explore.POOL_PEOPLE = POOL_PEOPLE      # explore's settings, set to this
    explore.N_STUDIES = N_STUDIES          # script's values for the check
    san = run_sanity(v4, explore, pk, fit, cap, ck)

    # --- scenarios ---
    scen = build_scenarios()
    scen_df = pd.DataFrame([{"scenario": s["name"], "effect": s["effect"],
                             "role": s["role"], **{k: str(v) for k, v in
                                                   s["nuis"].items()}}
                            for s in scen])
    need_y = {(e, n) for e, n in MIXED_CELLS} if RUN_MIXED_MODEL else set()
    PW, SG, BM, BMM, pools_for_mixed = [], [], [], [], {}
    for size in BLOCK_SIZES:
        for s in scen:
            st = cap.Streams(RANDOM_SEED, STREAM_KEY, "pool", size, s["name"],
                             s["effect"])
            keep = (s["effect"], s["name"]) in need_y
            pool = simulate_pool(pk[size], EFFECTS[s["effect"]], s["nuis"],
                                 st("people"), fit, keep_y=keep)
            person_thresholds(pool, size, TH)
            a, b, c, m = scenario_rows(pool, s, size, explore, crit, cap)
            PW += a
            SG += b
            BM += c
            BMM.append(m)
            if keep:
                pools_for_mixed[(size, s["effect"], s["name"])] = pool
            log(f"{size} | {s['role']:20s} | {s['effect']:30s} | "
                f"{s['name']}")
    pw, sg = pd.DataFrame(PW), pd.DataFrame(SG)
    bm, bmm = pd.DataFrame(BM), pd.DataFrame(BMM)

    # --- replicate pools ---
    log("replicate pools")
    R2 = []
    for size in BLOCK_SIZES:
        for eff in REPLICATE_EFFECTS:
            s = [x for x in scen if x["effect"] == eff
                 and x["name"] == "all nuisances (defaults)"][0]
            st = cap.Streams(RANDOM_SEED, STREAM_KEY, "replicate", size, eff)
            pool = simulate_pool(pk[size], EFFECTS[eff], s["nuis"],
                                 st("people"), fit)
            person_thresholds(pool, size, TH)
            a, _, _, _ = scenario_rows(pool, s, size, explore, crit, cap,
                                       tag="replicate")
            R2 += a
    r2 = pd.DataFrame(R2)
    r1 = pw[(pw["scenario"] == "all nuisances (defaults)")
            & pw["effect"].isin(REPLICATE_EFFECTS)]
    rep = replicate_compare(r1, r2)
    az = rep["z"].abs()
    ck.add("Monte Carlo", "headline cells agree between two independent "
           "pools (pool-to-pool error is small)",
           az.max() <= SANITY_MAX_Z and (az > 1.96).mean()
           <= SANITY_SHARE_BEYOND,
           f"{len(rep)} comparisons; largest |z| {az.max():.2f}; "
           f"{(az > 1.96).mean():.1%} beyond 1.96; SD z {rep['z'].std():.2f}")

    # --- general checks on the results ---
    nul = pw[(pw["scenario"] == "no nuisances") & (pw["effect"] ==
                                                   "no effect")]
    zz = (nul["rate"] - ALPHA) / nul["se"]
    ck.add("results", "with no effect and no nuisances, both tests stay "
           f"at or below {ALPHA} + 3 SE (measured Monte Carlo SE) at every N",
           (zz <= 3).all(), f"largest rate {nul['rate'].max():.4f}, largest "
           f"z {zz.max():+.2f}; mean rate t "
           f"{nul[nul['test'] == 't']['rate'].mean():.4f}, Wilcoxon "
           f"{nul[nul['test'] == 'wilcoxon']['rate'].mean():.4f}")
    def rises(g):
        r, e = g["rate"].to_numpy(), g["se"].to_numpy()
        return bool(np.all(np.diff(r) > -3 * np.sqrt(e[1:] ** 2
                                                     + e[:-1] ** 2) - 0.002))
    mono = pw[(pw["role"] == "power")].sort_values("N").groupby(
        ["size", "effect", "test", "feature"])[["rate", "se"]].apply(rises)
    ck.add("results", "power rises with N (within 3 SE of Monte Carlo "
           "noise between neighbouring N)",
           mono.all(), f"{int(mono.sum())} of {len(mono)} series")
    ex = bmm[bmm["effect"] == "no effect"]
    ck.add("results", "the share excluded for 3+ catch failures matches the "
           "lapse model (about 5% at mean lapse 0.1, no nuisances)",
           ex[ex["scenario"] == "no nuisances"]["excluded_share"].between(
               0.03, 0.07).all(),
           f"share excluded, no nuisances: "
           f"{ex[ex['scenario'] == 'no nuisances']['excluded_share'].round(3).tolist()}")

    rule = apply_rule(pw)
    log(f"rule: chosen {rule['chosen']}")

    # --- mixed model ---
    mx_rows, mixed_verdict = [], []
    if RUN_MIXED_MODEL:
        model = PERSON_MODEL
        check_mixed(mixed_templates(pk[BLOCK_SIZES[-1]], model),
                    cap.Streams(RANDOM_SEED, STREAM_KEY)("mixed_check"), ck)
        for size in BLOCK_SIZES:
            Zt = mixed_templates(pk[size], model)
            for eff, name in MIXED_CELLS:
                pool = pools_for_mixed[(size, eff, name)]
                st = cap.Streams(RANDOM_SEED, STREAM_KEY, "mixed", size, eff,
                                 name)
                for N in MIXED_N:
                    t0 = time.time()
                    idx, _, sb = draw_studies(pool, N, MIXED_STUDIES,
                                              st(f"N{N}"))
                    r = run_tests(pool, idx, explore, crit)
                    tsel = (pool["der_set"][idx] == SETS[1]).astype(int)
                    Z_rej = {f: [] for f in FEATURES}
                    T_rej = {f: [] for f in FEATURES}
                    convs, its = [], []
                    for c0 in range(0, MIXED_STUDIES, MIXED_CHUNK):
                        sl = slice(c0, c0 + MIXED_CHUNK)
                        Y = np.concatenate([pool["Y_D"][idx[sl]],
                                            pool["Y_O"][idx[sl]]], axis=2)
                        beta, se, s2, it_, cv = fit_mixed(Zt, tsel[sl], Y)
                        convs.append(cv)
                        its.append(it_)
                        for j, f in enumerate(FEATURES):
                            z = beta[:, 2 + j] / se[:, 2 + j]
                            Z_rej[f].append(stats.norm.sf(z) <= ALPHA)
                            T_rej[f].append(stats.t.sf(z, N - 1) <= ALPHA)
                    cv = np.concatenate(convs)
                    for f in FEATURES:
                        mz, mz_se = rate_se(np.concatenate(Z_rej[f]), sb)
                        mxz = np.concatenate(Z_rej[f]).astype(float)
                        gain_t = rate_se(mxz - r[("t", f)], sb)
                        gain_w = rate_se(mxz - r[("wilcoxon", f)], sb)
                        mx_rows.append({
                            "size": size, "effect": eff, "nuisances": name,
                            "N": N, "feature": FEATURE_LABEL[f],
                            "mixed_z": mz, "mixed_z_se": mz_se,
                            "mixed_t": float(np.mean(np.concatenate(
                                T_rej[f]))),
                            "paired_t": float(np.mean(r[("t", f)])),
                            "wilcoxon": float(np.mean(r[("wilcoxon", f)])),
                            "gain_over_t": gain_t[0],
                            "gain_over_t_se": gain_t[1],
                            "gain_over_wilcoxon": gain_w[0],
                            "gain_over_wilcoxon_se": gain_w[1],
                            "converged": float(cv.mean()),
                            "iterations_max": int(np.concatenate(its).max())})
                    log(f"mixed {size} {eff} / {name} N={N}: "
                        f"{time.time() - t0:.0f}s, converged {cv.mean():.3f}")
        mx = pd.DataFrame(mx_rows)
        ck.add("mixed model", "every mixed-model fit converged",
               (mx["converged"] == 1).all(),
               f"lowest share converged {mx['converged'].min():.4f}")
        se0m = np.sqrt(ALPHA * (1 - ALPHA) / MIXED_STUDIES)
        nulm = mx[mx["effect"] == "no effect"]
        lim_m = ALPHA + RULE_SE_MULT * np.maximum(nulm["mixed_z_se"], se0m)
        fp_ok = bool((nulm["mixed_z"] <= lim_m).all())
        lim_m = float(lim_m.max())
        hs = mx[(mx["effect"] == "half shift") & (mx["N"] == MIXED_GAIN_N)]
        against = [rule["chosen"]] if rule["chosen"] else list(TESTS)
        mixed_verdict = [
            f"False positives (Wald z): largest {nulm['mixed_z'].max():.4f} "
            f"against a limit of up to {lim_m:.4f} -> "
            f"{'controlled' if fp_ok else 'NOT controlled'}."]
        if not rule["chosen"]:
            mixed_verdict.append("No primary test was chosen by the rule, "
                                 "so the mixed model is compared with both.")
        worth_all = []
        for test in against:
            gains = hs["gain_over_t" if test == "t"
                       else "gain_over_wilcoxon"].to_numpy()
            worth = fp_ok and bool(np.all(gains >= MIXED_GAIN))
            worth_all.append(worth)
            mixed_verdict.append(
                f"Power gain over the {test} test under half shift at N = "
                f"{MIXED_GAIN_N} (size x feature): "
                f"{', '.join(f'{g:+.3f}' for g in gains)}; needed >= "
                f"{MIXED_GAIN} in every one -> "
                f"{'worth registering' if worth else 'not worth registering'}"
                ".")
        mixed_verdict.append(
            "VERDICT: the mixed model is "
            + ("WORTH REGISTERING." if all(worth_all) else
               "NOT worth registering by this criterion."))
    else:
        mx = pd.DataFrame()

    # --- write ---
    scen_df.to_csv(os.path.join(out, "scenarios.csv"), index=False)
    pw.to_csv(os.path.join(out, "paired_power.csv"), index=False)
    sg.to_csv(os.path.join(out, "secondary_sign.csv"), index=False)
    pd.concat([bmm, bm], ignore_index=True).to_csv(
        os.path.join(out, "block_measures.csv"), index=False)
    mx.to_csv(os.path.join(out, "mixed_model.csv"), index=False)
    san.to_csv(os.path.join(out, "sanity_explore.csv"), index=False)
    rep.to_csv(os.path.join(out, "pool_replicates.csv"), index=False)
    if MAKE_FIGURES:
        try:
            figures(pw, bmm, bm, out)
        except ImportError:
            log("matplotlib not installed: no figures")
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(jsonable(settings_dict()), f, indent=2)
    pd.DataFrame(ck.rows).to_csv(os.path.join(out, "checks.csv"),
                                 index=False)
    files = sorted(set(os.listdir(out)) | {"report.txt", "run_meta.json"})
    import scipy
    meta = {"started": started, "minutes": (time.time() - T0) / 60,
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__,
            "files": files}
    write_report(out, ck, scen_df, pw, sg, bm, bmm, mx, san, rep, rule,
                 mixed_verdict, meta)
    with open(os.path.join(out, "run_meta.json"), "w") as f:
        json.dump(jsonable({**{k: v for k, v in meta.items()
                               if k != "files"},
                            "output_dir": out, "platform": platform.platform(),
                            "input_sha256": hashes,
                            "chosen_test": rule["chosen"],
                            "rule": {k: v for k, v in rule.items()},
                            "mixed_verdict": mixed_verdict,
                            "checks_pass": len(ck.rows) - ck.n_fail,
                            "checks_fail": ck.n_fail}), f, indent=2)
    log(f"done: {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL; "
        f"primary test: {rule['chosen']}")
    log(f"report: {os.path.join(out, 'report.txt')}")


if __name__ == "__main__":
    main()
