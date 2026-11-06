"""
analyse_pair_capacity_v5.py  --  v5 step 1: can the stimuli supply two matched sets?

WHAT THIS ANSWERS
    The v5 design gives every participant two image-choice blocks of equal
    length. Each block has its own set of pairs, and which set goes with
    which block is counterbalanced. The paired comparison of per-person
    weights between the blocks is only fair if the two sets carry the same
    information for each weight. So, for each block size, type mix and
    colour-level set in SETTINGS, this script asks whether two sets exist
    that:

      * share no pair (a pair with its sides swapped is the same pair);
      * have identical trial-type counts and identical magnitude-bin counts;
      * have matched |x_sat| and |x_con| distributions, i.e. about equal
        information (sum of squared predictors) for each weight;
      * use the same scenes (by default every trial in one set has a
        partner in the other from the same scene and the same bin);
      * balance sides, and spread over the scenes;
      * leave enough unused pairs for each block's practice and catch
        trials.

    It is run once WITH and once WITHOUT the saturation level p65, and
    applies the decision rule: if two matched sets are feasible without
    p65, p65 is dropped.

HOW FEASIBILITY IS DECIDED (two stages)
    1. Counts, exactly. For each trial type an integer program allocates
       pairs over scene x magnitude bin x block, under the per-scene cap,
       the bin quotas and the no-shared-pair rule. "Infeasible" here is a
       proof, not a failed search. When it is infeasible the same program
       maximises what can be placed, which locates the shortfall by type
       and bin, and it is re-solved at higher caps to find the smallest cap
       that would work.
    2. Matching, on concrete draws. N_DRAWS pairs of sets are drawn (v4's
       greedy draw, extended to two blocks: a random pair goes to one
       block, and its nearest unused partner in |x_sat|, |x_con| goes to the
       other). Anchors come from the least-used scenes first, and once a
       type is drawn, which member of each couple goes to which set is
       chosen to equalise the information per weight (SPREAD_SCENES,
       BALANCE_ASSIGNMENT). Each draw is checked against the tolerances in
       SETTINGS.
    A combination is FEASIBLE when stage 1 succeeds, every draw could be
    built, and at least REQUIRED_PASS_SHARE of draws pass every check.

REUSED FROM THE v4 SCRIPTS (copied, not imported, so this file is standalone
and a later edit to a v4 script cannot change it)
    analyse_pairs_v4_corrected.py
        classify_pair()            -> classify_pair_v4(), used only to check
                                      that every pool row's type re-derives
                                      from its measured features
        SD_SAT, SD_FOG, FOG_SIGN and the type thresholds
    build_trials_v4_corrected.py
        pair_key(), keys_of()      -> unchanged
        BIN_EDGES_SD               -> unchanged
        apply_colour_cut()         -> folded into build_pool()
        assign_bins()              -> usable_bins(): same fixed edges, same
                                      "too few pairs" and "too few scenes for
                                      its share" rules, counted on unordered
                                      pairs and per block
        stratified_sample()        -> greedy_type(): same random bin order,
                                      sort-then-shuffle, cap counted across
                                      the whole draw, extended to two blocks
        Streams (per-type seeded generators, SHA-256 of the name)
                                   -> Streams, keyed by combination and draw
        _balanced_sides()          -> unchanged
        resolve_output_root()      -> resolve_output_dir(): never overwrites,
                                      and the folder is named after the key
                                      settings

INPUTS (paths relative to the project folder; run from there)
    PAIRS_CSV       data/pair_analysis_v4_ext/extended/all_pairs.csv   required
    MANIFEST_CSV    data/stimulus_set_v4_ext_haze_eq/stimulus_manifest.csv
                    optional: cross-checks features, `use` and the unedited
                    images. None or a missing file skips those checks.
    V4_TRIAL_LIST   data/frozen_trial_list_v4/trial_list.csv
                    optional: the information baseline from the frozen v4
                    list.

OUTPUTS (one new folder per run, never overwritten)
    capacity_report.txt       what was done, what was found, every check
    summary.csv               one row per level set x size x mix
    feasibility_by_type.csv   ILP result per type: placed, shortfall, min cap
    shortfall_by_bin.csv      where an infeasible combination falls short
    pool_by_bin.csv           unordered supply per level set, type and bin
    draw_checks.csv           every check on every draw
    checks.csv                every check in the report, PASS or FAIL
    example_sets/             the first draw of each feasible combination
    settings.json             every setting this run used

REQUIRES  numpy, pandas, scipy >= 1.9 (scipy.optimize.milp)
RUN       python analyse_pair_capacity_v5.py
"""

# ============================================================
# SETTINGS
# ============================================================

# --- Input / output ---
PAIRS_CSV = "data/pair_analysis_v4_ext/extended/all_pairs.csv"
MANIFEST_CSV = "data/stimulus_set_v4_ext_haze_eq/stimulus_manifest.csv"
V4_TRIAL_LIST = "data/frozen_trial_list_v4/trial_list.csv"

# OUTPUT_ROOT
#   Each run writes to OUTPUT_ROOT/<key settings>__s<hash>. The hash covers
#   every setting, so two runs that differ only in a setting not named in
#   the folder still get different folders. If that folder exists anyway,
#   the run writes to <folder>_2, _3, ... (ON_EXISTING = "new") or stops
#   (ON_EXISTING = "error").
OUTPUT_ROOT = "data/pair_capacity_v5"
ON_EXISTING = "new"

# --- The design questions ---
#
# BLOCK_SIZES
#   Analysed trials PER BLOCK. Practice, catch and consistency trials come
#   on top (practice and catch are reserved below; consistency repeats
#   reuse a block's own pairs and cost no capacity).
# BLOCK_SIZES = [90, 105, 120, 150]
BLOCK_SIZES = [120, 150, 180]

# TYPE_MIXES
#   Share of each block's analysed trials per type. Counts are rounded by
#   largest remainder (ties broken in TYPES order), so they always sum to
#   the block size; the realised counts are in the report.
# TYPE_MIXES = {
#     "v4":             {"sat_only": 1 / 3, "fog_only": 1 / 3,
#                        "together": 1 / 6, "conflict": 1 / 6},
#     "conflict_heavy": {"sat_only": 0.30, "fog_only": 0.30,
#                        "together": 0.15, "conflict": 0.25},
#     "equal":          {"sat_only": 0.25, "fog_only": 0.25,
#                        "together": 0.25, "conflict": 0.25},
# }
TYPE_MIXES = {
    "v4":             {"sat_only": 1 / 3, "fog_only": 1 / 3,
                       "together": 1 / 6, "conflict": 1 / 6},
    "conflict_heavy": {"sat_only": 0.30, "fog_only": 0.30,
                       "together": 0.15, "conflict": 0.25},
} 

# LEVEL_SETS
#   Which rendered levels may appear in a pair. A pair is dropped when
#   either image carries a dropped level. Colour labels as in all_pairs.csv:
#   p75, p65, k=1, p25, p10, p05, p01. Fog labels: p35, p25, p15, p10, p05.
# LEVEL_SETS = {
#     "with_p65":    {"drop_colour": [], "drop_fog": []},
#     "without_p65": {"drop_colour": ["p65"], "drop_fog": []},
# }

LEVEL_SETS = {
    "without_p65": {"drop_colour": ["p65"], "drop_fog": []},
}                                   # delete the "with_p65" entry


# P65_DECISION
#   (level set with p65, level set without p65). Decision rule: for a given
#   size and mix, if the set without p65 is feasible, p65 is dropped.
#   None skips the decision section.
# P65_DECISION = ("with_p65", "without_p65")
P65_DECISION = None   

# DISJOINT
#   What the two sets may not share.
#     "pair"   no pair in both sets (sides swapped = same pair). DEFAULT.
#     "image"  no image file in both sets. Stage 1 then solves the "pair"
#              problem, which is necessary but not sufficient, and the
#              draws decide.
#     "scene"  no scene in both sets; each scene goes to one set. Stage 1
#              solves the scene split exactly, jointly over the four types.
#              Requires SCENE_MATCH = "bin".
DISJOINT = "pair"

# SCENE_MATCH
#   How the two sets are tied together.
#     "scene_bin"  every trial in one set has a partner in the other from
#                  the SAME SCENE and the SAME MAGNITUDE BIN, the nearest
#                  unused pair in (|x_sat|, |x_con|). Each scene then gives
#                  each block the same number of trials of each type, so
#                  scene content is equal across blocks. DEFAULT.
#     "bin"        the partner comes from the same bin, any scene.
SCENE_MATCH = "scene_bin"

# MAX_PER_SCENE_PER_TYPE
#   Cap on trials of one type from one scene, WITHIN EACH BLOCK. v4 used 2
#   over its 60-trial types. Not scaled with block size.
MAX_PER_SCENE_PER_TYPE = 2

# MAX_CAP_SEARCH
#   When a combination is infeasible, stage 1 is re-solved at caps
#   MAX_PER_SCENE_PER_TYPE + 1 ... + MAX_CAP_SEARCH, to report the smallest
#   cap that would make it feasible.
MAX_CAP_SEARCH = 4

# --- Pair rules (the v4 rules, plus the new contrast rule) ---
#
# SAT_ONLY_MIN_DSAT
#   Saturation-only pairs need |d_sat| >= this (sat_mean units). v4's 31.1.
SAT_ONLY_MIN_DSAT = 31.1

# SAT_ONLY_CON_TOL
#   NEW in v5: saturation-only pairs must match in measured local contrast
#   within this many contrast_local units. 0.72 units = 0.077 corpus SD, the
#   same SD fraction as the 3-unit saturation rule for contrast-only pairs
#   (3 / 39.182 = 0.077; 0.077 x 9.348 = 0.72). None disables the rule.
SAT_ONLY_CON_TOL = 0.72

# FOG_ONLY_SAT_TOL
#   Contrast-only pairs must match in sat_mean within this. v4's 3.0; the
#   pool already enforces it, so this re-checks rather than filters.
FOG_ONLY_SAT_TOL = 3.0

# BALANCED_CONFLICT_ONLY
#   Conflict trials only from pairs flagged `balanced` by v4 step 4.
BALANCED_CONFLICT_ONLY = True

# --- Magnitude bins (v4's fixed edges, corpus-SD units of the driver) ---
#   Driver: |x_sat| for sat_only, |x_con| for the other types.
BIN_EDGES_SD = {
    "sat_only": (0.58, 0.79, 1.10, 1.80),
    "fog_only": (0.5894, 0.8259),
    "together": (0.5517, 0.8362),
    "conflict": (0.4506, 0.5714),
}

# MIN_PAIRS_PER_BIN
#   A bin with fewer UNORDERED pairs than this is treated as empty, and the
#   type's trials are spread evenly over the remaining bins (v4's rule).
#   v4 counted ordered pairs and used 20, which is 10 unordered.
MIN_PAIRS_PER_BIN = 10

# --- Reserved trials per block (drawn from pairs neither set uses) ---
#   Reserved so the build step cannot run out. The counts here only
#   reserve capacity; the build step sets the real counts.
N_PRACTICE_PER_BLOCK = 6
N_CATCH_PER_BLOCK = 10
PRACTICE_TYPES = ("sat_only", "fog_only", "together")
PRACTICE_MAG_QUANTILE = 0.60
CATCH_TYPE = "together"

# --- How the draws build matched sets ---
#
# SPREAD_SCENES
#   True: each next anchor pair comes from the scenes used least so far (ties
#   random), so the sets spread over the scenes. False: v4's purely random
#   order, which at 90 trials per set leaves about ten scenes unused.
SPREAD_SCENES = True

# BALANCE_ASSIGNMENT
#   True: once a type's matched couples are drawn, decide which member of
#   each couple goes to set A so that the information per weight is as equal
#   as possible (no pair enters or leaves; only its set changes). Applied only
#   when DISJOINT = "pair". False: the side of each couple is left as drawn.
BALANCE_ASSIGNMENT = True

# TYPE_TRIES
#   Each type is drawn this many times within a draw (different random
#   orders), and the attempt whose couples balance the information best is
#   kept. Matters most for the small types: 15-17 conflict couples cannot
#   always equalise both weights from one attempt. 1 = v4's single draw.
TYPE_TRIES = 5

# --- Draws and matching checks ---
N_DRAWS = 20
RANDOM_SEED = 42

# REQUIRED_PASS_SHARE
#   A combination is FEASIBLE only if at least this share of its draws pass
#   every check.
REQUIRED_PASS_SHARE = 0.90

# INFO_TOL_TOTAL
#   Max relative difference between the two sets in information (sum of
#   squared predictors) for each weight, over all analysed trials:
#   |I_A - I_B| / mean(I_A, I_B).
INFO_TOL_TOTAL = 0.05

# INFO_TOL_TYPE
#   The same, within each type, for the weights that type informs:
#   sat_only -> x_sat; fog_only -> x_con; together and conflict -> both.
INFO_TOL_TYPE = 0.10

# KS_MAX_D
#   Max two-sample Kolmogorov-Smirnov distance between the sets, for |x_sat|
#   over the trials that inform x_sat (sat_only, together, conflict) and for
#   |x_con| over those that inform x_con (fog_only, together, conflict).
#   The per-type KS of each type's driver is in draw_checks.csv.
KS_MAX_D = 0.10

# COS_DIFF_MAX / VIF_MAX
#   The predictor correlation that matters for a no-intercept model is the
#   uncentred one, cos = sum(x_sat * x_con) / sqrt(sum x_sat^2 sum x_con^2),
#   which does not depend on which image is A. The two sets must agree on it
#   within COS_DIFF_MAX, and each set's VIF = 1 / (1 - cos^2) must not
#   exceed VIF_MAX. The ordinary Pearson r on the presented sides is
#   reported too.
COS_DIFF_MAX = 0.05
VIF_MAX = 1.25

# MIN_SCENE_SHARE
#   Each set's analysed trials must come from at least this share of the
#   scenes that set may use: all scenes, or under DISJOINT = "scene" the
#   scenes assigned to it. 0.85 of 47 = 40 scenes.
MIN_SCENE_SHARE = 0.85

# TYPE_ORDER
#   Order in which types are drawn. Matters only when DISJOINT = "image",
#   where an image placed in one set is barred from the other for every
#   later type; scarcest first.
TYPE_ORDER = ("sat_only", "fog_only", "conflict", "together")

# --- Input checks ---
#
# EXPECTED_POOL
#   Row counts of the v4 extended all_pairs.csv (design document, section
#   6). A mismatch means PAIRS_CSV is not the file this design assumes.
#   None skips the check.
EXPECTED_POOL = {"total": 34366, "sat_only": 6282, "fog_only": 2136,
                 "together": 17666, "conflict": 8282,
                 "balanced_conflict": 4300}

# The v4 step-4 thresholds, copied from analyse_pairs_v4_corrected.py, used
# only to check that every pool row's type re-derives from its features.
V4_MIN_DSAT_SAT_ONLY = 17.0
V4_PAIR_SAT_TOL = 3.0
V4_MIN_DFOG_FOG_ONLY = 4.2
V4_MIN_DSAT_MIXED = 8.0
V4_MIN_DFOG_MIXED = 3.0
V4_BALANCE_LO = 0.6
V4_BALANCE_HI = 1.7

# Labels of the unedited photograph's levels (manifest / pair table).
ORIGINAL_COLOUR = "k=1"
ORIGINAL_FOG = "h=0"

# ILP_TIME_LIMIT
#   Seconds per integer program. The problems are small (hundreds of
#   variables) and solve in well under a second; the limit is a guard.
ILP_TIME_LIMIT = 60

# WRITE_EXAMPLE_SETS
#   Write draw 0 of every combination whose counts are feasible.
WRITE_EXAMPLE_SETS = True

# --- Constants (v4; do not change) ---
SD_SAT = 39.182
SD_FOG = 9.348
FOG_SIGN = -1.0
TYPES = ["sat_only", "fog_only", "together", "conflict"]


# ============================================================
# Imports
# ============================================================

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse, stats
try:
    from scipy.optimize import Bounds, LinearConstraint, milp
except ImportError as e:            # scipy < 1.9
    raise ImportError("This script needs scipy >= 1.9 for "
                      "scipy.optimize.milp.") from e


# ============================================================
# Small helpers
# ============================================================

def settings_dict():
    return {
        "PAIRS_CSV": PAIRS_CSV, "MANIFEST_CSV": MANIFEST_CSV,
        "V4_TRIAL_LIST": V4_TRIAL_LIST, "OUTPUT_ROOT": OUTPUT_ROOT,
        "BLOCK_SIZES": BLOCK_SIZES, "TYPE_MIXES": TYPE_MIXES,
        "LEVEL_SETS": LEVEL_SETS, "P65_DECISION": P65_DECISION,
        "DISJOINT": DISJOINT, "SCENE_MATCH": SCENE_MATCH,
        "MAX_PER_SCENE_PER_TYPE": MAX_PER_SCENE_PER_TYPE,
        "MAX_CAP_SEARCH": MAX_CAP_SEARCH,
        "SAT_ONLY_MIN_DSAT": SAT_ONLY_MIN_DSAT,
        "SAT_ONLY_CON_TOL": SAT_ONLY_CON_TOL,
        "FOG_ONLY_SAT_TOL": FOG_ONLY_SAT_TOL,
        "BALANCED_CONFLICT_ONLY": BALANCED_CONFLICT_ONLY,
        "BIN_EDGES_SD": BIN_EDGES_SD, "MIN_PAIRS_PER_BIN": MIN_PAIRS_PER_BIN,
        "N_PRACTICE_PER_BLOCK": N_PRACTICE_PER_BLOCK,
        "N_CATCH_PER_BLOCK": N_CATCH_PER_BLOCK,
        "PRACTICE_TYPES": PRACTICE_TYPES,
        "PRACTICE_MAG_QUANTILE": PRACTICE_MAG_QUANTILE,
        "CATCH_TYPE": CATCH_TYPE, "N_DRAWS": N_DRAWS,
        "RANDOM_SEED": RANDOM_SEED,
        "REQUIRED_PASS_SHARE": REQUIRED_PASS_SHARE,
        "INFO_TOL_TOTAL": INFO_TOL_TOTAL, "INFO_TOL_TYPE": INFO_TOL_TYPE,
        "KS_MAX_D": KS_MAX_D, "COS_DIFF_MAX": COS_DIFF_MAX,
        "VIF_MAX": VIF_MAX, "MIN_SCENE_SHARE": MIN_SCENE_SHARE,
        "TYPE_ORDER": TYPE_ORDER, "EXPECTED_POOL": EXPECTED_POOL,
        "SPREAD_SCENES": SPREAD_SCENES,
        "BALANCE_ASSIGNMENT": BALANCE_ASSIGNMENT, "TYPE_TRIES": TYPE_TRIES,
        "ILP_TIME_LIMIT": ILP_TIME_LIMIT,
        "SD_SAT": SD_SAT, "SD_FOG": SD_FOG,
    }


def check_settings():
    if DISJOINT not in ("pair", "image", "scene"):
        raise ValueError(f"DISJOINT must be pair, image or scene, "
                         f"got {DISJOINT!r}")
    if SCENE_MATCH not in ("scene_bin", "bin"):
        raise ValueError(f"SCENE_MATCH must be scene_bin or bin, "
                         f"got {SCENE_MATCH!r}")
    if DISJOINT == "scene" and SCENE_MATCH != "bin":
        raise ValueError("DISJOINT = 'scene' puts each scene in one set, so "
                         "a partner cannot come from the same scene. Set "
                         "SCENE_MATCH = 'bin'.")
    for name, mix in TYPE_MIXES.items():
        if set(mix) != set(TYPES):
            raise ValueError(f"TYPE_MIXES[{name!r}] must name exactly "
                             f"{TYPES}")
        if abs(sum(mix.values()) - 1) > 1e-9:
            raise ValueError(f"TYPE_MIXES[{name!r}] sums to "
                             f"{sum(mix.values())}, not 1")
    if P65_DECISION is not None:
        for s in P65_DECISION:
            if s not in LEVEL_SETS:
                raise ValueError(f"P65_DECISION names {s!r}, which is not "
                                 "in LEVEL_SETS")
    if set(TYPE_ORDER) != set(TYPES):
        raise ValueError(f"TYPE_ORDER must list exactly {TYPES}")


def fmt_num(x):
    return ("none" if x is None else
            (f"{x:g}" if isinstance(x, float) else str(x)))


def resolve_output_dir():
    """OUTPUT_ROOT/<key settings>__s<hash>, never overwriting a run.

    Adapted from resolve_output_root() in the v4 scripts."""
    blob = json.dumps(settings_dict(), sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode()).hexdigest()[:8]
    name = (f"disjoint-{DISJOINT}_match-{SCENE_MATCH}"
            f"_cap{MAX_PER_SCENE_PER_TYPE}"
            f"_satcut{fmt_num(SAT_ONLY_MIN_DSAT)}"
            f"_contol{fmt_num(SAT_ONLY_CON_TOL)}"
            f"_sizes{'-'.join(str(s) for s in BLOCK_SIZES)}"
            f"_levels-{'-'.join(LEVEL_SETS)}"
            f"_draws{N_DRAWS}_seed{RANDOM_SEED}__s{h}")
    base = os.path.join(OUTPUT_ROOT, name)

    def occupied(d):
        return os.path.exists(d)

    if not occupied(base):
        return base
    if ON_EXISTING == "error":
        raise FileExistsError(f"{base} exists. Move it, or set "
                              "ON_EXISTING = 'new'.")
    n = 2
    while occupied(f"{base}_{n}"):
        n += 1
    print(f"  NOTE: {base} exists; writing to {base}_{n} instead.")
    return f"{base}_{n}"


class Streams:
    """Named random generators, after build_trials_v4_corrected.Streams.

    Each (combination, draw, name) gets its own generator, seeded from the
    SHA-256 of the key rather than Python's salted hash(), so a run is
    reproducible and one draw cannot move another."""

    def __init__(self, seed, *key):
        self.seed = int(seed)
        self.key = "|".join(str(k) for k in key)
        self._made = {}

    def __call__(self, name):
        if name not in self._made:
            tag = int.from_bytes(hashlib.sha256(
                f"{self.key}|{name}".encode()).digest()[:8], "big")
            self._made[name] = np.random.default_rng([self.seed, tag])
        return self._made[name]


def pair_key(a, b):
    """Identify a pair by its IMAGES, ignoring which is A and which B.
    Copied unchanged from build_trials_v4_corrected.py."""
    return (a, b) if str(a) <= str(b) else (b, a)


def keys_of(df):
    """Copied unchanged from build_trials_v4_corrected.py."""
    return [pair_key(a, b) for a, b in zip(df.file_a, df.file_b)]


def balanced_sides(n, rng):
    """Exactly half True, at random; an odd extra goes either way at random.
    _balanced_sides() from build_trials_v4_corrected.py, for one group."""
    out = np.zeros(n, dtype=bool)
    idx = rng.permutation(n)
    half = n // 2
    if n % 2 and rng.random() < 0.5:
        half += 1
    out[idx[:half]] = True
    return out


def largest_remainder(total, mix):
    raw = {t: total * mix[t] for t in TYPES}
    base = {t: int(np.floor(raw[t] + 1e-9)) for t in TYPES}
    left = total - sum(base.values())
    order = sorted(TYPES, key=lambda t: (-(raw[t] - base[t]),
                                         TYPES.index(t)))
    for t in order[:left]:
        base[t] += 1
    return base


# EPS
#   Margin on every "<= threshold" comparison, so a value that equals a
#   threshold in exact arithmetic (a KS distance of exactly 0.1 is 9 of 90
#   trials) passes on every computer, whatever the last bit of rounding.
EPS = 1e-9


def rel_diff(a, b):
    m = (a + b) / 2
    return abs(a - b) / m if m > 0 else 0.0


class Checks:
    """Every check in the run, each PASS or FAIL (or SKIPPED with a reason,
    when its input is not there)."""

    def __init__(self):
        self.rows = []

    def add(self, scope, name, ok, detail=""):
        status = ok if isinstance(ok, str) else ("PASS" if ok else "FAIL")
        self.rows.append({"scope": scope, "check": name, "status": status,
                          "detail": detail})
        return status == "PASS"

    def frame(self):
        return pd.DataFrame(self.rows)


# ============================================================
# The v4 pair classification (copied from analyse_pairs_v4_corrected.py)
# ============================================================

def classify_pair_v4(sat_a, fog_a, sat_b, fog_b, same_row):
    """Type and corpus-SD balance of one ordered pair, exactly as v4 step 4
    classifies it (classify_pair, BALANCE_NORM = "corpus_sd"). Returns
    (type, balance) or None for an unusable pair."""
    d_sat = sat_a - sat_b
    d_fog = FOG_SIGN * (fog_a - fog_b)
    if same_row:
        if abs(d_sat) < V4_MIN_DSAT_SAT_ONLY:
            return None
        t = "sat_only"
    elif abs(d_sat) <= V4_PAIR_SAT_TOL and abs(d_fog) >= V4_MIN_DFOG_FOG_ONLY:
        t = "fog_only"
    elif abs(d_fog) >= V4_MIN_DFOG_MIXED and abs(d_sat) >= V4_MIN_DSAT_MIXED:
        t = "conflict" if ((d_fog > 0) == (d_sat > 0)) else "together"
    else:
        return None
    zc_sat, zc_fog = abs(d_sat) / SD_SAT, abs(d_fog) / SD_FOG
    bal = zc_sat / zc_fog if zc_fog > 0 else np.nan
    return t, bal


# ============================================================
# Loading and input checks
# ============================================================

REQUIRED_COLS = ["scene_id", "type", "colour_a", "fog_a_level", "colour_b",
                 "fog_b_level", "file_a", "file_b", "sat_a", "sat_b",
                 "fog_a", "fog_b", "d_sat", "d_fog", "pred_sat", "pred_fog",
                 "balanced"]


def as_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def load_pairs(ck, log):
    if not os.path.isfile(PAIRS_CSV):
        raise FileNotFoundError(
            f"Pair table not found: {PAIRS_CSV}\nRun this script from the "
            "project folder, or set PAIRS_CSV.")
    p = pd.read_csv(PAIRS_CSV)
    missing = [c for c in REQUIRED_COLS if c not in p.columns]
    ck.add("input", "pair table has the required columns", not missing,
           f"missing {missing}" if missing else f"{len(p.columns)} columns")
    if missing:
        raise ValueError(f"{PAIRS_CSV} is missing columns {missing}")
    p["balanced"] = as_bool(p["balanced"])
    log.append(f"pair table      {PAIRS_CSV}: {len(p):,} ordered pairs, "
               f"{p.scene_id.nunique()} scenes")

    # 1. counts match the v4 extended inventory
    counts = {"total": len(p)}
    counts.update(p.type.value_counts().to_dict())
    counts["balanced_conflict"] = int(((p.type == "conflict")
                                       & p.balanced).sum())
    if EXPECTED_POOL:
        bad = {k: (counts.get(k, 0), v) for k, v in EXPECTED_POOL.items()
               if counts.get(k, 0) != v}
        ck.add("input", "pool counts match the v4 extended inventory",
               not bad,
               ("all of " + ", ".join(f"{k} {v:,}" for k, v in
                                      EXPECTED_POOL.items()))
               if not bad else f"got vs expected: {bad}")
    else:
        ck.add("input", "pool counts match the v4 extended inventory",
               "SKIPPED", "EXPECTED_POOL is None")

    # 2. every ordered pair has its reverse, with negated differences
    fwd = p.set_index(["file_a", "file_b"])[["d_sat", "d_fog", "type"]]
    rev = p.set_index(["file_b", "file_a"])[["d_sat", "d_fog", "type"]]
    rev.index.names = ["file_a", "file_b"]
    j = fwd.join(rev, rsuffix="_r", how="left")
    ok = (j.d_sat_r.notna().all()
          and np.allclose(j.d_sat, -j.d_sat_r, atol=1e-6)
          and np.allclose(j.d_fog, -j.d_fog_r, atol=1e-6)
          and (j.type == j.type_r).all())
    ck.add("input", "every pair is listed in both directions, with negated "
           "differences and the same type", bool(ok),
           f"{int(j.d_sat_r.isna().sum())} without a reverse")

    # 3. differences and oriented predictors agree with the features
    ok_d = (np.allclose(p.d_sat, p.sat_a - p.sat_b, atol=1e-4)
            and np.allclose(p.d_fog, FOG_SIGN * (p.fog_a - p.fog_b),
                            atol=1e-4))
    ok_x = (np.allclose(p.pred_sat, -p.d_sat / SD_SAT, atol=1e-6)
            and np.allclose(p.pred_fog, p.d_fog / SD_FOG, atol=1e-6))
    ck.add("input", "d_sat, d_fog re-derive from the measured features",
           bool(ok_d))
    ck.add("input", "pred_sat = -d_sat/SD_SAT and pred_fog = d_fog/SD_FOG",
           bool(ok_x), f"SD_SAT {SD_SAT}, SD_FOG {SD_FOG}")

    # 4. types and the balanced flag re-derive with v4's classify_pair
    n_bad_t = n_bad_b = 0
    for r in p.itertuples(index=False):
        got = classify_pair_v4(r.sat_a, r.fog_a, r.sat_b, r.fog_b,
                               r.fog_a_level == r.fog_b_level)
        if got is None or got[0] != r.type:
            n_bad_t += 1
            continue
        if r.type == "conflict":
            bal = (np.isfinite(got[1])
                   and V4_BALANCE_LO <= got[1] <= V4_BALANCE_HI)
            n_bad_b += int(bal != bool(r.balanced))
    ck.add("input", "every type re-derives with v4 classify_pair",
           n_bad_t == 0, f"{n_bad_t} of {len(p):,} rows disagree")
    ck.add("input", "every conflict `balanced` flag re-derives "
           f"(corpus-SD balance in [{V4_BALANCE_LO}, {V4_BALANCE_HI}])",
           n_bad_b == 0, f"{n_bad_b} disagree")

    # 5. the unedited photograph never appears
    lv = pd.concat([p[["colour_a", "fog_a_level"]].set_axis(["c", "f"], axis=1),
                    p[["colour_b", "fog_b_level"]].set_axis(["c", "f"], axis=1)])
    n_unedited = int(((lv.c == ORIGINAL_COLOUR) & (lv.f == ORIGINAL_FOG)).sum())
    n_h0 = int((lv.f == ORIGINAL_FOG).sum())
    ck.add("input", "the unedited photograph (colour k=1, fog h=0) is in no "
           "pair", n_unedited == 0,
           f"{n_unedited} image slots; images at h=0 of any colour: {n_h0}")
    return p


def check_manifest(p, ck, log):
    if MANIFEST_CSV is None or not os.path.isfile(MANIFEST_CSV):
        why = ("MANIFEST_CSV is None" if MANIFEST_CSV is None
               else f"{MANIFEST_CSV} not found")
        for name in ("every pool image is in the manifest with use pair/both",
                     "pool features equal the manifest's measured features",
                     "every unedited image is grid-only"):
            ck.add("manifest", name, "SKIPPED", why)
        log.append(f"manifest        not read ({why})")
        return
    m = pd.read_csv(MANIFEST_CSV)
    log.append(f"manifest        {MANIFEST_CSV}: {len(m):,} images, "
               f"{m.scene_id.nunique()} scenes")
    need = {"filename", "sat_mean", "contrast_local"}
    if not need <= set(m.columns):
        for name in ("every pool image is in the manifest with use pair/both",
                     "pool features equal the manifest's measured features",
                     "every unedited image is grid-only"):
            ck.add("manifest", name, "SKIPPED",
                   f"manifest lacks {sorted(need - set(m.columns))}")
        return
    mi = m.set_index("filename")
    imgs = pd.concat([
        p[["file_a", "sat_a", "fog_a"]].set_axis(["f", "s", "c"], axis=1),
        p[["file_b", "sat_b", "fog_b"]].set_axis(["f", "s", "c"], axis=1),
    ]).drop_duplicates("f").set_index("f")
    missing = imgs.index.difference(mi.index)
    use_ok = True
    detail = f"{len(imgs):,} distinct images, {len(missing)} not in manifest"
    if "use" in mi.columns and not len(missing):
        bad_use = (~mi.loc[imgs.index, "use"].isin(["pair", "both"])).sum()
        use_ok = bad_use == 0
        detail += f", {int(bad_use)} with another `use`"
    ck.add("manifest", "every pool image is in the manifest with use "
           "pair/both", len(missing) == 0 and use_ok, detail)
    if len(missing) == 0:
        ds = (imgs.s - mi.loc[imgs.index, "sat_mean"]).abs().max()
        dc = (imgs.c - mi.loc[imgs.index, "contrast_local"]).abs().max()
        ck.add("manifest", "pool features equal the manifest's measured "
               "features", ds < 1e-4 and dc < 1e-4,
               f"max |diff| sat_mean {ds:.2e}, contrast_local {dc:.2e}")
    else:
        ck.add("manifest", "pool features equal the manifest's measured "
               "features", "SKIPPED", "images missing from the manifest")
    if {"colour_level", "fog_level"} <= set(m.columns):
        une = m[(m.colour_level == ORIGINAL_COLOUR)
                & (m.fog_level == ORIGINAL_FOG)]
        in_pool = une.filename.isin(imgs.index).sum()
        grid_only = ("use" not in m.columns
                     or (une.use == "grid").all())
        ck.add("manifest", "every unedited image is grid-only and in no "
               "pair", len(une) > 0 and in_pool == 0 and grid_only,
               f"{len(une)} unedited images in the manifest, {in_pool} "
               "in the pool")
    else:
        ck.add("manifest", "every unedited image is grid-only",
               "SKIPPED", "no colour_level / fog_level columns")


def load_v4_baseline(log):
    if V4_TRIAL_LIST is None or not os.path.isfile(V4_TRIAL_LIST):
        log.append(f"v4 baseline     not read ({V4_TRIAL_LIST} not found); "
                   "information is reported without it")
        return None
    t = pd.read_csv(V4_TRIAL_LIST)
    a = t[as_bool(t["analysed"])]
    xs, xc = a.pred_sat.to_numpy(), a.pred_fog.to_numpy()
    cos = float((xs * xc).sum() / np.sqrt((xs ** 2).sum() * (xc ** 2).sum()))
    b = {"n": len(a), "info_sat": float((xs ** 2).sum()),
         "info_con": float((xc ** 2).sum()), "cos": cos,
         "counts": a.trial_type.value_counts().to_dict()}
    log.append(f"v4 baseline     {V4_TRIAL_LIST}: {b['n']} analysed trials, "
               f"info x_sat {b['info_sat']:.1f}, x_con {b['info_con']:.1f}, "
               f"cos {cos:+.3f}")
    return b


# ============================================================
# Pools
# ============================================================

def build_pool(p, level_set, ck, funnel):
    """Unordered, rule-filtered pool for one level set.

    One row per pair, oriented so that A is the image the hypothesis
    predicts by the type's driver (x_sat > 0 for sat_only, x_con > 0 for
    the rest), with x_sat = pred_sat and x_con = pred_fog on that row."""
    spec = LEVEL_SETS[level_set]
    dc, df_ = set(spec["drop_colour"]), set(spec["drop_fog"])
    q = p[~p.colour_a.isin(dc) & ~p.colour_b.isin(dc)
          & ~p.fog_a_level.isin(df_) & ~p.fog_b_level.isin(df_)]
    driver_pos = np.where(q.type == "sat_only", q.pred_sat > 0,
                          q.pred_fog > 0)
    q = q[driver_pos].copy()
    keep = []
    for t in TYPES:
        steps = []
        s = q[q.type == t]
        steps.append(("1 after level filter", len(s)))
        if t == "sat_only":
            s = s[s.d_sat.abs() >= SAT_ONLY_MIN_DSAT]
            steps.append((f"2 |d_sat| >= {SAT_ONLY_MIN_DSAT}", len(s)))
            if SAT_ONLY_CON_TOL is not None:
                s = s[s.d_fog.abs() <= SAT_ONLY_CON_TOL]
                steps.append((f"3 |d_con| <= {SAT_ONLY_CON_TOL}", len(s)))
        if t == "fog_only":
            s = s[s.d_sat.abs() <= FOG_ONLY_SAT_TOL]
            steps.append((f"2 |d_sat| <= {FOG_ONLY_SAT_TOL}", len(s)))
        if t == "conflict" and BALANCED_CONFLICT_ONLY:
            s = s[s.balanced]
            steps.append(("2 balanced", len(s)))
        steps.append(("9 scenes", s.scene_id.nunique()))
        for step, n in steps:
            funnel.append({"level_set": level_set, "type": t, "step": step,
                           "unordered_pairs": n})
        keep.append(s)
    pool = pd.concat(keep, ignore_index=True)
    pool["x_sat"] = pool.pred_sat
    pool["x_con"] = pool.pred_fog
    drv = np.where(pool.type == "sat_only", pool.x_sat.abs(),
                   pool.x_con.abs())
    pool["driver_sd"] = drv
    pool["mag_bin"] = 0
    for t in TYPES:
        m = (pool.type == t).to_numpy()
        pool.loc[m, "mag_bin"] = np.digitize(drv[m], BIN_EDGES_SD[t]) + 1
    pool["key"] = keys_of(pool)
    pool = pool.sort_values(["type", "scene_id", "file_a", "file_b"],
                            kind="stable").reset_index(drop=True)

    # checks on the pool
    dup = pool.key.duplicated().sum()
    ck.add(level_set, "each unordered pair appears once in the pool",
           dup == 0, f"{len(pool):,} pairs, {dup} duplicates")
    lv = set(pool.colour_a) | set(pool.colour_b)
    fl = set(pool.fog_a_level) | set(pool.fog_b_level)
    ck.add(level_set, "no dropped level in the pool",
           not (lv & dc) and not (fl & df_),
           f"colour levels {sorted(lv)}; fog levels {sorted(fl)}")
    return pool


# ============================================================
# Bins and quotas
# ============================================================

def usable_bins(tp, n_trials, cap):
    """v4's assign_bins() rules, per block: a bin is usable if it holds at
    least MIN_PAIRS_PER_BIN unordered pairs AND its eligible scenes can
    supply its share under the cap; dropping a bin raises the others'
    shares, so the second rule is iterated. Under scene_bin matching a
    scene is eligible only with >= 2 pairs in the bin (one per block)."""
    n_bins = len(BIN_EDGES_SD[tp.type.iloc[0]]) + 1 if len(tp) else 0
    counts = {b: int((tp.mag_bin == b).sum()) for b in range(1, n_bins + 1)}
    need = 2 if SCENE_MATCH == "scene_bin" else 1
    per_scene = tp.groupby(["mag_bin", "scene_id"]).size()
    elig = {b: int((per_scene.loc[b] >= need).sum())
            if b in per_scene.index.get_level_values(0) else 0
            for b in counts}
    usable = [b for b, c in counts.items() if c >= MIN_PAIRS_PER_BIN]
    starved = []
    while usable:
        share = int(np.ceil(n_trials / len(usable)))
        short = [b for b in usable if elig[b] * cap < share]
        if not short:
            break
        starved += short
        usable = [b for b in usable if b not in short]
    dropped_thin = [b for b, c in counts.items()
                    if c < MIN_PAIRS_PER_BIN]
    return usable, counts, elig, dropped_thin, starved


def bin_quotas(usable, n):
    """Spread n evenly over the usable bins, the remainder to the lowest
    bins first (as v4 does)."""
    nb = len(usable)
    if nb == 0:
        return {}
    per = [n // nb] * nb
    for k in range(n % nb):
        per[k] += 1
    return dict(zip(sorted(usable), per))


# ============================================================
# Stage 1: integer programs
# ============================================================

def _milp(c, rows, lo, hi, ub, integrality=None):
    A = sparse.coo_matrix(
        ([v for _, _, v in rows], ([r for r, _, _ in rows],
                                   [k for _, k, _ in rows])),
        shape=(len(lo), len(c))).tocsr()
    cons = LinearConstraint(A, np.array(lo, float), np.array(hi, float))
    integ = np.ones(len(c)) if integrality is None else integrality
    res = milp(c=np.asarray(c, float), constraints=cons,
               integrality=integ, bounds=Bounds(np.zeros(len(c)),
                                                np.asarray(ub, float)),
               options={"time_limit": ILP_TIME_LIMIT})
    return res


def ilp_type(supply, quotas, cap, mode, rng=None):
    """One trial type, DISJOINT in ("pair", "image").

    supply  array scenes x bins of unordered pairs
    quotas  array of per-block quotas per bin
    mode    "feasible": exact quotas (random objective if rng, to give a
            random allocation); "max": place as much as possible, equal in
            both blocks.

    Returns (status, alloc[s, b, k]) with status "feasible", "infeasible"
    or "timeout"; alloc is None unless a solution exists."""
    S, B = supply.shape
    rows, lo, hi = [], [], []

    def row(entries, l, h):
        r = len(lo)
        rows.extend((r, k, v) for k, v in entries)
        lo.append(l)
        hi.append(h)

    if SCENE_MATCH == "scene_bin":
        # m[s,b]: matched couples (one pair per block) from scene s, bin b
        nv = S * B
        ub = [min(supply[s, b] // 2, cap) for s in range(S) for b in range(B)]
        for s in range(S):
            row([(s * B + b, 1) for b in range(B)], 0, cap)
        for b in range(B):
            q = quotas[b]
            row([(s * B + b, 1) for s in range(S)],
                q if mode == "feasible" else 0, q)
    else:
        # x[s,b,k]
        nv = S * B * 2

        def ix(s, b, k):
            return (s * B + b) * 2 + k

        ub = [min(supply[s, b], cap) for s in range(S) for b in range(B)
              for _ in (0, 1)]
        for s in range(S):
            for b in range(B):
                if supply[s, b]:
                    row([(ix(s, b, 0), 1), (ix(s, b, 1), 1)], 0,
                        supply[s, b])
            for k in (0, 1):
                row([(ix(s, b, k), 1) for b in range(B)], 0, cap)
        for b in range(B):
            q = quotas[b]
            for k in (0, 1):
                row([(ix(s, b, k), 1) for s in range(S)],
                    q if mode == "feasible" else 0, q)
            if mode == "max":
                row([(ix(s, b, 0), 1) for s in range(S)]
                    + [(ix(s, b, 1), -1) for s in range(S)], 0, 0)

    if mode == "max":
        c = -np.ones(nv)
    elif rng is not None:
        c = -rng.random(nv)
    else:
        c = np.zeros(nv)
    res = _milp(c, rows, lo, hi, ub)
    if res.status == 2 or (res.x is None and res.status != 1):
        return "infeasible", None
    if res.x is None:
        return "timeout", None
    x = np.rint(res.x).astype(int)
    # In "max" mode a time limit leaves the best placement found, not the
    # best possible: the shortfall is then an upper bound.
    found = ("feasible_limit" if (mode == "max" and res.status != 0)
             else "feasible")
    if SCENE_MATCH == "scene_bin":
        m = x.reshape(S, B)
        alloc = np.stack([m, m], axis=2)
    else:
        alloc = x.reshape(S, B, 2)
    return found, alloc


def ilp_scene_split(supplies, quotas, cap, mode, rng=None):
    """DISJOINT = "scene": all four types jointly, with a binary z[s] that
    puts scene s in set A (1) or set B (0). Returns
    (status, {type: alloc[s, b, k]}, z)."""
    types = list(supplies)
    S = next(iter(supplies.values())).shape[0]
    offs, nv = {}, 0
    for t in types:
        offs[t] = nv
        nv += supplies[t].shape[0] * supplies[t].shape[1] * 2
    zoff = nv
    nv += S
    rows, lo, hi = [], [], []

    def row(entries, l, h):
        r = len(lo)
        rows.extend((r, k, v) for k, v in entries)
        lo.append(l)
        hi.append(h)

    ub = np.zeros(nv)
    ub[zoff:] = 1
    for t in types:
        sup = supplies[t]
        _, B = sup.shape

        def ix(s, b, k, t=t, B=B):
            return offs[t] + (s * B + b) * 2 + k

        for s in range(S):
            for b in range(B):
                U = int(min(sup[s, b], cap))
                for k in (0, 1):
                    ub[ix(s, b, k)] = U
                if U == 0:
                    continue
                row([(ix(s, b, 0), 1), (ix(s, b, 1), 1)], 0, sup[s, b])
                row([(ix(s, b, 0), 1), (zoff + s, -U)], -np.inf, 0)
                row([(ix(s, b, 1), 1), (zoff + s, U)], -np.inf, U)
            for k in (0, 1):
                row([(ix(s, b, k), 1) for b in range(B)], 0, cap)
        for b in range(B):
            q = quotas[t][b]
            for k in (0, 1):
                row([(ix(s, b, k), 1) for s in range(S)],
                    q if mode == "feasible" else 0, q)
            if mode == "max":
                row([(ix(s, b, 0), 1) for s in range(S)]
                    + [(ix(s, b, 1), -1) for s in range(S)], 0, 0)
    if mode == "max":
        c = -np.ones(nv)
        c[zoff:] = 0
    elif rng is not None:
        c = -rng.random(nv)
    else:
        c = np.zeros(nv)
    res = _milp(c, rows, lo, hi, ub)
    if res.x is None:
        return ("timeout" if res.status == 1 else "infeasible"), None, None
    x = np.rint(res.x).astype(int)
    out = {}
    for t in types:
        S_, B = supplies[t].shape
        out[t] = x[offs[t]:offs[t] + S_ * B * 2].reshape(S_, B, 2)
    found = ("feasible_limit" if (mode == "max" and res.status != 0)
             else "feasible")
    return found, out, x[zoff:]


def supply_matrix(tp, scenes, bins):
    sidx = {s: i for i, s in enumerate(scenes)}
    bidx = {b: i for i, b in enumerate(bins)}
    M = np.zeros((len(scenes), len(bins)), dtype=int)
    g = tp[tp.mag_bin.isin(bins)].groupby(["scene_id", "mag_bin"]).size()
    for (s, b), n in g.items():
        M[sidx[s], bidx[b]] = n
    return M


def stage1(pools_t, counts, scenes, cap):
    """Exact count feasibility for one combination at one cap.

    Returns a dict: per type -> {usable, quotas, status, placed, shortfall
    by bin, nobin status}, plus overall status."""
    out = {"types": {}}
    info = {}
    for t in TYPES:
        tp = pools_t[t]
        usable, cnt, elig, thin, starved = usable_bins(tp, counts[t], cap)
        q = bin_quotas(usable, counts[t])
        info[t] = dict(usable=usable, quotas=q, bin_counts=cnt,
                       eligible_scenes=elig, thin=thin, starved=starved)
    if DISJOINT in ("pair", "image"):
        all_ok = True
        for t in TYPES:
            d = info[t]
            if not d["usable"]:
                d.update(status="infeasible", placed={}, shortfall={},
                         note="no usable bin")
                all_ok = False
                out["types"][t] = d
                continue
            bins = sorted(d["usable"])
            sup = supply_matrix(pools_t[t], scenes, bins)
            qv = np.array([d["quotas"][b] for b in bins])
            st, alloc = ilp_type(sup, qv, cap, "feasible")
            d["status"] = st
            d["shortfall_exact"] = True
            if st == "timeout":
                all_ok = False
                placed = np.zeros(len(bins), int)
                d["shortfall_exact"] = False
            elif st != "feasible":
                all_ok = False
                st2, alloc2 = ilp_type(sup, qv, cap, "max")
                placed = (alloc2[:, :, 0].sum(0) if alloc2 is not None
                          else np.zeros(len(bins), int))
                d["shortfall_exact"] = st2 == "feasible"
            else:
                placed = qv
            d["placed"] = {b: int(v) for b, v in zip(bins, placed)}
            d["shortfall"] = {b: int(d["quotas"][b] - d["placed"][b])
                              for b in bins}
            out["types"][t] = d
        sts = [out["types"][t]["status"] for t in TYPES]
        out["status"] = ("feasible" if all_ok else
                         "timeout" if "timeout" in sts else "infeasible")
    else:
        sup, qd, bins_of = {}, {}, {}
        ok_bins = True
        for t in TYPES:
            d = info[t]
            if not d["usable"]:
                ok_bins = False
            bins_of[t] = sorted(d["usable"])
            sup[t] = supply_matrix(pools_t[t], scenes, bins_of[t])
            qd[t] = np.array([d["quotas"][b] for b in bins_of[t]])
        if not ok_bins:
            st = "infeasible"
        else:
            st, alloc, z = ilp_scene_split(sup, qd, cap, "feasible")
        alloc2, st2 = None, None
        if st == "infeasible" and ok_bins:
            st2, alloc2, _ = ilp_scene_split(sup, qd, cap, "max")
        for t in TYPES:
            d = info[t]
            bins = bins_of[t]
            if st == "feasible":
                placed = qd[t]
            elif ok_bins and alloc2 is not None:
                placed = alloc2[t][:, :, 0].sum(0)
            else:
                placed = np.zeros(len(bins), int)
            d["status"] = st
            d["shortfall_exact"] = (st == "feasible"
                                    or (st == "infeasible"
                                        and st2 == "feasible"))
            d["placed"] = {b: int(v) for b, v in zip(bins, placed)}
            d["shortfall"] = {b: int(d["quotas"][b] - d["placed"][b])
                              for b in bins}
            out["types"][t] = d
        out["status"] = st
    return out


def stage1_nobins(pools_t, counts, scenes, cap):
    """Diagnostic: the same problem with the bin quotas removed (every pair
    of a type in one bin). Tells whether a shortfall is the bin quotas'
    doing or a lack of pairs and scenes outright."""
    res = {}
    if DISJOINT in ("pair", "image"):
        for t in TYPES:
            tp = pools_t[t].assign(mag_bin=1)
            sup = supply_matrix(tp, scenes, [1])
            st, _ = ilp_type(sup, np.array([counts[t]]), cap, "feasible")
            res[t] = st
    else:
        sup = {t: supply_matrix(pools_t[t].assign(mag_bin=1), scenes, [1])
               for t in TYPES}
        qd = {t: np.array([counts[t]]) for t in TYPES}
        st, _, _ = ilp_scene_split(sup, qd, cap, "feasible")
        res = {t: st for t in TYPES}
    return res


# ============================================================
# Stage 2: concrete matched draws
# ============================================================

class DrawState:
    def __init__(self):
        self.used = set()             # unordered keys, either block
        self.cnt = {}                 # (block, type, scene) -> trials
        self.img_block = {}           # image -> block (DISJOINT image)
        self.scene_block = None       # scene -> block (DISJOINT scene)
        self.scene_use = {}           # scene -> analysed trials, both sets
        self.info = {"A": np.zeros(2), "B": np.zeros(2)}  # sum x^2 so far

    def allowed(self, row, blk, t, use_cap=True):
        s = row["scene_id"]
        if use_cap and self.cnt.get((blk, t, s), 0) >= MAX_PER_SCENE_PER_TYPE:
            return False
        if DISJOINT == "scene" and self.scene_block[s] != blk:
            return False
        if DISJOINT == "image":
            for f in (row["file_a"], row["file_b"]):
                if self.img_block.get(f, blk) != blk:
                    return False
        return True

    def take(self, row, blk, t):
        self.used.add(row["key"])
        k = (blk, t, row["scene_id"])
        self.cnt[k] = self.cnt.get(k, 0) + 1
        if t in TYPES:
            s = row["scene_id"]
            self.scene_use[s] = self.scene_use.get(s, 0) + 1
        if DISJOINT == "image":
            self.img_block[row["file_a"]] = blk
            self.img_block[row["file_b"]] = blk


def _records(tp):
    cols = ["key", "scene_id", "file_a", "file_b", "mag_bin", "x_sat",
            "x_con"]
    return tp[cols].to_dict("records")


def greedy_type(tp, t, quotas, state, rng):
    """v4's stratified_sample(), for two blocks: bins in random order, the
    bin's pairs sorted then shuffled, and each anchor that fits goes to one
    block while its nearest unused partner (same bin; same scene under
    scene_bin) goes to the other. The anchor's block alternates, from a
    random start, so neither block is systematically the anchor.

    One change from v4 (SPREAD_SCENES): among the remaining candidates, the
    next anchor is taken from the scenes used least so far (over all types
    and both sets), ties in random order. A purely random draw of 90 trials
    leaves about ten of the 47 scenes unused.

    Returns (ok, couples) with couples a list of (index in set A, index in
    set B)."""
    recs = _records(tp)
    bins = np.array([r["mag_bin"] for r in recs])
    scn = np.array([r["scene_id"] for r in recs])
    ax = np.abs(np.array([[r["x_sat"], r["x_con"]] for r in recs]))
    couples = []
    couple = int(rng.integers(2))
    ok = True
    for b in rng.permutation(sorted(quotas)):
        want = quotas[b]
        idx = np.where(bins == b)[0]
        placed = 0
        remaining = list(rng.permutation(idx))
        while placed < want and remaining:
            if SPREAD_SCENES:
                use = np.array([state.scene_use.get(scn[k], 0)
                                for k in remaining])
                pos = int(np.flatnonzero(use == use.min())[0])
            else:
                pos = 0
            i = remaining.pop(pos)
            ri = recs[i]
            if ri["key"] in state.used:
                continue
            first = "A" if couple % 2 == 0 else "B"
            done = False
            for a in (first, "B" if first == "A" else "A"):
                o = "B" if a == "A" else "A"
                if not state.allowed(ri, a, t):
                    continue
                cand = idx[idx != i]
                if SCENE_MATCH == "scene_bin":
                    cand = cand[scn[cand] == ri["scene_id"]]
                best, bd = None, np.inf
                for j in cand:
                    rj = recs[j]
                    if rj["key"] in state.used:
                        continue
                    if DISJOINT == "image" and (
                            {rj["file_a"], rj["file_b"]}
                            & {ri["file_a"], ri["file_b"]}):
                        continue
                    if not state.allowed(rj, o, t):
                        continue
                    d = ((ax[j] - ax[i]) ** 2).sum()
                    if d < bd:
                        best, bd = j, d
                if best is None:
                    continue
                state.take(ri, a, t)
                state.take(recs[best], o, t)
                couples.append((i, best) if a == "A" else (best, i))
                couple += 1
                placed += 1
                done = True
                break
            if not done:
                continue
        if placed < want:
            ok = False
    return ok, couples


def balance_couples(tp, t, couples, state, rng):
    """Decide which member of each matched couple goes to set A, so that the
    information per weight is as equal as possible between the sets.

    Only the side of each couple changes; no pair enters or leaves, so
    every count, bin, scene and rule is untouched. Under scene_bin both
    members share a scene, so a flip cannot break the cap; under bin it is
    checked. Not applied when DISJOINT is image or scene (a flip would move
    images or scenes across sets).

    Objective: squared relative A-B difference in this type's information
    for each weight it informs (sat_only: x_sat; fog_only: x_con; mixed:
    both) plus the same for the running totals over all types placed so
    far. Couples are visited in random order and flipped while that lowers
    the objective, for up to 50 passes."""
    X = tp[["x_sat", "x_con"]].to_numpy(float) ** 2
    ws = {"sat_only": [0], "fog_only": [1]}.get(t, [0, 1])
    a = np.array([X[i] for i, _ in couples])
    b = np.array([X[j] for _, j in couples])
    flip = np.zeros(len(couples), dtype=bool)
    base_A = state.info["A"].copy()
    base_B = state.info["B"].copy()
    scenes = tp.scene_id.to_numpy()

    def obj(f):
        IA = np.where(f[:, None], b, a).sum(0)
        IB = np.where(f[:, None], a, b).sum(0)
        J = 0.0
        for w in ws:
            m = (IA[w] + IB[w]) / 2
            if m > 0:
                J += ((IA[w] - IB[w]) / m) ** 2
        TA, TB = base_A + IA, base_B + IB
        for w in (0, 1):
            m = (TA[w] + TB[w]) / 2
            if m > 0:
                J += ((TA[w] - TB[w]) / m) ** 2
        return J

    def cap_ok(k):
        i, j = couples[k]
        si, sj = scenes[i], scenes[j]
        if si == sj:
            return True
        # i moves A->B (or back), j the other way
        src_i, dst_i = (("A", "B") if not flip[k] else ("B", "A"))
        return (state.cnt.get((dst_i, t, si), 0) < MAX_PER_SCENE_PER_TYPE
                and state.cnt.get((src_i, t, sj), 0)
                < MAX_PER_SCENE_PER_TYPE)

    cur = obj(flip)
    if not BALANCE_ASSIGNMENT or DISJOINT != "pair":
        return couples, cur
    for _ in range(50):
        improved = False
        for k in rng.permutation(len(couples)):
            if not cap_ok(k):
                continue
            flip[k] = ~flip[k]
            new = obj(flip)
            if new < cur - 1e-12:
                cur = new
                improved = True
                i, j = couples[k]
                si, sj = scenes[i], scenes[j]
                if si != sj:
                    # keep the per-scene counts in step with the flip
                    s_from = "A" if flip[k] else "B"
                    s_to = "B" if flip[k] else "A"
                    state.cnt[(s_from, t, si)] -= 1
                    state.cnt[(s_to, t, si)] = state.cnt.get(
                        (s_to, t, si), 0) + 1
                    state.cnt[(s_to, t, sj)] -= 1
                    state.cnt[(s_from, t, sj)] = state.cnt.get(
                        (s_from, t, sj), 0) + 1
            else:
                flip[k] = ~flip[k]
        if not improved:
            break
    return ([((j, i) if f else (i, j)) for (i, j), f in zip(couples, flip)],
            cur)


def fill_from_alloc(tp, t, bins, alloc, scenes, state, rng):
    """Fallback when the greedy draw stalls but stage 1 found an allocation:
    take exactly the allocated number of pairs from each scene x bin x block.
    Under scene_bin, partners are still nearest-matched within the cell."""
    recs = _records(tp)
    ax = np.abs(np.array([[r["x_sat"], r["x_con"]] for r in recs]))
    by_cell = {}
    for i, r in enumerate(recs):
        by_cell.setdefault((r["scene_id"], r["mag_bin"]), []).append(i)
    couples = []
    for bi, b in enumerate(bins):
        listA, listB = [], []
        for si, s in enumerate(scenes):
            nA, nB = alloc[si, bi, 0], alloc[si, bi, 1]
            if nA == 0 and nB == 0:
                continue
            pool = [i for i in by_cell.get((s, b), [])
                    if recs[i]["key"] not in state.used]
            pool = [int(k) for k in rng.permutation(pool)]
            if SCENE_MATCH == "scene_bin":
                for c in range(nA):
                    i = pool.pop(0)
                    j = min(pool, key=lambda j: ((ax[j] - ax[i]) ** 2).sum())
                    pool.remove(j)
                    state.take(recs[i], "A", t)
                    state.take(recs[j], "B", t)
                    couples.append((i, j))
            else:
                for blk, n, lst in (("A", nA, listA), ("B", nB, listB)):
                    for _ in range(n):
                        k = pool.pop(0)
                        state.take(recs[k], blk, t)
                        lst.append(k)
        if listA:
            # pair the two sets' trials in this bin by rank of magnitude,
            # so the couples are nearest-matched for balance_couples
            listA.sort(key=lambda k: tuple(ax[k]))
            listB.sort(key=lambda k: tuple(ax[k]))
            couples += list(zip(listA, listB))
    return couples


def draw_reserves(pools_t, state, rng_p, rng_c):
    """Practice (round-robin over PRACTICE_TYPES, at or below the
    PRACTICE_MAG_QUANTILE of each pool's driver, one scene at most once per
    block) and catch (unused CATCH_TYPE pairs), per block, from pairs no
    set uses. The cap does not apply, as in v4."""
    out = {"A": [], "B": []}
    thr = {t: pools_t[t].driver_sd.quantile(PRACTICE_MAG_QUANTILE)
           for t in PRACTICE_TYPES}
    for blk in ("A", "B"):
        used_scenes = set()
        for i in range(N_PRACTICE_PER_BLOCK):
            t = PRACTICE_TYPES[i % len(PRACTICE_TYPES)]
            tp = pools_t[t]
            m = ((tp.driver_sd <= thr[t]) & ~tp.scene_id.isin(used_scenes)
                 & ~tp.key.isin(state.used))
            cand = tp.index[m.to_numpy()]
            got = None
            for j in rng_p.permutation(cand):
                r = tp.loc[j]
                if state.allowed(r, blk, t, use_cap=False):
                    got = j
                    break
            if got is None:
                continue
            r = tp.loc[got]
            state.take(r, blk, "practice")
            used_scenes.add(r.scene_id)
            out[blk].append(tp.loc[[got]].assign(role="practice",
                                                 trial_type="practice"))
        tp = pools_t[CATCH_TYPE]
        cand = tp.index[~tp.key.isin(state.used).to_numpy()]
        n = 0
        for j in rng_c.permutation(cand):
            if n >= N_CATCH_PER_BLOCK:
                break
            r = tp.loc[j]
            if r.key in state.used or not state.allowed(r, blk, "catch",
                                                        use_cap=False):
                continue
            state.take(r, blk, "catch")
            out[blk].append(tp.loc[[j]].assign(role="catch",
                                               trial_type="catch"))
            n += 1
    return {b: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
            for b, v in out.items()}


def one_draw(pools_t, counts, s1, scenes, combo, d):
    """Build one pair of sets. Returns (sets dict, how) where how is
    "greedy", "fallback" or "failed"."""
    st = Streams(RANDOM_SEED, combo, d)
    state = DrawState()
    how = "greedy"
    if DISJOINT == "scene":
        sup = {t: supply_matrix(pools_t[t], scenes,
                                sorted(s1["types"][t]["usable"]))
               for t in TYPES}
        qd = {t: np.array([s1["types"][t]["quotas"][b]
                           for b in sorted(s1["types"][t]["usable"])])
              for t in TYPES}
        status, alloc_all, z = ilp_scene_split(sup, qd,
                                               MAX_PER_SCENE_PER_TYPE,
                                               "feasible", st("split"))
        if status != "feasible":
            return None, "failed", None
        state.scene_block = {s: ("A" if z[i] else "B")
                             for i, s in enumerate(scenes)}
    parts = {"A": [], "B": []}
    for t in TYPE_ORDER:
        tp = pools_t[t]
        q = s1["types"][t]["quotas"]
        snap = (set(state.used), dict(state.cnt), dict(state.img_block),
                dict(state.scene_use))

        def restore(sn):
            (state.used, state.cnt, state.img_block,
             state.scene_use) = (set(sn[0]), dict(sn[1]), dict(sn[2]),
                                 dict(sn[3]))

        best = None
        for k in range(TYPE_TRIES):
            restore(snap)
            ok, couples = greedy_type(tp, t, q, state, st(f"{t}#{k}"))
            if not ok:
                if DISJOINT == "image":
                    continue
                restore(snap)
                bins = sorted(q)
                if DISJOINT == "scene":
                    alloc = alloc_all[t]
                else:
                    sup = supply_matrix(tp, scenes, bins)
                    stt, alloc = ilp_type(
                        sup, np.array([q[b] for b in bins]),
                        MAX_PER_SCENE_PER_TYPE, "feasible",
                        st(f"{t}#{k}_fallback"))
                    if stt != "feasible":
                        continue
                couples = fill_from_alloc(tp, t, bins, alloc, scenes, state,
                                          st(f"{t}#{k}_fill"))
            couples, J = balance_couples(tp, t, couples, state,
                                         st(f"{t}#{k}_balance"))
            if best is None or J < best[0]:
                best = (J, couples, ok,
                        (set(state.used), dict(state.cnt),
                         dict(state.img_block), dict(state.scene_use)))
        if best is None:
            return None, "failed", None
        _, couples, ok, after = best
        restore(after)
        if not ok:
            how = "fallback"
        X2 = tp[["x_sat", "x_con"]].to_numpy(float) ** 2
        for blk, pos in (("A", 0), ("B", 1)):
            idx = [c[pos] for c in couples]
            state.info[blk] += X2[idx].sum(0)
            parts[blk].append(tp.iloc[idx].assign(
                role="analysed", trial_type=t,
                couple=np.arange(len(couples))))
    res = draw_reserves(pools_t, state, st("practice"), st("catch"))
    sets = {}
    for blk in ("A", "B"):
        main = pd.concat(parts[blk], ignore_index=True)
        # sides: A (the driver-predicted image) on the left in exactly half
        # of each type's trials, and of the reserved trials
        main["a_is_left"] = False
        for t in TYPES:
            m = (main.trial_type == t).to_numpy()
            main.loc[m, "a_is_left"] = balanced_sides(int(m.sum()),
                                                      st(f"sides_{blk}_{t}"))
        r = res[blk]
        if len(r):
            r = r.copy()
            r["a_is_left"] = balanced_sides(len(r), st(f"sides_{blk}_res"))
        sets[blk] = pd.concat([main, r], ignore_index=True)
        sets[blk]["block_set"] = blk
    if state.scene_block is None:
        n_allowed = {"A": len(scenes), "B": len(scenes)}
    else:
        v = list(state.scene_block.values())
        n_allowed = {"A": v.count("A"), "B": v.count("B")}
    return sets, how, n_allowed


def evaluate(sets, counts, s1, n_allowed):
    """Every matching check on one draw. Returns dict of metrics and of
    check results (True/False, or None for not applicable)."""
    A, B = sets["A"], sets["B"]
    mA, mB = A[A.role == "analysed"], B[B.role == "analysed"]
    met, chk = {}, {}

    # counts
    tcA = mA.trial_type.value_counts().to_dict()
    tcB = mB.trial_type.value_counts().to_dict()
    chk["type counts equal the target in both sets"] = all(
        tcA.get(t, 0) == counts[t] == tcB.get(t, 0) for t in TYPES)
    want = {(t, int(b)): int(q) for t in TYPES
            for b, q in s1["types"][t]["quotas"].items()}
    bA = {(t, int(b)): int(n) for (t, b), n in
          mA.groupby(["trial_type", "mag_bin"]).size().items()}
    bB = {(t, int(b)): int(n) for (t, b), n in
          mB.groupby(["trial_type", "mag_bin"]).size().items()}
    chk["bin counts equal the quotas in both sets"] = (bA == want == bB)

    # disjointness and within-set uniqueness (all roles)
    kA, kB = set(A.key), set(B.key)
    chk["no pair in both sets"] = not (kA & kB)
    chk["no pair twice within a set"] = (len(kA) == len(A)
                                         and len(kB) == len(B))
    if DISJOINT == "image":
        iA = set(A.file_a) | set(A.file_b)
        iB = set(B.file_a) | set(B.file_b)
        chk["no image in both sets"] = not (iA & iB)
    else:
        chk["no image in both sets"] = None
    if DISJOINT == "scene":
        chk["no scene in both sets"] = not (set(A.scene_id)
                                            & set(B.scene_id))
    else:
        chk["no scene in both sets"] = None

    # pair rules
    both = pd.concat([A, B])
    s = both[both.type == "sat_only"]
    chk["sat_only: |d_sat| >= cut"] = bool(
        (s.d_sat.abs() >= SAT_ONLY_MIN_DSAT).all())
    chk["sat_only: |d_con| <= contrast tolerance"] = (
        bool((s.d_fog.abs() <= SAT_ONLY_CON_TOL).all())
        if SAT_ONLY_CON_TOL is not None else None)
    f = both[both.type == "fog_only"]
    chk["fog_only: |d_sat| <= 3"] = bool(
        (f.d_sat.abs() <= FOG_ONLY_SAT_TOL).all())
    cf = both[both.type == "conflict"]
    chk["conflict: all balanced"] = (bool(cf.balanced.astype(bool).all())
                                     if BALANCED_CONFLICT_ONLY else None)

    # information per weight
    for blk, m in (("A", mA), ("B", mB)):
        xs, xc = m.x_sat.to_numpy(float), m.x_con.to_numpy(float)
        met[f"info_sat_{blk}"] = float((xs ** 2).sum())
        met[f"info_con_{blk}"] = float((xc ** 2).sum())
        cos = float((xs * xc).sum() / np.sqrt((xs ** 2).sum()
                                              * (xc ** 2).sum()))
        met[f"cos_{blk}"] = cos
        met[f"vif_{blk}"] = 1 / (1 - cos ** 2)
        sign = np.where(m.a_is_left.astype(bool), 1.0, -1.0)
        met[f"pearson_lr_{blk}"] = float(np.corrcoef(xs * sign,
                                                     xc * sign)[0, 1])
        met[f"scenes_{blk}"] = int(m.scene_id.nunique())
        met[f"max_per_scene_{blk}"] = int(m.scene_id.value_counts().max())
        for t in TYPES:
            mt = m[m.trial_type == t]
            met[f"info_sat_{t}_{blk}"] = float((mt.x_sat ** 2).sum())
            met[f"info_con_{t}_{blk}"] = float((mt.x_con ** 2).sum())
            met[f"mean_abs_sat_{t}_{blk}"] = float(mt.x_sat.abs().mean())
            met[f"mean_abs_con_{t}_{blk}"] = float(mt.x_con.abs().mean())
    met["info_sat_rel_diff"] = rel_diff(met["info_sat_A"], met["info_sat_B"])
    met["info_con_rel_diff"] = rel_diff(met["info_con_A"], met["info_con_B"])
    chk[f"information x_sat within {INFO_TOL_TOTAL:.0%}"] = (
        met["info_sat_rel_diff"] <= INFO_TOL_TOTAL + EPS)
    chk[f"information x_con within {INFO_TOL_TOTAL:.0%}"] = (
        met["info_con_rel_diff"] <= INFO_TOL_TOTAL + EPS)
    worst = 0.0
    for t, ws in (("sat_only", ["sat"]), ("fog_only", ["con"]),
                  ("together", ["sat", "con"]),
                  ("conflict", ["sat", "con"])):
        for w in ws:
            rd = rel_diff(met[f"info_{w}_{t}_A"], met[f"info_{w}_{t}_B"])
            met[f"info_{w}_{t}_rel_diff"] = rd
            worst = max(worst, rd)
    met["info_type_worst_rel_diff"] = worst
    chk[f"information per type within {INFO_TOL_TYPE:.0%}"] = (
        worst <= INFO_TOL_TYPE + EPS)
    # KS over the trials where each predictor carries information: x_sat
    # on sat_only and the mixed types, x_con on fog_only and the mixed
    # types. The single-feature types' other predictor is held below
    # 0.077 SD by the matching rules, and its random order would otherwise
    # dominate the statistic while carrying almost no information.
    inf_s = ["sat_only", "together", "conflict"]
    inf_c = ["fog_only", "together", "conflict"]
    met["ks_sat"] = float(stats.ks_2samp(
        mA[mA.trial_type.isin(inf_s)].x_sat.abs(),
        mB[mB.trial_type.isin(inf_s)].x_sat.abs(),
        method="asymp").statistic)
    met["ks_con"] = float(stats.ks_2samp(
        mA[mA.trial_type.isin(inf_c)].x_con.abs(),
        mB[mB.trial_type.isin(inf_c)].x_con.abs(),
        method="asymp").statistic)
    for t in TYPES:
        w_ = "x_sat" if t == "sat_only" else "x_con"
        met[f"ks_{t}_driver"] = float(stats.ks_2samp(
            mA[mA.trial_type == t][w_].abs(),
            mB[mB.trial_type == t][w_].abs(), method="asymp").statistic)
    chk[f"KS |x_sat| <= {KS_MAX_D}"] = met["ks_sat"] <= KS_MAX_D + EPS
    chk[f"KS |x_con| <= {KS_MAX_D}"] = met["ks_con"] <= KS_MAX_D + EPS
    met["cos_diff"] = abs(met["cos_A"] - met["cos_B"])
    chk[f"predictor correlation (cos) differs by <= {COS_DIFF_MAX}"] = (
        met["cos_diff"] <= COS_DIFF_MAX + EPS)
    chk[f"VIF <= {VIF_MAX} in both sets"] = max(met["vif_A"],
                                                met["vif_B"]) <= VIF_MAX + EPS

    # sides
    side_ok = True
    for m in (mA, mB):
        for t in TYPES:
            mt = m[m.trial_type == t]
            n_left = int(mt.a_is_left.astype(bool).sum())
            if n_left not in (len(mt) // 2, (len(mt) + 1) // 2):
                side_ok = False
    chk["predicted image on the left in half of each type (+-1 if odd)"] = (
        side_ok)

    # scenes
    for blk in ("A", "B"):
        met[f"scenes_allowed_{blk}"] = n_allowed[blk]
    chk[f"each set uses >= {MIN_SCENE_SHARE:.0%} of the scenes it may use"] \
        = all(met[f"scenes_{b}"] >= MIN_SCENE_SHARE * n_allowed[b] - EPS
              for b in ("A", "B"))
    capA = mA.groupby(["trial_type", "scene_id"]).size().max()
    capB = mB.groupby(["trial_type", "scene_id"]).size().max()
    chk[f"<= {MAX_PER_SCENE_PER_TYPE} per scene per type in each set"] = (
        max(capA, capB) <= MAX_PER_SCENE_PER_TYPE)
    if SCENE_MATCH == "scene_bin":
        gA = mA.groupby(["trial_type", "scene_id", "mag_bin"]).size()
        gB = mB.groupby(["trial_type", "scene_id", "mag_bin"]).size()
        chk["same scenes: each scene x type x bin count equal in both sets"] \
            = gA.sort_index().equals(gB.sort_index())
    else:
        chk["same scenes: each scene x type x bin count equal in both sets"] \
            = None
    gA = mA.groupby(["trial_type", "scene_id"]).size()
    gB = mB.groupby(["trial_type", "scene_id"]).size()
    met["scene_count_l1"] = int(gA.sub(gB, fill_value=0).abs().sum())

    # reserves
    for blk, S_ in (("A", A), ("B", B)):
        met[f"practice_{blk}"] = int((S_.role == "practice").sum())
        met[f"catch_{blk}"] = int((S_.role == "catch").sum())
    chk["practice and catch reserved in full for both sets"] = all(
        met[f"practice_{b}"] == N_PRACTICE_PER_BLOCK
        and met[f"catch_{b}"] == N_CATCH_PER_BLOCK for b in ("A", "B"))
    return met, chk


# ============================================================
# Running the combinations
# ============================================================

def run_combo(level_set, size, mix_name, pools_t, scenes, out_dir, ck,
              feas_rows, short_rows, draw_rows):
    combo = f"{level_set}|{size}|{mix_name}"
    counts = largest_remainder(size, TYPE_MIXES[mix_name])
    s1 = stage1(pools_t, counts, scenes, MAX_PER_SCENE_PER_TYPE)
    nob = stage1_nobins(pools_t, counts, scenes, MAX_PER_SCENE_PER_TYPE)
    s1["nobins"] = nob

    # smallest cap that works, when the set cap does not
    min_cap = MAX_PER_SCENE_PER_TYPE if s1["status"] == "feasible" else None
    if s1["status"] != "feasible":
        for c in range(MAX_PER_SCENE_PER_TYPE + 1,
                       MAX_PER_SCENE_PER_TYPE + MAX_CAP_SEARCH + 1):
            if stage1(pools_t, counts, scenes, c)["status"] == "feasible":
                min_cap = c
                break
    s1["min_cap"] = min_cap

    for t in TYPES:
        d = s1["types"][t]
        feas_rows.append({
            "level_set": level_set, "block_size": size, "mix": mix_name,
            "type": t, "trials_per_block": counts[t],
            "pool_pairs": len(pools_t[t]),
            "pool_scenes": pools_t[t].scene_id.nunique(),
            "usable_bins": " ".join(map(str, sorted(d["usable"]))),
            "bins_dropped_thin": " ".join(map(str, d["thin"])),
            "bins_dropped_scenes": " ".join(map(str, d["starved"])),
            "quotas": " ".join(f"{b}:{q}" for b, q in
                               sorted(d["quotas"].items())),
            "status": d["status"],
            "placed_per_block": sum(d["placed"].values()),
            "shortfall_per_block": sum(d["shortfall"].values()),
            "shortfall_exact": d.get("shortfall_exact", True),
            "feasible_without_bin_quotas": nob[t],
            "min_cap_combination": min_cap if min_cap else
            f">{MAX_PER_SCENE_PER_TYPE + MAX_CAP_SEARCH}",
        })
        for b in sorted(d["quotas"]):
            short_rows.append({
                "level_set": level_set, "block_size": size, "mix": mix_name,
                "type": t, "bin": b, "quota_per_block": d["quotas"][b],
                "placed_per_block": d["placed"].get(b, 0),
                "shortfall_per_block": d["shortfall"].get(b, 0),
                "pairs_in_bin": d["bin_counts"].get(b, 0),
                "eligible_scenes": d["eligible_scenes"].get(b, 0)})

    res = {"level_set": level_set, "size": size, "mix": mix_name,
           "counts": counts, "s1": s1, "draws": []}
    if any(s1["types"][t]["status"] == "timeout" for t in TYPES):
        res["verdict"] = "UNDECIDED (solver time limit)"
        return res
    if DISJOINT != "image" and s1["status"] != "feasible":
        res["verdict"] = "INFEASIBLE"
        return res
    if DISJOINT == "image" and any(s1["types"][t]["status"] != "feasible"
                                   for t in TYPES):
        res["verdict"] = "INFEASIBLE"
        return res

    hows, mets, chks = [], [], []
    for d in range(N_DRAWS):
        sets, how, n_allowed = one_draw(pools_t, counts, s1, scenes, combo,
                                        d)
        hows.append(how)
        if sets is None:
            draw_rows.append({"level_set": level_set, "block_size": size,
                              "mix": mix_name, "draw": d, "built": how,
                              "all_checks_pass": False})
            continue
        met, chk = evaluate(sets, counts, s1, n_allowed)
        all_ok = all(v for v in chk.values() if v is not None)
        mets.append(met)
        chks.append(chk)
        row = {"level_set": level_set, "block_size": size, "mix": mix_name,
               "draw": d, "built": how, "all_checks_pass": all_ok}
        row.update(met)
        row.update({f"CHECK {k}": ("N/A" if v is None else
                                   ("PASS" if v else "FAIL"))
                    for k, v in chk.items()})
        draw_rows.append(row)
        if d == 0 and WRITE_EXAMPLE_SETS:
            ex = os.path.join(out_dir, "example_sets",
                              f"{level_set}_n{size}_{mix_name}")
            os.makedirs(ex, exist_ok=True)
            cols = ["block_set", "role", "trial_type", "couple", "scene_id",
                    "mag_bin", "file_a", "file_b", "a_is_left", "x_sat",
                    "x_con", "d_sat", "d_fog", "colour_a", "fog_a_level",
                    "colour_b", "fog_b_level", "balanced"]
            for blk in ("A", "B"):
                sets[blk][cols].to_csv(os.path.join(ex, f"set_{blk}.csv"),
                                       index=False)
    res["hows"] = hows
    res["mets"] = pd.DataFrame(mets)
    res["chks"] = chks
    n_built = sum(h != "failed" for h in hows)
    n_pass = sum(all(v for v in c.values() if v is not None) for c in chks)
    res["n_built"], res["n_pass"] = n_built, n_pass
    res["pass_share"] = n_pass / N_DRAWS
    ok = (n_built == N_DRAWS and res["pass_share"] >= REQUIRED_PASS_SHARE)
    res["verdict"] = "FEASIBLE" if ok else (
        "COUNTS FEASIBLE, DRAWS FAIL" if n_built else "DRAWS FAIL")
    # per-check pass shares
    names = list(chks[0]) if chks else []
    res["check_share"] = {
        n: (None if chks[0][n] is None else
            sum(bool(c[n]) for c in chks) / N_DRAWS) for n in names}
    scope = f"{level_set} n={size} {mix_name}"
    for n, sh in res["check_share"].items():
        if sh is None:
            continue
        ck.add(scope, f"draws: {n}", sh >= REQUIRED_PASS_SHARE,
               f"{sh:.0%} of {N_DRAWS} draws")
    return res


# ============================================================
# Report
# ============================================================

def describe(x):
    return f"{x.mean():7.1f} [{x.min():.1f}-{x.max():.1f}]"


def write_report(out_dir, log, ck, pools, funnel, bin_rows, results,
                 v4, t0):
    L = []
    w = L.append
    w("=" * 78)
    w("v5 STEP 1: PAIR CAPACITY FOR TWO MATCHED SETS")
    w("analyse_pair_capacity_v5.py")
    w("=" * 78)
    w(f"run finished  {time.strftime('%Y-%m-%d %H:%M:%S')}   "
      f"elapsed {time.time() - t0:.1f} s")
    w(f"output        {out_dir}")
    w("")
    w("1. WHAT THIS RUN DID")
    w("-" * 78)
    w("For every level set x block size x type mix it asked whether two sets")
    w("of analysed pairs (A and B, one per block) exist with identical type")
    w("and magnitude-bin counts, no shared pair, matched information for")
    w("each weight, balanced sides and a spread of scenes, plus each block's")
    w("practice and catch pairs. Stage 1 solves the counts exactly (integer")
    w("program); stage 2 builds concrete matched draws and checks them.")
    w("")
    w("Key settings")
    w(f"  block sizes (analysed / block)   {BLOCK_SIZES}")
    for k, mix in TYPE_MIXES.items():
        w(f"  mix {k:15s}  " + ", ".join(f"{t} {mix[t]:.3f}" for t in TYPES))
    for k, v in LEVEL_SETS.items():
        w(f"  level set {k:13s}  drops colour {v['drop_colour'] or '-'}, "
          f"fog {v['drop_fog'] or '-'}")
    w(f"  disjoint                         {DISJOINT}")
    w(f"  scene matching                   {SCENE_MATCH}")
    w(f"  cap per scene per type per block {MAX_PER_SCENE_PER_TYPE}")
    w(f"  sat_only |d_sat| >=              {SAT_ONLY_MIN_DSAT} "
      f"({SAT_ONLY_MIN_DSAT / SD_SAT:.3f} SD)")
    w(f"  sat_only |d_con| <=              {SAT_ONLY_CON_TOL} "
      + (f"({SAT_ONLY_CON_TOL / SD_FOG:.3f} SD)" if SAT_ONLY_CON_TOL else ""))
    w(f"  fog_only |d_sat| <=              {FOG_ONLY_SAT_TOL} "
      f"({FOG_ONLY_SAT_TOL / SD_SAT:.3f} SD)")
    w(f"  conflict balanced only           {BALANCED_CONFLICT_ONLY}")
    w(f"  spread scenes / balance couples  {SPREAD_SCENES} / "
      f"{BALANCE_ASSIGNMENT}, best of {TYPE_TRIES} tries per type")
    w(f"  reserved per block               {N_PRACTICE_PER_BLOCK} practice, "
      f"{N_CATCH_PER_BLOCK} catch")
    w(f"  draws per combination            {N_DRAWS}, feasible if >= "
      f"{REQUIRED_PASS_SHARE:.0%} pass every check")
    w(f"  tolerances                       information {INFO_TOL_TOTAL:.0%} "
      f"total / {INFO_TOL_TYPE:.0%} per type, KS <= {KS_MAX_D}, "
      f"cos diff <= {COS_DIFF_MAX}, VIF <= {VIF_MAX}, scenes >= "
      f"{MIN_SCENE_SHARE:.0%} of those allowed")
    w("")
    w("Inputs")
    for line in log:
        w("  " + line)
    w("")

    # pool
    w("2. THE POOL (unordered pairs, after each rule)")
    w("-" * 78)
    fu = pd.DataFrame(funnel)
    for ls in LEVEL_SETS:
        w(f"[{ls}]")
        for t in TYPES:
            sub = fu[(fu.level_set == ls) & (fu.type == t)]
            w(f"  {t:9s} " + " -> ".join(
                f"{r.step.split(' ', 1)[-1]}: {r.unordered_pairs:,}"
                for r in sub.itertuples()))
    if SAT_ONLY_CON_TOL is not None:
        for ls in LEVEL_SETS:
            sub = fu[(fu.level_set == ls) & (fu.type == "sat_only")]
            a = sub[sub.step.str.startswith("2")].unordered_pairs.iloc[0]
            b = sub[sub.step.str.startswith("3")].unordered_pairs.iloc[0]
            w(f"  contrast rule on sat_only [{ls}]: {a - b} of {a:,} pairs "
              f"removed")
        first = pools[next(iter(LEVEL_SETS))]
        s_ = first[first.type == "sat_only"]
        w("  (largest |d_con| among the remaining sat_only pairs: "
          f"{s_.d_fog.abs().max():.3f} units)")
    w("")
    w("Supply by magnitude bin: pairs / scenes with >= 1 pair / scenes with "
      ">= 2 pairs")
    br = pd.DataFrame(bin_rows)
    for ls in LEVEL_SETS:
        w(f"[{ls}]")
        for t in TYPES:
            sub = br[(br.level_set == ls) & (br.type == t)]
            w(f"  {t:9s} " + "  ".join(
                f"bin{r.bin}: {r.pairs}/{r.scenes_1}/{r.scenes_2}"
                for r in sub.itertuples()))
    w("")

    # feasibility
    w("3. FEASIBILITY")
    w("-" * 78)
    w("verdict per level set x size x mix. Stage 1 = exact counts; draws =")
    w("share of draws passing every check; fb = draws that needed the")
    w("stage-1 allocation because the greedy draw stalled.")
    w("")
    w(f"{'level set':12s} {'size':>4s} {'mix':15s} {'per-type counts':22s} "
      f"{'stage 1':10s} {'draws':>6s} {'fb':>3s}  verdict")
    for r in results:
        c = r["counts"]
        cs = "/".join(str(c[t]) for t in TYPES)
        ds = (f"{r['pass_share']:.0%}" if "pass_share" in r else "-")
        fb = (str(sum(h == "fallback" for h in r["hows"]))
              if "hows" in r else "-")
        w(f"{r['level_set']:12s} {r['size']:4d} {r['mix']:15s} {cs:22s} "
          f"{r['s1']['status']:10s} {ds:>6s} {fb:>3s}  {r['verdict']}")
    w("  (counts are sat_only/fog_only/together/conflict per block)")
    w("")
    bad = [r for r in results if r["verdict"] != "FEASIBLE"]
    if bad:
        w("Where the shortfall is")
        for r in bad:
            w(f"  {r['level_set']} n={r['size']} {r['mix']}: "
              f"{r['verdict']}")
            for t in TYPES:
                d = r["s1"]["types"][t]
                sf = {b: v for b, v in d["shortfall"].items() if v}
                notes = []
                if d["thin"]:
                    notes.append(f"bins {d['thin']} too thin")
                if d["starved"]:
                    notes.append(f"bins {d['starved']} too few scenes")
                if sf or notes or d["status"] != "feasible":
                    w(f"    {t:9s} stage 1 {d['status']}"
                      + (" (joint over types)" if DISJOINT == "scene"
                         else "")
                      + "; short per block "
                      f"by bin {sf or '-'}"
                      + ("" if d.get("shortfall_exact", True) else
                         " (an upper bound: the maximisation hit the "
                         "solver time limit)")
                      + f"; quotas {d['quotas']}; "
                      f"without bin quotas: {r['s1']['nobins'][t]}"
                      + (f"; {'; '.join(notes)}" if notes else ""))
            mc = r["s1"]["min_cap"]
            w(f"    smallest cap that makes the counts feasible: "
              f"{mc if mc else f'> {MAX_PER_SCENE_PER_TYPE + MAX_CAP_SEARCH}'}")
            if "pass_share" in r:
                w(f"    draws passing every check: {r['pass_share']:.0%} "
                  f"(needed {REQUIRED_PASS_SHARE:.0%})")
            if "check_share" in r:
                failing = {n: s for n, s in r["check_share"].items()
                           if s is not None and s < REQUIRED_PASS_SHARE}
                for n, s in failing.items():
                    w(f"    draws fail: {n} (passes in {s:.0%})")
        w("")
    else:
        w("Every combination is feasible.")
        w("")
    # notes on dropped bins in feasible combos
    notes = set()
    for r in results:
        for t in TYPES:
            d = r["s1"]["types"][t]
            if d["thin"] or d["starved"]:
                notes.add((r["level_set"], r["size"], r["mix"], t,
                           tuple(d["thin"]), tuple(d["starved"]),
                           tuple(sorted(d["usable"]))))
    if notes:
        w("Bins not used (v4 rule: too few pairs, or too few eligible scenes "
          "for their share)")
        for n in sorted(notes):
            w(f"  {n[0]} n={n[1]} {n[2]} {n[3]}: thin {list(n[4]) or '-'}, "
              f"too few scenes {list(n[5]) or '-'}; trials spread over bins "
              f"{list(n[6])}")
        w("")

    # matching and information
    w("4. MATCHING AND INFORMATION (feasible-count combinations; mean "
      "[min-max] over draws)")
    w("-" * 78)
    if v4:
        w(f"v4 frozen list (one session, {v4['n']} analysed trials): info "
          f"x_sat {v4['info_sat']:.1f}, x_con {v4['info_con']:.1f}, cos "
          f"{v4['cos']:+.3f}")
        w(f"  per analysed trial: x_sat {v4['info_sat'] / v4['n']:.3f}, "
          f"x_con {v4['info_con'] / v4['n']:.3f}")
    w("")
    w(f"{'level set':12s} {'size':>4s} {'mix':15s} "
      f"{'info x_sat (A)':>22s} {'info x_con (A)':>22s} "
      f"{'dI sat':>6s} {'dI con':>6s} {'KS s':>5s} {'KS c':>5s} "
      f"{'cos':>6s} {'VIF':>5s} {'r(L-R)':>6s} {'scenes':>6s}")
    for r in results:
        m = r.get("mets")
        if m is None or not len(m):
            continue
        w(f"{r['level_set']:12s} {r['size']:4d} {r['mix']:15s} "
          f"{describe(m.info_sat_A):>22s} {describe(m.info_con_A):>22s} "
          f"{m.info_sat_rel_diff.max():6.3f} {m.info_con_rel_diff.max():6.3f} "
          f"{m.ks_sat.max():5.3f} {m.ks_con.max():5.3f} "
          f"{m[['cos_A', 'cos_B']].mean().mean():+6.3f} "
          f"{m[['vif_A', 'vif_B']].max().max():5.3f} "
          f"{m[['pearson_lr_A', 'pearson_lr_B']].mean().mean():+6.3f} "
          f"{int(m[['scenes_A', 'scenes_B']].min().min()):6d}")
    w("  info = sum of squared predictors over a set's analysed trials (set "
      "A shown; B is")
    w("  within the dI columns, which give the WORST relative A-B difference "
      "over draws).")
    w("  KS = worst two-sample KS distance over draws, |x_sat| over sat_only+"
      "mixed trials,")
    w("  |x_con| over fog_only+mixed trials. cos = uncentred correlation of "
      "x_sat")
    w("  and x_con, the collinearity a no-intercept model sees; r(L-R) = "
      "Pearson r on the")
    w("  presented left-minus-right predictors. scenes = fewest scenes in "
      "either set.")
    w("")
    if v4:
        w("Each set's information as a share of the whole v4 session")
        for r in results:
            m = r.get("mets")
            if m is None or not len(m):
                continue
            s_ = m[["info_sat_A", "info_sat_B"]].mean().mean()
            c_ = m[["info_con_A", "info_con_B"]].mean().mean()
            w(f"  {r['level_set']:12s} n={r['size']:<4d} {r['mix']:15s} "
              f"x_sat {s_ / v4['info_sat']:5.2f}   x_con "
              f"{c_ / v4['info_con']:5.2f}")
        w("")

    # p65 decision
    if P65_DECISION is not None:
        w_set, wo_set = P65_DECISION
        w("5. DECISION: IS p65 DROPPED?")
        w("-" * 78)
        w("Rule: if two matched sets are feasible without p65, p65 is "
          "dropped.")
        idx = {(r["level_set"], r["size"], r["mix"]): r for r in results}
        drops = []
        for size in BLOCK_SIZES:
            for mix in TYPE_MIXES:
                rw = idx.get((w_set, size, mix))
                ro = idx.get((wo_set, size, mix))
                if rw is None or ro is None:
                    continue
                d = ro["verdict"] == "FEASIBLE"
                drops.append(d)
                line = (f"  n={size:<4d} {mix:15s} without p65: "
                        f"{ro['verdict']:28s} -> "
                        f"{'DROP p65' if d else 'KEEP p65'}")
                mw, mo = rw.get("mets"), ro.get("mets")
                if (mw is not None and len(mw) and mo is not None
                        and len(mo)):
                    ds = (mo[["info_sat_A", "info_sat_B"]].mean().mean()
                          / mw[["info_sat_A", "info_sat_B"]].mean().mean()
                          - 1)
                    dc = (mo[["info_con_A", "info_con_B"]].mean().mean()
                          / mw[["info_con_A", "info_con_B"]].mean().mean()
                          - 1)
                    line += (f"   info change from dropping p65: x_sat "
                             f"{ds:+.1%}, x_con {dc:+.1%}")
                w(line)
        if drops:
            if all(drops):
                w("  => p65 is DROPPED at every size and mix tested.")
            elif not any(drops):
                w("  => p65 is KEPT at every size and mix tested.")
            else:
                w("  => the decision depends on the size and mix; see the "
                  "lines above.")
        w("")

    # checks
    w("6. CHECKS")
    w("-" * 78)
    cf = ck.frame()
    for row in cf.itertuples():
        w(f"  {row.status:7s} [{row.scope}] {row.check}"
          + (f"  ({row.detail})" if row.detail else ""))
    n_f = int((cf.status == "FAIL").sum())
    n_p = int((cf.status == "PASS").sum())
    n_s = int((cf.status == "SKIPPED").sum())
    w("")
    w(f"  {n_p} PASS, {n_f} FAIL, {n_s} SKIPPED")
    w("  Draw checks are listed only for combinations whose counts are "
      "feasible; a check")
    w(f"  PASSES there when at least {REQUIRED_PASS_SHARE:.0%} of draws pass "
      "it. Every draw's checks are")
    w("  in draw_checks.csv. A combination whose counts are infeasible is "
      "reported in")
    w("  section 3 and has no draw checks.")
    w("")
    w("7. NOTES")
    w("-" * 78)
    w("* Feasibility of counts is exact for the bins and quotas as set (v4's "
      "fixed edges,")
    w("  trials spread evenly over usable bins). 'without bin quotas' in "
      "section 3 shows")
    w("  whether a shortfall is the quotas' doing.")
    if DISJOINT == "image":
        w("* DISJOINT = image: stage 1 solved the pair-disjoint problem, which "
          "is necessary")
        w("  but not sufficient; only the draws establish feasibility.")
    if DISJOINT == "scene":
        w("* DISJOINT = scene: stage 1 is one joint problem over the four "
          "types (the scene split")
        w("  is shared), so a type's 'infeasible' means the joint problem is "
          "infeasible, and")
        w("  its shortfall is from the joint maximum placement.")
    w("* A solver time limit is reported as UNDECIDED, never as infeasible.")
    w("* The draws are one way of building matched sets, not the build step "
      "itself; the")
    w("  build step (script 3) can use the same procedure with its own "
      "settings.")
    w("* Consistency repeats reuse a block's own pairs and need no capacity; "
      "they are not")
    w("  drawn here.")
    with open(os.path.join(out_dir, "capacity_report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    return n_f


def bin_supply_rows(pools):
    rows = []
    for ls, pool in pools.items():
        for t in TYPES:
            tp = pool[pool.type == t]
            for b in range(1, len(BIN_EDGES_SD[t]) + 2):
                g = tp[tp.mag_bin == b].groupby("scene_id").size()
                rows.append({"level_set": ls, "type": t, "bin": b,
                             "pairs": int(g.sum()),
                             "scenes_1": int((g >= 1).sum()),
                             "scenes_2": int((g >= 2).sum())})
    return rows


def summary_rows(results):
    rows = []
    for r in results:
        row = {"level_set": r["level_set"], "block_size": r["size"],
               "mix": r["mix"], "verdict": r["verdict"],
               "stage1": r["s1"]["status"],
               "min_cap": r["s1"]["min_cap"]}
        row.update({f"n_{t}": r["counts"][t] for t in TYPES})
        if "pass_share" in r:
            m = r["mets"]
            row.update({
                "draws_built": r["n_built"], "draws_pass": r["n_pass"],
                "draws_fallback": sum(h == "fallback" for h in r["hows"]),
                "info_sat_mean": m[["info_sat_A", "info_sat_B"]].mean().mean(),
                "info_con_mean": m[["info_con_A", "info_con_B"]].mean().mean(),
                "info_sat_rel_diff_max": m.info_sat_rel_diff.max(),
                "info_con_rel_diff_max": m.info_con_rel_diff.max(),
                "info_type_rel_diff_max": m.info_type_worst_rel_diff.max(),
                "ks_sat_max": m.ks_sat.max(), "ks_con_max": m.ks_con.max(),
                "cos_mean": m[["cos_A", "cos_B"]].mean().mean(),
                "cos_diff_max": m.cos_diff.max(),
                "vif_max": m[["vif_A", "vif_B"]].max().max(),
                "pearson_lr_mean": m[["pearson_lr_A",
                                      "pearson_lr_B"]].mean().mean(),
                "scenes_min": int(m[["scenes_A", "scenes_B"]].min().min()),
                "max_per_scene": int(m[["max_per_scene_A",
                                        "max_per_scene_B"]].max().max()),
            })
            for t in TYPES:
                row[f"info_sat_{t}"] = m[[f"info_sat_{t}_A",
                                          f"info_sat_{t}_B"]].mean().mean()
                row[f"info_con_{t}"] = m[[f"info_con_{t}_A",
                                          f"info_con_{t}_B"]].mean().mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    check_settings()
    print("v5 step 1: pair capacity for two matched sets")
    ck = Checks()
    log = []
    p = load_pairs(ck, log)
    check_manifest(p, ck, log)
    v4 = load_v4_baseline(log)
    out_dir = resolve_output_dir()
    os.makedirs(out_dir)
    print(f"  output: {out_dir}")

    funnel, pools = [], {}
    for ls in LEVEL_SETS:
        pools[ls] = build_pool(p, ls, ck, funnel)
    scenes = sorted(p.scene_id.unique())
    bin_rows = bin_supply_rows(pools)

    results, feas_rows, short_rows, draw_rows = [], [], [], []
    for ls in LEVEL_SETS:
        pools_t = {t: pools[ls][pools[ls].type == t].reset_index(drop=True)
                   for t in TYPES}
        for size in BLOCK_SIZES:
            for mix in TYPE_MIXES:
                t1 = time.time()
                r = run_combo(ls, size, mix, pools_t, scenes, out_dir, ck,
                              feas_rows, short_rows, draw_rows)
                results.append(r)
                print(f"  {ls:12s} n={size:<4d} {mix:15s} "
                      f"{r['verdict']:28s} ({time.time() - t1:.1f} s)")

    pd.DataFrame(funnel).to_csv(os.path.join(out_dir, "pool_funnel.csv"),
                                index=False)
    pd.DataFrame(bin_rows).to_csv(os.path.join(out_dir, "pool_by_bin.csv"),
                                  index=False)
    pd.DataFrame(feas_rows).to_csv(
        os.path.join(out_dir, "feasibility_by_type.csv"), index=False)
    pd.DataFrame(short_rows).to_csv(
        os.path.join(out_dir, "shortfall_by_bin.csv"), index=False)
    pd.DataFrame(draw_rows).to_csv(os.path.join(out_dir, "draw_checks.csv"),
                                   index=False)
    summary_rows(results).to_csv(os.path.join(out_dir, "summary.csv"),
                                 index=False)
    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump(settings_dict(), f, indent=2, default=str)
    n_fail = write_report(out_dir, log, ck, pools, funnel, bin_rows,
                          results, v4, t0)
    ck.frame().to_csv(os.path.join(out_dir, "checks.csv"), index=False)
    print(f"  report: {os.path.join(out_dir, 'capacity_report.txt')}")
    print(f"  {n_fail} check(s) FAIL; elapsed {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
