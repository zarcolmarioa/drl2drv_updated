"""
analyse_pair_capacity_v5b.py  --  v5b step 1: the pairs of the v5 stimulus
set, and can they supply two matched sets?

WHAT CHANGED FROM analyse_pair_capacity_v5.py (v5 -> v5b)
    v5 read v4's pair table (all_pairs.csv from analyse_pairs_v4_corrected.py)
    for the v4 stimuli. v5b reads the v5 STIMULUS SET (build_stimulus_set_v5.py:
    47 scenes x 49 cells, three contrast and three colour levels below and
    above the unedited photograph) and MAKES THE PAIRS ITSELF:

      * every unordered pair of images within a scene (47 x C(49, 2) =
        55,272), listed in both directions, saved as all_pairs.csv.gz for
        script 3;
      * the UNEDITED photograph may appear in a pair;
      * the pair TYPE is decided by MEASURED differences (corpus SD), not by
        level labels; the level labels and step distances are carried as
        labels only, because the cells with both features edited sit up to
        0.51 SD from their nominal colour level;
      * the MINIMUM differences are named sets in THRESHOLD_SETS, each run
        through the full feasibility analysis (they replace v5's level sets
        and the p65 decision, which are gone), and a cheap SUPPLY GRID
        reports the pairs and scenes of every type over many minimums;
      * magnitude BINS are quantiles of each type's driver in each threshold
        set's pool (v4's fixed edges were tied to v4's ladders); the bins by
        step distance are reported for reference;
      * STRADDLING pairs (one image below the original, one above, on
        contrast or colour) and pairs with the original are counted;
      * every pair carries the NORMALITY terms: distance from typical (corpus
        median, and the scene's own original), N = dist(B) - dist(A),
        positive = A is the more typical image, combined and per feature.
        The pool's and every draw's leak (N regressed on both direction
        terms, no intercept) is reported, for script 3's limit.

    The matching machinery (stage 1 integer programs, the two-block greedy
    draw, the checks on each draw) is v5's, unchanged; scripts 2 and 3 import
    it from here.

WHAT THIS ANSWERS
    The design gives every participant two image-choice blocks of equal
    length, each with its own set of pairs; which set goes with which block
    is counterbalanced. For each threshold set, block size and type mix, it
    asks whether two sets exist that share no pair, have identical type and
    magnitude-bin counts, matched information for each weight, the same
    scenes, balanced sides, and enough unused pairs for practice and catch.
    A combination is FEASIBLE when stage 1 (exact counts) succeeds, every
    draw is built, and at least REQUIRED_PASS_SHARE of draws pass every
    check.

PAIR TYPES (A shown against B; d_sat = sat_A - sat_B, d_fog = -(con_A -
con_B), so d_fog > 0 means A has the lower contrast)
    sat_only   |d_con| <= CON_TOL_SD           (the same contrast)
    fog_only   |d_sat| <= SAT_TOL_SD           (the same saturation)
    together   both above their tolerance, the lower-contrast image the
               less saturated
    conflict   both above their tolerance, the lower-contrast image the
               more saturated; usable only when balanced (corpus-SD
               balance in [BALANCE_LO, BALANCE_HI]), as v4
    none       both within tolerance
    Each threshold set then requires a minimum |difference| in corpus SD:
    on saturation (sat_only), contrast (fog_only), and both (together,
    conflict).

INPUTS (paths relative to the project folder; run from there)
    MANIFEST_CSV    the v5 stimulus set's stimulus_manifest.csv
    The set's FROZEN.json, next to it (checked when present: the manifest's
    SHA-256 must be the frozen one).
    V4_TRIAL_LIST   optional: the information baseline from the frozen v4
                    list.

OUTPUTS (one new folder per run, never overwritten)
    capacity_report.txt       what was done, what was found, every check
    all_pairs.csv.gz          every pair, both directions, all columns
    supply_grid.csv           pairs and scenes per type over many minimums
    bin_edges.csv             the quantile bin edges of each threshold set
    bins_by_steps.csv         the pools by step distance (reference)
    straddle.csv              straddling pairs and pairs with the original
    normality_pool.csv        the leak of each pool, by type
    summary.csv, feasibility_by_type.csv, shortfall_by_bin.csv,
    pool_by_bin.csv, pool_funnel.csv, draw_checks.csv, checks.csv
    example_sets/             the first draw of each feasible combination
    settings.json             every setting this run used

REQUIRES  numpy, pandas, scipy >= 1.9 (scipy.optimize.milp)
RUN       python analyse_pair_capacity_v5b.py
"""

# ============================================================
# SETTINGS
# ============================================================

# --- Input / output ---
MANIFEST_CSV = "data/stimulus_set_v5/7fd72262ba/stimulus_manifest.csv"
V4_TRIAL_LIST = "data/frozen_trial_list_v4/trial_list.csv"
OUTPUT_ROOT = "data/pair_capacity_v5b"
ON_EXISTING = "new"

# --- The design questions ---
BLOCK_SIZES = [120, 150, 180]
TYPE_MIXES = {
    "v4":             {"sat_only": 1 / 3, "fog_only": 1 / 3,
                       "together": 1 / 6, "conflict": 1 / 6},
    "conflict_heavy": {"sat_only": 0.30, "fog_only": 0.30,
                       "together": 0.15, "conflict": 0.25},
}

# --- The predictor divisors (corpus SD on the stimulus path; v5's) ---
SD_SAT = 39.182
SD_FOG = 9.348
FOG_SIGN = -1.0

# --- Pair types: what counts as "the same" (corpus SD) ---
# SAT_TOL_SD
#   fog_only pairs must match in sat_mean within this: v4's 3 units
#   (3 / 39.182 = 0.0766 SD).
# CON_TOL_SD
#   sat_only pairs must match in contrast_local within this: v5's 0.72
#   units (0.72 / 9.348 = 0.0770 SD), the same SD fraction.
SAT_TOL_SD = 3.0 / SD_SAT
CON_TOL_SD = 0.72 / SD_FOG

# BALANCE_LO / BALANCE_HI / BALANCED_CONFLICT_ONLY
#   A conflict pair is balanced when (|d_sat| / SD_SAT) / (|d_fog| / SD_FOG)
#   lies in [BALANCE_LO, BALANCE_HI] (v4's rule); only balanced conflict
#   pairs are used.
BALANCE_LO = 0.6
BALANCE_HI = 1.7
BALANCED_CONFLICT_ONLY = True

# --- Minimum differences (corpus SD, measured) ---
# THRESHOLD_SETS
#   Each set is run through the full feasibility analysis. Keys:
#     sat_only    |d_sat| >= this, for sat_only pairs
#     fog_only    |d_con| >= this, for fog_only pairs
#     mixed_sat   |d_sat| >= this, for together and conflict pairs
#     mixed_con   |d_con| >= this, for together and conflict pairs
#   One nominal step of the ladders is 1/3 SD; measured differences sit
#   near (not on) multiples of it, so the minimums are set a little below
#   the step they are meant to admit (0.30 admits one step, 0.60 two).
#   "v5units" is v5's rules in SD: 31.1, 4.2, 8 and 3 units.
THRESHOLD_SETS = {
    "v5units": {"sat_only": 31.1 / SD_SAT, "fog_only": 4.2 / SD_FOG,
                "mixed_sat": 8.0 / SD_SAT, "mixed_con": 3.0 / SD_FOG},
    "sd030":   {"sat_only": 0.30, "fog_only": 0.30,
                "mixed_sat": 0.20, "mixed_con": 0.30},
    "sd060":   {"sat_only": 0.60, "fog_only": 0.60,
                "mixed_sat": 0.20, "mixed_con": 0.30},
}

# SUPPLY_GRID
#   Counting only (no feasibility analysis): the unordered pairs and the
#   scenes of each type at every minimum listed, in supply_grid.csv and
#   the report. Mixed pairs are counted over every (mixed_sat, mixed_con).
SUPPLY_GRID = {
    "sat_only": [0.20, 0.30, 0.45, 0.60, 0.79, 1.00, 1.30],
    "fog_only": [0.20, 0.30, 0.45, 0.60, 0.79, 1.00, 1.30],
    "mixed_sat": [0.10, 0.20, 0.30, 0.45],
    "mixed_con": [0.20, 0.30, 0.45, 0.60],
}

# STRADDLE_TOL_SD
#   A pair straddles the original on a feature, by measurement, when one
#   image is more than this below the original and the other more than this
#   above it. The straddle by level label is counted too.
STRADDLE_TOL_SD = 0.02

# --- Magnitude bins (quantiles of each type's driver, per threshold set) ---
#   Driver: |x_sat| for sat_only, |x_con| for the other types. The edges
#   are the quantiles below of the type's unordered pool, rounded to
#   BIN_EDGE_DECIMALS, so they are reproducible; bin_edges.csv lists them.
#   v4 had five bins for sat_only and three for the others.
BIN_QUANTILES = {
    "sat_only": (0.2, 0.4, 0.6, 0.8),
    "fog_only": (1 / 3, 2 / 3),
    "together": (1 / 3, 2 / 3),
    "conflict": (1 / 3, 2 / 3),
}
BIN_EDGE_DECIMALS = 4

# --- Matching (v5's settings, unchanged) ---
DISJOINT = "pair"
SCENE_MATCH = "scene_bin"
MAX_PER_SCENE_PER_TYPE = 2
MAX_CAP_SEARCH = 4
MIN_PAIRS_PER_BIN = 10

# --- Reserved trials per block (drawn from pairs neither set uses) ---
N_PRACTICE_PER_BLOCK = 6
N_CATCH_PER_BLOCK = 10
PRACTICE_TYPES = ("sat_only", "fog_only", "together")
PRACTICE_MAG_QUANTILE = 0.60
CATCH_TYPE = "together"

SPREAD_SCENES = True
BALANCE_ASSIGNMENT = True
TYPE_TRIES = 5
N_DRAWS = 20
RANDOM_SEED = 42
REQUIRED_PASS_SHARE = 0.90
INFO_TOL_TOTAL = 0.05
INFO_TOL_TYPE = 0.10
KS_MAX_D = 0.10
COS_DIFF_MAX = 0.05
VIF_MAX = 1.25
MIN_SCENE_SHARE = 0.85
TYPE_ORDER = ("sat_only", "fog_only", "conflict", "together")

# --- Input checks ---
# EXPECTED_POOL
#   Unordered pair counts by type (before any minimum) from a first run,
#   to catch a different manifest later. None skips the check.
EXPECTED_POOL = None
N_SCENES_EXPECTED = 47
CELLS_PER_SCENE = 49

ILP_TIME_LIMIT = 60
WRITE_EXAMPLE_SETS = True

# --- Constants ---
TYPES = ["sat_only", "fog_only", "together", "conflict"]
NORMALITY_DEFS = ("corpus", "orig")   # distance from the corpus median /
                                      # from the scene's own original


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
        "MANIFEST_CSV": MANIFEST_CSV, "V4_TRIAL_LIST": V4_TRIAL_LIST,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "BLOCK_SIZES": BLOCK_SIZES, "TYPE_MIXES": TYPE_MIXES,
        "SD_SAT": SD_SAT, "SD_FOG": SD_FOG, "FOG_SIGN": FOG_SIGN,
        "SAT_TOL_SD": SAT_TOL_SD, "CON_TOL_SD": CON_TOL_SD,
        "BALANCE_LO": BALANCE_LO, "BALANCE_HI": BALANCE_HI,
        "BALANCED_CONFLICT_ONLY": BALANCED_CONFLICT_ONLY,
        "THRESHOLD_SETS": THRESHOLD_SETS, "SUPPLY_GRID": SUPPLY_GRID,
        "STRADDLE_TOL_SD": STRADDLE_TOL_SD,
        "BIN_QUANTILES": BIN_QUANTILES,
        "BIN_EDGE_DECIMALS": BIN_EDGE_DECIMALS,
        "DISJOINT": DISJOINT, "SCENE_MATCH": SCENE_MATCH,
        "MAX_PER_SCENE_PER_TYPE": MAX_PER_SCENE_PER_TYPE,
        "MAX_CAP_SEARCH": MAX_CAP_SEARCH,
        "MIN_PAIRS_PER_BIN": MIN_PAIRS_PER_BIN,
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
        "N_SCENES_EXPECTED": N_SCENES_EXPECTED,
        "CELLS_PER_SCENE": CELLS_PER_SCENE,
        "SPREAD_SCENES": SPREAD_SCENES,
        "BALANCE_ASSIGNMENT": BALANCE_ASSIGNMENT, "TYPE_TRIES": TYPE_TRIES,
        "ILP_TIME_LIMIT": ILP_TIME_LIMIT,
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
    if set(TYPE_ORDER) != set(TYPES):
        raise ValueError(f"TYPE_ORDER must list exactly {TYPES}")
    if set(BIN_QUANTILES) != set(TYPES):
        raise ValueError(f"BIN_QUANTILES must name exactly {TYPES}")
    keys = {"sat_only", "fog_only", "mixed_sat", "mixed_con"}
    for name, th in THRESHOLD_SETS.items():
        if set(th) != keys:
            raise ValueError(f"THRESHOLD_SETS[{name!r}] must name exactly "
                             f"{sorted(keys)}")
        # a minimum at or below its tolerance would let one pair be two types
        if th["sat_only"] <= SAT_TOL_SD or th["fog_only"] <= CON_TOL_SD:
            raise ValueError(f"THRESHOLD_SETS[{name!r}]: the sat_only and "
                             "fog_only minimums must exceed SAT_TOL_SD and "
                             "CON_TOL_SD")
        if th["mixed_sat"] <= SAT_TOL_SD or th["mixed_con"] <= CON_TOL_SD:
            raise ValueError(f"THRESHOLD_SETS[{name!r}]: the mixed minimums "
                             "must exceed SAT_TOL_SD and CON_TOL_SD")


def fmt_num(x):
    return ("none" if x is None else
            (f"{x:g}" if isinstance(x, float) else str(x)))


def resolve_output_dir():
    """OUTPUT_ROOT/<key settings>__s<hash>, never overwriting a run.

    Adapted from resolve_output_root() in the v4 scripts."""
    blob = json.dumps(settings_dict(), sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode()).hexdigest()[:8]
    name = (f"v5b_disjoint-{DISJOINT}_match-{SCENE_MATCH}"
            f"_cap{MAX_PER_SCENE_PER_TYPE}"
            f"_sets-{'-'.join(THRESHOLD_SETS)}"
            f"_sizes{'-'.join(str(s) for s in BLOCK_SIZES)}"
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
# The stimulus set and its pairs (NEW in v5b)
# ============================================================

# The image columns read from the manifest, and the names they get on each
# side of a pair (<name>_a, <name>_b). sat / fog keep v4's names (fog = the
# measured contrast_local), so d_sat, d_fog, pred_sat, pred_fog are v4's.
SIDE_COLS = {
    "filename": "file", "sat_mean": "sat", "contrast_local": "fog",
    "contrast_level": "contrast_level", "colour_level": "colour_level",
    "contrast_step": "contrast_step", "colour_step": "colour_step",
    "is_original": "is_original", "contrast_param": "contrast_param",
    "comp_k": "comp_k", "colour_k": "colour_k",
    "d_sat_sd": "change_sat", "d_con_sd": "change_con",
    "z_sat": "z_sat", "z_con": "z_con",
    "dist_corpus": "dist_corpus", "dist_corpus_sat": "dist_corpus_sat",
    "dist_corpus_con": "dist_corpus_con", "dist_orig": "dist_orig",
    "dist_orig_sat": "dist_orig_sat", "dist_orig_con": "dist_orig_con",
}
SCENE_COLS = ["scene_id", "category", "group", "indoor"]

# Set by read_manifest: the stimulus set's identity, for the report and
# for scripts 3-5.
STIMULUS_SET = {}


def as_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(ck, log):
    """The v5 stimulus manifest, checked."""
    if not os.path.isfile(MANIFEST_CSV):
        raise FileNotFoundError(
            f"Stimulus manifest not found: {MANIFEST_CSV}\nRun this script "
            "from the project folder, or set MANIFEST_CSV.")
    m = pd.read_csv(MANIFEST_CSV)
    need = SCENE_COLS + list(SIDE_COLS)
    missing = [c for c in need if c not in m.columns]
    ck.add("input", "manifest has the required columns", not missing,
           f"missing {missing}" if missing else f"{len(m.columns)} columns")
    if missing:
        raise ValueError(f"{MANIFEST_CSV} is missing columns {missing}")
    m["is_original"] = as_bool(m["is_original"])
    per = m.groupby("scene_id").size()
    ck.add("input", f"{N_SCENES_EXPECTED} scenes x {CELLS_PER_SCENE} images",
           len(per) == N_SCENES_EXPECTED and (per == CELLS_PER_SCENE).all(),
           f"{len(per)} scenes, {per.min()}-{per.max()} images each")
    ck.add("input", "image file names are unique", m.filename.is_unique,
           f"{len(m):,} rows")
    orig = m[m.is_original]
    ok_o = (orig.scene_id.is_unique and len(orig) == len(per)
            and ((orig.contrast_step == 0) & (orig.colour_step == 0)).all())
    ck.add("input", "one unedited image per scene (contrast and colour step "
           "0)", bool(ok_o), f"{len(orig)} unedited images")
    if "use" in m.columns:
        ck.add("input", "every image may be used in pairs (use = pair)",
               bool((m.use == "pair").all()),
               f"{int((m.use != 'pair').sum())} with another use")
    # the frozen stimulus set
    d = os.path.dirname(MANIFEST_CSV)
    fz = os.path.join(d, "FROZEN.json")
    sha = _sha256_file(MANIFEST_CSV)
    STIMULUS_SET.clear()
    STIMULUS_SET.update({"manifest": MANIFEST_CSV, "manifest_sha256": sha})
    if os.path.isfile(fz):
        with open(fz) as f:
            fr = json.load(f)
        STIMULUS_SET.update({
            "settings_hash": fr.get("settings_hash"),
            "stimuli_sha256": fr.get("stimuli_sha256"),
            "checksums_sha256": fr.get("checksums_sha256")})
        ck.add("input", "the manifest is the frozen stimulus set's "
               "(SHA-256 as in its FROZEN.json)",
               sha == fr.get("manifest_sha256"),
               f"set {fr.get('settings_hash')}, stimuli_sha256 "
               f"{str(fr.get('stimuli_sha256'))[:12]}...")
    else:
        ck.add("input", "the manifest is the frozen stimulus set's",
               "SKIPPED", f"{fz} not found")
    log.append(f"stimulus set    {MANIFEST_CSV}: {len(m):,} images, "
               f"{len(per)} scenes; manifest SHA-256 {sha[:12]}..."
               + (f"; stimuli_sha256 {STIMULUS_SET['stimuli_sha256'][:12]}"
                  "..." if STIMULUS_SET.get("stimuli_sha256") else ""))
    return m


def classify(d_sat, d_fog):
    """Type, |differences| in corpus SD, and balance of ordered pairs, from
    MEASURED differences (vectorised). Types as in the header."""
    zs = np.abs(d_sat) / SD_SAT
    zc = np.abs(d_fog) / SD_FOG
    same_s = zs <= SAT_TOL_SD
    same_c = zc <= CON_TOL_SD
    mixed = np.where((d_fog > 0) == (d_sat > 0), "conflict", "together")
    t = np.select([same_s & same_c, same_c, same_s],
                  ["none", "sat_only", "fog_only"], default=mixed)
    with np.errstate(divide="ignore", invalid="ignore"):
        bal = np.where(zc > 0, zs / zc, np.nan)
    balanced = np.isfinite(bal) & (bal >= BALANCE_LO) & (bal <= BALANCE_HI)
    return t, zs, zc, bal, balanced


def make_pairs(m):
    """Every pair of images within a scene, both directions."""
    out = []
    for sid, g in m.groupby("scene_id", sort=True):
        g = g.sort_values(["contrast_step", "colour_step"], kind="stable")
        n = len(g)
        i, j = np.triu_indices(n, 1)
        ia = np.r_[i, j]
        ib = np.r_[j, i]
        cols = {c: np.repeat(g[c].iloc[0], len(ia)) for c in SCENE_COLS}
        for c, nm in SIDE_COLS.items():
            v = g[c].to_numpy()
            cols[f"{nm}_a"] = v[ia]
            cols[f"{nm}_b"] = v[ib]
        out.append(pd.DataFrame(cols))
    p = pd.concat(out, ignore_index=True)
    p["d_sat"] = p.sat_a - p.sat_b
    p["d_fog"] = FOG_SIGN * (p.fog_a - p.fog_b)
    p["pred_sat"] = -p.d_sat / SD_SAT
    p["pred_fog"] = p.d_fog / SD_FOG
    t, zs, zc, bal, bd = classify(p.d_sat.to_numpy(), p.d_fog.to_numpy())
    p["type"] = t
    p["z_sat"] = zs                   # |d_sat| / SD_SAT  (v4's pair z)
    p["z_fog"] = zc                   # |d_fog| / SD_FOG
    p["balance"] = bal
    p["balanced"] = bd
    # labels only: nominal step distances
    p["d_contrast_steps"] = p.contrast_step_a - p.contrast_step_b
    p["d_colour_steps"] = p.colour_step_a - p.colour_step_b
    # straddling the original, by level label and by measurement
    p["straddle_contrast_level"] = p.contrast_step_a * p.contrast_step_b < 0
    p["straddle_colour_level"] = p.colour_step_a * p.colour_step_b < 0
    tol = STRADDLE_TOL_SD
    p["straddle_contrast_measured"] = (
        ((p.change_con_a < -tol) & (p.change_con_b > tol))
        | ((p.change_con_a > tol) & (p.change_con_b < -tol)))
    p["straddle_colour_measured"] = (
        ((p.change_sat_a < -tol) & (p.change_sat_b > tol))
        | ((p.change_sat_a > tol) & (p.change_sat_b < -tol)))
    p["includes_original"] = p.is_original_a | p.is_original_b
    # normality: positive = A is the more typical image
    for d in NORMALITY_DEFS:
        for suf in ("", "_sat", "_con"):
            p[f"N_{d}{suf}"] = p[f"dist_{d}{suf}_b"] - p[f"dist_{d}{suf}_a"]
    return p


def load_pairs(ck, log):
    """The ordered pair table of the v5 stimulus set (both directions),
    checked. Same name and signature as v5's, so scripts 2 and 3 import
    it unchanged."""
    m = read_manifest(ck, log)
    p = make_pairs(m)
    n_sc = m.scene_id.nunique()
    n_un = n_sc * CELLS_PER_SCENE * (CELLS_PER_SCENE - 1) // 2
    ck.add("input", "every pair of images within a scene, both directions",
           len(p) == 2 * n_un, f"{len(p):,} ordered pairs ({n_un:,} "
           "unordered)")
    log.append(f"pair table      made here: {len(p):,} ordered pairs "
               f"({len(p) // 2:,} unordered), {p.scene_id.nunique()} scenes")

    # every ordered pair has its reverse, with negated differences
    fwd = p.set_index(["file_a", "file_b"])[["d_sat", "d_fog", "type",
                                             "N_corpus", "N_orig"]]
    rev = p.set_index(["file_b", "file_a"])[["d_sat", "d_fog", "type",
                                             "N_corpus", "N_orig"]]
    rev.index.names = ["file_a", "file_b"]
    j = fwd.join(rev, rsuffix="_r", how="left")
    ok = (j.d_sat_r.notna().all()
          and np.allclose(j.d_sat, -j.d_sat_r, atol=1e-9)
          and np.allclose(j.d_fog, -j.d_fog_r, atol=1e-9)
          and np.allclose(j.N_corpus, -j.N_corpus_r, atol=1e-9)
          and np.allclose(j.N_orig, -j.N_orig_r, atol=1e-9)
          and (j.type == j.type_r).all())
    ck.add("input", "every pair is listed in both directions, with negated "
           "differences and normality terms and the same type", bool(ok),
           f"{int(j.d_sat_r.isna().sum())} without a reverse")
    ok_x = (np.allclose(p.pred_sat, -p.d_sat / SD_SAT, atol=1e-12)
            and np.allclose(p.pred_fog, p.d_fog / SD_FOG, atol=1e-12))
    ck.add("input", "pred_sat = -d_sat/SD_SAT and pred_fog = d_fog/SD_FOG",
           bool(ok_x), f"SD_SAT {SD_SAT}, SD_FOG {SD_FOG}")
    # the manifest's own changes agree with the features
    sd_ok = np.allclose((p.change_sat_a - p.change_sat_b) * SD_SAT,
                        p.d_sat, atol=2e-3)
    ck.add("input", "the manifest's changes from the original agree with "
           "its features (d_sat)", bool(sd_ok))

    un = p[p.file_a < p.file_b]
    counts = un.type.value_counts().to_dict()
    counts["balanced_conflict"] = int(((un.type == "conflict")
                                       & un.balanced).sum())
    counts["total"] = len(un)
    log.append("pairs by type   " + ", ".join(
        f"{k} {counts.get(k, 0):,}" for k in
        ["sat_only", "fog_only", "together", "conflict",
         "balanced_conflict", "none", "total"]) + " (unordered, before any "
        "minimum)")
    if EXPECTED_POOL:
        bad = {k: (counts.get(k, 0), v) for k, v in EXPECTED_POOL.items()
               if counts.get(k, 0) != v}
        ck.add("input", "pair counts match EXPECTED_POOL", not bad,
               "all match" if not bad else f"got vs expected: {bad}")
    else:
        ck.add("input", "pair counts match EXPECTED_POOL", "SKIPPED",
               "EXPECTED_POOL is None")
    return p


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

# BIN_EDGES[threshold set][type]: filled by build_pool, written to
# bin_edges.csv.
BIN_EDGES = {}


def n_bins(t):
    return len(BIN_QUANTILES[t]) + 1


def build_pool(p, threshold_set, ck, funnel):
    """Unordered, rule-filtered pool for one threshold set.

    One row per pair, oriented so that A is the image the hypothesis
    predicts by the type's driver (x_sat > 0 for sat_only, x_con > 0 for
    the rest), with x_sat = pred_sat and x_con = pred_fog on that row, and
    the normality terms N_* of that orientation. Magnitude bins are the
    BIN_QUANTILES of each type's driver in this pool."""
    th = THRESHOLD_SETS[threshold_set]
    q = p[p.type.isin(TYPES)]
    driver_pos = np.where(q.type == "sat_only", q.pred_sat > 0,
                          q.pred_fog > 0)
    q = q[driver_pos].copy()
    keep = []
    for t in TYPES:
        steps = []
        s = q[q.type == t]
        steps.append(("1 typed by measured differences", len(s)))
        if t == "sat_only":
            s = s[s.z_sat >= th["sat_only"] - EPS]
            steps.append((f"2 |d_sat| >= {th['sat_only']:.3f} SD", len(s)))
        elif t == "fog_only":
            s = s[s.z_fog >= th["fog_only"] - EPS]
            steps.append((f"2 |d_con| >= {th['fog_only']:.3f} SD", len(s)))
        else:
            s = s[(s.z_sat >= th["mixed_sat"] - EPS)
                  & (s.z_fog >= th["mixed_con"] - EPS)]
            steps.append((f"2 |d_sat| >= {th['mixed_sat']:.3f} and |d_con| "
                          f">= {th['mixed_con']:.3f} SD", len(s)))
            if t == "conflict" and BALANCED_CONFLICT_ONLY:
                s = s[s.balanced.astype(bool)]
                steps.append(("3 balanced", len(s)))
        steps.append(("9 scenes", s.scene_id.nunique()))
        for step, n in steps:
            funnel.append({"threshold_set": threshold_set, "type": t,
                           "step": step, "unordered_pairs": n})
        keep.append(s)
    pool = pd.concat(keep, ignore_index=True)
    pool["threshold_set"] = threshold_set
    pool["x_sat"] = pool.pred_sat
    pool["x_con"] = pool.pred_fog
    drv = np.where(pool.type == "sat_only", pool.x_sat.abs(),
                   pool.x_con.abs())
    pool["driver_sd"] = drv
    pool["driver_steps"] = np.where(pool.type == "sat_only",
                                    pool.d_colour_steps.abs(),
                                    pool.d_contrast_steps.abs())
    pool["mag_bin"] = 0
    edges = {}
    for t in TYPES:
        msk = (pool.type == t).to_numpy()
        e = (np.round(np.quantile(drv[msk], BIN_QUANTILES[t]),
                      BIN_EDGE_DECIMALS) if msk.any()
             else np.full(len(BIN_QUANTILES[t]), np.nan))
        edges[t] = [float(v) for v in e]
        pool.loc[msk, "mag_bin"] = np.digitize(drv[msk], e) + 1
    BIN_EDGES[threshold_set] = edges
    pool.attrs["bin_edges"] = edges
    pool["key"] = keys_of(pool)
    pool = pool.sort_values(["type", "scene_id", "file_a", "file_b"],
                            kind="stable").reset_index(drop=True)

    # checks on the pool
    dup = pool.key.duplicated().sum()
    ck.add(threshold_set, "each unordered pair appears once in the pool",
           dup == 0, f"{len(pool):,} pairs, {dup} duplicates")
    ok = all(np.all(np.diff(edges[t]) >= 0) for t in TYPES
             if np.isfinite(edges[t]).all())
    ck.add(threshold_set, "bin edges are in order", bool(ok),
           "; ".join(f"{t} {edges[t]}" for t in TYPES))
    return pool


def leak(N, xs, xc, terms=("sat", "con")):
    """The normality leak of a set of oriented pairs: N regressed on the
    direction terms together, no intercept (N ~ b_sat x_sat + b_con x_con),
    and the uncentred correlations of N with each term. `terms` = ("sat",)
    or ("con",) fits one term only, for a type in which the other is ~0
    (sat_only, fog_only), where a joint fit is meaningless."""
    N, xs, xc = (np.asarray(v, float) for v in (N, xs, xc))
    out = {"b_sat": np.nan, "b_con": np.nan, "cos_sat": np.nan,
           "cos_con": np.nan}
    if len(N) < 2:
        return out
    cols = {"sat": xs, "con": xc}
    X = np.column_stack([cols[t] for t in terms])
    b, *_ = np.linalg.lstsq(X, N, rcond=None)
    nn = np.sqrt((N ** 2).sum())
    for t, v in zip(terms, b):
        out[f"b_{t}"] = float(v)
    if nn > 0:
        out["cos_sat"] = float((N * xs).sum() / (nn * np.sqrt((xs ** 2).sum())))
        out["cos_con"] = float((N * xc).sum() / (nn * np.sqrt((xc ** 2).sum())))
    return out


def supply_grid(p):
    """Pairs and scenes of every type over SUPPLY_GRID's minimums
    (counting only), with the nominal step distances as labels."""
    q = p[p.type.isin(TYPES)]
    q = q[np.where(q.type == "sat_only", q.pred_sat > 0,
                   q.pred_fog > 0)]
    rows = []

    def row(t, ms, mc, s, steps_col):
        per = s.groupby("scene_id").size()
        st = s[steps_col].abs().value_counts().to_dict()
        r = {"type": t, "min_sat_sd": ms, "min_con_sd": mc,
             "pairs": len(s), "scenes": int((per >= 1).sum()),
             "scenes_ge4": int((per >= 4).sum()),
             "median_pairs_per_scene": float(per.median()) if len(per)
             else 0.0}
        for k in range(0, 7):
            r[f"steps_{k}"] = int(st.get(k, 0))
        rows.append(r)

    s0 = q[q.type == "sat_only"]
    for ms in SUPPLY_GRID["sat_only"]:
        row("sat_only", ms, np.nan, s0[s0.z_sat >= ms - EPS],
            "d_colour_steps")
    f0 = q[q.type == "fog_only"]
    for mc in SUPPLY_GRID["fog_only"]:
        row("fog_only", np.nan, mc, f0[f0.z_fog >= mc - EPS],
            "d_contrast_steps")
    for t in ("together", "conflict"):
        x0 = q[q.type == t]
        if t == "conflict" and BALANCED_CONFLICT_ONLY:
            x0 = x0[x0.balanced.astype(bool)]
        for ms in SUPPLY_GRID["mixed_sat"]:
            for mc in SUPPLY_GRID["mixed_con"]:
                row(t, ms, mc, x0[(x0.z_sat >= ms - EPS)
                                  & (x0.z_fog >= mc - EPS)],
                    "d_contrast_steps")
    return pd.DataFrame(rows)


def straddle_rows(pools):
    rows = []
    for ts, pool in pools.items():
        for t in TYPES + ["all"]:
            s = pool if t == "all" else pool[pool.type == t]
            rows.append({
                "threshold_set": ts, "type": t, "pairs": len(s),
                "straddle_contrast_level": int(s.straddle_contrast_level
                                               .sum()),
                "straddle_contrast_measured": int(
                    s.straddle_contrast_measured.sum()),
                "straddle_colour_level": int(s.straddle_colour_level.sum()),
                "straddle_colour_measured": int(
                    s.straddle_colour_measured.sum()),
                "includes_original": int(s.includes_original.sum()),
                "includes_original_as_A": int(s.is_original_a.sum()),
                "includes_original_as_B": int(s.is_original_b.sum())})
    return pd.DataFrame(rows)


def normality_pool_rows(pools):
    rows = []
    for ts, pool in pools.items():
        for t in TYPES + ["all"]:
            s = pool if t == "all" else pool[pool.type == t]
            for d in NORMALITY_DEFS:
                r = {"threshold_set": ts, "type": t, "definition": d,
                     "pairs": len(s),
                     "mean_N": float(s[f"N_{d}"].mean()) if len(s)
                     else np.nan}
                terms = {"sat_only": ("sat",), "fog_only": ("con",)}.get(
                    t, ("sat", "con"))
                r.update(leak(s[f"N_{d}"], s.x_sat, s.x_con, terms))
                rows.append(r)
    return pd.DataFrame(rows)


def bins_by_steps_rows(pools):
    rows = []
    for ts, pool in pools.items():
        g = (pool.groupby(["type", "mag_bin", "driver_steps"]).size()
             .rename("pairs").reset_index())
        g.insert(0, "threshold_set", ts)
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ============================================================
# Bins and quotas
# ============================================================

def usable_bins(tp, n_trials, cap):
    """v4's assign_bins() rules, per block: a bin is usable if it holds at
    least MIN_PAIRS_PER_BIN unordered pairs AND its eligible scenes can
    supply its share under the cap; dropping a bin raises the others'
    shares, so the second rule is iterated. Under scene_bin matching a
    scene is eligible only with >= 2 pairs in the bin (one per block)."""
    nb = n_bins(tp.type.iloc[0]) if len(tp) else 0
    counts = {b: int((tp.mag_bin == b).sum()) for b in range(1, nb + 1)}
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
    th = THRESHOLD_SETS[both.threshold_set.iloc[0]]
    s = both[both.type == "sat_only"]
    chk["sat_only: |d_sat| >= its minimum"] = bool(
        (s.z_sat >= th["sat_only"] - EPS).all())
    chk["sat_only: |d_con| <= CON_TOL_SD"] = bool(
        (s.z_fog <= CON_TOL_SD + EPS).all())
    f = both[both.type == "fog_only"]
    chk["fog_only: |d_con| >= its minimum"] = bool(
        (f.z_fog >= th["fog_only"] - EPS).all())
    chk["fog_only: |d_sat| <= SAT_TOL_SD"] = bool(
        (f.z_sat <= SAT_TOL_SD + EPS).all())
    mx = both[both.type.isin(["together", "conflict"])]
    chk["together, conflict: both differences >= their minimums"] = bool(
        ((mx.z_sat >= th["mixed_sat"] - EPS)
         & (mx.z_fog >= th["mixed_con"] - EPS)).all())
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
        # the normality leak of the analysed trials (reported here; limited
        # by script 3)
        for d in NORMALITY_DEFS:
            lk = leak(m[f"N_{d}"], xs, xc)
            for k, v in lk.items():
                met[f"leak_{d}_{k}_{blk}"] = v
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

def run_combo(threshold_set, size, mix_name, pools_t, scenes, out_dir, ck,
              feas_rows, short_rows, draw_rows):
    combo = f"{threshold_set}|{size}|{mix_name}"
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
            "threshold_set": threshold_set, "block_size": size, "mix": mix_name,
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
                "threshold_set": threshold_set, "block_size": size, "mix": mix_name,
                "type": t, "bin": b, "quota_per_block": d["quotas"][b],
                "placed_per_block": d["placed"].get(b, 0),
                "shortfall_per_block": d["shortfall"].get(b, 0),
                "pairs_in_bin": d["bin_counts"].get(b, 0),
                "eligible_scenes": d["eligible_scenes"].get(b, 0)})

    res = {"threshold_set": threshold_set, "size": size, "mix": mix_name,
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
            draw_rows.append({"threshold_set": threshold_set, "block_size": size,
                              "mix": mix_name, "draw": d, "built": how,
                              "all_checks_pass": False})
            continue
        met, chk = evaluate(sets, counts, s1, n_allowed)
        all_ok = all(v for v in chk.values() if v is not None)
        mets.append(met)
        chks.append(chk)
        row = {"threshold_set": threshold_set, "block_size": size, "mix": mix_name,
               "draw": d, "built": how, "all_checks_pass": all_ok}
        row.update(met)
        row.update({f"CHECK {k}": ("N/A" if v is None else
                                   ("PASS" if v else "FAIL"))
                    for k, v in chk.items()})
        draw_rows.append(row)
        if d == 0 and WRITE_EXAMPLE_SETS:
            ex = os.path.join(out_dir, "example_sets",
                              f"{threshold_set}_n{size}_{mix_name}")
            os.makedirs(ex, exist_ok=True)
            cols = ["block_set", "role", "trial_type", "couple", "scene_id",
                    "mag_bin", "file_a", "file_b", "a_is_left", "x_sat",
                    "x_con", "d_sat", "d_fog", "contrast_level_a",
                    "colour_level_a", "contrast_level_b", "colour_level_b",
                    "balanced", "includes_original", "N_corpus", "N_orig"]
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
    scope = f"{threshold_set} n={size} {mix_name}"
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
                 v4, t0, extra):
    L = []
    w = L.append
    w("=" * 78)
    w("v5b STEP 1: THE PAIRS OF THE v5 STIMULUS SET, AND CAPACITY FOR TWO "
      "MATCHED SETS")
    w("analyse_pair_capacity_v5b.py")
    w("=" * 78)
    w(f"run finished  {time.strftime('%Y-%m-%d %H:%M:%S')}   "
      f"elapsed {time.time() - t0:.1f} s")
    w(f"output        {out_dir}")
    w("")
    w("1. WHAT THIS RUN DID")
    w("-" * 78)
    w("It made every pair of images within each scene of the v5 stimulus "
      "set, typed")
    w("each by its measured differences, and for every threshold set x "
      "block size x")
    w("type mix asked whether two sets")
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
    w(f"  same saturation (fog_only)       |d_sat| <= {SAT_TOL_SD:.4f} SD "
      f"({SAT_TOL_SD * SD_SAT:g} units)")
    w(f"  same contrast (sat_only)         |d_con| <= {CON_TOL_SD:.4f} SD "
      f"({CON_TOL_SD * SD_FOG:g} units)")
    w(f"  conflict balance                 [{BALANCE_LO}, {BALANCE_HI}] "
      "(corpus SD)")
    w("  threshold sets (minimum |difference|, corpus SD; one nominal step "
      "= 0.333 SD)")
    for k, th in THRESHOLD_SETS.items():
        w(f"    {k:10s} sat_only {th['sat_only']:.3f}  fog_only "
          f"{th['fog_only']:.3f}  together/conflict sat {th['mixed_sat']:.3f}"
          f" con {th['mixed_con']:.3f}")
    w(f"  disjoint                         {DISJOINT}")
    w(f"  scene matching                   {SCENE_MATCH}")
    w(f"  cap per scene per type per block {MAX_PER_SCENE_PER_TYPE}")
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
    for ls in THRESHOLD_SETS:
        w(f"[{ls}]")
        for t in TYPES:
            sub = fu[(fu.threshold_set == ls) & (fu.type == t)]
            w(f"  {t:9s} " + " -> ".join(
                f"{r.step.split(' ', 1)[-1]}: {r.unordered_pairs:,}"
                for r in sub.itertuples()))
    w("")
    w("Supply over many minimums (unordered pairs / scenes; mixed types at "
      "the minimum")
    w("on saturation x contrast). Full table, with step labels: "
      "supply_grid.csv")
    g = extra["supply"]
    for t in ("sat_only", "fog_only"):
        col = "min_sat_sd" if t == "sat_only" else "min_con_sd"
        sub = g[g.type == t]
        w(f"  {t:9s} " + "  ".join(f">={getattr(r, col):.2f}: "
                                    f"{r.pairs:,}/{r.scenes}"
                                    for r in sub.itertuples()))
    for t in ("together", "conflict"):
        sub = g[g.type == t]
        w(f"  {t:9s} (rows: min sat; columns: min con "
          f"{', '.join(f'{v:.2f}' for v in SUPPLY_GRID['mixed_con'])})")
        for ms in SUPPLY_GRID["mixed_sat"]:
            ss = sub[np.isclose(sub.min_sat_sd, ms)]
            w(f"    >={ms:.2f}  " + "  ".join(f"{r.pairs:>6,}/{r.scenes:<2d}"
                                             for r in ss.itertuples()))
    w("")
    w("Bin edges (quantiles of each type's driver, corpus SD; bins by step "
      "distance:")
    w("bins_by_steps.csv)")
    for ts in THRESHOLD_SETS:
        w(f"  [{ts}] " + "; ".join(f"{t} {BIN_EDGES[ts][t]}" for t in TYPES))
    w("")
    w("Supply by magnitude bin: pairs / scenes with >= 1 pair / scenes with "
      ">= 2 pairs")
    br = pd.DataFrame(bin_rows)
    for ls in THRESHOLD_SETS:
        w(f"[{ls}]")
        for t in TYPES:
            sub = br[(br.threshold_set == ls) & (br.type == t)]
            w(f"  {t:9s} " + "  ".join(
                f"bin{r.bin}: {r.pairs}/{r.scenes_1}/{r.scenes_2}"
                for r in sub.itertuples()))
    w("")

    # feasibility
    w("3. FEASIBILITY")
    w("-" * 78)
    w("verdict per threshold set x size x mix. Stage 1 = exact counts; draws =")
    w("share of draws passing every check; fb = draws that needed the")
    w("stage-1 allocation because the greedy draw stalled.")
    w("")
    w(f"{'thresholds':12s} {'size':>4s} {'mix':15s} {'per-type counts':22s} "
      f"{'stage 1':10s} {'draws':>6s} {'fb':>3s}  verdict")
    for r in results:
        c = r["counts"]
        cs = "/".join(str(c[t]) for t in TYPES)
        ds = (f"{r['pass_share']:.0%}" if "pass_share" in r else "-")
        fb = (str(sum(h == "fallback" for h in r["hows"]))
              if "hows" in r else "-")
        w(f"{r['threshold_set']:12s} {r['size']:4d} {r['mix']:15s} {cs:22s} "
          f"{r['s1']['status']:10s} {ds:>6s} {fb:>3s}  {r['verdict']}")
    w("  (counts are sat_only/fog_only/together/conflict per block)")
    w("")
    bad = [r for r in results if r["verdict"] != "FEASIBLE"]
    if bad:
        w("Where the shortfall is")
        for r in bad:
            w(f"  {r['threshold_set']} n={r['size']} {r['mix']}: "
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
                notes.add((r["threshold_set"], r["size"], r["mix"], t,
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
    w(f"{'thresholds':12s} {'size':>4s} {'mix':15s} "
      f"{'info x_sat (A)':>22s} {'info x_con (A)':>22s} "
      f"{'dI sat':>6s} {'dI con':>6s} {'KS s':>5s} {'KS c':>5s} "
      f"{'cos':>6s} {'VIF':>5s} {'r(L-R)':>6s} {'scenes':>6s}")
    for r in results:
        m = r.get("mets")
        if m is None or not len(m):
            continue
        w(f"{r['threshold_set']:12s} {r['size']:4d} {r['mix']:15s} "
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
            w(f"  {r['threshold_set']:12s} n={r['size']:<4d} {r['mix']:15s} "
              f"x_sat {s_ / v4['info_sat']:5.2f}   x_con "
              f"{c_ / v4['info_con']:5.2f}")
        w("")

    w("Normality leak of the draws' analysed trials (mean [worst |b|] over "
      "draws, both sets):")
    w("N = dist(B) - dist(A), regressed on x_sat and x_con together, no "
      "intercept. Script 3")
    w("limits |b|; here it is only reported.")
    for r in results:
        m = r.get("mets")
        if m is None or not len(m):
            continue
        parts = []
        for dname in NORMALITY_DEFS:
            for k in ("b_sat", "b_con"):
                v = m[[f"leak_{dname}_{k}_A", f"leak_{dname}_{k}_B"]]
                parts.append(f"{dname} {k} {v.mean().mean():+.3f} "
                             f"[{v.abs().max().max():.3f}]")
        w(f"  {r['threshold_set']:10s} n={r['size']:<4d} {r['mix']:15s} "
          + "  ".join(parts))
    w("")

    w("5. STRADDLING PAIRS, THE ORIGINAL, AND NORMALITY IN THE POOLS")
    w("-" * 78)
    w("Pairs (unordered) whose images lie on both sides of the original, by "
      "level label and")
    w(f"by measurement (each image > {STRADDLE_TOL_SD} SD from the original, "
      "opposite sides), and")
    w("pairs that include the unedited photograph.")
    sr = extra["straddle"]
    w(f"  {'set':8s} {'type':9s} {'pairs':>7s} {'con lvl':>8s} "
      f"{'con meas':>9s} {'col lvl':>8s} {'col meas':>9s} {'original':>9s}")
    for r in sr.itertuples():
        w(f"  {r.threshold_set:8s} {r.type:9s} {r.pairs:7,d} "
          f"{r.straddle_contrast_level:8,d} {r.straddle_contrast_measured:9,d}"
          f" {r.straddle_colour_level:8,d} {r.straddle_colour_measured:9,d} "
          f"{r.includes_original:9,d}")
    w("")
    w("Normality in each pool: N regressed on x_sat and x_con (no "
      "intercept), and its")
    w("uncentred correlation with each; sat_only on x_sat alone, fog_only "
      "on x_con alone")
    w("(the other term is ~0 there). corpus = distance from the corpus "
      "median; orig =")
    w("from the scene's own original; positive N = A, the predicted image, "
      "is more typical)")
    nr = extra["normality"]
    w(f"  {'set':8s} {'type':9s} {'def':6s} {'b_sat':>7s} {'b_con':>7s} "
      f"{'cos_sat':>8s} {'cos_con':>8s} {'mean N':>7s}")
    for r in nr.itertuples():
        bs = "" if not np.isfinite(r.b_sat) else f"{r.b_sat:+7.3f}"
        bc = "" if not np.isfinite(r.b_con) else f"{r.b_con:+7.3f}"
        w(f"  {r.threshold_set:8s} {r.type:9s} {r.definition:6s} "
          f"{bs:>7s} {bc:>7s} {r.cos_sat:+8.3f} "
          f"{r.cos_con:+8.3f} {r.mean_N:+7.3f}")
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
    w("* Feasibility of counts is exact for the bins and quotas as set "
      "(quantile edges,")
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
            for b in range(1, n_bins(t) + 1):
                g = tp[tp.mag_bin == b].groupby("scene_id").size()
                rows.append({"threshold_set": ls, "type": t, "bin": b,
                             "pairs": int(g.sum()),
                             "scenes_1": int((g >= 1).sum()),
                             "scenes_2": int((g >= 2).sum())})
    return rows


def summary_rows(results):
    rows = []
    for r in results:
        row = {"threshold_set": r["threshold_set"], "block_size": r["size"],
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
    print("v5b step 1: the pairs of the v5 stimulus set, and capacity for "
          "two matched sets")
    ck = Checks()
    log = []
    p = load_pairs(ck, log)
    v4 = load_v4_baseline(log)
    out_dir = resolve_output_dir()
    os.makedirs(out_dir)
    print(f"  output: {out_dir}")
    p.to_csv(os.path.join(out_dir, "all_pairs.csv.gz"), index=False,
             float_format="%.8g", compression="gzip")

    supply = supply_grid(p)
    funnel, pools = [], {}
    for ts in THRESHOLD_SETS:
        pools[ts] = build_pool(p, ts, ck, funnel)
    scenes = sorted(p.scene_id.unique())
    bin_rows = bin_supply_rows(pools)
    extra = {"supply": supply, "straddle": straddle_rows(pools),
             "normality": normality_pool_rows(pools)}

    results, feas_rows, short_rows, draw_rows = [], [], [], []
    for ts in THRESHOLD_SETS:
        pools_t = {t: pools[ts][pools[ts].type == t].reset_index(drop=True)
                   for t in TYPES}
        for size in BLOCK_SIZES:
            for mix in TYPE_MIXES:
                t1 = time.time()
                r = run_combo(ts, size, mix, pools_t, scenes, out_dir, ck,
                              feas_rows, short_rows, draw_rows)
                results.append(r)
                print(f"  {ts:10s} n={size:<4d} {mix:15s} "
                      f"{r['verdict']:28s} ({time.time() - t1:.1f} s)")

    supply.to_csv(os.path.join(out_dir, "supply_grid.csv"), index=False,
                  float_format="%.4g")
    pd.DataFrame([{"threshold_set": ts, "type": t, "bin_edges_sd":
                   json.dumps(BIN_EDGES[ts][t]),
                   "quantiles": json.dumps(list(BIN_QUANTILES[t]))}
                  for ts in THRESHOLD_SETS for t in TYPES]).to_csv(
        os.path.join(out_dir, "bin_edges.csv"), index=False)
    bins_by_steps_rows(pools).to_csv(
        os.path.join(out_dir, "bins_by_steps.csv"), index=False)
    extra["straddle"].to_csv(os.path.join(out_dir, "straddle.csv"),
                             index=False)
    extra["normality"].to_csv(os.path.join(out_dir, "normality_pool.csv"),
                              index=False, float_format="%.5g")
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
        json.dump({**settings_dict(), "stimulus_set": STIMULUS_SET}, f,
                  indent=2, default=str)
    n_fail = write_report(out_dir, log, ck, pools, funnel, bin_rows,
                          results, v4, t0, extra)
    ck.frame().to_csv(os.path.join(out_dir, "checks.csv"), index=False)
    print(f"  report: {os.path.join(out_dir, 'capacity_report.txt')}")
    print(f"  {n_fail} check(s) FAIL; elapsed {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
