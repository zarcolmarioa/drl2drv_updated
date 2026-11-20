"""
simulate_c1_v5c.py  --  v5c power analysis, script 3: C1 under the v5c design,
                        plus the paired test at further strengths of
                        pair-specific preferences (a copy of
                        simulate_c1_v5b.py, v5b script 8, changed only where
                        listed below)

WHAT CHANGED FROM simulate_c1_v5b.py (v5b -> v5c)
    Only the inputs, the names and the comparison section. The C1 test, the
    attention gate, the secondary check, recruitment, every scenario,
    Part B, the person model, the numbers of studies and the seeds
    (STREAM_KEY "c1_v5b", kept) are v5b's.
      * INPUTS: script 2 (simulate_power_paired_v5c_adj.py) is imported as
        "script 7", with its settings pointing to v5c (v5c_180, script 1's
        thresholds); its run (PAIRED_RUN_DIR) gives the adjusted thresholds
        (Part B's sign test) and the SD 1.5 rows of Part B. The package is
        v5c_180 only; comparisons of 150 with 180 are "not applicable".
      * REGRESSION: "python simulate_c1_v5c.py --regression" runs this same
        code on v5b_180 (script 2 in its regression mode, script 2's
        regression run) and requires every 180-trial row of v5b's c1_v5b
        run (..._sed9376ba: c1.csv, c1_attention.csv, c1_replicates.csv,
        paired_pair_sd.csv) to be reproduced: rates identical, other values
        within 1e-12. A full rerun at 180 trials is fast, so the "slice" is
        every 180-trial cell. A normal run first looks for a passing
        regression of this very script file and runs it if there is none;
        a FAIL stops before the v5c run.
      * A COMPARISON section (report and comparison.csv): v5c next to
        v5b's registered results (c1_v5b, 180 trials), changes larger than
        COMPARE_FLAG flagged.
      * A PREFLIGHT lists every input path; N_WORKERS = None uses all CPUs.
      * Output under data/power_v5c/c1_v5c/, v5b's folder naming. Script
        numbers in the text are v5b's (script 7 = the paired script, v5c
        script 2); the report says so.

---- simulate_c1_v5b.py's description follows (v5b -> v5c: v5c_180 and
     simulate_power_paired_v5c_adj.py for v5b's packages and script 7) ----

simulate_c1_v5b.py  --  v5b power analysis, script 8: C1 under the v5b design,
                        plus the paired test at further strengths of
                        pair-specific preferences

WHAT CHANGED FROM simulate_c1_v5.py (v5 -> v5b)
    The C1 test, the attention gate, the secondary check, recruitment, the
    scenarios, Part B, the checks, the report and the figures are v5's.
    Changed:
      * Script 7 is now simulate_power_paired_v5b_adj.py (imported) with
        the REGISTERED per-person model: the strategy-adjusted model (the
        side model plus N_corpus and N_orig per block, Firth, no intercept;
        side and N terms are nuisance terms, never results). Its thresholds
        are read from script 7's adjusted run (PAIRED_RUN_DIR,
        thresholds_adjusted.csv); they are used by Part B's sign test only.
        The frozen v5b packages are read and verified through script 7's
        loader.
      * THE C1 STRATEGY. v5 gave a quarter of each group +0.5 on both
        features towards the more edited-looking image (v4's C1 strategy).
        In v5b the strategies act through N (dist(B) - dist(A), corpus SD):
        in the main rows a quarter of each group "pick the less typical"
        (-0.5 x N_corpus) and a separate quarter "pick the more edited"
        (-0.5 x N_orig), in every block. The opposite direction (more
        typical / less edited, +0.5) is a sensitivity row set. "Realistic"
        rows add script 7's default nuisances (with these two quarters in
        the main-row direction, and script 7's fatigue).
      * A NEW C1 THREAT: an episode-dependent strategy across the groups.
        The report group uses -k x N in its derealization block (its
        ordinary block: none) and the comparison group +k x N in its
        ordinary block (its expectation block: none), k = 0.25 / 0.5 / 1.0,
        both definitions of N, no effect. Reported: the false-claim rate per
        feature and for either feature, before and after the attention
        check.
      * Part B (the paired test at pair-specific SD 1.0 and 2.0) is
        unchanged, now with the adjusted model; script 7's SD 1.5 rows are
        read from its adjusted run.
      * PARALLEL: every scenario x size is one task in its own process
        (N_WORKERS, one thread each), with named random streams; results
        are gathered in a fixed order, so they do not depend on N_WORKERS
        (checked by rerunning one task in this process). The comparison
        group's pool is rebuilt in each task from its own stream (the same
        pool wherever it is used).
      * Relative paths only; a plain-language summary (summary.txt and the
        report's first section).

WHAT THIS ANSWERS
  PART A -- C1. Does the report group's derealization-block weight run
    higher than the comparison group's ordinary-block weight?
      * Test: one-sided Mann-Whitney per feature (report group higher),
        Holm across the two features (v4's _c1_test).
      * First the attention gate: are the comparison group less attentive?
        One-sided Mann-Whitney on catch failures over the session and on
        the share of responses under 300 ms (comparison group higher),
        Holm across the two (v4's _c1_gate). If it fires, C1 is
        inconclusive.
      * Secondary check: the report group's ORDINARY-block weights against
        the comparison group's ordinary-block weights, same test.
      * Recruitment as registered: each group fills the next slots of its
        rows of the frozen assignment table; anyone with 3 or more catch
        failures of 10 is excluded and uses up the slot; recruitment
        continues to the target. Report group N = 10-20 analysable;
        comparison group 20 and 30 (below 20, C1 is descriptive only, so
        not simulated).
    Reported: power (raw and after the gate), the false-claim rate with no
    effect per feature and for either feature, the rate when both groups
    share a one-direction preference, the episode-dependent strategy
    across the groups, how often the gate fires with equal attention and
    when either group answers randomly more often, and the secondary check.

  PART B -- supplement to script 7. Power of the primary paired test
    (paired t-test; the Wilcoxon as robustness check) for planning and half
    shift, all nuisances at their defaults, plus pair-specific preferences
    at SD 1.0 and 2.0 (script 7 has SD 1.5), so all three strengths have
    power figures; they are to be calibrated from the agreement on
    repeated trials in the pilot.

PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES
    No image-choice data exist. The responder share and shift, the strategy
    quarters, taste spread, lapse rates, the response-time model and every
    nuisance value are assumptions for planning. The report labels every
    figure that depends on them.

THE PERSON MODEL
    Script 7's (imported, unchanged): per block, the registered adjusted
    model (two weights + side term + N_corpus + N_orig, Firth) is fitted to
    choices simulated from script 7's person model (feature weights, scene
    terms, side bias, strategies k x N, fatigue, pair term, lapses).
    REPORT GROUP: derealization block (D) and ordinary block (O).
    COMPARISON GROUP: ordinary block and expectation block. In the
    simulation the expectation block takes script 7's "D" slot with no
    responders, i.e. it is simulated like the ordinary block; it is not
    used by C1. Each group's slots come from its own rows of
    assignment.csv.
    The C1 scenario (the main rows): half the report group respond (a
    derealization-only shift of 0.5 saturation, 0.7 contrast per corpus
    SD, both features at once); in each group a quarter pick the less
    typical image (-0.5 x N_corpus) and a separate quarter the more edited
    image (-0.5 x N_orig), in every block; personal lapse rates with mean
    0.1; no other nuisance. v4-style rows put the shift on one feature at a
    time. "Realistic" rows add script 7's default nuisances.
    Half shift: half of each responder's preference is shared by both of
    the report group's episodes (0.25/0.35 in both blocks, plus 0.25/0.35
    in the derealization block only). The comparison group does not share
    it, so C1 sees the full 0.5/0.7 and the secondary check sees the shared
    0.25/0.35.
    Attention: catch failures per block as in script 7 (5 per block,
    p = 0.75 x the block's mean lapse); responses under 300 ms over the
    analysed trials of both blocks, with probability
    lapse x RT_FAST_GIVEN_LAPSE + (1 - lapse) x RT_FAST_GIVEN_ENGAGED per
    trial (v4's model and values, read from the v4 C1 script).

REUSED (imported, not copied; SHA-256 of each file recorded and checked)
    simulate_c1_post_exclusion_v4.py   mw_p_batch, _c1_test, _c1_gate,
        RT_FAST_GIVEN_LAPSE, RT_FAST_GIVEN_ENGAGED, RT_FAST_CUTOFF_MS,
        C1_EXACT_MAX_N, ALPHA, GATE_ALPHA
    simulate_power_paired_v5b_adj.py (script 7, adjusted)
        load_stimulus_table, load_package (with its package checks),
        simulate_pool, draw_studies, rate_se, run_tests, person_thresholds,
        Fitter, EFFECTS, NUISANCE_OFF, NUISANCE_DEFAULT, N_GRID, TESTS and
        its person-model settings (PERSON_MODEL "adjusted")
    thresholds_v5b.py, simulate_power_v4_n10_20.py,
    explore_block_length_v5b.py, analyse_pair_capacity_v5b.py: through
        script 7 (fitting, sign-test critical values, the exact Wilcoxon,
        random streams), with script 7's hash checks
    One of script 7's settings is changed at run time, inside a task only
    and restored after use: LAPSE_MEAN, to give each group its own mean
    lapse rate in the attention rows; the slot table passed to
    simulate_pool selects the comparison group's rows.

INPUTS (relative to the project folder; run from there)
    data/frozen_trial_lists_v5b/v5b_150/, v5b_180/ and the stimulus set
        table (via script 7's loaders, checksums verified)
    PAIRED_RUN_DIR (script 7's adjusted run: thresholds_adjusted.csv,
        paired_power.csv, block_measures.csv, checks.csv, run_meta.json)
    the imported scripts above

OUTPUTS (one new folder per run under OUTPUT_ROOT; never overwritten)
    report.txt, summary.txt, checks.csv, settings.json, run_meta.json
    c1.csv                   per scenario, size, report N, comparison N
    c1_attention.csv         lapse, catch failures, fast responses and
                             exclusions per group and scenario
    c1_replicates.csv        headline cells on independent pools
    paired_pair_sd.csv       Part B
    c1_power.png, c1_gate.png, c1_strategy_threat.png, paired_pair_sd.png

REQUIRES  Python 3.9+, numpy, pandas, scipy; matplotlib for the figures
RUN       python simulate_c1_v5c.py               (v5c_180; runs the
                                                   regression first if needed)
          python simulate_c1_v5c.py --regression  (v5b_180 only)
          (each about 4-5 minutes with 2 workers)
"""

# One thread per process (BLAS and OpenMP), set before numpy is imported,
# so the result is the same for any N_WORKERS.
import os
import sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

# ============================================================
# SETTINGS
# ============================================================

# PROJECT_ROOT: None = the current folder (run from the project folder).
PROJECT_ROOT = None
# MODE
#   "v5c" (default): the v5c package. "regression": the same code on
#   v5b_180 with v5b's inputs (python simulate_c1_v5c.py --regression).
#   Script 2 is imported in the same mode (PAIRED_V5C_MODE); worker
#   processes inherit both through the environment.
MODE_ENV = "C1_V5C_MODE"
MODE = (os.environ.get(MODE_ENV)
        or ("regression" if "--regression" in sys.argv[1:] else "v5c"))
os.environ["PAIRED_V5C_MODE"] = MODE
BLOCK_SIZES = [180]

SCRIPT7 = "simulate_power_paired_v5c_adj.py"     # v5c script 2
C1_V4_SCRIPT = "simulate_c1_post_exclusion_v4.py"
EXPECTED_SHA256 = {
    SCRIPT7:
        "58a5640be3351ac80af350064fc8664409e1ac02f43039278bbe304f2bb98414",
    C1_V4_SCRIPT:
        "4d6c6fcee68d29c6e824b4ed99da3cd1edf6f372d6e9c06c49d30b0d137a0b13",
}
# Script 2's run (v5c_180): its thresholds (adjusted model), its SD 1.5
# rows beside Part B, and its record of the registered model.
PAIRED_RUN_DIR = ("data/power_v5c/paired_adj_v5c/sizes180_N10-20_model-adjusted_pool50000_studies10000_mixed0_seed42__s13156d40")
OUTPUT_ROOT = "data/power_v5c/c1_v5c"
MAKE_FIGURES = True
FIG_DPI = 150

# N_WORKERS / CHECK_ONE_WORKER: as in scripts 6 and 7 (None = all
#   available CPUs).
N_WORKERS = None
CHECK_ONE_WORKER = True

REPORT_GROUP = "participant"     # rows of assignment.csv
COMPARISON_GROUP = "control"

# --- Group sizes ---
REPORT_N = list(range(10, 21))   # analysable report-group participants
COMPARISON_N = [20, 30]          # analysable comparison-group participants
TABLE_N = [10, 15, 20]

# --- People (PLANNING ASSUMPTIONS, not estimates) ---
# Responders' shift: script 7's EFFECTS ("planning", "half shift",
# "no effect"). v4-style rows: one feature at a time, same share.
V4_STYLE_EFFECTS = {
    "v4-style: saturation only (0.5)": {"share": 0.5, "shift": (0.5, 0.0),
                                        "shift_sd": (0.0, 0.0),
                                        "shared": (0.0, 0.0)},
    "v4-style: contrast only (0.7)": {"share": 0.5, "shift": (0.0, 0.7),
                                      "shift_sd": (0.0, 0.0),
                                      "shared": (0.0, 0.0)},
}
# STRATEGY_SHARE / STRATEGY_K
#   The C1 strategy quarters (each group, every block, separate quarters):
#   main rows "pick the less typical" (k = -STRATEGY_K on N_corpus) and
#   "pick the more edited" (k = -STRATEGY_K on N_orig); sensitivity rows
#   the opposite direction (+STRATEGY_K: more typical / less edited).
#   Script 7's typical_* and less_edited_* nuisances, with a signed k.
STRATEGY_SHARE = 0.25
STRATEGY_K = 0.5
# LAPSE_EQUAL / LAPSE_PAIRS
#   Mean lapse per group (report, comparison). The main rows use
#   LAPSE_EQUAL; the attention rows use each pair, with no effect and with
#   the planning effect.
LAPSE_EQUAL = (0.1, 0.1)
LAPSE_PAIRS = [(0.1, 0.15), (0.1, 0.2), (0.15, 0.1), (0.2, 0.1)]
# SHARED_PREFERENCES
#   A one-direction preference in BOTH groups and every block (everyone,
#   towards the predicted image), with no effect.
SHARED_PREFERENCES = [(0.25, 0.35), (0.5, 0.7)]
# THREAT_K / THREAT_DEFS
#   The episode-dependent strategy across the groups (no effect): report
#   group -k x N in its derealization block, comparison group +k x N in its
#   ordinary block.
THREAT_K = [0.25, 0.5, 1.0]
THREAT_DEFS = ("corpus", "orig")

# --- Part B: pair-specific preferences for the paired test ---
PAIR_SDS = [1.0, 2.0]
PAIR_EFFECTS = ["planning", "half shift"]
PAIR_SD_IN_SCRIPT7 = 1.5

# --- Monte Carlo ---
#   Pools and sub-pools as in script 7 (its POOL_PEOPLE, N_BATCHES); each
#   C1 study draws both groups from the same sub-pool index, so every rate
#   has a measured SE that includes the pools' error.
N_STUDIES = 10_000
REPLICATE_ROWS = ["no effect", "planning"]

# --- Checks ---
CHECK_STUDIES = 300              # studies recomputed with SciPy
MAX_Z = 3.5
SHARE_BEYOND = 0.10
SCIPY_CHECK_CELLS = [("planning", 10, 20), ("planning", 20, 30),
                     ("no effect, lapse report 0.1 / comparison 0.2", 10, 20),
                     ("no effect, lapse report 0.1 / comparison 0.2", 20, 30)]

RANDOM_SEED = 42
STREAM_KEY = "c1_v5b"          # kept from v5b (the seeds stay as in v5b)

# --- v5b's registered results (comparison and regression reference) ---
REF_RUN = ("data/power_v5b/c1_v5b/sizes150-180_report10-20_comparison20-30"
           "_model-adjusted_pool50000_studies10000_seed42__sed9376ba")
COMPARE_FLAG = 0.05            # changes larger than this are flagged

# --- Regression (MODE "regression") ---
# REGRESSION_FILES: the files compared with REF_RUN (rows of 180 trials)
#   and their key columns. REGRESSION_EXACT_COLS: rates and shares, which
#   must be identical; other values within REGRESSION_TOL (v5b's "exact
#   apart from the last bits"). REGRESSION_RUN_FIRST: a v5c run first
#   looks for a passing regression run of this very script file and runs
#   one if there is none.
REGRESSION_ROOT = "data/power_v5c/c1_v5c/regression_v5b"
REGRESSION_FILES = [
    ("c1.csv", ["size", "scenario", "report_n", "comparison_n"]),
    ("c1_attention.csv", ["size", "scenario", "group"]),
    ("c1_replicates.csv", ["size", "scenario", "report_n", "comparison_n",
                           "measure"]),
    ("paired_pair_sd.csv", ["size", "effect", "pair_sd", "N", "test",
                            "feature"]),
]
REGRESSION_EXACT_COLS = (
    "raw_saturation", "raw_contrast", "raw_either", "gated_saturation",
    "gated_contrast", "gated_either", "gate_fires", "gate_fires_catch",
    "gate_fires_rt", "secondary_saturation", "secondary_contrast",
    "secondary_either", "excluded_share", "rate", "rate_1", "rate_2")
REGRESSION_TOL = 1e-12
REGRESSION_RUN_FIRST = True

if MODE == "regression":
    PAIRED_RUN_DIR = ("data/power_v5c/paired_adj_v5c/regression_v5b/sizes180_N10-20_model-adjusted_pool50000_studies10000_mixed0_seed42__se7f16b4c")
    OUTPUT_ROOT = REGRESSION_ROOT
elif MODE != "v5c":
    raise SystemExit(f"unknown MODE {MODE!r} (v5c or regression)")

# ============================================================
# Code
# ============================================================

import contextlib
import hashlib
import importlib.util
import io
import json
import multiprocessing as mp
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

FEATURES = ("sat", "con")
FEATURE_LABEL = {"sat": "saturation", "con": "contrast"}
T0 = time.time()
QUIET = False


def log(msg):
    if not QUIET:
        print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def root_path(*parts):
    """A path under the project folder: relative (to the current folder)
    when PROJECT_ROOT is None."""
    if PROJECT_ROOT:
        return os.path.join(PROJECT_ROOT, *parts)
    return os.path.join(*parts)


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
    "PROJECT_ROOT", "BLOCK_SIZES", "SCRIPT7", "C1_V4_SCRIPT",
    "EXPECTED_SHA256", "PAIRED_RUN_DIR", "OUTPUT_ROOT", "MAKE_FIGURES",
    "FIG_DPI", "N_WORKERS", "CHECK_ONE_WORKER", "REPORT_GROUP",
    "COMPARISON_GROUP", "REPORT_N", "COMPARISON_N", "TABLE_N",
    "V4_STYLE_EFFECTS", "STRATEGY_SHARE", "STRATEGY_K", "LAPSE_EQUAL",
    "LAPSE_PAIRS", "SHARED_PREFERENCES", "THREAT_K", "THREAT_DEFS",
    "PAIR_SDS", "PAIR_EFFECTS", "PAIR_SD_IN_SCRIPT7", "N_STUDIES",
    "REPLICATE_ROWS", "CHECK_STUDIES", "MAX_Z", "SHARE_BEYOND",
    "SCIPY_CHECK_CELLS", "RANDOM_SEED", "STREAM_KEY",
    "MODE", "REF_RUN", "COMPARE_FLAG", "REGRESSION_ROOT", "REGRESSION_FILES",
    "REGRESSION_EXACT_COLS", "REGRESSION_TOL", "REGRESSION_RUN_FIRST",
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


def settings_dict(s7):
    g = globals()
    d = {k: g[k] for k in SETTING_NAMES}
    d["script7_settings_used"] = {k: getattr(s7, k) for k in (
        "TASTE_SD", "LAPSE_MEAN", "LAPSE_BETWEEN_SD", "LAPSE_MAX",
        "CATCH_PER_BLOCK", "CATCH_P_FAIL_PER_LAPSE", "CATCH_FAIL_EXCLUDE",
        "EXCLUSION", "POOL_PEOPLE", "N_BATCHES", "PERSON_MODEL",
        "NUISANCE_OFF", "NUISANCE_DEFAULT", "EFFECTS", "N_GRID")}
    return d


def resolve_output_dir(s7):
    s = settings_dict(s7)
    s.pop("N_WORKERS")
    s.pop("REGRESSION_RUN_FIRST")      # cannot change the result
    h = hashlib.sha256(json.dumps(jsonable(s), sort_keys=True).encode()
                       ).hexdigest()[:8]
    name = (f"sizes{'-'.join(map(str, BLOCK_SIZES))}"
            f"_report{REPORT_N[0]}-{REPORT_N[-1]}"
            f"_comparison{'-'.join(map(str, COMPARISON_N))}"
            f"_model-{s7.PERSON_MODEL}"
            f"_pool{s7.POOL_PEOPLE}_studies{N_STUDIES}"
            f"_seed{RANDOM_SEED}__s{h}")
    base = root_path(OUTPUT_ROOT, name)
    out, k = base, 1
    while os.path.exists(out):
        k += 1
        out = f"{base}_run{k}"
    os.makedirs(out)
    return out


def import_file(fname):
    path = root_path(fname)
    spec = importlib.util.spec_from_file_location(
        os.path.splitext(fname)[0] + "_imported_s8", path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def load_checked(fname, ck, hashes):
    path = root_path(fname)
    ok = os.path.exists(path)
    ck.add("inputs", f"{fname} exists", ok, path)
    if not ok:
        raise SystemExit(f"missing {path}")
    digest = sha256_of(path)
    hashes[fname] = digest
    ck.add("inputs", f"{fname} is the expected version (SHA-256)",
           digest == EXPECTED_SHA256.get(fname), digest[:16] + "...")
    return import_file(fname)


@contextlib.contextmanager
def lapse_mean(s7, value):
    """Script 7's LAPSE_MEAN for one group's pool, restored afterwards."""
    old = s7.LAPSE_MEAN
    s7.LAPSE_MEAN = value
    try:
        yield
    finally:
        s7.LAPSE_MEAN = old


def comparison_slots(pk, size, s7):
    """The comparison group's slots in script 7's form: its expectation
    block takes the "D" slot (not used by C1), its ordinary block "O"."""
    a = pd.read_csv(root_path(s7.PACKAGES[size], "package",
                              "assignment.csv"))
    a = a[a["group"] == COMPARISON_GROUP].sort_values("slot")
    rows = []
    for r in a.itertuples():
        e_first = r.block1_episode == "expectation"
        rows.append({"der_list": r.block1_list if e_first else r.block2_list,
                     "ord_list": r.block2_list if e_first else r.block1_list,
                     "der_pos": 1 if e_first else 2})
    ok = (len(a) == 64 and set(a["block1_episode"]) | set(a["block2_episode"])
          == {"ordinary", "expectation"}
          and all((x.block1_episode == "ordinary")
                  != (x.block2_episode == "ordinary") for x in a.itertuples()))
    return {**pk, "slots": rows}, ok, len(a)


def add_fast(pool, size, c1, rng):
    """Share of responses under the cut-off over the analysed trials of
    both blocks (v4's response-time model, per block lapse)."""
    k = 0
    for b in ("D", "O"):
        lam = pool[f"lapse_{b}"]
        p = lam * c1.RT_FAST_GIVEN_LAPSE + (1 - lam) * c1.RT_FAST_GIVEN_ENGAGED
        k = k + rng.binomial(size, np.clip(p, 0, 1))
    pool["fast"] = k / (2.0 * size)
    pool["catch"] = pool["catch_D"] + pool["catch_O"]


def strategy_nuis(base, sign=-1.0):
    """The C1 strategy quarters on a nuisance set: sign -1 = less typical
    and more edited (main rows), +1 = more typical and less edited."""
    return {**base, "typical_share": STRATEGY_SHARE,
            "typical_k": sign * STRATEGY_K,
            "less_edited_share": STRATEGY_SHARE,
            "less_edited_k": sign * STRATEGY_K}


# ------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------

def threat_name(dname, k):
    lab = "less typical" if dname == "corpus" else "more edited"
    opp = "more typical" if dname == "corpus" else "less edited"
    return (f"no effect, episode-dependent strategy ({dname}, k {k:g}): "
            f"report group's derealization block picks the {lab}, "
            f"comparison group's ordinary block the {opp}")


def build_scenarios(s7):
    """Each: name, kind, report effect, nuisances of the report and the
    comparison group, lapse pair."""
    S = []
    base = strategy_nuis(s7.NUISANCE_OFF)
    opp = strategy_nuis(s7.NUISANCE_OFF, +1.0)
    real = strategy_nuis(s7.NUISANCE_DEFAULT)

    def add(name, kind, eff, nuis, lapses=LAPSE_EQUAL, nuis_c=None):
        S.append({"name": name, "kind": kind, "effect": eff,
                  "nuis": nuis, "nuis_c": nuis if nuis_c is None else nuis_c,
                  "lapses": lapses})

    for eff in ("no effect", "planning", "half shift"):
        add(eff, "main", s7.EFFECTS[eff], base)
    for name, eff in V4_STYLE_EFFECTS.items():
        add(name, "v4-style", eff, base)
    for sp in SHARED_PREFERENCES:
        add(f"no effect, both groups share +{sp[0]:g}/+{sp[1]:g}",
            "shared preference", s7.EFFECTS["no effect"],
            {**base, "dull": sp})
    for lp in LAPSE_PAIRS:
        for eff in ("no effect", "planning"):
            add(f"{eff}, lapse report {lp[0]:g} / comparison {lp[1]:g}",
                "attention", s7.EFFECTS[eff], base, lp)
    for eff in ("no effect", "planning", "half shift"):
        add(f"{eff}, script 7's default nuisances", "realistic",
            s7.EFFECTS[eff], real)
    for eff in ("no effect", "planning", "half shift"):
        add(f"{eff}, strategy quarters in the opposite direction (more "
            "typical / less edited)", "strategy sensitivity",
            s7.EFFECTS[eff], opp)
    for dname in THREAT_DEFS:
        for k in THREAT_K:
            add(threat_name(dname, k), "strategy threat",
                s7.EFFECTS["no effect"],
                {**base, "episode_strategy": (dname, -k, 0.0)},
                nuis_c={**base, "episode_strategy": (dname, 0.0, k)})
    names = [s["name"] for s in S]
    assert len(names) == len(set(names)), "duplicate scenario"
    return S


# ------------------------------------------------------------
# C1 per cell
# ------------------------------------------------------------

def c1_cell(rp, cp, nr, nc, rng, s7, c1):
    ir, ex_r, sb = s7.draw_studies(rp, nr, N_STUDIES, rng)
    ic, ex_c, sb2 = s7.draw_studies(cp, nc, N_STUDIES, rng)
    assert np.array_equal(sb, sb2)
    A = rp["est_D"][ir][:, :, :2]          # report, derealization block
    B = cp["est_O"][ic][:, :, :2]          # comparison, ordinary block
    r_sat, r_con = c1._c1_test(A, B)
    fired, f_catch, f_rt = c1._c1_gate(rp["catch"][ir], cp["catch"][ic],
                                       rp["fast"][ir], cp["fast"][ic])
    s_sat, s_con = c1._c1_test(rp["est_O"][ir][:, :, :2], B)
    live = ~fired
    out = {}
    for key, x in (("raw_saturation", r_sat), ("raw_contrast", r_con),
                   ("raw_either", r_sat | r_con),
                   ("gated_saturation", r_sat & live),
                   ("gated_contrast", r_con & live),
                   ("gated_either", (r_sat | r_con) & live),
                   ("gate_fires", fired), ("gate_fires_catch", f_catch),
                   ("gate_fires_rt", f_rt),
                   ("secondary_saturation", s_sat),
                   ("secondary_contrast", s_con),
                   ("secondary_either", s_sat | s_con)):
        m, se = s7.rate_se(x, sb)
        out[key] = m
        out[key + "_se"] = se
    out["excluded_report"] = float(ex_r.mean())
    out["excluded_comparison"] = float(ex_c.mean())
    n = CHECK_STUDIES
    return out, (A[:n], B[:n], rp["catch"][ir][:n], cp["catch"][ic][:n],
                 rp["fast"][ir][:n], cp["fast"][ic][:n], r_sat[:n],
                 r_con[:n], fired[:n])


def scipy_check(sample, c1, ck, label):
    """Recompute the decisions of CHECK_STUDIES studies with SciPy."""
    A, B, cr, cc, fr, fc, r_sat, r_con, fired = sample
    n = min(CHECK_STUDIES, len(A))
    bad = 0
    for s in range(n):
        p = []
        for j in range(2):
            m, k = A.shape[1], B.shape[1]
            meth = "exact" if m + k <= c1.C1_EXACT_MAX_N else "asymptotic"
            p.append(stats.mannwhitneyu(A[s, :, j], B[s, :, j],
                                        alternative="greater",
                                        method=meth,
                                        use_continuity=True).pvalue)
        lo, hi = min(p), max(p)
        rej = [False, False]
        if lo <= c1.ALPHA / 2:
            rej[int(np.argmin(p))] = True
            if hi <= c1.ALPHA:
                rej = [True, True]
        g = [stats.mannwhitneyu(cc[s], cr[s], alternative="greater",
                                method="asymptotic",
                                use_continuity=True).pvalue
             if not (np.all(cc[s] == cc[s][0]) and np.all(cr[s] == cc[s][0]))
             else 1.0,
             stats.mannwhitneyu(fc[s], fr[s], alternative="greater",
                                method="asymptotic",
                                use_continuity=True).pvalue
             if not (np.all(fc[s] == fc[s][0]) and np.all(fr[s] == fc[s][0]))
             else 1.0]
        gf = min(g) <= c1.GATE_ALPHA / 2
        bad += (rej[0] != bool(r_sat[s])) + (rej[1] != bool(r_con[s])) + (
            gf != bool(fired[s]))
    ck.add("tools", f"C1 test and gate decisions equal SciPy's Mann-Whitney "
           f"with Holm ({label}, {n} studies)", bad == 0,
           f"{bad} differing decisions")


# ------------------------------------------------------------
# Tasks
# ------------------------------------------------------------

_W = {}


def _worker_init(state, quiet=True):
    """Load script 7 and, through it, the scripts it imports (their hashes
    are checked once, in the main process)."""
    global QUIET
    QUIET = quiet
    s7 = import_file(SCRIPT7)
    s7.QUIET = True
    c1 = import_file(C1_V4_SCRIPT)
    v4 = s7.load_module(s7.V4_SIM_SCRIPT)
    cap = s7.load_module(s7.CAPACITY_SCRIPT)
    explore = s7.load_module(s7.EXPLORE_SCRIPT)
    s6 = s7.load_module(s7.THRESHOLDS_SCRIPT)
    _W.update(s7=s7, c1=c1, v4=v4, cap=cap, explore=explore,
              fit=s7.Fitter(v4, s6),
              crit=v4.sign_test_critical(max(max(REPORT_N), 20)),
              scen=build_scenarios(s7), **state)


def _pools(sc, size, st_r, st_c):
    """The report group's and the comparison group's pools of a scenario."""
    s7, c1, fit = _W["s7"], _W["c1"], _W["fit"]
    with lapse_mean(s7, sc["lapses"][0]):
        rp = s7.simulate_pool(_W["pk"][size], sc["effect"], sc["nuis"],
                              st_r("report_pool"), fit)
    add_fast(rp, size, c1, st_r("report_fast"))
    with lapse_mean(s7, sc["lapses"][1]):
        cp = s7.simulate_pool(_W["pkc"][size], s7.EFFECTS["no effect"],
                              sc["nuis_c"], st_c("pool"), fit)
    add_fast(cp, size, c1, st_c("fast"))
    return rp, cp


def comparison_key(sc):
    return json.dumps(jsonable([sc["nuis_c"], sc["lapses"][1]]),
                      sort_keys=True)


def task_scenario(size, i):
    s7, c1, cap = _W["s7"], _W["c1"], _W["cap"]
    sc = _W["scen"][i]
    st = cap.Streams(RANDOM_SEED, STREAM_KEY, size, sc["name"])
    stc = cap.Streams(RANDOM_SEED, STREAM_KEY, size, "comparison",
                      comparison_key(sc))
    rp, cp = _pools(sc, size, st, stc)
    att = []
    for grp, pool in (("report", rp), ("comparison", cp)):
        keep = ~pool["excluded"]
        att.append({
            "size": size, "scenario": sc["name"], "group": grp,
            "excluded_share": float(pool["excluded"].mean()),
            "lapse_mean": float(((pool["lapse_D"] + pool["lapse_O"])
                                 / 2)[keep].mean()),
            "catch_mean": float(pool["catch"][keep].mean()),
            "fast_share": float(pool["fast"][keep].mean()),
            "agreement_O": float(pool["agree_O"][keep].mean())})
    rows, samples = [], {}
    for nc in COMPARISON_N:
        for nr in REPORT_N:
            res, sample = c1_cell(rp, cp, nr, nc, st(f"studies_{nr}_{nc}"),
                                  s7, c1)
            rows.append({"size": size, "scenario": sc["name"],
                         "kind": sc["kind"], "report_n": nr,
                         "comparison_n": nc,
                         "lapse_report": sc["lapses"][0],
                         "lapse_comparison": sc["lapses"][1], **res})
            if (sc["name"], nr, nc) in SCIPY_CHECK_CELLS and \
                    size == BLOCK_SIZES[0]:
                samples[(sc["name"], nr, nc)] = sample
    return {"rows": rows, "att": att, "samples": samples}


def task_replicate(size, name):
    s7, c1, cap = _W["s7"], _W["c1"], _W["cap"]
    sc = [x for x in _W["scen"] if x["name"] == name][0]
    st = cap.Streams(RANDOM_SEED, STREAM_KEY, "replicate", size, name)
    stc = cap.Streams(RANDOM_SEED, STREAM_KEY, "replicate", size, name,
                      "comparison")
    rp, cp = _pools(sc, size, st, stc)
    out = []
    for nc in COMPARISON_N:
        for nr in TABLE_N:
            res, _ = c1_cell(rp, cp, nr, nc, st(f"studies_{nr}_{nc}"), s7, c1)
            out.append({"size": size, "scenario": name, "report_n": nr,
                        "comparison_n": nc, **res})
    return out


def task_pair(size, sd, eff):
    """Part B: the paired test with pair-specific preferences."""
    s7, cap = _W["s7"], _W["cap"]
    st = cap.Streams(RANDOM_SEED, STREAM_KEY, "pair_sd", size, sd, eff)
    nuis = {**s7.NUISANCE_DEFAULT, "pair_sd": sd}
    pool = s7.simulate_pool(_W["pk"][size], s7.EFFECTS[eff], nuis,
                            st("pool"), _W["fit"])
    s7.person_thresholds(pool, size, _W["TH"])
    agree = float(pool["agree_O"][~pool["excluded"]].mean())
    rows = []
    for N in s7.N_GRID:
        idx, _, sb = s7.draw_studies(pool, N, s7.N_STUDIES, st(f"N{N}"))
        r = s7.run_tests(pool, idx, _W["explore"], _W["crit"])
        for test in s7.TESTS:
            for f in FEATURES:
                p, se = s7.rate_se(r[(test, f)], sb)
                rows.append({"size": size, "effect": eff, "pair_sd": sd,
                             "N": N, "test": test,
                             "feature": FEATURE_LABEL[f], "rate": p,
                             "se": se, "agreement_O": agree,
                             "source": "this script"})
    return rows


TASKS = {"scenario": task_scenario, "replicate": task_replicate,
         "pair": task_pair}


def _run_task(kind, args):
    return TASKS[kind](*args)


def run_tasks(jobs, n_workers, state):
    if n_workers <= 1:
        _worker_init(state, quiet=False)
        return [_run_task(k, a) for k, a in jobs]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(n_workers, len(jobs)),
                             mp_context=ctx, initializer=_worker_init,
                             initargs=(state,)) as ex:
        futs = [ex.submit(_run_task, *j) for j in jobs]
        out = []
        for k, f in enumerate(futs):
            out.append(f.result())
            if (k + 1) % 10 == 0 or k + 1 == len(futs):
                log(f"  {k + 1} of {len(futs)} tasks done")
        return out


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


def figures(c1df, pair, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 1. C1 power (after the gate) by report N, 20 or 30 comparison
    fig, axes = plt.subplots(len(BLOCK_SIZES), 2, figsize=(10, 3.9 *
                             len(BLOCK_SIZES)), sharey=True, squeeze=False)
    for i, size in enumerate(BLOCK_SIZES):
        for j, f in enumerate(("contrast", "saturation")):
            ax = axes[i, j]
            for c, (sc, ls) in enumerate((("planning", "-"),
                                          ("half shift", "--"),
                                          ("no effect", ":"))):
                for k, nc in enumerate(COMPARISON_N):
                    q = c1df[(c1df["size"] == size) & (c1df["scenario"] == sc)
                             & (c1df["comparison_n"] == nc)]
                    ax.plot(q["report_n"], q[f"gated_{f}"], ls,
                            color=OKABE_ITO[k], marker="os"[k], ms=3,
                            label=f"{sc}, {nc} in the comparison group")
            ax.axhline(0.05, color="#888888", lw=0.8, ls=":")
            ax.set_ylim(0, 1)
            ax.set_title(f"{f.capitalize()}, {size} trials per block",
                         fontsize=10)
            ax.set_xlabel("Report-group participants (analysable)")
            if j == 0:
                ax.set_ylabel("Probability C1 is supported\n(after the "
                              "attention check)")
            _style(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle("C1: report group's derealization block against the "
                 "comparison group's ordinary block\n(responder share, "
                 "shift and attention model are planning assumptions)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out, "c1_power.png"), dpi=FIG_DPI)
    plt.close(fig)

    # 2. the gate
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    nc = COMPARISON_N[-1]
    rows = [("no effect", "equal attention")] + [
        (f"no effect, lapse report {a:g} / comparison {b:g}",
         f"report {a:g} / comparison {b:g}") for a, b in LAPSE_PAIRS]
    for i, (col, ylab) in enumerate((("gate_fires", "Share of studies where "
                                      "the attention\ncheck fires"),
                                     ("raw_either", "Share of studies with "
                                      "a C1 claim\nbefore the check, no "
                                      "effect"))):
        ax = axes[i]
        for c, (sc, lab) in enumerate(rows):
            for size, ls in zip(BLOCK_SIZES, ("-", "--")):
                q = c1df[(c1df["size"] == size) & (c1df["scenario"] == sc)
                         & (c1df["comparison_n"] == nc)]
                ax.plot(q["report_n"], q[col], ls, color=OKABE_ITO[c],
                        marker="o", ms=3, label=f"{lab}, {size} trials")
        ax.axhline(0.05, color="#888888", lw=0.8, ls=":")
        ax.set_xlabel("Report-group participants (analysable)")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.0 if i == 0 else 0.12)
        _style(ax)
    axes[0].legend(fontsize=6, frameon=False, loc="center right")
    fig.suptitle(f"The attention check, no effect, {nc} in the comparison "
                 "group (mean lapse rates are assumptions)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "c1_gate.png"), dpi=FIG_DPI)
    plt.close(fig)

    # 3. the episode-dependent strategy across the groups
    q = c1df[(c1df["kind"] == "strategy threat")
             & (c1df["report_n"] == max(REPORT_N))].copy()
    q["k"] = q["scenario"].str.extract(r"k ([0-9.]+)").astype(float)
    q["definition"] = q["scenario"].str.extract(r"\((corpus|orig),")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for j, col in enumerate(("raw_either", "gated_either")):
        ax = axes[j]
        c = 0
        for size in BLOCK_SIZES:
            for dn in THREAT_DEFS:
                for nc, ls in zip(COMPARISON_N, ("-", "--")):
                    r = q[(q["size"] == size) & (q["definition"] == dn)
                          & (q["comparison_n"] == nc)].sort_values("k")
                    ax.plot(r["k"], r[col], ls, color=OKABE_ITO[c],
                            marker="o", ms=3,
                            label=f"{size} trials, {dn}, {nc} comparison")
                c += 1
        ax.axhline(0.05, color="#888888", lw=0.8, ls=":")
        ax.set_xlabel("k (report D: -k x N; comparison O: +k x N)")
        ax.set_title("before the attention check" if j == 0
                     else "after the attention check", fontsize=10)
        if j == 0:
            ax.set_ylabel(f"Share of studies with a C1 claim\n(either "
                          f"feature, no effect, report N = {max(REPORT_N)})")
        _style(ax)
    axes[1].legend(fontsize=6, frameon=False)
    fig.suptitle("Episode-dependent strategy across the groups (adjusted "
                 "model; k values are assumptions)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "c1_strategy_threat.png"), dpi=FIG_DPI)
    plt.close(fig)

    # 4. Part B
    if pair is not None and not pair.empty:
        fig, axes = plt.subplots(len(BLOCK_SIZES), 2, figsize=(10, 3.9 *
                                 len(BLOCK_SIZES)), sharey=True,
                                 squeeze=False)
        sds = sorted(pair["pair_sd"].unique())
        for i, size in enumerate(BLOCK_SIZES):
            for j, f in enumerate(("contrast", "saturation")):
                ax = axes[i, j]
                for c, sd in enumerate(sds):
                    for eff, ls in (("planning", "-"), ("half shift", "--")):
                        q = pair[(pair["size"] == size)
                                 & (pair["pair_sd"] == sd)
                                 & (pair["effect"] == eff)
                                 & (pair["test"] == "t")
                                 & (pair["feature"] == f)]
                        ax.plot(q["N"], q["rate"], ls, color=OKABE_ITO[c],
                                marker="o", ms=3,
                                label=f"{eff}, pair-specific SD {sd:g}")
                ax.set_ylim(0, 1)
                ax.set_title(f"{f.capitalize()}, {size} trials per block",
                             fontsize=10)
                ax.set_xlabel("Report-group participants (analysable)")
                if j == 0:
                    ax.set_ylabel("Probability the paired t-test\nis "
                                  "significant")
                _style(ax)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, fontsize=7,
                   frameon=False)
        fig.suptitle("Primary paired t-test with pair-specific preferences "
                     "(all nuisances at defaults;\nstrengths to be "
                     "calibrated from the pilot's repeated trials)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(os.path.join(out, "paired_pair_sd.png"), dpi=FIG_DPI)
        plt.close(fig)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

def fmt(df, nd=3):
    return "  " + df.to_string(index=False, float_format=lambda x:
                               f"{x:.{nd}f}").replace("\n", "\n  ")


def c1_table(c1df, scen, cols, n_list=TABLE_N):
    q = c1df[(c1df["scenario"] == scen) & c1df["report_n"].isin(n_list)]
    return q[["size", "comparison_n", "report_n"] + cols]


def threat_table(c1df):
    q = c1df[(c1df["kind"] == "strategy threat")
             & c1df["report_n"].isin(TABLE_N)].copy()
    q["k"] = q["scenario"].str.extract(r"k ([0-9.]+)").astype(float)
    q["definition"] = q["scenario"].str.extract(r"\((corpus|orig),")
    return q[["size", "definition", "k", "comparison_n", "report_n",
              "raw_contrast", "raw_saturation", "raw_either",
              "gated_either", "gate_fires"]].sort_values(
        ["size", "definition", "k", "comparison_n", "report_n"])


def plain_summary(ck, c1df, pair, s7, meta7):
    L = []
    L.append(f"* Checks: {len(ck.rows) - ck.n_fail} of {len(ck.rows)} pass"
             + ("." if not ck.n_fail else f"; {ck.n_fail} FAIL (see "
                "checks.csv)."))
    L.append(f"* Per-person model: the registered {s7.PERSON_MODEL} model "
             "(side model + N_corpus + N_orig per block; side and N terms "
             "are nuisance terms, never results), from script 7's adjusted "
             f"run (primary test there: {meta7.get('chosen_test')}).")
    nr, nc = max(REPORT_N), COMPARISON_N
    for sc in ("planning", "half shift"):
        parts = []
        for size in BLOCK_SIZES:
            for c in nc:
                q = c1df[(c1df["scenario"] == sc) & (c1df["size"] == size)
                         & (c1df["report_n"] == nr)
                         & (c1df["comparison_n"] == c)].iloc[0]
                parts.append(f"{size}, {c} comparison: contrast "
                             f"{q['gated_contrast']:.2f}, saturation "
                             f"{q['gated_saturation']:.2f}")
        L.append(f"* C1 power after the attention check, {sc}, report N = "
                 f"{nr} (ASSUMPTIONS): " + "; ".join(parts) + ".")
    q = c1df[c1df["scenario"] == "no effect"]
    L.append(f"* No effect, equal attention: a C1 claim on either feature "
             f"in {q['raw_either'].min():.3f}-{q['raw_either'].max():.3f} of "
             f"studies; the attention check fires in "
             f"{q['gate_fires'].min():.3f}-{q['gate_fires'].max():.3f}.")
    t = c1df[c1df["kind"] == "strategy threat"].copy()
    t["k"] = t["scenario"].str.extract(r"k ([0-9.]+)").astype(float)
    g = t.groupby("k")[["raw_either", "gated_either"]].max()
    L.append("* Episode-dependent strategy across the groups (report D "
             "-kN, comparison O +kN; no effect; every N): largest false-"
             "claim rate on either feature "
             + ", ".join(f"{r.raw_either:.3f} at k {k:g} (after the check "
                         f"{r.gated_either:.3f})" for k, r in g.iterrows())
             + ".")
    a = c1df[c1df["kind"] == "attention"]
    a = a[a["scenario"].str.startswith("no effect")]
    L.append(f"* Unequal attention, no effect: C1 claim before the check up "
             f"to {a['raw_either'].max():.3f}, after it up to "
             f"{a['gated_either'].max():.3f}; the check fires in up to "
             f"{a['gate_fires'].max():.3f} of studies.")
    if pair is not None and not pair.empty:
        q = pair[(pair["N"] == 20) & (pair["test"] == "t")
                 & (pair["effect"] == "planning")]
        g = q.groupby("pair_sd")["rate"].agg(["min", "max"])
        L.append("* Part B, paired t-test at N = 20, planning, all defaults "
                 "+ pair-specific preferences: "
                 + "; ".join(f"SD {sd:g}: {r['min']:.2f}-{r['max']:.2f}"
                             for sd, r in g.iterrows()) + ".")
    return L


def write_report(out, ck, c1df, att, rep, pair, s7, meta, extra=()):
    L = []
    w = L.append
    w("C1 UNDER THE v5c DESIGN  (simulate_c1_v5c.py, v5c script 3; copy of "
      "v5b script 8)"
      + ("  [REGRESSION ON v5b_180]" if MODE == "regression" else ""))
    w("=" * 78)
    w(f"run       {meta['started']}  ({meta['minutes']:.1f} min, "
      f"{meta['workers']} worker process(es), one thread each)")
    w(f"output    {out}")
    w(f"python    {meta['python']}, numpy {meta['numpy']}, scipy "
      f"{meta['scipy']}, pandas {meta['pandas']}")
    w(f"script 7  {PAIRED_RUN_DIR}")
    w(f"package   {', '.join(s7.PACKAGES[s] for s in BLOCK_SIZES)}")
    w(SCRIPT_NAMES_NOTE)
    w("")
    w("0. IN PLAIN WORDS")
    w("-" * 78)
    for line in meta["summary"]:
        w(line)
    w("")
    w("PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES. No image-choice "
      "data exist.")
    w("The responder share and shift, the strategy quarters, taste spread, "
      "lapse rates, the")
    w("response-time model and the nuisance values were chosen for planning; "
      "every rate here")
    w("depends on them.")
    w("")
    w("1. WHAT WAS DONE")
    w("-" * 78)
    w("C1: one-sided Mann-Whitney per feature, report group's derealization-"
      "block weights")
    w("higher than the comparison group's ordinary-block weights, Holm "
      "across the two")
    w("features. Attention check first (catch failures over the session; "
      "share of")
    w(f"responses under {meta['rt_cutoff']} ms over the analysed trials of "
      "both blocks; comparison group")
    w("higher; Holm across the two): if it fires, C1 is inconclusive. "
      "Secondary check:")
    w("report group's ordinary-block weights against the comparison group's, "
      "same test.")
    w(f"Per-person model: script 7's registered {s7.PERSON_MODEL} model "
      "(side model + N_corpus +")
    w("N_orig per block, Firth, no intercept; side and N terms are nuisance "
      "terms, never results).")
    w(f"Pools of {s7.POOL_PEOPLE:,} people per group and scenario, each "
      f"fitted on both blocks; {N_STUDIES:,} studies")
    w(f"per cell; SE from {s7.N_BATCHES} sub-pools. Recruitment by slot with "
      "exclusion (3+ catch failures")
    w("of 10), as in script 7.")
    w("Main rows: half the report group respond (0.5 / 0.7 per SD, both "
      "features); in each")
    w(f"group a quarter pick the less typical image (-{STRATEGY_K} x "
      f"N_corpus) and a separate quarter")
    w(f"the more edited image (-{STRATEGY_K} x N_orig), every block; mean "
      f"lapse {LAPSE_EQUAL[0]} in both groups.")
    w("These strategies replace v5's quarter with +0.5 on both features "
      "towards the more")
    w("edited-looking image; they act through N, and the adjusted model's N "
      "terms take up")
    w("a strategy that is linear in N. [ASSUMPTIONS]")
    w(f"Response-time model (v4): P(fast | lapse) = {meta['p_fast_lapse']}, "
      f"P(fast | engaged) = {meta['p_fast_engaged']}.")
    w("")
    w("2. CHECKS")
    w("-" * 78)
    for r in ck.rows:
        w(f"  {r['status']}  [{r['section']}] {r['check']}")
        if r["detail"]:
            w(f"          {r['detail']}")
    w(f"\n  {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL")
    w("")
    w("3. C1 POWER (planning and half shift)")
    w("-" * 78)
    w("raw: C1 supported before the attention check; gated: supported and "
      "the check did")
    w("not fire (the registered verdict). Per feature after Holm; either = "
      "at least one.")
    cols = ["gated_contrast", "gated_saturation", "gated_either",
            "raw_either", "gate_fires"]
    for sc in ("planning", "half shift") + tuple(V4_STYLE_EFFECTS):
        w(f"\n  {sc}")
        w(fmt(c1_table(c1df, sc, cols)))
    w("")
    w("4. NO EFFECT: FALSE CLAIMS")
    w("-" * 78)
    cols = ["raw_contrast", "raw_saturation", "raw_either", "gated_either",
            "gate_fires"]
    for sc in ["no effect"] + [s for s in c1df["scenario"].unique()
                               if s.startswith("no effect, both groups")]:
        w(f"\n  {sc}")
        w(fmt(c1_table(c1df, sc, cols)))
    w("")
    w("5. THE EPISODE-DEPENDENT STRATEGY ACROSS THE GROUPS (a threat; no "
      "effect)")
    w("-" * 78)
    w("The report group picks the less typical / more edited image in its "
      "derealization")
    w("block (-k x N) and the comparison group the more typical / less "
      "edited image in its")
    w("ordinary block (+k x N); their other blocks have no strategy. C1 "
      "compares exactly")
    w("these two blocks. False-claim rate per feature (raw, after Holm) and "
      "for either")
    w("feature, before (raw) and after (gated) the attention check. "
      "[k values: ASSUMPTIONS]")
    w(fmt(threat_table(c1df)))
    w("")
    w("6. THE ATTENTION CHECK")
    w("-" * 78)
    w("Mean lapse per group (report / comparison). With the comparison group "
      "guessing more")
    w("often, its weights shrink towards zero, so a preference both groups "
      "share can make")
    w("the report group look higher (the shared-preference rows in section 4 "
      "show that")
    w("preference with equal attention). raw_either shows how often unequal "
      "attention alone")
    w("produces a C1 claim with no effect before the check; gated_either "
      "after it. When the")
    w("check fires, C1 is inconclusive, so with an effect (planning rows) a "
      "less attentive")
    w("comparison group also costs power.")
    cols = ["gate_fires", "gate_fires_catch", "gate_fires_rt", "raw_either",
            "gated_either"]
    for sc in [s for s in c1df["scenario"].unique() if "lapse report" in s]:
        w(f"\n  {sc}")
        w(fmt(c1_table(c1df, sc, cols)))
    w("")
    w("7. SCRIPT 7'S DEFAULT NUISANCES ADDED, AND THE STRATEGY QUARTERS "
      "REVERSED")
    w("-" * 78)
    cols = ["gated_contrast", "gated_saturation", "gated_either",
            "raw_either", "gate_fires"]
    for sc in [s for s in c1df["scenario"].unique()
               if "default nuis" in s or "opposite direction" in s]:
        w(f"\n  {sc}")
        w(fmt(c1_table(c1df, sc, cols)))
    w("")
    w("8. SECONDARY CHECK: report group's ordinary block against the "
      "comparison group's")
    w("-" * 78)
    w("Under half shift the report group's shared preference shows here; "
      "under planning")
    w("it should not (raw rates, before the attention check).")
    cols = ["secondary_contrast", "secondary_saturation", "secondary_either"]
    for sc in ("no effect", "planning", "half shift"):
        w(f"\n  {sc}")
        w(fmt(c1_table(c1df, sc, cols)))
    w("")
    w("9. ATTENTION MEASURES AND EXCLUSIONS BY GROUP (pool means)")
    w("-" * 78)
    w(fmt(att))
    w("")
    w("10. MONTE CARLO CHECK: headline cells on independent pools")
    w("-" * 78)
    w(fmt(rep[rep["report_n"].isin(TABLE_N)], 4))
    w("")
    w("11. PART B: THE PRIMARY PAIRED TEST WITH PAIR-SPECIFIC PREFERENCES")
    w("-" * 78)
    w("All nuisances at script 7's defaults plus a person x pair preference "
      "(logit SD; the")
    w(f"same on a pair's repeat); {s7.PERSON_MODEL} model. Paired t-test = "
      f"primary; Wilcoxon = robustness")
    w(f"check. SD {PAIR_SD_IN_SCRIPT7:g} rows are script 7's (read from its "
      "adjusted run); the others are")
    w("new here. Strengths to be calibrated from agreement on repeated "
      "trials in the pilot")
    w("[ASSUMPTIONS].")
    if pair is not None and not pair.empty:
        q = pair[pair["N"].isin(TABLE_N)]
        piv = q.pivot_table(index=["size", "effect", "pair_sd"],
                            columns=["test", "feature", "N"], values="rate")
        w("  " + piv.to_string(float_format=lambda x: f"{x:.3f}")
          .replace("\n", "\n  "))
        ag = pair.drop_duplicates(["size", "pair_sd"])[
            ["size", "pair_sd", "agreement_O"]]
        w("\n  Agreement on repeated trials (ordinary block, analysable "
          "people, planning pools):")
        w(fmt(ag))
    w("")
    for line in extra:
        w(line)
    w("15. FILES")
    w("-" * 78)
    for fn in meta["files"]:
        w(f"  {fn}")
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------
# v5c: preflight, regression on v5b_180, comparison with v5b (added in v5c)
# ------------------------------------------------------------

SCRIPT_NAMES_NOTE = (
    "Script numbers in this report are v5b's: script 6 = thresholds_v5c.py "
    "(v5c script 1), script 7 = simulate_power_paired_v5c_adj.py (v5c "
    "script 2), script 8 = simulate_c1_v5c.py (v5c script 3), script 8b = "
    "simulate_c1_attention_v5c.py (v5c script 4).")


def preflight(paths):
    """List every input path and whether it was found; stop before any
    computation if one is missing."""
    print(f"PREFLIGHT ({os.path.basename(__file__)}, mode {MODE}; run from "
          f"{os.path.abspath(root_path('.'))})", flush=True)
    missing = []
    for p in paths:
        ok = os.path.exists(root_path(p))
        print(f"  {'found  ' if ok else 'MISSING'}  {p}", flush=True)
        if not ok:
            missing.append(p)
    print("  (images in package/stimuli/ are checked when present; they are "
          "not needed)", flush=True)
    if MODE == "v5c":
        print(f"  regression runs are looked for in {REGRESSION_ROOT} (run "
              "now if none passed)", flush=True)
    if missing:
        raise SystemExit(f"PREFLIGHT FAILED: {len(missing)} input(s) missing "
                         "(listed above). Run from the working folder that "
                         "holds the scripts and data/.")
    print("  all inputs found\n", flush=True)


def package_paths(pk_dir, items=True):
    """The package files read through script 2's loader."""
    p = [f"{pk_dir}/checksums.sha256", f"{pk_dir}/FROZEN.json",
         f"{pk_dir}/package/design.json", f"{pk_dir}/package/assignment.csv",
         f"{pk_dir}/package/manifest.csv"]
    if items:
        p.append(f"{pk_dir}/package/items.csv")
    return p + [f"{pk_dir}/package/lists/{b}_o{k}.json" for b in ("A", "B")
                for k in (1, 2, 3, 4)]


def _passed(meta):
    return meta.get("checks_fail") == 0 and meta.get("regression_pass") is True


def find_regression_run(sha):
    """The newest regression run of this script file (same SHA-256),
    preferring one that passed."""
    root = root_path(REGRESSION_ROOT)
    if not os.path.isdir(root):
        return None
    runs = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "run_meta.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            m = json.load(f)
        if m.get("mode") == "regression" and m.get("script_sha256") == sha:
            runs.append((_passed(m), m.get("started", ""),
                         f"{REGRESSION_ROOT}/{d}", m))
    if not runs:
        return None
    runs.sort(key=lambda r: (r[0], r[1]))
    return runs[-1][2], runs[-1][3]


def regression_gate(ck, sha):
    """v5c mode: the regression on v5b_180 must have passed. A passing
    regression run of this very script file is reused; otherwise it is run
    now (this script with --regression). A failure stops here, before the
    v5c run."""
    found = find_regression_run(sha)
    if (found is None or not _passed(found[1])) and REGRESSION_RUN_FIRST:
        log("no passing regression run of this script version: running "
            f"python {os.path.basename(__file__)} --regression first")
        env = dict(os.environ, **{MODE_ENV: "regression",
                                  "PAIRED_V5C_MODE": "regression"})
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--regression"], env=env, check=False)
        found = find_regression_run(sha)
    sec = "regression (v5b_180)"
    if found is None:
        ck.add(sec, "a regression run of this script version exists", False,
               REGRESSION_ROOT)
        raise SystemExit("REGRESSION MISSING: run python "
                         f"{os.path.basename(__file__)} --regression")
    d, m = found
    reg = pd.read_csv(os.path.join(root_path(d), "regression.csv"))
    for r in reg.itertuples():
        ck.add(sec, f"{r.what} equals v5b's ({REF_RUN.split('/')[-2]}): "
               f"rates identical, other values within {REGRESSION_TOL:g}",
               r.status == "PASS", f"{r.rows_matched} of {r.rows_ref} rows "
               f"matched, {r.columns_compared} columns, max |difference| "
               f"{r.max_abs_diff:.1e} (rates {r.max_abs_diff_rates:.1e}; "
               f"bit-identical: {'yes' if r.identical else 'no'})")
    ok = _passed(m)
    ck.add(sec, "the regression run passed every check", ok,
           f"{d}: {m.get('checks_pass')} PASS, {m.get('checks_fail')} FAIL")
    if not ok:
        raise SystemExit(f"REGRESSION FAILED ({d}): the v5c run is not "
                         "started. See that run's checks.csv.")
    log(f"regression on v5b_180 passed: {d}")
    return {"dir": d, "meta": m, "table": reg}


def _read(p):
    return pd.read_csv(root_path(p), float_precision="round_trip",
                       low_memory=False)


def _compare_frames(mine, ref, keys):
    """ref's rows matched in mine on keys; every other column of ref
    compared (numbers: absolute difference, NaN = NaN; text: equal).
    Rates (REGRESSION_EXACT_COLS) must be identical, other numbers within
    REGRESSION_TOL."""
    ref = ref.reset_index(drop=True)
    mine = mine.reset_index(drop=True)
    for k in keys:                      # the same key types on both sides
        if ref[k].dtype != mine[k].dtype:
            ref[k], mine[k] = ref[k].astype(str), mine[k].astype(str)
    dup = ref.duplicated(keys).any() or mine.duplicated(keys).any()
    cols = [c for c in ref.columns if c not in keys]
    missing = [c for c in cols if c not in mine.columns]
    m = ref.merge(mine, on=keys, how="left", suffixes=("_ref", ""),
                  indicator=True)
    matched = int((m["_merge"] == "both").sum())
    worst, worst_rate, bad = 0.0, 0.0, []
    for c in cols:
        if c in missing:
            continue
        a, b = m[c], m[c + "_ref"]
        if (pd.api.types.is_numeric_dtype(a)
                and pd.api.types.is_numeric_dtype(b)):
            same_nan = a.isna() & b.isna()
            d = (a.astype(float) - b.astype(float)).abs()
            if (d.isna() & ~same_nan).any():
                bad.append(c)
            else:
                dm = float(d.where(~same_nan, 0.0).max())
                worst = max(worst, dm)
                if c in REGRESSION_EXACT_COLS:
                    worst_rate = max(worst_rate, dm)
        elif not ((a.astype(str) == b.astype(str)).fillna(False)
                  | (a.isna() & b.isna())).all():     # NA = NA
            bad.append(c)
    ok = (not dup and not missing and not bad and matched == len(ref)
          == len(mine) and len(ref) > 0 and worst_rate == 0.0
          and worst <= REGRESSION_TOL)
    note = "; ".join(x for x in (
        "duplicate keys" if dup else "",
        f"missing columns {missing}" if missing else "",
        f"differing columns {bad}" if bad else "") if x)
    return {"rows_ref": len(ref), "rows_mine": len(mine),
            "rows_matched": matched,
            "columns_compared": len(cols) - len(missing),
            "max_abs_diff": worst, "max_abs_diff_rates": worst_rate,
            "identical": worst == 0.0 and not bad,
            "status": "PASS" if ok else "FAIL", "note": note}


def regression_compare(out, ck):
    """MODE regression: this run (v5b_180, v5b's inputs) against v5b's
    registered run (REF_RUN), every row of 180 trials of the files in
    REGRESSION_FILES."""
    o = os.path.relpath(out, root_path(".")) if PROJECT_ROOT else out
    s = BLOCK_SIZES[0]
    rows = []
    for fn, keys in REGRESSION_FILES:
        ref = _read(f"{REF_RUN}/{fn}")
        if "size" in ref.columns:
            ref = ref[ref["size"].astype(str) == str(s)]
        mine = _read(f"{o}/{fn}")
        r = {"what": fn, "reference": f"{REF_RUN}/{fn}", "file": f"{o}/{fn}",
             **_compare_frames(mine, ref, keys)}
        rows.append(r)
        ck.add("regression (v5b_180)", f"{fn} equals v5b's "
               f"({REF_RUN.split('/')[-2]}, {s} trials): rates identical, "
               f"other values within {REGRESSION_TOL:g}",
               r["status"] == "PASS",
               f"{r['rows_matched']} of {r['rows_ref']} rows matched "
               f"({r['rows_mine']} here), {r['columns_compared']} columns, "
               f"max |difference| {r['max_abs_diff']:.1e} (rates "
               f"{r['max_abs_diff_rates']:.1e}; bit-identical: "
               f"{'yes' if r['identical'] else 'no'})"
               + (f"; {r['note']}" if r["note"] else ""))
    reg = pd.DataFrame(rows)
    reg.to_csv(os.path.join(out, "regression.csv"), index=False)
    return reg


def regression_lines(reg, info, title):
    L = [f"{title} REGRESSION ON v5b_180 (proves the copy did not change the "
         "method)", "-" * 78]
    if reg is None and info is not None:
        reg = info["table"]
        L.append(f"Regression run reused: {info['dir']}")
        L.append(f"  (the same script file, SHA-256 "
                 f"{info['meta'].get('script_sha256', '')[:16]}...; "
                 f"{info['meta'].get('checks_pass')} PASS, "
                 f"{info['meta'].get('checks_fail')} FAIL)")
    elif reg is not None:
        L.append("This run IS the regression: this script on v5b_180 with "
                 "v5b's inputs (script 2's")
        L.append(f"regression run), compared with {REF_RUN},")
    else:
        return []
    L.append("every row of 180 trials (the slice reproduced is all of v5b's "
             "180-trial cells): rates")
    L.append(f"and shares identical, other values within {REGRESSION_TOL:g} "
             "(v5b's 'exact apart from the last")
    L.append("bits'; another CPU or numerical library changes only the last "
             "bits of a floating-point sum).")
    L.append("")
    L.append("  " + reg[["what", "rows_ref", "rows_matched", "columns_compared",
                         "max_abs_diff", "max_abs_diff_rates", "identical",
                         "status"]].to_string(
        index=False, float_format=lambda x: f"{x:.1e}").replace("\n", "\n  "))
    L.append(f"\n  Regression: {'PASS' if (reg['status'] == 'PASS').all() else 'FAIL'}"
             " (a FAIL stops the v5c run).")
    L.append("")
    return L


def long_compare(mine, ref, keys, measures, group):
    """v5c (mine) against v5b (ref) for the given measure columns, long
    format, with the change and the flag."""
    rows = []
    m = ref[keys + measures].merge(mine[keys + measures], on=keys,
                                   how="outer", suffixes=("_v5b", "_v5c"))
    for c in measures:
        q = m[keys].copy()
        q["measure"] = c
        q["v5b"] = m[c + "_v5b"]
        q["v5c"] = m[c + "_v5c"]
        rows.append(q)
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "group", group)
    out["change"] = out["v5c"] - out["v5b"]
    out["flag"] = np.where(out["change"].abs() > COMPARE_FLAG,
                           f"CHANGED > {COMPARE_FLAG:g}", "")
    return out


def wide_compare(q, index, ncol, n_list):
    """Rows: index; columns: v5b and v5c at each N, the largest |change|
    and a flag '*' when it exceeds COMPARE_FLAG."""
    q = q[q[ncol].isin(n_list)].copy()
    q[ncol] = q[ncol].astype(int)
    for c in index:                     # whole numbers shown as integers
        if (pd.api.types.is_float_dtype(q[c]) and q[c].notna().all()
                and (q[c] % 1 == 0).all()):
            q[c] = q[c].astype(int)
    a = q.pivot_table(index=index, columns=ncol, values="v5b", sort=False)
    b = q.pivot_table(index=index, columns=ncol, values="v5c", sort=False)
    a.columns = [f"v5b N{n}" for n in a.columns]
    b.columns = [f"v5c N{n}" for n in b.columns]
    t = a.join(b)
    t["max|chg|"] = q.groupby(index, sort=False)["change"].apply(
        lambda x: x.abs().max())
    t["flag"] = np.where(t["max|chg|"] > COMPARE_FLAG, "*", "")
    return t.reset_index()


def preflight_paths():
    """Every input of this script (script 2's inputs are read through its
    loaders)."""
    s7mode = MODE
    pk = ("data/frozen_trial_lists_v5b/v5b_180" if s7mode == "regression"
          else "data/frozen_trial_lists_v5c/v5c_180")
    paths = [SCRIPT7, C1_V4_SCRIPT, "simulate_power_v4_n10_20.py",
             ("analyse_pair_capacity_v5b.py" if s7mode == "regression"
              else "analyse_pair_capacity_v5c.py"),
             ("explore_block_length_v5b.py" if s7mode == "regression"
              else "explore_block_length_v5c.py"), "thresholds_v5c.py",
             "data/stimulus_set_v5/7fd72262ba/stimulus_manifest.csv",
             "data/stimulus_set_v5/7fd72262ba/reference_stimulus_path.csv"]
    paths += package_paths(pk)
    paths += [f"{PAIRED_RUN_DIR}/{f}" for f in (
        "thresholds_adjusted.csv", "paired_power.csv", "block_measures.csv",
        "checks.csv", "run_meta.json")]
    paths += [f"{REF_RUN}/{fn}" for fn, _ in REGRESSION_FILES]
    return paths


C1_MEASURES = ["gated_contrast", "gated_saturation", "gated_either",
               "raw_contrast", "raw_saturation", "raw_either", "gate_fires",
               "secondary_either"]


def compare_with_v5b(out, c1df, att, pair):
    """The v5c results next to v5b's registered results (c1_v5b, 180
    trials): every C1 cell, the attention pool means and Part B."""
    s = BLOCK_SIZES[0]
    r1 = _read(f"{REF_RUN}/c1.csv")
    r1 = r1[r1["size"] == s]
    a = long_compare(c1df, r1, ["kind", "scenario", "comparison_n",
                                "report_n"], C1_MEASURES, "C1")
    r2 = _read(f"{REF_RUN}/c1_attention.csv")
    r2 = r2[r2["size"] == s]
    b = long_compare(att.rename(columns={"group": "pool"}),
                     r2.rename(columns={"group": "pool"}), ["scenario", "pool"],
                     ["excluded_share", "lapse_mean", "catch_mean",
                      "fast_share", "agreement_O"], "pool means by group")
    r3 = _read(f"{REF_RUN}/paired_pair_sd.csv")
    r3 = r3[r3["size"] == s]
    c = long_compare(pair[pair["size"] == s], r3,
                     ["effect", "pair_sd", "N", "test", "feature"], ["rate"],
                     "Part B: paired test with pair-specific preferences")
    comp = pd.concat([a, b, c], ignore_index=True)
    comp.to_csv(os.path.join(out, "comparison.csv"), index=False)
    return comp


def comparison_lines(comp):
    s = BLOCK_SIZES[0]
    L = ["14. COMPARISON WITH v5b'S REGISTERED RESULTS (c1_v5b, "
         f"{s} trials per block)", "-" * 78,
         f"v5b: {REF_RUN.split('/')[-1]}, rows of {s} trials. v5c: this run. "
         "The same streams and seeds,",
         "so the differences come from the lists. Shares of studies; report "
         "N 10/15/20 (every N in",
         f"comparison.csv); '*': a change larger than {COMPARE_FLAG:g}. "
         "gated = C1 after the attention check",
         "(the registered C1 claim), raw = before it, either = a claim on at "
         "least one feature.",
         "Comparisons of 150 with 180 trials: not applicable.", ""]
    q = comp[comp["group"] == "C1"]

    def block(title, sel, measures):
        t = q[sel & q["measure"].isin(measures)]
        L.append(f"  {title}")
        if t.empty:
            L.append("    (no rows)")
        else:
            L.append("  " + wide_compare(
                t, ["scenario", "comparison_n", "measure"], "report_n",
                TABLE_N).to_string(index=False, float_format=lambda x:
                                   f"{x:.3f}").replace("\n", "\n  "))
        L.append("")
    block("C1 POWER after the attention check (planning, half shift; "
          "comparison groups of 20 and 30)",
          q["scenario"].isin(["planning", "half shift"]),
          ["gated_contrast", "gated_saturation", "gated_either"])
    block("C1 FALSE-CLAIM RATE with no effect (equal attention), and how "
          "often the check fires",
          q["scenario"] == "no effect",
          ["raw_either", "gated_either", "gate_fires"])
    block("Other scenarios (v4-style, shared preference, unequal attention, "
          "realistic, strategy rows): a claim on either feature after the "
          "check",
          ~q["scenario"].isin(["planning", "half shift", "no effect"]),
          ["gated_either"])
    p = comp[comp["group"].str.startswith("Part B")]
    L.append("  Part B: the primary paired test, all defaults plus "
             "pair-specific preferences (SD 1.0, 1.5 from script 2, 2.0)")
    p = p.assign(row=p["effect"] + ", SD " + p["pair_sd"].map("{:g}".format)
                 + ", " + p["test"] + ", " + p["feature"])
    L.append("  " + wide_compare(p, ["row"], "N", TABLE_N).to_string(
        index=False, float_format=lambda x: f"{x:.3f}").replace("\n", "\n  "))
    L.append("")
    g = comp[comp["group"] == "pool means by group"]
    L.append(f"  Pool means by group (exclusions, lapse, catch, fast "
             f"responses, agreement): {int((g['flag'] != '').sum())} of "
             f"{len(g)} values changed by more than {COMPARE_FLAG:g} "
             "(comparison.csv).")
    allc = comp[comp["report_n"].isin(TABLE_N) | comp["report_n"].isna()]
    allc = allc[(allc["group"] != "Part B: paired test with pair-specific "
                 "preferences") | allc["N"].isin(TABLE_N)]
    L.append(f"  In all: {int((allc['flag'] != '').sum())} of {len(allc)} "
             f"compared values (N = {'/'.join(map(str, TABLE_N))}) changed "
             f"by more than {COMPARE_FLAG:g}.")
    L.append("")
    return L


def summary_v5c(comp, reg):
    L = []
    if isinstance(reg, pd.DataFrame):
        ok = (reg["status"] == "PASS").all()
        L.append(f"* Regression on v5b_180: {'PASS' if ok else 'FAIL'} "
                 f"({int((reg['status'] == 'PASS').sum())} of {len(reg)} "
                 "files reproduce v5b's c1_v5b run at 180 trials: rates "
                 f"identical, other values within {REGRESSION_TOL:g}).")
    elif reg is not None:
        t = reg["table"]
        L.append(f"* Regression on v5b_180: PASS ({len(t)} of {len(t)} files "
                 "reproduce v5b's c1_v5b run at 180 trials; " + reg["dir"]
                 + ").")
    if comp is None:
        return L
    q = comp[comp["group"] == "C1"]

    def cells(sc, meas, nc):
        r = q[(q["scenario"] == sc) & (q["measure"] == meas)
              & (q["comparison_n"] == nc) & q["report_n"].isin(TABLE_N)]
        return ", ".join(f"N{int(x.report_n)} {x.v5c:.2f} ({x.v5b:.2f})"
                         for x in r.sort_values("report_n").itertuples())
    L.append("* v5c against v5b (in brackets), 180 trials, C1 after the "
             "attention check (report group N 10/15/20):")
    for sc in ("planning", "half shift"):
        for nc in COMPARISON_N:
            L.append(f"    {sc}, comparison group {nc}: contrast "
                     f"{cells(sc, 'gated_contrast', nc)}; colour "
                     f"{cells(sc, 'gated_saturation', nc)}.")
    for nc in COMPARISON_N:
        L.append(f"    no effect, comparison group {nc}: a claim on either "
                 f"feature before the check {cells('no effect', 'raw_either', nc)}"
                 f"; after it {cells('no effect', 'gated_either', nc)}.")
    allc = comp[comp["report_n"].isin(TABLE_N) | comp["report_n"].isna()]
    allc = allc[(allc["group"] != "Part B: paired test with pair-specific "
                 "preferences") | allc["N"].isin(TABLE_N)]
    L.append(f"* {int((allc['flag'] != '').sum())} of {len(allc)} compared "
             f"values changed by more than {COMPARE_FLAG:g} (report section "
             "14, comparison.csv). 150 against 180 trials: not applicable.")
    return L


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    preflight(preflight_paths())
    ck = Checks()
    hashes = {}
    script_sha = sha256_of(os.path.abspath(__file__))
    if MODE == "regression":
        os.environ[MODE_ENV] = "regression"         # the workers
        reg_info = None
    else:                      # the regression must pass before anything else
        reg_info = regression_gate(ck, script_sha)
    s7 = load_checked(SCRIPT7, ck, hashes)
    s7.QUIET = True
    c1 = load_checked(C1_V4_SCRIPT, ck, hashes)
    n_workers = N_WORKERS or max(1, os.cpu_count() or 1)

    # --- script 7's inputs, through script 7's loaders and checks ---
    with open(os.path.join(root_path(s7.PACKAGES[BLOCK_SIZES[0]]),
                           "FROZEN.json")) as f:
        frozen0 = json.load(f)
    stim = s7.load_stimulus_table(frozen0, ck, hashes)
    pk = {size: s7.load_package(size, ck, hashes, stim, s7.LISTS)
          for size in BLOCK_SIZES}
    s7.load_module(s7.V4_SIM_SCRIPT, ck, hashes, frozen0["scripts_sha256"])
    cap = s7.load_module(s7.CAPACITY_SCRIPT, ck, hashes,
                         frozen0["scripts_sha256"])
    s7.load_module(s7.EXPLORE_SCRIPT, ck, hashes)
    s7.load_module(s7.THRESHOLDS_SCRIPT, ck, hashes)
    ck.add("inputs", "script 7's person model is the registered adjusted "
           "model", s7.PERSON_MODEL == "adjusted", s7.PERSON_MODEL)
    ck.add("inputs", "script 2 (imported as script 7) runs in this script's "
           "mode, on the matching package", s7.MODE == MODE,
           f"script 2 mode {s7.MODE}, package "
           + ", ".join(s7.PACKAGES[s] for s in BLOCK_SIZES))
    out = resolve_output_dir(s7)
    log(f"output folder {out}")
    log(f"{n_workers} worker process(es), one thread each")

    # --- script 7's adjusted run ---
    d7 = root_path(PAIRED_RUN_DIR)
    need = ["thresholds_adjusted.csv", "paired_power.csv",
            "block_measures.csv", "checks.csv", "run_meta.json"]
    ok = all(os.path.exists(os.path.join(d7, f)) for f in need)
    ck.add("inputs", "script 7's adjusted run exists (" + ", ".join(need)
           + ")", ok, PAIRED_RUN_DIR)
    if not ok:
        raise SystemExit(f"missing files in {d7}")
    for f in need:
        hashes[f"{PAIRED_RUN_DIR}/{f}"] = sha256_of(os.path.join(d7, f))
    c7 = pd.read_csv(os.path.join(d7, "checks.csv"))
    ck.add("inputs", "script 7's adjusted run passed every check",
           (c7["status"] == "PASS").all(),
           f"{(c7['status'] == 'PASS').sum()} PASS, "
           f"{(c7['status'] != 'PASS').sum()} FAIL")
    with open(os.path.join(d7, "run_meta.json")) as f:
        meta7 = json.load(f)
    ck.add("inputs", "script 7's run used the adjusted model",
           meta7.get("person_model") == "adjusted",
           f"person model {meta7.get('person_model')}, primary test "
           f"{meta7.get('chosen_test')}")
    ck.add("inputs", "script 7's run was made in this mode (script 2, "
           f"{MODE})", meta7.get("mode") == MODE,
           f"its mode {meta7.get('mode')}")
    same = [k for k, v in meta7["input_sha256"].items()
            if k.startswith(tuple(s7.PACKAGES.values())) and k in hashes]
    ck.add("inputs", "script 7's run read the same frozen package files",
           len(same) > 0 and all(meta7["input_sha256"][k] == hashes[k]
                                 for k in same),
           f"{len(same)} files compared")
    t = pd.read_csv(os.path.join(d7, "thresholds_adjusted.csv"))
    TH = {}
    for size in BLOCK_SIZES:
        for b in s7.SETS:
            q = t[(t["size"] == size) & (t["set"] == b)].set_index("feature")
            TH[(size, b)] = {
                "upper": np.array([q.loc["saturation", "upper"],
                                   q.loc["contrast", "upper"]]),
                "lower": np.array([q.loc["saturation", "lower"],
                                   q.loc["contrast", "lower"]])}
    ck.add("inputs", "the adjusted model's thresholds found for every size "
           "and set", len(TH) == len(BLOCK_SIZES) * len(s7.SETS),
           "; ".join(f"{s} {b}: sat {v['upper'][0]:+.3f}, con "
                     f"{v['upper'][1]:+.3f}" for (s, b), v in TH.items()))

    pkc = {}
    for size in BLOCK_SIZES:
        pkc[size], ok, n = comparison_slots(pk[size], size, s7)
        ck.add(f"package {size}", f"assignment table: 64 slots for the "
               f"{COMPARISON_GROUP} group, each with one ordinary and one "
               "expectation block", ok, f"{n} rows")

    # --- tool checks ---
    rng = cap.Streams(RANDOM_SEED, STREAM_KEY)("tools")
    worst = 0.0
    for m, n in ((6, 9), (8, 12)):
        A, B = rng.normal(0.3, 1, (100, m)), rng.normal(0, 1, (100, n))
        p, meth = c1.mw_p_batch(A, B)
        ref = [stats.mannwhitneyu(A[i], B[i], alternative="greater",
                                  method="exact").pvalue for i in range(100)]
        worst = max(worst, float(np.max(np.abs(p - ref))))
    ck.add("tools", "v4's exact Mann-Whitney equals SciPy's exact test",
           worst < 1e-10, f"max |difference| in p {worst:.1e}")
    A, B = rng.normal(0.3, 1, (100, 15)), rng.normal(0, 1, (100, 30))
    p, meth = c1.mw_p_batch(A, B)
    ref = [stats.mannwhitneyu(A[i], B[i], alternative="greater",
                              method="asymptotic",
                              use_continuity=True).pvalue for i in range(100)]
    ck.add("tools", "v4's normal-approximation Mann-Whitney (groups of "
           f"{15 + 30}, above its exact limit) equals SciPy's",
           meth == "asymptotic" and np.max(np.abs(p - ref)) < 1e-10,
           f"max |difference| in p {np.max(np.abs(p - ref)):.1e}")

    # --- the tasks ---
    scen = build_scenarios(s7)
    jobs = ([("scenario", (size, i)) for size in BLOCK_SIZES
             for i in range(len(scen))]
            + [("replicate", (size, name)) for size in BLOCK_SIZES
               for name in REPLICATE_ROWS]
            + [("pair", (size, sd, eff)) for size in BLOCK_SIZES
               for sd in PAIR_SDS for eff in PAIR_EFFECTS])
    state = {"pk": pk, "pkc": pkc, "TH": TH}
    log(f"{len(jobs)} tasks ({len(scen)} scenarios x {len(BLOCK_SIZES)} "
        "sizes, replicates, Part B)")
    res = run_tasks(jobs, n_workers, state)
    if CHECK_ONE_WORKER and n_workers > 1:
        _worker_init(state, quiet=False)
        log(f"rerunning task {jobs[0]} in this process (one worker)")
        again = task_scenario(*jobs[0][1])
        ok = (pd.DataFrame(again["rows"]).equals(pd.DataFrame(res[0]["rows"]))
              and again["att"] == res[0]["att"])
        ck.add("workers", f"one worker gives the same result as {n_workers}",
               ok, f"{jobs[0][1][0]} trials, '{scen[jobs[0][1][1]]['name']}'")
    elif CHECK_ONE_WORKER:
        ck.add("workers", "one worker gives the same result (run with one "
               "worker: nothing to compare)", True, "N_WORKERS = 1")

    by = {k: [r for (kind, _), r in zip(jobs, res) if kind == k]
          for k in TASKS}
    c1df = pd.DataFrame([row for r in by["scenario"] for row in r["rows"]])
    att = pd.DataFrame([row for r in by["scenario"] for row in r["att"]]
                       ).drop_duplicates(["size", "scenario", "group"])
    samples = {}
    for r in by["scenario"]:
        samples.update(r["samples"])
    for (name, nr, nc) in SCIPY_CHECK_CELLS:
        if (name, nr, nc) in samples:
            scipy_check(samples[(name, nr, nc)], c1, ck,
                        f"{name}, {nr} vs {nc}")
        else:
            ck.add("tools", f"SciPy check cell found: {name}, {nr} vs {nc}",
                   False, "")

    # --- replicate pools ---
    rep_rows = []
    for out_rows in by["replicate"]:
        for res_ in out_rows:
            o = c1df[(c1df["size"] == res_["size"])
                     & (c1df["scenario"] == res_["scenario"])
                     & (c1df["report_n"] == res_["report_n"])
                     & (c1df["comparison_n"] == res_["comparison_n"])
                     ].iloc[0]
            for key in ("gated_contrast", "gated_saturation", "gate_fires"):
                se = np.sqrt(o[key + "_se"] ** 2 + res_[key + "_se"] ** 2)
                rep_rows.append({"size": res_["size"],
                                 "scenario": res_["scenario"],
                                 "report_n": res_["report_n"],
                                 "comparison_n": res_["comparison_n"],
                                 "measure": key, "rate_1": o[key],
                                 "rate_2": res_[key],
                                 "z": (res_[key] - o[key]) / max(se, 1e-4)})
    rep = pd.DataFrame(rep_rows)
    az = rep["z"].abs()
    ck.add("Monte Carlo", "headline cells agree between independent pools",
           az.max() <= MAX_Z and (az > 1.96).mean() <= SHARE_BEYOND,
           f"{len(rep)} comparisons; largest |z| {az.max():.2f}; "
           f"{(az > 1.96).mean():.1%} beyond 1.96")

    # --- result checks ---
    nul = c1df[c1df["scenario"] == "no effect"]
    z = (nul["raw_either"] - 0.05) / nul["raw_either_se"]
    ck.add("results", "no effect, equal attention: the chance of a C1 claim "
           "on either feature stays at or below 0.05 + 3 SE (Holm controls "
           "it)", (z <= 3).all(),
           f"largest {nul['raw_either'].max():.4f}, mean "
           f"{nul['raw_either'].mean():.4f}")
    z = (nul["gate_fires"] - 0.05) / nul["gate_fires_se"]
    ck.add("results", "equal attention: the attention check fires at or "
           "below 0.05 + 3 SE", (z <= 3).all(),
           f"largest {nul['gate_fires'].max():.4f}, mean "
           f"{nul['gate_fires'].mean():.4f}")

    def rises(g):
        r = g["gated_either"].to_numpy()
        e = g["gated_either_se"].to_numpy()
        return bool(np.all(np.diff(r) > -3 * np.sqrt(e[1:] ** 2
                                                     + e[:-1] ** 2) - 0.002))
    mono = c1df[c1df["scenario"].isin(["planning", "half shift"])].sort_values(
        "report_n").groupby(["size", "scenario", "comparison_n"])[
        ["gated_either", "gated_either_se"]].apply(rises)
    ck.add("results", "C1 power rises with the report group's N (within "
           "Monte Carlo noise)", mono.all(),
           f"{int(mono.sum())} of {len(mono)} series")
    ex = att[(att["scenario"] == "no effect")]
    ck.add("results", "share excluded matches the lapse model in both "
           "groups (about 5% at mean lapse 0.1)",
           ex["excluded_share"].between(0.03, 0.07).all(),
           f"{ex['excluded_share'].round(3).tolist()}")
    hs = c1df[(c1df["scenario"] == "half shift")]
    pl = c1df[(c1df["scenario"] == "planning")]
    m = hs.merge(pl, on=["size", "report_n", "comparison_n"],
                 suffixes=("_h", "_p"))
    ck.add("results", "the secondary check sees the shared preference under "
           "half shift, not under planning (higher rate under half shift in "
           "every cell)",
           (m["secondary_either_h"] > m["secondary_either_p"]).all(),
           f"mean {m['secondary_either_h'].mean():.3f} vs "
           f"{m['secondary_either_p'].mean():.3f}")

    # --- Part B ---
    pair = pd.DataFrame([row for rows in by["pair"] for row in rows])
    q = pd.read_csv(os.path.join(d7, "paired_power.csv"))
    q = q[(q["role"] == "power (sensitivity)") & (q["lists"] == s7.LISTS)
          & q["scenario"].str.contains(f"SD {PAIR_SD_IN_SCRIPT7:g}")]
    bm7 = pd.read_csv(os.path.join(d7, "block_measures.csv"),
                      low_memory=False)
    a7 = {}
    for size in BLOCK_SIZES:
        r7 = bm7[(bm7["size"] == size) & (bm7["effect"] == "planning")
                 & (bm7["N"].astype(str) == "pool (analysable people)")
                 & bm7["scenario"].astype(str).str.contains(
                     f"SD {PAIR_SD_IN_SCRIPT7:g}")]
        a7[size] = float(r7["agree_O"].iloc[0]) if len(r7) else np.nan
    q = q.assign(pair_sd=PAIR_SD_IN_SCRIPT7, agreement_O=q["size"].map(a7),
                 source="script 7")[pair.columns]
    ck.add("Part B", f"script 7's SD {PAIR_SD_IN_SCRIPT7:g} rows found for "
           "both sizes and both effects", len(q) == len(BLOCK_SIZES)
           * len(PAIR_EFFECTS) * len(s7.N_GRID) * 4, f"{len(q)} rows")
    pair = pd.concat([pair, q], ignore_index=True).sort_values(
        ["size", "effect", "pair_sd", "test", "feature", "N"])
    piv = pair[pair["N"] == max(s7.N_GRID)].pivot_table(
        index=["size", "effect", "test", "feature"], columns="pair_sd",
        values="rate")
    sds = sorted(piv.columns)
    dec = (np.diff(piv[sds].to_numpy(), axis=1) < 0.02).all()
    ck.add("Part B", "power falls (or stays level) as pair-specific "
           f"preferences strengthen (N = {max(s7.N_GRID)})", dec,
           f"SD {', '.join(f'{s:g}' for s in sds)}")

    # --- write ---
    c1df.to_csv(os.path.join(out, "c1.csv"), index=False)
    att.to_csv(os.path.join(out, "c1_attention.csv"), index=False)
    rep.to_csv(os.path.join(out, "c1_replicates.csv"), index=False)
    pair.to_csv(os.path.join(out, "paired_pair_sd.csv"), index=False)
    if MAKE_FIGURES:
        try:
            figures(c1df, pair, out)
        except ImportError:
            log("matplotlib not installed: no figures")
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(jsonable(settings_dict(s7)), f, indent=2)
    # --- v5c: regression (MODE regression) or comparison with v5b ---
    if MODE == "regression":
        reg = regression_compare(out, ck)
        extra = regression_lines(reg, None, "13.")
        comp = None
    else:
        reg = None
        extra = regression_lines(None, reg_info, "13.")
        comp = compare_with_v5b(out, c1df, att, pair)
        extra += comparison_lines(comp)
    pd.DataFrame(ck.rows).to_csv(os.path.join(out, "checks.csv"),
                                 index=False)
    summary = plain_summary(ck, c1df, pair, s7, meta7)
    summary += summary_v5c(comp, reg if MODE == "regression" else reg_info)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("C1 UNDER THE v5c DESIGN (simulate_c1_v5c.py"
                + (", REGRESSION ON v5b_180" if MODE == "regression" else "")
                + "): IN PLAIN WORDS\n\n" + "\n".join(summary) + "\n")
    import scipy
    files = sorted(set(os.listdir(out)) | {"report.txt", "run_meta.json"})
    meta = {"started": started, "minutes": (time.time() - T0) / 60,
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__,
            "files": files, "rt_cutoff": c1.RT_FAST_CUTOFF_MS,
            "p_fast_lapse": c1.RT_FAST_GIVEN_LAPSE,
            "p_fast_engaged": c1.RT_FAST_GIVEN_ENGAGED,
            "workers": n_workers, "summary": summary}
    write_report(out, ck, c1df, att, rep, pair, s7, meta, extra)
    with open(os.path.join(out, "run_meta.json"), "w") as f:
        json.dump(jsonable({**{k: v for k, v in meta.items()
                               if k not in ("files", "summary")},
                            "output_dir": out, "platform": platform.platform(),
                            "cpu_count": os.cpu_count(),
                            "person_model": s7.PERSON_MODEL,
                            "input_sha256": hashes,
                            "mode": MODE,
                            "script": os.path.basename(__file__),
                            "script_sha256": script_sha,
                            "regression_pass": (
                                bool((reg["status"] == "PASS").all())
                                if MODE == "regression" else None),
                            "regression_run": (reg_info or {}).get("dir"),
                            "checks_pass": len(ck.rows) - ck.n_fail,
                            "checks_fail": ck.n_fail}), f, indent=2)
    log(f"done: {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL")
    log(f"report: {os.path.join(out, 'report.txt')}")


if __name__ == "__main__":
    main()
