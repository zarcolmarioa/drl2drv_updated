"""
nuisance_power_v5c.py  --  v5c power analysis, script 5: which nuisances
                           cost the primary test its power (a copy of
                           nuisance_power_v5b.py, changed only where listed
                           below)

WHAT CHANGED FROM nuisance_power_v5b.py (v5b -> v5c)
    Only the inputs, the names and the comparison section; the nine
    conditions, the common random numbers, the ranking, the additivity test,
    the anchors, the numbers of studies and the seeds (STREAM_KEY
    "nuisance_v5b", kept) are v5b's.
      * INPUTS: script 2 (simulate_power_paired_v5c_adj.py) is imported as
        "script 7", with its settings pointing to v5c (v5c_180, script 1's
        thresholds); PAIRED_RUN_DIR is script 2's run. v5c has one package
        (180 trials), so the ranking averages 6 cells (N = 10, 15, 20 x 2
        features) instead of v5b's 12, and "does 180 beat 150" is NOT
        APPLICABLE (no 150-trial package): the size comparison and its
        figure are left out and the report says so; fatigue's own cost at
        180 trials is still reported. The anchors and the fatigue
        conditions of the size comparison are still rerun with script 2's
        streams and must equal its run exactly.
      * REGRESSION: "python nuisance_power_v5c.py --regression" runs this
        same code on v5b_180 (script 2 in its regression mode, script 2's
        regression run) and requires every 180-trial row of v5b's
        nuisance_v5b run (..._sbb19480e: recoveries.csv, anchors.csv,
        fatigue_cost.csv) to be reproduced: rates identical, other values
        within 1e-12. ranking.csv is not compared directly (v5b averaged
        over both sizes); the comparison section recomputes v5b's ranking
        from its 180-trial cells. A normal run first looks for a passing
        regression of this very script file and runs it if there is none;
        a FAIL stops before the v5c run.
      * A COMPARISON section (report and comparison.csv): v5c next to v5b's
        registered results, changes larger than COMPARE_FLAG flagged.
      * A PREFLIGHT lists every input path; N_WORKERS = None uses all CPUs.
      * Output under data/power_v5c/nuisance_v5c/, v5b's folder naming.
        Script numbers in the text are v5b's; the report says so.

---- nuisance_power_v5b.py's description follows (v5b -> v5c: v5c_180 and
     simulate_power_paired_v5c_adj.py for v5b's packages and script 7) ----

nuisance_power_v5b.py  --  v5b power analysis, extra script: which nuisances
                           cost the primary test its power, and does 180
                           trials per block beat 150 once fatigue counts

WHAT CHANGED FROM nuisance_power_v5.py (v5 -> v5b)
    The leave-one-out logic, common random numbers, SE from sub-pools, the
    ranking, the additivity test, the checks, the report and the figure are
    v5's. Changed:
      * Script 7 is simulate_power_paired_v5b_adj.py (imported) with the
        REGISTERED per-person model, the strategy-adjusted model (side model
        + N_corpus + N_orig per block, Firth, no intercept); its adjusted
        run (PAIRED_RUN_DIR) is read; the frozen packages' checksums and the
        scripts' SHA-256 are verified through script 7's loaders, as in
        scripts 7 and 8.
      * THE NUISANCES are script 7's (v5b): taste stability, scene
        dependence, side bias, "memory makes it duller", the strategies
        "more typical" and "less edited" (a quarter each, k = 0.5), and
        fatigue (lapse +0.03 and preferences x0.939 per 100 trials). v5's
        "edited looks odd" and "block order" no longer exist: the two
        strategies replace the first, fatigue the second.
      * EXCLUSIONS: with fatigue on, switching a nuisance off can change
        catch failures and so exclusions; the conditions where that happens
        (fatigue off, no nuisances) are counted and reported, as v5 did for
        block order.
      * NEW: 150 AGAINST 180 TRIALS PER BLOCK, separately for planning and
        half shift, both features, N = 10, 15, 20: power at each size and
        the difference under all defaults, all defaults without fatigue,
        with fatigue doubled, and with 50% recovery at the rest. These rows
        exist in script 7's adjusted run; they are taken from it, after
        rerunning them with script 7's own streams and checking they match
        exactly. Fatigue's own power cost at each size comes from the
        matched family. A plain summary says whether the extra 30 trials
        per block buy a meaningful gain once fatigue is counted
        (differences of 0.02 or less are ties, as in script 2).
      * PARALLEL: every pool (anchor, size comparison, matched-family
        condition) is one task (N_WORKERS, one thread each), with named
        random streams; results are gathered in a fixed order, so they do
        not depend on N_WORKERS (checked by rerunning one task in this
        process). Relative paths only; a plain-language summary
        (summary.txt and the report's first section).

WHAT THIS ANSWERS
    Script 7 found that no nuisance creates false positives, but that all of
    them together lower the power of the primary test (the one-sided paired
    t-test on derealization-block weight minus ordinary-block weight). This
    script asks which of them cause that loss, by "leave one out": start
    from all nuisances at their defaults (script 7's NUISANCE_DEFAULT) and
    switch each off on its own (to its NUISANCE_OFF value):

      1. taste stability            0.7 -> 1 (taste the same in both blocks)
      2. scene dependence           person x scene preferences, SD x1 -> off
      3. side bias                  SD 0.25 logit -> 0
      4. "memory makes it duller"   +0.25 / +0.35 in everyone -> 0
      5. strategy "more typical"    a quarter, k 0.5 x N_corpus -> nobody
      6. strategy "less edited"     a quarter, k 0.5 x N_orig -> nobody
      7. fatigue                    lapse +0.03 and preferences x0.939 per
                                    100 trials -> both off together

    plus the two anchors: all defaults, and no nuisances. For each nuisance:
    the power returned when it is switched off (power without it minus power
    with all defaults), with its Monte Carlo SE; the seven ranked by the
    average over 12 cells (2 block sizes x N = 10, 15, 20 x 2 features),
    separately for planning and half shift; and whether the seven recoveries
    add up to the total drop (no nuisances minus all defaults).
    Power only; paired t-test only; fixed weights; pair-specific preferences
    off.

PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES
    The responders' shift, the nuisances' default values and every other
    person-model value are planning assumptions from script 7. The ranking
    says which of those ASSUMED nuisances matter most at their ASSUMED
    sizes; it is not a measurement of real participants.

COMMON RANDOM NUMBERS
    All nine conditions of one effect and block size are simulated from the
    same random draws (same people, same tastes, same strategies, same
    choices' random numbers) and the same study draws (which pool members
    are sampled into each study), so a recovery reflects only the nuisance
    switched off. Two consequences, stated in the report:
      * script 7's simulate_pool draws the person x scene deviations only
        when scene dependence is on, so switching it to exactly 0 would shift
        every later draw. "Scene dependence off" is therefore SD x 1e-9
        (SCENE_OFF_MULT): the draws stay aligned, and the deviations are
        below 1e-9 logit, numerically nothing.
      * changing the lapse rate (fatigue off, no nuisances) changes catch
        failures and hence who is excluded, so in those conditions some
        studies sample different people; the report gives how many.
    SEs: every rate and every difference is computed per study and its SE
    measured from script 7's sub-pools (each study draws from one sub-pool);
    for differences this is the SE of the paired difference, so it is small.
    The average over 12 cells gets its SE from the sub-pool averages.
    The 150 / 180 comparison uses independent pools (script 7's), so the SE
    of a difference there is sqrt(SE1^2 + SE2^2).

REUSED (imported, not copied; SHA-256 of each file recorded and checked)
    simulate_power_paired_v5b_adj.py (script 7): load_stimulus_table,
        load_package, simulate_pool, person_thresholds, draw_studies,
        run_tests, rate_se, scenario_rows (exact reproduction), Fitter,
        EFFECTS, NUISANCE_OFF, NUISANCE_DEFAULT, FATIGUE_VARIANTS and its
        settings (person model, pool size, studies, sub-pools, seed and
        stream key)
    through script 7: simulate_power_v4_n10_20.py, thresholds_v5b.py,
        explore_block_length_v5b.py, analyse_pair_capacity_v5b.py

CHECKS
    * the imported scripts are the expected versions (SHA-256); the frozen
      packages match their checksums (through script 7's loader);
    * EXACT REPRODUCTION: the two anchors, and the four fatigue conditions
      of the size comparison, rerun with script 7's own stream keys give its
      adjusted run's numbers to the last digit;
    * the matched anchors (this script's own streams) agree with script 7
      within Monte Carlo error;
    * the draws are shared: identical responders across the nine
      conditions; exclusions differ only where the lapse rate differs;
    * the total drop is positive (nuisances cost power).

INPUTS (relative to the project folder; run from there)
    PAIRED_RUN_DIR (script 7's adjusted run: paired_power.csv,
    thresholds_adjusted.csv, checks.csv, run_meta.json) and, through
    script 7's loaders, the frozen lists and the stimulus set table

OUTPUTS (one new folder per run under OUTPUT_ROOT; never overwritten)
    report.txt, summary.txt, checks.csv, settings.json, run_meta.json
    ranking.csv           average recovery per nuisance and effect, with SE
    recoveries.csv        per cell (size, N, feature): power and recovery
    anchors.csv           the anchor comparisons with script 7
    size_comparison.csv   150 against 180 under the four fatigue conditions
    fatigue_cost.csv      fatigue's own power cost per size (matched family)
    figure_nuisances.png  recoveries, planning beside half shift
    figure_sizes.png      150 against 180 under the four fatigue conditions

REQUIRES  Python 3.9+, numpy, pandas, scipy, matplotlib
RUN       python nuisance_power_v5c.py               (v5c_180; runs the
                                                   regression first if needed)
          python nuisance_power_v5c.py --regression  (v5b_180 only)
          (each about 4 minutes with 2 workers)
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

PROJECT_ROOT = None          # None = the current folder (run from there)
# MODE
#   "v5c" (default): the v5c package. "regression": the same code on
#   v5b_180 with v5b's inputs (python nuisance_power_v5c.py --regression).
#   Script 2 is imported in the same mode (PAIRED_V5C_MODE); worker
#   processes inherit both through the environment.
MODE_ENV = "NUISANCE_V5C_MODE"
MODE = (os.environ.get(MODE_ENV)
        or ("regression" if "--regression" in sys.argv[1:] else "v5c"))
os.environ["PAIRED_V5C_MODE"] = MODE
SCRIPT7 = "simulate_power_paired_v5c_adj.py"     # v5c script 2
EXPECTED_SHA256 = {
    SCRIPT7:
        "58a5640be3351ac80af350064fc8664409e1ac02f43039278bbe304f2bb98414",
}
PAIRED_RUN_DIR = ("data/power_v5c/paired_adj_v5c/sizes180_N10-20_model-adjusted_pool50000_studies10000_mixed0_seed42__s13156d40")
OUTPUT_ROOT = "data/power_v5c/nuisance_v5c"
FIG_DPI = 200

N_WORKERS = None
CHECK_ONE_WORKER = True

BLOCK_SIZES = [180]
N_LIST = [10, 15, 20]
EFFECTS_RUN = ["planning", "half shift"]
TEST = "t"                       # the registered primary test

# The seven nuisances: plain label (for the report, the figure and the
# sentence) and what "off" sets, starting from NUISANCE_DEFAULT. Values
# are script 7's NUISANCE_OFF values, except scene dependence (below).
SCENE_OFF_MULT = 1e-9            # "off" for scene dependence: see docstring
NUISANCES = [
    ("taste", "taste that differs between the two episodes",
     {"taste_corr": "OFF"}),
    ("scene", "scene-specific preferences",
     {"scene_sd_mult": SCENE_OFF_MULT}),
    ("side", "a preference for one side of the screen",
     {"side_sd": "OFF"}),
    ("dull", "a shared preference for the duller image",
     {"dull": "OFF"}),
    ("typical", "a quarter of people picking the more typical image",
     {"typical_share": "OFF"}),
    ("less_edited", "a quarter of people picking the less edited image",
     {"less_edited_share": "OFF"}),
    ("fatigue", "fatigue building over the session",
     {"fatigue_lapse": "OFF", "fatigue_scale": "OFF"}),
]
SHORT_LABEL = {"taste": "Taste differs between episodes",
               "scene": "Scene-specific preferences",
               "side": "Side preference",
               "dull": "Shared preference for duller images",
               "typical": "Some pick the more typical image",
               "less_edited": "Some pick the less edited image",
               "fatigue": "Fatigue over the session"}
# Conditions in which the lapse rate differs from all defaults (so
# exclusions may differ)
LAPSE_CHANGES = ["fatigue", "none"]

# --- The size comparison ---
#   Script 7's scenarios (all nuisances at defaults; and its fatigue
#   variants), taken from its adjusted run after an exact rerun.
SIZE_CONDITIONS = [
    ("all defaults", "all nuisances (defaults)", "power"),
    ("all defaults without fatigue",
     "all nuisances (defaults), no fatigue", "power (sensitivity)"),
    ("all defaults, fatigue doubled",
     "all nuisances (defaults), fatigue doubled", "power (sensitivity)"),
    ("all defaults, 50% recovery at the rest",
     "all nuisances (defaults), fatigue, 50% recovery at the rest",
     "power (sensitivity)"),
]
TIE = 0.02                       # script 2's POWER_TIE
EXTRA_TIME_TEXT = ("about 4 minutes per session, 6 when responses are slow, "
                   "by script 2's session estimate")

# --- Verdicts ---
NEGLIGIBLE = 0.01                # "costs almost nothing": average below
INTERACT_SE = 2.0                # "interacting": |sum - total| > 2 SE ...
INTERACT_MIN = 0.01              # ... and > 0.01
TOP_K = 3                        # nuisances named in the summary at most

# --- Anchor checks ---
EXACT_TOL = 0.0
MAX_Z = 3.5
SHARE_BEYOND = 0.10

RANDOM_SEED = 42
STREAM_KEY = "nuisance_v5b"    # kept from v5b (the seeds stay as in v5b)

# --- v5b's registered results (comparison and regression reference) ---
REF_RUN = ("data/power_v5b/nuisance_v5b/sizes150-180_N10-15-20_test-t"
           "_model-adjusted_pool50000_studies10000_seed42__sbb19480e")
COMPARE_FLAG = 0.05            # changes larger than this are flagged

# --- Regression (MODE "regression") ---
# REGRESSION_FILES: the files compared with REF_RUN (rows of 180 trials)
#   and their key columns. REGRESSION_EXACT_COLS: powers and recoveries
#   (shares of studies), which must be identical; other values within
#   REGRESSION_TOL (v5b's "exact apart from the last bits").
REGRESSION_ROOT = "data/power_v5c/nuisance_v5c/regression_v5b"
REGRESSION_FILES = [
    ("recoveries.csv", ["size", "effect", "N", "feature"]),
    ("anchors.csv", ["check", "size", "effect", "anchor", "N", "test",
                     "feature"]),
    ("fatigue_cost.csv", ["effect", "size", "N", "feature"]),
]
REGRESSION_EXACT_COLS = tuple(
    [f"power: {c}" for c in ["all defaults"]
     + [k for k, _, _ in NUISANCES] + ["none"]]
    + [f"recovery: {k}" for k, _, _ in NUISANCES]
    + ["total drop", "sum minus total", "this_script", "script7", "diff"])
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
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

FEATS = {"sat": "saturation", "con": "contrast"}
T0 = time.time()
QUIET = False


def log(msg):
    if not QUIET:
        print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def root_path(*parts):
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


SETTING_NAMES = ["PROJECT_ROOT", "SCRIPT7", "EXPECTED_SHA256",
                 "PAIRED_RUN_DIR", "OUTPUT_ROOT", "FIG_DPI", "N_WORKERS",
                 "CHECK_ONE_WORKER", "BLOCK_SIZES", "N_LIST", "EFFECTS_RUN",
                 "TEST", "SCENE_OFF_MULT", "NUISANCES", "SHORT_LABEL",
                 "LAPSE_CHANGES", "SIZE_CONDITIONS", "TIE", "EXTRA_TIME_TEXT",
                 "NEGLIGIBLE", "INTERACT_SE", "INTERACT_MIN", "TOP_K",
                 "EXACT_TOL", "MAX_Z", "SHARE_BEYOND", "RANDOM_SEED",
                 "STREAM_KEY", "MODE", "REF_RUN", "COMPARE_FLAG",
                 "REGRESSION_ROOT", "REGRESSION_FILES",
                 "REGRESSION_EXACT_COLS", "REGRESSION_TOL",
                 "REGRESSION_RUN_FIRST"]


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
        "POOL_PEOPLE", "N_STUDIES", "N_BATCHES", "PERSON_MODEL",
        "NUISANCE_OFF", "NUISANCE_DEFAULT", "EFFECTS", "RANDOM_SEED",
        "STREAM_KEY", "TASTE_SD", "LAPSE_MEAN")}
    return d


def resolve_output_dir(s7):
    s = settings_dict(s7)
    s.pop("N_WORKERS")
    s.pop("REGRESSION_RUN_FIRST")      # cannot change the result
    h = hashlib.sha256(json.dumps(jsonable(s), sort_keys=True).encode()
                       ).hexdigest()[:8]
    name = (f"sizes{'-'.join(map(str, BLOCK_SIZES))}"
            f"_N{'-'.join(map(str, N_LIST))}_test-{TEST}"
            f"_model-{s7.PERSON_MODEL}"
            f"_pool{s7.POOL_PEOPLE}_studies{s7.N_STUDIES}"
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
        os.path.splitext(fname)[0] + "_imported_nuis", path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def conditions(s7):
    """The nine conditions as nuisance dictionaries."""
    d = dict(s7.NUISANCE_DEFAULT)
    out = {"all defaults": d}
    for key, _, off in NUISANCES:
        c = dict(d)
        for field, val in off.items():
            c[field] = s7.NUISANCE_OFF[field] if val == "OFF" else val
        out[key] = c
    none = dict(s7.NUISANCE_OFF)
    none["scene_sd_mult"] = SCENE_OFF_MULT
    out["none"] = none
    return out


def batch_means(x, sb):
    K = sb.max() + 1
    x = np.asarray(x, float)
    return np.array([x[sb == k].mean() for k in range(K)])


# ------------------------------------------------------------
# Tasks
# ------------------------------------------------------------

_W = {}


def _worker_init(state, quiet=True):
    global QUIET
    QUIET = quiet
    s7 = import_file(SCRIPT7)
    s7.QUIET = True
    v4 = s7.load_module(s7.V4_SIM_SCRIPT)
    cap = s7.load_module(s7.CAPACITY_SCRIPT)
    explore = s7.load_module(s7.EXPLORE_SCRIPT)
    s6 = s7.load_module(s7.THRESHOLDS_SCRIPT)
    _W.update(s7=s7, v4=v4, cap=cap, explore=explore, fit=s7.Fitter(v4, s6),
              crit=v4.sign_test_critical(max(max(N_LIST), 20)),
              COND=conditions(s7), **state)


def task_exact(size, eff, name, role):
    """One of script 7's cells, with script 7's own stream keys."""
    s7, cap = _W["s7"], _W["cap"]
    nuis = {s["name"]: s for s in s7.build_scenarios()
            if s["effect"] == eff and s["lists"] == s7.LISTS}[name]["nuis"]
    st = cap.Streams(s7.RANDOM_SEED, s7.STREAM_KEY, "pool",
                     s7.STREAM_LISTS[s7.LISTS], size, name, eff)
    pool = s7.simulate_pool(_W["pk"][size], s7.EFFECTS[eff], nuis,
                            st("people"), _W["fit"])
    s7.person_thresholds(pool, size, _W["TH"])
    rows, _, _, _ = s7.scenario_rows(
        pool, {"name": name, "effect": eff, "role": role, "lists": s7.LISTS},
        size, _W["explore"], _W["crit"], cap)
    return rows


def task_family(size, eff, cond):
    """One condition of the matched family: this script's streams (shared
    by all nine conditions of a size and effect), and per N the studies'
    rejections, the study draws and the people's responder and exclusion
    flags."""
    s7, cap = _W["s7"], _W["cap"]
    st = cap.Streams(RANDOM_SEED, STREAM_KEY, "pool", size, eff)
    pool = s7.simulate_pool(_W["pk"][size], s7.EFFECTS[eff],
                            _W["COND"][cond], st("people"), _W["fit"])
    s7.person_thresholds(pool, size, _W["TH"])
    out = {"responder": pool["responder"], "excluded": pool["excluded"],
           "rej": {}, "idx": {}, "sb": None}
    for N in N_LIST:
        # a fresh stream per condition, the same for all nine: the same
        # study draws wherever exclusions are the same
        sts = cap.Streams(RANDOM_SEED, STREAM_KEY, "studies", size, eff)
        idx, _, sb = s7.draw_studies(pool, N, s7.N_STUDIES, sts(f"N{N}"))
        r = s7.run_tests(pool, idx, _W["explore"], _W["crit"])
        out["rej"][N] = {f: r[(TEST, f)] for f in FEATS}
        out["idx"][N] = idx
        out["sb"] = sb
    return out


TASKS = {"exact": task_exact, "family": task_family}


def _run_task(kind, args):
    return TASKS[kind](*args)


def run_tasks(jobs, n_workers, state):
    if n_workers <= 1:
        _worker_init(state, quiet=False)
        return [_run_task(k, a) for k, a in jobs]
    order = sorted(range(len(jobs)), key=lambda i: jobs[i][0] != "family")
    ctx = mp.get_context("spawn")
    res = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=min(n_workers, len(jobs)),
                             mp_context=ctx, initializer=_worker_init,
                             initargs=(state,)) as ex:
        futs = {i: ex.submit(_run_task, *jobs[i]) for i in order}
        for k, i in enumerate(order):
            res[i] = futs[i].result()
            if (k + 1) % 10 == 0 or k + 1 == len(jobs):
                log(f"  {k + 1} of {len(jobs)} tasks done")
    return res


def same_family(a, b):
    if not (np.array_equal(a["responder"], b["responder"])
            and np.array_equal(a["excluded"], b["excluded"])):
        return False
    for N in N_LIST:
        if not np.array_equal(a["idx"][N], b["idx"][N]):
            return False
        for f in FEATS:
            if not np.array_equal(a["rej"][N][f], b["rej"][N][f]):
                return False
    return True


# ------------------------------------------------------------
# Figures
# ------------------------------------------------------------

FORBIDDEN = [r"\bsat\b", r"\bcon\b", r"\bfog\b", r"_", r"\bnuis\b",
             r"\bodd\b", r"\bdull\b"]


def _scan(fig, bad):
    for t in fig.findobj(lambda a: hasattr(a, "get_text")):
        try:
            s = t.get_text()
        except Exception:
            continue
        for pat in FORBIDDEN:
            if s and re.search(pat, s):
                bad.append(s)


def figure(rank, out, bad):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = (rank[rank["effect"] == "planning"]
             .sort_values("recovery")["nuisance"].tolist())
    fig, axes = plt.subplots(1, len(EFFECTS_RUN), figsize=(10, 4.2),
                             sharey=True, sharex=True)
    for ax, eff in zip(np.atleast_1d(axes), EFFECTS_RUN):
        q = rank[rank["effect"] == eff].set_index("nuisance").loc[order]
        y = np.arange(len(order))
        ax.barh(y, q["recovery"], xerr=2 * q["recovery_se"],
                color="#0072B2", alpha=0.85, capsize=3,
                error_kw={"lw": 0.8})
        ax.axvline(0, color="#444444", lw=0.8)
        ax.axvline(NEGLIGIBLE, color="#888888", lw=0.8, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT_LABEL[k] for k in order], fontsize=8)
        ax.set_title(f"{'Planning' if eff == 'planning' else 'Half shift'}",
                     fontsize=10)
        ax.set_xlabel("Power returned when switched off\n(average of "
                      f"{n_cells()} cells; bars: 2 SE)", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Primary paired t-test: power each assumed nuisance costs "
                 "(dotted line: 0.01)", fontsize=10)
    fig.tight_layout()
    _scan(fig, bad)
    fig.savefig(os.path.join(out, "figure_nuisances.png"), dpi=FIG_DPI)
    plt.close(fig)


def figure_sizes(sz, out, bad):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
    fig, axes = plt.subplots(len(EFFECTS_RUN), 2, figsize=(10, 7),
                             sharey=True, squeeze=False)
    for i, eff in enumerate(EFFECTS_RUN):
        for j, f in enumerate(("contrast", "saturation")):
            ax = axes[i, j]
            for c, (lab, _, _) in enumerate(SIZE_CONDITIONS):
                q = sz[(sz["effect"] == eff) & (sz["feature"] == f)
                       & (sz["condition"] == lab)].sort_values("N")
                ax.errorbar(q["N"] + (c - 1.5) * 0.15, q["difference"],
                            yerr=2 * q["difference_se"], fmt="o-",
                            color=colours[c], ms=3, capsize=2, lw=1,
                            label=lab)
            ax.axhspan(-TIE, TIE, color="#eeeeee", zorder=0)
            ax.axhline(0, color="#444444", lw=0.8)
            ax.set_title(f"{'Planning' if eff == 'planning' else 'Half shift'}"
                         f", {f}", fontsize=10)
            ax.set_xlabel("Report-group participants (analysable)")
            if j == 0:
                ax.set_ylabel("Power at 180 minus power at 150\n(grey band: "
                              f"a tie, within {TIE})")
            ax.set_xticks(N_LIST)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color="#dddddd", lw=0.6)
    axes[0, -1].legend(fontsize=7, frameon=False)
    fig.suptitle("Primary paired t-test: 180 against 150 trials per block "
                 "(bars: 2 SE; assumptions)", fontsize=10)
    fig.tight_layout()
    _scan(fig, bad)
    fig.savefig(os.path.join(out, "figure_sizes.png"), dpi=FIG_DPI)
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

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
    reg = MODE == "regression"
    pk = ("data/frozen_trial_lists_v5b/v5b_180" if reg
          else "data/frozen_trial_lists_v5c/v5c_180")
    paths = [SCRIPT7, "simulate_power_v4_n10_20.py",
             ("analyse_pair_capacity_v5b.py" if reg
              else "analyse_pair_capacity_v5c.py"),
             ("explore_block_length_v5b.py" if reg
              else "explore_block_length_v5c.py"), "thresholds_v5c.py",
             "data/stimulus_set_v5/7fd72262ba/stimulus_manifest.csv",
             "data/stimulus_set_v5/7fd72262ba/reference_stimulus_path.csv"]
    paths += package_paths(pk)
    paths += [f"{PAIRED_RUN_DIR}/{f}" for f in (
        "paired_power.csv", "thresholds_adjusted.csv", "checks.csv",
        "run_meta.json")]
    paths += [f"{REF_RUN}/{fn}" for fn in ("recoveries.csv", "anchors.csv",
                                           "fatigue_cost.csv",
                                           "ranking.csv")]
    return paths


def n_cells():
    return len(BLOCK_SIZES) * len(N_LIST) * len(FEATS)


def compare_with_v5b(out, rank, cells, addv, fat_avg):
    """The v5c results next to v5b's registered results (nuisance_v5b).
    v5b ranked the nuisances over both block sizes (12 cells); the like-for-
    like comparison is v5b's ranking recomputed from its 180-trial cells
    (recoveries.csv, the same average over 6 cells as here)."""
    s = BLOCK_SIZES[0]
    rc = _read(f"{REF_RUN}/recoveries.csv")
    rc = rc[rc["size"] == s]
    rk = _read(f"{REF_RUN}/ranking.csv")
    rows = []
    for eff in EFFECTS_RUN:
        b = rc[rc["effect"] == eff]
        for key, label, _ in NUISANCES:
            v5b180 = float(b[f"recovery: {key}"].mean())
            mine = rank[(rank["effect"] == eff)
                        & (rank["nuisance"] == key)].iloc[0]
            reg_ = rk[(rk["effect"] == eff) & (rk["nuisance"] == key)]
            rows.append({"group": "ranking (average recovery)",
                         "effect": eff, "nuisance": key, "label": label,
                         "v5b_registered_150_180":
                             float(reg_["recovery"].iloc[0])
                             if len(reg_) else np.nan,
                         "v5b": v5b180, "v5c": float(mine["recovery"])})
        rows.append({"group": "ranking (average recovery)", "effect": eff,
                     "nuisance": "total drop", "label": "all nuisances "
                     "together (no nuisances minus all defaults)",
                     "v5b_registered_150_180": np.nan,
                     "v5b": float(b["total drop"].mean()),
                     "v5c": float(addv[addv["effect"] == eff][
                         "total_drop"].iloc[0])})
    r = pd.DataFrame(rows)
    for eff in EFFECTS_RUN:
        m = (r["effect"] == eff) & (r["nuisance"] != "total drop")
        r.loc[m, "rank_v5b"] = r.loc[m, "v5b"].rank(
            ascending=False, method="first")
        r.loc[m, "rank_v5c"] = r.loc[m, "v5c"].rank(
            ascending=False, method="first")
    r["rank_v5b"] = r["rank_v5b"].astype("Int64")
    r["rank_v5c"] = r["rank_v5c"].astype("Int64")
    r["change"] = r["v5c"] - r["v5b"]
    r["flag"] = np.where(r["change"].abs() > COMPARE_FLAG,
                         f"CHANGED > {COMPARE_FLAG:g}", "")
    meas = ([f"recovery: {k}" for k, _, _ in NUISANCES]
            + ["total drop", "power: all defaults", "power: none"])
    c = long_compare(cells[cells["size"] == s], rc,
                     ["effect", "N", "feature"], meas,
                     "per cell (180 trials)")
    comp = pd.concat([r, c], ignore_index=True)
    comp.to_csv(os.path.join(out, "comparison.csv"), index=False)
    return comp


def comparison_lines(comp):
    L = ["11. COMPARISON WITH v5b'S REGISTERED RESULTS (nuisance_v5b)",
         "-" * 78,
         f"v5b: {REF_RUN.split('/')[-1]}. v5b ranked the nuisances over 150 "
         "and 180 trials (12 cells);",
         "v5c has 180 trials only (6 cells: N = 10, 15, 20 x 2 features), so "
         "the like-for-like v5b",
         "value ('v5b') is v5b's average over its 6 cells at 180 trials, from "
         "its recoveries.csv. v5b's",
         "registered 12-cell value is shown beside it. The same streams and "
         "seeds, so the differences",
         f"come from the lists. '*'/flag: a change larger than "
         f"{COMPARE_FLAG:g}. 150 against 180 trials: not applicable.", ""]
    r = comp[comp["group"] == "ranking (average recovery)"]
    for eff in EFFECTS_RUN:
        q = r[r["effect"] == eff]
        L.append(f"  {eff}")
        L.append("  " + q[["label", "v5b_registered_150_180", "v5b", "v5c",
                           "change", "rank_v5b", "rank_v5c", "flag"]]
                 .rename(columns={"v5b_registered_150_180":
                                  "v5b (150+180)", "v5b": "v5b (180)",
                                  "v5c": "v5c (180)"})
                 .to_string(index=False, float_format=lambda x: f"{x:+.3f}")
                 .replace("\n", "\n  "))
        L.append("")
    c = comp[comp["group"] == "per cell (180 trials)"]
    L.append(f"  Per cell (180 trials, N 10/15/20, both features): "
             f"{int((c['flag'] != '').sum())} of {len(c)} recoveries and "
             f"powers changed by more than {COMPARE_FLAG:g} (comparison.csv).")
    q = c[c["measure"].isin(["power: all defaults", "power: none"])]
    L.append("  Power with all nuisances at defaults and with none (paired "
             "t-test):")
    L.append("  " + wide_compare(q.assign(row=q["effect"] + ", "
                                          + q["measure"] + ", "
                                          + q["feature"]),
                                 ["row"], "N", N_LIST).to_string(
        index=False, float_format=lambda x: f"{x:.3f}").replace("\n", "\n  "))
    L.append("")
    return L


def summary_v5c(comp, reg):
    L = []
    if isinstance(reg, pd.DataFrame):
        ok = (reg["status"] == "PASS").all()
        L.append(f"Regression on v5b_180: {'PASS' if ok else 'FAIL'} "
                 f"({int((reg['status'] == 'PASS').sum())} of {len(reg)} "
                 "files reproduce v5b's nuisance_v5b run at 180 trials: "
                 f"rates identical, other values within {REGRESSION_TOL:g}).")
    elif reg is not None:
        t = reg["table"]
        L.append(f"Regression on v5b_180: PASS ({len(t)} of {len(t)} files "
                 "reproduce v5b's nuisance_v5b run at 180 trials; "
                 + reg["dir"] + ").")
    if comp is None:
        return L
    r = comp[comp["group"] == "ranking (average recovery)"]
    for eff in EFFECTS_RUN:
        q = r[(r["effect"] == eff) & (r["nuisance"] != "total drop")]
        t = r[(r["effect"] == eff) & (r["nuisance"] == "total drop")].iloc[0]
        order_c = " > ".join(f"{x.nuisance} {x.v5c:+.3f}" for x in
                             q.sort_values("rank_v5c").itertuples())
        order_b = " > ".join(f"{x.nuisance} {x.v5b:+.3f}" for x in
                             q.sort_values("rank_v5b").itertuples())
        L.append(f"Nuisance ranking, {eff}, 180 trials (power returned when "
                 f"each is switched off, average of 6 cells): v5c {order_c}; "
                 f"v5b {order_b}. All together: v5c {t.v5c:.3f}, v5b "
                 f"{t.v5b:.3f}. Same order: "
                 f"{'yes' if list(q.sort_values('rank_v5c')['nuisance']) == list(q.sort_values('rank_v5b')['nuisance']) else 'no'}.")
    allc = comp
    L.append(f"{int((allc['flag'] != '').sum())} of {len(allc)} compared "
             f"values changed by more than {COMPARE_FLAG:g} (report section "
             "11, comparison.csv). 150 against 180 trials: not applicable "
             "(v5c has only the 180-trial package).")
    return L


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
    p = root_path(SCRIPT7)
    ok = os.path.exists(p)
    ck.add("inputs", f"{SCRIPT7} exists", ok, p)
    if not ok:
        raise SystemExit(f"missing {p}")
    hashes[SCRIPT7] = sha256_of(p)
    ck.add("inputs", f"{SCRIPT7} is the expected version (SHA-256)",
           hashes[SCRIPT7] == EXPECTED_SHA256[SCRIPT7],
           hashes[SCRIPT7][:16] + "...")
    s7 = import_file(SCRIPT7)
    s7.QUIET = True
    ck.add("inputs", "script 2 (imported as script 7) runs in this script's "
           "mode, on the matching package", s7.MODE == MODE,
           f"script 2 mode {s7.MODE}, package "
           + ", ".join(s7.PACKAGES[s] for s in BLOCK_SIZES))
    n_workers = N_WORKERS or max(1, os.cpu_count() or 1)
    # script 7's inputs, through its loaders and checks
    with open(os.path.join(root_path(s7.PACKAGES[BLOCK_SIZES[0]]),
                           "FROZEN.json")) as f:
        frozen0 = json.load(f)
    stim = s7.load_stimulus_table(frozen0, ck, hashes)
    pk = {size: s7.load_package(size, ck, hashes, stim, s7.LISTS)
          for size in BLOCK_SIZES}
    s7.load_module(s7.V4_SIM_SCRIPT, ck, hashes, frozen0["scripts_sha256"])
    s7.load_module(s7.CAPACITY_SCRIPT, ck, hashes, frozen0["scripts_sha256"])
    s7.load_module(s7.EXPLORE_SCRIPT, ck, hashes)
    s7.load_module(s7.THRESHOLDS_SCRIPT, ck, hashes)
    ck.add("inputs", "script 7's person model is the registered adjusted "
           "model", s7.PERSON_MODEL == "adjusted", s7.PERSON_MODEL)

    # --- script 7's adjusted run ---
    d7 = root_path(PAIRED_RUN_DIR)
    need = ["paired_power.csv", "thresholds_adjusted.csv", "checks.csv",
            "run_meta.json"]
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
           str(meta7.get("person_model")))
    ck.add("inputs", f"script 7's run was made in this mode ({MODE})",
           meta7.get("mode") == MODE, f"its mode {meta7.get('mode')}")
    same = [k for k in meta7["input_sha256"]
            if k.startswith(tuple(s7.PACKAGES.values())) and k in hashes]
    ck.add("inputs", "script 7's run read the same frozen package files",
           len(same) > 0 and all(meta7["input_sha256"][k] == hashes[k]
                                 for k in same), f"{len(same)} files")
    # round_trip: the floats exactly as script 7 wrote them
    pw7 = pd.read_csv(os.path.join(d7, "paired_power.csv"),
                      float_precision="round_trip")
    pw7 = pw7[pw7["lists"] == s7.LISTS]
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
    out = resolve_output_dir(s7)
    log(f"output folder {out}")
    log(f"{n_workers} worker process(es), one thread each")
    COND = conditions(s7)
    ck.add("setup", "the nine conditions differ from all defaults only in "
           "the nuisance switched off",
           all(sum(COND[k][f] != COND["all defaults"][f]
                   for f in COND["all defaults"]) == len(off)
               for k, _, off in NUISANCES), "")
    names7 = {v for _, v, _ in SIZE_CONDITIONS}
    ok = all(((pw7["scenario"] == n) & (pw7["effect"] == e)).any()
             for n in names7 | {"no nuisances"} for e in EFFECTS_RUN)
    ck.add("setup", "the size-comparison and anchor scenarios exist in "
           "script 7's run", ok, f"{len(names7) + 1} scenarios")
    fv = dict(s7.FATIGUE_VARIANTS)
    ck.add("setup", "script 7's fatigue variants are the size comparison's "
           "(no fatigue, doubled, 50% recovery)",
           fv["no fatigue"] == {"fatigue_lapse": 0.0, "fatigue_scale": 1.0}
           and fv["fatigue doubled"]["fatigue_lapse"]
           == 2 * s7.FATIGUE_LAPSE_PER_100
           and fv["fatigue, 50% recovery at the rest"]["fatigue_recovery"]
           == 0.5, str(fv))

    # --- tasks ---
    exact_cells = ([(size, eff, "no nuisances", "power (no nuisances)")
                    for size in BLOCK_SIZES for eff in EFFECTS_RUN]
                   + [(size, eff, name7, role)
                      for size in BLOCK_SIZES for eff in EFFECTS_RUN
                      for _, name7, role in SIZE_CONDITIONS])
    fam = [(size, eff, cond) for size in BLOCK_SIZES for eff in EFFECTS_RUN
           for cond in COND]
    jobs = ([("exact", a) for a in exact_cells]
            + [("family", a) for a in fam])
    state = {"pk": pk, "TH": TH}
    log(f"{len(jobs)} tasks ({len(exact_cells)} of script 7's cells, "
        f"{len(fam)} matched-family pools)")
    res = run_tasks(jobs, n_workers, state)
    if CHECK_ONE_WORKER and n_workers > 1:
        _worker_init(state, quiet=False)
        j0 = [k for k, j in enumerate(jobs) if j[0] == "family"][0]
        log(f"rerunning task {jobs[j0]} in this process (one worker)")
        again = task_family(*jobs[j0][1])
        ck.add("workers", f"one worker gives the same result as {n_workers}",
               same_family(again, res[j0]), f"{jobs[j0][1]}")
    elif CHECK_ONE_WORKER:
        ck.add("workers", "one worker gives the same result (run with one "
               "worker: nothing to compare)", True, "N_WORKERS = 1")

    # --- 1. exact reproduction of script 7 with its own stream keys ---
    anchor_rows = []
    exact_rows = []
    for (kind, args), r in zip(jobs, res):
        if kind != "exact":
            continue
        size, eff, name, role = args
        mine = pd.DataFrame(r)
        exact_rows.append(mine)
        ref = pw7[(pw7["scenario"] == name) & (pw7["effect"] == eff)
                  & (pw7["size"] == size) & (pw7["role"] == role)]
        m = mine.merge(ref, on=["N", "test", "feature"], suffixes=("", "_s7"))
        for rr in m.itertuples():
            anchor_rows.append({
                "check": "exact (script 7's stream keys)", "size": size,
                "effect": eff, "anchor": name, "N": rr.N, "test": rr.test,
                "feature": rr.feature, "this_script": rr.rate,
                "script7": rr.rate_s7, "diff": rr.rate - rr.rate_s7})
        log(f"exact reproduction {size} {eff} {name}: max |diff| "
            f"{np.max(np.abs(m['rate'] - m['rate_s7'])):.1e} over "
            f"{len(m)} cells")
    ex = pd.DataFrame(anchor_rows)
    n_expected = len(exact_cells) * len(s7.N_GRID) * len(s7.TESTS) * 2
    ck.add("anchors", "rerun with script 7's own stream keys, the anchors "
           "and the size-comparison conditions reproduce script 7's adjusted "
           "run exactly (every N, both tests, both features)",
           len(ex) == n_expected and (ex["diff"].abs() <= EXACT_TOL).all(),
           f"{len(ex)} of {n_expected} cells; largest |diff| "
           f"{ex['diff'].abs().max():.1e}")

    # --- 2. the matched family ---
    famres = {args: r for (kind, args), r in zip(jobs, res)
              if kind == "family"}
    cell_rows, rank_rows, crn_rows = [], [], []
    batch = {}          # (effect, cond) -> list of per-cell batch means
    fat_rows = []
    for size in BLOCK_SIZES:
        for eff in EFFECTS_RUN:
            pools = {cond: famres[(size, eff, cond)] for cond in COND}
            base = pools["all defaults"]
            same_resp = all(np.array_equal(p["responder"], base["responder"])
                            for p in pools.values())
            for cond, p in pools.items():
                crn_rows.append({
                    "size": size, "effect": eff, "condition": cond,
                    "same_responders": bool(np.array_equal(
                        p["responder"], base["responder"])),
                    "exclusion_differs": float(np.mean(
                        p["excluded"] != base["excluded"]))})
            ck.add("common draws", f"{size} trials, {eff}: identical "
                   f"responders in all {len(COND)} conditions", same_resp, "")
            for N in N_LIST:
                sb0 = base["sb"]
                rej = {cond: {f: p["rej"][N][f].astype(float) for f in FEATS}
                       for cond, p in pools.items()}
                for cond, p in pools.items():
                    crn_rows.append({
                        "size": size, "effect": eff, "condition": cond,
                        "N": N, "studies_same_people": float(np.mean(
                            np.all(p["idx"][N] == base["idx"][N], axis=1)))})
                for f, flab in FEATS.items():
                    d0 = rej["all defaults"][f]
                    total = rej["none"][f] - d0
                    ssum = np.zeros_like(d0)
                    row = {"size": size, "effect": eff, "N": N,
                           "feature": flab}
                    for cond in COND:
                        pwr, se = s7.rate_se(rej[cond][f], sb0)
                        row[f"power: {cond}"] = pwr
                        row[f"power SE: {cond}"] = se
                    for key, _, _ in NUISANCES:
                        dk = rej[key][f] - d0
                        ssum += dk
                        m, se = s7.rate_se(dk, sb0)
                        row[f"recovery: {key}"] = m
                        row[f"recovery SE: {key}"] = se
                        batch.setdefault((eff, key), []).append(
                            batch_means(dk, sb0))
                        if key == "fatigue":
                            batch.setdefault((eff, "fatigue", size),
                                             []).append(batch_means(dk, sb0))
                    m, se = s7.rate_se(total, sb0)
                    row["total drop"], row["total drop SE"] = m, se
                    batch.setdefault((eff, "total"), []).append(
                        batch_means(total, sb0))
                    inter = ssum - total
                    m, se = s7.rate_se(inter, sb0)
                    row["sum minus total"], row["sum minus total SE"] = m, se
                    batch.setdefault((eff, "interaction"), []).append(
                        batch_means(inter, sb0))
                    batch.setdefault((eff, "sum"), []).append(
                        batch_means(ssum, sb0))
                    batch.setdefault((eff, "anchor all"), []).append(
                        (size, N, flab, *s7.rate_se(d0, sb0)))
                    batch.setdefault((eff, "anchor none"), []).append(
                        (size, N, flab, *s7.rate_se(rej["none"][f], sb0)))
                    cell_rows.append(row)
                    fat_rows.append({"effect": eff, "size": size, "N": N,
                                     "feature": flab,
                                     "fatigue_cost": row["recovery: fatigue"],
                                     "fatigue_cost_se":
                                         row["recovery SE: fatigue"]})
    cells = pd.DataFrame(cell_rows)
    crn = pd.DataFrame(crn_rows)
    fat = pd.DataFrame(fat_rows)

    def avg(*key):
        B = np.mean(np.vstack(batch[key]), axis=0)  # per sub-pool
        return float(B.mean()), float(B.std(ddof=1) / np.sqrt(len(B)))

    for eff in EFFECTS_RUN:
        for key, label, _ in NUISANCES:
            m, se = avg(eff, key)
            rank_rows.append({"effect": eff, "nuisance": key,
                              "label": label, "recovery": m,
                              "recovery_se": se})
    fat_avg = pd.DataFrame([{"effect": eff, "size": size,
                             "fatigue_cost": avg(eff, "fatigue", size)[0],
                             "fatigue_cost_se": avg(eff, "fatigue", size)[1]}
                            for eff in EFFECTS_RUN for size in BLOCK_SIZES])

    def batch_diff(eff, k1, k2):
        """Average recovery of k1 minus k2, with SE from the sub-pools."""
        B = (np.mean(np.vstack(batch[(eff, k1)]), axis=0)
             - np.mean(np.vstack(batch[(eff, k2)]), axis=0))
        return float(B.mean()), float(B.std(ddof=1) / np.sqrt(len(B)))

    rank = pd.DataFrame(rank_rows)
    rank["rank"] = rank.groupby("effect")["recovery"].rank(
        ascending=False, method="first").astype(int)
    rank = rank.sort_values(["effect", "rank"])
    add_rows = []
    for eff in EFFECTS_RUN:
        t_, tse = avg(eff, "total")
        s_, sse = avg(eff, "sum")
        i, ise = avg(eff, "interaction")
        add_rows.append({"effect": eff, "total_drop": t_,
                         "total_drop_se": tse, "sum_of_seven": s_,
                         "sum_se": sse, "sum_minus_total": i,
                         "sum_minus_total_se": ise,
                         "ratio": s_ / t_ if t_ else np.nan,
                         "interacting": abs(i) > INTERACT_SE * ise
                         and abs(i) > INTERACT_MIN})
    addv = pd.DataFrame(add_rows)

    # --- checks on the family ---
    oth = crn[crn["N"].isna() & ~crn["condition"].isin(LAPSE_CHANGES)]
    ck.add("common draws", "exclusions identical to all defaults wherever "
           "the lapse rate is unchanged (all but fatigue off and no "
           "nuisances)", (oth["exclusion_differs"] == 0).all(),
           f"largest share differing {oth['exclusion_differs'].max():.4f}")
    od = crn[crn["N"].isna() & crn["condition"].isin(LAPSE_CHANGES)]
    sp = crn[crn["N"].notna()]
    ck.add("common draws", "studies sample the same people as with all "
           "defaults wherever exclusions are unchanged",
           (sp[~sp["condition"].isin(LAPSE_CHANGES)][
               "studies_same_people"] == 1).all(),
           f"fatigue off / no nuisances: {od['exclusion_differs'].mean():.1%}"
           " of people excluded differently, "
           f"{sp[sp['condition'].isin(LAPSE_CHANGES)]['studies_same_people'].mean():.1%}"
           " of studies with the same people")
    ar = []
    for eff in EFFECTS_RUN:
        for tag, name, role in (("anchor all", "all nuisances (defaults)",
                                 "power"),
                                ("anchor none", "no nuisances",
                                 "power (no nuisances)")):
            for size, N, flab, m, se in batch[(eff, tag)]:
                r7 = pw7[(pw7["scenario"] == name) & (pw7["effect"] == eff)
                         & (pw7["size"] == size) & (pw7["N"] == N)
                         & (pw7["test"] == TEST) & (pw7["feature"] == flab)
                         & (pw7["role"] == role)].iloc[0]
                ar.append({"check": "matched family (own streams)",
                           "size": size, "effect": eff, "anchor": name,
                           "N": N, "test": TEST, "feature": flab,
                           "this_script": m, "script7": r7["rate"],
                           "diff": m - r7["rate"],
                           "z": (m - r7["rate"]) / np.sqrt(se ** 2 +
                                                           r7["se"] ** 2)})
    mf = pd.DataFrame(ar)
    az = mf["z"].abs()
    ck.add("anchors", "the matched family's anchors agree with script 7 "
           "within Monte Carlo error (scene off as SD x1e-9 in no nuisances)",
           az.max() <= MAX_Z and (az > 1.96).mean() <= SHARE_BEYOND,
           f"{len(mf)} cells; largest |z| {az.max():.2f}; "
           f"{(az > 1.96).mean():.0%} beyond 1.96")
    ck.add("results", "nuisances cost power: total drop positive for both "
           "effects", (addv["total_drop"] > 0).all(),
           ", ".join(f"{r.effect} {r.total_drop:.3f}"
                     for r in addv.itertuples()))
    anchors = pd.concat([ex, mf], ignore_index=True)

    # --- 3. 150 against 180 (script 7's rows, checked above) ---
    sz_rows = []
    for eff in (EFFECTS_RUN if len(BLOCK_SIZES) >= 2 else []):   # v5c: n/a
        for lab, name7, role in SIZE_CONDITIONS:
            for N in N_LIST:
                for flab in FEATS.values():
                    r = {}
                    for size in BLOCK_SIZES:
                        q = pw7[(pw7["scenario"] == name7)
                                & (pw7["effect"] == eff)
                                & (pw7["role"] == role)
                                & (pw7["size"] == size) & (pw7["N"] == N)
                                & (pw7["test"] == TEST)
                                & (pw7["feature"] == flab)].iloc[0]
                        r[size] = (float(q["rate"]), float(q["se"]))
                    a, b = BLOCK_SIZES
                    d = r[b][0] - r[a][0]
                    dse = float(np.hypot(r[a][1], r[b][1]))
                    sz_rows.append({
                        "effect": eff, "condition": lab, "N": N,
                        "feature": flab, f"power_{a}": r[a][0],
                        f"se_{a}": r[a][1], f"power_{b}": r[b][0],
                        f"se_{b}": r[b][1], "difference": d,
                        "difference_se": dse,
                        "verdict": ("tie" if abs(d) <= TIE else
                                    (f"{b} higher" if d > 0
                                     else f"{a} higher"))})
    sz = pd.DataFrame(sz_rows)

    bad = []
    figure(rank, out, bad)
    if len(sz):
        figure_sizes(sz, out, bad)
    ck.add("figure", "figure text uses plain labels", not bad,
           "; ".join(bad[:3]))

    # --- write ---
    rank.to_csv(os.path.join(out, "ranking.csv"), index=False)
    cells.to_csv(os.path.join(out, "recoveries.csv"), index=False)
    anchors.to_csv(os.path.join(out, "anchors.csv"), index=False)
    sz.to_csv(os.path.join(out, "size_comparison.csv"), index=False)
    pd.concat([fat, fat_avg.assign(N="average of 6 cells",
                                   feature="both")],
              ignore_index=True).to_csv(
        os.path.join(out, "fatigue_cost.csv"), index=False)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(jsonable(settings_dict(s7)), f, indent=2)
    # --- v5c: regression (MODE regression) or comparison with v5b ---
    if MODE == "regression":
        reg = regression_compare(out, ck)
        extra = regression_lines(reg, None, "10.")
        comp = None
    else:
        reg = None
        extra = regression_lines(None, reg_info, "10.")
        comp = compare_with_v5b(out, rank, cells, addv, fat_avg)
        extra += comparison_lines(comp)
    pd.DataFrame(ck.rows).to_csv(os.path.join(out, "checks.csv"),
                                 index=False)
    minutes = (time.time() - T0) / 60
    summary = write_report(out, ck, rank, addv, cells, crn, mf, ex, s7,
                           started, minutes, batch_diff, sz, fat, fat_avg,
                           n_workers, extra)
    summary += summary_v5c(comp, reg if MODE == "regression" else reg_info)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("WHICH NUISANCES COST THE PRIMARY TEST POWER, AND 150 "
                "AGAINST 180 (nuisance_power_v5c.py"
                + (", REGRESSION ON v5b_180" if MODE == "regression" else "")
                + "): IN PLAIN WORDS\n\n"
                + "\n\n".join(summary) + "\n")
    with open(os.path.join(out, "run_meta.json"), "w") as f:
        json.dump(jsonable({
            "started": started, "minutes": minutes, "workers": n_workers,
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "output_dir": out,
            "person_model": s7.PERSON_MODEL,
            "input_sha256": hashes,
            "ranking": rank.to_dict("records"),
            "additivity": addv.to_dict("records"),
            "fatigue_cost": fat_avg.to_dict("records"),
            "mode": MODE, "script": os.path.basename(__file__),
            "script_sha256": script_sha,
            "regression_pass": (bool((reg["status"] == "PASS").all())
                                if MODE == "regression" else None),
            "regression_run": (reg_info or {}).get("dir"),
            "checks_pass": len(ck.rows) - ck.n_fail,
            "checks_fail": ck.n_fail}), f, indent=2)
    log(f"done: {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL")
    log(f"report: {os.path.join(out, 'report.txt')}")


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

def wrap(text, width=76, indent="  "):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width:
            lines.append(indent + cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    if cur:
        lines.append(indent + cur)
    return "\n".join(lines)


def classify(r):
    if r >= NEGLIGIBLE:
        return "costs"
    if r <= -NEGLIGIBLE:
        return "raises"
    return "nothing"


def order_comparison(rank, batch_diff):
    """Plain description of how the planning and half-shift rankings
    differ, with whether each swapped pair differs beyond 2 SE."""
    p = list(rank[rank["effect"] == "planning"].sort_values("rank")
             ["nuisance"])
    h = list(rank[rank["effect"] == "half shift"].sort_values("rank")
             ["nuisance"])
    if p == h:
        return True, "The order is the same under half shift."
    lab = dict(zip(rank["nuisance"], rank["label"]))
    swaps = []
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            a, b = p[i], p[j]
            if h.index(a) > h.index(b):
                d, se = batch_diff("half shift", b, a)
                sig = abs(d) > 2 * se
                swaps.append(f"{lab[a]} and {lab[b]} swap places under half "
                             f"shift (difference there {d:+.3f}, SE "
                             f"{se:.3f}; "
                             f"{'beyond' if sig else 'within'} 2 SE)")
    return False, ("The order differs: " + "; ".join(swaps) + ".")


def join(names):
    names = list(names)
    if not names:
        return ""
    return names[0] if len(names) == 1 else \
        ", ".join(names[:-1]) + " and " + names[-1]


def summary_sentences(rank, addv, batch_diff):
    p = rank[rank["effect"] == "planning"].sort_values("rank")
    p = p.assign(kind=p["recovery"].map(classify))
    big = p[p["kind"] == "costs"].head(TOP_K)
    small = p[p["kind"] == "nothing"]
    helps = p[p["kind"] == "raises"]
    same, order_txt = order_comparison(rank, batch_diff)
    a = addv.set_index("effect")
    big_txt = "; ".join(f"{r.label} ({r.recovery:.3f})"
                        for r in big.itertuples())
    plain = (
        f"Under planning, all nuisances together cost the primary test "
        f"{a.loc['planning', 'total_drop']:.3f} of power on average "
        f"(half shift: {a.loc['half shift', 'total_drop']:.3f}). The "
        f"largest costs come from {big_txt or 'none of them'} (power "
        f"returned when each is switched off on its own; average over "
        f"{', '.join(map(str, BLOCK_SIZES))} trials per block, N = 10, 15, "
        f"20 and both features). "
        + (f"Almost nothing (less than {NEGLIGIBLE} either way) is lost to "
           f"{join(small['label'])}. " if len(small) else "")
        + (f"{join(helps['label']).capitalize()} slightly RAISES power when "
           f"present ({'; '.join(f'{r.recovery:+.3f}' for r in helps.itertuples())}"
           f" when switched off). " if len(helps) else "")
        + order_txt + " "
        + ("The seven costs add up to the total drop approximately."
           if not a["interacting"].any() else
           f"The seven separate costs add up to "
           f"{a.loc['planning', 'ratio']:.0%} (planning) and "
           f"{a.loc['half shift', 'ratio']:.0%} (half shift) of the total "
           f"drop, so the nuisances are not additive (section 4)."))
    lo = big["recovery"].min() if len(big) else 0
    hi = big["recovery"].max() if len(big) else 0
    supp = (
        f"Most of the power lost to the assumed nuisances came from "
        f"{join(big['label']) or 'none of them'}: switching any one of these "
        f"off restored about {lo:.2f} to {hi:.2f} of the "
        f"{a.loc['planning', 'total_drop']:.2f} lost under the planning "
        f"scenario (average over {', '.join(map(str, BLOCK_SIZES))} trials "
        f"per block, sample sizes of 10 to 20 and both features)"
        + ("; the nuisances are not additive, so their separate costs do "
           "not sum to the total" if a["interacting"].any() else "")
        + "."
        + (f" {join(small['label']).capitalize()} cost almost no power"
           if len(small) else "")
        + (f", and {join(helps['label'])} slightly raised it"
           if len(helps) and len(small) else
           (f" {join(helps['label']).capitalize()} slightly raised it"
            if len(helps) else ""))
        + ("." if len(small) or len(helps) else ""))
    return plain, supp, same


def size_sentences(sz, fat_avg):
    """The plain summary of 150 against 180, per effect."""
    if len(BLOCK_SIZES) < 2:                    # v5c: one package
        fa = fat_avg.set_index(["effect", "size"])
        return [
            "150 against 180 trials per block: NOT APPLICABLE (v5c has only "
            f"the {BLOCK_SIZES[0]}-trial package). Fatigue itself costs "
            + "; ".join(f"{eff} {fa.loc[(eff, s), 'fatigue_cost']:.3f} (SE "
                        f"{fa.loc[(eff, s), 'fatigue_cost_se']:.3f})"
                        for eff in EFFECTS_RUN for s in BLOCK_SIZES)
            + f" at {BLOCK_SIZES[0]} trials (power returned when it is "
            "switched off; average of 6 cells). [ASSUMPTIONS: the effect, the "
            "nuisances and the fatigue model are planning values.]"]
    a, b = BLOCK_SIZES
    out = []
    for eff in EFFECTS_RUN:
        q = sz[sz["effect"] == eff]
        parts = []
        for lab, _, _ in SIZE_CONDITIONS:
            r = q[q["condition"] == lab]
            n_tie = int((r["verdict"] == "tie").sum())
            parts.append(f"{lab}: {r['difference'].mean():+.3f} on average "
                         f"({r['difference'].min():+.3f} to "
                         f"{r['difference'].max():+.3f}; {n_tie} of "
                         f"{len(r)} cells ties)")
        d0 = q[q["condition"] == "all defaults"]
        gain = d0["difference"].mean()
        n_win = int((d0["difference"] > TIE).sum())
        fa = fat_avg[fat_avg["effect"] == eff].set_index("size")
        if gain > TIE and n_win > len(d0) / 2:
            verdict = (f"a meaningful gain (on average more than {TIE}, and "
                       f"beyond a tie in {n_win} of {len(d0)} cells)")
        elif gain > TIE:
            verdict = (f"at most a borderline gain: on average just above "
                       f"the tie limit of {TIE}, but beyond a tie in only "
                       f"{n_win} of {len(d0)} cells")
        else:
            verdict = (f"no meaningful gain: on average a tie (within "
                       f"{TIE}), beyond a tie in {n_win} of {len(d0)} cells")
        out.append(
            f"{eff.capitalize()}: going from {a} to {b} trials per block "
            f"({EXTRA_TIME_TEXT}) changes the paired t-test's power by "
            + "; ".join(parts) + f". Fatigue itself costs "
            f"{fa.loc[a, 'fatigue_cost']:.3f} (SE "
            f"{fa.loc[a, 'fatigue_cost_se']:.3f}) at {a} and "
            f"{fa.loc[b, 'fatigue_cost']:.3f} (SE "
            f"{fa.loc[b, 'fatigue_cost_se']:.3f}) at {b} (power returned "
            f"when it is switched off; average of 6 cells). Once fatigue is "
            f"counted, the extra {b - a} trials per block buy {verdict}. "
            f"[ASSUMPTIONS: the effect, the nuisances and the fatigue model "
            f"are planning values.]")
    return out


def write_report(out, ck, rank, addv, cells, crn, mf, ex, s7, started,
                 minutes, batch_diff, sz, fat, fat_avg, n_workers, extra=()):
    L = []
    w = L.append
    plain, supp, same = summary_sentences(rank, addv, batch_diff)
    sizes_txt = size_sentences(sz, fat_avg)
    w("WHICH NUISANCES COST THE PRIMARY TEST POWER, AND 150 AGAINST 180  "
      "(nuisance_power_v5c.py, v5c script 5)"
      + ("  [REGRESSION ON v5b_180]" if MODE == "regression" else ""))
    w("=" * 78)
    w(f"run       {started}  ({minutes:.1f} min, {n_workers} worker "
      f"process(es), one thread each, {os.cpu_count()} cores)")
    w(f"output    {out}")
    w(f"model     {s7.PERSON_MODEL} (registered; side + N_corpus + N_orig "
      "per block, Firth, no intercept)")
    w(f"script 7  {PAIRED_RUN_DIR}")
    w(f"package   {', '.join(s7.PACKAGES[s] for s in BLOCK_SIZES)}")
    w(SCRIPT_NAMES_NOTE)
    w(f"checks    {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL "
      "(section 7)")
    w("")
    w(wrap("PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES. The "
           "nuisances and their default sizes are planning assumptions from "
           "script 7; this ranks the assumed nuisances at their assumed "
           "sizes, not anything measured in participants.", indent=""))
    w("")
    w("1. SUMMARY")
    w("-" * 78)
    w(wrap(plain))
    w("")
    for s in sizes_txt:
        w(wrap(s))
        w("")
    w("  Suggested sentence for the supplement:")
    w(wrap('"' + supp + '"', indent="    "))
    w("")
    w("2. WHAT WAS DONE")
    w("-" * 78)
    w(wrap("Paired t-test (the registered primary test), power only, fixed "
           "weights, planning and half shift, block size(s) "
           f"{', '.join(map(str, BLOCK_SIZES))}, N = "
           "10, 15, 20, both features; pair-specific preferences off; the "
           f"registered {s7.PERSON_MODEL} model. Nine conditions: all "
           "nuisances at script 7's defaults; each of the seven switched off "
           "on its own (to script 7's 'off' value); no nuisances. Power "
           "returned by a nuisance = power with it off minus power with all "
           f"defaults. Ranked by the unweighted average over the "
           f"{n_cells()} cells. "
           f"Script 7's person model, pools of {s7.POOL_PEOPLE:,} people, "
           f"{s7.N_STUDIES:,} studies per cell, SE from {s7.N_BATCHES} "
           "sub-pools."))
    w(wrap("v5's 'edited looks odd' and 'block order' no longer exist in "
           "v5b: the two strategies ('more typical', 'less edited', a "
           "quarter each at k = 0.5 per unit of N) replace the first, and "
           "fatigue (lapse +0.03 and preferences x0.939 per 100 main "
           "trials, anchored at block 1's mean) replaces the second."))
    w(wrap("Common random numbers: within an effect and block size, all "
           "nine conditions use the same random draws and the same study "
           "draws, so each recovery is a paired difference and its SE is "
           "that of the difference. 'Scene dependence off' is SD x "
           f"{SCENE_OFF_MULT:g} rather than exactly 0, because script 7 "
           "draws the scene deviations only when they are on and a 0 would "
           "shift every later draw; at x 1e-9 the deviations are numerically "
           "nothing. Switching fatigue off changes the lapse rate and so "
           "catch failures and who is excluded: in the fatigue-off and "
           "no-nuisance conditions some studies sample different people "
           "(section 6), which makes those two paired differences somewhat "
           "noisier."))
    w("")
    w(f"3. RANKING (power returned when switched off; average of {n_cells()} "
      "cells, with SE)")
    w("-" * 78)
    for eff in EFFECTS_RUN:
        q = rank[rank["effect"] == eff]
        w(f"\n  {eff}")
        for r in q.itertuples():
            note = {"costs": "",
                    "nothing": " (costs almost nothing)",
                    "raises": " (raises power when present)"}[
                classify(r.recovery)]
            w(f"    {r.rank}. {r.label:62s} {r.recovery:+.4f} "
              f"(SE {r.recovery_se:.4f}){note}")
    w("")
    w(wrap(order_comparison(rank, batch_diff)[1]))
    w("")
    for eff in EFFECTS_RUN:
        w(f"  At N = 20 (per block size and feature, {eff}):")
        q = cells[(cells["effect"] == eff) & (cells["N"] == 20)]
        cols = (["size", "feature"]
                + [f"recovery: {k}" for k, _, _ in NUISANCES]
                + ["total drop"])
        w("  " + q[cols].rename(columns=lambda c: c.replace("recovery: ",
                                                            ""))
          .to_string(index=False, float_format=lambda x: f"{x:+.3f}")
          .replace("\n", "\n  "))
        w("")
    w("4. DO THE SEVEN ADD UP?")
    w("-" * 78)
    w(wrap("Total drop = power with no nuisances minus power with all "
           "defaults. If the nuisances acted independently, the seven "
           "recoveries would sum to it. 'Interacting' if the sum differs "
           f"from the total by more than {INTERACT_SE:g} SE and by more "
           f"than {INTERACT_MIN}."))
    for r in addv.itertuples():
        w(f"  {r.effect:11s} total drop {r.total_drop:.4f} (SE "
          f"{r.total_drop_se:.4f}); sum of the seven {r.sum_of_seven:.4f} "
          f"(SE {r.sum_se:.4f});")
        w(f"  {'':11s} sum minus total {r.sum_minus_total:+.4f} (SE "
          f"{r.sum_minus_total_se:.4f}); ratio {r.ratio:.2f} -> "
          f"{'INTERACTING' if r.interacting else 'approximately additive'}")
    w(wrap("A ratio above 1 means switching nuisances off one at a time "
           "returns more power in total than switching them all off "
           "together (they overlap); below 1, that they compound (each costs "
           "more in the presence of the others)."))
    w("")
    w("5. 150 AGAINST 180 TRIALS PER BLOCK, WITH FATIGUE COUNTED")
    w("-" * 78)
    w(wrap("Power of the paired t-test at each size and the difference (180 "
           "minus 150), with its SE (independent pools: sqrt(SE1^2 + "
           f"SE2^2)); differences of {TIE} or less are ties (script 2's "
           "rule). The four conditions are script 7's scenarios; their rows "
           "are taken from script 7's adjusted run, after rerunning them "
           "with script 7's own streams and checking they match exactly "
           f"(section 7). The extra 30 trials per block take "
           f"{EXTRA_TIME_TEXT}. [ASSUMPTIONS]"))
    if len(BLOCK_SIZES) < 2:                    # v5c: one package
        w(f"  NOT APPLICABLE in v5c: there is only the {BLOCK_SIZES[0]}-"
          "trial package, so there is")
        w("  no 150 to compare with (size_comparison.csv is empty, "
          "figure_sizes.png is not made).")
        w(f"  Fatigue's own cost at {BLOCK_SIZES[0]} trials is below.")
    else:
        a, b = BLOCK_SIZES
        for eff in EFFECTS_RUN:
            q = sz[sz["effect"] == eff]
            w(f"\n  {eff}")
            w("  " + q[["condition", "N", "feature", f"power_{a}",
                        f"power_{b}", "difference", "difference_se",
                        "verdict"]].to_string(
                index=False, float_format=lambda x: f"{x:.3f}").replace(
                "\n", "\n  "))
    w("")
    w("  Fatigue's own power cost at each size (matched family: power "
      "returned when fatigue")
    w("  is switched off; average of 6 cells, then per cell):")
    w("  " + fat_avg.to_string(index=False, float_format=lambda x:
                               f"{x:+.4f}").replace("\n", "\n  "))
    w("  " + fat.to_string(index=False, float_format=lambda x:
                           f"{x:+.4f}").replace("\n", "\n  "))
    w("")
    w("6. COMMON DRAWS: HOW CLOSE THE MATCHING IS (exclusions)")
    w("-" * 78)
    w(wrap("Share of pool members whose exclusion (3+ catch failures) "
           "differs from all defaults, and share of studies that sample "
           "exactly the same people. Only conditions that change the lapse "
           "rate (fatigue off, no nuisances) can differ."))
    pc = crn[crn["N"].isna()].groupby("condition")["exclusion_differs"] \
        .mean()
    sp = crn[crn["N"].notna()].groupby("condition")["studies_same_people"] \
        .mean()
    w("  condition        people excluded differently   studies with the "
      "same people")
    for cond in pc.index:
        w(f"  {cond:16s} {pc[cond]:26.2%} {sp.get(cond, np.nan):30.1%}")
    w("")
    w("7. CHECKS")
    w("-" * 78)
    for r in ck.rows:
        w(f"  {r['status']}  [{r['section']}] {r['check']}")
        if r["detail"]:
            w(f"          {r['detail']}")
    w("")
    w("  Matched family's anchors against script 7 (paired t-test):")
    w("  " + mf[["effect", "anchor", "size", "N", "feature", "this_script",
                 "script7", "z"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}").replace("\n",
                                                                "\n  "))
    w("")
    w("8. RUN TIME")
    w("-" * 78)
    w(f"  This run: {minutes:.1f} min with {n_workers} worker process(es) "
      f"({len(BLOCK_SIZES) * len(EFFECTS_RUN) * 9} matched-family pools and "
      f"{len(BLOCK_SIZES) * len(EFFECTS_RUN) * (len(SIZE_CONDITIONS) + 1)} "
      "of script 7's pools rerun).")
    w("")
    for line in extra:
        w(line)
    w("12. FILES")
    w("-" * 78)
    for fn in sorted(set(os.listdir(out)) | {"report.txt", "run_meta.json",
                                             "summary.txt"}):
        w(f"  {fn}")
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    return [plain] + sizes_txt + ["Suggested sentence for the supplement: "
                                  + supp]


if __name__ == "__main__":
    main()
