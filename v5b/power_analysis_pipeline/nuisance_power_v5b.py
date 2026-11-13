"""
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
RUN       python nuisance_power_v5b.py   (about 1-2 minutes with 31 workers;
                                          about 8 minutes with 2)
"""

# One thread per process (BLAS and OpenMP), set before numpy is imported,
# so the result is the same for any N_WORKERS.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = None          # None = the current folder (run from there)
SCRIPT7 = "simulate_power_paired_v5b_adj.py"
EXPECTED_SHA256 = {
    SCRIPT7:
        "e7c7aa79c033e711d6f13c2aaf8dc75e66a558a152767c41e9696fc20afd1fd7",
}
PAIRED_RUN_DIR = ("data/power_v5b/paired_adj_v5b/sizes150-180_N10-20"
                  "_model-adjusted_pool50000_studies10000_mixed0_seed42"
                  "__s7c42ae4b")
OUTPUT_ROOT = "data/power_v5b/nuisance_v5b"
FIG_DPI = 200

N_WORKERS = None
CHECK_ONE_WORKER = True

BLOCK_SIZES = [150, 180]
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
STREAM_KEY = "nuisance_v5b"

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
                 "STREAM_KEY"]


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
            if s["effect"] == eff and s["lists"] == "v5b"}[name]["nuis"]
    st = cap.Streams(s7.RANDOM_SEED, s7.STREAM_KEY, "pool", "v5b", size,
                     name, eff)
    pool = s7.simulate_pool(_W["pk"][size], s7.EFFECTS[eff], nuis,
                            st("people"), _W["fit"])
    s7.person_thresholds(pool, size, _W["TH"])
    rows, _, _, _ = s7.scenario_rows(
        pool, {"name": name, "effect": eff, "role": role, "lists": "v5b"},
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
        ax.set_xlabel("Power returned when switched off\n(average of 12 "
                      "cells; bars: 2 SE)", fontsize=8)
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

def main():
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    ck = Checks()
    hashes = {}
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
    n_workers = N_WORKERS or max(1, (os.cpu_count() or 2) - 1)
    # script 7's inputs, through its loaders and checks
    with open(os.path.join(root_path(s7.PACKAGES[BLOCK_SIZES[0]]),
                           "FROZEN.json")) as f:
        frozen0 = json.load(f)
    stim = s7.load_stimulus_table(frozen0, ck, hashes)
    pk = {size: s7.load_package(size, ck, hashes, stim, "v5b")
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
    same = [k for k in meta7["input_sha256"]
            if k.startswith(tuple(s7.PACKAGES.values())) and k in hashes]
    ck.add("inputs", "script 7's run read the same frozen package files",
           len(same) > 0 and all(meta7["input_sha256"][k] == hashes[k]
                                 for k in same), f"{len(same)} files")
    # round_trip: the floats exactly as script 7 wrote them
    pw7 = pd.read_csv(os.path.join(d7, "paired_power.csv"),
                      float_precision="round_trip")
    pw7 = pw7[pw7["lists"] == "v5b"]
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
    for eff in EFFECTS_RUN:
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
    pd.DataFrame(ck.rows).to_csv(os.path.join(out, "checks.csv"),
                                 index=False)
    minutes = (time.time() - T0) / 60
    summary = write_report(out, ck, rank, addv, cells, crn, mf, ex, s7,
                           started, minutes, batch_diff, sz, fat, fat_avg,
                           n_workers)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("WHICH NUISANCES COST THE PRIMARY TEST POWER, AND 150 "
                "AGAINST 180 (nuisance_power_v5b.py): IN PLAIN WORDS\n\n"
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
        f"returned when each is switched off on its own; average over both "
        f"block sizes, N = 10, 15, 20 and both features). "
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
        f"scenario (average over block sizes, sample sizes of 10 to 20 and "
        f"both features)"
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
                 minutes, batch_diff, sz, fat, fat_avg, n_workers):
    L = []
    w = L.append
    plain, supp, same = summary_sentences(rank, addv, batch_diff)
    sizes_txt = size_sentences(sz, fat_avg)
    w("WHICH NUISANCES COST THE PRIMARY TEST POWER, AND 150 AGAINST 180  "
      "(nuisance_power_v5b.py)")
    w("=" * 78)
    w(f"run       {started}  ({minutes:.1f} min, {n_workers} worker "
      f"process(es), one thread each, {os.cpu_count()} cores)")
    w(f"output    {out}")
    w(f"model     {s7.PERSON_MODEL} (registered; side + N_corpus + N_orig "
      "per block, Firth, no intercept)")
    w(f"script 7  {PAIRED_RUN_DIR}")
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
           "weights, planning and half shift, block sizes 150 and 180, N = "
           "10, 15, 20, both features; pair-specific preferences off; the "
           f"registered {s7.PERSON_MODEL} model. Nine conditions: all "
           "nuisances at script 7's defaults; each of the seven switched off "
           "on its own (to script 7's 'off' value); no nuisances. Power "
           "returned by a nuisance = power with it off minus power with all "
           "defaults. Ranked by the unweighted average over the 12 cells. "
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
    w("3. RANKING (power returned when switched off; average of 12 cells, "
      "with SE)")
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
    a, b = BLOCK_SIZES
    for eff in EFFECTS_RUN:
        q = sz[sz["effect"] == eff]
        w(f"\n  {eff}")
        w("  " + q[["condition", "N", "feature", f"power_{a}", f"power_{b}",
                    "difference", "difference_se", "verdict"]].to_string(
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
    w("9. FILES")
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
