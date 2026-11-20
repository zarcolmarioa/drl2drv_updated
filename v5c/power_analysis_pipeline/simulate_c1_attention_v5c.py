"""
simulate_c1_attention_v5c.py  --  v5c power analysis, script 4: C1 when a
                                  shared preference meets unequal attention
                                  (a copy of simulate_c1_attention_v5b.py,
                                  v5b script 8b, changed only where listed
                                  below)

WHAT CHANGED FROM simulate_c1_attention_v5b.py (v5b -> v5c)
    Only the inputs, the names and the comparison section; the scenarios,
    the check and its other versions, the numbers of studies and the seeds
    (script 8's streams) are v5b's.
      * INPUTS: script 3 (simulate_c1_v5c.py) is imported as "script 8"
        (and through it script 2, simulate_power_paired_v5c_adj.py), with
        their settings pointing to v5c (v5c_180); C1_RUN_DIR is script 3's
        run. Comparisons of 150 with 180 are "not applicable".
      * REGRESSION: "python simulate_c1_attention_v5c.py --regression" runs
        this same code on v5b_180 (scripts 3 and 2 in their regression mode,
        script 3's regression run) and requires every 180-trial row of
        v5b's c1_attention_v5b run (..._s8e477098, c1_attention_shared.csv)
        to be reproduced: rates identical, other values within 1e-12 (the
        "slice" is every 180-trial cell). A normal run first looks for a
        passing regression of this very script file and runs it if there is
        none; a FAIL stops before the v5c run.
      * A COMPARISON section (report and comparison.csv): v5c next to v5b's
        registered results, changes larger than COMPARE_FLAG flagged.
      * A PREFLIGHT lists every input path; N_WORKERS = None uses all CPUs.
      * Output under data/power_v5c/c1_attention_v5c/, v5b's folder naming.
        Script numbers in the text are v5b's; the report says so.

---- simulate_c1_attention_v5b.py's description follows (v5b -> v5c:
     simulate_c1_v5c.py for script 8, v5c_180 for the v5b packages) ----

simulate_c1_attention_v5b.py  --  v5b power analysis, script 8b: C1 when a
                                  shared preference meets unequal attention

WHAT THIS ANSWERS
    The attention check exists for one situation, which script 8 did not
    simulate: both groups share a one-direction preference (here everyone
    leans towards the predicted image, +0.25/+0.35 or +0.5/+0.7 per SD on
    saturation/contrast, in every block) and the comparison group answers
    at random more often (mean lapse 0.15 or 0.2; report group 0.1). The
    comparison group's weights then shrink towards zero, so the shared
    preference makes the report group look higher: a C1 claim with no
    effect. This script runs that situation with no effect and with the
    planning effect, and reports, per scenario, size, report N (10-20) and
    comparison N (20, 30):

      * C1's false-claim rate (no effect) and power (planning), before and
        after the attention check, per feature and for either feature;
      * how often the check fires, and each of its parts (catch failures,
        responses under 300 ms; Holm'd as in the check, and each on its own
        at 0.05);
      * FOR INFORMATION ONLY, two other versions of the check on the same
        studies:
          - catch failures alone: one one-sided Mann-Whitney on catch
            failures at 0.05 (no response-time part, no Holm); C1 counts
            only if it does not fire;
          - the check as a sensitivity analysis: C1 is reported without the
            gate (the "before" rate), with the gate's verdict beside it
            (the share of claims the gate would flag).
    The registered check is unchanged (v4's _c1_gate: both parts, Holm).

HOW
    Script 8 (simulate_c1_v5b.py) is imported and not modified; its pools,
    streams, recruitment, C1 test and gate are used as they are. Each
    scenario's report and comparison pools come from script 8's named
    streams (scenario name; the comparison group's nuisances and lapse), so
    script 8's own scenarios are reproduced exactly here: the rows of
    script 8 that this script reruns for context (no effect and planning
    with equal attention, the shared-preference rows, the attention rows)
    must equal script 8's c1.csv exactly (checked). The new scenarios add
    the shared preference to the attention rows.

PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES
    The shared preference, the lapse rates, the response-time model, the
    responder share and shift, and the strategy quarters are assumptions
    for planning.

REUSED (imported, not copied; SHA-256 checked)
    simulate_c1_v5b.py (script 8)   build_scenarios, strategy_nuis,
        comparison_slots, comparison_key, _pools (pools, fast responses,
        catch counts), _worker_init, its settings (streams, N, studies)
    simulate_power_paired_v5b_adj.py (script 7, through script 8): the
        person model (adjusted), loaders with their package checks,
        draw_studies, rate_se
    simulate_c1_post_exclusion_v4.py: _c1_test, _c1_gate, mw_p_batch,
        GATE_ALPHA

INPUTS (relative to the project folder; run from there)
    the inputs of script 8, and C1_RUN_DIR (script 8's run: c1.csv,
    checks.csv, run_meta.json)

OUTPUTS (one new folder per run under OUTPUT_ROOT; never overwritten)
    report.txt, summary.txt, checks.csv, settings.json, run_meta.json
    c1_attention_shared.csv   every cell: all rates and their SE
    c1_attention_shared.png

REQUIRES  Python 3.9+, numpy, pandas, scipy; matplotlib for the figure
RUN       python simulate_c1_attention_v5c.py   (v5c_180; runs the
                                         regression first if needed)
          python simulate_c1_attention_v5c.py --regression  (v5b_180 only)
          (each about 3 minutes with 2 workers)
"""

# One thread per process (BLAS and OpenMP), set before numpy is imported.
import os
import sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = None      # None = the current folder (run from there)
# MODE
#   "v5c" (default): the v5c package. "regression": the same code on
#   v5b_180 with v5b's inputs (python simulate_c1_attention_v5c.py
#   --regression). Scripts 3 and 2 are imported in the same mode; worker
#   processes inherit it through the environment.
MODE_ENV = "C1ATT_V5C_MODE"
MODE = (os.environ.get(MODE_ENV)
        or ("regression" if "--regression" in sys.argv[1:] else "v5c"))
os.environ["C1_V5C_MODE"] = MODE
os.environ["PAIRED_V5C_MODE"] = MODE
# BLOCK_SIZES: v5c has one package (180 trials); the runs use script 3's
#   BLOCK_SIZES (checked to be the same).
BLOCK_SIZES = [180]
SCRIPT8 = "simulate_c1_v5c.py"            # v5c script 3
EXPECTED_SHA256 = {
    SCRIPT8:
        "28c6a7dd5f833a728d9a1209d116e531f0189a0e57156dd439e5f84f82a88e9c",
}
# Script 3's run: the rows rerun here for context must equal it exactly.
C1_RUN_DIR = ("data/power_v5c/c1_v5c/sizes180_report10-20_comparison20-30_model-adjusted_pool50000_studies10000_seed42__sb3711eee")
OUTPUT_ROOT = "data/power_v5c/c1_attention_v5c"
MAKE_FIGURES = True
FIG_DPI = 150

N_WORKERS = None
CHECK_ONE_WORKER = True

# SHARED / LAPSE_REPORT / LAPSE_COMPARISON / EFFECTS_RUN
#   Everyone in both groups, every block, towards the predicted image
#   (script 7's "dull" nuisance), on top of script 8's main-row nuisances
#   (the strategy quarters). Comparison group's mean lapse 0.1 (equal
#   attention, for reference), 0.15 and 0.2; report group 0.1.
SHARED = [(0.25, 0.35), (0.5, 0.7)]
LAPSE_REPORT = 0.1
LAPSE_COMPARISON = [0.1, 0.15, 0.2]
EFFECTS_RUN = ("no effect", "planning")

# CONTEXT_ROWS: script 8's own scenarios rerun here (reproduction check and
# context in the tables).
CONTEXT_ROWS = [
    "no effect", "planning",
    "no effect, both groups share +0.25/+0.35",
    "no effect, both groups share +0.5/+0.7",
    "no effect, lapse report 0.1 / comparison 0.15",
    "no effect, lapse report 0.1 / comparison 0.2",
    "planning, lapse report 0.1 / comparison 0.15",
    "planning, lapse report 0.1 / comparison 0.2",
]
# The other versions of the check (information only)
CATCH_ONLY_ALPHA = 0.05

TABLE_REPORT_N = [10, 15, 20]
RATE_TOL = 0.0           # reproduced rates must be identical

# --- v5b's registered results (comparison and regression reference) ---
REF_RUN = ("data/power_v5b/c1_attention_v5b/shared-preference_unequal-"
           "attention__s8e477098")
COMPARE_FLAG = 0.05            # changes larger than this are flagged

# --- Regression (MODE "regression"): as in script 3 ---
REGRESSION_ROOT = "data/power_v5c/c1_attention_v5c/regression_v5b"
REGRESSION_FILES = [
    ("c1_attention_shared.csv", ["size", "scenario", "report_n",
                                 "comparison_n"]),
]
REGRESSION_EXACT_COLS = (
    "raw_saturation", "raw_contrast", "raw_either", "gated_saturation",
    "gated_contrast", "gated_either", "gate_fires", "gate_fires_catch",
    "gate_fires_rt", "secondary_saturation", "secondary_contrast",
    "secondary_either", "catch_alone_fires", "rt_alone_fires",
    "catchonly_saturation", "catchonly_contrast", "catchonly_either",
    "claim_flagged_by_gate")
REGRESSION_TOL = 1e-12
REGRESSION_RUN_FIRST = True

if MODE == "regression":
    C1_RUN_DIR = ("data/power_v5c/c1_v5c/regression_v5b/sizes180_report10-20_comparison20-30_model-adjusted_pool50000_studies10000_seed42__sf93ddddc")
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

T0 = time.time()
QUIET = False


def log(msg):
    if not QUIET:
        print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


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


SETTING_NAMES = ["PROJECT_ROOT", "SCRIPT8", "EXPECTED_SHA256", "C1_RUN_DIR",
                 "OUTPUT_ROOT", "MAKE_FIGURES", "FIG_DPI", "N_WORKERS",
                 "CHECK_ONE_WORKER", "SHARED", "LAPSE_REPORT",
                 "LAPSE_COMPARISON", "EFFECTS_RUN", "CONTEXT_ROWS",
                 "CATCH_ONLY_ALPHA", "TABLE_REPORT_N", "RATE_TOL",
                 "MODE", "BLOCK_SIZES", "REF_RUN", "COMPARE_FLAG",
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


def settings_dict():
    g = globals()
    return {k: g[k] for k in SETTING_NAMES}


def resolve_output_dir():
    s = settings_dict()
    s.pop("N_WORKERS")
    s.pop("REGRESSION_RUN_FIRST")      # cannot change the result
    h = hashlib.sha256(json.dumps(jsonable(s), sort_keys=True).encode()
                       ).hexdigest()[:8]
    base = root_path(OUTPUT_ROOT, f"shared-preference_unequal-attention__s{h}")
    out, k = base, 1
    while os.path.exists(out):
        k += 1
        out = f"{base}_run{k}"
    os.makedirs(out)
    return out


def import_file(fname):
    path = root_path(fname)
    spec = importlib.util.spec_from_file_location(
        os.path.splitext(fname)[0] + "_imported_s8b", path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------

def scenarios(s8, s7):
    """Script 8's context rows (its own definitions), then the new ones."""
    S8 = {s["name"]: s for s in s8.build_scenarios(s7)}
    out = [dict(S8[n], new=False) for n in CONTEXT_ROWS]
    base = s8.strategy_nuis(s7.NUISANCE_OFF)
    for eff in EFFECTS_RUN:
        for sp in SHARED:
            for lc in LAPSE_COMPARISON:
                name = (f"{eff}, both groups share +{sp[0]:g}/+{sp[1]:g}, "
                        f"lapse report {LAPSE_REPORT:g} / comparison {lc:g}")
                if eff == "no effect" and lc == LAPSE_REPORT:
                    continue            # script 8's shared rows (context)
                nuis = {**base, "dull": sp}
                out.append({"name": name, "kind": "shared + attention",
                            "effect": s7.EFFECTS[eff], "nuis": nuis,
                            "nuis_c": nuis, "lapses": (LAPSE_REPORT, lc),
                            "new": True})
    names = [s["name"] for s in out]
    assert len(names) == len(set(names))
    return out


def describe(sc):
    """effect, shared preference and lapse pair of a scenario."""
    eff = "no effect" if sc["effect"]["share"] == 0 else "planning"
    d = sc["nuis"].get("dull", (0.0, 0.0))
    return {"effect": eff, "shared": f"+{d[0]:g}/+{d[1]:g}"
            if d[0] or d[1] else "none",
            "lapse_report": sc["lapses"][0],
            "lapse_comparison": sc["lapses"][1]}


# ------------------------------------------------------------
# One cell: script 8's C1 cell, with the other versions of the check
# ------------------------------------------------------------

def cell(rp, cp, nr, nc, rng):
    s7, c1 = _W["s8"]._W["s7"], _W["s8"]._W["c1"]
    s8 = _W["s8"]
    ir, ex_r, sb = s7.draw_studies(rp, nr, s8.N_STUDIES, rng)
    ic, ex_c, sb2 = s7.draw_studies(cp, nc, s8.N_STUDIES, rng)
    assert np.array_equal(sb, sb2)
    A = rp["est_D"][ir][:, :, :2]
    B = cp["est_O"][ic][:, :, :2]
    r_sat, r_con = c1._c1_test(A, B)
    cr, cc = rp["catch"][ir], cp["catch"][ic]
    fr, fc = rp["fast"][ir], cp["fast"][ic]
    fired, f_catch, f_rt = c1._c1_gate(cr, cc, fr, fc)
    s_sat, s_con = c1._c1_test(rp["est_O"][ir][:, :, :2], B)
    # the parts on their own (uncorrected) and the catch-only check
    p_catch, _ = c1.mw_p_batch(cc, cr, exact_max_n=0)
    p_rt, _ = c1.mw_p_batch(fc, fr, exact_max_n=0)
    lo, hi = np.minimum(p_catch, p_rt), np.maximum(p_catch, p_rt)
    holm = (lo <= c1.GATE_ALPHA / 2) | ((lo <= c1.GATE_ALPHA / 2)
                                        & (hi <= c1.GATE_ALPHA))
    same_gate = bool(np.array_equal(holm, fired))
    catch_alone = p_catch <= CATCH_ONLY_ALPHA
    rt_alone = p_rt <= c1.GATE_ALPHA
    either = r_sat | r_con
    live = ~fired
    out = {}
    for key, x in (
            # script 8's columns, computed as script 8 does
            ("raw_saturation", r_sat), ("raw_contrast", r_con),
            ("raw_either", either),
            ("gated_saturation", r_sat & live),
            ("gated_contrast", r_con & live),
            ("gated_either", either & live),
            ("gate_fires", fired), ("gate_fires_catch", f_catch),
            ("gate_fires_rt", f_rt),
            ("secondary_saturation", s_sat),
            ("secondary_contrast", s_con),
            ("secondary_either", s_sat | s_con),
            # the parts on their own
            ("catch_alone_fires", catch_alone),
            ("rt_alone_fires", rt_alone),
            # other version 1: catch failures alone
            ("catchonly_saturation", r_sat & ~catch_alone),
            ("catchonly_contrast", r_con & ~catch_alone),
            ("catchonly_either", either & ~catch_alone),
            # other version 2: the check as a sensitivity analysis (C1 =
            # the raw columns; the gate's verdict beside it)
            ("claim_flagged_by_gate", either & fired)):
        m, se = s7.rate_se(x, sb)
        out[key] = m
        out[key + "_se"] = se
    out["excluded_report"] = float(ex_r.mean())
    out["excluded_comparison"] = float(ex_c.mean())
    return out, same_gate


# ------------------------------------------------------------
# Tasks
# ------------------------------------------------------------

_W = {}


def _worker_init(state, quiet=True):
    global QUIET
    QUIET = quiet
    s8 = import_file(SCRIPT8)
    s8._worker_init(state, quiet=True)
    s7 = s8._W["s7"]
    _W.update(s8=s8, scen=scenarios(s8, s7))


def task(size, i):
    s8 = _W["s8"]
    cap = s8._W["cap"]
    sc = _W["scen"][i]
    # script 8's streams: the scenario's name, and the comparison group's
    # nuisances and lapse
    st = cap.Streams(s8.RANDOM_SEED, s8.STREAM_KEY, size, sc["name"])
    stc = cap.Streams(s8.RANDOM_SEED, s8.STREAM_KEY, size, "comparison",
                      s8.comparison_key(sc))
    rp, cp = s8._pools(sc, size, st, stc)
    rows, gate_ok = [], True
    for nc in s8.COMPARISON_N:
        for nr in s8.REPORT_N:
            res, ok = cell(rp, cp, nr, nc, st(f"studies_{nr}_{nc}"))
            gate_ok &= ok
            rows.append({"size": size, "scenario": sc["name"],
                         "kind": sc["kind"], "new": sc["new"],
                         "report_n": nr, "comparison_n": nc,
                         **describe(sc), **res})
    att = []
    for grp, pool in (("report", rp), ("comparison", cp)):
        keep = ~pool["excluded"]
        att.append({"size": size, "scenario": sc["name"], "group": grp,
                    "lapse_mean": float(((pool["lapse_D"] + pool["lapse_O"])
                                         / 2)[keep].mean()),
                    "catch_mean": float(pool["catch"][keep].mean()),
                    "fast_share": float(pool["fast"][keep].mean()),
                    "excluded_share": float(pool["excluded"].mean()),
                    "mean_weight_sat": float(pool["est_O"][keep, 0].mean()),
                    "mean_weight_con": float(pool["est_O"][keep, 1].mean())})
    return {"rows": rows, "att": att, "gate_ok": gate_ok}


def run_tasks(jobs, n_workers, state):
    if n_workers <= 1:
        _worker_init(state, quiet=False)
        return [task(*a) for a in jobs]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(n_workers, len(jobs)),
                             mp_context=ctx, initializer=_worker_init,
                             initargs=(state,)) as ex:
        futs = [ex.submit(task, *a) for a in jobs]
        out = []
        for k, f in enumerate(futs):
            out.append(f.result())
            if (k + 1) % 10 == 0 or k + 1 == len(futs):
                log(f"  {k + 1} of {len(futs)} tasks done")
        return out


# ------------------------------------------------------------
# Figure, tables, report
# ------------------------------------------------------------

def figure(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
    nr = max(TABLE_REPORT_N)
    q = df[(df["report_n"] == nr) & (df["comparison_n"] == 30)]
    sizes = sorted(q["size"].unique())
    fig, axes = plt.subplots(2, len(sizes), figsize=(5 * len(sizes), 7.5),
                             squeeze=False)
    for i, eff in enumerate(("no effect", "planning")):
        for j, size in enumerate(sizes):
            ax = axes[i, j]
            for c, sh in enumerate(["none"] + [f"+{a:g}/+{b:g}"
                                               for a, b in SHARED]):
                r = q[(q["size"] == size) & (q["effect"] == eff)
                      & (q["shared"] == sh)].sort_values("lapse_comparison")
                if r.empty:
                    continue
                ax.plot(r["lapse_comparison"], r["raw_either"], "-",
                        color=colours[c], marker="o", ms=3,
                        label=f"shared {sh}: before the check")
                ax.plot(r["lapse_comparison"], r["gated_either"], "--",
                        color=colours[c], marker="s", ms=3,
                        label=f"shared {sh}: after the check")
                ax.plot(r["lapse_comparison"], r["catchonly_either"], ":",
                        color=colours[c], marker="^", ms=3,
                        label=f"shared {sh}: catch-only check")
            ax.axhline(0.05, color="#888888", lw=0.8, ls=":")
            ax.set_title(f"{eff}, {size} trials per block (report N {nr}, "
                         "comparison 30)", fontsize=9)
            ax.set_xlabel("Comparison group's mean lapse (report group 0.1)")
            ax.set_ylabel("Share of studies with a C1 claim\n(either "
                          "feature)")
            ax.set_ylim(0, 1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color="#dddddd", linewidth=0.6)
    axes[0, -1].legend(fontsize=6, frameon=False)
    fig.suptitle("C1 when both groups share a preference and the comparison "
                 "group lapses more\n(shared preferences and lapse rates are "
                 "assumptions)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "c1_attention_shared.png"), dpi=FIG_DPI)
    plt.close(fig)


def fmt(df, nd=3):
    return "  " + df.to_string(index=False, float_format=lambda x:
                               f"{x:.{nd}f}").replace("\n", "\n  ")


def table(df, eff, cols):
    q = df[(df["effect"] == eff) & df["report_n"].isin(TABLE_REPORT_N)]
    q = q.sort_values(["size", "shared", "lapse_comparison", "comparison_n",
                       "report_n"])
    return q[["size", "shared", "lapse_comparison", "comparison_n",
              "report_n"] + cols]


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
    """Every input of this script (scripts 2 and 3's inputs are read through
    their loaders)."""
    reg = MODE == "regression"
    pk = ("data/frozen_trial_lists_v5b/v5b_180" if reg
          else "data/frozen_trial_lists_v5c/v5c_180")
    paths = [SCRIPT8, "simulate_power_paired_v5c_adj.py",
             "simulate_c1_post_exclusion_v4.py", "simulate_power_v4_n10_20.py",
             ("analyse_pair_capacity_v5b.py" if reg
              else "analyse_pair_capacity_v5c.py"),
             ("explore_block_length_v5b.py" if reg
              else "explore_block_length_v5c.py"), "thresholds_v5c.py",
             "data/stimulus_set_v5/7fd72262ba/stimulus_manifest.csv",
             "data/stimulus_set_v5/7fd72262ba/reference_stimulus_path.csv"]
    paths += package_paths(pk)
    paths += [f"{C1_RUN_DIR}/{f}" for f in ("c1.csv", "checks.csv",
                                            "run_meta.json")]
    paths += [f"{REF_RUN}/{fn}" for fn, _ in REGRESSION_FILES]
    return paths


ATT_MEASURES = ["raw_either", "gated_either", "gate_fires",
                "gate_fires_catch", "gate_fires_rt", "catchonly_either",
                "claim_flagged_by_gate", "gated_contrast", "gated_saturation"]


def compare_with_v5b(out, df):
    """The v5c results next to v5b's registered results (c1_attention_v5b,
    180 trials): every cell."""
    s = 180
    r = _read(f"{REF_RUN}/c1_attention_shared.csv")
    r = r[r["size"] == s]
    comp = long_compare(df, r, ["effect", "shared", "lapse_comparison",
                                "comparison_n", "report_n", "scenario"],
                        ATT_MEASURES, "attention check")
    comp.to_csv(os.path.join(out, "comparison.csv"), index=False)
    return comp


def comparison_lines(comp):
    L = ["14. COMPARISON WITH v5b'S REGISTERED RESULTS (c1_attention_v5b, "
         "180 trials per block)", "-" * 78,
         f"v5b: {REF_RUN.split('/')[-1]}, rows of 180 trials. v5c: this run. "
         "The same streams and",
         "seeds, so the differences come from the lists. Shares of studies "
         "at report N 10/15/20",
         f"(every N in comparison.csv); '*': a change larger than "
         f"{COMPARE_FLAG:g}. raw = C1 before the check,",
         "gated = after it (registered), gate_fires = the check fires; "
         "lapse_c = the comparison",
         "group's mean lapse (the report group's is "
         f"{LAPSE_REPORT:g}). Comparisons of 150 with 180: not applicable.",
         ""]
    q = comp.copy()
    q["lapse_c"] = q["lapse_comparison"]
    for eff, meas in (("no effect", ["raw_either", "gated_either",
                                     "gate_fires"]),
                      ("planning", ["gated_contrast", "gated_saturation",
                                    "gated_either", "gate_fires"])):
        t = q[(q["effect"] == eff) & q["measure"].isin(meas)]
        L.append(f"  {eff.upper()}: "
                 + ("false claims" if eff == "no effect" else "power"))
        L.append("  " + wide_compare(
            t, ["shared", "lapse_c", "comparison_n", "measure"], "report_n",
            TABLE_REPORT_N).to_string(index=False, float_format=lambda x:
                                      f"{x:.3f}").replace("\n", "\n  "))
        L.append("")
    allc = comp[comp["report_n"].isin(TABLE_REPORT_N)]
    L.append(f"  In all: {int((allc['flag'] != '').sum())} of {len(allc)} "
             f"compared values (report N = "
             f"{'/'.join(map(str, TABLE_REPORT_N))}) changed by more than "
             f"{COMPARE_FLAG:g}.")
    L.append("")
    return L


def summary_v5c(comp, reg):
    L = []
    if isinstance(reg, pd.DataFrame):
        ok = (reg["status"] == "PASS").all()
        L.append(f"* Regression on v5b_180: {'PASS' if ok else 'FAIL'} "
                 f"({int((reg['status'] == 'PASS').sum())} of {len(reg)} "
                 "files reproduce v5b's c1_attention_v5b run at 180 trials: "
                 f"rates identical, other values within {REGRESSION_TOL:g}).")
    elif reg is not None:
        t = reg["table"]
        L.append(f"* Regression on v5b_180: PASS ({len(t)} of {len(t)} files "
                 "reproduce v5b's c1_attention_v5b run at 180 trials; "
                 + reg["dir"] + ").")
    if comp is None:
        return L
    nr = max(TABLE_REPORT_N)
    q = comp[comp["report_n"] == nr]
    L.append(f"* v5c against v5b (in brackets), 180 trials, report N {nr}, "
             "comparison groups of 20 / 30:")
    for eff, meas, what in (("no effect", "raw_either",
                             "false claim before the check"),
                            ("no effect", "gated_either",
                             "false claim after the check"),
                            ("planning", "gated_either",
                             "power after the check"),
                            ("no effect", "gate_fires", "the check fires")):
        parts = []
        for (sh, lc), g in q[(q["effect"] == eff)
                             & (q["measure"] == meas)].groupby(
                ["shared", "lapse_comparison"], sort=False):
            g = g.sort_values("comparison_n")
            parts.append(f"shared {sh}, comparison lapse {lc:g}: "
                         + " / ".join(f"{r.v5c:.2f} ({r.v5b:.2f})"
                                      for r in g.itertuples()))
        L.append(f"    {eff}, {what}: " + "; ".join(parts) + ".")
    allc = comp[comp["report_n"].isin(TABLE_REPORT_N)]
    L.append(f"* {int((allc['flag'] != '').sum())} of {len(allc)} compared "
             f"values changed by more than {COMPARE_FLAG:g} (report section "
             "14, comparison.csv). 150 against 180 trials: not applicable.")
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
    p = root_path(SCRIPT8)
    ok = os.path.exists(p)
    ck.add("inputs", f"{SCRIPT8} exists", ok, p)
    if not ok:
        raise SystemExit(f"missing {p}")
    hashes[SCRIPT8] = sha256_of(p)
    ck.add("inputs", f"{SCRIPT8} is the version run for script 3 (SHA-256)",
           hashes[SCRIPT8] == EXPECTED_SHA256[SCRIPT8],
           hashes[SCRIPT8][:16] + "...")
    s8 = import_file(SCRIPT8)
    s8.QUIET = True
    # script 8's own input checks, through its loaders
    s7 = s8.load_checked(s8.SCRIPT7, ck, hashes)
    s7.QUIET = True
    ck.add("inputs", "scripts 3 and 2 (imported as scripts 8 and 7) run in "
           "this script's mode, on the matching package, with the same block "
           "size", s8.MODE == MODE and s7.MODE == MODE
           and list(s8.BLOCK_SIZES) == BLOCK_SIZES,
           f"modes {s8.MODE} / {s7.MODE}, package "
           + ", ".join(s7.PACKAGES[s] for s in s8.BLOCK_SIZES))
    s8.load_checked(s8.C1_V4_SCRIPT, ck, hashes)
    with open(os.path.join(root_path(s7.PACKAGES[s8.BLOCK_SIZES[0]]),
                           "FROZEN.json")) as f:
        frozen0 = json.load(f)
    stim = s7.load_stimulus_table(frozen0, ck, hashes)
    pk = {size: s7.load_package(size, ck, hashes, stim, s7.LISTS)
          for size in s8.BLOCK_SIZES}
    s7.load_module(s7.V4_SIM_SCRIPT, ck, hashes, frozen0["scripts_sha256"])
    s7.load_module(s7.CAPACITY_SCRIPT, ck, hashes, frozen0["scripts_sha256"])
    s7.load_module(s7.EXPLORE_SCRIPT, ck, hashes)
    s7.load_module(s7.THRESHOLDS_SCRIPT, ck, hashes)
    pkc = {}
    for size in s8.BLOCK_SIZES:
        pkc[size], ok, n = s8.comparison_slots(pk[size], size, s7)
        ck.add(f"package {size}", "assignment table: 64 slots for the "
               "comparison group", ok, f"{n} rows")
    out = resolve_output_dir()
    n_workers = N_WORKERS or max(1, os.cpu_count() or 1)
    log(f"output folder {out}")
    log(f"{n_workers} worker process(es), one thread each")

    # --- script 8's run ---
    d8 = root_path(C1_RUN_DIR)
    need = ["c1.csv", "checks.csv", "run_meta.json"]
    ok = all(os.path.exists(os.path.join(d8, f)) for f in need)
    ck.add("inputs", "script 8's run exists (" + ", ".join(need) + ")", ok,
           C1_RUN_DIR)
    if not ok:
        raise SystemExit(f"missing files in {d8}")
    for f in need:
        hashes[f"{C1_RUN_DIR}/{f}"] = sha256_of(os.path.join(d8, f))
    c8 = pd.read_csv(os.path.join(d8, "checks.csv"))
    ck.add("inputs", "script 8's run passed every check",
           (c8["status"] == "PASS").all(),
           f"{(c8['status'] == 'PASS').sum()} PASS, "
           f"{(c8['status'] != 'PASS').sum()} FAIL")
    with open(os.path.join(d8, "run_meta.json")) as f:
        meta8 = json.load(f)
    ck.add("inputs", "script 8's run used the adjusted model",
           meta8.get("person_model") == "adjusted",
           str(meta8.get("person_model")))
    # round_trip: the floats exactly as script 8 wrote them (pandas'
    # default parser can differ in the last bit)
    df8 = pd.read_csv(os.path.join(d8, "c1.csv"), float_precision="round_trip")

    # --- the tasks ---
    scen = scenarios(s8, s7)
    jobs = [(size, i) for size in s8.BLOCK_SIZES for i in range(len(scen))]
    state = {"pk": pk, "pkc": pkc, "TH": None}
    log(f"{len(jobs)} tasks ({len(scen)} scenarios x "
        f"{len(s8.BLOCK_SIZES)} sizes)")
    res = run_tasks(jobs, n_workers, state)
    if CHECK_ONE_WORKER and n_workers > 1:
        _worker_init(state, quiet=False)
        j0 = [k for k, (size, i) in enumerate(jobs) if scen[i]["new"]][0]
        log(f"rerunning task {jobs[j0]} in this process (one worker)")
        again = task(*jobs[j0])
        ck.add("workers", f"one worker gives the same result as {n_workers}",
               pd.DataFrame(again["rows"]).equals(
                   pd.DataFrame(res[j0]["rows"]))
               and again["att"] == res[j0]["att"],
               f"{jobs[j0][0]} trials, '{scen[jobs[j0][1]]['name']}'")
    elif CHECK_ONE_WORKER:
        ck.add("workers", "one worker gives the same result (run with one "
               "worker: nothing to compare)", True, "N_WORKERS = 1")

    df = pd.DataFrame([row for r in res for row in r["rows"]])
    att = pd.DataFrame([row for r in res for row in r["att"]])
    ck.add("check", "the gate recomputed from its two p-values (Holm) "
           "equals v4's _c1_gate in every study",
           all(r["gate_ok"] for r in res), f"{len(df)} cells")

    # --- reproduction of script 8 ---
    keys = ["size", "scenario", "report_n", "comparison_n"]
    cols = [c for c in df8.columns if c not in keys + ["kind",
                                                       "lapse_report",
                                                       "lapse_comparison"]]
    ctx = df[~df["new"]]
    m = ctx.merge(df8, on=keys, suffixes=("", "_8"))
    diff = max(float(np.max(np.abs(m[c] - m[c + "_8"]))) for c in cols)
    ck.add("reproduction", "script 8's scenarios rerun here equal its c1.csv "
           "(every column, every N)", len(m) == len(ctx) and len(m) > 0
           and diff <= RATE_TOL, f"{len(m)} of {len(ctx)} cells, "
           f"{len(cols)} columns; max |difference| {diff:.1e}")

    # --- checks on the results ---
    q = df[(df["effect"] == "no effect") & (df["shared"] == "none")
           & (df["lapse_comparison"] == LAPSE_REPORT)]
    ck.add("results", "no shared preference, equal attention: the catch-"
           "only check fires at about 0.05 (within 3 SE)",
           ((q["catch_alone_fires"] - 0.05) / q["catch_alone_fires_se"]
            <= 3).all(), f"largest {q['catch_alone_fires'].max():.4f}")
    q = df[df["lapse_comparison"] > LAPSE_REPORT]
    ck.add("results", "the gate fires at least as often when the comparison "
           "group lapses more (0.2 against 0.15, every cell)",
           all((g.sort_values("lapse_comparison")["gate_fires"].diff()
                .dropna() >= -0.005).all()
               for _, g in q.groupby(["size", "effect", "shared",
                                      "report_n", "comparison_n"])),
           "")

    # --- v5c: regression (MODE regression) or comparison with v5b ---
    df.to_csv(os.path.join(out, "c1_attention_shared.csv"), index=False)
    if MODE == "regression":
        reg = regression_compare(out, ck)
        extra = regression_lines(reg, None, "7.")
        comp = None
    else:
        reg = None
        extra = regression_lines(None, reg_info, "7.")
        comp = compare_with_v5b(out, df)
        extra += [x.replace("14. COMPARISON", "8. COMPARISON", 1)
                  for x in comparison_lines(comp)]

    # --- summary ---
    nr = max(TABLE_REPORT_N)
    S = []
    S.append(f"* Checks: {len(ck.rows) - ck.n_fail} of {len(ck.rows)} pass"
             + ("." if not ck.n_fail else f"; {ck.n_fail} FAIL."))
    S.append("* Situation: both groups share a preference towards the "
             "predicted image; the comparison group lapses more (report "
             f"group {LAPSE_REPORT:g}). Registered check = both parts, Holm. "
             "Rates at report N = "
             f"{nr}, comparison N 20 and 30, both sizes (ASSUMPTIONS).")
    for eff, what in (("no effect", "false claims"), ("planning", "power")):
        for sp in SHARED:
            sh = f"+{sp[0]:g}/+{sp[1]:g}"
            for lc in LAPSE_COMPARISON:
                r = df[(df["effect"] == eff) & (df["shared"] == sh)
                       & (df["lapse_comparison"] == lc)
                       & (df["report_n"] == nr)]
                if r.empty:
                    continue
                S.append(
                    f"* {eff}, shared {sh}, comparison lapse {lc:g} ({what}, "
                    f"either feature): before the check "
                    f"{r['raw_either'].min():.3f}-{r['raw_either'].max():.3f}"
                    f"; after {r['gated_either'].min():.3f}-"
                    f"{r['gated_either'].max():.3f}; catch-only check "
                    f"{r['catchonly_either'].min():.3f}-"
                    f"{r['catchonly_either'].max():.3f}; the check fires in "
                    f"{r['gate_fires'].min():.3f}-{r['gate_fires'].max():.3f}"
                    f" (catch part {r['gate_fires_catch'].min():.3f}-"
                    f"{r['gate_fires_catch'].max():.3f}, under-300-ms part "
                    f"{r['gate_fires_rt'].min():.3f}-"
                    f"{r['gate_fires_rt'].max():.3f}).")
    S.append("* As a sensitivity analysis (no gate), C1 is the 'before' "
             "rate; the gate's verdict beside it flags the share "
             "'claim_flagged_by_gate' of studies (report section 5).")
    S += summary_v5c(comp, reg if MODE == "regression" else reg_info)

    # --- write ---
    df.to_csv(os.path.join(out, "c1_attention_shared.csv"), index=False)
    if MAKE_FIGURES:
        try:
            figure(df, out)
        except ImportError:
            log("matplotlib not installed: no figure")
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(jsonable(settings_dict()), f, indent=2)
    pd.DataFrame(ck.rows).to_csv(os.path.join(out, "checks.csv"),
                                 index=False)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("C1 WITH A SHARED PREFERENCE AND UNEQUAL ATTENTION "
                "(simulate_c1_attention_v5c.py"
                + (", REGRESSION ON v5b_180" if MODE == "regression" else "")
                + "): IN PLAIN WORDS\n\n"
                + "\n".join(S) + "\n")
    import scipy
    minutes = (time.time() - T0) / 60
    L = []
    w = L.append
    w("C1 WITH A SHARED PREFERENCE AND UNEQUAL ATTENTION  "
      "(simulate_c1_attention_v5c.py, v5c script 4; copy of v5b script 8b)"
      + ("  [REGRESSION ON v5b_180]" if MODE == "regression" else ""))
    w("=" * 78)
    w(f"run       {started}  ({minutes:.1f} min, {n_workers} worker "
      "process(es), one thread each)")
    w(f"output    {out}")
    w(f"python    {platform.python_version()}, numpy {np.__version__}, "
      f"scipy {scipy.__version__}, pandas {pd.__version__}")
    w(f"script 8  {C1_RUN_DIR}")
    w(f"package   {', '.join(s7.PACKAGES[s] for s in s8.BLOCK_SIZES)}")
    w(SCRIPT_NAMES_NOTE)
    w("")
    w("0. IN PLAIN WORDS")
    w("-" * 78)
    L += S
    w("")
    w("PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES: the shared "
      "preference, the lapse")
    w("rates, the response-time model and the effect are chosen for "
      "planning.")
    w("")
    w("1. WHAT WAS DONE")
    w("-" * 78)
    w("Script 8's pools, recruitment, C1 test and registered attention check "
      "(imported).")
    w("Every person in both groups has a preference towards the predicted "
      "image in every")
    w("block (shared, +0.25/+0.35 or +0.5/+0.7 per SD); script 8's strategy "
      "quarters as in its")
    w(f"main rows; report group mean lapse {LAPSE_REPORT:g}, comparison "
      f"group {', '.join(f'{x:g}' for x in LAPSE_COMPARISON)}.")
    w("Columns: raw_* = C1 before the check (per feature after Holm; either "
      "= at least one);")
    w("gated_* = after the registered check (C1 counts only if the check "
      "does not fire);")
    w("gate_fires (any part), gate_fires_catch / gate_fires_rt = the parts "
      "as Holm'd in the")
    w("check; catch_alone_fires / rt_alone_fires = each part on its own at "
      "0.05;")
    w("catchonly_* = C1 under the catch-only check (information only); "
      "claim_flagged_by_gate")
    w("= share of studies with a C1 claim that the gate would flag (the "
      "check as a")
    w("sensitivity analysis; C1 is then the raw rate). shared 'none' rows "
      "are script 8's.")
    w("")
    w("2. CHECKS")
    w("-" * 78)
    for r in ck.rows:
        w(f"  {r['status']}  [{r['section']}] {r['check']}")
        if r["detail"]:
            w(f"          {r['detail']}")
    w(f"\n  {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL")
    w("")
    w("3. NO EFFECT: FALSE CLAIMS BEFORE AND AFTER THE CHECK")
    w("-" * 78)
    w(fmt(table(df, "no effect", ["raw_contrast", "raw_saturation",
                                  "raw_either", "gated_either",
                                  "catchonly_either",
                                  "claim_flagged_by_gate"])))
    w("")
    w("4. PLANNING: POWER BEFORE AND AFTER THE CHECK")
    w("-" * 78)
    w(fmt(table(df, "planning", ["raw_contrast", "raw_saturation",
                                 "raw_either", "gated_contrast",
                                 "gated_saturation", "gated_either",
                                 "catchonly_either",
                                 "claim_flagged_by_gate"])))
    w("")
    w("5. HOW OFTEN THE CHECK AND ITS PARTS FIRE")
    w("-" * 78)
    for eff in EFFECTS_RUN:
        w(f"\n  {eff}")
        w(fmt(table(df, eff, ["gate_fires", "gate_fires_catch",
                              "gate_fires_rt", "catch_alone_fires",
                              "rt_alone_fires"])))
    w("")
    w("6. POOL MEANS PER GROUP (lapse, catch failures of 10, share of "
      "responses under 300 ms,")
    w("   exclusions, mean ordinary-block weights: the comparison group's "
      "shrink with its lapses)")
    w("-" * 78)
    w(fmt(att))
    w("")
    for line in extra:
        w(line)
    w("9. FILES")
    w("-" * 78)
    for fn in sorted(set(os.listdir(out)) | {"report.txt",
                                             "run_meta.json"}):
        w(f"  {fn}")
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out, "run_meta.json"), "w") as f:
        json.dump(jsonable({"started": started, "minutes": minutes,
                            "workers": n_workers,
                            "python": platform.python_version(),
                            "numpy": np.__version__,
                            "scipy": scipy.__version__,
                            "pandas": pd.__version__, "output_dir": out,
                            "platform": platform.platform(),
                            "cpu_count": os.cpu_count(),
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
