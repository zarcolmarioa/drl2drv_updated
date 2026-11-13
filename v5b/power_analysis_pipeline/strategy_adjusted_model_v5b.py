"""
strategy_adjusted_model_v5b.py  --  v5b power analysis, script 7b: a
                                    strategy-adjusted per-person model

WHAT THIS ANSWERS
    Script 7 found that the episode-dependent strategy (everyone picks the
    less typical / more edited image in the derealization block and the
    opposite in the ordinary block) still gives false positives on the
    v5b lists (about 0.09-0.18 at k = 0.25-0.5, N = 20), because a little
    of N leaks into the predictors. This script tries a per-person model
    that adjusts for it, as a VARIANT beside the registered side model:

        adjusted model  = the side model (saturation, contrast, side term)
                          + two nuisance terms per block, N_corpus and
                          N_orig (Firth logistic regression, no intercept)

    The N terms' estimates are nuisance terms and are NEVER results. The
    researcher decides the registered model from these numbers; nothing
    here decides it. Reported, for both models side by side:

      1. thresholds (2.5% per tail, the same 20,000 random choosers as
         design.json and script 6) and detection at the planning weights
         (people at 0.5 / 0.7, lapse 0.1, no side bias, the same responses
         fitted with both models), and the largest detection loss of the
         adjusted model against the side model (as in script 6's rule (b));
      2. false positives with no effect (script 7's rule rows) and power
         (planning and half shift, all nuisances at their defaults, fixed
         and rescaled weights), paired t-test (the Wilcoxon in the CSV);
      3. the episode-dependent strategy rows (both definitions, both
         directions, k = 0.25 / 0.5 / 1.0) on the v5b lists.

HOW (a rerun of the affected cells only)
    Every cell is script 7's: the same pool (script 7's simulate_pool,
    the same named stream, so the same people and the same answers) and the
    same studies (the same stream). The responses are kept and every
    person is fitted twice: script 7's fit (the side model) and the
    adjusted model, on the analysed trials in item order (a fit does not
    depend on row order; checked). The side-model rates therefore equal
    script 7's paired_power.csv exactly (checked), and the difference
    between the models is not Monte Carlo noise.

REUSED (imported, not copied; SHA-256 checked)
    simulate_power_paired_v5b.py (script 7)  load_stimulus_table,
        load_package (with its package checks), load_thresholds,
        rescale_table, build_scenarios, simulate_pool, draw_studies,
        rate_se, t_greater, EFFECTS and every person-model setting
    thresholds_v5b.py (script 6)    fit_firth_batch_k, respond,
        thresholds_from, the threshold and planning settings
    simulate_power_v4_n10_20.py     simulate_many, fit_firth (check),
        FIRTH_ITER / FIRTH_TOL / FIRTH_MAX_STEP
    explore_block_length_v5b.py     wilcoxon_greater, scaled_scenario,
        rescale_factors (through script 7)
    analyse_pair_capacity_v5b.py    Streams

INPUTS (relative to the project folder; run from there)
    the inputs of script 7 (frozen v5b packages, the v5 packages, the
    stimulus set table, script 6's run, script 2's paired_power.csv) and
    PAIRED_RUN_DIR, script 7's run (paired_power.csv, checks.csv,
    run_meta.json)

OUTPUTS (one new folder per run under OUTPUT_ROOT; never overwritten)
    report.txt, summary.txt, checks.csv, settings.json, run_meta.json
    thresholds_detection.csv   part 1, per size, set, feature and model
    paired_rates.csv           parts 2 and 3: every cell, N 10-20, both
                               models, t-test and Wilcoxon
    nuisance_terms.csv         mean N-term estimates per block (nuisance;
                               never results)
    episode_strategy_models.png

REQUIRES  Python 3.9+, numpy, pandas, scipy; matplotlib for the figure
RUN       python strategy_adjusted_model_v5b.py
          (about 1 minute with 31 workers; 4 minutes with 2)
"""

# One thread per process (BLAS and OpenMP), set before numpy is imported,
# so the result is the same for any N_WORKERS.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

# ============================================================
# SETTINGS
# ============================================================

# PROJECT_ROOT: None = the current folder (run from the project folder).
PROJECT_ROOT = None

PAIRED_SCRIPT = "simulate_power_paired_v5b.py"
THRESHOLDS_SCRIPT = "thresholds_v5b.py"
V4_SIM_SCRIPT = "simulate_power_v4_n10_20.py"
CAPACITY_SCRIPT = "analyse_pair_capacity_v5b.py"
EXPLORE_SCRIPT = "explore_block_length_v5b.py"
# EXPECTED_SHA256: the versions delivered and run for scripts 6 and 7;
# the other scripts are checked by script 7's own rules (FROZEN.json and
# its EXPECTED_SHA256).
EXPECTED_SHA256 = {
    PAIRED_SCRIPT:
        "21ebba17c7fd7dc8c216fd4f72321755ef6b5a31ef39abeecc526f75ec17ad0d",
    THRESHOLDS_SCRIPT:
        "07b2423e480df29a36975e9072d2f5e81c9c71b908dfc281119111b30d4c4115",
}

# PAIRED_RUN_DIR: script 7's run; its side-model rates must be reproduced
# exactly here.
PAIRED_RUN_DIR = ("data/power_v5b/paired_v5b/sizes150-180_N10-20_model-side"
                  "_pool50000_studies10000_mixed2500_seed42__s41cc6f67")

OUTPUT_ROOT = "data/power_v5b/adjusted_model_v5b"
MAKE_FIGURES = True
FIG_DPI = 150

# N_WORKERS / CHECK_ONE_WORKER: as in scripts 6 and 7.
N_WORKERS = None
CHECK_ONE_WORKER = True

# The cells rerun (script 7's scenario names and roles; v5b lists only):
#   the rule rows (no effect), the headline power rows (planning and half
#   shift, all nuisances at defaults, fixed and rescaled weights), and
#   every episode-dependent strategy row (both directions).
RULE_ROLE = "null (rule)"
POWER_ROLES = ("power", "power (rescaled)")
POWER_EFFECTS = ("planning", "half shift")
EPISODE_ROLES = ("episode strategy", "episode strategy (reverse)")

# Tables show these N; paired_rates.csv has script 7's whole N_GRID.
TABLE_N = [10, 15, 20]

# Part 1: false-positive limit used to count cells, as script 7's rule
# (ALPHA + 2 SE, the cell's measured SE); and the detection-loss reference
# (script 6's rule (b): RULE_DETECTION_LOSS_MAX).
RULE_SE_MULT = 2.0

# Checks
REFIT_CHECK_PEOPLE = 2_000   # people per pool refitted with the side model
REFIT_TOL = 1e-9             # ... in item order, against script 7's fit
RATE_TOL = 1e-12             # side-model rates against script 7's run
THRESHOLD_TOL = 1e-12        # side-model thresholds against script 6's

RANDOM_SEED = 42
STREAM_KEY = "adjusted_model_v5b"

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
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

FEATURES = ("sat", "con")
FEATURE_LABEL = {"sat": "saturation", "con": "contrast"}
MODELS = ("side", "adjusted")
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


SETTING_NAMES = [
    "PROJECT_ROOT", "PAIRED_SCRIPT", "THRESHOLDS_SCRIPT", "V4_SIM_SCRIPT",
    "CAPACITY_SCRIPT", "EXPLORE_SCRIPT", "EXPECTED_SHA256", "PAIRED_RUN_DIR",
    "OUTPUT_ROOT", "MAKE_FIGURES", "FIG_DPI", "N_WORKERS", "CHECK_ONE_WORKER",
    "RULE_ROLE", "POWER_ROLES", "POWER_EFFECTS", "EPISODE_ROLES", "TABLE_N",
    "RULE_SE_MULT", "REFIT_CHECK_PEOPLE", "REFIT_TOL", "RATE_TOL",
    "THRESHOLD_TOL", "RANDOM_SEED", "STREAM_KEY",
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
    s = settings_dict()
    s.pop("N_WORKERS")
    h = hashlib.sha256(json.dumps(jsonable(s), sort_keys=True)
                       .encode()).hexdigest()[:8]
    base = root_path(OUTPUT_ROOT, f"side-vs-adjusted_seed{RANDOM_SEED}__s{h}")
    out, k = base, 1
    while os.path.exists(out):
        k += 1
        out = f"{base}_run{k}"
    os.makedirs(out)
    return out


def import_file(fname):
    path = root_path(fname)
    spec = importlib.util.spec_from_file_location(
        os.path.splitext(fname)[0] + "_imported_7b", path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------
# The models
# ------------------------------------------------------------

def design(items, model):
    """Item-order design of one set: side = x_sat, x_con, side;
    adjusted = + N_corpus, N_orig."""
    Z = np.column_stack([items["X"], items["side"]])
    if model == "adjusted":
        Z = np.column_stack([Z, items["N"]])
    return Z


def fitk(X, Y):
    v4, s6 = _W["v4"], _W["s6"]
    return s6.fit_firth_batch_k(X, Y, v4.FIRTH_ITER, v4.FIRTH_TOL,
                                v4.FIRTH_MAX_STEP)


# ------------------------------------------------------------
# Tasks (one per size x set for part 1, one per cell for parts 2-3)
# ------------------------------------------------------------

_W = {}


def _worker_init(state, quiet=True):
    global QUIET
    QUIET = quiet
    s7 = import_file(PAIRED_SCRIPT)
    s7.QUIET = True
    _W.update(s7=s7, v4=import_file(V4_SIM_SCRIPT),
              s6=import_file(THRESHOLDS_SCRIPT),
              cap=import_file(CAPACITY_SCRIPT),
              explore=import_file(EXPLORE_SCRIPT),
              scen=s7.build_scenarios(), **state)


def task_thresholds(size, b):
    """Part 1 for one size and set: the 20,000 random choosers of
    design.json (build_trials_v5b's stream), fitted with both models; and
    people at the planning weights, the same responses fitted with both."""
    v4, s6, cap = _W["v4"], _W["s6"], _W["cap"]
    it = _W["pk"][size]["lists"][f"items_{b}"]
    X2, side = it["X"], it["side"]
    rng = cap.Streams(s6.REPRO_SEED, s6.REPRO_STREAM_KEY, size)(
        s6.REPRO_STREAM_NAME.format(set=b))
    Yr = v4.simulate_many(X2, np.zeros((s6.N_THRESHOLD_PEOPLE, 2)), rng)
    st = cap.Streams(RANDOM_SEED, STREAM_KEY, "planning", size, b)
    W = np.tile(np.asarray(s6.PLANNING_WEIGHTS, float), (s6.N_PEOPLE, 1))
    U = st("uniforms").random((s6.N_PEOPLE, len(X2)))
    Yp = s6.respond(X2, side, W, np.zeros(s6.N_PEOPLE), s6.PLANNING_LAPSE, U)
    out = {"size": size, "set": b, "rows": []}
    for model in MODELS:
        Z = design(it, model)
        est, last = fitk(Z, Yr)
        th = s6.thresholds_from(est[:, :2])
        ep, lp = fitk(Z, Yp)
        out[f"th_{model}"] = th
        out[f"unconv_{model}"] = float(max(np.mean(last > 1e-6),
                                           np.mean(lp > 1e-6)))
        for j, f in enumerate(FEATURES):
            out["rows"].append({
                "size": size, "set": b, "model": model,
                "feature": FEATURE_LABEL[f],
                "upper": th["upper"][j], "lower": th["lower"][j],
                "null_sd": float(est[:, j].std()),
                "planning_weight": s6.PLANNING_WEIGHTS[j],
                "planning_mean": float(ep[:, j].mean()),
                "planning_sd": float(ep[:, j].std()),
                "detection": float(np.mean(ep[:, j] > th["upper"][j])),
                "opposite": float(np.mean(ep[:, j] < th["lower"][j])),
                "unconverged": out[f"unconv_{model}"]})
    Z5 = design(it, "adjusted")
    out["n_corr"] = np.corrcoef(Z5.T).tolist()
    return out


def effect_of(name, size):
    s7 = _W["s7"]
    if name.endswith(s7.RESCALED_SUFFIX):
        base = s7.EFFECTS[name[:-len(s7.RESCALED_SUFFIX)]]
        r = _W["resc"][size]
        return _W["explore"].scaled_scenario(base, (r["k_sat"], r["k_con"]))
    return s7.EFFECTS[name]


def task_cell(size, i):
    """One of script 7's cells: its pool (same stream, responses kept), every
    person fitted with both models, and its studies (same stream)."""
    s7, cap, explore = _W["s7"], _W["cap"], _W["explore"]
    s = _W["scen"][i]
    pk = _W["pk"][size]
    st = cap.Streams(s7.RANDOM_SEED, s7.STREAM_KEY, "pool", s["lists"], size,
                     s["name"], s["effect"])
    pool = s7.simulate_pool(pk, effect_of(s["effect"], size), s["nuis"],
                            st("people"), _W["fit7"], keep_y=True)
    est = {"side": {"D": pool["est_D"], "O": pool["est_O"]},
           "adjusted": {}}
    worst_refit = 0.0
    nterm = []
    for b, key in (("D", "der_set"), ("O", "ord_set")):
        ea = np.empty((len(pool["row"]), 5))
        unconv = 0.0
        for setname in s7.SETS:
            m = np.flatnonzero(pool[key] == setname)
            it = pk["lists"][f"items_{setname}"]
            Y = pool[f"Y_{b}"][m].astype(float)
            ea[m], last = fitk(design(it, "adjusted"), Y)
            unconv = max(unconv, float(np.mean(last > 1e-6)))
            c = m[:REFIT_CHECK_PEOPLE // 2]
            es, _ = fitk(design(it, "side"), pool[f"Y_{b}"][c].astype(float))
            worst_refit = max(worst_refit, float(np.max(np.abs(
                es - pool[f"est_{b}"][c]))))
        est["adjusted"][b] = ea
        keep = ~pool["excluded"]
        nterm.append({"size": size, "scenario": s["name"],
                      "effect": s["effect"], "role": s["role"], "block": b,
                      "mean_N_corpus_term": float(ea[keep, 3].mean()),
                      "mean_N_orig_term": float(ea[keep, 4].mean()),
                      "unconverged": unconv})
    sts = cap.Streams(s7.RANDOM_SEED, s7.STREAM_KEY, "studies", s["lists"],
                      size, s["name"], s["effect"], "")
    rows = []
    for N in s7.N_GRID:
        idx, n_excl, sb = s7.draw_studies(pool, N, s7.N_STUDIES,
                                          sts(f"N{N}"))
        for model in MODELS:
            D = est[model]["D"][idx, :2] - est[model]["O"][idx, :2]
            for j, f in enumerate(FEATURES):
                for test in s7.TESTS:
                    rej = (s7.t_greater(D[:, :, j]) if test == "t"
                           else explore.wilcoxon_greater(D[:, :, j])
                           ) <= s7.ALPHA
                    p, se = s7.rate_se(rej, sb)
                    rows.append({"size": size, "scenario": s["name"],
                                 "effect": s["effect"], "role": s["role"],
                                 "N": N, "model": model, "test": test,
                                 "feature": FEATURE_LABEL[f], "rate": p,
                                 "se": se,
                                 "mean_diff": float(D[:, :, j].mean())})
    return {"rows": rows, "nterm": nterm, "refit": worst_refit}


TASKS = {"thresholds": task_thresholds, "cell": task_cell}


def _run_task(kind, args):
    return TASKS[kind](*args)


# ------------------------------------------------------------
# Figure and report
# ------------------------------------------------------------

def figure(ep, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"side": "#0072B2", "adjusted": "#D55E00"}
    nn = max(TABLE_N)
    q = ep[(ep["N"] == nn) & (ep["test"] == "t")]
    sizes = sorted(q["size"].unique())
    fig, axes = plt.subplots(len(sizes), 2, figsize=(10, 4 * len(sizes)),
                             sharey=True, squeeze=False)
    for i, size in enumerate(sizes):
        for j, f in enumerate(("contrast", "saturation")):
            ax = axes[i, j]
            for model in MODELS:
                for (dn, dr), ls in zip([("corpus", "main"),
                                         ("orig", "main"),
                                         ("corpus", "reverse"),
                                         ("orig", "reverse")],
                                        ["-", "--", ":", "-."]):
                    r = q[(q["size"] == size) & (q["model"] == model)
                          & (q["definition"] == dn)
                          & (q["direction"] == dr)
                          & (q["feature"] == f)].sort_values("k")
                    if r.empty:
                        continue
                    ax.plot(r["k"], r["rate"], ls, color=colours[model],
                            marker="o", ms=3,
                            label=f"{model} model, {dn}, {dr}")
            ax.axhline(0.05, color="#888888", lw=0.8, ls=":")
            ax.set_ylim(0, None)
            ax.set_title(f"{f.capitalize()}, {size} trials per block",
                         fontsize=10)
            ax.set_xlabel("k of the episode-dependent strategy")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color="#dddddd", linewidth=0.6)
            if j == 0:
                ax.set_ylabel(f"Share of studies significant\n(paired "
                              f"t-test, N = {nn}, no effect)")
    axes[0, -1].legend(fontsize=6, frameon=False)
    fig.suptitle("Episode-dependent strategy, v5b lists: side model and "
                 "strategy-adjusted model\n(k values are assumptions)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "episode_strategy_models.png"),
                dpi=FIG_DPI)
    plt.close(fig)


def tab(df, nd=3):
    return "  " + df.to_string(index=False, float_format=lambda x:
                               f"{x:.{nd}f}").replace("\n", "\n  ")


def side_by_side(rates, sel, by):
    q = rates[sel & rates["N"].isin(TABLE_N) & (rates["test"] == "t")]
    q = q.assign(col=q["model"].str[:4] + ":N" + q["N"].astype(str))
    piv = q.pivot_table(index=by, columns="col", values="rate", sort=False)
    cols = [f"{m[:4]}:N{n}" for m in MODELS for n in TABLE_N]
    piv = piv[[c for c in cols if c in piv.columns]]
    for n in TABLE_N:
        if f"side:N{n}" in piv and f"adju:N{n}" in piv:
            piv[f"diff:N{n}"] = piv[f"adju:N{n}"] - piv[f"side:N{n}"]
    return piv.reset_index()


def main():
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    ck = Checks()
    hashes = {}
    out = resolve_output_dir()
    log(f"output folder {out}")
    n_workers = N_WORKERS or max(1, (os.cpu_count() or 2) - 1)
    log(f"{n_workers} worker process(es), one thread each")

    # --- the imported scripts ---
    for fn in (PAIRED_SCRIPT, THRESHOLDS_SCRIPT):
        p = root_path(fn)
        ok = os.path.exists(p)
        ck.add("inputs", f"{fn} exists", ok, p)
        if not ok:
            raise SystemExit(f"missing {p}")
        hashes[fn] = sha256_of(p)
        ck.add("inputs", f"{fn} is the version delivered and run (SHA-256)",
               hashes[fn] == EXPECTED_SHA256[fn], hashes[fn][:16] + "...")
    s7 = import_file(PAIRED_SCRIPT)
    s7.QUIET = True

    # --- script 7's inputs, through script 7's own loaders and checks ---
    with open(os.path.join(root_path(s7.PACKAGES[s7.BLOCK_SIZES[0]]),
                           "FROZEN.json")) as f:
        frozen0 = json.load(f)
    stim = s7.load_stimulus_table(frozen0, ck, hashes)
    pk = {size: s7.load_package(size, ck, hashes, stim, "v5b")
          for size in s7.BLOCK_SIZES}
    v4 = s7.load_module(s7.V4_SIM_SCRIPT, ck, hashes,
                        frozen0["scripts_sha256"])
    s7.load_module(s7.CAPACITY_SCRIPT, ck, hashes, frozen0["scripts_sha256"])
    explore = s7.load_module(s7.EXPLORE_SCRIPT, ck, hashes)
    s6 = s7.load_module(s7.THRESHOLDS_SCRIPT, ck, hashes)
    TH, meta6 = s7.load_thresholds(ck, hashes)
    resc_df, resc = s7.rescale_table(v4, explore, pk, ck)

    # --- script 7's run ---
    d7 = root_path(PAIRED_RUN_DIR)
    ok = all(os.path.exists(os.path.join(d7, fn)) for fn in
             ("paired_power.csv", "checks.csv", "run_meta.json"))
    ck.add("inputs", "script 7's run exists (paired_power.csv, checks.csv, "
           "run_meta.json)", ok, PAIRED_RUN_DIR)
    if not ok:
        raise SystemExit(f"missing files in {d7}")
    for fn in ("paired_power.csv", "checks.csv", "run_meta.json"):
        hashes[f"{PAIRED_RUN_DIR}/{fn}"] = sha256_of(os.path.join(d7, fn))
    c7 = pd.read_csv(os.path.join(d7, "checks.csv"))
    ck.add("inputs", "script 7's run passed every check",
           (c7["status"] == "PASS").all(),
           f"{(c7['status'] == 'PASS').sum()} PASS, "
           f"{(c7['status'] != 'PASS').sum()} FAIL")
    with open(os.path.join(d7, "run_meta.json")) as f:
        meta7 = json.load(f)
    ck.add("inputs", "script 7's run read the same script 6 thresholds",
           meta7["input_sha256"].get(
               f"{s7.THRESHOLDS_RUN_DIR}/thresholds.csv")
           == hashes.get(f"{s7.THRESHOLDS_RUN_DIR}/thresholds.csv"),
           f"primary test {meta7.get('chosen_test')}")
    pw7 = pd.read_csv(os.path.join(d7, "paired_power.csv"))

    # --- tool check: the adjusted fit against v4's single-person fit ---
    _W.update(v4=v4, s6=s6)
    it = pk[s7.BLOCK_SIZES[-1]]["lists"]["items_A"]
    Z5 = design(it, "adjusted")
    rng = np.random.default_rng(RANDOM_SEED)
    Y = (rng.random((30, len(Z5))) < 0.5).astype(float)
    b1, _ = fitk(Z5, Y)
    b2 = np.array([v4.fit_firth(Z5, Y[i]) for i in range(30)])
    ck.add("tools", "the adjusted-model fit (5 terms) equals v4's single-"
           "person fit_firth", np.max(np.abs(b1 - b2)) < 1e-9,
           f"max |difference| {np.max(np.abs(b1 - b2)):.1e}")

    # --- the cells ---
    scen = s7.build_scenarios()
    sel = [i for i, s in enumerate(scen) if s["lists"] == "v5b" and (
        s["role"] == RULE_ROLE
        or (s["role"] in POWER_ROLES and s["name"] == "all nuisances "
            "(defaults)" and s["effect"].replace(s7.RESCALED_SUFFIX, "")
            in POWER_EFFECTS)
        or s["role"] in EPISODE_ROLES)]
    jobs = ([("thresholds", (size, b)) for size in s7.BLOCK_SIZES
             for b in s7.SETS]
            + [("cell", (size, i)) for size in s7.BLOCK_SIZES for i in sel])
    state = {"pk": pk, "resc": resc}
    log(f"{len(jobs)} tasks ({len(sel)} cells x {len(s7.BLOCK_SIZES)} "
        "sizes, and thresholds)")
    _worker_init(state, quiet=False)        # this process (fit7 below)
    _W["fit7"] = s7.Fitter(v4, s6)
    if n_workers > 1:
        res = run_tasks(jobs, n_workers, state)
    else:
        res = [_run_task(k, a) for k, a in jobs]
    if CHECK_ONE_WORKER and n_workers > 1:
        j0 = [j for j in jobs if j[0] == "cell"][0]
        log(f"rerunning task {j0} in this process (one worker)")
        again = task_cell(*j0[1])
        r0 = res[jobs.index(j0)]
        ok = (pd.DataFrame(again["rows"]).equals(pd.DataFrame(r0["rows"]))
              and again["nterm"] == r0["nterm"])
        ck.add("workers", f"one worker gives the same result as {n_workers}",
               ok, f"{j0[1][0]} trials, '{scen[j0[1][1]]['name']}'")
    elif CHECK_ONE_WORKER:
        ck.add("workers", "one worker gives the same result (run with one "
               "worker: nothing to compare)", True, "N_WORKERS = 1")

    # --- part 1 ---
    th_res = [r for (k, _), r in zip(jobs, res) if k == "thresholds"]
    td = pd.DataFrame([row for r in th_res for row in r["rows"]])
    t6 = pd.read_csv(os.path.join(root_path(s7.THRESHOLDS_RUN_DIR),
                                  "thresholds.csv"))
    t6 = t6[(t6["model"] == "side")
            & (t6["source"] == "same choosers as design.json")]
    worst = 0.0
    for r in td[td["model"] == "side"].itertuples():
        q = t6[(t6["size"] == r.size) & (t6["set"] == r.set)
               & (t6["feature"] == r.feature)].iloc[0]
        worst = max(worst, abs(q["upper"] - r.upper),
                    abs(q["lower"] - r.lower))
    ck.add("part 1", "side-model thresholds reproduce script 6's (same "
           "20,000 random choosers)", worst <= THRESHOLD_TOL,
           f"max |difference| {worst:.1e}")
    ck.add("part 1", "every fit converged (random choosers and planning "
           "people, both models)", (td["unconverged"] == 0).all(),
           f"largest unconverged share {td['unconverged'].max():.5f}")
    piv = td.pivot_table(index=["size", "set", "feature"], columns="model",
                         values="detection")
    loss = (piv["side"] - piv["adjusted"])
    worst_loss = float(loss.max())
    wl = loss.idxmax()
    ncorr = {(r["size"], r["set"]): np.array(r["n_corr"]) for r in th_res}

    # --- parts 2 and 3 ---
    cells = [r for (k, _), r in zip(jobs, res) if k == "cell"]
    rates = pd.DataFrame([row for r in cells for row in r["rows"]])
    nterm = pd.DataFrame([row for r in cells for row in r["nterm"]])
    worst_refit = max(r["refit"] for r in cells)
    ck.add("cells", "refitting the side model on the kept responses in item "
           "order reproduces script 7's fit", worst_refit <= REFIT_TOL,
           f"max |difference| {worst_refit:.1e} ({REFIT_CHECK_PEOPLE:,} "
           "people per pool)")
    ck.add("cells", "every adjusted-model fit converged",
           (nterm["unconverged"] == 0).all(),
           f"largest unconverged share {nterm['unconverged'].max():.5f}")
    m = rates[rates["model"] == "side"].merge(
        pw7[pw7["lists"] == "v5b"], on=["size", "scenario", "effect", "N",
                                        "test", "feature"],
        suffixes=("", "_7"))
    dr = float(np.max(np.abs(m["rate"] - m["rate_7"]))) if len(m) else 1.0
    ck.add("cells", "side-model rates equal script 7's paired_power.csv "
           "(same pools, same studies)",
           len(m) == (rates["model"] == "side").sum() and dr <= RATE_TOL,
           f"{len(m)} rates; max |difference| {dr:.1e}")

    se0 = np.sqrt(s7.ALPHA * (1 - s7.ALPHA) / s7.N_STUDIES)
    nul = rates[(rates["role"] == RULE_ROLE) & rates["N"].isin(s7.RULE_N)
                & (rates["test"] == "t")].copy()
    nul["over"] = nul["rate"] > s7.ALPHA + RULE_SE_MULT * np.maximum(
        nul["se"], se0)
    nul_sum = nul.groupby("model").agg(mean_rate=("rate", "mean"),
                                       largest=("rate", "max"),
                                       cells_over_limit=("over", "sum"),
                                       cells=("rate", "size")).reset_index()

    ep = rates[rates["role"].isin(EPISODE_ROLES)].copy()
    ep["k"] = ep["scenario"].str.extract(r"k ([0-9.]+)").astype(float)
    ep["definition"] = ep["scenario"].str.extract(r"\((corpus|orig),")
    ep["direction"] = np.where(ep["role"].str.contains("reverse"),
                               "reverse", "main")
    ep_t = ep[ep["N"].isin(TABLE_N) & (ep["test"] == "t")]
    ep_tab = ep_t.assign(col=ep_t["model"].str[:4] + ":N"
                         + ep_t["N"].astype(str)).pivot_table(
        index=["size", "direction", "definition", "k", "feature"],
        columns="col", values="rate").reset_index()
    md = ep[(ep["N"] == TABLE_N[0]) & (ep["test"] == "t")].pivot_table(
        index=["size", "direction", "definition", "k", "feature"],
        columns="model", values="mean_diff").reset_index().rename(
        columns={"side": "mean_diff_side", "adjusted": "mean_diff_adj"})
    ep_tab = ep_tab.merge(md, on=["size", "direction", "definition", "k",
                                  "feature"])
    cols = (["size", "direction", "definition", "k", "feature"]
            + [f"{mm[:4]}:N{n}" for mm in MODELS for n in TABLE_N]
            + ["mean_diff_side", "mean_diff_adj"])
    ep_tab = ep_tab[cols]

    pwr = rates[rates["role"].isin(POWER_ROLES)]

    # --- write ---
    td.to_csv(os.path.join(out, "thresholds_detection.csv"), index=False)
    rates.to_csv(os.path.join(out, "paired_rates.csv"), index=False)
    nterm.to_csv(os.path.join(out, "nuisance_terms.csv"), index=False)
    if MAKE_FIGURES:
        try:
            figure(ep, out)
        except ImportError:
            log("matplotlib not installed: no figure")
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(jsonable(settings_dict()), f, indent=2)
    pd.DataFrame(ck.rows).to_csv(os.path.join(out, "checks.csv"),
                                 index=False)

    # --- summary ---
    nn = max(TABLE_N)
    S = []
    S.append(f"* Checks: {len(ck.rows) - ck.n_fail} of {len(ck.rows)} pass"
             + ("." if not ck.n_fail else f"; {ck.n_fail} FAIL."))
    S.append("* The adjusted model is the side model plus two nuisance terms "
             "per block (N_corpus, N_orig). Their estimates are never "
             "results. Nothing here decides the registered model.")
    S.append(f"* Part 1: largest detection loss of the adjusted model "
             f"against the side model at the planning weights: "
             f"{worst_loss:+.4f} ({wl[0]} {wl[1]}, {wl[2]}); script 6's "
             f"rule (b) allowed {s6.RULE_DETECTION_LOSS_MAX}.")
    for r in nul_sum.itertuples():
        S.append(f"* Part 2, rule rows with no effect ({r.model} model, "
                 f"paired t): mean {r.mean_rate:.4f}, largest "
                 f"{r.largest:.4f}, {int(r.cells_over_limit)} of "
                 f"{int(r.cells)} cells over 0.05 + 2 SE.")
    for eff in ("planning", "half shift"):
        for sfx, lab in (("", "fixed"), (s7.RESCALED_SUFFIX, "rescaled")):
            parts = []
            for size in s7.BLOCK_SIZES:
                for f in ("contrast", "saturation"):
                    q = pwr[(pwr["effect"] == eff + sfx)
                            & (pwr["size"] == size) & (pwr["N"] == nn)
                            & (pwr["test"] == "t") & (pwr["feature"] == f)
                            ].set_index("model")["rate"]
                    parts.append(f"{size} {f} {q['side']:.2f} -> "
                                 f"{q['adjusted']:.2f}")
            S.append(f"* Part 2, power at N = {nn}, {eff}, {lab} weights "
                     "(side -> adjusted): " + "; ".join(parts) + ".")
    for k in sorted(ep["k"].unique()):
        q = ep_t[(ep_t["N"] == nn) & (ep_t["k"] == k)]
        g = q.groupby("model")["rate"].max()
        S.append(f"* Part 3, episode-dependent strategy k {k:g} (both "
                 f"definitions and directions, paired t, N = {nn}): largest "
                 f"false-positive rate, side {g['side']:.3f}, adjusted "
                 f"{g['adjusted']:.3f}.")
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("THE STRATEGY-ADJUSTED PER-PERSON MODEL "
                "(strategy_adjusted_model_v5b.py): IN PLAIN WORDS\n\n"
                + "\n".join(S) + "\n")

    # --- report ---
    import scipy
    files = sorted(set(os.listdir(out)) | {"report.txt", "run_meta.json"})
    minutes = (time.time() - T0) / 60
    L = []
    w = L.append
    w("THE STRATEGY-ADJUSTED PER-PERSON MODEL  (strategy_adjusted_model_v5b"
      ".py, script 7b)")
    w("=" * 78)
    w(f"run       {started}  ({minutes:.1f} min, {n_workers} worker "
      "process(es), one thread each)")
    w(f"output    {out}")
    w(f"python    {platform.python_version()}, numpy {np.__version__}, "
      f"scipy {scipy.__version__}, pandas {pd.__version__}")
    w(f"script 7  {PAIRED_RUN_DIR}")
    w("")
    w("0. IN PLAIN WORDS")
    w("-" * 78)
    L += S
    w("")
    w("PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES: every rate with "
      "simulated")
    w("preferences, strategies or nuisances depends on values chosen for "
      "planning.")
    w("")
    w("1. WHAT WAS DONE")
    w("-" * 78)
    w("Adjusted model: Firth logistic regression without intercept on "
      "x_sat, x_con, side,")
    w("N_corpus and N_orig (each block fitted separately; N from the lists, "
      "corpus SD).")
    w("The N terms are nuisance terms: their estimates are never results "
      "(their means are")
    w("in nuisance_terms.csv only to show what they absorb).")
    w("Every cell is script 7's (same pool stream, same study stream); each "
      "person is fitted")
    w("with both models on the same answers, so the differences between the "
      "models are not")
    w("Monte Carlo noise. Script 7's side-model rates are reproduced exactly "
      "(section 2).")
    w("Correlations of the adjusted model's columns (x_sat, x_con, side, "
      "N_corpus, N_orig),")
    w("per size and set (the N terms cost precision where they correlate "
      "with the predictors):")
    for (size, b), C in ncorr.items():
        w(f"  {size} {b}: " + "  ".join(
            f"{a}-{c} {C[i, j]:+.3f}"
            for i, a in enumerate(("sat", "con", "side", "Nc", "No"))
            for j, c in enumerate(("sat", "con", "side", "Nc", "No"))
            if j > i))
    w("")
    w("2. CHECKS")
    w("-" * 78)
    for r in ck.rows:
        w(f"  {r['status']}  [{r['section']}] {r['check']}")
        if r["detail"]:
            w(f"          {r['detail']}")
    w(f"\n  {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL")
    w("")
    w("3. PART 1: THRESHOLDS AND DETECTION AT THE PLANNING WEIGHTS")
    w("-" * 78)
    w("Thresholds: 2.5% per tail of the 20,000 random choosers of "
      "design.json (the same")
    w(f"answers for both models). Planning: {s6.N_PEOPLE:,} people at "
      f"{s6.PLANNING_WEIGHTS}, lapse {s6.PLANNING_LAPSE}, no side bias;")
    w("the same answers fitted with both models; detection = share above "
      "the model's own upper")
    w("threshold. [planning weights and lapse: ASSUMPTIONS]")
    w(tab(td[["size", "set", "feature", "model", "upper", "lower",
              "null_sd", "planning_mean", "planning_sd", "detection",
              "opposite"]], 4))
    w("")
    w("  Detection loss (side minus adjusted):")
    w("  " + loss.rename("loss").reset_index().to_string(
        index=False, float_format=lambda x: f"{x:+.4f}").replace(
        "\n", "\n  "))
    w(f"  Largest loss {worst_loss:+.4f} ({wl[0]} {wl[1]}, {wl[2]}); "
      f"script 6's rule (b) allowed {s6.RULE_DETECTION_LOSS_MAX}"
      " (reference only).")
    w("")
    w("4. PART 2: FALSE POSITIVES WITH NO EFFECT (script 7's rule rows) AND "
      "POWER")
    w("-" * 78)
    w("Paired t-test, share of studies significant (one-sided, 0.05); "
      "side/adju = model; diff = adjusted - side.")
    w("Rule rows: script 7's rows used by the test-choice rule.")
    w(tab(nul_sum, 4))
    w("")
    w(tab(side_by_side(rates, rates["role"] == RULE_ROLE,
                       ["size", "scenario", "feature"]), 3))
    w("")
    w("Power, all nuisances at their defaults, fixed and rescaled weights "
      "[ASSUMPTIONS]:")
    w(tab(side_by_side(rates, rates["role"].isin(POWER_ROLES),
                       ["size", "effect", "feature"]), 3))
    w("")
    w("5. PART 3: THE EPISODE-DEPENDENT STRATEGY, v5b LISTS, BOTH MODELS")
    w("-" * 78)
    w("main: derealization -k x N, ordinary +k x N; reverse: the other way "
      "round. Paired")
    w("t-test, no effect; mean_diff: mean D - O weight difference over "
      "studies at N = 10.")
    w("[k values: ASSUMPTIONS]")
    w(tab(ep_tab, 3))
    w("")
    w("  Mean nuisance-term estimates per block (never results; they show "
      "what the terms absorb):")
    q = nterm[nterm["role"].isin(EPISODE_ROLES)].pivot_table(
        index=["size", "scenario"], columns="block",
        values=["mean_N_corpus_term", "mean_N_orig_term"], sort=False)
    w("  " + q.to_string(float_format=lambda x: f"{x:+.3f}").replace(
        "\n", "\n  "))
    w("")
    w("6. FILES")
    w("-" * 78)
    for fn in files:
        w(f"  {fn}")
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out, "run_meta.json"), "w") as f:
        json.dump(jsonable({
            "started": started, "minutes": minutes, "workers": n_workers,
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__,
            "output_dir": out, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "input_sha256": hashes,
            "largest_detection_loss": worst_loss,
            "largest_detection_loss_where": list(map(str, wl)),
            "checks_pass": len(ck.rows) - ck.n_fail,
            "checks_fail": ck.n_fail}), f, indent=2)
    log(f"done: {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL")
    log(f"report: {os.path.join(out, 'report.txt')}")


def _worker_init_fit(state):
    _worker_init(state)
    _W["fit7"] = _W["s7"].Fitter(_W["v4"], _W["s6"])


def run_tasks(jobs, n_workers, state):
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(n_workers, len(jobs)),
                             mp_context=ctx, initializer=_worker_init_fit,
                             initargs=(state,)) as ex:
        futs = [ex.submit(_run_task, *j) for j in jobs]
        out = []
        for k, f in enumerate(futs):
            out.append(f.result())
            if (k + 1) % 10 == 0 or k + 1 == len(futs):
                log(f"  {k + 1} of {len(futs)} tasks done")
        return out


if __name__ == "__main__":
    main()
