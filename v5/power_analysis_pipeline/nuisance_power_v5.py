"""
nuisance_power_v5.py  --  v5 power analysis, extra script: which nuisances
                          cost the primary test its power

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
      5. "edited looks odd"         a quarter at -0.5 -> nobody
      6. block order                second block: lapse +0.05 and
                                    preferences x0.9 -> both off together

    plus the two anchors: all defaults, and no nuisances. For each nuisance:
    the power returned when it is switched off (power without it minus power
    with all defaults), with its Monte Carlo SE; the six ranked by the
    average over 12 cells (2 block sizes x N = 10, 15, 20 x 2 features),
    separately for planning and half shift; and whether the six recoveries
    add up to the total drop (no nuisances minus all defaults).
    Power only; paired t-test only; pair-specific preferences off.

PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES
    The responders' shift, the nuisances' default values and every other
    person-model value are planning assumptions from script 7. The ranking
    says which of those ASSUMED nuisances matter most at their ASSUMED
    sizes; it is not a measurement of real participants.

COMMON RANDOM NUMBERS
    All eight conditions of one effect and block size are simulated from the
    same random draws (same people, same tastes, same choices' random
    numbers) and the same study draws (which pool members are sampled into
    each study), so a recovery reflects only the nuisance switched off.
    Two consequences, stated in the report:
      * script 7's simulate_pool draws the person x scene deviations only
        when scene dependence is on, so switching it to exactly 0 would shift
        every later draw. "Scene dependence off" is therefore SD x 1e-9
        (SCENE_OFF_MULT): the draws stay aligned, and the deviations are
        below 1e-9 logit, numerically nothing.
      * changing the second-block lapse changes catch failures and hence who
        is excluded, so in the block-order condition some studies sample
        different people; the report gives how many.
    SEs: every rate and every difference is computed per study and its SE
    measured from script 7's sub-pools (each study draws from one sub-pool);
    for differences this is the SE of the paired difference, so it is small.
    The average over 12 cells gets its SE from the sub-pool averages.

REUSED (imported, not copied; SHA-256 of each file recorded and checked)
    simulate_power_paired_v5.py (script 7): load_package, load_thresholds,
        simulate_pool, person_thresholds, draw_studies, run_tests, rate_se,
        scenario_rows (exact reproduction), Fitter, EFFECTS, NUISANCE_OFF,
        NUISANCE_DEFAULT and its settings (pool size, studies, sub-pools,
        seed and stream key)
    through script 7: simulate_power_v4_n10_20.py, thresholds_v5.py,
        explore_block_length_v5.py, analyse_pair_capacity_v5.py

CHECKS
    * the imported scripts are the expected versions (SHA-256);
    * EXACT REPRODUCTION: the two anchors rerun with script 7's own stream
      keys give script 7's numbers to the last digit;
    * the matched anchors (this script's own streams) agree with script 7
      within Monte Carlo error;
    * the draws are shared: identical responders and tastes across the eight
      conditions; exclusions differ only where the lapse rate differs;
    * the total drop is positive (nuisances cost power).

INPUTS (relative to PROJECT_ROOT)
    PAIRED_RUN_DIR (script 7's output: paired_power.csv, checks.csv) and,
    through script 7's loader, the frozen lists and script 6's thresholds

OUTPUTS (one new folder per run under OUTPUT_ROOT; never overwritten)
    report.txt, checks.csv, settings.json, run_meta.json
    ranking.csv           average recovery per nuisance and effect, with SE
    recoveries.csv        per cell (size, N, feature): power and recovery
    anchors.csv           both anchor comparisons with script 7
    figure_nuisances.png  recoveries, planning beside half shift

REQUIRES  Python 3.9+, numpy, pandas, scipy, matplotlib
RUN       python nuisance_power_v5.py        (about 5 minutes on 2 cores)
"""

# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = None          # None = the folder this script is in
SCRIPT7 = "simulate_power_paired_v5.py"
EXPECTED_SHA256 = {
    SCRIPT7:
        "14e09055fd122d032665d210713a3a32017c88dc0ea5610cb00a9aba106f366f",
}
PAIRED_RUN_DIR = ("data/power_v5/paired_v5/sizes150-180_N10-20_model-side"
                  "_pool50000_studies10000_mixed2500_seed42__se1c01101")
OUTPUT_ROOT = "data/power_v5/nuisance_v5"
FIG_DPI = 200

BLOCK_SIZES = [150, 180]
N_LIST = [10, 15, 20]
EFFECTS_RUN = ["planning", "half shift"]
TEST = "t"                       # the registered primary test

# The six nuisances: plain label (for the report, the figure and the
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
    ("odd", "a quarter of people avoiding the more edited image",
     {"odd_share": "OFF"}),
    ("order", "fatigue in the second block (more lapses, weaker "
     "preferences)", {"order_lapse": "OFF", "order_scale": "OFF"}),
]
SHORT_LABEL = {"taste": "Taste differs between episodes",
               "scene": "Scene-specific preferences",
               "side": "Side preference",
               "dull": "Shared preference for duller images",
               "odd": "Some avoid the more edited image",
               "order": "Second-block fatigue"}

# --- Verdicts ---
NEGLIGIBLE = 0.01                # "costs almost nothing": average below
INTERACT_SE = 2.0                # "interacting": |sum - total| > 2 SE ...
INTERACT_MIN = 0.01              # ... and > 0.01
TOP_K = 3                        # nuisances named in the summary at most

# --- Anchor checks ---
EXACT_TOL = 1e-12
MAX_Z = 3.5
SHARE_BEYOND = 0.10

RANDOM_SEED = 42
STREAM_KEY = "nuisance_v5"

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
import re
import time

import numpy as np
import pandas as pd

FEATS = {"sat": "saturation", "con": "contrast"}
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
                 "PAIRED_RUN_DIR", "OUTPUT_ROOT", "FIG_DPI", "BLOCK_SIZES",
                 "N_LIST", "EFFECTS_RUN", "TEST", "SCENE_OFF_MULT",
                 "NUISANCES", "SHORT_LABEL", "NEGLIGIBLE", "INTERACT_SE",
                 "INTERACT_MIN", "TOP_K", "EXACT_TOL", "MAX_Z",
                 "SHARE_BEYOND", "RANDOM_SEED", "STREAM_KEY"]


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
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
    h = hashlib.sha256(json.dumps(jsonable(settings_dict(s7)),
                                  sort_keys=True).encode()).hexdigest()[:8]
    name = (f"sizes{'-'.join(map(str, BLOCK_SIZES))}"
            f"_N{'-'.join(map(str, N_LIST))}_test-{TEST}"
            f"_pool{s7.POOL_PEOPLE}_studies{s7.N_STUDIES}"
            f"_seed{RANDOM_SEED}__s{h}")
    base = root_path(OUTPUT_ROOT, name)
    out, k = base, 1
    while os.path.exists(out):
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


def conditions(s7):
    """The eight conditions as nuisance dictionaries."""
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
# Figure
# ------------------------------------------------------------

FORBIDDEN = [r"\bsat\b", r"\bcon\b", r"\bfog\b", r"_", r"\bnuis\b",
             r"\bodd\b", r"\bdull\b"]


def figure(rank, out, bad):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = (rank[rank["effect"] == "planning"]
             .sort_values("recovery")["nuisance"].tolist())
    fig, axes = plt.subplots(1, len(EFFECTS_RUN), figsize=(10, 3.8),
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
    for t in fig.findobj(lambda a: hasattr(a, "get_text")):
        try:
            s = t.get_text()
        except Exception:
            continue
        for pat in FORBIDDEN:
            if s and re.search(pat, s):
                bad.append(s)
    fig.savefig(os.path.join(out, "figure_nuisances.png"), dpi=FIG_DPI)
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    ck = Checks()
    hashes = {}
    s7 = load_module(SCRIPT7, ck, hashes)
    ck7 = s7.Checks()
    with contextlib.redirect_stdout(io.StringIO()):
        v4 = s7.load_module(s7.V4_SIM_SCRIPT, ck7, hashes)
        cap = s7.load_module(s7.CAPACITY_SCRIPT, ck7, hashes)
        explore = s7.load_module(s7.EXPLORE_SCRIPT, ck7, hashes)
        s6 = s7.load_module(s7.THRESHOLDS_SCRIPT, ck7, hashes)
        TH = s7.load_thresholds(ck7, hashes)
        pk = {size: s7.load_package(size, ck7, hashes)
              for size in BLOCK_SIZES}
    for r in ck7.rows:
        ck.add("inputs (via script 7)", r["check"], r["status"] == "PASS",
               r["detail"])
    p7 = root_path(PAIRED_RUN_DIR, "paired_power.csv")
    ok = os.path.exists(p7)
    ck.add("inputs", "script 7's paired_power.csv exists", ok, p7)
    if not ok:
        raise SystemExit(p7)
    hashes["script 7 paired_power.csv"] = sha256_of(p7)
    c7 = pd.read_csv(root_path(PAIRED_RUN_DIR, "checks.csv"))
    ck.add("inputs", "script 7's run passed every check",
           (c7["status"] == "PASS").all(),
           f"{(c7['status'] == 'PASS').sum()} PASS")
    pw7 = pd.read_csv(p7)
    out = resolve_output_dir(s7)
    log(f"output folder {out}")
    fit = s7.Fitter(v4, s6)
    crit = v4.sign_test_critical(max(max(N_LIST), 20))
    COND = conditions(s7)
    ck.add("setup", "the eight conditions differ from all defaults only in "
           "the nuisance switched off",
           all(sum(COND[k][f] != COND["all defaults"][f]
                   for f in COND["all defaults"]) == len(off)
               for k, _, off in NUISANCES), "")

    # --- 1. exact reproduction of script 7 with its own stream keys ---
    anchor_rows = []
    for size in BLOCK_SIZES:
        for eff in EFFECTS_RUN:
            for name, nuis in (("all nuisances (defaults)",
                                s7.NUISANCE_DEFAULT),
                               ("no nuisances", s7.NUISANCE_OFF)):
                role = ("power" if name.startswith("all")
                        else "power (no nuisances)")
                st = cap.Streams(s7.RANDOM_SEED, s7.STREAM_KEY, "pool", size,
                                 name, eff)
                pool = s7.simulate_pool(pk[size], s7.EFFECTS[eff], nuis,
                                        st("people"), fit)
                s7.person_thresholds(pool, size, TH)
                rows, _, _, _ = s7.scenario_rows(
                    pool, {"name": name, "effect": eff, "role": role},
                    size, explore, crit, cap)
                mine = pd.DataFrame(rows)
                ref = pw7[(pw7["scenario"] == name) & (pw7["effect"] == eff)
                          & (pw7["size"] == size) & (pw7["role"] == role)]
                m = mine.merge(ref, on=["N", "test", "feature"],
                               suffixes=("", "_s7"))
                for r in m[m["N"].isin(N_LIST) & (m["test"] == TEST)
                           ].itertuples():
                    anchor_rows.append({
                        "check": "exact (script 7's stream keys)",
                        "size": size, "effect": eff, "anchor": name,
                        "N": r.N, "feature": r.feature, "this_script": r.rate,
                        "script7": r.rate_s7, "diff": r.rate - r.rate_s7})
                log(f"exact reproduction {size} {eff} {name}: "
                    f"max |diff| {np.max(np.abs(m['rate'] - m['rate_s7'])):.1e}"
                    f" over {len(m)} cells")
    ex = pd.DataFrame(anchor_rows)
    ck.add("anchors", "rerun with script 7's own stream keys, the two "
           "anchors reproduce script 7's numbers exactly",
           (ex["diff"].abs() <= EXACT_TOL).all(),
           f"{len(ex)} cells (paired t-test, N {N_LIST}); largest |diff| "
           f"{ex['diff'].abs().max():.1e}")

    # --- 2. the matched family ---
    cell_rows, rank_rows, crn_rows = [], [], []
    batch = {}          # (effect, cond) -> list of per-cell batch means
    for size in BLOCK_SIZES:
        for eff in EFFECTS_RUN:
            pools = {}
            for cond, nuis in COND.items():
                st = cap.Streams(RANDOM_SEED, STREAM_KEY, "pool", size, eff)
                pool = s7.simulate_pool(pk[size], s7.EFFECTS[eff], nuis,
                                        st("people"), fit)
                s7.person_thresholds(pool, size, TH)
                pools[cond] = pool
                log(f"pool {size} {eff} {cond}")
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
                   "responders in all eight conditions", same_resp, "")
            for N in N_LIST:
                rej, idxs, sb0 = {}, {}, None
                for cond, p in pools.items():
                    # a FRESH stream per condition: Streams caches one
                    # generator per name, so reusing it would continue the
                    # sequence instead of repeating it
                    st = cap.Streams(RANDOM_SEED, STREAM_KEY, "studies",
                                     size, eff)
                    idx, _, sb = s7.draw_studies(p, N, s7.N_STUDIES,
                                                 st(f"N{N}"))
                    r = s7.run_tests(p, idx, explore, crit)
                    rej[cond] = {f: r[(TEST, f)].astype(float)
                                 for f in FEATS}
                    idxs[cond] = idx
                    sb0 = sb
                for cond in COND:
                    crn_rows.append({
                        "size": size, "effect": eff, "condition": cond,
                        "N": N, "studies_same_people": float(np.mean(
                            np.all(idxs[cond] == idxs["all defaults"],
                                   axis=1)))})
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
    cells = pd.DataFrame(cell_rows)
    crn = pd.DataFrame(crn_rows)

    def avg(eff, key):
        B = np.mean(np.vstack(batch[(eff, key)]), axis=0)  # per sub-pool
        return float(B.mean()), float(B.std(ddof=1) / np.sqrt(len(B)))

    for eff in EFFECTS_RUN:
        for key, label, _ in NUISANCES:
            m, se = avg(eff, key)
            rank_rows.append({"effect": eff, "nuisance": key,
                              "label": label, "recovery": m,
                              "recovery_se": se})
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
        t, tse = avg(eff, "total")
        s_, sse = avg(eff, "sum")
        i, ise = avg(eff, "interaction")
        add_rows.append({"effect": eff, "total_drop": t, "total_drop_se": tse,
                         "sum_of_six": s_, "sum_se": sse,
                         "sum_minus_total": i, "sum_minus_total_se": ise,
                         "ratio": s_ / t if t else np.nan,
                         "interacting": abs(i) > INTERACT_SE * ise
                         and abs(i) > INTERACT_MIN})
    addv = pd.DataFrame(add_rows)

    # --- checks on the family ---
    oth = crn[crn["N"].isna() & (crn["condition"] != "order")
              & (crn["condition"] != "none")]
    ck.add("common draws", "exclusions identical to all defaults wherever "
           "the lapse rate is unchanged (all but block order and no "
           "nuisances)", (oth["exclusion_differs"] == 0).all(),
           f"largest share differing {oth['exclusion_differs'].max():.4f}")
    od = crn[crn["N"].isna() & crn["condition"].isin(["order", "none"])]
    sp = crn[crn["N"].notna()]
    ck.add("common draws", "studies sample the same people as with all "
           "defaults wherever exclusions are unchanged",
           (sp[~sp["condition"].isin(["order", "none"])][
               "studies_same_people"] == 1).all(),
           f"block order / no nuisances: {od['exclusion_differs'].mean():.1%}"
           " of people excluded differently, "
           f"{sp[sp['condition'].isin(['order', 'none'])]['studies_same_people'].mean():.1%}"
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
                           "N": N, "feature": flab, "this_script": m,
                           "script7": r7["rate"],
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

    bad = []
    figure(rank, out, bad)
    ck.add("figure", "figure text uses plain labels", not bad,
           "; ".join(bad[:3]))

    # --- write ---
    rank.to_csv(os.path.join(out, "ranking.csv"), index=False)
    cells.to_csv(os.path.join(out, "recoveries.csv"), index=False)
    anchors.to_csv(os.path.join(out, "anchors.csv"), index=False)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(jsonable(settings_dict(s7)), f, indent=2)
    pd.DataFrame(ck.rows).to_csv(os.path.join(out, "checks.csv"),
                                 index=False)
    minutes = (time.time() - T0) / 60
    write_report(out, ck, rank, addv, cells, crn, mf, ex, s7, started,
                 minutes, batch_diff)
    with open(os.path.join(out, "run_meta.json"), "w") as f:
        json.dump(jsonable({
            "started": started, "minutes": minutes,
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "output_dir": out,
            "input_sha256": hashes,
            "ranking": rank.to_dict("records"),
            "additivity": addv.to_dict("records"),
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


def summary_sentences(rank, addv, batch_diff):
    p = rank[rank["effect"] == "planning"].sort_values("rank")
    p = p.assign(kind=p["recovery"].map(classify))
    big = p[p["kind"] == "costs"].head(TOP_K)
    small = p[p["kind"] == "nothing"]
    helps = p[p["kind"] == "raises"]
    same, order_txt = order_comparison(rank, batch_diff)
    a = addv.set_index("effect")

    def join(names):
        names = list(names)
        if not names:
            return ""
        return names[0] if len(names) == 1 else \
            ", ".join(names[:-1]) + " and " + names[-1]
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
           f" when switched off); a likely reason, not tested here, is that "
           f"it offsets the shared preference for the duller image, pulling "
           f"those people's weights back towards zero, where each choice "
           f"carries more information. " if len(helps) else "")
        + order_txt + " "
        + ("The six costs add up to the total drop approximately."
           if not a["interacting"].any() else
           f"The six separate costs add up to only "
           f"{a.loc['planning', 'ratio']:.0%} (planning) and "
           f"{a.loc['half shift', 'ratio']:.0%} (half shift) of the total "
           f"drop: the nuisances compound, so together they cost more than "
           f"the sum of their separate costs (section 4)."))
    lo = big["recovery"].min() if len(big) else 0
    hi = big["recovery"].max() if len(big) else 0
    supp = (
        f"Most of the power lost to the assumed nuisances came from "
        f"{join(big['label']) or 'none of them'}: switching any one of these "
        f"off restored about {lo:.2f} to {hi:.2f} of the "
        f"{a.loc['planning', 'total_drop']:.2f} lost under the planning "
        f"scenario (average over block sizes, sample sizes of 10 to 20 and "
        f"both features)"
        + ("; the nuisances also compound, so together they cost more than "
           "the sum of these separate costs" if a["interacting"].any()
           else "")
        + "."
        + (f" {join(small['label']).capitalize()} cost almost no power"
           if len(small) else "")
        + (f", and {join(helps['label'])} slightly raised it"
           if len(helps) and len(small) else
           (f" {join(helps['label']).capitalize()} slightly raised it"
            if len(helps) else ""))
        + ("." if len(small) or len(helps) else ""))
    return plain, supp, same


def write_report(out, ck, rank, addv, cells, crn, mf, ex, s7, started,
                 minutes, batch_diff):
    L = []
    w = L.append
    plain, supp, same = summary_sentences(rank, addv, batch_diff)
    w("WHICH NUISANCES COST THE PRIMARY TEST POWER  (nuisance_power_v5.py)")
    w("=" * 78)
    w(f"run       {started}  ({minutes:.1f} min on {os.cpu_count()} "
      "cores)")
    w(f"output    {out}")
    w(f"checks    {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL "
      "(section 6)")
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
    w("  Suggested sentence for the supplement:")
    w(wrap('"' + supp + '"', indent="    "))
    w("")
    w("2. WHAT WAS DONE")
    w("-" * 78)
    w(wrap("Paired t-test (the registered primary test), power only, "
           "planning and half shift, block sizes 150 and 180, N = 10, 15, "
           "20, both features; pair-specific preferences off. Eight "
           "conditions: all nuisances at script 7's defaults; each of the "
           "six switched off on its own (to script 7's 'off' value); no "
           "nuisances. Power returned by a nuisance = power with it off "
           "minus power with all defaults. Ranked by the unweighted average "
           f"over the 12 cells. Script 7's person model, pools of "
           f"{s7.POOL_PEOPLE:,} people, {s7.N_STUDIES:,} studies per cell, "
           f"SE from {s7.N_BATCHES} sub-pools."))
    w(wrap("Common random numbers: within an effect and block size, all "
           "eight conditions use the same random draws and the same study "
           "draws, so each recovery is a paired difference and its SE is "
           "that of the difference. 'Scene dependence off' is SD x "
           f"{SCENE_OFF_MULT:g} rather than exactly 0, because script 7 "
           "draws the scene deviations only when they are on and a 0 would "
           "shift every later draw; at x 1e-9 the deviations are numerically "
           "nothing. Changing the second-block lapse changes catch failures "
           "and so who is excluded: in the block-order and no-nuisance "
           "conditions some studies sample different people (section 5), "
           "which makes those two paired differences somewhat noisier."))
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
    w("  At N = 20 (per block size and feature, planning):")
    q = cells[(cells["effect"] == "planning") & (cells["N"] == 20)]
    cols = ["size", "feature"] + [f"recovery: {k}" for k, _, _ in NUISANCES] \
        + ["total drop"]
    w("  " + q[cols].rename(columns=lambda c: c.replace("recovery: ", ""))
      .to_string(index=False, float_format=lambda x: f"{x:+.3f}")
      .replace("\n", "\n  "))
    w("")
    w("4. DO THE SIX ADD UP?")
    w("-" * 78)
    w(wrap("Total drop = power with no nuisances minus power with all "
           "defaults. If the nuisances acted independently, the six "
           "recoveries would sum to it. 'Interacting' if the sum differs "
           f"from the total by more than {INTERACT_SE:g} SE and by more "
           f"than {INTERACT_MIN}."))
    for r in addv.itertuples():
        w(f"  {r.effect:11s} total drop {r.total_drop:.4f} (SE "
          f"{r.total_drop_se:.4f}); sum of the six {r.sum_of_six:.4f} "
          f"(SE {r.sum_se:.4f});")
        w(f"  {'':11s} sum minus total {r.sum_minus_total:+.4f} (SE "
          f"{r.sum_minus_total_se:.4f}); ratio {r.ratio:.2f} -> "
          f"{'INTERACTING' if r.interacting else 'approximately additive'}")
    w(wrap("A ratio above 1 means switching nuisances off one at a time "
           "returns more power in total than switching them all off "
           "together (they overlap); below 1, that they compound (each costs "
           "more in the presence of the others)."))
    w("")
    w("5. COMMON DRAWS: HOW CLOSE THE MATCHING IS")
    w("-" * 78)
    pc = crn[crn["N"].isna()].groupby("condition")["exclusion_differs"] \
        .mean()
    sp = crn[crn["N"].notna()].groupby("condition")["studies_same_people"] \
        .mean()
    w("  condition        people excluded differently   studies with the "
      "same people")
    for cond in pc.index:
        w(f"  {cond:16s} {pc[cond]:26.2%} {sp.get(cond, np.nan):30.1%}")
    w("")
    w("6. CHECKS")
    w("-" * 78)
    for r in ck.rows:
        w(f"  {r['status']}  [{r['section']}] {r['check']}")
        if r["detail"]:
            w(f"          {r['detail']}")
    w("")
    w("  Anchors against script 7 (paired t-test):")
    w("  " + mf[["effect", "anchor", "size", "N", "feature", "this_script",
                 "script7", "z"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}").replace("\n",
                                                                "\n  "))
    w("")
    w("7. RUN TIME")
    w("-" * 78)
    w(f"  This run: {minutes:.1f} min on {os.cpu_count()} cores (36 pools of "
      f"{s7.POOL_PEOPLE:,} people, each")
    w("  fitted on both blocks, plus the studies). Expect about 5 minutes on "
      "a similar")
    w("  machine; time scales with the number of pools.")
    w("")
    w("8. FILES")
    w("-" * 78)
    for fn in sorted(set(os.listdir(out)) | {"report.txt",
                                             "run_meta.json"}):
        w(f"  {fn}")
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
