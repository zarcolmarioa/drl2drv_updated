"""
registration_tables_v5.py  --  v5 power analysis, script 9: the registered
                               numbers, tables, figures and decisions

WHAT THIS DOES
    Runs NO simulation. It reads the output folders of scripts 6, 7 and 8,
    checks that they passed every check and that they belong together
    (each later run recorded the SHA-256 of the files it read from the
    earlier one), recomputes the choice of primary test from script 7's
    results under the original rule and under the amended rule, and
    writes, for both block sizes:

      tables     thresholds per set and the chosen per-person model;
                 the primary test (paired t-test) with the Wilcoxon as
                 robustness check: power and false-positive rates under
                 planning and half shift, and the main sensitivity rows;
                 the paired test with pair-specific preferences;
                 the secondary sign test within each block;
                 C1 and the attention check; threats no test can remove;
                 the per-block attention measures
      figures    the v5 counterparts of the v4 preregistration's
                 detection, power and C1 figures, plus pair-specific
                 preferences; plain labels only
      text       registration_summary.txt: what is registered and why,
                 with the original rule, its amendment and the reason;
                 report.txt: what this script did, every check PASS or
                 FAIL, and the tables

PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES
    Every power and false-positive rate here comes from simulated people
    whose preferences, lapse rates and nuisances were chosen for planning.
    No image-choice data exist. The summary and every table say so.

REUSED
    Nothing is imported: the inputs are the CSV / JSON outputs of scripts
    6-8. Style follows registration_table_v4.py (provenance with SHA-256
    of every file read, checks first, Okabe-Ito colours). The rule
    computation repeats script 7's apply_rule on its paired_power.csv and
    is checked against what script 7 recorded.

INPUTS (relative to PROJECT_ROOT; folder names are settings)
    THRESHOLDS_RUN_DIR (script 6), PAIRED_RUN_DIR (script 7),
    C1_RUN_DIR (script 8)

OUTPUTS (one new folder per run under OUTPUT_ROOT; never overwritten)
    registration_summary.txt, report.txt, checks.csv, tables.txt
    table_thresholds.csv, table_primary.csv, table_sensitivity.csv,
    table_pair_preferences.csv, table_secondary_sign.csv, table_c1.csv,
    table_attention_check.csv, table_threats.csv, table_block_measures.csv
    figure_detection.png, figure_primary_power.png, figure_c1.png,
    figure_pair_preferences.png
    settings.json, run_meta.json

REQUIRES  Python 3.9+, numpy, pandas, scipy, matplotlib
RUN       python registration_tables_v5.py        (seconds)
"""

# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = None          # None = the folder this script is in
THRESHOLDS_RUN_DIR = ("data/power_v5/thresholds_v5/sizes150-180_choosers"
                      "20000_people10000_side0.25-1.5-popsd0.25_w0.5-0.7"
                      "_lapse0.1_seed42__s02be6391")
PAIRED_RUN_DIR = ("data/power_v5/paired_v5/sizes150-180_N10-20_model-side"
                  "_pool50000_studies10000_mixed2500_seed42__se1c01101")
C1_RUN_DIR = ("data/power_v5/c1_v5/sizes150-180_report10-20_comparison"
              "20-30_strategy0.25x0.5_pool50000_studies10000_seed42"
              "__s2ab231a6")
OUTPUT_ROOT = "data/power_v5/registration_v5"
FIG_DPI = 200

BLOCK_SIZES = [150, 180]
TABLE_N = [10, 15, 20]

# --- What is registered (decisions taken by the researcher and PI) ---
REGISTERED_MODEL = "side"            # script 6's recommendation
PRIMARY_TEST = "t"                   # paired t-test
ROBUSTNESS_TEST = "wilcoxon"         # exact Wilcoxon signed-rank
MIXED_MODEL_REGISTERED = False
DECISION_DATE = "2026-11-06"

# The rule for choosing the primary test: as fixed before script 7 ran,
# and as amended after it ran. Recorded verbatim in the summary.
ORIGINAL_RULE = (
    "A test qualifies if, in every rule row (no nuisance, each default "
    "nuisance alone, all defaults together; no effect), at N = 10, 15 and "
    "20, both block sizes and both features, its false-positive rate is "
    "at most 0.05 + 2 SE (that cell's Monte Carlo SE). Among qualifying "
    "tests, the one with the higher mean power over planning and half "
    "shift (all nuisances at their defaults; same N, sizes and features) "
    "wins; within 0.02 of each other, the Wilcoxon test.")
AMENDED_RULE = (
    "As the original rule, except that a test qualifies if no more of its "
    "rule cells exceed 0.05 + 2 SE than chance gives in 95% of runs for a "
    "test whose false-positive rate is exactly 0.05 (at most 5 of 96 "
    "cells).")
AMENDMENT_REASON = (
    "The original rule asks every one of 96 cells per test to stay below "
    "a limit set at 2 Monte Carlo SE. Each cell exceeds that limit with "
    "probability about 0.023 even for a perfectly calibrated test, so "
    "about 2.2 exceedances are expected by chance alone and an exactly "
    "calibrated test fails the rule in most runs. In script 7 each test "
    "exceeded the limit in exactly one cell, the same cell for both (block "
    "order, 180 trials, N = 20, contrast), by margins below 0.001 (t-test "
    "0.0579 against 0.0576; Wilcoxon 0.0567 against 0.0567), with mean "
    "false-positive rates of 0.050 (t) and 0.047 (Wilcoxon) over the rule "
    "rows. The amendment keeps the intent (no test whose false positives "
    "are inflated beyond chance qualifies) and removes the multiple-"
    "comparison artefact. It was made after script 7 had run, by the "
    "researcher, and is recorded here as such.")
RULE_K_SE = 2.0
RULE_CHANCE_QUANTILE = 0.95

# Script 6: how its side-model thresholds were computed (recorded in the
# summary).
THRESHOLD_NOTE = (
    "Both models' thresholds come from the same 20,000 random choosers "
    "(the stream that produced design.json), so the two models are "
    "compared without the Monte Carlo error of two separate threshold "
    "draws. In script 6's first run the side model's thresholds came "
    "from separate choosers; the detection-loss criterion (b) then failed "
    "on threshold noise (0.014, with a Monte Carlo SE of about 0.009) "
    "and the rule picked the two-weight model. Script 6 was changed to "
    "use the same choosers before its results were accepted; the change "
    "flipped the recommendation, and the researcher accepted it.")

# Script 8: the "realistic" rows and the expectation block.
REALISTIC_ROWS_NOTE = (
    "Script 8's 'realistic' C1 rows add script 7's default nuisances, "
    "except that script 7's 'edited looks odd' quarter (-0.5 on both "
    "features, against the more edited image) is replaced by v4's C1 "
    "strategy: a quarter of each group pick the more heavily edited-"
    "looking image (+0.5 on both features in every block). The two use "
    "the same slot of the person model, so they cannot both be on.")
EXPECTATION_ROLE = (
    "The comparison group's expectation block is a side benchmark. It "
    "decides nothing: it is used only in two exploratory, descriptive "
    "analyses, and no power simulation was run for them.")
EXPLORATORY_ANALYSES = [
    "Within the comparison group: expectation-block weights against the "
    "same people's ordinary-block weights (paired, per feature). Asks "
    "whether people who never had derealization expect it to look faded.",
    "Across groups: the report group's derealization-block weights against "
    "the comparison group's expectation-block weights (the v4 version of "
    "C1), reported beside C1.",
]
PAIR_SD_NOTE = (
    "Pair-specific preferences (a person's preference within a specific "
    "pair, repeated with the pair) are OFF in the defaults. Power is "
    "reported at SD 1.0, 1.5 and 2.0 logit; the strength will be "
    "calibrated from the agreement on repeated trials in the pilot "
    "(agreement in the simulation is shown beside each SD).")

# Words that must not appear in figure text (internal code names).
FORBIDDEN_FIGURE_TOKENS = [r"\bsat\b", r"\bcon\b", r"\bfog\b", r"\bder\b",
                           r"\bord\b", r"_", r"\bpred\b", r"\best\b",
                           r"\bo[1-4]\b", r"\bnuis\b", r"\bDPDR\b"]

# ============================================================
# Code
# ============================================================

import hashlib
import json
import os
import platform
import re
import time

import numpy as np
import pandas as pd
from scipy import stats

T0 = time.time()
FEATS = ("contrast", "saturation")
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
             "#56B4E9", "#000000"]


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


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
    "PROJECT_ROOT", "THRESHOLDS_RUN_DIR", "PAIRED_RUN_DIR", "C1_RUN_DIR",
    "OUTPUT_ROOT", "FIG_DPI", "BLOCK_SIZES", "TABLE_N", "REGISTERED_MODEL",
    "PRIMARY_TEST", "ROBUSTNESS_TEST", "MIXED_MODEL_REGISTERED",
    "DECISION_DATE", "ORIGINAL_RULE", "AMENDED_RULE", "AMENDMENT_REASON",
    "RULE_K_SE", "RULE_CHANCE_QUANTILE", "THRESHOLD_NOTE",
    "REALISTIC_ROWS_NOTE", "EXPECTATION_ROLE", "EXPLORATORY_ANALYSES",
    "PAIR_SD_NOTE", "FORBIDDEN_FIGURE_TOKENS",
]


def settings_dict():
    g = globals()
    return {k: g[k] for k in SETTING_NAMES}


def resolve_output_dir():
    h = hashlib.sha256(json.dumps(settings_dict(), sort_keys=True,
                                  default=str).encode()).hexdigest()[:8]
    name = (f"sizes{'-'.join(map(str, BLOCK_SIZES))}"
            f"_model-{REGISTERED_MODEL}_primary-{PRIMARY_TEST}__s{h}")
    base = root_path(OUTPUT_ROOT, name)
    out, k = base, 1
    while os.path.exists(out):
        k += 1
        out = f"{base}_run{k}"
    os.makedirs(out)
    return out


# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------

class Run:
    """One earlier script's output folder: every file read is hashed."""

    def __init__(self, label, rel, ck, hashes):
        self.label, self.dir = label, root_path(rel)
        self.ck, self.hashes = ck, hashes
        ok = os.path.isdir(self.dir)
        ck.add("inputs", f"{label}: output folder exists", ok, rel)
        if not ok:
            raise SystemExit(f"missing {self.dir}")
        c = self.csv("checks.csv")
        nf = int((c["status"] != "PASS").sum())
        ck.add("inputs", f"{label}: its run passed every check", nf == 0,
               f"{int((c['status'] == 'PASS').sum())} PASS, {nf} FAIL")
        self.meta = self.json("run_meta.json")
        self.settings = self.json("settings.json")

    def path(self, fn):
        p = os.path.join(self.dir, fn)
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        self.hashes[f"{self.label}/{fn}"] = sha256_of(p)
        return p

    def csv(self, fn, **kw):
        return pd.read_csv(self.path(fn), **kw)

    def json(self, fn):
        with open(self.path(fn)) as f:
            return json.load(f)


# ------------------------------------------------------------
# The rule for the primary test (script 7's apply_rule, both versions)
# ------------------------------------------------------------

def primary_rule(pw, s7set):
    alpha = s7set["ALPHA"]
    se0 = np.sqrt(alpha * (1 - alpha) / s7set["N_STUDIES"])
    rule_n = s7set["RULE_N"]
    nul = pw[(pw["role"] == "null (rule)") & pw["N"].isin(rule_n)].copy()
    nul["limit"] = alpha + RULE_K_SE * np.maximum(nul["se"], se0)
    pwr = pw[(pw["role"] == "power")
             & pw["effect"].isin(["planning", "half shift"])
             & pw["N"].isin(rule_n)]
    tests = sorted(pw["test"].unique())
    n_cells = len(nul) // len(tests)
    p1 = float(stats.norm.sf(RULE_K_SE))
    chance = int(stats.binom.ppf(RULE_CHANCE_QUANTILE, n_cells, p1))
    res = {"cells": n_cells, "p_cell": p1, "expected": n_cells * p1,
           "chance_max": chance, "tests": {}}
    for t in tests:
        q = nul[nul["test"] == t]
        ex = q[q["rate"] > q["limit"]]
        res["tests"][t] = {
            "n_exceed": len(ex),
            "exceed": [f"{r.scenario}, {r.size} trials, N = {r.N}, "
                       f"{r.feature}: {r.rate:.4f} > {r.limit:.4f}"
                       for r in ex.itertuples()],
            "mean_null": float(q["rate"].mean()),
            "power": float(pwr[pwr["test"] == t]["rate"].mean())}

    def choose(qual):
        ok = [t for t in tests if qual(res["tests"][t])]
        if not ok:
            return None, None
        if len(ok) == 1:
            return ok[0], None
        best = max(ok, key=lambda t: res["tests"][t]["power"])
        other = [t for t in ok if t != best][0]
        gap = res["tests"][best]["power"] - res["tests"][other]["power"]
        pref = s7set["RULE_PREFER"]
        return (pref if gap <= s7set["RULE_TIE"] and pref in ok
                else best), gap
    res["original_chosen"], res["original_gap"] = choose(
        lambda r: r["n_exceed"] == 0)
    res["amended_chosen"], res["amended_gap"] = choose(
        lambda r: r["n_exceed"] <= chance)
    for t in tests:
        r = res["tests"][t]
        r["original_qualifies"] = r["n_exceed"] == 0
        r["amended_qualifies"] = r["n_exceed"] <= chance
    return res


# ------------------------------------------------------------
# Tables
# ------------------------------------------------------------

TEST_LABEL = {"t": "paired t-test", "wilcoxon": "Wilcoxon signed-rank"}


def t_thresholds(thr):
    src = {"side": "same choosers as design.json",
           "two-weight": "design.json (registered)"}
    rows = []
    for size in BLOCK_SIZES:
        for s in ("A", "B"):
            for f in ("saturation", "contrast"):
                r = {"block size": size, "set": s, "feature": f}
                for model, lab in ((REGISTERED_MODEL, "registered"),
                                   ("two-weight", "two-weight (frozen "
                                    "in design.json)")):
                    if model == REGISTERED_MODEL and lab != "registered":
                        continue
                    q = thr[(thr["size"] == size) & (thr["set"] == s)
                            & (thr["feature"] == f) & (thr["model"] == model)
                            & (thr["source"] == src[model])].iloc[0]
                    r[f"{lab}: upper"] = round(q["upper"], 3)
                    r[f"{lab}: lower"] = round(q["lower"], 3)
                rows.append(r)
    return pd.DataFrame(rows)


def t_primary(pw):
    rows = []
    for size in BLOCK_SIZES:
        for N in sorted(pw["N"].unique()):
            r = {"block size": size, "N": N}
            for eff, lab in (("planning", "planning"),
                             ("half shift", "half shift")):
                for f in FEATS:
                    for t in (PRIMARY_TEST, ROBUSTNESS_TEST):
                        q = pw[(pw["role"] == "power") & (pw["effect"] == eff)
                               & (pw["size"] == size) & (pw["N"] == N)
                               & (pw["test"] == t) & (pw["feature"] == f)]
                        r[f"{lab}, {f}, {TEST_LABEL[t]}"] = q["rate"].iloc[0]
            for f in FEATS:
                for t in (PRIMARY_TEST, ROBUSTNESS_TEST):
                    q = pw[(pw["role"] == "null (rule)")
                           & (pw["scenario"] == "all nuisances (defaults)")
                           & (pw["size"] == size) & (pw["N"] == N)
                           & (pw["test"] == t) & (pw["feature"] == f)]
                    r[f"no effect, {f}, {TEST_LABEL[t]}"] = q["rate"].iloc[0]
            rows.append(r)
    return pd.DataFrame(rows)


def t_sensitivity(pw):
    rows = []

    def add(label, q):
        for size in BLOCK_SIZES:
            for N in TABLE_N:
                r = {"row": label, "block size": size, "N": N}
                for f in FEATS:
                    for t in (PRIMARY_TEST, ROBUSTNESS_TEST):
                        x = q[(q["size"] == size) & (q["N"] == N)
                              & (q["feature"] == f) & (q["test"] == t)]
                        r[f"{f}, {TEST_LABEL[t]}"] = (
                            x["rate"].max() if len(x) else np.nan)
                rows.append(r)

    for eff in ("a quarter respond", "a quarter respond, half shift",
                "all respond", "all respond, half shift"):
        add(f"{eff} (nuisances at defaults)",
            pw[(pw["role"] == "power") & (pw["effect"] == eff)])
    for eff in ("planning", "half shift"):
        add(f"{eff}, no nuisances",
            pw[(pw["role"] == "power (no nuisances)") & (pw["effect"] == eff)])
    add("worst no-effect rate over the rule rows (default nuisances)",
        pw[pw["role"] == "null (rule)"])
    add("worst no-effect rate over the sensitivity rows (ranges)",
        pw[pw["role"] == "null (sensitivity)"])
    return pd.DataFrame(rows)


def t_pair(pair, pw, bm):
    """Power with pair-specific preferences; SD 0 = the defaults (script
    7), with its agreement on repeats from script 7's pool means."""
    rows = []
    for size in BLOCK_SIZES:
        for eff in ("planning", "half shift"):
            base = pw[(pw["role"] == "power") & (pw["effect"] == eff)
                      & (pw["size"] == size)]
            for sd, src in [(0.0, base)] + [
                    (sd, pair[(pair["pair_sd"] == sd) & (pair["size"] == size)
                              & (pair["effect"] == eff)])
                    for sd in sorted(pair["pair_sd"].unique())]:
                for N in TABLE_N:
                    r = {"block size": size, "effect": eff,
                         "pair-specific SD": sd, "N": N}
                    if sd > 0:
                        a = src["agreement_O"].dropna()
                    else:
                        a = bm[(bm["size"] == size) & (bm["effect"] == eff)
                               & (bm["scenario"] == "all nuisances "
                                  "(defaults)")
                               & (bm["N"] == "pool (analysable people)")
                               ]["agree_O"].dropna()
                    r["agreement on repeats"] = (round(float(a.iloc[0]), 3)
                                                 if len(a) else np.nan)
                    for f in FEATS:
                        for t in (PRIMARY_TEST, ROBUSTNESS_TEST):
                            x = src[(src["N"] == N) & (src["feature"] == f)
                                    & (src["test"] == t)]
                            r[f"{f}, {TEST_LABEL[t]}"] = x["rate"].iloc[0]
                    rows.append(r)
    return pd.DataFrame(rows)


def t_secondary(sg):
    q = sg[sg["N"].isin(TABLE_N) & (
        ((sg["role"] == "power") & sg["effect"].isin(["planning",
                                                      "half shift"]))
        | ((sg["role"] == "null (rule)") & sg["scenario"].isin(
            ["no nuisances", "all nuisances (defaults)"])))]
    q = q.assign(row=np.where(q["effect"] == "no effect",
                              "no effect, " + q["scenario"],
                              q["effect"] + " (nuisances at defaults)"))
    piv = q.pivot_table(index=["row", "size", "N"],
                        columns=["block", "feature"], values="claim_rate",
                        sort=False).reset_index()
    piv.columns = [" ".join(c).strip() if isinstance(c, tuple) else c
                   for c in piv.columns]
    return piv.rename(columns={"size": "block size"})


def t_c1(c1):
    rows = []
    for sc in ("planning", "half shift", "no effect",
               "v4-style: saturation only (0.5)",
               "v4-style: contrast only (0.7)",
               "planning, script 7's default nuisances",
               "half shift, script 7's default nuisances",
               "no effect, script 7's default nuisances"):
        q = c1[(c1["scenario"] == sc) & c1["report_n"].isin(TABLE_N)]
        for r in q.itertuples():
            rows.append({
                "scenario": sc, "block size": r.size,
                "comparison group": r.comparison_n, "report group": r.report_n,
                "C1 supported, contrast": r.gated_contrast,
                "C1 supported, saturation": r.gated_saturation,
                "C1 supported, either": r.gated_either,
                "before the attention check, either": r.raw_either,
                "attention check fires": r.gate_fires,
                "secondary check, either": r.secondary_either})
    return pd.DataFrame(rows)


def t_attention(c1):
    q = c1[(c1["scenario"].str.contains("lapse report")
            | (c1["scenario"].isin(["no effect", "planning"])))
           & c1["report_n"].isin(TABLE_N)]
    out = q[["scenario", "size", "comparison_n", "report_n",
             "lapse_report", "lapse_comparison", "gate_fires",
             "gate_fires_catch", "gate_fires_rt", "raw_either",
             "gated_either"]].copy()
    return out.rename(columns={
        "size": "block size", "comparison_n": "comparison group",
        "report_n": "report group", "lapse_report": "mean lapse, report",
        "lapse_comparison": "mean lapse, comparison",
        "gate_fires": "check fires", "gate_fires_catch": "fires on catch",
        "gate_fires_rt": "fires on fast responses",
        "raw_either": "C1 claim before the check",
        "gated_either": "C1 supported after the check"})


def t_threats(pw):
    q = pw[(pw["role"] == "threat") & pw["N"].isin(TABLE_N)
           & (pw["test"] == PRIMARY_TEST)]
    piv = q.pivot_table(index=["scenario", "size", "N"], columns="feature",
                        values="rate", sort=False).reset_index()
    return piv.rename(columns={"size": "block size"})


def t_block(bm):
    q = bm[(bm["N"].isin([str(n) for n in TABLE_N]))
           & (bm["effect"] == "no effect")
           & (bm["scenario"].str.startswith("derealization lapse")
              | (bm["scenario"] == "no nuisances"))
           & ~bm["scenario"].str.contains("odd|duller")]
    q = q[["scenario", "size", "N", "consistency_flags", "catch_flags"]]
    return q.rename(columns={
        "size": "block size",
        "consistency_flags": "consistency flags the difference",
        "catch_flags": "catch failures flag the difference"})


# ------------------------------------------------------------
# Figures
# ------------------------------------------------------------

def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)


def _texts(fig):
    out = []
    for t in fig.findobj(lambda a: hasattr(a, "get_text")):
        try:
            s = t.get_text()
        except Exception:
            continue
        if s:
            out.append(s)
    return out


def save(fig, path, bad):
    """Save a figure after checking its text for internal code names."""
    for s in _texts(fig):
        for pat in FORBIDDEN_FIGURE_TOKENS:
            if re.search(pat, s):
                bad.append(f"{os.path.basename(path)}: '{s}'")
    fig.savefig(path, dpi=FIG_DPI)


def figures(det, pw, c1, pair, s6set, out, bad):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. detection
    fig, axes = plt.subplots(len(BLOCK_SIZES), 2, figsize=(9, 3.6 *
                             len(BLOCK_SIZES)), sharex=True, sharey=True,
                             squeeze=False)
    lapses = sorted(det["lapse"].unique())
    planning = dict(zip(("saturation", "contrast"),
                        s6set["PLANNING_WEIGHTS"]))
    for i, size in enumerate(BLOCK_SIZES):
        for j, (f, col) in enumerate((("contrast", "con_detected"),
                                      ("saturation", "sat_detected"))):
            ax = axes[i, j]
            for c, lam in enumerate(lapses):
                for s, ls in (("A", "-"), ("B", "--")):
                    q = det[(det["size"] == size) & (det["set"] == s)
                            & (det["lapse"] == lam)]
                    ax.plot(q["true_weight"], q[col], ls, marker="o", ms=3,
                            color=OKABE_ITO[c],
                            label=f"lapse rate {lam:g}, set {s}")
            ax.axvline(planning[f], color="#888888", lw=0.8, ls="-.")
            ax.axhline(0.025, color="#888888", lw=0.8, ls=":")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{f.capitalize()}, {size} trials per block",
                         fontsize=10)
            if i == len(BLOCK_SIZES) - 1:
                ax.set_xlabel("True weight (per corpus SD)")
            if j == 0:
                ax.set_ylabel("Probability a person's weight\nis detected "
                              "in the predicted direction")
            _style(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle("Detection of one person's preference (registered "
                 "per-person model)\ndash-dot: planning weight, an "
                 "assumption; dotted: 2.5% by chance", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, os.path.join(out, "figure_detection.png"), bad)
    plt.close(fig)

    # 2. primary test
    fig, axes = plt.subplots(len(BLOCK_SIZES), 2, figsize=(10, 3.9 *
                             len(BLOCK_SIZES)), sharey=True, squeeze=False)
    for i, size in enumerate(BLOCK_SIZES):
        for j, f in enumerate(FEATS):
            ax = axes[i, j]
            series = [("power", "planning", 0, "planning"),
                      ("power", "half shift", 1, "half shift"),
                      ("null (rule)", "no effect", 6, "no effect")]
            for role, eff, c, lab in series:
                for t, ls, lw in ((PRIMARY_TEST, "-", 1.6),
                                  (ROBUSTNESS_TEST, "--", 1.0)):
                    q = pw[(pw["role"] == role) & (pw["effect"] == eff)
                           & (pw["size"] == size) & (pw["feature"] == f)
                           & (pw["test"] == t)]
                    if eff == "no effect":
                        q = q[q["scenario"] == "all nuisances (defaults)"]
                    ax.plot(q["N"], q["rate"], ls, lw=lw, color=OKABE_ITO[c],
                            marker="o", ms=3,
                            label=f"{lab}, {TEST_LABEL[t]}")
            ax.axhline(0.05, color="#888888", lw=0.8, ls=":")
            ax.set_ylim(0, 1)
            ax.set_title(f"{f.capitalize()} ({'H1' if f == 'contrast' else 'H2'}"
                         f"), {size} trials per block", fontsize=10)
            ax.set_xlabel("Report-group participants (analysable)")
            if j == 0:
                ax.set_ylabel("Probability the one-sided test\nis "
                              "significant")
            _style(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle("Primary test: derealization-block weight minus ordinary-"
                 "block weight (paired t-test;\nWilcoxon as robustness "
                 "check). Nuisances at their defaults; all values are "
                 "planning assumptions", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, os.path.join(out, "figure_primary_power.png"), bad)
    plt.close(fig)

    # 3. C1
    fig, axes = plt.subplots(len(BLOCK_SIZES), 2, figsize=(10, 3.9 *
                             len(BLOCK_SIZES)), sharey=True, squeeze=False)
    for i, size in enumerate(BLOCK_SIZES):
        for j, f in enumerate(FEATS):
            ax = axes[i, j]
            for k, nc in enumerate(sorted(c1["comparison_n"].unique())):
                for sc, ls in (("planning", "-"), ("no effect", ":")):
                    q = c1[(c1["size"] == size) & (c1["scenario"] == sc)
                           & (c1["comparison_n"] == nc)]
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
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.suptitle("C1: report group's derealization block against the "
                 "comparison group's ordinary block\n(planning values and "
                 "the attention model are assumptions)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, os.path.join(out, "figure_c1.png"), bad)
    plt.close(fig)

    # 4. pair-specific preferences
    fig, axes = plt.subplots(len(BLOCK_SIZES), 2, figsize=(10, 3.9 *
                             len(BLOCK_SIZES)), sharey=True, squeeze=False)
    sds = [0.0] + sorted(pair["pair_sd"].unique())
    for i, size in enumerate(BLOCK_SIZES):
        for j, f in enumerate(FEATS):
            ax = axes[i, j]
            for c, sd in enumerate(sds):
                for eff, ls in (("planning", "-"), ("half shift", "--")):
                    if sd == 0:
                        q = pw[(pw["role"] == "power") & (pw["effect"] == eff)]
                    else:
                        q = pair[(pair["pair_sd"] == sd)
                                 & (pair["effect"] == eff)]
                    q = q[(q["size"] == size) & (q["feature"] == f)
                          & (q["test"] == PRIMARY_TEST)].sort_values("N")
                    lab = ("none (default)" if sd == 0 else f"SD {sd:g}")
                    ax.plot(q["N"], q["rate"], ls, color=OKABE_ITO[c],
                            marker="o", ms=3,
                            label=f"{eff}, pair-specific preferences {lab}")
            ax.set_ylim(0, 1)
            ax.set_title(f"{f.capitalize()}, {size} trials per block",
                         fontsize=10)
            ax.set_xlabel("Report-group participants (analysable)")
            if j == 0:
                ax.set_ylabel("Probability the paired t-test\nis "
                              "significant")
            _style(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.suptitle("Primary test with pair-specific preferences (other "
                 "nuisances at their defaults);\nthe strength will be "
                 "calibrated from agreement on repeated pilot trials",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, os.path.join(out, "figure_pair_preferences.png"), bad)
    plt.close(fig)


# ------------------------------------------------------------
# Text
# ------------------------------------------------------------

def tab(df, nd=3):
    return "  " + df.to_string(index=False, float_format=lambda x:
                               f"{x:.{nd}f}").replace("\n", "\n  ")


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


def pick(pw, role, eff, size, N, f, t, scen=None):
    q = pw[(pw["role"] == role) & (pw["effect"] == eff) & (pw["size"] == size)
           & (pw["N"] == N) & (pw["feature"] == f) & (pw["test"] == t)]
    if scen:
        q = q[q["scenario"] == scen]
    return float(q["rate"].iloc[0])


def write_summary(out, rr, s6meta, s7meta, thr_t, pw, c1, pair, bm, runs,
                  ck):
    L = []
    w = L.append
    w("REGISTRATION SUMMARY: v5 POWER ANALYSIS  (registration_tables_v5.py)")
    w("=" * 78)
    w(f"Written {time.strftime('%Y-%m-%d %H:%M')}; decisions dated "
      f"{DECISION_DATE}. Checks: {len(ck.rows) - ck.n_fail} PASS, "
      f"{ck.n_fail} FAIL (report.txt).")
    w("")
    w(wrap("PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES. Every power "
           "and false-positive rate below comes from simulated people whose "
           "preferences (responders' shift of 0.5 saturation and 0.7 "
           "contrast per corpus SD, half of the report group responding), "
           "taste spread, lapse rates and nuisances were chosen for "
           "planning. No image-choice data exist yet.", indent=""))
    w(wrap("Both block sizes (150 and 180 analysed trials per block) are "
           "reported; the size is chosen after the timed pilot.", indent=""))
    w("")
    w("1. WHAT IS REGISTERED")
    w("-" * 78)
    items = [
        f"Per-person model: Firth logistic regression without intercept, "
        f"fitted per block on the analysed trials, with a saturation weight, "
        f"a contrast weight and a side term (+1 when image A is on the left, "
        f"-1 on the right). The side term is a nuisance term; its estimate "
        f"is never reported as a result.",
        "Detection thresholds: per set and block size, the 2.5% tails of "
        "20,000 random choosers' weights under the registered model (table "
        "1). They replace the two-weight thresholds frozen in design.json.",
        f"PRIMARY test (H1 contrast, H2 saturation), in the report group: "
        f"one-sided paired t-test on the within-person difference "
        f"(derealization-block weight minus ordinary-block weight), alpha "
        f"0.05 each, uncorrected. The exact Wilcoxon signed-rank test on the "
        f"same differences is reported as a robustness check.",
        "The mixed-effects logistic model was evaluated and is NOT "
        "registered (section 4).",
        "SECONDARY: detected counts and the sign test within each block "
        "(v4's rule: d >= 5 and the exact one-sided binomial), with the "
        "registered thresholds.",
        "C1: section 7. The comparison group's expectation block: section 8.",
    ]
    for it in items:
        w(wrap("* " + it))
    w("")
    w("2. THE PER-PERSON MODEL AND ITS THRESHOLDS (script 6)")
    w("-" * 78)
    r6 = s6meta["rule"]
    w(wrap("Rule fixed before the run: register the side model only if (a) "
           "a side bias of 0.5 logit moves the two-weight model's false-"
           "detection rate above 0.035 in either tail or a weight by more "
           "than 0.05, and (b) the side model loses at most 0.01 of "
           "detection at the planning weights."))
    w(wrap(f"Outcome: (a) holds: worst tail rate {r6['tail']:.4f} "
           f"({r6['tail_where']}); worst weight shift {r6['shift']:.4f} "
           f"({r6['shift_where']}); the weight shift is what decides (a), "
           f"the tail rate is within Monte Carlo error of the limit. (b) "
           f"holds: largest detection loss {r6['loss']:.4f} "
           f"({r6['loss_where']}). Chosen: {s6meta['chosen_model']} model."))
    w(wrap("How the thresholds were computed: " + THRESHOLD_NOTE))
    w("")
    w(tab(thr_t))
    w("")
    w("3. THE PRIMARY TEST: HOW IT WAS CHOSEN (script 7)")
    w("-" * 78)
    w("  Original rule (fixed before script 7 ran):")
    w(wrap(ORIGINAL_RULE, indent="    "))
    w("  Amendment (made after script 7 ran):")
    w(wrap(AMENDED_RULE, indent="    "))
    w("  Reason:")
    w(wrap(AMENDMENT_REASON, indent="    "))
    w(f"  Recomputed from script 7's results ({rr['cells']} rule cells per "
      f"test; chance of one cell over the")
    w(f"  limit for an exactly calibrated test {rr['p_cell']:.4f}; expected "
      f"{rr['expected']:.1f}; 95% of runs give at most {rr['chance_max']}):")
    for t, r in rr["tests"].items():
        w(f"    {TEST_LABEL[t]:22s} cells over the limit {r['n_exceed']}; "
          f"mean false-positive rate {r['mean_null']:.4f}; mean power "
          f"{r['power']:.4f}")
        w(f"    {'':22s} original rule: "
          f"{'qualifies' if r['original_qualifies'] else 'does not qualify'}"
          f"; amended rule: "
          f"{'qualifies' if r['amended_qualifies'] else 'does not qualify'}")
        for c in r["exceed"]:
            w(f"    {'':22s} over the limit: {c}")
    w(f"  Original rule chooses: {rr['original_chosen'] or 'no test'}. "
      f"Amended rule chooses: "
      f"{TEST_LABEL.get(rr['amended_chosen'], 'no test')}"
      + (f" (power gap {rr['amended_gap']:+.4f}, above the tie margin)"
         if rr["amended_gap"] is not None else "") + ".")
    w(f"  REGISTERED: {TEST_LABEL[PRIMARY_TEST]} (primary); "
      f"{TEST_LABEL[ROBUSTNESS_TEST]} (robustness check).")
    w("")
    w("  Registered power and false-positive rates, paired t-test, nuisances "
      "at their defaults")
    w("  [ASSUMPTIONS] (Wilcoxon in table_primary.csv):")
    rows = []
    for size in BLOCK_SIZES:
        for N in TABLE_N:
            rows.append({
                "block size": size, "N": N,
                "planning contrast": pick(pw, "power", "planning", size, N,
                                          "contrast", PRIMARY_TEST),
                "planning saturation": pick(pw, "power", "planning", size, N,
                                            "saturation", PRIMARY_TEST),
                "half shift contrast": pick(pw, "power", "half shift", size,
                                            N, "contrast", PRIMARY_TEST),
                "half shift saturation": pick(pw, "power", "half shift",
                                              size, N, "saturation",
                                              PRIMARY_TEST),
                "no effect contrast": pick(
                    pw, "null (rule)", "no effect", size, N, "contrast",
                    PRIMARY_TEST, "all nuisances (defaults)"),
                "no effect saturation": pick(
                    pw, "null (rule)", "no effect", size, N, "saturation",
                    PRIMARY_TEST, "all nuisances (defaults)")})
    w(tab(pd.DataFrame(rows)))
    w("")
    w("4. THE MIXED-EFFECTS LOGISTIC MODEL: NOT REGISTERED")
    w("-" * 78)
    for line in s7meta["mixed_verdict"]:
        w(wrap(line))
    w(wrap("Criterion fixed before the run: at least 0.05 more power than "
           "the chosen test under half shift at N = 15, both sizes and both "
           "features, with false positives controlled."))
    w("")
    w("5. DEFAULTS AND SENSITIVITY")
    w("-" * 78)
    w(wrap("The nuisance defaults are unchanged from script 7 (taste "
           "stability 0.7; person x scene preferences, SD equal to the taste "
           "SD, stability 0.7; side bias SD 0.25 logit; 'memory makes it "
           "duller' +0.25/+0.35 in everyone; 'edited looks odd' in a quarter, "
           "-0.5; second block lapse +0.05 and preferences x0.9; equal "
           "lapse rates in the two episodes)."))
    w(wrap(PAIR_SD_NOTE))
    ps = t_pair(pair, pw, bm)
    ps = ps[(ps["N"] == 20)][["block size", "effect", "pair-specific SD",
                               "agreement on repeats",
                               f"contrast, {TEST_LABEL[PRIMARY_TEST]}",
                               f"saturation, {TEST_LABEL[PRIMARY_TEST]}"]]
    w("  Paired t-test power at N = 20 [ASSUMPTIONS]:")
    w(tab(ps))
    w("")
    w("6. THREATS NO TEST CAN REMOVE")
    w("-" * 78)
    w(wrap("A bias present in both blocks cancels in the paired difference "
           "only if both blocks express it equally. More lapses in the "
           "derealization block (a sensitivity row, kept out of the test "
           "choice) shrink its weights: with 'edited looks odd' this "
           "inflates false positives, with 'memory makes it duller' it makes "
           "the test conservative. A bias stronger in one episode cannot be "
           "told from an effect. Paired t-test, no effect, N = 15:"))
    th = pw[(pw["role"] == "threat") & (pw["N"] == 15)
            & (pw["test"] == PRIMARY_TEST)]
    piv = th.pivot_table(index="scenario", columns=["size", "feature"],
                         values="rate", sort=False)
    w("  " + piv.to_string(float_format=lambda x: f"{x:.3f}")
      .replace("\n", "\n  "))
    w("")
    w("7. C1 (script 8)")
    w("-" * 78)
    w(wrap("C1 compares the report group's derealization-block weights with "
           "the comparison group's ORDINARY-block weights (as the PI "
           "recommended): one-sided Mann-Whitney per feature, report group "
           "higher, Holm across the two features. First, the attention "
           "check: the comparison group against the report group on catch "
           "failures over the session and on the share of responses under "
           "300 ms (one-sided Mann-Whitney, Holm); if the comparison group "
           "looks less attentive, C1 is inconclusive. Secondary check: the "
           "two groups' ORDINARY-block weights, same test. Comparison group "
           "below 20: C1 descriptive only."))
    w(wrap("Scenario (v4's C1 scenario): half the report group respond "
           "(0.5 / 0.7 per SD, both features); a quarter of each group pick "
           "the more edited-looking image whatever the episode; mean lapse "
           "0.1. " + REALISTIC_ROWS_NOTE))
    rows = []
    for size in BLOCK_SIZES:
        for nc in sorted(c1["comparison_n"].unique()):
            for N in TABLE_N:
                g = lambda sc, col: float(c1[(c1["scenario"] == sc)
                                             & (c1["size"] == size)
                                             & (c1["comparison_n"] == nc)
                                             & (c1["report_n"] == N)][col]
                                          .iloc[0])
                rows.append({"block size": size, "comparison": nc,
                             "report": N,
                             "planning contrast": g("planning",
                                                    "gated_contrast"),
                             "planning saturation": g("planning",
                                                      "gated_saturation"),
                             "planning either": g("planning",
                                                  "gated_either"),
                             "no effect either": g("no effect",
                                                   "raw_either"),
                             "check fires, equal": g("no effect",
                                                     "gate_fires")})
    w("  C1 supported after the attention check [ASSUMPTIONS]:")
    w(tab(pd.DataFrame(rows)))
    lp = c1[c1["scenario"].str.contains("lapse report 0.1 / comparison")
            & c1["scenario"].str.startswith("no effect")]
    w(wrap(f"When the comparison group lapses more (mean 0.15 or 0.2 against "
           f"0.1), the check fires in {lp['gate_fires'].min():.0%}-"
           f"{lp['gate_fires'].max():.0%} of studies; before the check, "
           f"false C1 claims then reach at most {lp['raw_either'].max():.3f}. "
           f"With an effect, a slightly less attentive comparison group "
           f"therefore mostly costs power (C1 inconclusive) rather than "
           f"producing false claims (table_attention_check.csv)."))
    w("")
    w("8. THE COMPARISON GROUP'S EXPECTATION BLOCK")
    w("-" * 78)
    w(wrap(EXPECTATION_ROLE))
    for i, a in enumerate(EXPLORATORY_ANALYSES, 1):
        w(wrap(f"Exploratory {i}: {a}", indent="    "))
    w("")
    w("9. PER-BLOCK ATTENTION MEASURES (candidate check, not registered "
      "here)")
    w("-" * 78)
    w(wrap("How often catch failures and consistency on repeated trials "
           "flag more lapses in the derealization block (one-sided, 0.05), "
           "no effect [ASSUMPTIONS]; consistency only works if people have "
           "pair-specific preferences:"))
    w(tab(t_block(bm)))
    w("")
    w("10. PROVENANCE")
    w("-" * 78)
    for label, run in runs.items():
        w(f"  {label}: {os.path.relpath(run.dir, root_path())}")
    w("  SHA-256 of every file read: run_meta.json of this folder.")
    with open(os.path.join(out, "registration_summary.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


def write_report(out, ck, tables, bad, meta):
    L = []
    w = L.append
    w("REGISTRATION TABLES  (registration_tables_v5.py, script 9)")
    w("=" * 78)
    w(f"run       {meta['started']}  ({meta['seconds']:.1f} s)")
    w(f"output    {out}")
    w("")
    w("1. WHAT WAS DONE")
    w("-" * 78)
    w(wrap("Read the outputs of scripts 6, 7 and 8 (no simulation), checked "
           "that each passed every check and that they chain together "
           "through the SHA-256 each run recorded for the files it read, "
           "recomputed the primary-test rule (original and amended) from "
           "script 7's results and compared it with what script 7 recorded, "
           "and wrote the tables, the figures and registration_summary.txt. "
           "Planning assumptions are assumptions, not estimates.", indent=""))
    w("")
    w("2. CHECKS")
    w("-" * 78)
    for r in ck.rows:
        w(f"  {r['status']}  [{r['section']}] {r['check']}")
        if r["detail"]:
            w(f"          {r['detail']}")
    w(f"\n  {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL")
    w("")
    w("3. TABLES (also as CSV; tables.txt has them all)")
    w("-" * 78)
    for name, df in tables.items():
        w(f"\n  {name}  ({len(df)} rows)")
    w("")
    w("4. FILES")
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
    runs = {"script 6 (thresholds_v5.py)": Run("script 6", THRESHOLDS_RUN_DIR,
                                               ck, hashes),
            "script 7 (simulate_power_paired_v5.py)": Run(
                "script 7", PAIRED_RUN_DIR, ck, hashes),
            "script 8 (simulate_c1_v5.py)": Run("script 8", C1_RUN_DIR, ck,
                                                hashes)}
    r6, r7, r8 = runs.values()
    out = resolve_output_dir()
    log(f"output folder {out}")

    thr = r6.csv("thresholds.csv")
    det = r6.csv("detection_curve.csv")
    pw = r7.csv("paired_power.csv")
    sg = r7.csv("secondary_sign.csv")
    bm = r7.csv("block_measures.csv", low_memory=False)
    c1 = r8.csv("c1.csv")
    pair = r8.csv("paired_pair_sd.csv")
    pair = pair[pair["source"] == "this script"].copy()   # SD 1.5 from 7
    p15 = r8.csv("paired_pair_sd.csv")
    p15 = p15[p15["source"] == "script 7"]
    pair = pd.concat([pair, p15], ignore_index=True)

    # --- the runs belong together ---
    ck.add("chain", "script 7 read the thresholds.csv read here",
           r7.meta["input_sha256"].get("thresholds.csv")
           == hashes["script 6/thresholds.csv"],
           hashes["script 6/thresholds.csv"][:16] + "...")
    ck.add("chain", "script 8 read the thresholds.csv read here",
           r8.meta["input_sha256"].get("thresholds.csv")
           == hashes["script 6/thresholds.csv"], "")
    ck.add("chain", "script 8 read script 7's paired_power.csv read here",
           r8.meta["input_sha256"].get("script 7 paired_power.csv")
           == hashes["script 7/paired_power.csv"],
           hashes["script 7/paired_power.csv"][:16] + "...")
    ck.add("chain", "scripts 7 and 8 used the same frozen lists",
           all(r7.meta["input_sha256"].get(k) == v
               for k, v in r8.meta["input_sha256"].items()
               if k.startswith("data/frozen_trial_lists_v5")),
           f"{sum(k.startswith('data/frozen') for k in r8.meta['input_sha256'])}"
           " files")
    ck.add("chain", "script 7's per-person model is the registered one",
           r7.settings["PERSON_MODEL"] == REGISTERED_MODEL,
           r7.settings["PERSON_MODEL"])

    # --- decisions against the earlier recommendations ---
    ck.add("decisions", "the registered per-person model is script 6's "
           "recommendation", r6.meta["chosen_model"] == REGISTERED_MODEL,
           f"script 6: {r6.meta['chosen_model']}; registered: "
           f"{REGISTERED_MODEL}")
    rr = primary_rule(pw, r7.settings)
    same = all(rr["tests"][t]["n_exceed"]
               == r7.meta["rule"]["tests"][t]["n_exceed"]
               for t in rr["tests"]) and (
        rr["original_chosen"] == r7.meta["rule"]["chosen"])
    ck.add("decisions", "the original rule, recomputed here, gives what "
           "script 7 recorded", same,
           f"cells over the limit {[(t, r['n_exceed']) for t, r in rr['tests'].items()]}"
           f"; chosen {rr['original_chosen']}")
    ck.add("decisions", "the amended rule chooses the registered primary "
           "test", rr["amended_chosen"] == PRIMARY_TEST,
           f"amended rule: {rr['amended_chosen']} (gap "
           f"{rr['amended_gap']:+.4f}); registered: {PRIMARY_TEST}"
           if rr["amended_gap"] is not None else f"{rr['amended_chosen']}")
    ck.add("decisions", "the mixed model is not registered, as script 7's "
           "verdict says",
           (not MIXED_MODEL_REGISTERED)
           == ("NOT worth" in r7.meta["mixed_verdict"][-1]),
           r7.meta["mixed_verdict"][-1])

    # --- tables ---
    tables = {
        "table_thresholds": t_thresholds(thr),
        "table_primary": t_primary(pw),
        "table_sensitivity": t_sensitivity(pw),
        "table_pair_preferences": t_pair(pair, pw, bm),
        "table_secondary_sign": t_secondary(sg),
        "table_c1": t_c1(c1),
        "table_attention_check": t_attention(c1),
        "table_threats": t_threats(pw),
        "table_block_measures": t_block(bm),
    }
    expect = {"table_thresholds": 2 * 2 * len(BLOCK_SIZES),
              "table_primary": len(BLOCK_SIZES) * pw["N"].nunique(),
              "table_c1": 8 * len(BLOCK_SIZES) * 2 * len(TABLE_N)}
    for name, df in tables.items():
        n_nan = int(df.select_dtypes("number").isna().sum().sum())
        ok = len(df) > 0 and n_nan == 0 and (name not in expect
                                             or len(df) == expect[name])
        ck.add("tables", f"{name}: complete", ok,
               f"{len(df)} rows, {n_nan} empty numeric cells"
               + (f" (expected {expect[name]} rows)" if name in expect
                  else ""))
        df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    with open(os.path.join(out, "tables.txt"), "w") as f:
        f.write("REGISTRATION TABLES (planning assumptions are assumptions, "
                "not estimates)\n")
        for name, df in tables.items():
            f.write(f"\n{name}\n{'-' * len(name)}\n")
            f.write(df.to_string(index=False, float_format=lambda x:
                                 f"{x:.3f}") + "\n")
    q = tables["table_thresholds"]
    ck.add("tables", "registered thresholds: upper above 0, lower below 0, "
           "contrast wider than saturation",
           (q["registered: upper"] > 0).all()
           and (q["registered: lower"] < 0).all()
           and all(q[(q["feature"] == "contrast")]["registered: upper"]
                   .to_numpy() > q[(q["feature"] == "saturation")]
                   ["registered: upper"].to_numpy()), "")

    # --- figures ---
    bad = []
    figures(det[det["model"] == REGISTERED_MODEL], pw, c1, pair,
            r6.settings, out, bad)
    ck.add("figures", "figure text uses plain labels (no internal code "
           "names)", not bad, "; ".join(bad[:5]) if bad else "4 figures")

    write_summary(out, rr, r6.meta, r7.meta, tables["table_thresholds"], pw,
                  c1, pair, bm, runs, ck)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(settings_dict(), f, indent=2, default=str)
    pd.DataFrame(ck.rows).to_csv(os.path.join(out, "checks.csv"),
                                 index=False)
    files = sorted(set(os.listdir(out)) | {"report.txt", "run_meta.json"})
    meta = {"started": started, "seconds": time.time() - T0, "files": files}
    write_report(out, ck, tables, bad, meta)
    # the summary carries the check count: rewrite it now that all checks
    # are in
    write_summary(out, rr, r6.meta, r7.meta, tables["table_thresholds"], pw,
                  c1, pair, bm, runs, ck)
    with open(os.path.join(out, "run_meta.json"), "w") as f:
        json.dump({"started": started, "seconds": time.time() - T0,
                   "python": platform.python_version(),
                   "numpy": np.__version__, "pandas": pd.__version__,
                   "output_dir": out, "input_sha256": hashes,
                   "primary_rule": {k: v for k, v in rr.items()},
                   "checks_pass": len(ck.rows) - ck.n_fail,
                   "checks_fail": ck.n_fail}, f, indent=2, default=str)
    log(f"done: {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL")
    log(f"summary: {os.path.join(out, 'registration_summary.txt')}")


if __name__ == "__main__":
    main()
