"""
plot_power_seq_v4.py  (stimulus pipeline for the v4 axes, step 7 figures)

The figures for the preregistration, drawn from step 7's tables.

Separate from step 7 on purpose. Step 7 runs a simulation and takes a
couple of minutes; figure design is iterative and you should not have
to re-run a simulation to move a legend. Everything here reads CSVs
that step 7 has already written, so it finishes in seconds and can be
run as often as you like.

It also means the figures can be rebuilt months later from the saved
tables, without the trial lists or the stimuli being present.

Where this sits
---------------
  step 6  simulate_power_v4      one person: floors, crossover,
                                 additivity, trial count
  step 7  simulate_power_seq_v4  N people: responders, group test,
                                 sequential stopping
  this    plot_power_seq_v4      the figures, from step 7's tables

WHAT EACH FIGURE IS FOR
-----------------------
Not all of these belong in a preregistration. Pick from them:

  01_claim_vs_group   THE figure. The existence claim against the
                      group test, at every sample size, for each
                      responder share. This is the whole argument for
                      the analysis plan in one panel set: where the
                      two curves separate is what a mixture sample
                      does to an averaging test.

  02_null_check       The decision rule tested against a population
                      where every weight is exactly zero: observed
                      false-positive rate against what the binomial
                      says it should be. A reviewer who doubts the
                      counting rule is answered by this and nothing
                      else. Needs the spread = 0 scenarios.

  03_claim_by_pattern The existence claim for each feature, under each
                      assumption about which features responders
                      actually track. Shows that colour and fog are
                      close once you compare like with like.

  04_sensitivity      The claim against N for every assumed effect
                      weight and between-person spread. The answer to
                      "but what if the effect is weaker than you
                      assumed" -- it is already in the figure.

  05_evidence         Kusztor-style: the group Bayes factor's three
                      outcomes (effect / null / inconclusive) stacked
                      against N. Comparable with their figure, and it
                      shows the inconclusive band that a p-value
                      cannot express.

  06_conjunction      Joint outcomes: both features claimed, only
                      fog, only colour, neither. Needs step 7's
                      p_joint_* columns; skipped with a note if the
                      tables predate them.

  07_stopping         Where a sequentially monitored study ends up,
                      against the per-person equivalent. The argument
                      for a single final look.

Also written:

  bf_cutoffs.csv      the t value, and the observed effect size, at
                      which the Bayes factor crosses its thresholds,
                      per sample size. For the preregistration: "BF >
                      3 at N = 10 means t > ..." lets a reader check
                      the rule without re-running the simulation.

Inputs (per set, under INPUT_ROOT/<label>/)
-------------------------------------------
  responders.csv    every scenario x N            (step 7)
  lookup_by_n.csv   the headline patterns         (step 7)
  sequential.csv    stopping outcomes             (step 7)

Outputs (per set, under INPUT_ROOT/<label>/figures/)
----------------------------------------------------
  01_claim_vs_group.png ... 07_stopping.png, bf_cutoffs.csv

Usage
-----
    conda activate drl2drv
    python plot_power_seq_v4.py

Seconds. No simulation, no GPU, nothing to re-run.
"""

import os
import itertools

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

# --- Input ---
#
# INPUT_ROOT
#   Step 7's output directory. The figures are written into a
#   `figures/` subdirectory of each set inside it, so nothing step 7
#   wrote is ever touched.
INPUT_ROOT = "data/power_seq_v4_ec"

# LABELS
#   Which sets to draw. None means every subdirectory that holds a
#   responders.csv.
LABELS = None

# --- What to draw ---
#
# These must match the settings step 7 ran with, because the figures
# label themselves from them. They are read from the tables where
# possible and only used as fallbacks here.
FLOOR_ALPHA = 0.025
BF_THRESHOLD = 3.0
BF_SCALE = 0.707

# HEADLINE_*
#   The scenario the main figures are drawn for. "both" is the
#   realistic population: someone whose sense of unreality responds to
#   desaturation probably responds to contrast loss too.
HEADLINE_PATTERN = "both"
HEADLINE_WEIGHT = 1.0
HEADLINE_SPREAD = 1.0
HEADLINE_CORR = 0.0

# PLAN_N
#   The sample size the preregistration plans around, marked on the
#   figures. 14 because that is the last N at which two detections
#   suffice -- see step 7's note on k_required stepping up.
PLAN_N = 14

# SHARE_COLOURS
#   One colour per responder share, used consistently everywhere so
#   the figures can be read side by side.
SHARE_COLOURS = {1.0: "#1b3a5c", 0.5: "#2e7fa8",
                 0.2: "#d98324", 0.1: "#b23a48"}

# --- Style ---
FIG_DPI = 200
FONT_SIZE = 10
GRID_ALPHA = 0.25

# BF_CUTOFF_NS
#   Sample sizes to tabulate Bayes factor cut-offs for.
BF_CUTOFF_NS = [5, 7, 10, 12, 14, 15, 18, 20]

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ============================================================
# Loading
# ============================================================

def find_sets(root):
    """Every subdirectory of root that holds a responders.csv."""
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"{root} not found. Run step 7 "
            "(simulate_power_seq_v4.py) first, or point INPUT_ROOT at "
            "its output directory.")
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isfile(os.path.join(d, "responders.csv")):
            if LABELS is None or name in LABELS:
                out.append(name)
    if not out:
        raise FileNotFoundError(
            f"No set directories with a responders.csv under {root}.")
    return out


def load_set(root, label):
    """Step 7's tables for one set, plus what is missing."""
    d = os.path.join(root, label)
    resp = pd.read_csv(os.path.join(d, "responders.csv"))
    seq_path = os.path.join(d, "sequential.csv")
    seq = pd.read_csv(seq_path) if os.path.isfile(seq_path) else None

    missing = []
    if "p_joint_both" not in resp.columns:
        missing.append("p_joint_* (re-run step 7 for the conjunction "
                       "figure)")
    if not ((resp.pattern == "neither") & (resp.spread == 0.0)).any():
        missing.append("spread = 0 scenarios (add 0.0 to "
                       "SPREAD_FACTORS for the null check)")
    return resp, seq, missing


def headline(resp, pattern=None):
    """The headline slice: one pattern, one weight, one spread."""
    pat = HEADLINE_PATTERN if pattern is None else pattern
    g = resp[(resp.pattern == pat)
             & (resp.weight == HEADLINE_WEIGHT)
             & (resp.spread == HEADLINE_SPREAD)
             & (resp["w_corr"] == HEADLINE_CORR)]
    return g.sort_values(["share", "n"])


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    {os.path.basename(path)}")


def _mark_plan_n(ax):
    """The planned sample size, on every figure that has N on x."""
    ax.axvline(PLAN_N, color="0.45", lw=0.9, ls=":", zorder=0)


# ============================================================
# 01  The existence claim against the group test
# ============================================================

def fig_claim_vs_group(resp, label, out_dir):
    """The central figure: what each analysis buys, at every N.

    Two panels, same axes. Left is the per-person existence claim,
    right is the group Bayes factor. The gap between them IS the
    argument for the analysis plan, and it opens exactly where the
    responder share falls -- which is the situation this experiment is
    actually in.
    """
    g = headline(resp)
    if not len(g):
        print("    01 skipped: no headline rows")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for sh, gg in g.groupby("share"):
        c = SHARE_COLOURS.get(float(sh), None)
        axes[0].plot(gg.n, gg.p_claim_fog, marker="o", ms=3.5,
                     color=c, label=f"{sh:.0%} respond")
        axes[1].plot(gg.n, gg.p_bf_effect_fog, marker="o", ms=3.5,
                     color=c, label=f"{sh:.0%} respond")
    for ax, title, sub in (
            (axes[0], "PER PERSON",
             "enough participants individually clear\n"
             "their own floor to beat multiplicity"),
            (axes[1], "GROUP TEST",
             f"Bayes factor reaches {BF_THRESHOLD:g}\n"
             "(the conventional threshold)")):
        ax.set_title(f"{title}\n{sub}", fontsize=FONT_SIZE + 1)
        ax.set_xlabel("participants")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=GRID_ALPHA)
        _mark_plan_n(ax)
    axes[0].set_ylabel("probability")
    axes[0].legend(fontsize=FONT_SIZE - 1, loc="lower right")
    axes[0].annotate(f"planned N = {PLAN_N}", (PLAN_N, 0.03),
                     fontsize=FONT_SIZE - 2, color="0.35",
                     rotation=90, ha="right", va="bottom")
    # Without this the drop at N = 15 reads as an error in the
    # simulation rather than as the rule doing its job.
    ks = g[["n", "k_required"]].drop_duplicates().sort_values("n")
    step = ks[ks.k_required.diff() > 0]
    if len(step):
        n0 = int(step.n.iloc[0])
        axes[0].annotate(
            f"the bar rises from {int(ks[ks.n == n0 - 1].k_required.iloc[0])}"
            f" to {int(step.k_required.iloc[0])} detections at N = {n0}:\n"
            "a larger sample gives multiplicity more room,\n"
            "so the claim gets harder before it recovers",
            xy=(n0, 0.5), xytext=(n0 + 0.6, 0.30),
            fontsize=FONT_SIZE - 3, color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.45", lw=0.9,
                            shrinkA=0, shrinkB=4))
    fig.suptitle(
        f"What each analysis can establish  [{label}]\n"
        f"responders track {HEADLINE_PATTERN}; weight "
        f"{HEADLINE_WEIGHT:g}, spread {HEADLINE_SPREAD:g}",
        fontsize=FONT_SIZE + 2, y=1.04)
    _finish(fig, os.path.join(out_dir, "01_claim_vs_group.png"))


# ============================================================
# 02  The decision rule against a true null
# ============================================================

def fig_null_check(resp, label, out_dir):
    """Does the counting rule deliver the error rate it promises?

    The spread = 0 "neither" population is the only one in which every
    participant's true weight is exactly zero. Anything the rule flags
    there is a false positive, so the observed rate can be compared
    directly with the binomial prediction. If these do not coincide,
    nothing else in the analysis plan can be trusted.
    """
    Z = resp[(resp.pattern == "neither") & (resp.spread == 0.0)]
    if not len(Z):
        print("    02 skipped: no spread = 0 scenarios")
        return
    g = Z.groupby("n").agg(
        k=("k_required", "first"),
        claim_fog=("p_claim_fog", "mean"),
        claim_col=("p_claim_colour", "mean"),
        any_fog=("p_any_fog", "mean"),
        predicted=("study_false_pos", "first")).reset_index()
    g["pred_any"] = stats.binom.sf(0, g.n, FLOOR_ALPHA)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    ax = axes[0]
    ax.plot(g.n, g.predicted, color="k", lw=1.6, zorder=1,
            label="predicted (binomial)")
    ax.plot(g.n, g.claim_fog, "o", ms=5, color="#2e7fa8", zorder=2,
            label="observed, fog")
    ax.plot(g.n, g.claim_col, "s", ms=4.5, color="#d98324", zorder=2,
            mfc="none", label="observed, colour")
    ax.axhline(0.05, color="0.5", lw=0.8, ls="--")
    ax.annotate("5% ceiling", (g.n.min(), 0.0515), fontsize=FONT_SIZE - 2,
                color="0.4")
    ax.set_xlabel("participants")
    ax.set_ylabel("false-positive rate")
    ax.set_title("THE COUNTING RULE\nevery weight exactly zero:\n"
                 "does the rule hold its promised rate?",
                 fontsize=FONT_SIZE + 1)
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(fontsize=FONT_SIZE - 1)
    for n, k in zip(g.n, g.k):
        if n in (g.n.min(), 14, 15, g.n.max()):
            ax.annotate(f"k={int(k)}", (n, 0.001), fontsize=FONT_SIZE - 3,
                        ha="center", color="0.35")

    ax = axes[1]
    ax.plot(g.n, g.pred_any, color="k", lw=1.6, label="predicted")
    ax.plot(g.n, g.any_fog, "o", ms=5, color="#b23a48",
            label="observed")
    ax.axhline(0.05, color="0.5", lw=0.8, ls="--")
    ax.annotate("5% -- where a decision rule should sit",
                (g.n.min(), 0.058), fontsize=FONT_SIZE - 2,
                color="0.4")
    ax.set_xlabel("participants")
    ax.set_ylabel("rate")
    ax.set_title("WHY ONE DETECTION IS NOT A CLAIM\n"
                 "'at least one participant flagged',\n"
                 "in a study where nobody responds",
                 fontsize=FONT_SIZE + 1)
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(fontsize=FONT_SIZE - 1)

    fig.suptitle(f"The decision rule, verified  [{label}]",
                 fontsize=FONT_SIZE + 2, y=1.03)
    _finish(fig, os.path.join(out_dir, "02_null_check.png"))


# ============================================================
# 03  The claim for each feature, under each pattern
# ============================================================

def fig_claim_by_pattern(resp, label, out_dir):
    """Colour and fog, compared like with like.

    Each panel is a different assumption about what responders track.
    Read each feature's curve in the panel where that feature HAS an
    effect: colour's curve under "colour only" or "both", fog's under
    "fog only" or "both". Reading colour's curve under "fog only"
    measures the spread between people, not colour.
    """
    pats = [p for p in ("both", "fog only", "colour only")
            if (resp.pattern == p).any()]
    if not pats:
        print("    03 skipped: no patterns")
        return
    fig, axes = plt.subplots(1, len(pats), figsize=(5.0 * len(pats), 5.0),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, pat in zip(axes, pats):
        g = headline(resp, pat)
        for sh, gg in g.groupby("share"):
            if float(sh) not in (0.2, 0.5):
                continue
            c = SHARE_COLOURS.get(float(sh))
            ax.plot(gg.n, gg.p_claim_fog, marker="o", ms=3.5, color=c,
                    label=f"fog, {sh:.0%} respond")
            ax.plot(gg.n, gg.p_claim_colour, marker="s", ms=3.5,
                    color=c, ls="--", mfc="none",
                    label=f"colour, {sh:.0%} respond")
        true_c = "yes" if pat in ("both", "colour only") else "NO"
        true_f = "yes" if pat in ("both", "fog only") else "NO"
        ax.set_title(f"responders track: {pat}\n"
                     f"(true colour effect: {true_c};  "
                     f"true fog effect: {true_f})",
                     fontsize=FONT_SIZE)
        ax.set_xlabel("participants")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=GRID_ALPHA)
        _mark_plan_n(ax)
    axes[0].set_ylabel("P(existence claim)")
    axes[0].legend(fontsize=FONT_SIZE - 2, loc="upper left")
    fig.suptitle(
        f"The existence claim, per feature and per assumption  "
        f"[{label}]\nsolid = fog, dashed = colour",
        fontsize=FONT_SIZE + 2, y=1.05)
    _finish(fig, os.path.join(out_dir, "03_claim_by_pattern.png"))


# ============================================================
# 04  Sensitivity to the assumptions
# ============================================================

def fig_sensitivity(resp, label, out_dir):
    """What if the effect is weaker, or people differ more?

    The answer to the reviewer's question, drawn in advance. One panel
    per assumed effect weight, one line per between-person spread.
    """
    g = resp[(resp.pattern == HEADLINE_PATTERN)
             & (resp["w_corr"] == HEADLINE_CORR)
             & (resp.share.isin([0.2, 0.5]))]
    if not len(g):
        print("    04 skipped: no rows")
        return
    weights = sorted(g.weight.unique())
    spreads = sorted(s for s in g.spread.unique() if s > 0)
    fig, axes = plt.subplots(1, len(weights),
                             figsize=(4.6 * len(weights), 4.8),
                             sharey=True, squeeze=False)
    axes = axes[0]
    greys = plt.cm.viridis(np.linspace(0.15, 0.8, len(spreads)))
    for ax, w in zip(axes, weights):
        for sp, col in zip(spreads, greys):
            for sh, ls in ((0.5, "-"), (0.2, "--")):
                gg = g[(g.weight == w) & (g.spread == sp)
                       & (g.share == sh)].sort_values("n")
                if not len(gg):
                    continue
                ax.plot(gg.n, gg.p_claim_fog, ls=ls, color=col, lw=1.5,
                        label=f"spread {sp:g}, {sh:.0%} respond")
        ax.set_title(f"effect weight {w:g}\n"
                     + {0.5: "(below the floor -- "
                             "individually undetectable)",
                        1.0: "(the smallest effect of interest)",
                        1.5: "(a strong effect)"}.get(w, ""),
                     fontsize=FONT_SIZE)
        ax.set_xlabel("participants")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=GRID_ALPHA)
        _mark_plan_n(ax)
    axes[0].set_ylabel("P(existence claim, fog)")
    axes[-1].legend(fontsize=FONT_SIZE - 3, loc="upper left")
    fig.suptitle(
        f"Sensitivity to what cannot be known in advance  [{label}]\n"
        f"responders track {HEADLINE_PATTERN}; solid = 50% respond, "
        "dashed = 20%",
        fontsize=FONT_SIZE + 2, y=1.05)
    _finish(fig, os.path.join(out_dir, "04_sensitivity.png"))


# ============================================================
# 05  The group Bayes factor's three outcomes
# ============================================================

def fig_evidence(resp, label, out_dir):
    """Kusztor-style: what the group test concludes, against N.

    Three outcomes stacked, so the INCONCLUSIVE band is visible. That
    band is the reason to use a Bayes factor at all here: a p-value
    would fold it into "not significant" and lose the distinction
    between 'no effect' and 'not enough data'.
    """
    g = headline(resp)
    if not len(g):
        print("    05 skipped: no headline rows")
        return
    shares = sorted(g.share.unique(), reverse=True)
    fig, axes = plt.subplots(1, len(shares),
                             figsize=(3.6 * len(shares), 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, sh in zip(axes, shares):
        gg = g[g.share == sh].sort_values("n")
        ax.stackplot(
            gg.n, gg.p_bf_effect_fog, gg.p_bf_incon_fog,
            gg.p_bf_null_fog,
            colors=["#2e7fa8", "#d9d9d9", "#b23a48"],
            labels=[f"evidence for an effect (BF > {BF_THRESHOLD:g})",
                    "inconclusive",
                    f"evidence FOR the null (BF < 1/{BF_THRESHOLD:g})"])
        ax.set_title(f"{sh:.0%} of the sample responds",
                     fontsize=FONT_SIZE)
        ax.set_xlabel("participants")
        ax.set_xlim(gg.n.min(), gg.n.max())
        ax.set_ylim(0, 1)
        _mark_plan_n(ax)
    axes[0].set_ylabel("share of simulated studies")
    axes[-1].legend(fontsize=FONT_SIZE - 3, loc="lower left",
                    framealpha=0.9)
    fig.suptitle(
        f"What the GROUP test concludes  [{label}]\n"
        "the red band is the failure mode: evidence for the null "
        "while the effect is real",
        fontsize=FONT_SIZE + 2, y=1.04)
    _finish(fig, os.path.join(out_dir, "05_evidence.png"))


# ============================================================
# 06  The conjunction across features
# ============================================================

def fig_conjunction(resp, label, out_dir):
    """Both features claimed, only one, or neither.

    Not the product of the two separate claim rates: the same
    participants drive both, so the outcomes are correlated. Step 7
    records the joint distribution directly.
    """
    if "p_joint_both" not in resp.columns:
        print("    06 skipped: no p_joint_* columns "
              "(re-run step 7)")
        return
    g = headline(resp)
    if not len(g):
        print("    06 skipped: no headline rows")
        return
    shares = sorted(g.share.unique(), reverse=True)
    fig, axes = plt.subplots(1, len(shares),
                             figsize=(3.6 * len(shares), 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, sh in zip(axes, shares):
        gg = g[g.share == sh].sort_values("n")
        neither = 1.0 - (gg.p_joint_both + gg.p_joint_fog_only
                         + gg.p_joint_colour_only)
        ax.stackplot(
            gg.n, gg.p_joint_both, gg.p_joint_fog_only,
            gg.p_joint_colour_only, neither,
            colors=["#1b3a5c", "#2e7fa8", "#d98324", "#e8e8e8"],
            labels=["both features", "fog only", "colour only",
                    "neither"])
        ax.set_title(f"{sh:.0%} of the sample responds",
                     fontsize=FONT_SIZE)
        ax.set_xlabel("participants")
        ax.set_xlim(gg.n.min(), gg.n.max())
        ax.set_ylim(0, 1)
        _mark_plan_n(ax)
    axes[0].set_ylabel("share of simulated studies")
    axes[-1].legend(fontsize=FONT_SIZE - 3, loc="lower left",
                    framealpha=0.9)
    fig.suptitle(
        f"Which features the study ends up claiming  [{label}]\n"
        f"responders track {HEADLINE_PATTERN}, so both claims are "
        "true claims",
        fontsize=FONT_SIZE + 2, y=1.04)
    _finish(fig, os.path.join(out_dir, "06_conjunction.png"))


# ============================================================
# 07  Sequential stopping
# ============================================================

def fig_stopping(seq, label, out_dir):
    """Where a sequentially monitored study ends up.

    The left bars are the group Bayes factor read after every
    participant. The right panel is the per-person equivalent: how
    many participants until the first and second individual detection.
    The contrast is the argument for a single final look.
    """
    if seq is None or not len(seq):
        print("    07 skipped: no sequential.csv")
        return
    shares = sorted(seq.share.unique(), reverse=True)
    kinds = ["effect", "null", "inconclusive"]
    cols = ["#2e7fa8", "#b23a48", "#d9d9d9"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    ax = axes[0]
    xs = np.arange(len(shares))
    width = 0.8 / len(kinds)
    for j, (kind, c) in enumerate(zip(kinds, cols)):
        vals = []
        for sh in shares:
            r = seq[(seq.share == sh) & (seq.outcome == kind)]
            vals.append(float(r.share_of_studies.iloc[0])
                        if len(r) else np.nan)
        ax.bar(xs + j * width, vals, width, color=c, label=kind,
               edgecolor="white", lw=0.6)
    ax.set_xticks(xs + width)
    ax.set_xticklabels([f"{sh:.0%}" for sh in shares])
    ax.set_xlabel("share of the sample that responds")
    ax.set_ylabel("share of simulated studies")
    ax.set_title("GROUP TEST, monitored sequentially\n"
                 "every interim look is another chance\n"
                 "to cross the wrong threshold",
                 fontsize=FONT_SIZE + 1)
    ax.legend(fontsize=FONT_SIZE - 1)
    ax.grid(alpha=GRID_ALPHA, axis="y")

    ax = axes[1]
    for lab, mk, c in (("first 1 responder", "o", "#2e7fa8"),
                       ("first 2 responders", "s", "#1b3a5c")):
        ys, xs2 = [], []
        for i, sh in enumerate(shares):
            r = seq[(seq.share == sh) & (seq.outcome == lab)]
            if len(r):
                xs2.append(i)
                ys.append(float(r.median_stop_n.iloc[0]))
        ax.plot(xs2, ys, marker=mk, ms=7, color=c, label=lab)
    ax.axhline(PLAN_N, color="0.45", lw=0.9, ls=":")
    ax.annotate(f"planned N = {PLAN_N}", (0, PLAN_N + 0.3),
                fontsize=FONT_SIZE - 2, color="0.35")
    ax.set_xticks(np.arange(len(shares)))
    ax.set_xticklabels([f"{sh:.0%}" for sh in shares])
    ax.set_xlabel("share of the sample that responds")
    ax.set_ylabel("median participants needed")
    ax.set_title("PER PERSON\nhow many participants until the\n"
                 "first and second detection",
                 fontsize=FONT_SIZE + 1)
    ax.legend(fontsize=FONT_SIZE - 1)
    ax.grid(alpha=GRID_ALPHA)

    fig.suptitle(f"Why the group test gets ONE final look  [{label}]",
                 fontsize=FONT_SIZE + 2, y=1.03)
    _finish(fig, os.path.join(out_dir, "07_stopping.png"))


# ============================================================
# Bayes factor cut-offs
# ============================================================

def jzs_bf_delta(t, n, sided, scale=BF_SCALE, n_grid=2001):
    """JZS Bayes factor, delta form. Verbatim from step 7."""
    df = n - 1
    dhat = t / np.sqrt(n)
    width = np.sqrt(1.0 / n + dhat ** 2 / (2.0 * df))
    lo, hi = dhat - 20 * width, dhat + 20 * width
    if sided == "one":
        lo = max(lo, 0.0)
        if hi <= lo:
            hi = lo + 20 * width
    delta = np.linspace(lo, hi, n_grid)
    prior = stats.cauchy.pdf(delta, 0, scale) * (2.0 if sided == "one"
                                                 else 1.0)
    like = np.nan_to_num(stats.nct.pdf(t, df, delta * np.sqrt(n)),
                         nan=0.0)
    denom = stats.t.pdf(t, df)
    return _trapz(like * prior, delta) / denom if denom > 0 else np.nan


def write_bf_cutoffs(out_dir):
    """The t at which the Bayes factor crosses each threshold.

    For the preregistration. A reader can apply "BF > 3 at N = 10
    means t > 2.26" to published summary statistics without re-running
    anything, which is what makes a Bayes-factor decision rule
    checkable rather than merely stated.
    """
    rows = []
    for n in BF_CUTOFF_NS:
        for sided in ("one", "two"):
            for name, target in (("effect", BF_THRESHOLD),
                                 ("null", 1.0 / BF_THRESHOLD)):
                lo, hi = 0.0, 12.0
                # The factor rises with t, so bisect on it.
                f = lambda t: jzs_bf_delta(t, n, sided) - target
                if f(lo) > 0 or f(hi) < 0:
                    t_cut = np.nan
                else:
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        if f(mid) < 0:
                            lo = mid
                        else:
                            hi = mid
                    t_cut = 0.5 * (lo + hi)
                rows.append({"n": n, "sided": sided,
                             "threshold": name,
                             "bf": target,
                             "t_cutoff": t_cut,
                             "d_cutoff": t_cut / np.sqrt(n)})
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "bf_cutoffs.csv")
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"    {os.path.basename(path)}")
    one = df[(df.sided == "one") & (df.threshold == "effect")]
    print("      one-sided, BF > "
          f"{BF_THRESHOLD:g}:  " + ",  ".join(
              f"N={int(r.n)}: t>{r.t_cutoff:.2f}"
              for _, r in one.iterrows() if np.isfinite(r.t_cutoff)))
    return df


# ============================================================
# Main
# ============================================================

def main():
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.edgecolor": "0.3",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
    })
    print("=" * 60)
    print("FIGURES FOR STEP 7  (v4 pipeline)")
    print("=" * 60)
    print(f"  Input : {INPUT_ROOT}")

    labels = find_sets(INPUT_ROOT)
    print(f"  Sets  : {', '.join(labels)}")

    for label in labels:
        print(f"\n  [{label}]")
        resp, seq, missing = load_set(INPUT_ROOT, label)
        for m in missing:
            print(f"    NOTE: missing {m}")
        out_dir = os.path.join(INPUT_ROOT, label, "figures")
        os.makedirs(out_dir, exist_ok=True)

        if not len(headline(resp)):
            avail = sorted(resp.pattern.unique())
            print(f"    WARNING: no rows for HEADLINE_PATTERN "
                  f"{HEADLINE_PATTERN!r}; available: {avail}")

        fig_claim_vs_group(resp, label, out_dir)
        fig_null_check(resp, label, out_dir)
        fig_claim_by_pattern(resp, label, out_dir)
        fig_sensitivity(resp, label, out_dir)
        fig_evidence(resp, label, out_dir)
        fig_conjunction(resp, label, out_dir)
        fig_stopping(seq, label, out_dir)
        write_bf_cutoffs(out_dir)
        print(f"    -> {out_dir}")

    print("\n  Done. Nothing step 7 wrote was modified.")


if __name__ == "__main__":
    main()
