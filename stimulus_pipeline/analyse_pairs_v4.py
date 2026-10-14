"""
analyse_pairs_v4.py  (stimulus pipeline, v4 axes, step 4)

Enumerates every usable trial the stimulus set can support, classifies
it, checks whether the conflict trials are fair, and reports what step 5
needs in order to spread trials across scenes.

Where this sits
---------------
  step 1  calibrate_levels_v4    the five colour and five fog rungs
  step 2  select_scenes_v4       shortlist -> you screen by hand
  step 3  build_stimulus_set_v4  the screened set -> stimuli + manifest
  step 4  this one               what trials can the set support?
  step 5  build_trials_v4        pick the trials actually shown
  step 6  simulate_power_v4      is that trial list powered?

Why this is separate from building the trial list: the pair inventory
is a property of the STIMULUS SET and changes only when the stimuli
change. The trial list is a sample from it, and you will rebuild that
several times while tuning proportions. Keeping them apart means the
expensive enumeration runs once.

SELF-CONTAINED and lightweight: it reads stimulus_manifest.csv and
nothing else. No images, no encoder -- the manifest already carries
every measured feature. Runs in seconds.

TWO STIMULUS SETS
-----------------
MANIFESTS is a LIST. Each manifest is analysed into its own output
directory, and a side-by-side comparison is written on top. That is the
point of this step for v4: you have a plain-fog set and a
brightness-corrected set, and the question is which one gives the
better experiment. The thresholds are held identical across sets so the
comparison means something.

The four trial types
--------------------
Each trial shows two images of the SAME scene, from different cells of
the colour x fog grid. Differences come from MEASURED features, never
from the parameters, because the mapping from parameter to feature is
scene-dependent -- fog drains colour by different amounts in different
scenes.

  sat_only   Same fog row. Chroma scaling works on a* and b* in CIELAB
             and leaves L* alone, so the fog feature is unchanged along
             a row and ONLY colour differs. Measures colour sensitivity
             cleanly.

  fog_only   Different fog rows, but the two cells' measured
             saturations coincide. Necessary because fog drains colour
             as a side effect: an unmatched foggy-vs-clear comparison
             confounds fog with colour loss. Only these make the fog
             contrast interpretable.

  together   Foggier AND less saturated: the way the world works, since
             fog takes colour with it. Easiest comparison, largest
             expected effect.

  conflict   Foggier AND MORE saturated. The cues point opposite ways,
             so the participant must reveal which one they follow. The
             ONLY type that can show the crossover -- that people who
             describe their experience as foggy choose differently from
             those who describe it as lifeless.

CONFLICT BALANCE
----------------
A conflict trial is a fair test only if neither cue is handed an
advantage. Raw differences cannot be compared: sat_mean spans roughly
45 units per scene while contrast_local spans roughly 11, so a 13-unit
colour difference and a 4-unit fog difference are comparable, not
lopsided.

So each difference is expressed as a fraction of that SCENE's own range
for that feature:

    z_sat = |d_sat| / (that scene's sat_mean range)
    z_fog = |d_fog| / (that scene's contrast_local range)
    balance = z_sat / z_fog

Near 1 means the cues are equally strong; below 1 means fog dominates.

The physics is asymmetric: fog drains colour, so a large POSITIVE
saturation difference cannot accompany a large fog difference. In v2
the fog cue ran about 1.6x stronger across conflict pairs, which would
bias a simple proportion toward "fog wins" whatever participants were
tracking. (A regression is unaffected -- it estimates per-unit weights.)
The fix is to draw conflict trials only from the BALANCED subset, which
this script identifies and flags for step 5.

This asymmetry is also one of the clearest differences between the two
fog families, since the brightness-corrected fog drains less colour. The
comparison table reports it.

WHAT STEP 5 NEEDS FROM HERE
---------------------------
Step 5 caps how many trials any one scene may contribute, so the design
does not rest on a handful of scenes. This script reports, for each
trial type, how concentrated the pool is and how many scenes can supply
a given cap -- so the cap can be chosen from evidence rather than
guessed. See section 5 of the report.

Inputs
------
  MANIFESTS   one or more stimulus_manifest.csv from step 3

Outputs (under OUTPUT_ROOT)
---------------------------
  <set>/all_pairs.csv            every usable pair, classified, with
                                 raw and normalised differences and the
                                 `balanced` flag. STEP 5 READS THIS.
  <set>/pair_summary.csv         counts and magnitudes per type
  <set>/scene_capability.csv     per-scene counts, incl. balanced
  <set>/weak_scenes.csv          poor conflict contributors
  <set>/concentration.csv        how concentrated each type is
  <set>/pairs_feature_space.png  where each type sits
  <set>/conflict_balance.png     balance distribution and per scene
  <set>/magnitude_coverage.png   difference sizes, with bin edges
  comparison.csv                 the sets side by side
  pairs_report.txt               the whole thing, in words

Usage
-----
    conda activate drl2drv
    python analyse_pairs_v4.py
"""

import os
import time
import itertools
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

# --- Input / output ---
#
# MANIFESTS
#   (label, path) per stimulus set. The label names the output
#   subdirectory and the column in the comparison table. One entry is
#   fine; two gives the comparison this step exists for.
MANIFESTS = [
    ("haze",    "data/stimulus_set_v4_ec_haze_2/stimulus_manifest.csv"),
    ("haze_eq", "data/stimulus_set_v4_ec_haze_eq_2/stimulus_manifest.csv"),
]

OUTPUT_ROOT = "data/pair_analysis_v4_ec"

# ON_EXISTING
#   "new" writes to OUTPUT_ROOT_2, _3, ... rather than overwriting an
#   existing analysis. "error" stops instead.
ON_EXISTING = "new"

# --- Feature columns ---
FEAT_SAT = "sat_mean"
FEAT_FOG = "contrast_local"

# FOG_SIGN
#   Multiplier on the raw fog-feature difference so that POSITIVE
#   always means "image A is foggier". -1 for contrast_local, which fog
#   LOWERS. Applied once, in classify_pair.
#
#   Getting this backwards swaps `together` and `conflict`, because the
#   distinction turns on whether d_fog and d_sat share a sign.
FOG_SIGN = -1.0

# --- Trial-type definitions (feature units, 0-255) ---
#
# These MUST match step 3's, or the pair inventory here will disagree
# with the affordance check that admitted the scenes.
#
# MIN_DSAT_SAT_ONLY
#   Minimum colour difference for a same-fog pair to be a usable
#   saturation trial. Below this the two images look alike and the
#   trial carries no information.
MIN_DSAT_SAT_ONLY = 17.0

# PAIR_SAT_TOL
#   How closely two cells' saturations must match to count as
#   "saturation held constant" in a fog trial. 3 units on a 0-255 scale
#   is effectively identical.
PAIR_SAT_TOL = 3.0

# MIN_DFOG_FOG_ONLY
#   Minimum fog difference for a saturation-matched pair to be worth
#   showing.
MIN_DFOG_FOG_ONLY = 4.2

# MIN_DSAT_MIXED / MIN_DFOG_MIXED
#   A mixed pair must differ by at least this much on BOTH features, or
#   it is really a single-feature trial with noise. The saturation
#   threshold is lower because conflict pairs cannot achieve large
#   positive d_sat -- the physics resists it.
MIN_DSAT_MIXED = 8.0
MIN_DFOG_MIXED = 3.0

# --- Conflict balance ---
#
# BALANCE_LO / BALANCE_HI
#   A conflict pair counts as BALANCED when z_sat / z_fog falls in this
#   window. 0.6-1.7 is v2's, meaning neither cue is more than about
#   1.7x the other in normalised terms.
#
#   HELD FIXED ACROSS SETS on purpose. A window fitted to each set
#   separately would hide the very difference the comparison is for:
#   the corrected fog drains less colour, so its conflict pairs should
#   sit closer to 1 on their own. Let them.
#
#   The report prints what window each set's own data would suggest, so
#   you can see whether v2's is still the right one. Narrow (0.8-1.25)
#   for stricter fairness and fewer pairs; widen if too few scenes can
#   supply balanced conflict trials.
BALANCE_LO = 0.6
BALANCE_HI = 1.7

# MIN_BALANCED_PER_SCENE
#   A scene is flagged as a weak conflict contributor below this many
#   balanced conflict pairs. Set low: with ~40 conflict trials over ~47
#   scenes, each scene contributes one or two.
MIN_BALANCED_PER_SCENE = 3

# --- Magnitude bins ---
#
# N_MAGNITUDE_BINS
#   The dose-response analysis splits trials by difference size and
#   asks whether people choose the manipulated image more often when
#   the difference is larger. A rising curve is far stronger evidence
#   than a single proportion, because a response bias would give a flat
#   line. This is how many bins that analysis will use; the edges are
#   reported so step 5 can sample evenly across them.
N_MAGNITUDE_BINS = 3

# MIN_PAIRS_PER_BIN
#   Minimum pairs the whole set must afford in each bin, per type.
MIN_PAIRS_PER_BIN = 40

# --- What step 5 will ask for ---
#
# TRIAL_TARGETS
#   Roughly how many trials of each type step 5 will draw. Used only to
#   report whether the pool can supply them while spread across scenes
#   -- nothing here depends on the exact numbers, and they are easy to
#   change later. v2 used about 137 analysed trials in total.
TRIAL_TARGETS = {"sat_only": 60, "fog_only": 60,
                 "together": 30, "conflict": 30}

# CONCENTRATION_TOP_K
#   The report gives the share of each pool held by the K scenes that
#   contribute most. A pool where a few scenes hold most of the pairs
#   is one where a per-scene cap matters.
CONCENTRATION_TOP_K = 5

RANDOM_SEED = 42

TYPES = ["sat_only", "fog_only", "together", "conflict"]


# ============================================================
# Small helpers
# ============================================================

def resolve_output_root():
    """OUTPUT_ROOT, never overwriting an existing analysis."""
    def occupied(d):
        return os.path.isdir(d) and (
            os.path.isfile(os.path.join(d, "pairs_report.txt"))
            or os.path.isfile(os.path.join(d, "comparison.csv")))

    if not occupied(OUTPUT_ROOT):
        return OUTPUT_ROOT
    if ON_EXISTING == "error":
        raise FileExistsError(
            f"{OUTPUT_ROOT} already holds an analysis. Move it, or set "
            "ON_EXISTING = 'new'.")
    n = 2
    while occupied(f"{OUTPUT_ROOT}_{n}"):
        n += 1
    print(f"  NOTE: {OUTPUT_ROOT} already holds an analysis; writing "
          f"to {OUTPUT_ROOT}_{n} instead.")
    return f"{OUTPUT_ROOT}_{n}"


def short(scene_id):
    """A scene id trimmed for printing: the category part."""
    return scene_id.split("_000")[0][:28]


# ============================================================
# Pair enumeration
# ============================================================

def classify_pair(sat_a, fog_a, sat_b, fog_b, same_row,
                  sat_range, fog_range):
    """Classify one ordered pair (A shown against B).

    Returns (type, d_sat, d_fog, z_sat, z_fog, balance), or None if the
    pair is unusable -- either too similar to be informative or too
    ambiguous to interpret.

    Normalisation: each difference is divided by the SCENE's own range
    for that feature, so the two cues can be compared. Without it,
    d_sat = 13 and d_fog = 4 look lopsided when they are in fact
    comparable, because sat_mean spans roughly four times the range
    contrast_local does.
    """
    d_sat = sat_a - sat_b
    # FOG_SIGN keeps "positive means A is foggier" true whichever
    # feature names the fog axis. The test below depends on it.
    d_fog = FOG_SIGN * (fog_a - fog_b)

    if same_row:
        # Same fog row: d_fog is ~0 by construction, because chroma
        # scaling leaves L* alone.
        if abs(d_sat) < MIN_DSAT_SAT_ONLY:
            return None
        t = "sat_only"
    elif abs(d_sat) <= PAIR_SAT_TOL and abs(d_fog) >= MIN_DFOG_FOG_ONLY:
        t = "fog_only"
    elif abs(d_fog) >= MIN_DFOG_MIXED and abs(d_sat) >= MIN_DSAT_MIXED:
        # With A taken as the foggier image, is it also LESS saturated
        # (the natural pattern) or MORE saturated (the conflict case)?
        t = "conflict" if ((d_fog > 0) == (d_sat > 0)) else "together"
    else:
        return None

    z_sat = abs(d_sat) / sat_range if sat_range > 0 else np.nan
    z_fog = abs(d_fog) / fog_range if fog_range > 0 else np.nan
    bal = (z_sat / z_fog) if (z_fog and z_fog > 0) else np.nan
    return t, d_sat, d_fog, z_sat, z_fog, bal


def enumerate_pairs(manifest, label):
    """Every usable ordered within-scene pair.

    ORDERED rather than unordered because left/right position is
    counterbalanced in the experiment, and keeping both directions
    makes the difference distribution symmetric about zero.
    """
    print("\n" + "=" * 60)
    print(f"ENUMERATING PAIRS  [{label}]")
    print("=" * 60)

    # Row identity comes from the LEVEL LABEL where the manifest has
    # one. v2 compared the haze parameter as a float; the label is the
    # same test without the floating-point edge.
    row_col = "fog_level" if "fog_level" in manifest.columns else "haze"
    col_col = ("colour_level" if "colour_level" in manifest.columns
               else "chroma")

    out = []
    for sid, g in manifest.groupby("scene_id", sort=False):
        g = g.reset_index(drop=True)
        sat = g[FEAT_SAT].to_numpy(float)
        fog = g[FEAT_FOG].to_numpy(float)
        row = g[row_col].to_numpy()
        col = g[col_col].to_numpy()
        fname = (g["filename"].to_numpy() if "filename" in g.columns
                 else np.array([""] * len(g)))
        chroma = (g["chroma"].to_numpy() if "chroma" in g.columns
                  else np.full(len(g), np.nan))
        haze = (g["haze"].to_numpy() if "haze" in g.columns
                else np.full(len(g), np.nan))
        sat_range = float(sat.max() - sat.min())
        fog_range = float(fog.max() - fog.min())
        cat = g["category"].iloc[0] if "category" in g.columns else ""
        grp = g["group"].iloc[0] if "group" in g.columns else ""
        ind = g["indoor"].iloc[0] if "indoor" in g.columns else ""

        for i, j in itertools.permutations(range(len(g)), 2):
            r = classify_pair(sat[i], fog[i], sat[j], fog[j],
                              row[i] == row[j], sat_range, fog_range)
            if r is None:
                continue
            t, d_sat, d_fog, z_sat, z_fog, bal = r
            out.append({
                "scene_id": sid, "category": cat, "group": grp,
                "indoor": ind, "type": t,
                "colour_a": col[i], "fog_a_level": row[i],
                "colour_b": col[j], "fog_b_level": row[j],
                "chroma_a": chroma[i], "haze_a": haze[i],
                "chroma_b": chroma[j], "haze_b": haze[j],
                "file_a": fname[i], "file_b": fname[j],
                "sat_a": sat[i], "sat_b": sat[j],
                "fog_a": fog[i], "fog_b": fog[j],
                "d_sat": d_sat, "d_fog": d_fog,
                "z_sat": z_sat, "z_fog": z_fog, "balance": bal,
                "sat_range": sat_range, "fog_range": fog_range,
            })

    pairs = pd.DataFrame(out)
    if not len(pairs):
        raise RuntimeError(
            f"[{label}] No usable pairs at all. Check that the manifest "
            "has varied colour and fog levels, and that the thresholds "
            "are not set far above the set's actual ranges.")
    n_sc = manifest.scene_id.nunique()
    print(f"  {n_sc} scenes, {len(manifest):,} images -> "
          f"{len(pairs):,} usable pairs")
    counts = pairs.type.value_counts().reindex(TYPES).fillna(0).astype(int)
    for t, n in counts.items():
        print(f"    {t:10s} {n:7,d}   per scene {n / n_sc:6.1f}")
    return pairs


# ============================================================
# Conflict balance
# ============================================================

def analyse_balance(pairs, label, out_dir, report):
    """Are the conflict trials fair to both cues?"""
    print("\n" + "=" * 60)
    print(f"CONFLICT BALANCE  [{label}]")
    print("=" * 60)

    c = pairs[pairs.type == "conflict"].copy()
    if not len(c):
        print("  No conflict pairs at all. The crossover cannot be "
              "tested. Loosen MIN_DSAT_MIXED / MIN_DFOG_MIXED.")
        report.append(("conflict pairs", "0 - FAIL"))
        return c, False, {}

    print("  differences as a fraction of each scene's own range:")
    print(c[["z_sat", "z_fog"]].describe()
          .loc[["mean", "25%", "50%", "75%", "max"]].round(3).to_string())

    med = float(c.balance.median())
    print(f"\n  median balance ratio z_sat/z_fog: {med:.2f}")
    if med < 0.9:
        print(f"    The fog cue runs about {1 / med:.1f}x stronger than "
              "the colour cue in a typical conflict pair.")
        print("    Expected: fog drains colour, so a large positive "
              "d_sat cannot accompany a large d_fog. This is why "
              "conflict trials are drawn from the balanced subset "
              "rather than sampled at random.")
    elif med > 1.1:
        print(f"    The COLOUR cue runs about {med:.1f}x stronger here. "
              "Unusual, and worth a look: it means this set's fog "
              "barely drains colour.")

    c["balanced"] = c.balance.between(BALANCE_LO, BALANCE_HI)
    n_bal = int(c.balanced.sum())
    print(f"\n  balanced pairs ({BALANCE_LO}-{BALANCE_HI}): "
          f"{n_bal:,} of {len(c):,} ({n_bal / len(c):.0%})")

    # What window would this set's own data suggest? Printed, never
    # applied: a per-set window would mask the difference between the
    # families, which is the thing being compared.
    q = c.balance.quantile([0.25, 0.75])
    print(f"  for reference, this set's own middle 50% spans "
          f"{q.iloc[0]:.2f}-{q.iloc[1]:.2f}")

    scenes = pairs.scene_id.unique()
    per = (c[c.balanced].groupby("scene_id").size()
           .reindex(scenes, fill_value=0))
    n_ok = int((per >= MIN_BALANCED_PER_SCENE).sum())
    print(f"  scenes with >= {MIN_BALANCED_PER_SCENE} balanced "
          f"conflict pairs: {n_ok}/{len(per)}")

    weak = per[per < MIN_BALANCED_PER_SCENE]
    if len(weak):
        print("  weak contributors: " + ", ".join(
            f"{short(s)}({n})" for s, n in weak.items()))
        weak.rename("n_balanced_conflict").to_csv(
            os.path.join(out_dir, "weak_scenes.csv"))

    ok = n_ok >= 0.8 * len(per)
    print(f"\n  RESULT: {'PASS' if ok else 'CHECK'}")
    if not ok:
        print("  Many scenes cannot supply balanced conflict trials. "
              "Widen BALANCE_LO/HI, or accept that conflict trials "
              "come from a subset of scenes.")
    report.append(("median balance ratio", f"{med:.2f}"))
    report.append(("balanced conflict pairs",
                   f"{n_bal:,} of {len(c):,} ({n_bal / len(c):.0%})"))
    report.append(("scenes with balanced conflict",
                   f"{n_ok}/{len(per)}"))
    stats = {"median_balance": med, "n_conflict": len(c),
             "n_balanced": n_bal,
             "frac_balanced": n_bal / len(c),
             "scenes_with_balanced": n_ok, "n_scenes": len(per),
             "q25": float(q.iloc[0]), "q75": float(q.iloc[1])}
    return c, ok, stats


# ============================================================
# Magnitude coverage
# ============================================================

def analyse_coverage(pairs, label, report):
    """Do the differences span small, medium and large?

    Reports the bin edges step 5 should sample across, so the
    dose-response curve has data at every point.
    """
    print("\n" + "=" * 60)
    print(f"MAGNITUDE COVERAGE  [{label}]")
    print("=" * 60)

    rows = []
    for t, feat in [("sat_only", "d_sat"), ("fog_only", "d_fog"),
                    ("together", "d_fog"), ("conflict", "d_fog")]:
        sub = pairs[pairs.type == t]
        if not len(sub):
            rows.append({"type": t, "driver": feat, "bin": 0,
                         "lo": np.nan, "hi": np.nan, "n_pairs": 0,
                         "ok": False})
            continue
        mag = sub[feat].abs()
        edges = np.quantile(mag, np.linspace(0, 1, N_MAGNITUDE_BINS + 1))
        for b in range(N_MAGNITUDE_BINS):
            sel = (mag >= edges[b]) & (mag <= edges[b + 1])
            rows.append({"type": t, "driver": feat, "bin": b + 1,
                         "lo": round(float(edges[b]), 2),
                         "hi": round(float(edges[b + 1]), 2),
                         "n_pairs": int(sel.sum()),
                         "ok": int(sel.sum()) >= MIN_PAIRS_PER_BIN})

    cov = pd.DataFrame(rows)
    print(cov.to_string(index=False))
    ok = bool(cov.ok.all())
    print(f"\n  RESULT: {'PASS' if ok else 'CHECK'} "
          f"(>= {MIN_PAIRS_PER_BIN} per bin)")
    report.append(("magnitude coverage", "PASS" if ok else "CHECK"))
    return cov, ok


# ============================================================
# Per-scene capability and concentration
# ============================================================

def scene_capability(pairs, conflict, label, out_dir, report):
    """How many trials of each type each scene can contribute."""
    print("\n" + "=" * 60)
    print(f"PER-SCENE CAPABILITY  [{label}]")
    print("=" * 60)

    cap = (pairs.pivot_table(index="scene_id", columns="type",
                             values="d_sat", aggfunc="size")
           .reindex(columns=TYPES).fillna(0).astype(int))
    bal = (conflict[conflict.balanced].groupby("scene_id").size()
           .reindex(cap.index, fill_value=0)
           if len(conflict) else pd.Series(0, index=cap.index))
    cap["conflict_balanced"] = bal
    cap["best_conflict_dsat"] = (
        conflict[conflict.balanced].groupby("scene_id")["d_sat"]
        .apply(lambda x: x.abs().max()).reindex(cap.index).fillna(0)
        if len(conflict) else 0.0)

    meta = pairs.groupby("scene_id")[["category", "group",
                                      "indoor"]].first()
    cap = meta.join(cap)
    cap.to_csv(os.path.join(out_dir, "scene_capability.csv"))

    print(cap[TYPES + ["conflict_balanced"]].describe()
          .loc[["min", "25%", "50%", "75%", "max"]].round(0).to_string())
    print("\n  weakest scenes by balanced conflict pairs:")
    print(cap.nsmallest(6, "conflict_balanced")[
        ["category", "conflict_balanced", "best_conflict_dsat"]]
        .round(1).to_string())
    report.append(("scenes", str(len(cap))))
    return cap


def analyse_concentration(pairs, cap, label, out_dir, report):
    """How much of each pool rests on a few scenes?

    Step 5 caps how many trials one scene may contribute. That cap is
    only meaningful if enough scenes can reach it, so the numbers it
    needs are: how concentrated the pool is now, and how many scenes
    could supply the cap a given trial count implies.

    The point is not that concentration is wrong -- it is expected,
    since scenes differ -- but that a trial type resting on two or
    three scenes makes the design depend on those scenes being typical.
    """
    print("\n" + "=" * 60)
    print(f"CONCENTRATION, AND WHAT STEP 5 CAN ASK FOR  [{label}]")
    print("=" * 60)

    n_scenes = len(cap)
    rows = []
    for t in TYPES:
        col = "conflict_balanced" if t == "conflict" else t
        v = cap[col].sort_values(ascending=False)
        total = int(v.sum())
        if total == 0:
            continue
        topk = int(v.head(CONCENTRATION_TOP_K).sum())
        # how many scenes hold half the pool
        half = int((v.cumsum() <= total / 2).sum()) + 1
        want = TRIAL_TARGETS.get(t, 0)
        # spread perfectly, each scene gives this many
        need_cap = int(np.ceil(want / n_scenes)) if want else 0
        can = int((v >= max(need_cap, 1)).sum())
        rows.append({
            "type": t, "pool": total, "scenes": n_scenes,
            "per_scene_median": float(v.median()),
            "top1_share": v.iloc[0] / total,
            f"top{CONCENTRATION_TOP_K}_share": topk / total,
            "scenes_for_half": half,
            "trials_wanted": want,
            "cap_if_even": need_cap,
            "scenes_meeting_cap": can,
            "max_trials_at_cap": can * need_cap if need_cap else 0,
        })
    con = pd.DataFrame(rows)
    con.to_csv(os.path.join(out_dir, "concentration.csv"), index=False)

    print(con[["type", "pool", "per_scene_median", "top1_share",
               f"top{CONCENTRATION_TOP_K}_share", "scenes_for_half"]]
          .round(3).to_string(index=False))
    print()
    print("  can step 5 spread the trials it wants?")
    for r in con.itertuples():
        enough = r.max_trials_at_cap >= r.trials_wanted
        print(f"    {r.type:10s} wants {r.trials_wanted:3d}: cap "
              f"{r.cap_if_even} per scene, {r.scenes_meeting_cap}/"
              f"{n_scenes} scenes can supply it -> up to "
              f"{r.max_trials_at_cap:3d}  "
              f"{'OK' if enough else 'TIGHT'}")
    tight = [r.type for r in con.itertuples()
             if r.max_trials_at_cap < r.trials_wanted]
    if tight:
        print(f"\n  TIGHT for: {', '.join(tight)}. Either raise the cap "
              "(and accept more trials from fewer scenes), ask for "
              "fewer trials of that type, or loosen the thresholds.")
    report.append(("concentration", "written"))
    return con


# ============================================================
# Figures
# ============================================================

def make_figures(pairs, conflict, cov, label, out_dir):
    # --- Where each type sits ---
    fig, ax = plt.subplots(figsize=(8, 7))
    for t in TYPES:
        sub = pairs[pairs.type == t]
        if len(sub):
            ax.scatter(sub.d_sat, sub.d_fog, s=4, alpha=0.2, label=t)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("d sat_mean  (image A - image B)")
    ax.set_ylabel(f"d {FEAT_FOG}  (A - B, signed so + means A is foggier)")
    ax.set_title(f"Usable pairs by trial type  [{label}]\n"
                 "sat_only on the horizontal, fog_only on the vertical,\n"
                 "mixed types in the quadrants")
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pairs_feature_space.png"), dpi=150)
    plt.close(fig)

    # --- Conflict balance ---
    if len(conflict):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        axes[0].hist(conflict.balance.clip(0, 3), bins=60, color="C3",
                     alpha=0.8)
        axes[0].axvline(1.0, color="k", lw=1, ls="--",
                        label="equal cues")
        axes[0].axvspan(BALANCE_LO, BALANCE_HI, color="C2", alpha=0.15,
                        label="balanced window")
        axes[0].set_xlabel("balance ratio  z_sat / z_fog")
        axes[0].set_ylabel("conflict pairs")
        axes[0].set_title(f"Conflict-pair balance  [{label}]\n"
                          "(below 1 = the fog cue is the stronger one)")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)

        per = (conflict[conflict.balanced].groupby("scene_id").size()
               .reindex(sorted(pairs.scene_id.unique()), fill_value=0))
        axes[1].bar(range(len(per)), per.sort_values().to_numpy(),
                    color="C0")
        axes[1].axhline(MIN_BALANCED_PER_SCENE, color="k", lw=1,
                        ls="--", label=f"threshold "
                                       f"{MIN_BALANCED_PER_SCENE}")
        axes[1].set_xlabel("scenes (sorted)")
        axes[1].set_ylabel("balanced conflict pairs")
        axes[1].set_title("Balanced conflict pairs per scene")
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "conflict_balance.png"),
                    dpi=150)
        plt.close(fig)

    # --- Magnitude distributions with bin edges ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, t in zip(axes, TYPES):
        sub = pairs[pairs.type == t]
        feat = "d_sat" if t == "sat_only" else "d_fog"
        if not len(sub):
            ax.set_title(f"{t}  (none)")
            continue
        ax.hist(sub[feat].abs(), bins=40, color="C0", alpha=0.8)
        for _, r in cov[cov.type == t].iterrows():
            if np.isfinite(r.lo):
                ax.axvline(r.lo, color="k", lw=0.8, ls=":")
        ax.set_xlabel(f"|{feat}|")
        ax.set_title(f"{t}  (n={len(sub):,})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("pairs")
    fig.suptitle(f"Difference magnitudes, with dose-response bin edges "
                 f"[{label}]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "magnitude_coverage.png"), dpi=150)
    plt.close(fig)


def comparison_figure(results, root):
    """The two sets side by side, on the things that differ."""
    labels = [r["label"] for r in results]
    if len(labels) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    x = np.arange(len(TYPES))
    w = 0.8 / len(results)
    for k, r in enumerate(results):
        v = [r["counts"].get(t, 0) for t in TYPES]
        axes[0].bar(x + k * w, v, w, label=r["label"])
    axes[0].set_xticks(x + w * (len(results) - 1) / 2)
    axes[0].set_xticklabels(TYPES, rotation=20)
    axes[0].set_ylabel("usable pairs")
    axes[0].set_title("Pool size by trial type")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")

    for r in results:
        c = r["conflict"]
        if len(c):
            axes[1].hist(c.balance.clip(0, 3), bins=60, alpha=0.5,
                         label=r["label"], density=True)
    axes[1].axvline(1.0, color="k", lw=1, ls="--")
    axes[1].axvspan(BALANCE_LO, BALANCE_HI, color="C2", alpha=0.12)
    axes[1].set_xlabel("balance ratio  z_sat / z_fog")
    axes[1].set_ylabel("density")
    axes[1].set_title("Conflict balance\n"
                      "(closer to 1 is a fairer test)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    for k, r in enumerate(results):
        v = [r["stats"].get("frac_balanced", 0),
             r["stats"].get("scenes_with_balanced", 0)
             / max(r["stats"].get("n_scenes", 1), 1)]
        axes[2].bar([0 + k * w, 1 + k * w], v, w, label=r["label"])
    axes[2].set_xticks([w * (len(results) - 1) / 2,
                        1 + w * (len(results) - 1) / 2])
    axes[2].set_xticklabels(["share of conflict\npairs balanced",
                             "share of scenes with\n>= 3 balanced"])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Conflict usability")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(root, "comparison.png"), dpi=150)
    plt.close(fig)


# ============================================================
# One set
# ============================================================

def analyse_one(label, path, root):
    print("\n" + "#" * 60)
    print(f"# {label}")
    print("#" * 60)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Manifest not found: {path}\nRun step 3 "
            "(build_stimulus_set_v4.py) for this fog family first.")
    manifest = pd.read_csv(path)
    need = {"scene_id", FEAT_SAT, FEAT_FOG}
    missing = need - set(manifest.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if not ({"fog_level", "colour_level"} <= set(manifest.columns)
            or {"haze", "chroma"} <= set(manifest.columns)):
        raise ValueError(
            f"{path} has neither (fog_level, colour_level) nor "
            "(haze, chroma); one pair is needed to tell which cells "
            "share a fog row.")

    out_dir = os.path.join(root, label)
    os.makedirs(out_dir, exist_ok=True)

    report = []
    pairs = enumerate_pairs(manifest, label)
    conflict, ok_bal, stats = analyse_balance(pairs, label, out_dir,
                                              report)

    # Carry the balanced flag onto the main table so step 5 can filter
    # conflict trials without recomputing anything.
    pairs["balanced"] = False
    if len(conflict):
        pairs.loc[conflict.index, "balanced"] = conflict["balanced"]

    cov, ok_cov = analyse_coverage(pairs, label, report)
    cap = scene_capability(pairs, conflict, label, out_dir, report)
    con = analyse_concentration(pairs, cap, label, out_dir, report)
    make_figures(pairs, conflict, cov, label, out_dir)

    pairs.to_csv(os.path.join(out_dir, "all_pairs.csv"), index=False)
    cov.to_csv(os.path.join(out_dir, "pair_summary.csv"), index=False)

    counts = (pairs.type.value_counts().reindex(TYPES)
              .fillna(0).astype(int).to_dict())
    return {"label": label, "path": path, "manifest": manifest,
            "pairs": pairs, "conflict": conflict, "cov": cov,
            "cap": cap, "con": con, "counts": counts, "stats": stats,
            "report": report, "ok": ok_bal and ok_cov,
            "out_dir": out_dir}


# ============================================================
# Comparison
# ============================================================

def build_comparison(results, root):
    """One row per quantity, one column per set."""
    rows = {}

    def put(k, label, v):
        rows.setdefault(k, {})[label] = v

    for r in results:
        L = r["label"]
        m, p, s = r["manifest"], r["pairs"], r["stats"]
        n_sc = m.scene_id.nunique()
        put("scenes", L, n_sc)
        put("images", L, len(m))
        put("usable pairs", L, len(p))
        for t in TYPES:
            put(f"{t} pairs", L, r["counts"].get(t, 0))
            put(f"{t} per scene", L,
                round(r["counts"].get(t, 0) / n_sc, 1))
        put("median balance ratio", L,
            round(s.get("median_balance", np.nan), 3))
        put("fog cue stronger by", L,
            round(1 / s["median_balance"], 2)
            if s.get("median_balance") else np.nan)
        put("balanced conflict pairs", L, s.get("n_balanced", 0))
        put("share of conflict balanced", L,
            round(s.get("frac_balanced", 0), 3))
        put("scenes with >=3 balanced", L,
            f"{s.get('scenes_with_balanced', 0)}/{n_sc}")
        bc = r["cap"]["conflict_balanced"].sort_values(ascending=False)
        put("top-1 scene share of balanced conflict", L,
            round(bc.iloc[0] / bc.sum(), 3) if bc.sum() else np.nan)

    comp = pd.DataFrame(rows).T
    comp.index.name = "quantity"
    comp.to_csv(os.path.join(root, "comparison.csv"))
    return comp


# ============================================================
# Report
# ============================================================

def write_report(results, comp, root, t0):
    path = os.path.join(root, "pairs_report.txt")
    with open(path, "w") as f:
        def W(s=""):
            f.write(s + "\n")

        W("PAIR ANALYSIS REPORT  (v4 axes, step 4)")
        W("=" * 70)
        for r in results:
            W(f"{r['label']:10s} {r['path']}")
        W(f"Thresholds : sat_only >= {MIN_DSAT_SAT_ONLY}, "
          f"fog_only |d_sat| <= {PAIR_SAT_TOL} & "
          f"|d_fog| >= {MIN_DFOG_FOG_ONLY}")
        W(f"             mixed |d_sat| >= {MIN_DSAT_MIXED} & "
          f"|d_fog| >= {MIN_DFOG_MIXED}")
        W(f"Balanced   : {BALANCE_LO} <= z_sat/z_fog <= {BALANCE_HI}")
        W(f"Elapsed    : {time.time() - t0:.1f}s")
        W()

        if len(results) > 1:
            W("1. THE SETS SIDE BY SIDE")
            W("-" * 70)
            W("  Thresholds are identical across sets, so every")
            W("  difference below comes from the fog itself.")
            W()
            W(comp.to_string())
            W()
            W("  The balance ratio is the one to read first. It is the")
            W("  colour cue's strength divided by the fog cue's, both")
            W("  normalised by the scene's own range. Below 1 means fog")
            W("  dominates the conflict trials, which biases a simple")
            W("  proportion toward 'fog wins' whatever participants are")
            W("  tracking -- and it is why conflict trials are drawn")
            W("  from the balanced subset.")
            W()

        for i, r in enumerate(results, 2 if len(results) > 1 else 1):
            W(f"{i}. {r['label'].upper()}")
            W("-" * 70)
            for k, v in r["report"]:
                W(f"  {k:32s} {v}")
            W()
            W("  magnitude coverage:")
            W("  " + r["cov"].to_string(index=False).replace("\n",
                                                             "\n  "))
            W()
            W("  concentration, and what step 5 can ask for:")
            W("  " + r["con"].round(3).to_string(index=False)
              .replace("\n", "\n  "))
            W()

        n = len(results) + (2 if len(results) > 1 else 1)
        W(f"{n}. WHAT STEP 5 READS")
        W("-" * 70)
        for r in results:
            W(f"  {os.path.join(r['out_dir'], 'all_pairs.csv')}")
        W()
        W("  It carries the `balanced` flag, so step 5 filters conflict")
        W("  trials without recomputing anything, and `file_a`/`file_b`")
        W("  so it can write the trial list directly.")
        W()
        W("  Set step 5's per-scene cap from the concentration table")
        W("  above: the `cap_if_even` column is what a perfectly spread")
        W("  draw needs, and `scenes_meeting_cap` says how many scenes")
        W("  can actually supply it.")
        W()
        W("OVERALL: " + ("PASS" if all(r["ok"] for r in results)
                         else "CHECK -- see the failing section above"))
    return path


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("ANALYSE PAIRS  (v4, step 4)")
    print("=" * 60)
    root = resolve_output_root()
    os.makedirs(root, exist_ok=True)
    print(f"  sets   : {', '.join(l for l, _ in MANIFESTS)}")
    print(f"  output : {root}")

    results = [analyse_one(label, path, root)
               for label, path in MANIFESTS]

    comp = build_comparison(results, root)
    if len(results) > 1:
        comparison_figure(results, root)
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(comp.to_string())

    path = write_report(results, comp, root, t0)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        flag = "PASS" if r["ok"] else "CHECK"
        print(f"  {r['label']:10s} {len(r['pairs']):7,d} pairs   {flag}")
    print(f"\n  Report : {path}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print("\n  NEXT: step 5 (build_trials_v4) samples the trial list "
          "from all_pairs.csv.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
