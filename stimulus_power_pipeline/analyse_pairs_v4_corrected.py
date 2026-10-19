"""
analyse_pairs_v4_corrected.py  (stimulus pipeline, v4 axes, step 4)

Enumerates every usable trial the stimulus set can support, classifies
it, checks whether the conflict trials are fair, and reports what step 5
needs in order to spread trials across scenes.

Where this sits
---------------
  step 1  calibrate_levels_v4_corrected    the ladders, both directions
  step 2  select_scenes_v4_corrected       shortlist -> screened by hand
  step 3  build_stimulus_set_v4_corrected  the set -> stimuli + manifest
  step 4  this one                         what trials can the set
                                           support?
  step 5  build_trials_v4                  pick the trials actually shown
  step 6  simulate_power_v4                is that trial list powered?

WHAT CHANGED FROM analyse_pairs_v4.py
-------------------------------------
THE COMPARISON NOW ASKS A DIFFERENT QUESTION. The old script put the
two FOG FAMILIES side by side -- plain haze against the
brightness-corrected haze_eq -- because the open question then was
which family to build the experiment from. That question is answered:
haze_eq gave 15,004 usable pairs against haze's 13,560, and 2,030
colour-only pairs against 1,390, so haze_eq is what step 3 builds and
what the experiment ships.

The open question now is whether the two VIVID COLOUR LEVELS earn their
place. The rerun exists because of a specific weakness:

    colour is detected less often than fog. Colour-only pairs are
    confined to a single fog row, and in mixed pairs the haze drains
    colour anyway, so a participant whose choices follow colour is
    identified far less reliably than one whose choices follow fog.

p65 and p75 are the fix: they lengthen the colour axis upward, which
should produce more colour-only pairs and larger ones. So this script
now analyses ONE manifest TWICE -- once with the old colour levels and
once with the vivid levels included -- and the comparison table
measures exactly what they bought.

Both analyses run in a SINGLE invocation, as the two families did: one
output directory per variant, one comparison table, one report.

Three other changes:

  1. `use` FILTERING, which is new and necessary. The step-3 manifest
     now carries 45 cells per scene, and only 35 of them may become
     pair trials. The h = 0 row and the contrast-increase column are
     grid-block material: they carry `use = "grid"`. Feeding them to
     the pair classifier would invent trials the experiment never
     shows, and would classify the contrast cells -- whose fog feature
     moves UP -- under a rule written for a feature that only moves
     down.

  2. ORIENTED PREDICTORS. all_pairs.csv gains `pred_sat` and
     `pred_fog`, positive when image A is the one the hypothesis says
     will be chosen. The raw signed differences stay beside them. See
     the note on enumerate_pairs for why one is negated and the other
     is not.

  3. A REPRODUCTION CHECK. The `original` variant should reproduce the
     earlier pair table exactly, pair for pair, because step 3 proved
     the shared images are byte-identical. REPRODUCE_FROM points at it.

Why this is separate from building the trial list: the pair inventory
is a property of the STIMULUS SET and changes only when the stimuli
change. The trial list is a sample from it, and you will rebuild that
several times while tuning proportions. Keeping them apart means the
expensive enumeration runs once.

SELF-CONTAINED and lightweight: it reads stimulus_manifest.csv and
nothing else. No images, no encoder -- the manifest already carries
every measured feature. Runs in seconds.

TWO VARIANTS, ONE MANIFEST
--------------------------
VARIANTS is a LIST of (label, colour levels). Each is analysed into its
own output directory from the SAME manifest, and a side-by-side
comparison is written on top. The thresholds are held identical across
variants, so every difference in the comparison table comes from the
colour levels themselves and from nothing else.

This is also why the two variants must come from one manifest rather
than from two builds: reading one file twice guarantees the shared
cells are the same numbers, where two builds would only probably be.

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
  MANIFEST    ONE stimulus_manifest.csv from step 3, filtered to the
              cells whose `use` is in PAIR_USE
  VARIANTS    the colour levels each analysis keeps
  REPRODUCE_FROM  an earlier all_pairs.csv the control must reproduce

Outputs (under OUTPUT_ROOT)
---------------------------
  <variant>/all_pairs.csv        every usable pair, classified, with
                                 raw differences, the ORIENTED
                                 predictors in corpus-SD units
                                 (`pred_sat`, `pred_fog`) and in
                                 per-scene-range units
                                 (`pred_*_range`), the three normalised
                                 balances and the `balanced` flag.
                                 STEP 5 READS THIS.
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
# MANIFEST
#   ONE stimulus manifest, read once and analysed under each entry of
#   VARIANTS below. Step 3 builds the haze_eq set by default, which is
#   what the experiment ships; point this at the _haze directory
#   instead if you ever want the plain-fog set analysed.
MANIFEST = "data/stimulus_set_v4_ext_haze_eq/stimulus_manifest.csv"

# VARIANTS
#   (label, colour levels) per analysis. The label names the output
#   subdirectory and the column in the comparison table.
#
#   "original" is the colour ladder the earlier build had, and exists
#   to be checked against it: same levels, same images, so it must
#   reproduce the earlier pair table exactly. It is the control.
#
#   "extended" adds the two vivid levels. The difference between the
#   two columns of the comparison table is what p65 and p75 bought,
#   measured rather than assumed -- and the number that matters most is
#   the sat_only count, because colour-only trials are the scarce kind
#   and the reason for the rerun.
#
#   The fog levels are NOT varied. Every pair cell carries real haze
#   (the h = 0 row is grid-only), so both variants see the same five
#   fog rungs and the colour levels are the only thing that moves.
VARIANTS = [
    ("original", ["k=1", "p25", "p10", "p05", "p01"]),
    ("extended", ["p75", "p65", "k=1", "p25", "p10", "p05", "p01"]),
]

# PAIR_USE
#   Which `use` values may become a pair trial. The step-3 manifest
#   carries 45 cells per scene and only 35 are pair material:
#
#     pair   30/scene  the pair grid only
#     both    5/scene  the pair grid AND the fog grid screen
#     grid   10/scene  the h = 0 row and the three contrast cells --
#                      grid screens only, NEVER a pair
#
#   Dropping this filter would invent 10 cells/scene of trials the
#   experiment never shows, and would hand the contrast cells -- whose
#   fog feature moves UP, not down -- to a classifier whose `conflict`
#   and `together` rule assumes it only moves down.
#
#   A manifest with no `use` column is from an older build whose cells
#   were all pair material; it is accepted unchanged, with a note.
PAIR_USE = ("pair", "both")

# REPRODUCE_FROM
#   An earlier all_pairs.csv the `original` variant must reproduce,
#   pair for pair. Step 3 showed the 1,175 shared images are
#   byte-identical, so the pair table built from them has to be
#   identical too; anything else means this script changed what it
#   measures. Set to None to skip.
REPRODUCE_FROM = "data/pair_analysis_v4_ec/haze_eq/all_pairs.csv"

# REPRODUCE_LABEL
#   Which variant is the control. Must be one of the VARIANTS labels.
REPRODUCE_LABEL = "original"

OUTPUT_ROOT = "data/pair_analysis_v4_ext"

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
# BALANCE_NORM
#   What the two cue strengths are divided by before their ratio is
#   taken. THE SELECTION RULE DEPENDS ON THIS, so it is a setting
#   rather than a constant.
#
#     "corpus_sd"    each difference over the FEATURE's corpus SD.
#                    THE DEFAULT, and the one the registered model
#                    uses. Three reasons:
#
#                    1. The model fits differences in these units, so
#                       "balanced" should mean balanced in the model's
#                       units: balance = 1 then means a participant
#                       with equal weights in the registered model is
#                       at 50/50. Selecting in one set of units and
#                       analysing in another is incoherent.
#                    2. A selection rule should depend on the IMAGES,
#                       not on which levels happen to be in the ladder.
#                    3. It is ladder-independent, so two ladders can be
#                       compared on it at all.
#
#     "scene_range"  each difference over the scene's own range across
#                    the cells IN THIS ANALYSIS. What this script used
#                    to do, and the reason for the change: adding the
#                    two vivid colour levels grew the median sat range
#                    by 56% (54.3 -> 85.0), which shrank z_sat for
#                    every pair including the 15,004 that are
#                    bit-identical between the two ladders. Their
#                    median balance fell 0.668 -> 0.428 and the
#                    `balanced` flag flipped on 908 of 2,958 conflict
#                    pairs, with no pixel having changed.
#
#     "scene_base"   each difference over the SCENE's own unmanipulated
#                    value. Scene-adaptive like scene_range but
#                    ladder-independent, because the baseline does not
#                    move when levels are added. A Weber fraction.
#
#   All three are written to all_pairs.csv whatever this is set to --
#   `balance`, `balance_range` and `balance_base`. Only the one named
#   here drives the `balanced` flag and therefore step 5's conflict
#   trials.
BALANCE_NORM = "corpus_sd"

# SD_SAT / SD_FOG
#   The corpus SDs, from the `stimulus_path` population of step 1's
#   corpus_feature_stats_v4.csv (n = 20,000, measured the way a
#   stimulus is measured: original -> 512 -> 256). These are the same
#   numbers the registered model divides by, which is the point.
SD_SAT = 39.182
SD_FOG = 9.348

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
#   Kept at 0.6-1.7 for now. Whether to widen it is a question for the
#   step 6 simulations rather than for this script: conflict trials are
#   most informative for participants whose own indifference ratio lies
#   inside the window, so the right width depends on where those ratios
#   are assumed to sit, which is what step 6 varies.
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
                  sat_range, fog_range, base_sat, base_fog):
    """Classify one ordered pair (A shown against B).

    Returns (type, d_sat, d_fog, z_sat, z_fog, balance, balance_range,
    balance_base), or None if the pair is unusable -- either too
    similar to be informative or too ambiguous to interpret.

    WHY NORMALISE AT ALL: d_sat = 13 and d_fog = 4 look lopsided but
    are in fact comparable, because sat_mean spans roughly four times
    the range contrast_local does. Some common scale is needed before
    the two cues can be put in a ratio.

    WHICH SCALE is BALANCE_NORM's job, and all three are computed here
    so the choice stays visible in the output rather than being baked
    into one number:

      balance        corpus SD        -- ladder-independent, and the
                                        units the registered model
                                        fits. The selection rule.
      balance_range  the scene's range across the cells in THIS
                     analysis -- scene-adaptive, but moves when the
                     ladder changes. Reported only.
      balance_base   the scene's own unmanipulated value -- a Weber
                     fraction: scene-adaptive AND ladder-independent.
                     Reported only.

    balance_base is NaN when the scene's unmanipulated cell is not in
    the manifest, which is the normal case for an older build.
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

    def ratio(a, b):
        return (a / b) if (b and b > 0 and np.isfinite(b)) else np.nan

    # the three normalisations, in the order of the docstring
    zc_sat, zc_fog = abs(d_sat) / SD_SAT, abs(d_fog) / SD_FOG
    zr_sat = ratio(abs(d_sat), sat_range)
    zr_fog = ratio(abs(d_fog), fog_range)
    zb_sat = ratio(abs(d_sat), base_sat)
    zb_fog = ratio(abs(d_fog), base_fog)

    bal_corpus = ratio(zc_sat, zc_fog)
    bal_range = ratio(zr_sat, zr_fog)
    bal_base = ratio(zb_sat, zb_fog)

    # z_sat / z_fog are reported in the units the SELECTION uses, so
    # the describe() table in analyse_balance and the balance ratio
    # beside it are always on one scale.
    if BALANCE_NORM == "corpus_sd":
        z_sat, z_fog, bal = zc_sat, zc_fog, bal_corpus
    elif BALANCE_NORM == "scene_range":
        z_sat, z_fog, bal = zr_sat, zr_fog, bal_range
    elif BALANCE_NORM == "scene_base":
        z_sat, z_fog, bal = zb_sat, zb_fog, bal_base
    else:
        raise ValueError(
            "BALANCE_NORM must be 'corpus_sd', 'scene_range' or "
            f"'scene_base', got {BALANCE_NORM!r}")
    return t, d_sat, d_fog, z_sat, z_fog, bal, bal_range, bal_base


def enumerate_pairs(manifest, label, baseline=None):
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
        # the scene's UNMANIPULATED values, for the Weber definition.
        # They come from the full manifest rather than from these
        # cells, because the unmanipulated cell is grid-only and the
        # pair cells all carry haze.
        b = (baseline or {}).get(sid, (np.nan, np.nan))
        base_sat, base_fog = float(b[0]), float(b[1])
        cat = g["category"].iloc[0] if "category" in g.columns else ""
        grp = g["group"].iloc[0] if "group" in g.columns else ""
        ind = g["indoor"].iloc[0] if "indoor" in g.columns else ""

        for i, j in itertools.permutations(range(len(g)), 2):
            r = classify_pair(sat[i], fog[i], sat[j], fog[j],
                              row[i] == row[j], sat_range, fog_range,
                              base_sat, base_fog)
            if r is None:
                continue
            t, d_sat, d_fog, z_sat, z_fog, bal, bal_r, bal_b = r
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
                # Oriented predictors: POSITIVE when image A is the one
                # the hypothesis says will be chosen -- the less
                # colourful, the foggier.
                #
                # IN CORPUS-SD UNITS, the units the registered model
                # fits, so a coefficient is "per SD of the feature" and
                # the two are on one scale. The per-scene-range
                # versions follow as extra columns; step 5 carries both
                # through to the trial list under the same names, so a
                # name means one thing everywhere.
                #
                # Only d_sat is negated, and the asymmetry is real
                # rather than an oversight: d_sat is raw (sat_a -
                # sat_b, so positive means A is MORE colourful, the
                # opposite of the prediction), while d_fog has already
                # been through FOG_SIGN in classify_pair and so is
                # ALREADY positive when A is the foggier image. Negating
                # both would silently flip the fog predictor and turn
                # every fog coefficient in step 8 upside down.
                "pred_sat": -d_sat / SD_SAT,
                "pred_fog": d_fog / SD_FOG,
                "pred_sat_range": (-d_sat / sat_range
                                   if sat_range > 0 else np.nan),
                "pred_fog_range": (d_fog / fog_range
                                   if fog_range > 0 else np.nan),
                "z_sat": z_sat, "z_fog": z_fog, "balance": bal,
                # the two definitions NOT driving selection, carried so
                # the choice can be revisited without re-enumerating
                "balance_range": bal_r, "balance_base": bal_b,
                "sat_range": sat_range, "fog_range": fog_range,
                "base_sat": base_sat, "base_fog": base_fog,
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

    UNITS = {"corpus_sd": "in corpus SD",
             "scene_range": "as a fraction of the scene's own range",
             "scene_base": "as a fraction of the scene's own baseline"}
    print(f"  cue strengths, {UNITS[BALANCE_NORM]} "
          f"(BALANCE_NORM = {BALANCE_NORM!r}):")
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
    # variants, which is the thing being compared.
    q = c.balance.quantile([0.25, 0.75])
    print(f"  for reference, this set's own middle 50% spans "
          f"{q.iloc[0]:.2f}-{q.iloc[1]:.2f}")

    # The distribution INSIDE the window. Conflict trials are most
    # informative for a participant whose own indifference ratio lies
    # where these pairs are, so this is what step 6 needs in order to
    # decide whether 0.6-1.7 is the right width.
    kept = c.loc[c.balanced, "balance"]
    if len(kept):
        qq = kept.quantile([0, .1, .25, .5, .75, .9, 1]).round(3)
        print(f"\n  distribution of balance among the {len(kept):,} "
              "pairs the window keeps:")
        print("    " + "  ".join(f"p{int(p * 100):02d} {v:.2f}"
                                 for p, v in qq.items()))
        print(f"    mean {kept.mean():.3f}   "
              f"share below 1.0 {float((kept < 1).mean()):.0%}")

    # The other two definitions, for comparison only.
    for col, nm in (("balance_range", "scene range"),
                    ("balance_base", "scene baseline / Weber")):
        if col in c.columns and c[col].notna().any():
            v = c[col]
            inw = float(v.between(BALANCE_LO, BALANCE_HI).mean())
            print(f"  [reported only] {nm:24s} median "
                  f"{v.median():.3f}, {inw:.0%} inside the window")

    scenes = pairs.scene_id.unique()
    per = (c[c.balanced].groupby("scene_id").size()
           .reindex(scenes, fill_value=0))
    n_ok = int((per >= MIN_BALANCED_PER_SCENE).sum())
    print(f"  scenes with >= {MIN_BALANCED_PER_SCENE} balanced "
          f"conflict pairs: {n_ok}/{len(per)}")

    print(f"    per scene: min {int(per.min())}, median "
          f"{int(per.median())}, max {int(per.max())}")
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

def scene_baseline(m):
    """Each scene's UNMANIPULATED sat_mean and contrast_local.

    Taken from the full manifest BEFORE the `use` filter, because the
    unmanipulated cell (chroma 1, haze 0) is grid-only -- every pair
    cell carries real haze, so the pair rows cannot supply it.

    Needed only for balance_base, the Weber definition, which is a
    reported column rather than the selection rule. A manifest without
    that cell -- any older build -- gives an empty map and a column of
    NaN, which costs nothing.
    """
    if {"chroma", "haze"} <= set(m.columns):
        sel = np.isclose(m.chroma, 1.0) & np.isclose(m.haze, 0.0)
        if "kind" in m.columns:
            sel &= m.kind.to_numpy() != "contrast_up"
    elif {"colour_level", "fog_level"} <= set(m.columns):
        sel = ((m.colour_level == "k=1") & (m.fog_level == "h=0")).to_numpy()
    else:
        return {}
    b = m[sel]
    if not len(b):
        return {}
    b = b.drop_duplicates(subset="scene_id")
    return {r.scene_id: (getattr(r, FEAT_SAT), getattr(r, FEAT_FOG))
            for r in b.itertuples()}


def load_manifest():
    """Read the manifest ONCE, and keep only the pair-eligible cells.

    Read once rather than per variant because the variants have to be
    comparable: two reads of one file would be identical in practice,
    but one read makes it true by construction, and the shared cells
    are precisely what the comparison rests on.
    """
    if not os.path.isfile(MANIFEST):
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}\nRun step 3 "
            "(build_stimulus_set_v4_corrected.py) first.")
    m = pd.read_csv(MANIFEST)
    need = {"scene_id", FEAT_SAT, FEAT_FOG}
    missing = need - set(m.columns)
    if missing:
        raise ValueError(
            f"{MANIFEST} is missing columns: {sorted(missing)}")
    if not ({"fog_level", "colour_level"} <= set(m.columns)
            or {"haze", "chroma"} <= set(m.columns)):
        raise ValueError(
            f"{MANIFEST} has neither (fog_level, colour_level) nor "
            "(haze, chroma); one pair is needed to tell which cells "
            "share a fog row.")

    print("\n" + "=" * 60)
    print("MANIFEST")
    print("=" * 60)
    print(f"  {MANIFEST}")
    base = scene_baseline(m)
    n0 = len(m)
    print(f"  {n0:,} rows, {m.scene_id.nunique()} scenes, "
          f"{n0 // max(m.scene_id.nunique(), 1)} cells/scene")

    if "use" in m.columns:
        counts = m.use.value_counts()
        print("  by use: " + ", ".join(f"{u} {n:,}"
                                       for u, n in counts.items()))
        m = m[m.use.isin(PAIR_USE)].reset_index(drop=True)
        print(f"  keeping use in {PAIR_USE}: {len(m):,} pair cells "
              f"({len(m) // max(m.scene_id.nunique(), 1)}/scene)")
        if not len(m):
            raise RuntimeError(
                f"No rows with use in {PAIR_USE}. The manifest has "
                f"{sorted(counts.index)}; PAIR_USE must name the "
                "values step 3 wrote for pair-eligible cells.")
    else:
        print("  NOTE: no `use` column -- an older build, whose cells "
              "were all\n        pair material. Taking every row.")

    n_sc = m.scene_id.nunique()
    if base:
        print(f"  unmanipulated cell found for {len(base)}/{n_sc} "
              "scenes (for balance_base)")
    else:
        print("  no unmanipulated cell in the manifest; balance_base "
              "will be NaN.\n  It is a reported column only, so "
              "nothing depends on it.")

    # The h = 0 row should already be gone with the `use` filter; say so
    # if it is not, because a pair against an unmanipulated image is a
    # trial type the design does not have.
    if "fog_level" in m.columns and (m.fog_level == "h=0").any():
        n = int((m.fog_level == "h=0").sum())
        print(f"  WARNING: {n} cell(s) at fog_level 'h=0' survived the "
              "use filter.\n           Pairs in this design are always "
              "two manipulated images.")
    return m, base


def analyse_one(label, levels, manifest, baseline, root):
    print("\n" + "#" * 60)
    print(f"# {label}")
    print("#" * 60)

    if "colour_level" in manifest.columns:
        have = set(manifest.colour_level.unique())
        unknown = [c for c in levels if c not in have]
        if unknown:
            raise RuntimeError(
                f"[{label}] colour level(s) {unknown} are not in the "
                f"manifest.\n  present: {sorted(have)}\n  VARIANTS "
                "names levels by the label step 1 and step 3 use; a "
                "level that was\n  never rendered cannot be analysed.")
        manifest = manifest[manifest.colour_level.isin(levels)]
        manifest = manifest.reset_index(drop=True)
    else:
        print("  NOTE: no `colour_level` column; the variant's level "
              "list is ignored.")

    n_sc = max(manifest.scene_id.nunique(), 1)
    print(f"  colour levels: {', '.join(levels)}")
    print(f"  {len(manifest):,} cells  ({len(manifest) // n_sc}/scene = "
          f"{len(levels)} colour x "
          f"{len(manifest) // n_sc // max(len(levels), 1)} fog)")

    out_dir = os.path.join(root, label)
    os.makedirs(out_dir, exist_ok=True)

    report = []
    pairs = enumerate_pairs(manifest, label, baseline)
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

    # CANONICAL ROW ORDER, for the same reason the manifest has one:
    # step 5 shuffles this table to sample trials, so its order is
    # part of what determines the trial list. Sorting on the pair's
    # own identity makes the file independent of how the manifest
    # happened to be ordered.
    pairs = pairs.sort_values(["scene_id", "file_a", "file_b"],
                              kind="stable").reset_index(drop=True)
    pairs.to_csv(os.path.join(out_dir, "all_pairs.csv"), index=False)
    cov.to_csv(os.path.join(out_dir, "pair_summary.csv"), index=False)

    counts = (pairs.type.value_counts().reindex(TYPES)
              .fillna(0).astype(int).to_dict())
    return {"label": label, "path": MANIFEST, "levels": levels,
            "manifest": manifest,
            "pairs": pairs, "conflict": conflict, "cov": cov,
            "cap": cap, "con": con, "counts": counts, "stats": stats,
            "report": report, "ok": ok_bal and ok_cov,
            "out_dir": out_dir}


# ============================================================
# Reproduction of the earlier pair table
# ============================================================

def check_reproduction(results, root):
    """The control variant must reproduce the earlier pair table.

    Step 3 established that the 1,175 images the two builds share are
    byte-identical, so their measured features are identical, so the
    pair table built from them must be identical too -- same pairs,
    same types, same differences. If it is not, the change is in THIS
    script rather than in the stimuli, and every comparison below is
    measuring two different things.

    Compared on (scene_id, file_a, file_b), which is exact: the
    filenames did not change, because step 3 keeps the old naming for
    the cells the two builds share.
    """
    print("\n" + "=" * 60)
    print("REPRODUCTION OF THE EARLIER PAIR TABLE")
    print("=" * 60)
    if not REPRODUCE_FROM:
        print("  REPRODUCE_FROM is None; skipped.")
        return None, True
    if not os.path.isfile(REPRODUCE_FROM):
        print(f"  {REPRODUCE_FROM} not found; skipped.")
        print("  Point REPRODUCE_FROM at the earlier run's "
              "all_pairs.csv to enable this.")
        return None, True

    hit = [r for r in results if r["label"] == REPRODUCE_LABEL]
    if not hit:
        print(f"  no variant labelled {REPRODUCE_LABEL!r}; skipped.")
        return None, True
    new = hit[0]["pairs"]
    old = pd.read_csv(REPRODUCE_FROM)
    print(f"  reference: {REPRODUCE_FROM}")
    print(f"    {len(old):,} pairs   vs   {len(new):,} in "
          f"{REPRODUCE_LABEL!r}")

    key = ["scene_id", "file_a", "file_b"]
    if not set(key) <= set(old.columns):
        print(f"  the reference has no {key}; comparing counts only.")
        same = len(old) == len(new)
        print(f"  RESULT: {'PASS' if same else 'FAIL'}")
        return None, same

    o = old.set_index(key)
    n = new.set_index(key)
    only_old = o.index.difference(n.index)
    only_new = n.index.difference(o.index)
    shared = o.index.intersection(n.index)

    rows = {"pairs in both": len(shared),
            "only in the reference": len(only_old),
            "only in this run": len(only_new)}
    diffs = {}
    for c in ("type", "d_sat", "d_fog"):
        if c in o.columns and c in n.columns:
            a, b = o.loc[shared, c], n.loc[shared, c]
            if c == "type":
                diffs[c] = int((a.to_numpy() != b.to_numpy()).sum())
            else:
                diffs[c] = float(np.abs(a.to_numpy()
                                        - b.to_numpy()).max())
    for k, v in rows.items():
        print(f"    {k:24s} {v:,}")
    for k, v in diffs.items():
        print(f"    worst {k:18s} {v:,}" if isinstance(v, int)
              else f"    worst |d {k}|{'':10s} {v:.3e}")

    ok = (not len(only_old) and not len(only_new)
          and diffs.get("type", 0) == 0
          and diffs.get("d_sat", 0.0) < 1e-9
          and diffs.get("d_fog", 0.0) < 1e-9)
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    if not ok:
        if len(only_old) or len(only_new):
            print("  The two tables do not contain the same pairs. "
                  "Either the cell set\n  differs (check PAIR_USE and "
                  "the variant's colour levels) or a threshold\n  "
                  "moved.")
            for nm, idx in (("only in the reference", only_old),
                            ("only in this run", only_new)):
                if len(idx):
                    print(f"    first {nm}: {tuple(idx[0])}")
        else:
            print("  The same pairs, but different numbers. That is "
                  "this script's measurement\n  changing, not the "
                  "stimuli -- step 3 proved the shared images are "
                  "identical.")
    return rows, ok


# ============================================================
# Is the selection rule ladder-independent?
# ============================================================

def check_balance_invariance(results, root):
    """A pair present in two variants must get the SAME balance.

    This is the property the switch to corpus-SD units was made for,
    so it is checked rather than assumed. Under the old
    scene_range definition it failed badly: the 15,004 pairs common to
    both ladders had bit-identical d_sat and d_fog, yet their median
    balance moved 0.668 -> 0.428 and the `balanced` flag flipped on 908
    of 2,958 conflict pairs -- a selection rule responding to which
    levels were in the ladder rather than to the images.

    balance_range is EXPECTED to differ here and is reported alongside,
    because its moving is the whole reason the rule changed.
    """
    print("\n" + "=" * 60)
    print("BALANCE INVARIANCE ACROSS LADDERS")
    print("=" * 60)
    if len(results) < 2:
        print("  only one variant; nothing to compare.")
        return None, True

    a, b = results[0], results[1]
    key = ["scene_id", "file_a", "file_b"]
    A = a["pairs"].set_index(key)
    B = b["pairs"].set_index(key)
    shared = A.index.intersection(B.index)
    print(f"  {a['label']} vs {b['label']}: {len(shared):,} pairs in "
          "both")
    if not len(shared):
        print("  nothing shared; skipped.")
        return None, True

    # The balance columns are compared on CONFLICT pairs only, which
    # is where the rule is applied and the only place the ratio is
    # meaningful. A sat_only pair sits in one fog row, so its d_fog is
    # ~0 by construction (median 0.03 levels), and any ratio dividing
    # by it is enormous in both variants alike -- a difference of 1e5
    # between two meaningless numbers, which would swamp the check
    # without saying anything about it. The raw differences are
    # compared on every shared pair.
    conf = A.loc[shared, "type"].to_numpy() == "conflict"
    rows = {}
    for col in ("d_sat", "d_fog", "balance", "balance_range",
                "balance_base"):
        if col not in A.columns or col not in B.columns:
            continue
        x = A.loc[shared, col].to_numpy(float)
        y = B.loc[shared, col].to_numpy(float)
        if col.startswith("balance"):
            x, y = x[conf], y[conf]
        fin = np.isfinite(x) & np.isfinite(y)
        rows[col] = (float(np.abs(x[fin] - y[fin]).max())
                     if fin.any() else np.nan)

    flips = np.nan
    if "balanced" in A.columns and "balanced" in B.columns:
        ca = A.loc[shared]
        cb = B.loc[shared]
        m = (ca.type == "conflict").to_numpy()
        flips = int((ca.balanced.to_numpy()[m]
                     != cb.balanced.to_numpy()[m]).sum())
        n_conf = int(m.sum())

    print(f"\n  BALANCE_NORM = {BALANCE_NORM!r}")
    print(f"  worst difference on the shared pairs "
          f"(balance columns on the {int(conf.sum()):,} conflict "
          "pairs):")
    for k, v in rows.items():
        tag = ""
        if k == "balance":
            tag = "   <- the selection rule"
        elif k == "balance_range":
            tag = "   (expected to move; not the rule)"
        print(f"    {k:16s} {v:.3e}{tag}")
    if np.isfinite(flips):
        print(f"\n  `balanced` flips on {flips:,} of {n_conf:,} "
              "shared conflict pairs")

    ok = (rows.get("balance", np.inf) < 1e-9) and (flips == 0)
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("  The same images get the same balance under both "
              "ladders, so which\n  conflict trials are eligible does "
              "not depend on which levels were\n  rendered.")
    else:
        print("  The selection rule still moves with the ladder. With "
              "BALANCE_NORM =\n  'corpus_sd' it should not; "
              "'scene_range' is expected to fail this and\n  is kept "
              "only as a reported column.")
    rows["balanced_flips"] = flips
    pd.Series(rows).rename("worst_difference").to_csv(
        os.path.join(root, "balance_invariance.csv"))
    return rows, ok


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
        # The quantities the vivid levels were added to move: how many
        # colour-only trials there are, and how large they get. The
        # rerun exists because colour-only pairs are the scarce kind.
        so = p[p.type == "sat_only"]
        put("sat_only median |d_sat|", L,
            round(float(so.d_sat.abs().median()), 1) if len(so)
            else np.nan)
        put("sat_only p90 |d_sat|", L,
            round(float(so.d_sat.abs().quantile(0.90)), 1) if len(so)
            else np.nan)
        put("sat_only max |d_sat|", L,
            round(float(so.d_sat.abs().max()), 1) if len(so)
            else np.nan)
        put("colour levels", L, len(r.get("levels", [])) or np.nan)

    comp = pd.DataFrame(rows).T
    comp.index.name = "quantity"
    comp.to_csv(os.path.join(root, "comparison.csv"))
    return comp


# ============================================================
# Report
# ============================================================

def write_report(results, comp, root, t0, repro=None,
                 ok_repro=True, inv=None, ok_inv=True):
    path = os.path.join(root, "pairs_report.txt")
    with open(path, "w") as f:
        def W(s=""):
            f.write(s + "\n")

        W("PAIR ANALYSIS REPORT  (v4 axes, step 4)")
        W("=" * 70)
        W(f"Manifest   : {MANIFEST}")
        W(f"Pair cells : use in {PAIR_USE}  (the h = 0 row and the "
          "contrast")
        W("             column are grid-block material and excluded)")
        for r in results:
            W(f"{r['label']:10s} colour levels: "
              f"{', '.join(r.get('levels', []))}")
        W(f"Thresholds : sat_only >= {MIN_DSAT_SAT_ONLY}, "
          f"fog_only |d_sat| <= {PAIR_SAT_TOL} & "
          f"|d_fog| >= {MIN_DFOG_FOG_ONLY}")
        W(f"             mixed |d_sat| >= {MIN_DSAT_MIXED} & "
          f"|d_fog| >= {MIN_DFOG_MIXED}")
        W(f"Balanced   : {BALANCE_LO} <= z_sat/z_fog <= {BALANCE_HI}"
          f"   (z in {BALANCE_NORM!r} units)")
        if BALANCE_NORM == "corpus_sd":
            W(f"             SD_SAT {SD_SAT}, SD_FOG {SD_FOG} "
              "(stimulus_path population)")
        W(f"Elapsed    : {time.time() - t0:.1f}s")
        W()

        W("0. REPRODUCTION OF THE EARLIER PAIR TABLE")
        W("-" * 70)
        W("  The control variant uses the colour levels the earlier")
        W("  build had, on images step 3 showed to be byte-identical.")
        W("  So its pair table must be identical too. If it is not,")
        W("  this script's measurement changed, and the comparison")
        W("  below is between two different things.")
        W()
        if repro is None:
            W("  Skipped (no reference table).")
        else:
            for k, v in repro.items():
                W(f"    {k:24s} {v:,}")
            W()
            W(f"  RESULT: {'PASS' if ok_repro else 'FAIL'}")
        W()

        W("0b. IS THE SELECTION RULE LADDER-INDEPENDENT?")
        W("-" * 70)
        W(f"  BALANCE_NORM = {BALANCE_NORM!r}. A pair present under")
        W("  both ladders must get the same balance, or the rule is")
        W("  responding to which levels were rendered rather than to")
        W("  the images.")
        W()
        if inv is None:
            W("  Skipped (one variant only).")
        else:
            for k, v in inv.items():
                if k == "balanced_flips":
                    W(f"    {'`balanced` flips':20s} {v:,}")
                else:
                    W(f"    worst |d {k:14s}| {v:.3e}")
            W()
            W(f"  RESULT: {'PASS' if ok_inv else 'FAIL'}")
            W()
            W("  balance_range is EXPECTED to move here. It is kept as")
            W("  a reported column because its moving is the reason")
            W("  the selection rule changed, not a fault in it.")
        W()
        W("  NOTE on section 0: the reproduction check compares the")
        W("  pairs, their types and their raw differences, and those")
        W("  are unchanged. It does NOT compare the `balanced` flag,")
        W("  which differs from the earlier run by design -- that flag")
        W("  is what the normalisation change was for. Step 5 will")
        W("  therefore draw a different conflict set than the current")
        W("  trial list used.")
        W()

        if len(results) > 1:
            W("1. THE VARIANTS SIDE BY SIDE")
            W("-" * 70)
            W("  Thresholds are identical across variants, and both")
            W("  read the same manifest, so every difference below")
            W("  comes from the two vivid colour levels and nothing")
            W("  else.")
            W()
            W("  The row that matters most is `sat_only pairs`.")
            W("  Colour-only trials are the scarce kind -- they are")
            W("  confined to a single fog row, and in mixed pairs the")
            W("  haze drains colour anyway -- so a participant whose")
            W("  choices follow colour is identified less reliably")
            W("  than one whose choices follow fog. That is the")
            W("  weakness p65 and p75 were added to fix, and the")
            W("  sat_only count is whether they fixed it.")
            W()
            W(comp.to_string())
            W()
            W("  The balance ratio is the colour cue's strength over")
            W("  the fog cue's, both divided by the feature's corpus")
            W("  SD. Below 1 means fog dominates the conflict trials,")
            W("  which biases a simple proportion toward 'fog wins'")
            W("  whatever participants are tracking -- and it is why")
            W("  conflict trials are drawn from the balanced subset")
            W("  rather than sampled at random.")
            W()
            W("  This row IS comparable across the two columns, which")
            W("  it was not in the earlier version of this script.")
            W("  Balance was normalised by the scene's own range then,")
            W("  and the vivid levels grew that range by 56%, so the")
            W("  ratio fell for every pair -- including 15,004 that")
            W("  are bit-identical between the two ladders. A reader")
            W("  comparing the two columns would have concluded the")
            W("  vivid levels worsened the cue balance, when nothing")
            W("  about any image had changed. Section 0b checks that")
            W("  this cannot happen now.")
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
        ok_all = (all(r["ok"] for r in results) and ok_repro
                  and ok_inv)
        W("OVERALL: " + ("PASS" if ok_all
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
    print(f"  variants : {', '.join(l for l, _ in VARIANTS)}")
    print(f"  output   : {root}")

    manifest, baseline = load_manifest()
    results = [analyse_one(label, levels, manifest, baseline, root)
               for label, levels in VARIANTS]

    repro, ok_repro = check_reproduction(results, root)
    inv, ok_inv = check_balance_invariance(results, root)

    comp = build_comparison(results, root)
    if len(results) > 1:
        comparison_figure(results, root)
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(comp.to_string())

    path = write_report(results, comp, root, t0, repro, ok_repro,
                        inv, ok_inv)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        flag = "PASS" if r["ok"] else "CHECK"
        print(f"  {r['label']:10s} {len(r['pairs']):7,d} pairs   {flag}")
    if repro is not None:
        print(f"  {'reproduction':12s} "
              f"{'PASS' if ok_repro else 'FAIL'}")
    if inv is not None:
        print(f"  {'invariance':12s} "
              f"{'PASS' if ok_inv else 'FAIL'}")
    print(f"\n  Report : {path}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print("\n  NEXT: step 5 (build_trials_v4) samples the trial list "
          "from all_pairs.csv.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
