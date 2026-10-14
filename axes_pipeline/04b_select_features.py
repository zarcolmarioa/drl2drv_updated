"""
04b_select_features.py  (pipeline step 04b)

Chooses a set of low-level features that are "independent enough" from
the candidates computed in steps 04 and 04c, using ONLY the features'
relationships with each other. The CLIP axes are never loaded here, so
the choice cannot be tuned to the results.

Version 2
---------
  * Candidates: step 04's 18 features plus the CORPUS CANDIDATES of step
    04c (periph_blur, miniature_index, periph_lum, hue_spread, cast_a,
    cast_b; read from 04c's extra_definitions.json). 04c's sweep-only
    features (edge_blur, fine_texture, noise_sigma, edge_overshoot) are
    not candidates: in natural photos they belong to the fine-detail
    family that contrast_local already represents.
  * The new features come LAST in PRIORITY (fixed before any axis result
    was seen), so a VIF conflict removes a new feature before an
    established one. They are not stimulus features (not FORCED): if
    selected, whether they are kept is decided after a visual check of
    their extreme images (step 07).
  * Missing values: some new features are undefined for some images
    (hue_spread for greyscale photographs; the regional blur features
    for images with too few edges). Correlations are computed PAIRWISE,
    on the images where both features are defined; the image set itself
    is the same as version 1 (images with all 18 step-04 features
    defined at 256 and 336). With no missing values this is exactly the
    version-1 computation.
  * REPRODUCTION check: the version-1 procedure (the 18 step-04
    features, the version-1 priority list) is rerun on the same images
    and must reproduce the version-1 selection in PREVIOUS_SELECTION.
  * Output goes to a new folder; version 1's selection is kept, so later
    steps can use either (SELECTION_JSON in steps 07 and 09a).

The procedure (fixed in advance)
--------------------------------
Run twice, on two image sets, and saved side by side:
    colour_only   the step-04 subsample without greyscale photographs
                  (is_greyscale from step 04a)
    all_images    the full step-04 subsample
PRIMARY_SET says which one later steps use; switching it needs no
recomputation (both selections are in selected_features.json).

For each image set:
  1. WITHIN-CATEGORY values: every feature minus its category mean, so
     two features are not called redundant just because some kinds of
     scene are high on both.
  2. Spearman correlation (rho) between every pair of features, on
     those values. |rho| is used: features moving in opposite directions
     are just as redundant.
  3. Hierarchical clustering (average linkage) on the distance
     1 - |rho|. Groups are cut where the average |rho| between two
     groups falls below CLUSTER_THRESHOLD (0.7: features in one group
     share roughly half or more of their rank variation). The grouping
     is also reported at SENSITIVITY_THRESHOLDS (0.6, 0.8).
  4. One feature per group:
       * a group containing a stimulus feature (FORCED) is represented
         by it (if a group holds two stimulus features, both are kept
         and flagged: never put them in the same multivariate model);
       * otherwise the first feature in PRIORITY (an order written down
         before seeing any data: established, interpretable measures
         first) whose 256-vs-336 stability is at least STABILITY_MIN.
  5. Check the chosen set together: variance inflation factor (VIF) of
     each feature from the within-category Spearman correlations
     (VIF_j = [R^-1]_jj). A VIF of 5 means 80% of that feature is
     predictable from the others combined. Redundancy belongs to the
     set, not to the one feature whose VIF is high, so while any
     selected feature (stimulus features included) exceeds VIF_MAX:
       a. take the feature with the highest VIF (the target);
       b. try the non-forced features from the LOWEST in PRIORITY
          upwards (the target itself included, in its place); drop the
          first whose removal brings the target to VIF_MAX or below;
       c. if none does (possible only when the target is a stimulus
          feature made redundant by another stimulus feature), flag
          the target and move on.
     Removing a feature never raises another's VIF, so the check always
     ends. Stimulus features are never dropped. Every step is logged
     (vif_trace_<set>.csv). The a-priori order, not the data, decides
     which of several redundant features goes.
     (First version: only features themselves above VIF_MAX could be
     dropped. That removed lum_mean, the highest-priority feature,
     because its VIF was high through lum_skew, lum_p05 and lum_std
     together, and left brightness without a representative.)

Stability (256 vs 336)
----------------------
For every feature, the Spearman correlation between its value at 256 px
and at 336 px on the same images (the step-04 crop-336 subsample; 04c
measured its features on the same subsample). A feature whose ranking
of images depends on the measurement size is a poor representative.
Reported for all features; used in step 4. A group whose members all
fall below STABILITY_MIN keeps its first member anyway, flagged (the
case of the regional blur features, 0.80-0.84, each a group of its
own).

Inputs
------
  FEATURES_DIR   features_crop.npy, feature_definitions.json,
                 subsample_ids.npy, features_crop336_subsample.npy,
                 run_meta.json                                   (04)
  EXTRA_DIR      extra_crop.npy, extra_crop336_subsample.npy,
                 extra_definitions.json, subsample_ids.npy,
                 run_meta.json                                   (04c)
  PREVIOUS_SELECTION  version 1's selected_features.json (optional;
                 the reproduction check is skipped without it)
  INDEX_CSV      image_index.csv (categories)
  FLAGS_CSV      image_flags.csv from step 04a (is_greyscale)

Outputs (under OUTPUT_DIR)
--------------------------
  selected_features.json            both selections, primary, settings,
                                    and which step's file each feature
                                    comes from (feature_sources)
  selection_report.txt              summary (start here)
  groups_<set>.csv                  group of each feature at 0.6/0.7/0.8,
                                    selected or not, and why
  vif_<set>.csv                     VIF of the selected features, before
                                    and after the VIF check
  vif_trace_<set>.csv               every removal tried by the VIF check
  correlations_within_<set>.csv     within-category Spearman matrix
                                    (signed)
  correlations_raw_<set>.csv        Spearman matrix without centring
                                    (signed)
  stability_256_vs_336.csv          per feature
  pairwise_n_<set>.csv              images behind each correlation
                                    (pairwise; less than the set only
                                    where a feature is undefined)
  dendrogram_<set>.png              the clustering, with the cut lines
  correlation_heatmap_<set>.png     within-category rho WITH SIGN (blue
                                    negative, red positive), in
                                    dendrogram order

Usage
-----
    conda activate drl2drv
    python 04b_select_features.py

Runs in under a minute.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

# --- Inputs / output ---
FEATURES_DIR        = "data/features"                  # step 04
EXTRA_DIR           = "data/features_extra"            # step 04c (v2)
INDEX_CSV           = "data/selection/image_index.csv"
FLAGS_CSV           = "data/monochrome/image_flags.csv"
OUTPUT_DIR          = "data/feature_selection_v3"
PREVIOUS_SELECTION  = "data/feature_selection/selected_features.json"

# --- Step 04c candidates ---
# None = the "corpus_candidates" listed in 04c's extra_definitions.json.
EXTRA_CANDIDATES    = ["hue_spread", "cast_b"]

# --- Which selection later steps use ---
PRIMARY_SET         = "colour_only"      # or "all_images"

# --- Stimulus features: always kept ---
FORCED              = ["sat_mean", "contrast_local", "lum_p05"]

# --- Grouping ---
CLUSTER_THRESHOLD   = 0.7
SENSITIVITY_THRESHOLDS = [0.6, 0.8]
LINKAGE_METHOD      = "average"

# --- Representative choice for groups without a stimulus feature ---
# Order fixed before looking at the data: established and interpretable
# measures first; measures with known limitations (ceiling effects,
# near-duplicates of a measure listed earlier) last.
PRIORITY            = [
    "lum_mean",           # standard brightness measure
    "lum_std",            # RMS contrast, the standard global contrast
    "colorfulness",       # Hasler & Suesstrunk, widely used
    "chroma_mean",        # perceptual colour amount (CIELAB)
    "lab_b_mean",         # warm-cool
    "lab_a_mean",         # red-green
    "spectral_slope",     # standard natural-image statistic
    "edge_density",
    "dark_channel_mean",  # haze prior
    "lum_skew",
    "lum_entropy",
    "sat_std",
    "lum_p95",            # ceiling at 255 in >1% of images
    "lum_p50",            # near-duplicate of lum_mean
    "val_mean",           # near-duplicate of lum_mean
    # --- version 2: step 04c features, after all step-04 features,
    #     most reliable (256 vs 336 in 04c) first ---
    "periph_lum",         # tunnel vision (darkening)
    "hue_spread",         # hue variety
    "cast_b",             # colour temperature
    "cast_a",             # tint
    "miniature_index",    # miniature / tilt-shift; stability 0.84
    "periph_blur",        # tunnel vision (blur); stability 0.80
]
# The version-1 list (for the reproduction check): the step-04 part.
PRIORITY_V1         = PRIORITY[:15]
STABILITY_MIN       = 0.90      # 256-vs-336 Spearman

# --- Joint check ---
VIF_MAX             = 5.0


# ============================================================
# Helpers
# ============================================================

def P(name):
    return os.path.join(OUTPUT_DIR, name)


def centre_by_category(X, cats):
    """Subtract each category's mean from every column."""
    df = pd.DataFrame(X)
    return (df - df.groupby(cats).transform("mean")).to_numpy()


def spearman_matrix(X, return_n=False):
    """Spearman correlation matrix of the columns of X. Columns without
    NaNs are ranked together (exactly the version-1 computation); a pair
    involving a column with NaNs uses the rows where both are defined
    (pairwise)."""
    n, p = X.shape
    fin = np.isfinite(X)
    full = fin.all(axis=0)
    R = np.eye(p)
    N = np.full((p, p), n, dtype=np.int64)
    fi = np.where(full)[0]
    if len(fi):
        Rk = np.column_stack([rankdata(X[:, j]) for j in fi])
        R[np.ix_(fi, fi)] = np.corrcoef(Rk, rowvar=False) if len(fi) > 1 \
            else 1.0
    for i in range(p):
        for j in range(i + 1, p):
            if full[i] and full[j]:
                continue
            m = fin[:, i] & fin[:, j]
            N[i, j] = N[j, i] = int(m.sum())
            r = np.corrcoef(rankdata(X[m, i]), rankdata(X[m, j]))[0, 1] \
                if m.sum() > 2 else np.nan
            R[i, j] = R[j, i] = r
    for i in range(p):
        N[i, i] = int(fin[:, i].sum())
    return (R, N) if return_n else R


def vif_from_corr(R):
    return np.diag(np.linalg.inv(R))


def cluster(absr, names, threshold):
    D = 1.0 - absr
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method=LINKAGE_METHOD)
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    return Z, labels


# ============================================================
# Selection for one image set
# ============================================================

def select(X, cats, names, stability, set_name, priority=None,
           write=True):
    """The selection procedure on one image set. write=False runs it
    without writing files (reproduction check)."""
    if priority is None:
        priority = PRIORITY
    Xc = centre_by_category(X, cats)
    R_within, N_pair = spearman_matrix(Xc, return_n=True)
    R_raw = spearman_matrix(X)
    if not np.isfinite(R_within).all():
        raise RuntimeError(f"{set_name}: a correlation could not be "
                           "computed (too few images with both features).")
    absr = np.abs(R_within)
    min_eig = float(np.linalg.eigvalsh(R_within).min())

    if write:
        pd.DataFrame(R_within, index=names, columns=names).round(4).to_csv(
            P(f"correlations_within_{set_name}.csv"))
        pd.DataFrame(R_raw, index=names, columns=names).round(4).to_csv(
            P(f"correlations_raw_{set_name}.csv"))
        pd.DataFrame(N_pair, index=names, columns=names).to_csv(
            P(f"pairwise_n_{set_name}.csv"))

    Z, lab = cluster(absr, names, CLUSTER_THRESHOLD)
    sens = {t: cluster(absr, names, t)[1] for t in SENSITIVITY_THRESHOLDS}

    # --- one representative per group ---
    chosen, notes = [], {}
    for g in sorted(set(lab)):
        members = [names[i] for i in np.where(lab == g)[0]]
        forced = [m for m in members if m in FORCED]
        if forced:
            chosen += forced
            for m in forced:
                notes[m] = "stimulus feature"
            if len(forced) > 1:
                for m in forced:
                    notes[m] = ("stimulus feature; SHARES A GROUP with "
                                + ", ".join(f for f in forced if f != m)
                                + " -- never in the same model")
            continue
        ranked = [m for m in priority if m in members] + \
                 [m for m in members if m not in priority]
        stable = [m for m in ranked if stability.get(m, 1.0) >= STABILITY_MIN]
        pick = stable[0] if stable else ranked[0]
        chosen.append(pick)
        notes[pick] = ("first in priority order" if pick == ranked[0]
                       else "first stable in priority order")
        if not stable:
            notes[pick] += " (NO member meets STABILITY_MIN)"

    # --- joint VIF check (rule in the module docstring, step 5) ---
    def vifs(feats):
        ix = [names.index(c) for c in feats]
        return dict(zip(feats, vif_from_corr(R_within[np.ix_(ix, ix)])))

    initial_vif = vifs(chosen)
    rank = {f: i for i, f in enumerate(priority)}
    dropped, trace, unresolved, step = [], [], [], 0
    while True:
        v = vifs(chosen)
        high = {c: x for c, x in v.items()
                if x > VIF_MAX and c not in unresolved}
        if not high:
            break
        step += 1
        target = max(high, key=high.get)       # the highest VIF first
        # Non-forced features, lowest priority first (features not in
        # PRIORITY count as lowest). Removing the target itself always
        # fixes it, so a non-forced target is dropped only if no
        # lower-priority feature fixes it.
        cands = sorted((c for c in chosen if c not in FORCED),
                       key=lambda c: -rank.get(c, 999))
        fix = None
        for c in cands:
            rest = [f for f in chosen if f != c]
            after = 0.0 if c == target else vifs(rest)[target]
            ok = after <= VIF_MAX
            trace.append({"step": step, "target": target,
                          "target_vif": round(high[target], 3),
                          "try_removing": c,
                          "priority": rank.get(c, 999) + 1,
                          "target_vif_after": (round(after, 3)
                                               if c != target
                                               else "removed"),
                          "fixes_target": ok,
                          "action": "DROPPED" if ok else ""})
            if ok:
                fix = c
                break
        if fix is None:
            # only possible when the target is a stimulus feature
            unresolved.append(target)
            trace.append({"step": step, "target": target,
                          "target_vif": round(high[target], 3),
                          "try_removing": "", "priority": "",
                          "target_vif_after": "", "fixes_target": False,
                          "action": "no removal fixes it: flagged"})
            continue
        chosen.remove(fix)
        dropped.append((fix, f"{target} {high[target]:.2f}"))
        notes[fix] = (f"dropped by the VIF check: lowest-priority feature "
                      f"whose removal brings {target} (VIF "
                      f"{high[target]:.2f}) to {VIF_MAX:g} or below")

    vif = np.array([vifs(chosen)[c] for c in chosen])
    if not write:
        return {"selected": chosen}
    vif_df = pd.DataFrame({
        "feature": chosen,
        "vif": vif.round(3),
        "vif_before_check": [round(float(initial_vif[c]), 3)
                             for c in chosen],
        "forced": [c in FORCED for c in chosen]})
    vif_df.to_csv(P(f"vif_{set_name}.csv"), index=False)
    pd.DataFrame(trace, columns=["step", "target", "target_vif",
                                 "try_removing", "priority",
                                 "target_vif_after", "fixes_target",
                                 "action"]
                 ).to_csv(P(f"vif_trace_{set_name}.csv"), index=False)

    # --- per-feature table ---
    rows = []
    for i, nm in enumerate(names):
        others = [names[j] for j in np.where(lab == lab[i])[0] if j != i]
        max_other = max((absr[i, names.index(o)] for o in others),
                        default=np.nan)
        rows.append({
            "feature": nm,
            f"group_{CLUSTER_THRESHOLD:g}": int(lab[i]),
            **{f"group_{t:g}": int(sens[t][i]) for t in sens},
            "selected": nm in chosen,
            "why": notes.get(nm, "represented by another member"),
            "max_abs_rho_in_group": (round(float(max_other), 3)
                                     if others else np.nan),
            "stability_256_336": round(stability.get(nm, np.nan), 4),
        })
    groups = pd.DataFrame(rows).sort_values(
        [f"group_{CLUSTER_THRESHOLD:g}", "feature"])
    groups.to_csv(P(f"groups_{set_name}.csv"), index=False)

    # groups left without a selected member (after the VIF check)
    empty = []
    for gid in sorted(set(lab)):
        members = [names[i] for i in np.where(lab == gid)[0]]
        if not any(m in chosen for m in members):
            empty.append(members)

    plot_set(Z, R_within, names, set_name)
    return {"selected": chosen, "dropped_by_vif": dropped,
            "initial_vif": {c: float(x) for c, x in initial_vif.items()},
            "unresolved_vif": unresolved, "empty_groups": empty,
            "vif": dict(zip(chosen, map(float, vif))),
            "groups": groups, "labels": lab, "sens": sens,
            "R_within": R_within, "min_eigenvalue": min_eig,
            "n_pair_min": int(N_pair.min())}


def plot_set(Z, R, names, set_name):
    """Dendrogram (from |rho|) and the SIGNED within-category rho heatmap
    in dendrogram order (blue negative, red positive, white 0)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    d = dendrogram(Z, labels=names, ax=ax, leaf_rotation=90,
                   color_threshold=1 - CLUSTER_THRESHOLD)
    for t, st in [(CLUSTER_THRESHOLD, "-")] + \
            [(t, ":") for t in SENSITIVITY_THRESHOLDS]:
        ax.axhline(1 - t, color="k", ls=st, lw=0.8)
        ax.text(ax.get_xlim()[1], 1 - t, f" |rho|={t:g}", va="center",
                fontsize=8)
    ax.set_ylabel("1 - |within-category Spearman rho|")
    ax.set_title(f"Feature groups ({set_name}); solid line = primary cut")
    fig.tight_layout()
    fig.savefig(P(f"dendrogram_{set_name}.png"), dpi=130)
    plt.close(fig)

    order = [names.index(n) for n in d["ivl"]]
    M = R[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(d["ivl"], rotation=90, fontsize=8)
    ax.set_yticklabels(d["ivl"], fontsize=8)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                    fontsize=5.5,
                    color="w" if abs(M[i, j]) >= 0.6 else "k")
    fig.colorbar(im, label="within-category Spearman rho")
    ax.set_title(f"Within-category rho ({set_name}); "
                 "groups use |rho|")
    fig.tight_layout()
    fig.savefig(P(f"correlation_heatmap_{set_name}.png"), dpi=130)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("04b_select_features.py")
    print("=" * 70)

    # --- step 04 ---
    defs = json.load(open(os.path.join(FEATURES_DIR,
                                       "feature_definitions.json")))
    names04 = defs["crop"]
    F = np.load(os.path.join(FEATURES_DIR, "features_crop.npy"),
                mmap_mode="r")
    sub = np.load(os.path.join(FEATURES_DIR, "subsample_ids.npy"))
    F336 = np.load(os.path.join(FEATURES_DIR,
                                "features_crop336_subsample.npy"))
    X04 = np.asarray(F[sub], dtype=np.float64)

    # --- step 04c: same images, same subsample ---
    xdefs = json.load(open(os.path.join(EXTRA_DIR,
                                        "extra_definitions.json")))
    if xdefs.get("version") != 2:
        raise RuntimeError(f"{EXTRA_DIR} is not step 04c version 2.")
    meta04 = json.load(open(os.path.join(FEATURES_DIR, "run_meta.json")))
    meta4c = json.load(open(os.path.join(EXTRA_DIR, "run_meta.json")))
    for k in ("index_fingerprint", "n_images"):
        if meta04.get(k) != meta4c.get(k):
            raise RuntimeError(f"Steps 04 and 04c differ in {k}.")
    sub4c = np.load(os.path.join(EXTRA_DIR, "subsample_ids.npy"))
    if not np.array_equal(sub, sub4c):
        raise RuntimeError("Steps 04 and 04c used different subsamples.")
    extra = EXTRA_CANDIDATES or xdefs["corpus_candidates"]
    xcols = xdefs["crop"]
    ex = [xcols.index(f) for f in extra]
    E = np.load(os.path.join(EXTRA_DIR, "extra_crop.npy"), mmap_mode="r")
    E336 = np.load(os.path.join(EXTRA_DIR, "extra_crop336_subsample.npy"))
    XE = np.asarray(E[sub][:, ex], dtype=np.float64)
    XE336 = np.asarray(E336[:, ex], dtype=np.float64)

    names = names04 + extra
    clash = set(names04) & set(extra)
    # PRIORITY may list 04c features that are not candidates in this run
    # (EXTRA_CANDIDATES); they are simply unused. Anything else unknown
    # is an error.
    missing = [f for f in FORCED + PRIORITY
               if f not in names and f not in xcols]
    if clash or missing:
        raise KeyError(f"Feature names: duplicated {sorted(clash)}, "
                       f"unknown in settings {missing}")
    sources = {**{f: "04" for f in names04}, **{f: "04c" for f in extra}}
    X = np.hstack([X04, XE])
    F336all = np.hstack([np.asarray(F336, np.float64), XE336])

    idx = pd.read_csv(INDEX_CSV, usecols=["image_id", "category"])
    cats = idx["category"].to_numpy()[sub]
    flags = pd.read_csv(FLAGS_CSV, usecols=["image_id", "is_greyscale"])
    grey = flags.set_index("image_id").loc[sub, "is_greyscale"] \
        .to_numpy().astype(bool)

    # Image set as in version 1: all step-04 features defined at both
    # sizes. The 04c features may be undefined (pairwise correlations).
    ok = np.isfinite(X04).all(1) & np.isfinite(F336).all(1)
    print(f"Subsample: {len(sub):,} images, {len(np.unique(cats))} "
          f"categories; greyscale {grey.sum():,}; incomplete rows "
          f"(step 04) {(~ok).sum()}")
    print(f"Candidates: {len(names04)} from step 04 + {len(extra)} from "
          f"04c ({', '.join(extra)})")
    nan_rows = []
    for j, nm in enumerate(extra):
        v = XE[:, j]
        nan_rows.append({"feature": nm,
                         "pct_undefined_colour_only":
                             100 * float(np.isnan(v[ok & ~grey]).mean()),
                         "pct_undefined_all_images":
                             100 * float(np.isnan(v[ok]).mean())})
    nan_df = pd.DataFrame(nan_rows)

    # --- Stability 256 vs 336 (colour images, and all) ---
    stab_rows, stability = [], {}
    for j, nm in enumerate(names):
        both = ok & np.isfinite(X[:, j]) & np.isfinite(F336all[:, j])
        r_all = np.corrcoef(rankdata(X[both, j]),
                            rankdata(F336all[both, j]))[0, 1]
        m = both & ~grey
        r_col = np.corrcoef(rankdata(X[m, j]),
                            rankdata(F336all[m, j]))[0, 1]
        stab_rows.append({"feature": nm, "spearman_all": round(r_all, 4),
                          "spearman_colour_only": round(r_col, 4)})
        stability[nm] = r_col if PRIMARY_SET == "colour_only" else r_all
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(P("stability_256_vs_336.csv"), index=False)

    # --- Both selections ---
    sets = {"colour_only": ok & ~grey, "all_images": ok}
    res = {}
    for s, m in sets.items():
        print(f"\n--- {s}: {m.sum():,} images ---")
        res[s] = select(X[m], cats[m], names, stability, s)
        print("  selected:", ", ".join(res[s]["selected"]))

    # --- Reproduction check: version 1 on the same images ---
    repro = {}
    if PREVIOUS_SELECTION and os.path.exists(PREVIOUS_SELECTION):
        prev = json.load(open(PREVIOUS_SELECTION))
        i04 = [names.index(f) for f in names04]
        for s, m in sets.items():
            r1 = select(X[m][:, i04], cats[m], names04, stability, s,
                        priority=PRIORITY_V1, write=False)["selected"]
            repro[s] = sorted(r1) == sorted(prev["sets"][s])
        print("\nReproduction of version 1: "
              + ", ".join(f"{s} {'PASS' if v else 'FAIL'}"
                          for s, v in repro.items()))
        if not all(repro.values()):
            raise RuntimeError("The version-1 selection is not reproduced "
                               "from the same images; inputs changed.")
        prev_sets = prev["sets"]
    else:
        print("\nReproduction of version 1: skipped (no "
              "PREVIOUS_SELECTION)")
        prev_sets = None

    same = set(res["colour_only"]["selected"]) == \
        set(res["all_images"]["selected"])
    out = {
        "primary": PRIMARY_SET,
        "sets": {s: res[s]["selected"] for s in res},
        "dropped_by_vif": {s: [f for f, _ in res[s]["dropped_by_vif"]]
                           for s in res},
        "same_selection_in_both_sets": same,
        "forced": FORCED,
        "version": 2,
        "feature_sources": {f: sources[f]
                            for s in res for f in res[s]["selected"]},
        "feature_dirs": {"04": FEATURES_DIR, "04c": EXTRA_DIR},
        "extra_candidates": extra,
        "reproduces_version_1": repro or None,
        "settings": {"cluster_threshold": CLUSTER_THRESHOLD,
                     "sensitivity_thresholds": SENSITIVITY_THRESHOLDS,
                     "linkage": LINKAGE_METHOD, "vif_max": VIF_MAX,
                     "stability_min": STABILITY_MIN,
                     "priority": PRIORITY},
        "n_images": {s: int(m.sum()) for s, m in sets.items()},
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "note": "Chosen from feature-feature correlations only; the "
                "CLIP axes were not used.",
    }
    json.dump(out, open(P("selected_features.json"), "w"), indent=2)

    # --- Report ---
    lines = [
        "04b_select_features.py -- summary",
        "=" * 60,
        f"Date            : {out['date']}",
        f"Subsample       : {len(sub):,} images ({len(np.unique(cats))} "
        f"categories); colour-only {sets['colour_only'].sum():,}",
        f"Rule            : average-linkage groups at |rho| >= "
        f"{CLUSTER_THRESHOLD:g} (within-category Spearman); "
        f"VIF <= {VIF_MAX:g} (drop the lowest-priority feature that "
        f"fixes it)",
        f"Stimulus (kept) : {', '.join(FORCED)}",
        f"Candidates      : {len(names04)} (step 04) + {len(extra)} "
        f"(step 04c: {', '.join(extra)})",
        "Reproduction of version 1: "
        + (", ".join(f"{s} {'PASS' if v else 'FAIL'}"
                     for s, v in repro.items()) if repro else "skipped"),
        "The CLIP axes were not used.",
        "",
        "Undefined values of the 04c candidates (% of the images used):",
    ]
    for rr in nan_df.itertuples():
        lines.append(f"  {rr.feature:18s} colour-only "
                     f"{rr.pct_undefined_colour_only:5.2f}%   all images "
                     f"{rr.pct_undefined_all_images:5.2f}%")
    lines.append("")
    for s in res:
        r = res[s]
        g = r["groups"]
        gcol = f"group_{CLUSTER_THRESHOLD:g}"
        lines += [f"=== {s} ({sets[s].sum():,} images)"
                  + ("  [PRIMARY]" if s == PRIMARY_SET else ""), ""]
        lines.append(f"Groups at |rho| >= {CLUSTER_THRESHOLD:g}:")
        for gid, gg in g.groupby(gcol):
            mem = ", ".join(
                (f"*{f}*" if sel else f)
                for f, sel in zip(gg.feature, gg.selected))
            lines.append(f"  {gid:2d}: {mem}")
        lines.append("  (*selected*)")
        for t in SENSITIVITY_THRESHOLDS:
            n_t = len(set(r["sens"][t]))
            lines.append(f"  number of groups at {t:g}: {n_t} "
                         f"(vs {len(set(r['labels']))} at "
                         f"{CLUSTER_THRESHOLD:g})")
        lines.append("")
        lines.append("Selected features (VIF within category; before the "
                     "VIF check -> final):")
        for f, v in r["vif"].items():
            lines.append(f"  {f:18s} VIF {r['initial_vif'][f]:5.2f} -> "
                         f"{v:5.2f}"
                         + ("  [stimulus]" if f in FORCED else "")
                         + ("  [04c]" if sources[f] == "04c" else ""))
        if prev_sets is not None:
            new = [f for f in r["selected"] if f not in prev_sets[s]]
            gone = [f for f in prev_sets[s] if f not in r["selected"]]
            lines.append(f"Compared with version 1: added "
                         f"{', '.join(new) or 'none'}; removed "
                         f"{', '.join(gone) or 'none'}")
        lines.append(f"Correlation matrix: smallest eigenvalue "
                     f"{r['min_eigenvalue']:.3f} (must be > 0; pairwise "
                     f"correlations), fewest images behind a correlation "
                     f"{r['n_pair_min']:,}")
        if r["unresolved_vif"]:
            lines.append("NOTE: above VIF_MAX and no removal of a "
                         "non-stimulus feature fixes it: "
                         + ", ".join(r["unresolved_vif"]))
        for members in r["empty_groups"]:
            lines.append("NOTE: group without a selected member: "
                         + ", ".join(members))
        sel = r["selected"]
        ix = [names.index(f) for f in sel]
        Rs = r["R_within"][np.ix_(ix, ix)]
        lines.append("")
        lines.append("Within-category rho among the selected features "
                     "(signed):")
        w = max(len(f) for f in sel)
        lines.append(" " * (w + 4) + " ".join(f"{k + 1:>5d}"
                                             for k in range(len(sel))))
        for k, f in enumerate(sel):
            lines.append(f"  {k + 1:2d} {f:{w}s} "
                         + " ".join(f"{Rs[k, m]:+5.2f}" if m != k
                                    else "    ." for m in range(len(sel))))
        unstable = [f for f in r["selected"]
                    if stability.get(f, 1.0) < STABILITY_MIN]
        if unstable:
            lines.append("NOTE: selected but below STABILITY_MIN "
                         "(256 vs 336): " + ", ".join(unstable))
        high = [f for f, v in r["vif"].items() if v > VIF_MAX]
        if high:
            lines.append("NOTE: stimulus features above VIF_MAX: "
                         + ", ".join(high))
        for f, why in r["dropped_by_vif"]:
            lines.append(f"Dropped by the VIF check: {f} (fixes {why}); "
                         f"see vif_trace_{s}.csv")
        shared = g[g.why.str.contains("SHARES A GROUP")]
        if len(shared):
            lines.append("NOTE: stimulus features sharing a group: "
                         + ", ".join(shared.feature))
        lines.append("")
    lines += [
        f"Same selection in both sets: {'YES' if same else 'NO'}",
        "",
        "Stability 256 vs 336 px (Spearman, colour-only):",
    ]
    for _, rr in stab.sort_values("spearman_colour_only").iterrows():
        flag = "" if rr.spearman_colour_only >= STABILITY_MIN else \
            f"  < {STABILITY_MIN:g}"
        lines.append(f"  {rr.feature:18s} {rr.spearman_colour_only:.4f}"
                     f"{flag}")
    lines += ["", f"Elapsed: {time.time() - t0:.0f}s"]
    report = "\n".join(lines)
    open(P("selection_report.txt"), "w").write(report + "\n")
    print("\n" + report)


if __name__ == "__main__":
    main()
