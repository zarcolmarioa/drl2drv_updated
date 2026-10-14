"""
09a_axis_checks_v4.py  (pipeline step 09, part a, v4 run: checks on whole axes)

Does the within-category result of step 07 hold up? Every check below
re-estimates 07's within-category model under one change and reports how
much the result moves. Nothing is changed or selected by these checks.

v4 run (v4_protocol.md 1.18)
----------------------------
The same checks as version 3 (below), run on the nine v4 axes
(N1a: v4 only; the v3b results are in data/checks_v3), with the same
12 features (feature_selection_v3, N2), on the scores of
06_project_axes_v4.py (data/scores_v4) and the axes of 05_build_axes_v4.py
(data/axes_v4). Output: data/checks_v4.
  * Targets: every included v4 main axis. C1 is run for v4 only (version
    3 also ran v2); C13 holds the other EIGHT v4 axes constant; C14 and
    C16 are computed for v4.
  * Not run: C2 (no v4 sub-axes before Stage T, S2b) and C5 (v2 only).
  * C3: the v4 template versions (photo, place, world); C4: v4's centroid
    and "a scene" baselines; C8: A07 (fog); C10: the v4 axes.
  * NEW C17 PAIR OVERLAP (decision 13): for every pair of v4 axes whose
    within-category correlation is at least OVERLAP_R, each axis's links
    with the other axis held constant. The pairs are chosen from the point
    estimate of the within-category correlation, computed before the main
    evaluation. Also the text half and image half of the merge test
    (text cosine from 05, image r from C11, both >= MERGE_MIN). REPORT
    ONLY: nothing is merged (N4).
  * NEW C18 v3-v4 COMPARISON (protocol 6.5, fixed before any v4 result):
    within-category correlations between every v4 and v3b axis (with
    intervals); for every v3b axis its COUNTERPART (each the other's
    highest correlation, r >= COUNTERPART_R), SPLIT (two v4 axes with
    r >= COUNTERPART_R that both have it as their best match) or none;
    and whether each v3b link REPLICATES on its counterpart: same sign,
    |beta| >= LINK_MIN and an interval excluding 0, alone and with all
    controls (C1). The v3b estimates are computed here with the same
    resamples, and checked against data/checks_v3 (check R1). REPORT
    ONLY (N4).
  * NEW in the axis summary: whether each link also survives without the
    darkness and enclosure directions (R2b: the v4@nocontrol axis).

Version 3 (28 Sep)
------------------
  * New check C16 (poles): the overlap between the axes (C11) and the
    general factor (C14) recomputed with each axis's POSITIVE anchors
    only and with its NEGATIVE anchors only. Tests the hypothesis from
    09b that the "ordinary reality" negatives (real, accurate, usual,
    natural, ...) shared in meaning across axes produce much of the
    overlap. Output folder data/checks_v3; everything else unchanged.

Version 2 (28 Sep)
------------------
  * Features: the selection in data/feature_selection_v3 (12 features:
    version 1's 10 plus hue_spread and cast_b from step 04c), read from
    both folders as in 07.
  * ALL EIGHT v3b axes are targets (version 1: lifeless and foggy only).
  * New checks C13 (specificity), C14 (overlap with features and
    controls removed, and the principal components of the axes) and C15
    (hue_spread and toned photographs); mean |r| added to C11.
  * Each evaluation uses the images with every variable IT uses defined
    (version 1: every variable loaded). hue_spread is undefined for
    greyscale photographs, so C6 (greyscale) uses the selected features
    that greyscale photographs have (all but hue_spread) on both image
    sets.
  * The report starts with an AXIS SUMMARY and lists, for every axis, its
    strongest links (REPORT_TOP per axis) plus KEY_PAIRS.

The model (as in 07)
--------------------
Every variable minus its category mean; then the axis score regressed on
the selected features (standardised coefficients, "beta"). Where a check
adds covariates (control scores, other axes, lum_skew, dark channel)
they enter the same regression. "rho" is 07's one-feature-at-a-time
within-category Spearman correlation. Intervals: 95% from resampling
whole categories (N_BOOT); the same resamples are used for every check,
so differences between checks are not resampling noise.

Checks
------
  C1 CONTROLS      the control axes added as covariates, one at a time,
                   all together, and all except greyscale (greyscale
                   photos genuinely lack colour, so that control can
                   absorb real colour information; both are reported).
  C2 SUB-AXES      lifeless literal / non-literal sub-axes (v3a, v3b;
                   v3b-neg non-literal): alone, with the greyscale and
                   mood controls, and with all controls.
  C3 CAPTIONS      the same model with the "photo", "place" and "world"
                   caption versions of each v3 version.
  C4 BASELINES     positives against the caption centroid (preferred)
                   and against "a scene" (reference only).
  C5 NO BRIGHT     v2 foggy with and without "bright".
  C6 GREYSCALE     colour_only vs all_images, with and without the
                   greyscale control (features defined for greyscale
                   photographs only).
  C7 LUM_SKEW      lum_skew added to the features (exploratory).
  C8 DARK CHANNEL  foggy on dark_channel_mean with lum_mean and sat_mean,
                   and with the features: does foggy follow anything
                   haze-specific beyond brightness and saturation?
  C9 256 vs 336    the model on 04's subsample with features measured at
                   256 px and at 336 px (CLIP's own resolution).
  C10 CROP vs STRETCH  within rho with the old 11 features measured the
                   old way (stretched whole image) and on CLIP's crop.
  C11 OVERLAP      within-category correlations between the axes' image
                   scores (review Y2), with intervals for the primary set;
                   mean r (signed) and mean |r| per version.
  C12 DUPLICATES   images with identical embeddings (the same picture
                   stored more than once, often in two categories);
                   the results again with one copy of each kept.
  C13 SPECIFICITY  each v3b axis on the features with the OTHER seven v3b
                   axes as covariates, and with the other axes and all
                   controls. A link that survives belongs to that axis,
                   not to what the axes share (e.g. the lifeless / foggy
                   / unfamiliar cluster). Strict: the other axes also
                   carry part of the axis's own meaning.
  C14 OVERLAP ADJUSTED  the v3b overlap (C11) as partial correlations,
                   with the features removed, the controls removed, and
                   both. Shows how much of the overlap is shared image
                   properties and how much the generic "dull / gloomy"
                   direction. Also the principal components of each
                   correlation matrix: the share of the first component
                   says how strongly one general factor runs through all
                   eight axes, and its loadings which axes carry it.
  C15 HUE_SPREAD   hue_spread's low end contains many TONED photographs
                   (sepia, duotone: almost no colour, all of one hue),
                   a vintage style. Its betas with (a) the age control,
                   (b) toned photographs left out, (c) both. Toned =
                   hue_spread < TONED_HUE_MAX and chroma_mean <
                   TONED_CHROMA_MAX (a working rule for this check; the
                   contact sheet toned_examples.png shows what it
                   catches).
  C16 POLES        for every set in POLE_SETS, the within-category
                   correlations between the axes built three ways:
                     main      positives vs negatives (= C11);
                     positive  positives vs the vocabulary centroid c
                               (= 05's "@centroid" axis, from 06);
                     negative  c vs the negatives: unit(c - mean N),
                               built here from 05's caption embeddings
                               and projected on every image.
                   c is the mean of all captions of the set's included
                   axes, each once (as in 05). Every positive pole
                   contains -c and every negative pole +c, so the poles
                   of a set share one term, which by itself makes them
                   correlate. "vocabulary removed" partials out each
                   image's score on unit(c), which removes that shared
                   term exactly: that row is the one to compare. For
                   each: mean r, mean |r| and the first component's
                   share (intervals), and each pair. If the negative
                   poles overlap much more than the positive poles, the
                   negatives carry the general factor (G3).

Axis summary
------------
For every v3b axis: the number of LINKS in the base model (a feature
with |beta| >= LINK_MIN and an interval excluding 0), and how many keep
the same sign, |beta| >= LINK_MIN and an interval excluding 0 with all
controls (C1), with the other axes (C13), and with both.

Inputs
------
  INDEX_CSV, FLAGS_CSV, FEATURES_DIR, EXTRA_DIR, SELECTION_JSON,
  SCORES_DIR, EMB_PATH (C12 and C16) -- as in step 07.
  AXES_DIR (C16): step 05's axis_anchors.csv, captions.csv,
  caption_embeddings.npy, axis_catalog.csv, axis_vectors.npy.

Outputs (under OUTPUT_DIR)
--------------------------
  checks_report.txt          summary (start here)
  checks_results.csv         one row per check x configuration x image
                             set x axis x feature: estimate and interval
  axis_summary.csv           the axis summary
  overlap_within_<set>.csv   within-category correlation matrices (C11)
  overlap_primary_ci.csv     the primary set's pairs with intervals
  overlap_adjusted.csv       C14: every v3b pair, raw and partial
  axis_components.csv        C14: eigenvalues and loadings
  duplicate_groups.csv       image_ids with identical embeddings (C12)
  toned_images.csv           C15: images flagged as toned
  toned_examples.png         C15: a random sample of them
  poles_summary.csv          C16: mean r, mean |r|, first-component share
                             per set x poles x adjustment
  poles_overlap.csv          C16: every pair, with intervals
  poles_components.csv       C16: first-component loadings
  run_meta.json

Usage
-----
    conda activate drl2drv
    python 09a_axis_checks.py

Longer than version 1 (eight target axes; the ranks for the caption and
baseline variants of all eight are the slowest part); memory about 4-6 GB.
"""

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

INDEX_CSV       = "data/selection/image_index.csv"
FLAGS_CSV       = "data/monochrome/image_flags.csv"
FEATURES_DIR    = "data/features"
EXTRA_DIR       = "data/features_extra"          # 04c (version 2)
SELECTION_JSON  = "data/feature_selection_v3/selected_features.json"
SCORES_DIR      = "data/scores_v4"         # 06_project_axes_v4.py
EMB_PATH        = "data/embeddings/embeddings_768.npy"
AXES_DIR        = "data/axes_v4"           # 05_build_axes_v4.py (C16, C17)
OUTPUT_DIR      = "data/checks_v4"         # new folder (v3 run: data/checks_v3)
V3_CHECKS       = "data/checks_v3/checks_results.csv"   # check R1; None to skip
R1_TOL          = 1.0001e-4                # the v3 run's CSV is rounded to 4 decimals

PRIMARY_SET     = "v4"                     # the target set of this run
V3_SET          = "v3b"                    # comparison (C18)
NOCONTROL_SET   = "v4@nocontrol"           # R2b
MAIN_SETS       = [PRIMARY_SET]
TARGET_AXES     = None                     # None = every included v4 main axis
FOG_AXIS_ID     = "A07"                    # C8
CONTROLS        = ["greyscale", "mood", "quality", "age", "weather", "threat"]
EXTRA_FEATURES  = ["lum_skew", "dark_channel_mean"]      # from step 04
# key pairs: (v4 axis id, feature)
KEY_PAIRS       = [("A14", "contrast_local"), ("A14", "sat_mean"),
                   ("A13", "sat_mean"), ("A13", "contrast_local"),
                   ("A07", "contrast_local"), ("A07", "sat_mean"),
                   ("A07", "lum_p05"), ("A11", "contrast_local"),
                   ("A11", "lum_mean")]
OVERLAP_R       = 0.5      # C17: pairs analysed with the other axis held constant
MERGE_MIN       = 0.8      # decision 13: text cosine and image r (report only)
COUNTERPART_R   = 0.5      # C18 (protocol 6.5)
REPORT_TOP      = 3        # strongest links per axis shown in the report
LINK_MIN        = 0.05     # |beta| counted as a link in the axis summary

# --- C15: toned photographs ---
TONED_HUE_MAX   = 0.02     # hue_spread below: almost one hue
TONED_CHROMA_MAX = 10.0    # chroma_mean below: little colour (corpus
                           # mean ~14.6, SD ~9.1)
TONED_FIG_N     = 30
TONED_FIG_PX    = 256

# --- C16: poles ---
POLE_SETS       = [PRIMARY_SET]
POLE_CHECK_N    = 2000     # images used to check the projections against 06
POLE_MAX_DIFF   = 1e-3     # max |projection - 06's raw score|
POLE_MIN_COS    = 0.9999   # rebuilt axis vs 05's vector

MIN_IMAGES_PER_CATEGORY = 30
N_BOOT          = 1000
CI_LEVEL        = 0.95
RANDOM_SEED     = 42
DUP_CHUNK       = 100_000


# ============================================================
# Helpers (same estimators as step 07)
# ============================================================

def P(*n):
    return os.path.join(OUTPUT_DIR, *n)


def short(a):
    """'veiled / foggy' -> 'veiled'; 'A14 lifeless, dead, empty' -> 'A14 lifeless'."""
    return a.split(" / ")[0].split(",")[0]


def ci(a, level=CI_LEVEL):
    lo = (1 - level) / 2 * 100
    return np.nanpercentile(a, lo, axis=0), np.nanpercentile(a, 100 - lo, axis=0)


def cat_stats(V, codes, C):
    p = V.shape[1]
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(C + 1))
    n, m, M = np.zeros(C), np.zeros((C, p)), np.zeros((C, p, p))
    for c in range(C):
        rows = order[bounds[c]:bounds[c + 1]]
        X = np.asarray(V[rows], dtype=np.float64)
        n[c] = len(rows)
        if len(rows):
            m[c] = X.mean(0)
            Xc = X - m[c]
            M[c] = Xc.T @ Xc
    return n, m, M


def total(n, m, M, w, within):
    S = np.tensordot(w, M, axes=1)
    if within:
        return S
    N = (w * n).sum()
    mu = (w * n) @ m / N
    D = m - mu
    return S + np.einsum("c,ci,cj->ij", w * n, D, D)


def betas(S, F, A):
    Sxx = S[np.ix_(F, F)]
    B = np.linalg.solve(Sxx, S[np.ix_(F, A)])
    return B * np.sqrt(np.diag(Sxx))[:, None] / np.sqrt(np.diag(S)[A])[None, :]


def corrs(S, F, A):
    d = np.sqrt(np.diag(S))
    return S[np.ix_(F, A)] / np.outer(d[F], d[A])


def partial_corr(S, A, G):
    """Correlations among A with the variables G removed (residual
    covariance of A given G)."""
    Saa = S[np.ix_(A, A)]
    if G:
        Sag = S[np.ix_(A, G)]
        Saa = Saa - Sag @ np.linalg.solve(S[np.ix_(G, G)], Sag.T)
    d = np.sqrt(np.diag(Saa))
    return Saa / np.outer(d, d)


def upper(Rm):
    return Rm[np.triu_indices(len(Rm), 1)]


def first_share(Rm):
    """Share of the total variance on the first principal component of a
    correlation matrix."""
    ev = np.linalg.eigvalsh(Rm)
    return float(ev[-1] / ev.sum())


def vifs(S, F):
    d = np.sqrt(np.diag(S)[F])
    R = S[np.ix_(F, F)] / np.outer(d, d)
    return np.diag(np.linalg.inv(R))


def rank_cols(X):
    out = np.empty(X.shape, dtype=np.float32)
    for j in range(X.shape[1]):
        out[:, j] = rankdata(X[:, j])
    return out


def restrict(codes_all, keep):
    """Codes of the kept images, dropping small categories."""
    codes = codes_all[keep]
    cnt = np.bincount(codes, minlength=codes_all.max() + 1)
    ok = cnt >= MIN_IMAGES_PER_CATEGORY
    rows = ok[codes]
    remap = -np.ones(len(cnt), int)
    remap[np.where(ok)[0]] = np.arange(ok.sum())
    return np.where(keep)[0][rows], remap[codes[rows]], int(ok.sum())


def clip_box(w, h, n=336):
    """CLIP's crop in original pixel coordinates (same as step 04)."""
    if w <= h:
        nw, nh = n, int(n * h / w)
    else:
        nw, nh = int(n * w / h), n
    left = int(round((nw - n) / 2.0))
    top = int(round((nh - n) / 2.0))
    sx, sy = nw / w, nh / h
    return (left / sx, top / sy, (left + n) / sx, (top + n) / sy)


# ============================================================
# Generic evaluation of many configurations on one image set
# ============================================================

def evaluate(label, V, names, rows, codes, C, beta_cfg, rho_cfg,
             overlap_groups, rng, n_boot=N_BOOT, partial_cfg=()):
    """V: (N, p) variables; rows: images used; codes: their categories.
    beta_cfg: list of (check, config, axis_names, predictor_names,
    report_names). rho_cfg: list of (check, config, axis_names,
    feature_names). overlap_groups: {name: [axis names]}. partial_cfg:
    list of (name, axis_names, given_names). Only the variables used here
    are loaded, and an image is used if all of THEM are defined."""
    used = set()
    for _, _, ax, pr, _ in beta_cfg:
        used |= set(ax) | set(pr)
    for _, _, ax, fs in rho_cfg:
        used |= set(ax) | set(fs)
    for ax in overlap_groups.values():
        used |= set(ax)
    for _, ax, gv in partial_cfg:
        used |= set(ax) | set(gv)
    pos = {k: i for i, k in enumerate(names)}
    cols = sorted(pos[u] for u in used)
    ix = {names[c]: i for i, c in enumerate(cols)}
    X = np.asarray(V[rows[:, None], np.asarray(cols)[None, :]])
    good = np.isfinite(X).all(1)
    X, cd = X[good], codes[good]
    n_left = int((~good).sum())
    print(f"    [{label}] {len(X):,} images used ({n_left:,} with an "
          f"undefined value left out), {C} categories")
    n, m, M = cat_stats(X, cd, C)
    need_rank = sorted({ix[a] for _, _, ax, fs in rho_cfg for a in ax + fs})
    if need_rank:
        Xw = X[:, need_rank] - m[cd][:, need_rank]
        R = rank_cols(Xw)
        del Xw
        nr, mr, Mr = cat_stats(R, cd, C)
        del R
        rix = {j: k for k, j in enumerate(need_rank)}
    w1 = np.ones(C)
    out = []

    def run(S, Sr):
        res = {}
        for chk, cfg, ax, pr, rep in beta_cfg:
            F = [ix[p] for p in pr]
            A = [ix[a] for a in ax]
            try:
                res[("b", chk, cfg)] = betas(S, F, A)
            except np.linalg.LinAlgError:
                res[("b", chk, cfg)] = np.full((len(F), len(A)), np.nan)
        for chk, cfg, ax, fs in rho_cfg:
            F = [rix[ix[f]] for f in fs]
            A = [rix[ix[a]] for a in ax]
            res[("r", chk, cfg)] = corrs(Sr, F, A)
        for g, ax in overlap_groups.items():
            A = [ix[a] for a in ax]
            Rg = corrs(S, A, A)
            res[("o", g)] = Rg
            res[("om", g)] = np.array([upper(Rg).mean(),
                                       np.abs(upper(Rg)).mean()])
        for nm, ax, gv in partial_cfg:
            A = [ix[a] for a in ax]
            G = [ix[g] for g in gv]
            try:
                Rp = partial_corr(S, A, G)
                res[("p", nm)] = Rp
                res[("pm", nm)] = np.array([np.abs(upper(Rp)).mean(),
                                            first_share(Rp), upper(Rp).mean()])
            except np.linalg.LinAlgError:
                res[("p", nm)] = np.full((len(A), len(A)), np.nan)
                res[("pm", nm)] = np.full(3, np.nan)
        return res

    S0 = total(n, m, M, w1, True)
    Sr0 = total(nr, mr, Mr, w1, False) if need_rank else None
    point = run(S0, Sr0)
    boots = {k: np.zeros((n_boot,) + v.shape) for k, v in point.items()}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, C, C), minlength=C).astype(float)
        Sb = total(n, m, M, w, True)
        Srb = total(nr, mr, Mr, w, False) if need_rank else None
        rb = run(Sb, Srb)
        for k in boots:
            boots[k][b] = rb[k]
    for chk, cfg, ax, pr, rep in beta_cfg:
        k = ("b", chk, cfg)
        lo, hi = ci(boots[k])
        F = [ix[p] for p in pr]
        vf = vifs(S0, F)
        for j, a in enumerate(ax):
            for i, p in enumerate(pr):
                if p not in rep:
                    continue
                out.append({"check": chk, "config": cfg, "image_set": label,
                            "axis": a, "feature": p, "stat": "beta",
                            "value": point[k][i, j], "lo": lo[i, j],
                            "hi": hi[i, j], "vif": vf[i],
                            "n_images": int(len(X)), "n_categories": C})
    for chk, cfg, ax, fs in rho_cfg:
        k = ("r", chk, cfg)
        lo, hi = ci(boots[k])
        for j, a in enumerate(ax):
            for i, f in enumerate(fs):
                out.append({"check": chk, "config": cfg, "image_set": label,
                            "axis": a, "feature": f, "stat": "rho",
                            "value": point[k][i, j], "lo": lo[i, j],
                            "hi": hi[i, j], "vif": np.nan,
                            "n_images": int(len(X)), "n_categories": C})
    extras = {}
    for g in overlap_groups:
        lo, hi = ci(boots[("o", g)])
        mlo, mhi = ci(boots[("om", g)])
        extras[("o", g)] = (point[("o", g)], lo, hi)
        extras[("om", g)] = (point[("om", g)], mlo, mhi)
    for nm, _, _ in partial_cfg:
        lo, hi = ci(boots[("p", nm)])
        mlo, mhi = ci(boots[("pm", nm)])
        extras[("p", nm)] = (point[("p", nm)], lo, hi)
        extras[("pm", nm)] = (point[("pm", nm)], mlo, mhi)
    return out, extras


def build_poles(sets):
    """C16: for each set, the negative pole unit(c - mean N) of every
    included main axis and the unit vocabulary centroid, from step 05's
    caption embeddings. Also rebuilds the main axes and positive poles to
    check them against 05's vectors. Returns (vectors (m, 768), names,
    axes per set, check vectors {key: vector}, min cosine)."""
    anc = pd.read_csv(os.path.join(AXES_DIR, "axis_anchors.csv"))
    caps = pd.read_csv(os.path.join(AXES_DIR, "captions.csv"))
    CE = np.load(os.path.join(AXES_DIR, "caption_embeddings.npy")).astype(np.float64)
    cidx = dict(zip(caps["caption"], caps["row"]))
    acat = pd.read_csv(os.path.join(AXES_DIR, "axis_catalog.csv"))
    AV = np.load(os.path.join(AXES_DIR, "axis_vectors.npy")).astype(np.float64)
    arow = dict(zip(acat["key"], acat["row"]))

    def unit(v):
        return v / max(np.linalg.norm(v), 1e-12)

    def emb(cs):
        miss = [c for c in cs if c not in cidx]
        if miss:
            raise SystemExit(f"C16: captions not in captions.csv, e.g. {miss[:3]}")
        return CE[[cidx[c] for c in cs]]

    vecs, names, axes, check, cos = [], [], {}, {}, []
    for s in sets:
        mains = acat[(acat["set"] == s) & (acat["kind"] == "main") &
                     (acat["included"].astype(bool))]
        if not len(mains):
            raise SystemExit(f"C16: no included main axes for set {s}.")
        axes[s] = list(mains["axis"])
        cent = list(dict.fromkeys(c for k in mains["key"]
                                  for c in anc.loc[anc["key"] == k, "caption"]))
        c = emb(cent).mean(0)
        for r in mains.itertuples():
            rows = anc[anc["key"] == r.key]
            Pm = emb(rows.loc[rows["pole"] == "positive", "caption"]).mean(0)
            Nm = emb(rows.loc[rows["pole"] == "negative", "caption"]).mean(0)
            main_v, pos_v = unit(Pm - Nm), unit(Pm - c)
            cos.append(float(main_v @ unit(AV[arow[r.key]])))
            ck = f"{s}@centroid::{r.axis}"
            if ck not in arow:
                raise SystemExit(f"C16: {ck} not in 05's axis catalog.")
            cos.append(float(pos_v @ unit(AV[arow[ck]])))
            check[r.key], check[ck] = main_v, pos_v
            vecs.append(unit(c - Nm))
            names.append(f"{s}@negpole::{r.axis}")
        vecs.append(unit(c))
        names.append(f"{s}@vocab")
    return np.vstack(vecs), names, axes, check, min(cos)


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng_seed = RANDOM_SEED
    print("=" * 70)
    print("09a_axis_checks_v4.py")
    print("=" * 70)

    # --- inputs ---
    idx = pd.read_csv(INDEX_CSV, usecols=["image_id", "category", "path"])
    N = len(idx)
    cat_names = sorted(idx["category"].unique())
    codes_all = pd.Categorical(idx["category"], categories=cat_names).codes.astype(int)
    fl = pd.read_csv(FLAGS_CSV, usecols=["image_id", "is_greyscale"])
    colour = ~fl.set_index("image_id").reindex(np.arange(N))["is_greyscale"] \
        .astype(bool).to_numpy()
    defs = json.load(open(os.path.join(FEATURES_DIR, "feature_definitions.json")))
    sel = json.load(open(SELECTION_JSON))
    feats = sel["sets"][sel.get("primary", "colour_only")]
    sources = sel.get("feature_sources", {})
    crop, stretch = defs["crop"], defs["stretch"]
    Fc = np.load(os.path.join(FEATURES_DIR, "features_crop.npy"), mmap_mode="r")
    feats_new = [f for f in feats if sources.get(f, "04") == "04c"]
    E = E336 = None
    xcols = []
    if feats_new:
        xdefs = json.load(open(os.path.join(EXTRA_DIR, "extra_definitions.json")))
        m4 = os.path.join(FEATURES_DIR, "run_meta.json")
        m4c = os.path.join(EXTRA_DIR, "run_meta.json")
        if os.path.exists(m4) and os.path.exists(m4c):
            if json.load(open(m4)).get("index_fingerprint") != \
                    json.load(open(m4c)).get("index_fingerprint"):
                raise SystemExit("Steps 04 and 04c used different indexes.")
        xcols = xdefs["crop"]
        E = np.load(os.path.join(EXTRA_DIR, "extra_crop.npy"), mmap_mode="r")
        E336 = np.load(os.path.join(EXTRA_DIR, "extra_crop336_subsample.npy"))
        if E.shape[0] != N:
            raise SystemExit("04c features do not match the image index.")
        if not np.array_equal(
                np.load(os.path.join(FEATURES_DIR, "subsample_ids.npy")),
                np.load(os.path.join(EXTRA_DIR, "subsample_ids.npy"))):
            raise SystemExit("Steps 04 and 04c used different subsamples.")

    def src(f):
        return sources.get(f, "04")

    def feature_block(fn, rows=None, size=256):
        """(n, k) float32 of the named features at 256 (all images or
        rows) or 336 (the subsample), from 04 or 04c."""
        if size == 336:
            F336 = np.load(os.path.join(FEATURES_DIR,
                                        "features_crop336_subsample.npy"))
        out = []
        for f in fn:
            if src(f) == "04c":
                if f not in xcols:
                    raise SystemExit(f"{f} not in 04c's file.")
                a = E336[:, xcols.index(f)] if size == 336 else \
                    (E[:, xcols.index(f)] if rows is None
                     else E[rows, xcols.index(f)])
            else:
                if f not in crop:
                    raise SystemExit(f"{f} not in features_crop.")
                a = F336[:, crop.index(f)] if size == 336 else \
                    (Fc[:, crop.index(f)] if rows is None
                     else Fc[rows, crop.index(f)])
            out.append(np.asarray(a, dtype=np.float32))
        return np.column_stack(out)

    sc = pd.read_csv(os.path.join(SCORES_DIR, "score_catalog.csv"))
    Z = np.load(os.path.join(SCORES_DIR, "scores_z.npy"), mmap_mode="r")
    key2col = dict(zip(sc["key"], sc["col"]))
    ctrl_keys = [f"controls::{c}" for c in CONTROLS]

    # --- variables: features (selected + extras) and every axis ---
    fnames = feats + [f for f in EXTRA_FEATURES if f not in feats]
    print(f"Images {N:,}; features {len(feats)} ({len(feats_new)} from 04c: "
          f"{', '.join(feats_new) or 'none'}); loading {len(sc)} axis scores...")
    sc = sc.sort_values("col").reset_index(drop=True)
    if not (sc["col"].to_numpy() == np.arange(len(sc))).all():
        raise SystemExit("score_catalog.csv: columns are not 0..K-1.")
    Vf = feature_block(fnames)

    # --- C16: negative poles and the vocabulary centroid, on every image ---
    print("C16: building the negative poles and projecting every image...")
    Wp, pole_names, pole_axes, pole_check, pole_cos = build_poles(POLE_SETS)
    if pole_cos < POLE_MIN_COS:
        raise SystemExit(f"C16: rebuilt axes differ from step 05 (min cosine "
                         f"{pole_cos:.6f}).")
    for s in POLE_SETS:
        for a in pole_axes[s]:
            for k in (f"{s}::{a}", f"{s}@centroid::{a}"):
                if k not in key2col:
                    raise SystemExit(f"C16: {k} not in step 06's scores.")
    Emb = np.load(EMB_PATH, mmap_mode="r")
    if Emb.shape[0] != N:
        raise SystemExit("Embeddings do not match the image index.")
    ref_col = key2col[f"{POLE_SETS[0]}::{pole_axes[POLE_SETS[0]][0]}"]
    Xp = np.empty((N, len(pole_names)), np.float32)
    for a0 in range(0, N, DUP_CHUNK):
        b0 = min(a0 + DUP_CHUNK, N)
        x = np.asarray(Emb[a0:b0], dtype=np.float64) @ Wp.T
        x[~np.isfinite(np.asarray(Z[a0:b0, ref_col]))] = np.nan
        Xp[a0:b0] = x
    # check: the rebuilt main axes and positive poles reproduce 06's scores
    Rraw = np.load(os.path.join(SCORES_DIR, "scores_raw.npy"), mmap_mode="r")
    enc = np.where(np.isfinite(Xp[:, 0]))[0]
    pick = np.sort(np.random.default_rng(rng_seed).choice(
        enc, size=min(POLE_CHECK_N, len(enc)), replace=False))
    Ep = np.asarray(Emb[pick], dtype=np.float64)
    pole_diff = max(float(np.abs(Ep @ v - np.asarray(Rraw[pick, key2col[k]],
                                                     np.float64)).max())
                    for k, v in pole_check.items())
    print(f"  rebuilt axes vs 05: min cosine {pole_cos:.6f}; projections vs "
          f"06: max |difference| {pole_diff:.2e}")
    if pole_diff > POLE_MAX_DIFF:
        raise SystemExit("C16: projections differ from step 06 (wrong "
                         "embedding file?).")
    del Ep

    V = np.hstack([Vf, np.asarray(Z, dtype=np.float32), Xp])
    names = fnames + list(sc["key"]) + pole_names
    del Xp

    # features that greyscale photographs have (for C6)
    grey_nan = np.isnan(Vf[~colour]).mean(0) if (~colour).any() \
        else np.zeros(len(fnames))
    feats_c6 = [f for f in feats if grey_nan[fnames.index(f)] < 0.5]
    del Vf

    def K(s, a):
        return f"{s}::{a}"

    def exists(k):
        return k in key2col

    # --- configurations ---
    beta_cfg, rho_cfg, partial_cfg = [], [], []
    rep = feats
    axes_t = TARGET_AXES or [k.split("::", 1)[1] for k in sc["key"]
                             if k.startswith(f"{PRIMARY_SET}::")]
    tgt = [K(PRIMARY_SET, a) for a in axes_t if exists(K(PRIMARY_SET, a))]
    if not tgt:
        raise SystemExit(f"No {PRIMARY_SET} axes in {SCORES_DIR}.")

    def by_id(aid):
        """the v4 target whose name starts with the axis id (e.g. 'A14')."""
        m_ = [t for t in tgt if t.split("::", 1)[1].startswith(aid + " ")]
        return m_[0] if m_ else None
    tgt_v3 = [k for k in sc["key"] if k.startswith(f"{V3_SET}::")]
    tgt_nc = [K(NOCONTROL_SET, t.split("::", 1)[1]) for t in tgt
              if exists(K(NOCONTROL_SET, t.split("::", 1)[1]))]
    # C1 controls (v4); the v3b base and all-controls models for C18
    for ax in (tgt,):
        tag = ax[0].split("::")[0]
        beta_cfg.append(("C1", f"{tag}: base", ax, feats, rep))
        for c in ctrl_keys:
            beta_cfg.append(("C1", f"{tag}: + {c.split('::')[1]}", ax,
                             feats + [c], rep + [c]))
        beta_cfg.append(("C1", f"{tag}: + all controls", ax, feats + ctrl_keys,
                         rep + ctrl_keys))
        nog = [c for c in ctrl_keys if not c.endswith("greyscale")]
        beta_cfg.append(("C1", f"{tag}: + all but greyscale", ax, feats + nog,
                         rep + nog))
    # C18: v3b axes, base and all controls (the same resamples)
    beta_cfg.append(("C18", f"{V3_SET}: base", tgt_v3, feats, rep))
    beta_cfg.append(("C18", f"{V3_SET}: + all controls", tgt_v3, feats + ctrl_keys,
                     rep))
    # R2b: the nocontrol version of every v4 axis
    if tgt_nc:
        beta_cfg.append(("R2B", "base", tgt_nc, feats, rep))
    # (C2 sub-axes: not run for v4, S2b)
    # C3 captions, C4 baselines (C5: v2 only, not run)
    cap = [k for k in (K(f"{PRIMARY_SET}{t}", a.split("::", 1)[1]) for t in
                       ("", "@photo", "@place", "@world") for a in tgt) if exists(k)]
    beta_cfg.append(("C3", "base", cap, feats, rep))
    rho_cfg.append(("C3", "base", cap, feats))
    base = [k for k in (K(f"{PRIMARY_SET}{t}", a.split("::", 1)[1])
                        for t in ("", "@centroid", "@neutral") for a in tgt)
            if exists(k)]
    beta_cfg.append(("C4", "base", base, feats, rep))
    rho_cfg.append(("C4", "base", base, feats))
    # C6 greyscale: features defined for greyscale photographs, both sets
    c6_cfg = [("C6", f"{PRIMARY_SET}: base", tgt, feats_c6, feats_c6),
              ("C6", f"{PRIMARY_SET}: + greyscale", tgt,
               feats_c6 + ["controls::greyscale"], feats_c6)]
    beta_cfg += c6_cfg
    # C7 lum_skew
    prim_all = [k for k in sc["key"] if k.startswith(f"{PRIMARY_SET}::")]
    beta_cfg.append(("C7", "features", prim_all, feats, rep))
    beta_cfg.append(("C7", "+ lum_skew", prim_all, feats + ["lum_skew"],
                     rep + ["lum_skew"]))
    # C8 dark channel
    fog = [k for k in [by_id(FOG_AXIS_ID)] if k]
    beta_cfg.append(("C8", "dark + lum_mean + sat_mean", fog,
                     ["dark_channel_mean", "lum_mean", "sat_mean"],
                     ["dark_channel_mean", "lum_mean", "sat_mean"]))
    beta_cfg.append(("C8", "features + dark", fog, feats + ["dark_channel_mean"],
                     rep + ["dark_channel_mean"]))
    rho_cfg.append(("C8", "base", fog, ["dark_channel_mean"]))
    # C13 specificity: each target with the other targets as covariates
    for a in tgt:
        others = [t for t in tgt if t != a]
        beta_cfg.append(("C13", f"+ other axes | {a}", [a], feats + others, rep))
        beta_cfg.append(("C13", f"+ other axes + controls | {a}", [a],
                         feats + others + ctrl_keys, rep))
    # C14 overlap adjusted (primary targets)
    partial_cfg += [("raw", tgt, []), ("features", tgt, feats),
                    ("controls", tgt, ctrl_keys),
                    ("both", tgt, feats + ctrl_keys)]
    # C16 poles (separate evaluation: its matrices differ in size)
    c16_cfg = []
    for s in POLE_SETS:
        axs = pole_axes[s]
        voc = f"{s}@vocab"
        for kind, ks in (("main", [K(s, a) for a in axs]),
                         ("positive", [K(f"{s}@centroid", a) for a in axs]),
                         ("negative", [K(f"{s}@negpole", a) for a in axs])):
            for adj_, gv in (("nothing", []), ("vocabulary", [voc]),
                             ("vocabulary + features", [voc] + feats)):
                c16_cfg.append((f"{s}|{kind}|{adj_}", ks, gv))
    # rho for the targets and controls (reference)
    rho_cfg.append(("REF", "base", tgt + ctrl_keys, feats))
    # C11 overlap groups; C18: v4 and v3b together
    groups = {s: [k for k in sc["key"] if k.startswith(f"{s}::")] for s in MAIN_SETS}
    groups["controls+primary"] = [k for k in sc["key"]
                                  if k.startswith(f"{PRIMARY_SET}::")] + ctrl_keys
    groups["v4+v3b"] = tgt + tgt_v3

    # C17: v4 pairs with within-category |r| >= OVERLAP_R (point estimate,
    # colour_only, the same images and categories as the main evaluation)
    rows, codes, C = restrict(codes_all, colour)
    cols17 = [names.index(t) for t in tgt]
    X17 = np.asarray(V[rows[:, None], np.asarray(cols17)[None, :]], np.float64)
    ok17 = np.isfinite(X17).all(1)
    n17, m17, M17 = cat_stats(X17[ok17], codes[ok17], C)
    S17 = M17.sum(0)
    d17 = np.sqrt(np.diag(S17))
    R17 = S17 / np.outer(d17, d17)
    del X17
    pairs17 = [(tgt[i], tgt[j], R17[i, j]) for i in range(len(tgt))
               for j in range(i + 1, len(tgt)) if abs(R17[i, j]) >= OVERLAP_R]
    print(f"C17: {len(pairs17)} v4 pairs with within-category |r| >= {OVERLAP_R}: "
          + ", ".join(f"{short(a.split('::')[1])}~{short(b.split('::')[1])} {r:+.2f}"
                      for a, b, r in pairs17))
    for a, b, _ in pairs17:
        for x, y in ((a, b), (b, a)):
            beta_cfg.append(("C17", f"+ {y} | {x}", [x], feats + [y], rep))
            beta_cfg.append(("C17", f"+ {y} + controls | {x}", [x],
                             feats + [y] + ctrl_keys, rep))

    results = []
    print("Main evaluation (colour_only)...")
    main_cfg = [c for c in beta_cfg if c[0] != "C6"]
    r, extras = evaluate("colour_only", V, names, rows, codes, C, main_cfg,
                         rho_cfg, groups, np.random.default_rng(rng_seed),
                         partial_cfg=partial_cfg)
    results += r
    print("C16: overlap of the positive and negative poles...")
    _, extras16 = evaluate("colour_only", V, names, rows, codes, C, [], [], {},
                           np.random.default_rng(rng_seed), partial_cfg=c16_cfg)
    print("C6: greyscale (both image sets)...")
    for label, mask in (("colour_only", colour), ("all_images", np.ones(N, bool))):
        rows, codes, C = restrict(codes_all, mask)
        r, _ = evaluate(label, V, names, rows, codes, C, c6_cfg, [], {},
                        np.random.default_rng(rng_seed))
        results += r

    # --- C15 toned photographs ---
    toned = np.zeros(N, bool)
    have_hue = "hue_spread" in feats
    if have_hue:
        print("C15: hue_spread and toned photographs...")
        hue = feature_block(["hue_spread"])[:, 0]
        chroma = feature_block(["chroma_mean"])[:, 0]
        toned = colour & np.isfinite(hue) & (hue < TONED_HUE_MAX) & \
            (chroma < TONED_CHROMA_MAX)
        tid = np.where(toned)[0]
        pd.DataFrame({"image_id": tid, "category": idx["category"].to_numpy()[tid],
                      "hue_spread": hue[tid], "chroma_mean": chroma[tid]}) \
            .to_csv(P("toned_images.csv"), index=False, float_format="%.5g")
        age = "controls::age"
        c15 = [("C15", "base", tgt, feats, rep),
               ("C15", "+ age", tgt, feats + [age], rep)]
        for label, mask in (("colour_only", colour),
                            ("colour_only_no_toned", colour & ~toned)):
            rows, codes, C = restrict(codes_all, mask)
            r, _ = evaluate(label, V, names, rows, codes, C, c15, [], {},
                            np.random.default_rng(rng_seed))
            results += r
        # contact sheet
        if len(tid):
            pick = np.random.default_rng(rng_seed).choice(
                tid, size=min(TONED_FIG_N, len(tid)), replace=False)
            nc = 6
            nr_ = int(np.ceil(len(pick) / nc))
            fig, axs = plt.subplots(nr_, nc, figsize=(2.2 * nc, 2.5 * nr_),
                                    squeeze=False)
            for a_ in axs.ravel():
                a_.axis("off")
            for k, i in enumerate(pick):
                a_ = axs.ravel()[k]
                try:
                    with Image.open(idx["path"].iat[i]) as im:
                        im = im.convert("RGB")
                        a_.imshow(im.resize((TONED_FIG_PX, TONED_FIG_PX),
                                            Image.BILINEAR,
                                            box=clip_box(*im.size)))
                except Exception as e:
                    a_.text(0.5, 0.5, f"cannot open\n{e}", ha="center",
                            va="center", fontsize=6, color="red",
                            transform=a_.transAxes)
                a_.set_title(f"#{i} {idx['category'].iat[i][:16]}\n"
                             f"hue {hue[i]:.4f}  chroma {chroma[i]:.1f}",
                             fontsize=6, loc="left")
            fig.suptitle(f"C15: {len(pick)} random images flagged as toned "
                         f"(hue_spread < {TONED_HUE_MAX}, chroma_mean < "
                         f"{TONED_CHROMA_MAX}; {len(tid):,} flagged)",
                         fontsize=9, x=0.01, ha="left")
            fig.tight_layout()
            fig.savefig(P("toned_examples.png"), dpi=130)
            plt.close(fig)

    # --- C12 duplicates ---
    print("C12: exact duplicates (identical embeddings)...")
    hashes = np.empty(N, dtype=object)
    for a in range(0, N, DUP_CHUNK):
        b = min(a + DUP_CHUNK, N)
        blk = np.ascontiguousarray(Emb[a:b])
        for i in range(b - a):
            hashes[a + i] = hashlib.blake2b(blk[i].tobytes(), digest_size=16).digest()
    hs = pd.Series(hashes)
    dup_mask = hs.duplicated(keep=False).to_numpy()
    groups_d = hs[dup_mask].groupby(hs[dup_mask]).groups
    rows_d = []
    for gi, (h, members) in enumerate(groups_d.items()):
        mem = sorted(int(x) for x in members)
        if not all(np.array_equal(Emb[mem[0]], Emb[j]) for j in mem[1:]):
            continue
        cats = idx["category"].iloc[mem].tolist()
        for j, c in zip(mem, cats):
            rows_d.append({"group": gi, "image_id": j, "category": c,
                           "keep": j == mem[0], "cross_category": len(set(cats)) > 1})
    dup = pd.DataFrame(rows_d)
    dup.to_csv(P("duplicate_groups.csv"), index=False)
    drop = np.zeros(N, bool)
    if len(dup):
        drop[dup.loc[~dup["keep"], "image_id"].to_numpy()] = True
        rows, codes, C = restrict(codes_all, colour & ~drop)
        dc = [(("C12",) + c[1:]) for c in beta_cfg if c[0] == "C1" and
              c[1].endswith("base")]
        dr = [(("C12",) + c[1:]) for c in rho_cfg if c[0] == "REF"]
        r, _ = evaluate("colour_only_nodup", V, names, rows, codes, C, dc, dr, {},
                        np.random.default_rng(rng_seed))
        results += r

    # --- C9 256 vs 336 (subsample) ---
    print("C9: 256 vs 336 px on the subsample...")
    sub = np.load(os.path.join(FEATURES_DIR, "subsample_ids.npy"))
    ax9 = list(prim_all)
    Zs = np.asarray(Z[sub][:, [key2col[k] for k in ax9]], np.float32)
    for lab, Fx in (("256", feature_block(feats, rows=sub)),
                    ("336", feature_block(feats, size=336))):
        Vs = np.hstack([Fx, Zs])
        nm = feats + ax9
        cs = codes_all[sub]
        rws = np.where(colour[sub])[0]
        cnt = np.bincount(cs[rws], minlength=len(cat_names))
        okc = cnt >= 30
        rws = rws[okc[cs[rws]]]
        remap = -np.ones(len(cnt), int)
        remap[np.where(okc)[0]] = np.arange(okc.sum())
        r, _ = evaluate(f"subsample_{lab}", Vs, nm, rws, remap[cs[rws]],
                        int(okc.sum()), [("C9", lab, ax9, feats, feats)],
                        [("C9", lab, ax9, feats)], {},
                        np.random.default_rng(rng_seed))
        results += r

    # --- C10 crop vs stretch (old 11 features) ---
    print("C10: crop vs stretch...")
    Fs = np.load(os.path.join(FEATURES_DIR, "features_stretch.npy"), mmap_mode="r")
    ax10 = [k for k in sc["key"] if k.startswith(f"{PRIMARY_SET}::")]
    Vs = np.hstack([np.asarray(Fs, np.float32),
                    np.asarray(Fc[:, [crop.index(f) for f in stretch]], np.float32),
                    V[:, [names.index(k) for k in ax10]]])
    nm = [f"stretch:{f}" for f in stretch] + [f"crop:{f}" for f in stretch] + ax10
    rows, codes, C = restrict(codes_all, colour)
    r, _ = evaluate("colour_only", Vs, nm, rows, codes, C, [],
                    [("C10", "stretch", ax10, [f"stretch:{f}" for f in stretch]),
                     ("C10", "crop", ax10, [f"crop:{f}" for f in stretch])], {},
                    np.random.default_rng(rng_seed), n_boot=200)
    for x in r:
        x["feature"] = x["feature"].split(":", 1)[1]
    results += r
    del Vs

    res = pd.DataFrame(results)
    res.to_csv(P("checks_results.csv"), index=False, float_format="%.4f")

    # --- C11 overlap outputs ---
    for g in groups:
        pt = extras[("o", g)][0]
        ks = groups[g]
        lab = [k.split("::")[1] if "controls" not in k else k for k in ks]
        pd.DataFrame(pt, index=lab, columns=lab).to_csv(
            P(f"overlap_within_{g.replace('+', '_')}.csv"), float_format="%.4f")
    pt, lo, hi = extras[("o", PRIMARY_SET)]
    ks = groups[PRIMARY_SET]
    ov_rows = [{"axis_a": ks[i].split("::")[1], "axis_b": ks[j].split("::")[1],
                "r_within": pt[i, j], "lo": lo[i, j], "hi": hi[i, j]}
               for i in range(len(ks)) for j in range(i + 1, len(ks))]
    ovp = pd.DataFrame(ov_rows).sort_values("r_within", key=abs, ascending=False)
    ovp.to_csv(P("overlap_primary_ci.csv"), index=False, float_format="%.4f")

    # --- C14 outputs ---
    pnames = [nm for nm, _, _ in partial_cfg]
    tl = [t.split("::")[1] for t in tgt]
    adj_rows = []
    for i in range(len(tgt)):
        for j in range(i + 1, len(tgt)):
            row = {"axis_a": tl[i], "axis_b": tl[j]}
            for nm in pnames:
                p_, l_, h_ = extras[("p", nm)]
                row[f"r_{nm}"] = p_[i, j]
                row[f"lo_{nm}"] = l_[i, j]
                row[f"hi_{nm}"] = h_[i, j]
            adj_rows.append(row)
    adj = pd.DataFrame(adj_rows)
    adj.to_csv(P("overlap_adjusted.csv"), index=False, float_format="%.4f")
    comp_rows = []
    for nm in pnames:
        Rm = extras[("p", nm)][0]
        ev, vec = np.linalg.eigh(Rm)
        order = np.argsort(ev)[::-1]
        ev, vec = ev[order], vec[:, order]
        if vec[:, 0].sum() < 0:
            vec[:, 0] = -vec[:, 0]
        for k in range(len(ev)):
            comp_rows.append({"matrix": nm, "component": k + 1,
                              "eigenvalue": ev[k], "share": ev[k] / ev.sum(),
                              **{f"loading_{short(a)}": vec[i, k] * np.sqrt(ev[k])
                                 for i, a in enumerate(tl)}})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(P("axis_components.csv"), index=False, float_format="%.4f")

    # --- C16 outputs ---
    p16, s16, l16 = [], [], []
    for nm, ks, _ in c16_cfg:
        s_, kind, adj_ = nm.split("|")
        pt, lo, hi = extras16[("p", nm)]
        mm, mlo, mhi = extras16[("pm", nm)]
        lab = [k.split("::")[1] for k in ks]
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                p16.append({"set": s_, "poles": kind, "removed": adj_,
                            "axis_a": lab[i], "axis_b": lab[j], "r": pt[i, j],
                            "lo": lo[i, j], "hi": hi[i, j]})
        s16.append({"set": s_, "poles": kind, "removed": adj_,
                    "n_axes": len(ks), "mean_r": mm[2], "mean_r_lo": mlo[2],
                    "mean_r_hi": mhi[2], "mean_abs_r": mm[0],
                    "mean_abs_r_lo": mlo[0], "mean_abs_r_hi": mhi[0],
                    "first_share": mm[1], "first_share_lo": mlo[1],
                    "first_share_hi": mhi[1]})
        ev, vec = np.linalg.eigh(pt)
        v1 = vec[:, np.argmax(ev)] * np.sqrt(ev.max())
        if v1.sum() < 0:
            v1 = -v1
        for i, a in enumerate(lab):
            l16.append({"set": s_, "poles": kind, "removed": adj_, "axis": a,
                        "loading": v1[i]})
    p16, s16, l16 = pd.DataFrame(p16), pd.DataFrame(s16), pd.DataFrame(l16)
    p16.to_csv(P("poles_overlap.csv"), index=False, float_format="%.4f")
    s16.to_csv(P("poles_summary.csv"), index=False, float_format="%.4f")
    l16.to_csv(P("poles_components.csv"), index=False, float_format="%.4f")

    # --- report helpers ---
    def get(check, config, ax, f, stat="beta", iset="colour_only"):
        q = res[(res.check == check) & (res.config == config) & (res.axis == ax) &
                (res.feature == f) & (res.stat == stat) & (res.image_set == iset)]
        return q.iloc[0] if len(q) else None

    def fmt(r_):
        return "   n/a          " if r_ is None else \
            f"{r_.value:+.2f} [{r_.lo:+.2f},{r_.hi:+.2f}]"

    def v3(r_):
        return "   n/a" if r_ is None else f"{r_.value:+6.3f}"

    def is_link(r_):
        return r_ is not None and abs(r_.value) >= LINK_MIN and \
            (r_.lo > 0 or r_.hi < 0)

    base_cfg = f"{PRIMARY_SET}: base"
    # report pairs: strongest links per axis + key pairs
    pairs = []
    for a in tgt:
        bb = [(abs(get("C1", base_cfg, a, f).value), f) for f in feats]
        for _, f in sorted(bb, reverse=True)[:REPORT_TOP]:
            pairs.append((a, f))
    for aid, f in KEY_PAIRS:
        a = by_id(aid)
        if a in tgt and f in feats and (a, f) not in pairs:
            pairs.append((a, f))
    order_ax = {a: i for i, a in enumerate(tgt)}
    pairs.sort(key=lambda p: (order_ax[p[0]], feats.index(p[1])))

    def plabel(a, f, w=34):
        return f"{short(a.split('::')[1])[:14]} x {f}"[:w].ljust(w)

    # --- axis summary ---
    summ = []
    for a in tgt:
        c13 = f"+ other axes | {a}"
        c13c = f"+ other axes + controls | {a}"
        links = [f for f in feats if is_link(get("C1", base_cfg, a, f))]

        def survives(check, cfg, f):
            b0 = get("C1", base_cfg, a, f)
            r_ = get(check, cfg, a, f)
            return is_link(r_) and np.sign(r_.value) == np.sign(b0.value)
        s_ctrl = [f for f in links if survives("C1", f"{PRIMARY_SET}: + all controls", f)]
        s_oth = [f for f in links if survives("C13", c13, f)]
        s_both = [f for f in links if survives("C13", c13c, f)]
        nc_key = K(NOCONTROL_SET, a.split("::", 1)[1])

        def survives_nc(f):
            b0 = get("C1", base_cfg, a, f)
            r_ = get("R2B", "base", nc_key, f)
            return is_link(r_) and np.sign(r_.value) == np.sign(b0.value)
        s_r2b = [f for f in links if survives_nc(f)] if nc_key in tgt_nc else []
        best = max(s_both, key=lambda f: abs(get("C13", c13c, a, f).value),
                   default=None)
        summ.append({
            "axis": a.split("::")[1], "links_base": len(links),
            "survive_controls": len(s_ctrl), "survive_other_axes": len(s_oth),
            "survive_both": len(s_both), "survive_r2b": len(s_r2b),
            "surviving_r2b": ", ".join(s_r2b),
            "links": ", ".join(links),
            "surviving_both": ", ".join(s_both),
            "strongest_surviving": best or "",
            "beta_base": get("C1", base_cfg, a, best).value if best else np.nan,
            "beta_both": get("C13", c13c, a, best).value if best else np.nan})
    summ = pd.DataFrame(summ)
    summ.to_csv(P("axis_summary.csv"), index=False, float_format="%.4f")

    # --- report ---
    n_main = res[(res.check == "C1") & (res.image_set == "colour_only")]["n_images"]
    L = ["09a_axis_checks_v4.py -- summary", "=" * 78,
         f"Date     : {time.strftime('%Y-%m-%d %H:%M')}",
         f"Features : {', '.join(feats)}",
         f"           ({len(feats_new)} from step 04c: "
         f"{', '.join(feats_new) or 'none'})",
         f"Selection: {SELECTION_JSON}",
         f"Model    : within-category joint model as in 07 (beta = standardised "
         f"coefficient); {int(CI_LEVEL * 100)}% intervals from {N_BOOT} "
         "category resamples (the same resamples for every check)",
         f"Images   : colour_only unless stated; main evaluation "
         f"{int(n_main.iloc[0]):,} images" if len(n_main) else "",
         f"Targets  : the {len(tgt)} {PRIMARY_SET} axes", ""]

    L.append(f"AXIS SUMMARY: links = |beta| >= {LINK_MIN} with an interval "
             "excluding 0 (base model); how many keep that, with the same "
             "sign, after all controls (C1), the other axes (C13), and both")
    L.append("  r2b = also without the darkness and enclosure directions "
             "(v4@nocontrol, base model)")
    L.append(f"  {'axis':<24}{'base':>6}{'ctrl':>6}{'axes':>6}{'both':>6}{'r2b':>6}   "
             "strongest surviving both (base -> both)")
    for r_ in summ.itertuples():
        s = (f"{r_.strongest_surviving} ({r_.beta_base:+.3f} -> "
             f"{r_.beta_both:+.3f})") if r_.strongest_surviving else "none"
        L.append(f"  {short(r_.axis)[:24]:<24}{r_.links_base:>6}"
                 f"{r_.survive_controls:>6}{r_.survive_other_axes:>6}"
                 f"{r_.survive_both:>6}{r_.survive_r2b:>6}   {s}")
    L.append("  surviving both, per axis:")
    for r_ in summ.itertuples():
        L.append(f"    {short(r_.axis)[:24]:<24} {r_.surviving_both or 'none'}")
    L.append("")

    L.append(f"REPORT PAIRS: the {REPORT_TOP} strongest links of each axis "
             "(|base beta|) plus the key pairs")
    L.append("")
    L.append("C1 CONTROLS (beta; intervals in checks_results.csv)")
    c1 = ["base"] + [f"+ {c}" for c in CONTROLS] + ["+ all controls",
                                                   "+ all but greyscale"]
    heads = ["base"] + [c[:5] for c in CONTROLS] + ["all", "all-g"]
    L.append(f"  {'':<34}" + "".join(f"{h:>7}" for h in heads))
    for a, f in pairs:
        L.append(f"  {plabel(a, f)}" + "".join(
            f"{v3(get('C1', f'{PRIMARY_SET}: {c}', a, f)):>7}" for c in c1))
    L.append("")

    L.append(f"C13 SPECIFICITY (beta): base | + the other {len(tgt) - 1} axes | "
             "+ other axes and all controls")
    for a, f in pairs:
        L.append(f"  {plabel(a, f)} {fmt(get('C1', base_cfg, a, f))}  "
                 f"{fmt(get('C13', f'+ other axes | {a}', a, f))}  "
                 f"{fmt(get('C13', f'+ other axes + controls | {a}', a, f))}")
    L.append("")

    L.append(f"C3 CAPTIONS ({PRIMARY_SET}; beta: scene / photo / place / world)")
    for a, f in pairs:
        ax = a.split("::")[1]
        vals = [get("C3", "base", K(f"{PRIMARY_SET}{t}", ax), f)
                for t in ("", "@photo", "@place", "@world")]
        L.append(f"  {plabel(a, f)} " + " / ".join(v3(v) for v in vals))
    q3 = res[(res.check == "C3") & (res.stat == "beta")].copy()
    q3["ver"] = q3["axis"].str.split("::").str[0]
    q3["base_ver"] = q3["ver"].str.split("@").str[0]
    q3["ax"] = q3["axis"].str.split("::").str[1]
    w3 = q3.pivot_table(index=["base_ver", "ax", "feature"], columns="ver",
                        values="value")
    worst3 = 0.0
    for v in [PRIMARY_SET]:
        for t in ("@photo", "@place", "@world"):
            if v in w3 and f"{v}{t}" in w3:
                d = (w3[f"{v}{t}"] - w3[v]).abs().max()
                worst3 = max(worst3, float(d) if np.isfinite(d) else 0.0)
    L.append(f"  largest |template - scene| over all {PRIMARY_SET} axes and "
             f"features: {worst3:.3f}")
    L.append("")

    L.append(f"C4 BASELINES ({PRIMARY_SET}; beta: main / centroid / 'a scene')")
    for a, f in pairs:
        ax = a.split("::")[1]
        vals = [get("C4", "base", K(f"{PRIMARY_SET}{t}", ax), f)
                for t in ("", "@centroid", "@neutral")]
        L.append(f"  {plabel(a, f)} " + " / ".join(v3(v) for v in vals))
    L.append("")

    L.append(f"C6 GREYSCALE ({len(feats_c6)} features defined for greyscale "
             f"photographs; beta: colour_only / all_images, then with the "
             "greyscale control)")
    for a, f in pairs:
        if f not in feats_c6:
            continue
        g = [get("C6", f"{PRIMARY_SET}: base", a, f),
             get("C6", f"{PRIMARY_SET}: base", a, f, iset="all_images"),
             get("C6", f"{PRIMARY_SET}: + greyscale", a, f),
             get("C6", f"{PRIMARY_SET}: + greyscale", a, f, iset="all_images")]
        L.append(f"  {plabel(a, f)} {v3(g[0])} / {v3(g[1])}    + greyscale "
                 f"{v3(g[2])} / {v3(g[3])}")
    L.append("")

    L.append("C7 LUM_SKEW added (beta of lum_skew, and the largest change in "
             "another feature)")
    for k in prim_all:
        s = get("C7", "+ lum_skew", k, "lum_skew")
        ch = max(((abs(get("C7", "+ lum_skew", k, f).value -
                       get("C7", "features", k, f).value), f) for f in feats))
        L.append(f"  {short(k.split('::')[1]):<24} lum_skew {fmt(s)}   largest "
                 f"change {ch[0]:.3f} ({ch[1]})")
    L.append("")

    L.append(f"C8 DARK CHANNEL and {FOG_AXIS_ID} (fog; beta)")
    for k in fog:
        a = get("C8", "dark + lum_mean + sat_mean", k, "dark_channel_mean")
        b = get("C8", "features + dark", k, "dark_channel_mean")
        r_ = get("C8", "base", k, "dark_channel_mean", "rho")
        L.append(f"  {k:<26} rho alone {r_.value:+.2f}   with lum+sat {fmt(a)} "
                 f"(VIF {a.vif:.1f})   with the features {fmt(b)} "
                 f"(VIF {b.vif:.1f})")
    L.append("")

    L.append("C9 256 vs 336 px (subsample, colour_only; beta)")
    for a, f in pairs:
        x = get("C9", "256", a, f, iset="subsample_256")
        y = get("C9", "336", a, f, iset="subsample_336")
        L.append(f"  {plabel(a, f)} 256 {v3(x)}   336 {v3(y)}")
    q9 = res[res.check == "C9"].pivot_table(index=["axis", "feature", "stat"],
                                            columns="config", values="value")
    L.append(f"  largest |256 - 336| over all {PRIMARY_SET} axes and features: "
             f"{(q9['256'] - q9['336']).abs().max():.3f}")
    L.append("")

    L.append("C10 CROP vs STRETCH (old 11 features; within rho)")
    q10 = res[res.check == "C10"].pivot_table(index=["axis", "feature"],
                                              columns="config", values="value")
    dd = (q10["crop"] - q10["stretch"]).abs().sort_values(ascending=False)
    L.append(f"  largest |crop - stretch|: {dd.iloc[0]:.3f}; median {dd.median():.3f}")
    L += [f"    {i[0][:34]:<34} {i[1]:<15} stretch {q10.loc[i, 'stretch']:+.3f} "
          f"crop {q10.loc[i, 'crop']:+.3f}" for i in dd.index[:5]]
    L.append("")

    L.append("C11 OVERLAP: within-category correlations between axis scores "
             "(mean r, mean |r| with interval, highest |r|)")
    for g in MAIN_SETS:
        pt = extras[("o", g)][0]
        mm, mlo, mhi = extras[("om", g)]
        u = upper(pt)
        L.append(f"  {g:<10} mean r {mm[0]:+.3f}   mean |r| {mm[1]:.3f} "
                 f"[{mlo[1]:.3f}, {mhi[1]:.3f}]   highest |r| {np.abs(u).max():.3f}")
    L.append(f"  {PRIMARY_SET}: largest pairs")
    L += [f"    {short(r_.axis_a)} ~ {short(r_.axis_b)}: {r_.r_within:+.2f} "
          f"[{r_.lo:+.2f}, {r_.hi:+.2f}]" for r_ in ovp.head(8).itertuples()]
    ptc = extras[("o", "controls+primary")][0]
    kc = groups["controls+primary"]
    L.append(f"  {PRIMARY_SET} axes vs controls (within r):")
    L.append(f"    {'':<24}" + "".join(f"{c[:8]:>9}" for c in CONTROLS))
    for a in tgt:
        i = kc.index(a)
        L.append(f"    {short(a.split('::')[1])[:24]:<24}" + "".join(
            f"{ptc[i, kc.index(c)]:>+9.2f}" for c in ctrl_keys))
    L.append("")

    L.append(f"C14 OVERLAP ADJUSTED ({PRIMARY_SET}): partial within-category "
             "correlations between the axes")
    L.append(f"  {'removed':<22}{'mean |r|':>22}{'1st component share':>28}")
    plab = {"raw": "nothing (= C11)", "features": "features",
            "controls": "controls", "both": "features + controls"}
    for nm in pnames:
        mm, mlo, mhi = extras[("pm", nm)]
        L.append(f"  {plab[nm]:<22}{mm[0]:>8.3f} [{mlo[0]:.3f}, {mhi[0]:.3f}]"
                 f"{mm[1]:>12.3f} [{mlo[1]:.3f}, {mhi[1]:.3f}]")
    L.append(f"  (with {len(tgt)} unrelated axes the first component's share "
             f"would be about {1 / len(tgt):.3f})")
    ac3 = os.path.join(os.path.dirname(V3_CHECKS), "axis_components.csv") \
        if V3_CHECKS else None
    if ac3 and os.path.exists(ac3):
        q3_ = pd.read_csv(ac3)
        q3_ = q3_[q3_["component"] == 1].set_index("matrix")["share"]
        L.append(f"  {V3_SET} first-component share for comparison ({ac3}; 8 axes, "
                 f"{1 / 8:.3f} if unrelated): " + ", ".join(
                     f"{plab.get(k, k)} {v:.3f}" for k, v in q3_.items()))
    L.append("  largest pairs (raw -> features removed -> controls removed "
             "-> both removed):")
    top = adj.reindex(adj["r_raw"].abs().sort_values(ascending=False).index).head(8)
    for r_ in top.itertuples():
        L.append(f"    {short(r_.axis_a)[:14]:<14} ~ {short(r_.axis_b)[:14]:<14} "
                 f"{r_.r_raw:+.2f} -> {r_.r_features:+.2f} -> "
                 f"{r_.r_controls:+.2f} -> {r_.r_both:+.2f}")
    L.append("  first-component loadings (raw | features removed | controls "
             "removed | both):")
    c1s = comp[comp.component == 1].set_index("matrix")
    for a in tl:
        col = f"loading_{short(a)}"
        L.append(f"    {short(a)[:24]:<24}" + "".join(
            f"{c1s.loc[nm, col]:>+8.2f}" for nm in pnames))
    L.append("")

    L.append("C16 POLES: overlap with the positive anchors only and the "
             "negative anchors only")
    L.append("  main = positives vs negatives; positive = positives vs the "
             "vocabulary centroid;")
    L.append("  negative = the centroid vs the negatives. The poles of a set "
             "share the centroid,")
    L.append("  which alone makes them correlate: compare the rows with the "
             "vocabulary removed.")
    L.append(f"  {'set':<5}{'poles':<10}{'removed':<23}{'mean r':>8}"
             f"{'mean |r|':>22}{'1st share':>22}")
    for r_ in s16.itertuples():
        L.append(f"  {r_.set:<5}{r_.poles:<10}{r_.removed:<23}{r_.mean_r:>+8.3f}"
                 f"{r_.mean_abs_r:>8.3f} [{r_.mean_abs_r_lo:.3f}, "
                 f"{r_.mean_abs_r_hi:.3f}]{r_.first_share:>8.3f} "
                 f"[{r_.first_share_lo:.3f}, {r_.first_share_hi:.3f}]")
    for s in POLE_SETS:
        L.append(f"  ({s}: with {len(pole_axes[s])} unrelated axes the first "
                 f"share would be about {1 / len(pole_axes[s]):.3f})")
    v3dir = os.path.dirname(V3_CHECKS) if V3_CHECKS else None
    ps3 = os.path.join(v3dir, "poles_summary.csv") if v3dir else None
    if ps3 and os.path.exists(ps3):
        q3_ = pd.read_csv(ps3)
        q3_ = q3_[(q3_["set"] == V3_SET) & (q3_["removed"] == "vocabulary")]
        L.append(f"  {V3_SET} for comparison ({ps3}; vocabulary removed; "
                 f"{1 / 8:.3f} if unrelated):")
        for r_ in q3_.itertuples():
            L.append(f"  {r_.set:<5}{r_.poles:<10}{r_.removed:<23}{r_.mean_r:>+8.3f}"
                     f"{r_.mean_abs_r:>8.3f} [{r_.mean_abs_r_lo:.3f}, "
                     f"{r_.mean_abs_r_hi:.3f}]{r_.first_share:>8.3f} "
                     f"[{r_.first_share_lo:.3f}, {r_.first_share_hi:.3f}]")
    ps = POLE_SETS[0]
    L.append(f"  {ps} pairs (largest main r first): main | then with the "
             "vocabulary removed: main / positive / negative")
    pw = p16[p16["set"] == ps].copy()
    pw["col"] = pw["poles"] + "|" + pw["removed"]
    pw = pw.pivot_table(index=["axis_a", "axis_b"], columns="col", values="r")
    pw = pw.reindex(pw["main|nothing"].abs().sort_values(ascending=False).index)
    for (a_, b_), r_ in pw.head(12).iterrows():
        L.append(f"    {short(a_)[:14]:<14} ~ {short(b_)[:14]:<14} "
                 f"{r_['main|nothing']:+.2f} | {r_['main|vocabulary']:+.2f} / "
                 f"{r_['positive|vocabulary']:+.2f} / "
                 f"{r_['negative|vocabulary']:+.2f}")
    L.append(f"  {ps} first-component loadings, vocabulary removed "
             "(main / positive / negative):")
    lw = l16[(l16["set"] == ps) & (l16["removed"] == "vocabulary")] \
        .pivot_table(index="axis", columns="poles", values="loading")
    for a in pole_axes[ps]:
        L.append(f"    {short(a)[:24]:<24}{lw.loc[a, 'main']:>+8.2f}"
                 f"{lw.loc[a, 'positive']:>+8.2f}{lw.loc[a, 'negative']:>+8.2f}")
    L.append("")

    if have_hue:
        L.append(f"C15 HUE_SPREAD and toned photographs: {int(toned.sum()):,} "
                 f"colour images flagged ({100 * toned.sum() / colour.sum():.2f}%);"
                 " see toned_examples.png")
        L.append("  beta of hue_spread: base | + age | toned left out | "
                 "toned left out + age")
        for a in tgt:
            g = [get("C15", "base", a, "hue_spread"),
                 get("C15", "+ age", a, "hue_spread"),
                 get("C15", "base", a, "hue_spread", iset="colour_only_no_toned"),
                 get("C15", "+ age", a, "hue_spread",
                     iset="colour_only_no_toned")]
            L.append(f"  {short(a.split('::')[1])[:24]:<24}" +
                     "  ".join(fmt(x) for x in g))
        L.append("")

    # ---------------- C17 pair overlap and the merge test ----------------
    o_pt, o_lo, o_hi = extras[("o", PRIMARY_SET)]
    gk = groups[PRIMARY_SET]
    cos_path = os.path.join(AXES_DIR, f"axis_cosines_{PRIMARY_SET}.csv")
    tcos = pd.read_csv(cos_path, index_col=0) if os.path.exists(cos_path) else None
    L.append(f"C17 PAIR OVERLAP (decision 13; report only): v4 pairs with "
             f"within-category |r| >= {OVERLAP_R}")
    m_rows = []
    for i in range(len(gk)):
        for j in range(i + 1, len(gk)):
            a_, b_ = gk[i].split("::", 1)[1], gk[j].split("::", 1)[1]
            tc = float(tcos.loc[a_, b_]) if tcos is not None and a_ in tcos.index \
                and b_ in tcos.columns else np.nan
            m_rows.append({"axis_a": a_, "axis_b": b_, "text_cosine": tc,
                           "image_r_within": o_pt[i, j], "lo": o_lo[i, j],
                           "hi": o_hi[i, j],
                           "merge_criterion_met": bool(tc >= MERGE_MIN and
                                                       o_pt[i, j] >= MERGE_MIN)})
    mt = pd.DataFrame(m_rows)
    mt.to_csv(P("merge_test.csv"), index=False, float_format="%.4f")
    met = mt[mt["merge_criterion_met"]]
    L.append(f"  merge test (text cosine >= {MERGE_MIN} AND within-category image "
             f"r >= {MERGE_MIN}): " + ("none" if not len(met) else "; ".join(
                 f"{short(r_.axis_a)}~{short(r_.axis_b)}" for r_ in met.itertuples())))
    L.append("  highest image r (text cosine | within-category r [interval]):")
    for r_ in mt.reindex(mt["image_r_within"].abs().sort_values(ascending=False)
                         .index).head(6).itertuples():
        L.append(f"    {short(r_.axis_a):<16} ~ {short(r_.axis_b):<16} "
                 f"{r_.text_cosine:+.2f} | {r_.image_r_within:+.2f} "
                 f"[{r_.lo:+.2f}, {r_.hi:+.2f}]")
    if not pairs17:
        L.append(f"  no pair reaches |r| >= {OVERLAP_R}")
    for a, b, r0 in pairs17:
        for x, y in ((a, b), (b, a)):
            L.append(f"  {short(x.split('::')[1])} with {short(y.split('::')[1])} "
                     f"held constant (r {r0:+.2f}); its base links: base -> "
                     f"+ {short(y.split('::')[1])} -> + it and all controls")
            for f in feats:
                b0 = get("C1", base_cfg, x, f)
                if not is_link(b0):
                    continue
                b1 = get("C17", f"+ {y} | {x}", x, f)
                b2 = get("C17", f"+ {y} + controls | {x}", x, f)
                L.append(f"    {f:<15} {fmt(b0)} -> {fmt(b1)} -> {fmt(b2)}")
    L.append("")

    # ---------------- C18 v3-v4 comparison (protocol 6.5) ----------------
    x_pt, x_lo, x_hi = extras[("o", "v4+v3b")]
    n4 = len(tgt)
    Rx, Lx, Hx = x_pt[:n4, n4:], x_lo[:n4, n4:], x_hi[:n4, n4:]
    v3n = [k.split("::", 1)[1] for k in tgt_v3]
    v4n = [k.split("::", 1)[1] for k in tgt]
    pd.DataFrame(Rx, index=v4n, columns=v3n).to_csv(P("v4_v3b_within_r.csv"),
                                                    float_format="%.4f")
    cmp_rows = []
    for i in range(n4):
        for j in range(len(v3n)):
            cmp_rows.append({"v4_axis": v4n[i], "v3b_axis": v3n[j], "r": Rx[i, j],
                             "lo": Lx[i, j], "hi": Hx[i, j]})
    pd.DataFrame(cmp_rows).to_csv(P("v4_v3b_within_r_ci.csv"), index=False,
                                  float_format="%.4f")
    best4 = Rx.argmax(1)            # best v3b axis of each v4 axis
    match, rep_rows = {}, []
    for j, a3 in enumerate(v3n):
        i_best = int(Rx[:, j].argmax())
        cands = [i for i in range(n4) if best4[i] == j and Rx[i, j] >= COUNTERPART_R]
        if len(cands) >= 2:
            match[a3] = ("split", cands)
        elif best4[i_best] == j and Rx[i_best, j] >= COUNTERPART_R:
            match[a3] = ("counterpart", [i_best])
        else:
            match[a3] = ("none", [i_best])
    L.append("C18 v3-v4 COMPARISON (protocol 6.5, fixed in advance; report only)")
    L.append(f"  within-category r between every v4 and {V3_SET} axis "
             "(v4_v3b_within_r.csv); counterpart = each the other's highest "
             f"r and r >= {COUNTERPART_R}; split = two or more v4 axes with "
             f"r >= {COUNTERPART_R} whose best match it is")
    L.append(f"  {'':<16}" + "".join(f"{short(a)[:9]:>10}" for a in v3n))
    for i, a in enumerate(v4n):
        L.append(f"  {short(a)[:16]:<16}" + "".join(f"{Rx[i, j]:>+10.2f}"
                                                    for j in range(len(v3n))))
    L.append("")
    for a3 in v3n:
        kind, ii = match[a3]
        j = v3n.index(a3)
        who = ", ".join(f"{short(v4n[i])} {Rx[i, j]:+.2f} [{Lx[i, j]:+.2f}, "
                        f"{Hx[i, j]:+.2f}]" for i in ii)
        L.append(f"  {short(a3):<14} {kind:<12} {who}")
    L.append("")
    L.append(f"  REPLICATION of the {V3_SET} links on their counterparts: same "
             f"sign, |beta| >= {LINK_MIN}, interval excluding 0")
    for a3 in v3n:
        kind, ii = match[a3]
        if kind == "none":
            continue
        k3 = K(V3_SET, a3)
        for i in ii:
            k4 = tgt[i]
            L.append(f"  {short(a3)} -> {short(v4n[i])} ({kind})")
            for mdl, c3, c4 in (("alone", f"{V3_SET}: base", base_cfg),
                                ("+ all controls", f"{V3_SET}: + all controls",
                                 f"{PRIMARY_SET}: + all controls")):
                links3 = [f for f in feats if is_link(get("C18", c3, k3, f))]
                n_rep = 0
                for f in links3:
                    b3 = get("C18", c3, k3, f)
                    b4 = get("C1", c4, k4, f)
                    ok = is_link(b4) and np.sign(b4.value) == np.sign(b3.value)
                    n_rep += ok
                    rep_rows.append({"v3b_axis": a3, "v4_axis": v4n[i],
                                     "match": kind, "model": mdl, "feature": f,
                                     "beta_v3b": b3.value, "lo_v3b": b3.lo,
                                     "hi_v3b": b3.hi, "beta_v4": b4.value,
                                     "lo_v4": b4.lo, "hi_v4": b4.hi,
                                     "replicated": bool(ok)})
                L.append(f"    {mdl:<15} {n_rep} of {len(links3)} {V3_SET} links "
                         "replicated")
                for f in links3:
                    b3 = get("C18", c3, k3, f)
                    b4 = get("C1", c4, k4, f)
                    ok = is_link(b4) and np.sign(b4.value) == np.sign(b3.value)
                    L.append(f"      {f:<15} {V3_SET} {fmt(b3)}   v4 {fmt(b4)}"
                             f"   {'yes' if ok else 'no'}")
    pd.DataFrame(rep_rows).to_csv(P("replication.csv"), index=False,
                                  float_format="%.4f")
    L.append("")

    # ---------------- R1: the v3b estimates equal the v3 run's ----------------
    if V3_CHECKS and os.path.exists(V3_CHECKS):
        old = pd.read_csv(V3_CHECKS)
        old = old[(old["check"] == "C1") & (old["stat"] == "beta") &
                  (old["image_set"] == "colour_only") &
                  (old["config"].isin([f"{V3_SET}: base",
                                       f"{V3_SET}: + all controls"])) &
                  (old["feature"].isin(feats))]
        now = res[(res["check"] == "C18") & (res["stat"] == "beta")]
        mm_ = old.merge(now, on=["config", "axis", "feature"], suffixes=("_v3", "_now"))
        dmax = max(float((mm_[f"{c}_v3"] - mm_[f"{c}_now"]).abs().max())
                   for c in ("value", "lo", "hi")) if len(mm_) else np.nan
        ok1 = len(mm_) > 0 and dmax <= R1_TOL and len(mm_) == len(old)
        L.append(f"CHECK R1: {V3_SET} base and all-controls estimates vs {V3_CHECKS}: "
                 f"{len(mm_)} of {len(old)} rows matched; largest |diff| {dmax:.1e} -> "
                 f"{'PASS (CSV rounded to 4 decimals)' if ok1 else 'WARN: differs from the v3 run'}")
    else:
        L.append(f"CHECK R1: skipped ({V3_CHECKS} not found)")
    L.append("")

    L.append("C12 DUPLICATES (identical embeddings)")
    if len(dup):
        L.append(f"  {dup['group'].nunique():,} groups, {len(dup):,} images; "
                 f"{int((~dup['keep']).sum()):,} extra copies removed; groups "
                 f"spanning two or more categories: "
                 f"{dup.drop_duplicates('group')['cross_category'].sum():,}")
        worst = 0.0
        for a in tgt:
            for f in feats:
                x = get("C1", base_cfg, a, f)
                y = get("C12", base_cfg, a, f, iset="colour_only_nodup")
                if x is not None and y is not None:
                    worst = max(worst, abs(x.value - y.value))
        L.append(f"  largest |with copies - without| over all {PRIMARY_SET} "
                 f"axes and features: {worst:.4f}")
    else:
        L.append("  none found")
    L.append(f"\nElapsed: {(time.time() - t0) / 60:.1f} min")
    report = "\n".join(L)
    with open(P("checks_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    json.dump({"script": "09a_axis_checks_v4.py", "version": "v4",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "selection_json": SELECTION_JSON, "features": feats,
               "features_04c": feats_new, "features_c6": feats_c6,
               "extra_features": EXTRA_FEATURES, "controls": CONTROLS,
               "target_axes": axes_t, "link_min": LINK_MIN,
               "overlap_r": OVERLAP_R, "merge_min": MERGE_MIN,
               "counterpart_r": COUNTERPART_R, "v3_set": V3_SET,
               "pairs_c17": [[a, b, float(r)] for a, b, r in pairs17],
               "scores_dir": SCORES_DIR, "v3_checks": V3_CHECKS,
               "toned_rule": {"hue_spread_max": TONED_HUE_MAX,
                              "chroma_mean_max": TONED_CHROMA_MAX,
                              "n_flagged": int(toned.sum())},
               "pole_sets": POLE_SETS, "axes_dir": AXES_DIR,
               "pole_checks": {"min_cos_vs_05": pole_cos,
                               "max_diff_vs_06": pole_diff},
               "n_boot": N_BOOT, "seed": RANDOM_SEED},
              open(P("run_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
