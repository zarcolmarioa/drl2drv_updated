"""
09b_item_analysis_v4.py  (pipeline step 09, part b, v4: the captions of
                          each axis)

09a checked whole axes; 09c manipulated images. This script looks inside
the v4 axes: every caption pair is treated as an item of a questionnaire
and described on images, and each axis is compared with an axis built
from images alone. Descriptive only (v4_protocol.md, N4): nothing is
dropped or merged; the results inform the interpretation of each axis and
the discussion that follows.

Axes: the v4 main axes (primary) and the v3b main axes (comparison; 09a
matched them, §6.5).

Images: 04's stratified subsample (about 33,000 images, about 100 per
category), colour photographs with every selected feature defined. All
statistics are WITHIN CATEGORY (every variable minus its category mean),
as in 07 and 09a. Features: the selection in SELECTION_JSON (12 features).

What is scored
--------------
Every "item" is a direction in CLIP's space; an image's item score is its
embedding times that direction (the same as 06's raw score). For an axis
with positive captions P, negative captions N (v4: one per pair, a shared
negative counted once per pair, as 05 builds the axis) and the set's
CENTROID c (the mean of all captions of the set's included axes, each
caption once, as in 05):

  main            unit(mean P - mean N): the axis itself (checked against
                  05's vectors and 06's scores).
  positive pole   unit(mean P - c): the positives alone, against the
                  vocabulary (= 05's "@centroid" axis, checked).
  negative pole   unit(c - mean N): the negatives alone, oriented so that
                  higher = more of the axis. The main axis is, up to
                  scale, positive pole + negative pole, so these two show
                  which side a link comes from.
  pair item       (v4) e_pos - e_neg of one pair: the axis is the mean of
                  its pair items (decision 20). Used for the v4
                  reliability ("pair alpha"), item-rest r and
                  leave-one-pair-out.
  contrast item   for a positive caption a: e_a - mean N; for a negative
                  caption b: mean P - e_b (each distinct caption once).
                  The item of the v3 analysis; gives an alpha that is
                  comparable between v4 and v3b ("caption alpha").
  centroid item   for a positive caption: e_a - c; for a negative caption:
                  c - e_b. The caption's OWN meaning (against the whole
                  vocabulary). Used for the caption profiles, own r and
                  the cross-axis loadings.
  leave-one-out   v4: the main axis without one pair; v3b: without one
                  caption.
  groups          (N3) for the axes in GROUPS: unit(mean of the group's
                  positives - c) ("vs vocabulary") and the group's own
                  sub-axis unit(mean of the group's pair items) ("own
                  pairs").
  image-anchored  unit(mean image embedding of the top images - mean of
                  the bottom images) of each main axis, over the whole
                  colour corpus: "within" = the top and bottom IMG_TOP_PCT %
                  of every category; "global" = the IMG_GLOBAL_N highest and
                  lowest images overall.

Statistics
----------
  beta / r        each item's within-category standardised coefficients
                  on the selected features (joint model, as in 07) and its
                  marginal correlations, with 95% intervals from resampling
                  whole categories (N_BOOT; the same resamples for every
                  item).
  reliability     standardised Cronbach's alpha (with an interval) of the
                  v4 pair items and, for both sets, of the contrast items;
                  per pole (contrast items of that pole); alpha if a pair
                  (v4) or caption (v3b) is deleted; item-rest r (the item's
                  correlation with the axis rebuilt without it). Caveat:
                  contrast items of one pole share the other pole's mean,
                  which raises their correlations; alpha is an index of how
                  consistently the captions rank images, not a
                  questionnaire reliability.
  own r           correlation of the centroid item with its own main axis
                  (negative = the caption ranks images against its axis:
                  flag W).
  cross-loadings  correlation of each centroid item with every other main
                  axis of its set; flag O when another axis is stronger
                  than own r. Pairs of axes in SHARED_PAIRS are listed in
                  full.
  LOO influence   how much each axis's key features' beta change when one
                  pair (v4) or caption (v3b) is left out. Key features:
                  |beta| >= LINK_MIN with an interval excluding 0 (at most
                  KEY_MAX), plus the features in TRACE.
Flags: W = ranks images against its own axis; O = closer to another axis;
T = on the wrong side of the centroid in 05 (text); S = short caption
(SHORT_MAX_WORDS words or fewer).

Report (item_report.txt)
------------------------
  1. TRACE    the links to trace to their captions (decision of 6 Oct):
              which pole, which group, which captions carry each link
  2. GROUPS   movie / film / camera; veil / glass / wall / bubble;
              dead / empty / darkness
  3. SHARED   captions of two axes against both axes
  4. SHORT    one- and two-word captions; every flagged caption
  5. CAPTION TABLES (v4)   6. PAIR TABLES (v4)   7. LOO INFLUENCE
  8. RELIABILITY (v4 and v3b, counterparts side by side)
  9. IMAGE-ANCHORED AXES

Checks (the run stops if one fails)
-----------------------------------
  K0  the pairs of V4_YAML are the v4 anchors of 05 (same captions, same
      order), so pair ids can be attached.
  K1  every rebuilt main axis equals 05's vector (cosine >= VEC_MIN_COS).
  K2  every positive pole equals 05's "@centroid" axis.
  K3  the main-axis projections equal 06's raw scores.

Inputs
------
  INDEX_CSV, FLAGS_CSV, FEATURES_DIR, EXTRA_DIR, SELECTION_JSON (as 07)
  AXES_DIR     05_v4: axis_anchors.csv, captions.csv,
               caption_embeddings.npy, axis_catalog.csv, axis_vectors.npy,
               anchor_diagnostics.csv
  SCORES_DIR   06_v4: score_catalog.csv, scores_raw.npy
  EMB_PATH     step 03 embeddings (the file 06 used)
  V4_YAML      the Stage D2 axes (pair ids)

Outputs (under OUTPUT_DIR)
--------------------------
  item_report.txt          summary (start here)
  caption_table.csv        one row per distinct caption (both sets): group,
                           pairs, own r, strongest other axis, flags, 05's
                           text flag, the centroid item's beta on every
                           feature
  pair_table.csv           one row per v4 pair: item-rest r, alpha if
                           deleted, leave-one-out cosine, the pair item's
                           beta on every feature
  item_profiles.csv        every item x feature: beta and r with intervals
  pole_profiles.csv        the same for main / poles / groups / image axes
  reliability.csv          per axis: pair alpha (v4), caption alpha, per
                           pole, mean item-rest r, flag counts
  cross_loadings.csv       centroid item x every main axis of its set
  loo_influence.csv        leave-one-out change of the key features' beta
  group_matrix.csv         r and cosine between the groups of an axis
  image_anchored.csv       image-anchored axes vs the text axes
  image_axis_cosines_<set>.csv
  run_meta.json

Usage
-----
    conda activate drl2drv
    python 09b_item_analysis_v4.py
One pass over the embeddings (image-anchored axes) and one over the score
file; the rest runs on the subsample. Memory about 3-4 GB.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import yaml


# ============================================================
# SETTINGS
# ============================================================

INDEX_CSV       = "data/selection/image_index.csv"
FLAGS_CSV       = "data/monochrome/image_flags.csv"
FEATURES_DIR    = "data/features"
EXTRA_DIR       = "data/features_extra"          # 04c (version 2)
SELECTION_JSON  = "data/feature_selection_v3/selected_features.json"
AXES_DIR        = "data/axes_v4"                 # 05_build_axes_v4.py
SCORES_DIR      = "data/scores_v4"               # 06_project_axes_v4.py
EMB_PATH        = "data/embeddings/embeddings_768.npy"
V4_YAML         = "axes/v4/stageD2/axes_v4.yaml"
OUTPUT_DIR      = "data/items_v4"                # new folder (v3: data/items)

SETS            = ["v4", "v3b"]       # first = primary
PRIMARY_SET     = "v4"

# Links traced to their captions: (v4 axis id, feature)
TRACE = [("A13", "sat_mean"),          # dull x saturation (09c: causal)
         ("A14", "contrast_local"),    # lifeless x local contrast (09c: not)
         ("A14", "lum_mean"),          # lifeless x brightness (09c: darkening)
         ("A07", "contrast_local"),    # fog x local contrast (09c: causal)
         ("A07", "sat_mean"),          # fog x saturation (09c: via darkness)
         ("A11", "lum_mean"),          # flat x brightness (09c: not)
         ("A11", "contrast_local"),    # flat x local contrast (09c: not)
         ("A03", "spectral_slope")]    # film x spectral slope (09a C4)

# N3 groups of positive captions (pair ids of V4_YAML; pairs not listed
# belong to no group)
GROUPS = {
    "A03": {"picture / art": ["A03-01", "A03-06", "A03-09", "A03-10"],
            "movie / film": ["A03-02", "A03-03", "A03-04", "A03-05", "A03-12"],
            "camera": ["A03-07", "A03-11"]},
    "A05": {"veil": ["A05-01", "A05-11", "A05-20", "A05-21", "A05-22",
                     "A05-24", "A05-33", "A05-42", "A05-44", "A05-45",
                     "A05-46", "A05-50"],
            "glass / screen / window": ["A05-03", "A05-04", "A05-19",
                                        "A05-30", "A05-49", "A05-53",
                                        "A05-56", "A05-57", "A05-59",
                                        "A05-62", "A05-63"],
            "wall / barrier": ["A05-05", "A05-25", "A05-26", "A05-28",
                               "A05-29", "A05-36", "A05-37", "A05-43",
                               "A05-51", "A05-52"],
            "bubble / shell": ["A05-02", "A05-10", "A05-35", "A05-47",
                               "A05-54", "A05-55", "A05-60"]},
    "A14": {"dead / lifeless": ["A14-02", "A14-04", "A14-07", "A14-08",
                                "A14-10", "A14-14"],
            "empty / void": ["A14-01", "A14-06", "A14-09", "A14-11"],
            "darkness": ["A14-12", "A14-13"]},
}

# Pairs of axes whose captions are listed against both axes (v4 ids)
SHARED_PAIRS    = [("A13", "A14"), ("A05", "A07"), ("A01", "A05"),
                   ("A01", "A02"), ("A07", "A14"), ("A07", "A13")]

# v3b counterparts of each v4 axis (09a, C18), side by side in section 8
V3B_MATCH       = {"A01": ["dreamlike"], "A02": ["dreamlike"],
                   "A04": ["constructed / artificial world"],
                   "A05": ["glass-walled / enclosed"],
                   "A07": ["veiled / foggy"],
                   "A13": ["lifeless / deadened"],
                   "A14": ["lifeless / deadened"]}

SHORT_MAX_WORDS = 2        # captions of this many words or fewer: flag S
LINK_MIN        = 0.05     # |beta| for a key feature
KEY_MAX         = 4        # key features per axis (plus TRACE features)
REPORT_TOP      = 3        # pairs / captions listed per axis (LOO section)

# image-anchored axes
IMG_TOP_PCT     = 1.0      # top / bottom % of every category ("within")
IMG_GLOBAL_N    = 200      # top / bottom images overall ("global")

MIN_IMAGES_PER_CATEGORY = 30
N_BOOT          = 1000
CI_LEVEL        = 0.95
RANDOM_SEED     = 42
CHUNK           = 100_000

# verification of the rebuilt axes against steps 05 and 06
VEC_MIN_COS     = 0.9999   # rebuilt axis vs 05's vector
PROJ_MAX_DIFF   = 1e-3     # projection vs 06's raw score


# ============================================================
# Helpers
# ============================================================

def P(*n):
    return os.path.join(OUTPUT_DIR, *n)


def short(a):
    """Display name: 'fog, blurry, hazy' for v4, the first label for v3b."""
    if a[:1] == "A" and a[1:3].isdigit():
        return a.split(" ", 1)[1]
    return a.split(" / ")[0]


def aid(a):
    return a.split()[0] if a[:1] == "A" and a[1:3].isdigit() else ""


def unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def ci(a, level=CI_LEVEL):
    lo = (1 - level) / 2 * 100
    return np.nanpercentile(a, lo, axis=0), np.nanpercentile(a, 100 - lo, axis=0)


def upper(Rm):
    return Rm[np.triu_indices(len(Rm), 1)]


def std_alpha(Rm):
    """Standardised Cronbach's alpha from a correlation matrix."""
    k = len(Rm)
    if k < 2:
        return np.nan
    rb = upper(Rm).mean()
    return k * rb / (1 + (k - 1) * rb)


def cov_to_corr(S):
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


FSHORT = {"contrast_local": "contr", "lum_entropy": "entr",
          "spectral_slope": "slope", "lum_std": "lstd", "lum_mean": "lmean",
          "lum_p05": "p05", "sat_mean": "sat", "sat_std": "satsd",
          "lab_b_mean": "lab_b", "lab_a_mean": "lab_a", "hue_spread": "hue",
          "cast_b": "castb"}


def fs(f):
    return FSHORT.get(f, f[:6])


def nwords(t):
    return len(str(t).split())


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("09b_item_analysis_v4.py")
    print("=" * 70)
    checks = {}

    # --- index, colour flag, categories ---
    idx = pd.read_csv(INDEX_CSV, usecols=["image_id", "category"])
    N = len(idx)
    cat_names = sorted(idx["category"].unique())
    codes_all = pd.Categorical(idx["category"], categories=cat_names).codes.astype(int)
    fl = pd.read_csv(FLAGS_CSV, usecols=["image_id", "is_greyscale"])
    colour = ~fl.set_index("image_id").reindex(np.arange(N))["is_greyscale"] \
        .astype(bool).to_numpy()

    # --- features (as in 07 / 09a) ---
    defs = json.load(open(os.path.join(FEATURES_DIR, "feature_definitions.json")))
    sel = json.load(open(SELECTION_JSON))
    feats = sel["sets"][sel.get("primary", "colour_only")]
    sources = sel.get("feature_sources", {})
    crop = defs["crop"]
    Fc = np.load(os.path.join(FEATURES_DIR, "features_crop.npy"), mmap_mode="r")
    feats_new = [f for f in feats if sources.get(f, "04") == "04c"]
    Ex, xcols = None, []
    if feats_new:
        xdefs = json.load(open(os.path.join(EXTRA_DIR, "extra_definitions.json")))
        m4 = os.path.join(FEATURES_DIR, "run_meta.json")
        m4c = os.path.join(EXTRA_DIR, "run_meta.json")
        if os.path.exists(m4) and os.path.exists(m4c):
            if json.load(open(m4)).get("index_fingerprint") != \
                    json.load(open(m4c)).get("index_fingerprint"):
                raise SystemExit("Steps 04 and 04c used different indexes.")
        xcols = xdefs["crop"]
        Ex = np.load(os.path.join(EXTRA_DIR, "extra_crop.npy"), mmap_mode="r")
        if Ex.shape[0] != N:
            raise SystemExit("04c features do not match the image index.")

    def feature_block(fn, rows):
        out = []
        for f in fn:
            if sources.get(f, "04") == "04c":
                if f not in xcols:
                    raise SystemExit(f"{f} not in 04c's file.")
                a = Ex[rows, xcols.index(f)]
            else:
                if f not in crop:
                    raise SystemExit(f"{f} not in features_crop.")
                a = Fc[rows, crop.index(f)]
            out.append(np.asarray(a, dtype=np.float64))
        return np.column_stack(out)

    # --- axes (step 05) ---
    anc = pd.read_csv(os.path.join(AXES_DIR, "axis_anchors.csv"))
    caps = pd.read_csv(os.path.join(AXES_DIR, "captions.csv"))
    CE = np.load(os.path.join(AXES_DIR, "caption_embeddings.npy")).astype(np.float64)
    cidx = dict(zip(caps["caption"], caps["row"]))
    acat = pd.read_csv(os.path.join(AXES_DIR, "axis_catalog.csv"))
    AV = np.load(os.path.join(AXES_DIR, "axis_vectors.npy")).astype(np.float64)
    arow = dict(zip(acat["key"], acat["row"]))
    diag05 = {}
    dpath = os.path.join(AXES_DIR, "anchor_diagnostics.csv")
    if os.path.exists(dpath):
        d5 = pd.read_csv(dpath)
        diag05 = {(r.set, r.axis, r.pole, r.anchor): bool(r.wrong_side)
                  for r in d5.itertuples()}
    missing = [c for c in anc["caption"].unique() if c not in cidx]
    if missing:
        raise SystemExit(f"{len(missing)} anchor captions not in captions.csv, "
                         f"e.g. {missing[:3]}")
    doc = yaml.safe_load(open(V4_YAML))
    ypairs = {f"{ax['id']} {ax['name']}": ax["pairs"] for ax in doc["axes"]}

    # --- build the text items ---
    items, vecs = [], []
    axes_of, poles = {}, {}
    vec_cos, pole_cos, k0_bad = [], [], []

    def add(vec, **kw):
        items.append(kw)
        vecs.append(vec)
        return len(items) - 1

    for s in SETS:
        mains = acat[(acat["set"] == s) & (acat["kind"] == "main") &
                     (acat["included"].astype(bool))]
        if not len(mains):
            raise SystemExit(f"No included main axes for set {s} in axis_catalog.")
        axes_of[s] = list(mains["axis"])
        cent_caps = list(dict.fromkeys(
            c for k in mains["key"] for c in anc.loc[anc["key"] == k, "caption"]))
        cvec = CE[[cidx[c] for c in cent_caps]].mean(0)
        for r in mains.itertuples():
            a = r.axis
            rows = anc[anc["key"] == r.key]
            pa = rows[rows["pole"] == "positive"].reset_index(drop=True)
            na = rows[rows["pole"] == "negative"].reset_index(drop=True)
            if len(pa) != r.n_pos or len(na) != r.n_neg:
                raise SystemExit(f"{r.key}: anchor counts differ from the catalog.")
            Pe = CE[[cidx[c] for c in pa["caption"]]]
            Ne = CE[[cidx[c] for c in na["caption"]]]
            Pm, Nm = Pe.mean(0), Ne.mean(0)
            main_v = unit(Pm - Nm)
            vec_cos.append(float(main_v @ unit(AV[arow[r.key]])))
            ck = f"{s}@centroid::{a}"
            pos_v = unit(Pm - cvec)
            if ck in arow and arow[ck] >= 0:
                pole_cos.append(float(pos_v @ unit(AV[arow[ck]])))
            base = {"set": s, "axis": a}
            d = {"main": add(main_v, **base, type="main", pole="", anchor="",
                             caption=""),
                 "pos": add(pos_v, **base, type="pole", pole="positive",
                            anchor="", caption=""),
                 "neg": add(unit(cvec - Nm), **base, type="pole",
                            pole="negative", anchor="", caption=""),
                 "caps": [], "pairs": [], "cvec": cvec}
            # pair ids (v4)
            pid_of = {}
            if s == PRIMARY_SET:
                yp = ypairs.get(a)
                if yp is None or len(yp) != len(pa) or len(pa) != len(na) or any(
                        y["positive"] != p_ or y["negative"] != n_ for y, p_, n_ in
                        zip(yp, pa["anchor"], na["anchor"])):
                    k0_bad.append(a)
                    continue
                for i, y in enumerate(yp):
                    pe, ne_ = Pe[i], Ne[i]
                    kw = dict(base, pole="pair", anchor=y["anchor_id"],
                              caption=f"{y['positive']} || {y['negative']}")
                    kp = add(pe - ne_, **kw, type="pair")
                    rest = np.delete(Pe - Ne, i, axis=0).mean(0)
                    kl = add(unit(rest), **kw, type="loo")
                    d["pairs"].append({"pair": y["anchor_id"], "pos": y["positive"],
                                       "neg": y["negative"], "item": kp, "loo": kl,
                                       "i": i})
                    pid_of.setdefault(("positive", y["positive"]), []).append(
                        y["anchor_id"])
                    pid_of.setdefault(("negative", y["negative"]), []).append(
                        y["anchor_id"])
            # distinct captions: contrast and centroid items (v3b: LOO too)
            for pole, df, E_, sign in (("positive", pa, Pe, 1),
                                       ("negative", na, Ne, -1)):
                seen = {}
                for i, (an, cp) in enumerate(zip(df["anchor"], df["caption"])):
                    if cp in seen:
                        continue
                    seen[cp] = i
                    e = E_[i]
                    kw = dict(base, pole=pole, anchor=an, caption=cp)
                    con = e - Nm if sign > 0 else Pm - e
                    c_ = {"pole": pole, "anchor": an, "caption": cp,
                          "contrast": add(con, **kw, type="contrast"),
                          "centroid": add(sign * (e - cvec), **kw, type="centroid"),
                          "n_pairs": int((df["caption"] == cp).sum()),
                          "pairs": pid_of.get((pole, an), []), "loo": None}
                    if s != PRIMARY_SET and len(E_) > 1:
                        keep_ = (df["caption"] != cp).to_numpy()
                        rest = E_[keep_].mean(0)
                        lv = unit(rest - Nm) if sign > 0 else unit(Pm - rest)
                        c_["loo"] = add(lv, **kw, type="loo")
                    d["caps"].append(c_)
            # N3 groups (v4)
            if s == PRIMARY_SET and aid(a) in GROUPS:
                d["groups"] = {}
                byid = {q["pair"]: q for q in d["pairs"]}
                for g, members in GROUPS[aid(a)].items():
                    have = [byid[m] for m in members if m in byid]
                    gone = [m for m in members if m not in byid]
                    if gone:
                        print(f"  group '{g}' of {short(a)}: {gone} not among "
                              f"the pairs")
                    if len(have) < 2:
                        print(f"  group '{g}' of {short(a)}: {len(have)} pair(s), "
                              f"skipped")
                        continue
                    Ge = Pe[[q["i"] for q in have]]
                    Gd = (Pe - Ne)[[q["i"] for q in have]].mean(0)
                    kw = dict(base, pole="positive", anchor=g,
                              caption=f"{len(have)} pairs")
                    d["groups"][g] = {
                        "voc": add(unit(Ge.mean(0) - cvec), **kw, type="group"),
                        "own": add(unit(Gd), **kw, type="group_axis"),
                        "pairs": [q["pair"] for q in have],
                        "n": len(have)}
            poles[(s, a)] = d
    if k0_bad:
        raise SystemExit(f"K0 failed: the pairs of {V4_YAML} differ from 05's "
                         f"anchors for {k0_bad}.")
    checks["K0_pairs_match_yaml"] = True
    checks["K1_main_vs_05_min_cos"] = float(min(vec_cos))
    checks["K2_positive_pole_vs_05_centroid_min_cos"] = \
        float(min(pole_cos)) if pole_cos else None
    print(f"K0: pairs of {V4_YAML} match 05's anchors")
    print(f"Rebuilt {sum(len(v) for v in axes_of.values())} main axes: min cosine "
          f"with 05 {min(vec_cos):.6f}; positive pole vs 05's centroid axis "
          f"{checks['K2_positive_pole_vs_05_centroid_min_cos']}")
    if min(vec_cos) < VEC_MIN_COS:
        raise SystemExit("K1 failed: rebuilt main axes differ from step 05.")
    if pole_cos and min(pole_cos) < VEC_MIN_COS:
        raise SystemExit("K2 failed: positive poles differ from 05's centroid axes.")

    # axis by v4 id
    by_id = {aid(a): a for a in axes_of[PRIMARY_SET]}
    for g in GROUPS:
        if g not in by_id:
            print(f"  GROUPS: no v4 axis {g}")
    trace = [(by_id[x], f) for x, f in TRACE if x in by_id and f in feats]
    skipped = [(x, f) for x, f in TRACE if not (x in by_id and f in feats)]
    if skipped:
        print(f"  TRACE skipped (axis or feature not available): {skipped}")

    # --- scores: main axes over the whole corpus (one pass) ---
    sc = pd.read_csv(os.path.join(SCORES_DIR, "score_catalog.csv"))
    key2col = dict(zip(sc["key"], sc["col"]))
    Rraw = np.load(os.path.join(SCORES_DIR, "scores_raw.npy"), mmap_mode="r")
    if Rraw.shape[0] != N:
        raise SystemExit("scores_raw.npy does not match the image index.")
    mkeys = [(s, a) for s in SETS for a in axes_of[s]]
    mcols = [key2col[f"{s}::{a}"] for s, a in mkeys]
    print(f"Reading {len(mcols)} main-axis scores for {N:,} images...")
    Sm = np.empty((N, len(mcols)), np.float32)
    for a0 in range(0, N, CHUNK):
        b0 = min(a0 + CHUNK, N)
        Sm[a0:b0] = np.asarray(Rraw[a0:b0])[:, mcols]
    valid = colour & np.isfinite(Sm).all(1)
    print(f"  colour images with scores: {valid.sum():,}")

    # --- image-anchored axes: member lists ---
    vidx = np.where(valid)[0]
    vcodes = codes_all[vidx]
    lists = []
    q = IMG_TOP_PCT / 100.0
    for j, (s, a) in enumerate(mkeys):
        x = Sm[vidx, j]
        g = pd.Series(x).groupby(vcodes)
        rk = g.rank(method="first").to_numpy()
        nn = g.transform("size").to_numpy()
        m_ = np.maximum(1, np.floor(nn * q))
        lists.append((s, a, "within", "top", np.sort(vidx[rk > nn - m_])))
        lists.append((s, a, "within", "bottom", np.sort(vidx[rk <= m_])))
        o = np.argsort(x)
        lists.append((s, a, "global", "top", np.sort(vidx[o[-IMG_GLOBAL_N:]])))
        lists.append((s, a, "global", "bottom", np.sort(vidx[o[:IMG_GLOBAL_N]])))
    Emb = np.load(EMB_PATH, mmap_mode="r")
    if Emb.shape[0] != N:
        raise SystemExit("Embeddings do not match the image index.")
    print("Image-anchored axes: one pass over the embeddings...")
    sums = np.zeros((len(lists), Emb.shape[1]))
    for a0 in range(0, N, CHUNK):
        b0 = min(a0 + CHUNK, N)
        blk = None
        for li, L_ in enumerate(lists):
            i0, i1 = np.searchsorted(L_[4], [a0, b0])
            if i1 > i0:
                if blk is None:
                    blk = np.asarray(Emb[a0:b0], dtype=np.float64)
                sums[li] += blk[L_[4][i0:i1] - a0].sum(0)
    img_rows = []
    for j, (s, a) in enumerate(mkeys):
        for var in ("within", "global"):
            li_t = next(i for i, L_ in enumerate(lists)
                        if L_[:4] == (s, a, var, "top"))
            li_b = next(i for i, L_ in enumerate(lists)
                        if L_[:4] == (s, a, var, "bottom"))
            nt, nb = len(lists[li_t][4]), len(lists[li_b][4])
            v = unit(sums[li_t] / nt - sums[li_b] / nb)
            k = add(v, set=s, axis=a, type=f"image_{var}", pole="", anchor="",
                    caption=f"{nt} top / {nb} bottom")
            poles[(s, a)][f"image_{var}"] = k
            img_rows.append({"set": s, "axis": a, "variant": var, "n_top": nt,
                             "n_bottom": nb, "item": k})
    del Sm

    V = np.vstack(vecs)
    items = pd.DataFrame(items)
    K = len(items)
    print(f"{K} items in total")

    # --- subsample: colour, scored, features defined ---
    sub = np.load(os.path.join(FEATURES_DIR, "subsample_ids.npy")).astype(int)
    sub = np.unique(sub[valid[sub]])
    Fs_ = feature_block(feats, np.sort(sub))
    okf = np.isfinite(Fs_).all(1)
    keep = np.zeros(N, bool)
    keep[np.sort(sub)[okf]] = True
    kc = codes_all[keep]
    cnt = np.bincount(kc, minlength=len(cat_names))
    okc = cnt >= MIN_IMAGES_PER_CATEGORY
    keep &= okc[codes_all]
    rows = np.where(keep)[0]
    remap = -np.ones(len(cat_names), int)
    remap[np.where(okc)[0]] = np.arange(okc.sum())
    codes = remap[codes_all[rows]]
    C = int(okc.sum())
    order = np.argsort(codes, kind="stable")
    rows, codes = rows[order], codes[order]
    bounds = np.searchsorted(codes, np.arange(C + 1))
    n_img = len(rows)
    print(f"Subsample: {n_img:,} images in {C} categories "
          f"({len(sub) - okf.sum():,} with an undefined feature left out)")

    srt = np.argsort(rows)
    Es = np.empty((n_img, Emb.shape[1]))
    Es[srt] = np.asarray(Emb[rows[srt]], dtype=np.float64)
    F = np.empty((n_img, len(feats)))
    F[srt] = feature_block(feats, rows[srt])
    A = Es @ V.T
    diffs = []
    for s, a in mkeys:
        k = poles[(s, a)]["main"]
        ref = np.asarray(Rraw[rows[srt], key2col[f"{s}::{a}"]], np.float64)
        diffs.append(float(np.abs(A[srt, k] - ref).max()))
    checks["K3_projection_vs_06_max_diff"] = max(diffs)
    print(f"K3: max |difference| from 06's raw scores {max(diffs):.2e}")
    if max(diffs) > PROJ_MAX_DIFF:
        raise SystemExit("K3 failed: projections differ from step 06 (wrong "
                         "embedding file?).")
    del Es

    # --- within-category moments ---
    print("Within-category moments...")
    p = len(feats)
    Mff = np.zeros((C, p, p))
    Mfa = np.zeros((C, p, K))
    va = np.zeros((C, K))
    for c in range(C):
        sl = slice(bounds[c], bounds[c + 1])
        F[sl] -= F[sl].mean(0)
        A[sl] -= A[sl].mean(0)
        Mff[c] = F[sl].T @ F[sl]
        Mfa[c] = F[sl].T @ A[sl]
        va[c] = (A[sl] ** 2).sum(0)
    Aw = A
    del F
    Q = cov_to_corr(Aw.T @ Aw)

    def profile(Sff, Sfa, v):
        B = np.linalg.solve(Sff, Sfa)
        sf = np.sqrt(np.diagonal(Sff, axis1=-2, axis2=-1))
        sa = np.sqrt(v)
        beta = B * sf[..., :, None] / sa[..., None, :]
        r = Sfa / (sf[..., :, None] * sa[..., None, :])
        return beta, r

    rng = np.random.default_rng(RANDOM_SEED)
    Wb = np.stack([np.bincount(rng.integers(0, C, C), minlength=C)
                   for _ in range(N_BOOT)]).astype(float)
    beta0, r0 = profile(Mff.sum(0), Mfa.sum(0), va.sum(0))
    print(f"Bootstrap ({N_BOOT} category resamples)...")
    Bb = np.empty((N_BOOT, p, K), np.float32)
    Rb = np.empty((N_BOOT, p, K), np.float32)
    step = 50
    for b0 in range(0, N_BOOT, step):
        w = Wb[b0:b0 + step]
        bb, rr = profile(np.einsum("bc,cij->bij", w, Mff),
                         np.einsum("bc,cik->bik", w, Mfa), w @ va)
        Bb[b0:b0 + step], Rb[b0:b0 + step] = bb, rr
    blo, bhi = ci(Bb)
    rlo, rhi = ci(Rb)
    del Bb, Rb

    prof = []
    for k in range(K):
        it = items.iloc[k]
        for i, f in enumerate(feats):
            prof.append({"set": it.set, "axis": it.axis, "type": it.type,
                         "pole": it.pole, "anchor": it.anchor, "feature": f,
                         "beta": beta0[i, k], "beta_lo": blo[i, k],
                         "beta_hi": bhi[i, k], "r": r0[i, k], "r_lo": rlo[i, k],
                         "r_hi": rhi[i, k]})
    prof = pd.DataFrame(prof)
    prof.to_csv(P("item_profiles.csv"), index=False, float_format="%.4f")
    prof[prof["type"].isin(["main", "pole", "group", "group_axis",
                            "image_within", "image_global"])] \
        .to_csv(P("pole_profiles.csv"), index=False, float_format="%.4f")
    fi = {f: i for i, f in enumerate(feats)}

    # --- key features per axis ---
    keyf, repf = {}, {}
    for s, a in mkeys:
        k = poles[(s, a)]["main"]
        cand = [(abs(beta0[i, k]), f) for f, i in fi.items()
                if abs(beta0[i, k]) >= LINK_MIN and
                (blo[i, k] > 0 or bhi[i, k] < 0)]
        kf = [f for _, f in sorted(cand, reverse=True)[:KEY_MAX]]
        if s == PRIMARY_SET:
            kf += [f for ax, f in trace if ax == a and f not in kf]
        keyf[(s, a)] = kf
        repf[(s, a)] = kf if kf else \
            [feats[i] for i in np.argsort(-np.abs(beta0[:, k]))[:3]]

    def alpha_ci(J):
        kk = len(J)
        if kk < 2:
            return np.nan, np.nan, np.nan
        Mjj = np.zeros((C, kk, kk))
        for c in range(C):
            sl = slice(bounds[c], bounds[c + 1])
            X = Aw[sl][:, J]
            Mjj[c] = X.T @ X
        a0_ = std_alpha(cov_to_corr(Mjj.sum(0)))
        ab = np.array([std_alpha(cov_to_corr(np.tensordot(w, Mjj, 1)))
                       for w in Wb])
        lo_, hi_ = ci(ab)
        return a0_, lo_, hi_

    # --- reliability, item statistics, cross-loadings, LOO ---
    print("Reliability and item statistics...")
    group_of = {pid: g for ax, gs in GROUPS.items() for g, m in gs.items()
                for pid in m}
    rel_rows, cap_rows, pair_rows, cross_rows, loo_rows = [], [], [], [], []
    for s, a in mkeys:
        d = poles[(s, a)]
        km = d["main"]
        cps = d["caps"]
        J = [c_["contrast"] for c_ in cps]
        a_cap, a_cap_lo, a_cap_hi = alpha_ci(J)
        RJ = Q[np.ix_(J, J)]
        pos_i = [i for i, c_ in enumerate(cps) if c_["pole"] == "positive"]
        neg_i = [i for i, c_ in enumerate(cps) if c_["pole"] == "negative"]
        a_pos = std_alpha(RJ[np.ix_(pos_i, pos_i)])
        a_neg = std_alpha(RJ[np.ix_(neg_i, neg_i)])
        others = [(o, poles[(s, o)]["main"]) for o in axes_of[s] if o != a]
        n0 = len(cap_rows)
        irs_cap = []
        for i, c_ in enumerate(cps):
            kn = c_["centroid"]
            own = Q[kn, km]
            best, bname = 0.0, ""
            for o, ko in others:
                rr = Q[kn, ko]
                cross_rows.append({"set": s, "axis": a, "pole": c_["pole"],
                                   "anchor": c_["anchor"], "other_axis": o,
                                   "r": rr, "own_r": own})
                if abs(rr) > abs(best):
                    best, bname = rr, o
            t05 = diag05.get((s, a, c_["pole"], c_["anchor"]), np.nan)
            flags = ("W" if own < 0 else "") + \
                    ("O" if bname and abs(best) > own else "") + \
                    ("T" if t05 is True else "") + \
                    ("S" if nwords(c_["anchor"]) <= SHORT_MAX_WORDS else "")
            ir = Q[c_["contrast"], c_["loo"]] if c_["loo"] is not None else np.nan
            irs_cap.append(ir)
            grp = ""
            if s == PRIMARY_SET and c_["pole"] == "positive":
                grp = "; ".join(dict.fromkeys(group_of.get(x, "")
                                              for x in c_["pairs"] if
                                              group_of.get(x, "")))
            row = {"set": s, "axis": a, "pole": c_["pole"],
                   "anchor": c_["anchor"], "caption": c_["caption"],
                   "pairs": ", ".join(c_["pairs"]), "n_pairs": c_["n_pairs"],
                   "group": grp, "own_r": own,
                   "item_total_r": Q[c_["contrast"], km],
                   "item_rest_r": ir,
                   "alpha_if_deleted": std_alpha(np.delete(np.delete(RJ, i, 0),
                                                           i, 1)),
                   "strongest_other_axis": bname, "other_r": best,
                   "flags": flags, "text_wrong_side_05": t05,
                   "loo_cos": float(V[c_["loo"]] @ V[km])
                   if c_["loo"] is not None else np.nan}
            for f, fi_ in fi.items():
                row[f"beta_{f}"] = beta0[fi_, kn]
            cap_rows.append(row)
            if c_["loo"] is not None:
                for f in repf[(s, a)]:
                    bm, bl_ = beta0[fi[f], km], beta0[fi[f], c_["loo"]]
                    loo_rows.append({"set": s, "axis": a, "unit": "caption",
                                     "pole": c_["pole"], "anchor": c_["anchor"],
                                     "feature": f, "beta_main": bm,
                                     "beta_loo": bl_, "delta": bl_ - bm,
                                     "loo_cos": row["loo_cos"]})
        caps_here = cap_rows[n0:]
        rel = {"set": s, "axis": a, "n_pairs": len(d["pairs"]) or np.nan,
               "n_pos": len(pos_i), "n_neg": len(neg_i),
               "alpha_caption": a_cap, "alpha_caption_lo": a_cap_lo,
               "alpha_caption_hi": a_cap_hi, "alpha_pos": a_pos,
               "alpha_neg": a_neg,
               "n_W": sum("W" in x["flags"] for x in caps_here),
               "n_O": sum("O" in x["flags"] for x in caps_here),
               "n_T": sum("T" in x["flags"] for x in caps_here)}
        if d["pairs"]:
            JP = [q["item"] for q in d["pairs"]]
            a_p, a_p_lo, a_p_hi = alpha_ci(JP)
            RP = Q[np.ix_(JP, JP)]
            irs = []
            for i, q_ in enumerate(d["pairs"]):
                ir = Q[q_["item"], q_["loo"]]
                irs.append(ir)
                prow = {"set": s, "axis": a, "pair": q_["pair"],
                        "group": group_of.get(q_["pair"], ""),
                        "positive": q_["pos"], "negative": q_["neg"],
                        "item_rest_r": ir, "item_total_r": Q[q_["item"], km],
                        "alpha_if_deleted": std_alpha(
                            np.delete(np.delete(RP, i, 0), i, 1)),
                        "loo_cos": float(V[q_["loo"]] @ V[km])}
                for f, fi_ in fi.items():
                    prow[f"beta_{f}"] = beta0[fi_, q_["item"]]
                pair_rows.append(prow)
                for f in repf[(s, a)]:
                    bm, bl_ = beta0[fi[f], km], beta0[fi[f], q_["loo"]]
                    loo_rows.append({"set": s, "axis": a, "unit": "pair",
                                     "pole": "", "anchor": q_["pair"],
                                     "feature": f, "beta_main": bm,
                                     "beta_loo": bl_, "delta": bl_ - bm,
                                     "loo_cos": prow["loo_cos"]})
            rel.update({"alpha_pairs": a_p, "alpha_pairs_lo": a_p_lo,
                        "alpha_pairs_hi": a_p_hi,
                        "mean_item_rest_r": float(np.mean(irs)),
                        "min_item_rest_r": float(np.min(irs))})
        else:
            rel.update({"alpha_pairs": np.nan, "alpha_pairs_lo": np.nan,
                        "alpha_pairs_hi": np.nan,
                        "mean_item_rest_r": float(np.nanmean(irs_cap)),
                        "min_item_rest_r": float(np.nanmin(irs_cap))})
        rel_rows.append(rel)
    rel = pd.DataFrame(rel_rows)
    ct = pd.DataFrame(cap_rows)
    pt = pd.DataFrame(pair_rows)
    cross = pd.DataFrame(cross_rows)
    loo = pd.DataFrame(loo_rows)
    rel.to_csv(P("reliability.csv"), index=False, float_format="%.4f")
    ct.to_csv(P("caption_table.csv"), index=False, float_format="%.4f")
    pt.to_csv(P("pair_table.csv"), index=False, float_format="%.4f")
    cross.to_csv(P("cross_loadings.csv"), index=False, float_format="%.4f")
    loo.to_csv(P("loo_influence.csv"), index=False, float_format="%.4f")

    # --- groups: alpha within each group, r / cosine between groups ---
    gm_rows = []
    for a in axes_of[PRIMARY_SET]:
        d = poles[(PRIMARY_SET, a)]
        gs = d.get("groups", {})
        byid = {q["pair"]: q for q in d["pairs"]}
        for g, G in gs.items():
            JP = [byid[x]["item"] for x in G["pairs"]]
            G["alpha"] = std_alpha(Q[np.ix_(JP, JP)])
            x = pt[(pt["axis"] == a) & (pt["pair"].isin(G["pairs"]))]
            G["rest"] = float(x["item_rest_r"].mean())
            y = ct[(ct["set"] == PRIMARY_SET) & (ct["axis"] == a) &
                   (ct["pole"] == "positive") & (ct["group"] == g)]
            G["own_r"] = float(y["own_r"].mean()) if len(y) else np.nan
        gl = list(gs)
        for i, g1 in enumerate(gl):
            for g2 in gl[i + 1:]:
                gm_rows.append({"axis": a, "group_1": g1, "group_2": g2,
                                "r_vocab": Q[gs[g1]["voc"], gs[g2]["voc"]],
                                "cos_vocab": float(V[gs[g1]["voc"]] @
                                                   V[gs[g2]["voc"]]),
                                "r_own": Q[gs[g1]["own"], gs[g2]["own"]],
                                "cos_own": float(V[gs[g1]["own"]] @
                                                 V[gs[g2]["own"]])})
    gmat = pd.DataFrame(gm_rows)
    gmat.to_csv(P("group_matrix.csv"), index=False, float_format="%.4f")

    # --- image-anchored axes ---
    for r in img_rows:
        d = poles[(r["set"], r["axis"])]
        k, km = r["item"], d["main"]
        r["cos_text"] = float(V[k] @ V[km])
        r["r_within_text"] = float(Q[k, km])
        r["profile_agreement"] = float(np.corrcoef(beta0[:, k], beta0[:, km])[0, 1])
    img = pd.DataFrame(img_rows).drop(columns="item")
    img.to_csv(P("image_anchored.csv"), index=False, float_format="%.4f")
    for s in SETS:
        ax = axes_of[s]
        ki = [poles[(s, a)]["image_within"] for a in ax]
        kt = [poles[(s, a)]["main"] for a in ax]
        lab = [short(a) for a in ax]
        ci_ = pd.DataFrame(V[ki] @ V[ki].T, index=[f"img:{x}" for x in lab],
                           columns=lab)
        ct_ = pd.DataFrame(V[kt] @ V[kt].T, index=[f"text:{x}" for x in lab],
                           columns=lab)
        pd.concat([ci_, ct_]).to_csv(P(f"image_axis_cosines_{s}.csv"),
                                     float_format="%.4f")

    # ========================================================
    # Report
    # ========================================================
    L = []
    W_ = L.append

    def bci(k, f):
        i = fi[f]
        b, lo, hi = beta0[i, k], blo[i, k], bhi[i, k]
        star = "*" if (lo > 0 or hi < 0) else " "
        return f"{b:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"

    def kfl(s, a):
        if keyf[(s, a)]:
            return "key features: " + ", ".join(keyf[(s, a)])
        return "no key feature; the 3 strongest shown: " + ", ".join(repf[(s, a)])

    def wrap(txt, n=2, ind="    "):
        for i in range(0, len(txt), n):
            W_(ind + " | ".join(txt[i:i + n]))

    W_("09b ITEM ANALYSIS (v4)")
    W_("=" * 78)
    W_(f"Images: {n_img:,} colour images of 04's subsample, {C} categories "
       f"(within-category statistics).")
    W_(f"Features ({p}): {', '.join(feats)}")
    W_(f"Axes: {', '.join(f'{s} ({len(axes_of[s])})' for s in SETS)}; "
       f"{K} items; {N_BOOT} category resamples.")
    W_(f"Checks: K0 pairs match {V4_YAML}; K1 main axes vs 05 min cosine "
       f"{checks['K1_main_vs_05_min_cos']:.6f}; K2 positive pole vs 05's "
       f"centroid axis {checks['K2_positive_pole_vs_05_centroid_min_cos']}; "
       f"K3 projection vs 06 max diff {checks['K3_projection_vs_06_max_diff']:.1e}.")
    W_("")
    W_("How to read: beta = within-category standardised coefficient (joint "
       "model on all")
    W_("features, as in 07), 95% interval, * = interval excludes 0. The main "
       "axis is, up to")
    W_("scale, positive pole + negative pole, so a link that appears in one "
       "pole only comes")
    W_("from that pole's captions. A caption's beta is for its OWN meaning "
       "(against the whole")
    W_("vocabulary), oriented so that + = more of the axis (for a negative "
       "caption: + = the")
    W_("caption is LESS like the image). A pair's beta is for the pair item "
       "(positive minus its")
    W_("own negative). Flags: W = ranks images against its own axis; O = "
       "closer to another")
    W_("axis than to its own; T = wrong side of the centroid in 05 (text); S = "
       f"{SHORT_MAX_WORDS} words or fewer.")
    W_("")
    W_("Axis names: " + "; ".join(f"{aid(a)} = {short(a)}"
                                 for a in axes_of[PRIMARY_SET]))

    # --- 1. TRACE ---
    W_("")
    W_("1. LINKS TRACED TO THEIR CAPTIONS")
    W_("-" * 78)
    W_("  (09a and 07 used all colour images; the subsample gives slightly "
       "different betas.)")
    for a, f in trace:
        d = poles[(PRIMARY_SET, a)]
        W_("")
        W_(f"'{short(a)}' x {f}")
        W_(f"  main axis        {bci(d['main'], f)}")
        W_(f"  positive pole    {bci(d['pos'], f)}   (positive captions alone)")
        W_(f"  negative pole    {bci(d['neg'], f)}   (negative captions alone)")
        for g, G in d.get("groups", {}).items():
            W_(f"  group {g + ' (' + str(G['n']) + ')':<26} vs vocabulary "
               f"{bci(G['voc'], f)}; own pairs {bci(G['own'], f)}")
        sub_ = ct[(ct["set"] == PRIMARY_SET) & (ct["axis"] == a)]
        for pole in ("positive", "negative"):
            x = sub_[sub_["pole"] == pole].sort_values(f"beta_{f}",
                                                       ascending=False)
            W_(f"  {pole} captions, beta on {f} (high to low):")
            wrap([f"{r['anchor'][:44]}"
                  f"{' [' + r['flags'] + ']' if r['flags'] else ''}"
                  f" {r[f'beta_{f}']:+.2f}" for _, r in x.iterrows()])
        x = pt[pt["axis"] == a].sort_values(f"beta_{f}", ascending=False)
        W_(f"  pairs (positive - its negative), beta on {f}: highest "
           + ", ".join(f"{r['pair']} {r[f'beta_{f}']:+.2f}" for _, r in
                       x.head(3).iterrows())
           + "; lowest " + ", ".join(f"{r['pair']} {r[f'beta_{f}']:+.2f}" for _, r in
                                     x.tail(3).iloc[::-1].iterrows()))
        lx = loo[(loo["set"] == PRIMARY_SET) & (loo["axis"] == a) &
                 (loo["feature"] == f)].copy()
        lx["ad"] = lx["delta"].abs()
        W_("  largest leave-one-pair-out changes: " + "; ".join(
            f"{r.anchor} {r.delta:+.3f}"
            for r in lx.nlargest(REPORT_TOP, "ad").itertuples()))

    # --- 2. GROUPS ---
    W_("")
    W_("2. GROUPS WITHIN THREE AXES (N3)")
    W_("-" * 78)
    W_("  vs vocabulary = the group's positive captions against all captions; "
       "own pairs = the")
    W_("  group's own sub-axis (its positives minus their own negatives). r "
       "main = within-")
    W_("  category r with the whole axis. alpha = pair alpha within the "
       "group; rest = mean")
    W_("  item-rest r of its pairs (with the whole axis); own = mean own r of "
       "its positives.")
    for a in axes_of[PRIMARY_SET]:
        d = poles[(PRIMARY_SET, a)]
        gs = d.get("groups")
        if not gs:
            continue
        kf = repf[(PRIMARY_SET, a)]
        allf = list(dict.fromkeys(kf + [f for f in feats if f not in kf]))[:6]
        W_("")
        W_(f"'{short(a)}'   (features: key features first)")
        W_(f"  {'item':<46}" + "".join(f"{fs(f):>7}" for f in allf) + "  r main")

        def line(lbl, k):
            return f"  {lbl[:45]:<46}" + "".join(f"{beta0[fi[f], k]:>+7.3f}"
                                            for f in allf) + \
                f"  {Q[k, d['main']]:+.2f}"
        W_(line("main axis", d["main"]))
        W_(line("positive pole", d["pos"]))
        W_(line("negative pole", d["neg"]))
        for g, G in gs.items():
            W_(line(f"{g} ({G['n']}) vs vocabulary", G["voc"]))
            W_(line(f"{g} own pairs", G["own"]))
        for g, G in gs.items():
            W_(f"  {g}: alpha {G['alpha']:.2f}, rest {G['rest']:+.2f}, own "
               f"{G['own_r']:+.2f}  [{', '.join(G['pairs'])}]")
        x = gmat[gmat["axis"] == a]
        if len(x):
            W_("  between groups (r within categories / cosine; vs vocabulary, "
               "then own pairs):")
            for r in x.itertuples():
                W_(f"    {r.group_1} ~ {r.group_2}: {r.r_vocab:+.2f} / "
                   f"{r.cos_vocab:+.2f};  {r.r_own:+.2f} / {r.cos_own:+.2f}")
        ing = {p_ for G in gs.values() for p_ in G["pairs"]}
        out = [q["pair"] for q in d["pairs"] if q["pair"] not in ing]
        if out:
            W_(f"  in no group: {', '.join(out)}")

    # --- 3. SHARED ---
    W_("")
    W_("3. CAPTIONS OF TWO AXES AGAINST BOTH AXES (centroid items, r within)")
    W_("-" * 78)
    for x1, x2 in SHARED_PAIRS:
        if x1 not in by_id or x2 not in by_id:
            continue
        a1, a2 = by_id[x1], by_id[x2]
        d1, d2 = poles[(PRIMARY_SET, a1)], poles[(PRIMARY_SET, a2)]
        W_("")
        W_(f"'{short(a1)}' ~ '{short(a2)}': axes r {Q[d1['main'], d2['main']]:+.2f}"
           f", image-anchored r {Q[d1['image_within'], d2['image_within']]:+.2f}")
        for own, oth in ((a1, a2), (a2, a1)):
            x = cross[(cross["set"] == PRIMARY_SET) & (cross["axis"] == own) &
                      (cross["other_axis"] == oth)].copy()
            x["gap"] = x["r"] - x["own_r"]
            x = x.sort_values("gap", ascending=False)
            W_(f"  '{short(own)}' captions: own r / r with '{short(oth)}' "
               f"(most shared first)")
            wrap([f"{r.anchor[:40]} ({r.pole[:3]}) {r.own_r:+.2f}/{r.r:+.2f}"
                  for r in x.head(6).itertuples()])
            W_(f"    closer to '{short(oth)}' than to their own axis: "
               f"{int((x['r'] > x['own_r']).sum())} of {len(x)}")

    # --- 4. SHORT and FLAGGED ---
    W_("")
    W_(f"4. SHORT CAPTIONS ({SHORT_MAX_WORDS} words or fewer) AND CAPTIONS "
       "FLAGGED W OR T (v4)")
    W_("-" * 78)
    W_("  own = own r; other = strongest other axis (r); then the caption's "
       "beta on its axis's")
    W_("  key features. Captions flagged only O are in the caption tables "
       "(section 5).")
    v4c = ct[ct["set"] == PRIMARY_SET]
    sel_ = v4c[v4c["flags"].str.contains("W|T|S")]
    for a in axes_of[PRIMARY_SET]:
        x = sel_[sel_["axis"] == a]
        if not len(x):
            continue
        kf = repf[(PRIMARY_SET, a)]
        km = poles[(PRIMARY_SET, a)]["main"]
        W_(f"  '{short(a)}'  (main: " + ", ".join(
            f"{fs(f)} {beta0[fi[f], km]:+.2f}" for f in kf) + ")")
        for r in x.sort_values(["pole", "own_r"],
                               ascending=[False, True]).itertuples():
            oth = f"{short(r.strongest_other_axis)[:16]} {r.other_r:+.2f}"
            W_(f"    {r.pole[:3]} {r.anchor[:40]:<40} {r.flags:<4} own "
               f"{r.own_r:+.2f}  {oth:<24}" + " ".join(
                   f"{fs(f)} {getattr(r, f'beta_{f}'):+.2f}" for f in kf))
    W_("  Flag counts W / O / T / S per axis: " + "; ".join(
        f"{aid(a)} " + "/".join(
            str(int(v4c[v4c['axis'] == a]['flags'].str.contains(fl_).sum()))
            for fl_ in "WOTS") for a in axes_of[PRIMARY_SET]))

    # --- 5. CAPTION TABLES ---
    W_("")
    W_("5. CAPTION TABLES (v4; distinct captions; negatives shared by several "
       "pairs once)")
    W_("-" * 78)
    W_("  n = pairs using the caption; tot = r of its contrast item with the "
       "axis; own = own r;")
    W_("  other = strongest other axis (r); then the caption's own beta on the "
       "key features.")
    for a in axes_of[PRIMARY_SET]:
        r_ = rel[(rel["set"] == PRIMARY_SET) & (rel["axis"] == a)].iloc[0]
        kf = repf[(PRIMARY_SET, a)]
        km = poles[(PRIMARY_SET, a)]["main"]
        W_("")
        W_(f"{a}: pair alpha {r_.alpha_pairs:.2f} [{r_.alpha_pairs_lo:.2f}, "
           f"{r_.alpha_pairs_hi:.2f}]; caption alpha {r_.alpha_caption:.2f} "
           f"(pos {r_.alpha_pos:.2f}, neg {r_.alpha_neg:.2f}); {kfl(PRIMARY_SET, a)}")
        W_(f"  {'pole caption':<44}{'n':>3}{'tot':>6}{'own':>6}  {'other':<20}"
           f"{'fl':<5}" + "".join(f"{fs(f):>7}" for f in kf))
        W_(f"  {'main axis':<44}{'':>3}{'':>6}{'':>6}  {'':<20}{'':<5}" +
           "".join(f"{beta0[fi[f], km]:>+7.2f}" for f in kf))
        x = ct[(ct["set"] == PRIMARY_SET) & (ct["axis"] == a)]
        for pole in ("positive", "negative"):
            for r in x[x["pole"] == pole].sort_values(
                    "own_r", ascending=False).itertuples():
                oth = f"{short(r.strongest_other_axis)[:13]} {r.other_r:+.2f}"
                W_(f"  {pole[:3]} {r.anchor[:40]:<40}{r.n_pairs:>3}"
                   f"{r.item_total_r:>+6.2f}{r.own_r:>+6.2f}  {oth:<20}"
                   f"{r.flags:<5}" +
                   "".join(f"{getattr(r, f'beta_{f}'):>+7.2f}" for f in kf))

    # --- 6. PAIR TABLES ---
    W_("")
    W_("6. PAIR TABLES (v4; sorted by item-rest r)")
    W_("-" * 78)
    W_("  rest = item-rest r; a-del = pair alpha if the pair is deleted; "
       "group = N3 group;")
    W_("  then the pair item's beta on the key features.")
    for a in axes_of[PRIMARY_SET]:
        kf = repf[(PRIMARY_SET, a)]
        x = pt[pt["axis"] == a].sort_values("item_rest_r", ascending=False)
        W_("")
        W_(f"{a}")
        W_(f"  {'pair':<8}{'positive':<36}{'rest':>6}{'a-del':>6}  {'group':<12}"
           + "".join(f"{fs(f):>7}" for f in kf))
        for r in x.itertuples():
            W_(f"  {r.pair:<8}{r.positive[:35]:<36}{r.item_rest_r:>+6.2f}"
               f"{r.alpha_if_deleted:>6.2f}  {r.group[:11]:<12}" +
               "".join(f"{getattr(r, f'beta_{f}'):>+7.2f}" for f in kf))

    # --- 7. LOO ---
    W_("")
    W_(f"7. LEAVE-ONE-OUT INFLUENCE ON THE KEY FEATURES (top {REPORT_TOP}; v4 "
       "pairs, v3b captions)")
    W_("-" * 78)
    for s in SETS:
        for a in axes_of[s]:
            x = loo[(loo["set"] == s) & (loo["axis"] == a)].copy()
            if not len(x):
                continue
            x["ad"] = x["delta"].abs()
            g = x.groupby(["anchor", "pole"])["ad"].max().nlargest(REPORT_TOP)
            parts = []
            for (an, pole), _ in g.items():
                y = x[(x["anchor"] == an) & (x["pole"] == pole)]
                yy = y.loc[y["ad"].idxmax()]
                parts.append(f"{an[:24]}{' (' + pole[:3] + ')' if pole else ''}: "
                             f"{fs(yy.feature)} {yy.beta_main:+.3f}->"
                             f"{yy.beta_loo:+.3f}")
            W_(f"  {s:<4}{short(a)[:22]:<23}" + "; ".join(parts) +
               ("" if keyf[(s, a)] else "  (no key feature)"))

    # --- 8. RELIABILITY ---
    W_("")
    W_("8. RELIABILITY")
    W_("-" * 78)
    W_("  pair alpha (v4 only) and caption alpha (contrast items; comparable "
       "between v4 and")
    W_("  v3b); pos / neg = caption alpha within each pole; rest = mean "
       "item-rest r (v4 pairs,")
    W_("  v3b captions); W / O / T = flagged captions.")
    W_(f"  {'set':<5}{'axis':<34}{'pairs':>6}{'pair a':>8}{'interval':>15}"
       f"{'capt a':>8}{'pos':>6}{'neg':>6}{'rest':>6}{'W':>3}{'O':>3}{'T':>3}")

    def relline(r):
        np_ = "" if not np.isfinite(r.n_pairs) else f"{int(r.n_pairs)}"
        pa_ = (f"{r.alpha_pairs:>8.2f}  [{r.alpha_pairs_lo:.2f}, "
               f"{r.alpha_pairs_hi:.2f}]") if np.isfinite(r.alpha_pairs) \
            else f"{'':>8}  {'':>12} "
        return (f"  {r.set:<5}{short(r.axis)[:33]:<34}{np_:>6}{pa_}"
                f"{r.alpha_caption:>8.2f}{r.alpha_pos:>6.2f}{r.alpha_neg:>6.2f}"
                f"{r.mean_item_rest_r:>+6.2f}{r.n_W:>3}{r.n_O:>3}{r.n_T:>3}")
    for r in rel.itertuples():
        W_(relline(r))
    W_("  v4 axis and its v3b counterpart(s) (09a), caption alpha:")
    rx = rel.set_index(["set", "axis"])
    for a in axes_of[PRIMARY_SET]:
        m = [x for x in V3B_MATCH.get(aid(a), []) if ("v3b", x) in rx.index]
        if "v3b" not in SETS or not m:
            continue
        W_(f"    {short(a)[:30]:<31}{rx.loc[(PRIMARY_SET, a), 'alpha_caption']:.2f}"
           "   vs " + ", ".join(f"{short(x)} {rx.loc[('v3b', x), 'alpha_caption']:.2f}"
                                for x in m))

    # --- 9. IMAGE-ANCHORED ---
    W_("")
    W_("9. IMAGE-ANCHORED AXES (top minus bottom images of each axis)")
    W_("-" * 78)
    W_("  cos = cosine with the text axis; r = within-category r of the scores "
       "(subsample);")
    W_(f"  prof = correlation of the two feature profiles ({p} betas).")
    W_(f"  {'set':<5}{'axis':<34}{'within: cos':>12}{'r':>6}{'prof':>6}"
       f"{'global: cos':>13}{'r':>6}{'prof':>6}")
    for s in SETS:
        for a in axes_of[s]:
            w_ = img[(img["set"] == s) & (img["axis"] == a) &
                     (img["variant"] == "within")].iloc[0]
            g_ = img[(img["set"] == s) & (img["axis"] == a) &
                     (img["variant"] == "global")].iloc[0]
            W_(f"  {s:<5}{short(a)[:33]:<34}{w_.cos_text:>12.2f}"
               f"{w_.r_within_text:>+6.2f}{w_.profile_agreement:>+6.2f}"
               f"{g_.cos_text:>13.2f}{g_.r_within_text:>+6.2f}"
               f"{g_.profile_agreement:>+6.2f}")
    for s in SETS:
        ax = axes_of[s]
        ki = [poles[(s, a)]["image_within"] for a in ax]
        kt = [poles[(s, a)]["main"] for a in ax]
        W_(f"  {s}: mean |cosine| between axes, text "
           f"{np.abs(upper(V[kt] @ V[kt].T)).mean():.2f}, image-anchored "
           f"{np.abs(upper(V[ki] @ V[ki].T)).mean():.2f}; mean |r| of scores, "
           f"text {np.abs(upper(Q[np.ix_(kt, kt)])).mean():.2f}, image-anchored "
           f"{np.abs(upper(Q[np.ix_(ki, ki)])).mean():.2f}")
    W_("")
    W_("Files: caption_table.csv, pair_table.csv, item_profiles.csv, "
       "pole_profiles.csv,")
    W_("reliability.csv, cross_loadings.csv, loo_influence.csv, "
       "group_matrix.csv, image_anchored.csv,")
    W_("image_axis_cosines_<set>.csv")
    W_(f"Runtime {(time.time() - t0) / 60:.1f} min")
    open(P("item_report.txt"), "w").write("\n".join(L) + "\n")

    json.dump({"script": "09b_item_analysis_v4.py", "version": 1,
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "inputs": {"index_csv": INDEX_CSV, "flags_csv": FLAGS_CSV,
                          "features_dir": FEATURES_DIR, "extra_dir": EXTRA_DIR,
                          "selection_json": SELECTION_JSON, "axes_dir": AXES_DIR,
                          "scores_dir": SCORES_DIR, "emb_path": EMB_PATH,
                          "v4_yaml": V4_YAML},
               "settings": {"sets": SETS, "trace": TRACE, "groups": GROUPS,
                            "shared_pairs": SHARED_PAIRS,
                            "v3b_match": V3B_MATCH,
                            "short_max_words": SHORT_MAX_WORDS,
                            "link_min": LINK_MIN, "key_max": KEY_MAX,
                            "img_top_pct": IMG_TOP_PCT,
                            "img_global_n": IMG_GLOBAL_N,
                            "min_images_per_category": MIN_IMAGES_PER_CATEGORY,
                            "n_boot": N_BOOT, "seed": RANDOM_SEED},
               "features": feats, "n_images": int(n_img), "n_categories": C,
               "n_items": int(K), "checks": checks,
               "key_features": {f"{s}::{a}": v for (s, a), v in keyf.items()},
               "runtime_min": (time.time() - t0) / 60},
              open(P("run_meta.json"), "w"), indent=2)
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Report: {P('item_report.txt')}")


if __name__ == "__main__":
    main()
