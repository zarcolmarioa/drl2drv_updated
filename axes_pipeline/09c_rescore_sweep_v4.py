"""
09c_rescore_sweep_v4.py  (pipeline step 09, part c, v4: manipulation sweep
                          rescored on the v4 axes)

v4_protocol.md §6.5: "09c is rescored from the stored sweep embeddings; no
re-encoding." The sweep (09c_manipulation_sweep.py, version 2, output in
data/sweep_v2) manipulated 150 photographs at calibrated levels and stored
the un2CLIP embedding of every variant (embeddings.npy). This script scores
those embeddings on the v4 axes of 05_build_axes_v4.py and computes the same
statistics as the sweep did for v3b. Nothing is generated, measured or
encoded again: the manipulations, levels, feature changes and side effects
are the sweep's.

The question it answers (causal side of 07 / 09a): when a feature is
changed in a real photograph and nothing else, does the v4 axis that goes
with it in the corpus move? E.g. does blur, lower contrast or smoothing
raise A14 (lifeless) and A07 (fog), and does lower chroma raise A13
(dull)?

Statistics (as the sweep; v4 replaces v3b as the primary set)
------------------------------------------------------------
For each variant level and axis: d = the mean change over images in
corpus SD units of the axis (06_v4's z scale), relative to the same image's
level 0; a 95% interval from resampling images (N_BOOT); the share of
images moving in the mean's direction (same_sign). For the v4 axes also:
  unique   the change beyond what the other eight v4 axes' changes predict
           (within-category regression of each v4 axis on the other eight,
           colour images, from scores_v4).
  share    d^2 / sum of d^2 over the nine v4 axes (1 = only this axis
           moved; 0.11 = all moved equally).
  per SD   d / median change of the calibrated feature in corpus SD units.
  beyond   the change beyond what the change of the quality control
           ("beyond q") or of quality, mood and threat ("beyond qmt")
           predicts (within-category regression on those controls, colour
           images). Descriptive: it assumes the manipulation moves an axis
           and the controls in the ratio natural variation does.
New in v4:
  R2b      the same effects for v4@nocontrol (every v4 axis with the
           enclosure and darkness text directions removed). If an effect
           holds for v4 but not for v4@nocontrol, it runs through those
           directions.
  control directions  the enclosure and darkness text directions themselves
           (decision 17; rebuilt from 05_v4's caption embeddings exactly as
           05_v4 builds them; check R4), scored like an axis. Their corpus
           SD is estimated from SD_SAMPLE stored embeddings (check R5
           compares that estimate with 06_v4's SD for one axis).
  key axes table  for each axis in KEY_AXES, every manipulation at the top
           of the natural range, sorted by d.

Checks (the run stops if R1-R4 fail)
-----------------------------------
  R1  the axis vectors in AXES_DIR for every key the sweep scored
      (score_keys.csv: v2, v3b, controls ...) reproduce the sweep's stored
      raw scores (variant_scores.npy) from the stored embeddings.
  R2  d recomputed for the sweep's own report sets (v3b, v3b@centroid,
      controls, v2) with 06_v4's SDs equals the sweep's effects.csv.
  R3  the v4 axis vectors are unit length; the sweep's arrays are
      consistent (rows = images x variants, image-major, level 0 first).
  R4  the rebuilt control directions reproduce v4@nocontrol: projecting
      them out of each v4 vector gives the stored v4@nocontrol vector.
  R5  (reported) the sampled SD of one axis vs 06_v4's sd_raw.

Inputs
------
  SWEEP_DIR   (data/sweep_v2): embeddings.npy, variants.csv, levels.csv,
              side_effects.csv, corpus_feature_stats.csv, variant_scores.npy,
              score_keys.csv, effects.csv, sweep_images.csv
  AXES_DIR    (data/axes_v4): axis_catalog.csv, axis_vectors.npy,
              captions.csv, caption_embeddings.npy
  SCORES_DIR  (data/scores_v4): score_catalog.csv, scores_z.npy
  INDEX_CSV, FLAGS_CSV, EMB_PATH, EMB_STATUS

Outputs (under OUTPUT_DIR)
--------------------------
  sweep_report.txt     summary (start here)
  effects.csv          level x axis: d, interval, same-sign share, d per SD;
                       unique, share, beyond q / qmt for the v4 axes
  variant_scores.npy   (variants, keys) raw scores on the reported keys
  score_keys.csv       the columns of variant_scores.npy (with the SD used)
  run_meta.json

Usage
-----
    conda activate drl2drv
    python 09c_rescore_sweep_v4.py
A few minutes (most of it reading scores_z.npy for the corpus relation).
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

# --- Inputs ---
SWEEP_DIR       = "data/sweep_v2"           # 09c_manipulation_sweep.py v2
AXES_DIR        = "data/axes_v4"            # 05_build_axes_v4.py
SCORES_DIR      = "data/scores_v4"          # 06_project_axes_v4.py
INDEX_CSV       = "data/selection/image_index.csv"
FLAGS_CSV       = "data/monochrome/image_flags.csv"
EMB_PATH        = "data/embeddings/embeddings_768.npy"
EMB_STATUS      = "data/embeddings/status.npy"   # 1 = encoded
OUTPUT_DIR      = "data/sweep_v4"           # new folder

# --- Axes reported ---
PRIMARY_SET     = "v4"
NOCONTROL_SET   = "v4@nocontrol"            # R2b
CENTROID_SET    = "v4@centroid"             # positive poles
COMPARE_SET     = "v3b"                     # continuity with the sweep
QUALITY_KEY     = "controls::quality"       # + = "a bad photo"
ADJUST          = {"q": ["controls::quality"],
                   "qmt": ["controls::quality", "controls::mood",
                           "controls::threat"]}
KEY_AXES        = ["A07", "A13", "A14", "A11"]   # None = all v4 axes

# --- Control directions (as 05_build_axes_v4.py; check R4) ---
CTRL_TEMPLATE   = "a scene that looks {}"
CONTROLS = {
    "enclosure": [("enclosed", "open"), ("cramped", "spacious"),
                  ("like a closed room", "like a wide open landscape"),
                  ("like an indoor room", "like the outdoors")],
    "darkness":  [("dark", "bright"), ("dimly lit", "brightly lit"),
                  ("like nighttime", "like daytime"), ("shadowy", "sunlit")],
}
SD_SAMPLE       = 200_000                   # embeddings used for their SD

# --- Statistics ---
N_BOOT          = 2000
CI_LEVEL        = 0.95
SIDE_MIN        = 0.5       # side effects reported from |median| >= this SD

# --- Checks ---
R1_TOL          = 1e-3      # max |score diff| in corpus SD units
R2_TOL          = 2e-3      # max |d diff| (effects.csv is rounded to 1e-4)
R4_MIN_COS      = 0.9999
RANDOM_SEED     = 42


# ============================================================
# Helpers
# ============================================================

def P(*n):
    return os.path.join(OUTPUT_DIR, *n)


def S_(*n):
    return os.path.join(SWEEP_DIR, *n)


def ci(a, level=CI_LEVEL, axis=0):
    lo = (1 - level) / 2 * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanpercentile(a, lo, axis=axis),
                np.nanpercentile(a, 100 - lo, axis=axis))


def unit(v):
    return v / np.linalg.norm(v)


def axis_of(k):
    return k.split("::", 1)[1]


def lab(k):
    """Short column label: the v4 id (A07), else the first word."""
    a = axis_of(k)
    if k.startswith(f"{PRIMARY_SET}") and a[:1] == "A" and a[1:3].isdigit():
        return a.split()[0]
    return a.split(" / ")[0].split(",")[0][:6]


def stop(msg):
    raise SystemExit(msg)


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    checks = {}
    print("=" * 70)
    print("09c_rescore_sweep_v4.py")
    print("=" * 70)

    # --- the sweep ---
    var = pd.read_csv(S_("variants.csv"))
    levels = pd.read_csv(S_("levels.csv"))
    side = pd.read_csv(S_("side_effects.csv"))
    fstats = pd.read_csv(S_("corpus_feature_stats.csv")).set_index("feature")
    simg = pd.read_csv(S_("sweep_images.csv"))
    E = np.load(S_("embeddings.npy")).astype(np.float64)
    old_S = np.load(S_("variant_scores.npy"))
    old_keys = pd.read_csv(S_("score_keys.csv"))
    old_eff = pd.read_csv(S_("effects.csv"))
    ids = var["image_id"].drop_duplicates().to_numpy()
    nI = len(ids)
    nV = len(var) // nI
    feat_cols = [c for c in var.columns if c not in
                 ("row", "image_id", "family", "direction", "level", "param",
                  "clipped_new")]
    # R3: layout
    ok3 = (len(var) == nI * nV and E.shape == (nI * nV, 768)
           and old_S.shape[0] == nI * nV
           and (var["row"].to_numpy() == np.arange(nI * nV)).all()
           and (var["image_id"].to_numpy() == np.repeat(ids, nV)).all())
    V0 = var.iloc[:nV][["family", "direction", "level", "param"]] \
        .fillna({"direction": ""}).reset_index(drop=True)
    for i in range(1, nI):
        Vi = var.iloc[i * nV:(i + 1) * nV][["family", "direction", "level"]] \
            .fillna({"direction": ""}).reset_index(drop=True)
        ok3 &= Vi.equals(V0[["family", "direction", "level"]])
    ok3 &= V0.at[0, "family"] == "none"
    checks["R3_layout_ok"] = bool(ok3)
    if not ok3:
        stop("Check R3 failed: variants.csv / embeddings.npy are not "
             "image-major with the same variants per image.")
    E = E.reshape(nI, nV, 768)
    Fv = var[feat_cols].to_numpy(np.float64).reshape(nI, nV, len(feat_cols))
    print(f"Sweep: {nI} images x {nV} variants ({SWEEP_DIR})")

    # variant list and the calibrated feature of each family / direction
    lev_i = levels.fillna({"direction": ""}).set_index(
        ["family", "direction", "level"])
    var_list = []
    for v, r in V0.iterrows():
        if v == 0:
            var_list.append((None, 0.0, "level0", "", "", None))
            continue
        key = (r.family, r.direction, r.level)
        if key not in lev_i.index:
            stop(f"levels.csv has no row for {key}.")
        lr = lev_i.loc[key]
        var_list.append((r.family, float(r.param), r.level, r.direction,
                         lr["status"], lr["feature"]))
    fd = list(dict.fromkeys((f, d) for f, _, _, d, _, _ in var_list if f))
    use_lv = {}
    for f, _, lv, d, _, _ in var_list:
        if f:
            use_lv.setdefault((f, d), []).append(lv)

    # --- axes ---
    acat = pd.read_csv(os.path.join(AXES_DIR, "axis_catalog.csv"))
    AV = np.load(os.path.join(AXES_DIR, "axis_vectors.npy")).astype(np.float64)
    acat = acat[acat["row"] >= 0].sort_values("row")
    if not (acat["row"].to_numpy() == np.arange(len(AV))).all():
        stop("axis_catalog rows do not match axis_vectors.npy.")
    keys = list(acat["key"])
    kv = {k: AV[i] for i, k in enumerate(keys)}
    inc = dict(zip(acat["key"], acat["included"].astype(bool)))
    nerr = float(np.abs(np.linalg.norm(AV, axis=1) - 1).max())
    checks["R3_axes_max_norm_error"] = nerr
    if nerr > 1e-4:
        stop("Check R3 failed: axis vectors are not unit length.")
    sc = pd.read_csv(os.path.join(SCORES_DIR, "score_catalog.csv"))
    sd_raw = dict(zip(sc["key"], sc["sd_raw"]))

    def setkeys(set_):
        return [k for k in keys if k.startswith(f"{set_}::") and inc[k]]

    prim = setkeys(PRIMARY_SET)
    ncs = [f"{NOCONTROL_SET}::{axis_of(k)}" for k in prim]
    ncs = [k for k in ncs if k in kv]
    cents = [f"{CENTROID_SET}::{axis_of(k)}" for k in prim]
    cents = [k for k in cents if k in kv]
    ctrls = [k for k in keys if k.startswith("controls::")]
    comp = setkeys(COMPARE_SET)
    if not prim:
        stop(f"No '{PRIMARY_SET}' axes in {AXES_DIR}.")
    print(f"Axes: {len(prim)} {PRIMARY_SET}, {len(ncs)} {NOCONTROL_SET}, "
          f"{len(cents)} {CENTROID_SET}, {len(ctrls)} controls, "
          f"{len(comp)} {COMPARE_SET}")

    # --- R1: the sweep's stored scores ---
    common = [(int(r.col), r.key) for r in old_keys.itertuples() if r.key in kv]
    miss1 = [k for k in old_keys["key"] if k not in kv]
    Ef = E.reshape(nI * nV, 768)
    worst = 0.0
    for c, k in common:
        s = Ef @ kv[k]
        sd = sd_raw.get(k, np.nan)
        worst = max(worst, float(np.abs(s - old_S[:, c]).max() / sd))
    checks["R1_max_diff_sd"] = worst
    checks["R1_keys_compared"] = len(common)
    checks["R1_keys_missing"] = len(miss1)
    print(f"Check R1: {len(common)} stored score columns recomputed, max "
          f"|diff| {worst:.2e} SD ({len(miss1)} keys not in {AXES_DIR}) -> "
          f"{'PASS' if worst <= R1_TOL and common else 'FAIL'}")
    if not common or not worst <= R1_TOL:
        stop("Check R1 failed: the axes do not reproduce the sweep's scores.")

    # --- control directions and R4 ---
    caps = pd.read_csv(os.path.join(AXES_DIR, "captions.csv"))
    C = np.load(os.path.join(AXES_DIR, "caption_embeddings.npy")).astype(np.float64)
    cidx = dict(zip(caps["caption"], caps["row"]))
    ctrl_vec = {}
    for cn, prs in CONTROLS.items():
        need = [CTRL_TEMPLATE.format(x) for p in prs for x in p]
        m = [c for c in need if c not in cidx]
        if m:
            stop(f"Control captions not in captions.csv: {m}")
        ctrl_vec[cn] = unit(np.mean([C[cidx[CTRL_TEMPLATE.format(a)]]
                                     - C[cidx[CTRL_TEMPLATE.format(b)]]
                                     for a, b in prs], axis=0))
    Q = np.linalg.qr(np.stack(list(ctrl_vec.values()), axis=1))[0]
    cos4 = [float(unit(kv[k] - Q @ (Q.T @ kv[k])) @
                  kv[f"{NOCONTROL_SET}::{axis_of(k)}"])
            for k in prim if f"{NOCONTROL_SET}::{axis_of(k)}" in kv]
    checks["R4_min_cos"] = min(cos4) if cos4 else np.nan
    print(f"Check R4: control directions rebuilt; min cosine with "
          f"{NOCONTROL_SET} {checks['R4_min_cos']:.6f} -> "
          f"{'PASS' if cos4 and min(cos4) >= R4_MIN_COS else 'FAIL'}")
    if not cos4 or min(cos4) < R4_MIN_COS:
        stop("Check R4 failed: the control directions differ from 05_v4's.")
    dirs = [f"direction::{cn}" for cn in ctrl_vec]
    for cn, v_ in ctrl_vec.items():
        kv[f"direction::{cn}"] = v_

    # corpus SD of the control directions (sample of stored embeddings)
    Emb = np.load(EMB_PATH, mmap_mode="r")
    okE = np.load(EMB_STATUS) == 1 if os.path.exists(EMB_STATUS) \
        else np.ones(Emb.shape[0], bool)
    pool_ = np.where(okE)[0]
    smp = np.sort(rng.choice(pool_, min(SD_SAMPLE, len(pool_)), replace=False))
    Xs = np.asarray(Emb[smp], np.float64)
    Xs = Xs[np.isfinite(Xs).all(1) & (np.abs(Xs).sum(1) > 0)]
    for k in dirs:
        sd_raw[k] = float((Xs @ kv[k]).std())
    ref = prim[0]
    r5 = float((Xs @ kv[ref]).std() / sd_raw[ref])
    checks["R5_sample_sd_ratio"] = r5
    checks["R5_n_embeddings"] = int(len(Xs))
    print(f"Check R5: sampled SD / 06_v4 SD for {lab(ref)} = {r5:.4f} "
          f"({len(Xs):,} embeddings); control direction SDs "
          + ", ".join(f"{k.split('::')[1]} {sd_raw[k]:.4f}" for k in dirs))
    del Xs, Emb

    # --- scores and effects ---
    rep_keys = list(dict.fromkeys(prim + ncs + cents + ctrls + dirs + comp
                                  + [k for _, k in common]))
    miss_sd = [k for k in rep_keys if not np.isfinite(sd_raw.get(k, np.nan))]
    if miss_sd:
        stop(f"No corpus SD for {miss_sd[:5]} (score_catalog.csv).")
    kix = {k: j for j, k in enumerate(rep_keys)}
    AVr = np.stack([kv[k] for k in rep_keys])
    key_sd = np.array([sd_raw[k] for k in rep_keys])
    S = np.einsum("ivd,kd->ivk", E, AVr)
    Dz = (S - S[:, :1, :]) / key_sd[None, None, :]
    Df = Fv - Fv[:, :1, :]

    # --- R2: the sweep's own effects ---
    rec = []
    for v, (fam, p, lv, dirn, status, tgt) in enumerate(var_list):
        if fam is None:
            continue
        x = old_eff[(old_eff["family"] == fam) &
                    (old_eff["direction"].fillna("") == dirn) &
                    (old_eff["level"] == lv)]
        for r in x.itertuples():
            if r.axis in kix:
                rec.append(abs(Dz[:, v, kix[r.axis]].mean() - r.d))
    checks["R2_max_d_diff"] = float(max(rec)) if rec else np.nan
    checks["R2_compared"] = len(rec)
    print(f"Check R2: {len(rec)} effects of the sweep's report recomputed, "
          f"max |d diff| {checks['R2_max_d_diff']:.2e} -> "
          f"{'PASS' if rec and max(rec) <= R2_TOL else 'FAIL'}")
    if not rec or max(rec) > R2_TOL:
        stop("Check R2 failed: the sweep's effects are not reproduced.")

    # --- corpus relation (unique, beyond) ---
    print("Corpus relation between the v4 axes and the controls "
          "(within categories, colour images)...")
    idx = pd.read_csv(INDEX_CSV, usecols=["image_id", "category"])
    N = len(idx)
    fl = pd.read_csv(FLAGS_CSV, usecols=["image_id", "is_greyscale"])
    colour = ~fl.set_index("image_id").reindex(np.arange(N))["is_greyscale"] \
        .astype(bool).to_numpy()
    adj_keys = list(dict.fromkeys(k for v in ADJUST.values() for k in v))
    miss = [k for k in adj_keys if k not in kix]
    if miss:
        stop(f"ADJUST: {miss} not among the axes.")
    rel_keys = prim + adj_keys
    Z = np.load(os.path.join(SCORES_DIR, "scores_z.npy"), mmap_mode="r")
    if Z.shape[0] != N:
        stop("scores_z.npy rows do not match the image index.")
    zc = sc.set_index("key")["col"]
    zcols = [int(zc[k]) for k in rel_keys]
    codes = pd.Categorical(idx["category"]).codes.astype(int)
    Cn = codes.max() + 1
    cnt = np.zeros(Cn)
    sums = np.zeros((Cn, len(rel_keys)))
    T = np.zeros((len(rel_keys), len(rel_keys)))
    for a0 in range(0, N, 200_000):
        b0 = min(a0 + 200_000, N)
        X = np.asarray(Z[a0:b0], np.float64)[:, zcols]
        m = colour[a0:b0] & np.isfinite(X).all(1)
        X, cd = X[m], codes[a0:b0][m]
        cnt += np.bincount(cd, minlength=Cn)
        np.add.at(sums, cd, X)
        T += X.T @ X
    okc = cnt > 0
    Sw = T - np.einsum("c,ci,cj->ij", 1 / cnt[okc], sums[okc], sums[okc])
    nP = len(prim)
    Bu = np.zeros((nP, nP))
    for a in range(nP):
        o = [j for j in range(nP) if j != a]
        Bu[a, o] = np.linalg.solve(Sw[np.ix_(o, o)], Sw[o, a])
    Bc = {}
    for nm, ks in ADJUST.items():
        g = [rel_keys.index(k) for k in ks]
        Bc[nm] = np.linalg.solve(Sw[np.ix_(g, g)], Sw[np.ix_(g, range(nP))]).T
    dS = np.sqrt(np.diag(Sw))
    corr_q = {k: Sw[rel_keys.index(k), :nP] / (dS[rel_keys.index(k)] * dS[:nP])
              for k in adj_keys}
    R_un = 1 - np.array([Sw[a, a] - Sw[a, [j for j in range(nP) if j != a]]
                         @ Bu[a, [j for j in range(nP) if j != a]]
                         for a in range(nP)]) / np.diag(Sw)[:nP]

    # --- effects ---
    W = np.stack([np.bincount(rng.integers(0, nI, nI), minlength=nI)
                  for _ in range(N_BOOT)]).astype(float) / nI
    pi = [kix[k] for k in prim]
    eff_rows = []
    for v, (fam, p, lv, dirn, status, tgt) in enumerate(var_list):
        if fam is None:
            continue
        X = Dz[:, v, :]
        mean = X.mean(0)
        lo, hi = ci(W @ X)
        U = X[:, pi] - X[:, pi] @ Bu.T
        um = U.mean(0)
        ulo, uhi = ci(W @ U)
        bey = {}
        for nm, ks in ADJUST.items():
            Bm = X[:, pi] - X[:, [kix[k] for k in ks]] @ Bc[nm].T
            bl, bh = ci(W @ Bm)
            bey[nm] = (Bm.mean(0), bl, bh)
        dm = mean[pi]
        share = dm ** 2 / max((dm ** 2).sum(), 1e-12)
        tj = feat_cols.index(tgt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mc_sd = float(np.nanmedian(Df[:, v, tj])) / fstats.at[tgt, "sd"]
        for k in rep_keys:
            j = kix[k]
            st = ("v4" if k in prim else "v4@nocontrol" if k in ncs else
                  "v4@centroid" if k in cents else "controls" if k in ctrls
                  else "direction" if k in dirs else k.split("::")[0])
            row = {"family": fam, "direction": dirn, "level": lv, "param": p,
                   "status": status, "set": st, "axis": k, "d": mean[j],
                   "lo": lo[j], "hi": hi[j],
                   "same_sign": float((np.sign(X[:, j]) ==
                                       np.sign(mean[j])).mean()),
                   "feature": tgt, "feature_change_sd": mc_sd,
                   "d_per_sd": mean[j] / mc_sd if abs(mc_sd) >= 0.05
                   else np.nan}
            if k in prim:
                a = prim.index(k)
                row.update({"unique": um[a], "unique_lo": ulo[a],
                            "unique_hi": uhi[a], "share": share[a]})
                for nm, (bm_, bl_, bh_) in bey.items():
                    row.update({f"beyond_{nm}": bm_[a],
                                f"beyond_{nm}_lo": bl_[a],
                                f"beyond_{nm}_hi": bh_[a]})
            eff_rows.append(row)
    eff = pd.DataFrame(eff_rows)
    eff.to_csv(P("effects.csv"), index=False, float_format="%.4f")
    np.save(P("variant_scores.npy"), S.reshape(nI * nV, -1).astype(np.float32))
    pd.DataFrame({"col": range(len(rep_keys)), "key": rep_keys,
                  "sd_used": key_sd}).to_csv(P("score_keys.csv"), index=False)
    E_ = {k: g.set_index("axis") for k, g in
          eff.groupby(["family", "direction", "level"], sort=False)}

    def get(fam, dirn, lv):
        return E_.get((fam, dirn, lv))

    def natural_top(fam, dirn):
        lv = use_lv.get((fam, dirn), [])
        nat = [x for x in lv if x.startswith("p")]
        return nat[-1] if nat else (lv[0] if lv else None)

    def ext_top(fam, dirn):
        lv = use_lv.get((fam, dirn), [])
        return lv[-1] if lv else None

    def sig(lo_, hi_):
        return "*" if (lo_ > 0 or hi_ < 0) else " "

    # ========================================================
    # Report
    # ========================================================
    L = []
    W_ = L.append
    pl = [lab(k) for k in prim]
    W_("09c MANIPULATION SWEEP, RESCORED ON THE v4 AXES")
    W_("=" * 78)
    W_(f"Sweep    : {SWEEP_DIR} ({nI} images, {len(simg['category'].unique()) if 'category' in simg else '?'} "
       f"categories; {nV} variants per image); nothing re-encoded")
    W_(f"Axes     : {AXES_DIR}; SDs from {SCORES_DIR} (06_v4's z scale)")
    W_(f"Checks   : R1 stored scores max diff {checks['R1_max_diff_sd']:.1e} SD "
       f"({checks['R1_keys_compared']} keys); R2 sweep effects max d diff "
       f"{checks['R2_max_d_diff']:.1e} ({checks['R2_compared']}); R3 layout, "
       f"unit axes; R4 control directions min cos {checks['R4_min_cos']:.6f}; "
       f"R5 sampled SD ratio {checks['R5_sample_sd_ratio']:.3f}")
    W_("Effects  : d = mean change over images in corpus SD units of the axis,")
    W_("           relative to the same image's level 0; interval from "
       f"{N_BOOT} image resamples; * = interval excludes 0.")
    W_("Axes     : " + "; ".join(axis_of(k) for k in prim))
    W_("")

    W_("1. LEVELS (the sweep's; target = change of the calibrated feature; "
       "achieved = median over images)")
    W_("-" * 78)
    W_(f"  {'family':<13}{'dir':<5}{'level':<6}{'feature':<18}{'target':>9}"
       f"{'achieved':>10}{'param':>10}{'clip':>7}  status")
    for r in levels.fillna({"direction": ""}).itertuples():
        W_(f"  {r.family:<13}{r.direction:<5}{r.level:<6}{str(r.feature)[:17]:<18}"
           f"{r.target_change:>9.3g}{r.achieved_change:>10.3g}"
           f"{r.param:>10.4g}{r.clipped_new:>7.3f}  {r.status}")
    W_("")

    def table(title, pick, ks=None, extra=True):
        ks = ks or prim
        W_(title)
        W_("-" * 78)
        W_(f"  {'manipulation':<22}{'lvl':<5}" + "".join(f"{lab(k):>7}" for k in ks)
           + (f"{'qual':>7}  top: unique / beyond qmt / share" if extra else ""))
        for fam, dirn in fd:
            lv = pick(fam, dirn)
            x = get(fam, dirn, lv) if lv else None
            if x is None:
                continue
            ds = [x.at[k, "d"] for k in ks]
            s_ = "".join(f"{v:>+7.2f}" for v in ds)
            if extra:
                q = x.at[QUALITY_KEY, "d"] if QUALITY_KEY in x.index else np.nan
                top = ks[int(np.argmax(np.abs(ds)))]
                t_ = x.loc[top]
                s_ += (f"{q:>+7.2f}  {lab(top)}{sig(t_['lo'], t_['hi'])} "
                       f"{t_['unique']:+.2f} / {t_['beyond_qmt']:+.2f} / "
                       f"{t_['share']:.2f}")
            W_(f"  {fam + ' ' + dirn:<22}{lv:<5}" + s_)
        W_("")

    table("2. EFFECTS ON THE v4 AXES AT THE TOP OF THE NATURAL RANGE (p99 or "
          "p01, or the highest level reached)", natural_top)
    W_("   qual = quality control (+ = more like 'a bad photo'); top = the "
       "axis that moved most;")
    W_("   unique = change beyond what the other eight v4 axes predict; "
       "beyond qmt = change beyond")
    W_("   what quality, mood and threat predict (section 4); share = "
       "d^2 / sum d^2 (0.11 = all equal).")
    W_("")

    def beyond_table(title, nm, pick):
        W_(title)
        W_("-" * 78)
        W_(f"  {'manipulation':<22}{'lvl':<5}" + "".join(f"{x:>7}" for x in pl)
           + "   largest")
        for fam, dirn in fd:
            lv = pick(fam, dirn)
            x = get(fam, dirn, lv) if lv else None
            if x is None:
                continue
            bs_ = [x.at[k, f"beyond_{nm}"] for k in prim]
            j = int(np.argmax(np.abs(bs_)))
            r_ = x.loc[prim[j]]
            W_(f"  {fam + ' ' + dirn:<22}{lv:<5}" +
               "".join(f"{v:>+7.2f}" for v in bs_) +
               f"   {pl[j]}{sig(r_[f'beyond_{nm}_lo'], r_[f'beyond_{nm}_hi'])} "
               f"(raw {r_['d']:+.2f})")
        W_("")

    beyond_table("3. BEYOND THE QUALITY CONTROL, TOP OF THE NATURAL RANGE "
                 "(d minus what the change in 'a bad photo' predicts)", "q",
                 natural_top)
    W_("   Corpus relation (within categories, colour images), within r with "
       "quality:")
    W_("   " + ", ".join(f"{pl[i]} {corr_q[QUALITY_KEY][i]:+.2f}"
                         for i in range(nP)) + ".")
    W_("   Descriptive: it assumes the manipulation moves an axis and the "
       "control in the ratio")
    W_("   natural variation does; not a causal decomposition.")
    W_("")
    beyond_table("4. BEYOND QUALITY, MOOD AND THREAT, TOP OF THE NATURAL RANGE",
                 "qmt", natural_top)
    W_("   Within r with mood: " + ", ".join(
        f"{pl[i]} {corr_q['controls::mood'][i]:+.2f}" for i in range(nP))
        if "controls::mood" in corr_q else "")
    W_("   Within r with threat: " + ", ".join(
        f"{pl[i]} {corr_q['controls::threat'][i]:+.2f}" for i in range(nP))
        if "controls::threat" in corr_q else "")
    W_("   R^2 of each v4 axis on the other eight (how much 'unique' removes): "
       + ", ".join(f"{pl[i]} {R_un[i]:.2f}" for i in range(nP)) + ".")
    W_("")

    # --- 5. key axes ---
    kax = prim if KEY_AXES is None else \
        [k for k in prim if lab(k) in KEY_AXES]
    W_("5. KEY AXES: EVERY MANIPULATION AT THE TOP OF THE NATURAL RANGE, "
       "SORTED BY d")
    W_("-" * 78)
    W_("   d [interval]; same = share of images moving in d's direction; "
       "uniq = beyond the other v4")
    W_("   axes; qmt = beyond quality, mood and threat; noctl = the same axis "
       "without the enclosure /")
    W_("   darkness directions (R2b); /SD = d per corpus SD of the calibrated "
       "feature's change.")
    for k in kax:
        W_("")
        W_(f"  {axis_of(k)}")
        W_(f"    {'manipulation':<22}{'lvl':<5}{'d':>7} {'[interval]':<15}"
           f"{'same':>5}{'uniq':>7}{'qmt':>7}{'noctl':>7}{'/SD':>7}")
        nk = f"{NOCONTROL_SET}::{axis_of(k)}"
        rows = []
        for fam, dirn in fd:
            lv = natural_top(fam, dirn)
            x = get(fam, dirn, lv) if lv else None
            if x is None:
                continue
            r_ = x.loc[k]
            rows.append((r_["d"], fam, dirn, lv, r_,
                         x.at[nk, "d"] if nk in x.index else np.nan))
        for d_, fam, dirn, lv, r_, dn in sorted(rows, key=lambda t: -t[0]):
            W_(f"    {fam + ' ' + dirn:<22}{lv:<5}{d_:>+7.2f}"
               f"{sig(r_['lo'], r_['hi'])}[{r_['lo']:+.2f}, {r_['hi']:+.2f}] "
               f"{r_['same_sign']:>5.2f}"
               f"{r_['unique']:>+6.2f}{sig(r_['unique_lo'], r_['unique_hi'])}"
               f"{r_['beyond_qmt']:>+6.2f}{sig(r_['beyond_qmt_lo'], r_['beyond_qmt_hi'])}"
               f"{dn:>+7.2f}{r_['d_per_sd']:>+7.2f}")
    W_("")

    table("6. EFFECTS AT THE STRONGEST LEVEL (4x, or the highest reached)",
          ext_top)

    W_("7. DOSE-RESPONSE OF THE MOST AFFECTED v4 AXIS (d at each level)")
    W_("-" * 78)
    for fam, dirn in fd:
        x = eff[(eff["family"] == fam) & (eff["direction"] == dirn) &
                eff["axis"].isin(prim)]
        if not len(x):
            continue
        top = x.loc[x["d"].abs().idxmax(), "axis"]
        y = x[x["axis"] == top]
        W_(f"  {fam + ' ' + dirn:<22}{lab(top):<5}" +
           "  ".join(f"{r.level} {r.d:+.2f}" for r in y.itertuples()))
    W_("")

    W_(f"8. SIDE EFFECTS AT THE TOP OF THE NATURAL RANGE (the sweep's; features "
       f"other than the calibrated one with |median change| >= {SIDE_MIN} "
       "corpus SD; selected features *)")
    W_("-" * 78)
    sd_ = side.fillna({"direction": ""})
    for fam, dirn in fd:
        lv = natural_top(fam, dirn)
        x = sd_[(sd_["family"] == fam) & (sd_["direction"] == dirn) &
                (sd_["level"] == lv) & ~sd_["calibrated"].astype(bool)]
        x = x[x["median_change_sd"].abs() >= SIDE_MIN].copy()
        x = x.reindex(x["median_change_sd"].abs().sort_values(
            ascending=False).index)
        W_(f"  {fam + ' ' + dirn:<22}" + (", ".join(
            f"{r.feature}{'*' if bool(r.selected) else ''} "
            f"{r.median_change_sd:+.1f}" for r in x.itertuples()) or "none"))
    W_("")

    table(f"9. R2b: {NOCONTROL_SET} AT THE TOP OF THE NATURAL RANGE (the v4 "
          "axes without the enclosure and darkness directions)", natural_top,
          ncs, extra=False)
    W_("   Compare with section 2: an effect that shrinks here runs through "
       "the enclosure / darkness")
    W_("   directions (section 10 shows how far the manipulation moves them).")
    W_("")
    table("10. CONTROL AXES AND CONTROL DIRECTIONS AT THE TOP OF THE NATURAL "
          "RANGE", natural_top, ctrls + dirs, extra=False)
    W_("   enclos / darkne = the text directions of decision 17 (+ = more "
       "enclosed / darker); SD")
    W_(f"   estimated from {checks['R5_n_embeddings']:,} stored embeddings.")
    W_("")
    table(f"11. POSITIVE POLES ({CENTROID_SET}) AT THE TOP OF THE NATURAL RANGE",
          natural_top, cents, extra=False)
    table(f"12. {COMPARE_SET} AXES AT THE TOP OF THE NATURAL RANGE (continuity "
          "with the sweep's report)", natural_top, comp, extra=False)
    W_(f"Elapsed: {(time.time() - t0) / 60:.1f} min")
    open(P("sweep_report.txt"), "w").write("\n".join(L) + "\n")

    json.dump({"script": "09c_rescore_sweep_v4.py",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "sweep_dir": SWEEP_DIR, "axes_dir": AXES_DIR,
               "scores_dir": SCORES_DIR, "n_images": int(nI),
               "n_variants_per_image": int(nV), "checks": checks,
               "settings": {"primary": PRIMARY_SET, "nocontrol": NOCONTROL_SET,
                            "compare": COMPARE_SET, "adjust": ADJUST,
                            "key_axes": KEY_AXES, "controls": CONTROLS,
                            "sd_sample": SD_SAMPLE, "n_boot": N_BOOT,
                            "seed": RANDOM_SEED},
               "control_direction_sd": {k: sd_raw[k] for k in dirs},
               "runtime_min": (time.time() - t0) / 60},
              open(P("run_meta.json"), "w"), indent=2, default=float)
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Report: "
          f"{P('sweep_report.txt')}")


if __name__ == "__main__":
    main()
