"""
06_project_axes_v4.py  (pipeline step 06, v4 run)

Scores every image on every axis built in step 05, once, so that steps
07-09 all read the same numbers (review item B11).

v4 run (v4_protocol.md 1.15): the same script as 06_project_axes.py, with
  * the axes of 05_build_axes_v4.py (data/axes_v4: v2, v3 and v4 sets;
    the v2/v3 vectors are identical to the v3 run);
  * a new output folder (data/scores_v4);
  * "v4" among the main sets, and the v4 axes in the report next to v3b;
  * check S5: the v2/v3 columns equal those of the v3 run (data/scores)
    on the S3 images, so 07-09 give the same v2/v3 results as before.

Score
-----
  raw  = cosine between the image embedding (step 03, L2-normalised,
         un2CLIP projection) and the axis vector (step 05, L2-normalised)
       = embedding @ axis. No modality-gap correction (review item B1).
  z    = (raw - mean) / SD, with the mean and SD of each axis taken over
         ALL encoded images (decision of 27 Sep: one scale for both image
         sets). Standardising changes no correlation, ranking or eta^2;
         it only puts the axes on a common scale. Step 07 standardises
         again inside each analysis, on exactly the images it uses.

The mean and SD are saved, so raw and z can be converted into each other.
The same statistics inside `colour_only` (all images minus the greyscale
photographs flagged by 04a) are reported too.

All included axes of axis_catalog.csv are scored (264 in the v4 run),
including the "a scene" baselines (kept to document the change to the
centroid baseline) and the v4@nocontrol variant (R2b).

Checks (the run stops if one fails)
-----------------------------------
  S1 INPUTS     the image index is the one step 03 encoded (fingerprint),
                image_ids are 0..N-1, and the embedding rows match.
  S2 AXES       axis_vectors.npy has one unit-length row per included
                axis in axis_catalog.csv, in order.
  S3 RECOMPUTE  CHECK_N random images rescored in float64 from the
                embeddings: max |difference| <= RECOMPUTE_TOL.
  S4 Z          every z column has mean ~0 and SD ~1 over all images.
  S5 v3 RUN     [v4 run] every axis also in OLD_SCORES_DIR (matched by key)
                has the same raw scores on the S3 images (max |difference|
                <= RECOMPUTE_TOL). A failure is a warning: it means the
                v2/v3 axes differ from those of the v3 run. Skipped if
                OLD_SCORES_DIR is None.

Descriptives (reported; nothing is decided by them)
---------------------------------------------------
  * per axis: raw mean and SD (all images and colour_only), skewness and
    excess kurtosis of z, extreme percentiles. Why: the z-score assumes
    no extreme tails; heavy tails would argue for ranks or a robust
    scale in 07.
  * correlations between the image scores of the axes, pooled over all
    images (not within category; that is 07/09). Two uses:
      - how much the axes of each version overlap at the image level;
      - whether the "a scene" baselines share a common component
        (found in the text in 05): their mutual correlations are
        compared with those of the centroid baselines.

Inputs
------
  INDEX_CSV     data/selection/image_index.csv        (step 02)
  EMB_DIR       data/embeddings/                      (step 03)
  FLAGS_CSV     data/monochrome/image_flags.csv       (step 04a)
  AXES_DIR      data/axes_v4/                         (05_build_axes_v4.py)
  OLD_SCORES_DIR data/scores/                         (the v3 run; check S5)

Outputs (under OUTPUT_DIR)
--------------------------
  scores_raw.npy          (N, K) float32   cosine; NaN for unencoded images
  scores_z.npy            (N, K) float32   z-score over all images
  image_ids.npy           (N,)   int64     row i = image_id i
  score_catalog.csv       one row per column: col, key, set, kind, version,
                          template, axis, primary, mean_raw, sd_raw,
                          mean_raw_colour, sd_raw_colour, mean_z_colour,
                          sd_z_colour, skew_z, kurt_z, z_p001, z_p01,
                          z_p99, z_p999, z_min, z_max
  score_correlations_all.csv      (K, K) Pearson r of the scores, all images
  score_correlations_colour.csv   (K, K) the same in colour_only
  score_report.txt        summary (start here)
  run_meta.json           settings, input fingerprints, check results

Column j of both score arrays is the axis with `col` = j in
score_catalog.csv (the same as `row` in data/axes_v4/axis_catalog.csv).

Usage
-----
    conda activate drl2drv
    python 06_project_axes_v4.py

Reads the 3.4 GB embedding file twice in chunks; writes about 2.4 GB
(264 axes).
A few minutes.
"""

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INDEX_CSV       = "data/selection/image_index.csv"
EMB_DIR         = "data/embeddings"
FLAGS_CSV       = "data/monochrome/image_flags.csv"
AXES_DIR        = "data/axes_v4"          # 05_build_axes_v4.py
OUTPUT_DIR      = "data/scores_v4"        # new folder (the v3 run: data/scores)
OLD_SCORES_DIR  = "data/scores"           # check S5; None to skip

PRIMARY_SET     = "v3b"
V4_SET          = "v4"
MAIN_SETS       = ["v2", "v2_merged", "v3a", "v3b", "v3b_neg", V4_SET]

CHUNK           = 100_000
CHECK_N         = 1_000
RECOMPUTE_TOL   = 1e-5
Z_TOL           = 1e-3          # |mean| and |SD - 1| allowed for check S4
PERCENTILE_SAMPLE = 200_000     # images used for the extreme percentiles
RANDOM_SEED     = 42


# ============================================================
# Helpers
# ============================================================

def P(name):
    return os.path.join(OUTPUT_DIR, name)


def fingerprint(idx):
    """Same as step 03."""
    h = hashlib.sha1()
    for k, p in zip(idx["image_key"], idx["path"]):
        h.update(f"{k}\t{p}\n".encode("utf-8"))
    return h.hexdigest()


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Moments:
    """Running sums for means, SDs and correlations (float64)."""

    def __init__(self, k):
        self.n = np.zeros(k)
        self.s = np.zeros(k)
        self.ss = np.zeros(k)
        self.xtx = np.zeros((k, k))
        self.nr = 0                       # complete rows used for xtx

    def add(self, X):
        ok = ~np.isnan(X)
        Xf = np.where(ok, X, 0.0).astype(np.float64)
        self.n += ok.sum(0)
        self.s += Xf.sum(0)
        self.ss += (Xf * Xf).sum(0)
        full = ok.all(1)
        Xc = Xf[full]
        self.xtx += Xc.T @ Xc
        self.nr += int(full.sum())

    def mean(self):
        return self.s / self.n

    def sd(self):
        m = self.mean()
        return np.sqrt(np.maximum(self.ss / self.n - m * m, 0) * self.n
                       / (self.n - 1))

    def corr(self):
        # complete rows only; with no missing scores this equals n
        m = self.s / self.n
        cov = self.xtx / self.nr - np.outer(m, m)
        d = np.sqrt(np.diag(cov))
        return cov / np.outer(d, d)


def offdiag_mean(M):
    iu = np.triu_indices(len(M), 1)
    return float(M[iu].mean()), float(M[iu].max())


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    print("=" * 70)
    print("06_project_axes_v4.py")
    print("=" * 70)
    checks = {}

    # --- S1: inputs ---
    idx = pd.read_csv(INDEX_CSV, usecols=["image_id", "image_key", "path",
                                          "category"])
    n = len(idx)
    if not (idx["image_id"].to_numpy() == np.arange(n)).all():
        raise SystemExit("S1: image_id must be 0..N-1 in row order.")
    with open(os.path.join(EMB_DIR, "run_meta.json")) as f:
        emb_meta = json.load(f)
    fp = fingerprint(idx)
    if emb_meta.get("index_fingerprint") != fp:
        raise SystemExit("S1: image_index.csv differs from the one step 03 "
                         "encoded (fingerprint mismatch).")
    E = np.load(os.path.join(EMB_DIR, "embeddings_768.npy"), mmap_mode="r")
    status = np.load(os.path.join(EMB_DIR, "status.npy"))
    ids = np.load(os.path.join(EMB_DIR, "image_ids.npy"))
    if E.shape[0] != n or len(status) != n or not (ids == np.arange(n)).all():
        raise SystemExit(f"S1: embeddings ({E.shape[0]}) do not match the "
                         f"index ({n}).")
    ok_img = status == 1
    print(f"Images   : {n:,} ({int(ok_img.sum()):,} encoded)")
    checks["S1"] = {"n_images": n, "n_encoded": int(ok_img.sum()),
                    "index_fingerprint": fp}

    flags = pd.read_csv(FLAGS_CSV, usecols=["image_id", "is_greyscale"])
    flags = flags.set_index("image_id").reindex(np.arange(n))
    if flags["is_greyscale"].isna().any():
        raise SystemExit("S1: image_flags.csv does not cover every image_id.")
    colour = ~flags["is_greyscale"].astype(bool).to_numpy()
    print(f"Colour   : {int(colour.sum()):,} images in colour_only "
          f"({int((~colour).sum()):,} greyscale)")

    # --- S2: axes ---
    A = np.load(os.path.join(AXES_DIR, "axis_vectors.npy")).astype(np.float32)
    cat = pd.read_csv(os.path.join(AXES_DIR, "axis_catalog.csv"))
    cat = cat[cat["included"]].sort_values("row").reset_index(drop=True)
    k = len(cat)
    norms = np.linalg.norm(A, axis=1)
    if A.shape[0] != k or not (cat["row"].to_numpy() == np.arange(k)).all():
        raise SystemExit(f"S2: {A.shape[0]} vectors but {k} included axes, "
                         "or catalog rows not 0..K-1.")
    if np.abs(norms - 1).max() > 1e-4:
        raise SystemExit("S2: axis vectors are not unit length.")
    print(f"Axes     : {k} (from {AXES_DIR})")
    checks["S2"] = {"n_axes": k, "max_norm_error": float(np.abs(norms - 1).max()),
                    "axis_vectors_sha256": sha256(os.path.join(AXES_DIR,
                                                               "axis_vectors.npy"))}

    # --- Pass 1: raw scores ---
    fmt = np.lib.format
    raw = fmt.open_memmap(P("scores_raw.npy"), mode="w+", dtype=np.float32,
                          shape=(n, k))
    m_all, m_col = Moments(k), Moments(k)
    print("\nPass 1: raw scores...")
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        X = np.asarray(E[a:b], dtype=np.float32) @ A.T
        X[~ok_img[a:b]] = np.nan
        raw[a:b] = X
        m_all.add(X)
        m_col.add(X[colour[a:b]])
        print(f"  {b:>10,} / {n:,}", end="\r")
    raw.flush()
    print()
    mu, sd = m_all.mean(), m_all.sd()
    mu_c, sd_c = m_col.mean(), m_col.sd()

    # --- S3: recompute ---
    pick = np.sort(rng.choice(np.where(ok_img)[0], min(CHECK_N, int(ok_img.sum())),
                              replace=False))
    ref = np.asarray(E[pick], dtype=np.float64) @ A.astype(np.float64).T
    diff = float(np.abs(ref - raw[pick]).max())
    ok3 = diff <= RECOMPUTE_TOL
    print(f"  S3 recompute: {len(pick)} images, max |diff| {diff:.2e} -> "
          f"{'PASS' if ok3 else 'FAIL'}")
    checks["S3"] = {"n": int(len(pick)), "max_abs_diff": diff}
    if not ok3:
        raise SystemExit("S3: stored scores differ from recomputed ones.")

    # --- Pass 2: z-scores, higher moments ---
    zarr = fmt.open_memmap(P("scores_z.npy"), mode="w+", dtype=np.float32,
                           shape=(n, k))
    mz_all, mz_col = Moments(k), Moments(k)
    s3, s4 = np.zeros(k), np.zeros(k)
    zmin, zmax = np.full(k, np.inf), np.full(k, -np.inf)
    print("Pass 2: z-scores...")
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        Z = ((raw[a:b].astype(np.float64) - mu) / sd)
        zarr[a:b] = Z.astype(np.float32)
        Zf = np.where(np.isnan(Z), 0.0, Z)
        s3 += (Zf ** 3).sum(0)
        s4 += (Zf ** 4).sum(0)
        zmin = np.minimum(zmin, np.nanmin(Z, axis=0))
        zmax = np.maximum(zmax, np.nanmax(Z, axis=0))
        mz_all.add(Z)
        mz_col.add(Z[colour[a:b]])
        print(f"  {b:>10,} / {n:,}", end="\r")
    zarr.flush()
    print()
    np.save(P("image_ids.npy"), np.arange(n, dtype=np.int64))

    # --- S4: z check ---
    zm, zs = mz_all.mean(), mz_all.sd()
    ok4 = np.abs(zm).max() <= Z_TOL and np.abs(zs - 1).max() <= Z_TOL
    print(f"  S4 z-scores: max |mean| {np.abs(zm).max():.1e}, max |SD-1| "
          f"{np.abs(zs - 1).max():.1e} -> {'PASS' if ok4 else 'FAIL'}")
    checks["S4"] = {"max_abs_mean": float(np.abs(zm).max()),
                    "max_abs_sd_minus_1": float(np.abs(zs - 1).max())}
    if not ok4:
        raise SystemExit("S4: z-scores are not standardised.")

    # --- S5: v2/v3 columns equal the v3 run ---
    s5 = None
    if OLD_SCORES_DIR and os.path.exists(os.path.join(OLD_SCORES_DIR, "score_catalog.csv")):
        old_cat = pd.read_csv(os.path.join(OLD_SCORES_DIR, "score_catalog.csv"))
        old_col = dict(zip(old_cat["key"], old_cat["col"]))
        both = [(j, old_col[kk]) for j, kk in enumerate(cat["key"]) if kk in old_col]
        old_raw = np.load(os.path.join(OLD_SCORES_DIR, "scores_raw.npy"), mmap_mode="r")
        if old_raw.shape[0] != n:
            raise SystemExit("S5: the v3 run's scores have a different number of images.")
        new_j = [j for j, _ in both]
        old_j = [o for _, o in both]
        d5 = float(np.nanmax(np.abs(np.asarray(raw[pick])[:, new_j]
                                    - np.asarray(old_raw[pick])[:, old_j])))
        ok5 = d5 <= RECOMPUTE_TOL
        n_new = int(sum(1 for kk in cat["key"] if kk not in old_col))
        s5 = {"n_matched": len(both), "n_new": n_new, "max_abs_diff": d5, "pass": ok5}
        print(f"  S5 v3 run: {len(both)} axes matched, max |diff| {d5:.2e} -> "
              f"{'PASS' if ok5 else 'WARN'}; {n_new} new axes")
        checks["S5"] = s5
    else:
        print("  S5 v3 run: skipped")
        checks["S5"] = "skipped"

    # --- Descriptives ---
    nn = mz_all.n
    skew = s3 / nn
    kurt = s4 / nn - 3.0
    samp = np.sort(rng.choice(np.where(ok_img)[0],
                              min(PERCENTILE_SAMPLE, int(ok_img.sum())),
                              replace=False))
    Zs = np.asarray(zarr[samp], dtype=np.float64)
    pct = np.percentile(Zs, [0.1, 1, 99, 99.9], axis=0)
    sc = cat[["row", "key", "set", "kind", "version", "template", "axis",
              "primary"]].rename(columns={"row": "col"}).copy()
    sc["mean_raw"], sc["sd_raw"] = mu, sd
    sc["mean_raw_colour"], sc["sd_raw_colour"] = mu_c, sd_c
    sc["mean_z_colour"], sc["sd_z_colour"] = mz_col.mean(), mz_col.sd()
    sc["skew_z"], sc["kurt_z"] = skew, kurt
    sc["z_p001"], sc["z_p01"], sc["z_p99"], sc["z_p999"] = pct
    sc["z_min"], sc["z_max"] = zmin, zmax
    sc.to_csv(P("score_catalog.csv"), index=False, float_format="%.6g")

    R_all, R_col = mz_all.corr(), mz_col.corr()
    keys = cat["key"].tolist()
    pd.DataFrame(R_all, index=keys, columns=keys).to_csv(
        P("score_correlations_all.csv"), float_format="%.4f")
    pd.DataFrame(R_col, index=keys, columns=keys).to_csv(
        P("score_correlations_colour.csv"), float_format="%.4f")

    # --- Report ---
    L = ["06_project_axes_v4.py -- summary", "=" * 70,
         f"Date     : {time.strftime('%Y-%m-%d %H:%M')}",
         f"Images   : {n:,} ({int(ok_img.sum()):,} encoded; colour_only "
         f"{int(colour.sum()):,})",
         f"Axes     : {k} (all included axes of {AXES_DIR}/axis_catalog.csv)",
         "Score    : raw = cosine(image, axis); z over all images", "",
         "CHECKS",
         "  S1 inputs    : index fingerprint = step 03; ids and rows aligned",
         f"  S2 axes      : {k} unit vectors in catalog order",
         f"  S3 recompute : {len(pick)} images, max |diff| {diff:.1e} -> PASS",
         f"  S4 z-scores  : max |mean| {np.abs(zm).max():.1e}, max |SD-1| "
         f"{np.abs(zs - 1).max():.1e} -> PASS",
         ("  S5 v3 run    : skipped" if s5 is None else
          f"  S5 v3 run    : {s5['n_matched']} axes equal to {OLD_SCORES_DIR} "
          f"(max |diff| {s5['max_abs_diff']:.1e}) -> "
          f"{'PASS' if s5['pass'] else 'WARN: the v2/v3 axes differ from the v3 run'}"
          f"; {s5['n_new']} new axes"), ""]

    L += [f"DISTRIBUTIONS, {PRIMARY_SET} (primary), {V4_SET} and controls",
          "  raw mean / SD over all images; z mean / SD inside colour_only "
          "(shift caused by leaving out greyscale photos); skew and excess "
          "kurtosis of z (0 for a normal distribution); 0.1st and 99.9th "
          "percentiles of z"]
    for r in sc[sc["set"].isin([PRIMARY_SET, V4_SET]) | (sc["kind"] == "control")].itertuples():
        L.append(f"  {r.set:<9} {r.axis[:30]:<30} raw {r.mean_raw:+.4f} / "
                 f"{r.sd_raw:.4f}   colour z {r.mean_z_colour:+.3f} / "
                 f"{r.sd_z_colour:.3f}   skew {r.skew_z:+.2f}  kurt "
                 f"{r.kurt_z:+.2f}   z {r.z_p001:+.2f} .. {r.z_p999:+.2f}")
    heavy = sc.reindex(sc["kurt_z"].abs().sort_values(ascending=False).index)[:5]
    L += ["", "  Heaviest tails among all axes (largest |excess kurtosis|):"]
    L += [f"    {r.key[:58]:<58} skew {r.skew_z:+.2f}  kurt {r.kurt_z:+.2f}"
          for r in heavy.itertuples()]

    L += ["", "CORRELATIONS BETWEEN IMAGE SCORES (pooled over all images; "
          "not within category)",
          "  mean and highest correlation between the axes of each set"]
    for s_ in MAIN_SETS + [f"{m}@centroid" for m in MAIN_SETS] + \
            [f"{m}@neutral" for m in MAIN_SETS]:
        j = np.where(cat["set"].to_numpy() == s_)[0]
        if len(j) > 1:
            m_, x_ = offdiag_mean(R_all[np.ix_(j, j)])
            L.append(f"    {s_:<20} mean {m_:+.3f}   highest {x_:+.3f}")
    j = np.where(cat["set"].to_numpy() == PRIMARY_SET)[0]
    Rp = pd.DataFrame(R_all[np.ix_(j, j)],
                      index=[a.split(" / ")[0][:12] for a in cat["axis"].iloc[j]],
                      columns=[a.split(" / ")[0][:12] for a in cat["axis"].iloc[j]])
    L += ["", f"  {PRIMARY_SET}, all images:"]
    L += ["  " + x for x in Rp.round(2).to_string().splitlines()]
    Rc = pd.DataFrame(R_col[np.ix_(j, j)], index=Rp.index, columns=Rp.columns)
    dmax = float(np.abs(Rc.values - Rp.values).max())
    L += [f"  {PRIMARY_SET}, colour_only: largest difference from all images "
          f"{dmax:.3f}"]

    # v4: the same tables, the nocontrol version and the closest v3b axis
    j4 = np.where(cat["set"].to_numpy() == V4_SET)[0]
    if len(j4):
        n4 = [a[:3] for a in cat["axis"].iloc[j4]]
        R4 = pd.DataFrame(R_all[np.ix_(j4, j4)], index=n4, columns=n4)
        L += ["", f"  {V4_SET}, all images:"]
        L += ["  " + x for x in R4.round(2).to_string().splitlines()]
        R4c = pd.DataFrame(R_col[np.ix_(j4, j4)], index=n4, columns=n4)
        L += [f"  {V4_SET}, colour_only: largest difference from all images "
              f"{float(np.abs(R4c.values - R4.values).max()):.3f}"]
        keyj = {kk: j for j, kk in enumerate(cat["key"])}
        L += ["", f"  {V4_SET} against its nocontrol version (R2b) and the closest "
              f"{PRIMARY_SET} axis (pooled r; counterparts are decided in 09 within "
              "categories, protocol 6.5)"]
        for j in j4:
            ax_ = cat["axis"].iloc[j]
            jn = keyj.get(f"{V4_SET}@nocontrol::{ax_}")
            rn = R_all[j, jn] if jn is not None else np.nan
            jj = np.where(cat["set"].to_numpy() == PRIMARY_SET)[0]
            b = jj[int(np.argmax(np.abs(R_all[j, jj])))]
            L.append(f"    {ax_[:36]:<36} nocontrol r {rn:.3f}   closest {PRIMARY_SET}: "
                     f"{cat['axis'].iloc[b].split(' / ')[0][:20]:<20} r {R_all[j, b]:+.3f}")

    L += ["", f"Files in {OUTPUT_DIR}: scores_raw.npy ({n}, {k}), scores_z.npy "
          f"({n}, {k}), image_ids.npy, score_catalog.csv, "
          "score_correlations_all.csv, score_correlations_colour.csv",
          f"Elapsed: {(time.time() - t0) / 60:.1f} min"]
    report = "\n".join(L)
    with open(P("score_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)

    meta = {"script": "06_project_axes_v4.py",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "inputs": {"index_csv": INDEX_CSV, "emb_dir": EMB_DIR,
                       "flags_csv": FLAGS_CSV, "axes_dir": AXES_DIR,
                       "old_scores_dir": OLD_SCORES_DIR},
            "score": "raw = embedding @ axis (cosine); z over all encoded images",
            "n_images": n, "n_axes": k, "n_colour_only": int(colour.sum()),
            "seed": RANDOM_SEED, "checks": checks}
    with open(P("run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
