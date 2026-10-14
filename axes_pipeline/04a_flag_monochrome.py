"""
04a_flag_monochrome.py  (pipeline step 04a)

Flags black-and-white photographs, and single-tint photographs (sepia,
blue-toned, duotone), so later steps can run each analysis on colour
images only and on all images.

Why
---
About 3% of the corpus (32,744 images with mean chroma < 1) has no
colour at all, concentrated in categories such as fire_escape, alley,
elevator_shaft, jail_cell and corridor -- which also score high on the
lifeless axis. A black-and-white photograph is a photographic STYLE,
not a desaturated SCENE, so these images can inflate the relationship
between "lifeless" and low saturation. Toned monochrome photographs
have the same problem but can have fairly high mean chroma, so mean
chroma alone does not find them.

The script runs in two modes (MODE setting).

MODE = "compute"   (run first, ~15 minutes)
------------------------------------------
  1. For EVERY image (same central-square region and size as step 04),
     measures how much colour there is and how it is spread:

       chroma_mean         mean CIELAB chroma C*ab (as in step 04)
       chroma_p95          95th percentile of per-pixel chroma: how
                           colourful the most colourful 5% of the image
                           is. ~0-2 for black-and-white photographs
                           (only compression noise), clearly higher for
                           colour photographs even of grey scenes.
       colourful_frac      share of pixels with chroma > COLOURFUL_C
       hue_concentration   0-1; how much all the colour points the same
                           way (1 = a single hue everywhere). Computed on
                           pixels with chroma > HUE_MIN_C; NaN if fewer
                           than MIN_HUE_FRAC of pixels qualify.
       ab_sd               spread of the colours in the a*b* plane
       colour_given_L      0-1; how much of the colour variation is
                           explained by lightness alone. Toned
                           photographs are ~1 (each grey level maps to
                           one tint); colour photographs are lower,
                           because objects of the same lightness have
                           different colours. NaN if the image has
                           almost no colour variation.

  2. CALCULATED thresholds (Otsu's method: the split that best separates
     the values into two groups), with histograms to judge whether the
     data really form two groups:
       greyscale  on log10(chroma_p95 + 0.1), all images
       toned      on colour_given_L, among images that are not
                  greyscale and have chroma_mean < TONED_MAX_CHROMA

  3. A LABELLING PACK for checking those thresholds by eye:
       * GREY_SAMPLE images spread across the greyscale boundary and
         TONED_SAMPLE images spread across the toned boundary, plus
         N_REPEATS repeated images to measure your own consistency;
       * a thumbnail of each (the whole photograph), and
       * label_images.html, a page that shows them one at a time in
         random order, WITHOUT any measured value, with four buttons:
           1 greyscale   2 single tint   3 colour   4 unsure
         and saves your answers as a CSV file.

  Open label_images.html in a browser, label all images, download the
  CSV and save it as LABELS_CSV (default: <OUTPUT_DIR>/labels.csv).

MODE = "finalize"   (after labelling, ~1 minute)
-----------------------------------------------
  4. Reads your labels and CALCULATES the thresholds from them: for each
     rule, the cut-off that misclassifies the fewest labelled images.
     Reports the error at that cut-off, the error of the Otsu threshold,
     and your consistency on the repeated images.
  5. Writes image_flags.csv for every image, using THRESHOLD_SOURCE:
       "labels"  the label-based thresholds (recommended)
       "otsu"    the calculated thresholds from step 2
       "manual"  MANUAL_GREY_P95 and MANUAL_TONED_CGL

The flags:
  is_greyscale  chroma_p95 < t_grey
  is_toned      not greyscale, chroma_mean < TONED_MAX_CHROMA and
                colour_given_L >= t_toned
  is_monochrome is_greyscale or is_toned

Decide the thresholds before looking at any correlation with the axes,
and record the date and the source in the methods.

Inputs
------
  INDEX_CSV       data/selection/image_index.csv (step 02)
  FEATURES_DIR    data/features (step 04; for chroma_mean and the check)

Outputs (under OUTPUT_DIR)
--------------------------
  compute:
    colour_measures.npy / .csv    (N, 6) the measures above
    status.npy, run_meta.json
    histograms.png                distributions with Otsu thresholds
    measures_scatter.png          hue_concentration vs colour_given_L
    label_images/                 thumbnails for labelling
    label_images.html             the labelling page
    label_key.csv                 sample_id -> image_id, measures, reason
                                  (do not open while labelling)
    compute_report.txt
  finalize:
    image_flags.csv               image_id, flags, measures
    threshold_analysis.csv        errors at every candidate cut-off
    flags_report.txt

Usage
-----
    conda activate drl2drv
    python 04a_flag_monochrome.py          # MODE = "compute"
    # label the images, save labels.csv
    python 04a_flag_monochrome.py          # MODE = "finalize"
"""

import hashlib
import html
import json
import multiprocessing as mp
import os
import time
import warnings

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

MODE                = "finalize"     # "compute" or "finalize"

# --- Inputs / output ---
INDEX_CSV           = "data/selection/image_index.csv"
FEATURES_DIR        = "data/features"
OUTPUT_DIR          = "data/monochrome"

# Dry run: first TEST_LIMIT images only (None = all; separate OUTPUT_DIR)
TEST_LIMIT          = None

# --- Region (same as step 04) ---
FEATURE_SIZE        = 256

# --- Measures ---
# COLOURFUL_C    per-pixel chroma above which a pixel counts as
#                "colourful" (CIELAB units; ~10 is a clearly visible
#                colour).
# HUE_MIN_C      per-pixel chroma above which a pixel's hue is reliable
#                enough to use (below it, the hue is noise).
# MIN_HUE_FRAC   minimum share of such pixels for hue_concentration.
# L_BINS         lightness bins for colour_given_L.
COLOURFUL_C         = 10.0
HUE_MIN_C           = 5.0
MIN_HUE_FRAC        = 0.01
L_BINS              = 16
MIN_AB_VAR          = 1.0          # below this, colour_given_L is NaN

# --- Toned rule ---
# Toned monochrome is only considered among images with mean chroma
# below this: a strongly coloured image is not a toned photograph.
TONED_MAX_CHROMA    = 20.0

# --- Labelling pack ---
GREY_SAMPLE         = 150          # across the greyscale boundary
TONED_SAMPLE        = 150          # across the toned boundary
N_REPEATS           = 20           # shown twice, for consistency
SAMPLE_BINS         = 10           # strata across each boundary
THUMB_MAX_SIDE      = 512
RANDOM_SEED         = 42

# --- Finalize ---
LABELS_CSV          = None         # None -> <OUTPUT_DIR>/labels.csv
THRESHOLD_SOURCE    = "labels"     # "labels", "otsu" or "manual"
MANUAL_GREY_P95     = None         # chroma_p95 cut-off, if "manual"
MANUAL_TONED_CGL    = None         # colour_given_L cut-off, if "manual"

# --- Parallelism ---
N_WORKERS           = 24
CHUNK               = 64
FLUSH_EVERY_TASKS   = 100


MEASURES = ["chroma_mean", "chroma_p95", "colourful_frac",
            "hue_concentration", "ab_sd", "colour_given_L"]
LABELS = {1: "greyscale", 2: "toned", 3: "colour", 4: "unsure"}


# ============================================================
# Region (identical to step 04)
# ============================================================

def clip_box(w, h, n=336):
    if w <= h:
        nw, nh = n, int(n * h / w)
    else:
        nw, nh = int(n * w / h), n
    left = int(round((nw - n) / 2.0))
    top = int(round((nh - n) / 2.0))
    sx, sy = nw / w, nh / h
    return (left / sx, top / sy, (left + n) / sx, (top + n) / sy)


def central_square(im, size):
    return im.resize((size, size), Image.BILINEAR, box=clip_box(*im.size))


# ============================================================
# Measures
# ============================================================

def colour_measures(rgb_u8):
    lab = cv2.cvtColor(rgb_u8.astype(np.float32) / 255.0,
                       cv2.COLOR_RGB2Lab)
    L = lab[:, :, 0].ravel()
    a = lab[:, :, 1].ravel().astype(np.float64)
    b = lab[:, :, 2].ravel().astype(np.float64)
    C = np.sqrt(a ** 2 + b ** 2)

    chroma_mean = C.mean()
    chroma_p95 = np.percentile(C, 95)
    colourful = (C > COLOURFUL_C).mean()

    sel = C > HUE_MIN_C
    if sel.mean() >= MIN_HUE_FRAC:
        # Resultant length of the chroma vectors relative to their total
        # length: 1 when every coloured pixel has the same hue.
        hue_conc = np.hypot(a[sel].sum(), b[sel].sum()) / C[sel].sum()
    else:
        hue_conc = np.nan

    var_a, var_b = a.var(), b.var()
    ab_sd = np.sqrt(var_a + var_b)
    total = var_a + var_b
    if total >= MIN_AB_VAR:
        bins = np.minimum((L / 100.0 * L_BINS).astype(int), L_BINS - 1)
        n = np.bincount(bins, minlength=L_BINS).astype(np.float64)
        within = 0.0
        for arr in (a, b):
            s1 = np.bincount(bins, arr, minlength=L_BINS)
            s2 = np.bincount(bins, arr ** 2, minlength=L_BINS)
            ok = n > 0
            within += (s2[ok] - s1[ok] ** 2 / n[ok]).sum()
        cgl = 1.0 - (within / len(a)) / total
    else:
        cgl = np.nan
    return [chroma_mean, chroma_p95, colourful, hue_conc, ab_sd, cgl]


def image_measures(path):
    with Image.open(path) as im:
        sq = central_square(im.convert("RGB"), FEATURE_SIZE)
    return colour_measures(np.asarray(sq, dtype=np.uint8))


def _init_worker():
    cv2.setNumThreads(1)
    warnings.filterwarnings("ignore")


def work(task):
    ids, paths = task
    out = np.full((len(ids), len(MEASURES)), np.nan, np.float32)
    ok = np.zeros(len(ids), dtype=bool)
    for j, p in enumerate(paths):
        try:
            out[j] = image_measures(p)
            ok[j] = True
        except Exception:
            pass
    return ids, out, ok


# ============================================================
# Helpers
# ============================================================

def P(name):
    return os.path.join(OUTPUT_DIR, name)


def fingerprint(idx):
    h = hashlib.sha1()
    for k, p in zip(idx["image_key"], idx["path"]):
        h.update(f"{k}\t{p}\n".encode("utf-8"))
    return h.hexdigest()


def save_status(status):
    np.save(P("status.tmp.npy"), status)
    os.replace(P("status.tmp.npy"), P("status.npy"))


def otsu(values, n_bins=256):
    """Otsu's threshold: the cut that maximises the between-group
    variance of a 1-D sample."""
    v = values[np.isfinite(values)]
    hist, edges = np.histogram(v, bins=n_bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w = hist.astype(np.float64)
    w0 = np.cumsum(w)
    w1 = w0[-1] - w0
    m0 = np.cumsum(w * mids) / np.maximum(w0, 1e-12)
    m1 = ((w * mids).sum() - np.cumsum(w * mids)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    k = int(np.argmax(between[:-1]))
    return float(edges[k + 1])


def load_index():
    idx = pd.read_csv(INDEX_CSV)
    if TEST_LIMIT is not None:
        idx = idx.iloc[:TEST_LIMIT].copy()
    if not (idx["image_id"].to_numpy() == np.arange(len(idx))).all():
        raise RuntimeError("image_id must be 0..N-1 in row order.")
    return idx


GREY_TRANSFORM = lambda x: np.log10(x + 0.1)
GREY_INVERSE = lambda y: 10 ** y - 0.1


# ============================================================
# MODE = "compute"
# ============================================================

def compute_measures(idx):
    n = len(idx)
    meta = {"index_fingerprint": fingerprint(idx), "n_images": n,
            "settings": {"feature_size": FEATURE_SIZE,
                         "colourful_c": COLOURFUL_C,
                         "hue_min_c": HUE_MIN_C,
                         "min_hue_frac": MIN_HUE_FRAC, "l_bins": L_BINS,
                         "min_ab_var": MIN_AB_VAR}}
    fmt = np.lib.format
    if os.path.exists(P("run_meta.json")) and os.path.exists(P("status.npy")):
        old = json.load(open(P("run_meta.json")))
        diffs = [k for k in ("index_fingerprint", "n_images", "settings")
                 if old.get(k) != meta.get(k)]
        if diffs:
            raise RuntimeError(f"Existing run differs in {diffs}; use a "
                               "new OUTPUT_DIR or delete it.")
        M = fmt.open_memmap(P("colour_measures.npy"), mode="r+")
        status = np.load(P("status.npy"))
        print(f"Resuming: {int((status != 0).sum()):,} of {n:,} done.")
    else:
        M = fmt.open_memmap(P("colour_measures.npy"), mode="w+",
                            dtype=np.float32, shape=(n, len(MEASURES)))
        M[:] = np.nan
        status = np.zeros(n, dtype=np.int8)
        save_status(status)
    json.dump(meta, open(P("run_meta.json"), "w"), indent=2)

    todo = np.where(status == 0)[0]
    paths = idx["path"].to_numpy()
    tasks = [(todo[i:i + CHUNK], paths[todo[i:i + CHUNK]])
             for i in range(0, len(todo), CHUNK)]
    print(f"To measure: {len(todo):,} images")
    if tasks:
        with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
            k = 0
            try:
                for ids, out, ok in tqdm(pool.imap_unordered(work, tasks),
                                         total=len(tasks), unit="task"):
                    M[ids] = out
                    status[ids] = np.where(ok, 1, -1).astype(np.int8)
                    k += 1
                    if k % FLUSH_EVERY_TASKS == 0:
                        M.flush()
                        save_status(status)
            finally:
                M.flush()
                save_status(status)
    return np.asarray(M), status


def check_against_step04(M, status):
    """chroma_mean here and in step 04 come from the same region and
    formula; they must agree."""
    fd = json.load(open(os.path.join(FEATURES_DIR,
                                     "feature_definitions.json")))
    F = np.load(os.path.join(FEATURES_DIR, "features_crop.npy"),
                mmap_mode="r")
    j = fd["crop"].index("chroma_mean")
    ok = np.where(status == 1)[0]
    rng = np.random.default_rng(RANDOM_SEED)
    pick = np.sort(rng.choice(ok, size=min(5000, len(ok)), replace=False))
    d = np.abs(M[pick, 0] - F[pick, j])
    worst = float(np.nanmax(d))
    print(f"  check (chroma_mean vs step 04, {len(pick)} images): max abs "
          f"diff {worst:.2e} -> {'PASS' if worst < 1e-3 else 'FAIL'}")
    if worst >= 1e-3:
        raise RuntimeError("Measures do not match step 04: region or "
                           "image list differs.")
    return worst


def make_sample(M, rng, t_grey, t_toned):
    """Stratified samples across both boundaries, plus repeats."""
    cm, p95, cgl = M[:, 0], M[:, 1], M[:, 5]
    rows = []

    # Greyscale boundary: log-spaced strata from 0 to 4x the threshold.
    g = GREY_TRANSFORM(p95)
    lo, hi = np.nanmin(g), GREY_TRANSFORM(4 * max(t_grey, 0.25))
    edges = np.linspace(lo, hi, SAMPLE_BINS + 1)
    per = int(np.ceil(GREY_SAMPLE / SAMPLE_BINS))
    for k in range(SAMPLE_BINS):
        pool = np.where((g >= edges[k]) & (g < edges[k + 1]))[0]
        take = rng.choice(pool, size=min(per, len(pool)), replace=False)
        rows += [(int(i), "grey_boundary") for i in take]

    # Toned boundary: candidates as in the rule, strata across cgl.
    cand = np.where((p95 >= t_grey) & (cm < TONED_MAX_CHROMA)
                    & np.isfinite(cgl))[0]
    q = np.nanquantile(cgl[cand], [0.5, 0.999])
    edges = np.linspace(min(q[0], t_toned - 0.2), q[1], SAMPLE_BINS + 1)
    per = int(np.ceil(TONED_SAMPLE / SAMPLE_BINS))
    have = {i for i, _ in rows}
    for k in range(SAMPLE_BINS):
        pool = cand[(cgl[cand] >= edges[k]) & (cgl[cand] < edges[k + 1])]
        pool = np.array([i for i in pool if i not in have])
        if len(pool):
            take = rng.choice(pool, size=min(per, len(pool)), replace=False)
            rows += [(int(i), "toned_boundary") for i in take]
            have.update(int(i) for i in take)

    uniq = [i for i, _ in rows]
    reps = rng.choice(uniq, size=min(N_REPEATS, len(uniq)), replace=False)
    rows += [(int(i), "repeat") for i in reps]
    order = rng.permutation(len(rows))
    return [rows[o] for o in order]


def write_labelling_pack(idx, M, sample):
    img_dir = P("label_images")
    os.makedirs(img_dir, exist_ok=True)
    key = []
    for sid, (i, reason) in enumerate(sample):
        fname = f"{sid:04d}.jpg"
        with Image.open(idx["path"].iat[i]) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE), Image.LANCZOS)
            im.save(os.path.join(img_dir, fname), quality=92)
        key.append({"sample_id": sid, "file": fname, "image_id": i,
                    "reason": reason, "category": idx["category"].iat[i],
                    **{m: float(M[i, j]) for j, m in enumerate(MEASURES)}})
    pd.DataFrame(key).to_csv(P("label_key.csv"), index=False)

    items = json.dumps([{"id": k["sample_id"], "src": "label_images/"
                         + k["file"]} for k in key])
    page = LABEL_PAGE.replace("__ITEMS__", items).replace(
        "__N__", str(len(key)))
    with open(P("label_images.html"), "w", encoding="utf-8") as f:
        f.write(page)
    return len(key)


def plot_compute(M, t_grey, t_toned):
    cm, p95, cgl, hc = M[:, 0], M[:, 1], M[:, 5], M[:, 3]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    g = GREY_TRANSFORM(p95)
    ax[0].hist(g[np.isfinite(g)], bins=200, color="#555")
    ax[0].axvline(GREY_TRANSFORM(t_grey), color="C3",
                  label=f"Otsu: chroma_p95 = {t_grey:.2f}")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("log10(chroma_p95 + 0.1)")
    ax[0].set_ylabel("images (log scale)")
    ax[0].set_title("Greyscale: two groups should be visible")
    ax[0].legend()
    cand = (p95 >= t_grey) & (cm < TONED_MAX_CHROMA) & np.isfinite(cgl)
    ax[1].hist(cgl[cand], bins=200, color="#555")
    ax[1].axvline(t_toned, color="C3",
                  label=f"Otsu: colour_given_L = {t_toned:.3f}")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("colour_given_L")
    ax[1].set_title(f"Toned: non-greyscale images, chroma_mean < "
                    f"{TONED_MAX_CHROMA:g}")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(P("histograms.png"), dpi=130)
    plt.close(fig)

    rng = np.random.default_rng(RANDOM_SEED)
    s = np.where(np.isfinite(cgl) & np.isfinite(hc))[0]
    s = rng.choice(s, size=min(60000, len(s)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(cgl[s], hc[s], c=np.minimum(cm[s], 30), s=2,
                    cmap="viridis", alpha=0.5)
    ax.axvline(t_toned, color="C3", lw=1)
    ax.set_xlabel("colour_given_L")
    ax.set_ylabel("hue_concentration")
    fig.colorbar(sc, label="chroma_mean (capped at 30)")
    ax.set_title("Toned photographs: high on both axes, low chroma")
    fig.tight_layout()
    fig.savefig(P("measures_scatter.png"), dpi=130)
    plt.close(fig)


def run_compute():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    idx = load_index()
    print("=" * 70)
    print("04a_flag_monochrome.py  (MODE = compute)")
    print("=" * 70)
    print(f"Images: {len(idx):,}   Output: {OUTPUT_DIR}")
    M, status = compute_measures(idx)
    worst = check_against_step04(M, status)
    pd.DataFrame(M, columns=MEASURES).assign(
        image_id=idx["image_id"].to_numpy())[["image_id"] + MEASURES] \
        .to_csv(P("colour_measures.csv"), index=False, float_format="%.6g")

    cm, p95, cgl = M[:, 0], M[:, 1], M[:, 5]
    t_grey = float(GREY_INVERSE(otsu(GREY_TRANSFORM(p95))))
    cand = (p95 >= t_grey) & (cm < TONED_MAX_CHROMA) & np.isfinite(cgl)
    t_toned = otsu(cgl[cand])
    grey = p95 < t_grey
    toned = cand & (cgl >= t_toned)
    plot_compute(M, t_grey, t_toned)

    rng = np.random.default_rng(RANDOM_SEED)
    sample = make_sample(M, rng, t_grey, t_toned)
    n_lab = write_labelling_pack(idx, M, sample)

    meta = json.load(open(P("run_meta.json")))
    meta.update({"otsu_grey_chroma_p95": t_grey, "otsu_toned_cgl": t_toned,
                 "check_vs_step04_max_diff": worst,
                 "n_labelling_images": n_lab})
    json.dump(meta, open(P("run_meta.json"), "w"), indent=2)

    lines = [
        "04a_flag_monochrome.py -- compute",
        "=" * 60,
        f"Images           : {len(idx):,} (unreadable "
        f"{int((status == -1).sum())})",
        f"Check vs step 04 : chroma_mean max diff {worst:.1e}",
        "",
        "Calculated (Otsu) thresholds -- to be checked by labelling",
        f"  greyscale : chroma_p95 < {t_grey:.3f}   -> "
        f"{int(grey.sum()):,} images ({grey.mean():.2%})",
        f"  toned     : colour_given_L >= {t_toned:.3f} "
        f"(non-greyscale, chroma_mean < {TONED_MAX_CHROMA:g}) -> "
        f"{int(toned.sum()):,} images ({toned.mean():.2%})",
        "  (for reference, chroma_mean < 1: "
        f"{int((cm < 1).sum()):,} images)",
        "",
        "Look at histograms.png: a threshold is only meaningful if the",
        "histogram shows two groups with a gap or dip between them.",
        "",
        f"Labelling pack: {n_lab} images ({N_REPEATS} repeats)",
        f"  open {P('label_images.html')} in a browser, label all",
        "  images, press 'Download labels' and save the file as",
        f"  {LABELS_CSV or P('labels.csv')}",
        "  Do NOT open label_key.csv before labelling.",
        "",
        "Top categories among greyscale (Otsu):",
        idx.loc[grey, "category"].value_counts().head(10).to_string(),
        "",
        f"Elapsed: {(time.time() - t0) / 60:.1f} min",
    ]
    report = "\n".join(lines)
    open(P("compute_report.txt"), "w").write(report + "\n")
    print("\n" + report)


# ============================================================
# MODE = "finalize"
# ============================================================

def best_cutoff(values, y, direction):
    """Cut-off minimising errors. direction "below": positive if
    value < cut; "above": positive if value >= cut. Returns (cut,
    errors, table)."""
    v = np.asarray(values, float)
    y = np.asarray(y, bool)
    ok = np.isfinite(v)
    v, y = v[ok], y[ok]
    u = np.unique(v)
    cands = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2,
                            [u[-1] + 1e-6]])
    rows = []
    for c in cands:
        pred = v < c if direction == "below" else v >= c
        fn = int((y & ~pred).sum())
        fp = int((~y & pred).sum())
        rows.append((c, fn, fp, fn + fp))
    t = pd.DataFrame(rows, columns=["cutoff", "missed", "false_flag",
                                    "errors"])
    best = t[t.errors == t.errors.min()]
    cut = float(best.cutoff.median())          # middle of the best range
    return cut, int(t.errors.min()), t


def errors_at(values, y, cut, direction):
    v = np.asarray(values, float)
    y = np.asarray(y, bool)
    ok = np.isfinite(v)
    pred = v[ok] < cut if direction == "below" else v[ok] >= cut
    return int((y[ok] & ~pred).sum()), int((~y[ok] & pred).sum())


def run_finalize():
    t0 = time.time()
    idx = load_index()
    M = np.load(P("colour_measures.npy"))
    meta = json.load(open(P("run_meta.json")))
    if meta.get("n_images") != len(idx):
        raise RuntimeError("Index differs from the compute run.")
    key = pd.read_csv(P("label_key.csv"))
    lab_path = LABELS_CSV or P("labels.csv")
    lab = pd.read_csv(lab_path)
    d = key.merge(lab, on="sample_id", how="left")
    n_missing = int(d["label"].isna().sum())

    # --- Consistency on repeats ---
    rep_ids = d.loc[d.reason == "repeat", "image_id"]
    pairs = d[d.image_id.isin(rep_ids)].groupby("image_id")["label"]
    agree = [g.nunique() == 1 for _, g in pairs if g.notna().all()
             and len(g) == 2]
    consistency = float(np.mean(agree)) if agree else np.nan

    # One label per image (first showing), unsure/missing dropped.
    u = d[d.reason != "repeat"].dropna(subset=["label"])
    u = u[u.label.isin(["greyscale", "toned", "colour"])]

    # --- Greyscale threshold on chroma_p95 ---
    y_grey = (u.label == "greyscale").to_numpy()
    t_grey_lab, e_grey, tab_g = best_cutoff(u.chroma_p95, y_grey, "below")
    t_grey_otsu = meta["otsu_grey_chroma_p95"]
    fn_o, fp_o = errors_at(u.chroma_p95, y_grey, t_grey_otsu, "below")

    # --- Toned threshold on colour_given_L (among non-greyscale) ---
    ng = u[(u.label != "greyscale") & (u.chroma_mean < TONED_MAX_CHROMA)]
    y_ton = (ng.label == "toned").to_numpy()
    if y_ton.any() and (~y_ton).any():
        t_ton_lab, e_ton, tab_t = best_cutoff(ng.colour_given_L, y_ton,
                                              "above")
    else:
        t_ton_lab, e_ton, tab_t = np.nan, np.nan, pd.DataFrame()
    t_ton_otsu = meta["otsu_toned_cgl"]
    fn_t, fp_t = (errors_at(ng.colour_given_L, y_ton, t_ton_otsu, "above")
                  if len(ng) else (np.nan, np.nan))

    tab_g.assign(rule="greyscale_chroma_p95_below").to_csv(
        P("threshold_analysis.csv"), index=False)
    if len(tab_t):
        tab_t.assign(rule="toned_colour_given_L_above").to_csv(
            P("threshold_analysis.csv"), mode="a", header=False,
            index=False)

    # --- Chosen thresholds ---
    if THRESHOLD_SOURCE == "labels":
        t_grey, t_toned = t_grey_lab, t_ton_lab
    elif THRESHOLD_SOURCE == "otsu":
        t_grey, t_toned = t_grey_otsu, t_ton_otsu
    elif THRESHOLD_SOURCE == "manual":
        t_grey, t_toned = MANUAL_GREY_P95, MANUAL_TONED_CGL
    else:
        raise ValueError(THRESHOLD_SOURCE)
    if not np.isfinite(t_toned):
        raise RuntimeError("No toned threshold available; set "
                           "THRESHOLD_SOURCE or MANUAL_TONED_CGL.")

    cm, p95, cgl = M[:, 0], M[:, 1], M[:, 5]
    is_grey = p95 < t_grey
    is_toned = (~is_grey) & (cm < TONED_MAX_CHROMA) & np.isfinite(cgl) \
        & (cgl >= t_toned)
    flags = pd.DataFrame({
        "image_id": idx["image_id"].to_numpy(),
        "is_greyscale": is_grey, "is_toned": is_toned,
        "is_monochrome": is_grey | is_toned})
    for j, m in enumerate(MEASURES):
        flags[m] = M[:, j]
    flags.to_csv(P("image_flags.csv"), index=False, float_format="%.6g")

    n_u = len(u)
    lines = [
        "04a_flag_monochrome.py -- finalize",
        "=" * 60,
        f"Labels            : {lab_path}",
        f"Labelled          : {len(d) - n_missing} of {len(d)} showings "
        f"({n_missing} missing)",
        f"Used              : {n_u} images (greyscale "
        f"{int(y_grey.sum())}, toned {int((u.label == 'toned').sum())}, "
        f"colour {int((u.label == 'colour').sum())}); unsure dropped",
        f"Your consistency  : {consistency:.0%} on {len(agree)} repeated "
        "images",
        "",
        "Greyscale rule: chroma_p95 < t",
        f"  from labels : t = {t_grey_lab:.3f}, errors {e_grey} of {n_u} "
        f"({e_grey / max(n_u, 1):.1%})",
        f"  Otsu        : t = {t_grey_otsu:.3f}, missed {fn_o}, "
        f"false flags {fp_o}",
        "",
        f"Toned rule: colour_given_L >= t (non-greyscale, chroma_mean < "
        f"{TONED_MAX_CHROMA:g})",
        f"  from labels : t = {t_ton_lab:.3f}, errors {e_ton} of {len(ng)}",
        f"  Otsu        : t = {t_ton_otsu:.3f}, missed {fn_t}, "
        f"false flags {fp_t}",
        "",
        "Note: the labelled sample over-represents borderline images,",
        "so these error rates are upper bounds for the whole corpus.",
        "",
        f"Chosen ({THRESHOLD_SOURCE}): greyscale chroma_p95 < {t_grey:.3f};"
        f" toned colour_given_L >= {t_toned:.3f}",
        f"  is_greyscale  : {int(is_grey.sum()):,} ({is_grey.mean():.2%})",
        f"  is_toned      : {int(is_toned.sum()):,} ({is_toned.mean():.2%})",
        f"  is_monochrome : {int((is_grey | is_toned).sum()):,} "
        f"({(is_grey | is_toned).mean():.2%})",
        "",
        "Top categories among monochrome images:",
        idx.loc[is_grey | is_toned, "category"].value_counts().head(10)
        .to_string(),
        "",
        f"Written: {P('image_flags.csv')}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M')}  "
        f"({time.time() - t0:.0f}s)",
    ]
    report = "\n".join(lines)
    open(P("flags_report.txt"), "w").write(report + "\n")
    print(report)


# ============================================================
# Labelling page (self-contained; opened locally in a browser)
# ============================================================

LABEL_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Monochrome labelling</title>
<style>
 :root { --bg:#1b1b1b; --fg:#eee; --muted:#999; --btn:#333; --hi:#4a7; }
 body { margin:0; background:var(--bg); color:var(--fg);
        font:16px/1.4 system-ui, sans-serif; }
 #wrap { max-width:900px; margin:0 auto; padding:16px; text-align:center; }
 #q { font-size:18px; margin:8px 0 12px; }
 #img { max-width:100%; max-height:62vh; background:#000; }
 #btns { display:flex; gap:10px; justify-content:center; flex-wrap:wrap;
         margin:14px 0; }
 button { background:var(--btn); color:var(--fg); border:1px solid #555;
          border-radius:6px; padding:12px 18px; font-size:16px;
          cursor:pointer; }
 button:hover { border-color:var(--hi); }
 .k { color:var(--muted); font-size:13px; }
 #bar { color:var(--muted); margin-top:6px; }
 #done { display:none; margin-top:30px; }
</style></head><body><div id="wrap">
<div id="q">Is this photograph <b>black-and-white</b>, a
<b>single tint</b> (sepia, blue-toned...), or in <b>colour</b>?</div>
<img id="img" alt="">
<div id="btns">
 <button onclick="answer('greyscale')">Greyscale <span class="k">[1]</span></button>
 <button onclick="answer('toned')">Single tint <span class="k">[2]</span></button>
 <button onclick="answer('colour')">Colour <span class="k">[3]</span></button>
 <button onclick="answer('unsure')">Unsure <span class="k">[4]</span></button>
</div>
<div>
 <button onclick="back()">&larr; Back <span class="k">[Backspace]</span></button>
 <button onclick="download()">Download labels</button>
</div>
<div id="bar"></div>
<div id="done"><p>All __N__ images labelled. Press
<b>Download labels</b> and save the file as <code>labels.csv</code> in
the output folder.</p></div>
</div>
<script>
const items = __ITEMS__;
const KEY = "monochrome_labels_v1";
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let pos = items.findIndex(it => !(it.id in labels));
if (pos < 0) pos = items.length;
function save() { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
function show() {
  const n = Object.keys(labels).length;
  document.getElementById("bar").textContent =
    n + " of " + items.length + " labelled";
  if (pos >= items.length) {
    document.getElementById("img").style.display = "none";
    document.getElementById("done").style.display = "block";
    return;
  }
  document.getElementById("img").style.display = "";
  document.getElementById("done").style.display = "none";
  document.getElementById("img").src = items[pos].src;
}
function answer(a) {
  if (pos >= items.length) return;
  labels[items[pos].id] = a; save(); pos++; show();
}
function back() { if (pos > 0) { pos--; show(); } }
function download() {
  let csv = "sample_id,label\\n";
  for (const it of items) if (it.id in labels) csv += it.id + "," + labels[it.id] + "\\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {type: "text/csv"}));
  a.download = "labels.csv"; a.click();
}
document.addEventListener("keydown", e => {
  if (e.key === "1") answer("greyscale");
  else if (e.key === "2") answer("toned");
  else if (e.key === "3") answer("colour");
  else if (e.key === "4") answer("unsure");
  else if (e.key === "Backspace") { e.preventDefault(); back(); }
});
show();
</script></body></html>
"""


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    if MODE == "compute":
        run_compute()
    elif MODE == "finalize":
        run_finalize()
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}")
