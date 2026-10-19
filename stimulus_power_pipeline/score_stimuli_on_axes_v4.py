"""
score_stimuli_on_axes_v4.py  (stimulus pipeline, v4 axes: the final
                              stimuli scored on the CLIP axes)

Scores the 47 x 45 rendered stimuli of the haze_eq build on the v4 axes,
with the encoder and the scoring the axes were built with, and asks
whether the edits move the text-defined concepts in the intended
direction and more than the neighbouring ones.

Why this step exists
--------------------
The manipulation sweep (09c) edited 150 other corpus photographs,
calibrated the fog edit on another feature, and never combined fog and
colour. This script encodes the images participants will actually see.

CLIP is not a human observer. A pass here shows that the edits move the
TEXT-DEFINED concepts as intended; it does not show what participants
will see.

What it does
------------
  1. Encodes every stimulus exactly as 03_encode_images.py encodes corpus
     images (un2CLIP with its own projection, "openai" preprocessing,
     fp32, TF32 off). Keeps the 1024-d pooled vectors (float32) and the
     768-d embeddings, so rescoring never needs re-encoding.
  2. Reference: each scene's own k=1, h=0 stimulus (kind == "grid",
     empty contrast_level), which is byte-for-byte the unmanipulated
     512 px square. Its cosine with the stored corpus embedding of the
     same photograph is reported per scene (warning below REF_MIN_COS_WARN).
     The stimulus shows the same region step 03 encoded; only the
     resampling differs.
  3. Scores every image on the nine v4 axes, their v4@nocontrol versions,
     the six control axes and the enclosure / darkness directions, as
     06_project_axes_v4.py does: raw = embedding . axis; changes are
     divided by the axis's corpus SD (sd_raw, all encoded images).
  4. Cell effects: per scene, each cell minus the scene's reference, in
     corpus SD units; mean over scenes, a 95% interval from resampling
     scenes, the share of scenes moving in the mean's direction, the
     specificity contrasts d(A07)-d(A13), d(A07)-d(A11), d(A07)-d(A14),
     and each v4 axis's share of the total change (d^2 / sum d^2 over the
     nine axes, as 09c).
  5. Questions (a)-(d) with verdicts, stated before the results; (e)
     additivity of fog and colour, for the axes and for the two features;
     (f) the unused contrast-up cells, descriptive.
     The verdicts are recomputed with sd_raw_colour (only changes are
     reported) and for v4@nocontrol (sensitivity).
  6. Per feature unit: within each scene, the change of each key axis
     regressed (no intercept) on the measured feature changes over the
     41 manipulated colour x fog cells, oriented like the trial
     predictors (fog = -d contrast_local / 9.348, desaturation =
     -d sat_mean / 39.182). Slopes averaged across scenes.
  7. The frozen trial list: for the 180 analysed trials, the axis
     difference image A minus image B, against pred_fog and pred_sat:
     Pearson and uncentred correlations, overall and by type, and the
     joint no-intercept regression of dA07 and dA13 on both predictors.
     Intervals resample scenes.

Statistics
----------
Intervals: N_BOOT resamples of scenes (multinomial weights, as 09c),
percentile intervals. They describe scenes like these 47 hand-screened
ones, not photographs in general. They are descriptive checks and are
not corrected for multiple comparisons.

Verdicts for (a)-(d). Each level is classified on its tested quantity
(positive = the intended direction): "shown" (interval on the intended
side of 0), "not shown" (interval includes 0) or "reversed" (interval on
the other side). A question PASSES if every level is shown, is PARTIAL if
no level is reversed but at least one is not shown, and FAILS if any
level is reversed.

Checks (the run stops if a critical one fails)
----------------------------------------------
  M1  every manifest row has its file, every file has a manifest row;
      N_SCENES_EXPECTED x N_CELLS_EXPECTED images; all IMAGE_PX square.
  M2  every scene has exactly the cells of cell_plan.csv, exactly one
      reference and exactly one source_path; the plan has the expected
      colour and fog levels.
  M3  every scene matches exactly one corpus row (path, then category +
      filename; never the basename alone), and no two scenes share one.
  T1  the frozen trial list matches its entry in checksums.sha256
      (read only).
  T2  the expected number of analysed trials; file_a, file_b, left_file
      and right_file are manifest filenames; a_is_left agrees with the
      file columns; both images of a trial come from one scene.
  T3  pred_sat and pred_fog recomputed from the manifest features
      (build_trials_v4_corrected.py's definitions) equal the list's.
  A1  axis_vectors.npy matches axis_catalog.csv; vectors unit length;
      the nine v4 axes, their nocontrol versions and the report controls
      exist; every scored key has a corpus mean and SD.
  A2  the enclosure and darkness directions rebuilt from the caption
      embeddings reproduce the stored v4@nocontrol vectors (as
      09c_rescore_sweep_v4.py check R4).
  E1  encoder: every weight loaded (03 check 1).
  E2  encoder: un2CLIP projection applied (03 check 2).
  E3  encoder: agreement with the official loader on N_CHECK_IMAGES
      stimuli, if the `clip` package is installed (03 check 3).
  C1  the index is the one step 03 encoded (fingerprint; 06 check S1).
  C2  corpus reproduction: the 47 source photographs plus CORPUS_RANDOM_N
      random corpus images, opened as 03 opens them and re-encoded,
      reproduce the stored embeddings and pooled vectors.
  C3  their z-scores reproduce scores_z.npy, both from the stored
      embeddings (tests the scoring) and from the re-encoded ones
      (tests encoder and scoring together).
  F1  sat_mean and the other legacy features re-measured on REMEASURE_N
      stimulus PNGs (build_stimulus_set_v4_corrected.py's feature code)
      equal the manifest's.
  S1  every stimulus embedding is finite and unit length.
  S2  STIM_RECHECK_N stimuli re-encoded one at a time equal their stored
      rows (as 03 check 4).
  Warnings (do not stop): reference cosine below REF_MIN_COS_WARN;
  official-loader check skipped.

Inputs
------
  MANIFEST_CSV, CELL_PLAN_CSV, STIMULI_DIR   (build_stimulus_set_v4_corrected)
  TRIAL_LIST_CSV, CHECKSUMS_FILE             (freeze_trial_list_v4; read only)
  INDEX_CSV                                  (02)
  EMB_DIR: embeddings_768.npy, pooled_1024.npy, status.npy,
           image_ids.npy, run_meta.json      (03)
  AXES_DIR: axis_catalog.csv, axis_vectors.npy, captions.csv,
            caption_embeddings.npy           (05_build_axes_v4)
  SCORES_DIR: score_catalog.csv, scores_z.npy (06_project_axes_v4)
  UN2CLIP_CKPT_PATH, CLIP_MODEL_NAME         (as 03)
  the original photographs (source_path, and random corpus paths)

Outputs (under OUTPUT_DIR, never overwriting: _2, _3, ...)
---------------------------------------------------------
  report.txt                 summary (start here)
  embeddings_768.npy         (n_images, 768) float32, rows = image_scores.csv
  pooled_1024.npy            (n_images, 1024) float32
  image_scores.csv           one row per image: identity, features, raw_
                             and z_ scores on every scored key
  score_keys.csv             the scored keys: label, key, mean / SD used
  scene_effects.csv          one row per scene x cell: changes from the
                             reference (d_ = sd_raw, dcol_ = sd_raw_colour)
  cell_effects.csv           one row per cell x quantity x scale: mean,
                             interval, same-sign share, share, class
  question_levels.csv        (a)-(d) per level, on every scale and set
  additivity.csv             (e) per combined cell and quantity
  scene_slopes.csv           per scene x axis: slopes per feature SD, R^2
  trial_axis_differences.csv one row per analysed trial: A minus B
  trial_stats.csv            correlations and regressions, by type
  corpus_check.csv           per re-encoded corpus image
  reference_check.csv        per scene: reference vs stored corpus row
  heatmap_<axis>.png         one per key axis: colour level x fog level
  run_meta.json

Usage
-----
    conda activate drl2drv
    python score_stimuli_on_axes_v4.py

Dry run first (DRY_RUN = True, the default): 3 scenes, written to
OUTPUT_DIR + "_dryrun". Every check on files, the trial list, the axes
and the corpus still covers all 47 scenes; only the stimulus encoding
and the analyses are limited to the 3 scenes (the scenes with the most
analysed trials, so the trial section has something to show). With 3
scenes the intervals and verdicts mean nothing; the dry run tests the
pipeline. Then set DRY_RUN = False for the full run (a few minutes on a
GPU; most of it the encoder and reading the corpus files).

Code copied from other scripts (this script imports none of them) is
marked "From <script>, <function>" where it appears.
"""

import hashlib
import json
import os
import time
import warnings

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# OpenCV only computes the re-measured features (check F1); keep it in
# the calling thread, as the stimulus scripts do.
cv2.setNumThreads(0)


# ============================================================
# SETTINGS
# ============================================================
_G0 = set(globals())

# --- Run mode ---
# DRY_RUN           True: encode and analyse DRY_RUN_N_SCENES scenes only,
#                   writing to OUTPUT_DIR + DRY_RUN_SUFFIX. All file,
#                   trial-list, axis and corpus checks still cover every
#                   scene.
# DRY_RUN_N_SCENES  how many scenes a dry run encodes.
# DRY_RUN_SCENES    None: the scenes with the most analysed trials (ties
#                   by scene_id); or a list of scene_ids to use instead.
DRY_RUN          = False
DRY_RUN_N_SCENES = 3
DRY_RUN_SCENES   = None

# --- Inputs: stimuli (build_stimulus_set_v4_corrected.py, haze_eq) ---
MANIFEST_CSV   = "data/stimulus_set_v4_ext_haze_eq/stimulus_manifest.csv"
CELL_PLAN_CSV  = "data/stimulus_set_v4_ext_haze_eq/cell_plan.csv"
STIMULI_DIR    = "data/stimulus_set_v4_ext_haze_eq/stimuli"

# --- Inputs: the frozen trial list (read only; never written) ---
TRIAL_LIST_CSV = "data/frozen_trial_list_v4/trial_list.csv"
CHECKSUMS_FILE = "data/frozen_trial_list_v4/checksums.sha256"

# --- Inputs: corpus, axes, scores ---
INDEX_CSV      = "data/selection/image_index.csv"      # step 02
EMB_DIR        = "data/embeddings"                     # step 03
AXES_DIR       = "data/axes_v4"                        # 05_build_axes_v4.py
SCORES_DIR     = "data/scores_v4"                      # 06_project_axes_v4.py

# --- Encoder (the same values as 03_encode_images.py) ---
CLIP_MODEL_NAME   = "openai/clip-vit-large-patch14-336"
UN2CLIP_CKPT_PATH = "./pretrained_models/openai_vit_l_14_336.ckpt"
OPENAI_MODEL_NAME = "ViT-L/14@336px"   # official `clip` name (check E3)
BATCH_SIZE        = 32                 # images per forward pass (fp32)
ALLOW_TF32        = False              # as 03: full fp32, TF32 off
HASH_CHECKPOINT   = True               # record the checkpoint's SHA-256

# --- Output ---
# OUTPUT_DIR is never overwritten: if it holds files, _2, _3, ... is used.
OUTPUT_DIR     = "data/stimulus_axes_v4"
DRY_RUN_SUFFIX = "_dryrun"             # appended in a dry run

# --- The stimulus design (build_stimulus_set_v4_corrected.py) ---
N_SCENES_EXPECTED = 47
N_CELLS_EXPECTED  = 45                 # 7 colour x 6 fog + 3 contrast-up
IMAGE_PX          = 512                # every stimulus is IMAGE_PX square
COLOUR_IDENTITY   = "k=1"              # the scene's own colour
FOG_IDENTITY      = "h=0"              # the scene's own contrast
FOG_LEVELS        = ["p35", "p25", "p15", "p10", "p05"]   # lightest first
COLOUR_DOWN       = ["p25", "p10", "p05", "p01"]          # below k=1
COLOUR_UP         = ["p65", "p75"]                        # above k=1
GRID_KIND         = "grid"             # `kind` of the colour x fog cells
CONTRAST_UP_KIND  = "contrast_up"      # `kind` of the unused gain cells

# --- Trials ---
EXPECTED_ANALYSED = 180                # 60 sat_only, 60 fog_only, 30 + 30
REAL_TYPES        = ["sat_only", "fog_only", "together", "conflict"]

# --- Axes scored ---
# V4_IDS       the nine v4 axes, by id (the first word of `axis`).
# KEY_AXES     the axes with heatmaps, slopes and the detailed tables.
# REPORT_CONTROLS  control axes shown next to the key axes. Every
#              included `controls::` axis is scored either way.
V4_SET          = "v4"
NOCONTROL_SET   = "v4@nocontrol"
CONTROLS_SET    = "controls"
V4_IDS          = ["A01", "A02", "A03", "A04", "A05", "A07", "A11",
                   "A13", "A14"]
KEY_AXES        = ["A07", "A13", "A11", "A14"]
QUALITY_KEY     = "controls::quality"           # + = "a bad photo"
REPORT_CONTROLS = ["controls::quality", "controls::weather",
                   "controls::greyscale"]

# Control directions of decision 17, rebuilt exactly as
# 05_build_axes_v4.py / 09c_rescore_sweep_v4.py build them (check A2),
# and scored like an axis. Their corpus SD is estimated from SD_SAMPLE
# stored embeddings, as 09c_rescore_sweep_v4.py does.
CTRL_TEMPLATE   = "a scene that looks {}"
CONTROLS = {
    "enclosure": [("enclosed", "open"), ("cramped", "spacious"),
                  ("like a closed room", "like a wide open landscape"),
                  ("like an indoor room", "like the outdoors")],
    "darkness":  [("dark", "bright"), ("dimly lit", "brightly lit"),
                  ("like nighttime", "like daytime"), ("shadowy", "sunlit")],
}
SD_SAMPLE       = 200_000

# --- Features (the stimulus pipeline's) ---
# The SDs are those of the 20,000-image stimulus-path reference, the
# values the trial predictors use (build_trials_v4_corrected.py, SD_OF).
FEAT_SAT = "sat_mean"
FEAT_FOG = "contrast_local"
SD_SAT   = 39.182
SD_FOG   = 9.348

# --- Statistics ---
N_BOOT      = 2000          # scene resamples
CI_LEVEL    = 0.95
RANDOM_SEED = 42
# A trial-type correlation is marked uninformative when its predictor's
# SD within that type is below this (corpus-SD units): e.g. pred_fog in
# sat_only trials, which stay in one fog row by construction.
PRED_MIN_SD = 0.10

# --- Checks ---
N_CHECK_IMAGES       = 16       # E3: stimuli compared with the official loader
SAME_PIXELS_MIN_COS  = 0.9999   # E3a (as 03)
OWN_PREPROC_MIN_COS  = 0.99999  # E3b (as 03; a warning, as in 03)
CORPUS_RANDOM_N      = 32       # C2: random corpus images besides the 47 sources
CORPUS_EMB_MIN_COS   = 0.99999  # C2: re-encoded vs stored embedding (as 03 / 09c)
CORPUS_POOLED_MIN_COS = 0.9999  # C2: vs stored pooled vector (stored as float16)
Z_STORED_TOL         = 1e-4     # C3: z from stored embeddings vs scores_z.npy
Z_REENCODED_TOL      = 1e-3     # C3: z from re-encoded embeddings vs scores_z.npy
AXIS_NORM_TOL        = 1e-4     # A1 (as 06 / 09c)
NOCONTROL_MIN_COS    = 0.9999   # A2 (as 09c_rescore_sweep_v4.py R4)
REF_MIN_COS_WARN     = 0.99     # reference vs stored corpus row: warn below
STIM_RECHECK_N       = 8        # S2: stimuli re-encoded one at a time
STIM_RECHECK_MIN_COS = 0.99999  # S2 (as 03 check 4)
REMEASURE_N          = 12       # F1: stimulus PNGs re-measured
REMEASURE_RTOL       = 1e-4     # F1 (as the stimulus scripts' feature check)
PRED_TOL             = 1e-3     # T3: |pred difference| allowed, SD units

# --- Figures ---
# Diverging blue (lower) / grey / red (higher), as 05_build_axes_v4.py.
DIVERGING = ["#2a78d6", "#f0efec", "#d03b3b"]
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
FIG_DPI   = 150

_SETTING_NAMES = sorted(k for k in globals()
                        if k not in _G0 and not k.startswith("_"))


# ============================================================
# Run state, checks, stopping
# ============================================================

RUN = {"out": None, "checks": {}, "warnings": [], "t0": time.time(),
       "meta": {}}


def P(name):
    return os.path.join(RUN["out"], name)


def settings_dict():
    out = {}
    for k in _SETTING_NAMES:
        v = globals()[k]
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = repr(v)
    return out


def write_meta(status, message=None):
    if RUN["out"] is None:
        return
    meta = {"script": "score_stimuli_on_axes_v4.py",
            "status": status,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_min": round((time.time() - RUN["t0"]) / 60, 2),
            "dry_run": DRY_RUN,
            "settings": settings_dict(),
            "checks": RUN["checks"],
            "warnings": RUN["warnings"],
            **RUN["meta"]}
    if message:
        meta["stop_message"] = message
    with open(P("run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def stop(msg):
    """Stop the run, leaving run_meta.json and STOPPED.txt behind."""
    print(f"\nSTOPPED: {msg}")
    if RUN["out"] is not None:
        write_meta("stopped", msg)
        with open(P("STOPPED.txt"), "w") as f:
            f.write(msg + "\n\nChecks so far:\n")
            for k, v in RUN["checks"].items():
                f.write(f"  {k:<4} {v['status']:<7} {v['detail']}\n")
    raise SystemExit(msg)


def check(code, ok, detail, critical=True):
    status = "PASS" if ok else ("FAIL" if critical else "WARN")
    RUN["checks"][code] = {"status": status, "detail": detail}
    print(f"  [{status}] {code} {detail}")
    if not ok and critical:
        stop(f"Check {code} failed: {detail}")
    if not ok:
        RUN["warnings"].append(f"{code}: {detail}")
    return ok


def skip(code, detail):
    RUN["checks"][code] = {"status": "SKIP", "detail": detail}
    RUN["warnings"].append(f"{code} skipped: {detail}")
    print(f"  [SKIP] {code} {detail}")


def resolve_output_dir():
    """OUTPUT_DIR, never overwriting an earlier run.
    The pattern of build_stimulus_set_v4_corrected.py, resolve_output_dir():
    a folder already holding files is left alone and _2, _3, ... is used."""
    base = OUTPUT_DIR + (DRY_RUN_SUFFIX if DRY_RUN else "")

    def occupied(d):
        return os.path.isdir(d) and any(not f.startswith(".")
                                        for f in os.listdir(d))

    if not occupied(base):
        return base
    n = 2
    while occupied(f"{base}_{n}"):
        n += 1
    print(f"  NOTE: {base} already holds a run; writing to {base}_{n}.")
    return f"{base}_{n}"


# ============================================================
# Small helpers
# ============================================================

def sha256(path, chunk=1 << 24):
    """From 03_encode_images.py, sha256()."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fingerprint(idx):
    """From 03_encode_images.py, fingerprint() (also 06's S1)."""
    h = hashlib.sha1()
    for k, p in zip(idx["image_key"], idx["path"]):
        h.update(f"{k}\t{p}\n".encode("utf-8"))
    return h.hexdigest()


def unit(v):
    """From 09c_rescore_sweep_v4.py, unit()."""
    return v / np.linalg.norm(v)


def ci(a, level=CI_LEVEL, axis=0):
    """From 09c_rescore_sweep_v4.py, ci(): percentile interval."""
    lo = (1 - level) / 2 * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanpercentile(a, lo, axis=axis),
                np.nanpercentile(a, 100 - lo, axis=axis))


def boot_weights(n, rng):
    """Resampling weights, one row per resample: the multinomial
    weights of 09c_rescore_sweep_v4.py, main() ("effects")."""
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n)
                     for _ in range(N_BOOT)]).astype(float) / n


def summ(X, W):
    """Mean over the first axis (scenes), interval from the resampling
    weights W, and the share of scenes moving in the mean's direction
    (09c's same_sign)."""
    m = X.mean(0)
    lo, hi = ci(W @ X)
    ss = (np.sign(X) == np.sign(m)).mean(0)
    return m, lo, hi, ss


def classify(lo, hi, sign=1):
    """'shown' / 'not shown' / 'reversed' for an interval whose
    intended side of 0 is `sign`."""
    if sign < 0:
        lo, hi = -hi, -lo
    if lo > 0:
        return "shown"
    if hi < 0:
        return "reversed"
    return "not shown"


def verdict(classes):
    if any(c == "reversed" for c in classes):
        return "FAIL"
    if all(c == "shown" for c in classes):
        return "PASS"
    return "PARTIAL"


def fci(m, lo, hi):
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def mark(lo, hi):
    return "*" if (lo > 0 or hi < 0) else " "


def to_bool(s):
    if s.dtype == bool:
        return s
    m = {"true": True, "false": False, "1": True, "0": False,
         "yes": True, "no": False}
    out = s.astype(str).str.strip().str.lower().map(m)
    if out.isna().any():
        stop(f"Cannot read {s.name!r} as True/False: "
             f"{sorted(s[out.isna()].astype(str).unique())[:5]}")
    return out.astype(bool)


def cell_key(colour, fog, contrast):
    return f"{colour}|{fog}|{contrast}"


def cell_label(colour, fog, contrast=""):
    return f"{colour} {fog}" + (f" {contrast}" if contrast else "")


# ============================================================
# Feature code (check F1)
# From build_stimulus_set_v4_corrected.py: legacy_features() verbatim;
# measure_legacy() is its measure() reduced to the legacy fast path,
# which that function documents as bit-identical.
# ============================================================

FEATURE_SIZE = 256
LEGACY = ["lum_mean", "lum_std", "lum_p05", "lum_p50", "lum_p95",
          "sat_mean", "sat_std", "val_mean", "colorfulness",
          "contrast_local", "edge_density"]
CANNY_LOW, CANNY_HIGH = 100, 200


def legacy_features(rgb_u8):
    rgb = rgb_u8.astype(np.float32)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    gray_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    sigma = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mu = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny(gray_u8, CANNY_LOW, CANNY_HIGH)
    vals = {
        "lum_mean": gray.mean(), "lum_std": gray.std(),
        "lum_p05": np.percentile(gray, 5),
        "lum_p50": np.percentile(gray, 50),
        "lum_p95": np.percentile(gray, 95),
        "sat_mean": S.mean(), "sat_std": S.std(), "val_mean": V.mean(),
        "colorfulness": sigma + 0.3 * mu,
        "contrast_local": lap.std(),
        "edge_density": edges.mean() / 255.0,
    }
    return [float(vals[k]) for k in LEGACY], gray_u8, gray


def measure_legacy(rgb_u8):
    if rgb_u8.shape[0] != FEATURE_SIZE:
        rgb_u8 = np.asarray(Image.fromarray(rgb_u8).resize(
            (FEATURE_SIZE, FEATURE_SIZE), Image.BILINEAR), dtype=np.uint8)
    return dict(zip(LEGACY, legacy_features(rgb_u8)[0]))


# ============================================================
# Encoder
# From 03_encode_images.py: load_un2clip_state_dict(), remap_openai_to_hf(),
# load_model(), embed(), _to_rgb(), OpenAIPreprocess, official_check().
# torch is imported inside the functions (as 09c_manipulation_sweep.py,
# load_encoder(), does), so the rest of the script needs no torch.
# ============================================================

def load_un2clip_state_dict(path):
    """From 03_encode_images.py, load_un2clip_state_dict()."""
    import torch
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        raw = raw["model"]
    out = {}
    for k, v in raw.items():
        if k.startswith("visual."):
            k = k[len("visual."):]
        out[k] = v
    return out


def remap_openai_to_hf(sd):
    """From 03_encode_images.py, remap_openai_to_hf() (verbatim)."""
    d_model = sd["class_embedding"].shape[0]
    hf, unused = {}, []

    def block(rest, v):
        n, sub = rest.split(".", 1)
        p = f"encoder.layers.{n}."
        if sub in ("attn.in_proj_weight", "attn.in_proj_bias"):
            q, k, vv = v.split(d_model, dim=0)      # rows ordered q, k, v
            kind = "weight" if sub.endswith("weight") else "bias"
            return {p + f"self_attn.q_proj.{kind}": q,
                    p + f"self_attn.k_proj.{kind}": k,
                    p + f"self_attn.v_proj.{kind}": vv}
        for a, b in (("attn.out_proj.", "self_attn.out_proj."),
                     ("ln_1.", "layer_norm1."), ("ln_2.", "layer_norm2."),
                     ("mlp.c_fc.", "mlp.fc1."), ("mlp.c_proj.", "mlp.fc2.")):
            if sub.startswith(a):
                return {p + b + sub[len(a):]: v}
        return None

    for k, v in sd.items():
        if k == "proj":
            continue
        if k.startswith("transformer.resblocks."):
            r = block(k[len("transformer.resblocks."):], v)
        elif k.startswith("resblocks."):
            r = block(k[len("resblocks."):], v)
        elif k == "class_embedding":
            r = {"embeddings.class_embedding": v}
        elif k == "positional_embedding":
            r = {"embeddings.position_embedding.weight": v}
        elif k == "conv1.weight":
            r = {"embeddings.patch_embedding.weight": v}
        elif k.startswith("ln_pre."):
            r = {"pre_layrnorm." + k[len("ln_pre."):]: v}
        elif k.startswith("ln_post."):
            r = {"post_layernorm." + k[len("ln_post."):]: v}
        else:
            r = None
        if r is None:
            unused.append(k)
        else:
            hf.update(r)
    if "proj" not in sd:
        raise KeyError("un2CLIP checkpoint has no 'proj' key.")
    return hf, sd["proj"], unused


def load_model(device):
    """From 03_encode_images.py, load_model(). The failures that raise
    there are recorded as checks E1 / E2 here, and stop the run."""
    import torch
    from transformers import CLIPImageProcessor, CLIPModel
    print(f"  base model : {CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)
    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.text_model = None                 # the text tower is not needed

    print(f"  un2CLIP    : {UN2CLIP_CKPT_PATH}")
    if not os.path.exists(UN2CLIP_CKPT_PATH):
        stop(f"un2CLIP checkpoint not found: {UN2CLIP_CKPT_PATH}")
    sd = load_un2clip_state_dict(UN2CLIP_CKPT_PATH)
    hf_sd, proj, unused = remap_openai_to_hf(sd)

    # --- Check 1 (03): every weight loaded ---
    missing, unexpected = model.vision_model.load_state_dict(hf_sd,
                                                             strict=False)
    buffers = {n for n, _ in model.vision_model.named_buffers()}
    missing_w = [k for k in missing if k not in buffers]
    ok1 = not missing_w and not unexpected and not unused
    check("E1", ok1, f"weights: {len(hf_sd)} tensors loaded; missing "
          f"{len(missing_w)}, unexpected {len(unexpected)}, unused "
          f"checkpoint keys {len(unused)}"
          + ("" if ok1 else f" ({missing_w[:3]}, {list(unexpected)[:3]}, "
             f"{unused[:3]})"))

    # --- The fix of 03: use un2CLIP's projection ---
    with torch.no_grad():
        if tuple(proj.T.shape) != tuple(model.visual_projection.weight.shape):
            stop(f"proj shape {tuple(proj.shape)} does not fit "
                 f"visual_projection "
                 f"{tuple(model.visual_projection.weight.shape)}")
        std_w = model.visual_projection.weight.detach().clone()
        model.visual_projection.weight.copy_(proj.T)

    # --- Check 2 (03): projection in place ---
    w = model.visual_projection.weight.detach()
    ok2 = torch.equal(w, proj.T.to(w.dtype))
    changed = (proj.T.float() - std_w.float()).abs().max().item()
    check("E2", ok2, f"un2CLIP projection applied (differs from the "
          f"standard one by up to {changed:.2e})")

    model = model.to(device).eval()
    info = {"weights_loaded": len(hf_sd), "proj_applied": bool(ok2),
            "proj_max_diff_vs_standard": changed}
    return model, processor, sd, info


def embed(model, pixels):
    """From 03_encode_images.py, embed(): (pooled_1024,
    embedding_768_normalised). The @torch.inference_mode() decorator is
    a context manager here, so torch is imported lazily."""
    import torch
    with torch.inference_mode():
        pooled = model.vision_model(pixel_values=pixels).pooler_output
        emb = model.visual_projection(pooled)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return pooled, emb


def _to_rgb(im):
    """From 03_encode_images.py, _to_rgb()."""
    return im.convert("RGB")


class OpenAIPreprocess:
    """From 03_encode_images.py, OpenAIPreprocess: exactly
    clip._transform(n_px): Resize (bicubic, on the PIL image) ->
    CenterCrop -> RGB -> ToTensor -> Normalize."""

    def __init__(self, n_px, mean, std):
        from torchvision.transforms import (CenterCrop, Compose,
                                            InterpolationMode,
                                            Normalize, Resize, ToTensor)
        self.t = Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            _to_rgb,
            ToTensor(),
            Normalize(tuple(mean), tuple(std)),
        ])

    def __call__(self, im):
        return self.t(im)


class Un2CLIPEncoder:
    """The encoder of step 03, with its start-up checks E1-E3.

    Images are opened as 03's ImageSet.__getitem__ opens them --
    `with Image.open(path) as im: preprocess(im)` -- with no conversion
    before the resize; RGB conversion happens after the centre crop,
    inside the preprocessing, as in 03. That is the route for the corpus
    images (check C2) and for the stimuli alike.
    """

    def __init__(self):
        import torch
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = ALLOW_TF32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  device     : {self.device}")
        self.model, self.processor, self.sd, self.info = \
            load_model(self.device)
        n_px = self.processor.crop_size["height"]
        self.pre = OpenAIPreprocess(n_px, self.processor.image_mean,
                                    self.processor.image_std)
        self.d_pool = self.model.visual_projection.weight.shape[1]
        self.d_emb = self.model.visual_projection.weight.shape[0]
        self.info.update({
            "device": self.device, "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "preprocessing": {"mode": "openai", "size": n_px,
                              "mean": list(self.processor.image_mean),
                              "std": list(self.processor.image_std)},
            "dims": {"pooled": int(self.d_pool),
                     "embedding": int(self.d_emb)}})

    def encode_paths(self, paths, batch_size=None, what="images"):
        """(pooled float32, embedding float32) for a list of files."""
        torch = self.torch
        bs = batch_size or BATCH_SIZE
        n = len(paths)
        Pz = np.zeros((n, self.d_pool), np.float32)
        Ez = np.zeros((n, self.d_emb), np.float32)
        t1 = time.time()
        for a in range(0, n, bs):
            b = min(a + bs, n)
            px = []
            for p in paths[a:b]:
                try:
                    with Image.open(p) as im:
                        px.append(self.pre(im))
                except Exception as exc:
                    stop(f"Cannot read {p}: {type(exc).__name__}: {exc}")
            x = torch.stack(px).to(self.device)
            p_, e_ = embed(self.model, x)
            Pz[a:b] = p_.float().cpu().numpy()
            Ez[a:b] = e_.float().cpu().numpy()
            if n > bs:
                print(f"    {what}: {b}/{n} ({time.time() - t1:.0f}s)",
                      end="\r")
        if n > bs:
            print()
        return Pz, Ez

    def official_check(self, paths):
        """From 03_encode_images.py, official_check(): embeddings against
        the official un2CLIP loader (clip.load + visual.load_state_dict),
        on the same pixels (E3a) and each with its own preprocessing
        (E3b). Here on stimulus images rather than the first index rows."""
        torch = self.torch
        try:
            import clip
        except ImportError:
            skip("E3", "official loader: the OpenAI 'clip' package is not "
                 "installed (E1-E2 still guarantee the projection)")
            return {"official_check": "skipped"}
        ref, ref_pre = clip.load(OPENAI_MODEL_NAME, device="cpu", jit=False)
        ref.visual.load_state_dict(self.sd)          # strict, as un2CLIP does
        ref = ref.float().to(self.device).eval()
        imgs = []
        for p in paths[:N_CHECK_IMAGES]:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB").copy())

        def cos(a, b):
            a = a.float() / a.float().norm(dim=-1, keepdim=True)
            b = b.float() / b.float().norm(dim=-1, keepdim=True)
            return (a * b).sum(-1).cpu().numpy()

        with torch.inference_mode():
            px_ref = torch.stack([ref_pre(im) for im in imgs]).to(self.device)
            e_ref = ref.encode_image(px_ref)
            _, e_same = embed(self.model, px_ref)
            px_own = torch.stack([self.pre(im) for im in imgs]).to(self.device)
            _, e_own = embed(self.model, px_own)
        c_same, c_own = cos(e_ref, e_same), cos(e_ref, e_own)
        del ref
        if self.device == "cuda":
            torch.cuda.empty_cache()
        check("E3", c_same.min() >= SAME_PIXELS_MIN_COS,
              f"official loader, same pixels: min cos {c_same.min():.6f} "
              f"({len(imgs)} stimuli)")
        check("E3b", c_own.min() >= OWN_PREPROC_MIN_COS,
              f"official vs 'openai' preprocessing: min cos "
              f"{c_own.min():.6f}", critical=False)
        return {"official_check": "run",
                "same_pixels_min_cos": float(c_same.min()),
                "own_preproc_min_cos": float(c_own.min())}


# ============================================================
# 1. Manifest, cell plan and files
# ============================================================

def load_manifest():
    print("\nStimuli")
    need = ["scene_id", "category", "use", "kind", "colour_level",
            "fog_level", "contrast_level", "chroma", "haze",
            "contrast_gain", "filename", "source_path", FEAT_SAT, FEAT_FOG]
    man = pd.read_csv(MANIFEST_CSV)
    miss = [c for c in need if c not in man.columns]
    if miss:
        stop(f"{MANIFEST_CSV} has no column(s) {miss}.")
    for c in ("scene_id", "kind", "use", "colour_level", "fog_level",
              "filename", "source_path", "category"):
        man[c] = man[c].astype(str)
    man["contrast_level"] = man["contrast_level"].fillna("").astype(str)
    man["cell"] = [cell_key(*t) for t in zip(man.colour_level, man.fog_level,
                                             man.contrast_level)]

    # --- M1: files, counts, size ---
    on_disk = {f for f in os.listdir(STIMULI_DIR)
               if not f.startswith(".")
               and os.path.isfile(os.path.join(STIMULI_DIR, f))}
    names = set(man.filename)
    dup = int(man.filename.duplicated().sum())
    no_file = sorted(names - on_disk)
    no_row = sorted(on_disk - names)
    nsc = man.scene_id.nunique()
    n_exp = N_SCENES_EXPECTED * N_CELLS_EXPECTED
    bad_size = []
    modes = {}
    for f in sorted(names & on_disk):
        with Image.open(os.path.join(STIMULI_DIR, f)) as im:
            if im.size != (IMAGE_PX, IMAGE_PX):
                bad_size.append(f"{f} {im.size}")
            modes[im.mode] = modes.get(im.mode, 0) + 1
    ok = (not dup and not no_file and not no_row and nsc == N_SCENES_EXPECTED
          and len(man) == n_exp and not bad_size)
    check("M1", ok,
          f"{len(man)} manifest rows ({nsc} scenes; expected "
          f"{N_SCENES_EXPECTED} x {N_CELLS_EXPECTED} = {n_exp}); "
          f"{len(on_disk)} files in {STIMULI_DIR}; rows without a file "
          f"{len(no_file)}, files without a row {len(no_row)}, duplicate "
          f"filenames {dup}; not {IMAGE_PX}x{IMAGE_PX}: {len(bad_size)}; "
          f"modes {modes}"
          + (f"; e.g. {(no_file + no_row + bad_size)[:3]}" if not ok else ""))

    # --- M2: cell plan, reference, source path ---
    plan = pd.read_csv(CELL_PLAN_CSV)
    for c in ("kind", "colour_level", "fog_level", "use"):
        plan[c] = plan[c].astype(str)
    plan["contrast_level"] = plan["contrast_level"].fillna("").astype(str)
    plan["cell"] = [cell_key(*t) for t in zip(
        plan.colour_level, plan.fog_level, plan.contrast_level)]
    problems = []
    if plan.cell.duplicated().any() or len(plan) != N_CELLS_EXPECTED:
        problems.append(f"cell plan has {len(plan)} rows "
                        f"({int(plan.cell.duplicated().sum())} duplicates)")
    g = plan[plan.kind == GRID_KIND]
    want_c = set(COLOUR_UP) | {COLOUR_IDENTITY} | set(COLOUR_DOWN)
    want_f = {FOG_IDENTITY} | set(FOG_LEVELS)
    if set(g.colour_level) != want_c or set(g.fog_level) != want_f \
            or len(g) != len(want_c) * len(want_f):
        problems.append(f"grid cells: colour {sorted(set(g.colour_level))}, "
                        f"fog {sorted(set(g.fog_level))}, {len(g)} cells")
    ref_cell = cell_key(COLOUR_IDENTITY, FOG_IDENTITY, "")
    if ref_cell not in set(g.cell):
        problems.append("no reference cell in the plan")
    plan_cells = set(plan.cell)
    bad_sc = [s for s, d in man.groupby("scene_id")
              if set(d.cell) != plan_cells or d.cell.duplicated().any()]
    if bad_sc:
        problems.append(f"{len(bad_sc)} scene(s) differ from the cell plan, "
                        f"e.g. {bad_sc[:3]}")
    is_ref = ((man.kind == GRID_KIND) & (man.colour_level == COLOUR_IDENTITY)
              & (man.fog_level == FOG_IDENTITY) & (man.contrast_level == ""))
    nref = man[is_ref].groupby("scene_id").size().reindex(
        man.scene_id.unique(), fill_value=0)
    if (nref != 1).any():
        problems.append(f"scenes without exactly one reference: "
                        f"{list(nref[nref != 1].index[:3])}")
    nsrc = man.groupby("scene_id").source_path.nunique()
    if (nsrc != 1).any():
        problems.append(f"scenes with more than one source_path: "
                        f"{list(nsrc[nsrc != 1].index[:3])}")
    check("M2", not problems,
          "every scene has the 45 planned cells, one reference "
          f"({COLOUR_IDENTITY}, {FOG_IDENTITY}, kind {GRID_KIND!r}, no "
          "contrast_level) and one source_path" if not problems
          else "; ".join(problems))
    man["is_ref"] = is_ref
    order = {c: i for i, c in enumerate(plan.cell)}
    man["_o"] = man.cell.map(order)
    man = man.sort_values(["scene_id", "_o"], kind="stable") \
        .drop(columns="_o").reset_index(drop=True)
    RUN["meta"]["manifest_sha256"] = sha256(MANIFEST_CSV)
    RUN["meta"]["cell_plan_sha256"] = sha256(CELL_PLAN_CSV)
    return man, plan


# ============================================================
# 2. Scenes -> corpus rows
# ============================================================

def match_scenes(man):
    print("\nScene -> corpus match")
    idx = pd.read_csv(INDEX_CSV, usecols=lambda c: c in
                      ("image_id", "image_key", "category", "path"))
    for c in ("image_id", "image_key", "category", "path"):
        if c not in idx.columns:
            stop(f"{INDEX_CSV} has no {c!r} column.")
    if not (idx["image_id"].to_numpy() == np.arange(len(idx))).all():
        stop("image_id must be 0..N-1 in row order (as written by step 02).")
    sc = man.groupby("scene_id", sort=True).agg(
        source_path=("source_path", "first"),
        category=("category", "first")).reset_index()
    by_path, by_cf = {}, {}
    for i, p, c in zip(idx.image_id.to_numpy(), idx.path.astype(str),
                       idx.category.astype(str)):
        by_path.setdefault(os.path.normpath(p), []).append(int(i))
    rows, methods, bad = [], [], []
    need_cf = []
    for r in sc.itertuples():
        hits = by_path.get(os.path.normpath(r.source_path), [])
        if hits:
            rows.append(hits)
            methods.append("path")
        else:
            rows.append(None)
            methods.append(None)
            need_cf.append(r.Index)
    if need_cf:                         # fallback: category + filename
        for i, p, c in zip(idx.image_id.to_numpy(), idx.path.astype(str),
                           idx.category.astype(str)):
            by_cf.setdefault((c, os.path.basename(p)), []).append(int(i))
        for k in need_cf:
            r = sc.iloc[k]
            rows[k] = by_cf.get((r.category,
                                 os.path.basename(r.source_path)), [])
            methods[k] = "category+filename"
    for r, h, m in zip(sc.itertuples(), rows, methods):
        if len(h) != 1:
            bad.append(f"{r.scene_id} ({m}: {len(h)} rows)")
    image_ids = [h[0] if len(h) == 1 else -1 for h in rows]
    shared = pd.Series(image_ids)[lambda s: s >= 0].duplicated().sum()
    sc["image_id"] = image_ids
    sc["match"] = methods
    nm = pd.Series(methods).value_counts().to_dict()
    check("M3", not bad and not shared,
          f"{len(sc)} scenes, each matching exactly one corpus row "
          f"({nm})" if not bad and not shared else
          f"{len(bad)} scene(s) without exactly one match, e.g. {bad[:3]}; "
          f"{int(shared)} corpus row(s) shared by two scenes")
    sc["corpus_path"] = idx.path.to_numpy()[sc.image_id.to_numpy()]
    return sc, idx


# ============================================================
# 3. The frozen trial list
# ============================================================

def load_trials(man):
    print("\nFrozen trial list (read only)")
    if not os.path.isfile(CHECKSUMS_FILE):
        stop(f"{CHECKSUMS_FILE} not found: the frozen list cannot be "
             "verified.")
    entries = {}
    with open(CHECKSUMS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            h, p = parts
            p = p.strip().lstrip("*")
            entries[os.path.normpath(p)] = h.lower()
    want = os.path.basename(TRIAL_LIST_CSV)
    if want not in entries:
        stop(f"{CHECKSUMS_FILE} has no entry for {want!r} at the top "
             f"level (entries: {sorted(entries)[:5]} ...).")
    got = sha256(TRIAL_LIST_CSV)
    check("T1", got == entries[want],
          f"{TRIAL_LIST_CSV} sha256 {got[:16]}... "
          f"{'matches' if got == entries[want] else 'DOES NOT match'} "
          f"{CHECKSUMS_FILE}")
    RUN["meta"]["trial_list_sha256"] = got

    t = pd.read_csv(TRIAL_LIST_CSV)
    need = ["file_a", "file_b", "left_file", "right_file", "a_is_left",
            "pred_sat", "pred_fog", "trial_type", "analysed"]
    miss = [c for c in need if c not in t.columns]
    if miss:
        stop(f"{TRIAL_LIST_CSV} has no column(s) {miss}.")
    t["analysed"] = to_bool(t["analysed"])
    an = t[t.analysed].copy()
    an["a_is_left"] = to_bool(an["a_is_left"])
    if "trial" not in an.columns:
        an["trial"] = an.index
    for c in ("file_a", "file_b", "left_file", "right_file", "trial_type"):
        an[c] = an[c].astype(str)
    names = set(man.filename)
    notin = sorted({f for c in ("file_a", "file_b", "left_file",
                                "right_file") for f in an[c]} - names)
    side = np.where(an.a_is_left, (an.left_file == an.file_a)
                    & (an.right_file == an.file_b),
                    (an.left_file == an.file_b) & (an.right_file == an.file_a))
    scene_of = dict(zip(man.filename, man.scene_id))
    sa = an.file_a.map(scene_of)
    sb = an.file_b.map(scene_of)
    cross = int((sa != sb).sum()) if not notin else -1
    types = an.trial_type.value_counts().to_dict()
    ok = (len(an) == EXPECTED_ANALYSED and not notin and side.all()
          and cross == 0)
    check("T2", ok,
          f"{len(an)} analysed trials (expected {EXPECTED_ANALYSED}; "
          f"{types}); filenames not in the manifest {len(notin)}"
          + (f" e.g. {notin[:3]}" if notin else "")
          + f"; a_is_left inconsistent {int((~side).sum())}; "
          f"trials across two scenes {cross}")
    an["scene_id"] = sa.to_numpy()

    # --- T3: the predictors, recomputed from the manifest ---
    # From build_trials_v4_corrected.py, package(): pred_sat = -d_sat / SD,
    # pred_fog = FOG_SIGN * (fog_a - fog_b) / SD with FOG_SIGN = -1.
    f_sat = dict(zip(man.filename, man[FEAT_SAT].astype(float)))
    f_fog = dict(zip(man.filename, man[FEAT_FOG].astype(float)))
    ps = -(an.file_a.map(f_sat) - an.file_b.map(f_sat)) / SD_SAT
    pf = -(an.file_a.map(f_fog) - an.file_b.map(f_fog)) / SD_FOG
    dps = float(np.abs(ps - an.pred_sat).max())
    dpf = float(np.abs(pf - an.pred_fog).max())
    check("T3", dps <= PRED_TOL and dpf <= PRED_TOL,
          f"pred_sat and pred_fog recomputed from the manifest features "
          f"(image A minus image B): max |diff| {dps:.2e} and {dpf:.2e} "
          "(positive = A less colourful / foggier)")
    an["pred_sat_check"] = ps.to_numpy()
    an["pred_fog_check"] = pf.to_numpy()
    return an.reset_index(drop=True)


# ============================================================
# 4. Axes
# ============================================================

def load_axes():
    print("\nAxes")
    acat = pd.read_csv(os.path.join(AXES_DIR, "axis_catalog.csv"))
    AV = np.load(os.path.join(AXES_DIR, "axis_vectors.npy")).astype(np.float64)
    # From 09c_rescore_sweep_v4.py, main() ("axes"): rows >= 0 have vectors.
    acat = acat[acat["row"] >= 0].sort_values("row")
    rows_ok = (acat["row"].to_numpy() == np.arange(len(AV))).all()
    nerr = float(np.abs(np.linalg.norm(AV, axis=1) - 1).max())
    kv = {k: AV[i] for i, k in enumerate(acat["key"])}
    inc = dict(zip(acat["key"], acat["included"].astype(str).str.lower()
                   .isin(["true", "1"])))
    v4 = acat[(acat["set"] == V4_SET)]
    v4 = v4[v4["key"].map(inc)]
    ids = {a.split()[0]: k for a, k in zip(v4["axis"], v4["key"])}
    problems = []
    if not rows_ok:
        problems.append("axis_catalog rows do not match axis_vectors.npy")
    if nerr > AXIS_NORM_TOL:
        problems.append(f"axis vectors not unit length ({nerr:.1e})")
    if sorted(ids) != sorted(V4_IDS):
        problems.append(f"v4 axes found {sorted(ids)}, expected {V4_IDS}")
    v4keys = [ids[i] for i in V4_IDS if i in ids]
    nckeys = [f"{NOCONTROL_SET}::{k.split('::', 1)[1]}" for k in v4keys]
    miss_nc = [k for k in nckeys if k not in kv]
    if miss_nc:
        problems.append(f"no stored nocontrol vector for {miss_nc[:3]}")
    ctrl = acat[(acat["set"] == CONTROLS_SET)]
    ctrlkeys = [k for k in ctrl["key"] if inc[k]]
    miss_c = [k for k in REPORT_CONTROLS + [QUALITY_KEY] if k not in ctrlkeys]
    if miss_c:
        problems.append(f"control axes {miss_c} not found; available "
                        f"{ctrlkeys}")
    sc = pd.read_csv(os.path.join(SCORES_DIR, "score_catalog.csv"))
    scd = sc.set_index("key")
    allkeys = v4keys + nckeys + ctrlkeys
    miss_sd = [k for k in allkeys if k not in scd.index
               or not np.isfinite(scd.at[k, "sd_raw"])]
    if miss_sd:
        problems.append(f"no corpus SD in score_catalog.csv for "
                        f"{miss_sd[:3]}")
    check("A1", not problems,
          f"{len(AV)} vectors, unit length (max error {nerr:.1e}); "
          f"{len(v4keys)} v4 axes, {len(nckeys)} nocontrol, "
          f"{len(ctrlkeys)} controls; corpus mean and SD for each"
          if not problems else "; ".join(problems))

    # --- A2: control directions, rebuilt as 05_v4 / 09c_rescore_v4 (R4) ---
    caps = pd.read_csv(os.path.join(AXES_DIR, "captions.csv"))
    C = np.load(os.path.join(AXES_DIR, "caption_embeddings.npy")).astype(
        np.float64)
    cidx = dict(zip(caps["caption"], caps["row"]))
    need = [CTRL_TEMPLATE.format(x) for prs in CONTROLS.values()
            for p in prs for x in p]
    m = [c for c in need if c not in cidx]
    if m:
        stop(f"Control captions not in captions.csv: {m}")
    ctrl_vec = {}
    for cn, prs in CONTROLS.items():
        ctrl_vec[cn] = unit(np.mean([C[cidx[CTRL_TEMPLATE.format(a)]]
                                     - C[cidx[CTRL_TEMPLATE.format(b)]]
                                     for a, b in prs], axis=0))
    Q = np.linalg.qr(np.stack(list(ctrl_vec.values()), axis=1))[0]
    cos4 = [float(unit(kv[k] - Q @ (Q.T @ kv[k])) @ kv[n])
            for k, n in zip(v4keys, nckeys)]
    check("A2", min(cos4) >= NOCONTROL_MIN_COS,
          f"enclosure and darkness directions rebuilt from the caption "
          f"embeddings; nocontrol vectors reproduced, min cos "
          f"{min(cos4):.6f}")

    # --- the keys scored, in one table ---
    rows = []
    for k in v4keys:
        rows.append(("v4", k.split("::", 1)[1].split()[0], k))
    for k in nckeys:
        rows.append(("v4@nocontrol",
                     "nc_" + k.split("::", 1)[1].split()[0], k))
    for k in ctrlkeys:
        rows.append(("controls", "ctrl_" + k.split("::", 1)[1].split()[0]
                     .replace("/", "_"), k))
    keys = pd.DataFrame(rows, columns=["set", "label", "key"])
    for c in ("col", "mean_raw", "sd_raw", "mean_raw_colour",
              "sd_raw_colour"):
        keys[c] = [scd.at[k, c] if c in scd.columns else np.nan
                   for k in keys.key]
    keys["sd_source"] = "score_catalog.csv"
    dirs = pd.DataFrame([("direction", f"dir_{cn}", f"direction::{cn}")
                         for cn in ctrl_vec], columns=["set", "label", "key"])
    for cn in ctrl_vec:
        kv[f"direction::{cn}"] = ctrl_vec[cn]
    keys = pd.concat([keys, dirs], ignore_index=True)
    keys["V"] = [kv[k] for k in keys.key]
    if keys.label.duplicated().any():
        stop(f"Duplicate key labels: {list(keys.label[keys.label.duplicated()])}")
    RUN["meta"]["axis_vectors_sha256"] = sha256(
        os.path.join(AXES_DIR, "axis_vectors.npy"))
    return keys


def direction_sd(keys, Emb, status, rng):
    """Corpus mean and SD of the control directions from SD_SAMPLE
    stored embeddings: 09c_rescore_sweep_v4.py, main() ("corpus SD of the
    control directions"). Also the sampled / stored SD ratio of A07 (R5
    there), as a sanity figure."""
    pool_ = np.where(status == 1)[0]
    smp = np.sort(rng.choice(pool_, min(SD_SAMPLE, len(pool_)),
                             replace=False))
    Xs = np.asarray(Emb[smp], np.float64)
    Xs = Xs[np.isfinite(Xs).all(1) & (np.abs(Xs).sum(1) > 0)]
    for i in np.where(keys.set == "direction")[0]:
        s = Xs @ keys.V.iat[i]
        keys.at[i, "mean_raw"] = float(s.mean())
        keys.at[i, "sd_raw"] = float(s.std())
        keys.at[i, "sd_source"] = f"sample of {len(Xs):,} embeddings"
    j = int(np.where(keys.label == "A07")[0][0])
    r5 = float((Xs @ keys.V.iat[j]).std() / keys.sd_raw.iat[j])
    return len(Xs), r5


# ============================================================
# 5. Corpus reproduction (checks C1-C3)
# ============================================================

def corpus_checks(enc, idx, scenes, keys, rng):
    print("\nCorpus reproduction")
    with open(os.path.join(EMB_DIR, "run_meta.json")) as f:
        emb_meta = json.load(f)
    fp = fingerprint(idx)
    Emb = np.load(os.path.join(EMB_DIR, "embeddings_768.npy"), mmap_mode="r")
    Pool = np.load(os.path.join(EMB_DIR, "pooled_1024.npy"), mmap_mode="r")
    status = np.load(os.path.join(EMB_DIR, "status.npy"))
    ids = np.load(os.path.join(EMB_DIR, "image_ids.npy"))
    n = len(idx)
    ok1 = (emb_meta.get("index_fingerprint") == fp and Emb.shape[0] == n
           and Pool.shape[0] == n and len(status) == n
           and (ids == np.arange(n)).all())
    check("C1", ok1, f"image_index.csv is the index step 03 encoded "
          f"(fingerprint {fp[:12]}...; {n:,} rows aligned with the "
          "embeddings)")
    src = scenes.image_id.to_numpy()
    if (status[src] != 1).any():
        stop(f"Source photographs not encoded in step 03: "
             f"{list(scenes.scene_id[status[src] != 1][:3])}")
    pool_ = np.setdiff1d(np.where(status == 1)[0], src)
    rnd = rng.choice(pool_, min(CORPUS_RANDOM_N, len(pool_)), replace=False)
    rows = np.concatenate([src, rnd]).astype(np.int64)
    role = ["scene source"] * len(src) + ["random"] * len(rnd)
    paths = idx.path.to_numpy()[rows].tolist()
    Pn, En = enc.encode_paths(paths, what="corpus images")
    o = np.argsort(rows)                      # memmap reads in row order
    inv = np.empty_like(o)
    inv[o] = np.arange(len(o))
    E_st = np.asarray(Emb[rows[o]], np.float64)[inv]
    P_st = np.asarray(Pool[rows[o]], np.float64)[inv]
    cos_e = (En * E_st).sum(1) / np.linalg.norm(E_st, axis=1)
    cos_p = (Pn * P_st).sum(1) / (np.linalg.norm(Pn, axis=1)
                                  * np.linalg.norm(P_st, axis=1))
    check("C2", cos_e.min() >= CORPUS_EMB_MIN_COS
          and cos_p.min() >= CORPUS_POOLED_MIN_COS,
          f"{len(rows)} corpus images re-encoded ({len(src)} scene sources "
          f"+ {len(rnd)} random), opened as 03 opens them: min cos with "
          f"stored embeddings {cos_e.min():.7f}, with stored pooled vectors "
          f"{cos_p.min():.6f}")

    sc_keys = keys[keys.col.notna() & (keys.set != "direction")]
    cols = sc_keys.col.astype(int).to_numpy()
    A = np.stack(sc_keys.V.to_list())
    mu = sc_keys.mean_raw.to_numpy(float)
    sd = sc_keys.sd_raw.to_numpy(float)
    Z = np.load(os.path.join(SCORES_DIR, "scores_z.npy"), mmap_mode="r")
    if Z.shape[0] != n:
        stop("scores_z.npy rows do not match the image index.")
    Zst = np.asarray(Z[rows[o]], np.float64)[inv][:, cols]
    z_st = (E_st @ A.T - mu) / sd
    z_re = (En.astype(np.float64) @ A.T - mu) / sd
    d_st = np.abs(z_st - Zst).max(1)
    d_re = np.abs(z_re - Zst).max(1)
    check("C3", d_st.max() <= Z_STORED_TOL and d_re.max() <= Z_REENCODED_TOL,
          f"z-scores on {len(cols)} axes vs scores_z.npy: max |diff| "
          f"{d_st.max():.1e} from the stored embeddings (tol "
          f"{Z_STORED_TOL:g}), {d_re.max():.1e} from the re-encoded ones "
          f"(tol {Z_REENCODED_TOL:g})")
    pd.DataFrame({"image_id": rows, "role": role, "path": paths,
                  "cos_embedding": cos_e, "cos_pooled": cos_p,
                  "max_abs_dz_stored_emb": d_st,
                  "max_abs_dz_reencoded": d_re}).to_csv(
        P("corpus_check.csv"), index=False, float_format="%.8g")
    stored_src = E_st[:len(src)]
    return Emb, status, stored_src, {
        "n": int(len(rows)), "min_cos_emb": float(cos_e.min()),
        "min_cos_pooled": float(cos_p.min()),
        "max_dz_stored": float(d_st.max()),
        "max_dz_reencoded": float(d_re.max())}


# ============================================================
# 6. Re-measured features (check F1)
# ============================================================

def remeasure_check(man_sel, rng):
    print("\nFeature re-measurement")
    feats = [f for f in LEGACY if f in man_sel.columns]
    pick = rng.choice(len(man_sel), min(REMEASURE_N, len(man_sel)),
                      replace=False)
    worst, where = 0.0, ""
    for i in pick:
        r = man_sel.iloc[int(i)]
        with Image.open(os.path.join(STIMULI_DIR, r.filename)) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        m = measure_legacy(rgb)
        for f in feats:
            a, b = m[f], float(r[f])
            if np.isnan(a) and np.isnan(b):
                continue
            rel = abs(a - b) / max(abs(b), 1e-6)
            if not np.isfinite(rel) or rel > worst:
                worst, where = (rel if np.isfinite(rel) else np.inf), \
                    f"{r.filename} {f}"
    check("F1", worst <= REMEASURE_RTOL,
          f"{len(pick)} stimulus PNGs re-measured ({len(feats)} features, "
          f"incl. {FEAT_SAT} and {FEAT_FOG}): max relative difference from "
          f"the manifest {worst:.1e}" + (f" ({where})" if worst > 0 else ""))


# ============================================================
# 7. Effects
# ============================================================

def scene_arrays(man_sel, plan, raw, keys):
    """R[scene, cell, key] raw scores and F[scene, cell, 2] features
    (sat_mean, contrast_local), cells in cell-plan order."""
    sids = sorted(man_sel.scene_id.unique())
    cells = list(plan.cell)
    si = {s: i for i, s in enumerate(sids)}
    ci_ = {c: i for i, c in enumerate(cells)}
    R = np.full((len(sids), len(cells), raw.shape[1]), np.nan)
    F = np.full((len(sids), len(cells), 2), np.nan)
    for r, (s, c, fs, ff) in enumerate(zip(man_sel.scene_id, man_sel.cell,
                                           man_sel[FEAT_SAT], man_sel[FEAT_FOG])):
        R[si[s], ci_[c]] = raw[r]
        F[si[s], ci_[c]] = (fs, ff)
    if np.isnan(R).any():
        stop("Some scene x cell has no score (internal error).")
    return sids, cells, R, F


def cell_effects(plan, sids, cells, R, F, keys, W):
    ref = cells.index(cell_key(COLOUR_IDENTITY, FOG_IDENTITY, ""))
    sd = keys.sd_raw.to_numpy(float)
    sdc = keys.sd_raw_colour.to_numpy(float)
    D = (R - R[:, ref:ref + 1, :]) / sd                    # sd_raw
    with np.errstate(invalid="ignore", divide="ignore"):
        Dc = (R - R[:, ref:ref + 1, :]) / sdc              # sd_raw_colour
    dF = F - F[:, ref:ref + 1, :]
    dsat = dF[:, :, 0] / SD_SAT
    dcl = dF[:, :, 1] / SD_FOG
    lab = list(keys.label)
    v4i = [lab.index(i) for i in V4_IDS]
    nci = [lab.index("nc_" + i) for i in V4_IDS]
    J = {i: lab.index(i) for i in V4_IDS}
    JN = {i: lab.index("nc_" + i) for i in V4_IDS}

    rows = []
    pl = plan.set_index("cell")
    for c, ck in enumerate(cells):
        if c == ref:
            continue
        p = pl.loc[ck]
        base = {"cell": cell_label(p.colour_level, p.fog_level,
                                   p.contrast_level),
                "kind": p.kind, "use": p.use, "colour_level": p.colour_level,
                "fog_level": p.fog_level,
                "contrast_level": p.contrast_level, "n_scenes": len(sids)}
        for scale, X in (("sd_raw", D[:, c, :]), ("sd_raw_colour", Dc[:, c, :])):
            m, lo, hi, ss = summ(X, W)
            sh = np.full(len(lab), np.nan)
            dm = m[v4i]
            sh[v4i] = dm ** 2 / max((dm ** 2).sum(), 1e-12)
            for j, k in enumerate(lab):
                if scale == "sd_raw_colour" and keys.set.iat[j] != "v4":
                    continue
                rows.append({**base, "scale": scale, "set": keys.set.iat[j],
                             "quantity": k, "key": keys.key.iat[j],
                             "d": m[j], "lo": lo[j], "hi": hi[j],
                             "same_sign": ss[j], "share": sh[j],
                             "interval": "excludes 0" if mark(lo[j], hi[j]) == "*"
                             else "includes 0"})
            sets = [("v4", J)] + ([("v4@nocontrol", JN)]
                                  if scale == "sd_raw" else [])
            for set_, JJ in sets:
                for other in ("A13", "A11", "A14"):
                    x = X[:, JJ["A07"]] - X[:, JJ[other]]
                    m1, l1, h1, s1 = summ(x[:, None], W)
                    rows.append({**base, "scale": scale, "set": set_,
                                 "quantity": f"A07-{other}", "key": "",
                                 "d": m1[0], "lo": l1[0], "hi": h1[0],
                                 "same_sign": s1[0], "share": np.nan,
                                 "interval": "excludes 0" if mark(l1[0], h1[0]) == "*"
                                 else "includes 0"})
        for q, x in (("d_sat_mean_sd", dsat[:, c]), ("d_contrast_local_sd",
                                                     dcl[:, c])):
            m1, l1, h1, s1 = summ(x[:, None], W)
            rows.append({**base, "scale": "feature_sd", "set": "feature",
                         "quantity": q, "key": "", "d": m1[0], "lo": l1[0],
                         "hi": h1[0], "same_sign": s1[0], "share": np.nan,
                         "interval": "excludes 0" if mark(l1[0], h1[0]) == "*"
                         else "includes 0"})
    return D, Dc, dsat, dcl, pd.DataFrame(rows), v4i, nci


QUESTIONS = {
    "a": "every fog level (at the original colour) raises fog (A07) more "
         "than vividless (A13)",
    "b": "every colour level below the original raises vividless (A13) "
         "more than fog (A07)",
    "c": "the colour levels above the original lower vividless (A13)",
    "d": "every fog level (at the original colour) moves fog (A07) more "
         "than flat (A11) and more than lifeless (A14)",
}


def question_levels(X, lab_index, cells, W):
    """Per-level tests of (a)-(d) on one change array X[scene, cell, key]
    (one set, one scale); lab_index maps 'A07' etc. to columns of X."""
    J = lab_index
    out = []

    def add(q, level, cellname, tested, x, sign):
        m, lo, hi, ss = summ(x[:, None], W)
        out.append({"question": q, "level": level, "cell": cellname,
                    "tested": tested, "intended": "> 0" if sign > 0 else "< 0",
                    "mean": m[0], "lo": lo[0], "hi": hi[0],
                    "same_sign": ss[0],
                    "class": classify(lo[0], hi[0], sign)})

    for f in FOG_LEVELS:
        c = cells.index(cell_key(COLOUR_IDENTITY, f, ""))
        add("a", f, f"{COLOUR_IDENTITY} {f}", "d(A07) - d(A13)",
            X[:, c, J["A07"]] - X[:, c, J["A13"]], 1)
    for k in COLOUR_DOWN:
        c = cells.index(cell_key(k, FOG_IDENTITY, ""))
        add("b", k, f"{k} {FOG_IDENTITY}", "d(A13) - d(A07)",
            X[:, c, J["A13"]] - X[:, c, J["A07"]], 1)
    for k in COLOUR_UP:
        c = cells.index(cell_key(k, FOG_IDENTITY, ""))
        add("c", k, f"{k} {FOG_IDENTITY}", "d(A13)", X[:, c, J["A13"]], -1)
    for f in FOG_LEVELS:
        c = cells.index(cell_key(COLOUR_IDENTITY, f, ""))
        for o in ("A11", "A14"):
            add("d", f"{f} vs {o}", f"{COLOUR_IDENTITY} {f}",
                f"d(A07) - d({o})", X[:, c, J["A07"]] - X[:, c, J[o]], 1)
    return pd.DataFrame(out)


def additivity(cells, D, dsat, dcl, keys, W):
    """(e): per scene, d(k,h) - d(k,h=0) - d(k=1,h), for every combined
    cell (colour != k=1, fog != h=0)."""
    lab = list(keys.label)
    rows = []
    colours = COLOUR_UP + COLOUR_DOWN
    quant = [(l, D[:, :, j]) for j, l in enumerate(lab)
             if keys.set.iat[j] in ("v4", "controls")]
    quant += [("d_sat_mean_sd", dsat), ("d_contrast_local_sd", dcl)]
    for k in colours:
        cc = cells.index(cell_key(k, FOG_IDENTITY, ""))
        for f in FOG_LEVELS:
            c = cells.index(cell_key(k, f, ""))
            cf = cells.index(cell_key(COLOUR_IDENTITY, f, ""))
            for q, X in quant:
                obs, pred = X[:, c], X[:, cc] + X[:, cf]
                inter = obs - pred
                M = np.stack([obs, pred, inter], 1)
                m, lo, hi, ss = summ(M, W)
                rows.append({"cell": f"{k} {f}", "colour_level": k,
                             "fog_level": f, "quantity": q,
                             "observed": m[0], "sum_of_singles": m[1],
                             "interaction": m[2], "lo": lo[2], "hi": hi[2],
                             "same_sign": ss[2],
                             "interval": "excludes 0" if mark(lo[2], hi[2]) == "*"
                             else "includes 0"})
    return pd.DataFrame(rows)


def feature_slopes(plan, sids, cells, D, dsat, dcl, keys, W):
    """Per scene: y = b_fog * fog + b_desat * desat, no intercept, over
    the manipulated colour x fog cells; fog = -d contrast_local / SD_FOG,
    desat = -d sat_mean / SD_SAT (oriented like pred_fog / pred_sat)."""
    ref = cells.index(cell_key(COLOUR_IDENTITY, FOG_IDENTITY, ""))
    pl = plan.set_index("cell")
    use = [i for i, c in enumerate(cells)
           if pl.loc[c].kind == GRID_KIND and i != ref]
    lab = list(keys.label)
    targets = [l for l in KEY_AXES] + [f"nc_{l}" for l in KEY_AXES] + \
        [lab[j] for j in range(len(lab)) if keys.key.iat[j] == QUALITY_KEY]
    rows, B = [], {}
    for s, sid in enumerate(sids):
        X = np.stack([-dcl[s, use], -dsat[s, use]], 1)
        rx = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
        for t in targets:
            y = D[s, use, lab.index(t)]
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ b
            r2 = 1 - float(res @ res) / max(float(y @ y), 1e-12)
            B.setdefault(t, []).append(b)
            rows.append({"scene_id": sid, "axis": t,
                         "slope_per_sd_fog": b[0],
                         "slope_per_sd_desaturation": b[1],
                         "r2_uncentred": r2, "r_fog_desat": rx,
                         "n_cells": len(use)})
    per = pd.DataFrame(rows)
    summary = []
    for t in targets:
        Bt = np.array(B[t])
        m, lo, hi, ss = summ(Bt, W)
        r2 = per[per.axis == t].r2_uncentred
        summary.append({"axis": t, "fog": (m[0], lo[0], hi[0], ss[0]),
                        "desat": (m[1], lo[1], hi[1], ss[1]),
                        "r2_median": float(r2.median()),
                        "r2_min": float(r2.min())})
    contr = {}
    for name, a, b, col in (("fog slope: A07 - A13", "A07", "A13", 0),
                            ("desaturation slope: A13 - A07", "A13", "A07", 1),
                            ("fog slope: A07 - A11", "A07", "A11", 0),
                            ("fog slope: A07 - A14", "A07", "A14", 0)):
        x = np.array(B[a])[:, col] - np.array(B[b])[:, col]
        m, lo, hi, ss = summ(x[:, None], W)
        contr[name] = (m[0], lo[0], hi[0], ss[0])
    return per, summary, contr, len(use)


# ============================================================
# 8. Trials
# ============================================================

def wstats(Wt, x, y, centred):
    """Weighted Pearson (centred) or uncentred correlation, one value per
    row of Wt (trial weights)."""
    S = Wt.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        if centred:
            mx, my = Wt @ x / S, Wt @ y / S
            sxy = Wt @ (x * y) / S - mx * my
            sxx = Wt @ (x * x) / S - mx * mx
            syy = Wt @ (y * y) / S - my * my
        else:
            sxy, sxx, syy = Wt @ (x * y), Wt @ (x * x), Wt @ (y * y)
        return sxy / np.sqrt(sxx * syy)


def wreg2(Wt, x1, x2, y):
    """Weighted no-intercept regression y = b1 x1 + b2 x2, per row of Wt.
    Returns (n_rows, 2)."""
    a, b_, c = Wt @ (x1 * x1), Wt @ (x1 * x2), Wt @ (x2 * x2)
    u, v = Wt @ (x1 * y), Wt @ (x2 * y)
    det = a * c - b_ * b_
    with np.errstate(invalid="ignore", divide="ignore"):
        b1 = (c * u - b_ * v) / det
        b2 = (a * v - b_ * u) / det
    bad = np.abs(det) < 1e-12 * np.maximum(a * c, 1e-300)
    b1[bad] = np.nan
    b2[bad] = np.nan
    return np.stack([b1, b2], 1)


def trial_analysis(tr, rng):
    """Correlations and joint regressions on the analysed trials,
    overall and by type; intervals resample scenes."""
    out_rows = []
    res = {}

    def one(name, sub):
        n = len(sub)
        if n < 3:
            return None
        sc = sorted(sub.scene_id.unique())
        si = {s: i for i, s in enumerate(sc)}
        Wsc = boot_weights(len(sc), rng)
        Wt = Wsc[:, [si[s] for s in sub.scene_id]] * len(sc)
        one_ = np.ones((1, n))
        pf = sub.pred_fog.to_numpy(float)
        ps = sub.pred_sat.to_numpy(float)
        sdf, sds = float(pf.std()), float(ps.std())
        info = {"type": name, "n_trials": n, "n_scenes": len(sc),
                "sd_pred_fog": sdf, "sd_pred_sat": sds}
        r = {"info": info, "corr": [], "reg": None, "agree": []}
        for pname, x, sdx in (("pred_fog", pf, sdf), ("pred_sat", ps, sds)):
            for ax_ in ("A07", "A13"):
                y = sub[f"dA_{ax_}"].to_numpy(float)
                uninf = sdx < PRED_MIN_SD
                row = {"type": name, "statistic": "correlation",
                       "predictor": pname, "axis": ax_, "n_trials": n,
                       "n_scenes": len(sc), "uninformative": uninf}
                for cen, lab_ in ((True, "pearson"), (False, "uncentred")):
                    est = float(wstats(one_, x, y, cen)[0])
                    lo, hi = ci(wstats(Wt, x, y, cen))
                    row.update({lab_: est, f"{lab_}_lo": float(lo),
                                f"{lab_}_hi": float(hi)})
                nz = x != 0
                row["sign_agreement"] = float(
                    (np.sign(y[nz]) == np.sign(x[nz])).mean()) if nz.any() \
                    else np.nan
                out_rows.append(row)
                r["corr"].append(row)
        if sdf >= PRED_MIN_SD and sds >= PRED_MIN_SD:
            reg = {}
            Bs = {}
            for ax_ in ("A07", "A13"):
                y = sub[f"dA_{ax_}"].to_numpy(float)
                est = wreg2(one_, pf, ps, y)[0]
                bt = wreg2(Wt, pf, ps, y)
                Bs[ax_] = bt
                lo, hi = ci(bt)
                reg[ax_] = {"b_fog": (est[0], lo[0], hi[0]),
                            "b_sat": (est[1], lo[1], hi[1])}
                for j, pn in enumerate(("pred_fog", "pred_sat")):
                    out_rows.append({"type": name, "statistic":
                                     "joint regression coefficient",
                                     "predictor": pn, "axis": ax_,
                                     "n_trials": n, "n_scenes": len(sc),
                                     "estimate": est[j], "lo": lo[j],
                                     "hi": hi[j]})
            y7 = sub["dA_A07"].to_numpy(float)
            y13 = sub["dA_A13"].to_numpy(float)
            e7, e13 = wreg2(one_, pf, ps, y7)[0], wreg2(one_, pf, ps, y13)[0]
            for cname, est, bt in (
                    ("b_fog(A07) - b_fog(A13)", e7[0] - e13[0],
                     Bs["A07"][:, 0] - Bs["A13"][:, 0]),
                    ("b_sat(A13) - b_sat(A07)", e13[1] - e7[1],
                     Bs["A13"][:, 1] - Bs["A07"][:, 1])):
                lo, hi = ci(bt)
                reg[cname] = (est, float(lo), float(hi),
                              classify(float(lo), float(hi)))
                out_rows.append({"type": name, "statistic":
                                 "joint regression contrast",
                                 "predictor": cname, "axis": "",
                                 "n_trials": n, "n_scenes": len(sc),
                                 "estimate": est, "lo": float(lo),
                                 "hi": float(hi)})
            r["reg"] = reg
        return r

    res["all analysed"] = one("all analysed", tr)
    for t in REAL_TYPES:
        sub = tr[tr.trial_type == t]
        res[t] = one(t, sub) if len(sub) else None
    return res, pd.DataFrame(out_rows)


# ============================================================
# 9. Figures
# ============================================================

def heatmap(lab, key, D, cells, plan, effects, n_scenes, path):
    colours = COLOUR_UP[::-1] + [COLOUR_IDENTITY] + COLOUR_DOWN
    fogs = [FOG_IDENTITY] + FOG_LEVELS
    M = np.full((len(fogs), len(colours)), np.nan)
    S = np.full(M.shape, "", dtype=object)
    e = effects[(effects.scale == "sd_raw") & (effects.quantity == lab)
                & (effects.kind == GRID_KIND)]
    e = {(k, f): (d, lo, hi) for k, f, d, lo, hi in
         zip(e.colour_level, e.fog_level, e.d, e.lo, e.hi)}
    for i, f in enumerate(fogs):
        for j, k in enumerate(colours):
            if k == COLOUR_IDENTITY and f == FOG_IDENTITY:
                M[i, j] = 0.0
                S[i, j] = "ref"
                continue
            d, lo, hi = e[(k, f)]
            M[i, j] = d
            S[i, j] = f"{d:+.2f}" + mark(lo, hi).strip()
    lim = max(float(np.nanmax(np.abs(M))), 0.05)
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            rgb = cmap(norm(M[i, j]))[:3]
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            ax.text(j, i, S[i, j], ha="center", va="center", fontsize=9,
                    color=INK if lum > 0.45 else SURFACE)
    ax.set_xticks(range(len(colours)), colours, fontsize=9, color=INK)
    ax.set_yticks(range(len(fogs)), fogs, fontsize=9, color=INK)
    ax.set_xlabel("colour level  (vivid  <-  original  ->  drained)",
                  fontsize=9, color=INK_MUTED)
    ax.set_ylabel("fog level  (original at top)", fontsize=9, color=INK_MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mean change from the scene's original (corpus SD)",
                 fontsize=8, color=INK_MUTED)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED)
    cb.outline.set_visible(False)
    ax.set_title(f"{key}\nmean over {n_scenes} scenes; * = 95% scene-"
                 "resampled interval excludes 0", fontsize=10, color=INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, facecolor=SURFACE)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    t0 = RUN["t0"]
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    print("=" * 70)
    print("score_stimuli_on_axes_v4.py" + ("   [DRY RUN]" if DRY_RUN else ""))
    print("=" * 70)
    out = resolve_output_dir()
    os.makedirs(out, exist_ok=True)
    RUN["out"] = out
    print(f"Output: {out}")
    rng_main = np.random.default_rng(RANDOM_SEED)
    rng_corpus = np.random.default_rng([RANDOM_SEED, 1])
    rng_boot = np.random.default_rng([RANDOM_SEED, 2])
    rng_trial = np.random.default_rng([RANDOM_SEED, 3])
    rng_dir = np.random.default_rng([RANDOM_SEED, 4])

    # --- inputs and their checks (all scenes, also in a dry run) ---
    man, plan = load_manifest()
    scenes, idx = match_scenes(man)
    trials = load_trials(man)
    keys = load_axes()

    # --- scenes to encode ---
    if not DRY_RUN:
        sel = list(scenes.scene_id)
    elif DRY_RUN_SCENES:
        sel = list(DRY_RUN_SCENES)
        bad = [s for s in sel if s not in set(scenes.scene_id)]
        if bad:
            stop(f"DRY_RUN_SCENES not in the manifest: {bad}")
    else:
        n_tr = trials.groupby("scene_id").size()
        cnt = pd.DataFrame({"scene_id": scenes.scene_id.to_numpy()})
        cnt["n"] = cnt.scene_id.map(n_tr).fillna(0).astype(int)
        cnt = cnt.sort_values(["n", "scene_id"], ascending=[False, True])
        sel = list(cnt.scene_id[:DRY_RUN_N_SCENES])
    sel = sorted(sel)
    man_sel = man[man.scene_id.isin(sel)].reset_index(drop=True)
    sc_sel = scenes[scenes.scene_id.isin(sel)].reset_index(drop=True)
    print(f"\nScenes encoded: {len(sel)} of {len(scenes)}"
          + (f"  (dry run: {', '.join(sel)})" if DRY_RUN else ""))
    RUN["meta"]["scenes_encoded"] = sel

    # --- feature re-measurement (no encoder needed) ---
    remeasure_check(man_sel, rng_main)

    # --- encoder ---
    print("\nEncoder")
    enc = Un2CLIPEncoder()
    stim_paths = [os.path.join(STIMULI_DIR, f) for f in man_sel.filename]
    enc.info.update(enc.official_check(stim_paths))
    if HASH_CHECKPOINT:
        enc.info["un2clip_ckpt_sha256"] = sha256(UN2CLIP_CKPT_PATH)
    RUN["meta"]["encoder"] = enc.info

    # --- corpus reproduction ---
    Emb, status, stored_src, cinfo = corpus_checks(enc, idx, scenes, keys,
                                                   rng_corpus)
    RUN["meta"]["corpus_check"] = cinfo
    n_smp, r5 = direction_sd(keys, Emb, status, rng_dir)
    RUN["meta"]["direction_sd_sample"] = {"n": n_smp,
                                          "A07_sampled_sd_ratio": r5}

    # --- encode the stimuli ---
    print("\nEncoding the stimuli")
    Pz, Ez = enc.encode_paths(stim_paths, what="stimuli")
    nrm = np.linalg.norm(Ez, axis=1)
    check("S1", np.isfinite(Ez).all() and np.isfinite(Pz).all()
          and np.abs(nrm - 1).max() < 1e-4,
          f"{len(Ez)} stimulus embeddings finite, norm "
          f"{nrm.min():.6f} - {nrm.max():.6f}")
    pick = rng_main.choice(len(stim_paths), min(STIM_RECHECK_N,
                                                len(stim_paths)),
                           replace=False)
    cs = []
    for i in pick:
        _, e1 = enc.encode_paths([stim_paths[int(i)]], batch_size=1)
        cs.append(float(e1[0] @ Ez[int(i)]))
    check("S2", min(cs) >= STIM_RECHECK_MIN_COS,
          f"{len(cs)} stimuli re-encoded one at a time: min cos with their "
          f"stored rows {min(cs):.7f}")
    np.save(P("embeddings_768.npy"), Ez.astype(np.float32))
    np.save(P("pooled_1024.npy"), Pz.astype(np.float32))

    # --- scores ---
    A = np.stack(keys.V.to_list())
    mu = keys.mean_raw.to_numpy(float)
    sd = keys.sd_raw.to_numpy(float)
    raw = Ez.astype(np.float64) @ A.T
    z = (raw - mu) / sd
    idcols = ["scene_id", "filename", "use", "kind", "colour_level",
              "fog_level", "contrast_level", "chroma", "haze",
              "contrast_gain", "is_ref", FEAT_SAT, FEAT_FOG]
    isc = man_sel[idcols].copy()
    isc.insert(0, "row", np.arange(len(isc)))
    isc = pd.concat([isc, pd.DataFrame(raw, columns=[f"raw_{l}" for l in
                                                     keys.label]),
                     pd.DataFrame(z, columns=[f"z_{l}" for l in keys.label])],
                    axis=1)
    isc.to_csv(P("image_scores.csv"), index=False, float_format="%.6g")
    keys.drop(columns="V").to_csv(P("score_keys.csv"), index=False,
                                  float_format="%.8g")

    # --- reference diagnostic ---
    refrow = man_sel.index[man_sel.is_ref].to_numpy()
    ref_sid = man_sel.scene_id.to_numpy()[refrow]
    srcpos = {s: i for i, s in enumerate(scenes.scene_id)}
    st = stored_src[[srcpos[s] for s in ref_sid]]
    cref = (Ez[refrow].astype(np.float64) * st).sum(1) / np.linalg.norm(
        st, axis=1)
    refdf = pd.DataFrame({"scene_id": ref_sid,
                          "image_id": scenes.image_id.to_numpy()[
                              [srcpos[s] for s in ref_sid]],
                          "match": scenes.match.to_numpy()[
                              [srcpos[s] for s in ref_sid]],
                          "cos_reference_vs_stored": cref,
                          "warning": cref < REF_MIN_COS_WARN})
    refdf.to_csv(P("reference_check.csv"), index=False, float_format="%.7f")
    low = refdf[refdf.warning]
    check("R1", not len(low),
          f"reference stimulus vs stored corpus embedding of the same "
          f"photograph: min cos {cref.min():.5f}, median "
          f"{np.median(cref):.5f}; below {REF_MIN_COS_WARN}: {len(low)}"
          + (f" ({', '.join(low.scene_id[:5])})" if len(low) else ""),
          critical=False)

    # --- effects ---
    print("\nEffects")
    sids, cells, R, F = scene_arrays(man_sel, plan, raw, keys)
    W = boot_weights(len(sids), rng_boot)
    D, Dc, dsat, dcl, eff, v4i, nci = cell_effects(plan, sids, cells, R, F,
                                                   keys, W)
    eff.to_csv(P("cell_effects.csv"), index=False, float_format="%.5g")
    # scene_effects.csv
    lab = list(keys.label)
    ref = cells.index(cell_key(COLOUR_IDENTITY, FOG_IDENTITY, ""))
    pl = plan.set_index("cell")
    rows = []
    for s, sid in enumerate(sids):
        for c, ck in enumerate(cells):
            if c == ref:
                continue
            p = pl.loc[ck]
            r = {"scene_id": sid, "cell": cell_label(p.colour_level,
                                                      p.fog_level,
                                                      p.contrast_level),
                 "kind": p.kind, "use": p.use,
                 "colour_level": p.colour_level, "fog_level": p.fog_level,
                 "contrast_level": p.contrast_level,
                 "d_sat_mean": F[s, c, 0] - F[s, ref, 0],
                 "d_contrast_local": F[s, c, 1] - F[s, ref, 1],
                 "d_sat_mean_sd": dsat[s, c], "d_contrast_local_sd": dcl[s, c]}
            for j, l in enumerate(lab):
                r[f"d_{l}"] = D[s, c, j]
            for j in v4i:
                r[f"dcol_{lab[j]}"] = Dc[s, c, j]
            for o in ("A13", "A11", "A14"):
                r[f"d_A07-{o}"] = D[s, c, lab.index("A07")] - \
                    D[s, c, lab.index(o)]
            rows.append(r)
    pd.DataFrame(rows).to_csv(P("scene_effects.csv"), index=False,
                              float_format="%.6g")

    # --- questions (a)-(d) on three scale / set combinations ---
    Jv4 = {i: lab.index(i) for i in V4_IDS}
    Jnc = {i: lab.index("nc_" + i) for i in V4_IDS}
    ql = {}
    for name, X, J in (("v4 / sd_raw", D, Jv4),
                       ("v4 / sd_raw_colour", Dc, Jv4),
                       ("v4@nocontrol / sd_raw", D, Jnc)):
        q = question_levels(X, J, cells, W)
        q.insert(0, "analysis", name)
        ql[name] = q
    qall = pd.concat(ql.values(), ignore_index=True)
    qall.to_csv(P("question_levels.csv"), index=False, float_format="%.5g")

    add = additivity(cells, D, dsat, dcl, keys, W)
    add.to_csv(P("additivity.csv"), index=False, float_format="%.5g")

    per, slope_sum, slope_con, n_slope_cells = feature_slopes(
        plan, sids, cells, D, dsat, dcl, keys, W)
    per.to_csv(P("scene_slopes.csv"), index=False, float_format="%.5g")

    # --- trials ---
    print("Trials")
    rowpos = {f: i for i, f in enumerate(man_sel.filename)}
    tr = trials[trials.file_a.isin(rowpos) & trials.file_b.isin(rowpos)] \
        .reset_index(drop=True)
    ia = tr.file_a.map(rowpos).to_numpy(int)
    ib = tr.file_b.map(rowpos).to_numpy(int)
    sdc = keys.sd_raw_colour.to_numpy(float)
    dA = (raw[ia] - raw[ib]) / sd
    keep = ["trial", "trial_type", "scene_id", "file_a", "file_b",
            "left_file", "right_file", "a_is_left", "pred_sat", "pred_fog",
            "pred_sat_check", "pred_fog_check"] + \
        [c for c in ("d_sat", "d_fog", "mag_bin") if c in tr.columns]
    tdf = tr[keep].copy()
    for j, l in enumerate(lab):
        tdf[f"dA_{l}"] = dA[:, j]
    for j in v4i:
        tdf[f"dAcol_{lab[j]}"] = (raw[ia, j] - raw[ib, j]) / sdc[j]
    tdf.to_csv(P("trial_axis_differences.csv"), index=False,
               float_format="%.6g")
    tres, tstats = trial_analysis(tdf, rng_trial)
    tstats.to_csv(P("trial_stats.csv"), index=False, float_format="%.5g")

    # --- figures ---
    for l in KEY_AXES:
        heatmap(l, keys.key.iat[lab.index(l)], D, cells, plan, eff, len(sids),
                P(f"heatmap_{l}.png"))

    # --- report ---
    write_report(man, man_sel, scenes, sel, keys, refdf, eff, qall, add,
                 per, slope_sum, slope_con, n_slope_cells, tdf, tres, cells,
                 plan, len(trials), r5, n_smp)
    write_meta("finished")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Report: "
          f"{P('report.txt')}")


# ============================================================
# Report
# ============================================================

def write_report(man, man_sel, scenes, sel, keys, refdf, eff, qall, add,
                 per, slope_sum, slope_con, n_slope_cells, tdf, tres, cells,
                 plan, n_trials_all, r5, n_smp):
    L = []
    W_ = L.append
    nS = len(sel)
    lab = list(keys.label)
    k07 = keys.key.iat[lab.index("A07")]
    k13 = keys.key.iat[lab.index("A13")]
    ctl = [keys.label.iat[np.where(keys.key == k)[0][0]]
           for k in REPORT_CONTROLS]

    W_("SCORING THE FINAL STIMULI ON THE v4 AXES  (score_stimuli_on_axes_v4.py)")
    W_("=" * 78)
    W_(f"Date      : {time.strftime('%Y-%m-%d %H:%M')}")
    if DRY_RUN:
        W_(f"MODE      : DRY RUN -- {nS} of {len(scenes)} scenes encoded "
           f"({', '.join(sel)}).")
        W_("            Intervals and verdicts from 3 scenes mean nothing; "
           "this run tests the pipeline.")
    else:
        W_(f"Mode      : full run, {nS} scenes")
    W_(f"Stimuli   : {MANIFEST_CSV} ({len(man_sel)} of {len(man)} images "
       "encoded)")
    W_(f"Trials    : {TRIAL_LIST_CSV} ({len(tdf)} analysed trials with both "
       "images encoded)")
    W_(f"Encoder   : {CLIP_MODEL_NAME} + un2CLIP "
       f"({os.path.basename(UN2CLIP_CKPT_PATH)}), own projection, 'openai' "
       "preprocessing, fp32, TF32 off (as 03)")
    W_(f"Axes      : {AXES_DIR}; corpus means and SDs from "
       f"{SCORES_DIR}/score_catalog.csv")
    W_("")
    W_("WHAT THIS IS, AND WHAT IT IS NOT")
    W_("-" * 78)
    W_("  CLIP is not a human observer. This checks that the edits move the "
       "text-defined")
    W_("  concepts (A07 fog, A13 vividless, ...) in the intended direction "
       "and more than the")
    W_("  neighbouring concepts. It does not show what participants will "
       "see.")
    W_("")

    # ---- 1 checks ----
    W_("1. CHECKS")
    W_("-" * 78)
    for k, v in RUN["checks"].items():
        W_(f"  {k:<4} {v['status']:<5} {v['detail']}")
    if RUN["warnings"]:
        W_("  Warnings:")
        for w in RUN["warnings"]:
            W_(f"    - {w}")
    W_(f"  Control directions: corpus SD estimated from {n_smp:,} stored "
       f"embeddings (sampled SD / stored SD for A07 = {r5:.4f}).")
    W_("")

    # ---- 2 criteria ----
    W_("2. CRITERIA, STATED BEFORE THE RESULTS")
    W_("-" * 78)
    W_("  Changes: per scene, each cell minus the scene's own k=1, h=0 "
       "stimulus, in units of")
    W_("  the axis's corpus SD (sd_raw over all 1.1M encoded images, as 06 "
       "and 09c).")
    W_(f"  Intervals: {int(CI_LEVEL * 100)}% percentile intervals from "
       f"{N_BOOT} resamples of scenes. They describe scenes")
    W_("  like these hand-screened ones, not photographs in general. These "
       "are descriptive")
    W_("  checks: the intervals are NOT corrected for multiple comparisons.")
    W_("")
    W_("  Each level is tested on a quantity whose intended sign is given. "
       "Classes:")
    W_("    shown      interval entirely on the intended side of 0")
    W_("    not shown  interval includes 0 (a weak step, not a wrong one)")
    W_("    reversed   interval entirely on the other side of 0")
    W_("  Verdict per question: PASS if every level is shown; PARTIAL if no "
       "level is reversed")
    W_("  but at least one is not shown; FAIL if any level is reversed.")
    W_("")
    W_(f"  (a) {QUESTIONS['a']}.")
    W_(f"      Levels {', '.join(FOG_LEVELS)} at {COLOUR_IDENTITY}; tested "
       "d(A07) - d(A13), intended > 0.")
    W_(f"  (b) {QUESTIONS['b']}.")
    W_(f"      Levels {', '.join(COLOUR_DOWN)} at {FOG_IDENTITY}; tested "
       "d(A13) - d(A07), intended > 0.")
    W_(f"  (c) {QUESTIONS['c']}.")
    W_(f"      Levels {', '.join(COLOUR_UP)} at {FOG_IDENTITY}; tested "
       "d(A13), intended < 0.")
    W_(f"  (d) {QUESTIONS['d']}.")
    W_(f"      Levels {', '.join(FOG_LEVELS)} at {COLOUR_IDENTITY}; tested "
       "d(A07) - d(A11) and d(A07) - d(A14), intended > 0")
    W_("      (10 tests).")
    W_("")
    W_("  Primary: v4 axes, sd_raw. Also reported: the verdicts with "
       "sd_raw_colour (each axis")
    W_("  divided by its SD over colour images; only changes are listed), "
       "and v4@nocontrol")
    W_("  (sensitivity only: the darkness direction also reads colour, so "
       "nocontrol understates")
    W_("  colour effects).")
    W_("")

    # ---- 3 reference ----
    W_("3. REFERENCE IMAGE")
    W_("-" * 78)
    W_("  Each scene's k=1, h=0 stimulus is the unmanipulated 512 px square "
       "of CLIP's own region")
    W_("  of the photograph. Its cosine with the stored corpus embedding of "
       "the same photograph")
    W_(f"  (only the resampling differs; warning below {REF_MIN_COS_WARN}):")
    c = refdf.cos_reference_vs_stored
    W_(f"    min {c.min():.5f}   median {c.median():.5f}   max {c.max():.5f}"
       f"   below {REF_MIN_COS_WARN}: {int(refdf.warning.sum())}")
    for r in refdf.sort_values("cos_reference_vs_stored").head(
            len(refdf) if nS <= 10 else 5).itertuples():
        W_(f"    {r.scene_id:<44} {r.cos_reference_vs_stored:.5f}"
           f"  ({r.match}){'  WARNING' if r.warning else ''}")
    if nS > 10:
        W_("    (lowest five shown; every scene in reference_check.csv)")
    W_("")

    # ---- 4 questions ----
    W_("4. QUESTIONS (a)-(d)   [v4 axes, sd_raw]")
    W_("-" * 78)
    qp = qall[qall.analysis == "v4 / sd_raw"]
    verdicts = {}
    for q in "abcd":
        d = qp[qp.question == q]
        v = verdict(d["class"])
        verdicts[q] = v
        W_(f"  ({q}) {QUESTIONS[q]}")
        W_(f"      {'level':<14}{'tested':<18}{'mean [95% interval]':<26}"
           f"{'same':>5}  class")
        for lv, ts, m, lo, hi, ss, cl in zip(d.level, d.tested, d["mean"],
                                             d.lo, d.hi, d.same_sign,
                                             d["class"]):
            W_(f"      {lv:<14}{ts:<18}{fci(m, lo, hi):<26}{ss:>5.2f}  {cl}")
        ns = list(d.level[d["class"] == "not shown"])
        rv = list(d.level[d["class"] == "reversed"])
        W_(f"      VERDICT: {v}"
           + (f"   not shown: {', '.join(ns)}" if ns else "")
           + (f"   reversed: {', '.join(rv)}" if rv else ""))
        if q == "a" and v != "PASS":
            W_("      On the trials shown (section 9, joint no-intercept "
               "regression on all analysed")
            W_("      trials; does pred_fog track A07 more than A13 with "
               "pred_sat held fixed?):")
            rg = tres.get("all analysed")
            if rg and rg["reg"]:
                g = rg["reg"]
                for ax_ in ("A07", "A13"):
                    b = g[ax_]["b_fog"]
                    W_(f"        b_fog on d{ax_:<4} {fci(*b)}")
                e = g["b_fog(A07) - b_fog(A13)"]
                W_(f"        b_fog(A07) - b_fog(A13) {fci(*e[:3])}  -> {e[3]}")
            else:
                W_("        (not estimable on the trials encoded)")
        W_("")

    W_("  4b. Sensitivity: v4@nocontrol (sd_raw)")
    qn = qall[qall.analysis == "v4@nocontrol / sd_raw"]
    for q in "abcd":
        d = qn[qn.question == q]
        dp = qp[qp.question == q]
        ch = [f"{a}: {x} -> {y}" for a, x, y in zip(d.level, dp["class"],
                                                    d["class"]) if x != y]
        W_(f"      ({q}) {verdict(d['class']):<8}"
           + (f"  levels changed: {'; '.join(ch)}" if ch
              else "  (same classes as v4)"))
    W_("")
    W_("  4c. With sd_raw_colour instead of sd_raw (only changes listed)")
    qc = qall[qall.analysis == "v4 / sd_raw_colour"]
    any_change = False
    for q in "abcd":
        d = qc[qc.question == q]
        dp = qp[qp.question == q]
        vc = verdict(d["class"])
        ch = [f"{a}: {x} -> {y}" for a, x, y in zip(d.level, dp["class"],
                                                    d["class"]) if x != y]
        if vc != verdicts[q] or ch:
            any_change = True
            W_(f"      ({q}) verdict {verdicts[q]} -> {vc}"
               + (f"; levels: {'; '.join(ch)}" if ch else ""))
    if not any_change:
        W_("      No verdict and no level class changes.")
    sdr = keys.set_index("label")
    W_("      SDs (raw cosine units), all images / colour images: "
       + ", ".join(f"{i} {sdr.at[i, 'sd_raw']:.4f}/"
                   f"{sdr.at[i, 'sd_raw_colour']:.4f}"
                   for i in ("A07", "A13", "A11", "A14")))
    W_("")

    # ---- 5 additivity ----
    W_("5. (e) DO FOG AND COLOUR COMBINE ADDITIVELY?")
    W_("-" * 78)
    W_("  Per scene: interaction = d(k,h) - d(k,h=0) - d(k=1,h), for the "
       f"{len(COLOUR_UP + COLOUR_DOWN) * len(FOG_LEVELS)} combined cells;")
    W_("  0 = additive. Feature rows separate non-additivity in the stimuli "
       "from that in CLIP.")
    show = ["A07", "A13", "A11", "A14"] + ctl[:1] + ["d_sat_mean_sd",
                                                     "d_contrast_local_sd"]
    W_(f"  {'quantity':<22}{'cells *':>8}{'+':>4}{'-':>4}"
       f"{'median |inter|':>16}{'median |obs|':>14}")
    for qn_ in show:
        d = add[add.quantity == qn_]
        ex = d[d.interval == "excludes 0"]
        W_(f"  {qn_:<22}{len(ex):>5}/{len(d):<2}{int((ex.interaction > 0).sum()):>4}"
           f"{int((ex.interaction < 0).sum()):>4}"
           f"{d.interaction.abs().median():>16.3f}"
           f"{d.observed.abs().median():>14.3f}")
    W_("  (cells * = intervals excluding 0; + / - = their sign; units: "
       "corpus SD of the axis or")
    W_("  of the feature)")
    for qn_ in ("A07", "A13"):
        W_("")
        W_(f"  Interaction for {qn_} (rows fog, columns colour; * interval "
           "excludes 0):")
        cols = COLOUR_UP[::-1] + COLOUR_DOWN
        W_("    " + " " * 6 + "".join(f"{x:>9}" for x in cols))
        d = add[add.quantity == qn_].set_index(["colour_level", "fog_level"])
        for f in FOG_LEVELS:
            W_(f"    {f:<6}" + "".join(
                f"{d.at[(k, f), 'interaction']:>+8.2f}"
                f"{mark(d.at[(k, f), 'lo'], d.at[(k, f), 'hi'])}"
                for k in cols))
    W_("")

    # ---- 6 contrast-up ----
    W_("6. (f) THE CONTRAST-UP CELLS (rendered, not shown; descriptive)")
    W_("-" * 78)
    cu = eff[(eff.kind == CONTRAST_UP_KIND) & (eff.scale.isin(
        ["sd_raw", "feature_sd"]))]
    qs = ["A07", "A13", "A11", "A14"] + ctl + ["d_contrast_local_sd",
                                                "d_sat_mean_sd"]
    for cl, d in cu.groupby("contrast_level", sort=True):
        d = d[d.set != "v4@nocontrol"].set_index("quantity")
        W_(f"  gain {cl}: " + "; ".join(
            f"{q} {fci(d.at[q, 'd'], d.at[q, 'lo'], d.at[q, 'hi'])}"
            for q in qs[:4]))
        W_(f"  {'':<13}" + "; ".join(
            f"{q} {fci(d.at[q, 'd'], d.at[q, 'lo'], d.at[q, 'hi'])}"
            for q in qs[4:]))
    W_("")

    # ---- 7 cell effects ----
    W_("7. CELL EFFECTS: KEY AXES AND CONTROLS (mean d, corpus SD; * "
       "interval excludes 0)")
    W_("-" * 78)
    W_("   top = the v4 axis with the largest share d^2 / sum d^2 over the "
       "nine (0.11 = all equal)")
    hdr = ["A07", "A13", "A11", "A14"] + ctl
    W_(f"  {'cell':<12}" + "".join(f"{h[:7]:>9}" for h in hdr)
       + f"{'dsat':>7}{'dcl':>7}  top")
    g = eff[(eff.scale == "sd_raw") & (eff.kind == GRID_KIND)
            & eff.set.isin(["v4", "controls"]) & (eff.key != "")]
    gf = eff[(eff.scale == "feature_sd") & (eff.kind == GRID_KIND)]
    for cl in g.cell.drop_duplicates():
        d = g[g.cell == cl].set_index("quantity")
        f_ = gf[gf.cell == cl].set_index("quantity")
        v4d = d[d.set == "v4"]
        top = v4d.share.idxmax()
        W_(f"  {cl:<12}" + "".join(
            f"{d.at[h, 'd']:>+8.2f}{mark(d.at[h, 'lo'], d.at[h, 'hi'])}"
            for h in hdr)
           + f"{f_.at['d_sat_mean_sd', 'd']:>+7.2f}"
           f"{f_.at['d_contrast_local_sd', 'd']:>+7.2f}  {top} "
           f"{v4d.at[top, 'share']:.2f}")
    W_("  dsat / dcl = change of sat_mean / contrast_local in feature SD "
       f"units ({SD_SAT} / {SD_FOG}).")
    W_("  Every cell, axis, interval and set: cell_effects.csv. Heatmaps: "
       "heatmap_<axis>.png.")
    W_("")

    # ---- 8 slopes ----
    W_("8. PER FEATURE UNIT (within-scene regression, no intercept)")
    W_("-" * 78)
    W_(f"  Per scene, over its {n_slope_cells} manipulated colour x fog "
       "cells: d(axis) = b_fog * fog + b_desat * desat,")
    W_(f"  fog = -d contrast_local / {SD_FOG}, desat = -d sat_mean / "
       f"{SD_SAT} (oriented like pred_fog / pred_sat:")
    W_("  positive = foggier / less colourful). Slopes are axis SD per "
       "feature SD, averaged over")
    W_("  scenes; R^2 is uncentred (no-intercept model).")
    W_(f"  {'axis':<18}{'per SD fog':<26}{'per SD desaturation':<26}"
       f"{'R2 med (min)':>14}")
    for s in slope_sum:
        W_(f"  {s['axis']:<18}{fci(*s['fog'][:3]):<26}"
           f"{fci(*s['desat'][:3]):<26}"
           f"{s['r2_median']:>8.2f} ({s['r2_min']:.2f})")
    W_("  Specificity of the slopes:")
    for k, v in slope_con.items():
        W_(f"    {k:<32}{fci(*v[:3])}  same sign {v[3]:.2f}")
    rx = per.drop_duplicates("scene_id").r_fog_desat
    W_(f"  Within-scene r(fog, desat) over the cells: median {rx.median():+.2f}"
       f" (range {rx.min():+.2f} to {rx.max():+.2f}).")
    W_("")

    # ---- 9 trials ----
    W_("9. THE TRIALS PARTICIPANTS WILL SEE")
    W_("-" * 78)
    W_("  dA = axis(image A) - axis(image B) in corpus SD, oriented like "
       "pred_*: pred_fog > 0 =")
    W_("  A foggier, pred_sat > 0 = A less colourful. Expected: pred_fog "
       "with dA07 and pred_sat")
    W_("  with dA13 positive. Intervals resample scenes. 'uninformative' = "
       f"predictor SD < {PRED_MIN_SD}")
    W_("  within the type (nearly constant by construction).")
    if DRY_RUN:
        W_(f"  DRY RUN: only the {len(tdf)} analysed trials whose images "
           "are both encoded.")
    for t in ["all analysed"] + REAL_TYPES:
        r = tres.get(t)
        if r is None:
            W_(f"  {t}: no trials" + (" in the encoded scenes" if DRY_RUN
                                      else ""))
            continue
        i = r["info"]
        W_("")
        W_(f"  {t}: {i['n_trials']} trials, {i['n_scenes']} scenes; SD of "
           f"pred_fog {i['sd_pred_fog']:.2f}, of pred_sat "
           f"{i['sd_pred_sat']:.2f}")
        W_(f"    {'pair':<20}{'Pearson r [interval]':<26}"
           f"{'uncentred r [interval]':<26}{'signs':>6}")
        for row in r["corr"]:
            nm = f"{row['predictor']} x d{row['axis']}"
            W_(f"    {nm:<20}"
               f"{fci(row['pearson'], row['pearson_lo'], row['pearson_hi']):<26}"
               f"{fci(row['uncentred'], row['uncentred_lo'], row['uncentred_hi']):<26}"
               f"{row['sign_agreement']:>6.2f}"
               + ("  uninformative" if row["uninformative"] else ""))
        if r["reg"]:
            g = r["reg"]
            W_("    joint regression, no intercept: dA = b_fog * pred_fog + "
               "b_sat * pred_sat")
            for ax_ in ("A07", "A13"):
                W_(f"      d{ax_:<5} b_fog {fci(*g[ax_]['b_fog']):<24} "
                   f"b_sat {fci(*g[ax_]['b_sat'])}")
            for cname in ("b_fog(A07) - b_fog(A13)", "b_sat(A13) - b_sat(A07)"):
                e = g[cname]
                W_(f"      {cname:<25} {fci(*e[:3])}  -> {e[3]}")
        else:
            W_("    joint regression: not fitted (a predictor is nearly "
               "constant in this type)")
    W_("  signs = share of trials where the axis difference has the "
       "predictor's sign.")
    W_("")

    # ---- 10 files ----
    W_("10. FILES")
    W_("-" * 78)
    W_(f"  {RUN['out']}: report.txt, embeddings_768.npy, pooled_1024.npy "
       "(float32), image_scores.csv,")
    W_("  score_keys.csv, scene_effects.csv, cell_effects.csv, "
       "question_levels.csv, additivity.csv,")
    W_("  scene_slopes.csv, trial_axis_differences.csv, trial_stats.csv, "
       "corpus_check.csv,")
    W_("  reference_check.csv, " + ", ".join(f"heatmap_{l}.png"
                                           for l in KEY_AXES)
       + ", run_meta.json")
    W_(f"  Elapsed: {(time.time() - RUN['t0']) / 60:.1f} min")
    with open(P("report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    RUN["meta"]["verdicts_v4_sd_raw"] = verdicts


if __name__ == "__main__":
    main()
