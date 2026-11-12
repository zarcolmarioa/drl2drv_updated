"""
build_trials_v5b.py  --  v5b step 3: build the two block lists (sets A and B)
with the normality leak cancelled

WHAT CHANGED FROM build_trials_v5.py (v5 -> v5b)
    * The pairs are the v5 STIMULUS SET's (47 scenes x 49 cells), made by
      script 1 (analyse_pair_capacity_v5b.py, imported). Pair types come
      from measured differences; the minimum differences are a named
      threshold set (THRESHOLD_SET, default v5units); the mix is
      balanced_mixed (0.30 / 0.30 / 0.20 / 0.20).
    * THE NORMALITY LIMIT. For each set's analysed trials, the normality
      term N (positive = image A, the predicted image, is the more typical)
      is regressed on both direction terms together, with no intercept:
          N ~ b_sat * x_sat + b_con * x_con
      under both definitions of "typical" (the corpus median, and the
      scene's own original). Every |b_sat| and |b_con| must be at most
      LEAK_LIMIT (0.05) in both sets under both definitions. The uncentred
      correlations of N with each term are reported.
    * CONSTRUCTION. A random draw never meets the limit under the corpus
      definition (script 2: b_sat about +0.19, b_con about -0.12), so each
      candidate is REPAIRED, not filtered: an integer program keeps as many
      of the draw's pairs as it can and replaces the rest with unused pairs
      from the SAME scene x type x magnitude-bin cells, so every count, bin,
      scene and per-scene cap of the draw is unchanged, while
        - the leak moments sum(x * N) are held within REPAIR_TARGET of
          sum(x^2) for each term, definition and set (a near-diagonal
          version of |b| <= REPAIR_TARGET),
        - the information per weight is matched between the sets within
          REPAIR_INFO_TOL_TOTAL overall and REPAIR_INFO_TOL_TYPE per type
          (tighter than script 1's 5% and 10%),
        - no pair is in both sets, and the draw's reserved practice and
          catch pairs are left out.
      The repaired sets are then checked EXACTLY: the no-intercept fit must
      give |b| <= LEAK_LIMIT everywhere, and every one of script 1's checks
      must pass (script 1's evaluate: counts, bins, information, KS, cos,
      VIF, scenes, sides, reserves). If the exact fit misses, the target is
      tightened (x REPAIR_TIGHTEN) and the repair re-solved, up to
      REPAIR_TRIES times. A candidate whose repair fails is dropped.
      WIDE REPAIR: only when fewer than WIDE_MIN_PASSING candidates pass at
      a size, the failed ones are repaired again with the scenes inside each
      type x bin free to change (counts, bins and scene matching between
      the sets still exact); the report says when this was used.
    * SELECTION. For each weight, the weaker set's information is divided
      by the mean weaker-set information of the UNREPAIRED candidates (that
      pass script 1's checks) at that size; the candidate whose SMALLER
      relative value is highest is shipped, ties broken on the smaller worst
      |b|, then the lower candidate number. The same rule applied to the
      unrepaired candidates gives the UNCONSTRAINED selection, against which
      the information cost of the limit is reported for each weight.
    * PARALLEL. Candidates are drawn and repaired in N_WORKERS separate
      processes, each limited to one thread (BLAS and the solver). Every
      candidate has its own named random streams and results are gathered
      and selected in candidate order, so the result is the same for any
      number of workers; the chosen candidate is rebuilt in the main
      process and must give byte-identical sets.
    * UNEDITED IMAGES are allowed (v5 forbade them) and counted.
    * IMAGES are checked against the stimulus set's checksums.sha256 as
      well as copied, hashed and re-measured (v4's feature code).
    * LIST ROWS carry the level columns of each image, the image z values,
      the normality terms (N_corpus, N_orig and their parts per feature)
      and the direction terms, in every row (analysed, consistency,
      practice and catch), for scripts 4-9.
    * REPORT ADDITIONS, before vs after repair, for the shipped sets and as
      a summary over candidates: where the images sit relative to the
      original (above / at / below, per feature and per side), pairs that
      straddle the original, pairs with the unedited image, and the leak
      by trial type (reported, not limited). The leak of the practice and
      catch trials is reported (not limited).

WHAT THIS DOES (per block size in BLOCK_SIZES; one package each)
    1. CANDIDATES: N_CANDIDATES draws with script 1's one_draw (script 1's
       and script 2's seeds: candidate d is their draw d), each repaired.
    2. SELECTION as above.
    3. CONSISTENCY trials: CONSISTENCY_RATE of each block, stratified by
       type, the same couples in both sets, sides swapped (v5).
    4. PRACTICE per block position (PRACTICE_BY_BLOCK), v4's rule (v5).
    5. CATCH per set: CATCH_PER_BLOCK in 'no_images' mode (v5).
    6. IMAGES: copied, compared with the source and the set's checksums,
       re-measured. Any failure leaves BUILD_FAILED.txt and exits 1.
    7. THRESHOLDS per set from N_THRESHOLD_PEOPLE random choosers (v5).

INPUTS (relative to the project folder; run from there)
    analyse_pair_capacity_v5b.py, simulate_power_v4_n10_20.py
    STIMULUS_SET_DIR    the frozen v5 stimulus set (stimulus_manifest.csv,
                        stimuli/, checksums.sha256, FROZEN.json)
    SCRIPT1_RUN_DIR     optional: script 1's run (all_pairs.csv.gz,
                        bin_edges.csv), for the check that the pairs and bins
                        are script 1's
    SCRIPT2_RUN_DIR     optional: script 2's run (normality_leak.csv), for
                        the check that the candidates are script 2's draws

OUTPUTS: one folder per block size, OUTPUT_ROOT/<settings>__s<hash>/
    set_A.csv, set_B.csv       each set's analysed, consistency and catch
                               rows, unordered, with a stable item_id
    practice_block1.csv, practice_block2.csv
    candidates.csv             every candidate before and after repair:
                               information, leak, positions, selection
    leak_by_type.csv           the leak by trial type, before and after
    set_info.json              counts, leak, thresholds, provenance
    manifest.csv               every shipped image: SHA-256, re-measured
    stimuli/                   the images
    build_report.txt, checks.csv, settings.json
    BUILD_FAILED.txt           only when a check fails

REQUIRES  numpy, pandas, scipy >= 1.9, opencv-python, Pillow
RUN       python build_trials_v5b.py
"""

# One thread per process (BLAS and OpenMP), set before numpy is imported,
# so a run gives the same result for any N_WORKERS.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

# ============================================================
# SETTINGS
# ============================================================

# --- Imported scripts (in the project folder) ---
CAPACITY_SCRIPT = "analyse_pair_capacity_v5b.py"
V4_SIM_SCRIPT = "simulate_power_v4_n10_20.py"

# CAPACITY_OVERRIDES
#   Script 1 settings to change for this build only. Empty = script 1's
#   file as saved (MANIFEST_CSV is always set from STIMULUS_SET_DIR). Every
#   script 1 setting in force goes to settings.json.
CAPACITY_OVERRIDES = {}

# --- The stimulus set ---
STIMULUS_SET_DIR = "data/stimulus_set_v5/7fd72262ba"
MANIFEST_CSV = f"{STIMULUS_SET_DIR}/stimulus_manifest.csv"
STIMULI_SRC = f"{STIMULUS_SET_DIR}/stimuli"
CHECKSUMS_FILE = f"{STIMULUS_SET_DIR}/checksums.sha256"
FROZEN_FILE = f"{STIMULUS_SET_DIR}/FROZEN.json"

# SCRIPT1_RUN_DIR / SCRIPT2_RUN_DIR
#   Optional cross-checks. A folder, or a pattern that must match exactly
#   one folder; None, no match or several matches = the check is SKIPPED
#   (the report says why).
SCRIPT1_RUN_DIR = "data/pair_capacity_v5b/*__s554900b6"
SCRIPT2_RUN_DIR = "data/block_length_v5b/*"

# --- The design ---
BLOCK_SIZES = [180, 150]           # one package per size; 180 primary
THRESHOLD_SET = "v5units"          # a key of script 1's THRESHOLD_SETS
MIX = "balanced_mixed"             # added to script 1's TYPE_MIXES here
MIX_SHARES = {"sat_only": 0.30, "fog_only": 0.30,
              "together": 0.20, "conflict": 0.20}

# --- Candidates ---
N_CANDIDATES = 40                  # per size

# N_WORKERS
#   Processes that draw and repair candidates. None = the number of cores
#   minus one; 1 = no parallelism (everything in this process). The result
#   does not depend on it.
N_WORKERS = None

# --- The normality limit (analysed trials only) ---
# LEAK_LIMIT
#   Largest allowed |b_sat| and |b_con| of the no-intercept fit, in each set,
#   under each definition (corpus median and the scene's original).
LEAK_LIMIT = 0.05

# TYPE_LIMITS
#   Further limits, per trial type, on top of the overall one: type ->
#   (term, limit). N of that type's analysed trials is regressed on the ONE
#   direction term named, no intercept, and |b| <= limit is required in each
#   set under each definition. One term suits sat_only (x_con ~ 0), fog_only
#   (x_sat ~ 0) and conflict (x_sat moves in lockstep with x_con, cos about
#   -0.96, so one term describes it). Together trials are not limited (only
#   reported). {} = the overall limit alone (the first v5b build).
TYPE_LIMITS = {"sat_only": ("sat", 0.05),
               "fog_only": ("con", 0.05),
               "conflict": ("con", 0.10)}

# TYPE_TARGET_SHARE
#   The repair holds each type's single-term b within this share of its
#   limit (a one-term fit is exact, so a small margin is enough).
TYPE_TARGET_SHARE = 0.95

# LIMIT_STEPS / COMPARE_LIMIT_STEPS
#   With COMPARE_LIMIT_STEPS, every candidate is also repaired under each
#   step below (each a list of TYPE_LIMITS keys added to the overall limit)
#   and the report gives, per step: the candidates passing, the information
#   of the set each step would ship and of every candidate against the first
#   step (the overall limit alone, i.e. the first v5b build), and where the
#   images sit. The shipped sets use all of TYPE_LIMITS. Each step adds
#   about one more candidate run's time.
COMPARE_LIMIT_STEPS = True
LIMIT_STEPS = [("overall only", []),
               ("+ single-feature", ["sat_only", "fog_only"]),
               ("+ conflict", ["sat_only", "fog_only", "conflict"])]

# COST_RULE
#   (largest information cost, fewest passing candidates) at which a step is
#   judged acceptable in the report: its cost against the step before, for
#   each weight, on the shipped sets and on the mean over candidates. The
#   report states the verdict; it does not change what is shipped.
COST_RULE = (0.05, 5)

# REPAIR_TARGET / REPAIR_TIGHTEN / REPAIR_TRIES
#   The repair holds |sum(x*N)| <= target * sum(x^2) per term, definition
#   and set. If the exact fit then misses LEAK_LIMIT, the target is
#   multiplied by REPAIR_TIGHTEN and the repair re-solved, up to
#   REPAIR_TRIES solves in all.
REPAIR_TARGET = 0.04
REPAIR_TIGHTEN = 0.75
REPAIR_TRIES = 3

# REPAIR_INFO_TOL_TOTAL / REPAIR_INFO_TOL_TYPE
#   Relative A-B difference in information the repair may leave, overall
#   and per type, for each weight (script 1 checks 5% and 10%).
REPAIR_INFO_TOL_TOTAL = 0.03
REPAIR_INFO_TOL_TYPE = 0.08

# REPAIR_MIP_GAP / REPAIR_TIME_LIMIT
#   The solver stops once the pairs kept are within REPAIR_MIP_GAP of the
#   most that could be kept (0.02: within about 7 of 360). This stop is
#   deterministic, so the result does not depend on the computer's speed;
#   proving the exact maximum often takes minutes and gains a pair or two.
#   REPAIR_TIME_LIMIT (seconds per solve) is a safety net: a solve stopped
#   by it keeps the best repair found so far, and the report counts such
#   solves, because where they stop can depend on the computer's speed.
REPAIR_MIP_GAP = 0.02
REPAIR_TIME_LIMIT = 60

# WIDE_MIN_PASSING
#   When fewer candidates than this pass at a size, the failed ones are
#   repaired again with the scenes within each type x bin free to change.
WIDE_MIN_PASSING = 5

# WIDE_MAX_PER_CELL
#   The wide repair offers the solver every pair of the draw and up to this
#   many other pairs per scene x type x bin cell (drawn at random with the
#   candidate's own stream), so it finishes in about a minute; None = every
#   pair (much slower, and then often stopped by the time limit).
WIDE_MAX_PER_CELL = 12

# SELECTION_ROUND
#   Values are rounded to this many decimals before ranking, so a tie is a
#   tie rather than a difference in the last bit of floating point.
SELECTION_ROUND = 6

# --- Trials added to each block (v5's) ---
# CONSISTENCY_RATE
#   Consistency repeats per analysed trial: v4's 15 per 180 (1 in 12). The
#   count per block is floor(block size * rate): 12 at 150, 15 at 180.
CONSISTENCY_RATE = 1 / 12

# PRACTICE_BY_BLOCK / PRACTICE_TYPES / PRACTICE_MAG_QUANTILE
#   Practice trials before the first and the second block, taking the types
#   in turn; v4's types and magnitude cap (conflict is never practised).
PRACTICE_BY_BLOCK = (6, 3)
PRACTICE_TYPES = ("sat_only", "fog_only", "together")
PRACTICE_MAG_QUANTILE = 0.60

# CATCH_PER_BLOCK / CATCH_MODE / CATCH_POOL_TYPE
#   Catch rows per set. EXCLUSION RULE (recorded in set_info.json): exclude
#   a participant with CATCH_EXCLUDE_AT or more failures over the session's
#   2 x CATCH_PER_BLOCK catch trials; failures are also recorded per block.
CATCH_PER_BLOCK = 5
CATCH_MODE = "no_images"
CATCH_POOL_TYPE = "together"
CATCH_EXCLUDE_AT = 3

# --- Images ---
IMAGE_PX = 512                  # every shipped image must be this size
FEATURE_TARGET_SIZE = (256, 256)  # v4's measurement convention; keep
# REMEASURE_TOL
#   Largest allowed |re-measured - table| for sat_mean and contrast_local.
#   The images are copied unchanged; anything visible means the wrong file.
REMEASURE_TOL = 1e-4

# --- Thresholds ---
N_THRESHOLD_PEOPLE = 20_000
THRESHOLD_ALPHA = 0.025

# --- Reported (v4's interpretation of the planning weights) ---
V4_CHOICE_PROB = (0.6191, 0.6104)
PLANNING_WEIGHTS = (0.5, 0.7)
INFORMATIVE_SD = 0.20
SET_THRESHOLD_TOL = 0.05

OUTPUT_ROOT = "data/trials_v5b"
ON_EXISTING = "new"
RANDOM_SEED = 42


# ============================================================
# Imports
# ============================================================

import glob
import hashlib
import importlib.util
import json
import multiprocessing as mp
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

W_NAMES = ("sat", "con")
MAIN_TYPES = ["sat_only", "fog_only", "together", "conflict"]
SCRIPT_NAME = "build_trials_v5b.py"


def load_module(path, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Run this script from the project folder, "
            "where the v4 and v5b scripts are.")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def features_of(path, target_size=FEATURE_TARGET_SIZE):
    """sat_mean and contrast_local of a file. VERBATIM from v4
    (build_trials_v4_corrected._features_of, freeze_trial_list_v4.features_of):
    resize to target_size, mean HSV saturation, SD of the Laplacian of the
    grey image."""
    import cv2
    from PIL import Image
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB").resize(target_size,
                                                  Image.BILINEAR),
                         dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    S = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)[:, :, 1]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(S.mean()), float(cv2.Laplacian(g, cv2.CV_32F).std())


def image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def resolve_run_dir(pattern):
    """A folder, or a glob pattern that matches exactly one folder.
    Returns (path or None, reason)."""
    if not pattern:
        return None, "not set"
    hits = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if len(hits) == 1:
        return hits[0].replace(os.sep, "/"), ""
    if not hits:
        return None, f"no folder matches {pattern}"
    return None, (f"{len(hits)} folders match {pattern}; set the one to use "
                  f"({', '.join(os.path.basename(h) for h in hits[:4])}"
                  f"{', ...' if len(hits) > 4 else ''})")


def n_workers():
    if N_WORKERS is None:
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(N_WORKERS))


# ============================================================
# Settings, output folder
# ============================================================

def settings_dict(cap, size):
    own = {k: v for k, v in globals().items()
           if k.isupper() and not k.startswith("_")
           and k not in ("W_NAMES", "MAIN_TYPES", "SCRIPT_NAME",
                         "KEEP_COLS")}
    own["BLOCK_SIZE"] = size
    script1 = {k: getattr(cap, k) for k in sorted(dir(cap))
               if k.isupper() and not k.startswith("_")
               and k not in ("STIMULUS_SET", "BIN_EDGES")}
    return {"script3": own, "script1_in_force": script1}


def check_settings(cap):
    if THRESHOLD_SET not in cap.THRESHOLD_SETS:
        raise ValueError(f"THRESHOLD_SET {THRESHOLD_SET!r} is not in script "
                         "1's THRESHOLD_SETS")
    if set(MIX_SHARES) != set(cap.TYPES) or abs(sum(MIX_SHARES.values())
                                                 - 1) > 1e-9:
        raise ValueError(f"MIX_SHARES must name {cap.TYPES} and sum to 1")
    if MIX in cap.TYPE_MIXES and any(
            abs(cap.TYPE_MIXES[MIX][t] - MIX_SHARES[t]) > 1e-12
            for t in cap.TYPES):
        raise ValueError(f"script 1 defines mix {MIX!r} with other shares")
    if cap.DISJOINT != "pair" or cap.SCENE_MATCH != "scene_bin":
        raise ValueError("The repair keeps each scene x type x bin cell, so "
                         "it needs script 1's DISJOINT = 'pair' and "
                         "SCENE_MATCH = 'scene_bin'.")
    if CATCH_MODE != "no_images":
        raise ValueError("Only CATCH_MODE 'no_images' is implemented by the "
                         "experiment.")
    if len(PRACTICE_BY_BLOCK) != 2:
        raise ValueError("PRACTICE_BY_BLOCK needs two counts")
    if not 0 < REPAIR_TARGET <= LEAK_LIMIT:
        raise ValueError("REPAIR_TARGET must be in (0, LEAK_LIMIT]")
    if not 0 < REPAIR_TIGHTEN < 1 or REPAIR_TRIES < 1:
        raise ValueError("REPAIR_TIGHTEN must be in (0, 1) and REPAIR_TRIES "
                         ">= 1")
    for t, v in TYPE_LIMITS.items():
        if (t not in MAIN_TYPES or len(v) != 2 or v[0] not in W_NAMES
                or not v[1] > 0):
            raise ValueError(f"TYPE_LIMITS[{t!r}] must be (term 'sat' or "
                             "'con', limit > 0) for a trial type")
    for name, ts in LIMIT_STEPS:
        if not set(ts) <= set(TYPE_LIMITS):
            raise ValueError(f"LIMIT_STEPS {name!r} names types not in "
                             "TYPE_LIMITS")
    for k in CAPACITY_OVERRIDES:
        if not hasattr(cap, k):
            raise ValueError(f"CAPACITY_OVERRIDES names {k!r}, which script "
                             "1 does not have")


def resolve_output_dir(cap, size):
    """The hash leaves out N_WORKERS, which cannot change the result."""
    s = settings_dict(cap, size)
    s["script3"].pop("N_WORKERS", None)
    blob = json.dumps(s, sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode()).hexdigest()[:8]
    name = (f"v5b_n{size}_{MIX}_{THRESHOLD_SET}_cand{N_CANDIDATES}"
            f"_leak{LEAK_LIMIT:g}_seed{RANDOM_SEED}__s{h}")
    base = f"{OUTPUT_ROOT}/{name}"
    if not os.path.exists(base):
        return base
    if ON_EXISTING == "error":
        raise FileExistsError(f"{base} exists")
    n = 2
    while os.path.exists(f"{base}_{n}"):
        n += 1
    print(f"  NOTE: {base} exists; writing to {base}_{n} instead.")
    return f"{base}_{n}"


# ============================================================
# Candidate measures (draw and repaired sets alike)
# ============================================================

def analysed(s):
    return s[s.role == "analysed"]


def worst_b(cap, met, defs=None):
    """Largest |b_sat| or |b_con| over both sets and the given definitions
    (default: all)."""
    defs = defs or cap.NORMALITY_DEFS
    return max(abs(met[f"leak_{d}_{k}_{b}"]) for d in defs
               for k in ("b_sat", "b_con") for b in ("A", "B"))


def position_metrics(cap, m):
    """Where the images of the trials in m sit relative to the scene's
    unedited photograph, by measurement: the share of images above, at
    (within STRADDLE_TOL_SD) and below the original on each feature, for
    image A (the predicted image), image B and both; the pairs that
    straddle the original; the pairs with the unedited image."""
    tol = cap.STRADDLE_TOL_SD
    out = {}
    for feat, col in (("sat", "change_sat"), ("con", "change_con")):
        v = {"A": m[f"{col}_a"].to_numpy(float),
             "B": m[f"{col}_b"].to_numpy(float)}
        v["all"] = np.r_[v["A"], v["B"]]
        for side, x in v.items():
            n = max(len(x), 1)
            out[f"pos_{feat}_{side}_above"] = float((x > tol).sum() / n)
            out[f"pos_{feat}_{side}_at"] = float((np.abs(x) <= tol).sum() / n)
            out[f"pos_{feat}_{side}_below"] = float((x < -tol).sum() / n)
    out["straddle_sat"] = int(m.straddle_colour_measured.astype(bool).sum())
    out["straddle_con"] = int(m.straddle_contrast_measured.astype(bool)
                              .sum())
    out["with_original"] = int(m.includes_original.astype(bool).sum())
    return out


TYPE_TERMS = {"sat_only": ("sat",), "fog_only": ("con",)}


def type_b(cap, sets, types=None):
    """The single-term leak of each limited type: N of the type's analysed
    trials on its TYPE_LIMITS term, no intercept, per definition and set."""
    out = {}
    for t in (types if types is not None else TYPE_LIMITS):
        term = TYPE_LIMITS[t][0]
        for blk in ("A", "B"):
            m = analysed(sets[blk])
            m = m[m.trial_type == t]
            for dfn in cap.NORMALITY_DEFS:
                lk = cap.leak(m[f"N_{dfn}"], m.x_sat, m.x_con, (term,))
                out[f"type_b_{t}_{dfn}_{blk}"] = lk[f"b_{term}"]
    return out


def type_worst(out, t):
    return max(abs(v) for k, v in out.items()
               if k.startswith(f"type_b_{t}_"))


def leak_rows_of(cap, m, base, label):
    """The leak of the trials in m, by trial type and overall, under each
    definition. One term for sat_only (x_sat) and fog_only (x_con), where
    the other term is ~0; both for the mixed types and overall (script 1's
    convention)."""
    rows = []
    groups = ([(t, m[m.trial_type == t], TYPE_TERMS.get(t, ("sat", "con")))
               for t in MAIN_TYPES if (m.trial_type == t).any()]
              + [("all", m, ("sat", "con"))])
    # the single-term fits of limited types not already fitted that way
    groups += [(t, m[m.trial_type == t], (TYPE_LIMITS[t][0],))
               for t in TYPE_LIMITS if (m.trial_type == t).any()
               and TYPE_TERMS.get(t) != (TYPE_LIMITS[t][0],)]
    for t, s, terms in groups:
        for dfn in cap.NORMALITY_DEFS:
            r = {**base, "set": label, "type": t, "definition": dfn,
                 "n": len(s), "terms": "+".join(terms)}
            r.update(cap.leak(s[f"N_{dfn}"], s.x_sat, s.x_con, terms))
            xs_, xc_ = s.x_sat.to_numpy(float), s.x_con.to_numpy(float)
            den = np.sqrt((xs_ ** 2).sum() * (xc_ ** 2).sum())
            r["cos_x"] = float((xs_ * xc_).sum() / den) if den > 0 else np.nan
            rows.append(r)
    return rows


def summarise(cap, met, chk, sets):
    """One candidate's values, before or after repair."""
    failed = [k for k, v in chk.items() if v is False]
    out = {"pass": not failed, "failed_checks": "; ".join(failed)}
    for blk in ("A", "B"):
        for w in W_NAMES:
            out[f"info_{w}_{blk}"] = met[f"info_{w}_{blk}"]
    for w in W_NAMES:
        a, b = out[f"info_{w}_A"], out[f"info_{w}_B"]
        out[f"weaker_set_info_{w}"] = min(a, b)
        out[f"total_info_{w}"] = a + b
        out[f"ab_rel_diff_info_{w}"] = met[f"info_{w}_rel_diff"]
    out["info_type_worst_rel_diff"] = met["info_type_worst_rel_diff"]
    for d in cap.NORMALITY_DEFS:
        for k in ("b_sat", "b_con", "cos_sat", "cos_con"):
            for blk in ("A", "B"):
                out[f"leak_{d}_{k}_{blk}"] = met[f"leak_{d}_{k}_{blk}"]
        out[f"worst_abs_b_{d}"] = worst_b(cap, met, (d,))
    out["worst_abs_b"] = worst_b(cap, met)
    tb = type_b(cap, sets)
    out.update(tb)
    for t in TYPE_LIMITS:
        out[f"type_worst_{t}"] = type_worst(tb, t)
    out["ks_sat"], out["ks_con"] = met["ks_sat"], met["ks_con"]
    out["cos_diff"] = met["cos_diff"]
    out["vif_max"] = max(met["vif_A"], met["vif_B"])
    out["scenes_A"], out["scenes_B"] = met["scenes_A"], met["scenes_B"]
    m = pd.concat([analysed(sets["A"]), analysed(sets["B"])])
    out.update(position_metrics(cap, m))
    return out


def prefixed(p, d):
    return {f"{p}{k}": v for k, v in d.items()}


# ============================================================
# The repair: an integer program over the unused pairs
# ============================================================

class _Rows:
    """Sparse constraint rows lo <= a . y <= hi, built row by row."""

    def __init__(self, ncol):
        self.ncol = ncol
        self.r, self.c, self.v, self.lo, self.hi = [], [], [], [], []

    def add(self, cols, vals, lo, hi):
        cols = np.asarray(cols, dtype=np.int64)
        vals = np.broadcast_to(np.asarray(vals, float), cols.shape)
        nz = vals != 0
        self.r.append(np.full(int(nz.sum()), len(self.lo), dtype=np.int64))
        self.c.append(cols[nz])
        self.v.append(vals[nz])
        self.lo.append(lo)
        self.hi.append(hi)

    def constraint(self):
        from scipy import sparse
        from scipy.optimize import LinearConstraint
        A = sparse.csr_matrix(
            (np.concatenate(self.v), (np.concatenate(self.r),
                                      np.concatenate(self.c))),
            shape=(len(self.lo), self.ncol))
        return LinearConstraint(A, np.array(self.lo, float),
                                np.array(self.hi, float))


def solve_repair(cap, Q, codes, need, inA, inB, target, mode, types=(),
                 shrink=1.0):
    """The integer program. Variables yA, yB (one per usable pair): is the
    pair in set A / set B. Maximise the draw's pairs kept, subject to
      cells  each scene x type x bin cell holds the draw's count in each set
             (mode 'cells'); or each type x bin its quota, each cell the
             same count in A and B, and <= MAX_PER_SCENE_PER_TYPE per scene
             and type (mode 'wide');
      leak   |sum y x N| <= target * sum y x^2, per term, definition, set;
             and for each type in `types`, over that type's trials and its
             one TYPE_LIMITS term, |sum y x N| <= limit * TYPE_TARGET_SHARE
             * shrink * sum y x^2 (exact for a one-term fit);
      info   A-B relative difference <= REPAIR_INFO_TOL_TOTAL per weight,
             and <= REPAIR_INFO_TOL_TYPE per type and weight;
      disjoint  yA + yB <= 1."""
    from scipy import sparse
    from scipy.optimize import Bounds, LinearConstraint, milp
    n = len(Q)
    R = _Rows(2 * n)
    idx_all = np.arange(n)
    if mode == "cells":
        for code, idx in codes["cell"].items():
            kA, kB = need["cellA"].get(code, 0), need["cellB"].get(code, 0)
            R.add(idx, 1.0, kA, kA)
            R.add(n + idx, 1.0, kB, kB)
    else:
        for code, idx in codes["tb"].items():
            q = need["tb"].get(code, 0)
            R.add(idx, 1.0, q, q)
            R.add(n + idx, 1.0, q, q)
        for code, idx in codes["cell"].items():
            R.add(np.r_[idx, n + idx],
                  np.r_[np.ones(len(idx)), -np.ones(len(idx))], 0, 0)
        for code, idx in codes["ts"].items():
            R.add(idx, 1.0, 0, cap.MAX_PER_SCENE_PER_TYPE)
            R.add(n + idx, 1.0, 0, cap.MAX_PER_SCENE_PER_TYPE)
    xs, xc = Q.x_sat.to_numpy(float), Q.x_con.to_numpy(float)
    for dfn in cap.NORMALITY_DEFS:
        N = Q[f"N_{dfn}"].to_numpy(float)
        for x in (xs, xc):
            g, m2 = x * N, x * x
            for off in (0, n):
                R.add(off + idx_all, g - target * m2, -np.inf, 0)
                R.add(off + idx_all, g + target * m2, 0, np.inf)

    tt_ = Q.trial_type.to_numpy()
    for t in types:
        term, lim = TYPE_LIMITS[t]
        tgt = lim * TYPE_TARGET_SHARE * shrink
        mt = (tt_ == t).astype(float)
        x = (xs if term == "sat" else xc) * mt
        for dfn in cap.NORMALITY_DEFS:
            N = Q[f"N_{dfn}"].to_numpy(float)
            g, m2 = x * N, x * x
            for off in (0, n):
                R.add(off + idx_all, g - tgt * m2, -np.inf, 0)
                R.add(off + idx_all, g + tgt * m2, 0, np.inf)

    def info_rows(q, tol):
        cols = np.r_[idx_all, n + idx_all]
        R.add(cols, np.r_[q * (1 - tol / 2), -q * (1 + tol / 2)], -np.inf, 0)
        R.add(cols, np.r_[-q * (1 + tol / 2), q * (1 - tol / 2)], -np.inf, 0)

    for x in (xs, xc):
        info_rows(x * x, REPAIR_INFO_TOL_TOTAL)
    tt = Q.trial_type.to_numpy()
    for t, ws in (("sat_only", [xs]), ("fog_only", [xc]),
                  ("together", [xs, xc]), ("conflict", [xs, xc])):
        mt = (tt == t).astype(float)
        if mt.any():
            for x in ws:
                info_rows(x * x * mt, REPAIR_INFO_TOL_TYPE)
    D = sparse.hstack([sparse.identity(n, format="csr"),
                       sparse.identity(n, format="csr")], format="csr")
    c = -np.r_[inA, inB].astype(float)
    with warnings.catch_warnings():
        # "threads" is passed to HiGHS verbatim; scipy warns that it does
        warnings.simplefilter("ignore", RuntimeWarning)
        res = milp(c, constraints=[R.constraint(),
                                   LinearConstraint(D, -np.inf, 1.0)],
                   integrality=np.ones(2 * n), bounds=Bounds(0, 1),
                   options={"time_limit": REPAIR_TIME_LIMIT,
                            "mip_rel_gap": REPAIR_MIP_GAP,
                            "threads": 1, "disp": False})
    return res


def rebuild(cap, Q, yA, yB, draw_sets, st):
    """The repaired sets: the chosen pairs, coupled within each scene x type
    x bin cell in order of the type's driver (the i-th smallest in A with
    the i-th smallest in B), sides balanced per type as in script 1, and
    the draw's reserved practice and catch rows unchanged."""
    out = {}
    sel = {"A": Q[yA], "B": Q[yB]}
    parts = {"A": [], "B": []}
    for t in cap.TYPE_ORDER:
        fr = {}
        for blk in ("A", "B"):
            s = sel[blk][sel[blk].trial_type == t].copy()
            s["_drv"] = np.where(t == "sat_only", s.x_sat.abs(),
                                 s.x_con.abs())
            fr[blk] = s.sort_values(["scene_id", "mag_bin", "_drv", "file_a",
                                     "file_b"], kind="stable")
        ca = list(zip(fr["A"].scene_id, fr["A"].mag_bin))
        cb = list(zip(fr["B"].scene_id, fr["B"].mag_bin))
        if ca != cb:
            raise RuntimeError(f"repair: the {t} cells differ between the "
                               "sets")
        for blk in ("A", "B"):
            parts[blk].append(fr[blk].drop(columns="_drv").assign(
                role="analysed", trial_type=t,
                couple=np.arange(len(fr[blk]))))
    for blk in ("A", "B"):
        main = pd.concat(parts[blk], ignore_index=True)
        main["a_is_left"] = False
        for t in cap.TYPES:
            m = (main.trial_type == t).to_numpy()
            main.loc[m, "a_is_left"] = cap.balanced_sides(
                int(m.sum()), st(f"repair_sides_{blk}_{t}"))
        res = draw_sets[blk][draw_sets[blk].role != "analysed"]
        s = pd.concat([main, res], ignore_index=True)
        s["block_set"] = blk
        out[blk] = s
    return out


def _kid(df):
    return (df.file_a.astype(str) + "|" + df.file_b.astype(str)).to_numpy()


def repair(cap, sets, counts, s1, n_allowed, st, mode, types=()):
    W = _W
    P = W["P_all"]
    mA, mB = analysed(sets["A"]), analysed(sets["B"])
    res_rows = pd.concat([s[s.role != "analysed"] for s in sets.values()])
    reserve = set(_kid(res_rows))

    def cell_of(m):
        return (m.trial_type + "|" + m.scene_id + "|"
                + m.mag_bin.astype(str)).to_numpy()

    need = {"cellA": pd.Series(cell_of(mA)).value_counts().to_dict(),
            "cellB": pd.Series(cell_of(mB)).value_counts().to_dict(),
            "tb": {f"{t}|{b}": int(q) for t in cap.TYPES
                   for b, q in s1["types"][t]["quotas"].items() if q > 0}}
    keep = ~np.isin(W["kid"], list(reserve))
    if mode == "cells":
        keep &= np.isin(W["cell"], list(set(need["cellA"])
                                         | set(need["cellB"])))
    else:
        keep &= np.isin(W["tb"], list(need["tb"]))
        if WIDE_MAX_PER_CELL:
            # every pair of the draw, and up to WIDE_MAX_PER_CELL others
            # per scene x type x bin cell, chosen at random (named stream)
            drawn = np.isin(W["kid"], list(_kid(mA)) + list(_kid(mB)))
            rng = st("wide_sample")
            pos = np.flatnonzero(keep & ~drawn)
            cells = W["cell"][pos]
            take = np.zeros(len(P), dtype=bool)
            for c, ix in pd.Series(cells).groupby(cells, sort=True) \
                    .indices.items():
                ix = pos[np.asarray(ix)]
                if len(ix) > WIDE_MAX_PER_CELL:
                    ix = np.sort(rng.choice(ix, WIDE_MAX_PER_CELL,
                                            replace=False))
                take[ix] = True
            keep &= drawn | take
    Q = P[keep].reset_index(drop=True)
    codes = {}
    for name in ("cell", "tb", "ts"):
        arr = W[name][keep]
        codes[name] = {c: np.asarray(v) for c, v in
                       pd.Series(arr).groupby(arr, sort=True).indices.items()}
    kid = W["kid"][keep]
    inA = np.isin(kid, list(_kid(mA))).astype(float)
    inB = np.isin(kid, list(_kid(mB))).astype(float)
    info = {"repair_mode": mode, "repair_vars": 2 * len(Q),
            "repair_solves": 0, "repair_seconds": 0.0,
            "repair_time_limit_hits": 0, "repair_target": np.nan,
            "repair_status": "", "kept_A": np.nan, "kept_B": np.nan}
    target, shrink = REPAIR_TARGET, 1.0
    new, met, chk, ok = None, None, None, False
    for k in range(REPAIR_TRIES):
        t1 = time.time()
        res = solve_repair(cap, Q, codes, need, inA, inB, target, mode,
                           types, shrink)
        info["repair_solves"] += 1
        info["repair_seconds"] += time.time() - t1
        info["repair_time_limit_hits"] += int(res.status == 1)
        info["repair_target"] = target
        if res.x is None:
            info["repair_status"] = (f"no repair found (solver status "
                                     f"{res.status}: {res.message[:60]})")
            break
        y = res.x > 0.5
        yA, yB = y[:len(Q)], y[len(Q):]
        new = rebuild(cap, Q, yA, yB, sets, st)
        met, chk = cap.evaluate(new, counts, s1, n_allowed)
        info["kept_A"] = int((yA & (inA > 0)).sum())
        info["kept_B"] = int((yB & (inB > 0)).sum())
        failed = [c for c, v in chk.items() if v is False]
        wb = worst_b(cap, met)
        tb = type_b(cap, new, types)
        t_bad = [t for t in types
                 if type_worst(tb, t) > TYPE_LIMITS[t][1] + cap.EPS]
        lim_ok = wb <= LEAK_LIMIT + cap.EPS and not t_bad
        if lim_ok and not failed:
            ok = True
            info["repair_status"] = "ok"
            break
        if not lim_ok and k + 1 < REPAIR_TRIES:
            if wb > LEAK_LIMIT + cap.EPS:
                target *= REPAIR_TIGHTEN
            if t_bad:
                shrink *= REPAIR_TIGHTEN
            continue
        info["repair_status"] = (
            (f"limit missed (worst |b| {wb:.3f}"
             + (f"; types {t_bad}" if t_bad else "") + ")")
            if not lim_ok else "script 1 checks fail: " + "; ".join(failed))
        break
    return {"sets": new, "met": met, "chk": chk, "pass": ok, "info": info}


# ============================================================
# Workers (one process each; also used in this process when N_WORKERS = 1)
# ============================================================

_W = {}


def _one_thread(milp):
    """scipy's milp with HiGHS limited to one thread. HiGHS keeps one
    thread pool per process, set by its first solve; a later solve that asks
    for a different number of threads fails (no solution). So every solve
    in every process, script 1's included, asks for one thread."""
    if getattr(milp, "_one_thread", False):
        return milp

    def wrapped(*args, **kw):
        opts = dict(kw.pop("options", None) or {})
        opts["threads"] = 1
        with warnings.catch_warnings():
            # "threads" is passed to HiGHS verbatim; scipy warns that it does
            warnings.simplefilter("ignore", RuntimeWarning)
            return milp(*args, options=opts, **kw)

    wrapped._one_thread = True
    return wrapped


def apply_overrides(cap):
    for k, v in CAPACITY_OVERRIDES.items():
        setattr(cap, k, v)
    if hasattr(cap, "milp"):
        cap.milp = _one_thread(cap.milp)
    cap.MANIFEST_CSV = MANIFEST_CSV
    cap.TYPE_MIXES = {**cap.TYPE_MIXES, MIX: dict(MIX_SHARES)}


def _setup_worker(cap, payload):
    _W.clear()
    _W.update(payload)
    _W["cap"] = cap
    P = pd.concat([payload["pools_t"][t].assign(trial_type=t)
                   for t in cap.TYPES], ignore_index=True)
    _W["P_all"] = P
    _W["kid"] = _kid(P)
    tb = (P.trial_type + "|" + P.mag_bin.astype(str)).to_numpy()
    _W["tb"] = tb
    _W["cell"] = (P.trial_type + "|" + P.scene_id + "|"
                  + P.mag_bin.astype(str)).to_numpy()
    _W["ts"] = (P.trial_type + "|" + P.scene_id).to_numpy()


def _worker_init(payload):
    cap = load_module(CAPACITY_SCRIPT, "analyse_pair_capacity_v5b")
    apply_overrides(cap)
    _setup_worker(cap, payload)


def _task(args):
    """Draw candidate d at one size with script 1's one_draw, check it, and
    repair it. Returns a dict of values and both versions of the sets."""
    size, d, mode, types = args
    cap = _W["cap"]
    t0 = time.time()
    counts, s1 = _W["counts"][size], _W["s1"][size]
    combo = f"{THRESHOLD_SET}|{size}|{MIX}"     # script 1's and 2's seeds
    out = {"size": size, "candidate": d, "step": "+".join(types) or "overall"}
    sets, how, n_allowed = cap.one_draw(_W["pools_t"], counts, s1,
                                        _W["scenes"], combo, d)
    out["built"] = how
    if sets is None:
        out.update({"draw_pass": False, "rep_pass": False,
                    "repair_mode": mode, "repair_status": "draw not built",
                    "seconds": time.time() - t0, "leak_rows": [],
                    "sets_draw": None, "sets_rep": None})
        return out
    met0, chk0 = cap.evaluate(sets, counts, s1, n_allowed)
    out.update(prefixed("draw_", summarise(cap, met0, chk0, sets)))
    base = {"size": size, "candidate": d}
    rows = []
    for blk in ("A", "B"):
        rows += leak_rows_of(cap, analysed(sets[blk]),
                             {**base, "stage": "before"}, blk)
    st = cap.Streams(cap.RANDOM_SEED, "build_trials_v5b", combo, d, mode)
    rep = repair(cap, sets, counts, s1, n_allowed, st, mode, types)
    out.update(rep["info"])
    if rep["sets"] is not None:
        out.update(prefixed("rep_", summarise(cap, rep["met"], rep["chk"],
                                              rep["sets"])))
        for blk in ("A", "B"):
            rows += leak_rows_of(cap, analysed(rep["sets"][blk]),
                                 {**base, "stage": "after"}, blk)
    out["rep_pass"] = rep["pass"]
    out["leak_rows"] = rows
    out["sets_draw"] = sets
    out["sets_rep"] = rep["sets"]
    out["seconds"] = time.time() - t0
    return out


def run_tasks(tasks, payload, cap, nw, label):
    """Run the tasks in nw processes (or here when nw == 1) and return the
    results in task order."""
    got = {}
    t0 = time.time()

    def note(k, t, r):
        print(f"    {label} {k}/{len(tasks)}  n={t[0]} candidate {t[1]:2d} "
              f"[{'+'.join(t[3]) or 'overall'}]: "
              f"{r.get('repair_status', '')}  ({time.time() - t0:.0f} s)",
              flush=True)

    if nw == 1:
        _setup_worker(cap, payload)
        for k, t in enumerate(tasks, 1):
            got[t] = _task(t)
            note(k, t, got[t])
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=nw, mp_context=ctx,
                                 initializer=_worker_init,
                                 initargs=(payload,)) as ex:
            futs = {ex.submit(_task, t): t for t in tasks}
            for k, f in enumerate(as_completed(futs), 1):
                t = futs[f]
                got[t] = f.result()
                note(k, t, got[t])
    return [got[t] for t in tasks]


# ============================================================
# Selection
# ============================================================

def select(cand):
    """For each weight, the weaker set's information relative to the mean
    of the unrepaired candidates that pass script 1's checks; score = the
    smaller relative value. Ranked among passing candidates by score (high
    first), then the worst |b| (small first), then the candidate number.
    Done for the repaired sets ('rep') and the unrepaired ('draw')."""
    cand = cand.copy()
    base = cand[cand.draw_pass.fillna(False).astype(bool)]
    if not len(base):
        base = cand[cand.built.isin(["greedy", "fallback"])]
    ref = {w: float(base[f"draw_weaker_set_info_{w}"].mean())
           for w in W_NAMES}
    chosen = {}
    for stg in ("draw", "rep"):
        for w in W_NAMES:
            col = f"{stg}_weaker_set_info_{w}"
            cand[f"{stg}_rel_{w}"] = (cand[col] / ref[w] if col in cand
                                      else np.nan)
        cand[f"{stg}_score"] = cand[[f"{stg}_rel_sat",
                                     f"{stg}_rel_con"]].min(axis=1)
        ok = cand[cand[f"{stg}_pass"].fillna(False).astype(bool)].copy()
        cand[f"{stg}_rank"] = np.nan
        cand[f"{stg}_chosen"] = False
        if not len(ok):
            chosen[stg] = None
            continue
        ok["_s"] = -ok[f"{stg}_score"].round(SELECTION_ROUND)
        ok["_b"] = ok[f"{stg}_worst_abs_b"].round(SELECTION_ROUND)
        ok = ok.sort_values(["_s", "_b", "candidate"], kind="stable")
        rank = {c: i + 1 for i, c in enumerate(ok.candidate)}
        cand[f"{stg}_rank"] = cand.candidate.map(rank)
        chosen[stg] = int(ok.candidate.iloc[0])
        cand[f"{stg}_chosen"] = cand.candidate == chosen[stg]
    return chosen, cand, ref


# ============================================================
# Rows
# ============================================================

# Every list row (analysed, consistency, practice, catch) carries these,
# where the pair has them: the images' level columns and edit parameters,
# their measured features and changes from the original (corpus SD), the
# images' z values, the pair's differences, direction terms and type, the
# straddle and unedited-image flags, the distances from typical and the
# normality terms (combined and per feature), and the magnitude bin.
KEEP_COLS = [
    "scene_id", "category", "group", "indoor", "type",
    "file_a", "file_b",
    "contrast_level_a", "colour_level_a", "contrast_level_b",
    "colour_level_b", "contrast_step_a", "colour_step_a", "contrast_step_b",
    "colour_step_b", "is_original_a", "is_original_b",
    "contrast_param_a", "comp_k_a", "colour_k_a",
    "contrast_param_b", "comp_k_b", "colour_k_b",
    "sat_a", "sat_b", "fog_a", "fog_b",
    "change_sat_a", "change_con_a", "change_sat_b", "change_con_b",
    "z_sat_a", "z_con_a", "z_sat_b", "z_con_b",
    "d_sat", "d_fog", "pred_sat", "pred_fog", "z_sat", "z_fog",
    "balance", "balanced", "d_contrast_steps", "d_colour_steps",
    "straddle_contrast_measured", "straddle_colour_measured",
    "includes_original",
    "dist_corpus_a", "dist_corpus_b", "dist_orig_a", "dist_orig_b",
    "N_corpus", "N_corpus_sat", "N_corpus_con",
    "N_orig", "N_orig_sat", "N_orig_con",
    "mag_bin", "driver_sd", "key"]


def with_sides(df, a_is_left):
    df = df.copy()
    df["a_is_left"] = np.asarray(a_is_left, dtype=bool)
    df["left_file"] = np.where(df.a_is_left, df.file_a, df.file_b)
    df["right_file"] = np.where(df.a_is_left, df.file_b, df.file_a)
    df["left_path"] = "stimuli/" + df.left_file
    df["right_path"] = "stimuli/" + df.right_file
    return df


def main_rows(sets, blk):
    m = sets[blk][sets[blk].role == "analysed"].copy()
    keep = [c for c in KEEP_COLS if c in m.columns]
    out = m[keep + ["trial_type", "couple", "a_is_left"]].copy()
    out["couple"] = out.couple.astype(int)
    out["mag_driver"] = np.where(out.trial_type == "sat_only", "d_sat",
                                 "d_fog")
    out["analysed"] = True
    out["catch_mode"] = None
    out["repeat_of"] = None
    order = {t: i for i, t in enumerate(MAIN_TYPES)}
    out["_o"] = out.trial_type.map(order)
    out = out.sort_values(["_o", "couple"], kind="stable").drop(columns="_o")
    out = with_sides(out, out.a_is_left)
    out.insert(0, "item_id", [f"{blk}{i + 1:03d}" for i in range(len(out))])
    out.insert(1, "set", blk)
    return out.reset_index(drop=True)


def consistency_rows(mainA, mainB, n_cons, rng):
    """Stratified by type (largest remainder over the type counts); the
    SAME couples in both sets, so the repeats are matched. Each repeat is
    the original pair with its sides swapped. (v5, unchanged)"""
    counts = mainA.trial_type.value_counts().to_dict()
    total = sum(counts.values())
    share = {t: counts.get(t, 0) / total for t in MAIN_TYPES}
    alloc = {t: int(np.floor(n_cons * share[t])) for t in MAIN_TYPES}
    left = n_cons - sum(alloc.values())
    rema = sorted(MAIN_TYPES, key=lambda t: (-(n_cons * share[t] - alloc[t]),
                                             MAIN_TYPES.index(t)))
    for t in rema[:left]:
        alloc[t] += 1
    out = {}
    picks = []
    for t in MAIN_TYPES:
        couples = sorted(mainA.loc[mainA.trial_type == t, "couple"])
        if alloc[t]:
            picks += [(t, c) for c in rng.choice(couples, alloc[t],
                                                 replace=False)]
    for blk, main in (("A", mainA), ("B", mainB)):
        rows = []
        for k, (t, c) in enumerate(picks):
            orig = main[(main.trial_type == t) & (main.couple == c)].iloc[0]
            r = orig.copy()
            r["item_id"] = f"{blk}c{k + 1:02d}"
            r["trial_type"] = "consistency"
            r["analysed"] = False
            r["repeat_of"] = orig.item_id
            r["repeat_type"] = t
            rows.append(r)
        df = pd.DataFrame(rows)
        df = with_sides(df, ~df.a_is_left.astype(bool))
        out[blk] = df.reset_index(drop=True)
    return out, alloc


def practice_rows(pools_t, used, n, label, rng):
    """v4's practice rule: the types in turn, at or below the
    PRACTICE_MAG_QUANTILE of each type's driver, one scene at most once.
    (v5, unchanged)"""
    want = [PRACTICE_TYPES[i % len(PRACTICE_TYPES)] for i in range(n)]
    rows, scenes = [], set()
    for t in want:
        tp = pools_t[t]
        thr = tp.driver_sd.quantile(PRACTICE_MAG_QUANTILE)
        m = ((tp.driver_sd <= thr) & ~tp.scene_id.isin(scenes)
             & ~tp.key.isin(used))
        cand = tp[m]
        if not len(cand):
            raise RuntimeError(f"No {t} pair left for {label} practice")
        r = cand.iloc[int(rng.integers(len(cand)))]
        rows.append(r)
        scenes.add(r.scene_id)
        used.add(r.key)
    df = pd.DataFrame(rows)
    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep].copy()
    df["trial_type"] = "practice"
    df["practice_type"] = want
    df["mag_driver"] = np.where(df.type == "sat_only", "d_sat", "d_fog")
    df["mag_bin"] = 0
    df["analysed"] = False
    df["catch_mode"] = None
    df["repeat_of"] = None
    df = with_sides(df, cap_sides(len(df), rng))
    df.insert(0, "item_id", [f"{label}_{i + 1:02d}" for i in range(len(df))])
    df.insert(1, "set", label)
    return df.reset_index(drop=True)


def catch_rows(pools_t, used, blk, rng):
    """(v5, unchanged)"""
    tp = pools_t[CATCH_POOL_TYPE]
    cand = tp[~tp.key.isin(used)]
    rows, scenes = [], set()
    for j in rng.permutation(len(cand)):
        if len(rows) >= CATCH_PER_BLOCK:
            break
        r = cand.iloc[j]
        if r.scene_id in scenes or r.key in used:
            continue
        rows.append(r)
        scenes.add(r.scene_id)
        used.add(r.key)
    if len(rows) < CATCH_PER_BLOCK:
        raise RuntimeError(f"Only {len(rows)} catch pairs for set {blk}")
    df = pd.DataFrame(rows)
    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep].copy()
    df["trial_type"] = "catch"
    df["mag_driver"] = "d_fog"
    df["mag_bin"] = 0
    df["analysed"] = False
    df["catch_mode"] = CATCH_MODE
    df["repeat_of"] = None
    df = with_sides(df, cap_sides(len(df), rng))
    df.insert(0, "item_id", [f"{blk}x{i + 1:02d}" for i in range(len(df))])
    df.insert(1, "set", blk)
    return df.reset_index(drop=True)


_CAP = None


def cap_sides(n, rng):
    return _CAP.balanced_sides(n, rng)


# ============================================================
# Images
# ============================================================

def read_checksums(ck):
    """The stimulus set's checksums.sha256 (path -> SHA-256), checked
    against its FROZEN.json. None when it cannot be read."""
    if not os.path.isfile(CHECKSUMS_FILE):
        ck.add("images", "the stimulus set's checksums.sha256 is present",
               False, f"{CHECKSUMS_FILE} not found")
        return None
    sums = {}
    with open(CHECKSUMS_FILE) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln:
                h, rel = ln.split("  ", 1)
                sums[rel] = h
    ck.add("images", "the stimulus set's checksums.sha256 is present", True,
           f"{len(sums):,} files listed")
    if os.path.isfile(FROZEN_FILE):
        with open(FROZEN_FILE) as f:
            fr = json.load(f)
        ck.add("images", "checksums.sha256 is the frozen one (SHA-256 as in "
               "FROZEN.json)",
               sha256_of(CHECKSUMS_FILE) == fr.get("checksums_sha256"),
               f"set {fr.get('settings_hash')}")
    else:
        ck.add("images", "checksums.sha256 is the frozen one (SHA-256 as in "
               "FROZEN.json)", False, f"{FROZEN_FILE} not found")
    return sums


def ship_images(rows, out_dir, ck, manifest):
    """Copy, hash and re-measure every referenced image; compare each copy
    with its source and with the stimulus set's checksums.sha256. Returns
    (manifest frame, list of failures)."""
    sums = read_checksums(ck)
    exp = {}
    for r in rows.itertuples():
        exp.setdefault(r.file_a, (r.sat_a, r.fog_a))
        exp.setdefault(r.file_b, (r.sat_b, r.fog_b))
    need = sorted(exp)
    missing = [f for f in need if not os.path.isfile(f"{STIMULI_SRC}/{f}")]
    ck.add("images", "every referenced image exists in STIMULI_SRC",
           not missing, f"{len(need)} needed, {len(missing)} missing"
           + (f": {missing[:5]}" if missing else ""))
    if missing:
        return None, [f"missing: {f}" for f in missing]
    dst = f"{out_dir}/stimuli"
    os.makedirs(dst)
    recs, bad_copy, bad_feat, bad_size, bad_man, bad_sum = ([], [], [], [],
                                                            [], [])
    mi = manifest.set_index("filename") if manifest is not None else None
    for f in need:
        src = f"{STIMULI_SRC}/{f}"
        out = f"{dst}/{f}"
        shutil.copy2(src, out)
        h_src, h_out = sha256_of(src), sha256_of(out)
        if h_src != h_out:
            bad_copy.append(f)
        if sums is not None and sums.get(f"stimuli/{f}") != h_out:
            bad_sum.append(f)
        size = image_size(out)
        if size != (IMAGE_PX, IMAGE_PX):
            bad_size.append(f"{f} {size}")
        s, c = features_of(out)
        es, ec = exp[f]
        if max(abs(s - es), abs(c - ec)) > REMEASURE_TOL:
            bad_feat.append(f"{f} (sat {s:.3f} vs {es:.3f}, contrast "
                            f"{c:.3f} vs {ec:.3f})")
        if mi is not None and f in mi.index:
            if max(abs(s - mi.loc[f, "sat_mean"]),
                   abs(c - mi.loc[f, "contrast_local"])) > REMEASURE_TOL:
                bad_man.append(f)
        recs.append({"filename": f, "sha256": h_out, "sat_mean": s,
                     "contrast_local": c, "table_sat_mean": es,
                     "table_contrast_local": ec, "width": size[0],
                     "height": size[1]})
    ck.add("images", "every copy is byte-identical to its source (SHA-256)",
           not bad_copy, f"{len(bad_copy)} differ" + (
               f": {bad_copy[:5]}" if bad_copy else ""))
    if sums is not None:
        ck.add("images", "every copy matches the stimulus set's "
               "checksums.sha256", not bad_sum, f"{len(bad_sum)} of "
               f"{len(need)} differ or are not listed" + (
                   f": {bad_sum[:5]}" if bad_sum else ""))
    ck.add("images", f"every image is {IMAGE_PX} x {IMAGE_PX}", not bad_size,
           f"{len(bad_size)} differ" + (f": {bad_size[:5]}"
                                        if bad_size else ""))
    ck.add("images", f"re-measured sat_mean and contrast_local equal the "
           f"pair table (|diff| <= {REMEASURE_TOL:g})", not bad_feat,
           f"{len(bad_feat)} of {len(need)} differ" + (
               f": {bad_feat[:3]}" if bad_feat else ""))
    if mi is not None:
        ck.add("images", "re-measured features equal the stimulus manifest",
               not bad_man, f"{len(bad_man)} differ")
    fails = ([f"copy differs: {f}" for f in bad_copy]
             + [f"not as in checksums.sha256: {f}" for f in bad_sum]
             + [f"wrong size: {f}" for f in bad_size]
             + [f"features differ: {f}" for f in bad_feat]
             + [f"manifest differs: {f}" for f in bad_man])
    if sums is None:
        fails.append(f"checksums.sha256 not read ({CHECKSUMS_FILE})")
    return pd.DataFrame(recs), fails


# ============================================================
# Thresholds (v5, unchanged)
# ============================================================

def thresholds(v4, X, rng):
    est, last = v4.fit_firth_batch(X, v4.simulate_many(
        X, np.zeros((N_THRESHOLD_PEOPLE, 2)), rng))
    return {"upper": np.percentile(est, 100 * (1 - THRESHOLD_ALPHA), axis=0),
            "lower": np.percentile(est, 100 * THRESHOLD_ALPHA, axis=0),
            "null_sd": est.std(axis=0),
            "unconverged": float(np.mean(last > 1e-6))}


def choice_prob(v4, X):
    out = {}
    for k, w in enumerate(W_NAMES):
        a = np.abs(X[:, k])
        info = a >= INFORMATIVE_SD
        med = float(np.median(a[info]))
        out[w] = (med, int(info.sum()),
                  v4.as_probability(PLANNING_WEIGHTS[k], med))
    return out


# ============================================================
# Report helpers
# ============================================================

def q_str(s, qs=(0.0, 0.1, 0.5, 0.9, 1.0)):
    s = pd.Series(s).abs()
    if not len(s):
        return "-"
    return " / ".join(f"{s.quantile(q):.3f}" for q in qs)


def per_type_table(main, w):
    for blk in ("A", "B"):
        m = main[blk]
        for t in MAIN_TYPES:
            mt = m[m.trial_type == t]
            w(f"  {blk}  {t:9s} n={len(mt):3d}  info sat "
              f"{(mt.pred_sat ** 2).sum():7.2f}  con "
              f"{(mt.pred_fog ** 2).sum():6.2f}  scenes "
              f"{mt.scene_id.nunique():2d}  A-left {int(mt.a_is_left.sum()):3d}")
            w(f"       |x_sat| min/10/50/90/max {q_str(mt.pred_sat)}")
            w(f"       |x_con| min/10/50/90/max {q_str(mt.pred_fog)}")


def fmt_b(r, d, blk):
    return (f"b_sat {r[f'leak_{d}_b_sat_{blk}']:+.3f}  b_con "
            f"{r[f'leak_{d}_b_con_{blk}']:+.3f}  (cos {r[f'leak_{d}_cos_sat_{blk}']:+.3f}"
            f" / {r[f'leak_{d}_cos_con_{blk}']:+.3f})")


# ============================================================
# One package
# ============================================================

NORM_FIELDS = ["pred_sat", "pred_fog", "a_is_left", "d_sat", "d_fog",
               "z_sat", "z_fog", "mag_bin", "N_corpus", "N_corpus_sat",
               "N_corpus_con", "N_orig", "N_orig_sat", "N_orig_con"]


def cand_table(results):
    rows = []
    for r in results:
        rows.append({k: v for k, v in r.items()
                     if k not in ("sets_draw", "sets_rep", "leak_rows")})
    return pd.DataFrame(rows)


def step_table(cap, R):
    """One row per LIMIT_STEPS step: candidates passing, the set it would
    ship (by the same selection), its information, leaks and positions, the
    mean information over its passing candidates, and the cost against the
    first step and the step before (on the shipped sets, and paired by
    candidate over candidates passing both). None without the steps."""
    if not R.get("steps"):
        return None
    rows, cands = [], []
    for name, types, cand in R["steps"]:
        chosen, c2, _ = select(cand)
        ok = c2[c2.rep_pass.fillna(False).astype(bool)]
        r = {"step": name, "types": "+".join(types) or "-",
             "passing": len(ok), "chosen": chosen["rep"]}
        if chosen["rep"] is not None:
            cr = c2[c2.candidate == chosen["rep"]].iloc[0]
            for w_ in W_NAMES:
                r[f"weaker_{w_}"] = cr[f"rep_weaker_set_info_{w_}"]
                r[f"total_{w_}"] = cr[f"rep_total_info_{w_}"]
            r["worst_abs_b"] = cr.rep_worst_abs_b
            for t in TYPE_LIMITS:
                r[f"type_worst_{t}"] = cr[f"rep_type_worst_{t}"]
            for k in cr.index:
                if k.startswith("rep_pos_") or k in (
                        "rep_straddle_sat", "rep_straddle_con",
                        "rep_with_original"):
                    r[k[4:]] = cr[k]
        for w_ in W_NAMES:
            r[f"mean_weaker_{w_}"] = ok[f"rep_weaker_set_info_{w_}"].mean()
        for k in ok.columns:
            if k.startswith("rep_pos_") or k in (
                    "rep_straddle_sat", "rep_straddle_con",
                    "rep_with_original"):
                r[f"mean_{k[4:]}"] = ok[k].mean()
        rows.append(r)
        cands.append(c2.set_index("candidate"))
    T = pd.DataFrame(rows)
    for i in range(len(T)):
        for ref_i, tag in ((0, "vs_first"), (max(i - 1, 0), "vs_prev")):
            for w_ in W_NAMES:
                a, b = T.loc[i].get(f"weaker_{w_}"), T.loc[ref_i].get(
                    f"weaker_{w_}")
                T.loc[i, f"shipped_{tag}_{w_}"] = (a / b - 1 if pd.notna(a)
                                                   and pd.notna(b) else np.nan)
                c, c0 = cands[i], cands[ref_i]
                both = c.index[c.rep_pass.fillna(False).astype(bool)
                               & c0.rep_pass.fillna(False).astype(bool)]
                T.loc[i, f"paired_{tag}_{w_}"] = float(
                    (c.loc[both, f"rep_weaker_set_info_{w_}"]
                     / c0.loc[both, f"rep_weaker_set_info_{w_}"] - 1).mean()
                ) if len(both) else np.nan
        worst = min(T.loc[i, f"{k}_vs_prev_{w_}"] for k in ("shipped",
                                                             "paired")
                    for w_ in W_NAMES)
        T.loc[i, "largest_cost_vs_prev"] = -worst
        T.loc[i, "meets_cost_rule"] = bool(
            (i == 0 or -worst <= COST_RULE[0] + 1e-12)
            and T.loc[i, "passing"] >= COST_RULE[1])
    return T


def write_failed(out_dir, size, cand, ck, why, t0):
    cand.to_csv(f"{out_dir}/candidates.csv", index=False)
    ck.frame().to_csv(f"{out_dir}/checks.csv", index=False)
    with open(f"{out_dir}/BUILD_FAILED.txt", "w") as f:
        f.write(f"THE BUILD FAILED. Do not use this folder.\n\n{why}\n")
    with open(f"{out_dir}/build_report.txt", "w") as f:
        f.write(f"v5b STEP 3, {size} ANALYSED TRIALS PER BLOCK: FAILED\n\n"
                f"{why}\n\nelapsed {time.time() - t0:.0f} s\n"
                "candidates.csv lists every candidate and why its repair "
                "failed.\n")


def build_one(cap, v4, size, R, pools_t, scenes, p_log, pre_checks, run,
              t0):
    ck = cap.Checks()
    for row in pre_checks:
        ck.add(*row)
    out_dir = resolve_output_dir(cap, size)
    os.makedirs(out_dir)
    print(f"\n  n={size}: output {out_dir}")
    counts, s1 = R["counts"], R["s1"]
    ck.add("design", "the counts are feasible (script 1 stage 1)",
           s1["status"] == "feasible", s1["status"])

    # --- candidates and the selection ---
    chosen, cand, ref = select(R["cand"])
    print(f"    shipped with the overall limit and per-type limits "
          f"{dict(TYPE_LIMITS) or 'none'}")
    n_built = int(cand.built.isin(["greedy", "fallback"]).sum())
    n_draw = int(cand.draw_pass.fillna(False).astype(bool).sum())
    n_rep = int(cand.rep_pass.fillna(False).astype(bool).sum())
    ck.add("candidates", "at least one repaired candidate passes the limit "
           "and every script 1 check", n_rep > 0,
           f"{n_rep} of {N_CANDIDATES} (draws built {n_built}, draws "
           f"passing script 1's checks before repair {n_draw})")
    ck.add("candidates", "wide repair (scenes within a bin free to change)",
           "USED" if R["wide_used"] else "NOT NEEDED",
           (f"{len(R['wide_candidates'])} candidates re-repaired: "
            f"{R['wide_candidates']}") if R["wide_used"] else
           f"{n_rep} passed with the cells kept; it runs when fewer than "
           f"{WIDE_MIN_PASSING} pass and a failed candidate can be "
           "re-repaired")
    if chosen["rep"] is None:
        why = (f"No repaired candidate passes at n={size} (of "
               f"{N_CANDIDATES}).")
        write_failed(out_dir, size, cand, ck, why, t0)
        return out_dir, 1, [why]
    d = chosen["rep"]
    res = R["results"][d]
    sets, sets0 = res["sets_rep"], res["sets_draw"]
    crow = cand[cand.candidate == d].iloc[0]
    print(f"    {n_rep}/{N_CANDIDATES} candidates pass after repair; chosen "
          f"candidate {d} (unconstrained choice: {chosen['draw']})")

    # --- the shipped sets: script 1's checks, the limit, reproducibility ---
    n_allowed = {"A": len(scenes), "B": len(scenes)}
    met, chk = cap.evaluate(sets, counts, s1, n_allowed)
    for k, v in chk.items():
        ck.add("sets (script 1 checks)", k,
               "SKIPPED" if v is None else bool(v),
               "" if v is not None else "not applicable")
    for dfn in cap.NORMALITY_DEFS:
        for blk in ("A", "B"):
            bs, bc = (met[f"leak_{dfn}_b_sat_{blk}"],
                      met[f"leak_{dfn}_b_con_{blk}"])
            ck.add("normality limit", f"set {blk}, {dfn} definition: |b_sat| "
                   f"and |b_con| <= {LEAK_LIMIT}",
                   max(abs(bs), abs(bc)) <= LEAK_LIMIT + cap.EPS,
                   f"b_sat {bs:+.4f}, b_con {bc:+.4f}")
    tb = type_b(cap, sets)
    for t, (term, lim) in TYPE_LIMITS.items():
        for dfn in cap.NORMALITY_DEFS:
            for blk in ("A", "B"):
                v = tb[f"type_b_{t}_{dfn}_{blk}"]
                ck.add("normality limit", f"set {blk}, {dfn} definition, {t}"
                       f" (N on x_{term} alone): |b| <= {lim:g}",
                       abs(v) <= lim + cap.EPS, f"b_{term} {v:+.4f}")
    ok_rep, det = R["repro"]
    ck.add("candidates", "the chosen candidate rebuilt in this process (one "
           "worker) gives byte-identical sets", ok_rep, det)
    ck.add(*R["s2check"])

    # --- rows ---
    st = cap.Streams(RANDOM_SEED, "build_trials_v5b", size)
    main = {b: main_rows(sets, b) for b in ("A", "B")}
    n_cons = int(np.floor(size * CONSISTENCY_RATE + 1e-9))
    cons, alloc = consistency_rows(main["A"], main["B"], n_cons,
                                   st("consistency"))
    used = set(main["A"].key) | set(main["B"].key)
    prac = {}
    for i, n in enumerate(PRACTICE_BY_BLOCK):
        label = f"P{i + 1}"
        prac[label] = practice_rows(pools_t, used, n, label,
                                    st(f"practice{i + 1}"))
    catch = {b: catch_rows(pools_t, used, b, st(f"catch{b}"))
             for b in ("A", "B")}
    full = {b: pd.concat([main[b], cons[b], catch[b]], ignore_index=True)
            for b in ("A", "B")}
    for b in ("A", "B"):
        full[b]["couple"] = full[b].couple.astype("Int64")   # blank on catch

    # --- checks on the rows ---
    th = cap.THRESHOLD_SETS[THRESHOLD_SET]
    for b in ("A", "B"):
        m, c, x = main[b], cons[b], catch[b]
        tc = m.trial_type.value_counts().to_dict()
        ck.add(f"set {b}", "analysed type counts equal the target",
               all(tc.get(t, 0) == counts[t] for t in MAIN_TYPES),
               ", ".join(f"{t} {tc.get(t, 0)}" for t in MAIN_TYPES))
        so = m[m.trial_type == "sat_only"]
        ck.add(f"set {b}", f"sat_only: |d_con| <= {cap.CON_TOL_SD:.4f} SD and "
               f"|d_sat| >= {th['sat_only']:.4f} SD",
               bool((so.z_fog <= cap.CON_TOL_SD + cap.EPS).all()
                    and (so.z_sat >= th["sat_only"] - cap.EPS).all()),
               f"largest |d_con| {so.z_fog.max():.4f}, smallest |d_sat| "
               f"{so.z_sat.min():.4f} SD")
        fo = m[m.trial_type == "fog_only"]
        ck.add(f"set {b}", f"fog_only: |d_sat| <= {cap.SAT_TOL_SD:.4f} SD and "
               f"|d_con| >= {th['fog_only']:.4f} SD",
               bool((fo.z_sat <= cap.SAT_TOL_SD + cap.EPS).all()
                    and (fo.z_fog >= th["fog_only"] - cap.EPS).all()),
               f"largest |d_sat| {fo.z_sat.max():.4f}, smallest |d_con| "
               f"{fo.z_fog.min():.4f} SD")
        mx = m[m.trial_type.isin(["together", "conflict"])]
        cf = m[m.trial_type == "conflict"]
        ck.add(f"set {b}", "together, conflict: both differences >= their "
               "minimums; conflict balanced",
               bool(((mx.z_sat >= th["mixed_sat"] - cap.EPS)
                     & (mx.z_fog >= th["mixed_con"] - cap.EPS)).all()
                    and cf.balanced.astype(bool).all()))
        ck.add(f"set {b}", f"{len(c)} consistency trials = floor({size} x "
               f"{CONSISTENCY_RATE:.4f})", len(c) == n_cons,
               ", ".join(f"{t} {alloc[t]}" for t in MAIN_TYPES))
        orig = m.set_index("item_id")
        ok_rep = all(
            r.repeat_of in orig.index
            and orig.loc[r.repeat_of, "key"] == r.key
            and bool(orig.loc[r.repeat_of, "a_is_left"]) != bool(r.a_is_left)
            and np.isclose(orig.loc[r.repeat_of, "pred_sat"], r.pred_sat)
            and np.isclose(orig.loc[r.repeat_of, "pred_fog"], r.pred_fog)
            for r in c.itertuples())
        ck.add(f"set {b}", "every consistency trial repeats an analysed pair "
               "of its set with the sides swapped", bool(ok_rep))
        ck.add(f"set {b}", f"{CATCH_PER_BLOCK} catch trials, mode "
               f"'{CATCH_MODE}', distinct scenes",
               len(x) == CATCH_PER_BLOCK and x.scene_id.is_unique
               and (x.catch_mode == CATCH_MODE).all())
        side_bad = []
        for t in MAIN_TYPES:
            mt = m[m.trial_type == t]
            nl = int(mt.a_is_left.sum())
            if nl not in (len(mt) // 2, (len(mt) + 1) // 2):
                side_bad.append(f"{t} {nl} of {len(mt)}")
        ck.add(f"set {b}", "image A (the predicted image) on the left in half "
               "of each type's trials (+-1 if odd)", not side_bad,
               "; ".join(side_bad))
        nl = int(x.a_is_left.sum())
        ck.add(f"set {b}", "catch sides balanced (+-1)",
               nl in (len(x) // 2, (len(x) + 1) // 2), f"{nl} of {len(x)}")
    ck.add("consistency", "the same couples are repeated in both sets "
           "(matched)", sorted(zip(cons["A"].repeat_type,
                                   main["A"].set_index("item_id")
                                   .loc[cons["A"].repeat_of, "couple"]))
           == sorted(zip(cons["B"].repeat_type,
                         main["B"].set_index("item_id")
                         .loc[cons["B"].repeat_of, "couple"])))
    for label, df in prac.items():
        n_want = PRACTICE_BY_BLOCK[int(label[1]) - 1]
        types = list(df.type)
        want = [PRACTICE_TYPES[i % len(PRACTICE_TYPES)]
                for i in range(n_want)]
        thr_ok = all(
            r.driver <= pools_t[r.type].driver_sd.quantile(
                PRACTICE_MAG_QUANTILE) + 1e-12
            for r in df.assign(driver=np.where(
                df.type == "sat_only", df.pred_sat.abs(),
                df.pred_fog.abs())).itertuples())
        ck.add(f"practice {label}", f"{n_want} trials of types {want} in "
               "turn, distinct scenes", types == want
               and df.scene_id.is_unique, ", ".join(types))
        ck.add(f"practice {label}", f"every difference at or below the "
               f"{PRACTICE_MAG_QUANTILE:.0%} quantile of its type", thr_ok)
    allrows = pd.concat([full["A"], full["B"]] + list(prac.values()),
                        ignore_index=True)
    first = allrows[allrows.trial_type != "consistency"]
    dup = int(first.key.duplicated().sum())
    ck.add("all rows", "no pair is used twice anywhere (consistency repeats "
           "aside): sets, practice and catch are disjoint", dup == 0,
           f"{len(first)} pairs, {dup} duplicates")
    miss = [c for c in NORM_FIELDS if c not in allrows.columns
            or allrows[c].isna().any()]
    ck.add("all rows", "every row carries the direction, difference, bin and "
           "normality fields", not miss,
           f"missing or empty: {miss}" if miss else
           f"{len(NORM_FIELDS)} fields in {len(allrows)} rows")
    n_orig = {lab: int(df.includes_original.astype(bool).sum())
              for lab, df in [("A", main["A"]), ("B", main["B"])]}
    ck.add("all rows", "unedited images are allowed (counted, not limited)",
           "NOTE", f"analysed pairs with the unedited image: A "
           f"{n_orig['A']}, B {n_orig['B']}; all rows "
           f"{int(allrows.includes_original.astype(bool).sum())}")

    # --- leak of the trials outside the limit (reported only) ---
    extra_leak = []
    base = {"size": size, "candidate": d, "stage": "shipped"}
    pp = pd.concat(list(prac.values()), ignore_index=True)
    for lab, df in [("practice P1+P2", pp), ("catch A", catch["A"]),
                    ("catch B", catch["B"])]:
        df2 = df.assign(x_sat=df.pred_sat, x_con=df.pred_fog,
                        trial_type="other")
        extra_leak += leak_rows_of(cap, df2, base, lab)

    # --- choice probability and thresholds per set ---
    info = {}
    for b in ("A", "B"):
        X = main[b][["pred_sat", "pred_fog"]].to_numpy(float)
        tt = time.time()
        th_ = thresholds(v4, X, st(f"thresholds{b}"))
        cp = choice_prob(v4, X)
        info[b] = {"th": th_, "cp": cp, "X": X, "sec": time.time() - tt}
        ck.add(f"set {b}", "choice probability at the planning weights "
               "(reported; v4's for reference)", "NOTE",
               f"saturation {cp['sat'][2]:.4f} (v4 {V4_CHOICE_PROB[0]}), "
               f"contrast {cp['con'][2]:.4f} (v4 {V4_CHOICE_PROB[1]})")
        ck.add(f"set {b}", "every threshold fit converged",
               th_["unconverged"] == 0,
               f"share unconverged {th_['unconverged']:.5f}")
    rel = max(abs(info["A"]["th"]["upper"][k] - info["B"]["th"]["upper"][k])
              / np.mean([info["A"]["th"]["upper"][k],
                         info["B"]["th"]["upper"][k]]) for k in range(2))
    ck.add("thresholds", f"the two sets' upper thresholds agree within "
           f"{SET_THRESHOLD_TOL:.0%}", rel <= SET_THRESHOLD_TOL,
           f"largest relative difference {rel:.1%}")

    # --- write the lists before the images, so a failure leaves them ---
    for b in ("A", "B"):
        full[b].drop(columns=["key"]).to_csv(f"{out_dir}/set_{b}.csv",
                                             index=False)
    for i, label in enumerate(prac):
        prac[label].drop(columns=["key"]).to_csv(
            f"{out_dir}/practice_block{i + 1}.csv", index=False)
    cand.to_csv(f"{out_dir}/candidates.csv", index=False)
    lk = pd.DataFrame([r for c_ in R["results"].values()
                       for r in c_["leak_rows"]] + extra_leak)
    lk.to_csv(f"{out_dir}/leak_by_type.csv", index=False)
    steps = step_table(cap, R)
    if steps is not None:
        steps.to_csv(f"{out_dir}/limit_steps.csv", index=False)
    with open(f"{out_dir}/settings.json", "w") as f:
        json.dump({**settings_dict(cap, size),
                   "run": {"workers": run["workers"]}}, f, indent=2,
                  default=str)

    # --- images ---
    manifest = (pd.read_csv(MANIFEST_CSV) if os.path.isfile(MANIFEST_CSV)
                else None)
    man, fails = ship_images(allrows.drop_duplicates("item_id"), out_dir, ck,
                             manifest)
    if man is not None:
        man.to_csv(f"{out_dir}/manifest.csv", index=False)

    # --- set_info.json ---
    set_info = {
        "version": "v5b", "block_size": size, "mix": MIX,
        "mix_shares": MIX_SHARES, "threshold_set": THRESHOLD_SET,
        "threshold_set_values_sd": th,
        "chosen_candidate": d, "unconstrained_choice": chosen["draw"],
        "selection": ("max over candidates of min over weights of the "
                      "weaker set's information / mean weaker-set "
                      "information of the unrepaired candidates; ties: "
                      "smaller worst |b|, then lower candidate"),
        "selection_reference": ref,
        "normality_limit": {
            "limit": LEAK_LIMIT, "fit": "N ~ b_sat x_sat + b_con x_con, no "
            "intercept, analysed trials of each set",
            "definitions": list(cap.NORMALITY_DEFS),
            "shipped": {f"{dfn}_{b}": {k: met[f"leak_{dfn}_{k}_{b}"] for k in
                                       ("b_sat", "b_con", "cos_sat",
                                        "cos_con")}
                        for dfn in cap.NORMALITY_DEFS for b in ("A", "B")},
            "type_limits": {
                t: {"term": term, "limit": lim, "fit": f"N ~ b x_{term}, no "
                    "intercept, the type's analysed trials of each set",
                    "shipped": {f"{dfn}_{b}": tb[f"type_b_{t}_{dfn}_{b}"]
                                for dfn in cap.NORMALITY_DEFS
                                for b in ("A", "B")}}
                for t, (term, lim) in TYPE_LIMITS.items()},
            "repair": {"mode": crow.repair_mode,
                       "kept_A": int(crow.kept_A),
                       "kept_B": int(crow.kept_B),
                       "target": float(crow.repair_target),
                       "solves": int(crow.repair_solves)}},
        "counts_per_set": {"analysed": size, **counts,
                           "consistency": n_cons,
                           "consistency_by_type": alloc,
                           "catch": CATCH_PER_BLOCK},
        "practice_by_block": list(PRACTICE_BY_BLOCK),
        "catch": {"mode": CATCH_MODE, "per_block": CATCH_PER_BLOCK,
                  "exclude_at_or_above": CATCH_EXCLUDE_AT,
                  "exclusion_scope": "session (both blocks)",
                  "record_failures_per_block": True},
        "thresholds": {b: {"n_random_choosers": N_THRESHOLD_PEOPLE,
                           "tail": THRESHOLD_ALPHA,
                           "estimator": "Firth (v4 fit_firth_batch)",
                           "upper": {"sat": info[b]["th"]["upper"][0],
                                     "con": info[b]["th"]["upper"][1]},
                           "lower": {"sat": info[b]["th"]["lower"][0],
                                     "con": info[b]["th"]["lower"][1]}}
                       for b in ("A", "B")},
        "choice_probability_at_planning_weights": {
            b: {"sat": info[b]["cp"]["sat"][2],
                "con": info[b]["cp"]["con"][2]} for b in ("A", "B")},
        "predictors": (f"pred_sat = -d_sat / {cap.SD_SAT}, pred_fog = d_fog "
                       f"/ {cap.SD_FOG}; positive = image A is the "
                       "predicted choice"),
        "normality": ("N = dist(B) - dist(A); positive = image A is the more "
                      "typical; corpus = distance from the corpus median, "
                      "orig = distance from the scene's unedited photograph"),
        "stimulus_set": {"dir": STIMULUS_SET_DIR, **{
            k: v for k, v in cap.STIMULUS_SET.items() if k != "manifest"}},
        "script1": {"name": CAPACITY_SCRIPT,
                    "sha256": sha256_of(CAPACITY_SCRIPT)},
        "script3": {"name": SCRIPT_NAME, "sha256": sha256_of(__file__)},
        "workers": run["workers"],
    }
    with open(f"{out_dir}/set_info.json", "w") as f:
        json.dump(set_info, f, indent=2, default=float)

    # --- report ---
    write_report(cap, out_dir, size, ck, p_log, cand, ref, chosen, R, res,
                 sets, sets0, main, cons, catch, prac, info, counts, n_cons,
                 alloc, met, man, fails, extra_leak, run, t0, steps)
    ck.frame().to_csv(f"{out_dir}/checks.csv", index=False)
    cf = ck.frame()
    n_fail = int((cf.status == "FAIL").sum())
    if fails or n_fail:
        bad = cf[cf.status == "FAIL"]
        with open(f"{out_dir}/BUILD_FAILED.txt", "w") as f:
            f.write("THE BUILD FAILED. Do not use this folder.\n\n"
                    + "\n".join(f"FAIL [{r.scope}] {r.check} ({r.detail})"
                                for r in bad.itertuples())
                    + ("\n\n" + "\n".join(fails) if fails else "") + "\n")
    return out_dir, n_fail, fails


# ============================================================
# Report
# ============================================================

def _pct(a, b):
    return (a / b - 1) if b else np.nan


def _mmm(s, f="{:.3f}"):
    s = pd.Series(s).dropna()
    if not len(s):
        return "-"
    return (f"{f.format(s.mean())} [{f.format(s.min())} to "
            f"{f.format(s.max())}]")


def position_lines(w, before, after, label_b="before", label_a="after"):
    w(f"  {'':22s} {'image A (predicted)':>21s}  {'image B':>21s}  "
      f"{'all images':>21s}")
    w(f"  {'':22s} " + "  ".join(["above    at  below"] * 3))
    for feat, lab in (("sat", "saturation"), ("con", "contrast")):
        for tag, P in ((label_b, before), (label_a, after)):
            cells = []
            for side in ("A", "B", "all"):
                cells.append(" ".join(
                    f"{P[f'pos_{feat}_{side}_{k}']:5.0%}"
                    for k in ("above", "at", "below")))
            w(f"  {lab:10s} {tag:11s} " + "   ".join(cells))
    w(f"  pairs straddling the original (measured): saturation "
      f"{before['straddle_sat']} -> {after['straddle_sat']}, contrast "
      f"{before['straddle_con']} -> {after['straddle_con']}")
    w(f"  pairs with the unedited image: {before['with_original']} -> "
      f"{after['with_original']}")


def step_lines(w, T):
    w("5b. THE LIMIT STEPS: WHAT EACH ADDED LIMIT COSTS")
    w("-" * 78)
    w("Every candidate repaired under each step; each step's set chosen by "
      "the same rule. The first")
    w("step is the overall limit alone (the first v5b build). Cost = change "
      "in the weaker set's")
    w("information: on the set each step would ship, and the mean over "
      "candidates passing both steps.")
    w(f"{'step':18s} {'pass':>4s} {'cand':>4s} {'weaker sat':>10s} "
      f"{'weaker con':>10s}   {'vs first: shipped':>19s} {'paired':>14s}   "
      f"{'vs previous: shipped':>20s} {'paired':>14s}")
    for r in T.itertuples():
        def pc(tag, kind):
            return (f"{100 * getattr(r, f'{kind}_{tag}_sat'):+5.1f}/"
                    f"{100 * getattr(r, f'{kind}_{tag}_con'):+5.1f}%")
        w(f"{r.step:18s} {r.passing:4d} {('-' if pd.isna(r.chosen) else int(r.chosen))!s:>4s} "
          f"{r.weaker_sat:10.2f} {r.weaker_con:10.2f}   "
          f"{pc('vs_first', 'shipped'):>19s} {pc('vs_first', 'paired'):>14s}"
          f"   {pc('vs_prev', 'shipped'):>20s} {pc('vs_prev', 'paired'):>14s}")
    w("  (percentages: saturation / contrast; negative = information lost)")
    w("Leak of each step's shipped set: worst overall |b|, and the worst "
      "single-term |b| per limited type:")
    for r in T.itertuples():
        w(f"  {r.step:18s} overall {r.worst_abs_b:.3f}   " + "   ".join(
            f"{t} {getattr(r, f'type_worst_{t}'):.3f}" for t in TYPE_LIMITS))
    w(f"Cost rule: a step is acceptable when it costs at most "
      f"{COST_RULE[0]:.0%} of either weight's information against the")
    w(f"step before (shipped and paired) and at least {COST_RULE[1]} "
      "candidates pass:")
    for r in T.itertuples():
        w(f"  {r.step:18s} largest cost vs previous "
          f"{100 * r.largest_cost_vs_prev:+.1f}%, {r.passing} passing -> "
          + ("ACCEPTABLE" if r.meets_cost_rule else "NOT ACCEPTABLE"))
    w("Where the images sit, each step's shipped set (both sets), share above"
      " / at / below the original:")
    w(f"  {'':18s} {'':4s} {'image A':>17s}   {'image B':>17s}   "
      f"{'all':>17s}   straddle s/c  with orig")
    for r in T.itertuples():
        for feat, lab in (("sat", "sat"), ("con", "con")):
            cells = "   ".join(" ".join(
                f"{getattr(r, f'pos_{feat}_{side}_{k}'):4.0%}"
                for k in ("above", "at", "below"))
                for side in ("A", "B", "all"))
            extra = (f"   {int(r.straddle_sat):4d}/{int(r.straddle_con):<4d}"
                     f"   {int(r.with_original):5d}") if feat == "sat" else ""
            w(f"  {r.step if feat == 'sat' else '':18s} {lab:4s} {cells}"
              + extra)
    w("Mean over each step's passing candidates:")
    for r in T.itertuples():
        for feat in ("sat", "con"):
            cells = "   ".join(" ".join(
                f"{getattr(r, f'mean_pos_{feat}_{side}_{k}'):4.0%}"
                for k in ("above", "at", "below"))
                for side in ("A", "B", "all"))
            extra = (f"   {r.mean_straddle_sat:5.1f}/{r.mean_straddle_con:<5.1f}"
                     f" {r.mean_with_original:5.1f}") if feat == "sat" else ""
            w(f"  {r.step if feat == 'sat' else '':18s} {feat:4s} {cells}"
              + extra)
    w("  limit_steps.csv holds these figures.")
    w("")


def write_report(cap, out_dir, size, ck, p_log, cand, ref, chosen, R, res,
                 sets, sets0, main, cons, catch, prac, info, counts, n_cons,
                 alloc, met, man, fails, extra_leak, run, t0, steps=None):
    L = []
    w = L.append
    d = chosen["rep"]
    crow = cand[cand.candidate == d].iloc[0]
    passing = cand[cand.rep_pass.fillna(False).astype(bool)]
    dpass = cand[cand.draw_pass.fillna(False).astype(bool)]
    both = cand[cand.rep_pass.fillna(False).astype(bool)
                & cand.draw_pass.fillna(False).astype(bool)]
    defs = cap.NORMALITY_DEFS
    w("=" * 78)
    w(f"v5b STEP 3: THE TWO BLOCK LISTS, {size} ANALYSED TRIALS PER BLOCK")
    w(f"{SCRIPT_NAME}")
    w("=" * 78)
    w(f"run finished  {time.strftime('%Y-%m-%d %H:%M:%S')}   elapsed "
      f"{time.time() - t0:.0f} s in all")
    w(f"output        {out_dir}")
    w(f"workers       {run['workers']} process(es), one thread each; the "
      f"candidates (both sizes) took {run['cand_seconds']:.0f} s")
    if fails or (ck.frame().status == "FAIL").any():
        w("")
        w("!!! A CHECK FAILED -- BUILD_FAILED.txt is in this folder and "
          "scripts 4 and 5 will refuse it. See section 11. !!!")
    w("")
    w("1. WHAT THIS RUN DID")
    w("-" * 78)
    w(f"Threshold set '{THRESHOLD_SET}', mix '{MIX}' ("
      + ", ".join(f"{t} {counts[t]}" for t in MAIN_TYPES) + " per set).")
    w(f"Script 1's draw rules (cap {cap.MAX_PER_SCENE_PER_TYPE} per scene per "
      f"type per set, matching '{cap.SCENE_MATCH}', disjoint "
      f"'{cap.DISJOINT}').")
    w(f"{N_CANDIDATES} candidates (script 1's one_draw; candidate d = script "
      "1's and script 2's draw d), each REPAIRED:")
    w("  an integer program keeps as many of the draw's pairs as it can and "
      "replaces the rest")
    w("  with unused pairs from the same scene x type x bin cells, holding "
      f"the leak moments within {REPAIR_TARGET:g}")
    w(f"  (tightened x{REPAIR_TIGHTEN:g} if the exact fit misses, up to "
      f"{REPAIR_TRIES} solves) and the information matched within "
      f"{REPAIR_INFO_TOL_TOTAL:.0%} / {REPAIR_INFO_TOL_TYPE:.0%} per type.")
    w(f"LIMIT: the no-intercept fit N ~ b_sat x_sat + b_con x_con on each "
      f"set's analysed trials gives |b| <= {LEAK_LIMIT:g}")
    w(f"under both definitions ({', '.join(defs)}), and every script 1 "
      "check passes.")
    if TYPE_LIMITS:
        w("PER-TYPE LIMITS (one term, no intercept, each set, both "
          "definitions): " + "; ".join(
              f"{t} N on x_{term} |b| <= {lim:g}"
              for t, (term, lim) in TYPE_LIMITS.items()) + ".")
    else:
        w("No per-type limits (TYPE_LIMITS is empty).")
    w("SELECTION: the weaker set's information per weight relative to the "
      "mean of the unrepaired")
    w(f"candidates (saturation {ref['sat']:.2f}, contrast {ref['con']:.2f}); "
      "the highest smaller value wins, ties on the smaller worst |b|.")
    w(f"Per set: {n_cons} consistency, {CATCH_PER_BLOCK} catch. Practice: "
      f"{PRACTICE_BY_BLOCK[0]} before block 1, {PRACTICE_BY_BLOCK[1]} before "
      "block 2.")
    w("")
    w("Inputs and provenance")
    for line in p_log:
        w("  " + line)
    w("")

    w("2. CANDIDATES: REPAIR AND SELECTION")
    w("-" * 78)
    w(f"draws built {int(cand.built.isin(['greedy', 'fallback']).sum())}, "
      f"passing script 1's checks before repair {len(dpass)}; PASSING AFTER "
      f"REPAIR {len(passing)} of {N_CANDIDATES}.")
    if R["wide_used"]:
        w(f"WIDE REPAIR WAS USED: fewer than {WIDE_MIN_PASSING} passed with "
          "the cells kept, so candidates "
          f"{R['wide_candidates']} were repaired again with the scenes within "
          "each type x bin free to change.")
    else:
        w(f"Wide repair not used ({len(passing)} passed with the cells kept; "
          f"it runs when fewer than {WIDE_MIN_PASSING} pass).")
    nt = int(cand.repair_time_limit_hits.fillna(0).sum())
    w(f"Solves stopped by the {REPAIR_TIME_LIMIT} s limit: {nt} (where a "
      "stopped solve ends can depend on the computer's speed).")
    w(f"Repair time per candidate: {_mmm(cand.repair_seconds, '{:.1f}')} s.")
    w(f"Chosen: candidate {d}. Unconstrained choice (same rule, unrepaired "
      f"draws): candidate {chosen['draw']}.")
    w("")
    w(f"{'cand':>4s} {'draw':>4s} {'mode':>5s} {'kept A/B':>9s} {'solv':>4s} "
      f"{'worst|b| before':>15s} {'after':>6s} {'rel sat':>7s} {'rel con':>7s}"
      f" {'score':>6s} {'rank':>4s}")
    for r in cand.sort_values(["rep_rank", "candidate"],
                              na_position="last").itertuples():
        if not r.built or r.built not in ("greedy", "fallback"):
            w(f"{r.candidate:4d}   (draw not built)")
            continue
        kept = (f"{int(r.kept_A)}/{int(r.kept_B)}"
                if pd.notna(r.kept_A) else "-")
        aft = (f"{r.rep_worst_abs_b:6.3f}" if pd.notna(
            getattr(r, "rep_worst_abs_b", np.nan)) else "     -")
        rk = "" if pd.isna(r.rep_rank) else int(r.rep_rank)
        line = (f"{r.candidate:4d} {'yes' if r.draw_pass else 'no':>4s} "
                f"{r.repair_mode:>5s} {kept:>9s} {int(r.repair_solves):4d} "
                f"{r.draw_worst_abs_b:15.3f} {aft} "
                f"{r.rep_rel_sat:7.3f} {r.rep_rel_con:7.3f} "
                f"{r.rep_score:6.3f} {rk!s:>4s}")
        if r.rep_chosen:
            line += "  <- SHIPPED"
        if not r.rep_pass:
            line += f"  FAILS: {r.repair_status}"
        w(line)
    w("  draw = passes script 1's checks before repair; kept = the draw's "
      "pairs kept per set;")
    w("  rel = weaker-set information / the reference above; score = the "
      "smaller rel.")
    w("")

    w("3. THE NORMALITY LEAK (analysed trials; LIMITED)")
    w("-" * 78)
    w(f"Shipped candidate {d}, before -> after repair (b from the "
      "no-intercept fit; cos = uncentred correlation):")
    for dfn in defs:
        for blk in ("A", "B"):
            before = {k.replace("draw_", ""): v for k, v in crow.items()
                      if k.startswith("draw_leak")}
            after = {k.replace("rep_", ""): v for k, v in crow.items()
                     if k.startswith("rep_leak")}
            w(f"  {dfn:6s} set {blk}  before {fmt_b(before, dfn, blk)}")
            w(f"  {'':6s}        after  {fmt_b(after, dfn, blk)}")
    w("Over candidates (mean [min to max] of b over both sets):")
    for dfn in defs:
        for k in ("b_sat", "b_con"):
            bb = pd.concat([dpass[f"draw_leak_{dfn}_{k}_{b}"]
                            for b in ("A", "B")])
            aa = pd.concat([passing[f"rep_leak_{dfn}_{k}_{b}"]
                            for b in ("A", "B")])
            w(f"  {dfn:6s} {k}: unrepaired (passing draws) {_mmm(bb)};  "
              f"repaired (passing) {_mmm(aa)}")
    pl = run["pool_leak"]
    w("The whole pool, for reference: " + "; ".join(
        f"{dfn} b_sat {pl[dfn]['b_sat']:+.3f} b_con {pl[dfn]['b_con']:+.3f}"
        for dfn in defs))
    w("")

    w("4. THE LEAK BY TRIAL TYPE")
    w("-" * 78)
    if TYPE_LIMITS:
        w("LIMITED types, the fit that is limited (N on one term, no "
          "intercept), before -> after repair:")
        for t, (term, lim) in TYPE_LIMITS.items():
            for dfn in defs:
                w(f"  {t:9s} {dfn:6s} N on x_{term} (limit {lim:g})  " + "   "
                  .join(f"{blk}: {crow[f'draw_type_b_{t}_{dfn}_{blk}']:+.3f}"
                        f" -> {crow[f'rep_type_b_{t}_{dfn}_{blk}']:+.3f}"
                        for blk in ("A", "B")))
        if len(passing):
            w("  over passing candidates, largest |b| per candidate (both "
              "sets, both definitions), mean [min to max]:")
            for t in TYPE_LIMITS:
                w(f"    {t:9s} before {_mmm(passing[f'draw_type_worst_{t}'])}"
                  f"   after {_mmm(passing[f'rep_type_worst_{t}'])}")
        w("")
    w("Every type, script 1's convention (one term for sat_only and "
      "fog_only, both for the mixed types;")
    w("together " + ("is" if "together" not in TYPE_LIMITS else "is also")
      + " reported only):")
    lr = pd.DataFrame(res["leak_rows"])
    lr = lr[lr.terms == [("+".join(TYPE_TERMS.get(t, ("sat", "con")))
                          if t in MAIN_TYPES else "sat+con")
                         for t in lr.type]]
    w(f"Shipped candidate {d}, b before -> after:")
    for t in MAIN_TYPES:
        for dfn in defs:
            parts = []
            for blk in ("A", "B"):
                g = lr[(lr.type == t) & (lr.definition == dfn)
                       & (lr.set == blk)]
                b0 = g[g.stage == "before"].iloc[0]
                b1 = g[g.stage == "after"].iloc[0]
                terms = b0.terms.split("+")
                parts.append(f"{blk}: " + ", ".join(
                    f"b_{tt} {b0[f'b_{tt}']:+.3f}->{b1[f'b_{tt}']:+.3f}"
                    for tt in terms))
            w(f"  {t:9s} {dfn:6s} " + "   ".join(parts))
    w("Uncentred correlations of N with x_sat / x_con (cos_sat / cos_con), "
      "before -> after, and the")
    w("correlation of x_sat with x_con within the type (cos_x):")
    for t in MAIN_TYPES:
        for dfn in defs:
            parts = []
            for blk in ("A", "B"):
                g = lr[(lr.type == t) & (lr.definition == dfn)
                       & (lr.set == blk)]
                b0 = g[g.stage == "before"].iloc[0]
                b1 = g[g.stage == "after"].iloc[0]
                parts.append(f"{blk}: {b0.cos_sat:+.3f}/{b0.cos_con:+.3f} -> "
                             f"{b1.cos_sat:+.3f}/{b1.cos_con:+.3f} (cos_x "
                             f"{b1.cos_x:+.2f})")
            w(f"  {t:9s} {dfn:6s} " + "   ".join(parts))
    w("  In together and conflict trials x_sat and x_con move together "
      "(|cos_x| near 1 for conflict),")
    w("  so the two b's of a mixed type are individually unstable: only "
      "their sum (together) or")
    w("  difference (conflict) is well determined. The cos values are the "
      "steadier guide there.")
    allr = pd.DataFrame([r for c_ in R["results"].values()
                         for r in c_["leak_rows"]])
    if len(allr) and len(passing):
        sub = allr[allr.candidate.isin(passing.candidate)]
        w("Over passing candidates, mean b (both sets), before -> after:")
        for t in MAIN_TYPES:
            for dfn in defs:
                terms = TYPE_TERMS.get(t, ("sat", "con"))
                g = sub[(sub.type == t) & (sub.definition == dfn)
                        & (sub.terms == "+".join(terms))]
                w(f"  {t:9s} {dfn:6s} " + ", ".join(
                    f"b_{tt} {g[g.stage == 'before'][f'b_{tt}'].mean():+.3f}"
                    f"->{g[g.stage == 'after'][f'b_{tt}'].mean():+.3f}"
                    for tt in terms))
    w("Trials outside the limit (shipped; both terms; few trials, so "
      "noisy):")
    el = pd.DataFrame(extra_leak)
    for lab in el.set.unique():
        for dfn in defs:
            g = el[(el.set == lab) & (el.definition == dfn)].iloc[0]
            w(f"  {lab:15s} {dfn:6s} n={int(g.n):2d}  b_sat {g.b_sat:+.3f} "
              f"b_con {g.b_con:+.3f}  (cos {g.cos_sat:+.3f} / "
              f"{g.cos_con:+.3f})")
    w("  Consistency repeats are copies of analysed pairs: their leak is "
      "the analysed trials'.")
    w("")

    w("5. THE INFORMATION COST OF THE LIMIT")
    w("-" * 78)
    du = cand[cand.candidate == chosen["draw"]].iloc[0] \
        if chosen["draw"] is not None else None
    w(f"Shipped (repaired candidate {d}) against the unconstrained selection "
      f"(unrepaired candidate {chosen['draw']}):")
    for wn, lab in (("sat", "saturation"), ("con", "contrast")):
        if du is None:
            w(f"  {lab}: no unrepaired candidate passes script 1's checks")
            continue
        a, b = crow[f"rep_weaker_set_info_{wn}"], du[f"draw_weaker_set_info_{wn}"]
        a2, b2 = crow[f"rep_total_info_{wn}"], du[f"draw_total_info_{wn}"]
        w(f"  {lab:10s} weaker set {a:8.2f} vs {b:8.2f} ({_pct(a, b):+.1%});"
          f"  both sets {a2:8.2f} vs {b2:8.2f} ({_pct(a2, b2):+.1%})")
    if len(both):
        w(f"Each candidate against its own draw ({len(both)} candidates "
          "passing before and after), % change, mean [min to max]:")
        for wn, lab in (("sat", "saturation"), ("con", "contrast")):
            ws = both[f"rep_weaker_set_info_{wn}"] / both[
                f"draw_weaker_set_info_{wn}"] - 1
            ts = both[f"rep_total_info_{wn}"] / both[
                f"draw_total_info_{wn}"] - 1
            w(f"  {lab:10s} weaker set {_mmm(100 * ws, '{:+.1f}')}   both "
              f"sets {_mmm(100 * ts, '{:+.1f}')}")
    w("Mean over passing candidates, both sets: " + "; ".join(
        f"{lab} unrepaired {dpass[f'draw_total_info_{wn}'].mean():.2f}, "
        f"repaired {passing[f'rep_total_info_{wn}'].mean():.2f}"
        for wn, lab in (("sat", "saturation"), ("con", "contrast"))))
    w("")

    if steps is not None and len(steps):
        step_lines(w, steps)
    w("6. WHERE THE IMAGES SIT RELATIVE TO THE ORIGINAL")
    w("-" * 78)
    w(f"Share of images above / at (within {cap.STRADDLE_TOL_SD} SD) / below "
      "the scene's unedited photograph, by measurement.")
    w(f"Shipped candidate {d}, both sets, before vs after repair:")
    position_lines(w, {k[5:]: v for k, v in crow.items()
                       if k.startswith("draw_pos_") or k in (
                           "draw_straddle_sat", "draw_straddle_con",
                           "draw_with_original")},
                   {k[4:]: v for k, v in crow.items()
                    if k.startswith("rep_pos_") or k in (
                        "rep_straddle_sat", "rep_straddle_con",
                        "rep_with_original")})
    for blk in ("A", "B"):
        w(f"Shipped set {blk} alone:")
        position_lines(w, position_metrics(cap, analysed(sets0[blk])),
                       position_metrics(cap, analysed(sets[blk])))
    if len(both):
        mb = {k[5:]: both[k].mean() for k in both.columns
              if k.startswith("draw_pos_") or k in (
                  "draw_straddle_sat", "draw_straddle_con",
                  "draw_with_original")}
        ma = {k[4:]: both[k].mean() for k in both.columns
              if k.startswith("rep_pos_") or k in (
                  "rep_straddle_sat", "rep_straddle_con",
                  "rep_with_original")}
        for k in ("straddle_sat", "straddle_con", "with_original"):
            mb[k], ma[k] = f"{mb[k]:.1f}", f"{ma[k]:.1f}"
        w(f"Mean over the {len(both)} candidates passing before and after "
          "(both sets):")
        position_lines(w, mb, ma)
    w("  The concern: cancelling the corpus leak could pull the sets toward "
      "one end of a ladder")
    w("  and re-create v4's 'edited = duller' pattern; compare image A's "
      "rows before and after.")
    w("")

    w("7. HOW WELL THE TWO SETS MATCH (shipped)")
    w("-" * 78)
    per_type_table(main, w)
    w("")
    for b in ("A", "B"):
        X = info[b]["X"]
        cos = float((X[:, 0] * X[:, 1]).sum()
                    / np.sqrt((X[:, 0] ** 2).sum() * (X[:, 1] ** 2).sum()))
        w(f"  set {b}: info sat {(X[:, 0] ** 2).sum():7.2f}, con "
          f"{(X[:, 1] ** 2).sum():6.2f}, cos {cos:+.3f}, VIF "
          f"{1 / (1 - cos ** 2):.3f}, scenes {main[b].scene_id.nunique()}, "
          f"A on the left {int(main[b].a_is_left.sum())} of {len(main[b])}")
    w(f"  information A-B: saturation {met['info_sat_rel_diff']:.2%}, "
      f"contrast {met['info_con_rel_diff']:.2%}; worst per type "
      f"{met['info_type_worst_rel_diff']:.2%}")
    w(f"  KS distance A-B: |x_sat| {met['ks_sat']:.3f}, |x_con| "
      f"{met['ks_con']:.3f}; cos difference {met['cos_diff']:.3f}")
    w("")

    w("8. WHAT THE PLANNING WEIGHTS MEAN, AND THE THRESHOLDS")
    w("-" * 78)
    for b in ("A", "B"):
        cp, th = info[b]["cp"], info[b]["th"]
        w(f"  set {b}: typical informative |x_sat| {cp['sat'][0]:.3f} "
          f"({cp['sat'][1]} trials) -> P {cp['sat'][2]:.4f};  |x_con| "
          f"{cp['con'][0]:.3f} ({cp['con'][1]} trials) -> P "
          f"{cp['con'][2]:.4f}")
        w(f"         thresholds ({N_THRESHOLD_PEOPLE:,} random choosers, "
          f"{info[b]['sec']:.0f} s): saturation {th['upper'][0]:+.3f} / "
          f"{th['lower'][0]:+.3f}, contrast {th['upper'][1]:+.3f} / "
          f"{th['lower'][1]:+.3f}")
    w(f"  v4 interpretation: {V4_CHOICE_PROB[0]:.4f} / {V4_CHOICE_PROB[1]:.4f}"
      "; v4 thresholds +0.286 / +0.488 (180 trials, one list).")
    w("  The choice probabilities are reported, not checked: script 2 "
      "showed v5's differences differ from v4's.")
    w("")

    w("9. CONSISTENCY, CATCH AND PRACTICE")
    w("-" * 78)
    w(f"  consistency per set: {n_cons} ("
      + ", ".join(f"{t} {alloc[t]}" for t in MAIN_TYPES)
      + "), the same couples in A and B, sides swapped")
    for b in ("A", "B"):
        w(f"  catch, set {b}: " + ", ".join(
            f"{r.item_id} {r.scene_id.split('_000')[0]}"
            for r in catch[b].itertuples()))
    for label, df in prac.items():
        w(f"  practice {label}: " + ", ".join(
            f"{r.type} {r.scene_id.split('_000')[0]}"
            for r in df.itertuples()))
    w("")

    w("10. IMAGES")
    w("-" * 78)
    if man is not None:
        ds = (man.sat_mean - man.table_sat_mean).abs().max()
        dc = (man.contrast_local - man.table_contrast_local).abs().max()
        w(f"  {len(man)} images copied from {STIMULI_SRC}; checked against "
          f"{CHECKSUMS_FILE};")
        w(f"  largest re-measured difference: sat_mean {ds:.2e}, "
          f"contrast_local {dc:.2e}")
    if fails:
        w(f"  FAILURES ({len(fails)}):")
        for f in fails[:40]:
            w(f"    {f}")
        if len(fails) > 40:
            w(f"    ... and {len(fails) - 40} more (BUILD_FAILED.txt)")
    w("")

    w("11. CHECKS")
    w("-" * 78)
    cf = ck.frame()
    for row in cf.itertuples():
        w(f"  {row.status:10s} [{row.scope}] {row.check}"
          + (f"  ({row.detail})" if row.detail else ""))
    w("")
    w(f"  {int((cf.status == 'PASS').sum())} PASS, "
      f"{int((cf.status == 'FAIL').sum())} FAIL, "
      f"{int(cf.status.isin(['SKIPPED']).sum())} SKIPPED, "
      f"{int((~cf.status.isin(['PASS', 'FAIL', 'SKIPPED'])).sum())} "
      "NOTE / USED / NOT NEEDED")
    w("")
    w("12. NOTES")
    w("-" * 78)
    w("* Nothing here is ordered. Rows are sorted by type and couple; "
      "script 4 makes the orders.")
    w("* item_id is stable within this build: A001..., Ac01... "
      "(consistency), Ax01... (catch),")
    w("  P1_01... and P2_01... (practice before block 1 and 2). repeat_of "
      "names the original's item_id.")
    w("* 'couple' links each analysed trial to its matched partner in the "
      "other set (same scene,")
    w("  type and bin, paired in order of the type's driver).")
    w("* Every row carries pred_sat, pred_fog, a_is_left, N_corpus, N_orig "
      "and their parts, for scripts 6-9.")
    w("* candidates.csv: every candidate before (draw_*) and after (rep_*) "
      "repair; leak_by_type.csv: the")
    w("  leak by type for every candidate, and for the shipped practice and "
      "catch trials.")
    w("* BUILD_FAILED.txt is written when any check fails.")
    with open(f"{out_dir}/build_report.txt", "w") as f:
        f.write("\n".join(L) + "\n")


# ============================================================
# Cross-checks with scripts 1 and 2
# ============================================================

def check_script1(cap, p):
    """The pair table and the bin edges equal script 1's run."""
    rows = []
    d, why = resolve_run_dir(SCRIPT1_RUN_DIR)
    names = ("the pairs equal script 1's all_pairs.csv.gz (files, type, "
             "differences, normality terms, to its 8 significant digits)",
             f"the {THRESHOLD_SET} bin edges equal script 1's bin_edges.csv")
    if d is None:
        return [("input", n, "SKIPPED", why) for n in names]
    cols = ["file_a", "file_b", "type", "d_sat", "d_fog", "N_corpus",
            "N_orig"]
    f = f"{d}/all_pairs.csv.gz"
    if os.path.isfile(f):
        theirs = pd.read_csv(f, usecols=cols)
        j = p[cols].merge(theirs, on=["file_a", "file_b"], how="inner",
                          suffixes=("", "_s1"))
        # script 1 writes 8 significant digits (float_format "%.8g"), so a
        # value can differ by up to half a unit in its 8th digit
        rel = max(float(((j[c] - j[f"{c}_s1"]).abs()
                         / np.maximum(j[c].abs(), 1e-12)).max())
                  for c in cols[3:]) if len(j) else np.inf
        ok_v = all(np.allclose(j[c], j[f"{c}_s1"], rtol=1e-7, atol=1e-12)
                   for c in cols[3:]) if len(j) else False
        ok = (len(theirs) == len(p) == len(j)
              and (j.type == j.type_s1).all() and ok_v)
        rows.append(("input", names[0], bool(ok),
                     f"{d}: {len(theirs):,} vs {len(p):,} rows, "
                     f"{len(j):,} matched, largest relative difference "
                     f"{rel:.1e} (script 1 saves 8 significant digits)"))
    else:
        rows.append(("input", names[0], "SKIPPED", f"{f} not found"))
    f = f"{d}/bin_edges.csv"
    if os.path.isfile(f):
        be = pd.read_csv(f)
        be = be[be.threshold_set == THRESHOLD_SET]
        mine = cap.BIN_EDGES[THRESHOLD_SET]
        ok = len(be) == len(mine) and all(
            np.allclose(json.loads(r.bin_edges_sd), mine[r.type], atol=1e-12)
            for r in be.itertuples())
        rows.append(("input", names[1], bool(ok),
                     "; ".join(f"{t} {mine[t]}" for t in cap.TYPES)))
    else:
        rows.append(("input", names[1], "SKIPPED", f"{f} not found"))
    return rows


def check_script2(cand, size):
    """The unrepaired candidates' leak equals script 2's for the same draws
    (script 2 lists the draws that pass script 1's checks)."""
    name = ("candidates are script 2's draws (their leak equals script 2's "
            "normality_leak.csv)")
    d, why = resolve_run_dir(SCRIPT2_RUN_DIR)
    if d is None:
        return ("candidates", name, "SKIPPED", why)
    f = f"{d}/normality_leak.csv"
    if not os.path.isfile(f):
        return ("candidates", name, "SKIPPED", f"{f} not found")
    s2 = pd.read_csv(f)
    s2 = s2[(s2.tset == THRESHOLD_SET) & (s2["size"] == size)
            & (s2.mix == MIX) & (s2.draw < N_CANDIDATES)]
    if not len(s2):
        return ("candidates", name, "SKIPPED",
                f"{d} has no {THRESHOLD_SET} n={size} {MIX} draws")
    c = cand.set_index("candidate")
    worst, n = 0.0, 0
    for r in s2.itertuples():
        if r.draw not in c.index or not c.loc[r.draw, "draw_pass"]:
            worst = np.inf
            continue
        for k in ("b_sat", "b_con", "cos_sat", "cos_con"):
            v = c.loc[r.draw, f"draw_leak_{r.definition}_{k}_{r.set}"]
            worst = max(worst, abs(v - getattr(r, k)))
        n += 1
    return ("candidates", name, bool(worst <= 1e-9),
            f"{d}: {s2.draw.nunique()} draws, {n} rows compared, largest "
            f"difference {worst:.1e}")


def sets_bytes(sets):
    return {b: sets[b].drop(columns=["key"]).to_csv(index=False)
            for b in ("A", "B")}


# ============================================================
# Main
# ============================================================

def main():
    global _CAP
    t0 = time.time()
    print("v5b step 3: build the two block lists, normality leak cancelled")
    cap = load_module(CAPACITY_SCRIPT, "analyse_pair_capacity_v5b")
    apply_overrides(cap)
    v4 = load_module(V4_SIM_SCRIPT, "simulate_power_v4_n10_20")
    _CAP = cap
    check_settings(cap)
    nw = n_workers()
    p_log = [f"script 1   {CAPACITY_SCRIPT}  sha256 "
             f"{sha256_of(CAPACITY_SCRIPT)[:16]}",
             f"v4 sim     {V4_SIM_SCRIPT}  sha256 "
             f"{sha256_of(V4_SIM_SCRIPT)[:16]}",
             f"overrides  {CAPACITY_OVERRIDES or 'none'} (MANIFEST_CSV = "
             f"{MANIFEST_CSV})",
             f"stimuli    {STIMULI_SRC}"]
    s1_ck = cap.Checks()
    s1_log = []
    p = cap.load_pairs(s1_ck, s1_log)
    p_log += s1_log
    pool = cap.build_pool(p, THRESHOLD_SET, s1_ck, [])
    pools_t = {t: pool[pool.type == t].reset_index(drop=True)
               for t in cap.TYPES}
    scenes = sorted(p.scene_id.unique())
    cf = s1_ck.frame()
    if (cf.status == "FAIL").any():
        raise RuntimeError("Script 1's input checks fail:\n"
                           + cf[cf.status == "FAIL"].to_string())
    p_log.append(f"script 1's input checks: {len(cf)} run, all PASS or "
                 "SKIPPED")
    p_log.append(f"pool       {THRESHOLD_SET}: " + ", ".join(
        f"{t} {len(pools_t[t]):,}" for t in cap.TYPES) + " pairs")
    pre = [(r.scope if r.scope != "input" else "input",
            r.check, r.status, r.detail) for r in cf.itertuples()]
    pre += check_script1(cap, p)
    print(f"  pairs made; {nw} worker(s)")

    # --- stage 1 per size ---
    counts, s1s = {}, {}
    for size in BLOCK_SIZES:
        counts[size] = cap.largest_remainder(size, cap.TYPE_MIXES[MIX])
        s1s[size] = cap.stage1(pools_t, counts[size], scenes,
                               cap.MAX_PER_SCENE_PER_TYPE)
        if s1s[size]["status"] != "feasible":
            raise RuntimeError(f"n={size}: counts not feasible "
                               f"({s1s[size]['status']})")
    # the solver must accept one thread here after script 1's solves
    from scipy.optimize import Bounds, LinearConstraint, milp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        probe = milp(-np.ones(2), constraints=[LinearConstraint(
            np.ones((1, 2)), 0, 1)], integrality=np.ones(2),
            bounds=Bounds(0, 1), options={"threads": 1})
    if probe.x is None:
        raise RuntimeError("The solver (HiGHS) refused a one-thread solve "
                           f"after script 1's solves ({probe.message}).")
    payload = {"pools_t": pools_t, "scenes": scenes, "counts": counts,
               "s1": s1s}

    # --- candidates: draw and repair, in parallel ---
    # the shipped limits (all of TYPE_LIMITS) and, with COMPARE_LIMIT_STEPS,
    # each step of LIMIT_STEPS; a type tuple keeps TYPE_LIMITS' order
    full = tuple(TYPE_LIMITS)
    steps = []
    if COMPARE_LIMIT_STEPS:
        for name, ts in LIMIT_STEPS:
            steps.append((name, tuple(t for t in TYPE_LIMITS if t in ts)))
    if full not in [ts for _, ts in steps]:
        steps.append(("shipped", full))
    tc = time.time()
    tasks = [(s, d, "cells", ts) for _, ts in steps for s in BLOCK_SIZES
             for d in range(N_CANDIDATES)]
    print(f"  {len(tasks)} repairs: {N_CANDIDATES} candidates x "
          f"{len(BLOCK_SIZES)} sizes x {len(steps)} limit step(s)")
    res = run_tasks(tasks, payload, cap, nw, "repair")
    by = {(r["size"], r["candidate"], t[3]): r for r, t in zip(res, tasks)}
    wide = {}
    for size in BLOCK_SIZES:
        n_ok = sum(by[(size, d, full)]["rep_pass"]
                   for d in range(N_CANDIDATES))
        if n_ok < WIDE_MIN_PASSING:
            lst = [d for d in range(N_CANDIDATES)
                   if not by[(size, d, full)]["rep_pass"]
                   and by[(size, d, full)]["built"] in ("greedy", "fallback")]
            if lst:
                wide[size] = lst
    wtasks = [(s, d, "wide", full) for s in wide for d in wide[s]]
    if wtasks:
        print(f"  WIDE REPAIR for {len(wtasks)} candidates (fewer than "
              f"{WIDE_MIN_PASSING} passed at "
              f"{', '.join(f'n={s}' for s in wide)})")
        for r in run_tasks(wtasks, payload, cap, nw, "wide"):
            k = (r["size"], r["candidate"], full)
            r["cells_status"] = by[k].get("repair_status")
            by[k] = r
    cand_seconds = time.time() - tc

    xs, xc = pool.x_sat, pool.x_con
    pool_leak = {dfn: cap.leak(pool[f"N_{dfn}"], xs, xc)
                 for dfn in cap.NORMALITY_DEFS}
    run = {"workers": nw, "cand_seconds": cand_seconds,
           "pool_leak": pool_leak}

    # --- per size: the reproducibility check, then the package ---
    if nw > 1:
        _setup_worker(cap, payload)
    out = []
    for size in BLOCK_SIZES:
        results = {d: by[(size, d, full)] for d in range(N_CANDIDATES)}
        cand = cand_table([results[d] for d in range(N_CANDIDATES)])
        if "cells_status" not in cand:
            cand["cells_status"] = None
        R = {"cand": cand, "results": results, "counts": counts[size],
             "s1": s1s[size], "wide_used": size in wide,
             "wide_candidates": wide.get(size, []),
             "steps": ([(name, ts, cand if ts == full else cand_table(
                 [by[(size, d, ts)] for d in range(N_CANDIDATES)]))
                 for name, ts in steps] if COMPARE_LIMIT_STEPS else [])}
        chosen, _, _ = select(cand)
        if chosen["rep"] is not None:
            d = chosen["rep"]
            mode = results[d]["repair_mode"]
            again = _task((size, d, mode, full))
            if again["sets_rep"] is None:
                R["repro"] = (False, f"candidate {d} ({mode}): the rebuild "
                              f"gave no sets ({again['repair_status']})")
            else:
                a = sets_bytes(results[d]["sets_rep"])
                b = sets_bytes(again["sets_rep"])
                R["repro"] = (a == b, f"candidate {d} ({mode}), {nw} "
                              "worker(s) in the run")
        else:
            R["repro"] = ("SKIPPED", "no candidate passes")
        R["s2check"] = check_script2(cand, size)
        out.append(build_one(cap, v4, size, R, pools_t, scenes, p_log, pre,
                             run, t0))
    any_fail = False
    for out_dir, n_fail, fails in out:
        print(f"  {out_dir}: {n_fail} check(s) FAIL")
        any_fail |= bool(n_fail) or bool(fails)
    print(f"  workers {nw}; candidates {cand_seconds:.0f} s; elapsed "
          f"{time.time() - t0:.0f} s")
    if any_fail:
        print("\n  BUILD FAILED: see the report(s) above. A folder with "
              "BUILD_FAILED.txt must not be used.")
        sys.exit(1)


if __name__ == "__main__":
    main()
