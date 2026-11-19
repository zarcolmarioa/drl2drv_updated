"""
preview_pair_rules_r3_v5b.py  --  v5b: pair preview round 3: final rules and
a leak check

WHAT THIS DOES
    The third round of the pair preview (round 1: preview_pair_rules_v5b.py,
    run f8c2aa13ca; round 2: preview_pair_rules_r2_v5b.py, run b5493951ff).
    The pair typing, the gallery and the summary mode are as in round 2; the
    rules, the groups and the LEAK CHECK are new.

    1. LEAK CHECK (the main part; indicative: build_trials_v5b.py is the real
       test). For the final rules under every combination of three levers
       (COLOUR_ONLY_ALLOW_HIGH_3, COLOUR_ONLY_ALLOW_LOW_2,
       CONTRAST_ONLY_ALLOW_LOW_2) and, as a reference, the current v5b_180
       pool ("v5units": the current minimums, no level rules, no caps):
         - pool leak: the no-intercept regression of N on x_sat and x_con
           over every candidate pair, per type and overall, both normality
           definitions, and the share of pairs where image A is the more
           typical one;
         - achievable leak: LEAK_N_DRAWS draws of two disjoint sets (A, B)
           like the real ones (54 colour only, 54 contrast only, 36 together,
           36 conflict; within a type every scene once before any twice),
           then a simple swap step (one pair at a time, within the same type
           and number of levels apart, scenes capped as in the draw) that
           lowers the worst |b| relative to its limit; how often each limit
           is met after 0, 10, 20, 40, 80 and 150 swaps per set, the swaps
           needed, and the best |b| reached. The reference's results are
           the yardstick: the shipped lists did meet the limits from it;
         - what drives the leak: the cells (type x levels apart x where the
           pair sits) that pull each b away from 0, and the pairs (allowed,
           just excluded, or the fallback's) that would help most;
         - the best sets found: their pairs over the cells, and the pairs
           the swap step brought in;
         - a plain-language conclusion at the top of report.txt.
       The leak function is copied from analyse_pair_capacity_v5b.py
       (leak(), the function build_trials_v5b.py uses), and is checked by
       reproducing the shipped v5b_180 leaks from its items.csv.

    2. GALLERY (default levers): per type 15 typical pairs (spread over
       distance, where the pair sits and position), no odd pairs, 10 just-
       excluded pairs (near the final boundaries, split evenly across
       EXCLUDED_CATEGORIES), and 12 anchors (6 from round 1, rated in both
       rounds 1 and 2: those that agreed first, then some that did not; 6
       from round 2: 3 "Too similar" and 3 "Fine"), unmarked in the gallery.
       Labels stay hidden until "hide labels" is switched off, and then
       show only on rated pairs.

FINAL RULES (levels nominal, -3 to +3; "lower"/"higher" = the two images'
levels on the feature that changes; measured differences in corpus SD)
    colour only    3-4 levels apart with the lower image at -1 or below; or
                   (COLOUR_ONLY_ALLOW_LOW_2) 2 levels apart with both images
                   at or below 0; or (COLOUR_ONLY_ALLOW_HIGH_3, off) 3 levels
                   apart with both at or above 0 (0 vs +3); >= 0.30 SD
                   measured; the same nominal contrast level, within +-2
    contrast only  3-4 levels apart, anywhere; or (CONTRAST_ONLY_ALLOW_LOW_2)
                   2 levels apart with the higher image at -1 or below;
                   >= 0.45 SD measured; the same colour level, within +-2
    together,      exactly 2/2 levels (colour/contrast); >= 0.20 SD colour,
    conflict       >= 0.32 SD contrast, <= 1.2 SD each (measured); conflict
                   balanced; at most one feature reaches +-3
    Where a pair sits (on the changing feature; on contrast for mixed
    pairs): "low" both images at or below 0, "high" both at or above 0,
    otherwise "straddling". Position: touching +-3 or middle (as before).

THE LEAK (as build_trials_v5b.py)
    Each pair is oriented so that image A is the one the hypothesis
    predicts by the type's driver: the less saturated in colour-only pairs,
    the lower-contrast in the other types (in conflict pairs, the more
    saturated). x_sat = -(sat_A - sat_B) / SD_SAT, x_con = -(con_A - con_B) /
    SD_FOG; N = dist(B) - dist(A) (positive: A is the more typical), with
    dist the manifest's dist_corpus or dist_orig. Limits (per set, both
    definitions): overall |b_sat|, |b_con| <= 0.05 (both terms, no
    intercept); colour only on x_sat alone <= 0.05; contrast only on x_con
    alone <= 0.05; conflict on x_con alone <= 0.10; together not limited.

EXCLUSION (EXCLUDE_SCENES, EXCLUDE_IMAGES, EXCLUDE_REASON; also on the
command line: --exclude-scenes, --exclude-images, --exclude-reason; the
same names can be added to analyse_pair_capacity_v5b.py)
    Scenes (all 49 images) and single image files are left out of every
    pool, list and gallery. With both lists empty the run is exactly the
    base run (BASE_RUN, 2883e744a4): the same settings hash and the same
    files (only the run line of report.txt and run_meta.json differ). With
    an exclusion the hash changes, and the script also computes everything
    without it ("before"), checks that this equals BASE_RUN file by file,
    and reports before and after: the scenes left out (category,
    indoor/outdoor, categories that disappear), supply, pool and achievable
    leak (the same draws and seeds), scene reuse in the lists, and the
    gallery pairs replaced. The gallery keeps every pair of the base run
    that is not affected, with its side and number; each affected pair is
    replaced, with the seed, by one of the same type and group (typical:
    the same spread cell; excluded: the same category; anchors: the same
    source round, rating kind and type). An unknown scene or file name
    stops the script before anything is written.

SCENE CHECK (optional output, --scene-check)
    scene_check.html (same buttons, download and hidden labels as the
    gallery) with 6 pairs under the final rules for each scene in
    SCENE_CHECK_SCENES (one per type, plus one more colour-only and one more
    contrast-only pair at another distance) and for
    SCENE_CHECK_N_COMPARISON comparison scenes drawn with the seed from the
    remaining scenes, matched for indoor/outdoor; scene_check.csv lists
    them. The rule, fixed before any rating: a scene is excluded if
    SCENE_CHECK_RULE_MIN or more of its 6 pairs are rated "Too similar".
    Summary mode reports each scene's count against it.

COPIED (with source comments): from analyse_pair_capacity_v5b.py SD_SAT,
SD_FOG, FOG_SIGN, SAT_TOL_SD, CON_TOL_SD, BALANCE_LO, BALANCE_HI, the
"v5units" minimums, NORMALITY_DEFS, EPS, Streams, classify, make_pairs,
leak; from build_trials_v5b.py LEAK_LIMIT and TYPE_LIMITS.

SUMMARY MODE
    python preview_pair_rules_r3_v5b.py --ratings <csv> [<csv> ...]
    As round 2 (each file's round found from its settings hash or pair
    keys), with round-3 definitions for every round: each round on its own
    and all rounds together; colour only and contrast only by distance x
    where the pair sits (low / straddling / high), mixed types by level
    combination; excluded pairs by rule; odd pairs by reason (rounds 1-2);
    the anchors' ratings in every round they were shown, and agreement.
    Anchor repeats are left out of the pooled counts. Ratings of excluded
    scenes or images are left out of the main tables and reported in their
    own, per scene and round, split by whether the pair meets the final
    rules, with the remaining scenes for comparison and every scene's share
    of "Too similar" (ratings_by_scene.csv). Scene-check ratings are scored
    against the rule.

CHECKS (report.txt; the script stops on a failure)
    as round 2, plus: the leak function reproduces the shipped v5b_180
    leaks (FROZEN.json); the reference pool reproduces the current counts;
    every drawn set has the right size, comes from its pool, and A and B are
    disjoint; the leak check is reproducible; typical pairs meet the final
    rules and excluded pairs break exactly what their category replaces;
    the anchors are found in the round-1 and round-2 files (which belong to
    their galleries); the settings hash differs from rounds 1 and 2, whose
    folders are untouched.

INPUTS (relative to the repository root)
    MANIFEST_CSV, its folder's checksums.sha256, FROZEN.json and stimuli/;
    V5B180_DIRS: the first that holds v5b_180's FROZEN.json and
    package/items.csv (the freeze script's output under data/, or its copy
    in the repository);
    --anchors: the round-1 and round-2 ratings files, in that order;
    ROUND1_GALLERY, ROUND2_GALLERY (for which pairs were anchors, and checks)

OUTPUTS: OUTPUT_ROOT/<hash of the settings>/ (a rerun: <hash>_run2, ...)
    report.txt (conclusion first), gallery.html, images/, examples.csv,
    supply.csv, leak_pool.csv (pool leak), leak_cells.csv (each cell's
    pull), leak_helpers.csv and leak_helper_cells.csv (pairs and cells that
    would help), leak_draws.csv (every set, before and after the swaps, met
    at each swap budget), leak_achievable.csv (per rule set),
    leak_feasible_cells.csv (cells in sets that met every limit),
    leak_best_cells.csv and leak_best_swaps.csv (the best draws), checks.csv,
    settings.json, run_meta.json

REQUIRES  Python 3.9+, numpy, pandas
RUN       python v5b/trial_design_pipeline/preview_pair_rules_r3_v5b.py \
              --anchors <round-1 ratings csv> <round-2 ratings csv>
          [--exclude-scenes S ...] [--exclude-images F ...] [--scene-check]
          (from the repository root; about 1.5 minutes, 3 with an
          exclusion, mostly the leak check. With "--exclude-scenes" and
          "--exclude-images" given without names, both lists are empty and
          the run is the base run.)
"""

import argparse
import difflib
import hashlib
import html
import itertools
import json
import math
import os
import platform
import re
import shutil
import sys
import textwrap
import time

import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

MANIFEST_CSV = "data/stimulus_set_v5/7fd72262ba/stimulus_manifest.csv"
OUTPUT_ROOT = "data/pair_preview_v5b"
V5B180_DIRS = ["data/frozen_trial_lists_v5b/v5b_180",
               "v5b/trial_design_data/frozen_trial_lists_v5b/v5b_180"]
ROUND = 3
ROUND1_HASH = "f8c2aa13ca"
ROUND2_HASH = "b5493951ff"
ROUND1_GALLERY = "data/pair_preview_v5b/f8c2aa13ca"
ROUND2_GALLERY = "data/pair_preview_v5b/b5493951ff"
RANDOM_SEED = 42

# --- Copied from analyse_pair_capacity_v5b.py (pair typing; unchanged) ---
SD_SAT = 39.182
SD_FOG = 9.348
FOG_SIGN = -1.0
SAT_TOL_SD = 3.0 / SD_SAT
CON_TOL_SD = 0.72 / SD_FOG
BALANCE_LO = 0.6
BALANCE_HI = 1.7
CURRENT_MIN = {"sat_only": 31.1 / SD_SAT, "fog_only": 4.2 / SD_FOG,
               "mixed_sat": 8.0 / SD_SAT, "mixed_con": 3.0 / SD_FOG}
NORMALITY_DEFS = ("corpus", "orig")
EXPECTED_CURRENT = {"sat_only": 3290, "fog_only": 3976, "together": 20283,
                    "conflict": 8571}

# --- Final rules ---
MAX_LEVEL = 3
COLOUR_ONLY_LEVELS = (3, 4)              # levels apart, inclusive ...
COLOUR_ONLY_LOWER_MAX = -1               # ... with the lower image at or below
COLOUR_ONLY_ALLOW_LOW_2 = True           # 2 apart, both at or below:
COLOUR_ONLY_LOW_2_MAX_HIGH = 0
COLOUR_ONLY_ALLOW_HIGH_3 = False         # fallback: 3 apart, both >= 0
COLOUR_ONLY_MIN_SD = 0.30
COLOUR_ONLY_SHARED_CONTRAST_MAX = 2      # the shared contrast level, +-
CONTRAST_ONLY_LEVELS = (3, 4)            # anywhere
CONTRAST_ONLY_ALLOW_LOW_2 = True         # 2 apart, higher image at or below:
CONTRAST_ONLY_LOW_2_MAX_HIGH = -1
CONTRAST_ONLY_MIN_SD = 0.45
CONTRAST_ONLY_SHARED_COLOUR_MAX = 2
TOGETHER_LEVELS = [(2, 2)]               # (colour, contrast) levels apart
CONFLICT_LEVELS = [(2, 2)]
MIXED_MIN_SAT_SD = CURRENT_MIN["mixed_sat"]  # 0.20 SD
MIXED_MIN_CON_SD = CURRENT_MIN["mixed_con"]  # 0.32 SD
MIXED_MAX_SD = 1.2
CONFLICT_BALANCED = True

# Candidate pairs under the default rules (computed from the manifest and
# reported before this script was written; checked)
EXPECTED_CANDIDATES = {"sat_only": 1880, "fog_only": 1644, "together": 987,
                       "conflict": 987}

# --- Leak check ---
# Copied from build_trials_v5b.py: LEAK_LIMIT (overall |b_sat|, |b_con|)
# and TYPE_LIMITS (type -> (term, limit)); together is not limited.
LEAK_LIMIT = 0.05
TYPE_LIMITS = {"sat_only": ("sat", 0.05),
               "fog_only": ("con", 0.05),
               "conflict": ("con", 0.10)}
LEAK_SET_SIZES = {"sat_only": 54, "fog_only": 54, "together": 36,
                  "conflict": 36}
LEAK_N_DRAWS = 100               # per rule set; each draw = sets A and B
LEAK_MAX_SWAPS = 150             # per set (of 180 pairs)
LEAK_TRY_MEMBERS = None          # pairs tried for removal per swap (None: all)
LEAK_SWAP_TARGET = 0.8           # stop once every |b| <= 0.8 x its limit
                                 # (build_trials' REPAIR_TARGET 0.04 of 0.05)
LEAK_BUDGETS = (0, 10, 20, 40, 80, 150)   # swaps per set, reported
LEAK_MOST = 0.5                  # "most draws": share of draws, both sets
LEAK_N_HELPERS = 15              # pairs listed per type in leak_helpers.csv
LEVERS = ["COLOUR_ONLY_ALLOW_HIGH_3", "COLOUR_ONLY_ALLOW_LOW_2",
          "CONTRAST_ONLY_ALLOW_LOW_2"]

# --- Just excluded: categories near the final boundaries ---
# Each pair meets every final condition except those in "replaces".
# Keys: "levels" (levels apart of the changing feature), "where" (where the
# pair sits), "high_eq" (the higher-contrast image at this level), "combos"
# ((colour, contrast) levels apart).
EXCLUDED_CATEGORIES = {
    "sat_only": [
        {"name": "2 levels straddling or above 0", "replaces": ["levels"],
         "levels": [2], "where": ["straddling", "high"]},
        {"name": "0 vs +3", "replaces": ["levels"], "levels": [3],
         "where": ["high"]},
        {"name": "5 levels apart", "replaces": ["levels"], "levels": [5]}],
    "fog_only": [
        {"name": "2 levels, higher image at 0", "replaces": ["levels"],
         "levels": [2], "high_eq": 0},
        {"name": "5 levels apart", "replaces": ["levels"], "levels": [5]}],
    "together": [
        {"name": "2/1", "replaces": ["levels"], "combos": [(2, 1)]},
        {"name": "1/2", "replaces": ["levels"], "combos": [(1, 2)]},
        {"name": "3/2", "replaces": ["levels"], "combos": [(3, 2)]}],
    "conflict": [
        {"name": "2/1", "replaces": ["levels"], "combos": [(2, 1)]},
        {"name": "1/2", "replaces": ["levels"], "combos": [(1, 2)]},
        {"name": "3/2", "replaces": ["levels"], "combos": [(3, 2)]}]}

# --- The gallery ---
N_TYPICAL = 15
N_ODD = 0
N_EXCLUDED = 10
ANCHORS_FROM_R1 = {"agree": 4, "disagree": 2}   # rated in rounds 1 and 2
ANCHORS_FROM_R2 = {"Too similar": 3, "Fine": 3}  # rated once, in round 2
TYPICAL_DIMS = {"sat_only": ["levels", "where", "position"],
                "fog_only": ["levels", "where", "position"],
                "together": ["position", "where"],
                "conflict": ["position", "where"]}
ODDNESS_WEIGHTS = {"abs_z_sat": 1.0, "abs_z_con": 1.0, "comp_boost": 1.0,
                   "clip_2lv": 1.0, "abs_colour_dev": 1.0,
                   "abs_contrast_dev": 1.0, "dist_orig": 1.0,
                   "envelope_n": 1.0}
IMAGE_PX = 512
GAP_PX = 32
BACKGROUND = "#7f7f7f"
RATINGS = ["Fine", "Too similar", "Too different", "Looks odd"]
DEFAULT_ORDER = "shuffled"

# --- Exclusion (the same names can be added to analyse_pair_capacity_v5b.py)
# Scenes (all 49 images) and single image files left out of every pool, list
# and gallery. With both lists empty the run is the base run (BASE_RUN).
EXCLUDE_SCENES = ["bathroom_00001909", "dining_hall_00001273",
                  "gymnasium_indoor_00003333", "shower_00000779", "auditorium_00001394"]
EXCLUDE_IMAGES = []
EXCLUDE_REASON = ("edits look too similar to the unedited photo and to each "
                  "other in the preview galleries (rounds 1-3)")
BASE_RUN = "data/pair_preview_v5b/2883e744a4"    # the run without exclusion

# --- Scene check (optional output: --scene-check) ---
SCENE_CHECK_SCENES = ["bathroom_00001909", "dining_hall_00001273",
                      "gymnasium_indoor_00003333", "shower_00000779",
                      "auditorium_00001394"]
SCENE_CHECK_N_COMPARISON = 5         # matched for indoor/outdoor
SCENE_CHECK_EXTRA_TYPES = ["sat_only", "fog_only"]   # the 5th and 6th pair
SCENE_CHECK_RULE_MIN = 4             # of 6 rated "Too similar": exclude
                                     # (fixed before any rating)

# ============================================================
# Code
# ============================================================

TYPES = ["sat_only", "fog_only", "together", "conflict"]
TYPE_LABEL = {"sat_only": "colour only", "fog_only": "contrast only",
              "together": "together", "conflict": "conflict"}
GROUPS = ["typical"] + (["odd"] if N_ODD else []) + ["excluded"]
SELECTED = GROUPS + ["anchor"]
WHERE = ["low", "straddling", "high"]
EPS = 1e-9          # copied from analyse_pair_capacity_v5b.py
COMPONENT_LABEL = {
    "abs_z_sat": "saturation far from the corpus (|z_sat|)",
    "abs_z_con": "contrast far from the corpus (|z_con|)",
    "comp_boost": "chroma boost after haze (log comp_k > 0)",
    "clip_2lv": "clipping (clip_2lv)",
    "abs_colour_dev": "colour off its nominal level (|colour_dev_sd|)",
    "abs_contrast_dev": "contrast off its nominal level (|contrast_dev_sd|)",
    "dist_orig": "far from the unedited photograph (dist_orig)",
    "envelope_n": "envelope violations (envelope_n)"}
SCENE_COLS = ["scene_id", "category", "group", "indoor"]
SIDE_COLS = {"filename": "file", "sat_mean": "sat",
             "contrast_local": "fog", "contrast_level": "contrast_level",
             "colour_level": "colour_level", "contrast_step":
             "contrast_step", "colour_step": "colour_step",
             "is_original": "is_original"}
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


SETTING_KEYS = [
    "MANIFEST_CSV", "OUTPUT_ROOT", "V5B180_DIRS", "ROUND", "ROUND1_HASH",
    "ROUND2_HASH", "ROUND1_GALLERY", "ROUND2_GALLERY", "RANDOM_SEED",
    "SD_SAT", "SD_FOG", "FOG_SIGN", "SAT_TOL_SD", "CON_TOL_SD", "BALANCE_LO",
    "BALANCE_HI", "CURRENT_MIN", "NORMALITY_DEFS", "EXPECTED_CURRENT",
    "MAX_LEVEL", "COLOUR_ONLY_LEVELS", "COLOUR_ONLY_LOWER_MAX",
    "COLOUR_ONLY_ALLOW_LOW_2", "COLOUR_ONLY_LOW_2_MAX_HIGH",
    "COLOUR_ONLY_ALLOW_HIGH_3", "COLOUR_ONLY_MIN_SD",
    "COLOUR_ONLY_SHARED_CONTRAST_MAX", "CONTRAST_ONLY_LEVELS",
    "CONTRAST_ONLY_ALLOW_LOW_2", "CONTRAST_ONLY_LOW_2_MAX_HIGH",
    "CONTRAST_ONLY_MIN_SD", "CONTRAST_ONLY_SHARED_COLOUR_MAX",
    "TOGETHER_LEVELS", "CONFLICT_LEVELS", "MIXED_MIN_SAT_SD",
    "MIXED_MIN_CON_SD", "MIXED_MAX_SD", "CONFLICT_BALANCED",
    "EXPECTED_CANDIDATES", "LEAK_LIMIT", "TYPE_LIMITS", "LEAK_SET_SIZES",
    "LEAK_N_DRAWS", "LEAK_MAX_SWAPS", "LEAK_TRY_MEMBERS",
    "LEAK_SWAP_TARGET", "LEAK_BUDGETS", "LEAK_MOST", "LEAK_N_HELPERS", "LEVERS",
    "EXCLUDED_CATEGORIES", "N_TYPICAL", "N_ODD", "N_EXCLUDED",
    "ANCHORS_FROM_R1", "ANCHORS_FROM_R2", "TYPICAL_DIMS", "ODDNESS_WEIGHTS",
    "IMAGE_PX", "GAP_PX", "BACKGROUND", "RATINGS", "DEFAULT_ORDER"]


SCENE_CHECK_RUN = False      # set by --scene-check (not a setting)


def exclusion_active():
    return bool(EXCLUDE_SCENES or EXCLUDE_IMAGES)


def settings_dict(with_exclusion=True):
    """The settings; the exclusion lists are added only when not empty, so
    that with both lists empty the hash is the base run's."""
    g = globals()
    d = {k: g[k] for k in SETTING_KEYS}
    if with_exclusion and exclusion_active():
        d["EXCLUDE_SCENES"] = sorted(EXCLUDE_SCENES)
        d["EXCLUDE_IMAGES"] = sorted(EXCLUDE_IMAGES)
    return d


def settings_hash(with_exclusion=True):
    blob = json.dumps(settings_dict(with_exclusion), sort_keys=True,
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def validate_exclusion(m):
    """Unknown or repeated scene and file names (a list of messages)."""
    errs = []
    scenes, files = set(m.scene_id), set(m.filename)
    lists = [("EXCLUDE_SCENES", "scene", EXCLUDE_SCENES, scenes),
             ("EXCLUDE_IMAGES", "image file", EXCLUDE_IMAGES, files)]
    if SCENE_CHECK_RUN:
        lists.append(("SCENE_CHECK_SCENES", "scene", SCENE_CHECK_SCENES,
                      scenes))
    for setting, kind, given, known in lists:
        for x in given:
            if x not in known:
                near = difflib.get_close_matches(x, sorted(known), n=3)
                errs.append(f"{setting}: unknown {kind} '{x}'"
                            + (f" (close: {', '.join(near)})" if near
                               else ""))
        dup = sorted({x for x in given if list(given).count(x) > 1})
        if dup:
            errs.append(f"{setting}: listed twice: {', '.join(dup)}")
    return errs


def excluded_mask(df, scene="scene_id", files=("file_a", "file_b")):
    m = df[scene].isin(EXCLUDE_SCENES)
    for f in files:
        m |= df[f].isin(EXCLUDE_IMAGES)
    return m


def run_folders(h):
    base = os.path.join(OUTPUT_ROOT, h)
    out = [base] if os.path.isdir(base) else []
    k = 2
    while os.path.isdir(f"{base}_run{k}"):
        out.append(f"{base}_run{k}")
        k += 1
    return out


def new_output_dir(h):
    base = os.path.join(OUTPUT_ROOT, h)
    if not os.path.exists(base):
        return base
    k = 2
    while os.path.exists(f"{base}_run{k}"):
        k += 1
    return f"{base}_run{k}"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def folder_digest(path):
    if not os.path.isdir(path):
        return None
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for fn in sorted(files):
            p = os.path.join(root, fn)
            h.update(os.path.relpath(p, path).encode() + b"\0")
            h.update(sha256_file(p).encode())
    return h.hexdigest()


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, section, what, ok, detail=""):
        status = "PASS" if bool(ok) else "FAIL"
        self.rows.append({"section": section, "check": what,
                          "status": status, "detail": str(detail)})
        log(f"{status}  [{section}] {what}  {detail}")
        return bool(ok)

    @property
    def n_fail(self):
        return sum(r["status"] == "FAIL" for r in self.rows)


class Streams:
    """Named random generators (copied from analyse_pair_capacity_v5b.py):
    each name gets its own generator, seeded from the SHA-256 of the key."""

    def __init__(self, seed, *key):
        self.seed = int(seed)
        self.key = "|".join(str(k) for k in key)
        self._made = {}

    def __call__(self, name):
        if name not in self._made:
            tag = int.from_bytes(hashlib.sha256(
                f"{self.key}|{name}".encode()).digest()[:8], "big")
            self._made[name] = np.random.default_rng([self.seed, tag])
        return self._made[name]


def as_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


# ------------------------------------------------------------
# Pair typing and the leak (copied from analyse_pair_capacity_v5b.py)
# ------------------------------------------------------------

def classify(d_sat, d_fog):
    """Copied from analyse_pair_capacity_v5b.py: type, |differences| in
    corpus SD and balance of ordered pairs, from MEASURED differences."""
    zs = np.abs(d_sat) / SD_SAT
    zc = np.abs(d_fog) / SD_FOG
    same_s = zs <= SAT_TOL_SD
    same_c = zc <= CON_TOL_SD
    mixed = np.where((d_fog > 0) == (d_sat > 0), "conflict", "together")
    t = np.select([same_s & same_c, same_c, same_s],
                  ["none", "sat_only", "fog_only"], default=mixed)
    with np.errstate(divide="ignore", invalid="ignore"):
        bal = np.where(zc > 0, zs / zc, np.nan)
    balanced = np.isfinite(bal) & (bal >= BALANCE_LO) & (bal <= BALANCE_HI)
    return t, zs, zc, bal, balanced


def make_pairs(m):
    """Copied from analyse_pair_capacity_v5b.py (the columns this script
    needs, plus the normality distances): every pair of images within a
    scene, both directions, typed by measured differences."""
    out = []
    for sid, g in m.groupby("scene_id", sort=True):
        g = g.sort_values(["contrast_step", "colour_step"], kind="stable")
        n = len(g)
        i, j = np.triu_indices(n, 1)
        ia = np.r_[i, j]
        ib = np.r_[j, i]
        cols = {c: np.repeat(g[c].iloc[0], len(ia)) for c in SCENE_COLS}
        for c, nm in SIDE_COLS.items():
            v = g[c].to_numpy()
            cols[f"{nm}_a"] = v[ia]
            cols[f"{nm}_b"] = v[ib]
        for d in NORMALITY_DEFS:
            v = g[f"dist_{d}"].to_numpy()
            cols[f"dist_{d}_a"] = v[ia]
            cols[f"dist_{d}_b"] = v[ib]
        out.append(pd.DataFrame(cols))
    p = pd.concat(out, ignore_index=True)
    p["d_sat"] = p.sat_a - p.sat_b
    p["d_fog"] = FOG_SIGN * (p.fog_a - p.fog_b)
    p["pred_sat"] = -p.d_sat / SD_SAT
    p["pred_fog"] = p.d_fog / SD_FOG
    t, zs, zc, bal, bd = classify(p.d_sat.to_numpy(), p.d_fog.to_numpy())
    p["type"] = t
    p["z_sat"] = zs
    p["z_fog"] = zc
    p["balance"] = bal
    p["balanced"] = bd
    p["d_contrast_steps"] = p.contrast_step_a - p.contrast_step_b
    p["d_colour_steps"] = p.colour_step_a - p.colour_step_b
    # normality: positive = A is the more typical image (copied)
    for d in NORMALITY_DEFS:
        p[f"N_{d}"] = p[f"dist_{d}_b"] - p[f"dist_{d}_a"]
    return p


def leak(N, xs, xc, terms=("sat", "con")):
    """Copied from analyse_pair_capacity_v5b.py (used by
    build_trials_v5b.py): the normality leak of a set of oriented pairs: N
    regressed on the direction terms together, no intercept (N ~ b_sat
    x_sat + b_con x_con), and the uncentred correlations of N with each
    term. `terms` = ("sat",) or ("con",) fits one term only."""
    N, xs, xc = (np.asarray(v, float) for v in (N, xs, xc))
    out = {"b_sat": np.nan, "b_con": np.nan, "cos_sat": np.nan,
           "cos_con": np.nan}
    if len(N) < 2:
        return out
    cols = {"sat": xs, "con": xc}
    X = np.column_stack([cols[t] for t in terms])
    b, *_ = np.linalg.lstsq(X, N, rcond=None)
    nn = np.sqrt((N ** 2).sum())
    for t, v in zip(terms, b):
        out[f"b_{t}"] = float(v)
    if nn > 0:
        out["cos_sat"] = float((N * xs).sum() / (nn * np.sqrt((xs ** 2).sum())))
        out["cos_con"] = float((N * xc).sum() / (nn * np.sqrt((xc ** 2).sum())))
    return out


# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------

def read_manifest():
    m = pd.read_csv(MANIFEST_CSV)
    m["is_original"] = as_bool(m["is_original"])
    return m


def read_inputs(ck):
    d = os.path.dirname(MANIFEST_CSV)
    ok = os.path.isfile(MANIFEST_CSV)
    ck.add("inputs", "the stimulus manifest exists", ok, MANIFEST_CSV)
    if not ok:
        return None, None
    m = read_manifest()
    need = (SCENE_COLS + list(SIDE_COLS)
            + ["z_sat", "z_con", "comp_k", "clip_2lv", "colour_dev_sd",
               "contrast_dev_sd", "dist_orig", "envelope_n"]
            + [f"dist_{x}" for x in NORMALITY_DEFS])
    miss = sorted(set(c for c in need if c not in m.columns))
    ck.add("inputs", "the manifest has the columns used", not miss,
           f"missing {miss}" if miss else f"{len(m.columns)} columns")
    if miss:
        return None, None
    per = m.groupby("scene_id").size()
    ck.add("inputs", "47 scenes x 49 images, file names unique",
           len(per) == 47 and (per == 49).all() and m.filename.is_unique,
           f"{len(per)} scenes, {len(m):,} images")
    cs = os.path.join(d, "checksums.sha256")
    sums = {}
    if os.path.isfile(cs):
        with open(cs) as f:
            for line in f:
                if line.strip():
                    h, rel = line.strip().split(None, 1)
                    sums[rel.lstrip("*").strip()] = h
    ck.add("inputs", "checksums.sha256 exists", bool(sums), cs)
    man_sha = sha256_file(MANIFEST_CSV)
    ck.add("inputs", "the manifest matches checksums.sha256",
           sums.get("stimulus_manifest.csv") == man_sha, man_sha[:16] + "...")
    fz = os.path.join(d, "FROZEN.json")
    if os.path.isfile(fz):
        with open(fz) as f:
            fr = json.load(f)
        ck.add("inputs", "the manifest is the frozen stimulus set's "
               "(FROZEN.json)", fr.get("manifest_sha256") == man_sha,
               f"set {fr.get('settings_hash')}")
    stim = os.path.join(d, "stimuli")
    missing, bad = [], []
    for fn in m.filename:
        p = os.path.join(stim, fn)
        if not os.path.isfile(p):
            missing.append(fn)
        elif sha256_file(p) != sums.get(f"stimuli/{fn}"):
            bad.append(fn)
    ck.add("inputs", f"every image file exists ({len(m):,})", not missing,
           f"{len(missing)} missing" + (f", e.g. {missing[:3]}"
                                        if missing else ""))
    ck.add("inputs", f"all {len(m):,} images match checksums.sha256",
           not missing and not bad, f"{len(bad)} differ"
           + (f", e.g. {bad[:3]}" if bad else ""))
    return m, {"manifest_sha256": man_sha, "stimuli_dir": stim}


# ------------------------------------------------------------
# Pair fields, orientation, rules
# ------------------------------------------------------------

def where_of(a, b):
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return np.where(hi <= 0, "low", np.where(lo >= 0, "high", "straddling"))


def add_pair_fields(u):
    """Distances, +-3 flags, position, levels, shared level, where the pair
    sits, pair key, and the leak orientation (A = the image predicted by
    the type's driver, as build_trials_v5b.py)."""
    u = u.copy()
    u["pair_key"] = u.file_a + "__vs__" + u.file_b     # file_a < file_b
    u["dcol"] = u.d_colour_steps.abs()
    u["dcon"] = u.d_contrast_steps.abs()
    u["col3"] = ((u.colour_step_a.abs() == MAX_LEVEL)
                 | (u.colour_step_b.abs() == MAX_LEVEL))
    u["con3"] = ((u.contrast_step_a.abs() == MAX_LEVEL)
                 | (u.contrast_step_b.abs() == MAX_LEVEL))
    u["lo_col"] = np.minimum(u.colour_step_a, u.colour_step_b)
    u["hi_col"] = np.maximum(u.colour_step_a, u.colour_step_b)
    u["lo_con"] = np.minimum(u.contrast_step_a, u.contrast_step_b)
    u["hi_con"] = np.maximum(u.contrast_step_a, u.contrast_step_b)
    is_s, is_f = u.type == "sat_only", u.type == "fog_only"
    touch = np.where(is_s, u.col3, np.where(is_f, u.con3, u.col3 | u.con3))
    u["position"] = np.where(touch, "touching +-3", "middle")
    u["levels"] = np.where(
        is_s, u.dcol.astype(str), np.where(
            is_f, u.dcon.astype(str),
            u.dcol.astype(str) + "/" + u.dcon.astype(str)))
    shared = np.where(is_s & (u.dcon == 0), u.contrast_step_a,
                      np.where(is_f & (u.dcol == 0), u.colour_step_a,
                               np.nan))
    u["shared_level"] = pd.Series(shared, index=u.index).map(
        lambda x: "" if pd.isna(x) else f"{int(x):+d}".replace("+0", "0"))
    u["where"] = np.where(is_s, where_of(u.colour_step_a, u.colour_step_b),
                          where_of(u.contrast_step_a, u.contrast_step_b))
    # leak orientation (build_trials_v5b.py / analyse_pair_capacity build_pool)
    flip = np.where(is_s, u.pred_sat < 0, u.pred_fog < 0)
    sgn = np.where(flip, -1.0, 1.0)
    u["x_sat"] = u.pred_sat * sgn
    u["x_con"] = u.pred_fog * sgn
    for d in NORMALITY_DEFS:
        u[f"N_{d}"] = u[f"N_{d}"] * sgn
    u["file_A"] = np.where(flip, u.file_b, u.file_a)
    u["file_B"] = np.where(flip, u.file_a, u.file_b)
    return u


def lever_flags(**over):
    f = {k: globals()[k] for k in LEVERS}
    f.update(over)
    return f


def in_combos(u, combos):
    m = pd.Series(False, index=u.index)
    for a, b in combos:
        m |= (u.dcol == a) & (u.dcon == b)
    return m


def conditions(u, flags=None):
    """The final conditions per type under the levers in flags:
    {type: {name: (label, mask)}}."""
    f = flags or lever_flags()
    lo, hi = COLOUR_ONLY_LEVELS
    lev_s = u.dcol.between(lo, hi) & (u.lo_col <= COLOUR_ONLY_LOWER_MAX)
    lab_s = (f"colour {lo}-{hi} levels apart with the lower image at "
             f"{COLOUR_ONLY_LOWER_MAX} or below")
    if f["COLOUR_ONLY_ALLOW_LOW_2"]:
        lev_s |= (u.dcol == 2) & (u.hi_col <= COLOUR_ONLY_LOW_2_MAX_HIGH)
        lab_s += (", or 2 with both at or below "
                  f"{COLOUR_ONLY_LOW_2_MAX_HIGH}")
    if f["COLOUR_ONLY_ALLOW_HIGH_3"]:
        lev_s |= (u.dcol == 3) & (u.lo_col >= 0)
        lab_s += ", or 3 with both at or above 0 (fallback)"
    flo, fhi = CONTRAST_ONLY_LEVELS
    lev_f = u.dcon.between(flo, fhi)
    lab_f = f"contrast {flo}-{fhi} levels apart"
    if f["CONTRAST_ONLY_ALLOW_LOW_2"]:
        lev_f |= (u.dcon == 2) & (u.hi_con <= CONTRAST_ONLY_LOW_2_MAX_HIGH)
        lab_f += (", or 2 with the higher image at "
                  f"{CONTRAST_ONLY_LOW_2_MAX_HIGH} or below")
    C = {
        "sat_only": {
            "levels": (lab_s, lev_s),
            "min": (f"colour >= {COLOUR_ONLY_MIN_SD:g} SD measured",
                    u.z_sat >= COLOUR_ONLY_MIN_SD - EPS),
            "shared": (f"same contrast level, within "
                       f"+-{COLOUR_ONLY_SHARED_CONTRAST_MAX}",
                       (u.dcon == 0) & (u.contrast_step_a.abs()
                                        <= COLOUR_ONLY_SHARED_CONTRAST_MAX))},
        "fog_only": {
            "levels": (lab_f, lev_f),
            "min": (f"contrast >= {CONTRAST_ONLY_MIN_SD:g} SD measured",
                    u.z_fog >= CONTRAST_ONLY_MIN_SD - EPS),
            "shared": (f"same colour level, within "
                       f"+-{CONTRAST_ONLY_SHARED_COLOUR_MAX}",
                       (u.dcol == 0) & (u.colour_step_a.abs()
                                        <= CONTRAST_ONLY_SHARED_COLOUR_MAX))}}
    for t, combos in (("together", TOGETHER_LEVELS),
                      ("conflict", CONFLICT_LEVELS)):
        C[t] = {
            "levels": ("levels apart (colour/contrast) "
                       + ", ".join(f"{a}/{b}" for a, b in combos),
                       in_combos(u, combos)),
            "min": (f"measured >= {MIXED_MIN_SAT_SD:.2f} SD colour and "
                    f">= {MIXED_MIN_CON_SD:.2f} SD contrast",
                    (u.z_sat >= MIXED_MIN_SAT_SD - EPS)
                    & (u.z_fog >= MIXED_MIN_CON_SD - EPS)),
            "cap": (f"measured <= {MIXED_MAX_SD:g} SD on each feature",
                    (u.z_sat <= MIXED_MAX_SD + EPS)
                    & (u.z_fog <= MIXED_MAX_SD + EPS)),
            "pm3": (f"at most one feature reaches +-{MAX_LEVEL}",
                    ~(u.col3 & u.con3))}
    if CONFLICT_BALANCED:
        C["conflict"]["balanced"] = (
            f"balanced [{BALANCE_LO}, {BALANCE_HI}]",
            u.balanced.astype(bool))
    return C


def meets_mask(u, C):
    m = pd.Series(False, index=u.index)
    for t in TYPES:
        ok = u.type == t
        for _, (_, mask) in C[t].items():
            ok &= mask
        m |= ok
    return m


def current_mask(u):
    """The current v5b_180 pool ('v5units'): the minimums, conflict
    balanced; no level rules, no caps."""
    mx = ((u.z_sat >= CURRENT_MIN["mixed_sat"] - EPS)
          & (u.z_fog >= CURRENT_MIN["mixed_con"] - EPS))
    return (((u.type == "sat_only")
             & (u.z_sat >= CURRENT_MIN["sat_only"] - EPS))
            | ((u.type == "fog_only")
               & (u.z_fog >= CURRENT_MIN["fog_only"] - EPS))
            | ((u.type == "together") & mx)
            | ((u.type == "conflict") & mx & u.balanced.astype(bool)))


def category_mask(u, spec):
    m = pd.Series(True, index=u.index)
    if "levels" in spec:
        dist = np.where(u.type == "sat_only", u.dcol, u.dcon)
        m &= pd.Series(np.isin(dist, spec["levels"]), index=u.index)
    if "where" in spec:
        m &= u["where"].isin(spec["where"])
    if "high_eq" in spec:
        m &= u.hi_con == spec["high_eq"]
    if "combos" in spec:
        m &= in_combos(u, spec["combos"])
    return m


def apply_rules(u, C):
    u = u.copy()
    failed = pd.Series([[] for _ in range(len(u))], index=u.index,
                       dtype=object)
    meets = pd.Series(False, index=u.index)
    cat = pd.Series("", index=u.index)
    for t in TYPES:
        sel = u.type == t
        sub = u[sel]
        lists = [[] for _ in range(len(sub))]
        for name, (_, mask) in C[t].items():
            for k in np.flatnonzero(~mask[sel].to_numpy()):
                lists[k].append(name)
        failed[sel] = pd.Series(lists, index=sub.index, dtype=object)
        meets[sel] = [len(x) == 0 for x in lists]
        for spec in EXCLUDED_CATEGORIES[t]:
            rep = set(spec["replaces"])
            ok = category_mask(sub, spec) & pd.Series(
                [set(x) == rep for x in lists], index=sub.index)
            ix = ok[ok].index
            cat[ix] = np.where(cat[ix] == "", spec["name"],
                               cat[ix] + "; " + spec["name"])
    u["failed"] = failed
    u["meets_r3"] = meets
    u["excl_cat"] = cat
    return u


def failed_text(t, failed, C):
    return "; ".join(C[t][f][0] for f in failed) if failed else ""


def build_pairs(m):
    p = make_pairs(m)
    u = p[p.file_a < p.file_b].copy()
    u = add_pair_fields(u)
    C = conditions(u)
    u = apply_rules(u, C)
    u["current"] = current_mask(u)
    return u.reset_index(drop=True), C


# ------------------------------------------------------------
# The leak check
# ------------------------------------------------------------

def limit_list():
    """(name, type or 'overall', term, limit) of every limited |b|."""
    out = [("overall b_sat", "overall", "sat", LEAK_LIMIT),
           ("overall b_con", "overall", "con", LEAK_LIMIT)]
    for t, (term, lim) in TYPE_LIMITS.items():
        out.append((f"{TYPE_LABEL[t]} b_{term}", t, term, lim))
    return out


def limit_group_names():
    g = {"overall": ["overall b_sat", "overall b_con"]}
    for t, (term, _) in TYPE_LIMITS.items():
        g[TYPE_LABEL[t]] = [f"{TYPE_LABEL[t]} b_{term}"]
    return g


def cell_of(q):
    """Leak cell: type x levels apart x where the pair sits (mixed: x
    position too)."""
    return np.where(q.type.isin(["sat_only", "fog_only"]),
                    q.type.map(TYPE_LABEL) + " | " + q.levels + " | "
                    + q["where"],
                    q.type.map(TYPE_LABEL) + " | " + q.levels + " | "
                    + q.position + " | contrast " + q["where"])


class LeakPool:
    """A rule set's pool as arrays for fast draws and swaps."""

    def __init__(self, q):
        q = q.reset_index(drop=True)
        self.q = q
        self.n = len(q)
        self.t = q.type.map({t: i for i, t in enumerate(TYPES)}).to_numpy()
        sc = pd.Categorical(q.scene_id)
        self.scene = sc.codes.astype(int)
        self.n_scenes = len(sc.categories)
        swap_cell = q.type + "|" + q.levels
        self.swap_cell = pd.Categorical(swap_cell).codes.astype(int)
        self.cell_index = {c: np.flatnonzero(self.swap_cell == c)
                           for c in np.unique(self.swap_cell)}
        xs = q.x_sat.to_numpy(float)
        xc = q.x_con.to_numpy(float)
        self.cS = np.column_stack([xs * xs, xs * xc, xc * xc])
        self.cN = np.zeros((self.n, len(NORMALITY_DEFS), 2))
        for k, d in enumerate(NORMALITY_DEFS):
            N = q[f"N_{d}"].to_numpy(float)
            self.cN[:, k, 0] = xs * N
            self.cN[:, k, 1] = xc * N
        self.by_type_scene = {}
        for ti in range(len(TYPES)):
            idx = np.flatnonzero(self.t == ti)
            self.by_type_scene[ti] = {s: idx[self.scene[idx] == s]
                                      for s in np.unique(self.scene[idx])}
        self.limits = limit_list()


def sums_of(P, members):
    S = np.zeros((len(TYPES), 3))
    NS = np.zeros((len(NORMALITY_DEFS), len(TYPES), 2))
    for ti in range(len(TYPES)):
        mm = members[P.t[members] == ti]
        S[ti] = P.cS[mm].sum(axis=0)
        NS[:, ti, :] = P.cN[mm].sum(axis=0)
    return S, NS


def b_values(S, NS, tt=None, dS=None, dN=None):
    """The limited b's for sums S (types x 3) and NS (defs x types x 2),
    optionally changed by k candidate deltas on type tt. Returns an array
    (k, defs, limits) of b."""
    if dS is None:
        dS = np.zeros((1, 3))
        dN = np.zeros((1, NS.shape[0], 2))
        tt = 0
    k = dS.shape[0]
    St = S.sum(axis=0)[None, :] + dS                       # (k, 3)
    Nt = NS.sum(axis=1)[None, :, :] + dN                   # (k, D, 2)
    det = St[:, 0] * St[:, 2] - St[:, 1] ** 2
    det = np.where(np.abs(det) < 1e-300, np.nan, det)
    bs = (St[:, 2:3] * Nt[:, :, 0] - St[:, 1:2] * Nt[:, :, 1]) / det[:, None]
    bc = (St[:, 0:1] * Nt[:, :, 1] - St[:, 1:2] * Nt[:, :, 0]) / det[:, None]
    out = [bs, bc]
    for t, (term, _) in TYPE_LIMITS.items():
        ti = TYPES.index(t)
        j = 0 if term == "sat" else 1
        ci = 0 if term == "sat" else 2
        den = S[ti, ci] + (dS[:, ci] if ti == tt else np.zeros(k))
        num = NS[None, :, ti, j] + (dN[:, :, j] if ti == tt else 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(num / den[:, None])
    return np.stack(out, axis=2)                           # (k, D, L)


def score_of(B, lims):
    r = np.abs(B) / lims[None, None, :]
    r = np.where(np.isfinite(r), r, np.inf)
    return r.max(axis=(1, 2))


def draw_set(P, rng, in_use):
    members = []
    for ti, t in enumerate(TYPES):
        need = LEAK_SET_SIZES[t]
        got = []
        avail = {s: [i for i in ix if not in_use[i]]
                 for s, ix in P.by_type_scene[ti].items()}
        avail = {s: v for s, v in avail.items() if v}
        while len(got) < need and avail:
            for s in rng.permutation(sorted(avail)):
                if len(got) >= need:
                    break
                v = avail[s]
                k = int(rng.integers(len(v)))
                got.append(v.pop(k))
                if not v:
                    del avail[s]
        members += got
    m = np.array(members, int)
    in_use[m] = True
    return m


def caps_of(P):
    cap = np.zeros(len(TYPES), int)
    for ti, t in enumerate(TYPES):
        ns = max(len(P.by_type_scene[ti]), 1)
        cap[ti] = math.ceil(LEAK_SET_SIZES[t] / ns)
    return cap


def improve_set(P, members, in_use, lims):
    """Greedy swaps, one pair at a time: the members are tried in the order
    of how much their removal would lower the score (all of them, or the
    first LEAK_TRY_MEMBERS), and the first whose best replacement (same type
    and levels apart, unused, scene within the cap) lowers the score is
    swapped. Stops when every |b| <= LEAK_SWAP_TARGET x its limit, after
    LEAK_MAX_SWAPS, or when no swap helps. Returns the members, the swaps
    and the b's after each swap (the first entry: the draw)."""
    members = members.copy()
    S, NS = sums_of(P, members)
    cap = caps_of(P)
    cnt = np.zeros((len(TYPES), P.n_scenes), int)
    np.add.at(cnt, (P.t[members], P.scene[members]), 1)
    B = b_values(S, NS)[0]
    cur = score_of(B[None], lims)[0]
    traj, swaps = [B], []
    n_try = LEAK_TRY_MEMBERS or len(members)
    for _ in range(LEAK_MAX_SWAPS):
        if cur <= LEAK_SWAP_TARGET:
            break
        rem = np.empty(len(members))
        for ti in range(len(TYPES)):
            pos = np.flatnonzero(P.t[members] == ti)
            if len(pos):
                mm = members[pos]
                rem[pos] = score_of(b_values(S, NS, ti, -P.cS[mm],
                                             -P.cN[mm]), lims)
        done = False
        for pi in np.argsort(rem, kind="stable")[:n_try]:
            m = members[pi]
            ti = P.t[m]
            cand = P.cell_index[P.swap_cell[m]]
            cand = cand[~in_use[cand]]
            if not len(cand):
                continue
            c = cnt[ti, P.scene[cand]] - (P.scene[cand] == P.scene[m])
            cand = cand[c < cap[ti]]
            if not len(cand):
                continue
            Bc = b_values(S, NS, ti, P.cS[cand] - P.cS[m],
                          P.cN[cand] - P.cN[m])
            sc = score_of(Bc, lims)
            j = int(np.argmin(sc))
            if sc[j] < cur - 1e-12:
                new = cand[j]
                S[ti] += P.cS[new] - P.cS[m]
                NS[:, ti, :] += P.cN[new] - P.cN[m]
                cnt[ti, P.scene[m]] -= 1
                cnt[ti, P.scene[new]] += 1
                in_use[m] = False
                in_use[new] = True
                members[pi] = new
                swaps.append((int(m), int(new)))
                traj.append(Bc[j])
                cur = sc[j]
                done = True
                break
        if not done:
            break
    return members, swaps, traj


def met_groups(B, P):
    """{group: met} for a (defs x limits) array of b."""
    lims = {x[0]: x[3] for x in P.limits}
    names = [x[0] for x in P.limits]
    out = {}
    for g, nms in limit_group_names().items():
        out[g] = all(np.all(np.abs(B[:, names.index(nm)]) <= lims[nm] + EPS)
                     for nm in nms)
    out["all"] = all(out.values())
    return out


def run_leak(name, P, st):
    """LEAK_N_DRAWS draws of sets A and B; the swap step on each; which
    limits are met after 0, 10, ... swaps (LEAK_BUDGETS)."""
    lims = np.array([x[3] for x in P.limits])
    names = [x[0] for x in P.limits]
    cells = cell_of(P.q)
    rows, best, reuse = [], None, []
    drawn_cells, ok_cells, n_ok = {}, {}, 0
    for k in range(LEAK_N_DRAWS):
        rng = st(f"leak|{name}|draw{k}")
        in_use = np.zeros(P.n, bool)
        sets = {}
        for blk in ("A", "B"):
            m0 = draw_set(P, rng, in_use)
            m1, sw, traj = improve_set(P, m0, in_use, lims)
            B0, B1 = traj[0], traj[-1]
            s1 = score_of(B1[None], lims)[0]
            sets[blk] = (m0, m1, sw, s1)
            reuse += reuse_rows(P, m0, name, k, blk, "drawn")
            reuse += reuse_rows(P, m1, name, k, blk, "after swaps")
            r = {"rules": name, "draw": k, "set": blk,
                 "n_pairs": len(m1), "swaps": len(sw),
                 "score_before": score_of(B0[None], lims)[0],
                 "score_after": s1}
            first = [i for i, Bx in enumerate(traj)
                     if met_groups(Bx, P)["all"]]
            r["swaps to meet all"] = first[0] if first else np.nan
            for bud in LEAK_BUDGETS:
                Bx = traj[min(bud, len(traj) - 1)]
                for g, ok in met_groups(Bx, P).items():
                    r[f"met {g} @{bud}"] = ok
            for when, Bx in (("before", B0), ("after", B1)):
                for di, d in enumerate(NORMALITY_DEFS):
                    for li, nm in enumerate(names):
                        r[f"{when} {d} {nm}"] = Bx[di, li]
            rows.append(r)
            for c in cells[m0]:
                drawn_cells[c] = drawn_cells.get(c, 0) + 1
            if met_groups(B1, P)["all"]:
                n_ok += 1
                for c in cells[m1]:
                    ok_cells[c] = ok_cells.get(c, 0) + 1
        dis = not (set(sets["A"][1]) & set(sets["B"][1]))
        rows[-1]["A and B disjoint"] = rows[-2]["A and B disjoint"] = dis
        worst = max(sets["A"][3], sets["B"][3])
        if best is None or worst < best[0]:
            best = (worst, k, sets)
    n_sets = 2 * LEAK_N_DRAWS
    fc = pd.DataFrame([{"rules": name, "cell": c,
                        "mean pairs per set, drawn": drawn_cells.get(c, 0)
                        / n_sets,
                        "mean pairs per set, sets meeting all limits":
                        (ok_cells.get(c, 0) / n_ok) if n_ok else np.nan,
                        "sets meeting all limits": n_ok}
                       for c in sorted(set(drawn_cells) | set(ok_cells))])
    return pd.DataFrame(rows), best, fc, reuse


def reuse_rows(P, members, name, k, blk, stage):
    """Scene use per type in one set: each scene of the type's pool once
    before any twice means the counts differ by at most 1."""
    cap = caps_of(P)
    rows = []
    for ti, t in enumerate(TYPES):
        mm = members[P.t[members] == ti]
        ids = np.array(sorted(P.by_type_scene[ti]), int)
        c = np.bincount(P.scene[mm], minlength=P.n_scenes)[ids]
        rows.append({"rules": name, "draw": k, "set": blk, "stage": stage,
                     "type": TYPE_LABEL[t], "pairs": len(mm),
                     "scenes in pool": len(ids),
                     "scenes used": int((c > 0).sum()),
                     "scenes used twice or more": int((c > 1).sum()),
                     "most uses of one scene": int(c.max()) if len(c) else 0,
                     "cap": int(cap[ti]),
                     "each scene once before any twice":
                         bool(len(c) and c.max() - c.min() <= 1)})
    return rows


def achievable_summary(name, df, P):
    lims = {x[0]: x[3] for x in P.limits}
    groups = limit_group_names()
    out = {"rules": name}
    top = max(LEAK_BUDGETS)

    def worst(df_, nm, when):
        return np.maximum.reduce([df_[f"{when} {d} {nm}"].abs()
                                  for d in NORMALITY_DEFS])
    for bud in LEAK_BUDGETS:
        out[f"sets meeting all @{bud}"] = df[f"met all @{bud}"].mean()
        out[f"draws meeting all @{bud}"] = df.groupby("draw")[
            f"met all @{bud}"].all().mean()
        for g in groups:
            out[f"sets meeting {g} @{bud}"] = df[f"met {g} @{bud}"].mean()
    for g, nms in groups.items():
        wb = np.maximum.reduce([worst(df, nm, "after") for nm in nms])
        out[f"best |b| {g}"] = float(np.min(wb))
        out[f"median |b| {g}"] = float(np.median(wb))
        out[f"limit {g}"] = lims[nms[0]]
    out["median swaps to meet all"] = float(df["swaps to meet all"].median())
    out["sets never meeting all"] = float(df["swaps to meet all"].isna()
                                          .mean())
    out["median swaps"] = df.swaps.median()
    out["median score before"] = df.score_before.median()
    out["median score after"] = df.score_after.median()
    out["top budget"] = top
    return out


def pool_leak_rows(name, q):
    rows = []
    for t in TYPES + ["overall"]:
        s = q if t == "overall" else q[q.type == t]
        if t == "overall":
            terms = ("sat", "con")
        elif t in TYPE_LIMITS:
            terms = (TYPE_LIMITS[t][0],)
        else:
            terms = ("sat", "con")
        r = {"rules": name, "type": "overall" if t == "overall"
             else TYPE_LABEL[t], "pairs": len(s),
             "scenes": s.scene_id.nunique(), "terms": "+".join(terms)}
        for d in NORMALITY_DEFS:
            lk = leak(s[f"N_{d}"], s.x_sat, s.x_con, terms)
            for term in terms:
                r[f"b_{term} {d}"] = lk[f"b_{term}"]
            r[f"share A more typical {d}"] = float((s[f"N_{d}"] > 0).mean()) \
                if len(s) else np.nan
        rows.append(r)
    return rows


def cell_rows(name, q):
    """Each cell's pull on its type's limited b (or, for together, the
    overall fit): sum(x N) over the cell / sum(x^2) over the type; the
    pulls of a type's cells add up to its b."""
    rows = []
    q = q.assign(cell=cell_of(q))
    for t in TYPES:
        s = q[q.type == t]
        term = TYPE_LIMITS.get(t, ("sat", None))[0]
        x = s[f"x_{term}"].to_numpy(float)
        den = (x ** 2).sum()
        for c, g in s.groupby("cell"):
            xg = g[f"x_{term}"].to_numpy(float)
            r = {"rules": name, "type": TYPE_LABEL[t], "cell": c,
                 "pairs": len(g), "term": term,
                 "share of x^2": (xg ** 2).sum() / den if den else np.nan}
            for d in NORMALITY_DEFS:
                r[f"pull {d}"] = (xg * g[f"N_{d}"]).sum() / den \
                    if den else np.nan
                r[f"cell b {d}"] = (xg * g[f"N_{d}"]).sum() / (xg ** 2).sum()
                r[f"share A more typical {d}"] = float(
                    (g[f"N_{d}"] > 0).mean())
            rows.append(r)
    return rows


def helper_rows(u, pool, name):
    """Pairs that would help most: for each type, the marginal change in
    |b| (both definitions) of the type's limited fit (together: the
    overall fit) if the pair were added to the pool, scaled by the limit;
    over the allowed pool, the just-excluded categories and the fallback
    cell."""
    rows, crows = [], []
    for t in TYPES:
        s = pool[pool.type == t]
        cand = u[(u.type == t) & (u.meets_r3 | (u.excl_cat != "")
                                  | ((t == "sat_only") & (u.dcol == 3)
                                     & (u.lo_col >= 0) & (u.dcon == 0)))]
        cand = cand[cand.type == t]
        if t in TYPE_LIMITS:
            term, lim = TYPE_LIMITS[t]
            x = s[f"x_{term}"].to_numpy(float)
            den = (x ** 2).sum()
            xc = cand[f"x_{term}"].to_numpy(float)
            gain = np.zeros(len(cand))
            for d in NORMALITY_DEFS:
                b = (x * s[f"N_{d}"]).sum() / den
                db = xc * (cand[f"N_{d}"].to_numpy(float) - b * xc) / den
                gain += (np.abs(b) - np.abs(b + db)) / lim
        else:
            s = pool                      # the overall fit (all types)
            X = s[["x_sat", "x_con"]].to_numpy(float)
            XtX = X.T @ X
            inv = np.linalg.inv(XtX)
            Xc = cand[["x_sat", "x_con"]].to_numpy(float)
            gain = np.zeros(len(cand))
            for d in NORMALITY_DEFS:
                b = inv @ (X.T @ s[f"N_{d}"].to_numpy(float))
                res = cand[f"N_{d}"].to_numpy(float) - Xc @ b
                db = (Xc @ inv) * res[:, None]
                gain += (np.abs(b).sum() - np.abs(b + db).sum(axis=1)) \
                    / LEAK_LIMIT
        call = cand.assign(gain=gain, cell=cell_of(cand), status=np.where(
            cand.meets_r3, "allowed", np.where(cand.excl_cat != "",
                                               "excluded: " + cand.excl_cat,
                                               "fallback 0 vs +3")))
        for (cl, stt), g in call.groupby(["cell", "status"]):
            crows.append({"rules": name, "type": TYPE_LABEL[t], "cell": cl,
                          "status": stt, "pairs": len(g),
                          "share that help": float((g.gain > 0).mean()),
                          "mean help": float(g.gain.mean()),
                          "share A more typical corpus":
                              float((g.N_corpus > 0).mean()),
                          "share A more typical orig":
                              float((g.N_orig > 0).mean())})
        c = call.sort_values("gain", ascending=False)
        c = c.head(LEAK_N_HELPERS)
        for r in c.itertuples():
            rows.append({
                "rules": name, "type": TYPE_LABEL[t], "pair_key": r.pair_key,
                "image_A": r.file_A, "levels": r.levels, "where": r.where,
                "position": r.position,
                "status": ("allowed" if r.meets_r3 else
                           (f"excluded: {r.excl_cat}" if r.excl_cat
                            else "fallback 0 vs +3")),
                "N_corpus": r.N_corpus, "N_orig": r.N_orig,
                "A more typical": ("both" if r.N_corpus > 0 and r.N_orig > 0
                                   else "corpus" if r.N_corpus > 0
                                   else "orig" if r.N_orig > 0 else "neither"),
                "x_sat": r.x_sat, "x_con": r.x_con,
                "help (rank score)": r.gain})
    return rows, crows


def reproduce_shipped(m, ck):
    """The leak function applied to the shipped v5b_180 sets reproduces
    FROZEN.json's recomputed leaks."""
    found, seen = None, []
    for d0 in V5B180_DIRS:
        fz = os.path.join(d0, "FROZEN.json")
        it = os.path.join(d0, "package", "items.csv")
        have = [f for f in (fz, it) if os.path.isfile(f)]
        seen.append(f"{d0}: " + ("both found" if len(have) == 2 else
                                 "missing " + ", ".join(
                                     os.path.relpath(f, d0)
                                     for f in (fz, it) if f not in have)))
        if len(have) == 2:
            found = d0
            break
    ok = found is not None
    ck.add("leak", "v5b_180 FROZEN.json and package/items.csv exist", ok,
           f"using {found}" if ok else "; ".join(seen))
    if not ok:
        return pd.DataFrame()
    with open(fz) as f:
        ref = json.load(f)["normality"]["recomputed"]
    items = pd.read_csv(it)
    items = items[as_bool(items.analysed)]
    d = m.set_index("filename")
    rows, worst = [], 0.0
    fits = {"overall (both terms)": (None, ("sat", "con")),
            "sat_only on x_sat alone": ("sat_only", ("sat",)),
            "fog_only on x_con alone": ("fog_only", ("con",)),
            "conflict on x_con alone": ("conflict", ("con",))}
    for blk in ("A", "B"):
        q = items[items.set == blk]
        xs = -(d.loc[q.file_a, "sat_mean"].to_numpy()
               - d.loc[q.file_b, "sat_mean"].to_numpy()) / SD_SAT
        xc = FOG_SIGN * (d.loc[q.file_a, "contrast_local"].to_numpy()
                         - d.loc[q.file_b, "contrast_local"].to_numpy()) \
            / SD_FOG
        for dfn in NORMALITY_DEFS:
            N = (d.loc[q.file_b, f"dist_{dfn}"].to_numpy()
                 - d.loc[q.file_a, f"dist_{dfn}"].to_numpy())
            for r in ref:
                if r["set"] != blk or r["definition"] != dfn:
                    continue
                t, terms = fits[r["fit"]]
                k = np.ones(len(q), bool) if t is None else \
                    (q.type == t).to_numpy()
                lk = leak(N[k], xs[k], xc[k], terms)
                b = lk[r["term"]]
                rows.append({"set": blk, "definition": dfn, "fit": r["fit"],
                             "term": r["term"], "b here": b,
                             "b FROZEN.json": r["b"], "n": int(k.sum())})
                worst = max(worst, abs(b - r["b"]))
    ck.add("leak", "the leak function reproduces the shipped v5b_180 leaks "
           "(FROZEN.json, 20 values)", len(rows) == 20 and worst < 1e-5,
           f"{len(rows)} values, largest difference {worst:.1e}")
    return pd.DataFrame(rows)


def leak_check(u, st, ck, label="", expected_ref=None):
    """Every rule set: pool leak, cells, helpers, achievable leak. `label`
    is added to the check names (the run after an exclusion)."""
    expected_ref = expected_ref or EXPECTED_CURRENT
    rulesets = [("reference: v5b_180 pool (v5units)", u[u.current])]
    for combo in itertools.product([False, True], repeat=3):
        f = dict(zip(LEVERS, combo))
        name = ("fallback " + ("on" if f["COLOUR_ONLY_ALLOW_HIGH_3"]
                               else "off")
                + " · colour low-2 " + ("on" if f["COLOUR_ONLY_ALLOW_LOW_2"]
                                        else "off")
                + " · contrast low-2 "
                + ("on" if f["CONTRAST_ONLY_ALLOW_LOW_2"] else "off"))
        if f == lever_flags():
            name += " (default)"
        rulesets.append((name, u[meets_mask(u, conditions(u, f))]))
    cur = {t: int((rulesets[0][1].type == t).sum()) for t in TYPES}
    ck.add("leak", "the reference pool reproduces the current counts"
           + label, cur == expected_ref,
           ", ".join(f"{TYPE_LABEL[t]} {cur[t]:,}" for t in TYPES))
    pool_rows, cells, helpers, draws, summ, best = [], [], [], [], [], {}
    feas, helper_cells, reuse = [], [], []
    ok_sets, disjoint = True, True
    for name, q in rulesets:
        t1 = time.time()
        pool_rows += pool_leak_rows(name, q)
        cells += cell_rows(name, q)
        if name.endswith("(default)"):
            hr, hc = helper_rows(u, q, name)
            helpers += hr
            helper_cells = hc
        P = LeakPool(q)
        df, b, fc, ru = run_leak(name, P, st)
        draws.append(df)
        feas.append(fc)
        reuse += ru
        summ.append(achievable_summary(name, df, P))
        best[name] = (P, b)
        for blk in ("A", "B"):
            m1 = b[2][blk][1]
            ok_sets &= (len(m1) == sum(LEAK_SET_SIZES.values())
                        and len(set(m1)) == len(m1))
        disjoint &= not (set(b[2]["A"][1]) & set(b[2]["B"][1]))
        log(f"leak: {name}: {len(q):,} pairs, {LEAK_N_DRAWS} draws, "
            f"{time.time() - t1:.0f}s")
    ck.add("leak", "every drawn set has the set sizes, from its pool, "
           "pairs unique" + label, ok_sets and all(
               (d_.n_pairs == sum(LEAK_SET_SIZES.values())).all()
               for d_ in draws), f"{sum(LEAK_SET_SIZES.values())} per set")
    ck.add("leak", "sets A and B of every draw are disjoint" + label,
           disjoint and all(
        d_["A and B disjoint"].all() for d_ in draws), "")
    # reproducibility: the first draw of the default rules again
    dname, dq = [x for x in rulesets if x[0].endswith("(default)")][0]
    P = LeakPool(dq)
    keep = globals()["LEAK_N_DRAWS"]
    globals()["LEAK_N_DRAWS"] = 1
    again = run_leak(dname, P, st.__class__(st.seed, st.key))[0]
    globals()["LEAK_N_DRAWS"] = keep
    first = pd.concat(draws)
    first = first[(first.rules == dname) & (first.draw == 0)]
    ck.add("leak", "the leak check is reproducible (first draw of the "
           "default rules again)" + label, np.allclose(
               again.score_after.to_numpy(), first.score_after.to_numpy()),
           f"{first.score_after.round(4).tolist()}")
    return {"pool": pd.DataFrame(pool_rows), "cells": pd.DataFrame(cells),
            "helpers": pd.DataFrame(helpers),
            "helper_cells": pd.DataFrame(helper_cells),
            "draws": pd.concat(draws, ignore_index=True),
            "summary": pd.DataFrame(summ), "best": best,
            "feasible_cells": pd.concat(feas, ignore_index=True),
            "reuse": pd.DataFrame(reuse),
            "rulesets": [n for n, _ in rulesets]}


def best_tables(L):
    """For each rule set's best draw: pairs per cell (draw and after the
    swaps), and every pair the swaps brought in."""
    crow, srow = [], []
    for name, (P, b) in L["best"].items():
        worst, k, sets = b
        q = P.q.assign(cell=cell_of(P.q))
        for blk in ("A", "B"):
            m0, m1, sw, s1 = sets[blk]
            c0 = q.loc[m0, "cell"].value_counts()
            c1 = q.loc[m1, "cell"].value_counts()
            for c in sorted(set(c0.index) | set(c1.index)):
                crow.append({"rules": name, "draw": k, "set": blk,
                             "cell": c, "pairs drawn": int(c0.get(c, 0)),
                             "pairs after swaps": int(c1.get(c, 0))})
            ins = set(m1) - set(m0)
            outs = set(m0) - set(m1)
            for i in sorted(ins):
                r = q.loc[i]
                srow.append({"rules": name, "draw": k, "set": blk,
                             "in/out": "in", "cell": r.cell,
                             "pair_key": r.pair_key, "image_A": r.file_A,
                             "N_corpus": r.N_corpus, "N_orig": r.N_orig,
                             "x_sat": r.x_sat, "x_con": r.x_con})
            for i in sorted(outs):
                r = q.loc[i]
                srow.append({"rules": name, "draw": k, "set": blk,
                             "in/out": "out", "cell": r.cell,
                             "pair_key": r.pair_key, "image_A": r.file_A,
                             "N_corpus": r.N_corpus, "N_orig": r.N_orig,
                             "x_sat": r.x_sat, "x_con": r.x_con})
    return pd.DataFrame(crow), pd.DataFrame(srow)


def calibration_budget(s, ref):
    """The fewest swaps per set (of LEAK_BUDGETS) with which the reference
    pool meets every limit in both sets in most draws (else the largest)."""
    for bud in LEAK_BUDGETS:
        if s.loc[ref, f"draws meeting all @{bud}"] >= LEAK_MOST:
            return bud
    return max(LEAK_BUDGETS)


def conclusion(L):
    """A short plain-language conclusion."""
    s = L["summary"].set_index("rules")
    groups = list(limit_group_names())
    ref = [n for n in s.index if n.startswith("reference")][0]
    dflt = [n for n in s.index if n.endswith("(default)")][0]
    combos = [n for n in s.index if n != ref]
    top = max(LEAK_BUDGETS)
    cal = calibration_budget(s, ref)
    n_set = sum(LEAK_SET_SIZES.values())
    mc = 2 * math.sqrt(0.25 / LEAK_N_DRAWS)

    def p(n, bud):
        return s.loc[n, f"draws meeting all @{bud}"]

    def binding(n, bud):
        f = {g: 1 - s.loc[n, f"sets meeting {g} @{bud}"] for g in groups}
        g = max(f, key=f.get)
        return g, f[g]

    def swaps_txt(n):
        v = s.loc[n, "median swaps to meet all"]
        never = s.loc[n, "sets never meeting all"]
        return ((f"median {v:.0f} swaps per set" if np.isfinite(v)
                 else "no median: most sets never meet them")
                + (f"; {never:.0%} of sets never meet them within {top}"
                   if never > 0 else ""))
    out = []
    out.append(
        f"Reference, the v5b_180 pool (the shipped lists met every limit "
        f"from it with build_trials_v5b.py's exact repair): plain draws meet "
        f"every limit in both sets in {p(ref, 0):.0%} of draws; with the "
        f"swap step it takes {swaps_txt(ref)} (of {n_set} pairs). With "
        f"{cal} swaps per set the reference meets every limit in "
        f"{p(ref, cal):.0%} of draws; that is the yardstick below: a rule "
        f"set that needs many more swaps than the reference is that much "
        f"harder for build_trials_v5b.py. Shares are +-{mc:.0%} or so "
        f"({LEAK_N_DRAWS} draws).")
    g, f = binding(dflt, cal)
    out.append(
        f"Final rules at their defaults: every limit met in both sets in "
        f"{p(dflt, 0):.0%} of plain draws, {p(dflt, cal):.0%} with {cal} "
        f"swaps per set, {p(dflt, top):.0%} with up to {top}; "
        f"{swaps_txt(dflt)}. With {cal} swaps the binding limit is {g} "
        f"(missed in {f:.0%} of sets); after all swaps its best |b| is "
        f"{s.loc[dflt, f'best |b| {g}']:.3f} and the median "
        f"{s.loc[dflt, f'median |b| {g}']:.3f} (limit "
        f"{s.loc[dflt, f'limit {g}']:.2f}).")
    out.append(
        "Median swaps per set needed to meet every limit: " + "; ".join(
            f"{short_rules(n)} {s.loc[n, 'median swaps to meet all']:.0f}"
            for n in s.index) + " (F = fallback, C2 = colour 2-level low "
        "pairs, K2 = contrast -3/-1 pairs).")
    good = [n for n in combos if p(n, cal) >= LEAK_MOST]
    if good:
        best = max(good, key=lambda n: (p(n, cal),
                                        -s.loc[n, "median swaps to meet all"]))
        out.append(
            f"Combinations meeting every limit in most draws with {cal} swaps "
            f"per set (as the reference): " + "; ".join(
                f"{n} ({p(n, cal):.0%})" for n in good)
            + f". The most reliable: {best}.")
    else:
        good2 = [n for n in combos if p(n, top) >= LEAK_MOST]
        closest = max(combos, key=lambda n: (p(n, cal), p(n, top),
                                             -s.loc[n, "median score after"]))
        g2, f2 = binding(closest, cal)
        out.append(
            f"No combination of the three levers meets every limit in most "
            f"draws with the reference's {cal} swaps per set. The closest is "
            f"{closest} ({p(closest, cal):.0%}; binding: {g2}, missed in "
            f"{f2:.0%} of sets). "
            + (f"With up to {top} swaps per set, most draws meet them under: "
               + "; ".join(f"{n} ({p(n, top):.0%})" for n in good2) + "."
               if good2 else
               f"Even with up to {top} swaps per set, none meets them in "
               f"most draws."))
    out.append("This is indicative: the draws and the swap step are "
               "simpler than build_trials_v5b.py (no magnitude bins, "
               "information or position constraints, no exact repair). "
               "build_trials_v5b.py is the real test.")
    return out


# ------------------------------------------------------------
# Oddness (as round 1; used for the labels only)
# ------------------------------------------------------------

def image_oddness(m):
    raw = pd.DataFrame({
        "abs_z_sat": m.z_sat.abs(), "abs_z_con": m.z_con.abs(),
        "comp_boost": np.maximum(np.log(m.comp_k), 0.0),
        "clip_2lv": m["clip_2lv"],
        "abs_colour_dev": m.colour_dev_sd.abs(),
        "abs_contrast_dev": m.contrast_dev_sd.abs(),
        "dist_orig": m.dist_orig, "envelope_n": m.envelope_n})
    stats_ = pd.DataFrame({"mean": raw.mean(), "sd": raw.std(ddof=0)})
    z = (raw - stats_["mean"]) / stats_["sd"].replace(0, 1.0)
    contrib = z * pd.Series(ODDNESS_WEIGHTS)
    out = pd.DataFrame({"filename": m.filename,
                        "score": contrib.sum(axis=1)})
    top = np.argsort(-contrib.to_numpy(), axis=1)[:, :2]
    cols = contrib.columns.to_numpy()
    vals = contrib.to_numpy()
    out["reason1"] = cols[top[:, 0]]
    out["reason2"] = cols[top[:, 1]]
    out["reason1_value"] = vals[np.arange(len(vals)), top[:, 0]]
    out["reason2_value"] = vals[np.arange(len(vals)), top[:, 1]]
    return out.set_index("filename"), stats_


def pair_oddness(u, img):
    a = img.loc[u.file_a.to_numpy()].reset_index(drop=True)
    b = img.loc[u.file_b.to_numpy()].reset_index(drop=True)
    use_a = (a.score >= b.score).to_numpy()
    w = a.where(np.repeat(use_a[:, None], a.shape[1], axis=1), b)
    u = u.copy()
    u["odd_score"] = w.score.to_numpy()
    u["odd_image"] = np.where(use_a, u.file_a, u.file_b)
    u["reasons"] = [f"{COMPONENT_LABEL[r1]} ({v1:+.2f}); "
                    f"{COMPONENT_LABEL[r2]} ({v2:+.2f})"
                    for r1, v1, r2, v2 in zip(w.reason1, w.reason1_value,
                                              w.reason2, w.reason2_value)]
    u["reason1"] = w.reason1.to_numpy()
    u["reason2"] = w.reason2.to_numpy()
    return u


# ------------------------------------------------------------
# Gallery selection
# ------------------------------------------------------------

def even_split(total, keys):
    n = len(keys)
    if n == 0:
        return {}
    base, extra = divmod(total, n)
    return {k: base + (1 if i < extra else 0) for i, k in enumerate(keys)}


def pick_spread(c, n, used_scenes, rng, order_cols=None, indoor=None):
    """As round 2: an unused scene first; then the indoor class furthest
    below its share; then order_cols; then random."""
    if n <= 0 or c.empty:
        return c.iloc[0:0]
    c = c.copy()
    c["_r"] = rng.random(len(c))
    c = c.sort_values((order_cols or []) + ["_r"], kind="stable")
    picked = []
    rows = list(c.itertuples(index=True))
    while len(picked) < n and rows:
        best, best_key = None, None
        for i, r in enumerate(rows):
            bal = 0.0
            if indoor is not None:
                counts, share = indoor
                bal = (counts.get(r.indoor, 0) + 1) / share[r.indoor]
            key = (r.scene_id in used_scenes, bal, i)
            if best_key is None or key < best_key:
                best, best_key = i, key
        r = rows.pop(best)
        picked.append(r.Index)
        used_scenes.add(r.scene_id)
        if indoor is not None:
            indoor[0][r.indoor] = indoor[0].get(r.indoor, 0) + 1
    return c.loc[picked].drop(columns="_r")


def pick_typical(new, t, n, rng, share):
    """As round 2: greedy spread over TYPICAL_DIMS[t]."""
    dims = TYPICAL_DIMS[t]
    values = {d: sorted(new[d].unique()) for d in dims}
    counts = {d: {v: 0 for v in values[d]} for d in dims}
    used, indoor, got = set(), ({}, share), []
    for _ in range(n):
        rest = new[~new.index.isin(got)]
        if rest.empty:
            break
        best, best_key = None, None
        for cell, g in rest.groupby(dims, sort=True):
            cell = cell if isinstance(cell, tuple) else (cell,)
            score = sum(counts[d][v] * len(values[d])
                        for d, v in zip(dims, cell))
            key = (bool(g.scene_id.isin(used).all()), score, rng.random())
            if best_key is None or key < best_key:
                best, best_key = (cell, g), key
        cell, g = best
        k = pick_spread(g, 1, used, rng, indoor=indoor)
        got += list(k.index)
        for d, v in zip(dims, cell):
            counts[d][v] += 1
    return got


def select(u, st, share):
    chosen, notes, excl_info = [], [], []
    for t in TYPES:
        tp = u[u.type == t]
        new = tp[tp.meets_r3]
        got = pick_typical(new, t, N_TYPICAL, st(f"typical|{t}"), share)
        if len(got) < N_TYPICAL:
            notes.append(f"{TYPE_LABEL[t]}, typical: {len(got)} of "
                         f"{N_TYPICAL} ({len(new)} candidates)")
        parts = [u.loc[got].assign(selected_as="typical", group_rule="")]
        if N_ODD:
            cand = new[~new.index.isin(got)].copy()
            cand["_neg"] = -cand.odd_score
            k = pick_spread(cand, N_ODD, set(), st(f"odd|{t}"),
                            order_cols=["_neg"])
            parts.append(u.loc[k.index].assign(selected_as="odd",
                                               group_rule=""))
        cand_by = {}
        for spec in EXCLUDED_CATEGORIES[t]:
            c = tp[tp.excl_cat.str.split("; ").map(
                lambda x, s=spec["name"]: s in x)]
            excl_info.append({"type": TYPE_LABEL[t],
                              "category": spec["name"],
                              "replaces": ", ".join(spec["replaces"]),
                              "candidates": len(c),
                              "scenes": c.scene_id.nunique()})
            if len(c):
                cand_by[spec["name"]] = c
        rules = list(cand_by)
        quota = even_split(N_EXCLUDED, rules)
        short = sum(max(q - len(cand_by[r]), 0) for r, q in quota.items())
        while short:
            moved = False
            for r in rules:
                if short and len(cand_by[r]) > quota[r]:
                    quota[r] += 1
                    short -= 1
                    moved = True
            if not moved:
                break
        used, indoor = set(), ({}, share)
        for r in rules:
            q = min(quota[r], len(cand_by[r]))
            k = pick_spread(cand_by[r], q, used, st(f"excluded|{t}|{r}"),
                            indoor=indoor)
            parts.append(u.loc[k.index].assign(selected_as="excluded",
                                               group_rule=r))
        none = [s["name"] for s in EXCLUDED_CATEGORIES[t]
                if s["name"] not in cand_by]
        if none:
            notes.append(f"{TYPE_LABEL[t]}, excluded: no candidates for "
                         + "; ".join(none))
        n_exc = sum(len(p) for p in parts if (p.selected_as == "excluded")
                    .all() and len(p))
        if n_exc < N_EXCLUDED:
            notes.append(f"{TYPE_LABEL[t]}, excluded: {n_exc} of "
                         f"{N_EXCLUDED}")
        chosen += parts
    return pd.concat(chosen), notes, pd.DataFrame(excl_info)


def read_ratings(path):
    r = pd.read_csv(path, keep_default_na=False, dtype=str)
    if "note" not in r.columns:
        r["note"] = ""
    return r


def pick_anchors(u, r1, r2, r2_anchor_keys, taken, rng):
    """6 from round 1 (rated in rounds 1 and 2: ANCHORS_FROM_R1 agreeing
    and disagreeing) and 6 from round 2 (rated once there:
    ANCHORS_FROM_R2), spread over the types, one scene each where
    possible. A shortfall in one kind passes to the other kind of the same
    round."""
    notes, picked, used = [], [], set()
    pool1, pool2 = anchor_pools(u, r1, r2, r2_anchor_keys, taken)
    for pool, want in ((pool1, ANCHORS_FROM_R1), (pool2, ANCHORS_FROM_R2)):
        want = dict(want)
        avail = {k: int((pool.kind == k).sum()) for k in want}
        short = sum(max(want[k] - avail[k], 0) for k in want)
        for k in want:
            want[k] = min(want[k], avail[k])
        for k in want:
            while short and avail[k] > want[k]:
                want[k] += 1
                short -= 1
        if short:
            notes.append(f"anchors from {pool.source.iloc[0] if len(pool) else 'a round'}: "
                         f"{short} fewer than set")
        for k, n in want.items():
            c = pool[pool.kind == k]
            # spread over types: the type with fewest anchors so far first
            got = []
            for _ in range(n):
                have = pd.Series([u.loc[i, "type"] for i in
                                  [x for p in picked for x in p.index]
                                  + got], dtype=object).value_counts()
                c2 = c[~c.index.isin(got)]
                if c2.empty:
                    break
                c2 = c2.assign(_h=c2.type.map(lambda t: have.get(t, 0)))
                c2 = c2[c2._h == c2._h.min()].drop(columns="_h")
                kk = pick_spread(c2, 1, used, rng)
                got += list(kk.index)
            picked.append(pool.loc[got])
    a = pd.concat(picked) if picked else u.iloc[0:0]
    return a, notes


def anchor_pools(u, r1, r2, r2_anchor_keys, taken):
    """The anchor candidates: rated in rounds 1 and 2 (round 2's anchors;
    kind agree/disagree) and rated once, in round 2 (kind = the rating)."""
    a1 = r1[r1.rating != ""].set_index("pair_key").rating
    a2 = r2[r2.rating != ""].set_index("pair_key").rating
    twice = sorted((set(a1.index) & set(a2.index) & set(r2_anchor_keys))
                   - set(taken))
    pool1 = u[u.pair_key.isin(twice)].copy()
    pool1["r1_rating"] = pool1.pair_key.map(a1)
    pool1["r2_rating"] = pool1.pair_key.map(a2)
    pool1["kind"] = np.where(pool1.r1_rating == pool1.r2_rating, "agree",
                             "disagree")
    pool1["source"] = "round 1 (rated in rounds 1 and 2)"
    once = sorted(set(a2.index) - set(a1.index) - set(taken))
    pool2 = u[u.pair_key.isin(once)].copy()
    pool2["r1_rating"] = ""
    pool2["r2_rating"] = pool2.pair_key.map(a2)
    pool2["kind"] = pool2.r2_rating
    pool2["source"] = "round 2"
    return pool1, pool2


def replace_affected(u, allsel, st, r1, r2, r2_anchor_keys):
    """Each pair touching an excluded scene or image is replaced, in place,
    by one of the same type and group: typical from the same spread cell,
    excluded from the same category, anchors from the same source round,
    rating kind and type (else relaxed, and recorded); one new scene where
    possible; a seeded stream per replaced pair."""
    rows, info, notes = [], [], []
    chosen = set(allsel.pair_key)
    xkeys = set(u.pair_key[u.x_excl])
    for idx, r in allsel.iterrows():
        if r.pair_key not in xkeys:
            rows.append(allsel.loc[[idx]])
            continue
        t, grp = r.type, r.selected_as
        rng = st(f"replace|{r.pair_key}")
        keep = allsel[~allsel.pair_key.isin(xkeys)]
        used = set(keep[(keep.type == t)
                        & (keep.selected_as == grp)].scene_id)
        ok = ~u.x_excl & ~u.pair_key.isin(chosen) & (u.type == t)
        matched, rule = "", r.group_rule
        if grp == "typical":
            cand = u[ok & u.meets_r3]
            cell = cand[np.logical_and.reduce(
                [cand[d] == r[d] for d in TYPICAL_DIMS[t]])]
            matched = ("same cell (" + ", ".join(
                f"{d} {r[d]}" for d in TYPICAL_DIMS[t]) + ")")
            if cell.empty:
                cell, matched = cand, "no pair left in the cell: any cell"
            c = cell
        elif grp == "excluded":
            cand = u[ok]
            c = cand[cand.excl_cat.str.split("; ").map(
                lambda x: r.group_rule in x)]
            matched = f"same category ({r.group_rule})"
            if c.empty:
                c = cand[cand.excl_cat != ""]
                matched = "no pair left in the category: any category"
        else:
            p1, p2 = anchor_pools(u, r1, r2, r2_anchor_keys, chosen)
            pool = p1 if str(r.source).startswith("round 1") else p2
            pool = pool[~pool.x_excl]
            c = pool[(pool.kind == r.kind) & (pool.type == t)]
            matched = f"same source, rating ({r.kind}) and type"
            if c.empty:
                c = pool[pool.kind == r.kind]
                matched = f"same source and rating ({r.kind}); another type"
            if c.empty:
                c = pool
                matched = "same source; another rating"
        if c.empty:
            notes.append(f"{TYPE_LABEL[t]}, {grp}: no replacement for "
                         f"{r.pair_key}")
            info.append({"type": TYPE_LABEL[t], "group": grp,
                         "old pair": r.pair_key, "old scene": r.scene_id,
                         "new pair": "", "new scene": "", "matched": "none"})
            continue
        k = pick_spread(c, 1, used, rng)
        new = c.loc[k.index]
        if grp == "excluded":
            rule = (r.group_rule if "same category" in matched
                    else new.excl_cat.iloc[0].split("; ")[0])
        new = new.assign(selected_as=grp, group_rule=rule,
                         replaces_pair=r.pair_key)
        if grp != "anchor":
            for col in ("r1_rating", "r2_rating", "source", "kind"):
                new[col] = ""
        rows.append(new)
        chosen.add(new.pair_key.iloc[0])
        info.append({"type": TYPE_LABEL[t], "group": grp,
                     "old pair": r.pair_key, "old scene": r.scene_id,
                     "new pair": new.pair_key.iloc[0],
                     "new scene": new.scene_id.iloc[0], "matched": matched,
                     "category": rule if grp == "excluded" else "",
                     "anchor kind": r.kind if grp == "anchor" else ""})
    out = pd.concat(rows)
    if "replaces_pair" not in out.columns:
        out["replaces_pair"] = ""
    out["replaces_pair"] = out.replaces_pair.fillna("")
    return out, pd.DataFrame(info), notes


# ------------------------------------------------------------
# Supply
# ------------------------------------------------------------

def supply_table(u):
    rows = []
    q = u[u.meets_r3]
    for t in TYPES:
        g0 = q[q.type == t]
        rows.append({"type": TYPE_LABEL[t], "levels": "all", "where": "all",
                     "position": "all", "pairs": len(g0),
                     "scenes": g0.scene_id.nunique()})
        for k, g in g0.groupby(["levels", "where", "position"]):
            rows.append({"type": TYPE_LABEL[t], "levels": k[0],
                         "where": k[1], "position": k[2], "pairs": len(g),
                         "scenes": g.scene_id.nunique()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Examples and the gallery (as round 2)
# ------------------------------------------------------------

def level_text(contrast_level, colour_level):
    return f"{contrast_level} {colour_level}"


def predicted_text(r, left_is_a):
    lo_sat = "A" if r.sat_a < r.sat_b else "B"
    lo_con = "A" if r.fog_a < r.fog_b else "B"
    side = {"A": "left" if left_is_a else "right",
            "B": "right" if left_is_a else "left"}
    if r.type == "sat_only":
        return f"{side[lo_sat]} (lower saturation)"
    if r.type == "fog_only":
        return f"{side[lo_con]} (lower contrast)"
    if lo_sat == lo_con:
        return f"{side[lo_sat]} (lower saturation and lower contrast)"
    return (f"{side[lo_sat]} by saturation, {side[lo_con]} by contrast "
            "(conflict)")


def build_examples(sel, rng, C):
    rows = []
    for r in sel.itertuples():
        left_a = bool(rng.random() < 0.5)
        A = (r.file_a, r.contrast_level_a, r.colour_level_a)
        B = (r.file_b, r.contrast_level_b, r.colour_level_b)
        L, Rr = (A, B) if left_a else (B, A)
        anchor = r.selected_as == "anchor"
        if anchor:
            gal_group = "typical" if r.meets_r3 else "excluded"
            rule = "" if r.meets_r3 else (
                r.excl_cat or failed_text(r.type, r.failed, C))
        else:
            gal_group = r.selected_as
            rule = r.group_rule
        rows.append({
            "pair_key": r.pair_key, "type": TYPE_LABEL[r.type],
            "group": gal_group, "selected_as": r.selected_as,
            "anchor": anchor,
            "anchor_source": r.source if anchor else "",
            "anchor_round1_rating": r.r1_rating if anchor else "",
            "anchor_round2_rating": r.r2_rating if anchor else "",
            "scene": r.scene_id, "category": r.category,
            "indoor": r.indoor,
            "left_file": L[0], "right_file": Rr[0],
            "left_levels": level_text(L[1], L[2]),
            "right_levels": level_text(Rr[1], Rr[2]),
            "levels_apart": r.levels,
            "colour_levels_apart": int(r.dcol),
            "contrast_levels_apart": int(r.dcon),
            "where": r.where, "position": r.position,
            "shared_level": r.shared_level,
            "d_sat_sd": float(r.z_sat), "d_con_sd": float(r.z_fog),
            "balance": float(r.balance) if np.isfinite(r.balance)
            else np.nan,
            "predicted": predicted_text(r, left_a),
            "image_A_leak": r.file_A,
            "N_corpus": float(r.N_corpus), "N_orig": float(r.N_orig),
            "odd_score": float(r.odd_score), "odd_image": r.odd_image,
            "reasons": r.reasons if r.selected_as == "odd" else "",
            "reason1": r.reason1 if r.selected_as == "odd" else "",
            "reason2": r.reason2 if r.selected_as == "odd" else "",
            "all_reasons": r.reasons,
            "exclusion_rule": rule,
            "conditions_failed": failed_text(r.type, r.failed, C),
            "meets_final_rules": bool(r.meets_r3)})
    return pd.DataFrame(rows)


def label_lines(e):
    l1 = f"{e.type} · {e.group} · scene {e.scene} ({e.category}, {e.indoor})"
    shared = (f" · shared level {e.shared_level}" if e.shared_level != ""
              else "")
    where = "" if not e.where else (
        f" · {e.where}" + (" (contrast)" if "/" in str(e.levels_apart)
                           else ""))
    l2 = (f"left {e.left_levels}  |  right {e.right_levels}  ·  levels apart: "
          f"{e.levels_apart}  ·  measured: saturation {e.d_sat_sd:.2f} SD, "
          f"contrast {e.d_con_sd:.2f} SD  ·  {e.position}{where}{shared}")
    l3 = f"predicted: {e.predicted}"
    if e.group == "excluded":
        l4 = f"excluded by: {e.exclusion_rule}"
    elif e.group == "odd":
        l4 = f"oddness {e.odd_score:+.2f}: {e.reasons}"
    else:
        l4 = f"oddness {e.odd_score:+.2f}"
    return [l1, l2, l3, l4]


GALLERY_JS = r"""
const KEY = "pair_preview_v5b_" + DATA.hash;
let R = {};
try { R = JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch (e) { R = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(R)); } catch (e) {} }
const list = document.getElementById("list");
const cards = {};
function el(tag, attrs, text) {
  const e = document.createElement(tag);
  for (const k in (attrs || {})) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}
function rated(p) { return !!(R[p.pair_key] && R[p.pair_key].rating); }
DATA.pairs.forEach(p => {
  const c = el("div", {"class": "card", "id": "p" + p.n});
  c.appendChild(el("div", {"class": "num"}, "Pair " + p.n));
  const im = el("div", {"class": "imgs"});
  im.appendChild(el("img", {"src": "images/" + p.left_file, "alt": "left", "loading": "lazy"}));
  im.appendChild(el("img", {"src": "images/" + p.right_file, "alt": "right", "loading": "lazy"}));
  c.appendChild(im);
  const lab = el("div", {"class": "label"});
  p.label.forEach(t => lab.appendChild(el("div", {}, t)));
  c.appendChild(lab);
  const rt = el("div", {"class": "rate"});
  DATA.ratings.forEach(r => {
    const l = el("label");
    const i = el("input", {"type": "radio", "name": "r" + p.n, "value": r});
    if (R[p.pair_key] && R[p.pair_key].rating === r) i.checked = true;
    i.addEventListener("change", () => {
      R[p.pair_key] = Object.assign(R[p.pair_key] || {}, {rating: r, t: new Date().toISOString()});
      save(); c.classList.add("rated"); count();
    });
    l.appendChild(i); l.appendChild(document.createTextNode(" " + r));
    rt.appendChild(l);
  });
  const note = el("input", {"type": "text", "class": "note", "placeholder": "note (optional)"});
  note.value = (R[p.pair_key] && R[p.pair_key].note) || "";
  note.addEventListener("input", () => {
    R[p.pair_key] = Object.assign(R[p.pair_key] || {}, {note: note.value});
    save();
  });
  rt.appendChild(note);
  c.appendChild(rt);
  if (rated(p)) c.classList.add("rated");
  cards[p.n] = c;
});
function count() {
  const n = DATA.pairs.filter(rated).length;
  document.getElementById("count").textContent = n + " of " + DATA.pairs.length + " rated";
}
function render() {
  const ft = document.getElementById("ftype").value;
  const fg = document.getElementById("fgroup").value;
  const un = document.getElementById("unrated").checked;
  const ord = document.getElementById("order").value;
  const idx = ord === "grouped" ? DATA.order_grouped : DATA.order_shuffled;
  list.innerHTML = "";
  idx.forEach(i => {
    const p = DATA.pairs[i];
    if (ft && p.type !== ft) return;
    if (fg && p.group !== fg) return;
    if (un && rated(p)) return;
    list.appendChild(cards[p.n]);
  });
}
function csvCell(v) {
  v = (v === undefined || v === null) ? "" : String(v);
  return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
}
function download() {
  const cols = ["pair_key", "pair_number", "settings_hash", "type", "group", "scene",
                "left_levels", "right_levels", "levels_apart", "d_sat_sd", "d_con_sd",
                "rating", "note", "rated_at"];
  const lines = [cols.join(",")];
  DATA.pairs.slice().sort((a, b) => a.n - b.n).forEach(p => {
    const r = R[p.pair_key] || {};
    lines.push([p.pair_key, p.n, DATA.hash, p.type, p.group, p.scene, p.left_levels,
                p.right_levels, p.levels_apart, p.d_sat_sd.toFixed(4), p.d_con_sd.toFixed(4),
                r.rating || "", r.note || "", r.t || ""].map(csvCell).join(","));
  });
  const blob = new Blob([lines.join("\n") + "\n"], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ratings_" + DATA.hash + ".csv";
  document.body.appendChild(a); a.click(); a.remove();
}
document.getElementById("hide").addEventListener("change", e => {
  document.body.classList.toggle("hide", e.target.checked);
});
["ftype", "fgroup", "unrated", "order"].forEach(id =>
  document.getElementById(id).addEventListener("change", render));
document.getElementById("dl").addEventListener("click", download);
document.getElementById("order").value = DATA.default_order;
count(); render();
"""


def number_pairs(ex, order_h):
    """Pair numbers: a seeded permutation of the rows (as the base run)."""
    rng = Streams(RANDOM_SEED, "preview_pair_rules_r3_v5b", order_h)("order")
    n = len(ex)
    shuffled = rng.permutation(n)
    number = np.empty(n, int)
    number[shuffled] = np.arange(1, n + 1)
    ex = ex.copy()
    ex["pair_number"] = number
    return ex, shuffled, number


def write_gallery(ex, out, h, order_h=None, fname="gallery.html",
                  title=None, bar=None, groups=None):
    """The gallery page. order_h: the hash whose seed numbers the pairs
    (the base run's, so that kept pairs keep their numbers); the defaults
    give the round-3 gallery."""
    groups = groups or GROUPS
    title = title or f"Pair preview v5b round 3 ({h})"
    bar = bar or f"round 3 · settings {h}"
    ex, shuffled, number = number_pairs(ex, order_h or h)
    n = len(ex)
    grouped = sorted(range(n), key=lambda i: (
        list(TYPE_LABEL.values()).index(ex.type.iloc[i]),
        groups.index(ex.group.iloc[i]), number[i]))
    pairs = []
    for i, e in enumerate(ex.itertuples()):
        pairs.append({"n": int(number[i]), "pair_key": e.pair_key,
                      "type": e.type, "group": e.group, "scene": e.scene,
                      "left_file": e.left_file, "right_file": e.right_file,
                      "left_levels": e.left_levels,
                      "right_levels": e.right_levels,
                      "levels_apart": e.levels_apart,
                      "d_sat_sd": e.d_sat_sd, "d_con_sd": e.d_con_sd,
                      "label": label_lines(e)})
    data = {"hash": h, "pairs": pairs, "ratings": RATINGS,
            "order_shuffled": [int(i) for i in shuffled],
            "order_grouped": [int(i) for i in grouped],
            "default_order": DEFAULT_ORDER}
    opts_t = "".join(f'<option value="{html.escape(v)}">{html.escape(v)}'
                     "</option>" for v in TYPE_LABEL.values())
    opts_g = "".join(f'<option value="{g}">{g}</option>' for g in groups)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ background: {BACKGROUND}; color: #111; margin: 0;
         font-family: system-ui, -apple-system, Segoe UI, sans-serif; }}
  #bar {{ position: sticky; top: 0; z-index: 2; background: #5f5f5f;
          color: #fff; padding: 8px 16px; display: flex; flex-wrap: wrap;
          gap: 16px; align-items: center; font-size: 14px; }}
  #bar select, #bar button {{ font-size: 14px; }}
  #list {{ padding: 8px 16px 64px; }}
  .card {{ width: {2 * IMAGE_PX + GAP_PX}px; margin: 28px auto; }}
  .num {{ font-size: 13px; margin-bottom: 4px; color: #222; }}
  .imgs {{ display: flex; gap: {GAP_PX}px; }}
  .imgs img {{ width: {IMAGE_PX}px; height: {IMAGE_PX}px; display: block; }}
  .label {{ font-size: 13px; margin-top: 6px; line-height: 1.45;
            display: none; }}
  .card.rated .label {{ display: block; }}
  body.hide .card .label {{ display: none; }}
  .rate {{ margin-top: 8px; display: flex; flex-wrap: wrap; gap: 14px;
           align-items: center; font-size: 14px; }}
  .note {{ flex: 1; min-width: 220px; font-size: 13px; }}
</style></head>
<body class="hide">
<div id="bar">
  <label title="Labels are shown only for pairs already rated"><input type="checkbox" id="hide" checked> hide labels</label>
  <label>type <select id="ftype"><option value="">all</option>{opts_t}</select></label>
  <label>group <select id="fgroup"><option value="">all</option>{opts_g}</select></label>
  <label><input type="checkbox" id="unrated"> unrated only</label>
  <label>order <select id="order"><option value="shuffled">shuffled</option><option value="grouped">grouped</option></select></label>
  <button id="dl">Download ratings</button>
  <span id="count"></span>
  <span>{bar}</span>
</div>
<div id="list"></div>
<script>
const DATA = {json.dumps(data)};
{GALLERY_JS}
</script>
</body></html>
"""
    with open(os.path.join(out, fname), "w", encoding="utf-8") as f:
        f.write(page)
    return ex


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

def tab(df, nd=3):
    return "  " + df.to_string(index=False, float_format=lambda x:
                               f"{x:.{nd}f}").replace("\n", "\n  ")


def vc(s):
    c = s.value_counts()

    def key(v):
        try:
            return (0, float(str(v).split("/")[0]), str(v))
        except ValueError:
            order = WHERE + ["middle", "touching +-3"]
            return (1, order.index(v) if v in order else 9, str(v))
    return ", ".join(f"{v} {c[v]}" for v in sorted(c.index, key=key))


def ordered(ex, col="selected_as", groups=SELECTED):
    return ex.assign(type=pd.Categorical(ex.type, list(TYPE_LABEL.values()),
                                         ordered=True),
                     **{col: pd.Categorical(ex[col], groups, ordered=True)})


def short_rules(n):
    return (n.replace("reference: v5b_180 pool (v5units)", "REFERENCE v5units")
            .replace("fallback ", "F:").replace(" · colour low-2 ", " C2:")
            .replace(" · contrast low-2 ", " K2:"))


def write_report(out, ck, h, info, R):
    L = []
    w = L.append
    w("PAIR PREVIEW, ROUND 3: FINAL RULES AND A LEAK CHECK  "
      "(preview_pair_rules_r3_v5b.py)")
    w("=" * 78)
    w(f"run       {time.strftime('%Y-%m-%d %H:%M:%S')}  "
      f"({R['minutes']:.1f} min)")
    w(f"output    {out}")
    w(f"settings  {h}  (round 1: {ROUND1_HASH}, round 2: {ROUND2_HASH})")
    w(f"manifest  {MANIFEST_CSV} (SHA-256 "
      f"{info['manifest_sha256'][:16]}...)")
    w(f"anchors   {R['anchors_files'][0]}, {R['anchors_files'][1]}")
    w(f"checks    {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL")
    w("")
    w("CONCLUSION OF THE LEAK CHECK (indicative)")
    w("-" * 78)
    for para in R["conclusion"]:
        words, line = para.split(), "  "
        for wd in words:
            if len(line) + len(wd) + 1 > 78:
                w(line)
                line = "  "
            line += ("" if line == "  " else " ") + wd
        w(line)
        w("")
    for line in R.get("exclusion_lines", []):
        w(line)
    w("1. FINAL RULES (settings; levers: " + ", ".join(
        f"{k} = {globals()[k]}" for k in LEVERS) + ")")
    w("-" * 78)
    for t in TYPES:
        w(f"  {TYPE_LABEL[t]}:")
        for name, (lab, _) in R["C"][t].items():
            w(f"    [{name}] {lab}")
    w("  Where a pair sits (changing feature; contrast for mixed pairs): "
      "low = both at or below 0,")
    w("  high = both at or above 0, otherwise straddling.")
    w("")
    w("2. SUPPLY UNDER THE FINAL RULES (default levers)")
    w("-" * 78)
    sp = R["supply"]
    w(tab(sp[sp.levels == "all"][["type", "pairs", "scenes"]]))
    w(("  Before the exclusion (as expected): " if exclusion_active() else
       "  Expected (reported before): ") + ", ".join(
        f"{TYPE_LABEL[t]} {n:,}" for t, n in EXPECTED_CANDIDATES.items()))
    w("  Current counts reproduced: " + ", ".join(
        f"{TYPE_LABEL[t]} {R['cur'][t]:,}" for t in TYPES))
    q = R["u"][R["u"].meets_r3]
    for t in TYPES:
        g = q[q.type == t]
        w(f"    {TYPE_LABEL[t]}: levels {vc(g.levels)}; where {vc(g['where'])}"
          f"; position {vc(g.position)}")
    w("")
    w("3. LEAK CHECK")
    w("-" * 78)
    w("  A = the image predicted by the type's driver (colour only: less "
      "saturated; others: lower")
    w("  contrast). N = dist(B) - dist(A): positive when A is the more "
      "typical. Limits per set and")
    w("  definition: overall |b_sat|, |b_con| <= "
      f"{LEAK_LIMIT}; " + "; ".join(
          f"{TYPE_LABEL[t]} on x_{term} alone <= {lim}"
          for t, (term, lim) in TYPE_LIMITS.items()) + ".")
    w("")
    w("  3.1 The shipped v5b_180 sets, recomputed here (they met the limits"
      " from the reference pool):")
    w(tab(R["shipped"], 4))
    w("")
    w("  3.2 Pool leak (every candidate pair; b of the limited fit; together "
      "is fitted on both terms")
    w("      and not limited). Rule sets: F = fallback (0 vs +3), C2 = "
      "colour 2-level low pairs,")
    w("      K2 = contrast -3/-1 pairs.")
    pl = R["leak"]["pool"].copy()
    pl["rules"] = pl.rules.map(short_rules)
    cols = ["rules", "type", "pairs", "terms"] + [
        c for c in pl.columns if c.startswith("b_")] + [
        c for c in pl.columns if c.startswith("share")]
    w(tab(pl[cols]))
    w("")
    w(f"  3.3 Achievable leak: {LEAK_N_DRAWS} draws per rule set, two "
      "disjoint sets per draw ("
      + ", ".join(f"{v} {TYPE_LABEL[k]}" for k, v in LEAK_SET_SIZES.items())
      + "),")
    w("      then the swap step: one pair at a time, within type and levels "
      "apart, scenes capped as in")
    w(f"      the draw; stops when every |b| <= {LEAK_SWAP_TARGET:g} x its "
      f"limit, or after {LEAK_MAX_SWAPS} swaps.")
    s = R["leak"]["summary"].copy()
    sref = s.set_index("rules")
    ref = [n for n in sref.index if n.startswith("reference")][0]
    cal = calibration_budget(sref, ref)
    s["rules"] = s.rules.map(short_rules)
    groups = list(limit_group_names())
    w("      DRAWS in which both sets meet every limit, by swaps allowed per "
      "set (0 = plain draws):")
    tbl = pd.DataFrame({"rules": s.rules})
    for bud in LEAK_BUDGETS:
        tbl[f"@{bud}"] = [f"{x:.0%}" for x in s[f"draws meeting all @{bud}"]]
    tbl["median swaps to meet all"] = s["median swaps to meet all"].map(
        lambda x: f"{x:.0f}" if np.isfinite(x) else "-")
    tbl["sets never"] = [f"{x:.0%}" for x in s["sets never meeting all"]]
    w(tab(tbl))
    w("")
    w(f"      SETS meeting each limit with {cal} swaps per set (the "
      "reference's yardstick) and, in")
    w("      brackets, with plain draws:")
    tb1 = pd.DataFrame({"rules": s.rules})
    for g in groups:
        tb1[g] = [f"{a:.0%} ({b:.0%})" for a, b in
                  zip(s[f"sets meeting {g} @{cal}"], s[f"sets meeting {g} @0"])]
    w(tab(tb1))
    w("")
    w("      Best / median worst |b| after the swap step (over both "
      "definitions; limit in brackets):")
    tb2 = pd.DataFrame({"rules": s.rules})
    for g in groups:
        tb2[g] = [f"{a:.3f} / {b:.3f} ({c:.2f})" for a, b, c in
                  zip(s[f"best |b| {g}"], s[f"median |b| {g}"],
                      s[f"limit {g}"])]
    w(tab(tb2))
    w("")
    w("  3.4 What drives the leak (default rules): each cell's pull on its "
      "type's b (sum of x N over")
    w("      the cell / sum of x^2 over the type; the pulls add up to the "
      "type's pool b). Together")
    w("      has no limit of its own and its joint fit is unstable (x_sat ~ "
      "x_con at 2/2), so its")
    w("      cells are shown on x_sat for orientation only. Largest first:")
    ce = R["leak"]["cells"]
    ce = ce[ce.rules.str.endswith("(default)")].copy()
    ce["|pull|"] = np.maximum.reduce([ce[f"pull {d}"].abs()
                                      for d in NORMALITY_DEFS])
    for t in TYPES:
        c = ce[ce.type == TYPE_LABEL[t]].sort_values("|pull|",
                                                     ascending=False)
        w(f"    {TYPE_LABEL[t]}:")
        w(tab(c[["cell", "pairs", "share of x^2"]
                + [f"pull {d}" for d in NORMALITY_DEFS]
                + [f"share A more typical {d}" for d in NORMALITY_DEFS]]
              .head(8)).replace("\n", "\n  "))
    w("")
    w("      Which pairs would help: for every allowed, just-excluded or "
      "fallback pair, the change in")
    w("      |b| (both definitions, scaled by the limit) if it were added to "
      "the pool (together: the")
    w("      overall fit). By cell, the cells that help most first "
      "(share of their pairs that help):")
    hc = R["leak"]["helper_cells"]
    for t in TYPES:
        c = hc[hc.type == TYPE_LABEL[t]].sort_values("mean help",
                                                     ascending=False)
        if len(c):
            w(f"    {TYPE_LABEL[t]}:")
            w(tab(c[["cell", "status", "pairs", "share that help",
                     "share A more typical corpus",
                     "share A more typical orig"]].head(6))
              .replace("\n", "\n  "))
    w(f"      The {LEAK_N_HELPERS} single pairs that help most per type: "
      "leak_helpers.csv.")
    w("")
    w("  3.5 What a feasible list looks like: mean pairs per set in each "
      "cell, in the sets that met")
    w("      every limit after the swaps, against the plain draws (changes of "
      "0.5 or more; all cells in")
    w("      leak_feasible_cells.csv):")
    fc = R["leak"]["feasible_cells"]
    show = [n for n in R["leak"]["rulesets"] if n.startswith("reference")
            or n.endswith("(default)")]
    for name in show:
        c = fc[fc.rules == name].copy()
        n_ok = int(c["sets meeting all limits"].max()) if len(c) else 0
        c["change"] = (c["mean pairs per set, sets meeting all limits"]
                       - c["mean pairs per set, drawn"])
        c = c[c.change.abs() >= 0.5].sort_values("change")
        w(f"    {short_rules(name)} ({n_ok} of {2 * LEAK_N_DRAWS} sets met "
          "every limit):")
        if len(c):
            w(tab(c[["cell", "mean pairs per set, drawn",
                     "mean pairs per set, sets meeting all limits",
                     "change"]], 1).replace("\n", "\n  "))
        else:
            w("      no cell changes by 0.5 pairs or more")
    w("")
    w("  3.6 The best draw per rule set (lowest worst score over A and B): "
      "the pairs the swap step")
    w("      took out and brought in, net by cell (every pair in "
      "leak_best_swaps.csv):")
    bs = R["best_swaps"]
    for name in R["leak"]["rulesets"]:
        c = bs[bs.rules == name]
        if not len(c):
            w(f"    {short_rules(name)}: no swaps")
            continue
        g = c.groupby(["cell", "in/out"]).size().unstack(fill_value=0)
        for col in ("in", "out"):
            if col not in g.columns:
                g[col] = 0
        g["net"] = g["in"] - g["out"]
        g = g[g.net != 0].sort_values("net")
        for line in textwrap.wrap(
                f"{short_rules(name)}: {int((c['in/out'] == 'in').sum())} "
                "swaps; net: " + "; ".join(f"{k} {int(v):+d}"
                                            for k, v in g.net.items()),
                width=76, initial_indent="    ", subsequent_indent="      "):
            w(line)
    bc = R["best_cells"]
    dflt = [n for n in R["leak"]["rulesets"] if n.endswith("(default)")][0]
    c = bc[bc.rules == dflt].groupby("cell")[[
        "pairs drawn", "pairs after swaps"]].sum().reset_index()
    w("    Default rules, the best draw's pairs per cell (A + B):")
    w(tab(c).replace("\n", "\n  "))
    w("")
    w("4. THE PAIRS SHOWN")
    w("-" * 78)
    ex = ordered(R["ex"])
    g_ = ex.groupby(["type", "selected_as"], observed=True).agg(
        pairs=("pair_key", "size"), scenes=("scene", "nunique"))
    w(f"  per type and group (set: typical {N_TYPICAL}, odd {N_ODD}, "
      f"excluded {N_EXCLUDED}; anchors 12 in all):")
    w("  " + g_.to_string().replace("\n", "\n  "))
    w("  Typical pairs per dimension:")
    typ = ex[ex.selected_as == "typical"]
    dims = {"levels": "levels_apart", "position": "position",
            "where": "where"}
    for t in TYPES:
        g = typ[typ.type == TYPE_LABEL[t]]
        w(f"    {TYPE_LABEL[t]}: " + "; ".join(
            f"{d} {vc(g[dims[d]])}" for d in TYPICAL_DIMS[t]))
    c = ex[ex.selected_as != "anchor"].groupby(
        ["type", "selected_as", "levels_apart", "where", "position"],
        observed=True).size().rename("pairs").reset_index()
    w(tab(c))
    io_ = ex.groupby(["type", "selected_as", "indoor"], observed=True).size(
        ).unstack(fill_value=0)
    w("  by indoor class (the 47 scenes: " + ", ".join(
        f"{k} {v}" for k, v in R["scene_split"].items()) + "):")
    w("  " + io_.to_string().replace("\n", "\n  "))
    if R["notes"]:
        w("  Why some counts are lower:")
        for n_ in R["notes"]:
            w(f"    - {n_}")
    w("")
    w("5. JUST EXCLUDED")
    w("-" * 78)
    w("  Each pair meets every final condition except the one its category "
      "replaces. Split evenly")
    w("  in the order listed (shortfalls passed on); at random, one scene "
      "each, indoor balanced.")
    e2 = R["excl"].copy()
    shown = ex[ex.selected_as == "excluded"].groupby(
        ["type", "exclusion_rule"], observed=True).size()
    e2["shown"] = [int(shown.get((t, r), 0)) for t, r in
                   zip(e2.type, e2.category)]
    w(tab(e2))
    w("")
    w("6. ANCHORS")
    w("-" * 78)
    an = ex[ex.selected_as == "anchor"].sort_values(["type",
                                                     "anchor_source"])
    w(tab(an[["type", "anchor_source", "anchor_round1_rating",
              "anchor_round2_rating", "scene", "levels_apart", "where",
              "group"]]))
    w("")
    for line in R.get("scene_check_lines", []):
        w(line)
    w("7. CHECKS")
    w("-" * 78)
    for r in ck.rows:
        w(f"  {r['status']}  [{r['section']}] {r['check']}")
        if r["detail"]:
            w(f"          {r['detail']}")
    w("")
    w("8. FILES")
    w("-" * 78)
    for fn in sorted(set(os.listdir(out)) | {"report.txt"}):
        w(f"  {fn}")
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


def stop(out, ck, why):
    failed = [f"{r['check']} ({r['detail']})" for r in ck.rows
              if r["status"] == "FAIL"]
    why += "\n\nFailed:\n" + "\n".join(f"  - {f}" for f in failed)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(ck.rows).to_csv(os.path.join(out, "checks.csv"),
                                 index=False)
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write("PAIR PREVIEW, ROUND 3: STOPPED\n\n" + why + "\n\n"
                + "\n".join(f"{r['status']}  [{r['section']}] {r['check']}  "
                            f"{r['detail']}" for r in ck.rows) + "\n")
    log(f"STOPPED: {why} (see {out}/report.txt)")
    sys.exit(1)


# ------------------------------------------------------------
# Gallery mode
# ------------------------------------------------------------

def check_ratings_file(ck, path, label, gallery, ghash):
    ok = os.path.isfile(path)
    ck.add("anchors", f"the {label} ratings file exists", ok, path)
    if not ok:
        return None
    r = read_ratings(path)
    need = {"pair_key", "rating", "type"}
    ck.add("anchors", f"the {label} file has pair_key, type, rating",
           need <= set(r.columns), f"{len(r)} rows")
    exf = os.path.join(gallery, "examples.csv")
    if os.path.isfile(exf):
        keys = set(pd.read_csv(exf, keep_default_na=False).pair_key)
        ck.add("anchors", f"the {label} file belongs to its gallery "
               f"({gallery})", set(r.pair_key) <= keys, f"{len(r)} rows")
    else:
        ck.add("anchors", f"the {label} file belongs to its gallery",
               "settings_hash" not in r.columns
               or (r.settings_hash == ghash).all(),
               f"{gallery} not found here; checked the file only")
    return r


def main_gallery(anchors):
    ck = Checks()
    active = exclusion_active()
    h = settings_hash()
    base_h = settings_hash(with_exclusion=False)    # == h without exclusion
    out = new_output_dir(h)
    log(f"output folder {out}")
    watch = [ROUND1_GALLERY, ROUND2_GALLERY] + ([BASE_RUN] if active else [])
    digests = {g: folder_digest(g) for g in watch}
    ck.add("rounds", "the settings hash differs from rounds 1 and 2",
           h not in (ROUND1_HASH, ROUND2_HASH),
           f"{h} vs {ROUND1_HASH}, {ROUND2_HASH}")
    ck.add("rounds", "the output folder is not round 1's or round 2's",
           os.path.basename(out).split("_run")[0]
           not in (ROUND1_HASH, ROUND2_HASH), out)
    m, info = read_inputs(ck)
    if ck.n_fail:
        stop(out, ck, "an input check failed")
    if active:
        ck.add("exclusion", "the excluded scenes and images are in the "
               "manifest (else the run would have stopped)",
               not validate_exclusion(m),
               f"{len(EXCLUDE_SCENES)} scenes, {len(EXCLUDE_IMAGES)} images")
        ck.add("exclusion", "the settings hash with the exclusion differs "
               "from the base run's", h != base_h, f"{h} vs {base_h}")
    r1 = check_ratings_file(ck, anchors[0], "round-1", ROUND1_GALLERY,
                            ROUND1_HASH)
    r2 = check_ratings_file(ck, anchors[1], "round-2", ROUND2_GALLERY,
                            ROUND2_HASH)
    if ck.n_fail:
        stop(out, ck, "a ratings file is missing or not as expected")
    ex2f = os.path.join(ROUND2_GALLERY, "examples.csv")
    if os.path.isfile(ex2f):
        e2 = pd.read_csv(ex2f, keep_default_na=False, dtype=str)
        r2_anchor_keys = set(e2[e2.selected_as == "anchor"].pair_key)
        src = ex2f
    else:
        r2_anchor_keys = set(r1.pair_key) & set(r2.pair_key)
        src = "keys in both files"
    ck.add("anchors", "round 2's anchors are known", bool(r2_anchor_keys),
           f"{len(r2_anchor_keys)} from {src}")

    u, C = build_pairs(m)
    u["x_excl"] = excluded_mask(u)
    cur = {t: int((u.current & (u.type == t)).sum()) for t in TYPES}
    ck.add("typing", "the pair typing reproduces the current counts "
           "(EXPECTED_CURRENT)", cur == EXPECTED_CURRENT,
           ", ".join(f"{TYPE_LABEL[t]} {cur[t]:,}" for t in TYPES))
    cand = {t: int((u.meets_r3 & (u.type == t)).sum()) for t in TYPES}
    ck.add("typing", "the final-rule candidate counts are as reported "
           "(EXPECTED_CANDIDATES)", cand == EXPECTED_CANDIDATES,
           ", ".join(f"{TYPE_LABEL[t]} {cand[t]:,}" for t in TYPES))
    shipped = reproduce_shipped(m, ck)
    if ck.n_fail:
        stop(out, ck, "a check on the typing, the counts or the shipped "
             "v5b_180 leaks failed")

    log("leak check: "
        f"{LEAK_N_DRAWS} draws x 9 rule sets ...")
    LK0 = leak_check(u, Streams(RANDOM_SEED, "preview_pair_rules_r3_v5b",
                                base_h), ck)
    ua = u[~u.x_excl] if active else u
    cur_a = {t: int((ua.current & (ua.type == t)).sum()) for t in TYPES}
    if active:
        log("leak check after the exclusion (same draws and seeds) ...")
        LK = leak_check(ua, Streams(RANDOM_SEED, "preview_pair_rules_r3_v5b",
                                    base_h), ck,
                        label=" (after the exclusion)", expected_ref=cur_a)
    else:
        LK = LK0
    best_cells, best_swaps = best_tables(LK)

    img, odd_stats = image_oddness(m)
    u = pair_oddness(u, img)
    ua = u[~u.x_excl] if active else u
    sc = m.drop_duplicates("scene_id").indoor.value_counts()
    share = (sc / sc.sum()).to_dict()

    def run_selection():
        s2 = Streams(RANDOM_SEED, "preview_pair_rules_r3_v5b", base_h,
                     "gallery")
        sel, notes, excl = select(u, s2, share)
        a, an_notes = pick_anchors(u, r1, r2, r2_anchor_keys,
                                   set(sel.pair_key), s2("anchors"))
        a = a.assign(selected_as="anchor", group_rule="")
        return sel, notes + an_notes, excl, a

    sel0, notes, excl, anc0 = run_selection()
    sel2, _, _, anc2 = run_selection()
    allsel0 = pd.concat([sel0, anc0])
    for col in ("r1_rating", "r2_rating", "source", "kind"):
        allsel0[col] = allsel0[col].fillna("") if col in allsel0 else ""
    rep_info = pd.DataFrame()
    if active:
        allsel, rep_info, rep_notes = replace_affected(
            u, allsel0, Streams(RANDOM_SEED, "preview_pair_rules_r3_v5b",
                                base_h, "gallery"), r1, r2, r2_anchor_keys)
        notes = notes + rep_notes
        for i, row in excl.iterrows():
            t = [k for k, v in TYPE_LABEL.items() if v == row.type][0]
            c = ua[(ua.type == t) & ua.excl_cat.str.split("; ").map(
                lambda x, nm=row.category: nm in x)]
            excl.loc[i, "candidates"] = len(c)
            excl.loc[i, "scenes"] = c.scene_id.nunique()
    else:
        allsel = allsel0
    sel = allsel[allsel.selected_as != "anchor"]
    anc = allsel[allsel.selected_as == "anchor"]
    ck.add("selection", "the selection is reproducible (same settings, same "
           "pairs)", list(sel0.pair_key) + list(anc0.pair_key)
           == list(sel2.pair_key) + list(anc2.pair_key),
           f"{len(sel0) + len(anc0)} pairs")
    ok_new = sel[sel.selected_as.isin(["typical", "odd"])]
    ck.add("selection", "every typical pair meets the final rules (and is "
           "of its type)", bool(ok_new.meets_r3.all()),
           f"{len(ok_new)} pairs")
    exc = sel[sel.selected_as == "excluded"]
    spec_of = {(t, s_["name"]): s_ for t in TYPES
               for s_ in EXCLUDED_CATEGORIES[t]}
    bad = []
    for r in exc.itertuples():
        s_ = spec_of[(r.type, r.group_rule)]
        if not (set(r.failed) == set(s_["replaces"])
                and bool(category_mask(u.loc[[r.Index]], s_).iloc[0])):
            bad.append(r.pair_key)
    ck.add("selection", "every excluded pair breaks exactly the rule it is "
           "listed under", not bad, f"{len(exc)} pairs"
           + (f"; wrong: {bad[:3]}" if bad else ""))
    want = sum(ANCHORS_FROM_R1.values()) + sum(ANCHORS_FROM_R2.values())
    a1 = r1.set_index("pair_key")
    a2 = r2.set_index("pair_key")
    in_files = all(
        (k in a2.index and a2.loc[k, "rating"] == rr2)
        and (rr1 == "" or (k in a1.index and a1.loc[k, "rating"] == rr1))
        for k, rr1, rr2 in zip(anc.pair_key, anc.r1_rating, anc.r2_rating))
    ck.add("anchors", "the anchors are found in the round-1 and round-2 "
           "files with the ratings used", bool(in_files),
           f"{len(anc)} anchors: " + ", ".join(
               f"{k} {v}" for k, v in anc.kind.value_counts().items()))
    t_ok = all(a2.loc[k, "type"] == TYPE_LABEL[t]
               for k, t in zip(anc.pair_key, anc.type))
    ck.add("anchors", "the anchors' typing agrees with the ratings files",
           bool(t_ok), "")
    ck.add("anchors", f"{want} anchors, or the report says why fewer",
           len(anc) == want or any("anchors" in n_ for n_ in notes),
           f"{len(anc)}")
    ck.add("selection", "no pair is shown twice (anchors included)",
           allsel.pair_key.is_unique, f"{len(allsel)} pairs")
    sizes = {"typical": N_TYPICAL, "excluded": N_EXCLUDED}
    if N_ODD:
        sizes["odd"] = N_ODD
    got = sel.groupby(["type", "selected_as"]).size()
    lower = [f"{TYPE_LABEL[t]}/{g}: {int(got.get((t, g), 0))} of {n}"
             for t in TYPES for g, n in sizes.items()
             if got.get((t, g), 0) < n]
    ck.add("selection", "the number of pairs per type and group is as set, "
           "or the report (section 4) says why it is lower",
           not lower or bool(notes),
           "all as set" if not lower else "; ".join(lower))
    ex0 = build_examples(allsel0, Streams(RANDOM_SEED,
                                          "preview_pair_rules_r3_v5b",
                                          base_h, "gallery")("sides"), C)
    if active:
        reps = allsel[allsel.replaces_pair != ""]
        ex_rep = build_examples(reps, Streams(
            RANDOM_SEED, "preview_pair_rules_r3_v5b", base_h,
            "gallery")("replacement sides"), C)
        ex_rep.index = reps.replaces_pair.to_numpy()
        ex = ex0.copy()
        ex["replaces_pair"] = ""
        for i in range(len(ex)):
            k = ex.pair_key.iloc[i]
            if k in ex_rep.index:
                row = ex_rep.loc[k]
                for col in ex_rep.columns:
                    ex.iat[i, ex.columns.get_loc(col)] = row[col]
                ex.iat[i, ex.columns.get_loc("replaces_pair")] = k
    else:
        ex = ex0
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    files = sorted(set(ex.left_file) | set(ex.right_file))
    missing = [fn for fn in files if not os.path.isfile(
        os.path.join(info["stimuli_dir"], fn))]
    bad = []
    for fn in files:
        if fn in missing:
            continue
        src = os.path.join(info["stimuli_dir"], fn)
        dst = os.path.join(out, "images", fn)
        shutil.copy2(src, dst)
        if sha256_file(dst) != sha256_file(src):
            bad.append(fn)
    an_files = set(ex[ex.anchor].left_file) | set(ex[ex.anchor].right_file)
    ck.add("anchors", "the anchors' images exist (stimuli and copied)",
           not (an_files & set(missing)) and all(
               os.path.isfile(os.path.join(out, "images", f))
               for f in an_files), f"{len(an_files)} images")
    ck.add("gallery", "the images used are copied to images/ unchanged",
           not bad and not missing, f"{len(files)} images")
    ex = write_gallery(ex, out, h, order_h=base_h)
    ex = ex.sort_values("pair_number")
    cols = ["pair_number"] + [c for c in ex.columns if c != "pair_number"]
    ex[cols].to_csv(os.path.join(out, "examples.csv"), index=False)
    supply = supply_table(ua)
    supply.to_csv(os.path.join(out, "supply.csv"), index=False)
    LK["pool"].to_csv(os.path.join(out, "leak_pool.csv"), index=False)
    LK["cells"].to_csv(os.path.join(out, "leak_cells.csv"), index=False)
    LK["helpers"].to_csv(os.path.join(out, "leak_helpers.csv"), index=False)
    LK["draws"].to_csv(os.path.join(out, "leak_draws.csv"), index=False)
    LK["summary"].to_csv(os.path.join(out, "leak_achievable.csv"),
                         index=False)
    best_cells.to_csv(os.path.join(out, "leak_best_cells.csv"), index=False)
    best_swaps.to_csv(os.path.join(out, "leak_best_swaps.csv"), index=False)
    LK["feasible_cells"].to_csv(os.path.join(out, "leak_feasible_cells.csv"),
                                index=False)
    LK["helper_cells"].to_csv(os.path.join(out, "leak_helper_cells.csv"),
                              index=False)
    excl_lines = []
    if active:
        base_ok = check_base_run(ck, u, LK0, ex0, base_h)
        check_removed(ck, u, LK0, LK)
        check_exclusion_outputs(ck, ex, allsel, rep_info, LK)
        supply0 = supply_table(u)
        supply0.to_csv(os.path.join(out, "supply_before_exclusion.csv"),
                       index=False)
        LK0["pool"].to_csv(os.path.join(out, "leak_pool_before_exclusion"
                                        ".csv"), index=False)
        LK0["summary"].to_csv(os.path.join(
            out, "leak_achievable_before_exclusion.csv"), index=False)
        LK0["draws"].to_csv(os.path.join(out, "leak_draws_before_exclusion"
                                         ".csv"), index=False)
        LK["reuse"].to_csv(os.path.join(out, "leak_scene_reuse.csv"),
                           index=False)
        rep_info.to_csv(os.path.join(out, "replacements.csv"), index=False)
        excl_lines = exclusion_report(m, u, ua, supply0, supply, LK0, LK,
                                      rep_info, base_h, base_ok)
    sc_lines, sc_meta = [], None
    if SCENE_CHECK_RUN:
        sc_lines, sc_meta = scene_check(ck, u, m, out, h, base_h, C, info)
    sj = {"settings": settings_dict(), "settings_hash": h}
    if active:
        sj["exclusion"] = {"EXCLUDE_SCENES": EXCLUDE_SCENES,
                           "EXCLUDE_IMAGES": EXCLUDE_IMAGES,
                           "EXCLUDE_REASON": EXCLUDE_REASON,
                           "BASE_RUN": BASE_RUN,
                           "base_settings_hash": base_h}
    if sc_meta:
        sj["scene_check"] = sc_meta
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(sj, f, indent=2, default=str)
    meta = {"script": "preview_pair_rules_r3_v5b.py",
            "script_sha256": sha256_file(os.path.abspath(__file__)),
            "settings_hash": h, "round": ROUND,
            "manifest_sha256": info["manifest_sha256"],
            "anchors_files": anchors,
            "anchors_sha256": [sha256_file(a) for a in anchors],
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "started": time.strftime("%Y-%m-%d %H:%M:%S",
                                     time.localtime(T0))}
    if active:
        meta["base_settings_hash"] = base_h
    with open(os.path.join(out, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for g, d0 in digests.items():
        ck.add("rounds", f"the folder {g} is untouched",
               folder_digest(g) == d0, g if d0 else f"{g} not found here")
    pd.DataFrame(ck.rows).to_csv(os.path.join(out, "checks.csv"),
                                 index=False)
    write_report(out, ck, h, info, {
        "minutes": (time.time() - T0) / 60, "anchors_files": anchors,
        "C": C, "supply": supply, "cur": cur, "u": ua, "ex": ex,
        "notes": notes, "excl": excl, "scene_split": sc.to_dict(),
        "shipped": shipped, "leak": LK, "best_cells": best_cells,
        "best_swaps": best_swaps, "conclusion": conclusion(LK),
        "exclusion_lines": excl_lines, "scene_check_lines": sc_lines})
    if ck.n_fail:
        log(f"STOPPED: {ck.n_fail} check(s) failed; see {out}/report.txt")
        sys.exit(1)
    log(f"done: {len(ex)} pairs; open {out}/gallery.html")


# ------------------------------------------------------------
# Exclusion: checks and report
# ------------------------------------------------------------

def check_base_run(ck, u, LK0, ex0, base_h):
    """The results without the exclusion equal BASE_RUN, file by file."""
    sj = os.path.join(BASE_RUN, "settings.json")
    if not os.path.isfile(sj):
        ck.add("exclusion", "the results without the exclusion equal the "
               "base run (BASE_RUN)", False, f"{BASE_RUN} not found")
        return False
    with open(sj) as f:
        bs = json.load(f)
    ck.add("exclusion", "the settings without the exclusion give the base "
           "run's hash", bs.get("settings_hash") == base_h,
           f"{base_h} vs {bs.get('settings_hash')} ({BASE_RUN})")
    ck.add("exclusion", "the leak check uses the base run's number of draws",
           bs["settings"].get("LEAK_N_DRAWS") == LEAK_N_DRAWS,
           f"{LEAK_N_DRAWS}")
    bc, bsw = best_tables(LK0)
    exn = number_pairs(ex0, base_h)[0].sort_values("pair_number")
    cols = ["pair_number"] + [c for c in exn.columns if c != "pair_number"]
    mine = {"supply.csv": supply_table(u),
            "leak_pool.csv": LK0["pool"], "leak_cells.csv": LK0["cells"],
            "leak_helpers.csv": LK0["helpers"],
            "leak_helper_cells.csv": LK0["helper_cells"],
            "leak_draws.csv": LK0["draws"],
            "leak_achievable.csv": LK0["summary"],
            "leak_feasible_cells.csv": LK0["feasible_cells"],
            "leak_best_cells.csv": bc, "leak_best_swaps.csv": bsw,
            "examples.csv": exn[cols]}
    diff = []
    for fn, df in mine.items():
        fp = os.path.join(BASE_RUN, fn)
        if not os.path.isfile(fp):
            diff.append(f"{fn} missing")
            continue
        with open(fp) as f:
            if f.read() != df.to_csv(index=False):
                diff.append(fn)
    ok = not diff
    ck.add("exclusion", "the results without the exclusion equal the base "
           f"run, file by file ({len(mine)} files)", ok,
           "identical" if ok else "differ: " + ", ".join(diff))
    return ok


def check_removed(ck, u, LK0, LK):
    """Supply and pool leak after = before minus the removed pairs."""
    xk = set(u.pair_key[u.x_excl])
    bad, n_rm = [], {}
    for name in LK0["rulesets"]:
        q0 = LK0["best"][name][0].q
        q1 = LK["best"][name][0].q
        if set(q1.pair_key) != set(q0.pair_key) - xk:
            bad.append(f"{name}: pairs")
        rows = pd.DataFrame(pool_leak_rows(name, q0[~q0.pair_key.isin(xk)]))
        got = LK["pool"][LK["pool"].rules == name].reset_index(drop=True)
        num = [c for c in rows.columns if c.startswith(("b_", "share",
                                                        "pairs", "scenes"))]
        if not np.allclose(rows[num].to_numpy(float),
                           got[num].to_numpy(float), equal_nan=True,
                           rtol=0, atol=1e-12):
            bad.append(f"{name}: pool leak")
        n_rm[name] = len(q0) - len(q1)
    s0 = supply_table(u)
    s1 = supply_table(u[~u.x_excl])
    for t in TYPES:
        a = int(s0[(s0.type == TYPE_LABEL[t])
                   & (s0.levels == "all")].pairs.iloc[0])
        b = int(s1[(s1.type == TYPE_LABEL[t])
                   & (s1.levels == "all")].pairs.iloc[0])
        rm = int((u.meets_r3 & u.x_excl & (u.type == t)).sum())
        if a - b != rm:
            bad.append(f"supply {t}")
    ck.add("exclusion", "supply and pool leak after the exclusion = before "
           "minus the removed pairs (every rule set; the pool leak "
           "recomputed)", not bad, "; ".join(bad) if bad else
           "removed per rule set: " + ", ".join(
               f"{short_rules(k)} {v:,}" for k, v in n_rm.items()))


def check_exclusion_outputs(ck, ex, allsel, rep_info, LK):
    hit = excluded_mask(ex, "scene", ("left_file", "right_file"))
    ck.add("exclusion", "no gallery pair, anchor or example uses an excluded "
           "scene or image", not hit.any() and not excluded_mask(
               allsel).any(), f"{int(hit.sum())} found" if hit.any()
           else f"{len(ex)} pairs")
    an = rep_info[rep_info.group == "anchor"] if len(rep_info) else rep_info
    relaxed = int((~an.matched.str.startswith("same source, rating")).sum()
                  ) if len(an) else 0
    ck.add("exclusion", "anchors from excluded scenes are replaced, from the "
           "same source round, rating and type where any is left (else "
           "relaxed and listed in section E5)",
           bool(len(an) == 0 or (an["new pair"] != "").all()),
           f"{len(an)} replaced, {relaxed} relaxed")
    ru = LK["reuse"]
    dr = ru[ru.stage == "drawn"]
    ck.add("exclusion", "every drawn set uses each scene once per type "
           "before any twice (after the exclusion)",
           bool(dr["each scene once before any twice"].all()),
           f"{len(dr) // len(TYPES)} sets")
    sw = ru[ru.stage == "after swaps"]
    ck.add("exclusion", "after the swaps no scene exceeds the per-type cap",
           bool((sw["most uses of one scene"] <= sw["cap"]).all()), "")


def verdict(pb, pa, n):
    se = math.sqrt(max(pb * (1 - pb), 1e-9) / n
                   + max(pa * (1 - pa), 1e-9) / n)
    d = pa - pb
    if abs(d) <= 2 * se:
        return "no clear change"
    return "easier" if d > 0 else "harder"


def exclusion_report(m, u, ua, supply0, supply, LK0, LK, rep_info, base_h,
                     base_ok):
    L = []
    w = L.append
    sc = m.drop_duplicates("scene_id").set_index("scene_id")
    left = sc.drop(EXCLUDE_SCENES)
    w("EXCLUSION")
    w("-" * 78)
    w(f"  reason: {EXCLUDE_REASON}")
    w(f"  base run (no exclusion): {BASE_RUN} (settings {base_h}); "
      "everything below is after")
    w("  the exclusion unless marked 'before'. The results without the "
      "exclusion "
      + ("equal the base run file by file." if base_ok else
         "DO NOT equal the base run (see the checks)."))
    w("")
    w("  E1. Scenes and images left out")
    if EXCLUDE_SCENES:
        t = sc.loc[EXCLUDE_SCENES, ["category", "group", "indoor"]] \
            .reset_index()
        w(tab(t).replace("\n", "\n  "))
    if EXCLUDE_IMAGES:
        w("    images: " + ", ".join(EXCLUDE_IMAGES))
    b0, b1 = sc.indoor.value_counts(), left.indoor.value_counts()
    w("    indoor / outdoor / ambiguous scenes: " + " / ".join(
        str(int(b0.get(k, 0))) for k in ("indoor", "outdoor", "ambiguous"))
      + " before, " + " / ".join(
        str(int(b1.get(k, 0))) for k in ("indoor", "outdoor", "ambiguous"))
      + " after")
    g0, g1 = sc.group.value_counts(), left.group.value_counts()
    w("    scene groups: " + "; ".join(
        f"{k} {int(g0[k])} -> {int(g1.get(k, 0))}" for k in sorted(g0.index)
        if g0[k] != g1.get(k, 0)) + " (others unchanged)")
    gone = sorted(set(sc.category) - set(left.category))
    w("    categories that disappear: " + (", ".join(gone) if gone
                                            else "none"))
    kept = sorted(set(sc.loc[EXCLUDE_SCENES, "category"]) - set(gone))
    if kept:
        w("    categories that remain with fewer scenes: " + ", ".join(
            f"{c} ({int((left.category == c).sum())} left)" for c in kept))
    w("")
    w("  E2. Supply under the final rules (default levers): pairs and "
      "scenes per type")
    rows = []
    for t in TYPES:
        a = supply0[(supply0.type == TYPE_LABEL[t])
                    & (supply0.levels == "all")].iloc[0]
        b = supply[(supply.type == TYPE_LABEL[t])
                   & (supply.levels == "all")].iloc[0]
        rows.append({"type": TYPE_LABEL[t], "pairs before": int(a.pairs),
                     "pairs after": int(b.pairs),
                     "removed": int(a.pairs - b.pairs),
                     "scenes before": int(a.scenes),
                     "scenes after": int(b.scenes)})
    w(tab(pd.DataFrame(rows)).replace("\n", "\n  "))
    cur0 = {t: int((u.current & (u.type == t)).sum()) for t in TYPES}
    cur1 = {t: int((ua.current & (ua.type == t)).sum()) for t in TYPES}
    w("    the current (v5units) pool: " + ", ".join(
        f"{TYPE_LABEL[t]} {cur0[t]:,} -> {cur1[t]:,}" for t in TYPES))
    w("")
    w("  E3. Leak before and after (same draws and seeds; "
      f"{LEAK_N_DRAWS} draws per rule set)")
    w("    Pool leak, the limited b (corpus / orig), before -> after:")
    p0, p1 = LK0["pool"], LK["pool"]
    rows = []
    for name in LK["rulesets"]:
        for t in [TYPE_LABEL[x] for x in TYPE_LIMITS] + ["overall"]:
            a = p0[(p0.rules == name) & (p0.type == t)].iloc[0]
            b = p1[(p1.rules == name) & (p1.type == t)].iloc[0]
            terms = a.terms.split("+")
            rows.append({"rules": short_rules(name), "type": t, **{
                f"b_{tm}": " / ".join(
                    f"{a[f'b_{tm} {d}']:+.3f} -> {b[f'b_{tm} {d}']:+.3f}"
                    for d in NORMALITY_DEFS) for tm in terms}})
    w(tab(pd.DataFrame(rows).fillna("")).replace("\n", "\n  "))
    s0 = LK0["summary"].set_index("rules")
    s1 = LK["summary"].set_index("rules")
    ref = [n for n in s0.index if n.startswith("reference")][0]
    cal = calibration_budget(s0, ref)
    top = max(LEAK_BUDGETS)
    n = 2 * LEAK_N_DRAWS
    w("")
    w(f"    Achievable leak, before -> after: draws meeting every limit with "
      f"{cal} swaps per set (the")
    w(f"    reference's yardstick before the exclusion) and with {top}; "
      "median swaps to meet every")
    w("    limit; sets meeting each limit with "
      f"{cal} swaps, and whether that is easier, harder or")
    w("    no clear change (beyond 2 standard errors of the difference, "
      f"{n} sets each):")
    rows, changes = [], []
    groups = list(limit_group_names())
    for name in LK["rulesets"]:
        r = {"rules": short_rules(name),
             f"draws @{cal}": f"{s0.loc[name, f'draws meeting all @{cal}']:.0%}"
             f" -> {s1.loc[name, f'draws meeting all @{cal}']:.0%}",
             f"draws @{top}": f"{s0.loc[name, f'draws meeting all @{top}']:.0%}"
             f" -> {s1.loc[name, f'draws meeting all @{top}']:.0%}",
             "median swaps": f"{s0.loc[name, 'median swaps to meet all']:.0f}"
             f" -> {s1.loc[name, 'median swaps to meet all']:.0f}"}
        for g in groups:
            pb = s0.loc[name, f"sets meeting {g} @{cal}"]
            pa = s1.loc[name, f"sets meeting {g} @{cal}"]
            v = verdict(pb, pa, n)
            r[g] = f"{pb:.0%} -> {pa:.0%} {v}"
            if v != "no clear change":
                changes.append(f"{g} under {short_rules(name)} ({v}: "
                               f"{pb:.0%} -> {pa:.0%})")
        rows.append(r)
    w(tab(pd.DataFrame(rows)).replace("\n", "\n  "))
    dflt = [x for x in LK["rulesets"] if x.endswith("(default)")][0]
    w("")
    n_cmp = len(LK["rulesets"]) * len(groups)
    w(f"    ({n_cmp} comparisons: about {round(0.05 * n_cmp)} would pass 2 "
      "standard errors by chance alone.)")
    if changes:
        w("    Limits clearly easier or harder after the exclusion:")
        for c in changes:
            for line in textwrap.wrap(c, 72, initial_indent="      - ",
                                      subsequent_indent="        "):
                w(line)
    else:
        w("    No limit becomes clearly easier or harder to meet after the "
          "exclusion, under any rule")
        w("    set (all differences within Monte Carlo error).")
    w(f"    Default rules: every limit met in "
      f"{s0.loc[dflt, f'draws meeting all @{cal}']:.0%} -> "
      f"{s1.loc[dflt, f'draws meeting all @{cal}']:.0%} of draws with "
      f"{cal} swaps; median swaps "
      f"{s0.loc[dflt, 'median swaps to meet all']:.0f} -> "
      f"{s1.loc[dflt, 'median swaps to meet all']:.0f}.")
    w("")
    w("  E4. Lists: scene use per type and set (after the exclusion, "
      "default rules, every draw)")
    ru = LK["reuse"]
    ru = ru[ru.rules == dflt]
    rows = []
    for t in TYPES:
        d0 = ru[(ru.type == TYPE_LABEL[t]) & (ru.stage == "drawn")]
        d1 = ru[(ru.type == TYPE_LABEL[t]) & (ru.stage == "after swaps")]
        rows.append({
            "type": TYPE_LABEL[t], "pairs per set": int(d0.pairs.iloc[0]),
            "scenes": int(d0["scenes in pool"].iloc[0]),
            "scenes used twice, drawn": f"{d0['scenes used twice or more'].min()}"
            f"-{d0['scenes used twice or more'].max()}",
            "each once before any twice": f"{d0['each scene once before any twice'].mean():.0%}",
            "most uses after swaps (cap)":
                f"{d1['most uses of one scene'].max()} ({d1.cap.iloc[0]})"})
    w(tab(pd.DataFrame(rows)).replace("\n", "\n  "))
    w(f"    Two disjoint sets were drawn in all {LEAK_N_DRAWS} draws of "
      "every rule set (see the checks).")
    w("")
    w("  E5. Gallery pairs replaced (the others are as in the base run, "
      "with the same number and")
    w("      side; replacements.csv):")
    if len(rep_info):
        w(tab(rep_info[["type", "group", "old scene", "new scene",
                        "matched"]]).replace("\n", "\n  "))
    else:
        w("    none")
    w("")
    return L


# ------------------------------------------------------------
# Scene check
# ------------------------------------------------------------

def scene_check(ck, u, m, out, h, base_h, C, info):
    """6 pairs per checked scene and per comparison scene; a small
    gallery of its own."""
    st = Streams(RANDOM_SEED, "preview_pair_rules_r3_v5b", base_h,
                 "scene check")
    sc = m.drop_duplicates("scene_id").set_index("scene_id")
    checked = list(SCENE_CHECK_SCENES)
    rest = sorted(s for s in sc.index if s not in checked
                  and s not in EXCLUDE_SCENES)
    labels = [sc.loc[s, "indoor"] for s in checked]
    labels = (labels * (SCENE_CHECK_N_COMPARISON // max(len(labels), 1) + 1)
              )[:SCENE_CHECK_N_COMPARISON]
    rng = st("comparison scenes")
    comp = []
    for lab in labels:
        pool = [s for s in rest if sc.loc[s, "indoor"] == lab
                and s not in comp]
        pool = pool or [s for s in rest if s not in comp]
        comp.append(pool[int(rng.integers(len(pool)))])
    rows = []
    for role, scenes in (("checked", checked), ("comparison", comp)):
        for s_ in scenes:
            r = st(f"pairs|{s_}")
            q = u[u.meets_r3 & (u.scene_id == s_)]
            first = {}
            for t in TYPES + SCENE_CHECK_EXTRA_TYPES:
                c = q[(q.type == t) & ~q.pair_key.isin(
                    [x.pair_key for x in rows])]
                if t in first:
                    c2 = c[c.levels != first[t]]
                    c = c2 if len(c2) else c
                if c.empty:
                    continue
                pick = c.iloc[[int(r.integers(len(c)))]]
                first.setdefault(t, pick.levels.iloc[0])
                rows.append(pick.assign(selected_as="scene check",
                                        group_rule="", role=role).iloc[0])
    sel = pd.DataFrame(rows)
    for col in ("r1_rating", "r2_rating", "source", "kind"):
        sel[col] = ""
    ex = build_examples(sel, st("sides"), C)
    ex["group"] = "scene check"
    ex["role"] = sel.role.to_numpy()
    for fn in sorted(set(ex.left_file) | set(ex.right_file)):
        dst = os.path.join(out, "images", fn)
        if not os.path.isfile(dst):
            shutil.copy2(os.path.join(info["stimuli_dir"], fn), dst)
    hsc = f"{h}_scene_check"
    ex = write_gallery(ex, out, hsc, order_h=f"{base_h}_scene_check",
                       fname="scene_check.html",
                       title=f"Scene check v5b ({h})",
                       bar=f"scene check · settings {h}",
                       groups=["scene check"])
    ex = ex.sort_values("pair_number")
    cols = ["pair_number", "role"] + [c for c in ex.columns
                                      if c not in ("pair_number", "role")]
    ex[cols].to_csv(os.path.join(out, "scene_check.csv"), index=False)
    per = ex.groupby(["role", "scene"]).size()
    ck.add("scene check", "6 pairs under the final rules per scene "
           f"({len(checked)} checked, {len(comp)} comparison)",
           bool((per == len(TYPES) + len(SCENE_CHECK_EXTRA_TYPES)).all()
                and ex.meets_final_rules.all()),
           f"{len(ex)} pairs")
    lab_c = sorted(sc.loc[checked, "indoor"])
    lab_k = sorted(sc.loc[comp, "indoor"])
    ck.add("scene check", "comparison scenes are other scenes, matched for "
           "indoor/outdoor where possible",
           not (set(comp) & set(checked)) and len(set(comp)) == len(comp),
           f"checked {', '.join(lab_c)}; comparison {', '.join(lab_k)}")
    L = ["SCENE CHECK (scene_check.html, scene_check.csv)", "-" * 78]
    L += textwrap.wrap(
        f"The rule, fixed before any rating: a scene is excluded if "
        f"{SCENE_CHECK_RULE_MIN} or more of its "
        f"{len(TYPES) + len(SCENE_CHECK_EXTRA_TYPES)} pairs are rated "
        "'Too similar'. Its purpose is a documented, pre-data reason for "
        "the exclusion. The pairs meet the final rules: one per type, plus "
        "one more colour-only and one more contrast-only pair at another "
        "distance; shuffled, sides at random, labels hidden until rated. "
        "Comparison scenes: drawn with the seed from the scenes not checked "
        "or excluded, matched for indoor/outdoor where possible.", 76,
        initial_indent="  ", subsequent_indent="  ")
    t = pd.DataFrame({"scene": checked + comp,
                      "role": ["checked"] * len(checked)
                      + ["comparison"] * len(comp)})
    t["category"] = t.scene.map(sc.category)
    t["indoor"] = t.scene.map(sc.indoor)
    t["pairs (levels apart)"] = [
        ", ".join(f"{a} {b}" for a, b in zip(
            ex[ex.scene == s_].type, ex[ex.scene == s_].levels_apart))
        for s_ in t.scene]
    L += [tab(t), ""]
    meta = {"SCENE_CHECK_SCENES": checked, "comparison_scenes": comp,
            "SCENE_CHECK_N_COMPARISON": SCENE_CHECK_N_COMPARISON,
            "SCENE_CHECK_EXTRA_TYPES": SCENE_CHECK_EXTRA_TYPES,
            "SCENE_CHECK_RULE_MIN": SCENE_CHECK_RULE_MIN,
            "rule": f"exclude a scene if >= {SCENE_CHECK_RULE_MIN} of its "
                    "6 pairs are rated 'Too similar' (fixed before rating)"}
    return L, meta


# ------------------------------------------------------------
# Summary mode
# ------------------------------------------------------------

def galleries():
    """{gallery: set of pair keys}: each run folder's examples.csv, and its
    scene_check.csv as '<folder>#scene_check'."""
    out = {}
    if not os.path.isdir(OUTPUT_ROOT):
        return out
    for d in sorted(os.listdir(OUTPUT_ROOT)):
        f = os.path.join(OUTPUT_ROOT, d, "examples.csv")
        if os.path.isfile(f):
            out[os.path.join(OUTPUT_ROOT, d)] = set(pd.read_csv(
                f, keep_default_na=False).pair_key)
        f = os.path.join(OUTPUT_ROOT, d, "scene_check.csv")
        if os.path.isfile(f):
            out[os.path.join(OUTPUT_ROOT, d) + "#scene_check"] = set(
                pd.read_csv(f, keep_default_na=False).pair_key)
    return out


def gallery_csv(g):
    if g.endswith("#scene_check"):
        return os.path.join(g[:-len("#scene_check")], "scene_check.csv")
    return os.path.join(g, "examples.csv")


def folder_hash(folder):
    if folder.endswith("#scene_check"):
        return folder_hash(folder[:-len("#scene_check")]) + "_scene_check"
    return os.path.basename(folder).split("_run")[0]


def find_gallery(r, path, gals):
    keys = set(r.pair_key)
    if "settings_hash" in r.columns and r.settings_hash.nunique() == 1:
        hint = r.settings_hash.iloc[0]
    else:
        mt = re.search(r"ratings_([0-9a-f]{10}(_scene_check)?)",
                       os.path.basename(path))
        hint = mt.group(1) if mt else None
    fits = [g for g, k in gals.items() if keys <= k]
    if hint:
        pref = [g for g in fits if folder_hash(g) == hint]
        fits = pref or fits
    if not fits:
        return None, hint
    return fits[-1], hint


def round_name(folder, h_now):
    fh = folder_hash(folder)
    if fh.endswith("_scene_check"):
        return "scene check"
    if fh == ROUND1_HASH:
        return "round 1"
    if fh == ROUND2_HASH:
        return "round 2"
    if fh == h_now:
        return f"round {ROUND}"
    return f"settings {fh}"


def rating_counts(df, by):
    cats = RATINGS + ["(unrated)"]
    if df.empty:
        return pd.DataFrame(columns=by + cats + ["n"])
    t = df.groupby(by + ["rating"]).size().unstack(fill_value=0)
    for c in cats:
        if c not in t.columns:
            t[c] = 0
    t = t[cats]
    t["n"] = t.sum(axis=1)
    return t.reset_index()


def main_summary(paths):
    h = settings_hash()
    gals = galleries()
    m = read_manifest()
    u, C = build_pairs(m)
    fields = u.set_index("pair_key")[
        ["type", "levels", "position", "where", "meets_r3", "scene_id"]]
    fields = fields.assign(type=fields.type.map(TYPE_LABEL))
    rows, info = [], []
    for i, path in enumerate(paths, 1):
        r = read_ratings(path)
        g, hint = find_gallery(r, path, gals)
        if g is None:
            sys.exit(f"{path}: no gallery under {OUTPUT_ROOT} holds all its "
                     f"pair keys (hash hint {hint})")
        ex = pd.read_csv(gallery_csv(g), keep_default_na=False, dtype=str)
        for col, dflt in (("selected_as", None), ("anchor", "False"),
                          ("reason1", ""), ("reason2", ""),
                          ("exclusion_rule", ""), ("role", "")):
            if col not in ex.columns:
                ex[col] = ex["group"] if dflt is None else dflt
        j = r[["pair_key", "rating", "note"]].merge(
            ex[["pair_key", "selected_as", "anchor", "reason1", "reason2",
                "exclusion_rule", "role"]], on="pair_key", how="left")
        j = j.join(fields, on="pair_key")
        j["rating"] = j.rating.replace("", "(unrated)")
        j["round"] = round_name(g, h)
        j["file"] = i
        rows.append(j)
        info.append({"file": i, "path": path, "gallery": g,
                     "round": round_name(g, h), "pairs": len(r),
                     "rated": int((j.rating != "(unrated)").sum()),
                     "unknown_keys": int(j.type.isna().sum())})
    allr = pd.concat(rows, ignore_index=True)
    allr["is_anchor"] = allr.anchor.astype(str).str.lower() == "true"
    fa = allr.pair_key.str.split("__vs__").str[0]
    fb = allr.pair_key.str.split("__vs__").str[1]
    allr["x_excl"] = (allr.scene_id.isin(EXCLUDE_SCENES) | fa.isin(
        EXCLUDE_IMAGES) | fb.isin(EXCLUDE_IMAGES))
    scr = allr[allr["round"] == "scene check"]
    xr = allr[allr.x_excl & (allr["round"] != "scene check")]
    every = allr[allr["round"] != "scene check"]
    allr = allr[~allr.x_excl & (allr["round"] != "scene check")]
    allr["dist_where"] = np.where(
        allr.type.isin(["colour only", "contrast only"]),
        allr.levels.astype(str) + " | " + allr["where"].astype(str),
        allr.levels.astype(str))
    runs = run_folders(h)
    base = runs[-1] if runs else info[-1]["gallery"]
    k = 1
    while os.path.exists(os.path.join(base, f"ratings_{k}")):
        k += 1
    out = os.path.join(base, f"ratings_{k}")
    os.makedirs(out)
    for i, path in enumerate(paths, 1):
        shutil.copy2(path, os.path.join(out, f"ratings_input_{i}.csv"))
    tables = []

    def add(name, df):
        if len(df):
            tables.append(df.assign(table=name))
        return df

    def drop(df):
        return df.drop(columns=["round", "table"], errors="ignore")

    L = ["SUMMARY OF RATINGS  (preview_pair_rules_r3_v5b.py --ratings)",
         "=" * 78, f"output    {out}", "", "Files:", tab(pd.DataFrame(info))]
    if exclusion_active():
        L += [f"  Excluded: scenes {', '.join(EXCLUDE_SCENES) or 'none'}; "
              f"images {', '.join(EXCLUDE_IMAGES) or 'none'}.",
              f"  Reason: {EXCLUDE_REASON}",
              f"  Their {len(xr)} ratings are left out of the main tables "
              "and reported in EXCLUDED SCENES below."]
    if len(scr):
        L += [f"  Scene-check ratings ({len(scr)}) are reported in SCENE "
              "CHECK below, not in the main tables."]
    L += [
         "  Breakdowns use round-3 definitions for every round (from the "
         "pair key): colour only and",
         "  contrast only by levels apart | where the pair sits (low = both "
         "at or below 0, high =",
         "  both at or above 0, else straddling); mixed types by level "
         "combination (colour/contrast)."]
    single = ["colour only", "contrast only"]
    for rnd, d in allr.groupby("round", sort=True):
        L += ["", "=" * 78, rnd.upper(), "=" * 78]
        a = add(f"{rnd}: type x group",
                rating_counts(d, ["round", "type", "selected_as"]))
        L += ["By type and group:", tab(drop(a))]
        dn = d[~d.is_anchor]
        b = add(f"{rnd}: single types by levels apart x where",
                rating_counts(dn[dn.type.isin(single)],
                              ["round", "type", "levels", "where"]))
        c = add(f"{rnd}: mixed types by level combination",
                rating_counts(dn[~dn.type.isin(single)],
                              ["round", "type", "levels"]))
        L += ["", "Colour only and contrast only by levels apart x where "
              "(anchors left out):", tab(drop(b)),
              "", "Mixed types by level combination:", tab(drop(c))]
        xd = d[d.selected_as == "excluded"]
        if len(xd):
            e = add(f"{rnd}: excluded by rule",
                    rating_counts(xd, ["round", "type", "exclusion_rule"]))
            L += ["", "Excluded pairs by rule:", tab(drop(e))]
        od = d[d.selected_as == "odd"]
        rs = pd.concat([od.assign(reason=od.reason1),
                        od.assign(reason=od.reason2)])
        rs = rs[rs.reason.fillna("") != ""]
        if len(rs):
            e = add(f"{rnd}: odd by reason",
                    rating_counts(rs, ["round", "type", "reason"]))
            L += ["", "Odd pairs by reason:", tab(drop(e))]
    rounds = sorted(allr["round"].unique())
    if len(rounds) > 1:
        d = allr[~allr.is_anchor & (allr.rating != "(unrated)")]
        per = d.groupby(["type", "dist_where", "round"]).size().unstack(
            fill_value=0)
        per.columns = [f"n {c}" for c in per.columns]
        pc = rating_counts(d, ["type", "dist_where"]).join(
            per, on=["type", "dist_where"]).join(
            d.groupby(["type", "dist_where"]).meets_r3.mean().rename(
                "share meeting final rules"), on=["type", "dist_where"])
        add("all rounds together", pc)
        L += ["", "=" * 78, "ALL ROUNDS TOGETHER (rated pairs; anchor "
              "repeats left out)", "=" * 78,
              "Single types: levels apart | where; mixed: level combination."
              " 'share meeting final",
              "rules': the share of the cell's rated pairs allowed now.",
              tab(pc)]
    an_keys = set(allr[allr.is_anchor].pair_key)
    if an_keys:
        a = allr[allr.pair_key.isin(an_keys) & (allr.rating != "(unrated)")]
        wide = a.pivot_table(index=["pair_key", "type", "levels", "where"],
                             columns="round", values="rating",
                             aggfunc="first").reset_index()
        rcols = [c for c in wide.columns if str(c).startswith("round")
                 or str(c).startswith("settings")]
        wide["ratings"] = wide[rcols].notna().sum(axis=1)
        wide["all agree"] = [
            ("yes" if len(set(v.dropna())) == 1 else "no")
            if v.notna().sum() >= 2 else "" for _, v in
            wide[rcols].iterrows()]
        add("anchors", wide)
        L += ["", "=" * 78, "ANCHORS ACROSS ROUNDS", "=" * 78,
              tab(wide.drop(columns="pair_key").fillna(""))]
        rr = wide[wide["all agree"] != ""]
        if len(rr):
            L.append(f"  all ratings agree: {(rr['all agree'] == 'yes').sum()}"
                     f" of {len(rr)} pairs rated at least twice "
                     f"({(rr['all agree'] == 'yes').mean():.0%})")
        for x, y in itertools.combinations(rcols, 2):
            both = wide[wide[x].notna() & wide[y].notna()]
            if len(both):
                L.append(f"  {x} vs {y}: {(both[x] == both[y]).sum()} of "
                         f"{len(both)} agree "
                         f"({(both[x] == both[y]).mean():.0%})")
    if exclusion_active() or len(scr):
        L += excluded_scene_tables(every, xr, add)
    if len(scr):
        L += scene_check_tables(scr, add)
    notes = pd.concat([allr, xr, scr])
    notes = notes[notes.note.fillna("").str.strip() != ""][
        ["round", "type", "selected_as", "scene_id", "rating", "note"]]
    if len(notes):
        L += ["", "Notes:", tab(notes)]
    summ = pd.concat(tables, ignore_index=True)
    summ = summ[["table"] + [c for c in summ.columns if c != "table"]]
    summ.to_csv(os.path.join(out, "ratings_summary.csv"), index=False)
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"summary written to {out}")


def excluded_scene_tables(every, xr, add):
    """Evidence: each excluded scene's ratings per round, split by whether
    the pair meets the final rules; the remaining scenes for comparison;
    every scene's share of 'Too similar'."""
    L = []
    d = every[every.rating != "(unrated)"].copy()
    d["final rules"] = np.where(d.meets_r3.fillna(False).astype(bool), "meets",
                                "does not meet")
    d["scenes"] = np.where(d.x_excl, d.scene_id, "remaining scenes")
    if exclusion_active():
        L += ["", "=" * 78, "EXCLUDED SCENES: THE EVIDENCE", "=" * 78,
              "Ratings per round, split by whether the pair meets the final "
              "rules (round-1 pairs",
              "were mostly 1 level apart and too similar in every scene). "
              "Anchor repeats included.",
              "Excluded scenes, one by one:"]
        a = add("excluded scenes: ratings by round and final rules",
                rating_counts(d[d.x_excl], ["scenes", "round",
                                            "final rules"]))
        L.append(tab(a.drop(columns="table", errors="ignore")))
        b = add("remaining scenes: ratings by round and final rules",
                rating_counts(d[~d.x_excl], ["scenes", "round",
                                             "final rules"]))
        L += ["", "All remaining scenes together, for comparison:",
              tab(b.drop(columns="table", errors="ignore"))]
    per = []
    for (sid, rnd), g in d.groupby(["scene_id", "round"]):
        f = g[g.meets_r3.fillna(False).astype(bool)]
        per.append({"scene": sid, "round": rnd, "excluded": bool(
            g.x_excl.any()), "rated": len(g),
            "too similar": int((g.rating == "Too similar").sum()),
            "rated, final rules": len(f),
            "too similar, final rules": int((f.rating == "Too similar")
                                            .sum())})
    per = pd.DataFrame(per)
    if len(per):
        tot = per.groupby(["scene", "excluded"])[[
            "rated", "too similar", "rated, final rules",
            "too similar, final rules"]].sum().reset_index()
        tot["share too similar"] = tot["too similar"] / tot.rated
        tot["share too similar, final rules"] = (
            tot["too similar, final rules"]
            / tot["rated, final rules"].replace(0, np.nan))
        wide = per.pivot_table(index="scene", columns="round",
                               values="too similar", aggfunc="sum")
        rated = per.pivot_table(index="scene", columns="round",
                                values="rated", aggfunc="sum")
        for c in wide.columns:
            tot[f"{c}: too similar / rated"] = tot.scene.map(
                lambda x, c=c: (f"{int(wide.loc[x, c])}/{int(rated.loc[x, c])}"
                                if x in wide.index and not pd.isna(
                                    wide.loc[x, c]) else ""))
        tot = tot.sort_values("share too similar", ascending=False)
        add("by scene: share too similar (all rounds)", tot)
        top = tot.head(10)
        rest = tot[tot.excluded & ~tot.scene.isin(top.scene)]
        show = pd.concat([top, rest])
        show = show.assign(rank=[int(np.flatnonzero(
            tot.scene.to_numpy() == x)[0]) + 1 for x in show.scene])
        L += ["", "Every scene's share of 'Too similar' over all rounds "
              "(the top 10, and the excluded",
              f"scenes wherever they rank, of {len(tot)}; all scenes in "
              "ratings_summary.csv):",
              tab(show[["rank", "scene", "excluded", "rated",
                        "share too similar", "share too similar, "
                        "final rules"] + [c for c in show.columns
                                          if c.endswith("/ rated")]], 2)]
    return L


def scene_check_tables(scr, add):
    """Each scene's count of 'Too similar' against the rule fixed before
    rating."""
    n_pairs = len(TYPES) + len(SCENE_CHECK_EXTRA_TYPES)
    rows = []
    for (sid, role), g in scr.groupby(["scene_id", "role"]):
        ts = int((g.rating == "Too similar").sum())
        rated = int((g.rating != "(unrated)").sum())
        rows.append({"scene": sid, "role": role, "pairs": len(g),
                     "rated": rated, "too similar": ts,
                     "rule: exclude": "yes" if ts >= SCENE_CHECK_RULE_MIN
                     else ("no" if rated == len(g)
                           and ts < SCENE_CHECK_RULE_MIN else
                           "not all rated"),
                     "excluded in these settings": sid in EXCLUDE_SCENES})
    t = pd.DataFrame(rows).sort_values(["role", "too similar"],
                                       ascending=[True, False])
    add("scene check: too similar per scene", t)
    return ["", "=" * 78, "SCENE CHECK", "=" * 78,
            f"The rule, fixed before rating: a scene is excluded if "
            f"{SCENE_CHECK_RULE_MIN} or more of its {n_pairs} pairs are "
            "rated 'Too similar'.", tab(t)]


def main():
    ap = argparse.ArgumentParser(description="Pair preview round 3 with a "
                                 "leak check (v5b).")
    ap.add_argument("--anchors", nargs=2, metavar=("ROUND1_CSV",
                                                   "ROUND2_CSV"),
                    help="the round-1 and round-2 ratings files (needed to "
                    "build the gallery)")
    ap.add_argument("--ratings", nargs="+", help="one or more downloaded "
                    "ratings files: summarise them instead")
    ap.add_argument("--exclude-scenes", nargs="*", metavar="SCENE",
                    help="replaces EXCLUDE_SCENES (no names: none)")
    ap.add_argument("--exclude-images", nargs="*", metavar="FILE",
                    help="replaces EXCLUDE_IMAGES (no names: none)")
    ap.add_argument("--exclude-reason", help="replaces EXCLUDE_REASON")
    ap.add_argument("--scene-check", action="store_true",
                    help="also write scene_check.html (SCENE_CHECK_SCENES)")
    a = ap.parse_args()
    g = globals()
    if a.exclude_scenes is not None:
        g["EXCLUDE_SCENES"] = list(a.exclude_scenes)
    if a.exclude_images is not None:
        g["EXCLUDE_IMAGES"] = list(a.exclude_images)
    if a.exclude_reason is not None:
        g["EXCLUDE_REASON"] = a.exclude_reason
    g["SCENE_CHECK_RUN"] = bool(a.scene_check)
    if (exclusion_active() or SCENE_CHECK_RUN) and os.path.isfile(
            MANIFEST_CSV):
        errs = validate_exclusion(read_manifest())
        if errs:
            sys.exit("preview_pair_rules_r3_v5b.py: the exclusion settings "
                     "are not valid (nothing was written):\n  - "
                     + "\n  - ".join(errs))
    if a.ratings:
        main_summary(a.ratings)
    elif a.anchors:
        main_gallery(a.anchors)
    else:
        ap.error("--anchors <round-1 csv> <round-2 csv> is needed to build "
                 "the gallery (or --ratings <csv> ... to summarise)")


if __name__ == "__main__":
    main()
