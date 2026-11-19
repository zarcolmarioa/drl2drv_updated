"""
make_trial_orders_v5c.py  --  v5c step 4: trial orders, assignment table,
package

WHAT CHANGED FROM make_trial_orders_v5b.py (v5b -> v5c)
    A copy of make_trial_orders_v5b.py. The order construction and balance
    rules, the assignment table, the package format and the checks are
    unchanged. Changed:
      * INPUTS AND NAMES: script 3 is build_trials_v5c.py's run (rule set
        final_r3, 42 scenes, practice 3 + 3); package v5c_180; output
        data/trial_orders_v5c/; every v5b label, report and README line is
        v5c.
      * PRACTICE: the counts and the feedback come from script 3's
        set_info.json (practice_by_block, practice_feedback_by_block_
        position); PRACTICE_FEEDBACK_TRIALS (3) practice trials give
        feedback. design.json and the README record them; a check confirms
        that design.json, set_info.json and the practice files agree.
      * SCENE GAP: v5b's rules and settings (gap 30, fixed; 4 orders per
        set). With 42 scenes instead of 47 the report states the largest
        gap at which all orders can be built and the gap used; if 30 cannot
        be reached the script stops (after reporting, for information, the
        largest gap below 30 that works) and never lowers it.
      * BALANCE AGAINST v5b_180 (report section 3b): the orders' mean
        position by type and by bin, the correlation of position with the
        size of the difference, how alike the orders are, the realised scene
        gap and the attempts needed, against v5b_180's package.
      * design.json AND THE README also record: the rule set and its rules
        in words (checked against set_info.json), the excluded scenes and
        the reason, the bin quotas, the leak limits and shipped values,
        script 3's run folder and the SHA-256 of its files (stimuli/ apart),
        the stimulus-set provenance and the Places365 note.
      * EXPERIMENT CHANGES: the experiment (dpdr_experiment_v5) already runs
        the two-block design with v5b_180, so the README lists only: the
        set name, the practice override in config.js, and any difference
        in row fields from v5b_180 (a check compares the field lists).
      * REGRESSION (REGRESSION_CHECK): the same code, run on v5b's 180
        build with v5b's settings, must give v5b_180's lists, practice,
        assignment table, preload, items.csv and design.json (apart from
        the names), up to the image step, compared in memory.
      * PREFLIGHT: every input is listed, found or missing, before any long
        computation.
      * STOP_BEFORE_IMAGES: a testing aid (default False).

WHAT CHANGED FROM make_trial_orders_v5.py (v5 -> v5b)
    The order construction, balance rules, order checks, assignment logic
    and package format are v5's. Changed:
      * INPUTS: script 3 is build_trials_v5b.py (the v5 stimulus set, the
        normality limits). The refusal checks are kept; the agreement of
        set_info.json with the set files now covers the counts per type and
        the practice counts too.
      * NO GRID: the assignment factors are block order x set mapping x
        order (2 x 2 x 4 = 16 cells, each 4 times in 64 slots); the grid set
        code is gone from design.json, assignment.json and the README. The
        README lists the experiment code that still refers to the grid.
      * LIST ROWS: v5's fields plus N_corpus, N_orig and includes_original.
        z_sat and z_fog keep their meaning (the pair's differences in SD).
        Every column of script 3's rows (image-level values with _a / _b
        suffixes, levels, normality parts, ...) is in items.csv, one row per
        item_id; checks confirm that each list holds its set's rows, apart
        from order, and that every list row agrees with items.csv.
      * PRACTICE: both practice files; design.json records that the
        practice before block 1 has feedback and the practice before block 2
        has none.
      * COUNTS AND BREAKS come from script 3's files (200 main trials per
        block at 180, 167 at 150; breaks every 40 / 34).
      * STIMULUS PROVENANCE: the stimulus set's folder and the SHA-256 of
        its checksums.sha256 are recorded in design.json and the README;
        every packaged image is checked against that file.
      * NORMALITY: design.json carries script 3's limits and shipped leak
        values, its run folder and the SHA-256 of its set and info files.
        The report recomputes the leak and the information from the packaged
        lists (they must equal script 3's), and gives each order's leak over
        its first segment (information only).
      * NAMES: packages v5b_{size}; output data/trial_orders_v5b/.
      * Relative paths only.

WHAT THIS DOES
    For each script 3 folder in SCRIPT3_DIRS (one per block size):

    1. REFUSES a folder that holds BUILD_FAILED.txt, or whose checks.csv has
       a FAIL, or whose set_info.json does not match its own set files.
    2. ORDERS. N_ORDERS orders of each set's block (analysed + consistency +
       catch trials), built by a seeded constructive search:
         * catch trials: one in each of SEGMENTS parts of the block, never
           in the first or last CATCH_EDGE trials; the parts end where the
           breaks fall (every BREAK_EVERY main trials), so each stretch
           between breaks holds one catch trial;
         * every other trial type is placed under a RUNNING BALANCE: after
           any number of trials, each type's count stays within one of its
           share (consistency's share starts after CONSISTENCY_MIN_DISTANCE),
           which spreads every type evenly across the block;
         * at least SCENE_GAP trials between two trials of the same scene;
         * each consistency repeat after its original and at least
           CONSISTENCY_MIN_DISTANCE trials from it;
         * no more than MAX_TYPE_RUN trials of one type in a row, and no
           more than MAX_SIDE_RUN trials in a row with the predicted image on
           the same side.
       N_ORDER_CANDIDATES valid orders are built per set; the N_ORDERS kept
       are the best balanced (mean position by type, by magnitude bin, and
       the correlation of position with the size of difference), subject to
       a limit on how alike two orders may be. Orders of set A and set B are
       built independently. The report states the LARGEST scene gap at which
       all orders could be built, and the gap used.
    3. ASSIGNMENT TABLE. For each group, N_SLOTS slots. Each slot gives the
       block order (which episode first), the set mapping (which set goes
       with which episode) and the order index (the same in both blocks).
       The three factors are fully crossed (2 x 2 x 4 = 16 cells; every
       cell 4 times in 64 slots), and the sequence is built
       to keep every prefix as balanced as possible: the report gives the
       balance at the first 10, 15, 20, 30 and all slots. Both groups use
       the same table, so they are balanced alike. Deterministic. Slots
       beyond N_SLOTS reuse the table (slot n -> row ((n - 1) mod N_SLOTS)
       + 1); a check confirms the balance after any number of slots then
       equals the balance after (that number mod N_SLOTS) slots.
    4. PACKAGE, in the format of the plan: design.json, assignment.json,
       lists/, practice/, items.csv, preload.json, stimuli/, manifest.csv,
       README.txt, and a CSV copy of every JSON.

REUSED
    Nothing is imported from v4 or script 1. Script 3's outputs are read as
    files; the leak is computed with script 1's leak() formula (copied).
    The list row format is v4's trial_list.json format (the experiment's
    REQUIRED_FIELDS), with extra fields for the two-block design.

INPUTS   SCRIPT3_DIRS (script 3's output folders, one per size; each with
         its stimuli/), V5B_PACKAGE_DIR (v5b_180's run: field list, balance
         comparison, regression), REGRESSION_SCRIPT3_DIR (v5b's 180 build)
OUTPUTS  one folder per size, OUTPUT_ROOT/<settings>__s<hash>/
    package/                 the experiment package (copy to the experiment
                             as data/sets/<PACKAGE_NAME>/)
    orders_report.txt        what was done, what was found, every check
    checks.csv               every check, PASS or FAIL
    orders_summary.csv       one row per order: balance and constraints
    order_candidates.csv     every valid candidate order's scores
    assignment_balance.csv   balance of the assignment at each prefix
    settings.json            every setting this run used
    balance_vs_v5b.csv       (v5c) each order's balance, v5c and v5b_180
    regression_v5b.csv       (v5c) the regression, file by file

    items.csv (package)      every item, all of script 3's columns

RUN      python make_trial_orders_v5c.py
"""

# ============================================================
# SETTINGS
# ============================================================

# SCRIPT3_DIRS
#   Script 3's output folder for each block size.
SCRIPT3_DIRS = {
    180: ("data/trials_v5c/v5c_n180_balanced_mixed_final_r3_cand40_leak0.05"
          "_seed42__s38fa29a2"),
}

# SCRIPT3_VERSION
#   The version script 3's set_info.json must state.
SCRIPT3_VERSION = "v5c"

# PACKAGE_NAME
#   The package folder's name; the experiment loads it as
#   data/sets/<name>/. {size} is replaced by the block size.
PACKAGE_NAME = "v5c_{size}"
VERSION = "v5c"

# --- Orders ---
N_ORDERS = 4                 # kept per set
N_ORDER_CANDIDATES = 40      # valid orders built per set before choosing
MAX_ATTEMPTS = 100_000        # restarts allowed per set to reach them

# SEGMENTS
#   The block is divided into this many parts for the type-spread check,
#   and holds one catch trial in each (so it must equal the number of catch
#   trials per set). The parts end exactly where the breaks fall (see
#   BREAK_EVERY), so each stretch between two breaks is one segment.
SEGMENTS = 5

# BREAK_EVERY
#   Main trials (analysed, consistency and catch; practice not counted)
#   between two breaks within a block.
#     "per_segment"  ceil(main trials per block / SEGMENTS), worked out per
#                    package (150 per block: 167 main trials -> 34;
#                    180: 200 -> 40). The default.
#     an integer     that number for every package.
#   Segment boundaries are BREAK_EVERY, 2 x BREAK_EVERY, ..., so the block
#   must split into exactly SEGMENTS parts; the script stops otherwise.
#   Recorded per package in design.json (timing) and the README.
BREAK_EVERY = "per_segment"

# TYPE_SPREAD_TOL
#   In every segment, each type's count must be within this of its expected
#   count (consistency is checked only from CONSISTENCY_MIN_DISTANCE on).
TYPE_SPREAD_TOL = 1

# SCENE_GAP / SCENE_GAP_SEARCH_MAX / SCENE_GAP_POLICY
#   Two trials of the same scene must be at least SCENE_GAP trials apart
#   (positions i < j with j - i >= SCENE_GAP; 30 means at least 29 other
#   trials between them). Catch trials show no images and do not count.
#   The same gap applies to both block sizes. The largest gap at which every
#   order of both sets can still be built is searched upward from SCENE_GAP
#   to SCENE_GAP_SEARCH_MAX and reported (v5's sets: 35 at 150 per block;
#   v5's earlier search from 5 found 34 at 150 and 36 at 180; 30 leaves a
#   margin).
#     "fixed"    build every order with SCENE_GAP (the default)
#     "largest"  build with the largest achievable gap
#   The realised smallest gap of every order is reported and recorded in
#   design.json.
SCENE_GAP = 30
SCENE_GAP_SEARCH_MAX = 40
SCENE_GAP_POLICY = "fixed"
GAP_SEARCH_ATTEMPTS = 20_000  # restarts per set per gap in the search

CONSISTENCY_MIN_DISTANCE = 40
CATCH_EDGE = 10              # no catch in the first or last 10 trials
MAX_TYPE_RUN = 4
MAX_SIDE_RUN = 4

# ORIGINAL_WEIGHT
#   When an analysed trial with a consistency repeat competes with others
#   of its type, it is this many times as likely to be placed next, so the
#   originals come early enough for their repeats to fit.
ORIGINAL_WEIGHT = 3.0

# SCENE_WEIGHT_POWER
#   Items are also weighted by (trials of their scene still to place) **
#   this, so scenes with many trials are spread out first and the end of the
#   block is not left with one scene's trials only. 0 switches it off.
SCENE_WEIGHT_POWER = 1.0

# Balance tolerances, as a share of the block (mean position 0.5 = centre)
POS_TOL_TYPE = 0.05          # mean position of each analysed type
POS_TOL_BIN = 0.10           # mean position of each type x magnitude bin
POS_CORR_TOL = 0.10          # |corr(position, |x_sat|)| and |x_con|
ORDER_CORR_MAX = 0.30        # largest |rank correlation| between two orders

# --- Assignment ---
GROUPS = {"participant": ["derealization", "ordinary"],
          "control": ["ordinary", "expectation"]}
N_SLOTS = 64
SLOT_PARAM = "slot"          # the link parameter that carries the slot
BALANCE_PREFIXES = [10, 15, 20, 30, 64]
ASSIGN_RESTARTS = 60
BALANCE_TOL = 1.0            # largest allowed deviation from an even split,
                             # per factor and for block order x set mapping

# SLOT_WRAP_CYCLES
#   Slots beyond N_SLOTS reuse the table: slot n uses row
#   ((n - 1) mod N_SLOTS) + 1 (slot 65 -> row 1, slot 130 -> row 2). Stated
#   in design.json, assignment.json and the README. The check that this
#   keeps the balance runs over this many passes through the table
#   (4 = slots 1 to 256).
SLOT_WRAP_CYCLES = 4

# --- Session settings recorded in design.json ---
REST_BETWEEN_BLOCKS = True

# PRACTICE_FEEDBACK
#   Whether the practice trials before block 1 and before block 2 give
#   feedback (by block position). v5c: "from_script3" = script 3's
#   set_info.json (practice_feedback_by_block_position); a pair (block 1,
#   block 2) overrides it. Recorded in design.json; the main trials never
#   give feedback.
PRACTICE_FEEDBACK = "from_script3"

# PRACTICE_FEEDBACK_TRIALS
#   How many practice trials give feedback, in a block whose practice has
#   feedback (script 2's PRACTICE_FEEDBACK_TRIALS: 3, i.e. all three).
#   set_info.json's practice_feedback_trials is used when it has one.
PRACTICE_FEEDBACK_TRIALS = 3
CATCH_KEYS = ("ArrowUp", "ArrowDown")
CATCH_MAX_RUN = 2            # same target at most twice in a row (v4 rule)

OUTPUT_ROOT = "data/trial_orders_v5c"
ON_EXISTING = "new"
RANDOM_SEED = 42

# --- v5c ---
# V5B_PACKAGE_DIR
#   v5b_180's script 4 run: its package's field lists (the list rows must
#   keep v5b_180's fields), its orders (the balance comparison, section 3b)
#   and the regression's reference.
V5B_PACKAGE_DIR = ("data/trial_orders_v5b/v5b_n180_orders4_gap30-fixed"
                   "_break40_slots64_seed42__s20227fe1")

# REGRESSION_CHECK
#   Run this script's code on v5b's 180 build (REGRESSION_SCRIPT3_DIR) with
#   v5b's settings (REGRESSION_SETTINGS) up to the image step, in memory, and
#   compare its lists and practice (JSON and CSV), assignment table,
#   preload, items.csv and design.json with V5B_PACKAGE_DIR's package, apart
#   from the names (REGRESSION_NAME_KEYS). A mismatch is a FAIL. It repeats
#   the gap search and the orders for v5b's sets (about 5 minutes).
REGRESSION_CHECK = True
REGRESSION_SCRIPT3_DIR = ("data/trials_v5b/v5b_n180_balanced_mixed_v5units"
                          "_cand40_leak0.05_seed42__s262763fa")
REGRESSION_SETTINGS = {"PRACTICE_FEEDBACK": (True, False),
                       "PACKAGE_NAME": "v5b_{size}", "VERSION": "v5b",
                       "SCRIPT3_VERSION": "v5b", "BLOCK_SIZE": 180}
REGRESSION_NAME_KEYS = ["version", "package", "source.script4",
                        "source.script4_sha256"]
# REGRESSION_RTOL: numbers in design.json are equal within this relative
# difference (the leak recomputed from the lists is a least-squares fit,
# whose last digit can differ between computers and library versions);
# the report says whether they are exact. Everything else must be exact.
REGRESSION_RTOL = 1e-9

# GAP_REPORT_BELOW
#   If SCENE_GAP cannot be reached, search downward (to 10) for the largest
#   gap that works, report it, and stop. The gap is never lowered.
GAP_REPORT_BELOW = True

# EXPERIMENT_REPO / V5B_PACKAGE_NAME
#   The experiment that runs the package, and the package it runs now.
EXPERIMENT_REPO = "zarcolmarioa/dpdr_experiment_v5"
V5B_PACKAGE_NAME = "v5b_180"

# RULES_IN_WORDS / RULES_LEVERS / RULES_MIN_SD
#   The rule set's rules, in words, for design.json and the README. Checked
#   against script 3's set_info.json: the rule set's name, its levers and
#   its minimum differences (rounded as in the words) must match.
RULES_IN_WORDS = {"final_r3": [
    "colour only: the two images 3 or 4 colour levels apart with the lower "
    "image at or below -1, or 2 levels apart with both at or below 0; at "
    "least 0.30 SD apart in measured saturation; the same contrast level, "
    "within +-2 of the original",
    "contrast only: 3 or 4 contrast levels apart, or 2 levels apart with "
    "the higher image at or below -1; at least 0.45 SD apart in measured "
    "contrast; the same colour level, within +-2 of the original",
    "together and conflict: exactly 2 levels apart on both features; at "
    "least 0.20 SD (saturation) and 0.32 SD (contrast) apart, at most 1.2 "
    "SD on each; at most one of the two features at +-3; conflict pairs "
    "balanced",
    "pair types come from the measured differences; magnitude bins are "
    "the levels apart (2 / 3 / 4) for colour only and contrast only, one "
    "bin for together and conflict, with quotas in proportion to the pool"]}
RULES_LEVERS = {"final_r3": {"COLOUR_ONLY_ALLOW_HIGH_3": False,
                             "COLOUR_ONLY_ALLOW_LOW_2": True,
                             "CONTRAST_ONLY_ALLOW_LOW_2": True}}
RULES_MIN_SD = {"final_r3": {"sat_only": 0.30, "fog_only": 0.45,
                             "mixed_sat": 0.20, "mixed_con": 0.32}}

# PLACES365_NOTE
#   The images' origin and licence (as v5b's frozen README).
PLACES365_NOTE = (
    "The images are edited versions of photographs from Places365, whose "
    "copyright belongs to the images' owners. This package can be archived "
    "and cited, but the images should not be redistributed publicly until "
    "their licence is confirmed. The stimulus set's sources.csv (the "
    "Places365 file of every scene) and its checksums allow anyone with "
    "Places365 to rebuild the images and verify them.")

# STOP_BEFORE_IMAGES
#   A testing aid: True builds and checks everything up to the image step,
#   copies no image, and fails the image check, so the folder cannot be
#   frozen. Keep False for a real build.
STOP_BEFORE_IMAGES = False


# ============================================================
# Imports
# ============================================================

import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

ANALYSED = ["sat_only", "fog_only", "together", "conflict"]
FILL_TYPES = ANALYSED + ["consistency"]
SD_SAT, SD_FOG = 39.182, 9.348

LIST_FIELDS = ["trial", "trial_type", "analysed", "scene_id", "category",
               "left_path", "right_path", "a_is_left", "pred_sat",
               "pred_fog", "d_sat", "d_fog", "z_sat", "z_fog", "mag_bin",
               "catch_mode", "repeat_of",
               # v5 additions
               "list_id", "set", "order", "item_id", "repeat_of_item",
               "pair_type", "couple", "file_a", "file_b",
               # v5b additions
               "N_corpus", "N_orig", "includes_original"]

# List fields that list_rows() makes or renames, so they are not compared
# with script 3's columns of the same name.
MADE_FIELDS = {"trial", "repeat_of", "list_id", "set", "order",
               "repeat_of_item", "pair_type", "couple"}
NORMALITY_DEFS = ("corpus", "orig")

# (v5c) The experiment changes are worked out per package: see
# experiment_changes().


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rng_for(*key):
    tag = int.from_bytes(hashlib.sha256(
        "|".join(str(k) for k in key).encode()).digest()[:8], "big")
    return np.random.default_rng([RANDOM_SEED, tag])


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, scope, name, ok, detail=""):
        status = ok if isinstance(ok, str) else ("PASS" if ok else "FAIL")
        self.rows.append({"scope": scope, "check": name, "status": status,
                          "detail": detail})
        return status == "PASS"

    def frame(self):
        return pd.DataFrame(self.rows)


def settings_dict(size):
    own = {k: v for k, v in globals().items()
           if k.isupper() and not k.startswith("_")
           and k not in ("ANALYSED", "FILL_TYPES", "SD_SAT", "SD_FOG",
                         "LIST_FIELDS", "MADE_FIELDS", "NORMALITY_DEFS",
                         "FACTORS", "README", "NOT_HASHED")}
    own["BLOCK_SIZE"] = size
    own["SCRIPT3_DIR"] = SCRIPT3_DIRS[size]
    return own


def break_every_for(N):
    """Main trials between breaks for a block of N main trials."""
    if BREAK_EVERY == "per_segment":
        return -(-N // SEGMENTS)
    return int(BREAK_EVERY)


def segment_edges(N):
    """Segment boundaries = break positions: 0, B, 2B, ..., N."""
    B = break_every_for(N)
    edges = list(range(0, N, B)) + [N]
    if len(edges) - 1 != SEGMENTS:
        raise ValueError(
            f"a break every {B} main trials splits a block of {N} into "
            f"{len(edges) - 1} parts, but SEGMENTS = {SEGMENTS}; the parts "
            "between breaks are the segments, so these must agree")
    return np.array(edges, dtype=int)


# not settings of the build: left out of the folder's hash
NOT_HASHED = ("REGRESSION_CHECK", "STOP_BEFORE_IMAGES")


def resolve_output_dir(size, B):
    s = {k: v for k, v in settings_dict(size).items() if k not in NOT_HASHED}
    blob = json.dumps(s, sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode()).hexdigest()[:8]
    name = (f"v5c_n{size}_orders{N_ORDERS}_gap{SCENE_GAP}-{SCENE_GAP_POLICY}"
            f"_break{B}_slots{N_SLOTS}_seed{RANDOM_SEED}__s{h}")
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
# Reading script 3
# ============================================================

def read_script3(d, size, ck, version=None):
    if not os.path.isdir(d):
        raise FileNotFoundError(f"script 3 folder not found: {d}")
    if os.path.exists(os.path.join(d, "BUILD_FAILED.txt")):
        raise RuntimeError(
            f"{d} contains BUILD_FAILED.txt: a check of script 3 failed "
            "there. This folder must not be used.")
    cf = pd.read_csv(os.path.join(d, "checks.csv"))
    fails = cf[cf.status == "FAIL"]
    if len(fails):
        raise RuntimeError(f"{d}: script 3 reports {len(fails)} failing "
                           "check(s):\n" + fails.to_string())
    ck.add("script 3", "folder has no BUILD_FAILED.txt and no failing check",
           True, f"{len(cf)} checks")
    info = json.load(open(os.path.join(d, "set_info.json")))
    sets = {b: pd.read_csv(os.path.join(d, f"set_{b}.csv"),
                           keep_default_na=False, na_values=[""])
            for b in ("A", "B")}
    prac = {i: pd.read_csv(os.path.join(d, f"practice_block{i}.csv"),
                           keep_default_na=False, na_values=[""])
            for i in (1, 2)}
    man = pd.read_csv(os.path.join(d, "manifest.csv"))
    ok = (info["block_size"] == size
          and info.get("version") == (version or SCRIPT3_VERSION))
    cnt = info["counts_per_set"]
    for b in ("A", "B"):
        s = sets[b]
        ok &= int((s.trial_type.isin(ANALYSED)).sum()) == size
        ok &= int(cnt["analysed"]) == size
        for t in ANALYSED:
            ok &= int((s.trial_type == t).sum()) == int(cnt[t])
        ok &= int((s.trial_type == "consistency").sum()) == cnt["consistency"]
        ok &= int((s.trial_type == "catch").sum()) == cnt["catch"]
    for i in (1, 2):
        ok &= len(prac[i]) == int(info["practice_by_block"][i - 1])
    ck.add("script 3", "set files agree with set_info.json (version, size, "
           "counts per type, consistency, catch, practice)", bool(ok),
           f"{size} analysed (" + ", ".join(f"{t} {cnt[t]}" for t in ANALYSED)
           + f"), {cnt['consistency']} consistency, {cnt['catch']} catch per "
           f"set; practice {info['practice_by_block']}")
    return sets, prac, info, man


# ============================================================
# One order
# ============================================================

def catch_positions(N, n_catch, rng):
    edges = segment_edges(N)
    pos = []
    for k in range(SEGMENTS):
        lo = max(edges[k], CATCH_EDGE)
        hi = min(edges[k + 1], N - CATCH_EDGE)
        if hi <= lo:
            raise RuntimeError("segment too short for a catch trial")
        pos.append(int(rng.integers(lo, hi)))
    return pos[:n_catch]


def build_order(items, G, rng):
    """One order of a set's block, or (None, reason). items: DataFrame with
    item_id, trial_type, scene_id, a_is_left, repeat_of (item_id or NaN)."""
    N = len(items)
    tt = items.trial_type.to_numpy()
    tb = list(zip(tt, items.mag_bin.to_numpy()))
    scn = items.scene_id.to_numpy()
    side = items.a_is_left.astype(bool).to_numpy()
    rep_of = items.repeat_of.to_numpy()
    idx_of = {iid: k for k, iid in enumerate(items.item_id)}
    is_orig = np.zeros(N, dtype=bool)
    for r in rep_of:
        if isinstance(r, str) and r:
            is_orig[idx_of[r]] = True
    catches = [k for k in range(N) if tt[k] == "catch"]
    if len(catches) != SEGMENTS:
        raise RuntimeError(f"{len(catches)} catch trials but SEGMENTS = "
                           f"{SEGMENTS}: one catch per segment needs them "
                           "equal")
    cpos = catch_positions(N, len(catches), rng)
    order = [None] * N
    for k, p in zip(rng.permutation(catches), cpos):
        order[p] = int(k)
    M = N - len(catches)
    n_t = {t: int((tt == t).sum()) for t in FILL_TYPES}
    n_tb = {}
    for k in range(N):
        n_tb[tb[k]] = n_tb.get(tb[k], 0) + 1
    placed_tb = {key: 0 for key in n_tb}
    remaining = {t: [k for k in range(N) if tt[k] == t] for t in FILL_TYPES}
    placed = {t: 0 for t in FILL_TYPES}
    pos_of = {}
    last_scene_pos = {}
    scene_left = {}
    for k in range(N):
        if tt[k] != "catch":
            scene_left[scn[k]] = scene_left.get(scn[k], 0) + 1
    D = CONSISTENCY_MIN_DISTANCE
    i = 0                                   # filled non-catch trials
    for p in range(N):
        if order[p] is not None:
            pos_of[order[p]] = p
            continue

        n_an = sum(n_t[t] for t in ANALYSED)
        a_done = sum(placed[t] for t in ANALYSED)

        def expected(t):
            # analysed types: against the number of analysed trials placed;
            # consistency: its own schedule, from trial D on
            if t == "consistency":
                return n_t[t] * max(0, i + 1 - D) / max(M - D, 1)
            return n_t[t] * (a_done + 1) / n_an

        an = [t for t in ANALYSED if remaining[t]
              and placed[t] <= expected(t) + 1e-9]
        deficit = {t: expected(t) - placed[t] for t in an}
        rng.shuffle(an)
        an.sort(key=lambda t: -deficit[t])
        cons_due = (remaining["consistency"]
                    and placed["consistency"] < expected("consistency"))
        types = (["consistency"] if cons_due else []) + an
        if remaining["consistency"] and not cons_due:
            types.append("consistency")
        chosen = None
        # recent image trials, for run limits
        before = [order[q] for q in range(p - 1, -1, -1)
                  if order[q] is not None]
        recent = before[:MAX_TYPE_RUN]
        # side runs count image trials only (catch trials show no images)
        recent_img = [k for k in before if tt[k] != "catch"][:MAX_SIDE_RUN]
        for t in types:
            if (len(recent) >= MAX_TYPE_RUN
                    and all(tt[k] == t for k in recent[:MAX_TYPE_RUN])):
                continue
            cand = []
            for k in remaining[t]:
                if p - last_scene_pos.get(scn[k], -10 ** 9) < G:
                    continue
                if (len(recent_img) >= MAX_SIDE_RUN
                        and all(side[q] == side[k]
                                for q in recent_img[:MAX_SIDE_RUN])):
                    continue
                if t == "consistency":
                    o = idx_of[rep_of[k]]
                    if o not in pos_of or p - pos_of[o] < D:
                        continue
                cand.append(k)
            if not cand:
                continue
            # within the type, the magnitude bin furthest behind its share
            # (ties at random), so every bin is spread across the block too
            bins = sorted({tb[k] for k in cand})
            def_b = {key: n_tb[key] * (placed[t] + 1) / n_t[t]
                     - placed_tb[key] for key in bins}
            top = max(def_b.values())
            pick_bins = [key for key in bins if def_b[key] >= top - 1e-9]
            key = pick_bins[int(rng.integers(len(pick_bins)))]
            cand = [k for k in cand if tb[k] == key]
            # most-constrained first: items from scenes with more trials
            # still to place are more likely to be taken now, so the end of
            # the block is not left with one scene's trials only
            w = np.array([(ORIGINAL_WEIGHT if is_orig[k] else 1.0)
                          * scene_left[scn[k]] ** SCENE_WEIGHT_POWER
                          for k in cand])
            chosen = cand[int(rng.choice(len(cand), p=w / w.sum()))]
            break
        if chosen is None:
            return None, f"dead end at trial {p + 1} of {N}"
        order[p] = chosen
        pos_of[chosen] = p
        remaining[tt[chosen]].remove(chosen)
        placed[tt[chosen]] += 1
        placed_tb[tb[chosen]] += 1
        last_scene_pos[scn[chosen]] = p
        scene_left[scn[chosen]] -= 1
        i += 1
    return order, None


# ============================================================
# Measuring an order
# ============================================================

def order_frame(items, order):
    df = items.iloc[order].reset_index(drop=True).copy()
    df["position"] = np.arange(len(df))
    return df


def runs(values):
    best, cur = 1, 1
    for a, b in zip(values[:-1], values[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return best if len(values) else 0


def measure_order(df):
    """Every constraint and balance figure of one ordered block."""
    N = len(df)
    rel = df.position / (N - 1)
    out = {}
    # type spread per segment (segments end at the breaks)
    edges = segment_edges(N)
    seg = np.searchsorted(edges, df.position, side="right") - 1
    # Expected counts per segment follow the design: one catch per
    # segment; consistency spread evenly from CONSISTENCY_MIN_DISTANCE on;
    # the analysed types share the remaining places in proportion to their
    # counts (so they fill more of the first segment, where no repeat can
    # yet go).
    worst = 0.0
    n_cons = int((df.trial_type == "consistency").sum())
    exp_cons, exp_slots = [], []
    for k in range(SEGMENTS):
        lo = max(edges[k], CONSISTENCY_MIN_DISTANCE)
        span = max(0, edges[k + 1] - lo)
        ec = n_cons * span / max(N - CONSISTENCY_MIN_DISTANCE, 1)
        exp_cons.append(ec)
        exp_slots.append(edges[k + 1] - edges[k] - 1 - ec)
    tot_slots = sum(exp_slots)
    for t in FILL_TYPES + ["catch"]:
        m = (df.trial_type == t).to_numpy()
        n_tt = m.sum()
        for k in range(SEGMENTS):
            got = int((m & (seg == k)).sum())
            if t == "consistency":
                exp = exp_cons[k]
            elif t == "catch":
                exp = 1.0
            else:
                exp = n_tt * exp_slots[k] / tot_slots
            worst = max(worst, abs(got - exp))
    out["type_spread_worst"] = worst
    # scene gap among image trials
    img = df[df.trial_type != "catch"]
    gaps = []
    for _, g in img.groupby("scene_id"):
        p = np.sort(g.position.to_numpy())
        if len(p) > 1:
            gaps.append(np.diff(p).min())
    out["scene_gap_min"] = int(min(gaps)) if gaps else N
    # consistency
    pos = dict(zip(df.item_id, df.position))
    dist = [pos[r.item_id] - pos[r.repeat_of]
            for r in df[df.trial_type == "consistency"].itertuples()]
    out["consistency_min_distance"] = int(min(dist)) if dist else N
    # catch
    cp = df.position[df.trial_type == "catch"].to_numpy()
    out["catch_first"] = int(cp.min()) + 1
    out["catch_last_from_end"] = int(N - cp.max())
    out["catch_per_segment"] = "/".join(
        str(int(((seg == k) & (df.trial_type == "catch")).sum()))
        for k in range(SEGMENTS))
    # runs
    out["max_type_run"] = runs(list(df.trial_type))
    out["max_side_run"] = runs(list(img.a_is_left.astype(bool)))
    # balance
    for t in ANALYSED + ["consistency", "catch"]:
        out[f"mean_pos_{t}"] = float(rel[df.trial_type == t].mean())
    worst_t = max(abs(out[f"mean_pos_{t}"] - 0.5) for t in ANALYSED)
    out["mean_pos_type_worst"] = worst_t
    wb = 0.0
    for (t, b), g in df[df.trial_type.isin(ANALYSED)].groupby(
            ["trial_type", "mag_bin"]):
        v = float(rel[g.index].mean())
        out[f"mean_pos_{t}_bin{b}"] = v
        wb = max(wb, abs(v - 0.5))
    out["mean_pos_bin_worst"] = wb
    a = df[df.trial_type.isin(ANALYSED)]
    out["corr_pos_abs_x_sat"] = float(np.corrcoef(
        a.position, a.pred_sat.abs())[0, 1])
    out["corr_pos_abs_x_con"] = float(np.corrcoef(
        a.position, a.pred_fog.abs())[0, 1])
    out["score"] = (worst_t + 0.5 * wb + 0.5 * max(
        abs(out["corr_pos_abs_x_sat"]), abs(out["corr_pos_abs_x_con"])))
    return out


def order_checks(m, G):
    """Each constraint and tolerance as (name, passed, detail)."""
    return [
        (f"type spread: every type within {TYPE_SPREAD_TOL} of its share in "
         f"each of {SEGMENTS} segments",
         m["type_spread_worst"] <= TYPE_SPREAD_TOL + 1e-9,
         f"worst {m['type_spread_worst']:.2f}"),
        (f"scene gap >= {G}", m["scene_gap_min"] >= G,
         f"smallest {m['scene_gap_min']}"),
        (f"consistency repeats after their originals, >= "
         f"{CONSISTENCY_MIN_DISTANCE} apart",
         m["consistency_min_distance"] >= CONSISTENCY_MIN_DISTANCE,
         f"smallest {m['consistency_min_distance']}"),
        (f"one catch per segment (= per stretch between breaks), none in "
         f"the first or last {CATCH_EDGE}",
         m["catch_per_segment"] == "/".join(["1"] * SEGMENTS)
         and m["catch_first"] > CATCH_EDGE
         and m["catch_last_from_end"] > CATCH_EDGE,
         f"per segment {m['catch_per_segment']}, first at "
         f"{m['catch_first']}, last {m['catch_last_from_end']} from the "
         "end"),
        (f"same type at most {MAX_TYPE_RUN} in a row",
         m["max_type_run"] <= MAX_TYPE_RUN, f"longest {m['max_type_run']}"),
        (f"predicted image on the same side at most {MAX_SIDE_RUN} in a row",
         m["max_side_run"] <= MAX_SIDE_RUN, f"longest {m['max_side_run']}"),
        (f"mean position of each analysed type within {POS_TOL_TYPE} of the "
         "centre", m["mean_pos_type_worst"] <= POS_TOL_TYPE,
         f"worst {m['mean_pos_type_worst']:.3f}"),
        (f"mean position of each type x magnitude bin within {POS_TOL_BIN} "
         "of the centre", m["mean_pos_bin_worst"] <= POS_TOL_BIN,
         f"worst {m['mean_pos_bin_worst']:.3f}"),
        (f"|corr(position, size of difference)| <= {POS_CORR_TOL}",
         max(abs(m["corr_pos_abs_x_sat"]),
             abs(m["corr_pos_abs_x_con"])) <= POS_CORR_TOL,
         f"x_sat {m['corr_pos_abs_x_sat']:+.3f}, x_con "
         f"{m['corr_pos_abs_x_con']:+.3f}"),
    ]


def valid(m, G):
    return all(ok for _, ok, _ in order_checks(m, G))


def rank_corr(o1, o2, items):
    """Spearman correlation of the positions the same items take in two
    orders."""
    p1 = np.empty(len(items))
    p2 = np.empty(len(items))
    p1[o1] = np.arange(len(items))
    p2[o2] = np.arange(len(items))
    return float(np.corrcoef(p1, p2)[0, 1])


def candidate_orders(items, G, n_want, max_attempts, rng):
    """Valid orders (all constraints and tolerances met) until n_want or
    the attempt budget. Returns (list of (order, measures), stats)."""
    got, reasons, attempts = [], {}, 0
    while len(got) < n_want and attempts < max_attempts:
        attempts += 1
        o, why = build_order(items, G, rng)
        if o is None:
            reasons[why.split(" at ")[0]] = reasons.get(
                why.split(" at ")[0], 0) + 1
            continue
        m = measure_order(order_frame(items, o))
        if not valid(m, G):
            bad = [n for n, ok, _ in order_checks(m, G) if not ok]
            key = "built, but fails: " + bad[0]
            reasons[key] = reasons.get(key, 0) + 1
            continue
        got.append((o, m))
    return got, {"attempts": attempts, "valid": len(got), "reasons": reasons}


def pick_orders(cands, items):
    """The N_ORDERS best-balanced candidates, each at most ORDER_CORR_MAX
    rank-correlated with those already kept."""
    ranked = sorted(range(len(cands)), key=lambda k: cands[k][1]["score"])
    kept = []
    for k in ranked:
        if all(abs(rank_corr(cands[k][0], cands[j][0], items))
               <= ORDER_CORR_MAX for j in kept):
            kept.append(k)
        if len(kept) == N_ORDERS:
            break
    return kept


# ============================================================
# Assignment
# ============================================================

FACTORS = ["block_order", "set_mapping", "order"]
_ASSIGN_ROWS = None


def cells():
    return [(bo, sm, k) for bo in (0, 1) for sm in (0, 1)
            for k in range(1, N_ORDERS + 1)]


def levels(f):
    return list(range(1, N_ORDERS + 1)) if f == "order" else [0, 1]


def prefix_dev(rows):
    """Largest deviation from an even split, per factor and for each pair
    of factors, over a list of cells."""
    P = len(rows)
    out = {}
    arr = np.array(rows)
    for i, f in enumerate(FACTORS):
        lv = levels(f)
        out[f] = max(abs((arr[:, i] == v).sum() - P / len(lv)) for v in lv)
    for i in range(len(FACTORS)):
        for j in range(i + 1, len(FACTORS)):
            li, lj = levels(FACTORS[i]), levels(FACTORS[j])
            out[f"{FACTORS[i]} x {FACTORS[j]}"] = max(
                abs(((arr[:, i] == a) & (arr[:, j] == b)).sum()
                    - P / (len(li) * len(lj))) for a in li for b in lj)
    return out


def assignment_table():
    """Deterministic: ASSIGN_RESTARTS greedy constructions, each going
    through the 16 cells in cycles (every cell once per cycle) and taking,
    at each slot, the remaining cell that keeps the prefix most balanced
    (ties at random); the construction with the best balance summed over
    every prefix is kept."""
    all_cells = cells()
    best, best_score = None, np.inf
    for r in range(ASSIGN_RESTARTS):
        rng = rng_for("assignment", r)
        rows, score = [], 0.0
        pool = []
        for s in range(N_SLOTS):
            if not pool:
                pool = list(all_cells)
            sc = []
            for c in pool:
                d = prefix_dev(rows + [c])
                sc.append(sum(d[f] for f in FACTORS)
                          + 0.5 * sum(v for k, v in d.items()
                                      if " x " in k))
            sc = np.array(sc)
            cand = np.flatnonzero(sc <= sc.min() + 1e-9)
            c = pool.pop(int(cand[int(rng.integers(len(cand)))]))
            rows.append(c)
            score += sc.min() * (3.0 if (s + 1) in BALANCE_PREFIXES else 1.0)
        if score < best_score:
            best, best_score = rows, score
    return best


def slot_rule_text():
    return (f"slot n uses row ((n - 1) mod {N_SLOTS}) + 1 of the table: "
            f"slots 1-{N_SLOTS} use rows 1-{N_SLOTS}, slot {N_SLOTS + 1} "
            f"uses row 1, slot {2 * N_SLOTS + 2} uses row 2, and so on")


def assignment_json(rows, sets_present):
    out = {"slots_per_group": N_SLOTS, "slot_param": SLOT_PARAM,
           "slot_rule": slot_rule_text(),
           "factors": {
               "block_order": "0 = the group's first-listed episode first, "
                              "1 = second-listed first",
               "set_mapping": "0 = set A with the first-listed episode, "
                              "1 = set B with it",
               "order": "order index, the same in both blocks"},
           "groups": {}}
    for grp, eps in GROUPS.items():
        lst = []
        for s, (bo, sm, k) in enumerate(rows, start=1):
            set_of = {eps[0]: "A" if sm == 0 else "B",
                      eps[1]: "B" if sm == 0 else "A"}
            seq = eps if bo == 0 else eps[::-1]
            blocks = []
            for pos, ep in enumerate(seq, start=1):
                st = set_of[ep]
                blocks.append({"position": pos, "episode": ep, "set": st,
                               "list": f"{st}_o{k}",
                               "practice": f"practice/block{pos}.json"})
            lst.append({"slot": s, "block_order": bo, "set_mapping": sm,
                        "order": k, "blocks": blocks})
        out["groups"][grp] = lst
    return out


# ============================================================
# v5b: stimulus set, items, normality
# ============================================================

def leak(N, xs, xc, terms=("sat", "con")):
    """The normality leak of a set of oriented pairs: N regressed on the
    direction terms together, no intercept (N ~ b_sat x_sat + b_con x_con),
    and the uncentred correlations of N with each term. `terms` = ("sat",)
    or ("con",) fits one term only, for a type in which the other is ~0
    (sat_only, fog_only), where a joint fit is meaningless.
    (VERBATIM from analyse_pair_capacity_v5b.py, script 1.)"""
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


def stimulus_provenance(info, ck):
    """The stimulus set script 3 used (set_info.json: stimulus_set.dir): its
    checksums.sha256 must be the frozen one (FROZEN.json) and the one script
    3 recorded. Returns the provenance for design.json and the checksums.
    (v5b's stimulus_set_check, split in two so the package can be made up
    to the image step without the images.)"""
    st = info.get("stimulus_set", {})
    d = str(st.get("dir", "")).rstrip("/")
    ck_file, fr_file = f"{d}/checksums.sha256", f"{d}/FROZEN.json"
    out = {"dir": d, "run": os.path.basename(d),
           "checksums_file": ck_file, "checksums_sha256": None,
           "manifest_sha256": st.get("manifest_sha256"),
           "stimuli_sha256": st.get("stimuli_sha256")}
    if not d or not os.path.isfile(ck_file):
        ck.add("stimulus set", "the stimulus set's checksums.sha256 is "
               "present", False, f"{ck_file} not found")
        return out, {}
    h = sha256_of(ck_file)
    out["checksums_sha256"] = h
    fr = json.load(open(fr_file)) if os.path.isfile(fr_file) else {}
    out["settings_hash"] = fr.get("settings_hash")
    ck.add("stimulus set", "checksums.sha256 is the frozen one (FROZEN.json) "
           "and the one script 3 recorded (set_info.json)",
           h == fr.get("checksums_sha256") and h == st.get("checksums_sha256"),
           f"{d}: SHA-256 {h[:16]}...")
    sums = {}
    with open(ck_file) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln:
                hh, rel = ln.split("  ", 1)
                sums[rel] = hh
    return out, sums


def stimulus_image_check(sums, pkg_sha, ck):
    bad = [f for f, hh in pkg_sha.items() if sums.get(f"stimuli/{f}") != hh]
    ck.add("stimulus set", "every packaged image matches the stimulus set's "
           "checksums.sha256", not bad,
           f"{len(pkg_sha)} images, {len(bad)} differ or are not listed"
           + (f": {bad[:5]}" if bad else ""))


def items_table(sets, prac):
    """One row per item_id: sets A and B (analysed, consistency, catch) and
    the practice trials, with every column of script 3's rows."""
    parts = [sets[b].assign(source=f"set_{b}") for b in ("A", "B")]
    parts += [prac[i].assign(source=f"practice_block{i}") for i in (1, 2)]
    it = pd.concat(parts, ignore_index=True, sort=False)
    cols = ["item_id", "source"] + [c for c in it.columns
                                    if c not in ("item_id", "source")]
    it = it[cols]
    return it, it.source.value_counts().to_dict()


def _same(a, b):
    na = a is None or (isinstance(a, float) and np.isnan(a))
    nb = b is None or (isinstance(b, float) and np.isnan(b))
    if na or nb:
        return na and nb
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        def truth(v):
            return str(v).strip().lower() in ("true", "1", "1.0")
        return truth(a) == truth(b)
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    return abs(fa - fb) <= 1e-9 * max(1.0, abs(fa))


def mismatches(df, src):
    """item_id.field for every field of df's rows that differs from src
    (indexed by item_id); fields that list_rows() makes are skipped."""
    cols = [c for c in df.columns if c in src.columns
            and c not in MADE_FIELDS and c != "item_id"]
    out = []
    for r in df.to_dict("records"):
        if r["item_id"] not in src.index:
            out.append(f"{r['item_id']} missing")
            continue
        s = src.loc[r["item_id"]]
        out += [f"{r['item_id']}.{c}" for c in cols if not _same(r[c], s[c])]
    return out


def compare_lists(lists, prac_rows, sets, prac, items, ck):
    ck.add("items", "items.csv has one row per item_id",
           bool(items.item_id.is_unique), f"{len(items)} items")
    for lid, df in sorted(lists.items()):
        b = lid[0]
        src = sets[b].set_index("item_id")
        same = sorted(df.item_id) == sorted(src.index)
        bad = mismatches(df, src)
        ck.add("lists", f"{lid}: exactly set {b}'s rows of script 3, apart "
               "from order (every item once, every carried field equal)",
               same and not bad, f"{len(df)} rows" + (
                   f"; differ: {bad[:5]}" if bad else ""))
    for i, df in sorted(prac_rows.items()):
        src = prac[i].set_index("item_id")
        same = list(df.item_id) == list(src.index)
        bad = mismatches(df, src)
        ck.add("lists", f"practice block {i}: script 3's practice rows, in "
               "its order, every carried field equal", same and not bad,
               f"{len(df)} rows" + (f"; differ: {bad[:5]}" if bad else ""))
    allrows = pd.concat(list(lists.values()) + list(prac_rows.values()),
                        ignore_index=True)
    idx = items.set_index("item_id")
    missing = sorted(set(allrows.item_id) - set(idx.index))
    bad = mismatches(allrows, idx)
    shared = [c for c in allrows.columns if c in idx.columns
              and c not in MADE_FIELDS and c != "item_id"]
    ck.add("items", "every list and practice row's item_id is in items.csv, "
           "and the fields they share agree", not missing and not bad,
           f"{len(allrows)} rows, {len(shared)} shared fields"
           + (f"; missing {missing[:5]}" if missing else "")
           + (f"; differ {bad[:5]}" if bad else ""))


def limits_text(info):
    nl = info["normality_limit"]
    t = (f"overall |b_sat|, |b_con| <= {nl['limit']:g} (both terms, no "
         "intercept, both definitions, each set)")
    for tp, v in nl.get("type_limits", {}).items():
        t += f"; {tp} N on x_{v['term']} alone |b| <= {v['limit']:g}"
    return t


def normality_from_lists(lists, info, d3, ck):
    """The leak (overall, and per limited type) and the information of each
    list's analysed rows. They must be the same in every order of a set
    (the orders only rearrange it) and equal to script 3's (set_info.json;
    the information from candidates.csv's row of the chosen candidate)."""
    nl = info["normality_limit"]
    tl = nl.get("type_limits", {})
    rows = []
    for lid, df in sorted(lists.items()):
        a = df[df.trial_type.isin(ANALYSED)]
        r = {"list_id": lid, "set": lid[0], "n": len(a),
             "info_sat": float((a.pred_sat ** 2).sum()),
             "info_con": float((a.pred_fog ** 2).sum())}
        for dfn in NORMALITY_DEFS:
            lk = leak(a[f"N_{dfn}"], a.pred_sat, a.pred_fog)
            for k, v in lk.items():
                r[f"{dfn}_{k}"] = v
            for tp, v in tl.items():
                m = a[a.trial_type == tp]
                lt = leak(m[f"N_{dfn}"], m.pred_sat, m.pred_fog,
                          (v["term"],))
                r[f"{tp}_{dfn}_b_{v['term']}"] = lt[f"b_{v['term']}"]
        rows.append(r)
    T = pd.DataFrame(rows)
    num = [c for c in T.columns if c not in ("list_id", "set", "n")]
    spread = max(float((g[num].max() - g[num].min()).abs().max())
                 for _, g in T.groupby("set"))
    ck.add("normality", "every order of a set gives the same leak and "
           "information (the orders only rearrange the set)",
           spread <= 1e-9, f"largest difference between orders {spread:.1e}")
    cand = pd.read_csv(os.path.join(d3, "candidates.csv"))
    crow = cand[cand.candidate == info["chosen_candidate"]].iloc[0]
    diffs = []
    for r in T.itertuples():
        b = r.set
        for dfn in NORMALITY_DEFS:
            sh = nl["shipped"][f"{dfn}_{b}"]
            for k in ("b_sat", "b_con", "cos_sat", "cos_con"):
                diffs.append(abs(getattr(r, f"{dfn}_{k}") - sh[k]))
            for tp, v in tl.items():
                diffs.append(abs(getattr(r, f"{tp}_{dfn}_b_{v['term']}")
                                 - v["shipped"][f"{dfn}_{b}"]))
        for w in ("sat", "con"):
            diffs.append(abs(getattr(r, f"info_{w}")
                             - crow[f"rep_info_{w}_{b}"]))
    worst = max(diffs)
    ck.add("normality", "the leak (overall and per limited type, both "
           "definitions) and the information recomputed from the packaged "
           "lists equal script 3's", worst <= 1e-9,
           f"{len(diffs)} values compared, largest difference {worst:.1e}")
    first = T.groupby("set").first()
    summary = {b: {k: (float(v) if isinstance(v, (float, np.floating))
                       else int(v)) for k, v in first.loc[b].items()
                   if k != "list_id"} for b in first.index}
    return {"table": T, "summary": summary, "worst": worst,
            "spread": spread}


def first_segment_leak(lists, edges):
    """Each order's leak over its first segment (trials 1 to the first
    break), analysed trials only: both terms, no intercept, both
    definitions, with the uncentred correlations. Information only."""
    L = int(edges[1])
    out = []
    for lid, df in sorted(lists.items()):
        seg = df[(df.trial <= L) & df.trial_type.isin(ANALYSED)]
        r = {"list_id": lid, "segment_trials": L, "n_analysed": len(seg)}
        for dfn in NORMALITY_DEFS:
            lk = leak(seg[f"N_{dfn}"], seg.pred_sat, seg.pred_fog)
            for k, v in lk.items():
                r[f"{dfn}_{k}"] = v
        out.append(r)
    return out


def normality_lines(w, norm, first_seg, edges, info):
    nl = info["normality_limit"]
    tl = nl.get("type_limits", {})
    w("6. NORMALITY AND INFORMATION (THE ORDERS DO NOT CHANGE THEM)")
    w("-" * 78)
    w("The orders only rearrange each set's trials, so the leak and the "
      "information are script 3's.")
    w(f"Script 3's limits: {limits_text(info)}.")
    w("Recomputed from the packaged lists (analysed rows):")
    T = norm["table"]
    w(f"  {'list':6s} {'n':>4s} {'info sat':>9s} {'info con':>9s}  "
      f"{'corpus b_sat':>12s} {'b_con':>7s}  {'orig b_sat':>10s} "
      f"{'b_con':>7s}  " + "  ".join(f"{tp} x_{v['term']}"
                                     for tp, v in tl.items()))
    for r in T.itertuples():
        cells = []
        for tp, v in tl.items():
            term = v["term"]
            cells.append("/".join(
                f"{getattr(r, f'{tp}_{d}_b_{term}'):+.3f}"
                for d in NORMALITY_DEFS))
        w(f"  {r.list_id:6s} {r.n:4d} {r.info_sat:9.2f} {r.info_con:9.2f}  "
          f"{r.corpus_b_sat:+12.4f} {r.corpus_b_con:+7.4f}  "
          f"{r.orig_b_sat:+10.4f} {r.orig_b_con:+7.4f}  " + "  ".join(cells))
    w("  (per-type columns: corpus / orig)")
    w(f"  largest difference between the orders of a set "
      f"{norm['spread']:.1e}; largest difference from script 3 "
      f"{norm['worst']:.1e}")
    L = int(edges[1])
    w("")
    w(f"First segment: trials 1-{L} of each order (the stretch before the "
      "first break), analysed trials only;")
    w("both terms, no intercept; cos = uncentred correlation. INFORMATION "
      "ONLY: few trials, so b moves a lot")
    w("(cos is steadier); nothing is limited here.")
    w(f"  {'list':6s} {'n':>3s}   {'corpus b_sat':>12s} {'b_con':>7s} "
      f"{'cos_sat':>7s} {'cos_con':>7s}   {'orig b_sat':>10s} {'b_con':>7s} "
      f"{'cos_sat':>7s} {'cos_con':>7s}")
    for r in first_seg:
        w(f"  {r['list_id']:6s} {r['n_analysed']:3d}   "
          f"{r['corpus_b_sat']:+12.3f} {r['corpus_b_con']:+7.3f} "
          f"{r['corpus_cos_sat']:+7.3f} {r['corpus_cos_con']:+7.3f}   "
          f"{r['orig_b_sat']:+10.3f} {r['orig_b_con']:+7.3f} "
          f"{r['orig_cos_sat']:+7.3f} {r['orig_cos_con']:+7.3f}")
    w("  (first_segment_leak.csv holds these figures)")
    w("")


# ============================================================
# Catch targets
# ============================================================

def catch_targets(n, block_pos, rng):
    """n targets, as balanced as n allows (block 1 gets the extra Up, block
    2 the extra Down), at most CATCH_MAX_RUN identical in a row."""
    up = (n + 1) // 2 if block_pos == 1 else n // 2
    for _ in range(10_000):
        seq = [CATCH_KEYS[0]] * up + [CATCH_KEYS[1]] * (n - up)
        seq = list(rng.permutation(seq))
        if runs(seq) <= CATCH_MAX_RUN:
            return [str(x) for x in seq]
    raise RuntimeError("no catch sequence meets CATCH_MAX_RUN")


# ============================================================
# Package
# ============================================================

def list_rows(df, list_id, st, k):
    """Rows of one list in the experiment's format, trial numbers 1..N."""
    df = df.reset_index(drop=True).copy()
    df["trial"] = np.arange(1, len(df) + 1)
    num = dict(zip(df.item_id, df.trial))
    df["repeat_of_item"] = df.repeat_of.where(df.repeat_of.notna(), None)
    df["repeat_of"] = [num[r] if isinstance(r, str) and r else None
                       for r in df.repeat_of_item]
    df["list_id"] = list_id
    df["set"] = st
    df["order"] = k
    df["pair_type"] = df["type"]
    df["catch_mode"] = df.catch_mode.where(df.catch_mode.notna(), None)
    df["couple"] = df.couple.where(df.couple.notna(), None) \
        if "couple" in df.columns else None
    return df[LIST_FIELDS]


def to_json_records(df):
    recs = []
    for r in df.to_dict("records"):
        o = {}
        for k, v in r.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                o[k] = None
            elif isinstance(v, (np.integer,)):
                o[k] = int(v)
            elif isinstance(v, (np.floating,)):
                o[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                o[k] = bool(v)
            else:
                o[k] = v
        if o.get("couple") is not None:
            o["couple"] = int(o["couple"])
        if o.get("repeat_of") is not None:
            o["repeat_of"] = int(o["repeat_of"])
        recs.append(o)
    return recs


def write_json_csv(obj_or_df, path_json, csv_df=None):
    with open(path_json, "w") as f:
        json.dump(obj_or_df, f, indent=1)
    if csv_df is not None:
        csv_df.to_csv(path_json[:-5] + ".csv", index=False)


def flatten(d, prefix=""):
    rows = []
    if isinstance(d, dict):
        for k, v in d.items():
            rows += flatten(v, f"{prefix}{k}.")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            rows += flatten(v, f"{prefix}{i}.")
    else:
        rows.append((prefix[:-1], d))
    return rows


README = """V5C EXPERIMENT PACKAGE  ({name})
====================================================================

Two image-choice blocks per participant, {size} analysed trials each.
Generated by make_trial_orders_v5c.py from build_trials_v5c.py's sets
(script 3 run: {s3run}).

FILES
  design.json       the design: counts, groups and episodes, lists, catch
                    targets per block position, break and rest settings,
                    per-set thresholds, order constraints, practice and its
                    feedback, the rule set, the excluded scenes, the bin
                    quotas, the normality limits, provenance
  assignment.json   per group, {slots} slots. The invitation link carries
                    ?{param}=<n>; slots are handed out in sequence per
                    group and never reused. Each slot names, for block
                    positions 1 and 2, the episode, the set and the list.
                    See SLOTS for slots beyond {slots}. There is no grid
                    block and no grid set code.
  lists/<set>_o<k>.json
                    one block's main trials (analysed, consistency, catch)
                    in presentation order. Row format = {v5b_name}'s (v4's
                    trial_list.json, plus list_id, set, order, item_id,
                    repeat_of_item, pair_type, couple, file_a, file_b,
                    N_corpus, N_orig, includes_original).
                    repeat_of = the block-local trial number of the
                    original (consistency rows only). z_sat and z_fog are
                    the PAIR's differences in SD units (as v4).
  practice/block1.json, practice/block2.json
                    the practice trials before block 1 ({p1} trials, with
                    feedback: {fb1}) and block 2 ({p2} trials, with
                    feedback: {fb2}), the same for every participant;
                    feedback on {fbn} practice trials of a block with
                    feedback (design.json: practice_feedback_by_block_
                    position, practice_feedback_trials)
  items.csv         one row per item_id (sets A and B: analysed,
                    consistency and catch; practice), with every column of
                    script 3's rows: image levels and edit parameters,
                    image-level values with _a / _b suffixes (z_sat_a,
                    z_con_a, change_sat_a, dist_corpus_a, ...), the
                    normality terms and their parts (N_corpus_sat, ...),
                    straddle flags. Join responses to it by item_id.
  preload.json      every image any list or practice row shows. Catch rows
                    show no images ('no_images' mode), so images used only
                    by catch rows ship in stimuli/ but are not listed.
  stimuli/          the images, byte-identical to the stimulus set
                    {stimset} (each checked against its checksums.sha256,
                    SHA-256 {cksha})
  manifest.csv      every shipped image, its SHA-256 and measured features
  *.csv             a CSV copy of every JSON

RULE SET ({rule_name})
{rules}

EXCLUDED SCENES
  {excluded}
  Reason: {excl_reason}
  {n_scenes} scenes are used.

BIN QUOTAS (per set)
{quotas}

PREDICTORS
  pred_sat = -d_sat / 39.182, pred_fog = d_fog / 9.348, always A minus B;
  positive = image A is the predicted choice. a_is_left says which side A
  is on.

NORMALITY
  N_corpus, N_orig = dist(B) - dist(A); positive = image A is the more
  typical (corpus = distance from the corpus median, orig = from the
  scene's unedited photograph). Script 3 limited the leak of the analysed
  trials of each set: {limits}.
  Shipped: {shipped}
  design.json: normality.

CATCH
  {catch} catch rows per block, no images; targets per block position in
  design.json. Exclude a participant with {excl} or more failures over the
  session; record failures per block too.

BREAKS
  A break every {brk} main trials within each block (main trials = the
  {main} rows of a list: analysed, consistency and catch; practice is not
  counted), i.e. after trials {breaks}. {brk} = ceil({main} / {segs}) when
  set "per_segment" (this package: {brk_rule}). The stretches between breaks
  are the {segs} segments the orders are built on: {parts} trials, one
  catch trial in each. A rest between the two blocks: {rest}.
  design.json: timing.break_every, timing.breaks_after_trial.

ORDERS
  Two trials of the same scene at least {gap} trials apart (the largest gap
  at which every order could be built: {largest}); {n_orders} orders per set.

SLOTS
  {slot_rule}.
  The balance after any number of slots then equals the balance after
  (that number mod {slots}) slots: every {slots} slots each of the 16 cells
  (block order x set mapping x order) has been used equally often again
  (checked by the build for {wrap} slots).

SOURCE
  script 3 run {s3dir}; the SHA-256 of each of its files (stimuli/ apart)
  is in design.json: source.script3_files_sha256 ({n3files} files).
  Stimulus set {stimrun} (folder {stimset}); manifest SHA-256 {mansha}.

IMAGES AND ARCHIVING
{places}

EXPERIMENT CHANGES NEEDED (not made here; {exp_repo})
  The experiment already runs the two-block design with {v5b_name}. For
  this package:
{changes}
"""


def _wrap(text, indent="  ", width=74):
    import textwrap
    return "\n".join(textwrap.wrap(text, width, initial_indent=indent,
                                   subsequent_indent=indent + "  "))


# ============================================================
# v5c: practice, rule set, source, experiment changes, fields
# ============================================================

def practice_design(info, prac, feedback_setting, ck):
    """The practice counts and feedback: from set_info.json (or the
    setting), checked against the practice files."""
    counts = [int(x) for x in info["practice_by_block"]]
    fb_si = info.get("practice_feedback_by_block_position")
    if feedback_setting == "from_script3":
        if fb_si is None:
            raise RuntimeError("PRACTICE_FEEDBACK = 'from_script3', but "
                               "script 3's set_info.json has no "
                               "practice_feedback_by_block_position")
        fb = {str(k): bool(v) for k, v in fb_si.items()}
        source = "script 3's set_info.json"
    else:
        fb = {"1": bool(feedback_setting[0]), "2": bool(feedback_setting[1])}
        source = "the setting PRACTICE_FEEDBACK"
    ntr = int(info.get("practice_feedback_trials", PRACTICE_FEEDBACK_TRIALS))
    ntr_src = ("set_info.json" if "practice_feedback_trials" in info
               else "the setting PRACTICE_FEEDBACK_TRIALS (set_info.json "
               "does not record it)")
    files = [len(prac[1]), len(prac[2])]
    return {"counts": counts, "feedback": fb, "feedback_source": source,
            "feedback_trials": ntr, "feedback_trials_source": ntr_src,
            "files": files, "set_info_feedback": fb_si}


def practice_agreement(design, pr, ck):
    """design.json, set_info.json and the practice files agree."""
    fb_si = pr["set_info_feedback"]
    fb_si = ({str(k): bool(v) for k, v in fb_si.items()} if fb_si is not
             None else None)
    ok = (design["practice_by_block"] == pr["counts"] == pr["files"]
          and design["practice_feedback_by_block_position"] == pr["feedback"]
          and (fb_si is None or fb_si == pr["feedback"])
          and all(pr["feedback_trials"] <= n for n in pr["counts"]))
    ck.add("practice", "design.json, script 3's set_info.json and the practice"
           " files agree (trials per block, feedback per block)", ok,
           f"design.json {design['practice_by_block']} feedback "
           f"{design['practice_feedback_by_block_position']}; set_info.json "
           f"{pr['counts']} feedback {fb_si}; files {pr['files']}; feedback "
           f"on {pr['feedback_trials']} trials ({pr['feedback_trials_source']})")


def rule_set_design(info, ck):
    """The rule set for design.json, checked against set_info.json."""
    rs = info.get("rule_set")
    if not rs:
        return None
    name = rs.get("name")
    words = RULES_IN_WORDS.get(name)
    lev = rs.get("final_rules_levers", {})
    th = info.get("threshold_set_values_sd", {})
    mins = RULES_MIN_SD.get(name, {})
    ok_l = lev == RULES_LEVERS.get(name)
    ok_m = bool(mins) and all(abs(round(float(th.get(k, np.nan)), 2) - v)
                              < 1e-9 for k, v in mins.items())
    ck.add("rule set", f"the rules in words ({name}) match script 3's "
           "set_info.json: the rule set, its levers and its minimum "
           "differences", words is not None and ok_l and ok_m,
           f"levers {lev}; minimums (SD) " + ", ".join(
               f"{k} {float(th.get(k, np.nan)):.3f}" for k in mins))
    return {"name": name, "rules": words,
            "final_rules": rs.get("final_rules"), "exclude": rs.get("exclude"),
            "bins": rs.get("bins"), "quotas": rs.get("quotas"),
            "levers": lev, "minimum_differences_sd": th}


def script3_file_hashes(d3):
    """SHA-256 of every file of script 3's run folder, stimuli/ apart (the
    images are checked against the manifest and the stimulus set)."""
    out = {}
    for root, dirs, fs in os.walk(d3):
        dirs[:] = sorted(x for x in dirs if not (
            root == d3 and x == "stimuli"))
        for f in sorted(fs):
            p = os.path.join(root, f)
            out[os.path.relpath(p, d3).replace(os.sep, "/")] = sha256_of(p)
    return dict(sorted(out.items()))


def v5b_fields(ck):
    """v5b_180's list and practice field lists (JSON keys and CSV
    headers)."""
    pkg = os.path.join(V5B_PACKAGE_DIR, "package")
    out = {}
    for sub in ("lists", "practice"):
        for f in sorted(os.listdir(os.path.join(pkg, sub))):
            p = os.path.join(pkg, sub, f)
            if f.endswith(".json"):
                recs = json.load(open(p))
                out[f"{sub}/{f}"] = list(recs[0].keys()) if recs else []
            elif f.endswith(".csv"):
                out[f"{sub}/{f}"] = open(p).readline().rstrip("\n").split(",")
    return out


def field_differences(lists, prac_rows, ref):
    """This package's fields against v5b_180's, per kind of file."""
    mine = {"lists json": list(to_json_records(next(iter(
        lists.values())).head(1))[0].keys()),
            "lists csv": list(next(iter(lists.values())).columns),
            "practice json": list(to_json_records(prac_rows[1].head(1))[0]
                                  .keys()),
            "practice csv": list(prac_rows[1].columns)}
    theirs = {}
    for k, v in ref.items():
        kind = ("lists" if k.startswith("lists/") else "practice") + (
            " json" if k.endswith(".json") else " csv")
        theirs.setdefault(kind, []).append(v)
    rows = []
    for kind, m in mine.items():
        vs = theirs.get(kind, [])
        same_all = all(v == vs[0] for v in vs) if vs else False
        t = vs[0] if vs else []
        rows.append({"kind": kind, "same": bool(vs) and same_all and m == t,
                     "added": [c for c in m if c not in t],
                     "removed": [c for c in t if c not in m],
                     "order_differs": [c for c in m if c in t] != [
                         c for c in t if c in m],
                     "n_files_v5b": len(vs)})
    return rows


def experiment_changes(name, pr, fdiff):
    fb = pr["feedback"]
    out = [f"config.js: set CONFIG.data.set and CONFIG.data.available_sets "
           f"to '{name}' (now '{V5B_PACKAGE_NAME}').",
           "config.js: remove the practice override "
           "(practice_trials_by_block_position, practice_feedback_by_block_"
           f"position): design.json now states {pr['counts'][0]} + "
           f"{pr['counts'][1]} practice trials, feedback before block 1 "
           f"{'yes' if fb['1'] else 'no'} and block 2 "
           f"{'yes' if fb['2'] else 'no'}, on {pr['feedback_trials']} trials "
           "of a block with feedback."]
    diff = [r for r in fdiff if not r["same"]]
    if not diff:
        out.append(f"row fields: none differ from {V5B_PACKAGE_NAME}'s (lists"
                   " and practice, JSON and CSV), so the experiment code "
                   "works unchanged.")
    else:
        for r in diff:
            out.append(f"row fields ({r['kind']}): added {r['added']}, "
                       f"removed {r['removed']}"
                       + (", order differs" if r["order_differs"] else ""))
    return out


# ============================================================
# Building a package (in memory), then writing it
# ============================================================

def gap_search(sets, size, ck, label=""):
    """v5b's search upward from SCENE_GAP; v5c: if SCENE_GAP fails, the
    largest gap below it that works is searched for (information only) and
    the run stops."""
    gap_rows = []
    largest = None
    t = time.time()
    for G in range(SCENE_GAP, SCENE_GAP_SEARCH_MAX + 1):
        ok_all = True
        for b in ("A", "B"):
            got, stats = candidate_orders(sets[b], G, N_ORDERS,
                                          GAP_SEARCH_ATTEMPTS,
                                          rng_for("gap", size, b, G))
            gap_rows.append({"gap": G, "set": b, **{k: v for k, v in
                                                   stats.items()
                                                   if k != "reasons"},
                             "reasons": json.dumps(stats["reasons"])})
            ok_all &= stats["valid"] >= N_ORDERS
        print(f"    {label}gap {G}: "
              + ("all orders built" if ok_all else "not every order built")
              + f" ({time.time() - t:.0f} s)", flush=True)
        if ok_all:
            largest = G
        else:
            break
    gap_tab = pd.DataFrame(gap_rows)
    ck.add("scene gap", f"all {N_ORDERS} orders of both sets can be built "
           f"with a scene gap of at least {SCENE_GAP}", largest is not None,
           f"largest achievable {largest} (searched {SCENE_GAP}-"
           f"{SCENE_GAP_SEARCH_MAX}, {GAP_SEARCH_ATTEMPTS} attempts per "
           "set per gap)")
    if largest is None:
        below = None
        if GAP_REPORT_BELOW:
            for G in range(SCENE_GAP - 1, 9, -1):
                ok_all = all(candidate_orders(
                    sets[b], G, N_ORDERS, GAP_SEARCH_ATTEMPTS,
                    rng_for("gap", size, b, G))[1]["valid"] >= N_ORDERS
                    for b in ("A", "B"))
                if ok_all:
                    below = G
                    break
        why = gap_tab[gap_tab.gap == SCENE_GAP].to_string()
        raise RuntimeError(
            f"n={size}: the orders cannot be built with a scene gap of "
            f"{SCENE_GAP}. STOPPED: the gap is not lowered. "
            + (f"For information: the largest gap that works is {below}."
               if below else "No gap from 10 up works either.")
            + "\n" + why)
    return gap_tab, largest, time.time() - t


def construct(size, d3, ck, practice_feedback, pkg_name, version,
              s3_version, label=""):
    """Everything up to the image step, in memory: the orders, the
    assignment, the list rows, items, design.json. (v5b's build_size, with
    the writing moved to write_package.)"""
    T = {}
    sets, prac, info, man = read_script3(d3, size, ck, s3_version)
    main_n = len(sets["A"])
    B = break_every_for(main_n)
    edges = segment_edges(main_n)        # stops here if they disagree
    parts = np.diff(edges).tolist()
    breaks = edges[1:-1].tolist()
    ck.add("breaks", f"a break every {B} main trials "
           + (f"(= ceil({main_n} / {SEGMENTS}))"
              if BREAK_EVERY == "per_segment" else "(fixed)")
           + f" gives exactly {SEGMENTS} stretches, which are the segments",
           len(parts) == SEGMENTS and len(sets["B"]) == main_n,
           f"breaks after trials {breaks}; stretches {parts}")
    ck.add("breaks", f"no short final stretch: every stretch has at least "
           f"{B} - {SEGMENTS - 1} = {B - SEGMENTS + 1} trials",
           min(parts) >= B - SEGMENTS + 1, f"shortest {min(parts)}")

    # --- the largest achievable scene gap ---
    gap_tab, largest, T["gap_search"] = gap_search(sets, size, ck, label)
    G_use = SCENE_GAP if SCENE_GAP_POLICY == "fixed" else largest
    print(f"    {label}scene gap: largest achievable {largest}, used {G_use}",
          flush=True)

    # --- orders ---
    t = time.time()
    orders, summary, cand_rows, ostats = {}, [], [], {}
    for b in ("A", "B"):
        cands, stats = candidate_orders(sets[b], G_use, N_ORDER_CANDIDATES,
                                        MAX_ATTEMPTS,
                                        rng_for("orders", size, b))
        ostats[b] = stats
        ck.add(f"orders {b}", f"{N_ORDER_CANDIDATES} valid candidate orders "
               "built", stats["valid"] >= N_ORDER_CANDIDATES,
               f"{stats['valid']} in {stats['attempts']} attempts; failures "
               f"{stats['reasons']}")
        for j, (o, m) in enumerate(cands):
            cand_rows.append({"set": b, "candidate": j, **m})
        kept = pick_orders(cands, sets[b])
        ck.add(f"orders {b}", f"{N_ORDERS} orders kept, pairwise |rank "
               f"correlation| <= {ORDER_CORR_MAX}", len(kept) == N_ORDERS,
               f"candidates {kept}")
        for k, j in enumerate(kept, start=1):
            o, m = cands[j]
            orders[(b, k)] = order_frame(sets[b], o)
            row = {"list_id": f"{b}_o{k}", "set": b, "order": k,
                   "candidate": j, **m}
            for name, ok, det in order_checks(m, G_use):
                ck.add(f"{b}_o{k}", name, bool(ok), det)
            summary.append(row)
        corr = [abs(rank_corr(cands[i][0], cands[j][0], sets[b]))
                for x, i in enumerate(kept) for j in kept[x + 1:]]
        summary_corr = max(corr) if corr else 0.0
        ck.add(f"orders {b}", "every order is a permutation of the set",
               all(sorted(orders[(b, k)].item_id) == sorted(sets[b].item_id)
                   for k in range(1, N_ORDERS + 1)))
        ck.add(f"orders {b}", "the kept orders differ (largest pairwise "
               f"|rank correlation| <= {ORDER_CORR_MAX})",
               summary_corr <= ORDER_CORR_MAX, f"{summary_corr:.3f}")
        print(f"    {label}orders {b}: {stats['valid']} valid in "
              f"{stats['attempts']} attempts ({time.time() - t:.0f} s)",
              flush=True)
    summ = pd.DataFrame(summary)
    T["orders"] = time.time() - t

    # --- assignment (the same table for every size; built once) ---
    global _ASSIGN_ROWS
    if _ASSIGN_ROWS is None:
        _ASSIGN_ROWS = assignment_table()
    rows = _ASSIGN_ROWS
    bal = []
    for P in sorted(set(BALANCE_PREFIXES + [N_SLOTS])):
        if P > N_SLOTS:
            continue
        dv = prefix_dev(rows[:P])
        bal.append({"slots": P, **dv})
    bal = pd.DataFrame(bal)
    cell_counts = pd.Series([tuple(r) for r in rows]).value_counts()
    ck.add("assignment", f"every one of the {len(cells())} cells "
           f"(block order x set mapping x order) appears "
           f"{N_SLOTS // len(cells())} times in {N_SLOTS} slots",
           len(cell_counts) == len(cells())
           and (cell_counts == N_SLOTS // len(cells())).all(),
           f"counts {sorted(set(cell_counts))}")
    for P in BALANCE_PREFIXES:
        r = bal[bal.slots == P].iloc[0]
        worst = max(r[f] for f in FACTORS)
        ck.add("assignment", f"first {P} slots: every factor within "
               f"{BALANCE_TOL:g} of an even split, and block order x set "
               f"mapping too", worst <= BALANCE_TOL + 1e-9
               and r["block_order x set_mapping"] <= BALANCE_TOL + 1e-9,
               f"worst factor {worst:g}, block order x set mapping "
               f"{r['block_order x set_mapping']:g}")
    W = SLOT_WRAP_CYCLES * N_SLOTS
    wrapped = [rows[(n - 1) % N_SLOTS] for n in range(1, W + 1)]
    dev_all = [prefix_dev(wrapped[:P]) for P in range(1, W + 1)]
    keys = list(dev_all[0])
    at_full = [P for P in range(N_SLOTS, W + 1, N_SLOTS)]
    zero_ok = all(abs(dev_all[P - 1][k]) < 1e-9 for P in at_full
                  for k in keys)
    same_ok = all(abs(dev_all[P - 1][k] - dev_all[P % N_SLOTS - 1][k]) < 1e-9
                  for P in range(N_SLOTS + 1, W + 1) if P % N_SLOTS
                  for k in keys)
    worst_first = {k: max(d[k] for d in dev_all[:N_SLOTS]) for k in keys}
    worst_all = {k: max(d[k] for d in dev_all) for k in keys}
    ck.add("assignment", f"slots beyond {N_SLOTS} (row ((n - 1) mod "
           f"{N_SLOTS}) + 1): after every {N_SLOTS} slots, up to {W}, every "
           "factor and every pair of factors is exactly balanced", zero_ok,
           f"checked at {at_full}")
    ck.add("assignment", f"slots beyond {N_SLOTS}: the balance after n slots "
           f"equals the balance after (n mod {N_SLOTS}) slots for every n up "
           f"to {W}, so it is never worse than within the first {N_SLOTS}",
           same_ok and all(abs(worst_all[k] - worst_first[k]) < 1e-9
                           for k in keys),
           "worst over every prefix up to "
           f"{W}: " + ", ".join(f"{f} {worst_all[f]:g}" for f in FACTORS)
           + f", block order x set mapping "
           f"{worst_all['block_order x set_mapping']:g}")
    for P in BALANCE_PREFIXES:
        for c in range(1, SLOT_WRAP_CYCLES):
            Q = c * N_SLOTS + P
            if Q <= W and Q not in set(bal.slots):
                bal = pd.concat([bal, pd.DataFrame(
                    [{"slots": Q, **dev_all[Q - 1]}])], ignore_index=True)
    bal = bal.sort_values("slots").reset_index(drop=True)
    asg = assignment_json(rows, sets)

    # --- list rows (in memory) ---
    lists = {}
    for (b, k), df in sorted(orders.items()):
        lid = f"{b}_o{k}"
        lists[lid] = list_rows(df, lid, b, k)
    prac_rows = {}
    for i in (1, 2):
        p = prac[i].copy()
        p["couple"] = None
        if "repeat_of" not in p:
            p["repeat_of"] = None
        prac_rows[i] = list_rows(p, f"practice_block{i}", "practice", 0)
    flat = []
    for grp, lst in asg["groups"].items():
        for r in lst:
            flat.append({"group": grp, "slot": r["slot"],
                         "block_order": r["block_order"],
                         "set_mapping": r["set_mapping"],
                         "order": r["order"],
                         **{f"block{b['position']}_{k}": b[k]
                            for b in r["blocks"]
                            for k in ("episode", "set", "list")}})
    shown = set()
    for df in list(lists.values()) + list(prac_rows.values()):
        img = df[df.trial_type != "catch"]
        shown |= set(img.left_path) | set(img.right_path)
    preload = sorted(shown)
    files = set()
    for df in list(lists.values()) + list(prac_rows.values()):
        files |= set(df.file_a) | set(df.file_b)
    items, item_src = items_table(sets, prac)
    compare_lists(lists, prac_rows, sets, prac, items, ck)
    norm = normality_from_lists(lists, info, d3, ck)
    first_seg = first_segment_leak(lists, edges)
    ck.add("package", "every preloaded image ships in stimuli/",
           all(p[len("stimuli/"):] in files for p in preload),
           f"{len(preload)} preloaded, {len(files)} shipped")
    stim, sums = stimulus_provenance(info, ck)
    pr = practice_design(info, prac, practice_feedback, ck)

    rngc = rng_for("catch_targets", size)
    catch_seq = {str(pos): catch_targets(info["counts_per_set"]["catch"],
                                         pos, rngc) for pos in (1, 2)}
    gap_of = dict(zip(summ.list_id, summ.scene_gap_min))
    name = pkg_name.format(size=size)
    design = {
        "version": version, "package": name, "block_size": size,
        "main_trials_per_block": main_n,
        "counts_per_block": info["counts_per_set"],
        "practice_by_block": info["practice_by_block"],
        "groups": {g: {"episodes": e} for g, e in GROUPS.items()},
        "lists": {lid: {"set": lid[0], "order": int(lid.split("_o")[1]),
                        "file": f"lists/{lid}.json", "n": len(df),
                        "scene_gap_min": int(gap_of[lid])}
                  for lid, df in lists.items()},
        "practice": {"1": "practice/block1.json",
                     "2": "practice/block2.json"},
        "practice_feedback_by_block_position": pr["feedback"],
        "items": {"file": "items.csv", "key": "item_id",
                  "rows": len(items),
                  "content": "every column of script 3's rows, one row per "
                             "item (sets A and B, practice)"},
        "assignment": {"file": "assignment.json", "slot_param": SLOT_PARAM,
                       "slots_per_group": N_SLOTS,
                       "allocation": "in sequence per group, never reused",
                       "slot_rule": slot_rule_text()},
        "timing": {"break_every": B,
                   "break_rule": (f"ceil(main trials per block / segments) "
                                  f"= ceil({main_n} / {SEGMENTS})"
                                  if BREAK_EVERY == "per_segment"
                                  else "fixed"),
                   "break_counts": "main trials of the block (analysed, "
                                   "consistency, catch); practice not "
                                   "counted",
                   "breaks_after_trial": breaks,
                   "stretch_lengths": parts,
                   "rest_between_blocks": REST_BETWEEN_BLOCKS},
        "catch": {**info["catch"],
                  "targets_by_block_position": catch_seq},
        "thresholds_by_set": info["thresholds"],
        "choice_probability_at_planning_weights":
            info["choice_probability_at_planning_weights"],
        "predictors": info["predictors"],
        "normality": {"definition": info["normality"],
                      **info["normality_limit"],
                      "recomputed_from_lists": norm["summary"]},
        "stimulus_set": stim,
        "order_constraints": {
            "segments": SEGMENTS, "segment_edges": edges.tolist(),
            "segments_end_at_breaks": True,
            "type_spread_tol": TYPE_SPREAD_TOL,
            "scene_gap_used": G_use, "scene_gap_policy": SCENE_GAP_POLICY,
            "scene_gap_largest_achievable": largest,
            "consistency_min_distance": CONSISTENCY_MIN_DISTANCE,
            "catch_edge": CATCH_EDGE, "max_type_run": MAX_TYPE_RUN,
            "max_side_run": MAX_SIDE_RUN},
        "source": {"script3_dir": d3,
                   "script3_run": os.path.basename(d3.rstrip("/")),
                   "script3_set_A_sha256": sha256_of(
                       os.path.join(d3, "set_A.csv")),
                   "script3_set_B_sha256": sha256_of(
                       os.path.join(d3, "set_B.csv")),
                   "script3_set_info_sha256": sha256_of(
                       os.path.join(d3, "set_info.json")),
                   "script3_sha256": info.get("script3", {}).get("sha256"),
                   "script1_sha256": info.get("script1", {}).get("sha256"),
                   "script4": "make_trial_orders_v5c.py",
                   "script4_sha256": sha256_of(__file__),
                   "chosen_candidate": info.get("chosen_candidate")},
    }
    # --- v5c additions ---
    design["practice_feedback_trials"] = pr["feedback_trials"]
    design["rule_set"] = rule_set_design(info, ck)
    design["excluded_scenes"] = {
        "scenes": info.get("excluded_scenes", []),
        "images": info.get("excluded_images", []),
        "reason": info.get("exclude_reason", ""),
        "scenes_used": int(info.get("scenes_usable", 0)) or None}
    design["bin_quotas"] = {"quotas": info.get("bin_quotas"),
                            "levels_apart": info.get("bin_levels_apart")}
    design["images_note"] = PLACES365_NOTE
    design["source"]["script3_files_sha256"] = script3_file_hashes(d3)
    design = json.loads(json.dumps(design))       # as it will be written
    return {"size": size, "d3": d3, "sets": sets, "prac": prac,
            "info": info, "man": man, "main_n": main_n, "B": B,
            "edges": edges, "parts": parts, "breaks": breaks,
            "gap_tab": gap_tab, "largest": largest, "G_use": G_use,
            "summ": summ, "cand_rows": cand_rows, "ostats": ostats,
            "bal": bal, "rows": rows, "asg": asg, "flat": pd.DataFrame(flat),
            "lists": lists, "prac_rows": prac_rows, "preload": preload,
            "files": files, "items": items, "norm": norm,
            "first_seg": first_seg, "stim": stim, "sums": sums, "pr": pr,
            "design": design, "name": name, "times": T}


def shipped_text(info):
    nl = info["normality_limit"]
    parts = []
    for k, v in nl.get("shipped", {}).items():
        parts.append(f"{k} b_sat {v['b_sat']:+.3f} b_con {v['b_con']:+.3f}")
    for tp, v in nl.get("type_limits", {}).items():
        parts.append(f"{tp} " + ", ".join(
            f"{k} {x:+.3f}" for k, x in v.get("shipped", {}).items()))
    return "; ".join(parts)


def write_package(R, pkg, ck, changes):
    """The package files (v5b's format), the images, the README."""
    os.makedirs(os.path.join(pkg, "lists"))
    os.makedirs(os.path.join(pkg, "practice"))
    for lid, lr in R["lists"].items():
        write_json_csv(to_json_records(lr),
                       os.path.join(pkg, "lists", f"{lid}.json"), lr)
    for i, pr_ in R["prac_rows"].items():
        write_json_csv(to_json_records(pr_),
                       os.path.join(pkg, "practice", f"block{i}.json"), pr_)
    write_json_csv(R["asg"], os.path.join(pkg, "assignment.json"), R["flat"])
    write_json_csv(R["preload"], os.path.join(pkg, "preload.json"),
                   pd.DataFrame({"path": R["preload"]}))
    files, man, d3 = R["files"], R["man"], R["d3"]
    t = time.time()
    if STOP_BEFORE_IMAGES:
        ck.add("package", "the image step ran (images copied and checked)",
               False, "STOP_BEFORE_IMAGES = True: a test run, not a package "
               "to use")
    else:
        os.makedirs(os.path.join(pkg, "stimuli"))
        src = os.path.join(d3, "stimuli")
        for f in sorted(files):
            s = os.path.join(src, f)
            if not os.path.isfile(s):
                raise RuntimeError(f"{f} is not in {src}")
            shutil.copy2(s, os.path.join(pkg, "stimuli", f))
        man_pkg = man[man.filename.isin(files)].sort_values("filename")
        man_pkg.to_csv(os.path.join(pkg, "manifest.csv"), index=False)
        pkg_sha = {f: sha256_of(os.path.join(pkg, "stimuli", f))
                   for f in sorted(files)}
        bad_hash = [f for f in sorted(files)
                    if pkg_sha[f] != man.set_index("filename").loc[f,
                                                                   "sha256"]]
        ck.add("package", "every shipped image matches script 3's manifest "
               "(SHA-256)", not bad_hash, f"{len(files)} images, "
               f"{len(bad_hash)} differ")
        stimulus_image_check(R["sums"], pkg_sha, ck)
    R["times"]["images"] = time.time() - t
    R["items"].to_csv(os.path.join(pkg, "items.csv"), index=False)
    design = R["design"]
    with open(os.path.join(pkg, "design.json"), "w") as f:
        json.dump(design, f, indent=1)
    pd.DataFrame(flatten(design), columns=["key", "value"]).to_csv(
        os.path.join(pkg, "design.csv"), index=False)
    info, pr, stim = R["info"], R["pr"], R["stim"]
    rs = design.get("rule_set") or {}
    q = (design.get("bin_quotas") or {}).get("quotas") or {}
    lv = (design.get("bin_quotas") or {}).get("levels_apart") or {}
    ex = design.get("excluded_scenes") or {}
    with open(os.path.join(pkg, "README.txt"), "w") as f:
        f.write(README.format(
            name=R["name"], size=R["size"], slots=N_SLOTS, param=SLOT_PARAM,
            catch=info["counts_per_set"]["catch"],
            excl=info["catch"]["exclude_at_or_above"],
            brk=R["B"], main=R["main_n"], segs=SEGMENTS,
            breaks=", ".join(map(str, R["breaks"])),
            brk_rule=("per_segment" if BREAK_EVERY == "per_segment"
                      else f"fixed at {R['B']}"),
            parts=" / ".join(map(str, R["parts"])),
            rest="yes" if REST_BETWEEN_BLOCKS else "no",
            slot_rule=slot_rule_text()[0].upper() + slot_rule_text()[1:],
            wrap=f"{SLOT_WRAP_CYCLES * N_SLOTS}",
            s3run=os.path.basename(d3.rstrip("/")), s3dir=d3,
            n3files=len(design["source"]["script3_files_sha256"]),
            p1=pr["counts"][0], p2=pr["counts"][1],
            fb1="yes" if pr["feedback"]["1"] else "no",
            fb2="yes" if pr["feedback"]["2"] else "no",
            fbn=pr["feedback_trials"],
            stimset=stim["dir"], stimrun=stim["run"],
            cksha=stim["checksums_sha256"], mansha=stim["manifest_sha256"],
            limits=limits_text(info), shipped=shipped_text(info),
            rule_name=rs.get("name", "-"),
            rules="\n".join(_wrap("* " + r) for r in (rs.get("rules")
                                                      or ["-"])),
            excluded=", ".join(ex.get("scenes") or []) or "none",
            excl_reason=ex.get("reason") or "-",
            n_scenes=ex.get("scenes_used") or "-",
            quotas="\n".join(
                f"  {t:9s} " + " / ".join(
                    f"{n} ({lv.get(t, {}).get(b, '?')} levels apart)"
                    if t in ("sat_only", "fog_only") else str(n)
                    for b, n in q[t].items()) for t in q) or "  -",
            gap=R["G_use"], largest=R["largest"], n_orders=N_ORDERS,
            places=_wrap(PLACES365_NOTE),
            exp_repo=EXPERIMENT_REPO, v5b_name=V5B_PACKAGE_NAME,
            changes="\n".join(_wrap("* " + c) for c in changes)))


# ============================================================
# v5c: the regression against v5b_180
# ============================================================

def _jnorm(o):
    return json.loads(json.dumps(o))


def _drop_key(d, dotted):
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d.get(p, {}) if isinstance(d, dict) else {}
    if isinstance(d, dict):
        d.pop(parts[-1], None)


def _diff_keys(a, b, prefix="", rtol=0.0, worst=None):
    """Dotted keys at which two JSON objects differ; numbers (not booleans)
    are equal within rtol relative. worst: a list collecting the largest
    absolute numeric difference."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(prefix + str(k))
            else:
                out += _diff_keys(a[k], b[k], f"{prefix}{k}.", rtol, worst)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff_keys(x, y, f"{prefix}{i}.", rtol, worst)
        return out
    num = (isinstance(a, (int, float)) and isinstance(b, (int, float))
           and not isinstance(a, bool) and not isinstance(b, bool))
    if num and a != b:
        d = abs(a - b)
        if worst is not None:
            worst.append(d)
        return [] if d <= rtol * max(1.0, abs(b)) else [prefix[:-1]]
    return [] if a == b else [prefix[:-1] or "(root)"]


def run_regression():
    """v5b's 180 build with v5b's settings, up to the image step, in memory;
    compared with V5B_PACKAGE_DIR's package."""
    t0 = time.time()
    st = REGRESSION_SETTINGS
    size = int(st["BLOCK_SIZE"])
    ck = Checks()
    print(f"\n  REGRESSION: {REGRESSION_SCRIPT3_DIR} with v5b's settings",
          flush=True)
    R = construct(size, REGRESSION_SCRIPT3_DIR, ck, st["PRACTICE_FEEDBACK"],
                  st["PACKAGE_NAME"], st["VERSION"], st["SCRIPT3_VERSION"],
                  label="regression ")
    pkg = os.path.join(V5B_PACKAGE_DIR, "package")
    rows = []

    def add(what, eq, note=""):
        rows.append({"file": what, "equal": bool(eq), "note": note})

    for lid, lr in R["lists"].items():
        js = _jnorm(to_json_records(lr))
        add(f"lists/{lid}.json", js == json.load(open(
            os.path.join(pkg, "lists", f"{lid}.json"))))
        add(f"lists/{lid}.csv", lr.to_csv(index=False) == open(
            os.path.join(pkg, "lists", f"{lid}.csv")).read())
    for i, pr_ in R["prac_rows"].items():
        add(f"practice/block{i}.json", _jnorm(to_json_records(pr_))
            == json.load(open(os.path.join(pkg, "practice",
                                           f"block{i}.json"))))
        add(f"practice/block{i}.csv", pr_.to_csv(index=False) == open(
            os.path.join(pkg, "practice", f"block{i}.csv")).read())
    want = sorted(f for f in os.listdir(os.path.join(pkg, "lists"))
                  if f.endswith(".json"))
    add("lists/ (the same list files)", sorted(
        f"{lid}.json" for lid in R["lists"]) == want, f"{len(want)} lists")
    add("assignment.json", _jnorm(R["asg"]) == json.load(open(
        os.path.join(pkg, "assignment.json"))))
    add("assignment.csv", R["flat"].to_csv(index=False) == open(
        os.path.join(pkg, "assignment.csv")).read())
    add("preload.json", _jnorm(R["preload"]) == json.load(open(
        os.path.join(pkg, "preload.json"))))
    add("items.csv", R["items"].to_csv(index=False) == open(
        os.path.join(pkg, "items.csv")).read())
    theirs = json.load(open(os.path.join(pkg, "design.json")))
    mine = _jnorm(R["design"])
    extra = sorted(k for k in mine if k not in theirs)
    extra += sorted(f"source.{k}" for k in mine["source"]
                    if k not in theirs.get("source", {}))
    for k in extra:
        _drop_key(mine, k)
    names = {k: None for k in REGRESSION_NAME_KEYS}
    for k in REGRESSION_NAME_KEYS:
        _drop_key(mine, k)
        _drop_key(theirs, k)
    exact = _diff_keys(mine, theirs)
    worst = []
    dk = _diff_keys(mine, theirs, rtol=REGRESSION_RTOL, worst=worst)
    how = ("exact" if not exact else
           f"equal within {REGRESSION_RTOL:g} relative: {len(exact)} numbers "
           f"differ in the last digits (largest {max(worst):.1e}, e.g. "
           f"{exact[0]})" if not dk else f"differ at {dk[:6]}")
    add("design.json", not dk, how + f"; names left out: {list(names)}; "
        "v5c-only keys not compared: " + ", ".join(extra))
    nf = int((ck.frame().status == "FAIL").sum())
    add("the regression's own checks", nf == 0, f"{nf} FAIL of "
        f"{len(ck.rows)}")
    ok = all(r["equal"] for r in rows)
    sec = time.time() - t0
    print(f"  REGRESSION: {'PASS' if ok else 'FAIL'} ({sec:.0f} s)",
          flush=True)
    return {"ok": ok, "rows": rows, "seconds": sec, "R": R, "ck": ck}


# ============================================================
# v5c: the orders' balance against v5b_180's
# ============================================================

def measure_list(df):
    """v5b's measure_order on a list (position = trial - 1)."""
    d = pd.DataFrame({
        "position": df.trial.astype(int) - 1,
        "trial_type": df.trial_type, "mag_bin": df.mag_bin,
        "pred_sat": df.pred_sat.astype(float),
        "pred_fog": df.pred_fog.astype(float), "scene_id": df.scene_id,
        "a_is_left": df.a_is_left.astype(bool), "item_id": df.item_id,
        "repeat_of": df.repeat_of_item})
    return measure_order(d.reset_index(drop=True))


def pair_corr(lists):
    """|rank correlation| of every pair of orders of a set (positions of
    the same items)."""
    out = {}
    for b in ("A", "B"):
        ids = sorted(k for k in lists if k.startswith(b))
        pos = {k: dict(zip(lists[k].item_id, lists[k].trial)) for k in ids}
        v = []
        for x, i in enumerate(ids):
            for j in ids[x + 1:]:
                it = sorted(pos[i])
                v.append(abs(float(np.corrcoef(
                    [pos[i][q] for q in it], [pos[j][q] for q in it])[0, 1])))
        out[b] = v
    return out


def v5b_attempts():
    """v5b_180's attempts for its 40 candidate orders per set (its
    checks.csv)."""
    import re
    f = os.path.join(V5B_PACKAGE_DIR, "checks.csv")
    out = {}
    if os.path.isfile(f):
        c = pd.read_csv(f)
        for r in c[c.check.str.contains("valid candidate orders built",
                                        na=False)].itertuples():
            m = re.match(r"(\d+) in (\d+) attempts", str(r.detail))
            if m:
                out[r.scope.split()[-1]] = (int(m.group(1)), int(m.group(2)))
    return out


def balance_vs_v5b(R):
    """Each order's balance, this package and v5b_180's package."""
    ref = {}
    d = os.path.join(V5B_PACKAGE_DIR, "package", "lists")
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            ref[f[:-5]] = pd.DataFrame(json.load(open(os.path.join(d, f))))
    rows = []
    for ver, L in ((VERSION, R["lists"]), ("v5b_180", ref)):
        for lid, df in sorted(L.items()):
            m = measure_list(df)
            rows.append({"package": ver, "list_id": lid, **m})
    T = pd.DataFrame(rows)
    pc = {VERSION: pair_corr(R["lists"]), "v5b_180": pair_corr(ref)}
    gaps = {VERSION: R["gap_tab"], "v5b_180": (
        pd.read_csv(os.path.join(V5B_PACKAGE_DIR, "scene_gap_search.csv"))
        if os.path.isfile(os.path.join(V5B_PACKAGE_DIR,
                                       "scene_gap_search.csv")) else None)}
    return T, pc, gaps


def balance_lines(w, T, pc, gaps, R):
    w("3b. BALANCE AGAINST v5b_180 (did the tighter scene constraint cost "
      "anything?)")
    w("-" * 78)
    n_sc = {VERSION: int(R["info"].get("scenes_usable") or 0) or "?",
            "v5b_180": 47}
    w(f"Each kept order of both packages, measured the same way (v5b's "
      f"measure_order on the lists): {VERSION} {n_sc[VERSION]} scenes, "
      f"v5b_180 {n_sc['v5b_180']} scenes; both at scene gap {SCENE_GAP}.")
    w("Mean position (0 = first trial, 1 = last), mean over the 8 orders "
      "[worst |mean - 0.5| of any order]:")
    vers = [VERSION, "v5b_180"]
    w(f"  {'type':12s} " + "".join(f"{v:>24s}" for v in vers))
    for t in ANALYSED + ["consistency", "catch"]:
        cells = []
        for v in vers:
            s = T[T.package == v][f"mean_pos_{t}"]
            cells.append(f"{s.mean():.3f} [{(s - 0.5).abs().max():.3f}]")
        w(f"  {t:12s} " + "".join(f"{c:>24s}" for c in cells))
    w("Mean position by type x magnitude bin, worst |mean - 0.5| over the "
      "bins of a type, worst order")
    w("  (the bins differ: v5c by levels apart, v5b by quantiles of the "
      "difference):")
    for t in ANALYSED:
        cells = []
        for v in vers:
            g = T[T.package == v]
            cs = [c for c in g.columns if c.startswith(f"mean_pos_{t}_bin")
                  and g[c].notna().any()]
            vals = (g[cs] - 0.5).abs().max().max() if cs else np.nan
            cells.append(f"{len(cs)} bins, worst {vals:.3f}")
        w(f"  {t:12s} " + "".join(f"{c:>24s}" for c in cells))
    w(f"  tolerance {POS_TOL_TYPE} per type, {POS_TOL_BIN} per type x bin")
    w("corr(position, size of difference), mean [largest |r|] over the "
      "orders:")
    for k, lab in (("corr_pos_abs_x_sat", "|x_sat|"),
                   ("corr_pos_abs_x_con", "|x_con|")):
        cells = []
        for v in vers:
            s = T[T.package == v][k]
            cells.append(f"{s.mean():+.3f} [{s.abs().max():.3f}]")
        w(f"  {lab:12s} " + "".join(f"{c:>24s}" for c in cells))
    w(f"  tolerance {POS_CORR_TOL}")
    w("How alike the orders are: |rank correlation| between two orders of "
      "a set, mean [largest]:")
    for b in ("A", "B"):
        cells = []
        for v in vers:
            x = pc[v][b]
            cells.append(f"{np.mean(x):.3f} [{max(x):.3f}]" if x else "-")
        w(f"  set {b:8s} " + "".join(f"{c:>24s}" for c in cells))
    w(f"  limit {ORDER_CORR_MAX}")
    w("Other measures, mean [worst] over the orders:")
    for k, lab, f in (("score", "selection score", "{:.3f}"),
                      ("type_spread_worst", "type spread", "{:.2f}"),
                      ("scene_gap_min", "smallest scene gap", "{:.1f}"),
                      ("max_type_run", "longest type run", "{:.1f}"),
                      ("max_side_run", "longest side run", "{:.1f}")):
        cells = []
        for v in vers:
            s = T[T.package == v][k]
            worst = s.min() if k == "scene_gap_min" else s.max()
            cells.append(f"{f.format(s.mean())} [{f.format(worst)}]")
        w(f"  {lab:18s} " + "".join(f"{c:>18s}" for c in cells))
    att = v5b_attempts()
    w("Attempts the builder needed (40 valid candidate orders per set, at "
      f"gap {SCENE_GAP}):")
    for b in ("A", "B"):
        mine = R["ostats"][b]
        th = att.get(b)
        w(f"  set {b}: {VERSION} {mine['valid']} in {mine['attempts']}; "
          f"v5b_180 " + (f"{th[0]} in {th[1]}" if th else "-"))
    for v in vers:
        g = gaps.get(v)
        if g is not None and len(g):
            ok = g.groupby("gap").valid.min()
            lg = ok[ok >= N_ORDERS]
            w(f"  largest scene gap, {v}: "
              f"{int(lg.index.max()) if len(lg) else '-'}")
    w("  (balance_vs_v5b.csv: every measure of every order)")
    w("")


# ============================================================
# One package
# ============================================================

def build_size(size, ck_global, pre_rows):
    t0 = time.time()
    ck = Checks()
    for r in pre_rows:
        ck.add("preflight", f"{r['input'].strip()} found", True, r["path"])
    d3 = SCRIPT3_DIRS[size]
    reg = run_regression() if REGRESSION_CHECK else None
    print(f"\n  MAIN BUILD: {d3}", flush=True)
    R = construct(size, d3, ck, PRACTICE_FEEDBACK, PACKAGE_NAME, VERSION,
                  SCRIPT3_VERSION)
    out = resolve_output_dir(size, R["B"])
    os.makedirs(out)
    print(f"\n  n={size}: output {out}")
    # --- v5c: practice agreement, fields against v5b_180 ---
    practice_agreement(R["design"], R["pr"], ck)
    ref_fields = v5b_fields(ck)
    fdiff = field_differences(R["lists"], R["prac_rows"], ref_fields)
    ck.add("fields", f"the list and practice rows keep {V5B_PACKAGE_NAME}'s "
           "fields (JSON keys and CSV headers, in order)",
           all(r["same"] for r in fdiff), "; ".join(
               f"{r['kind']}: " + ("same" if r["same"] else
                                   f"added {r['added']}, removed "
                                   f"{r['removed']}")
               + f" ({r['n_files_v5b']} v5b files)" for r in fdiff))
    changes = experiment_changes(R["name"], R["pr"], fdiff)
    if reg is not None:
        for r in reg["rows"]:
            ck.add("regression", f"with v5b's settings on v5b's 180 build: "
                   f"{r['file']} equals {V5B_PACKAGE_NAME}'s (names apart)",
                   r["equal"], r["note"])
    pkg = f"{out}/package"
    write_package(R, pkg, ck, changes)
    T, pc, gaps = balance_vs_v5b(R)
    # --- outputs ---
    R["summ"].to_csv(os.path.join(out, "orders_summary.csv"), index=False)
    pd.DataFrame(R["cand_rows"]).to_csv(
        os.path.join(out, "order_candidates.csv"), index=False)
    R["bal"].to_csv(os.path.join(out, "assignment_balance.csv"), index=False)
    R["gap_tab"].to_csv(os.path.join(out, "scene_gap_search.csv"),
                        index=False)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(settings_dict(size), f, indent=2, default=str)
    pd.DataFrame(R["first_seg"]).to_csv(
        os.path.join(out, "first_segment_leak.csv"), index=False)
    T.to_csv(os.path.join(out, "balance_vs_v5b.csv"), index=False)
    if reg is not None:
        pd.DataFrame(reg["rows"]).to_csv(
            os.path.join(out, "regression_v5b.csv"), index=False)
    write_report(out, size, ck, R, t0, T, pc, gaps, reg, changes, fdiff)
    ck.frame().to_csv(os.path.join(out, "checks.csv"), index=False)
    n_fail = int((ck.frame().status == "FAIL").sum())
    ck_global.append((out, n_fail))
    return out, n_fail


def write_report(out, size, ck, R, t0, T, pc, gaps, reg, changes, fdiff):
    d3, info, design = R["d3"], R["info"], R["design"]
    gap_tab, largest, G_use = R["gap_tab"], R["largest"], R["G_use"]
    summ, bal, rows, asg = R["summ"], R["bal"], R["rows"], R["asg"]
    lists, preload, files = R["lists"], R["preload"], R["files"]
    norm, first_seg, edges, items = (R["norm"], R["first_seg"], R["edges"],
                                     R["items"])
    L = []
    w = L.append
    w("=" * 78)
    w(f"v5c STEP 4: TRIAL ORDERS, ASSIGNMENT AND PACKAGE, {size} PER BLOCK")
    w("make_trial_orders_v5c.py")
    w("=" * 78)
    w(f"run finished  {time.strftime('%Y-%m-%d %H:%M:%S')}   elapsed "
      f"{time.time() - t0:.0f} s")
    tm_ = R["times"]
    w(f"run times     gap search {tm_['gap_search']:.0f} s, orders "
      f"{tm_['orders']:.0f} s, images {tm_.get('images', 0):.0f} s"
      + (f"; regression {reg['seconds']:.0f} s" if reg else
         "; regression not run"))
    w(f"output        {out}")
    w(f"script 3      {d3}")
    if STOP_BEFORE_IMAGES:
        w("")
        w("!!! STOPPED BEFORE THE IMAGE STEP (STOP_BEFORE_IMAGES = True): a "
          "test run; the package has no images and cannot be frozen. !!!")
    w("")
    w("1. WHAT THIS RUN DID")
    w("-" * 78)
    w(f"{N_ORDERS} orders per set from {N_ORDER_CANDIDATES} valid candidates, "
      f"each block of {design['main_trials_per_block']} main trials")
    w(f"({size} analysed, {info['counts_per_set']['consistency']} "
      f"consistency, {info['counts_per_set']['catch']} catch). Constraints:")
    w(f"  {SEGMENTS} segments, every type within {TYPE_SPREAD_TOL} of its "
      "share in each (running balance)")
    w(f"  scene gap >= {G_use}; consistency >= {CONSISTENCY_MIN_DISTANCE} "
      f"after its original; one catch per segment,")
    w(f"  none in the first/last {CATCH_EDGE}; type runs <= "
      f"{MAX_TYPE_RUN}; side runs <= {MAX_SIDE_RUN}")
    w(f"  balance: mean position per type within {POS_TOL_TYPE}, per "
      f"type x bin within {POS_TOL_BIN}, |corr(position,")
    w(f"  size of difference)| <= {POS_CORR_TOL}; orders at most "
      f"{ORDER_CORR_MAX} rank-correlated with each other")
    pr = R["pr"]
    w(f"Practice: {pr['counts'][0]} before block 1, {pr['counts'][1]} before "
      f"block 2; feedback block 1 {'yes' if pr['feedback']['1'] else 'no'}, "
      f"block 2 {'yes' if pr['feedback']['2'] else 'no'} ({pr['feedback_source']});"
      f" feedback on {pr['feedback_trials']} trials of a block with feedback"
      f" ({pr['feedback_trials_source']}).")
    w("")
    w("2. SCENE GAP")
    w("-" * 78)
    w(f"Scenes in the sets: {info.get('scenes_usable', '?')} (v5b: 47). "
      f"Largest gap at which all {N_ORDERS} orders of both sets could be "
      f"built: {largest}")
    w(f"(searched upward from {SCENE_GAP}, {GAP_SEARCH_ATTEMPTS} attempts "
      f"per set per gap). Gap used: {G_use} (policy "
      f"'{SCENE_GAP_POLICY}'). The margin is {largest - SCENE_GAP} trial(s).")
    for G, g in gap_tab.groupby("gap"):
        w(f"  gap {G:2d}: " + "; ".join(
            f"set {r.set} {r.valid} valid in {r.attempts} attempts"
            for r in g.itertuples()))
    last = gap_tab[gap_tab.gap == gap_tab.gap.max()]
    if (last.valid < N_ORDERS).any():
        w(f"  why the search stopped at gap {gap_tab.gap.max()}: "
          + "; ".join(f"set {r.set}: {r.reasons}" for r in last.itertuples()
                      if r.valid < N_ORDERS))
    w("Realised smallest scene gap of each kept order: " + ", ".join(
        f"{r.list_id} {r.scene_gap_min}" for r in summ.itertuples()))
    w(f"  (every order must reach {G_use}; the builder places a scene's next "
      "trial as soon as the gap allows, so the smallest gap usually equals "
      "the gap set)")
    w("")
    w("3. THE ORDERS")
    w("-" * 78)
    w(f"{'list':6s} {'cand':>4s} {'gap':>4s} {'cons d':>6s} {'catch/seg':>10s}"
      f" {'type run':>8s} {'side run':>8s} {'spread':>6s} {'pos type':>8s} "
      f"{'pos bin':>7s} {'r sat':>6s} {'r con':>6s}")
    for r in summ.itertuples():
        w(f"{r.list_id:6s} {r.candidate:4d} {r.scene_gap_min:4d} "
          f"{r.consistency_min_distance:6d} {r.catch_per_segment:>10s} "
          f"{r.max_type_run:8d} {r.max_side_run:8d} "
          f"{r.type_spread_worst:6.2f} {r.mean_pos_type_worst:8.3f} "
          f"{r.mean_pos_bin_worst:7.3f} {r.corr_pos_abs_x_sat:+6.3f} "
          f"{r.corr_pos_abs_x_con:+6.3f}")
    w("  gap = smallest scene gap; cons d = smallest repeat distance; "
      "spread = worst type count")
    w("  deviation in a segment; pos type / pos bin = worst |mean position "
      "- 0.5|; r = corr(position, |x|).")
    w("")
    w("Mean position (0 = first trial, 1 = last) by trial type:")
    tcols = ANALYSED + ["consistency", "catch"]
    w(f"  {'list':6s} " + " ".join(f"{t:>11s}" for t in tcols))
    for r in summ.itertuples():
        w(f"  {r.list_id:6s} " + " ".join(
            f"{getattr(r, 'mean_pos_' + t):11.3f}" for t in tcols))
    w("  (consistency sits late by design: each repeat follows its original "
      f"by >= {CONSISTENCY_MIN_DISTANCE})")
    w("Mean position by magnitude bin, per analysed type:")
    bcols = sorted([c for c in summ.columns if "_bin" in c
                    and c.startswith("mean_pos_") and "worst" not in c])
    for t in ANALYSED:
        cs = [c for c in bcols if c.startswith(f"mean_pos_{t}_bin")]
        w(f"  {t:9s} " + "   ".join(
            f"{c.split('_bin')[1]}: " + "/".join(f"{v:.2f}" for v in summ[c])
            for c in cs) + "   (per list, in the order above)")
    w("")
    balance_lines(w, T, pc, gaps, R)
    w("4. ASSIGNMENT")
    w("-" * 78)
    w("Groups: " + "; ".join(f"{g} = {' + '.join(e)}"
                             for g, e in GROUPS.items())
      + f". {N_SLOTS} slots per group, the")
    w(f"same table for both groups; the link carries ?{SLOT_PARAM}=<n>.")
    w("Balance: the largest deviation from an even split, per factor and per "
      "pair of factors:")
    cols = [c for c in bal.columns if c != "slots"]
    nf = len(FACTORS)
    w(f"  {'slots':>5s} " + " ".join(f"{c[:20]:>20s}" for c in cols[:nf]))
    for r in bal.itertuples(index=False):
        w(f"  {r[0]:5d} " + " ".join(f"{v:20g}" for v in r[1:nf + 1]))
    w("  pairs of factors:")
    for r in bal.itertuples(index=False):
        d = dict(zip(bal.columns, r))
        w(f"  {d['slots']:5d} " + ", ".join(
            f"{c} {d[c]:g}" for c in cols[nf:]))
    w("  First 16 slots (block order, set mapping, order):")
    w("    " + "  ".join(f"{s + 1}:{''.join(map(str, c))}"
                         for s, c in enumerate(rows[:16])))
    w("  participant slot 1: " + json.dumps(
        asg["groups"]["participant"][0]["blocks"]))
    w(f"Slots beyond {N_SLOTS}: {slot_rule_text()}. After every {N_SLOTS} "
      "slots each cell has been used equally often, so the balance after "
      f"n slots equals the balance after n mod {N_SLOTS}; checked up to "
      f"{SLOT_WRAP_CYCLES * N_SLOTS} slots (rows above {N_SLOTS} in the "
      "table are those wrapped prefixes).")
    w("")
    w("5. PACKAGE")
    w("-" * 78)
    w(f"  package/  ->  copy to the experiment as data/sets/"
      f"{design['package']}/")
    w(f"  lists: {', '.join(sorted(lists))}; practice: block1 "
      f"({info['practice_by_block'][0]}), block2 "
      f"({info['practice_by_block'][1]})")
    w(f"  images: {len(files)} " + ("to ship (NOT copied: STOP_BEFORE_IMAGES)"
                                    if STOP_BEFORE_IMAGES else "shipped")
      + f", {len(preload)} preloaded (catch-only images ship but are not "
      "preloaded)")
    w(f"  catch targets: block 1 {design['catch']['targets_by_block_position']['1']}, "
      f"block 2 {design['catch']['targets_by_block_position']['2']}")
    tm = design["timing"]
    w(f"  breaks: every {tm['break_every']} main trials ({tm['break_rule']})"
      f", after trials {tm['breaks_after_trial']};")
    w(f"    stretches {tm['stretch_lengths']}, each one segment with one "
      "catch trial; rest between blocks: "
      f"{'yes' if tm['rest_between_blocks'] else 'no'}")
    w("")
    w("5b. WHAT design.json AND THE README RECORD (v5c)")
    w("-" * 78)
    rs = design.get("rule_set") or {}
    w(f"  rule set {rs.get('name')}: levers {rs.get('levers')}")
    for r in rs.get("rules") or []:
        w(_wrap("* " + r, "    "))
    ex = design.get("excluded_scenes") or {}
    w(f"  excluded scenes: {', '.join(ex.get('scenes') or []) or 'none'} "
      f"(images {ex.get('images')}); {ex.get('scenes_used')} scenes used")
    w(_wrap(f"reason: {ex.get('reason')}", "    "))
    q = (design.get("bin_quotas") or {}).get("quotas") or {}
    w("  bin quotas per set: " + "; ".join(
        f"{t} " + "/".join(str(v) for v in q[t].values()) for t in q))
    w(_wrap(f"leak limits: {limits_text(info)}", "  "))
    w(_wrap(f"shipped: {shipped_text(info)}", "  "))
    w(f"  script 3: {d3}; SHA-256 of "
      f"{len(design['source']['script3_files_sha256'])} files "
      "(stimuli/ apart) in design.json: source.script3_files_sha256")
    st = design["stimulus_set"]
    w(f"  stimulus set {st['run']}: checksums.sha256 SHA-256 "
      f"{st['checksums_sha256']}")
    w("  Places365 note: in design.json (images_note) and the README")
    w("")
    normality_lines(w, norm, first_seg, edges, info)
    w("  items.csv: " + f"{len(items)} items, {items.shape[1]} columns "
      "(every column of script 3's rows)")
    w("")
    w("7. EXPERIMENT CHANGES NEEDED (listed in the README; not made here)")
    w("-" * 78)
    w(f"  {EXPERIMENT_REPO} already runs the two-block design with "
      f"{V5B_PACKAGE_NAME}. For {design['package']}:")
    for c in changes:
        w(_wrap("* " + c, "  "))
    w("Row fields against " + V5B_PACKAGE_NAME + ":")
    for r in fdiff:
        w(f"  {r['kind']:14s} " + ("same" if r["same"] else
                                   f"added {r['added']}, removed "
                                   f"{r['removed']}"
                                   + (", order differs" if r["order_differs"]
                                      else ""))
          + f" ({r['n_files_v5b']} {V5B_PACKAGE_NAME} files)")
    w("")
    w("7b. REGRESSION AGAINST " + V5B_PACKAGE_NAME)
    w("-" * 78)
    if reg is None:
        w("  not run (REGRESSION_CHECK = False)")
    else:
        w(f"  This script's code on {REGRESSION_SCRIPT3_DIR}, with v5b's "
          f"settings {json.dumps(REGRESSION_SETTINGS)},")
        w(f"  up to the image step, against {V5B_PACKAGE_DIR}/package:")
        for r in reg["rows"]:
            w(f"  {'equal ' if r['equal'] else 'DIFFERS'} {r['file']}"
              + (f"  ({r['note']})" if r["note"] else ""))
        Rr = reg["R"]
        w(f"  regression gap search: largest {Rr['largest']} (v5b_180's "
          "report: 37); orders A " + f"{Rr['ostats']['A']['attempts']}, B "
          f"{Rr['ostats']['B']['attempts']} attempts")
        w(f"  RESULT: {'PASS' if reg['ok'] else 'FAIL'}")
    w("")
    w("8. CHECKS")
    w("-" * 78)
    cf = ck.frame()
    for row in cf.itertuples():
        w(f"  {row.status:7s} [{row.scope}] {row.check}"
          + (f"  ({row.detail})" if row.detail else ""))
    w("")
    w(f"  {int((cf.status == 'PASS').sum())} PASS, "
      f"{int((cf.status == 'FAIL').sum())} FAIL, "
      f"{int((cf.status == 'SKIPPED').sum())} SKIPPED")
    with open(os.path.join(out, "orders_report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


# ============================================================
# v5c: preflight
# ============================================================

def preflight():
    rows = []

    def add(what, path, ok, required=True, note=""):
        rows.append({"input": what, "path": str(path), "found": bool(ok),
                     "required": required, "note": note})

    def run_dir(label, d, files, images):
        add(label, d, os.path.isdir(d))
        if not os.path.isdir(d):
            return
        for f in files:
            add(f"  {f}", f"{d}/{f}", os.path.isfile(f"{d}/{f}"))
        if images is not None:
            n = (len([x for x in os.listdir(f"{d}/stimuli")])
                 if os.path.isdir(f"{d}/stimuli") else 0)
            add("  stimuli/", f"{d}/stimuli", n > 0, images,
                f"{n} files" + ("" if images else
                                "; not needed (STOP_BEFORE_IMAGES)"))

    s3files = ["set_A.csv", "set_B.csv", "practice_block1.csv",
               "practice_block2.csv", "set_info.json", "checks.csv",
               "manifest.csv", "candidates.csv", "settings.json"]
    for size, d in SCRIPT3_DIRS.items():
        run_dir(f"script 3 run (n={size})", d, s3files,
                not STOP_BEFORE_IMAGES)
        si = f"{d}/set_info.json"
        if os.path.isfile(si):
            sd = json.load(open(si)).get("stimulus_set", {}).get("dir", "")
            add("stimulus set (from set_info)", sd, os.path.isdir(sd))
            for f in ("checksums.sha256", "FROZEN.json"):
                add(f"  {f}", f"{sd}/{f}", os.path.isfile(f"{sd}/{f}"))
    run_dir(f"{V5B_PACKAGE_NAME} run (fields, balance)", V5B_PACKAGE_DIR,
            ["checks.csv", "scene_gap_search.csv", "package/design.json",
             "package/assignment.json", "package/assignment.csv",
             "package/items.csv", "package/preload.json",
             "package/lists/A_o1.json", "package/practice/block1.json"],
            None)
    if REGRESSION_CHECK:
        run_dir("v5b 180 build (regression)", REGRESSION_SCRIPT3_DIR,
                s3files, None)
    else:
        rows.append({"input": "v5b 180 build (regression)",
                     "path": REGRESSION_SCRIPT3_DIR, "found": None,
                     "required": False,
                     "note": "REGRESSION_CHECK = False: not run, not needed"})
    print("  preflight:")
    for r in rows:
        tag = ("skipped" if r["found"] is None else "found  " if r["found"]
               else ("MISSING" if r["required"] else "absent "))
        print(f"    {tag} {r['input']:34s} {r['path']}"
              + (f"  ({r['note']})" if r["note"] else ""))
    miss = [r for r in rows if r["required"] and r["found"] is False]
    if miss:
        print(f"\n  STOPPED before any computation: {len(miss)} required "
              "input(s) missing:")
        for r in miss:
            print(f"    {r['input'].strip()}: {r['path']}")
        print("  Run the script from the working folder, with the data under "
              "data/, or set the paths at the top of the script.")
        raise SystemExit(2)
    return [r for r in rows if r["found"]]


def main():
    t0 = time.time()
    print("v5c step 4: trial orders, assignment and package")
    if SCENE_GAP_POLICY not in ("fixed", "largest"):
        raise ValueError("SCENE_GAP_POLICY must be 'fixed' or 'largest'")
    if BREAK_EVERY != "per_segment" and not (
            isinstance(BREAK_EVERY, int) and BREAK_EVERY > 0):
        raise ValueError("BREAK_EVERY must be 'per_segment' or a positive "
                         "integer")
    if not (PRACTICE_FEEDBACK == "from_script3" or (
            isinstance(PRACTICE_FEEDBACK, (tuple, list))
            and len(PRACTICE_FEEDBACK) == 2)):
        raise ValueError("PRACTICE_FEEDBACK must be 'from_script3' or a "
                         "pair of booleans")
    pre = preflight()
    results = []
    for size in SCRIPT3_DIRS:
        build_size(size, results, pre)
    for out, n_fail in results:
        print(f"  {out}: {n_fail} check(s) FAIL")
    print(f"  elapsed {time.time() - t0:.0f} s")
    if any(n for _, n in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
