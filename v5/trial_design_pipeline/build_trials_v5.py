"""
build_trials_v5.py  --  v5 step 3: build the two block lists (sets A and B)

WHAT THIS DOES
    For each block size in BLOCK_SIZES it builds one package: two matched
    sets of pairs (A and B; which set goes with which block is decided per
    participant by script 4's assignment table), each with its own
    consistency and catch trials, plus the practice trials for the first
    and the second block, the stimulus images, and the per-set thresholds.
    Nothing here is ordered: script 4 makes the trial orders.

    1. CANDIDATES. N_CANDIDATES pairs of matched sets are drawn with
       script 1's own functions (imported from analyse_pair_capacity_v5.py:
       stage1, one_draw, evaluate), under script 1's settings. Candidate d
       uses script 1's seed for draw d, so candidates 0-19 are script 1's
       and script 2's draws. Every candidate is checked with script 1's
       matching checks.
    2. SELECTION. Among the candidates that pass every check, the one to
       ship is chosen by SELECTION_KEYS, applied in order: by default the
       one whose WEAKER set has the most contrast information, ties broken
       on the smallest A-B difference in contrast information, then on the
       weaker set's saturation information. Every candidate's values are in
       candidates.csv and the report.
    3. CONSISTENCY trials: CONSISTENCY_RATE of each block (v4's 1 in 12),
       stratified by type. They are drawn as matched COUPLES, so set A and
       set B repeat each other's partner trials; each repeat swaps sides.
    4. PRACTICE, per block POSITION (the same for everyone, because the set
       to block mapping is counterbalanced): PRACTICE_BY_BLOCK trials of
       PRACTICE_TYPES in turn, at or below the PRACTICE_MAG_QUANTILE of each
       type's differences (v4's rule), one trial per scene within a block,
       from pairs neither set uses. Drawn before catch (as in v4), so a
       change to the catch count cannot move the practice trials.
    5. CATCH trials, per set: CATCH_PER_BLOCK rows in 'no_images' mode,
       from unused 'together' pairs (v4's rule). The pair is never shown.
    6. IMAGES. Every image any row references is copied from STIMULI_SRC,
       the copy is compared byte for byte with the source (SHA-256), and
       sat_mean and contrast_local are RE-MEASURED from the copy with v4's
       feature code and compared with the pair table. A missing image, a
       copy that differs, or a re-measured feature outside REMEASURE_TOL
       stops the build: the report is written, BUILD_FAILED.txt is left in
       the folder (scripts 4 and 5 refuse such a folder), and the script
       exits with an error.
    7. THRESHOLDS per set: N_THRESHOLD_PEOPLE random choosers fitted with
       v4's registered estimator (Firth); the 2.5% and 97.5% percentiles of
       each weight. Recorded in set_info.json for design.json.

    PREFER_LARGE_DIFFS (off by default): when on, the draw prefers pairs
    with larger differences on each type's driver (mainly contrast), within
    the fixed magnitude bins, so a later power analysis can test whether
    more information per trial helps. The report then compares the
    information and the range of differences with an unweighted draw.

REUSED
    analyse_pair_capacity_v5.py (imported): load_pairs, build_pool, stage1,
        one_draw, evaluate, largest_remainder, balanced_sides, Streams,
        Checks, pair_key; greedy_type is copied (with a weighted order) only
        for PREFER_LARGE_DIFFS.
    simulate_power_v4_n10_20.py (imported): fit_firth_batch, simulate_many,
        informative_medians, as_probability.
    v4 feature code (_features_of in build_trials_v4_corrected.py,
        features_of in freeze_trial_list_v4.py): copied, as v4 does, so the
        re-measurement is independent of the code that built the table.

INPUTS (run from the project folder)
    analyse_pair_capacity_v5.py, simulate_power_v4_n10_20.py and the inputs
    script 1 reads; STIMULI_SRC (the rendered stimulus set); MANIFEST_CSV
    (optional cross-check of the re-measured features); SCRIPT1_RUN_DIR
    (optional, for the draw-0 reproduction check).

OUTPUTS: one folder per block size, OUTPUT_ROOT/<settings>__s<hash>/
    set_A.csv, set_B.csv     each set's analysed, consistency and catch rows,
                             unordered, with a stable item_id
    practice_block1.csv, practice_block2.csv
    candidates.csv           every candidate's values and the selection
    set_info.json            counts, thresholds, catch rule, provenance
    manifest.csv             every shipped image: SHA-256, re-measured
                             features
    stimuli/                 the images
    build_report.txt, checks.csv, settings.json
    BUILD_FAILED.txt         only when an image check fails

RUN       python build_trials_v5.py
"""

# ============================================================
# SETTINGS
# ============================================================

# --- Imported scripts (in the project folder) ---
CAPACITY_SCRIPT = "analyse_pair_capacity_v5.py"
V4_SIM_SCRIPT = "simulate_power_v4_n10_20.py"

# CAPACITY_OVERRIDES
#   Script 1 settings to change for this build only. Empty = script 1's
#   file as saved. Every script 1 setting in force goes to settings.json.
CAPACITY_OVERRIDES = {}

# SCRIPT1_RUN_DIR
#   Script 1's output folder with example sets for these sizes, for the
#   check that candidate 0 is script 1's draw 0. None or missing = skipped.
SCRIPT1_RUN_DIR = ("data/pair_capacity_v5/disjoint-pair_match-scene_bin_cap2"
                   "_satcut31.1_contol0.72_sizes120-150-180_levels-without_p65"
                   "_draws20_seed42__s4d22db73")

# --- The design ---
BLOCK_SIZES = [150, 180]        # one package per size
MIX = "conflict_heavy"          # a key of script 1's TYPE_MIXES
LEVEL_SET = "without_p65"       # a key of script 1's LEVEL_SETS

# --- Candidates and selection ---
N_CANDIDATES = 50

# SELECTION_KEYS
#   Applied in order to the candidates that pass every check; each is
#   (metric, "max" or "min"). Metrics available:
#     weaker_set_info_con    min over A and B of sum x_con^2
#     weaker_set_info_sat    min over A and B of sum x_sat^2
#     ab_diff_info_con       |A - B| in sum x_con^2
#     ab_diff_info_sat       |A - B| in sum x_sat^2
#     total_info_con, total_info_sat   A + B
SELECTION_KEYS = [("weaker_set_info_con", "max"),
                  ("ab_diff_info_con", "min"),
                  ("weaker_set_info_sat", "max")]

# SELECTION_ROUND
#   Metrics are rounded to this many decimals before ranking, so a tie is a
#   tie rather than a difference in the last bit of floating point.
SELECTION_ROUND = 6

# --- Optional: prefer larger differences (OFF by default) ---
#
# PREFER_LARGE_DIFFS
#   When True, within each magnitude bin the draw takes pairs in a random
#   order weighted by (driver / bin median) ** PREFER_STRENGTH, so larger
#   differences come first; the bins and their quotas are unchanged. The
#   driver is |x_sat| for sat_only and |x_con| for the other types; only the
#   types in PREFER_TYPES are weighted. The report compares the result with
#   an unweighted draw of the same candidates.
PREFER_LARGE_DIFFS = False
PREFER_STRENGTH = 2.0
PREFER_TYPES = ("fog_only", "together", "conflict")

# --- Trials added to each block ---
#
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
STIMULI_SRC = "data/stimulus_set_v4_ext_haze_eq/stimuli"
MANIFEST_CSV = "data/stimulus_set_v4_ext_haze_eq/stimulus_manifest.csv"
IMAGE_PX = 512                  # every shipped image must be this size
FEATURE_TARGET_SIZE = (256, 256)  # v4's measurement convention; keep
# REMEASURE_TOL
#   Largest allowed |re-measured - table| for sat_mean and contrast_local.
#   The images are copied unchanged, so v4's own package re-measured with a
#   shift of 0.000; anything visible means the wrong file.
REMEASURE_TOL = 1e-4

# --- Thresholds ---
N_THRESHOLD_PEOPLE = 20_000
THRESHOLD_ALPHA = 0.025

# --- Checks (as script 2) ---
V4_CHOICE_PROB = (0.6191, 0.6104)
CHOICE_PROB_TOL = 0.02
PLANNING_WEIGHTS = (0.5, 0.7)
INFORMATIVE_SD = 0.20
SET_THRESHOLD_TOL = 0.05

OUTPUT_ROOT = "data/trials_v5"
ON_EXISTING = "new"
RANDOM_SEED = 42


# ============================================================
# Imports
# ============================================================

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

W_NAMES = ("sat", "con")
MAIN_TYPES = ["sat_only", "fog_only", "together", "conflict"]


def load_module(path, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Run this script from the project folder, "
            "where the v4 and v5 scripts are.")
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


# ============================================================
# Settings, output folder
# ============================================================

def settings_dict(cap, size):
    own = {k: v for k, v in globals().items()
           if k.isupper() and not k.startswith("_")
           and k not in ("W_NAMES", "MAIN_TYPES")}
    own["BLOCK_SIZE"] = size
    script1 = {k: getattr(cap, k) for k in sorted(dir(cap))
               if k.isupper() and not k.startswith("_")}
    return {"script3": own, "script1_in_force": script1}


def check_settings(cap):
    if MIX not in cap.TYPE_MIXES:
        raise ValueError(f"MIX {MIX!r} is not in script 1's TYPE_MIXES")
    if LEVEL_SET not in cap.LEVEL_SETS:
        raise ValueError(f"LEVEL_SET {LEVEL_SET!r} is not in script 1's "
                         "LEVEL_SETS")
    allowed = {"weaker_set_info_con", "weaker_set_info_sat",
               "ab_diff_info_con", "ab_diff_info_sat", "total_info_con",
               "total_info_sat"}
    for k, d in SELECTION_KEYS:
        if k not in allowed or d not in ("max", "min"):
            raise ValueError(f"SELECTION_KEYS entry {(k, d)!r} is not "
                             f"valid; metrics: {sorted(allowed)}")
    if CATCH_MODE != "no_images":
        raise ValueError("Only CATCH_MODE 'no_images' is implemented by the "
                         "experiment.")
    if len(PRACTICE_BY_BLOCK) != 2:
        raise ValueError("PRACTICE_BY_BLOCK needs two counts")
    for k in CAPACITY_OVERRIDES:
        if not hasattr(cap, k):
            raise ValueError(f"CAPACITY_OVERRIDES names {k!r}, which script "
                             "1 does not have")


def resolve_output_dir(cap, size):
    blob = json.dumps(settings_dict(cap, size), sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode()).hexdigest()[:8]
    name = (f"v5_n{size}_{MIX}_{LEVEL_SET}_cand{N_CANDIDATES}"
            f"_prefer-{'on' if PREFER_LARGE_DIFFS else 'off'}"
            f"_seed{RANDOM_SEED}__s{h}")
    base = os.path.join(OUTPUT_ROOT, name)
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
# Optional weighted draw (PREFER_LARGE_DIFFS)
# ============================================================

def make_weighted_greedy(cap):
    """Script 1's greedy_type, copied, with ONE change: within each bin the
    pairs are taken in a random order weighted by (driver / bin median) **
    PREFER_STRENGTH (exponential-race keys, i.e. weighted sampling without
    replacement) instead of a uniform shuffle. Everything else -- the
    scene spreading, the nearest partner, the cap -- is unchanged."""

    def greedy_type(tp, t, quotas, state, rng):
        recs = cap._records(tp)
        bins = np.array([r["mag_bin"] for r in recs])
        scn = np.array([r["scene_id"] for r in recs])
        ax = np.abs(np.array([[r["x_sat"], r["x_con"]] for r in recs]))
        drv = ax[:, 0] if t == "sat_only" else ax[:, 1]
        couples = []
        couple = int(rng.integers(2))
        ok = True
        for b in rng.permutation(sorted(quotas)):
            want = quotas[b]
            idx = np.where(bins == b)[0]
            placed = 0
            if t in PREFER_TYPES and len(idx):
                wgt = (drv[idx] / np.median(drv[idx])) ** PREFER_STRENGTH
                keys = -np.log(rng.random(len(idx))) / np.maximum(wgt, 1e-12)
                remaining = list(idx[np.argsort(keys, kind="stable")])
            else:
                remaining = list(rng.permutation(idx))
            while placed < want and remaining:
                if cap.SPREAD_SCENES:
                    use = np.array([state.scene_use.get(scn[k], 0)
                                    for k in remaining])
                    pos = int(np.flatnonzero(use == use.min())[0])
                else:
                    pos = 0
                i = remaining.pop(pos)
                ri = recs[i]
                if ri["key"] in state.used:
                    continue
                first = "A" if couple % 2 == 0 else "B"
                for a in (first, "B" if first == "A" else "A"):
                    o = "B" if a == "A" else "A"
                    if not state.allowed(ri, a, t):
                        continue
                    cand = idx[idx != i]
                    if cap.SCENE_MATCH == "scene_bin":
                        cand = cand[scn[cand] == ri["scene_id"]]
                    best, bd = None, np.inf
                    for j in cand:
                        rj = recs[j]
                        if rj["key"] in state.used:
                            continue
                        if cap.DISJOINT == "image" and (
                                {rj["file_a"], rj["file_b"]}
                                & {ri["file_a"], ri["file_b"]}):
                            continue
                        if not state.allowed(rj, o, t):
                            continue
                        d = ((ax[j] - ax[i]) ** 2).sum()
                        if d < bd:
                            best, bd = j, d
                    if best is None:
                        continue
                    state.take(ri, a, t)
                    state.take(recs[best], o, t)
                    couples.append((i, best) if a == "A" else (best, i))
                    couple += 1
                    placed += 1
                    break
            if placed < want:
                ok = False
        return ok, couples

    return greedy_type


# ============================================================
# Candidates
# ============================================================

def cand_metrics(sets):
    out = {}
    for blk in ("A", "B"):
        m = sets[blk][sets[blk].role == "analysed"]
        out[f"info_sat_{blk}"] = float((m.x_sat ** 2).sum())
        out[f"info_con_{blk}"] = float((m.x_con ** 2).sum())
    for w in W_NAMES:
        a, b = out[f"info_{w}_A"], out[f"info_{w}_B"]
        out[f"weaker_set_info_{w}"] = min(a, b)
        out[f"ab_diff_info_{w}"] = abs(a - b)
        out[f"total_info_{w}"] = a + b
    return out


def draw_candidates(cap, pools_t, counts, s1, scenes, size, weighted):
    """N_CANDIDATES draws with script 1's one_draw (script 1's seeds), each
    evaluated with script 1's checks. Returns (rows, {d: sets})."""
    combo = f"{LEVEL_SET}|{size}|{MIX}"
    original = cap.greedy_type
    if weighted:
        cap.greedy_type = make_weighted_greedy(cap)
    rows, kept = [], {}
    try:
        for d in range(N_CANDIDATES):
            sets, how, n_allowed = cap.one_draw(pools_t, counts, s1, scenes,
                                                combo, d)
            row = {"candidate": d, "built": how}
            if sets is None:
                row["passes"] = False
                rows.append(row)
                continue
            met, chk = cap.evaluate(sets, counts, s1, n_allowed)
            failed = [k for k, v in chk.items() if v is False]
            row.update(cand_metrics(sets))
            row["passes"] = not failed
            row["failed_checks"] = "; ".join(failed)
            row["ks_sat"], row["ks_con"] = met["ks_sat"], met["ks_con"]
            row["cos_A"], row["cos_B"] = met["cos_A"], met["cos_B"]
            rows.append(row)
            kept[d] = (sets, met, chk)
    finally:
        cap.greedy_type = original
    return pd.DataFrame(rows), kept


def driver_spread(cand_w, kept_w, cand_u, kept_u):
    """Per type, the quantiles of each type's driver (|x_sat| for sat_only,
    |x_con| otherwise) over the analysed trials of every PASSING candidate,
    weighted against unweighted, and the shipped candidate under each
    rule."""
    rows = []
    _, cw = select(cand_w)
    _, cu = select(cand_u)
    ship_w = int(cw.loc[cw.chosen.fillna(False).astype(bool),
                        "candidate"].iloc[0])
    ship_u = int(cu.loc[cu.chosen.fillna(False).astype(bool),
                        "candidate"].iloc[0])
    for label, cand, kept, ship in (("weighted", cand_w, kept_w, ship_w),
                                    ("unweighted", cand_u, kept_u, ship_u)):
        ok = set(cand.loc[cand.passes, "candidate"])
        for scope, ids in (("all passing", ok), ("shipped", {ship})):
            frames = []
            for d in ids:
                for blk in ("A", "B"):
                    s_ = kept[d][0][blk]
                    frames.append(s_[s_.role == "analysed"])
            allm = pd.concat(frames)
            for t in MAIN_TYPES:
                m = allm[allm.trial_type == t]
                drv = (m.x_sat if t == "sat_only" else m.x_con).abs()
                rows.append({"draw": label, "scope": scope, "type": t,
                             "q10": drv.quantile(0.1),
                             "median": drv.median(),
                             "q90": drv.quantile(0.9), "max": drv.max(),
                             "info_driver_per_set": float((drv ** 2).sum())
                             / (2 * len(ids))})
    return pd.DataFrame(rows), ship_u


def select(cand):
    ok = cand[cand.passes].copy()
    if not len(ok):
        return None, cand
    sort_cols, asc = [], []
    for k, d in SELECTION_KEYS:
        ok[f"_r_{k}"] = ok[k].round(SELECTION_ROUND)
        sort_cols.append(f"_r_{k}")
        asc.append(d == "min")
    sort_cols.append("candidate")       # last resort: the lowest index
    asc.append(True)
    ok = ok.sort_values(sort_cols, ascending=asc, kind="stable")
    rank = {c: i + 1 for i, c in enumerate(ok.candidate)}
    cand = cand.copy()
    cand["rank"] = cand.candidate.map(rank)
    cand["chosen"] = cand.candidate == ok.candidate.iloc[0]
    return int(ok.candidate.iloc[0]), cand


# ============================================================
# Rows
# ============================================================

KEEP_COLS = ["scene_id", "category", "group", "indoor", "type",
             "colour_a", "fog_a_level", "colour_b", "fog_b_level",
             "chroma_a", "haze_a", "chroma_b", "haze_b",
             "file_a", "file_b", "sat_a", "sat_b", "fog_a", "fog_b",
             "d_sat", "d_fog", "pred_sat", "pred_fog", "z_sat", "z_fog",
             "balance", "balanced", "mag_bin", "key"]


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
    out = out.rename(columns={"trial_type": "trial_type"})
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
    the original pair with its sides swapped."""
    counts = mainA.trial_type.value_counts().to_dict()
    total = sum(counts.values())
    share = {t: counts.get(t, 0) / total for t in MAIN_TYPES}
    alloc = None
    # reuse script 1's largest remainder on the type shares
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
    PRACTICE_MAG_QUANTILE of each type's driver, one scene at most once."""
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

def ship_images(rows, out_dir, ck, manifest):
    """Copy, hash and re-measure every referenced image. Returns
    (manifest frame, list of failures)."""
    exp = {}
    for r in rows.itertuples():
        exp.setdefault(r.file_a, (r.sat_a, r.fog_a))
        exp.setdefault(r.file_b, (r.sat_b, r.fog_b))
    need = sorted(exp)
    missing = [f for f in need
               if not os.path.isfile(os.path.join(STIMULI_SRC, f))]
    ck.add("images", "every referenced image exists in STIMULI_SRC",
           not missing, f"{len(need)} needed, {len(missing)} missing"
           + (f": {missing[:5]}" if missing else ""))
    if missing:
        return None, [f"missing: {f}" for f in missing]
    dst = os.path.join(out_dir, "stimuli")
    os.makedirs(dst)
    recs, bad_copy, bad_feat, bad_size, bad_man = [], [], [], [], []
    mi = manifest.set_index("filename") if manifest is not None else None
    for f in need:
        src = os.path.join(STIMULI_SRC, f)
        out = os.path.join(dst, f)
        shutil.copy2(src, out)
        h_src, h_out = sha256_of(src), sha256_of(out)
        if h_src != h_out:
            bad_copy.append(f)
        size = image_size(out)
        if size != (IMAGE_PX, IMAGE_PX):
            bad_size.append(f"{f} {size}")
        s, c = features_of(out)
        es, ec = exp[f]
        ds, dc = abs(s - es), abs(c - ec)
        if max(ds, dc) > REMEASURE_TOL:
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
    else:
        ck.add("images", "re-measured features equal the stimulus manifest",
               "SKIPPED", "MANIFEST_CSV not found")
    fails = ([f"copy differs: {f}" for f in bad_copy]
             + [f"wrong size: {f}" for f in bad_size]
             + [f"features differ: {f}" for f in bad_feat]
             + [f"manifest differs: {f}" for f in bad_man])
    return pd.DataFrame(recs), fails


# ============================================================
# Thresholds
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
    rows = []
    for blk in ("A", "B"):
        m = main[blk]
        for t in MAIN_TYPES:
            mt = m[m.trial_type == t]
            rows.append((blk, t, len(mt), (mt.pred_sat ** 2).sum(),
                         (mt.pred_fog ** 2).sum(), q_str(mt.pred_sat),
                         q_str(mt.pred_fog), mt.scene_id.nunique(),
                         int(mt.a_is_left.sum())))
    for r in rows:
        w(f"  {r[0]}  {r[1]:9s} n={r[2]:3d}  info sat {r[3]:7.2f}  con "
          f"{r[4]:6.2f}  scenes {r[7]:2d}  A-left {r[8]:3d}")
        w(f"       |x_sat| min/10/50/90/max {r[5]}")
        w(f"       |x_con| min/10/50/90/max {r[6]}")


# ============================================================
# One package
# ============================================================

def build_one(cap, v4, size, pools_t, scenes, p_log, t0):
    ck = cap.Checks()
    out_dir = resolve_output_dir(cap, size)
    os.makedirs(out_dir)
    print(f"\n  n={size}: output {out_dir}")
    counts = cap.largest_remainder(size, cap.TYPE_MIXES[MIX])
    s1 = cap.stage1(pools_t, counts, scenes, cap.MAX_PER_SCENE_PER_TYPE)
    ck.add("design", "the counts are feasible (script 1 stage 1)",
           s1["status"] == "feasible", s1["status"])
    if s1["status"] != "feasible":
        raise RuntimeError(f"n={size}: counts not feasible ({s1['status']})")

    # --- candidates ---
    cand, kept = draw_candidates(cap, pools_t, counts, s1, scenes, size,
                                 PREFER_LARGE_DIFFS)
    comparison = None
    spread = None
    if PREFER_LARGE_DIFFS:
        comparison, kept_u = draw_candidates(cap, pools_t, counts, s1,
                                             scenes, size, False)
        spread = driver_spread(cand, kept, comparison, kept_u)
    n_pass = int(cand.passes.sum())
    ck.add("candidates", "at least one candidate passes every script 1 "
           "check", n_pass > 0, f"{n_pass} of {N_CANDIDATES}")
    chosen, cand = select(cand)
    if chosen is None:
        raise RuntimeError(f"n={size}: no candidate passes every check")
    sets, met, chk = kept[chosen]
    print(f"    {n_pass}/{N_CANDIDATES} candidates pass; chosen "
          f"candidate {chosen}")

    # reproducibility: the chosen candidate rebuilt from its seed
    combo = f"{LEVEL_SET}|{size}|{MIX}"
    original = cap.greedy_type
    if PREFER_LARGE_DIFFS:
        cap.greedy_type = make_weighted_greedy(cap)
    try:
        again, _, _ = cap.one_draw(pools_t, counts, s1, scenes, combo, chosen)
    finally:
        cap.greedy_type = original
    same = all(list(again[b].key) == list(sets[b].key) for b in ("A", "B"))
    ck.add("candidates", "the chosen candidate is rebuilt identically from "
           "its seed", same, f"candidate {chosen}")

    # draw 0 against script 1's example set
    ex = (os.path.join(SCRIPT1_RUN_DIR, "example_sets",
                       f"{LEVEL_SET}_n{size}_{MIX}")
          if SCRIPT1_RUN_DIR else None)
    if PREFER_LARGE_DIFFS:
        ck.add("candidates", "candidate 0 equals script 1's draw 0",
               "SKIPPED", "PREFER_LARGE_DIFFS changes the draws")
    elif ex and os.path.isdir(ex) and 0 in kept:
        ok0 = True
        for blk in ("A", "B"):
            saved = pd.read_csv(os.path.join(ex, f"set_{blk}.csv"))
            ok0 &= (set(map(tuple, saved[["file_a", "file_b"]].to_numpy()))
                    == set(map(tuple, kept[0][0][blk][["file_a", "file_b"]]
                               .to_numpy())))
        ck.add("candidates", "candidate 0 equals script 1's draw 0 "
               "(example set)", ok0)
    else:
        ck.add("candidates", "candidate 0 equals script 1's draw 0",
               "SKIPPED", "no example set for this size in SCRIPT1_RUN_DIR")

    # --- the chosen sets: script 1's checks, listed one by one ---
    for k, v in chk.items():
        ck.add("sets (script 1 checks)", k,
               "SKIPPED" if v is None else bool(v),
               "" if v is not None else "not applicable")

    # --- rows ---
    st = cap.Streams(RANDOM_SEED, "build_trials_v5", size)
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

    # --- checks on the rows ---
    for b in ("A", "B"):
        m, c, x = main[b], cons[b], catch[b]
        tc = m.trial_type.value_counts().to_dict()
        ck.add(f"set {b}", "analysed type counts equal the target",
               all(tc.get(t, 0) == counts[t] for t in MAIN_TYPES),
               ", ".join(f"{t} {tc.get(t, 0)}" for t in MAIN_TYPES))
        so = m[m.trial_type == "sat_only"]
        tol = cap.SAT_ONLY_CON_TOL
        ck.add(f"set {b}", f"saturation-only pairs: measured |d_con| <= "
               f"{tol} (contrast-matching rule)",
               bool((so.d_fog.abs() <= tol).all()),
               f"largest {so.d_fog.abs().max():.3f}, median "
               f"{so.d_fog.abs().median():.3f} contrast units")
        ck.add(f"set {b}", f"saturation-only pairs: |d_sat| >= "
               f"{cap.SAT_ONLY_MIN_DSAT}",
               bool((so.d_sat.abs() >= cap.SAT_ONLY_MIN_DSAT).all()),
               f"smallest {so.d_sat.abs().min():.2f}")
        fo = m[m.trial_type == "fog_only"]
        ck.add(f"set {b}", f"contrast-only pairs: |d_sat| <= "
               f"{cap.FOG_ONLY_SAT_TOL}",
               bool((fo.d_sat.abs() <= cap.FOG_ONLY_SAT_TOL).all()),
               f"largest {fo.d_sat.abs().max():.3f}")
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
        for lab, df in (("consistency", c), ("catch", x)):
            nl = int(df.a_is_left.sum())
            if lab == "catch":
                ck.add(f"set {b}", "catch sides balanced (+-1)",
                       nl in (len(df) // 2, (len(df) + 1) // 2),
                       f"{nl} of {len(df)}")
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
    lv = set(allrows.colour_a) | set(allrows.colour_b)
    dropped = set(cap.LEVEL_SETS[LEVEL_SET]["drop_colour"])
    ck.add("all rows", "no dropped level and no unedited image in any row",
           not (lv & dropped)
           and not ((allrows.fog_a_level == cap.ORIGINAL_FOG).any()
                    or (allrows.fog_b_level == cap.ORIGINAL_FOG).any()),
           f"colour levels {sorted(lv)}")

    # --- choice probability and thresholds per set ---
    info = {}
    for b in ("A", "B"):
        X = main[b][["pred_sat", "pred_fog"]].to_numpy(float)
        th = thresholds(v4, X, st(f"thresholds{b}"))
        cp = choice_prob(v4, X)
        info[b] = {"th": th, "cp": cp, "X": X}
        ck.add(f"set {b}", "planning weights give v4's choice probabilities "
               f"within {CHOICE_PROB_TOL}",
               abs(cp["sat"][2] - V4_CHOICE_PROB[0]) <= CHOICE_PROB_TOL
               and abs(cp["con"][2] - V4_CHOICE_PROB[1]) <= CHOICE_PROB_TOL,
               f"saturation {cp['sat'][2]:.4f}, contrast {cp['con'][2]:.4f}")
        ck.add(f"set {b}", "every threshold fit converged",
               th["unconverged"] == 0,
               f"share unconverged {th['unconverged']:.5f}")
    rel = max(abs(info["A"]["th"]["upper"][k] - info["B"]["th"]["upper"][k])
              / np.mean([info["A"]["th"]["upper"][k],
                         info["B"]["th"]["upper"][k]]) for k in range(2))
    ck.add("thresholds", f"the two sets' upper thresholds agree within "
           f"{SET_THRESHOLD_TOL:.0%}", rel <= SET_THRESHOLD_TOL,
           f"largest relative difference {rel:.1%}")

    # --- write the lists before the images, so a failure leaves them ---
    for b in ("A", "B"):
        full[b].drop(columns=["key"]).to_csv(
            os.path.join(out_dir, f"set_{b}.csv"), index=False)
    for i, label in enumerate(prac):
        prac[label].drop(columns=["key"]).to_csv(
            os.path.join(out_dir, f"practice_block{i + 1}.csv"), index=False)
    cand.to_csv(os.path.join(out_dir, "candidates.csv"), index=False)
    if comparison is not None:
        comparison.to_csv(os.path.join(out_dir,
                                       "candidates_unweighted.csv"),
                          index=False)
    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump(settings_dict(cap, size), f, indent=2, default=str)

    # --- images ---
    manifest = (pd.read_csv(MANIFEST_CSV)
                if MANIFEST_CSV and os.path.isfile(MANIFEST_CSV) else None)
    man, fails = ship_images(allrows.drop_duplicates("item_id"), out_dir, ck,
                             manifest)
    if man is not None:
        man.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)

    # --- set_info.json ---
    set_info = {
        "version": "v5", "block_size": size, "mix": MIX,
        "level_set": LEVEL_SET, "chosen_candidate": chosen,
        "selection_keys": SELECTION_KEYS,
        "prefer_large_diffs": PREFER_LARGE_DIFFS,
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
        "predictors": "pred_sat = -d_sat / 39.182, pred_fog = d_fog / 9.348; "
                      "positive = image A is the predicted choice",
        "script1_sha256": sha256_of(CAPACITY_SCRIPT),
        "script3_sha256": sha256_of(os.path.abspath(__file__)),
    }
    with open(os.path.join(out_dir, "set_info.json"), "w") as f:
        json.dump(set_info, f, indent=2, default=float)

    # --- report ---
    if spread is not None:
        spread[0].to_csv(os.path.join(out_dir, "prefer_spread.csv"),
                         index=False)
    write_report(cap, out_dir, size, ck, p_log, cand, comparison, chosen,
                 main, cons, catch, prac, info, counts, n_cons, alloc, met,
                 man, fails, t0, spread)
    ck.frame().to_csv(os.path.join(out_dir, "checks.csv"), index=False)
    n_fail = int((ck.frame().status == "FAIL").sum())
    if fails:
        with open(os.path.join(out_dir, "BUILD_FAILED.txt"), "w") as f:
            f.write("THE IMAGE CHECKS FAILED. Do not use this folder.\n\n"
                    + "\n".join(fails) + "\n")
    return out_dir, n_fail, fails


def write_report(cap, out_dir, size, ck, p_log, cand, comparison, chosen,
                 main, cons, catch, prac, info, counts, n_cons, alloc, met,
                 man, fails, t0, spread=None):
    L = []
    w = L.append
    w("=" * 78)
    w(f"v5 STEP 3: THE TWO BLOCK LISTS, {size} ANALYSED TRIALS PER BLOCK")
    w("build_trials_v5.py")
    w("=" * 78)
    w(f"run finished  {time.strftime('%Y-%m-%d %H:%M:%S')}   elapsed "
      f"{time.time() - t0:.0f} s")
    w(f"output        {out_dir}")
    if fails:
        w("")
        w("!!! THE IMAGE CHECKS FAILED -- BUILD_FAILED.txt is in this folder "
          "and scripts 4 and 5 will refuse it. See section 6. !!!")
    w("")
    w("1. WHAT THIS RUN DID")
    w("-" * 78)
    w(f"Mix '{MIX}' ({', '.join(f'{t} {counts[t]}' for t in MAIN_TYPES)} "
      f"per set), level set '{LEVEL_SET}',")
    w(f"script 1's draw rules (cap {cap.MAX_PER_SCENE_PER_TYPE} per scene "
      f"per type per block, matching '{cap.SCENE_MATCH}', disjoint "
      f"'{cap.DISJOINT}',")
    w(f"saturation-only |d_sat| >= {cap.SAT_ONLY_MIN_DSAT} and |d_con| <= "
      f"{cap.SAT_ONLY_CON_TOL}, contrast-only |d_sat| <= "
      f"{cap.FOG_ONLY_SAT_TOL}).")
    w(f"{N_CANDIDATES} candidate pairs of sets were drawn; the shipped one "
      "was chosen by:")
    for k, d in SELECTION_KEYS:
        w(f"    {d} {k}")
    w(f"Per set: {n_cons} consistency, {CATCH_PER_BLOCK} catch. Practice: "
      f"{PRACTICE_BY_BLOCK[0]} before block 1, {PRACTICE_BY_BLOCK[1]} before "
      "block 2.")
    w(f"Prefer larger differences: {'ON' if PREFER_LARGE_DIFFS else 'off'}.")
    w("")
    w("Inputs and provenance")
    for line in p_log:
        w("  " + line)
    w("")

    w("2. CANDIDATES AND THE SELECTION")
    w("-" * 78)
    w(f"{int(cand.passes.sum())} of {N_CANDIDATES} pass every script 1 "
      f"check. Chosen: candidate {chosen}.")
    w(f"{'cand':>4s} {'pass':>4s} {'rank':>4s}  {'con A':>7s} {'con B':>7s} "
      f"{'weaker':>7s} {'|A-B|':>6s}  {'sat A':>7s} {'sat B':>7s} "
      f"{'weaker':>7s}")
    for r in cand.sort_values(["rank", "candidate"],
                              na_position="last").itertuples():
        if not hasattr(r, "info_con_A") or pd.isna(r.info_con_A):
            w(f"{r.candidate:4d}   no            (not built)")
            continue
        w(f"{r.candidate:4d} {'yes' if r.passes else 'NO':>4s} "
          f"{('' if pd.isna(r.rank) else int(r.rank)):>4}  "
          f"{r.info_con_A:7.2f} {r.info_con_B:7.2f} "
          f"{r.weaker_set_info_con:7.2f} {r.ab_diff_info_con:6.2f}  "
          f"{r.info_sat_A:7.2f} {r.info_sat_B:7.2f} "
          f"{r.weaker_set_info_sat:7.2f}"
          + ("   <- SHIPPED" if r.chosen else "")
          + ("" if r.passes else f"   fails: {r.failed_checks}"))
    ok = cand[cand.passes]
    w(f"Over passing candidates, weaker-set contrast information: min "
      f"{ok.weaker_set_info_con.min():.2f}, median "
      f"{ok.weaker_set_info_con.median():.2f}, max "
      f"{ok.weaker_set_info_con.max():.2f}.")
    w("")

    if comparison is not None:
        w("2b. PREFERRING LARGER DIFFERENCES: WHAT IT CHANGED")
        w("-" * 78)
        co = comparison[comparison.passes]
        for wn, lab in (("con", "contrast"), ("sat", "saturation")):
            w(f"  {lab} information per set, mean over passing candidates: "
              f"weighted {ok[[f'info_{wn}_A', f'info_{wn}_B']].mean().mean():.2f}"
              f", unweighted "
              f"{co[[f'info_{wn}_A', f'info_{wn}_B']].mean().mean():.2f} "
              f"({ok[[f'info_{wn}_A', f'info_{wn}_B']].mean().mean() / co[[f'info_{wn}_A', f'info_{wn}_B']].mean().mean() - 1:+.1%})")
        cu = co
        w(f"  shipped set, weaker-set contrast information: weighted "
          f"{ok.weaker_set_info_con.max():.2f} (candidate {chosen}), "
          f"unweighted {cu.weaker_set_info_con.max():.2f} (candidate "
          f"{spread[1]})")
        w("  The range of each type's driver (|x_sat| for sat_only, |x_con| "
          "otherwise), 10% / median / 90% / max,")
        w("  and that type's driver information per set:")
        sp = spread[0]
        for scope in ("all passing", "shipped"):
            w(f"  [{scope} candidates]")
            for t in MAIN_TYPES:
                parts = []
                for lab in ("unweighted", "weighted"):
                    r = sp[(sp.draw == lab) & (sp.scope == scope)
                           & (sp.type == t)].iloc[0]
                    parts.append(f"{lab:10s} {r.q10:.3f} / {r['median']:.3f} "
                                 f"/ {r.q90:.3f} / {r['max']:.3f}  info "
                                 f"{r.info_driver_per_set:6.2f}")
                w(f"    {t:9s} " + "\n              ".join(parts))
        w("  candidates_unweighted.csv and prefer_spread.csv hold the "
          "unweighted draws and these figures.")
        w("")

    w("3. HOW WELL THE TWO SETS MATCH (the shipped candidate)")
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
      f"{met['ks_con']:.3f}")
    w("  scene x type x bin counts identical in A and B: see the checks")
    w("")
    w("  Saturation-only pairs, measured contrast difference |d_con| "
      "(contrast units):")
    for b in ("A", "B"):
        so = main[b][main[b].trial_type == "sat_only"]
        w(f"    set {b}: min/10/50/90/max {q_str(so.d_fog)}   (rule: <= "
          f"{cap.SAT_ONLY_CON_TOL})")
    w("")
    w("4. WHAT THE PLANNING WEIGHTS MEAN, AND THE THRESHOLDS")
    w("-" * 78)
    for b in ("A", "B"):
        cp, th = info[b]["cp"], info[b]["th"]
        w(f"  set {b}: typical informative |x_sat| {cp['sat'][0]:.3f} "
          f"({cp['sat'][1]} trials) -> P {cp['sat'][2]:.4f};  |x_con| "
          f"{cp['con'][0]:.3f} ({cp['con'][1]} trials) -> P "
          f"{cp['con'][2]:.4f}")
        w(f"         thresholds ({N_THRESHOLD_PEOPLE:,} random choosers): "
          f"saturation {th['upper'][0]:+.3f} / {th['lower'][0]:+.3f}, "
          f"contrast {th['upper'][1]:+.3f} / {th['lower'][1]:+.3f}")
    w(f"  v4 interpretation: {V4_CHOICE_PROB[0]:.4f} / {V4_CHOICE_PROB[1]:.4f}"
      "; v4 thresholds +0.286 / +0.488 (180 trials, one list).")
    w("")
    w("5. CONSISTENCY, CATCH AND PRACTICE")
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
    w("6. IMAGES")
    w("-" * 78)
    if man is not None:
        ds = (man.sat_mean - man.table_sat_mean).abs().max()
        dc = (man.contrast_local - man.table_contrast_local).abs().max()
        w(f"  {len(man)} images copied from {STIMULI_SRC}; largest "
          f"re-measured difference: sat_mean {ds:.2e}, contrast_local "
          f"{dc:.2e}")
    if fails:
        w(f"  FAILURES ({len(fails)}):")
        for f in fails[:40]:
            w(f"    {f}")
        if len(fails) > 40:
            w(f"    ... and {len(fails) - 40} more (BUILD_FAILED.txt)")
    w("")
    w("7. CHECKS")
    w("-" * 78)
    cf = ck.frame()
    for row in cf.itertuples():
        w(f"  {row.status:7s} [{row.scope}] {row.check}"
          + (f"  ({row.detail})" if row.detail else ""))
    w("")
    w(f"  {int((cf.status == 'PASS').sum())} PASS, "
      f"{int((cf.status == 'FAIL').sum())} FAIL, "
      f"{int((cf.status == 'SKIPPED').sum())} SKIPPED")
    w("")
    w("8. NOTES")
    w("-" * 78)
    w("* Nothing here is ordered. Rows are sorted by type and couple; "
      "script 4 makes the orders.")
    w("* item_id is stable within this build: A001..., Ac01... "
      "(consistency), Ax01... (catch),")
    w("  P1_01... and P2_01... (practice before block 1 and 2). repeat_of "
      "names the original's item_id.")
    w("* 'couple' links each analysed trial to its matched partner in the "
      "other set.")
    w("* Thresholds are per set, from random choosers, and go to "
      "design.json via set_info.json.")
    with open(os.path.join(out_dir, "build_report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


# ============================================================
# Main
# ============================================================

def main():
    global _CAP
    t0 = time.time()
    print("v5 step 3: build the two block lists")
    cap = load_module(CAPACITY_SCRIPT, "analyse_pair_capacity_v5")
    v4 = load_module(V4_SIM_SCRIPT, "simulate_power_v4_n10_20")
    _CAP = cap
    for k, v in CAPACITY_OVERRIDES.items():
        setattr(cap, k, v)
    check_settings(cap)
    p_log = [f"script 1   {CAPACITY_SCRIPT}  sha256 "
             f"{sha256_of(CAPACITY_SCRIPT)[:16]}",
             f"v4 sim     {V4_SIM_SCRIPT}  sha256 "
             f"{sha256_of(V4_SIM_SCRIPT)[:16]}",
             f"overrides  {CAPACITY_OVERRIDES or 'none'}",
             f"stimuli    {STIMULI_SRC}"]
    s1_ck = cap.Checks()
    s1_log = []
    p = cap.load_pairs(s1_ck, s1_log)
    cap.check_manifest(p, s1_ck, s1_log)
    p_log += s1_log
    pool = cap.build_pool(p, LEVEL_SET, s1_ck, [])
    pools_t = {t: pool[pool.type == t].reset_index(drop=True)
               for t in cap.TYPES}
    scenes = sorted(p.scene_id.unique())
    bad_in = s1_ck.frame()
    bad_in = bad_in[bad_in.status == "FAIL"]
    if len(bad_in):
        raise RuntimeError("Script 1's input checks fail:\n"
                           + bad_in.to_string())
    p_log.append(f"script 1's input checks: {len(s1_ck.rows)} run, all "
                 "PASS or SKIPPED")

    results = []
    for size in BLOCK_SIZES:
        results.append(build_one(cap, v4, size, pools_t, scenes, p_log, t0))
    any_fail = False
    for out_dir, n_fail, fails in results:
        print(f"  {out_dir}: {n_fail} check(s) FAIL")
        any_fail |= bool(n_fail) or bool(fails)
    print(f"  elapsed {time.time() - t0:.0f} s")
    if any_fail:
        print("\n  BUILD FAILED: see the report(s) above. A folder with "
              "BUILD_FAILED.txt must not be used.")
        sys.exit(1)


if __name__ == "__main__":
    main()
