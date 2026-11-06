"""
make_trial_orders_v5.py  --  v5 step 4: trial orders, assignment table, package

WHAT THIS DOES
    For each script 3 folder in SCRIPT3_DIRS (one per block size):

    1. REFUSES a folder that holds BUILD_FAILED.txt, or whose checks.csv has
       a FAIL, or whose set_info.json does not match its own set files.
    2. ORDERS. N_ORDERS orders of each set's block (analysed + consistency +
       catch trials), built by a seeded constructive search:
         * catch trials: one in each of SEGMENTS equal parts of the block,
           never in the first or last CATCH_EDGE trials;
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
       with which episode), the order index (the same in both blocks) and
       the grid set code. The four factors are fully crossed (2 x 2 x 4 x 2
       = 32 cells; every cell twice in 64 slots), and the sequence is built
       to keep every prefix as balanced as possible: the report gives the
       balance at the first 10, 15, 20, 30 and all slots. Both groups use
       the same table, so they are balanced alike. Deterministic.
    4. PACKAGE, in the format of the plan: design.json, assignment.json,
       lists/, practice/, preload.json, stimuli/, manifest.csv, README.txt,
       and a CSV copy of every JSON.

REUSED
    Nothing is imported from v4. Script 3's outputs are read as files.
    The list row format is v4's trial_list.json format (the experiment's
    REQUIRED_FIELDS), with extra fields for the two-block design.

INPUTS   SCRIPT3_DIRS (script 3's output folders, one per size)
OUTPUTS  one folder per size, OUTPUT_ROOT/<settings>__s<hash>/
    package/                 the experiment package (copy to the experiment
                             as data/sets/<PACKAGE_NAME>/)
    orders_report.txt        what was done, what was found, every check
    checks.csv               every check, PASS or FAIL
    orders_summary.csv       one row per order: balance and constraints
    order_candidates.csv     every valid candidate order's scores
    assignment_balance.csv   balance of the assignment at each prefix
    settings.json            every setting this run used

RUN      python make_trial_orders_v5.py
"""

# ============================================================
# SETTINGS
# ============================================================

# SCRIPT3_DIRS
#   Script 3's output folder for each block size.
SCRIPT3_DIRS = {
    150: ("data/trials_v5/v5_n150_conflict_heavy_without_p65_cand50"
          "_prefer-off_seed42__s6fe003f5"),
    180: ("data/trials_v5/v5_n180_conflict_heavy_without_p65_cand50"
          "_prefer-off_seed42__see842861"),
}

# PACKAGE_NAME
#   The package folder's name; the experiment loads it as
#   data/sets/<name>/. {size} is replaced by the block size.
PACKAGE_NAME = "v5_{size}"

# --- Orders ---
N_ORDERS = 4                 # kept per set
N_ORDER_CANDIDATES = 40      # valid orders built per set before choosing
MAX_ATTEMPTS = 20_000        # restarts allowed per set to reach them

# SEGMENTS
#   The block is divided into this many equal parts for the type-spread
#   check, and holds one catch trial in each (so it must equal the number of
#   catch trials per set).
SEGMENTS = 5

# TYPE_SPREAD_TOL
#   In every segment, each type's count must be within this of its expected
#   count (consistency is checked only from CONSISTENCY_MIN_DISTANCE on).
TYPE_SPREAD_TOL = 1

# SCENE_GAP / SCENE_GAP_SEARCH_MAX / SCENE_GAP_POLICY
#   Two trials of the same scene must be at least SCENE_GAP trials apart
#   (positions i < j with j - i >= SCENE_GAP; 5 means at least four other
#   trials between them). Catch trials show no images and do not count.
#   The largest gap at which every order of both sets can still be built is
#   searched up to SCENE_GAP_SEARCH_MAX and reported.
#     "min"      build with SCENE_GAP (the default)
#     "largest"  build with the largest achievable gap
SCENE_GAP = 5
SCENE_GAP_SEARCH_MAX = 40
SCENE_GAP_POLICY = "min"
GAP_SEARCH_ATTEMPTS = 2_000  # restarts per set per gap in the search

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

# --- Session settings recorded in design.json ---
BREAK_EVERY = 41
REST_BETWEEN_BLOCKS = True
CATCH_KEYS = ("ArrowUp", "ArrowDown")
CATCH_MAX_RUN = 2            # same target at most twice in a row (v4 rule)

OUTPUT_ROOT = "data/trial_orders_v5"
ON_EXISTING = "new"
RANDOM_SEED = 42


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
               "pair_type", "couple", "file_a", "file_b"]


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
                         "LIST_FIELDS")}
    own["BLOCK_SIZE"] = size
    own["SCRIPT3_DIR"] = SCRIPT3_DIRS[size]
    return own


def resolve_output_dir(size):
    blob = json.dumps(settings_dict(size), sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode()).hexdigest()[:8]
    name = (f"v5_n{size}_orders{N_ORDERS}_gap{SCENE_GAP}-{SCENE_GAP_POLICY}"
            f"_slots{N_SLOTS}_seed{RANDOM_SEED}__s{h}")
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
# Reading script 3
# ============================================================

def read_script3(d, size, ck):
    if not os.path.isdir(d):
        raise FileNotFoundError(f"script 3 folder not found: {d}")
    if os.path.exists(os.path.join(d, "BUILD_FAILED.txt")):
        raise RuntimeError(
            f"{d} contains BUILD_FAILED.txt: script 3's image checks failed "
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
    ok = info["block_size"] == size
    cnt = info["counts_per_set"]
    for b in ("A", "B"):
        s = sets[b]
        ok &= int((s.trial_type.isin(ANALYSED)).sum()) == size
        ok &= int((s.trial_type == "consistency").sum()) == cnt["consistency"]
        ok &= int((s.trial_type == "catch").sum()) == cnt["catch"]
    ck.add("script 3", "set files agree with set_info.json (size and counts)",
           bool(ok), f"{size} analysed, {cnt['consistency']} consistency, "
           f"{cnt['catch']} catch per set")
    return sets, prac, info, man


# ============================================================
# One order
# ============================================================

def catch_positions(N, n_catch, rng):
    edges = np.round(np.linspace(0, N, SEGMENTS + 1)).astype(int)
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
    # type spread per segment
    edges = np.round(np.linspace(0, N, SEGMENTS + 1)).astype(int)
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
        (f"one catch per segment, none in the first or last {CATCH_EDGE}",
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

FACTORS = ["block_order", "set_mapping", "order", "grid_set_code"]
_ASSIGN_ROWS = None


def cells():
    return [(bo, sm, k, g) for bo in (0, 1) for sm in (0, 1)
            for k in range(1, N_ORDERS + 1) for g in (0, 1)]


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
    through the 32 cells in cycles (every cell once per cycle) and taking,
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


def assignment_json(rows, sets_present):
    out = {"slots_per_group": N_SLOTS, "slot_param": SLOT_PARAM,
           "factors": {
               "block_order": "0 = the group's first-listed episode first, "
                              "1 = second-listed first",
               "set_mapping": "0 = set A with the first-listed episode, "
                              "1 = set B with it",
               "order": "order index, the same in both blocks",
               "grid_set_code": "the grid set swap (0 or 1), as the "
                                "experiment's set_code"},
           "groups": {}}
    for grp, eps in GROUPS.items():
        lst = []
        for s, (bo, sm, k, g) in enumerate(rows, start=1):
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
                        "order": k, "grid_set_code": g, "blocks": blocks})
        out["groups"][grp] = lst
    return out


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


README = """V5 EXPERIMENT PACKAGE  ({name})
====================================================================

Two image-choice blocks per participant, {size} analysed trials each.
Generated by make_trial_orders_v5.py from build_trials_v5.py's sets.

FILES
  design.json       the design: counts, groups and episodes, lists, catch
                    targets per block position, break and rest settings,
                    per-set thresholds, order constraints, provenance
  assignment.json   per group, {slots} slots. The invitation link carries
                    ?{param}=<n>; slots are handed out in sequence per
                    group and never reused. Each slot names, for block
                    positions 1 and 2, the episode, the set and the list,
                    and the grid set code.
  lists/<set>_o<k>.json
                    one block's main trials (analysed, consistency, catch)
                    in presentation order. Row format = v4's
                    trial_list.json, plus list_id, set, order, item_id,
                    repeat_of_item, pair_type, couple, file_a, file_b.
                    repeat_of = the block-local trial number of the
                    original (consistency rows only).
  practice/block1.json, practice/block2.json
                    the practice trials before block 1 and block 2, the
                    same for every participant
  preload.json      every image any list or practice row shows. Catch rows
                    show no images ('no_images' mode), so images used only
                    by catch rows ship in stimuli/ but are not listed.
  stimuli/          the images, byte-identical to the rendered stimulus set
  manifest.csv      every shipped image, its SHA-256 and measured features
  *.csv             a CSV copy of every JSON

PREDICTORS
  pred_sat = -d_sat / 39.182, pred_fog = d_fog / 9.348, always A minus B;
  positive = image A is the predicted choice. a_is_left says which side A
  is on.

CATCH
  {catch} catch rows per block, no images; targets per block position in
  design.json. Exclude a participant with {excl} or more failures over the
  session; record failures per block too.
"""


def build_size(size, ck_global):
    t0 = time.time()
    ck = Checks()
    d3 = SCRIPT3_DIRS[size]
    sets, prac, info, man = read_script3(d3, size, ck)
    out = resolve_output_dir(size)
    os.makedirs(out)
    print(f"\n  n={size}: output {out}")

    # --- the largest achievable scene gap ---
    gap_rows = []
    largest = None
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
        why = gap_tab[gap_tab.gap == SCENE_GAP].to_string()
        raise RuntimeError(f"n={size}: no orders at scene gap {SCENE_GAP}:\n"
                           + why)
    G_use = SCENE_GAP if SCENE_GAP_POLICY == "min" else largest
    print(f"    scene gap: largest achievable {largest}, used {G_use}")

    # --- orders ---
    orders, summary, cand_rows = {}, [], []
    for b in ("A", "B"):
        cands, stats = candidate_orders(sets[b], G_use, N_ORDER_CANDIDATES,
                                        MAX_ATTEMPTS,
                                        rng_for("orders", size, b))
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
    summ = pd.DataFrame(summary)

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
           f"(block order x set mapping x order x grid code) appears "
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
    asg = assignment_json(rows, sets)

    # --- package ---
    name = PACKAGE_NAME.format(size=size)
    pkg = os.path.join(out, "package")
    os.makedirs(os.path.join(pkg, "lists"))
    os.makedirs(os.path.join(pkg, "practice"))
    lists = {}
    for (b, k), df in sorted(orders.items()):
        lid = f"{b}_o{k}"
        lr = list_rows(df, lid, b, k)
        lists[lid] = lr
        write_json_csv(to_json_records(lr),
                       os.path.join(pkg, "lists", f"{lid}.json"), lr)
    prac_rows = {}
    for i in (1, 2):
        p = prac[i].copy()
        p["couple"] = None
        pr = list_rows(p, f"practice_block{i}", "practice", 0)
        prac_rows[i] = pr
        write_json_csv(to_json_records(pr),
                       os.path.join(pkg, "practice", f"block{i}.json"), pr)
    flat = []
    for grp, lst in asg["groups"].items():
        for r in lst:
            flat.append({"group": grp, "slot": r["slot"],
                         "block_order": r["block_order"],
                         "set_mapping": r["set_mapping"],
                         "order": r["order"],
                         "grid_set_code": r["grid_set_code"],
                         **{f"block{b['position']}_{k}": b[k]
                            for b in r["blocks"]
                            for k in ("episode", "set", "list")}})
    write_json_csv(asg, os.path.join(pkg, "assignment.json"),
                   pd.DataFrame(flat))
    shown = set()
    for df in list(lists.values()) + list(prac_rows.values()):
        img = df[df.trial_type != "catch"]
        shown |= set(img.left_path) | set(img.right_path)
    preload = sorted(shown)
    write_json_csv(preload, os.path.join(pkg, "preload.json"),
                   pd.DataFrame({"path": preload}))
    # stimuli: every image any row references (catch rows included)
    files = set()
    for df in list(lists.values()) + list(prac_rows.values()):
        files |= set(df.file_a) | set(df.file_b)
    os.makedirs(os.path.join(pkg, "stimuli"))
    src = os.path.join(d3, "stimuli")
    for f in sorted(files):
        s = os.path.join(src, f)
        if not os.path.isfile(s):
            raise RuntimeError(f"{f} is not in {src}")
        shutil.copy2(s, os.path.join(pkg, "stimuli", f))
    man_pkg = man[man.filename.isin(files)].sort_values("filename")
    man_pkg.to_csv(os.path.join(pkg, "manifest.csv"), index=False)
    bad_hash = [f for f in sorted(files)
                if sha256_of(os.path.join(pkg, "stimuli", f))
                != man.set_index("filename").loc[f, "sha256"]]
    ck.add("package", "every shipped image matches script 3's manifest "
           "(SHA-256)", not bad_hash, f"{len(files)} images, "
           f"{len(bad_hash)} differ")
    ck.add("package", "every preloaded image ships in stimuli/",
           all(p[len("stimuli/"):] in files for p in preload),
           f"{len(preload)} preloaded, {len(files)} shipped")

    rngc = rng_for("catch_targets", size)
    catch_seq = {str(pos): catch_targets(info["counts_per_set"]["catch"],
                                         pos, rngc) for pos in (1, 2)}
    main_n = len(sets["A"])
    design = {
        "version": "v5", "package": name, "block_size": size,
        "main_trials_per_block": main_n,
        "counts_per_block": info["counts_per_set"],
        "practice_by_block": info["practice_by_block"],
        "groups": {g: {"episodes": e} for g, e in GROUPS.items()},
        "lists": {lid: {"set": lid[0], "order": int(lid.split("_o")[1]),
                        "file": f"lists/{lid}.json", "n": len(df)}
                  for lid, df in lists.items()},
        "practice": {"1": "practice/block1.json",
                     "2": "practice/block2.json"},
        "assignment": {"file": "assignment.json", "slot_param": SLOT_PARAM,
                       "slots_per_group": N_SLOTS,
                       "allocation": "in sequence per group, never reused"},
        "timing": {"break_every": BREAK_EVERY,
                   "rest_between_blocks": REST_BETWEEN_BLOCKS},
        "catch": {**info["catch"],
                  "targets_by_block_position": catch_seq},
        "thresholds_by_set": info["thresholds"],
        "choice_probability_at_planning_weights":
            info["choice_probability_at_planning_weights"],
        "predictors": info["predictors"],
        "order_constraints": {
            "segments": SEGMENTS, "type_spread_tol": TYPE_SPREAD_TOL,
            "scene_gap_used": G_use, "scene_gap_largest_achievable": largest,
            "consistency_min_distance": CONSISTENCY_MIN_DISTANCE,
            "catch_edge": CATCH_EDGE, "max_type_run": MAX_TYPE_RUN,
            "max_side_run": MAX_SIDE_RUN},
        "source": {"script3_dir": d3,
                   "script3_set_info_sha256": sha256_of(
                       os.path.join(d3, "set_info.json")),
                   "script3_sha256": info.get("script3_sha256"),
                   "script1_sha256": info.get("script1_sha256"),
                   "script4_sha256": sha256_of(os.path.abspath(__file__)),
                   "chosen_candidate": info.get("chosen_candidate")},
    }
    with open(os.path.join(pkg, "design.json"), "w") as f:
        json.dump(design, f, indent=1)
    pd.DataFrame(flatten(design), columns=["key", "value"]).to_csv(
        os.path.join(pkg, "design.csv"), index=False)
    with open(os.path.join(pkg, "README.txt"), "w") as f:
        f.write(README.format(name=name, size=size, slots=N_SLOTS,
                              param=SLOT_PARAM,
                              catch=info["counts_per_set"]["catch"],
                              excl=info["catch"]["exclude_at_or_above"]))

    # --- outputs ---
    summ.to_csv(os.path.join(out, "orders_summary.csv"), index=False)
    pd.DataFrame(cand_rows).to_csv(os.path.join(out,
                                                "order_candidates.csv"),
                                   index=False)
    bal.to_csv(os.path.join(out, "assignment_balance.csv"), index=False)
    gap_tab.to_csv(os.path.join(out, "scene_gap_search.csv"), index=False)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(settings_dict(size), f, indent=2, default=str)
    write_report(out, size, ck, d3, info, gap_tab, largest, G_use, summ,
                 bal, rows, asg, design, lists, preload, files, t0)
    ck.frame().to_csv(os.path.join(out, "checks.csv"), index=False)
    n_fail = int((ck.frame().status == "FAIL").sum())
    ck_global.append((out, n_fail))
    return out, n_fail


def write_report(out, size, ck, d3, info, gap_tab, largest, G_use, summ, bal,
                 rows, asg, design, lists, preload, files, t0):
    L = []
    w = L.append
    w("=" * 78)
    w(f"v5 STEP 4: TRIAL ORDERS, ASSIGNMENT AND PACKAGE, {size} PER BLOCK")
    w("make_trial_orders_v5.py")
    w("=" * 78)
    w(f"run finished  {time.strftime('%Y-%m-%d %H:%M:%S')}   elapsed "
      f"{time.time() - t0:.0f} s")
    w(f"output        {out}")
    w(f"script 3      {d3}")
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
    w("")
    w("2. SCENE GAP")
    w("-" * 78)
    w(f"Largest gap at which all {N_ORDERS} orders of both sets could be "
      f"built: {largest}")
    w(f"(searched upward from {SCENE_GAP}, {GAP_SEARCH_ATTEMPTS} attempts "
      f"per set per gap). Gap used: {G_use} (policy "
      f"'{SCENE_GAP_POLICY}').")
    for G, g in gap_tab.groupby("gap"):
        w(f"  gap {G:2d}: " + "; ".join(
            f"set {r.set} {r.valid} valid in {r.attempts} attempts"
            for r in g.itertuples()))
    last = gap_tab[gap_tab.gap == gap_tab.gap.max()]
    if (last.valid < N_ORDERS).any():
        w(f"  why the search stopped at gap {gap_tab.gap.max()}: "
          + "; ".join(f"set {r.set}: {r.reasons}" for r in last.itertuples()
                      if r.valid < N_ORDERS))
    w("  (the realised smallest gap of each kept order is in section 3)")
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
    w("4. ASSIGNMENT")
    w("-" * 78)
    w("Groups: " + "; ".join(f"{g} = {' + '.join(e)}"
                             for g, e in GROUPS.items())
      + f". {N_SLOTS} slots per group, the")
    w(f"same table for both groups; the link carries ?{SLOT_PARAM}=<n>.")
    w("Balance: the largest deviation from an even split, per factor and per "
      "pair of factors:")
    cols = [c for c in bal.columns if c != "slots"]
    w(f"  {'slots':>5s} " + " ".join(f"{c[:20]:>20s}" for c in cols[:4]))
    for r in bal.itertuples(index=False):
        w(f"  {r[0]:5d} " + " ".join(f"{v:20g}" for v in r[1:5]))
    w("  pairs of factors:")
    for r in bal.itertuples(index=False):
        d = dict(zip(bal.columns, r))
        w(f"  {d['slots']:5d} " + ", ".join(
            f"{c} {d[c]:g}" for c in cols[4:]))
    w("  First 16 slots (block order, set mapping, order, grid code):")
    w("    " + "  ".join(f"{s + 1}:{''.join(map(str, c))}"
                         for s, c in enumerate(rows[:16])))
    w("  participant slot 1: " + json.dumps(
        asg["groups"]["participant"][0]["blocks"]))
    w("")
    w("5. PACKAGE")
    w("-" * 78)
    w(f"  package/  ->  copy to the experiment as data/sets/"
      f"{design['package']}/")
    w(f"  lists: {', '.join(sorted(lists))}; practice: block1 "
      f"({info['practice_by_block'][0]}), block2 "
      f"({info['practice_by_block'][1]})")
    w(f"  images: {len(files)} shipped, {len(preload)} preloaded "
      "(catch-only images ship but are not preloaded)")
    w(f"  catch targets: block 1 {design['catch']['targets_by_block_position']['1']}, "
      f"block 2 {design['catch']['targets_by_block_position']['2']}")
    nb = (design["main_trials_per_block"] - 1) // BREAK_EVERY
    last_part = design["main_trials_per_block"] - nb * BREAK_EVERY
    w(f"  breaks: every {BREAK_EVERY} main trials -> {nb} per block; the "
      f"last part of each block has {last_part} trials")
    w("")
    w("6. CHECKS")
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


def main():
    t0 = time.time()
    print("v5 step 4: trial orders, assignment and package")
    if SCENE_GAP_POLICY not in ("min", "largest"):
        raise ValueError("SCENE_GAP_POLICY must be 'min' or 'largest'")
    results = []
    for size in SCRIPT3_DIRS:
        build_size(size, results)
    for out, n_fail in results:
        print(f"  {out}: {n_fail} check(s) FAIL")
    print(f"  elapsed {time.time() - t0:.0f} s")
    if any(n for _, n in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
