"""
freeze_trial_lists_v5.py  --  v5 step 5: verify, then freeze

WHAT THIS DOES, per block size in SCRIPT4_DIRS
    1. INPUTS. Refuses a script 4 folder with a failing check, and a
       script 3 folder (named in script 4's settings) with BUILD_FAILED.txt
       or a failing check.
    2. PROVENANCE. The script files on disk must be the ones that built the
       folders (SHA-256 recorded in set_info.json and design.json), and
       their SETTINGS must equal the saved settings.json of each build.
    3. REBUILD. Scripts 3 and 4 are rerun from those settings into
       REBUILD_ROOT, and every output file must be byte-identical to the
       original. Three kinds of file are compared after removing what
       legitimately differs, and the report says so: the two reports (the
       'run finished' and 'output' lines) and settings.json (OUTPUT_ROOT).
    4. CHECKS on the package, written here independently of scripts 3 and
       4, after v4's freeze_trial_list_v4.py and extended to every list:
         v4 check 1  complete: every file the design names exists
         v4 check 2  copies agree: every JSON against its CSV copy
         v4 check 3  the image set closes: every referenced image ships,
                     every shipped image is referenced, the manifest names
                     exactly those files, the preload is exactly the images
                     shown (catch-only images ship but are not preloaded)
         v4 check 4  re-measure: sat_mean and contrast_local measured from
                     the shipped files rebuild every row's d_sat, d_fog,
                     pred_sat and pred_fog, and every file's SHA-256 matches
                     the manifest
         v4 check 5  design invariants: counts, sides, catch and
                     consistency rows, per list
         v4 check 6  the builds' own checks all PASS
       and the new checks: every list is a permutation of its set; no pair
       in both sets; every order meets its constraints (scene gap, type
       spread, repeat distance, catch placement, runs, balance), re-measured
       here, and its realised scene gap equals the one design.json records;
       the breaks (every ceil(main trials / segments) trials, falling on
       the segment boundaries, one catch per stretch, no short final
       stretch, stated in the README); practice; the assignment table
       (slots, lists, episodes, cells, balance at 10/15/20/30 slots, and
       the rule for slots beyond the table: stated in design.json,
       assignment.json and the README, and the balance after n slots equal
       to the balance after n mod slots); catch targets; thresholds in
       design.json equal set_info.json.
    5. FREEZE, only if every check passes: copies the package and the
       builds' records to FROZEN_ROOT/<package>/, writes checksums.sha256
       (every file except checksums.sha256, FROZEN.json and
       freeze_report.txt), verifies it, and writes FROZEN.json and
       freeze_report.txt. If a check fails, nothing is frozen: the report
       goes to the rebuild folder and the script exits with an error.

REUSED
    v4's freeze checks (freeze_trial_list_v4.py: check_complete,
    check_three_copies, check_images_close, check_remeasure, check_design,
    check_settings, write_checksums, verify_checksums, write_record) are
    re-implemented here for a package of several lists; features_of is
    copied verbatim from v4, as v4's freeze script does, so the
    re-measurement is independent of the build.
    build_trials_v5.py and make_trial_orders_v5.py are IMPORTED for the
    rebuild only.

RUN      python freeze_trial_lists_v5.py
"""

# ============================================================
# SETTINGS
# ============================================================

# SCRIPT4_DIRS
#   Script 4's output folder for each block size.
SCRIPT4_DIRS = {
    150: ("data/trial_orders_v5/"
          "v5_n150_orders4_gap30-fixed_break34_slots64_seed42__s7fff70ee"),
    180: ("data/trial_orders_v5/"
          "v5_n180_orders4_gap30-fixed_break40_slots64_seed42__seea0c639"),
}

# SLOT_WRAP_CYCLES
#   The slot rule (slot n uses row ((n - 1) mod slots) + 1) is checked here
#   over this many passes through the table.
SLOT_WRAP_CYCLES = 4

# The scripts that built them (in the project folder)
CAPACITY_SCRIPT = "analyse_pair_capacity_v5.py"
BUILD_SCRIPT = "build_trials_v5.py"
ORDERS_SCRIPT = "make_trial_orders_v5.py"
V4_SIM_SCRIPT = "simulate_power_v4_n10_20.py"

FROZEN_ROOT = "data/frozen_trial_lists_v5"
REBUILD_ROOT = "data/frozen_trial_lists_v5_rebuild"

# REMEASURE_TOL / PRED_TOL
#   Largest |re-measured - stored| for sat_mean and contrast_local (and the
#   differences built from them), and for the predictors.
REMEASURE_TOL = 1e-4
PRED_TOL = 1e-5

FEATURE_TARGET_SIZE = (256, 256)   # v4's measurement convention; keep
SD_SAT = 39.182
SD_FOG = 9.348
FOG_SIGN = -1.0

# Lines removed from the two reports before comparing a rebuild
REPORT_VOLATILE = ("run finished", "output")


# ============================================================
# Imports
# ============================================================

import filecmp
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

ANALYSED = ["sat_only", "fog_only", "together", "conflict"]
UNHASHED = ("checksums.sha256", "FROZEN.json", "freeze_report.txt")


def load_module(path, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found (run from the project "
                                "folder)")
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
    """VERBATIM from v4 (freeze_trial_list_v4.features_of)."""
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


def pair_key(a, b):
    return (a, b) if str(a) <= str(b) else (b, a)


def runs(values):
    best, cur = (1, 1) if len(values) else (0, 0)
    for a, b in zip(values[:-1], values[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return best


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, scope, name, ok, detail=""):
        status = ok if isinstance(ok, str) else ("PASS" if ok else "FAIL")
        self.rows.append({"scope": scope, "check": name, "status": status,
                          "detail": str(detail)})
        return status == "PASS"

    def frame(self):
        return pd.DataFrame(self.rows)


def walk_files(root, skip_dirs=()):
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn
                       if os.path.relpath(os.path.join(dp, d), root)
                       not in skip_dirs)
        for f in sorted(fn):
            out.append(os.path.relpath(os.path.join(dp, f), root))
    return sorted(out)


def new_dir(base):
    if not os.path.exists(base):
        return base
    n = 2
    while os.path.exists(f"{base}_{n}"):
        n += 1
    return f"{base}_{n}"


def norm_json(obj, drop=()):
    if isinstance(obj, dict):
        return {k: norm_json(v, drop) for k, v in obj.items()
                if k not in drop}
    if isinstance(obj, list):
        return [norm_json(v, drop) for v in obj]
    return obj


# ============================================================
# 1-2. Inputs and provenance
# ============================================================

def no_fail(path):
    cf = pd.read_csv(path)
    return cf[cf.status == "FAIL"]


def inputs_and_provenance(size, d4, ck):
    if not os.path.isdir(d4):
        raise FileNotFoundError(f"script 4 folder not found: {d4}")
    s4 = json.load(open(os.path.join(d4, "settings.json")))
    d3 = s4["SCRIPT3_DIR"]
    if os.path.exists(os.path.join(d3, "BUILD_FAILED.txt")):
        raise RuntimeError(f"{d3} contains BUILD_FAILED.txt; it cannot be "
                           "frozen.")
    f3, f4 = (no_fail(os.path.join(d3, "checks.csv")),
              no_fail(os.path.join(d4, "checks.csv")))
    ck.add("v4 check 6: builds", "script 3's checks all PASS (no "
           "BUILD_FAILED.txt)", len(f3) == 0, f"{len(f3)} FAIL")
    ck.add("v4 check 6: builds", "script 4's checks all PASS", len(f4) == 0,
           f"{len(f4)} FAIL")
    info = json.load(open(os.path.join(d3, "set_info.json")))
    design = json.load(open(os.path.join(d4, "package", "design.json")))
    for label, path, want in (
            ("script 1", CAPACITY_SCRIPT, info["script1_sha256"]),
            ("script 3", BUILD_SCRIPT, info["script3_sha256"]),
            ("script 4", ORDERS_SCRIPT,
             design["source"]["script4_sha256"])):
        got = sha256_of(path)
        ck.add("provenance", f"{label} on disk is the one that built the "
               f"folder ({path})", got == want,
               f"recorded {want[:16]}, on disk {got[:16]}")
    ck.add("provenance", "design.json names this script 3 folder and its "
           "set_info.json", design["source"]["script3_dir"] == d3
           and design["source"]["script3_set_info_sha256"]
           == sha256_of(os.path.join(d3, "set_info.json")))
    return d3, s4, info, design


# ============================================================
# 3. Rebuild
# ============================================================

def compare_trees(orig, new, ck, scope, skip_dirs=()):
    """Every file byte-identical, except the volatile report lines and
    settings.json's OUTPUT_ROOT. Returns rows for the comparison table."""
    a = walk_files(orig, skip_dirs)
    b = walk_files(new, skip_dirs)
    rows = []
    ck.add(scope, "the rebuild writes exactly the same files",
           a == b, f"{len(a)} files; only in original "
           f"{sorted(set(a) - set(b))[:5]}, only in rebuild "
           f"{sorted(set(b) - set(a))[:5]}")
    bad = []
    for f in sorted(set(a) & set(b)):
        pa, pb = os.path.join(orig, f), os.path.join(new, f)
        how = "bytes"
        if f.endswith("_report.txt"):
            how = "report minus volatile lines"
            la = [x for x in open(pa).read().splitlines()
                  if not x.startswith(REPORT_VOLATILE)
                  and not x.lstrip().startswith(REPORT_VOLATILE)]
            lb = [x for x in open(pb).read().splitlines()
                  if not x.startswith(REPORT_VOLATILE)
                  and not x.lstrip().startswith(REPORT_VOLATILE)]
            # the elapsed time sits on the 'run finished' line
            same = la == lb
        elif os.path.basename(f) == "settings.json":
            how = "settings minus OUTPUT_ROOT"
            ja, jb = json.load(open(pa)), json.load(open(pb))
            drop = ("OUTPUT_ROOT",)
            same = norm_json(ja, drop) == norm_json(jb, drop)
        else:
            same = filecmp.cmp(pa, pb, shallow=False)
        rows.append({"scope": scope, "file": f, "compared_as": how,
                     "identical": same})
        if not same:
            bad.append(f)
    ck.add(scope, "every file is byte-identical (reports and settings.json "
           "as stated)", not bad,
           f"{len(rows)} compared, {len(bad)} differ {bad[:5]}")
    return rows


def rebuild(size, d3, d4, s4, ck, stage):
    """Rerun scripts 3 and 4 with their saved settings into `stage`."""
    rows = []
    # --- script 3 ---
    b3 = load_module(BUILD_SCRIPT, "build_trials_v5_rebuild")
    cap = b3.load_module(b3.CAPACITY_SCRIPT, "analyse_pair_capacity_v5")
    v4 = b3.load_module(b3.V4_SIM_SCRIPT, "simulate_power_v4_n10_20")
    b3._CAP = cap
    for k, v in b3.CAPACITY_OVERRIDES.items():
        setattr(cap, k, v)
    saved3 = json.load(open(os.path.join(d3, "settings.json")))
    now3 = json.loads(json.dumps(b3.settings_dict(cap, size),
                                 default=str))
    same3 = (norm_json(saved3, ("OUTPUT_ROOT",))
             == norm_json(now3, ("OUTPUT_ROOT",)))
    ck.add("provenance", "script 3's settings (and script 1's, in force) "
           "equal the build's settings.json", same3)
    b3.OUTPUT_ROOT = os.path.join(stage, "script3")
    # prepared exactly as build_trials_v5.main() prepares it, so the
    # report's provenance lines come out the same
    s1_ck = cap.Checks()
    s1_log = []
    p = cap.load_pairs(s1_ck, s1_log)
    cap.check_manifest(p, s1_ck, s1_log)
    pool = cap.build_pool(p, b3.LEVEL_SET, s1_ck, [])
    pools_t = {t: pool[pool.type == t].reset_index(drop=True)
               for t in cap.TYPES}
    scenes = sorted(p.scene_id.unique())
    log3 = [f"script 1   {b3.CAPACITY_SCRIPT}  sha256 "
            f"{sha256_of(b3.CAPACITY_SCRIPT)[:16]}",
            f"v4 sim     {b3.V4_SIM_SCRIPT}  sha256 "
            f"{sha256_of(b3.V4_SIM_SCRIPT)[:16]}",
            f"overrides  {b3.CAPACITY_OVERRIDES or 'none'}",
            f"stimuli    {b3.STIMULI_SRC}"] + s1_log + [
        f"script 1's input checks: {len(s1_ck.rows)} run, all PASS or "
        "SKIPPED"]
    new3, n_fail3, fails3 = b3.build_one(cap, v4, size, pools_t, scenes,
                                         log3, time.time())
    rows += compare_trees(d3, new3, ck, "rebuild: script 3")
    # --- script 4 ---
    b4 = load_module(ORDERS_SCRIPT, "make_trial_orders_v5_rebuild")
    now4 = json.loads(json.dumps(b4.settings_dict(size), default=str))
    same4 = (norm_json(s4, ("OUTPUT_ROOT",))
             == norm_json(now4, ("OUTPUT_ROOT",)))
    ck.add("provenance", "script 4's settings equal the build's "
           "settings.json", same4)
    b4.OUTPUT_ROOT = os.path.join(stage, "script4")
    res = []
    new4, _ = b4.build_size(size, res)
    rows += compare_trees(d4, new4, ck, "rebuild: script 4")
    return rows


# ============================================================
# 4. Package checks
# ============================================================

def load_list(path_json):
    return pd.DataFrame(json.load(open(path_json)))


def same_frame(js, cs):
    """A JSON list of records against its CSV copy: same columns, same
    rows, values equal (None = empty, floats to 1e-12)."""
    if list(js.columns) != list(cs.columns) or len(js) != len(cs):
        return False
    for c in js.columns:
        a, b = js[c], cs[c]
        an, bn = a.isna(), b.isna()
        if not (an == bn).all():
            return False
        a, b = a[~an], b[~bn]
        try:
            if not np.allclose(a.astype(float), b.astype(float), rtol=0,
                               atol=1e-12):
                return False
        except (TypeError, ValueError):
            if not (a.astype(str).str.lower().values
                    == b.astype(str).str.lower().values).all():
                return False
    return True


def package_checks(size, d3, d4, info, design, ck):
    pkg = os.path.join(d4, "package")
    stim = os.path.join(pkg, "stimuli")

    # --- v4 check 1: complete ---
    need = ["design.json", "design.csv", "assignment.json", "assignment.csv",
            "preload.json", "preload.csv", "manifest.csv", "README.txt"]
    need += [v["file"] for v in design["lists"].values()]
    need += [v["file"][:-5] + ".csv" for v in design["lists"].values()]
    need += list(design["practice"].values())
    need += [v[:-5] + ".csv" for v in design["practice"].values()]
    missing = [f for f in need if not os.path.isfile(os.path.join(pkg, f))]
    ck.add("v4 check 1: complete", "every file the design names exists, and "
           "stimuli/", not missing and os.path.isdir(stim),
           f"{len(need)} files; missing {missing}")
    if missing:
        return

    # --- v4 check 2: copies agree ---
    lists = {lid: load_list(os.path.join(pkg, v["file"]))
             for lid, v in design["lists"].items()}
    prac = {k: load_list(os.path.join(pkg, v))
            for k, v in design["practice"].items()}
    bad = []
    for name, df in list(lists.items()) + [(f"practice{k}", v)
                                           for k, v in prac.items()]:
        f = (design["lists"][name]["file"] if name in design["lists"]
             else design["practice"][name[-1]])
        cs = pd.read_csv(os.path.join(pkg, f[:-5] + ".csv"))
        if not same_frame(df, cs):
            bad.append(name)
    asg = json.load(open(os.path.join(pkg, "assignment.json")))
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
    if not same_frame(pd.DataFrame(flat),
                      pd.read_csv(os.path.join(pkg, "assignment.csv"))):
        bad.append("assignment")
    preload = json.load(open(os.path.join(pkg, "preload.json")))
    if list(pd.read_csv(os.path.join(pkg, "preload.csv")).path) != preload:
        bad.append("preload")
    dcsv = pd.read_csv(os.path.join(pkg, "design.csv"),
                       keep_default_na=False)

    def flatten(dd, prefix=""):
        out = []
        if isinstance(dd, dict):
            for k, v in dd.items():
                out += flatten(v, f"{prefix}{k}.")
        elif isinstance(dd, list):
            for i, v in enumerate(dd):
                out += flatten(v, f"{prefix}{i}.")
        else:
            out.append((prefix[:-1], dd))
        return out
    fl = flatten(design)
    if (len(fl) != len(dcsv) or list(dcsv.key) != [k for k, _ in fl]):
        bad.append("design")
    ck.add("v4 check 2: copies agree", "every JSON equals its CSV copy "
           "(lists, practice, assignment, preload, design)", not bad,
           f"differ: {bad}")

    # --- v4 check 3: the image set closes ---
    allrows = pd.concat(list(lists.values()) + list(prac.values()),
                        ignore_index=True)
    referenced = set(allrows.file_a) | set(allrows.file_b)
    shipped = set(os.listdir(stim))
    man = pd.read_csv(os.path.join(pkg, "manifest.csv"))
    shown = allrows[allrows.trial_type != "catch"]
    shown_paths = sorted(set(shown.left_path) | set(shown.right_path))
    catch_only = referenced - (set(shown.file_a) | set(shown.file_b))
    paths_ok = all(
        (r.left_path == "stimuli/" + (r.file_a if r.a_is_left else r.file_b))
        and (r.right_path == "stimuli/"
             + (r.file_b if r.a_is_left else r.file_a))
        for r in allrows.itertuples())
    ck.add("v4 check 3: images close", "every referenced image ships, and "
           "every shipped image is referenced", referenced == shipped,
           f"{len(referenced)} referenced, {len(shipped)} shipped; missing "
           f"{sorted(referenced - shipped)[:3]}, unreferenced "
           f"{sorted(shipped - referenced)[:3]}")
    ck.add("v4 check 3: images close", "manifest.csv names exactly the "
           "shipped files", set(man.filename) == shipped
           and man.filename.is_unique)
    ck.add("v4 check 3: images close", "preload.json is exactly the images "
           "shown; the catch-only images ship but are not preloaded",
           preload == shown_paths,
           f"{len(preload)} preloaded, {len(catch_only)} catch-only")
    ck.add("v4 check 3: images close", "left_path / right_path follow "
           "a_is_left, file_a and file_b on every row", paths_ok)

    # --- v4 check 4: re-measure ---
    feat, bad_hash = {}, []
    mi = man.set_index("filename")
    for f in sorted(shipped):
        path = os.path.join(stim, f)
        feat[f] = features_of(path)
        if sha256_of(path) != mi.loc[f, "sha256"]:
            bad_hash.append(f)
    dman = max(max(abs(feat[f][0] - mi.loc[f, "sat_mean"]),
                   abs(feat[f][1] - mi.loc[f, "contrast_local"]))
               for f in shipped)
    ck.add("v4 check 4: re-measure", "every shipped image's SHA-256 matches "
           "the manifest", not bad_hash, f"{len(bad_hash)} differ")
    ck.add("v4 check 4: re-measure", "re-measured features equal the "
           f"manifest (<= {REMEASURE_TOL:g})", dman <= REMEASURE_TOL,
           f"largest difference {dman:.2e}")
    worst = {"d": 0.0, "pred": 0.0}
    for r in allrows.itertuples():
        sa, fa = feat[r.file_a]
        sb, fb = feat[r.file_b]
        d_sat, d_fog = sa - sb, FOG_SIGN * (fa - fb)
        worst["d"] = max(worst["d"], abs(d_sat - r.d_sat),
                         abs(d_fog - r.d_fog))
        worst["pred"] = max(worst["pred"],
                            abs(-d_sat / SD_SAT - r.pred_sat),
                            abs(d_fog / SD_FOG - r.pred_fog))
    ck.add("v4 check 4: re-measure", "the shipped images rebuild every row's "
           f"d_sat and d_fog (<= {REMEASURE_TOL:g}) and pred_sat and "
           f"pred_fog (<= {PRED_TOL:g})",
           worst["d"] <= REMEASURE_TOL and worst["pred"] <= PRED_TOL,
           f"largest: d {worst['d']:.2e}, pred {worst['pred']:.2e}")

    # --- breaks and segments ---
    cnt = design["counts_per_block"]
    oc = design["order_constraints"]
    tm = design["timing"]
    N0 = design["main_trials_per_block"]
    S = oc["segments"]
    B = tm["break_every"]
    want_edges = list(range(0, N0, B)) + [N0]
    per_seg_rule = str(tm.get("break_rule", "")).startswith("ceil")
    ck.add("breaks", f"break_every = ceil({N0} / {S}) = {-(-N0 // S)}"
           if per_seg_rule else f"break_every fixed at {B}",
           (B == -(-N0 // S)) if per_seg_rule else B > 0,
           f"design.json: {B} ({tm.get('break_rule')})")
    ck.add("breaks", "the breaks fall on the segment boundaries: segment "
           "edges = 0, B, 2B, ..., N, and breaks_after_trial and "
           "stretch_lengths agree", oc.get("segment_edges") == want_edges
           and len(want_edges) - 1 == S
           and tm.get("breaks_after_trial") == want_edges[1:-1]
           and tm.get("stretch_lengths") == np.diff(want_edges).tolist(),
           f"edges {oc.get('segment_edges')}, breaks "
           f"{tm.get('breaks_after_trial')}")
    ck.add("breaks", f"no short final stretch: every stretch >= {B} - "
           f"{S - 1}", min(np.diff(want_edges)) >= B - S + 1,
           f"stretches {np.diff(want_edges).tolist()}")
    readme = open(os.path.join(pkg, "README.txt")).read()
    rd = " ".join(readme.split())
    ck.add("breaks", "the README states the break interval and positions",
           f"A break every {B} main trials" in rd
           and ", ".join(map(str, want_edges[1:-1])) in rd)

    # --- v4 check 5 and the new checks: lists ---
    sets3 = {b: pd.read_csv(os.path.join(d3, f"set_{b}.csv"),
                            keep_default_na=False, na_values=[""])
             for b in ("A", "B")}
    for lid, df in lists.items():
        st = design["lists"][lid]["set"]
        N = len(df)
        tc = df.trial_type.value_counts().to_dict()
        ok_counts = (all(tc.get(t, 0) == cnt[t] for t in ANALYSED)
                     and tc.get("consistency", 0) == cnt["consistency"]
                     and tc.get("catch", 0) == cnt["catch"]
                     and int(df.analysed.sum()) == size
                     and list(df.trial) == list(range(1, N + 1))
                     and N == design["main_trials_per_block"])
        ck.add(f"v4 check 5: {lid}", "counts, trial numbers 1..N, analysed "
               "flags", ok_counts, ", ".join(f"{k} {v}"
                                             for k, v in sorted(tc.items())))
        ck.add(f"v4 check 5: {lid}", "the list is a permutation of its "
               f"set's items (set {st})",
               sorted(df.item_id) == sorted(sets3[st].item_id)
               and (df.set == st).all())
        cat = df[df.trial_type == "catch"]
        ck.add(f"v4 check 5: {lid}", "catch rows: no_images, not analysed, "
               "no repeat", (cat.catch_mode == "no_images").all()
               and (~cat.analysed.astype(bool)).all()
               and cat.repeat_of.isna().all()
               and df[df.trial_type != "catch"].catch_mode.isna().all())
        con = df[df.trial_type == "consistency"]
        by_trial = df.set_index("trial")
        ok_rep, dmin = True, N
        for r in con.itertuples():
            o = by_trial.loc[int(r.repeat_of)]
            same_pair = (pair_key(o.file_a, o.file_b)
                         == pair_key(r.file_a, r.file_b))
            ok_rep &= (same_pair and bool(o.a_is_left) != bool(r.a_is_left)
                       and int(r.repeat_of) < r.trial
                       and o.item_id == r.repeat_of_item
                       and bool(o.analysed))
            dmin = min(dmin, r.trial - int(r.repeat_of))
        ck.add(f"order {lid}", "every repeat follows its original (same "
               f"pair, sides swapped), >= {oc['consistency_min_distance']} "
               "apart", ok_rep and dmin >= oc["consistency_min_distance"],
               f"smallest distance {dmin}")
        img = df[df.trial_type != "catch"]
        gmin = N
        for _, g in img.groupby("scene_id"):
            t = np.sort(g.trial.to_numpy())
            if len(t) > 1:
                gmin = min(gmin, int(np.diff(t).min()))
        ck.add(f"order {lid}", f"scene gap >= {oc['scene_gap_used']}, and "
               "the realised gap equals design.json's scene_gap_min",
               gmin >= oc["scene_gap_used"]
               and gmin == design["lists"][lid].get("scene_gap_min"),
               f"smallest {gmin}, recorded "
               f"{design['lists'][lid].get('scene_gap_min')}")
        edges = np.array(want_edges)
        seg = np.searchsorted(edges, df.trial - 1, side="right") - 1
        per = [int(((seg == k) & (df.trial_type == "catch")).sum())
               for k in range(S)]
        cpos = cat.trial.to_numpy()
        ck.add(f"order {lid}", f"one catch per segment, none in the first or "
               f"last {oc['catch_edge']}", per == [1] * S
               and cpos.min() > oc["catch_edge"]
               and N - cpos.max() + 1 > oc["catch_edge"],
               f"per segment {per}, at {sorted(cpos.tolist())}")
        n_cons = len(con)
        D = oc["consistency_min_distance"]
        exp_c, slots = [], []
        for k in range(S):
            span = max(0, edges[k + 1] - max(edges[k], D))
            ec = n_cons * span / max(N - D, 1)
            exp_c.append(ec)
            slots.append(edges[k + 1] - edges[k] - 1 - ec)
        worst_sp = 0.0
        for t in ANALYSED + ["consistency"]:
            m = (df.trial_type == t).to_numpy()
            for k in range(S):
                got = int((m & (seg == k)).sum())
                exp = (exp_c[k] if t == "consistency"
                       else m.sum() * slots[k] / sum(slots))
                worst_sp = max(worst_sp, abs(got - exp))
        ck.add(f"order {lid}", f"type spread within "
               f"{oc['type_spread_tol']} per segment",
               worst_sp <= oc["type_spread_tol"] + 1e-9,
               f"worst {worst_sp:.2f}")
        ck.add(f"order {lid}", f"type runs <= {oc['max_type_run']}, side runs "
               f"<= {oc['max_side_run']}",
               runs(list(df.trial_type)) <= oc["max_type_run"]
               and runs(list(img.a_is_left.astype(bool)))
               <= oc["max_side_run"],
               f"type {runs(list(df.trial_type))}, side "
               f"{runs(list(img.a_is_left.astype(bool)))}")
        side_bad = []
        for t in ANALYSED:
            mt = df[df.trial_type == t]
            nl = int(mt.a_is_left.astype(bool).sum())
            if nl not in (len(mt) // 2, (len(mt) + 1) // 2):
                side_bad.append(t)
        ck.add(f"v4 check 5: {lid}", "image A on the left in half of each "
               "analysed type (+-1 if odd)", not side_bad, side_bad)
    pairs = {st: set(pair_key(a, b) for a, b in
                     zip(sets3[st].file_a, sets3[st].file_b))
             for st in ("A", "B")}
    ck.add("sets", "no pair in both sets", not (pairs["A"] & pairs["B"]),
           f"{len(pairs['A'] & pairs['B'])} shared")
    for st in ("A", "B"):
        ids = [sorted(df.item_id) for lid, df in lists.items()
               if design["lists"][lid]["set"] == st]
        ck.add("sets", f"every order of set {st} holds the same items",
               all(x == ids[0] for x in ids), f"{len(ids)} orders")

    # practice
    used = pairs["A"] | pairs["B"]
    pp = set()
    for k, df in prac.items():
        n_want = design["practice_by_block"][int(k) - 1]
        ok = (len(df) == n_want and df.scene_id.is_unique
              and (df.trial_type == "practice").all()
              and (~df.analysed.astype(bool)).all())
        keys = set(pair_key(a, b) for a, b in zip(df.file_a, df.file_b))
        ck.add(f"practice block {k}", f"{n_want} trials, distinct scenes, "
               "no pair from either set or the other practice block",
               ok and not (keys & used) and not (keys & pp),
               ", ".join(df.pair_type))
        pp |= keys

    # assignment
    groups = design["groups"]
    n_slots = asg["slots_per_group"]
    for grp, lst in asg["groups"].items():
        eps = groups[grp]["episodes"]
        ok = [r["slot"] for r in lst] == list(range(1, n_slots + 1))
        for r in lst:
            b = r["blocks"]
            ok &= len(b) == 2 and [x["position"] for x in b] == [1, 2]
            ok &= sorted(x["episode"] for x in b) == sorted(eps)
            ok &= {x["set"] for x in b} == {"A", "B"}
            ok &= all(x["list"] in design["lists"]
                      and x["list"] == f"{x['set']}_o{r['order']}"
                      for x in b)
            first = eps[0] if r["block_order"] == 0 else eps[1]
            ok &= b[0]["episode"] == first
            set_first_ep = [x["set"] for x in b if x["episode"] == eps[0]][0]
            ok &= set_first_ep == ("A" if r["set_mapping"] == 0 else "B")
        ck.add("assignment", f"{grp}: slots 1..{n_slots}; each slot has both "
               "episodes, both sets, existing lists, the same order index, "
               "and agrees with its block order and set mapping", bool(ok))
    rows = [(r["block_order"], r["set_mapping"], r["order"],
             r["grid_set_code"]) for r in asg["groups"]["participant"]]
    rows_c = [(r["block_order"], r["set_mapping"], r["order"],
               r["grid_set_code"]) for r in asg["groups"]["control"]]
    ck.add("assignment", "both groups use the same table", rows == rows_c)
    vc = pd.Series(rows).value_counts()
    n_orders = len({r[2] for r in rows})
    n_cells = 2 * 2 * n_orders * 2
    ck.add("assignment", f"all {n_cells} cells appear equally often in "
           f"{n_slots} slots", len(vc) == n_cells
           and vc.nunique() == 1, f"{len(vc)} cells, counts "
           f"{sorted(set(vc))}")
    for P in (10, 15, 20, 30):
        arr = np.array(rows[:P])
        dev = []
        for i, lv in enumerate(([0, 1], [0, 1],
                                list(range(1, n_orders + 1)), [0, 1])):
            dev.append(max(abs((arr[:, i] == v).sum() - P / len(lv))
                           for v in lv))
        pair = max(abs(((arr[:, 0] == a) & (arr[:, 1] == b)).sum() - P / 4)
                   for a in (0, 1) for b in (0, 1))
        ck.add("assignment", f"first {P} slots: every factor and block order "
               "x set mapping within 1 of an even split",
               max(dev) <= 1 and pair <= 1,
               f"factors {dev}, block order x set mapping {pair}")

    # slots beyond the table
    rule = (f"slot n uses row ((n - 1) mod {n_slots}) + 1")
    ck.add("assignment", f"design.json, assignment.json and the README state "
           f"that {rule}",
           str(design["assignment"].get("slot_rule", "")).startswith(rule)
           and str(asg.get("slot_rule", "")).startswith(rule)
           and rule.lower() in rd.lower())
    W = SLOT_WRAP_CYCLES * n_slots
    wrapped = [rows[(n - 1) % n_slots] for n in range(1, W + 1)]
    levels = [[0, 1], [0, 1], list(range(1, n_orders + 1)), [0, 1]]

    def cell_dev(P):
        """Deviation of every factor level and every pair of factor levels
        from an even split after the first P slots."""
        arr = np.array(wrapped[:P])
        out = []
        for i in range(4):
            out += [(arr[:, i] == v).sum() - P / len(levels[i])
                    for v in levels[i]]
            for j in range(i + 1, 4):
                out += [((arr[:, i] == a) & (arr[:, j] == b)).sum()
                        - P / (len(levels[i]) * len(levels[j]))
                        for a in levels[i] for b in levels[j]]
        return np.array(out)
    devs = [cell_dev(P) for P in range(1, W + 1)]
    full = all(np.abs(devs[P - 1]).max() < 1e-9
               for P in range(n_slots, W + 1, n_slots))
    same = all(np.abs(devs[P - 1] - devs[P % n_slots - 1]).max() < 1e-9
               for P in range(n_slots + 1, W + 1) if P % n_slots)
    worst = max(np.abs(d).max() for d in devs)
    worst64 = max(np.abs(d).max() for d in devs[:n_slots])
    ck.add("assignment", f"with that rule, up to {W} slots: exact balance "
           f"after every {n_slots}, and the balance after n slots equals "
           f"the balance after n mod {n_slots} (every factor and pair of "
           "factors)", full and same and abs(worst - worst64) < 1e-9,
           f"largest deviation over all prefixes {worst:g}, within the "
           f"first {n_slots} {worst64:g}")

    # catch targets
    seqs = design["catch"]["targets_by_block_position"]
    ok = True
    for k in ("1", "2"):
        sq = seqs[k]
        ok &= len(sq) == cnt["catch"]
        ok &= set(sq) <= {"ArrowUp", "ArrowDown"}
        ok &= runs(sq) <= 2
        ok &= abs(sq.count("ArrowUp") - sq.count("ArrowDown")) <= 1
    ck.add("catch", "targets per block position: one per catch row, Up/Down "
           "only, balanced within 1, at most 2 identical in a row",
           bool(ok), json.dumps(seqs))

    # thresholds
    ck.add("thresholds", "design.json's per-set thresholds equal "
           "set_info.json", design["thresholds_by_set"]
           == info["thresholds"])


# ============================================================
# 5. Freeze
# ============================================================

def write_checksums(root):
    lines = []
    for f in walk_files(root):
        if f in UNHASHED:
            continue
        lines.append(f"{sha256_of(os.path.join(root, f))}  {f}")
    with open(os.path.join(root, "checksums.sha256"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(lines)


def verify_checksums(root, ck):
    want = {}
    for line in open(os.path.join(root, "checksums.sha256")):
        h, f = line.rstrip("\n").split("  ", 1)
        want[f] = h
    files = [f for f in walk_files(root) if f not in UNHASHED]
    bad = [f for f in files if want.get(f) != sha256_of(os.path.join(root,
                                                                      f))]
    ck.add("freeze", "every frozen file matches checksums.sha256, and every "
           "file is listed", not bad and set(files) == set(want),
           f"{len(files)} files, {len(bad)} differ")


def write_report(path, size, ck, d3, d4, frozen, t0, cmp_rows):
    L = []
    w = L.append
    w("=" * 78)
    w(f"v5 STEP 5: FREEZE, {size} PER BLOCK")
    w("freeze_trial_lists_v5.py")
    w("=" * 78)
    w(f"run finished  {time.strftime('%Y-%m-%d %H:%M:%S')}   elapsed "
      f"{time.time() - t0:.0f} s")
    w(f"script 3      {d3}")
    w(f"script 4      {d4}")
    w(f"frozen        {frozen or 'NOT FROZEN (a check failed)'}")
    w("")
    cf = ck.frame()
    n_f = int((cf.status == "FAIL").sum())
    w(f"RESULT: {int((cf.status == 'PASS').sum())} PASS, {n_f} FAIL, "
      f"{int((cf.status == 'SKIPPED').sum())} SKIPPED"
      + ("" if n_f else " -- frozen"))
    w("")
    w("REBUILD")
    w("-" * 78)
    t = pd.DataFrame(cmp_rows)
    if len(t):
        for (scope, how), g in t.groupby(["scope", "compared_as"]):
            w(f"  {scope:20s} {how:28s} {int(g.identical.sum())} of "
              f"{len(g)} identical")
    w("")
    w("CHECKS")
    w("-" * 78)
    for r in cf.itertuples():
        w(f"  {r.status:7s} [{r.scope}] {r.check}"
          + (f"  ({r.detail})" if r.detail else ""))
    w("")
    w("Reports are compared without their 'run finished' and 'output' "
      "lines, and settings.json")
    w("without OUTPUT_ROOT: those record when and where a build ran, not "
      "what it built.")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return n_f


def freeze_one(size, d4):
    t0 = time.time()
    ck = Checks()
    print(f"\n  n={size}: {d4}")
    d3, s4, info, design = inputs_and_provenance(size, d4, ck)
    name = design["package"]
    stage = new_dir(os.path.join(REBUILD_ROOT, name))
    os.makedirs(stage)
    print(f"    rebuilding into {stage}")
    cmp_rows = rebuild(size, d3, d4, s4, ck, stage)
    print("    checking the package")
    package_checks(size, d3, d4, info, design, ck)
    pd.DataFrame(cmp_rows).to_csv(os.path.join(stage,
                                               "rebuild_comparison.csv"),
                                  index=False)
    n_fail = int((ck.frame().status == "FAIL").sum())
    if n_fail:
        ck.frame().to_csv(os.path.join(stage, "checks.csv"), index=False)
        write_report(os.path.join(stage, "freeze_report.txt"), size, ck, d3,
                     d4, None, t0, cmp_rows)
        print(f"    {n_fail} check(s) FAIL: NOT frozen. Report: "
              f"{os.path.join(stage, 'freeze_report.txt')}")
        return None, n_fail
    frozen = new_dir(os.path.join(FROZEN_ROOT, name))
    shutil.copytree(os.path.join(d4, "package"),
                    os.path.join(frozen, "package"))
    prov = os.path.join(frozen, "provenance")
    shutil.copytree(d3, os.path.join(prov, "script3"),
                    ignore=shutil.ignore_patterns("stimuli"))
    shutil.copytree(d4, os.path.join(prov, "script4"),
                    ignore=shutil.ignore_patterns("package"))
    shutil.copy2(os.path.join(stage, "rebuild_comparison.csv"),
                 os.path.join(prov, "rebuild_comparison.csv"))
    ck.frame().to_csv(os.path.join(prov, "freeze_checks.csv"), index=False)
    n_files = write_checksums(frozen)
    verify_checksums(frozen, ck)
    ck.frame().to_csv(os.path.join(prov, "freeze_checks.csv"), index=False)
    # the checks file changed after hashing; rehash and verify once more
    n_files = write_checksums(frozen)
    ck2 = Checks()
    verify_checksums(frozen, ck2)
    if ck2.rows[-1]["status"] != "PASS":
        raise RuntimeError("checksums do not verify after the final write")
    record = {
        "frozen": time.strftime("%Y-%m-%d %H:%M:%S"),
        "package": name, "block_size": size,
        "files_hashed": n_files,
        "checksums_sha256": sha256_of(os.path.join(frozen,
                                                   "checksums.sha256")),
        "design_json_sha256": sha256_of(os.path.join(frozen, "package",
                                                     "design.json")),
        "assignment_json_sha256": sha256_of(os.path.join(
            frozen, "package", "assignment.json")),
        "script3_dir": d3, "script4_dir": d4,
        "scripts_sha256": {p: sha256_of(p) for p in
                           (CAPACITY_SCRIPT, BUILD_SCRIPT, ORDERS_SCRIPT,
                            os.path.abspath(__file__))},
        "checks": {"pass": int((ck.frame().status == "PASS").sum()),
                   "fail": 0},
        "unhashed": list(UNHASHED),
        "use": f"copy package/ to the experiment as data/sets/{name}/; "
               "verify checksums.sha256 before any analysis",
    }
    with open(os.path.join(frozen, "FROZEN.json"), "w") as f:
        json.dump(record, f, indent=2)
    write_report(os.path.join(frozen, "freeze_report.txt"), size, ck, d3, d4,
                 frozen, t0, cmp_rows)
    print(f"    frozen: {frozen} ({n_files} files hashed)")
    return frozen, 0


def main():
    t0 = time.time()
    print("v5 step 5: verify and freeze")
    out = []
    for size, d4 in SCRIPT4_DIRS.items():
        out.append(freeze_one(size, d4))
    print(f"  elapsed {time.time() - t0:.0f} s")
    if any(n for _, n in out):
        sys.exit(1)


if __name__ == "__main__":
    main()
