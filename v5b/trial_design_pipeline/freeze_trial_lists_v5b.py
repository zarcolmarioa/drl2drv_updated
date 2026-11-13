"""
freeze_trial_lists_v5b.py  --  v5b step 5: verify, then freeze

WHAT CHANGED FROM freeze_trial_lists_v5.py (v5 -> v5b)
    The structure (inputs, provenance, rebuild, independent checks, freeze
    only if everything passes) and v4's checks are v5's. Changed:
      * INPUTS: script 4 = make_trial_orders_v5b.py's runs; scripts 1, 3, 4
        are the v5b ones. Provenance compares each script on disk with the
        SHA-256 its build recorded (script 3: the per-type build's
        set_info.json, since build_trials_v5b.py was edited in place); the
        v4 simulation script with the 16 characters script 3's report
        records.
      * REBUILD: scripts 3 and 4 rerun as they were run, each as a separate
        process with one thread per process, in a workspace inside
        REBUILD_ROOT (the scripts copied byte for byte, the inputs under
        data/ linked, the output roots not), so script 4 is rebuilt from the
        REBUILT script 3 folders and the whole chain is reproduced. Each
        command and its log are kept. Nothing may be written outside
        REBUILD_ROOT. The run time and worker count are removed before the
        byte-for-byte comparison (listed in the report).
      * NEW CHECKS (written here): N_corpus and N_orig recomputed from the
        re-measured shipped images (corpus reference and scene originals
        from the stimulus set), the overall and per-type normality limits
        verified in both sets with their margins, and equal to design.json;
        items.csv; the stimulus set's checksums; practice and its feedback;
        no grid (16 cells x 4 = 64 slots); the README's list of experiment
        changes; no absolute path in any output.
      * ARCHIVING: FROZEN.json records the stimulus set's run and the hash
        of its checksums.sha256; a top-level README.txt (hashed) gives the
        use line, the stimulus provenance and the Places365 note.
      * Relative paths everywhere (v5 keyed its own hash by an absolute
        path).

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
    build_trials_v5b.py and make_trial_orders_v5b.py are IMPORTED only to
    compare their settings with the builds'; the rebuild RUNS them.

RUN      python freeze_trial_lists_v5b.py
"""

# ============================================================
# SETTINGS
# ============================================================

# SCRIPT4_DIRS
#   Script 4's output folder for each block size.
SCRIPT4_DIRS = {
    180: ("data/trial_orders_v5b/"
          "v5b_n180_orders4_gap30-fixed_break40_slots64_seed42__s20227fe1"),
    150: ("data/trial_orders_v5b/"
          "v5b_n150_orders4_gap30-fixed_break34_slots64_seed42__s0506d7c8"),
}

# SLOT_WRAP_CYCLES
#   The slot rule (slot n uses row ((n - 1) mod slots) + 1) is checked here
#   over this many passes through the table.
SLOT_WRAP_CYCLES = 4

# The scripts that built them (in the project folder)
CAPACITY_SCRIPT = "analyse_pair_capacity_v5b.py"
BUILD_SCRIPT = "build_trials_v5b.py"
ORDERS_SCRIPT = "make_trial_orders_v5b.py"
V4_SIM_SCRIPT = "simulate_power_v4_n10_20.py"

FROZEN_ROOT = "data/frozen_trial_lists_v5b"
REBUILD_ROOT = "data/frozen_trial_lists_v5b_rebuild"

# The stimulus set the packages were built from
STIMULUS_SET_DIR = "data/stimulus_set_v5/7fd72262ba"
STIMULUS_SET_RUN = "stimulus_set_v5/7fd72262ba"

# REFERENCE_ROUNDED
#   (median, SD) of sat_mean and contrast_local that the stimulus set's
#   corpus reference (reference_stimulus_path.csv: p50, sd) must round to
#   at 3 decimals. The exact values are used.
REFERENCE_ROUNDED = {"sat_mean": (66.448, 39.182),
                     "contrast_local": (26.909, 9.348)}

# The agreed normality limits (analysed trials, each set, both
# definitions); design.json must record exactly these.
OVERALL_LIMIT = 0.05
TYPE_LIMITS = {"sat_only": ("sat", 0.05), "fog_only": ("con", 0.05),
               "conflict": ("con", 0.10)}
NORMALITY_DEFS = ("corpus", "orig")

# PRACTICE_EXPECTED: (rows, feedback) before block 1 and before block 2
PRACTICE_EXPECTED = ((6, True), (3, False))

# REMEASURE_TOL / PRED_TOL
#   Largest |re-measured - stored| for sat_mean and contrast_local (and the
#   differences built from them), and for the predictors.
REMEASURE_TOL = 1e-4
PRED_TOL = 1e-5
# NORM_TOL: recomputed N against items.csv and the lists, and recomputed
# b and cos against design.json
NORM_TOL = 1e-4

FEATURE_TARGET_SIZE = (256, 256)   # v4's measurement convention; keep
SD_SAT = 39.182
SD_FOG = 9.348
FOG_SIGN = -1.0

# --- The rebuild ---
# One thread per process for the rebuilt scripts (as script 3 ran)
REBUILD_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                       "MKL_NUM_THREADS")

# What legitimately differs between two runs (run time, worker count),
# removed before comparing; nothing else is:
#   report lines starting with REPORT_VOLATILE; REPORT_SUBS in the reports
#   and in checks.csv; the CSV_VOLATILE_COLS columns; the
#   JSON_VOLATILE_KEYS keys (at any depth).
REPORT_VOLATILE = ("run finished", "workers", "Repair time per candidate")
REPORT_SUBS = [(r"random choosers, \d+ s\)", "random choosers, _ s)"),
               (r"\d+ worker\(s\) in the run", "_ worker(s) in the run")]
CSV_VOLATILE_COLS = {"candidates.csv": ("seconds", "repair_seconds")}

# Script 3's limit-step comparison (limit_steps.csv and report section 5b)
# repairs every candidate under the lighter steps too, for the cost table
# only. A solve there that reaches script 3's time limit can end
# differently on a busier computer, so the AVERAGES over those candidates
# may differ between runs; they never ship. In limit_steps.csv these
# columns are compared loosely (largest difference reported); every other
# column (each step's chosen candidate, number passing, the chosen sets'
# information, leak and positions, and meets_cost_rule) must be identical.
# Report section 5b holds the same numbers and is left out of the line
# comparison.
STEPS_FILE = "limit_steps.csv"
STEPS_LOOSE_PREFIXES = ("mean_", "paired_")
STEPS_LOOSE_COLS = ("largest_cost_vs_prev",)
REPORT_SKIP_SECTIONS = [("5b. THE LIMIT STEPS", "6. ")]
JSON_VOLATILE_KEYS = {"settings.json": ("workers",),
                      "set_info.json": ("workers",)}

# WATCH_TREES
#   Checked file by file for writes during the rebuild (with the project's
#   and data/'s top level): nothing outside REBUILD_ROOT may change.
WATCH_TREES = ["data/trials_v5b", "data/trial_orders_v5b",
               "data/pair_capacity_v5b", "data/block_length_v5b",
               STIMULUS_SET_DIR, FROZEN_ROOT]

# Absolute paths: no output text file may contain the project folder's
# path or a match of these patterns.
ABS_PATH_PATTERNS = [r"(?<![\w.~])/(home|Users|tmp|mnt|root|var|opt|private"
                     r"|media|srv)/", r"(?<![A-Za-z])[A-Za-z]:\\"]
TEXT_EXT = (".json", ".csv", ".txt", ".md", ".sha256", ".cff", ".log")


# ============================================================
# Imports
# ============================================================

import filecmp
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True   # no __pycache__ in the project folder

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
    if not os.path.isdir(d3):
        raise FileNotFoundError(f"script 3 folder not found: {d3}")
    f3, f4 = (no_fail(os.path.join(d3, "checks.csv")),
              no_fail(os.path.join(d4, "checks.csv")))
    ck.add("v4 check 6: builds", "script 3's checks all PASS (no "
           "BUILD_FAILED.txt)", len(f3) == 0, f"{len(f3)} FAIL")
    ck.add("v4 check 6: builds", "script 4's checks all PASS", len(f4) == 0,
           f"{len(f4)} FAIL")
    info = json.load(open(os.path.join(d3, "set_info.json")))
    design = json.load(open(os.path.join(d4, "package", "design.json")))
    for label, path, want in (
            ("script 1", CAPACITY_SCRIPT, info["script1"]["sha256"]),
            ("script 3 (the per-type build that made this folder)",
             BUILD_SCRIPT, info["script3"]["sha256"]),
            ("script 4", ORDERS_SCRIPT,
             design["source"]["script4_sha256"])):
        got = sha256_of(path)
        ck.add("provenance", f"{label} on disk is the one that built the "
               f"folder ({path})", got == want,
               f"recorded {want[:16]}, on disk {got[:16]}")
    rec = [ln.split("sha256")[-1].strip() for ln in open(os.path.join(
        d3, "build_report.txt")).read().splitlines()
        if ln.strip().startswith("v4 sim")]
    got = sha256_of(V4_SIM_SCRIPT)[:16]
    ck.add("provenance", "the v4 simulation script on disk is the one script "
           f"3 used ({V4_SIM_SCRIPT}; its report records 16 characters)",
           bool(rec) and rec[0] == got,
           f"recorded {rec[0] if rec else 'nothing'}, on disk {got}")
    src = design["source"]
    ck.add("provenance", "design.json names this script 3 folder and the "
           "SHA-256 of its set_A.csv, set_B.csv and set_info.json",
           src["script3_dir"] == d3 and all(
               src[f"script3_{k}_sha256"] == sha256_of(os.path.join(d3, fn))
               for k, fn in (("set_A", "set_A.csv"), ("set_B", "set_B.csv"),
                             ("set_info", "set_info.json"))))
    # the scripts' settings on disk equal the builds' settings.json
    b3 = load_module(BUILD_SCRIPT, "build_trials_v5b_settings")
    cap = b3.load_module(b3.CAPACITY_SCRIPT,
                         "analyse_pair_capacity_v5b_settings")
    b3.apply_overrides(cap)
    saved3 = json.load(open(os.path.join(d3, "settings.json")))
    now3 = json.loads(json.dumps(b3.settings_dict(cap, size), default=str))
    ck.add("provenance", "script 3's settings (and script 1's, in force) "
           "equal the build's settings.json (its 'run' record aside)",
           norm_json(saved3, ("run",)) == norm_json(now3, ("run",)))
    b4 = load_module(ORDERS_SCRIPT, "make_trial_orders_v5b_settings")
    now4 = json.loads(json.dumps(b4.settings_dict(size), default=str))
    ck.add("provenance", "script 4's settings equal the build's "
           "settings.json", now4 == s4)
    return d3, s4, info, design


# ============================================================
# 3. Rebuild
# ============================================================

def split_steps(path):
    """limit_steps.csv as (the columns that must be identical, the columns
    compared loosely), all as text."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    loose = [c for c in df.columns if c.startswith(STEPS_LOOSE_PREFIXES)
             or c in STEPS_LOOSE_COLS]
    return df.drop(columns=loose), df[loose]


def steps_loose_diff(pa, pb):
    """Largest |difference| between two limit_steps.csv files in the
    loosely compared columns, and the column it is in."""
    _, a = split_steps(pa)
    _, b = split_steps(pb)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return np.inf, "the loose columns differ in shape"
    best, where = 0.0, "-"
    for c in a.columns:
        x = pd.to_numeric(a[c], errors="coerce").to_numpy(float)
        y = pd.to_numeric(b[c], errors="coerce").to_numpy(float)
        d = np.nanmax(np.abs(x - y)) if len(x) else 0.0
        if np.isfinite(d) and d > best:
            best, where = float(d), c
    return best, where


def normalised(path):
    """(how, content) of a file for the rebuild comparison, with what
    legitimately differs between two runs removed (run time, worker count);
    content None = compare bytes."""
    base = os.path.basename(path)
    if base == STEPS_FILE:
        exact, _ = split_steps(path)
        return ("steps: exact, but averages over candidates loose",
                exact.to_csv(index=False))
    if base.endswith("_report.txt"):
        lines, skip = [], None
        for x in open(path).read().splitlines():
            if skip is not None:
                if x.startswith(skip):
                    skip = None
                else:
                    continue
            hit = [end for start, end in REPORT_SKIP_SECTIONS
                   if x.startswith(start)]
            if hit:
                skip = hit[0]
                continue
            if x.lstrip().startswith(REPORT_VOLATILE):
                continue
            for pat, rep in REPORT_SUBS:
                x = re.sub(pat, rep, x)
            lines.append(x)
        return "report minus run time and workers", lines
    if base == "checks.csv":
        t = open(path).read()
        for pat, rep in REPORT_SUBS:
            t = re.sub(pat, rep, t)
        return "checks minus worker count", t
    if base in CSV_VOLATILE_COLS:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        drop = [c for c in CSV_VOLATILE_COLS[base] if c in df.columns]
        return ("CSV minus " + ", ".join(CSV_VOLATILE_COLS[base]),
                df.drop(columns=drop).to_csv(index=False))
    if base in JSON_VOLATILE_KEYS:
        return ("JSON minus " + ", ".join(JSON_VOLATILE_KEYS[base]),
                norm_json(json.load(open(path)), JSON_VOLATILE_KEYS[base]))
    return "bytes", None


def compare_trees(orig, new, ck, scope, skip_dirs=()):
    """Every file byte-identical, except what normalised() removes. Returns
    rows for the comparison table."""
    a = walk_files(orig, skip_dirs)
    b = walk_files(new, skip_dirs) if os.path.isdir(new) else []
    rows = []
    ck.add(scope, "the rebuild writes exactly the same files",
           a == b, f"{len(a)} files; only in original "
           f"{sorted(set(a) - set(b))[:5]}, only in rebuild "
           f"{sorted(set(b) - set(a))[:5]}")
    bad = []
    for f in sorted(set(a) & set(b)):
        pa, pb = os.path.join(orig, f), os.path.join(new, f)
        how, ca = normalised(pa)
        if ca is None:
            same = filecmp.cmp(pa, pb, shallow=False)
        else:
            same = ca == normalised(pb)[1]
        rows.append({"scope": scope, "file": f, "compared_as": how,
                     "identical": same})
        if not same:
            bad.append(f)
        if os.path.basename(f) == STEPS_FILE:
            d, col = steps_loose_diff(pa, pb)
            ck.add(scope, f"{f}: the averages over candidates in the "
                   "comparison steps (" + ", ".join(
                       [f"{p}*" for p in STEPS_LOOSE_PREFIXES]
                       + list(STEPS_LOOSE_COLS)) + ") are compared "
                   "loosely: they can depend on the computer's speed and "
                   "never ship", "NOTE" if np.isfinite(d) else False,
                   f"largest difference {d:.4g} (in {col}); everything else "
                   "in the file identical: " + ("yes" if same else "NO"))
    ck.add(scope, "every file is byte-identical (the files named in the "
           "REBUILD section after removing run time and worker count)",
           not bad and bool(rows), f"{len(rows)} compared, {len(bad)} "
           f"differ {bad[:5]}")
    return rows


def snapshot():
    """Size and modification time of every file and folder outside
    REBUILD_ROOT that the rebuild could write to: the project's top level,
    data/'s top level, and the WATCH_TREES in full."""
    snap = {}
    rb = os.path.normpath(REBUILD_ROOT)
    for top in (".", "data"):
        if not os.path.isdir(top):
            continue
        for e in sorted(os.listdir(top)):
            p = os.path.normpath(os.path.join(top, e))
            if p == rb or rb.startswith(p + os.sep) and p != "data":
                continue
            if p == "data":
                continue          # its entries are listed below
            st = os.lstat(p)
            snap[p] = (os.path.isdir(p), st.st_size if not os.path.isdir(p)
                       else 0, st.st_mtime_ns)
    for tree in WATCH_TREES:
        if os.path.isdir(tree):
            for f in walk_files(tree):
                p = os.path.join(tree, f)
                st = os.lstat(p)
                snap[p] = (False, st.st_size, st.st_mtime_ns)
    return snap


def rebuild_all(stage, meta):
    """Scripts 3 and 4 rerun, as they were run, in a workspace inside the
    rebuild folder: the four build scripts copied there byte for byte, every
    input under data/ linked (read only), and the two builds' output roots
    left out, so each build writes inside the workspace and script 4 reads
    the REBUILT script 3 folders. Each step is a separate process from the
    workspace, with one thread per process (REBUILD_THREAD_VARS)."""
    rows = []
    work = f"{stage}/work"
    os.makedirs(f"{work}/data")
    os.makedirs(f"{stage}/logs")
    scripts = (CAPACITY_SCRIPT, BUILD_SCRIPT, ORDERS_SCRIPT, V4_SIM_SCRIPT)
    for s in scripts:
        shutil.copy2(s, f"{work}/{s}")
    rows.append(("rebuild", "the four build scripts are copied byte for "
                 "byte into the rebuild workspace",
                 all(sha256_of(s) == sha256_of(f"{work}/{s}")
                     for s in scripts), ", ".join(scripts)))
    outs = set()
    for size, (d3, s4, info, design) in meta.items():
        s3 = json.load(open(os.path.join(d3, "settings.json")))
        outs |= {s3["script3"]["OUTPUT_ROOT"], s4["OUTPUT_ROOT"]}
    outs |= {FROZEN_ROOT, REBUILD_ROOT}
    skip = {os.path.normpath(o).split(os.sep)[1] for o in outs
            if os.path.normpath(o).split(os.sep)[0] == "data"}
    linked = []
    for e in sorted(os.listdir("data")):
        if e in skip:
            continue
        os.symlink(os.path.relpath(os.path.join("data", e),
                                   f"{work}/data"), f"{work}/data/{e}")
        linked.append(e)
    env = dict(os.environ)
    env.update({v: "1" for v in REBUILD_THREAD_VARS})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    thr = " ".join(f"{v}=1" for v in REBUILD_THREAD_VARS)
    cmds = []
    before = snapshot()
    for tag, script in (("script3", BUILD_SCRIPT), ("script4", ORDERS_SCRIPT)):
        t1 = time.time()
        print(f"    {tag}: python {script} (in {work})", flush=True)
        with open(f"{stage}/logs/{tag}.log", "w") as lg:
            r = subprocess.run([sys.executable, script], cwd=work, env=env,
                               stdout=lg, stderr=subprocess.STDOUT)
        cmds.append({"step": tag, "cwd": work,
                     "command": f"{thr} python {script}",
                     "exit_code": r.returncode,
                     "seconds": round(time.time() - t1, 1),
                     "log": f"logs/{tag}.log"})
        rows.append(("rebuild", f"{tag} ({script}) ran to the end in the "
                     "workspace (exit code 0)", r.returncode == 0,
                     f"exit code {r.returncode}, {time.time() - t1:.0f} s; "
                     f"log {stage}/logs/{tag}.log"))
    after = snapshot()
    changed = sorted(k for k in set(before) | set(after)
                     if before.get(k) != after.get(k))
    rows.append(("rebuild", f"nothing was written outside {REBUILD_ROOT} "
                 "during the rebuild", not changed,
                 f"{len(before)} files and folders watched (the project's "
                 "top level, data/'s top level, " + ", ".join(WATCH_TREES)
                 + f"); changed {changed[:5]}"))
    with open(f"{stage}/commands.json", "w") as f:
        json.dump({"workspace": work, "linked_from_data": linked,
                   "not_linked": sorted(skip), "scripts_copied":
                   list(scripts), "commands": cmds}, f, indent=2)
    with open(f"{stage}/commands.txt", "w") as f:
        f.write("# run from the project folder; 'python' is the interpreter "
                "that ran the freeze\n")
        for c in cmds:
            f.write(f"cd {c['cwd']} && {c['command']}   # exit "
                    f"{c['exit_code']}, {c['seconds']} s, {c['log']}\n")
    return {"work": work, "commands": cmds, "rows": rows,
            "linked": linked, "skipped": sorted(skip),
            "d3": {size: f"{work}/{m[0]}" for size, m in meta.items()},
            "d4": {size: f"{work}/{SCRIPT4_DIRS[size]}"
                   for size in meta}}


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
    feat, bad_hash, sha_of = {}, [], {}
    mi = man.set_index("filename")
    for f in sorted(shipped):
        path = os.path.join(stim, f)
        feat[f] = features_of(path)
        sha_of[f] = sha256_of(path)
        if sha_of[f] != mi.loc[f, "sha256"]:
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
    rows = [(r["block_order"], r["set_mapping"], r["order"])
            for r in asg["groups"]["participant"]]
    rows_c = [(r["block_order"], r["set_mapping"], r["order"])
              for r in asg["groups"]["control"]]
    ck.add("assignment", "both groups use the same table", rows == rows_c)
    vc = pd.Series(rows).value_counts()
    n_orders = len({r[2] for r in rows})
    n_cells = 2 * 2 * n_orders
    keys_ok = all(set(r) == {"slot", "block_order", "set_mapping", "order",
                             "blocks"}
                  for lst in asg["groups"].values() for r in lst)
    ck.add("assignment", "no grid: block order x set mapping x order = 16 "
           "cells, each 4 times in 64 slots, and no other factor in a slot",
           n_cells == 16 and len(vc) == 16 and set(vc) == {4}
           and n_slots == 64 and keys_ok,
           f"{len(vc)} cells, counts {sorted(set(vc))}, {n_slots} slots")
    ck.add("assignment", f"all {n_cells} cells appear equally often in "
           f"{n_slots} slots", len(vc) == n_cells
           and vc.nunique() == 1, f"{len(vc)} cells, counts "
           f"{sorted(set(vc))}")
    for P in (10, 15, 20, 30):
        arr = np.array(rows[:P])
        dev = []
        for i, lv in enumerate(([0, 1], [0, 1],
                                list(range(1, n_orders + 1)))):
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
    levels = [[0, 1], [0, 1], list(range(1, n_orders + 1))]

    def cell_dev(P):
        """Deviation of every factor level and every pair of factor levels
        from an even split after the first P slots."""
        arr = np.array(wrapped[:P])
        out = []
        for i in range(3):
            out += [(arr[:, i] == v).sum() - P / len(levels[i])
                    for v in levels[i]]
            for j in range(i + 1, 3):
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

    # --- v5b: the new independent checks ---
    items = pd.read_csv(os.path.join(pkg, "items.csv"),
                        keep_default_na=False, na_values=[""])
    stim_rec = stimulus_set_checks(design, sha_of, ck)
    man_set = pd.read_csv(f"{STIMULUS_SET_DIR}/stimulus_manifest.csv")
    ref = read_reference(man_set, ck)
    orig, orig_how = scene_originals(man_set, ck)
    items_checks(items, lists, prac, feat, ck)
    T = normality_checks(items, lists, prac, feat, ref, orig, design, ck)
    practice_checks_v5b(design, prac, ck)
    grid_checks(pkg, ck)
    readme_checks(readme, ck)
    return {"stimulus_set": stim_rec, "normality": T, "reference": ref,
            "originals": orig_how}


# ============================================================
# 4b. v5b checks (written here, independent of scripts 3 and 4)
# ============================================================

def leak(N, xs, xc, terms=("sat", "con")):
    """N regressed on the direction terms, no intercept, and the uncentred
    correlations of N with each term. (Script 1's formula, written out
    again here: the fit is part of what this script checks.)"""
    N, xs, xc = (np.asarray(v, float) for v in (N, xs, xc))
    out = {"b_sat": np.nan, "b_con": np.nan, "cos_sat": np.nan,
           "cos_con": np.nan}
    if len(N) < 2:
        return out
    cols = {"sat": xs, "con": xc}
    X = np.column_stack([cols[t] for t in terms])
    b, *_ = np.linalg.lstsq(X, N, rcond=None)
    for t, v in zip(terms, b):
        out[f"b_{t}"] = float(v)
    nn = np.sqrt((N ** 2).sum())
    if nn > 0:
        out["cos_sat"] = float((N * xs).sum()
                               / (nn * np.sqrt((xs ** 2).sum())))
        out["cos_con"] = float((N * xc).sum()
                               / (nn * np.sqrt((xc ** 2).sum())))
    return out


def stimulus_set_checks(design, sha_of, ck):
    """The package's stimulus set is the frozen one, and every shipped image
    is listed in its checksums.sha256. Returns its record for FROZEN.json."""
    st = design.get("stimulus_set", {})
    ck_file = f"{STIMULUS_SET_DIR}/checksums.sha256"
    fr_file = f"{STIMULUS_SET_DIR}/FROZEN.json"
    rec = {"run": STIMULUS_SET_RUN, "dir": STIMULUS_SET_DIR,
           "checksums_sha256": None}
    ck.add("stimulus set", f"design.json names the stimulus set "
           f"{STIMULUS_SET_DIR}", str(st.get("dir", "")).rstrip("/")
           == STIMULUS_SET_DIR, f"design.json: {st.get('dir')}")
    if not (os.path.isfile(ck_file) and os.path.isfile(fr_file)):
        ck.add("stimulus set", "the stimulus set's checksums.sha256 and "
               "FROZEN.json are present", False, f"{STIMULUS_SET_DIR}")
        return rec
    fr = json.load(open(fr_file))
    h = sha256_of(ck_file)
    rec.update({"checksums_sha256": h,
                "manifest_sha256": fr.get("manifest_sha256"),
                "stimuli_sha256": fr.get("stimuli_sha256"),
                "settings_hash": fr.get("settings_hash")})
    ck.add("stimulus set", "the checksums.sha256 hash in design.json equals "
           "the frozen stimulus set's (its FROZEN.json, and the file itself)",
           st.get("checksums_sha256") == fr.get("checksums_sha256") == h,
           f"design.json {str(st.get('checksums_sha256'))[:16]}, FROZEN.json "
           f"{str(fr.get('checksums_sha256'))[:16]}, file {h[:16]}")
    sums = {}
    for ln in open(ck_file):
        ln = ln.rstrip("\n")
        if ln:
            hh, rel = ln.split("  ", 1)
            sums[rel] = hh
    bad = [f for f, hh in sha_of.items() if sums.get(f"stimuli/{f}") != hh]
    ck.add("stimulus set", "every shipped image matches the stimulus set's "
           "checksums.sha256", not bad, f"{len(sha_of)} images, {len(bad)} "
           "differ or are not listed" + (f": {bad[:5]}" if bad else ""))
    return rec


def read_reference(man_set, ck):
    """The corpus reference that produced the stimulus set's z values: the
    exact median (p50) and SD of sat_mean and contrast_local."""
    p = f"{STIMULUS_SET_DIR}/reference_stimulus_path.csv"
    r = pd.read_csv(p).set_index("feature")
    ref = {f: (float(r.at[f, "p50"]), float(r.at[f, "sd"]))
           for f in REFERENCE_ROUNDED}
    ok = all(round(ref[f][0], 3) == REFERENCE_ROUNDED[f][0]
             and round(ref[f][1], 3) == REFERENCE_ROUNDED[f][1]
             for f in ref)
    ck.add("normality", "the stimulus set's corpus reference rounds to "
           "median 66.448 / 26.909 and SD 39.182 / 9.348 (sat_mean / "
           "contrast_local); the exact values are used", ok,
           "; ".join(f"{f} median {m:.6f}, SD {s:.6f}"
                     for f, (m, s) in ref.items()))
    (ms, ss), (mc, sc) = ref["sat_mean"], ref["contrast_local"]
    dz = max(float(((man_set.sat_mean - ms) / ss - man_set.z_sat).abs().max()),
             float(((man_set.contrast_local - mc) / sc
                    - man_set.z_con).abs().max()))
    ck.add("normality", "those values reproduce the stimulus set's z_sat and "
           f"z_con (<= {NORM_TOL:g})", dz <= NORM_TOL,
           f"largest difference {dz:.1e}")
    return ref


def scene_originals(man_set, ck):
    """Each scene's unedited photograph (sat_mean, contrast_local), from the
    stimulus set's table; re-measured from its stimuli/ when present."""
    o = man_set[man_set.is_original.astype(str).str.lower() == "true"]
    vals = {r.scene_id: (float(r.sat_mean), float(r.contrast_local))
            for r in o.itertuples()}
    ck.add("normality", "the stimulus set's table has one original per "
           "scene", o.scene_id.is_unique
           and len(o) == man_set.scene_id.nunique(),
           f"{len(o)} originals, {man_set.scene_id.nunique()} scenes")
    src = f"{STIMULUS_SET_DIR}/stimuli"
    if all(os.path.isfile(f"{src}/{f}") for f in o.filename):
        worst = 0.0
        for r in o.itertuples():
            s, c = features_of(f"{src}/{r.filename}")
            worst = max(worst, abs(s - r.sat_mean),
                        abs(c - r.contrast_local))
            vals[r.scene_id] = (s, c)
        ck.add("normality", f"the scene originals re-measured from {src} "
               f"equal the table (<= {REMEASURE_TOL:g}); the re-measured "
               "values are used", worst <= REMEASURE_TOL,
               f"{len(o)} originals, largest difference {worst:.2e}")
        how = f"re-measured from {src}"
    else:
        ck.add("normality", "the scene originals re-measured from the "
               "stimulus set's stimuli/", "SKIPPED",
               f"{src} is not here: the table's values are used")
        how = "the stimulus set's table"
    return vals, how


def norm_terms(fa, fb, orig, ref):
    """N = distance(B) - distance(A), combined and per feature, from the
    corpus median (z) and from the scene's original (change in SD)."""
    (ms, ss), (mc, sc) = ref["sat_mean"], ref["contrast_local"]

    def dists(f):
        zs, zc = (f[0] - ms) / ss, (f[1] - mc) / sc
        os_, oc = (f[0] - orig[0]) / ss, (f[1] - orig[1]) / sc
        return {"corpus": (np.hypot(zs, zc), abs(zs), abs(zc)),
                "orig": (np.hypot(os_, oc), abs(os_), abs(oc))}
    da, db = dists(fa), dists(fb)
    return {f"N_{d}{suf}": db[d][k] - da[d][k]
            for d in NORMALITY_DEFS
            for k, suf in enumerate(("", "_sat", "_con"))}


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


# fields script 4 makes or renames in the list rows (not script 3's values)
LIST_MADE = {"trial", "repeat_of", "list_id", "set", "order",
             "repeat_of_item", "pair_type", "couple"}


def items_checks(items, lists, prac, feat, ck):
    ck.add("items", "items.csv has one row per item_id",
           bool(items.item_id.is_unique), f"{len(items)} items")
    idx = items.set_index("item_id")
    rows = pd.concat(list(lists.values()) + list(prac.values()),
                     ignore_index=True)
    missing = sorted(set(rows.item_id) - set(idx.index))
    shared = [c for c in rows.columns if c in idx.columns
              and c not in LIST_MADE and c != "item_id"]
    bad = []
    for r in rows.to_dict("records"):
        if r["item_id"] in idx.index:
            s = idx.loc[r["item_id"]]
            bad += [f"{r['item_id']}.{c}" for c in shared
                    if not _same(r[c], s[c])]
    ck.add("items", "every list and practice row's item_id is in items.csv, "
           "and the fields they share are equal", not missing and not bad,
           f"{len(rows)} rows, {len(shared)} shared fields"
           + (f"; missing {missing[:5]}" if missing else "")
           + (f"; differ {bad[:5]}" if bad else ""))
    worst = 0.0
    for r in items.itertuples():
        for f, s, c in ((r.file_a, r.sat_a, r.fog_a),
                        (r.file_b, r.sat_b, r.fog_b)):
            worst = max(worst, abs(feat[f][0] - s), abs(feat[f][1] - c))
    ck.add("items", "items.csv's image features equal the re-measured "
           f"shipped images (<= {REMEASURE_TOL:g})", worst <= REMEASURE_TOL,
           f"largest difference {worst:.2e}")


def normality_checks(items, lists, prac, feat, ref, orig, design, ck):
    """Recompute N for every item from the re-measured images, compare with
    items.csv and the lists, and verify the limits in both sets."""
    rec = {}
    for r in items.itertuples():
        rec[r.item_id] = norm_terms(feat[r.file_a], feat[r.file_b],
                                    orig[r.scene_id], ref)
    cols = list(next(iter(rec.values())))
    worst_i = max(abs(rec[r["item_id"]][c] - float(r[c]))
                  for r in items.to_dict("records") for c in cols)
    ck.add("normality", "N_corpus and N_orig, combined and per feature, "
           "recomputed from the re-measured images equal items.csv "
           f"(<= {NORM_TOL:g})", worst_i <= NORM_TOL,
           f"{len(items)} items x {len(cols)} terms, largest difference "
           f"{worst_i:.1e}")
    rows = pd.concat(list(lists.values()) + list(prac.values()),
                     ignore_index=True)
    worst_l = max(abs(rec[r.item_id][f"N_{d}"] - getattr(r, f"N_{d}"))
                  for r in rows.itertuples() for d in NORMALITY_DEFS)
    ck.add("normality", "...and equal N_corpus and N_orig in every list and "
           f"practice row (<= {NORM_TOL:g})", worst_l <= NORM_TOL,
           f"{len(rows)} rows, largest difference {worst_l:.1e}")

    nd = design.get("normality", {})
    tl = nd.get("type_limits", {})
    ok_lim = (nd.get("limit") == OVERALL_LIMIT
              and set(tl) == set(TYPE_LIMITS)
              and all(tl[t]["term"] == TYPE_LIMITS[t][0]
                      and tl[t]["limit"] == TYPE_LIMITS[t][1]
                      for t in TYPE_LIMITS))
    ck.add("normality", "design.json records the agreed limits (overall "
           f"{OVERALL_LIMIT:g}; " + "; ".join(
               f"{t} on x_{v[0]} {v[1]:g}" for t, v in TYPE_LIMITS.items())
           + ")", ok_lim)

    table, diffs = [], []
    for b in ("A", "B"):
        a = items[(items.source == f"set_{b}")
                  & items.trial_type.isin(ANALYSED)]
        xs = np.array([-(feat[r.file_a][0] - feat[r.file_b][0]) / SD_SAT
                       for r in a.itertuples()])
        xc = np.array([FOG_SIGN * (feat[r.file_a][1] - feat[r.file_b][1])
                       / SD_FOG for r in a.itertuples()])
        tt = a.trial_type.to_numpy()
        for d in NORMALITY_DEFS:
            N = np.array([rec[i][f"N_{d}"] for i in a.item_id])
            lk = leak(N, xs, xc)
            for k in ("b_sat", "b_con"):
                table.append({"set": b, "definition": d, "fit": "overall "
                              "(both terms)", "term": k, "b": lk[k],
                              "cos": lk[k.replace("b_", "cos_")],
                              "limit": OVERALL_LIMIT,
                              "margin": OVERALL_LIMIT - abs(lk[k]),
                              "n": len(a)})
            sh = nd.get("shipped", {}).get(f"{d}_{b}", {})
            diffs += [abs(lk[k] - sh.get(k, np.inf))
                      for k in ("b_sat", "b_con", "cos_sat", "cos_con")]
            for t, (term, lim) in TYPE_LIMITS.items():
                m = tt == t
                lt = leak(N[m], xs[m], xc[m], (term,))
                bt = lt[f"b_{term}"]
                table.append({"set": b, "definition": d, "fit": f"{t} on "
                              f"x_{term} alone", "term": f"b_{term}",
                              "b": bt, "cos": lt[f"cos_{term}"],
                              "limit": lim, "margin": lim - abs(bt),
                              "n": int(m.sum())})
                diffs.append(abs(bt - tl.get(t, {}).get("shipped", {})
                                 .get(f"{d}_{b}", np.inf)))
    T = pd.DataFrame(table)
    for (b, d, fit), g in T.groupby(["set", "definition", "fit"],
                                    sort=False):
        ck.add("normality limit", f"set {b}, {d} definition, {fit}: "
               f"|b| <= {g.limit.iloc[0]:g} (recomputed, judged exactly)",
               bool((g.b.abs() <= g.limit).all()),
               ", ".join(f"{r.term} {r.b:+.4f} (margin {r.margin:.4f})"
                         for r in g.itertuples()))
    worst = max(diffs)
    ck.add("normality", "the recomputed leak equals the values design.json "
           f"records (overall b and cos, per-type b; <= {NORM_TOL:g})",
           worst <= NORM_TOL, f"{len(diffs)} values, largest difference "
           f"{worst:.1e}")
    return T


def practice_checks_v5b(design, prac, ck):
    fb = design.get("practice_feedback_by_block_position", {})
    want = {str(i + 1): v for i, v in enumerate(PRACTICE_EXPECTED)}
    ok = all(len(prac[k]) == n and fb.get(k) is f
             for k, (n, f) in want.items())
    ck.add("practice", "block 1: " + f"{PRACTICE_EXPECTED[0][0]} rows with "
           f"feedback; block 2: {PRACTICE_EXPECTED[1][0]} rows without (as "
           "design.json records)", ok
           and design.get("practice_by_block") == [n for n, _ in
                                                   PRACTICE_EXPECTED],
           f"rows {[len(prac[k]) for k in sorted(prac)]}, feedback {fb}")


def json_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from json_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from json_keys(v)


def grid_checks(pkg, ck):
    files = walk_files(pkg)
    gf = [f for f in files if "grid" in f.lower()]
    gk = []
    for f in files:
        if f.endswith(".json"):
            gk += [f"{f}: {k}" for k in json_keys(json.load(open(
                os.path.join(pkg, f)))) if "grid" in k.lower()]
        elif f.endswith(".csv") and not f.startswith("stimuli/"):
            head = open(os.path.join(pkg, f)).readline()
            if "grid" in head.lower():
                gk.append(f"{f}: header")
    ck.add("no grid", "no grid file, and no grid field in any JSON or CSV "
           "header of the package", not gf and not gk,
           f"files {gf[:3]}, fields {gk[:3]}")


def readme_checks(readme, ck):
    rd = " ".join(readme.split())
    ok = ("EXPERIMENT CHANGES NEEDED" in readme and "record.js" in rd
          and "item_id" in rd and "grid" in rd.lower())
    ck.add("README", "the README lists the experiment changes needed: the "
           "grid code, and saving item_id in record.js", ok)


def abs_path_hits(root, skip_dirs=()):
    """Text files under root that contain an absolute path."""
    pats = [re.compile(p) for p in ABS_PATH_PATTERNS]
    here = os.getcwd()
    hits = []
    for f in walk_files(root, skip_dirs):
        if not f.endswith(TEXT_EXT):
            continue
        t = open(os.path.join(root, f), errors="replace").read()
        if here in t or any(p.search(t) for p in pats):
            hits.append(f)
    return hits


def text_has_abs_path(t):
    return (os.getcwd() in t
            or any(re.search(p, t) for p in ABS_PATH_PATTERNS))


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


def frozen_readme(name, size, design, stim, d3, d4, scripts):
    return f"""FROZEN TRIAL LISTS  {name}  ({size} analysed trials per block)
====================================================================

USE
  Copy package/ to the experiment as data/sets/{name}/. Verify
  checksums.sha256 before any analysis (sha256sum -c checksums.sha256,
  from this folder). package/README.txt describes the package.

WHAT IS HERE
  package/      the experiment package, byte-identical to script 4's output
                ({d4}/package)
  provenance/   the builds' records: script 3 ({d3}, without its images),
                script 4 (without its package), the rebuild comparison and
                commands, and this freeze's checks
  README.txt    this file
  checksums.sha256, FROZEN.json, freeze_report.txt

STIMULUS SET
  {stim.get('run')} (folder {stim.get('dir')})
  SHA-256 of its checksums.sha256: {stim.get('checksums_sha256')}
  manifest SHA-256: {stim.get('manifest_sha256')}
  Every image in package/stimuli/ is checked against that checksums file.

IMAGES AND ARCHIVING
  The images are edited versions of photographs from Places365, whose
  copyright belongs to the images' owners. This package can be archived
  and cited, but the images should not be redistributed publicly until
  their licence is confirmed. The stimulus set's sources.csv (the
  Places365 file of every scene) and its checksums allow anyone with
  Places365 to rebuild the images and verify them.

SCRIPTS (SHA-256)
""" + "".join(f"  {k}  {v}\n" for k, v in scripts.items())


def freeze_one(size, d3, d4, s4, info, design, ck, rb, stage, t0):
    name = design["package"]
    print(f"\n  n={size}: comparing the rebuild with {d3} and {d4}")
    cmp_rows = compare_trees(d3, rb["d3"][size], ck, "rebuild: script 3")
    cmp_rows += compare_trees(d4, rb["d4"][size], ck, "rebuild: script 4")
    print("    checking the package")
    res = package_checks(size, d3, d4, info, design, ck) or {}
    hits = ([f"package/{h}" for h in abs_path_hits(
        os.path.join(d4, "package"))]
        + [f"script3/{h}" for h in abs_path_hits(d3, ("stimuli",))]
        + [f"script4/{h}" for h in abs_path_hits(d4, ("package",))])
    ck.add("relative paths", "no file to be frozen contains an absolute path",
           not hits, f"{len(hits)} files" + (f": {hits[:5]}" if hits else ""))
    sdir = os.path.join(stage, name)
    os.makedirs(sdir)
    pd.DataFrame(cmp_rows).to_csv(os.path.join(sdir, "rebuild_comparison.csv"),
                                  index=False)
    n_fail = int((ck.frame().status == "FAIL").sum())
    if n_fail:
        ck.frame().to_csv(os.path.join(sdir, "checks.csv"), index=False)
        write_report(os.path.join(sdir, "freeze_report.txt"), size, ck, d3,
                     d4, None, t0, cmp_rows, rb, res)
        print(f"    {n_fail} check(s) FAIL: NOT frozen. Report: "
              f"{os.path.join(sdir, 'freeze_report.txt')}")
        return None, n_fail
    frozen = new_dir(os.path.join(FROZEN_ROOT, name))
    shutil.copytree(os.path.join(d4, "package"),
                    os.path.join(frozen, "package"))
    prov = os.path.join(frozen, "provenance")
    shutil.copytree(d3, os.path.join(prov, "script3"),
                    ignore=shutil.ignore_patterns("stimuli"))
    shutil.copytree(d4, os.path.join(prov, "script4"),
                    ignore=shutil.ignore_patterns("package"))
    os.makedirs(os.path.join(prov, "rebuild"))
    shutil.copy2(os.path.join(sdir, "rebuild_comparison.csv"),
                 os.path.join(prov, "rebuild", "rebuild_comparison.csv"))
    for f in ("commands.json", "commands.txt"):
        shutil.copy2(os.path.join(stage, f), os.path.join(prov, "rebuild", f))
    me = os.path.basename(__file__)
    scripts = {p: sha256_of(p) for p in (CAPACITY_SCRIPT, BUILD_SCRIPT,
                                         ORDERS_SCRIPT, V4_SIM_SCRIPT)}
    scripts[me] = sha256_of(__file__)
    stim = res.get("stimulus_set", {})
    with open(os.path.join(frozen, "README.txt"), "w") as f:
        f.write(frozen_readme(name, size, design, stim, d3, d4, scripts))
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
    T = res.get("normality")
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
        "items_csv_sha256": sha256_of(os.path.join(frozen, "package",
                                                   "items.csv")),
        "script3_dir": d3, "script4_dir": d4,
        "scripts_sha256": scripts,
        "stimulus_set": stim,
        "normality": {
            "overall_limit": OVERALL_LIMIT,
            "type_limits": {t: {"term": v[0], "limit": v[1]}
                            for t, v in TYPE_LIMITS.items()},
            "recomputed": ([] if T is None else
                           T.round(6).to_dict("records")),
            "reference": res.get("reference"),
            "originals": res.get("originals")},
        "rebuild": {"folder": stage, "commands": rb["commands"]},
        "checks": {"pass": int((ck.frame().status == "PASS").sum()),
                   "fail": 0},
        "unhashed": list(UNHASHED),
        "use": f"copy package/ to the experiment as data/sets/{name}/; "
               "verify checksums.sha256 before any analysis",
        "images": "Places365 photographs, copyright of their owners: archive "
                  "and cite, but do not redistribute the images publicly "
                  "until the licence is confirmed (README.txt)",
    }
    with open(os.path.join(frozen, "FROZEN.json"), "w") as f:
        json.dump(record, f, indent=2)
    # the last check: nothing frozen holds an absolute path (the report is
    # checked as text before it is written)
    hits = abs_path_hits(frozen)
    ok = ck.add("relative paths", "no frozen file contains an absolute path "
                "(checked after writing; not in provenance/freeze_checks.csv)",
                not hits, f"{len(hits)} files" + (f": {hits[:5]}"
                                                  if hits else ""))
    rp = os.path.join(frozen, "freeze_report.txt")
    txt = write_report(None, size, ck, d3, d4, frozen, t0, cmp_rows, rb, res)
    if not ok or text_has_abs_path(txt):
        shutil.rmtree(frozen)
        ck.add("relative paths", "the freeze report holds no absolute path",
               not text_has_abs_path(txt))
        write_report(os.path.join(sdir, "freeze_report.txt"), size, ck, d3,
                     d4, None, t0, cmp_rows, rb, res)
        print("    an absolute path was found: the frozen folder was "
              f"removed. Report: {os.path.join(sdir, 'freeze_report.txt')}")
        return None, 1
    with open(rp, "w") as f:
        f.write(txt)
    print(f"    frozen: {frozen} ({n_files} files hashed)")
    return frozen, 0


def write_report(path, size, ck, d3, d4, frozen, t0, cmp_rows, rb, res):
    L = []
    w = L.append
    w("=" * 78)
    w(f"v5b STEP 5: FREEZE, {size} PER BLOCK")
    w("freeze_trial_lists_v5b.py")
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
      f"{int((cf.status == 'SKIPPED').sum())} SKIPPED, "
      f"{int((cf.status == 'NOTE').sum())} NOTE"
      + ("" if n_f or not frozen else " -- frozen"))
    w("")
    w("REBUILD")
    w("-" * 78)
    w(f"Workspace {rb['work']}: the four build scripts copied byte for byte; "
      "every entry of data/ linked")
    w(f"(read only) except the builds' output roots ({', '.join(rb['skipped'])}"
      "), so script 3 writes there and")
    w("script 4 reads the REBUILT script 3 folders. Each step a separate "
      "process, one thread each:")
    for c in rb["commands"]:
        w(f"  cd {c['cwd']} && {c['command']}")
        w(f"      exit {c['exit_code']}, {c['seconds']:.0f} s, log "
          f"{c['log']}")
    t = pd.DataFrame(cmp_rows)
    if len(t):
        for (scope, how), g in t.groupby(["scope", "compared_as"]):
            w(f"  {scope:20s} {how:38s} {int(g.identical.sum())} of "
              f"{len(g)} identical")
    w("What legitimately differs between two runs, and is removed before "
      "comparing:")
    w("  reports: the lines starting " + ", ".join(
        f"'{x}'" for x in REPORT_VOLATILE) + "; the seconds in")
    w("    'random choosers, N s)' and the count in 'N worker(s) in the "
      "run' (also in script 3's checks.csv);")
    w("  " + "; ".join(f"{k}: the columns {', '.join(v)}"
                       for k, v in CSV_VOLATILE_COLS.items())
      + "; " + "; ".join(f"{k}: the key(s) {', '.join(v)}"
                         for k, v in JSON_VOLATILE_KEYS.items()) + ".")
    w(f"  {STEPS_FILE}: the averages over candidates in script 3's comparison "
      "steps (" + ", ".join([f"{p}*" for p in STEPS_LOOSE_PREFIXES]
                            + list(STEPS_LOOSE_COLS)) + ")")
    w("    are compared loosely, with the largest difference stated in the "
      "checks; every other column,")
    w("    meets_cost_rule included, must be identical. Report section 5b "
      "(the same numbers) is left")
    w("    out of the line comparison. (A solve in those steps that reaches "
      "script 3's time limit can end")
    w("    differently on a busier computer; those candidates never ship.)")
    w("  Everything else, including every list, set, image and design file, "
      "is compared byte for byte.")
    w("")
    T = res.get("normality") if res else None
    if T is not None and len(T):
        w("NORMALITY (recomputed here from the re-measured shipped images)")
        w("-" * 78)
        ref = res.get("reference", {})
        w("Corpus reference (stimulus set): " + "; ".join(
            f"{f} median {m:.6f}, SD {s:.6f}" for f, (m, s) in ref.items()))
        w(f"Scene originals: {res.get('originals')}.")
        w(f"  {'set':3s} {'definition':10s} {'fit':26s} {'term':6s} "
          f"{'b':>8s} {'cos':>7s} {'limit':>6s} {'margin':>7s} {'n':>4s}")
        for r in T.itertuples():
            w(f"  {r.set:3s} {r.definition:10s} {r.fit:26s} {r.term:6s} "
              f"{r.b:+8.4f} {r.cos:+7.3f} {r.limit:6.2f} {r.margin:7.4f} "
              f"{r.n:4d}")
        w("  margin = limit - |b|; the limits are judged on these values "
          "exactly.")
        w("")
    w("CHECKS")
    w("-" * 78)
    for r in cf.itertuples():
        w(f"  {r.status:7s} [{r.scope}] {r.check}"
          + (f"  ({r.detail})" if r.detail else ""))
    txt = "\n".join(L) + "\n"
    if path:
        with open(path, "w") as f:
            f.write(txt)
    return txt


def main():
    t0 = time.time()
    print("v5b step 5: verify and freeze")
    ck, meta = {}, {}
    for size, d4 in SCRIPT4_DIRS.items():
        ck[size] = Checks()
        print(f"\n  n={size}: {d4}")
        meta[size] = inputs_and_provenance(size, d4, ck[size])
    stage = new_dir(os.path.join(REBUILD_ROOT, "rebuild"))
    os.makedirs(stage)
    print(f"\n  rebuilding scripts 3 and 4 into {stage}")
    rb = rebuild_all(stage, meta)
    out = []
    for size, d4 in SCRIPT4_DIRS.items():
        for row in rb["rows"]:
            ck[size].add(*row)
        d3, s4, info, design = meta[size]
        out.append(freeze_one(size, d3, d4, s4, info, design, ck[size], rb,
                              stage, t0))
    print(f"\n  elapsed {time.time() - t0:.0f} s")
    if any(n for _, n in out):
        sys.exit(1)


if __name__ == "__main__":
    main()
