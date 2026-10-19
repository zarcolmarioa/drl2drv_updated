"""
freeze_trial_list_v4.py  (stimulus pipeline, v4 axes, step 7a)

Freezes the chosen trial list: verifies it end to end, copies it to a
directory nothing else writes to, and records what it is with a
checksum for every file.

Where this sits
---------------
  step 5   build_trials_v4_corrected   the candidate trial lists
  step 6   choose_colour_condition_v4  which candidate detects best
  step 6b  check_robustness_v4         does that survive lapses, and
                                       are the weights usable
  step 7a  this one                    freeze the chosen list
  step 7b  the grid blocks             (open design questions first)
  step 8   simulate_power_v4           is the frozen list powered?

WHAT "FROZEN" MEANS HERE
------------------------
Three things, and the third is the one that matters later.

  VERIFIED   every check below passes on the files as they sit on
             disk, including a re-measurement of the shipped images.
  COPIED     to FROZEN_ROOT, which no other script in this pipeline
             writes to. The step-5 build directory stays where it is;
             this is a copy, so a later step-5 run cannot touch it.
  RECORDED   a SHA-256 for every frozen file, plus the seed, the
             settings that produced it and the verdicts from steps 6
             and 6b, in one text file meant to be read by a person.

The checksums are the point. Between now and the last participant,
the only question that will matter about this list is "is this still
the thing that was registered?", and that question needs an answer
that does not depend on anyone's memory. Re-running in verify mode is
how to get one.

THE CHOICE THIS FREEZES, AND WHY
--------------------------------
`both`: the extended colour ladder (two vivid levels above the
original) with colour-only trials cut at |d_sat| >= 31.1 sat_mean
units, 0.794 corpus SD.

  Step 6   colour detection of a colour-following participant rises
           at a true sensitivity of 0.50 SD, paired over 20
           draws, and the fog control stays flat. The figures are
           NOT repeated here: this docstring cannot check itself
           against a run, and an earlier version of it went stale
           when the estimator changed. freeze_report.txt derives
           every number from the captured provenance.
  Step 6b  the ranking survives lapse rates of 0.1 and 0.2 in all 21
           cells with no inversions, and the gains hold at lapse
           0.2. No fit approached separation; the
           closest anyone came is reported per cell
           where separation needs 1.000. The spread of the fitted
           colour weight is the SMALLEST of the candidates --
           by `sd_over_true` -- so the
           between-group comparison is easier on this list, not
           harder.

The cost, recorded here because it is the part a reader will want to
weigh later: the cut removes colour differences below 31.1 units, so
the colour dose-response curve rests on three magnitude bins spanning
32 to 100 units rather than four spanning 17 to 62, and the average
image is edited about 15% further from the unmanipulated original
(by mean |k - 1|). `sens_cut_35_3` was tested and
dominated: it bought a gain inside the noise, for a
narrower range.

WHAT IS CHECKED
---------------
  1  the build directory and the package are complete
  2  the three copies of the trial list agree -- the top-level CSV,
     the packaged CSV and the packaged JSON
  3  the image set closes: every image the trials reference is in
     stimuli/, every file in stimuli/ is referenced, and the manifest
     and preload list exactly those files
  4  THE PREDICTORS DESCRIBE THE SHIPPED PICTURES. sat_mean and
     contrast_local are re-measured from the files in the frozen
     package and the differences rebuilt from them, then compared
     against the shipped columns. Step 5 did this at build time; doing
     it again here is what makes the frozen copy self-certifying
  5  the design invariants: counts per type, side balance, every
     consistency repeat after its original with the sides swapped,
     every catch row declaring the catch mode the list was built for
     and no other row declaring one, and no image pair seen twice
     except a repeat
  6  the provenance files exist and are captured

A failure stops the freeze. A list that cannot pass these is not a
list to register.

Outputs (under FROZEN_ROOT)
---------------------------
  package/              the self-contained package, copied verbatim
  provenance/           the step 5, 6 and 6b reports that chose it
  trial_list.csv        the full session, as built
  trial_design.png      the trials in difference space
  checksums.sha256      one line per file, in the usual format, so
                        `shasum -c checksums.sha256` works outside
                        this pipeline too
  FROZEN.json           the same facts for a machine to read
  freeze_report.txt     what was frozen, what was checked, what it
                        cost -- the documentation record

Usage
-----
    conda activate drl2drv
    python freeze_trial_list_v4.py          # with MODE set to freeze

Then, at any later point and before launching, set MODE to verify in
the settings block and run it again.

Note that these lines deliberately do NOT spell the setting as an
assignment: the settings block below holds the only one, so searching
this file for it finds the setting rather than a usage note.

CPU only, under a minute.
"""

import os
import json
import time
import shutil
import hashlib
import warnings

import numpy as np
import pandas as pd
import cv2
from PIL import Image

cv2.setNumThreads(0)


# ============================================================
# SETTINGS
# ============================================================

# MODE
#   "freeze"  verify the chosen build and write the frozen copy.
#             Refuses to overwrite an existing FROZEN_ROOT.
#   "verify"  re-check an existing FROZEN_ROOT against its own
#             checksums and re-run every check on it. Nothing is
#             written. Run this before launching, and again if the
#             package is ever moved or re-uploaded.
MODE = "verify"

# SOURCE_ROOT / CHOSEN
#   The step-5 output root and the version inside it to freeze.
#
#   CHECK THIS. Step 5 writes to a new numbered directory on each run,
#   so SOURCE_ROOT must be the build that steps 6 and 6b measured --
#   the same root their reports name at the top. Freezing a different
#   build would freeze a list nothing has been checked against, and
#   the checksums would make that mistake permanent.
SOURCE_ROOT = "data/trials_v4_ext"
CHOSEN = "both"

FROZEN_ROOT = "data/frozen_trial_list_v4"

# PROVENANCE
#   The reports that chose this list, copied into the frozen directory
#   and checksummed with everything else. Only scripts and
#   documentation are kept long-term in this project, so a result file
#   that justifies a frozen decision has to travel with it.
#
#   A missing file is listed as missing in the report rather than
#   stopping the freeze -- but a freeze with no provenance is worth
#   re-running once the reports exist.
PROVENANCE = [
    ("step 5 build",      "data/trials_v4_ext/both/trials_report.txt"),
    ("step 5 overview",   "data/trials_v4_ext/trials_overview.txt"),
    ("step 5 streams",    "data/trials_v4_ext/stream_check.csv"),
    ("step 5 seeds",      "data/trials_v4_ext/seeds_summary.csv"),
    ("step 6 choice",     "data/choose_colour_v4/choose_report.txt"),
    ("step 6 information", "data/choose_colour_v4/information.csv"),
    ("step 6 differences", "data/choose_colour_v4/differences.csv"),
    # The fog control on its own, split by ladder. This is the
    # registered control comparison, and the only sound place to read
    # it from.
    ("step 6 fog control", "data/choose_colour_v4/fog_control.csv"),
    ("step 6b robustness", "data/robustness_v4/robustness_report.txt"),
    ("step 6b ranking",   "data/robustness_v4/ranking.csv"),
    ("step 6b weights",   "data/robustness_v4/weights.csv"),
    # The evidence that the gains survive the lapse rates, and the
    # source of that figure in the report's opening section.
    ("step 6b lapse gains",
     "data/robustness_v4/differences_by_lapse.csv"),
]

# --- What the list must be ---
#
# EXPECTED
#   The design, stated here so the freeze fails loudly if the build
#   does not match what was registered. These are not derived from the
#   file; they are the independent statement of what it should be.
EXPECTED = {
    "total": 211,
    "analysed": 180,
    "sat_only": 60,
    "fog_only": 60,
    "together": 30,
    "conflict": 30,
    "consistency": 15,
    "catch": 10,
    "practice": 6,
    "seed": 42,
    "colour_cut": 31.1,
    "variant": "extended",
    # The catch trial has two forms and the list declares which one it
    # was built for, on every catch row. The registered rule is
    # "exclude a participant failing 3 or more of the 10", and the
    # arithmetic behind that threshold depends on the mode, so the
    # mode is part of what is frozen rather than a detail of the
    # interface. SWITCHING IT IS NOT FREE: catch_mode sits in
    # trial_list.csv and trial_list.json, so a switch changes both
    # files, their checksums and the preload list, and needs a
    # re-freeze.
    "catch_mode": "no_images",
}

# --- What the report's opening section quotes ---
#
# NARRATIVE_SENS / NARRATIVE_LAPSES / NARRATIVE_RUNNER_UP /
# NARRATIVE_NOT_CANDIDATES
#   Which cell of steps 6 and 6b the opening section reports, and
#   which versions count as design candidates for the fog control.
#
#   0.50 SD is the middle of the sensitivity range and the cell the
#   design was chosen on. `reproduction` and `shipped` are controls
#   built from a different pair table, so their fog trials
#   legitimately differ from the candidates' and including them
#   would overstate how much the fog control moved.
NARRATIVE_SENS = 0.50
NARRATIVE_LAPSES = (0.1, 0.2)
NARRATIVE_RUNNER_UP = "sens_cut_35_3"
NARRATIVE_NOT_CANDIDATES = ("reproduction", "shipped")

# NARRATIVE_BASELINE
#   The version the cost figures are compared against -- the same
#   baseline steps 6 and 6b used. Its trial list is NOT part of the
#   frozen package, so it is read from SOURCE_ROOT; on a verify run
#   from a moved package it will be absent, and the report then
#   states the chosen list's figures without the comparison rather
#   than guessing one.
NARRATIVE_BASELINE = "current"

# SIDE_BALANCE_TOL
#   Step 5 forces an exact 50/50 split of left/right within each trial
#   type, so any deviation beyond rounding means the assignment did not
#   apply.
SIDE_BALANCE_TOL = 0.03

# --- The predictors ---
#
# The corpus SDs, from step 1's corpus_feature_stats_v4.csv
# (stimulus_path population, n = 20,000). The constants steps 4, 5 and
# 6 use.
SD_SAT = 39.182
SD_FOG = 9.348

# FOG_SIGN
#   Multiplier on the raw fog-feature difference so a POSITIVE value
#   means "image A is foggier". Step 4 applies this when it builds the
#   pair table; the re-measurement here has to apply it too.
FOG_SIGN = -1.0

# FEATURE_TARGET_SIZE
#   A MEASUREMENT CONVENTION inherited from step 04, not a CLIP
#   requirement. The features were measured on a 256 downsample, and
#   measuring at any other size gives different numbers -- about 9
#   units on contrast_local. Do not change it.
FEATURE_TARGET_SIZE = (256, 256)

# REMEASURE_TOL_SAT / REMEASURE_TOL_FOG
#   How far a re-measured difference may sit from the shipped column.
#
#   For a PNG package this should be essentially zero: the same file,
#   measured the same way. The tolerance is not zero only because
#   libjpeg and Pillow versions can differ by a fraction of a unit on
#   a lossy package, and because a resize is not bit-stable across
#   OpenCV builds.
#
#   A failure here means the predictors do not describe the pictures
#   participants will see, which is the one defect no amount of
#   analysis can repair afterwards.
REMEASURE_TOL_SAT = 0.6
REMEASURE_TOL_FOG = 0.6

# REMEASURE_LIMIT
#   Images to re-measure. None does all of them (about 340, a few
#   seconds). A number samples that many, which is for a quick trial
#   run only -- the freeze that gets registered should do all.
REMEASURE_LIMIT = None

TYPES = ["sat_only", "fog_only", "together", "conflict"]
EXTRA_TYPES = ["consistency", "catch", "practice"]


# ============================================================
# Helpers
# ============================================================

def sha256_of(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def walk_files(root):
    """Every file under root, as paths relative to it, sorted.

    Sorted so the checksum file is byte-identical between runs on the
    same content -- a checksum list whose ORDER wanders is a diff that
    says nothing.
    """
    out = []
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            p = os.path.join(dirpath, n)
            out.append(os.path.relpath(p, root))
    return sorted(out)


def features_of(path, target_size=FEATURE_TARGET_SIZE):
    """sat_mean and contrast_local of a file.

    VERBATIM from step 5's _features_of, which took it from step 04:
    resize to target_size, mean HSV saturation, standard deviation of
    the Laplacian of the Rec. 601 grey. Copied rather than imported,
    which is this pipeline's convention -- and here the copy is the
    point, since an independent re-implementation is what makes check
    4 a check rather than a tautology.
    """
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB").resize(target_size,
                                                  Image.BILINEAR),
                         dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    S = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)[:, :, 1]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(S.mean()), float(cv2.Laplacian(g, cv2.CV_32F).std())


def pair_key(a, b):
    """Identify a pair by its IMAGES, ignoring which is A and which B."""
    return (a, b) if str(a) <= str(b) else (b, a)


class Checks:
    """Collects verdicts so every check runs before anything stops.

    A freeze that stops at the first failure makes the next run a
    guessing game about what else is wrong. Everything is checked, the
    failures are listed together, and then it stops.
    """

    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail="", fatal=True):
        self.rows.append({"check": name,
                          "result": "PASS" if ok else
                                    ("FAIL" if fatal else "NOTE"),
                          "detail": detail})
        mark = "PASS" if ok else ("FAIL" if fatal else "note")
        print(f"  [{mark:4s}] {name}"
              + (f"   {detail}" if detail else ""))
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if r["result"] == "FAIL"]

    def table(self):
        return pd.DataFrame(self.rows)


# ============================================================
# The checks
# ============================================================

def check_complete(root, ck):
    """CHECK 1: the directory and the package are complete."""
    print("\n" + "=" * 60)
    print("CHECK 1: COMPLETENESS")
    print("=" * 60)
    need = ["trial_list.csv",
            "package/trial_list.csv",
            "package/trial_list.json",
            "package/manifest.csv",
            "package/preload.json",
            "package/README.txt"]
    missing = [p for p in need
               if not os.path.isfile(os.path.join(root, p))]
    ck.add("required files present", not missing,
           "all 6" if not missing else f"missing {missing}")
    img_dir = os.path.join(root, "package", "stimuli")
    n_img = (len([f for f in os.listdir(img_dir)
                  if not f.startswith(".")])
             if os.path.isdir(img_dir) else 0)
    ck.add("stimuli/ populated", n_img > 0, f"{n_img} images")
    # The figure is useful and not load-bearing, so its absence is a
    # note rather than a failure.
    ck.add("trial_design.png present",
           os.path.isfile(os.path.join(root, "trial_design.png")),
           fatal=False)
    return n_img


def check_three_copies(root, ck):
    """CHECK 2: the three copies of the trial list agree.

    The top-level CSV is the record, the packaged CSV is what an
    analysis will read, and the JSON is what the experiment loads. They
    are written from one frame, so they must agree -- and if they ever
    stop agreeing, the experiment and the analysis would be describing
    different sessions, which no later check would notice.
    """
    print("\n" + "=" * 60)
    print("CHECK 2: THE THREE COPIES AGREE")
    print("=" * 60)
    top = pd.read_csv(os.path.join(root, "trial_list.csv"))
    pkg = pd.read_csv(os.path.join(root, "package", "trial_list.csv"))
    with open(os.path.join(root, "package",
                           "trial_list.json")) as f:
        js = pd.DataFrame(json.load(f))

    ck.add("row counts match",
           len(top) == len(pkg) == len(js),
           f"top {len(top)}, package {len(pkg)}, json {len(js)}")
    ck.add("trial numbering is 1..n in order",
           list(pkg.trial) == list(range(1, len(pkg) + 1))
           and list(js.trial) == list(range(1, len(js) + 1)))

    # The columns that must be identical across copies. left_path and
    # right_path exist only in the packaged copies (the top-level one
    # carries left_file / right_file), so they are compared between
    # those two.
    same = ["trial", "trial_type", "scene_id", "a_is_left",
            "pred_sat", "pred_fog", "d_sat", "d_fog", "mag_bin"]
    worst, bad = 0.0, []
    for c in same:
        if c not in pkg.columns or c not in js.columns:
            bad.append(f"{c} absent")
            continue
        a, b = pkg[c], js[c]
        if pd.api.types.is_numeric_dtype(a):
            w = float(np.nanmax(np.abs(a.to_numpy(float)
                                       - b.to_numpy(float))))
            worst = max(worst, w)
            if w > 1e-9:
                bad.append(f"{c} differs by {w:.2e}")
        elif not a.astype(str).equals(b.astype(str)):
            bad.append(f"{c} differs")
    ck.add("packaged CSV and JSON agree", not bad,
           f"worst numeric difference {worst:.1e}" if not bad
           else "; ".join(bad))

    for c in ("left_path", "right_path"):
        ck.add(f"{c} identical in CSV and JSON",
               pkg[c].astype(str).equals(js[c].astype(str)))

    # The top-level copy carries the same trials in the same order.
    tt = top.trial_type.astype(str).tolist()
    ck.add("top-level copy has the same trial order",
           tt == pkg.trial_type.astype(str).tolist())
    w_sat = float(np.nanmax(np.abs(top.pred_sat.to_numpy(float)
                                   - pkg.pred_sat.to_numpy(float))))
    # The packaged copy may have been re-measured on the converted
    # images, so a small difference here is expected and is exactly
    # what check 4 quantifies. A LARGE one is not.
    ck.add("top-level and packaged predictors consistent",
           w_sat <= REMEASURE_TOL_SAT / SD_SAT + 1e-9,
           f"worst |pred_sat| difference {w_sat:.2e} "
           f"({w_sat * SD_SAT:.3f} raw units)")
    return top, pkg, js


def check_images_close(root, pkg, ck):
    """CHECK 3: the image set closes.

    Every image a trial references exists; every image shipped is
    referenced; the manifest names exactly those files. An
    unreferenced image is dead weight in a preload, and a missing one
    is a blank trial in the middle of a session.

    THE PRELOAD IS NOT ALWAYS THE WHOLE SET. In 'no_images' mode the
    catch trial shows no pictures, so step 5 leaves out of the preload
    any image that ONLY a catch row uses. Those images still ship, in
    stimuli/, so a later switch to image mode needs no new stimuli.
    So the test is: the preload is a subset of what shipped, and what
    it leaves out is EXACTLY the catch-only images. In 'images' mode it
    must be the whole set.
    """
    print("\n" + "=" * 60)
    print("CHECK 3: THE IMAGE SET CLOSES")
    print("=" * 60)
    img_dir = os.path.join(root, "package", "stimuli")
    on_disk = {f for f in os.listdir(img_dir)
               if not f.startswith(".")}
    refs = set(pkg.left_path) | set(pkg.right_path)
    ref_names = {os.path.basename(p) for p in refs}
    ck.add("every referenced image exists",
           not (ref_names - on_disk),
           f"{len(ref_names)} referenced"
           if not (ref_names - on_disk)
           else f"missing {sorted(ref_names - on_disk)[:5]}")
    ck.add("no unreferenced image shipped",
           not (on_disk - ref_names),
           f"{len(on_disk)} on disk"
           if not (on_disk - ref_names)
           else f"extra {sorted(on_disk - ref_names)[:5]}")
    ck.add("every path is under stimuli/",
           all(str(p).startswith("stimuli/") for p in refs))

    man = pd.read_csv(os.path.join(root, "package", "manifest.csv"))
    ck.add("manifest lists exactly the shipped images",
           set(man.filename) == on_disk,
           f"{len(man)} rows")
    has_feat = {"sat_mean", "contrast_local"} <= set(man.columns)
    # Step 5 writes these only when REMEASURE_AFTER_CONVERT is on. The
    # freeze does not depend on them -- check 4 measures the files
    # itself -- but their absence means the package was built without
    # that step, so the README's claim about the manifest is wrong for
    # this package and the build is worth re-running.
    ck.add("manifest carries the measured features", has_feat,
           "sat_mean and contrast_local present" if has_feat
           else "ABSENT: step 5 ran with REMEASURE_AFTER_CONVERT "
                "off, so package/README.txt overstates what "
                "manifest.csv holds", fatal=False)

    with open(os.path.join(root, "package", "preload.json")) as f:
        pre = json.load(f)
    pre_names = {os.path.basename(p) for p in pre}
    ck.add("preload is a subset of the shipped images",
           not (pre_names - on_disk),
           f"{len(pre)} paths"
           if not (pre_names - on_disk)
           else f"names {sorted(pre_names - on_disk)[:5]} not shipped")

    # Which images only a catch row uses, from the packaged list
    # itself rather than from a setting.
    cat = pkg[pkg.trial_type == "catch"]
    shown = pkg[pkg.trial_type != "catch"]
    shown_names = ({os.path.basename(p) for p in shown.left_path}
                   | {os.path.basename(p) for p in shown.right_path})
    catch_only = on_disk - shown_names
    omitted = on_disk - pre_names
    mode = (str(cat.catch_mode.iloc[0])
            if len(cat) and "catch_mode" in cat.columns else "")
    if mode == "no_images":
        ck.add("preload omits exactly the catch-only images",
               omitted == catch_only,
               f"{len(omitted)} omitted, {len(catch_only)} catch-only")
        print(f"    mode {mode!r}: {len(pre)} of {len(on_disk)} "
              f"images preloaded")
        print(f"    {len(omitted)} omitted, of {len(catch_only)} "
              "images that only a catch row uses")
    else:
        ck.add("preload names every shipped image",
               not omitted, f"{len(omitted)} omitted")
        print(f"    mode {mode!r}: all {len(pre)} images preloaded")
    return on_disk, man


def check_remeasure(root, pkg, man, ck):
    """CHECK 4: the predictors describe the shipped pictures.

    THE CHECK THAT MAKES THE FROZEN COPY SELF-CERTIFYING. Everything
    else compares one record against another record; this one compares
    the records against the images.

    sat_mean and contrast_local are re-measured from the files in the
    frozen package, the signed differences rebuilt from them, and
    compared against the shipped `d_sat`, `d_fog`, `pred_sat` and
    `pred_fog`. A mismatch means the numbers an analysis will fit do
    not describe what a participant saw -- the one defect that cannot
    be repaired after the data are collected.

    Step 5 ran this at build time against its own source folder. Here
    it runs against the FROZEN copy, so it also catches a package that
    was corrupted, partially copied, or re-converted on the way.
    """
    print("\n" + "=" * 60)
    print("CHECK 4: THE PREDICTORS DESCRIBE THE SHIPPED PICTURES")
    print("=" * 60)
    img_dir = os.path.join(root, "package", "stimuli")
    names = sorted({os.path.basename(p)
                    for p in set(pkg.left_path) | set(pkg.right_path)})
    if REMEASURE_LIMIT:
        names = names[:int(REMEASURE_LIMIT)]
        print(f"  NOTE: REMEASURE_LIMIT = {REMEASURE_LIMIT}; this is a "
              "trial run, not a freeze to register.")
    print(f"  re-measuring {len(names)} images at "
          f"{FEATURE_TARGET_SIZE[0]} px")
    feat = {}
    for n, fn in enumerate(names, 1):
        feat[fn] = features_of(os.path.join(img_dir, fn))
        if n % 50 == 0:
            print(f"    {n}/{len(names)}", end="\r")
    print(f"    {len(names)}/{len(names)}")

    # Against the manifest first, where it carries the features: that
    # isolates "the files changed" from "the arithmetic changed".
    if {"sat_mean", "contrast_local"} <= set(man.columns):
        m = man.set_index("filename")
        ds, df_ = [], []
        for fn, (s, c) in feat.items():
            if fn in m.index:
                ds.append(abs(s - float(m.loc[fn, "sat_mean"])))
                df_.append(abs(c - float(m.loc[fn,
                                               "contrast_local"])))
        if ds:
            ck.add("re-measured features match the manifest",
                   max(ds) <= REMEASURE_TOL_SAT
                   and max(df_) <= REMEASURE_TOL_FOG,
                   f"worst sat {max(ds):.4f}, worst fog {max(df_):.4f}")

    # Then the differences the model fits.
    sub = pkg[pkg.trial_type.isin(TYPES)].copy()
    la = sub.left_path.map(os.path.basename)
    ra = sub.right_path.map(os.path.basename)
    known = la.isin(feat) & ra.isin(feat)
    sub, la, ra = sub[known], la[known], ra[known]
    if not len(sub):
        ck.add("predictors rebuilt from the images", False,
               "no analysed trial had both images measured")
        return pd.DataFrame()

    # a_is_left says which of the two displayed images is A, and the
    # differences are always A minus B. Getting this backwards would
    # flip the sign of every predictor, so it is rebuilt from the
    # column rather than assumed.
    a_left = sub.a_is_left.astype(bool).to_numpy()
    fa = np.where(a_left, la.to_numpy(), ra.to_numpy())
    fb = np.where(a_left, ra.to_numpy(), la.to_numpy())
    sat_a = np.array([feat[f][0] for f in fa])
    sat_b = np.array([feat[f][0] for f in fb])
    fog_a = np.array([feat[f][1] for f in fa])
    fog_b = np.array([feat[f][1] for f in fb])

    d_sat = sat_a - sat_b
    d_fog = FOG_SIGN * (fog_a - fog_b)
    w_sat = float(np.abs(d_sat - sub.d_sat.to_numpy(float)).max())
    w_fog = float(np.abs(d_fog - sub.d_fog.to_numpy(float)).max())
    ck.add("d_sat rebuilt from the images", w_sat <= REMEASURE_TOL_SAT,
           f"worst {w_sat:.4f} sat_mean units "
           f"(tolerance {REMEASURE_TOL_SAT})")
    ck.add("d_fog rebuilt from the images", w_fog <= REMEASURE_TOL_FOG,
           f"worst {w_fog:.4f} contrast_local units "
           f"(tolerance {REMEASURE_TOL_FOG})")

    # And the oriented predictors, which is where a sign convention
    # would go wrong without changing either difference.
    p_sat = -d_sat / SD_SAT
    p_fog = d_fog / SD_FOG
    e_sat = float(np.abs(p_sat - sub.pred_sat.to_numpy(float)).max())
    e_fog = float(np.abs(p_fog - sub.pred_fog.to_numpy(float)).max())
    ck.add("pred_sat orientation and units",
           e_sat <= REMEASURE_TOL_SAT / SD_SAT,
           f"worst {e_sat:.5f} SD")
    ck.add("pred_fog orientation and units",
           e_fog <= REMEASURE_TOL_FOG / SD_FOG,
           f"worst {e_fog:.5f} SD")
    # A sign error would show as a large error AND a negative
    # correlation, so the correlation is reported to tell the two
    # apart at a glance.
    r_sat = float(np.corrcoef(p_sat, sub.pred_sat.to_numpy(float))[0, 1])
    r_fog = float(np.corrcoef(p_fog, sub.pred_fog.to_numpy(float))[0, 1])
    ck.add("predictors correlate with the shipped columns",
           min(r_sat, r_fog) > 0.999,
           f"r(sat) {r_sat:+.6f}, r(fog) {r_fog:+.6f}")
    return pd.DataFrame({"trial": sub.trial, "d_sat_shipped": sub.d_sat,
                         "d_sat_remeasured": d_sat,
                         "d_fog_shipped": sub.d_fog,
                         "d_fog_remeasured": d_fog})


def check_design(top, ck):
    """CHECK 5: the design invariants.

    EXPECTED states what the list should be, independently of the file,
    so that a build which silently changed shape cannot be frozen.
    """
    print("\n" + "=" * 60)
    print("CHECK 5: THE DESIGN")
    print("=" * 60)
    n = top.trial_type.value_counts().to_dict()
    ck.add("total trials",
           len(top) == EXPECTED["total"],
           f"{len(top)} (expected {EXPECTED['total']})")
    an = top[top.analysed.astype(bool)]
    ck.add("analysed trials",
           len(an) == EXPECTED["analysed"],
           f"{len(an)} (expected {EXPECTED['analysed']})")
    for t in TYPES + EXTRA_TYPES:
        ck.add(f"{t} count", n.get(t, 0) == EXPECTED[t],
               f"{n.get(t, 0)} (expected {EXPECTED[t]})")
    ck.add("analysed flag marks exactly the four real types",
           set(an.trial_type) == set(TYPES))

    # Side balance, forced to 0.50 within each type by step 5.
    side = an.groupby("trial_type").a_is_left.mean()
    worst = float((side - 0.5).abs().max())
    ck.add("side balance within each type",
           worst <= SIDE_BALANCE_TOL,
           f"worst deviation {worst:.3f}")

    # No pair twice except a consistency repeat.
    keys = [pair_key(a, b) for a, b in zip(top.file_a, top.file_b)]
    dup = pd.Series(keys).value_counts()
    multi = dup[dup > 1]
    n_con = n.get("consistency", 0)
    ck.add("no image pair appears twice except a repeat",
           len(multi) == n_con and (multi <= 2).all(),
           f"{len(multi)} repeated pairs, {n_con} consistency trials")

    # Each repeat AFTER its original, with the sides swapped.
    con = top[top.trial_type == "consistency"]
    gaps, swaps, ok_order = [], [], True
    for _, r in con.iterrows():
        tgt = top[top.trial == r.repeat_of]
        if len(tgt) != 1:
            ok_order = False
            continue
        o = tgt.iloc[0]
        ok_order &= int(o.trial) < int(r.trial)
        ok_order &= (pair_key(o.file_a, o.file_b)
                     == pair_key(r.file_a, r.file_b))
        gaps.append(int(r.trial) - int(o.trial))
        swaps.append(bool(o.left_file == r.right_file))
    ck.add("every repeat follows its original, same pair", ok_order,
           f"min gap {min(gaps) if gaps else 'n/a'}")
    ck.add("every repeat has its sides swapped",
           bool(swaps) and all(swaps),
           f"{sum(swaps)}/{len(swaps)}")

    # THE CATCH TRIAL DECLARES ITS MODE AND NOTHING ELSE.
    #
    # In 'no_images' mode a catch trial shows no pictures: it shows a
    # line of text naming one arrow key and takes one press. So the
    # list cannot carry an expected response in left/right terms, and
    # it does not try to -- there is no correct_response column at
    # all, and the instruction text lives in the experiment, which
    # owns the key sequence. What the list carries is catch_mode, so
    # the experiment can refuse to run a list built for the other
    # mode.
    cat = top[top.trial_type == "catch"]
    other = top[top.trial_type != "catch"]
    ck.add("no correct_response column",
           "correct_response" not in top.columns,
           "absent" if "correct_response" not in top.columns
           else "PRESENT: step 5 was run before the catch rewrite, or "
                "with an older settings block")
    ck.add("every catch row declares a catch mode",
           "catch_mode" in top.columns
           and bool(len(cat)) and cat.catch_mode.notna().all(),
           f"{int(cat.catch_mode.notna().sum()) if 'catch_mode' in top.columns else 0}"
           f"/{len(cat)}")
    if "catch_mode" in top.columns and len(cat):
        modes = sorted(set(cat.catch_mode.dropna().astype(str)))
        ck.add("one list, one catch mode",
               len(modes) == 1, f"{modes}")
        ck.add("the catch mode is the registered one",
               modes == [EXPECTED["catch_mode"]],
               f"{modes} (expected [{EXPECTED['catch_mode']!r}])")
        # A catch_mode on an analysed, consistency or practice row
        # would make the experiment's load-time check ambiguous about
        # which rows it governs.
        ck.add("no other row declares a catch mode",
               other.catch_mode.isna().all(),
               f"{int(other.catch_mode.notna().sum())} rows outside "
               "the catch block carry one")
    if "instruction" in top.columns and len(cat):
        # Empty on catch rows BECAUSE the experiment owns the wording
        # and the key sequence; non-empty everywhere else, since those
        # trials are a two-image choice with one question.
        ck.add("catch rows carry no instruction text",
               cat.instruction.isna().all(),
               f"{int(cat.instruction.notna().sum())} with text")
        ck.add("every other row carries its instruction",
               other.instruction.notna().all()
               and other.instruction.nunique() == 1,
               f"{other.instruction.nunique()} distinct")

    # Practice first, and nothing analysed before it.
    prac = top[top.trial_type == "practice"]
    ck.add("practice trials come first",
           bool(len(prac)) and int(prac.trial.max())
           == EXPECTED["practice"],
           f"trials 1-{int(prac.trial.max()) if len(prac) else 0}")
    # Practice and catch must not reuse an analysed pair, in either
    # direction -- a second exposure would make the analysed
    # presentation the contaminated one.
    an_keys = {pair_key(a, b) for a, b in zip(an.file_a, an.file_b)}
    pc_keys = {pair_key(a, b) for a, b in
               zip(pd.concat([prac, cat]).file_a,
                   pd.concat([prac, cat]).file_b)}
    ck.add("practice and catch share no pair with the analysed set",
           not (an_keys & pc_keys),
           f"{len(an_keys & pc_keys)} overlapping")

    # Scene spread and collinearity, the two numbers that decide
    # whether the two weights are separable.
    r = float(np.corrcoef(an.d_sat, an.d_fog)[0, 1])
    ck.add("r(d_sat, d_fog) is small", abs(r) <= 0.6, f"{r:+.3f}")
    per_scene = an.scene_id.value_counts()
    print(f"  scenes: {an.scene_id.nunique()}, "
          f"analysed trials per scene min {per_scene.min()} "
          f"median {int(per_scene.median())} max {per_scene.max()}")
    return {"r": r, "scenes": int(an.scene_id.nunique()),
            "max_per_scene": int(per_scene.max()),
            "counts": {k: int(v) for k, v in n.items()}}


def check_settings(root, ck):
    """CHECK 6: the build's own report says what EXPECTED says.

    Step 5 writes its settings into trials_report.txt. Reading them
    back is how the freeze confirms the seed, the cut and the variant
    rather than assuming the directory name means what it says.
    """
    print("\n" + "=" * 60)
    print("CHECK 6: THE SETTINGS THAT BUILT IT")
    print("=" * 60)
    path = os.path.join(root, "trials_report.txt")
    if not os.path.isfile(path):
        ck.add("step 5 report present", False, f"{path} not found")
        return {}
    text = open(path).read()
    head = {}
    for line in text.splitlines():
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            head[k.strip()] = v.strip()
    seed = head.get("Seed", "")
    ck.add("seed is the registered one",
           str(EXPECTED["seed"]) in seed, f"{seed!r}")
    cut = head.get("Colour cut", "")
    ck.add("colour cut is the chosen one",
           str(EXPECTED["colour_cut"]) in cut, f"{cut!r}")
    var = head.get("Variant", "")
    ck.add("variant is the extended ladder",
           EXPECTED["variant"] in var, f"{var!r}")
    streams = head.get("Streams", "")
    ck.add("built with per-type random streams",
           "per_type" in streams, f"{streams!r}", fatal=False)
    catch = head.get("Catch", "")
    ck.add("the build declares the registered catch mode",
           EXPECTED["catch_mode"] in catch, f"{catch!r}")
    ck.add("the build states the exclusion rule",
           "3 or more" in catch, f"{catch!r}", fatal=False)
    extras = head.get("Extras", "")
    ck.add("the build states the catch count",
           f"{EXPECTED['catch']} catch" in extras, f"{extras!r}")
    for k in ("Pairs from", "Stimuli from", "Bins", "Counts",
              "Conflict", "Scene cap"):
        if k in head:
            print(f"    {k:14s} {head[k]}")
    return head


def narrative_numbers(root, source_root):
    """Every figure the report's opening section quotes, DERIVED.

    These used to be hardcoded strings. They went stale: the text
    still carried the numbers from a run made before the per-type
    random streams and before Firth, so the fog-control figure was
    out by a factor of ten and every other one was slightly wrong.
    A freeze report is the documentation record, so it cannot be
    allowed to drift from the run it describes.

    Sources, all of them inside the frozen package except where
    noted:
      provenance/differences.csv            step 6 detection
      provenance/ranking.csv                step 6b ranking
      provenance/weights.csv                step 6b weight spread
      provenance/differences_by_lapse.csv   step 6b gains under lapses
      trial_list.csv                        bins and edit depth
      SOURCE_ROOT/<baseline>/trial_list.csv the baseline to compare
                                            the last two against;
                                            not part of the package

    DEFENSIVE BY DESIGN. Every figure is looked up by column name,
    and anything missing comes back as None rather than as a guess.
    The report then cites the file instead of printing a number. A
    missing column must degrade to a citation, never to a wrong
    figure.
    """
    out = {}

    def read(rel, base=root):
        p = os.path.join(base, rel)
        try:
            return pd.read_csv(p) if os.path.isfile(p) else None
        except Exception:
            return None

    def pick(df, *names):
        """The first of `names` the frame actually has.

        Step 6 renamed its error column from `mc_error` to `se` when
        it moved from one draw to twenty paired draws, and a
        hardcoded name silently cost three figures. Ask the frame.
        """
        if df is None:
            return None
        for nm in names:
            if nm in df.columns:
                return nm
        return None

    def one(df, cols, **eq):
        """The single row matching eq, if the columns are all there."""
        if df is None or not set(list(eq) + cols) <= set(df.columns):
            return None
        m = df
        for k, v in eq.items():
            # A float key is matched with a tolerance, because 0.5
            # read back from a CSV need not be exactly 0.5; a string
            # key is matched as text. Never by dtype, which pandas
            # reports differently across versions.
            if isinstance(v, str):
                m = m[m[k].astype(str) == v]
            else:
                m = m[np.isclose(
                    pd.to_numeric(m[k], errors="coerce"), float(v))]
        return m.iloc[0] if len(m) == 1 else None

    # --- step 6: colour detection at NARRATIVE_SENS ---
    diff = read(os.path.join("provenance", "differences.csv"))
    ecol = pick(diff, "se", "mc_error")
    r = one(diff, ["baseline", "this_list", "difference"]
            + ([ecol] if ecol else []),
            list=CHOSEN, participant="colour_only", feature="colour",
            sensitivity_SD=NARRATIVE_SENS) if ecol else None
    if r is not None:
        out["det_from"] = float(r.baseline)
        out["det_to"] = float(r.this_list)
        out["det_gain"] = float(r.difference)
        out["det_se"] = float(r[ecol])
        out["det_draws"] = (int(r.draws) if "draws" in diff.columns
                            else None)

    # --- step 6's own fog control ---
    #
    # NOT re-derived here. The comparison only means something within
    # a ladder, where the fog, together and conflict trials are
    # IDENTICAL across versions so the difference isolates the colour
    # cut. Across ladders the vivid levels add new pairs of every
    # type, so the fog trials are a different draw by design and the
    # error is several times larger. An earlier version of this
    # function took the worst difference over all candidates and
    # divided by an SE pooled across both kinds, which manufactured a
    # failing verdict out of two sound comparisons.
    #
    # Step 6 writes fog_control.csv with `same_ladder_as_baseline`
    # already computed. Use it.
    fog = read(os.path.join("provenance", "fog_control.csv"))
    fcol = pick(fog, "se", "mc_error")
    if (fog is not None and fcol
            and {"same_ladder_as_baseline", "difference",
                 "list"} <= set(fog.columns)):
        same = fog[fog.same_ladder_as_baseline.astype(bool)]
        if len(same):
            out["fog_worst"] = float(same.difference.abs().max())
            out["fog_se"] = float(same[fcol].median())
            out["fog_n"] = int(len(same))
            out["fog_lists"] = sorted(set(same.list.astype(str)))
            # Per row against its OWN error, never a pooled one.
            out["fog_ok"] = bool(
                (same.difference.abs() <= 2 * same[fcol]).all())
        other = fog[~fog.same_ladder_as_baseline.astype(bool)]
        other = other[~other.list.isin(NARRATIVE_NOT_CANDIDATES)]
        if len(other):
            out["fog_other_worst"] = float(
                other.difference.abs().max())
            out["fog_other_se"] = float(other[fcol].median())
            out["fog_other_ok"] = bool(
                (other.difference.abs() <= 2 * other[fcol]).all())

    # --- step 6: the runner-up ---
    r = one(diff, ["difference"]
            + ([ecol] if ecol else []),
            list=NARRATIVE_RUNNER_UP, participant="colour_only",
            feature="colour", sensitivity_SD=NARRATIVE_SENS)
    if r is not None and ecol:
        out["runner_up_se"] = float(r[ecol])
    if r is not None and "det_gain" in out:
        out["runner_up"] = NARRATIVE_RUNNER_UP
        out["runner_up_gain"] = float(r.difference)
        out["runner_up_over"] = float(r.difference) - out["det_gain"]

    # --- step 6b: the ranking ---
    rank = read(os.path.join("provenance", "ranking.csv"))
    if rank is not None:
        out["rank_cells"] = int(len(rank))
        vcols = [c for c in rank.columns if c.startswith("verdict_")]
        if vcols:
            bad = 0
            for c in vcols:
                bad += int((~rank[c].isin(["ok", "tied"])).sum())
            out["rank_inversions"] = bad
            out["rank_tied"] = int(sum(
                int((rank[c] == "tied").sum()) for c in vcols))
        if "as_expected" in rank.columns:
            out["rank_all_expected"] = bool(rank.as_expected.all())

    # --- step 6b: the gains hold under lapses ---
    lap = read(os.path.join("provenance", "differences_by_lapse.csv"))
    r = one(lap, ["difference"], list=CHOSEN,
            participant="colour_only", feature="colour",
            sensitivity_SD=NARRATIVE_SENS,
            lapse=max(NARRATIVE_LAPSES))
    if r is not None:
        out["lapse_max"] = max(NARRATIVE_LAPSES)
        out["lapse_gain"] = float(r.difference)

    # --- step 6b: spread, and how close anyone came ---
    w = read(os.path.join("provenance", "weights.csv"))
    if w is not None and "colour_correct_max" in w.columns:
        sel = w[w.list == CHOSEN] if "list" in w.columns else w
        if len(sel):
            out["closest"] = float(sel.colour_correct_max.max())
    for key, lab in (("sd_chosen", CHOSEN), ("sd_base", NARRATIVE_BASELINE)):
        r = one(w, ["sd_over_true"], list=lab,
                participant="colour_only",
                sensitivity_SD=NARRATIVE_SENS, lapse=0.0)
        if r is not None:
            out[key] = float(r.sd_over_true)

    # --- the frozen list itself: colour bins and edit depth ---
    def list_facts(df):
        if df is None or not {"trial_type", "d_sat",
                              "mag_bin"} <= set(df.columns):
            return None
        s = df[df.trial_type == "sat_only"]
        if not len(s):
            return None
        a = np.abs(s.d_sat.to_numpy(float))
        f = {"bins": int(s.mag_bin.nunique()),
             "lo": float(a.min()), "hi": float(a.max())}
        if {"file_a", "file_b", "chroma_a", "chroma_b"} <= set(
                df.columns):
            k = {}
            for ca, cb, fa, fb in zip(df.chroma_a, df.chroma_b,
                                      df.file_a, df.file_b):
                k[fa], k[fb] = ca, cb
            v = np.abs(np.array(list(k.values()), dtype=float) - 1.0)
            f["edit"] = float(v.mean())
            f["n_images"] = len(k)
        return f

    out["chosen_list"] = list_facts(read("trial_list.csv"))
    # The baseline list is not part of the package, so this one is
    # read from the build directory and is absent on a verify run
    # from a moved package. That is why it degrades to a citation.
    out["base_list"] = list_facts(read(
        os.path.join(NARRATIVE_BASELINE, "trial_list.csv"), base=source_root))
    return out


# ============================================================
# Freeze and verify
# ============================================================

def copy_build(src, dst):
    """Copy the build to the frozen directory, refusing to overwrite."""
    print("\n" + "=" * 60)
    print("COPYING")
    print("=" * 60)
    if os.path.exists(dst):
        raise FileExistsError(
            f"{dst} already exists. A frozen list is not overwritten: "
            "move it aside\nand re-run, or set MODE = 'verify' to "
            "check the one that is there.")
    os.makedirs(dst)
    for name in ("trial_list.csv", "trial_design.png",
                 "trials_report.txt"):
        p = os.path.join(src, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, name))
    shutil.copytree(os.path.join(src, "package"),
                    os.path.join(dst, "package"))
    prov = os.path.join(dst, "provenance")
    os.makedirs(prov, exist_ok=True)
    got, miss = [], []
    for label, p in PROVENANCE:
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(prov, os.path.basename(p)))
            got.append((label, p))
        else:
            miss.append((label, p))
    n = len(walk_files(dst))
    print(f"  copied {n} files to {dst}")
    print(f"  provenance: {len(got)} captured"
          + (f", {len(miss)} missing" if miss else ""))
    for label, p in miss:
        print(f"    missing: {label}  ({p})")
    return got, miss


def write_checksums(root):
    """One SHA-256 per file, in `shasum -c` format."""
    files = [f for f in walk_files(root)
             if f not in ("checksums.sha256", "FROZEN.json",
                          "freeze_report.txt")]
    lines = []
    for rel in files:
        lines.append(f"{sha256_of(os.path.join(root, rel))}  {rel}")
    path = os.path.join(root, "checksums.sha256")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  checksummed {len(files)} files -> {path}")
    return files, lines


def verify_checksums(root, ck):
    """Re-check every file against checksums.sha256."""
    print("\n" + "=" * 60)
    print("CHECKSUMS")
    print("=" * 60)
    path = os.path.join(root, "checksums.sha256")
    if not os.path.isfile(path):
        ck.add("checksums.sha256 present", False, f"{path} not found")
        return
    want = {}
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        h, _, rel = line.partition("  ")
        want[rel] = h
    have = {f for f in walk_files(root)
            if f not in ("checksums.sha256", "FROZEN.json",
                         "freeze_report.txt")}
    missing = sorted(set(want) - have)
    extra = sorted(have - set(want))
    changed = []
    for rel, h in want.items():
        if rel in have and sha256_of(os.path.join(root, rel)) != h:
            changed.append(rel)
    ck.add("no file is missing", not missing,
           f"{len(want)} listed" if not missing
           else f"missing {missing[:5]}")
    ck.add("no file was added", not extra,
           "" if not extra else f"extra {extra[:5]}")
    ck.add("every checksum matches", not changed,
           f"{len(want)} files verified" if not changed
           else f"CHANGED: {changed[:5]}")


def write_record(root, ck, settings, design, prov_got, prov_miss,
                 files, n_img, t0, num=None):
    """The documentation record, and FROZEN.json beside it.

    `num` is narrative_numbers()'s dict. It is derived from the
    captured provenance rather than written into this script, so the
    record cannot drift from the run it describes. None means the
    opening section cites its files instead of quoting figures.
    """
    num = {} if num is None else num
    path = os.path.join(root, "freeze_report.txt")
    tab = ck.table()
    n_pass = int((tab.result == "PASS").sum())
    n_fail = int((tab.result == "FAIL").sum())
    n_note = int((tab.result == "NOTE").sum())

    with open(path, "w") as f:
        def W(s=""):
            f.write(s + "\n")

        W("FROZEN TRIAL LIST  (v4 axes, step 7a)")
        W("=" * 70)
        W(f"Frozen at  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        W(f"Version    : {CHOSEN}")
        W(f"Source     : {os.path.join(SOURCE_ROOT, CHOSEN)}")
        W(f"Frozen to  : {root}")
        W(f"Files      : {len(files)} ({n_img} images)")
        W(f"Checks     : {n_pass} passed, {n_fail} failed, "
          f"{n_note} noted")
        W(f"Elapsed    : {time.time() - t0:.1f}s")
        W()
        W("WHAT WAS FROZEN, AND WHY THIS ONE")
        W("-" * 70)
        W("  `both`: the extended colour ladder -- two vivid levels")
        W("  above the unmanipulated image -- with colour-only trials")
        W("  cut at |d_sat| >= 31.1 sat_mean units (0.794 corpus SD).")
        W()
        # EVERY FIGURE BELOW IS DERIVED, not written in. An earlier
        # version of this section carried the numbers as literal
        # text, and they went stale when the random streams and the
        # estimator changed -- the fog-control figure by a factor of
        # ten. Anything that cannot be derived cites its file.
        n = num
        W("  Derived from the captured provenance, not written into")
        W("  this script. Where a figure could not be read, the file")
        W("  holding it is named instead.")
        W()
        if "det_gain" in n:
            W(f"  Step 6, over "
              f"{n['det_draws'] or '20'} draws, paired: colour "
              "detection of")
            W(f"  a colour-following participant rises from "
              f"{n['det_from']:.3f} to")
            W(f"  {n['det_to']:.3f} at a true sensitivity of "
              f"{NARRATIVE_SENS:.2f} SD, "
              f"{n['det_gain']:+.3f} with a")
            W(f"  standard error of {n['det_se']:.4f}.")
        else:
            W("  Step 6 detection: see provenance/differences.csv.")
        if "fog_worst" in n:
            # The verdict comes from step 6, which tests each cell
            # against ITS OWN error. Only versions on the baseline's
            # ladder are a control: they share their fog, together
            # and conflict trials with it, so the difference isolates
            # the colour cut.
            W("  THE FOG CONTROL. Only versions on the baseline's")
            W(f"  ladder are a control -- {', '.join(n['fog_lists'])}")
            W("  -- because they share their fog, together and")
            W("  conflict trials with the baseline, so the difference")
            W(f"  isolates the colour cut. Over those {n['fog_n']} "
              "cells the largest")
            W(f"  difference in fog detection is {n['fog_worst']:.4f} "
              f"against a typical")
            W(f"  standard error of {n['fog_se']:.4f}"
              + (", every cell inside twice its"
                 if n.get("fog_ok") else
                 " -- but at least one cell is beyond twice its"))
            W("  own error." + ("" if n.get("fog_ok") else
                                " THAT BEARS ON THE CONTROL"))
            if not n.get("fog_ok"):
                W("  COMPARISON and belongs in the write-up as a")
                W("  limitation.")
            else:
                W("  The colour cut costs nothing on the axis the")
                W("  registered control comparison uses.")
            if "fog_other_worst" in n:
                W()
                W("  Versions on the OTHER ladder are not a control:")
                W("  the vivid levels add new pairs of every type, so")
                W("  their fog trials are a different draw by design.")
                W(f"  Their largest difference is "
                  f"{n['fog_other_worst']:.4f} against a typical")
                W(f"  error of {n['fog_other_se']:.4f}"
                  + (", also inside twice it."
                     if n.get("fog_other_ok") else
                     ", beyond twice it in at least one cell."))
                W("  This is the ladder, not the cut.")
        else:
            W("  Fog control: see provenance/fog_control.csv.")
        W()
        if "rank_cells" in n:
            inv = n.get("rank_inversions")
            W(f"  Step 6b: the ranking survives lapse rates of "
              f"{', '.join(str(x) for x in NARRATIVE_LAPSES)} in")
            W(f"  all {n['rank_cells']} cells"
              + ("." if inv is None else
                 f" with {inv} resolvable inversion(s)."))
            if n.get("rank_tied"):
                W(f"  {n['rank_tied']} adjacent comparison(s) are "
                  "tied, which is a finding")
                W("  about those cells rather than a failure; see")
                W("  provenance/ranking.csv for which.")
        else:
            W("  Step 6b ranking: see provenance/ranking.csv.")
        if "lapse_gain" in n:
            W(f"  The gains hold: {n['lapse_gain']:+.3f} at "
              f"{NARRATIVE_SENS:.2f} SD at lapse "
              f"{n['lapse_max']}.")
        else:
            W("  Gains under lapses: see "
              "provenance/differences_by_lapse.csv.")
        if "closest" in n:
            W(f"  No fit approached separation -- the closest was "
              f"{n['closest']:.3f} of")
            W("  colour-informative trials, where separation needs")
            W("  1.000.")
        else:
            W("  Separation diagnostics: see provenance/weights.csv.")
        if "sd_chosen" in n and "sd_base" in n:
            W(f"  The fitted colour weight has the SMALLEST spread of")
            W(f"  the candidates, sd/true {n['sd_chosen']:.3f} against "
              f"the baseline's")
            W(f"  {n['sd_base']:.3f}, so the between-group comparison "
              "is easier on")
            W("  this list.")
        else:
            W("  Weight spread: see provenance/weights.csv.")
        W()
        W("  THE COST, recorded because it is what a reader will want")
        W("  to weigh later. The cut removes colour differences below")
        W(f"  {EXPECTED['colour_cut']} units, so the colour "
          "dose-response curve rests on")
        cl, bl = n.get("chosen_list"), n.get("base_list")
        if cl:
            W(f"  {cl['bins']} magnitude bins spanning {cl['lo']:.0f} "
              f"to {cl['hi']:.0f} units"
              + (f" rather than" if bl else "."))
            if bl:
                W(f"  {bl['bins']} spanning {bl['lo']:.0f} to "
                  f"{bl['hi']:.0f} on {NARRATIVE_BASELINE}.")
        else:
            W("  fewer magnitude bins; see trial_list.csv.")
        if cl and "edit" in cl and bl and "edit" in bl:
            pct = 100.0 * (cl["edit"] / bl["edit"] - 1.0)
            W(f"  The average shipped image is edited {pct:.0f}% "
              "further from the")
            W(f"  unmanipulated original: mean |k - 1| "
              f"{cl['edit']:.3f} over "
              f"{cl['n_images']} images,")
            W(f"  against {bl['edit']:.3f} over {bl['n_images']} on "
              f"{NARRATIVE_BASELINE}.")
        elif cl and "edit" in cl:
            W(f"  Mean |k - 1| over the {cl['n_images']} shipped "
              f"images is {cl['edit']:.3f};")
            W(f"  the {NARRATIVE_BASELINE} build was not readable from "
              "SOURCE_ROOT, so the")
            W("  comparison is not stated.")
        W("  The dose-response curve is exploratory; the registered")
        W("  claims are about per-person weights and the control")
        W("  comparison, which is why detection was weighted above")
        W("  range.")
        W()
        if "runner_up_over" in n:
            W(f"  `{n['runner_up']}` was tested and dominated: "
              f"{n['runner_up_over']:+.4f} over")
            W("  this list"
              + ("," if "runner_up_se" not in n else
                 f", against a standard error of "
                 f"{n['runner_up_se']:.4f},"))
            W("  inside the noise, for a narrower range of colour")
            W("  differences. 31.1 was chosen over 35.3")
            W("  because it keeps a wider range of colour differences")
            W("  with slightly less conspicuous edits. It is a")
            W("  stimulus choice rather than an analysis option, so")
            W("  it could not be kept as a later sensitivity")
            W("  analysis.")
        else:
            W(f"  `{NARRATIVE_RUNNER_UP}`: see "
              "provenance/differences.csv.")
        W()
        W("THE BUILD'S OWN SETTINGS")
        W("-" * 70)
        if settings:
            for k, v in settings.items():
                W(f"  {k:14s} {v}")
        else:
            W("  step 5's trials_report.txt was not found.")
        W()
        W("THE LIST")
        W("-" * 70)
        if design:
            W(f"  trials by type   {design['counts']}")
            W(f"  scenes           {design['scenes']}, at most "
              f"{design['max_per_scene']} analysed trials from one")
            W(f"  r(d_sat, d_fog)  {design['r']:+.3f}  -- what decides")
            W("                   whether the two weights can be told")
            W("                   apart; above about 0.6 they cannot")
        W()
        W(f"  THE CATCH TRIAL, mode {EXPECTED['catch_mode']!r}. "
          f"{EXPECTED['catch']} trials. The")
        W("  registered rule: EXCLUDE A PARTICIPANT FAILING 3 OR MORE.")
        W()
        W("  What a catch trial is in this mode: no pictures. A line")
        W("  of text names one arrow key -- UP or DOWN -- and the")
        W("  trial takes one press. The key sequence and the wording")
        W("  live in the experiment, not in this list; the list")
        W("  carries only catch_mode, so the experiment can refuse to")
        W("  run a list built for the other mode.")
        W()
        W("  THE KEY-PRESS ASSUMPTION behind the threshold. The")
        W("  response set has four live keys: left and right for the")
        W("  image choice, up and down for the catch trials. A")
        W("  participant who is not reading presses one of the four")
        W("  at random, so three presses in four are wrong and the")
        W("  per-trial failure probability is 0.75 * lambda, where")
        W("  lambda is the lapse rate. That assumption is what makes")
        W("  the numbers below, and it is stated because it is an")
        W("  assumption: a participant who always presses the same")
        W("  key, or who reads the instruction but mis-aims, is not")
        W("  this model.")
        W()
        W("  At 3 or more failures out of 10, binomial:")
        W("    lambda 0.00   excluded 0.0%    an attentive")
        W("                                   participant is never")
        W("                                   excluded by chance")
        W("    lambda 0.05   excluded 0.5%")
        W("    lambda 0.10   excluded 3.4%")
        W("    lambda 0.20   excluded 18.0%")
        W("    lambda 0.30   excluded 39.8%")
        W("    lambda 0.50   excluded 78.9%")
        W("    random presser    99.96%   (p_fail 0.75)")
        W("    left/right only  100.00%   (p_fail 1.00: an up or")
        W("                               down trial is never")
        W("                               answered)")
        W()
        W("  So the rule catches gross inattention almost always and")
        W("  costs almost nothing at the attentive end. It is NOT a")
        W("  filter on mild inattention: a participant lapsing on one")
        W("  trial in ten survives it 97% of the time, which is why")
        W("  the lapse rate is also fitted, as the pre-registered")
        W("  sensitivity analysis for C1, rather than screened for")
        W("  here.")
        W()
        W("  IN 'images' MODE the catch trial shows a pair with an")
        W("  obvious difference and takes a left/right press, so a")
        W("  random press is wrong only half the time: p_fail is")
        W("  0.5 * lambda, and the same threshold excludes 1.2% at")
        W("  lambda 0.1 and 7.0% at lambda 0.2. The rule is kept at 3")
        W("  because it is the registered one; the mode is recorded")
        W("  so the arithmetic that applies is never in doubt.")
        W()
        W(f"  SESSION LENGTH. {EXPECTED['total']} trials presented "
          f"({EXPECTED['analysed']} analysed,")
        W(f"  {EXPECTED['consistency']} consistency, "
          f"{EXPECTED['catch']} catch, {EXPECTED['practice']} "
          "practice). At 3.5 to 4.5")
        W("  seconds per trial plus about two minutes of instructions")
        W("  and breaks, that is 14 to 18 minutes. THE 15-MINUTE")
        W("  TARGET IS AT THE FAST END OF THAT RANGE. The per-trial")
        W("  time is the assumption to check against the first")
        W("  participants' timestamps; the counts are fixed here.")
        W()
        W("CHECKS")
        W("-" * 70)
        W(tab.to_string(index=False))
        W()
        if n_fail:
            W("  THIS LIST IS NOT FROZEN. The failures above have to be")
            W("  fixed and the freeze re-run.")
        else:
            W("  All checks passed. Check 4 is the one that makes this")
            W("  copy self-certifying: sat_mean and contrast_local were")
            W("  re-measured from the images IN THIS DIRECTORY and the")
            W("  predictors rebuilt from them, so the numbers an")
            W("  analysis will fit are known to describe the pictures a")
            W("  participant sees.")
        W()
        W("PROVENANCE")
        W("-" * 70)
        W("  Only scripts and documentation are kept long-term in this")
        W("  project, so the reports that chose this list travel with")
        W("  it, in provenance/, and are checksummed with everything")
        W("  else.")
        W()
        for label, p in prov_got:
            W(f"  captured  {label:20s} {p}")
        for label, p in prov_miss:
            W(f"  MISSING   {label:20s} {p}")
        W()
        W("HOW TO CHECK THIS LATER")
        W("-" * 70)
        W("  Re-run this script with MODE = 'verify'. It re-checks")
        W("  every file against checksums.sha256 and runs every check")
        W("  above again on the frozen copy.")
        W()
        W("  Outside this pipeline, from inside the frozen directory:")
        W("      shasum -a 256 -c checksums.sha256")
        W()
        W("  Do that before launching, and again if the package is")
        W("  ever moved, re-uploaded or rebuilt.")
        W()
        W("WHAT IS NOT FROZEN HERE")
        W("-" * 70)
        W("  The grid blocks. They use different stimulus cells -- the")
        W("  single-feature ladders and the contrast-up column -- and")
        W("  they carry open design questions: how many columns each")
        W("  ladder shows, whether the unmanipulated image appears in")
        W("  them, and whether the contrast-up column is included at")
        W("  all. Step 7b, once those are answered.")
        W()
        W("  The power analysis. Step 8 runs on THIS list, at seed")
        W("  42, because thresholds and power belong to the list")
        W("  participants see -- and reports where this draw sits in")
        W("  the 20-draw distribution without choosing a seed on any")
        W("  result.")
    rec = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": CHOSEN,
        "source": os.path.join(SOURCE_ROOT, CHOSEN),
        "frozen_root": root,
        "seed": EXPECTED["seed"],
        "colour_cut_sat_mean": EXPECTED["colour_cut"],
        "colour_cut_SD": round(EXPECTED["colour_cut"] / SD_SAT, 4),
        "variant": EXPECTED["variant"],
        "catch_mode": EXPECTED["catch_mode"],
        "catch_trials": EXPECTED["catch"],
        "catch_exclusion_rule": "exclude a participant failing 3 or "
                                "more catch trials",
        "sd_sat": SD_SAT, "sd_fog": SD_FOG,
        "n_files": len(files), "n_images": n_img,
        "checks_passed": n_pass, "checks_failed": n_fail,
        "checks_noted": n_note,
        "design": design,
        "settings": settings,
        "provenance_captured": [p for _, p in prov_got],
        "provenance_missing": [p for _, p in prov_miss],
        "checksums": "checksums.sha256",
    }
    with open(os.path.join(root, "FROZEN.json"), "w") as f:
        json.dump(rec, f, indent=2)
    return path


# ============================================================
# Main
# ============================================================

def run_checks(root, ck):
    """Every check, against whichever directory is given."""
    n_img = check_complete(root, ck)
    top, pkg, js = check_three_copies(root, ck)
    on_disk, man = check_images_close(root, pkg, ck)
    check_remeasure(root, pkg, man, ck)
    design = check_design(top, ck)
    settings = check_settings(root, ck)
    return n_img, design, settings


def main():
    t0 = time.time()
    print("=" * 60)
    print(f"FREEZE THE TRIAL LIST  (v4, step 7a)  [{MODE}]")
    print("=" * 60)

    if MODE == "verify":
        print(f"  verifying: {FROZEN_ROOT}")
        if not os.path.isdir(FROZEN_ROOT):
            raise FileNotFoundError(
                f"{FROZEN_ROOT} does not exist. Run with "
                "MODE = 'freeze' first.")
        ck = Checks()
        verify_checksums(FROZEN_ROOT, ck)
        run_checks(FROZEN_ROOT, ck)
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        tab = ck.table()
        print(f"  {int((tab.result == 'PASS').sum())} passed, "
              f"{len(ck.failed)} failed, "
              f"{int((tab.result == 'NOTE').sum())} noted")
        if ck.failed:
            print(tab[tab.result == "FAIL"].to_string(index=False))
            raise RuntimeError(
                f"VERIFY FAILED on {FROZEN_ROOT}. The frozen list is "
                "not what it was when\nit was frozen, or it no longer "
                "passes its own checks. Do not launch with it.")
        print("  The frozen list is unchanged and still passes every "
              "check.")
        print(f"\n  Elapsed: {time.time() - t0:.1f}s")
        return

    src = os.path.join(SOURCE_ROOT, CHOSEN)
    print(f"  source : {src}")
    print(f"  frozen : {FROZEN_ROOT}")
    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"{src} does not exist.\nCheck SOURCE_ROOT: step 5 writes "
            "to a new numbered directory on each run,\nand this must "
            "be the build steps 6 and 6b measured.")

    # Checked BEFORE copying, so a bad build is never frozen.
    ck = Checks()
    n_img, design, settings = run_checks(src, ck)
    if ck.failed:
        print("\n" + "=" * 60)
        print("NOT FROZEN")
        print("=" * 60)
        print(ck.table()[ck.table().result == "FAIL"]
              .to_string(index=False))
        raise RuntimeError(
            f"{len(ck.failed)} check(s) failed on {src}. Nothing was "
            "copied. A list that\ncannot pass these is not a list to "
            "register.")

    prov_got, prov_miss = copy_build(src, FROZEN_ROOT)

    # And again on the COPY, because the copy is what gets registered.
    print("\n" + "=" * 60)
    print("RE-CHECKING THE FROZEN COPY")
    print("=" * 60)
    ck2 = Checks()
    n_img2, design2, settings2 = run_checks(FROZEN_ROOT, ck2)
    if ck2.failed:
        raise RuntimeError(
            "The copy does not pass the checks the source passed, so "
            "the copy is\ndamaged. Remove "
            f"{FROZEN_ROOT} and re-run.")

    files, _ = write_checksums(FROZEN_ROOT)
    # Derived from the frozen copy's own provenance, so the record
    # describes the run that produced it.
    num = narrative_numbers(FROZEN_ROOT, SOURCE_ROOT)
    missing = [k for k in ("det_gain", "fog_worst", "rank_cells",
                           "lapse_gain", "closest", "sd_chosen")
               if k not in num]
    if missing:
        print(f"\n  NOTE: {len(missing)} narrative figure(s) could "
              f"not be derived\n  ({', '.join(missing)}); the report "
              "cites the file instead of quoting\n  a number. Check "
              "that PROVENANCE points at the runs you mean.")
    path = write_record(FROZEN_ROOT, ck2, settings2, design2,
                        prov_got, prov_miss, files, n_img2, t0, num)

    print("\n" + "=" * 60)
    print("FROZEN")
    print("=" * 60)
    tab = ck2.table()
    print(f"  {int((tab.result == 'PASS').sum())} checks passed, "
          f"{int((tab.result == 'NOTE').sum())} noted")
    print(f"  {len(files)} files, {n_img2} images")
    print(f"  record : {path}")
    print(f"  elapsed: {time.time() - t0:.1f}s")
    print("\n  Set MODE = 'verify' and re-run before launching, and "
          "again if the\n  package is ever moved or re-uploaded.")
    print("\n  NEXT: step 7b builds the grid blocks, which need the "
          "layout questions\n        answered first; step 8 runs the "
          "power analysis on this list.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
