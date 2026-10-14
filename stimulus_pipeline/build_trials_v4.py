"""
build_trials_v4.py  (stimulus pipeline, v4 axes, step 5)

Samples the trials a participant actually sees, orders the session, and
writes a self-contained package ready to upload.

Where this sits
---------------
  step 3  build_stimulus_set_v4  the stimuli and the manifest
  step 4  analyse_pairs_v4       every usable pair -> all_pairs.csv
  step 5  this one               the trial list the experiment runs
  step 6  simulate_power_v4      is that trial list powered?

SELF-CONTAINED and lightweight: it reads all_pairs.csv and the rendered
images, nothing else. Step 4 has already classified every pair and
flagged the balanced conflict ones.

TWO TRIAL LISTS
---------------
PAIR_TABLES is a LIST: one entry per stimulus set. Each gets its own
trial list and its own package, built with identical settings and a
shared RANDOM_SEED, so the two sessions differ only in which fog the
images carry.

THE TRIAL COUNTS, AND WHY
-------------------------
40 saturation-only, 40 fog-only, 40 mixed (half together, half
conflict) = 120 analysed trials, as in v2.

40 is not arbitrary. This is a design in which EACH PARTICIPANT IS A
DATA POINT: the analysis estimates a per-person weight on each cue and
then asks how those weights differ between people -- whether someone
who describes their experience as foggy weighs the cues differently
from someone who describes it as lifeless. So the count is set by the
precision of the INDIVIDUAL estimate, not the group one. A proportion
from 40 trials has a standard error near sqrt(0.25/40) = 0.079, which
sits just below the expected spread between people. Below about 30 the
within-person noise starts to dominate and individual differences
become unreadable -- and individual differences are the whole reason a
small sample is worth running.

If piloting shows the session is too long, cut all three together, and
cut N_MIXED LAST: the conflict trials inside it carry the most
information, being the only ones that can show the crossover.

WHAT IS AND IS NOT ANALYSED
---------------------------
  analysed      the four real types only, 120 trials
  consistency   repeats of pairs already shown, sides swapped. EXCLUDED
                from the analysis on purpose: they are an INDEPENDENT
                quality check, and the agreement rate is what tells you
                whether a flat weight means "does not track these
                features" or "was not attending". A check that is also
                part of the analysis it is checking is no check at all.
  catch         the question is replaced by a task-irrelevant
                instruction, so the response tests attention
  practice      unanalysed by definition

Practice and catch pairs are drawn from pairs the analysed set never
uses, so no pair is ever seen twice except a consistency repeat -- and
there the repeat comes SECOND, so the analysed presentation is the
uncontaminated one.

Inputs
------
  PAIR_TABLES   all_pairs.csv per set (step 4), plus where that set's
                rendered images live (step 3)

Outputs (per set, under OUTPUT_ROOT/<label>/)
---------------------------------------------
  trial_list.csv      the full session, in presentation order
  trial_design.png    the trials in difference space, and bin counts
  trials_report.txt   counts and checks
  package/            self-contained, ready to upload:
    trial_list.csv      relative paths
    trial_list.json     the same trials for the experiment to load
    stimuli/            ONLY the images this list uses
    manifest.csv        every shipped image with its features
    preload.json        the paths, for fetching up front
    README.txt          how to read it all

Usage
-----
    conda activate drl2drv
    python build_trials_v4.py
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cv2.setNumThreads(0)


# ============================================================
# SETTINGS
# ============================================================

# --- Input / output ---
#
# PAIR_TABLES
#   (label, all_pairs.csv, stimuli directory) per stimulus set.
#   The label names the output subdirectory. The stimuli directory is
#   that set's own rendered images -- getting these crossed would build
#   a trial list whose predictors describe different pictures from the
#   ones shown, which the packaging guard below is there to catch.
# PAIR_TABLES = [
#     ("haze",    "data/pair_analysis_v4/haze/all_pairs.csv",
#      "data/stimulus_set_v4_haze/stimuli"),
#     ("haze_eq", "data/pair_analysis_v4/haze_eq/all_pairs.csv",
#      "data/stimulus_set_v4_haze_eq/stimuli"),
# ]

# PAIR_TABLES = [
#     ("haze",    "data/pair_analysis_v4/haze/all_pairs.csv",
#      "data/stimulus_set_v4_2_haze_2/stimuli"),
#     ("haze_eq", "data/pair_analysis_v4/haze_eq/all_pairs.csv",
#      "data/stimulus_set_v4_2_haze_eq/stimuli"),
# ]

PAIR_TABLES = [
    ("haze_eq", "data/pair_analysis_v4_ec/haze_eq/all_pairs.csv",
     "data/stimulus_set_v4_ec_haze_eq_2/stimuli"),
    ("haze",    "data/pair_analysis_v4_ec/haze/all_pairs.csv",
     "data/stimulus_set_v4_ec_haze_2/stimuli"),
]

OUTPUT_ROOT = "data/trials_v4_ec"

# ON_EXISTING
#   "new" writes to OUTPUT_ROOT_2, _3, ... rather than overwriting.
#   A trial list and its package must stay in step, so a half-replaced
#   directory is worse than a duplicate one.
ON_EXISTING = "new"

# --- Trial counts ---
#
# N_SAT / N_FOG / N_MIXED
#   Real trials of each kind. Equal counts by design. See the header
#   for why 40.
N_SAT   = 60 # 40
N_FOG   = 60 # 40
N_MIXED = 60 # 40

# MIXED_CONFLICT_FRAC
#   Share of the mixed trials that are conflict rather than together.
#   Half and half: `together` gives the largest effect and acts as the
#   floor check, `conflict` gives the crossover.
MIXED_CONFLICT_FRAC = 0.5

# N_CONSISTENCY
#   Pairs repeated later in the session with the sides swapped, so the
#   repeat is not recognisable. Gives a per-participant agreement rate,
#   which is what lets you tell "does not track these features" from
#   "was not attending". Essential when each participant is a data
#   point.
N_CONSISTENCY = 15

# N_CATCH
#   Trials whose instruction is replaced by something task-irrelevant.
#   Suzuki et al. used two per block of 32 and excluded participants
#   failing four of eight. With 6 you can set an exclusion rule of
#   three or more failures.
N_CATCH = 6

# N_PRACTICE
#   Unanalysed trials at the start, drawn from the `together` pool
#   because those have the most obvious difference.
N_PRACTICE = 6

# --- Sampling constraints ---
#
# N_MAGNITUDE_BINS
#   Trials are spread evenly across this many bins of difference size,
#   matching the dose-response analysis.
N_MAGNITUDE_BINS = 3

# BALANCED_CONFLICT_ONLY
#   Draw conflict trials only from pairs step 4 flagged as balanced.
#   The fog cue runs about 1.65x stronger than the colour cue across
#   raw conflict pairs -- step 4 measured exactly that on both sets --
#   so an unfiltered draw would bias a simple proportion toward "fog
#   wins" whatever participants are tracking. Turning this off exists
#   only so the effect of the constraint can be measured.
BALANCED_CONFLICT_ONLY = True

# MAX_PAIRS_PER_SCENE_PER_TYPE
#   Cap on how many trials of one type any single scene contributes, so
#   no photograph dominates.
#
#   Counted ACROSS the whole draw for that type, not within each
#   magnitude bin -- see stratified_sample for why that distinction
#   matters and what v2 did instead.
#
#   1 is deliberate and stricter than v2's value. Step 4 checked it:
#   with 47 scenes and 40 trials per type, enough scenes can supply one
#   pair of every type, so a cap of 1 costs nothing. It matters because
#   two scenes (courtyard and the train platform) have fog that barely
#   drains colour and so hold an outsized share of the conflict pool --
#   at a cap of 1 they contribute one conflict trial each, like
#   everyone else.
#
#   The ceiling is one trial per scene per type, so 4 analysed trials
#   from any one scene out of 120. Raise it if a pool is too thin to
#   fill a type, which the draw will say plainly.
#
#   None computes a looser cap from the scene count, as v2 did.
MAX_PAIRS_PER_SCENE_PER_TYPE = 2

# --- Presentation ---
#
# CONSISTENCY_MIN_GAP
#   Minimum trials between a pair and its repeat, so the repeat is not
#   recognised as one.
CONSISTENCY_MIN_GAP = 25

# CATCH_INSTRUCTION
#   Shown in place of the normal question on catch trials.
CATCH_INSTRUCTION = "Click the image on the RIGHT."

# MAIN_INSTRUCTION
#   The question asked on every real trial. Kept here so the wording
#   lives with the trial list rather than only in the experiment
#   software.
MAIN_INSTRUCTION = ("Which of these looks more like how the world "
                    "looked to you?")

# --- Packaging ---
PACKAGE = True

# SHIP_FORMAT / SHIP_KWARGS
#   Format for the packaged images.
#
#   "png"  lossless, zero drift, nothing to justify.
#   "jpeg" roughly 3-5x smaller. ALWAYS pass subsampling=0 (4:4:4):
#          JPEG's default halves the resolution of the colour channels,
#          and saturation is a dependent variable here, so that is
#          exactly the wrong economy.
#
#   Do NOT resize. Features were measured on a 256 downsample of the
#   512 image, and that downsampling smooths fine detail. A file that
#   is already smaller skips the smoothing, and contrast_local lands
#   about 9 units away.
SHIP_FORMAT = "png"
SHIP_KWARGS = dict(format="PNG")
# For JPEG instead:
# SHIP_FORMAT = "jpeg"
# SHIP_KWARGS = dict(format="JPEG", quality=95, subsampling=0)

# REMEASURE_AFTER_CONVERT
#   Recompute sat_mean and contrast_local on the CONVERTED files and
#   overwrite d_sat, d_fog, z_sat and z_fog in the packaged trial list,
#   so the predictors describe the images participants actually see
#   rather than their lossless ancestors. Pointless for PNG (nothing
#   changes), worth it for any lossy format -- and it doubles as a
#   check that STIMULI_SRC really is the right folder.
REMEASURE_AFTER_CONVERT = True

# REMEASURE_MAX_SHIFT
#   Guard on the re-measurement. Conversion shifts these features by a
#   fraction of a unit. A shift beyond this means the stimuli directory
#   is not the one the pair table was built from -- the two fog
#   families being the obvious way to get that wrong -- and the script
#   stops rather than silently overwriting the predictors with
#   unrelated values.
REMEASURE_MAX_SHIFT = 0.6

# FEATURE_TARGET_SIZE
#   Only used by the re-measurement. A MEASUREMENT CONVENTION inherited
#   from step 04, not a CLIP requirement. Do not change it.
FEATURE_TARGET_SIZE = (256, 256)

# FOG_SIGN
#   Multiplier on the raw fog-feature difference so a POSITIVE value
#   always means "image A is foggier". Used ONLY by the re-measurement:
#   the sampling reads d_fog straight from all_pairs.csv, where step 4
#   has already applied the sign.
FOG_SIGN = -1.0

# JSON_COLUMNS
#   Columns written to trial_list.json, the version a web experiment
#   loads. `instruction` is deliberately omitted: the question is the
#   same on every real trial, and the catch instruction belongs in the
#   experiment code rather than repeated on every row. The CSV keeps it
#   as the authoritative record.
#
#   NaN becomes null in JSON. correct_response is set only on catch
#   trials and repeat_of only on consistency trials, so those are null
#   on most rows -- treat null as "not applicable".
JSON_COLUMNS = [
    "trial", "trial_type", "analysed", "scene_id", "category",
    "left_path", "right_path", "a_is_left",
    "d_sat", "d_fog", "z_sat", "z_fog", "mag_bin",
    "correct_response", "repeat_of",
]

RANDOM_SEED = 42

REAL_TYPES = ["sat_only", "fog_only", "together", "conflict"]


# ============================================================
# Small helpers
# ============================================================

def resolve_output_root():
    """OUTPUT_ROOT, never overwriting an existing build."""
    def occupied(d):
        if not os.path.isdir(d):
            return False
        return any(os.path.isfile(os.path.join(d, lab,
                                               "trial_list.csv"))
                   for lab, _, _ in PAIR_TABLES)

    if not occupied(OUTPUT_ROOT):
        return OUTPUT_ROOT
    if ON_EXISTING == "error":
        raise FileExistsError(
            f"{OUTPUT_ROOT} already holds trial lists. Move it, or set "
            "ON_EXISTING = 'new'.")
    n = 2
    while occupied(f"{OUTPUT_ROOT}_{n}"):
        n += 1
    print(f"  NOTE: {OUTPUT_ROOT} already holds trial lists; writing "
          f"to {OUTPUT_ROOT}_{n} instead.")
    return f"{OUTPUT_ROOT}_{n}"


def pair_key(a, b):
    """Identify a pair by its IMAGES, ignoring which is A and which B.

    Pairs are enumerated in both directions by step 4, so (a, b) and
    (b, a) are two rows describing the SAME two pictures. Everything
    that asks "has the participant seen these two images together
    already?" must use this key.

    v2 compared the ordered tuple in one place and the sorted one in
    another. The mismatch let a pair drawn as a real trial come back as
    practice in the opposite order -- the exact double-exposure the
    separate practice pool exists to prevent -- and left the repeat
    ordering unable to find its original. Rare, silent, and avoided
    entirely by having one key.
    """
    return (a, b) if str(a) <= str(b) else (b, a)


def keys_of(df):
    return [pair_key(a, b) for a, b in zip(df.file_a, df.file_b)]


# ============================================================
# Sampling
# ============================================================

def stratified_sample(pool, n, rng, mag_col, n_bins,
                      max_per_scene=None):
    """Draw `n` pairs spread across magnitude bins and scenes.

    Random sampling would clump: most pairs sit at mid-range
    magnitudes, so a uniform draw leaves the extreme bins thin and the
    dose-response curve unsupported at its ends. Binning first, then
    drawing equally from each bin, fixes that.

    THE SCENE CAP IS COUNTED ACROSS THE WHOLE DRAW, not within each
    bin. Counting per bin -- which is what v2 did -- quietly multiplies
    the cap by the number of bins: a "cap of 1" became one trial per
    scene per bin, so three. That is exactly the concentration the cap
    exists to prevent, and on this set it let one scene supply seven of
    the analysed trials.

    Bins are visited in a random order so that no bin systematically
    gets first pick of the scenes.
    """
    if not len(pool):
        raise ValueError("Empty pool; cannot sample.")

    pool = pool.copy()
    mag = pool[mag_col].abs()
    # Rank-based binning, so bins hold equal numbers of pairs whatever
    # the shape of the magnitude distribution.
    pool["mag_bin"] = pd.qcut(mag.rank(method="first"), n_bins,
                              labels=range(1, n_bins + 1))

    per_bin = [n // n_bins] * n_bins
    for k in range(n % n_bins):
        per_bin[k] += 1
    want_of = dict(zip(range(1, n_bins + 1), per_bin))

    order = list(range(1, n_bins + 1))
    rng.shuffle(order)

    out, taken, used = [], set(), {}
    for b in order:
        want = want_of[b]
        sub = pool[pool.mag_bin == b]
        if not len(sub):
            print(f"    WARNING: bin {b} is empty")
            continue
        sub = sub.sample(frac=1.0, random_state=rng.integers(1 << 30))
        # Greedy pass: take a pair only if its pictures have not been
        # used already (in either direction) and its scene still has
        # room under the cap.
        rows = []
        for pos, (sid, fa, fb) in enumerate(
                zip(sub.scene_id, sub.file_a, sub.file_b)):
            if len(rows) >= want:
                break
            k = pair_key(fa, fb)
            if k in taken:
                continue
            if (max_per_scene is not None
                    and used.get(sid, 0) >= max_per_scene):
                continue
            rows.append(pos)
            taken.add(k)
            used[sid] = used.get(sid, 0) + 1
        if len(rows) < want:
            print(f"    WARNING: bin {b} wanted {want}, could place "
                  f"{len(rows)} under a cap of {max_per_scene} per "
                  "scene")
        out.append(sub.iloc[rows])

    got = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return got.sort_values("mag_bin").reset_index(drop=True)


def build_main_trials(pairs, rng, report):
    """Sample the real trials of each type, plus practice and catch.

    Practice and catch pairs are drawn SEPARATELY and never appear
    among the analysed trials: any pair seen twice gives the second
    presentation the advantage of prior exposure.
    """
    print("\n" + "=" * 60)
    print("SAMPLING TRIALS")
    print("=" * 60)

    n_scenes = pairs.scene_id.nunique()
    cap = MAX_PAIRS_PER_SCENE_PER_TYPE
    if cap is None:
        # A little headroom over perfect balance, so a scene short of
        # one type does not block the draw.
        cap = int(np.ceil(max(N_SAT, N_FOG, N_MIXED) / n_scenes)) + 2
    print(f"  {n_scenes} scenes, cap {cap} trial(s) per scene per type")

    n_conflict = int(round(N_MIXED * MIXED_CONFLICT_FRAC))
    n_together = N_MIXED - n_conflict

    conflict_pool = pairs[pairs.type == "conflict"]
    if BALANCED_CONFLICT_ONLY:
        if "balanced" not in pairs.columns:
            raise KeyError(
                "all_pairs.csv has no `balanced` column; re-run step 4 "
                "(analyse_pairs_v4.py).")
        conflict_pool = conflict_pool[conflict_pool.balanced]

    specs = [
        ("sat_only", N_SAT, "d_sat", pairs[pairs.type == "sat_only"]),
        ("fog_only", N_FOG, "d_fog", pairs[pairs.type == "fog_only"]),
        ("together", n_together, "d_fog",
         pairs[pairs.type == "together"]),
        ("conflict", n_conflict, "d_fog", conflict_pool),
    ]

    chunks = []
    for name, want, mag_col, pool in specs:
        print(f"\n  {name}: want {want}, pool {len(pool):,}")
        if len(pool) < want:
            raise RuntimeError(
                f"Pool for {name} has only {len(pool)} pairs but "
                f"{want} trials were requested. Loosen the thresholds "
                "in step 4, lower the trial count, or raise "
                "MAX_PAIRS_PER_SCENE_PER_TYPE.")
        got = stratified_sample(pool, want, rng, mag_col,
                                N_MAGNITUDE_BINS, cap)
        if len(got) < want:
            raise RuntimeError(
                f"Drew only {len(got)} of {want} {name} trials at a "
                f"cap of {cap} per scene. Raise "
                "MAX_PAIRS_PER_SCENE_PER_TYPE or lower the count.")
        got = got.assign(trial_type=name, mag_driver=mag_col)
        print(f"    drew {len(got)}, {got.scene_id.nunique()} scenes, "
              f"bins {sorted(got.mag_bin.value_counts().to_dict().items())}")
        chunks.append(got)
        report.append((f"{name} trials", str(len(got))))

    main = pd.concat(chunks, ignore_index=True)

    # --- Practice and catch, from pairs NOT used as real trials ---
    #
    # `together` pairs are used for both: they have the most obvious
    # difference, which is what practice wants and what makes a catch
    # trial's replaced instruction stand out rather than blend in.
    used = set(keys_of(main))
    tog = pairs[pairs.type == "together"]
    avail = tog[~pd.Series(keys_of(tog), index=tog.index).isin(used)]
    need = N_PRACTICE + N_CATCH
    if len(avail) < need:
        raise RuntimeError(
            f"Only {len(avail)} unused `together` pairs are available "
            f"for {N_PRACTICE} practice + {N_CATCH} catch trials.")
    extra = stratified_sample(avail, need, rng, "d_fog", 1, None)
    prac = extra.head(N_PRACTICE).assign(trial_type="practice",
                                         mag_driver="d_fog")
    catch = extra.tail(N_CATCH).assign(trial_type="catch",
                                       mag_driver="d_fog")
    print(f"\n  practice + catch: drew {len(extra)} from "
          f"{len(avail):,} unused pairs (no overlap with the analysed "
          "trials, in either direction)")
    report.append(("practice trials", str(len(prac))))
    report.append(("catch trials", str(len(catch))))
    return main, prac, catch


# ============================================================
# Assemble the session
# ============================================================

def _balanced_sides(labels, rng):
    """Exactly half of each group gets True, assigned at random.

    Independent coin flips give a binomial spread -- with 40 trials per
    type the realised split can easily land at 0.40 or 0.53. That
    matters because the model has no intercept: the design is
    symmetric, so there is no baseline preference to absorb. A
    participant with a side preference plus unequal side assignment
    creates exactly such a constant, and with nowhere to go it leaks
    into whichever weight correlates with the imbalance.

    Simulated on a participant who tracks NOTHING but prefers one side,
    the spurious weight was:

        side preference     realised 0.46     forced 50/50
        mild  (0.5)             0.059            0.016
        moderate (1.0)          0.136            0.004
        strong (2.0)            0.214            0.006

    Not fatal against a minimum detectable weight near 0.86, but free
    to remove, and it pushes one weight rather than both -- unhelpful
    in a design whose headline result is b_sat against b_fog.

    Odd group sizes get the extra trial assigned at random, so no group
    is systematically biased.
    """
    out = np.zeros(len(labels), dtype=bool)
    for lab in pd.unique(labels):
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        if len(idx) % 2 and rng.random() < 0.5:
            half += 1
        out[idx[:half]] = True
    return out


def _assign_sides(df, labels, rng):
    """left/right assignment. `swap` means image B is shown on the left.

    The stored d_sat and d_fog always refer to A minus B, so the
    analysis is unaffected; left_file / right_file are what the
    software displays.
    """
    df = df.copy()
    swap = ~_balanced_sides(labels, rng)
    df["left_file"] = np.where(swap, df.file_b, df.file_a)
    df["right_file"] = np.where(swap, df.file_a, df.file_b)
    df["a_is_left"] = ~swap
    return df


def assemble(main, prac, catch, rng, report):
    """Add practice, consistency and catch trials, randomise sides, and
    order the session.

    Types are interleaved rather than blocked, so a participant does
    not meet a run of trials in which only the colour changes and work
    out the manipulation.
    """
    print("\n" + "=" * 60)
    print("ASSEMBLING THE SESSION")
    print("=" * 60)

    main = main.sample(frac=1.0,
                       random_state=RANDOM_SEED).reset_index(drop=True)
    # Balanced WITHIN each trial type -- see _balanced_sides.
    main = _assign_sides(main, main.trial_type.to_numpy(), rng)

    # --- Consistency repeats ---
    # Drawn from across the types, repeated with the sides swapped so
    # the repeat is not recognisable as one.
    idx = rng.choice(len(main), N_CONSISTENCY, replace=False)
    rep = main.iloc[idx].copy()
    rep["left_file"], rep["right_file"] = (rep.right_file.to_numpy(),
                                           rep.left_file.to_numpy())
    rep["a_is_left"] = ~rep.a_is_left
    rep["trial_type"] = "consistency"
    # repeat_of is NOT set here. `idx` indexes the shuffled frame, and
    # the rows are about to be reordered twice and renumbered, so a
    # value stored now would point at nothing. Resolved at the end.

    catch = _assign_sides(catch, np.full(len(catch), "catch"), rng)
    catch["correct_response"] = "right"
    prac = _assign_sides(prac, np.full(len(prac), "practice"), rng)

    # --- Order: practice first, then everything interleaved ---
    body = pd.concat([main, rep, catch], ignore_index=True)
    body = body.sample(
        frac=1.0, random_state=RANDOM_SEED + 1).reset_index(drop=True)

    # Space each repeat from its original, AND put the repeat second.
    # If the repeat came first, the later presentation would be the
    # analysed one -- so the analysed trial would be the contaminated
    # one, which is backwards.
    body = _space_repeats(body, rng)
    body = _order_repeats_last(body)

    session = pd.concat([prac, body], ignore_index=True)
    session.insert(0, "trial", range(1, len(session) + 1))
    session["instruction"] = np.where(
        session.trial_type == "catch", CATCH_INSTRUCTION,
        MAIN_INSTRUCTION)
    # `analysed` marks the trials entering the regression and the
    # proportion test: the four real types only. See the header for why
    # consistency trials are excluded.
    session["analysed"] = session.trial_type.isin(REAL_TYPES)
    session = _resolve_repeat_of(session)

    print(f"  {len(session)} trials total")
    print(session.trial_type.value_counts().to_string())
    report.append(("total trials", str(len(session))))
    report.append(("analysed trials", str(int(session.analysed.sum()))))
    report.append(("consistency (checked, not analysed)",
                   str(int((session.trial_type == "consistency").sum()))))
    return session


def _resolve_repeat_of(session):
    """Point each consistency trial at the trial it repeats.

    Must run AFTER `trial` is assigned. Storing a positional index
    before the reordering steps would leave every value pointing at an
    unrelated trial -- worse than an empty column, because the
    consistency rate is computed by pairing each repeat with its
    original and asking whether the same image was chosen. Pairing
    against unrelated trials returns eleven coin flips and an agreement
    rate near 50% for every participant, however attentive.

    Matching is by the unordered image pair. Practice and catch draw
    from pairs the analysed set never uses, so the only other trial
    sharing a repeat's pictures is its original.
    """
    session = session.reset_index(drop=True)
    key = keys_of(session)
    session["repeat_of"] = pd.NA

    for i, r in session.iterrows():
        if r.trial_type != "consistency":
            continue
        hits = [j for j, k in enumerate(key)
                if k == key[i]
                and session.loc[j, "trial_type"] != "consistency"]
        if len(hits) != 1:
            raise RuntimeError(
                f"Consistency trial {r.trial} matches {len(hits)} "
                "non-consistency trials by image pair; expected 1.")
        session.loc[i, "repeat_of"] = int(session.loc[hits[0], "trial"])

    # --- Assertions: both would have caught the bug described above ---
    con = session[session.trial_type == "consistency"]
    for _, r in con.iterrows():
        tgt = session[session.trial == r.repeat_of]
        if len(tgt) != 1:
            raise RuntimeError(
                f"repeat_of {r.repeat_of} does not resolve to a trial.")
        tgt = tgt.iloc[0]
        if pair_key(tgt.file_a, tgt.file_b) != pair_key(r.file_a,
                                                        r.file_b):
            raise RuntimeError(
                f"Trial {r.trial} points at trial {r.repeat_of}, which "
                "shows a different image pair.")
        if tgt.trial >= r.trial:
            raise RuntimeError(
                f"Trial {r.trial} points at trial {r.repeat_of}, which "
                "is not earlier. The repeat must follow its original, "
                "so the analysed trial is the uncontaminated one.")
    print(f"    repeat_of resolved and verified for {len(con)} "
          "consistency trials")
    return session


def _order_repeats_last(body):
    """Ensure each consistency repeat follows its original.

    Swaps the two rows' POSITIONS where necessary, leaving every other
    trial where it is, so the spacing achieved by _space_repeats
    survives.
    """
    body = body.reset_index(drop=True)
    pos = {}
    for i, k in enumerate(keys_of(body)):
        pos.setdefault(k, []).append(i)

    order = list(range(len(body)))
    n_swapped = 0
    for idxs in pos.values():
        if len(idxs) != 2:
            continue
        a, b = sorted(idxs)
        if body.loc[a, "trial_type"] == "consistency":
            order[a], order[b] = order[b], order[a]
            n_swapped += 1
    if n_swapped:
        print(f"    reordered {n_swapped} repeat(s) to follow their "
              "originals")
    return body.iloc[order].reset_index(drop=True)


def _space_repeats(body, rng, max_tries=400):
    """Nudge consistency repeats away from their originals.

    A repeat sitting a few trials after its original would be
    recognised, and the agreement rate would measure memory rather than
    consistency.
    """
    body = body.reset_index(drop=True)
    bad = []
    for _ in range(max_tries):
        pos = {}
        for i, k in enumerate(keys_of(body)):
            pos.setdefault(k, []).append(i)
        bad = [v for v in pos.values()
               if len(v) > 1
               and (max(v) - min(v)) < CONSISTENCY_MIN_GAP]
        if not bad:
            return body
        # Move the later member of the closest offending pair.
        v = min(bad, key=lambda x: max(x) - min(x))
        i = max(v)
        j = int(rng.integers(0, len(body)))
        order = list(range(len(body)))
        order[i], order[j] = order[j], order[i]
        body = body.iloc[order].reset_index(drop=True)
    print(f"    NOTE: could not space every repeat by "
          f"{CONSISTENCY_MIN_GAP} trials; {len(bad)} remain close.")
    return body


# ============================================================
# Checks
# ============================================================

def check_session(session, label, report):
    """Sanity checks on the finished list."""
    print("\n" + "=" * 60)
    print(f"CHECKS  [{label}]")
    print("=" * 60)
    an = session[session.analysed]

    # Collinearity of the two predictors across the analysed trials.
    # This is the number deciding whether the two weights are
    # separately estimable.
    r = float(np.corrcoef(an.d_sat, an.d_fog)[0, 1])
    vif = 1.0 / max(1e-12, 1 - r ** 2)
    print(f"  r(d_sat, d_fog) over analysed trials : {r:+.3f}")
    print(f"  VIF                                  : {vif:.2f}")
    if abs(r) > 0.6:
        print("    HIGH. The two weights will be hard to separate. "
              "Check the trial-type proportions.")
    report.append(("r(d_sat, d_fog)", f"{r:+.3f}"))
    report.append(("VIF", f"{vif:.2f}"))

    # Side balance per type. Forced to 0.50 within each type, so
    # anything else means the assignment did not take.
    side = an.groupby("trial_type").a_is_left.agg(["mean", "size"])
    print("\n  proportion with image A on the left, by type:")
    print(side.round(3).to_string())
    worst = float((side["mean"] - 0.5).abs().max())
    if worst > 0.03:
        print(f"    WARNING: worst deviation from 0.50 is {worst:.3f}; "
              "side balancing may not have applied.")
    report.append(("side balance (worst dev)", f"{worst:.3f}"))

    # Scene spread.
    per_scene = an.scene_id.value_counts()
    print(f"\n  analysed trials per scene: min {per_scene.min()}, "
          f"median {int(per_scene.median())}, max {per_scene.max()} "
          f"over {an.scene_id.nunique()} scenes")
    report.append(("scenes used", str(an.scene_id.nunique())))
    report.append(("max trials from one scene", str(per_scene.max())))

    # No pair may be seen twice except a consistency repeat.
    k = keys_of(session)
    dup = pd.Series(k).value_counts()
    multi = dup[dup > 1]
    n_con = int((session.trial_type == "consistency").sum())
    if len(multi) != n_con or (multi > 2).any():
        raise RuntimeError(
            f"{len(multi)} image pairs appear more than once but there "
            f"are {n_con} consistency trials. Every repeat must be a "
            "consistency trial and no pair may appear three times.")
    print(f"\n  repeated image pairs: {len(multi)}, all consistency "
          "repeats")

    # Consistency repeats: gap, and whether the sides were swapped. A
    # swapped repeat tests that the choice survives a change of
    # position as well as a gap in time.
    con = session[session.trial_type == "consistency"]
    if len(con):
        gaps, swaps = [], []
        for _, r2 in con.iterrows():
            o = session[session.trial == r2.repeat_of].iloc[0]
            gaps.append(int(r2.trial) - int(o.trial))
            swaps.append(bool(o.left_file == r2.right_file))
        print(f"  consistency repeats: min gap {min(gaps)}, median "
              f"{int(np.median(gaps))}, sides swapped "
              f"{sum(swaps)}/{len(swaps)}")
        report.append(("consistency min gap", str(min(gaps))))
        if sum(swaps) != len(swaps):
            print("    WARNING: not every repeat has its sides swapped.")

    print("\n  trials per magnitude bin, by type:")
    print(pd.crosstab(an.trial_type, an.mag_bin).to_string())
    return r, vif


def make_figure(session, label, out_dir):
    an = session[session.analysed]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for t, g in an.groupby("trial_type"):
        axes[0].scatter(g.d_sat, g.d_fog, s=28, alpha=0.75, label=t)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].axvline(0, color="k", lw=0.6)
    axes[0].set_xlabel("d sat_mean  (A - B)")
    axes[0].set_ylabel("d contrast_local  (A - B, + means A is foggier)")
    axes[0].set_title(f"The trials, in difference space  [{label}]\n"
                      "(spread, not clumped, is what makes the two "
                      "weights separable)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    ct = pd.crosstab(an.trial_type, an.mag_bin)
    ct.plot(kind="bar", stacked=True, ax=axes[1], rot=0)
    axes[1].set_xlabel("trial type")
    axes[1].set_ylabel("trials")
    axes[1].set_title("Magnitude bins per type\n"
                      "(the dose-response analysis needs all three)")
    axes[1].legend(title="bin", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trial_design.png"), dpi=150)
    plt.close(fig)


# ============================================================
# Packaging
# ============================================================

def _features_of(path, target_size):
    """sat_mean and contrast_local of a file, measured exactly as step
    04 measured the corpus: resize to target_size, then compute.

    contrast_local is the standard deviation of the Laplacian -- the
    same computation the pipeline uses throughout, so the values are
    comparable with the manifest.
    """
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB").resize(target_size,
                                                  Image.BILINEAR),
                         dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    S = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)[:, :, 1]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(S.mean()), float(cv2.Laplacian(g, cv2.CV_32F).std())


def package(session, label, stimuli_src, out_dir, report):
    """Write a self-contained folder ready to upload.

    Only the images the trial list references are copied -- about 200
    of the 1,175 rendered, which is the difference between a package
    that preloads and one that does not.
    """
    print("\n" + "=" * 60)
    print(f"PACKAGING  [{label}]")
    print("=" * 60)
    pkg_dir = os.path.join(out_dir, "package")
    img_dir = os.path.join(pkg_dir, "stimuli")
    os.makedirs(img_dir, exist_ok=True)

    used = sorted(set(session.file_a) | set(session.file_b))
    print(f"  {len(used)} distinct images from {stimuli_src}")
    print(f"  format: {SHIP_FORMAT} {SHIP_KWARGS}")

    ext = ".png" if SHIP_FORMAT.lower() == "png" else ".jpg"
    rename, rows, total = {}, [], 0
    for n, fn in enumerate(used, 1):
        src = os.path.join(stimuli_src, fn)
        if not os.path.isfile(src):
            raise FileNotFoundError(
                f"Stimulus not found: {src}\nIs STIMULI_SRC for "
                f"{label!r} the folder this pair table was built from?")
        out_name = os.path.splitext(fn)[0] + ext
        dst = os.path.join(img_dir, out_name)
        with Image.open(src) as im:
            im.convert("RGB").save(dst, **SHIP_KWARGS)
        rename[fn] = out_name
        n_bytes = os.path.getsize(dst)
        total += n_bytes
        rec = {"filename": out_name, "source": fn,
               "kB": round(n_bytes / 1024, 1)}
        if REMEASURE_AFTER_CONVERT:
            sat, fog = _features_of(dst, FEATURE_TARGET_SIZE)
            rec["sat_mean"], rec["contrast_local"] = sat, fog
        rows.append(rec)
        if n % 50 == 0:
            print(f"    {n}/{len(used)}", end="\r")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(pkg_dir, "manifest.csv"), index=False)
    mb = total / 1024 / 1024
    print(f"\n  wrote {len(used)} images, {mb:.1f} MB "
          f"({total / len(used) / 1024:.0f} kB each)")

    out = session.copy()
    out["left_path"] = "stimuli/" + out.left_file.map(rename)
    out["right_path"] = "stimuli/" + out.right_file.map(rename)

    # --- Re-measure the predictors on the converted files ---
    if REMEASURE_AFTER_CONVERT:
        m = manifest.set_index("source")
        sat_a = m.loc[out.file_a, "sat_mean"].to_numpy()
        sat_b = m.loc[out.file_b, "sat_mean"].to_numpy()
        fog_a = m.loc[out.file_a, "contrast_local"].to_numpy()
        fog_b = m.loc[out.file_b, "contrast_local"].to_numpy()

        old_sat, old_fog = out.d_sat.to_numpy(), out.d_fog.to_numpy()
        new_sat = sat_a - sat_b
        new_fog = FOG_SIGN * (fog_a - fog_b)

        # z_* are the normalised versions: each difference divided by
        # that scene's own range. Rescale rather than recompute, since
        # the ranges are unchanged by conversion.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_s = np.where(np.abs(old_sat) > 1e-9,
                               new_sat / old_sat, 1.0)
            ratio_f = np.where(np.abs(old_fog) > 1e-9,
                               new_fog / old_fog, 1.0)
        out["z_sat"] = out.z_sat.to_numpy() * np.abs(ratio_s)
        out["z_fog"] = out.z_fog.to_numpy() * np.abs(ratio_f)
        out["d_sat"], out["d_fog"] = new_sat, new_fog

        d_s = float(np.abs(new_sat - old_sat).max())
        d_f = float(np.abs(new_fog - old_fog).max())
        print(f"  re-measured: worst |d_sat| shift {d_s:.3f}, "
              f"worst |d_fog| shift {d_f:.3f}")
        report.append(("remeasure d_sat shift", f"{d_s:.3f}"))

        # A shift of tens of units means the files are not the ones the
        # pair table describes -- the two fog families being the
        # obvious way to get that wrong -- and re-measuring would
        # silently overwrite the predictors with unrelated values.
        if d_s > REMEASURE_MAX_SHIFT or d_f > REMEASURE_MAX_SHIFT:
            raise RuntimeError(
                f"[{label}] Re-measured features shifted by {d_s:.1f} "
                f"(sat) and {d_f:.1f} (fog), far beyond what conversion "
                f"can cause (limit {REMEASURE_MAX_SHIFT}).\n"
                f"STIMULI_SRC = {stimuli_src!r} probably does not hold "
                "the images this trial list was built from -- check "
                "that the fog family matches the pair table.")

    cols = ["trial", "trial_type", "analysed", "instruction",
            "scene_id", "category", "left_path", "right_path",
            "a_is_left", "d_sat", "d_fog", "z_sat", "z_fog",
            "mag_bin", "correct_response", "repeat_of"]
    out[[c for c in cols if c in out.columns]].to_csv(
        os.path.join(pkg_dir, "trial_list.csv"), index=False)

    missing = [c for c in JSON_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(
            f"JSON_COLUMNS names columns that do not exist: {missing}")
    out[JSON_COLUMNS].to_json(
        os.path.join(pkg_dir, "trial_list.json"),
        orient="records", indent=1)
    print(f"  trial_list.json: {len(JSON_COLUMNS)} columns, "
          f"{len(out)} records")

    with open(os.path.join(pkg_dir, "preload.json"), "w") as f:
        json.dump(["stimuli/" + n for n in sorted(rename.values())],
                  f, indent=1)

    n_an = int(session.analysed.sum())
    with open(os.path.join(pkg_dir, "README.txt"), "w") as f:
        f.write(f"EXPERIMENT PACKAGE  ({label})\n")
        f.write("=" * 56 + "\n\n")
        f.write(f"  trials        {len(session)} ({n_an} analysed)\n")
        f.write(f"  images        {len(used)}\n")
        f.write(f"  format        {SHIP_FORMAT} {SHIP_KWARGS}\n")
        f.write(f"  total size    {mb:.1f} MB\n")
        f.write(f"  re-measured   {REMEASURE_AFTER_CONVERT}\n")
        f.write(f"  stimuli from  {stimuli_src}\n\n")
        f.write("FILES\n")
        f.write("  trial_list.csv   one row per trial, in order\n")
        f.write("  trial_list.json  the same trials as an array of\n")
        f.write("                   objects, for the experiment to\n")
        f.write("                   load. No `instruction` column --\n")
        f.write("                   keep the question text in code.\n")
        f.write("                   null means 'not applicable'.\n")
        f.write("  stimuli/         the images, referenced by\n")
        f.write("                   left_path and right_path\n")
        f.write("  manifest.csv     every image with its measured\n")
        f.write("                   sat_mean and contrast_local\n")
        f.write("  preload.json     all image paths, for fetching\n")
        f.write("                   before the first trial\n\n")
        f.write("TRIAL LIST\n")
        f.write("  trial        presentation order\n")
        f.write("  trial_type   sat_only / fog_only / together /\n")
        f.write("               conflict / consistency / catch /\n")
        f.write("               practice\n")
        f.write("  analysed     True ONLY for the four real types.\n")
        f.write("               Practice, catch and consistency are\n")
        f.write("               quality checks, excluded from every\n")
        f.write("               analysis.\n")
        f.write("  instruction  the question to show. Catch trials\n")
        f.write("               carry a different one.\n")
        f.write("  left_path    image for the left position\n")
        f.write("  right_path   image for the right position\n")
        f.write("  a_is_left    whether the left image is image A.\n")
        f.write("               The analysis needs it to map a side\n")
        f.write("               to a choice, because d_sat and d_fog\n")
        f.write("               are always A minus B.\n")
        f.write("  z_sat,z_fog  normalised differences -- the\n")
        f.write("               regression predictors\n")
        f.write("  d_sat,d_fog  raw differences, in feature units\n")
        f.write("  mag_bin      magnitude bin, for the\n")
        f.write("               dose-response analysis\n")
        f.write("  correct_response  catch trials only\n")
        f.write("  repeat_of    consistency trials only: the trial\n")
        f.write("               this one repeats\n\n")
        f.write("RECORDING RESPONSES\n")
        f.write("  Record which SIDE was clicked (left/right) and the\n")
        f.write("  reaction time. Combine with a_is_left to recover\n")
        f.write("  which image was chosen.\n")

    print(f"  package written to {pkg_dir}")
    report.append(("package images", str(len(used))))
    report.append(("package size", f"{mb:.1f} MB"))
    return out


# ============================================================
# One set
# ============================================================

def build_one(label, pairs_csv, stimuli_src, root):
    print("\n" + "#" * 60)
    print(f"# {label}")
    print("#" * 60)
    if not os.path.isfile(pairs_csv):
        raise FileNotFoundError(
            f"Pair table not found: {pairs_csv}\nRun step 4 "
            "(analyse_pairs_v4.py) first.")
    pairs = pd.read_csv(pairs_csv)
    for c in ("scene_id", "type", "file_a", "file_b", "d_sat", "d_fog"):
        if c not in pairs.columns:
            raise ValueError(f"{pairs_csv} has no {c!r} column.")

    out_dir = os.path.join(root, label)
    os.makedirs(out_dir, exist_ok=True)

    # One generator per set, seeded identically, so the two sessions
    # are built by the same process rather than one inheriting the
    # other's draws.
    rng = np.random.default_rng(RANDOM_SEED)
    report = []

    main, prac, catch = build_main_trials(pairs, rng, report)
    session = assemble(main, prac, catch, rng, report)
    r, vif = check_session(session, label, report)
    make_figure(session, label, out_dir)
    session.to_csv(os.path.join(out_dir, "trial_list.csv"), index=False)

    packaged = None
    if PACKAGE:
        packaged = package(session, label, stimuli_src, out_dir, report)

    path = os.path.join(out_dir, "trials_report.txt")
    with open(path, "w") as f:
        f.write(f"TRIAL LIST REPORT  (v4 axes, step 5) -- {label}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Pairs from  : {pairs_csv}\n")
        f.write(f"Stimuli from: {stimuli_src}\n")
        f.write(f"Counts      : {N_SAT} sat_only, {N_FOG} fog_only, "
                f"{N_MIXED} mixed "
                f"({MIXED_CONFLICT_FRAC:.0%} conflict)\n")
        f.write(f"Extras      : {N_CONSISTENCY} consistency, "
                f"{N_CATCH} catch, {N_PRACTICE} practice\n")
        f.write(f"Conflict    : "
                f"{'balanced subset only' if BALANCED_CONFLICT_ONLY else 'all'}\n")
        f.write(f"Scene cap   : {MAX_PAIRS_PER_SCENE_PER_TYPE} per "
                "type\n")
        f.write(f"Seed        : {RANDOM_SEED}\n\n")
        for k, v in report:
            f.write(f"  {k:36s} {v}\n")
        f.write("\nTRIALS BY TYPE\n")
        f.write(session.trial_type.value_counts().to_string() + "\n")
        f.write("\nMAGNITUDE BINS (analysed trials)\n")
        an = session[session.analysed]
        f.write(pd.crosstab(an.trial_type, an.mag_bin).to_string())
        f.write("\n\nWHAT TO CHECK BY EYE\n")
        f.write("  r(d_sat, d_fog) should be small: it is what decides\n")
        f.write("  whether the two weights can be told apart. Above\n")
        f.write("  about 0.6 they cannot.\n")
    return {"label": label, "session": session, "packaged": packaged,
            "report": report, "r": r, "vif": vif, "out_dir": out_dir,
            "path": path}


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("BUILD TRIALS  (v4, step 5)")
    print("=" * 60)
    root = resolve_output_root()
    os.makedirs(root, exist_ok=True)
    n_an = N_SAT + N_FOG + N_MIXED
    print(f"  sets    : {', '.join(l for l, _, _ in PAIR_TABLES)}")
    print(f"  analysed: {n_an} trials "
          f"({N_SAT}/{N_FOG}/{N_MIXED})")
    print(f"  plus    : {N_CONSISTENCY} consistency, {N_CATCH} catch, "
          f"{N_PRACTICE} practice = "
          f"{n_an + N_CONSISTENCY + N_CATCH + N_PRACTICE} presented")
    print(f"  output  : {root}")

    results = [build_one(label, pairs_csv, src, root)
               for label, pairs_csv, src in PAIR_TABLES]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        s = r["session"]
        print(f"  {r['label']:10s} {len(s):3d} trials "
              f"({int(s.analysed.sum())} analysed)   "
              f"r(d_sat,d_fog) {r['r']:+.3f}   -> {r['out_dir']}")
    print(f"\n  Elapsed: {time.time() - t0:.1f}s")
    print("\n  NEXT: step 6 (simulate_power_v4) checks whether these "
          "trial lists are powered.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
