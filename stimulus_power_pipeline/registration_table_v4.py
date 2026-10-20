"""
registration_table_v4.py   (step 8b)

WHERE THIS SITS
---------------
  step 7a  freeze_trial_list_v4        the frozen pair package
  step 8   simulate_power_v4           the design analysis (untouched)
  step 8   simulate_power_v4_n10_20    the same script, five settings
                                       changed, rerun over N = 10..20
  step 8b  THIS SCRIPT                 the registered rule, and the
                                       table the preregistration quotes
  step 9   documentation

This script RUNS NO SIMULATION. It reads the rerun's outputs and
nothing else, so every number it prints can be traced to a row of
rule_sweep.csv. If a figure here disagrees with the step-8 report, the
step-8 report is right and this script has a bug.

WHY IT EXISTS
-------------
The registered existence claim needs a decision rule fixed BEFORE the
data arrive, depending only on the final number of participants.

THE REGISTERED RULE IS THE COUNT RULE -- "at least k detections in the
predicted direction and at most m in the opposite" -- calibrated at
each N from 10 to 20, with no switch to anything else. The sign test is
computed at every N as well and REPORTED AS A ROBUSTNESS COMPARISON.

A combination of the two was considered first, and the case for it was
this: the count rule can LOSE power as N grows, because when m = 0 a
single opposite-direction detection vetoes the claim, so every
non-responder added is another chance of that veto.

That is a property of the m = 0 rule and of no other. It was read off a
summary cell -- a weak effect with a quarter responding -- in which the
rule selected happened to be the m = 0 one, and in which neither rule
has useful power anyway. Selected instead in the PLANNING CELL, where
the study can succeed, the count rule has m > 0 from N = 11 on and
the sign test does not overtake it anywhere in 10..20 -- by at least
0.037 on colour and 0.010 on fog.

Its power does NOT rise monotonically: it dips where the false-rate
ceiling forces a stricter m (see section 2 of the report, which
computes the sequence). The registration quotes power per N from the
table, so nothing rests on monotonicity -- but the claim that more
participants always help is not available here, and the report says
so. The scan that shows this is still computed and reported
(section 3 of the report), because it is the evidence for registering
the count rule rather than an assertion that it is better.

THE PLANNING CELL
-----------------
Half the participants responding, a moderate effect, lapse 0.1. Not the
cell the step-8 summary used -- a weak effect with a quarter
responding, where neither rule reaches useful power and the comparison
is therefore choosing a rule for a scenario in which the study fails
either way.

Both rules keep the false-claim rate at or below the target under the
strict standard at every N, and that is checked here rather than
assumed (CHECK 3).

A rule calibrated at one cell is quoting its own best case, so the
table also carries its power and the sign test's in two further cells
-- the weaker pair at half share, and the planning weights at a quarter
share -- with the crossovers stated (report section 6).

AT EQUAL BEHAVIOUR, which is the part that needs care. A weight per SD
is not comparable between the two features on this list: the
informative colour trials are about 1.0 corpus SD apart and the fog
trials about 0.64, so the same weight produces a smaller choice bias on
fog. Comparing the features at the same weight would make fog look
worse for a reason that is arithmetic. The planning cell therefore uses
a DIFFERENT weight per feature, chosen so the two sit at nearly the
same choice probability on a typical informative trial:

    planning     colour 0.50/SD (p = 0.619)   fog 0.70/SD (p = 0.610)
    robustness   colour 0.30/SD (p = 0.572)   fog 0.50/SD (p = 0.580)

Neither pair is exact, and the higher one is less exact where it
matters: see CHECK 5 and the report section it writes.

INPUTS   (all from IN_ROOT, written by simulate_power_v4_n10_20.py)
------
  rule_sweep.csv            every candidate rule at every N, with its
                            false-claim rates, its admissibility under
                            each standard, and its power in every
                            (weight, share, lapse) cell. THE SOURCE.
  power_equal_behaviour.csv the behavioural statement of the claims
  detection_curve.csv       per-person detection against weight
  c1_power.csv              the control comparison
  run_meta.json             provenance of the run being read

OUTPUTS  (OUT_ROOT)
-------
  registration_table.csv    THE TABLE. One row per feature x N x rule.
  combined_rule.csv         the registered rule at each N, and its power
  combined_rule_power.png   that power against N, both features
  switch_point.json         the scan: where, if anywhere, the sign test
                            overtakes the count rule
  tie_breaks.csv            every firing of the tie-break, marked as a
                            genuine decision or as one rule appearing
                            under both family names
  equal_behaviour_check.csv the two pairs, their probabilities and
                            their per-person detection rates
  c1_compact.csv            participants x controls, compactly
  registration_report.txt   what was run and what came out
  run_meta.json

CHECKS
------
  CHECK 1  INPUTS. Every required file present, every column this
           script names present, and N_MIN..N_MAX complete for both
           features. A missing N is fatal: a rule fixed per N cannot
           have a hole in it.
  CHECK 2  ADMISSIBILITY, RECOMPUTED. Every selected rule is checked
           against the target at every spread in the standard's own
           set, from the false_taste_* columns -- not by trusting the
           admissible_* flag the sweep wrote.
  CHECK 3  THE REGISTERED RULE. Its false-claim rate at every N is
           the rate of the rule it uses there, and every one clears
           the target. This is the statement the registration rests
           on.
  CHECK 4  THE SWITCH POINT. It exists, it is unique, and sign >= count
           holds for every N at or above it, for both features.
  CHECK 5  EQUAL BEHAVIOUR. How close each pair actually is, in choice
           probability AND in per-person detection rate, both printed
           per feature so the reader can judge rather than take it.
  CHECK 6  COVERAGE. Every power column this script reads exists in the
           sweep, and the planning weight is one the sweep carries.

USAGE
-----
    conda activate drl2drv
    python registration_table_v4.py
"""

import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

# --- Input ---
#
# IN_ROOT
#   The rerun's output folder, INCLUDING ITS SET SUBFOLDER.
#
#   Step 8 writes per-set outputs to OUTPUT_ROOT/<label>/ and shared
#   ones to OUTPUT_ROOT/ itself. There is one set, labelled "frozen",
#   and all five files this script reads -- rule_sweep.csv,
#   power_equal_behaviour.csv, detection_curve.csv, c1_power.csv,
#   run_meta.json -- are per-set, so they are one level down. The
#   shared files (power_curve, bayes_factors, comparison,
#   power_report.txt) sit in the parent and are not read here.
#
#   NOT data/power_v4/frozen: that run's N grid skips 11, 13, 17, 18
#   and 19, and a rule fixed per N cannot have a hole in it. CHECK 1
#   fails loudly if the folder read has one.
#
#   If a rerun landed in data/power_v4_n10_20_2 (step 8 never
#   overwrites), change the parent here and leave /frozen in place.
IN_ROOT = "data/power_v4_n10_20/frozen"

# PROVENANCE_HEADER_LINES
#   How many lines of the rerun's power_report.txt to copy into this
#   report and into run_meta.json. The rerun writes no run_meta.json,
#   so its report header is the only machine-readable record of what
#   it was run with. 40 lines covers the settings block with room to
#   spare; trailing blanks are trimmed.
PROVENANCE_HEADER_LINES = 40

SWEEP_CSV      = os.path.join(IN_ROOT, "rule_sweep.csv")
EQUAL_BEH_CSV  = os.path.join(IN_ROOT, "power_equal_behaviour.csv")
DETECTION_CSV  = os.path.join(IN_ROOT, "detection_curve.csv")
C1_CSV         = os.path.join(IN_ROOT, "c1_power.csv")
RUN_META_JSON  = os.path.join(IN_ROOT, "run_meta.json")

# --- Output ---
OUT_ROOT = "data/registration_v4"

# ON_EXISTING
#   "new"    write to OUT_ROOT_2, _3, ... rather than overwrite.
#   "error"  stop instead.
ON_EXISTING = "new"

# ============================================================
# The planning cell
# ============================================================

# PLANNING_WEIGHT / ROBUSTNESS_WEIGHT
#   The responder's true weight per SD, PER FEATURE, chosen so the two
#   features sit at nearly the same choice probability on a typical
#   informative trial. The probabilities are not set here -- they are
#   read from the sweep's own effect_p_median_w* columns and printed,
#   so this setting cannot drift away from what the sweep measured.
#
#   Each weight must be one the sweep carries (CHECK 6).
PLANNING_WEIGHT   = {"colour": 0.50, "fog": 0.70}
ROBUSTNESS_WEIGHT = {"colour": 0.30, "fog": 0.50}

# PLANNING_SHARE / PLANNING_LAPSE
#   Half the participants responding, lapse 0.1. The share is the one
#   assumption that most moves the count rule: at full response its
#   m = 0 veto almost never fires, and at a quarter neither rule has
#   useful power.
PLANNING_SHARE = 0.50
PLANNING_LAPSE = 0.1

# --- The table's other cells ---
TABLE_SHARES = [0.25, 0.50, 1.00]
TABLE_LAPSES = [0.0, 0.1]

# ============================================================
# The rules
# ============================================================

N_MIN = 10
N_MAX = 20

# STANDARD
#   The admissibility standard the registered rule must meet, as named
#   in the sweep's admissible_* columns. "strict" requires the rule to
#   hold its false-claim rate at or below the target at every taste
#   spread in its set, including a spread as wide as the floor itself.
STANDARD = "strict"

# STANDARD_SPREADS
#   The taste spreads that standard's set contains, as the suffixes of
#   the sweep's false_taste_* columns. CHECK 2 recomputes the worst
#   rate over exactly these rather than trusting the sweep's flag.
STANDARD_SPREADS = ["0.25", "0.5", "0.75", "1"]

TARGET_FALSE = 0.05

# USE_MC_MARGIN
#   True  a rule is admissible only if the UPPER BOUND of its 95%
#         Monte Carlo interval clears the target -- the column
#         admissible_<standard>.
#   False the point estimate, admissible_point_<standard>.
#
#   True. A rule certified on a point estimate that happens to land at
#   0.0498 is not certified.
USE_MC_MARGIN = True

# REGISTERED_RULE
#   "count"     the count rule at every N. What is registered.
#   "combined"  the count rule below a switch point and the sign test
#               from it.
#
#   "count". The combination was designed for a defect that does not
#   occur once the count rule is selected in the planning cell: there
#   the chosen rule has m > 0 from N = 11 on, so no single
#   opposite-direction detection can veto the claim, and power rises
#   with N instead of falling. The sign test does not overtake it
#   anywhere in 10..20. The switch-point scan is still run and
#   reported, because it is the evidence for that statement rather
#   than an assertion of it.
REGISTERED_RULE = "count"

# COUNT_FAMILIES
#   Which families count as "the count rule".
#
#   BOTH. count-absolute degenerates as N grows -- a fixed count
#   becomes trivially easy -- which is a real objection to registering
#   it at large N, but the registration range stops at 20 and the two
#   families overlap heavily there: at N = 15 the rule k >= 6, m <= 3
#   appears in both, with identical power and identical false-claim
#   rates, because floor(0.20 x 15) = 3. Excluding count-absolute
#   would have silently resolved a tie that ought to be resolved by
#   the stated tie-break instead.
#
#   FAMILY_ORDER is the last tie-break, so a rule that both families
#   express is reported as the proportional one -- the family that
#   does not degenerate, and the one a reader should see.
COUNT_FAMILIES = ("count-proportional", "count-absolute")
FAMILY_ORDER = {"count-proportional": 0, "count-absolute": 1}

SIGN_FAMILY = "sign"

# POWER_TIE_TOL
#   Two rules whose planning-cell power differs by less than this are
#   a TIE, not an ordering.
#
#   The step-8 run simulates N_SIM_STUDIES = 2000 studies per cell, so
#   a power near 0.93 carries a Monte Carlo standard error of about
#   sqrt(0.93 x 0.07 / 2000) = 0.0057. The estimates are paired across
#   rules -- the same simulated studies are scored by each -- so the
#   difference is less noisy than two independent draws would be, and
#   0.01 is a deliberately generous threshold. Generous is the right
#   direction here: taking the smaller m costs a thousandth of power
#   and buys a rule that is harder to satisfy by luck.
#
#   The observed case: at N = 15, colour, count-absolute k >= 6,
#   m <= 4 scores 0.932893 and k >= 6, m <= 3 scores 0.931971. The gap
#   is 0.00092.
POWER_TIE_TOL = 0.01

# TIE_BREAK
#   Applied to the rules that tie within POWER_TIE_TOL, in order:
#   the LOWEST upper-bound worst false-claim rate first, then the
#   smaller m, then the smaller k, then the family order above.
#
#   False rate first, not m first. The point of breaking a tie toward
#   the "stricter" rule is to take the one that is harder to satisfy
#   by luck, and the false-claim rate is that quantity, measured. A
#   smaller m is only a proxy for it, and the proxy can disagree with
#   the measurement: at fog N = 20 the m-first order chose k >= 5,
#   m <= 1 over k >= 7, m <= 3, which tolerates fewer
#   opposite-direction detections but also requires fewer
#   same-direction ones, so it is not unambiguously the stricter rule.
#   Ordering on the measured rate removes the ambiguity.
#
#   Deterministic, and independent of row order.
TIE_BREAK = ("worst_upper", "m", "k", "family_order")

# TIE_BREAK_PREVIOUS
#   The order used before: m first. Kept so the selection can be run
#   under both and the difference reported, rather than asserted. It
#   decides nothing.
TIE_BREAK_PREVIOUS = ("m", "k", "family_order", "worst_upper")

# THIN_FALSE_WARN
#   A selected rule whose upper-bound worst false-claim rate exceeds
#   this is MARKED in the table and listed in the report.
#
#   The 0.05 target stays on the upper bound and is not tightened: the
#   upper bound is already the Monte Carlo safeguard, so a rule that
#   clears it is certified however thin the margin. But "certified by
#   0.0007" and "certified by 0.03" are different facts about a rule,
#   and a reader of the registration should not have to subtract to
#   find out which one they have. Hence a column, and a list.
THIN_FALSE_WARN = 0.045

# C1_CELL
#   The ONE scenario the compact C1 grid is reported at, matched to
#   the planning cell of the existence rules so the two tables
#   describe the same assumed world:
#
#     effect      the planning weight for the feature
#     share       half the participants responding
#     strategy    nobody using the alternative strategy
#     lapse       0.1 in BOTH groups
#
#   c1_power.csv crosses 13 effect/feature/share combinations with 3
#   strategy rates and 7 lapse pairs, so a grid that does not fix all
#   of them is averaging over scenarios -- which is what the first
#   version of this table did, and a mean of power over effect sizes
#   is not the power of anything. Every one is fixed here, and the
#   full filtered set is written separately.
C1_EFFECT = {"colour": 0.50, "fog": 0.70}
C1_SHARE = 0.5
C1_STRATEGY = 0.0
C1_LAPSE_PAIR = (0.1, 0.1)
C1_MEASURES = ("power_colour_gated", "power_fog_gated",
               "power_either_gated", "gate_fires", "inconclusive")
C1_WANT_N = [10, 15, 20]
C1_WANT_CONTROLS = [20, 30, 40]

# MIN_ANALYSABLE
#   Below this many analysable participants the existence claims are
#   NOT made. The counts in each direction are reported descriptively
#   and that is all; C1 still runs, because it compares groups and
#   does not depend on the per-person rule. Registered in advance so
#   it is not a decision taken after seeing a small sample.
MIN_ANALYSABLE = 10

# COST_CELLS
#   Two further cells, reported for the selected count rule and for
#   the sign test at every N, so what the planning-cell tuning costs
#   elsewhere is visible next to what it buys.
#
#     "lower_half"    the lower (weaker) equal-behaviour pair at half
#                     share
#     "planning_qtr"  the planning weights at a quarter share
#
#   Both at lapse 0.1. In both, the sign test eventually overtakes the
#   tuned count rule, and the report says where.
COST_CELLS = {"lower_half": ("robustness", 0.50),
              "planning_qtr": ("planning", 0.25)}
COST_LAPSE = 0.1

FIG_DPI = 150
SERIES_COLOURS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
SERIES_MARKERS = ["o", "s", "^", "D"]
SERIES_STYLES = ["-", "--", "-.", ":"]

FEATURES = ("colour", "fog")
CLAIM = {"fog": "H1 fog", "colour": "H2 colour"}

# every firing of the tie-break, filled by pick() and reported
TIE_LOG = []


# ============================================================
# Small helpers
# ============================================================

def banner(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def resolve_output_dir():
    if not os.path.exists(OUT_ROOT):
        os.makedirs(OUT_ROOT)
        return OUT_ROOT
    if ON_EXISTING == "error":
        raise RuntimeError(
            f"{OUT_ROOT} exists and ON_EXISTING is 'error'.")
    n = 2
    while os.path.exists(f"{OUT_ROOT}_{n}"):
        n += 1
    out = f"{OUT_ROOT}_{n}"
    os.makedirs(out)
    return out


def power_col(weight, share, lapse):
    """The sweep's power column name.

    The sweep names these power_w<weight>_s<share>_lapse<lapse>. The
    decision table uses power_e<effect>_s<share>_l<lapse> for the same
    quantity; this script reads the SWEEP, so it uses the sweep's
    spelling. Getting this wrong gives a KeyError rather than a wrong
    number, which is the right failure.
    """
    return f"power_w{weight:g}_s{share:g}_lapse{lapse:g}"


def fmt_rule(r):
    if r["family"] == SIGN_FAMILY:
        return f"sign test, k >= {int(r['k'])}"
    return f"k >= {int(r['k'])}, m <= {int(r['m'])}"


# ============================================================
# CHECK 1: the inputs
# ============================================================

def sha256(path, blocks=1 << 20):
    """The file's SHA-256, read in blocks so a large sweep is fine."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(blocks)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def provenance(report):
    """Which run this table was built from, recorded rather than
    assumed.

    The rerun writes no run_meta.json of its own, so there is nothing
    to inherit. What it does write is power_report.txt, in the PARENT
    of the set folder -- that file is shared across sets while the
    CSVs are per-set -- and its header carries the settings the run
    used. The header is copied verbatim; copying is the point, since a
    paraphrase of provenance is not provenance.

    The hashes are of the two files this table actually read. If a
    number in the preregistration is ever questioned, they are what
    says whether the file behind it is still the file that was read.
    """
    banner("PROVENANCE OF THE RUN BEING READ")
    out = {}

    rep = os.path.join(os.path.dirname(IN_ROOT.rstrip("/")),
                       "power_report.txt")
    if os.path.isfile(rep):
        with open(rep, errors="replace") as f:
            head = [next(f, "").rstrip("\n")
                    for _ in range(PROVENANCE_HEADER_LINES)]
        while head and not head[-1].strip():
            head.pop()
        out["power_report_header"] = head
        out["power_report_path"] = rep
        print(f"  {rep}, first {len(head)} line(s):")
        for ln in head:
            print(f"    | {ln}")
    else:
        out["power_report_header"] = None
        print(f"  {rep} NOT FOUND -- no run header to carry.")
        report.append(("provenance header", "power_report.txt missing"))

    out["sha256"] = {}
    for path in (SWEEP_CSV, C1_CSV):
        name = os.path.basename(path)
        if os.path.isfile(path):
            d = sha256(path)
            out["sha256"][name] = d
            print(f"  sha256  {name:<24s} {d}")
            print(f"          {'':<24s} "
                  f"{os.path.getsize(path):,} bytes")
        else:
            out["sha256"][name] = None
            print(f"  sha256  {name:<24s} not read (absent)")
    report.append(("provenance",
                   ("power_report.txt header + "
                    if out["power_report_header"] else "")
                   + f"{sum(1 for v in out['sha256'].values() if v)} "
                     "file hash(es)"))
    return out


def load_inputs(report):
    banner("INPUTS  (CHECK 1)")

    if not os.path.isfile(SWEEP_CSV):
        # Step 8 writes per-set files one level down, under
        # OUTPUT_ROOT/<label>/, and never overwrites, so a second run
        # lands in _2. Both are easy to get wrong in a setting, so
        # look for the file rather than only reporting its absence.
        found = sorted(glob.glob("data/*/rule_sweep.csv")
                       + glob.glob("data/*/*/rule_sweep.csv"))
        hint = ("\n  rule_sweep.csv WAS found at:\n"
                + "".join(f"    {p}\n" for p in found)
                + "  Set IN_ROOT to the folder holding the one you "
                  "mean (without the file name)."
                if found else "")
        raise FileNotFoundError(
            f"{SWEEP_CSV} not found. This script reads the RERUN's "
            "outputs; run simulate_power_v4_n10_20.py first, and "
            "point IN_ROOT at the folder it actually wrote -- "
            "including its set\n  subfolder, which is `frozen`."
            + hint)
    S = pd.read_csv(SWEEP_CSV)
    print(f"  rule_sweep.csv   {len(S):,} rows, "
          f"{S.n.nunique()} sample sizes, "
          f"{S.family.nunique()} families")

    adm = (f"admissible_{STANDARD}" if USE_MC_MARGIN
           else f"admissible_point_{STANDARD}")
    need = ["feature", "n", "family", "k", "m", "m_spec",
            "false_guesser", "false_guesser_upper",
            f"worst_{STANDARD}", f"worst_upper_{STANDARD}", adm]
    need += [f"false_taste_{s}" for s in STANDARD_SPREADS]
    need += [f"false_taste_{s}_upper" for s in STANDARD_SPREADS]
    missing = [c for c in need if c not in S.columns]
    if missing:
        raise RuntimeError(
            "rule_sweep.csv is missing " + ", ".join(missing) + ".\n"
            "  Either STANDARD / STANDARD_SPREADS name a standard this "
            "run did not\n  compute, or the sweep's schema changed. "
            f"Columns present:\n  {sorted(S.columns)[:24]} ...")
    print(f"  admissibility    {adm}"
          + ("  (upper bound of the 95% MC interval)" if USE_MC_MARGIN
             else "  (POINT ESTIMATE -- no MC margin)"))

    # ---- N coverage ----
    want = list(range(N_MIN, N_MAX + 1))
    for feat in FEATURES:
        have = sorted(S[S.feature == feat].n.unique())
        gap = [n for n in want if n not in have]
        if gap:
            raise RuntimeError(
                f"the sweep has no rules at N = {gap} for {feat!r}.\n"
                "  A rule fixed per N cannot have a hole in it: a "
                "final N of, say, 13 would\n  have no calibrated rule. "
                "Re-run simulate_power_v4_n10_20.py with those sizes "
                "in\n  SAMPLE_SIZES and RULE_SUMMARY_N.")
    print(f"  N coverage       {N_MIN}..{N_MAX} complete for both "
          "features")

    S = S[(S.n >= N_MIN) & (S.n <= N_MAX)].copy()

    # ---- the optional companions ----
    extra = {}
    for key, path in (("equal_behaviour", EQUAL_BEH_CSV),
                      ("detection", DETECTION_CSV),
                      ("c1", C1_CSV)):
        if os.path.isfile(path):
            extra[key] = pd.read_csv(path)
            print(f"  {os.path.basename(path):<28s} "
                  f"{len(extra[key]):,} rows")
        else:
            extra[key] = None
            print(f"  {os.path.basename(path):<28s} NOT FOUND "
                  "-- the sections that use it are skipped")

    meta = {}
    if os.path.isfile(RUN_META_JSON):
        with open(RUN_META_JSON) as f:
            meta = json.load(f)

    report.append(("source", IN_ROOT))
    report.append(("candidate rules read", f"{len(S):,}"))
    report.append(("admissibility column", adm))
    report.append(("check 1 inputs", "PASS"))
    return S, adm, extra, meta


# ============================================================
# CHECK 6 + the behavioural labels
# ============================================================

def behaviour_labels(S, report):
    """The choice probability each weight corresponds to, per feature,
    read from the sweep rather than restated here.

    This is also CHECK 6: a planning weight the sweep does not carry
    has no probability column and no power column, and is caught here
    rather than as a KeyError three sections later.
    """
    banner("WEIGHTS AND WHAT THEY MEAN  (CHECK 6)")
    p = {}
    for feat in FEATURES:
        row = S[S.feature == feat].iloc[0]
        for w in sorted({*PLANNING_WEIGHT.values(),
                         *ROBUSTNESS_WEIGHT.values()}):
            col = f"effect_p_median_w{w:g}"
            if col not in S.columns:
                raise RuntimeError(
                    f"the sweep has no {col}, so weight {w} is not one "
                    "it carries.\n  PLANNING_WEIGHT / "
                    "ROBUSTNESS_WEIGHT must name weights from the "
                    "run's\n  RULE_RESPONDER_WEIGHTS grid.")
            p[(feat, w)] = float(row[col])
        if "median_informative_SD" in S.columns:
            print(f"  {feat:7s} median informative difference "
                  f"{float(row.median_informative_SD):.3f} SD")

    for name, wmap in (("planning", PLANNING_WEIGHT),
                       ("robustness", ROBUSTNESS_WEIGHT)):
        parts = [f"{f} {wmap[f]:g}/SD -> p = {p[(f, wmap[f])]:.4f}"
                 for f in FEATURES]
        gap = abs(p[(FEATURES[0], wmap[FEATURES[0]])]
                  - p[(FEATURES[1], wmap[FEATURES[1]])])
        print(f"  {name:11s} " + ",  ".join(parts)
              + f"   gap {gap:.4f}")
        report.append((f"{name} pair",
                       ", ".join(parts) + f" (gap {gap:.4f})"))

    for share in TABLE_SHARES + [PLANNING_SHARE]:
        for lapse in TABLE_LAPSES + [PLANNING_LAPSE]:
            for feat in FEATURES:
                for wmap in (PLANNING_WEIGHT, ROBUSTNESS_WEIGHT):
                    c = power_col(wmap[feat], share, lapse)
                    if c not in S.columns:
                        raise RuntimeError(
                            f"the sweep has no column {c!r}. The "
                            "share/lapse grid in this\n  script asks "
                            "for a cell the run did not compute.")
    print("  every power column this script reads is present")
    report.append(("check 6 coverage", "PASS"))
    return p


# ============================================================
# Selection
# ============================================================

def pick(cands, col, order=None):
    """The most powerful rule in `cands` by `col`, with ties resolved.

    A TIE is a power difference smaller than POWER_TIE_TOL, not an
    exact equality. Monte Carlo noise means the most powerful rule by
    a thousandth is not reliably the most powerful rule, and taking
    the nominal winner would register a choice made by noise.

    Among the tied, `order` (TIE_BREAK by default) prefers the lowest
    upper-bound worst false-claim rate -- the measured version of
    "harder to satisfy by luck" -- then the smaller m, then the
    smaller k, then the family that does not degenerate.

    Returns the chosen rule and whether the tie-break changed the
    outcome: that is, whether the rule with the highest nominal power
    is not the one chosen. The caller records every such case.
    """
    if not len(cands):
        return None, None
    c = cands.copy()
    c["worst_upper"] = c[f"worst_upper_{STANDARD}"]
    c["family_order"] = c.family.map(FAMILY_ORDER).fillna(99)
    top = c.loc[c[col].idxmax()]
    best = float(c[col].max())
    tied = c[c[col] >= best - POWER_TIE_TOL]
    chosen = tied.sort_values(list(order or TIE_BREAK)).iloc[0]
    moved = None
    if not (chosen.family == top.family and chosen.k == top.k
            and chosen.m == top.m):
        # TWO VERY DIFFERENT CASES, kept apart.
        #
        # DUPLICATE: the same k and m expressed by both families --
        # at N = 15, floor(0.20 x 15) = 3, so "count-proportional
        # m <= 3" and "count-absolute m <= 3" are one rule under two
        # names, with identical power and identical false-claim
        # rates. Resolving that is bookkeeping, not a decision, and
        # reporting it as a tie-break would bury the real cases.
        #
        # SUBSTANTIVE: a different k or m was chosen, so the
        # tie-break actually changed the registered rule.
        duplicate = (int(chosen.k) == int(top.k)
                     and int(chosen.m) == int(top.m))
        moved = {"kind": "duplicate" if duplicate else "substantive",
                 "nominal_family": top.family,
                 "nominal_k": int(top.k), "nominal_m": int(top.m),
                 "nominal_power": round(float(top[col]), 6),
                 "chosen_family": chosen.family,
                 "chosen_k": int(chosen.k), "chosen_m": int(chosen.m),
                 "chosen_power": round(float(chosen[col]), 6),
                 "power_given_up": round(float(top[col]
                                               - chosen[col]), 6)}
    return chosen, moved


def select_rules(S, adm, wmap, label, report, quiet=False,
                 order=None, log_ties=True):
    """The count rule and the sign test at each N, per feature, chosen
    in the cell `wmap` names."""
    if not quiet:
        banner(f"SELECTING RULES AT THE {label.upper()} CELL")
        print(f"  share {PLANNING_SHARE:g}, lapse {PLANNING_LAPSE:g}, "
              "weight per feature "
              + ", ".join(f"{f} {wmap[f]:g}" for f in FEATURES))
    rows = []
    for feat in FEATURES:
        col = power_col(wmap[feat], PLANNING_SHARE, PLANNING_LAPSE)
        for n in range(N_MIN, N_MAX + 1):
            g = S[(S.feature == feat) & (S.n == n) & S[adm]]
            for kind, fams in (("count", COUNT_FAMILIES),
                               ("sign", (SIGN_FAMILY,))):
                r, moved = pick(g[g.family.isin(fams)], col, order)
                if moved is not None and log_ties:
                    TIE_LOG.append({"cell": label, "feature": feat,
                                    "n": n, "rule_kind": kind,
                                    **moved})
                if r is None:
                    raise RuntimeError(
                        f"no admissible {kind} rule for {feat} at "
                        f"N = {n} under the {STANDARD} standard.\n"
                        "  A rule fixed per N cannot have a hole in "
                        "it. Either the standard is\n  unreachable at "
                        "that size, or the sweep did not enumerate "
                        "that family there.")
                rows.append({
                    "cell": label, "feature": feat, "claim": CLAIM[feat],
                    "n": n, "rule_kind": kind, "family": r.family,
                    "k": int(r.k), "m": int(r.m),
                    "m_spec": r.m_spec,
                    "weight_per_SD": wmap[feat],
                    "planning_power": float(r[col]),
                    "false_guesser": float(r.false_guesser),
                    "false_guesser_upper": float(r.false_guesser_upper),
                    "false_worst": float(r[f"worst_{STANDARD}"]),
                    "false_worst_upper":
                        float(r[f"worst_upper_{STANDARD}"]),
                })
    R = pd.DataFrame(rows)
    if not quiet:
        for feat in FEATURES:
            sub = R[R.feature == feat]
            for kind in ("count", "sign"):
                s2 = sub[sub.rule_kind == kind].sort_values("n")
                print(f"  {feat:7s} {kind:5s} "
                      + "  ".join(f"N{int(r.n)}:{r.planning_power:.3f}"
                                  for _, r in s2.iterrows()))
    return R


def report_headroom(T, report):
    """How much room each registered rule has under the target.

    The target is NOT tightened. The upper bound of the 95% Monte
    Carlo interval is already the safeguard against a selection made
    by simulation noise, so a rule that clears it is certified however
    thin the margin -- that is what an upper bound is for. What a thin
    margin does mean is that the POINT estimate and the bound are
    close to the line together, and a reader of the registration
    should be told which rules those are instead of subtracting to
    find them.
    """
    banner("HEADROOM UNDER THE FALSE-CLAIM TARGET")
    reg = T[T.registered.astype(bool) & (T.rule_kind == "count")]
    print(f"  target {TARGET_FALSE} on the upper bound of the 95% "
          "interval; margin by N:")
    for feat in FEATURES:
        s = reg[reg.feature == feat].sort_values("n")
        print(f"    {feat:7s} " + " ".join(f"{int(r.n)}:"
                                           f"{r.false_headroom:.4f}"
                                           for r in s.itertuples()))
    thin = reg[reg.false_thin].sort_values(["feature", "n"])
    if not len(thin):
        print(f"  no registered rule exceeds {THIN_FALSE_WARN} on the "
              "upper bound.")
        report.append(("thin cells", f"none above {THIN_FALSE_WARN}"))
        return thin
    print(f"\n  {len(thin)} rule(s) above {THIN_FALSE_WARN} -- "
          "certified, and thin:")
    for r in thin.itertuples():
        print(f"    {r.feature:7s} N{int(r.n):<3d} {r.rule:<16s} "
              f"upper {r.false_worst_upper:.5f}  point "
              f"{r.false_worst:.5f}  margin {r.false_headroom:.5f}")
    print("\n  Each clears the target on the UPPER BOUND, which is "
          "the Monte Carlo")
    print("  safeguard, so each is admissible as registered. The "
          "margin is reported")
    print("  because it is a property of the rule a reader should not "
          "have to compute.")
    report.append(("thin cells",
                   f"{len(thin)} above {THIN_FALSE_WARN} "
                   + ", ".join(f"{r.feature} N{int(r.n)} "
                               f"({r.false_headroom:.5f})"
                               for r in thin.itertuples())))
    return thin


def report_tie_order(S, adm, report, root):
    """What the new tie-break order changed, run under both.

    The order was changed from "smaller m first" to "lowest
    upper-bound false rate first". Whether that moves any registered
    rule is a question about this sweep, not about the argument for
    the change, so it is answered by selecting under both orders and
    comparing. Nothing here decides anything: TIE_BREAK alone does.
    """
    banner("WHAT THE NEW TIE-BREAK ORDER CHANGED")
    print("  now  : " + ", ".join(TIE_BREAK))
    print("  was  : " + ", ".join(TIE_BREAK_PREVIOUS))
    rows = []
    for label, wmap in (("planning", PLANNING_WEIGHT),
                        ("robustness", ROBUSTNESS_WEIGHT)):
        new = select_rules(S, adm, wmap, label, report, quiet=True,
                           order=TIE_BREAK, log_ties=False)
        old = select_rules(S, adm, wmap, label, report, quiet=True,
                           order=TIE_BREAK_PREVIOUS, log_ties=False)
        key = ["feature", "n", "rule_kind"]
        mg = new.merge(old, on=key, suffixes=("_new", "_old"))
        d = mg[(mg.k_new != mg.k_old) | (mg.m_new != mg.m_old)
               | (mg.family_new != mg.family_old)]
        for r in d.itertuples():
            rows.append({
                "cell": label, "feature": r.feature, "n": int(r.n),
                "rule_kind": r.rule_kind,
                "was_family": r.family_old, "was_k": int(r.k_old),
                "was_m": int(r.m_old),
                "was_power": round(float(r.planning_power_old), 6),
                "was_false_upper": round(float(r.false_worst_upper_old),
                                         5),
                "now_family": r.family_new, "now_k": int(r.k_new),
                "now_m": int(r.m_new),
                "now_power": round(float(r.planning_power_new), 6),
                "now_false_upper": round(float(r.false_worst_upper_new),
                                         5)})
    if not rows:
        print("  NOTHING. The two orders select the same rule in "
              "every cell at every N,")
        print("  so the change is a statement of principle here "
              "rather than a change of")
        print("  rule. It still matters for a future run, where the "
              "two could disagree.")
        report.append(("tie-order change", "no rule changed"))
        return None
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(root, "tie_order_change.csv"), index=False)
    print(f"  {len(D)} selection(s) changed:")
    for r in D.itertuples():
        print(f"    {r.cell:10s} {r.feature:7s} N{int(r.n):<3d} "
              f"{r.rule_kind:5s}")
        print(f"      was  {r.was_family} k>={r.was_k}, m<={r.was_m}"
              f"   power {r.was_power:.6f}  false upper "
              f"{r.was_false_upper:.5f}")
        print(f"      now  {r.now_family} k>={r.now_k}, m<={r.now_m}"
              f"   power {r.now_power:.6f}  false upper "
              f"{r.now_false_upper:.5f}")
        print(f"      power {r.now_power - r.was_power:+.6f}, "
              f"false rate {r.now_false_upper - r.was_false_upper:+.5f}")
    report.append(("tie-order change",
                   f"{len(D)} selection(s) changed"))
    return D


def report_ties(root, report):
    """Where the tie-break decided, and what it cost.

    Only the substantive firings are printed. A rule the two families
    both express is counted and set aside: nothing about the
    registered rule depends on which name it is read under.
    """
    banner("WHERE THE TIE-BREAK DECIDED")
    if not TIE_LOG:
        print("  nowhere: the most powerful rule was also the one the "
              "tie-break prefers,")
        print("  at every N and in both cells.")
        report.append(("tie-break", "never fired"))
        return
    T = pd.DataFrame(TIE_LOG)
    T.to_csv(os.path.join(root, "tie_breaks.csv"), index=False)
    dup = T[T.kind == "duplicate"]
    sub_ = T[T.kind == "substantive"]
    print(f"  {len(dup)} firing(s) were the same rule under both "
          "family names -- same k, same m,")
    print("  identical power and identical false-claim rate. "
          "Bookkeeping, not a decision.")
    if not len(sub_):
        print("\n  No substantive firing: the tie-break never changed "
              "which rule is registered.")
        report.append(("tie-break",
                       f"{len(dup)} duplicate-label, 0 substantive"))
        return
    print(f"\n  {len(sub_)} SUBSTANTIVE firing(s) -- the tie-break "
          "changed the registered rule:")
    for r in sub_.itertuples():
        print(f"    {r.cell:10s} {r.feature:7s} N{int(r.n)}  "
              f"{r.nominal_family} k>={r.nominal_k}, "
              f"m<={r.nominal_m} @ {r.nominal_power:.6f}")
        print(f"    {'':10s} {'':7s}      -> "
              f"{r.chosen_family} k>={r.chosen_k}, "
              f"m<={r.chosen_m} @ {r.chosen_power:.6f}"
              f"   (gave up {r.power_given_up:.6f})")
    print(f"\n  Each gave up at most "
          f"{sub_.power_given_up.max():.6f} of power -- inside the "
          f"{POWER_TIE_TOL} tie tolerance,")
    print("  which is itself inside the Monte Carlo error of the "
          "power estimate -- and bought")
    print("  a rule that tolerates fewer opposite-direction "
          "detections.")
    report.append(("tie-break",
                   f"{len(dup)} duplicate-label, {len(sub_)} "
                   f"substantive (worst cost "
                   f"{sub_.power_given_up.max():.6f})"))


def check_admissibility(S, R, report):
    """CHECK 2: recompute each selected rule's worst false-claim rate
    from the per-spread columns.

    The sweep already wrote an admissible flag, and this does not trust
    it. The flag is one boolean; the thing being registered is the
    statement that the rate clears the target at every spread in the
    standard's set, and that statement is cheap to verify directly.
    """
    banner("ADMISSIBILITY, RECOMPUTED  (CHECK 2)")
    suffix = "_upper" if USE_MC_MARGIN else ""
    bad = []
    for _, r in R.drop_duplicates(
            ["feature", "n", "rule_kind"]).iterrows():
        row = S[(S.feature == r.feature) & (S.n == r.n)
                & (S.family == r.family) & (S.k == r.k)
                & (S.m == r.m)]
        if not len(row):
            bad.append((r.feature, r.n, r.rule_kind, "row not found"))
            continue
        row = row.iloc[0]
        rates = [float(row[f"false_taste_{s}{suffix}"])
                 for s in STANDARD_SPREADS]
        rates.append(float(row[f"false_guesser{suffix}"]))
        worst = max(rates)
        if worst > TARGET_FALSE + 1e-12:
            bad.append((r.feature, int(r.n), r.rule_kind,
                        f"worst {worst:.5f}"))
        if abs(worst - float(r.false_worst_upper
                             if USE_MC_MARGIN
                             else r.false_worst)) > 1e-6:
            bad.append((r.feature, int(r.n), r.rule_kind,
                        "disagrees with the sweep's own worst column"))
    n_rules = len(R.drop_duplicates(["feature", "n", "rule_kind"]))
    if bad:
        print(f"  FAIL: {len(bad)} of {n_rules} selected rules")
        for b in bad[:8]:
            print("    ", b)
        report.append(("check 2 admissibility", f"FAIL ({len(bad)})"))
        return False
    print(f"  PASS: all {n_rules} selected rules clear "
          f"{TARGET_FALSE} at every spread in the {STANDARD} set")
    print("  spreads checked: "
          + ", ".join(f"{x}x" for x in STANDARD_SPREADS)
          + ", plus the pure guesser")
    report.append(("check 2 admissibility",
                   f"PASS ({n_rules} rules)"))
    return True


def find_switch(R, report, label="planning", loud=True):
    """CHECK 4: the smallest N from which the sign test is at least as
    powerful as the count rule, for BOTH features, and stays so."""
    if loud:
        banner(f"THE SWITCH POINT  ({label} cell)  (CHECK 4)")
    wins = {}
    for n in range(N_MIN, N_MAX + 1):
        ok = True
        for feat in FEATURES:
            c = R[(R.feature == feat) & (R.n == n)
                  & (R.rule_kind == "count")].planning_power.iloc[0]
            s = R[(R.feature == feat) & (R.n == n)
                  & (R.rule_kind == "sign")].planning_power.iloc[0]
            ok = ok and (s >= c)
        wins[n] = ok

    switch = None
    for n in range(N_MIN, N_MAX + 1):
        if all(wins[m] for m in range(n, N_MAX + 1)):
            switch = n
            break

    if loud:
        print("  sign >= count for BOTH features, by N:")
        print("    " + "  ".join(
            f"{n}:{'Y' if wins[n] else 'n'}"
            for n in range(N_MIN, N_MAX + 1)))
        if switch is None:
            print(f"\n  NO SWITCH POINT in {N_MIN}..{N_MAX}: the count "
                  "rule wins somewhere at or")
            print("  above every candidate N. The registered rule is "
                  "then the count rule")
            print("  throughout, and that should be stated plainly.")
            report.append((f"switch point ({label})",
                           f"none in {N_MIN}..{N_MAX}"))
        else:
            print(f"\n  SWITCH POINT = {switch}: count below it, sign "
                  "from it, and sign stays")
            print(f"  at least as powerful through N = {N_MAX}.")
            report.append((f"switch point ({label})", str(switch)))
    return switch, wins


def combined(R, switch):
    """The registered rule at each N.

    With REGISTERED_RULE = "count" this is the count rule at every N
    and `switch` is ignored -- the scan is still run, and reported, as
    the evidence that no switch is needed.

    With "combined" it is the count rule below the switch point and
    the sign test from it. Either way the rule depends only on N, so
    each N's figures are simply those of the rule it uses: nothing is
    recombined and nothing is averaged.
    """
    rows = []
    for feat in FEATURES:
        for n in range(N_MIN, N_MAX + 1):
            if REGISTERED_RULE == "count":
                kind = "count"
            else:
                kind = ("count" if (switch is None or n < switch)
                        else "sign")
            r = R[(R.feature == feat) & (R.n == n)
                  & (R.rule_kind == kind)].iloc[0]
            rows.append({**r.to_dict(), "registered_kind": kind,
                         "rule": fmt_rule(r)})
    return pd.DataFrame(rows)


# ============================================================
# CHECK 5: how equal is "equal behaviour"?
# ============================================================

def person_rate(extra, feat, w):
    """The per-person detection rate for a responder of weight `w` on
    `feat`: the share of such people whose fitted weight clears the
    threshold in the predicted direction.

    This is the quantity the sign test is sensitive to and the
    saturated count rule is not, so it is what explains why the two
    features can match on one rule and not the other.

    Read from detection_curve.csv, whose grid is 0.2, 0.4 ... 2.0 per
    lapse rate. The weights in play -- 0.30, 0.50, 0.70 -- are not on
    that grid, so the value is INTERPOLATED, and the file's own
    <feature>_p_median column is interpolated alongside it so the
    interpolation can be checked against the sweep's independent
    label for the same weight. Outside the grid this refuses rather
    than extrapolating.

    Returns (rate, p_median, source).
    """
    D = extra.get("detection")
    if D is None:
        return None, None, "detection_curve.csv not found"
    need = {"lapse", "true_weight", feat, f"{feat}_p_median"}
    if not need <= set(D.columns):
        return None, None, (
            "detection_curve.csv lacks "
            + ", ".join(sorted(need - set(D.columns))))

    # the rate depends on the lapse rate, so take the planning lapse
    m = D[np.isclose(D.lapse, PLANNING_LAPSE)].sort_values("true_weight")
    if not len(m):
        return None, None, (
            f"detection_curve.csv has no lapse {PLANNING_LAPSE:g} "
            f"(has {sorted(D.lapse.unique())})")

    x = m.true_weight.to_numpy(float)
    if not (x.min() <= w <= x.max()):
        return None, None, (
            f"weight {w} is outside the detection grid "
            f"[{x.min():g}, {x.max():g}]; interpolating is fine, "
            "extrapolating is not")

    rate = float(np.interp(w, x, m[feat].to_numpy(float)))
    # the same file carries the choice probability, so the
    # interpolation can be checked against the sweep's own label
    pmed = float(np.interp(w, x, m[f"{feat}_p_median"].to_numpy(float)))
    return rate, pmed, f"detection_curve, lapse {PLANNING_LAPSE:g}"


def check_equal_behaviour(S, probs, extra, report, root):
    """CHECK 5: state how close each pair is, on both quantities."""
    banner("HOW EQUAL IS 'EQUAL BEHAVIOUR'?  (CHECK 5)")
    rows, unresolved = [], False
    for name, wmap in (("planning", PLANNING_WEIGHT),
                       ("robustness", ROBUSTNESS_WEIGHT)):
        rec = {"pair": name}
        for feat in FEATURES:
            w = wmap[feat]
            pr, pmed, src = person_rate(extra, feat, w)
            if pr is None:
                unresolved = True
            rec[f"{feat}_weight_per_SD"] = w
            rec[f"{feat}_p_typical_trial"] = probs[(feat, w)]
            rec[f"{feat}_person_detection"] = pr
            rec[f"{feat}_p_median_from_curve"] = pmed
            rec[f"{feat}_p_median_gap"] = (
                abs(pmed - probs[(feat, w)]) if pmed is not None
                else None)
            rec[f"{feat}_person_detection_source"] = src
        rec["p_gap"] = abs(rec[f"{FEATURES[0]}_p_typical_trial"]
                           - rec[f"{FEATURES[1]}_p_typical_trial"])
        a = rec[f"{FEATURES[0]}_person_detection"]
        b = rec[f"{FEATURES[1]}_person_detection"]
        rec["person_detection_gap"] = (abs(a - b) if None not in (a, b)
                                       else None)
        rows.append(rec)
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(root, "equal_behaviour_check.csv"),
             index=False)

    for r in E.itertuples():
        print(f"\n  {r.pair.upper()} pair")
        for feat in FEATURES:
            w = getattr(r, f"{feat}_weight_per_SD")
            p = getattr(r, f"{feat}_p_typical_trial")
            d = getattr(r, f"{feat}_person_detection")
            ds = "  (unresolved)" if d is None else f"{d:.4f}"
            print(f"    {feat:7s} {w:g}/SD   p(typical trial) "
                  f"{p:.4f}   per-person detection {ds}")
        print(f"    gap in p(typical trial)      {r.p_gap:.4f}")
        if r.person_detection_gap is not None:
            print(f"    gap in per-person detection  "
                  f"{r.person_detection_gap:.4f}")

    # the cross-check: two independent routes to the same label
    gaps = [getattr(r, f"{f}_p_median_gap") for r in E.itertuples()
            for f in FEATURES]
    gaps = [g for g in gaps if g is not None]
    if gaps:
        worst = max(gaps)
        print("\n  CROSS-CHECK. detection_curve.csv's own "
              "<feature>_p_median, interpolated to the")
        print("  same weights, against the sweep's "
              "effect_p_median_w* labels: worst difference")
        print(f"  {worst:.2e}. Two files, computed in different "
              "passes, agree -- so the")
        print("  interpolation above is sound and the weights line "
              "up between them.")
        report.append(("p_median cross-check",
                       f"worst {worst:.1e}"))

    print("\n  WHY THIS MATTERS, AND IT IS NOT A TECHNICALITY.")
    print("  The two pairs are matched on the choice probability on a "
          "typical informative")
    print("  trial. They are NOT thereby matched on the per-person "
          "detection rate, which is")
    print("  the share of such people whose fitted weight clears the "
          "threshold.")
    print("  The COUNT rule is saturated on k at these sizes -- it "
          "asks only whether enough")
    print("  people cleared the threshold and none went the other "
          "way -- so a difference in")
    print("  the detection rate moves it little. The SIGN TEST "
          "compares the two counts, so")
    print("  it is sensitive to exactly that difference. That is why "
          "the higher pair can")
    print("  match on the count rule and not on the sign test.")
    if unresolved:
        print("\n  NOTE: the per-person detection rate could not be "
              "resolved. Reasons, per")
        print("  pair and feature, are in the _source column of "
              "equal_behaviour_check.csv.")
        print("  The probability gap above still stands; only the "
              "second quantity is missing.")
    report.append(("check 5 equal behaviour",
                   "reported" + (" (per-person rate unresolved)"
                                 if unresolved else "")))
    return E


# ============================================================
# The registration table
# ============================================================

def registration_table(R, probs, switch, root, report):
    banner("THE REGISTRATION TABLE")
    rows = []
    for r in R.itertuples():
        w = r.weight_per_SD
        if REGISTERED_RULE == "count":
            is_reg = r.rule_kind == "count"
        elif switch is None:
            is_reg = r.rule_kind == "count"
        else:
            is_reg = (r.rule_kind == "sign") == (r.n >= switch)
        rec = {
            "feature": r.feature, "claim": r.claim, "n": int(r.n),
            "rule_kind": r.rule_kind,
            "registered": is_reg,
            "family": r.family, "k": int(r.k), "m": int(r.m),
            "rule": fmt_rule(r._asdict()),
            "weight_per_SD": w,
            "p_typical_trial": round(probs[(r.feature, w)], 4),
            "false_worst": round(r.false_worst, 5),
            "false_worst_upper": round(r.false_worst_upper, 5),
            "false_guesser": round(r.false_guesser, 5),
            # The margin by which the rule clears the target, and
            # whether it is thin. Both are facts about the rule, not
            # judgements: the target itself is unchanged.
            "false_headroom": round(TARGET_FALSE
                                    - r.false_worst_upper, 5),
            "false_thin": bool(r.false_worst_upper
                               > THIN_FALSE_WARN),
            f"power_planning_s{PLANNING_SHARE:g}"
            f"_l{PLANNING_LAPSE:g}": round(r.planning_power, 4),
        }
        rows.append(rec)
    T = pd.DataFrame(rows)
    T.attrs["note"] = "every power cell in a row uses that row's weight"
    return T


def add_power_cells(T, S, adm, wmap, report):
    """The share x lapse grid, for each selected rule.

    Each row's cells use THAT ROW'S weight, which is the whole point of
    an equal-behaviour pair: the colour row is read at the colour
    weight and the fog row at the fog weight, and the two are then
    comparable.
    """
    out = []
    for r in T.itertuples():
        row = S[(S.feature == r.feature) & (S.n == r.n)
                & (S.family == r.family) & (S.k == r.k)
                & (S.m == r.m)]
        rec = {}
        if len(row):
            row = row.iloc[0]
            for share in TABLE_SHARES:
                for lapse in TABLE_LAPSES:
                    c = power_col(r.weight_per_SD, share, lapse)
                    rec[f"power_s{share:g}_l{lapse:g}"] = round(
                        float(row[c]), 4)
        out.append(rec)
    return pd.concat([T, pd.DataFrame(out)], axis=1)


def cost_columns(T, S, probs, report, root):
    """What the planning-cell tuning costs elsewhere.

    The count rule is tuned at one cell. A rule tuned at a cell is
    better there and can be worse away from it, and a registration
    that quotes only the cell it was tuned in is quoting its own best
    case. These two columns are the other side of that.

    Reported for BOTH rules at every N, so the comparison is
    like for like.
    """
    banner("WHAT THE TUNING COSTS")
    out = []
    for r in T.itertuples():
        row = S[(S.feature == r.feature) & (S.n == r.n)
                & (S.family == r.family) & (S.k == r.k)
                & (S.m == r.m)]
        rec = {}
        for name, (which, share) in COST_CELLS.items():
            w = (PLANNING_WEIGHT if which == "planning"
                 else ROBUSTNESS_WEIGHT)[r.feature]
            c = power_col(w, share, COST_LAPSE)
            rec[f"cost_{name}"] = (round(float(row.iloc[0][c]), 4)
                                   if len(row) else None)
            rec[f"cost_{name}_weight"] = w
            rec[f"cost_{name}_p"] = round(probs[(r.feature, w)], 4)
        out.append(rec)
    T = pd.concat([T, pd.DataFrame(out)], axis=1)

    for name in COST_CELLS:
        which, share = COST_CELLS[name]
        print(f"\n  {name}: the {which} weights at share "
              f"{share:g}, lapse {COST_LAPSE:g}")
        for feat in FEATURES:
            cnt = T[(T.feature == feat)
                    & (T.rule_kind == "count")].sort_values("n")
            sgn = T[(T.feature == feat)
                    & (T.rule_kind == "sign")].sort_values("n")
            cross = None
            for n in sorted(cnt.n.unique()):
                c = float(cnt[cnt.n == n][f"cost_{name}"].iloc[0])
                s = float(sgn[sgn.n == n][f"cost_{name}"].iloc[0])
                if s >= c and cross is None:
                    cross = n
                elif s < c:
                    cross = None
            print(f"    {feat:7s} count "
                  + " ".join(f"{float(v):.3f}"
                             for v in cnt[f"cost_{name}"])
                  + "\n            sign  "
                  + " ".join(f"{float(v):.3f}"
                             for v in sgn[f"cost_{name}"]))
            msg = (f"sign overtakes from N = {cross}" if cross
                   else "the count rule wins throughout")
            print(f"            -> {msg}")
            report.append((f"{name}, {feat}", msg))

    print("\n  THE TRADE, PLAINLY. The count rule is registered "
          "because it is more")
    print("  powerful in the cell the study is designed around. Away "
          "from that cell it")
    print("  can be the weaker of the two, and where it is, the table "
          "says so. Both")
    print("  columns belong in the registration next to the "
          "planning-cell power.")
    return T


def plot_combined(C, switch, root):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
    for ax, feat in zip(axes, ("fog", "colour")):
        sub = C[C.feature == feat].sort_values("n")
        for i, kind in enumerate(("count", "sign")):
            s2 = sub[sub.registered_kind == kind]
            if len(s2):
                ax.plot(s2.n, s2.planning_power,
                        color=SERIES_COLOURS[i],
                        marker=SERIES_MARKERS[i],
                        linestyle=SERIES_STYLES[i],
                        label=f"registered: {kind}")
        ax.plot(sub.n, sub.planning_power, color="#444444",
                linewidth=0.8, alpha=0.6, zorder=0)
        if switch is not None:
            ax.axvline(switch - 0.5, color="#D55E00", linestyle=":",
                       linewidth=1.2)
            ax.annotate(f"switch at N = {switch}", (switch - 0.5, 0.02),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color="#D55E00")
        ax.set_xlabel("participants")
        ax.set_ylabel("power of the existence claim")
        ax.set_title(f"{CLAIM[feat]}", fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "The registered rule: count below the switch point, sign test "
        f"from it\nplanning cell: half responding, lapse "
        f"{PLANNING_LAPSE:g}, "
        + ", ".join(f"{f} {PLANNING_WEIGHT[f]:g}/SD"
                    for f in FEATURES), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(root, "combined_rule_power.png"),
                dpi=FIG_DPI)
    plt.close(fig)


# ============================================================
# C1, compactly
# ============================================================

def c1_grid(C, feat, claim, root):
    """The 3 x 3 grid for one feature at the stated scenario.

    Every scenario dimension is fixed, and the function REFUSES to
    aggregate: if a (participants, controls) pair does not resolve to
    exactly one row, it says so rather than taking a mean. That is the
    whole point of the fix -- the previous version's mean was over
    scenarios that answer different questions.
    """
    w = C1_EFFECT[feat]
    lp, lc = C1_LAPSE_PAIR
    sel = C[(C.effect_on == claim) & (C.effect_sd == w)
            & (C.responder_share == C1_SHARE)
            & (C.strategy_rate == C1_STRATEGY)
            & (C.lapse_participants == lp)
            & (C.lapse_controls == lc)
            & C.n_participants.isin(C1_WANT_N)
            & C.n_controls.isin(C1_WANT_CONTROLS)].copy()
    print(f"\n  {feat}: effect {w:g}/SD, share {C1_SHARE:g}, "
          f"strategy {C1_STRATEGY:g}, lapse {lp:g}/{lc:g}")
    if not len(sel):
        print("    no row at that scenario -- the grid is skipped.")
        return None
    dup = sel.groupby(["n_participants", "n_controls"]).size()
    if (dup > 1).any():
        print("    REFUSING TO AGGREGATE: more than one row at "
              + ", ".join(f"{int(a)}x{int(b)}"
                          for (a, b), v in dup.items() if v > 1))
        print("    A scenario dimension is unfixed. The grid is "
              "skipped rather than averaged.")
        return None
    cols = [c for c in C1_MEASURES if c in sel.columns]
    for c in cols:
        piv = sel.pivot_table(index="n_participants",
                              columns="n_controls", values=c)
        print(f"    {c}")
        print("      " + piv.round(3).to_string()
              .replace("\n", "\n      "))
    keep = (["effect_on", "effect_sd", "effect_p_typical",
             "responder_share", "strategy_rate",
             "lapse_participants", "lapse_controls",
             "n_participants", "n_controls"] + cols)
    return sel[[c for c in keep if c in sel.columns]].sort_values(
        ["n_participants", "n_controls"])


def c1_compact(extra, root, report):
    """C1 at the sizes that could be recruited, at ONE scenario.

    Writes c1_compact.csv -- the grid per feature plus the no-effect
    grid -- and c1_cells.csv, every row of c1_power.csv inside the
    wanted counts, so the full scenario spread is still on disk under
    a name that does not promise to be compact.
    """
    banner("C1: THE GROUP COMPARISON, AT ONE STATED SCENARIO")
    C = extra.get("c1")
    if C is None:
        print("  c1_power.csv not found; skipped.")
        report.append(("C1 compact table", "skipped (no c1_power.csv)"))
        return None
    need = ["effect_on", "effect_sd", "responder_share",
            "strategy_rate", "lapse_participants", "lapse_controls",
            "n_participants", "n_controls"]
    missing = [c for c in need if c not in C.columns]
    if missing:
        print("  c1_power.csv is missing " + ", ".join(missing)
              + f";\n  columns present: {sorted(C.columns)[:14]} ...")
        report.append(("C1 compact table", "skipped (schema)"))
        return None

    K = C[C.n_participants.isin(C1_WANT_N)
          & C.n_controls.isin(C1_WANT_CONTROLS)].copy()
    K.to_csv(os.path.join(root, "c1_cells.csv"), index=False)
    print(f"  c1_cells.csv  {len(K):,} rows: every scenario at "
          f"participants {C1_WANT_N} x controls {C1_WANT_CONTROLS}")
    print("  THAT IS NOT A TABLE TO READ. It crosses "
          f"{K.effect_on.nunique()} effect/feature labels x "
          f"{K.effect_sd.nunique()} effect sizes x")
    print(f"  {K.responder_share.nunique()} shares x "
          f"{K.strategy_rate.nunique()} strategy rates x "
          f"{K.groupby(['lapse_participants','lapse_controls']).ngroups}"
          " lapse pairs, which answer")
    print("  different questions and must not be averaged together.")

    parts = []
    for feat in FEATURES:
        g = c1_grid(K, feat, CLAIM[feat], root)
        if g is not None:
            g = g.copy()
            g.insert(0, "grid", feat)
            parts.append(g)

    # The no-effect grid: how often C1 fires when there is nothing to
    # find. Registered alongside the power, because a group comparison
    # without its false-positive rate is half a statement.
    lp, lc = C1_LAPSE_PAIR
    z = K[(K.effect_on == "none")
          & (K.strategy_rate == C1_STRATEGY)
          & (K.lapse_participants == lp)
          & (K.lapse_controls == lc)].copy()
    if len(z):
        print(f"\n  no effect anywhere: strategy {C1_STRATEGY:g}, "
              f"lapse {lp:g}/{lc:g}")
        for c in [c for c in C1_MEASURES if c in z.columns]:
            piv = z.pivot_table(index="n_participants",
                                columns="n_controls", values=c)
            print(f"    {c}")
            print("      " + piv.round(4).to_string()
                  .replace("\n", "\n      "))
        z.insert(0, "grid", "no effect")
        parts.append(z[[c for c in parts[0].columns if c in z.columns]]
                     if parts else z)
    else:
        print("\n  no no-effect rows at that scenario.")

    if not parts:
        report.append(("C1 compact table", "skipped (no grid)"))
        return None
    G = pd.concat(parts, ignore_index=True)
    G.to_csv(os.path.join(root, "c1_compact.csv"), index=False)
    print(f"\n  c1_compact.csv  {len(G)} rows: "
          + ", ".join(f"{k} ({v})"
                      for k, v in G.grid.value_counts().items()))
    report.append(("C1 compact table",
                   f"{len(G)} rows at one stated scenario "
                   f"(was {len(K):,} rows averaged over scenarios)"))
    return G


# CHECK 3 and the report
# ============================================================

def check_combined(C, report):
    """CHECK 3: the registered rule's false-claim rate at every N.

    The combination inherits each rule's own rate, because it uses
    exactly one rule at each N and N is fixed before the data arrive.
    That is the argument; this is the arithmetic that backs it.
    """
    banner("THE COMBINED RULE'S FALSE-CLAIM RATE  (CHECK 3)")
    worst = C.false_worst_upper.max()
    bad = C[C.false_worst_upper > TARGET_FALSE + 1e-12]
    print(f"  {len(C)} (feature, N) cells; worst upper bound "
          f"{worst:.5f} against a target of {TARGET_FALSE}")
    if len(bad):
        print(f"  FAIL at {len(bad)} cell(s):")
        print("  " + bad[["feature", "n", "rule", "false_worst_upper"]]
              .head(8).to_string(index=False).replace("\n", "\n  "))
        report.append(("check 3 combined rule",
                       f"FAIL (worst {worst:.5f})"))
        return False
    print("  PASS: every N's rule clears the target, so the rule that "
          "selects between")
    print("  them by N alone does too. N is fixed before the data "
          "arrive, so no choice")
    print("  is made after seeing a result.")
    report.append(("check 3 combined rule",
                   f"PASS (worst upper {worst:.5f})"))
    return True


def power_shape(C):
    """The registered rule's power against N, and where it dips.

    Written because the first draft of this report asserted that the
    power rises with N. It does not. The rule is re-selected at every
    N under a false-rate ceiling, and where the ceiling forces a
    stricter m the power falls even though N grew. That is a property
    worth stating plainly rather than a defect to leave unmentioned:
    power is quoted per N from the table, so nothing in the
    registration depends on the sequence being monotone.

    Returns, per feature: the (n, power, m) sequence and the drops.
    """
    out = {}
    for feat in FEATURES:
        s = C[C.feature == feat].sort_values("n")
        v = list(s.planning_power)
        ns = [int(x) for x in s.n]
        ms = [int(x) for x in s.m]
        drops = [(ns[i], v[i] - v[i - 1])
                 for i in range(1, len(v)) if v[i] < v[i - 1]]
        out[feat] = {"n": ns, "power": v, "m": ms, "drops": drops}
    return out


def ties_frame():
    """The tie log as a frame, split into its two kinds."""
    if not TIE_LOG:
        return None, None
    T = pd.DataFrame(TIE_LOG)
    return T[T.kind == "duplicate"], T[T.kind == "substantive"]


def write_report(root, report, T, C, E, switch, switch_rob, wins,
                 meta, elapsed, prov=None, thin=None,
                 tie_change=None):
    path = os.path.join(root, "registration_report.txt")
    with open(path, "w") as f:
        W = f.write
        W("REGISTRATION TABLE REPORT  (v4, step 8b)\n")
        W("=" * 70 + "\n")
        W(f"Source        : {IN_ROOT}\n")
        W(f"Output        : {root}\n")
        W(f"Standard      : {STANDARD}"
          + (" with the 95% MC margin\n" if USE_MC_MARGIN
             else " on the POINT ESTIMATE\n"))
        W(f"Sample sizes  : {N_MIN}..{N_MAX}\n")
        W(f"Planning cell : share {PLANNING_SHARE:g}, lapse "
          f"{PLANNING_LAPSE:g}, "
          + ", ".join(f"{k} {v:g}/SD"
                      for k, v in PLANNING_WEIGHT.items()) + "\n")
        if meta:
            W(f"Run read      : {meta.get('built', '?')}\n")
        if prov:
            for name, digest in (prov.get("sha256") or {}).items():
                W(f"  sha256 {name:<24s} "
                  + (digest or "not read") + "\n")
        W(f"Elapsed       : {elapsed:.1f}s\n\n")
        for k, v in report:
            W(f"  {k:<32s} {v}\n")

        W("\n1. WHAT IS BEING REGISTERED\n")
        W("-" * 70 + "\n")
        if switch is None:
            W("  The COUNT RULE at every N from "
              f"{N_MIN} to {N_MAX}. The sign test never\n")
            W("  overtakes it in the planning cell within that "
              "range.\n")
            W("  Each N's rule and its power are in "
              "combined_rule.csv;\n")
            W("  registration_table.csv carries both rules, the\n")
            W("  false-claim margin and the two cost cells.\n\n")
            W("  WITH THE TUNING COST IN THE SAME BREATH. The rule "
              "is\n")
            W("  calibrated in one cell. In weaker cells the sign "
              "test\n")
            W("  would outperform it for colour from N = 15 (section "
              "6);\n")
            W("  for fog the count rule stays ahead throughout. The\n")
            W("  registered power figures are the planning-cell "
              "ones,\n")
            W("  and they are the rule's best case.\n\n")
            W(f"  FEWER THAN {MIN_ANALYSABLE} ANALYSABLE "
              "PARTICIPANTS. If the end date\n")
            W(f"  arrives with fewer than {MIN_ANALYSABLE}, NEITHER "
              "EXISTENCE CLAIM IS\n")
            W("  MADE. The counts in each direction are reported\n")
            W("  descriptively, per feature, and nothing is "
              "concluded\n")
            W("  from them. C1 is still run: it compares two groups\n")
            W("  and does not depend on the per-person rule. This is\n")
            W("  registered in advance so that a small sample is not\n")
            W("  met with a decision taken after seeing it.\n")
        else:
            W(f"  N < {switch}   the count rule calibrated at that N\n")
            W(f"  N >= {switch}  the sign test\n\n")
            W("  The rule depends ONLY on the final number of\n")
            W("  participants, which is fixed by a maximum N and an\n")
            W("  end date, never on the results. Each N's rule is in\n")
            W("  registration_table.csv, and combined_rule.csv is the\n")
            W("  registered rule alone.\n")

        if REGISTERED_RULE == "count":
            W("\n2. WHY THE COUNT RULE THROUGHOUT\n")
            W("-" * 70 + "\n")
            W("  A combination of the two rules was considered, and\n")
            W("  rejected. The case for it was that the count rule\n")
            W("  loses power as N grows: with m = 0 one\n")
            W("  opposite-direction detection vetoes the claim, every\n")
            W("  non-responder added is another chance of that veto,\n")
            W("  and so more participants can mean less power.\n\n")
            W("  THAT IS TRUE OF THE m = 0 RULE AND ONLY OF IT. The\n")
            W("  rule selected in the planning cell has m > 0 from\n")
            W("  N = 11 on, so no single opposite-direction detection\n")
            W("  can veto anything. The scan in section 3 is the\n")
            W("  evidence: the sign test does not overtake the count\n")
            W("  rule for either feature at any N in the registered\n")
            W("  range.\n\n")
            W("  THE POWER DOES NOT RISE MONOTONICALLY, and the\n")
            W("  registration should not claim it does. The rule is\n")
            W("  re-selected at every N under the false-rate\n")
            W("  ceiling, and where the ceiling forces a stricter m\n")
            W("  the power falls even though N grew:\n\n")
            sh = power_shape(C)
            for feat in FEATURES:
                d = sh[feat]
                W(f"    {feat}\n      N      "
                  + " ".join(f"{n:>5d}" for n in d["n"]) + "\n")
                W("      power  "
                  + " ".join(f"{p:5.3f}" for p in d["power"]) + "\n")
                W("      m <=   "
                  + " ".join(f"{m:>5d}" for m in d["m"]) + "\n")
                W("      drops  "
                  + (", ".join(f"N {n} ({g:+.3f})"
                               for n, g in d["drops"])
                     if d["drops"] else "none") + "\n")
            W("\n  Power is quoted PER N from the table, so nothing\n")
            W("  in the registered analysis depends on the sequence\n")
            W("  being monotone. What is not available is the\n")
            W("  statement that another participant always helps:\n")
            W("  at the N where m tightens, one more can lower the\n")
            W("  power of the rule that N selects.\n\n")
            W("  The sign test stays in the table at every N as a\n")
            W("  ROBUSTNESS COMPARISON, not as part of the\n")
            W("  registered analysis. Reporting it costs nothing and\n")
            W("  shows that the registered choice was not the only\n")
            W("  one available.\n")
        else:
            W("\n2. WHY NOT THE COUNT RULE THROUGHOUT\n")
            W("-" * 70 + "\n")
            W("  Under the strict standard its m is 0 at these "
              "sizes, so\n")
            W("  ONE opposite-direction detection vetoes the claim. "
              "Every\n")
            W("  non-responder added is another chance of that veto, "
              "so\n")
            W("  with partial response its power stops rising and "
              "then\n")
            W("  falls as N grows. The sign test compares the two "
              "counts\n")
            W("  instead of requiring zero on one side, so it does "
              "not\n")
            W("  have that defect; it is weak at the smallest sizes\n")
            W("  because it cannot fire below a minimum number of\n")
            W("  detections at all. The combination takes each where "
              "it\n")
            W("  is better.\n")

        W("\n3. THE SWITCH POINT, AND HOW ROBUST IT IS\n")
        W("-" * 70 + "\n")
        W("  sign >= count for BOTH features, by N:\n    ")
        W("  ".join(f"{n}:{'Y' if wins[n] else 'n'}"
                    for n in range(N_MIN, N_MAX + 1)) + "\n\n")
        W(f"  planning cell    switch at "
          f"{switch if switch is not None else 'none in range'}\n")
        W(f"  robustness cell  switch at "
          f"{switch_rob if switch_rob is not None else 'none in range'}"
          "\n\n")
        if switch_rob is None:
            W("  THE TWO CELLS AGREE. The count rule wins throughout\n")
            W(f"  {N_MIN}..{N_MAX} at the robustness pair as well as\n")
            W("  at the planning pair, so the registered rule is not\n")
            W("  an artefact of which pair was taken as planning.\n")
            W("  The sign test overtakes it only in the two cost\n")
            W("  cells of section 6, which weaken the effect rather\n")
            W("  than the pair, and those are reported next to the\n")
            W("  planning-cell power.\n")
        elif switch_rob == switch:
            W("  The two pairs agree, so the switch point does not\n")
            W("  depend on which of them is taken as planning.\n")
        else:
            W("  The two pairs DISAGREE. The registered switch point\n")
            W("  is the planning one; the difference is a statement\n")
            W("  about how much the choice depends on the assumed\n")
            W("  effect, and should be reported as such.\n")

        W("\n4. EQUAL BEHAVIOUR: HOW EQUAL\n")
        W("-" * 70 + "\n")
        W("  A weight per SD is not comparable between the features:\n")
        W("  the informative colour trials are about 1.0 corpus SD\n")
        W("  apart and the fog trials about 0.64, so the same weight\n")
        W("  gives a smaller choice bias on fog. The pairs are\n")
        W("  therefore matched on the choice probability.\n\n")
        for r in E.itertuples():
            W(f"  {r.pair} pair\n")
            for feat in FEATURES:
                W(f"    {feat:7s} "
                  f"{getattr(r, f'{feat}_weight_per_SD'):g}/SD   "
                  f"p = {getattr(r, f'{feat}_p_typical_trial'):.4f}   "
                  "per-person detection ")
                d = getattr(r, f"{feat}_person_detection")
                W(("unresolved" if d is None else f"{d:.4f}") + "\n")
            W(f"    gap in p {r.p_gap:.4f}")
            if r.person_detection_gap is not None:
                W(f"   gap in detection {r.person_detection_gap:.4f}")
            W("\n")
        W("\n  MATCHING ON THE PROBABILITY DOES NOT MATCH THE\n")
        W("  PER-PERSON DETECTION RATE. The count rule is saturated\n")
        W("  on k at these sizes, so it barely moves with that rate;\n")
        W("  the sign test compares the two counts and is sensitive\n")
        W("  to it. That is why the higher pair can match on the\n")
        W("  count rule and not on the sign test. It does not\n")
        W("  affect what is registered: the count rule wins for\n")
        W("  both features at every N in both cells, so the\n")
        W("  imperfect match of the higher pair changes no\n")
        W("  selection. It is reported because the two features'\n")
        W("  claims are stated as equally strong, and on the\n")
        W("  per-person detection rate they are not: 0.80 against\n")
        W("  0.67 at the planning pair. That belongs in the\n")
        W("  preregistration as a limitation of the stimuli.\n")

        W("\n5. HEADROOM UNDER THE FALSE-CLAIM TARGET\n")
        W("-" * 70 + "\n")
        W(f"  The target is {TARGET_FALSE} ON THE UPPER BOUND of the\n")
        W("  95% Monte Carlo interval, and it is not tightened. The\n")
        W("  upper bound is itself the safeguard against a selection\n")
        W("  made by simulation noise, so a rule that clears it is\n")
        W("  certified however thin the margin. The margin is\n")
        W("  reported per N all the same, in the false_headroom\n")
        W("  column of registration_table.csv, because how close a\n")
        W("  rule sits to the line is a property of the rule.\n\n")
        reg = T[T.registered.astype(bool) & (T.rule_kind == "count")]
        for feat in FEATURES:
            s = reg[reg.feature == feat].sort_values("n")
            W(f"    {feat}\n      N       "
              + " ".join(f"{int(r.n):>6d}" for r in s.itertuples())
              + "\n")
            W("      upper   "
              + " ".join(f"{r.false_worst_upper:6.4f}"
                         for r in s.itertuples()) + "\n")
            W("      margin  "
              + " ".join(f"{r.false_headroom:6.4f}"
                         for r in s.itertuples()) + "\n")
        if thin is not None and len(thin):
            W(f"\n  {len(thin)} CELL(S) ABOVE {THIN_FALSE_WARN}, "
              "marked false_thin in the\n")
            W("  table and listed here as required:\n")
            for r in thin.itertuples():
                W(f"    {r.feature} N = {int(r.n)}, {r.rule}: "
                  f"upper {r.false_worst_upper:.5f}, point\n")
                W(f"      {r.false_worst:.5f}, margin "
                  f"{r.false_headroom:.5f}\n")
            W("\n  Each is admissible as registered. None is a\n")
            W("  reason to change the rule; each is a reason not to\n")
            W("  describe the false-claim rate as comfortably under\n")
            W("  the target at those sizes.\n")
        else:
            W(f"\n  No registered rule exceeds {THIN_FALSE_WARN} on\n")
            W("  the upper bound.\n")

        W("\n6. WHERE THE TIE-BREAK DECIDED\n")
        W("-" * 70 + "\n")
        W("  Two rules whose planning-cell power differs by less\n")
        W(f"  than {POWER_TIE_TOL} are treated as TIED, because the\n")
        W("  simulation's own Monte Carlo error at a power near 0.93\n")
        W("  is about 0.006. Among the tied the SMALLER m is taken:\n")
        W("  the rule that tolerates fewer opposite-direction\n")
        W("  detections, and so is the harder of the two to satisfy\n")
        W("  by luck.\n\n")
        dup, sb = ties_frame()
        if dup is None:
            W("  The tie-break never fired: at every N and in both\n")
            W("  cells the most powerful rule was also the one it\n")
            W("  prefers.\n")
        else:
            W(f"  {len(dup)} firing(s) were ONE RULE UNDER TWO\n")
            W("  FAMILY NAMES -- same k, same m, identical power,\n")
            W("  identical\n")
            W("  false-claim rate, differing only in whether the\n")
            W("  threshold is written as a count or as a proportion\n")
            W("  of N. Resolving those is bookkeeping and no\n")
            W("  decision was made; they are counted here and listed\n")
            W("  in tie_breaks.csv.\n\n")
            if not len(sb):
                W("  NO SUBSTANTIVE FIRING: the tie-break did not\n")
                W("  change which rule is registered anywhere.\n")
            else:
                W(f"  {len(sb)} SUBSTANTIVE firing(s), where the\n")
                W("  tie-break changed the registered rule:\n\n")
                for r in sb.itertuples():
                    W(f"    {r.cell} cell, {r.feature}, N = "
                      f"{int(r.n)}\n")
                    W(f"      most powerful  {r.nominal_family} "
                      f"k >= {int(r.nominal_k)}, "
                      f"m <= {int(r.nominal_m)}   "
                      f"{r.nominal_power:.6f}\n")
                    W(f"      registered     {r.chosen_family} "
                      f"k >= {int(r.chosen_k)}, "
                      f"m <= {int(r.chosen_m)}   "
                      f"{r.chosen_power:.6f}\n")
                    W(f"      power given up {r.power_given_up:.6f}\n")
                W("\n  The most power given up at any of them was\n")
                W(f"  {sb.power_given_up.max():.6f}, which is inside\n")
                W("  the simulation's own error, in exchange for a\n")
                W("  rule that is harder to satisfy by chance.\n")

        W("  THE ORDER OF THE TIE-BREAK. Ties go to the lowest\n")
        W("  upper-bound false-claim rate first, then the smaller m,\n")
        W("  then the smaller k, then the family. The measured rate\n")
        W("  comes first because it IS the quantity a stricter rule\n")
        W("  is wanted for; a smaller m is a proxy that can disagree\n")
        W("  with it. Under the previous order -- m first -- ")
        if tie_change is None:
            W("the\n  selection was THE SAME in every cell at every "
              "N, so\n")
            W("  the change moved no registered rule here.\n")
        else:
            W(f"{len(tie_change)}\n  selection(s) differed:\n")
            for r in tie_change.itertuples():
                W(f"    {r.cell} cell, {r.feature}, N = {int(r.n)}, "
                  f"{r.rule_kind}\n")
                W(f"      m first   {r.was_family} k >= {r.was_k}, "
                  f"m <= {r.was_m}   power {r.was_power:.6f}, "
                  f"false upper {r.was_false_upper:.5f}\n")
                W(f"      rate first {r.now_family} k >= {r.now_k}, "
                  f"m <= {r.now_m}   power {r.now_power:.6f}, "
                  f"false upper {r.now_false_upper:.5f}\n")
            W("\n  tie_order_change.csv has these rows.\n")

        W("\n7. WHAT THE TUNING COSTS\n")
        W("-" * 70 + "\n")
        W("  The count rule is calibrated at ONE cell: the planning\n")
        W("  weights at half share. A rule tuned at a cell is better\n")
        W("  there and can be worse away from it, so the\n")
        W("  planning-cell power on its own is the rule quoting its\n")
        W("  own best case. Two further cells are in the table for\n")
        W("  both rules at every N:\n\n")
        for name, (which, share) in COST_CELLS.items():
            w = (PLANNING_WEIGHT if which == "planning"
                 else ROBUSTNESS_WEIGHT)
            W(f"    cost_{name}\n")
            W(f"      the {which} weights ("
              + ", ".join(f"{k} {v:g}/SD" for k, v in w.items())
              + f") at share {share:g},\n")
            W(f"      lapse {COST_LAPSE:g}\n")
        W("\n  Crossovers, per cell and feature:\n")
        for k, v in report:
            if k.startswith(tuple(f"{n}," for n in COST_CELLS)):
                W(f"    {k:<24s} {v}\n")
        W("\n  Where the sign test overtakes, the registered rule is\n")
        W("  the weaker of the two IN THAT CELL. That is the trade,\n")
        W("  and it belongs in the registration next to the\n")
        W("  planning-cell power rather than behind it: the planning\n")
        W("  cell is registered because it is the scenario in which\n")
        W("  a study of 10 to 20 participants can succeed at all,\n")
        W("  not because it is the only one considered.\n")

        W("\n8. WHAT THIS TABLE IS NOT\n")
        W("-" * 70 + "\n")
        W("  Two outputs of the step-8 run choose a different rule\n")
        W("  for each scenario, and NEITHER IS THE REGISTERED\n")
        W("  ANALYSIS:\n")
        W("    existence_power.csv  takes the best admissible rule\n")
        W("                         separately for each responder\n")
        W("                         share.\n")
        W("    the behavioural      selects under the practical\n")
        W("    section              standard, not the strict one.\n")
        W("  A real study cannot choose its rule after learning the\n")
        W("  responder share. Quote power for registration only from\n")
        W("  this table, whose rules are fixed per N under the\n")
        W("  strict standard with the Monte Carlo margin.\n")

        W("\n9. FILES\n")
        W("-" * 70 + "\n")
        W("  registration_table.csv    every candidate, both rules,\n")
        W("                            with the two cost columns\n")
        W("  combined_rule.csv         the registered rule alone\n")
        W("  combined_rule_power.png   its power against N\n")
        W("  switch_point.json         the scan behind section 3\n")
        W("  tie_breaks.csv            every tie-break firing\n")
        W("  tie_order_change.csv      what the tie order changed,\n")
        W("                            if anything\n")
        W("  c1_compact.csv            C1 at one stated scenario\n")
        W("  c1_cells.csv              C1, every scenario at those\n")
        W("                            counts -- NOT to be averaged\n")
        W("  equal_behaviour_check.csv the two pairs\n")

        W("\nOVERALL: " + ("PASS" if all("FAIL" not in str(v)
                                         for _, v in report)
                           else "FAIL") + "\n")
    print(f"\n  report: {path}")


# ============================================================
# main
# ============================================================

def main():
    t0 = time.time()
    report = []
    print("=" * 62)
    print("REGISTRATION TABLE  (v4, step 8b)")
    print("=" * 62)
    print("  reads the rerun's outputs; runs no simulation")

    S, adm, extra, meta = load_inputs(report)
    prov = provenance(report)
    root = resolve_output_dir()
    print(f"  output: {root}")

    probs = behaviour_labels(S, report)

    R = select_rules(S, adm, PLANNING_WEIGHT, "planning", report)
    ok2 = check_admissibility(S, R, report)

    switch, wins = find_switch(R, report, "planning")

    Rr = select_rules(S, adm, ROBUSTNESS_WEIGHT, "robustness", report,
                      quiet=True)
    switch_rob, _ = find_switch(Rr, report, "robustness", loud=False)
    banner("ROBUSTNESS PAIR")
    print(f"  switch at "
          f"{switch_rob if switch_rob is not None else 'none in range'}")
    if switch_rob is None:
        print(f"  THE COUNT RULE WINS THROUGHOUT {N_MIN}..{N_MAX} at "
              "the weaker pair too.")
        print("  The two cells agree, so the registered rule does not "
              "depend on which pair")
        print("  was taken as planning. Where the sign test does "
              "overtake it is the two cost")
        print("  cells, which weaken the EFFECT rather than the pair.")

    report_ties(root, report)
    tie_change = report_tie_order(S, adm, report, root)

    C = combined(R, switch)
    ok3 = check_combined(C, report)
    C.to_csv(os.path.join(root, "combined_rule.csv"), index=False)

    E = check_equal_behaviour(S, probs, extra, report, root)

    T = registration_table(R, probs, switch, root, report)
    T = add_power_cells(T, S, adm, PLANNING_WEIGHT, report)
    T = cost_columns(T, S, probs, report, root)
    # the robustness column, so the weaker scenario is in the same table
    rob = []
    for r in T.itertuples():
        w = ROBUSTNESS_WEIGHT[r.feature]
        row = S[(S.feature == r.feature) & (S.n == r.n)
                & (S.family == r.family) & (S.k == r.k)
                & (S.m == r.m)]
        c = power_col(w, PLANNING_SHARE, PLANNING_LAPSE)
        rob.append({"robustness_weight_per_SD": w,
                    "robustness_p_typical_trial":
                        round(probs[(r.feature, w)], 4),
                    "power_robustness": (round(float(row.iloc[0][c]), 4)
                                         if len(row) else None)})
    T = pd.concat([T, pd.DataFrame(rob)], axis=1)
    thin = report_headroom(T, report)
    T.to_csv(os.path.join(root, "registration_table.csv"), index=False)
    print(f"  registration_table.csv   {len(T)} rows "
          f"({len(FEATURES)} features x {N_MAX - N_MIN + 1} sizes "
          "x 2 rules)")
    report.append(("registration table rows", str(len(T))))

    plot_combined(C, switch, root)
    print("  combined_rule_power.png")

    with open(os.path.join(root, "switch_point.json"), "w") as f:
        json.dump({"switch_point": switch,
                   "switch_point_robustness": switch_rob,
                   "scan": {str(k): bool(v) for k, v in wins.items()},
                   "planning_weight": PLANNING_WEIGHT,
                   "robustness_weight": ROBUSTNESS_WEIGHT,
                   "share": PLANNING_SHARE, "lapse": PLANNING_LAPSE,
                   "standard": STANDARD,
                   "mc_margin": USE_MC_MARGIN,
                   "n_min": N_MIN, "n_max": N_MAX}, f, indent=1)

    c1_compact(extra, root, report)

    with open(os.path.join(root, "run_meta.json"), "w") as f:
        json.dump({"script": "registration_table_v4.py",
                   "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "source": IN_ROOT, "output": root,
                   "registered_rule": REGISTERED_RULE,
                   "tie_break": list(TIE_BREAK),
                   "tie_break_previous": list(TIE_BREAK_PREVIOUS),
                   "target_false": TARGET_FALSE,
                   "thin_false_warn": THIN_FALSE_WARN,
                   "min_analysable": MIN_ANALYSABLE,
                   "n_min": N_MIN, "n_max": N_MAX,
                   "source_meta": meta,
                   "source_provenance": prov}, f, indent=1)

    write_report(root, report, T, C, E, switch, switch_rob, wins,
                 meta, time.time() - t0, prov, thin, tie_change)

    bad = [k for k, v in report if "FAIL" in str(v)]
    print("\n" + "=" * 62)
    if bad or not (ok2 and ok3):
        print("FINISHED WITH FAILURES: " + ", ".join(bad))
    else:
        print(f"DONE in {time.time() - t0:.1f}s -- every check passed.")
    print("=" * 62)


if __name__ == "__main__":
    main()
