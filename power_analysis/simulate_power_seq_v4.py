"""
simulate_power_seq_v4.py  (stimulus pipeline for the v4 axes, step 7)

What N PEOPLE can tell you, when only some of them respond.

Step 6 answers the per-person question: given one participant's 180
trials, how strong must their preference be before it can be told from
noise? That question is settled, and the answer is good -- floors near
0.66 (colour) and 0.56 (fog).

This script answers the other half, and it is the half where this
design is weak. The sample is not chosen: a pool of high-CDS
respondents receives a questionnaire, and whoever reports changes in
world perception does the task. That is expected to be about ten
people, and -- this is the part the usual power analysis ignores --
only SOME of those ten need carry the effect. A questionnaire that
selects for reported experience does not guarantee behaviour.

A sample like that breaks the standard group test, because the standard
group test AVERAGES. If two of ten people have a strong effect and
eight have none, the average is weak, and the test reports nothing --
while the finding sits intact in those two participants. So the
quantity to plan around is not the group mean. It is

    HOW MANY of the N people individually clear their own floor.

That number survives a mixture, because non-responders simply are not
counted. This script simulates it, alongside the group test, so the
preregistration can say which is primary and show why.

Where this sits
---------------
  step 1  calibrate_levels_v4   the five colour and five fog rungs
  step 2  select_scenes_v4      shortlist -> you screen by hand
  step 3  build_stimulus_set_v4 the stimuli and the manifest
  step 4  analyse_pairs_v4      pair inventory and thresholds
  step 5  build_trials_v4       the trial list
  step 6  simulate_power_v4     one person: floors, crossover,
                                additivity, trial count
  step 7  this one              N people: responders, group test,
                                sequential stopping

WHAT IS SIMULATED, AND WHAT IS ASSUMED
--------------------------------------
Measured, not assumed:
  - the trial list, read from step 5, with its real magnitudes
  - therefore the per-person noise, and therefore the floors, which
    are recomputed here from that same trial list rather than typed in

Assumed, and swept rather than guessed:
  - EFFECT_WEIGHTS, how strong the effect is in a person who has it.
    No pilot exists, so this cannot be measured. v2 declared 1.0 as
    the smallest effect of interest, justified by what it means in
    behaviour: a weight of 1.0 is a 73% preference at the largest
    difference the stimuli can show, 62% at a typical one. 0.5 and 1.5
    bracket it. The same three values are kept here, so the two
    experiments remain comparable.
  - SPREAD_FACTORS, how much responders differ from each other.
  - RESPONDER_SHARES, what fraction of the sample responds at all.
    This is new in v4; v2 assumed everyone carried the effect. Share
    1.0 reproduces v2's assumption exactly, so the v2 rows are still
    in the output.

Because all three are swept, no single guess is load-bearing, and the
preregistration can show how each conclusion depends on them.

THE FLOORS ARE PER FEATURE, AND STAY THAT WAY
---------------------------------------------
The colour weight is estimated less precisely than the fog weight --
0.66 against 0.56 at the time of writing -- because the colour
differences, measured against each scene's own range, are smaller
relative to that range. Averaging the two floors into one would apply
too lax a test to colour and too strict a test to fog, and the
difference between the number of colour-trackers and fog-trackers is
one of the findings. So both are carried separately throughout.

Inputs
------
  TRIAL_LISTS    trial_list.csv per set, from step 5

Outputs (per set, under OUTPUT_ROOT/<label>/)
---------------------------------------------
  lookup_by_n.csv       THE TABLE. One row per (responder share, N):
                        what you could claim with that many people.
                        Read p_claim_fog, not p_any_fog: the first is
                        the chance of enough detections to beat
                        multiplicity, the second the chance of one,
                        which a study with no effect produces 22% of
                        the time at N = 10.
  responders.csv        the full grid: every scenario x N
  sequential.csv        when a sequentially-monitored study would stop
  false_null.csv        how often the group test says 'no effect' when
                        a minority genuinely responds
  lookup_by_n.png       the headline table as curves
  responder_recovery.png  recovered responder count vs the truth
  false_null.png        the group test's failure mode, drawn
  sequential.png        the distribution of stopping points

Shared, under OUTPUT_ROOT/
--------------------------
  comparison.csv        the sets side by side
  power_seq_report.txt  the whole thing

Usage
-----
    conda activate drl2drv
    python simulate_power_seq_v4.py

CPU only. The full grid is a few minutes; QUICK_GRID = True cuts it to
the headline scenario in seconds.
"""

import os
import time
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

# --- Input / output ---
#
# TRIAL_LISTS
#   (label, trial_list.csv) per stimulus set. The label names the
#   output subdirectory and the column in the comparison table.
TRIAL_LISTS = [
    ("haze_eq", "data/trials_v4_ec/haze_eq/trial_list.csv"),
    ("haze",    "data/trials_v4_ec/haze/trial_list.csv"),
]

OUTPUT_ROOT = "data/power_seq_v4"

# ON_EXISTING
#   "new" writes to OUTPUT_ROOT_2, _3, ... rather than overwriting an
#   existing analysis. "error" stops instead.
ON_EXISTING = "new"

# --- Predictors (must match step 6) ---
PRED_SAT = "z_sat"
PRED_FOG = "z_fog"

# FOG_SIGN is already applied by step 4, so d_fog is positive when A is
# foggier. Only the saturation sign needs flipping here; see load_trials.

# --- The population ---
#
# PATTERNS
#   Which features carry the effect, multiplied by the effect weight:
#   (colour, fog). Verbatim from v2 so the two are comparable.
PATTERNS = {
    "colour only": (1.0, 0.0),
    "fog only":    (0.0, 1.0),
    "both":        (1.0, 1.0),
    "neither":     (0.0, 0.0),
}

# EFFECT_WEIGHTS
#   How strong the effect is in a person who HAS it. A weight is on the
#   z scale: going from no difference to the scene's full range shifts
#   the log-odds of choosing the manipulated image by this much.
#       0.5 -> 56% at half the range, 62% at the full range
#       1.0 -> 62% at half the range, 73% at the full range
#       1.5 -> 68% at half the range, 82% at the full range
#   v2 declared 1.0 the smallest effect of interest on exactly this
#   reasoning, and v4's design happens to be calibrated to it: step 6's
#   weight for 80% detection came out at 0.97 (colour) and 0.82 (fog).
#   A person at 0.5 sits BELOW both floors and is individually
#   undetectable, which is what "below the smallest effect of interest"
#   should mean.
EFFECT_WEIGHTS = [0.5, 1.0, 1.5]

# SPREAD_FACTORS
#   SD of the true weights among RESPONDERS, as a multiple of the
#   effect weight.
#       0.5  similar responders -- nearly all show the preference
#       1.0  some strongly, some barely
#       2.0  preferences in both directions; the mean is a minority
#   Applies to both features, including one whose mean is zero: in a
#   "fog only" population people still vary in colour, around zero.
#   0.0 is not a population anyone believes in -- it makes every
#   responder identical. It is there as the TRUE NULL: combined with
#   the "neither" pattern it gives people whose weights are exactly
#   zero, which is the only scenario that can verify the per-person
#   false-alarm rate and the study-level rule actually deliver what
#   they promise. Without it the rule is asserted from a formula and
#   never checked. Keep it.
SPREAD_FACTORS = [0.0, 0.5, 1.0, 2.0]

# WEIGHT_CORRELATIONS
#   Correlation between a responder's true colour and fog weights.
#   0 = unrelated; 0.5 = people who track one tend to track the other.
WEIGHT_CORRELATIONS = [0.0, 0.5]

# RESPONDER_SHARES
#   Fraction of the sample that carries the effect at all. NEW in v4.
#   The rest have true weights of exactly (0, 0) -- they completed the
#   questionnaire and the task, and the cues do not move their
#   judgements.
#
#   1.0 is v2's assumption and is kept first so the v2 rows are
#   present. 0.1 is the pessimistic case the design has to survive.
#
#   Each person is a responder independently with this probability, so
#   the actual number in any one simulated study varies -- which is
#   itself part of the answer.
RESPONDER_SHARES = [1.0, 0.5, 0.2, 0.1]

# --- Sample sizes ---
#
# N_LOOKS
#   Participant counts to evaluate. Unlike v2, which fixed interim
#   looks at 5 / 10 / 15, this spans the whole plausible range,
#   because N is set by the questionnaire rather than chosen. The
#   output is a lookup table: when the questionnaire returns 7 people,
#   read row 7.
N_LOOKS = list(range(4, 21))

# N_RUNS
#   Simulated studies per scenario. v2 used 1000; Kusztor et al. used
#   100. 400 gives a standard error of at most 2.5 percentage points,
#   which is finer than any decision here turns on.
N_RUNS = 400

# --- The headline scenario ---
#
# The one shown in the figures and the report's main table. Every other
# combination is in responders.csv.
HEADLINE_PATTERN = "fog only"

# LOOKUP_PATTERNS
#   The lookup table is written for each of these, one block apiece.
#   One pattern is not enough: under "fog only" the true colour effect
#   is ZERO, so p_claim_colour there measures the spread rather than
#   colour's power, and reading it as colour's power is a mistake the
#   earlier single-pattern table invited. "both" is the most realistic
#   -- someone whose sense of unreality responds to desaturation
#   probably responds to contrast loss too -- and is the block to plan
#   with.
LOOKUP_PATTERNS = ["both", "fog only", "colour only"]
HEADLINE_WEIGHT = 1.0
HEADLINE_SPREAD = 1.0
HEADLINE_CORR = 0.0

# QUICK_GRID
#   True runs ONLY the headline pattern/weight/spread/corr, across all
#   responder shares and all N. Seconds instead of minutes. Use it
#   while changing the script; set False for the real run.
QUICK_GRID = False

# --- Per-person classification ---
#
# RIDGE
#   Penalty in the per-participant fit. MUST match step 6, whose floors
#   this script uses, and the eventual check_session_v4.py, since the
#   simulated analysis has to be the real analysis. Step 6 uses 1e-6.
#
#   Note this differs from v2's 0.25. A ridge shrinks the weights
#   toward zero, so a floor computed at one ridge does not apply at
#   another; the two must move together or not at all.
RIDGE = 1e-6

# FLOOR_ALPHA
#   One-sided false-alarm rate for calling one person a responder. The
#   floor is the (1 - alpha) percentile of the weight a true guesser
#   would produce. 0.025 matches step 6's 1.96-sigma convention.
FLOOR_ALPHA = 0.025

# N_FLOOR_PEOPLE
#   Simulated guessers used to locate the floors. A floor is a TAIL
#   percentile, so it needs far more people than a mean or an SD would:
#   at 4,000 only about 100 sit above it, and the estimate wobbles by
#   enough to matter -- the two fog families came out 0.74 and 0.70 on
#   colour, a gap that runs opposite to their SDs and is therefore
#   mostly noise. At 20,000 there are 500 above it.
N_FLOOR_PEOPLE = 20_000

# STUDY_ALPHA
#   The false-positive rate for the STUDY, not for one participant.
#
#   This is the correction the per-person floors do not make. Each
#   participant is tested at FLOOR_ALPHA, which is fine for one person
#   -- but every extra participant is another chance to be wrong. With
#   ten people at 2.5% each, the chance that AT LEAST ONE pure guesser
#   is called a responder is 1 - 0.975^10 = 22%. So "we found one
#   responder" is something a study with no effect at all produces
#   nearly a quarter of the time, and it cannot support an existence
#   claim on its own.
#
#   Requiring TWO independent detections fixes it: the chance of two or
#   more false alarms among ten people is 2.5%. The script works out,
#   for each sample size, the smallest number of detections that keeps
#   the study-level rate at or below STUDY_ALPHA, and reports how often
#   that number is reached. THAT is the existence claim, and it is the
#   number to preregister -- it is roughly half what the "at least one"
#   column suggests.
STUDY_ALPHA = 0.05

# FLOOR_OVERRIDE
#   (colour, fog) to use instead of the recomputed floors. Leave None.
#   Set it only to reproduce an older run; a floor typed in by hand
#   will silently stop matching the trial list.
FLOOR_OVERRIDE = None

# --- Group test ---
#
# ALPHA / ONE_TAILED for the t-test; the hypothesis is directional, so
# one-tailed, matching step 6.
ALPHA = 0.05
ONE_TAILED = True

# BF_SCALE
#   Cauchy prior scale on the standardised effect size. 0.707 is the
#   JZS default and what Kusztor et al. used.
BF_SCALE = 0.707

# BF_THRESHOLD
#   A Bayes factor this far from 1 counts as evidence. 3 is the
#   conventional "moderate". BF10 > 3 is evidence for an effect;
#   BF10 < 1/3 is evidence for the null; between them is INCONCLUSIVE,
#   and being able to say that is the main reason to use Bayes factors
#   here at all -- it is not a power advantage, it is the ability to
#   distinguish 'no effect' from 'not enough data'.
BF_THRESHOLD = 3.0

# BF_T_RANGE / BF_T_STEP
#   The Bayes factor depends only on t and N, so it is tabulated once
#   on a grid of t and every simulated study is a lookup. Beyond
#   +-BF_T_RANGE the factor is clipped, which never matters: a t that
#   large is already decisive.
#   The step is 0.1 rather than something finer because log BF is
#   very smooth in t: interpolating at this step reproduces the
#   directly integrated value to within 6e-4 relative, which is two
#   orders of magnitude below this script's own Monte Carlo error.
#   Finer steps cost minutes and buy nothing.
BF_T_RANGE = 12.0
BF_T_STEP = 0.1

# BF_CACHE
#   Where to keep the tabulated factors between runs. Building them is
#   the slowest part of the script by far -- every other stage is
#   sub-second -- and they depend only on N_LOOKS, TESTS and BF_SCALE,
#   never on the trial list. So they are computed once, written here,
#   and reused until one of those three changes. Set to None to
#   recompute every time.
BF_CACHE = "data/power_seq_v4_bf_cache.npz"

# TESTS
#   Which tabulations to build. "one" for the directional per-feature
#   group test, "two" for the crossover, which has no predicted
#   direction at group level.
TESTS = ["one", "two"]

# --- Sequential monitoring ---
#
# SEQ_MIN_N
#   Smallest sample at which the Bayes factor is read. Below about 4 it
#   is almost never conclusive either way and reading it only adds
#   opportunities to stop early by luck.
SEQ_MIN_N = 4

RANDOM_SEED = 42

TYPES = ["sat_only", "fog_only", "together", "conflict"]
FEATURES = ["colour", "fog"]

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ============================================================
# Output directory
# ============================================================

def resolve_output_root():
    """OUTPUT_ROOT, never overwriting an existing analysis."""
    def occupied(d):
        return os.path.isdir(d) and os.path.isfile(
            os.path.join(d, "power_seq_report.txt"))

    if not occupied(OUTPUT_ROOT):
        return OUTPUT_ROOT
    if ON_EXISTING == "error":
        raise FileExistsError(
            f"{OUTPUT_ROOT} already holds an analysis. Move it, or set "
            "ON_EXISTING = 'new'.")
    n = 2
    while occupied(f"{OUTPUT_ROOT}_{n}"):
        n += 1
    print(f"  NOTE: {OUTPUT_ROOT} already holds an analysis; writing "
          f"to {OUTPUT_ROOT}_{n} instead.")
    return f"{OUTPUT_ROOT}_{n}"


# ============================================================
# Model  (verbatim from step 6, so the two cannot drift apart)
# ============================================================

def fit_logit(X, y, n_iter=60, ridge=RIDGE):
    """Newton-Raphson logistic regression, no intercept.

    No intercept because the design is symmetric: each pair appears
    with A on the left or right at random, and the predictors are
    signed differences, so there is no baseline preference to absorb.
    """
    b = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-X @ b))
        W = np.clip(p * (1 - p), 1e-9, None)
        H = (X * W[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


def fit_logit_batch(X, Y, n_iter=60, ridge=RIDGE):
    """fit_logit for many simulated people at once.

    X is (n_trials, k), Y is (n_people, n_trials). Returns
    (n_people, k). Verified against fit_logit at start-up.
    """
    P, k = Y.shape[0], X.shape[1]
    b = np.zeros((P, k))
    eye = ridge * np.eye(k)
    for _ in range(n_iter):
        # Clipped because a simulated participant who responds almost
        # deterministically drives the weights up until exp overflows.
        # The probability is 0 or 1 either side of the clip, so this
        # changes no result -- it only silences the warning.
        p = 1.0 / (1.0 + np.exp(-np.clip(b @ X.T, -500.0, 500.0)))
        w = np.clip(p * (1 - p), 1e-9, None)
        H = np.einsum("pt,ti,tj->pij", w, X, X) + eye
        g = (Y - p) @ X
        step = np.linalg.solve(H, g[..., None])[..., 0]
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


def simulate_many(X, W, rng):
    """Choices of many people at once. W is (n_people, 2) true weights."""
    p = 1.0 / (1.0 + np.exp(-(W @ X.T)))
    return (rng.random(p.shape) < p).astype(float)


def draw_population(n, mean, sd, corr, rng):
    """True (b_sat, b_fog) for n people from a bivariate normal.

    mean is (colour, fog); both features share the same SD; corr is the
    correlation between a person's two weights.
    """
    if sd == 0:
        return np.tile(np.asarray(mean, float), (n, 1))
    cov = np.array([[sd ** 2, corr * sd ** 2],
                    [corr * sd ** 2, sd ** 2]])
    return rng.multivariate_normal(np.asarray(mean, float), cov, size=n)


def draw_mixture(n_studies, n_people, pattern, weight, spread, corr,
                 share, rng):
    """A mixture population: `share` of people respond, the rest do not.

    Returns (n_studies, n_people, 2) true weights and the boolean
    responder mask of the same leading shape. Each person is a
    responder independently, so the NUMBER of responders in a study
    varies -- which is part of what this script is measuring.
    """
    total = n_studies * n_people
    mean = (pattern[0] * weight, pattern[1] * weight)
    W = np.zeros((total, 2))
    is_resp = rng.random(total) < share
    k = int(is_resp.sum())
    if k:
        W[is_resp] = draw_population(k, mean, spread * weight, corr, rng)
    return (W.reshape(n_studies, n_people, 2),
            is_resp.reshape(n_studies, n_people))


# ============================================================
# Bayes factor  (tabulated once, then looked up)
# ============================================================

def jzs_bf10_gform(t, n, scale=BF_SCALE):
    """Reference implementation: JZS BF10 in the g form, two-sided.

    Kept only to check the fast form below at start-up.
    """
    df = n - 1

    def integrand(g):
        return ((1 + n * g) ** -0.5
                * (1 + t ** 2 / ((1 + n * g) * df)) ** (-(df + 1) / 2.0)
                * scale / np.sqrt(2 * np.pi)
                * g ** -1.5 * np.exp(-scale ** 2 / (2 * g)))

    null = (1 + t ** 2 / df) ** (-(df + 1) / 2.0)
    val, _ = quad(integrand, 0, np.inf, limit=200)
    return val / null


def jzs_bf_delta(t, n, sided, scale=BF_SCALE, n_grid=2001):
    """JZS Bayes factor in the delta form, one- or two-sided.

    The likelihood of the observed t for each possible effect size
    delta (a noncentral t density), averaged over a Cauchy prior on
    delta, divided by the likelihood at delta = 0. For the one-sided
    test the prior is truncated to delta > 0 and doubled, so it still
    integrates to 1.

    Integrated on a fixed grid around where the likelihood is
    non-negligible: fast, vectorisable, and checked against the g form
    at start-up.
    """
    df = n - 1
    dhat = t / np.sqrt(n)
    width = np.sqrt(1.0 / n + dhat ** 2 / (2.0 * df))
    lo, hi = dhat - 20 * width, dhat + 20 * width
    if sided == "one":
        lo = max(lo, 0.0)
        if hi <= lo:
            hi = lo + 20 * width
    delta = np.linspace(lo, hi, n_grid)
    prior = stats.cauchy.pdf(delta, 0, scale) * (2.0 if sided == "one"
                                                 else 1.0)
    like = np.nan_to_num(stats.nct.pdf(t, df, delta * np.sqrt(n)),
                         nan=0.0)
    marg = _trapz(like * prior, delta)
    denom = stats.t.pdf(t, df)
    return marg / denom if denom > 0 else np.nan


def _cache_key():
    """What the tabulated factors depend on, and nothing else."""
    return (f"n{'_'.join(map(str, N_LOOKS))}"
            f"|t{BF_T_RANGE}:{BF_T_STEP}"
            f"|s{BF_SCALE}|{'_'.join(TESTS)}")


def build_bf_tables():
    """log Bayes factor on a grid of t, for every N and every test.

    The factor depends only on t and N, so every simulated study is a
    lookup rather than an integration. Also checks the fast form
    against the reference and prints the worst disagreement.

    This is the slowest stage in the script -- the noncentral t density
    is expensive and it is evaluated once per grid point per N per test
    -- so the result is cached on disk under BF_CACHE and reused while
    N_LOOKS, TESTS, BF_SCALE and the t grid are unchanged.
    """
    grid = np.round(np.arange(-BF_T_RANGE, BF_T_RANGE + 1e-9,
                              BF_T_STEP), 10)
    key = _cache_key()
    if BF_CACHE and os.path.isfile(BF_CACHE):
        try:
            z = np.load(BF_CACHE, allow_pickle=False)
            if str(z["key"]) == key:
                tables = {(n, sided): z[f"t_{n}_{sided}"]
                          for n in N_LOOKS for sided in TESTS}
                print(f"  reusing tabulated factors from {BF_CACHE}")
                return grid, tables, float(z["worst"])
            print("  cached factors are for different settings; "
                  "rebuilding")
        except Exception as e:
            print(f"  could not read {BF_CACHE} ({e}); rebuilding")

    tables = {}
    todo = len(N_LOOKS) * len(TESTS)
    done = 0
    t_start = time.time()
    for n in N_LOOKS:
        for sided in TESTS:
            bf = np.array([jzs_bf_delta(t, n, sided) for t in grid])
            tables[(n, sided)] = np.log(np.clip(bf, 1e-300, None))
            done += 1
            if done % 6 == 0 or done == todo:
                el = time.time() - t_start
                print(f"    {done}/{todo} tables, {el:.0f}s elapsed, "
                      f"~{el / done * (todo - done):.0f}s left")

    worst = 0.0
    for n in (N_LOOKS[0], N_LOOKS[len(N_LOOKS) // 2], N_LOOKS[-1]):
        for t in [-3.0, -1.0, 0.0, 0.5, 1.5, 2.5, 4.0, 8.0]:
            ref = jzs_bf10_gform(t, n)
            new = jzs_bf_delta(t, n, "two")
            worst = max(worst, abs(new / ref - 1))
    print(f"  Bayes factor, fast form vs reference: largest relative "
          f"difference {worst:.2e}")
    if worst > 1e-3:
        warnings.warn("Bayes factor integration disagrees with the "
                      "reference by more than 0.1%")

    if BF_CACHE:
        try:
            os.makedirs(os.path.dirname(BF_CACHE) or ".",
                        exist_ok=True)
            np.savez_compressed(
                BF_CACHE, key=key, worst=worst, grid=grid,
                **{f"t_{n}_{sided}": tables[(n, sided)]
                   for n in N_LOOKS for sided in TESTS})
            print(f"  tabulated factors cached to {BF_CACHE}")
        except Exception as e:
            print(f"  could not write {BF_CACHE} ({e}); continuing")
    return grid, tables, worst


def lookup_bf(t, n, sided, grid, tables):
    """Bayes factors for an array of t values at sample size n."""
    t = np.nan_to_num(np.asarray(t, dtype=float), nan=0.0,
                      posinf=BF_T_RANGE, neginf=-BF_T_RANGE)
    t = np.clip(t, -BF_T_RANGE, BF_T_RANGE)
    return np.exp(np.interp(t, grid, tables[(n, sided)]))


def t_stats(values):
    """One-sample t for each row of a (n_studies, n_people) array.

    A row whose people all give the same weight has zero variance and
    an infinite t; that is clipped by lookup_bf, and it is the right
    answer anyway -- perfect agreement IS decisive.
    """
    n = values.shape[1]
    m = values.mean(axis=1)
    s = values.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, m / (s / np.sqrt(n)),
                        np.where(m > 0, BF_T_RANGE,
                                 np.where(m < 0, -BF_T_RANGE, 0.0)))


# ============================================================
# Loading  (verbatim from step 6)
# ============================================================

def load_trials(path, label):
    """The analysed trials and the two oriented predictors."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Trial list not found: {path}\nRun step 5 "
            "(build_trials_v4.py) first.")
    trials = pd.read_csv(path)
    for c in ("d_sat", "d_fog", PRED_SAT, PRED_FOG):
        if c not in trials.columns:
            raise ValueError(f"{path} has no {c!r} column.")
    if "analysed" in trials.columns:
        trials = trials[trials.analysed.astype(bool)]
    trials = trials.reset_index(drop=True)
    if not len(trials):
        raise ValueError(f"{path} has no analysed trials.")

    # Signed standardised predictors, oriented so a POSITIVE weight
    # always means "prefers the more manipulated image": the foggier
    # one, and the LESS saturated one -- hence the sign flip on d_sat.
    X = np.column_stack([
        -np.sign(trials.d_sat) * trials[PRED_SAT].abs(),
        np.sign(trials.d_fog) * trials[PRED_FOG].abs(),
    ]).astype(float)
    r = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    print(f"  [{label}] {len(trials)} analysed trials, "
          f"r(predictors) = {r:+.3f}, VIF = {1 / (1 - r ** 2):.2f}")
    return trials, X, r


# ============================================================
# The floors
# ============================================================

def find_floors(X, label):
    """Per-feature floors, recomputed from THIS trial list.

    Simulate people who track nothing, fit them, and take the
    (1 - FLOOR_ALPHA) percentile of each weight. A person above it is
    called a responder on that feature, and a true guesser is called
    one FLOOR_ALPHA of the time.

    Recomputed rather than typed in, so the floors cannot drift out of
    step with the trial list or the ridge.
    """
    if FLOOR_OVERRIDE is not None:
        f = np.asarray(FLOOR_OVERRIDE, dtype=float)
        print(f"  [{label}] floors OVERRIDDEN: colour {f[0]:.3f}, "
              f"fog {f[1]:.3f}")
        return f
    rng = np.random.default_rng(RANDOM_SEED + 101)
    W = np.zeros((N_FLOOR_PEOPLE, 2))
    est = fit_logit_batch(X, simulate_many(X, W, rng))
    floors = np.percentile(est, 100 * (1 - FLOOR_ALPHA), axis=0)
    print(f"  [{label}] floors at {FLOOR_ALPHA:.1%} false alarms: "
          f"colour {floors[0]:.3f}, fog {floors[1]:.3f}")
    print(f"           (null scatter: colour {est[:, 0].std():.3f}, "
          f"fog {est[:, 1].std():.3f})")
    return floors


def required_detections(n, person_alpha=None, study_alpha=None):
    """Detections needed before the finding beats multiplicity.

    The smallest k for which the chance of k or more FALSE detections
    among n participants stays at or below study_alpha, when each
    participant is tested at person_alpha.

    The count of false alarms is binomial, so this is a lookup rather
    than an approximation: k = 1 is almost never enough, k = 2 carries
    samples up to about 14, and k = 3 is needed from there to 20.
    """
    pa = FLOOR_ALPHA if person_alpha is None else person_alpha
    sa = STUDY_ALPHA if study_alpha is None else study_alpha
    for k in range(1, n + 1):
        if stats.binom.sf(k - 1, n, pa) <= sa:
            return k
    return n + 1          # even unanimity would not clear the bar


# ============================================================
# The core: one scenario, all sample sizes
# ============================================================

def run_scenario(X, floors, pattern_name, pattern, weight, spread,
                 corr, share, grid, tables, rng):
    """Simulate N_RUNS studies at the largest N, then read every N.

    Drawing once at N_LOOKS[-1] and taking the first N people of each
    study is what makes the sequential analysis coherent: the study
    that stopped at 8 is the SAME study seen earlier, not an
    independent one.
    """
    n_max = N_LOOKS[-1]
    W, is_resp = draw_mixture(N_RUNS, n_max, pattern, weight, spread,
                              corr, share, rng)
    flat = W.reshape(-1, 2)
    est = fit_logit_batch(X, simulate_many(X, flat, rng))
    est = est.reshape(N_RUNS, n_max, 2)

    # Per-person detection, per feature, against that feature's floor.
    hit = np.stack([est[:, :, 0] > floors[0],
                    est[:, :, 1] > floors[1]], axis=-1)

    rows = []
    for n in N_LOOKS:
        e, h, resp = est[:, :n, :], hit[:, :n, :], is_resp[:, :n]
        k_req = required_detections(n)
        row = {"pattern": pattern_name, "weight": weight,
               "spread": spread, "w_corr": corr, "share": share, "n": n,
               "true_responders_mean": float(resp.sum(axis=1).mean()),
               "k_required": k_req,
               "study_false_pos": float(
                   stats.binom.sf(k_req - 1, n, FLOOR_ALPHA))}
        for k, feat in enumerate(FEATURES):
            cnt = h[:, :, k].sum(axis=1)
            row[f"count_{feat}_mean"] = float(cnt.mean())
            row[f"count_{feat}_p10"] = float(np.percentile(cnt, 10))
            row[f"count_{feat}_p90"] = float(np.percentile(cnt, 90))
            row[f"p_any_{feat}"] = float((cnt >= 1).mean())
            row[f"p_two_{feat}"] = float((cnt >= 2).mean())
            # THE existence claim: enough detections to beat
            # multiplicity at this sample size.
            row[f"p_claim_{feat}"] = float((cnt >= k_req).mean())
            row[f"rate_{feat}"] = float(cnt.mean() / n)

            t = t_stats(e[:, :, k])
            bf = lookup_bf(t, n, "one", grid, tables)
            row[f"bf_median_{feat}"] = float(np.median(bf))
            row[f"p_bf_effect_{feat}"] = float(
                (bf > BF_THRESHOLD).mean())
            row[f"p_bf_null_{feat}"] = float(
                (bf < 1.0 / BF_THRESHOLD).mean())
            row[f"p_bf_incon_{feat}"] = float(
                ((bf >= 1.0 / BF_THRESHOLD)
                 & (bf <= BF_THRESHOLD)).mean())
            crit = (stats.t.ppf(1 - ALPHA, n - 1) if ONE_TAILED
                    else stats.t.ppf(1 - ALPHA / 2, n - 1))
            row[f"p_t_sig_{feat}"] = float(
                (t > crit).mean() if ONE_TAILED
                else (np.abs(t) > crit).mean())

        # JOINT outcomes across the two features. Recorded here
        # because they cannot be reconstructed afterwards from the two
        # separate claim rates -- P(both) is not P(fog) x P(colour),
        # since the same participants drive both.
        c_col = h[:, :, 0].sum(axis=1) >= k_req
        c_fog = h[:, :, 1].sum(axis=1) >= k_req
        row["p_joint_both"] = float((c_col & c_fog).mean())
        row["p_joint_fog_only"] = float((c_fog & ~c_col).mean())
        row["p_joint_colour_only"] = float((c_col & ~c_fog).mean())
        row["p_joint_neither"] = float((~c_col & ~c_fog).mean())

        # The crossover, at group level: is the group's fog weight
        # bigger than its colour weight? Paired, and two-sided,
        # because at group level the direction is not predicted -- the
        # design is meant to find colour-trackers too.
        d = e[:, :, 1] - e[:, :, 0]
        td = t_stats(d)
        bfd = lookup_bf(td, n, "two", grid, tables)
        row["p_bf_crossover"] = float((bfd > BF_THRESHOLD).mean())
        row["crossover_mean"] = float(d.mean())
        rows.append(row)
    return pd.DataFrame(rows), est, hit, is_resp


def sequential_stops(X, floors, pattern, weight, spread, corr, share,
                     grid, tables, rng):
    """When a sequentially-monitored study would stop, and on what.

    Walks N up from SEQ_MIN_N, reading the group Bayes factor on the
    fog weight at every step, and records the first N at which it
    crosses BF_THRESHOLD (stop: effect) or its reciprocal (stop:
    null). A study that never crosses either runs out of participants
    and is recorded as inconclusive.

    Also records the first N at which at least one, and at least two,
    individual participants clear their floor -- the per-person
    equivalent, and the comparison the preregistration rests on.
    """
    n_max = N_LOOKS[-1]
    W, is_resp = draw_mixture(N_RUNS, n_max, pattern, weight, spread,
                              corr, share, rng)
    est = fit_logit_batch(
        X, simulate_many(X, W.reshape(-1, 2), rng)).reshape(
            N_RUNS, n_max, 2)
    hit_fog = est[:, :, 1] > floors[1]

    stop_n = np.full(N_RUNS, np.nan)
    stop_kind = np.array(["inconclusive"] * N_RUNS, dtype=object)
    for n in [k for k in N_LOOKS if k >= SEQ_MIN_N]:
        live = np.isnan(stop_n)
        if not live.any():
            break
        t = t_stats(est[live, :n, 1])
        bf = lookup_bf(t, n, "one", grid, tables)
        eff = bf > BF_THRESHOLD
        nul = bf < 1.0 / BF_THRESHOLD
        idx = np.where(live)[0]
        stop_n[idx[eff]] = n
        stop_kind[idx[eff]] = "effect"
        stop_n[idx[nul]] = n
        stop_kind[idx[nul]] = "null"

    cum = np.cumsum(hit_fog, axis=1)
    def first_at(k):
        reached = cum >= k
        n = np.where(reached.any(axis=1),
                     reached.argmax(axis=1) + 1, np.nan)
        return n

    return pd.DataFrame({
        "stop_n": stop_n, "stop_kind": stop_kind,
        "first_n_one_responder": first_at(1),
        "first_n_two_responders": first_at(2),
        "true_responders": is_resp.sum(axis=1),
    })


# ============================================================
# Per set
# ============================================================

def scenario_grid():
    """Every (pattern, weight, spread, corr, share) to simulate.

    Two reductions, both because the cells would otherwise be wasted:
      - the "neither" pattern has a mean of zero, so the responder
        share changes nothing; only share 1.0 is run for it.
      - QUICK_GRID keeps only the headline pattern/weight/spread/corr.
    """
    if QUICK_GRID:
        combos = [(HEADLINE_PATTERN, PATTERNS[HEADLINE_PATTERN],
                   HEADLINE_WEIGHT, HEADLINE_SPREAD, HEADLINE_CORR, s)
                  for s in RESPONDER_SHARES]
        combos += [("neither", PATTERNS["neither"], HEADLINE_WEIGHT,
                    HEADLINE_SPREAD, HEADLINE_CORR, 1.0)]
        return combos
    out = []
    for pname, pat in PATTERNS.items():
        shares = [1.0] if pname == "neither" else RESPONDER_SHARES
        for w, sp, co, sh in itertools.product(
                EFFECT_WEIGHTS, SPREAD_FACTORS, WEIGHT_CORRELATIONS,
                shares):
            out.append((pname, pat, w, sp, co, sh))
    return out


def analyse_one(label, path, grid, tables, root):
    print("\n" + "#" * 60)
    print(f"# {label}")
    print("#" * 60)
    out_dir = os.path.join(root, label)
    os.makedirs(out_dir, exist_ok=True)

    trials, X, r = load_trials(path, label)
    floors = find_floors(X, label)
    rng = np.random.default_rng(RANDOM_SEED)

    combos = scenario_grid()
    print(f"\n  {len(combos)} scenarios x {N_RUNS} studies x "
          f"{N_LOOKS[-1]} people")
    chunks = []
    t0 = time.time()
    for i, (pname, pat, w, sp, co, sh) in enumerate(combos, 1):
        df, _, _, _ = run_scenario(X, floors, pname, pat, w, sp, co,
                                   sh, grid, tables, rng)
        chunks.append(df)
        if i % 10 == 0 or i == len(combos):
            el = time.time() - t0
            print(f"    {i}/{len(combos)}  {el:.0f}s elapsed, "
                  f"~{el / i * (len(combos) - i):.0f}s left")
    resp = pd.concat(chunks, ignore_index=True)
    resp.to_csv(os.path.join(out_dir, "responders.csv"), index=False)

    # The headline table: the lookup by N, one block per share.
    pats = [q for q in LOOKUP_PATTERNS if q in set(resp.pattern)]
    look = resp[resp.pattern.isin(pats)
                & (resp.weight == HEADLINE_WEIGHT)
                & (resp.spread == HEADLINE_SPREAD)
                & (resp["w_corr"] == HEADLINE_CORR)].copy()
    cols = ["pattern", "share", "n", "true_responders_mean",
            "count_fog_mean", "count_fog_p10", "count_fog_p90",
            "k_required", "p_claim_fog", "p_claim_colour",
            "p_any_fog", "study_false_pos",
            "p_joint_both", "p_joint_fog_only", "p_joint_colour_only",
            "p_bf_effect_fog", "p_bf_null_fog", "p_bf_incon_fog",
            "p_bf_crossover"]
    cols = [c for c in cols if c in look.columns]
    look = look[cols].sort_values(["pattern", "share", "n"])
    look.to_csv(os.path.join(out_dir, "lookup_by_n.csv"), index=False)

    # The group test's failure mode: how often it reports evidence FOR
    # the null when a minority genuinely responds.
    fn = resp[(resp.pattern != "neither")
              & (resp.share < 1.0)][
        ["pattern", "weight", "spread", "w_corr", "share", "n",
         "p_bf_null_fog", "p_bf_effect_fog", "p_any_fog"]]
    fn.to_csv(os.path.join(out_dir, "false_null.csv"), index=False)

    # Sequential monitoring, headline scenario, one table per share.
    seq_rows = []
    for sh in RESPONDER_SHARES:
        s = sequential_stops(X, floors, PATTERNS[HEADLINE_PATTERN],
                             HEADLINE_WEIGHT, HEADLINE_SPREAD,
                             HEADLINE_CORR, sh, grid, tables, rng)
        for kind in ["effect", "null", "inconclusive"]:
            sel = s.stop_kind == kind
            # An inconclusive study never stopped, so its stop_n is NaN
            # by construction; nanmedian of an all-NaN slice warns.
            vals = s.stop_n[sel].to_numpy(dtype=float)
            med = (float(np.nanmedian(vals))
                   if np.isfinite(vals).any() else np.nan)
            seq_rows.append({
                "share": sh, "outcome": kind,
                "share_of_studies": float(sel.mean()),
                "median_stop_n": med})
        seq_rows.append({
            "share": sh, "outcome": "first 1 responder",
            "share_of_studies": float(
                np.isfinite(s.first_n_one_responder).mean()),
            "median_stop_n": float(
                np.nanmedian(s.first_n_one_responder))})
        seq_rows.append({
            "share": sh, "outcome": "first 2 responders",
            "share_of_studies": float(
                np.isfinite(s.first_n_two_responders).mean()),
            "median_stop_n": float(
                np.nanmedian(s.first_n_two_responders))})
    seq = pd.DataFrame(seq_rows)
    seq.to_csv(os.path.join(out_dir, "sequential.csv"), index=False)

    _plot_lookup(look, label, out_dir)
    _plot_recovery(resp, label, out_dir)
    _plot_false_null(resp, label, out_dir)
    _plot_sequential(seq, label, out_dir)

    return {"label": label, "path": path, "trials": trials, "r": r,
            "floors": floors, "resp": resp, "look": look, "fn": fn,
            "seq": seq, "out_dir": out_dir}


# ============================================================
# Figures
# ============================================================

def _plot_lookup(look, label, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for sh, g in look.groupby("share"):
        lab = f"{sh:.0%} respond"
        axes[0].plot(g.n, g.p_any_fog, marker="o", label=lab)
        axes[1].plot(g.n, g.p_bf_effect_fog, marker="o", label=lab)
        axes[2].plot(g.n, g.count_fog_mean, marker="o", label=lab)
    axes[0].set_title("PER PERSON\nat least one participant clears\n"
                      "their own fog floor")
    axes[1].set_title("GROUP TEST\nBayes factor reaches "
                      f"{BF_THRESHOLD:g}\n(the conventional threshold)")
    axes[2].set_title("THE COUNT\nhow many participants clear\n"
                      "the floor, on average")
    for ax in axes[:2]:
        ax.set_ylim(0, 1.02)
        ax.axhline(0.8, color="k", lw=0.7, ls=":")
    for ax in axes:
        ax.set_xlabel("participants")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"What N people buy  [{label}]  --  "
                 f"{HEADLINE_PATTERN}, weight {HEADLINE_WEIGHT:g}, "
                 f"spread {HEADLINE_SPREAD:g}", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lookup_by_n.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_recovery(resp, label, out_dir):
    """Does the responder COUNT recover the responder SHARE?"""
    g = resp[(resp.pattern == HEADLINE_PATTERN)
             & (resp.weight == HEADLINE_WEIGHT)
             & (resp.spread == HEADLINE_SPREAD)
             & (resp["w_corr"] == HEADLINE_CORR)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for sh, gg in g.groupby("share"):
        ax.plot(gg.n, gg.rate_fog, marker="o", label=f"true {sh:.0%}")
        ax.axhline(sh, color="k", lw=0.5, ls=":")
    ax.set_xlabel("participants")
    ax.set_ylabel("share called responders")
    ax.set_title("Recovered share vs the truth\n"
                 "(dotted lines are the true shares)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for sh, gg in g.groupby("share"):
        ax.fill_between(gg.n, gg.count_fog_p10, gg.count_fog_p90,
                        alpha=0.18)
        ax.plot(gg.n, gg.count_fog_mean, marker="o",
                label=f"true {sh:.0%}")
    ax.set_xlabel("participants")
    ax.set_ylabel("number called responders")
    ax.set_title("The count, with its 10th-90th percentile band\n"
                 "(the band is what a single study could return)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"The responder count  [{label}]", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "responder_recovery.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_false_null(resp, label, out_dir):
    """The group test concluding 'no effect' when the effect is real."""
    g = resp[(resp.pattern == HEADLINE_PATTERN)
             & (resp.weight == HEADLINE_WEIGHT)
             & (resp.spread == HEADLINE_SPREAD)
             & (resp["w_corr"] == HEADLINE_CORR)
             & (resp.share < 1.0)]
    if not len(g):
        return
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for sh, gg in g.groupby("share"):
        ax.plot(gg.n, gg.p_bf_null_fog, marker="o",
                label=f"{sh:.0%} respond")
    ax.set_xlabel("participants")
    ax.set_ylabel(f"P(BF10 < 1/{BF_THRESHOLD:g})")
    ax.set_title("The group test's failure mode\n"
                 "evidence FOR the null, when the effect is genuinely\n"
                 "present in a minority of participants")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "false_null.png"), dpi=150)
    plt.close(fig)


def _plot_sequential(seq, label, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    kinds = ["effect", "null", "inconclusive"]
    width = 0.8 / len(kinds)
    xs = np.arange(len(RESPONDER_SHARES))
    for j, kind in enumerate(kinds):
        vals = [float(seq[(seq.share == sh)
                          & (seq.outcome == kind)]
                      .share_of_studies.iloc[0])
                for sh in RESPONDER_SHARES]
        ax.bar(xs + j * width, vals, width, label=kind)
    ax.set_xticks(xs + width)
    ax.set_xticklabels([f"{sh:.0%}" for sh in RESPONDER_SHARES])
    ax.set_xlabel("responder share")
    ax.set_ylabel("share of simulated studies")
    ax.set_title("Where a sequentially-monitored study ends up\n"
                 f"(group Bayes factor on the fog weight, threshold "
                 f"{BF_THRESHOLD:g}, N up to {N_LOOKS[-1]})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sequential.png"), dpi=150)
    plt.close(fig)


# ============================================================
# Comparison and report
# ============================================================

def build_comparison(results, root):
    rows = {}

    def put(k, label, v):
        rows.setdefault(k, {})[label] = v

    for res in results:
        L = res["label"]
        put("analysed trials", L, len(res["trials"]))
        put("r(predictors)", L, round(res["r"], 3))
        put("floor, colour", L, round(float(res["floors"][0]), 3))
        put("floor, fog", L, round(float(res["floors"][1]), 3))
        lk = res["look"]
        for sh in RESPONDER_SHARES:
            g = lk[(lk.share == sh) & (lk.n == 10)]
            if len(g):
                k = int(g.k_required.iloc[0])
                put(f"P(existence claim, {k}+ detections) at N=10, "
                    f"{sh:.0%} respond", L,
                    round(float(g.p_claim_fog.iloc[0]), 3))
                put(f"P(>=1 detection) at N=10, {sh:.0%} respond", L,
                    round(float(g.p_any_fog.iloc[0]), 3))
                put(f"P(group BF>{BF_THRESHOLD:g}) at N=10, "
                    f"{sh:.0%} respond", L,
                    round(float(g.p_bf_effect_fog.iloc[0]), 3))
    comp = pd.DataFrame(rows).T
    comp.index.name = "quantity"
    comp.to_csv(os.path.join(root, "comparison.csv"))
    return comp


def write_report(results, comp, bf_worst, root, t0):
    path = os.path.join(root, "power_seq_report.txt")
    with open(path, "w") as f:
        def W(s=""):
            f.write(s + "\n")

        W("SEQUENTIAL / RESPONDER POWER REPORT  (v4 axes, step 7)")
        W("=" * 70)
        for r in results:
            W(f"{r['label']:10s} {r['path']}  "
              f"({len(r['trials'])} analysed trials)")
        W(f"Scenarios  : {len(scenario_grid())}"
          f"{'  (QUICK_GRID)' if QUICK_GRID else ''}, "
          f"{N_RUNS} studies each, N = {N_LOOKS[0]}..{N_LOOKS[-1]}")
        W(f"Headline   : {HEADLINE_PATTERN}, weight "
          f"{HEADLINE_WEIGHT:g}, spread {HEADLINE_SPREAD:g}, "
          f"corr {HEADLINE_CORR:g}")
        W(f"Prior      : JZS, scale {BF_SCALE}; threshold "
          f"{BF_THRESHOLD:g}")
        W(f"Per-person : floors at {FLOOR_ALPHA:.1%} false alarms, "
          f"ridge {RIDGE:g}")
        W(f"BF check   : fast form vs reference, worst relative "
          f"difference {bf_worst:.1e}")
        W(f"Elapsed    : {time.time() - t0:.1f}s")
        W()

        W("WHAT THIS SCRIPT IS FOR")
        W("-" * 70)
        W("  The sample is not chosen here -- the questionnaire")
        W("  decides it, and only some of the people it selects need")
        W("  carry the effect. Both facts break the standard group")
        W("  test, which averages: a real effect in a minority")
        W("  averages away to nothing.")
        W()
        W("  So the tables below report TWO things side by side at")
        W("  every sample size: how often at least one participant")
        W("  individually clears their floor, and how often the group")
        W("  test reaches its threshold. Where they disagree is the")
        W("  argument for making the per-person analysis primary.")
        W()

        if len(results) > 1:
            W("1. THE SETS SIDE BY SIDE")
            W("-" * 70)
            W(comp.to_string())
            W()

        start = 2 if len(results) > 1 else 1
        for i, r in enumerate(results, start):
            W(f"{i}. {r['label'].upper()}")
            W("-" * 70)
            W(f"  floors: colour {r['floors'][0]:.3f}, "
              f"fog {r['floors'][1]:.3f}")
            W("  Recomputed from this trial list, not typed in. Both")
            W("  are carried separately everywhere below: averaging")
            W("  them would test colour too leniently and fog too")
            W("  strictly, and the difference between the number of")
            W("  colour-trackers and fog-trackers is a finding.")
            W()

            W("  THE LOOKUP TABLE")
            W("  One row per (responder share, N). When the")
            W("  questionnaire returns 7 eligible people, read the")
            W("  rows with n = 7.")
            W()
            W("    share            fraction of the sample that truly")
            W("                     responds")
            W("    count_fog_*      how many are CALLED responders:")
            W("                     mean, and the 10th-90th percentile")
            W("                     band a single study could return")
            W("    k_required       detections needed before the")
            W("                     finding beats multiplicity at this")
            W(f"                     N, at a {STUDY_ALPHA:.0%} "
              "study-level rate")
            W("    pattern          WHICH FEATURES THE RESPONDERS")
            W("                     ACTUALLY TRACK. Read each claim")
            W("                     column against its own row's")
            W("                     pattern: under 'fog only' the true")
            W("                     colour effect is ZERO, so")
            W("                     p_claim_colour there is driven by")
            W("                     the spread between people, not by")
            W("                     colour. 'both' is the realistic")
            W("                     case and the one to plan with.")
            W("    p_claim_fog      P(reaching k_required on fog)")
            W("                     -- THE EXISTENCE CLAIM")
            W("    p_claim_colour   the same, for colour")
            W("    p_joint_*        both features claimed / only fog /")
            W("                     only colour. Not the product of the")
            W("                     two columns above: the same people")
            W("                     drive both.")
            W("    p_any_fog        P(at least ONE detection). NOT the")
            W("                     existence claim: with ten people at")
            W(f"                     {FLOOR_ALPHA:.1%} each, one false "
              "alarm turns up")
            W("                     22% of the time even when nothing")
            W("                     is there. Shown for contrast only.")
            W("    study_false_pos  the rate k_required actually buys")
            W("    p_bf_effect_fog  P(group BF reaches "
              f"{BF_THRESHOLD:g})")
            W("    p_bf_null_fog    P(group BF reaches "
              f"1/{BF_THRESHOLD:g}) -- evidence FOR")
            W("                     the null, i.e. the failure mode")
            W("    p_bf_incon_fog   P(inconclusive)")
            W("    p_bf_crossover   P(group shows fog > colour)")
            W()
            lk = r["look"]
            for pat in [q for q in LOOKUP_PATTERNS
                        if q in set(lk.pattern)]:
                W(f"  --- responders track: {pat.upper()} ---")
                blk = lk[lk.pattern == pat].drop(columns=["pattern"])
                W("  " + blk.round(3).to_string(index=False)
                  .replace("\n", "\n  "))
                W()
            W("  p_claim DOES NOT RISE SMOOTHLY WITH N. It steps DOWN")
            W("  wherever k_required steps up -- from N = 14 to N = 15")
            W("  the bar moves from 2 detections to 3, and the claim")
            W("  gets HARDER for a while even though the sample is")
            W("  larger. It takes about three more participants to")
            W("  recover. That is not an artefact: a bigger sample")
            W("  gives multiplicity more room, and the rule pays for")
            W("  it. Worth knowing before recruiting the 15th person,")
            W("  and worth stating in the preregistration so the")
            W("  threshold is fixed by the planned N rather than")
            W("  chosen once the sample is in.")
            W()

            # The comparison that carries the preregistration.
            W("  PER-PERSON VERSUS GROUP, AT THE LIKELY SAMPLE SIZE")
            # Read from "both" where available: it is the realistic
            # population, and the only one in which BOTH claim columns
            # mean what their names say.
            cmp_pat = ("both" if "both" in set(lk.pattern)
                       else lk.pattern.iloc[0])
            W(f"    (responders track: {cmp_pat})")
            for sh in RESPONDER_SHARES:
                g = lk[(lk.pattern == cmp_pat) & (lk.share == sh)
                       & (lk.n == 10)]
                if not len(g):
                    continue
                g = g.iloc[0]
                W(f"    {sh:.0%} respond, N = 10:")
                W(f"       EXISTENCE CLAIM ({int(g.k_required)}+ "
                  f"detections)")
                W(f"         on fog                           "
                  f"{g.p_claim_fog:.0%}")
                W(f"         on colour                        "
                  f"{g.p_claim_colour:.0%}")
                W(f"       (at least one detection            "
                  f"{g.p_any_fog:.0%} -- not a claim)")
                W(f"       group BF reaches {BF_THRESHOLD:g}            "
                  f"       {g.p_bf_effect_fog:.0%}")
                W(f"       group BF reaches 1/{BF_THRESHOLD:g} "
                  f"(says NO effect) {g.p_bf_null_fog:.0%}")
                W(f"       responders found: {g.count_fog_mean:.1f} "
                  f"on average, {g.count_fog_p10:.0f} to "
                  f"{g.count_fog_p90:.0f} in 8 studies out of 10")
            W()
            W("  The first line of each block is the one to")
            W("  preregister, and the second is there to show why. One")
            W("  detection among ten participants, each tested at")
            W(f"  {FLOOR_ALPHA:.1%}, happens 22% of the time in a study")
            W("  where nobody responds at all; two happens 2.5% of the")
            W("  time. So the claim needs the count, not the")
            W("  existence of a single flagged participant.")
            W()
            W("  The responder count with its band is the other number")
            W("  to state in advance. A band that wide is honest about")
            W("  what ten people can establish, and it is still more")
            W("  than the group test returns.")
            W()

            W("  SEQUENTIAL MONITORING")
            W("  Reading the group Bayes factor after each participant")
            W(f"  from N = {SEQ_MIN_N}, stopping at "
              f"{BF_THRESHOLD:g} or 1/{BF_THRESHOLD:g}. The last two")
            W("  rows of each block are the per-person equivalent:")
            W("  how many participants until the first, and the")
            W("  second, individual detection.")
            W()
            W("  " + r["seq"].round(3).to_string(index=False)
              .replace("\n", "\n  "))
            W()

            W("  THE GROUP TEST'S FAILURE MODE")
            W("  How often the group test reports evidence FOR the")
            W("  null while the effect is genuinely present in a")
            W("  minority. This is the number that justifies demoting")
            W("  it to secondary -- not its power, but the fact that")
            W("  it can actively mislead.")
            W()
            fn = r["fn"]
            fn = fn[(fn.pattern == HEADLINE_PATTERN)
                    & (fn.weight == HEADLINE_WEIGHT)
                    & (fn.spread == HEADLINE_SPREAD)
                    & (fn["w_corr"] == HEADLINE_CORR)
                    & (fn.n.isin([6, 10, 15, 20]))]
            if len(fn):
                W("  " + fn[["share", "n", "p_bf_null_fog",
                             "p_bf_effect_fog", "p_any_fog"]]
                  .round(3).to_string(index=False)
                  .replace("\n", "\n  "))
            W()

        W("WHAT TO WRITE IN THE PREREGISTRATION")
        W("-" * 70)
        W("  0. THE EXISTENCE CLAIM AND ITS COUNT. The claim is")
        W("     supported when at least k_required participants")
        W("     independently exceed the floor on the SAME feature,")
        W("     where k_required is read from the table above for the")
        W("     sample size actually obtained -- 2 for samples up to")
        W("     about 14, 3 from there to 20. State the number in")
        W("     advance. This turns the multiplicity problem into a")
        W("     counting rule and needs no correction formula.")
        W()
        W("  1. PRIMARY, per participant. Each participant's colour")
        W("     and fog weights are fitted from their own 180 trials.")
        W("     A weight above that feature's floor counts as a")
        W(f"     detection, one-sided, {FLOOR_ALPHA:.1%} false alarms.")
        W("     Both floors are stated in advance and differ by")
        W("     feature.")
        W()
        W("  2. SECONDARY, the responder count. Reported as a count")
        W("     with an interval, not as a proportion with a p-value.")
        W("     The interval is wide at N = 10 and the table above")
        W("     says how wide, in advance.")
        W()
        W("  3. SECONDARY AND UNDERPOWERED, the group test. Kept for")
        W("     comparability with Kusztor et al. Declare, with the")
        W("     numbers above, that it has low sensitivity at the")
        W("     anticipated N and a substantial chance of yielding")
        W("     evidence for the null even when a minority truly")
        W("     responds.")
        W()
        W("  4. THE INCONCLUSIVE CLAUSE. If the group Bayes factor")
        W(f"     falls between 1/{BF_THRESHOLD:g} and "
          f"{BF_THRESHOLD:g}, or fewer than a stated number of")
        W("     participants complete the task, the group-level")
        W("     question is reported as INCONCLUSIVE and inference is")
        W("     restricted to the per-participant level. Saying this")
        W("     in advance is what separates 'absence of evidence'")
        W("     from 'evidence of absence' -- and it is the only")
        W("     reason the Bayes factor is worth keeping, since it")
        W("     buys no power.")
        W()
        W("  5. WHAT IS ASSUMED. The effect's strength in a responder")
        W("     cannot be measured without a pilot, so it is swept:")
        W(f"     weights {EFFECT_WEIGHTS}, spreads {SPREAD_FACTORS},")
        W(f"     responder shares {RESPONDER_SHARES}. Every")
        W("     conclusion above is reported per assumption in")
        W("     responders.csv, so no single guess is load-bearing.")
        W()
        W("OVERALL: see the lookup table; there is no single verdict.")
    print(f"\n  Report : {path}")


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    root = resolve_output_root()
    os.makedirs(root, exist_ok=True)

    print("=" * 60)
    print("SEQUENTIAL / RESPONDER POWER  (v4 pipeline, step 7)")
    print("=" * 60)
    print(f"  Output : {root}")

    # The batch fit must agree with the single fit, or every number
    # here is wrong in a way nothing else would reveal.
    rng = np.random.default_rng(0)
    Xc = rng.normal(size=(180, 2))
    Yc = (rng.random((7, 180)) < 0.5).astype(float)
    single = np.array([fit_logit(Xc, y) for y in Yc])
    batch = fit_logit_batch(Xc, Yc)
    worst_fit = float(np.max(np.abs(single - batch)))
    print(f"  fit_logit vs fit_logit_batch: worst difference "
          f"{worst_fit:.2e}")
    if worst_fit > 1e-6:
        raise AssertionError("the batch fit disagrees with the single "
                             "fit; the simulation cannot be trusted.")

    print("\n  tabulating Bayes factors ...")
    grid, tables, bf_worst = build_bf_tables()

    results = []
    for label, path in TRIAL_LISTS:
        results.append(analyse_one(label, path, grid, tables, root))

    comp = build_comparison(results, root)
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("THE SETS SIDE BY SIDE")
        print("=" * 60)
        print(comp.to_string())

    write_report(results, comp, bf_worst, root, t0)
    print(f"  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
