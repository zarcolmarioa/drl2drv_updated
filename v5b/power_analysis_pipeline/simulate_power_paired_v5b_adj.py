"""
simulate_power_paired_v5b_adj.py  --  v5b power analysis, script 7 with the
                                      registered strategy-adjusted model

WHAT CHANGED FROM simulate_power_paired_v5b.py (side -> adjusted model)
    simulate_power_paired_v5b.py and its run are not modified. Changed:
      * PERSON_MODEL is a setting: "adjusted" (default; registered) or
        "side". The adjusted model is the side model plus two nuisance
        terms per block, N_corpus and N_orig (the lists' N, corpus SD):
        Firth logistic regression, no intercept, columns x_sat, x_con,
        side, N_corpus, N_orig. The side and N terms are nuisance terms;
        their estimates are never results. It is fitted on the analysed
        trials in item order, per block and set, exactly as script 7b
        (strategy_adjusted_model_v5b.py) did.
      * Its own thresholds: 2.5% per tail of script 6's 20,000 random
        choosers (design.json's stream), fitted with the adjusted model;
        they must equal script 7b's, and the secondary sign test uses them.
      * Everything model-dependent is rerun with the adjusted model on the
        v5b lists: false positives (rule and sensitivity rows), threats,
        the episode-dependent strategy rows, power (fixed and rescaled,
        every scenario), the test-choice rule (amended and original), the
        sign test, the per-block attention measures, the Monte Carlo
        replicates and the figures. The v5-list comparison rows stay with
        the side model, as in script 7.
      * The side model is shown beside the adjusted model as sensitivity
        rows in the headline tables (false positives, power, strategy
        rows); its rates are READ from script 7's run (PAIRED_RUN_DIR), not
        recomputed.
      * The mixed model is not rerun (RUN_MIXED_MODEL = False): it was
        assessed with the side model in script 7's run and was not worth
        registering; the report says so.
      * Reproduction checks: a slice of cells run with the side model
        (REPRO_SIDE_SLICE) and the v5-list rows must equal script 7's
        paired_power.csv (and secondary_sign.csv) exactly; with the
        adjusted model, every cell script 7b ran must equal its
        paired_rates.csv exactly.
      * The report and summary record why and when the model was chosen.
    The random streams are script 7's (STREAM_KEY "paired_v5b"), so every
    pool holds the same simulated people and answers as in script 7.

SCRIPT 7 (simulate_power_paired_v5b.py) CHANGED FROM simulate_power_paired_v5.py (v5 -> v5b)
    The person model, tests, recruitment, mixed model, per-block measures,
    the sign test, the Monte Carlo checks, the report and the figures are
    v5's. Changed:
      * INPUTS: the frozen v5b packages (each checksums.sha256 verified
        first), script 6's v5b run (the side model and its thresholds,
        registered by the researcher), explore_block_length_v5b.py for the
        sanity check, and the v5b stimulus set table for the "less edited"
        definition on the v5 lists (below).
      * STRATEGIES replace "edited looks odd". Two strategies that need no
        feature preference: "more typical" (a person adds +k x N_corpus to
        the logit: picks the image nearer the corpus median) and "less
        edited" (+k x N_orig: picks the image nearer the scene's original).
        N = dist(B) - dist(A) per trial, in corpus SD (the lists' N_corpus,
        N_orig). Each strategy is used by its own quarter of the people
        (separate quarters) at k = 0.5, both in "all nuisances
        (defaults)"; sensitivity rows: half the people, k = 0.25 and 1.0;
        "all nuisances (high ends)": half each (everyone uses one), k 1.0.
        A strategy matters only through the lists' leak (N regressed on the
        predictors); script 3 limited it to |b| <= 0.05.
      * EPISODE-DEPENDENT STRATEGY (a threat no test can remove): everyone
        uses -k x N in the derealization block and +k x N in the ordinary
        block, k = 0.25 / 0.5 / 1.0, both definitions; the reverse direction
        as information. The same rows run on the v5 lists (data/frozen_
        trial_lists_v5), whose N is computed here from their manifest.csv
        (the image features) with the originals from the v5b stimulus set
        table (the same 47 scenes; features measured on the stimulus path
        photograph -> 512 px -> 256 px), so v5 and v5b can be compared.
      * FATIGUE replaces "block order": it scales with trials. Per 100 main
        trials of the session (analysed, consistency and catch; practice
        not counted), the lapse rate rises by 0.030 and every preference is
        multiplied by 0.939, anchored at block 1's average (block 1's mean
        lapse and preferences are unchanged). Calibration: at 150 trials
        (167 main trials per block) the second block then has lapse +0.050
        and preferences x0.900 on average, v5's "block order" values; at
        180 (200 main trials) +0.060 and x0.881. Sensitivity rows: none,
        doubled, and 50% recovery at the rest (half of block 1's
        accumulated fatigue is gone when block 2 starts).
      * RESCALED WEIGHTS: planning and half shift (all nuisances at
        defaults) are also run with their effect parts rescaled per size by
        script 2's rescale_factors on the frozen lists (so the typical
        informative trial gives v4's choice probabilities); the fixed
        weights stay the headline.
      * THE RULE for the primary test is v5's AMENDED rule: a test
        qualifies if no more of its rule cells exceed ALPHA + 2 SE than
        chance gives in 95% of runs for an exactly calibrated test. The
        original (every cell) is reported beside it.
      * PARALLEL: every pool, sanity cell and mixed-model cell is one task
        in its own process (N_WORKERS, one thread each), with its own named
        random streams; results are gathered in a fixed order, so they do
        not depend on N_WORKERS (checked by rerunning one task in this
        process).
      * Relative paths only (run from the project folder); a plain-language
        summary (summary.txt and the report's first section).

WHAT THIS ANSWERS
    Each report-group participant does a derealization-episode block and an
    ordinary-episode block. Per block, the registered per-person model
    (script 6's choice: Firth logistic regression without intercept, with a
    saturation weight, a contrast weight and a side term) gives the weights.
    The PRIMARY test, per feature, is a one-sided test of the within-person
    difference (derealization-block weight minus ordinary-block weight)
    across the report group: H1 contrast, H2 saturation, each at alpha 0.05,
    one-sided, uncorrected. Candidates: the paired t-test and the exact
    Wilcoxon signed-rank test. This script:

      0. SANITY CHECK. Under explore_block_length_v5b.py's simplified person
         model, its own functions (imported, unmodified) and this script's
         simulation give the same power on the frozen lists, within Monte
         Carlo error; and the frozen lists' power lies within the range
         explore found over its draws of each size.
      1. FALSE POSITIVES with no effect: with no nuisance, with each
         nuisance alone (default value, then the ends of its range), with
         all at their defaults and all at their high ends.
      2. THREATS NO TEST CAN REMOVE, as their own rows: more lapses in the
         derealization block (+0.05, +0.10, +0.15), alone and with the
         strategies or "memory makes it duller"; a bias that differs between
         the episodes; and the episode-dependent strategy (v5b and v5 lists).
      3. POWER: planning, a quarter responding, all responding, each with
         its half-shift version beside it; with all nuisances at their
         defaults (the headline) and with none; planning and half shift
         also with rescaled weights.
      4. THE CHOICE OF PRIMARY TEST by the rule fixed in SETTINGS.
      5. A MIXED-EFFECTS LOGISTIC MODEL over all trials and people as an
         alternative: does it gain enough power, especially under half
         shift, to be worth registering?
      6. THE SECONDARY sign test on detected counts within each block.
      7. PER-BLOCK ATTENTION MEASURES: catch failures and consistency
         (agreement between a trial and its repeat), per block, and how
         often they pick up a lapse difference between the episodes.
    N = 10-20 report-group participants (analysable), both block sizes.

PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES
    No image-choice data exist. The share of responders, their shift
    (0.5 saturation, 0.7 contrast per corpus SD between the two blocks;
    0.25 / 0.35 under half shift), taste spread, lapse rates, and every
    nuisance value below are assumptions for planning. The report labels
    every figure that depends on them.

THE PERSON MODEL (the truth; the fit is always the registered model)
    For person i, block b (D = derealization, O = ordinary) and choice
    trial t showing scene s(t), with x_sat, x_con the trial's predictors,
    side_t = +1 if image A is on the left, -1 if on the right, and N_t the
    trial's typicality difference (corpus or original):

      pref = sum_f (w_ibf + u_i,s(t),b,f) x_tf
             + k_typ_i N_corpus,t + k_led_i N_orig,t + e_b N_episode,t
      logit P(choose A) = fat_t x pref + sum_f v_s(t),f x_tf
                          + side_ib * side_t + g_i,pair(t)
      w_iO = taste_iO + shared_i + dull
      w_iD = taste_iD + shared_i + dull + responder_i * shift
             + differential bias
      P    = (1 - lapse_ibt) logistic(.) + lapse_ibt / 2

    taste_iD, taste_iO   bivariate normal, SD TASTE_SD, correlation
                         taste_corr (1 = the same in both blocks)
    u_isb                person x scene deviation in the two weights,
                         SD scene_sd_mult x TASTE_SD, correlation
                         scene_corr between blocks (the same 47 scenes
                         appear in both blocks), so trials of the same
                         scene are not independent
    v_s                  a scene effect shared by everyone (common_scene_sd)
    side_ib              Normal(side_mean, side_sd), correlation side_corr
                         between blocks; + = prefers the left image
    dull                 everyone, both blocks, towards the predicted image
                         ("memory makes it duller")
    k_typ_i, k_led_i     the strategies: typical_share of people use
                         "more typical" (k_typ = typical_k), a separate
                         less_edited_share use "less edited" (k_led =
                         less_edited_k), both blocks
    e_b                  the episode-dependent strategy (threat rows only):
                         everyone, -k in the derealization block and +k in
                         the ordinary block (or the reverse), on one
                         definition of N
    fat_t, lapse_ibt     fatigue: with f_t = (session main-trial position
                         - block 1's mean position) / 100, fat_t =
                         fatigue_scale ** f_t and the lapse rises by
                         fatigue_lapse x f_t; fatigue_recovery takes that
                         share of block 1's accumulated fatigue off before
                         block 2
    lapse_ib             personal lapse rate, Gamma with mean LAPSE_MEAN
                         and SD LAPSE_BETWEEN_SD x mean (v4's model), the
                         same in both episodes by default; + episode_lapse
                         in the derealization block (threat rows only);
                         lapse_drift makes it rise across a block's trials
                         (centred, so the block mean is unchanged)
    Catch trials: 5 per block, each failed with probability
    CATCH_P_FAIL_PER_LAPSE x the block's mean lapse rate (v4's model); a
    person with 3 or more failures over the session is excluded and
    recruitment continues. Consistency repeats (12 or 15 per block, not
    analysed) are simulated with the same model: agreement = the same image
    chosen on the original and its repeat (the repeat shows it on the other
    side).
    pair_sd adds a person x pair preference for image A (logit), the same
    on the original and its repeat; without it (the default) people agree
    with themselves on repeats at about chance, because their only
    preferences are the small feature tastes.
    Counterbalancing: participant k takes the next slot of the frozen
    assignment table (block order, set mapping, trial order); excluded
    people use up their slots, as in the real study.

REUSED (imported, not copied; SHA-256 of each file recorded and checked)
    simulate_power_v4_n10_20.py   fit_firth_batch (two predictors),
                                  fit_firth (single person),
                                  sign_test_critical (Table 4's rule),
                                  FIRTH_ITER / FIRTH_TOL / FIRTH_MAX_STEP
    thresholds_v5b.py (script 6)  fit_firth_batch_k (the side model)
    explore_block_length_v5b.py   paired_pool, study_index, paired_tests
                                  (the sanity check, run unmodified),
                                  wilcoxon_greater (the exact one-sided
                                  Wilcoxon, vectorised; checked against
                                  SciPy here), choice_probability,
                                  rescale_factors, scaled_scenario (the
                                  rescaled weights)
    analyse_pair_capacity_v5b.py  Streams (named random generators)
    The expected SHA-256 of simulate_power_v4_n10_20.py and
    analyse_pair_capacity_v5b.py is read from the packages' FROZEN.json.

THE MIXED MODEL
    One logistic model per study over all analysed trials of both blocks
    and all people, no intercept: fixed effects saturation, contrast,
    saturation x block, contrast x block (block coded +1/2 derealization,
    -1/2 ordinary) and the side term; random person slopes on all of them,
    uncorrelated. Fitted by the Laplace approximation (each person's
    random effects at their posterior mode, Newton steps) with EM updates
    of the fixed effects and variances. One-sided Wald z on the two
    block-by-feature effects, SE from the generalised-least-squares form
    [sum_i (Sigma + V_i)^-1]^-1, V_i the person's own sampling covariance;
    the t(N-1) reference is reported beside it. No R is used; the fit is
    checked by recovering known parameters.

INPUTS (relative to the project folder; run from there)
    data/frozen_trial_lists_v5b/v5b_150/, v5b_180/   (checksums verified)
    data/frozen_trial_lists_v5/v5_150/, v5_180/      (the comparison rows;
        checksums verified; manifest.csv, assignment.csv, design.json and
        the lists are read)
    data/stimulus_set_v5/7fd72262ba/   stimulus_manifest.csv (originals'
        features; its SHA-256 must equal FROZEN.json's) and
        reference_stimulus_path.csv (corpus median and SD)
    THRESHOLDS_RUN_DIR (script 6's output: thresholds.csv, checks.csv,
        run_meta.json)
    EXPLORE_RESULTS_CSV (explore_block_length_v5b's paired_power.csv)
    the four imported scripts above

OUTPUTS (one new folder per run under OUTPUT_ROOT, named after the key
settings plus a hash of all settings; never overwritten)
    report.txt                 what was done, what was found, every check
    summary.txt                the plain-language summary
    checks.csv                 every check, PASS or FAIL
    scenarios.csv              every scenario and its settings
    paired_power.csv           per scenario, size, N, test, feature: rate
    secondary_sign.csv         the sign test within each block
    block_measures.csv         catch failures and consistency per block,
                               and how often each flags a lapse difference
    mixed_model.csv            the mixed model beside the paired tests
    sanity_explore.csv         the sanity check
    pool_replicates.csv        headline cells on two independent pools
    leak.csv                   the leak of N on the predictors, v5b and v5
    fatigue_calibration.csv    the fatigue model's block means
    rescale_factors.csv        script 2's rescale factors on the lists
    power_vs_n.png, false_positives.png, block_measures.png,
    episode_strategy.png
    settings.json, run_meta.json
    (this version: mixed_model.csv is not written; paired_power.csv has a
    "model" column; side_model_script7.csv holds the side-model rows read
    from script 7's run; thresholds_adjusted.csv the adjusted model's
    thresholds; model_choice.csv the numbers behind the model choice)

REQUIRES  Python 3.9+, numpy, pandas, scipy; matplotlib for the figures
RUN       python simulate_power_paired_v5b_adj.py
          (about 1-3 minutes with 31 workers; 11 minutes with 2)
"""

# One thread per process (BLAS and OpenMP), set before numpy is imported,
# so the result is the same for any N_WORKERS.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

# ============================================================
# SETTINGS
# ============================================================

# --- Where things are (relative to the project folder) ---
# PROJECT_ROOT: None = the current folder (run the script from the project
# folder); paths are then used and recorded as relative paths.
PROJECT_ROOT = None
PACKAGES = {150: "data/frozen_trial_lists_v5b/v5b_150",
            180: "data/frozen_trial_lists_v5b/v5b_180"}
# V5_PACKAGES: the v5 frozen lists, for the episode-dependent strategy rows
# only (v5 versus v5b). Their N is computed here (see STIMULUS_SET_DIR).
V5_PACKAGES = {150: "data/frozen_trial_lists_v5/v5_150",
               180: "data/frozen_trial_lists_v5/v5_180"}
# STIMULUS_SET_DIR: the v5b stimulus set the packages were built from
# (FROZEN.json "stimulus_set"); its table gives the originals' features
# and the corpus median and SD.
STIMULUS_SET_DIR = "data/stimulus_set_v5/7fd72262ba"
BLOCK_SIZES = [150, 180]
SETS = ("A", "B")
ORDERS = (1, 2, 3, 4)
GROUP = "participant"        # the report group's rows of assignment.csv

V4_SIM_SCRIPT = "simulate_power_v4_n10_20.py"
CAPACITY_SCRIPT = "analyse_pair_capacity_v5b.py"
EXPLORE_SCRIPT = "explore_block_length_v5b.py"
THRESHOLDS_SCRIPT = "thresholds_v5b.py"
# EXPECTED_SHA256: scripts not recorded in the packages' FROZEN.json (the
# other two are checked against FROZEN.json's scripts_sha256).
#   explore_block_length_v5b.py: the committed version (script 2's run).
#   thresholds_v5b.py: the version delivered and run for script 6.
EXPECTED_SHA256 = {
    EXPLORE_SCRIPT:
        "dc38cc67b547eaa39cf0743143838fb0eeee84b4585ee5172613e3996024c526",
    THRESHOLDS_SCRIPT:
        "07b2423e480df29a36975e9072d2f5e81c9c71b908dfc281119111b30d4c4115",
}

# THRESHOLDS_RUN_DIR / PERSON_MODEL
#   Script 6's output folder, and the per-person model registered from it
#   ("side": two weights + side term, chosen by the rule and registered by
#   the researcher; "two-weight": the original model). The thresholds are
#   read from its thresholds.csv: for "side" the rows with source "same
#   choosers as design.json" (they replace the frozen ones; the lists are
#   not re-frozen), for "two-weight" the rows "design.json (registered)".
THRESHOLDS_RUN_DIR = ("data/power_v5b/thresholds_v5b/sizes150-180_choosers"
                      "20000_people10000_side0.25-1.5-popsd0.25_w0.5-0.7"
                      "_lapse0.1_seed42__s02abb51a")
# PERSON_MODEL (this version): "adjusted" (registered; the side model +
#   N_corpus + N_orig, thresholds computed here from script 6's random
#   choosers) or "side" (script 7's model; thresholds from script 6).
PERSON_MODEL = "adjusted"
THRESHOLD_SOURCE = {"side": "same choosers as design.json",
                    "two-weight": "design.json (registered)"}

# PAIRED_RUN_DIR: script 7's run (the side model): its side-model rates
#   are shown beside the adjusted model's, and the side-model slice and
#   the v5-list rows must reproduce it exactly. Its mixed-model verdict is
#   quoted.
PAIRED_RUN_DIR = ("data/power_v5b/paired_v5b/sizes150-180_N10-20_model-side"
                  "_pool50000_studies10000_mixed2500_seed42__s41cc6f67")
# ADJ_RUN_DIR: script 7b's run (side against adjusted): the adjusted
#   thresholds and every rate it computed must be reproduced exactly; the
#   numbers behind the model choice are read from it.
ADJ_RUN_DIR = ("data/power_v5b/adjusted_model_v5b/side-vs-adjusted"
               "_seed42__sce30772f")
# REPRO_SIDE_SLICE: (size, scenario, effect) cells also run with the side
#   model and compared with script 7's run (the v5-list rows always are).
REPRO_SIDE_SLICE = [
    (150, "no nuisances", "no effect"),
    (180, "all nuisances (defaults)", "half shift"),
    (150, "episode-dependent strategy (corpus, k 0.5): derealization picks "
          "the less typical, ordinary the more typical", "no effect"),
]
# SIDE_BESIDE: show script 7's side-model rates beside the headline tables.
SIDE_BESIDE = True
SIDE_LABEL = "side (script 7)"
RATE_TOL = 0.0      # reproduced rates must be identical
THRESHOLD_TOL = 1e-12

# EXPLORE_RESULTS_CSV / EXPLORE_MIX / EXPLORE_TSET
#   explore_block_length_v5b's paired_power.csv, and the mix and threshold
#   set whose draws match the frozen lists (fixed weights).
EXPLORE_RESULTS_CSV = (
    "data/block_length_v5b/v5b_sets-v5units-sd060_sizes120-150-180_mixes5"
    "_draws20_w0.5-0.7_lapse0.1_null20000_seed42__sad6f14a4/"
    "paired_power.csv")
EXPLORE_MIX = "balanced_mixed"
EXPLORE_TSET = "v5units"

OUTPUT_ROOT = "data/power_v5b/paired_adj_v5b"
MAKE_FIGURES = True
FIG_DPI = 150

# N_WORKERS / CHECK_ONE_WORKER
#   Processes for the tasks (pools, sanity cells, mixed-model cells). None
#   = the number of cores minus one; 1 = everything in this process. The
#   result does not depend on it. CHECK_ONE_WORKER reruns the first pool
#   in this process and requires identical results.
N_WORKERS = None
CHECK_ONE_WORKER = True

# --- People (PLANNING ASSUMPTIONS, not estimates) ---
#
# TASTE_SD
#   Between-person SD of personal taste on (saturation, contrast): v4's
#   non-responder spread, 0.5 x the v4 thresholds (0.286, 0.488).
TASTE_SD = (0.143, 0.244)

# LAPSE_MEAN / LAPSE_BETWEEN_SD / LAPSE_MAX
#   Personal lapse rates: Gamma with this mean and SD LAPSE_BETWEEN_SD x
#   mean, capped at LAPSE_MAX (v4's model, simulate_power_v4_n10_20).
LAPSE_MEAN = 0.1
LAPSE_BETWEEN_SD = 0.5
LAPSE_MAX = 0.9

# Catch trials and exclusion (registered rule; v4's catch model)
CATCH_PER_BLOCK = 5
CATCH_P_FAIL_PER_LAPSE = 0.75
CATCH_FAIL_EXCLUDE = 3          # 3 or more of 10 over the session
EXCLUSION = True

# EFFECTS
#   share: share of responders; shift: their derealization-only shift
#   (saturation, contrast) per corpus SD, with between-responder SD
#   shift_sd; shared: a responder preference present in BOTH blocks.
#   "half shift": half of each responder's preference is shared by both
#   episodes, so only 0.25 / 0.35 separates the blocks.
EFFECTS = {
    "no effect": {"share": 0.0, "shift": (0.0, 0.0),
                  "shift_sd": (0.0, 0.0), "shared": (0.0, 0.0)},
    "planning": {"share": 0.5, "shift": (0.5, 0.7),
                 "shift_sd": (0.0, 0.0), "shared": (0.0, 0.0)},
    "half shift": {"share": 0.5, "shift": (0.25, 0.35),
                   "shift_sd": (0.0, 0.0), "shared": (0.25, 0.35)},
    "a quarter respond": {"share": 0.25, "shift": (0.5, 0.7),
                          "shift_sd": (0.0, 0.0), "shared": (0.0, 0.0)},
    "a quarter respond, half shift": {"share": 0.25, "shift": (0.25, 0.35),
                                      "shift_sd": (0.0, 0.0),
                                      "shared": (0.25, 0.35)},
    "all respond": {"share": 1.0, "shift": (0.5, 0.7),
                    "shift_sd": (0.0, 0.0), "shared": (0.0, 0.0)},
    "all respond, half shift": {"share": 1.0, "shift": (0.25, 0.35),
                                "shift_sd": (0.0, 0.0),
                                "shared": (0.25, 0.35)},
}
# RESCALE_EFFECTS
#   Also run with all nuisances at their defaults and the effect parts
#   (shift, shared, shift spread) multiplied per size and feature by
#   script 2's rescale_factors on the frozen lists (effect names get
#   RESCALED_SUFFIX). The fixed weights stay the headline.
RESCALE_EFFECTS = ("planning", "half shift")
RESCALED_SUFFIX = " (rescaled weights)"

# STRATEGY_SHARE / STRATEGY_K
#   The two strategies at their defaults: each used by its own quarter of
#   the people (separate quarters), k = 0.5 per unit of N (corpus SD).
STRATEGY_SHARE = 0.25
STRATEGY_K = 0.5

# FATIGUE_LAPSE_PER_100 / FATIGUE_SCALE_PER_100
#   Per 100 main trials: lapse + 0.030, preferences x 0.939, anchored at
#   block 1's mean position. Calibrated so that at 150 trials (167 main
#   trials per block) block 2 has v5's "block order" values on average
#   (lapse +0.05; preferences 0.939 ** 1.67 = x0.900).
FATIGUE_LAPSE_PER_100 = 0.030
FATIGUE_SCALE_PER_100 = 0.939
FATIGUE_RECOVERY_ROW = 0.5     # the "recovery at the rest" sensitivity row

# NUISANCES
#   NUISANCE_OFF: none (taste identical in both blocks, no scene effects,
#   no side bias, no shared bias, no strategy, no fatigue, equal lapses;
#   personal lapse rates still differ between people).
#   NUISANCE_DEFAULT: the plausible values agreed for planning.
#   The ranges swept are in SENSITIVITY_ROWS below.
NUISANCE_OFF = {
    "taste_corr": 1.0,            # taste correlation between blocks
    "scene_sd_mult": 0.0,         # person x scene SD, x TASTE_SD
    "scene_corr": 0.7,            # its correlation between blocks
    "common_scene_sd": 0.0,       # scene effect shared by everyone
    "side_mean": 0.0,             # side bias, logit (+ = left)
    "side_sd": 0.0,
    "side_corr": 0.8,             # its correlation between blocks
    "dull": (0.0, 0.0),           # "memory makes it duller", everyone
    "typical_share": 0.0,         # "more typical": share of people ...
    "typical_k": STRATEGY_K,      # ... and their k (x N_corpus)
    "less_edited_share": 0.0,     # "less edited": a separate share ...
    "less_edited_k": STRATEGY_K,  # ... and their k (x N_orig)
    "fatigue_lapse": 0.0,         # lapse + per 100 main trials
    "fatigue_scale": 1.0,         # preferences x per 100 main trials
    "fatigue_recovery": 0.0,      # share of block 1's fatigue gone at rest
    "episode_lapse": 0.0,         # derealization block: + lapse (threat)
    "differential_bias": (0.0, 0.0),   # derealization block only (threat)
    "episode_strategy": None,     # (definition, k_D, k_O) (threat)
    "lapse_drift": 0.0,           # lapse rise across a block, centred
    "pair_sd": 0.0,               # person x pair preference for image A,
                                  # logit SD; the same on a pair's repeat
}
FATIGUE_DEFAULT = {"fatigue_lapse": FATIGUE_LAPSE_PER_100,
                   "fatigue_scale": FATIGUE_SCALE_PER_100}
FATIGUE_DOUBLE = {"fatigue_lapse": 2 * FATIGUE_LAPSE_PER_100,
                  "fatigue_scale": FATIGUE_SCALE_PER_100 ** 2}
STRATEGIES_DEFAULT = {"typical_share": STRATEGY_SHARE,
                      "less_edited_share": STRATEGY_SHARE}
NUISANCE_DEFAULT = {**NUISANCE_OFF,
                    "taste_corr": 0.7, "scene_sd_mult": 1.0,
                    "scene_corr": 0.7, "side_sd": 0.25,
                    "dull": (0.25, 0.35), **STRATEGIES_DEFAULT,
                    **FATIGUE_DEFAULT}
NUISANCE_HIGH = {**NUISANCE_OFF,
                 "taste_corr": 0.0, "scene_sd_mult": 2.0, "scene_corr": 0.0,
                 "side_mean": 0.2, "side_sd": 0.5,
                 "dull": (0.5, 0.7),
                 "typical_share": 0.5, "typical_k": 1.0,
                 "less_edited_share": 0.5, "less_edited_k": 1.0,
                 **FATIGUE_DOUBLE}

# The single-nuisance rows used by the test-choice rule (each alone, on
# NUISANCE_OFF, at its default value).
DEFAULT_ROWS = [
    ("taste stability 0.7", {"taste_corr": 0.7}),
    ("scene dependence (SD x1, stability 0.7)",
     {"scene_sd_mult": 1.0, "scene_corr": 0.7}),
    ("side bias (SD 0.25)", {"side_sd": 0.25}),
    ("memory makes it duller (+0.25/+0.35, everyone)",
     {"dull": (0.25, 0.35)}),
    ("strategy: more typical (a quarter, k 0.5)",
     {"typical_share": STRATEGY_SHARE}),
    ("strategy: less edited (a quarter, k 0.5)",
     {"less_edited_share": STRATEGY_SHARE}),
    ("fatigue (per 100 trials: lapse +0.030, preferences x0.939)",
     FATIGUE_DEFAULT),
]
# Sensitivity rows (reported; not used by the rule), each alone on
# NUISANCE_OFF.
SENSITIVITY_ROWS = [
    ("taste stability 0.5", {"taste_corr": 0.5}),
    ("taste stability 0.3", {"taste_corr": 0.3}),
    ("taste stability 0", {"taste_corr": 0.0}),
    ("scene dependence SD x0.5", {"scene_sd_mult": 0.5}),
    ("scene dependence SD x2", {"scene_sd_mult": 2.0}),
    ("scene dependence, stability 0", {"scene_sd_mult": 1.0,
                                       "scene_corr": 0.0}),
    ("scene dependence, stability 1", {"scene_sd_mult": 1.0,
                                       "scene_corr": 1.0}),
    ("common scene effect SD 0.1", {"common_scene_sd": 0.1}),
    ("side bias SD 0.5", {"side_sd": 0.5}),
    ("side bias SD 0.25, mean 0.2", {"side_sd": 0.25, "side_mean": 0.2}),
    ("memory makes it duller +0.5/+0.7", {"dull": (0.5, 0.7)}),
    ("strategy: more typical, half (k 0.5)", {"typical_share": 0.5}),
    ("strategy: more typical, a quarter, k 0.25",
     {"typical_share": STRATEGY_SHARE, "typical_k": 0.25}),
    ("strategy: more typical, a quarter, k 1.0",
     {"typical_share": STRATEGY_SHARE, "typical_k": 1.0}),
    ("strategy: less edited, half (k 0.5)", {"less_edited_share": 0.5}),
    ("strategy: less edited, a quarter, k 0.25",
     {"less_edited_share": STRATEGY_SHARE, "less_edited_k": 0.25}),
    ("strategy: less edited, a quarter, k 1.0",
     {"less_edited_share": STRATEGY_SHARE, "less_edited_k": 1.0}),
    ("fatigue doubled (per 100 trials: lapse +0.060, preferences x0.882)",
     FATIGUE_DOUBLE),
    ("fatigue, 50% recovery at the rest",
     {**FATIGUE_DEFAULT, "fatigue_recovery": FATIGUE_RECOVERY_ROW}),
    ("lapse rising across the block (+0.10)", {"lapse_drift": 0.10}),
    ("pair-specific preferences SD 1.0", {"pair_sd": 1.0}),
    ("pair-specific preferences SD 2.0", {"pair_sd": 2.0}),
]
# FATIGUE_VARIANTS: all nuisances at their defaults with fatigue replaced
# (no effect, planning and half shift; sensitivity rows).
FATIGUE_VARIANTS = [
    ("no fatigue", {"fatigue_lapse": 0.0, "fatigue_scale": 1.0}),
    ("fatigue doubled", FATIGUE_DOUBLE),
    ("fatigue, 50% recovery at the rest",
     {**FATIGUE_DEFAULT, "fatigue_recovery": FATIGUE_RECOVERY_ROW}),
]
# PAIR_SD_FOR_ROWS
#   A person's preference for one image of a specific pair, beyond the two
#   features (logit SD), the same when the pair is repeated: what makes
#   people agree with themselves on repeats. Not part of the agreed
#   defaults (0); the rows using it show what it does to the tests, to
#   power and to the consistency measure. A value for the PI to calibrate.
PAIR_SD_FOR_ROWS = 1.5
# Threats no test can remove (reported as their own rows; not in the rule).
EPISODE_LAPSES = [0.05, 0.10, 0.15]
THREAT_INTERACTION_LAPSE = 0.10
DIFFERENTIAL_BIAS = (0.10, 0.14)   # dulling 40% stronger in derealization
# EPISODE_STRATEGY_K / EPISODE_STRATEGY_DEFS
#   The episode-dependent strategy: everyone -k x N in the derealization
#   block and +k x N in the ordinary block (main rows; the reverse as
#   information), on the v5b lists and on the v5 lists.
EPISODE_STRATEGY_K = [0.25, 0.5, 1.0]
EPISODE_STRATEGY_DEFS = ("corpus", "orig")

# --- Studies ---
N_GRID = list(range(10, 21))      # analysable report-group participants
TABLE_N = [10, 15, 20]            # the N shown in report tables
# POOL_PEOPLE / N_STUDIES / N_BATCHES
#   People per pool (one pool per scenario and size, each person fitted on
#   both blocks) and studies per cell, drawn from the pool with
#   replacement. A finite pool adds its own Monte Carlo error: about
#   sqrt(N x phi^2 / POOL_PEOPLE) on a power near 0.5 at N = 20, which is
#   0.013 for 20,000 people and 0.008 for 50,000. With 10,000 studies
#   (0.005) the total is then <= 0.01. Every rate's SE is MEASURED: the pool
#   is split into N_BATCHES disjoint sub-pools, each study draws from one,
#   and the SE is the spread of the sub-pool rates / sqrt(N_BATCHES); it
#   includes the pool's error.
POOL_PEOPLE = 50_000
N_STUDIES = 10_000
N_BATCHES = 20
TESTS = ("t", "wilcoxon")
ALPHA = 0.05

# --- The rule for choosing the primary test (fixed before the run) ---
#   v5's AMENDED rule: a test qualifies if, over the rule rows (no
#   nuisance, each default nuisance alone, all defaults together; no
#   effect), at every N in RULE_N, both sizes and both features, no more of
#   its cells exceed ALPHA + RULE_SE_MULT x SE than chance gives in
#   RULE_CHANCE_QUANTILE of runs for a test whose false-positive rate is
#   exactly ALPHA (binomial; 5 of 108 cells here, as 5 of 96 in v5). SE is
#   that cell's measured Monte Carlo SE (pool and studies; never below the
#   binomial SE of ALPHA over N_STUDIES). The original rule (no cell over
#   the limit) is reported beside it.
#   Among qualifying tests, the one with the higher mean power over
#   planning and half shift (all nuisances at defaults, fixed weights),
#   RULE_N, both sizes and both features wins; within RULE_TIE of each
#   other, RULE_PREFER.
RULE_N = [10, 15, 20]
RULE_SE_MULT = 2.0
RULE_AMENDED = True
RULE_CHANCE_QUANTILE = 0.95
RULE_TIE = 0.02
RULE_PREFER = "wilcoxon"

# --- Sanity check against explore_block_length_v5b ---
#   Its scenarios run with its own functions on the frozen lists, and with
#   this script's simulation set to its simplified model; agreement if no
#   |z| exceeds SANITY_MAX_Z and at most SANITY_SHARE_BEYOND of |z| exceed
#   1.96. The frozen lists' power must lie within explore's range over its
#   draws, widened by SANITY_RANGE_SE standard errors.
SANITY_SCENARIOS = ["no effect", "planning", "half shift"]
SANITY_N = [10, 15, 20]
SANITY_MAX_Z = 3.5
SANITY_SHARE_BEYOND = 0.10
SANITY_RANGE_SE = 3.0

# --- Pool replicates (Monte Carlo check) ---
#   These scenarios (all nuisances at defaults) are rerun on an independent
#   pool; same criteria as the sanity check.
REPLICATE_EFFECTS = ["no effect", "planning", "half shift"]

# --- Checks on the inputs ---
N_TOL = 1e-5        # N rebuilt from the tables vs the lists' N (corpus SD)
LEAK_TOL = 2e-6     # leak recomputed here vs FROZEN.json (6 decimals there)
FATIGUE_TOL = 1e-9  # block means of the fatigue model vs their formulas

# --- The mixed model ---
#   Not rerun in this version: it was assessed with the side model in
#   script 7's run (PAIRED_RUN_DIR) and was not worth registering. Setting
#   RUN_MIXED_MODEL = True is not supported here (the mixed model has no
#   N terms).
RUN_MIXED_MODEL = False
MIXED_STUDIES = 2_500
MIXED_N = [10, 15, 20]
MIXED_CELLS = [("no effect", "no nuisances"),
               ("no effect", "all nuisances (defaults)"),
               ("planning", "all nuisances (defaults)"),
               ("half shift", "all nuisances (defaults)")]
MIXED_MAX_ITER = 300
MIXED_TOL = 1e-5               # largest change of a fixed effect per step
MIXED_CHUNK = 250              # studies fitted at once (memory)
MIXED_GAIN = 0.05              # "worth registering": power gain under
MIXED_GAIN_N = 15              # half shift at this N, both sizes, both
                               # features, with false positives controlled

# --- Per-block attention measures ---
#   Per study: consistency lower in the derealization block (one-sided
#   paired t on each person's agreement difference) and more catch
#   failures there (one-sided exact sign test, ties dropped), at ALPHA.

# --- Seeds ---
RANDOM_SEED = 42
STREAM_KEY = "paired_v5b"      # script 7's streams: the same pools and studies

# ============================================================
# Code
# ============================================================

import contextlib
import hashlib
import importlib.util
import io
import json
import multiprocessing as mp
import platform
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

FEATURES = ("sat", "con")
FEATURE_LABEL = {"sat": "saturation", "con": "contrast"}
ANALYSED_TYPES = ("sat_only", "fog_only", "together", "conflict")
STRAT_LABEL = {"corpus": "more typical (N_corpus)",
               "orig": "less edited (N_orig)"}
T0 = time.time()
QUIET = False      # True in worker processes


def log(msg):
    if not QUIET:
        print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def root_path(*parts):
    """A path under the project folder: relative (to the current folder)
    when PROJECT_ROOT is None."""
    if PROJECT_ROOT:
        return os.path.join(PROJECT_ROOT, *parts)
    return os.path.join(*parts)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
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


SETTING_NAMES = [
    "PROJECT_ROOT", "PACKAGES", "V5_PACKAGES", "STIMULUS_SET_DIR",
    "BLOCK_SIZES", "SETS", "ORDERS", "GROUP",
    "V4_SIM_SCRIPT", "CAPACITY_SCRIPT", "EXPLORE_SCRIPT", "THRESHOLDS_SCRIPT",
    "EXPECTED_SHA256", "THRESHOLDS_RUN_DIR", "PERSON_MODEL",
    "THRESHOLD_SOURCE", "EXPLORE_RESULTS_CSV", "EXPLORE_MIX", "EXPLORE_TSET",
    "OUTPUT_ROOT", "N_WORKERS", "CHECK_ONE_WORKER",
    "MAKE_FIGURES", "FIG_DPI", "TASTE_SD", "LAPSE_MEAN", "LAPSE_BETWEEN_SD",
    "LAPSE_MAX", "CATCH_PER_BLOCK", "CATCH_P_FAIL_PER_LAPSE",
    "CATCH_FAIL_EXCLUDE", "EXCLUSION", "EFFECTS", "RESCALE_EFFECTS",
    "RESCALED_SUFFIX", "STRATEGY_SHARE", "STRATEGY_K",
    "FATIGUE_LAPSE_PER_100", "FATIGUE_SCALE_PER_100", "FATIGUE_RECOVERY_ROW",
    "NUISANCE_OFF", "NUISANCE_DEFAULT", "NUISANCE_HIGH", "DEFAULT_ROWS",
    "SENSITIVITY_ROWS", "FATIGUE_VARIANTS",
    "EPISODE_LAPSES", "THREAT_INTERACTION_LAPSE", "DIFFERENTIAL_BIAS",
    "EPISODE_STRATEGY_K", "EPISODE_STRATEGY_DEFS", "PAIR_SD_FOR_ROWS",
    "N_GRID", "TABLE_N", "POOL_PEOPLE", "N_STUDIES", "N_BATCHES", "TESTS",
    "ALPHA", "RULE_N", "RULE_SE_MULT", "RULE_AMENDED",
    "RULE_CHANCE_QUANTILE", "RULE_TIE", "RULE_PREFER", "SANITY_SCENARIOS",
    "SANITY_N", "SANITY_MAX_Z", "SANITY_SHARE_BEYOND", "SANITY_RANGE_SE",
    "REPLICATE_EFFECTS", "N_TOL", "LEAK_TOL", "FATIGUE_TOL",
    "RUN_MIXED_MODEL", "MIXED_STUDIES", "MIXED_N",
    "MIXED_CELLS", "MIXED_MAX_ITER", "MIXED_TOL", "MIXED_CHUNK",
    "MIXED_GAIN", "MIXED_GAIN_N", "RANDOM_SEED", "STREAM_KEY",
    "PAIRED_RUN_DIR", "ADJ_RUN_DIR", "REPRO_SIDE_SLICE", "SIDE_BESIDE",
    "SIDE_LABEL", "RATE_TOL", "THRESHOLD_TOL",
]


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def settings_dict():
    g = globals()
    return {k: g[k] for k in SETTING_NAMES}


def resolve_output_dir():
    s = settings_dict()
    s.pop("N_WORKERS")              # cannot change the result
    h = hashlib.sha256(json.dumps(jsonable(s), sort_keys=True)
                       .encode()).hexdigest()[:8]
    name = (f"sizes{'-'.join(map(str, BLOCK_SIZES))}"
            f"_N{N_GRID[0]}-{N_GRID[-1]}_model-{PERSON_MODEL}"
            f"_pool{POOL_PEOPLE}_studies{N_STUDIES}"
            f"_mixed{MIXED_STUDIES if RUN_MIXED_MODEL else 0}"
            f"_seed{RANDOM_SEED}__s{h}")
    base = root_path(OUTPUT_ROOT, name)
    out, k = base, 1
    while os.path.exists(out):          # never overwrite an earlier run
        k += 1
        out = f"{base}_run{k}"
    os.makedirs(out)
    return out


def load_module(fname, ck=None, hashes=None, expected=None):
    """Import a script from its file. With ck, check that it exists and that
    its SHA-256 is the expected one (expected, else EXPECTED_SHA256)."""
    path = root_path(fname)
    if ck is not None:
        ok = os.path.exists(path)
        ck.add("inputs", f"{fname} exists", ok, path)
        if not ok:
            raise SystemExit(f"missing {path}")
        digest = sha256_of(path)
        hashes[fname] = digest
        want = (expected or EXPECTED_SHA256).get(fname)
        ck.add("inputs", f"{fname} is the expected version (SHA-256"
               + (", FROZEN.json" if expected else "") + ")",
               digest == want, f"on disk {digest[:16]}..., expected "
               f"{str(want)[:16]}...")
    spec = importlib.util.spec_from_file_location(
        os.path.splitext(fname)[0] + "_imported", path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------
# Inputs: packages, the stimulus table, thresholds
# ------------------------------------------------------------

def read_checksums(pkg_dir):
    rows = {}
    with open(os.path.join(pkg_dir, "checksums.sha256")) as f:
        for line in f:
            if line.strip():
                d, rel = line.strip().split(None, 1)
                rows[rel.lstrip("*").strip()] = d
    return rows


def verify_package(rel_dir, label, ck, hashes, needed):
    """checksums.sha256 first: every non-image entry must be present and
    match; images are verified when present (not needed here). Every file
    read must be listed. Returns the package folder and FROZEN.json."""
    pkg = root_path(rel_dir)
    sums = read_checksums(pkg)
    images = [r for r in sums if r.startswith("package/stimuli/")]
    others = [r for r in sums if r not in images]
    bad = [r for r in others if not os.path.exists(os.path.join(pkg, r))
           or sha256_of(os.path.join(pkg, r)) != sums[r]]
    present = [r for r in images if os.path.exists(os.path.join(pkg, r))]
    bad_img = [r for r in present
               if sha256_of(os.path.join(pkg, r)) != sums[r]]
    ck.add(label, "every non-image file matches checksums.sha256", not bad,
           f"{len(others) - len(bad)} of {len(others)} match"
           + (f"; mismatched or missing: {bad[:5]}" if bad else ""))
    ck.add(label, "every image present matches checksums.sha256 (images "
           "are not needed here)", not bad_img,
           f"{len(present)} of {len(images)} present, {len(bad_img)} differ")
    missing = [r for r in needed if r not in sums]
    ck.add(label, "every file read here is listed in checksums.sha256",
           not missing, f"{len(needed)} files"
           + (f"; not listed: {missing}" if missing else ""))
    with open(os.path.join(pkg, "FROZEN.json")) as f:
        frozen = json.load(f)
    ck.add(label, "FROZEN.json records the hash of this checksums.sha256",
           frozen.get("checksums_sha256")
           == sha256_of(os.path.join(pkg, "checksums.sha256")),
           f"package {frozen.get('package')}")
    hashes[f"{rel_dir}/FROZEN.json"] = sha256_of(
        os.path.join(pkg, "FROZEN.json"))
    hashes[f"{rel_dir}/checksums.sha256"] = sha256_of(
        os.path.join(pkg, "checksums.sha256"))
    for r in needed:
        hashes[f"{rel_dir}/{r}"] = sums.get(r)
    return pkg, frozen


def load_stimulus_table(frozen, ck, hashes):
    """The originals' features and the corpus median and SD, from the v5b
    stimulus set the packages were built from."""
    d = root_path(STIMULUS_SET_DIR)
    man = os.path.join(d, "stimulus_manifest.csv")
    ref = os.path.join(d, "reference_stimulus_path.csv")
    ok = os.path.exists(man) and os.path.exists(ref)
    ck.add("stimulus set", "stimulus_manifest.csv and "
           "reference_stimulus_path.csv exist", ok, STIMULUS_SET_DIR)
    if not ok:
        raise SystemExit(f"missing {man} or {ref}")
    h = sha256_of(man)
    hashes[f"{STIMULUS_SET_DIR}/stimulus_manifest.csv"] = h
    hashes[f"{STIMULUS_SET_DIR}/reference_stimulus_path.csv"] = \
        sha256_of(ref)
    ck.add("stimulus set", "stimulus_manifest.csv is the one the packages "
           "were built from (SHA-256 in FROZEN.json)",
           h == frozen["stimulus_set"].get("manifest_sha256"),
           f"{h[:16]}..., FROZEN.json "
           f"{str(frozen['stimulus_set'].get('manifest_sha256'))[:16]}...")
    sm = pd.read_csv(man)
    r = pd.read_csv(ref).set_index("feature")
    med = r.loc[["sat_mean", "contrast_local"], "p50"].to_numpy(float)
    sd = r.loc[["sat_mean", "contrast_local"], "sd"].to_numpy(float)
    ok = (np.allclose(sd, [39.182, 9.348], atol=5e-4))
    ck.add("stimulus set", "corpus SDs equal the predictors' (39.182, "
           "9.348)", ok, f"median {med.round(6).tolist()}, SD "
           f"{sd.round(6).tolist()}")
    orig = sm[sm["is_original"].astype(str) == "True"].set_index(
        "scene_id")[["sat_mean", "contrast_local"]]
    ck.add("stimulus set", "one original per scene (47 scenes)",
           len(orig) == 47 and orig.index.is_unique, f"{len(orig)}")
    return {"orig": orig, "med": med, "sd": sd}


def n_from_features(fa, fb, scene, feat, stim):
    """N_corpus and N_orig per trial: dist(B) - dist(A), dist = hypot of the
    features' z (corpus: from the corpus median; orig: from the scene's
    original), in corpus SD. feat: filename -> (sat_mean, contrast_local)."""
    med, sd = stim["med"], stim["sd"]
    a = feat.loc[fa].to_numpy(float)
    b = feat.loc[fb].to_numpy(float)
    o = stim["orig"].loc[scene].to_numpy(float)

    def dist(v, c):
        z = (v - c) / sd
        return np.hypot(z[:, 0], z[:, 1])
    return (dist(b, med) - dist(a, med), dist(b, o) - dist(a, o))


def leak_fit(X, n):
    """N regressed on the two predictors, no intercept (script 3's leak)."""
    b = np.linalg.lstsq(X, n, rcond=None)[0]
    return float(b[0]), float(b[1])


def load_package(size, ck, hashes, stim=None, version="v5b"):
    """Lists of this size, as arrays ready for simulation. version "v5b":
    the frozen v5b package (N from its lists); "v5": the v5 package, N
    computed from its manifest.csv and the stimulus table's originals."""
    rel = PACKAGES[size] if version == "v5b" else V5_PACKAGES[size]
    label = f"package {version} {size}"
    needed = (["package/design.json", "package/assignment.csv",
               "package/manifest.csv"]
              + (["package/items.csv"] if version == "v5b" else [])
              + [f"package/lists/{b}_o{k}.json" for b in SETS
                 for k in ORDERS])
    pkg, frozen = verify_package(rel, label, ck, hashes, needed)
    with open(os.path.join(pkg, "package", "design.json")) as f:
        design = json.load(f)
    counts = design["counts_per_block"]
    man = pd.read_csv(os.path.join(pkg, "package", "manifest.csv")
                      ).set_index("filename")[["sat_mean", "contrast_local"]]
    lists, scenes = {}, None
    item_order = {}
    for b in SETS:
        for k in ORDERS:
            lid = f"{b}_o{k}"
            with open(os.path.join(pkg, "package", "lists",
                                   f"{lid}.json")) as f:
                L = pd.DataFrame(json.load(f))
            if scenes is None:
                scenes = sorted(L.loc[L["trial_type"] != "catch",
                                      "scene_id"].dropna().unique())
            lists[lid] = L
    sidx = {s: i for i, s in enumerate(scenes)}
    out = {}
    worst_n, worst_d = 0.0, 0.0
    for lid, L in lists.items():
        b = lid[0]
        choice = L["trial_type"] != "catch"
        an = (L["analysed"] == True).to_numpy()        # noqa: E712
        ch = choice.to_numpy()
        n_an = int(an.sum())
        cons = L[L["trial_type"] == "consistency"]
        by_trial = {int(t): i for i, t in enumerate(L["trial"])}
        rep = np.array([by_trial[int(t)] for t in cons["trial"]])
        orig = np.array([by_trial[int(t)] for t in cons["repeat_of"]])
        same_a = (L.loc[cons.index, "file_a"].to_numpy()
                  == L["file_a"].to_numpy()[orig])
        if b not in item_order:
            item_order[b] = sorted(L.loc[an, "item_id"])
        pos_item = {it: i for i, it in enumerate(item_order[b])}
        X = L[["pred_sat", "pred_fog"]].to_numpy(float)
        X = np.nan_to_num(X)
        side = np.where(L["a_is_left"].fillna(False).astype(bool), 1.0, -1.0)
        ci = np.full(len(L), -1)
        ci[an] = [pos_item[it] for it in L.loc[an, "item_id"]]
        ci[cons.index.to_numpy()] = [pos_item[it] for it in
                                     cons["repeat_of_item"]]
        # N per row (choice rows; catch rows 0)
        Nc, No = np.zeros(len(L)), np.zeros(len(L))
        c = L[choice]
        nc, no = n_from_features(c["file_a"], c["file_b"], c["scene_id"],
                                 man, stim)
        if version == "v5b":
            worst_n = max(worst_n, float(np.max(np.abs(np.r_[
                nc - c["N_corpus"].to_numpy(float),
                no - c["N_orig"].to_numpy(float)]))))
            Nc[ch] = c["N_corpus"].to_numpy(float)
            No[ch] = c["N_orig"].to_numpy(float)
        else:
            Nc[ch], No[ch] = nc, no
        fa, fb = man.loc[c["file_a"]].to_numpy(), man.loc[c["file_b"]
                                                          ].to_numpy()
        worst_d = max(worst_d, float(np.max(np.abs(np.r_[
            (fa[:, 0] - fb[:, 0]) - c["d_sat"].to_numpy(float),
            (fb[:, 1] - fa[:, 1]) - c["d_fog"].to_numpy(float)]))))
        out[lid] = {
            "set": b, "choice_item": ci,
            "X": X, "side": side, "Nc": Nc, "No": No,
            "scene": np.array([sidx.get(s, 0) for s in
                               L["scene_id"].fillna(scenes[0])]),
            "choice": ch, "analysed": an,
            "pos": np.arange(len(L)) / (len(L) - 1),
            "rep": rep, "orig": orig, "same_a": same_a,
            "item_cols": np.array([pos_item[it] for it in
                                   L.loc[an, "item_id"]]),
            "n_cons": len(cons), "n_catch": int((~choice).sum()),
        }
        ok = (n_an == counts["analysed"] and len(cons) == counts["consistency"]
              and out[lid]["n_catch"] == counts["catch"]
              and bool(np.all(same_a))
              and len(L) == design["main_trials_per_block"]
              and set(L.loc[an, "trial_type"]) <= set(ANALYSED_TYPES))
        ck.add(label, f"{lid}: analysed, consistency and catch counts equal "
               "design.json; every repeat shows the same image A",
               ok, f"{n_an} analysed, {len(cons)} repeats, "
               f"{out[lid]['n_catch']} catch, {len(L)} main trials")
    ck.add(label, "the lists' d_sat and d_fog equal manifest.csv's features "
           "(A - B saturation, B - A contrast)", worst_d <= 1e-4,
           f"max |difference| {worst_d:.1e}")
    if version == "v5b":
        ck.add(label, "N_corpus and N_orig rebuilt from manifest.csv and the "
               "stimulus table's originals equal the lists' (validates the "
               "computation used for the v5 lists)", worst_n <= N_TOL,
               f"max |difference| {worst_n:.1e} (corpus SD)")
        it = pd.read_csv(os.path.join(pkg, "package", "items.csv"),
                         keep_default_na=False, na_values=[""])
    # item-order design matrices per set (for the mixed model and leak)
    leak = []
    for b in SETS:
        L0 = out[f"{b}_o1"]
        an_rows = np.flatnonzero(L0["analysed"])
        Xi = np.empty((len(an_rows), 2))
        si = np.empty(len(an_rows))
        Ni = np.empty((len(an_rows), 2))
        Xi[L0["item_cols"]] = L0["X"][an_rows]
        si[L0["item_cols"]] = L0["side"][an_rows]
        Ni[L0["item_cols"], 0] = L0["Nc"][an_rows]
        Ni[L0["item_cols"], 1] = L0["No"][an_rows]
        out[f"items_{b}"] = {"X": Xi, "side": si, "N": Ni}
        if version == "v5b":
            q = it[(it["source"] == f"set_{b}")
                   & (it["analysed"].astype(str) == "True")
                   ].sort_values("item_id")
            same = (list(q["item_id"]) == item_order[b] and np.allclose(
                q[["N_corpus", "N_orig"]].to_numpy(float), Ni, rtol=0,
                atol=1e-12))
            ck.add(label, f"set {b}: the analysed rows' N equal items.csv's",
                   same, f"{len(q)} items")
        for j, dname in enumerate(EPISODE_STRATEGY_DEFS):
            bs, bc = leak_fit(Xi, Ni[:, j])
            leak.append({"lists": version, "size": size, "set": b,
                         "definition": dname, "b_sat": bs, "b_con": bc,
                         "N_sd": float(Ni[:, j].std()),
                         "n": len(an_rows)})
    if version == "v5b":
        rec = pd.DataFrame(frozen["normality"]["recomputed"])
        rec = rec[rec["fit"] == "overall (both terms)"]
        worst = 0.0
        for r in leak:
            q = rec[(rec["set"] == r["set"])
                    & (rec["definition"] == r["definition"])]
            for t in ("b_sat", "b_con"):
                worst = max(worst, abs(float(q[q["term"] == t]["b"].iloc[0])
                                       - r[t]))
        ck.add(label, "the leak recomputed here equals FROZEN.json's "
               "normality.recomputed (overall fit, both definitions)",
               worst <= LEAK_TOL, f"max |difference| {worst:.1e}")
    asg = pd.read_csv(os.path.join(pkg, "package", "assignment.csv"))
    asg = asg[asg["group"] == GROUP].sort_values("slot").reset_index(drop=True)
    ck.add(label, f"assignment table: 64 slots for the {GROUP} group, each "
           "with a derealization and an ordinary block",
           len(asg) == 64 and set(asg["block1_episode"]) | set(
               asg["block2_episode"]) == {"derealization", "ordinary"},
           f"{len(asg)} rows")
    rows = []
    for r in asg.itertuples():
        d_first = r.block1_episode == "derealization"
        rows.append({"der_list": r.block1_list if d_first else r.block2_list,
                     "ord_list": r.block2_list if d_first else r.block1_list,
                     "der_pos": 1 if d_first else 2})
    return {"lists": out, "slots": rows, "n_scenes": len(scenes),
            "design": design, "main_trials": design["main_trials_per_block"],
            "frozen": frozen, "leak": leak, "version": version}


def load_thresholds(ck, hashes):
    d = root_path(THRESHOLDS_RUN_DIR)
    p = os.path.join(d, "thresholds.csv")
    ok = os.path.exists(p)
    ck.add("inputs", "script 6's thresholds.csv exists", ok, p)
    if not ok:
        raise SystemExit(f"missing {p}")
    for fn in ("thresholds.csv", "checks.csv", "run_meta.json"):
        hashes[f"{THRESHOLDS_RUN_DIR}/{fn}"] = sha256_of(os.path.join(d, fn))
    c = pd.read_csv(os.path.join(d, "checks.csv"))
    ck.add("inputs", "script 6's run passed every check",
           (c["status"] == "PASS").all(),
           f"{(c['status'] == 'PASS').sum()} PASS, "
           f"{(c['status'] != 'PASS').sum()} FAIL")
    with open(os.path.join(d, "run_meta.json")) as f:
        meta = json.load(f)
    ck.add("inputs", "script 6 chose the side model (the registered "
           "adjusted model adds the N terms to it)",
           meta.get("chosen_model") == "side",
           f"script 6: {meta.get('chosen_model')}")
    t = pd.read_csv(p)
    t = t[(t["model"] == "side")
          & (t["source"] == THRESHOLD_SOURCE["side"])]
    th = {}
    for size in BLOCK_SIZES:
        for b in SETS:
            q = t[(t["size"] == size) & (t["set"] == b)].set_index("feature")
            th[(size, b)] = {
                "upper": np.array([q.loc["saturation", "upper"],
                                   q.loc["contrast", "upper"]]),
                "lower": np.array([q.loc["saturation", "lower"],
                                   q.loc["contrast", "lower"]])}
    ck.add("inputs", "thresholds found for every size and set",
           len(th) == len(BLOCK_SIZES) * len(SETS),
           "; ".join(f"{s} {b}: sat {v['upper'][0]:+.3f}/{v['lower'][0]:+.3f}"
                     f", con {v['upper'][1]:+.3f}/{v['lower'][1]:+.3f}"
                     for (s, b), v in th.items()))
    return th, meta


# ------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------

def episode_name(dname, k, reverse=False):
    lab = "less typical" if dname == "corpus" else "more edited"
    opp = "more typical" if dname == "corpus" else "less edited"
    if not reverse:
        return (f"episode-dependent strategy ({dname}, k {k:g}): "
                f"derealization picks the {lab}, ordinary the {opp}")
    return (f"episode-dependent strategy ({dname}, k {k:g}, reverse): "
            f"derealization picks the {opp}, ordinary the {lab}")


def build_scenarios():
    """Every scenario, in report order. lists: which frozen lists ("v5b";
    "v5" for the comparison rows)."""
    S = []

    def add(name, effect, nuis, role, lists="v5b"):
        S.append({"name": name, "effect": effect, "nuis": dict(nuis),
                  "role": role, "lists": lists})

    add("no nuisances", "no effect", NUISANCE_OFF, "null (rule)")
    for name, o in DEFAULT_ROWS:
        add(name, "no effect", {**NUISANCE_OFF, **o}, "null (rule)")
    add("all nuisances (defaults)", "no effect", NUISANCE_DEFAULT,
        "null (rule)")
    for name, o in SENSITIVITY_ROWS:
        add(name, "no effect", {**NUISANCE_OFF, **o}, "null (sensitivity)")
    add("all nuisances (high ends)", "no effect", NUISANCE_HIGH,
        "null (sensitivity)")
    for vname, o in FATIGUE_VARIANTS:
        add(f"all nuisances (defaults), {vname}", "no effect",
            {**NUISANCE_DEFAULT, **o}, "null (sensitivity)")
    dull = dict(DEFAULT_ROWS)["memory makes it duller (+0.25/+0.35, "
                              "everyone)"]
    for el in EPISODE_LAPSES:
        add(f"derealization lapse +{el:g}", "no effect",
            {**NUISANCE_OFF, "episode_lapse": el}, "threat")
    for el in EPISODE_LAPSES:
        add(f"derealization lapse +{el:g}, with both strategies (defaults)",
            "no effect", {**NUISANCE_OFF, **STRATEGIES_DEFAULT,
                          "episode_lapse": el}, "threat")
    for el in EPISODE_LAPSES:
        add(f"derealization lapse +{el:g}, with memory makes it duller",
            "no effect", {**NUISANCE_OFF, **dull, "episode_lapse": el},
            "threat")
    add(f"all nuisances (defaults), derealization lapse "
        f"+{THREAT_INTERACTION_LAPSE:g}", "no effect",
        {**NUISANCE_DEFAULT, "episode_lapse": THREAT_INTERACTION_LAPSE},
        "threat")
    add(f"all nuisances (defaults), dulling stronger in derealization "
        f"(+{DIFFERENTIAL_BIAS[0]:g}/+{DIFFERENTIAL_BIAS[1]:g})",
        "no effect", {**NUISANCE_DEFAULT,
                      "differential_bias": DIFFERENTIAL_BIAS}, "threat")
    ps = {"pair_sd": PAIR_SD_FOR_ROWS}
    for el in EPISODE_LAPSES:
        add(f"derealization lapse +{el:g}, with pair-specific preferences "
            f"SD {PAIR_SD_FOR_ROWS:g}", "no effect",
            {**NUISANCE_OFF, **ps, "episode_lapse": el}, "threat")
    for lists, sfx in (("v5b", ""), ("v5", " [v5 lists]")):
        for rev in (False, True):
            for dname in EPISODE_STRATEGY_DEFS:
                for k in EPISODE_STRATEGY_K:
                    es = (dname, k, -k) if rev else (dname, -k, k)
                    role = "episode strategy" + (" (reverse)" if rev
                                                 else "")
                    add(episode_name(dname, k, rev) + sfx, "no effect",
                        {**NUISANCE_OFF, "episode_strategy": es},
                        role + (" [v5 lists]" if lists == "v5" else ""),
                        lists=lists)
    add(f"all nuisances (defaults) + pair-specific preferences SD "
        f"{PAIR_SD_FOR_ROWS:g}", "no effect", {**NUISANCE_DEFAULT, **ps},
        "null (sensitivity)")
    for eff in EFFECTS:
        if eff == "no effect":
            continue
        add("all nuisances (defaults)", eff, NUISANCE_DEFAULT, "power")
        add("no nuisances", eff, NUISANCE_OFF, "power (no nuisances)")
    for eff in RESCALE_EFFECTS:
        add("all nuisances (defaults)", eff + RESCALED_SUFFIX,
            NUISANCE_DEFAULT, "power (rescaled)")
    for eff in ("planning", "half shift"):
        add(f"all nuisances (defaults) + pair-specific preferences SD "
            f"{PAIR_SD_FOR_ROWS:g}", eff, {**NUISANCE_DEFAULT, **ps},
            "power (sensitivity)")
        for vname, o in FATIGUE_VARIANTS:
            add(f"all nuisances (defaults), {vname}", eff,
                {**NUISANCE_DEFAULT, **o}, "power (sensitivity)")
    names = [(s["name"], s["effect"]) for s in S]
    assert len(names) == len(set(names)), "duplicate scenario"
    return S


def effect_of(effect, size, resc):
    """The effect dict of an effect name; rescaled ones use script 2's
    scaled_scenario with this size's factors."""
    if effect.endswith(RESCALED_SUFFIX):
        base = EFFECTS[effect[:-len(RESCALED_SUFFIX)]]
        k = (resc[size]["k_sat"], resc[size]["k_con"])
        return _W["explore"].scaled_scenario(base, k)
    return EFFECTS[effect]


# ------------------------------------------------------------
# Estimation helpers
# ------------------------------------------------------------

class Fitter:
    def __init__(self, v4, s6):
        self.v4, self.s6 = v4, s6

    def __call__(self, X, Y):
        if X.shape[1] == 2:
            return self.v4.fit_firth_batch(X, Y)[0]
        return self.s6.fit_firth_batch_k(X, Y, self.v4.FIRTH_ITER,
                                         self.v4.FIRTH_TOL,
                                         self.v4.FIRTH_MAX_STEP)[0]


def model_X(X2, side, model):
    """List-order design of the two-weight and side models (the adjusted
    model is fitted in item order: adjusted_design)."""
    return X2 if model == "two-weight" else np.column_stack([X2, side])


def adjusted_design(items):
    """Item-order design of the adjusted model for one set: x_sat, x_con,
    side, N_corpus, N_orig (script 7b's)."""
    return np.column_stack([items["X"], items["side"], items["N"]])


# ------------------------------------------------------------
# Fatigue
# ------------------------------------------------------------

def fatigue_offsets(M, nuis):
    """Fatigue position f (in 100s of main trials, anchored at block 1's
    mean position) of a block's first trial, for the first and the second
    block. f_t = offset + row index / 100."""
    t1 = (M - 1) / 2.0
    first = -t1 / 100.0
    second = (M - nuis["fatigue_recovery"] * M - t1) / 100.0
    return first, second


def fatigue_calibration(pk_by_size, ck):
    """Block means of the fatigue model at its default, doubled and
    recovery values, and the check that block 1's mean is unchanged and
    block 2's equals the formula."""
    rows = []
    worst = 0.0
    for size in BLOCK_SIZES:
        M = pk_by_size[size]["main_trials"]
        idx = np.arange(M) / 100.0
        for name, o in [("default", FATIGUE_DEFAULT),
                        ("doubled", FATIGUE_DOUBLE),
                        ("50% recovery at the rest",
                         {**FATIGUE_DEFAULT,
                          "fatigue_recovery": FATIGUE_RECOVERY_ROW})]:
            nu = {**NUISANCE_OFF, **o}
            f1, f2 = fatigue_offsets(M, nu)
            r = {"size": size, "main_trials_per_block": M, "fatigue": name}
            for blk, off in (("block 1", f1), ("block 2", f2)):
                f = off + idx
                r[f"{blk} lapse change"] = float(nu["fatigue_lapse"]
                                                 * f.mean())
                r[f"{blk} preference factor (mean)"] = float(
                    (nu["fatigue_scale"] ** f).mean())
                r[f"{blk} preference factor (at mean position)"] = float(
                    nu["fatigue_scale"] ** f.mean())
            want = (1 - nu["fatigue_recovery"]) * M / 100.0
            worst = max(worst, abs(r["block 1 lapse change"]),
                        abs(r["block 2 lapse change"]
                            - nu["fatigue_lapse"] * want),
                        abs(r["block 2 preference factor (at mean "
                              "position)"]
                            - nu["fatigue_scale"] ** want))
            rows.append(r)
    df = pd.DataFrame(rows)
    ck.add("fatigue", "block 1's mean lapse is unchanged and block 2's mean "
           "lapse and preference factor follow the formula", worst
           <= FATIGUE_TOL, f"max |difference| {worst:.1e}")
    d = df[(df["fatigue"] == "default")].set_index("size")
    if 150 in d.index:
        ok = (abs(d.loc[150, "block 2 lapse change"] - 0.05) < 0.001
              and abs(d.loc[150, "block 2 preference factor (at mean "
                             "position)"] - 0.90) < 0.001)
        ck.add("fatigue", "calibration: at 150 trials block 2 has v5's "
               "'block order' values (lapse +0.05, preferences x0.90)", ok,
               f"lapse {d.loc[150, 'block 2 lapse change']:+.4f}, "
               f"preferences x{d.loc[150, 'block 2 preference factor (at mean position)']:.4f}")
    return df


# ------------------------------------------------------------
# The person model: one pool of people, each fitted on both blocks
# ------------------------------------------------------------

def simulate_pool(pk, eff, nuis, rng, fit, P=POOL_PEOPLE, model=PERSON_MODEL,
                  lapse_fixed=False, keep_y=False):
    """P people of one scenario; person j takes slot row j mod 64."""
    lists, slots = pk["lists"], pk["slots"]
    nsc = pk["n_scenes"]
    M = pk["main_trials"]
    row = np.arange(P) % len(slots)
    der_list = np.array([slots[r]["der_list"] for r in row])
    ord_list = np.array([slots[r]["ord_list"] for r in row])
    der_pos = np.array([slots[r]["der_pos"] for r in row])
    tsd = np.asarray(TASTE_SD)

    resp = rng.random(P) < eff["share"]
    shift = (np.asarray(eff["shift"]) + rng.normal(0, 1, (P, 2))
             * np.asarray(eff["shift_sd"]))
    shift = np.where(resp[:, None], shift, 0.0)
    shared = np.where(resp[:, None], np.asarray(eff["shared"]), 0.0)

    rho = nuis["taste_corr"]
    z1, z2 = rng.normal(0, 1, (P, 2)), rng.normal(0, 1, (P, 2))
    taste_O = z1 * tsd
    taste_D = (rho * z1 + np.sqrt(max(1 - rho ** 2, 0.0)) * z2) * tsd
    common = np.asarray(nuis["dull"])
    w = {"O": taste_O + shared + common,
         "D": taste_D + shared + common + shift
         + np.asarray(nuis["differential_bias"])}

    # the strategies: separate groups of people
    us = rng.random(P)
    s1 = nuis["typical_share"]
    s2 = s1 + nuis["less_edited_share"]
    k_typ = np.where(us < s1, nuis["typical_k"], 0.0)
    k_led = np.where((us >= s1) & (us < s2), nuis["less_edited_k"], 0.0)
    es = nuis["episode_strategy"]

    ssd = nuis["scene_sd_mult"] * tsd
    if nuis["scene_sd_mult"] > 0:
        rs = nuis["scene_corr"]
        a = rng.normal(0, 1, (P, nsc, 2))
        c = rng.normal(0, 1, (P, nsc, 2))
        u = {"O": a * ssd,
             "D": (rs * a + np.sqrt(max(1 - rs ** 2, 0.0)) * c) * ssd}
    else:
        u = {"O": None, "D": None}
    v = (rng.normal(0, nuis["common_scene_sd"], (nsc, 2))
         if nuis["common_scene_sd"] > 0 else None)

    rsd = nuis["side_corr"]
    y1, y2 = rng.normal(0, 1, P), rng.normal(0, 1, P)
    side = {"O": nuis["side_mean"] + nuis["side_sd"] * y1,
            "D": nuis["side_mean"] + nuis["side_sd"]
            * (rsd * y1 + np.sqrt(max(1 - rsd ** 2, 0.0)) * y2)}

    if lapse_fixed or LAPSE_MEAN <= 0:
        lam = np.full(P, LAPSE_MEAN)
    else:
        shape = 1.0 / LAPSE_BETWEEN_SD ** 2
        lam = np.clip(rng.gamma(shape, LAPSE_MEAN / shape, P), 0, LAPSE_MAX)
    # fatigue: the position of each block's first trial (100s of trials)
    f_first, f_second = fatigue_offsets(M, nuis)
    second = {"D": der_pos == 2, "O": der_pos == 1}
    fat0, lam_base, lam_b = {}, {}, {}
    for b in ("D", "O"):
        fat0[b] = np.where(second[b], f_second, f_first)
        lam_base[b] = lam + (nuis["episode_lapse"] if b == "D" else 0.0)
        fmean = fat0[b] + (M - 1) / 200.0          # block mean position
        lam_b[b] = np.clip(lam_base[b] + nuis["fatigue_lapse"] * fmean,
                           0.0, LAPSE_MAX)

    n_items = len(pk["lists"][f"items_{SETS[0]}"]["X"])
    k = {"two-weight": 2, "side": 3, "adjusted": 5}[model]
    # the adjusted model is fitted after the list loop, in item order, per
    # block and set (as script 7b); the others per list (as script 7)
    item_fit = model == "adjusted"
    out = {"row": row, "batch": (np.arange(P) // len(slots)) % N_BATCHES,
           "der_set": np.array([s[0] for s in der_list]),
           "ord_set": np.array([s[0] for s in ord_list]),
           "der_pos": der_pos, "responder": resp,
           "lapse_D": lam_b["D"], "lapse_O": lam_b["O"]}
    fat_on = (nuis["fatigue_scale"] != 1.0) or (nuis["fatigue_lapse"] != 0.0)
    for b, lst in (("D", der_list), ("O", ord_list)):
        est = np.empty((P, k))
        agree = np.empty(P)
        Yitem = (np.empty((P, n_items), dtype=np.uint8)
                 if keep_y or item_fit else None)
        for lid in np.unique(lst):
            m = np.flatnonzero(lst == lid)
            L = lists[lid]
            ch = np.flatnonzero(L["choice"])
            X = L["X"][ch]
            sc = L["scene"][ch]
            pref = w[b][m] @ X.T
            if u[b] is not None:
                pref += np.einsum("ptf,tf->pt", u[b][m][:, sc, :], X)
            pref += (k_typ[m][:, None] * L["Nc"][ch][None, :]
                     + k_led[m][:, None] * L["No"][ch][None, :])
            if es is not None:
                kb = es[1] if b == "D" else es[2]
                Nd = L["Nc"] if es[0] == "corpus" else L["No"]
                pref += kb * Nd[ch][None, :]
            f = fat0[b][m][:, None] + ch[None, :] / 100.0   # 100s of trials
            if fat_on and nuis["fatigue_scale"] != 1.0:
                pref *= nuis["fatigue_scale"] ** f
            eta = pref + side[b][m][:, None] * L["side"][ch][None, :]
            if v is not None:
                eta += (v[sc] * X).sum(1)[None, :]
            if nuis.get("pair_sd", 0) > 0:
                g = rng.normal(0, nuis["pair_sd"], (len(m), n_items))
                eta += g[:, L["choice_item"][ch]]
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
            lt = lam_base[b][m][:, None] + nuis["lapse_drift"] * (
                L["pos"][ch][None, :] - 0.5)
            if fat_on and nuis["fatigue_lapse"] != 0.0:
                lt = lt + nuis["fatigue_lapse"] * f
            lt = np.clip(lt, 0.0, LAPSE_MAX)
            p = (1 - lt) * p + lt / 2
            Y = (rng.random(p.shape) < p).astype(float)
            # column positions within the choice rows
            pos_in_ch = np.full(len(L["choice"]), -1)
            pos_in_ch[ch] = np.arange(len(ch))
            an = pos_in_ch[np.flatnonzero(L["analysed"])]
            Ya = Y[:, an]
            if not item_fit:
                Xa = model_X(L["X"][L["analysed"]],
                             L["side"][L["analysed"]], model)
                est[m] = fit(Xa, Ya)
            yr, yo = Y[:, pos_in_ch[L["rep"]]], Y[:, pos_in_ch[L["orig"]]]
            agree[m] = np.mean(np.where(L["same_a"][None, :], yr == yo,
                                        yr != yo), axis=1)
            if keep_y or item_fit:
                Yitem[np.ix_(m, L["item_cols"])] = Ya.astype(np.uint8)
        if item_fit:
            set_of = np.array([s[0] for s in lst])
            for sname in SETS:
                mm = np.flatnonzero(set_of == sname)
                est[mm] = fit(adjusted_design(lists[f"items_{sname}"]),
                              Yitem[mm].astype(float))
        out[f"est_{b}"] = est
        out[f"agree_{b}"] = agree
        out[f"catch_{b}"] = rng.binomial(
            CATCH_PER_BLOCK, np.clip(CATCH_P_FAIL_PER_LAPSE * lam_b[b], 0, 1))
        if keep_y:
            out[f"Y_{b}"] = Yitem
    out["excluded"] = (out["catch_D"] + out["catch_O"]) >= CATCH_FAIL_EXCLUDE
    return out


def person_thresholds(pool, size, TH):
    for b, key in (("D", "der_set"), ("O", "ord_set")):
        up = np.array([TH[(size, s)]["upper"] for s in SETS])
        lo = np.array([TH[(size, s)]["lower"] for s in SETS])
        i = (pool[key] == SETS[1]).astype(int)
        pool[f"up_{b}"], pool[f"lo_{b}"] = up[i], lo[i]


# ------------------------------------------------------------
# Studies: recruitment by slot, with exclusion
# ------------------------------------------------------------

def draw_studies(pool, N, S, rng, exclusion=EXCLUSION):
    """S studies of N analysable people. Study s draws from sub-pool
    s mod N_BATCHES only. Candidate k takes slot k (row (k-1) mod 64) and is
    drawn from that sub-pool's people of that row; anyone excluded uses up
    the slot and recruitment continues. Returns the people (S x N), the
    number excluded per study and each study's sub-pool."""
    rows, bat = pool["row"], pool["batch"]
    nrow = rows.max() + 1
    K = bat.max() + 1
    members = [[np.flatnonzero((rows == r) & (bat == k)) for r in range(nrow)]
               for k in range(K)]
    cnt = np.array([[len(m) for m in mk] for mk in members])
    M = np.zeros((K, nrow, cnt.max()), dtype=int)
    for k in range(K):
        for r in range(nrow):
            M[k, r, :cnt[k, r]] = members[k][r]
    excl = pool["excluded"] if exclusion else np.zeros(len(rows), bool)
    sb = np.arange(S) % K
    width = int(np.ceil(1.5 * N)) + 10
    while True:
        slot_row = np.arange(width) % nrow
        pick = (rng.random((S, width)) * cnt[sb[:, None], slot_row[None, :]]
                ).astype(int)
        cand = M[sb[:, None], slot_row[None, :], pick]
        ok = ~excl[cand]
        if (ok.sum(1) >= N).all():
            break
        width *= 2
    pos = np.argsort(~ok, axis=1, kind="stable")[:, :N]
    idx = np.take_along_axis(cand, pos, axis=1)
    return idx, pos[:, -1] + 1 - N, sb


def rate_se(x, sb):
    """Mean of a per-study quantity and its Monte Carlo SE from the
    sub-pool means (includes the pool's error)."""
    x = np.asarray(x, float)
    K = sb.max() + 1
    m = np.array([x[sb == k].mean() for k in range(K)])
    p, n = float(x.mean()), len(x)
    # never below the binomial SE of the studies (with p(1-p) floored at
    # 5/n, so a rate of exactly 0 or 1 does not get an SE of 0)
    floor = np.sqrt(max(p * (1 - p), 5.0 / n) / n) if set(np.unique(x)) <= \
        {0.0, 1.0} else 0.0
    return p, float(max(m.std(ddof=1) / np.sqrt(K), floor))


def t_greater(d):
    n = d.shape[1]
    s = d.std(1, ddof=1)
    t = d.mean(1) / (s / np.sqrt(n))
    return stats.t.sf(t, n - 1)


def run_tests(pool, idx, explore, crit):
    """Per study: paired tests per feature, the sign test per block, and
    the per-block attention tests."""
    res = {}
    D = pool["est_D"][idx, :2] - pool["est_O"][idx, :2]        # S x N x 2
    for j, f in enumerate(FEATURES):
        res[("t", f)] = t_greater(D[:, :, j]) <= ALPHA
        res[("wilcoxon", f)] = explore.wilcoxon_greater(D[:, :, j]) <= ALPHA
        res[("mean_diff", f)] = D[:, :, j].mean(1)
        for b in ("D", "O"):
            e = pool[f"est_{b}"][idx, j]
            a = (e > pool[f"up_{b}"][idx, j]).sum(1)
            d = a + (e < pool[f"lo_{b}"][idx, j]).sum(1)
            res[(f"sign_{b}", f)] = a >= crit[d]
    ag = pool["agree_D"][idx] - pool["agree_O"][idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        res[("consistency_flags", "-")] = np.nan_to_num(
            t_greater(-ag), nan=1.0) <= ALPHA
    cd = pool["catch_D"][idx] - pool["catch_O"][idx]
    npos, nnz = (cd > 0).sum(1), (cd != 0).sum(1)
    pc = np.where(nnz > 0, stats.binom.sf(npos - 1, np.maximum(nnz, 1), 0.5),
                  1.0)
    res[("catch_flags", "-")] = pc <= ALPHA
    return res


# ------------------------------------------------------------
# The mixed model (Laplace approximation, EM updates)
# ------------------------------------------------------------

def mixed_templates(pk, model):
    """Design rows per set mapping: derealization block on set A or B.
    Columns: sat, con, sat x block, con x block (block +1/2 derealization,
    -1/2 ordinary) [, side]."""
    out = []
    for der in SETS:
        orr = [s for s in SETS if s != der][0]
        parts = []
        for s, dcode in ((der, 0.5), (orr, -0.5)):
            it = pk["lists"][f"items_{s}"]
            Z = np.column_stack([it["X"], dcode * it["X"]])
            if model != "two-weight":
                Z = np.column_stack([Z, it["side"]])
            parts.append(Z)
        out.append(np.vstack(parts))
    return np.stack(out)                     # (2, T, k)


def fit_mixed(Zt, tsel, Y, max_iter=MIXED_MAX_ITER, tol=MIXED_TOL):
    """Many studies at once. Zt (2, T, k) design templates; tsel (S, N)
    which template each person uses; Y (S, N, T) responses.
    Returns beta (S, k), its SE (S, k), sigma^2 (S, k), iterations used,
    and whether each study converged."""
    S, N, T = Y.shape
    k = Zt.shape[2]
    St = (Zt[:, :, :, None] * Zt[:, :, None, :]).reshape(2, T, k * k)
    PN = S * N
    y = Y.reshape(PN, T).astype(float)
    tf = tsel.reshape(PN)
    grp = [np.flatnonzero(tf == j) for j in range(2)]
    b = np.zeros((PN, k))
    beta = np.zeros((S, k))
    s2 = np.full((S, k), 0.1)
    conv = np.zeros(S, bool)
    it_used = np.zeros(S, int)

    def info(bb):
        A = np.empty((PN, k * k))
        r = np.empty((PN, T))
        for j in range(2):
            g = grp[j]
            if len(g) == 0:
                continue
            eta = bb[g] @ Zt[j].T
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            A[g] = (p * (1 - p)) @ St[j]
            r[g] = y[g] - p
        grad = np.empty((PN, k))
        for j in range(2):
            g = grp[j]
            if len(g):
                grad[g] = r[g] @ Zt[j]
        return A.reshape(PN, k, k), grad

    for it in range(max_iter):
        bet_p = np.repeat(beta, N, axis=0)
        s2_p = np.repeat(s2, N, axis=0)
        for _ in range(2):                  # Newton steps to the modes
            A, g = info(b)
            g = g - (b - bet_p) / s2_p
            H = A + np.einsum("pk,kl->pkl", 1.0 / s2_p, np.eye(k))
            step = np.linalg.solve(H, g[:, :, None])[:, :, 0]
            step = np.clip(step, -2.0, 2.0)
            b = b + step
        A, _ = info(b)
        H = A + np.einsum("pk,kl->pkl", 1.0 / s2_p, np.eye(k))
        Hd = np.diagonal(np.linalg.inv(H), axis1=1, axis2=2)
        bs = b.reshape(S, N, k)
        beta_new = bs.mean(1)
        s2_new = np.maximum(((bs - beta_new[:, None]) ** 2
                             + Hd.reshape(S, N, k)).mean(1), 1e-8)
        change = np.abs(beta_new - beta).max(1)
        newly = (~conv) & (change < tol)
        it_used[~conv] = it + 1
        conv |= newly
        beta = np.where(conv[:, None] & ~newly[:, None], beta, beta_new)
        s2 = np.where(conv[:, None] & ~newly[:, None], s2, s2_new)
        if conv.all():
            break
    A, _ = info(b)
    V = np.linalg.inv(A + 1e-10 * np.eye(k))              # person sampling cov
    s2_p = np.repeat(s2, N, axis=0)
    W = np.linalg.inv(V + np.einsum("pk,kl->pkl", s2_p, np.eye(k)))
    Wsum = W.reshape(S, N, k, k).sum(1)
    cov = np.linalg.inv(Wsum)
    se = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    return beta, se, s2, it_used, conv


def check_mixed(Zt, rng, ck):
    """Recover known parameters: 40 studies of 150 people simulated from the
    mixed model itself (no lapse); the z-scores of the fixed effects must
    look standard normal."""
    k = Zt.shape[2]
    beta = np.array([0.4, 0.6, 0.3, 0.4, 0.1][:k])
    sd = np.array([0.15, 0.25, 0.2, 0.3, 0.2][:k])
    S, N = 40, 150
    tsel = rng.integers(0, 2, (S, N))
    bi = beta + rng.normal(0, 1, (S, N, k)) * sd
    Z = Zt[tsel]                                            # S,N,T,k
    p = 1 / (1 + np.exp(-np.einsum("sntk,snk->snt", Z, bi)))
    Y = (rng.random(p.shape) < p).astype(np.uint8)
    est, se, s2, its, conv = fit_mixed(Zt, tsel, Y)
    z = (est - beta) / se
    ok = (abs(z.mean()) < 0.35 and 0.7 < z.std() < 1.35 and conv.all())
    ck.add("mixed model", "recovers known fixed effects (40 studies of 150 "
           "people simulated from the model): z-scores centred, SD near 1, "
           "every fit converged", ok,
           f"mean z {z.mean():+.2f}, SD z {z.std():.2f}, iterations "
           f"{its.min()}-{its.max()}; mean estimated SD "
           f"{np.sqrt(s2).mean(0).round(2).tolist()} (true {sd.tolist()})")



# ------------------------------------------------------------
# Sanity check against explore_block_length_v5b (one task per size and
# scenario; the checks are made in the main process)
# ------------------------------------------------------------

def sanity_rows(v4, explore, pk, fit, cap, size, scen, ex_res):
    rows = []
    XA = pk["lists"]["items_A"]["X"]
    XB = pk["lists"]["items_B"]["X"]
    st = cap.Streams(RANDOM_SEED, STREAM_KEY, "sanity", size, scen)
    # explore's own functions, unmodified (its POOL_PEOPLE and N_STUDIES
    # module settings are set to this script's values in every process)
    esc = explore.PAIRED_SCENARIOS[scen]
    Dx, on_A, _, _ = explore.paired_pool(v4, XA, XB, esc, st("explore_pool"))
    # this script, set to explore's model
    eff = EFFECTS[scen]
    pool = simulate_pool(pk, eff, NUISANCE_OFF, st("own_pool"), fit,
                         P=POOL_PEOPLE, model="two-weight", lapse_fixed=True)
    D = pool["est_D"][:, :2] - pool["est_O"][:, :2]
    for N in SANITY_N:
        ix = explore.study_index(on_A, N, st(f"explore_idx{N}"))
        pe = explore.paired_tests(Dx, ix)
        idx, _, sb = draw_studies(pool, N, N_STUDIES, st(f"own_idx{N}"),
                                  exclusion=False)
        for j, f in enumerate(FEATURES):
            d = D[idx, j]
            own = {"t": rate_se(t_greater(d) <= ALPHA, sb),
                   "wilcoxon": rate_se(
                       explore.wilcoxon_greater(d) <= ALPHA, sb)}
            for test in TESTS:
                rx = float(np.mean(pe[(test, j)] <= ALPHA))
                ro, se_o = own[test]
                # same pool size and study count on both sides, so the same
                # Monte Carlo SE: the difference has sqrt(2) x it
                se = np.sqrt(2) * max(se_o, 1e-4)
                er = ex_res[(ex_res["size"] == size)
                            & (ex_res["mix"] == EXPLORE_MIX)
                            & (ex_res["tset"] == EXPLORE_TSET)
                            & (ex_res["weights"] == "fixed")
                            & (ex_res["scenario"] == scen)
                            & (ex_res["N"] == N)
                            & (ex_res["test"] == test)][f"power_{f}"]
                rows.append({
                    "size": size, "scenario": scen, "N": N,
                    "test": test, "feature": FEATURE_LABEL[f],
                    "explore_functions": rx, "this_script": ro,
                    "se_each": se_o, "z": (ro - rx) / se,
                    "draws_min": er.min(), "draws_mean": er.mean(),
                    "draws_max": er.max(), "n_draws": len(er)})
    return rows


def sanity_checks(df, ck):
    az = df["z"].abs()
    ck.add("sanity", "this script reproduces explore_block_length_v5b's "
           "functions under its simplified model (power and false positives "
           "on the frozen lists)",
           az.max() <= SANITY_MAX_Z and (az > 1.96).mean()
           <= SANITY_SHARE_BEYOND,
           f"{len(df)} comparisons; largest |z| {az.max():.2f}; "
           f"{(az > 1.96).mean():.1%} beyond 1.96; mean z {df['z'].mean():+.2f}")
    se = df["se_each"].clip(lower=0.002)
    inside = ((df["explore_functions"] >= df["draws_min"]
               - SANITY_RANGE_SE * se)
              & (df["explore_functions"] <= df["draws_max"]
                 + SANITY_RANGE_SE * se))
    df["within_draw_range"] = inside
    ck.add("sanity", "the frozen lists' power lies within explore's range "
           f"over its draws of the same size and mix (+/- "
           f"{SANITY_RANGE_SE:g} SE)",
           inside.all() and (df["n_draws"] > 0).all(),
           f"{int(inside.sum())} of {len(df)} cells; draws per cell "
           f"{int(df['n_draws'].min())}-{int(df['n_draws'].max())}")
    return df


# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------

def scenario_rows(pool, scen, size, explore, crit, cap, tag="", sign=True):
    rows, sign_rows, bm_rows = [], [], []
    st = cap.Streams(RANDOM_SEED, STREAM_KEY, "studies", scen["lists"], size,
                     scen["name"], scen["effect"], tag)
    keep = ~pool["excluded"]
    for N in N_GRID:
        idx, n_excl, sb = draw_studies(pool, N, N_STUDIES, st(f"N{N}"))
        r = run_tests(pool, idx, explore, crit)
        base = {"lists": scen["lists"], "size": size,
                "scenario": scen["name"], "effect": scen["effect"],
                "role": scen["role"], "N": N}
        for test in TESTS:
            for f in FEATURES:
                p, se = rate_se(r[(test, f)], sb)
                rows.append({**base, "test": test,
                             "feature": FEATURE_LABEL[f], "rate": p,
                             "se": se,
                             "mean_diff": float(np.mean(
                                 r[("mean_diff", f)])),
                             "excluded_per_study": float(np.mean(n_excl))})
        if sign:
            for b, lab in (("D", "derealization"), ("O", "ordinary")):
                for f in FEATURES:
                    p, se = rate_se(r[(f"sign_{b}", f)], sb)
                    sign_rows.append({**base, "block": lab,
                                      "feature": FEATURE_LABEL[f],
                                      "claim_rate": p, "se": se})
        pc, sc_ = rate_se(r[("consistency_flags", "-")], sb)
        pk_, sk = rate_se(r[("catch_flags", "-")], sb)
        bm_rows.append({**base, "consistency_flags": pc,
                        "consistency_se": sc_, "catch_flags": pk_,
                        "catch_se": sk})
    means = {"lists": scen["lists"], "size": size, "scenario": scen["name"],
             "effect": scen["effect"],
             "role": scen["role"], "N": "pool (analysable people)",
             "excluded_share": float(pool["excluded"].mean()),
             "catch_D": float(pool["catch_D"][keep].mean()),
             "catch_O": float(pool["catch_O"][keep].mean()),
             "agree_D": float(pool["agree_D"][keep].mean()),
             "agree_O": float(pool["agree_O"][keep].mean()),
             "lapse_D": float(pool["lapse_D"][keep].mean()),
             "lapse_O": float(pool["lapse_O"][keep].mean()),
             "lapse_D_all": float(pool["lapse_D"].mean()),
             "lapse_O_all": float(pool["lapse_O"].mean()),
             "mean_est_diff_sat": float((pool["est_D"][keep, 0]
                                         - pool["est_O"][keep, 0]).mean()),
             "mean_est_diff_con": float((pool["est_D"][keep, 1]
                                         - pool["est_O"][keep, 1]).mean())}
    return rows, sign_rows, bm_rows, means


def replicate_compare(a, b):
    m = a.merge(b, on=["lists", "size", "scenario", "effect", "N", "test",
                       "feature"], suffixes=("_1", "_2"))
    se = np.sqrt(m["se_1"] ** 2 + m["se_2"] ** 2).clip(lower=1e-4)
    m["z"] = (m["rate_2"] - m["rate_1"]) / se
    return m


def apply_rule(pw):
    """v5's rule, with the amendment (RULE_AMENDED): a test qualifies if no
    more of its cells exceed the limit than chance gives in
    RULE_CHANCE_QUANTILE of runs; the original asks for none."""
    se0 = np.sqrt(ALPHA * (1 - ALPHA) / N_STUDIES)
    nul = pw[(pw["role"] == "null (rule)") & pw["N"].isin(RULE_N)].copy()
    nul["limit"] = ALPHA + RULE_SE_MULT * np.maximum(nul["se"], se0)
    pwr = pw[(pw["role"] == "power")
             & pw["effect"].isin(["planning", "half shift"])
             & pw["N"].isin(RULE_N)]
    res = {"limit": float(nul["limit"].mean()),
           "limit_range": (float(nul["limit"].min()),
                           float(nul["limit"].max())), "tests": {}}
    n_cells = len(nul) // len(TESTS)
    res["cells"] = n_cells
    p1 = float(stats.norm.sf(RULE_SE_MULT))
    res["expected_exceed_if_exact"] = n_cells * p1
    res["chance_95"] = int(stats.binom.ppf(RULE_CHANCE_QUANTILE, n_cells, p1))
    res["amended"] = RULE_AMENDED
    for test in TESTS:
        q = nul[nul["test"] == test]
        ex = q[q["rate"] > q["limit"]]
        q_orig = len(ex) == 0
        q_amend = len(ex) <= res["chance_95"]
        res["tests"][test] = {
            "qualifies": q_amend if RULE_AMENDED else q_orig,
            "qualifies_original": q_orig, "qualifies_amended": q_amend,
            "n_exceed": len(ex),
            "exceed_cells": [f"{r.scenario}, {r.size}, N={r.N}, "
                             f"{r.feature}: {r.rate:.4f} > {r.limit:.4f}"
                             for r in ex.itertuples()],
            "worst": float(q["rate"].max()),
            "worst_where": (lambda r: f"{r['scenario']}, {r['size']}, "
                            f"N={r['N']}, {r['feature']}")(
                q.loc[q["rate"].idxmax()]),
            "mean_null": float(q["rate"].mean()),
            "power": float(pwr[pwr["test"] == test]["rate"].mean())}

    def choose(key):
        ok = [t for t in TESTS if res["tests"][t][key]]
        if not ok:
            return None, None
        if len(ok) == 1:
            return ok[0], None
        best = max(ok, key=lambda t: res["tests"][t]["power"])
        other = [t for t in ok if t != best][0]
        gap = res["tests"][best]["power"] - res["tests"][other]["power"]
        return (RULE_PREFER if gap <= RULE_TIE and RULE_PREFER in ok
                else best), gap
    res["chosen"], gap = choose("qualifies")
    if gap is not None:
        res["gap"] = gap
    res["chosen_original_rule"], _ = choose("qualifies_original")
    return res


# ------------------------------------------------------------
# Parallel tasks
# ------------------------------------------------------------

_W = {}          # per-process state


def _worker_init(state, quiet=True):
    """Load the imported scripts in this process (their hashes are checked
    once, in the main process) and keep the inputs."""
    global QUIET
    QUIET = quiet
    v4 = load_module(V4_SIM_SCRIPT)
    cap = load_module(CAPACITY_SCRIPT)
    explore = load_module(EXPLORE_SCRIPT)
    s6 = load_module(THRESHOLDS_SCRIPT)
    explore.POOL_PEOPLE = POOL_PEOPLE      # explore's settings, set to this
    explore.N_STUDIES = N_STUDIES          # script's values for the check
    _W.update(v4=v4, cap=cap, explore=explore, s6=s6, fit=Fitter(v4, s6),
              crit=v4.sign_test_critical(max(N_GRID)),
              scen=build_scenarios(), **state)


def _pool_for(s, size, keep_y=False, model=None):
    """A scenario's pool (script 7's stream), fitted with `model` (default:
    PERSON_MODEL on the v5b lists, the side model on the v5 lists), with
    that model's thresholds."""
    pkx = _W["pk"] if s["lists"] == "v5b" else _W["pk5"]
    if model is None:
        model = PERSON_MODEL if s["lists"] == "v5b" else "side"
    st = _W["cap"].Streams(RANDOM_SEED, STREAM_KEY, "pool", s["lists"], size,
                           s["name"], s["effect"])
    pool = simulate_pool(pkx[size], effect_of(s["effect"], size, _W["resc"]),
                         s["nuis"], st("people"), _W["fit"], keep_y=keep_y,
                         model=model)
    person_thresholds(pool, size, _W["TH"] if model == PERSON_MODEL
                      else _W["TH_side"])
    return pool


def _model_of(s):
    return PERSON_MODEL if s["lists"] == "v5b" else "side"


def _tag(rows, model):
    for r in rows:
        r["model"] = model
    return rows


def task_pool(size, i):
    s = _W["scen"][i]
    pool = _pool_for(s, size)
    a, b, c, m = scenario_rows(pool, s, size, _W["explore"], _W["crit"],
                               _W["cap"], sign=(s["lists"] == "v5b"))
    md = _model_of(s)
    return _tag(a, md), _tag(b, md), _tag(c, md), {**m, "model": md}


def task_side_repro(size, i):
    """A cell with the side model (script 7's fit and thresholds): must
    equal script 7's run exactly."""
    s = _W["scen"][i]
    pool = _pool_for(s, size, model="side")
    a, b, _, _ = scenario_rows(pool, s, size, _W["explore"], _W["crit"],
                               _W["cap"], sign=(s["lists"] == "v5b"))
    return a, b


def task_thresholds(size, b):
    """The adjusted model's thresholds: script 6's 20,000 random choosers
    (design.json's stream, script 7b's part 1), fitted with the adjusted
    model in item order."""
    v4, cap, s6 = _W["v4"], _W["cap"], _W["s6"]
    it = _W["pk"][size]["lists"][f"items_{b}"]
    rng = cap.Streams(s6.REPRO_SEED, s6.REPRO_STREAM_KEY, size)(
        s6.REPRO_STREAM_NAME.format(set=b))
    Yr = v4.simulate_many(it["X"], np.zeros((s6.N_THRESHOLD_PEOPLE, 2)), rng)
    est, last = s6.fit_firth_batch_k(adjusted_design(it), Yr, v4.FIRTH_ITER,
                                     v4.FIRTH_TOL, v4.FIRTH_MAX_STEP)
    th = s6.thresholds_from(est[:, :2])
    return {"size": size, "set": b, "upper": th["upper"],
            "lower": th["lower"], "null_sd": est[:, :2].std(axis=0),
            "unconverged": float(np.mean(last > 1e-6))}


def task_replicate(size, eff):
    s = [x for x in _W["scen"] if x["effect"] == eff
         and x["name"] == "all nuisances (defaults)"][0]
    st = _W["cap"].Streams(RANDOM_SEED, STREAM_KEY, "replicate", size, eff)
    pool = simulate_pool(_W["pk"][size], EFFECTS[eff], s["nuis"],
                         st("people"), _W["fit"], model=PERSON_MODEL)
    person_thresholds(pool, size, _W["TH"])
    a, _, _, _ = scenario_rows(pool, s, size, _W["explore"], _W["crit"],
                               _W["cap"], tag="replicate")
    return _tag(a, PERSON_MODEL)


def task_sanity(size, scen):
    return sanity_rows(_W["v4"], _W["explore"], _W["pk"][size], _W["fit"],
                       _W["cap"], size, scen, _W["ex_res"])


def task_mixed(size, cell, N):
    """The mixed model on one cell and N; the pool is the scenario's pool
    (same stream), rebuilt here with its responses kept."""
    eff, name = MIXED_CELLS[cell]
    s = [x for x in _W["scen"] if x["effect"] == eff and x["name"] == name
         and x["lists"] == "v5b"][0]
    pool = _pool_for(s, size, keep_y=True)
    Zt = mixed_templates(_W["pk"][size], PERSON_MODEL)
    st = _W["cap"].Streams(RANDOM_SEED, STREAM_KEY, "mixed", size, eff, name)
    t0 = time.time()
    idx, _, sb = draw_studies(pool, N, MIXED_STUDIES, st(f"N{N}"))
    r = run_tests(pool, idx, _W["explore"], _W["crit"])
    tsel = (pool["der_set"][idx] == SETS[1]).astype(int)
    Z_rej = {f: [] for f in FEATURES}
    T_rej = {f: [] for f in FEATURES}
    convs, its = [], []
    for c0 in range(0, MIXED_STUDIES, MIXED_CHUNK):
        sl = slice(c0, c0 + MIXED_CHUNK)
        Y = np.concatenate([pool["Y_D"][idx[sl]], pool["Y_O"][idx[sl]]],
                           axis=2)
        beta, se, s2, it_, cv = fit_mixed(Zt, tsel[sl], Y)
        convs.append(cv)
        its.append(it_)
        for j, f in enumerate(FEATURES):
            z = beta[:, 2 + j] / se[:, 2 + j]
            Z_rej[f].append(stats.norm.sf(z) <= ALPHA)
            T_rej[f].append(stats.t.sf(z, N - 1) <= ALPHA)
    cv = np.concatenate(convs)
    rows = []
    for f in FEATURES:
        mz, mz_se = rate_se(np.concatenate(Z_rej[f]), sb)
        mxz = np.concatenate(Z_rej[f]).astype(float)
        gain_t = rate_se(mxz - r[("t", f)], sb)
        gain_w = rate_se(mxz - r[("wilcoxon", f)], sb)
        rows.append({
            "size": size, "effect": eff, "nuisances": name,
            "N": N, "feature": FEATURE_LABEL[f],
            "mixed_z": mz, "mixed_z_se": mz_se,
            "mixed_t": float(np.mean(np.concatenate(T_rej[f]))),
            "paired_t": float(np.mean(r[("t", f)])),
            "wilcoxon": float(np.mean(r[("wilcoxon", f)])),
            "gain_over_t": gain_t[0], "gain_over_t_se": gain_t[1],
            "gain_over_wilcoxon": gain_w[0],
            "gain_over_wilcoxon_se": gain_w[1],
            "converged": float(cv.mean()),
            "iterations_max": int(np.concatenate(its).max()),
            "seconds": time.time() - t0})
    return rows


TASKS = {"pool": task_pool, "replicate": task_replicate,
         "sanity": task_sanity, "mixed": task_mixed,
         "side_repro": task_side_repro, "thresholds": task_thresholds}


def _run_task(kind, args):
    return TASKS[kind](*args)


def run_tasks(jobs, n_workers, state):
    """jobs: list of (kind, args). Results in the order of jobs. The long
    mixed-model jobs are submitted first; the order of the results does not
    depend on it."""
    if n_workers <= 1:
        _worker_init(state, quiet=False)     # this process (fixes script 7,
        return [_run_task(k, a) for k, a in jobs]    # which needed > 1)
    order = sorted(range(len(jobs)), key=lambda i: jobs[i][0] != "mixed")
    ctx = mp.get_context("spawn")
    res = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=min(n_workers, len(jobs)),
                             mp_context=ctx, initializer=_worker_init,
                             initargs=(state,)) as ex:
        futs = {i: ex.submit(_run_task, *jobs[i]) for i in order}
        done = 0
        for i in order:
            res[i] = futs[i].result()
            done += 1
            if done % 25 == 0 or done == len(jobs):
                log(f"  {done} of {len(jobs)} tasks done")
    return res


def same_rows(a, b):
    """Two task results identical (exact)."""
    for x, y in zip(a, b):
        if isinstance(x, dict):
            if pd.Series(x).to_dict() != pd.Series(y).to_dict():
                return False
        elif not pd.DataFrame(x).equals(pd.DataFrame(y)):
            return False
    return True


def rescale_table(v4, explore, pk, ck):
    """Script 2's rescale_factors on the frozen lists (mean of the two
    sets' typical informative trial), per size."""
    rows, resc = [], {}
    for size in BLOCK_SIZES:
        g = pd.DataFrame([explore.choice_probability(
            v4, pk[size]["lists"][f"items_{b}"]["X"]) for b in SETS])
        r = explore.rescale_factors(v4, g)
        resc[size] = r
        rows.append({"size": size, **r})
    df = pd.DataFrame(rows)
    ok = (np.allclose(df["p_rescaled_sat"], explore.V4_CHOICE_PROB[0],
                      atol=1e-9)
          and np.allclose(df["p_rescaled_con"], explore.V4_CHOICE_PROB[1],
                          atol=1e-9))
    ck.add("rescaled weights", "the rescaled weights give v4's choice "
           "probabilities on the lists' typical informative trial",
           ok, "; ".join(f"{r.size}: k_sat {r.k_sat:.3f}, k_con "
                         f"{r.k_con:.3f}" for r in df.itertuples()))
    return df, resc


def mixed_summary(mx, rule, ck):
    ck.add("mixed model", "every mixed-model fit converged",
           (mx["converged"] == 1).all(),
           f"lowest share converged {mx['converged'].min():.4f}")
    se0m = np.sqrt(ALPHA * (1 - ALPHA) / MIXED_STUDIES)
    nulm = mx[mx["effect"] == "no effect"]
    lim_m = ALPHA + RULE_SE_MULT * np.maximum(nulm["mixed_z_se"], se0m)
    fp_ok = bool((nulm["mixed_z"] <= lim_m).all())
    lim_m = float(lim_m.max())
    hs = mx[(mx["effect"] == "half shift") & (mx["N"] == MIXED_GAIN_N)]
    against = [rule["chosen"]] if rule["chosen"] else list(TESTS)
    verdict = [
        f"False positives (Wald z): largest {nulm['mixed_z'].max():.4f} "
        f"against a limit of up to {lim_m:.4f} -> "
        f"{'controlled' if fp_ok else 'NOT controlled'}."]
    if not rule["chosen"]:
        verdict.append("No primary test was chosen by the rule, so the "
                       "mixed model is compared with both.")
    worth_all = []
    for test in against:
        gains = hs["gain_over_t" if test == "t"
                   else "gain_over_wilcoxon"].to_numpy()
        worth = fp_ok and bool(np.all(gains >= MIXED_GAIN))
        worth_all.append(worth)
        verdict.append(
            f"Power gain over the {test} test under half shift at N = "
            f"{MIXED_GAIN_N} (size x feature): "
            f"{', '.join(f'{g:+.3f}' for g in gains)}; needed >= "
            f"{MIXED_GAIN} in every one -> "
            f"{'worth registering' if worth else 'not worth registering'}.")
    verdict.append("VERDICT: the mixed model is "
                   + ("WORTH REGISTERING." if all(worth_all) else
                      "NOT worth registering by this criterion."))
    return verdict


# ------------------------------------------------------------
# Figures
# ------------------------------------------------------------

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
             "#56B4E9", "#000000"]


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)


def figures(pw, bm_means, bm, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    TEST_LABEL = {"t": "paired t-test", "wilcoxon": "Wilcoxon signed-rank"}
    # 1. power against N
    fig, axes = plt.subplots(len(BLOCK_SIZES), 2, figsize=(10, 4 *
                             len(BLOCK_SIZES)), sharey=True, squeeze=False)
    series = [("planning", "power", 0), ("half shift", "power", 1),
              ("planning", "power (no nuisances)", 2),
              ("half shift", "power (no nuisances)", 4),
              ("no effect", "null (rule)", 6)]
    for i, size in enumerate(BLOCK_SIZES):
        for j, f in enumerate(("con", "sat")):
            ax = axes[i, j]
            for eff, role, c in series:
                for test, ls in (("t", "-"), ("wilcoxon", "--")):
                    q = pw[(pw["size"] == size) & (pw["effect"] == eff)
                           & (pw["role"] == role) & (pw["test"] == test)
                           & (pw["feature"] == FEATURE_LABEL[f])]
                    if eff == "no effect":
                        q = q[q["scenario"] == "all nuisances (defaults)"]
                    if q.empty:
                        continue
                    lab = (f"{eff}, "
                           f"{'nuisances at defaults' if role != 'power (no nuisances)' else 'no nuisances'}"
                           f", {TEST_LABEL[test]}")
                    ax.plot(q["N"], q["rate"], ls, color=OKABE_ITO[c],
                            marker="o", ms=3, label=lab)
            ax.axhline(ALPHA, color="#888888", lw=0.8, ls=":")
            ax.set_title(f"{FEATURE_LABEL[f].capitalize()}, {size} trials per "
                         "block", fontsize=10)
            ax.set_ylim(0, 1)
            _style(ax)
            ax.set_xlabel("Report-group participants (analysable)")
            if j == 0:
                ax.set_ylabel("Probability the one-sided paired\ntest is "
                              "significant")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.suptitle("Primary test: derealization-block weight minus "
                 "ordinary-block weight\n(responder shares, shifts and "
                 "nuisances are planning assumptions)", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "power_vs_n.png"), dpi=FIG_DPI)
    plt.close(fig)

    # 2. false positives across scenarios at N = 15
    nn = 15 if 15 in N_GRID else N_GRID[len(N_GRID) // 2]
    q = pw[(pw["effect"] == "no effect") & (pw["N"] == nn)
           & (pw["lists"] == "v5b")
           & ~pw["role"].str.contains("reverse")]
    names = list(dict.fromkeys(q["scenario"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(names) + 2),
                             sharey=True)
    for j, f in enumerate(("con", "sat")):
        ax = axes[j]
        for k_, (size, test) in enumerate([(s, t) for s in BLOCK_SIZES
                                           for t in TESTS]):
            r = q[(q["size"] == size) & (q["test"] == test)
                  & (q["feature"] == FEATURE_LABEL[f])].set_index(
                "scenario").reindex(names)
            y = np.arange(len(names)) + (k_ - 1.5) * 0.15
            ax.plot(r["rate"], y, "o", ms=3.5, color=OKABE_ITO[k_],
                    label=f"{size} trials, {TEST_LABEL[test]}")
        ax.axvline(ALPHA, color="#888888", lw=0.8)
        ax.set_title(f"{FEATURE_LABEL[f].capitalize()} (N = {nn})",
                     fontsize=10)
        ax.set_xlabel("Share of studies significant with no effect")
        _style(ax)
    axes[0].set_yticks(np.arange(len(names)))
    axes[0].set_yticklabels(names, fontsize=7)
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle("False positives with no effect, by nuisance (rows from "
                 "'derealization lapse' on are threats no test can "
                 "remove)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "false_positives.png"), dpi=FIG_DPI)
    plt.close(fig)

    # 3. per-block measures picking up an episode lapse difference
    fig, axes = plt.subplots(2, len(BLOCK_SIZES), figsize=(10, 7.2),
                             sharey=True, sharex=True, squeeze=False)
    variants = [("", "no pair-specific preferences"),
                (f", with pair-specific preferences SD {PAIR_SD_FOR_ROWS:g}",
                 f"pair-specific preferences SD {PAIR_SD_FOR_ROWS:g}")]
    for row, (suffix, title) in enumerate(variants):
        sel = [("no nuisances", "no lapse difference", 0)] if not suffix \
            else []
        sel += [(f"derealization lapse +{el:g}{suffix}",
                 f"derealization lapse +{el:g}", c + 1)
                for c, el in enumerate(EPISODE_LAPSES)]
        for i, size in enumerate(BLOCK_SIZES):
            ax = axes[row, i]
            for name, lab, c in sel:
                r = bm[(bm["size"] == size) & (bm["scenario"] == name)
                       & (bm["effect"] == "no effect")]
                if r.empty:
                    continue
                ax.plot(r["N"], r["consistency_flags"], "-",
                        color=OKABE_ITO[c], marker="o", ms=3,
                        label=f"{lab}: consistency")
                ax.plot(r["N"], r["catch_flags"], "--", color=OKABE_ITO[c],
                        marker="s", ms=3, label=f"{lab}: catch trials")
            ax.axhline(ALPHA, color="#888888", lw=0.8, ls=":")
            ax.set_title(f"{size} trials per block, {title}", fontsize=9)
            if row == 1:
                ax.set_xlabel("Report-group participants (analysable)")
            _style(ax)
        axes[row, 0].set_ylabel("Share of studies flagging more\nlapses "
                                "in the derealization block")
        axes[row, -1].legend(fontsize=6, frameon=False)
    fig.suptitle("Do per-block consistency and catch failures pick up a "
                 "lapse difference? (lapse values are assumptions)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "block_measures.png"), dpi=FIG_DPI)
    plt.close(fig)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

def tab(df, nd=3):
    return "  " + df.to_string(index=False, float_format=lambda x:
                               f"{x:.{nd}f}").replace("\n", "\n  ")


def wide(pw, rows_filter, n_list, by=("scenario",)):
    q = pw[rows_filter & pw["N"].isin(n_list)].copy()
    q["col"] = (q["test"].map({"t": "t", "wilcoxon": "W"}) + ":"
                + q["feature"].str[:3] + ":N" + q["N"].astype(str))
    piv = q.pivot_table(index=list(by), columns="col", values="rate",
                        sort=False)
    cols = [f"{t}:{f}:N{n}" for f in ("con", "sat") for t in ("t", "W")
            for n in n_list]
    return piv[[c for c in cols if c in piv.columns]]




def fig_episode(pwb, out):
    """The episode-dependent strategy: false positives against k at N = 20
    (paired t-test): this run's model on the v5b lists, the side model on
    the v5b lists (script 7) and on the v5 lists."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nn = max(TABLE_N)
    q = pwb[pwb["role"].isin(["episode strategy",
                              "episode strategy [v5 lists]"])
            & (pwb["N"] == nn) & (pwb["test"] == "t")].copy()
    q["k"] = q["scenario"].str.extract(r"k ([0-9.]+)").astype(float)
    q["definition"] = q["scenario"].str.extract(r"\((corpus|orig),")
    fig, axes = plt.subplots(len(BLOCK_SIZES), 2, figsize=(10, 4 *
                             len(BLOCK_SIZES)), sharey=True, squeeze=False)
    series = [("v5b", PERSON_MODEL, "-"), ("v5b", SIDE_LABEL, ":"),
              ("v5", "side", "--")]
    for i, size in enumerate(BLOCK_SIZES):
        for j, f in enumerate(("con", "sat")):
            ax = axes[i, j]
            for lists, model, ls in series:
                for dn, c in (("corpus", 0), ("orig", 1)):
                    r = q[(q["size"] == size) & (q["lists"] == lists)
                          & (q["model"] == model) & (q["definition"] == dn)
                          & (q["feature"] == FEATURE_LABEL[f])
                          ].sort_values("k")
                    if r.empty:
                        continue
                    ax.plot(r["k"], r["rate"], ls, color=OKABE_ITO[c],
                            marker="o", ms=3,
                            label=f"{lists} lists, {model} model, "
                                  f"{STRAT_LABEL[dn]}")
            ax.axhline(ALPHA, color="#888888", lw=0.8, ls=":")
            ax.set_title(f"{FEATURE_LABEL[f].capitalize()}, {size} trials "
                         "per block", fontsize=10)
            ax.set_xlabel("k (derealization -k x N, ordinary +k x N)")
            ax.set_ylim(0, 1)
            _style(ax)
            if j == 0:
                ax.set_ylabel(f"Share of studies significant\n(paired "
                              f"t-test, N = {nn}, no effect)")
    axes[0, -1].legend(fontsize=6, frameon=False)
    fig.suptitle(f"Episode-dependent strategy: false positives, {PERSON_MODEL}"
                 " model and side model\n(k values are assumptions)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "episode_strategy.png"), dpi=FIG_DPI)
    plt.close(fig)


def episode_table(pwb, leak):
    """The episode-dependent strategy rows, every model shown: rates at
    TABLE_N (paired t-test and Wilcoxon), the mean difference, and the
    shift the leak predicts for a model without N terms, (k_D - k_O) x b."""
    q = pwb[pwb["role"].str.startswith("episode strategy")
            & pwb["N"].isin(TABLE_N)].copy()
    q["k"] = q["scenario"].str.extract(r"k ([0-9.]+)").astype(float)
    q["definition"] = q["scenario"].str.extract(r"\((corpus|orig),")
    q["direction"] = np.where(q["role"].str.contains("reverse"), "reverse",
                              "main")
    lk = pd.DataFrame(leak).groupby(["lists", "size", "definition"])[
        ["b_sat", "b_con"]].mean().reset_index()
    q = q.merge(lk, on=["lists", "size", "definition"], how="left")
    sgn = np.where(q["direction"] == "main", -2.0, 2.0)
    q["leak_shift"] = np.where(q["feature"] == "saturation",
                               sgn * q["k"] * q["b_sat"],
                               sgn * q["k"] * q["b_con"])
    q["col"] = (q["test"].map({"t": "t", "wilcoxon": "W"}) + ":N"
                + q["N"].astype(str))
    idx = ["lists", "size", "direction", "definition", "k", "feature",
           "model"]
    piv = q.pivot_table(index=idx, columns="col", values="rate", sort=False)
    md = q[q["N"] == TABLE_N[0]].groupby(idx, sort=False)[
        ["mean_diff", "leak_shift"]].first()
    piv = piv.join(md)
    cols = [f"{t}:N{n}" for t in ("t", "W") for n in TABLE_N]
    out = piv[[c for c in cols if c in piv.columns]
              + ["mean_diff", "leak_shift"]].reset_index()
    rank = {PERSON_MODEL: 0, SIDE_LABEL: 1, "side": 2}
    out["_r"] = out["model"].map(rank)
    out["_l"] = (out["lists"] != "v5b").astype(int)
    return out.sort_values(["_l", "size", "direction", "definition", "k",
                            "feature", "_r"], kind="stable").drop(
        columns=["_r", "_l"]).reset_index(drop=True)


def plain_summary(ck, pw, pwb, rule, resc_df, ept, hist, TH):
    L = []
    chosen = rule["chosen"]
    L.append(f"* Checks: {len(ck.rows) - ck.n_fail} of {len(ck.rows)} pass"
             + ("." if ck.n_fail == 0 else f"; {ck.n_fail} FAIL (see "
                "checks.csv)."))
    L += hist["summary_lines"]
    L.append(f"* {PERSON_MODEL} model thresholds (2.5% per tail, script 6's "
             "20,000 random choosers; sign test only): "
             + "; ".join(f"{s} {b}: sat {v['upper'][0]:+.3f}/"
                         f"{v['lower'][0]:+.3f}, con {v['upper'][1]:+.3f}/"
                         f"{v['lower'][1]:+.3f}" for (s, b), v in TH.items())
             + ".")
    t = rule["tests"]
    L.append(f"* Primary test (v5's amended rule, at most "
             f"{rule['chance_95']} of {rule['cells']} cells over the limit): "
             f"{chosen or 'NONE QUALIFIES'}. Cells over the limit: "
             + ", ".join(f"{k} {v['n_exceed']}" for k, v in t.items())
             + f". Under the original rule (no cell): "
             f"{rule['chosen_original_rule'] or 'none qualifies'}.")
    test = chosen or "t"
    side_on = SIDE_LABEL in set(pwb["model"])

    def r(e, role, size, f, model, N=20):
        q = pwb[(pwb["role"] == role) & (pwb["effect"] == e)
                & (pwb["size"] == size) & (pwb["test"] == test)
                & (pwb["feature"] == f) & (pwb["N"] == N)
                & (pwb["model"] == model) & (pwb["lists"] == "v5b")]
        return float(q["rate"].iloc[0]) if len(q) else np.nan
    for eff in ("planning", "half shift"):
        parts = []
        for size in BLOCK_SIZES:
            for f in ("contrast", "saturation"):
                a = r(eff, "power", size, f, PERSON_MODEL)
                b = r(eff + RESCALED_SUFFIX, "power (rescaled)", size, f,
                      PERSON_MODEL)
                sd = (f"; side {r(eff, 'power', size, f, SIDE_LABEL):.2f}"
                      if side_on else "")
                parts.append(f"{size} {f} {a:.2f} (rescaled {b:.2f}{sd})")
        L.append(f"* Power at N = 20, {eff}, all nuisances at defaults "
                 f"({PERSON_MODEL} model, {test} test; ASSUMPTIONS): "
                 + "; ".join(parts) + ".")
    for model in [PERSON_MODEL] + ([SIDE_LABEL] if side_on else []):
        nul = pwb[(pwb["role"] == "null (rule)") & (pwb["test"] == test)
                  & (pwb["model"] == model)]
        L.append(f"* False positives with no effect over the rule rows "
                 f"(model '{model}', {test} test): mean "
                 f"{nul['rate'].mean():.3f}, largest {nul['rate'].max():.3f}.")
    col = f"t:N{max(TABLE_N)}"
    e = ept[ept["direction"].isin(["main", "reverse"])]
    for lists, model in (("v5b", PERSON_MODEL), ("v5b", SIDE_LABEL),
                         ("v5", "side")):
        q = e[(e["lists"] == lists) & (e["model"] == model)]
        if q.empty or col not in q:
            continue
        g = q.groupby("k")[col].max()
        L.append(f"* Episode-dependent strategy, {lists} lists, model "
                 f"'{model}' (both definitions and directions, paired t, N = "
                 f"{max(TABLE_N)}): largest false-positive rate "
                 + ", ".join(f"{v:.3f} at k {k:g}" for k, v in g.items())
                 + ".")
    L.append("* Mixed model: not rerun; " + hist["mixed_short"])
    L.append("* Rescale factors (script 2's rescale_factors on the lists): "
             + "; ".join(f"{r_.size}: saturation x{r_.k_sat:.3f}, contrast "
                         f"x{r_.k_con:.3f}" for r_ in resc_df.itertuples())
             + ".")
    return L


def model_history(meta6, meta7, td7b, r7b):
    """The numbers behind the model choice (script 6, script 7, script 7b)
    and the report lines that record it."""
    loss = (td7b.pivot_table(index=["size", "set", "feature"],
                             columns="model", values="detection")
            .eval("side - adjusted"))
    q = r7b[(r7b["N"] == 20) & (r7b["test"] == "t")]
    ep = q[q["role"].str.startswith("episode strategy")].copy()
    ep["k"] = ep["scenario"].str.extract(r"k ([0-9.]+)").astype(float)
    epk = ep.groupby(["model", "k"])["rate"].max().unstack(0)
    pw_ = q[q["role"].isin(["power", "power (rescaled)"])].pivot_table(
        index=["size", "effect", "feature"], columns="model", values="rate")
    ploss = float((pw_["side"] - pw_["adjusted"]).max())
    rows = ([{"quantity": "detection loss (side - adjusted), planning "
              "weights", "where": f"{a} {b}, {c}", "value": v}
             for (a, b, c), v in loss.items()]
            + [{"quantity": f"largest false-positive rate, episode-dependent "
                f"strategy, k {k:g}, N 20, paired t", "where": m,
                "value": float(epk.loc[k, m])}
               for k in epk.index for m in epk.columns]
            + [{"quantity": "largest power loss (side - adjusted), N 20, "
                "paired t, headline cells", "where": "", "value": ploss}])
    ks = list(epk.index)
    side_fp = " / ".join(f"{epk.loc[k, 'side']:.3f}" for k in ks)
    adj_fp = float(epk["adjusted"].max())
    kk = " / ".join(f"{k:g}" for k in ks)
    choice = [
        "MODEL CHOICE. The registered per-person model is the strategy-"
        "adjusted model: the side",
        "model plus two nuisance terms per block, N_corpus and N_orig "
        "(Firth, no intercept).",
        "The side and N terms are nuisance terms; their estimates are never "
        "results. The side",
        "model is reported as a sensitivity analysis.",
        "History: script 6 chose the side model over the two-weight model "
        "by its rule, by a",
        f"margin of {abs(meta6['rule']['shift'] - 0.05):.4f} "
        f"({meta6['rule']['shift_where']}, shift "
        f"{meta6['rule']['shift']:.4f} against 0.05;",
        "caused by that set's side-predictor correlation; at 180 trials "
        "alone the rule would choose",
        "the two-weight model). Script 7 (side model) found that the "
        "episode-dependent strategy",
        "inflated false positives; script 7b compared the side model with "
        "the adjusted model on",
        "the same simulated people. The adjusted model was chosen AFTER "
        "these simulations and",
        "BEFORE any data were collected, because the episode-dependent "
        "strategy inflated the side",
        f"model's false-positive rate to {side_fp} (k = {kk}; N = 20, paired "
        "t, largest over both",
        f"definitions and directions) and the adjusted model's to at most "
        f"{adj_fp:.3f}. The cost was",
        f"at most {float(loss.max()):.3f} in detection at the planning "
        f"weights and {ploss:.3f} in power at N = 20.",
        "The N terms only remove strategies that act linearly on N_corpus "
        "or N_orig (in the",
        "logit); a strategy of another form is not removed by them. The "
        "lists were not re-frozen.",
    ]
    mixed = meta7.get("mixed_verdict") or []
    mixed_lines = (["Not rerun here. The mixed model was assessed with the "
                    "side model in script 7's run",
                    f"({PAIRED_RUN_DIR}):"]
                   + ["  " + x for x in mixed]
                   + ["It has no N terms; it is not part of the registration."])
    mixed_short = ("assessed with the side model in script 7's run: "
                   + (mixed[-1] if mixed else "verdict not found") + "")
    summary = [
        "* Registered per-person model: the strategy-adjusted model (side "
        "model + N_corpus + N_orig per block, Firth, no intercept); side and "
        "N terms are nuisance terms, never results; the side model is a "
        "sensitivity analysis.",
        "* Chosen after the simulations and before any data: the episode-"
        f"dependent strategy inflated the side model's false-positive rate "
        f"to {side_fp} (k = {kk}) and the adjusted model's to at most "
        f"{adj_fp:.3f}; the cost was at most {float(loss.max()):.3f} in "
        f"detection and {ploss:.3f} in power at N = 20 (script 7b). The N "
        "terms only remove strategies that act linearly on N_corpus or "
        "N_orig.",
        f"* Script 6 had chosen the side model (over the two-weight model) "
        f"by a margin of {abs(meta6['rule']['shift'] - 0.05):.4f}.",
    ]
    return {"rows": pd.DataFrame(rows), "choice": choice,
            "mixed_lines": mixed_lines, "mixed_short": mixed_short,
            "summary_lines": summary, "side_fp": side_fp, "adj_fp": adj_fp,
            "loss": float(loss.max()), "ploss": ploss}




def write_report(out, ck, scen_df, pw, sg, bm, bmm, mx, san, rep, rule,
                 mixed_verdict, meta, leak_df, fat_df, resc_df, ept, pwb,
                 hist):
    L = []
    w = L.append
    w("THE PRIMARY PAIRED TEST  (simulate_power_paired_v5b_adj.py, v5b "
      f"script 7, {PERSON_MODEL} model)")
    w("=" * 78)
    w(f"run       {meta['started']}  ({meta['minutes']:.1f} min, "
      f"{meta['workers']} worker process(es), one thread each)")
    w(f"output    {out}")
    w(f"python    {meta['python']}, numpy {meta['numpy']}, scipy "
      f"{meta['scipy']}, pandas {meta['pandas']}")
    w("packages  " + ", ".join(PACKAGES[s] for s in BLOCK_SIZES)
      + "; comparison: " + ", ".join(V5_PACKAGES[s] for s in BLOCK_SIZES))
    w("")
    w("0. IN PLAIN WORDS")
    w("-" * 78)
    for line in meta["summary"]:
        w(line)
    w("")
    w("PLANNING ASSUMPTIONS ARE ASSUMPTIONS, NOT ESTIMATES. No image-choice "
      "data exist.")
    w("The responder shares and shifts, taste spread, lapse rates and every "
      "nuisance")
    w("value below were chosen for planning; every power and false-positive "
      "rate here")
    w("depends on them.")
    w("")
    w("1. WHAT WAS DONE")
    w("-" * 78)
    for line in hist["model_lines"]:
        w(line)
    w("Primary test per feature: one-sided, derealization-block weight minus "
      "ordinary-")
    w(f"block weight, alpha {ALPHA}, uncorrected; paired t-test and exact "
      "Wilcoxon.")
    w(f"Pools of {POOL_PEOPLE:,} people per scenario and size, each fitted "
      f"on both blocks; {N_STUDIES:,}")
    w("studies per cell drawn from the pool. Every rate's Monte Carlo SE is "
      "measured from")
    w(f"{N_BATCHES} disjoint sub-pools (it includes the pool's own error): "
      f"largest power SE {pw['se'].max():.4f},")
    w(f"largest SE of a no-effect rate "
      f"{pw[pw['effect'] == 'no effect']['se'].max():.4f} (columns 'se' in "
      "the CSV files).")
    w("Recruitment: each participant takes the next slot of the frozen "
      "assignment table;")
    w(f"anyone with {CATCH_FAIL_EXCLUDE}+ catch failures of "
      f"{2 * CATCH_PER_BLOCK} is excluded and uses up the slot.")
    w(f"N = {N_GRID[0]}-{N_GRID[-1]} analysable people; tables show N = "
      f"{', '.join(map(str, TABLE_N))}, paired_power.csv has every N.")
    w("The person model is in the script's docstring. Values:")
    w(f"  taste SD {TASTE_SD}; personal lapse Gamma, mean {LAPSE_MEAN}, SD "
      f"{LAPSE_BETWEEN_SD} x mean;")
    w(f"  catch p(fail) = {CATCH_P_FAIL_PER_LAPSE} x lapse, "
      f"{CATCH_PER_BLOCK} per block.")
    w("  Nuisances at their defaults [ASSUMPTIONS]:")
    for k_, v in NUISANCE_DEFAULT.items():
        if v != NUISANCE_OFF[k_]:
            w(f"    {k_:18s} {v}")
    w("  (scene SD = scene_sd_mult x taste SD; dull in per-SD weight units; "
      "side in logit;")
    w("  strategies: k per unit of N, N in corpus SD; fatigue per 100 main "
      "trials)")
    w("")
    w("  STRATEGIES (replace v5's 'edited looks odd'). 'More typical': "
      "+k x N_corpus;")
    w("  'less edited': +k x N_orig; N = dist(B) - dist(A), so +k favours the "
      "image nearer")
    w("  the corpus median or the scene's original. Separate quarters at "
      "the defaults.")
    w("")
    w("  FATIGUE (replaces v5's 'block order'). Per 100 main trials of the "
      f"session: lapse +{FATIGUE_LAPSE_PER_100},")
    w(f"  preferences x{FATIGUE_SCALE_PER_100}, anchored at block 1's mean "
      "position. Calibrated so that at 150")
    w("  trials block 2 has v5's 'block order' values (lapse +0.05, "
      "preferences x0.90). Block means:")
    w(tab(fat_df, 4))
    w("")
    w("  RESCALED WEIGHTS: script 2's rescale_factors on the frozen lists "
      "(the factors k")
    w("  multiply the effect parts of planning and half shift, so the "
      "typical informative")
    w("  trial gives v4's choice probabilities 0.6191 / 0.6104):")
    w(tab(resc_df[["size", "median_sat", "k_sat", "median_con", "k_con",
                   "p_fixed_sat", "p_fixed_con"]], 4))
    w("")
    w("  THE V5 LISTS (comparison rows only): their N is computed from "
      "their manifest.csv")
    w("  (each image's sat_mean and contrast_local); the scenes' originals "
      "are not in the v5")
    w("  lists, so their features come from the v5b stimulus set table "
      f"({STIMULUS_SET_DIR}),")
    w("  the same 47 scenes, measured on the stimulus path photograph -> "
      "512 px -> 256 px.")
    w("  The same computation reproduces the v5b lists' N (check in section "
      "2).")
    w("")
    w("2. CHECKS")
    w("-" * 78)
    for r in ck.rows:
        w(f"  {r['status']}  [{r['section']}] {r['check']}")
        if r["detail"]:
            w(f"          {r['detail']}")
    w(f"\n  {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL")
    w("")
    w("3. SANITY CHECK: explore_block_length_v5b's simplified model on the "
      "frozen lists")
    w("-" * 78)
    w("explore_functions: explore's paired_pool / study_index / paired_tests "
      "run")
    w(f"unmodified (its pool and study settings set to {POOL_PEOPLE:,} and "
      f"{N_STUDIES:,}); this_script: this script's simulation")
    w("set to the same model (two-weight fit, lapse 0.1 for everyone, taste "
      "the same in")
    w(f"both blocks, no exclusion). draws: explore's range over its draws "
      f"({EXPLORE_MIX}, {EXPLORE_TSET}, fixed weights).")
    w(tab(san[["size", "scenario", "N", "test", "feature",
                "explore_functions", "this_script", "se_each", "z",
                "draws_min", "draws_max"]]))
    w("")
    w("4. FALSE POSITIVES WITH NO EFFECT")
    w("-" * 78)
    w("Share of studies where the one-sided paired test is significant. "
      "t = paired")
    w("t-test, W = Wilcoxon; con = contrast (H1), sat = saturation (H2). "
      "Monte Carlo SE about")
    w(f"{pw[pw['role'] == 'null (rule)']['se'].mean():.4f}. Rule limit per "
      f"cell {ALPHA} + {RULE_SE_MULT:g} SE: "
      f"{rule['limit_range'][0]:.4f}-{rule['limit_range'][1]:.4f}.")
    for role, title in (("null (rule)", "Rows used by the test-choice rule "
                         "(each default nuisance alone, then all)"),
                        ("null (sensitivity)", "Sensitivity rows (the ranges"
                         "; not used by the rule)"),
                        ("threat", "THREATS NO TEST CAN REMOVE (more lapses "
                         "in the derealization block,\nalone and with a "
                         "shared bias; a bias that differs between episodes)"
                         )):
        w(f"\n  {title}")
        for size in BLOCK_SIZES:
            t_ = wide(pwb, (pwb["role"] == role) & (pwb["size"] == size),
                      TABLE_N, by=("scenario", "model"))
            w(f"\n  {size} trials per block")
            w("  " + t_.to_string(float_format=lambda x: f"{x:.3f}")
              .replace("\n", "\n  "))
    w("\n  Why the threat rows are threats: a bias present in both blocks "
      "cancels in the")
    w("  difference only if both blocks express it equally. More lapses in "
      "the")
    w("  derealization block shrink its weights towards zero: a shared "
      "preference for the")
    w("  predicted image ('duller') then looks like a negative difference; a "
      "strategy acts")
    w("  on the weights only through the lists' leak, so its interaction "
      "with lapses is")
    w("  small. A bias stronger in one episode is indistinguishable from an "
      "effect.")
    w("  Section 11 shows how well the per-block attention measures pick up "
      "a lapse difference.")
    w("")
    w("5. THE EPISODE-DEPENDENT STRATEGY: v5b LISTS AGAINST v5 LISTS "
      "(a threat no test can remove)")
    w("-" * 78)
    w("Everyone uses a strategy that differs between the episodes: main "
      "rows, -k x N in the")
    w("derealization block and +k x N in the ordinary block (picks the less "
      "typical / more")
    w("edited image when derealized); reverse rows the other way round "
      "(information). A")
    w("strategy reaches a model WITHOUT N terms only through the leak b (N "
      "regressed on the")
    w("predictors, no intercept): the weight difference it creates is about "
      "(k_D - k_O) x b,")
    w("before logistic attenuation and lapses ('leak_shift'; 'mean_diff' is "
      "the simulated mean")
    w("difference of the estimates). v5b's lists were built with |b| <= "
      "0.05; v5's were not.")
    w(f"model: {PERSON_MODEL} (this run, v5b lists); '{SIDE_LABEL}' = the "
      "side model's rates read")
    w("from script 7's run (sensitivity); the v5 lists use the side model "
      "(rerun here, equal to script 7).")
    w("  The adjusted model's N terms take up a strategy that is linear in "
      "N, so its mean_diff")
    w("  stays near 0.")
    w("\n  The leak (mean over the two sets is used for leak_shift):")
    w(tab(leak_df, 4))
    w("")
    w(tab(ept, 3))
    w("")
    w("6. POWER [responder shares and shifts: ASSUMPTIONS]")
    w("-" * 78)
    for role, title in (("power", "All nuisances at their defaults, fixed "
                         "weights (the headline)"),
                        ("power (rescaled)", "All nuisances at their "
                         "defaults, rescaled weights (cautious; beside the "
                         "headline)"),
                        ("power (no nuisances)", "No nuisances"),
                        ("power (sensitivity)", "All nuisances at their "
                         "defaults plus pair-specific preferences, or with "
                         "fatigue changed (sensitivity rows)")):
        w(f"\n  {title}")
        for size in BLOCK_SIZES:
            t_ = wide(pwb, (pwb["role"] == role) & (pwb["size"] == size),
                      TABLE_N, by=("scenario", "effect", "model"))
            w(f"\n  {size} trials per block")
            w("  " + t_.to_string(float_format=lambda x: f"{x:.3f}")
              .replace("\n", "\n  "))
    w("\n  Planning and half shift at every N (all nuisances at defaults; "
      "fixed and rescaled):")
    q = pw[pw["role"].isin(["power", "power (rescaled)"])
           & pw["effect"].str.match(r"^(planning|half shift)")]
    piv = q.pivot_table(index=["size", "effect", "N"],
                        columns=["test", "feature"], values="rate")
    w("  " + piv.to_string(float_format=lambda x: f"{x:.3f}")
      .replace("\n", "\n  "))
    w("")
    w("7. THE CHOICE OF PRIMARY TEST")
    w("-" * 78)
    w("The rule (v5's, as amended): a test qualifies if, over the rule rows "
      "(section 4),")
    w(f"at N = {', '.join(map(str, RULE_N))}, both sizes and both features, "
      f"no more of its cells exceed {ALPHA} + {RULE_SE_MULT:g} SE")
    w("(that cell's Monte Carlo SE) than chance gives in "
      f"{RULE_CHANCE_QUANTILE:.0%} of runs for a test whose false-")
    w(f"positive rate is exactly {ALPHA}. Among qualifying tests, the higher "
      "mean power over planning")
    w("and half shift (all nuisances at defaults, fixed weights, same N, "
      f"sizes, features) wins; within {RULE_TIE}")
    w(f"of each other, the {RULE_PREFER} test. The original rule (no cell "
      "over the limit) is shown beside it.")
    w(f"  Cells per test: {rule['cells']}; about "
      f"{rule['expected_exceed_if_exact']:.1f} expected over the limit by "
      f"chance, at most {rule['chance_95']} in "
      f"{RULE_CHANCE_QUANTILE:.0%} of runs.")
    for test, r in rule["tests"].items():
        w(f"  {test:9s} qualifies (amended): "
          f"{'yes' if r['qualifies_amended'] else 'NO'}; (original): "
          f"{'yes' if r['qualifies_original'] else 'no'}; cells over the "
          f"limit {r['n_exceed']}; worst {r['worst']:.4f} "
          f"({r['worst_where']});")
        w(f"            mean false-positive rate {r['mean_null']:.4f}; mean "
          f"power {r['power']:.4f}")
        for c in r["exceed_cells"]:
            w(f"            over the limit: {c}")
    if rule["chosen"] is None:
        w("\n  NO TEST QUALIFIES. The rule chooses nothing; see the cells "
          "over the limit.")
    else:
        extra = (f" (power gap {rule['gap']:+.4f})" if "gap" in rule
                 else "")
        w(f"\n  RECOMMENDED PRIMARY TEST: {rule['chosen']}{extra}.")
    w(f"  Under the original rule: {rule['chosen_original_rule'] or 'none'}.")
    w("")
    w("8. THE MIXED-EFFECTS LOGISTIC MODEL (alternative)")
    w("-" * 78)
    if mx is None or mx.empty:
        for line in hist["mixed_lines"]:
            w("  " + line)
    else:
        w(f"  {MIXED_STUDIES:,} studies per cell; the paired tests on the "
          "SAME studies beside it.")
        w("  z: one-sided Wald z; t(N-1): the same statistic against t with "
          "N-1 df.")
        w(tab(mx.drop(columns=["seconds"]), 3))
        w("")
        for line in mixed_verdict:
            w("  " + line)
    w("")
    w("9. SECONDARY: SIGN TEST ON DETECTED COUNTS WITHIN EACH BLOCK")
    w("-" * 78)
    w("Per block, each person's weight is classified with the per-set "
      "thresholds of")
    w(f"the {PERSON_MODEL} model (computed here from script 6's random "
      "choosers); the claim needs")
    w("d >= 5 detections and the exact one-sided")
    w("binomial (v4's Table 4 rule). Derealization block: the secondary "
      "H1/H2.")
    w("Ordinary block: the same rule, as a reference.")
    w("Note: within one block the sign test asks whether a directional "
      "preference exists;")
    w("it cannot tell a preference shared by both episodes (e.g. 'memory "
      "makes it duller',")
    w("one of the default nuisances) from a derealization effect, so with "
      "nuisances at")
    w("their defaults it claims in many studies with no derealization "
      "effect (second table).")
    w("That is why the paired test is primary.")
    for role in ("power", "null (rule)"):
        q = sg[(sg["role"] == role) & sg["N"].isin(TABLE_N)]
        if role == "null (rule)":
            q = q[q["scenario"].isin(["no nuisances",
                                      "all nuisances (defaults)"])]
        q = q.copy()
        q["col"] = (q["block"].str[:3] + ":" + q["feature"].str[:3] + ":N"
                    + q["N"].astype(str))
        piv = q.pivot_table(index=["size", "scenario", "effect"],
                            columns="col", values="claim_rate", sort=False)
        cols = [f"{b}:{f}:N{n}" for b in ("der", "ord") for f in ("con",
                                                                 "sat")
                for n in TABLE_N]
        w("")
        w("  " + piv[[c for c in cols if c in piv.columns]].to_string(
            float_format=lambda x: f"{x:.3f}").replace("\n", "\n  "))
    w("")
    w("10. MONTE CARLO CHECK: headline cells on two independent pools")
    w("-" * 78)
    w(tab(rep[rep["N"].isin(TABLE_N)][["size", "effect", "N", "test",
                                        "feature", "rate_1", "rate_2",
                                        "se_1", "z"]], 4))
    w("")
    w("11. PER-BLOCK ATTENTION MEASURES")
    w("-" * 78)
    w("Means over analysable people in each pool: lapse rate, catch failures "
      "(of 5),")
    w("consistency (share of repeats answered the same way), per block; and "
      "the share")
    w("excluded. Then, per study, how often each measure flags more lapses "
      "in the")
    w(f"derealization block (consistency: one-sided paired t; catch: "
      f"one-sided sign test; alpha {ALPHA}).")
    sel = bmm[(bmm["effect"] == "no effect") & (bmm["lists"] == "v5b")][[
        "size", "scenario", "excluded_share", "lapse_D", "lapse_O",
        "catch_D", "catch_O", "agree_D", "agree_O"]]
    w(tab(sel, 3))
    q = bm[(bm["effect"] == "no effect") & bm["N"].isin(TABLE_N)
           & (bm["lists"] == "v5b")].copy()
    piv = q.pivot_table(index=["size", "scenario"], columns="N",
                        values=["consistency_flags", "catch_flags"],
                        sort=False)
    w("")
    w("  " + piv.to_string(float_format=lambda x: f"{x:.3f}")
      .replace("\n", "\n  "))
    w("")
    w("12. FILES")
    w("-" * 78)
    for fn in meta["files"]:
        w(f"  {fn}")
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def read_run(d, files, label, ck, hashes):
    ok = all(os.path.exists(os.path.join(root_path(d), f)) for f in files)
    ck.add("inputs", f"{label} exists ({', '.join(files)})", ok, d)
    if not ok:
        raise SystemExit(f"missing files in {d}")
    for f in files:
        hashes[f"{d}/{f}"] = sha256_of(os.path.join(root_path(d), f))
    c = pd.read_csv(os.path.join(root_path(d), "checks.csv"))
    ck.add("inputs", f"{label} passed every check",
           (c["status"] == "PASS").all(),
           f"{(c['status'] == 'PASS').sum()} PASS, "
           f"{(c['status'] != 'PASS').sum()} FAIL")
    with open(os.path.join(root_path(d), "run_meta.json")) as f:
        return json.load(f)


def compare_exact(a, b, keys, col, what, ck):
    m = a.merge(b, on=keys, suffixes=("", "_ref"))
    d = (float(np.max(np.abs(m[col] - m[col + "_ref"]))) if len(m)
         else np.inf)
    ck.add("reproduction", what, len(m) == len(a) and len(a) > 0
           and d <= RATE_TOL, f"{len(m)} of {len(a)} rows matched; max "
           f"|difference| {d:.1e}")


def main():
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    ck = Checks()
    hashes = {}
    if RUN_MIXED_MODEL:
        raise SystemExit("RUN_MIXED_MODEL = True is not supported in this "
                         "version (see SETTINGS)")
    out = resolve_output_dir()
    log(f"output folder {out}")
    log(f"person model: {PERSON_MODEL}")
    n_workers = N_WORKERS or max(1, (os.cpu_count() or 2) - 1)
    log(f"{n_workers} worker process(es), one thread each")

    # --- the frozen packages first, the stimulus table, the v5 lists ---
    with open(os.path.join(root_path(PACKAGES[BLOCK_SIZES[0]]),
                           "FROZEN.json")) as f:
        frozen0 = json.load(f)
    stim = load_stimulus_table(frozen0, ck, hashes)
    pk = {size: load_package(size, ck, hashes, stim, "v5b")
          for size in BLOCK_SIZES}
    same = all(pk[s]["frozen"]["stimulus_set"] == frozen0["stimulus_set"]
               and {k: pk[s]["frozen"]["scripts_sha256"].get(k)
                    for k in (V4_SIM_SCRIPT, CAPACITY_SCRIPT)}
               == {k: frozen0["scripts_sha256"].get(k)
                   for k in (V4_SIM_SCRIPT, CAPACITY_SCRIPT)}
               for s in BLOCK_SIZES)
    ck.add("inputs", "the packages record the same stimulus set and the "
           "same imported scripts", same, frozen0["stimulus_set"]["run"])
    pk5 = {size: load_package(size, ck, hashes, stim, "v5")
           for size in BLOCK_SIZES}

    # --- the imported scripts ---
    v4 = load_module(V4_SIM_SCRIPT, ck, hashes, frozen0["scripts_sha256"])
    cap = load_module(CAPACITY_SCRIPT, ck, hashes, frozen0["scripts_sha256"])
    explore = load_module(EXPLORE_SCRIPT, ck, hashes)
    s6 = load_module(THRESHOLDS_SCRIPT, ck, hashes)
    fit = Fitter(v4, s6)
    TH_side, meta6 = load_thresholds(ck, hashes)
    p = root_path(EXPLORE_RESULTS_CSV)
    ok = os.path.exists(p)
    ck.add("inputs", "explore's paired_power.csv exists", ok, p)
    if not ok:
        raise SystemExit(f"missing {p}")
    hashes[EXPLORE_RESULTS_CSV] = sha256_of(p)
    ex_res = pd.read_csv(p)
    crit = v4.sign_test_critical(max(N_GRID))

    # --- script 7's and script 7b's runs ---
    meta7 = read_run(PAIRED_RUN_DIR, ["paired_power.csv",
                                      "secondary_sign.csv", "checks.csv",
                                      "run_meta.json"],
                     "script 7's run (side model)", ck, hashes)
    pw7 = pd.read_csv(os.path.join(root_path(PAIRED_RUN_DIR),
                                   "paired_power.csv"))
    sg7 = pd.read_csv(os.path.join(root_path(PAIRED_RUN_DIR),
                                   "secondary_sign.csv"))
    meta7b = read_run(ADJ_RUN_DIR, ["thresholds_detection.csv",
                                    "paired_rates.csv", "checks.csv",
                                    "run_meta.json"],
                      "script 7b's run (side against adjusted)", ck, hashes)
    td7b = pd.read_csv(os.path.join(root_path(ADJ_RUN_DIR),
                                    "thresholds_detection.csv"))
    r7b = pd.read_csv(os.path.join(root_path(ADJ_RUN_DIR),
                                   "paired_rates.csv"))
    t6 = f"{THRESHOLDS_RUN_DIR}/thresholds.csv"
    ck.add("inputs", "scripts 7 and 7b read the same script 6 thresholds as "
           "this run", meta7["input_sha256"].get(t6) == hashes.get(t6)
           and meta7b["input_sha256"].get(t6) == hashes.get(t6),
           f"{str(hashes.get(t6))[:16]}...")
    ck.add("inputs", "script 7b read script 7's run used here",
           meta7b["input_sha256"].get(f"{PAIRED_RUN_DIR}/paired_power.csv")
           == hashes.get(f"{PAIRED_RUN_DIR}/paired_power.csv"),
           PAIRED_RUN_DIR)
    hist = model_history(meta6, meta7, td7b, r7b)

    # --- tool checks ---
    rng = cap.Streams(RANDOM_SEED, STREAM_KEY)("tool_checks")
    worst = 0.0
    for n in (10, 15, 20):
        d = rng.normal(0.2, 1, (200, n))
        pv = explore.wilcoxon_greater(d)
        ref = np.array([stats.wilcoxon(r, alternative="greater",
                                       method="exact").pvalue for r in d])
        worst = max(worst, float(np.max(np.abs(pv - ref))))
    ck.add("tools", "explore's vectorised Wilcoxon equals SciPy's exact test",
           worst < 1e-12, f"max |difference| in p {worst:.1e}")
    d = rng.normal(0.2, 1, (200, 12))
    ref = stats.ttest_1samp(d, 0, axis=1, alternative="greater").pvalue
    ck.add("tools", "the paired t-test equals SciPy's", np.max(np.abs(
        t_greater(d) - ref)) < 1e-12, "")
    ck.add("tools", "sign-test critical values: none below 5 detections, "
           "5 of 5 at d = 5 (v4's Table 4 rule)",
           np.all(np.isinf(crit[:5])) and crit[5] == 5,
           f"d=5..10: {crit[5:11].tolist()}")
    it = pk[BLOCK_SIZES[-1]]["lists"]["items_A"]
    for model, Z in (("side", model_X(it["X"], it["side"], "side")),
                     ("adjusted", adjusted_design(it))):
        Y = (rng.random((30, len(Z))) < 0.5).astype(float)
        b1 = fit(Z, Y)
        b2 = np.array([v4.fit_firth(Z, Y[i]) for i in range(30)])
        ck.add("tools", f"the {model}-model fit equals v4's single-person "
               "fit_firth", np.max(np.abs(b1 - b2)) < 1e-9,
               f"max |difference| {np.max(np.abs(b1 - b2)):.1e}")

    resc_df, resc = rescale_table(v4, explore, pk, ck)
    fat_df = fatigue_calibration(pk, ck)
    leak_df = pd.DataFrame([r for s in BLOCK_SIZES
                            for r in pk[s]["leak"] + pk5[s]["leak"]])

    # --- phase 1: the model's thresholds ---
    state = {"pk": pk, "pk5": pk5, "TH": TH_side, "TH_side": TH_side,
             "resc": resc, "ex_res": ex_res}
    th_rows = []
    if PERSON_MODEL == "adjusted":
        jobs1 = [("thresholds", (size, b)) for size in BLOCK_SIZES
                 for b in SETS]
        log("phase 1: the adjusted model's thresholds")
        res1 = run_tasks(jobs1, n_workers, state)
        TH = {}
        worst, unconv = 0.0, 0.0
        for r in res1:
            TH[(r["size"], r["set"])] = {"upper": r["upper"],
                                         "lower": r["lower"]}
            unconv = max(unconv, r["unconverged"])
            for j, f in enumerate(FEATURES):
                q = td7b[(td7b["model"] == "adjusted")
                         & (td7b["size"] == r["size"])
                         & (td7b["set"] == r["set"])
                         & (td7b["feature"] == FEATURE_LABEL[f])].iloc[0]
                worst = max(worst, abs(q["upper"] - r["upper"][j]),
                            abs(q["lower"] - r["lower"][j]))
                th_rows.append({"size": r["size"], "set": r["set"],
                                "model": "adjusted",
                                "feature": FEATURE_LABEL[f],
                                "upper": r["upper"][j],
                                "lower": r["lower"][j],
                                "null_sd": r["null_sd"][j]})
        ck.add("thresholds", "every random chooser's adjusted-model fit "
               "converged", unconv == 0, f"unconverged share {unconv:.5f}")
        ck.add("thresholds", "the adjusted model's thresholds equal script "
               "7b's (same 20,000 random choosers)", worst <= THRESHOLD_TOL,
               f"max |difference| {worst:.1e}")
    else:
        TH = TH_side
    state["TH"] = TH
    th_df = pd.DataFrame(th_rows)

    # --- phase 2: every simulation, as tasks ---
    scen = build_scenarios()
    scen_df = pd.DataFrame([{"scenario": s["name"], "effect": s["effect"],
                             "role": s["role"], "lists": s["lists"],
                             "model": (PERSON_MODEL if s["lists"] == "v5b"
                                       else "side"),
                             **{k: str(v) for k, v in s["nuis"].items()}}
                            for s in scen])
    pos = {(s["name"], s["effect"]): i for i, s in enumerate(scen)}
    slice_i = []
    for size, name, eff in REPRO_SIDE_SLICE:
        ok = (name, eff) in pos and size in BLOCK_SIZES
        ck.add("reproduction", f"slice cell found: {size}, {name[:60]}, "
               f"{eff}", ok, "")
        if ok:
            slice_i.append((size, pos[(name, eff)]))
    jobs = ([("sanity", (size, sc)) for size in BLOCK_SIZES
             for sc in SANITY_SCENARIOS]
            + [("pool", (size, i)) for size in BLOCK_SIZES
               for i in range(len(scen))]
            + [("replicate", (size, eff)) for size in BLOCK_SIZES
               for eff in REPLICATE_EFFECTS])
    if PERSON_MODEL != "side":
        jobs += [("side_repro", a) for a in slice_i]
    log(f"phase 2: {len(jobs)} tasks ({len(scen)} scenarios x "
        f"{len(BLOCK_SIZES)} sizes, sanity, replicates, side-model slice)")
    res = run_tasks(jobs, n_workers, state)
    _worker_init(state, quiet=False)        # this process
    if CHECK_ONE_WORKER and n_workers > 1:
        i0 = [i for i, j in enumerate(jobs) if j[0] == "pool"][0]
        log(f"rerunning task {jobs[i0]} in this process (one worker)")
        again = task_pool(*jobs[i0][1])
        ck.add("workers", f"one worker gives the same result as {n_workers} "
               f"({jobs[i0][1][0]} trials, '{scen[jobs[i0][1][1]]['name']}', "
               f"{scen[jobs[i0][1][1]]['effect']})",
               same_rows(res[i0], again),
               "power, sign test, block measures and pool means compared")
    elif CHECK_ONE_WORKER:
        ck.add("workers", "one worker gives the same result (run with one "
               "worker: nothing to compare)", True, "N_WORKERS = 1")

    by = {k: [r for (kind, _), r in zip(jobs, res) if kind == k]
          for k in TASKS}
    san = sanity_checks(pd.DataFrame([r for rows in by["sanity"]
                                      for r in rows]), ck)
    PW, SG, BM, BMM = [], [], [], []
    for a, b, c, m in by["pool"]:
        PW += a
        SG += b
        BM += c
        BMM.append(m)
    pw, sg = pd.DataFrame(PW), pd.DataFrame(SG)
    bm, bmm = pd.DataFrame(BM), pd.DataFrame(BMM)

    # --- reproduction checks ---
    keys = ["lists", "size", "scenario", "effect", "N", "test", "feature"]
    skeys = ["lists", "size", "scenario", "effect", "N", "block", "feature"]
    v5rows = pw[pw["lists"] == "v5"]
    compare_exact(v5rows[keys + ["rate"]], pw7[keys + ["rate"]], keys,
                  "rate", "the v5-list rows (side model) equal script 7's "
                  "paired_power.csv", ck)
    if PERSON_MODEL != "side":
        sp = pd.DataFrame([r for a, _ in by["side_repro"] for r in a])
        ss = pd.DataFrame([r for _, b in by["side_repro"] for r in b])
        compare_exact(sp[keys + ["rate"]], pw7[keys + ["rate"]], keys,
                      "rate", "side model on the slice: rates equal script "
                      "7's paired_power.csv", ck)
        compare_exact(ss[skeys + ["claim_rate"]], sg7[skeys + ["claim_rate"]],
                      skeys, "claim_rate", "side model on the slice: sign "
                      "test equals script 7's secondary_sign.csv", ck)
        ra = r7b[r7b["model"] == PERSON_MODEL].assign(lists="v5b")
        mine = pw[pw["lists"] == "v5b"].merge(
            ra[keys], on=keys)[keys + ["rate"]]
        ck.add("reproduction", "every cell script 7b ran is in this run",
               len(mine) == len(ra), f"{len(mine)} of {len(ra)}")
        compare_exact(mine, ra[keys + ["rate"]], keys, "rate",
                      f"{PERSON_MODEL} model: rates equal script 7b's "
                      "paired_rates.csv on the cells it ran", ck)
    else:
        mine = pw[pw["lists"] == "v5b"]
        compare_exact(mine[keys + ["rate"]], pw7[keys + ["rate"]], keys,
                      "rate", "side model: every rate equals script 7's "
                      "paired_power.csv", ck)

    r2 = pd.DataFrame([r for rows in by["replicate"] for r in rows])
    r1 = pw[(pw["scenario"] == "all nuisances (defaults)")
            & pw["effect"].isin(REPLICATE_EFFECTS)
            & (pw["lists"] == "v5b")]
    rep = replicate_compare(r1.drop(columns=["model"]),
                            r2.drop(columns=["model"]))
    az = rep["z"].abs()
    ck.add("Monte Carlo", "headline cells agree between two independent "
           "pools (pool-to-pool error is small)",
           az.max() <= SANITY_MAX_Z and (az > 1.96).mean()
           <= SANITY_SHARE_BEYOND,
           f"{len(rep)} comparisons; largest |z| {az.max():.2f}; "
           f"{(az > 1.96).mean():.1%} beyond 1.96; SD z {rep['z'].std():.2f}")

    # --- general checks on the results ---
    nul = pw[(pw["scenario"] == "no nuisances") & (pw["effect"] ==
                                                   "no effect")]
    zz = (nul["rate"] - ALPHA) / nul["se"]
    ck.add("results", "with no effect and no nuisances, both tests stay "
           f"at or below {ALPHA} + 3 SE (measured Monte Carlo SE) at every N",
           (zz <= 3).all(), f"largest rate {nul['rate'].max():.4f}, largest "
           f"z {zz.max():+.2f}; mean rate t "
           f"{nul[nul['test'] == 't']['rate'].mean():.4f}, Wilcoxon "
           f"{nul[nul['test'] == 'wilcoxon']['rate'].mean():.4f}")

    def rises(g):
        r, e = g["rate"].to_numpy(), g["se"].to_numpy()
        return bool(np.all(np.diff(r) > -3 * np.sqrt(e[1:] ** 2
                                                     + e[:-1] ** 2) - 0.002))
    mono = pw[pw["role"].isin(["power", "power (rescaled)"])].sort_values(
        "N").groupby(["size", "effect", "test", "feature"])[
        ["rate", "se"]].apply(rises)
    ck.add("results", "power rises with N (within 3 SE of Monte Carlo "
           "noise between neighbouring N)",
           mono.all(), f"{int(mono.sum())} of {len(mono)} series")
    ex = bmm[(bmm["effect"] == "no effect") & (bmm["lists"] == "v5b")]
    ck.add("results", "the share excluded for 3+ catch failures matches the "
           "lapse model (about 5% at mean lapse 0.1, no nuisances)",
           ex[ex["scenario"] == "no nuisances"]["excluded_share"].between(
               0.03, 0.07).all(),
           f"share excluded, no nuisances: "
           f"{ex[ex['scenario'] == 'no nuisances']['excluded_share'].round(3).tolist()}")
    rule = apply_rule(pw[pw["lists"] == "v5b"])
    log(f"rule: chosen {rule['chosen']} (original rule: "
        f"{rule['chosen_original_rule']})")

    # --- the side model beside (read from script 7's run) ---
    pw["model"] = pw["model"].fillna(PERSON_MODEL)
    side7 = pw7[(pw7["lists"] == "v5b")].copy()
    side7["model"] = SIDE_LABEL
    if SIDE_BESIDE and PERSON_MODEL != "side":
        pwb = pd.concat([pw, side7], ignore_index=True)
    else:
        pwb = pw.copy()
    order = {k: i for i, k in enumerate(pos)}
    rank = {PERSON_MODEL: 0, SIDE_LABEL: 1, "side": 2}
    pwb["_o"] = [order.get((a, b), 10 ** 6) for a, b in
                 zip(pwb["scenario"], pwb["effect"])]
    pwb["_r"] = pwb["model"].map(rank).fillna(9)
    pwb = pwb.sort_values(["_o", "_r", "size", "N", "test", "feature"],
                          kind="stable").drop(columns=["_o", "_r"])
    ept = episode_table(pwb, leak_df.to_dict("records"))
    mx = pd.DataFrame()
    mixed_verdict = []

    lines = [f"Per-person model: {PERSON_MODEL} model, Firth, no intercept, "
             "fitted per block on the analysed",
             "trials" + (" (columns x_sat, x_con, side, N_corpus, N_orig, "
                         "in item order, per block and set, as script 7b)."
                         if PERSON_MODEL == "adjusted" else
                         " (script 7's side model)."),
             "Thresholds (secondary sign test only): "
             + ("computed here from script 6's 20,000 random choosers (equal "
                "to script 7b's);"
                if PERSON_MODEL == "adjusted" else "script 6's;"),
             "they replace the frozen two-weight thresholds; the lists are "
             "not re-frozen.",
             f"Side-model rows labelled '{SIDE_LABEL}' are read from script "
             f"7's run ({PAIRED_RUN_DIR}),",
             "not recomputed (sensitivity rows). The pools and studies are "
             "script 7's (same streams).", ""] + hist["choice"] + [""]
    hist["model_lines"] = lines

    # --- write ---
    scen_df.to_csv(os.path.join(out, "scenarios.csv"), index=False)
    pw.to_csv(os.path.join(out, "paired_power.csv"), index=False)
    side7.to_csv(os.path.join(out, "side_model_script7.csv"), index=False)
    sg.to_csv(os.path.join(out, "secondary_sign.csv"), index=False)
    pd.concat([bmm, bm], ignore_index=True).to_csv(
        os.path.join(out, "block_measures.csv"), index=False)
    san.to_csv(os.path.join(out, "sanity_explore.csv"), index=False)
    rep.to_csv(os.path.join(out, "pool_replicates.csv"), index=False)
    leak_df.to_csv(os.path.join(out, "leak.csv"), index=False)
    fat_df.to_csv(os.path.join(out, "fatigue_calibration.csv"), index=False)
    resc_df.to_csv(os.path.join(out, "rescale_factors.csv"), index=False)
    ept.to_csv(os.path.join(out, "episode_strategy.csv"), index=False)
    if len(th_df):
        th_df.to_csv(os.path.join(out, "thresholds_adjusted.csv"),
                     index=False)
    hist["rows"].to_csv(os.path.join(out, "model_choice.csv"), index=False)
    if MAKE_FIGURES:
        try:
            figures(pw[pw["lists"] == "v5b"], bmm, bm, out)
            fig_episode(pwb, out)
        except ImportError:
            log("matplotlib not installed: no figures")
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(jsonable(settings_dict()), f, indent=2)
    pd.DataFrame(ck.rows).to_csv(os.path.join(out, "checks.csv"),
                                 index=False)
    summary = plain_summary(ck, pw, pwb, rule, resc_df, ept, hist, TH)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(f"THE PRIMARY PAIRED TEST, {PERSON_MODEL.upper()} MODEL "
                "(simulate_power_paired_v5b_adj.py): IN PLAIN WORDS\n\n"
                + "\n".join(summary) + "\n")
    files = sorted(set(os.listdir(out)) | {"report.txt", "run_meta.json"})
    import scipy
    meta = {"started": started, "minutes": (time.time() - T0) / 60,
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__,
            "files": files, "workers": n_workers, "summary": summary}
    write_report(out, ck, scen_df, pw, sg, bm, bmm, mx, san, rep, rule,
                 mixed_verdict, meta, leak_df, fat_df, resc_df, ept, pwb,
                 hist)
    with open(os.path.join(out, "run_meta.json"), "w") as f:
        json.dump(jsonable({**{k: v for k, v in meta.items()
                               if k not in ("files", "summary")},
                            "output_dir": out, "platform": platform.platform(),
                            "cpu_count": os.cpu_count(),
                            "person_model": PERSON_MODEL,
                            "thresholds": {f"{s} {b}": v
                                           for (s, b), v in TH.items()},
                            "input_sha256": hashes,
                            "chosen_test": rule["chosen"],
                            "chosen_test_original_rule":
                                rule["chosen_original_rule"],
                            "rule": {k: v for k, v in rule.items()},
                            "rescale_factors": resc_df.to_dict("records"),
                            "mixed_model": hist["mixed_short"],
                            "model_choice": hist["rows"].to_dict("records"),
                            "checks_pass": len(ck.rows) - ck.n_fail,
                            "checks_fail": ck.n_fail}), f, indent=2)
    log(f"done: {len(ck.rows) - ck.n_fail} PASS, {ck.n_fail} FAIL; "
        f"model {PERSON_MODEL}; primary test: {rule['chosen']}")
    log(f"report: {os.path.join(out, 'report.txt')}")


if __name__ == "__main__":
    main()
