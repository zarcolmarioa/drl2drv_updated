# The analysis script: plan

**Status:** 21 Oct 2026. Both scripts are written and validated on simulated
data (`analysis_pipeline/`, outputs in `data_analysis/`). Only the reader for the
experiment's CSV files is pending.

**Decisions taken after the prompt was sent,** now in the scripts:
- **Which image was chosen** is derived in the analysis from the key pressed
  and the frozen list's `a_is_left`, and checked. A test with reversed answers
  confirms the run stops.
- **The state items** use a 0–6 scale.
- **More than 20 participants or 30 controls:** the first 20 / 30 by
  completion time are analysed and the rest listed.
- **The cap on extreme weights** is colour 8.378 and fog 10.269, computed on
  the frozen list (the power script describes it but never computed it).
- **The attention check and C1** run on the analysable people, after the
  catch exclusion.
- **H3** is the angle atan2(b_fog, b_sat).
- **Flat word raters** are excluded from the word-rating correlations only.

**What this file is.**
- **Part 1** explains what the analysis does, how the scripts are built, which
  simulated data test them, and why.
- **Part 2** is a prompt for another Claude session that will write the
  scripts.

**The way of working.** The session writes and edits the scripts for you to
download. You run them on your computer and upload outputs for the session to
check. When the final version is ready, you put the scripts in a new folder and
commit them. The tested script is filed with the preregistration, with a
checksum, like the frozen trial list.

---

## Part 1. What the analysis does, and why

### 1. Two scripts

- **`simulate_study_data_v4.py`** makes fake datasets whose true answer is
  known.
- **`analyse_study_v4.py`** analyses a dataset, real or simulated.

Both follow the project's script conventions:
- a SETTINGS block and no command-line arguments;
- their own output folder, never overwriting an earlier run;
- a report and a `run_meta.json`;
- checks that stop the run when something is wrong.

**Why two.** Testing the analysis on data whose answer is known is the only way
to show, before any real data exist, that the script reaches the right
conclusions and that its false-claim rate is the registered one.

### 2. How the analysis script is built

- **Python, reusing the power script's own code.** The Firth fit, the
  thresholds, the decision rule, the Bayes factor, the cap on extreme weights,
  the prevalence estimate and the attention check are copied from
  `simulate_power_v4_n10_20.py` (the run the registered numbers come from;
  its functions are identical to `_corrected.py`), because scripts are
  standalone. A test
  shows the copies give identical results.

  *Why:* the registered error rates and power were computed with that code. If
  the analysis used different code, its error rates would not be the ones
  registered.
- **Two layers.**
  - A *loader* turns the experiment's CSV files into tidy tables: one row per
    trial, per person, per grid screen.
  - The *analysis* only ever sees those tables.

  *Why:* the experiment's output format is changing now (the experiment-code
  changes). The analysis can be written and tested now on tidy tables. The
  loader is finished last, against the updated data dictionary and a few real
  test sessions from the new experiment.
- **Inputs.** All inputs are read from paths in SETTINGS, in the usual project
  layout (`data/…`):
  - the session CSV files, downloaded from OSF;
  - the frozen trial list, whose checksums are verified before use;
  - the registration table (`combined_rule.csv`);
  - the grid table (`grid_trials.csv`);
  - **two private files that stay on your computer:**
    - the survey file with the CDS item scores;
    - a small participant table: ID, group, tier (A–D or control).

    These two are needed only for the real data. For the simulated data, the
    simulation script writes fake versions of both, in the same format, so
    nothing private is needed to develop and test the scripts.

  The interview is **not** an input; see section 5.
- **Outputs.**
  - A report, tables (one row per person; one row per test) and a few
    figures.
  - Complete tables, not article-ready layouts.
  - Shareable outputs use pseudonyms (P01…, C01…). The key linking pseudonyms
    to IDs is written to a separate private file.

### 3. What the analysis does

#### 3.1 Preparing the data (registered rules)

- **Join to the frozen trial list.** Each session is joined to the frozen
  trial list by trial number. Which image was chosen is derived from the key
  pressed and which side image A was on. The predictors are taken from the
  trial list, not from the data file.
- **Who is analysable.** A person counts if:
  - the session is complete;
  - it is not a test or development session;
  - the images fitted the screen;
  - they failed fewer than 3 of the 10 catch trials;
  - and it is their first complete session.
- **Partial exclusions.**
  - Flat word ratings (the same rating for all 10 words) are left out of the
    word-rating analysis only.
  - A missing grid, word-rating or state block is left out of that analysis
    only.
- **Separate reporting.** Tier D participants are reported separately and do
  not count toward N.
- **Final numbers.** The script reports N for participants and controls, and
  how many people each rule removed.

*Why:* these are the registered rules (`experimental_design.md`, section 2.4).
The script applies them mechanically, so nothing is decided after seeing the
data.

#### 3.2 Primary: H1 (fog) and H2 (colour)

1. **Fit each person's model:** a Firth logistic regression with no
   intercept on `pred_fog` and `pred_sat`, over the 180 analysed trials. Each
   weight is reported with its interval.
2. **Classify each person, per feature:** detected in the predicted
   direction, detected in the opposite direction, or not detected. The
   thresholds are colour +0.286 / −0.290 and fog +0.488 / −0.486.
3. **Look up the rule (k, m)** for the final N in the registration table, and
   state the verdict for each feature: supported or not.
4. **With fewer than 10 analysable participants,** there is no verdict; the
   counts are reported descriptively.

*Why:* this is the registered test. Each claim is held at or below 5% false
claims, including when people without the effect have personal tastes.

#### 3.3 Primary: C1 (participants against controls)

1. **The attention check comes first.** It compares catch failures, and the
   share of responses under 300 ms, between the groups, with controls expected
   to be worse. It uses one-sided Mann–Whitney tests, Holm across the two
   measures.
2. **C1 itself:** a one-sided Mann–Whitney test per feature (participants'
   weights higher), Holm across the two features.
3. **If the attention check fires,** C1 is "inconclusive" and the analysis
   restricted to people with zero catch failures is reported beside it.
4. **With fewer than 20 analysable controls,** C1 is reported descriptively.

*Why:* a group that lapses more has weights pulled toward zero, which would
fake a difference. The check catches that first.

#### 3.4 Secondary

Per feature:
- a one-sided t-test and a JZS Bayes factor (scale 0.707) on the weights;
- reported with and without the cap on extreme weights.
  - **The cap** is the Firth fit for the strongest follower of one cue, short
    of separation. Every trial with |predictor| ≥ 0.20 is answered as the cue
    predicts, every other trial is answered y = 0.5, and the other weight is
    free.
  - **Its values** on the frozen list are colour 8.378 and fog 10.269. They
    are flat across the informative-trial threshold, so the choice of 0.20
    does not matter.
  - **Use:** each weight is clipped to [−cap, +cap]. The power script
    describes the construction but never computed it, so these values are
    stated in the preregistration.

*Why:* it answers "does the group lean on average?". One person who answers
every trial as predicted gets a huge weight and can dominate a mean; the cap
stops that.

#### 3.5 Exploratory and descriptive (reported with intervals, no claims)

- **Prevalence.**
  - The counts in each direction.
  - The estimate under three assumed sensitivities: weak (0.30/SD), moderate
    (0.50/SD) and strong (0.70/SD). Each is labelled with what it assumes, has
    its interval, and draws no "excludes zero" inference.
- **The Suzuki-style analysis:** on the 60 single-feature trials of each kind,
  each person's count of predicted choices (39 of 60 or more counts as
  following the feature), plus a group t-test on the proportions.
- **H3:** each person's contrast-to-colour balance, as registered: the weight
  ratio, compared between people only. It is computed as the angle of the
  weight pair, which ranks people exactly as the ratio does but stays defined
  when the colour weight is near zero or negative. Also reported: the count of
  people detected on both features.
- **E1, descriptions against weights:**
  - CDS item 13 against the fog weight, and item 2 against the colour weight
    (item 2 is reported but is not a colour test);
  - the word ratings: each target (foggy, faded) minus the person's mean of
    the four baseline words, against the weights.

  Reported per person and as correlations with intervals.
- **E2, the grid:**
  - the shift from normal to episode, per feature, in feature units, using
    each chosen cell's `sd_from_original`;
  - reported with it: the fog answers on the original column in each block
    (censored), "unedited" choices separately, "none of these", the set code
    and the cell size.
- **Dose-response and additivity:** the share of predicted choices by
  difference size and trial type.
- **Quality and covariates:**
  - consistency, response times, catch failures;
  - gamma, pixel density;
  - the delay between interview and task;
  - the state items.

*Why:* these are the registered exploratory analyses; nothing new is added. They
describe and qualify the primary results, but no claim rests on them.

### 4. The simulated data

The simulated data are made with **the same model as the power simulation**:
each person has a weight per feature, a lapse rate, and possibly a personal
taste. Choices are drawn on the frozen trial list.

**Scenarios with a known answer:**

| Scenario | What should come out |
|---|---|
| Nobody follows either feature (random choosers) | no claim, for either feature |
| Nobody follows, but personal tastes in random directions (worst case) | no claim, in at least 95% of simulated studies |
| Fog only, half the participants, moderate strength | H1 supported most of the time, H2 not |
| Colour only, half the participants | H2 supported, H1 not |
| Both features | both supported most of the time |
| Controls using a "shared idea" strategy | C1 tested as normal |
| Controls lapsing more than participants | the attention check fires; C1 inconclusive |
| N = 9 participants | descriptive only, no verdict |
| N = 10, 15, 20 | the rule for that N is used |
| 19 controls | C1 descriptive |

**Messy-data cases, for the loader:**
- test IDs and a development session (excluded);
- a person with two sessions (first complete one used);
- 3 catch failures (excluded);
- images that did not fit (excluded);
- flat word ratings (out of the word analysis only);
- a missing grid or state block;
- a Tier D person (reported separately);
- a missing interview date.

**The checks the script must pass:**
1. **The weights** equal the power script's Firth fit on the same data, to
   rounding precision.
2. **The thresholds and rules** used are exactly the registered ones.
3. **The clear scenarios** come out as built.
4. **False claims.** Over many simulated studies with no effect (for example
   1,000 at N = 10, 15 and 20), the share with a false claim stays within the
   registration table's figures for that N, allowing for simulation noise.
5. **Every exclusion** is counted correctly.

*Why:* checks 1 and 2 show the analysis is the analysis that was simulated.
Checks 3 and 4 show it reaches the right conclusions. Check 5 shows the
registered exclusion rules are applied as written.

### 5. Left out on purpose

- **The interview.** The codebook is not preregistered and the quality of the
  interview data is not known yet. The interview part of E1 is therefore not
  in this script and is not simulated. If the interview data turn out usable,
  that analysis is added later, reported as exploratory.
- **Article-ready tables.** The script writes complete tables; formatting for
  the article is done later.

### 6. Order of work

1. The simulation script, and the analysis on tidy tables, tested against the
   checks above. Can start now.
2. The loader for the experiment's CSV files. Once the experiment changes are
   merged, the session reads the updated `DATA_DICTIONARY.md`. You then run a
   few test sessions of the new experiment (both groups) and upload the CSVs.
3. A final run on simulated raw CSVs. The script is then frozen with a checksum
   and filed with the preregistration.

---

## Part 2. Prompt for the other session

Copy everything between the two lines into a new session that has access to the
`drl2drv_updated` and `dpdr_experiment` repositories.

---

You are going to write the analysis script for a preregistered study, and a
second script that simulates data to test it. You have no earlier context, so
this message gives you what you need. **Read it all, then reply as described
under "How to work" before writing any code.**

### The study in brief

People who have had episodes of derealization see two edited versions of the
same photograph on each trial. They press the left or right arrow key to choose
the one that looks more like how their surroundings looked during one specific
episode. The versions differ in local contrast (a "fog"-like reduction,
predictor `pred_fog`), in colour saturation (`pred_sat`), or in both. Each
person makes 180 analysed choices.

A **control group** does the same task, imagining how the surroundings might
look to someone experiencing derealization.

The design is preregistered, so the analysis below is fixed. Do not add,
remove or change any test, threshold or rule. If something seems wrong, ask.

### Where things are

- **`drl2drv_updated`** (the research project; read only):
  - **`stimulus_power_pipeline/simulate_power_v4_corrected.py`, the power
    simulation.** The registered error rates and power were computed with its
    functions. Read its SETTINGS and the functions you will reuse:
    - `fit_firth` and `fit_firth_batch`;
    - `compute_floors` and `classify` (the thresholds and the three-way
      classification);
    - `rule_mask_count` (the k/m rule);
    - `jzs_bayes_factor`;
    - the cap construction (see "The caps" in SETTINGS and where it is
      computed);
    - `prevalence_posterior_shared`, `prevalence_moment` and `hpdi`;
    - `mann_whitney_one_sided`, `holm_two` and `_c1_gate`;
    - `lapse_p` and `simulate_lapsing`;
    - `verify_frozen` and `load_trials`.
  - **`stimulus_power_pipeline/registration_table_v4.py`,** which selected the
    registered rule for each N.
  - **`data_power_v4/registration_v4/`:**
    - `combined_rule.csv`: the registered rule (k, m) for each feature and
      each N from 10 to 20, with its false-claim rates;
    - `registration_report.txt`.
  - **`data_power_v4/power_v4_n10_20/`:** the power outputs, including
    `frozen/detection_curve.csv` (sensitivities for prevalence) and
    `frozen/rule_sweep.csv`.
  - **`data_trials/frozen_trial_list_v4/`:** the frozen trial list
    (`trial_list.csv`, `checksums.sha256`, `package/trial_list.json`).
  - **`data_trials/grid_block_v4/package/`:** `grid_trials.json`,
    `grid_trials.csv` and `interface_spec.md` (the grid blocks).
  - **Docs:**
    - `docs/experimental_design.md`: the registered design. Sections 2.2–2.4
      are the hypotheses, analyses and exclusion rules.
    - `docs/v4_stimulus_power_explained.md`: sections 12–13 are the power
      analysis and the registered numbers.
- **`dpdr_experiment`** (the browser experiment; read only):
  - `DATA_DICTIONARY.md`: the output columns.
  - **It is being updated right now** by another session, to a new format
    (new session order, a `group` column, new word ids, two grid blocks, state
    items, `interview_date`). Until that is merged, write the loader against
    the expected columns listed below, and keep it separate so it can be
    finished later.

**Running conventions.** The scripts run on the user's computer from a project
folder that holds the scripts at its top level and all data under `data/`.
Every path is relative to that folder and set in SETTINGS. So, for example, the
frozen list is at `data/frozen_trial_list_v4/` and the registration table at
`data/registration_v4/combined_rule.csv`. The user downloads your scripts, runs
them, and uploads outputs for you to check. Other conventions:
- every script is standalone, with a SETTINGS block and no command-line
  arguments;
- it writes to its own output folder, never overwriting an earlier run (a
  rerun goes to `_2`, `_3`, …);
- it writes a human-readable report and a `run_meta.json` (settings, input
  checksums, check results);
- it has checks that stop the run when something is wrong.

Environment: Python 3 with numpy, pandas, scipy and matplotlib.

### Two scripts to write

1. **`simulate_study_data_v4.py`**: makes simulated datasets with known
   answers (see "Simulated data").
2. **`analyse_study_v4.py`**: analyses one dataset, real or simulated.

**Reuse the power script's code; do not rewrite it.** Copy the functions listed
above into the scripts, as each script is standalone. Mark them as copies. Add a
start-up check that, on a fixed simulated dataset, the copied functions give
results identical to the power script's, to rounding precision. The registered
error rates were computed with that code, so the analysis must be that code.

**Structure of the analysis script: two layers.**
- **The loader** reads the session CSV files and produces tidy tables:
  - per trial: person, group, trial, trial type, analysed, side pressed,
    chose A, RT;
  - per person: session-level fields, exclusion flags, word ratings, state
    items;
  - per grid screen.
- **The analysis** reads only the tidy tables.

The simulation script writes the tidy tables directly, and later also raw CSVs
in the experiment's format.

### Inputs (paths in SETTINGS)

- **The session CSV files:** a folder of them.
- **The frozen trial list.** Verify `checksums.sha256` before use and stop on
  any mismatch. Use `pred_sat` and `pred_fog` from `trial_list.csv`, joined by
  trial number; never use `z_sat` or `z_fog`, which are unsigned.
- **`combined_rule.csv`,** and `grid_trials.csv` for the grid.
- **Two private files that stay on the user's computer:**
  - the survey file. It is an `.xlsx` with one row per respondent: `ResponseId`
    (the participant ID used by the experiment) and the CDS item scores
    `Q01_score` … `Q29_score`;
  - a participant table: `participant_id`, `group` (`participant` or
    `control`), `tier` (A, B, C, D, or empty for controls).
- **Expected columns in the new experiment format** (confirm against the
  updated `DATA_DICTIONARY.md` when it exists):
  - session-level: `participant_id`, `group`, `interview_date`, `is_test`,
    `is_dev`, `consent_given`, `images_fit`, `device_pixel_ratio`,
    `gamma_estimate`, `set_code`;
  - pair rows: the trial number, the design trial type, the key pressed, RT;
    catch rows carry `catch_pass`;
  - word ratings: one column per word id: `foggy`, `faded`, `veiled`,
    `lifeless`, `unreal`, `dreamlike`, `distant`, `like_a_picture`,
    `oppressive`, `threatening`;
  - grid rows: `block` (`episode`/`normal`), `set_code`, `scene_set`,
    `screen`, `feature`, `column_order`, `scene_ids`,
    `chosen_display_column`, `chosen_level_index`, `chose_none`,
    `free_text`, `rt_ms`, `cell_px`;
  - state items: `state_q02`, `state_q13`, `state_q19`, `state_q29`.

### What the analysis does, in order

**A. Who is analysable** (registered rules; apply them mechanically and report
how many each removes):
- **Excluded from everything:**
  - test or development sessions;
  - sessions without consent;
  - images that did not fit (`images_fit` false);
  - 3 or more failures on the 10 catch trials.
- **Repeat sessions:** only the first complete session per ID is used; later
  ones are listed.
- **Partial exclusions:**
  - flat word ratings (the same rating for all 10 words) are left out of the
    word-rating analysis only;
  - a missing grid, word or state block is left out of that analysis only.
- **Tier D participants** are analysed and reported separately. They do not
  count toward N, and they are not in H1/H2 or C1.
- **Not exclusions:** consistency trials (a quality report only) and fast
  responses (used only in the attention check).
- **Report** the final N of analysable participants and controls.

**B. Per-person model.** For each analysable person:
- fit a Firth logistic regression, no intercept: chose A ~ `pred_fog` +
  `pred_sat`, over the 180 analysed trials, using the power script's
  `fit_firth`;
- report both weights with 95% profile penalised-likelihood intervals;
- classify each weight against the registered thresholds: colour +0.286 /
  −0.290, fog +0.488 / −0.486. Read them from the power outputs, or recompute
  them with the power script's `compute_floors` on the frozen list with the
  same seed and number of simulated guessers, and check they match;
- the classes are: detected in the predicted direction, in the opposite
  direction, or not detected.

**C. H1 (fog) and H2 (colour)**, primary, two separate claims each at 5%:
- **With the final participant N from 10 to 20:** take (k, m) for that N and
  feature from `combined_rule.csv`. The claim is supported if at least k
  people are detected in the predicted direction and at most m in the
  opposite direction. Report the counts, the rule, the verdict, and the rule's
  registered false-claim rate and power.
- **With N below 10:** no verdict; report the counts descriptively.
- **More than the maximum** (sessions completing at the same time): use the
  first 20 participants by session completion time, and list the rest
  separately as beyond the maximum. The same applies to controls, with the
  first 30.

**D. C1 (participants against controls)**, primary:
1. **The attention check:** catch failures per person, and the share of image
   trials with RT under 300 ms. Compare them between the groups with one-sided
   Mann–Whitney tests (controls worse), Holm across the two, alpha 0.05, using
   the power script's `_c1_gate`.
2. **C1:** a one-sided Mann–Whitney test per feature (participants' weights
   higher than controls'), Holm across the two features.
3. **If the check fires,** C1 is "inconclusive". The same test on people with
   zero catch failures is reported beside it, if each group has enough people
   (the power script's minimum).
4. **With fewer than 20 analysable controls,** C1 is descriptive.

**E. Secondary**, per feature, for participants:
- a one-sided one-sample t-test on the weights (mean > 0), and a one-sided
  JZS Bayes factor, scale 0.707, via `jzs_bayes_factor`;
- both computed with and without the cap on extreme weights. The cap is
  colour 8.378 and fog 10.269, computed on the frozen list by the construction
  described under "The caps" in the power script's SETTINGS. Each weight is
  clipped to [−cap, +cap].

**F. Exploratory and descriptive** (intervals, no claims):
- **Prevalence**, per feature:
  - the counts in each direction;
  - the estimate and 95% interval (`prevalence_moment`,
    `prevalence_posterior_shared`, `hpdi`) under three assumed sensitivities.
    Each is labelled with the weight it assumes: weak 0.30/SD, moderate
    0.50/SD, strong 0.70/SD;
  - the sensitivities are read from `frozen/detection_curve.csv` at lapse 0,
    as in the power script. For reference: colour 0.50 / 0.86 / 0.99, fog
    0.22 / 0.51 / 0.79;
  - no "excludes zero" inference.
- **Suzuki-style:** for each person, the number of predicted choices on the
  60 `sat_only` trials and on the 60 `fog_only` trials (39 of 60 or more
  counts as following the feature), plus a one-sample t-test of the
  proportions against 0.5 across participants.
- **H3:** the registered contrast-to-colour weight ratio, compared between
  people only, computed as the angle atan2(b_fog, b_sat) of the raw weights.
  This ranks people as the ratio does, but stays defined when b_sat is near
  zero or negative. Report its distribution with no per-person test, plus the
  count of people detected on both features.
- **E1, descriptions against weights:**
  - CDS item 13 against the fog weight, and item 2 against the colour weight
    (item 2 is reported, not a colour test);
  - word ratings: foggy minus the person's mean of the four baseline words
    (unreal, dreamlike, distant, like a picture) against the fog weight, and
    the same for faded against the colour weight.

  Reported per person and as Spearman correlations with bootstrap intervals.
  Flat raters are excluded from the word-rating correlations only; they stay
  in the CDS-item analyses.
- **E2, the grid**, per person and feature:
  - the normal-to-episode shift in feature units: for each screen, the mean
    `sd_from_original` of the chosen column over the three rows (from
    `grid_trials.csv`), averaged over the two screens of that feature in each
    block, episode minus normal;
  - also reported: fog answers on the original column (`level_index` 0) in
    each block, which are censored, so count them; choices of the unedited
    column, reported separately; "none of these"; the set code; the cell size.
- **Dose-response and additivity:** the share of predicted choices by
  `mag_bin` and trial type, per group.
- **Quality and covariates, per person:**
  - consistency (agreement on the 15 repeated pairs), RT summary, catch
    failures;
  - `gamma_estimate`, `device_pixel_ratio`, grid `cell_px`;
  - the interview-to-task delay in days;
  - the state items.

**G. Outputs:**
- a report;
- tables: per person, per test, prevalence, grid, quality;
- a few figures: weights per person with thresholds; the C1 distributions;
  the grid shifts;
- `run_meta.json`.

Complete tables, not article-ready layouts. Shareable outputs use pseudonyms
(P01…, C01…, D01… for Tier D); the key is written to a separate file clearly
named as private.

### Simulated data (`simulate_study_data_v4.py`)

**The generating model.** Use the power script's model (`lapse_p` /
`simulate_lapsing`) on the frozen trial list. Each simulated person has a
colour weight and a fog weight per corpus SD, a lapse rate, and optionally a
personal taste (a symmetric random weight within a stated fraction of the
threshold, as in the power script's taste nulls). Simulate catch failures and
fast RTs from the lapse rate, as the power script does. Simulate plausible word
ratings, grid choices, session fields and CDS scores; they need not be
realistic, only well-formed. **Also write fake versions of the two private
inputs**, in the same format as the real ones: a survey `.xlsx` (`ResponseId`,
`Q01_score` … `Q29_score`) and the participant table (`participant_id`,
`group`, `tier`). Simulated IDs follow the real pattern (`R_` plus 15 letters and
digits). The scripts can then be developed and tested without the user's private
files.

**Scenarios** (each a separate dataset, with a truth file):
- **No effect:**
  - all random choosers;
  - the worst taste null (spread 1× the threshold).
- **Effects, at the registered planning cell** (half the participants
  responding; colour 0.50/SD, fog 0.70/SD; lapse 0.1):
  - fog only;
  - colour only;
  - both;
  - everyone responding.
- **Controls:**
  - no strategy;
  - a quarter using a "shared idea" strategy (weight 0.5/SD, as in the power
    script's C1 settings);
  - lapsing more than participants (the attention check must fire).
- **Sizes:** participant N = 9, 10, 15 and 20; controls 19 and 30.
- **Messy cases**, in one dataset:
  - test IDs and a dev session;
  - a person with two sessions;
  - 3 catch failures;
  - `images_fit` false;
  - flat word ratings;
  - a missing grid block;
  - a missing state block;
  - a Tier D person;
  - a missing interview date.

### Checks the scripts must pass (report each as passed or failed)

1. **The copied functions** give results identical to the power script's on a
   fixed dataset.
2. **The thresholds and the rules** used equal the registered ones (to the
   precision stored).
3. **The clear scenarios** give the expected verdicts:
   - no claim under no effect;
   - descriptive at N = 9;
   - C1 inconclusive when the controls lapse more.
4. **False claims:** over 1,000 simulated studies with no effect (random
   choosers, and the worst taste null) at N = 10, 15 and 20, the share with a
   false H1 or H2 claim is within the Monte Carlo error of the rule's
   registered false-claim rate in `combined_rule.csv` (at or below 5%).
5. **The messy dataset:** every exclusion and partial exclusion is applied and
   counted correctly.

### Not included

- **The interview.** Its codebook is not preregistered and its quality is
  unknown; it may be analysed later, separately.
- **Article-ready formatting.**

### How to work

- **First reply, before writing any code:**
  1. your understanding, in a few sentences, of the study and of what the two
     scripts must do;
  2. which power-script functions you will copy, and anything in them that
     does not match this message;
  3. a plan, in this order: simulation, then analysis on tidy tables, then
     the checks, then the loader;
  4. your questions: anything unclear, or anything in the repositories that
     conflicts with this message.

  **Then wait for the user's go.**
- **Deliver the scripts as files for the user to download,** in stages. The
  user runs them locally and uploads outputs for you to check.
- **Do not change any registered number, test or rule.** If the repositories
  disagree with this message, ask.
- **The loader is finished last,** after the experiment update is merged.
  Then read the updated `DATA_DICTIONARY.md`; the user will upload CSV files
  from test sessions of both groups.

---
