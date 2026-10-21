# Preregistration writing guide — recommendations and rationale

**Project:** Derealization episode-reconstruction image-choice study (frozen v4 stimulus list)  
**Prepared:** 21 October 2026  
**Purpose:** Living checklist for drafting, illustrating, checking, and ultimately freezing the preregistration. This is a **writing and evidence plan**, not a replacement for the registered protocol or the separate [issue tracker](preregistration_issue_tracker.md).  
**Status convention:** `DRAFT NOW` = can prepare from present records; `VERIFY BEFORE SUBMISSION` = wording or evidence must be settled before the final manuscript is frozen; `BEFORE RECRUITMENT` = implementation/ethics gate; `RESULTS LATER` = specify an output now but do not populate it with observed data until collection.

## 1. Overall recommendation

**Start the preregistration now.** The frozen 211-trial design, the analysis definitions, the N=10–20 count-rule table, and the existing power outputs provide a substantial basis for the manuscript. **Do not submit unconditional false-claim assertions or end-to-end C1 power estimates until the outstanding assumptions and procedural mismatches are addressed.** Do not freeze the analysis software until its raw-data reader and trial-integrity checks have been tested.

Use Kusztor et al.'s Stage 1 Registered Report chiefly as an example of **connecting questions → hypotheses → methods → analysis → sample-size justification → possible interpretations**, and of illustrating *possible outcomes* using simulations. The published paper is especially useful for framing limits on perceptual conclusions. **Do not import their specific statistical method:** they used Bayesian mixed-effects model comparisons, sequential Bayes-factor stopping, and other tasks. This study uses preregistered **per-person Firth logistic coefficients**, **directional count rules** for H1/H2, and **group comparisons** for C1, with **one final analysis**, not sequential hypothesis testing.

The distinction from Kusztor et al. should be visible throughout: **their participants judged image realness relative to normal everyday experience or adjusted images to match ordinary experience; our primary choice task asks people with identified episodes to select images resembling how surroundings looked during a *particular remembered episode*.** This difference motivates the study but does **not** prove that choices reflect sensory changes rather than reconstruction, framing, general expectations, or response tendencies.

## 2. Recommended manuscript structure and content

The items below are recommendations for the document itself. Each includes **what**, **why**, and **what is needed**. Tick them off as each section becomes manuscript-ready.

### R01 — State the scientific question narrowly and explicitly
- [ ] **Priority:** Essential · **Status:** `DRAFT NOW`.
- **Include:** A short introduction distinguishing (i) altered *reported/remembered appearance* of surroundings, (ii) perceived *realness* of an image, and (iii) a demonstrable change in visual processing. Explain why retrospective image matching is worth testing following Kusztor et al.'s null findings on other tasks.
- **Why:** These are related but nonidentical constructs. Readers must know exactly what the experiment can measure. A positive choice pattern would support **systematic episode-related visual matching**; it cannot, alone, establish that visual encoding was altered during the original episode.
- **Evidence/source:** `docs/experimental_design.md` §§1–2; Kusztor Stage 1, pp. 4–8; published article, abstract and discussion (including the El Greco discussion).

### R02 — Show a hypothesis-to-inference matrix, modelled on the Stage 1 Study Design Template
- [ ] **Priority:** Essential · **Status:** `DRAFT NOW`, revise wording before freeze.
- **Include:** One table with rows H1 (fog/local contrast), H2 (colour/saturation), C1 (participants vs controls), and separate rows marked *secondary* or *exploratory* as space permits. Suggested columns: research question; operational prediction; analysed observations; fitting/statistical test; decision criterion; sample-size/eligibility conditions; outcomes and allowed interpretations; principal alternative explanations.
- **Why:** The Stage 1 manuscript's Study Design Template (p. 19) makes the inferential logic inspectable. This is especially important here because H1/H2 are **existence/directional-count claims**, C1 is a **group-difference claim**, and H3, grids, words, prevalence, and dose–response do **not** have the same confirmatory status.
- **Important interpretation entries:** H1/H2 *supported*; H1/H2 *not supported*; an abundance of *opposite-direction* detections (descriptive, not a registered opposite-direction existence claim); C1 *supported*, *not supported*, *inconclusive after the attention gate*, and *descriptive only* when fewer than 20 analysable controls remain. With fewer than 10 analysable participants, H1/H2 are descriptive only.
- **Do not call “not supported” evidence for H0.** Unlike Kusztor's Bayes-factor analysis, the present H1/H2 rule is not designed to accumulate affirmative evidence for absence.
- **Evidence/source:** `docs/experimental_design.md` §§2.2–2.4; `docs/analysis_script_plan.md` §§3.2–3.5; `data_power_v4/registration_v4/combined_rule.csv`; Kusztor Stage 1, p. 19. **Issue:** C02.

### R03 — Define exactly what is manipulated and what the coefficients mean
- [ ] **Priority:** Essential · **Status:** `DRAFT NOW`.
- **Include:** A short technical description distinguishing **loss of local contrast** (the experiment's “fog” axis) from optical blur or a literal fog overlay, and **reduced saturation** (colour axis). State that the contrast manipulation can also reduce colour, so selected contrast-only pairs are saturation-matched; explain *together* versus *conflict* trials. Specify oriented predictors from the frozen list: `pred_sat = -d_sat / 39.182`, `pred_fog = +d_fog / 9.348`. The model uses **signed** predictors, not unsigned `z_sat`/`z_fog`.
- **Why:** Your central contribution depends on attributing choices to interpretable stimulus directions, not on ambiguous labels. It also explains why the trial design separates correlated perceptual dimensions.
- **Evidence/source:** `docs/experimental_design.md` §3; `docs/v4_stimulus_power_explained.md` §§2, 4–11, 13; frozen trial list.

### R04 — Document the exact experiment and quality-control trials
- [ ] **Priority:** Essential · **Status:** `DRAFT NOW`; update with verified deployed version before recruitment.
- **Include:** 211 presented trials = **6 practice + 60 saturation-only + 60 contrast-only + 30 together + 30 conflict + 15 consistency + 10 no-image catches**. The **180** trials of the four feature types are the confirmatory model input. The 15 consistency trials and 10 catch trials serve different purposes. State left/right balance, trial order frozen in a versioned file, trial identity/checksum, and that repeated scenes are intentional. Describe task instructions by group and the main session sequence (consent → ID/display calibration → episode anchoring → practice/pairs → episode grid → words → normal grid → optional state → upload).
- **Why:** Makes it possible to reconstruct the experimental exposure and prevents a reader mistaking checks for additional independent observations. Images, side assignments and order cannot be silently regenerated once registered.
- **Evidence/source:** `docs/experimental_design.md` §§3–4; `docs/experiment_code_changes.md`; `data_trials/frozen_trial_list_v4/package/trial_list.json`.
- **Caveat:** The experimental browser was still being updated when reviewed; describe **the protocol to be implemented**, then verify that the released implementation matches it. **Issues:** C01, C03.

### R05 — Specify sampling, stopping, inclusion and exclusions before seeing results
- [ ] **Priority:** Essential · **Status:** `DRAFT NOW`, **`VERIFY BEFORE SUBMISSION`**.
- **Include:** Participant and control selection from the earlier CDS survey/questionnaire; clear difference between the participant's experienced episode and the control's imagined scenario; priority tiers and Tier D; recruitment waves/time limits; first complete session rule; ceilings of **20 analysable participants and 30 controls**; what occurs if the pool is exhausted; **no sequential evidence-based stopping**; the threshold of **10 analysable participants** for H1/H2 claims and **20 analysable controls** for confirmatory C1. State exclusions: incomplete/development sessions, no consent, images not fitting, and **≥3 failed catches out of 10**; report separate partial exclusions for exploratory blocks.
- **Why:** The sample is small, selected, and recruitment-limited. A preregistration is informative only if final N and eligibility cannot be chosen in response to observed coefficient estimates.
- **Evidence/source:** `docs/experimental_design.md` §2.4; `docs/analysis_script_plan.md` §3.1; `analysis_pipeline/analyse_study_v4.py` exclusion logic.
- **Implementation note:** The analysis should verify the **exact set of 211 unique frozen trial IDs** and recompute/check catch correctness against the keys, rather than trusting row count and `catch_pass` alone. **Issue:** C03.

### R06 — Specify H1/H2 fitting and decision rules with all directional details
- [ ] **Priority:** Essential · **Status:** `DRAFT NOW`, **`VERIFY BEFORE SUBMISSION`**.
- **Include:** A **no-intercept, per-person Firth bias-reduced logistic regression** of chose-A on `pred_sat` and `pred_fog` over 180 analysed trials; choice reconstructed from pressed side and frozen `a_is_left`; report coefficient estimates and **95% profile intervals**. The frozen per-person thresholds are **colour +0.286/−0.290** and **fog +0.488/−0.486**. Each participant is classified, for each feature, as **predicted-direction**, **opposite-direction**, or **not detected**. For each possible final N=10–20, use the **fixed pre-selected `(k,m)` count rule** (at least `k` predicted detections, at most `m` opposite detections), separately for H1 fog and H2 colour; do not select the rule after examining results.
- **Why:** This is the inferential heart of the preregistration. Stating only “logistic regression followed by a group test” would misdescribe what determines an H1/H2 claim.
- **Evidence/source:** `docs/analysis_script_plan.md` §3.2; `docs/v4_stimulus_power_explained.md` §§12–13; `combined_rule.csv`. **Issue:** C02 on interpretation; C04 on claim calibration domain.

### R07 — Qualify the H1/H2 error-rate claim precisely
- [ ] **Priority:** Essential · **Status:** **`VERIFY BEFORE SUBMISSION`**.
- **Include:** Say that the **estimated ≤5% false-claim rate is conditional on the null models used for calibration**: random choosers and **zero-centred symmetric individual taste distributions** at the specified spreads, with the simulation's choice/lapse/dependence assumptions. The quoted Monte Carlo upper margins are *estimated bounds under these models*, **not an unconditional guarantee for arbitrary real preferences**; the per-cell upper intervals are not simultaneous post-selection guarantees. Clearly distinguish the probability of falsely supporting **each separate feature claim** from the chance of supporting **either H1 or H2**.
- **Why:** An entire nonresponder population that systematically prefers desaturated or lower-contrast images is not exchangeable across directions. A directional count rule may be triggered without episode-specific visual memory. A few registered cells have narrow margins (notably colour N=18). The theoretical claim must not be stronger than what was simulated.
- **Needed work:** Prefer a preregistered **sensitivity analysis** varying directional taste means/asymmetric mixtures. Re-evaluate selected rules independently in larger fresh null simulations if claiming strong calibration confidence; **do not** silently retune the primary rules after inspecting the data.
- **Evidence/source:** `stimulus_power_pipeline/simulate_power_v4_n10_20.py` `classification_rates`; `docs/v4_stimulus_power_explained.md` §12; `registration_table.csv`. **Issues:** C04, C11, S01.

### R08 — Report what negative, contrary and inconclusive outcomes would mean
- [ ] **Priority:** Essential · **Status:** **`VERIFY BEFORE SUBMISSION`**.
- **Include:** A paragraph/table of permitted conclusions. No H1/H2 claim when rule fails; no conclusion of *absence* unless a separate, appropriately justified exclusion/equivalence analysis is registered. State power at actual N; negative results may mean small effects, few responders, high lapses, or limited image range. Opposite-direction detections should be described, not converted into an unplanned confirmatory claim. The grid's original-endpoint censoring and “none of these” choices also affect interpretation. State that C1 only directly compares groups under different recall/imagination and interview conditions.
- **Why:** This addresses the largest overinterpretation risk. In the current simulation, even the specified planning alternatives frequently fail at N=10 (colour power 0.591, fog 0.443). Kusztor could discuss support for H0 because it registered Bayesian evidence thresholds; this analysis cannot treat “not supported” equivalently.
- **Evidence/source:** `docs/experimental_design.md` §2.3; `combined_rule.csv`; Kusztor Stage 1, pp. 17–19; published article discussion. **Issues:** C02, S03.

### R09 — Specify C1 as a separate confirmatory comparison with a conditional verdict
- [ ] **Priority:** Essential · **Status:** **`VERIFY BEFORE SUBMISSION`**.
- **Include:** C1 tests participants' **larger positive per-person weights** against controls, per feature, using **one-sided Mann–Whitney** tests and **Holm correction across the two features**. Before interpretation, an attention gate compares **whether controls are worse** on catch failures and share of responses under 300 ms, using **two one-sided Mann–Whitney tests with Holm correction across attention measures**. If it fires, C1 is **inconclusive**, not negative. Both gate and group tests run **after** the registered exclusion procedure. With <20 analysable controls, C1 is descriptive. Include the zero-catch-failure follow-up as specified by the plan, clearly labelled secondary.
- **Why:** Greater lapses in controls can pull their coefficients towards zero and create an apparent participant advantage; the gate protects against that interpretation. The phrase “groups differ” incorrectly implies a two-sided attention gate. Moreover, even a valid C1 difference cannot uniquely attribute a mechanism to altered perception.
- **Needed work:** Before quoting its power as **end-to-end power**, **repeat/extend C1 simulations with the ≥3-catch-failure exclusion actually applied**. Alternatively state clearly that the existing values describe a simplified procedure.
- **Evidence/source:** `docs/experimental_design.md` §2.2; `docs/analysis_script_plan.md` §3.3; `docs/v4_stimulus_power_explained.md` §12. **Issues:** C05, C08, S03.

### R10 — Separate secondary and exploratory analyses and correct H3 terminology
- [ ] **Priority:** Important · **Status:** `DRAFT NOW`, H3 **`VERIFY BEFORE SUBMISSION`**.
- **Include:** Secondary group-level one-sided t-tests and JZS Bayes factors (scale 0.707), with and without registered coefficient caps (**colour ±8.378**, **fog ±10.269**). Mark **prevalence estimates, Suzuki-style single-feature proportions, H3 balance, descriptions/word/CDS associations, grids, dose–response/additivity, state items, and quality/covariates** as exploratory/descriptive with intervals. Explain the assumed weak/moderate/strong sensitivities behind prevalence estimates. Specify that `atan2(b_fog,b_sat)` describes the **direction of a signed two-dimensional weight vector**, not a globally rank-preserving replacement for `b_fog / b_sat` across quadrants; preserve and show both raw coefficients.
- **Why:** These analyses make findings interpretable and detect artefacts, but many comparisons and limited sample sizes make them unsuitable as extra undisclosed confirmatory claims. Incorrect ratio language would make H3 mathematically ambiguous.
- **Evidence/source:** `docs/analysis_script_plan.md` §§3.4–3.5; `analysis_pipeline/analyse_study_v4.py`. **Issue:** C07.

### R11 — Include a short feasibility, quality and transparency section
- [ ] **Priority:** Important · **Status:** `DRAFT NOW`; test results before **`BEFORE RECRUITMENT`**.
- **Include:** What has already been validated (frozen list/counters; registered probability arithmetic; simulation datasets; tidy-data analysis checks) versus what remains (browser implementation, real raw-CSV import, full consent/upload path). Record any **non-confirmatory technical pilot** separately from recruited study data: completion time, device fit, understanding, catch performance, consistency, RT distribution, grid functionality, missingness, uploads, language checks. Label synthetic data and pilot outputs explicitly.
- **Why:** Kusztor Stage 1 includes empirical pilots and repeat-response reliability displays (pp. 13–14). This study has extensive *computational* validation but, from the materials reviewed, does not yet have an analogous human test of the frozen 211-trial browser experience. A small practical pilot is useful; a large new behavioural study is **not** a prerequisite to start writing.
- **Caution:** Do not use pilot effects to revise the registered primary rule without documenting a new pre-data design and rerunning its calibration. **Issues:** C01, C03, S02.

### R12 — Handle consent, ethics and data protection as part of the preregistered procedure
- [ ] **Priority:** Essential for protocol completeness · **Status:** **`BEFORE RECRUITMENT`** (wording agreed before final filing).
- **Include:** Approved consent language, what is uploaded at ID entry versus study completion, whether optional contact details survive abandonment, what is withdrawn on request and how, treatment of episode free text/interview dates, separation of linkable survey identifiers and shareable outputs, retention/OSF-access plans, and what data will be shared. Enforce consent in production; verify English/Japanese approved wording and the exact version deployed.
- **Why:** The proposed draft consent says closing the page saves nothing, while the then-current browser separately uploads optional contact information immediately. That discrepancy must be resolved before the consent procedure is final. Pseudonyms and short location/episode descriptions can still be sensitive.
- **Evidence/source:** `docs/experiment_code_changes.md` §12; browser `src/participant-id.js`; `docs/analysis_script_plan.md`. **Issues:** C06, C12, S04, S05.

### R13 — Define frozen-code and data provenance
- [ ] **Priority:** Important · **Status:** `DRAFT NOW`, **`BEFORE RECRUITMENT`** for completion.
- **Include:** Links/version/checksum for the exact frozen trial list, the registered `(k,m)` table, the simulator/settings and the tested analysis script; repository commit corresponding to the deployed browser; package/data dictionary; output folders; which rows or private tables are not shared. Acknowledge the browser may be updated to **read the frozen v4 package**, but the package and preregistered rules must not be silently changed.
- **Why:** The actual statistical plan is partly executable. Provenance is how later reviewers distinguish what was fixed *before* observations from subsequent engineering fixes.
- **Evidence/source:** `docs/v4_stimulus_power_explained.md` §§10, 12–13; `docs/analysis_script_plan.md` §§1–2; `data_analysis/analysis_v4_validation/run_meta.json`. **Issues:** C01, C03, C09–C10.

## 3. Figure and table plan, in priority order

**Rule for all visuals:** Every simulated example must be labelled **“simulated; not observed participant data”**. Diagrams of the task should use actual frozen stimulus images when available, never generated stand-ins presented as stimuli. Any figure showing *operating characteristics* must identify the generative assumptions and match the registered analysis/exclusions, or clearly disclose the differences. Figures need not reproduce the form of Kusztor's paper; they must clarify *your* hypotheses.

### Main manuscript: figures

#### F01 — Conceptual figure: what the experiment predicts and what it does not establish
- [ ] **Priority:** Essential · **Status:** `DRAFT NOW`.
- **Show:** A normal-looking scene and two visual-edit directions (reduced saturation, reduced local contrast), together with a simple “remembered episode → image-pair match” schematic. Distinguish the **behavioural prediction** from a **possible perceptual mechanism**; optionally juxtapose the control's imagined scenario. Avoid diagram arrows implying proven causality.
- **Why:** Like Kusztor Stage 1 **Figure 1**, it ties the theoretical question to observable choices, before any statistics are introduced. The manuscript's main conceptual risk is confusing image matching with proof of an earlier sensory distortion.
- **Inputs:** Actual frozen visual edits; `docs/v4_stimulus_pipeline_explained.md`; `docs/experimental_design.md` §1.

#### F02 — Actual stimuli and task flow
- [ ] **Priority:** Essential · **Status:** `DRAFT NOW`, validate screenshots **`BEFORE RECRUITMENT`**.
- **Show:** One or two **real original images and true edited versions**; a sample two-alternative forced-choice trial; one saturation-only, one contrast-only and ideally one *conflict* comparison; a brief flow of pair trials, catch and consistency trials, episode grid, words, normal grid. Clarify that the original is not equally represented as a high/low endpoint on both grids: the contrast grid has no level above original and can be censored. Label photos as examples, not as arbitrary calibrated display colors.
- **Why:** Like Kusztor Stage 1 **Figures 4–6**, this lets reviewers understand what participants actually see and how different dimensions are separated. It also exposes why cross-feature confounds, screen calibration and repeat scenes matter.
- **Inputs:** `data_trials/frozen_trial_list_v4/package/`, `docs/experimental_design.md` §§3–4, `data_trials` grid package/interface spec. **Issue:** C01 for final code screenshots.

#### F03 — Illustrative possible H1/H2 outcomes (**new high priority after reading Kusztor**)
- [ ] **Priority:** Essential · **Status:** `DRAFT NOW` with accurate simulated cases.
- **Show:** Simulated distributions or participant-level outcomes for **(a) a directional responder mixture**, **(b) symmetric nonresponders with personal tastes**, **(c) directionally biased nonresponders**, and optionally **(d) an opposite-direction pattern**. Display each feature's **predicted/opposite/not-detected** counts, the **actual registered** `(k,m)` condition at a clearly fixed N (e.g., 15), and whether the rule supports the feature. Simulations are illustrative and must not quietly redefine the null model or the rules.
- **Why:** Kusztor Stage 1 **Figures 2–3** explicitly show alternative result patterns. The analogous figure for this study should reveal **what the *decision rule* concludes**, not merely an average sigmoid. It is a safeguard against calling a visually attractive preference curve an H1/H2 finding.
- **Inputs:** `analysis_pipeline/simulate_study_data_v4.py`, `analysis_pipeline/analyse_study_v4.py`; validation scenarios in `data_analysis/analysis_v4_validation/` where suitable. A **small plotting script may be needed**, but the primary analysis should not be changed to create an illustration.

#### F04 — H1/H2 operating characteristics across final N
- [ ] **Priority:** Essential · **Status:** `DRAFT NOW` from existing outputs.
- **Show:** Two panels, H1 fog and H2 colour. X-axis = **final analysable participant N=10–20**; y-axis = probability of **meeting the registered positive rule**. Distinguish the **planning scenario** from **weaker effects at the same responder share** and **lower responder share at planning effect strength**. Label the prevalence, effect size and lapse assumptions. Do **not smooth away dips** from discrete `(k,m)` changes. A caption should say **non-support is not evidence of H0**.
- **Why:** The planning scenario looks much stronger than the weaker alternatives; hiding sensitivity would exaggerate what a null outcome could tell us. This adapts the explanatory purpose of Kusztor Stage 1's **Figures 9–10**, not their Bayesian sequential analysis.
- **Existing numbers at N=20:** planning colour **0.969**, fog **0.797**; weaker half-share colour **0.253**, fog **0.550**; quarter-share planning-strength colour **0.188**, fog **0.382**.
- **Inputs:** `data_power_v4/registration_v4/registration_table.csv`, registered **count-rule** rows; `docs/v4_stimulus_power_explained.md` §13. **No whole-pipeline rerun required** to plot the already committed scenarios.

#### F05 — False-claim calibration and its domain
- [ ] **Priority:** Essential, can be supplementary if space is tight · **Status:** `DRAFT NOW`, caption **`VERIFY BEFORE SUBMISSION`**.
- **Show:** Estimated worst tested-null false-claim probability and corresponding per-cell 95% Monte Carlo upper estimate for each of **11 N values × 2 features**, with a reference line at **0.05**. Mark thin-margin cases (notably colour N=18 and fog N=12). State the **symmetric zero-centred taste null**, random choosers, tested spreads and that the interval is **not a simultaneous model-free guarantee**. Optionally supplement with sensitivity results for asymmetric preferences.
- **Why:** The threshold “5%” is part of the scientific claim, but the model assumptions are equally important. The figure keeps them visible and makes the significance of near-boundary calibration honest.
- **Inputs:** `combined_rule.csv`, `registration_table.csv`; separate asymmetric-null sensitivity results when available. **Issues:** C04, C11, S01.

#### F06 — C1 power and inconclusiveness
- [ ] **Priority:** Important · **Status:** **`VERIFY BEFORE SUBMISSION`** for final numerical values.
- **Show:** Gated colour and fog C1 power versus number of participants (e.g. 10, 15, 20) for **20 and 30 controls** under **one fully stated scenario** (responder share, strategy prevalence, lapses). Include the chance of **attention-gate inconclusiveness**, either as separate bars/panels or in an adjacent small table. Make clear that with <20 analysable controls the real analysis is descriptive.
- **Why:** H1/H2 power does not transfer to C1. Under the existing simplified planning scenario, **20 participants + 30 controls** yield only **0.606 colour** and **0.574 fog** gated power. The gate may prevent a group claim even when a coefficient difference exists. Readers need the distinction.
- **Inputs:** `data_power_v4/registration_v4/c1_compact.csv` (provisional), `c1_cells.csv`; after rerunning C1 with the registered **≥3 catch-failure exclusion**, replace/endnote the estimates. **Issue:** C05.

### Main manuscript: tables

#### T01 — Hypothesis / analysis / interpretation matrix
- [ ] **Priority:** Essential · **Status:** `DRAFT NOW`.
- **Contents:** R02 in compact form. **Why:** A single traceable account of every confirmatory claim and permitted conclusion.

#### T02 — Registered H1/H2 count-rule lookup and operating characteristics
- [ ] **Priority:** Essential · **Status:** `DRAFT NOW`.
- **Contents:** For **each final N=10–20**, show `(k,m)` separately for **colour H2** and **fog H1**, planning-cell power, and optionally upper estimated false-claim rate/headroom. If too wide, put the full 11-row table in a preregistration appendix and retain a condensed 10/15/20 example in the text. Specify the table is **fixed** rather than chosen based on results.
- **Why:** Without this table, a reader cannot reconstruct the actual H1/H2 decision at the observed sample size. **Source:** `combined_rule.csv`; `registration_table.csv`.

#### T03 — Sample flow, exclusion and output definitions
- [ ] **Priority:** Important · **Status:** `DRAFT NOW` (empty results template).
- **Contents:** Inclusion/exclusion ordering, treatment of repeated attempts, cutoff at analysable N, controls below 20, Tier D, and partial data for words/grids/state. Columns planned for counts by group, but **do not fill with invented counts**.
- **Why:** Ensures exclusions do not silently change the denominator and separates full-session exclusions from exploratory-analysis missingness.

### Supplementary / optional explanatory figures

#### FS01 — Per-person detection sensitivity versus true feature weight
- [ ] **Priority:** Useful, not needed in main text · **Status:** `DRAFT NOW`.
- **Show:** Rate of classification in the registered direction against simulated true weight for each feature across lapse rates; thresholds in the caption. **Why:** Distinguishes the **individual detection stage** from the **group count-rule stage**, and explains why the coefficient scale matters.
- **Inputs:** `data_power_v4/power_v4_n10_20/frozen/detection_curve.csv`.

#### FS02 — Robustness to asymmetric taste and scene dependence
- [ ] **Priority:** Important if sensitivities reveal meaningful deviations · **Status:** **`VERIFY BEFORE SUBMISSION`**.
- **Show:** Estimated H1/H2 false-claim probabilities using **directional** nonresponder preferences, **within-scene correlated choice tendencies**, and optionally temporally clustered lapses, with the primary registered rules held **unchanged**. Report plausible parameter ranges and explain any dependence on assumptions.
- **Why:** These are precisely where the original model-based calibration may misrepresent actual choice behaviour. The frozen 180 analysed rows reuse **47 scenes**, **45 repeatedly**, with up to seven appearances. Kusztor's realness-rating task, by contrast, avoids showing the same image at different levels to the same participant; their second task uses repeats deliberately for consistency.
- **Inputs:** Adapted simulation of the frozen list; `docs/v4_stimulus_power_explained.md`, frozen `trial_list.json`. **Issues:** S01, S02.

#### FS03 — Non-confirmatory pilot feasibility and response consistency
- [ ] **Priority:** Optional but recommended once tested · **Status:** `BEFORE RECRUITMENT`.
- **Show:** Browser completion/dropout, trial RT, catch failure distribution, 15-trial repeat consistency, image fit, and upload/schema success. **Why:** Demonstrates executable feasibility as Kusztor Stage 1's pilot figures did, without confusing these observations with participants in the confirmatory sample.

## 4. Plan the eventual *results* figures now, but do not populate them as results

These outputs should appear in the preregistration's analysis/visualization plan; observed examples belong in the later article. The existing analysis already saves three key figures and numerous tables.

### O01 — Participant-level coefficient estimates and intervals
- [ ] **Status:** `RESULTS LATER`. Use `figures/weights_per_person.png`; separately for colour and fog show each person's fitted weight, profile interval, **both** registered classification thresholds, and predicted/opposite/undetected labels. If intervals are clipped visually, disclose it.
- **Why:** This displays heterogeneity and prevents the group rule from hiding strongly contrary individual responses.

### O02 — A small H1/H2 verdict table linked directly to the coefficients
- [ ] **Status:** `RESULTS LATER`. Report feature; final N; predicted-direction count; opposite-direction count; undetected count; registered `(k,m)`; supported/not supported/descriptive; short qualification of inference.
- **Why:** Per-person coefficient plots alone cannot show how the registered *study-level* decision was reached. The table is the most direct audit of H1/H2.

### O03 — Participant/control coefficients *and* attention-quality distributions
- [ ] **Status:** `RESULTS LATER`. Use `figures/c1_distributions.png` and `tables/tests.csv`/`quality.csv`; display both feature weights, catch-failure counts and sub-300-ms response shares, group sizes, p-values/Holm interpretation and explicit C1 gate verdict.
- **Why:** A significant group difference without the gate result is incomplete and potentially misleading.

### O04 — Episode-minus-normal grid shifts
- [ ] **Status:** `RESULTS LATER`. Use `figures/grid_shifts.png`, `grid_summary.csv` and `grid_per_person.csv`. Plot signed shift separately for colour and contrast; report **original-endpoint censoring**, “none of these” selections, image-set allocation and pixel-fit diagnostics.
- **Why:** A small shift can reflect a measurement boundary rather than similarity of actual perceptions; the grid is descriptive, not an independent confirmatory H1/H2 test.

### O05 — Two-dimensional balance and secondary description correlations
- [ ] **Status:** `RESULTS LATER`. For H3, prefer a **scatterplot of `(b_sat,b_fog)` with axis zero lines/quadrants** over relying on a single angle ranking. For E1/word/CDS analyses, report scatterplots or compact coefficient tables with intervals, labelled exploratory.
- **Why:** The signed angle and ratio do **not** have identical global rankings when colour weights can be negative. The two-dimensional figure is faithful to the model and avoids overly precise inference from small subgroups. **Issue:** C07.

### O06 — Recruitment and quality accounting
- [ ] **Status:** `RESULTS LATER`. Present a compact participant flow table from `exclusion_counts.csv`/`exclusions_per_session.csv`, plus preplanned sensitivity/quality summaries. Keep names, contact, raw episode notes and linkage keys private.
- **Why:** Transparency about analysed N and removed/partial data is required to judge generalizability and exclusions.

**Already planned/available as script products:** `per_person.csv`, `tests.csv`, `exclusion_counts.csv`, `prevalence.csv`, `secondary_tests.csv`, `suzuki_tests.csv`, `h3_balance.csv`, `e1_correlations.csv`, `dose_response.csv`, `grid_per_person.csv`, `grid_summary.csv`, `quality.csv`, plus the three PNGs above. These are **planned output names**, not a claim that real observations already exist.

## 5. Necessary analytical work and what *not* to rerun

This section prioritizes work needed for accurate preregistration claims and executable reproducibility. **A sensitivity analysis is not automatically a proposal to change the frozen decision rules.**

| Work item | Priority for document | Is code/rerun needed? | Why / acceptable disposition | Issue tracker |
|---|---|---|---|---|
| Correct negative-result prose and outcomes table | **Before final submission** | Documentation | Current non-support cannot “rule out” moderate effects. | C02 |
| Restrict the 5% false-claim wording to actually simulated nulls | **Before final submission** | Documentation | Do not give a universal taste-bias guarantee. | C04 |
| Sensitivity to asymmetric/non-zero-mean nonresponder tastes | **Strongly recommended before final calibration claims** | **Yes, targeted simulation** | Test directional bias and mixture nulls with registered rules unchanged; report limits. | C04, S01 |
| Sensitivity to repeated scenes/within-person dependence | **Strongly recommended before final calibration claims** | **Yes, targeted simulation** | Independent trial draws may miss scene-specific choices or correlated lapses. | S02 |
| C1 simulation implementing catch-based exclusions before gate/group tests | **Before quoting C1 rates as end-to-end power/error** | **Yes, C1 update/rerun** | Current analytic selection and simulation differ. Qualify preliminary C1 estimates if not rerun. | C05 |
| Independent additional runs for selected rule headroom | **Recommended when presenting strong ≤5% numerical certification** | **Optional larger fresh Monte Carlo** | Existing per-cell upper estimates are not simultaneous post-selection coverage; some margins thin. | C11 |
| Exact frozen trial identity and catch-key integrity checks | **Before analysis-script freeze** | **Yes, code + negative tests** | Validate 211 distinct IDs per complete session, expected catches and response-key truth. | C03 |
| Raw browser CSV → analysis reader; both groups and languages | **Before analysis-script freeze and data collection** | **Yes, code + end-to-end tests** | The current `load_raw` is intentionally unfinished. | C03, C01 |
| Align C1 one-sided gate description; repair H3 angle interpretation | **Before final submission** | Mostly documentation; possibly small plotting/report edit | Avoid contradictory specified tests and mathematically false ratio interpretation. | C07, C08 |
| Consent/upload reconciliation and approved bilingual deployed flow | **Before recruitment; document procedure before filing** | **Yes, browser/consent QA** | Avoid false assurance about data retained after closing. | C06, C12 |
| Minor stale implementation/document pointers | **Before repository archival as practicable** | Documentation | Do not let historical notes masquerade as current protocols. | C09, C10 |

**Not required now:** Do **not** automatically rerun the entire unchanged H1/H2 power pipeline simply to draft the figures, replace the preselected `(k,m)` table after observing data, import Kusztor's sequential Bayes-factor stopping rule, or invent empirical pilot results. The registered trinomial count-rule arithmetic has already been independently recomputed from the stated symmetric null classification rates; that verification is **not** itself a fresh simulation of the raw choices or real-world null behaviour.

## 6. How the Kusztor documents inform these recommendations—and where they do not

| Kusztor source | Transferable lesson | Our adaptation / explicit non-transfer |
|---|---|---|
| **Stage 1 pp. 4–7, Figs. 1–3** | Conceptual figure and *simulated* visual examples of alternative outcomes | F01 and F03, with **actual H1/H2 directional classifications and registered `(k,m)` verdicts**. Do not copy their regression or imply their curves are our data. |
| **Stage 1 pp. 8–11, Figs. 4–6** | Display actual manipulated stimuli and task interactions | F02: our forced-choice pairs, contrast/saturation confounds, checks and grids, rather than their 8-point rating and slider tasks. |
| **Stage 1 pp. 13–14, Figs. 7–8 and repeatability supplement** | Feasibility pilots, attention checks and consistency described before collection | R11 and FS03; keep technical/pilot data outside confirmatory analysis. |
| **Stage 1 pp. 14–16** | Fully specified model, priors, exclusions and decision threshold before data | R05–R10: **Firth/logistic classification + count rules + C1 MW/Holm**, not their Bayesian mixed-effects comparison. |
| **Stage 1 pp. 16–18, Figs. 9–10** | Simulate **the planned decision procedure**, displaying positive, null and inconclusive outcomes | F04 and F06. Our H1/H2 outcomes are *support/not support*; **not** evidence for H0. Unlike their registered sequential design, ours is one final look. Note their Fig. 9 disclosed a simplified model that excluded a planned interaction—avoid treating a simplified simulation as the full pipeline. |
| **Stage 1 p. 19, Study Design Template** | Table mapping scientific question, analysis, sample size and interpretation | T01, including C1 attention-gate **inconclusive** status. |
| **Published article (2025), abstract/results and discussion** | Their registered tests mostly favoured null models, and they discuss behavioural judgments, metacognition and El Greco/perceptual-interpretation limits | R01/R08: episode-related memory matching is a **different operationalization**, but the result is not a direct assay of sensory change. |
| **Published article, Methods (realness-rating stimulus exposure)** | They avoid repeating a scene at different manipulation levels within the same rating task | FS02: investigate **our** intentional scene reuse, rather than treating their exposure scheme as compulsory. Their image-adjustment task **does** present images twice as a consistency measure. |

**Published source specifics:** Kusztor et al., *Lifting the veil: probing altered visual perception in derealization*, *Neuroscience of Consciousness* (2025), article `niaf045`, DOI `10.1093/nc/niaf045`. Their reported Ns were **200** (realness rating) and **125** (image adjustment) across the included conditions. These figures are **not** a sample-size prescription for the present study, because both the questions and decisions differ.

## 7. Recommended writing and production order

1. [ ] **Write now:** R01–R06, R08–R10 and first drafts of R11–R13; produce T01/T02, F01/F02. Keep language on questionable operating characteristics conditional.
2. [ ] **Plot committed outputs now:** F04/F05 and optional FS01 using the **registered** rule rows, labelled planning and weaker scenarios; draft F03 with explicit simulated labels. These are data presentation jobs, not reasons to rerun the entire power simulator.
3. [ ] **Resolve confirmatory calibration claims:** symmetric/non-symmetric taste sensitivity, within-scene dependence, optional selected-rule fresh calibration; update captions and limitations (R07, F05, FS02).
4. [ ] **Make C1 operating characteristics reflect the registered whole procedure:** catch exclusion → gate → one-sided group tests; then finalise R09/F06.
5. [ ] **Finish executable QA:** browser v4 deployment, consent/optional-contact wording, bilingual accuracy, strict trial/catch validation, raw CSV importer, complete sessions from both groups, analysis regression tests, checksums/commits (R04–R05, R11–R13).
6. [ ] **Freeze and archive:** store exact manuscript version, complete outputs, code versions and preregistration timestamp **before primary data collection**. Document any necessary deviations transparently; do not conceal changes as if prespecified.
7. [ ] **After collection only:** populate O01–O06 from the registered analysis. Separate confirmatory findings, exploratory observations, calibration sensitivity results and limitations.

## 8. Source and input index

**Current-study primary design and analysis documents**
- [`docs/experimental_design.md`](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/docs/experimental_design.md) — study description, eligibility, hypotheses, trial details, grids, words.
- [`docs/analysis_script_plan.md`](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/docs/analysis_script_plan.md) — fit, classifications, C1 gate, secondary and exploratory outputs; includes historical build prompt.
- [`docs/experiment_code_changes.md`](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/docs/experiment_code_changes.md) — v4 browser implementation and consent/ID plan (some steps pending).
- [`docs/v4_stimulus_power_explained.md`](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/docs/v4_stimulus_power_explained.md) — stimulus/grid design, planning cells, error calibration, C1 and registered N lookup, §§12–14.
- [`stimulus_power_pipeline/simulate_power_v4_n10_20.py`](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/stimulus_power_pipeline/simulate_power_v4_n10_20.py) — relevant original power simulator.
- [`analysis_pipeline/analyse_study_v4.py`](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/analysis_pipeline/analyse_study_v4.py) and [`analysis_pipeline/simulate_study_data_v4.py`](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/analysis_pipeline/simulate_study_data_v4.py) — analysis and validation-data generation.

**Tables and figures to build from**
- [`data_power_v4/registration_v4/combined_rule.csv`](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/data_power_v4/registration_v4/combined_rule.csv) — 22 registered rule rows for N=10–20.
- [`data_power_v4/registration_v4/registration_table.csv`](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/data_power_v4/registration_v4/registration_table.csv) — planning and weaker-scenario power, rates, upper estimates/headroom.
- [`data_power_v4/registration_v4/c1_compact.csv`](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/data_power_v4/registration_v4/c1_compact.csv) — conditional/provisional C1 planning summary.
- [`data_power_v4/registration_v4/c1_cells.csv`](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/data_power_v4/registration_v4/c1_cells.csv) — fuller scenario sweep for C1.
- [`data_power_v4/power_v4_n10_20/frozen/detection_curve.csv`](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/data_power_v4/power_v4_n10_20/frozen/detection_curve.csv) — individual detection sensitivity.
- [`data_analysis/analysis_v4_validation/validation_report.txt`](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/data_analysis/analysis_v4_validation/validation_report.txt) — existing simulated-data validation report (inspected, not newly executed for this guide).
- [`data_trials/frozen_trial_list_v4/package/trial_list.json`](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/data_trials/frozen_trial_list_v4/package/trial_list.json) — frozen trial details and repetition structure.

**Example Registered Report and published context**
- [Kusztor et al., *Stage1Manuscript_final.docx*](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/docs/experimental_design_articles/Stage1Manuscript_final.docx) — esp. Figures 1–10, Stage 1 Study Design Template p. 19, and supplementary El Greco discussion.
- [Kusztor et al., *Lifting the veil* (published PDF)](https://github.com/zarcolmarioa/drl2drv_updated/blob/master/docs/experimental_design_articles/Kusztor%20et%20al%20-%20Lifting%20the%20veil%20probing%20altered%20visual%20perception%20in%20derealization.pdf) — final Registered Report, results and methodological limits.

## 9. Evidence boundaries

This guide draws on inspected repository sources, the **attached** Stage 1 DOCX and final article PDF, and the previous independent technical review. The registered count-rule probabilities were previously checked by direct trinomial arithmetic conditional on the supplied symmetric null-classification rates. **For this guide, I did not execute new power simulations, rerun end-to-end browser sessions, inspect real participant responses, or newly validate the preregistration-ready consent text.** Existing simulated validation outputs are records of earlier runs, not evidence that the unfinished real-CSV reader or v4 browser update is already working.

**What is sound enough to draft around:** The frozen trial-count design, the preselected rule table and its arithmetic under the stated null assumptions, and the planned confirmatory versus exploratory separation are a coherent basis for a preregistration, provided the above qualifications and remaining implementation checks are resolved before the final protocol is frozen.
