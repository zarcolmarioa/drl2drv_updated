# Stimulus pipeline and power analysis: changes before preregistration

## How we will work

This is our usual working pattern:
- You write one script at a time.
- I run it locally.
- You tell me which output files to upload so you can check that the step worked.
- We move on only once the checks pass.

New scripts follow the same conventions as the scripts we have written together:
- a header that explains where the script sits in the pipeline and why;
- a settings block at the top with commented choices;
- a report file;
- built-in checks;
- figures or contact sheets where a check is visual;
- outputs written to a new folder rather than over an earlier run.

Only the scripts and the documentation are kept long-term. Result files stay on my computer. So the documentation has to record what was run and the key outcomes.

Read this whole plan first. Then, in this reply only:
1. Check that you have every file listed under "Files needed". Tell me which ones are missing, and I will upload them.
2. Confirm your understanding of the plan: summarise each step in your own words, and list anything that is unclear, that seems wrong, or that conflicts with how the scripts currently work.

**Do not write any script yet.** I will answer your questions and then ask you to proceed in a later message. When I do, write only the first script (Step 1), and do not write later steps until I ask.

## Background: why these changes

A review of the stimulus set, the trial list and the power analysis found several problems that need fixing before the study is preregistered. They fall into seven groups.

1. **Colour is harder to detect than fog.** In the trial list, colour differences are small relative to each scene's colour range:
   - colour-only pairs stay within one fog level;
   - in mixed pairs, haze also drains colour.

   So a participant whose choices follow colour is detected much less often than one whose choices follow fog. Two changes should help: add more-vivid colour levels above the original, and draw the colour-only trials from the larger colour differences. These need testing before we adopt them.
2. **Units of the weights.** Each pair's differences are divided by that scene's range on the grid. So a weight on fog and a weight on colour are not in comparable units. Changing the grid (for example, adding levels) also changes what a given weight means. Effect sizes in the power analysis must therefore be defined in raw feature units that do not depend on the grid.
3. **Predictor signs.** The trial list stores the colour and fog differences as unsigned magnitudes. The power scripts sign them correctly, but the documentation describes them as signed and gives a model that uses them as they are. An analysis written from the documentation would be wrong.
4. **The existence claim assumes non-responders choose at random.** Real people without the effect may still have small personal preferences in either direction: some like hazier pictures, some clearer ones. Counting detections in one direction only can then produce a false claim. Each person should be tested in both directions, and the decision rule should require detections in the predicted direction clearly to outnumber those in the opposite direction.
5. **A "pick the more edited picture" strategy.** A participant who reasons "derealization looks unreal, so choose the more altered image" gives positive weights on both features and a curve that rises with difference size. So a rising dose-response does not rule out a response strategy, and the documentation must stop claiming that it does. Whether the effect is specific to the experience rests on a control group, which will be added to the design. It must also be added to the simulations.
6. **Simulation realism.**
   - Simulated participants never lapse.
   - Effect sizes were carried over from an earlier design.
   - The "stop at a certain sample size" advice in the documentation comes from a jump in a fixed count rule, not from a real loss of power.
   - The Bayes factor figures quoted in the documentation are two-sided, while the plan is one-sided.

   There will be no pilot, so lapses and effect sizes are shown as ranges of scenarios.
7. **A second grid block.** The experiment will get a "normal appearance" grid block: which column looks most like how the surroundings normally look. To measure each person's shift from "normal" to "episode", both grid blocks need levels above the original, the original itself, and the existing edited levels.

## Files needed

Scripts and documents (check which ones you already have, and list the ones I need to upload):
- stimulus scripts:
  - `calibrate_levels_v4.py`
  - `select_scenes_v4.py`
  - `make_screening_page.py`
  - `build_stimulus_set_v4.py`
  - `analyse_pairs_v4.py`
  - `build_trials_v4.py`
- power scripts:
  - `simulate_power_v4.py`
  - `simulate_power_seq_v4.py`
  - `plot_power_seq_v4.py`
- documents:
  - `v4_stimulus_pipeline_explained.md`
  - `v4_power_analysis_explained.md`
  - `experimental_design.md`
- any shared module these scripts import.

Output files from earlier runs, when a step needs them (ask me at that step, not before):
- the run reports of the current stimulus build, pair analysis and trial list;
- the current `trial_list.json`;
- an existing `grid_trials.json` from the experiment, so the grid format matches.

## The plan, in pipeline order

### Step 1. Calibration: levels above the original (`calibrate_levels_v4.py`)

What to change:
- **More-vivid colour levels.** Add two colour levels above the original, calibrated the same way as the existing ones (target corpus percentiles of the colour feature, this time above the median). Choose the spacing to match the existing steps, and explain your choice.
- **Make chroma above 1 work.** The chroma edit needs the maximum in-gamut chroma per pixel for multipliers above 1. The current context function does not supply it. Compute it from the image's lightness and hue with the existing gamut function, after the haze step, because haze is applied first.
- **Contrast levels for the grid blocks.** Add a level with no haze (the original) and one or two levels with more contrast than the original. Use the inverse of the same haze model if that works, so the feature moves along the same path, and report clipping. These are for the grid blocks only, not for the pairs.
- **Missing percentiles.** Write the percentiles that the band sweep in step 2 looks up but that are currently missing.
- **Defaults and comments.**
  - Set the script's defaults to the configuration of the final stimulus set: the equal-luminance haze family, and the paths of the final run.
  - Update comments that contradict the final run.
  - Describe the fog manipulation as "contrast reduction (haze model), used to model the fog complaint".

Checks:
- The existing levels are unchanged.
- Each new level moves the measured feature in the right direction for a typical image.
- Gamut clipping at the vivid levels is reported.

### Step 2. Scene selection report (`select_scenes_v4.py`, `make_screening_page.py`)

What to change:
- **Band sweep.** When a percentile column is missing, the band sweep currently skips the feature filter silently. So most rows of the table in the report and the documentation are wrong. Fix the lookup, raise an error when a column is missing, and rerun **only the sweep**.
- **Screening tool path.** Point the screening tool at the final folder.

Check: the selected scenes are exactly the same as before.

### Step 3. Render the new levels (`build_stimulus_set_v4.py`)

What to change:
- **Render the new levels.** Render the new colour levels for every scene and every fog level. Render the contrast levels for the grid blocks.
- **Mark each image's use** in the manifest: pair grid, grid block only, or both.
- **Default configuration.** Build only the equal-luminance haze set by default, and keep the other family as a commented alternative.

Checks:
- Existing images are reproduced pixel-identical.
- The colour feature increases with each colour level in every scene, and no scene's steps collapse because of clipping.
- Contact sheets of a few scenes at the vivid levels, so I can check whether they look artificial.

### Step 4. Pair analysis (`analyse_pairs_v4.py`)

What to change:
- **Two runs.** Run on the original levels only, and on the grid extended with the vivid colour levels.
- **Oriented predictors.** Add one column for colour and one for fog: positive when image A is the one predicted to be chosen (less colourful, or lower contrast). Also keep the raw signed feature differences.
- **Defaults and comments.** Update the defaults and stale comments.

Check: the original-levels run reproduces the existing pair table.

### Step 5. Trial lists (`build_trials_v4.py`)

What to change:
- **Option to change the colour-only sampling.** Add a setting that draws the colour-only trials from the larger colour differences, instead of evenly across magnitude bins. Let me choose where the cut sits, and report how much smaller-difference range is lost (this affects the exploratory dose-response).
- **Practice trials.** At the moment they are drawn only from unused pairs where one image is both hazier and less colourful. So every practice trial shows a "more edited versus less edited" contrast, which may nudge participants toward picking the more edited image. Draw them instead from a mix of pair types at small-to-medium differences, still from pairs the analysed trials do not use. Catch trials can stay as they are.
- **Oriented predictors.** Write the oriented predictor columns into `trial_list.csv` and `trial_list.json`.
- **Header.** Update the header and comments to the current trial counts.
- **Four versions**, everything else identical and the same seed:
  - **current:** original levels, current sampling;
  - **extra levels only:** vivid levels, current sampling;
  - **larger differences only:** original levels, new sampling;
  - **both changes:** vivid levels and new sampling.

Checks:
- The current version reproduces the existing trial list.
- For each version, the summary tables and difference-space figure.

### Step 6. Choose the colour condition (new script, alongside the power scripts)

Goal: compare the four trial lists on how often a participant with a **fixed** sensitivity is detected.

Use the existing per-person fitting and threshold functions from the power scripts. Do not wait for the Step 8 changes: the comparison only needs the per-person parts, which those changes do not alter.

Two rules make the comparison fair:
1. **Raw units.** Fix each simulated participant's sensitivity in raw feature units (log-odds per raw difference in the colour feature and in the contrast feature), never in the grid-scaled units. Pick several sensitivity values that give detection rates well away from floor and ceiling on the current list.
2. **Thresholds per version.** Compute each version's thresholds from random choosers on that version's own trial list.

Report, for each version:
- **detection:** colour detection at each sensitivity, and the same for fog, so I can see that fog does not get worse;
- **false detections:** the false-detection rate of random choosers, as a check;
- **ranges:** the range of colour and fog differences used;
- **trade-offs:** for example, larger colour edits also look more edited.

Then **stop**. I will choose the version and tell you.

### Step 7. Freeze the trial list, and build the grid blocks

**Freeze the chosen trial list.**
- Rebuild it once with the chosen settings.
- Compute a checksum of `trial_list.json` and of the package.
- Write the checksum, the settings and the seed into the stimulus documentation.
- Tell me which files to keep: I need the same package for the experiment code.

From here on, every script that reads the trial list verifies its checksum.

**New script for the grid blocks.** It writes the grid files the experiment reads, in the same format as the existing `grid_trials.json`. It covers:
- **two blocks:** the episode block and the normal-appearance block;
- **same columns in both blocks**, per feature: levels above the original, the original, and the edited levels. This lets us measure each person's shift from "normal" to "episode" in grid steps;
- **forward and reversed column order** for each feature, as now;
- **two matched scene sets**, assigned to the two blocks and swapped between participants, so nobody sees the same photograph in both blocks.

The experiment code itself is handled elsewhere. Tell me what it will need to change to run two grid blocks and to read the new trial-list columns.

### Step 8. Power analysis (`simulate_power_v4.py`, `simulate_power_seq_v4.py`, `plot_power_seq_v4.py`)

Run everything on the frozen trial list, after verifying its checksum.

What to change:
1. **Read the oriented predictors.** Read them from the trial list rather than signing them inside the script, and assert that they match the script's own signing. Keep the per-person fitting function in a form the future analysis script can import unchanged. The registered thresholds are only valid if the analysis fits exactly the same model.
2. **Two-sided per-person thresholds.** Compute an upper and a lower threshold per feature from the same random choosers, at the same percentile the current threshold uses, one at each end. Classify each person per feature as detected in the predicted direction, detected in the opposite direction, or not detected.
3. **Taste scenario.** Non-responders have small preferences centred on zero, at several spreads, in addition to the pure random chooser.
4. **Candidate decision rules** of the form "at least k detections in the predicted direction and at most m in the opposite direction". For each rule and sample size, report:
   - the false-claim rate under random choosers and under each taste spread;
   - the power under each effect scenario.

   Report this per feature, and for the two claims (colour, fog) together. Also compute population prevalence inference (Ince et al., 2021, eLife, "Bayesian inference of population prevalence") as an alternative to a fixed count. **Do not pick the rule.** I will decide it separately and come back to you.
5. **Effect sizes in raw units.** Define them as in Step 6, and translate each scenario into something readable, such as "chooses the lower-contrast image X% of the time at a typical difference". Keep the responder-share and spread scenarios.
6. **Lapses.** Add several lapse rates (a share of trials answered at random), from none to a high value that would still pass the catch and consistency checks.
7. **Control group.** Controls do the same task, imagining how the world looks to someone experiencing derealization. Simulate them with weights drawn from a "shared concept" distribution, which includes a "pick the more edited image" strategy (positive on both features). Compare participants' per-person weights with controls' weights, one-sided. Report power across numbers of participants and controls.
8. **Group test.** Run it on the per-person weights with a **one-sided** Bayes factor, and regenerate the null-evidence figures (evidence against, inconclusive) with that same one-sided factor.
9. **Sample sizes.** Extend the range past the point where the fixed count rule jumps, and show what extra participants gain. Remove the advice to stop early because of the jump.
10. **Tidy up.**
    - Update the stale headers, for example the ones still calling the proportion test confirmatory.
    - Mark the current trial count in the trial-count sweep.
    - Align the output folder of the simulation with the input folder of the plot script.

Checks:
- With pure random choosers, the per-person false-detection rate at each threshold is close to the nominal rate.
- Rerunning the old one-directional rule on the current settings reproduces the old figures.

When the rule is chosen, run once more with that rule. That run's results go into the preregistration.

### Step 9. Documentation (alongside each step)

**`v4_stimulus_pipeline_explained.md`:**
- the new levels, and how they were calibrated;
- the corrected band-sweep table;
- the description of the fog manipulation as contrast reduction;
- the statement about display rescaling, reworded: the within-pair ratio is preserved only under linear rescaling, and display gamma is not linear;
- the grid blocks;
- the frozen list's checksum and settings.

**`v4_power_analysis_explained.md`:**
- the units of the weights, and why fog and colour weights cannot be compared within a person;
- the oriented predictors and the exact model;
- remove the claim that a rising dose-response rules out response strategies;
- the both-direction rule;
- the taste, lapse and control scenarios;
- the one-sided Bayes factor figures;
- the sample-size discussion.

**`experimental_design.md`:** fix only the model statement and the predictor signs for now. The rest will be rewritten after the hypotheses are finalised.

At the end of each step, give me:
- a short plain-text summary: what changed, which files were written and where, and each check with pass or fail;
- the list of files to upload.
