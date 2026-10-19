# drl2drv review: Axes pipeline

Exported 2026-10-15. Technical, methodological and results assessment. Every issue is listed, resolved or not, with the outcome of the discussion and your notes.

18 issues, 6 resolved. Severity: 0 critical, 6 major, 12 minor.

| ID | Issue | Severity | Outcome | Resolved |
|---|---|---|---|---|
| AX-T1 | Stage C as committed doesn't reproduce the saved captions | minor | Proposed | no |
| AX-T2 | Stage C word check lets a meaning-reversing edit through | minor | Open | no |
| AX-T3 | Stage V reads its input from a folder that isn't in the repo | minor | Agreed | yes |
| AX-T4 | Documentation disagrees with the code in four places | minor | Open | no |
| AX-M1 | The corpus links measure CLIP's photo vocabulary, not what derealization looks like | major | Agreed | yes |
| AX-M2 | The colour link of vividless (A13) rests mostly on CLIP's reading of the word "vivid" | major | Agreed | no |
| AX-M3 | Both experiment axes lean towards darkness, and the stimuli leave darkness out | major | Agreed | yes |
| AX-M4 | One person made all the judgement calls, with no second rater | major | Limitation | no |
| AX-M5 | The lighting rule for negatives isn't applied consistently | minor | Agreed | no |
| AX-M6 | The reliability figures are inflated by repeated contrast words | minor | Agreed | no |
| AX-M7 | Some control checks partly control for the construct itself | minor | Agreed | no |
| AX-M8 | The feature set was cut after axis results had been seen | minor | Open | yes |
| AX-M9 | CLIP doesn't read "less", "lacking" or "un-", and the negation check didn't test them | minor | Proposed | no |
| AX-R1 | The corpus predicts a difference in weight ratio, not a crossover | major | Agreed | no |
| AX-R2 | The fog manipulation used in the stimuli wasn't shown to be specific | major | Agreed | no |
| AX-R3 | The darkness sensitivity check is itself confounded with colour | minor | Open | yes |
| AX-R4 | The link sizes are modest, and the "link" threshold is lenient | minor | Agreed | no |
| AX-R5 | The image-side numbers can't be checked from the repository | minor | Open | yes |

## Technical assessment

### [ ] AX-T1. Stage C as committed doesn't reproduce the saved captions

**Status:** not resolved · severity minor · confirmed · outcome: Proposed

**Where:** `axes_pipeline/v4_stageC_check.py, lines 115 (USE_FLUENCY) and 119 (ACCEPT_SUGGESTIONS)`

**What's wrong:** The committed file has USE_FLUENCY = False and ACCEPT_SUGGESTIONS = False, but the saved Stage C outputs were made with both switched on. I re-ran the script on the repository's own files: 105 of the 290 captions come back as "open", because the 105 rows decided by accepting a suggestion are stored with an empty decision. The support table changes too (fog, A07: 33 → 23 captioned texts; vividless, A13: 11 → 9).

**Why it matters:** A rerun as committed overwrites stageC_captions.csv with 105 blank captions. Stage D's prepare script then stops, so nothing downstream is silently corrupted, but the pipeline can't be reproduced from the code, and the review file doesn't say which captions were chosen by rule and which by you.

**Suggested fix:** Set ACCEPT_SUGGESTIONS = True and record that USE_FLUENCY was True for the final run, or better, write the accepted suggestion into the decision column so the result no longer depends on the flags. Save the settings in check_meta.json and re-run once to confirm identical output.

**Outcome of the discussion:**

> Your note (15 Oct): this stage was run in another session; the script was edited several times afterwards, and the last version may never have been run.
> 
> Proposed resolution: no rerun needed. Set ACCEPT_SUGGESTIONS = True and USE_FLUENCY = True in the committed file, with a comment that these are the settings of the final run, so the file matches the saved outputs. Optionally run it once to confirm the output is identical.

**Your notes:**

> I remember I asked the other session to do this for me. So, there was no need to run it again. I also remember this script was updated a few times. I might have run the previous version with the flags and the latest version was only corrected but never run.

### [ ] AX-T2. Stage C word check lets a meaning-reversing edit through

**Status:** not resolved · severity minor · confirmed · outcome: Open

**Where:** `axes_pipeline/v4_stageC_check.py, same_root() (lines 254–261)`

**What's wrong:** The prefix rule treats "colorless" and "colored" as forms of the same word (the stem "colorles" starts with "color"). A caption that turns "colorless" into "colored" passes the word check and isn't flagged as a deleted content word. I tested this directly with the script's own check().

**Why it matters:** The check exists to stop the models changing meaning. No final caption is affected (I went through all 24 captions with form changes), so the bug is latent, but it would let a reversal through on any rerun with new model outputs.

**Suggested fix:** Exclude pairs that differ by a negating affix (-less, -free, un-, in-, non-) from same_root(), or reject a caption form that drops a negating suffix present in the quote.

**Outcome of the discussion:**

> Not discussed.

**Your notes:**

> None.

### [x] AX-T3. Stage V reads its input from a folder that isn't in the repo

**Status:** resolved · severity minor · confirmed · outcome: Agreed

**Where:** `axes_pipeline/v4_stageV_verify.py, line 112 (STAGEA_DIR = "stageA_v4")`

**What's wrong:** The 34 Stage A files are committed in axes/v4/stageA, not stageA_v4. The docstrings say to run from drl2drv_updated/, but every relative path in the axes scripts (PDF_DIR, axes/v4/..., data/...) only resolves from inside axes_pipeline/. With the path corrected, the script reproduces stageA_verified.csv exactly (1,718 phrases; I re-ran it against the PDFs).

**Why it matters:** Someone rerunning from the repo gets "Missing Stage A files" or wrong paths.

**Suggested fix:** Point STAGEA_DIR to axes/v4/stageA and change the "run from" note to axes_pipeline/ in every script.

**Outcome of the discussion:**

> Your note (15 Oct): the paths may need updating if Stage V is rerun. Agreed: update STAGEA_DIR and the working-folder note at that point.

**Your notes:**

> Although confirmed, the paths might need to be updated if we are to run this again.

### [ ] AX-T4. Documentation disagrees with the code in four places

**Status:** not resolved · severity minor · confirmed · outcome: Open

**Where:** `docs/v4_axes_pipeline_explained.md §2.4, §2.6, §3.3; axes/v4/stageV/stageV_report.txt`

**What's wrong:** (a) §2.4 says Stage C scores fluency with GPT-2; the code has it switched off (USE_FLUENCY = False, "dropped 6 Oct"). (b) §3.3 says 04c's OUTPUT_DIR is data/features_extra_v2; the code says data/features_extra, with a comment calling that version 1's folder. (c) §2.6 quotes darkness leans of 0.32 / 0.35 / 0.41 for fog / vividless / lifeless. Those are the pre-revision values (0.318 / 0.345 / 0.414); on the final D2 axes they are 0.318 / 0.355 / 0.433 (I recomputed them from the cached CLIP vectors). (d) Stage V's report lists 9 overlapping spans as "not merged", although all 9 were merged by review.

**Why it matters:** Small, but a reader who checks one of these will doubt the rest.

**Suggested fix:** Correct the four statements; make clear which 04c folder holds version 2.

**Outcome of the discussion:**

> Not discussed.

**Your notes:**

> None.

## Methodological assessment

### [x] AX-M1. The corpus links measure CLIP's photo vocabulary, not what derealization looks like

**Status:** resolved · severity major · confirmed · outcome: Agreed

**Where:** `docs/v4_axes_pipeline_explained.md §1, §7, §8; 07_category_confounds_v4.py; 09a C1`

**What's wrong:** CLIP learned "foggy" from photographs of real fog and "vivid" from saturated photographs. Several fog positives are metaphors ("as if seen in the fog" from "the dpdr fog", "as if navigating through fog in slow motion", "as if blinded by a mist"), but the template "a scene that looks …" makes CLIP read them as literal weather. The checks show this: the "foggy weather" control absorbs fog's saturation link (−0.30 → −0.01).

**Why it matters:** The axis–feature correlations show which image features CLIP associates with the words patients use. They are not evidence that derealization looks like low local contrast or low saturation. As a principled way to choose manipulations they are fine; claims beyond that overreach, and the experiment is what tests the link.

**Suggested fix:** Present the axes pipeline as a translation of patient vocabulary into candidate image manipulations, and keep claims about perception for the experiment. State the literal-reading limitation in the methods.

**Outcome of the discussion:**

> Agreed (15 Oct). Use the suggested fix: present the axes pipeline as a translation of patient vocabulary into candidate image manipulations, leave claims about perception to the experiment, and state the literal-reading limitation in the methods.

**Your notes:**

> None.

### [ ] AX-M2. The colour link of vividless (A13) rests mostly on CLIP's reading of the word "vivid"

**Status:** not resolved · severity major · confirmed · outcome: Agreed

**Where:** `axes/v4/stageD2/axes_v4.yaml (vividless, A13); axes/v4/stageB/stageB_final_axes.csv; 09b pole results (doc §6.2)`

**What's wrong:** 7 of the 10 negatives of vividless (A13) contain "vivid" or "vividness". In 4 of them the patient's own phrase contains it too ("less vivid", "less vivid or intense", "like things lacking vividness", "lacking in normal color and vividness"). Only 3 negatives add "vivid" where the patient's phrase didn't: "dull → vivid", "duller → vivid", "as if seen with attenuated perception → as if seen with vivid perception". The larger effect is how CLIP reads the patient phrases: it largely ignores "less" and "lacking". Along the axis, "less vivid", "less vivid or intense" and "like things lacking vividness" sit at the midpoint of all captions (−0.01 to 0.00), against +0.12 to +0.15 for "dull", "duller" and "colorless"; their pairs work only through the "vivid" side. This fits the pole results of 09b: the saturation link comes mostly from the negative pole (−0.34) and less from the positive pole (−0.15). The positives are also only weakly visual: 25% vision texts, 2 first-person phrases, 43% from one document (Pienkos 2022), and the keeping rule is passed only with all sense classes pooled.

**Why it matters:** The colour arm of the experiment rests on vividless (A13). Its link with saturation is carried mostly by CLIP's reading of "vivid", and less by the phrases in which patients describe a loss ("dull", "colorless", "reduced in intensity").

**Suggested fix:** Run the rebuild test (prompt in the outcome below) and report how much of the saturation link survives each variant. Report the pair-level β from 09b (pair_table.csv), which needs no midpoint. State the thin support plainly.

**Outcome of the discussion:**

> Correction (15 Oct): the first version of this issue said the 7 "vivid" negatives were all words the LLMs chose. In 4 of them the patient's phrase already contains "vivid"; only 3 add it. The issue text is corrected and the mechanism added (CLIP ignores "less" and "lacking"; see AX-M9).
> 
> The midpoint (15 Oct): any midpoint splits the axis into two poles that add up to the full axis, so where it sits decides how a link is shared between them. The average of all 364 captions is reasonable but not neutral, so the pole numbers are relative. Tests that need no midpoint are preferable.
> 
> The 32,061 images (15 Oct): 09b scores well over a thousand caption, pair and pole directions, each with 1,000 category resamples; it uses step 04's stratified sample (about 100 colour images per category) to keep this manageable. Each category then counts about equally, whereas the full-corpus β of −0.40 weights large categories more.
> 
> Is the test still needed? (15 Oct) Yes for vividless. Text-only measurements (no images needed): the 3 pairs whose negative adds "vivid" make up 40% of the vividless axis (30% would be their equal share), and without them the axis changes noticeably (cosine 0.911 with the published one). The 3 pairs with "less" or "lacking" make up 21%, and without them the axis barely changes (cosine 0.982). So the open question is whether the saturation link survives without the added "vivid" negatives, and that needs the images.
> 
> Agreed (15 Oct): run one test for both axes. It replaces the earlier vividless-only prompt and also covers AX-M5 (fog's lighting negatives) and AX-M9.
> 
> Prompt for the caption-variant test (for a session that has the data/ folders):
> 
> Write a standalone script, test_axis_variants_v4.py, in the style of the v4 pipeline (SETTINGS block, no command-line arguments, its own output folder, a report and run_meta.json). It rebuilds the fog axis (A07, key "v4::A07 fog, blurry, hazy") and the vividless axis (A13, key "v4::A13 dull, less vivid, colorless") in several variants and repeats step 07's within-category model for each, to show which caption pairs carry their links with local contrast and saturation.
> 
> Build every variant from the pairs in axes/v4/stageD2/axes_v4.yaml, in the template "a scene that looks {}", encoded exactly as 05_build_axes_v4.py does (encode_hf). The axis is unit(mean of the pair differences), except in "equal".
> Vividless (A13):
>   V0        the published axis (check: cosine with its vector in data/axes_v4 >= 0.9999)
>   no_vivid  without the 3 pairs whose negative adds "vivid": A13-01, A13-03, A13-04
>   swap      those 3 negatives replaced by Stage D candidates that passed the filters: A13-01 -> "as if seen with full perception", A13-03 and A13-04 -> "colorful"
>   no_lack   without the 3 pairs whose positive CLIP barely reads: A13-08, A13-09, A13-10
>   equal     all pairs, each pair difference scaled to unit length before averaging
> Fog (A07):
>   V0        the published axis (same check)
>   no_light  without the 7 pairs whose negative adds lighting: A07-01, A07-03, A07-05, A07-06, A07-09, A07-10, A07-28
>   no_lack   without A07-04, A07-23, A07-28, A07-29, A07-30
>   equal     as for vividless
> 
> For each variant report: its cosine with V0; the correlation of its image scores with V0's over all colour images (data/embeddings); and step 07's within-category model on colour images with the 12 features of data/feature_selection_v3: within-category beta for sat_mean, contrast_local, sat_std and lum_mean, with 95% intervals from 1,000 resamples of whole categories (the same resamples for every variant). Reuse 07's functions (cat_stats, pooled, joint_from) rather than rewriting them. Write variants.csv, betas.csv and a short report on how much of each axis's key link (fog x contrast_local, vividless x sat_mean) survives in each variant.

**Your notes:**

> None.

### [x] AX-M3. Both experiment axes lean towards darkness, and the stimuli leave darkness out

**Status:** resolved · severity major · confirmed · outcome: Agreed

**Where:** `axes/v4/check17_D2/check17_report.txt; doc §2.6; definition of vividless (A13) in axes_v4.yaml`

**What's wrong:** Cosine with the darkness direction on the final axes: fog (A07) 0.32, vividless (A13) 0.36 (9 of its 10 pairs lean dark), lifeless (A14) 0.43. That is as large as the enclosure lean of veil (A05) (0.31) that triggered a revision. Check 17 only asks whether particular suspect words cause a lean, so a lean built into the quality itself is never judged; it was accepted. The definition of vividless (A13) includes "dimmed". The stimulus fog (haze_eq) restores mean brightness and the chroma change keeps L*, so no stimulus is darker than its original.

**Why it matters:** If participants remember their surroundings as dim, the stimuli can't show it. The corpus links also partly run through darkness: with the darkness and enclosure directions removed, fog × saturation falls from −0.30 to −0.13 and vividless × saturation from −0.40 to −0.30.

**Suggested fix:** Decide explicitly about brightness: either justify leaving it out and list it as a limitation, or add a darker level to the design (or to the grid ratings). Report the nocontrol results beside the main ones, as planned.

**Outcome of the discussion:**

> Explained in the chat on 15 Oct: what check 17 computes (control directions built from sentence pairs; lean = cosine between the axis and a control direction) and why veil (A05) was revised. Both experiment axes also lean towards dimness at image level (β with lum_mean: fog −0.10, vividless −0.09).
> 
> Agreed (15 Oct): the report states that brightness was kept constant on purpose, to isolate colour and fine detail. The darkness results of check 17 may not be included in the final report. Issue closed.

**Your notes:**

> None.

### [ ] AX-M4. One person made all the judgement calls, with no second rater

**Status:** not resolved · severity major · confirmed · outcome: Limitation

**Where:** `stageV_review.csv; stageB_review.csv; stageB_axes.csv; stageB_clip_selection.csv; stageC_review.csv; stageD_review.csv; stageD2_revision.csv`

**What's wrong:** Agreement is reported between the two LLMs (kappa 0.77–0.92; adjusted Rand 0.70), but every human decision that settled their disagreements was made by one reviewer: 29 first-person disputes (all decided "no"), 128 Stage B texts, the axis definitions, the CLIP selection, 202 Stage C rows (45 dropped as not_scene), 243 Stage D negatives and the Stage D2 drop and replace lists. Some calls are close: vividless (A13) was kept for CLIP although it fails the vision-only rule, while far away (A10), altered size (A12) and strange (A16) were dropped as "doubtful".

**Why it matters:** The final axes depend on these decisions, and a reviewer will ask how reproducible they are.

**Suggested fix:** Have a second person independently redo a random subset (for example 20–25% of the Stage B reviews, the Stage C not_scene drops and the Stage D negatives for fog (A07) and vividless (A13)) and report agreement. Write the reason for each CLIP-selection call into stageB_clip_selection.csv.

**Outcome of the discussion:**

> Decided (15 Oct): report as a limitation. Possible wording: "All human review decisions in building the axes (resolving disagreements between the two models, defining the axes, choosing captions and negatives) were made by a single reviewer, and their reliability was not assessed."

**Your notes:**

> None.

### [ ] AX-M5. The lighting rule for negatives isn't applied consistently

**Status:** not resolved · severity minor · confirmed · outcome: Agreed

**Where:** `axes_pipeline/v4_stageD2_revise.py, lines 121–144; fog (A07) pairs in axes/v4/stageD2/axes_v4.yaml`

**What's wrong:** Stage D2 bans light words (daylight, lit, light …) from every new negative and replaced the veil pair A05-06 because its negative "brings in light". The fog axis (A07) still keeps 7 negatives that add lighting: "as if seen in daylight" (twice), "as if seen in the daylight", "as if lit by daylight", "as if objects were surrounded by daylight", "as if navigating through daylight in slow motion" and "shadowy → well-lit", where the positive is itself a darkness word. The people rule has the same gap: the lifeless pair A14-11 ("an inhabited space") was kept while lifeless pairs A14-01 and A14-09 were replaced for bringing in people.

**Why it matters:** The effect on the fog axis is small (without the 7 pairs: cosine 0.97 with the full axis, darkness lean 0.30 instead of 0.32), but the rule isn't applied evenly, which a reviewer will notice.

**Suggested fix:** Apply the light rule to fog (A07), or record why fog is exempt. The "no_light" variant of the test in AX-M2 shows what changes without the 7 pairs (on text alone they make up 29% of the fog axis, and the axis without them has cosine 0.969 with the published one).

**Outcome of the discussion:**

> Agreed (15 Oct), to do: apply the light rule to fog (A07), or record why it is exempt. The "no_light" variant of the caption-variant test (AX-M2) shows what changes.

**Your notes:**

> None.

### [ ] AX-M6. The reliability figures are inflated by repeated contrast words

**Status:** not resolved · severity minor · confirmed · outcome: Agreed

**Where:** `05_build_axes_v4.py split-half (doc §4.1); 09b pair alpha (doc §6.2)`

**What's wrong:** Split-half 0.95 (fog, A07) and 0.94 (vividless, A13), and pair alpha 0.93 / 0.94, measure whether halves of the caption list point the same way in CLIP space. Many pairs share the same contrast word: vividless (A13) has "vivid" in 7 of 10 negatives; fog (A07) has "clear" in 9 of 29 negatives and "daylight" in 6.

**Why it matters:** Halves drawn from such a list agree almost by construction. The numbers show the text direction is stable, not that it measures the construct; calling them "reliability" invites the wrong reading.

**Suggested fix:** Call them stability of the text direction. Add a leave-one-word-out check (drop all pairs sharing a contrast word) to show how much one word carries.

**Outcome of the discussion:**

> Agreed (15 Oct), to do: rename split-half reliability and pair alpha as "stability of the text direction" in the documentation and the article.

**Your notes:**

> None.

### [ ] AX-M7. Some control checks partly control for the construct itself

**Status:** not resolved · severity minor · confirmed · outcome: Agreed

**Where:** `09a_axis_checks_v4.py C1; doc §6.1 and §8 (Argument 1)`

**What's wrong:** The weather control is "a scene in foggy weather" vs "a scene in clear weather", close to the fog axis (A07) itself. When it absorbs fog's saturation link, the doc reads this as saturation being a side effect of fog. But removing "foggy weather" removes most of what the fog axis is, so the same reasoning would remove any of its links. The quality control ("a bad photo") overlaps blur in the same way (+1.37 SD in the sweep).

**Why it matters:** "Survives all controls" and "absorbed by a control" are used as evidence for the fog–contrast pairing. With near-duplicate controls these results are hard to read in either direction.

**Suggested fix:** Report the weather control separately and don't use it as evidence for the pairing. Base the argument on controls that are clearly different constructs (mood, threat, greyscale, age).

**Outcome of the discussion:**

> Explained in the chat on 15 Oct: the weather control ("a scene in foggy weather" vs "clear weather") is close to the fog axis itself, so adding it soaks up any genuine fog link; its absorption of fog × saturation (−0.30 → −0.01) is not evidence that saturation is a side effect. Fog × local contrast survived even this (−0.15).
> 
> Agreed (15 Oct): the weather control will not be reported. The pairing argument rests on the other controls and the manipulation sweep.

**Your notes:**

> None.

### [x] AX-M8. The feature set was cut after axis results had been seen

**Status:** resolved · severity minor · suspected · outcome: Open

**Where:** `04b_select_features.py line 170 (EXTRA_CANDIDATES); 07_category_confounds_v4.py docstring, lines 28–32`

**What's wrong:** 04b says the CLIP axes are never loaded, so the selection can't be tuned to the results. But the 04c candidates were cut from six features to two (hue_spread, cast_b) after the 16-feature runs of step 07 (the v3 run and the first v4 run) had produced axis results. The four were dropped after a visual check of their extreme images.

**Why it matters:** The visual check is a fair reason, but the order of events gives a reviewer a forking-paths argument.

**Suggested fix:** Record when and why the four were dropped, and report the key experiment links under the 16-feature model as a sensitivity row (the superseded run in data/confounds_v4 already has them).

**Outcome of the discussion:**

> Not discussed.

**Your notes:**

> The order of events will not be reported, only the final set of features we used.

### [ ] AX-M9. CLIP doesn't read "less", "lacking" or "un-", and the negation check didn't test them

**Status:** not resolved · severity minor · confirmed · outcome: Proposed

**Where:** `v4_stageD_prepare.py line 65 and v4_stageD_check.py lines 162–163 (NEGATION); decision D9; axes/v4/check17_D2/check17_cache_clip.npz`

**What's wrong:** Stage D's check D9 dropped pairs whose positive contained an explicit negation word (not, no, nothing, without …) when CLIP couldn't separate them. Words and prefixes that express a lack weren't covered. CLIP barely separates these positives from their negatives: "like things lacking vividness" vs "like things having vividness" cosine 0.982, "less vivid" vs "vivid" 0.963, "indefinite" vs "definite" 0.968, "as if its clarity is hindered" vs "as if its clarity is maintained" 0.970 (median pair cosine: fog 0.886, vividless 0.943). Along each final axis these positives sit at or below the average of all 364 captions: fog (A07) "unclear", "indefinite", "vague", "shadowy", "as if its clarity is hindered"; vividless (A13) "less vivid", "less vivid or intense", "like things lacking vividness"; veil (A05) "less transparent".

**Why it matters:** Because each axis is the average of its pairs' raw differences, pairs that CLIP barely separates count for little. The 5 fog pairs make up 5% of the fog axis (17% would be their equal share), and the axis is almost unchanged without them (cosine 0.992). The 3 vividless pairs make up 21% (equal share 30%; cosine 0.982 without them). So these phrases distort the axes little, but they are under-represented: the axes are shaped mostly by the captions CLIP reads well, such as "foggy", "as if seen in a fog", "dull" and "colorless".

**Suggested fix:** Extend the D9 check to words and affixes of lack (less, lacking, reduced, un-, in-, -less) and report each pair's share of its axis. No image test is needed for this point alone; the "no_lack" variants of the test in AX-M2 cover it.

**Outcome of the discussion:**

> Text-only measurements added (15 Oct): these pairs contribute little to the axes (fog 5%, vividless 21%), so the effect is under-representation rather than distortion. Proposed: extend the D9 check as suggested, and rely on the "no_lack" variants of the AX-M2 test; no separate test.

**Your notes:**

> None.

## Results assessment

### [ ] AX-R1. The corpus predicts a difference in weight ratio, not a crossover

**Status:** not resolved · severity major · confirmed · outcome: Agreed

**Where:** `doc §7 and §8 (within-category β tables)`

**What's wrong:** Both axes link to both features. Fog: saturation −0.30, local contrast −0.27 (the saturation link is slightly the larger). Vividless: saturation −0.40, local contrast −0.19. So someone whose experience matches "fog" is predicted to prefer both less saturated and less detailed images, and so is someone matching "vividless"; what differs is the ratio, about 1.1 : 1 against 2.1 : 1. With darkness and enclosure removed the contrast is clearer (fog −0.13 vs −0.24; vividless −0.30 vs −0.15).

**Why it matters:** Q3, the crossover between people, is the motivating hypothesis. The axes evidence supports a modest difference in relative weights, not opposite preferences, and that sets the effect size Q3 has to be powered for.

**Suggested fix:** State the expected pattern in these terms in the hypotheses, and carry the predicted weight ratios into the power analysis (to be checked in that part).

**Outcome of the discussion:**

> Your position (15 Oct): the corpus correlations identify which features to change (less saturation, less local contrast). They can't predict a weight ratio, which would change with another dataset or CLIP model.
> 
> My reply: agreed, the numeric ratios shouldn't go into the hypotheses. The corpus shows that both axes relate to both features, so it doesn't by itself support a prediction that people who endorse "foggy" and people who endorse vividless words weight different features.
> 
> Q3 (experimental_design.md §2.2): "Does the adjective a person endorses predict which feature their choices track?" It is called a crossover because the pattern is expected to flip between people, and it is a between-people question because each participant contributes one set of ratings and one pair of weights.
> 
> Agreed (15 Oct): state the Q3 prediction as coming from what patients describe, not from the corpus. Whether the current design can answer Q3 goes to the review of the hypotheses, design and power analysis.
> 
> Action for the experiment (15 Oct): the design and the experiment still say "lifeless" for what is now the vividless axis (A13). Make the naming consistent in the word ratings, the documentation and the analysis.

**Your notes:**

> None.

### [ ] AX-R2. The fog manipulation used in the stimuli wasn't shown to be specific

**Status:** not resolved · severity major · confirmed · outcome: Agreed

**Where:** `doc §6.3 and §8 (Argument 3); 09c_manipulation_sweep.py lines 940 and 959–960`

**What's wrong:** In the sweep, haze_eq (the fog family the stimuli use) raises vividless almost as much as fog (+1.29 vs +1.54 SD), while plain haze separates them much better (+1.44 vs +0.75). The sweep also differs from the stimuli: haze was calibrated on dark_channel_mean, not contrast_local; haze_eq reused those parameters; the 150 images and their levels aren't the 47 stimulus scenes and grid levels; and no haze + chroma combination was tested, although 60 of the 180 analysed trials combine them.

**Why it matters:** The experiment needs fog stimuli that read as fog and colour stimuli that read as colour. The evidence for the stimuli actually shown is indirect, and for haze_eq it points the wrong way.

**Suggested fix:** Encode the 47 × 25 rendered stimuli with un2CLIP and score them on fog (A07) and vividless (A13) (the code exists in 09c). Report each cell's change from the original image, per fog family. That also settles the haze vs haze_eq choice on evidence.

**Outcome of the discussion:**

> Agreed (15 Oct): score the rendered stimuli on the axes before running the experiment. To do after the stimulus pipeline review, so the test can use whatever changes come out of it.
> 
> Prompt for the stimulus-scoring script (for the session that built the stimuli):
> 
> Write a standalone script, score_stimuli_on_axes_v4.py, in the style of the v4 pipeline (SETTINGS block, no command-line arguments, its own output folder, a report and run_meta.json).
> 
> Goal: check that the rendered stimuli move the CLIP axes the way the design assumes. Fog levels should raise fog (A07, key "v4::A07 fog, blurry, hazy") more than vividless (A13, key "v4::A13 dull, less vivid, colorless"); colour levels should do the reverse.
> 
> Inputs: both stimulus sets (data/stimulus_set_v4_ec_haze_eq_2 and data/stimulus_set_v4_ec_haze_2: stimuli/ and stimulus_manifest.csv); axis_vectors.npy and axis_catalog.csv from data/axes_v4; mean_raw and sd_raw from data/scores_v4/score_catalog.csv.
> 
> Steps:
> 1. Encode every rendered image exactly as 03_encode_images.py does (un2CLIP with its own projection, the "openai" preprocessing, fp32). Reuse its loading code.
> 2. For each of the 47 scenes, also encode the unmanipulated original through the same 512 px path build_stimulus_set_v4.py uses. This is the reference. Report its cosine with the stored corpus embedding of the same image_id.
> 3. Score every image on the nine v4 axes, their v4@nocontrol versions and the quality control, in corpus SD units ((raw - mean_raw) / sd_raw).
> 4. For each grid cell (fog level x colour level, per fog family): the change from the scene's original, averaged over scenes, with a 95% interval from resampling scenes; the difference d(fog) - d(vividless); and each axis's share of the total change across the nine axes.
> 5. Per scene, regress the change in fog and in vividless on the measured changes in contrast_local and sat_mean from the manifest, to give effects per feature unit.
> 6. Write cell_effects.csv, scene_effects.csv, a 5 x 5 heatmap per axis and fog family, and a report that answers: (a) does every fog level raise fog more than vividless? (b) does every colour level raise vividless more than fog? (c) which fog family (haze or haze_eq) is more specific? (d) do haze and chroma combine additively in the axis scores?
> 
> Do a dry run on 3 scenes first.

**Your notes:**

> None.

### [x] AX-R3. The darkness sensitivity check is itself confounded with colour

**Status:** resolved · severity minor · confirmed · outcome: Open

**Where:** `doc §6.3 (last bullet) and §7 (nocontrol table)`

**What's wrong:** Removing the darkness and enclosure directions changes several links a lot (lifeless × saturation −0.24 → −0.06; fog × lum_mean −0.10 → +0.10), but the darkness direction itself responds to colour (chroma down moves it +1.23 SD). The doc says so. Neither version separates "dark" from "colourless" cleanly.

**Why it matters:** It limits what can be said about how much of the colour links is really darkness.

**Suggested fix:** Build the darkness control from brightness-only sentence pairs and check in the sweep that chroma down doesn't move it; or rely on lum_mean, which is already in the feature model, and drop the text control.

**Outcome of the discussion:**

> Not discussed.

**Your notes:**

> This test might not be reported in the final article, so this is not a priority.

### [ ] AX-R4. The link sizes are modest, and the "link" threshold is lenient

**Status:** not resolved · severity minor · confirmed · outcome: Agreed

**Where:** `doc §5 (link rule) and §7`

**What's wrong:** Within categories, fog × local contrast is ρ = −0.29 (about 8% shared variance) and vividless × saturation ρ = −0.50 (about 25%). A "link" is any |β| ≥ 0.05 whose interval excludes 0, a low bar with 1.07 million images in 332 categories.

**Why it matters:** The axes explain a modest share of feature variation within a scene type. That is worth saying when setting expectations for the experiment.

**Suggested fix:** Report effect sizes as shared variance as well as β, and qualify small links by size.

**Outcome of the discussion:**

> Explained in the chat on 15 Oct: shared variance is ρ² (fog × local contrast about 8%, vividless × saturation about 25%); the 0.05 threshold reflects what a million images can detect, not size.
> 
> Are these numbers a function of the dataset? (15 Oct) Yes. They depend on the corpus (Places365, filtered to mostly empty scenes, and how much each feature varies within a category), the CLIP model and captions, how the features are measured (central square at 256 px), and, for β, which other features are in the model. They describe this corpus, not a general property of fog or vividness, and they don't predict how strongly participants will respond to the stimuli.
> 
> Agreed (15 Oct): the fog × local contrast relationship is clearly there, and modest in size (about 8% shared variance) for this image corpus, these captions and this CLIP model. Report sizes as descriptive of this corpus (ρ² or R² next to β), describe links by size rather than calling every β ≥ 0.05 a link, and don't generalise the numbers beyond the corpus.

**Your notes:**

> None.

### [x] AX-R5. The image-side numbers can't be checked from the repository

**Status:** resolved · severity minor · confirmed · outcome: Open

**Where:** `.gitignore (axes_pipeline/data/); doc §10`

**What's wrong:** The doc says every number can be traced to a file in the project folder, but data/ is git-ignored, so the corpus results aren't in the repo. I re-ran and confirmed the text stages (Stage V, B and D reproduce exactly; C doesn't with the committed settings, see AX-T1) and recomputed the text-space numbers (axis cosines, darkness leans). None of the image-based numbers in §4–8 could be checked.

**Why it matters:** The headline results (β tables, sweep effects, pole decomposition) are unverified by anyone but the author.

**Suggested fix:** Share the small result files (score_catalog.csv, within_between.csv, checks_results.csv, pole_profiles.csv, sweep_v4/effects.csv; a few MB) or commit them, so the numbers can be verified.

**Outcome of the discussion:**

> Not discussed.

**Your notes:**

> This will be double cheked once we finished to review all the parts of the project.

