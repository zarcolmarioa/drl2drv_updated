# How the v4 trial list was finalised, the grid blocks built, and the design analysed

This document explains, for a reader with no other context, the second round of
work on the v4 image experiment: the changes made to the stimulus pipeline before
preregistration, the two new steps that chose and froze the trial list, the grid
blocks, and the design analysis that produced the registered decision rules.

- **Project folder:** all paths are relative to the project folder
  (`drl2drv_updated/`). Scripts are in its pipeline folder.
- **Script conventions:** every script is standalone, with a SETTINGS block at
  the top and no command-line arguments. Each writes to its own output folder, a
  report (`*_report.txt`), and never overwrites an existing result — a second run
  writes to `<dir>_2`, `_3`, and so on (`ON_EXISTING = "new"`).
- **Scope.** How the photographs were chosen, how the two manipulations work,
  what the two features are, and how the ladders are calibrated are covered in
  the companion document, *How the v4 experiment stimuli were selected, edited
  and measured*, referred to below as **the stimulus document**. This document
  covers only what changed in this round and what the new steps do. Where a
  mechanism is unchanged, it is not restated here; there should be exactly one
  account of each thing in the project.
- **Every number comes from a script's SETTINGS block or from a run report.**
  Section 17 lists the figures that still need to be read out of reports this
  document's author did not have to hand.

## Guide: sections and scripts

| Section | Step | Script |
|---|---|---|
| 1 | What this round changed, and why | – |
| 2 | Units: the idea the whole round turns on | – |
| 3 | Step 1: levels above the original | `calibrate_levels_v4.py` |
| 4 | Step 2: the band-sweep correction | `select_scenes_v4_corrected.py`, `make_screening_page_corrected.py` |
| 5 | Step 3: the cell plan, and what each image is for | `build_stimulus_set_v4_corrected.py` |
| 6 | Step 4: oriented predictors | `analyse_pairs_v4_corrected.py` |
| 7 | Step 5: four candidate trial lists | `build_trials_v4_corrected.py` |
| 8 | Step 6: choosing the colour condition | `choose_colour_condition_v4.py` |
| 9 | Step 6b: does the choice survive lapses? | `check_robustness_v4.py` |
| 10 | Step 7a: freezing the list | `freeze_trial_list_v4.py` |
| 11 | Step 7b: the grid blocks | `build_grid_block_v4.py` |
| 12 | Step 8: the design analysis | `simulate_power_v4.py` |
| 13 | The registered design in numbers | (from the reports) |
| 14 | Known properties and limitations | – |
| 15 | Directory map | – |
| 16 | What is not included | – |
| 17 | Open items | – |

---

## 1. What this round changed, and why

A review of the stimulus set, the trial list and the design analysis found seven
problems. They are listed here because every change downstream is traceable to
one of them, and because several are the kind of error that would have survived
into the preregistration unnoticed.

1. **Colour was harder to detect than fog.** Colour-only pairs stayed within one
   fog level, and in mixed pairs haze drained colour as a side effect, so a
   participant whose choices followed colour was caught far less often than one
   whose choices followed fog. Two remedies were proposed and tested rather than
   adopted: colour levels **above** the original, and drawing colour-only trials
   from the larger colour differences.
2. **The weights were in grid-dependent units.** Each pair's difference was
   divided by that scene's range on the grid, so a weight on fog and a weight on
   colour were not comparable, and changing the grid changed what a given weight
   meant. Effect sizes had to be redefined in raw feature units.
3. **The predictor signs were documented wrongly.** The trial list stored
   unsigned magnitudes; the power scripts signed them internally, but the
   documentation described them as signed and gave a model that used them as
   they stood. An analysis written from the documentation would have been wrong.
4. **The existence claim assumed non-responders choose at random.** Real people
   without the effect still have small personal preferences in either direction.
   Counting detections in one direction only can then manufacture a claim out of
   taste.
5. **A "pick the more edited picture" strategy could mimic the effect.** Such a
   participant gives positive weights on *both* features and a curve that rises
   with difference size, so a rising dose-response does not rule out a response
   strategy. Specificity has to rest on a control group instead.
6. **The simulations were not realistic.** Simulated participants never lapsed,
   effect sizes were inherited from an earlier design, the stopping advice came
   from a jump in a fixed-count rule rather than from any loss of power, and the
   quoted Bayes factors were two-sided while the plan was one-sided.
7. **A second grid block was needed.** To measure each person's shift from
   "normal" to "episode", the experiment needed a normal-appearance block as
   well as the episode block.

---

## 2. Units: the idea the whole round turns on

The stimulus document explains that a **rung** is defined by the measured
feature, not the parameter. This round extends the same discipline to the
analysis, and it is worth stating plainly because it is what makes problems 2 and
3 go away.

**Predictors are in corpus-SD units, oriented, and written into the trial list.**
For every analysed trial the list now carries two columns:

```
pred_sat = -d_sat / SD_SAT        SD_SAT = 39.182
pred_fog = +d_fog / SD_FOG        SD_FOG =  9.348
```

where `d_sat` and `d_fog` are the measured feature differences between image A
and image B, and the SDs are those of the two features across the 20,000-image
corpus reference. Three things follow:

- **Only `d_sat` is negated.** Fog *lowers* `contrast_local`, so a raw negative
  fog difference already means "A is foggier"; colour runs the other way.
  Both predictors are therefore positive when image A is the one a responder is
  predicted to choose, and a positive weight always means "prefers the image
  that looks more like derealization".
- **The units do not depend on the grid.** Adding a colour level above the
  original changes what trials exist; it does not change what a weight of 1.0
  means. This is why the extra levels could be tested at all.
- **The signing happens once, in the trial list.** Step 8 reads the columns
  rather than signing them internally, and then rebuilds them from `d_sat` and
  `d_fog` purely as a check that the list and the script agree. The registered
  thresholds are only valid if the eventual analysis fits exactly the same
  model, so there must be one definition of the model, not two.

The earlier scene-range magnitudes `z_sat` and `z_fog` still appear in the pair
inventory as descriptive quantities. **They are retired as predictors** and no
registered quantity uses them.

**Per-person fitting is Firth's bias-reduced logistic regression**, with no
intercept, on the two oriented predictors. Firth was chosen because it is always
finite under separation — a participant who answers perfectly consistently would
give infinite maximum-likelihood weights — and because it is equivariant to
rescaling the predictors, so the corpus-SD scaling does not change the inference.

---

## 3. Step 1: levels above the original

**Script:** `calibrate_levels_v4.py`. **Output:** `data/levels_v4_ext/`.

The mechanism is unchanged (stimulus document, section 3): a rung is the
parameter at which the median image's measured feature reaches a named corpus
percentile. Four things were added.

**Colour levels above the original.** `colour_up` is a new dimension in
`levels.csv`, calibrated to percentiles **above** the median rather than below,
with the same solver. Two rungs are used, chosen to match the spacing of the
existing ones: `p65` (k = 1.3302) and `p75` (k = 1.7295). A third, `p90`, was
solved and **deliberately rejected**: the corpus p90 of `sat_mean` needs a
chroma multiplier past the gamut ceiling, so it came back `capped (out of range)`
having achieved +0.89 SD against a +1.55 SD target. A level that cannot reach its
target is not a calibrated level, and `ALLOW_CAPPED_LEVELS = False` is what stops
it being rendered as though it were.

**Making chroma above 1 work.** The existing chroma edit multiplies a\* and b\*
in CIELAB. For `k > 1` that runs straight out of gamut, so `m_chroma` now takes
the maximum in-gamut chroma per pixel and blends toward it:

```
C' = C + H · (1 − exp(−(k−1)·C / H)),    H = max(cmax − C, 0)
```

which approaches the ceiling asymptotically rather than clipping into it. The
ceiling is computed from the image **as it stands after the haze step**, not from
the unmodified square. Haze moves both lightness and hue, and the ceiling is a
function of both, so a ceiling taken from the original belongs to a different
image: at the deep fog rungs it would let chroma run past the gamut and the
soft-saturating blend would stop protecting anything. For `k ≤ 1` nothing is
computed and the code path is byte-for-byte the old one — which is what makes
the reproduction check in step 3 a real test rather than a tautology.

**Contrast levels for the grid blocks.** The fog screen needed a column showing
*more* contrast than the original. The inverse of the haze model does not supply
one usefully, so a separate family was calibrated: `contrast_gain_srgb`,

```
I' = m + g·(I − m)   on sRGB values, m = the image's mean grey,
                     followed by a mean-restoring linear-light exposure
```

This is the same family of operation as `PIL.ImageEnhance.Contrast` at factor
`g`, verified to within one level on five real scenes, though not identical to it
because of the exposure step. A linear-light variant was tried and rejected: at
the same clipping budget it solved no level at all, and on bright scenes a linear
gain crushes a wide band of real shadow detail. Three gains were carried forward
— 1.0875, 1.1100 and 1.1500 — all marked **provisional**, to be judged on the
pictures at the size participants would see them. Section 11 records how that
ended.

**Missing percentiles.** The reference statistics file was written with
`p01…p50`, `p75`, `p90` and `p99` but not `p55`, `p60`, `p65` or `p70` — the very
columns step 2's band sweep looks up. They are now written, and the lookup raises
rather than skipping (section 4).

---

## 4. Step 2: the band-sweep correction

**Scripts:** `select_scenes_v4_corrected.py`, `make_screening_page_corrected.py`.

One bug, and it is instructive. `band_counts()` looked a percentile column up in
the reference table and, when the column was missing, **silently skipped the
feature filter for that band**. Because the reference file lacked `p55`–`p70`,
four of the five rows in the band sweep reported `after_features` equal to
`after_axes` — the feature filter had never been applied at all. The sweep table
in the stimulus document was therefore wrong for those rows, and the table is
what the 35–65% band was chosen from.

The fix: look the column up, and **raise** when it is missing. Only the sweep was
re-run. The required check — that the selected scenes are exactly the same as
before — passed, because the chosen band's row was one of the ones that had been
computed correctly.

The screening tool changed only in `BASE_DIR`, which follows the selection output
folder because the page references its crops by relative path.

---

## 5. Step 3: the cell plan, and what each image is for

**Script:** `build_stimulus_set_v4_corrected.py`. **Output:**
`data/stimulus_set_v4_ext_haze_eq/`.

The grid per scene grew from 5 × 5 = 25 cells to **45**, and the script gained a
single table that says what every one of them is for.

**The ladder as built:**

```
colour   p75=1.7295, p65=1.3302, k=1=1.0000, p25=0.6516, p10=0.4247,
         p05=0.3204, p01=0.1679                            (7 columns)
fog      h=0=0.0000, p35=0.0771, p25=0.1403, p15=0.2245,
         p10=0.2870, p05=0.3875                            (6 rows)
contrast 1.0875, 1.1100, 1.1500   (contrast_gain_srgb, grid only, provisional)
```

7 × 6 = 42 grid cells plus 3 contrast cells = 45 per scene; 47 scenes = **2,115
images**.

**One cell plan, read by everything.** `build_cells()` writes `cell_plan.csv`,
and the measuring pass, the contact sheets and the stimulus writer all go through
it. The older script rebuilt the colour × fog loop separately in each of the
three, which was harmless while every cell was the same kind of thing and is not
once there are three kinds.

**The `use` column** is what steps 4, 5 and 7 filter on:

| `use` | cells/scene | meaning |
|---|---|---|
| `pair` | 30 | pair trials only |
| `both` | 5 | pair trials **and** a grid screen |
| `grid` | 10 | grid screens only — never a pair trial |

A cell is on a grid screen exactly when **one of its two levels is the
original**, so the two grid screens are a cross through the plan: the fog screen
holds colour at the original and walks down the fog rungs; the colour screen
holds fog at the original and walks across the colour levels.

**The `h = 0` row is not in the pair set.** `PAIR_INCLUDES_UNMANIPULATED_FOG =
False` keeps it out. Every pair in the earlier build was two manipulated images,
because `h = 0` cells did not exist; keeping that true is what lets step 4
reproduce the old pair table on the levels the two builds share. There is a real
argument for admitting them — a pair against the untouched photograph is the
cleanest possible comparison — but that is a design change, not a bug fix, and it
was not taken.

**Checks, final run.** Eight, all reported:

| check | result |
|---|---|
| 1 feature code | PASS (1.1e-09) |
| 2 envelope | PASS, 2.0% of cells outside the corpus 1–99% range |
| 3 fog families | NOTE (0.468) — the per-scene equivalence of the stimulus document |
| 4 affordance | PASS, 47/47 scenes afford all four pair types |
| 5 coverage | PASS, every type has ≥ 40 pairs in each of 3 magnitude bins |
| 6 colour ladder | PASS, **0 of 282** scene × fog-rung ladders affected |
| 7 contrast damage | reported, no threshold (see below) |
| 8 reproduction | **PASS, 1,175 of 1,175 images byte-identical, ladder parameters identical** |

**Check 8 is the one that matters most.** It compares every cell this build
shares with the earlier one, parameters first and then the PNG bytes. Parameters
first because a moved ladder *should* give different images, and a bare hash
cannot tell that from a broken renderer. All 1,175 shared images came back
byte-identical with identical parameters: **the twenty new cells per scene are
additions to the existing set, not a new set.** Trial lists and any data already
collected on the shared cells stay valid.

**Check 6 covers the new vivid rungs.** `sat_mean` is monotone across all seven
colour levels in every scene at every fog rung, with no collapsed step — a
collapsed step would give a pair trial with no stimulus difference, which the
analysis would score as a coin flip and treat as evidence.

**Check 7, contrast damage.** `flattened` is the share of pixels at least 15
levels below white in the original that render as exactly 255; `crushed` is the
same at the black end.

| gain | scenes with zero damage | worst flattened | median Δ contrast_local |
|---|---|---|---|
| 1.0875 | 40 of 47 | 0.0075 | 2.03 |
| 1.1100 | 22 of 47 | 0.0248 | 2.53 |
| 1.1500 | 1 of 47 | 0.1109 | 3.36 |

The highest gain was rendered **blind** — step 1's sweep stopped at 1.110 — and
exists only so the pictures could be judged.

**Two full-resolution diagnostics** were added for the new cells, drawn with PIL
at exactly 512 px per cell: `sheets_vivid/` for the colour rungs above the
original and `sheets_contrast/` for the gains. They answer the question the
240 px contact sheets cannot — does the edit leave visible damage when you zoom
in — and the separation exists because of a lesson from step 1, where a contact
sheet rendered at 221 px made a real contrast change look like no change at all.

---

## 6. Step 4: oriented predictors

**Script:** `analyse_pairs_v4_corrected.py`.

Two runs under identical thresholds: the **original** levels only (5 colour
levels) and the grid **extended** with the two vivid levels (7). The first is the
control — it must reproduce the existing pair table — and the second is what step
5 samples from. Only cells with `use in ("pair", "both")` are read, so the `h = 0`
row and the contrast column are excluded as grid-block material.

The inventory gained the oriented predictor columns described in section 2,
alongside the raw signed feature differences, which are kept because every pair
classification is made from measured features and a reader needs to be able to
check the orientation rather than trust it.

**Reproduction (section 0 of the report): PASS.** 15,004 pairs in both, **0 only
in the reference and 0 only in this run.** The control variant uses the colour
levels the earlier build had, on images step 3 proved byte-identical, so its pair
table had to be identical too — and if it had not been, the comparison below
would have been between two different things.

**The two variants side by side**, identical thresholds, same manifest, so every
difference comes from the two vivid levels and nothing else:

| quantity | original | extended |
|---|---|---|
| images | 1,175 | 1,645 |
| usable pairs | 15,004 | 34,366 |
| **`sat_only` pairs** | **2,030** | **6,282** |
| `sat_only` per scene | 43.2 | 133.7 |
| `fog_only` pairs | 1,750 | 2,136 |
| `together` pairs | 8,266 | 17,666 |
| `conflict` pairs | 2,958 | 8,282 |
| balanced conflict pairs | 1,548 | 4,300 |
| share of conflict balanced | 0.523 | 0.519 |
| median balance ratio | 0.70 | 0.97 |
| fog cue stronger by | 1.43× | 1.03× |
| `sat_only` median \|d_sat\| | 26.3 | 35.3 |
| `sat_only` max \|d_sat\| | 67.3 | 108.7 |
| scenes with ≥ 3 balanced conflict | 47/47 | 47/47 |

`sat_only` is the row that matters. Colour-only trials are the scarce kind —
confined to a single fog row, with haze draining colour in mixed pairs — and
problem 1 is exactly that scarcity. The vivid levels **triple** the pool and
raise the median colour difference from 26.3 to 35.3 units.

**The balance normalisation had to change, and the check that caught it.**
Section 0b asks whether the selection rule is **ladder-independent**: a pair
present under both ladders must get the same balance, or the rule is responding
to which levels were rendered rather than to the images.

Under the old normalisation — balance divided by the **scene's own range** —
it is not. The vivid levels grow that range by 56%, so the ratio falls for every
pair, **including the 15,004 that are bit-identical between the two ladders**.
The diagnostic run records the damage: `balance_range` moves by up to 1.240 and
the `balanced` flag **flips on 908 pairs**, FAIL. A reader comparing the two
columns would have concluded that the vivid levels worsened the cue balance,
when nothing about any image had changed.

The build therefore uses `BALANCE_NORM = "corpus_sd"`: both cues divided by their
feature's corpus SD, which does not move when the ladder changes. Section 0b then
passes with **0 flips** and a worst balance difference of 0.000e+00, and the
balance row in the table above is comparable across the two columns — which it
was not before. `balance_range` is kept as a reported column precisely because
its moving is the reason the rule changed.

One consequence to carry forward: the `balanced` flag differs from the earlier
run **by design**, so step 5 draws a different conflict set than the previously
shipped trial list did. The reproduction check compares the pairs, their types
and their raw differences — all unchanged — and deliberately does not compare
`balanced`.

Magnitude coverage passes for every type in both variants, and concentration is
low: the single largest scene contributes 4.4% of the balanced conflict pool in
the original variant and 3.6% in the extended one.

---

## 7. Step 5: four candidate trial lists

**Script:** `build_trials_v4_corrected.py`. **Output:**
`data/trials_v4_ext/<version>/`.

Three changes and then a fan-out.

**Colour-only sampling.** A setting draws the colour-only trials from the larger
colour differences instead of evenly across magnitude bins. The cut point is a
setting, and the report states how much small-difference range is lost — which
matters, because the exploratory dose-response needs data at the low end.

**Practice trials.** Previously drawn only from unused pairs where one image was
both hazier *and* less colourful, so every practice trial showed a "more edited
versus less edited" contrast — exactly the demonstration that would nudge a
participant toward the strategy in problem 5. They are now drawn from a mix of
types (`PRACTICE_TYPES = ("sat_only", "fog_only", "together")`) at
small-to-medium differences, capped at the 60th percentile of each pool
(`PRACTICE_MAG_QUANTILE = 0.60`) so practice is representative rather than a run
of the most obvious pairs. Practice is drawn **before** catch, so changing the
number of catch trials does not change which pairs practice uses. Catch trials
were left as they were.

**Four versions**, everything else identical and the same seed:

| version | levels | colour-only sampling |
|---|---|---|
| `current` | original | even across bins |
| `extra_levels` | with vivid colour | even across bins |
| `larger_diffs` | original | larger differences |
| `both` | with vivid colour | larger differences |

A `reproduction` variant is built alongside them — the old ladder **and** the
old rank-based bins — as the control that must reproduce the existing list.

**The seven versions as built** (the two `sens_cut_*` rows are alternative cut
points added so the cut could be chosen from evidence):

| version | variant | colour cut | `sat_only` median \|d_sat\| | r(d_sat, d_fog) |
|---|---|---|---|---|
| `current` | original | none | 31.0 | −0.086 |
| `extra_levels` | extended | none | 35.7 | −0.106 |
| `larger_diffs` | original | 31.1 | 43.4 | −0.118 |
| **`both`** | extended | 31.1 | 50.1 | −0.110 |
| `sens_cut_25_5` | extended | 25.5 | 43.4 | −0.111 |
| `sens_cut_35_3` | extended | 35.3 | 50.1 | −0.110 |
| `reproduction` | legacy | none | 25.3 | −0.111 |

Every version has VIF 1.01, 180 analysed trials and the same type counts; they
differ only in which pairs fill them.

**A bug worth recording, because it invalidated the first comparison.** Step 5
originally drew every trial type off **one** random stream. The colour cut
filters colour-only pairs and nothing else, so it should leave `fog_only`,
`together` and `conflict` untouched — but moving the stream changed them all:
`current` and `larger_diffs` shared **1 of 60** `fog_only` trials, 1 of 30
`together` and 0 of 30 `conflict`. Two consequences: the fog condition was not a
control, and colour information leaked in from types the cut never touches —
`both` gained about a sixth of its apparent advantage over `current` from its
`together` trials alone. Sampling now runs on **per-type streams**, 13 generators
each seeded from `(seed, draw name)`, and both step 5 and step 6 verify the
invariant.

**The shipped list** (`both`, seed 42):

```
colour cut              |d_sat| >= 31.1 sat_mean units (0.794 SD)
colour pairs after cut  3,768 of 6,282 (60%); dropped range 17.0-31.1
sat_only bins           fixed value edges; bins 1 and 2 empty, 3 usable
trials presented        211   180 analysed, 15 consistency, 10 catch, 6 practice
r(d_sat, d_fog)         -0.115    VIF 1.01
side balance            worst deviation from 0.50 is 0.000
scenes used             47        max 7 trials from one scene
consistency min gap     34
catch                   10 trials, mode 'no_images', exclude at >= 3 failures
re-measurement shift    0.000 (PNG, not resized)
package                 349 images, 105.8 MB; 334 preloaded
```

Two changes from the earlier list are visible here. **Catch trials went from 6 to
10** and are shown in `no_images` mode, which is what makes the exclusion rule at
≥ 3 failures workable (section 12). And the magnitude bins are now **fixed value
ranges in corpus-SD units, shared by every version**, so "bin 2" means the same
difference everywhere; a bin holding fewer than 20 pairs is treated as empty and
its trials spread over those that remain. Under the colour cut the two lowest
`sat_only` bins are empty by construction, which is the cut doing its job.

The required check — that `current` reproduces the existing trial list — is what
the `reproduction` variant is for. Its consequence is in section 8.

---

## 8. Step 6: choosing the colour condition

**Script:** `choose_colour_condition_v4.py`. **New in this round.**

The four versions are compared on **one** question: how often is a participant
with a *fixed* sensitivity caught? Two rules make the comparison fair, and both
follow from section 2.

1. **Sensitivity is fixed in raw feature units** — log-odds per raw difference in
   `sat_mean` and in `contrast_local` — never in grid-scaled units. A
   participant with a given sensitivity is then literally the same participant
   across the four lists, which is the only way the comparison means anything.
2. **Thresholds are computed per version**, from random choosers on that
   version's own trial list. A version that happens to produce a looser threshold
   would otherwise look better simply for being noisier.

The script reuses the per-person fitting and threshold functions rather than
reimplementing them, and reports per version: colour detection at each
sensitivity, fog detection at each sensitivity (so it is visible that fog does
not get worse), the false-detection rate of random choosers as a check, and the
range of colour and fog differences used.

**Scale of the run.** 20 independent draws per version (122 draws in all),
4,000 simulated people per cell per draw, 20,000 per floor, sensitivities 0.3 to
1.0 per SD, floors at 2.5% false alarms per person per feature, one-sided.
Elapsed 327.9 s.

**Why many draws.** A trial list is 180 trials sampled from a pool of thousands,
so two versions can differ because one is a better design or because one drew
better. The first run of this script used **one draw per version and misled**: the
fog control read 0.531 for `current` against 0.508 for `larger_diffs` at 0.50 SD,
against a Monte Carlo error of 0.011 — an apparently real cost of the colour cut
that was in fact two different draws of fog trials, caused by the stream bug in
section 7. Simulated people are now seeded by **draw index, not by version**, so
within an index every version faces the same people and differences are paired.

**Information and floors.** Information is the sum of squared predictors in
corpus-SD units — what the logistic fit has to work with — and the floor is the
weight 2.5% of pure guessers exceed by luck *on that draw*:

| version | colour information | colour floor (SD) | fog floor (SD) |
|---|---|---|---|
| `shipped` (the old list) | 58.9 | 0.508 | 0.486 |
| `current` | 71.9 | 0.467 | 0.486 |
| `larger_diffs` | 96.7 | 0.401 | 0.486 |
| `extra_levels` | 140.4 | 0.336 | 0.482 |
| `sens_cut_25_5` | 158.3 | 0.315 | 0.483 |
| **`both`** | **186.6** | **0.290** | 0.482 |

The fog floor barely moves, which is the point: nothing the colour cut does
should touch fog.

**Detection: share of people caught, by true sensitivity, mean over 20 draws.**

| sensitivity (SD) | `shipped` | `current` | `larger_diffs` | `extra_levels` | `sens_cut_25_5` | **`both`** |
|---|---|---|---|---|---|---|
| 0.30 | 0.218 | 0.240 | 0.307 | 0.414 | 0.457 | **0.520** |
| 0.40 | 0.337 | 0.382 | 0.488 | 0.630 | 0.686 | **0.756** |
| **0.50** | 0.470 | 0.547 | 0.676 | 0.812 | 0.856 | **0.908** |
| 0.60 | 0.636 | 0.696 | 0.820 | 0.920 | 0.948 | **0.974** |
| 0.70 | 0.760 | 0.819 | 0.914 | 0.973 | 0.986 | **0.994** |
| 1.00 | 0.965 | 0.980 | 0.996 | 1.000 | 1.000 | **1.000** |

At the middle sensitivity the chosen version catches **0.908** of participants
against **0.547** for `current` and 0.470 for the previously shipped list.
Between-draw SDs are 0.005–0.024, so the ordering is not a draw artefact. Both
changes contribute, and they compound: the vivid levels alone give 0.812, the cut
alone 0.676.

**Checks.** False alarms PASS (worst 0.0032 against a 0.025 floor); the fog
control PASS (worst 0.0010, SE 0.0009) — fog detection does not get worse;
stream separation PASS, with `current` and `larger_diffs` now sharing all 20 of
20 draws identically on `fog_only`, `together` and `conflict`; and five estimator
checks including equivariance under a change of predictor scale (8.2e-12).

**The estimator changed here, and the old one was inert.** The per-person fit was
plain maximum likelihood with a ridge of 1e-06, nominally there to keep the fit
finite under separation. The predictors were in raw feature units, so the
diagonal that ridge was added to ran to tens of thousands and the ridge was
**3.6e-11 of it**; refitting at 1e-3 moved the weights by less than 1e-5. The
fits were maximum-likelihood in practice and unprotected in principle. Excluding
separable fits was considered and rejected — a participant who answers every
colour trial as predicted is the strongest evidence this design can produce, and
dropping them would bias against the hypothesis. **Firth is always finite,
differs from the MLE by about 2% away from separation, and removes most of the
small-sample bias**, which matters because the control comparison reads
magnitudes between groups rather than just signs. The floors use the same
estimator, so the shrinkage applies to guessers and responders alike.

**One finding the preregistration must carry.** The `shipped` and `current` lists
use the same ladder, the same pool and the same 60 colour trials. They differ
only in their **magnitude bins**: the shipped list used rank-based quantile bins,
which hold equal numbers of pairs and so concentrate trials in the middle, while
fixed value edges push trials into the tails. Information is a sum of squares, so
large differences count disproportionately — and colour information rises from
58.9 to 71.9, **+22%, with no change to the stimuli or the pool**. The
`reproduction` variant (same ladder *and* same bins as the shipped list) sits
beside `shipped` at 58.9, which is the check that binning is the cause. **The
baseline moved for a reason unrelated to anything the four versions were designed
to vary, and that should be stated.**

The script **does not pick a version**. It stops, and the choice is made
separately. The chosen version was `both`.

---

## 9. Step 6b: does the choice survive lapses?

**Script:** `check_robustness_v4.py`. **New in this round.**

Two checks before anything is frozen.

- **Does the ranking survive response lapses?** A lapse is a trial answered at
  random. If the ordering of the four versions flips once participants lapse at a
  realistic rate, the step-6 ranking is an artefact of assuming perfect
  attention.
- **Are the fitted weights stable enough to compare between groups?** The
  control-group comparison in step 8 is a test on per-person weights, so the
  weights have to be estimable at the planned trial counts before that comparison
  can be designed around them.

Four lists (`current`, `extra_levels`, `sens_cut_25_5`, `both`), 20 draws each,
lapse rates 0.0 / 0.1 / 0.2, 4,000 people per cell, 20,000 per floor. Elapsed
725.4 s.

**The ranking survives.** Expected order from step 6 is
`extra_levels < sens_cut_25_5 < both`, and it holds in **every** sensitivity ×
lapse cell with **0 resolvable inversions**. Adjacent gaps are tested against
paired standard errors, and only a gap reversing by more than 2 SE would count.

**Detection at 0.50 SD, by lapse rate:**

| lapse | `current` | `extra_levels` | `sens_cut_25_5` | **`both`** |
|---|---|---|---|---|
| 0.0 | 0.544 | 0.814 | 0.860 | **0.909** |
| 0.1 | 0.459 | 0.722 | 0.775 | **0.839** |
| 0.2 | 0.377 | 0.615 | 0.670 | **0.738** |

Lapsing costs every list roughly the same share, so it compresses the absolute
numbers without reordering them. At a 20% lapse rate the chosen list still
catches 0.738 where the current one catches 0.377.

**The floors do not depend on the lapse rate**, which they must not: a lapse is a
trial answered at random, and a pure guesser is already answering at random, so
lapsing cannot change a guesser. This is verified two ways — the guessers'
responses come back **bit-identical** under lapsing, and the floors measured
under each lapse rate agree to within 1.47 SD of a sampling difference.

**The weights are usable.** Across every cell: **no separated fits, no extreme
weights** (|colour weight| > 5 per SD), **no unconverged fits** — all three
shares exactly 0.00000. The closest any simulated participant came to separation
was answering 0.867 of colour trials as predicted, where 1.000 is needed.
Noise-to-signal at 0.50 SD falls from 0.490 on `current` to **0.323** on `both`.
Refitting every cell with the other estimator moves a weight by at most
**0.024 SD (2.3%)**, moves a floor by at most 2.2%, and moves a detection rate by
at most 0.0019 — 0 of 504 cells beyond 0.01.

The conclusion both checks support: the step-6 ordering is not an artefact of
assuming perfect attention, and the per-person weights are precise enough for the
between-group comparison step 8 is built on.

---

## 10. Step 7a: freezing the list

**Script:** `freeze_trial_list_v4.py`. **Output:**
`data/frozen_trial_list_v4/`.

The chosen list is rebuilt once with the chosen settings, verified end to end,
copied into a directory nothing else writes to, and recorded with a **SHA-256 for
every file** in `checksums.sha256`, plus `FROZEN.json` and `freeze_report.txt`
describing what it is.

Three files are deliberately **not** in `checksums.sha256` —
`checksums.sha256` itself, `FROZEN.json` and `freeze_report.txt` — because a
file cannot contain its own hash. `FROZEN.json` is cross-checked separately
instead, so the exclusion does not create a hole.

**From here on, every script that reads the trial list verifies the checksums
before using it.** Step 8 does this at start-up and refuses to run on a package
that does not match.

The frozen package contains `trial_list.csv`, `trial_design.png`,
`trials_report.txt` and the complete `package/` directory the experiment serves.

---

## 11. Step 7b: the grid blocks

**Script:** `build_grid_block_v4.py`. **New in this round.** **Output:**
`data/grid_block_v4/`.

### What the grid block is

Two blocks of four screens each. Every screen shows one feature at every level at
once: the **columns** are the levels of that feature in display order, the
**rows** are three different scenes at the same levels so that no single
photograph drives the choice. The participant picks the **column** that looks
most like how the world looked.

| block | when | question | "none of these" | free text |
|---|---|---|---|---|
| episode | after the pair trials | how your surroundings looked during the episode | yes | yes, when `none` is chosen |
| normal | end of the session | how your surroundings normally look | **no** | no |

The normal block has no `none` button on purpose: "my surroundings normally look
like none of these" is not an answer the shift can use, and an empty normal
answer loses the whole pair.

**It is descriptive only.** It locates an experience on the manipulated range; it
is not a test of the hypothesis, and no registered analysis uses it. Each feature
is asked twice, forward and with the column order reversed, with different
scenes, so that the two answers can be compared for stability and so that
answering by position — always the leftmost or always the rightmost column — is
detectable.

### The columns

```
fog      h=0 (the original), p35, p25, p15, p10, p05       (6 columns)
colour   p75, k=1 (the original), p25, p10, p05, p01        (6 columns)
```

Six on each screen, which matters because the experiment computes **one cell size
for the whole block from the widest screen in the file**; unequal counts would
make one screen smaller than it needs to be.

**The contrast-up column was dropped.** All three gains were judged against
`sheets_contrast/`, step 3's full-resolution diagnostic at 512 px, and none was
distinguishable from the original there. The grid shows cells at 149–239 px, so a
difference invisible at 512 px cannot appear in the grid. The damage figures were
never the binding constraint — gain 1.0875 does no measured damage at all on 40
of 47 scenes and is still invisible.

Why it is so weak is worth recording, because the measurement says otherwise. The
gain holds the mean fixed and moves only the **histogram tails**: +7.0 on
`lum_p95` and −6.4 on `lum_p05` at 1.0875, rising to +11.3 and −10.7 at 1.1500.
That change lives in a small fraction of the frame, and downscaling averages it
away. Haze moves the same statistics further *and* lays a coherent veil over
every pixel, which is what survives a thumbnail. **Equal movement in
`contrast_local` SD is not equal visibility in the two directions**, so the
contrast-up gains and the fog rungs are not comparable on that number.

**`p65` was dropped from the grid**, and from the grid only — it remains in the
pair trials at every fog rung. It leaves six columns on both screens, and it
leaves `p75` (25 percentile points above the original) facing `p25` (25 below).
Note the limit of that symmetry: it is exact in **percentile points** and not in
feature units. On the twelve colour scenes shipped, the first step up is
**+0.885 SD** and the first step down **−0.627 SD**, a ratio of 1.41. (`p65`
would have been +0.472 SD, a ratio of 0.75 — closer to symmetric, at the cost of
halving the reach above the original.)

Six columns each does **not** put the original in the same position on both
screens: it is 1st of 6 on the fog screen and 2nd of 6 on the colour screen. So
comparing *which column* people choose across the two features is partly a
position effect. The shift within each feature is unaffected, and that is the
quantity the analysis uses.

### The fog axis is censored at the original

With no column above the original, a participant whose vision is clearer than the
photograph as taken has to answer on the end column. **In the normal-appearance
block that is the common case, not the edge case**, because most people's normal
vision is not foggy.

A fog answer on the original column is therefore a **lower bound** on that
person's normal, and the normal-to-episode shift computed from it is a lower
bound on the shift. The bias is conservative — it can only shrink the shift — but
it is declared rather than absorbed, and **the count of fog answers on the
original column in each block is reported alongside the shift.**

This is also why `chosen_level_index` must be recorded and not only the display
column: the censored answers are exactly `level_index == 0` on a fog screen, and
the display column moves when the column order reverses.

### Two matched scene sets

The shift is a within-person comparison of two blocks, so a photograph may not
appear in both — recognising it would anchor the second answer. 24 scenes are
chosen as **12 matched pairs**, one member of each pair to set A and the other to
set B, **in the same slot** (same screen, same row). The two sets are then
swapped between participants, so neither set is systematically the episode set.

Matching by pairs rather than by set means is deliberate: matched set means can
hide a set that is bimodal, and a pair bound to a slot makes the two blocks
comparable row by row rather than only on average.

Pairs are matched on the baseline `sat_mean` and `contrast_local` of the scene's
own original cell, in corpus-SD units so that a within-pair distance is in the
same units as an effect in step 8. A headroom criterion on contrast damage was
available as a strict secondary, applied lexicographically — best matching first,
then the best matching among headroom-eligible scenes, taken only if it costs no
more than a stated budget — but it is inactive now that there is no contrast-up
column to have headroom for.

A pair is assigned to a feature by its **span**, the distance its columns travel
on that scene: the fog screens get the pairs with the widest local-contrast range
and the colour screens those with the widest saturation range. A scene with a
narrow span shows six columns that look much the same, which asks for a
distinction the stimulus does not offer.

### Nothing was rendered

Step 3 already renders every cell a grid screen needs and marks it
`use = "grid"` or `"both"`. This step reads those rows and **ships copies**. In
particular the colour columns above the original exist for every scene already,
so no additional rendering pass was needed and step 3 was not re-run. The feature
code and the filename helpers are **imported** from step 3's script rather than
copied, so there is one definition of each and the two cannot drift.

The frozen pair package is not read, not written and not re-checksummed; its
images are a disjoint set of cells.

### Result, final run

```
scenes available                47
grid-usable images             705
fog columns                      6   h=0, p35, p25, p15, p10, p05
colour columns                   6   p75, k=1, p25, p10, p05, p01
realised cell size             149 px at 1366x768, 239 px at 1920x1080
matching total              0.6201 SD over 12 pairs (mean 0.0517)
worst pair                  0.0698 SD;  worst set-mean difference 0.0031 SD
cells on screens               144   over 2 sets x 4 screens x 3 rows
fog range on the screens     -2.16 to +0.00 SD
colour range on the screens  -1.78 to +1.05 SD
rendered but not used          188 images, all with a recorded reason
pair reuse                     1-8 pair trials per grid scene (median 5)
images shipped                 144 at 512 px, 43.7 MB
```

Checks: inputs agree with the cell plan; every column resolves to a file for
every scene; the colour span reproduces step 3's own `colour_ladder.csv`
**exactly** over all 47 scenes; every column step is monotone and above the
floor; the two sets are disjoint with one pair per slot; the layout fits in both
windows; and every shipped image is re-measured with step 3's own code
(worst relative difference 2.0e-16) and hashed against its source
(**144 of 144 byte-identical**).

**188 rendered images are not shown** — 141 contrast-up and 47 `p65 @ h=0` — and
`unused_cells.md` records each with its reason. Nothing was deleted or
re-rendered; the images remain in the stimulus set with their measured features.

The package carries `grid_trials.json` (the layout), `grid_trials.csv` (one row
per cell with measured features and `sd_from_original`), `manifest.csv`,
`preload.json`, `README.txt` and `interface_spec.md` — the contract stating what
the experiment must render and what it must record.

**Measure the shift in feature units, not grid steps.** One step is not a
constant amount of anything: on the shipped screens the first colour step up is
+0.885 SD and the first down −0.627 SD, while the first fog step is −0.415 SD.
`grid_trials.csv` carries `sd_from_original` per cell for exactly this.

---

## 12. Step 8: the design analysis

**Scripts:** `simulate_power_v4.py` (the original run) and
`simulate_power_v4_n10_20.py` (the same script with five settings changed, rerun
over N = 10..20), then `registration_table_v4.py`, which runs no simulation and
selects the registered rule from the rerun's sweep.
**Outputs:** `data/power_v4/`, `data/power_v4_n10_20/` and
`data/registration_v4/`.

`simulate_power_v4.py` was rewritten rather than edited. It runs on the frozen
list after verifying its checksums, and it produces the figures that go into the
preregistration. The rerun is a copy of it with **five values changed** —
`OUTPUT_ROOT`, `SAMPLE_SIZES`, `RULE_SUMMARY_N`, `C1_PARTICIPANT_N` and
`C1_CONTROL_ABSOLUTE` — and not one line of code or comment, so the two runs are
comparable by construction.

The registered figures in this document and in section 13 come from the rerun and
the registration table, **not** from the original run, whose sample-size grid
skipped N = 11, 13, 17, 18 and 19 — and a rule fixed per N cannot have a hole in
it. The two input files the registration table read are recorded by their
SHA-256 in its `run_meta.json`:

- `rule_sweep.csv` — `0d5d03282c954924…`
- `c1_power.csv` — `f7033f3160e9fe4d…`

### Per-person classification, in both directions

For each feature, thresholds are computed from **random choosers on the frozen
list**: an upper and a lower threshold at the same percentile, one at each end.
Each person is then classified per feature as **detected in the predicted
direction**, **detected in the opposite direction**, or **not detected**.

The registered thresholds:

| | upper | lower |
|---|---|---|
| colour | **+0.286** | −0.290 |
| fog | **+0.488** | −0.486 |

The ratio between the two features' thresholds is **1.70**, against **1.68**
predicted from the information each feature's trials carry — an independent
arithmetic check that the thresholds are a property of the design rather than of
the simulation.

Testing both directions is what answers problem 4. A person with a small taste
for hazier pictures is no longer counted as evidence, because the rule requires
detections in the predicted direction clearly to outnumber those in the opposite
direction.

### The decision rules, and how one was chosen

Candidate rules take the form **"at least *k* detections in the predicted
direction and at most *m* in the opposite direction"**. Three families were
swept:

- **count-absolute** — a fixed count degenerates as N grows, since it becomes
  trivially easy. That is a reason not to register it at a large N, and the
  registered range stops at 20, where the two count families overlap heavily:
  at N = 15, `floor(0.20 × 15) = 3`, so "m ≤ 3" is one rule under two names with
  identical power and identical false-claim rates. It is therefore **searched**
  rather than excluded, because excluding it would silently resolve ties that
  the stated tie-break should resolve;
- **count-proportional** — the family the registered rule comes from, and the
  one a rule expressible both ways is reported as;
- **the sign test** — bounded by alpha by construction, and unable to fire at all
  below 5 detections. **Not registered**, but computed at every N and reported
  as a **robustness comparison**.

### The registered rule

**The count rule alone, calibrated at each N from 10 to 20, with no switch to
anything else.** Section 13 is the table, and it is the authoritative statement
of the rule and of its power.

A combination — the count rule below a switch point and the sign test from it —
was considered first and rejected. The case for it was that the count rule can
*lose* power as N grows: when m = 0 a single opposite-direction detection vetoes
the claim, so every non-responder added is another chance of that veto. **That is
true of the m = 0 rule and of no other.** It was read off a summary cell — a weak
effect with a quarter of participants responding — in which the rule selected
happened to be the m = 0 one and in which neither rule reaches useful power
anyway.

Selected instead in the **planning cell** — half the participants responding, a
moderate effect, lapse 0.1 — the chosen rule has m > 0 from N = 11 on, and the
sign test does not overtake it at any N in 10..20, for either feature, in
either the planning or the robustness cell. The margin is at least **+0.037** on
colour and **+0.010** on fog. The scan behind that statement is still computed
and reported, because it is the evidence for registering the count rule rather
than an assertion that it is better.

### Ties, and how they are broken

Two rules whose planning-cell power differs by less than **0.01** are treated as
*tied*. With 2,000 simulated studies per cell, a power near 0.93 carries a Monte
Carlo standard error of about 0.006, so the more powerful rule by a thousandth is
not reliably the more powerful rule, and taking the nominal winner would register
a choice made by noise.

Among the tied, the rule with the **lowest upper-bound worst false-claim rate**
is taken, then the smaller m, then the smaller k, then the family. The measured
rate comes first because it *is* the quantity a stricter rule is wanted for; a
smaller m is a proxy for it that can disagree — at fog N = 20 the proxy would
choose `k ≥ 5, m ≤ 1` over `k ≥ 7, m ≤ 3`, which tolerates fewer
opposite-direction detections but also requires fewer same-direction ones, so it
is not unambiguously the stricter rule.

On the registered run the tie-break decided **five** selections — colour at
N = 11, 15 and 19, fog at N = 11 and 20 — at a cost of at most **0.007215** of
power, which is inside the simulation's own error. A further **39** firings were
one rule appearing under both family names, with identical k, m, power and
false-claim rate; those are bookkeeping and are recorded as such rather than
reported as decisions.

Both orders were run, and they **selected the same rule in every cell at every
N**: in each of the five, the rule with the smaller m also carried the lower
false-claim rate, so the proxy and the measurement agreed. The ordering is
therefore a statement of principle on this run rather than a change of rule. It
is stated because a future run could have the two disagree, and then the
registered rule must not depend on which of them was written down first.

### Integer calibration: the power does not rise monotonically

The rule is re-selected at every N under the false-claim ceiling, and the
threshold m can only take integer values. Where the ceiling forces a stricter m,
the power **falls even though N grew** — most visibly at N = 13, where m drops
from 2 to 1 and power falls by 0.066 on colour and 0.034 on fog, and at N = 20 on
fog, where the twentieth participant lowers power from 0.825 to 0.797.

This is a property of calibrating an integer threshold against a fixed error
rate, not a defect in the rule and not a reason to prefer a smaller N. **Power is
quoted per N from the table in section 13, which is authoritative**, so nothing
in the registered analysis depends on the sequence being monotone. What is *not*
available is the claim that another participant always helps.

### Headroom under the false-claim target

The target is **0.05 on the upper bound** of the 95% Monte Carlo interval, and it
is not tightened. The upper bound is itself the safeguard against a selection
made by simulation noise, so a rule that clears it is certified however thin the
margin. The margin is nevertheless reported for every N, in the `false_headroom` column
of `registration_table.csv` and in section 13.

Three registered rules sit above 0.045 and are marked:

| | rule | upper bound | point estimate | margin |
|---|---|---|---|---|
| colour N = 18 | `k ≥ 6, m ≤ 2` | 0.04933 | 0.04739 | **0.00067** |
| fog N = 12 | `k ≥ 5, m ≤ 2` | 0.04755 | 0.04496 | **0.00245** |
| fog N = 17 | `k ≥ 5, m ≤ 1` | 0.04516 | 0.04491 | **0.00484** |

Each is admissible as registered. None is a reason to change the rule; each is a
reason not to describe the false-claim rate as comfortably under the target at
those sizes.

### What the tuning costs

The rule is calibrated at one cell, so the planning-cell power is its best case.
`registration_table.csv` therefore carries both rules' power in two further
cells at every N — the weaker equal-behaviour pair at half share
(`cost_lower_half`), and the planning weights at a quarter share
(`cost_planning_qtr`), both at lapse 0.1. In both, **the sign test would
outperform the registered rule for colour from N = 15**; for fog the count rule
stays ahead throughout. That belongs beside the registered power, not behind
it: the planning cell is registered because it is the scenario in which a study
of 10 to 20 participants can succeed at all, not because it is the only
one considered.

Each rule was scored for **false-claim rate** under random choosers and under
each taste spread, and for **power** under each effect scenario, per feature and
for the two claims together.

Admissibility is judged on a stated set of taste spreads. Three standards were
defined — `practical` {0.25, 0.5}, `practical_at_boundary` {0.25, 0.5, 0.75} and
`strict` {0.25, 0.5, 0.75, 1.0} — and the **strict** standard is registered: the
rule must hold its false-claim rate at or below alpha at every spread in that
set, including a taste spread as wide as the floor itself.

**Admissibility carries a Monte Carlo margin.** A rule is admissible only if the
**upper bound of a 95% interval** for its false-claim rate clears the target, not
the point estimate. The standard error comes from the delta method, with
`SE(p) = sqrt(p(1−2p)/(2N))` for the symmetrised trinomial rate. On the final
run this moved **9 selections**: for example strict / colour / N = 12, where the
point estimate 0.0498 has an upper bound of 0.05023, so the rule `k ≥ 3, m ≤ 0`
is replaced by `k ≥ 4, m ≤ 0` at a power cost of 0.112. Strict on its own set
scores **0.04826** and all 72 tabulated rules clear the target on the upper
bound.

**A two-stage control-based alternative was costed and rejected.** It would have
estimated the taste spread from the control group and then chosen the standard
accordingly. It exceeds alpha at its own boundary — 0.113 as first specified,
0.064 for both repaired variants — so it cannot be registered. Its results are
kept in the report, because they are the reason the strict standard was chosen
rather than an assumption. The control-based taste-spread estimate is reported
descriptively only.

That section reuses the sweep's **measured** null rates rather than measuring its
own. Two Monte Carlo estimates of the same quantity disagree, and a rule the
sweep had certified then scored above alpha in the control section purely as an
artefact of measuring twice. The nulls are shared per spread, and they are
**symmetrised** exactly as the sweep does — `p = (p_predicted + p_opposite)/2`,
used for both directions — because a zero-centred taste preference is symmetric
by construction, so the two counts estimate one quantity. Strategy-contaminated
nulls are **not** symmetrised: the strategy loads positively on both features, so
its asymmetry is the signal. After both fixes the two sections agree to
**0.00000000** across 32 matched rule × spread cells.

### Effect sizes, stated behaviourally

Scenarios are defined in raw feature units, as in step 6, and then translated
into something a reader can judge: **the probability of choosing the predicted
image on a typical informative trial**. That statement is primary; the per-SD
weights are secondary. Responder-share and spread scenarios are kept, and
**lapse rates** are swept from none up to a value that would still pass the catch
and consistency checks.

There is no pilot, so nothing here is a point prediction: effect sizes and lapse
rates are presented as ranges of scenarios.

### Prevalence

Population prevalence is estimated as an alternative to a fixed count, and is
registered **descriptively only** — reported with its interval, for both
features, with no "excludes zero" inference.

The estimator needed correcting. The obvious one, α̂ = opposite/n, is biased by a
factor of (1 − γ): the opposite-direction rate is contaminated by the responders
who are not detected. The model was reparameterised as

```
c = (1 − γ)·α        p_predicted = γ·sensitivity + c        p_opposite = c
```

giving the moment estimator **γ̂ = (predicted − opposite) / (n × sensitivity)**,
which scored a worst deviation of 0.000000 against the realised prevalence. The
joint (γ, α) posterior does **not** identify the pair — a true γ of 0 gives a
posterior mode of 0.86 — which is why the reparameterisation is necessary rather
than cosmetic.

**What that check does and does not establish.** The worst deviation of 0.000000
is a statement that the estimator is unbiased **by construction under its own
model**, verified by exact enumeration over the joint (predicted, opposite)
counts. It is arithmetic, not evidence: it says the algebra is right, not that
the model holds of real participants. It should not be reported as an empirical
validation.

**The enumeration assumes one sensitivity.** It divides by the detection rate of
a **weak responder, 0.30/SD**, interpolated from this run's own detection curve
at lapse 0. A prevalence estimate is only as good as the sensitivity it divides
by, and reporting it at a single value states the result as if that value were
known. The three strengths bracket it:

| responder | weight | colour sensitivity | fog sensitivity |
|---|---|---|---|
| weak | 0.30/SD | 0.50 | 0.22 |
| moderate | 0.50/SD | 0.86 | 0.51 |
| strong | 0.70/SD | 0.99 | 0.79 |

**The registered form.** Prevalence is reported as (a) the **counts** in each
direction, per feature, which assume nothing, and (b) the **estimate under each
of the three sensitivities above**, each with its interval, labelled with the
sensitivity assumed. No "excludes zero" inference is drawn from any of them. The
fog column of that table is systematically lower than the colour column for the
reason in section 14: the same person is detected less often on fog, so the same
count divides by a smaller number.

`prevalence_estimated.csv` was not present in the run read for this document, so
the typical interval widths are not quoted here. They are in that file as
`E_hpdi_hi − E_hpdi_lo` and should be added before the preregistration is filed.

### The control group (C1)

Controls do the same task, imagining how the world looks to someone experiencing
derealization. They are simulated with weights drawn from a **shared concept**
distribution that explicitly includes the "pick the more edited image" strategy —
positive on both features. This is what answers problem 5: specificity rests on
the comparison with controls, not on the shape of the dose-response.

The comparison is **two one-sided Mann–Whitney tests** on the per-person weights,
Holm-corrected across the two features. Power is reported across numbers of
participants and controls, and it **depends strongly on the responder share and
on how much of the shared-concept distribution is strategy**; both dependencies
are stated beside the result rather than buried.

The group test uses a **one-sided** JZS Bayes factor computed over the effect
size, and the null-evidence figures were regenerated with that same one-sided
factor. An earlier version declared a one-sided setting and never used it, so the
quoted factors were two-sided; the rewritten form agrees with an independent
g-integral formulation to 3e-11.

#### C1 at the sizes that could be recruited

`c1_compact.csv`, at **one fully stated scenario**. Every scenario dimension is
fixed — effect size, responder share, strategy rate and both lapse rates — and
the figures must not be averaged across them: `c1_power.csv` crosses 13
effect/feature/share combinations with 3 strategy rates and 7 lapse pairs, which
answer different questions. The full filtered set is in `c1_cells.csv` and is
not a table to read.

**effect on colour** — 0.5/SD, share 0.5, strategy 0, lapse 0.1/0.1

| participants | 20 controls | 30 controls | 40 controls |
|---|---|---|---|
| 10 | 0.494 | 0.547 | 0.563 |
| 15 | 0.603 | 0.662 | 0.697 |
| 20 | 0.674 | 0.747 | 0.779 |

The figure is `power_colour_gated`: the Holm-corrected colour test, counted
after the attention gate.

**effect on fog** — 0.7/SD, share 0.5, strategy 0, lapse 0.1/0.1

| participants | 20 controls | 30 controls | 40 controls |
|---|---|---|---|
| 10 | 0.424 | 0.496 | 0.531 |
| 15 | 0.517 | 0.587 | 0.626 |
| 20 | 0.594 | 0.654 | 0.724 |

The figure is `power_fog_gated`, on the same basis.

**no effect anywhere** — share 0.5, strategy 0, lapse 0.1/0.1. This is the
**false-positive rate**, and it is reported beside the power because a group
comparison without it is half a statement.

| participants × controls | colour | fog | either |
|---|---|---|---|
| 10 × 20 | 0.0265 | 0.0225 | 0.0470 |
| 10 × 30 | 0.0225 | 0.0215 | 0.0425 |
| 10 × 40 | 0.0245 | 0.0195 | 0.0430 |
| 15 × 20 | 0.0235 | 0.0190 | 0.0420 |
| 15 × 30 | 0.0240 | 0.0180 | 0.0405 |
| 15 × 40 | 0.0205 | 0.0310 | 0.0500 |
| 20 × 20 | 0.0225 | 0.0205 | 0.0425 |
| 20 × 30 | 0.0215 | 0.0265 | 0.0465 |
| 20 × 40 | 0.0265 | 0.0225 | 0.0485 |

**What ten more controls buy.** From 20 participants and 30 controls to 40
controls:

- colour: 0.747 → 0.779 (**+0.033**)
- fog: 0.654 → 0.724 (**+0.070**)

The gain is larger on fog than on colour, which is the argument for the "40 if
they are easy to recruit" branch of the plan. At 20 participants, ten more
controls buy about as much on fog as five more participants would.

Two properties of the no-effect grid are worth naming. The per-feature rate sits
between **0.018 and 0.031** and the two-feature disjunction between **0.041 and
0.050**, so Holm is holding the familywise rate where it should rather than
leaving the pair of tests to inflate it. And **there is no cross-feature
leakage**: when the effect is on colour, the fog test fires at 0.032–0.046, and
when it is on fog, the colour test fires at 0.031–0.040 — both at the null rate
in the table above. A C1 result therefore names a feature, which is what the
specificity claim needs.

### The attention gate

Participants are screened on catch-trial failures and on fast responses. Both
measures are combined with **Holm correction across the two**, not by OR-ing two
0.05 tests — which gave a realised false-alarm rate of 0.09. After Holm the
matched false-alarm rate is 0.0380 on average and 0.0508 at worst.

`GATE_ACTION = "inconclusive"`: a failed gate makes the result inconclusive
rather than silently dropping data. A pre-registered zero-catch-failure secondary
analysis is available, and is valid at the planned sizes (false positives ≤
0.0725, usable in ≥ 96.5% of runs) but **unavailable when both groups lapse at
0.2**, where only 5.4% of runs have a usable subsample.

Catch arithmetic: with four live keys, the probability a lapsing participant
fails a given catch trial is `0.75λ`, over 10 catch trials, with exclusion at
**≥ 3 failures**. Response times are recorded on every pair trial, so the RT
component of the gate is registered **unconditionally**. An RT re-check was
briefly registered as a second gate and then demoted to a diagnostic: it fires
99.9% of the time whether or not an effect is real, so the secondary's power is
exactly zero in every cell.

### Sequential stopping

**Not registered.** The plan is a single final analysis at a maximum N and an end
date. The earlier advice to stop at a particular sample size came from a jump in
a fixed-count rule rather than from any loss of power, and the sample-size sweep
now extends past that jump and shows what extra participants actually buy.

Two scripts were retired as part of this: `simulate_power_seq_v4.py` and
`plot_power_seq_v4.py`. They are superseded and **must not be run** — they use
the retired `z_sat` / `z_fog` predictors, a ridge penalty rather than Firth, and
point at trial lists that no longer exist. Their audit is in this step's
documentation.

### Checks

The two required ones pass: with pure random choosers the per-person
false-detection rate at each threshold is close to nominal, and re-running the
old one-directional rule on the current settings reproduces the old figures.
Beyond those, the script verifies the frozen package's checksums, cross-checks
`FROZEN.json`, asserts the oriented predictors in the list match its own signing,
validates its own settings before any work begins, and locates the shipping draw
by its flag rather than by seed — an early version filtered on seed alone across
seven versions × twenty draws and took the first match, which gave a detection
rate of 0.864 where the correct figure was 0.050.

Figures use an Okabe-Ito palette validated for colour-vision deficiency, with
markers and line styles as a secondary encoding so no series is distinguished by
hue alone.

---

## 13. The registered design in numbers

```
analysed trials                 180   60 sat_only / 60 fog_only / 30 together / 30 conflict
presented trials                211   + 15 consistency, 10 catch, 6 practice
images in the frozen package    349
catch trials                     10   no-image mode; exclude at >= 3 failures
seed                              42   trial list version "both"
predictors                 pred_sat = -d_sat/39.182,  pred_fog = +d_fog/9.348
per-person model            Firth bias-reduced logistic, no intercept
registered thresholds    colour +0.286 / -0.290,  fog +0.488 / -0.486
threshold ratio                 1.70   (1.68 predicted from trial information)
admissibility standard        strict   {0.25, 0.5, 0.75, 1.0} x floor
                                       certified on the upper bound of a 95% MC interval
strict rate on its own set   0.04826
registered rule          the count rule alone, calibrated per N, no switch
participants              10 to 20   maximum 20 or the end date, whichever first
controls                        30   40 if they are easy to recruit
analysis                    single   no sequential stopping
grid block                 2 blocks x 4 screens x 3 rows, 6 columns per screen
grid scene sets            2 x 12, matched pairs, swapped between participants
grid package                    144   images, 43.7 MB
```

### The registered rule at each N

**These are the authoritative power figures.** Each row is fixed before the data
arrive and depends only on the final number of analysable participants, which is
set by a maximum N and an end date, never by a result. Power is at the planning
cell: half the participants responding, colour 0.50/SD (p = 0.6191 on a typical
informative trial), fog 0.70/SD (p = 0.6104), lapse 0.1. "Margin" is how far
the rule's upper-bound worst false-claim rate sits below the 0.05 target; ⚠ marks
a margin under 0.005.

| N | colour rule | colour power | margin | fog rule | fog power | margin |
|---|---|---|---|---|---|---|
| 10 | k >= 4, m <= 0 | 0.591 | 0.0276 | k >= 4, m <= 0 | 0.443 | 0.0271 |
| 11 | k >= 5, m <= 2 | 0.834 | 0.0172 | k >= 5, m <= 2 | 0.565 | 0.0149 |
| 12 | k >= 5, m <= 2 | 0.841 | 0.0053 | k >= 5, m <= 2 | 0.584 | 0.0024 ⚠ |
| 13 | k >= 5, m <= 1 | 0.775 | 0.0157 | k >= 5, m <= 1 | 0.550 | 0.0144 |
| 14 | k >= 5, m <= 1 | 0.861 | 0.0112 | k >= 5, m <= 1 | 0.703 | 0.0101 |
| 15 | k >= 6, m <= 3 | 0.932 | 0.0111 | k >= 5, m <= 1 | 0.798 | 0.0070 |
| 16 | k >= 6, m <= 2 | 0.919 | 0.0130 | k >= 5, m <= 1 | 0.780 | 0.0053 |
| 17 | k >= 6, m <= 2 | 0.915 | 0.0065 | k >= 5, m <= 1 | 0.760 | 0.0048 ⚠ |
| 18 | k >= 6, m <= 2 | 0.954 | 0.0007 ⚠ | k >= 5, m <= 1 | 0.803 | 0.0053 |
| 19 | k >= 7, m <= 3 | 0.969 | 0.0151 | k >= 5, m <= 1 | 0.825 | 0.0062 |
| 20 | k >= 7, m <= 3 | 0.969 | 0.0079 | k >= 5, m <= 1 | 0.797 | 0.0079 |

The power does not rise monotonically with N; see "integer calibration" in
section 12. The figures above are per N and are the ones to quote.

**Fewer than 10 analysable participants.** If the end date arrives with fewer
than 10, **neither existence claim is made.** The counts in each direction
are reported descriptively, per feature, and nothing is concluded from them. C1
is still run: it compares two groups and does not depend on the per-person rule.
This is registered in advance so that a small sample is not met with a decision
taken after seeing it.

**Not the registered analysis.** Two outputs of the step-8 run select a different
rule for each scenario, and neither is quotable as registered power:
`existence_power.csv`, which takes the best admissible rule separately for each
responder share, and the behavioural section, which selects under the
`practical` standard rather than the `strict` one. A real study cannot choose its
rule after learning its responder share. Registered power comes only from the
table above.

---

## 14. Known properties and limitations

1. **The fog axis has no level above the original.** The contrast-up column was
   dropped as invisible, so the normal-appearance block's fog answers are
   censored at the original (section 11). The count of such answers is reported
   per block and the shift treated as a lower bound.
2. **The two features' claims are not equally detectable.** The informative
   colour trials are about 1.0 corpus SD apart and the fog trials about 0.64, so
   the same sensitivity produces a weaker choice bias on fog. The scenario pairs
   are matched on the **choice probability** — within 0.0087 at the planning
   pair — and that still does not match the **per-person detection rate**: at
   the planning pair a person is detected **0.8017** of the time on colour
   against **0.6675** on fog, a gap of **0.1342**. At the weaker pair the two
   agree to 0.0100. A person affected equally on both features will therefore
   show a weaker signal on fog, so the fog claim's lower power and lower
   prevalence estimate are properties of **the stimuli** — not of the analysis,
   and not of the phenomenon. This belongs in the preregistration as a
   limitation.
3. **The colour axis is not symmetric in feature units.** The first step up is
   1.41× the first step down, because the ladder is calibrated on corpus
   percentiles and that distribution is not symmetric in value.
4. **The original sits in a different column on the two grid screens.** Comparing
   which column people pick across the two features is partly a position effect;
   the within-feature shift is not affected.
5. **Specificity rests on the control group, not on the dose-response.** A rising
   dose-response is consistent with a "pick the more edited image" strategy, and
   any claim of specificity that cites the curve alone is unsupported.
6. **The two features' weights are not comparable within a person.** They are in
   corpus-SD units of two different features. A person with a larger fog weight
   than colour weight is not thereby "more sensitive to fog"; only comparisons of
   the same weight between people or groups are meaningful.
7. **Prevalence is descriptive, and conditional on an assumed sensitivity.** It
   is reported with an interval and no inferential claim. The correction divides
   by the detection rate of a responder of assumed strength, so the estimate is
   reported under three stated sensitivities rather than one (section 12), and
   the unbiasedness check establishes the algebra under the model rather than the
   model itself.
8. **The registered power is calibrated at one cell.** In weaker cells the sign
   test would outperform the registered count rule for colour from N = 15. Both
   cost cells are in `registration_table.csv`, and the crossover is stated in
   section 12 beside the registered power rather than after it.
9. **Three registered rules clear the false-claim target by less than 0.005.**
   Colour N = 18 (margin 0.00067), fog N = 12 (0.00245) and fog N = 17 (0.00484).
   Each is admissible on the upper bound of the Monte Carlo interval, which is
   the safeguard, but the margin is thin and is marked in section 13. The
   false-claim rate should not be described as comfortably under the target at
   those three sizes.
10. **Power is not monotone in N.** The threshold *m* is an integer calibrated
    against a fixed error rate, so at some N a stricter rule is forced and the
    power falls although N grew. The per-N figures in section 13 are
    authoritative; "more participants always help" is not a claim this design
    supports.
11. **The C1 power depends on quantities not yet known.** Responder share and the
    strategy content of the shared-concept distribution both move it
    substantially; the reported figures are scenario-conditional.
12. **No pilot.** Every effect size and lapse rate is a scenario, not an estimate.
13. **The set swap is binomial, not exact.** The client draws it with probability
    0.5 and records it, because the study has no server-side counter. At 20
    participants the split is 10 ± 2 about three quarters of the time and exactly
    even only half the time; at 10 participants it is 5 ± 1 about two thirds of
    the time. **The imbalance matters more at these sizes than it did at the
    sample sizes this design was first drawn for**, so the recorded swap code
    must be in every data file or the imbalance cannot be checked at all.
14. **188 rendered images are unused.** They remain in the stimulus set;
    `unused_cells.md` is the record of why.

---

## 15. Directory map

| Script | Reads | Writes |
|---|---|---|
| `calibrate_levels_v4.py` | corpus features and index | `data/levels_v4_ext/` |
| `select_scenes_v4_corrected.py` | `data/levels_v4_ext/` | `data/select_scenes_v4_ext/` |
| `make_screening_page_corrected.py` | `data/select_scenes_v4_ext/` | `screening.html` beside the crops |
| `build_stimulus_set_v4_corrected.py` | `screening_list_done.csv`, `data/levels_v4_ext/` | `data/stimulus_set_v4_ext_haze_eq/` |
| `analyse_pairs_v4_corrected.py` | `stimulus_manifest.csv` | `data/pair_analysis_v4_ext/` |
| `build_trials_v4_corrected.py` | `all_pairs.csv` + `stimuli/` | `data/trials_v4_ext/<version>/` |
| `choose_colour_condition_v4.py` | the four candidate lists | `data/colour_choice_v4/` |
| `check_robustness_v4.py` | the four candidate lists | `data/robustness_v4/` |
| `freeze_trial_list_v4.py` | `data/trials_v4_ext/<chosen>/` | `data/frozen_trial_list_v4/` |
| `build_grid_block_v4.py` | `data/stimulus_set_v4_ext_haze_eq/`, step 3's script | `data/grid_block_v4/` |
| `simulate_power_v4.py` | `data/frozen_trial_list_v4/` | `data/power_v4/` |

**Reading the folder names.** The suffix `_ext` marks this round's run — the one
whose ladders carry levels above the original. Folders carrying `_ec` are from
the previous round and are superseded; they remain readable because step 3's
reproduction check compares against them. Suffixes appear on **folders only**; no
script filename carries one.

---

## 16. What is not included

**Covered by the stimulus document, not here:** the corpus and the scene
selection, the two manipulations and their order, the three image sizes, the
verbatim-copy convention, the pair types and their thresholds, and why `haze_eq`
was chosen over plain haze.

**Covered by the axes document:** how the nine v4 axes were built and validated,
and the argument for pairing fog with `contrast_local` and vividness with
`sat_mean`.

**Outside all three documents:** the experiment's front-end code, the
session-level analysis scripts, and the final preregistration text.

---

## 17. Open items

This section is the only place that names the working plan,
`prompt_stimulus_power.md`, because these are gaps measured against it rather
than properties of what was built.

### Departures from the plan, to be declared as such

1. **The grid JSON is not the format the plan specified.** The plan asks for grid
   files "in the same format as the existing `grid_trials.json`". That file was
   never supplied, and a new schema was written. The structural change is
   unavoidable — the old format is a flat list of four screens, with nowhere to
   put two blocks and two swapped sets — but four field renames are not
   (`column` → `display_column`, `level_label` → `column_label`,
   `"haze"` → `"fog"`, and `n_columns` dropped). Keeping the old field names
   inside the new structure would let the existing grid rendering code be reused.
   This needs the real `grid_trials.json` to check against.
2. **The fog grid screen has no level above the original.** The plan requires
   "levels above the original, the original, and the edited levels" in both
   blocks, per feature. Colour has it; fog does not, for the reason in section
   11. This is a deliberate, documented deviation and should be stated as one.
3. **The shift is measured in feature units, not grid steps.** The plan says grid
   steps. Feature units are the better choice — the plan's own objection to
   grid-dependent units applies — but it is still a departure.

### Not done

4. **`experimental_design.md` was never supplied and so was never corrected.**
   The plan asks for two fixes in it: the model statement and the predictor
   signs. Those are exactly the errors in problem 3 of section 1, and nothing
   else in this round corrects them, because the corrections live in the scripts.
   If the preregistration's model is written from that document, the wrong model
   goes into it.
5. **The consolidated list of experiment-code changes.** The plan asks for a
   statement of what the front end needs in order to run two grid blocks and to
   read the new trial-list columns. The pieces exist — the catch sequence and
   catch-mode setting in `config.js`, `EXPECTED_COUNTS` in `validate.js`,
   dropping `correct_response` from the data file, and the grid package's
   `interface_spec.md` — but they have never been collected into one note.
   Regarding the new columns specifically: the front end needs to do nothing with
   `pred_sat` and `pred_fog` beyond not copying them into the data file.
6. **The remaining documentation edits.** `v4_stimulus_pipeline_explained.md`
   needs the new levels and how they were calibrated, the corrected band-sweep
   table, the fog manipulation described as contrast reduction, the grid blocks,
   and the frozen list's checksum and settings. It also needs one specific
   rewording: the statement about display rescaling should say that the
   within-pair ratio is preserved only under **linear** rescaling, and that
   display gamma is not linear. `v4_power_analysis_explained.md` needs the
   corresponding rewrite.

### Sources for the figures in this document

Every number comes from the run report of a final build. The reports used:

| section | step | report | run |
|---|---|---|---|
| 3 | 1 | `levels_report.txt` | — |
| 5 | 3 | `build_report.txt` | 10-19 |
| 6 | 4 | `pairs_report.txt`, `comparison.csv` | 10-17 17:37 |
| 7 | 5 | `trials_report.txt` | 10-18 11:02 |
| 7 | 5 | `comparison.csv`, `bins_by_version.csv` | 10-17 17:41 |
| 8 | 6 | `choose_report.txt` | 10-18 11:26 |
| 9 | 6b | `robustness_report.txt` | 10-18 16:28 |
| 11 | 7b | `grid_report.txt` | 10-19 12:37 |
| 12 | 8 | `power_report.txt` | 10-19 09:49 |
| 12, 13 | 8 | `power_report.txt`, N = 10..20 rerun | see `registration_v4/run_meta.json` |
| 12, 13 | 8b | `registration_report.txt` | 2026-10-19 23:19:20 |

**Two step-4 runs exist and only one is the build.** The run at 10-17 17:37 uses
`BALANCE_NORM = "corpus_sd"`, passes section 0b with 0 flips, reports OVERALL
PASS, and writes `data/pair_analysis_v4_ext/` — which is the folder step 5 reads.
The run at 10-17 17:39 uses `BALANCE_NORM = "scene_range"`, fails 0b with 908
flips, and writes `data/pair_analysis_v4_ext_legacy/`. The second is the
**diagnostic** that establishes why the normalisation changed, not a build, and
its balance figures should not be quoted as the set's.

**The step-5 seven-version comparison predates the catch-count change.** Its
tables show 207 presented trials, because they were built before catch went from
6 to 10. The shipped list's own report (10-18 11:02) shows 211 and is the
authoritative count. Nothing else in the comparison is affected: the analysed
trials, the collinearity and the colour-difference distributions are all
properties of the 180 analysed trials, which did not change.
