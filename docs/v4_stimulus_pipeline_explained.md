# How the v4 experiment stimuli were selected, edited and measured

This document explains, for a reader with no other context, how the stimuli for
the v4 image experiment were built: which photographs were chosen, how they were
edited along two visual dimensions, how the edits were calibrated, and what was
measured to check that the set does what the design needs.

- **Project folder:** all paths are relative to the project folder
  (`drl2drv_updated/`). Scripts are in its pipeline folder.
- **Script conventions:** every script is standalone, with a SETTINGS block at
  the top and no command-line arguments. Each writes to its own output folder, a
  report (`*_report.txt`), and never overwrites an existing result — a second run
  writes to `<dir>_2`, `_3`, and so on (`ON_EXISTING = "new"`).
- **Scope:** only the construction of the stimuli. Where the two axes came from,
  and why each is paired with its image feature, is a separate part of the
  project — see the companion document, *How the v4 derealization axes were
  built, validated and linked to image features*, referred to below as **the axes
  document**. The experimental design (task wording, screening questionnaire,
  response grid) and the power analysis are not covered here.
- **Every number in this document comes from a script's SETTINGS block or from
  the run report of the final build.** Where a value was changed between runs,
  the value given is the one the final set was built with.

## Guide: sections and scripts

| Section | Step | Script(s) |
|---|---|---|
| 1 | Key ideas, and the vocabulary | – |
| 2 | Geometry, and what a "rung" means | – |
| 3 | Step 1: calibrate the two ladders | `calibrate_levels_v4.py` |
| 4.1 | Step 2: shortlist candidate scenes | `select_scenes_v4.py` |
| 4.2 | Step 2b: screen the shortlist by hand | `make_screening_page.py` + by hand |
| 5 | Step 3: render the stimulus set | `build_stimulus_set_v4.py` |
| 6 | Step 4: enumerate every usable pair | `analyse_pairs_v4.py` |
| 7 | Step 5: sample the trial list and package it | `build_trials_v4.py` |
| 8 | The final set in numbers | (from the reports) |
| 9 | Known properties and limitations | – |
| 10 | Directory map, and what the folder names mean | – |
| 11 | Suggested wording for a methods section | – |
| 12 | What is not included | – |

---

## 1. Key ideas, and the vocabulary

**The experiment.** On each trial a participant sees two versions of the *same*
photograph side by side and chooses which one looks more like how the world
looked to them. The two versions differ along one or both of two visual
dimensions. From each participant's choices, the analysis estimates a weight on
each dimension — how much that person's choices were driven by colour, and how
much by fog.

**The two dimensions.** They come from two of the nine v4 axes (axes document,
section 8):

| | Fog dimension | Colour dimension |
|---|---|---|
| Axis | `v4::A07 fog, blurry, hazy` | `v4::A13 dull, less vivid, colorless` |
| Image feature | `contrast_local` | `sat_mean` |
| Within-category β | −0.27 | −0.40 |
| Manipulation | `haze_eq` | `chroma` |

**The two features, stated exactly.** These definitions are inherited verbatim
from step 04 of the axes pipeline and are easy to misread:

- **`contrast_local`** is the **standard deviation of the Laplacian of the grey
  image** (OpenCV Rec. 601 grey, 3×3 Laplacian). It measures fine-detail and
  edge energy. It is **not** RMS contrast and not a windowed local contrast; the
  old pipeline's `sharpness` feature was its square. Prose in this project
  sometimes calls it "local contrast" as a readable label — that is the same
  measurement, not a different one.
- **`sat_mean`** is the **mean HSV saturation** (OpenCV, 0–255).

`FOG_SIGN = -1.0` appears in several scripts. It flips the raw
`contrast_local` difference so that a positive value always means "image A is
foggier", because fog *lowers* this feature.

**The grid.** Each kept scene is rendered as a 5 × 5 grid: five colour levels
(the unmanipulated image plus four chroma reductions) crossed with five fog
levels. That is 25 **cells** per scene. A **ladder** is one dimension's list of
five levels; a **rung** is one level on it.

**The four pair types.** Every trial shows two cells of the same scene. The pair
is classified from the cells' *measured* features, never from the parameters:

- **`sat_only`** — same fog row, so only colour differs. Chroma scaling leaves
  L\* alone, so `contrast_local` is unchanged along a row.
- **`fog_only`** — different fog rows whose measured saturations happen to
  coincide. Necessary because fog drains colour as a side effect: an unmatched
  foggy-versus-clear comparison confounds fog with colour loss.
- **`together`** — foggier *and* less saturated. The way the world works, since
  fog takes colour with it. The largest expected effect.
- **`conflict`** — foggier *and more* saturated. The cues point opposite ways, so
  the choice reveals which one the participant follows. The only type that can
  show a crossover between people.

**Balance.** Raw differences on the two features are not comparable: `sat_mean`
spans roughly 45 units within a scene while `contrast_local` spans roughly 11. So
each difference is divided by that **scene's own range** for that feature
(`z_sat`, `z_fog`), and `balance = z_sat / z_fog`. Near 1 means neither cue is
handed an advantage.

---

## 2. Geometry, and what a "rung" means

This section is the conceptual key to the pipeline. A reader who skips it will
think the ladders are arbitrary.

### Three image sizes, used for three different things

| Size | Used for | Setting |
|---|---|---|
| 336 px | computing the region CLIP sees, via `clip_box` | `CLIP_PX = 336` |
| 512 px | applying the manipulations, and the shipped stimulus | `IMAGE_SIZE = 512` |
| 256 px | measuring the features | `FEATURE_SIZE = 256` |

Every scene is first reduced to the **central square** — exactly the region CLIP
sees, reproduced to within rounding by `clip_box` — then resized to 512 px in one
step. Manipulations are applied at 512. Features are then measured on a further
downsample to 256, because that is the size at which the corpus was measured.

### The edit space and the measurement space are not the same

There are four distinct places where the images are edited in one set of units
and measured in another:

1. **Size.** Manipulated at 512, measured at 256. A stimulus therefore travels
   original → 512 → 256, whereas the corpus travelled original → 256. The extra
   resize matters: `contrast_local` is the only one of the usual contrast
   measures that is strongly resize-sensitive, because resampling smooths exactly
   the fine detail it measures. The final calibration run measured the worst such
   discrepancy as a ratio of **0.9186** — the two-resize path reads
   `contrast_local` about 8% lower.
2. **Colour space.** `m_chroma` edits **CIELAB chroma** C\*ab; the feature
   measured is **HSV saturation**. These are related but not identical
   quantities, so `k = 0.65` does *not* mean "saturation × 0.65".
3. **Light space.** `m_haze_eq`'s restoring gain is applied in **linear light**,
   while the quantity it restores (mean grey) and all features are measured in
   **sRGB**. That is why the gain is found by a 14-step bisection rather than
   solved in closed form.
4. **Parameter versus feature.** The manipulation parameters are `h` (haze
   strength) and `k` (chroma multiplier); the design cares about
   `contrast_local` and `sat_mean`.

### What a rung means, and why that resolves all four

**The ladders are calibrated on the measured feature, not on the parameter.** A
rung named `p25` means: *this image's measured feature, on the stimulus
measurement path, now sits at the corpus 25th percentile.* The `h` and `k` values
are **outputs** of a search, not inputs chosen by hand.

This is why the mismatches above cost nothing. Whatever the edit does in CIELAB
or in linear light, and however the extra resize shifts things, the rung is
defined by where the feature lands when measured the way the corpus was measured.
The calibration run sets `REFERENCE_MODE = "stimulus"` for exactly this reason:
the corpus reference percentiles are recomputed on the stimulus measurement path
(20,000 images) rather than taken from the corpus feature file, so target and
measurement are in the same units.

**One rule follows, and it is load-bearing: the corpus reference and the stimuli
must be measured on the same path.** If the reference percentiles are taken from
the corpus-path file while the stimuli are measured on the stimulus path, the fog
ladder is wrong by roughly the 8% above, which propagates into every fog level.

### The verbatim-copy convention

The feature code (`legacy_features`, `all_features`) and the manipulation
functions (`m_haze`, `m_haze_eq`, `m_chroma`, `srgb_to_lin`, `lin_to_srgb_f`,
`mean_grey`, `to_u8`, `central_square`, `clip_box`) are **copied byte-for-byte**
into every script that needs them, rather than imported from a shared module.
Each script also runs a start-up check that recomputes features for a few images
and compares them against the stored corpus values (`FEATURE_CHECK_RTOL = 1e-4`;
all runs reported `1.1e-09`).

This duplication is deliberate. A reader's instinct will be to refactor it into
one module; doing so would put the corpus/stimulus comparability at the mercy of
an import path, which is the one thing this pipeline cannot afford to get wrong.

### The two manipulations, exactly

- **`m_haze`:** `I' = (1 − h)·I + h·A`, where **`A` is the image's own airlight**
  — the mean RGB of the brightest 1% of pixels of the **unmodified** 512 px
  square (`HAZE_AIRLIGHT_TOP = 1.0`). It carries that scene's own highlight
  colour and is typically bright, not mid-grey. The formulation follows He, Sun &
  Tang (2009).
- **`m_haze_eq`:** `m_haze`, then a linear-light gain restoring the original mean
  grey, found by 14-step bisection. This is the family the stimuli were built
  with.
- **`m_chroma`:** convert to CIELAB, multiply a\* and b\* by `k`, leave L\*
  unchanged, convert back, with gamut mapping (`GAMUT_ITERS = 16`,
  `GAMUT_MARGIN = 0.995`).

**Order matters:** `MANIP_ORDER = ("haze", "chroma")`. Haze goes first because
the airlight must be taken from the unmodified image. One consequence of the
linearity of the haze blend is worth recording: because `A` is constant across
the image and the Laplacian of a constant is zero, `contrast_local` after plain
haze is exactly `(1 − h)` times its original value, and the *colour* of the
airlight cancels out of the fog feature entirely.

---

## 3. Step 1: calibrate the two ladders

**Script:** `calibrate_levels_v4.py`. **Output:** `data/levels_v4_ec/`.

Note that the script's name carries no suffix while its output folder does. The
folder name records the run (section 10); the filename does not.

The problem this step solves: a given `h` removes different amounts of detail
from different photographs, so a ladder defined in parameter units would not mean
the same thing from scene to scene. The ladder is therefore defined in corpus
percentiles of the measured feature.

What the script does:

1. **Defines the target population.** `STATS_POPULATION = "everyday"` restricts
   the reference to the 82 everyday scene categories listed in
   `everyday_categories.csv` (`MIN_POPULATION = 1000`), colour photographs only.
   The final run: 1,111,881 corpus images → 1,078,535 colour → **209,119** in the
   `everyday_colour` population → 185,480 eligible for calibration.
2. **Computes the corpus reference percentiles on the stimulus measurement
   path.** `REFERENCE_MODE = "stimulus"`, `N_REFERENCE_IMAGES = 20_000`, with its
   own seed (`REFERENCE_SEED = 20250101`) so changing the calibration sample does
   not move the reference.
3. **Draws a calibration sample** of `N_CALIB_IMAGES = 150` images (seed 42,
   `MIN_SIDE = 512`, `EXCLUDE_GREYSCALE = True`).
4. **Builds a monotonic grid** for each manipulation — haze from 0 to
   `HAZE_MAX = 0.95`, chroma from 1 to `CHROMA_MIN = 0.0`, `GRID_N = 24` steps —
   measuring both features at every grid point on every image.
5. **Solves for each rung.** For each target percentile in
   `TARGET_PERCENTILES = [45, 40, 35, 30, 25, 20, 15, 10, 5, 1]`, finds the
   parameter at which the **median** image's measured feature change equals the
   gap between the corpus median and that percentile. Levels that would clip
   pixels beyond `CLIP_MAX_NEW = 0.01` are capped.
6. **Calibrates three dimensions:** `colour` (chroma), `fog` (plain haze) and
   `fog_eq` (haze_eq). Rung for rung, `fog` and `fog_eq` are calibrated to the
   **same `contrast_local` targets**, so a rung name denotes the same intended
   detail loss in both families.
7. **Assembles candidate ladders** (`CANDIDATE_LADDERS`) and reports each one's
   span and cross-talk, so the choice is made from evidence.
8. **Runs eight start-up and consistency checks**, all of which passed in the
   final run: feature code (`1.1e-09`), percentile reproduction (0.000 SD),
   monotonic grids, identity points (h = 0 and k = 1 move nothing, ±0.000),
   sample-versus-reference agreement (1.4 SE), and an envelope check.

**Result.** 30 levels calibrated, 0 capped. The candidate ladders, with each
one's span in corpus SD and its worst cross-talk — defined as the worst
`|cross / own|` ratio over the rungs, i.e. how much a manipulation moves the
*other* dimension's feature relative to its own:

| Ladder | Dimension | Parameters | Span | Worst cross-talk |
|---|---|---|---|---|
| `colour_v2` **(chosen)** | colour | k = 1, 0.652, 0.425, 0.320, 0.168 | 1.35 SD | **0.02** |
| `fogeq_v2` **(chosen)** | fog_eq | h = 0.077, 0.140, 0.224, 0.287, 0.387 | 1.11 SD | **0.73** |
| `fog_v2` | fog (plain haze) | h = 0.140, 0.240, 0.359, 0.436, 0.549 | 1.11 SD | 1.20 |
| `fog_mild` | fog | h = 0.046 … 0.240 | 0.53 SD | 1.51 |
| `fog_wide` | fog | h = 0.093 … 0.549 | 1.24 SD | 1.34 |
| `fog_deep` | fog | h = 0.093 … 0.718 | 1.70 SD | 1.34 |
| `fogeq_wide` | fog_eq | h = 0.024 … 0.387 | 1.37 SD | 0.82 |
| `fogeq_deep` | fog_eq | h = 0.050 … 0.568 | 1.70 SD | 0.74 |

**Read the cross-talk column.** Scaling chroma barely touches `contrast_local`
(0.02). The reverse is not true: hazing an image desaturates it as a side effect,
and plain haze moves `sat_mean` *more than it moves its own feature*
(1.20 > 1). Brightness-equated haze cuts that to 0.73. This asymmetry is the
single most important property of the stimulus set, and the main reason the
design needs `conflict` trials — see sections 6 and 9.

The final ladders, as the stimulus set carries them:

```
colour  k=1 (unmanipulated), p25=0.6516, p10=0.4247, p05=0.3204, p01=0.1679
fog_eq  h: p35=0.0771, p25=0.1403, p15=0.2245, p10=0.2870, p05=0.3875
```

**Envelope check (CHECK 6).** The step also asks what each level does to the
features the design does *not* intend to move, against the corpus 1–99% range.
Eight violations, all at the deepest rungs, the largest being `fog p01` lifting
`lum_p05` 3.708 SD above the corpus p99. Deepest rungs are meant to be extreme;
this is reported so the choice is deliberate rather than accidental. The chosen
ladders stop before `fog p01`.

**Files:** `levels.csv` (one row per dimension × level: target, parameter,
achieved change, share reaching target, cross-talk), `per_image_levels.csv`,
`corpus_feature_stats_v4.csv` (the reference percentiles — read by steps 2 and
3), `calibration_grid.csv`, example grids, `levels_report.txt`.

---

## 4. Step 2: choose the scenes

### 4.1 Shortlist the candidates

**Script:** `select_scenes_v4.py`. **Output:** `data/select_scenes_v4_ec/`.

Reads `data/levels_v4_ec/levels.csv` and
`data/levels_v4_ec/corpus_feature_stats_v4.csv`. Opens no model and needs no GPU:
the axis scores of the unmanipulated scenes are a lookup from step 06 of the axes
pipeline.

What the script does:

1. **Filters the corpus** to plausible candidates: short side at least
   `MIN_SIDE = 512`; `EXCLUDE_GREYSCALE`, `EXCLUDE_TONED` and
   `EXCLUDE_DUPLICATES` all true; `EXCLUDE_PEOPLE` with
   `MAX_PERSON_FRAC = 0.0`; restricted to the 82 everyday categories.
2. **Applies neutrality bands.** A scene must start out *unremarkable* on both
   axes and both features, or it arrives already foggy or already drab and has no
   room to move. Axis band `AXIS_PCTL_LOW = 35.0` to
   `AXIS_PCTL_HIGH = 65.0`; feature bands 30–75% on both `sat_mean` and
   `contrast_local`, applied in the `everyday_colour` population. A band sweep is
   reported so the setting is chosen from data:

   | band | after axes | after features |
   |---|---|---|
   | 25–75% | 32,543 | 9,489 |
   | 30–70% | 20,776 | 20,776 |
   | **35–65% (chosen)** | **11,757** | **11,757** |
   | 40–60% | 5,201 | 5,201 |
   | 45–55% | 1,303 | 1,303 |

3. **Measures reachability.** For `N_CANDIDATES = 2000` candidates
   (`MAX_PER_CATEGORY = 500`), renders all 25 cells at 512 px and measures each.
   A scene passes when its measured change reaches `REACH_FRACTION = 0.6` of the
   level's target — a rule in the same units as the calibration, so it needs no
   re-derivation when the grid changes. It also requires the scene to afford at
   least one **saturation-matched fog pair**: two cells whose `sat_mean` differ by
   no more than `PAIR_SAT_TOL = 3.0` while `contrast_local` differ by at least
   `MIN_FOG_GAP = 4.2`.
4. **Shortlists** `N_FINAL = 1800`, round-robin across categories so the list is
   spread rather than filled from whichever categories sort first, then writes
   crops and contact sheets for screening.

**Result.** 2,000 candidates from 82 categories; **2,000 of 2,000 passed
reachability**; shortlist 1,800 scenes across 82 categories, with 1,800 crops, 40
contact sheets and 40 full grid sheets for spot-checking. Reachability margins
were comfortable: `reach_fog` ranged from 0.695 to 1.300 (median 0.963) against
the 0.6 threshold, and every scene's best saturation-matched fog gap was at least
8.15 against the 4.2 requirement.

**One check warns, and it is worth understanding (CHECK 4).** The script is
written to produce *one* scene list serving both fog families, on the reasoning
that they are calibrated to the same feature targets. CHECK 4 verifies this
rather than assuming it: on 24 scenes × 5 rungs it renders both families at
matched rung names and compares the achieved `contrast_local` change. The worst
relative difference was **0.468**, against `FOG_EQUIV_TOL = 0.02` — a WARN. Step
3 repeats the check and reports the same 0.468.

The reason is that **the calibration is a median and the check is per scene.**
Rung for rung the two families agree for the median image by construction, but
the mapping from dose to measured feature is scene-dependent and the two families
have *different* scene-dependence, because haze_eq's restoring gain is a
darkening whose effect on fine detail varies with the scene's tone distribution.
Hence the very different doses — haze_eq's `p25` is h = 0.140 against plain
haze's 0.240 — and a per-scene spread of up to 47%.

This did not affect the selection, for three reasons that can be checked in the
reports: reachability excluded nobody (2,000 of 2,000), so the ladder changed
nothing about eligibility; the shortlist was drawn by category round-robin under
a fixed seed, not by fog reach; and the screening crops are **unmanipulated**, so
the fog family is invisible at the stage where the final scenes were chosen.
Step 3 then verified affordance per scene on the real haze_eq stimuli and passed
47 of 47. It is recorded here because it came out right rather than being safe by
construction.

**Files:** `candidates.csv`, `band_counts.csv`, `reachability.csv`,
`grid_measurements.csv`, `shortlist.csv`, `screening_list.csv`,
`crops/<scene_id>.jpg`, `contact_<n>.png`, `grid_sheets/<id>.png`,
`selection.png`, `reachability.png`, `select_report.txt`.

### 4.2 Screen the shortlist by hand

**Script:** `make_screening_page.py` (the tool). **The decision is made by a
human.** This is the one stage where judgement, not a threshold, selects the
stimuli.

The script builds `screening.html`, a self-contained page needing no server or
internet. It shows one shortlisted crop at a time, large, on a neutral surround,
records keep/discard as a coloured frame, saves after every decision, and can be
closed and reopened. Scenes are grouped by category with a per-category counter,
because earlier rounds produced six gymnasia out of twenty indoor scenes. **Not
pressing keep means discard:** anything left undecided is written out as
`keep = 0`, so stopping early still gives a well-defined result. Pressing Save
downloads `screening_list_done.csv`, and step 3 reads the rows with `keep == 1`.

**The criteria applied**, in the screener's own terms:

- The photograph shows an **easily recognisable place**.
- **No people, animals or comparable foreground subjects.**
- The environment is one **most people will have experienced** something similar
  to, so the scene is not exotic or specialist.
- A **spread of lighting conditions and times of day** across the kept set.
- A **similar number of indoor and outdoor** scenes.

Note that these are judgements about the *unmanipulated* scene. The crops carry
no haze and no chroma change, so screening cannot be influenced by how a
particular scene responds to the manipulations.

**Result.** 47 scenes kept from 1,800 screened. The `reason` column of the
worksheet was left blank, so the criteria above are the authoritative record of
why scenes were kept; per-scene rationales were not recorded.

---

## 5. Step 3: render the stimulus set

**Script:** `build_stimulus_set_v4.py`. **Output:**
`data/stimulus_set_v4_ec_haze_eq_2/` (and, for the comparison,
`data/stimulus_set_v4_ec_haze_2/`).

The script is run **once per fog family**, with `FOG_FAMILY` and `FOG_DIMENSION`
set accordingly (`haze` / `fog`, or `haze_eq` / `fog_eq`); the output directory
name records which. `RENDER_MODE = "preview"` measures everything and runs every
check without writing images; `RENDER_MODE = "full"` writes the stimuli and the
manifest. The final set was built with `FOG_FAMILY = haze_eq` and
`RENDER_MODE = "full"`.

What the script does:

1. **Reads the screened list** (`screening_list_done.csv`, rows with
   `keep == 1`) and the ladders from `data/levels_v4_ec/`.
2. **Renders every cell** of every kept scene: 5 colour × 5 fog = 25 cells, with
   `INCLUDE_UNMANIPULATED_COLOUR = True` making the first colour column the
   untouched image. Manipulations at 512 px in `MANIP_ORDER`.
3. **Measures every rendered cell** at 256 px and writes the measurements into
   the manifest. From here on, *everything downstream uses these measured
   values*, never the parameters.
4. **Runs five checks:**
   - **CHECK 1, feature code** — recomputes features for a few images against
     the stored corpus values. PASS (`1.1e-09`).
   - **CHECK 2, envelope** — which cells take a feature the design does not
     intend to move outside the corpus 1–99% range, now with *both*
     manipulations applied, which step 1 could only ask one at a time.
     `ENVELOPE_MAX_CELL_FRAC = 0.25`. PASS, **2.1% of cells outside**.
   - **CHECK 3, fog families** — the per-scene equivalence of section 4.1. NOTE
     (0.468) against `FOG_EQUIV_TOL = 0.05`.
   - **CHECK 4, affordance** — can each scene supply at least
     `MIN_PAIRS_PER_TYPE = 3` pairs of all four types, on its measured features?
     PASS, **47 of 47 scenes afford all types**.
   - **CHECK 5, coverage** — are there at least `MIN_PAIRS_PER_BIN = 40` pairs in
     each of `N_MAGNITUDE_BINS = 3` difference-magnitude bins, per type? PASS.
5. **Reports the balance of the final set** on an indoor/outdoor label
   (`BALANCE_BY = "indoor"`). The label is a judgement written into the script's
   settings, not a corpus column, and is carried in the manifest so it can serve
   as a covariate.

**Result.** 47 scenes, 37 categories, **1,175 stimuli** written (47 × 25).
Indoor/outdoor 22 / 22 with 3 ambiguous. At most 3 scenes from any one category
(`downtown`, `japanese_garden`). Groups: domestic 10, street/urban 9, academic 9,
everyday outdoor 8, transit indoor 8, retail/food 3. Elapsed 64.5 s.

Envelope detail (CHECK 2), cells outside the corpus range by feature:

| feature | cells | worst (SD outside) |
|---|---|---|
| `lum_p95` | 50 | 1.99 |
| `lum_p05` | 35 | 0.95 |
| `val_mean` | 26 | 0.69 |
| `lum_mean` | 25 | 0.43 |
| `lum_skew` | 25 | 0.79 |
| `lum_p50` | 10 | 0.08 |

For context, the previously piloted set had 95% of its images outside the corpus
p99 on `lum_p05` at its own top haze level. 2.1% is a large improvement and is
reported rather than silently accepted.

**Files:** `stimuli/` (the 1,175 PNGs), `stimulus_manifest.csv` (every image with
its measured features, level labels, parameters, category, group and indoor
label), `stimulus_set.csv`, `grid_cells.csv`, `envelope_violations.csv`,
`pair_affordance.csv`, `fog_equivalence.csv`, grid sheets, `build_report.txt`.

---

## 6. Step 4: enumerate every usable pair

**Script:** `analyse_pairs_v4.py`. **Output:** `data/pair_analysis_v4_ec/`.

Reads `stimulus_manifest.csv` and nothing else — no images, no encoder — and runs
in seconds. `MANIFESTS` is a list, so both fog families are analysed under
**identical thresholds** and compared side by side. That comparison is the point
of this step: it is how the fog family was chosen.

What the script does:

1. **Enumerates every ordered within-scene pair** of cells and classifies it from
   the measured features. Ordered rather than unordered because left/right
   position is counterbalanced, which keeps the difference distribution symmetric
   about zero. Thresholds:
   - `sat_only`: same fog row and `|d_sat| ≥ MIN_DSAT_SAT_ONLY = 17.0`
   - `fog_only`: `|d_sat| ≤ PAIR_SAT_TOL = 3.0` and
     `|d_fog| ≥ MIN_DFOG_FOG_ONLY = 4.2`
   - mixed: `|d_sat| ≥ MIN_DSAT_MIXED = 8.0` and
     `|d_fog| ≥ MIN_DFOG_MIXED = 3.0`, split into `together` or `conflict` by
     whether `d_fog` and `d_sat` share a sign
2. **Assesses conflict balance.** Computes `z_sat`, `z_fog` and their ratio, and
   flags a conflict pair as `balanced` when the ratio falls in
   `BALANCE_LO = 0.6` to `BALANCE_HI = 1.7`. The window is **held fixed across
   families on purpose** — fitting it per family would hide the very difference
   being compared — and the report prints what window each family's own data
   would suggest.
3. **Checks magnitude coverage** across three bins, and reports the bin edges so
   step 5 can sample evenly.
4. **Reports per-scene capability and concentration** — how much of each pool
   rests on a few scenes, and how many scenes could supply a given per-scene cap
   — so step 5's cap is chosen from evidence.

**Result: the two families side by side, identical thresholds.**

| quantity | haze | haze_eq |
|---|---|---|
| scenes / images | 47 / 1,175 | 47 / 1,175 |
| usable pairs | 13,560 | **15,004** |
| `sat_only` pairs | 1,390 | **2,030** |
| `fog_only` pairs | **2,238** | 1,750 |
| `together` pairs | 8,056 | **8,266** |
| `conflict` pairs | 1,876 | **2,958** |
| median balance ratio | 0.580 | **0.591** |
| fog cue stronger by | 1.72× | **1.69×** |
| balanced conflict pairs | 812 | **1,318** |
| share of conflict balanced | 0.433 | **0.446** |
| scenes with ≥ 3 balanced conflict | 46/47 | **47/47** |

**Why `haze_eq` was chosen.** It wins on nearly everything the design depends on:
a 58% larger conflict pool and a 62% larger balanced conflict pool, the only
family where every scene can supply balanced conflict trials, a better envelope
(2.1% of cells outside versus 12.3%), and much lower cross-talk onto the colour
feature (0.73 versus 1.20 — see section 3). It loses on `fog_only` pairs
(1,750 versus 2,238), which is ample for 60 trials. The one cost is conceptual:
`haze_eq` is a larger edit than a plain blend, since it adds a tone change.

**Note the balance figures.** In *both* families the fog cue runs about 1.7×
stronger than the colour cue across raw conflict pairs. This is physical, not a
flaw: fog drains colour, so a large *positive* saturation difference cannot
accompany a large fog difference. Left uncorrected it would bias a simple
proportion toward "fog wins" whatever participants were tracking — which is why
step 5 draws conflict trials only from the balanced subset. A regression on
per-unit weights is unaffected either way.

**Files, per family:** `all_pairs.csv` (**step 5 reads this**; every usable pair
with raw and normalised differences, the `balanced` flag, and `file_a` / `file_b`),
`pair_summary.csv`, `scene_capability.csv`, `weak_scenes.csv`,
`concentration.csv`, three figures. Plus `comparison.csv`, `comparison.png` and
`pairs_report.txt` on top.

---

## 7. Step 5: sample the trial list and package it

**Script:** `build_trials_v4.py`. **Output:** `data/trials_v4_ec/<family>/`.

Reads `all_pairs.csv` and the rendered images. Both families are built with
identical settings and a shared seed, so the two sessions differ only in which
fog the images carry.

**The trial counts.** `N_SAT = 60`, `N_FOG = 60`, `N_MIXED = 60` with
`MIXED_CONFLICT_FRAC = 0.5` → 60 `sat_only`, 60 `fog_only`, 30 `together`, 30
`conflict` = **180 analysed trials**. Plus `N_CONSISTENCY = 15`,
`N_CATCH = 6`, `N_PRACTICE = 6` → **207 presented**.

The counts are set by the precision of the **individual** estimate, not the group
one, because this is a design in which each participant is a data point: the
analysis estimates a per-person weight on each cue. A proportion from 60 trials
has a standard error near √(0.25/60) ≈ 0.065. Below about 30 trials per type the
within-person noise starts to dominate and individual differences become
unreadable.

What the script does:

1. **Samples each type with stratified draws** across three magnitude bins, so
   the dose-response curve has data at both ends rather than clumping at
   mid-range magnitudes. Bins are visited in random order so no bin gets first
   pick of the scenes.
2. **Caps per-scene contribution.** `MAX_PAIRS_PER_SCENE_PER_TYPE = 2`, counted
   **across the whole draw** for that type, not within each bin — counting per
   bin silently multiplies the cap by the number of bins. The cap matters because
   a couple of scenes have fog that barely drains colour and so hold an outsized
   share of the conflict pool.
3. **Draws conflict trials from the balanced subset only**
   (`BALANCED_CONFLICT_ONLY = True`).
4. **Draws practice and catch trials from `together` pairs the analysed set never
   uses**, so no pair is ever seen twice except a consistency repeat.
5. **Randomises sides to a forced 50/50 within each trial type** rather than by
   independent coin flips. The model has no intercept, so a participant with a
   side preference plus an unequal side assignment creates a constant with
   nowhere to go, which leaks into whichever weight correlates with the
   imbalance. Simulated on a participant tracking nothing but preferring one
   side, the spurious weight fell from 0.214 to 0.006 at a strong preference.
6. **Orders the session:** practice first, then everything interleaved rather
   than blocked, so no run of trials reveals the manipulation. Consistency
   repeats are spaced at least `CONSISTENCY_MIN_GAP = 25` trials from their
   originals, have their sides swapped, and are forced to come **second**, so the
   analysed presentation is the uncontaminated one.
7. **Checks the finished list** — collinearity of the two predictors, side
   balance per type, scene spread, no pair seen twice except repeats, and that
   every `repeat_of` resolves to an earlier trial showing the same image pair.
8. **Packages** only the images the list references, re-measuring
   `sat_mean` and `contrast_local` on the converted files
   (`REMEASURE_AFTER_CONVERT = True`) so the predictors describe what
   participants actually see. `REMEASURE_MAX_SHIFT = 0.6` guards against the
   stimuli directory being the wrong fog family. Images ship as PNG and are
   **not resized** — a smaller file skips the 512→256 smoothing and
   `contrast_local` would land about 9 units away.

**Result (haze_eq).** 207 trials, 180 analysed, 15 consistency, 6 catch, 6
practice. **`r(d_sat, d_fog) = −0.111`, VIF 1.01** — the number that decides
whether the two weights can be told apart; above about 0.6 they cannot. Side
balance: worst deviation from 0.50 is 0.000. 46 scenes used, at most 7 trials
from one scene. Consistency minimum gap 28. Re-measurement shift 0.000 (PNG).
Package: 342 images, 100.8 MB.

**What is and is not analysed.** Only the four real types. Consistency trials are
excluded on purpose: they are an independent quality check, and the agreement
rate is what distinguishes "does not track these features" from "was not
attending". A check that is also part of the analysis it is checking is no check
at all.

**Files:** `trial_list.csv` (full session in presentation order),
`trial_design.png`, `trials_report.txt`, and `package/` containing
`trial_list.csv` with relative paths, `trial_list.json`, `stimuli/`,
`manifest.csv`, `preload.json` and `README.txt`.

---

## 8. The final set in numbers

```
corpus                       1,111,881 images, 332 categories
everyday_colour population     209,119 images, 82 categories
reference for percentiles       20,000 images, stimulus measurement path
calibration sample                 150 images, seed 42
candidates measured              2,000  (2,000 passed reachability)
shortlist screened by hand       1,800  crops, unmanipulated
scenes kept                         47  in 37 categories, 22 indoor / 22 outdoor / 3 ambiguous
cells per scene                     25  (5 colour x 5 fog)
stimuli rendered                 1,175  PNG, 512 px
usable pairs (haze_eq)          15,004  of which 2,958 conflict, 1,318 balanced
analysed trials                    180  60 sat_only / 60 fog_only / 30 together / 30 conflict
presented trials                   207  + 15 consistency, 6 catch, 6 practice
predictor collinearity          -0.111  VIF 1.01
package                            342  images, 100.8 MB
```

---

## 9. Known properties and limitations

1. **Both ladders are one-directional.** Fog runs from none toward more, colour
   from full toward less. There is no extra-vivid or extra-crisp rung. The
   estimated weights describe sensitivity to the *loss* of detail and the *loss*
   of saturation, not two-sided sensitivity.
2. **The cross-talk is asymmetric.** Chroma barely moves `contrast_local`
   (worst ratio 0.02), but haze_eq moves `sat_mean` substantially (0.73; plain
   haze 1.20). A participant responding to the fog manipulation is partly also
   seeing a saturation change. This is why `conflict` trials exist and why they
   are drawn from the balanced subset.
3. **The two fog families are not matched sets.** At a shared rung name they
   differ in dose and in achieved detail removal per scene, by up to 47%
   (section 4.1). Nothing downstream depends on the equivalence, because every
   predictor is a measured per-stimulus difference — but the two sets should not
   be described as differing only in brightness.
4. **`contrast_local` is resize-sensitive.** About 8% per extra resize on this
   material. The corpus reference and the stimuli must be measured on the same
   path, and packaged images must not be resized.
5. **Deepest cells leave the corpus envelope.** 2.1% of cells take some
   unintended feature outside the corpus 1–99% range, worst `lum_p95` at 1.99 SD.
   This is far better than the previously piloted set and is reported so the
   choice is deliberate.
6. **One scene was retained deliberately despite envelope violations** — a night
   scene kept for diversity of lighting. Four of its five violating features are
   inherited from the unmanipulated photograph rather than caused by the
   manipulations.
7. **Display conditions are uncontrolled.** Participants' monitors differ in
   gamma, and browsers may render the 512 px stimuli at another size. Absolute
   feature values therefore drift from what was measured. The within-pair
   *ratio* survives any linear display rescaling, because resampling and the
   haze blend commute, and the analysis uses only within-pair differences — so
   this is a limitation on the absolute dose, not on the comparison.
8. **Per-scene screening rationales were not recorded.** The criteria in section
   4.2 are the authoritative record; the worksheet's `reason` column is empty.

---

## 10. Directory map, and what the folder names mean

| Script | Reads | Writes |
|---|---|---|
| `calibrate_levels_v4.py` | `data/selection/`, `data/features/`, `data/monochrome/`, `everyday_categories.csv` | `data/levels_v4_ec/` |
| `select_scenes_v4.py` | `data/levels_v4_ec/`, `data/scores_v4/`, `data/features/`, `data/monochrome/`, `data/checks_v4/` | `data/select_scenes_v4_ec/` |
| `make_screening_page.py` | `data/select_scenes_v4_ec/` | `screening.html` beside the crops; you download `screening_list_done.csv` |
| `build_stimulus_set_v4.py` | `screening_list_done.csv`, `data/levels_v4_ec/` | `data/stimulus_set_v4_ec_<family>[_n]/` |
| `analyse_pairs_v4.py` | `stimulus_manifest.csv` per family | `data/pair_analysis_v4_ec/<family>/` |
| `build_trials_v4.py` | `all_pairs.csv` + that family's `stimuli/` | `data/trials_v4_ec/<family>/` |

**Reading the folder names.** Suffixes are runs and variants, not versions of the
axes. They appear on **folders only** — no script filename carries one; each
script points at the folders it needs through its SETTINGS block.

| Suffix | Meaning |
|---|---|
| `_ec` | the run whose corpus reference percentiles were computed on the stimulus measurement path (section 2). **These are the folders the final set comes from.** |
| `_haze` / `_haze_eq` | which fog family that build used |
| `_2`, `_3`, … | a re-run; scripts never overwrite (`ON_EXISTING = "new"`) |

The final chain, which every number in this document comes from:

```
data/levels_v4_ec/                        -> levels_report.txt
data/select_scenes_v4_ec/                 -> select_report.txt
data/stimulus_set_v4_ec_haze_eq_2/        -> build_report.txt   (the shipped set)
data/stimulus_set_v4_ec_haze_2/           -> the comparison set
data/pair_analysis_v4_ec/{haze,haze_eq}/  -> pairs_report.txt
data/trials_v4_ec/{haze_eq,haze}/         -> trials_report.txt
```

Earlier folders without `_ec` are from the run whose reference percentiles were
taken from the corpus measurement path. They are superseded and should not be
read for v4 results.

**One setting to be aware of when reading the scripts.** `FOG_FAMILY` /
`FOG_DIMENSION` are toggled between runs, so a given copy of
`select_scenes_v4.py` or `build_stimulus_set_v4.py` shows whichever family was
run last, not necessarily the shipped one. The **reports** record what each run
actually used, and are authoritative. The shipped set is `haze_eq`.

---

## 11. Suggested wording for a methods section

> Stimuli were built from Places365 scene photographs restricted to 82 everyday
> categories (209,119 colour images). Two visual dimensions were manipulated:
> saturation, by scaling CIELAB chroma at constant lightness, and fog, by
> blending each image toward its own airlight (the mean RGB of its brightest 1%
> of pixels) and then restoring the original mean luminance with a linear-light
> gain. Five levels of each dimension were calibrated on the *measured* image
> feature rather than on the manipulation parameter: each level moves the median
> image's feature from the corpus median to a specified corpus percentile, where
> saturation is mean HSV saturation and fog is the standard deviation of the
> Laplacian of the greyscale image. Manipulations were applied to the central
> square at 512 px and features measured at 256 px, matching the measurement
> path used for the reference corpus. Candidate scenes were required to sit in
> the central 35–65% of both axes and the central 30–75% of both features, to
> reach at least 60% of every level's target, and to afford at least one
> saturation-matched fog pair; 1,800 shortlisted scenes were then screened by
> hand for recognisability, absence of people and animals, familiarity of the
> environment, and spread of lighting and of indoor versus outdoor settings,
> yielding 47 scenes in 37 categories (22 indoor, 22 outdoor, 3 ambiguous). Each
> scene was rendered as a 5 × 5 grid, giving 1,175 stimuli, and every cell was
> re-measured; all pair classifications and all analysis predictors use these
> measured values rather than the manipulation parameters. Trials paired two
> cells of the same scene and were classified as saturation-only (fog held
> constant), fog-only (measured saturation matched to within 3 units),
> congruent, or conflicting. Because haze desaturates as a side effect, the fog
> cue was about 1.7 times stronger than the colour cue across raw conflicting
> pairs; conflicting trials were therefore drawn only from pairs where the two
> cues, normalised by each scene's own feature range, fell within a factor of
> 0.6–1.7 of each other. The final list contained 180 analysed trials (60
> saturation-only, 60 fog-only, 30 congruent, 30 conflicting), sampled evenly
> across three difference-magnitude bins with at most two trials per scene per
> type, plus 15 consistency repeats, 6 catch and 6 practice trials. The two
> predictors were close to orthogonal across the analysed trials (r = −0.11,
> VIF = 1.01).

---

## 12. What is not included

**Covered by the companion axes document, not here:** how the nine v4 axes were
built from the literature, how they were validated, how the axis–feature
correlations were computed, and the argument for pairing fog with
`contrast_local` and vividless with `sat_mean`.

**Part of the project but outside this document:** the experimental design (task
wording, the word-rating screen, the screening questionnaire, the response grid),
the power analysis (`simulate_power_v4.py`, `simulate_power_seq_v4.py`), and the
session-level analysis scripts.

**Needed only to re-run these scripts:**

| Item | Used by |
|---|---|
| The Places365 images (paths in `image_index.csv`) | steps 1, 2, 3 |
| `data/selection/image_index.csv` (step 02 of the axes pipeline) | steps 1, 2, 3 |
| `data/features/features_crop.npy` and `feature_definitions.json` (step 04) | steps 1, 2, 3 |
| `data/monochrome/image_flags.csv` (step 04a) | steps 1, 2 |
| `data/checks_v4/duplicate_groups.csv`, `toned_images.csv` (step 09a) | step 2 |
| `data/scores_v4/` (step 06) | step 2 |
| `everyday_categories.csv` | steps 1, 2, 3 |

No GPU and no encoder is needed anywhere in this pipeline: the axis scores of the
unmanipulated scenes are a lookup from the axes pipeline, and no step in the
stimulus pipeline re-encodes an image. Step 1 takes about 5 minutes, step 2 about
4 minutes, step 3 about 1 minute, and steps 4 and 5 seconds. The hand screening
in step 2b is the long part.
