# drl2drv: remembered appearance of derealization episodes

This repository holds the code and documentation for a study of what the
surroundings looked like during an episode of derealization.

**The task.** People who have had a recognisable episode see two versions of the
same photograph and choose the one that looks more like how the world looked
during one specific recent episode. The two versions differ in:

- **local contrast**, a contrast reduction that models the "fog" or "haze"
  complaint;
- **colour saturation**, which models the "faded colour" complaint.

Each person makes 180 analysed choices. A per-person model estimates how
strongly their choices follow each feature. A control group does the same task,
imagining how the world might look to someone experiencing derealization.

The study is being preregistered. The hypotheses, the decision rules and the
sample-size reasoning are summarised in `docs/v4_stimulus_power_explained.md`.

The experiment itself (the web front end) lives in a separate repository,
`dpdr_experiment`.

---

## How the project is organised

The work runs in three stages. Each stage uses the outputs of the one before.

| Stage | Folder | What it does | Main document |
|---|---|---|---|
| 1. Axes | `axes_pipeline/` | Builds text-defined "derealization axes" in an image-embedding space (un2CLIP) from patients' own descriptions, and links them to measurable image features. This is why the study manipulates local contrast and saturation. | `docs/v4_axes_pipeline_explained.md` |
| 2. Stimuli and power | `stimulus_power_pipeline/` | Calibrates the manipulation levels, selects the scenes, renders the stimuli, builds and freezes the trial list, builds the grid blocks, runs the power analysis, selects the registered decision rule, and scores the final stimuli on the axes. | `docs/v4_stimulus_power_explained.md` |
| 3. Experiment | separate repository `dpdr_experiment` | The browser task that participants run. | that repository's README |

### Stage 2 scripts, in the order they run

| Step | Script | Output folder (under `data/`) |
|---|---|---|
| 1 | `calibrate_levels_v4_corrected.py` | `levels_v4_ext/` |
| 2 | `select_scenes_v4_corrected.py`, then `make_screening_page_corrected.py` (hand screening) | `select_scenes_v4_ext/` |
| 3 | `build_stimulus_set_v4_corrected.py` | `stimulus_set_v4_ext_haze_eq/` |
| 4 | `analyse_pairs_v4_corrected.py` | `pair_analysis_v4_ext/` |
| 5 | `build_trials_v4_corrected.py` | `trials_v4_ext/<version>/` |
| 6 | `choose_colour_condition_v4.py` | `choose_colour_v4/` |
| 6b | `check_robustness_v4.py` | `robustness_v4/` |
| 7a | `freeze_trial_list_v4.py` | `frozen_trial_list_v4/` (checksummed; never written to again) |
| 7b | `build_grid_block_v4.py` | `grid_block_v4/` |
| 8 | `simulate_power_v4_corrected.py` | `power_v4/` |
| 8 (rerun) | `simulate_power_v4_n10_20.py`, a copy of step 8 with five settings changed so that every N from 10 to 20 is simulated | `power_v4_n10_20/` |
| 8b | `registration_table_v4.py`, which selects the registered rule for each N from the rerun (no simulation) | `registration_v4/` |
| check | `score_stimuli_on_axes_v4.py`, which scores the final stimuli on the CLIP axes | `stimulus_axes_v4/` |

---

## Folder map

| Path | Contents | Status |
|---|---|---|
| `axes_pipeline/` | Stage 1 scripts; `axes/v4/` holds the hand-reviewed stage files (quotes, captions, negatives); `dpd_selected_scientific_literature/` holds the literature the quotes come from | current |
| `stimulus_power_pipeline/` | Stage 2 scripts (table above) | current |
| `data_power_v4/` | Selected outputs of stage 2, committed so they can be checked: `power_v4/` and `power_v4_n10_20/` (power reports and tables), `registration_v4/` (the registered rule for each N and the compact control-group table), `stimulus_axes_v4/` (axis scores of the final stimuli) | current results |
| `docs/` | Explanatory documents (below) | mixed, see below |
| `docs/claude_discussion/` | Records of the review discussions | record |
| `stimulus_pipeline/`, `power_analysis/` | Earlier versions of the stage 2 scripts | **superseded; do not run** (to be moved to an archive folder) |

### Documents in `docs/`

| File | Status |
|---|---|
| `v4_axes_pipeline_explained.md` | current: how the axes were built and validated |
| `v4_stimulus_power_explained.md` | current: what changed in the last round, the trial list, grids, axis scoring of the final stimuli, power and the registered rule for each N |
| `experimental_design.md` | current: the design as registered, including the parts not yet in the experiment code (marked *code change pending*) |
| `v4_stimulus_pipeline_explained.md` | partly superseded: still the reference for the corpus, scene selection, manipulations and pair types; out of date on what changed in the last round (see the note at its top) |
| `v4_power_analysis_explained.md` | superseded by `v4_stimulus_power_explained.md` |
| `prompt_stimulus_power.md` | the working plan given to the session that built stage 2 |
| `experimental_design_articles/` | key articles the design is compared against |

---

## Running the scripts

- **Layout.** The scripts are written to run from a single project folder that
  holds all scripts at its top level and all data under `data/`. Every path in
  the scripts is relative to that folder. This repository keeps the scripts in
  pipeline subfolders for readability, so copy a script to the project folder,
  or run it from there.
- **Conventions.** Every script is standalone:
  - a SETTINGS block at the top, and no command-line arguments;
  - its own output folder, never overwriting an earlier run (a rerun writes to
    `_2`, `_3`, …);
  - a human-readable report and a `run_meta.json`;
  - checks that stop the run when something is wrong.
- **Data not in the repository.** The image corpus, its embeddings and features,
  the un2CLIP checkpoint and the full stimulus set are too large to commit. The
  `.gitignore` lists the local data folders.
- **Environment.** Python 3 with numpy, pandas, scipy, Pillow, OpenCV and
  matplotlib; the axes and scoring steps also need PyTorch, torchvision and
  transformers, and a GPU is recommended.

---

## Where to start reading

1. This README.
2. `docs/v4_stimulus_power_explained.md`, sections 1 and 2 (what changed and the
   units), then section 13 (the registered design in numbers).
3. `docs/v4_axes_pipeline_explained.md` for why these two image features were
   chosen.
