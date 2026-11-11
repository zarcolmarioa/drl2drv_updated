# drl2drv v5 stimulus set

Stimulus set for the v5 image-choice experiment of the drl2drv project:
47 scenes x 49 versions (one unedited) = 2,303 images,
512 x 512 px PNG.

Built by `build_stimulus_set_v5.py` (a copy is in `provenance/`), settings
hash `7fd72262ba`, on 2026-11-11. Every path in this folder is relative to it.

## The design

Each scene is shown in a 7 x 7 grid of versions:

* **Contrast** (`contrast_level` c-3 ... c0 ... c+3, `contrast_step` -3 ... 3).
  Below the original: v4's brightness-equated haze (`haze_eq`). Above it:
  the same blend run backwards on RGB (`dehaze_rgb`, v4's `m_contrast_up`).
  At every contrast level a chroma factor (`comp_k`) restores the scene's
  mean saturation, so contrast levels leave `sat_mean` unchanged.
* **Colour** (`colour_level` s-3 ... s0 ... s+3, `colour_step` -3 ... 3):
  v4's CIELAB chroma edit, scaling a*, b* below the original and blending
  toward each pixel's in-gamut ceiling above it (`colour_k`).
* **Levels**: equal steps in corpus SD, solved separately for every scene,
  at -1, -2/3, -1/3, 0, +1/3, +2/3, +1 SD of `contrast_local` (SD
  9.348) and of `sat_mean` (SD 39.182), measured on the
  stimulus path (photograph -> 512 px -> 256 px). The colour edit is
  applied after the contrast edit.
* The cell c0/s0 (`is_original`) is the unedited photograph (its central
  512 px square).

Why this design: Chosen after test_edit_levels_v5.py, run 4aa278d54d (committed as v5/stimulus_levels_data/4aa278d54d/), and a visual check of its viewer. Method A4_haze_cc_rgb: contrast down with v4's brightness-equated haze (haze_eq), contrast up with the same blend run backwards on RGB (v4's m_contrast_up, 'dehaze_rgb'), and at every contrast level a chroma factor that restores the scene's mean saturation, so the contrast levels leave saturation unchanged. Colour: v4's CIELAB chroma edit in both directions. Levels: definition (b), equal steps in corpus SD solved per scene, at +-1/3, +-2/3 and +-1 SD on both features, so the unedited original sits in the middle of both ladders and 'edited' points both ways. No clipping budget: the upward contrast levels clip (reported, not used to cap or exclude), and the +3 images were judged acceptable. All 47 v4 scenes, none set aside, for variety. No grid block in v5.

## Precision

On the two ladders through the original, colour levels are within
0.002 SD of their targets and contrast levels within 0.049 SD
(77 of 282 contrast levels miss by more than 0.01 SD). Where
both features are edited, the colour edit's effect depends on the
contrast edit before it: those cells sit up to 0.51 SD from the
nominal colour level (`colour_dev_sd`). Use each image's measured values
(`d_con_sd`, `d_sat_sd`, the feature columns), not the nominal levels.

The upward contrast levels clip (pixels pushed past 0 or 255): median
over scenes c+1 7%, c+2 15%, c+3 27%. This is reported (`clip_*` columns), not used to
cap levels or drop images.

## Files

| file | content |
|---|---|
| `stimuli/` | the 2,303 images, `<scene_id>__c<k>__s<k>.png` (k = n3 ... 0 ... p3; p = plus, n = minus) |
| `stimulus_manifest.csv` | one row per image: scene, levels, file name, edit parameters, targets, every measurement |
| `cell_plan.csv` | the 49 cells of every scene |
| `sources.csv` | the 47 source photographs: Places365 file ID, SHA-256, size |
| `level_deviations.csv` | each scene's 12 levels: target, achieved, parameter |
| `cell_deviations.csv` | each cell's distance from its nominal levels, summarised over scenes |
| `level_summary.csv` | per level: measured change, cross-talk, clipping, artefacts, distance from typical |
| `reach.csv`, `reach_grid.csv` | the per-scene parameter search the levels were solved from |
| `reference_stimulus_path.csv` | the 20,000-image corpus reference (mean, SD, percentiles) |
| `envelope_violations.csv` | images with a feature outside the corpus 1-99% range (reported) |
| `identity.csv`, `unedited.csv` | the no-change checks |
| `report.txt`, `checks.csv` | the build report and every check |
| `settings.json`, `run_meta.json` | settings, versions, timings |
| `provenance/build_stimulus_set_v5.py` | the script that built this folder |
| `CITATION.cff` | how to cite (placeholders to complete) |
| `checksums.sha256`, `FROZEN.json` | integrity: `sha256sum -c checksums.sha256` |

Main manifest columns: `scene_id`, `category`, `group`, `indoor`,
`contrast_level`, `colour_level`, `contrast_step`, `colour_step`,
`is_original`, `use` (all `pair`), `filename`, `places365_file`;
`contrast_param` (haze h, negative = dehaze), `comp_k`, `colour_k`;
`contrast_target_sd`, `colour_target_sd`, `d_con_sd`, `d_sat_sd`,
`contrast_dev_sd`, `colour_dev_sd`; `z_sat`, `z_con`; `dist_corpus`,
`dist_orig` (combined) and `_sat`, `_con` (per feature); the 18 corpus
features and `fine_texture`, `noise_sigma`, `edge_overshoot`; clipping
(`clip_0.5lv` ... `clip_8lv`), `halo_share`, `noise_ratio`,
`band_new_empty`; `envelope_n`, `envelope_features`; `png_sha256`.

## Sources and licence

The images are derived from photographs of the Places365 dataset
(Zhou, Lapedriza, Khosla, Oliva & Torralba, 2018, "Places: A 10 million
image database for scene recognition", IEEE TPAMI 40(6), 1452-1464).
The Places365 documentation states that the copyright of all images
belongs to the image owners.

**Licence of this set: to be confirmed.** The source photographs are
not included. `sources.csv` lists each one's Places365 file ID (relative
to the dataset's image folder) and SHA-256, so the set can be rebuilt
exactly: obtain Places365, check the hashes, and run the script in
`provenance/` (it also needs the corpus files listed in its settings).

## Citation

See `CITATION.cff` (authors, version and DOI to be completed).
