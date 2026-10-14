# How the v4 derealization axes were built, validated and linked to image features

This document explains, for a reader with no other context, how the nine **v4 axes** were created, how they were tested, how their correlations with low-level image features were computed, and why the experiment pairs **fog with local contrast** and **vividless with saturation**.

- **Project folder:** all paths are relative to the project folder (`drl2drv_updated/`). Scripts are in its pipeline folder.
- **Script conventions:** every script is standalone, with a SETTINGS block at the top and no command-line arguments. Each writes to its own output folder, a report (`*_report.txt`) and a `run_meta.json` (settings, input fingerprints, check results).
- **Scope:** only the steps that produced the final axes and their validation. How the experiment's stimuli are made is a separate part of the project and is not covered here.

## Context: earlier axis versions (v2 and v3)

The project has had three generations of derealization axes. **v4 is the one used now**, and it is the only one this document describes. The earlier versions matter here only because their axes still appear in many v4 files.

- **v2:** the original axes. A language model extracted anchor words from the literature with the help of an example in its prompt. The negative anchors were antonyms the model invented, and many anchors can't be traced to a source. The earlier experimental design and its pilot data used **v2's "foggy" and "lifeless" axes**. In the new design, **v4's A07 (fog) and A13 (vividless)** take their place.
- **v3:** a rule-based revision of v2 (merges, removals, captions in the "a scene that looks …" template).
- **v4:** built from scratch, from verified quotes, with matched negatives (section 2). It was built without reference to v2 or v3.

The scripts of steps 05–09 keep every axis set in the same files and check that the earlier sets' results are unchanged. So when a v4 file contains a `set` column or `set::label` keys, you will see these sets:

| `set` | What it is |
|---|---|
| `v2` | the original v2 axes |
| `v2_merged` | v2 with "world-as-staged / scripted" and "simulation / digital world" merged into one axis |
| `v3a` | v3: v2 revised by rules |
| `v3b` | v3a without positive anchors that can't be traced to a source (the primary v3 version) |
| `v3b_neg` | v3b also without untraceable negative anchors |
| `v3a@…`, `v3b@…`, `v2@…` | template and baseline variants of those sets |
| `controls` | the six control axes (quality, age, weather, mood, threat, greyscale). Shared by every version and part of v4's validation (section 4.1). |
| `v4`, `v4@…` | the v4 axes and their variants: **the ones to use** |

**For any v4 analysis, select the rows with `set == "v4"`** (or `"v4@nocontrol"` for the sensitivity version), or match on the full key. Section 9 lists which files contain the earlier sets and which files to ignore. This document reports no v2 or v3 results.

## Guide: sections and scripts

| Section | Step | Script(s) |
|---|---|---|
| 2.1 | Stage A: phrase extraction from the literature | none (LLM chats) |
| 2.2 | Stage V: verify every quote against its PDF | `v4_stageV_verify.py` |
| 2.3 | Stage B: group phrases into candidate axes, then select axes CLIP can score | `v4_stageB_prepare.py`, LLM chats, `v4_stageB_compare.py` |
| 2.4 | Stage C: turn phrases into captions | `v4_stageC_prepare.py`, LLM chats, `v4_stageC_check.py` (+ `v4_captions_by_axis.py` for viewing) |
| 2.5 | Stage D: write the ordinary-perception negative for each caption | `v4_stageD_prepare.py`, LLM chats, `v4_stageD_check.py` |
| 2.6 | Scene-property check of the text axes | `v4_check17_scene_properties.py` |
| 2.7 | Stage D2: revise the pairs; the final axes | `v4_stageD2_revise.py` |
| 3.1 | Image content measurement and image selection | `01_measure_image_content.py` (reused), `02_select_images.py` |
| 3.2 | Image encoding | `03_encode_images.py` |
| 3.3 | Image features | `04_compute_features.py`, `04a_flag_monochrome.py`, `04c_compute_extra_features.py` |
| 3.4 | Feature selection (the 12 features) | `04b_select_features.py` |
| 4.1 | Build the axis vectors and test their stability | `05_build_axes_v4.py` |
| 4.2 | Score every image on every axis | `06_project_axes_v4.py` |
| 5 | How the axis–feature correlations were computed | `07_category_confounds_v4.py` |
| 6.1 | Validation: axis checks (controls, other axes, overlap) | `09a_axis_checks_v4.py` |
| 6.2 | Validation: item analysis (each caption pair) | `09b_item_analysis_v4.py` |
| 6.3 | Validation: manipulation sweep (causal test) | `09c_manipulation_sweep.py`, `09c_rescore_sweep_v4.py` |
| 7 | Main findings: the coefficient tables | (from 07, 09a, 09c) |
| 8 | Why fog ↔ `contrast_local` and vividless ↔ `sat_mean` | (from 07, 09a, 09b, 09c) |
| 9 | Earlier axis sets (v2, v3) in the v4 files, and misleading folder names | (05, 06, 07, 09a, 09b, 09c) |
| 10 | What this project folder contains and what is not included | – |

---

## 1. Key ideas in one page

- **What an axis is.** CLIP places images and sentences in the same 768-dimensional space. An axis is a direction in that space built from sentences. Each sentence is the template **"a scene that looks {caption}"**, filled with a caption that describes a derealization experience (the positive, e.g. "foggy") or ordinary perception (its matched negative, e.g. "clear").
- **v4 axes are built from pairs.** Every positive caption has one matched negative that changes only the word or words carrying the quality. The axis is the average difference between the positive and negative sentences of all its pairs, scaled to length 1.
- **Every positive comes from the literature.** Captions are minimal edits of verbatim quotes from 9 published sources containing first-person accounts of derealization. Each quote was checked against its PDF by a script.
- **An image's score** on an axis is the cosine between the image's CLIP embedding and the axis vector. It is converted to a z-score with the corpus mean and SD, computed over 1,111,881 Places365 scene photographs.
- **Correlations are within scene category.** Scene categories differ both in features and in scores (beaches are bright and colourful; jail cells are dim and grey). The analysis asks a narrower question: among images of the **same** category, do images that score higher on an axis differ in a feature? Both the score and the feature are centred on their category mean before any statistic is computed.
- **The two axes used in the experiment** are **A07 "fog, blurry, hazy"** (called *fog* here) and **A13 "dull, less vivid, colorless"** (called *vividless* here). The image features paired with them are local contrast (`contrast_local`) and mean saturation (`sat_mean`).

**Labels in the files.** Every file from step 05 on labels an axis as id + name (`A07 fog, blurry, hazy`) and keys it as `set::label` (`v4::A07 fog, blurry, hazy`). The same files also hold variants of each axis (`v4@photo`, `v4@place`, `v4@world`, `v4@centroid`, `v4@neutral`, `v4@nocontrol`) and the earlier axis sets (see "Context" above and section 9). **Always match on the full key**, never on a substring such as "fog", which also matches v2's "veiled / foggy".

---

## 2. Building the axes from the literature (Stages A–D2)

### Sources

Nine documents with first-person accounts of depersonalization/derealization. PDFs are in `dpd_selected_scientific_literature/`.

| ID | Document |
|---|---|
| CIAUNICA2022 | Ciaunica et al. (2022). Exploration of self- and world-experiences in depersonalization traits. |
| DILKES2024 | Dilkes (2024). Metaphor use in depersonalization derealization. |
| FINO2024 | Fino et al. (2024). Dispositional traits, characteristic adaptations, and narrative identity reconstructions in individuals with depersonalization and derealization. |
| MIYAZONO2021 | Miyazono (2021). Visual experience without presentational phenomenology. |
| PIENKOS2022 | Pienkos et al. (2022). "Robbed of my life": The felt loss of familiar and engaged presence in depersonalization/derealization disorder. |
| RADOVIC2002 | Radovic & Radovic (2002). Feelings of unreality: A conceptual and phenomenological analysis of the language of depersonalization. |
| SIERRA2001 | Sierra et al. (2001). The phenomenological stability of depersonalization: Comparing the old with the new. |
| SIMEON2023 | Simeon & Abugel (2023). *Feeling Unreal* (book). |
| TANAKA2018 | Tanaka (2018). What is it like to be disconnected from the body. |

**Principles applied throughout Stages A–D:**
- **Quotes only:** every positive anchor comes from a verbatim quote, verified against the PDF.
- **Nothing suggested by the prompts:** the prompts contain no derealization examples. The only illustration is from an unrelated topic (taste).
- **Phrases before axes:** phrases are collected before any grouping.
- **Two models:** each LLM stage is run independently by two models (Claude and ChatGPT), in fresh chats with memory, custom instructions and project files disabled. Their agreement is reported.
- **Rules first:** rules are fixed before each stage runs, and every decision is logged.

### 2.1 Stage A: phrase extraction (LLM chats, no script)

- **What:** one new chat per document per model. The model received one PDF and a fixed prompt asking for every verbatim phrase that describes how the world, objects or other people look or feel. Nothing was grouped at this stage.
- **Each item recorded:**
  - `quote`: the shortest verbatim span;
  - `context`: the full sentence;
  - `pdf_page`, `printed_page`;
  - `speaker`: participant/patient, historical case, questionnaire item, diagnostic criteria, author description, other;
  - `target`: surroundings, objects, other people, perceptual situation, mixed, unclear;
  - `sense`: vision, hearing, touch, other sense, not sense-specific;
  - `state`: altered, ordinary, unclear.
- **Two passes per chat:** pass 2 asks for missed items. The book received one pass.
- **Output:** `axes/v4/stageA/stageA_<model>_<DOC_ID>_pass1.json` / `_pass2.json` (34 files, never edited).

### 2.2 Stage V: verification and merging

**Script:** `v4_stageV_verify.py`. **Output:** `axes/v4/stageV/`.

What the script does:
1. **Checks the files:** reads the 34 Stage A files and the 9 PDFs, and checks fields, label values and ids.
2. **Matches every quote to its PDF.** The comparison keeps only letters and digits, so line breaks, hyphens and punctuation can't cause a mismatch. Each quote is marked:
   - VERIFIED (found on the stated page or within one page of it);
   - VERIFIED_WRONG_PAGE (page corrected);
   - NEAR_MATCH (similarity ≥ 0.95; confirmed or rejected by hand);
   - NOT_FOUND (dropped).
3. **Merges the two passes** of each model, then **merges the two models**. Items from the same sentence and page where one quote contains the other become one *phrase*.
4. **Labels each phrase.** It is first-person when every label given to it is participant, patient or historical case. Agreement between the models is computed per label (Cohen's kappa).

**Result:** 1,718 verified phrases.

**Files:**
- `stageA_verified.csv`: the phrase pool, one row per phrase;
- `stageV_items.csv`: every item with its verification status;
- `stageV_near_matches.csv`, `stageV_review.csv`: hand decisions;
- `stageV_problems.csv`, `stageA_run_log.csv`, `stageV_report.txt`.

### 2.3 Stage B: grouping into axes, and selection for CLIP

**Scripts:** `v4_stageB_prepare.py`, then the LLM chats, then `v4_stageB_compare.py`. **Output:** `axes/v4/stageB/`, with the model outputs in `axes/v4/stageB/outputs/`.

What happens:
1. **Prepare** (`v4_stageB_prepare.py`):
   - A phrase is kept if either model labelled it `altered`. Phrases labelled only `ordinary` are set aside for Stage D (`stageB_ordinary.csv`).
   - Phrases whose only sense labels are hearing, touch or another non-visual sense are excluded.
   - Identical texts are merged into one *text*: 1,140 texts.
   - The texts are shuffled once (fixed seed) and the prompt files are written.
2. **Group, in one chat per model.** B1: the model proposes groups, each naming one quality, with a name and a one-sentence definition. B2: it assigns every text to one of its groups, to OTHER, or to NOT_WORLD (descriptions of the self, body, emotions…). The models saw only the numbered texts, with no sources or counts.
3. **Compare** (`v4_stageB_compare.py`):
   - checks that every text was assigned once by each model;
   - measures agreement between the groupings (adjusted Rand index 0.70 on texts both models grouped);
   - builds a cross-walk between the two models' groups;
   - assigns a text automatically when both models' groups lead to the same axis;
   - writes a review file for the rest, which were decided by hand.
4. **Keeping rules,** fixed before the run. An axis is kept only if:
   - it has at least 5 distinct texts from at least 2 documents;
   - it has at least one first-person phrase;
   - it has one meaning (otherwise it is split);
   - the quality can plausibly show in a scene.

   Result: 32 axes, of which 18 are kept.
5. **Selection for CLIP,** with four criteria written before any CLIP output was seen:
   - (1) a property of the image, not of the viewer;
   - (2) possible in any scene;
   - (3) describable in caption words;
   - (4) not essentially one of the control axes (quality, mood, threat).

   Result: **10 axes selected** (`stageB_clip_selection.csv`).

**Files:**
- `stageB_texts.csv`;
- `stageB_assignments.csv`: both models' groups for every text;
- `stageB_crosswalk.csv`;
- `stageB_final_assignment.csv`: text → axis;
- `stageB_final_axes.csv`: support and keeping rules;
- `stageB_axes.csv`, `stageB_clip_selection.csv` (written by hand);
- `stageB_compare_report.txt`.

### 2.4 Stage C: captions

**Scripts:** `v4_stageC_prepare.py`, then the LLM chats, then `v4_stageC_check.py`. **Output:** `axes/v4/stageC/`.

What happens:
1. **The task.** Each phrase gets the smallest edit that makes "a scene that looks {caption}" grammatical. The content words must be the source's own. Only function words may be added ("like", "as if", "with", "the"…), plus "seen" in "as if seen through …", and words restored from directly beside the quote. Grammatical forms may change; synonyms and negations are not allowed.
2. **Two models** caption every phrase, in batches. A model may drop a phrase only as `cannot` (no grammatical caption possible) or `outside` (outside the axis's definition for CLIP).
3. **The check script** (`v4_stageC_check.py`) tests every caption for:
   - **words:** every word comes from the quote, its context, or the allowed list;
   - **form:** the caption fits "a scene that looks ___";
   - **deleted content words;**
   - **fluency:** a GPT-2 score;
   - **meta words** ("feeling", "experience", pronouns…).

   A caption is accepted automatically when both models wrote the same caption and it passes every check. The rest go to a review file.
4. **Four conventions** settle most differences between the models:
   - a clause takes "as if";
   - barrier and fog nouns take "as if seen through / in / with …" (so CLIP doesn't score pictures of veils or glass);
   - first-person words are removed;
   - hedges are removed.
5. **Review by hand.** Phrases that describe a feeling, judgement, person or event rather than how a scene looks are dropped (`not_scene`).
6. **Support is recounted** and the keeping rules applied again. A15 (vivid, heightened) was dropped: its support rested on one source describing a drug-induced experience.

**Result:** 274 captioned texts, 243 distinct captions in **9 axes**.

**Files:**
- `stageC_captions.csv`: final, with quote → caption and every edit logged;
- `stageC_candidates.csv`: both models' captions and checks;
- `stageC_review.csv`;
- `stageC_axes.csv`: support after Stage C;
- `stageC_check_report.txt`;
- `v4_captions_by_axis.html`: every caption in context, for reading.

### 2.5 Stage D: the negative of each caption

**Scripts:** `v4_stageD_prepare.py`, then the LLM chats, then `v4_stageD_check.py`. **Output:** `axes/v4/stageD/`.

What happens:
1. **The unit** is one negative per *anchor*, i.e. per distinct caption (243 anchors).
2. **Candidates.** Each model proposes 3 negatives per anchor. The rules:
   - change only the word or words that carry the quality, at most two;
   - keep the same form;
   - describe ordinary perception, not the opposite extreme;
   - no negation ("not", "un-"…);
   - no generic words ("normal", "ordinary"…).
3. **Filters** (`v4_stageD_check.py`) reject candidates that are identical to the positive, contain a negation, change the form, or change more than 2 content words.
4. **Flags** mark candidates for the reviewer:
   - possibly ungrammatical (a CoLA classifier);
   - generic (proposed for many anchors or several axes);
   - CLIP separates the pair poorly;
   - closer in CLIP to other axes than to its own;
   - uses a word that belongs to another axis.
5. **Suggestion.** For each anchor the script suggests the passing candidate with the fewest flags, preferring those proposed by both models and those consistent within the axis. Every suggestion was checked by hand. The reviewer could pick another candidate, an ordinary phrase from the sources, or their own negative, or drop the pair.
6. **Negation test.** Pairs whose positive contains a negation were dropped when CLIP hardly separated them.

**Typical negatives:** unreal → real; dream → waking life; simulation → physical; foggy → clear; dull → vivid; lifeless → alive.

**Result:** 232 pairs (later revised in Stage D2).

**Files:**
- `stageD_anchors.csv`;
- `stageD_candidates.csv`: every candidate with filters and flags;
- `stageD_review.csv`;
- `stageD_pairs.csv`;
- `axes_v4.yaml`: the Stage D version, superseded by Stage D2;
- `stageD_check_report.txt`.

### 2.6 Scene-property check

**Script:** `v4_check17_scene_properties.py`. **Output:** `axes/v4/check17/` (first run) and `axes/v4/check17_D2/` (after Stage D2).

**Why.** A negative can add a scene property by accident. For example, "as if in a bubble → as if in open space" adds openness. The axis would then partly measure "outdoor scene" rather than the experience.

What the script does:
1. **Builds two control directions** in CLIP's text space, the same way as an axis, from sentence pairs that differ in one property only:
   - *enclosure*: enclosed / cramped / like a closed room / like an indoor room vs open / spacious / like a wide open landscape / like the outdoors;
   - *darkness*: dark / dimly lit / like nighttime / shadowy vs bright / brightly lit / like daytime / sunlit.
2. **Computes the cosine** of every axis with each control. This is how much the axis leans towards that property.
3. **For axes with suspect negatives** (open / air / space, daylight / lit, light), recomputes the lean without those pairs.
4. **Applies a fixed rule:** an axis's suspect negatives are revised if removing them lowers its lean by at least a third and by at least 0.05.

**Result:**
- **A05 had to be revised:** its enclosure lean was 0.31, mostly from "open" negatives.
- **Fog, vividless and lifeless lean towards darkness** (0.32, 0.35, 0.41), but the lean is in the qualities themselves ("foggy" vs "clear" already reads as darker). It can't be removed by changing negatives. It was accepted and is reported with every result through a sensitivity version of each axis with both control directions removed (**v4@nocontrol**, built in step 05).

**Files:**
- `check17_axes.csv`: each axis's cosine with each control, with and without suspect pairs;
- `check17_pairs.csv`: every pair's share of the axis and its cosine with each control;
- `check17_controls.csv`;
- `check17_report.txt`.

### 2.7 Stage D2: revision of the pairs, and the final axes

**Script:** `v4_stageD2_revise.py`. **Output:** `axes/v4/stageD2/`.

What the script does:
1. **Proposes an action** for each of the 232 pairs:
   - **drop:** the positive isn't about how a scene looks (touch, time, body, judgement, emotion…), or it contains a negation;
   - **replace:** the negative brings in content of its own (people, places, light), or, in A05, adds an openness word. The replacement is another Stage D candidate that passed the filters, or the reviewer's own negative, which must pass the same filters;
   - **check:** borderline pairs, kept unless decided otherwise;
   - **keep:** everything else.
2. **Every non-trivial row is reviewed by hand** in `stageD2_revision.csv`.
3. **Writes the final axes.** The scene-property check is then run again on them: A05's enclosure lean fell to 0.27, and the remainder, which comes from its positives (veil, glass wall), was accepted.

**Result:** 191 pairs. The final file is **`axes/v4/stageD2/axes_v4.yaml`**:
- `template`: "a scene that looks {}";
- per axis: `id`, `name`, `definition`, and a list of `pairs`. Each pair has `anchor_id`, `positive`, `negative`, and provenance (`phrase_ids`, `documents`, `sense_classes`, `revision`).

**Readable axis names come from this YAML.**

**Other files:** `stageD2_pairs.csv` (final pairs and how each was decided), `stageD2_axes.csv`, `stageD2_report.txt`.

**The nine v4 axes:**

| id | name | definition | pairs |
|---|---|---|---|
| A01 | unreal, fake, artificial | The world or things in it look unreal, fake or artificial. | 33 |
| A02 | dreamlike, in a dream | The world or one's life seems like a dream or like being in a dream. | 25 |
| A03 | movie, picture, film set | The world appears as a film, picture or staged scene that is being viewed rather than lived. | 11 |
| A04 | simulation, video game | The world seems to be a simulation, video game or other digital or virtual reality. | 21 |
| A05 | veil, glass wall, barrier, bubble | Something like a veil, film, pane of glass, invisible wall or enclosing bubble seems to stand between the person and the world. | 44 |
| **A07** | **fog, blurry, hazy** | The world appears foggy, blurred, cloudy or indistinct. | 29 |
| A11 | flat, two-dimensional | Objects, people or the world look flat and lacking depth. | 6 |
| **A13** | **dull, less vivid, colorless** | The world appears dimmed, faded or reduced in color, vividness and intensity. | 10 |
| A14 | lifeless, dead, empty | The world or the things in it appear dead, lifeless, empty or void. | 12 |

---

## 3. Images and image features

### 3.1 Image content and selection

**Scripts:** `01_measure_image_content.py` (reused as is; its outputs are in `data/places_content_analysis/per_category/`) and `02_select_images.py`. **Output of 02:** `data/selection/`.

1. **Step 01** segmented every Places365 image and recorded the fraction of pixels covered by people, animals, vehicles, objects, structure and nature.
2. **Step 02** applies a content filter so the images are scenes rather than pictures of people or objects. An image is kept if:
   - person ≤ 0.10;
   - animal ≤ 0.10;
   - object ≤ 0.30;
   - foreground (person + animal + vehicle + object) ≤ 0.40;
   - its category isn't one of 33 excluded categories (dominated by people or objects, sports fields, borderline shops and markets).
3. **It checks** that every kept file is on disk and readable.
4. **It gives every image a stable `image_id`.** Row *i* of every later array is `image_id` *i*.

**Result:** 1,111,881 images in 332 scene categories.

**Files:**
- `image_index.csv`: `image_id`, `image_key` = "category/filename", `category` (Places365 form with a slash, e.g. `balcony/interior`), `path`, size, aspect ratio, content fractions;
- `selection_summary.csv`;
- `selection_report.txt`.

### 3.2 Image encoding

**Script:** `03_encode_images.py`. **Output:** `data/embeddings/`.

1. **Loads the model:** un2CLIP (`openai_vit_l_14_336.ckpt`) on CLIP ViT-L/14-336, including un2CLIP's own final image projection. Start-up checks verify that the projection is loaded.
2. **Preprocesses** each image exactly as the official CLIP code does: bicubic resize of the short side to 336 on the PIL image, centre crop of 336, CLIP normalisation. CLIP therefore sees the **central square** of each image.
3. **Saves the final-layer embedding** after the projection, L2-normalised (768-d), and the pooled vector before the projection (1024-d). Unreadable images get NaN rows and status −1.
4. **Checks alignment:** at the end it re-encodes random images and compares them with the stored rows.

**Files:** `embeddings_768.npy` (N × 768), `pooled_1024.npy`, `image_ids.npy`, `status.npy`, `run_meta.json`.

### 3.3 Image features

All features are measured on **exactly the region CLIP sees**: the central square, resized to 256 × 256 px. Grey is OpenCV's Rec. 601 grey (0–255).

**`04_compute_features.py`. Output:** `data/features/`.
- Computes 18 candidate features for every image:
  - **brightness:** `lum_mean`, `lum_p50`, `val_mean`;
  - **brightness distribution:** `lum_std`, `lum_p05`, `lum_p95`, `lum_skew`, `lum_entropy`;
  - **amount of colour:** `sat_mean`, `sat_std`, `colorfulness`, `chroma_mean`;
  - **colour direction:** `lab_a_mean`, `lab_b_mean`;
  - **fine detail and structure:** `contrast_local`, `edge_density`, `spectral_slope`;
  - **haze:** `dark_channel_mean`.
- Also measures the same features at 336 px on a stratified subsample of about 100 images per category, used for stability checks.
- **Files:**
  - `features_crop.npy` / `.csv` (N × 18);
  - `feature_definitions.json`;
  - `subsample_ids.npy`: about 100 per category, used by 04b and 09b;
  - `features_crop336_subsample.npy`;
  - `status.npy`;
  - `features_report.txt`.

**`04a_flag_monochrome.py`. Output:** `data/monochrome/`.
- **Why:** about 3% of the corpus are black-and-white photographs. They are a photographic style, not a desaturated scene, and would inflate colour links.
- **What it does:** measures how much colour each image has and how it is spread (e.g. the 95th percentile of per-pixel chroma). It sets thresholds by Otsu's method and flags greyscale and toned (sepia, duotone) photographs.
- **The primary analyses use colour photographs only** (`is_greyscale` = False).
- **Files:** `image_flags.csv` (per image: `is_greyscale`, toning flags, measures), `colour_measures.npy`, `threshold_analysis.csv`, `flags_report.txt`.

**`04c_compute_extra_features.py`. Output:** `data/features_extra/`, the folder 04b, 07, 09a, 09b and 09c read. The script's OUTPUT_DIR setting currently says `data/features_extra_v2`: if 04c is rerun as it stands, it writes there, and the later steps would have to be pointed to that folder.
- Computes 10 additional candidate features on the same region. Two of them enter the final set:
  - `hue_spread` (hue variety): 1 − |Σ(a\*, b\*)| / Σ chroma over chromatic pixels; 0 when all colour points the same way. Undefined for greyscale photographs.
  - `cast_b` (colour of the light): mean CIELAB b\* of the brightest non-clipped pixels; + yellow/warm, − blue/cool.
- Also records `native_side`, the original image's short side.
- **Files:** `extra_crop.npy` / `.csv`, `extra_definitions.json`, `extra_report.txt`, validity tables.

### 3.4 Feature selection: the 12 features

**Script:** `04b_select_features.py` (version 2). **Output:** `data/feature_selection_v3/` (set as the script's OUTPUT_DIR for this run).

The selection uses only the features' relations with each other. **The CLIP axes are never loaded,** so the choice can't be tuned to the results.

1. **Data:** colour images of 04's subsample. Every feature is centred on its category mean.
2. **Correlations:** Spearman |ρ| between every pair of features.
3. **Clustering:** hierarchical clustering on 1 − |ρ| (average linkage), cut where groups share less than |ρ| = 0.7.
4. **One feature per group.** The stimulus features `sat_mean`, `contrast_local` and `lum_p05` are forced in. Otherwise the first feature in a priority order fixed in advance is chosen, provided it is stable between 256 and 336 px.
5. **Redundancy check:** within-category VIF of the chosen set. While a feature exceeds the VIF limit, the lowest-priority non-forced feature whose removal fixes it is dropped.

**The 12 selected features** (primary set: colour images):

| Feature | Readable name | What it measures | From |
|---|---|---|---|
| `contrast_local` | local contrast | SD of the Laplacian (3×3) of grey: fine-detail and edge energy, not RMS contrast | 04 |
| `lum_entropy` | tonal entropy | entropy of the 256-bin grey histogram | 04 |
| `spectral_slope` | spectral slope | slope of the amplitude spectrum; higher = less fine detail relative to coarse structure | 04 |
| `lum_std` | global contrast | SD of grey | 04 |
| `lum_mean` | brightness | mean grey | 04 |
| `lum_p05` | shadow level | 5th percentile of grey | 04 |
| `sat_mean` | saturation | mean HSV saturation (0–255) | 04 |
| `sat_std` | saturation spread | SD of HSV saturation | 04 |
| `lab_b_mean` | yellowness | mean CIELAB b\* | 04 |
| `lab_a_mean` | redness | mean CIELAB a\* | 04 |
| `hue_spread` | hue variety | see 3.3 | 04c |
| `cast_b` | colour of the light | see 3.3 | 04c |

**Files:** `selected_features.json` (the selection and which step's file each feature comes from), `selection_report.txt`, group, VIF and correlation tables.

---

## 4. Axis vectors and image scores

### 4.1 Building the axes and testing their stability

**Script:** `05_build_axes_v4.py`. **Output:** `data/axes_v4/`.

What the script does:
1. **Encodes every caption** in the template "a scene that looks {caption}" with the official OpenAI CLIP ViT-L/14-336 text encoder (pooled output, text projection, L2 normalisation).
2. **Checks the encoder:** tokens, agreement with the official CLIP (minimum cosine 0.9999997), and that the template is filled correctly.
3. **Builds each v4 axis:** axis = normalise(mean of the positive sentences − mean of the negative sentences). Each pair counts once, so this equals the mean of the paired differences. A negative shared by several pairs counts once per pair.
4. **Builds variants and diagnostics:**
   - the same axis in three other templates ("a photo that looks", "a place that looks", "the world looks"): `v4@photo`, `@place`, `@world`;
   - baselines `v4@centroid` (positives against the mean of all captions) and `v4@neutral` (against "a scene");
   - **`v4@nocontrol`**: the axis with the darkness and enclosure directions of 2.6 projected out. A sensitivity version, never primary;
   - anchor diagnostics: is each positive on the positive side of its axis;
   - axis–axis cosines;
   - each axis's cosine with the two control directions.
5. **Builds six control axes** from single caption pairs:
   - quality: "a bad photo" vs "a good photo";
   - age: "an old photo" vs "a new photo";
   - weather: "a scene in foggy weather" vs "a scene in clear weather";
   - mood: "a sad scene" vs "a happy scene";
   - threat: "a scary scene" vs "a peaceful scene";
   - greyscale: "a black and white photo" vs "a colour photo".
6. **Tests stability** on 20,000 random images: split-half, leave-one-out, bootstrap and m-curve. Halves are drawn **by pair**, so a positive stays with its negative.
   - **Split-half** works like this. The axis's pairs are randomly split into two halves 200 times. Each half builds its own axis, every image is scored on both, and the two sets of scores are correlated. The Spearman–Brown formula corrects the correlation to full length.

**Result:**
- **Split-half reliability:** A02 0.97, **A07 0.95**, **A13 0.94**, A05 0.93, A01 0.91, A14 0.89, A11 0.84, A04 0.83, A03 0.71.
- **Leave-one-out:** minimum cosine 0.95–1.00.
- **Overlap:** no two v4 axes have a text cosine ≥ 0.8; the highest is A13–A14 at 0.38.

**Files:**
- `axis_vectors.npy`: one unit-length row per axis, all sets and variants;
- `axis_catalog.csv`: `row`, `key`, `set`, `kind`, `template`, `axis`, `n_pos`, `n_neg`, `included`, `primary`;
- `axis_anchors.csv`, `captions.csv`, `caption_embeddings.npy`;
- `anchor_diagnostics.csv`, `baseline_variants.csv`, `axis_cosines_<set>.csv`;
- `stability_summary.csv`, `stability_loo.csv`, `stability_mcurve.csv`;
- `v4_axes.csv`: pairs, documents, control cosines, nocontrol version;
- `build_report.txt`.

### 4.2 Scoring every image

**Script:** `06_project_axes_v4.py`. **Output:** `data/scores_v4/`.

1. **Raw score:** for every image and every axis, raw = embedding · axis, the cosine, since both are unit length.
2. **z-score:** z = (raw − `mean_raw`) / `sd_raw`, with the mean and SD over all 1,111,881 images. Any new image encoded as in step 03 can be put on the same scale with these two numbers.
3. **Checks:** the axis vectors are unit length and match the catalog, and rows are aligned with `image_id`.

**Files:**
- `scores_raw.npy`, `scores_z.npy`: 1,111,881 × 264, row = `image_id`, column = `col` in the catalog;
- `image_ids.npy`;
- `score_catalog.csv`: `col`, `key`, `set`, `kind`, `mean_raw`, `sd_raw`, the same over colour images (`mean_raw_colour`, `sd_raw_colour`), and distribution percentiles;
- `score_correlations_all.csv`, `score_correlations_colour.csv`;
- `score_report.txt`.

The two experiment axes:

| key | mean_raw | sd_raw | skew / excess kurtosis |
|---|---|---|---|
| `v4::A07 fog, blurry, hazy` | −0.0457 | 0.0322 | +1.00 / +2.26 (a minority of hazy scenes score very high) |
| `v4::A13 dull, less vivid, colorless` | +0.0251 | 0.0481 | −0.12 / −0.14 |

---

## 5. How the axis–feature correlations were computed

**Script:** `07_category_confounds_v4.py`. **Output:** `data/confounds_v4_12f/`.

**Data:**
- **Images:** colour photographs with all 12 features defined (1,071,373 images).
- **Categories:** the 332 categories, each with at least 30 images.

**The within-category model:**
1. **Centre both variables.** For each category, subtract the category mean from the axis z-score **and** from every feature. What remains is how an image differs from typical images of its own kind of scene. Centring both variables is essential: centring only the axis would mix in the differences between categories.
2. **within ρ:** Spearman correlation between the centred axis score and one centred feature. One feature at a time; a simple association.
3. **within β (the main statistic):** standardised coefficient of a feature in a regression of the centred axis score on **all 12** centred features together. It answers this: within a category, when this feature is 1 SD higher and the other 11 are held fixed, how many SD does the axis score change?
4. **between β:** the same regression on the 332 category means (one category = one observation). Together with the within model, this separates "kinds of place that are grey score higher" from "within a kind of place, greyer images score higher".
5. **Consistency:** the within statistics computed inside each category separately. "Same sign" is the share of categories whose estimate has the same sign as the pooled one.
6. **Uncertainty:** 95% intervals from 1,000 resamples of whole categories. Category-clustered standard errors are a cross-check.
7. **A link:** |β| ≥ 0.05 with an interval excluding 0.
8. **Sensitivity (R2b):** the same model for `v4@nocontrol`, reported next to every result.
9. **Control share:** the share of each axis's within-category variance explained by the quality, mood and threat controls together.

**Files:**
- `within_between.csv`: one row per image set × axis × feature, with:
  - `within_rho` (and interval);
  - `within_beta` (and interval, `within_beta_se_cluster`);
  - `between_rho`, `between_beta` (and intervals);
  - `consistency_r_same_sign`, `consistency_beta_same_sign`;
  - `vif_within`, `n_images`, `n_categories`;
- `eta_squared.csv`: share of each axis's variance that lies between categories;
- `control_share.csv`, `consistency.csv`;
- `category_means_axes_<set>.csv`, `category_means_features_<set>.csv`;
- `mixed_subsample.csv`: a mixed-model cross-check;
- figures;
- `confound_report.txt`.

---

## 6. Validation tests

### 6.1 Axis checks

**Script:** `09a_axis_checks_v4.py`. **Output:** `data/checks_v4/`.

Every check refits the within-category model of section 5 (colour images, the 12 features, 1,000 category resamples, the same resamples for every check), with something added or changed:
- **C1 controls:** the six control axes added as covariates, one at a time and all together.
- **C3 templates:** the axis built in the "photo", "place" and "world" templates.
- **C4 baselines:** the positive captions against the caption centroid, which shows which pole a link comes from.
- **C6 greyscale:** colour images vs all images.
- **C9 256 vs 336 px:** features measured at CLIP's own resolution.
- **C11 overlap:** within-category correlations between the axes' image scores.
- **C12 duplicates:** results with identical images removed.
- **C13 specificity:** each axis with the other eight v4 axes as covariates. A link that survives belongs to that axis, not to what the axes share.
- **C14 overlap adjusted:** the overlap with features and controls removed, and the principal components of the nine axes.
- **C16 poles:** the overlap computed separately for the positive and the negative poles.
- **C17 pair overlap:** for pairs of axes correlated at least 0.5, each link with the other axis held constant.

**Results for the experiment axes:**

| | Base β | + all controls and the other 8 axes | Comment |
|---|---|---|---|
| Vividless × `sat_mean` | −0.40 | **−0.19** (survives) | 7 of its 11 links survive |
| Fog × `contrast_local` | −0.27 | **−0.15** (survives) | its only link that survives |
| Fog × `sat_mean` | −0.30 | **−0.01 with all controls** (gone) | absorbed by the "foggy weather" control |

**Fog and vividless are not a pair to merge.** The only pair at r ≥ 0.5 is vividless–lifeless (A13–A14, r 0.67; text cosine 0.38, below the merge threshold of 0.8). Holding lifeless constant, vividless keeps its saturation link (−0.40 → −0.27) and loses its local-contrast link (−0.19 → −0.04). The colour link belongs to vividless.

**Files:**
- `checks_results.csv`: every check × configuration × axis × feature, with estimate and interval;
- `axis_summary.csv`;
- `overlap_within_<set>.csv`, `overlap_adjusted.csv`, `axis_components.csv`;
- `poles_*.csv`;
- `duplicate_groups.csv`, `toned_images.csv`;
- `checks_report.txt`.

### 6.2 Item analysis

**Script:** `09b_item_analysis_v4.py`. **Output:** `data/items_v4/`.

**Data:** 04's stratified subsample (32,061 colour images, 332 categories), all statistics within category.

What the script does:
1. **Treats every caption pair as a questionnaire item:** a pair item = positive sentence − negative sentence, scored on every image.
2. **Reliability:**
   - pair alpha (Cronbach's α of an axis's pair items);
   - item-rest r (each pair against the axis rebuilt without it);
   - alpha if a pair is deleted.
3. **Leave-one-pair-out:** how much each axis's key feature links change when one pair is left out.
4. **Traces each link to its poles and captions:**
   - the positive pole = positives vs the caption centroid;
   - the negative pole = centroid vs negatives;
   - each caption's own β on the 12 features.
5. **Cross-loadings:** each caption's correlation with every other axis.
6. **Image-anchored axes:** the mean embedding of the top images minus that of the bottom images of each axis, compared with the text axis.

**Results for the experiment axes:**
- **Reliability:** pair alpha **A13 0.94, A07 0.93** (range over the nine axes 0.65–0.96).
- **Stability:** no single pair moves a key link by more than about 0.03.
- **Weak pairs:** 26 of 191 pairs agree weakly with their axis (item-rest r < 0.2), mostly in A01 and A05. Fog has 2 and vividless 0.
- **Where the links come from:**
  - fog × local contrast comes **entirely from its positive captions** (positive pole −0.29, negative pole −0.02): describing fog, blur and haze is what selects low-detail images;
  - vividless × saturation comes mostly from its negatives ("vivid": negative pole −0.34, positive pole −0.15).

**Files:**
- `pair_table.csv`: one row per pair, with item-rest r, alpha if deleted and β on every feature;
- `caption_table.csv`;
- `reliability.csv`, `cross_loadings.csv`, `loo_influence.csv`;
- `pole_profiles.csv`, `item_profiles.csv`, `image_anchored.csv`;
- `item_report.txt`.

### 6.3 Manipulation sweep: a causal test

**Scripts:**
- `09c_manipulation_sweep.py` makes and encodes the variants. **Output:** `data/sweep_v2/`. The "_v2" is the sweep's own version (version 2 rebuilt six manipulations so they can't clip pixel values), not the v2 axes.
- `09c_rescore_sweep_v4.py` scores the stored embeddings on the v4 axes. **Output:** `data/sweep_v4/`.

**Why:** the correlations of section 5 are observational. The sweep changes one image property in real photographs and measures whether the axis score moves.

What the scripts do:
1. **Choose 150 colour photographs,** one per randomly drawn category. Each has original short side ≥ 512 px, all selected features defined, and no duplicate. Each is cut to CLIP's central square at 512 px.
2. **Apply 17 manipulations,** each changing one property. Three matter here:
   - **chroma:** CIELAB chroma × k at constant lightness and hue;
   - **haze:** I′ = (1 − h)·I + h·A, where A is the image's own airlight, the mean RGB of its brightest 1% of pixels;
   - **haze_eq:** haze followed by a linear-light gain that restores the original mean grey, so brightness is held constant.

   Others include blur, smoothing, sharpening, contrast, exposure and grain.
3. **Calibrate the levels on the measured feature, not on the parameter.** Each level moves the median image's feature by the gap between the corpus median and a corpus percentile (p75 / p90 / p99, or p25 / p10 / p01 for decreases). The levels therefore stay within the range of real photographs. Levels that would clip pixel values are capped.
4. **Measure every variant's features** (same code as steps 04/04c) and **encode it as in step 03.**
5. **Compute the rescored effect:** d = the mean change in axis score from the same image's unmanipulated version, in corpus SD units, with 95% intervals from resampling images. Also:
   - **unique:** the change beyond what the other eight axes predict;
   - **share:** the share of the total change across the nine axes that falls on this axis;
   - the change beyond the quality, mood and threat controls;
   - the effect on `v4@nocontrol`.
6. **Run start-up checks:** the stored scores and effects are reproduced from the stored embeddings.

**Files:**
- `data/sweep_v2/`: `sweep_images.csv`, `levels.csv`, `variants.csv` (each image × level: parameter, clipped share, all features), `embeddings.npy`, `side_effects.csv`. Its `effects.csv` and `sweep_report.txt` score the variants on the **earlier axes** (v2, v3b), not on v4: do not use them for v4;
- `data/sweep_v4/`: `effects.csv` (level × axis: d, interval, same sign, unique, share, beyond controls), `variant_scores.npy`, `score_keys.csv`, `sweep_report.txt`.

**Results at the top of the natural range** (d in corpus SD):

| Manipulation | Fog (A07) | Vividless (A13) | "Bad photo" control |
|---|---|---|---|
| Chroma down (less saturation) | +0.55 | **+1.59** | +0.48 |
| Chroma up | −0.25 | **−1.68** | +0.64 |
| Haze | **+1.44** | +0.75 | +0.80 |
| Haze at equal brightness (haze_eq) | **+1.54** | +1.29 | +1.30 |
| Smoothing (removes fine detail) | **+1.48** | – | – |
| Blur (lowers local contrast) | **+0.79** | +0.32 | +1.37 |
| Sharpening (raises local contrast) | **−0.41** | – | – |

- **Vividless responds to colour, in both directions, and specifically.** Chroma up puts 71% of the total change across the nine axes on vividless, the most specific effect in the sweep.
- **Fog responds to haze and to loss of detail, in both directions.** Its change beyond the other axes is +0.57 to +1.01.
- **The darkness control direction also reads colour** (chroma down moves it +1.23). So `v4@nocontrol` removes some real colour signal and understates colour links. This is stated wherever R2b is reported.

---

## 7. Main findings: coefficient tables

**All nine v4 axes × the 12 features: within-category joint β** (colour images; 1,071,373 images; 332 categories; from `within_between.csv`). **Bold** = a link (|β| ≥ 0.05, 95% interval excluding 0). The two experiment pairs are fog × `contrast_local` and vividless × `sat_mean`.

| Axis | `sat_mean` | `contrast_local` | `sat_std` | `hue_spread` | `cast_b` | `lab_b_mean` | `lab_a_mean` | `lum_mean` | `lum_p05` | `lum_std` | `lum_entropy` | `spectral_slope` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A01 unreal, fake, artificial | **+0.09** | **+0.05** | +0.01 | −0.03 | **+0.08** | **−0.16** | +0.03 | **−0.19** | **+0.08** | **+0.09** | **+0.05** | +0.02 |
| A02 dreamlike, in a dream | **+0.19** | +0.02 | +0.04 | +0.01 | −0.01 | **−0.15** | +0.01 | +0.01 | **+0.09** | **+0.08** | **+0.08** | −0.03 |
| A03 movie, picture, film set | −0.04 | +0.04 | +0.01 | −0.02 | +0.01 | +0.01 | +0.02 | −0.03 | **−0.05** | +0.00 | +0.04 | **+0.13** |
| A04 simulation, video game | **+0.06** | +0.02 | −0.01 | +0.03 | −0.02 | −0.02 | **−0.09** | **−0.18** | **+0.12** | **+0.07** | +0.02 | +0.00 |
| A05 veil, glass wall, barrier, bubble | **−0.09** | −0.02 | +0.01 | **−0.06** | +0.05 | **−0.07** | +0.04 | **−0.14** | **+0.05** | **+0.06** | +0.03 | +0.03 |
| **A07 fog, blurry, hazy** | **−0.30** | **−0.27** | **−0.06** | **−0.17** | −0.02 | **+0.10** | **+0.07** | **−0.10** | +0.02 | **+0.05** | **−0.06** | −0.01 |
| A11 flat, two-dimensional | +0.01 | **−0.20** | −0.03 | **+0.06** | **−0.08** | −0.05 | **−0.05** | **+0.21** | −0.02 | +0.00 | **−0.10** | **−0.12** |
| **A13 dull, less vivid, colorless** | **−0.40** | **−0.19** | **−0.23** | **−0.16** | **−0.15** | **+0.13** | **−0.09** | **−0.09** | −0.03 | **+0.06** | **−0.10** | **−0.05** |
| A14 lifeless, dead, empty | **−0.24** | **−0.28** | **−0.13** | **−0.12** | −0.05 | −0.04 | +0.03 | **−0.06** | **−0.08** | **+0.07** | **−0.20** | −0.04 |

**The same model with the darkness and enclosure directions removed (`v4@nocontrol`, sensitivity only):**

| Axis | `sat_mean` | `contrast_local` | `sat_std` | `hue_spread` | `cast_b` | `lab_b_mean` | `lab_a_mean` | `lum_mean` | `lum_p05` | `lum_std` | `lum_entropy` | `spectral_slope` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A01 unreal, fake, artificial | **+0.14** | **+0.06** | +0.03 | +0.00 | **+0.08** | **−0.18** | +0.01 | **−0.13** | **+0.09** | **+0.09** | **+0.06** | +0.00 |
| A02 dreamlike, in a dream | **+0.14** | +0.01 | +0.02 | −0.01 | −0.01 | **−0.13** | +0.02 | −0.04 | **+0.08** | **+0.09** | **+0.06** | −0.03 |
| A03 movie, picture, film set | −0.02 | **+0.05** | +0.02 | −0.01 | +0.01 | +0.01 | +0.02 | −0.01 | −0.05 | +0.00 | +0.04 | **+0.12** |
| A04 simulation, video game | **+0.10** | +0.02 | +0.01 | **+0.06** | −0.02 | −0.04 | **−0.10** | **−0.13** | **+0.12** | **+0.07** | +0.03 | −0.01 |
| A05 veil, glass wall, barrier, bubble | +0.05 | +0.00 | **+0.06** | +0.02 | **+0.06** | **−0.14** | +0.00 | +0.00 | **+0.08** | **+0.06** | **+0.07** | +0.00 |
| **A07 fog, blurry, hazy** | **−0.13** | **−0.24** | +0.03 | **−0.08** | +0.01 | **+0.06** | **+0.05** | **+0.10** | **+0.10** | +0.03 | +0.03 | −0.01 |
| A11 flat, two-dimensional | −0.02 | **−0.20** | −0.03 | +0.04 | **−0.08** | −0.02 | −0.04 | **+0.18** | −0.01 | −0.01 | **−0.08** | **−0.10** |
| **A13 dull, less vivid, colorless** | **−0.30** | **−0.15** | **−0.19** | **−0.10** | **−0.15** | **+0.11** | **−0.12** | +0.04 | +0.03 | +0.04 | −0.02 | **−0.05** |
| A14 lifeless, dead, empty | **−0.06** | **−0.26** | **−0.05** | −0.03 | −0.02 | **−0.10** | +0.02 | **+0.15** | −0.01 | +0.05 | **−0.12** | −0.04 |

**Detail for the two experiment axes** (colour images; 95% intervals in brackets):

| Axis × feature | within ρ | within β | between β | same sign (β), share of categories | β without darkness/enclosure |
|---|---|---|---|---|---|
| **Fog × `contrast_local`** | −0.29 [−0.30, −0.28] | **−0.27** [−0.28, −0.26] | −0.72 [−0.95, −0.51] | 100% | −0.24 |
| Fog × `sat_mean` | −0.26 [−0.28, −0.25] | −0.30 [−0.31, −0.28] | −0.18 [−0.40, −0.00] | 93% | −0.13 |
| Fog × `sat_std` | −0.18 [−0.20, −0.17] | −0.06 [−0.07, −0.04] | +0.12 [−0.04, +0.29] | 74% | +0.03 |
| **Vividless × `sat_mean`** | −0.50 [−0.51, −0.49] | **−0.40** [−0.42, −0.39] | −0.60 [−0.79, −0.42] | 100% | −0.30 |
| Vividless × `contrast_local` | −0.23 [−0.24, −0.22] | −0.19 [−0.20, −0.18] | −0.08 [−0.28, +0.15] | 99% | −0.15 |
| Vividless × `sat_std` | −0.42 [−0.43, −0.41] | −0.23 [−0.24, −0.22] | −0.34 [−0.49, −0.17] | 99% | −0.19 |

**Reading the tables:**
- Fog and vividless both lean towards less colour and less fine detail, but in different proportions.
- Vividless is above all a colour axis: amount, spread and variety of colour.
- Fog is the axis whose local-contrast link is the most consistent (every category) and the most robust (it survives the controls, the other axes and the removal of darkness/enclosure).

---

## 8. Why fog is paired with `contrast_local` and vividless with `sat_mean`

Both axes correlate with both features, so the choice needs a justification. Three arguments support it, each with caveats.

### Argument 1: each axis gets the feature that separates it from the other axis

The experiment needs two features that **tell fog and vividless apart**, not just two features that correlate with each.

| Within-category β | `sat_mean` | `contrast_local` |
|---|---|---|
| Vividless (A13) | **−0.40** | −0.19 |
| Fog (A07) | −0.30 | **−0.27** |

- Saturation is more strongly tied to vividless than to fog (−0.40 vs −0.30).
- Local contrast is more strongly tied to fog than to vividless (−0.27 vs −0.19).
- Using saturation for both would base the two dimensions on one feature, and they could not be separated. This pairing is the only one in which each axis gets the feature it owns more.

**The other tests agree:**
- 09a: with all controls, fog's saturation link vanishes (−0.30 → −0.01), while its local-contrast link survives the controls and the other axes (−0.15).
- 09a: with lifeless held constant, vividless keeps saturation and loses local contrast.
- 09b: fog's local-contrast link comes entirely from its positive captions (fog, blur, haze).

**Caveat:** in the within-category model alone, fog's saturation β is slightly larger than its local-contrast β (−0.30 vs −0.27, intervals not overlapping). The argument is about which feature **separates** the axes, not which is fog's largest link. Both features are measured for every image or stimulus and can enter an analysis together, so fog's saturation link can be estimated and reported rather than ignored.

### Argument 2: in fog, low saturation is a side effect

In real fog, light scattered by the air mixes into everything seen. That has two effects:
1. **It lowers contrast and detail.** This is what defines fog: things fade with distance.
2. **It washes out colour,** as a by-product.

The sweep shows the same order. Haze lowers local contrast in proportion to its strength (for plain haze, by the factor 1 − h up to rounding, since a blend with a constant colour scales the Laplacian). It lowers saturation as a side effect.

**Saturation spread fits this.** Vividless is linked to both the amount and the spread of colour (`sat_mean` −0.40, `sat_std` −0.23), as expected for an axis about colour itself. Fog's link with saturation spread is weak (β −0.06, and +0.03 without darkness/enclosure). That fits a general fading of colour rather than colour being fog's defining property. The axis's own words ("fog, blurry, hazy") describe loss of clarity, not loss of colour.

**Caveats:**
- This is an interpretation consistent with the data, not a test.
- With **haze_eq** (haze with brightness restored), local contrast falls a little more than with plain haze, because the restoring gain is a dimming. The statement is then "in proportion to haze strength", not "exactly 1 − h".

### Argument 3: changing each feature moves its own axis (manipulation sweep, 6.3)

| Manipulation | Fog | Vividless |
|---|---|---|
| Chroma down (lowers saturation) | +0.55 | **+1.59** |
| Haze at equal brightness (haze_eq) | **+1.54** | +1.29 |
| Plain haze | **+1.44** | +0.75 |
| Blur / smoothing (lower local contrast) | **+0.79 / +1.48** | +0.32 / – |
| Sharpening (raises local contrast) | **−0.41** | – |

- **Lowering saturation mainly moves vividless.** 62% of the total change across the nine axes falls on it.
- **Removing fine detail moves fog, in both directions:** removing detail raises it, adding detail lowers it.
- **Each manipulation moves its own axis most,** which is the pattern Argument 1 predicts.

**Caveats:**
- **haze_eq is only weakly specific.** It raises vividless almost as much as fog (+1.29 vs +1.54), because it also lowers saturation (Argument 2's side effect) and the re-darkening reads as duller to CLIP. Plain haze separates the two better (+1.44 vs +0.75). Report this rather than claim a clean separation.
- **Detail manipulations are not pure.** Blur also makes images look like a worse photo (quality control +1.37).
- **The sweep tested each manipulation alone,** on 150 photographs, at its own calibrated levels (haze strengths up to h = 0.50, chroma down to k = 0.198). Results apply to those manipulations and levels. Combinations of manipulations were never tested. When images are manipulated differently, or combined, repeat the same test on those images: the mean change from the unmanipulated version on fog, vividless and the "bad photo" control, with intervals from resampling images.

### Suggested wording

> We quantified the fog dimension with local contrast and the vividless dimension with saturation. Both features correlate with both axes, but each is more strongly linked to one: saturation to vividless (within-category β = −0.40 vs −0.30 for fog), local contrast to fog (−0.27 vs −0.19 for vividless). Using saturation for both dimensions would have confounded them. This fits the physics of fog, in which scattered light reduces contrast and detail, with desaturation as a by-product; consistently, fog is only weakly related to saturation spread (β = −0.06), whereas vividless is related to both the amount and the spread of colour (−0.40, −0.23). A manipulation sweep on 150 photographs supported the pairing: lowering chroma mainly raised vividless (+1.59 SD vs +0.55 for fog), whereas haze and loss of fine detail mainly raised fog (haze at equal brightness +1.54 SD, smoothing +1.48, sharpening −0.41), although brightness-matched haze also raised vividless (+1.29), partly through its side effect on saturation.

---

## 9. Earlier axis sets (v2, v3) in the v4 files

Several scripts keep the earlier sets in their files, or compare with the earlier run, to check that nothing changed for them. For v4 work, filter or ignore as below.

| Script | What it does with v2/v3 | What to do for v4 |
|---|---|---|
| `05_build_axes_v4.py` | Builds every set in one file: v2, v2_merged, v3a, v3b and v3b_neg with their variants, the controls and v4 (264 vectors). Also writes v4 × v3b cosine tables. | In `axis_vectors.npy` / `axis_catalog.csv`, select v4 by `key` or `set`. Ignore `v4_v3b_cosines.csv` and `v4_v3b_score_r.csv`. |
| `06_project_axes_v4.py` | Scores all 264 axes. Check S5 confirms the v2/v3 scores equal the earlier run's (`data/scores`). | In `scores_raw.npy` / `scores_z.npy`, use the `col` of the v4 key from `score_catalog.csv`. |
| `07_category_confounds_v4.py` | Analyses v2, v2_merged, v3a, v3b and v3b_neg next to v4. Check R1 confirms they equal the earlier run (`data/confounds_v3`). | In `within_between.csv` and the other tables, filter `set == "v4"` (or `"v4@nocontrol"`). |
| `09a_axis_checks_v4.py` | Reads the v3b results for comparison, checks the v3b estimates equal `data/checks_v3` (R1), and runs a v3–v4 comparison (C18). | Ignore `replication.csv`, `v4_v3b_within_r.csv`, `v4_v3b_within_r_ci.csv` and the comparison sections of `checks_report.txt`. Everything else is v4. |
| `09b_item_analysis_v4.py` | Runs the same item analysis on the v3b axes as a comparison set. | Filter the tables on the v4 axes. |
| `09c_manipulation_sweep.py` | Scored the variants on the earlier run's axes (`data/axes`: v2, v3b and the controls). | Use `data/sweep_v2/` only for the images, levels, variants and embeddings. Its `effects.csv` and `sweep_report.txt` are for the earlier axes. |
| `09c_rescore_sweep_v4.py` | Its start-up checks recompute the stored v2/v3b scores to confirm the embeddings are intact. | Nothing: its outputs (`data/sweep_v4/`) are v4 only. |

Stages A–D2 and steps 01–04c have nothing to do with the earlier axis sets.

**Folder names that look like axis versions but aren't:**

| Folder | What the version in its name means |
|---|---|
| `data/feature_selection/` | feature selection **version 1** (10 features); used only by 04b's check that version 1 is reproduced |
| `data/feature_selection_v3/` | the feature selection used everywhere in v4 (04b version 2: the 12 features); the "_v3" is the run, not the v3 axes |
| `data/features_extra/` | 04c version 2 (see 3.3 for its OUTPUT_DIR setting) |
| `data/sweep_v2/` | sweep version 2; the variants used by the v4 rescoring |
| `data/confounds_v4_12f/` | step 07 for v4, run with the 12 features |

**Folders of the earlier run** (`data/axes`, `data/scores`, `data/confounds_v3`, `data/checks_v3`, `data/items`) hold v2 and v3 only and are not part of the v4 pipeline. The one exception is `data/checks_v3/duplicate_groups.csv`, which is identical to `data/checks_v4/duplicate_groups.csv` (same code and inputs).

---

## 10. What this project folder contains, and what is not included

This folder holds the scripts listed in the guide table, `axes/v4/` (the outputs of Stages A–D2) and the data folders of every v4 step. **It is complete for reading and using the v4 results**: every number in this document can be traced to a file in it.

**Check that `axes/v4/` contains its subfolders:** `stageA`, `stageV`, `stageB` (with `outputs/`), `stageC`, `stageD`, `check17`, `check17_D2` and `stageD2`. Above all it needs `stageD2/axes_v4.yaml`, which holds the readable axis names, definitions and pairs.

**`data/embeddings/`** (`embeddings_768.npy`, `status.npy`, `image_ids.npy`; about 3.4 GB) is read by steps 06, 07, 09a, 09b and both 09c scripts. If it is not in the folder, those scripts can't run. It is also needed to check a new encoding setup against the corpus (re-encode a few corpus images and compare them with the stored embeddings; the cosine should be about 1). To put a new image on the v4 scale, only `data/scores_v4/score_catalog.csv` is needed: z = (embedding · axis − `mean_raw`) / `sd_raw`, with the axis vector from `data/axes_v4/axis_vectors.npy` (row = `col`).

**Not included on purpose. Needed only to rerun scripts:**

| Item | Used by |
|---|---|
| `dpd_selected_scientific_literature/` (the 9 PDFs) | `v4_stageV_verify.py` |
| `categories_places365.csv` | `02_select_images.py` |
| The Places365 images (paths in `image_index.csv`) | 01, 02, 03, 04, 04a, 04c, 09c |
| `pretrained_models/openai_vit_l_14_336.ckpt` (un2CLIP weights) | 03 and 09c, and any encoding of new images |
| `axes/axes_consolidated_v2.yaml`, `axes/v3/axes_v3.yaml`, `axes/v3/anchor_tags_final.csv` (the v2/v3 axis definitions) | `05_build_axes_v4.py`, which builds the earlier sets together with v4 (check its SETTINGS) |
| `data/scores`, `data/confounds_v3`, `data/checks_v3` (the earlier run) | The equality checks of 06 (S5), 07 (R1) and 09a (R1). Each can be switched off by setting it to `None` in the script's SETTINGS. |
| `data/axes`, `data/scores`, `data/checks_v3/duplicate_groups.csv` | `09c_manipulation_sweep.py`. It doesn't need rerunning: its outputs are in `data/sweep_v2/`. If it is rerun, point `DUP_CSV` to `data/checks_v4/duplicate_groups.csv`. |
| `data/encoded_all_categories_un2clip_v2_full/`, `results/methodological_analysis_un2clip_v2_full/` (the old pipeline) | Optional comparisons in 02, 04, 05 and 07; can be switched off |
