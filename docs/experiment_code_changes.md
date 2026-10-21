# Experiment code: the change list

**Status:** 20 Oct 2026, sent to the session making the changes (in progress). It applies to the experiment repository
(`dpdr_experiment`) as of its latest commit ("Add grid block (contrast_local
only)").

**What this file is.** It lists every change the online experiment needs so that
it runs the design as registered (`docs/experimental_design.md` in
`drl2drv_updated`).
- **Part 1** explains each change and why it was decided.
- **Part 2** is a prompt for another Claude session that will make the changes.

**Ready to send (20 Oct).** `text-polish` is merged into `main`, the site
deploys from `main`, and the stimuli on `main` are verified (step 0).

**Decided 20 Oct:**
- the control definition text (item 6);
- the group in the link, `&group=control` (item 4);
- the interview date in the link (item 11);
- a consent screen with placeholder text (item 12);
- every new block can be switched off (item 18);
- participants whose screen cannot show the images are stopped at the start
  (item 13).

---

## Part 1. The changes, and why

### 0. Before any code changes: the v4 stimuli and the starting point

**The stimuli are in place (checked 20 Oct).** Both packages are committed to the
experiment repository, on the `text-polish` branch:
- **`data/sets/v4/`, the pair trials:**
  - all 354 files match the frozen checksums of `frozen_trial_list_v4`;
  - 211 trials in the registered counts, 180 analysed;
  - 349 images, 334 of them preloaded (the catch-only images are not);
  - no extra files.
- **`data/sets/v4/grid/`, the grid:** identical to the grid package, without
  `previews/`. It has 144 images, all present and preloaded.

The archived copies in `drl2drv_updated/data_trials/` stay as the record. The
other session only verifies the copies; it does not copy anything.

**The starting point.** `text-polish` contains the fixes from 21 Sep (local CSV
save, queued uploads, the grid in the upload test, text polish) plus the v4
stimuli. `main` has not moved since the branch was made, so the merge has no
conflicts. Merge `text-polish` into `main` and switch GitHub Pages to `main`
before sending the prompt; the other session then works from `main`.

**Why.**
- **The stimuli.** The experiment still runs the pre-v4 stimuli (sets
  `lum_p05` and `contrast_local`). The v4 packages are the frozen, checksummed
  stimuli the registration describes. Copying them unchanged, with no
  conversion, keeps them identical to the frozen files. That is also why the
  grid decision was to adapt the experiment to the package, not the package to
  the experiment.
- **The checksums.** Checking the copies against them before any code change
  rules out a copying error.

### 1. The new stimulus set and trial list

**What.**
- Make `v4` the default set, and take the old sets out of `available_sets`.
  Their folders stay in the repository.
- Update the loader and the validator to the v4 trial-list format:
  - 211 trials: 6 practice, 60 colour-only, 60 contrast-only, 30 together,
    30 conflict, 15 consistency and 10 catch;
  - new columns `pred_sat`, `pred_fog` and `catch_mode`;
  - no `correct_response` column;
  - catch-only images are shipped but not listed in `preload.json`.
- Update the expected counts and the comments that still say 137 / 143 trials
  or 227 images.

**Why.** The registered analysis uses this frozen list, and the experiment
presents it in file order without randomising anything. `correct_response` was
dropped on purpose: the expected catch key lives in the experiment's settings,
and a column the experiment ignored is how an earlier mismatch survived.

### 2. Catch trials: 10, balanced 5 up / 5 down

**What.** Extend the catch key sequence in `config.js` from 6 to 10 targets, 5
up and 5 down, in a fixed mixed order. The validator should check that the
sequence length equals the number of catch trials in the list.

**Why.** The exclusion rule is "3 or more failures out of 10 catch trials". It
was set together with the power analysis, and it needs 10 trials to be
workable. Catch trials show no images (`no_images` mode), as now.

### 3. Session order

**What.** The new order, the same for participants and controls:
1. consent screen (item 12), then the ID screen;
2. full screen and viewport check;
3. brightness, with the viewing-conditions note (item 5);
4. gamma matching;
5. anchoring screen (item 6);
6. practice;
7. pair trials;
8. episode grid;
9. word ratings;
10. normal-appearance grid;
11. state items, optional (item 10);
12. upload and end.

**Why.** The word ratings used to come first, as a predictor. They now come
after the main task and the episode grid, so that the rating words (foggy,
faded…) cannot prime the image choices. The cost is accepted: ratings given
after the task may echo the choices, so they are a descriptive measure. The
normal-appearance grid comes last so it cannot shape the episode answers.

### 4. Participants and controls: one experiment, two groups

**What.**
- Every session belongs to a group, `participant` or `control`, recorded on
  every row.
- The group changes only the wording: the anchoring screen, the trial prompt,
  the episode-grid question and the word-rating question. Everything else is
  identical: trials, order, grids, words and catch trials.

**Why.** Controls are people from the same survey who have never had the
experience. They do the same task, imagining how the world might look to
someone experiencing derealization. The comparison between the groups is one
of the primary tests, so the two sessions must differ in nothing but the
framing.

**How the experiment knows the group (decided).** A URL parameter on the
invitation link, `&group=control` or `&group=participant`, alongside the `pid`
already there. The session refuses to start without a valid group and says the
link is incomplete. This needs no input from the participant and cannot be
answered wrongly.

### 5. Viewing-conditions note

**What.** Add a short paragraph to the brightness screen asking participants to
switch off night mode, blue-light or colour filters if they can. It should also
say not to worry if they do not know whether these are on or how to switch them
off. Nothing is confirmed or recorded.

**Why.** Decided as an instruction only, with no check and no covariate. A
yes/no confirmation would measure little, because many people do not know
whether a filter is on.

### 6. Anchoring screen

**What.** One new screen before the practice.
- **Participants** bring to mind one specific episode, ideally a recent one
  they remember clearly, and answer every question about that episode. An
  optional text box asks for a few words identifying it (for example, where
  they were). The screen **does not mention the interview**.
- **Controls** read a short definition with no perceptual words, and answer
  about how the surroundings might look to someone experiencing it.

Every later prompt then says "during that episode" (participants) or refers back
to the definition (controls).

**Why.**
- Without anchoring, each choice could refer to a different episode, or to an
  average of several.
- Leaving the interview out of the screen keeps the task from depending on the
  interview's content.
- The text box lets the participant's own words identify the episode, without
  asking anything perceptual.
- The control text is preregistered word for word.

**The control definition (decided).** "Derealization is an experience in which
one's surroundings feel strange or detached, as if one were cut off from the
world." The design document's earlier example said "unreal", which is one of the
baseline rating words; the wording rules forbid describing the condition with
words the task measures. The Japanese needs the research assistant's check.

### 7. Trial prompt

**What.** The question above every pair becomes:
- **participants:** "Which looks more like how your surroundings looked during
  that episode?" (now: "…during an episode?");
- **controls:** "Which looks more like how the surroundings might look to
  someone experiencing derealization?"

**Why.** It points every choice at the anchored episode. The wording rules stay:
the condition is named, never described, and the question is about the
surroundings.

### 8. Word ratings: new list, new position, new wording

**What.**
- **The list** (10 words, 0–6 scale, random order, order recorded):

  | Role | EN | JA (draft, to be checked) | id |
  |---|---|---|---|
  | target | foggy | 霞がかった | `foggy` |
  | target | faded | 色あせた | `faded` |
  | near the targets | veiled | ベールがかかったような | `veiled` |
  | near the targets | lifeless | 生気のない | `lifeless` |
  | baseline | unreal | 現実味のない | `unreal` |
  | baseline | dreamlike | 夢のような | `dreamlike` |
  | baseline | distant | 遠く感じる | `distant` |
  | baseline | like a picture | 絵のような | `like_a_picture` |
  | filler (low mood) | oppressive | 重苦しい | `oppressive` |
  | filler (anxiety) | threatening | 不穏な | `threatening` |

- **The promoted target columns** become `rating_foggy` and `rating_faded`.
  `lifeless` is no longer a target.
- **The wording.** Participants: "During that episode, how well does each word
  describe how your surroundings looked?" Controls: the same question about how
  the surroundings might look to someone experiencing derealization. The old
  text says the ratings come "before the main task"; that is no longer true.

**Why.**
- **The colour target.** The colour feature is saturation, and the colour axis
  is "vividless", so the colour target must name colour: "faded". "Lifeless"
  stays as a nearby word, to see whether people choose it instead of the
  target.
- **The baseline words** are derealization words that are not about contrast
  or colour. Each target is read relative to the person's mean of the four, so
  general endorsement of derealization words is not mistaken for a specific
  one.
- **The two fillers** (one low mood, one anxiety) show whether high ratings
  reflect general distress.
- **The old fillers** (crowded, echoing, …) were dropped as inappropriate.

### 9. The grid blocks: two blocks, the new package format, the set swap

**What.** Adapt the loader, validator and grid phase to the v4 grid package.
`interface_spec.md` in the package is the contract. In short:
- **Two blocks of four screens.**
  - Episode block, after the pair trials. "None of these" is offered, followed
    by a free-text box when it is chosen.
  - Normal-appearance block, after the word ratings. There is no "none of
    these" button.
- **The package format changed.** It is one object with blocks, a set
  assignment and two scene sets (A and B), each with screens. Columns carry
  `display_column`, `column_label` and `level_index`, and the feature is
  `fog` / `colour` rather than `haze`. `preload.json` is `{"images": [...]}`.
  The current validator rejects all of this.
- **The set swap.** At session start, draw a code, 0 or 1, with equal
  probability, and record it. Code 0: episode block shows set A, normal block
  shows set B. Code 1: the reverse.
- **Record per screen:**
  - the block, the set code and the set shown;
  - the screen number, the feature and the column order;
  - the three scene ids;
  - the chosen display column **and** the chosen level index;
  - whether "none of these" was chosen, and the free text;
  - the response time, the cell size and the window size.
- **The cell size:** one size for the whole block, computed from the widest
  screen.
- **Questions.**
  - Participants, episode block: which column looks most like how the
    surroundings looked during that episode.
  - Normal block, both groups: how your surroundings normally look.
  - Controls, episode block: how the surroundings might look to someone
    experiencing derealization.

**Why.**
- **Two blocks.** The shift from "normal" to "episode" in each person gives
  them their own baseline. It is a descriptive measure.
- **Two matched scene sets, swapped between people.** Nobody sees the same
  photograph in both blocks, which would anchor the second answer, and neither
  set is always the episode set.
- **Both the column and the level index.** The display position moves when the
  column order is reversed, and the fog answers on the original column are
  censored. Those are exactly the answers with level index 0 on a fog screen.
- **No "none" in the normal block.** "My surroundings normally look like none
  of these" cannot be used to compute the shift.

**The grid's cell size.** Six 512 px images side by side cannot fit on any
laptop, so the grid shrinks its images: one cell size per block, computed from
the window, between 100 and 240 px.
- **Typical sizes in full screen:** about 135 px at the smallest window the
  pair gate lets through (1056 × 680), 172 px at 1280 × 720, 187 px at
  1366 × 768, and 240 px at 1920 × 1080.
- **Why this is accepted.** The grid is descriptive and is not used by any
  registered test. The columns were chosen to stay distinguishable at these
  sizes; the contrast-up column was dropped because it was not.
- **Its limits.** Smaller cells make neighbouring fog columns harder to tell
  apart, and high-density screens show more detail at the same size.
- **What is done about it.** The cell size and the device pixel ratio are
  recorded on every grid row, so they can be reported and used as covariates.
  There is no extra gate.

### 10. State items (optional block)

**What.** A short block at the end with the four derealization items of the
Cambridge Depersonalisation Scale (items 2, 13, 19 and 29) in the CDS state
format ("right now"). It is a switchable block, on or off in `config.js`.

**Why.** The task measures a remembered episode, and the person's state while
doing the task may colour it. A short state measure at the end gives a
covariate. It is optional because it lengthens the session.

**The wording.** Use the exact wording of items 2, 13, 19 and 29 from the
CDS-29 survey your respondents answered: the English and the Japanese version in
your Qualtrics survey. Copy them into the text files, in the placeholders the
other session builds. There are two reasons not to take them from a website.
- **Consistency.** Participants should see the same wording they answered
  months earlier. Online versions differ in translation and even in item
  numbering; one form I checked numbers its items differently from your
  survey.
- **Copyright.** The scale is a published, copyrighted instrument, so I have
  not copied its items into this file.

**The time frame and the response format.** I found no standard published
"state" version of the CDS. The original asks about frequency and duration over
the past six months. Proposed:
- each item is shown with the lead-in "Right now, at this moment:";
- each is answered on the same 0–6 scale as the word ratings, from "not at all"
  (0) to "completely" (6), with radio buttons and no default answer;
- the four items are on one screen, in a fixed order.

Using the same scale as the word ratings means the block can reuse the
word-rating screen's code. State it in the preregistration as an adaptation.
The Japanese lead-in and anchors need the research assistant's check.

### 11. Interview date

**What.** Record the date of the participant's interview on every row.

**Why.** The delay between interview and task is a covariate, because the task
measures a remembered appearance.

**How (decided).** A second URL parameter on the participant's invitation link,
for example `&idate=2026-11-03`. It is empty for controls. The research
assistant schedules and runs the interviews, so she adds the date when she makes
each participant's link. The participant is not asked, since they may not
remember the date exactly. Her own schedule remains the backup record.

### 12. Consent screen (with placeholder text)

**What.**
- **A new first screen**, before the ID screen, with the consent text, an "I
  agree" checkbox and two buttons.
- **If the participant agrees,** the session continues, and `consent_given =
  true` is recorded on every row.
- **If they do not,** a short thank-you screen ends the session. Nothing is
  uploaded.
- **The text is a placeholder.** It is a generic online-study consent text
  with bracketed gaps, adapted from common university templates. It will be
  replaced with the approved text.
- **Where the text lives.** Like every participant-facing string, it goes in
  `text/en/instructions.js` and `text/ja/instructions.js`, in a new `consent`
  section, so replacing it later means editing those two files only.
- **A switch.** The screen is a switchable block, `CONFIG.blocks.consent`.

**Why.** The design lists consent at the start of the session, and the
experiment has none. Building the screen now, with placeholder text, means only
the wording changes once it is approved.

**The placeholder text** (English; the other session drafts the Japanese):

> **Consent to take part**
>
> You are invited to take part in a research study about how people perceive
> their surroundings. It is run by [researcher names] at [institution].
>
> - **What you will do:** look at pairs of photographs and choose between
>   them, then answer a few short questions. It takes about [XX] minutes.
> - **Risks:** we do not expect any risks beyond those of everyday computer
>   use. Some questions ask about unusual experiences; you may stop at any
>   time.
> - **Benefits:** there is no direct benefit to you. [Compensation, if any.]
> - **Your data:** your responses are stored under a study ID, without your
>   name. Results are reported only for groups or without identifying
>   information. [Storage period and who has access.] [Whether anonymised data
>   may be shared publicly, e.g. on OSF.]
> - **Voluntary participation:** taking part is voluntary. You may stop at any
>   time by closing this page, and nothing will be saved. [How to withdraw data
>   after submission.]
> - **Contact:** [researcher email]. For questions about your rights as a
>   participant: [ethics committee contact]. [Ethics approval number.]
>
> ☐ I have read the information above, I am at least 18 years old, and I agree
> to take part.
>
> [I agree — continue]  [I do not agree]

### 13. Screens too small for the images

**What.** Today a participant whose window cannot fit both 512 px images can
continue after a warning, and then sees cut-off images (`images_fit = false`).

**How the check works now.**
- **Scaling is off.** `display.allow_scaling` is `false`, so each image is
  always drawn at 512 × 512 CSS pixels and never shrunk to fit. The option to
  shrink exists; an earlier version may have used it.
- **The registered design needs it off.** The image features were measured
  on the 512 px images, and shrinking would change the fine detail, which is
  what the contrast feature measures.
- **The measurement.** After switching to full screen, the experiment
  measures the window. `images_fit` is `true` when it is at least 1056 px wide
  (two images and the gap). `viewport_ok` also requires the recommended
  1120 × 680.
- **The warning.** If the window is smaller, a warning appears. Below 800 px
  wide the participant cannot go on; between 800 and 1056 px they can press
  SPACE and continue with cut-off images.
- **In practice** it rarely fires. A full-screen laptop at 1366 × 768, or a
  1920 × 1080 screen at 150% system scaling (1280 × 720), passes. High-density
  screens (Retina) also pass: the images are drawn at 512 CSS pixels and
  enlarged by the screen, not cut off.

**Why it matters.** A session with cut-off images is not analysable under the
registered exclusion rules, so the person would do the whole task and then be
excluded.

**Decided.** When the images do not fit, stop at the start,
with a message asking the participant to use a larger window or screen and come
back. Do not offer a "continue anyway" button. Check the height as well as the
width, so that the images and the question above them are never cut off at the
bottom. Keep `images_fit` in the data as a check.

### 14. Length and break text

**What.** Time a full session with the new order and 211 trials. Update the
length stated to participants in both languages, and set `break_every` so the
breaks split the pair block into roughly equal parts.

**Why.** The current text and the break spacing assume about 140 trials.

### 15. Japanese text

**What.** Every new or changed participant-facing string needs a Japanese
version. Write drafts and collect them, with their English, in one file for the
research assistant to check.

**Why.** The study runs in Japanese. The English and Japanese versions must say
the same thing, and the anchoring and control texts are preregistered word for
word.

### 16. Documentation

**What.** Update `README.md`, `SETUP.md` and `DATA_DICTIONARY.md` to the new
design:
- the new columns: `group`, `interview_date`, the episode note, the grid
  fields, the word ids, the state items;
- the new counts;
- the session order;
- no `correct_response`.

**Why.** The data dictionary is how the data will be read and analysed.
Today it describes the old design (6 catch trials, 137 trials, "lifeless" as a
target).

### 17. Launch checklist (do not apply yet)

**What.** A short list in the README of what to switch before real data
collection:
- `debug: false`;
- the real researcher contact;
- the version string bumped (for example `1.0.0`);
- `allow_manual_id` decided;
- the test IDs kept but excluded at analysis;
- DataPipe "Enable data collection" switched on for both destinations;
- the OSF components private.

**Why.** The settings are still in development mode. They must stay that way
while testing, so the list is written now and applied at launch.

### 18. Every block can be switched off

**What.** Each block of the session has its own on/off switch in
`CONFIG.blocks`, as the existing blocks do. Each new block gets one:
- `consent`;
- `anchoring`;
- `grid_episode` and `grid_normal`, separately;
- `state`.

`words` and the rest keep theirs. The dev page lists every switch.

**Why.** It lets one part be tested without sitting through the whole session.
It is how the experiment is already built.

### Not changed

- **Device pixel ratio.** On high-density screens the 512 px images are
  upscaled. The experiment already records `device_pixel_ratio`, and it is used
  as a covariate. No code change.
- **The pair-trial mechanics** (fixation, blank, arrow keys, practice feedback,
  catch lockout) stay as they are.

---

## Part 2. Prompt for the other session

Copy everything between the two lines into a new session that has access to
both repositories, `dpdr_experiment` and `drl2drv_updated`. (`text-polish` was
merged into `main` on 20 Oct, and the site deploys from `main`.)

---

You are going to update a browser experiment so that it runs a new, finalised
study design. The experiment is the `dpdr_experiment` repository: jsPsych 7,
hosted on GitHub Pages, data uploaded through DataPipe to OSF. You have no
earlier context, so this message gives you everything you need. You need two
repositories:
- `zarcolmarioa/dpdr_experiment`: the experiment, which you change;
- `zarcolmarioa/drl2drv_updated`: the research project, which holds the
  archived stimuli and their checksums. Read it only.

After the first step below, read the experiment repository (`README.md`,
`SETUP.md`, `DATA_DICTIONARY.md`, `config.js`, `src/`, `text/`) and the two
stimulus packages before changing anything.

### The study in brief

People who have had episodes of derealization (the surroundings feel strange or
unreal) are shown two edited versions of the same photograph on each trial. They
press the left or right arrow key to choose the one that looks more like how
their surroundings looked during one specific episode. The two versions differ in
local contrast (a "fog"-like reduction), in colour saturation, or in both. A
per-person model later estimates how strongly each person's choices follow each
feature.

A **control group** of people who have never had such episodes does exactly the
same task, imagining how the surroundings might look to someone experiencing
derealization. The two groups must differ only in the wording of a few prompts.

The design is preregistered, so the stimuli, the trial order and the rules below
are fixed. Do not randomise, reorder, resample or regenerate anything that comes
from the stimulus packages.

### The stimulus packages, and the first step

The new stimuli are already committed in the experiment repository, in
`data/sets/v4/` (pair trials) and `data/sets/v4/grid/` (grid). The frozen
originals are archived in `drl2drv_updated` under `data_trials/`. Your **first
step**, before any code change, is to verify that the copies are exact:

1. **The pair package.** From `data/sets/v4/` in `dpdr_experiment`, check
   every file against the frozen checksums. Take the lines of
   `drl2drv_updated/data_trials/frozen_trial_list_v4/checksums.sha256` that
   start with `package/`, strip that prefix, and run `sha256sum -c`. All 354
   must pass.
2. **The grid package.** `data/sets/v4/grid/` must be byte-identical to
   `drl2drv_updated/data_trials/grid_block_v4/package/`, ignoring that
   folder's `previews/` (for example, `diff -r --exclude=previews`).
3. **If anything fails,** stop and tell the user. Never edit, re-encode,
   resize or rename these files.

What the packages contain:

- **The pair-trial package (`data/sets/v4/`):**
  - `trial_list.json`: an array of 211 trials in presentation order: 6
    practice (trials 1–6), 60 `sat_only`, 60 `fog_only`, 30 `together`,
    30 `conflict`, 15 `consistency` and 10 `catch`. 180 rows have
    `analysed: true`.
  - **Its columns:** `trial`, `trial_type`, `analysed`, `scene_id`,
    `category`, `left_path`, `right_path`, `a_is_left`, `pred_sat`,
    `pred_fog`, `d_sat`, `d_fog`, `z_sat`, `z_fog`, `mag_bin`,
    `catch_mode` and `repeat_of`.

  Differences from the current format:
  - new columns `pred_sat`, `pred_fog` and `catch_mode`. `catch_mode` is
    `no_images` on catch rows and null elsewhere; `repeat_of` is set only on
    consistency rows;
  - **no `correct_response` column**;
  - `preload.json` is a plain list of 334 paths. The 15 images used only by
    catch rows are shipped in `stimuli/` but not listed, because catch trials
    show no images.
- **The grid package (`data/sets/v4/grid/`).** Its `interface_spec.md` is the
  contract for what the grid must render and record: read it in full. In
  outline:
  - `grid_trials.json` is one object with `version`, `blocks` (episode and
    normal, with their `none_button` and `free_text_on_none` flags),
    `set_assignment` (rule `random_recorded`, and the two codes), `layout`
    (the cell-size constants), `feature_sd` and `sets`;
  - `sets.A` and `sets.B` are each a **list** of four screens;
  - each screen has `screen`, `feature` (`fog` or `colour`) and
    `column_order`, plus:
    - `columns` in display order, each with `display_column`,
      `column_label` and `level_index`, among other fields;
    - `rows`, three scenes, each with `scene_id` and `images` in the same
      order as `columns`;
  - `preload.json` is `{"images": [...]}`, 144 paths.

  The current loader, validator and `src/phases/grid.js` expect an older
  format and will reject this one.

Do not edit anything inside these two package folders.

### What to change

1. **Default set.** Make `v4` the default stimulus set, remove the old sets
   from `available_sets` (leave their folders in place), and update the loader
   and `src/validate.js` to the v4 format:
   - expected counts as above;
   - accept the new columns, and do not require `correct_response`;
   - catch rows in `no_images` mode need no preloaded images.

   Remove every use of `correct_response` from the code (`record.js`,
   `mock.js`, the validator), and update the comments that still say 137 or 143
   trials or 227 images.

2. **Catch trials.** Extend `CONFIG.catch_trials.sequence` to 10 targets, 5
   `ArrowUp` and 5 `ArrowDown`, in a fixed mixed order with no more than two
   identical targets in a row. Add a validator check that the sequence length
   equals the number of catch rows. Everything else about catch trials stays
   as it is.

3. **Session order** (the same for both groups):
   1. consent screen (new, item 14);
   2. ID screen;
   3. full screen and viewport check;
   4. brightness confirmation;
   5. gamma matching;
   6. anchoring screen (new);
   7. practice;
   8. pair trials;
   9. episode grid block;
   10. word ratings;
   11. normal-appearance grid block;
   12. state items (new, optional);
   13. save;
   14. completion screen.

   **Every block has its own switch in `CONFIG.blocks`**, as the existing ones
   do: add `consent`, `anchoring`, `grid_episode`, `grid_normal` (separately)
   and `state`, and make the dev page list them all.

   The word ratings currently run before the image task; move them. Update the
   header comment of `main.js` and any comment that says the ratings must come
   first. The images should still start downloading in the background early in
   the session, as now.

4. **Group.** Read the group from a URL parameter, `group=participant` or
   `group=control`. If it is missing or invalid, do not start: show the same
   kind of "incomplete link" message the ID screen uses. Stamp `group` on
   every row. The group changes only the wording of:
   - the anchoring screen;
   - the trial prompt;
   - the episode-grid question;
   - the word-rating instruction and question.

   Keep all group-dependent strings in `text/en/` and `text/ja/`. Allow the
   dev page to choose the group.

5. **Viewing-conditions note.** On the brightness screen, add a short
   paragraph: if they can, participants should switch off night mode,
   blue-light or colour filters; if they do not know whether these are on, or
   how to switch them off, they should not worry and simply continue. Nothing
   is confirmed or recorded.

6. **Anchoring screen** (new, before the practice). Proposed English wording,
   to be kept unless the user changes it:
   - **Participants:** "Please bring to mind one specific episode in which your
     surroundings felt different from usual, ideally a recent one that you
     remember clearly. Throughout this study, please answer about that
     episode." Then an **optional** text box: "In a few words, which episode
     is it? (for example, where you were)". Record the text as
     `episode_note`, or empty.
   - **Controls:** "Derealization is an experience in which one's surroundings
     feel strange or detached, as if one were cut off from the world. In this
     study, please imagine how the surroundings might look to someone
     experiencing derealization." No text box.
   - Do not mention any interview.
   - Do not describe the experience with visual words (foggy, blurred,
     faded, dull, colourless, unreal, dreamlike, flat, veiled, as if behind
     glass). This rule applies to every instruction you write.

7. **Trial prompt.**
   - Participants: "Which looks more like how your surroundings looked during
     that episode?"
   - Controls: "Which looks more like how the surroundings might look to
     someone experiencing derealization?"

8. **Word ratings.** Replace the list in `text/en/words.js` and
   `text/ja/words.js` with these ten words. The ids become column names and
   must be exactly as given:

   | id | EN | JA |
   |---|---|---|
   | `foggy` | foggy | 霞がかった |
   | `faded` | faded | 色あせた |
   | `veiled` | veiled | ベールがかかったような |
   | `lifeless` | lifeless | 生気のない |
   | `unreal` | unreal | 現実味のない |
   | `dreamlike` | dreamlike | 夢のような |
   | `distant` | distant | 遠く感じる |
   | `like_a_picture` | like a picture | 絵のような |
   | `oppressive` | oppressive | 重苦しい |
   | `threatening` | threatening | 不穏な |

   The 0–6 scale, the random order (recorded) and the hand-built grid stay as
   they are.
   - **The two targets** are `foggy` and `faded`. In `record.js`, the promoted
     session columns become `rating_foggy` and `rating_faded`, and
     `rating_lifeless` goes.
   - **The instruction and the one-line question** become, for participants,
     "During that episode, how well does each word describe how your
     surroundings looked?"; for controls, the same about how the surroundings
     might look to someone experiencing derealization.
   - **Remove** any text saying the ratings come before the main task, and
     update the file headers, which describe the old rationale.

9. **Grid blocks.** Rewrite the grid loading, validation and phase for the v4
   grid package, following `interface_spec.md`:
   - **Two blocks.**
     - `episode`, after the pair trials. It has a "None of these" button, and
       a free-text box after "none" (the existing `grid_none_text` feature):
       on for this block only.
     - `normal`, after the word ratings. No "none" button and no free text.
   - **The set swap.** At session start draw `set_code` = 0 or 1 with
     probability 0.5, record it on every row, and use it for the whole session.
     Code 0: episode block shows set A, normal block shows set B. Code 1: the
     reverse.
   - **Rendering.** Present each set's screens in `screen` order. `columns` is
     already in display order: do not reverse it again. A click anywhere in a
     column selects it. Compute **one** cell size for the whole block from the
     widest screen, using the formula in `interface_spec.md`.
   - **Record per screen:**
     - `block`, `set_code`, `scene_set`, `screen`, `feature`, `column_order`,
       `scene_ids`;
     - `chosen_display_column` (1-based, as displayed) **and**
       `chosen_level_index`;
     - `chose_none`, `free_text`;
     - `rt_ms`, `cell_px`, `inner_w`, `inner_h`.

     Do not copy level parameters or measured features into the data; they
     are in `grid_trials.csv`.
   - **Questions.**
     - Episode block: participants, "Which column looks most like how your
       surroundings looked during that episode?"; controls, "Which column
       looks most like how the surroundings might look to someone experiencing
       derealization?"
     - Normal block, both groups: "Which column looks most like how your
       surroundings normally look?"

     Keep the existing short instruction screen before each block, adapted to
     the two blocks.
   - **Remove** the old per-set `grid_sets` mechanism if the new structure
     makes it unnecessary, and the `haze`/`chroma` field names.

10. **State items** (new, optional block at the end, `CONFIG.blocks.state`,
    default on).
    - **The items:** four items from a published questionnaire, numbered 2, 13,
      19 and 29. The user will paste their exact English and Japanese wording.
      Put clearly marked placeholders in `text/en/` and `text/ja/` (for
      example `'[CDS item 2: paste wording]'`), so that only the strings need
      filling in.
    - **The screen:** one screen with the lead-in "Right now, at this moment:"
      (placeholder for the Japanese), then the four items in a fixed order.
    - **The scale:** each item on the same 0–6 scale as the word ratings, from
      "not at all" (0) to "completely" (6). Use radio buttons with no default
      answer, and reuse the word-rating screen's layout and code where
      possible.
    - **Record** one column per item: `state_q02`, `state_q13`, `state_q19`,
      `state_q29`.

11. **Interview date.** Read an optional URL parameter `idate` in the form
    `YYYY-MM-DD`, validate it, and stamp it on every row as `interview_date`
    (empty if absent or invalid). Participants are not asked for it. The
    researcher who schedules the interviews adds it to each participant's
    link.

12. **Images that do not fit.** `display.allow_scaling` stays `false`: images
    are always 512 × 512 CSS pixels. When the full-screen window cannot show
    both images and the question without cutting them off, by width or by
    height, do not offer "continue anyway". Stop with a message asking the
    participant to enlarge the window, or to use a larger screen, and to reopen
    the link. Keep `images_fit` and the viewport columns in the data.

13. **Length and breaks.** Run a full session in `dev.html` and time it. Update
    the stated session length in both languages, and set
    `CONFIG.timing.break_every` so that the breaks split the 205 non-practice
    pair trials into roughly equal parts. Update the break text if it
    mentions the old totals.

14. **Consent screen** (new, the first screen, `CONFIG.blocks.consent`).
    - **Layout:** the consent text, an "I agree" checkbox, and two buttons:
      "I agree, continue" (enabled only when the box is ticked) and "I do not
      agree".
    - **Agreeing** continues the session and stamps `consent_given = true` on
      every row.
    - **Declining** shows a short thank-you screen and ends the session with
      **nothing uploaded**: no response data and no contact file.
    - **The text is a placeholder** that will be replaced with the approved
      wording. Put it in a new `consent` section of
      `text/en/instructions.js` and `text/ja/instructions.js`, so replacing it
      means editing those two files only, and mark it clearly as a
      placeholder in a comment.
    - **The English placeholder:**

      > **Consent to take part.** You are invited to take part in a research
      > study about how people perceive their surroundings. It is run by
      > [researcher names] at [institution].
      >
      > - **What you will do:** look at pairs of photographs and choose between
      >   them, then answer a few short questions. It takes about [XX]
      >   minutes.
      > - **Risks:** we do not expect any risks beyond those of everyday
      >   computer use. Some questions ask about unusual experiences; you may
      >   stop at any time.
      > - **Benefits:** there is no direct benefit to you. [Compensation, if
      >   any.]
      > - **Your data:** your responses are stored under a study ID, without
      >   your name. Results are reported only for groups or without
      >   identifying information. [Storage period and who has access.]
      >   [Whether anonymised data may be shared publicly, e.g. on OSF.]
      > - **Voluntary participation:** taking part is voluntary. You may stop
      >   at any time by closing this page, and nothing will be saved. [How to
      >   withdraw data after submission.]
      > - **Contact:** [researcher email]. For questions about your rights as a
      >   participant: [ethics committee contact]. [Ethics approval number.]
      >
      > Checkbox: "I have read the information above, I am at least 18 years
      > old, and I agree to take part."

    - **Japanese:** write a draft in the same structure, keeping the bracketed
      gaps.

15. **Japanese.** Write a Japanese draft for every new or changed string, in
    the existing style of `text/ja/`. List each one, with its English, in a new
    file, `text/ja/STRINGS_TO_CHECK.md`, for a native speaker to check.

16. **Documentation.** Update:
    - `DATA_DICTIONARY.md`: every new column; the new counts; the grid fields;
      the word ids; the state items; no `correct_response`; the session order;
      the group;
    - `README.md` and `SETUP.md`: how to build the invitation links with
      `pid`, `group` and `idate`, and the session order.

    Add to the README a **launch checklist**, but do **not** apply it now:
    - `debug: false`;
    - a real `researcher_contact`;
    - the version bumped to `1.0.0`;
    - `allow_manual_id` decided;
    - the test IDs excluded at analysis;
    - DataPipe data collection enabled for both destinations;
    - the OSF components private.

    Set `experiment.version` to a new development version, for example
    `0.9.0-v4`.

### How to work

- **Start** by checking the packages (the first step above), then read the
  repository and the packages. Before changing any code, reply with:
  1. the result of the package check;
  2. your understanding, in a few sentences, of the study and of what this
     list asks you to change;
  3. a short plan: the files you will change, in what order;
  4. anything in this list that is unclear or conflicts with what you found
     in the code, as questions.

  **Then wait for the user's go.**
- **Work on a new branch from `main`** and open a pull request. Keep the existing coding style and
  the module structure. Every setting stays in `config.js`.
- **Do not change** the trial mechanics (fixation, blank, keys, practice
  feedback, catch lockout), the calibration tasks, the ID and contact handling,
  or the upload code, except where this list says so.
- **Test** before declaring it done:
  - the validator passes on the v4 packages;
  - a full session runs in local mode, in both languages and for both groups
    (headless Chromium via Playwright is fine if available);
  - check one downloaded CSV of each group: every listed column is present,
    there is one recorded row for each of the 211 trials (practice included),
    8 grid rows, and the word, state and session columns.

  Report what you tested and what you could not.
- **Ask the user** whenever something in this list is ambiguous or conflicts
  with the code, rather than guessing.

---
