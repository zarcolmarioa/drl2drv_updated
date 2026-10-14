# Derealization image-choice experiment — how the experiment works

**Purpose of this file.** A self-contained description of the experimental
design: what the participant is asked at each step, how a trial is built, and
why each element is there. It assumes no prior context.

**Code:** <https://github.com/zarcolmarioa/dpdr_experiment> (browser experiment,
jsPsych, runs online). File paths below refer to that repository.

**What this file does not cover:** how the phenomenological axes were derived,
how the scenes were chosen and the images generated, participants and
recruitment, and the statistical analysis plan. Those are documented
separately. Exact trial counts, stimulus-set names and image counts are
deliberately omitted here because they change; the authoritative values are in
the repository's trial list (`data/sets/<set>/trial_list.json`) and
configuration (`config.js`).

---

## 1. The experiment in one paragraph

Participants are people who report episodes of depersonalization or
derealization (DPDR) and who have already been interviewed about those
experiences. In the session they first rate how well a list of adjectives
describes how their surroundings *looked* during an episode. They then do a
long run of two-alternative forced-choice (2AFC) trials: on each trial they see
two edited versions of the same photograph side by side and press the left or
right arrow key to choose the one that looks more like how their surroundings
looked during an episode. Finally they see a small number of screens showing a
whole range of edited versions at once and choose the column closest to their
experience, or "none of these". The session ends by uploading the data.

---

## 2. The questions the design answers

### 2.1 Background

People with derealization describe the world as faded, flat, lifeless, foggy,
veiled, as if seen through glass. The obvious experimental move — manipulate
how "vivid" a photograph is and ask whether people with derealization prefer
the less vivid one — has been tried and produced a null result: physical image
vividness did not explain the sense of unreality.

This design assumes the reason may be that "vividness" is not one thing. Two
different complaints are bundled inside it:

- a **colour** complaint (*lifeless*, drained, flat) — physically, reduced
  chroma;
- a **clarity** complaint (*foggy*, veiled, as if behind glass) — physically,
  light scattering, i.e. haze.

These two can be manipulated independently and measured separately, so they can
be put into competition within a single trial. That is what this experiment
does.

### 2.2 The three questions

**Q1 — the "lifeless" question.** When two versions of the same scene differ
only in colour, do participants choose the less colourful one?

**Q2 — the "foggy" question.** When two versions differ only in haze, with
colour held matched, do participants choose the foggier one?

Q1 and Q2 are each answered per participant, as a proportion of choices, and
then at the group level.

**Q3 — the crossover, between participants.** Does the adjective a person
endorses predict which feature their choices track? A person who rates *foggy*
highly should show the stronger haze effect; a person who rates *lifeless*
highly should show the stronger colour effect.

Q3 is the question that motivates the whole structure. A single "vividness"
manipulation cannot answer it, because it cannot tell the two complaints apart.
Q1 and Q2 establish whether there is anything to attribute; Q3 attributes it.

A further question is asked by the final block (§4.6): does the manipulated
range cover the experience at all, or does the participant find none of the
images like it?

### 2.3 What a null result would mean

A null is a realistic outcome and the design is built so that it is
interpretable rather than merely disappointing. If choices do not track either
feature, that is evidence that the way people *describe* the experience does
not correspond to these low-level image properties — which is a result about
the vocabulary-to-perception mapping, not a failed experiment. For this to hold,
it must be impossible to attribute a null to a broken task, which is why the
session carries attention checks, consistency checks and display calibration.

---

## 3. The stimulus space and how a trial is built

### 3.1 The grid

Every scene (one photograph) exists in a **5 × 5 grid of versions**: five
levels of **chroma** (colour) crossed with five levels of **haze**. Haze is
applied with a physical scattering model, so it lifts the dark regions, reduces
local contrast and washes colour toward the airlight, as real haze does.

Two properties of the grid matter for the task:

- **No grid point is the untouched photograph.** The lowest haze level is a
  mild manipulation rather than zero. If one version were the original,
  participants could sort "edited" from "unedited" instead of responding to the
  features themselves.
- **Chroma does not reach zero.** A fully achromatic image reads as a
  black-and-white photograph — a different thing from a drained world.

Each image carries *measured* feature values, not just the parameters used to
make it, because each image is solved individually to hit its target.

### 3.2 Both images of a pair come from the same scene

On every trial the two images are two points in the grid of **the same
photograph**. Nothing about content differs between them: same room, same
objects, same composition.

This is a deliberate choice with a known cost. Pairing two *different*
photographs would make the editing less obvious, but then any preference could
be a preference for the content of one photograph rather than for the
manipulation. Same-scene pairing removes content variance entirely, which is
the largest single gain in precision available with a small sample. The cost —
that a participant may notice that the pictures are versions of one another —
is handled by the instruction wording (§5), by having no unedited version to
compare against, and by the quality-control trials.

### 3.3 The four analysed trial types

Each type is a different geometric relation between the two grid points, and
each answers a different part of the question.

| Trial type | How the two versions differ | What it is for |
|---|---|---|
| **Colour-only** (`sat_only`) | Same haze level, different chroma | Q1. Any choice is attributable to colour alone. |
| **Haze-only** (`fog_only`) | Different haze levels, with chroma chosen so that *measured* saturation matches | Q2. Haze drains colour as a side effect, so an unmatched foggy-versus-clear pair would confound haze with colour loss. Only colour-matched pairs isolate haze. |
| **Together** (`together`) | One image is both foggier *and* less colourful | The two features point the same way, as they do in natural haze. Informs the *sum* of the two sensitivities. |
| **Conflict** (`conflict`) | One image is foggier but *more* colourful | The two features point in opposite directions. Informs the *difference* between the two sensitivities — this is the type that separates them, and the direct test of Q3 within a participant. |

The haze-only type is the reason the stimulus space is a grid rather than a
single haze ladder: producing a colour-matched haze contrast requires moving in
two dimensions at once.

A column of the grid is never used on its own: moving straight down changes
both features together in a way that cannot be attributed.

### 3.4 Trial order is fixed in a file, not randomised at run time

The sequence of trials, the pairing, which image appears on the left, and the
feature differences for each trial are all computed in advance and stored in
`data/sets/<set>/trial_list.json`. The experiment presents them in file order
and randomises nothing. The reason is that the pairs are chosen deliberately —
to spread the feature differences evenly, to include pairs where one feature
differs a lot and the other barely at all, and to balance left/right exactly —
and run-time randomisation would undo that. Regenerating the trial list changes
the experiment with no change to the code.

Each trial row carries the standardised feature differences between its two
images (`z_sat`, `z_fog`, signed so that positive means image A is the less
colourful or the foggier one), which are the predictors used in analysis.

---

## 4. The session, step by step

Each step below is a block that can be switched on or off in `config.js`.

### 4.1 Participant identification

One screen asks for the participant ID issued at recruitment; name and email
are optional. The ID is what links the session to that participant's
questionnaire score and interview; it can also be prefilled from the invitation
link. Name and email, if given, are uploaded as a *separate* file to a separate
destination, so the response data never contains identifying information.

### 4.2 Display calibration

The manipulations *are* colour and shadow lifting, so the display is part of
the measurement. Four things happen, in this order:

1. **Full screen**, so the viewport is fixed for the whole session.
2. **Viewport check.** The two images are shown at a fixed pixel size for every
   participant — they are never scaled to fit — so that the physical size of the
   manipulation does not vary between people. A viewport too small to fit them
   produces a warning, and a very small one stops the session.
3. **Brightness.** The participant is asked to set screen brightness to maximum
   and confirm it. A dimmed screen hides exactly the dark-region differences
   that haze produces.
4. **Gamma.** A short luminance-matching task, repeated a few times, in which
   the participant adjusts a grey patch until it matches a checkerboard. This
   does not correct anything; it yields a per-participant estimate of the
   display's gamma that can be used as a covariate.

### 4.3 Word ratings

A list of adjectives is presented, and for each one the participant rates, on a
**0–6 scale**, how well it describes how their surroundings **looked** during an
episode of depersonalization or derealization.

Three design points:

- **The targets are *foggy* and *lifeless*.** These are the two predictors for
  Q3.
- **The rest of the list is fillers**, chosen to sit away from the two
  manipulated dimensions (they concern things like crowding, sound, weight or
  pace rather than colour or clarity). They make indiscriminate endorsement
  detectable: someone who rates every word highly is not describing an
  experience, and their target ratings carry no information. A filler that was
  itself about colour or clarity would correlate with the targets and defeat
  the purpose.
- **Ratings, not a forced choice between words.** This gives two continuous
  predictors rather than one categorical one, which is what the crossover
  analysis needs.

This block runs **before** the images, because it is the predictor: rated
afterwards, it would be contaminated by the fifteen minutes the participant had
just spent looking at manipulated photographs.

The instructions are shown on their own screen, and the rating grid on the
next, so the whole grid fits on screen without scrolling.

### 4.4 Practice

A few practice trials, identical in structure to the main trials. On the first
of them the chosen image is briefly outlined. The feedback teaches the key
mapping and nothing else: it says there is no right answer and says nothing
about *why* the participant chose what they chose. Confirming a reason would
endorse the choice as meaningful and invite consistency-seeking on later
trials.

### 4.5 The main block (2AFC)

**One trial:** a fixation cross, then the two images side by side with the
question above them, then a short blank.

**The question, in English:**

> Which looks more like how your surroundings looked to you during an episode?

**The response:** the left or right arrow key, self-paced, no time limit. Only
the *side* chosen is recorded; which image that was is reconstructed in analysis
from the trial list, which records which side image A was on. Keys rather than
mouse clicks keep the response time interpretable and the hand position fixed.

**Side balancing:** within each trial type, image A appears on the left in
exactly half the trials, so a bias toward one key cannot masquerade as a
preference for one manipulation.

**Breaks:** a short break screen appears periodically, stating progress.

Interleaved in the block, and excluded from all analysis, are two kinds of
quality-control trial:

- **Consistency trials.** A pair shown earlier is presented again much later
  with the sides swapped. Agreement across the two presentations measures
  whether the participant's responses are stable. The side swap means agreement
  cannot be produced by pressing the same key.
- **Attention checks (catch trials).** These show **no images at all**. The
  screen displays a short instruction to press the **up** or **down** arrow key.
  Removing the images is what makes the check work: with two pictures on screen,
  the trained "pick a side" response fires before anything is read, so an
  inattentive-but-habituated participant would pass. With nothing to choose
  between, the only available action is to read. The target is never left or
  right, so a reflexive side press fails immediately. Keys are inert for a
  moment when the screen appears, so a press carried over from the previous
  trial cannot land, and only the first press counts — a trial that waited for
  the correct key could only ever end in success and would discriminate nothing.

### 4.6 The grid block

After all pairwise trials, the participant sees a few screens, each laid out as
a matrix: **columns are levels of one feature** (haze on some screens, chroma on
others) and **rows are different scenes**. Clicking anywhere in a column selects
that whole column. A separate **"None of these"** button sits below, and the
choice can be changed before continuing.

**The question:** which column looks most like how their surroundings looked
during an episode.

**Why it exists.** The pairwise trials only ever ask "which of these two is
closer?". A participant can answer that consistently while finding neither
image anything like their experience. The grid shows the whole manipulated
range at once and asks an *absolute* question: where on this range was the
experience? Three outcomes are informative — everyone choosing the most extreme
column means the range does not go far enough; clustering in the middle means
the range is about right; "none of these" means the manipulations do not
capture the experience at all. That last answer is available nowhere else in
the session, and given the earlier null result it is a real possibility.

**Why it is last.** Seeing the whole range laid out reveals what is being
manipulated, so it cannot precede the pairwise trials.

**How it is used.** Descriptively, as a range check reported in counts. It is
not a test of Q1–Q3.

**Two further details.** Each feature is shown on more than one screen, once
with the levels running from least to most manipulated and once in the reverse
order, so that choosing a *level* can be told apart from choosing a *position*
on the screen. And the rows of each screen are different scenes, so no single
photograph drives the choice.

Unlike the pairwise panels, the grid cells cannot be shown at a fixed size — a
row of full-size images fits no laptop — so the cell size is computed from the
window and recorded on every row as a covariate.

### 4.7 End of session

The data are uploaded. If the upload fails, the participant is offered a button
that downloads their data file so it can be sent by email, rather than the
session being lost silently.

---

## 5. Instruction wording rules

The wording of every participant-facing screen follows four rules. They are not
stylistic; breaking them would damage the measurement. All participant-facing
text lives in `text/en/` and `text/ja/` (the study runs in English and
Japanese), and nothing displayed to a participant is written anywhere else in
the code.

1. **The condition is named, never described.** The instructions say "an episode
   of depersonalization or derealization". They never say "unreal",
   "dreamlike", "as if behind glass", "faded" or similar. Those words name
   perceptual qualities — and those qualities are exactly what the word ratings
   and the image manipulations measure. Using them in the instructions would
   prime both the predictor and the outcome.
2. **The judgment is anchored on the surroundings, not the self.**
   Depersonalization concerns the self feeling unreal; derealization concerns
   the world. Every stimulus is a scene, so the instruction must point at the
   world, or participants recall self-focused experiences that the images cannot
   represent.
3. **The manipulation is acknowledged but unspecified.** Both images are the
   same scene, so participants will notice that something was done. The
   instructions say the pictures "have been adjusted in different ways", which
   closes the question without hinting at colour or haze.
4. **Nothing suggests a detection task.** "Which looks more edited / distorted /
   altered" would convert a judgment about resemblance into a discrimination
   task — something people are good at, and which measures something else
   entirely.

---

## 6. What is recorded

The output is one CSV per session, one row per screen, with session-level
information (participant ID, language, stimulus set, display measurements,
calibration results, the target word ratings) copied onto every row.

The rows that matter:

| Block | What it holds |
|---|---|
| `word_ratings` | one rating per adjective |
| `pairs` | one row per pairwise trial: which images, the trial type, the standardised feature differences, which side was pressed, response time |
| `grid` | one row per grid screen: the chosen column, the level that column corresponds to, or "none" |

Two conventions are worth knowing before reading the data:

- **Only the side pressed is recorded.** Which image was chosen is derived in
  analysis by combining the side with the trial list's record of which side
  image A was on.
- **`trial_type` means two different things.** jsPsych writes its own
  `trial_type` column naming the plugin that produced the row; the design's
  trial type (colour-only, haze-only, together, conflict, …) is in a separate
  column.

The full column list is documented in `DATA_DICTIONARY.md` in the repository.

---

## 7. Where things are in the code

| Path | What it is |
|---|---|
| `config.js` | every setting: which blocks run, keys, display size, stimulus set, language, hosting |
| `data/sets/<set>/trial_list.json` | the trials: order, pairs, sides, feature differences |
| `data/sets/<set>/grid/grid_trials.json` | the grid screens: columns, levels, scenes |
| `src/phases/pairs.js` | the pairwise block, including practice and the attention checks |
| `src/phases/words.js` | the word-rating block |
| `src/phases/grid.js` | the grid block |
| `src/calibration/` | full screen, viewport check, brightness, gamma |
| `text/en/`, `text/ja/` | every participant-facing string |
| `src/validate.js` | pre-flight checks run before anything is displayed |
| `DATA_DICTIONARY.md` | the output columns |
| `dev.html` | developer entry point: jump straight to any single block |
