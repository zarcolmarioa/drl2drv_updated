# Derealization image-choice experiment: how the experiment works

**Purpose of this file.** A self-contained description of the experimental
design: who takes part, what the participant is asked at each step, how a trial
is built, what is analysed, and why each element is there. It assumes no prior
context.

**Status (20 Oct 2026).** This file describes the design **as registered**. Some
of it is not yet in the experiment code: the new session order, the anchoring
screen, the new word list, the two grid blocks and the optional state block.
Those changes are listed for the experiment repository and are marked
*(code change pending)* below.

**Code:** <https://github.com/zarcolmarioa/dpdr_experiment> (browser experiment,
jsPsych, runs online). File paths in sections 4–7 refer to that repository.

**Companion documents (this repository):**
- `docs/v4_axes_pipeline_explained.md`: why these two image features. They come
  from text-defined "derealization axes" built from patients' own descriptions.
- `docs/v4_stimulus_power_explained.md`: how the stimuli, the trial list and
  the grid blocks were built, and the design analysis behind the decision rules.
  Its section 13 is the registered design in numbers.
- `docs/v4_stimulus_pipeline_explained.md`: the corpus, scene selection and the
  two manipulations in detail.

The analysis plan is stated in the preregistration; section 2 summarises it.

---

## 1. The experiment in one paragraph

Participants are people who report at least one recognisable episode of
derealization. They were selected from an online survey and interviewed about
their experiences beforehand. In the session they bring to mind **one specific
episode** and then do a long run of two-alternative forced-choice trials. On
each trial they see two edited versions of the same photograph side by side,
and press the left or right arrow key to choose the one that looks more like how
their surroundings looked during that episode. The two versions differ in
**local contrast**, in **colour saturation**, or in both.

They then see grid screens showing a whole range of versions at once, first
about the episode and later about how their surroundings normally look. They
also rate a short list of words. A **control group** with no such episodes does
the same task, imagining how the world might look to someone experiencing
derealization.

---

## 2. The questions the design answers

### 2.1 Background

People with derealization describe the world as faded, flat, lifeless, foggy,
veiled, as if seen through glass. Manipulating how "vivid" a photograph is, and
asking whether people with derealization find less vivid images less real, has
produced a null result (Kusztor et al.). This design differs from that work in
three ways:

- **The question.** Kusztor et al. asked how real images looked compared with
  everyday life. This study asks which image resembles the altered episode
  itself.
- **Timing.** If someone is in the altered state while looking, the test images
  are seen through the same altered perception, and differences can cancel out
  (the "El Greco" problem). Here participants look while not in an episode and
  compare with a remembered one.
- **Two features, not one.** "Vividness" bundles two complaints, and the design
  separates them:
  - **clarity** (*foggy*, veiled): physically, a loss of **local contrast**;
  - **colour** (*faded*, drained): physically, a loss of **saturation**.

  The two are manipulated independently, measured on every image, and put into
  competition within single trials. Kusztor et al. themselves recommend
  combining a task with an interview about the same experience, which this
  design does.

Claims are about the **remembered appearance** of a specific episode. Memory is
the only access anyone has to past episodes. Remembered scenes tend to be
recalled as clearer and more colourful than they were, which works *against*
the hypotheses, so a positive result is conservative.

### 2.2 Hypotheses

**H1, primary: contrast.** When two versions of a scene differ in local
contrast with saturation matched, participants choose the lower-contrast
version as more like the episode.

**H2, primary: colour.** When two versions differ in saturation with contrast
matched, participants choose the less colourful version.

H1 and H2 are each tested **per person, in both directions**:
- **The model.** For each person, a Firth bias-reduced logistic regression
  with no intercept:

  ```
  P(choose image A) = logistic(b_fog · pred_fog + b_sat · pred_sat)
  ```

  `pred_fog` and `pred_sat` are the feature differences between image A and
  image B, in corpus-SD units, signed so that **positive means image A is the
  predicted choice**:
  - `pred_fog = +d_fog / 9.348`: positive when A has lower local contrast;
  - `pred_sat = −d_sat / 39.182`: positive when A is less colourful.
- **Thresholds.** Each weight is compared with thresholds set on 20,000
  simulated random choosers, at which only 2.5% of them pass each end:
  - contrast +0.488 / −0.486;
  - colour +0.286 / −0.290.

  Each person is classed as detected in the predicted direction, in the
  opposite direction, or not detected.
- **The claim.** It succeeds if at least *k* people are detected in the
  predicted direction and at most *m* in the opposite direction. *k* and *m*
  are fixed in advance for every final sample size from 10 to 20, and each rule
  keeps the false-claim rate at or below 5%, including when people without the
  effect have personal tastes in either direction. H1 and H2 are two separate
  claims.

**C1, primary: participants against controls.** Participants' weights are
higher than controls', for each feature. The test is a one-sided Mann–Whitney
test per feature, Holm-corrected across the two. An attention check comes
first: if the groups differ in catch-trial failures or in very fast responses,
C1 is reported as inconclusive. Only a difference is informative, because a
similar result cannot separate memory of the experience from a shared idea of
what derealization "should" look like.

**Secondary.** A group-level one-sided t-test and Bayes factor (JZS, scale
0.707) on the per-person weights, per feature: does the group lean on average?
It is reported with and without a cap on extreme weights (the weight of a
person who answers every informative trial as predicted, just short of
separation), because one such person can dominate a mean. Proportions of
choices are used only in the exploratory analysis of the single-feature trials
that follows Suzuki et al.

**Exploratory, reported with intervals.**
- People differ in their contrast-to-colour balance, compared between people
  only.
- A person's descriptions predict their weights. The descriptions come from the
  CDS items, the interview (coded before the participant does the task), and
  the word ratings.
- The grid blocks and the shift from "normal" to "episode".
- Dose-response, additivity of the two features, and prevalence.
- A per-person binomial analysis of the single-feature trials, following
  Suzuki et al.

With fewer than 10 analysable participants, no H1 or H2 claim is made.

The rule for each sample size, the power figures and their assumptions are in
`docs/v4_stimulus_power_explained.md`, sections 12 and 13.

### 2.3 What a null result would mean

A null rules out "about half or more of the participants follow the feature at
a moderate strength". It does not rule out weak or rare effects, and each
person's weights with their intervals show how large an effect each person
could still have.

Two further cases:
- **The grid shows choices at the extremes or "none of these".** The
  manipulated range does not reach the experience.
- **The repeated trials show low consistency.** The task or memory is too
  noisy to conclude.

### 2.4 Who takes part, when recruitment stops, and who is analysed

**Where people come from.** Participants and controls come from one pool: the
respondents to an earlier online survey with the Cambridge Depersonalisation
Scale (CDS-29), who are sent a short online questionnaire. Both groups have
therefore read the same CDS wording. Data collection starts when the
questionnaire is sent, and the questionnaire stays open for **4 weeks**.

Below, **AfS** is the sum of the CDS item scores for items 2, 13, 19 and 29,
the Alienation from Surroundings factor of Sierra et al. (2005).

**Participants.** A respondent is eligible if they:
- answer Yes or Not sure to having had an experience in which they or the world
  around them felt different from usual, in a way that is hard to put into
  words;
- give an interpretable answer to how long it lasted (not an episode that
  occurred only while falling asleep, or under alcohol or medication);
- are willing to be contacted.

Eligible respondents are invited in priority tiers, formed from whether they say
"the world around you" was what felt different and from their AfS score (6 or
above, or below 6). Within a tier, the most recent episode comes first. Everyone
is interviewed before doing the task. Respondents with a high CDS total but
little or no AfS endorsement (total 70 or above, AfS 2 or below) are contrast
cases for the interview; if they also do the task, their results are **reported
separately** and they do not count toward the sample size for H1 and H2.

**Controls.** A respondent is a control candidate if they:
- answer No to having had such an experience ("Not sure" is excluded);
- are willing to be contacted;
- have AfS = 0, that is, none of CDS items 2, 13, 19 and 29 endorsed at all;
- have a CDS total of **20 or below**. This cut-off was fixed before
  registration. It is the median total among the respondents with AfS = 0.

All candidates are invited. Controls are not interviewed.

**When recruitment stops.**
- **Participants:** at **20 analysable participants**, or when every eligible
  candidate has been invited and has done the task, declined, or not replied
  within 3 weeks of a reminder, whichever comes first.
- **Second wave.** If fewer than 20 participants are analysable when the pool is
  exhausted, the questionnaire may be sent once more, open for 4 weeks, with the
  same rules.
- **Controls:** the same rule, with a maximum of **30**. If fewer than 20
  controls complete, a reserve band (AfS = 0 and a CDS total of 21 to 35) is
  invited; this is the controls' second wave.

**Who is analysable.** The thresholds and the decision rules were computed for
exactly the 180 analysed trials of the frozen trial list, so a person counts
only with all 180. The experiment makes this all or nothing: every trial needs
a key press, there is no time limit, and nothing is saved until the end. A
person is analysable if:
- the session is complete and uploaded;
- it is not a test or development session;
- both images fitted on the screen, so no image was cut off;
- they failed fewer than 3 of the 10 catch trials.

One restart is allowed; the first complete session is analysed, and restarts
are reported. People who are not analysable do not count toward the sample
size, and recruitment continues within the pool. The same rules apply to
controls. With fewer than 10 analysable participants, no H1 or H2 claim is made
(section 2.2). With fewer than 20 analysable controls, C1 is reported
descriptively.

**Excluded from one analysis only.**
- **Flat word ratings** (the same rating for all 10 words): left out of the
  word-rating analysis only. The ratings come after the task and are used only
  in an exploratory analysis.
- **A missing grid, word-rating or state block:** that person is left out of
  that analysis only. None of these blocks is needed for H1 or H2.

**Not exclusions.**
- The 15 consistency trials are a quality report only.
- Very fast responses (under 300 ms on image trials) are used only in the
  attention check that comes before C1, which compares the two groups.

**The first sessions.** The first sessions also check that the interview fits
its time limit and that the experiment runs as intended. They count as normal
sessions if neither the interview guide nor the experiment is changed
afterwards; otherwise they are reported separately.

---

## 3. The stimulus space and how a trial is built

### 3.1 The cell grid

Each of **47 scenes** exists in a grid of edited versions.

- **Colour** has seven levels: two above the original (more vivid), the
  original, and four below (less colourful). The colour edit scales chroma in
  CIELAB. Above the original it saturates softly toward the in-gamut ceiling,
  so colours never clip.
- **Contrast** has six levels: the original and five steps of contrast
  reduction. The contrast edit blends the image toward a haze and restores the
  mean brightness (`haze_eq`). **It also lowers saturation**: at the original
  colour it lowers saturation by about two thirds as many corpus SD as it
  lowers local contrast. Pairs are built so that this does not confound the
  analysis (section 3.3).

Levels are calibrated on **corpus percentiles** of the two measured features,
and every image carries its *measured* `contrast_local` and `sat_mean`, not just
the parameters used to make it. Images are 512 px square.

The untouched photograph (original colour, original contrast) **never appears
in a pair trial**, so participants cannot sort "edited" from "unedited". It
does appear in the grid blocks, where the original is one of the columns.

### 3.2 Both images of a pair come from the same scene

On every trial the two images are two versions of **the same photograph**, so
nothing about content differs between them. This removes content preferences
entirely, which is the largest single gain in precision available with a small
sample. The cost, that participants notice the pictures are versions of one
another, is handled by the instruction wording (section 5) and the
quality-control trials.

### 3.3 The four analysed trial types (180 trials)

| Trial type | n | How the two versions differ | What it is for |
|---|---|---|---|
| **Colour-only** (`sat_only`) | 60 | Same contrast level, different colour | H2: any choice is attributable to colour. Only larger colour differences are used (\|Δ saturation\| ≥ 31.1 units). |
| **Contrast-only** (`fog_only`) | 60 | Different contrast levels, with colour levels chosen so that *measured* saturation matches within 3 units | H1: the contrast edit drains colour, so only saturation-matched pairs isolate contrast. |
| **Together** (`together`) | 30 | One image has both lower contrast *and* less colour | The features point the same way; informs the sum of the two weights. |
| **Conflict** (`conflict`) | 30 | One image has lower contrast but *more* colour | The features point in opposite directions; this is the type that separates the two weights. |

The magnitudes are spread over fixed bins, and side is balanced exactly within
each type.

### 3.4 Trial order is fixed in a file

The sequence, the pairing, the side of each image and the feature differences
are computed in advance, frozen, and checksummed (`frozen_trial_list_v4`). The
experiment presents them in file order and randomises nothing.

The trial list carries `pred_sat` and `pred_fog`, which are the predictors used
in the analysis. It also carries `z_sat` and `z_fog`, which are **unsigned
sizes** kept for reference only; the model must not use them.

**Presented trials: 211.** That is 180 analysed, 15 consistency, 10 catch and 6
practice.

---

## 4. The session, step by step

The order is the same for participants and controls. Each step is a block that
can be switched on or off in `config.js`.

| # | Step | Participants | Controls |
|---|---|---|---|
| 1 | Identification and consent | ✓ | ✓ |
| 2 | Display calibration, with the viewing-conditions note | ✓ | ✓ |
| 3 | Anchoring screen *(code change pending)* | one specific episode | short definition text |
| 4 | Practice | ✓ | ✓ |
| 5 | Pair trials, with consistency and catch trials | ✓ | ✓ |
| 6 | Episode grid *(code change pending)* | ✓ | ✓ |
| 7 | Word ratings, about that episode *(moved; new list)* | ✓ | ✓ |
| 8 | Normal-appearance grid *(code change pending)* | ✓ | ✓ |
| 9 | State items, optional *(code change pending)* | ✓ | ✓ |
| 10 | End and upload | ✓ | ✓ |

### 4.1 Identification

One screen asks for the participant ID issued at recruitment. Name and email are
optional and, if given, are uploaded as a separate file to a separate
destination, so the response data never contain identifying information. The
**interview date** is recorded *(code change pending)* so the delay between
interview and task can be used as a covariate.

### 4.2 Display calibration

The manipulations *are* contrast and colour, so the display is part of the
measurement.
1. **Full screen.**
2. **Viewport check:** pair images are shown at a fixed pixel size and never
   scaled to fit.
3. **Brightness set to maximum,** confirmed.
4. **A short gamma-matching task,** which gives a per-participant covariate.

The instructions also ask participants to switch off night mode, blue-light or
colour filters if they can, and say not to worry if they do not know whether
these are on or how to switch them off. Nothing is verified.

### 4.3 Anchoring screen *(code change pending)*

Participants bring to mind **one specific episode**, ideally a recent one they
remember clearly, and answer every question about that episode. There is an
optional box for a few words identifying it. The screen does not mention the
interview. Every later prompt says "during that episode".

Controls instead read the same short definition, which contains no perceptual
words (for example: "Derealization is an experience in which one's
surroundings feel unreal or detached, as if one were cut off from the world"),
and answer about how the world might look to someone experiencing it. The text
is preregistered word for word.

### 4.4 Practice

There are 6 practice trials, drawn from colour-only, contrast-only and together
pairs at small to medium differences. The feedback teaches the key mapping only:
it says there is no right answer and nothing about *why* a choice was made.

### 4.5 The main block

**One trial:** a fixation cross, then the two images side by side with the
question above them, then a short blank. The question is: "Which looks more
like how your surroundings looked during that episode?" The response is the
left or right arrow key, self-paced, and the response time is recorded. Short
break screens appear periodically.

Two kinds of quality-control trial are interleaved and excluded from the
analysis:
- **15 consistency trials:** a pair shown earlier is repeated much later with
  the sides swapped. Agreement is reported per person as a **quality measure
  only**; it is not an exclusion rule.
- **10 catch trials:** these show **no images**, only an instruction to press
  the up or down arrow, balanced 5 and 5. A reflexive left or right press
  fails, and only the first press counts. Participants with **3 or more
  failures** are excluded.

### 4.6 The grid blocks *(code change pending)*

There are two blocks of four screens each. On each screen the **columns are the
levels of one feature**, with 6 columns per screen, and the **three rows are
different scenes**, so no single photograph drives the choice. Clicking a
column selects it. Each feature is shown twice, once with the columns running
one way and once reversed, so choosing a *level* can be told apart from
choosing a *position*.

| Block | When | Question | "None of these" |
|---|---|---|---|
| Episode | after the pair trials | how the surroundings looked during that episode | yes, with an optional free-text box |
| Normal appearance | after the word ratings | how the surroundings **normally** look | no |

- **Scene sets.** The two blocks use two matched sets of 12 scenes, swapped
  between participants. The draw is recorded in every data file.
- **What is analysed.** The shift from normal to episode is analysed per
  feature, in feature units, descriptively.
- **Three properties to keep in mind:**
  - the contrast axis has no level above the original, so contrast answers in
    the normal block are censored at the original and are counted;
  - the contrast screen's columns also lose some colour;
  - the block order is fixed, and the grid photographs also appear in the pair
    trials.

### 4.7 Word ratings *(moved after the episode grid; new list)*

For each word, participants rate on a **0–6 scale** how well it describes how
their surroundings looked during that episode. Controls rate how the
surroundings might look to someone experiencing derealization.

| Role | Words (EN) | Japanese (draft, being checked by a native speaker) |
|---|---|---|
| Targets | foggy, faded | 霞がかった, 色あせた |
| Close to the targets | veiled, lifeless | ベールがかかったような, 生気のない |
| Derealization baseline | unreal, dreamlike, distant, like a picture | 現実味のない, 夢のような, 遠く感じる, 絵のような |
| Fillers | oppressive, threatening | 重苦しい, 不穏な |

**How the words are used**
- Each target is read **relative to the person's baseline**, the mean of the
  four baseline words, so a general tendency to endorse derealization words is
  not mistaken for a specific one.
- "Veiled" and "lifeless" show whether people choose a nearby word instead of
  the target.
- The two fillers, one about low mood and one about anxiety, show whether high
  ratings reflect general distress.

The block comes after the episode grid so that the words do not prime the main
task. The cost is that ratings given after the task may echo the choices just
made, so they are a descriptive predictor.

### 4.8 State items *(optional; code change pending)*

A short block at the end: the four derealization items of the Cambridge
Depersonalisation Scale (items 2, 13, 19 and 29), in the CDS state format
("right now"). It is built as a switchable block so that it can be used or not.

### 4.9 End of session

The data are uploaded. If the upload fails, the participant can download the
data file and send it by email.

---

## 5. Instruction wording rules

All participant-facing text lives in `text/en/` and `text/ja/`. The wording
follows four rules, because breaking them would damage the measurement.

1. **The condition is named, never described.** Instructions name the episode
   but never describe it as foggy, faded, unreal, dreamlike, or as if behind
   glass. Those are the qualities the task and the word ratings measure. The
   word ratings necessarily show such words, which is one reason they come
   after the main task. The control text follows the same rule.
2. **The judgment is about the surroundings, not the self.** Every stimulus is
   a scene, so every prompt points at how the surroundings looked.
3. **The manipulation is acknowledged but unspecified.** The pictures "have
   been adjusted in different ways", with no hint at contrast or colour.
4. **Nothing suggests a detection task.** "Which looks more edited" would turn
   resemblance into discrimination.

---

## 6. What is recorded

One CSV per session, one row per screen. Session-level information is copied
onto every row: participant ID, language, display measurements, calibration
results, word ratings and the grid set code.

| Block | What it holds |
|---|---|
| `pairs` | one row per pair trial: trial number, trial type, which side was pressed, response time |
| `grid` | one row per grid screen: block, set, the chosen display column, the chosen **level index**, or "none" |
| word ratings | one rating per word |
| state items | one rating per item, if the block is on |

- **Which image was chosen** is derived in analysis from the side pressed and
  the frozen trial list. The predictors `pred_sat` and `pred_fog` are joined
  from the trial list by trial number; they are not copied into the data file.
- **There is no `correct_response` column.** The expected catch response lives
  in the experiment's configuration.
- **`trial_type` means two things.** jsPsych's column names the plugin; the
  design's trial type is a separate column.

The full column list is in `DATA_DICTIONARY.md` in the experiment repository.

---

## 7. Where things are in the experiment code

| Path | What it is |
|---|---|
| `config.js` | every setting: which blocks run, keys, display size, stimulus set, catch sequence, language |
| `data/sets/<set>/trial_list.json` | the trials: order, pairs, sides, predictors |
| `data/sets/<set>/grid/` | the grid package: blocks, sets, screens, images |
| `src/phases/pairs.js` | the pair block, including practice, consistency and catch trials |
| `src/phases/words.js` | the word-rating block |
| `src/phases/grid.js` | the grid blocks |
| `src/calibration/` | full screen, viewport check, brightness, gamma |
| `text/en/`, `text/ja/` | every participant-facing string |
| `src/validate.js` | checks run before anything is displayed |
| `DATA_DICTIONARY.md` | the output columns |
| `dev.html` | developer entry point: jump to any single block |
