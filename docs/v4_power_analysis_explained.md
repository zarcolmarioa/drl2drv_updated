# The v4 power analysis: what the experiment can and cannot answer

This document explains, for a reader with no other context, the power analysis
for the v4 image experiment: what was simulated, what came out, which claims the
design supports, which it does not, and how the number of trials and the number
of participants change the answer.

- **Project folder:** all paths are relative to the project folder
  (`drl2drv_updated/`). Scripts are in its pipeline folder.
- **Script conventions:** every script is standalone, with a SETTINGS block at
  the top and no command-line arguments. Each writes its own output folder, a
  report (`*_report.txt`) and CSV tables, and never overwrites an existing
  result.
- **Scope:** the power analysis and the analysis plan it supports. Two companion
  documents cover the rest: *How the v4 derealization axes were built, validated
  and linked to image features* (where the two visual dimensions come from), and
  *How the v4 experiment stimuli were selected, edited and measured* (how the
  trials were built). The experimental design — task wording, screening
  questionnaire, response grid — is described elsewhere and is not discussed
  here; section 10 lists only the assumptions the power analysis depends on.
- **Every number in this document is taken from the final run's report or CSV
  tables.** Several earlier runs exist with superseded values; section 13 says
  which files are the final ones and how to tell.

## Guide: sections and scripts

| Section | Step | Script(s) |
|---|---|---|
| 1 | Key ideas: two levels, weights, floors | – |
| 2 | The analysis plan: what is claimed | – |
| 3 | Step 6: one participant — floors, crossover, additivity, trial count | `simulate_power_v4.py` |
| 4 | Step 7: N participants — responders, group test, stopping | `simulate_power_seq_v4.py` |
| 5 | The preregistration figures | `plot_power_seq_v4.py` |
| 6 | What the design can answer | (from 6 and 7) |
| 7 | What it cannot answer, and why | (from 6 and 7) |
| 8 | How the number of participants changes the answer | (from 7) |
| 9 | How the number of trials changes the answer | (from 6) |
| 10 | Assumptions the whole analysis rests on | – |
| 11 | Reading the raw tables: three traps | – |
| 12 | Suggested wording for the preregistration | – |
| 13 | Which files are final, and what is not included | – |

---

## 1. Key ideas: two levels, weights, floors

### The experiment, in one paragraph

On each of 180 analysed trials a participant sees two versions of the same
photograph and chooses which looks more like how the world looked to them during
an episode of derealization. The two versions differ in **saturation**
(`sat_mean`), in **fine detail** (`contrast_local`, the standard deviation of the
Laplacian of the grey image), or in both. Trials come in four types — colour
only, fog only, both cues agreeing (`together`), and the two cues opposed
(`conflict`). The stimulus document covers how these were built.

### What a weight is

Each participant's choices are fitted with a logistic model, no intercept:

```
    P(choose image A) = 1 / (1 + exp(-(b_sat * z_sat + b_fog * z_fog)))
```

`z_sat` and `z_fog` are the differences between the two images, each divided by
that scene's own range for that feature, and **oriented so a positive weight
always means "prefers the more manipulated image"** — the foggier one, and the
less saturated one. The two weights say what that person's recalled experience
tracks: colour, fog, both, or neither. Standardising means `b_sat = 1.0` and
`b_fog = 1.0` mean the same thing, and makes the numbers comparable with the
earlier v2 study.

To translate a weight into behaviour: a weight of **1.06 means choosing the
manipulated image 63% of the time at a typical difference and 74% at the
largest**. A weight of 0.85 means 61% and 70%. These are moderate effects, not
heroic ones.

### What a floor is

A **floor** is the smallest weight that can be told apart from a guesser *in a
single participant*. It is obtained by simulating 20,000 participants whose true
weights are exactly zero, fitting each one's 180 trials, and taking the
**97.5th percentile** of the resulting estimates — a one-sided test at 2.5%
false alarms.

**It is a percentile, not 1.96 × SD.** The colour weight's null distribution is
right-skewed, so the normal approximation understates the floor by about 11% and
would run the per-person test at roughly 4% false alarms instead of 2.5%. Three
of the uploaded reports show the superseded pair (0.662 / 0.564); the final ones
show the correct pair:

```
    floor, colour   0.735
    floor, fog      0.594
```

The two are carried **separately** everywhere. Averaging them would test colour
too leniently and fog too strictly, and the difference between the number of
colour-trackers and fog-trackers is itself a finding.

### The two levels, which answer different questions

This distinction governs everything below.

| | **Per-participant level** | **Group level** |
|---|---|---|
| Unit | one person's 180 trials | one number per person, across people |
| Statistic | `b_sat`, `b_fog` from the logistic fit | one-sample t-test on each person's proportion of "more manipulated" choices, plus a JZS Bayes factor (scale 0.707) |
| Question | does *this individual* track the feature? | is the *average* participant above chance? |
| Depends on trials? | yes — more trials lower the floor | no — once a person has a proportion, the test knows nothing about how many trials produced it |

**They can disagree in both directions.** Many people with small consistent
effects can make the group test significant while no individual is detectable. A
few strong responders among ten can support an individual-level claim while the
group mean is swamped by non-responders. The second case is the one this design
expects, and it is why the per-participant level carries the primary claim.

---

## 2. The analysis plan: what is claimed

**Primary, per participant, one claim per axis.** Each participant's colour and
fog weights are fitted from their own 180 trials. A weight above that feature's
floor counts as a **detection** (one-sided, 2.5% false alarms, both floors
stated in advance). The claim for an axis is supported when **at least
`k_required` participants independently exceed the floor on that same feature**.

> In at least *k* participants, local contrast reliably determined which of two
> images was judged to resemble their remembered episode.

and, in parallel,

> In at least *k* participants, saturation reliably determined which of two
> images was judged to resemble their remembered episode.

**The counting rule handles multiplicity without a correction formula.**
`k_required` is the smallest *k* for which P(≥ *k* false alarms among *N*
participants, each at 2.5%) ≤ 5%. It is **2 for samples up to 14** and **3 from
15 to 20**, and it is read from the table in section 8 for the sample size
actually obtained. The number must be stated before data collection.

**Secondary: the responder count.** Reported as a count with an interval, never
as a proportion with a p-value. The interval is wide at small N and section 8
says how wide, in advance.

**Secondary and underpowered: the group test.** Kept for comparability with
Kusztor et al. (2025). The plan declares in advance that it has low sensitivity
at the anticipated N and a substantial chance of returning evidence *for* the
null even when a minority genuinely responds.

**Supporting: dose-response.** The proportion computed separately within each of
three difference-magnitude bins. A rising curve cannot be produced by a response
bias — someone who always clicks left, or always prefers the darker picture,
gives a flat line because a bias does not care how large the difference is. This
is a bias check and a group-level descriptive pattern, **not a per-person test**:
section 7 gives the reason.

**The inconclusive clause.** If the group Bayes factor falls between 1/3 and 3,
or fewer than a stated number of participants complete the task, the group-level
question is reported as INCONCLUSIVE and inference is restricted to the
per-participant level. Saying this in advance is what separates "absence of
evidence" from "evidence of absence", and it is the only reason the Bayes factor
is worth keeping, since it buys no power.

**Exploratory: the crossover and additivity.** Both are reported with intervals
and labelled exploratory. Section 7 explains why they cannot be anything else.

---

## 3. Step 6: one participant

**Script:** `simulate_power_v4.py`. **Output:** `data/power_v4_ec/`.

Reads the trial lists produced by step 5 of the stimulus pipeline. No images and
no encoder: the trial list already carries every measured difference. The
logistic fit is a short Newton solver, so the whole run takes seconds.

**Two trial lists are analysed side by side** (`haze` and `haze_eq`) under
identical settings, because the stimulus pipeline built both. The quantities
that depend on the trial list differ slightly between them; the group-power and
Bayes-factor tables depend only on sample size and effect size, so they are
computed once and shared.

What the script does:

1. **Simulates a participant** with two given true weights, responding
   probabilistically to the real trial list through the model in section 1.
2. **Fits each simulated participant's own trials** and records the recovered
   weights, so the scatter of recovered weights is a property of *this* trial
   list.
3. **Computes the floors** from 20,000 null participants, as the 97.5th
   percentile (`FLOOR_ALPHA = 0.025`). The report prints the 1.96 × SD value
   alongside for comparison.
4. **Builds a per-person detection curve**: for each true weight, the share of
   simulated participants whose estimate clears the floor.
5. **Tests the crossover** — whether one person's two weights differ from *each
   other*, simulated at a baseline weight of 0.7 on both cues.
6. **Tests additivity** — whether the two cues combine more or less than
   additively, using an interaction term that must be **antisymmetric under the
   side swap** (section 7 explains why the obvious predictor cannot be used),
   both within one person and pooled across several.
7. **Computes the expected dose-response curve** across the trial list's
   magnitude bins.
8. **Sweeps the trial count** (20 to 100 trials per type) to show what more
   trials would buy.
9. **Computes group-level power** — the one-sample t-test and the JZS Bayes
   factor — as a function of N and effect size.

### Result: recovery works

```
        scenario  true_b_sat  true_b_fog  mean_b_sat  sd_b_sat  mean_b_fog  sd_b_fog
   tracks colour         1.0         0.0       0.999     0.383       0.004     0.304
      tracks fog         0.0         1.0       0.020     0.382       0.989     0.304
     tracks both         0.7         0.7       0.695     0.393       0.716     0.336
  tracks neither         0.0         0.0       0.011     0.338       0.010     0.288
```

The weights are recovered without bias and without leaking into each other — a
pure colour-tracker returns `b_fog = 0.004`. Predictor collinearity is
`r = +0.123`, VIF 1.02, so the two weights are separately estimable. That is a
property the trial list was built for.

### Result: the per-person detection curve

```
    colour  floor 0.74, 80% detection at 1.06
    fog     floor 0.59, 80% detection at 0.85

   true_weight  colour   fog
           0.2   0.092 0.077
           0.4   0.213 0.225
           0.6   0.378 0.502
           0.8   0.560 0.755
           1.0   0.757 0.923
           1.2   0.890 0.972
           1.4   0.970 0.993
           1.6   0.993 1.000
```

Read it this way: at exactly the floor you catch a responder about half the time;
the weight for 80% detection is the number to plan with. **Fog is easier than
colour at every true weight.** A parallel claim on the colour axis is therefore a
weaker claim, and should be presented as such rather than symmetrically.

Weak responders are missed: at a true weight of 0.6, colour detection is 38% and
fog 50%. That is the honest limit of 180 trials.

### Result: the two trial lists are near-identical

```
                                  haze_eq     haze
floor, colour                       0.735    0.692
floor, fog                          0.594    0.583
weight for 80% detection, colour    1.065    1.006
weight for 80% detection, fog       0.853    0.857
gap for 80% crossover detection     1.423    1.424
individual weight SD                0.313    0.326
```

Power did not decide the fog family. `haze_eq` was chosen on stimulus grounds —
a larger conflict pool, a better envelope, lower cross-talk onto the colour
feature — as the stimulus document's section 6 sets out.

### Result: group-level power, and the Bayes factor's ceiling

```
  Power (one-sample t-test):
  d    0.3   0.5   0.8   1.0   1.5
  8   0.18  0.36  0.65  0.82  0.99
  10  0.22  0.43  0.75  0.90  1.00
  15  0.28  0.56  0.91  0.97  1.00
  20  0.37  0.68  0.96  1.00  1.00

  Bayes factors under a TRUE null:
   n  median_BF10  p_evidence_against  p_inconclusive
   8        0.421               0.000           0.982
  10        0.390               0.305           0.658
  15        0.321               0.540           0.430
  20        0.285               0.642           0.343
  40        0.213               0.778           0.207
```

The second table is the important one for a small study. At N = 10, a study where
nothing is there has a 31% chance of being able to say so and a 66% chance of
being inconclusive. Kusztor et al. reached BF₀₁ > 1000 at N = 200; here the
realistic ceiling is "moderate evidence against an effect". That is still far
better than reporting "p = 0.34, we conclude nothing", which is why the Bayes
factor is kept — but it must be declared as a ceiling.

**Files:** `floors.csv`, `recovery.csv`, `detection.csv`, `crossover.csv`,
`additivity.csv`, `additivity_pooled.csv`, `dose_response.csv`,
`trial_sweep.csv`, `group_power.csv`, `bf_null.csv`, `comparison.csv`, figures,
`power_report.txt`.

---

## 4. Step 7: N participants

**Script:** `simulate_power_seq_v4.py`. **Output:** `data/power_seq_v4/`.

Step 6 asks what one participant can show. Step 7 asks what a *sample* can show,
and its central insight is that **the sample is a mixture**: some participants
genuinely track these features and some do not, and the share that does is
unknown.

What the script does:

1. **Recomputes the floors from the trial list** rather than taking step 6's
   numbers on trust. They agree: 0.735 and 0.594.
2. **Builds a scenario grid** — 312 scenarios, 400 simulated studies each, at
   every sample size from 4 to 20:
   - `PATTERNS`: which features responders track — `colour only`, `fog only`,
     `both`, `neither`
   - `EFFECT_WEIGHTS = [0.5, 1.0, 1.5]` — how strong a responder's effect is
   - `SPREAD_FACTORS = [0.0, 0.5, 1.0, 2.0]` — how much responders differ from
     each other; **0.0 is the true null**, where every weight is exactly its
     stated value
   - `WEIGHT_CORRELATIONS = [0.0, 0.5]`
   - `RESPONDER_SHARES = [1.0, 0.5, 0.2, 0.1]` — the fraction of the sample that
     genuinely responds
3. **Applies the counting rule.** For each N it computes `k_required` as the
   smallest *k* with P(≥ *k* false alarms) ≤ 5% at a 2.5% per-person rate, and
   reports the false-positive rate that rule actually buys.
4. **Records both marginal and joint outcomes** — the claim for fog, for colour,
   and the four joint possibilities (both claimed, fog only, colour only,
   neither), which cannot be reconstructed from the marginals.
5. **Runs the group test** inside every scenario, recording whether the Bayes
   factor reaches 3 (effect), 1/3 (evidence for the null) or neither
   (inconclusive).
6. **Simulates sequential monitoring** — reading the group Bayes factor after
   each participant from N = 4 and stopping at 3 or 1/3 — against the per-person
   equivalent.
7. **Checks the Bayes-factor machinery.** Computed on a cached t-grid and
   verified against the reference form to a worst relative difference of
   7.2 × 10⁻¹¹.

### Result: the counting rule is verified, not just derived

With `spread = 0.0` and the `neither` pattern, every simulated weight is exactly
zero — a genuine null. Across all 102 such cells the observed false-alarm rate
tracks the binomial prediction:

```
    n   k  predicted  observed fog  observed colour   P(at least one)
    10  2     0.0246        0.0238           0.0258            0.2150
    14  2     0.0466        0.0392           0.0467            0.2800
    15  3     0.0057        0.0054           0.0042            0.2954
    20  3     0.0130        0.0154           0.0142            0.3725
```

Worst absolute deviation over all true-null cells: **0.0175**. A reviewer who
doubts the rule is answered by this table and nothing else.

The right-hand column is the reason the rule exists. **At least one** detection
among ten participants each tested at 2.5% happens **22% of the time when nobody
responds at all.** The claim therefore needs the *count*, not the existence of a
single flagged participant.

### Result: the existence claim, by sample size and responder share

P(the fog claim succeeds), from the final lookup table:

```
share    0.1    0.2    0.5    1.0
n
 4     0.030  0.118  0.392  0.870
 6     0.072  0.205  0.620  0.978
 8     0.148  0.322  0.785  0.998
10     0.228  0.442  0.895  1.000
12     0.300  0.512  0.948  1.000
14     0.378  0.630  0.978  1.000
15     0.130  0.372  0.908  1.000   <- k_required steps 2 -> 3
17     0.165  0.450  0.960  1.000
20     0.248  0.570  0.985  1.000
```

**Read it across, not down.** The responder share moves the answer far more than
the sample size does. At a share of 1.0 any N ≥ 6 gives ≥ 0.978; at a share of
0.2 even N = 14 gives 0.630; at 0.1, 0.378. The share cannot be known in
advance, which is itself a reason to preregister an existence claim rather than a
prevalence claim.

### Result: sequential monitoring is not worth doing

```
   share            outcome  share_of_studies  median_stop_n
     1.0             effect             0.978            5.0
     1.0               null             0.015            4.0
     0.5             effect             0.730            9.0
     0.5               null             0.152            4.0
     0.2             effect             0.268            9.0
     0.2               null             0.385            5.0
     0.1             effect             0.135            7.0
     0.1               null             0.615            5.0
```

At low responder shares a monitored study stops early **for the wrong reason**:
at a share of 0.1 it stops with "evidence for the null" 62% of the time, at a
median of five participants — before the handful of real responders have had a
chance to appear. The per-person equivalent behaves better (the first individual
detection arrives at a median of four participants even at a share of 0.2). The
plan therefore takes **a single final look**, with no interim monitoring.

**Files:** `responders.csv` (every scenario × N), `lookup_by_n.csv` (the headline
patterns), `sequential.csv`, `comparison.csv`, `power_seq_report.txt`.

---

## 5. The preregistration figures

**Script:** `plot_power_seq_v4.py`. **Output:**
`data/power_seq_v4/<set>/figures/`.

Deliberately separate from step 7: a simulation takes a couple of minutes and
figure design is iterative, so this reads only the CSVs step 7 already wrote and
finishes in seconds. It also means the figures can be rebuilt months later from
the saved tables, with neither the trial lists nor the stimuli present.

| Figure | What it is for |
|---|---|
| `01_claim_vs_group` | **The figure.** The existence claim against the group test, at every sample size, for each responder share. The whole argument for the analysis plan in one panel set: where the two curves separate is what a mixture sample does to an averaging test. |
| `02_null_check` | The decision rule against a population where every weight is exactly zero — observed false-positive rate against the binomial prediction. Needs the `spread = 0` scenarios. |
| `03_claim_by_pattern` | The claim for each feature under each assumption about which features responders track. |
| `04_sensitivity` | The claim against N for every assumed effect weight and between-person spread — the answer to "but what if the effect is weaker than you assumed". |
| `05_evidence` | Kusztor-style: the group Bayes factor's three outcomes (effect / null / inconclusive) stacked against N. Comparable with their figure, and it shows the inconclusive band a p-value cannot express. |
| `06_conjunction` | Joint outcomes: both features claimed, only fog, only colour, neither. |
| `07_stopping` | Where a sequentially monitored study ends up, against the per-person equivalent. The argument for a single final look. |

Also written: `bf_cutoffs.csv` — the t value and observed effect size at which the
Bayes factor crosses its thresholds, per sample size, so a reader can check the
rule without re-running anything.

`PLAN_N = 14` marks the planned sample size on the figures. Not all seven belong
in a preregistration; `01`, `02` and `04` carry the argument.

---

## 6. What the design can answer

| Claim | Level | Status | Powered? |
|---|---|---|---|
| Fog determined choices in at least *k* individuals | per person | **primary** | yes — ≥ 0.98 if most respond; 0.63 at N = 14 if one in five responds |
| Saturation determined choices in at least *k* individuals | per person | **primary** | yes, but weaker than fog at every effect size |
| How many participants were responders | per person | secondary | reported as a count with a wide interval |
| The average participant was above chance | group | secondary | underpowered: 0.43 at N = 10, d = 0.5 |
| There is evidence *for* the null | group | secondary | 31% at N = 10; declared as a ceiling |
| Choices rise with difference size (bias check) | group | supporting | descriptive only — see section 7 |

The per-person route is **better powered than the group route** at these sample
sizes. At N = 10 the existence claim reaches 0.895 with half the sample
responding; the group t-test needs d ≈ 1.0 to reach 0.90. That asymmetry is the
reason the primary claim is the individual-level one.

---

## 7. What it cannot answer, and why

**1. The crossover: whether one person weighs fog more than colour.** The floors
each ask whether *one* weight differs from zero. The crossover asks whether the
*two* weights differ from each other, and the noise on a difference is larger
than the noise on either part:

```
    scatter of the difference      0.519
    critical values                -0.97 .. +0.96
    gap caught 80% of the time      1.42
```

A gap of 1.42 is larger than the whole weight of a typical strong responder. A
person can clear the fog floor, miss the colour floor, and still not clear this.
Exploratory, with intervals.

**2. Additivity, in one person.** γ is the proportional change in the combined
evidence at the largest differences: +0.5 means the two cues together carry 50%
more weight than their sum, 0 is additive. The two-weight model used everywhere
else assumes γ = 0 and so cannot test it. Within one participant:

```
   true_gamma  detected
         0.50     0.051
         1.00     0.101
         2.00     0.192
         4.00     0.508
         6.00     0.708
```

Detection never reaches 80% on a grid extending to γ = 6, a value far outside
anything plausible. **Not estimable per person at all.**

Pooled across participants it becomes answerable, because the mixed trials
accumulate:

```
   n_people  informing_trials  false_alarm  gamma_at_80%
          1               180        0.017           NaN
          5               900        0.057         1.861
         10              1800        0.110         1.229
         15              2700        0.093         1.112
```

Even pooled across ten participants the detectable γ is 1.23. And this treats
everyone as sharing one γ, so it is an **upper bound** on what a mixed-effects
fit would manage — real people differ, and that spread is not simulated.

**A technical reason the obvious predictor cannot be used.** The natural
interaction term `d_sat × d_fog` is *not antisymmetric* under the A/B side swap.
Because sides are randomised, a predictor must flip sign when the two images
exchange places, and a product of two differences does not. The script uses an
antisymmetric substitute and asserts the property at run time; without it the
additivity term would be unestimable rather than merely weak.

**3. Dose-response, in one person.** The expected curve, if the effect is real:

```
  trial_type  mag_bin  prop    sd
    sat_only        1 0.566 0.110
    sat_only        3 0.617 0.107
    fog_only        1 0.574 0.110
    fog_only        3 0.642 0.104
    together        1 0.664 0.157
    together        3 0.726 0.145
    conflict        1 0.495 0.166
    conflict        3 0.514 0.170
```

The rise from the smallest to the largest magnitude bin is +0.051 for `sat_only`,
+0.067 for `fog_only`, +0.061 for `together` and +0.019 for `conflict` — mean
+0.050, against per-person SDs of 0.10 to 0.17. **A five-point rise is not
visible in one participant.** `together` is not even monotone in expectation, and
`conflict` is essentially flat — expected, since conflict trials are scored by
fog while the two cues oppose. Dose-response is a group-level pattern and a bias
check, not a per-person test.

**4. Whether these features *drive* the experience.** The design establishes that
a participant's choices were determined by a visual property. It does not
establish that the property causes, constitutes, or is necessary to the
phenomenology. People describe derealization in metaphoric visual language, and a
reliable association between that language and a measurable image property is
what this design can deliver.

**5. How common the pattern is.** The claim is about individuals. The design
gives no estimate of prevalence in the DPDR population, and the responder share
— which drives all the power figures — is an assumption that is swept, not
measured.

**6. A clean "nobody responds" result from the mixture simulation.** The
`neither` pattern at non-zero spread is *not* a true null: weights are drawn
around zero with a spread inherited from v2, so a substantial fraction exceed the
floor by chance. This is not a bug — it is what forced the existence framing in
the first place — but a reader comparing `neither` rows against a 2.5% expectation
will be confused. The true null is `neither` **with `spread = 0.0`**, which is
the set of cells used for the verification in section 4.

---

## 8. How the number of participants changes the answer

The sample is not chosen by the power analysis — it is set by how many eligible
people the screening questionnaire returns. What the analysis provides is a
lookup table: for the N actually obtained, what can be claimed and at what rate.

**The rule and its consequence.** `k_required` is 2 up to N = 14 and 3 from
N = 15. Because the rule is discrete, **power falls when N crosses 15**:

```
share 0.2:   N=14 -> 0.630      N=15 -> 0.372      N=20 -> 0.570
share 0.5:   N=14 -> 0.978      N=15 -> 0.908      N=20 -> 0.985
share 0.1:   N=14 -> 0.378      N=15 -> 0.130      N=20 -> 0.248
```

Twenty participants do not recover what fourteen gave you at a share of 0.2 or
0.1. The mechanism: the worst study-level false-positive rate at N = 14 with
k = 2 is 0.0466, just inside 5%. At N = 15 that rule would exceed 5%, so k = 3 is
forced, which overshoots to 0.0057 — valid but conservative. **N = 14 is where
the rule is as tight as it can be while remaining honest.**

The practical implications:

- **Fourteen is a better stopping point than anything from 15 to 19.** If
  recruitment is uncertain, stopping at 14 is strictly preferable.
- **Small samples are not useless.** At a share of 0.5, six participants already
  give 0.620 and ten give 0.895.
- **Below about N = 8 the group test cannot return evidence for the null at all**
  (`p_evidence_against = 0.000` at N = 8), so the inconclusive clause will fire.
- **The responder count's interval is wide and should be stated in advance.** At
  N = 10 with every participant responding at a weight around 1, the expected
  number *called* responders is 6.5 with a 10–90 band of 4 to 8 — because the
  population has spread, so many responders sit below the detectable weight.

The colour axis is consistently harder. At a share of 1.0 and N = 10, the fog
claim succeeds 1.000 of the time and the colour claim 0.748; at a share of 0.5,
0.895 against 0.415.

---

## 9. How the number of trials changes the answer

Trials affect the per-person floor and nothing else. The group test depends on
the number of *people*, so it cannot be improved in the stimulus pipeline at all.

From step 6's sweep (haze_eq, the per-person noise floor by trials per type):

```
   n_per_type  analysed_trials  weight_sd  min_detectable_weight
           20               60     0.6089                  1.171
           30               90     0.4737                  0.942
           40              120     0.4108                  0.807
           60              180     0.3360                  0.682
           80              240     0.2843                  0.552
          100              300     0.2518                  0.503
```

The current design sits at 60 per type, 180 analysed trials. Cutting to 40 per
type would raise the floor from 0.682 to 0.807 — roughly a 20% worse floor for a
third fewer trials. Doubling to 100 per type would buy 0.503, but 300 analysed
trials plus the quality checks is a long session for this population.

**Why 180 and not fewer.** The count is set by the precision of the *individual*
estimate, because each participant is a data point. A proportion from 60 trials
of one type has a standard error near √(0.25/60) ≈ 0.065, which sits below the
expected spread between people. Below about 30 trials per type the within-person
noise dominates and individual differences — the whole reason a small sample is
worth running — become unreadable.

**If the floor is still too high for the effect you expect**, step 6's own advice,
in order of how far back it sends you: more trials per type (step 5 only),
different type proportions (step 5 only), larger image differences (steps 1 and 3,
re-render).

---

## 10. Assumptions the whole analysis rests on

Stated plainly, because every number above is conditional on them.

1. **The per-person weight distribution is carried over from v2, not measured in
   this design.** The effect weights (0.5, 1.0, 1.5), the between-person spreads
   and the weight correlations are v2's. This is the largest piece of borrowed
   scaffolding in the analysis. It is swept rather than fixed — every conclusion
   is reported per assumption in `responders.csv`, so no single guess is
   load-bearing — but the sweep's *range* is still an assumption.
2. **Each participant has a single stable referent across all 180 trials.** The
   per-person fit treats every trial as a question about the same remembered
   experience. If the referent drifts during the session, the fitted weight is
   attenuated and the floors are optimistic. The consistency trials (15 repeats
   with sides swapped, spaced at least 25 trials apart) provide a check on this,
   and they are excluded from the analysis precisely so they can serve as one.
3. **Trials are independent.** No learning, fatigue or carry-over is modelled.
4. **The logistic model with no intercept is the right form.** The design is
   symmetric, so there is no baseline preference to absorb — which is also why
   step 5 forces exactly 50/50 side assignment within each trial type rather than
   leaving it to coin flips.
5. **The responder share is unknown and swept**, not estimated. It is the single
   largest driver of the results.
6. **The floors are computed on the real trial list**, so they inherit whatever
   is true of the stimulus set. They are not generic.

---

## 11. Reading the raw tables: three traps

**1. The headline row is not what it looks like.** The headline scenario in step
7's report is `fog only` — the pattern where the true *colour* weight is zero. So
`p_claim_colour` in that row is a **false-alarm rate, not colour's power**. The
lookup table covers three patterns (`both`, `fog only`, `colour only`) for this
reason; take colour's power from a pattern in which colour is non-zero.

**2. `p_any_fog` is not the existence claim.** It is P(at least one detection),
which reaches 22% at N = 10 when nobody responds. The claim is `p_claim_fog`,
which requires `k_required` detections. The two columns sit next to each other.

**3. Superseded floors appear in earlier reports.** Any report showing
`floor, colour 0.662` and `floor, fog 0.564` used the 1.96 × SD approximation and
is superseded. The final pair is **0.735 / 0.594**.

---

## 12. Suggested wording for the preregistration

> Each participant completes 180 analysed two-alternative trials and contributes
> two weights, fitted by logistic regression without an intercept on the
> standardised within-pair differences in mean saturation and in fine detail.
> For each feature, the smallest weight distinguishable from chance in a single
> participant was established by simulating 20,000 null participants through the
> actual trial list and taking the 97.5th percentile of the recovered estimates:
> 0.735 for saturation and 0.594 for fine detail, corresponding to choosing the
> manipulated image on approximately 63% and 61% of trials at a typical
> difference. A weight above the relevant floor counts as a detection
> (one-sided, 2.5%). The primary claim for each feature is supported when at
> least k participants independently exceed that feature's floor, where k is the
> smallest count for which the probability of that many false alarms across the
> obtained sample does not exceed 5% — two for samples up to 14 participants,
> three from 15 to 20. This counting rule was verified by simulation against a
> population in which every weight is exactly zero, with observed study-level
> false-positive rates within 0.018 of the binomial prediction across all 102
> such scenarios. Detection probability for a single participant reaches 80% at
> a weight of 1.06 for saturation and 0.85 for fine detail; weaker responders
> (weight 0.6) are detected 38% and 50% of the time. Because the proportion of
> genuine responders cannot be known in advance, power is reported across
> responder shares of 1.0, 0.5, 0.2 and 0.1 and across assumed effect weights
> and between-person spreads; at a sample of 14 the fine-detail claim succeeds
> on 97.8%, 63.0% and 37.8% of simulated studies at shares of 0.5, 0.2 and 0.1
> respectively. A group-level one-sample t-test and JZS Bayes factor
> (scale 0.707) are reported for comparability with prior work and are declared
> in advance to be underpowered at this sample size: at N = 10 the t-test has
> 43% power against d = 0.5, and a study in which no effect exists has a 31%
> chance of returning evidence for the null and a 66% chance of being
> inconclusive. If the group Bayes factor falls between 1/3 and 3, the
> group-level question is reported as inconclusive and inference is restricted
> to the per-participant level. No interim analyses will be conducted: a
> sequentially monitored design was simulated and stops early for the wrong
> reason at low responder shares, concluding evidence for the null in 62% of
> studies at a share of 0.1. The within-person crossover (whether one
> participant weighs the two features differently) requires a weight gap of 1.42
> for 80% detection and the additivity of the two cues is not estimable within a
> participant at any plausible value; both are reported as exploratory with
> intervals.

---

## 13. Which files are final, and what is not included

**Several runs exist for both steps.** The decisive marker is the per-person
floor: the final runs show 0.735 / 0.594, earlier ones 0.662 / 0.564.

| Step | Final report | Distinguishing feature |
|---|---|---|
| 6 | `power_report.txt`, 417 lines | floors **0.735 / 0.594** (percentile method) |
| 7 | `power_seq_report.txt`, 510 lines | **312 scenarios** — the extra 78 against earlier runs are the `spread = 0.0` true-null cells |

Final companion tables: `lookup_by_n.csv` (69 rows), `responders.csv` (5,304
rows), `comparison.csv` (19 rows).

| Script | Reads | Writes |
|---|---|---|
| `simulate_power_v4.py` | step 5's `trial_list.csv` and step 4's `all_pairs.csv`, per set | `data/power_v4_ec/` |
| `simulate_power_seq_v4.py` | the same trial lists | `data/power_seq_v4/` |
| `plot_power_seq_v4.py` | `data/power_seq_v4/<set>/*.csv` | `data/power_seq_v4/<set>/figures/` |

All three are CPU-only and read no images. Step 6 takes about 15 seconds, step 7
about two minutes, the figures seconds.

**Not covered here:** the construction of the axes and of the stimuli (the two
companion documents); the experimental design — task wording, the word-rating
screen, the screening questionnaire, the response grid — which is described
separately; and the session-level analysis script that will apply the same model
to real data, which is not yet written. When it is, it must use the same ridge
(`1e-6`) as steps 6 and 7, or the fitted weights will not be comparable with the
floors stated above.
