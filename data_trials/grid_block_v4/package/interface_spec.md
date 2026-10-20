# Grid block: interface spec

Package `v4-grid-1`, built 2026-10-19 12:38:17 from `data/stimulus_set_v4_ext_haze_eq`.

The grid block is **descriptive only**. Nothing recorded here enters a registered
test; the registered analyses use the pair trials alone.

## The two blocks

| block | when | question | 'none of these' | free text |
|---|---|---|---|---|
| episode | after the pair trials | which column looks most like your surroundings looked during the episode | yes | yes, when none is chosen |
| normal | at the end of the session | which column looks most like your surroundings normally look | no | no |

The normal block has no `none` button on purpose: "my surroundings normally look
like none of these" is not an answer the normal-to-episode shift can use, and an
empty normal answer loses the whole pair.

## The set swap

`set_assignment.rule` is `random_recorded`.

At session start the client draws `code` from {0, 1} with probability 0.5 each,
records it, and uses it for the whole session:

```
code 0 -> episode block shows set A, normal block shows set B
code 1 -> episode block shows set B, normal block shows set A
```

No server state is needed, which this study does not have. The counterbalance is
therefore **binomial, not exact**: at 60 participants the split is 30 +/- 4 about two
thirds of the time. The code must be in the data file for every participant, or
the imbalance cannot be checked or adjusted for.

## Rendering a screen

For the chosen set, take `sets[<set>]` and present its screens in `screen` order.
For each screen:

- `columns` is the layout **left to right as displayed**. `column_order` says whether
  that is the forward or the reversed arrangement; the order in the file is already
  the display order, so do not reverse it again.
- each entry of `rows` has `images` in the **same order as `columns`**. Build the grid
  from those files; nothing is pre-composited.
- cell size: `floor(min(byWidth, byHeight, 240))`, at least 100, with
  `byWidth = (innerWidth - 48 - (cols-1)*22)/cols - 14` and
  `byHeight = (innerHeight - 190 - (rows-1)*6)/rows`.
- the participant picks a **column**, not a cell. A click anywhere in a column
  selects that column.
- the cell size is ONE size for the whole block, computed from the WIDEST screen
  in the file (6 columns here). Do not size each screen by its own column
  count.

## What to record, per screen

| field | what |
|---|---|
| `block` | `episode` or `normal` |
| `set_code` | the `code` drawn for this participant |
| `scene_set` | `A` or `B`, the set shown on this screen |
| `screen` | the screen number from the file |
| `feature` | `fog` or `colour` |
| `column_order` | `forward` or `reversed`, as shipped |
| `scene_ids` | the three scenes, in row order |
| `chosen_display_column` | 1-based position of the chosen column AS DISPLAYED |
| `chosen_level_index` | the `level_index` of that column: 0 is the scene's own original, negative is more than normal, positive is less |
| `chose_none` | true when 'none of these' was used (episode block only) |
| `free_text` | the typed answer, when `none` was chosen |
| `rt_ms` | from screen onset to the confirming click |
| `cell_px` | the cell size the layout computed, which varies by window and bears on whether the end columns were distinguishable |
| `inner_w, inner_h` | the window at the time of the screen |

Record **both** `chosen_display_column` and `chosen_level_index`. The position moves
when the column order reverses and the index does not, so storing only one of them
makes the reversed screens unreadable without the package in hand.

Do **not** copy the level parameters, the chroma, the haze or the measured features
into the data file. They are in `grid_trials.csv`, keyed by set, screen, row and
column, and a second copy is a second thing that can disagree.

## Analysis note

The normal-to-episode shift is computed in **feature units**, not in grid steps:
the contrast-up step is about +0.22 SD and the fog steps are -0.13 SD and larger,
so a step is not a constant amount of anything. `grid_trials.csv` carries
`sd_from_original` per cell for exactly this.

The original sits at position 1 on the fog screen and 2 on the colour
screen, so comparing *which column* people choose across the two features is partly
a position effect. The shift within each feature is unaffected.

### The fog axis is censored at the original

The fog screen has **no column clearer than the photograph as taken**. A
participant whose vision is clearer than that has to answer on the original
column, and in the normal-appearance block that is the common case rather than
the edge case -- most people's normal vision is not foggy.

So a fog answer on the original column is a **lower bound** on that person's
normal, and the normal-to-episode shift computed from it is a lower bound on the
shift. The bias is conservative; it can only shrink the shift. It is declared
rather than absorbed, and **the count of fog answers on the original column in
each block is reported alongside the shift**.

This is why `chosen_level_index` must be recorded: the censored answers are
exactly `level_index == 0` on a `fog` screen, and they cannot be identified from
the display column alone once the column order reverses.
