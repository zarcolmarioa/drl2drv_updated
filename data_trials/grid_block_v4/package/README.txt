GRID BLOCK PACKAGE  (v4, step 7b)
==============================================================

  version      v4-grid-1
  built        2026-10-19 12:38:17
  from         data/stimulus_set_v4_ext_haze_eq
  blocks       2 (episode, normal)
  screens      4 per block, 3 scenes each
  scene sets   2 x 12, swapped between participants
  images       144, 512x512, 43.7 MB

DESCRIPTIVE ONLY
  The grid block asks WHERE on the manipulated range an
  experience sits. It is not a test of the hypothesis: the
  registered analyses use the pair trials alone.

THE TWO BLOCKS
  episode   with 'none of these', free text when none
  normal    no 'none of these'

  A participant sees ONE set in the episode block and the
  OTHER in the normal block, so no photograph appears in
  both. Which way round is the set assignment, below.

FILES
  grid_trials.json  the layout. `sets` has one entry per
                    scene set; each screen lists `columns`
                    left to right and `rows`, whose
                    `images` are in the same order as the
                    columns.
  grid_trials.csv   one row per cell, with the measured
                    features and the distance from that
                    scene's own original, in corpus SD.
  manifest.csv      every shipped image, its size and its
                    SHA-256.
  preload.json      the image paths, for fetching up front.
  interface_spec.md what the experiment must render and
                    record.

CENSORING, for whoever analyses this
  The fog screen has NO column clearer than the
  photograph as taken. A participant whose vision is
  clearer answers on the original column, which in the
  normal block is the common case. Those answers are
  LOWER BOUNDS, and so is any shift computed from
  them. Report the count of fog answers on the
  original column in each block.

DISPLAY
  Cells are capped at 240 px and the layout
  computes the real size from the window; at seven
  columns and three rows that is about 149 px on a
  1366-wide window and about 234 px on a 1920-wide one.
  Local contrast is fine detail, so do not shrink further:
  100 px is the floor, and hitting it means the
  layout no longer fits.
