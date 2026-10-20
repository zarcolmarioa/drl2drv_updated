# Rendered but not used: the grid block

From `data/stimulus_set_v4_ext_haze_eq`, built by step 3. Step 7b shows a subset of the cells step 3
renders; this is the rest, with the reason each one is not shown. **Nothing was deleted
or re-rendered** -- the images are still in the stimulus set and still carry their
measured features.

## `contrast_up` -- 141 images

Rendered as three candidate gains (1.0875, 1.110, 1.150) for a column ABOVE the original on the fog screen, and dropped. No gain was distinguishable from the original on step 3's sheets_contrast/ diagnostic, which is drawn at the full 512 px; the grid shows cells at 149-239 px, so a difference invisible at 512 px cannot appear there. The damage figures were never the binding constraint: 1.0875 does no measured damage at all on 40 of 47 scenes and is still invisible. The gain holds the mean fixed and moves only the histogram tails, which downscaling averages away, whereas haze lays a coherent veil over every pixel.

## `p65@h=0` -- 47 images

Rendered as the first colour rung above the original and dropped from the GRID only -- p65 remains in the pair trials at every fog rung. Two reasons: it leaves six columns on both grid screens, so the whole block computes one cell size; and it makes the first step on each side of the original the same size, since p75 is 25 percentile points above the original and p25 is 25 below.

## What the grid does show

- **fog**: `h=0`, `p35`, `p25`, `p15`, `p10`, `p05` (6 columns)
- **colour**: `p75`, `k=1`, `p25`, `p10`, `p05`, `p01` (6 columns)

The pair trials are unaffected by all of this: every cell named above is
`use = "grid"` in the manifest and was never in the pair inventory.
