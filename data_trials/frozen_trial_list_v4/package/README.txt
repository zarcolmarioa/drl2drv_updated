EXPERIMENT PACKAGE  (both)
========================================================

  trials        211 (180 analysed)
  images        349
  format        png {'format': 'PNG'}
  total size    105.8 MB
  re-measured   True
  stimuli from  data/stimulus_set_v4_ext_haze_eq/stimuli

FILES
  trial_list.csv   one row per trial, in order
  trial_list.json  the same trials as an array of
                   objects, for the experiment to
                   load. No `instruction` column --
                   keep the question text in code.
                   null means 'not applicable'.
  stimuli/         the images, referenced by
                   left_path and right_path
  manifest.csv     every image with its measured
                   sat_mean and contrast_local
  preload.json     the image paths to fetch before
                   the first trial. FETCH EXACTLY
                   THIS LIST. In 'no_images' catch
                   mode it omits images used only by
                   catch rows, which ship in stimuli/
                   but are never shown.
                   before the first trial

TRIAL LIST
  trial        presentation order
  trial_type   sat_only / fog_only / together /
               conflict / consistency / catch /
               practice
  analysed     True ONLY for the four real types.
               Practice, catch and consistency are
               quality checks, excluded from every
               analysis.
  instruction  the question to show. Catch trials
               carry a different one.
  left_path    image for the left position
  right_path   image for the right position
  a_is_left    whether the left image is image A.
               The analysis needs it to map a side
               to a choice, because d_sat and d_fog
               are always A minus B.
  pred_sat     THE REGISTERED MODEL'S COLOUR
  pred_fog     AND FOG PREDICTORS. See below.
  d_sat,d_fog  raw signed differences, in feature
               units, always A minus B
  z_sat,z_fog  the same differences normalised, as
               step 4 computed them. Reported for
               reference; the model fits pred_sat
               and pred_fog.
  pred_sat_range, pred_fog_range
               the predictors divided by each
               scene's own feature range instead
  mag_bin      magnitude bin, for the
               dose-response analysis
  catch_mode   catch trials only: which kind of
               catch trial this list expects, either
               'no_images' (an arrow key named by an
               instruction, no pictures shown) or
               'images' (the pair shown, a side to
               click). THE EXPERIMENT MUST CHECK
               THIS AGAINST ITS OWN SETTING AND STOP
               IF THEY DISAGREE. There is no
               correct_response column: the expected
               response lives in the experiment, with
               the instruction text.
  repeat_of    consistency trials only: the trial
               this one repeats


THE PREDICTORS
  pred_sat and pred_fog are what the registered
  model fits. Two things about them matter:

  UNITS. Both are in CORPUS-SD UNITS -- the raw
  difference over that feature's corpus SD (sat_mean
  39.182, contrast_local 9.348, from the 20,000-image
  stimulus-path reference). A coefficient is then
  'per SD of the feature', and the two are on one
  scale and directly comparable.

  SIGN. POSITIVE MEANS IMAGE A IS THE PREDICTED
  CHOICE -- A is the less colourful image
  (pred_sat) or the foggier one (pred_fog). So a
  POSITIVE coefficient means the participant chose
  as predicted. Note this is NOT simply d/SD:
  pred_sat is the NEGATED d_sat, because d_sat is
  positive when A is MORE colourful. pred_fog is
  d_fog unchanged, because step 4 already applied
  the sign there. Negating both would flip the fog
  coefficient and nothing would announce it.

RECORDING RESPONSES
  Record which SIDE was clicked (left/right) and the
  reaction time. Combine with a_is_left to recover
  which image was chosen.
