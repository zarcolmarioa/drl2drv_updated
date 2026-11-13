FROZEN TRIAL LISTS  v5b_180  (180 analysed trials per block)
====================================================================

USE
  Copy package/ to the experiment as data/sets/v5b_180/. Verify
  checksums.sha256 before any analysis (sha256sum -c checksums.sha256,
  from this folder). package/README.txt describes the package.

WHAT IS HERE
  package/      the experiment package, byte-identical to script 4's output
                (data/trial_orders_v5b/v5b_n180_orders4_gap30-fixed_break40_slots64_seed42__s20227fe1/package)
  provenance/   the builds' records: script 3 (data/trials_v5b/v5b_n180_balanced_mixed_v5units_cand40_leak0.05_seed42__s262763fa, without its images),
                script 4 (without its package), the rebuild comparison and
                commands, and this freeze's checks
  README.txt    this file
  checksums.sha256, FROZEN.json, freeze_report.txt

STIMULUS SET
  stimulus_set_v5/7fd72262ba (folder data/stimulus_set_v5/7fd72262ba)
  SHA-256 of its checksums.sha256: 10d969db80459817b14c3dfb2b2da4d5aeb680e1f505b18df4b15f7865a7e075
  manifest SHA-256: 69f0529f9f539ad25e830806517f8f759389b3a5c9011b31e8ede5596727cfa6
  Every image in package/stimuli/ is checked against that checksums file.

IMAGES AND ARCHIVING
  The images are edited versions of photographs from Places365, whose
  copyright belongs to the images' owners. This package can be archived
  and cited, but the images should not be redistributed publicly until
  their licence is confirmed. The stimulus set's sources.csv (the
  Places365 file of every scene) and its checksums allow anyone with
  Places365 to rebuild the images and verify them.

SCRIPTS (SHA-256)
  analyse_pair_capacity_v5b.py  f7ab2c91e5a0be23e1d0f08ff96876d785487633c0dadff2781cf16826f45df4
  build_trials_v5b.py  b6d597006d540ace5dd86a191d860263966aec1d26350847b6ca51de752c45a8
  make_trial_orders_v5b.py  ecf4877835738b8aca8134fb36ae4647362d6568e1a753726d7bb531cf1e135e
  simulate_power_v4_n10_20.py  852f3b33eb349153fbff9f95c50e1cd3c4f84f68b204d79a23f020f471bd1730
  freeze_trial_lists_v5b.py  1b2f4a4775b1da65e18c4e3b36a6265d4a90c528d21de85454f093a053b147fc
