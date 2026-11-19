FROZEN TRIAL LISTS  v5c_180  (180 analysed trials per block)
====================================================================

USE
  Copy package/ to the experiment as data/sets/v5c_180/. Verify
  checksums.sha256 before any analysis (sha256sum -c checksums.sha256,
  from this folder). package/README.txt describes the package.

WHAT IS HERE
  package/      the experiment package, byte-identical to script 4's output
                (data/trial_orders_v5c/v5c_n180_orders4_gap30-fixed_break40_slots64_seed42__s14169caa/package)
  provenance/   the builds' records: script 3 (data/trials_v5c/v5c_n180_balanced_mixed_final_r3_cand40_leak0.05_seed42__s38fa29a2, without its images),
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
  analyse_pair_capacity_v5c.py  d12a047011880dc71c5d0f6e741a28b87a636c34df39fd76722fb5ddff19baaf
  build_trials_v5c.py  fd78ce4c912c7081649cd851f28a3fb795534dc75c9043e69bb844348c0d84cf
  make_trial_orders_v5c.py  e2faf36d33d789dbad33a43018b24e1e0e24e09fe343c0cb5418b35048d3cf46
  simulate_power_v4_n10_20.py  852f3b33eb349153fbff9f95c50e1cd3c4f84f68b204d79a23f020f471bd1730
  freeze_trial_lists_v5c.py  4ab4f787710198f2346e4090262c17ae7950c64a736f48b0f614c49bd1b62018
