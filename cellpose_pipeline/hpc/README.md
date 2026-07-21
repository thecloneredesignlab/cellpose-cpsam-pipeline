# HPC layout

`submit_full_fusion_production.sh` is the public full-production entry point. It
calls `orchestrate_cellpose_cpsam_full_array.sh`, which constructs the Slurm
dependency graph and submits the production workers retained in this directory.

`submit_classification_only_full.sh` is the public full-time-course
classification-only entry point. It treats the existing Combined, Brightfield,
Dead, Nuclei, nucleus-core, and nucleated-only segmentation masks as immutable.
It creates a new `results/classification_<timestamp>/` root and submits only:

```text
post-segmentation manifest
├── original classification array ── original merge
└── nucleated-only classification array ── nucleated-only merge
```

Both arrays call the production `08_fuse_multichannel_classification.py`, whose
defaults contain the final d0-calibrated object-aware method. No `--timepoint`
filter is passed, so every field in the complete experiment is processed. The
default array resources match the classification stage of full production: one
CPU, 4 GB, 12 hours per field, `xxlarge`, and no GPU. The array has no explicit
concurrency throttle, so Slurm controls how many tasks run simultaneously. Each
merge uses one CPU, 8 GB, and 12 hours.

After both merges complete, a final CPU job runs
`scripts/analysisi/04_plot_well_counts_over_time.py` for both classification
branches. It writes the plate-layout live/dead time-course PNG and PDF plus the
per-image and per-well/time CSV tables under
`analysis/well_count_timecourses/{fusion,fusion-nucleated-only}/` in the new
result root.

- `Parameter_calibration/`: numbered tuning, test, validation, and legacy-run
  launchers;
- `analysisi/`: numbered downstream analysis and partial-analysis launchers.

Production workers stay at the top level because they are direct dependencies
of the orchestrator. No script in the two subdirectories is required by the
default full-production dependency graph.
