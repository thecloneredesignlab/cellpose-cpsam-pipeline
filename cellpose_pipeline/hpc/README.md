# HPC layout

`submit_full_fusion_production.sh` is the only public full-production entry
point. It calls `orchestrate_cellpose_cpsam_full_array.sh`, which constructs the
Slurm dependency graph and submits the production workers retained in this
directory.

- `Parameter_calibration/`: numbered tuning, test, validation, and legacy-run
  launchers;
- `analysisi/`: numbered downstream analysis and partial-analysis launchers.

Production workers stay at the top level because they are direct dependencies
of the orchestrator. No script in the two subdirectories is required by the
default full-production dependency graph.
