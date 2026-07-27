# HPC analysis launchers

These are the SIF-backed counterparts under
`cellpose_pipeline/Docker/hpc/analysisi/`. They preserve the source launchers'
Slurm resources and result contracts while routing runtime tools through the
shared Apptainer layer.

These numbered scripts run downstream or partial analyses against existing
segmentation results. The full production workflow continues to use the worker
scripts in the parent `hpc/` directory.

- `05_run_well_count_timecourse_plots.sh` builds strict-completeness well/time
  tables and plate-layout count plots for both classification branches.
- `06_run_dose_response_analysis.sh` consumes those tables and atomically
  rebuilds the complete 123-file AUC, Day-4, Day-5, GR, and excess-lethal-
  fraction output tree for the original, nucleated-only, and authoritative
  fusion-consensus branches.
- `07_run_full_classification_report.sh` validates production completeness and
  frozen configuration provenance, then packages the self-contained full-
  cohort HTML, canonical artifact JSON, and build receipt.
- `08_submit_classification_reports_backfill.sh` submits the calibration
  report and full-cohort dose-response rebuild in parallel, then starts the
  full-cohort report only after both succeed. Its defaults target the completed
  historical `classification_20260723_101944` production result and its
  historical `late_dead_trajectory_optimization_20260723_024151` calibration.
  New production submissions use
  `death_classification_consensus_optimization_20260725_213646`.

In `submit_classification_only_full.sh`, these jobs run in the order
`well counts -> dose response -> full report`; each starts only after the
preceding job succeeds.
