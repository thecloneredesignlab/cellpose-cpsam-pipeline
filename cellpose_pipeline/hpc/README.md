# HPC layout

`submit_full_fusion_production.sh` is the public full-production entry point. It
calls `orchestrate_cellpose_cpsam_full_array.sh`, which constructs the Slurm
dependency graph and submits the production workers retained in this directory.

`submit_classification_only_full.sh` is the public full-time-course
classification-only entry point. It treats the existing Combined, Brightfield,
Dead, Nuclei, nucleus-core, and nucleated-only segmentation masks as immutable.

`submit_broad_phenotype_shadow_full.sh` creates a sibling
`results/broad_phenotype_shadow_<timestamp>/` workflow without changing those
segmentation masks or the current viability/trajectory outputs. Its first DAG
ends at region-annotation HTML and cannot cross the human-review barrier. The
host entry points are thin counterparts of the SIF-backed implementations under
`cellpose_pipeline/Docker/hpc/`; all scientific Python and R work uses the
pinned SIF. See the
[broad-phenotype shadow workflow](../../docs/broad_phenotype_shadow_workflow.md)
for its object, feature, model, sharded-inference and promotion contracts.
Formal CPU stages default to the `xxlarge` QOS with requests no longer than
12 hours. The submitter reads the live QOS `MaxWall` and rejects an incompatible
request before creating or submitting any job.
It also embeds the frozen archive hash in every `sbatch --wrap`, verifies and
extracts the tracked code in node-local temporary storage, and maps that source
read-only to the canonical snapshot path inside the SIF. Queued work is
therefore independent of later pulls and of writable managed shared-filesystem
mode bits.

`Parameter_calibration/29_run_broad_phenotype_shadow_test.sh` is restricted to
`hpctpa3pc0009` and writes only below
`results/Tests_and_Parameters_calibration/`.

`submit_reference_cell_state_shadow.sh` creates an independent historical
three-class reference axis under
`results/reference_cell_state_shadow_<timestamp>/`. It reuses only the frozen
development representative universe, recomputes UMAP from nine promoted-shape
features, builds the BF/Nuclei morphology workspace, and stops at the human
region-submission barrier. It neither binds nor writes the current
classification root. The calibration counterpart
`Parameter_calibration/30_run_reference_cell_state_shadow_test.sh` is hard
restricted to `hpctpa3pc0009` and the calibration result tree. See the
[reference cell-state workflow](../../docs/reference_cell_state_shadow_workflow.md).

`submit_reference_cell_state_shadow_v2.sh` is the historical-core-parity
replacement.
It adds historical existing-UMAP projection, authoritative diagnostic DBSCAN,
pinned representative selection on the stable-polygon branch, two exact
manual-review barriers, the
12-feature historical grouped glmnet model, sharded prediction, and optional
post-freeze comparison. V1 roots remain audit-only. V2 calibration uses
`Parameter_calibration/31_run_reference_cell_state_shadow_v2_test.sh` directly
on `hpctpa3pc0009`; the V2 entry is locked to the separately versioned,
66-package, A30-verified parity SIF. If DBSCAN has no stable
cluster, receipts explicitly disclose the user-approved Seed1/Seed2 sampling
adaptation; that branch is not described as exact historical sampling.
It creates a new `results/classification_<timestamp>/` root and submits only:

```text
post-segmentation manifest
├── original classification array ── original merge
└── nucleated-only classification array ── nucleated-only merge
                                      │
                                      └── late-death preparation
                                                        │
                                                        └── 80-well refinement array
                                                                       │
                                                                       └── refinement finalize
                                                                                      │
                                                                                      └── well-count plots
                                                                                                │
                                                                                                └── dose-response analysis
                                                                                                          │
                                                                                                          └── full-cohort HTML report
```

Both arrays call the production `08_fuse_multichannel_classification.py`, whose
defaults contain the final d0-calibrated object-aware method. After both merged
branches exist, `run_late_dead_trajectory_prepare.sh` calibrates density from
the run's 320 d0 fields and freezes the shared reference state. The unthrottled
`run_late_dead_trajectory_well_array_task.sh` array assigns one well, including
both segmentation views, to each task. It identifies persistent multi-site late
field collapse and rescues eligible live-labelled objects only when at least two
independent morphology/red-mass signals support death and strong live evidence
is absent. `run_late_dead_trajectory_finalize.sh` requires successful receipts
from every well before publishing merged summaries and the production GO/NO-GO.
The refinement updates both classification branches, per-field/merged summaries,
rescued-object masks, annotations, and affected QC overlays without changing
any segmentation mask.

No `--timepoint` filter is passed, so every field in the complete experiment is
processed. The default array resources match the classification stage of full
production: one CPU, 4 GB, 12 hours per field, `xxlarge`, and no GPU. The array
has no explicit concurrency throttle, so Slurm controls how many tasks run
simultaneously. Each merge uses one CPU, 8 GB, and 12 hours. The late-death
preparation uses 32 CPUs, 256 GB, and 12 hours. Each of the 80 well tasks uses
one CPU, 48 GB, and four hours; there is no explicit array throttle. Finalize
uses one CPU, 32 GB, and six hours. All stages use `xxlarge` and no GPU. A failed
well writes its traceback immediately and can be retried by array index; the
finalize job runs after the array settles but only succeeds when all 80 well
receipts and all 54,400 field rows pass their invariants.

After refinement completes, a final CPU job runs
`scripts/analysisi/04_plot_well_counts_over_time.py` for both classification
branches. It writes the plate-layout live/dead time-course PNG and PDF plus the
per-image and per-well/time CSV tables under
`analysis/well_count_timecourses/{fusion,fusion-nucleated-only}/` in the new
result root.

The next dependent CPU job rebuilds all 82 dose-response outputs under
`analysis/dose_response/`: AUC, exact Day-4, exact Day-5, growth-rate
inhibition, and excess-lethal-fraction analyses for both branches. The final
job validates the classification and refinement row counts, the zero-failure
ledger, the 85-time-point contract, the 82-file dose-response inventory, and
the frozen calibration configuration before writing:

```text
analysis/reports/
├── DEAD_CLASSIFICATION_FULL_COHORT_REPORT.html
├── DEAD_CLASSIFICATION_FULL_COHORT_REPORT.artifact.json
└── DEAD_CLASSIFICATION_FULL_COHORT_REPORT.build.json
```

The HTML is self-contained and uses the same navigation, vertically aligned
before/final QC panels, high-resolution lightbox, zoom, and drag behavior as
the d0 improvement report. Report generation requires the CellPose Python
environment, the locally installed Node.js runtime, and the Data Analytics
report package recorded in the worker.

- `Parameter_calibration/`: numbered tuning, test, validation, and legacy-run
  launchers;
- `analysisi/`: numbered downstream analysis and partial-analysis launchers.

Production workers stay at the top level because they are direct dependencies
of the orchestrator. The numbered production modules include
`13_build_late_dead_trajectory_dataset.py` and
`14_apply_late_dead_trajectory_refinement.py`; the parameter-calibration
wrappers call the same shared implementation so tuning and production cannot
silently diverge. No script in the two subdirectories is required by the
default segmentation-to-classification production dependency graph. The
classification-only entry point explicitly includes analysis workers
`analysisi/05_run_well_count_timecourse_plots.sh`,
`analysisi/06_run_dose_response_analysis.sh`, and
`analysisi/07_run_full_classification_report.sh`.
