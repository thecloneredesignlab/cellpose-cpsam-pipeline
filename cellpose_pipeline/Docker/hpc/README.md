# HPC layout

## SIF-backed counterpart

This directory is the one-to-one Apptainer counterpart of
`cellpose_pipeline/hpc/`. The original tree remains the module/conda version.
Slurm stays on the host; Python, Node/npm, `pdftoppm`, Cellpose, CUDA user-space
libraries, and bundled Cellpose models run inside the supplied SIF.

Set the runtime contract before invoking any script:

```bash
export HPC_CONTAINER_IMAGE=/absolute/readable/path/to/cellpose-runtime.sif
export HPC_CONTAINER_BINDS=/absolute/project-and-data-root
export REPORT_PLUGIN_ROOT=/absolute/path/to/data-analytics-plugin
```

`REPORT_PLUGIN_ROOT` is required only by report workflows. Those workers append
that exact directory to the bind list. Do not bind a whole home directory.

The runtime uses `apptainer exec --cleanenv`, a temporary container home/cache,
the project root, and the explicit comma-separated bind list. Slurm variables
and thread controls are forwarded through a narrow allowlist. GPU workers add
`--nv` automatically when Slurm exposes an allocated GPU; set
`HPC_CONTAINER_GPU=1` or `HPC_CONTAINER_GPU=0` only for a deliberate manual
override.

Preview supported submission workflows without calling `sbatch`, for example:

```bash
DRY_RUN_SUBMIT=1 \
  bash cellpose_pipeline/Docker/hpc/submit_classification_only_full.sh
```

The two target-only preview adapters in
`Parameter_calibration/20_submit_high_density_bf_combined_optimization.sh` and
`analysisi/04_submit_postsegmentation_analysis_chain.sh` also accept
`DRY_RUN_SUBMIT=1`.

Runtime command requirements are recorded in `sif_runtime_commands.tsv`.
`SIF_PROVENANCE.md` records the verified immutable SIF checksum without
committing its deployment path.

`submit_broad_phenotype_shadow_full.sh` is the SIF-backed broad-morphology
shadow entry point. It consumes frozen Stage 06 outputs without invoking
segmentation or changing the existing viability/trajectory products. Phase A
stops after annotation HTML generation; later human-reviewed stages are
resumed explicitly. The complete write, model-identity, sharded-inference and
scientific-interpretation contract is documented in
[`docs/broad_phenotype_shadow_workflow.md`](../../../docs/broad_phenotype_shadow_workflow.md).
Formal stages default to the 12-hour `xxlarge` contract, and the entry point
checks the live QOS `MaxWall` before any Slurm submission.
Workers execute through `sbatch --wrap` from a hash-verified node-local
extraction of the run's frozen tracked-code archive. The extraction is mounted
read-only at the canonical snapshot path inside the SIF, avoiding Slurm-spool,
mutable-checkout, and managed shared-filesystem mode-bit drift.

`submit_reference_cell_state_shadow.sh` is the separate historical-reference
entry point. It imports the frozen development representative universe into a
new `reference_cell_state_shadow_<timestamp>` root, recomputes a nine-feature
UMAP, builds a BF/Nuclei morphology-reference workspace, and stops before the
human region submission. The current classification root and Dead channel are
not container binds. Its compute-node calibration entry is
`Parameter_calibration/30_run_reference_cell_state_shadow_test.sh`. The full
contract is in
[`docs/reference_cell_state_shadow_workflow.md`](../../../docs/reference_cell_state_shadow_workflow.md).

`submit_reference_cell_state_shadow_v2.sh` adds the V2 historical projection,
exact-review, historical-model and sharded-prediction chain. Stable-polygon
runs use pinned reference sampling. A no-stable-cluster run instead records
`historical_core_parity_with_disclosed_no_stable_cluster_sampling_adaptation`;
its well-balanced Seed1 and sampling-only Seed2 surrogate are disclosed V2
adaptations and are never claimed as historical-production reproduction.
Each formal V2 stage keeps its original immutable submission ledger plus an
append-only attempt ledger. On an explicit resume, the frozen-archive
submitter queries the latest allocation with `sacct`: PENDING/RUNNING jobs are
reported without duplication, COMPLETED jobs are reused only after full
product/hash validation, and FAILED/CANCELLED/TIMEOUT-class states create a
new attempt against the latest upstream job. A COMPLETED job with absent or
changed products fails closed. Prediction recovery may resubmit the frozen
full array (`%64`); already-published shard generations are independently
verified and reused by each task before the new finalize attempt runs. The
initial ledger and original summary are never deleted or rewritten.

Before a new formal polygon run, the independent expanded-labelability
calibration entry is
`Parameter_calibration/32_run_reference_cell_state_expanded_projection_test.sh`.
It is direct-only on `hpctpa3pc0009`, writes only beneath
`Tests_and_Parameters_calibration`, compares shape9/classifier12/expanded39 on
the same cells, and stops for morphology-evidence review. It cannot submit a
formal job or put expanded39 features into the historical classifier.

The dataset-adapted successor is
`Parameter_calibration/33_run_multimodal_cell_state_v3_test.sh`. It abandons
reference-parity as an objective, uses zero-aware Nuclei evidence and equal
Shape/Brightfield/Nuclei distance contributions, and stops at a 500-cell blind
image-review barrier. It is direct-only on `hpctpa3pc0009`; no V3 formal Slurm
submission is permitted until that calibration and human-label gate pass.

The continuation trust boundary is the run-owned code archive, archive
receipt, and shared extracted snapshot. All three must agree. The public
caller cannot assert the private frozen-reexec flag: a one-use, owner-private
token binds the verified archive, canonical V2 root, extracted project, and
submitter before any frozen child can run. This detects checkout drift and
single-artifact tampering; it does not protect against an actor who can
coherently replace every run-owned trust artifact under the same Unix account.

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
the d0 improvement report. Report generation uses Python and Node.js from the
SIF plus the explicitly bound Data Analytics report package.

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
