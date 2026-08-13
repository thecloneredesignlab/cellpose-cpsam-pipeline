# Reference cell-state shadow workflow V2

## Scope

V2 produces a historical three-class cell-state axis (`live_cell`,
`dead_cell`, `multinucleated_cell`) without changing the current
viability/trajectory classifier. V1 results and entry points remain audit
records; no V2 command may resume, overwrite, or reinterpret a V1 root.

Formal roots are new siblings named
`results/reference_cell_state_shadow_v2_<timestamp>/`. Calibration roots are
new children of `results/Tests_and_Parameters_calibration/` named
`reference_cell_state_shadow_v2_test_<timestamp>/`.

## What “reference-consistent” means

Code-level parity requires all of the following:

- the pinned Cell Phenotype Annotator checkout and selected source-function
  hashes;
- the historical preprocessing, PCA/UMAP, DBSCAN optimizer and fallback;
- the pinned `get_all_cell_lines_overlay_representatives` and
  `get_spatially_uniform_representatives` calls with seed 1;
- pinned Seed1/Seed2 sampling only when stable polygon regions exist;
- exact manual-evidence import and conflict adjudication on both branches;
- the historical 12-feature multinomial glmnet classifier, grouped by well,
  with site plus elapsed time nested metadata and 5-by-5 grouped CV;
- the frozen SUM-159 plate identity mapping and dependency/runtime locks.

Data-result acceptance additionally requires both human-review barriers, a
final accepted model and exact full-universe shard coverage. A stable-polygon
run records `historical_core_parity_with_pinned_stable_polygon_sampling`.
When no stable cluster exists, the run instead records
`historical_core_parity_with_disclosed_no_stable_cluster_sampling_adaptation`:
the historical projection/classifier core remains pinned, but Seed1's
all-unassigned 500-cell design and Seed2's sampling-only pseudo-multi
probability surrogate are explicit V2 adaptations, not exact reference-repo
sampling or a historical-production reproduction claim.

Pixel calibration is unavailable and is never inferred. Current pixel-unit
features are adapted to historical names only; V2 makes no physical-micron
parity claim.

## Phase A and diagnostic-cluster branch

The formal entry point is:

```bash
V2_STAGE=phase-a \
  bash cellpose_pipeline/hpc/submit_reference_cell_state_shadow_v2.sh
```

Phase A imports the frozen 32,000-cell development universe, derives the
canonical SUM-159 plate identities, runs historical projection, imports its
existing coordinates into CPA, creates the diagnostic cluster overlay and
renders BF/Nuclei evidence using Combined masks for object localization.
Diagnostic clusters are metadata only and are never classifier targets.

The optimized DBSCAN outcome selects one of two human paths:

```text
valid multi-cluster candidate -> polygon region barrier
one-cluster fallback          -> adapted all-unassigned Seed1 exact review barrier
```

The one-cluster fallback is the reference optimizer's fail-closed diagnostic
outcome. V2 does not force a visually pleasing cluster. With user confirmation
it can still finish an independent classification when the UMAP has no stable
clusters: Seed1 becomes a blinded, well-balanced sample of at most 500 cells,
at most 8 per well, and Seed2 uses an initial-model probability surrogate only
to populate review buckets. Neither adapted sampling step is called pinned
reference-exact, and no surrogate label enters training.

## Human and model stages

For a multi-cluster outcome, export the polygon submission and run
`V2_STAGE=post-region`. This imports the exact polygon result, selects the
historical Seed1 set, and renders the exact BF/Nuclei review HTML. For a
one-cluster fallback, Phase A already renders Seed1.

After Seed1 manual submission, run `V2_STAGE=post-seed1` for the polygon path
or `V2_STAGE=post-fallback-seed1` for the fallback path. The worker imports the
exact selected universe without CPA resampling, trains the initial model,
selects the fixed Seed2 targeted buckets and renders the exact Seed2 review.

After Seed2 manual submission, run `V2_STAGE=post-seed2`. Any label conflict
stops at `HUMAN_ADJUDICATION_REQUIRED` and atomically freezes
`human_review/merged_adjudication_barrier/`. Complete an adjudication TSV
inside the V2 root, then run the distinct `V2_STAGE=post-adjudication` with
`REVIEW_ADJUDICATION=/absolute/path.tsv`. Before either merge, script 38 fully
revalidates both immutable review imports against the exact selection,
render/crop, submission, and output hashes. The adjudicated merge still
publishes atomically to `human_review/merged/`; it never overwrites the barrier.
A successful merge trains the final historical model.

All review pages show BF and Nuclei evidence and use Combined segmentation
masks only for localization/cropping. UMAP/cluster/default labels are never
accepted as manual training evidence.

## Acceptance, prediction and comparison

`V2_STAGE=predict` submits:

```text
final-model acceptance -> original-feature array (1-N%64) -> exact merge
```

The merged independent axis is
`predictions/reference_cell_state_predictions.tsv`, with exact columns
`model_id`, `cell_id`, `reference_cell_state_class_id`, and
`prediction_status`. Its receipt is
`predictions/REFERENCE_CELL_STATE_SHADOW_GO_NO_GO.json`. Each array task first
atomically installs one co-located generation under
`prediction_shards_v2/shards/<well>/<key>__original/`, containing the wide TSV
and `prediction_receipt.json`; probability columns retain the historical
`live_cell`, `dead_cell`, `multinucleated_cell` order.

Only after this result is frozen may `V2_STAGE=compare` bind a current
classification root. Comparison writes a third independent root and is a
descriptive cross-classification, not an accuracy table. No comparison result
feeds back into the V2 model.

Formal continuation always reuses the same V2 root and stage name. The frozen
submitter serializes concurrent orchestrators and queries the latest attempt
with `sacct`:

- PENDING/RUNNING-family states are reported without another submission;
- COMPLETED is reusable only after the stage's authoritative receipts and
  artifact hashes pass a fresh audit;
- FAILED/CANCELLED/TIMEOUT and other classified terminal failures append a new
  attempt while preserving the original ledger and summary;
- missing/ambiguous/unknown scheduler state, or COMPLETED with missing output,
  fails closed.

For the prediction DAG, each new node depends on the latest upstream attempt.
If an old downstream job remains active on a failed upstream dependency, V2
waits for that job to become terminal instead of cancelling or duplicating it.
Then a replacement full array is safe: immutable completed shards verify and
reuse themselves, failed/missing shards publish new generations, and a new
finalize attempt binds the replacement array job. Attempt history is retained
in `workflow_status/submission_<stage>_attempts_v2.tsv`.

## HPC and container contract

Formal jobs use Slurm without a node, constraint, exclusion or GPU request.
Every submission runs through `/usr/bin/env -i`; ambient `SBATCH_*` variables
cannot alter placement. Workers verify the full SIF SHA and its read-only
SquashFS root, execute a node-local archive copy, hide `/share`, then restore
only explicit binds. Before model freeze, Dead, Combined RGB and current
classification results remain invisible. Legacy/current `NO_GO` is not read;
all V2-specific gates remain mandatory.

The final parity image is versioned separately as
`cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models-reference-v2-parity.sif`.
It locks 66 R binaries, including the real dplyr/tidyr/purrr/stringr/ggplot2
namespaces required by selected reference functions. The identity file is
activated only after the image was built from commit
`c6414c6d49046e820c1f6971f45aa405428af0fe`, converted from registry digest
`sha256:a27b0e4367d89ec9f012a2523f800df7b3cceaf095fcdcb395039d59297b2d74`,
and verified on `hpctpa3pc0009`. Its frozen SIF SHA-256 is
`b3fda3cf5de4934c7533471f99b5d137d6f0431aa9dfa145b5da27d9e1687752`
for 6,932,799,488 bytes. Any live path, SHA, byte-size, package-version, or
read-only-rootfs mismatch remains fail-closed.

## Calibration

Calibration must be launched only after SSH to `hpctpa3pc0009`:

```bash
bash cellpose_pipeline/hpc/Parameter_calibration/31_run_reference_cell_state_shadow_v2_test.sh
```

It is direct execution, never `sbatch`, and may write only below
`results/Tests_and_Parameters_calibration/`. It uses the same frozen archive,
container identity and human barriers as formal execution.
