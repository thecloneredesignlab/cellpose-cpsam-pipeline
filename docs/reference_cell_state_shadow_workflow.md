# Reference cell-state shadow workflow

## Purpose and isolation

This workflow reproduces the historical `cell-phenotype-annotator` method as
an independent shadow classifier. It never replaces the production
viability/trajectory classifier and never writes to its result root.

The frozen V1 ontology is mutually exclusive and ordered:

1. `multinucleated_cell`: two or more distinct nuclei in the anchored Combined
   object; this class takes precedence;
2. `dead_cell`: when multinucleation is absent, Brightfield morphology supports
   death;
3. `live_cell`: neither higher-priority class applies and Brightfield morphology
   supports a live cell.

Insufficient evidence remains unassigned or is skipped during review. It is
not a fourth trainable class.

The model uses only the historical promoted-shape profile, mapped to current
pixel-unit columns:

```text
area_px2
perimeter_px
roundness
aspect_ratio
extent
solidity
equivalent_diameter_px
major_axis_px
minor_axis_px
```

Nuclei is review-only support. Dead, Combined RGB, current state/confidence,
trajectory, and current predictions are neither bound nor read before the
reference model is frozen.

## Phase A

Formal Phase A reuses the frozen 32,000-cell development representative
project from:

```text
results/broad_phenotype_shadow_20260812_075437
```

It verifies the parent projection, row universe, cells/features lockstep,
split/selection manifests, and source hashes. It copies the same development
cell universe, writes the exact nine features, installs the three new classes,
and recomputes UMAP. Parent UMAP coordinates and broad-phenotype labels are not
imported.

The Phase A entry point is:

```bash
bash cellpose_pipeline/hpc/submit_reference_cell_state_shadow.sh
```

It creates a new sibling root:

```text
results/reference_cell_state_shadow_<timestamp>/
```

The single CPU Slurm job performs:

```text
parent import -> CPA validate -> 9-feature UMAP -> CPA annotate
              -> morphology overlay/atlas -> human region barrier
```

The tracked code commit is archived at submission. Slurm copies and verifies
that archive on the compute node, executes the node-local worker, and mounts
the node-local source read-only at the canonical snapshot path in the latest
SIF. The parent project, raw BF/Nuclei data, source segmentation masks, and
pinned reference checkout are read-only binds. Only the new reference root is
writable. No GPU or node constraint is requested.

## Historical morphology reference

The generic CPA annotation page contains only points and polygons. The
historical LTEE workflow instead displayed real morphology beside the UMAP.
`27_build_reference_morphology_workspace.py` restores that behavior without
modifying the immutable CPA annotation generation:

- at most 300 deterministic, context-balanced real cells are selected in UMAP
  space;
- Brightfield cutouts use the Combined mask for transparent background and a
  visible object boundary;
- one source-pixel scale is used across the overlay, preserving relative cell
  size;
- Nuclei is displayed separately in the atlas as human evidence;
- source images, crops, coordinates, annotation identity, and output files are
  SHA-256 bound.

Open `morphology_reference/annotation_workspace.html`. Its left pane is the
unmodified CPA polygon editor and its right pane is the morphology overlay and
atlas. Region boundaries are provisional labels only.

## Human barriers and model

1. Draw regions with the morphology workspace and export
   `region_submission.json`.
2. Run authoritative CPA `annotation-import`; CPA recomputes every polygon
   assignment.
3. Build the balanced image review: 50 cells per class plus 100 unassigned,
   maximum 8 per well, shortage policy `fail`.
4. Confirm, correct, or skip each crop and export `review_submission.json`.
5. Run authoritative `review-import`. Only confirmed/corrected rows with
   confidence at least 0.8 are eligible.
6. Train grouped nested-CV multinomial glmnet: well grouping, 5 outer folds,
   5 inner folds, fold-local preprocessing, `alpha=1`, `lambda.1se`.
7. Accept an immutable model generation and run field-sharded full prediction.

The merged reference result is:

```text
predictions/reference_cell_state_predictions.tsv
```

with exact columns:

```text
model_id
cell_id
reference_cell_state_class_id
prediction_status
```

It does not contain or overwrite `state`, `final_state`, or current classifier
probabilities.

## Comparison boundary

`29_compare_current_vs_reference_cell_state.py` runs only after both axes are
frozen and writes a third, independent comparison root. The 4-by-3 table is a
cross-classification/association table, not an accuracy table. For the shared
dead endpoint, current `uncertain`/`artifact` and unavailable reference rows are
abstentions. `multinucleated_cell` remains its own reference prediction and is
never silently renamed `live`.

True sensitivity, specificity, or method superiority requires a later blinded
heldout gold review. The precomparison receipt therefore records
`PRECOMPARISON_NO_GOLD_STANDARD` and `accuracy_claimed=false`.

## Calibration

Calibration runs only after login-node SSH to `hpctpa3pc0009` and only below:

```text
results/Tests_and_Parameters_calibration/
```

Use:

```bash
bash cellpose_pipeline/hpc/Parameter_calibration/30_run_reference_cell_state_shadow_test.sh
```

It consumes the completed 100-cell broad calibration parent
`broad_phenotype_shadow_test_20260812_073844`, uses the same latest SIF, and
does not call `sbatch`.
