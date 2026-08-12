# Script layout

The top-level numbered scripts are the production pipeline stages. Their
numbers describe the logical analysis order; the HPC orchestrator may run
independent branches in parallel.

1. `01_segment_images.py`
2. `02_call_high_density_from_nuclei_masks.py`
3. `03_calibrate_dead_combined_blue.py`
4. `04_segment_dead_with_combined_blue_consensus.py`
5. `05_merge_dead_consensus_outputs.py`
6. `06_build_postsegmentation_field_manifest.py`
7. `07_build_nucleated_cell_branch.py`
8. `08_fuse_multichannel_classification.py`
9. `09_apply_shape_aware_nucleus_splits.py`
10. `10_merge_shape_strict_shards.py`
11. `11_render_shape_aware_nucleus_split_qc.py`
12. `12_finalize_shape_strict_qc.py`

## Broad-phenotype active-learning V1

`24_prepare_broad_phenotype_active_learning_project.py` creates an immutable
`review.queue` overlay for the pinned Cell Phenotype Annotator source CLI. It
accepts only the physical development representative project, one canonical
representative prediction generation, and one authoritative review-import
`reviewed_labels.tsv` generation:

```bash
python cellpose_pipeline/scripts/24_prepare_broad_phenotype_active_learning_project.py \
  --project "$SHADOW_ROOT/projection_input/representative_umap/project.yml" \
  --shadow-root "$SHADOW_ROOT" \
  --prediction-dir "$MODEL_DIR/predictions/$PREDICTION_ID" \
  --reviewed-labels "$REVIEW_IMPORT_DIR/reviewed_labels.tsv" \
  --queue-id uncertainty_v1 \
  --max-cells 500 \
  --max-per-group 8 \
  --seed 20260812
```

The helper writes
`$SHADOW_ROOT/active_learning/<queue-id>/{project.yml,input_manifest.tsv,receipt.json,USAGE.md}`.
Run the resulting project through `20_run_cellphenotypeannotator_stage.py
--stage review-build --check-config` before the actual review build. Selection
and immutable `review_id` generation remain owned by the pinned reference
implementation.

This V1 interface is targeted validation only. It excludes heldout cells, the
full 38.6M-cell universe, and viability inputs. It neither merges labels across
review IDs nor automatically retrains; one authoritative reviewed-label
generation is frozen as history for each queue.

Other directories are separated by purpose:

- `Parameter_calibration/`: tuning, candidate screening, and validation;
- `analysisi/`: inventory, downstream analysis, plots, QC, and audits;
- `_shared/`: imported implementation modules that are not standalone stages.
