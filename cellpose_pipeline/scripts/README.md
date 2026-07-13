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

Other directories are separated by purpose:

- `Parameter_calibration/`: tuning, candidate screening, and validation;
- `analysisi/`: inventory, downstream analysis, plots, QC, and audits;
- `_shared/`: imported implementation modules that are not standalone stages.
