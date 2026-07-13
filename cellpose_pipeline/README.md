# SUM159 Doxorubicin Cellpose Pipeline

This folder contains a Cellpose v4 / CellposeSAM segmentation workflow and first-pass live/dead classification workflow for Incucyte TIFFs in:

`../20260619_SUM159_Doxorubicin_Cyclophosphamide/20260626_SUM159_AC_Exp_1`

The local conda environment is expected to be named `cellpose`.

For running this pipeline on another machine, see [../docs/server_runbook.md](../docs/server_runbook.md).

## Quick Start

Run commands from the experiment root:

```bash
cd /Volumes/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1
```

1. Build an image inventory:

```bash
conda run -n cellpose python cellpose_pipeline/scripts/00_inventory.py
```

2. Run CellposeSAM segmentation with the tuned folder profiles. Images in `Brightfield/`, `Dead/`, and `Nuclei/` are assigned model and parameters from the parent folder name:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/18_run_segmentation_classification_workflow.py \
  --dir /path/to/20260626_SUM159_AC_Exp1_SeparateImages \
  --recursive \
  --skip-unknown-profiles \
  --force-classification \
  --run-name separate_images_profiles
```

Outputs are written under:

```text
cellpose_pipeline/workflow_runs/<run-name>/
  segmentations/Brightfield/
  segmentations/Dead/
  segmentations/Nuclei/
  metadata/
  qc/segmentation_overlays/
  classification/features/
  classification/predictions/
  classification/qc/label_overlays/
```

For quick testing with an existing mask, skip segmentation by providing `--mask-image`:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/18_run_segmentation_classification_workflow.py \
  --image-path cellpose_pipeline/annotation_images/raw/001_SUM159_AC_A10_1_00d00h00m.tif \
  --mask-image /private/tmp/cellpose_cpsam_bench_1_rerun/001_SUM159_AC_A10_1_00d00h00m_cp_masks.tif \
  --run-name benchmark_001_existing_mask
```

The tuned SeparateImages workflow now runs segmentation, segmentation QC overlays, and the retained first-pass classifier by default in the HPC entrypoints. Set `SEGMENTATION_ONLY=1` or pass `--segmentation-only` when only masks and metadata are needed.

## Nuclear analysis and cross-channel calibration

The nuclear workflow keeps a high-recall Cellpose extent mask and an
intensity-supported core mask with matching instance IDs. The core reduces the
effect of red-channel overexposure on density, nuclear/cytoplasmic ratios, and
cell-boundary diagnostics.

Post-segmentation analysis and optional BF/Combined boundary calibration are
implemented in:

- `scripts/31_analyze_nuclear_cell_alignment.py`: nuclear morphology,
  multinucleation, nuclear/cytoplasmic ratios, and mismatch types;
- `scripts/33_diagnose_registration_and_refine_cell_masks.py`: global offset
  diagnosis and nucleus-aware local candidates;
- `scripts/34_score_nucleus_aware_cell_refinement.py`: raw-edge, area, topology,
  and cross-method guardrails;
- `scripts/35_render_nucleus_aware_cell_refinement_qc.py`: focused before/after
  QC mosaics;
- `hpc/run_nucleus_aware_cell_refinement_*.sh`: reproducible HPC runners.
- `scripts/36_screen_shape_aware_nucleus_splits.py`: shape gating, stable
  fluorescence-peak detection, and optional merged-nucleus splits;
- `scripts/37_score_shape_aware_nucleus_splits.py`: foreground, core, density,
  morphology, and multinucleation guardrails;
- `scripts/38_render_shape_aware_nucleus_split_qc.py`: per-object split QC;
- `hpc/run_shape_aware_nucleus_split_*.sh`: reproducible shape-screening,
  validation, and final-QC runners.

The calibrated cell masks and shape-aware nucleus splits are separate
sensitivity-analysis layers and do not overwrite the original production masks. See
[`../docs/nuclei_segmentation_optimization.md`](../docs/nuclei_segmentation_optimization.md)
for the retained nuclear parameters and the full 20-field validation.

## Optional Annotation Prep

These scripts are retained for selecting representative images and preparing a future supervised training set from manually reviewed failure cases. They are not required for the current off-the-shelf CellposeSAM workflow.

Select representative images for annotation:

```bash
conda run -n cellpose python cellpose_pipeline/scripts/01_select_annotation_images.py --n-images 96
```

Open the Cellpose GUI and manually annotate the selected images:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib conda run -n cellpose cellpose --dir cellpose_pipeline/annotation_images/raw
```

Save annotations from the GUI as Cellpose `_seg.npy` files. For each image, the mask file should sit next to the TIFF and have the same basename ending in `_seg.npy`.

Split annotated images into train/test sets:

```bash
conda run -n cellpose python cellpose_pipeline/scripts/02_prepare_training_split.py
```

## Notes

- Cellpose `4.2.1.1` is required. The main workflow exits if another Cellpose package version is active.
- The env currently needs `KMP_DUPLICATE_LIB_OK=TRUE` to avoid a duplicate OpenMP runtime abort.
- The scripts also set `MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib` because the default Matplotlib cache path is not writable in this sandbox.
- Auto profiles use folder-specific CellposeSAM models: `Brightfield -> cpsam`, `Dead -> cpsam_v2`, `Nuclei -> cpsam_v2`.
- Auto profiles use single-channel `channel_axis=None` and external percentile normalization with Cellpose `normalize=False`.

## Expected Layout

```text
cellpose_pipeline/
  annotation_images/raw/      selected TIFFs for manual annotation
  train/                      annotated training image/mask pairs
  test/                       annotated held-out image/mask pairs
  workflow_runs/              CellposeSAM segmentation, classification, and QC outputs
  manifests/                  inventories, selected files, split manifests
```

## Annotation Guidance

Start with 50-100 images before training. Include:

- early, middle, and late timepoints
- sparse and dense wells
- dying/debris-heavy treated wells
- multiple sites from the same well
- a few difficult images that default Cellpose handles poorly

After the first model, review QC overlays and add 20-40 more annotations from failure cases.
