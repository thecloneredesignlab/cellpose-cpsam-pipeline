# SUM159 Doxorubicin Cellpose Pipeline

This folder contains a Cellpose v4 / CellposeSAM segmentation and first-pass live/dead classification workflow for the Incucyte TIFFs in:

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

2. Run CellposeSAM segmentation and the first-pass live/dead/transitional classifier together for one or more images:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/18_run_segmentation_classification_workflow.py \
  --image-path cellpose_pipeline/annotation_images/raw/001_SUM159_AC_A10_1_00d00h00m.tif \
  --run-name benchmark_001
```

Outputs are written under:

```text
cellpose_pipeline/workflow_runs/<run-name>/
  segmentations/
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

The current classifier is a first-pass rule baseline: it uses whole-mask color features and hard thresholds. Interior-mask and local-background correction are still needed before scaling this across the full dataset.

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

- Cellpose `4.0.7` is installed in the `cellpose` environment on this machine.
- The env currently needs `KMP_DUPLICATE_LIB_OK=TRUE` to avoid a duplicate OpenMP runtime abort.
- The scripts also set `MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib` because the default Matplotlib cache path is not writable in this sandbox.
- The default pretrained model is `cpsam`, which is already cached locally.
- The TIFFs sampled here are RGB arrays with shape `(1040, 1408, 3)`, so the workflow default is `--channel-axis 2`.

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
