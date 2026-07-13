# Server Runbook: CellposeSAM Segmentation + Cell-State Classification

This runbook describes how to run the SUM159 CellposeSAM segmentation and first-pass live/dead/transitional classification pipeline on another server.

## What the Pipeline Does

For each selected SeparateImages Incucyte TIFF:

1. Select a CellposeSAM model and tuned parameters from the parent folder name.
2. Save Cellpose mask TIFFs.
3. Write per-image metadata containing Cellpose version, model, parameters, and preprocessing settings.
4. Generate segmentation QC overlays.
5. Run the retained first-pass cell-state classifier and write features, predictions, summaries, and label overlays.

Set `SEGMENTATION_ONLY=1` only when masks and metadata are needed without QC/classification outputs.

## Required Files

Copy or mount the experiment folder containing:

```text
20260619_SUM159_Doxorubicin_Cyclophosphamide/
cellpose_pipeline/
docs/
```

The raw TIFFs are expected under:

```text
20260619_SUM159_Doxorubicin_Cyclophosphamide/20260626_SUM159_AC_Exp1_SeparateImages/
```

If the raw TIFF directory differs, rerun inventory and pass explicit paths to the scripts.

## Software Requirements

Use Python `3.10` if possible. The current working environment used:

```text
cellpose 4.2.1.1
torch 2.7.1
numpy
pandas
tifffile
scikit-image
scikit-learn
opencv-python
matplotlib
```

The model used is:

```text
Brightfield -> cpsam
Dead -> cpsam_v2
Nuclei -> cpsam_v2
```

On the current machine, `cpsam` was already cached in:

```text
~/.cellpose/models/cpsam
~/.cellpose/models/cpsam_v2
```

On a new server, Cellpose may download the model on first use if it is not cached. If the server has no internet access, copy the cached `cpsam` file into the server user's `~/.cellpose/models/` directory.

## Environment Setup

One working conda setup is:

```bash
conda create -n cellpose python=3.10 -y
conda activate cellpose
pip install "cellpose==4.2.1.1" tifffile pandas scikit-image scikit-learn opencv-python matplotlib
```

Verify the environment:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl_cellpose \
python -c "import cellpose, torch, tifffile, pandas, skimage, cv2; print('ok'); print('mps', torch.backends.mps.is_available()); print('cuda', torch.cuda.is_available())"
```

If the server has CUDA, test Cellpose GPU support before launching a large run. If it is CPU-only, expect `cpsam` to be slow.

## Runtime Notes

On the current CPU-only machine:

- one full-resolution RGB image took about 7 minutes with `cpsam`
- the 192-image run is roughly a 23-25 hour job
- output is resumable with `--skip-existing`
- one failed image can be skipped with `--continue-on-error`

GPU performance may be much faster, but should be benchmarked on the target server.

## Initial Setup in the Experiment Folder

Run from the experiment root:

```bash
cd /path/to/N01_Incucyte_SUM159_Doxorubicin_Test1
```

Build or refresh the inventory:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/analysisi/01_inventory.py
```

Select the 192-image off-the-shelf evaluation set:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/Parameter_calibration/03_select_off_the_shelf_eval_images.py --n-images 192
```

This writes:

```text
cellpose_pipeline/eval_off_the_shelf/raw/
cellpose_pipeline/eval_off_the_shelf/manifests/eval_sample.csv
```

## Run Segmentation

Start a resumable sequential run:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/01_segment_images.py \
  --dir /share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1/20260619_SUM159_Doxorubicin_Cyclophosphamide/20260626_SUM159_AC_Exp1_SeparateImages \
  --recursive \
  --skip-unknown-profiles \
  --force-classification \
  --run-name separate_images_cpsam_profiles \
  --continue-on-error
```

For an HPC Slurm array run:

```bash
PYTHON_BIN=/path/to/cellpose/python \
bash cellpose_pipeline/hpc/Parameter_calibration/01_submit_separate_images_segmentation.sh \
  /share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1/20260619_SUM159_Doxorubicin_Cyclophosphamide/20260626_SUM159_AC_Exp1_SeparateImages \
  separate_images_cpsam_profiles
```

## Generate Missing Overlays

This can be run at any time while the workflow is still running. It skips overlays that already exist.

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/analysisi/03_make_missing_workflow_overlays.py \
  --run-dir cellpose_pipeline/workflow_runs/overnight_cpsam_192 \
  --manifest cellpose_pipeline/eval_off_the_shelf/manifests/eval_sample.csv
```

It writes:

```text
cellpose_pipeline/workflow_runs/overnight_cpsam_192/qc/segmentation_overlays/
cellpose_pipeline/workflow_runs/overnight_cpsam_192/classification/qc/label_overlays/
```

Classification overlays are saved as side-by-side PNGs: normalized original image on the left and state-colored classification overlay on the right.

To repair or regenerate overlays for a single image stem:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/analysisi/03_make_missing_workflow_overlays.py \
  --run-dir cellpose_pipeline/workflow_runs/overnight_cpsam_192 \
  --manifest cellpose_pipeline/eval_off_the_shelf/manifests/eval_sample.csv \
  --overlay-type classification \
  --force \
  --stem SUM159_AC_B3_3_01d06h00m
```

## Monitor Progress

Check active processes:

```bash
pgrep -af "cpsam_overnight|01_segment_images.py|cellpose --image_path"
```

Count completed segmentations:

```bash
find cellpose_pipeline/workflow_runs/overnight_cpsam_192/segmentations \
  -maxdepth 1 -type f -name '*_cp_masks.tif' | wc -l
```

Count completed classifications:

```bash
find cellpose_pipeline/workflow_runs/overnight_cpsam_192/classification/predictions \
  -maxdepth 1 -type f -name '*_summary.csv' | wc -l
```

Count overlays:

```bash
find cellpose_pipeline/workflow_runs/overnight_cpsam_192/qc/segmentation_overlays \
  -maxdepth 1 -type f -name '*_segmentation_overlay.png' | wc -l

find cellpose_pipeline/workflow_runs/overnight_cpsam_192/classification/qc/label_overlays \
  -maxdepth 1 -type f -name '*_state_overlay.png' | wc -l
```

Check failures:

```bash
cat cellpose_pipeline/workflow_runs/overnight_cpsam_192/failures.tsv
```

No `failures.tsv` means no failures have been recorded yet.

## Output Structure

For run name `overnight_cpsam_192`:

```text
cellpose_pipeline/workflow_runs/overnight_cpsam_192/
  segmentations/
    *_cp_masks.tif
  classification/
    features/
      *_per_cell_features.csv
    predictions/
      *_per_cell_predictions.csv
      *_summary.csv
    qc/
      label_overlays/
        *_state_overlay.png
  qc/
    segmentation_overlays/
      *_segmentation_overlay.png
  failures.tsv
  overnight.log
```

## Resuming or Re-running

The workflow defaults to `--skip-existing`, so rerunning the same command should skip existing mask files and classify any missing outputs.

To force regeneration of segmentation masks, pass:

```bash
--no-skip-existing
```

To force overlay regeneration, pass:

```bash
--force
```

to `analysisi/03_make_missing_workflow_overlays.py`.

## Known Limitations

- CellposeSAM `cpsam` is slow on CPU.
- The current classifier is mask-level only; dead cells missed by segmentation cannot be recovered by classification.
- The first-pass classifier uses whole-mask color features and hard thresholds.
- Interior-mask and local-background correction should be implemented before relying on full-dataset biological conclusions.
- Manual orange-circle annotations should be used as validation hints, not direct training labels, unless the circled object maps cleanly to a blue/cyan Cellpose mask.
