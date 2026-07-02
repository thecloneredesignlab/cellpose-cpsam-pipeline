# Server Runbook: CellposeSAM Segmentation + Cell-State Classification

This runbook describes how to run the SUM159 CellposeSAM segmentation and first-pass live/dead/transitional classification pipeline on another server.

## What the Pipeline Does

For each selected RGB Incucyte TIFF:

1. Run CellposeSAM `cpsam` segmentation.
2. Save Cellpose mask TIFFs.
3. Classify each segmented object as:
   - `live`
   - `dead`
   - `transitional`
   - `artifact`
   - `uncertain`
4. Write per-cell features and per-image summaries.
5. Generate missing segmentation and classification overlays.

The current classifier is a first-pass rule baseline. It uses whole-mask color features and hard thresholds. Interior-mask erosion and local-background correction are documented as next improvements before final full-dataset analysis.

## Required Files

Copy or mount the experiment folder containing:

```text
20260619_SUM159_Doxorubicin_Cyclophosphamide/
cellpose_pipeline/
docs/
```

The raw TIFFs are expected under:

```text
20260619_SUM159_Doxorubicin_Cyclophosphamide/20260626_SUM159_AC_Exp_1/
```

If the raw TIFF directory differs, rerun inventory and pass explicit paths to the scripts.

## Software Requirements

Use Python `3.10` if possible. The current working environment used:

```text
cellpose 4.0.7
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
cpsam
```

On the current machine, `cpsam` was already cached in:

```text
~/.cellpose/models/cpsam
```

On a new server, Cellpose may download the model on first use if it is not cached. If the server has no internet access, copy the cached `cpsam` file into the server user's `~/.cellpose/models/` directory.

## Environment Setup

One working conda setup is:

```bash
conda create -n cellpose python=3.10 -y
conda activate cellpose
pip install "cellpose==4.0.7" tifffile pandas scikit-image scikit-learn opencv-python matplotlib
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
conda run -n cellpose python cellpose_pipeline/scripts/00_inventory.py
```

Select the 192-image off-the-shelf evaluation set:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/06_select_off_the_shelf_eval_images.py --n-images 192
```

This writes:

```text
cellpose_pipeline/eval_off_the_shelf/raw/
cellpose_pipeline/eval_off_the_shelf/manifests/eval_sample.csv
```

## Run Segmentation + Classification

Start a resumable sequential run:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/18_run_segmentation_classification_workflow.py \
  --dir cellpose_pipeline/eval_off_the_shelf/raw \
  --run-name overnight_cpsam_192 \
  --continue-on-error
```

For a detached run using `screen`:

```bash
mkdir -p cellpose_pipeline/workflow_runs/overnight_cpsam_192
screen -dmS cpsam_overnight bash -lc '
cd /path/to/N01_Incucyte_SUM159_Doxorubicin_Test1 &&
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/18_run_segmentation_classification_workflow.py \
  --dir cellpose_pipeline/eval_off_the_shelf/raw \
  --run-name overnight_cpsam_192 \
  --continue-on-error \
  > cellpose_pipeline/workflow_runs/overnight_cpsam_192/overnight.log 2>&1
'
```

## Generate Missing Overlays

This can be run at any time while the workflow is still running. It skips overlays that already exist.

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/19_make_missing_workflow_overlays.py \
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
conda run -n cellpose python cellpose_pipeline/scripts/19_make_missing_workflow_overlays.py \
  --run-dir cellpose_pipeline/workflow_runs/overnight_cpsam_192 \
  --manifest cellpose_pipeline/eval_off_the_shelf/manifests/eval_sample.csv \
  --overlay-type classification \
  --force \
  --stem SUM159_AC_B3_3_01d06h00m
```

## Monitor Progress

Check active processes:

```bash
pgrep -af "cpsam_overnight|18_run_segmentation_classification_workflow.py|cellpose --image_path"
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

to `19_make_missing_workflow_overlays.py`.

## Known Limitations

- CellposeSAM `cpsam` is slow on CPU.
- The current classifier is mask-level only; dead cells missed by segmentation cannot be recovered by classification.
- The first-pass classifier uses whole-mask color features and hard thresholds.
- Interior-mask and local-background correction should be implemented before relying on full-dataset biological conclusions.
- Manual orange-circle annotations should be used as validation hints, not direct training labels, unless the circled object maps cleanly to a blue/cyan Cellpose mask.
