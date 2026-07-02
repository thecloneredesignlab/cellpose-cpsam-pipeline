# SUM159 CellposeSAM Pipeline

Reusable CellposeSAM segmentation and first-pass live/dead/transitional classification workflow for SUM159 Incucyte RGB TIFFs.

## Contents

- `cellpose_pipeline/scripts/`: runnable inventory, sampling, segmentation, classification, and overlay scripts.
- `cellpose_pipeline/README.md`: quick-start commands from an experiment folder.
- `docs/server_runbook.md`: server setup and production run instructions.
- `docs/cellpose_sam_off_the_shelf_evaluation_plan.md`: off-the-shelf CellposeSAM evaluation plan and rationale.
- `docs/cell_state_classification_plan.md`: cell-state classification plan and known limitations.
- `env/environment.yml`: conda environment with Cellpose `4.0.7` / CellposeSAM support.

## Data

Raw TIFFs and generated outputs are not included. The scripts expect the experiment folder to contain the raw image tree documented in `docs/server_runbook.md`, or explicit paths passed on the command line.

## Quick Start

Create the environment:

```bash
conda env create -f env/environment.yml
```

Run from an experiment root containing `cellpose_pipeline/` and the raw TIFF directory:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/00_inventory.py

KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/06_select_off_the_shelf_eval_images.py --n-images 192

KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/18_run_segmentation_classification_workflow.py \
  --dir cellpose_pipeline/eval_off_the_shelf/raw \
  --run-name overnight_cpsam_192 \
  --continue-on-error
```

Generate missing overlays:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib \
conda run -n cellpose python cellpose_pipeline/scripts/19_make_missing_workflow_overlays.py \
  --run-dir cellpose_pipeline/workflow_runs/overnight_cpsam_192 \
  --manifest cellpose_pipeline/eval_off_the_shelf/manifests/eval_sample.csv
```

See `docs/server_runbook.md` for full operational details.
