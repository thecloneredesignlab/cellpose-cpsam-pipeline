# Off-the-Shelf CellposeSAM Evaluation Plan

## Goal

Before fine-tuning Cellpose, test whether the installed off-the-shelf CellposeSAM model produces usable SUM159 Incucyte segmentations on this dataset.

The evaluation should answer:

1. Does CellposeSAM detect individual cells or cell bodies well enough for downstream counts/confluence-style metrics?
2. Which threshold/diameter settings are robust across early, mid, and late timepoints?
3. Which failure modes remain and should be prioritized for manual annotation/training?

## Current Dataset Facts

From `cellpose_pipeline/manifests/image_inventory.csv`:

- 27,200 TIFF images
- 80 wells
- 4 sites per well
- 85 parsed timepoints
- time range: 0 to 168 hours
- sampled TIFF shape checked directly: `(1040, 1408, 3) uint8`

The images are RGB TIFFs, so Cellpose commands should use:

```bash
--channel_axis 2
```

## Models to Try

### Primary Model

Use the installed CellposeSAM model:

```text
cpsam
```

Rationale:

- `cellpose` env reports Cellpose `4.0.7`.
- `from cellpose import models; models.MODEL_NAMES` returns `['cpsam']`.
- The `cpsam` weights are already cached locally in `~/.cellpose/models/cpsam`.
- This is the only true CellposeSAM model exposed by the installed package.

### Optional Legacy Baselines

The local Cellpose cache also contains:

```text
cytotorch_0
cytotorch_1
cytotorch_2
cytotorch_3
```

These are older Cellpose cytoplasm-family model files, not CellposeSAM. I would not treat them as primary candidates, but they are useful as a sanity baseline if `cpsam` performs poorly. Test them only after the `cpsam` parameter sweep, and only if Cellpose v4 accepts them as explicit `--pretrained_model` paths.

I would not test a nuclei model first because these are RGB Incucyte whole-cell/brightfield-style images, not nuclear fluorescence images.

## Representative Sampling Strategy

Do not evaluate on all 27,200 images initially. Build a representative evaluation subset that is large enough to expose failure modes but small enough to inspect manually.

### Evaluation Set Size

Use 192 images for the first off-the-shelf evaluation:

- 12 temporal bins
- 16 images per temporal bin
- covers 0 to 168 hours

This is twice the current 96-image annotation set and gives enough coverage to assess dense, sparse, dying, and late-treatment morphologies.

### Stratification Axes

Sample across:

- `elapsed_hours`: 12 equal-width bins across 0-168 hours
- `well`: all 80 wells represented across the full set
- `site`: balance sites 1-4
- plate position: distribute across rows A-H and columns 2-11

Because no treatment map is currently encoded in the filenames, plate position should be used as a proxy for treatment diversity until a plate map is added.

### Sampling Procedure

Use the inventory CSV as the source of truth.

For each time bin:

1. Sort candidate images by well, site, and elapsed time.
2. Select 16 images using evenly spaced indices through the sorted candidate list.
3. Rotate the index offsets by time bin so different wells/sites are selected at different timepoints.
4. Enforce site balance so each bin contains roughly 4 images per site.
5. Avoid selecting the same well too often unless needed to fill a bin.

Expected output:

```text
cellpose_pipeline/eval_off_the_shelf/raw/
cellpose_pipeline/eval_off_the_shelf/manifests/eval_sample.csv
```

Use symlinks rather than copying TIFFs.

### Optional Enrichment Pass

After the first 192 images, add a small challenge set of 32 images if needed:

- very dense late images
- very sparse or nearly empty images
- strong debris/dead-cell fields
- images with illumination artifacts
- wells where `cpsam` produces extreme counts or near-zero counts

This challenge set should not replace the representative sample; it should be reported separately.

## Parameter Grid

Run `cpsam` over the 192-image evaluation set with a compact parameter grid.

### Diameter

Test:

```text
20, 30, 45, 60
```

Rationale:

- default Cellpose diameter is 30 px
- SUM159 cells and enlarged treated cells may span a wider size range
- late timepoints may contain large flattened cells or PGCC-like cells

### Cell Probability Threshold

Test:

```text
-2.0, -1.0, 0.0, 1.0
```

Rationale:

- lower values recover faint/low-contrast cells but may oversegment debris
- higher values reduce false positives but may miss dim cells

### Flow Threshold

Start with:

```text
0.4
```

Then test:

```text
0.0, 0.8
```

only for the best 2-3 diameter/cellprob settings.

### Minimum Size

Test:

```text
15, 50
```

Rationale:

- `15` preserves small cells and fragments
- `50` suppresses debris-like objects

### First-Pass Grid

Initial grid:

```text
cpsam x diameter {20,30,45,60} x cellprob {-2,-1,0,1} x min_size {15,50}
```

This is 32 runs over 192 images. That is manageable and should expose useful trends.

## Output Layout

Write all off-the-shelf outputs under:

```text
cellpose_pipeline/eval_off_the_shelf/
```

Recommended structure:

```text
raw/                         symlinked evaluation TIFFs
manifests/eval_sample.csv    selected image metadata
results/
  cpsam_d20_cp-2_ms15/
  cpsam_d20_cp-2_ms50/
  ...
qc/
  overlays/
  montage/
summary/
  per_image_metrics.csv
  per_setting_summary.csv
  selected_failure_cases.csv
```

Do not mix these outputs with the future fine-tuned model outputs in `cellpose_pipeline/segmentations`.

## Metrics to Compute

For each image and setting:

- number of masks
- mask area median, mean, and interquartile range
- total segmented area fraction
- fraction of masks touching image borders
- small-object fraction
- large-object fraction
- runtime per image

These metrics are not a replacement for visual QC, but they quickly flag settings that oversegment debris or miss most cells.

## Visual QC

For each parameter setting, generate overlay PNGs.

Review at least:

- 4 early sparse images
- 4 mid-timepoint proliferating images
- 4 late dense images
- 4 late treated/debris-heavy images
- all images with extreme mask counts or area fractions

Score each inspected image:

```text
0 = unusable
1 = detects some cells but major under/oversegmentation
2 = usable for rough aggregate metrics
3 = good single-cell segmentation
```

Record common failure modes:

- undersegmentation of touching cells
- oversegmentation of debris
- missed low-contrast cells
- merged colonies/clumps
- poor handling of enlarged treated cells
- segmentation of background artifacts

## Decision Criteria

Proceed without fine-tuning only if one `cpsam` setting satisfies:

- median visual QC score at least 2 across time bins
- no time bin has median score 0
- cell count and area fraction trends are biologically plausible over time
- failure cases are rare enough for the intended downstream analysis

Fine-tune Cellpose if:

- no off-the-shelf setting reaches score 2 on late or treated images
- counts vary mainly with image artifacts rather than cell density
- large treated cells are consistently missed or fragmented
- debris is consistently segmented as cells

## Original Proposed Implementation Steps

This section records the initial off-the-shelf evaluation plan. The active implementation now uses
`cellpose_pipeline/scripts/01_segment_images.py` for CellposeSAM segmentation plus
classification and `cellpose_pipeline/scripts/analysisi/03_make_missing_workflow_overlays.py` for overlay backfill.

1. Add `Parameter_calibration/03_select_off_the_shelf_eval_images.py`.
   - Read `cellpose_pipeline/manifests/image_inventory.csv`.
   - Select 192 representative images.
   - Create symlinks in `cellpose_pipeline/eval_off_the_shelf/raw`.
   - Write `eval_sample.csv`.

2. Add `07_run_off_the_shelf_grid.sh`.
   - Run Cellpose v4 with `--pretrained_model cpsam`.
   - Use `--channel_axis 2`.
   - Save masks and outlines per parameter setting.
   - Use `KMP_DUPLICATE_LIB_OK=TRUE`.
   - Use `MPLCONFIGDIR=cellpose_pipeline/tmp/matplotlib`.

3. Add `08_summarize_off_the_shelf_grid.py`.
   - Read each setting's mask outputs.
   - Compute per-image mask count and area summaries.
   - Write per-image and per-setting CSV summaries.

4. Add `09_make_off_the_shelf_qc.py`.
   - Generate overlays for representative and extreme-case images.
   - Create one montage per parameter setting.

5. Review overlays and summaries.
   - Pick the best off-the-shelf setting.
   - Identify failure cases for manual annotation.

6. Decide whether to fine-tune.
   - If `cpsam` is adequate, use the chosen setting for the full dataset.
   - If not, annotate failure cases and proceed with training in `cellpose_pipeline/train`.

## Initial Recommendation

Start with CellposeSAM `cpsam`, `diameter=30`, `cellprob_threshold=0.0`, `flow_threshold=0.4`, and `min_size=15` as the baseline run.

Then compare against:

```text
diameter = 20, 45, 60
cellprob_threshold = -1.0, -2.0, 1.0
min_size = 50
```

I would only test the legacy cached `cytotorch_*` models if the `cpsam` grid is clearly inadequate.
