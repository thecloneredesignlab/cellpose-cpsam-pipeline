# Live/Dead/Transitional Cell Classification Plan

## Goal

After CellposeSAM segmentation, classify each segmented object into:

```text
live          red
dead          blue
transitional  purple / mixed red-blue
uncertain     low-confidence or likely artifact
```

The classification should operate per segmented cell mask and produce per-cell labels plus aggregate well/site/timepoint summaries.

Important caveat: Cellpose overlay colors are random visualization colors and must not be used for biological classification. Classification must use the original RGB TIFF pixel values inside each Cellpose mask.

## Observation From First Annotated Image

For `001_SUM159_AC_A10_1_00d00h00m.tif`, manual review found:

- about 10 dead cells
- zero transitional cells
- most cells are live/red
- dead cells are small blue/cyan objects

The manually edited TIFF in `~/Downloads/001_SUM159_AC_A10_1_00d00h00m.tif` has orange circles around the dead cells. A first extraction pass detected 10 orange circles, matching the manual count. However, intersecting the circle neighborhoods with Cellpose masks showed an important issue: only a subset of circled cells mapped to masks with clearly blue/cyan raw RGB signal. Several circles overlapped or were closest to larger red Cellpose masks, because the blue dead cell was small, adjacent to a live cell, or not segmented as its own object.

Implementation consequence:

- Per-mask live/dead classification is necessary but not sufficient.
- The pipeline should also report circled/manual dead-cell cases where no blue Cellpose mask is recovered.
- For evaluation, compare manual dead-cell points/circles to nearby mask IDs and flag segmentation misses separately from classification mistakes.
- Transitional calls should be conservative; in this first image, an appropriate classifier should return zero or near-zero transitional cells.

## Inputs

Required:

- raw RGB TIFF image
- Cellpose mask TIFF for the same image
- image metadata from `cellpose_pipeline/manifests/image_inventory.csv`

Known image format:

```text
raw image: (1040, 1408, 3) uint8
mask:      (1040, 1408) uint16
```

The existing `cellpose` conda environment has the needed libraries:

- `numpy`
- `pandas`
- `tifffile`
- `scikit-image`
- `scikit-learn`
- `opencv-python`
- `matplotlib`

## Output Layout

Write all classification outputs under:

```text
cellpose_pipeline/classification/
```

Recommended structure:

```text
features/
  per_cell_features.csv
labels/
  manual_cell_labels.csv
models/
  cell_state_classifier.joblib
predictions/
  per_cell_predictions.csv
  per_image_state_summary.csv
qc/
  label_overlays/
  review_tiles/
  confusion_matrix.png
  calibration_curve.png
```

## Per-Cell Feature Extraction

For every segmented object:

1. Load raw RGB image and mask.
2. For each mask id, create an interior mask.
   - Erode the object by 1-2 pixels where possible.
   - Use the original mask if erosion removes the object.
   - This reduces edge/background contamination.
3. Create a local background ring around the object.
   - Dilate the object by 5-10 pixels.
   - Subtract the original object mask.
   - Exclude pixels belonging to any other object.
4. Compute raw and background-corrected color features.
5. Compute morphology and QC features.

### Color Features

For each cell, compute:

- mean, median, standard deviation for `R`, `G`, `B`
- 10th, 50th, and 90th percentiles for `R`, `G`, `B`
- local-background-corrected `R`, `G`, `B`
- normalized channel fractions:

```text
r_frac = R / (R + G + B + eps)
g_frac = G / (R + G + B + eps)
b_frac = B / (R + G + B + eps)
```

- log color ratios:

```text
log_R_over_B = log((R + eps) / (B + eps))
log_R_over_G = log((R + eps) / (G + eps))
log_B_over_G = log((B + eps) / (G + eps))
```

- HSV features:

```text
hue_mean
saturation_mean
value_mean
hue_circular_variance
```

- Lab features:

```text
L_mean
a_mean
b_mean
```

### Purple / Transitional Features

Purple cells should have both red and blue signal with less green contribution. Compute explicit mixed-color features:

```text
red_blue_sum       = r_frac + b_frac
red_blue_balance   = 1 - abs(r_frac - b_frac)
purple_score       = min(r_frac, b_frac) - g_frac
red_blue_intensity = min(R_bg_corrected, B_bg_corrected)
```

These should help distinguish:

- strongly red live cells
- strongly blue dead cells
- mixed red-blue transitional cells
- low-saturation gray/background artifacts

### Morphology and QC Features

For each mask:

- area
- perimeter
- eccentricity
- solidity
- major/minor axis length
- mean intensity/saturation
- border-touching flag
- local background brightness
- mask id and source image metadata

These features are useful for identifying artifacts and debris, not just biological state.

## Manual Dead-Cell Annotation Import

The orange-circle annotation can be used to bootstrap validation labels, but the orange pixels themselves are not biological signal.

Recommended import procedure:

1. Detect orange annotation circles in the edited TIFF using HSV thresholds.
2. Estimate each circle center and radius/bounding box.
3. Search for Cellpose mask IDs near the circle center and inside the circle.
4. Compute raw RGB features from the original unedited TIFF, not from the edited TIFF.
5. Assign a manual `dead` label only when the matched mask's raw pixels are blue/cyan or when the mask is clearly the circled object.
6. If the circled object has no matching mask, record it as a `segmentation_miss_dead_cell`.

This distinction matters: a circled dead cell next to a live red mask should not turn the red mask into a dead training example.

For the first image, the manual observation should be used as validation expectation:

```text
dead cells:          about 10
transitional cells:  0
most segmented cells: live
```

## Baseline Classifier

Start with an interpretable color-rule baseline before training a model.

### Rule-Based Labels

Use background-corrected normalized channels:

```text
live_score = r_frac - max(b_frac, g_frac)
dead_score = b_frac - max(r_frac, g_frac)
transition_score = min(r_frac, b_frac) - g_frac
```

Initial rule:

```text
live          if live_score is high and saturation is high
dead          if dead_score is high and saturation is high
transitional  if transition_score is high or red_blue_balance is high
uncertain     if saturation is low, object is too small, or scores conflict
```

Do not hard-code final thresholds upfront. Estimate thresholds from the evaluation data:

1. Extract features from the off-the-shelf segmentation sample.
2. Plot `r_frac`, `b_frac`, `log_R_over_B`, and `purple_score`.
3. Use quantiles and visual inspection to set provisional thresholds.
4. Generate prediction overlays and review.

This baseline gives a fast sanity check and creates weak labels that can help prioritize manual review.

For the first image, tune the rule-based baseline against the manual expectation:

- obvious blue/cyan objects should be `dead`
- red objects should be `live`
- mixed purple should be rare or absent
- tiny low-saturation objects should be `artifact` or `uncertain`
- any discrepancy between the 10 manual dead circles and mask-level dead calls should be reviewed as either segmentation miss, merged object, or classifier error

## First-Pass Implementation Limitation

The current first-pass classifier in `cellpose_pipeline/scripts/17_classify_cell_states.py` uses whole-mask color features and hard-coded rule thresholds. It does not yet implement the plan's interior-mask erosion or local-background correction.

This is acceptable for the first benchmark image, where the output was qualitatively reasonable and produced zero transitional calls as expected. It should not be treated as final for full-dataset analysis. Before scaling across wells, timepoints, and treatment conditions, update the classifier to:

- compute features from an eroded cell interior when possible
- compute a local background ring for each cell
- use background-corrected red/green/blue and HSV/Lab features
- calibrate thresholds or train a supervised classifier on manually reviewed cells
- keep segmentation misses separate from live/dead classification mistakes

## Supervised Classifier

If the rule-based classifier is imperfect, train a small supervised classifier.

### Manual Label Set

Create review tiles for individual segmented cells, not whole images.

Recommended first label set:

```text
live:          300 cells
dead:          300 cells
transitional:  300 cells
artifact:      100 cells
```

Sample cells across:

- 12 time bins from 0-168 hours
- wells A-H and columns 2-11
- sites 1-4
- small, medium, and large masks
- red, blue, purple, and low-saturation color quantiles
- early sparse fields and late dense/debris-heavy fields

Include an `artifact` class because segmentation will sometimes capture debris, bubbles, background texture, or partial border objects.

### Model Choice

Train in this order:

1. multinomial logistic regression
2. random forest
3. gradient boosting or histogram gradient boosting

Start with logistic regression because it is interpretable and likely sufficient if color separation is strong. Use random forest if color/background/morphology interactions matter.

Inputs:

- color features
- background-corrected color features
- HSV/Lab features
- morphology/QC features

Target classes:

```text
live
dead
transitional
artifact
```

For reporting, keep `artifact` separate from biological states.

### Probability Handling

Do not force every cell into live/dead/transitional.

Use probability thresholds:

```text
label = argmax probability if max_probability >= 0.70
label = uncertain otherwise
```

Also mark as uncertain if:

- object area is outside expected cell range
- saturation is very low
- object touches image border
- local background correction is unstable

## QC and Validation

### Per-Cell QC

Generate overlays where masks are colored by predicted state:

```text
live          red
dead          blue
transitional  purple
artifact      yellow
uncertain     gray
```

For each model/rule version, review:

- 4 early images
- 4 mid-timepoint images
- 4 late images
- 4 dense/debris-heavy images
- images with extreme live/dead fractions
- images with many uncertain cells

### Aggregate QC

For each well/site/timepoint:

- total segmented cells
- live count and fraction
- dead count and fraction
- transitional count and fraction
- artifact/uncertain fraction
- segmented area by state

Expected sanity checks:

- live fraction should be high in healthy early fields
- dead fraction should rise in toxic/treatment conditions
- transitional fraction may rise before dead fraction
- abrupt state changes between adjacent timepoints should be inspected
- artifact/uncertain fraction should not dominate unless the image is genuinely poor

### Validation Metrics

For supervised classification:

- confusion matrix
- per-class precision, recall, F1
- balanced accuracy
- calibration curve
- review of high-confidence mistakes

Keep validation split by image/well, not random cells only, so nearby cells from the same image do not leak into both train and test.

## Proposed Implementation Steps

1. Add `10_extract_cell_state_features.py`.
   - Inputs: raw image directory, mask directory, inventory CSV.
   - Output: `classification/features/per_cell_features.csv`.
   - One row per segmented object.

2. Add `11_make_cell_review_tiles.py`.
   - Sample diverse cells for manual labeling.
   - Save small RGB crops with mask outline.
   - Write `classification/labels/review_tile_manifest.csv`.

3. Add `12_apply_rule_based_cell_state_classifier.py`.
   - Compute provisional live/dead/transitional/artifact labels.
   - Save `classification/predictions/per_cell_predictions_rule_v1.csv`.
   - Generate state-colored overlays.

4. Add `13_train_cell_state_classifier.py`.
   - Read manual labels.
   - Train logistic regression and random forest.
   - Save model and validation reports.

5. Add `14_apply_cell_state_classifier.py`.
   - Apply trained classifier to selected evaluation masks or the full dataset.
   - Save per-cell predictions and per-image summaries.

6. Add `15_make_cell_state_qc_overlays.py`.
   - Generate overlays colored by predicted state.
   - Create montage sheets for review.

## Initial Recommendation

Start with the `cpsam` segmentation already generated for the benchmark image and build the first feature extraction and rule-based classifier around that.

Then expand to a small representative segmentation set:

```text
12 images total
1 image per temporal bin
balanced across sites and plate rows
```

Use those 12 images to tune the rule-based classifier and generate review tiles. If live/dead/transitional colors separate cleanly, the rule-based classifier may be enough. If not, label roughly 1,000 cells and train the supervised classifier.

## Risks and Mitigations

### Risk: Cellpose Masks Include Debris

Mitigation:

- include `artifact` as a class
- use size, saturation, solidity, and background features
- keep uncertain/artifact predictions out of biological live/dead fractions

### Risk: Illumination or Color Drift Across Time

Mitigation:

- use local background correction
- normalize color within image or field
- track per-image background color in summaries
- inspect timepoints with extreme background shifts

### Risk: Transitional Cells Are Ambiguous

Mitigation:

- treat transitional as probabilistic, not a hard color threshold
- preserve class probabilities
- report both hard labels and continuous live/dead/purple scores

### Risk: Overlay Review Bias

Mitigation:

- review raw RGB crops alongside label overlays
- use fixed state colors for predictions
- avoid Cellpose random mask colors when judging biological class
