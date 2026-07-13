# Nuclear morphometry and Nuclei–BF–Combined alignment analysis

## Purpose

`cellpose_pipeline/scripts/analysisi/05_analyze_nuclear_cell_alignment.py` analyzes the
existing instance masks without rerunning Cellpose. It addresses five linked
questions:

1. What are the size, shape, and nuclear-channel intensity characteristics of
   each segmented nucleus?
2. Which BF and Combined cell instance contains each nucleus, and do the two
   cell-mask methods agree?
3. Does the observed nuclear-size distribution contain clearly separable size
   modes?
4. What are the nuclear-to-cell and nuclear-to-cytoplasm area ratios?
5. Which nuclei are unmatched, partially outside a cell, or cross cell-instance
   boundaries?

The analysis keeps all measurements in object-level tables so that the
classification thresholds can be recalibrated after manual review.

## Local command

```bash
PYENV_VERSION=CellPose python \
  cellpose_pipeline/scripts/analysisi/05_analyze_nuclear_cell_alignment.py \
  --run-root segmentation_parameter_tuning/largetest_full_fusion_20260708_213504 \
  --input-root SUM159_AC_Exp1_SeparateImages_largetest \
  --out-dir segmentation_parameter_tuning/largetest_full_fusion_20260708_213504/nuclear_cell_alignment_analysis
```

The local TIFF metadata contains a 72-DPI display tag but no trustworthy
microscopy pixel-size calibration. Therefore, the current run reports pixels
and square pixels. If a calibrated scale becomes available, add:

```bash
--pixel-size-um <micrometers_per_pixel>
```

## Inputs and identity matching

For each field key (`well_site_DDdhhHmm`), the script requires:

- `Nuclei/segmentations/*_cp_masks.tif`
- `Brightfield/segmentations/*_cp_masks.tif`
- `Combined/segmentations/*_cp_masks.tif`
- the corresponding raw Nuclei, BF, and Combined images

All three masks must share the same field keys, shape, and pixel coordinate
system. Duplicate or incomplete keys stop the analysis instead of being silently
dropped.

## Nuclear morphometry

For every nuclear instance, the script records:

- area and equivalent diameter;
- major/minor axis lengths, axis ratio, eccentricity, and orientation;
- perimeter, circularity, convex area, solidity, and bounding-box extent;
- centroid, bounding box, and image-edge contact;
- nuclear-channel mean, standard deviation, P10, median, and P90 intensity;
- local background median/MAD and local robust signal-to-noise ratio.

The second-moment axes use the same convention commonly used by region-property
tools: axis length is four times the square root of the corresponding covariance
eigenvalue.

A nucleus is eligible for high-confidence multinucleation counting when it:

- has area at least 35 px²;
- does not touch the image edge; and
- has solidity at least 0.60.

These conditions are configurable and do not delete objects from the detailed
tables.

## Nucleus-to-cell matching

Matching is based on pixel overlap, not label-number equality or centroid
distance alone. For each nucleus and each cell-mask method, the script calculates:

- the primary and secondary overlapping cell instances;
- the fraction of nuclear area in the primary cell;
- the fraction outside all cell masks;
- nucleus–cell IoU and the fraction of the cell occupied by that nucleus;
- the number of intersected and significantly intersected cells;
- whether the nuclear centroid is inside the primary cell;
- distance from the nuclear centroid to the nearest cell-instance boundary.

The default per-method categories are:

- `contained`: at least 80% of the nucleus is in one cell and its centroid is in
  that cell;
- `crosses_cells`: the nucleus meaningfully overlaps at least two cell instances
  or at least 10% lies in the secondary cell;
- `unmatched`: less than 20% of the nucleus overlaps the cell mask;
- `partly_outside`: the nucleus is matched but at least 20% lies outside cell
  masks;
- `low_containment`: a residual matched category between these conditions.

An intersection is significant when it contains at least three pixels and at
least 5% of the nuclear area.

## BF–Combined consensus

BF and Combined cell labels are unrelated identifiers. The script therefore
constructs a cell-to-cell pixel-overlap matrix and identifies the maximum-overlap
partner in each direction.

`strict_consensus` requires all of the following:

1. the nucleus is `contained` in one Combined cell;
2. the nucleus is `contained` in one BF cell; and
3. the assigned BF and Combined cells are reciprocal-best overlap partners.

`acceptable_alignment` relaxes containment to 50% and permits a one-way-best
cell pair. `actionable_mismatch` flags unmatched/low-overlap assignments,
boundary crossings, and BF–Combined assignment conflicts.

## Optional nucleus-aware boundary calibration

The original BF and Combined masks remain the primary segmentation outputs.
When a sensitivity analysis is needed, scripts `Parameter_calibration/15`–`17` provide a separate,
traceable calibration layer:

1. search ±6 px for a systematic channel offset;
2. construct local cell-boundary candidates only where intensity-supported
   nucleus cores identify reciprocal-best BF/Combined cell pairs;
3. reject candidates that lose labels, materially alter cell areas, increase
   fragmentation, reduce raw BF edge support, or destabilize multinucleation;
4. render before/after crops for visual review.

For the current 20-field SUM159 set, no global offset was supported. The selected
`safe10_relabel` candidate permits at most 10 changed pixels per nucleus and
profile and preserves at least 35 px² of every source cell label. It is stored
under `nucleus_aware_cell_refinement_safe_hpc_20260711/candidates/` and must not
silently replace the baseline masks. Alignment gains are partly circular because
nucleus cores construct the candidate; raw-image edge evidence and manual review
remain required guardrails.

## Optional shape-aware merged-nucleus screen

Scripts 36–38 add a second optional sensitivity layer for Cellpose instances
that may contain two adjacent nuclei. A split requires an abnormal parent shape,
two spatially separated fluorescence peaks, a substantial peak-to-valley drop,
stability across four smoothing scales, plausible child areas, improved child
shape, and BF/Combined reciprocal cell support. Watershed is restricted to the
original parent foreground; disconnected regions without a marker are rejected.

For the current 20-field set, the retained `shape_strict` screen proposed only
7 splits among 40,174 nuclei. Density calls remained unchanged, so the primary
extent/core masks remain the production result and the split masks are reported
only as a sensitivity analysis.

## Multinucleation

The script summarizes nuclei separately for every Combined and BF cell:

- `confirmed_multinucleated`: at least two high-confidence contained nuclei;
- `candidate_multinucleated`: at least two non-fragment nuclei with at least 50%
  containment but fewer than two high-confidence nuclei;
- `single_nucleus`: one high-confidence or moderate nucleus;
- `no_confident_nucleus`: none of the above.

This is a mask-derived phenotype. It is not a ground-truth biological call until
the candidate cells are reviewed against the raw image.

## Nuclear-to-cell and nuclear-to-cytoplasm ratios

For every cell, nuclear area is counted directly from all Nuclei-mask pixels that
fall inside that cell instance. This avoids double counting and does not depend
on the primary assignment category.

\[
R_{N/cell} = \frac{A_{nucleus\ inside\ cell}}{A_{cell}}
\]

\[
R_{N/cytoplasm} =
\frac{A_{nucleus\ inside\ cell}}
{A_{cell} - A_{nucleus\ inside\ cell}}
\]

High ratios can arise from biological nuclear enlargement, an undersized cell
mask, an oversized nuclear mask, or combinations of these effects. BF and
Combined ratios and the QC overlay should be interpreted together.

## Nuclear-size separability

The script provides two complementary outputs:

- descriptive `small`, `medium`, and `large` bins using the eligible-nucleus
  Q25/Q75 area thresholds; and
- a one-dimensional Gaussian-mixture comparison with one, two, and three
  components fitted to log nuclear area and selected by BIC.

Adjacent components are considered cleanly separated only when Ashman's D is at
least 2 and every component contains at least 5% of eligible nuclei. When this
criterion fails, the quantile classes remain useful for stratified review but
must not be presented as distinct biological size populations.

## Composite confidence

The transparent composite score combines:

- nuclear morphology quality: 20%;
- Combined assignment: 35%;
- BF assignment: 30%;
- BF–Combined cell-pair consensus: 15%.

All component measurements and the final score are retained in the CSV outputs,
so downstream analyses can change the weights without rerunning segmentation.

## Outputs

- `nucleus_features.csv`
- `nucleus_cell_mapping.csv`
- `cell_nuclear_summary.csv`
- `cell_mask_correspondence.csv`
- `segmentation_mismatch.csv`
- `field_summary.csv`
- `density_stratified_summary.csv`
- `cell_pair_nuclear_status_concordance.csv`
- `input_inventory.csv`
- `summary.json`
- `analysis_summary.md`
- `plots/*.svg`
- `qc_overlays/*.png`

`segmentation_mismatch.csv` contains every nucleus that fails strict consensus
and includes separate `actionable_mismatch` and `review_recommended` flags.

## Validation and limitations

The run validates row-count conservation, unique nucleus/cell identities,
fraction bounds, complete field coverage, and consistency between strict-match
and review counts. The outputs measure internal agreement among predicted masks;
they do not estimate sensitivity, specificity, or biological accuracy without a
manual reference set.
