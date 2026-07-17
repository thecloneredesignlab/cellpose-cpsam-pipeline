# SUM159 CellposeSAM Pipeline

This project processes SUM159 Incucyte multichannel time-course images. It performs cell, dead-signal, and nucleus segmentation; multichannel cell-state fusion; and nucleus-supported sensitivity analyses. The production environment requires Cellpose `4.2.1.1`. The raw input root must contain matching fields in at least four directories: `Brightfield/`, `Combined/`, `Dead/`, and `Nuclei/`.

## 1. Complete Workflow Overview

The following sections describe the full workflow launched by the production HPC submission. This overview does not list internal code calls. Instead, each step explains what is read, what is done, what is produced, and why the result is needed.

### Step 1: Input preflight and task-list construction

- **Reads:** TIFF images and filenames from the four raw channel directories. The current production configuration skips `Dead_Uncalibrated/`.
- **Process:** Check the number of channel directories, total image count, empty channels, one-to-one Dead/Combined pairing, and duplicate field keys. Build separate task lists for nucleus preprocessing, main segmentation, and field-level fusion.
- **Produces:** Channel- and field-indexed task lists plus a per-channel image-count report.
- **Purpose:** Slurm arrays require stable, unique task indices. Preflight validation detects missing images, duplicates, and channel mismatches before a large GPU workflow is submitted.

### Step 2: Cohort-wide Dead and Combined-blue intensity calibration

- **Reads:** All paired Dead images and the blue and blue-excess signals from the corresponding Combined images.
- **Process:** Scan the cohort in shards, calculate robust intensity quantiles, backgrounds, histograms, cross-channel correlations, and image-level outliers, and then merge the shards into global and per-image calibration values.
- **Produces:** A Dead/Combined-blue calibration JSON, per-image calibration CSV, outlier table, global histograms, and a calibration QC figure.
- **Purpose:** Exposure and background drift vary across wells and time points. Cohort calibration places every image on a comparable scale and provides fixed intensity anchors for the Dead-primary and Combined-blue consensus logic.

### Step 3: Nuclei pre-segmentation

- **Reads:** Raw nuclear fluorescence images from `Nuclei/`.
- **Process:** Segment nuclei with the production CellposeSAM profile while preserving both a high-recall nucleus-extent mask and an intensity-supported nucleus-core mask.
- **Produces:** A nucleus-extent mask, nucleus-core mask, segmentation metadata, segmentation summary, and QC overlay for every field.
- **Purpose:** Extent masks support nucleus counts and morphology measurements. Core masks are more robust to overexposure and provide the nuclear evidence used for density calls, cell-boundary support, fusion classification, and later nucleus-splitting analyses.

### Step 4: High-density field calling

- **Reads:** Nuclei masks from Step 3, preferring nucleus-core masks when available.
- **Process:** Calculate nucleus count, nuclear mask fraction, and median nearest-neighbor distance for every field, then apply the retained production thresholds.
- **Produces:** `density_calls.csv`, with density metrics and a `high_density` flag for every field.
- **Purpose:** High-density Brightfield and Combined images require different segmentation profiles from ordinary-density images. The density table allows the main segmentation stage to select the appropriate profile per field.

### Step 5: Main Brightfield, Combined, and Dead segmentation

- **Reads:** Raw Brightfield, Combined, and Dead images; the calibration artifacts from Step 2; the density table from Step 4; and the Combined image paired with each Dead image.
- **Process:**
  - Select ordinary- or high-density CellposeSAM profiles for Brightfield and Combined.
  - Segment both the calibrated Dead-primary signal and the Combined-blue signal. Retain matched objects, and rescue blue-only objects only when the raw Dead image provides sufficient evidence.
  - Record the model, preprocessing, thresholds, object counts, and other provenance for each image.
- **Produces:** Brightfield, Combined, and Dead instance masks, per-image metadata, segmentation summaries, and QC overlays. Together with the Nuclei results from Step 3, this completes the four-channel base segmentation.
- **Purpose:** These four instance-mask sets form the shared spatial basis for cell counting, state classification, nucleus-to-cell matching, and morphology analysis. Dead consensus reduces single-channel false negatives while constraining Combined-blue false positives.

### Step 6: Dead-consensus merge and completeness validation

- **Reads:** Per-field Dead-consensus masks, field summaries, object-provenance records, and blue-only candidate records.
- **Process:** Verify that all expected fields completed, reject duplicate or unexpected keys, and merge all field-level outputs.
- **Produces:** Cohort-level Dead segmentation and field summaries, object provenance, a blue-only candidate table, aggregate JSON statistics, and a success marker.
- **Purpose:** Fusion classification requires both the final Dead mask and proof that every field is complete. Object provenance explains whether each final dead object came from the Dead-primary signal, a two-channel match, or a Combined-blue rescue.

### Step 7: Post-segmentation field manifest

- **Reads:** Four-channel raw images, four-channel instance masks, nucleus-core masks, Dead summaries, and the canonical field-key list.
- **Process:** Index the large result tree once, validate every required input for every field, and write one compact JSON record per field.
- **Produces:** `field_manifest.tsv`, `records/<well>/<field>.json`, a manifest summary, and a success marker.
- **Purpose:** The production result tree contains many files. Re-scanning it in every array task would be slow and could select an incorrect same-named file. The manifest fixes the exact input paths once for all post-segmentation stages.

### Step 8: Multichannel fusion classification on the original cell-mask branch

- **Reads:** The Combined raw image and Combined cell mask, Brightfield mask, Dead-consensus mask, Nuclei extent/core masks, and the field record from Step 7.
- **Process:** Use Combined cell instances as the primary objects, match Brightfield, Dead, and Nuclei evidence to each cell, and extract multichannel per-cell features. The RGB baseline retains `transitional` and `uncertain` as intermediate labels, while final fusion states are currently `live`, `dead`, or `artifact`.
- **Produces:** Per-cell feature tables, per-cell state tables, per-field summaries, a cohort cell-count summary, failure records, and state QC overlays under `classification_fusion/`.
- **Purpose:** A single-channel color threshold is not sufficient for stable cell-state classification. Fusion combines morphology, dead-signal, and nuclear evidence on the same cell instance and is the primary cell-state result.

### Step 9: Nucleated-only sensitivity branch

- **Reads:** Original Brightfield and Combined cell masks, Nuclei core/extent masks, and the field manifest.
- **Process:** Without modifying the original production masks, filter Brightfield and Combined instances to retain only cells with nuclear evidence. Repeat multichannel fusion classification on this branch.
- **Produces:** Filtered cell masks, filter decisions, and field summaries under `nucleated_only/`, plus a second fusion result under `classification_fusion_nucleated_only/`.
- **Purpose:** This branch measures the effect of anuclear debris and boundary artifacts on cell counts and state fractions. It is a sensitivity-analysis layer and does not replace the original production branch.

### Step 10: Strict shape-aware nucleus splitting and final QC

- **Reads:** Original Nuclei extent/core masks, original or nucleated-only cell masks, fusion results, raw Nuclei images, and the field manifest.
- **Process:** Attempt marker-watershed splitting only for large, shape-suspicious nuclei with two stable fluorescence peaks. Validate foreground conservation, peak stability, shape improvement, and cell support. Process the original and nucleated-only branches independently.
- **Produces:** Candidate nucleus masks, field summaries, split-event tables, diagnostics, aggregate JSON/Markdown reports, QC images, and contact sheets under `shape_strict/` and `shape_strict_nucleated_only/`.
- **Purpose:** Cellpose can merge adjacent nuclei into one instance. The strict split result measures the effect of merged nuclei on nucleus counts, morphology, and multinucleation analyses. It remains a separate result layer and never overwrites the original nucleus masks.

The main production output layout is:

```text
full_fusion_shape_strict_<timestamp>/
  Brightfield/                         Brightfield segmentation
  Combined/                            Combined segmentation
  Dead/                                Dead/Combined-blue consensus segmentation
  Nuclei/                              Nucleus-extent and nucleus-core masks
  qc/density_calls.csv                 Per-field density calls
  workflow_status/postsegmentation_manifest/
  classification_fusion/              Fusion using original cell masks
  nucleated_only/                      Cell masks retaining nuclear evidence
  classification_fusion_nucleated_only/
  shape_strict/                        Strict splitting on the original branch
  shape_strict_nucleated_only/         Strict splitting on the nucleated-only branch
  logs/                                Slurm stdout/stderr
```

## 2. HPC Submission Commands

The production entry point contains the current server defaults for the project, input, and result directories. Before the first run, or whenever storage locations change, verify `BASE`, `PROJECT_DIR`, `INPUT_ROOT`, and `RESULTS_ROOT` at the top of the entry point. The current production configuration expects `108800` images across four channels and skips `Dead_Uncalibrated/`.

### 2.1 Validate tasks and dependencies without submitting jobs

```bash
cd /share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1/cellpose-cpsam-pipeline_v2

DRY_RUN_SUBMIT=1 \
bash cellpose_pipeline/hpc/submit_full_fusion_production.sh
```

The dry run checks input counts, task lists, resources, and the complete dependency graph. It does not call `sbatch`.

### 2.2 Submit the complete production workflow

```bash
cd /share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1/cellpose-cpsam-pipeline_v2

bash cellpose_pipeline/hpc/submit_full_fusion_production.sh
```

Do not submit the internal orchestrator or workers directly. The production entry point creates the complete `afterok` dependency graph and prints every stage-level job ID. Results are written by default to:

```text
<RESULTS_ROOT>/full_fusion_shape_strict_<YYYYMMDD_HHMMSS>/
```

Inspect queue and accounting state with:

```bash
squeue -u "$USER"

sacct -j <job_id> \
  --format=JobID,JobName%35,State,ExitCode,Elapsed,Timelimit,AllocTRES%50
```

Disappearance from `squeue` alone does not prove success. Final verification must combine `sacct`, stage success markers, summary tables, and expected output counts.

## 3. Analysis Scripts

Analysis scripts are stored in `cellpose_pipeline/scripts/analysisi/`. Their numbering is independent of the production and parameter-calibration sequences. The following examples assume execution from the repository root:

```bash
conda activate cellpose

export PROJECT_DIR="$(pwd)"
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export INPUT_ROOT=/path/to/20260626_SUM159_AC_Exp1_SeparateImages
export RUN_ROOT=/path/to/full_fusion_shape_strict_<timestamp>
export ANALYSIS_ROOT="$RUN_ROOT/analysis"
export KMP_DUPLICATE_LIB_OK=TRUE
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cpsam_mpl_analysis"
```

### 3.1 `01_inventory.py`: Inventory raw images

- **Function:** Recursively scan TIFF files, parse well, site, day, and time metadata, and summarize the image distribution.
- **Reads:** A raw image directory.
- **Produces:** `image_inventory.csv`, optionally including file size.
- **Use:** Validate data completeness and provide the sampling frame for representative-image selection.

```bash
"$PYTHON_BIN" -I cellpose_pipeline/scripts/analysisi/01_inventory.py \
  --raw-dir "$INPUT_ROOT" \
  --out "$ANALYSIS_ROOT/image_inventory.csv" \
  --include-size
```

### 3.2 `02_classify_cell_states.py`: Classify one image

- **Function:** Extract color, area, and related features from one RGB image and its instance mask, then assign `live`, `dead`, `transitional`, `artifact`, or `uncertain` states.
- **Reads:** One raw RGB image and one same-sized instance mask.
- **Produces:** A per-cell feature CSV, prediction CSV, image-level summary CSV, and state overlay.
- **Use:** Inspect one image or support legacy single-channel workflows. Full production results should use multichannel fusion classification instead.

```bash
"$PYTHON_BIN" -I cellpose_pipeline/scripts/analysisi/02_classify_cell_states.py \
  --raw-image /path/to/image.tif \
  --mask-image /path/to/image_cp_masks.tif \
  --out-dir "$ANALYSIS_ROOT/single_image_classification"
```

### 3.3 `03_make_missing_workflow_overlays.py`: Backfill missing QC overlays

- **Function:** Recombine raw images, segmentation masks, and existing state predictions to create missing segmentation or classification overlays. Existing overlays are skipped by default.
- **Reads:** A standalone workflow result containing `segmentations/` and `classification/`, plus a raw-image manifest.
- **Produces:** Segmentation and/or classification overlay PNGs.
- **Use:** Repair missing QC images from interrupted or earlier runs without repeating segmentation or classification.

```bash
"$PYTHON_BIN" -I cellpose_pipeline/scripts/analysisi/03_make_missing_workflow_overlays.py \
  --run-dir /path/to/standalone_workflow_run \
  --manifest /path/to/image_manifest.csv \
  --overlay-type both
```

Add `--force --stem <image-stem>` to regenerate one field.

### 3.4 `04_plot_well_counts_over_time.py`: Plot well-level cell counts over time

- **Function:** Select the primary fusion, nucleated-only sensitivity, or legacy Combined RGB-only classification; validate count identities and experiment completeness; aggregate imaging sites by well and time; and plot live/dead trajectories.
- **Reads:** By default, `classification_fusion/summaries/cell_count_summary.csv` under a production run. It can also read `classification_fusion_nucleated_only/` or legacy `Combined/classification/predictions/*_summary.csv` outputs.
- **Produces:** Per-image counts, well-by-time aggregated counts with treatment metadata, and PNG/PDF time-course figures in a branch-specific output directory.
- **Use:** Compare cell-state trajectories while preserving the physical 8-by-12 plate arrangement. The plate layout labels doxorubicin concentration, ploidy, cyclophosphamide condition, and replicate. Columns 1 and 12 are retained as empty plate positions; `--layout compact` omits them.

```bash
"$PYTHON_BIN" -I cellpose_pipeline/scripts/analysisi/04_plot_well_counts_over_time.py \
  "$RUN_ROOT" \
  --branch fusion \
  --layout plate \
  --y-axis shared \
  --strict-completeness \
  --expected-timepoints 85 \
  --expected-sites 4
```

`--branch auto` is the default and prefers the primary fusion summary. Use
`--branch fusion-nucleated-only` for the sensitivity branch or
`--branch legacy-combined` only when reproducing the earlier RGB-only result.
The default machine-readable plate map is
`cellpose_pipeline/scripts/analysisi/resources/SUM159_AC_Experiment1_PlateMap.csv`;
the original Excel workbook is retained beside it as the source document.

### 3.5 `05_analyze_nuclear_cell_alignment.py`: Analyze nuclear morphology and nucleus-to-cell alignment

- **Function:** Measure nuclear area, shape, and intensity; match nuclei independently to Brightfield and Combined cell instances; and identify unmatched nuclei, multinucleated cells, nuclear-to-cell ratios, nuclear-to-cytoplasm ratios, and cross-channel mismatch classes.
- **Reads:** A completed production result root and the four-channel raw input root.
- **Produces:** Nucleus-, cell-, field-, and density-stratified CSV files; `summary.json`; `analysis_summary.md`; statistical plots; and per-field QC overlays.
- **Use:** Quantify nucleus-segmentation reliability and cell-boundary agreement, and provide the baseline for cell-boundary and nucleus-splitting calibration.

```bash
"$PYTHON_BIN" -I cellpose_pipeline/scripts/analysisi/05_analyze_nuclear_cell_alignment.py \
  --run-root "$RUN_ROOT" \
  --input-root "$INPUT_ROOT" \
  --out-dir "$ANALYSIS_ROOT/nuclear_cell_alignment"
```

### 3.6 `06_audit_largetest_full_pipeline.py`: Audit a completed result

- **Function:** Check four-channel raw images, masks, metadata, overlays, nucleus-core masks, the density table, fusion results, shape-strict outputs, and optional logs for fatal errors.
- **Reads:** The raw input root, completed result root, expected field count per channel, and an optional pipeline log.
- **Produces:** `final_audit.json` and `final_audit.md`. The command exits nonzero when a required check fails.
- **Use:** Confirm field counts and cross-stage consistency before downstream statistical analysis. The default expected count of `20` is only appropriate for the large-test subset; full data must provide the true field count explicitly.

```bash
"$PYTHON_BIN" -I cellpose_pipeline/scripts/analysisi/06_audit_largetest_full_pipeline.py \
  --input-root "$INPUT_ROOT" \
  --run-root "$RUN_ROOT" \
  --expected-per-profile <expected-fields-per-channel> \
  --pipeline-log /path/to/pipeline.log
```

### 3.7 `07_plot_dose_response_curves.py`: Compare 2N and 4N Hill responses

- **Function:** Convert the live-cell time courses into one normalized response per well, fit bounded four-parameter Hill curves, and compare 2N with 4N separately for doxorubicin alone and doxorubicin plus cyclophosphamide.
- **Reads:** The branch-specific `well_time_cell_state_counts.csv` produced by `04_plot_well_counts_over_time.py`, supplied directly or discovered under a production run.
- **Normalization:** The default `auc` metric divides each well's live-cell trajectory by its own starting count, integrates that trajectory over time, and divides the result by the 0 nM well from the same plate-row replicate. Thus every replicate's vehicle response is 1 before fitting. `--metric endpoint` applies the same baseline and paired-vehicle normalization to the endpoint or a configurable late-time window.
- **Produces:** Exactly two distinct dose-response plots, each as PNG and PDF: one for doxorubicin alone and one for doxorubicin plus cyclophosphamide. Each compares 2N with 4N, shows individual replicate wells and dose means, and annotates EC50, Hill slope, 95% confidence intervals, and R-squared. Per-well normalized responses, dose summaries, and fit parameters are also written as CSV files.
- **Interpretation:** Fits use both plate-row replicates at every dose. The confidence intervals quantify nonlinear-fit uncertainty but should be interpreted cautiously because there are only two replicate series per ploidy and treatment condition.

```bash
"$PYTHON_BIN" -I cellpose_pipeline/scripts/analysisi/07_plot_dose_response_curves.py \
  "$RUN_ROOT" \
  --branch fusion \
  --metric auc
```

To fit the final live-cell response instead, use `--metric endpoint`. Add
`--endpoint-hours <hours>` to choose an earlier endpoint or
`--endpoint-window-hours <hours>` to average a late-time window.

## 4. Parameter-Calibration Scripts

Parameter-calibration scripts are stored in `cellpose_pipeline/scripts/Parameter_calibration/`. Their numbering follows the logical sequence of data preparation, broad screening, focused validation, and final QC. They are not called by the default production workflow. Use a new output directory for each calibration run, and update production configuration only after reviewing the calibration results.

Define common paths first:

```bash
export PARAM_DIR="$PROJECT_DIR/cellpose_pipeline/scripts/Parameter_calibration"
export TUNING_ROOT=/path/to/parameter_calibration_results
export BASELINE_RUN="$RUN_ROOT"
export LEGACY_BASELINE_RUN=/path/to/legacy_workflow_run
mkdir -p "$TUNING_ROOT"
```

`BASELINE_RUN` refers to the current production layout: `<run>/<channel>/segmentations/`. Sections 4.7-4.13 describe retained early large-test tools that require a legacy layout containing either `LEGACY_BASELINE_RUN/segmentation_summary.csv` or `LEGACY_BASELINE_RUN/segmentations/<channel>/`. Do not pass a current production directory directly to those legacy interfaces.

Display every option for any script with:

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/<script>.py" --help
```

### 4.1 `01_select_annotation_images.py`: Select images for manual annotation

- **Function:** Select representative TIFFs across wells and time strata, copy or symlink them into an annotation directory, and write a manifest.
- **Purpose:** Prevent annotation from being concentrated in one time point or density regime and ensure that reviewed/training images cover the major experimental conditions.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/01_select_annotation_images.py" \
  --raw-dir "$INPUT_ROOT" \
  --out-dir "$TUNING_ROOT/annotation_images" \
  --manifest "$TUNING_ROOT/annotation_manifest.csv" \
  --n-images 96
```

### 4.2 `02_prepare_training_split.py`: Build train/test splits

- **Function:** Read annotated images and Cellpose `_seg.npy` files, split them reproducibly into train and test directories, and write a split manifest.
- **Purpose:** Create a reproducible held-out test set and avoid data leakage caused by manually moving files.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/02_prepare_training_split.py" \
  --annotation-dir "$TUNING_ROOT/annotation_images" \
  --train-dir "$TUNING_ROOT/train" \
  --test-dir "$TUNING_ROOT/test" \
  --manifest "$TUNING_ROOT/train_test_split.csv"
```

### 4.3 `03_select_off_the_shelf_eval_images.py`: Select an off-the-shelf evaluation set

- **Function:** Sample representative images across wells and time points from an inventory and write an evaluation directory plus manifest.
- **Purpose:** Compare untrained CellposeSAM configurations on a fixed sample rather than changing test images while reviewing results.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/03_select_off_the_shelf_eval_images.py" \
  --inventory "$ANALYSIS_ROOT/image_inventory.csv" \
  --out-dir "$TUNING_ROOT/off_the_shelf_eval" \
  --manifest "$TUNING_ROOT/off_the_shelf_eval_manifest.csv" \
  --n-images 192
```

### 4.4 `04_extract_circled_dead_annotation.py`: Extract circled dead-cell annotations

- **Function:** Detect orange circles in a manually edited image, locate the corresponding mask, and record raw RGB features, match status, and any usable training label.
- **Purpose:** Convert manual circles into structured validation data while distinguishing matched dead masks, segmentation misses, and neighboring-object mismatches.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/04_extract_circled_dead_annotation.py" \
  --annotated-image /path/to/circled_image.tif \
  --raw-image /path/to/raw_image.tif \
  --mask-image /path/to/mask.tif \
  --out "$TUNING_ROOT/circled_dead_cells.csv"
```

### 4.5 `05_tune_cpsam_parameters.py`: Screen base CellposeSAM parameters

- **Function:** Screen or validate model, diameter, threshold, and preprocessing combinations for Brightfield, Dead, and Nuclei. Write candidate masks, overlays, configurations, and summary tables.
- **Purpose:** Establish the base model and parameter ranges for each channel.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/05_tune_cpsam_parameters.py" \
  --input-root "$INPUT_ROOT" \
  --out-root "$TUNING_ROOT/cpsam_parameters" \
  --stage screen \
  --use-gpu
```

After screening, run a larger validation with `--stage validate --config-json /path/to/selected_configs.json`.

### 4.6 `06_cross_channel_parameter_analysis.py`: Score candidates across channels

- **Function:** Read one or more tuning summaries, construct matched Brightfield/Dead/Nuclei combinations, calculate cross-channel constraints and ranks, and render QC contact sheets.
- **Purpose:** Avoid selecting parameters from single-channel object counts alone and require biologically plausible relationships across signals.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/06_cross_channel_parameter_analysis.py" \
  --input-root "$INPUT_ROOT" \
  --summary-csv /path/to/screen/summary.csv /path/to/validate/summary.csv \
  --out-root "$TUNING_ROOT/cross_channel_scoring"
```

### 4.7 `07_evaluate_largetest_recommended_profiles.py`: Evaluate recommended profiles on the large test

- **Function:** Measure object counts, spatial matching, density-stratified behavior, and flagged fields across matched Brightfield, Dead, and Nuclei triplets, and render QC panels.
- **Purpose:** Determine whether parameters recommended from a small sample generalize to a larger and more difficult dataset.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/07_evaluate_largetest_recommended_profiles.py" \
  --input-root "$INPUT_ROOT" \
  --run-dir "$LEGACY_BASELINE_RUN" \
  --out-dir "$TUNING_ROOT/largetest_evaluation"
```

### 4.8 `08_largetest_retune.py`: Retune channel profiles on the large test

- **Function:** Use an existing run as a baseline, rerun selected channel candidates on the large test, and save masks, prepared-image previews, overlays, scores, and a report.
- **Purpose:** Optimize parameters for low-signal, high-density, or anomalous fields that were not represented in the smaller screen.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/08_largetest_retune.py" \
  --input-root "$INPUT_ROOT" \
  --baseline-run "$LEGACY_BASELINE_RUN" \
  --out-root "$TUNING_ROOT/largetest_retune" \
  --stage round1 \
  --use-gpu
```

### 4.9 `09_largetest_dead_global_filter_tune.py`: Tune Dead global scaling and object filters

- **Function:** Run Dead candidates with fixed global intensity scaling, then filter objects using area, aspect ratio, raw-signal increments, and SNR. Save candidate masks, object statistics, scores, and a report.
- **Purpose:** Reduce background false positives caused by local normalization while preserving real low-signal dead objects.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/09_largetest_dead_global_filter_tune.py" \
  --input-root "$INPUT_ROOT" \
  --baseline-run "$LEGACY_BASELINE_RUN" \
  --out-root "$TUNING_ROOT/dead_global_filter" \
  --use-gpu
```

### 4.10 `10_score_dead_global_filter_candidates.py`: Rescore Dead candidates without inference

- **Function:** Read saved candidate masks and object statistics from Section 4.9, apply new object-filter combinations, and update scores, key-field overlays, and the report.
- **Purpose:** Change filtering thresholds without repeating expensive Cellpose GPU inference.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/10_score_dead_global_filter_candidates.py" \
  --input-root "$INPUT_ROOT" \
  --baseline-run "$LEGACY_BASELINE_RUN" \
  --run-root "$TUNING_ROOT/dead_global_filter/dead_global_filter_round1"
```

### 4.11 `11_score_dead_a11_target_filters.py`: Tune A11-targeted false-positive filters

- **Function:** Perform a finer object-filter search on existing Dead candidates, target a specified A11 object count, and simultaneously audit other key and high-density fields.
- **Purpose:** Correct A11-like anomalous background without overfitting one well and degrading the rest of the cohort.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/11_score_dead_a11_target_filters.py" \
  --input-root "$INPUT_ROOT" \
  --baseline-run "$LEGACY_BASELINE_RUN" \
  --candidate-run-root "$TUNING_ROOT/dead_global_filter/dead_global_filter_round1" \
  --out-root "$TUNING_ROOT/dead_a11_filters"
```

### 4.12 `12_tune_combined_bf_like_preprocess.py`: Tune BF-like preprocessing for Combined images

- **Function:** Transform Combined RGB images into alternative BF-like representations and compare masks, object counts, baseline relationships, and QC figures.
- **Purpose:** Combined images contain several color signals, and a simple grayscale transform can alter cell boundaries. This screen selects a more stable morphology input.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/12_tune_combined_bf_like_preprocess.py" \
  --input-root "$INPUT_ROOT/Combined" \
  --baseline-run "$LEGACY_BASELINE_RUN" \
  --out-root "$TUNING_ROOT/combined_bf_like" \
  --use-gpu
```

### 4.13 `13_tune_high_density_spatial_profiles.py`: Early high-density spatial screen

- **Function:** Read nucleus masks from `segmentations/Nuclei/` in a legacy workflow, call high-density fields, and compare Brightfield/Combined candidates using nuclear spatial evidence.
- **Purpose:** Evaluate under-segmentation and merged cell boundaries in dense fields.
- **Note:** This is a retained early all-in-one screen. Sections 4.19-4.21 are preferred for current production high-density calibration.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/13_tune_high_density_spatial_profiles.py" \
  --input-root "$INPUT_ROOT" \
  --nuclei-run "$LEGACY_BASELINE_RUN" \
  --out-root "$TUNING_ROOT/high_density_spatial" \
  --use-gpu
```

### 4.14 `14_tune_nuclei_segmentation.py`: Tune Nuclei segmentation

- **Function:** Screen inference and mask-reconstruction configurations separately, reuse Cellpose flows, and compare extent/core masks, object morphology, and multichannel center matching.
- **Purpose:** Preserve high recall in nucleus extents while maintaining nucleus-core robustness to overexposure, rather than optimizing object count alone.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/14_tune_nuclei_segmentation.py" \
  --input-root "$INPUT_ROOT" \
  --baseline-run "$BASELINE_RUN" \
  --out-dir "$TUNING_ROOT/nuclei_tuning" \
  --use-gpu \
  --save-masks \
  --save-previews
```

### 4.15 `15_diagnose_registration_and_refine_cell_masks.py`: Diagnose registration and generate cell-boundary candidates

- **Function:** Measure global Nuclei-to-Brightfield/Combined offsets and create local cell-mask repair candidates only where nucleus-core evidence is strong. The script does not split or merge cell labels.
- **Purpose:** Separate true channel registration errors from local cell-boundary errors and generate conservative candidates for independent scoring.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/15_diagnose_registration_and_refine_cell_masks.py" \
  --run-root "$BASELINE_RUN" \
  --out-root "$TUNING_ROOT/nucleus_aware_cell_refinement"
```

### 4.16 `16_score_nucleus_aware_cell_refinement.py`: Score cell-boundary candidates with independent guardrails

- **Function:** Compare candidates with the baseline nucleus alignment while also enforcing raw-image edge support, label connectivity, area stability, and Brightfield/Combined agreement.
- **Purpose:** Alignment improvement is partly circular because nucleus cores construct the candidates. Independent guardrails prevent unrealistic cell boundaries from being accepted only because they contain nuclei better.
- **Prerequisite:** Run the analysis in Section 3.5 for every candidate first. The corresponding HPC calibration workflow is normally used for large-scale validation.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/16_score_nucleus_aware_cell_refinement.py" \
  --baseline-run-root "$BASELINE_RUN" \
  --baseline-summary "$ANALYSIS_ROOT/nuclear_cell_alignment/summary.json" \
  --candidates-root "$TUNING_ROOT/nucleus_aware_cell_refinement/candidates" \
  --input-root "$INPUT_ROOT" \
  --out-dir "$TUNING_ROOT/nucleus_aware_cell_refinement/scoring"
```

### 4.17 `17_render_nucleus_aware_cell_refinement_qc.py`: Render before/after cell-boundary QC

- **Function:** Use repair events to select changed regions and render side-by-side crops of baseline and candidate cell masks.
- **Purpose:** Visually confirm that a recommended boundary change follows plausible cell structure before accepting its parameters.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/17_render_nucleus_aware_cell_refinement_qc.py" \
  --baseline-run-root "$BASELINE_RUN" \
  --candidate-run-root /path/to/selected_candidate \
  --input-root "$INPUT_ROOT" \
  --repair-events "$TUNING_ROOT/nucleus_aware_cell_refinement/repair_events.csv" \
  --candidate-tag <candidate-tag> \
  --out-dir "$TUNING_ROOT/nucleus_aware_cell_refinement/qc"
```

### 4.18 `18_score_shape_aware_nucleus_splits.py`: Score nucleus-splitting candidates

- **Function:** Compare shape-aware candidates with the baseline using nucleus-count changes, peak stability, shape improvement, multinucleation rate, cell support, and cross-method agreement.
- **Purpose:** Constrain over-splitting with several conservative checks when complete manual truth is unavailable. A candidate remains a sensitivity layer and must not replace production masks.
- **Prerequisite:** Candidate masks and nucleus-to-cell analysis results must already exist for every candidate. The numbered HPC calibration stages 16-19 normally perform this screen and validation.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/18_score_shape_aware_nucleus_splits.py" \
  --baseline-run-root "$BASELINE_RUN" \
  --baseline-summary "$ANALYSIS_ROOT/nuclear_cell_alignment/summary.json" \
  --screen-root /path/to/shape_screen \
  --candidates-root /path/to/shape_candidates \
  --out-dir "$TUNING_ROOT/shape_split_scoring"
```

### 4.19 `19_tune_high_density_bf_combined.py`: Run current high-density Brightfield/Combined candidates

- **Function:** Run configured Brightfield and Combined candidates only on fields marked high density in `density_calls.csv`. Save masks, complete inference configuration, and field-level metrics without modifying Nuclei masks.
- **Purpose:** Restrict expensive GPU inference to fields that need high-density profiles and provide reproducible inputs for pairwise scoring.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/19_tune_high_density_bf_combined.py" \
  --input-root "$INPUT_ROOT" \
  --nuclei-run-root "$BASELINE_RUN" \
  --density-calls "$BASELINE_RUN/qc/density_calls.csv" \
  --candidate-config cellpose_pipeline/configs/high_density_bf_combined_candidates.json \
  --out-root "$TUNING_ROOT/high_density_bf_combined/screen" \
  --use-gpu
```

### 4.20 `20_score_high_density_bf_combined.py`: Score high-density candidate pairs

- **Function:** Pair Brightfield and Combined candidates and compare nucleus-core support, shape artifacts, foreground/boundary agreement, nucleus-multiplicity agreement, and calibration/validation stability.
- **Purpose:** Production requires a coordinated Brightfield/Combined pair, not two individually optimal candidates that disagree with one another.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/20_score_high_density_bf_combined.py" \
  --screen-root "$TUNING_ROOT/high_density_bf_combined/screen" \
  --out-dir "$TUNING_ROOT/high_density_bf_combined/screen/scoring"
```

The primary decision artifact is `selected_pair.json`. The baseline pair is retained when no candidate clears all guardrails.

### 4.21 `21_render_high_density_bf_combined_qc.py`: Render high-density visual QC

- **Function:** Read `selected_pair.json`, render baseline-versus-selected Brightfield and Combined comparisons for every high-density field, and build an aggregate contact sheet.
- **Purpose:** Review boundary changes and cross-channel disagreement before updating the production configuration.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/21_render_high_density_bf_combined_qc.py" \
  --screen-root "$TUNING_ROOT/high_density_bf_combined/screen" \
  --scoring-dir "$TUNING_ROOT/high_density_bf_combined/screen/scoring" \
  --out-dir "$TUNING_ROOT/high_density_bf_combined/qc"
```

These three stages can also be submitted as one dependency-linked HPC workflow with `cellpose_pipeline/hpc/Parameter_calibration/20_submit_high_density_bf_combined_optimization.sh`.

### 4.22 `22_tune_combined_blue_segmentation.py`: Screen Combined-blue dead-signal candidates

- **Function:** Run coarse or fine candidates on cohort-calibrated Combined blue/blue-excess signals and compare Dead-baseline matching, raw-signal support, and cell support.
- **Purpose:** Select an auxiliary configuration that can recover Dead false negatives without generating excessive blue-background false positives.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/22_tune_combined_blue_segmentation.py" \
  --input-root "$INPUT_ROOT" \
  --baseline-run "$BASELINE_RUN" \
  --calibration-json "$BASELINE_RUN/dead_preprocess_calibration/final/dead_combined_blue_calibration.json" \
  --out-root "$TUNING_ROOT/combined_blue_screen" \
  --candidate-set coarse \
  --use-gpu
```

Outputs include field metrics, a candidate summary, and `best_candidate.json`.

### 4.23 `23_tune_dead_calibrated_segmentation.py`: Screen calibrated Dead candidates

- **Function:** Run Dead candidates using the global calibration JSON and per-image calibration map, then compare object retention, cell/nucleus support, and baseline matching.
- **Purpose:** Verify that the selected calibration and Dead thresholds remain stable across the full cohort.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/23_tune_dead_calibrated_segmentation.py" \
  --input-root "$INPUT_ROOT" \
  --baseline-run "$BASELINE_RUN" \
  --calibration-json "$BASELINE_RUN/dead_preprocess_calibration/final/dead_combined_blue_calibration.json" \
  --calibration-map "$BASELINE_RUN/dead_preprocess_calibration/final/dead_combined_blue_image_calibration_map.csv" \
  --out-root "$TUNING_ROOT/dead_calibrated_screen" \
  --use-gpu
```

Outputs include field metrics, a candidate summary, and `best_candidate.json`.

### 4.24 `24_render_dead_consensus_qc.py`: Render Dead-consensus QC

- **Function:** Combine the raw Dead image, Combined-blue signal, primary Dead mask, Combined-blue candidates, and final consensus mask into per-field QC panels and paginated contact sheets.
- **Purpose:** Confirm that rescued objects have raw Dead evidence and identify systematic consensus false positives or false negatives.

```bash
"$PYTHON_BIN" -I "$PARAM_DIR/24_render_dead_consensus_qc.py" \
  --input-root "$INPUT_ROOT" \
  --dead-run "$BASELINE_RUN/Dead" \
  --out-dir "$TUNING_ROOT/dead_consensus_qc" \
  --page-size 4
```

Do not promote calibration results into production configuration until cross-field scoring, independent guardrails, and visual QC are complete. Historical screen directories and candidate masks must not directly replace production results.
