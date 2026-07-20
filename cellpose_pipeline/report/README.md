# Full-cohort results analysis report

`generate_results_analysis_report.py` builds the English analysis guide for a
completed production result tree. Scientific outputs are read-only. Report
rendering and HTML assembly are performed locally; the HPC result tree is used
only as a source for the minimal six-sample input snapshot.

## Report organization

The report contains:

- a technical and cohort-level summary;
- the production result-directory map and key machine-readable outputs;
- a detailed Methods chapter derived from the repository `README.md`;
- the recommended workflow diagram rendered from
  `docs/recommended_full_production_workflow.pdf` inside Methods;
- a single Figures chapter containing all result visuals.

The Figures chapter uses a fixed sample order: three high-density fields,
followed by three low-density fields. It first shows six per-sample
segmentation composites and then six per-sample classification/shape_strict
composites. A final cohort-level chart summarizes final Dead-object provenance.

Each segmentation composite contains:

1. nuclei extent and intensity-supported core;
2. Brightfield segmentation;
3. Combined segmentation;
4. Dead-primary candidates;
5. Combined-blue candidates;
6. final Dead consensus with provenance colors;
7. the nucleated-only mask filter.

Each classification/shape_strict composite contains the all-cell branch first
and the nucleated-only branch second:

1. all-cell fusion classification;
2. all-cell shape_strict baseline versus strict result;
3. nucleated-only fusion classification;
4. nucleated-only shape_strict baseline versus strict result.

Every result figure has a number, title, panel labels, color legend, and a
how-to-read explanation. Processing-step sections outside Figures contain text
and tables only. The workflow diagram is a numbered method diagram and is not
part of the result-figure sequence.

## Image quality and HTML behavior

QC boundaries, labels, and panel geometry are rendered directly from source
TIFFs and masks at 6x pixel density. The images are not enlarged copies of old
QC panels. High-resolution images use WebP quality 96 and carry 600 PPI
metadata. A bounded copy is stored in the portable artifact for validation;
the full 6x image is embedded into the final HTML.

The final HTML is self-contained and has no sibling-image dependency. It adds:

- a fixed left navigation panel on wide screens;
- automatic active-section highlighting;
- automatic expansion of the active navigation group and collapse of inactive
  groups;
- a narrow-screen Contents drawer;
- responsive image frames with internal scrollbars disabled;
- a click-to-inspect high-resolution image viewer for every QC figure. The
  normal page view keeps the full figure visible; clicking a figure opens the
  embedded 6x image in a lightbox with mouse-wheel zoom, button zoom, 1:1 pixel
  view, fit-to-screen reset, and drag-to-pan navigation.

## v3 Source Paths

The current v3 HPC checkout is:

```text
/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/cellpose-cpsam-pipeline-v3
```

The HPC production result tree documented by this report is:

```text
/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Cyclophosphamide/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/full_fusion_shape_strict_20260711_155940
```

The local report artifact root is:

```text
/Users/4482173/Documents/GitHub/cellpose-cpsam-pipeline-v3/results
```

## Minimal local input snapshot

The recommended local input root is:

```text
/Users/4482173/Documents/GitHub/cellpose-cpsam-pipeline-v3/results/full_fusion_shape_strict_20260711_155940/report_inputs_6samples/
```

It contains only:

- the 24 raw TIFFs for the six selected fields;
- the corresponding Nuclei, Brightfield, Combined, Dead, nucleated-only, and
  shape_strict masks;
- the corresponding classification predictions and Dead provenance tables;
- full-cohort summary tables required for report metrics and the provenance
  chart.

Production manifests can retain pre-move absolute paths. The generator first
checks the recorded path and then resolves the same relative result path or
`raw_input/<channel>/` path inside the compact local snapshot. shape_strict
paths are resolved before generic `Nuclei/` paths so strict masks cannot be
silently replaced by baseline masks.

## Generate and package locally

```bash
/Users/4482173/.pyenv/versions/CellPose/bin/python \
  cellpose_pipeline/report/generate_results_analysis_report.py \
  --run-root /Users/4482173/Documents/GitHub/cellpose-cpsam-pipeline-v3/results/full_fusion_shape_strict_20260711_155940/report_inputs_6samples \
  --artifact-json /tmp/RESULTS_ANALYSIS_GUIDE.artifact.json \
  --high-resolution-images-json /tmp/RESULTS_ANALYSIS_GUIDE.hq-images.json \
  --output-html /Users/4482173/Documents/GitHub/cellpose-cpsam-pipeline-v3/results/full_fusion_shape_strict_20260711_155940/RESULTS_ANALYSIS_GUIDE.html \
  --plugin-root /path/to/data-analytics-plugin \
  --sample-keys HD1_KEY HD2_KEY HD3_KEY LD1_KEY LD2_KEY LD3_KEY \
  --force
```

`--workflow-pdf` and `--methods-readme` default to the repository workflow PDF
and root README. `--debug-qc-dir` is optional and writes high-resolution review
copies; the HTML never depends on that directory. The temporary artifact and
high-resolution sidecar can be removed after the final HTML passes browser QA.

The command reports the selected field keys, canonical and high-resolution
image sizes, 6x render scale, 600 PPI metadata, image byte count, portable
artifact size, navigation counts, no-scroll image-frame counts, and whether the
high-resolution lightbox was enabled.

## d0 Dead-classification improvement report

`generate_dead_classification_improvement_report.py` converts
`DEAD_CLASSIFICATION_IMPROVEMENT_COMPARISON.md` and the complete 320-field d0
audit into an English, self-contained technical report. It rebuilds five
cohort-level charts directly from the saved CSV evidence, creates focused
six-panel QC figures for the fixed E2, F5, and H9 sentinel fields, and adds two
deterministic six-case galleries for live-override proxies and context-aware
death misses. The source segmentation, classification outputs, and QC overlays
are read-only.

```bash
/Users/4482173/.pyenv/versions/CellPose/bin/python -I \
  cellpose_pipeline/report/generate_dead_classification_improvement_report.py \
  --input-root /path/to/complete/SeparateImages \
  --result-root /path/to/complete/full_fusion_shape_strict_run \
  --timepoint d0 \
  --plugin-root /path/to/data-analytics-plugin \
  --force
```

`--input-root` and `--result-root` point to the complete experiment trees, not
to a separately copied d0 directory. The generator validates that the selected
`00d00h00m` raw images, original masks, nucleated-only masks, and nucleus-core
masks have the same field-key set. The two source-root arguments are optional
when rebuilding solely from an already completed audit, but they must be
provided together when source provenance is validated.

To build branch-aware field records for d0 directly from the complete trees:

```bash
python -I cellpose_pipeline/scripts/06_build_postsegmentation_field_manifest.py \
  --input-root /path/to/complete/SeparateImages \
  --run-root /path/to/complete/full_fusion_shape_strict_run \
  --timepoint d0 \
  --out-dir results/dead_d0_classification_audit/d0_field_manifest
```

The resulting records reference the original complete trees in place. No raw
images, masks, or result tables are copied into a d0-only source directory.

Default outputs are written under `results/dead_d0_classification_audit/`:

- `DEAD_CLASSIFICATION_IMPROVEMENT_REPORT.html`: portable report with fixed
  navigation, responsive layout, native charts and tables, and a high-resolution
  QC lightbox;
- `DEAD_CLASSIFICATION_IMPROVEMENT_REPORT.artifact.json`: validated canonical
  report artifact and bounded data snapshot;
- `DEAD_CLASSIFICATION_IMPROVEMENT_REPORT.build.json`: build receipt, dataset
  row counts, image checks, HTML enhancement counts, and chart map.

The generator discovers the most recently installed Data Analytics plugin when
`--plugin-root` is omitted. `--debug-figure-dir` optionally writes PNG review
copies, and `--high-resolution-images-json` optionally retains the temporary
image sidecar used during packaging. Neither is required by the final HTML.

### Current-run HPC layout

The same generator also detects a completed current-run audit root containing
the following directories:

```text
classification_original/
classification_nucleated_only/
automated_audit/
annotations/
detector_stress/
```

This layout produces a current-run technical report without requiring the
historical intermediate calibration directories. The report includes both
classification branches, cell-state composition, confirmed Death-object
relations, object-aware event composition, uncertainty, operational proxy
audit metrics, detector stress results, and E2/F5/H9 final QC examples. A full
historical audit root continues to produce the original multi-round comparison
report.

On the SUM159 HPC workflow, the single entry point is:

```bash
bash cellpose_pipeline/hpc/Parameter_calibration/25_dead_d0_classification_audit.sh
```

It embeds the per-field worker mode, validates all 320 fields, and generates
the artifact JSON, build receipt, and self-contained HTML only after the
classification, audit, annotations, stress test, and QC checks succeed.
