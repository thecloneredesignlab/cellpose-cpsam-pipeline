# Multimodal cell-state V4 workflow

`multimodal_cell_state_v4` is a new independent SUM159 cell-state axis. V1,
V2, and V3 result roots remain immutable audit records. V4 does not read,
modify, replace, or tune the current production classifier before its own model
is frozen.

## Scientific target

The primary endpoint is binary death status: `dead` versus `non_dead`, with
`uncertain_or_unreviewable` retained as an abstention. All death stages belong
to the same primary `dead` class. The secondary death-stage descriptor is
conditional on a primary dead call:

- `dead_marker_positive`;
- `dead_marker_weak_or_transition`;
- `dead_marker_negative_nucleus_lost_like`;
- `death_stage_indeterminate`.

The representation target is one broad death-enriched UMAP region. Internal
continuous geometry may reflect marker intensity, nuclear loss, and other
death-stage variation. The target is not three artificial death clusters.

## Allowed evidence and blinding

Before model freeze, V4 may read only the development universe and:

- Brightfield raw images;
- Nuclei raw fluorescence images;
- Dead raw fluorescence images;
- original Combined object masks;
- Nuclei extent masks, with the conservative core mask as an explicit fallback;
- development-well identity needed for grouped sampling and validation.

The existing Dead segmentation, Combined RGB, current labels/probabilities,
trajectory outputs, and heldout cells are forbidden. Raw Dead fluorescence is
an independent measurement, not an existing classifier output.

## Four feature blocks

The primary label-free representation has four blocks.

1. Shape: area, roundness, aspect ratio, extent, and solidity.
2. Brightfield: complementary object intensity, contrast, boundary, focus,
   edge, and texture summaries. A deterministic development-only
   absolute-Spearman filter removes later candidates at correlation 0.95 or
   above.
3. Nuclei support: supported absence, excess nucleus count, conditional
   nucleus-to-cell area fraction, largest-nucleus fraction, nucleus-area CV,
   small-fragment fraction, and spatial dispersion. Generic Nuclei GLCM or
   chromatin-texture claims are excluded.
4. Dead fluorescence: supported absence, cell-to-ring median and P90 contrast,
   within-object P90-P10, positive fraction, integrated excess per area,
   largest positive-component fraction, excess component count, and spatial
   dispersion. Existing Dead masks are not used.

`nuclei_overlap_area_px2` is not a projection feature because cell area times
nuclear area fraction already determines it.

## Zero-aware measurement states

Nuclear absence and Dead-signal absence are separate indicators, not ordinary
low continuous values. Conditional topology values are fitted only when the
corresponding measurement is supported and are mapped to neutral zero after
scaling when unavailable.

Nuclei status is one of `present_supported`, `absent_supported`,
`possible_nuclei_mask_miss`, or `low_quality_nuclei_signal`. A zero nuclear
area fraction is therefore never used to make a missing or failed nucleus look
numerically like an ordinary small nucleus.

Dead status is one of `dead_signal_present_supported`,
`dead_signal_absent_supported`, `dead_signal_saturated`, or
`dead_signal_background_uncertain`. Dead absence is not equivalent to live.
A cell with disrupted Brightfield morphology, supported nuclear loss, and
absent Dead fluorescence can still be reviewed as late marker-negative death.
Saturated or background-uncertain Dead measurements are not silently treated
as absence. Saturated cells retain their censored-high continuous Dead evidence
and an explicit saturated audit status; background-uncertain measurements are
mapped to the neutral point rather than made falsely positive or negative.

## Equal-block UMAP and death-resolution UMAP

All transform fitting uses development cells only. Retained continuous values
are transformed, winsorized, and robustly scaled. Each block is then scaled by
its observed mean pairwise squared distance. In the primary UMAP, Shape,
Brightfield, Nuclei, and Dead must each contribute 25%, within two percentage
points, regardless of feature count.

The label-free balanced UMAP is selected by neighborhood preservation and
cross-seed stability. It is used to audit the representation and choose a
well-balanced 300-cell anchor set. It is not required to isolate death.

After all 300 anchors are independently reviewed, V4 compares only three
predeclared geometries:

- 25/25/25/25 balanced blocks;
- 20/20/25/35, emphasizing Dead and retaining Nuclei;
- 20/30/20/30, emphasizing Dead and Brightfield.

Each profile is evaluated over the frozen UMAP grid. Selection combines
leave-well-out anchor-label balanced accuracy (45%), local dead-anchor purity
(20%), the fraction of dead anchors in the largest connected dead-neighbor
component (20%), and cross-seed stability (15%). The connectivity term encodes
the requested single broad death region while allowing continuous internal
death-stage geometry. This creates the death-resolution UMAP. Labels are used
only at this declared selection step, not to manufacture UMAP coordinates.

Diagnostic clusters are navigation strata only. They are never death labels.

## Three-channel visual evidence

`annotation_workspace.html` contains the authoritative CPA UMAP interface plus
three coordinate-matched references using the same representative cell IDs
and object bounds:

- Brightfield morphology;
- Dead fluorescence;
- Nuclei fluorescence.

The atlas also shows all three crops side by side. Fluorescence display windows
are fitted once over the development image set and locked per channel. Per-cell
autocontrast is forbidden in authoritative views because it would erase
absolute Dead/Nuclei signal differences.

## Human barriers and label authority

### Barrier 1: independent anchors

Exactly 300 cells, at most eight per well, are selected across wells,
diagnostic clusters, and Nuclei/Dead measurement statuses. The reviewer sees
only the same cell in Brightfield, Dead, and Nuclei with the Combined-mask
outline. UMAP location, polygon membership, condition, time, current
classification, and sampling bucket are hidden. Every row requires a primary
death label, conditional death-stage descriptor, nuclear-state descriptor,
confidence, reviewer, timestamp, and explicit inspection.

High/medium-confidence `dead` and `non_dead` anchors select the death-resolution
UMAP. They do not yet constitute the final training set.

### Barrier 2: broad death region

The selected death-resolution UMAP is rendered with the same three morphology
references. The user draws a deliberately broad `dead` polygon intended to
cover death at every stage. The polygon is provisional and is used only for
navigation and sampling.

V4 then freezes 500 cells, at most eight per well, equally across:

- death-region interior;
- death-region boundary on the inside;
- death-region boundary on the outside;
- distant exterior.

The exact three-channel review hides these bucket identities. Only imported
high/medium-confidence manual `dead` or `non_dead` labels are eligible for
training. Polygon assignments, UMAP coordinates, diagnostic clusters, and
sampling strata are never predictors or training truth.

## Model and evaluation contract

The final model uses the retained measurement features plus only the
predeclared cross-block interactions in the V4 config. Time, condition, well,
UMAP coordinates, cluster, and current predictions are forbidden predictors.
Outer and inner validation are grouped by well; every preprocessing, pruning,
interaction, and hyperparameter decision is fitted within each training fold.
The elastic-net alpha grid is 0, 0.25, 0.5, 0.75, and 1 with the
one-standard-error lambda rule.

Technical acceptance requires class support across wells, finite probabilities,
one prediction per cell, class-specific metrics, per-well stability, block and
interaction ablations, and an explicit unavailable status for missing evidence.
Only after model freeze may the 16 heldout wells be opened.

Primary heldout reporting treats all death stages as `dead`. Secondary reports
stratify dead predictions by marker-positive, transition, marker-negative
nucleus-loss-like, and indeterminate review states. Those secondary states do
not replace the primary death label.

## HPC boundaries

Calibration entry:

```bash
bash cellpose_pipeline/hpc/Parameter_calibration/34_run_multimodal_cell_state_v4_test.sh
```

Calibration must run directly on `hpctpa3pc0028`, outside Slurm, without GPU,
and may write only below
`results/Tests_and_Parameters_calibration/multimodal_cell_state_v4_test_*`.
The verified current SIF path and SHA are checked at runtime. The explicit bind
allowlist includes raw Brightfield, Nuclei, and Dead plus Combined/Nuclei masks;
the existing Dead segmentation and current classification roots remain
invisible.

Formal execution is intentionally not submitted before both human reviews and
the separate model-acceptance gate are complete. Its future generation must
use a new `results/multimodal_cell_state_v4_*` root and Slurm without node,
constraint, exclude, or GPU pinning. V1-V3 results are never resumed or
overwritten.

## Implemented stage interfaces

- `48`/`49`: per-field extraction and exact merge;
- `50`: four-block preprocessing, audit, balanced UMAP, and frozen candidate
  death-resolution grids;
- `51`/`52`: balanced CPA project and three-channel annotation workspace;
- `53`/`55`/`58`: independent anchor selection, exact render, and immutable
  human-evidence import;
- `56`: label-bound death-resolution UMAP selection and clean CPA project;
- CPA `annotation-import`: authoritative broad polygon import;
- `57`/`55`/`58`: interior/boundary/exterior selection, exact render, and
  immutable final review import.
- `59`: explicit, one-barrier-at-a-time post-human orchestration for anchor
  import, death-resolution projection, broad-region review, and final review
  import.

The calibration driver stops at Barrier 1. It does not fabricate either human
submission and does not fit a classifier before both reviews are complete.
The currently implemented V4 chain ends after immutable import of the broad
region review; model training, full prediction, and formal Slurm publication
remain separately gated rather than being represented as completed code.

The same direct-node launcher resumes one explicit human transition at a time
when `V4_SHADOW_ROOT` is the existing V4 calibration root:

```bash
V4_ACTION=post-anchor ANCHOR_SUBMISSION=/absolute/path/inside/v4/root/multimodal_v4_review_submission.json \
  bash cellpose_pipeline/hpc/Parameter_calibration/34_run_multimodal_cell_state_v4_test.sh

V4_ACTION=post-region REGION_SUBMISSION=/absolute/path/inside/v4/root/region_submission.json \
  bash cellpose_pipeline/hpc/Parameter_calibration/34_run_multimodal_cell_state_v4_test.sh

V4_ACTION=post-review BROAD_REVIEW_SUBMISSION=/absolute/path/inside/v4/root/multimodal_v4_review_submission.json \
  bash cellpose_pipeline/hpc/Parameter_calibration/34_run_multimodal_cell_state_v4_test.sh
```

Each action verifies and reuses completed immutable sub-generations. It cannot
skip its required submission or read a submission outside the V4 root.
