# Multimodal cell-state V3 workflow

`multimodal_cell_state_v3` is an independent, dataset-adapted cell-state axis.
It does not replace, modify, or tune the current production classification.
The historical V1/V2 roots remain immutable audit records; reference-repo
parity is no longer a scientific objective for V3.

## Inputs and blinding

Before model freeze, V3 may read only:

- Brightfield raw images;
- Nuclei raw fluorescence images;
- original Combined object masks;
- Nuclei extent masks, with the conservative core mask as an explicit fallback;
- development-well metadata and the frozen 32,000-cell development universe.

Dead images, Combined RGB, current labels/probabilities, trajectory outputs,
and all heldout cells are unavailable in the container. Nuclei fluorescence is
used for mask-support QC and sensitivity analyses, not for generic GLCM or
chromatin-texture claims.

## Feature blocks

The primary representation has three blocks.

1. Shape: area, roundness, aspect ratio, extent, and solidity.
2. Brightfield: complementary intensity, contrast, boundary, focus, edge, and
   texture candidates. A deterministic development-only absolute-Spearman
   filter removes a later feature when correlation is at least 0.95.
3. Nuclei support: `nucleus_absent`, excess nucleus count, conditional
   nucleus-to-cell area fraction, largest-nucleus fraction, nucleus-area CV,
   small-fragment fraction, and spatial dispersion.

`nuclei_overlap_area_px2` is retained only as an extraction audit value and is
not a projection feature because it is redundant with cell area and area
fraction.

Area fractions and nucleus topology use the full Nuclei extent segmentation;
the core-seed mask is accepted only when an extent mask is unavailable. This
keeps `nuclei_area_fraction_if_present` interpretable as nuclear area divided
by Combined-object area rather than core-seed area divided by cell area.

Nuclei zero values are not treated as ordinary low measurements. Supported
absence is encoded by `nucleus_absent=1`; the conditional quantitative values
are missing, fitted only among supported-present cells, and mapped to neutral
zero after scaling. Strong raw Nuclei signal without a mask is
`possible_nuclei_mask_miss`, not automatic death. Insufficient raw-signal QC is
`low_quality_nuclei_signal`, also neutral for absence.
Both unreliable statuses are also mapped to neutral after scaling for
`nuclei_count_excess`; a failed mask therefore cannot masquerade as a true
zero-nucleus observation.

## Preprocessing and block equality

All fitting uses development cells only. Continuous values are transformed as
declared in `configs/multimodal_cell_state_v3.json`, winsorized, median/MAD
scaled with an IQR fallback, and audited for constants. `nucleus_absent` uses a
Bernoulli prevalence scale.

Block weights are not based on feature count or PCA loading count. Each block
is scaled to the same mean pairwise squared distance and then multiplied by
the square root of one third. The primary representation must place Shape,
Brightfield, and Nuclei within two percentage points of one-third each.

## Projection calibration

The primary profile is compared with the previous expanded39 audit, an
unpruned zero-aware profile, an unbalanced profile, a Nuclei-QC sensitivity
profile, and three minus-one-block ablations. PCA feeds UMAP. The primary UMAP
grid covers neighbors 15/30/50, minimum distance 0.05/0.1/0.3, and five fixed
seeds. Run choice uses only input-neighborhood preservation and mean
cross-seed neighborhood stability. It does not use labels, Dead, current
predictions, or heldout data.

All retained PCA components are passed to UMAP. PCA is therefore an orthogonal
rotation rather than a variance-truncation step, so the one-third block
distance contract is not undone by keeping only high-variance PCs.

HDBSCAN clusters are diagnostic navigation and review-sampling strata only.
They are never cell-state labels. A separate dead-cell island is not required;
the scientific gate is the subsequent blind image-label mapping.

## Human-label authority

The UMAP/cluster atlas is an audit display. Polygon assignments are not
accepted as V3 training labels. The authoritative Seed-1 set contains exactly
500 cells, at most eight per well, with audited coverage of diagnostic cluster
and Nuclei measurement status. Its review page displays only Brightfield,
Nuclei support, and the Combined-mask outline. Every cell must be explicitly
inspected and labeled `live_cell`, `dead_cell`, `multinucleated_cell`, or
`uncertain`, with confidence and reviewer provenance.

The calibration deliberately stops at
`blind_exact_review_submission_required`. No classifier is fitted before the
complete review submission is imported and verified.

## Model, heldout, and full prediction

After the human barrier, only high/medium-confidence manual labels are
eligible. Training will use well-grouped nested cross-validation; preprocessing,
feature pruning, block weights, and model hyperparameters are fitted inside
each training fold. The candidate family is multinomial elastic net over alpha
0, 0.25, 0.5, 0.75, and 1 with the one-standard-error lambda rule. Feature and
block ablations are reported alongside class-specific and per-well stability.

Only after the feature schema, label policy, and model are frozen may the 16
heldout wells be opened for blind evaluation. Full inference remains
field-sharded, immutable, exact-coverage checked, and published under V3-only
column names. A later comparison stage reads frozen V3 and current predictions
into a third root; it cannot write back to either classifier.

## HPC execution

Calibration entry:

```bash
bash cellpose_pipeline/hpc/Parameter_calibration/33_run_multimodal_cell_state_v3_test.sh
```

It must run directly on `hpctpa3pc0009`, outside Slurm, without GPU, and may
write only below
`results/Tests_and_Parameters_calibration/multimodal_cell_state_v3_test_*`.
The latest verified SIF path and SHA are checked at runtime. Formal execution
will use a new `results/multimodal_cell_state_v3_*` generation, Slurm without
node/constraint/exclude/GPU pinning, and only after calibration and human-label
GO decisions.
