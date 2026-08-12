# Broad-phenotype shadow workflow

## Scope

This workflow adds an independent broad-morphology axis to the existing
classification pipeline. It does not replace or edit Cellpose segmentation,
Combined-object identities, Dead-channel discovery/override, either
original/nucleated classification branch, or late-death trajectory refinement.
Existing viability labels and their RGB evidence remain authoritative until a
separate, prospectively defined comparison approves a narrower replacement.

The production connection is the frozen Stage 06 field manifest. For each
selected branch, one stable object identifier is constructed as
`<branch>|<field-key>|<Combined-mask-label>`. Brightfield intensity, texture,
edge and mask-shape measurements are joined to three Nuclei support
measurements. Dead pixels and every existing state, prediction, probability,
confidence, review, and trajectory field are excluded from the model
allowlist.

The pinned `cell-phenotype-annotator` checkout is executed in source mode from
a read-only bind. Its source is not copied into this repository or the SIF.
The dependency lock freezes the commit, tree, LICENSE, DESCRIPTION and
read-only execution mode.

## Data flow

```text
frozen Stage 06 manifest and records
  -> audited path resolution and per-asset identity freeze
  -> per-field BF + Combined-mask + Nuclei feature shards
  -> full streaming audit tables (not loaded into R)
  -> development-only representative project
  -> deterministic UMAP
  -> region annotation HTML
  -> HUMAN BARRIER: accepted region submission
  -> deterministic image-review set
  -> HUMAN BARRIER: authoritative reviewed labels
  -> grouped nested-CV multinomial elastic net (alpha 0.5)
  -> frozen model acceptance receipt
  -> per-field full-universe prediction shards
  -> streaming coverage/probability merge
  -> independent broad_phenotype axis and shadow GO/NO-GO
```

The original branch is the V1 production object universe. Nucleated branch
support is retained for a later explicit comparison; branches must never be
mixed in one model generation.

## Feature and sampling contracts

The JSON feature configuration is the only model allowlist. It contains 52
numeric measurements. Localization coordinates and provenance stay in the
audited shard but are excluded from fitting. The primary model contains BF,
Combined-mask shape and Nuclei support features. The Nuclei comparator flag is
provenance, not a predictor. Dead and current classification artifacts are
recorded as `not_read` in every feature receipt.

The full experiment is too large for the reference package's in-memory table
and prediction representation. Full `cells.tsv` and `features.tsv` are retained
only as streamed audit artifacts. The development project first selects at
most 20 stable-hash fields per development well and then at most 25 cells per
field and 500 cells per well. Heldout wells contribute zero rows. This bounds
the source-image universe and the reference package's image cache while
preserving grouped sampling.

The development/heldout split is not a random well hash. The frozen SUM159
plate map defines each treatment condition as
`(doxorubicin_nm, ploidy, cyclophosphamide)` with paired plate-layout replicate
wells. The plate map does not establish independent biological cultures. Within
each `ploidy x cyclophosphamide` stratum, dose ranks
1, 4, 7 and 10 (0, 12.5, 100 and 800 nM in this plate) are preregistered for
heldout evaluation. Exactly one replicate from each selected pair is held out,
with replicate 1/2 assignments alternated and balanced (8 each); the paired
well remains in development. The other conditions keep both replicates in
development. This yields 16 heldout and 64 development wells, preserves
development support for all 40 treatment conditions, and limits later claims
to the 16 preregistered heldout condition pairs. The adapter freezes the plate
map hash plus 80-row well and 40-row condition audit tables.

Every trainable class requests 50 review rows globally. Review grouping is the
well with at most eight total selected rows per well; therefore a satisfied
class quota must span at least seven independent wells. The nontrainable
uncertainty class has zero training quota. Quota shortage fails rather than
being silently reallocated. The training stage independently validates grouped
outer and inner folds. The optional `explicit_unassigned` bucket also has quota
zero; ordinary unassigned cells request 100 rows, so review does not depend on
an annotator drawing an arbitrary exclusion region.

## Mutually exclusive annotation rule

Annotators apply the following first-match priority tree to the anchored
Combined object. The tree is about morphology, not viability.

1. `irregular_fragmented`: grossly fragmented, discontinuous, or severely
   irregular object.
2. `multinucleated`: at least two distinct nuclei, if rule 1 is absent.
3. `enlarged_flat`: dominantly enlarged, spread, or flat, if rules 1-2 are
   absent.
4. `rounded_compact`: dominantly rounded and compact, if rules 1-3 are absent.
5. `typical_mononuclear`: exactly one evident nucleus and none of rules 1-4.
6. `other_morphology`: reviewable object matching none of rules 1-5.
7. `phenotype_uncertain`: nontrainable; image/mask evidence is insufficient to
   apply the tree.

Brightfield and Nuclei are the only review displays, both overlaid with the
selected Combined mask. Combined RGB and Dead are deliberately hidden during
region annotation and cell review so the new morphology labels are blinded to
the current RGB/viability evidence. They may be compared only after labels and
the shadow model are frozen; neither is a predictor.

## Active learning V1

After a representative model and its canonical representative predictions
exist, `24_prepare_broad_phenotype_active_learning_project.py` creates a frozen
model-uncertainty queue project inside the same shadow root. Selection itself
remains in the pinned reference implementation. The queue excludes heldout
cells, already completed review rows, the full prediction universe, and every
viability artifact.

V1 active learning is targeted validation only. The reference package accepts
one authoritative `reviewed_labels.tsv` generation and does not automatically
merge labels from different review IDs. Consequently, an active-learning
review is not automatically appended to the baseline training labels and does
not trigger automatic retraining.

## Provenance and write boundaries

Each formal run gets a new
`results/broad_phenotype_shadow_<timestamp>/` directory. Test outputs are
restricted to
`results/Tests_and_Parameters_calibration/broad_phenotype_shadow_test_<timestamp>/`.
Before any worker starts, the submitter archives the verified clean Git commit
into a content-hashed code snapshot inside the new shadow root. Slurm jobs are
submitted with `--wrap` pointing to that absolute snapshot, so Slurm spool
relocation and later pulls of the deployment checkout cannot change queued or
resumed code. The snapshot receives its own read-only nested SIF bind, and every
resume re-extracts the frozen archive in temporary storage to verify bytewise
identity.
Slurm jobs receive an explicit stage-specific environment allowlist; host
`SBATCH_*`, shell startup, language-runtime, loader and Apptainer/Singularity
injection variables are not inherited by the worker process.
The raw dataset, frozen segmentation run, selected current classification run,
legacy NO_GO receipt and reference checkout are read-only. Only the new shadow
root is writable inside Apptainer.

The legacy viability NO_GO is hashed and retained as
`informational_only`; it is never changed or used to approve this workflow.
The broad workflow publishes its own decision receipt. Prediction requires a
model-acceptance receipt tying the model to this shadow's representative
project, class table, feature allowlist, authoritative review generation,
train-stage receipt and source/runtime identities. Every prediction task
verifies that receipt and its SHA before loading the model.

The runtime is the immutable SIF at
`/share/lab_crd/taoli/Docker/cellpose-cpsam-pipeline_hpc-cellpose-4.2.1.1-models.sif`
with SHA-256
`a6a49f3c87252c9034b3b4a1c92716716c4bf43b8e24a7919f725ce3b2857427`.
R stages use R 4.2.3, `glmnet` 4.1-10 and `uwot` 0.2.4 from that SIF.

## Acceptance and scientific interpretation

Technical acceptance requires exact field and object coverage, no duplicate or
unknown IDs, immutable input hashes, probability rows summing to one for every
available prediction, explicit unavailable rows, zero writes to frozen/current
outputs, and successful grouped-CV/model-generation verification. Full
prediction is sharded by field; a single in-memory full prediction is forbidden.

UMAP regions and the classifier use the same development feature space. Thus
nested CV measures reproducibility of manually confirmed development labels;
it is not an external estimate of biological accuracy. Heldout wells are
unseen prediction cohorts, not ground truth. Promotion remains NO_GO until a
frozen model is evaluated against independently blind-reviewed heldout cells
under preregistered class, calibration, coverage, and operational thresholds.
Only after that gate may a local replacement of an existing RGB threshold be
considered. The broad-morphology axis itself never overwrites viability.
