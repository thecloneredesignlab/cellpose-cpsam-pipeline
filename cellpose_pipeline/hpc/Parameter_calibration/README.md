# HPC parameter calibration

These launchers and workers are numbered independently from production. They
cover stand-alone segmentation tests, large-test workflows, nuclei and cell
boundary optimization, shape-split validation, and high-density BF/Combined
screening. `24_legacy_submit_sum159_ac_exp1_full.sh` is retained for history but
is not the current production entry point.

`27_generate_late_dead_d0_d5_report.sh` is the final reporting-only entry for
the completed no-ground-truth death-classification calibration. It reads the
immutable calibration root
`death_classification_consensus_optimization_20260725_213646`, follows its
recorded d0 audit provenance, and writes the combined pre-production d0 +
Day-5 report under `report_final/`. It does not rerun classification,
optimization, or segmentation.

```bash
bash cellpose_pipeline/hpc/Parameter_calibration/27_generate_late_dead_d0_d5_report.sh
```

The wrapper requires the CellPose Python environment, Node.js, and the deployed
Data Analytics report package. It validates 7,360 completed feature shards,
zero failed shards, zero d0 objects changed by the late-stage method, and at
least eight embedded high-resolution figures before reporting success.

`31_run_reference_cell_state_shadow_v2_test.sh` is the V2 historical-reference
calibration entry. It must be run directly on `hpctpa3pc0009`, never through
Slurm, and writes only under `results/Tests_and_Parameters_calibration/`. It
uses the same frozen-code, exact-review, and parity-SIF contracts as formal V2.

`33_run_multimodal_cell_state_v3_test.sh` runs the dataset-adapted multimodal
V3 representation directly on `hpctpa3pc0009` with no Slurm submission, GPU,
or node fallback. It reads only Brightfield, Nuclei, Combined masks, Nuclei
masks, and the frozen development universe; Dead, Combined RGB, current
classification, trajectories, and heldout wells remain invisible. Its only
output namespace is
`results/Tests_and_Parameters_calibration/multimodal_cell_state_v3_test_*`, and
it stops at the 500-cell blind exact-review barrier.
