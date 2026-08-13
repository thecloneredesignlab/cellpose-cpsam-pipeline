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
bash cellpose_pipeline/Docker/hpc/Parameter_calibration/27_generate_late_dead_d0_d5_report.sh
```

The wrapper requires `HPC_CONTAINER_IMAGE`, the explicit data bind root, and
`REPORT_PLUGIN_ROOT`. Python and Node.js come from the SIF; the deployed Data
Analytics report package is explicitly bound at runtime. It validates 7,360 completed feature shards,
zero failed shards, zero d0 objects changed by the late-stage method, and at
least eight embedded high-resolution figures before reporting success.

`31_run_reference_cell_state_shadow_v2_test.sh` delegates the V2
historical-reference calibration. It is fail-closed unless the host is exactly
`hpctpa3pc0009`, execution is outside Slurm, the output is below
`results/Tests_and_Parameters_calibration/`, and the final parity SIF identity
is verified.
