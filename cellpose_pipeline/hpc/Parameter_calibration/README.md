# HPC parameter calibration

These launchers and workers are numbered independently from production. They
cover stand-alone segmentation tests, large-test workflows, nuclei and cell
boundary optimization, shape-split validation, and high-density BF/Combined
screening. `24_legacy_submit_sum159_ac_exp1_full.sh` is retained for history but
is not the current production entry point.

`27_generate_late_dead_d0_d5_report.sh` is the final reporting-only entry for
the completed late-death calibration. It reads the immutable calibration root
`late_dead_trajectory_optimization_20260723_024151`, follows its recorded d0
audit provenance, and writes the combined pre-production d0 + Day-5 report
under `report_final/`. It does not rerun classification, optimization, or
segmentation.

```bash
bash cellpose_pipeline/hpc/Parameter_calibration/27_generate_late_dead_d0_d5_report.sh
```

The wrapper requires the CellPose Python environment, Node.js, and the deployed
Data Analytics report package. It validates 7,360 completed feature shards,
zero failed shards, zero d0 objects changed by the late-stage method, and at
least eight embedded high-resolution figures before reporting success.
