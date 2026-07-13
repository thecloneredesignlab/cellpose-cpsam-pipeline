# Full-cohort Dead calibration and dual-branch analysis

The production workflow now calibrates Dead preprocessing from the complete
paired Dead/Combined cohort before Dead segmentation. The calibration array is
independent of Nuclei segmentation, so both start after preflight. The main
cell-segmentation array depends on both the density table and calibration merge.

## Selected parameters

- Dead primary: `cpsam_v2`, diameter 22, cellprob -2.75.
- Dead normalization: per-image background anchor bounded by the full-cohort
  5th--95th percentile range; the intensity span remains cohort fixed.
- Combined dead signal: `max(B - (R + G) / 2, 0)` with cohort-fixed scaling.
- Combined-blue model: `cpsam`, diameter 22, cellprob -3.0.
- Final mask: all raw-evidence-filtered Dead primary objects plus only unmatched
  Combined-blue objects that pass relaxed Dead raw-evidence rescue thresholds.

The machine-readable parameter record is
`cellpose_pipeline/configs/dead_combined_blue_consensus_v1.json`.

## Output branches

The all-cell branch keeps the existing BF/Combined masks. The
`nucleated_only` branch relabels BF and Combined masks after removing instances
without a nucleus-core centroid or an extent-centroid fallback. Each branch has
its own classification fusion, shape_strict outputs, summaries, and QC.

## Production order

```text
environment preflight
  +-- Nuclei GPU array -> density table
  +-- Dead/Combined calibration CPU array -> calibration merge
                    \                         /
                     BF / Combined / Dead GPU array
                                |
                     Dead consensus report merge
                       /                    \
          all-cell classification       nucleated mask filter
                    |                         |
          all-cell shape_strict     nucleated classification
                                              |
                                  nucleated shape_strict
```

Use `cellpose_pipeline/hpc/submit_cellpose_cpsam_full_array.sh`. Its defaults
request `xxlarge`, a 12-hour wall time, and an A30 GPU only for segmentation
arrays. Calibration, density, fusion, branch filtering, merge, and shape jobs
explicitly remove inherited GPU requests.
