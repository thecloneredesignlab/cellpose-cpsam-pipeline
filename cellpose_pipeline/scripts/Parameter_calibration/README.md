# Parameter calibration scripts

These scripts are numbered independently from the production pipeline. The
order follows preparation, broad parameter screening, focused validation, and
final QC. They may call analysis utilities from `../analysisi/` and shared
implementation code from `../_shared/`.

The production pipeline must not import a tuning CLI directly. Reusable
Combined-blue preprocessing is therefore implemented in
`../_shared/combined_blue_segmentation_utils.py` and imported by both the
production consensus stage and `22_tune_combined_blue_segmentation.py`.
