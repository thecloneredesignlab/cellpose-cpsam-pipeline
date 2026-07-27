# Current Death Classification Workflow

Method version: `death_classification_consensus_v2_20260725`

This diagram documents the current production classification-only workflow.
Segmentation is immutable: the workflow reads the existing original and
nucleated-only masks, performs the per-field multichannel classification,
applies the density- and time-calibrated late-death consensus refinement, and
then publishes the final classifications, uncertainty bounds, audit tables,
and reports.

![Current death classification workflow](death_classification_workflow.svg)

Printable single-page vector version:
[`death_classification_workflow.pdf`](death_classification_workflow.pdf).

The editable Mermaid source is
[`death_classification_workflow.mmd`](death_classification_workflow.mmd).

## Current final-state contract

- Confirmed base-stage Dead calls are never demoted by the late-stage method.
- Late-death rescue can only promote an eligible live cell to Dead.
- Conflicting evidence is routed to `uncertain` instead of being forced to
  live or Dead.
- Supplemental confirmed Dead objects remain independently countable and can
  spatially overlap a live cell mask.
- The authoritative field table uses the original branch counts and retains
  the nucleated-only branch as a diagnostic view.
- The current branch-final disagreement rule applies to every time point. It
  can therefore route a d0 live object to `uncertain`, although d0 is ineligible
  for late-death rescue.
- The method has no manual biological ground truth; confidence and GO/NO-GO
  gates are operational validation only.
