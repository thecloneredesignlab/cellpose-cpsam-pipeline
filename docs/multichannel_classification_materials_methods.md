# Materials and Methods: Multichannel Cell Segmentation and Live/Dead Classification

## Overview

Cell number and viability were quantified from matched multichannel microscopy
images using a deterministic segmentation and rule-based classification
pipeline. Four image profiles were analyzed for each field and time point:
Combined RGB, Brightfield, Dead-channel, and Nuclei-channel images. The
Combined RGB image was used as the primary cell-object anchor for final
counting. Brightfield and Nuclei segmentations were used as supporting evidence
for countable non-dead cells, whereas the Dead-channel segmentation and raw
Dead-channel intensity were used to identify dead cells and correct the RGB
classification.

For each image key, the final outputs were total cell count, live cell count,
dead cell count, live fraction, dead fraction, per-cell state assignments, and
a quality-control overlay drawn on the Combined RGB image.

## Image Organization and Field Matching

Images were organized into four profile-specific folders:

```text
Combined/
Brightfield/
Dead/
Nuclei/
```

Images belonging to the same field and time point were matched by a normalized
key extracted from file names:

```text
[well]_[site]_[day]d[hour]h[minute]m
```

For example, `C9_4_00d18h00m` denotes well C9, site 4, day 0, hour 18, minute
0. The elapsed time in hours was calculated as:

```text
elapsed_hours = day * 24 + hour + minute / 60
```

Only objects present in the Combined segmentation were eligible for final
cell-state assignment. Segmentations from the other profiles were used only to
support or modify the classification of these Combined objects.

## Cellpose Segmentation

Segmentation was performed using Cellpose version 4.2.1.1 with the `cpsam` and
`cpsam_v2` pretrained models. Each profile was segmented with a profile-specific
parameter set. Cellpose internal normalization was disabled for all automatic
profiles; images were normalized by the pipeline before model inference.

### Segmentation Image Normalization

For percentile-normalized images, pixel intensities were scaled as:

```text
I_norm = clip((I - P_low) / (P_high - P_low), 0, 1)
```

where `P_low` and `P_high` are the profile-specific lower and upper intensity
percentiles. If the upper percentile was less than or equal to the lower
percentile, the upper value was set to `P_low + 1`.

For fixed normalization, intensities were scaled as:

```text
I_norm = clip((I - fixed_low) / (fixed_high - fixed_low), 0, 1)
```

Combined RGB images were converted to grayscale luma before percentile
normalization:

```text
I_luma = 0.299 * R + 0.587 * G + 0.114 * B
```

### Segmentation Parameters

The following profile-specific parameters were used for Cellpose segmentation:

| Profile | Model | Diameter | Flow threshold | Cellprob threshold | Min size | Input scaling | Preprocessing | Postprocessing |
|---|---:|---:|---:|---:|---:|---|---|---|
| Combined | `cpsam` | 25 | 0.0 | -1.75 | 20 | percentile 1.0-99.0 after RGB luma conversion | luma conversion and percentile scaling | none |
| Brightfield | `cpsam` | 25 | 0.0 | -1.75 | 20 | percentile 1.0-99.0 | percentile scaling | none |
| Dead | `cpsam_v2` | 22 | 0.0 | -3.0 | 12 | fixed 7.6-55.0 | fixed scaling | raw-signal filtering |
| Nuclei | `cpsam_v2` | 24 | 0.0 | -2.75 | 5 | percentile 0.5-99.9 | percentile scaling | none |

The Cellpose model was evaluated with:

```text
channel_axis = None
normalize = False
diameter = profile-specific value
flow_threshold = profile-specific value
cellprob_threshold = profile-specific value
min_size = profile-specific value
```

### Dead-Channel Segmentation Filtering

Dead-channel masks were further filtered using raw Dead-channel signal. For
each candidate Dead-channel object, local background was estimated from a
bounding-box window expanded by 14 pixels. Background pixels were first selected
from pixels with Dead mask label 0. If fewer than 25 such pixels were available,
non-current-label pixels in the local window were used; if this still produced
fewer than 25 pixels, the full image was used as the fallback background.

For each Dead-channel object:

```text
bg_median = median(background)
bg_sigma = max(MAD(background) * 1.4826, 1e-6)
mean_delta = raw_mean - bg_median
p90_delta = raw_p90 - bg_median
snr = (raw_p90 - bg_median) / bg_sigma
```

Dead-channel objects were retained only if all criteria were satisfied:

| Criterion | Threshold |
|---|---:|
| object area | 20 to 2500 pixels |
| bounding-box aspect ratio | <= 4.0 |
| mean intensity above local background | >= 0.60 |
| 90th percentile intensity above local background | >= 2.00 |
| raw 90th percentile intensity | >= 15.0 |
| local signal-to-noise ratio | >= 3.00 |

Retained Dead-channel objects were relabeled consecutively and used as
Dead-channel evidence during final classification.

## High-Density Field Handling

Nuclei segmentation was used to identify high-density fields before running the
main segmentation array. For each Nuclei mask, the following density metrics
were computed:

- number of Nuclei objects
- Nuclei mask area fraction
- median Nuclei object area
- median nearest-neighbor distance between Nuclei centroids

A field was classified as high density if any of the following conditions were
met:

```text
nuclei_count >= 4000
nuclei_mask_fraction >= 0.22
nuclei_median_nearest_neighbor_distance <= 16.0 pixels
```

High-density calls were used only to adjust Brightfield segmentation. In
high-density fields, Brightfield images were segmented with `cpsam`, diameter
20, flow threshold 0.0, cell probability threshold -2.75, minimum size 5, and
percentile normalization from 1.0 to 99.0. Other profiles retained the standard
parameters listed above.

## Combined RGB Feature Extraction

The Combined mask defined the final object set. For each Combined mask label,
RGB features were extracted from raw Combined-image pixels within the mask.
Features included object area, centroid, bounding box, median RGB intensity,
mean RGB intensity, channel fractions, log channel ratios, mean saturation, red
excess, blue excess, purple score, and red-blue balance.

RGB channel fractions were calculated from median RGB intensity:

```text
rgb_sum = median_R + median_G + median_B + 1e-9
r_frac = median_R / rgb_sum
g_frac = median_G / rgb_sum
b_frac = median_B / rgb_sum
```

Log channel ratios were calculated with a pseudocount of 1:

```text
log_B_over_R = log((median_B + 1) / (median_R + 1))
log_R_over_B = log((median_R + 1) / (median_B + 1))
```

Mean saturation was calculated per pixel as:

```text
saturation = (max(R, G, B) - min(R, G, B)) / max(R, G, B)
```

and averaged over all pixels in the object. Additional color summary features
were calculated as:

```text
red_excess = r_frac - max(g_frac, b_frac)
blue_excess = b_frac - max(r_frac, g_frac)
purple_score = min(r_frac, b_frac) - g_frac
red_blue_balance = 1 - abs(r_frac - b_frac)
```

## RGB Baseline Classification

Each Combined object was first assigned an RGB baseline state using deterministic
color and size rules. Objects with area less than 35 pixels were classified as
artifacts. Objects with mean saturation below 0.08 were classified as
uncertain.

Objects were classified as RGB-dead if either of the following rules was true:

```text
b_frac >= 0.40
blue_excess >= 0.055
log_B_over_R >= 0.25
```

or:

```text
median_B >= 175
b_frac >= 0.38
blue_excess >= 0.035
```

Objects were classified as RGB-transitional if all of the following conditions
were true:

```text
min(r_frac, b_frac) >= 0.34
g_frac <= 0.285
red_blue_balance >= 0.92
purple_score >= 0.055
mean_saturation >= 0.18
```

Objects were classified as RGB-live if:

```text
r_frac >= 0.335
red_excess >= 0.015
```

Remaining objects with area less than 80 pixels and mean saturation below 0.16
were classified as artifacts. All other remaining objects were classified as
uncertain at the RGB-baseline stage.

## Brightfield and Nuclei Support Evidence

Brightfield and Nuclei masks were used to determine whether each Combined
object was supported by independent segmentation evidence. Centroids were
computed for all mask labels in each profile. A Combined object was considered
Brightfield-supported if the nearest Brightfield centroid was within 35 pixels
of the Combined centroid. A Combined object was considered Nuclei-supported if
the nearest Nuclei centroid was within 35 pixels of the Combined centroid.

An additional Nuclei-inclusion metric was computed by assigning each Nuclei
centroid to the Combined label containing the rounded centroid coordinate. The
number of Nuclei centroids inside each Combined object was recorded. Brightfield
and Nuclei evidence were used only as support for countable non-dead cells and
did not directly produce dead-cell calls.

## Dead-Channel Fusion Evidence

Dead-channel evidence was used as the primary correction for dead-cell calls.
For each retained Dead-channel object, local raw-signal statistics were
recomputed and the object was assigned to a Combined object by overlap or
centroid proximity.

Overlap assignment was preferred. A Dead-channel object was assigned to the
Combined label with the largest overlap if either of the following conditions
was met:

```text
dead_overlap_fraction >= 0.03
combined_overlap_fraction >= 0.002
```

where:

```text
dead_overlap_fraction = overlap_pixels / dead_object_area
combined_overlap_fraction = overlap_pixels / combined_object_area
```

If overlap assignment did not pass these thresholds, the Dead object was
assigned by centroid proximity when the nearest Combined centroid was within
25 pixels.

If multiple Dead-channel objects assigned to the same Combined object, the
highest-scoring Dead object was retained. The Dead evidence score was:

```text
score =
  snr
  + 0.25 * p90_delta
  + 10.0 * dead_overlap_fraction
  + 5.0 * combined_overlap_fraction
  - 0.02 * min(distance, 100.0)
```

Any Combined object with valid Dead-channel evidence was assigned a final dead
state, regardless of the initial RGB baseline state.

## Final Cell-State Assignment

Final cell state was assigned to each Combined object using a hierarchical
decision rule. Objects with area less than 35 pixels were assigned as artifacts.
If valid Dead-channel evidence was present, the object was assigned as dead. If
the RGB baseline also classified the object as dead, the reason was recorded as
RGB-dead with Dead-channel support; otherwise, the reason was recorded as a
Dead-channel override.

In the absence of Dead-channel evidence, RGB artifacts without Brightfield or
Nuclei support were retained as artifacts. RGB-uncertain objects were also
assigned as artifacts when they lacked Brightfield support, Nuclei support, and
internal Nuclei centroids, and when their area was less than or equal to 80
pixels.

All remaining non-dead Combined objects were assigned as live if they satisfied
any of the following criteria:

```text
RGB baseline state is live
Brightfield support is present
Nuclei support is present
one or more Nuclei centroids fall inside the Combined object
```

Combined objects without Dead-channel evidence that did not meet the criteria
above were assigned as live with low confidence by default. Thus, the final
counting states in the current implementation were live, dead, and artifact.
The RGB-baseline transitional and uncertain labels were retained as intermediate
features but were not used as final counting states.

## Cell Count and Viability Quantification

Per-image counts were computed from final Combined-object states. The total
number of masks was defined as the number of processed Combined labels. Artifact
objects were excluded from the total cell count:

```text
total_cell_count = total_masks - artifact_count
```

Live and dead cell counts were defined as the numbers of final live and final
dead objects, respectively:

```text
live_cell_count = number of final live objects
dead_cell_count = number of final dead objects
```

Fractions were calculated using total cell count as the denominator:

```text
live_fraction = live_cell_count / total_cell_count
dead_fraction = dead_cell_count / total_cell_count
```

For each image, the output summary also included counts of RGB-live objects,
RGB-dead objects, Dead-channel overrides, Brightfield-supported objects,
Nuclei-supported objects, and total Nuclei centroids inside Combined objects.

## Quality-Control Visualization

For each Combined image, a side-by-side QC PNG was generated. The left panel
shows the normalized Combined RGB image, and the right panel shows the same
image with final state colors blended over the Combined mask labels.

The Combined RGB image was normalized for display using the 1st and 99th
intensity percentiles. State colors were blended with alpha 0.55:

```text
overlay_pixel = 0.45 * normalized_RGB + 0.55 * state_color
```

Final state colors were:

| State | Color |
|---|---|
| live | red |
| dead | blue |
| artifact | yellow |
| uncertain | gray |
| transitional | purple |

The uncertain and transitional colors are retained for compatibility with the
legacy classifier, although the current fusion classifier does not emit these
as final counting states.

## Output Tables

The fusion workflow generated per-object feature tables, compact per-object
prediction tables, per-image summaries, an aggregate summary, and QC overlays.
Per-object features included Combined RGB features, Brightfield support, Nuclei
support, Dead-channel signal metrics, final state, final reason, confidence, and
whether the object was countable.

Per-image summaries contained:

- image identifier and normalized key
- well, site, day, hour, minute, and elapsed time
- total masks
- total cell count
- live cell count
- dead cell count
- artifact count
- dead fraction
- live fraction
- RGB-live and RGB-dead counts
- Dead-channel override count
- Brightfield-supported count
- Nuclei-supported count
- number of Nuclei centroids inside Combined objects

Per-image summary files were merged into a final `cell_count_summary.csv` table
sorted by well, elapsed time, and key.

## Computational Execution

The production workflow was run as a dependency-based Slurm pipeline. Nuclei
segmentation was run first to generate high-density calls. A second segmentation
array then processed Combined, Brightfield, and Dead images. After all
segmentation tasks completed, a CPU-only fusion classification array processed
one Combined key per task. A final CPU-only merge job combined all per-image
summary files into the aggregate cell-count table.

The production workflow used GPU resources for Cellpose segmentation and CPU
resources for fusion classification. The fusion classification did not invoke
the legacy RGB-only classifier and did not require GPU resources.
