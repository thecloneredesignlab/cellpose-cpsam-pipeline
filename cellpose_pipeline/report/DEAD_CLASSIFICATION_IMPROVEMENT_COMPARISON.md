# d0 Dead Classification: Comparison of the Original and Iteratively Improved Results

## Technical Summary

This report compares the original classification results with the results obtained after several rounds of dead-classification improvement across all 320 SUM159 zero-time-point (`00d00h00m`) fields. The comparison covers both analysis branches: the original Combined-cell-mask branch (`original`) and the branch restricted to nucleated cells (`nucleated-only`).

The original workflow had two opposing problems:

1. **Live cells were misclassified as dead.** Once a Dead mask passed the basic signal filter and was assigned to a Combined cell, the original classifier directly changed that cell to dead. It did not require the Dead signal to represent the entire cell and did not protect cells whose RGB state was clearly live. Weak Dead-channel signal, weak blue signal in the Combined image, or a small dead object adjacent to a live cell could therefore overwrite the state of the neighboring live cell.
2. **Dead objects were missed or lost during classification.** The first attempt to reduce false positives added RGB-context and overlap restrictions, but those restrictions also rejected strong death signals in F5 and H9 when the death object was adjacent to or overlapping a much larger red live-cell region. In addition, the original association structure retained only one Dead object per Combined cell and discarded Dead objects that could not be assigned to a Combined cell. Consequently, some objects already present in the upstream Dead segmentation masks never entered the final classification result.

The final solution no longer assumes that one Dead object must be equivalent to one Combined cell. The workflow now maintains separate cell-level and object-level states. An RGB-live cell remains live, while a strong neighboring Dead object is counted independently. This design removes the previous tradeoff between improving death recall and increasing live-cell false positives.

In the automated audit without manual ground truth, the final results were:

- Both branches recorded all **8,289/8,289** upstream Dead segmentation objects.
- Both branches detected **1,189/1,189** high-confidence strong reference objects, corresponding to an operational recall of **100%**.
- The number of RGB-live cells classified as dead decreased from **4,485** to **0** in the original branch and from **4,562** to **0** in the nucleated-only branch.
- The maximum per-field live-to-dead false-positive proxy was **0%** across all 320 fields in both branches.
- Each branch produced and passed validation for 320 cell-state QC images and 320 Dead-object QC images.

These results demonstrate improved classification and association for the existing segmentation masks. They do not establish biological sensitivity or specificity. Without manual annotations or an independent death assay, true dead cells for which the upstream Dead segmentation produced no mask cannot be measured.

## Comparison Scope and Result Locations

### Complete source trees and zero-time-point selection

```text
Raw images:
/Volumes/Protable Disk/Project/HighPloidy_CostBenefits/20260619_SUM159_Doxorubicin_Cyclophosphamide/20260626_SUM159_AC_Exp1_SeparateImages

Pipeline results:
/Volumes/Protable Disk/Project/HighPloidy_CostBenefits/20260619_SUM159_Doxorubicin_Cyclophosphamide/results/full_fusion_shape_strict_20260711_155940
```

The analysis selects the exact `00d00h00m` field keys directly from these complete trees. It does not depend on the separately copied `results/20260619_SUM159_Doxorubicin_Cyclophosphamide_d0` directory. Field records retain the original raw-image and result paths, while all other timepoints remain untouched and are excluded by the explicit timepoint selector. The original Combined, Brightfield, Nuclei, shape-strict, and Dead segmentation masks were frozen during this comparison.

### Final object-aware results

Original branch:

```text
/Users/4482173/Documents/GitHub/cellpose-cpsam-pipeline-v3/results/dead_d0_classification_audit/object_aware_all_d0_v4
```

Nucleated-only branch:

```text
/Users/4482173/Documents/GitHub/cellpose-cpsam-pipeline-v3/results/dead_d0_classification_audit/object_aware_all_d0_v4_nucleated_only
```

Final automated audit:

```text
/Users/4482173/Documents/GitHub/cellpose-cpsam-pipeline-v3/results/dead_d0_classification_audit/object_aware_automated_reference_audit_v4
```

Independent raw-signal stress test:

```text
/Users/4482173/Documents/GitHub/cellpose-cpsam-pipeline-v3/results/dead_d0_classification_audit/dead_object_detection_stress_v2
```

## Metric Definitions and Evidence Boundaries

Because no manual ground-truth table is available, this report uses the following operational proxy metrics.

### Live false-positive proxy

```text
Number of RGB-live cells whose final state is dead
-------------------------------------------------
Total number of RGB-live cells
```

This is a conservative false-positive upper bound: every RGB-live-to-dead transition is treated as a possible live-cell false positive. It cannot detect live cells that were already incorrectly labeled dead by the RGB rule itself.

### Automated strong-death reference recall

The final object-level reference set requires that a Dead object already exists in the upstream segmentation mask and meets all of the following criteria:

- `keep_signal = True`;
- `dead_p90_delta >= 40`;
- `dead_snr >= 100`.

Object-level operational recall is defined as:

```text
Strong reference objects retained and confirmed in the final output
-------------------------------------------------------------------
All strong reference objects
```

This reference set is independent of the final cell state, but it still depends on the existing Dead segmentation mask. It measures whether an existing mask is lost during classification; it does not measure death objects that were never segmented from the raw image.

## Why the Original Workflow Misclassified Live Cells as Dead

### 1. The basic Dead-signal filter was permissive

The original classifier first filtered candidate Dead masks using their area, shape, and local Dead-channel intensity. The base filter allowed relatively weak objects into the association stage, including thresholds such as `dead_p90_delta >= 2` and `dead_snr >= 3`. These thresholds were suitable for retaining candidates, but were not sufficient evidence that an entire Combined cell was dead.

### 2. Any retained and assigned Dead object overrode the cell state

In the original `final_state_for()` implementation, a Combined cell was classified as dead whenever a Dead object had been assigned to it, even if the cell's RGB state was clearly live. These cases were typically recorded with the reason `dead_channel_override`.

The implementation therefore made the following invalid equivalence:

```text
A Dead object spatially matches a Combined cell
                       equals
The entire Combined cell is dead
```

When a shrunken, round, cytoplasm-depleted dead cell was located beside a larger attached live cell, the small Dead mask could overlap the large Combined live mask or fall within the nearest-centroid matching distance. The original workflow then transferred the state of the death object to the entire live cell.

### 3. Weak blue signal near red live-cell signal increased override risk

In fields such as E2, weak Dead-channel fluorescence produced a weak blue or cyan component in the Combined RGB image. When this signal was near a red live cell:

- the Dead mask could pass the candidate filter;
- the Dead mask could partially overlap, or be assigned by proximity to, a larger Combined live cell;
- the original logic did not determine whether the Dead signal represented the whole cell or a separate neighboring object;
- the live cell was consequently overwritten by the Dead evidence.

The original false positives therefore arose from the combination of permissive signal retention, spatial association, and unconditional cell-state override. They could not be robustly solved by changing a single intensity threshold.

## Why the Original Workflow Missed Dead Objects

### 1. Objects without a Combined-cell assignment were discarded

The original `compute_dead_evidence()` skipped an object whenever `assigned_label <= 0`. A shrunken or detached dead cell, or a death object approximately the size of a nucleus, may have no corresponding Combined cell mask. Such an object could have a very strong Dead-channel signal and still never enter the per-cell classification table.

### 2. Only one Dead object was retained for each Combined cell

The original implementation used a `best_by_combined` dictionary and retained only the highest-scoring Dead object for each Combined cell. If multiple small death objects were close to the same large Combined mask, the remaining objects were overwritten and could not be counted independently.

At the strong-direct stage, the source data contained 8,289 Dead masks, but only the following objects were represented in the old per-cell outputs:

| Branch | Source Dead masks | Represented in old per-cell output | Missing from per-cell output | Representation rate |
|---|---:|---:|---:|---:|
| Original | 8,289 | 6,953 | 1,336 | 83.88% |
| Nucleated-only | 8,289 | 5,984 | 2,305 | 72.19% |

Of the 1,189 objects meeting the final strong-reference definition, the old association structure represented only:

| Branch | Strong objects | Visible in old structure | Missing from old structure | Representation rate |
|---|---:|---:|---:|---:|
| Original | 1,189 | 1,150 | 39 | 96.72% |
| Nucleated-only | 1,189 | 1,132 | 57 | 95.21% |

These losses occurred during classification association; they do not indicate that the Dead segmentation itself failed to generate the object.

### 3. The first live-cell protection rule was overly conservative

The first context-aware revision allowed an RGB-live cell to be changed to dead only when the Dead evidence had sufficient intensity and high cell-level overlap. This eliminated most false positives, but the true strong Dead objects in F5 and H9 occupied only a small portion of a much larger red Combined mask:

- F5 cell 83: `dead_p90_delta = 233.94` and `dead_snr = 369.23`, but the Combined overlap fraction was only `0.179`;
- H9 cell 113: `dead_p90_delta = 87.74` and `dead_snr = 161.55`, but the Combined overlap fraction was only `0.095`.

Because their cell-level overlap was low, both strong death objects were incorrectly rejected in the first revision. At this stage, automated-reference recall was only 67.11% in the original branch and 61.53% in the nucleated-only branch.

## Iterative Improvement Process

### Round 1: RGB-context and Dead-intensity gating

The first revision no longer allowed arbitrary Dead evidence to override a cell state. It added the following measurements:

- mean, P90, and maximum intensity inside each Dead mask;
- local background median and MAD-based sigma;
- `p90_delta` and `SNR`;
- fraction of the Dead object covered by the Combined cell;
- fraction of the Combined cell covered by the Dead object;
- Combined-cell RGB state, cell area, and support from other channels.

This reduced the RGB-live-to-dead proxy from approximately 5.7%–6.0% to approximately 0.014%. However, because the rule still required substantial cell-level overlap, many strong death objects were rejected and recall decreased markedly.

### Round 2: Strong-direct override

The second revision added a high-intensity direct-override rule:

```text
dead_p90_delta >= 35
dead_snr >= 90
```

An object meeting these thresholds could be restored as death evidence even when its Combined overlap was low. The evidence was also evaluated before the minimum-cell-area rule so that a shrunken death object was not rejected solely because its Combined mask was small.

This restored the old cell-associated automated-reference recall to 100% and recovered the F5 and H9 death evidence. However, it re-exposed the structural defect of the original design: a strong Dead object still had to change a Combined cell to dead. Although the aggregate live false-positive proxy remained below 0.5%, per-field performance was unstable:

| Branch | Aggregate live FP proxy | Fields at or above 0.5% | Fields at or above 1% | Maximum per-field value |
|---|---:|---:|---:|---:|
| Original | 0.426% | 114/320 | 47/320 | 2.82% |
| Nucleated-only | 0.457% | 123/320 | 51/320 | 2.88% |

Increasing only the direct-signal threshold could therefore not simultaneously solve the F5/H9 death-detection problem and the E2-type live-cell protection problem.

### Final solution: separate cell state from Dead-object state

The final version uses two layers of output.

#### 1. Complete Dead-object ledger

Every upstream Dead segmentation mask is written to the object table, including objects that fail the signal filter, cannot be matched to a Combined cell, or share the same Combined-cell assignment with another Dead object. Both final branches satisfy:

```text
Source-mask object count = 8,289
Object-ledger row count  = 8,289
Per-field mismatches     = 0
```

#### 2. Multi-object association

The workflow no longer uses a one-object-per-Combined-cell structure. Every object is retained independently and assigned one of the following relations:

- `same_cell`: evidence supports that the object belongs to the Combined cell;
- `overlapping_live_dead_multi_nucleus`: a confirmed death object overlaps a live cell region containing at least two nucleus cores;
- `live_with_death_signal`: a confirmed death object overlaps a live cell region containing one nucleus core;
- `adjacent_or_overlapping_dead_uncertain`: a confirmed death object is retained when nucleus evidence cannot resolve adjacency from overlap;
- `adjacent_candidate`: adjacent evidence is insufficient and is retained for audit without being counted as confirmed death;
- `dead_only`: a strong death object has no matched Combined cell;
- `merged_multiple_objects`: multiple independent confirmed objects occur near the same Combined cell;
- `rejected_signal` or `unassigned_candidate`: retained for audit without promotion to confirmed death.

#### 3. Local RGB and Dead-channel confirmation

In addition to the RGB state of the entire Combined cell, the final object table measures local RGB features inside each Dead mask. A death object is confirmed when either:

- the Dead channel meets the strong-direct criteria; or
- local RGB inside the Dead mask is classified as dead, with `p90_delta >= 35` and `SNR >= 50`.

This design retains the locally strong blue death object in H9 even when it overlaps a red signal, without changing the entire red live cell to dead.

#### 4. Protection of RGB-live cells

If a Combined cell has an RGB-live state, neighboring Dead evidence no longer directly rewrites that state:

- the cell prediction remains `live`;
- a strong death object is recorded in one of the confirmed overlap relations and contributes to `supplemental_dead_object_count`;
- weak or inconsistent evidence is recorded as `adjacent_candidate` for QC only.

This removes the previous requirement that a death object must borrow a live-cell label in order to be counted.

#### 5. Cell-scale attribution for RGB-uncertain cells

The initial object-aware implementation protected cells already classified as RGB-live, but it still treated every retained Dead object assigned to an RGB-uncertain cell as `same_cell`. This remaining shortcut caused the Figure 5 case 3 error: C6 Dead mask 7 covered only 6.47% of a 1,761-pixel Combined cell, yet its strong local signal changed the entire neighboring cell to dead.

The corrected association rule now requires cell-scale spatial support before a confirmed object may change an RGB-uncertain cell:

- coverage of at least 45% of the Combined mask; or
- for a compact Combined cell of at most 500 pixels, coverage of at least 30%.

Objects that fail this cell-level support test are not discarded. The Combined cell remains live, while the confirmed object is written to an independent death-mask layer and contributes through `supplemental_dead_object_count`. This distinguishes the two Figure 5 controls:

- C6 case 3: cell area 1,761 pixels and coverage 0.0647; corrected to live plus `live_with_death_signal`;
- F2 case 6: cell area 413 pixels and coverage 0.3438; retained as dead plus `same_cell`.

Across all 320 fields, this refinement changed 203 RGB-uncertain cell records from dead to live in each branch while preserving all 1,416 confirmed Dead objects. These are rule-based attribution changes, not manually verified biological error counts.

#### 6. Nucleus-aware overlapping mask representation

The final workflow uses the frozen Nuclei core masks to count nucleus centroids in each Combined cell and Dead object. Nucleus evidence does not alter upstream segmentation:

- two or more nuclei increase support for overlapping live and death objects;
- one nucleus is interpreted conservatively as a live cell carrying death signal, possibly during a death transition;
- zero reliable nuclei leaves adjacency versus overlap unresolved.

The pixel output is stored as two independent instance-label layers. Channel 0 contains cell IDs and channel 1 contains confirmed Dead-object IDs, so both values can be nonzero at the same pixel. The confirmed-death layer is drawn above the cell-state layer in final QC.

#### 7. Object-aware counting

Each final per-image summary contains:

- `dead_cell_count`: cells whose Combined-cell state is dead;
- `supplemental_dead_object_count`: `dead_only`, confirmed live/death-overlap relations, and additional confirmed objects near an already represented cell;
- `object_aware_dead_count`: the sum of the two components.

Downstream analysis should use `object_aware_dead_count` as the final death count rather than reading only `dead_cell_count`.

## Final Multilevel Annotation and Classification Definitions

The final audit now writes one cohort-wide annotation ledger:

```text
results/dead_d0_classification_audit/final_multilevel_annotations.csv
```

The file contains both final branches and uses `annotation_level = cell` or `annotation_level = death_object`. Cell and Death-object rows have stable `record_id` values; a Death-object row links to its assigned cell through `linked_cell_record_id`. This long-table design preserves one row per biological/segmentation unit and avoids duplicating a cell when several Death objects share the same Combined mask. All original per-cell and per-Death-object source columns are retained with `cell_` and `death_` prefixes.

The corresponding validated summary is:

```text
results/dead_d0_classification_audit/final_annotation_summary.csv
```

Every summary row identifies its annotation level, final class, subclass, count, denominator name, denominator value, and percentage.

### Final cell classification

- `live`: a countable Combined cell without an accepted cell-level death override. It may overlap a separately confirmed supplemental Death object.
- `dead`: a Combined cell with a matched same-cell Dead object whose RGB/Dead evidence and cell-scale support pass the final override rule.
- `transitional`: a reserved final state for unresolved mixed live/death appearance.
- `uncertain`: a reserved final state when the decision rules cannot assign live or dead.
- `artifact`: a Combined mask excluded by size or unsupported-artifact criteria.

### Final Death-object classification

- `dead`: a retained Dead mask confirmed by strong direct Dead-channel evidence or local dual-channel RGB support.
- `candidate`: a retained Dead-channel signal that does not meet final confirmation criteria.
- `rejected`: a segmented Dead mask that does not pass `keep_signal`.

The two final classifications describe different units. A result such as `cell = live` plus `Death object = dead` is intentional: the Combined cell is retained as live while the overlapping confirmed Death mask is preserved and counted independently.

### Classification uncertainty

Cell uncertainty is reported within each final state. A cell is flagged when its classification confidence is `medium` or `low`, when its final state is explicitly `uncertain`, or when it is linked to a confirmed object with `adjacent_or_overlapping_dead_uncertain` attribution. Death-object uncertainty is reported separately for confirmed objects with uncertain spatial attribution and for retained-but-unconfirmed candidates. It is not equivalent to a manual-label error probability.

## Final Annotation Composition

### Final cell states and within-state uncertainty

| Branch | Live | Uncertain live annotations | Dead | Uncertain dead annotations | Transitional | Explicit uncertain state | Artifact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 88,724 | 10,234 (11.53%) | 1,164 | 344 (29.55%) | 0 | 0 | 427 |
| Nucleated-only | 80,561 | 3,973 (4.93%) | 935 | 189 (20.21%) | 0 | 0 | 10 |

### Confirmed Death objects by final relation

| Branch | Same cell | Overlapping live/dead, multiple nuclei | Live with death signal | Dead only | Spatially uncertain | Merged additional objects | Total confirmed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 758 | 341 | 269 | 34 | 7 | 7 | 1,416 |
| Nucleated-only | 715 | 346 | 287 | 59 | 0 | 9 | 1,416 |

The confirmed-Death table counts Death objects, not cells. In the original branch, 615 confirmed Death objects link to cells whose final state remains live, 767 link to final dead cells, and 34 have no linked cell. The corresponding nucleated-only counts are 630, 727, and 59.

## Like-for-Like Comparison Across Iterations

### Live false-positive proxy

This metric uses the same RGB-live denominator at every stage and can therefore be compared directly.

| Branch | Stage | RGB-live cells | RGB-live to dead | Live FP proxy |
|---|---|---:|---:|---:|
| Original | Original result | 78,493 | 4,485 | 5.714% |
| Original | Context-aware | 78,493 | 11 | 0.014% |
| Original | Strong-direct | 78,493 | 334 | 0.426% |
| Original | Final object-aware | 78,493 | 0 | 0.000% |
| Nucleated-only | Original result | 76,588 | 4,562 | 5.957% |
| Nucleated-only | Context-aware | 76,588 | 11 | 0.014% |
| Nucleated-only | Strong-direct | 76,588 | 350 | 0.457% |
| Nucleated-only | Final object-aware | 76,588 | 0 | 0.000% |

Relative to the original results, the final object-aware version eliminated Dead-channel forced overrides for 4,485 and 4,562 RGB-live cells, respectively. This is an improvement in the operational false-positive proxy, not a manual-label-confirmed false-positive count.

### Death-evidence coverage

| Branch | Stage | Reference unit | Reference count | Detected | Operational recall |
|---|---|---|---:|---:|---:|
| Original | Context-aware | Cell-associated automated reference | 1,645 | 1,104 | 67.11% |
| Original | Strong-direct | Cell-associated automated reference | 1,645 | 1,645 | 100.00% |
| Original | Final object-aware | Complete strong-object reference | 1,189 | 1,189 | 100.00% |
| Nucleated-only | Context-aware | Cell-associated automated reference | 1,427 | 878 | 61.53% |
| Nucleated-only | Strong-direct | Cell-associated automated reference | 1,427 | 1,427 | 100.00% |
| Nucleated-only | Final object-aware | Complete strong-object reference | 1,189 | 1,189 | 100.00% |

The context-aware and strong-direct stages used a reference unit that had already been associated with a cell. The final object-aware stage uses an object-level reference constructed from all Dead masks. The final rows are therefore not numerically identical metrics. Their advantage is that unassigned objects and same-cell collisions are no longer missing from the denominator.

## Historical Stage Reproducibility

The context-aware, strong-direct, and early object-aware stages were generated during iterative calibration before each working-tree state was committed independently. Their saved prediction tables, audit tables, and selected QC overlays are therefore treated as immutable comparison evidence. Re-running the current final classifier cannot reproduce those intermediate algorithms and must not be presented as if it did.

The report records the complete development sequence in `debugging_stage_provenance.json`. For every round it lists the scientific objective, thresholds and decision rules, per-field and merge command templates, output artifacts, code provenance, and whether the stage is computationally reproducible or requires a frozen reference. `FROZEN_REFERENCE_SHA256.txt` verifies the historical bundle byte-for-byte. The final nucleus-aware dual-layer v4 classification and detector stress test remain reproducible from commit `208128b7` or later and are recomputed on HPC from the complete d0 source trees.

The combined report therefore uses two explicitly separated evidence classes:

- frozen historical outputs for the original, context-aware, strong-direct, and early object-aware development comparison;
- newly computed HPC outputs for final v4 classification, multilevel annotations, automated audit, detector stress, masks, and QC.

### Final count composition

| Branch | Original dead-cell calls | Final dead-cell count | Final supplemental Dead objects | Final object-aware death count | Change from original calls |
|---|---:|---:|---:|---:|---:|
| Original | 6,936 | 1,164 | 658 | 1,822 | -73.73% |
| Nucleated-only | 5,980 | 935 | 701 | 1,636 | -72.64% |

The large decrease in dead calls is primarily due to removal of unconditional Dead-channel overrides of RGB-live cells. Without manual labels, the entire difference cannot be described as corrected errors. It reflects a change in classification unit from any cell with matched Dead evidence to cell-associated death plus independently confirmed death objects.

## Behavior of Key Sentinel Fields

| Sample and object | Original problem | Context-aware | Strong-direct | Final object-aware |
|---|---|---|---|---|
| `E2_1_00d00h00m`, cell 32 / Dead mask 2 | Weak Dead/blue signal near a red live cell created a false-positive risk | Cell remained live; weak evidence was rejected | Cell remained live | Cell remained live; object retained as unconfirmed `adjacent_candidate` |
| `F5_1_00d00h00m`, cell 83 / Dead mask 6 | A small strong death object beside a large live region was missed by the first revision | Cell remained live, but the strong death object was also rejected | Strong object recovered, but cell changed to dead | Four nuclei support probable live/dead overlap; both masks are retained |
| `H9_4_00d00h00m`, cell 113 / Dead mask 18 | Clear Dead positivity overlapped red signal and was missed because of low cell-level overlap | Cell remained live, but the strong death object was rejected | Strong object recovered, but cell changed to dead | One nucleus supports `live_with_death_signal`; both masks are retained |

The final architecture no longer requires a choice between preserving the live-cell label and detecting the adjacent death object.

The expanded Figure 5 cohort additionally includes the H8, C6, and D8 one-nucleus examples. Cases 1, 3, and 5 remain live at the cell level while their confirmed death masks are overlaid and counted. F2 case 6 remains `dead + same_cell`.

## Final QC and Output Changes

Each branch contains:

- 320 per-cell feature tables;
- 320 per-cell prediction tables;
- 320 Dead-object ledger tables;
- 320 per-image summaries;
- 320 cell-state overlays;
- 320 Dead-object overlays.
- 320 cell-state instance masks;
- 320 confirmed-death instance masks;
- 320 two-channel overlap masks;
- 320 nucleus-aware overlap overlays.

All cell-state, Dead-object, and overlap QC images passed PNG decoding validation. Every two-channel TIFF was also checked against its two component masks.

The principal new outputs are:

```text
dead_objects/<image>_dead_object_features.csv
qc/dead_object_overlays/<image>_dead_object_overlay.png
masks/cell_state/<image>_cell_state_masks.tif
masks/confirmed_dead/<image>_confirmed_dead_masks.tif
overlap_masks/<image>_cell_dead_overlap_masks.tif
qc/overlap_state_overlays/<image>_overlap_state_overlay.png
results/dead_d0_classification_audit/final_multilevel_annotations.csv
results/dead_d0_classification_audit/final_annotation_summary.csv
```

Important new summary fields include:

- `segmented_dead_object_count`;
- `retained_dead_object_count`;
- `strong_dead_object_count`;
- `confirmed_dead_object_count`;
- `dead_only_object_count`;
- `adjacent_dead_object_count`;
- `overlapping_live_dead_multi_nucleus_count`;
- `live_with_death_signal_count`;
- `nucleus_unresolved_dead_overlap_count`;
- `merged_multiple_dead_object_count`;
- `supplemental_dead_object_count`;
- `object_aware_dead_count`.

## Why the Independent Raw-Signal Detector Was Not Added to Production

An independent robust global seed detector was evaluated to determine whether candidates could be found directly from the raw Dead image outside the existing Dead segmentation. This detector did not inspect the final cell-state decision.

The stress-test results were:

| Branch | Strong references | Detected by global seed | Raw-seed recall | Synthetic relocation recall | Shifted-seed live-cell hit rate |
|---|---:|---:|---:|---:|---:|
| Original | 1,189 | 655 | 55.09% | 24.64% | 0.0968% |
| Nucleated-only | 1,189 | 655 | 55.09% | 24.64% | 0.0979% |

Although the live-cell hit rate in the shifted-channel negative control was below 0.5%, the detector recovered only 55.09% of the existing strong reference objects and therefore failed the 99.95% detection requirement. It remains a parameter-calibration stress test and was not used to modify production masks or the final object-aware results.

## Limitations and Claims That Cannot Be Made From the Current Metrics

1. **The 100% result is not biological death sensitivity.** It means that every existing Dead mask meeting the automated strong-reference definition entered the final result.
2. **The 0% result is not the biological live-cell false-positive rate.** It means that no RGB-live cell was forcibly overwritten by Dead evidence; the RGB rule itself may still be wrong.
3. **Upstream segmentation false negatives have not been quantified.** If a true dead cell has no Dead segmentation mask, it is absent from the denominator of 8,289 objects.
4. **Object-aware counting changes the unit used for death counts.** Downstream summaries must use `object_aware_dead_count` and distinguish cell-associated deaths from supplemental death objects.
5. **Without manual labels, validation is limited to consistency checks, negative controls, and known failure modes.** An independent death marker, experimental positive controls, or a limited expert review would be needed to estimate biological sensitivity, specificity, and object-count accuracy.

## Recommended Use of the Final Results

1. Use `object_aware_dead_count` as the primary d0 death-count summary.
2. Retain `dead_cell_count` and `supplemental_dead_object_count` so the source of every death count remains identifiable.
3. Use `final_multilevel_annotations.csv` for cohort-wide cell/Death analysis and `final_annotation_summary.csv` for reported totals with explicit denominators.
4. Use `dead_objects/*.csv` only when tracing an individual field back to its source object table.
5. Use `overlap_masks/*.tif` when downstream analysis must preserve simultaneous cell and confirmed-death labels at the same pixels.
6. After any parameter or code change, require both branches to maintain the 8,289-object ledger invariant, zero misses among the 1,189 strong reference objects, a per-field live FP proxy below 0.5%, and complete 320/320 validation for all masks and QC types.
7. Do not promote segmentation-external candidates to confirmed death until a new independent raw detector passes both high-recall and channel-shift negative-control requirements.

## Implementation and Audit Files

Production classification code:

```text
cellpose_pipeline/scripts/08_fuse_multichannel_classification.py
```

Audit without manual ground truth:

```text
cellpose_pipeline/scripts/Parameter_calibration/25_audit_dead_classification_without_ground_truth.py
```

Independent detector stress test:

```text
cellpose_pipeline/scripts/Parameter_calibration/26_stress_test_dead_object_detection.py
```

Primary evidence tables:

```text
results/dead_d0_classification_audit/d0_branch_before_after_metrics.csv
results/dead_d0_classification_audit/automated_reference_audit/automated_reference_metrics.csv
results/dead_d0_classification_audit/automated_reference_audit_final/automated_reference_metrics.csv
results/dead_d0_classification_audit/object_aware_automated_reference_audit_v4/automated_reference_metrics.csv
results/dead_d0_classification_audit/object_aware_automated_reference_audit_v4/per_field_metrics.csv
results/dead_d0_classification_audit/dead_object_detection_stress_v2/detection_stress_metrics.csv
results/dead_d0_classification_audit/final_multilevel_annotations.csv
results/dead_d0_classification_audit/final_annotation_summary.csv
```
