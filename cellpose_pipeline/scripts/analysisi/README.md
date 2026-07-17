# Analysis scripts

These scripts consume images, masks, or completed workflow results to produce
inventories, classifications, overlays, quantitative summaries, plots, or
audits. They are numbered independently from production and parameter tuning.

`resources/` contains analysis inputs that are not executable scripts. The
original SUM159 plate-map workbook is retained there together with a normalized
CSV used by the well time-course plotter. The CSV records each analyzed well's
plate coordinate, doxorubicin concentration, ploidy, cyclophosphamide condition,
and replicate without requiring Excel-formatting interpretation at runtime.

The normalized layout applies the workbook's dose labels to each two-row
replicate block: A/B are 2N with cyclophosphamide, C/D are 2N without
cyclophosphamide, E/F are 4N without cyclophosphamide, and G/H are 4N with
cyclophosphamide. Columns 2-11 contain 0, 3.125, 6.25, 12.5, 25, 50, 100, 200,
400, and 800 nM doxorubicin, respectively.

`07_plot_dose_response_curves.py` consumes the normalized well-time table from
`04_plot_well_counts_over_time.py`. Its default response is baseline-normalized
live-cell AUC divided by the matched vehicle AUC within each replicate plate
row. It writes separate doxorubicin-alone and doxorubicin-plus-cyclophosphamide
2N-versus-4N Hill plots together with the underlying normalized values and fit
parameters. A default run creates parallel `auc/`, `day4/`, and `day5/` output
directories; Day 4 and Day 5 use exact 96 h and 120 h live-cell responses with
the same baseline and paired-vehicle normalization.

The default run also creates `gr/` with two cross-day growth-rate inhibition
figures: one for doxorubicin alone and one for doxorubicin plus
cyclophosphamide. Each figure places fixed-top Day 4 and Day 5 GR curves above
paired `Delta GR = GR(4N) - GR(2N)` panels. GR = 1 denotes growth at the matched
control rate, GR = 0 denotes cytostasis, and GR < 0 denotes net cell loss. The
doxorubicin-alone reference is the matched untreated well; the combination
reference is the matched cyclophosphamide-only well at 0 nM doxorubicin. CSV
outputs retain per-well GR values, fit parameters, paired-replicate bootstrap
bands, paired differences, and integrated Delta-GR summaries. With only two
replicate series, bootstrap intervals are descriptive rather than strong
inferential evidence.

The same run creates `death/` with endpoint excess-lethal-fraction figures for
the two treatment conditions. At each exact endpoint, viable fraction is
`live / (live + dead)`, and the plotted response is
`1 - viable_fraction(treated) / viable_fraction(matched_control)`. Untreated
wells are the doxorubicin-alone controls; cyclophosphamide-only wells are the
combination controls. Each figure places Day 4 and Day 5 fixed-zero lethal
fraction curves above `Delta excess LF = excess LF(4N) - excess LF(2N)` panels,
where positive values indicate more adjusted death in 4N. Annotations report
LF50, LEC50, and LFmax, and CSV outputs retain raw and adjusted fractions,
ambiguity bounds, fits, paired bootstrap bands, paired differences, and
integrated Delta excess-LF summaries. Artifacts are excluded; transitional and
uncertain cells are omitted from the point estimate and represented in the
stored ambiguity bounds. This endpoint analysis does not estimate cumulative
death, a death rate, or the division-normalized death term used by GRADE.
