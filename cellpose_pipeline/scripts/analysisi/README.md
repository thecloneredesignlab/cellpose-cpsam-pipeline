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
