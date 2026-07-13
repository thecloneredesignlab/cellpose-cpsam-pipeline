# 红色荧光细胞核分割优化与 HPC 验证

## 目标与数据

本次优化针对红色荧光核通道曝光偏长、边缘模糊且视觉面积偏大的问题。核分割首先服务于密度估计，同时还要支持核大小/形态、多核细胞、核质比，以及与 Brightfield（BF）和 Combined 细胞 mask 的一致性分析。

HPC 数据与结果根目录：

- 图像：`/share/lab_crd/lab_crd/HighPloidy_CostBenefits/data/BreastCancerCellLines/SUM-159/N01_Incucyte_SUM159_Doxorubicin_Test1/SUM159_AC_Exp1_SeparateImages_largetest`
- 既有结果：`.../results/test/largetest_full_fusion_20260708_213504`
- 最终验证：`.../results/test/nuclei_optimization_final_hpc_20260710`
- 环境：`cellpose_cpsam`，`cellpose==4.2.1.1`

## 参数筛选结论

代表视野粗筛比较了 7 种归一化/预处理方案、2 个直径和多组 `cellprob_threshold`、`flow_threshold`、`min_size`。普通的逐图 `0.5–99.9` 百分位归一化优于背景扣除、top-hat、band-pass 和固定强度归一化；后几类方案在弱信号视野中更容易漏核。直径 24 优于 22，`flow_threshold=0` 比 0.4 更能保持高密度视野的核数。

全 20 个视野的稳健精调使用中位数、P10 和最差视野约束。最终保留原高召回轮廓参数：

| 参数 | 最终值 |
|---|---:|
| model | `cpsam_v2` |
| diameter | 24 px |
| normalization | per-image percentile 0.5–99.9 |
| cellprob_threshold | -2.75 |
| flow_threshold | 0.0 |
| min_size | 5 px² |

没有强行提高 `cellprob_threshold`。例如 `-2.5` 的中位面积仅缩小 2.6%，但 C11 弱信号视野从 273 核降到 245 核，中心召回仅 88.6%；`-2.25` 的全视野 P10 核芯计数保留率进一步降至 91.6%。由于核分割是密度计算基础，这种漏核代价高于有限的边缘收缩收益。

## 双层核 mask

流程现在同时输出两种一一对应的实例 mask：

1. `extent mask`：高召回 Cellpose 轮廓，用于核的外形、面积上界和与细胞边界的重叠分析。
2. `core mask`：在每个 extent 内，根据局部背景、MAD 稳健噪声、对象内 35% 强度分位、`SNR >= 2` 和 1 px 腐蚀生成的强度支持核芯。实例 ID 与 extent 保持一致，不拆分、不合并。

默认核芯参数为：`min_extent_area=15`、`min_core_area=5`、`local_bg_margin=10`、`smooth_sigma=1`、`snr_threshold=2`、`object_quantile=0.35`、`erode_px=1`。

这种设计把“是否存在一个核”和“曝光影响下核边缘在哪里”分开处理：高召回 extent 防止弱核丢失，core 降低荧光晕染对面积、密度和细胞边界判断的影响。

## 密度判定

生产密度表优先使用 core mask：

- `nuclei_count >= 4000`，或
- `median nearest-neighbor distance <= 16 px`。

`mask_fraction >= 0.22` 仍作为报告字段保留，但默认不能单独触发高密度，因为该指标直接受红色荧光曝光和边缘膨胀影响。最终 20 个视野仍有 9 个被判为高密度，与既有调用完全一致。

## 最终 HPC 结果

Slurm `18337576` 在 A30 节点上完成了 20 张图的生产流程重跑；Slurm `18338216` 完成最终对齐统计刷新。两者均为 `COMPLETED (0:0)`。

- 20/20 extent mask 与既有生产结果逐文件二进制一致，总核数 40,174，确认高召回基础没有改变。
- 生成 20/20 core mask；40,078 个核有有效 core，96 个小于阈值的 extent 被排除。
- extent 中位面积 143 px²（Q25–Q75：102–183）；core 中位面积 86 px²；单核 core/extent 面积比中位数为 60.4%。
- 加权细胞边界跨越率：Combined 从 36.17% 降至 22.87%，BF 从 38.38% 降至 24.98%。core 用于定位/计数时，曝光边缘导致的跨细胞现象明显减少。
- 核大小的 3 成分 log-area 模型未达到预设的清晰分离标准，因此当前只能可靠报告连续面积和描述性 small/medium/large 分位组，不能把三个组解释为已验证的生物学亚群。
- Combined mask：1,797 个确认多核细胞，占可计数细胞 5.16%；BF mask：2,024 个，占 5.80%。两种方法的确认多核 Jaccard 为 63.75%，四分类核状态一致率为 89.66%。
- 中位核质面积比（nuclear/cytoplasm）：extent 版本为 Combined 0.726、BF 0.728；抗曝光的 core 版本为 Combined 0.337、BF 0.337。每个细胞的上下界详细值均在 `cell_nuclear_summary.csv`。
- 19,390/40,174 个核（48.27%）存在可操作的 BF/Combined 对齐问题。非高密度视野为 15.16%，高密度视野为 53.98%，说明主要限制来自拥挤细胞的细胞 mask，而不是单纯的核荧光边缘。

## 第二阶段：核引导的细胞边界校准

在不改变核分割的前提下，进一步检查了 BF/Combined 细胞 mask 是否存在系统性配准偏移，并测试核芯引导的局部细胞边界修正。整数位移搜索范围为 ±6 px；Combined 和 BF 的全视野最优位移中位数均为 `(dy=0, dx=0)`，质心包含率增益中位数接近 0。因此没有证据支持对任一通道做全局平移。

局部修正遵循以下约束：

- 只使用 BF 和 Combined 中均有至少 80% 核芯支持、且两种细胞 mask 为 reciprocal-best 配对的核；
- 不拆分、不合并细胞实例，不改变核实例；
- 只重新标记核轮廓内原本属于其他细胞的像素，不填充背景；
- 每个核在每种细胞 mask 中最多改变 10 px；
- 任何被覆盖的原细胞标签修正后仍须至少保留 35 px²。

激进候选虽然可把可操作不吻合率降低到 14.39%–27.95%，但会改变 35.7%–51.0% 核附近的边界、删除小细胞标签，并使确认多核率增加 4–9 个百分点，因此被明确排除。随后测试的 `safe1/3/5/10_relabel` 只允许每核每通道改变 1/3/5/10 px，并用原图边缘、细胞面积、标签连通性、多核敏感性和 BF/Combined 一致性共同筛选。

最终推荐 `safe10_relabel` 作为**可选的下游校准 mask**，而不是覆盖原始生产 mask：

| 指标 | 原始 mask | `safe10_relabel` | 变化 |
|---|---:|---:|---:|
| 可操作不吻合 | 19,390 / 40,174（48.27%） | 17,744 / 40,174（44.17%） | -1,646，-4.10 pp |
| 严格一致 | 20,255 / 40,174（50.42%） | 21,892 / 40,174（54.49%） | +1,637，+4.07 pp |
| 高密度不吻合率 | 53.98% | 49.40% | -4.58 pp |
| 非高密度不吻合率 | 15.16% | 13.87% | -1.28 pp |
| Combined 确认多核率 | 5.16% | 5.93% | +0.76 pp |
| BF 确认多核率 | 5.80% | 6.49% | +0.69 pp |
| 两方法多核 Jaccard | 63.75% | 71.43% | +7.68 pp |
| 四分类状态一致率 | 89.66% | 90.07% | +0.41 pp |

该候选修正了 7,396/40,078 个有核芯的核，但实际只改变 Combined 0.0920%、BF 0.0875% 的全图像素；没有细胞标签增加或消失，细胞总数不变。受影响细胞面积相对变化 Q95 为 Combined 4.17%、BF 3.95%，新增碎片化比例分别为 0.32% 和 0.26%。独立于核对齐指标，修正位置的原图边缘梯度相对原边界提高 Combined 1.54%、BF 1.77%。

核质比对该修正不敏感：Combined/BF 的 extent 核质比中位数由 0.726/0.728 变为 0.723/0.727，core 版本由 0.337/0.337 变为 0.337/0.338。20 个视野均生成全图叠加图和每图 6 个最大改动区域的前后对照图；重点复核的 C2、E3_2、E3_3、C11、A11、A3、B3 和 G8_2 未见细胞合并、标签消失或大范围边界跳变。

由于候选本身使用核芯构造，对齐改善仍不是独立真值准确率。推荐保留原始 mask 作为主结果，同时把 `safe10_relabel` 用于多核、核质比和跨通道一致性的敏感性分析；最严格的跨方法多核集合是 `cell_pair_nuclear_status_concordance.csv` 中 BF 与 Combined 均确认的 1,788 对细胞。

第二阶段 Slurm 记录为：初始候选筛查/验证 `18344945`、`18344961`；标签保护候选筛查/验证 `18344995`、`18344996`、`18345000`；最终全图与局部 QC `18345025`。所有任务均为 `COMPLETED (0:0)`。

## 第三阶段：形状与稳定双峰诊断

当前流程已经测量长短轴、轴比、偏心率、圆度和 solidity，但这些指标此前只用于 QC，没有拆分 Cellpose 实例。为了检查“两个相邻核被合并为一个核”的剩余风险，本阶段只使用现有核通道、`safe10_relabel` 细胞 mask 和已有 extent/core，生成不覆盖生产结果的候选拆分层。

候选必须在原 extent 内完成 marker watershed，核前景像素严格守恒。双峰在 Gaussian sigma 0.8、1.0、1.2 和 1.5 下重复检测，以排除依赖单一平滑尺度的亮度纹理。最终 `shape_strict` 要求：

- parent area ≥190 px²；
- `solidity ≤0.90`、`circularity ≤0.58` 或 `axis ratio ≥1.70` 至少满足一项；
- 两个峰距离 ≥8 px，较弱峰 SNR ≥3.5；
- 第三峰相对强度 ≤0.80；
- 峰间强度谷下降 ≥1.3 个稳健噪声标准差且 ≥18%；
- 四种平滑尺度下的双峰位置稳定率 ≥80%；
- 两个子核均 ≥35 px²，且较小子核至少占 parent 的 22%；
- 面积加权形状评分至少提高 0.05；
- 两个子核均在 BF 和 Combined 中获得 ≥80% 包含率、质心一致及 reciprocal-best 细胞配对支持；
- watershed 必须覆盖完整 parent，任何无 marker 的断开分量直接拒绝。

全 20 个视野比较结果：

| 候选 | 拆分 parent | 核计数增加 | 形状收益中位数 | 稳定性 P10 | 两子核均获细胞支持 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `shape_sensitive` | 269 | 0.67% | 0.040 | 50% | 50.6% | 排除：4 个小子核且稳定性不足 |
| `shape_balanced` | 57 | 0.14% | 0.066 | 75% | 59.6% | 仅作宽松敏感性结果 |
| `shape_evidence` | 13 | 0.03% | 0.097 | 100% | 53.8% | 保留为相邻细胞合并候选 |
| `shape_strict` | 7 | 0.017% | 0.097 | 100% | 100% | 推荐的高置信候选层 |
| `shape_ultra` | 1 | 0.002% | 0.211 | 100% | 100% | 证据强但覆盖不足 |

`shape_strict` 将核数从 40,174 增至 40,181；7 个新增子核均生成有效 core，没有新增 `<35 px²` fragment，20/20 密度调用保持不变。形态异常标记从 58 降至 57。Combined 确认多核细胞由 2,062 增至 2,068，BF 由 2,264 增至 2,269；两种方法同时确认的多核细胞对由 1,788 增至 1,793，多核 Jaccard 从 71.434% 变为 71.463%。extent 核质比不变，Combined core 核质比只从 0.33690 变为 0.33696。

7 个对象分布在 E3_2（2 个）、F7（4 个）和 G7（1 个）。逐对象图显示多数具有双叶和稳定双峰，但仍有少数可能是单个不规则、死亡或分裂期核。因此最终决策是：**生产 extent/core 保持不变，`shape_strict` 仅作为 7 个高置信合并核的候选拆分和密度/多核敏感性层。** 形状校准对总密度的影响只有 0.017%，说明当前核计数对形状合并问题已经相当稳定。

最终成功任务为：双视野烟雾测试 `18345315`，全量筛查 `18345330`，最终前景守恒重筛 `18345371`，最终验证 `18345374`，最终全图与逐对象 QC `18345394`；均为 `COMPLETED (0:0)`。

## 主要输出

最终结果目录 `nuclei_optimization_final_hpc_20260710` 包含：

- `Nuclei/segmentations/`：高召回 extent masks。
- `Nuclei/nucleus_core_seeds/`：强度支持 core masks。
- `Nuclei/metadata/`：每图参数、extent/core 数量、面积和核芯诊断。
- `qc/density_calls.csv`：核芯计数、最近邻距离、各触发条件和最终密度调用。
- `nuclear_cell_alignment_analysis/nucleus_features.csv`：核大小、形态、强度、core/extent 比值和大小分组。
- `nuclear_cell_alignment_analysis/cell_nuclear_summary.csv`：每细胞核数、多核状态、核/细胞面积比和核质比。
- `nuclear_cell_alignment_analysis/segmentation_mismatch.csv`：所有需复核核及具体不吻合类型。
- `nuclear_cell_alignment_analysis/qc_overlays/`：BF、Combined、核轮廓和失败类别叠加图。
- `nuclear_cell_alignment_analysis/summary.json` 与 `analysis_summary.md`：完整汇总。

第二阶段结果目录 `nucleus_aware_cell_refinement_safe_hpc_20260711` 包含：

- `registration_diagnostics.csv`：逐视野、逐通道整数位移诊断。
- `repair_summary.csv` 与 `repair_events.csv`：候选级和逐核修正记录。
- `candidates/safe10_relabel/Combined|Brightfield/segmentations/`：推荐的可选校准 mask。
- `candidates/safe10_relabel/alignment/`：20 视野完整统计与全图叠加图。
- `validation/candidate_comparison.csv`：多维候选评分与 guardrail 结果。
- `validation/field_comparison.csv`：逐视野改善幅度。
- `validation/decision.json` 与 `validation_report.md`：机器可读及文字决策。
- `validation/qc_comparisons/`：20 张四联前后对照图。

形状诊断结果目录 `shape_aware_nucleus_split_hpc_20260711` 包含：

- `shape_multipeak_diagnostics.csv`：所有宽松形状门控对象的双峰与形态诊断。
- `split_events.csv`：逐候选、逐 parent 的峰、强度谷、形状和细胞支持证据。
- `candidate_summary.csv` 与 `screen_report.md`：五档候选筛查汇总。
- `candidates/shape_strict/Nuclei/`：推荐的可选拆分 extent/core masks。
- `candidates/shape_strict/alignment/`：完整核形态、核质比、多核与全视野 QC。
- `validation/candidate_comparison.csv` 与 `field_audit.csv`：候选和逐视野 guardrail。
- `validation/decision.json` 与 `validation_report.md`：最终决策。
- `validation/qc/shape_strict/`：覆盖全部 7 个对象的逐区域四联图。

## 代码入口

- `cellpose_pipeline/scripts/nuclei_segmentation_utils.py`：核芯生成与校准指标。
- `cellpose_pipeline/scripts/32_tune_nuclei_segmentation.py`：Cellpose flow 复用、参数筛选和稳健评分。
- `cellpose_pipeline/scripts/18_run_segmentation_classification_workflow.py`：生产分割和 extent/core 双输出。
- `cellpose_pipeline/scripts/28_call_high_density_from_nuclei_masks.py`：稳健密度调用。
- `cellpose_pipeline/scripts/29_fuse_multichannel_classification.py`：下游融合优先使用 core 质心/计数证据。
- `cellpose_pipeline/scripts/31_analyze_nuclear_cell_alignment.py`：核形态、多核、核质比和多通道一致性分析。
- `cellpose_pipeline/scripts/33_diagnose_registration_and_refine_cell_masks.py`：全局位移诊断与核引导局部边界候选。
- `cellpose_pipeline/scripts/34_score_nucleus_aware_cell_refinement.py`：原图边缘、面积、拓扑和跨方法一致性 guardrail。
- `cellpose_pipeline/scripts/35_render_nucleus_aware_cell_refinement_qc.py`：逐区域前后对照图。
- `cellpose_pipeline/hpc/run_nucleus_aware_cell_refinement_*.sh`：HPC 筛查、验证和最终 QC 入口。
- `cellpose_pipeline/scripts/36_screen_shape_aware_nucleus_splits.py`：形状门控、稳定双峰检测和候选拆分。
- `cellpose_pipeline/scripts/37_score_shape_aware_nucleus_splits.py`：核前景、core、密度、多核和形态 guardrail。
- `cellpose_pipeline/scripts/38_render_shape_aware_nucleus_split_qc.py`：逐拆分对象的 Nuclei/Combined/BF 四联图。
- `cellpose_pipeline/hpc/run_shape_aware_nucleus_split_*.sh`：形状筛查、验证和最终 QC 入口。

## 限制

这些结果是预测 mask 之间的内部一致性和稳健性评估，并非人工标注真值上的准确率。core 是抗曝光的保守面积下界，extent 是高召回面积上界，两者都不能直接等同于真实物理核边界。`safe10_relabel` 和 `shape_strict` 都应保留为独立的校准层，不能静默替换生产 BF/Combined 或 Nuclei mask。若后续获得像素尺寸和人工核边界/细胞边界标注，应以分层抽样真值重新校准面积、包含率、多核阈值、局部修正上限和合并核拆分标准。
