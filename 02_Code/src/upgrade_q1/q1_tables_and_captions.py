#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Q1 upgraded tables, figure captions, highlights, front-matter, and supplementary markdown."""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FORMAL_NEW = ROOT / '10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108'
FORMAL_OLD = FORMAL_NEW / 'analysis/frozen_baseline'
Q1_DIR = FORMAL_NEW / 'analysis/q1_spatial_calibration'
OLD_METHOD = FORMAL_OLD / 'analysis/method_strengthening'
DOCS_OLD = FORMAL_OLD / 'manuscript'
DOCS_NEW = FORMAL_NEW / 'manuscript'

TITLE_EN = (
    'Leaf-masked UAV RGB decouples nitrogen deficit from canopy openness and stabilizes few-shot cross-environment yield prediction in mulched cotton'
)
TITLE_ZH = '叶片掩膜无人机 RGB 解耦缺氮与冠层空隙度并稳定覆膜棉花少样本跨环境产量预测'

AUTHORS_LINE = (
    'Shu-Yang Yu<sup>a,b,c,d</sup>, Cundong Li<sup>c</sup>, Pengcheng Li<sup>a,b,d</sup>, '
    'Zhanlei Pan<sup>a,b,d</sup>, Jian Wang<sup>a,b,d</sup>, Guilan Sun<sup>a,b,d</sup>, '
    'Yaopeng Zhang<sup>a,b,d</sup>, Junhong Li<sup>a,b,d</sup>, Menghua Zhai<sup>a,b,d</sup>, '
    'Wenqi Zhao<sup>a,b,d</sup>, Xiaolei Ren<sup>a,b,d</sup>, Ningning Zhang<sup>a,b,c,d</sup>, '
    'Xin Li<sup>a,b,c,d</sup>, Zhenggui Zhang<sup>a,b,d,*</sup>'
)

AFFILIATIONS_BLOCK = """<sup>a</sup> State Key Laboratory of Cotton Bio-breeding and Integrated Utilization, Institute of Cotton Research of the Chinese Academy of Agricultural Sciences, Anyang, Henan 455000, China

<sup>b</sup> Xinjiang Key Laboratory for Crop Gene Editing and Germplasm Innovation, Institute of Western Agriculture, Chinese Academy of Agricultural Sciences, Changji, Xinjiang 831100, China

<sup>c</sup> College of Agronomy, Hebei Agricultural University, Baoding, Hebei 071001, China

<sup>d</sup> Changji Station, Chinese Agrosystem Long-Term Observation Network, Changji, Xinjiang 831100, China

\\* Corresponding author: Zhenggui Zhang (Email: [corresponding author email])"""

KEYWORDS_LINE = (
    '*Gossypium hirsutum*; Low-altitude remote sensing; Foreground segmentation; '
    'Nutrient diagnosis; Spatial resolution'
)

ABSTRACT_EN = (
    'In mulched drip-fertigated cotton, whole-plot visible indices mix foliage with bright plastic mulch and soil, '
    'confounding canopy nitrogen (N) deficit with canopy openness while exposing vision models to background shortcuts. '
    'Across three Xinjiang N-rate trials (0–450 kg N ha<sup>-1</sup>; cv. Xinluzao 53; *n* = 64), Canopeo leaf masking on '
    'sub-centimeter unmanned aerial vehicle (UAV) RGB orthomosaics separated pure-leaf chromaticity from contiguous canopy openness. '
    'Within 92–107 days after sowing (DAS), decoupling raised severe N-input deficit (≤ 75 versus ≥ 150 kg N ha<sup>-1</sup>) '
    'receiver operating characteristic area under the curve (ROC AUC) from 0.334 for excess green (ExG) to 0.879–0.926 '
    '(0.907 across fields versus 0.698 for mixed cover+ExG) and explained 66.7% of within-block yield variance (out-of-sample '
    '*R*<sup>2</sup> = 0.521). Without fertilizer records, source-standardized decoupled RGB predicted adjacent-field yield at '
    '*k* = 3 *R*<sup>2</sup> = 0.58 in-season (cross-year root mean squared error, RMSE = 940.3 versus 1053.7 kg ha<sup>-1</sup>) '
    'and 0.74 (735.0 kg ha<sup>-1</sup>) with 181 DAS boll filtering. With recorded N rates, three-plot zero-N (+3N0) and UAV '
    'quantile-stratified *k* = 3 calibration reached 656.2 and 601.1 kg ha<sup>-1</sup> RMSE (92.7% and 95.6% 90%-interval '
    'coverage; *N*-only: 610.4). Unmasked pretrained ResNet-50 failed under +3N0 (904.6 at *p* = 52) from overparameterization '
    '(*p* = 12: 694.5) and mulch shortcuts; leaf masking (*p* = 12) cut RMSE to 626.1 (+3N0) and 578.9 kg ha<sup>-1</sup> '
    '(stratified *k* = 3; 90.8% coverage). Pixel-mixing simulations (3.3 mm–3 m) showed screening requires ≤ 2–3 cm flights, '
    'whereas 0.2–3 m unmasked normalized green-red difference index (NGRDI) preserved N-integrated yield gains.'
)

ABSTRACT_ZH = (
    '在覆膜滴灌水肥一体化棉田中，整区可见光指数将叶片与高反光地膜及土壤混杂，既混淆冠层缺氮与冠层空隙度，又使视觉模型暴露于背景捷径风险。'
    '本研究基于新疆 3 个施氮梯度试验（0–450 kg N ha⁻¹；新陆早 53 号；*n* = 64），在亚厘米级无人机（UAV）RGB 正射影像上应用 Canopeo 叶片掩膜，'
    '分离纯叶片色度与连续冠层空隙度。播后 92–107 天（DAS）内，解耦将重度施氮不足（≤ 75 vs. ≥ 150 kg N ha⁻¹）的 ROC AUC 由超绿指数（ExG）的 0.334 '
    '提升至 0.879–0.926（跨田块 0.907 vs. 混合覆盖度+ExG 的 0.698），并解释了 66.7% 的区组内产量方差（样本外 *R*² = 0.521）。无施肥记录下，'
    '源域标准化解耦 RGB 对相邻田块估产的 *k* = 3 *R*² 在生育期内达 0.58（跨年均方根误差 RMSE = 940.3 vs. 1053.7 kg ha⁻¹），结合 181 DAS 棉铃过滤后达 0.74'
    '（735.0 kg ha⁻¹）。在已知施氮量下，3 小区零氮对照（+3N0）与无人机分位数分层 *k* = 3 校准的 RMSE 分别为 656.2 和 601.1 kg ha⁻¹'
    '（90% 预测区间覆盖率 92.7% 和 95.6%；仅施氮基线 *N*-only 为 610.4）。未掩膜预训练 ResNet-50 在 +3N0 校准下失效（*p* = 52 时为 904.6），'
    '源于过度参数化（*p* = 12 时为 694.5）与地膜背景捷径；施加同维度叶片掩膜（*p* = 12）后，RMSE 降至 626.1（+3N0）和 578.9 kg ha⁻¹'
    '（分层 *k* = 3，覆盖率 90.8%）。3.3 mm–3 m 像元混合模拟表明，缺氮与空隙筛查需采用 ≤ 2–3 cm 航拍，而 0.2–3 m 粗尺度未掩膜归一化绿红差值指数（NGRDI）'
    '在结合施氮记录时仍保持正向估产增益。'
)

HIGHLIGHTS = [
    'Canopeo leaf masking detects severe N-input deficit (92–107 DAS ROC AUC 0.88–0.93).',
    'Foliar vigor and canopy openness explain 66.7% of within-block yield variance.',
    'Source-standardized RGB predicts adjacent-field yield (*k* = 3 *R*<sup>2</sup> = 0.58–0.74).',
    'GSD needs split by task: ≤ 2–3 cm for screening; 0.2–3 m NGRDI aids yield models.',
    'PCA reduction and leaf masking fix ResNet-50 +3N0 error (904.6 to 626.1 kg ha<sup>-1</sup>).',
]

ROLES_ORDER = [
    ('2023_F1', '2023_F1', 'BlockCV', '2023 F1 CV', 24),
    ('2024_F1', '2024_F1', 'BlockCV', '2024 F1 CV', 19),
    ('2024_F2', '2024_F2', 'BlockCV', '2024 F2 CV', 21),
    ('2024_F1', '2024_F2', 'adjacent_field', '2024 F1 → 2024 F2', 21),
    ('2024_F2', '2024_F1', 'adjacent_field', '2024 F2 → 2024 F1', 19),
    ('2023_F1', '2024_F1', 'cross_year', '2023 F1 → 2024 F1', 19),
    ('2023_F1', '2024_F2', 'cross_year', '2023 F1 → 2024 F2', 21),
    ('2024_F1', '2023_F1', 'cross_year', '2024 F1 → 2023 F1', 24),
    ('2024_F2', '2023_F1', 'cross_year', '2024 F2 → 2023 F1', 24),
]

TRANSFER_ORDER = [
    ('2024_F1', '2024_F2', 'adjacent_field', '2024 F1 → 2024 F2', 18),
    ('2024_F2', '2024_F1', 'adjacent_field', '2024 F2 → 2024 F1', 16),
    ('2023_F1', '2024_F1', 'cross_year', '2023 F1 → 2024 F1', 16),
    ('2023_F1', '2024_F2', 'cross_year', '2023 F1 → 2024 F2', 18),
    ('2024_F1', '2023_F1', 'cross_year', '2024 F1 → 2023 F1', 21),
    ('2024_F2', '2023_F1', 'cross_year', '2024 F2 → 2023 F1', 21),
]


def fmt_ci(val: float, lo: float, hi: float) -> str:
    def _s(x: float) -> str:
        if abs(x) < 0.05:
            x = 0.0
        return f'{x:.1f}'.replace('-', '−')
    return f'{_s(val)} [{_s(lo)}, {_s(hi)}]'


def fmt_num(x: float) -> str:
    if abs(x) < 0.05:
        x = 0.0
    return f'{x:.1f}'.replace('-', '−')


def build_main_captions() -> list[tuple[str, str, str]]:
    """Return (fig_num, stem, caption_en) for Figures 1-6."""
    return [
        (
            '1',
            'Figure_1_design_and_label_budgets',
            '**Figure 1.** Study site, randomized complete block (RCB) layout, and UAV acquisition schedule. '
            '(a) Alar Experimental Station location in Xinjiang, China (40.6° N, 81.3° E), UAV RGB orthomosaic layouts '
            'for 2023 F1 (*n* = 24; dashed inner borders mark one extra fertilized plot per block), 2024 F2 (*n* = 21), and '
            '2024 F1 (*n* = 19 complete-case; 0–450 kg N ha<sup>-1</sup>), plus a single-plot zoom showing the 0.5 m buffered ROI '
            'and 3.3 mm slices at 78, 128, and 181 DAS. '
            '(b) Schedule of 33 nadir RGB flights by DAS and GDD<sub>10</sub>, marking the four shared canopy stages and 181 DAS boll opening.',
        ),
        (
            '2',
            'Figure_2_RGB_observations',
            '**Figure 2.** Canopy cover dynamics, N yield response, segmentation, and two-dimensional N-deficit and openness diagnosis. '
            '(a) In-season canopy cover trajectories (57–157 DAS, *n* = 64; shaded ribbons: ±1 SD for N0 and N300; gray band: four shared stages). '
            '(b) Seed-cotton yield, block-centered quadratic N-response curves (95% OLS CIs), and agronomic optima (dashed lines). '
            '(c) Sub-plot segmentation (1.80 m × 1.50 m; 2024 F1 Plot 4, N150, 128 DAS) separating pure-leaf NGRDI from contiguous openings, '
            'with cross-row cover transects comparing intact Plot 19 and gap-impaired Plot 4. '
            '(d) Diagnostic plane (*I*<sub>Gap</sub> versus *I*<sub>Leaf</sub>, *n* = 64; dotted lines: −1.0, 0.0, +1.0 Mg ha<sup>-1</sup> iso-anomalies; color: Δ*Y*<sub>blk</sub>). '
            '(e) Apparent within-environment *r* versus exact within-(*N*, block) partial *r* (*n* = 64, *df* = 36).',
        ),
        (
            '3',
            'Figure_3_RMSE_reduction',
            '**Figure 3.** Canopy background segmentation, uncalibrated (*k* = 0) adjacent-field prediction, and yield gains beyond known N rates (*n* = 64). '
            '(a) Pixel-level ExG and NGRDI density distributions for pure foliage (solid) versus mulch/soil background (dashed) at 141/142 DAS. '
            '(b) Observed versus predicted yield under *k* = 0 adjacent-field prediction (*n* = 40; dashed line: 1:1; shaded cone: ±10%) for RGB-only Mixed ExG (92–142 DAS), '
            'RGB-only Decoupled + *B*<sub>morph</sub> (*z*<sub>blk</sub>, 92–181 DAS; other normalizations in Table S2), and *N*+Leaf+*T*+*B*<sub>morph</sub>. '
            '(c) Paired Total ΔRMSE relative to *N*-only across nine tasks and their mean (*k* = 0). '
            '(d) Within-*N* ΔRMSE across nine tasks (bars: 27-resample block intervals; open symbols: *α* = 10<sup>5</sup>) and mean (±1 SE).',
        ),
        (
            '4',
            'Figure_4_multiscale_spatial_scaling',
            '**Figure 4.** Simulated pixel-mixing sensitivity (3.3 mm–3 m), N-deficit diagnostic decay, and 181 DAS boll filtering. '
            '(a) Multi-scale slices (2.4 m × 2.4 m; 2024 F1 Plot 11, N225) and cross-row profiles at 128 and 181 DAS across GSDs. '
            '(b) Four-stage mean canopy cover distributions and plot trajectories (*n* = 64) across GSDs. '
            '(c) Upper: trait correlation (*r*) with 3.3 mm benchmarks (NA: zero variance); lower: 92–142 DAS severe N-input deficit LOBO ROC AUC (dotted line: 5 cm bound; Table S1). '
            '(d) Nine-task mean Total ΔRMSE (±1 SE) across GSDs. '
            '(e) Representative 181 DAS morphological filtering (1.50 m × 1.50 m) and 64-plot component area spectrum. '
            '(f) Paired 181 DAS Total ΔRMSE across six cross-environment transfers (bars: block intervals; mean: ±1 SE).',
        ),
        (
            '5',
            'Figure_5_fewshot_calibration_paradox',
            '**Figure 5.** Orthogonal transfer MSE decomposition, N-dependent slope distortion, and few-shot plot calibration. '
            '(a) Six-task mean orthogonal MSE decomposition (Within-*N*, Between-*N*, Bias<sup>2</sup>) on fertilized holdouts across *k* = 0, +3N0, and *N*-only oracle shift, '
            'comparing Canopeo leaf-masked ResNet-50 (Masked RN50-12, *p* = 12) with unmasked RN50-12 (*p* = 12) and RN50-52 (*p* = 52; Table 3). '
            '(b) Task-level +3N0 stacked MSE across six cross-environment transfers. '
            '(c) N-response curves in 2023 F1 → 2024 F2 showing additive shifts at N300. '
            '(d) Plot-level yield residuals (predicted minus observed) in 2023 F1 → 2024 F2 for fertilized plots (*n* = 18) and N0 checks (*n* = 3). '
            '(e) Six-task mean RMSE across calibration budgets (*k* = 0–6) for *N*+Leaf+*T* under four sampling strategies (shaded band: 10th–90th random percentiles). '
            '(f) Multi-model quantile-stratified convergence (*k* = 0–6) on identical pre-harvest *N*+Leaf+*T* quantile subsets.',
        ),
        (
            '6',
            'Figure_6_forecast_timing_and_coverage',
            '**Figure 6.** Forward-only forecast horizon, plot-level 90% prediction intervals (PIs), and calibration coverage–RMSE trajectories. '
            '(a) Forward-only recursive Total ΔRMSE relative to *N*-only using all flights up to each DAS cutoff (without cross-year stage alignment; shaded bands: 27-resample block intervals) '
            'in three representative tasks (remaining six in Fig. S1). '
            '(b) Ordered plot-level 90% PI ribbons (N75–N450) in 2024 F2 → 2024 F1 (*n* = 16) and 2023 F1 → 2024 F2 (*n* = 18; crimson circle: missed by unmasked ResNet-50+3N0; teal diamond: missed by *N*+Leaf+*T*+3N0). '
            '(c) Empirical 90% PI coverage before (*k* = 0) and after (+3N0) calibration across six cross-environment transfers and mean (±1 SE; Table S5). '
            '(d) Six-task mean 90% PI coverage versus RMSE connecting *k* = 0 (open) to +3N0 (filled, ±1 SE).',
        ),
    ]


def build_zh_captions() -> str:
    return """# 主图中文图注（内部对照）

**图 1.** 研究区空间布局、随机完全区组（RCB）设计与低空遥感采集时间轴。(a) 新疆阿拉尔试验站区位（40.6° N，81.3° E）、2023 F1（*n* = 24，虚线内框示每区组额外施氮小区）、2024 F2（*n* = 21）与 2024 F1（*n* = 19 完全案例；0–450 kg N ha⁻¹）的无人机 RGB 正射布局，以及单小区 0.5 m 缓冲核心区与 78、128、181 DAS 的 3.3 mm 切片。(b) 33 架次航拍按 DAS 与 GDD₁₀ 排列的时间轴，标示 4 个共同冠层期与 181 DAS 吐絮期。

**图 2.** 冠层覆盖度动态、氮肥产量响应、冠层分割与二维氮限制及空隙诊断。(a) 生育期冠层覆盖度动态（57–157 DAS，*n* = 64；阴影带为 N0 与 N300 的 ±1 SD；灰窗为 4 个共同期）。(b) 籽棉产量、区组中心化二次氮响应曲线（95% OLS 置信区间）及农学最适施氮量虚线。(c) 2024 F1（Plot 4，N150，128 DAS；1.80 m × 1.50 m）冠层分割（纯叶片 NGRDI 与连续空隙）及同氮完整小区（Plot 19）与缺苗小区（Plot 4）跨行覆盖度剖面。(d) 二维诊断平面（*I*<sub>Gap</sub> 对 *I*<sub>Leaf</sub>，*n* = 64；点线为 −1.0、0.0、+1.0 Mg ha⁻¹ 等距平线；颜色为 Δ*Y*<sub>blk</sub>）。(e) 环境内表观 *r* 与精确控制（*N*，区组）偏相关 *r* 对比（*n* = 64，*df* = 36）。

**图 3.** 冠层背景分割、未校准（*k* = 0）邻田产量迁移及超越已知施氮量的预测增益（*n* = 64）。(a) 141/142 DAS 纯叶片（实线）与地膜/裸土背景（虚线）的 ExG 与 NGRDI 像元密度分布。(b) *k* = 0 邻田迁移（*n* = 40；虚线 1:1，灰锥 ±10%）下 RGB-only 混合 ExG（92–142 DAS）、RGB-only 解耦 + *B*<sub>morph</sub>（*z*<sub>blk</sub>，92–181 DAS；其余标准化见表 S2）与 *N*+Leaf+*T*+*B*<sub>morph</sub> 的实测-预测对比。(c) 9 项任务及均值相对 *N*-only 的总 ΔRMSE（*k* = 0）。(d) 9 项任务 Within-*N* ΔRMSE（27 次区组重抽样区间；空心示 *α* = 10⁵）及均值（±1 SE）。

**图 4.** 模拟像元混合空间分辨率敏感性（3.3 mm–3 m）、氮限制诊断衰减与 181 DAS 吐絮棉铃过滤。(a) 2024 F1（Plot 11，N225；2.4 m × 2.4 m）在 128 与 181 DAS 的多尺度切片与跨行剖面。(b) 四期平均冠层覆盖度随 GSD 变化的分布与小区轨迹（*n* = 64）。(c) 上：四类指标与 3.3 mm 基准的相关系数 *r*（NA 示零方差）；下：92–142 DAS 重度缺氮 LOBO ROC AUC（竖点线为 5 cm 上限；表 S1）。(d) 9 任务平均总 ΔRMSE（±1 SE）多尺度曲线。(e) 181 DAS 形态学过滤示意（1.50 m × 1.50 m）与 64 小区连通面积谱。(f) 181 DAS 6 项迁移配对总 ΔRMSE（均值 ±1 SE）。

**图 5.** 正交迁移 MSE 分解、氮依赖斜率畸变与微区田间校准。(a) 施氮留出集上 *k* = 0、+3N0 与 *N*-only 收获后平移参考的 6 任务平均正交 MSE 分解，对比含掩膜 ResNet-50（Masked RN50-12，*p* = 12）与未掩膜 RN50-12（*p* = 12）及 RN50-52（*p* = 52；表 3）。(b) 6 项迁移任务在 +3N0 下的正交 MSE 堆叠对比。(c) 2023 F1 → 2024 F2 施氮响应曲线及 N300 处截距平移。(d) 2023 F1 → 2024 F2 施氮小区（*n* = 18）与 N0 对照（*n* = 3）的逐小区残差。(e) *N*+Leaf+*T* 在 *k* = 0–6 下四类采样策略的 RMSE 收敛曲线（阴影带为随机 10%–90% 分位）。(f) 多模型在相同收获前 *N*+Leaf+*T* 分位数子集（*k* = 0–6）上的校准收敛曲线。

**图 6.** 前向递归预测提前期、逐小区 90% 预测区间（PI）与校准覆盖率-RMSE 轨迹。(a) 三个代表性任务利用截止日期前全部可用航次（非对齐档案频率；阴影带为 27 次区组区间）相对 *N*-only 的前向递归总 ΔRMSE（其余 6 项见图 S1）。(b) 2024 F2 → 2024 F1（*n* = 16）与 2023 F1 → 2024 F2（*n* = 18）按施氮排序的 90% PI 带（红圈：未掩膜 ResNet-50+3N0 漏覆；青菱：*N*+Leaf+*T*+3N0 漏覆）。(c) 校准前后（*k* = 0 与 +3N0）6 项迁移及均值（±1 SE）的 90% PI 覆盖率（表 S5）。(d) 连接 *k* = 0（空心）与 +3N0（实心，±1 SE）的 6 任务平均覆盖率-RMSE 轨迹。
"""


def build_main_tables_md() -> str:
    ms_int = pd.read_csv(Q1_DIR / 'multiscale_intervals_vs_N.csv')
    nat = ms_int[ms_int.prefix == '3.3 mm']
    step1 = pd.read_csv(OLD_METHOD / 'REFRAME_STEP1_INTERVALS.csv').query("family=='representation_vs_N'")
    blocked = pd.read_csv(OLD_METHOD / 'blocked_inference.csv')
    res_df = step1.query("candidate=='N_ResNet50_PCA52_common4'")
    rgb_df = pd.concat([
        blocked.query("model=='N_RGB52_common4' and base=='N'"),
        step1.query("candidate=='N_RGB52_common4'"),
    ])

    # Table 2 rows
    t2_lines = [
        '| Task | *n* | Scale | *N*-only RMSE | Δ *N*+*C*<sub>early</sub> | Δ *N*+*C*+*T* | Δ *N*+Leaf+*T* | Δ *N*+RGB52 | Δ *N*+ResNet-50 | Δ *N*+*C*+*T*+*B*<sub>morph</sub> |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for src, tgt, ev, disp, n_p in ROLES_ORDER:
        for sc in ('Total', 'Within-N'):
            r_ce = nat[(nat.source == src) & (nat.target == tgt) & (nat.evaluation == ev) & (nat.model == 'N_Cearly') & (nat.measure == sc)].iloc[0]
            r_ct = nat[(nat.source == src) & (nat.target == tgt) & (nat.evaluation == ev) & (nat.model == 'N_C_T') & (nat.measure == sc)].iloc[0]
            r_lf = nat[(nat.source == src) & (nat.target == tgt) & (nat.evaluation == ev) & (nat.model == 'N_Leaf_T') & (nat.measure == sc)].iloc[0]
            r_bm = nat[(nat.source == src) & (nat.target == tgt) & (nat.evaluation == ev) & (nat.model == 'N_C_T_Bmorph') & (nat.measure == sc)].iloc[0]
            r_52 = rgb_df[(rgb_df.source == src) & (rgb_df.target == tgt) & (rgb_df.evaluation == ev) & (rgb_df.measure == sc)].iloc[0]
            r_rn = res_df[(res_df.source == src) & (res_df.target == tgt) & (res_df.evaluation == ev) & (res_df.measure == sc)].iloc[0]
            base_rmse = fmt_num(float(r_ct.RMSE_N))
            t2_lines.append(
                f'| {disp} | {n_p} | {sc} | {base_rmse} | '
                f'{fmt_ci(r_ce.gain_absolute, r_ce.cluster_low, r_ce.cluster_high)} | '
                f'{fmt_ci(r_ct.gain_absolute, r_ct.cluster_low, r_ct.cluster_high)} | '
                f'{fmt_ci(r_lf.gain_absolute, r_lf.cluster_low, r_lf.cluster_high)} | '
                f'{fmt_ci(r_52.gain_absolute, r_52.cluster_low, r_52.cluster_high)} | '
                f'{fmt_ci(r_rn.gain_absolute, r_rn.cluster_low, r_rn.cluster_high)} | '
                f'{fmt_ci(r_bm.gain_absolute, r_bm.cluster_low, r_bm.cluster_high)} |'
            )

    # Table 3 rows: Orthogonal 4-way decomposition on non-N0 holdout (including Canopy-Masked ResNet-50)
    dec_base = pd.read_csv(Q1_DIR / 'hybrid_transfer_decomposition.csv')
    dec_mask = pd.read_csv(Q1_DIR / 'resnet_masked_p12_table3_decomposition.csv')
    dec = pd.concat([dec_base, dec_mask], ignore_index=True)
    t3_models = [
        ('N__zero_label', '*N*-only (*k* = 0)'),
        ('N_C_T__zero_label', '*N*+*C*+*T* (*k* = 0)'),
        ('N_Leaf_T__zero_label', '*N*+Leaf+*T* (*k* = 0)'),
        ('N_ResNet50_Masked_p12__zero_label', '*N*+Masked ResNet-50 (*p* = 12, *k* = 0)'),
        ('N_ResNet50_Unmasked_p12__zero_label', '*N*+Unmasked ResNet-50 (*p* = 12, *k* = 0)'),
        ('N_ResNet50_PCA52_common4__zero_label', '*N*+Unmasked ResNet-50 (*p* = 52, *k* = 0)'),
        ('N__plus_3N0', '*N*+3N0'),
        ('N_Leaf_T__plus_3N0', '*N*+Leaf+*T*+3N0'),
        ('N_ResNet50_Masked_p12__plus_3N0', '*N*+Masked ResNet-50 (*p* = 12)+3N0'),
        ('N_ResNet50_Unmasked_p12__plus_3N0', '*N*+Unmasked ResNet-50 (*p* = 12)+3N0'),
        ('N_ResNet50_PCA52_common4__plus_3N0', '*N*+Unmasked ResNet-50 (*p* = 52)+3N0'),
        ('N__oracle', '*N*-only post-harvest shift ref.'),
    ]
    t3_lines = [
        '| Direction | *n* | Model and calibration regime | Total RMSE | Signed bias | Between-*N* RMSE | Within-*N* RMSE |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for src, tgt, _, disp, n_h in TRANSFER_ORDER:
        sub_d = dec[(dec.source == src) & (dec.target == tgt)]
        for mid, m_lbl in t3_models:
            g = sub_d[sub_d.model_id == mid].set_index('measure')
            vals = [fmt_ci(g.loc[m, 'estimate'], g.loc[m, 'low_95'], g.loc[m, 'high_95']) for m in ('Total', 'Bias', 'Between-N', 'Within-N')]
            t3_lines.append(f'| {disp} | {n_h} | {m_lbl} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |')

    return f"""# **Main Tables**

**Table 1.** Experimental environments, UAV RGB sensor acquisition, canopy segmentation and multi-scale feature hierarchy, modeling modes, and field calibration protocols.

| Category | Item or feature set | Schedule, sensor parameters, and mathematical definition | Sample support and agronomic role |
| --- | --- | --- | --- |
| Field trial support | 2023 F1 (Cross-year benchmark) | 24 gross plots (3 RCB blocks × 8 plots in 4 × 2 grids; 7 N rates 0–450 kg N ha<sup>-1</sup> plus one extra plot per block: N375 in B1, N300 in B2, N225 in B3); 7 flights (85–181 DAS) | *n* = 24 complete-case plots; common anchors: 92 (1360 °C·d), 107 (1694 °C·d), 130 (2149 °C·d), 142 (2360.6 °C·d), and 181 DAS |
| Field trial support | 2024 F1 (Primary development) | 21 gross plots (3 RCB blocks × 7 N rates: 0, 75, 150, 225, 300, 375, and 450 kg N ha<sup>-1</sup>); 13 flights (57–181 DAS); 0.50 m inward-buffered ROIs (15.14–15.75 m<sup>2</sup>) | *n* = 19 complete-case plots (Block 3 plots 20 [N225] and 24 [N75] clipped at 92/106 DAS; *n* = 21 at 181 DAS); anchors: 92, 106, 128, 141, and 181 DAS |
| Field trial support | 2024 F2 (Adjacent-field target) | 21 gross plots (3 RCB blocks × 7 N rates: 0–450 kg N ha<sup>-1</sup>) separated from 2024 F1 by a 36.6 m buffer under identical management; 13 flights (63–181 DAS) | *n* = 21 complete-case plots; shared anchors: 92, 106, 128, and 141 DAS plus post-defoliation boll opening at 181 DAS (2914.5 °C·d) |
| UAV RGB acquisition | DJI Phantom 4 RTK (1-inch CMOS) | 20 MP global mechanical shutter at 25.0 m altitude (2.0 m s<sup>-1</sup>, 80%/75% overlap), solar-noon default AE/AWB; approximately 6.8 mm native optical GSD exported at 3.3 mm grid spacing in DJI Terra | 33 co-registered 8-bit sRGB GeoTIFF orthomosaics; valid-pixel integer block averaging to 1 cm, 5 cm, 20 cm, 1 m, and 3 m simulated GSDs |
| Recorded N baseline (*N*) | Quadratic N response [1, *N*, *N*<sup>2</sup>] | Unpenalized seasonal N rate and squared N rate in ordinary least squares (OLS) and Frisch–Waugh–Lovell PartialRidge (*p* = 2) | Agronomic diminishing-return benchmark across three within-field LOBO CVs, two adjacent-field transfers, and four cross-year transfers |
| Mixed-pixel canopy (*C*<sub>early</sub>, *C*, *T*) | Cover, mixed ExG, and 4-date trajectory | Canopeo leaf mask *M*<sub>leaf</sub> (*R* < 0.95*G*, *B* < 0.95*G*, 2*G* − *R* − *B* > 20), whole-ROI ExG (2*g* − *r* − *b*), trapezoidal integral *T*<sub>auc,mean</sub>, and *T*<sub>late,change</sub> | Conventional mixed-pixel single-stage (*p* = 2) and four-stage (*p* = 4) canopy representations susceptible to exposed mulch and soil glare |
| Segmented leaf & openness traits (Leaf, Gap) | Pure-leaf NGRDI and canopy-openness metrics | NGRDI<sub>leaf</sub> = (*G* − *R*)/(*G* + *R*) strictly on *M*<sub>leaf</sub>; contiguous canopy openings ≥ 0.04 m<sup>2</sup> (*S*<sub>gap,large</sub>, *T*<sub>gap,auc</sub>); cross-row cover SD (*S*<sub>prof,across</sub>) | Physically separates foliar chromaticity (*N*+Leaf+*T*, *p* = 4) from canopy openness and stand gaps across 3.3 mm–3 m GSD |
| Two-dimensional diagnosis (*I*<sub>Leaf</sub>, *I*<sub>Gap</sub>) | Foliar vigor versus openness plane | *I*<sub>Leaf</sub> = [*z*(NGRDI<sub>leaf</sub>) + *z*(*T*<sub>NGRDI,leaf</sub>) + *z*(*T*<sub>late,change</sub>)] / 3; *I*<sub>Gap</sub> = [*z*(*S*<sub>gap,large</sub>) + *z*(*T*<sub>gap,auc</sub>) + *z*(*S*<sub>prof,across</sub>)] / 3 | In-season severe N-input deficit screening (≤ 75 versus ≥ 150 kg N ha<sup>-1</sup>, *n* = 17 versus 47) at 92–107 DAS and 92–142 DAS; 4-quadrant yield attribution |
| RGB-only yield prediction | Prior-free Ridge without N records | L2 Ridge on mixed or decoupled RGB traits (*p* = 2–5) under source-only standardization, commercial N225 / N0+N225 strip scaling, or within-trial *z*<sub>blk</sub> | Evaluates yield mapping in unrecorded fields across in-season (92–142 DAS) and post-defoliation (+181 DAS) windows at *k* = 0 and *k* = 3 (Table S2) |
| Deep & empirical benchmarks | RGB52 and Unmasked/Masked ResNet-50 | 13 visible indices × 4 dates (*p* = 52; Table S3); fixed ImageNet ResNet-50 embeddings on whole-ROI versus *M*<sub>leaf</sub>-masked 224 × 224 crops (*p* = 4, 12, 52 nested PCA) | Tests whether check-plot calibration degradation stems from unmasked plastic-mulch background shortcuts or PCA dimensionality (Table 3; Table S4) |
| Post-defoliation bolls (*B*<sub>raw</sub>, *B*<sub>morph</sub>) | 181 DAS connected-component filter | Pre-specified 0.5–120 cm<sup>2</sup> component area filter on fixed HSV brightness–saturation white mask (*V* ≥ 140/255, *S* ≤ 0.30; fixed without target yield tuning) | Supplementary near-harvest evaluation suppressing soil speckle (< 0.5 cm<sup>2</sup>) and elongated inter-row plastic-mulch glare (> 120 cm<sup>2</sup>; Fig. 4e) |
| Field calibration regimes | Uncalibrated (*k* = 0) and *k* = 1–6 plots | Source-only (*k* = 0), N0 checks, standard N225/N300 strips, block-balanced random, and pre-harvest UAV quantile-stratified sampling | Evaluated strictly out-of-sample on common fertilized holdouts (*n* = 21, 16, and 18 in 2023 F1, 2024 F1, and 2024 F2; post-harvest *N*-only mean-offset ref.) |

Note: Baseline topsoil (0–20 cm) across the three sandy-loam Calcaric Fluvisol trials had pH 8.1–8.3, electrical conductivity (EC, 1:5 soil-to-water) 0.4–1.2 dS m<sup>-1</sup>, organic matter 11.4–13.2 g kg<sup>-1</sup>, alkali-hydrolyzable N 48–56 mg kg<sup>-1</sup>, Olsen-P 16–21 mg kg<sup>-1</sup>, and NH<sub>4</sub>OAc-K 185–210 mg kg<sup>-1</sup>. Transparent polyethylene film mulch (0.01 mm thickness, 2.05 m width) covered six rows in a wide-narrow (66 + 10 cm) configuration. Urea (46% N) was applied 20% basally before sowing and 80% across five equal drip-fertigation splits at 65, 78, 90, 102, and 115 DAS (92 and 106/107 DAS flights precede final fertigation cutoff). Hand-harvested central plot yields (22.80 m<sup>2</sup>) were air-dried to constant storage moisture (approximately 8%–11%; cv. Xinluzao 53 regional lint turnout 41%–43%).

**Table 2.** Paired Total and Within-N RMSE reductions over the quadratic *N*-only benchmark across image representations under uncalibrated evaluation (*k* = 0) on the 64-plot common support.

{chr(10).join(t2_lines)}

Note: Values are in kg ha<sup>-1</sup> (*N*+ResNet-50 denotes whole-ROI *p* = 52; Canopeo-masked ResNet-50 ablations are detailed in Table 3 and Table S4). Brackets report conditional block intervals across all 27 ordered resamples of the *B* = 3 target blocks (for *B* = 3, the 2.5th and 97.5th percentiles coincide with the three-block sample minimum and maximum). Positive Δ indicates lower candidate RMSE than *N*-only. Entries at 0.0 [0.0, 0.0] in 2023 F1 → 2024 F1/F2 reflect single-source 3-block inner CV selecting the conservative penalty ceiling *α* = 10<sup>5</sup>; pooling two source environments (6 blocks, *n* = 40–45) eliminates single-year shrinkage while preserving calibrated gains (Table S7).

**Table 3.** Orthogonal four-way transfer error decomposition on the common non-N0 fertilized holdout across uncalibrated (*k* = 0), three-plot N0 calibrated (+3N0), and post-harvest *N*-only shift reference regimes, comparing decoupled *N*+Leaf+*T* against Canopeo-masked (*p* = 12) and matched/high-dimensional unmasked (*p* = 12, 52) *N*+ResNet-50.

{chr(10).join(t3_lines)}

Note: Values are in kg ha<sup>-1</sup> with unified fixed-center 27-resample block intervals (spanning the 3-block sample range). Across any holdout prediction vector, Total MSE equals Bias<sup>2</sup> + Between-*N* MSE + Within-*N* MSE; scalar intercept shifts modify Bias and Total RMSE while leaving Between-*N* and Within-*N* RMSE invariant. Decomposing the 278.5 kg ha<sup>-1</sup> +3N0 reduction from unmasked *p* = 52 (904.6 kg ha<sup>-1</sup>) to masked *p* = 12 (626.1 kg ha<sup>-1</sup>) separates dimensionality reduction alone (*p* = 52 → 12 unmasked: −210.1 kg ha<sup>-1</sup> to 694.5 kg ha<sup>-1</sup>) from matched-dimension Canopeo leaf masking (−68.4 kg ha<sup>-1</sup> at *p* = 12, cutting Between-*N* RMSE from 348.9 to 320.8 kg ha<sup>-1</sup> and +3N0 Bias magnitude from +370.1 to −114.5 kg ha<sup>-1</sup>; −209.2 kg ha<sup>-1</sup> at *p* = 52 from 904.6 to 695.4 kg ha<sup>-1</sup>; Table S4).
"""


def build_supp_tables_extra() -> str:
    """Build curated Supplementary Tables S1-S7."""
    # --- Table S1: Multi-scale GSD sensitivity with same-window 92-107 DAS & 92-142 DAS ROC AUC ---
    phys = pd.read_csv(Q1_DIR / 'multiscale_physical_degradation.csv').sort_values('gsd_m')
    ms_int = pd.read_csv(Q1_DIR / 'multiscale_intervals_vs_N.csv')
    sw_diag = pd.read_csv(Q1_DIR / 'n_limitation_same_window_multiscale.csv')

    s1_lines = [
        '| Spatial resolution (GSD) | Mean cover (%) | Cover MAE vs. 3.3 mm (pp) | Large-gap *r* | Row SD *r* | Boll *r* | 92–107 DAS AUC (ExG / *C*+*T* / Dec. *p*=4 / *p*=8) | 92–142 DAS AUC (*C*+*T* / Dec. *p*=8) | Mean ΔTotal *N*+*C*+*T* | Mean ΔTotal *N*+Leaf+*T* | Mean ΔTotal *N*+*C*+*T*+*B*<sub>morph</sub> |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for r in phys.itertuples():
        g = r.gsd_label
        gt_ct = ms_int[(ms_int.prefix == g) & (ms_int.model == 'N_C_T') & (ms_int.measure == 'Total')].gain_absolute.mean()
        gt_lf = ms_int[(ms_int.prefix == g) & (ms_int.model == 'N_Leaf_T') & (ms_int.measure == 'Total')].gain_absolute.mean()
        gt_bm = ms_int[(ms_int.prefix == g) & (ms_int.model == 'N_C_T_Bmorph') & (ms_int.measure == 'Total')].gain_absolute.mean()

        sub_e = sw_diag[(sw_diag.gsd_label == g) & (sw_diag.window == 'Early_92_107')].set_index('model')['LOBO_ROC_AUC']
        sub_f = sw_diag[(sw_diag.gsd_label == g) & (sw_diag.window == 'Full_92_142')].set_index('model')['LOBO_ROC_AUC']
        early_str = f"{sub_e['Mixed_ExG']:.3f} / {sub_e['Mixed_CT']:.3f} / {sub_e['Decoupled_p4']:.3f} / {sub_e['Decoupled_p8']:.3f}"
        full_str = f"{sub_f['Mixed_CT']:.3f} / {sub_f['Decoupled_p8']:.3f}"

        row_r_str = 'NA' if pd.isna(r.row_profile_r_vs_native) else f'{r.row_profile_r_vs_native:.3f}'
        boll_r_str = 'NA' if pd.isna(r.boll_r_vs_native) else f'{r.boll_r_vs_native:.3f}'.replace('-', '−')
        s1_lines.append(
            f'| {g} | {r.mean_canopy_cover_pct:.1f} | {r.cover_mae_vs_native_pp:.2f} | '
            f'{r.gap_r_vs_native:.3f} | {row_r_str} | {boll_r_str} | '
            f'{early_str} | {full_str} | '
            f'{fmt_num(gt_ct)} | {fmt_num(gt_lf)} | {fmt_num(gt_bm)} |'
        )

    # --- Table S2: RGB-only normalizations (Zero-Leakage Source-Std, Commercial Strips, Within-Trial z_blk) & N-integrated ---
    r2_norm = pd.read_csv(Q1_DIR / 'ablation_rgb_only_normalization_regimes.csv')
    pf = pd.read_csv(Q1_DIR / 'direction_b_prior_free_summary.csv')

    def _norm_row(regime: str, spec: str) -> tuple[str, str, str, str, str, str, str, str]:
        sub = r2_norm[(r2_norm.regime == regime) & (r2_norm.specification == spec)]
        bcv = sub[sub.evaluation == 'BlockCV']
        adj = sub[sub.evaluation == 'adjacent_field']
        cyr = sub[sub.evaluation == 'cross_year']
        return (
            fmt_num(bcv.RMSE_k0_all.mean()),
            f'{bcv.R2_k0_all.mean():.2f}'.replace('-', '−'),
            fmt_num(adj.RMSE_k0_all.mean()),
            fmt_num(adj.RMSE_k3_ho.mean()),
            f'{adj.R2_k3_ho.mean():.2f}'.replace('-', '−'),
            fmt_num(cyr.RMSE_k0_all.mean()),
            fmt_num(cyr.RMSE_k3_ho.mean()),
            f'{cyr.R2_k3_ho.mean():.2f}'.replace('-', '−'),
        )

    s2_lines = [
        '| Model specification | Window (DAS) | Target normalization regime | BlockCV RMSE (*k*=0) | BlockCV *R*<sup>2</sup> | Adj.-field RMSE (*k*=0) | Adj.-field RMSE (*k*=3) | Adj.-field *R*<sup>2</sup> (*k*=3) | Cross-year RMSE (*k*=0) | Cross-year RMSE (*k*=3) | Cross-year *R*<sup>2</sup> (*k*=3) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    s2_specs = [
        ('Pure_Source_Std', 'RGB_Mixed_ExG', 'RGB-only Mixed ExG', '92–142', 'Source-only std. (0 target strips)'),
        ('Pure_Source_Std', 'RGB_Mixed_CT', 'RGB-only Mixed *C*+*T*', '92–142', 'Source-only std. (0 target strips)'),
        ('Pure_Source_Std', 'RGB_Mixed_CT_Bmorph', 'RGB-only Mixed *C*+*T* + *B*<sub>morph</sub>', '92–181', 'Source-only std. (0 target strips)'),
        ('Pure_Source_Std', 'RGB_Decoupled_InSeason', 'RGB-only Decoupled Leaf+Gap+*T*', '92–142', 'Source-only std. (0 target strips)'),
        ('Pure_Source_Std', 'RGB_Decoupled_Bmorph', 'RGB-only Decoupled + *B*<sub>morph</sub>', '92–181', 'Source-only std. (0 target strips)'),
        ('Strip_N225_Only', 'RGB_Mixed_CT_Bmorph', 'RGB-only Mixed *C*+*T* + *B*<sub>morph</sub>', '92–181', 'Standard N225 strip-centered'),
        ('Strip_N225_Only', 'RGB_Decoupled_InSeason', 'RGB-only Decoupled Leaf+Gap+*T*', '92–142', 'Standard N225 strip-centered'),
        ('Strip_N225_Only', 'RGB_Decoupled_Bmorph', 'RGB-only Decoupled + *B*<sub>morph</sub>', '92–181', 'Standard N225 strip-centered'),
        ('Strip_N0_N225', 'RGB_Decoupled_InSeason', 'RGB-only Decoupled Leaf+Gap+*T*', '92–142', 'Two-point N0 + N225 strip scale'),
        ('Trial_MultiRate_z_blk', 'RGB_Mixed_CT_Bmorph', 'RGB-only Mixed *C*+*T* + *B*<sub>morph</sub>', '92–181', 'Within-trial block *z*<sub>blk</sub>'),
    ]
    for reg, sp, disp_m, win_s, reg_lbl in s2_specs:
        vals = _norm_row(reg, sp)
        s2_lines.append(f'| {disp_m} | {win_s} | {reg_lbl} | ' + ' | '.join(vals) + ' |')

    pf_models = [
        ('RGB_Mixed_ExG', 'RGB-only Mixed ExG', '92–142', 'Within-trial block *z*<sub>blk</sub>'),
        ('RGB_Mixed_CT', 'RGB-only Mixed *C*+*T*', '92–142', 'Within-trial block *z*<sub>blk</sub>'),
        ('RGB_Decoupled_InSeason', 'RGB-only Decoupled Leaf+Gap+*T*', '92–142', 'Within-trial block *z*<sub>blk</sub>'),
        ('RGB_Decoupled_Bmorph', 'RGB-only Decoupled + *B*<sub>morph</sub>', '92–181', 'Within-trial block *z*<sub>blk</sub> (Fig. 3b)'),
        ('Agron_N_Only', 'Recorded-*N* baseline (*N*-only)', 'Sowing plan', 'Known plot N rate (source std.)'),
        ('Integrated_N_Decoupled', 'N-integrated Mixed *C*+*T*+*B*<sub>morph</sub>', '92–181', 'Known plot N rate (source std.)'),
        ('Integrated_N_Leaf_Bmorph', 'N-integrated Decoupled Leaf+*T*+*B*<sub>morph</sub>', '92–181', 'Known plot N rate (source std.; Fig. 3b)'),
    ]
    for mk, mdisp, win_s, reg_lbl in pf_models:
        sub_m = pf[pf.specification == mk]
        bcv = sub_m[sub_m.evaluation == 'BlockCV']
        adj = sub_m[sub_m.evaluation == 'adjacent_field']
        cyr = sub_m[sub_m.evaluation == 'cross_year']
        s2_lines.append(
            f'| {mdisp} | {win_s} | {reg_lbl} | '
            f'{fmt_num(bcv.RMSE_k0_all.mean())} | {f"{bcv.R2_k0_all.mean():.2f}".replace("-", "−")} | '
            f'{fmt_num(adj.RMSE_k0_all.mean())} | {fmt_num(adj.RMSE_k3.mean())} | {f"{adj.R2_k3.mean():.2f}".replace("-", "−")} | '
            f'{fmt_num(cyr.RMSE_k0_all.mean())} | {fmt_num(cyr.RMSE_k3.mean())} | {f"{cyr.R2_k3.mean():.2f}".replace("-", "−")} |'
        )

    # --- Table S4: Few-shot calibration curves k = 0..6 including Masked & Unmasked ResNet-50 ---
    fc = pd.read_csv(Q1_DIR / 'fewshot_calibration_curves.csv')
    ff = fc[fc.support == 'unsampled_fertilized']
    rn_fc = pd.read_csv(Q1_DIR / 'resnet_mask_pca_fewshot_curves.csv')
    rn_cal = pd.read_csv(Q1_DIR / 'resnet_mask_pca_calibration.csv')
    piv = pd.concat([
        ff[['specification', 'strategy', 'k', 'mean_RMSE']],
        rn_fc[['specification', 'strategy', 'k', 'mean_RMSE']],
    ], ignore_index=True).groupby(['specification', 'strategy', 'k'])['mean_RMSE'].mean().unstack('k')

    spec_names = [
        ('N', '*N*-only'),
        ('N_C_T', '*N*+*C*+*T*'),
        ('N_Leaf_T', '*N*+Leaf+*T*'),
        ('N_ResNet50_Masked_p4', '*N*+Masked ResNet-50 (*p* = 4)'),
        ('N_ResNet50_Masked_p12', '*N*+Masked ResNet-50 (*p* = 12)'),
        ('N_ResNet50_Masked_p52', '*N*+Masked ResNet-50 (*p* = 52)'),
        ('N_ResNet50_Unmasked_p4', '*N*+Unmasked ResNet-50 (*p* = 4)'),
        ('N_ResNet50_Unmasked_p12', '*N*+Unmasked ResNet-50 (*p* = 12)'),
        ('N_ResNet50_PCA52_common4', '*N*+Unmasked ResNet-50 (*p* = 52)'),
    ]
    strat_names = [
        ('Random_all', 'Random *k* plots (unstratified)'),
        ('Block_Rand_fert', 'Block-balanced random fertilized'),
        ('N0_check', 'Zero-N check strips (N0)'),
        ('Farmer_N225', 'Standard strips (N225)'),
        ('Farmer_N300', 'Standard strips (N300)'),
        ('UAV_Stratified_Pred', 'UAV quantile-stratified'),
    ]

    s4_lines = [
        '| Model specification | Calibration sampling strategy | *k* = 0 | *k* = 1 | *k* = 2 | *k* = 3 | *k* = 4 | *k* = 5 | *k* = 6 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    rn_cal_mean = rn_cal.groupby('specification')[['k0_Total_RMSE', 'plus3N0_Total_RMSE', 'N225_k3_RMSE']].mean()
    for sp_k, sp_disp in spec_names:
        for st_k, st_disp in strat_names:
            if (sp_k, st_k) in piv.index:
                row = piv.loc[(sp_k, st_k)]
                cells = [fmt_num(row[k]) if k in row and pd.notna(row[k]) else '—' for k in range(7)]
                s4_lines.append(f'| {sp_disp} | {st_disp} | ' + ' | '.join(cells) + ' |')
            elif sp_k in rn_cal_mean.index and st_k in ('N0_check', 'Farmer_N225'):
                k0_v = fmt_num(float(rn_cal_mean.loc[sp_k, 'k0_Total_RMSE']))
                k3_v = fmt_num(float(rn_cal_mean.loc[sp_k, 'plus3N0_Total_RMSE' if st_k == 'N0_check' else 'N225_k3_RMSE']))
                s4_lines.append(f'| {sp_disp} | {st_disp} | {k0_v} | — | — | {k3_v} | — | — | — |')

    # --- Table S5: Empirical 90% PI coverage including Masked & matched-dimension Unmasked ResNet-50 ---
    df5 = pd.read_csv(Q1_DIR / 'calibrated_interval_coverage.csv')
    df5_mask = pd.read_csv(Q1_DIR / 'resnet_masked_p12_coverage.csv').rename(columns={'model': 'model_raw'})
    df5_mask['model'] = df5_mask['model_raw'].map({
        'N_ResNet50_Masked_p12__zero_label': 'N+Masked ResNet-50 (p=12)',
        'N_ResNet50_Masked_p12__plus_3N0': 'N+Masked ResNet-50 (p=12)+3N0',
        'N_ResNet50_Unmasked_p12__zero_label': 'N+Unmasked ResNet-50 (p=12)',
        'N_ResNet50_Unmasked_p12__plus_3N0': 'N+Unmasked ResNet-50 (p=12)+3N0',
    })
    df5_all = pd.concat([df5[['source', 'target', 'model', 'coverage', 'mean_width_kg_ha']],
                         df5_mask.dropna(subset=['model'])[['source', 'target', 'model', 'coverage', 'mean_width_kg_ha']]], ignore_index=True)

    s5_lines = [
        '| Model specification | 2024 F1 → F2 | 2024 F2 → F1 | 2023 F1 → 2024 F1 | 2023 F1 → 2024 F2 | 2024 F1 → 2023 F1 | 2024 F2 → 2023 F1 | Six-task mean (%) | Mean width (kg ha<sup>-1</sup>) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    mod_disp_map = [
        ('N-only', '*N*-only (*k* = 0)'),
        ('N+3N0', '*N*+3N0'),
        ('N+C+T', '*N*+*C*+*T* (*k* = 0)'),
        ('N+C+T+3N0', '*N*+*C*+*T*+3N0'),
        ('N+Leaf+T', '*N*+Leaf+*T* (*k* = 0)'),
        ('N+Leaf+T+3N0', '*N*+Leaf+*T*+3N0'),
        ('N+Masked ResNet-50 (p=12)', '*N*+Masked ResNet-50 (*p* = 12, *k* = 0)'),
        ('N+Masked ResNet-50 (p=12)+3N0', '*N*+Masked ResNet-50 (*p* = 12)+3N0'),
        ('N+Unmasked ResNet-50 (p=12)', '*N*+Unmasked ResNet-50 (*p* = 12, *k* = 0)'),
        ('N+Unmasked ResNet-50 (p=12)+3N0', '*N*+Unmasked ResNet-50 (*p* = 12)+3N0'),
        ('N+ResNet-50', '*N*+Unmasked ResNet-50 (*p* = 52, *k* = 0)'),
        ('N+ResNet-50+3N0', '*N*+Unmasked ResNet-50 (*p* = 52)+3N0'),
    ]
    for mod_lbl, mod_disp in mod_disp_map:
        covs = []
        widths = []
        for src, tgt, _, _, _ in TRANSFER_ORDER:
            r5 = df5_all[(df5_all.source == src) & (df5_all.target == tgt) & (df5_all.model == mod_lbl)].iloc[0]
            covs.append(100.0 * float(r5.coverage))
            widths.append(float(r5.mean_width_kg_ha))
        cov_cells = ' | '.join(f'{c:.1f}' for c in covs)
        s5_lines.append(f'| {mod_disp} | {cov_cells} | {sum(covs)/len(covs):.1f} | {sum(widths)/len(widths):.1f} |')

    # --- Table S6: 2x2 Index (ExG vs NGRDI) x Masking (Whole-ROI vs Canopeo-Leaf) + Gap Multi-Scale Ablation ---
    ab2 = pd.read_csv(Q1_DIR / 'ablation_2x2_index_mask_multiscale.csv')
    s6_lines = [
        '| Spatial resolution (GSD) | ΔTotal *N*+Whole ExG | ΔTotal *N*+Whole NGRDI | ΔTotal *N*+Leaf ExG | ΔTotal *N*+Leaf NGRDI | ΔTotal *N*+Leaf+Gap+*T* | ΔWithin-*N* *N*+Whole ExG | ΔWithin-*N* *N*+Whole NGRDI | ΔWithin-*N* *N*+Leaf ExG | ΔWithin-*N* *N*+Leaf NGRDI | ΔWithin-*N* *N*+Leaf+Gap+*T* |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for gsd_lbl in ['3.3 mm', '1 cm', '5 cm', '20 cm', '1 m', '3 m']:
        sub_g = ab2[ab2.gsd_label == gsd_lbl].set_index('specification')
        t_vals = [fmt_num(float(sub_g.loc[sp, 'mean_9task_Delta_Total'])) for sp in ('N_Whole_ExG', 'N_Whole_NGRDI', 'N_Leaf_ExG', 'N_Leaf_NGRDI', 'N_Leaf_Gap_T')]
        w_vals = [fmt_num(float(sub_g.loc[sp, 'mean_9task_Delta_WithinN'])) for sp in ('N_Whole_ExG', 'N_Whole_NGRDI', 'N_Leaf_ExG', 'N_Leaf_NGRDI', 'N_Leaf_Gap_T')]
        s6_lines.append(f'| {gsd_lbl} | ' + ' | '.join(t_vals + w_vals) + ' |')

    # --- Table S7: 6-Block Pooled-Source LOEO Cross-Validation ---
    loeo = pd.read_csv(Q1_DIR / 'ablation_6block_pooled_source_loeo.csv')
    s7_lines = [
        '| Pooled source environments (6 blocks) | Target environment | *n*<sub>src</sub> | Model specification | Selected *α* | Uncalibrated Total RMSE | Within-*N* RMSE | ΔWithin-*N* vs. *N*-only | +3N0 Total RMSE | Δ(+3N0) vs. *N*+3N0 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    sp_map_s7 = {'N': '*N*-only', 'N_C_T': '*N*+*C*+*T*', 'N_Leaf_T': '*N*+Leaf+*T*'}
    for row in loeo.itertuples():
        a_str = '0 (unpenalized)' if row.specification == 'N' else (f'{int(row.selected_alpha)}' if row.selected_alpha >= 1 else f'{row.selected_alpha:.1f}')
        s7_lines.append(
            f'| {row.sources_pooled.replace("_", " ").replace("+", " + ")} | {row.target.replace("_", " ")} | {row.n_source_plots} | '
            f'{sp_map_s7[row.specification]} | {a_str} | {fmt_num(row.Total_RMSE)} | {fmt_num(row.WithinN_RMSE)} | '
            f'{fmt_num(row.Delta_WithinN_vs_N)} | {fmt_num(row.plus3N0_RMSE)} | {fmt_num(row.Delta_plus3N0_vs_N)} |'
        )

    return f"""**Table S1.** Simulated pixel-mixing spatial resolution sensitivity across six ground sampling distances (exported 3.3 mm to 3 m GSD): apparent canopy cover bias, sub-canopy trait correlations (*r*), same-window early (92–107 DAS) and full-season (92–142 DAS) severe N-input deficit diagnostic ROC AUC, and nine-task mean Total ΔRMSE (kg ha<sup>-1</sup>; *n* = 64 complete-case plots).

{chr(10).join(s1_lines)}

Note: Coarser spatial resolutions were simulated from 3.3 mm orthomosaics inside 0.50 m inward-buffered ROIs via integer valid-pixel block averaging prior to canopy segmentation; higher-altitude flights incur additional optical point-spread blur and structure-from-motion smoothing, making ≤ 2–3 cm native flight GSD recommended for canopy-openness mapping and full-season 92–142 DAS diagnosis (whereas coarse 0.2–3 m unmasked NGRDI retains positive N-integrated yield ΔRMSE; Table S6). Morphologically filtered 181 DAS open-boll fraction (0.5–120 cm<sup>2</sup>) decays to zero variance (NA) at ≥ 20 cm GSD (where single pixels exceed 120 cm<sup>2</sup>), and cross-row closure SD decays to zero variance (NA) at 3 m GSD. Spatial LOBO L2 logistic regression (*C* = 0.5; ROC AUC = 0.914–0.930 across *C* ∈ [0.05, 5.0]) classifies severe N-input deficit (≤ 75 kg N ha<sup>-1</sup>, *n* = 17 in the 64 complete-case plots after clipping 2024 F1 Block 3 Plot 24 [N75]) versus N-sufficient canopies (≥ 150 kg N ha<sup>-1</sup>, *n* = 47) under strictly matched temporal windows: at 3.3 mm across 92–107 DAS, cross-field leave-one-field-out (LOFO) ROC AUC is 0.468 for Mixed ExG (*p* = 2), 0.698 for Mixed *C*+*T* (*p* = 4), and 0.907 for size-matched Decoupled RGB (*p* = 4; LOBO PR-AUC = 0.880 for *p* = 8 versus 0.206 for ExG). Feature enumeration: 92–107 DAS Dec. *p* = 4 uses [92 & 106/107 DAS NGRDI<sub>leaf</sub>, *S*<sub>gap,large</sub>, *S*<sub>prof,across</sub>]; Dec. *p* = 8 adds 92 & 106/107 DAS *f*<sub>cover</sub>, early integral, and early cover change; 92–142 DAS Dec. *p* = 8 uses [128/130 & 141/142 DAS NGRDI<sub>leaf</sub>, *T*<sub>NGRDI,leaf</sub>, *T*<sub>late,change</sub>, *S*<sub>gap,large</sub>, *T*<sub>gap,auc</sub>, *S*<sub>prof,across</sub>, *T*<sub>auc,mean</sub>].

**Table S2.** RGB-only (without fertilizer records) versus N-integrated (with recorded N rates) seed-cotton yield prediction across zero-leakage source-only standardization, commercial single-rate (N225) and two-point (N0 + N225) reference strips, and within-trial multi-rate block standardization (*z*<sub>blk</sub>) before (*k* = 0) and after (*k* = 3) calibration (*n* = 64).

{chr(10).join(s2_lines)}

Note: RMSE values are in kg ha<sup>-1</sup>; *k* = 0 is evaluated across all *n* target plots and *k* = 3 (three block-balanced median-vigor plots) is evaluated strictly on unsampled *n* − 3 holdout plots. Source-only standardization scales target features strictly by source-domain training means and SDs (zero target plots or reference strips). Standard N225 strip-centering subtracts the target farm's routine N225 strip optical mean scaled by source SD without requiring zero-N checks or multi-rate blocks, whereas two-point N0 + N225 scaling divides by the three-plot (N225 − N0) difference and inflates variance when reference-strip contrasts narrow across years.

**Table S3.** Mathematical definitions and invalid-denominator rules for the 13 empirical visible vegetation indices (RGB13) forming the four-stage 52-feature (*N*+RGB52) benchmark.

| Index | Formula (chromaticity *r*, *g*, *b*) | Channels | Invalid denominator rule | Primary reference |
| --- | --- | --- | --- | --- |
| ExB | 1.4*b* − *g* | *g*, *b* | Not applicable | Mesas-Carrascosa et al. (2020) |
| ExG | 2*g* − *r* − *b* | *r*, *g*, *b* | Not applicable | Woebbecke et al. (1995) |
| ExR | 1.4*r* − *g* | *r*, *g* | Not applicable | Meyer and Neto (2008) |
| ExGR | 3*g* − 2.4*r* − *b* | *r*, *g*, *b* | Not applicable | Meyer and Neto (2008) |
| GBRI | *g* / *b* | *g*, *b* | *b* = 0 masked as missing | Bendig et al. (2015) |
| GLI | (2*g* − *r* − *b*) / (2*g* + *r* + *b*) | *r*, *g*, *b* | 2*g* + *r* + *b* = 0 masked | Bendig et al. (2015) |
| IKAW | (*r* − *b*) / (*r* + *b*) | *r*, *b* | *r* + *b* = 0 masked | Kawashima and Nakatani (1998) |
| MGRVI | (*g*<sup>2</sup> − *r*<sup>2</sup>) / (*g*<sup>2</sup> + *r*<sup>2</sup>) | *r*, *g* | *g*<sup>2</sup> + *r*<sup>2</sup> = 0 masked | Bendig et al. (2015) |
| NGBDI | (*g* − *b*) / (*g* + *b*) | *g*, *b* | *g* + *b* = 0 masked | Kawashima and Nakatani (1998); Zhi et al. (2025) |
| NGRDI | (*g* − *r*) / (*g* + *r*) | *r*, *g* | *g* + *r* = 0 masked | Tucker (1979); Gitelson et al. (2002) |
| RGBVI | (*g*<sup>2</sup> − *rb*) / (*g*<sup>2</sup> + *rb*) | *r*, *g*, *b* | *g*<sup>2</sup> + *rb* = 0 masked | Bendig et al. (2015) |
| RGRI | *r* / *g* | *r*, *g* | *g* = 0 masked | Bendig et al. (2015) |
| VARI | (*g* − *r*) / (*g* + *r* − *b*) | *r*, *g*, *b* | *g* + *r* − *b* = 0 masked | Gitelson et al. (2002) |

Note: Normalized chromaticity coordinates are defined per valid ROI pixel as *r* = *R*/(*R* + *G* + *B*), *g* = *G*/(*R* + *G* + *B*), and *b* = *B*/(*R* + *G* + *B*) for *R* + *G* + *B* > 0. Under identical *N*+*C*+*T* inputs across all nine tasks, Frisch–Waugh–Lovell PartialRidge achieved lower Total and Within-*N* RMSE than conventional joint L2 Ridge and source-nested random forests.

**Table S4.** Unweighted six-task mean transfer RMSE (kg ha<sup>-1</sup>) on unsampled fertilized plots across target calibration budgets (*k* = 0 to 6 plots), sampling strategies, and Canopeo-masked versus unmasked ResNet-50 PCA dimensionalities (*p* = 4, 12, and 52).

{chr(10).join(s4_lines)}

Note: Single-rate check strategies (N0, N225, and N300) include up to three replicates per target field (*k* ≤ 3), and sampled calibration plots are strictly excluded from holdout evaluation on every draw. Because excluding *k* extreme-quantile plots slightly shrinks and eases the remaining *n* − *k* holdout, the matched same-subset idealized post-harvest *N*-only mean-offset benchmark (oracle shift) declines from 594.0 (*k* = 0) to 571.6 (*k* = 3) and 528.0 kg ha<sup>-1</sup> (*k* = 6), confirming that every calibrated strategy remains strictly above its matched same-subset benchmark; a no-UAV N-rate-stratified baseline yields 602.4 (*N*-only) and 611.2 kg ha<sup>-1</sup> (*N*+Leaf+*T*) at *k* = 3, and 588.7 and 580.2 kg ha<sup>-1</sup> at *k* = 6. Inside each training split, ResNet-50 PCA is fitted strictly on training-fold plot-date feature vectors (4 dates × *n*<sub>train</sub> = 48–96 centered vectors, rank ≤ 47–95 across folds) before concatenating the top 1, 3, or 13 PCs per date (*p* = 4, 12, or 52). Reducing dimensionality from *p* = 52 to *p* = 12 accounts for −210.1 kg ha<sup>-1</sup> of the +3N0 reduction (904.6 to 694.5 kg ha<sup>-1</sup>), while imputing non-leaf pixels with the ImageNet channel mean prior to normalization via *M*<sub>leaf</sub> provides an additional matched-dimension gain of −68.4 kg ha<sup>-1</sup> at *p* = 12 (626.1 versus 694.5 kg ha<sup>-1</sup>) and −209.2 kg ha<sup>-1</sup> at *p* = 52 (695.4 versus 904.6 kg ha<sup>-1</sup>); under commercial quantile-stratified *k* = 3 sampling, compact unmasked features (*p* = 4, 12: 583.0, 587.7 kg ha<sup>-1</sup>) perform comparably to masked *p* = 12 (578.9 kg ha<sup>-1</sup>).

**Table S5.** Empirical coverage (%) and mean interval width (kg ha<sup>-1</sup>) of nominal 90% fixed-center source out-of-fold (OOF) prediction intervals on the non-N0 fertilized holdout (*n* = 16–21 per target field) before (*k* = 0) and after (+3N0) three-plot calibration.

{chr(10).join(s5_lines)}

Note: Source out-of-fold calibration quantiles follow the finite-sample order statistic ⌈0.90(*n*<sub>src</sub> + 1)⌉ across *n*<sub>src</sub> = 21 (2023 F1, rank 20/21), 16 (2024 F1, rank 16/16), and 18 (2024 F2, rank 18/18) fertilized source plots; for *n*<sub>src</sub> ≤ 18, this finite-sample ceiling coincides with the sample maximum, contributing to conservative single-task post-calibration coverages of 81.2%–100.0% alongside wide interval diameters (1.9–3.5 Mg ha<sup>-1</sup>). At matched *p* = 12 under +3N0, Canopeo leaf-masked *N*+Masked ResNet-50+3N0 achieves 88.9% coverage with a 776.2 kg ha<sup>-1</sup> narrower interval than unmasked *p* = 12 (2446.3 versus 3222.5 kg ha<sup>-1</sup>, 92.9% coverage), whereas unmasked *p* = 52 drops to 74.6% (27.8% in 2023 F1 → 2024 F2). Evaluated separately on the unsampled *n* − 3 holdout under commercial UAV quantile-stratified *k* = 3 calibration, six-task mean 90% PI empirical coverage reaches 93.0% for *N*-only (610.4 kg ha<sup>-1</sup>), 95.6% for *N*+Leaf+*T* (601.1 kg ha<sup>-1</sup>, width 2634.7 kg ha<sup>-1</sup>), 90.8% for *N*+Masked ResNet-50 (*p* = 12; 578.9 kg ha<sup>-1</sup>, width 2446.3 kg ha<sup>-1</sup>), 92.1% for *N*+Unmasked ResNet-50 (*p* = 12; 587.7 kg ha<sup>-1</sup>, width 3222.5 kg ha<sup>-1</sup>), and 89.7% for *N*+Unmasked ResNet-50 (*p* = 52; 684.0 kg ha<sup>-1</sup>).

**Table S6.** Two-by-two chromatic-index (ExG versus NGRDI) by leaf-masking (Whole-ROI versus Canopeo-Leaf) and canopy-openness (*N*+Leaf+Gap+*T*) ablation across six spatial resolutions (nine-task mean Total and Within-*N* ΔRMSE relative to *N*-only, kg ha<sup>-1</sup>).

{chr(10).join(s6_lines)}

Note: Each four-feature model pairs fractional green cover, *T*<sub>auc,mean</sub>, and *T*<sub>late,change</sub> with one chromatic formulation; *N*+Leaf+Gap+*T* adds 128/130 DAS *S*<sub>gap,large</sub> and *S*<sub>prof,across</sub> (*p* = 6). At sub-decimeter scales (3.3 mm–5 cm), Canopeo leaf segmentation adds +13.2 to +14.5 kg ha<sup>-1</sup> Total ΔRMSE beyond whole-ROI NGRDI (34.1–35.0 versus 19.6–21.8 kg ha<sup>-1</sup>), whereas at ≥ 20 cm–3 m apparent Canopeo cover saturates (97.4%–99.8%) so leaf-masked and whole-ROI metrics converge while blue-sensitive ExG turns negative (−7.9 to −138.5 kg ha<sup>-1</sup>) and green–red ratio NGRDI remains positive (+29.3 to +39.2 kg ha<sup>-1</sup>). At 181 DAS (*k* = 0 across six fertilized holdouts), *N*+*C*+*T*+*B*<sub>morph</sub> lowers mean Total RMSE from 1032.0 (*N*+*C*+*T*) to 650.1 kg ha<sup>-1</sup> by simultaneously reducing domain bias magnitude, Between-*N* RMSE (379.0 to 325.0 kg ha<sup>-1</sup>), and Within-*N* RMSE (423.6 to 375.3 kg ha<sup>-1</sup>, versus 439.5 kg ha<sup>-1</sup> for unfiltered *B*<sub>raw</sub>).

**Table S7.** Six-block pooled-source leave-one-environment-out (LOEO) cross-validation across the three environments, resolving single-source three-block ridge penalty boundary selection.

{chr(10).join(s7_lines)}

Note: Pooling two source trials provides six spatial blocks (*n*<sub>src</sub> = 40–45) for inner block CV, selecting interior penalties (*α* = 0.1–100) when source domains share consistent slopes: *N*+Leaf+*T* reduces uncalibrated RMSE by 64.6 kg ha<sup>-1</sup> (Within-*N* −48.5 kg ha<sup>-1</sup>) on 2024 F2 and +3N0 RMSE by 135.4 kg ha<sup>-1</sup> on 2023 F1, whereas releasing the penalty to *α* = 0.1 on 2024 F1 + 2024 F2 → 2023 F1 increases uncalibrated RMSE by 324.0 kg ha<sup>-1</sup> (1474.9 versus 1150.9 kg ha<sup>-1</sup>, reflecting inter-annual radiometric/slope offset prior to local intercept shift) and 2023 F1 + 2024 F2 → 2024 F1 still retains *α* = 10<sup>5</sup>. Additional pre-topping, photometric, statistical, and runtime audits: (i) restricted strictly to 92–107 DAS before final N fertigation (*I*<sub>Leaf,early</sub> = mean *z* of 92 & 106/107 DAS NGRDI<sub>leaf</sub> and early cover change; *I*<sub>Gap,early</sub> = mean *z* of 92 & 106/107 DAS *S*<sub>gap,large</sub> and *S*<sub>prof,across</sub>), the early 2D plane explains *R*<sup>2</sup> = 0.813 (*R*<sup>2</sup><sub>LOBO</sub> = 0.790) of within-block anomalies (which include multi-rate N variance; strictly out-of-sample training-standardized 92–142 DAS 2D plane without target *z*<sub>blk</sub> yields *R*<sup>2</sup><sub>LOBO</sub> = 0.521 and *R*<sup>2</sup><sub>LOFO</sub> = 0.527), and single-date 78 DAS squaring NGRDI<sub>leaf</sub> achieves ROC AUC = 0.922–0.933 in 2024 F1/F2 (moderate 75–150 versus 225–300 kg N ha<sup>-1</sup> ROC AUC = 0.696; excess 375–450 versus 225–300 ROC AUC = 0.623); (ii) bare-soil reference patches showed 1.77× inter-annual median DN scaling (0.762 in 2023 F1 versus 1.347 and 1.272 in 2024 F1/F2); offline photometric stress tests on 15 plot orthomosaic crops confirmed that uniform exposure scaling (×0.70–×1.30) left NGRDI<sub>leaf</sub> virtually unchanged (|ΔNGRDI<sub>leaf</sub>| ≤ 0.0002) while shifting Canopeo 2*G* − *R* − *B* > 20 mask IoU to 0.882–0.930 (−7.6 to +3.9 pp cover), whereas simulated overcast blue-shift AWB (*R*×0.90, *G*×0.95, *B*×1.12) reduced IoU to 0.677 (−18.1 pp cover, ΔNGRDI<sub>leaf</sub> = +0.039); (iii) Benjamini–Hochberg FDR adjustment of Fig. 2e partial correlations (*df* = 36) confirms significant retention for *T*<sub>auc,mean</sub> (*q* = 0.0086) and all three structural openness metrics (*q* = 0.0086–0.0232; *C*<sub>cover</sub> *q* = 0.0698, ExG *q* = 0.1015), while exact 2<sup>9</sup> = 512 deduplicated whole-block sign-flip permutations across the nine unique physical target blocks yield *P*<sub>exact</sub> = 0.066 across cross-environment transfers (7/9 positive blocks; BlockCV *P*<sub>exact</sub> = 0.656); and (iv) 4-thread CPU feature extraction required 49.0 ms plot<sup>-1</sup> (32.5 s ha<sup>-1</sup>) at 3.3 mm, 14.7 ms plot<sup>-1</sup> (9.8 s ha<sup>-1</sup>) at 1 cm, and 0.59 ms plot<sup>-1</sup> (0.39 s ha<sup>-1</sup>) at 5 cm versus 13.6 ms plot<sup>-1</sup> (9.0 s ha<sup>-1</sup>) for ResNet-50 (RGB-only Leaf+Gap+*T* *p* = 4 comprises [*I*<sub>Leaf</sub>, *I*<sub>Gap</sub>, *f*<sub>cover</sub>, *T*<sub>auc,mean</sub>]).
"""


def write_supporting_markdown_files() -> None:
    DOCS_NEW.mkdir(parents=True, exist_ok=True)
    (FORMAL_NEW / 'captions').mkdir(parents=True, exist_ok=True)

    # 1. Highlights
    for h in HIGHLIGHTS:
        vis_h = re.sub(r'\*|</?su[bp]>', '', h)
        assert len(vis_h) <= 85, f'Highlight too long ({len(vis_h)}): {vis_h}'
    (DOCS_NEW / 'SAT_CottonRGB_Highlights_20260901.md').write_text(
        '# **Highlights**\n\n' + '\n'.join(f'- {h}' for h in HIGHLIGHTS) + '\n',
        encoding='utf-8',
    )

    # 2. Cover Letter (synchronized with updated title, metrics, and AIA framing)
    cover_p = DOCS_NEW / 'SAT_CottonRGB_Cover_Letter_AuthorCompletionRequired_20260901.md'
    cover_p.write_text(
        f"""# **Cover Letter**

Dear Editor-in-Chief,

We are pleased to submit our manuscript entitled "{TITLE_EN}" for consideration as a Research Article in *Artificial Intelligence in Agriculture*.

Deploying computer vision and machine learning for low-altitude crop monitoring faces two persistent reliability bottlenecks under field domain shift: background confounding in mulched row crops and failure of high-dimensional deep representations under small-sample local calibration. Across 33 sub-centimeter nadir UAV RGB orthomosaics (25.0 m altitude) and three multi-rate (0–450 kg N ha<sup>-1</sup>) arid Xinjiang cotton trials (*n* = 64 complete-case plots), our study introduces a physically grounded leaf-masking, shortcut-diagnosis, and few-shot cross-environment calibration framework. Separating Canopeo-masked pure-leaf chromaticity (*I*<sub>Leaf</sub>) from contiguous canopy openness (*I*<sub>Gap</sub>) raises same-window (92–107 DAS) severe N-input deficit ROC AUC from 0.334 (ExG) to 0.879–0.926 (0.907 across fields) and supports strictly out-of-sample RGB-only adjacent-field yield prediction (*k* = 3 *R*<sup>2</sup> = 0.58 in-season; *R*<sup>2</sup> = 0.74 with 181 DAS boll filtering). Controlled equal-dimension ablations with orthogonal transfer MSE decomposition further explain why pretrained ResNet-50 features from unmasked plots degrade after three-plot zero-N check calibration (+3N0; 904.6 kg ha<sup>-1</sup> at *p* = 52) as an interaction between overparameterization (reducing *p* from 52 to 12 lowered RMSE to 694.5 kg ha<sup>-1</sup>) and plastic-mulch background shortcuts, which Canopeo leaf masking resolves (626.1 kg ha<sup>-1</sup> under +3N0 with 776.2 kg ha<sup>-1</sup> narrower 90% prediction intervals; 578.9 kg ha<sup>-1</sup> under commercial UAV quantile-stratified *k* = 3 sampling, 90.8% empirical interval coverage). Finally, simulated pixel-mixing (3.3 mm–3 m) defines task-stratified spatial resolution bounds (≤ 2–3 cm flights for screening versus 0.2–3 m unmasked NGRDI for N-integrated yield modeling). These findings on trustworthy, data-efficient agricultural vision, shortcut mitigation, and uncertainty-calibrated transfer align directly with the scope and readership of *Artificial Intelligence in Agriculture*.

We confirm that this manuscript is our original work, has not been published previously, and is not currently under consideration for publication elsewhere. All authors have approved the manuscript and agree with its submission to *Artificial Intelligence in Agriculture*.

Thank you for your time and consideration. We look forward to your positive response.

Sincerely,

Zhenggui Zhang, Ph.D., Professor (on behalf of all co-authors)
State Key Laboratory of Cotton Bio-breeding and Integrated Utilization
Institute of Cotton Research of the Chinese Academy of Agricultural Sciences
Anyang, Henan 455000, China
Email: [corresponding author email]
""",
        encoding='utf-8',
    )

    # 3. Title Page
    affiliations_only = AFFILIATIONS_BLOCK.split('\n\n\\* Corresponding author:')[0]
    (DOCS_NEW / 'SAT_CottonRGB_Title_Page_AuthorCompletionRequired_20260901.md').write_text(
        f"""# **{TITLE_EN}**

Article type: Research Article

## **Authors and affiliations**

{AUTHORS_LINE}

{affiliations_only}

## **Corresponding author**

\\* Corresponding author: Zhenggui Zhang (Email: [corresponding author email]), State Key Laboratory of Cotton Bio-breeding and Integrated Utilization, Institute of Cotton Research of the Chinese Academy of Agricultural Sciences, Anyang, Henan 455000, China.

## **Declarations**

Funding: This research was supported by the Natural Science Foundation of Xinjiang Uygur Autonomous Region (2024D01B91, 2024D01B92), the Tianshan Talent Training Program (2024TSYCQNTJ0026), the Tingzhou Science and Technology Innovation Team (2023CT09), the China Postdoctoral Science Foundation (2024T171024, 2024M763621), and the Chinese Agrosystem Long-Term Observation Network (CALTON-CJ) Changji Station.

Competing interests: The authors declare no conflicts of interest.

Data availability: Processed plot-level multi-scale optical features, uncalibrated and calibrated model predictions, and analysis and plotting scripts are publicly available at https://github.com/wlcbysy/cotton-uav-rgb-leaf-masking-calibration/releases/tag/v1.0.1. Raw GeoTIFF orthomosaics are available from the corresponding author upon reasonable request.
""",
        encoding='utf-8',
    )

    # 4. Declaration of Interests
    (DOCS_NEW / 'Declaration_of_Interests.md').write_text(
        f"""# **Declaration of Competing Interest**

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in the manuscript entitled "{TITLE_EN}".

Authors: Shu-Yang Yu, Cundong Li, Pengcheng Li, Zhanlei Pan, Jian Wang, Guilan Sun, Yaopeng Zhang, Junhong Li, Menghua Zhai, Wenqi Zhao, Xiaolei Ren, Ningning Zhang, Xin Li, and Zhenggui Zhang.
""",
        encoding='utf-8',
    )

    # 5. Main Figures & Captions
    caps = build_main_captions()
    fig_md = '# Main figures\n\n' + '\n\n'.join(
        f'![{cap}](../figures/main/{stem}.png)' for _, stem, cap in caps
    ) + '\n'
    (DOCS_NEW / 'SAT_CottonRGB_Figures_V22.md').write_text(fig_md, encoding='utf-8')

    cap_only_md = '# **Figure captions**\n\n' + '\n\n'.join(cap for _, _, cap in caps) + '\n'
    (DOCS_NEW / 'SAT_CottonRGB_Figure_Captions_20260901.md').write_text(cap_only_md, encoding='utf-8')
    (FORMAL_NEW / 'captions/CAPTIONS.md').write_text(cap_only_md, encoding='utf-8')
    (DOCS_NEW / 'Figure_Captions_ZH.md').write_text(build_zh_captions(), encoding='utf-8')

    # 6. Bilingual Title/Abstract/Keywords
    (DOCS_NEW / 'SAT_CottonRGB_Title_Abstract_Keywords_ZH.md').write_text(
        f"""# {TITLE_ZH}

## English title

{TITLE_EN}

## 中文摘要（内部对照）

{ABSTRACT_ZH}

**中文关键词：** 陆地棉；低空遥感；冠层分割；养分诊断；空间分辨率

## Abstract

{ABSTRACT_EN}

**Keywords:** {KEYWORDS_LINE}
""",
        encoding='utf-8',
    )

    # 7. Main Tables
    (DOCS_NEW / 'SAT_CottonRGB_Main_Tables_V22.md').write_text(build_main_tables_md(), encoding='utf-8')

    # 8. Curated Supplementary Material (Figures S1-S2 & Tables S1-S7)
    curated_tables = build_supp_tables_extra().strip()
    supp_figs_block = '\n\n'.join([
        '![**Figure S1.** Forward-only recursive Total ΔRMSE trajectories relative to *N*-only across calendar (DAS) and thermal-time (GDD<sub>10</sub>) scales for the six tasks complementing Fig. 6a, using all available flights up to each date without cross-year stage alignment (shaded ribbons: 27-resample block intervals).](../figures/supplementary/Figure_S1_recursive_timing.png)',
        '![**Figure S2.** Plot-level yield residual maps (*Ŷ* − *Y*, Mg ha<sup>-1</sup>) and Queen-contiguity (≤ 12 m) Moran\'s *I* on the 2024 F1 orthomosaic (*k* = 3 N0 checks, *n* = 16 fertilized holdout plots). (a,b) Adjacent-field transfer (2024 F2 → 2024 F1) for *N*+Leaf+*T*+3N0 (a) and unmasked *N*+ResNet-50+3N0 (b). (c,d) Cross-year transfer (2023 F1 → 2024 F1) for *N*+Leaf+*T*+3N0 (c) and unmasked *N*+ResNet-50+3N0 (d).](../figures/supplementary/Figure_S2_spatial_residuals_2024F1.png)',
    ])

    supp_refs_block = '\n\n'.join([
        'Bendig, J., Yu, K., Aasen, H., Bolten, A., Bennertz, S., Broscheit, J., Gnyp, M.L., Bareth, G., 2015. Combining UAV-based plant height from crop surface models, visible, and near infrared vegetation indices for biomass monitoring in barley. Int. J. Appl. Earth Obs. Geoinf. 39, 79-87. https://doi.org/10.1016/j.jag.2015.02.012',
        'Gitelson, A.A., Kaufman, Y.J., Stark, R., Rundquist, D., 2002. Novel algorithms for remote estimation of vegetation fraction. Remote Sens. Environ. 80(1), 76-87. https://doi.org/10.1016/S0034-4257(01)00289-9',
        'Kawashima, S., Nakatani, M., 1998. An algorithm for estimating chlorophyll content in leaves using a video camera. Ann. Bot. 81(1), 49-54. https://doi.org/10.1006/anbo.1997.0544',
        'Mesas-Carrascosa, F.-J., de Castro, A.I., Torres-Sanchez, J., Trivino-Tarradas, P., Jimenez-Brenes, F.M., Garcia-Ferrer, A., Lopez-Granados, F., 2020. Classification of 3D point clouds using color vegetation indices for precision viticulture and digitizing applications. Remote Sens. 12(2), 317. https://doi.org/10.3390/rs12020317',
        'Meyer, G.E., Neto, J.C., 2008. Verification of color vegetation indices for automated crop imaging applications. Comput. Electron. Agric. 63(2), 282-293. https://doi.org/10.1016/j.compag.2008.03.009',
        'Tucker, C.J., 1979. Red and photographic infrared linear combinations for monitoring vegetation. Remote Sens. Environ. 8(2), 127-150. https://doi.org/10.1016/0034-4257(79)90013-0',
        'Woebbecke, D.M., Meyer, G.E., Von Bargen, K., Mortensen, D.A., 1995. Color indices for weed identification under various soil, residue, and lighting conditions. Trans. ASAE 38(1), 259-269. https://doi.org/10.13031/2013.27838',
        'Zhi, X., Chen, Q., Han, Y., Yang, B., Wang, Y., Wu, F., Xiong, S., Jiao, Y., Ma, Y., Shang, S., Lin, T., Lei, Y., Li, Y., 2025. Multi-modal feature integration from UAV-RGB imagery for high-precision cotton phenotyping: A paradigm shift toward cost-effective agricultural remote sensing. Comput. Electron. Agric. 239, 111002. https://doi.org/10.1016/j.compag.2025.111002',
    ])

    new_supp = (
        f'# **Supplementary materials for {TITLE_EN}**\n\n'
        f'{AUTHORS_LINE}\n\n'
        f'{AFFILIATIONS_BLOCK}\n\n'
        '## **1. Supplementary Figures S1–S2**\n\n'
        f'{supp_figs_block}\n\n'
        '## **2. Supplementary Tables S1–S7**\n\n'
        f'{curated_tables}\n\n'
        '## **3. Supplementary References**\n\n'
        f'{supp_refs_block}\n'
    )
    (DOCS_NEW / 'SAT_CottonRGB_Supplementary_Material_20260901.md').write_text(new_supp, encoding='utf-8')

    new_supp_tables = (
        f'# **Supplementary Tables S1–S7**\n\n{curated_tables}\n\n'
        f'## **Supplementary References**\n\n{supp_refs_block}\n'
    )
    (DOCS_NEW / 'SAT_CottonRGB_Supplementary_Tables_20260901.md').write_text(new_supp_tables, encoding='utf-8')


if __name__ == '__main__':
    write_supporting_markdown_files()
    print('Wrote supporting Q1 tables, captions, highlights, and curated supplementary markdown files.')
