# 实现与实验协议

## 计划覆盖矩阵

| 原计划任务 | 实现 | 运行入口 |
|---|---|---|
| Stage 0 冻结 UniverSeg | `segmentation.Segmenter`；严格非空 support，logits→sigmoid | `K=1 bash scripts/run_random.sh` |
| Patient-level 数据和隔离 | `data.read_manifest`、`split_manifest.py`，跨患者重复内容检查 | `check_data.sh` |
| Random K=1/2/4/8/16 | 无放回随机病例，排除同患者 | `K=8 bash scripts/run_random.sh` |
| ResNet50 / DINOv2 KNN | 官方预训练网络，归一化 cosine | `run_knn.sh --set encoder=dinov2` |
| Case bank / feature cache | 训练病例、图像/掩码内容与实际 encoder 权重指纹 | `build_bank.sh` |
| Oracle / worst / gap | 每个 query 的同一随机候选池逐一分割 | `oracle.sh` |
| Failure representation | 预测边界、困难区池化、10 维形态描述 | `features.describe` |
| Context / TTA uncertainty | 独立 context 子集方差；四个可精确逆变换的翻转 | `--set uncertainty=context/tta/none` |
| Tyche stochastic uncertainty | 官方 Tyche-TS 多样本方差 | `run_tyche.sh` |
| Candidate composite score | appearance + boundary + shape + failure | `run_failure.sh` |
| 单病例真实 downstream utility | train/val 上固定 backbone Dice 标签 | `train_utility.sh` |
| Regression / pairwise / listwise | 同一 query/state 组内 ranking + MSE 校准 | `--set loss=...` |
| Redundancy-aware selection | utility - λ max support similarity | `run_mmr.sh` |
| Marginal utility / set-conditioned | selected-set mean pool、candidate 交互、冗余和 cardinality | `train_marginal.sh`, `run_set.sh` |
| 三类 case roles | appearance / morphology / failure 三角色各选一个，排除重复 | `run_roles.sh` |
| Adaptive-K | uncertainty 分段 K + MMR | `run_adaptive.sh` |
| Learnable stopping | 预测 ΔDice < epsilon 后停止（先满足 min_k） | `run_adaptive_learned.sh` |
| 区域修复/损害分解 | 相对粗预测的错误修复/新增错误，用双头监督预测 | `train_correction.sh`, `run_repair.sh` |
| 纠错覆盖与代价停止 | 按区域覆盖累积修复潜力并惩罚损害；达到 min_k 后可停止 | `run_repair_cover.sh`, `run_repair_adaptive.sh` |
| 数据集 / 跨中心 | ISIC、fundus manifest/config；domain allowlists | `CONFIG=... --set bank_domains=...` |
| Ablations | K、特征、selection、uncertainty、loss、one/two-pass | `run_ablations.sh` |
| 指标 / utility / diversity / cost | per-case JSON、病例/患者汇总、support 次数 | `report.sh` |
| 差异统计 | patient-clustered paired bootstrap | `compare_results.py` |

计划中的 future directions（3D CT/MRI、signed-distance CNN 形态编码、FAISS 大规模索引）不是当前 2D 第一篇实现的默认功能。
方案中的 Iris 和 ICL-SAM 是相关工作引用，不混入当前 backbone 对比。

## 创新实现细节

### 1. Utility-aware retrieval

输入由 query/candidate global feature、差值、乘积构成，按 feature_mode 加入形态、边界和困难区。
冻结 encoder；MLP 学习真实 `Dice(F(q,{r}), yq)`。
query 的预测形态和 uncertainty 来自 coarse support 的第一次分割，**不使用 query GT**。
`feature_mode=image` 关闭第一次分割，成为真正的一阶段 utility baseline。

每个训练 step 以一个 query 为单位；pairwise/listwise 只在同一个 query、相同 selected-set 条件内排序。
不比较来自不同 query 的任意候选，否则标签顺序没有检索意义。
pairwise loss 为所有严格有序 pair 的 logistic loss；相同标签无 ranking 项。
加 MSE 辅助项，避免仅排序分数未经标定而被拿来执行 stopping。

### 2. Failure-aware second pass

coarse KNN → frozen segmentation → context variance 或 TTA variance →
query boundary/hard-region/shape → re-score candidates → second segmentation。
困难区定义为 uncertainty 严格大于指定分位数，空区域退回 global spatial mean；
不会读取实际错误像素图，因为测试阶段不存在 query GT。
历史 support 可使用 GT 边界，困难区域候选表示以该边界为代理。
**这里的 failure-aware 是可观测 uncertainty 的代理，不等于已知真实错误位置。**

TTA 使用 identity/H-flip/V-flip/HV-flip，对 query/support 同步变换并逆变换预测；
避免非整像素旋转和缩放引入额外的插值方差。
Context 子集数可配置，若所有可用 support 必须全部取用，则方差可能自然为 0。

### 3. Set-conditioned complementarity

集合输入增加 `mean(z_s)`、candidate 与 pooled-set 的逐维乘积、最大相似度、归一化集合长度。
对 support 顺序不敏感。训练标签：

`ΔU(r|S) = Dice(F(q,S∪{r}), yq) - Dice(F(q,S), yq)`。

空集合定义 baseline utility 为 0，不调用需要非空 support 的分割模型。
每个 query 默认两条轨迹：一条真值 greedy，一条随机，覆盖不同训练状态；
所有候选在相同状态上计算真实 ΔDice，保存负收益。
推理每加一个 support 重新评价尚未选中的候选。
当前 set feature 使用初次 failure 表示和递增集合，不在每一步重新生成 uncertainty；
这是训练与推理一致的协议选择。

### 4. Adaptive-K

heuristic：初次 uncertainty mean 按阈值映射 K，再做 MMR。
learned：按预测边际收益逐步加入，达到 min_k 且下一收益低于 epsilon 后停止。
阈值只允许在 val 上确定，测试不根据标签调整。
必须报告最终 K 和所有分割调用的累计 support_evaluations；
二次检索最终 K 较小并不必然意味着整条流程更省算力。

### 5. 区域修复与损害预测

固定同一 query 的 coarse support 得到初次预测 `P0`，并用训练 GT `Y` 对每个候选的
单独分割结果 `Pr = F(q,{r})` 计算区域标签。将 128×128 图像分为 `correction_grid²` 个区域，
每个区域 `A` 的两个标签为：

`repair(A,r) = |{p∈A: P0(p)≠Y(p), Pr(p)=Y(p)}| / |A|`

`harm(A,r) = |{p∈A: P0(p)=Y(p), Pr(p)≠Y(p)}| / |A|`

二者分别是错误像素被修正、原本正确像素被破坏的**区域像素占比**。
`repair-harm` 精确对应区域 pixel accuracy 增量，**不是 Dice 增量**。
训练 GT 只用于生成这两个标签；推理使用 query 图像、初次预测、uncertainty、
候选 support 已知标签及图像特征，不访问 query GT。区域状态还包括局部图像特征、
初次预测均值、边界密度和位置坐标。

共享 MLP 对每个 query/candidate/region 输出 repair、harm、unchanged 三类质量，
softmax 保证非负且修复+损害≤1；只保存前两项。
监督损失是区域 MSE 加同 query 候选的净像素收益 pairwise/listwise 排序；
`repair` 直接选择预测净收益最高的 K 个候选。
一次监督生成需要每个 query 的 N 次单候选分割，加初次预测/uncertainty 成本；
与现有单病例 utility 生成同阶，低于集合 marginal 的大量组合推理。

### 6. 纠错覆盖驱动的互补选集

对于已选集合 S，用单候选 repair/harm 预测定义代理目标：

`C(S)=Σ_A w_A [max_{r∈S} repair_hat(A,r) - λ_h max_{r∈S} harm_hat(A,r)]`。

`w_A` 可按初次 uncertainty 增权，默认均匀；同一区域第二个相似候选的额外
repair 覆盖通常较小。`repair_cover` 贪心选择固定 K，`repair_adaptive` 在达到
`min_k` 后，若最佳候选边际代理收益不超过 `correction_cost` 则停止。
完整推理仍把最终 support set 送入冻结分割器；最后的 Dice 来自**实际联合预测**。
`C(S)` 只是基于单候选结果的覆盖代理，增加病例可能降低实际 Dice；
`max harm` 惩罚也不能保证目标单调、次模或临床安全。

新增监督文件仅保存 query 基础特征、区域状态、候选 ID 和小型标签。
训练时通过病例库 ID 取一次保存的 candidate 特征，临时展开输入；不重复落盘
`N × regions × feature_dim` 向量。旧 utility/marginal 缓存协议保持不变。

消融需固定同一 bank、候选池、初次 support、K、backbone、encoder：
全局 utility / 单候选区域 repair 排序 / coverage 贪心 / Adaptive-K；
repair-only（λ_h=0）、repair−harm、uniform/uncertainty weights、2×2/4×4/8×8。
报告 Dice、NSD、修复/损害占比、Dice 下降病例比例、实际 K、推理次数；
参数只在 val 上选择。

相关工作边界：Iris 已有预测后对象级检索，Context Matters 已联合研究 support
选择与失败检测，Cycle Context Verification 已利用模型自身预测改进上下文，
Uncertainty-Gated Region-Level Retrieval 已研究不确定区域检索及检索伤害。
这里应验证“候选级双结果监督 + 纠错覆盖选集”的增量，而不能宣称首次区域检索或
首次分析负迁移。参考：[Iris](https://arxiv.org/html/2503.19359v1)、
[Context Matters](https://arxiv.org/html/2608.05333v1)、
[CCV](https://papers.miccai.org/miccai-2025/paper/0254_paper.pdf)、
[Uncertainty-Gated Region-Level Retrieval](https://arxiv.org/pdf/2512.18082)。

## 实验公平性与防泄漏

1. 所有常规方法使用相同 train-only bank、task 约束、patient 排除规则。
2. train query 可来自训练集，但不能检索到同一患者任何病例。
3. val/test 患者全局互斥；val 标签只用于监督验证和模型选择。
4. 所有 query 特征都从图像与模型预测构造；评估完成选择和预测之后才读取 query mask。
5. 内容哈希为了审计可能读取 mask 文件，但哈希值不进入网络或检索打分。
6. 测试集不能进入 `generate` 的监督接口。禁止 test tuning。
7. 缓存/监督/模型校验 manifest、encoder 实际权重、segmenter 实际权重及特征协议指纹。
8. utility、marginal 与 correction checkpoint 不可混用；改变 uncertainty 或 feature_mode 需要匹配的新监督。
9. Oracle 是独立 GT 诊断；同池 random/knn/oracle/worst 公平对比。池外 KNN 和多 support 方法不能直接减同池 Oracle 得到所谓全局 gap。
10. 未验证 pretrained dataset overlap 时不得使用 unseen task 结论。把目标任务与 UniverSeg MegaMedical 清单逐项比对并保存审计证据。

## 指标和资源口径

- 分割评估在 128×128 网格上，间距按原始 H/W 补偿；这不等价于原分辨率结果。
- HD95 和 NSD 使用 surfel 面积加权。NSD tolerance 的单位同 manifest spacing。
- 单方空 mask 的 HD95 为 null；均值只统计有限样本并单独报告 undefined 数。
- timing 在 GPU 上同步，含每个 query encoder、初次分割、uncertainty、检索和最终分割；不含模型加载、bank 建设、读取 GT、保存图像。
- `retrieval_seconds = elapsed - segmentation_seconds`，包含 encoder、query I/O 和 Python 调度，不能解读成纯向量搜索时间。
- GPU memory 是单 query 流程峰值已分配显存，包含已加载 backbone/encoder；非全进程 RSS。
- Utility@K 可选且需额外运行 K 次单 support 推理；独立计时，不纳入方法在线开销。
- Correction supervision 约 Q×N 次单候选推理；compact 存储限制磁盘，不能降低标签生成成本。
- Oracle 和 marginal supervision 成本高：约 Q×N 和 Q×T×Σ(N-depth) 次推理；先在服务器小规模确认再扩展。
- 当前按 query 生成标签落盘，避免全部 pair features 常驻 GPU；bank CPU 图像/掩码随数据量线性增长。

## 可复现性

按 seed 设置 Python/NumPy/Torch；每个 query 重置稳定的 case seed，使缓存跳过不会改变该 query 的随机 context。
确定性设置采用 warn_only：部分 CUDA 算子可能无法完全确定，因此仍需多 seed 评估。
setup 锁定 PyTorch/UniverSeg/Tyche 版本，记录其他依赖 freeze 与 UniverSeg revision；实际 encoder 和 segmenter 权重纳入 hash。
重跑论文实验前还应固定 DINOv2 源码 revision，可用 encoder_repo 本地路径。

## 交付验证边界

此次交付未执行医学数据实验，也未报告收益。静态 AST/Bash 校验只能发现语法问题，
不能证明上游动态依赖、权重下载或特定 CUDA 环境已通过集成测试。
`tests/test_invariants.py` 提供无需权重的逻辑测试；在安装后的 Linux 环境执行。
