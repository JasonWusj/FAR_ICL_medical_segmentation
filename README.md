# FAR-ICL: Failure-Aware Retrieval for Medical In-Context Segmentation

围绕 **Similar → Useful → Corrective → Complementary** 的完整研究实现。
冻结分割 backbone 与图像 encoder，训练 utility / marginal-utility 检索器。
原始方案见 [实施计划](docs/implementation_plan.md)，实现对应关系见 [研究协议](docs/RESEARCH_PROTOCOL.md)。

**交付状态：代码实现与静态检查完成；未在本地下载模型权重、运行训练、分割推理或医学数据实验。没有预设的实验成绩。**
运行前需要自行准备有权限使用的数据、患者元信息和 Linux 环境。此项目不能保证未实际执行的 GPU 兼容性或算法提升。

## 已实现的方法

| 方法 | 一键训练/验证（默认 val） | 是否训练检索器 |
|---|---|---|
| Random context | `bash scripts/run_random.sh` | 否 |
| Appearance KNN | `bash scripts/run_knn.sh` | 否 |
| Appearance + Shape | `bash scripts/run_shape.sh` | 否 |
| Failure-aware second pass | `bash scripts/run_failure.sh` | 否，组合相似度 |
| Utility Top-K | `bash scripts/run_utility.sh` | 是，MLP |
| Utility + MMR | `bash scripts/run_mmr.sh` | 是，复用 utility |
| Set-conditioned marginal utility | `bash scripts/run_set.sh` | 是，集合条件 MLP |
| Uncertainty Adaptive-K + MMR | `bash scripts/run_adaptive.sh` | 是，utility |
| Learned stopping Adaptive-K | `bash scripts/run_adaptive_learned.sh` | 是，marginal utility |
| 区域修复/损害排序 | `bash scripts/run_repair.sh` | 是，区域修复/损害预测 |
| 纠错覆盖互补选集 | `bash scripts/run_repair_cover.sh` | 是，复用区域模型 |
| 纠错收益自适应停止 | `bash scripts/run_repair_adaptive.sh` | 是，复用区域模型 |
| Appearance / Morphology / Failure roles | `bash scripts/run_roles.sh` | 否，最多 3 个不同病例 |
| Oracle gap | `bash scripts/oracle.sh` | 否，仅使用 GT 的诊断 |
| Tyche-TS backbone / uncertainty | `bash scripts/run_tyche.sh` | 默认 utility + MMR |

每种方法都有独立 `scripts/eval_<method>.sh`，仅验证，不生成监督、不训练。
`run_*` 对可学习方法依次生成 train/val 监督、训练、选择验证集最佳 checkpoint、验证；
无训练参数的方法直接执行评估，避免把基线推理称为训练。

## Linux 快速开始

建议 Python 3.10/3.11、NVIDIA GPU。默认安装 PyTorch 2.5.1 / torchvision 0.20.1 CUDA 12.4 wheel；
请根据服务器驱动选择 `TORCH_INDEX_URL`。脚本可从任意工作目录启动。

```bash
git clone https://github.com/JasonWusj/FAR_ICL_medical_segmentation.git
cd FAR_ICL_medical_segmentation
bash scripts/setup_linux.sh
# CPU 安装选项（完整实验仍建议 GPU）
# TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash scripts/setup_linux.sh
```

脚本创建 `.venv`，安装项目和官方 UniverSeg，记录 `environment.lock.txt` 与上游 commit。
首次使用预训练 encoder/backbone 会联网下载权重。支持离线权重：
`--set encoder_weights=/path/resnet50.pth --set segmenter_weights=/path/universeg.pth`。
DINOv2 使用 `--set encoder=dinov2`；离线同时设置 `encoder_repo` 为官方源码目录及 `encoder_weights`。

### 1. 准备数据清单

复制 [manifest 模板](configs/manifest.example.csv)，填入真实数据。每行是一个 2D 二值分割任务：

```csv
case_id,patient_id,image_path,mask_path,split,task,domain,spacing_y,spacing_x
case001,patient001,images/001.png,masks/001.png,train,lesion,center_a,1,1
```

- `case_id` 唯一；`patient_id` 来自真实患者元数据且应全局唯一，不能用 slice/image ID 代替患者 ID。
- `split` 是 `train/val/test`；病例库仅取 train，同患者和不同 task 的 support 均被排除。
- 图片路径相对于 CSV 目录；支持 PNG/JPEG/TIFF 等 PIL 图像和二维/HWC `.npy`，不直接读取 3D DICOM/NIfTI。
- 灰度/RGB 图像归一化，分割输入灰度 128×128；mask 最近邻缩放。
- 二值标签 0/1 或 0/255 自动处理；多类别必须配置 `mask_values`，例如 cup/disc 应按实际编码分别配置。
- 眼底 palette PNG 保留标签索引；不接受彩色 RGB mask 自动猜类别。
- `spacing_y/x` 默认 1，表示原图像素间距。物理间距未知时距离单位是**原图像素**，不能报告为 mm。
- 严格检查患者跨 split、重复 case ID、跨患者/跨 split 的相同图像文件内容及缺失文件。
  内容哈希不能识别不同压缩或重采样版本，仍须核对患者元数据。

未划分的清单可用：

```bash
.venv/bin/python scripts/split_manifest.py data/isic/all.csv data/isic/manifest.csv --seed 42
bash scripts/check_data.sh
```

split 工具按患者随机分组；不是官方划分或分层划分的替代品。正式实验优先填写官方 patient-level split。
`query_domains` 仅过滤 val/test；训练 query 的中心可另用 `train_query_domains` 配置。
训练集每个 task 至少需要两个不同患者。验证和测试也必须有相同 task 的训练 support。

可先在 Linux 上用指定病例执行一次接口 smoke test（本次未执行）：

```bash
.venv/bin/python scripts/smoke_backbone.py --query case003 --support case001
```

### 2. 分别训练和验证

在 `configs/isic.yaml` 修改 manifest/output。眼底使用 `CONFIG=configs/fundus.yaml`。

```bash
# 先确认 Oracle gap，输出同一候选池的 random / KNN / oracle / worst
bash scripts/oracle.sh

# 每项都能独立运行，自动处理对应的监督与 checkpoint
CUDA_VISIBLE_DEVICES=0 bash scripts/run_utility.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mmr.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_set.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_adaptive_learned.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_repair.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_repair_cover.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_repair_adaptive.sh

# 仅训练；自动生成 train/val 监督，可断点恢复
bash scripts/train_utility.sh
bash scripts/train_marginal.sh
bash scripts/train_correction.sh

# 仅验证已训练模型；自动按当前配置定位 checkpoint
bash scripts/eval_utility.sh
bash scripts/eval_set.sh
bash scripts/eval_repair_cover.sh
# 或指定已有 checkpoint
bash scripts/eval_set.sh --checkpoint /absolute/path/to/best.pt

# 锁定验证集选择后的配置，再运行测试
SPLIT=test bash scripts/eval_set.sh
SPLIT=test K=8 bash scripts/eval_knn.sh

# 其他数据、encoder、loss、K、seed
CONFIG=configs/fundus.yaml SEED=43 K=4 bash scripts/run_mmr.sh --set encoder=dinov2
bash scripts/run_utility.sh --set loss=listwise
bash scripts/run_utility.sh --set feature_mode=image  # 单次分割的 appearance utility 消融
bash scripts/run_failure.sh --set uncertainty=tta
bash scripts/run_set.sh --set retrieval_diagnostics=true
```

`PYTHON=/absolute/python` 可指定解释器；默认优先 `.venv/bin/python`。
`CONFIG/SPLIT/SEED/K` 是所有方法脚本共用的环境变量。
额外配置通过可重复的 `--set key=value`，列表使用 shell 引号，例如
`--set 'adaptive_ks=[1,2,8]'`。`candidate_n` 必须覆盖 k/initial_k/max_k。
可用病例少于 K 时记录实际 K，不复制 support 补足。

`run_*` 默认 `--resume`。监督按 query 缓存；中断后跳过已完成 query。
`last.pt` 保存 optimizer、epoch、早停状态；`best.pt` 仅按 val loss 选择。
改变 loss/lr/hidden_dim 会使用不同 checkpoint 目录。
增加 `--set epochs=80` 可继续训练；`--force` 重建监督。
如需从头训练而不恢复，直接运行 CLI `train`（不带 `--resume`）。
不要同时向同一 output/config 写入；多卡独立实验请设置不同 output。

### 3. 消融与跨中心实验

```bash
# 一键完整消融（计算量大；本次交付没有执行）
SEEDS='42 43 44' bash scripts/run_ablations.sh
# 患者级跨中心：病例库 center_a，query center_b
bash scripts/run_set.sh --set 'bank_domains=[center_a]' --set 'query_domains=[center_b]' \
  --set output=runs/cross_center
# Tyche 独立安装/运行，独立输出和监督；不要跨 backbone 混用 ranker
INSTALL_TYCHE=1 bash scripts/setup_linux.sh
METHOD=set bash scripts/run_tyche.sh
```

消融涵盖 K=1/2/4/8/16、五组特征、三种 uncertainty、三种 loss、各选择策略、one/two-pass，
以及新模块的区域修复排序 / 纠错覆盖 / 自适应停止、harm 权重和 uncertainty 权重。
区域模型默认使用 4×4 网格；`--set correction_grid=8` 可检验空间粒度。
`correction_harm_weight` 控制新出错像素的惩罚；`correction_cost` 控制预测边际纠错收益的停止阈值。
这些参数只能在验证集选择，测试集不再调整。
Tyche uncertainty 使用独立脚本和配置，避免默认消融强制安装第二 backbone。

### 4. 结果与分析

```bash
bash scripts/report.sh
.venv/bin/python scripts/compare_results.py runs/isic/results/RUN_A/per_case.json \
  runs/isic/results/RUN_B/per_case.json
```

输出路径：

```text
runs/<dataset>/
  cache/<bank-hash>/                         # CPU tensor case cache
  supervision/<protocol-hash>/<kind>/<split>/ # query-grouped training labels
  checkpoints/<protocol-hash>/<kind>/<training-hash>/
    best.pt, last.pt, history.json
  results/<run>/
    config.json, provenance.json, per_case.json, summary.json
    *_mask.png, *_maps.npz, *_explanation.png # 可关闭 save_predictions
  results.csv, dice_comparison.png
```

指标：Dice、IoU、HD95、surfel-weighted NSD、患者宏平均 Dice、diversity、实际 K、
初次/二次分割收益、总时间、分割时间、其余检索流程时间、分割调用数、累计 support 使用数、GPU 峰值。
新区域方法额外输出原错误像素修复比例、原正确像素损害比例、二次分割 Dice 下降病例比例，
并在每个 query 保存所选病例的区域修复/损害预测和 `*_explanation.png` 对照图。两类比例除以全图像素数，
两者之差等于像素准确率变化，**不等于 Dice 变化**。
`retrieval_diagnostics=true` 额外输出所选病例单独分割 Dice 均值 `Utility@K`，
其计算不计入正常推理时间。它与最终集合 Dice 含义不同。
对空集：双方为空 Dice/IoU/NSD=1、HD95=0；单方为空 NSD=0、HD95=null，汇总显式报告缺失数量。
不把单方为空静默当成 HD95=0。

Oracle 仅是**相同随机候选池、K=1** 内上界，不能当作不同候选池或 K>1 的全局上界。
报告图只来自实际输出 JSON，不填充假结果。患者 bootstrap 用于配对差异，不代替多种子实验。

## 实现结构

- `data.py / bank.py`：患者隔离、预处理、train-only bank、内容和权重指纹缓存。
- `features.py`：ResNet50 / DINOv2、边界/困难区域池化、10 维 morphology descriptor。
- `segmentation.py / tyche_adapter.py`：冻结 UniverSeg 与官方 Tyche-TS。
- `retrieval.py`：utility MLP、regression/pairwise/listwise、MMR、集合条件选择与停止。
- `correction.py`：区域修复/损害双目标、纠错覆盖选集与代价感知停止。
- `pipeline.py`：统一监督生成、label-free 检索、评估与 oracle。
- `training.py`：按 query/state 排序、验证、恢复和 checkpoint。
- `report.py`：结果表、图及 patient-clustered bootstrap。

## 检查与限制

```bash
python3 scripts/static_check.py        # 无第三方依赖，不加载模型
.venv/bin/ruff check far_icl scripts tests
.venv/bin/pytest -q                    # 合成逻辑单元测试，不下载权重
```

本次仅执行静态检查，测试文件已提供但没有运行依赖 PyTorch 的测试。
目标硬件上的安装、单样本接口 smoke test、训练收敛、完整消融须在实际服务器验证。
所有方法贡献是待验证的研究假设，不代表已证明的新颖性或性能优势。
区域修复/损害监督使用单候选相对于粗预测的变化；互补覆盖是选集代理目标，
不保证联合支持集的 Dice 提升、次模性或医学安全。
新增监督按 query 保存基本特征、候选 ID、4×4 小型修复/损害标签，训练时按 ID 从病例库重建输入；
不把每个候选×区域的展开向量落盘，缓解此前 marginal 展开缓存占用。
原 `marginal` 监督仍采用展开格式，约 350 GB 的默认估算并未因这项新增创新自动下降。
UniverSeg 的预训练数据重叠必须按官方 MegaMedical 列表审计，未经核对不能声称 unseen task。
128×128 评估可能损失微小结构；本文实现是 2D 单目标协议，3D 切片工程与学习式 shape CNN 属于后续扩展。

## 上游来源

- [UniverSeg 官方代码](https://github.com/JJGO/UniverSeg)：ICCV 2023，模型输出 logits，在 wrapper 中 sigmoid。
- [Tyche 官方代码](https://github.com/mariannerakic/Tyche)：CVPR 2024，适配 Tyche-TS 的 `tychets`。
- [DINOv2 官方代码](https://github.com/facebookresearch/dinov2)：ViT-S/14 patch/global features。
- [DeepMind surface-distance](https://github.com/google-deepmind/surface-distance)：表面积加权边界指标。

上游代码、权重、数据分别遵循各自许可证；UniverSeg 预训练权重的使用限制需遵守其模型许可证。
