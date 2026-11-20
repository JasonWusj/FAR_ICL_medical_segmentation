# 冻结 UniverSeg 的四组基线实验

以下命令供操作者在 Linux GPU 服务器执行。仓库测试使用合成输入。真实 ISIC 指标、K=64 在目标显卡上的显存和耗时均待服务器验证。

## 1. 安装环境

在项目根目录执行。已有 CUDA 版 PyTorch 的 Featurize 环境可直接复用当前解释器。

```bash
cd /home/featurize/work/FAR_ICL_medical_segmentation
export PYTHON="$(command -v python)"
INSTALL_TYCHE=0 PYTHON="$PYTHON" bash scripts/setup_existing_torch.sh
"$PYTHON" -c 'import torch, torchvision, universeg; print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))'
```

新环境可改用下面的安装方式。示例为 CUDA 12.4 wheel，按服务器驱动选择兼容版本。脚本创建仓库内的 `.venv`，安装 torch 2.5.1、torchvision 0.20.1 和固定提交的官方 UniverSeg。

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 INSTALL_TYCHE=0 bash scripts/setup_linux.sh
export PYTHON="$PWD/.venv/bin/python"
```

## 2. 数据和 manifest

使用 ISIC 2018 **Task 1 病灶分割**。已有数据时直接将 `DATA_DIR` 指向解压目录，只读挂载也可。仅在新建、可写的数据目录上执行下面的下载块。[ISIC 官方下载页](https://challenge.isic-archive.com/data/#2018)

```bash
export DATA_DIR=/home/featurize/work/datasets/isic2018_task1
mkdir -p "$DATA_DIR"
(
  set -e
  cd "$DATA_DIR"
  base=https://isic-archive.s3.amazonaws.com/challenges/2018
  for split in Training Validation; do
    for archive in "ISIC2018_Task1-2_${split}_Input" "ISIC2018_Task1_${split}_GroundTruth"; do
      wget -c "$base/$archive.zip"
      unzip -nq "$archive.zip"
    done
  done
)
```

解压后的文件结构如下。每个 split 的图像和掩码 ID 必须一一对应。

```text
$DATA_DIR/
├── ISIC2018_Task1-2_Training_Input/ISIC_<id>.jpg
├── ISIC2018_Task1_Training_GroundTruth/ISIC_<id>_segmentation.png
├── ISIC2018_Task1-2_Validation_Input/ISIC_<id>.jpg
└── ISIC2018_Task1_Validation_GroundTruth/ISIC_<id>_segmentation.png
```

已有 manifest 可直接将 `MANIFEST` 设为其绝对路径。下面的示例仅在目标文件不存在时创建，并核查全部图像/掩码、身份约束和计数。默认有 2,594 张训练图、100 张验证图。ISIC 缺少已核验的患者映射，使用图像级身份。

```bash
export IDENTITY_SCOPE=image
export MANIFEST=/home/featurize/work/isic2018_baselines_train_val.csv
if [[ ! -e "$MANIFEST" ]]; then
  "$PYTHON" scripts/prepare_isic2018_exploratory.py --data-root "$DATA_DIR" --output "$MANIFEST"
fi
"$PYTHON" -m far_icl.cli check-data --config configs/isic.yaml \
  --set "manifest=$MANIFEST" --set "identity_scope=$IDENTITY_SCOPE"
```

如需在以后评估官方 1,000 张测试图，用上面的下载块将 `for split in Training Validation` 换成 `for split in Test`，在**新的 CSV 路径**生成 manifest 时追加 `--include-test`，再选择 `SPLIT=test`。比较不同方法时始终使用同一份 manifest 和同一 split。

## 3. 运行命令

导出一次公共参数。默认 K=2、N=max(64,K)、SEED=42，合法训练病例不足 K 时会报错。下例选择独立的结果目录。`CANDIDATE_N=128` 使 K=64 时，MMR 和 Dual 仍有候选可选。N=K 时，从同一 N 个不同候选中选 K 个，重排无法改变支持成员。

```bash
export OUT_BASE=/home/featurize/work/far_icl_baseline_audit
export FARICL_BANK_CACHE_ROOT="$OUT_BASE/baselines/cache"
export CUDA_VISIBLE_DEVICES=0
export SEED=42 K=2 CANDIDATE_N=128 SPLIT=val
export MMR_LAMBDA=0.5 RANDOM_REPEATS=1
export BASELINES=experiments/baselines
"$PYTHON" -m experiments.baselines.run --help
```

默认使用官方 UniverSeg v1 与 torchvision 的 ResNet50 IMAGENET1K_V2 权重。服务器首次使用会获取相应预训练权重。离线环境可将 `UNIVERSEG_WEIGHTS`、`RESNET50_WEIGHTS` 分别设为相应 state dict 的本地绝对路径。四组共享这两个模型、同一查询列表、特征缓存及指标口径。

| 单独执行 | 选择规则 |
| --- | --- |
| `bash "$BASELINES/universeg_random/run.sh"` | 每个查询从合法训练库无放回均匀抽 K 个。 |
| `bash "$BASELINES/universeg_knn/run.sh"` | 冻结 ResNet50 全局特征的余弦相似度 Top-K。 |
| `bash "$BASELINES/universeg_mmr/run.sh"` | 先取最相关病例，再最大化 λ×查询相似度 − (1−λ)×与已选成员的最大相似度。λ 可取 [0,1]，默认 0.5。 |
| `bash "$BASELINES/universeg_dual_similarity/run.sh"` | **Dual Similarity Adapted**。默认先用 2 个近邻粗分割（可通过 INITIAL_K 设置），在候选中按语义余弦和预测掩码形状余弦之和排序。 |

可以单独运行，也可以用下面的命令一次运行四组。再次运行时，入口核对代码版本、权重、配置和查询列表，跳过已完成的匹配结果，补跑缺少的组或随机重复。同名目录中的结果不完整时会报错，另设 `OUT_BASE` 可保留原文件并重新运行。

```bash
bash "$BASELINES/run_all_baselines.sh"
```

随机支持重复抽样示例（每个查询 10 次）和多 K、多种子全套示例。以下为可选方案。同一配置只需运行一次。

```bash
OUT_BASE="$OUT_BASE/random10" K=64 RANDOM_REPEATS=10 bash "$BASELINES/universeg_random/run.sh"

for k in 2 8 16 64; do
  for seed in 42 43; do
    OUT_BASE="$OUT_BASE/grid" K="$k" SEED="$seed" RANDOM_REPEATS=10 bash "$BASELINES/run_all_baselines.sh"
  done
done
```

若已有实验记录为 **K=2、N=12、验证集、种子 42/43**，请先导出该次的 `MANIFEST`，对齐原配置的权重和预处理，再运行下列对照。增大 K 会改变支持预算，增大 N 会改变可供 MMR 或 Dual 重排的集合，两者应单独报告。

```bash
for seed in 42 43; do
  OUT_BASE="$OUT_BASE/matched_k2" SPLIT=val K=2 CANDIDATE_N=12 SEED="$seed" RANDOM_REPEATS=1 \
    bash "$BASELINES/run_all_baselines.sh"
done
```

## 4. 查看和汇总结果

- 结果位于 `$OUT_BASE/baselines/results/<运行标签>/`。标签含方法、K、N、split、seed 及配置签名。Random 的每次抽样各有一套结果。比较多次采样的 Dice/IoU 时，先对同一查询的 R 个指标取均值，再对 Q 张查询等权取均值，样本数报告 Q，另列 R 和种子。HD95 另外列出有限值数和未定义数。
- `per_case.json` 记录实际支持 ID 与逐病例指标，`summary.json` 给出统计。`config.json` 保存参数，`provenance.json` 提供查询 ID、权重/缓存签名和代码版本。
- 默认共享特征缓存位于 `$OUT_BASE/baselines/cache/<签名>/`，方法、K 和随机种子可共用。若需在不同输出根复用缓存，可设置同一 `FARICL_BANK_CACHE_ROOT`。

```bash
RESULT_ROOT="$OUT_BASE/baselines"  # 对照和网格示例分别改为 "$OUT_BASE/matched_k2/baselines"、"$OUT_BASE/grid/baselines"
"$PYTHON" -m far_icl.cli report --config configs/isic.yaml --set "output=$RESULT_ROOT"
"$PYTHON" - "$RESULT_ROOT/results.csv" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        print(row["run"], *(f"{key}={row.get(key) or 'NA'}" for key in ("n_cases", "dice", "iou", "hd95", "hd95_undefined_cases")))
PY
```

`results.csv` 每行是一次运行。Dice、IoU 为逐图均值（0–1），HD95 只汇总有限值，单侧空掩码记未定义，双空记 0。ISIC manifest 使用单位像素间距，代码按原始宽高对 128 网格上的表面距离换算，结果单位是原图像素；其他数据的物理单位取决于 manifest 中给出的间距。

## 5. 协议与公开结果的对照

- **主干。** 官方固定版本的 `universeg(pretrained=True)` 加载 v1 权重，推理全程冻结。查询是 `[B,1,128,128]`，支持图与二值掩码均是 `[B,K,1,128,128]`。图像双线性缩放，仓库以加权灰度并逐图 min-max 归一化到 [0,1]，掩码用最近邻。logits 经过一次 sigmoid，以严格 `p > 0.5` 得到前景。[官方接口与固定模型版本](https://github.com/JJGO/UniverSeg/tree/833a0c34c65e38d675e21bd48ddec6797cc03259)
- **检索。** 训练集构成支持库；检索限同任务且排除相同身份。ResNet50 接收 128 的 RGB 图再缩放到 224，使用 ImageNet 均值与标准差，对末端特征图全局平均池化，再用 L2 归一化向量计算余弦。KNN 与 MMR 只用查询图像。Dual Adapted 另用初始 UniverSeg 预测估计查询形状，与支持真值的形状描述子比较。现有 FAR-ICL 的学习式 utility-MMR 与此处的无训练标准 MMR 是两组实验。
- **Gao 等的 Dual Similarity Checkup。** 原方法用按语义相似度与排名加权的支持掩码估算目标尺度先验，再筛选并比较删除、重复采样及对补充支持作随机缩放等策略。本适配保留两种相似度和固定 K，未实现原文的先验加权筛选、重复采样及增强。记录时使用 `Adapted` 标签。[原文算法 1 与实验设置](https://doi.org/10.1109/TMI.2024.3440311)
- **数值比较。** Gao 等在 2,594 张 ISIC 图像内做五折，四折供检索、一折作查询。其标准 UniverSeg 默认 K=64，每查询随机抽取支持做 10 次预测。该文 Dice 以百分数报告，HD95 表标注为像素。本手册沿用官方 train/val/test 划分。对比时并列注明来源、划分、K、随机次数及聚合方式、距离单位，外部报告值单独列示。[完整方法与对照协议，IV 节与表 I、II](https://pure.amsterdamumc.nl/files/142768030/Boosting-your-context-by-dual-similarity-checkup-for-in-context-learning-medical-image-segmentation.pdf)
