# 病例检索 + In-Context Medical Segmentation：可落地实现方案

> 目标：以 **UniverSeg** 为基础分割器，先复现“随机 / 相似病例检索”的 baseline，再逐步实现
> **Failure-Aware Retrieval + Set-Conditioned Retrieval + Adaptive-K**。
> 核心研究问题：**最相似的病例，不一定是对当前分割最有帮助的病例。**

---

## 1. 推荐的主 Baseline

### 1.1 首选：UniverSeg

**官方仓库：**
https://github.com/JJGO/UniverSeg

**论文：**
*UniverSeg: Universal Medical Image Segmentation*，ICCV 2023
https://openaccess.thecvf.com/content/ICCV2023/html/Butoi_UniverSeg_Universal_Medical_Image_Segmentation_ICCV_2023_paper.html

推荐它作为第一版 baseline 的原因：

1. 官方代码和预训练权重公开；
2. 推理接口非常简单；
3. 天然就是 `query + support image-mask pairs -> query mask`；
4. 不需要先训练一个新的 segmentation backbone；
5. 我们可以把绝大部分实验变量集中在 **support / case retrieval** 上；
6. 很适合做 Oracle Retrieval、Top-K Retrieval、Set Retrieval 等消融；
7. 模型规模和输入尺寸都比较友好，第一版验证成本低。

官方接口大致如下：

```python
from universeg import universeg

model = universeg(pretrained=True)

pred = model(
    target_image,      # [B, 1, H, W]
    support_images,    # [B, S, 1, H, W]
    support_labels,    # [B, S, 1, H, W]
)
```

官方实现要求输入做 min-max normalization 到 `[0, 1]`，默认空间尺寸为 `128 x 128`。

### 1.2 安装

```bash
git clone https://github.com/JJGO/UniverSeg.git external/UniverSeg

pip install -r external/UniverSeg/requirements.txt

export PYTHONPATH=$PYTHONPATH:$(realpath external/UniverSeg)
```

也可以直接：

```bash
pip install git+https://github.com/JJGO/UniverSeg.git
```

---

## 2. 第二阶段 Baseline：Tyche

**官方仓库：**
https://github.com/mariannerakic/Tyche

Tyche 同样属于医学图像 In-Context Segmentation，但它可以输出多个 stochastic segmentation。

这对于我们后面的：

```text
预测分歧
   ↓
uncertainty map
   ↓
寻找当前模型不确定的位置
   ↓
重新检索能够纠正这些位置的病例
```

非常合适。

所以建议：

- **第一版：UniverSeg**
- **第二版：Tyche 替换 / 补充 UniverSeg**
- 不要一开始同时改 backbone 和 retrieval，否则实验变量太多。

---

# 3. 论文核心假设

传统病例检索通常优化：

\[
r^* = \arg\max_r Sim(x_q, x_r)
\]

即：

> 给 query 找“最像”的病例。

但我们真正希望优化的是：

\[
r^* =
\arg\max_r
Performance(
F(x_q,r),
y_q
)
\]

即：

> 找到“最能帮助当前 query 得到正确分割”的病例。

因此：

\[
\boxed{Similarity \neq Utility}
\]

我们的课题可以围绕这个差异展开。

推荐论文主标题方向：

> **Beyond Similarity: Failure-Aware and Set-Conditioned Retrieval for In-Context Medical Image Segmentation**

或者：

> **Retrieve What You Fail: Failure-Aware Case Retrieval for In-Context Medical Image Segmentation**

---

# 4. 第一件必须做的实验：Oracle Retrieval Gap

不要马上训练复杂模型。

第一件事是验证：

> **病例选择到底是不是当前任务的瓶颈？**

## 4.1 实验方法

对每个 query：

```text
query q
  │
  ├── candidate case 1 → UniverSeg → Dice1
  ├── candidate case 2 → UniverSeg → Dice2
  ├── candidate case 3 → UniverSeg → Dice3
  │
  ...
  └── candidate case N → UniverSeg → DiceN
```

假设从 case bank 随机抽 50 个候选病例：

\[
C_q=\{r_1,\dots,r_{50}\}
\]

计算：

\[
u(q,r_i)=Dice(F(q,r_i),y_q)
\]

然后比较：

### Random

随机选一个 support。

### Appearance Top-1

图像 embedding 最相似的 support。

### Oracle Top-1

真正使 Dice 最高的 support：

\[
r_{oracle}=\arg\max_i u(q,r_i)
\]

### Worst

使分割性能最低的病例。

## 4.2 我们最想看到的情况

例如：

| Strategy | Dice |
|---|---:|
| Random | 0.72 |
| Image Similarity Top-1 | 0.76 |
| Oracle Top-1 | 0.86 |
| Worst | 0.48 |

则：

\[
RetrievalGap = 0.86-0.76=0.10
\]

说明：

> 现有 embedding 可以找到“像的病例”，但离“真正有帮助的病例”还有很大距离。

这个 **Retrieval Gap** 就是后面论文的动机。

如果：

```text
Similarity Top-1 = 0.84
Oracle          = 0.85
```

则说明这个数据集上病例 retrieval 不是主要瓶颈，不值得继续投入太多。

---

# 5. 第一版工程结构

建议不要直接在 UniverSeg 源码里乱改。

保持它基本不变，把创新写在外面。

```text
far_icl/
│
├── external/
│   └── UniverSeg/
│
├── configs/
│   ├── isic.yaml
│   └── fundus.yaml
│
├── datasets/
│   ├── base.py
│   ├── isic.py
│   └── fundus.py
│
├── case_bank/
│   ├── build_bank.py
│   ├── feature_cache.py
│   └── case_bank.py
│
├── encoders/
│   ├── image_encoder.py
│   ├── shape_encoder.py
│   └── failure_encoder.py
│
├── retrieval/
│   ├── random_retriever.py
│   ├── knn_retriever.py
│   ├── utility_reranker.py
│   ├── set_selector.py
│   └── adaptive_k.py
│
├── segmentation/
│   └── universeg_wrapper.py
│
├── uncertainty/
│   ├── context_disagreement.py
│   └── tta_uncertainty.py
│
├── scripts/
│   ├── 00_test_universeg.py
│   ├── 01_build_case_bank.py
│   ├── 02_random_baseline.py
│   ├── 03_knn_baseline.py
│   ├── 04_oracle_gap.py
│   ├── 05_train_utility_ranker.py
│   ├── 06_failure_aware_eval.py
│   └── 07_set_retrieval_eval.py
│
├── utils/
│   ├── metrics.py
│   ├── seed.py
│   └── visualization.py
│
├── train.py
├── evaluate.py
└── README.md
```

原则：

> **UniverSeg 是固定的分割器；我们的研究对象是 retrieval。**

第一阶段最好直接 freeze segmentation model。

---

# 6. 数据划分：一定注意 Patient Leakage

医疗数据尤其要注意：

```text
同一个 patient 的不同 slice
```

不能同时出现在：

```text
query/test set
```

和：

```text
case bank
```

否则 nearest-neighbor retrieval 很容易产生严重的数据泄漏。

正确方式：

```text
Patient-level split

Train patients
  ├── train query
  └── train case bank

Validation patients
  └── validation query

Test patients
  └── test query

Retrieval Case Bank
  └── 只能来自允许使用的 support / train patients
```

如果做跨中心实验：

```text
Hospital A
    ↓
Case Bank

Hospital B
    ↓
Queries
```

这个实验会更加符合“历史病例辅助新病例”的实际故事。

---

# 7. Baseline 1：Random Context

第一版先验证 UniverSeg 跑通。

对于 query：

```python
support_ids = random.sample(case_bank, k)
```

获得：

```python
support_images
support_masks
```

然后：

```python
pred = universeg(
    query,
    support_images,
    support_masks
)
```

分别测：

```text
K = 1
K = 2
K = 4
K = 8
K = 16
```

记录：

- Dice
- IoU
- HD95
- NSD / Surface Dice
- inference time

这就是最基本的 ICL baseline。

---

# 8. Baseline 2：Appearance KNN Retrieval

## 8.1 Case Bank

给每个病例提前抽 image embedding：

\[
z_i=E_{img}(x_i)
\]

保存：

```text
case_id
patient_id
image_path
mask_path
image_embedding
```

推荐第一版 encoder：

### 简单版

- ResNet50 ImageNet feature

### 推荐版

- DINOv2 feature

如果医学灰度图是单通道，可以：

```python
img = img.repeat(3, 1, 1)
```

再送入 encoder。

## 8.2 Retrieval

Query feature：

\[
z_q=E_{img}(x_q)
\]

Cosine similarity：

\[
s_i=\frac{z_q^\top z_i}{\|z_q\|\|z_i\|}
\]

取：

\[
TopK(s_i)
\]

第一版不用 FAISS 都可以。

病例数大后再换：

```bash
pip install faiss-cpu
```

---

# 9. 创新 1：Failure-Aware Second-Pass Retrieval

这是推荐作为主创新之一的部分。

普通检索：

```text
Query
  ↓
global similarity
  ↓
Top-K
  ↓
segmentation
```

我们的检索：

```text
Query
  ↓
coarse retrieval
  ↓
first segmentation
  ↓
Where does the model fail?
  ↓
failure representation
  ↓
retrieve corrective cases
  ↓
second segmentation
```

数学形式：

\[
\hat y_q^{(0)}=F(x_q,S_q^{(0)})
\]

从第一次预测得到：

\[
U_q
\]

其中 \(U_q\) 是 uncertainty map。

选择困难区域：

\[
R_q=\{p:U_q(p)>\tau\}
\]

再从 feature map 提取困难区域表示：

\[
z_q^{hard}=Pool(E(x_q)[R_q])
\]

然后进行第二次 retrieval。

---

# 10. UniverSeg 是确定性模型，怎么得到 Uncertainty？

有三个简单办法。

## 方法 A：Context Disagreement

用不同 support subset 重复推理：

```text
Support set 1 → prediction 1
Support set 2 → prediction 2
Support set 3 → prediction 3
...
Support set M → prediction M
```

得到：

\[
p_1(x),...,p_M(x)
\]

然后：

\[
U(x)=Var[p_1(x),...,p_M(x)]
\]

这个方法非常适合当前课题，因为：

> “不同病例造成预测分歧大的地方”本身就说明当前 context 不稳定。

## 方法 B：TTA

对 query 做：

```text
original
horizontal flip
small rotation
scale
```

然后恢复到原坐标。

计算：

\[
U(x)=Var[p_1(x),...,p_M(x)]
\]

## 方法 C：换 Tyche

第二阶段直接使用：

https://github.com/mariannerakic/Tyche

Tyche 会生成多个 plausible segmentation。

则：

\[
U(x)=Var[\hat y^{(1)}(x),...,\hat y^{(M)}(x)]
\]

这是论文后期很值得加入的版本。

---

# 11. Failure Representation

只用全局 embedding 不够。

我们希望描述：

> 当前模型“哪里不会”。

可以建立：

\[
z_q^{fail}=[z_q^{boundary},z_q^{uncertainty},z_q^{shape},z_q^{global}]
\]

## 11.1 Boundary Feature

从第一次预测：

\[
\hat y_q
\]

获取边界：

\[
B_q=Dilate(\hat y_q)-Erode(\hat y_q)
\]

从 feature map 中提取：

\[
z_q^{boundary}=Pool(F_q[B_q])
\]

## 11.2 Uncertain Region Feature

\[
R_q=U_q>\tau
\]

\[
z_q^{uncertainty}=Pool(F_q[R_q])
\]

## 11.3 Shape Feature

历史病例有 GT：

\[
z_i^{shape}=E_{shape}(y_i)
\]

Query 没有 GT，但第一次预测有：

\[
z_q^{shape}=E_{shape}(\hat y_q^{(0)})
\]

第一版甚至不用神经网络。

直接用 morphology descriptor：

```text
area
perimeter
centroid
eccentricity
compactness
connected components
bounding-box ratio
major/minor axis
```

之后再升级成：

```text
mask
 ↓
signed distance map
 ↓
small CNN
 ↓
shape embedding
```

---

# 12. Candidate Score

一个很自然的第二阶段打分：

\[
Score(q,r)=\alpha S_{global}+\beta S_{boundary}+\gamma S_{shape}+\delta S_{failure}
\]

例如：

\[
S_{global}=cos(z_q^{global},z_r^{global})
\]

\[
S_{shape}=cos(z_q^{shape},z_r^{shape})
\]

最终：

```python
score = (
      alpha * global_sim
    + beta  * boundary_sim
    + gamma * shape_sim
    + delta * failure_sim
)
```

---

# 13. 创新 2：Learn Utility，而不是 Learn Similarity

这是整个方案中很重要的一步。

对于训练 query \(q\) 和 candidate case \(r_i\)，直接运行：

\[
\hat y_{q,i}=F(q,r_i)
\]

得到：

\[
u_{q,i}=Dice(\hat y_{q,i},y_q)
\]

这个 \(u\) 就是：

> candidate 对 query 的真实 downstream utility。

## 13.1 训练 Utility Predictor

学习：

\[
G_\phi(q,r)\rightarrow \hat u(q,r)
\]

目标：

\[
\hat u(q,r)\approx Dice(F(q,r),y_q)
\]

输入可以是：

\[
[z_q,z_r,|z_q-z_r|,z_q\odot z_r,z_q^{fail},z_r^{shape}]
\]

模型第一版直接 MLP：

```text
features
   ↓
Linear
ReLU
Linear
ReLU
Linear
   ↓
predicted utility
```

不要一上来 Transformer。

---

# 14. Ranking Loss 比 Regression 更适合

因为最终不是特别在意预测：

```text
utility = 0.8126
```

我们更关心：

```text
case A 是否比 case B 更值得选
```

因此可以做 pairwise ranking。

如果：

\[
u_A>u_B
\]

则训练：

\[
G(q,A)>G(q,B)
\]

简单 loss：

\[
L_{rank}=-\log \sigma(G(q,A)-G(q,B))
\]

也可以使用 listwise ranking。

---

# 15. 创新 3：Set-Conditioned / Redundancy-Aware Retrieval

这是第二个很推荐作为论文核心贡献的模块。

普通 Top-K：

\[
r_1,r_2,r_3=Top3(Score(q,r))
\]

问题：

```text
r1 ≈ r2 ≈ r3
```

三个病例可能高度重复。

我们真正要优化：

\[
P(segmentation\ improves\mid q,S)
\]

而不是：

\[
P(single\ case\ is\ useful\mid q)
\]

## 15.1 简单版：MMR

选择：

\[
r_t=\arg\max_r[Utility(q,r)-\lambda\max_{s\in S}Sim(r,s)]
\]

第一项：

> 这个病例有多有帮助。

第二项：

> 它和已经选中的病例是否重复。

## 15.2 强版本：Marginal Utility

定义：

\[
\Delta U(r|S)=Dice(F(q,S\cup r),y_q)-Dice(F(q,S),y_q)
\]

训练：

\[
G_\phi(q,S,r)\approx\Delta U(r|S)
\]

推理：

\[
r_1=\arg\max_r G(q,\varnothing,r)
\]

\[
r_2=\arg\max_r G(q,\{r_1\},r)
\]

\[
r_3=\arg\max_r G(q,\{r_1,r_2\},r)
\]

这就从：

```text
Top-K Retrieval
```

升级成：

```text
Sequential Case Selection
```

---

# 16. 三种 Case Role

还可以做一个很医学化的解释。

不要找 3 个相同功能的病例。

人为定义三种角色：

### Appearance Anchor

用于提供：

- modality
- intensity
- global anatomy

### Morphology Anchor

用于提供：

- lesion size
- topology
- boundary
- morphology

### Failure Anchor

专门用于当前：

- fuzzy boundary
- small lesion
- difficult region
- uncertainty area

最后：

\[
S=\{r_A,r_M,r_F\}
\]

这个设计的优点是可解释性非常强。

可以画出三张 reference：

```text
Query

Appearance reference
Morphology reference
Failure reference

Final prediction
```

---

# 17. 创新 4：Adaptive-K

固定：

```text
K = 8
```

通常没有必要。

简单病例可能 `K=1` 就够了，困难病例才需要 `K=4/8`。

可以定义：

\[
K_q=f(uncertainty,retrieval\ confidence,prediction\ consistency)
\]

例如：

```python
if uncertainty < t1:
    k = 1
elif uncertainty < t2:
    k = 2
else:
    k = 4
```

后面再做 learnable stopping：

\[
\hat{\Delta U}_{next}<\epsilon
\]

则停止继续检索。

---

# 18. 推荐的最终框架

```text
                    Historical Case Bank
              ┌──────────────────────────┐
              │ image embedding          │
              │ GT mask                  │
              │ shape embedding          │
              │ boundary feature         │
              │ patient/domain metadata  │
              └────────────┬─────────────┘
                           │
                           ↓
Query ──→ Global Retriever ──→ Top-N Candidates
  │
  │
  └──────────────────────────────┐
                                 ↓
                       Initial Supports
                                 │
                                 ↓
                           UniverSeg
                                 │
                         Initial Mask
                                 │
                    ┌────────────┴────────────┐
                    ↓                         ↓
              Uncertainty                 Shape
                    ↓                         ↓
                Failure Encoder ←────────────┘
                    │
                    ↓
               Utility Reranker
                    │
                    ↓
           Set-Conditioned Selector
                    │
                 r1,r2,r3
                    │
                    ↓
                UniverSeg
                    │
                    ↓
                Final Mask
```

---

# 19. 最终 Score 可以写成

\[
Score(r|q,S)=
\alpha S_{appearance}+
\beta S_{shape}+
\gamma S_{failure}+
\delta \hat U(q,r)-
\lambda Redundancy(r,S)
\]

其中 \(\hat U(q,r)\) 表示预测该病例能给当前 segmentation 带来的收益。

---

# 20. 第一版建议用什么数据？

## 路线 A：皮肤病灶

优点：

- 2D；
- 单目标；
- mask 清晰；
- pipeline 最容易 debug；
- 可以快速验证 retrieval gap。

例如 ISIC 系列。

非常适合作为 Proof of Concept。

## 路线 B：眼底

可以做：

- optic disc
- optic cup
- vessel

优点：

- shape 很重要；
- domain shift 明显；
- 很适合研究 morphology retrieval。

## 路线 C：腹部 CT / MRI

论文故事更强，但第一版不建议马上开始，因为有：

- 3D preprocessing；
- slice correspondence；
- spacing；
- intensity；
- GPU memory；
- patient-level retrieval。

最好等 2D 方法跑通再做。

---

# 21. 一个非常重要的问题：UniverSeg 是否见过数据？

UniverSeg 本身是在大量公开医学数据上训练过的。

因此正式做论文实验时必须确认：

> 测试数据是否出现在 UniverSeg 的训练集合中。

否则不能轻易声称 `unseen task`。

第一阶段工程验证无所谓。

但正式实验应该：

1. 查 UniverSeg MegaMedical 数据列表；
2. 优先采用它论文里的 held-out tasks；
3. 或者选它没有使用的新数据；
4. 所有 comparison 使用完全一样的 segmentation backbone。

这样创新效果才能明确归因于 retrieval。

---

# 22. 最推荐的实验表

## Main Result

| Method | K | Dice ↑ | HD95 ↓ | NSD ↑ |
|---|---:|---:|---:|---:|
| Random Context | 1 | | | |
| Random Context | 4 | | | |
| Appearance KNN | 1 | | | |
| Appearance KNN | 4 | | | |
| Appearance + Shape | 4 | | | |
| Failure-Aware | 4 | | | |
| Failure + Set-aware | 4 | | | |
| Adaptive-K | dynamic | | | |
| Oracle Retrieval | 1 | upper bound | | |

---

# 23. 必做 Ablation

### A. Retrieval Feature

```text
Image
Image + Shape
Image + Boundary
Image + Failure
Image + Shape + Failure
```

### B. K

```text
K = 1
K = 2
K = 4
K = 8
K = 16
```

### C. Selection Strategy

```text
Random
KNN
Utility Top-K
MMR
Set-conditioned
```

### D. Uncertainty

```text
No uncertainty
TTA uncertainty
Context disagreement
Tyche uncertainty
```

### E. Two-stage Retrieval

```text
One-pass retrieval
Two-pass failure-aware retrieval
```

---

# 24. 除 Dice 外，强烈建议分析这些指标

## Retrieval Utility

\[
Utility@K
\]

表示所选病例对下游 segmentation 的实际贡献。

## Oracle Gap

\[
Gap=Dice_{oracle}-Dice_{method}
\]

越小越好。

## Context Diversity

例如：

\[
Diversity(S)=1-\frac{1}{|S|(|S|-1)}\sum_{i\neq j}cos(z_i,z_j)
\]

检查 Set-Aware 方法是否真的降低了 context redundancy。

## Efficiency

报告：

```text
number of support cases
retrieval latency
segmentation latency
GPU memory
```

一个非常漂亮的实验目标是：

> 我们选出的 3 个病例，效果优于 KNN 的 8 / 16 个病例。

---

# 25. 推荐实施顺序

不要一次实现所有创新。

## Stage 0：跑通 UniverSeg

目标：

```text
query + manually chosen support
      ↓
prediction
```

## Stage 1：Random Baseline

实现随机 K support，得到第一个 Dice。

## Stage 2：Embedding KNN

实现：

```text
DINO / ResNet
    ↓
case embeddings
    ↓
cosine Top-K
```

这就是最重要的 baseline。

## Stage 3：Oracle Retrieval

计算：

```text
50 candidates
    ↓
逐个运行 UniverSeg
    ↓
best support
```

得到 retrieval upper bound。

**如果 Oracle Gap 不明显，先不要继续。**

## Stage 4：Utility Ranker

生成：

```text
(query, candidate, downstream Dice)
```

训练数据。

训练 small MLP / ranking network 预测 candidate utility。

## Stage 5：Failure-Aware Retrieval

加入：

```text
initial prediction
uncertainty
boundary
hard region embedding
```

做第二次 retrieval。

## Stage 6：Set-Aware Selection

先用 MMR 验证 diversity 是否有效，再做 learned marginal utility。

## Stage 7：Adaptive-K

最后做。它很适合作为 additional contribution，但不应该抢主创新。

---

# 26. 第一篇版本尽量保持简单

推荐第一篇模型只保留三个核心贡献：

### Contribution 1：Utility-aware case retrieval

不再寻找视觉最相似病例，而是学习：

\[
P(segmentation\ improvement\mid q,r)
\]

### Contribution 2：Failure-aware re-retrieval

利用第一次预测的不确定区域重新寻找能够纠正当前失败模式的病例。

### Contribution 3：Set-conditioned complementary selection

选择相互补充的 context，而不是独立 Top-K。

这样论文故事非常统一：

```text
Similar
    ↓
Useful
    ↓
Corrective
    ↓
Complementary
```

---

# 27. 第一版最小可行系统（MVP）

如果现在马上开始写代码，第一阶段只做：

```text
1. Clone UniverSeg
2. 准备一个 2D segmentation dataset
3. patient-level split
4. 建 case bank
5. Random support
6. DINO / ResNet Top-K support
7. Oracle Top-1 support
8. 画 Retrieval Gap
```

只要出现：

```text
Oracle >> Similarity Top-K
```

这个课题就值得继续。

---

# 28. 关键伪代码

```python
# --------------------------------------------------
# Build Case Bank
# --------------------------------------------------

for case in train_cases:
    img = load_image(case.image)
    mask = load_mask(case.mask)

    z_img = image_encoder(img)
    z_shape = shape_descriptor(mask)

    bank.add(
        case_id=case.id,
        patient_id=case.patient_id,
        image=img,
        mask=mask,
        image_feature=z_img,
        shape_feature=z_shape,
    )


# --------------------------------------------------
# Query
# --------------------------------------------------

query = load_query(query_id)

zq = image_encoder(query)

candidates = bank.top_n(
    feature=zq,
    n=50,
    exclude_patient=query.patient_id,
)


# --------------------------------------------------
# First retrieval
# --------------------------------------------------

initial_supports = candidates[:4]

pred0 = segmenter(
    query,
    support_images(initial_supports),
    support_masks(initial_supports),
)


# --------------------------------------------------
# Estimate Failure
# --------------------------------------------------

uncertainty = estimate_uncertainty(
    query=query,
    candidates=candidates,
    segmenter=segmenter,
)

failure_feature = failure_encoder(
    query,
    pred0,
    uncertainty,
)


# --------------------------------------------------
# Re-rank
# --------------------------------------------------

for candidate in candidates:

    candidate.score = utility_ranker(
        query_feature=zq,
        failure_feature=failure_feature,
        candidate_image_feature=candidate.image_feature,
        candidate_shape_feature=candidate.shape_feature,
    )


# --------------------------------------------------
# Complementary Selection
# --------------------------------------------------

selected = []

while len(selected) < K:

    best = None
    best_score = -1e9

    for candidate in candidates:

        if candidate in selected:
            continue

        redundancy = max_similarity(
            candidate,
            selected,
        )

        score = candidate.score - lambda_red * redundancy

        if score > best_score:
            best = candidate
            best_score = score

    selected.append(best)


# --------------------------------------------------
# Final Segmentation
# --------------------------------------------------

pred = segmenter(
    query,
    support_images(selected),
    support_masks(selected),
)
```

---

# 29. 建议的 Git Commit 顺序

```text
commit 1
universeg inference works

commit 2
dataset + patient split

commit 3
random context baseline

commit 4
feature cache + case bank

commit 5
knn case retrieval

commit 6
oracle retrieval evaluation

commit 7
retrieval-gap analysis

commit 8
utility dataset generation

commit 9
utility ranker

commit 10
failure-aware reranking

commit 11
set-aware selection

commit 12
adaptive-k experiments
```

这样后面做消融很方便。

---

# 30. 参考工作

## UniverSeg — ICCV 2023

Paper:
https://openaccess.thecvf.com/content/ICCV2023/html/Butoi_UniverSeg_Universal_Medical_Image_Segmentation_ICCV_2023_paper.html

Code:
https://github.com/JJGO/UniverSeg

作用：

> 首选 segmentation backbone / ICL baseline。

## Tyche — CVPR 2024

Code:
https://github.com/mariannerakic/Tyche

作用：

> stochastic in-context segmentation；后期可以直接用来构建 uncertainty map。

## Iris — CVPR 2025

Paper:
https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Show_and_Segment_Universal_Medical_Image_Segmentation_via_In-Context_Learning_CVPR_2025_paper.html

作用：

> 重点参考 object-level context retrieval、context encoding、in-context inference。

建议：

> 第一版工程先从官方 UniverSeg 开始，代码链更短，更便于控制变量。

## ICL-SAM — MIDL 2024

Code:
https://github.com/jiesihu/ICL-SAM

作用：

> 可以参考 UniverSeg 与 SAM 组合的工程方式。

---

# 31. 最终研究路线总结

```text
                   Existing
                      │
                      ↓
             Similarity Retrieval
                      │
                      ↓
                    Top-K
                      │
                      ↓
                Segmentation


                   Ours
                    │
                    ↓
            Coarse Similarity Search
                    │
                    ↓
             Initial Segmentation
                    │
                    ↓
           Failure / Uncertainty
                    │
                    ↓
             Utility Prediction
                    │
                    ↓
       Complementary Case Selection
                    │
                    ↓
              Final Segmentation
```

核心观点：

\[
\boxed{
The\ most\ similar\ case\ is\ not\ necessarily\ the\ most\ useful\ case.
}
\]

进一步：

\[
\boxed{
The\ best\ context\ set\ should\ contain\ cases\ that\ correct\ different\ failure\ modes.
}
\]

这是整个项目最值得坚持的研究主线。
