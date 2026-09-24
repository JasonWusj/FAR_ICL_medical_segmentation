# ISIC 2018 exploratory comparison suite

All launchers read the same manifest (2,594 train / 100 validation / 1,000 test), use image-level identity, and write to **separate folders** under `OUT_BASE`. The source ZIP files and extracted images can remain outside this repository. The common primary metrics are mean Dice, IoU, and ISIC-style thresholded Jaccard (`IoU < 0.65` is scored as zero). This is an exploratory comparison: no patient IDs are available, and the test split has already been examined in earlier runs. It cannot establish patient-independent generalization or a new blind ISIC challenge score. Check the pretrained UniverSeg/Tyche training data for ISIC overlap before claiming an unseen-dataset result.

On Featurize, after `git pull --ff-only` and the existing `bash scripts/setup_linux.sh` setup, run one folder at a time:

```bash
cd /home/featurize/work/FAR_ICL_medical_segmentation
export MANIFEST=/home/featurize/work/isic2018_full_with_test.csv
export OUT_BASE=/home/featurize/work/far_icl_paper_runs
export TORCH_HOME=/home/featurize/work/.cache/torch
export CUDA_VISIBLE_DEVICES=0
bash experiments/same_backbone/universeg_knn/run.sh
bash experiments/same_backbone/dual_similarity_adapted/run.sh
bash experiments/same_backbone/far_icl_repair/run.sh
bash experiments/published_icl/tyche/run.sh
bash experiments/published_icl/ires_s3_adapted/run.sh
bash experiments/supervised/unet/run.sh
bash experiments/supervised/nnunet/run.sh
bash experiments/compare_all.sh
```

The Iris launcher is conditional:

```bash
IRIS_PRED_DIR=/path/to/iris_128px_test_pngs bash experiments/published_icl/iris/run.sh
```

It accepts one `case_id.png` binary mask per test image at 128×128 pixels. We have not found an author-released ISIC-ready Iris implementation and weights, so that folder scores **external predictions only** and does not claim an Iris reproduction. The [Iris paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Show_and_Segment_Universal_Medical_Image_Segmentation_via_In-Context_Learning_CVPR_2025_paper.pdf) is for a 3D method; a 2D ISIC adapter would need separate validation.

## Groups and provenance

| Group | Folder | What is run | Status |
|---|---|---|---|
| Same frozen backbone | `same_backbone/universeg_knn` | Official [UniverSeg](https://github.com/JJGO/UniverSeg) weights + ResNet50 cosine KNN, K=2 | Reproducible baseline under this project's preprocessing |
| Same frozen backbone | `same_backbone/far_icl_repair` | Frozen UniverSeg + trained FAR-ICL repair selector | Proposed method |
| Same frozen backbone | `same_backbone/dual_similarity_adapted` | Semantic cosine + predicted-mask shape cosine | **Adaptation**, not the [Dual Similarity Checkup](https://doi.org/10.1109/TMI.2024.3440311) paper's weighted sampling and augmentation |
| Published ICL | `published_icl/tyche` | Official [Tyche](https://github.com/mariannerakic/Tyche) model with the same ResNet50 KNN supports | Official model, adapted ISIC support protocol |
| Published ICL | `published_icl/ires_s3_adapted` | Per-query ResNet50/DINOv2 retrieval-encoder choice using S³ erosion score | **Adaptation** of [IRES](https://ojs.aaai.org/index.php/AAAI/article/view/39819): K=2, no P2R optimization |
| Published ICL | `published_icl/iris` | Score externally supplied Iris masks | Requires separately obtained author implementation/weights |
| Supervised | `supervised/unet` | Train 2D U-Net from scratch; validation Dice selects checkpoint | Project implementation of [U-Net](https://arxiv.org/abs/1505.04597) |
| Supervised | `supervised/nnunet` | Official [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), 2D fold 0 | Official training code, explicit train/validation split |

The three groups answer different questions. The same-backbone group isolates support selection. Published ICL compares end-to-end systems or published selection ideas. Supervised models measure how much labeled ISIC training can buy; they **do not** have the same training budget as frozen ICL. Report these groups separately and include trainable parameters, pretraining data, support count, training GPU-hours, inference time, and GPU memory in the paper.

`published_icl/tyche/run.sh` requires the official code checkout; on a fresh server run `INSTALL_TYCHE=1 bash scripts/setup_linux.sh` once. `published_icl/ires_s3_adapted/run.sh` downloads DINOv2 weights on first use. `supervised/nnunet/run.sh` installs `nnunetv2==2.8.1` into `.venv` if absent; its default trainer can take much longer than the other experiments and may need lower memory settings on an RTX 2080 Ti. Each method writes `results/*/per_case.json`, `summary.json`, and provenance. The comparison script requires matching manifest hash, identity scope, and case IDs and writes `comparison/test_table.csv` plus paired image-bootstrap intervals.

Set `K`, `SEED`, `CANDIDATE_N`, `OUT_BASE`, or `MANIFEST` in the shell to change a run. For the supervised U-Net, `UNET_EPOCHS`, `UNET_PATIENCE`, and `UNET_BATCH_SIZE` are available. Do not tune these on the official test set; choose them on validation only. For nnU-Net, `NNUNET_DATASET_ID` and `NNUNET_TRAINER` can be overridden. Use a new `OUT_BASE` when changing the protocol so that prior checkpoints and caches are not silently reused.
