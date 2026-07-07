# EDMPrompt: Efficient Dynamic Memory Prompt for Anomaly Detection

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

## 📖 Introduction

**EDMPrompt** is a CLIP-based visual anomaly detection framework that leverages prompt learning to distinguish between normal and anomalous samples without requiring full model fine-tuning. By learning a small set of learnable prompt tokens and employing a **Multi-Perspective Feature Generation (MPFG)** module and a  **Contrastive Anomaly Boundary(CAB)** module , the model adapts pre-trained vision-language knowledge to the anomaly detection domain efficiently.

### Key Features

- 🎯 **Prompt-driven Anomaly Detection** — Learns dual prompts (normal vs. abnormal) to enable zero-shot-style generalization across diverse industrial and medical datasets
- 🔬 **SignalPrompt** — Generates fine-grained, multi-perspective text features through learnable "signal" tokens and cross-attention via MPFG and refine them by CAB.
- 🧠 **BasePrompt** — Lightweight learnable prompt adaptation on top of frozen CLIP (ViT-L/14@336px) backbone
- 📊 **Comprehensive Evaluation** — Supports image-level, pixel-level, and joint metrics (AUROC, AP, AUPRO) across 14 datasets
- 🏭 **Industrial & Medical Coverage** — Out-of-the-box support for MVTec AD, VisA, BTAD, MPDD, SDD, DTD, and multiple medical imaging benchmarks


## 📦 Supported Datasets

| Category | Datasets |
|----------|----------|
| **Industrial** | MVTec AD, VisA, BTAD, MPDD, SDD, DTD |
| **Medical — Colonoscopy** | Kvasir, CVC-ClinicDB, CVC-ColonDB, Endo |
| **Medical — Radiology** | BrainMRI, HeadCT, BR35 |
| **Medical — Other** | ISBI (skin)|

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA 11.8+ 
- PyTorch 2.6.0

### Installation

```bash
# Clone the repository
git clone https://github.com/lake-user/EDMPrompt.git
cd EDMPrompt

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Training

```bash
python train.py \
    --train_data_path ./dataset/mvtec \
    --dataset mvtec \
    --save_path ./checkpoint \
    --epoch 10 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --image_size 518 \
    --depth 9 \
    --n_ctx 12 \
    --t_n_ctx 4 \
    --seed 111
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_data_path` | `./dataset/mvtec` | Path to training dataset |
| `--dataset` | `mvtec` | Dataset name (see supported datasets) |
| `--save_path` | `./checkpoint` | Directory to save model checkpoints |
| `--epoch` | `10` | Number of training epochs |
| `--batch_size` | `8` | Training batch size |
| `--learning_rate` | `0.001` | Learning rate for Adam optimizer |
| `--image_size` | `518` | Input image resolution |
| `--depth` | `9` | Learnable text embedding depth |
| `--n_ctx` | `12` | Number of learnable prompt context tokens |
| `--t_n_ctx` | `4` | Length of compound text embeddings |
| `--print_freq` | `1` | Logging frequency (epochs) |
| `--save_freq` | `1` | Checkpoint save frequency (epochs) |
| `--seed` | `111` | Random seed for reproducibility |

### Testing (Single Dataset)

```bash
# Image + Pixel combined evaluation
python test.py \
    --data_path ./dataset/mvtec \
    --dataset mvtec \
    --checkpoint_path ./checkpoint/epoch_7.pth \
    --save_path ./results/mvtec \
    --metrics image-pixel-level \
    --image_size 518
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | — | Path to test dataset |
| `--checkpoint_path` | — | Path to trained model checkpoint |
| `--metrics` | `pixel-level` | Evaluation mode: `pixel-level`, `image-level`, `image-pixel-level` |

## 📊 Evaluation Metrics

| Metric | Level | Description |
|--------|-------|-------------|
| **AUROC (Image)** | Image-level | Area under the ROC curve for anomaly classification |
| **AP (Image)** | Image-level | Average precision for anomaly classification |
| **AUROC (Pixel)** | Pixel-level | Area under the ROC curve for pixel-wise anomaly segmentation |
| **AUPRO** | Pixel-level | Area under the Per-Region-Overlap curve (up to 30% FPR) |

## 📊 Experimental Results

> **Legend:** **Bold** = best result  |  *Italic* = second-best result

### Industrial Datasets

#### Industrial · Image-level (AUROC, AP)

| Dataset | CoOp (IJCV'22) | WinCLIP (CVPR'23) | VAND (CVPR'23) | AnomalyCLIP (ICLR'24) | FAPrompt (ICCV'25) | **EDMPrompt** (Ours) |
| --- | --- | --- | --- | --- | --- | --- |
| MVTec AD | (88.8, 94.8) | (91.8, 96.5) | (86.1, 93.5) | **(91.5, 96.2)** | (90.8, 94.9) | *(91.1, 95.5)* |
| VisA | (62.8, 68.1) | (78.1, 81.2) | (78.0, 81.4) | (82.1, 85.4) | **(83.3, 85.9)** | *(83.0, 86.0)* |
| MPDD | (55.1, 64.2) | (63.6, 69.9) | (73.0, 80.2) | (77.0, 82.0) | *(77.8, 81.4)* | **(79.7, 85.3)** |
| BTAD | (66.8, 77.4) | (68.2, 70.9) | (73.6, 68.6) | (88.3, 87.3) | *(91.2, 90.7)* | **(92.7, 91.9)** |
| SDD | (74.9, 65.1) | (84.3, 77.4) | (79.8, 71.4) | *(84.7, 80.0)* | (84.4, 79.3) | **(87.4, 83.7)** |
| DTD | (83.1, 91.9) | (93.2, 92.6) | (86.4, 95.0) | (93.5, 97.0) | *(95.9, 98.4)* | **(96.6, 98.5)** |

#### Industrial · Pixel-level (AUROC, PRO)

| Dataset | CoOp (IJCV'22) | WinCLIP (CVPR'23) | VAND (CVPR'23) | AnomalyCLIP (ICLR'24) | FAPrompt (ICCV'25) | **EDMPrompt** (Ours) |
| --- | --- | --- | --- | --- | --- | --- |
| MVTec AD | (33.3, 6.6) | (85.1, 64.6) | (87.6, 44.0) | **(91.1, 81.4)** | *(90.6, 81.6)* | (89.7, 83.0) |
| VisA | (24.2, 3.8) | (79.6, 56.8) | (94.2, 86.8) | (95.5, 87.0) | (95.6, 85.8) | *(95.5, 88.1)* |
| MPDD | (15.4, 2.3) | (76.4, 48.9) | (94.1, 83.2) | *(96.5, 88.7)* | (96.0, 86.0) | **(96.9, 89.8)** |
| BTAD | (28.7, 3.8) | (72.7, 27.3) | (60.8, 25.0) | (94.2, 74.8) | *(95.6, 73.0)* | **(96.8, 77.9)** |
| SDD | (28.9, 7.1) | (68.8, 24.2) | (79.8, 65.1) | (90.6, 67.8) | *(94.0, 73.1)* | **(94.6, 75.5)** |
| DTD | (55.8, 36.1) | (83.9, 57.8) | (95.3, 86.9) | (97.9, 92.3) | *(98.1, 91.8)* | **(98.5, 91.2)** |

### Medical Datasets

#### Medical · Image-level (AUROC, AP)

| Dataset | CoOp (IJCV'22) | WinCLIP (CVPR'23) | VAND (CVPR'23) | AnomalyCLIP (ICLR'24) | FAPrompt (ICCV'25) | **EDMPrompt** (Ours) |
| --- | --- | --- | --- | --- | --- | --- |
| HeadCT | (78.4, 78.8) | (81.8, 80.2) | (89.1, 89.4) | (93.4, 91.6) | *(93.9, 93.5)* | **(95.4, 95.9)** |
| BrainMRI | (61.3, 44.9) | (86.6, 91.5) | (89.3, 90.9) | (90.3, 92.2) | *(95.0, 95.0)* | **(95.2, 96.3)** |
| Br35H | (86.0, 87.5) | (80.5, 82.2) | (93.1, 92.9) | (94.6, 94.7) | *(97.2, 96.7)* | **(97.8, 97.7)** |

#### Medical · Pixel-level (AUROC, PRO)

| Dataset | CoOp (IJCV'22) | WinCLIP (CVPR'23) | VAND (CVPR'23) | AnomalyCLIP (ICLR'24) | FAPrompt (ICCV'25) | **EDMPrompt** (Ours) |
| --- | --- | --- | --- | --- | --- | --- |
| ISIC | (51.7, 15.9) | (83.3, 55.1) | (89.4, 77.2) | (89.7, 78.4) | *(90.6, 80.3)* | **(92.6, 85.8)** |
| CVC-ColonDB | (40.5, 2.60) | (70.3, 32.5) | (78.4, 64.6) | (81.9, 71.3) | *(83.2, 72.9)* | **(84.4, 73.4)** |
| CVC-ClinicDB | (34.8, 2.40) | (51.2, 13.8) | (80.5, 60.7) | (82.9, 67.8) | *(83.4, 68.6)* | **(85.6, 72.5)** |
| Kvasir | (44.1, 3.50) | (69.7, 24.5) | (75.0, 36.2) | (78.9, 45.6) | *(80.7, 47.6)* | **(83.2, 51.7)** |
| Endo | (40.6, 3.90) | (68.2, 28.3) | (81.9, 54.9) | (84.1, 63.6) | *(85.9, 65.1)* | **(87.3, 69.5)** |

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{edmprompt,
  title={EDMPrompt:Towards Effcient Generation of Discriminative Multi-Prompt Features for Zero-Shot Anomaly Detection},
  author={Dongyang Zhao, Quan Yuan, Rui Pan, Ke Tan, Xiaoyuan Fu, Guiyang Luo, and Jinglin Li},
  journal={},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

This project builds upon the following excellent open-source works:
- [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) — CLIP-based anomaly detection with object-agnostic prompt learning
- [FAPrompt](https://github.com/mala-lab/faprompt) — Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection 
