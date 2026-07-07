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

### Industrial Datasets

<p align="center"><em>Comparison of different methods on industrial datasets. Best in <span style="color:#e74c3c"><b>red</b></span>, second best in <span style="color:#3498db"><b>blue</b></span>.</em></p>

<table align="center">
  <thead>
    <tr>
      <th colspan="2" style="border-bottom:2px solid #666; padding:8px 12px"></th>
      <th style="border-bottom:2px solid #666; padding:8px 12px" align="center">CoOp<br><sub>(IJCV'22)</sub></th>
      <th style="border-bottom:2px solid #666; padding:8px 12px" align="center">WinCLIP<br><sub>(CVPR'23)</sub></th>
      <th style="border-bottom:2px solid #666; padding:8px 12px" align="center">VAND<br><sub>(CVPR'23)</sub></th>
      <th style="border-bottom:2px solid #666; padding:8px 12px" align="center">AnomalyCLIP<br><sub>(ICLR'24)</sub></th>
      <th style="border-bottom:2px solid #666; padding:8px 12px" align="center">FAPrompt<br><sub>(ICCV'25)</sub></th>
      <th style="border-bottom:2px solid #666; padding:8px 12px" align="center"><b>EDMPrompt</b><br><sub>(Ours)</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr><td rowspan="6" style="border-right:1px solid #ddd; padding:8px 12px; vertical-align:middle" align="center"><b>Image-level</b><br><sub>(AUROC, AP)</sub></td></tr>
    <tr>
      <td style="padding:6px 10px">MVTec AD</td>
      <td align="center">(88.8, 94.8)</td>
      <td align="center">(91.8, 96.5)</td>
      <td align="center">(86.1, 93.5)</td>
      <td align="center"><span style="color:#e74c3c"><b>(91.5, 96.2)</b></span></td>
      <td align="center">(90.8, 94.9)</td>
      <td align="center"><span style="color:#3498db"><b>(91.1, 95.5)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">VisA</td>
      <td align="center">(62.8, 68.1)</td>
      <td align="center">(78.1, 81.2)</td>
      <td align="center">(78.0, 81.4)</td>
      <td align="center">(82.1, 85.4)</td>
      <td align="center"><span style="color:#e74c3c"><b>(83.3, 85.9)</b></span></td>
      <td align="center"><span style="color:#3498db"><b>(83.0, 86.0)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">MPDD</td>
      <td align="center">(55.1, 64.2)</td>
      <td align="center">(63.6, 69.9)</td>
      <td align="center">(73.0, 80.2)</td>
      <td align="center">(77.0, <span style="color:#3498db"><b>82.0</b></span>)</td>
      <td align="center"><span style="color:#3498db"><b>(77.8, 81.4)</b></span></td>
      <td align="center"><span style="color:#e74c3c"><b>(79.7, 85.3)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">BTAD</td>
      <td align="center">(66.8, 77.4)</td>
      <td align="center">(68.2, 70.9)</td>
      <td align="center">(73.6, 68.6)</td>
      <td align="center">(88.3, 87.3)</td>
      <td align="center"><span style="color:#3498db"><b>(91.2, 90.7)</b></span></td>
      <td align="center"><span style="color:#e74c3c"><b>(92.7, 91.9)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">SDD</td>
      <td align="center">(74.9, 65.1)</td>
      <td align="center">(84.3, 77.4)</td>
      <td align="center">(79.8, 71.4)</td>
      <td align="center"><span style="color:#3498db"><b>(84.7, 80.0)</b></span></td>
      <td align="center">(84.4, 79.3)</td>
      <td align="center"><span style="color:#e74c3c"><b>(87.4, 83.7)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">DTD-Synthetic</td>
      <td align="center">(83.1, 91.9)</td>
      <td align="center">(93.2, 92.6)</td>
      <td align="center">(86.4, 95.0)</td>
      <td align="center">(93.5, 97.0)</td>
      <td align="center"><span style="color:#3498db"><b>(95.9, 98.4)</b></span></td>
      <td align="center"><span style="color:#e74c3c"><b>(96.6, 98.5)</b></span></td>
    </tr>
    <tr style="border-top:1px solid #ddd"><td colspan="8"></td></tr>
    <tr><td rowspan="6" style="border-right:1px solid #ddd; padding:8px 12px; vertical-align:middle" align="center"><b>Pixel-level</b><br><sub>(AUROC, PRO)</sub></td></tr>
    <tr>
      <td style="padding:6px 10px">MVTec AD</td>
      <td align="center">(33.3, 6.6)</td>
      <td align="center">(85.1, 64.6)</td>
      <td align="center">(87.6, 44.0)</td>
      <td align="center"><span style="color:#e74c3c"><b>(91.1, 81.4)</b></span></td>
      <td align="center"><span style="color:#3498db"><b>(90.6, 81.6)</b></span></td>
      <td align="center">(89.7, <span style="color:#e74c3c"><b>83.0</b></span>)</td>
    </tr>
    <tr>
      <td style="padding:6px 10px">VisA</td>
      <td align="center">(24.2, 3.8)</td>
      <td align="center">(79.6, 56.8)</td>
      <td align="center">(94.2, 86.8)</td>
      <td align="center">(95.5, <span style="color:#3498db"><b>87.0</b></span>)</td>
      <td align="center"><span style="color:#e74c3c"><b>(95.6, 85.8)</b></span></td>
      <td align="center"><span style="color:#3498db"><b>(95.5, 88.1)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">MPDD</td>
      <td align="center">(15.4, 2.3)</td>
      <td align="center">(76.4, 48.9)</td>
      <td align="center">(94.1, 83.2)</td>
      <td align="center"><span style="color:#3498db"><b>(96.5, 88.7)</b></span></td>
      <td align="center">(96.0, 86.0)</td>
      <td align="center"><span style="color:#e74c3c"><b>(96.9, 89.8)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">BTAD</td>
      <td align="center">(28.7, 3.8)</td>
      <td align="center">(72.7, 27.3)</td>
      <td align="center">(60.8, 25.0)</td>
      <td align="center">(94.2, <span style="color:#3498db"><b>74.8</b></span>)</td>
      <td align="center"><span style="color:#3498db"><b>(95.6, 73.0)</b></span></td>
      <td align="center"><span style="color:#e74c3c"><b>(96.8, 77.9)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">SDD</td>
      <td align="center">(28.9, 7.1)</td>
      <td align="center">(68.8, 24.2)</td>
      <td align="center">(79.8, 65.1)</td>
      <td align="center">(90.6, 67.8)</td>
      <td align="center"><span style="color:#3498db"><b>(94.0, 73.1)</b></span></td>
      <td align="center"><span style="color:#e74c3c"><b>(94.6, 75.5)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">DTD-Synthetic</td>
      <td align="center">(55.8, 36.1)</td>
      <td align="center">(83.9, 57.8)</td>
      <td align="center">(95.3, 86.9)</td>
      <td align="center">(97.9, <span style="color:#e74c3c"><b>92.3</b></span>)</td>
      <td align="center"><span style="color:#3498db"><b>(98.1, 91.8)</b></span></td>
      <td align="center"><span style="color:#e74c3c"><b>(98.5, 91.2)</b></span></td>
    </tr>
  </tbody>
</table>

### Medical Datasets

<p align="center"><em>Comparison of different methods on medical datasets. Best in <span style="color:#e74c3c"><b>red</b></span>, second best in <span style="color:#3498db"><b>blue</b></span>.</em></p>

<table align="center">
  <thead>
    <tr>
      <th colspan="2" style="border-bottom:2px solid #666; padding:8px 12px"></th>
      <th style="border-bottom:2px solid #666; padding:8px 12px" align="center">CoOp<br><sub>(IJCV'22)</sub></th>
      <th style="border-bottom:2px solid #666; padding:8px 12px" align="center">WinCLIP<br><sub>(CVPR'23)</sub></th>
      <th style="border-bottom:2px solid #666; padding:8px 12px" align="center">VAND<br><sub>(CVPR'23)</sub></th>
      <th style="border-bottom:2px solid #666; padding:8px 12px" align="center">AnomalyCLIP<br><sub>(ICLR'24)</sub></th>
      <th style="border-bottom:2px solid #666; padding:8px 12px" align="center">FAPrompt<br><sub>(ICCV'25)</sub></th>
      <th style="border-bottom:2px solid #666; padding:8px 12px" align="center"><b>EDMPrompt</b><br><sub>(Ours)</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr><td rowspan="3" style="border-right:1px solid #ddd; padding:8px 12px; vertical-align:middle" align="center"><b>Image-level</b><br><sub>(AUROC, AP)</sub></td></tr>
    <tr>
      <td style="padding:6px 10px">HeadCT</td>
      <td align="center">(78.4, 78.8)</td>
      <td align="center">(81.8, 80.2)</td>
      <td align="center">(89.1, 89.4)</td>
      <td align="center">(93.4, 91.6)</td>
      <td align="center"><span style="color:#3498db"><b>(93.9, 93.5)</b></span></td>
      <td align="center"><span style="color:#e74c3c"><b>(95.4, 95.9)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">BrainMRI</td>
      <td align="center">(61.3, 44.9)</td>
      <td align="center">(86.6, 91.5)</td>
      <td align="center">(89.3, 90.9)</td>
      <td align="center">(90.3, 92.2)</td>
      <td align="center"><span style="color:#3498db"><b>(95.0, 95.0)</b></span></td>
      <td align="center"><span style="color:#e74c3c"><b>(95.2, 96.3)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">Br35H</td>
      <td align="center">(86.0, 87.5)</td>
      <td align="center">(80.5, 82.2)</td>
      <td align="center">(93.1, 92.9)</td>
      <td align="center">(94.6, 94.7)</td>
      <td align="center"><span style="color:#3498db"><b>(97.2, 96.7)</b></span></td>
      <td align="center"><span style="color:#e74c3c"><b>(97.8, 97.7)</b></span></td>
    </tr>
    <tr style="border-top:1px solid #ddd"><td colspan="8"></td></tr>
    <tr><td rowspan="5" style="border-right:1px solid #ddd; padding:8px 12px; vertical-align:middle" align="center"><b>Pixel-level</b><br><sub>(AUROC, PRO)</sub></td></tr>
    <tr>
      <td style="padding:6px 10px">ISIC</td>
      <td align="center">(51.7, 15.9)</td>
      <td align="center">(83.3, 55.1)</td>
      <td align="center">(89.4, 77.2)</td>
      <td align="center">(89.7, 78.4)</td>
      <td align="center"><span style="color:#3498db"><b>(90.6, 80.3)</b></span></td>
      <td align="center"><span style="color:#e74c3c"><b>(92.6, 85.8)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">CVC-ColonDB</td>
      <td align="center">(40.5, 2.60)</td>
      <td align="center">(70.3, 32.5)</td>
      <td align="center">(78.4, 64.6)</td>
      <td align="center">(81.9, 71.3)</td>
      <td align="center"><span style="color:#3498db"><b>(83.2, 72.9)</b></span></td>
      <td align="center"><span style="color:#e74c3c"><b>(84.4, 73.4)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">CVC-ClinicDB</td>
      <td align="center">(34.8, 2.40)</td>
      <td align="center">(51.2, 13.8)</td>
      <td align="center">(80.5, 60.7)</td>
      <td align="center">(82.9, 67.8)</td>
      <td align="center"><span style="color:#3498db"><b>(83.4, 68.6)</b></span></td>
      <td align="center"><span style="color:#e74c3c"><b>(85.6, 72.5)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">Kvasir</td>
      <td align="center">(44.1, 3.50)</td>
      <td align="center">(69.7, 24.5)</td>
      <td align="center">(75.0, 36.2)</td>
      <td align="center">(78.9, 45.6)</td>
      <td align="center"><span style="color:#3498db"><b>(80.7, 47.6)</b></span></td>
      <td align="center"><span style="color:#e74c3c"><b>(83.2, 51.7)</b></span></td>
    </tr>
    <tr>
      <td style="padding:6px 10px">Endo</td>
      <td align="center">(40.6, 3.90)</td>
      <td align="center">(68.2, 28.3)</td>
      <td align="center">(81.9, 54.9)</td>
      <td align="center">(84.1, 63.6)</td>
      <td align="center"><span style="color:#3498db"><b>(85.9, 65.1)</b></span></td>
      <td align="center"><span style="color:#e74c3c"><b>(87.3, 69.5)</b></span></td>
    </tr>
  </tbody>
</table>

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{edmprompt,
  title={EDMPrompt: Efficient Dynamic Memory Prompt for Anomaly Detection},
  author={},
  journal={},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

This project builds upon the following excellent open-source works:
- [OpenAI CLIP](https://github.com/openai/CLIP) — Vision-language pre-training
- [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) — CLIP-based anomaly detection with object-agnostic prompt learning
- [CoOp / CoCoOp](https://github.com/KaiyangZhou/CoOp) — Context Optimization for vision-language models
