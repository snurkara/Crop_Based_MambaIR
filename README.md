# Crop_Based_MambaIR: Object-Centered Aerial Image Deblurring via Two-Stage MambaIR Fine-Tuning

A crop-based finetuning pipeline for aerial image deblurring using [MambaIR](https://github.com/csguoh/MambaIR).

The model is finetuned in two stages:
1. **Full-image finetune** — trained on full-image blur/sharp image pairs
2. **Crop-based finetune** — trained on object-level crops extracted via YOLOv8, allowing the model to focus on actual aerial objects rather than background

---

## Repository Structure

```
CROP_BASED_MAMBAIR/
├── configs/
│   ├── train_MambaIR_Deblur_Finetune.yml     ← Stage 1 training config
│   └── train_MambaIR_CropDeblur.yml          ← Stage 2 training config
├── notebooks/
│   ├── 01_data_preparation_full_image.ipynb  ← Resize + blur synthesis
│   ├── 02_training_full_image.ipynb          ← Stage 1 finetuning
│   ├── 03_data_preparation_crop.ipynb        ← YOLO crop + blur synthesis
│   ├── 04_training_crop.ipynb                ← Stage 2 finetuning
│   └── 05_inference.ipynb                    ← Inference pipeline
└── LICENSE
```

---

## Pipeline

```
Input aerial image
       ↓
  YOLOv8 — detect all objects (no custom training needed)
       ↓
  Crop each detected object region (256×256)
       ↓
  MambaIR — deblur each crop
       ↓
  Paste deblurred crops back into original image
       ↓
  Output image
```

---

## Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/snurkara/Crop_Based_MambaIR.git
cd Crop_Based_MambaIR
```

### 2. Clone MambaIR

```bash
git clone https://github.com/csguoh/MambaIR.git
cd MambaIR
```

### 3. Install dependencies

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install causal-conv1d
pip install mamba-ssm
pip install basicsr einops timm ultralytics
```

> Tested on Google Colab A100 GPU with CUDA 12.8 and PyTorch 2.10.0.

### 4. Run notebooks in order

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_data_preparation_full_image.ipynb` | Resize images and apply synthetic blur |
| 2 | `02_training_full_image.ipynb` | Stage 1 finetuning on full images |
| 3 | `03_data_preparation_crop.ipynb` | Extract object crops via YOLOv8 and apply blur |
| 4 | `04_training_crop.ipynb` | Stage 2 finetuning on object crops |
| 5 | `05_inference.ipynb` | Run inference on new images |

---

## Dataset

The Custom Aerial Object Deblurring Dataset consists of 701 aerial images containing flying objects (drones, aircraft, birds etc.) split as follows:

| Split | Images |
|-------|--------|
| Train | 490 |
| Val   | 105 |
| Test  | 106 |

These 701 images were selected from the Det-Fly dataset [2] and the AOD4 dataset [3]. 

---

## Pretrained Weights

Download the original MambaIR pretrained weights (Gaussian Denoising, σ=25) from:

Base pretrained model: ColorDN_MambaIR_level25.pth 
- [HuggingFace — csguoh/MambaIR](https://huggingface.co/cguoh/MambaIR/tree/main/MambaIRv1_ckpt)

The finetuned checkpoints are available in the [v1.0-paper GitHub Release] (https://github.com/snurkara/Crop_Based_MambaIR/releases/tag/v1.0-paper) :

- `net_g_7000.pth` — Stage 1 full-image finetune
- `net_g_10000.pth` — Stage 2 crop-based finetune result

---

## Results

### Stage 1 — Full Image Finetune

| Metric | Best Value | Iteration |
|--------|-----------|-----------|
| PSNR   | 40.16 dB  | 7000      |
| SSIM   | 0.9825    | 7000      |

### Stage 2 — Crop-based Finetune

| Metric | Best Value | Iteration |
|--------|-----------|-----------|
| PSNR   | 37.71 dB  | 10000     |
| SSIM   | 0.9618    | 10000     |

---

## Citation

If you use this work, please also cite the original MambaIR paper:

```bibtex
@inproceedings{guo2025mambair,
  title={MambaIR: A simple baseline for image restoration with state-space model},
  author={Guo, Hang and Li, Jinmin and Dai, Tao and Ouyang, Zhihao and Ren, Xudong and Xia, Shu-Tao},
  booktitle={European Conference on Computer Vision},
  pages={222--241},
  year={2024},
  organization={Springer}
}
```

---

## Acknowledgements

This project builds on [MambaIR](https://github.com/csguoh/MambaIR) by Guo et al.
Object detection is performed using [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics.

## References

[1] H. Guo, J. Li, T. Dai, Z. Ouyang, X. Ren, & S. T. Xia, “Mambair: A simple baseline for image restoration with state-space model,” in *European Conference on Computer Vision* (pp. 222-241), Cham: Springer Nature Switzerland, Sep. 2024. [Online]. Available: https://doi.org/10.1007/978-3-031-72649-1_13

[2] J. Wu, “Det-Fly,” GitHub repository. Accessed: May 24, 2026. [Online]. Available: https://github.com/Jake-WU/Det-Fly

[3] V. Soni, D. Shah, J. Joshi, S. Gite, B. Pradhan, and A. Alamri, “Introducing AOD4: A dataset for air borne object detection,” Data in Brief, vol. 56, Art. no. 110801, 2024, doi: https://doi.org/10.1016/j.dib.2024.110801 [Online]. Available: https://data.mendeley.com/datasets/cd5z895tr2/1
