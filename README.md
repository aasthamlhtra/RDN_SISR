# Residual Dense Network for Image Super-Resolution

This repository contains the implementation and experimentation of a **Residual Dense Network (RDN)** for **Single Image Super-Resolution (SISR)**. Our work is based on the architecture proposed by Y. Zhang et al., with architectural modifications and performance evaluation using the DIV2K and standard benchmark datasets.

---

## Project Highlights

-  Implemented RDN in **PyTorch** with modular components  
-  Extended original RDN by increasing the number of residual dense blocks  
-  Achieved **PSNR of 29.144 dB** with 16 RDBs and **29.264 dB** with 32 RDBs on DIV2K validation  
-  Evaluated performance on Set5, Set14, B100, Urban100 datasets  
-  Conducted depth-performance tradeoff analysis  

---

##  Background

**Single Image Super-Resolution (SISR)** is the task of reconstructing a high-resolution image from a low-resolution input. RDN leverages:

- Local dense connectivity (within RDBs)  
- Local residual learning  
- Global feature fusion  
- Sub-pixel convolution for upscaling  

Compared to previous models (SRCNN, VDSR, SRDenseNet), RDN provides superior performance while avoiding the computational overhead of pre-upscaling.

---

## 🛠 Methodology

- **Architecture**:
  - 2-layer shallow feature extractor (SFENet)
  - Stack of Residual Dense Blocks (RDBs)
  - Global feature fusion layer (GFF)
  - Upsampling module using PixelShuffle (UPNet)
- **Loss Function**: L1 Loss
- **Optimizer**: Adam
- **Scale Factor**: ×4 Super-Resolution
- **Datasets**:
  - Training: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
  - Testing: Set5, Set14, B100, Urban100

---

##  Results

### PSNR on Benchmark Datasets

| Dataset   | PSNR (×4)   |
|-----------|-------------|
| Set5      | 31.565 dB   |
| Set14     | 28.230 dB   |
| B100      | 27.304 dB   |
| Urban100  | 25.265 dB   |

### Model Comparison

| Model Config | RDBs | PSNR (DIV2K) | Epoch | Relative Training Time |
|--------------|------|--------------|-------|-------------------------|
| Baseline     | 16   | 29.144 dB    | 18    | 1×                      |
| Deepened     | 32   | 29.264 dB    | 19    | ~3×                     |

>  Increasing model depth provided only marginal PSNR gains with significant increase in training time.

---




