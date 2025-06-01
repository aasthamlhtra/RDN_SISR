# RDN_SISR
This repository contains the implementation and experimentation of a Residual Dense Network (RDN) for Single Image Super-Resolution (SISR). Our work is based on the architecture proposed by Y. Zhang et al., with architectural modifications and performance evaluation using the DIV2K and standard benchmark datasets.

Project Highlights
Implemented RDN in PyTorch with modular components

Extended original RDN by increasing the number of residual dense blocks

Achieved PSNR of 29.144 dB with 16 RDBs and 29.264 dB with 32 RDBs on DIV2K validation

Evaluated performance on Set5, Set14, B100, Urban100 datasets

Conducted depth-performance tradeoff analysis


Single Image Super-Resolution (SISR) is the task of reconstructing a high-resolution image from a low-resolution input. RDN leverages:

Local dense connectivity (within RDBs)

Local residual learning

Global feature fusion

Sub-pixel convolution for upscaling

Compared to previous models (SRCNN, VDSR, SRDenseNet), RDN provides superior performance while avoiding the computational overhead of pre-upscaling.
