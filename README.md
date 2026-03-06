# Obstacle Segmentation in Orchard Images (PyTorch)

This repository contains a Python script for training and evaluating a deep learning model for **binary semantic segmentation** of obstacles in RGB images (e.g. tree trunks, wires) using **PyTorch**.

> ⚠️ **Dataset is not included.**  
> The project is designed to work with a private dataset annotated in a CVAT-like format (RGB images + overlay masks). You need to provide your own data with a compatible structure.

---

## Features

- Training pipeline for **binary segmentation** (background vs obstacle).
- Custom `Dataset` for images + overlay masks (obstacle drawn on top of the original image).
- On-the-fly preprocessing:
  - image loading, resizing, normalization,
  - light data augmentation (brightness/contrast/saturation, horizontal flip),
  - conversion of overlays to binary masks by pixel-wise comparison.
- Support for **train/validation split**.
- Training loop with:
  - weighted cross-entropy loss (to handle class imbalance),
  - optimizer with learning rate scheduler,
  - validation metrics: **IoU** and **Dice** for the obstacle class.
- Inference script that:
  - generates binary masks for new images,
  - saves semi-transparent orange overlays for visual inspection.


