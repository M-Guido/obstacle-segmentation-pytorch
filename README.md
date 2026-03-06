# Obstacle Segmentation in Orchard Images (PyTorch)

This repository contains a Python script for training and evaluating a deep learning model for **binary semantic segmentation** of obstacles in RGB images (e.g. tree trunks, wires) using **PyTorch**.

> ⚠️ **Dataset is not included.**  
> The project is designed to work with a private dataset annotated in a CVAT-like format (RGB images + overlay masks).  
> You need to provide your own data with a compatible structure.

---

## Features

- Training pipeline for **binary segmentation** (background vs obstacle).
- Custom `Dataset` for images + overlay masks (obstacle drawn on top of the original image).
- On-the-fly preprocessing:
  - image loading, resizing, normalization,
  - light data augmentation (brightness / contrast / saturation changes, horizontal flip).
- Support for **train/validation split**.
- Training loop with:
  - **weighted cross-entropy loss** (handles class imbalance),
  - **AdamW** optimizer with cosine learning rate scheduler,
  - validation metrics: **IoU** and **Dice** for the obstacle class.
- Inference:
  - generation of binary obstacle masks,
  - semi-transparent orange overlays for visual inspection.

---

## Repository structure

Suggested structure:

''obstacle-segmentation-pytorch/
├─ obstacle_segmentation.py   # main training + inference script
├─ README.md
└─ (your private data – NOT in this repo)
   ├─ images/
   │   └─ Train/              # RGB images
   └─ annotations/
       └─ out_mask/           # overlays: {base}_obstacle_overlay.png''

##Train the model

##Run:

-python obstacle_segmentation.py

-The script will:

-create a train/validation split (e.g. 80/20),

-train the model for a configurable number of epochs,

-print training and validation loss as well as IoU/Dice for the obstacle class,

-save the best model (highest validation IoU) to
segformer_obstacle.pth inside OUT_ROOT.

##Configurable training parameters

You can adjust key parameters in the train(...) function, such as:

-num_epochs – total number of training epochs,

-lr – learning rate,

-batch_size – mini-batch size,

-image_size – target resolution for training (e.g. (512, 512)),

-class weights in the loss function – to balance background vs obstacle.

##Changing these allows you to trade off training time vs quality and tune the model for your specific dataset.
