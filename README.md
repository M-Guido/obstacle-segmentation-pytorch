# 🌳 Obstacle Segmentation in Orchard Images (PyTorch)

Deep learning pipeline for **binary semantic segmentation** of obstacles (e.g. tree trunks, wires, poles) in RGB orchard images using **PyTorch**.

The project is built for structured agricultural environments where reliable obstacle detection is critical for autonomous navigation.

> ⚠️ **Dataset is NOT included**
> The training script expects a private dataset annotated in a CVAT-like format (RGB images + overlay masks).
> You must provide your own data following the expected structure.

---

## 🧠 What This Project Does

* Trains a **binary segmentation model** (background vs obstacle)
* Handles **class imbalance** using weighted cross-entropy
* Computes **IoU and Dice** metrics for the obstacle class
* Supports train/validation split
* Generates visual **semi-transparent overlays** for inspection
* Saves the best-performing model automatically

---

# 📁 Repository Structure

```
obstacle-segmentation-pytorch/
├─ obstacle_segmentation.py   # training + inference script
├─ README.md
└─ (your private dataset - not included)
   ├─ images/
   │   └─ Train/              # RGB input images
   └─ annotations/
       └─ out_mask/           # overlay masks: {base}_obstacle_overlay.png
```

### Expected Dataset Format

Each image should have a corresponding overlay mask:

```
images/Train/img_001.png
annotations/out_mask/img_001_obstacle_overlay.png
```

Overlay masks must:

* Match image resolution
* Contain obstacle regions clearly marked
* Be convertible to binary segmentation masks

---

# 🚀 Training

Run:

```bash
python obstacle_segmentation.py
```

The script will automatically:

1. Create an **80/20 train-validation split**
2. Train for the configured number of epochs
3. Print:

   * Training loss
   * Validation loss
   * IoU (Intersection over Union)
   * Dice score (F1)
4. Save the best model (highest validation IoU) to:

```
OUT_ROOT/segformer_obstacle.pth
```

---

# ⚙️ Training Configuration

You can modify parameters inside the `train(...)` function:

| Parameter       | Description                           |
| --------------- | ------------------------------------- |
| `num_epochs`    | Number of training epochs             |
| `lr`            | Learning rate                         |
| `batch_size`    | Mini-batch size                       |
| `image_size`    | Target resolution (e.g. `(512, 512)`) |
| `class_weights` | Loss weighting for imbalance          |

### Trade-offs

* Larger `image_size` → better detail, higher GPU memory usage
* Higher `batch_size` → faster training, more memory required
* Stronger class weighting → better small obstacle detection

---

# 🏗 Training Pipeline Details

### Preprocessing

* Image loading
* Resize
* Normalization
* Light augmentation:

  * brightness
  * contrast
  * saturation
  * horizontal flip

### Loss Function

* **Weighted Cross-Entropy**

  * Mitigates strong background dominance

### Optimizer

* **AdamW**
* Cosine learning rate scheduler

### Metrics (Validation)

* **IoU** (Intersection over Union)
* **Dice coefficient**

Both computed for the **obstacle class only**.

---

# 🔎 Inference

After training:

* The best model weights are saved
* Binary masks are generated
* Semi-transparent overlays are produced for visual inspection

This enables quick qualitative evaluation in orchard environments.

---

# 🧪 Tips for Best Results

* Ensure masks are clean and consistent
* Avoid annotation noise around thin structures (e.g. wires)
* Consider:

  * Increasing image resolution for thin obstacles
  * Stronger class weighting for rare obstacles
  * More augmentation for better generalization

---
