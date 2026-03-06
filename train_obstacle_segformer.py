import os
import numpy as np
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
from transformers import SegformerForSemanticSegmentation

# --- TensorBoard (bezpieczny import) ---
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# ============================================
# 0. USTAWIENIA ŚCIEŻEK (Twoje)
# ============================================

IMAGES_DIR = r"C:\Users\majmo\OneDrive\Desktop\Praktyki\images\Train"
MASKS_DIR  = r"C:\Users\majmo\OneDrive\Desktop\Praktyki\annotations\out_mask"

OUT_ROOT   = r"C:\Users\majmo\OneDrive\Desktop\Praktyki"
MODEL_OUT  = os.path.join(OUT_ROOT, "segformer_obstacle.pth")

# predykcje osobno:
PRED_TRAIN_DIR = os.path.join(OUT_ROOT, "predictions_train")
PRED_VAL_DIR   = os.path.join(OUT_ROOT, "predictions_val")

TB_ROOT    = os.path.join(OUT_ROOT, "tb_logs")  # tu zapisze metryki


# ============================================
# 1. Dataset – obrazy + maski z overlayu
# ============================================

class ObstacleDatasetFromOverlay(Dataset):
    """
    Obrazy: IMAGES_DIR
    Overlay-mask: MASKS_DIR, nazwa: {base}_obstacle_overlay.png
    Maska binarna: różnica (overlay vs oryginał).
    """

    def __init__(self, images_dir, masks_dir, image_size=(512, 512), augment=True, diff_thr=20):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.augment = augment
        self.diff_thr = diff_thr

        self.samples = []
        img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

        for dirpath, _, filenames in os.walk(self.images_dir):
            for fname in filenames:
                if not fname.lower().endswith(img_exts):
                    continue
                img_path = os.path.join(dirpath, fname)
                base = os.path.splitext(os.path.basename(fname))[0]
                overlay_path = os.path.join(self.masks_dir, f"{base}_obstacle_overlay.png")
                if os.path.exists(overlay_path):
                    self.samples.append((img_path, overlay_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                "Nie znaleziono par obraz–overlay.\n"
                f"Obrazy: {self.images_dir}\n"
                f"Overlay: {self.masks_dir}\n"
                "Sprawdź nazwę: {base}_obstacle_overlay.png"
            )

        if self.augment:
            self.img_transform = T.Compose([
                T.Resize(image_size),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.img_transform = T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.rgb_resize = T.Resize(image_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, overlay_path = self.samples[idx]

        img_pil = Image.open(img_path).convert("RGB")
        ov_pil  = Image.open(overlay_path).convert("RGB")

        img_r = self.rgb_resize(img_pil)
        ov_r  = self.rgb_resize(ov_pil)

        img_np = np.array(img_r, dtype=np.int16)
        ov_np  = np.array(ov_r, dtype=np.int16)

        diff = np.abs(ov_np - img_np).sum(axis=2)  # 0..765
        mask01 = (diff >= self.diff_thr).astype(np.int64)

        if self.augment and np.random.rand() < 0.5:
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            mask01 = np.fliplr(mask01).copy()

        image = self.img_transform(img_pil)
        mask_tensor = torch.from_numpy(mask01)

        return image, mask_tensor


# ============================================
# 2. Dataloadery
# ============================================

def get_dataloaders(images_dir, masks_dir, image_size=(512, 512), batch_size=4, train_split=0.8, num_workers=0):
    dataset = ObstacleDatasetFromOverlay(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=image_size,
        augment=True,
        diff_thr=20,
    )

    n_total = len(dataset)
    n_train = int(train_split * n_total)
    n_val   = n_total - n_train

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    # walidacja bez augmentacji
    val_ds.dataset.augment = False
    val_ds.dataset.img_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, train_ds, val_ds


# ============================================
# 3. Model – SegFormer
# ============================================

def get_model(num_classes=2):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model


# ============================================
# 4. Trening + TensorBoard
# ============================================

def train(
    images_dir,
    masks_dir,
    num_epochs=180,
    lr=4.5e-3,
    batch_size=4,
    image_size=(512, 512),
    train_split=0.8,
    num_workers=0,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, train_ds, val_ds = get_dataloaders(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=image_size,
        batch_size=batch_size,
        train_split=train_split,
        num_workers=num_workers,
    )

    model = get_model(num_classes=2).to(device)

    class_weights = torch.tensor([0.4, 0.6], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=lr * 0.01,
    )

    # --- TensorBoard init ---
    writer = None
    run_dir = None
    if SummaryWriter is not None:
        os.makedirs(TB_ROOT, exist_ok=True)
        run_dir = os.path.join(TB_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
        writer = SummaryWriter(log_dir=run_dir)
        print("TensorBoard logs:", run_dir)
    else:
        print("TensorBoard: brak pakietu 'tensorboard' -> metryki NIE będą logowane.")

    best_val_iou = -1.0
    eps = 1e-7

    for epoch in range(num_epochs):
        # ---------- TRAIN ----------
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            logits = model(pixel_values=images).logits

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # ---------- VAL ----------
        model.eval()
        val_loss = 0.0
        total_inter = total_union = total_pred = total_targ = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks  = masks.to(device)

                logits = model(pixel_values=images).logits
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

                loss = criterion(logits, masks)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)

                inter = torch.sum((preds == 1) & (masks == 1)).item()
                union = torch.sum((preds == 1) | (masks == 1)).item()
                psum  = torch.sum(preds == 1).item()
                tsum  = torch.sum(masks == 1).item()

                total_inter += inter
                total_union += union
                total_pred  += psum
                total_targ  += tsum

        val_loss /= max(1, len(val_loader))
        val_iou  = float(total_inter / (total_union + eps))
        val_dice = float((2.0 * total_inter) / (total_pred + total_targ + eps))

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{num_epochs} | lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_IoU={val_iou:.4f} | val_Dice={val_dice:.4f}"
        )

        # --- TensorBoard scalars ---
        if writer is not None:
            writer.add_scalar("lr", current_lr, epoch)
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/iou", val_iou, epoch)
            writer.add_scalar("val/dice", val_dice, epoch)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  Saved best model (val_IoU={best_val_iou:.4f}) -> {MODEL_OUT}")

    if writer is not None:
        writer.flush()
        writer.close()

    return model, train_ds, val_ds


# ============================================
# 5. Inferencja – overlay
# ============================================

def save_orange_overlay(image_pil: Image.Image, pred01: np.ndarray, out_path: str, alpha: int = 120):
    base = image_pil.convert("RGBA")
    h, w = pred01.shape

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., 0] = 255
    overlay[..., 1] = 165
    overlay[..., 2] = 0
    overlay[..., 3] = (pred01.astype(np.uint8) * alpha)

    overlay_img = Image.fromarray(overlay, mode="RGBA")
    out = Image.alpha_composite(base, overlay_img)
    out.save(out_path)


@torch.no_grad()
def predict_from_split(
    split_ds,
    model,
    out_dir,
    image_size=(512, 512),
    device=None,
    obstacle_threshold: float = 0.6,
    max_images=None,
):
    """
    split_ds: Subset zwrócony przez random_split (train_ds albo val_ds)
    out_dir: folder docelowy
    Zapisuje: maskę pred + overlay na oryginale.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    model.eval()

    img_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_dataset = split_ds.dataset
    indices = split_ds.indices

    if max_images is not None:
        indices = indices[:max_images]

    for idx in indices:
        img_path, _overlay_path = base_dataset.samples[idx]

        image_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image_pil.size
        orig_size = (orig_h, orig_w)

        x = img_transform(image_pil).unsqueeze(0).to(device)
        logits = model(pixel_values=x).logits
        logits = F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)

        probs = torch.softmax(logits, dim=1)
        p_obstacle = probs[0, 1]
        pred = (p_obstacle > obstacle_threshold).cpu().numpy().astype(np.uint8)

        base = os.path.splitext(os.path.basename(img_path))[0]

        mask_path = os.path.join(out_dir, base + "_pred.png")
        Image.fromarray(pred * 255).save(mask_path)

        overlay_path = os.path.join(out_dir, base + "_overlay.png")
        save_orange_overlay(image_pil, pred, overlay_path, alpha=120)

    print(f"Saved split predictions to: {out_dir}")


# ============================================
# 6. MAIN
# ============================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, train_ds, val_ds = train(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        num_epochs=180,
        lr=2.5e-3,   # zostawiam jak podałeś w MAIN
        batch_size=4,
        image_size=(512, 512),
        train_split=0.6,
        num_workers=0,
        device=device,
    )

    # Predykcje tylko z treningu (split)
    predict_from_split(
        split_ds=train_ds,
        model=model,
        out_dir=PRED_TRAIN_DIR,
        image_size=(512, 512),
        device=device,
        obstacle_threshold=0.5,
        max_images=None,  # np. 20 jeśli chcesz mniej
    )

    # Predykcje tylko z walidacji (split)
    predict_from_split(
        split_ds=val_ds,
        model=model,
        out_dir=PRED_VAL_DIR,
        image_size=(512, 512),
        device=device,
        obstacle_threshold=0.5,
        max_images=None,
    )
