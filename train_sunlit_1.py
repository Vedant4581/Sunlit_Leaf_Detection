# train_sunlit.py
import os
import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.tiramisu import FCDenseNet103   # from the repo
from datasets.sunlit import SunlitDataset   # the file above

# --------------------------- Metrics (binary) ---------------------------

class BinarySegMetrics:
    """
    Accumulates binary segmentation metrics over *cropped* regions (original sizes).
    Classes: 0 = background, 1 = foreground.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0  # true positives (class 1)
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.correct = 0
        self.total = 0
        self.inter = [0, 0]  # per-class intersection
        self.union = [0, 0]  # per-class union

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y: torch.Tensor,
               orig_h: torch.Tensor, orig_w: torch.Tensor):
        """
        logits: [B,2,512,512], y: [B,512,512]
        orig_h/orig_w: [B] per-sample original sizes
        """
        preds = torch.argmax(logits, dim=1)  # [B,512,512]
        B = preds.shape[0]

        for b in range(B):
            H = int(orig_h[b].item())
            W = int(orig_w[b].item())
            pb = preds[b, :H, :W]
            yb = y[b, :H, :W]

            # confusion terms for foreground=1
            tp = ((pb == 1) & (yb == 1)).sum().item()
            fp = ((pb == 1) & (yb == 0)).sum().item()
            fn = ((pb == 0) & (yb == 1)).sum().item()
            tn = ((pb == 0) & (yb == 0)).sum().item()
            self.tp += tp; self.fp += fp; self.fn += fn; self.tn += tn

            # accuracy (pixel)
            self.correct += (pb == yb).sum().item()
            self.total   += yb.numel()

            # mIoU components per class
            for c in (0, 1):
                pred_c = (pb == c)
                targ_c = (yb == c)
                inter = (pred_c & targ_c).sum().item()
                union = (pred_c | targ_c).sum().item()
                self.inter[c] += inter
                self.union[c] += union

    def compute(self) -> Dict[str, float]:
        acc = self.correct / max(1, self.total)

        precision = self.tp / max(1, (self.tp + self.fp))
        recall    = self.tp / max(1, (self.tp + self.fn))
        f1        = (2 * precision * recall) / max(1e-7, (precision + recall))

        # Dice (foreground). For binary with hard preds, Dice == F1.
        dice = (2 * self.tp) / max(1, (2*self.tp + self.fp + self.fn))

        iou_bg = self.inter[0] / max(1, self.union[0])
        iou_fg = self.inter[1] / max(1, self.union[1])
        miou   = 0.5 * (iou_bg + iou_fg)

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "dice": dice,
            "miou": miou
        }

# --------------------------- Training / Eval ---------------------------

def validate(model, loader, device):
    model.eval()
    metrics = BinarySegMetrics()
    with torch.no_grad():
        for x, y, oh, ow in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            # safety: make sure logits spatial size matches labels
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            metrics.update(logits, y, oh, ow)
    return metrics.compute()

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", dynamic_ncols=True)
    for x, y, _, _ in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)                 # [B,2,H,W]
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        loss = criterion(logits, y)       # CE expects class indices in y
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pbar.set_postfix(loss=loss.item())
    return running_loss / max(1, len(loader.dataset))

def main(args):
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model (random init, as per repo/paper)
    model = FCDenseNet103(n_classes=2).to(device)

    # Datasets / Loaders
    train_ds = SunlitDataset(args.data, "train", augment=True)
    val_ds   = SunlitDataset(args.data, "val",   augment=False)
    test_ds  = SunlitDataset(args.data, "test",  augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=args.bs, shuffle=False,
                              num_workers=args.workers, pin_memory=True, drop_last=False)

    # Loss & Optimizer & Scheduler (per Tiramisu recipe)
    if args.class_weights:
        # e.g. --class-weights 0.3 0.7
        w_bg, w_fg = map(float, args.class_weights)
        weights = torch.tensor([w_bg, w_fg], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma) \
                if args.lr_gamma < 1.0 else None

    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, "best.pt")
    best_miou = -1.0

    # --------------------------- Training loop ---------------------------
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)

        vm = validate(model, val_loader, device)
        val_miou = vm["miou"]

        print(f"[Epoch {epoch:03d}] "
              f"train_loss={train_loss:.6f} | "
              f"VAL: Acc={vm['accuracy']:.4f} mIoU={vm['miou']:.4f} "
              f"Dice={vm['dice']:.4f} F1={vm['f1']:.4f} "
              f"Prec={vm['precision']:.4f} Rec={vm['recall']:.4f}")

        # save best by mIoU with useful metadata
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "val_metrics": vm,
                "args": vars(args),
            }, best_path)
            print(f"✔ Saved new best to {best_path} (mIoU={best_miou:.4f})")

        if scheduler is not None:
            scheduler.step()

    # --------------------------- Testing (best checkpoint) ---------------------------
    print("\n==> Evaluating best checkpoint on TEST split")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    tm = validate(model, test_loader, device)

    print(f"[TEST] Acc={tm['accuracy']:.4f} mIoU={tm['miou']:.4f} "
          f"Dice={tm['dice']:.4f} F1={tm['f1']:.4f} "
          f"Prec={tm['precision']:.4f} Rec={tm['recall']:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./dataset", help="Root directory with images/ and masks/ subfolders")
    ap.add_argument("--bs", type=int, default=4, help="Batch size")
    ap.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (RMSprop)")
    ap.add_argument("--lr_gamma", type=float, default=0.995, help="Exponential LR decay gamma (<1 to enable)")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--out", default="runs/fcd103_sunlit", help="Output directory to save checkpoints")
    ap.add_argument("--class-weights", nargs=2, metavar=("W_BG","W_FG"),
                    help="Optional class weights for CrossEntropyLoss, e.g. --class-weights 0.3 0.7")

    args = ap.parse_args()
    main(args)
