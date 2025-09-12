# train_sunlit.py
import os
import argparse
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.tiramisu import FCDenseNet103           # original Tiramisu
from models.ca_denseunetpp import CADenseUNetPP     # CA-DenseUNet++
from models.densenext_unet import DenseNeXtUNet     # <-- NEW: DenseNeXt-UNet
from datasets.sunlit import SunlitDataset           # pads to 512, returns (x,y,oh,ow)


# --------------------------- Metrics (binary) ---------------------------

class BinarySegMetrics:
    """Accumulates binary segmentation metrics over cropped regions (original sizes)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0; self.fp = 0; self.fn = 0; self.tn = 0
        self.correct = 0; self.total = 0
        self.inter = [0, 0]; self.union = [0, 0]

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y: torch.Tensor,
               orig_h: torch.Tensor, orig_w: torch.Tensor):
        preds = torch.argmax(logits, dim=1)
        B = preds.shape[0]
        for b in range(B):
            H = int(orig_h[b].item()); W = int(orig_w[b].item())
            pb = preds[b, :H, :W]; yb = y[b, :H, :W]
            tp = ((pb == 1) & (yb == 1)).sum().item()
            fp = ((pb == 1) & (yb == 0)).sum().item()
            fn = ((pb == 0) & (yb == 1)).sum().item()
            tn = ((pb == 0) & (yb == 0)).sum().item()
            self.tp += tp; self.fp += fp; self.fn += fn; self.tn += tn
            self.correct += (pb == yb).sum().item()
            self.total   += yb.numel()
            for c in (0, 1):
                pred_c = (pb == c); targ_c = (yb == c)
                inter = (pred_c & targ_c).sum().item()
                union = (pred_c | targ_c).sum().item()
                self.inter[c] += inter; self.union[c] += union

    def compute(self) -> Dict[str, float]:
        acc = self.correct / max(1, self.total)
        precision = self.tp / max(1, (self.tp + self.fp))
        recall    = self.tp / max(1, (self.tp + self.fn))
        f1        = (2 * precision * recall) / max(1e-7, (precision + recall))
        dice      = (2 * self.tp) / max(1, (2*self.tp + self.fp + self.fn))
        iou_bg    = self.inter[0] / max(1, self.union[0])
        iou_fg    = self.inter[1] / max(1, self.union[1])
        miou      = 0.5 * (iou_bg + iou_fg)
        return {"accuracy": acc, "precision": precision, "recall": recall,
                "f1": f1, "dice": dice, "miou": miou}


# --------------------------- Training / Eval ---------------------------

@torch.no_grad()
def validate(model, loader, device, channels_last: bool = False, amp: bool = True):
    model.eval()
    metrics = BinarySegMetrics()
    for x, y, oh, ow in loader:
        if channels_last:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(x)

        logits = outputs[-1] if isinstance(outputs, list) else outputs
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

        metrics.update(logits, y, oh, ow)
    return metrics.compute()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs,
                    scaler: torch.cuda.amp.GradScaler, grad_accum: int = 1,
                    channels_last: bool = False, amp: bool = True,
                    ds_weights: Optional[List[float]] = None):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", dynamic_ncols=True)

    optimizer.zero_grad(set_to_none=True)
    for step, (x, y, _, _) in enumerate(pbar, start=1):
        if channels_last:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(x)
            if isinstance(outputs, list):
                outs = [o if o.shape[-2:] == y.shape[-2:] else
                        F.interpolate(o, size=y.shape[-2:], mode="bilinear", align_corners=False)
                        for o in outputs]
                if ds_weights is None or len(ds_weights) != len(outs):
                    w = [1.0/len(outs)] * len(outs)
                else:
                    w = ds_weights
                loss = sum(wi * criterion(oi, y) for wi, oi in zip(w, outs)) / grad_accum
            else:
                logits = outputs
                if logits.shape[-2:] != y.shape[-2:]:
                    logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(logits, y) / grad_accum

        scaler.scale(loss).backward()

        if step % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += (loss.item() * grad_accum) * x.size(0)
        pbar.set_postfix(loss=(loss.item() * grad_accum))

    if (step % grad_accum) != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / max(1, len(loader.dataset))


def human_readable_count(n: int) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000.0:
            return f"{n:.0f}{unit}"
        n /= 1000.0
    return f"{n:.0f}T"


def main(args):
    torch.backends.cudnn.benchmark = True

    # CPU for now (your current setting)
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # --------------------------- Model ---------------------------
    if args.model == "densenext_unet":
        model = DenseNeXtUNet(
            n_classes=2,
            in_channels=args.in_ch,   # 3 for RGB; set 5 if you add extra channels
            growth=16,
            dense_layers=(4,5),
            cnx_depths=(3,3,3),
            bot_depth=4,
            stem_out=48,
            dims_cnx=(256,384,512,640),
            drop_path=0.0
        ).to(device)
    elif args.model == "ca_denseunetpp":
        model = CADenseUNetPP(
            n_classes=2,
            norm=args.norm,
            deep_supervision=bool(args.ds),
            ca_encoder_stages=tuple(args.ca_enc_stages),
            ca_fusion_min_j=int(args.ca_fusion_min_j),
        ).to(device)
        if args.ds:
            print("Deep supervision: ON (validation/test use deepest head only).")
    else:
        model = FCDenseNet103(n_classes=2).to(device)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # ---- Print parameter counts ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: total={human_readable_count(total_params)} "
          f"trainable={human_readable_count(trainable_params)} "
          f"({total_params:,} / {trainable_params:,})")

    # --------------------------- Data ---------------------------
    train_ds = SunlitDataset(args.data, "train", augment=True)
    val_ds   = SunlitDataset(args.data, "val",   augment=False)
    test_ds  = SunlitDataset(args.data, "test",  augment=False)

    print(f"Dataset sizes -> train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=args.bs, shuffle=False,
                              num_workers=args.workers, pin_memory=True, drop_last=False)

    # --------------------------- Loss / Optim ---------------------------
    if args.class_weights:
        w_bg, w_fg = map(float, args.class_weights)
        weights = torch.tensor([w_bg, w_fg], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"Using weighted CE: w_bg={w_bg}, w_fg={w_fg}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma) \
                if args.lr_gamma < 1.0 else None

    # AMP scaler (keep disabled on CPU by passing --amp 0)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))

    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, "best.pt")
    best_miou = -1.0

    # --------------------------- Train Loop ---------------------------
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs,
            scaler=scaler, grad_accum=args.grad_accum,
            channels_last=bool(args.channels_last), amp=bool(args.amp),
            ds_weights=args.ds_weights
        )

        vm = validate(model, val_loader, device,
                      channels_last=bool(args.channels_last), amp=bool(args.amp))
        val_miou = vm["miou"]

        print(f"[Epoch {epoch:03d}] "
              f"train_loss={train_loss:.6f} | "
              f"VAL: Acc={vm['accuracy']:.4f} mIoU={vm['miou']:.4f} "
              f"Dice={vm['dice']:.4f} F1={vm['f1']:.4f} "
              f"Prec={vm['precision']:.4f} Rec={vm['recall']:.4f}")

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

    # --------------------------- Test ---------------------------
    print("\n==> Evaluating best checkpoint on TEST split")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    tm = validate(model, test_loader, device,
                  channels_last=bool(args.channels_last), amp=bool(args.amp))
    print(f"[TEST] Acc={tm['accuracy']:.4f} mIoU={tm['miou']:.4f} "
          f"Dice={tm['dice']:.4f} F1={tm['f1']:.4f} "
          f"Prec={tm['precision']:.4f} Rec={tm['recall']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./dataset", help="Root directory with images/ and masks/ subfolders")

    # Model selection & options
    ap.add_argument("--model", default="densenext_unet",
                    choices=["densenext_unet", "ca_denseunetpp", "tiramisu"],
                    help="Choose model backbone")
    ap.add_argument("--in-ch", type=int, default=3,
                    help="Input channels (3 for RGB; set 5 if you add extra channels)")
    ap.add_argument("--norm", default="gn", choices=["bn","gn"],
                    help="Normalization (used by CA-DenseUNet++; ignored by DenseNeXt-UNet)")
    ap.add_argument("--ds", type=int, default=0,
                    help="Deep supervision for CA-DenseUNet++ (1=yes, 0=no)")
    ap.add_argument("--ds-weights", nargs="*", type=float, default=None,
                    help="Deep supervision weights (e.g. --ds-weights 0.5 0.3 0.2)")
    ap.add_argument("--ca-enc-stages", dest="ca_enc_stages", nargs="*", type=int, default=[0,1],
                    help="Encoder stages to place CoordAtt (CA-DenseUNet++)")
    ap.add_argument("--ca-fusion-min-j", dest="ca_fusion_min_j", type=int, default=2,
                    help="Apply CoordAtt on fusion nodes with depth j >= this value (CA-DenseUNet++)")
    ap.add_argument("--dnu-dims", nargs=4, type=int, default=[128, 192, 256, 320],
                help="DenseNeXt-UNet dims for stages 2/3/4 and bottleneck, e.g. 192 288 384 512")
    ap.add_argument("--dnu-depths", nargs=3, type=int, default=[2, 2, 2],
                    help="ConvNeXt block counts for stages 2/3/4, e.g. 3 3 3")
    ap.add_argument("--dnu-growth", type=int, default=12, help="Dense block growth (early stages)")
    ap.add_argument("--dnu-stem", type=int, default=32, help="Stem out channels")
    ap.add_argument("--dnu-bot-depth", type=int, default=2, help="Bottleneck ConvNeXt depth")
    ap.add_argument("--dnu-drop-path", type=float, default=0.0, help="Stochastic depth for ConvNeXt blocks")
    ap.add_argument("--dnu-checkpoint", type=int, default=1, help="Gradient checkpointing in ConvNeXt stages (1/0)")


    # Training setup
    ap.add_argument("--bs", type=int, default=2, help="Batch size (per step)")
    ap.add_argument("--grad-accum", type=int, default=2, help="Accumulate this many steps before optimizer.step()")
    ap.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (RMSprop)")
    ap.add_argument("--lr_gamma", type=float, default=0.995, help="Exponential LR decay gamma (<1 to enable)")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--out", default="runs/densenext_unet", help="Output directory to save checkpoints")
    ap.add_argument("--class-weights", nargs=2, metavar=("W_BG","W_FG"),
                    help="Optional class weights for CrossEntropyLoss, e.g. --class-weights 0.3 0.7")

    # Memory / precision knobs (keep off on CPU)
    ap.add_argument("--amp", type=int, default=0, help="Use mixed precision (CUDA only) (1=yes, 0=no)")
    ap.add_argument("--channels-last", type=int, default=0, help="Use channels_last memory format (0/1)")

    args = ap.parse_args()
    main(args)
