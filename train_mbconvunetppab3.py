import os
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- Models ----

from models.Abalation.mbnext_unetpp_ablate import MBNeXtUNetPP


# ---- Dataset (pads to 512, returns (x, y, orig_h, orig_w)) ----
from datasets.sunlit import SunlitDataset


# ============================== Metrics ==============================

class BinarySegMetrics:
    """
    Accumulates binary segmentation metrics on the *original (unpadded)* region.
    Classes: 0 = background, 1 = foreground.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0; self.fp = 0; self.fn = 0; self.tn = 0
        self.correct = 0; self.total = 0
        self.inter = [0, 0]; self.union = [0, 0]

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y: torch.Tensor,
               orig_h: torch.Tensor, orig_w: torch.Tensor):
        # logits [B,2,512,512], y [B,512,512]
        preds = torch.argmax(logits, dim=1)
        B = preds.shape[0]
        for b in range(B):
            H = int(orig_h[b].item())
            W = int(orig_w[b].item())
            pb = preds[b, :H, :W]
            yb = y[b, :H, :W]

            tp = ((pb == 1) & (yb == 1)).sum().item()
            fp = ((pb == 1) & (yb == 0)).sum().item()
            fn = ((pb == 0) & (yb == 1)).sum().item()
            tn = ((pb == 0) & (yb == 0)).sum().item()
            self.tp += tp; self.fp += fp; self.fn += fn; self.tn += tn

            self.correct += (pb == yb).sum().item()
            self.total   += yb.numel()

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
        dice      = (2 * self.tp) / max(1, (2*self.tp + self.fp + self.fn))
        iou_bg = self.inter[0] / max(1, self.union[0])
        iou_fg = self.inter[1] / max(1, self.union[1])
        miou   = 0.5 * (iou_bg + iou_fg)
        return {"accuracy": acc, "precision": precision, "recall": recall,
                "f1": f1, "dice": dice, "miou": miou}


# ============================== Losses ==============================

class SoftDiceLoss(nn.Module):
    """Binary soft-Dice on foreground channel from 2-class logits."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # logits: [B,2,H,W]; y: [B,H,W] in {0,1}
        p = torch.softmax(logits, dim=1)[:, 1]  # [B,H,W]
        y1 = (y == 1).float()
        inter = (p * y1).sum(dim=(1, 2))
        denom = (p + y1).sum(dim=(1, 2))
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class ComboLoss(nn.Module):
    """alpha * CE + (1 - alpha) * SoftDice."""
    def __init__(self, alpha: float = 0.5,
                 class_weights: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.05,
                 smooth: float = 1e-6):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights,
                                      label_smoothing=label_smoothing)
        self.dice = SoftDiceLoss(smooth)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.ce(logits, y) + (1.0 - self.alpha) * self.dice(logits, y)


# ============================== EMA ==============================

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: p.detach().clone()
                       for k, p in model.state_dict().items()
                       if isinstance(p, torch.Tensor) and p.dtype.is_floating_point}
        self._backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.state_dict().items():
            if k in self.shadow and isinstance(p, torch.Tensor) and p.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self._backup = {}
        for k, p in model.state_dict().items():
            if k in self.shadow and isinstance(p, torch.Tensor) and p.dtype.is_floating_point:
                self._backup[k] = p.detach().clone()
                p.copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for k, v in self._backup.items():
            model.state_dict()[k].copy_(v)
        self._backup = {}


# ============================== TTA ==============================

@torch.no_grad()
def predict_tta(model: nn.Module, x: torch.Tensor, amp: bool, device_type: str) -> torch.Tensor:
    """
    Returns averaged logits over flips.
    If model returns list (deep supervision), we use the deepest head for TTA.
    """
    outs = []

    def _forward(inp):
        with torch.amp.autocast(device_type=device_type, enabled=amp):
            out = model(inp)
        return out[-1] if isinstance(out, list) else out

    # identity
    outs.append(_forward(x))
    # hflip
    xf = torch.flip(x, dims=[-1])
    outs.append(torch.flip(_forward(xf), dims=[-1]))
    # vflip
    xf = torch.flip(x, dims=[-2])
    outs.append(torch.flip(_forward(xf), dims=[-2]))
    # hvflip
    xf = torch.flip(x, dims=[-2, -1])
    outs.append(torch.flip(_forward(xf), dims=[-2, -1]))

    return torch.stack(outs, dim=0).mean(dim=0)


# ============================== Train / Eval ==============================

def human_readable_count(n: int) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000.0:
            return f"{n:.0f}{unit}"
        n /= 1000.0
    return f"{n:.0f}T"


def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             channels_last: bool, amp: bool, use_tta: bool,
             ema: Optional[ModelEMA]) -> Dict[str, float]:
    if ema is not None:
        ema.apply_to(model)
    model.eval()

    device_type = "cuda" if device.type == "cuda" else "cpu"
    metrics = BinarySegMetrics()
    with torch.no_grad():
        for x, y, oh, ow in loader:
            if channels_last:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if use_tta:
                logits = predict_tta(model, x, amp=amp and device.type == "cuda", device_type=device_type)
            else:
                with torch.amp.autocast(device_type=device_type, enabled=amp and device.type == "cuda"):
                    out = model(x)
                logits = out[-1] if isinstance(out, list) else out

            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

            metrics.update(logits, y, oh, ow)

    if ema is not None:
        ema.restore(model)
    return metrics.compute()


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, total_epochs: int, scaler: "torch.amp.GradScaler",
                    grad_accum: int, channels_last: bool, amp: bool,
                    ds_weights: Optional[List[float]], ema: Optional[ModelEMA]) -> float:
    model.train()
    device_type = "cuda" if device.type == "cuda" else "cpu"

    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", dynamic_ncols=True)

    optimizer.zero_grad(set_to_none=True)
    for step, (x, y, _, _) in enumerate(pbar, start=1):
        if channels_last:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, enabled=amp and device.type == "cuda"):
            outputs = model(x)

            if isinstance(outputs, list):
                outs = [o if o.shape[-2:] == y.shape[-2:] else
                        F.interpolate(o, size=y.shape[-2:], mode="bilinear", align_corners=False)
                        for o in outputs]
                if not ds_weights or len(ds_weights) != len(outs):
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
            if ema is not None:
                ema.update(model)

        running_loss += (loss.item() * grad_accum) * x.size(0)
        pbar.set_postfix(loss=(loss.item() * grad_accum))

    # Flush tail grads if we didn't hit the boundary
    if (step % grad_accum) != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)

    return running_loss / max(1, len(loader.dataset))


# ============================== Main ==============================

def main(args):
    # Device
    device = torch.device("cuda:2" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Using device: {device}")

    # --------------------------- Model ---------------------------
    
    if args.model == "mbnext_unetpp":
        model = MBNeXtUNetPP(
            n_classes=2,
            in_channels=args.in_ch,
            stage0_out=96, stage1_out=160,         # keep skip widths compatible
            stage0_blocks=3, stage1_blocks=4,
            stage0_exp=4, stage1_exp=6,
            stage0_k=5, stage1_k=5,
            stage0_se=0.0, stage1_se=0.25,
            norm_mb="gn", gn_groups=8,
            dims_cnx=tuple(args.dnu_dims),         # e.g., 224 320 416 544
            cnx_depths=tuple(args.dnu_depths),
            bot_depth=int(args.dnu_bot_depth),
            drop_path=float(args.dnu_drop_path),
            use_checkpoint=bool(args.dnu_checkpoint),
            dec_depth=int(args.dec_depth),
            deep_supervision=True,
            enable_A=False,
            enable_B=True
        ).to(device)


    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Parameter counts
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
                              num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=max(1, args.bs), shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=max(1, args.bs), shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=False)

    # --------------------------- Loss ---------------------------
    if args.class_weights:
        w_bg, w_fg = map(float, args.class_weights)
        weights = torch.tensor([w_bg, w_fg], dtype=torch.float32, device=device)
        print(f"Using weighted loss: w_bg={w_bg}, w_fg={w_fg}")
    else:
        weights = None

    if args.loss == "combo":
        criterion = ComboLoss(alpha=args.combo_alpha,
                              class_weights=weights,
                              label_smoothing=args.ce_smooth,
                              smooth=1e-6)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.ce_smooth)

    # --------------------------- Optim / Sched ---------------------------
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.sched == "exp" and args.lr_gamma < 1.0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cosine_T0, T_mult=args.cosine_Tmul)
    else:
        scheduler = None

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and device.type == "cuda")

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    # Output dir
    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, "best.pt")
    best_miou = -1.0

    # --------------------------- Train Loop ---------------------------
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs,
            scaler=scaler, grad_accum=args.grad_accum,
            channels_last=bool(args.channels_last), amp=bool(args.amp),
            ds_weights=args.ds_weights, ema=ema
        )

        vm = validate(model, val_loader, device,
                      channels_last=bool(args.channels_last), amp=bool(args.amp),
                      use_tta=bool(args.tta), ema=ema)
        val_miou = vm["miou"]

        print(f"[Epoch {epoch:03d}] "
              f"train_loss={train_loss:.6f} | "
              f"VAL: Acc={vm['accuracy']:.4f} mIoU={vm['miou']:.4f} "
              f"Dice={vm['dice']:.4f} F1={vm['f1']:.4f} "
              f"Prec={vm['precision']:.4f} Rec={vm['recall']:.4f}")

        # save best by mIoU
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
            print(f"? Saved new best to {best_path} (mIoU={best_miou:.4f})")

        if scheduler is not None:
            if args.sched == "cosine":
                scheduler.step(epoch)  # CosineAnnealingWarmRestarts expects epoch or fraction
            else:
                scheduler.step()

    # --------------------------- Test ---------------------------
    print("\n==> Evaluating best checkpoint on TEST split")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    tm = validate(model, test_loader, device,
                  channels_last=bool(args.channels_last), amp=bool(args.amp),
                  use_tta=bool(args.tta), ema=ema)

    print(f"[TEST] Acc={tm['accuracy']:.4f} mIoU={tm['miou']:.4f} "
          f"Dice={tm['dice']:.4f} F1={tm['f1']:.4f} "
          f"Prec={tm['precision']:.4f} Rec={tm['recall']:.4f}")
    print(f"Best checkpoint path: {best_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./dataset", help="Root directory with images/ and masks/ subfolders")
    ap.add_argument("--device", default="cuda:1", choices=["cuda:1","cpu"], help="Force device (default tries CUDA)")

    # Model selection
    ap.add_argument("--model", default="mbnext_unetpp",
                    choices=["densenext_unetpp", "densenext_unet", "ca_denseunetpp", "tiramisu","mbnext_unetpp"],
                    help="Choose model backbone")
    ap.add_argument("--in-ch", type=int, default=3, help="Input channels (RGB=3)")

    # DenseNeXt sizing knobs (also used by -unetpp)
    ap.add_argument("--dnu-dims", nargs=4, type=int, default=[224, 320, 416, 544],
                    help="Dims for ConvNeXt stages 2/3/4 and bottleneck, e.g. 224 320 416 544")
    ap.add_argument("--dnu-depths", nargs=3, type=int, default=[3, 3, 3],
                    help="ConvNeXt block counts for stages 2/3/4, e.g. 3 3 3")
    ap.add_argument("--dnu-growth", type=int, default=16, help="Dense growth (early stages)")
    ap.add_argument("--dnu-stem", type=int, default=48, help="Stem out channels")
    ap.add_argument("--dnu-bot-depth", type=int, default=3, help="Bottleneck ConvNeXt depth")
    ap.add_argument("--dnu-drop-path", type=float, default=0.0, help="Stochastic depth for ConvNeXt blocks")
    ap.add_argument("--dnu-checkpoint", type=int, default=1, help="Gradient checkpointing (1/0)")
    ap.add_argument("--dec-depth", type=int, default=1, help="UNet++ columns depth (for densenext_unetpp)")

    # CA-DenseUNet++ extras
    ap.add_argument("--norm", default="gn", choices=["bn","gn"],
                    help="Normalization for CA-DenseUNet++")
    ap.add_argument("--ds", type=int, default=1, help="Deep supervision for CA-DenseUNet++ (1/0)")
    ap.add_argument("--ds-weights", nargs="*", type=float, default=None,
                    help="Deep supervision weights (e.g. --ds-weights 0.3 0.3 0.4)")
    ap.add_argument("--ca-enc-stages", dest="ca_enc_stages", nargs="*", type=int, default=[0,1],
                    help="Encoder stages to place CoordAtt, e.g. --ca-enc-stages 0 1")
    ap.add_argument("--ca-fusion-min-j", dest="ca_fusion_min_j", type=int, default=2,
                    help="Apply CoordAtt on fusion nodes with depth j >= this value")

    # Training setup
    ap.add_argument("--bs", type=int, default=1, help="Batch size per step")
    ap.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    ap.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (RMSprop)")
    ap.add_argument("--lr_gamma", type=float, default=0.995, help="Gamma for ExponentialLR")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--out", default="runs/mbnunetab3", help="Output directory")

    # Loss / scheduler / EMA / TTA
    ap.add_argument("--loss", choices=["ce", "combo"], default="combo", help="Loss: CE or CE+Dice combo")
    ap.add_argument("--combo-alpha", type=float, default=0.5, help="Weight for CE in ComboLoss (1-alpha for Dice)")
    ap.add_argument("--ce-smooth", type=float, default=0.05, help="Label smoothing for CE")
    ap.add_argument("--class-weights", nargs=2, metavar=("W_BG","W_FG"), help="Optional CE class weights, e.g. 0.3 0.7")
    ap.add_argument("--sched", choices=["exp","cosine","none"], default="cosine", help="LR scheduler")
    ap.add_argument("--cosine-T0", type=int, default=10, help="Cosine warm-restart T0")
    ap.add_argument("--cosine-Tmul", type=int, default=2, help="Cosine warm-restart T_mult")
    ap.add_argument("--ema", type=int, default=0, help="Use EMA for evaluation (0/1)")
    ap.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay")
    ap.add_argument("--tta", type=int, default=1, help="Flip TTA at val/test (0/1)")

    # Memory tuning
    ap.add_argument("--amp", type=int, default=1, help="Mixed precision on CUDA (0/1)")
    ap.add_argument("--channels-last", type=int, default=1, help="Use channels_last memory format (0/1)")

    args = ap.parse_args()
    main(args)
