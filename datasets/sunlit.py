# datasets/sunlit.py
import os
import glob
import random
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# Your dataset specifics:
# - Images:  RGB .jpg (or .jpeg)
# - Masks:   PNG where white (>=128) = sunlit/foreground (1), black (<128) = background (0)
# - Pairing: image "abc.jpg" ↔ mask "abc_L.png"
# - No resizing. Pad to 512x512 (pad right & bottom). Return original (H,W).

IMG_EXTS  = (".jpg", ".jpeg")
MASK_EXTS = (".png",)

def _pad_to_512(t: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pad a CHW (image) or HW (mask) tensor to 512x512 on the right and bottom.
    Returns (padded_tensor, (orig_h, orig_w)).
    """
    if t.ndim == 3:
        _, h, w = t.shape
    else:
        h, w = t.shape

    pad_h = 512 - h
    pad_w = 512 - w
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"Found size {h}x{w} > 512; this loader only pads up to 512.")

    # pad = (left, top, right, bottom)
    pad = (0, 0, pad_w, pad_h)
    padded = TF.pad(t, pad, fill=0)
    return padded, (h, w)

def _collect(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
    return paths

def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

class SunlitDataset(Dataset):
    """
    Expects directory layout:

      root/
        images/
          train/*.jpg
          val/*.jpg
          test/*.jpg
        masks/
          train/*_L.png
          val/*_L.png
          test/*_L.png

    For each image "<stem>.jpg", pairs the mask "<stem>_L.png".
    Loads original sizes, optional flips (train only), pads to 512x512 (right & bottom),
    normalizes image, and returns (img_512, mask_512, orig_h, orig_w).
    """
    def __init__(self, root: str, split: str = "train", augment: bool = False):
        super().__init__()
        assert split in {"t_small", "val", "test"}

        self.root = root
        self.split = split
        self.augment = (split == "train") and augment

        img_dir  = os.path.join(root, "images", split)
        mask_dir = os.path.join(root, "masks",  split)

        img_paths  = _collect(img_dir, IMG_EXTS)
        mask_paths = _collect(mask_dir, MASK_EXTS)

        # index masks by their full stem ("abc_L")
        masks_by_stem = {_stem(p): p for p in mask_paths}

        pairs = []
        missing = []
        for ip in img_paths:
            s = _stem(ip)           # "abc"
            mask_key = f"{s}_L"     # expect "abc_L"
            mp = masks_by_stem.get(mask_key, None)
            if mp is not None:
                pairs.append((ip, mp))
            else:
                missing.append((ip, mask_key))

        if missing:
            examples = "\n".join([f"- image: {ip}  expected mask stem: {mk}" for ip, mk in missing[:8]])
            raise FileNotFoundError(
                f"[SunlitDataset] Could not find masks for {len(missing)} images in split='{split}'.\n"
                f"Expected mask filenames as '<image_stem>_L.png'. Examples:\n{examples}\n"
                f"Check your masks directory: {mask_dir}"
            )

        # deterministic order
        self.pairs = sorted(pairs, key=lambda t: t[0])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        img_p, msk_p = self.pairs[i]

        # RGB JPG image
        img = Image.open(img_p).convert("RGB")

        # PNG mask, white sunlit, black background
        msk = Image.open(msk_p).convert("L")          # 0..255 grayscale

        # to tensors
        img_t = TF.to_tensor(img)                     # [3,H,W], float32 in [0,1]
        msk_np = np.array(msk, dtype=np.uint8)        # [H,W], 0..255
        msk_bin = (msk_np >= 128).astype(np.uint8)    # threshold: white→1, black→0
        mask_t = torch.from_numpy(msk_bin).long()     # [H,W], {0,1}

        # simple flips before padding (keeps original size)
        if self.augment:
            if random.random() < 0.5:
                img_t = TF.hflip(img_t)
                mask_t = TF.hflip(mask_t)
            if random.random() < 0.2:
                img_t = TF.vflip(img_t)
                mask_t = TF.vflip(mask_t)

        # pad to 512x512 (right & bottom)
        img_t, (oh, ow) = _pad_to_512(img_t)
        mask_t, _ = _pad_to_512(mask_t)

        # normalize (ImageNet stats)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

        # return original H/W as tensors for easy batching
        return img_t, mask_t, torch.tensor(oh, dtype=torch.int32), torch.tensor(ow, dtype=torch.int32)
