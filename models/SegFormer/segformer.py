import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
except Exception as e:
    raise ImportError(
        "Please install transformers: `pip install transformers`"
    ) from e


class SegFormer2C(nn.Module):
    #Thin wrapper around Hugging Face `SegformerForSemanticSegmentation`
    #that always returns 2-class logits [B,2,H,W] (upsampled to the input size).

    #Notes
    #-----
    # We set `ignore_mismatched_sizes=True` so you can use
    # any `nvidia/segformer-...` checkpoint and still train with 2 classes.
    # As per the official implementation, raw logits are at 1/4 resolution;
    # we bilinearly upsample to the exact input H×W for your trainer/metrics.
    def __init__(
        self,
        backbone: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()

        if pretrained:
            # Load pretrained backbone + decode head; replace classifier to 2 classes.
            config = SegformerConfig.from_pretrained(backbone, num_labels=num_labels)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                backbone,
                config=config,
                ignore_mismatched_sizes=True,  # lets us change num_labels safely
            )
        else:
            # From scratch config (tiny-ish defaults); edit as needed.
            config = SegformerConfig(
                num_labels=num_labels,
                hidden_sizes=[32, 64, 160, 256],
                depths=[2, 2, 2, 2],
                decoder_hidden_size=256,
            )
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W] tensor.
        Returns: logits [B,2,H,W] (upsampled to x size).
        """
        H, W = x.shape[-2:]
        out = self.model(pixel_values=x, return_dict=True)
        # HF SegFormer yields logits at stride 4 by default; upsample to input size.
        logits = out.logits  # [B,2,H/4,W/4] (per HF impl), we upsample below.
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits
