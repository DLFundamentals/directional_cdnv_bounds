import os
import torch
from typing import Dict, Any


def export_teacher_encoder_only(
    lightly_mae_module,
    out_path: str,
    extra_meta: Dict[str, Any] | None = None,
):
    """
    Saves only the teacher encoder/backbone weights + minimal metadata.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # In our LightlyMAE module:
    # lightly_mae_module.backbone is MaskedVisionTransformerTIMM
    # We want the underlying ViT weights for use in Stage 2.
    state = lightly_mae_module.backbone.state_dict()

    meta = {
        "type": "teacher_encoder_only",
        "vit_name": lightly_mae_module.cfg.stage.vit_name,
        "mask_ratio_stage1": float(lightly_mae_module.cfg.stage.mask_ratio),
        "patch_size": int(lightly_mae_module.patch_size),
        "sequence_length": int(lightly_mae_module.sequence_length),
    }
    if extra_meta:
        meta.update(extra_meta)

    payload = {"state_dict": state, "meta": meta}
    torch.save(payload, out_path)
