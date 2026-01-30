import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from types import SimpleNamespace
from typing import List, Union
from PIL import Image
import torch
import torch.nn as nn
from model.vlam_embedder import InternVL3Embedder
from model.action_expert.flow_matching_post import FlowmatchingActionHeadd
import time

class caar(nn.Module):
    def __init__(self, config: dict):
        super().__init__() 
        self.config = config
        self._device = config.get("device", "cuda")
        self.embedder = InternVL3Embedder(model_name=cfg.pretrain_vlm_path, device=self._device)

        self.action_expert = FlowmatchingActionHeadd(cfg=SimpleNamespace(
            embed_dim=config.get("embed_dim", 896),
            horizon=config.get("horizon", 16),
            action_dim=config.get("action_dim", 7),
            state_dim=config.get("state_dim", 7),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 8),
            dropout=config.get("dropout", 0.0),
            device=self._device,
        )).to(self._device)

    def get_vlm_embeddings(self, images, image_mask, prompt):
        if images is None or len(images) == 0:
            raise ValueError("Must provide at least one image (PIL.Image). Got `images=None` or empty list.")
        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image=images,
            image_mask=image_mask,
            text_prompt=prompt,
        )

    def predict_actions(self, fused_tokens, fused_mask, state, actions_gt=None):
        if actions_gt is None:
            return self.action_expert.get_actions(fused_tokens, fused_mask, state)
        else:
            return self.action_expert(fused_tokens, fused_mask, state, actions_gt=actions_gt)

    @torch.no_grad()
    def inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str,
        state: Union[list, torch.Tensor],
    ) -> torch.Tensor:
        t_start = time.time()
        fused_tokens, fused_mask = self.get_vl_embeddings(
                        images=images,
                        image_mask=image_mask,
                        prompt=prompt,
                    )
        t_end = time.time()
        print(f"[Timing] Inference time: {t_end - t_start:.4f} seconds")
        return self.predict_actions(fused_tokens, fused_mask, state)

    def forward(self, fused_tokens, fused_mask, state, actions_gt=None):
        return self.predict_actions(fused_tokens, fused_mask, state, actions_gt)
