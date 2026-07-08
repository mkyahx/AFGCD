import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.mlp import Mlp
from timm.layers.weight_init import trunc_normal_tf_
from timm.models.layers import trunc_normal_


class Simple_Cross_Attention_Alpha(nn.Module):
    def __init__(self, feat_dim=768, mlp_ratio=4.0, alpha=1.0):
        super().__init__()

        if alpha < 0:
            raise ValueError(f"alpha ({alpha}) must be non-negative.")

        self.D_sqrt = feat_dim**-0.5
        self.alpha = float(alpha)

        self.learnable_query = nn.Parameter(torch.empty(1, feat_dim))
        self.mlp_norm = nn.LayerNorm(feat_dim, eps=1e-6)
        self.mlp = Mlp(feat_dim, int(feat_dim * mlp_ratio))

        trunc_normal_tf_(self.learnable_query, std=self.D_sqrt)

    def _align_patch_mask(self, patch_mask, batch_size, token_count, device):
        if patch_mask is None:
            return torch.ones(batch_size, token_count, device=device, dtype=torch.float32)

        patch_mask = patch_mask.to(device=device, dtype=torch.float32)

        if patch_mask.dim() == 4 and patch_mask.shape[1] == 1:
            patch_mask = patch_mask.squeeze(1)

        if patch_mask.dim() == 3:
            patch_mask = patch_mask.unsqueeze(1)
        elif patch_mask.dim() == 2:
            if patch_mask.shape[-1] == token_count:
                return (patch_mask > 0.5).float()

            side = int(math.sqrt(patch_mask.shape[-1]))
            if side * side != patch_mask.shape[-1]:
                raise ValueError(
                    f"Cannot reshape patch mask with shape {tuple(patch_mask.shape)} to a square grid."
                )
            patch_mask = patch_mask.view(batch_size, 1, side, side)
        else:
            raise ValueError(f"Unsupported patch mask shape: {tuple(patch_mask.shape)}")

        target_side = int(math.sqrt(token_count))
        if target_side * target_side != token_count:
            raise ValueError(f"Token count {token_count} is not a square patch grid.")

        if patch_mask.shape[-2:] != (target_side, target_side):
            patch_mask = F.interpolate(patch_mask, size=(target_side, target_side), mode="nearest")

        return (patch_mask.squeeze(1).reshape(batch_size, token_count) > 0.5).float()

    def forward(self, x, patch_mask=None):
        B, num_tokens, _ = x.shape

        q = self.learnable_query.repeat(B, 1).unsqueeze(1)
        k, v = x, x
        raw_attn = q @ k.transpose(-2, -1)
        attn_logits = raw_attn * self.D_sqrt

        if patch_mask is not None and self.alpha != 1.0:
            aligned_patch_mask = self._align_patch_mask(patch_mask, B, num_tokens - 1, x.device)
            cls_mask = torch.ones(B, 1, device=x.device, dtype=aligned_patch_mask.dtype)
            full_mask = torch.cat([cls_mask, aligned_patch_mask], dim=1)

            alpha_mask = torch.where(
                full_mask > 0.5,
                torch.ones_like(full_mask),
                torch.full_like(full_mask, self.alpha),
            ).clamp_min(1e-12)
            log_alpha_mask = alpha_mask.log().unsqueeze(1)
            attn_logits = attn_logits + log_alpha_mask
            raw_attn = raw_attn + log_alpha_mask

        attn_sm = attn_logits.softmax(dim=-1)
        x = (attn_sm @ v).squeeze(1)

        x = x + self.mlp(self.mlp_norm(x))
        return x, raw_attn.squeeze(1)


class Aux_Head(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.head = nn.Linear(feat_dim, num_classes)
        trunc_normal_(self.head.weight, std=0.02)
        self.head.bias.data.zero_()

    def forward(self, x):
        x = self.norm(x)
        x = self.head(x)
        return x


class Token_Importance_Measurer_Alpha(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=768, alpha=1.0):
        super().__init__()
        self.sim_cross_attn = Simple_Cross_Attention_Alpha(
            feat_dim=feat_dim,
            alpha=alpha,
        )
        self.aux_head = Aux_Head(feat_dim, num_classes)

    def forward(self, x, patch_mask=None):
        weighted_feat, attn = self.sim_cross_attn(x, patch_mask=patch_mask)
        pred = self.aux_head(weighted_feat)
        return attn, pred
