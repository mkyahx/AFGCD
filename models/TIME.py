import torch
import torch.nn as nn
from timm.layers.mlp import Mlp
from timm.models.layers import trunc_normal_
from timm.layers.weight_init import trunc_normal_tf_

class Simple_Cross_Attention(nn.Module):
    def __init__(self, feat_dim=768, mlp_ratio=4.0, mask_alpha=1.0):
        super().__init__()
        
        self.D_sqrt = feat_dim**-0.5
        self.learnable_query = nn.Parameter(torch.empty(1, feat_dim))
        self.mlp_norm = nn.LayerNorm(feat_dim, eps=1e-6)
        self.mlp = Mlp(feat_dim, int(feat_dim * mlp_ratio))
        self.mask_alpha = mask_alpha

        trunc_normal_tf_(self.learnable_query, std=self.D_sqrt)

    def forward(self, x, patch_mask=None):
        B = x.shape[0]
        # QKV
        q = self.learnable_query.repeat(B, 1).unsqueeze(1)
        k, v = x, x
        attn_raw = (q @ k.transpose(-2, -1)) * self.D_sqrt
        
        attn_sm = attn_raw.softmax(dim=-1)

        # Soft mask (post-softmax): foreground keeps weight=1, background scaled by mask_alpha, then renormalize.
        # patch_mask: (B, P), 1 = foreground (keep), 0 = background (suppress)
        if patch_mask is not None and self.mask_alpha != 1.0:
            N = attn_sm.shape[-1]
            P = patch_mask.shape[-1]
            if N == P + 1:
                cls_keep = torch.ones(B, 1, dtype=patch_mask.dtype, device=patch_mask.device)
                full_mask = torch.cat([cls_keep, patch_mask], dim=1)  # (B, N)
                scale = torch.where(
                    full_mask.bool(),
                    torch.ones_like(full_mask),
                    torch.full_like(full_mask, float(self.mask_alpha)),
                )
                attn_sm = attn_sm * scale.unsqueeze(1)
                attn_sm = attn_sm / (attn_sm.sum(dim=-1, keepdim=True) + 1e-12)

        x = (attn_sm @ v).squeeze(1)
        # FFN
        x = x + self.mlp(self.mlp_norm(x))
        
        # Return unbiased raw attention for correct pruning during both train and test!
        return x, attn_raw.squeeze(1)

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

class Token_Importance_Measurer(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=768, mask_alpha=1.0):
        super().__init__()
        self.sim_cross_attn = Simple_Cross_Attention(feat_dim=feat_dim, mask_alpha=mask_alpha)
        self.aux_head = Aux_Head(feat_dim, num_classes)

    def forward(self, x, patch_mask=None):
        weighted_feat, attn = self.sim_cross_attn(x, patch_mask=patch_mask)
        pred = self.aux_head(weighted_feat)
        return attn, pred
