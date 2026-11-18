"""Protocol-aware mask embedding (PAME) for Mask2Former.

Mask2Former draws each instance mask as  mask = mask_embed(query) . pixel_features.
Here mask_embed is split into a shared trunk (first two layers) and one output
layer per annotation protocol (source dataset), the same design as `pshared` in
ProtocolRoIHead. Queries, classification and the pixel decoder stay shared.

Training: each image uses the output layer of its source (`src_id` in metainfo,
from the AddSourceId transform).
Inference: `test_protocol` selects one layer, or `test_weights` blends them
(blending embeddings == blending mask logits, since the mask is linear in them).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.models.dense_heads import Mask2FormerHead


@MODELS.register_module()
class ProtocolMask2FormerHead(Mask2FormerHead):

    def __init__(self, *args, num_protocols=3, test_protocol=0, **kwargs):
        super().__init__(*args, **kwargs)
        c = self.mask_embed[0].in_features
        out = self.mask_embed[-1].out_features
        trunk = list(self.mask_embed.children())[:-1]
        del self.mask_embed
        self.mask_embed_trunk = nn.Sequential(*trunk)
        self.mask_embed_out = nn.ModuleList([nn.Linear(c, out) for _ in range(num_protocols)])
        self.num_protocols = num_protocols
        self.test_protocol = test_protocol
        self.test_weights = None
        self._w = None              # (B, K) per-image protocol weights for the current forward

    def _set_weights(self, batch_data_samples, device):
        B = len(batch_data_samples)
        src = [ds.metainfo.get('src_id') for ds in batch_data_samples]
        if self.training:
            assert all(s is not None for s in src), 'training images need src_id (AddSourceId)'
            w = F.one_hot(torch.tensor(src, device=device), self.num_protocols).float()
        elif self.test_weights is not None:
            w = torch.tensor(self.test_weights, device=device, dtype=torch.float).expand(B, -1)
        else:
            w = F.one_hot(torch.full((B,), self.test_protocol, device=device, dtype=torch.long),
                          self.num_protocols).float()
        self._w = w

    def _forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        cls_pred = self.cls_embed(decoder_out)
        e = self.mask_embed_trunk(decoder_out)                         # (B, Q, C)
        outs = torch.stack([o(e) for o in self.mask_embed_out], 1)     # (B, K, Q, C')
        mask_embed = (self._w[:, :, None, None] * outs).sum(1)         # (B, Q, C')
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode='bilinear', align_corners=False)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = (attn_mask.sigmoid() < 0.5).detach()
        return cls_pred, mask_pred, attn_mask

    def forward(self, x, batch_data_samples):
        self._set_weights(batch_data_samples, x[0].device)
        return super().forward(x, batch_data_samples)
