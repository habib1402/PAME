"""Protocol-aware RoI head (PAME for Mask R-CNN-type detectors).

Different datasets draw tooth masks with different protocols (root extent,
boundary style). A single mask head averages them. This head keeps one shared
detector/classifier and gives each annotation protocol (source dataset) its own
mask output.

Config:
  num_protocols : number of mask heads (1 = single shared head)
  shared_convs  : convs shared by all heads before the protocol-specific part
                  (the mask_head config then holds only the protocol part)
  mask_sources  : source ids whose masks are used in the mask loss
                  (None = all). Other sources still train RPN, box, class.
  test_protocol : head used at inference (settable after loading)
  test_weights  : optional list of per-head weights; if set, inference blends
                  the heads' mask logits (sum_k w_k * head_k), overriding test_protocol
Each training image must carry `src_id` in its metainfo (AddSourceId transform).
"""
import copy
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.models.roi_heads import StandardRoIHead


@MODELS.register_module()
class ProtocolRoIHead(StandardRoIHead):

    def __init__(self, *args, num_protocols=1, shared_convs=0, mask_sources=None,
                 test_protocol=0, **kwargs):
        self.num_protocols = num_protocols
        self.n_shared = shared_convs
        self.mask_sources = mask_sources
        self.test_protocol = test_protocol
        self.test_weights = None
        self._src = None
        super().__init__(*args, **kwargs)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = MODELS.build(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        c = mask_head.get('in_channels', 256)
        self.mask_trunk = nn.Sequential(*[
            ConvModule(c, c, 3, padding=1, conv_cfg=None, norm_cfg=None)
            for _ in range(self.n_shared)])
        self.mask_heads = nn.ModuleList(
            [MODELS.build(copy.deepcopy(mask_head)) for _ in range(self.num_protocols)])

    @property
    def mask_head(self):
        return self.mask_heads[self.test_protocol]

    @property
    def with_mask(self):
        return hasattr(self, 'mask_heads')

    def _head_of(self, src):
        return src if self.num_protocols > 1 else 0

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None, heads=None):
        if rois is not None:
            feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
        else:
            feats = bbox_feats[pos_inds]
        feats = self.mask_trunk(feats)
        if heads is None and self.test_weights is not None:
            preds = sum(w * self.mask_heads[k](feats)
                        for k, w in enumerate(self.test_weights) if w > 0)
        elif heads is None:
            preds = self.mask_heads[self.test_protocol](feats)
        else:
            preds = None
            for h in heads.unique().tolist():
                sel = heads == h
                p = self.mask_heads[h](feats[sel])
                if preds is None:
                    preds = p.new_zeros((feats.shape[0],) + p.shape[1:])
                preds[sel] = p
        return dict(mask_preds=preds, mask_feats=feats)

    def loss(self, x, rpn_results_list, batch_data_samples):
        self._src = [int(ds.metainfo['src_id']) for ds in batch_data_samples]
        return super().loss(x, rpn_results_list, batch_data_samples)

    def mask_loss(self, x, sampling_results, bbox_feats, batch_gt_instances):
        keep = [i for i, s in enumerate(self._src)
                if self.mask_sources is None or s in self.mask_sources]
        keep = [i for i in keep if len(sampling_results[i].pos_priors) > 0]
        if not keep:
            zero = sum(p.sum() for p in self.mask_heads.parameters()) * 0.
            zero = zero + sum(p.sum() for p in self.mask_trunk.parameters()) * 0.
            return dict(loss_mask=dict(loss_mask=zero), mask_preds=None)
        sr = [sampling_results[i] for i in keep]
        gt = [batch_gt_instances[i] for i in keep]
        rois = bbox2roi([r.pos_priors for r in sr])
        heads = torch.cat([rois.new_full((len(r.pos_priors),), self._head_of(self._src[i]),
                                         dtype=torch.long) for r, i in zip(sr, keep)])
        res = self._mask_forward(x, rois, heads=heads)
        lt = self.mask_heads[0].loss_and_target(
            mask_preds=res['mask_preds'], sampling_results=sr,
            batch_gt_instances=gt, rcnn_train_cfg=self.train_cfg)
        res.update(loss_mask=lt['loss_mask'])
        return res
