"""Data transforms for multi-protocol tooth segmentation.

AddSourceId   : tags every training image with the index of its source dataset (annotation protocol).
RandomFlipFDI : horizontal flip that also mirrors FDI tooth numbers. A flipped panoramic radiograph shows the
                patient's right on the image right, so tooth 11 must become 21 (quadrants 1<->2, 3<->4, 5<->6,
                7<->8). The stock RandomFlip moves boxes and masks but keeps the labels.
"""
import re

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import RandomFlip
from mmdet.registry import TRANSFORMS

_SWAP_Q = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7}


@TRANSFORMS.register_module()
class AddSourceId(BaseTransform):
    """Set ``results['src_id']`` to the index of the first pattern contained in the image path.

    Args:
        patterns (sequence[str]): one substring per source dataset, in protocol order
            (e.g. ``('tufts', 'o2pr', 'akudental')`` for directories named after the datasets).
    """

    def __init__(self, patterns):
        self.patterns = [p.lower() for p in patterns]

    def transform(self, results):
        p = results['img_path'].lower()
        ids = [i for i, s in enumerate(self.patterns) if s in p]
        assert ids, f'AddSourceId: no source pattern matches {results["img_path"]}'
        results['src_id'] = ids[0]
        return results


def _fdi(name):
    m = re.search(r'(\d{2})$', str(name))
    return int(m.group(1)) if m else None


@TRANSFORMS.register_module()
class RandomFlipFDI(RandomFlip):
    """RandomFlip that also mirrors FDI class labels on horizontal flips.

    Args:
        class_names (sequence[str]): dataset classes in label order; names must end with the FDI code
            (e.g. ``'11'`` or ``'tooth-11'``).
    """

    def __init__(self, class_names, **kwargs):
        super().__init__(**kwargs)
        idx = {f: i for i, n in enumerate(class_names) if (f := _fdi(n)) is not None}
        perm = np.arange(len(class_names))
        for f, i in idx.items():
            q, t = divmod(f, 10)
            m = _SWAP_Q.get(q, q) * 10 + t
            if m in idx:
                perm[i] = idx[m]
        self.label_perm = perm

    def _flip(self, results):
        super()._flip(results)
        if results.get('flip_direction') == 'horizontal' and results.get('gt_bboxes_labels') is not None:
            results['gt_bboxes_labels'] = self.label_perm[results['gt_bboxes_labels']]
