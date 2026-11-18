"""Paired per-image comparison of two prediction files: Wilcoxon test and bootstrap 95% CI of the mean difference.

    python tools/significance.py --ann data/annotations/test_AKU_OPG.json --a naive.pkl --b pame.pkl
"""
import argparse, contextlib, copy, io, json, os, pickle, sys
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import detections, to_fdi  # noqa: E402


def per_image(G, P, cid, ids):
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        D = G.loadRes(copy.deepcopy(detections(P, cid)))
        for i in ids:
            E = COCOeval(G, D, 'segm'); E.params.imgIds = [i]; E.evaluate(); E.accumulate(); E.summarize()
            out[i] = 100 * max(E.stats[0], 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    a = ap.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        G = COCO(a.ann)
    cid = {to_fdi(c['name']): c['id'] for c in json.load(open(a.ann))['categories']}
    ids = sorted(G.getImgIds())
    pa = per_image(G, pickle.load(open(a.a, 'rb')), cid, ids); pb = per_image(G, pickle.load(open(a.b, 'rb')), cid, ids)
    d = np.array([pb[i] - pa[i] for i in ids])
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    print(f'mean per-image difference (B - A) {d.mean():+.2f}  95% CI [{np.percentile(boot, 2.5):+.2f}, '
          f'{np.percentile(boot, 97.5):+.2f}]  B better on {100 * (d > 0).mean():.1f}% of images  '
          f'Wilcoxon p = {wilcoxon(d).pvalue:.2e}')


if __name__ == '__main__':
    main()
