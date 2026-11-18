"""Score saved predictions (tools/infer.py) on a unified test set.

Segmentation: COCO mask mAP / AP50, class-aware over the FDI classes present in the ground truth, and
class-agnostic mask mAP. Numbering: detections with score >= 0.5, class-agnostic NMS 0.5, arch-order decoding
(tools/numbering.py), Hungarian matching on box IoU >= 0.5; F1 of correctly numbered teeth and the rate of
perfect charts.

  python tools/evaluate.py --pred preds/pame_Dual.pkl --ann data/annotations/test_Dual_labeled.json [--json out.json]
"""
import argparse, collections, contextlib, copy, io, json, os, pickle, sys
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from numbering import ArchDecoder, fdi_jaw, iou, nms_agnostic  # noqa: E402


def to_fdi(n):
    n = str(n)
    n = n[6:] if n.startswith('tooth-') else n
    return int(n) if n.isdigit() else None


def detections(P, cid):
    dets = []
    for img_id, r in P['images'].items():
        for k in range(len(r['scores'])):
            f = to_fdi(P['classes'][r['labels'][k]])
            if f not in cid:
                continue
            rle = dict(r['rles'][k])
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode()
            dets.append(dict(image_id=img_id, category_id=cid[f], segmentation=rle, score=float(r['scores'][k])))
    return dets


def segm(P, gt_file, cid):
    dets, out = detections(P, cid), {}
    with contextlib.redirect_stdout(io.StringIO()):
        G = COCO(gt_file)
        for agn in (False, True):
            E = COCOeval(G, G.loadRes(copy.deepcopy(dets)), 'segm')
            E.params.useCats = 0 if agn else 1
            E.evaluate(); E.accumulate(); E.summarize()
            out['agnostic' if agn else 'class_aware'] = dict(mAP=100 * E.stats[0], AP50=100 * E.stats[1])
    return out


def numbering(P, G, tau=0.5, extra_cost=9.0):
    classes = [to_fdi(c) for c in P['classes']]
    valid = [i for i, f in enumerate(classes) if f is not None]
    cls = [classes[i] for i in valid]
    dec = ArchDecoder(cls, extra_cost=extra_cost, side_agnostic=True)
    gcat = {c['id']: to_fdi(c['name']) for c in G['categories']}
    gt = collections.defaultdict(list)
    for a in G['annotations']:
        x, y, w, h = a['bbox']; gt[a['image_id']].append(([x, y, x + w, y + h], gcat[a['category_id']]))
    C = collections.Counter()
    for img_id, r in P['images'].items():
        g = gt.get(img_id, []); gb = np.array([b for b, _ in g], float).reshape(-1, 4); gl = [f for _, f in g]
        m = r['scores'] >= tau
        b, s, fs = r['bboxes'][m], r['scores'][m], r['full_scores'][m].astype(np.float64)
        keep = nms_agnostic(b, s) if len(b) else np.array([], int)
        b, fs = b[keep], fs[keep]
        probs = fs[:, valid]; probs = probs / np.clip(probs.sum(1, keepdims=True), 1e-9, None)
        cen = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]
        lab = dec.decode(cen, probs) if len(b) else []
        sel = [i for i, l in enumerate(lab) if l is not None]
        C['gt'] += len(gl); C['pred'] += len(sel); corr = 0
        if sel and gl:
            ov = iou(b[sel], gb); ri, ci = linear_sum_assignment(-ov)
            corr = sum(ov[i, j] >= 0.5 and lab[sel[i]] == gl[j] for i, j in zip(ri, ci))
        C['correct'] += corr; C['img'] += 1; C['perfect'] += (corr == len(gl) and len(sel) == len(gl))
    rec = 100 * C['correct'] / max(C['gt'], 1); pre = 100 * C['correct'] / max(C['pred'], 1)
    return dict(F1=2 * rec * pre / max(rec + pre, 1e-9), recall=rec, precision=pre,
                perfect_chart=100 * C['perfect'] / max(C['img'], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True); ap.add_argument('--ann', required=True); ap.add_argument('--json')
    a = ap.parse_args()
    G = json.load(open(a.ann)); cid = {to_fdi(c['name']): c['id'] for c in G['categories']}
    P = pickle.load(open(a.pred, 'rb'))
    r = dict(segmentation=segm(P, a.ann, cid), numbering=numbering(P, G))
    s, n = r['segmentation'], r['numbering']
    print(f"mask mAP {s['class_aware']['mAP']:.1f}  AP50 {s['class_aware']['AP50']:.1f}  "
          f"class-agnostic mAP {s['agnostic']['mAP']:.1f}  numbering F1 {n['F1']:.2f}  perfect charts {n['perfect_chart']:.1f}%")
    if a.json:
        json.dump(r, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
