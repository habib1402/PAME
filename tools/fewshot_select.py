"""Pick the best protocol layer or blend for a new site from K labeled images, report mAP on the rest
(mean over random draws).

    python tools/fewshot_select.py --ann data/annotations/test_AKU_OPG.json T=h0.pkl O=h1.pkl A=h2.pkl b121=b121.pkl --K 3
"""
import argparse, contextlib, copy, io, json, os, pickle, random, sys
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import detections, to_fdi  # noqa: E402


def mAP(G, dt, ids):
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(G, dt, 'segm'); E.params.imgIds = list(ids); E.evaluate(); E.accumulate(); E.summarize()
    return 100 * E.stats[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True); ap.add_argument('--K', type=int, nargs='+', default=[1, 3, 5, 10])
    ap.add_argument('--draws', type=int, default=20); ap.add_argument('candidates', nargs='+', help='LABEL=preds.pkl')
    a = ap.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        G = COCO(a.ann)
    cid = {to_fdi(c['name']): c['id'] for c in json.load(open(a.ann))['categories']}
    DT = {}
    for c in a.candidates:
        lab, path = c.split('=', 1)
        with contextlib.redirect_stdout(io.StringIO()):
            DT[lab] = G.loadRes(copy.deepcopy(detections(pickle.load(open(path, 'rb')), cid)))
    ids = sorted(G.getImgIds())
    full = {l: mAP(G, d, ids) for l, d in DT.items()}
    print('full-set mAP: ' + ', '.join(f'{l} {v:.1f}' for l, v in full.items()))
    print(f'oracle: {max(full, key=full.get)} {max(full.values()):.1f}')
    for K in a.K:
        rng = random.Random(0); res, picks = [], []
        for _ in range(a.draws):
            sel = rng.sample(ids, K); rest = [i for i in ids if i not in set(sel)]
            sc = {l: mAP(G, d, sel) for l, d in DT.items()}
            best = max(sc, key=sc.get); picks.append(best); res.append(mAP(G, DT[best], rest))
        print(f'K={K:2d}: {np.mean(res):.1f} +- {np.std(res):.1f}  picked ' +
              str({l: picks.count(l) for l in DT if picks.count(l)}))


if __name__ == '__main__':
    main()
