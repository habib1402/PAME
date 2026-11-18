"""Write predictions as a COCO-style results file (RLE masks, no images).

    python tools/export_predictions.py --pred preds/x.pkl --ann data/annotations/test_AKU_OPG.json --out results/x.json
"""
import argparse, json, os, pickle, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import to_fdi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True); ap.add_argument('--ann', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    P = pickle.load(open(a.pred, 'rb')); G = json.load(open(a.ann))
    name = {im['id']: os.path.basename(im['file_name']) for im in G['images']}
    out = []
    for i, r in P['images'].items():
        for k in range(len(r['scores'])):
            rle = dict(r['rles'][k])
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode()
            out.append(dict(file_name=name[i], fdi=to_fdi(P['classes'][r['labels'][k]]), score=round(float(r['scores'][k]), 4),
                            bbox=[round(float(v), 1) for v in r['bboxes'][k]], segmentation=rle))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'))
    print(f'{len(out)} detections on {len(P["images"])} images -> {a.out}')


if __name__ == '__main__':
    main()
