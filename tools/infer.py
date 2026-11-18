"""Run a model on a test set and save boxes, scores, labels and RLE masks.

    python tools/infer.py CONFIG CKPT --ann data/annotations/test_Dual_labeled.json --out preds/x.pkl \
        [--protocol 1 | --blend 0.25,0.5,0.25]
"""
import argparse, json, os, pickle
import numpy as np
import pycocotools.mask as mu
from mmdet.apis import inference_detector, init_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('checkpoint')
    ap.add_argument('--ann', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--protocol', type=int); ap.add_argument('--blend')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    model = init_detector(a.config, a.checkpoint, device=a.device)
    head = getattr(model, 'roi_head', None) or model.panoptic_head
    if a.protocol is not None:
        head.test_protocol = a.protocol
    if a.blend:
        head.test_weights = [float(w) for w in a.blend.split(',')]
    classes = list(model.dataset_meta['classes'])
    d = json.load(open(a.ann))
    res = {'classes': classes, 'images': {}}
    for k, im in enumerate(d['images']):
        r = inference_detector(model, im['file_name']).pred_instances
        m = r.masks.cpu().numpy().astype(np.uint8)
        labels = r.labels.cpu().numpy()
        res['images'][im['id']] = dict(
            bboxes=r.bboxes.cpu().numpy().astype(np.float32), labels=labels.astype(np.int16),
            scores=r.scores.cpu().numpy().astype(np.float32),
            full_scores=np.eye(len(classes) + 1)[labels].astype(np.float16),
            rles=[mu.encode(np.asfortranarray(x)) for x in m])
        if k % 50 == 0:
            print(f'{k}/{len(d["images"])}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pickle.dump(res, open(a.out, 'wb'))
    print('saved', a.out)


if __name__ == '__main__':
    main()
