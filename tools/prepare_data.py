"""Build the unified 52-class FDI COCO annotations used in the paper from the original dataset downloads.

No images or annotations are redistributed: every dataset is read from its own download (see docs/DATASETS.md),
converted to one FDI label space, cleaned (duplicates removed, AKUDENTAL lower-jaw quadrants corrected), and split
according to the lists in splits/. The result is checked against those lists.

Usage:
  python tools/prepare_data.py \
      --tufts data/tufts_seed0            # output of tools/prepare_tufts.sh (seed 0)
      --o2pr data/raw/o2pr                # Roboflow COCO export: train/ and valid/ with _annotations.coco.json
      --akudental data/raw/akudental      # AKUDENTAL repo folder with images/, annotations/, folds/fold_0/labels/
      --aku-opg data/raw/aku_opg          # extracted AKU OPG archive (contains dataset/folder 1, 2, 3)
      --dual data/raw/dual_labeled        # Dual-labeled: images1/ and labels/
      --out data/annotations
"""
import argparse, glob, json, os, re
import numpy as np
from PIL import Image

FDI = [q * 10 + t for q in (1, 2, 3, 4) for t in range(1, 9)] + [q * 10 + t for q in (5, 6, 7, 8) for t in range(1, 6)]
CATS = [{'id': i + 1, 'name': str(f), 'supercategory': 'tooth'} for i, f in enumerate(FDI)]
CID = {f: i + 1 for i, f in enumerate(FDI)}
SPLITS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'splits')


def name_to_fdi(n):
    n = str(n)
    n = n[6:] if n.startswith('tooth-') else n
    return int(n) if n.isdigit() and int(n) in CID else None


def lead_int(s):
    m = re.match(r'\s*(\d+)', s)
    return int(m.group(1)) if m else None


def find_img(folder, stem):
    for ext in ('.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.bmp'):
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    return None


def labelme_to_coco(pairs, get_fdi, swap_lower=False):
    """LabelMe polygons -> COCO with FDI categories (teeth only). swap_lower: mirror quadrants 3<->4, 7<->8."""
    images, anns = [], []
    for k, (jf, img) in enumerate(pairs):
        if img is None:
            continue
        d = json.load(open(jf)); w, h = Image.open(img).size
        images.append({'id': k + 1, 'file_name': os.path.abspath(img), 'width': w, 'height': h})
        for s in d['shapes']:
            f = get_fdi(s)
            if f not in CID or s.get('shape_type', 'polygon') != 'polygon' or len(s['points']) < 3:
                continue
            if swap_lower and f // 10 in (3, 4, 7, 8):
                f = {3: 4, 4: 3, 7: 8, 8: 7}[f // 10] * 10 + f % 10
            pts = np.array(s['points'], float); x0, y0 = pts.min(0); x1, y1 = pts.max(0)
            anns.append({'id': len(anns) + 1, 'image_id': k + 1, 'category_id': CID[f], 'iscrowd': 0,
                         'bbox': [x0, y0, x1 - x0, y1 - y0], 'area': float((x1 - x0) * (y1 - y0)),
                         'segmentation': [pts.reshape(-1).tolist()]})
    return {'images': images, 'annotations': anns, 'categories': CATS}


def subset(d, src, keep=None, drop=()):
    """(images, annotations) of a COCO dict with unified categories; keep/drop by relative name."""
    cmap = {c['id']: name_to_fdi(c['name']) for c in d['categories']}
    imgs, ids = [], set()
    for im in d['images']:
        rel = relname(src, im['file_name'])
        if (keep is not None and rel not in keep) or rel in drop:
            continue
        im = dict(im); im['source'] = src; imgs.append(im); ids.add(im['id'])
    anns = []
    for a in d['annotations']:
        f = cmap.get(a['category_id'])
        if a['image_id'] in ids and f is not None:
            a = dict(a); a['category_id'] = CID[f]; a.setdefault('iscrowd', 0); anns.append(a)
    return imgs, anns


def relname(src, path):
    return path.split('/dataset/')[-1] if src == 'AKU_OPG' else os.path.basename(path)


def with_dir(d, img_dir):
    d = dict(d); d['images'] = [dict(im, file_name=os.path.abspath(os.path.join(img_dir, im['file_name']))) for im in d['images']]
    return d


def read_list(name):
    return [tuple(l.rstrip('\n').split('\t')[:2]) for l in open(f'{SPLITS}/{name}.txt') if l.strip()]


def write(out, name, parts):
    images, anns, nid, aid = [], [], 1, 1
    for imgs, ann in parts:
        remap = {}
        for im in imgs:
            im = dict(im); remap[im['id']] = nid; im['id'] = nid; nid += 1; images.append(im)
        for a in ann:
            a = dict(a); a['image_id'] = remap[a['image_id']]; a['id'] = aid; aid += 1; anns.append(a)
    json.dump({'images': images, 'annotations': anns, 'categories': CATS}, open(f'{out}/{name}.json', 'w'))
    got = sorted((im['source'], relname(im['source'], im['file_name'])) for im in images)
    ref = f'{SPLITS}/{name}.txt'
    status = ''
    if os.path.exists(ref):
        status = 'matches splits/' if got == sorted(read_list(name)) else 'MISMATCH with splits/ (check your downloads)'
    print(f'{name:22} {len(images):5d} images {len(anns):6d} teeth  {status}')


def main():
    ap = argparse.ArgumentParser()
    for k in ('tufts', 'o2pr', 'akudental', 'aku_opg', 'dual', 'out'):
        ap.add_argument('--' + k.replace('_', '-'), required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    drop = {}
    for l in open(f'{SPLITS}/removed_duplicates.txt'):
        ds, rel = l.rstrip('\n').split('\t')[:2]; drop.setdefault(ds, set()).add(rel)

    tufts = {s: subset(with_dir(json.load(open(f'{a.tufts}/annotations/instances_{s}.json')), f'{a.tufts}/{s}'), 'Tufts')
             for s in ('train', 'val', 'test')}
    o2pr = {s: subset(with_dir(json.load(open(f'{a.o2pr}/{s}/_annotations.coco.json')), f'{a.o2pr}/{s}'), 'O2PR')
            for s in ('train', 'valid')}
    ak = labelme_to_coco([(j, find_img(f'{a.akudental}/images', os.path.basename(j)[:-5]))
                          for j in sorted(glob.glob(f'{a.akudental}/annotations/*.json'))],
                         lambda s: lead_int(s['label']), swap_lower=True)
    F = f'{a.akudental}/folds/fold_0/labels'
    # AKUDENTAL official fold 0: keep the images whose file stem is listed in folds/fold_0/labels/<split>
    aku_all = subset(ak, 'AKUDENTAL', drop=drop.get('AKUDENTAL', ()))
    aku = {}
    for s in ('train', 'val', 'test'):
        stems = {os.path.splitext(x)[0] for x in os.listdir(f'{F}/{s}')}
        keep = {im['id'] for im in aku_all[0] if os.path.splitext(os.path.basename(im['file_name']))[0] in stems}
        aku[s] = ([im for im in aku_all[0] if im['id'] in keep], [x for x in aku_all[1] if x['image_id'] in keep])
    pairs = []
    for folder in sorted(glob.glob(f'{a.aku_opg}/**/dataset/*/', recursive=True)) or sorted(glob.glob(f'{a.aku_opg}/dataset/*/')):
        ann_dir = [d for d in glob.glob(folder + 'ann*') if os.path.isdir(d)][0]
        pairs += [(j, find_img(folder + 'OPGs', os.path.basename(j)[:-5])) for j in sorted(glob.glob(ann_dir + '/*.json'))]
    opg = subset(labelme_to_coco(pairs, lambda s: s.get('group_id')), 'AKU_OPG', drop=drop.get('AKU_OPG', ()))
    dual = subset(labelme_to_coco([(j, find_img(f'{a.dual}/images1', os.path.basename(j)[:-5]))
                                   for j in sorted(glob.glob(f'{a.dual}/labels/*.json'))], lambda s: lead_int(s['label'])),
                  'Dual_labeled', drop=drop.get('Dual_labeled', ()))

    o = a.out
    write(o, 'train_TOA', [tufts['train'], o2pr['train'], aku['train']])
    write(o, 'train_TO', [tufts['train'], o2pr['train']]); write(o, 'train_TA', [tufts['train'], aku['train']])
    write(o, 'train_O', [o2pr['train']])
    write(o, 'val_TA', [tufts['val'], aku['val']])
    write(o, 'test_Tufts', [tufts['test']]); write(o, 'test_O2PR', [o2pr['valid']]); write(o, 'test_AKUDENTAL', [aku['test']])
    write(o, 'test_AKU_OPG', [opg]); write(o, 'test_Dual_labeled', [dual])
    for n in sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{SPLITS}/fewshot_*.txt')):
        keep = {rel for _, rel in read_list(n)}
        write(o, n, [subset({'images': o2pr['train'][0], 'annotations': o2pr['train'][1], 'categories': CATS}, 'O2PR', keep=keep)])


if __name__ == '__main__':
    main()
