"""Initialize an untrained protocol slot of a PAME checkpoint as a blend of trained output layers
(start point for few-shot fine-tuning of one layer, configs/fewshot/).

  python tools/init_new_layer.py CKPT OUT --slot 1 --weights 0.75,0,0.25
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint'); ap.add_argument('out')
    ap.add_argument('--slot', type=int, required=True); ap.add_argument('--weights', required=True)
    a = ap.parse_args()
    w = [float(x) for x in a.weights.split(',')]
    ck = torch.load(a.checkpoint, map_location='cpu'); sd = ck['state_dict']; p = 'panoptic_head.mask_embed_out.'
    for t in ('weight', 'bias'):
        sd[f'{p}{a.slot}.{t}'] = sum(wk * sd[f'{p}{k}.{t}'] for k, wk in enumerate(w) if wk)
    torch.save(dict(state_dict=sd, meta=ck.get('meta', {})), a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
