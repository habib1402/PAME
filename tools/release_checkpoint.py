"""Prepare a checkpoint for public release: keep only the model weights (drops optimizer state, about 3x smaller)
and print its SHA-256 for the model zoo table.

  python tools/release_checkpoint.py work_dirs/pame_m2f_swin-t_toa/epoch_50.pth pame_m2f_swin-t_toa_seed0.pth
"""
import argparse, hashlib
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('checkpoint'); ap.add_argument('out'); a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location='cpu')
    meta = {k: v for k, v in ck.get('meta', {}).items() if k in ('dataset_meta', 'mmdet_version', 'epoch', 'iter')}
    torch.save(dict(state_dict=ck['state_dict'], meta=meta), a.out)
    h = hashlib.sha256(open(a.out, 'rb').read()).hexdigest()
    print(f'saved {a.out}  sha256 {h}')


if __name__ == '__main__':
    main()
