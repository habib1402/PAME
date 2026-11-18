"""Train with an MMDetection 3.x config (the `pame` plugin is loaded through custom_imports in the configs).

  python tools/train.py configs/pame_m2f_swin-t_toa.py --work-dir work_dirs/pame_m2f_swin-t_toa [--cfg-options randomness.seed=1]
"""
import argparse
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmdet.utils import register_all_modules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('--work-dir', required=True)
    ap.add_argument('--cfg-options', nargs='+', action=DictAction)
    a = ap.parse_args()
    register_all_modules()
    cfg = Config.fromfile(a.config)
    if a.cfg_options:
        cfg.merge_from_dict(a.cfg_options)
    cfg.work_dir = a.work_dir
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
