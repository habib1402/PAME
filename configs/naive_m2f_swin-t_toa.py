# Baseline: standard Mask2Former Swin-T trained on the naive merge of Tufts + O2PR + AKUDENTAL.
_base_ = ['./_base_/m2f_swin-t_model.py', './_base_/data_tooth.py']
