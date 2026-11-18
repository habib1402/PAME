# Leave-one-dataset-out PAME: Tufts + O2PR (AKUDENTAL unseen). The layer slot of the held-out
# protocol stays untrained (used later by few-shot fine-tuning).
_base_ = ['../pame_m2f_swin-t_toa.py']
train_dataloader = dict(dataset=dict(ann_file='data/annotations/train_TO.json'))
