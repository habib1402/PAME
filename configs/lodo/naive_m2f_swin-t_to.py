# Leave-one-dataset-out baseline: naive merge of Tufts + O2PR (AKUDENTAL unseen).
_base_ = ['../naive_m2f_swin-t_toa.py']
train_dataloader = dict(dataset=dict(ann_file='data/annotations/train_TO.json'))
