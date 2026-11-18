# Leave-one-dataset-out baseline: naive merge of Tufts + AKUDENTAL (O2PR unseen).
_base_ = ['../naive_m2f_swin-t_toa.py']
train_dataloader = dict(dataset=dict(ann_file='data/annotations/train_TA.json'))
