# Few-shot fine-tuning of ONE output layer for an unseen protocol (O2PR) on the leave-one-out Tufts + AKUDENTAL
# PAME model. Layer slot 1 (O2PR, never trained) is initialized as the 0.75 / 0.25 blend of the Tufts and
# AKUDENTAL layers (tools/init_new_layer.py), then only this layer is trained on K labeled O2PR images.
# Other draws: --cfg-options train_dataloader.dataset.ann_file=data/annotations/fewshot_O2PR_K10_d1.json
# Test with: model.panoptic_head.test_protocol = 1.
_base_ = ['../lodo/pame_m2f_swin-t_ta.py']
K = 10
load_from = 'work_dirs/pame_m2f_swin-t_ta/init_slot1_b75.pth'
train_dataloader = dict(batch_size=2, num_workers=2, persistent_workers=False,
                        sampler=dict(_delete_=True, type='InfiniteSampler', shuffle=True),
                        dataset=dict(ann_file=f'data/annotations/fewshot_O2PR_K{K}.json'))
optim_wrapper = dict(optimizer=dict(lr=1e-4),          # 1e-3 destabilizes the masked attention
                     paramwise_cfg=dict(_delete_=True, custom_keys={
                         'panoptic_head.mask_embed_out.1': dict(lr_mult=1.0, decay_mult=1.0),
                         '': dict(lr_mult=0.0, decay_mult=0.0)}))
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=300, val_interval=100000)
param_scheduler = [dict(type='ConstantLR', factor=1.0, by_epoch=False, begin=0, end=300)]
val_cfg = None
val_dataloader = None
val_evaluator = None
default_hooks = dict(checkpoint=dict(_delete_=True, type='CheckpointHook', interval=300, by_epoch=False, max_keep_ckpts=1),
                     logger=dict(type='LoggerHook', interval=20))
log_processor = dict(by_epoch=False)
