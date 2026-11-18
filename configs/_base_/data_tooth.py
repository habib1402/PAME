# Unified 52-class FDI tooth data (build with tools/prepare_data.py). File names in the JSON files are absolute.
custom_imports = dict(imports=['pame'], allow_failed_imports=False)
ann_root = 'data/annotations/'
classes = tuple(str(q * 10 + t) for q in (1, 2, 3, 4) for t in range(1, 9)) + \
          tuple(str(q * 10 + t) for q in (5, 6, 7, 8) for t in range(1, 6))
metainfo = dict(classes=classes)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='Pad', size=(1024, 512), pad_val=dict(img=0)),
    dict(type='RandomFlipFDI', prob=0.5, class_names=classes),   # flip that also mirrors FDI numbers
    dict(type='PackDetInputs')
]
test_pipeline = [   # no Pad at test time (Pad would overwrite img_shape and squeeze the masks)
    dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
train_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True), batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='CocoDataset', metainfo=metainfo, data_root='', data_prefix=dict(img=''),
                 ann_file=ann_root + 'train_TOA.json', filter_cfg=dict(filter_empty_gt=True, min_size=32),
                 pipeline=train_pipeline, backend_args=None))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', metainfo=metainfo, data_root='', data_prefix=dict(img=''),
                 ann_file=ann_root + 'val_TA.json', test_mode=True, pipeline=test_pipeline, backend_args=None))
test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric', ann_file=ann_root + 'val_TA.json', metric=['bbox', 'segm'], backend_args=None)
test_evaluator = val_evaluator
