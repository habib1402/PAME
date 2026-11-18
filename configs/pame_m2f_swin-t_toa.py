# PAME: shared Mask2Former Swin-T, one linear mask-embedding output layer per annotation protocol
# (0 Tufts, 1 O2PR, 2 AKUDENTAL). Training routes every image to the layer of its source dataset.
# Inference: model.panoptic_head.test_protocol = k (one layer) or .test_weights = [w0, w1, w2] (blend).
_base_ = ['./naive_m2f_swin-t_toa.py']
model = dict(panoptic_head=dict(type='ProtocolMask2FormerHead', num_protocols=3, test_protocol=1))
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddSourceId', patterns=('Tufts', 'O2PR', 'AKUDENTAL')),   # matched against the image path
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='Pad', size=(1024, 512), pad_val=dict(img=0)),
    dict(type='RandomFlipFDI', prob=0.5, class_names={{_base_.classes}}),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip',
                                          'flip_direction', 'src_id'))
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
