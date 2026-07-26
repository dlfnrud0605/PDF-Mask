import time as _time
work_dir = f'work_dirs/{_time.strftime("%Y%m%d_%H%M%S")}_400e_lr002'

custom_imports = dict(imports=['scripts.custom_hooks'], allow_failed_imports=False)

_base_ = [
    'mmocr::textdet/dbnetpp/_base_dbnetpp_resnet50-dcnv2_fpnc.py',
    'mmocr::textdet/_base_/default_runtime.py',
    'mmocr::textdet/_base_/schedules/schedule_sgd_1200e.py',
]

model = dict(
    det_head=dict(
        module_loss=dict(type='DBModuleLoss', shrink_ratio=0.6, min_sidelength=2),
        postprocessor=dict(text_repr_type='poly', unclip_ratio=1.05, mask_thr=0.4)),
    backbone=dict(
        _delete_=True,
        type='CLIPResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=None,
    )
)

data_root = 'data/train_dataset'

# [최적화] 학습 파이프라인 정의
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(type='TorchVisionWrapper', op='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    dict(type='ImgAugWrapper', args=[['Fliplr', 0.5], ['Resize', [0.3, 1.0]]]),
    dict(type='RandomCrop', min_side_ratio=0.5),
    dict(type='Resize', scale=(1280, 1280), keep_ratio=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1280, 1280), keep_ratio=True),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

custom_train = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=2),
    pipeline=train_pipeline,
)

custom_val = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='textdet_val.json',
    test_mode=True,
    pipeline=test_pipeline,
)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='SizeAwareBatchSampler', drop_last=True),
    collate_fn=dict(type='dynamic_pad_collate'),
    dataset=custom_train,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=custom_val,
)

test_dataloader = val_dataloader

val_evaluator = dict(type='HmeanIOUMetric')
test_evaluator = val_evaluator


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=400, val_interval=20)

custom_hooks = [
    dict(
        type='CheckpointKeyGuard',
        checkpoint_path='models/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.pth',
        critical_prefix='',
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=20,
        max_keep_ckpts=5,
        save_best='icdar/hmean',
        rule='greater',
    ),
    logger=dict(type='LoggerHook', interval=10),
)

# [최적화] 사전 학습된 모델 로드 (0점 문제 해결 및 학습 가속화)
load_from = 'models/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.pth'

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(lr=0.002)
)

param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=400),
]

auto_scale_lr = dict(base_batch_size=16)
