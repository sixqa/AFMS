_base_ = ['mmpose::_base_/default_runtime.py']

codec = dict(type='SimCCLabel', input_size=(256, 256), sigma=(12, 12),
             simcc_split_ratio=2.0, normalize=False, use_dark=False)

work_dir = 'runs/kp_crayfish'

stage2_num_epochs = 30
train_cfg = dict(max_epochs=1000, val_interval=1)

optim_wrapper = dict(type='OptimWrapper',
                      optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
                      paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0,
                                          bypass_duplicate=True))

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=4e-3*0.05,
         begin=1000//2, end=1000, T_max=1000//2,
         by_epoch=True, convert_to_iter_based=True),
]

auto_scale_lr = dict(base_batch_size=1024)

metainfo = dict(
    classes=('CRAYFISH',),
    dataset_name='crayfish_KEYPOINTS_keypoint',
    keypoint_info={
        0:  {"name": "tou1", "id": 0, "color": [255, 0, 0], "type": "", "swap": ""},
        1:  {"name": "zhonbu", "id": 1, "color": [0, 255, 0], "type": "", "swap": ""},
        2:  {"name": "wei1", "id": 2, "color": [0, 0, 255], "type": "", "swap": ""}
    },
    skeleton_info={
        0:  {"link": ("tou1", "zhonbu"), "id": 0, "color": [255, 0, 0]},
        1:  {"link": ("zhonbu", "wei1"), "id": 1, "color": [0, 255, 0]}
    },
    joint_weights=[1.0] * 3,
    sigmas=[0.025] * 3,
    paper_info={'author': 'Cheng-long ZHANG', 'title': 'AFMS Keypoint Detection', 'container': 'SDAU university', 'year': '2025', 'homepage': 'www.sdau.edu.cn'},
)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(type='PoseDataPreprocessor',
                           mean=[123.675, 116.28, 103.53],
                           std=[58.395, 57.12, 57.375], bgr_to_rgb=True),
    backbone=dict(_scope_='mmdet', type='CSPNeXt', arch='P5',
                  expand_ratio=0.5, deepen_factor=0.33, widen_factor=0.5,
                  out_indices=(4,), channel_attention=True,
                  norm_cfg=dict(type='SyncBN'), act_cfg=dict(type='SiLU'),
                  init_cfg=dict(type='Pretrained', prefix='backbone.',
                                checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e-ea671761.pth')),
    head=dict(type='RTMCCHead', in_channels=512, out_channels=3,
              input_size=(256, 256), in_featuremap_size=(8, 8),
              simcc_split_ratio=2.0, final_layer_kernel_size=7,
              gau_cfg=dict(hidden_dims=256, s=128, expansion_factor=2,
                           dropout_rate=0., drop_path=0., act_fn='SiLU',
                           use_rel_bias=False, pos_enc=False),
              loss=dict(type='KLDiscretLoss', use_target_weight=True,
                        beta=10., label_softmax=True),
              decoder=codec),
    test_cfg=dict(flip_test=False))

backend_args = dict(backend='local')
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    #dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomBBoxTransform', scale_factor=[0.8, 1.2], rotate_factor=30),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='Albumentation',
         transforms=[
             dict(type='ChannelShuffle', p=0.5), dict(type='CLAHE', p=0.5),
             dict(type='ColorJitter', p=0.5),
             dict(type='CoarseDropout', max_holes=4, max_height=0.3,
                  max_width=0.3, min_holes=1, min_height=0.2, min_width=0.2, p=0.5)]),
    dict(type='GenerateTarget', encoder=codec), dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)), dict(type='PackPoseInputs')
]
train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    #dict(type='RandomFlip', direction='horizontal'),
    #dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform', shift_factor=0.,
         scale_factor=[0.75, 1.25], rotate_factor=60),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='Albumentation',
         transforms=[
             dict(type='Blur', p=0.1),
             dict(type='MedianBlur', p=0.1),
             dict(type='CoarseDropout', max_holes=1, max_height=0.4,
                  max_width=0.4, min_holes=1, min_height=0.2, min_width=0.2, p=0.5)]),
    dict(type='GenerateTarget', encoder=codec), dict(type='PackPoseInputs')
]

train_dataloader = dict(
    batch_size=8, num_workers=8, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='CocoDataset', data_root='datasets/crayfish/',
                metainfo=metainfo,
                data_mode='topdown', ann_file='train_coco.json',
                data_prefix=dict(img='train/'), pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8, num_workers=8, persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(type='CocoDataset', data_root='datasets/crayfish/',
                metainfo=metainfo,
                data_mode='topdown', ann_file='val_coco.json',
                data_prefix=dict(img='val/'), pipeline=val_pipeline))

test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3,
                    save_best='PCK', rule='greater'),
    logger=dict(interval=1))
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         update_buffers=True, priority=49),
    dict(
        type='hooks.hooks.ConditionalPipelineSwitchHook',
        monitor='PCK',
        rule='greater',
        patience=15,
        min_delta=0.001,
        switch_pipeline=train_pipeline_stage2,
        strict=True
    ),
    dict(type='EarlyStoppingHook', monitor='PCK',
         rule='greater', patience=20, min_delta=0.001),
]

val_evaluator = [
    dict(type='CocoMetric', ann_file='datasets/crayfish/val_coco.json'),
    dict(type='PCKAccuracy', prefix=''),
    dict(type='AUC', prefix=''),
    dict(type='NME', prefix='', norm_mode='keypoint_distance', keypoint_indices=[0, 1])
]

test_evaluator = val_evaluator
visualizer = dict(vis_backends=[dict(type='LocalVisBackend')])
