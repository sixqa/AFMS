default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3,
                    save_best='coco/bbox_mAP', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(cudnn_benchmark=False,
               mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
               dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]

visualizer = dict(type='mmdet.DetLocalVisualizer',
                  vis_backends=[dict(type='LocalVisBackend')],
                  name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
work_dir = 'runs/det_carp'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=500,
                 val_interval=1,
                 dynamic_intervals=[(480, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=0.0002, begin=150,
         end=300, T_max=150, by_epoch=True, convert_to_iter_based=True)
]
optim_wrapper = dict(type='OptimWrapper',
                      optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
                      paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0,
                                          bypass_duplicate=True))

auto_scale_lr = dict(base_batch_size=16)

train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0,
         max_cached_images=20, random_pop=False),
    dict(type='RandomResize', scale=(1280, 1280), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'), dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='CachedMixUp', img_scale=(640, 640), ratio_range=(1.0, 1.0),
         max_cached_images=10, random_pop=False, pad_val=(114, 114, 114), prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape',
                                          'img_shape', 'scale_factor'))
]

metainfo = dict(
    classes=('CARP',),
    dataset_name='carp_KEYPOINTS_keypoint',
    keypoint_info={
        0:  {"name": "P1", "id": 0, "color": [255, 0, 0], "type": "", "swap": ""},
        1:  {"name": "P2", "id": 1, "color": [0, 255, 0], "type": "", "swap": ""},
        2:  {"name": "P3", "id": 2, "color": [0, 0, 255], "type": "", "swap": ""},
        3:  {"name": "P4", "id": 3, "color": [255, 255, 0], "type": "", "swap": ""},
        4:  {"name": "P5", "id": 4, "color": [255, 0, 255], "type": "", "swap": ""},
        5:  {"name": "P6", "id": 5, "color": [0, 255, 255], "type": "", "swap": ""},
        6:  {"name": "P7", "id": 6, "color": [255, 128, 0], "type": "", "swap": ""},
        7:  {"name": "P8", "id": 7, "color": [128, 0, 255], "type": "", "swap": ""},
        8:  {"name": "P9", "id": 8, "color": [0, 128, 255], "type": "", "swap": ""},
        9:  {"name": "P10", "id": 9, "color": [255, 0, 128], "type": "", "swap": ""},
        10:  {"name": "P11", "id": 10, "color": [128, 255, 0], "type": "", "swap": ""},
        11:  {"name": "P12", "id": 11, "color": [0, 255, 128], "type": "", "swap": ""}
    },
    skeleton_info={
        0:  {"link": ("P1", "P2"), "id": 0, "color": [255, 0, 0]},
        1:  {"link": ("P2", "P3"), "id": 1, "color": [0, 255, 0]},
        2:  {"link": ("P3", "P4"), "id": 2, "color": [0, 0, 255]},
        3:  {"link": ("P4", "P5"), "id": 3, "color": [255, 255, 0]},
        4:  {"link": ("P5", "P6"), "id": 4, "color": [255, 0, 255]},
        5:  {"link": ("P6", "P7"), "id": 5, "color": [0, 255, 255]},
        6:  {"link": ("P7", "P8"), "id": 6, "color": [255, 150, 100]},
        7:  {"link": ("P8", "P9"), "id": 7, "color": [150, 255, 100]},
        8:  {"link": ("P9", "P10"), "id": 8, "color": [100, 100, 150]},
        9:  {"link": ("P10", "P11"), "id": 9, "color": [150, 100, 100]},
        10:  {"link": ("P11", "P12"), "id": 10, "color": [150, 100, 200]},
        11:  {"link": ("P12", "P1"), "id": 11, "color": [100, 255, 150]},
        12:  {"link": ("P2", "P12"), "id": 12, "color": [150, 200, 100]},
        13:  {"link": ("P3", "P11"), "id": 13, "color": [255, 215, 0]},
        14:  {"link": ("P3", "P10"), "id": 14, "color": [0, 255, 255]},
        15:  {"link": ("P5", "P7"), "id": 15, "color": [255, 0, 0]}
    },
    joint_weights=[1.0] * 12,
    sigmas=[0.025] * 12,
    paper_info={'author': 'Cheng-long ZHANG', 'title': 'AFMS Keypoint Detection', 'container': 'SDAU university', 'year': '2025', 'homepage': 'www.sdau.edu.cn'},
)

train_dataloader = dict(
    batch_size=8, num_workers=8, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='CocoDataset', data_root='datasets/carp/', metainfo=metainfo,
                 ann_file='train_coco.json', data_prefix=dict(img='train/'),
                 filter_cfg=dict(filter_empty_gt=True, min_size=32),
                 pipeline=train_pipeline), pin_memory=True)

val_dataloader = dict(
    batch_size=8, num_workers=8, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', data_root='datasets/carp/', metainfo=metainfo,
                 ann_file='val_coco.json', data_prefix=dict(img='val/'),
                 test_mode=True, pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file='datasets/carp/val_coco.json',
                      metric=['bbox'], format_only=False)

test_evaluator = val_evaluator

tta_model = dict(type='DetTTAModel',
                 tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale': (640, 640),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (320, 320),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (960, 960),
            'keep_ratio': True
        }],
                    [{
                        'type': 'RandomFlip',
                        'prob': 1.0
                    }, {
                        'type': 'RandomFlip',
                        'prob': 0.0
                    }],
                    [{
                        'type': 'Pad',
                        'size': (960, 960),
                        'pad_val': {
                            'img': (114, 114, 114)
                        }
                    }],
                    [{
                        'type':
                        'PackDetInputs',
                        'meta_keys':
                        ('img_id', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'flip', 'flip_direction')
                    }]])
]

model = dict(
    type='RTMDet',
    data_preprocessor=dict(type='DetDataPreprocessor',
                           mean=[103.53, 116.28, 123.675],
                           std=[57.375, 57.12, 58.395],
                           bgr_to_rgb=False),
    backbone=dict(type='CSPNeXt', arch='P5', expand_ratio=0.5,
                  deepen_factor=0.167, widen_factor=0.375,
                  channel_attention=True, norm_cfg=dict(type='SyncBN'),
                  act_cfg=dict(type='SiLU', inplace=True),
                  init_cfg=dict(type='Pretrained', prefix='backbone.',
                                checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth')),
    neck=dict(type='CSPNeXtPAFPN', in_channels=[96, 192, 384],
              out_channels=96, num_csp_blocks=1,
              expand_ratio=0.5, norm_cfg=dict(type='SyncBN'),
              act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(type='RTMDetSepBNHead', num_classes=1,
                   in_channels=96, stacked_convs=2, feat_channels=96,
                   anchor_generator=dict(type='MlvlPointGenerator', offset=0,
                                         strides=[8, 16, 32]),
                   bbox_coder=dict(type='DistancePointBBoxCoder'),
                   loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True, beta=2.0,
                                 loss_weight=1.0),
                   loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                   with_objectness=False, exp_on_reg=False,
                   share_conv=True, pred_kernel_size=1,
                   norm_cfg=dict(type='SyncBN'), act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
                   allowed_border=-1, pos_weight=-1, debug=False),
    test_cfg=dict(nms_pre=30000, min_bbox_size=0, score_thr=0.001,
                  nms=dict(type='nms', iou_threshold=0.65), max_per_img=300))

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=(640, 640), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)), dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49
    ),
    dict(
        type='hooks.hooks.ConditionalPipelineSwitchHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        patience=15,
        min_delta=0.001,
        switch_pipeline=train_pipeline_stage2,
        strict=True
    ),
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        patience=20,
        min_delta=0.001,
        strict=True
    ),
]
