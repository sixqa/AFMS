#!/usr/bin/env python3
"""
AFMS 配置生成器
================
根据 YAML 文件生成 det/*.py / kp/*.py 训练配置。

用法：
    python tools/generate.py configs/carp.yaml       # 指定 YAML 文件
    python tools/generate.py configs/carp.yaml crab.yaml  # 多个
"""

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("需要 PyYAML: pip install pyyaml")
    sys.exit(1)


# ── 颜色板 ──
COLOR_PALETTE = [
    [255, 0, 0],     [0, 255, 0],     [0, 0, 255],     [255, 255, 0],
    [255, 0, 255],   [0, 255, 255],   [255, 128, 0],   [128, 0, 255],
    [0, 128, 255],   [255, 0, 128],   [128, 255, 0],   [0, 255, 128],
    [192, 192, 192], [128, 128, 0],   [128, 0, 128],   [0, 128, 128],
]

SKELETON_COLORS = [
    [255, 0, 0],     [0, 255, 0],     [0, 0, 255],     [255, 255, 0],
    [255, 0, 255],   [0, 255, 255],   [255, 150, 100], [150, 255, 100],
    [100, 100, 150], [150, 100, 100], [150, 100, 200], [100, 255, 150],
    [150, 200, 100], [255, 215, 0],   [0, 255, 255],   [255, 0, 0],
    [128, 0, 128],   [0, 128, 128],   [128, 128, 0],   [192, 192, 192],
    [255, 255, 255], [0, 0, 0],
]


def _build_keypoint_info(kp_names: list[str], flip_map: list = None) -> str:
    lines = []
    for i, name in enumerate(kp_names):
        c = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        if flip_map is not None and i < len(flip_map) and flip_map[i] is not None:
            swap_name = kp_names[flip_map[i]]
        else:
            swap_name = ""
        lines.append(
            f'        {i}:  {{"name": "{name}", "id": {i}, '
            f'"color": {c}, "type": "", "swap": "{swap_name}"}}'
        )
    return ",\n".join(lines)


def _build_skeleton_info(skeleton: list[list[str]]) -> str:
    lines = []
    for i, (a, b) in enumerate(skeleton):
        c = SKELETON_COLORS[i % len(SKELETON_COLORS)]
        lines.append(
            f'        {i}:  {{"link": ("{a}", "{b}"), "id": {i}, '
            f'"color": {c}}}'
        )
    return ",\n".join(lines)


def _build_paper_info() -> dict:
    return {
        'author': 'Cheng-long ZHANG',
        'title': 'AFMS Keypoint Detection',
        'container': 'SDAU university',
        'year': '2025',
        'homepage': 'www.sdau.edu.cn',
    }


# ── 默认模型参数（可通过 YAML model: 块覆盖） ──
DET_DEFAULTS = {
    "deepen_factor": 0.167, "widen_factor": 0.375,
    "in_channels": [96, 192, 384], "neck_out_channels": 96,
    "num_csp_blocks": 1, "exp_on_reg": False,
    "batch_size": 8, "base_lr": 4e-3, "num_workers": 8,
    "val_interval": 1, "early_stop_patience": 20, "stage2_patience": 15, "min_delta": 0.001,
    "backbone_pretrain": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth",
}

KP_DEFAULTS = {
    "backbone_scale": "s",  # s / m / l
    "input_size": [256, 256], "simcc_split_ratio": 2.0,
    "batch_size": 8, "base_lr": 4e-3, "num_workers": 8,
    "val_interval": 1, "early_stop_patience": 20, "stage2_patience": 15, "min_delta": 0.001,
    "backbone_pretrain": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e-ea671761.pth",
}

DET_BACKBONE_SCALES = {
    "tiny": {"deepen_factor": 0.167, "widen_factor": 0.375, "in_channels": [96, 192, 384], "neck_out_channels": 96, "pretrain": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth"},
    "s":    {"deepen_factor": 0.33,  "widen_factor": 0.5,   "in_channels": [128, 256, 512], "neck_out_channels": 128, "pretrain": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e-ea671761.pth"},
    "m":    {"deepen_factor": 0.67,  "widen_factor": 0.75,  "in_channels": [192, 384, 768], "neck_out_channels": 192, "pretrain": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb32-ms-coco/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230907_183225-5b5ca5f6.pth"},
    "l":    {"deepen_factor": 1.0,   "widen_factor": 1.0,   "in_channels": [256, 512, 1024], "neck_out_channels": 256, "pretrain": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth"},
}

KP_BACKBONE_SCALES = {
    "s": {"deepen_factor": 0.33, "widen_factor": 0.5},
    "m": {"deepen_factor": 0.67, "widen_factor": 0.75},
    "l": {"deepen_factor": 1.0,  "widen_factor": 1.0},
}


def merge_model_params(cfg: dict) -> tuple[dict, dict]:
    """从 YAML 中提取 model 覆盖，与默认值合并"""
    model_cfg = cfg.get("model", {})
    det = dict(DET_DEFAULTS)
    kp = dict(KP_DEFAULTS)
    # 根据默认 backbone_scale 填充 deepen_factor/widen_factor
    bs = kp.get("backbone_scale", "s")
    if bs in KP_BACKBONE_SCALES:
        for k, v in KP_BACKBONE_SCALES[bs].items():
            if k not in kp:
                kp[k] = v

    # 检测模型覆盖
    md = model_cfg.get("det", {})
    if isinstance(md.get("backbone"), str):
        scale = md["backbone"].lower()
        if scale in DET_BACKBONE_SCALES:
            det.update(DET_BACKBONE_SCALES[scale])
    for k in ["deepen_factor", "widen_factor", "in_channels", "neck_out_channels",
              "num_csp_blocks", "exp_on_reg", "batch_size", "base_lr",
              "num_workers", "val_interval", "early_stop_patience", "stage2_patience", "min_delta"]:
        if k in md:
            det[k] = md[k]

    # 关键点模型覆盖
    mk = model_cfg.get("kp", {})
    if isinstance(mk.get("backbone"), str):
        scale = mk["backbone"].lower()
        if scale in KP_BACKBONE_SCALES:
            kp.update(KP_BACKBONE_SCALES[scale])
    for k in ["backbone_scale", "input_size", "simcc_split_ratio",
              "batch_size", "base_lr", "num_workers", "early_stop_patience", "stage2_patience", "min_delta"]:
        if k in mk:
            kp[k] = mk[k]

    return det, kp


# ── 检测配置模板（RTMDet-Tiny） ──
DET_TEMPLATE = r"""default_scope = 'mmdet'

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
work_dir = 'runs/det_{SPL}'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={DET_EPOCHS},
                 val_interval=1,
                 dynamic_intervals=[({DET_EPOCHS_MINUS_20}, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=0.0002, begin=150,
         end=300, T_max=150, by_epoch=True, convert_to_iter_based=True)
]
optim_wrapper = dict(type='OptimWrapper',
                      optimizer=dict(type='AdamW', lr={DET_LR}, weight_decay=0.05),
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
    classes=('{SP}',),
    dataset_name='{SPL}_KEYPOINTS_keypoint',
    keypoint_info={
{KP_INFO}
    },
    skeleton_info={
{SK_INFO}
    },
    joint_weights=[1.0] * {N_KP},
    sigmas=[0.025] * {N_KP},
    paper_info={PAPER_INFO},
)

train_dataloader = dict(
    batch_size={DET_BS}, num_workers={DET_NW}, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='CocoDataset', data_root='{DS}', metainfo=metainfo,
                 ann_file='train_coco.json', data_prefix=dict(img='train/'),
                 filter_cfg=dict(filter_empty_gt=True, min_size=32),
                 pipeline=train_pipeline), pin_memory=True)

val_dataloader = dict(
    batch_size={DET_BS}, num_workers={DET_NW}, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', data_root='{DS}', metainfo=metainfo,
                 ann_file='val_coco.json', data_prefix=dict(img='val/'),
                 test_mode=True, pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file='{DS}val_coco.json',
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
                  deepen_factor={DET_DF}, widen_factor={DET_WF},
                  channel_attention=True, norm_cfg=dict(type='SyncBN'),
                  act_cfg=dict(type='SiLU', inplace=True),
                  init_cfg=dict(type='Pretrained', prefix='backbone.',
                                checkpoint='{DET_PRETRAIN}')),
    neck=dict(type='CSPNeXtPAFPN', in_channels={DET_IN_CH},
              out_channels={DET_OUT_CH}, num_csp_blocks={DET_NUM_CSP},
              expand_ratio=0.5, norm_cfg=dict(type='SyncBN'),
              act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(type='RTMDetSepBNHead', num_classes=1,
                   in_channels={DET_OUT_CH}, stacked_convs=2, feat_channels={DET_OUT_CH},
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
        patience={DET_STAGE2_PATIENCE},
        min_delta={DET_MIN_DELTA},
        switch_pipeline=train_pipeline_stage2,
        strict=True
    ),
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        rule='greater',
        patience={DET_PATIENCE},
        min_delta={DET_MIN_DELTA},
        strict=True
    ),
]
"""


# ── 关键点配置模板（RTMPose-S + SimCC） ──
KP_TEMPLATE = r"""_base_ = ['mmpose::_base_/default_runtime.py']

codec = dict(type='SimCCLabel', input_size={KP_IS}, sigma=(12, 12),
             simcc_split_ratio={KP_SCR}, normalize=False, use_dark=False)

work_dir = 'runs/kp_{SPL}'

stage2_num_epochs = 30
train_cfg = dict(max_epochs={KP_EPOCHS}, val_interval=1)

optim_wrapper = dict(type='OptimWrapper',
                      optimizer=dict(type='AdamW', lr={KP_LR}, weight_decay=0.05),
                      paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0,
                                          bypass_duplicate=True))

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=4e-3*0.05,
         begin={KP_EPOCHS}//2, end={KP_EPOCHS}, T_max={KP_EPOCHS}//2,
         by_epoch=True, convert_to_iter_based=True),
]

auto_scale_lr = dict(base_batch_size=1024)

metainfo = dict(
    classes=('{SP}',),
    dataset_name='{SPL}_KEYPOINTS_keypoint',
    keypoint_info={
{KP_INFO}
    },
    skeleton_info={
{SK_INFO}
    },
    joint_weights=[1.0] * {N_KP},
    sigmas=[0.025] * {N_KP},
    paper_info={PAPER_INFO},
)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(type='PoseDataPreprocessor',
                           mean=[123.675, 116.28, 103.53],
                           std=[58.395, 57.12, 57.375], bgr_to_rgb=True),
    backbone=dict(_scope_='mmdet', type='CSPNeXt', arch='P5',
                  expand_ratio=0.5, deepen_factor={KP_DF}, widen_factor={KP_WF},
                  out_indices=(4,), channel_attention=True,
                  norm_cfg=dict(type='SyncBN'), act_cfg=dict(type='SiLU'),
                  init_cfg=dict(type='Pretrained', prefix='backbone.',
                                checkpoint='{KP_PRETRAIN}')),
    head=dict(type='RTMCCHead', in_channels=512, out_channels={N_KP},
              input_size={KP_IS}, in_featuremap_size=({KP_IS0_DIV_32}, {KP_IS1_DIV_32}),
              simcc_split_ratio={KP_SCR}, final_layer_kernel_size=7,
              gau_cfg=dict(hidden_dims=256, s=128, expansion_factor=2,
                           dropout_rate=0., drop_path=0., act_fn='SiLU',
                           use_rel_bias=False, pos_enc=False),
              loss=dict(type='KLDiscretLoss', use_target_weight=True,
                        beta=10., label_softmax=True),
              decoder=codec),
    test_cfg=dict(flip_test={FLIP_TEST}))

backend_args = dict(backend='local')
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    {RANDOM_FLIP}dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomBBoxTransform', scale_factor=[0.8, 1.2], rotate_factor=30),
    dict(type='TopdownAffine', input_size={KP_IS}),
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
    dict(type='TopdownAffine', input_size={KP_IS}), dict(type='PackPoseInputs')
]
train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    {RANDOM_FLIP}dict(type='RandomFlip', direction='horizontal'),
    {RANDOM_FLIP}dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform', shift_factor=0.,
         scale_factor=[0.75, 1.25], rotate_factor=60),
    dict(type='TopdownAffine', input_size={KP_IS}),
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
    batch_size={KP_BS}, num_workers={KP_NW}, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='CocoDataset', data_root='{DS}',
                metainfo=metainfo,
                data_mode='topdown', ann_file='train_coco.json',
                data_prefix=dict(img='train/'), pipeline=train_pipeline))

val_dataloader = dict(
    batch_size={KP_BS}, num_workers={KP_NW}, persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(type='CocoDataset', data_root='{DS}',
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
        patience={KP_STAGE2_PATIENCE},
        min_delta={KP_MIN_DELTA},
        switch_pipeline=train_pipeline_stage2,
        strict=True
    ),
    dict(type='EarlyStoppingHook', monitor='PCK',
         rule='greater', patience={KP_PATIENCE}, min_delta={KP_MIN_DELTA}),
]

val_evaluator = [
    dict(type='CocoMetric', ann_file='{DS}val_coco.json'),
    dict(type='PCKAccuracy', prefix=''),
    dict(type='AUC', prefix=''),
    dict(type='NME', prefix='', norm_mode='keypoint_distance', keypoint_indices=[0, 1])
]

test_evaluator = val_evaluator
visualizer = dict(vis_backends=[dict(type='LocalVisBackend')])
"""


def generate(cfg: dict) -> tuple[str, str]:
    sp = cfg["species"]["name"].upper()
    spl = sp.lower()
    ds = cfg["dataset"]["root"].rstrip("/") + "/"
    dt = cfg["training"]

    kp_names = cfg["keypoints"]
    skeleton = cfg.get("skeleton", [])
    n_kp = len(kp_names)
    flip_map = cfg.get("flip_map", None)
    use_flip = cfg.get("use_flip", flip_map is not None)

    kp_info = _build_keypoint_info(kp_names, flip_map)
    sk_info = _build_skeleton_info(skeleton)
    paper_info = _build_paper_info()

    det_params, kp_params = merge_model_params(cfg)
    det_params_kp = det_params  # rename for readabiity
    kp_is = tuple(kp_params["input_size"])

    def fill(template, **extra):
        ctx = {
            "SP": sp, "SPL": spl, "DS": ds,
            "N_KP": n_kp,
            "PAPER_INFO": paper_info,
            "DET_EPOCHS": dt["det_epochs"],
            "DET_EPOCHS_MINUS_20": dt["det_epochs"] - 20,
            "DET_PATIENCE": det_params["early_stop_patience"],
            "DET_STAGE2_PATIENCE": det_params["stage2_patience"],
            "DET_MIN_DELTA": det_params["min_delta"],
            "KP_EPOCHS": dt["kp_epochs"],
            "KP_EPOCHS_MINUS_STAGE2": dt["kp_epochs"] - 30,
            "KP_PATIENCE": kp_params["early_stop_patience"],
            "KP_STAGE2_PATIENCE": kp_params["stage2_patience"],
            "KP_MIN_DELTA": kp_params["min_delta"],
            "KP_INFO": kp_info,
            "SK_INFO": sk_info,
            # 翻转相关参数
            "FLIP_TEST": "True" if use_flip else "False",
            "RANDOM_FLIP": "" if use_flip else "#",
            # 检测模型参数
            "DET_DF": det_params["deepen_factor"],
            "DET_WF": det_params["widen_factor"],
            "DET_IN_CH": det_params["in_channels"],
            "DET_OUT_CH": det_params["neck_out_channels"],
            "DET_NUM_CSP": det_params["num_csp_blocks"],
            "DET_BS": det_params["batch_size"],
            "DET_LR": det_params["base_lr"],
            "DET_NW": det_params["num_workers"],
            "DET_PRETRAIN": det_params.get("pretrain", det_params.get("backbone_pretrain", DET_DEFAULTS["backbone_pretrain"])),
            # 关键点模型参数
            "KP_DF": kp_params["deepen_factor"],
            "KP_WF": kp_params["widen_factor"],
            "KP_IS": kp_is,
            "KP_IS0": kp_is[0], "KP_IS1": kp_is[1],
            "KP_IS0_DIV_32": kp_is[0] // 32, "KP_IS1_DIV_32": kp_is[1] // 32,
            "KP_SCR": kp_params["simcc_split_ratio"],
            "KP_BS": kp_params["batch_size"],
            "KP_LR": kp_params["base_lr"],
            "KP_NW": kp_params["num_workers"],
            "KP_PRETRAIN": kp_params.get("pretrain", kp_params.get("backbone_pretrain", KP_DEFAULTS["backbone_pretrain"])),
        }
        ctx.update(extra)
        result = template
        for k, v in ctx.items():
            result = result.replace("{" + k + "}", str(v))
        return result

    det_cfg = fill(DET_TEMPLATE)
    kp_cfg = fill(KP_TEMPLATE)

    return det_cfg, kp_cfg


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0)

    out_det = Path(__file__).parent.parent / "configs" / "det"
    out_kp = Path(__file__).parent.parent / "configs" / "kp"
    out_det.mkdir(parents=True, exist_ok=True)
    out_kp.mkdir(parents=True, exist_ok=True)

    generated = 0
    for arg in sys.argv[1:]:
        yaml_path = Path(arg)
        if not yaml_path.exists():
            print(f"  [跳过] {yaml_path} 不存在")
            continue

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        sp = cfg["species"]["name"].upper()
        spl = sp.lower()

        print(f"  [{sp}]", end=" ")

        try:
            det_cfg, kp_cfg = generate(cfg)
            (out_det / f"{spl}.py").write_text(det_cfg)
            (out_kp / f"{spl}.py").write_text(kp_cfg)
            print(f"det/{spl}.py  kp/{spl}.py  OK")
            generated += 1
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n完成！已从 {generated} 个 YAML 生成配置。")
    if generated > 0:
        print(f"  训练: python tools/train_det.py --species <物种名>")
        print(f"         python tools/train_kp.py --species <物种名>")


if __name__ == "__main__":
    main()
