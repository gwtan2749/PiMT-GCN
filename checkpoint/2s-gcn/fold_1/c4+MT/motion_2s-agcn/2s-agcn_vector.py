ann_file = '/home/ivi/tgw/videos/pickleball/pickleball_dataset_rtmpose_C4_RealDist_Motion.pkl'
auto_scale_lr = dict(base_batch_size=32, enable=True)
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmaction.datasets.transforms.custom_transforms',
        'custom_multi_task',
    ])
dataset_type = 'PoseDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
left_kp = [
    1,
    3,
    5,
    7,
    9,
    11,
    13,
    15,
]
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        graph_cfg=dict(layout='coco', mode='spatial'),
        in_channels=4,
        num_person=1,
        type='AAGCN'),
    cls_head=dict(
        dynamic_weight=False,
        fixed_aux_weight=0.5,
        gating_in_loss=False,
        in_channels=256,
        is_mlp_bounce=False,
        num_classes=9,
        pcgrad_projection='global',
        type='MultiTaskGCNHead',
        use_hard_gating=False,
        use_pcgrad=False),
    type='RecognizerGCN')
optim_wrapper = dict(
    optimizer=dict(
        lr=0.05, momentum=0.9, nesterov=True, type='SGD', weight_decay=0.0005))
param_scheduler = [
    dict(
        T_max=100,
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min=0,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=42)
resume = True
right_kp = [
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file=
        '/home/ivi/tgw/videos/pickleball/pickleball_dataset_rtmpose_C4_RealDist_Motion.pkl',
        pipeline=[
            dict(dataset='coco', feats=[
                'j',
            ], type='GenSkeFeat'),
            dict(
                clip_len=100,
                num_clips=1,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(num_person=1, type='FormatGCNInput'),
            dict(type='PackMultiTaskInputs'),
        ],
        split='val',
        test_mode=True,
        type='PoseDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='AccMetric'),
]
test_pipeline = [
    dict(dataset='coco', feats=[
        'j',
    ], type='GenSkeFeat'),
    dict(
        clip_len=100, num_clips=1, test_mode=True, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=1, type='FormatGCNInput'),
    dict(type='PackMultiTaskInputs'),
]
train_cfg = dict(
    max_epochs=100, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        dataset=dict(
            ann_file=
            '/home/ivi/tgw/videos/pickleball/pkl_datasets/6folds_cross_val_4/fold_1/dataset_C4_RealDist_Motion.pkl',
            pipeline=[
                dict(dataset='coco', feats=[
                    'j',
                ], type='GenSkeFeat'),
                dict(
                    flip_ratio=0.5,
                    left_kp=[
                        1,
                        3,
                        5,
                        7,
                        9,
                        11,
                        13,
                        15,
                    ],
                    mode='vector',
                    right_kp=[
                        2,
                        4,
                        6,
                        8,
                        10,
                        12,
                        14,
                        16,
                    ],
                    type='VectorRandomFlip'),
                dict(clip_len=100, type='UniformSampleFrames'),
                dict(type='PoseDecode'),
                dict(num_person=1, type='FormatGCNInput'),
                dict(type='PackMultiTaskInputs'),
            ],
            split='train',
            type='PoseDataset'),
        times=5,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(dataset='coco', feats=[
        'j',
    ], type='GenSkeFeat'),
    dict(
        flip_ratio=0.5,
        left_kp=[
            1,
            3,
            5,
            7,
            9,
            11,
            13,
            15,
        ],
        mode='vector',
        right_kp=[
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
        ],
        type='VectorRandomFlip'),
    dict(clip_len=100, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=1, type='FormatGCNInput'),
    dict(type='PackMultiTaskInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file=
        '/home/ivi/tgw/videos/pickleball/pkl_datasets/6folds_cross_val_4/fold_1/dataset_C4_RealDist_Motion.pkl',
        pipeline=[
            dict(dataset='coco', feats=[
                'j',
            ], type='GenSkeFeat'),
            dict(
                clip_len=100,
                num_clips=1,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(num_person=1, type='FormatGCNInput'),
            dict(type='PackMultiTaskInputs'),
        ],
        split='val',
        test_mode=True,
        type='PoseDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='AccMetric'),
]
val_pipeline = [
    dict(dataset='coco', feats=[
        'j',
    ], type='GenSkeFeat'),
    dict(
        clip_len=100, num_clips=1, test_mode=True, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=1, type='FormatGCNInput'),
    dict(type='PackMultiTaskInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'work_dirs/b_final_cv_2/expfinal_01_c4_mt_fixed_2s-agcn_fold_1/motion_2s-agcn'
