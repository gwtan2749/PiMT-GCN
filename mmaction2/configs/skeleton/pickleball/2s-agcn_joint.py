_base_ = '../../_base_/default_runtime.py'
# 告诉框架导入你刚才建的 custom_transforms.py 文件
custom_imports = dict(imports=['mmaction.datasets.transforms.custom_transforms', 'custom_multi_task'], allow_failed_imports=False)

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='AAGCN',
        in_channels=4,
        num_person=1,
        graph_cfg=dict(layout='coco', mode='spatial')),  # degenerate AAGCN to AGCN
    # cls_head=dict(type='GCNHead', num_classes=6, in_channels=256)
    cls_head=dict(type='MultiTaskGCNHead',
                  num_classes=9, 
                  in_channels=256,
                  fixed_aux_weight=0.5,      # 👉 新增：落地辅助任务的权重，0.5是个不错的起点)
                  use_hard_gating=False,
                  gating_in_loss=False,
                  dynamic_weight=False
    )
)

dataset_type = 'PoseDataset'

ann_file = '/home/ivi/tgw/videos/pickleball/pickleball_dataset_rtmpose_C4_RealDist_Motion.pkl'

# 定义镜像对称的关节点索引对 (COCO 17格式示例)
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    
    # 🌟 唯一开启的数据增强：左右翻转！完美兼容 COCO 17点，解决左撇子
    dict(type='SkeletonRandomFlip', prob=0.5, left_kp=left_kp,right_kp=right_kp),
    
    # ❌ 坚决删除 RandomShift 和 RandomRot2D，保护 C4！
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    # dict(type='PackActionInputs') # ❌ 注释掉官方的
    dict(type='PackMultiTaskInputs') # ✅ 替换成我们能抓取 is_bounce 的自定义包
]

val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    # dict(type='PackActionInputs')  # ❌ 注释掉官方的
    dict(type='PackMultiTaskInputs') # ✅ 替换成我们能抓取 is_bounce 的自定义包
]
test_pipeline = val_pipeline
 
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='train')))
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='val',
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = [dict(type='AccMetric')]

test_evaluator = val_evaluator
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=100, # 🌟 修正：把它改成和 max_epochs 一样大
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005, nesterov=True))

default_hooks = dict(checkpoint=dict(interval=1), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=32)

# 新增可视化配置，开启 Tensorboard 后端
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')  # 🚀 增加这一行！
    ]
)
