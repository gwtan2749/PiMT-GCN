# 继承原来的 joint 基础配置
_base_ = ['./dgstgcn_joint.py']

custom_imports = dict(imports=['mmaction.datasets.transforms.custom_transforms', 'custom_multi_task'], allow_failed_imports=False)

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

# 🌟 缝合版训练流水线：结合缩放与自定义翻转，同时剔除有害的 Shift 破坏！
train_pipeline = [
    # ❌ 依然坚决不加 PreNormalize2D，保护原生态向量！
        # （注：虽然神经网络的拟合能力可以“死记硬背”克服固定的平移偏差，
        #  但 Fold-1 94.08% 的成绩证明：卸下解密负担，喂给它原汁原味的无偏差向量，单体性能才能彻底破壁！）
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    
    # 🌟 使用我们手写的 VectorRandomFlip，完美解决向量和角度的翻转数学逻辑
    dict(type='VectorRandomFlip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp, mode='vector'),
    
    # ❌ 坚决没有 RandomShift 和 RandomRot2D (保护物理先验)
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackMultiTaskInputs')
]

val_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackMultiTaskInputs')
]
test_pipeline=val_pipeline
train_dataloader = dict(dataset=dict(dataset=dict(pipeline=train_pipeline)))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))