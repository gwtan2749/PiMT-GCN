_base_ = ['./dgstgcn_vector.py']

custom_imports = dict(imports=['mmaction.datasets.transforms.custom_transforms', 'custom_multi_task'], allow_failed_imports=False)
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

train_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['j']), 
    # 🌟 Angle 专属：mode 改成 'angle'，确保 sin(Channel 1) 正确翻转
    dict(type='VectorRandomFlip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp, mode='angle'),
    
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackMultiTaskInputs')
]
train_dataloader = dict(dataset=dict(dataset=dict(pipeline=train_pipeline)))