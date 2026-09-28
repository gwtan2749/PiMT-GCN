# PiMT-GCN

PiMT-GCN is a skeleton-based pickleball action recognition project built on [MMAction2](https://github.com/open-mmlab/mmaction2). It provides 2S-AGCN and DG-STGCN models with joint, bone, motion, and angle feature configurations. The `custom_multi_task.py` module implements the PiMT multi-task classification head.

MMAction2 handles dataset loading, training, validation, and testing. Given a `.pkl` dataset in MMAction2 `PoseDataset` format and a matching configuration, MMAction2 can run the complete workflow.

## Directory Structure

```text
PiMT-GCN/
├── PKB-9/data/fold_1/              # Fold-1 datasets
├── mmaction2/
│   ├── configs/skeleton/pickleball/ # PiMT configurations
│   ├── custom_multi_task.py         # Multi-task classification head
│   └── tools/train.py, tools/test.py
├── checkpoint/                      # Saved configurations and weights
└── README.md
```

Available files under `PKB-9/data/fold_1`:

```text
dataset_joint.pkl
dataset_joint_Motion.pkl
dataset_joint_Bone.pkl
dataset_joint_Angle.pkl
dataset_C4_RealDist.pkl
dataset_C4_RealDist_Motion.pkl
dataset_C4_RealDist_Bone.pkl
dataset_C4_RealDist_Angle.pkl
```

## Environment Setup

Install Python, PyTorch, and a compatible CUDA runtime when using a GPU. Install MMAction2 from source according to the official guide:

```bash
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
```

Alternatively, install the MMAction2 source included in this repository:

```bash
cd ./mmaction2
pip install -v -e .
```

## Copy PiMT Files into MMAction2

If a different MMAction2 source directory is used, run these commands from the PiMT-GCN repository root:

```bash
cp -r ./mmaction2/configs/skeleton/pickleball \
      ../mmaction2/configs/skeleton/
cp ./mmaction2/custom_multi_task.py ../mmaction2/
```

`custom_multi_task.py` must be placed at the MMAction2 project root because the configurations import it as `custom_multi_task`.

## Dataset and Configuration

The `ann_file` field must point to an existing `.pkl` file. Commands are run from the `mmaction2` directory, so a repository dataset can be referenced as:

```python
ann_file = '../PKB-9/data/fold_1/dataset_C4_RealDist_Motion.pkl'
```

Change the filename for another representation. The configurations use `split='train'` and `split='val'`; the dataset must contain the split information required by MMAction2 `PoseDataset`.

## Experiment Settings

The `--channels` and `--task` options from the ablation workflow correspond to the following settings:

| Setting | Dataset files | Input channels | Meaning |
| --- | --- | ---: | --- |
| `c3` | `dataset_joint*.pkl` | 3 | C3 single-task or multi-task experiment |
| `c4` | `dataset_C4_RealDist*.pkl` | 4 | C4 single-task experiment |
| `c4+MT` | `dataset_C4_RealDist*.pkl` | 4 | C4 with `MultiTaskGCNHead` |

`MT` means multi-task learning: action classification plus binary `is_bounce` prediction. The provided head uses `fixed_aux_weight=0.5`. Use `--task single` to set the auxiliary-task weight to zero, or `--task multi_fixed` with `--fixed-aux-weight` to select another fixed weight. Historical options filtered by `custom_multi_task.py` are not required for the main PiMT workflow.

| Modality | Configuration |
| --- | --- |
| Joint | `dgstgcn_joint.py` or `2s-agcn_joint.py` |
| Motion/Bone | `dgstgcn_vector.py` or `2s-agcn_vector.py` |
| Angle | `dgstgcn_angle.py` or `2s-agcn_angle.py` |

For example, DG-STGCN `c4+MT` Joint uses `dataset_C4_RealDist.pkl`, `dgstgcn_joint.py`, and `model.backbone.in_channels=4`. Its Motion variant uses `dataset_C4_RealDist_Motion.pkl` and `dgstgcn_vector.py`.

## Training

Run commands from the MMAction2 root:

```bash
cd ./mmaction2
python tools/train.py configs/skeleton/pickleball/dgstgcn_joint.py \
  --work-dir work_dirs/pimt_dgstgcn_joint
```

Other configurations:

```bash
python tools/train.py configs/skeleton/pickleball/dgstgcn_vector.py \
  --work-dir work_dirs/pimt_dgstgcn_vector
python tools/train.py configs/skeleton/pickleball/2s-agcn_joint.py \
  --work-dir work_dirs/pimt_2sagcn_joint
```

The default training schedule is `max_epochs=100`. Resume training with:

```bash
python tools/train.py configs/skeleton/pickleball/dgstgcn_joint.py \
  --work-dir work_dirs/pimt_dgstgcn_joint --resume
```

Multi-GPU execution follows MMAction2 `tools/dist_train.sh`. Configuration values can be overridden with `--cfg-options`.

### c3 Training

```bash
bash tools/dist_train.sh configs/skeleton/pickleball/dgstgcn_joint.py 2 \
  --work-dir work_dirs/pimt_dgstgcn_c3_joint \
  --cfg-options \
  train_dataloader.dataset.dataset.ann_file=../PKB-9/data/fold_1/dataset_joint.pkl \
  val_dataloader.dataset.ann_file=../PKB-9/data/fold_1/dataset_joint.pkl \
  model.backbone.in_channels=3 \
  model.cls_head.dynamic_weight=False \
  model.cls_head.fixed_aux_weight=0.0
```

### c4+MT Training

```bash
bash tools/dist_train.sh configs/skeleton/pickleball/dgstgcn_joint.py 2 \
  --work-dir work_dirs/pimt_dgstgcn_c4_mt_joint \
  --cfg-options \
  train_dataloader.dataset.dataset.ann_file=../PKB-9/data/fold_1/dataset_C4_RealDist.pkl \
  val_dataloader.dataset.ann_file=../PKB-9/data/fold_1/dataset_C4_RealDist.pkl \
  model.backbone.in_channels=4 \
  model.cls_head.dynamic_weight=False \
  model.cls_head.fixed_aux_weight=0.5
```

For Motion, Bone, or Angle, replace both the configuration and dataset file.

## Testing

The standard MMAction2 test command takes a configuration and a checkpoint:

```bash
cd ./mmaction2
python tools/test.py \
  configs/skeleton/pickleball/dgstgcn_joint.py \
  ../checkpoint/dgstgcn/fold_1/c4+MT/joint/best_acc_top1_epoch_92.pth
```

To save metrics and predictions:

```bash
python tools/test.py \
  configs/skeleton/pickleball/dgstgcn_joint.py \
  ../checkpoint/dgstgcn/fold_1/c4+MT/joint/best_acc_top1_epoch_92.pth \
  --work-dir work_dirs/test_dgstgcn_joint \
  --dump work_dirs/test_dgstgcn_joint/predictions.pkl
```

Distributed `c4+MT` testing:

```bash
bash tools/dist_test.sh configs/skeleton/pickleball/dgstgcn_joint.py \
  ../checkpoint/dgstgcn/fold_1/c4+MT/joint/best_acc_top1_epoch_92.pth 2 \
  --cfg-options \
  test_dataloader.dataset.ann_file=../PKB-9/data/fold_1/dataset_C4_RealDist.pkl \
  model.backbone.in_channels=4
```

For c3, use the matching `dataset_joint*.pkl` file and set `model.backbone.in_channels=3`. The configuration, dataset representation, channel count, and checkpoint must match.

## Checkpoints

Checkpoints are organized by model, fold, feature setting, and task setting:

```text
checkpoint/dgstgcn/fold_1/c4+MT/joint/
├── dgstgcn_joint.py
└── best_acc_top1_epoch_92.pth
```

`best_acc_top1_epoch_*.pth` is the best validation Top-1 checkpoint. The adjacent `.py` file is a configuration backup.

## Troubleshooting

1. **Cannot import `custom_multi_task`**: place `custom_multi_task.py` at the MMAction2 project root and run commands from that directory.
2. **Dataset not found**: check `ann_file` and verify the file exists under `../PKB-9/data/fold_1`.
3. **Checkpoint mismatch**: verify the model, feature type, class count, input channels, and multi-task setting.

## Submission Note

`mmaction2/train_cv_ablation.sh` and `mmaction2/test_cv_ablation.sh` are local reference scripts for cross-validation and ablation experiments. They are excluded from the project submission and should not be committed.

## References

- [MMAction2 repository](https://github.com/open-mmlab/mmaction2)
- [MMAction2 installation guide](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html)
- [MMAction2 fine-tuning and testing guide](https://mmaction2.readthedocs.io/en/latest/user_guides/finetune.html)
