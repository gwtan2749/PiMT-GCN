import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.registry import MODELS, TRANSFORMS
from mmaction.models.heads.gcn_head import GCNHead
from mmengine.structures import LabelData

# ==========================================
# 1. 自定义数据流管道：抓取 is_bounce 弹地标签
# ==========================================
@TRANSFORMS.register_module()
class PackMultiTaskInputs(TRANSFORMS.get('PackActionInputs')):
    """打包包含主任务动作标签与弹地辅助标签的多任务输入."""
    
    def __call__(self, results):
        packed_results = super().__call__(results)
        data_sample = packed_results['data_samples']
        
        # 弹地二分类标签 (0/1, 缺失或未知样本设为 -1)
        bounce_val = results.get('is_bounce', -1)
        bounce_label = LabelData(item=torch.tensor([bounce_val], dtype=torch.long))
        data_sample.set_field(bounce_label, 'bounce_label')
        
        return packed_results

# ==========================================
# 2. 纯粹 C3 固定权重多任务分类头 (无门控 + 防崩溃容错)
# ==========================================
@MODELS.register_module()
class MultiTaskGCNHead(GCNHead):
    """
    C3 纯净固定权重多任务分类头：
    - 主任务：骨架动作识别分类 (Action Classification)
    - 辅助任务：球体是否弹地二分类 (Bounce State Prediction)
    - 联合损失：loss = loss_cls + fixed_aux_weight * loss_bounce
    """
    def __init__(self, 
                 num_classes, 
                 in_channels, 
                 fixed_aux_weight=0.5, 
                 is_mlp_bounce=False, 
                 **kwargs):
        
        # 🛡️ 核心防御：彻底清洗配置文件或历史权重中残留的门控/消融废弃参数
        # 防止传入底层 GCNHead/BaseHead 引发 "got an unexpected keyword argument" 崩溃
        deprecated_keys = [
            'use_hard_gating', 'gating_mode', 'gating_penalty', 'gating_in_loss', 'hard_gating_value',
            'dynamic_weight', 'dynamic_loss_type', 'use_pcgrad', 'pcgrad_projection',
            'log_var_init', 'max_precision_cls', 'min_precision_cls', 'max_precision_bounce',
            'min_precision_bounce', 'penalty_scale', 'caaw_penalty_scale', 'aux_ratio_limit',
            'is_use_temp', 'lr_mult', 'max_epoch', 'is_learnable_alpha'
        ]
        for key in deprecated_keys:
            kwargs.pop(key, None)
            
        super().__init__(num_classes, in_channels, **kwargs)
        
        # C3 固定辅助损失权重与网络架构配置
        self.fixed_aux_weight = float(fixed_aux_weight)
        self.is_mlp_bounce = is_mlp_bounce
        
        # 辅助弹地预测头分支 (线性层或轻量 MLP)
        if self.is_mlp_bounce:
            self.mlp_bounce = nn.Sequential(
                nn.Linear(in_channels, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 1)
            )
        else:
            self.fc_bounce = nn.Linear(in_channels, 1)

    def forward(self, x):
        """主干前向特征池化与双任务打分."""
        # 针对时空骨架图特征进行自适应全局平均池化 (2D ~ 5D 输入兼容)
        if x.dim() == 5:
            N, M, C, T, V = x.shape
            x = x.view(N * M, C, T, V)
            x = x.mean(-1).mean(-1)
            x = x.view(N, M, C).mean(dim=1)
        elif x.dim() == 4:
            x = x.mean(-1).mean(-1)
        elif x.dim() == 3:
            x = x.mean(dim=1)
        elif x.dim() == 2:
            pass  
        else:
            raise ValueError(f"输入特征维度异常，预期为 2D~5D，实际得到 {x.dim()}D")
            
        if self.dropout is not None:
            x = self.dropout(x)
            
        fc_cls = getattr(self, 'fc_cls', getattr(self, 'fc', None))
        if fc_cls is None:
            raise AttributeError("未在基类中找到分类全连接层。")
            
        # 主任务分类分值
        cls_score = fc_cls(x)
        
        # 辅助任务弹地分值 (Logits)
        if self.is_mlp_bounce:  
            bounce_score = self.mlp_bounce(x)
        else:
            bounce_score = self.fc_bounce(x)
        
        return cls_score, bounce_score

    def compute_raw_losses(self, cls_score, bounce_score, batch_data_samples):
        """计算主任务分类与弹地辅助任务的未加权原始损失."""
        # 1. 提取主动作分类标签
        labels_list = []
        for ds in batch_data_samples:
            label = None
            if hasattr(ds, 'gt_label') and hasattr(ds.gt_label, 'label'):
                label = ds.gt_label.label
            elif hasattr(ds, 'gt_labels') and hasattr(ds.gt_labels, 'item'):
                label = ds.gt_labels.item
            elif hasattr(ds, 'gt_label') and isinstance(ds.gt_label, torch.Tensor):
                label = ds.gt_label
            elif hasattr(ds, 'gt_label') and hasattr(ds.gt_label, 'item'):
                label = ds.gt_label.item
                
            if label is not None:
                labels_list.append(label.view(-1))
            else:
                labels_list.append(torch.tensor([0], device=cls_score.device))
                
        labels = torch.cat(labels_list).to(cls_score.device)
        loss_cls_raw = self.loss_cls(cls_score, labels)
        
        # 2. 提取弹地二分类标签并进行有效样本掩码过滤
        bounce_labels_list = []
        for ds in batch_data_samples:
            if hasattr(ds, 'bounce_label') and hasattr(ds.bounce_label, 'item'):
                bounce_labels_list.append(ds.bounce_label.item.view(-1))
            else:
                bounce_labels_list.append(torch.tensor([-1], device=cls_score.device))
                
        bounce_labels = torch.cat(bounce_labels_list).float().to(cls_score.device)
        valid_mask = bounce_labels >= 0
        
        if valid_mask.sum() > 0:
            loss_bounce_raw = F.binary_cross_entropy_with_logits(
                bounce_score[valid_mask].squeeze(-1),
                bounce_labels[valid_mask]
            )
        else:
            loss_bounce_raw = bounce_score.sum() * 0.0 
            
        return loss_cls_raw, loss_bounce_raw

    def loss(self, x, batch_data_samples, **kwargs):
        """计算多任务联合损失：主任务损失 + 固定权重辅助任务损失."""
        cls_score, bounce_score = self.forward(x)
        loss_cls_raw, loss_bounce_raw = self.compute_raw_losses(cls_score, bounce_score, batch_data_samples)
        
        losses = dict()
        losses['loss_cls'] = loss_cls_raw
        losses['loss_bounce'] = loss_bounce_raw * self.fixed_aux_weight
        return losses

    def predict(self, x, batch_data_samples, **kwargs):
        """测试/推理评估流程：纯主干输出，无需先验门控干预."""
        cls_score, _ = self.forward(x)
        return self.predict_by_feat(cls_score, batch_data_samples, **kwargs)