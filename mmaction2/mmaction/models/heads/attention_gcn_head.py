import torch
import torch.nn as nn
from mmaction.models.heads import GCNHead
from mmaction.registry import MODELS


class TemporalAttention(nn.Module):
    """
    针对骨架动作识别的时序注意力机制 (Temporal Attention)
    用于自动寻找“击球决定性瞬间”并赋予高权重。
    """
    def __init__(self, in_channels, reduction=4):
        super(TemporalAttention, self).__init__()
        # 使用 1D 卷积来跨时序提取帧的重要性特征
        self.attention_net = nn.Sequential(
            # 考虑到动作的连贯性，使用 kernel_size=3 感受相邻帧
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels // reduction),
            nn.ReLU(inplace=True),
            # 压缩为 1 个通道，即每一帧对应 1 个得分
            nn.Conv1d(in_channels // reduction, 1, kernel_size=1),
            # Softmax 确保所有帧的注意力权重加起来等于 1
            nn.Softmax(dim=-1) 
        )

    def forward(self, x):
        """
        输入:
            x: GCN Backbone 的输出特征，形状通常为 (N, M, C, T, V)
               N=Batch, M=人数, C=通道, T=帧数, V=关节数
        返回:
            out: 加权后的特征 (N, M, C, T, V)
            attn_weights: 每一帧的注意力分布 (N, T)
        """
        N, M, C, T, V = x.size()

        # 1. 空间与人数维度的全局平均池化 (GAP)
        # 把每个人的姿态特征压缩，只关注 "在这个时刻，发生了多剧烈的运动"
        # 形状变化: (N, M, C, T, V) -> (N, C, T)
        x_pool = x.mean(dim=(1, 4))

        # 2. 计算时序注意力权重
        # 形状变化: (N, C, T) -> (N, 1, T)
        attn_weights = self.attention_net(x_pool)

        # 3. 将权重广播回原始特征维度
        # 形状重塑: (N, 1, T) -> (N, 1, 1, T, 1) 以便与 x 逐元素相乘
        attn_weights_broadcast = attn_weights.unsqueeze(1).unsqueeze(-1)

        # 4. 加权融合：将重要帧的特征放大，无关准备帧的特征缩小
        # 乘以 T 是为了保持特征数值的总量级不缩水 (因为 Softmax 的值通常很小，平均只有 1/T)
        out = x * attn_weights_broadcast * T 

        return out
    
@MODELS.register_module()
class AttentionGCNHead(GCNHead):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__(in_channels=in_channels, num_classes=num_classes, **kwargs)
        # 初始化时序注意力机制
        self.temporal_attn = TemporalAttention(in_channels=in_channels)
        
    def forward(self, x):
        # 1. 过一遍注意力机制，突出击球关键帧
        # x 的形状此时为 (N, M, C, T, V)
        x = self.temporal_attn(x)
        
        # 2. 交给父类的代码，继续执行全局池化和线性分类层
        return super().forward(x)