import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from mmaction.registry import MODELS

# MMAction2 原生的骨骼图结构提取
from mmaction.models.utils import Graph

# ==========================================================
# 1. 初始化工具函数 (Init Functions - 来源于 pyskl/gcns/utils/init_func.py)
# ==========================================================
def conv_init(conv):
    """初始化卷积层权重"""
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    """初始化批归一化层权重"""
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

# ==========================================================
# 2. 时序卷积组件 (TCN Modules - 来源于 tcn.py 与 msg3d_utils.py)
# ==========================================================
class unit_tcn(BaseModule):
    """基础时序卷积单元 (来源于 pyskl/gcns/utils/tcn.py)"""
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0, init_cfg=None):
        super(unit_tcn, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))

    def init_weights(self):
        conv_init(self.conv)
        if not isinstance(self.bn, nn.Identity):
            bn_init(self.bn, 1)


class MSTCN(BaseModule):
    """多尺度时序卷积网络 (来源于 pyskl/gcns/utils/msg3d_utils.py)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=[1, 2, 3, 4], residual=True, act_cfg=dict(type='ReLU'), tcn_dropout=0, init_cfg=None):
        super(MSTCN, self).__init__(init_cfg=init_cfg)
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        branch_channels_rem = out_channels - branch_channels * (self.num_branches - 1)

        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        # 多尺度空洞卷积分支
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                build_activation_layer(act_cfg),
                unit_tcn(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # MaxPool 分支
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            build_activation_layer(act_cfg),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        # 1x1 分支
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels_rem, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels_rem)
        ))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

        self.act = build_activation_layer(act_cfg)
        self.drop = nn.Dropout(tcn_dropout)

    def forward(self, x):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        out = self.act(out)
        out = self.drop(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, _BatchNorm):
                bn_init(m, 1)

# ==========================================================
# 3. 图卷积组件 (GCN Modules - 来源于 pyskl/gcns/utils/gcn.py)
# ==========================================================
class CTRGC(BaseModule):
    """自适应通道拓扑推理图卷积核"""
    def __init__(self, in_channels, out_channels, rel_reduction=8, init_cfg=None):
        super(CTRGC, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
            
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x, A=None, alpha=1):
        # Input: N, C, T, V
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0) 
        x1 = torch.einsum('ncuv,nctu->nctv', x1, x3)
        return x1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, _BatchNorm):
                bn_init(m, 1)

class unit_ctrgcn(BaseModule):
    """核心 CTR-GCN 空间处理单元"""
    def __init__(self, in_channels, out_channels, A, init_cfg=None):
        super(unit_ctrgcn, self).__init__(init_cfg=init_cfg)
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels

        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()

        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.A = nn.Parameter(A.clone())
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = None
        for i in range(self.num_subset):
            z = self.convs[i](x, self.A[i], self.alpha)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, _BatchNorm):
                bn_init(m, 1)
        # Residual Zero-init：极其重要的初始化技巧，增强深层网络的稳定性和表现
        bn_init(self.bn, 1e-6)

# ==========================================================
# 4. 主干网络结构 (CTRGCN Backbone - 来源于 pyskl/gcns/ctrgcn.py)
# ==========================================================
class CTRGCNBlock(BaseModule):
    """CTR-GCN 基础构建块 (包含 GCN 与 TCN)"""
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, kernel_size=5, dilations=[1, 2], tcn_dropout=0, init_cfg=None, **kwargs):
        super(CTRGCNBlock, self).__init__(init_cfg=init_cfg)
        self.gcn1 = unit_ctrgcn(in_channels, out_channels, A)
        self.tcn1 = MSTCN(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations,
            residual=False,
            tcn_dropout=tcn_dropout)
        
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

    def init_weights(self):
        self.tcn1.init_weights()
        self.gcn1.init_weights()
        if isinstance(self.residual, unit_tcn):
            self.residual.init_weights()


@MODELS.register_module()
class CTRGCN(BaseModule):
    """
    自适应通道图卷积网络 (Channel-wise Topology Refinement Graph Convolution)
    适配于 MMAction2 的完全形态
    """
    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 num_person=2,
                 init_cfg=None,
                 **kwargs):
        
        # 兼容处理：将传入的 pretrained 字符串路径转化为 MMEngine 标准的 init_cfg
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
            
        super(CTRGCN, self).__init__(init_cfg=init_cfg)

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.num_person = num_person
        self.base_channels = base_channels

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))

        # 网络块堆叠逻辑
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'tcn_dropout'}
        modules = [CTRGCNBlock(in_channels, base_channels, A.clone(), residual=False, **kwargs0)]
        for i in range(2, num_stages + 1):
            in_channels = base_channels
            out_channels = base_channels * (1 + (i in inflate_stages))
            stride = 1 + (i in down_stages)
            modules.append(CTRGCNBlock(base_channels, out_channels, A.clone(), stride=stride, **kwargs))
            base_channels = out_channels
        self.net = nn.ModuleList(modules)

    def init_weights(self):
        """覆盖 MMEngine BaseModule 的初始化方式"""
        # 调用父类处理预训练权重的加载逻辑
        super().init_weights() 
        
        # 仅在没有传入预训练配置时，执行底层的自定义初始化，防止预训练权重被覆盖
        if self.init_cfg is None:
            bn_init(self.data_bn, 1)
            for module in self.net:
                module.init_weights()

    def forward(self, x):
        # 核心：恢复正确的骨骼点数据输入维度拆解
        # MMAction2 的骨骼数据默认传入形状为 (N, M, T, V, C)
        N, M, T, V, C = x.size()
        
        # 将输入排列转换为模型底层计算所需的 (N, M, V, C, T) 并展平处理 BatchNorm1D
        x = x.permute(0, 1, 3, 4, 2).contiguous() 
        x = self.data_bn(x.view(N, M * V * C, T))
        
        # 还原维度，并合并 N(批次) 与 M(人数) 送入核心网络: Shape 变为 (N * M, C, T, V)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # 逐层前向传播
        for gcn in self.net:
            x = gcn(x)

        # 最终输出还原回包含 M 维度的张量: Shape 为 (N, M, C_out, T_out, V)
        x = x.reshape((N, M) + x.shape[1:])
        return x