from mmcv.transforms import BaseTransform
from mmaction.registry import TRANSFORMS
import numpy as np
import math

@TRANSFORMS.register_module()
class RandomShift(BaseTransform):
    """随机平移骨架坐标
    Args:
        shift_range (float): 平移范围（通常在归一化坐标下使用，例如 0.1 代表 10%）
        prob (float): 触发概率，默认 0.5
    """
    def __init__(self, shift_range=0.1, prob=0.5):
        self.shift_range = shift_range
        self.prob = prob

    def transform(self, results):
        # 根据概率决定是否触发增强
        if np.random.rand() > self.prob:
            return results

        if 'keypoint' in results:
            # 提取关键点数据 (通常 Shape 为 M, T, V, C)
            # M: 人数, T: 帧数, V: 关节点数, C: 坐标维度(通常 C 的前两位是 x 和 y)
            kpts = results['keypoint'].copy()
            
            # 计算随机偏移量
            shift_x = np.random.uniform(-self.shift_range, self.shift_range)
            shift_y = np.random.uniform(-self.shift_range, self.shift_range)
            
            # 对所有的 X 和 Y 坐标施加平移 (针对整个视频序列同步平移)
            kpts[..., 0] += shift_x
            kpts[..., 1] += shift_y
            
            results['keypoint'] = kpts

        return results
    

@TRANSFORMS.register_module()
class RandomRot2D(BaseTransform):
    """随机二维旋转骨架坐标
    Args:
        angle_range (float): 最大旋转角度 (度)，比如 10 代表在 -10度 到 10度 之间随机旋转
        prob (float): 触发概率，默认 0.5
    """
    def __init__(self, angle_range=10, prob=0.5):
        self.angle_range = angle_range
        self.prob = prob

    def transform(self, results):
        # 概率判断
        if np.random.rand() > self.prob:
            return results

        if 'keypoint' in results:
            # 提取关键点数据 (M, T, V, C)
            kpts = results['keypoint'].copy()
            
            # 随机生成一个旋转角度，并转换为弧度
            angle = np.random.uniform(-self.angle_range, self.angle_range)
            theta = angle * math.pi / 180.0
            
            # 预计算 cos 和 sin
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            
            # 提取当前的 X 和 Y 坐标
            x = kpts[..., 0]
            y = kpts[..., 1]
            
            # 运用二维旋转矩阵计算新的坐标
            # (注意：通常这个增强放在 PreNormalize2D 之后，此时人的中心已经是对齐到原点0了)
            new_x = x * cos_theta - y * sin_theta
            new_y = x * sin_theta + y * cos_theta
            
            # 将新坐标写回
            kpts[..., 0] = new_x
            kpts[..., 1] = new_y
            
            results['keypoint'] = kpts

        return results
    
@TRANSFORMS.register_module()
class SkeletonRandomFlip(BaseTransform):
    """专门针对骨架数据的随机镜像翻转
    Args:
        prob (float): 触发概率，默认 0.5
        left_kp (list): 左侧关节点索引列表
        right_kp (list): 右侧关节点索引列表
    """
    def __init__(self, prob=0.5, left_kp=None, right_kp=None):
        self.prob = prob
        self.left_kp = left_kp if left_kp is not None else []
        self.right_kp = right_kp if right_kp is not None else []

    def transform(self, results):
        if np.random.rand() > self.prob:
            return results

        if 'keypoint' in results:
            kpts = results['keypoint'].copy()
            
            # 1. 翻转 X 坐标
            # 注意：这个增强必须放在 PreNormalize2D 之后！
            # 因为 PreNormalize2D 会把骨盆坐标平移到 (0,0) 作为中心点，
            # 此时翻转左右只需要一个简单的负号： x = -x
            kpts[..., 0] = -kpts[..., 0]
            
            # 2. 交换左右对称关节的索引 (解决正反手判断的最关键步骤)
            if self.left_kp and self.right_kp:
                for l, r in zip(self.left_kp, self.right_kp):
                    temp = kpts[..., l, :].copy()
                    kpts[..., l, :] = kpts[..., r, :]
                    kpts[..., r, :] = temp
                    
            results['keypoint'] = kpts

        return results
    

@TRANSFORMS.register_module()
class VectorRandomFlip(BaseTransform):
    """用于多模态骨架识别的自定义翻转模块，专为 Bone, Motion, Angle 设计。
    
    Args:
        flip_ratio (float): 执行翻转的概率，默认 0.5。
        left_kp (list[int]): 左侧关节点的索引列表。
        right_kp (list[int]): 右侧关节点的索引列表。
        mode (str): 模态类型，必须是 'vector' (用于Bone/Motion) 或 'angle'。
    """
    def __init__(self, flip_ratio=0.5, left_kp=None, right_kp=None, mode='vector'):
        self.flip_ratio = flip_ratio
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.mode = mode.lower()
        
        assert self.mode in ['vector', 'angle'], "mode 必须是 'vector' 或 'angle'"
        if left_kp is not None and right_kp is not None:
            assert len(self.left_kp) == len(self.right_kp), "左右关节点列表长度必须一致"

    def transform(self, results):
        # 以一定的概率执行翻转
        if np.random.rand() > self.flip_ratio:
            return results
        
        if 'keypoint' in results:
            # 取出坐标数据，标准的 shape 通常为 (M, T, V, C) 
            # M: 人数, T: 帧数, V: 关节点数, C: 通道数
            kpts = results['keypoint'] 
            
            # ==========================================
            # 1. 核心逻辑：针对不同模态进行特定通道的“取反”
            # ==========================================
            if self.mode == 'vector':
                # Bone 和 Motion: X轴分量 (Channel 0) 直接取相反数，Y轴不变
                kpts[..., 0] = -kpts[..., 0]
            elif self.mode == 'angle':
                # Angle: cos 保持不变 (Channel 0)，sin 取相反数 (Channel 1)
                kpts[..., 1] = -kpts[..., 1]
                
            # ==========================================
            # 2. 交换左右关节点的索引
            # ==========================================
            if self.left_kp is not None and self.right_kp is not None:
                # 使用 numpy 的高级索引一次性完成交换
                temp = kpts[..., self.left_kp, :].copy()
                kpts[..., self.left_kp, :] = kpts[..., self.right_kp, :]
                kpts[..., self.right_kp, :] = temp
                
            results['keypoint'] = kpts
            
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(flip_ratio={self.flip_ratio}, mode={self.mode})'
        return repr_str