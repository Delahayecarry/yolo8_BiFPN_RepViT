import torch
import torch.nn as nn 
import torch.nn.functional as F


class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        
        # 调整特征图的尺寸，使它们相同
        target_size = x[0].shape[-2:]  # 以第一个特征图的尺寸为目标尺寸
        resized_feature_maps = [F.interpolate(fm, size=target_size, mode='nearest') for fm in x]
        
        return torch.cat(resized_feature_maps, self.d)

class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        
        # 调整特征图的尺寸，使它们相同
        target_size = x[0].shape[-2:]  # 以第一个特征图的尺寸为目标尺寸
        resized_feature_maps = [F.interpolate(fm, size=target_size, mode='nearest') for fm in x]
        
        return torch.cat(resized_feature_maps, self.d)
