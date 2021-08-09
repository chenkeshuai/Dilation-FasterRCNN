from __future__ import print_function

import torch
import torch.nn as nn


class Correlation(nn.Module):
    """Correlation Congruence for Knowledge Distillation, ICCV 2019.
    The authors nicely shared the code with me. I restructured their code to be 
    compatible with my running framework. Credits go to the original author"""
    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, f_s, f_t):
        # delta：两个特征矩阵的差
        delta = torch.abs(f_s - f_t)
        # delta[:-1]* delta[1:]中的第i行表示第i和第i+1个batch的delta特征向量点乘
        # 求和后表示第i和第i+1个batch的相异程度
        # 求平均表示两两batch的平均delta相异程度
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss


