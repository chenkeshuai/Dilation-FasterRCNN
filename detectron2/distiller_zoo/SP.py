from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        '''对于老师和学生网络输出的每一个元素计算相似性损失'''
        return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def similarity_loss(self, f_s, f_t):
        '''损失函数'''

        # bsz: batch size
        # f_s: [batch_size, h, w]
        bsz = f_s.shape[0]

        # f_s: [batch_size, h, w] -> [batch_size, hxw]
        # f_t: [batch_size, h, w] -> [batch_size, hxw]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        # G_s: [batch_size, batch_size]为对称矩阵
        G_s = torch.mm(f_s, torch.t(f_s))
        G_t = torch.mm(f_t, torch.t(f_t))

        # 归一化后，G_s(i,j)表示students features第i和第j批数据的相似程度
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.nn.functional.normalize(G_t)

        # 相似度矩阵的差
        G_diff = G_t - G_s

        # 计算相似度差值矩阵G_diff的元素均值
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)

        return loss
