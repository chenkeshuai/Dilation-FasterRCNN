from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        # f_s: [batch_size, h, w] -> student: [batch_size, hxw]
        # f_t: [batch_size, h, w] -> teacher: [batch_size, hxw]
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)
        pdb.set_trace()

        # RKD 距离损失
        with torch.no_grad(): 
            # 老师特征图各行向量之间的欧氏距离矩阵，并归一化
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        # 学生特征图各行向量之间的欧氏距离矩阵，并归一化
        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD 角度损失
        with torch.no_grad():
            # 获取单位方向向量
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            # cos<angle> = e1 * e2
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        # 获取单位方向向量
        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        # cos<angle> = e1 * e2
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        # 计算距离损失和角度损失的加权和
        loss = self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        '''
        计算e各行向量之间的欧式距离
        例如:
            input:  [[a11,a12],[a21,a12]]
            output: [[0,  d12],[d21,  0]]
                if squared: d12 = d21 =      (a21-a11)^2+(a22-a12)^2
                else:       d12 = d21 = sqrt((a21-a11)^2+(a22-a12)^2)
        '''
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res
