import torch
import torch.nn as nn
import math
from layers.Revin import *
import matplotlib.pyplot as plt
import os
import seaborn as sns

class TimeVarKP_AR(nn.Module):
    """
    Koopman Predictor with DMD (analysitical solution of Koopman operator)
    Utilize local variations within individual sliding window to predict the future of time-variant term
    """

    def __init__(self,
                 enc_in=8,
                 input_len=96,
                 pred_len=96,
                 seg_len=48,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None,
                 multistep=False,
                 ):
        super(TimeVarKP_AR, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep
        self.encoder, self.decoder = encoder, decoder
        self.freq = math.ceil(self.input_len / self.seg_len)  # segment number of input
        self.step = math.ceil(self.pred_len / self.seg_len)  # segment number of output
        self.padding_len = self.seg_len * self.freq - self.input_len

        self.dynamics =  KPLayer_AR()

    def forward(self, x):
        # x: B L C
        B, L, C = x.shape

        res = torch.cat((x[:, L - self.padding_len:, :], x), dim=1)  # 进行一个padding 为了后续切分Patch的时候可以被整除

        res = res.chunk(self.freq, dim=1)  # F x B P C, P means seg_len （将某一个维度进行分块操作，self.freq是分成几块）
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)  # B F PC (F=patch_num,P=patch_len.C=dim)

        res = self.encoder(res)  # B F H
        x_rec, x_pred = self.dynamics(res, self.step)  # B F H, B S H

        x_rec = self.decoder(x_rec)  # B F PC 映射回输出的维度
        x_rec = x_rec.reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # B L C

        x_pred = self.decoder(x_pred)  # B S PC
        x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]  # B S C

        return x_rec, x_pred


class KPLayer_AR(nn.Module):
    """
    A demonstration of finding one step transition of linear system by DMD iteratively
    """

    def __init__(self):
        super(KPLayer_AR, self).__init__()

        self.K = None  # B E E

    def one_step_forward(self, z, return_rec=False, return_K=False):
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]  # 一个是Z_back，一个是Z_force

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution  # B E E  利用最小二乘的方法求解出K_var
        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_pred = torch.bmm(z[:, -1:, :], self.K)  # bmm是矩阵相乘，利用回顾窗口的最后一个patch乘上对应的K_var得到预测窗口的结果
        if return_rec:  # 回顾
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)  # 利用前一个patch作为输入，预测后一个patch(作为重建结果)
            return z_rec, z_pred

        return z_pred

    def forward(self, z, pred_len=1):
        assert pred_len >= 1, 'prediction length should not be less than 1'
        z_rec, z_pred = self.one_step_forward(z, return_rec=True)
        z_preds = [z_pred]
        for i in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            z_preds.append(z_pred)
        z_preds = torch.cat(z_preds, dim=1)
        return z_rec, z_preds
