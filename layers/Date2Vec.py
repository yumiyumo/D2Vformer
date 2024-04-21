import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Date2Vec(nn.Module):
    def __init__(self, in_features, out_features, d_features,d_mark):
        # For D2V, here the input in_features refers to seq_len and
        # out_features refers to how many sinusoids are used to characterise the corresponding each D
        super(Date2Vec, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(d_features, 1, 1, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        nn.init.uniform_(self.w, 0, 2 * torch.pi)
        self.b = nn.parameter.Parameter(torch.randn(d_features, 1, 1, out_features - 1))
        self.f = torch.sin
        self.w_transform_t=nn.Parameter(torch.randn((d_mark,d_features)),requires_grad=True)
        self.w_transform_s = nn.Parameter(torch.randn((d_mark,d_features)), requires_grad=True)

    def forward(self, data, tau):
        output=self.D2V(data, tau, self.f, self.w, self.b, self.w0, self.b0)
        return output

    def D2V(self,data, tau, f, w, b, w0, b0):
        _,D,_=data.shape
        tau=tau.unsqueeze(-1).repeat(1,1,1,D)#(B,d_mark,L+O,d_feature)
        tau=tau.transpose(1,-1)#(B,d_feature,L+O,d_mark)
        mark=tau.shape[-1]
        w_trend = torch.matmul(data, w0).unsqueeze(-2)
        w_season = torch.matmul(data, w).unsqueeze(-2)

        w_trend = w_trend.unsqueeze(2).repeat(1,1,mark,1,1)
        w_season = w_season.unsqueeze(2).repeat(1, 1, mark, 1, 1)
        tau = tau.unsqueeze(2).transpose(2, -1)
        v2 = torch.matmul(tau, w_trend) + b0
        v1 = f(torch.matmul(tau, w_season) + b)

        self.w_trend=w_trend
        self.w_season=w_season
        # Melting away the mark dimension (B,D,L,k)
        v1 = torch.mean(v1,dim=2)
        v2 = torch.mean(v2,dim=2)

        return torch.cat([v2, v1], -1)