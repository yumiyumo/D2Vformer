from layers.koopa_related import *
from layers.koopa_AR_related import *
from layers.Date2Vec import *
from layers.Fusion_Block import *
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class D2Vformer(nn.Module):
    def __init__(self, configs):
        super(D2Vformer, self).__init__()
        # Changing variable names
        self.configs=configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.d_feature = configs.d_feature
        self.d_model = configs.d_model

        # Encoder
        self.all_linear = nn.Linear(self.seq_len, self.d_model)
        self.season_linear = nn.Linear(self.seq_len, self.d_model)
        self.trend_linear_decoder = nn.Linear(self.d_model, self.pred_len)

        # Mapping time-invariant data
        K_init = torch.randn(self.d_model, self.d_model)
        U, _, V = torch.svd(K_init)  # stable initialization
        self.K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.K.weight.data = torch.mm(U, V.t())

        # Decoder
        self.project = nn.Linear(self.d_model, self.pred_len)

        # Encoding of time information
        self.linear_t = nn.Linear(self.seq_len,self.d_model)
        self.linear_t_out = nn.Linear(self.d_model, self.seq_len)

        #Revin
        self.revin = RevIN(self.d_feature, affine=True, subtract_last= False)

        # Date2Vec
        self.Date2Vec = Date2Vec(self.d_model,configs.T2V_outmodel,self.d_feature,d_mark=configs.d_mark)

        # Fusion
        self.fusion = Fusion_Block(args=configs)
        self.patch_len=configs.patch_len
        self.stride=configs.stride

        # Fourier splitting-->Decomposition of timing data
        self.mask_spectrum=configs.mask_spectrum
        self.disentanglement = FourierFilter(self.mask_spectrum)

        # BN Layer
        self.bn_inv=nn.BatchNorm1d(self.d_model)
        self.bn_var = nn.BatchNorm1d(self.d_feature)
        self.bn_pred=nn.BatchNorm1d(self.d_feature)

    def do_patching(self,z,stride):
        # Patching
        z=nn.functional.pad(z, (0, stride))
        # real doing patching
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
        return z


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # revin   in[B,L,D]    out[B,L,D]
        x_enc = self.revin(x_enc, 'norm')
        # Decomposition into time-varying and time-invariant components
        x_var, x_inv = self.disentanglement(x_enc)

        '''The following treatment of time-invariant components'''
        # The duration L of the time-invariant information is mapped to the high-dimensional d_model-->Encoder
        trend_output = self.trend_linear(x_inv.permute(0, 2, 1)).permute(0, 2, 1)
        season_output = self.season_linear(x_inv.permute(0, 2, 1)).permute(0, 2, 1)
        encoder_output = trend_output + season_output  # (B,L,D)
        encoder_output=self.bn_inv(encoder_output)
        y_inv = self.K(encoder_output.transpose(-1, -2)).transpose(-1, -2) # Koopa中时不变中的全局K算子
        # Decoder
        y_inv = self.project(y_inv.permute(0, 2, 1)).permute(0, 2, 1)

        '''The following operations are performed on time-varying data'''
        # Getting the date input for D2V
        D2V_input=torch.cat((x_mark_enc,x_mark_dec[:,-self.pred_len:,:]),dim=-2)
        D2V_input=D2V_input.transpose(-1,-2) #(B,D,L)

        # Encoding the time code of history
        t_history = self.linear_t(D2V_input[:, :, :self.seq_len])
        D2V_input = torch.cat([t_history, D2V_input[:, :, -self.pred_len:]], dim=-1)#(B,d_mark,L+O)


        all_output_var = self.all_linear(x_var.permute(0, 2, 1))
        season_output_var = self.season_linear(x_var.permute(0, 2, 1))
        trend_output_var =self.trend_linear_decoder(all_output_var-season_output_var).transpose(-1,-2)

        encoder_output_var=season_output_var
        encoder_output_var=self.bn_var(encoder_output_var)

        # The relationship between historical and predicted positions is extracted by inputting the encoded historical information into D2V
        D2V_output = self.Date2Vec(encoder_output_var, D2V_input)  # [B,D,L+O,k]
        D2V_output = D2V_output.permute(0, 2, 1, 3)

        # Date coding of model history input data
        D2V_x_date = D2V_output[:, :self.d_model, :, :]
        # Downscaling high-dimensional temporal information
        D2V_x_date = self.linear_t_out(D2V_x_date.transpose(1, -1)).transpose(1, -1)
        # Date coding of model predicted locations
        D2V_y_date = D2V_output[:, -self.pred_len:, :, :]

        # Cut patch, here y also have to do path because my Attention calculation later is calculated by P_L*k, so here also have to do path slice
        x_var = self.do_patching(x_var.transpose(-1, 1), self.configs.stride)  # (B,D,P_N,P_L)
        D2V_x_date = self.do_patching(D2V_x_date.transpose(1, -1), self.configs.stride)  # (B,K,D,P_N,P_L)
        D2V_y_date = self.do_patching(D2V_y_date.transpose(1, -1), self.configs.stride)  # (B,K,D,P_N,P_L)

        B, k, D, P_N, P_L = D2V_x_date.shape
        D2V_x_date = D2V_x_date.reshape(B, P_N, D, P_L * k)
        B, k, D, P_N, P_L = D2V_y_date.shape
        D2V_y_date = D2V_y_date.reshape(B, P_N, D, P_L * k)

        # Based on the extracted future and history DATE relationships, and the input history sequence, future results are obtained
        prediction_var = self.fusion(x_var, D2V_x_date, D2V_y_date)

        # Integrate the output of each component
        prediction=prediction_var+y_inv+trend_output_var
        prediction=self.bn_pred(prediction.transpose(-1,-2)).transpose(-1,-2)

        # Denorm
        prediction = self.revin(prediction, 'denorm')

        return prediction # [B, O, D]






