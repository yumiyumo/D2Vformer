import math
import torch
import torch.nn as nn

class Fusion_Block(nn.Module):
    def __init__(self,args):
        super(Fusion_Block, self).__init__()

        self.dropout = nn.Dropout(args.dropout)
        self.activation=nn.GELU()
        # Feed Forward
        self.conv1 = nn.Conv1d(in_channels=args.d_feature, out_channels=args.d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_feature, kernel_size=1)
        # Attention norm
        self.norm1 = nn.BatchNorm1d(args.d_feature)
        self.norm2 = nn.BatchNorm1d(args.d_feature)
        # Decoder norm
        self.norm3 = torch.nn.BatchNorm1d(args.d_feature)
        # Decoder projection
        self.fc_out=  nn.Linear(args.d_model, args.d_feature, bias=True)
        # Calculated directly from the formula
        patch_num = int((args.pred_len - args.patch_len) / args.stride + 1) + 1
        self.linear_out=nn.Linear(patch_num*args.patch_len,args.pred_len)



    def forward(self, x, x_date, y_date):
        # x:(B,D,P_N,P_L),
        # x_date:(B,L,D,K),
        # y_date:(B,O,D,K),
        # return:y(B,O,D)

        B,L,D,k = x_date.shape
        # Attention
        scale = 1. / math.sqrt(k)
        scores = torch.einsum("bldk,bodk->bdlo", x_date, y_date)
        A = torch.softmax(scale * scores, dim=-2) # For L × O Softmax on L
        V = torch.einsum("bdnl,bdnp->bdpl", x, A)
        # Attention norm
        B,D,_,_=V.shape
        x=V.reshape(B,D,-1)
        x=self.linear_out(x)
        y = x = self.norm1(x)
        # Feed forward
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        # ResNet+Attention norm
        y=self.norm2((x + y)).transpose(-1,-2)
        return y

