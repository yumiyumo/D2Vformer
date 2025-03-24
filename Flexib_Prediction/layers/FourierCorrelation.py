import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    Get modes in the frequency domain:
    'random' means sampling randomly;
    'else' means selecting the lowest modes.
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


class FourierBlock(nn.Module):
    """
    Fourier block for frequency domain representation learning.
    It performs FFT, linear transform, and inverse FFT.
    """

    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print(f'modes={modes}, index={self.index}')

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(3, in_channels // 3, out_channels // 3, len(self.index), dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        # Complex multiplication (batch, in_channels, x) * (in_channels, out_channels, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for i, wi in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, wi], self.weights1[:, :, :, i])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)


class FourierCrossAttention(nn.Module):
    """
    Fourier Cross Attention layer for frequency domain representation learning.
    It performs FFT, linear transform, attention mechanism, and inverse FFT.
    """

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)
        print(f'modes_q={len(self.index_q)}, index_q={self.index_q}')
        print(f'modes_kv={len(self.index_kv)}, index_kv={self.index_kv}')

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(3, in_channels // 3, out_channels // 3, len(self.index_q), dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # Perform attention on frequency domain
        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_)
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception(f'{self.activation} activation function is not implemented')

        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return (out, None)
