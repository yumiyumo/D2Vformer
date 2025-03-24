import math
import torch
import torch.nn as nn

# === Fourier Filter ===
class FourierFilter(nn.Module):
    """
    Fourier filter that separates time-variant and time-invariant components.
    """
    def __init__(self, mask_spectrum):
        super(FourierFilter, self).__init__()
        self.mask_spectrum = mask_spectrum

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)  # FFT along time dimension
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0  # Mask dominant frequencies
        x_var = torch.fft.irfft(xf * mask, dim=1)  # Time-variant component
        x_inv = x - x_var  # Time-invariant component
        return x_var, x_inv

# === MLP Encoder/Decoder ===
class MLP(nn.Module):
    """
    Multi-layer perceptron for encoding/decoding high-dimensional time series representations.
    """
    def __init__(self, f_in, f_out, hidden_dim=128, hidden_layers=2, dropout=0.05, activation='tanh'):
        super(MLP, self).__init__()
        act_fn = nn.ReLU() if activation == 'relu' else nn.Tanh()
        layers = [nn.Linear(f_in, hidden_dim), act_fn, nn.Dropout(dropout)]
        for _ in range(hidden_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_fn, nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)  # Shape: [B, S, f_out]

# === Time-Variant Koopman Predictor ===
class TimeVarKP(nn.Module):
    """
    Koopman predictor using DMD for time-variant dynamics.
    """
    def __init__(self, enc_in=8, input_len=96, pred_len=96, seg_len=24, dynamic_dim=128,
                 encoder=None, decoder=None, multistep=False):
        super(TimeVarKP, self).__init__()
        self.encoder, self.decoder = encoder, decoder
        self.step = 1
        self.dynamics = KPLayerApprox() if multistep else KPLayer()

    def forward(self, T2V_x_date, x_var, T2V_y_date):
        B, L, C, K = T2V_x_date.shape
        _, O, _, _ = T2V_y_date.shape

        T2V_x_date = self.encoder(T2V_x_date.view(B, L, -1))  # [B, L, H]
        T2V_y_date = self.encoder(T2V_y_date.view(B, O, -1))  # [B, O, H]

        x_rec, x_pred = self.dynamics(T2V_x_date, x_var, T2V_y_date, self.step)
        return self.decoder(x_rec), self.decoder(x_pred)

# === Koopman Layer (One-step) ===
class KPLayer(nn.Module):
    """
    Solves linear Koopman system using iterative one-step updates (DMD).
    """
    def __init__(self):
        super(KPLayer, self).__init__()
        self.num = 3  # Iterations for refinement
        self.K = None

    def one_step_forward(self, z, label, pred_z):
        pred_sum, rec_sum = 0, 0
        for _ in range(self.num):
            self.K = torch.linalg.lstsq(z, label).solution
            pred = torch.bmm(pred_z, self.K)
            pred_sum += pred
            rec = torch.bmm(z, self.K)
            rec_sum += rec
            label -= rec
        return rec_sum, pred_sum

    def forward(self, z, label, pred_z, pred_len=1):
        z_rec, z_pred = self.one_step_forward(z, label, pred_z)
        preds = [z_pred]
        for _ in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            preds.append(z_pred)
        return z_rec, torch.cat(preds, dim=1)

# === Koopman Layer (Multi-step Approximation) ===
class KPLayerApprox(nn.Module):
    """
    Multi-step approximation of Koopman operator using DMD.
    """
    def __init__(self):
        super(KPLayerApprox, self).__init__()
        self.K = None
        self.K_step = None

    def forward(self, z, pred_len=1):
        B, input_len, E = z.shape
        assert input_len > 1, "Input length must be greater than 1"
        x, y = z[:, :-1], z[:, 1:]
        self.K = torch.linalg.lstsq(x, y).solution

        if torch.isnan(self.K).any():
            self.K = torch.eye(E).to(z.device).unsqueeze(0).repeat(B, 1, 1)

        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)

        if pred_len <= input_len:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any():
                self.K_step = torch.eye(E).to(z.device).unsqueeze(0).repeat(B, 1, 1)
            z_pred = torch.bmm(z[:, -pred_len:], self.K_step)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():
                self.K_step = torch.eye(E).to(z.device).unsqueeze(0).repeat(B, 1, 1)
            preds, temp = [], z
            for _ in range(math.ceil(pred_len / input_len)):
                temp = torch.bmm(temp, self.K_step)
                preds.append(temp)
            z_pred = torch.cat(preds, dim=1)[:, :pred_len, :]

        return z_rec, z_pred

# === Time-Invariant Koopman Predictor ===
class TimeInvKP(nn.Module):
    """
    Koopman predictor for time-invariant dynamics using a learnable Koopman operator.
    """
    def __init__(self, input_len=96, pred_len=96, dynamic_dim=128, encoder=None, decoder=None):
        super(TimeInvKP, self).__init__()
        self.encoder, self.decoder = encoder, decoder

        # Initialize Koopman operator using orthogonal decomposition
        K_init = torch.randn(dynamic_dim, dynamic_dim)
        U, _, V = torch.svd(K_init)
        self.K = nn.Linear(dynamic_dim, dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())

    def forward(self, x):
        x = self.encoder(x.transpose(1, 2))     # [B, C, L] -> [B, C, H]
        x = self.K(x)                           # Apply learnable Koopman operator
        x = self.decoder(x)                     # Map to prediction
        return x.transpose(1, 2)                # [B, S, C]
