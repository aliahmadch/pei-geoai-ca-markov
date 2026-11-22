"""Network classifiers of the PEI comparison for the five-class legend (FT-Transformer, VCHNet, Swin-GNNet, FMTUNet,
TimeSenCLIP), with the settings that are tuned exposed as arguments."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASS = 5
N_FEAT = 13


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features=N_FEAT, d=32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d))

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class FTTransformer(nn.Module):
    def __init__(self, d=32, n_layers=2, ff=64, dropout=0.1, n_heads=4, n_features=N_FEAT, n_class=N_CLASS):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=ff, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d, n_class)

    def forward(self, x):
        tok = self.tokenizer(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        z = self.encoder(torch.cat([cls, tok], dim=1))
        return self.head(z[:, 0])


class VCHNet(nn.Module):
    def __init__(self, d=48, dropout=0.1, in_ch=13, n_class=N_CLASS):
        super().__init__()
        self.sitt3 = nn.Conv2d(in_ch, d, 3, padding=1)
        self.sitt5 = nn.Conv2d(in_ch, d, 5, padding=2)
        self.fuse = nn.Conv2d(d * 2, d, 1)
        self.norm = nn.BatchNorm2d(d)
        encoder = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=2 * d, batch_first=True, dropout=dropout)
        self.blocks = nn.TransformerEncoder(encoder, num_layers=2)
        self.fcff = nn.Sequential(nn.Conv1d(d, d, 1), nn.GELU(), nn.Conv1d(d, d, 1))
        self.head = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, n_class))

    def forward(self, x):
        h = F.gelu(self.fuse(torch.cat([self.sitt3(x), self.sitt5(x)], dim=1)))
        h = self.norm(h)
        tok = h.flatten(2).transpose(1, 2)
        z1 = self.blocks(tok)
        z2 = z1.transpose(1, 2)
        z2 = z2 + self.fcff(z2)
        pooled = torch.cat([z1.mean(1), z2.mean(2)], dim=1)
        return self.head(pooled)


class WindowAttention(nn.Module):
    def __init__(self, d=48, n_heads=4, window=3):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.window = window
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)

    def forward(self, x, shifted=False):
        b, n, d = x.shape
        side = int(n ** 0.5)
        h = x.view(b, side, side, d)
        w = self.window
        pad = (w - side % w) % w
        if pad:
            h = F.pad(h, (0, 0, 0, pad, 0, pad))
        if shifted:
            h = torch.roll(h, shifts=(-w // 2, -w // 2), dims=(1, 2))
        hp, wp = h.shape[1], h.shape[2]
        nh, nw = hp // w, wp // w
        windows = h.view(b, nh, w, nw, w, d).permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, d)
        qkv = self.qkv(windows).reshape(-1, w * w, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.unbind(2)
        attn = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)) / (d ** 0.5)
        attn = attn.softmax(-1)
        out = (attn @ v.transpose(1, 2)).transpose(1, 2).reshape(-1, w * w, d)
        out = self.proj(out)
        out = out.view(b, nh, nw, w, w, d).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, d)
        if shifted:
            out = torch.roll(out, shifts=(w // 2, w // 2), dims=(1, 2))
        if pad:
            out = out[:, :side, :side]
        return out.reshape(b, n, d)


class SwinGNNet(nn.Module):
    def __init__(self, d=48, in_ch=13, n_class=N_CLASS):
        super().__init__()
        self.embed = nn.Conv2d(in_ch, d, 3, padding=1)
        self.w1 = WindowAttention(d)
        self.w2 = WindowAttention(d)
        self.gcn = nn.Linear(d, d)
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Linear(d, n_class)

    def forward(self, x):
        h = F.gelu(self.embed(x))
        tok = h.flatten(2).transpose(1, 2)
        tok = tok + self.w1(tok, shifted=False)
        tok = tok + self.w2(tok, shifted=True)
        graph = tok.mean(1, keepdim=True)
        tok = tok + F.gelu(self.gcn(tok + graph))
        logits = self.head(tok.mean(1))
        return logits / self.temp.clamp(min=0.1)


class CPUMamba(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(d, d, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(d * 2, d)

    def forward(self, x):
        y, _ = self.gru(x)
        return self.proj(y)


class FMTUNet(nn.Module):
    def __init__(self, d=32, dropout=0.1, in_ch=13, n_class=N_CLASS):
        super().__init__()
        self.opt = nn.Sequential(nn.Conv2d(10, d, 3, padding=1), nn.GELU(), nn.Conv2d(d, d, 3, padding=1))
        self.anc = nn.Sequential(nn.Conv2d(3, d, 3, padding=1), nn.GELU())
        self.fuse = nn.Conv2d(d * 2, d, 1)
        self.down = nn.Conv2d(d, d * 2, 3, stride=2, padding=1)
        self.ssm = CPUMamba(d * 2)
        enc = nn.TransformerEncoderLayer(d_model=d * 2, nhead=4, dim_feedforward=2 * d, batch_first=True, dropout=dropout)
        self.tr = nn.TransformerEncoder(enc, num_layers=1)
        self.up = nn.ConvTranspose2d(d * 2, d, 2, stride=2)
        self.head = nn.Conv2d(d * 2, n_class, 1)

    def forward(self, x):
        o = self.opt(x[:, :10])
        a = self.anc(x[:, 10:])
        f = F.gelu(self.fuse(torch.cat([o, a], 1)))
        dwn = F.gelu(self.down(f))
        b, c, h, w = dwn.shape
        seq = dwn.flatten(2).transpose(1, 2)
        seq = self.tr(self.ssm(seq))
        dwn = seq.transpose(1, 2).view(b, c, h, w)
        up = F.gelu(self.up(dwn))
        if up.shape[-2:] != f.shape[-2:]:
            up = F.interpolate(up, size=f.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.head(torch.cat([up, f], 1))
        cy, cx = logits.shape[-2] // 2, logits.shape[-1] // 2
        return logits[:, :, cy, cx]


class TimeSenCLIP(nn.Module):
    def __init__(self, d=48, in_ch=13, n_class=N_CLASS):
        super().__init__()
        self.visual = nn.Sequential(nn.Conv2d(in_ch, d, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.text = nn.Parameter(torch.randn(n_class, d) * 0.02)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        img = F.normalize(self.visual(x).flatten(1), dim=-1)
        txt = F.normalize(self.text, dim=-1)
        return self.scale * img @ txt.t()


ARCH = {"ft_transformer": (FTTransformer, ("d", "n_layers", "ff", "dropout")), "vchnet": (VCHNet, ("d", "dropout")),
        "swingnnet": (SwinGNNet, ("d",)), "fmtunet": (FMTUNet, ("d", "dropout")), "timesenclip": (TimeSenCLIP, ("d",))}


def make_net(name, hp):
    cls, keys = ARCH[name]
    return cls(**{k: hp[k] for k in keys})
