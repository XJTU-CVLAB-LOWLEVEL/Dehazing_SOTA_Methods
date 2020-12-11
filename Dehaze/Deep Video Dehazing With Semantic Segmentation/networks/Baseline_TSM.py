# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# 3D DB + 2D SR
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.PWC_Net import Backward

def make_model(args, parent=False):
    return RDN(args)

def TemporalShift(x, n_segment, fold_div):
    nt, c, h, w = x.size()
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)

    fold = c // fold_div
    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
    out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

    return out.view(nt, c, h, w)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = TemporalShift(x, n_segment=5, fold_div=8)
        out = self.conv(out)
        #out += x
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (16, 6, 32),
            'Test': (1, 1, 3),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=2)
        self.SFENet3 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=2)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            #nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        self.GFF1 = nn.Sequential(*[
            nn.Conv3d(G0, G0, (3, 3, 3), padding=(0, 1, 1), stride=1),
            nn.Conv3d(G0, G0, (3, 3, 3), padding=(0, 1, 1), stride=1),
        ])

        self.GFF2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, G0 * 4, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(G0, G0 * 4, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
        ])
        self.conv_out = nn.Conv2d(G0, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1)

    def forward(self, x):
        n,c,h,w = x.size()
        frames = c //3
        inputs = []
        for i in range(frames):
            f__1 = self.SFENet1(x[:, i * 3:(i + 1) * 3])
            if i == frames // 2:
                base_HR = f__1
            f__1 = self.SFENet2(f__1)
            f__1 = self.SFENet3(f__1)
            if i == frames // 2:
                base_LR = f__1
            inputs.append(f__1)
            n, c, h, w = f__1.size()

        inputs = torch.cat(inputs, 1).view(-1, 64, h, w)

        RDBs_out = []
        for i in range(self.D):
            inputs = self.RDBs[i](inputs)
            RDBs_out.append(inputs)

        res = self.GFF(torch.cat(RDBs_out, 1))
        res = res.view(n, frames, 64, h, w).permute(0,2,1,3,4).contiguous()
        res = self.GFF1(res)
        res = res.view(n, 64, h, w)
        res = self.GFF2(res)
        x += res
        x = self.UPNet(x)  + base_HR


        return self.conv_out(x) # deblur_out

