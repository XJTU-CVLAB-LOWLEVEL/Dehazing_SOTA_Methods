# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# 3D DB + 2D SR
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.PWC_Net import Backward


def make_model(args, parent=False):
    return RDN(args)


class RDB_3DConv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=(3, 3, 3)):
        super(RDB_3DConv, self).__init__()
        Cin = inChannels
        G = growRate
        self.reflection_padding = (kSize[0] // 2, kSize[1] // 2, kSize[2] // 2, kSize[0] // 2, kSize[1] // 2,
                                   kSize[2] // 2)  # (padLeft, padRight, padTop, padBottom, padFront, padBack)
        self.conv = nn.Sequential(*[
            nn.Conv3d(Cin, G, kSize, padding=0, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = F.pad(x, self.reflection_padding, mode='replicate')
        out = self.conv(out)
        return torch.cat((x, out), 1)


class RDB3D(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB3D, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_3DConv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv3d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


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
        out = self.conv(x)
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
            'A': (10, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=2)
        self.SFENet3 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=2)

        # Redidual dense blocks and dense feature fusion
        self.RDB3Ds = nn.ModuleList()
        for i in range(self.D // 2):
            self.RDB3Ds.append(
                RDB3D(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF1 = nn.Sequential(*[
            nn.Conv3d(self.D // 2 * G0, G0, 1, padding=0, stride=1),
            nn.Conv3d(G0, G0, (3, 3, 3), padding=(0, 1, 1), stride=1),
            nn.Conv3d(G0, G0, (3, 3, 3), padding=(0, 1, 1), stride=1),
        ])
        # self.conv_deblur = nn.Conv2d(G0, 3, kSize, padding=(kSize - 1) // 2, stride=1)
        self.GFF_2d = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # 2D RDN
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF2 = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, G0 * 4, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(G0, G0 * 4, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
        ])
        self.conv_out = nn.Conv2d(G0, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1)


    def forward(self, x):
        n, c, h, w = x.size()
        frames = c // 3
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

        inputs = torch.cat(inputs, 1).view(n, frames, 64, h, w).permute(0, 2, 1, 3, 4).contiguous()

        RDB3Ds_out = []
        '''
        for i in range(self.D//2):
            inputs = self.RDB3Ds[i](inputs)
            inputs_warp = []
            for i in range(frames):
                if i == frames // 2:
                    inputs_warp.append(inputs[:,:, i:i+1])
                    continue
                inputs_warp.append(Backward(tensorInput=inputs[:,:, i], tensorFlow=flowmaps[:, i*2:(i+1)*2]).view(n,64,1,h,w))

            RDB3Ds_out.append(torch.cat(inputs_warp, 2))
        '''
        for i in range(self.D // 2):
            inputs = self.RDB3Ds[i](inputs)
            RDB3Ds_out.append(inputs)

        res_3d = self.GFF1(torch.cat(RDB3Ds_out, 1))
        res_2d = self.GFF_2d(res_3d.view(n, 64, h, w))
        res_2d += base_LR
        # deblur_out = self.conv_deblur(res_2d)

        RDBs_out = []
        res = res_2d
        for i in range(self.D):
            res_2d = self.RDBs[i](res_2d)
            RDBs_out.append(res_2d)

        x = self.GFF2(torch.cat(RDBs_out, 1))
        x += res
        x = self.UPNet(x)  + base_HR


        return self.conv_out(x) # deblur_out

