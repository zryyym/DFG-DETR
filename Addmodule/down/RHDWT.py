import torch
import torch.nn as nn

from pytorch_wavelets import DWTForward


class single_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(single_conv, self).__init__()
        self.s_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.s_conv(x)
        return x


class RHDWT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.identety = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=2, padding=1)
        self.DWT = DWTForward(J=1, wave='haar')
        self.dconv_encode = single_conv(4 * in_dim, out_dim)

    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        a = DMT1_yh[0]
        list_tensor.append(DMT1_yl)
        for i in range(3):
            list_tensor.append(a[:, :, i, :, :])
        return torch.cat(list_tensor, 1)

    def forward(self, x):
        DMT1_yl, DMT1_yh = self.DWT(x)
        DMT1 = self._transformer(DMT1_yl, DMT1_yh)
        x1 = self.dconv_encode(DMT1)

        res = self.identety(x)
        out = torch.add(x1, res)
        return out