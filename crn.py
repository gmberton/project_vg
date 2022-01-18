import torch
import torch.nn as nn
import torch.nn.functional as F


class CRN(nn.Module):
    """CRN layer implementation"""

    def __init__(self, args, dim=256):
        """
        Args:

        """
        super(CRN, self).__init__()
        self.dim = dim
        self.args = args
        self.last_reweight_mask = None

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(dim, 32, 3, device=self.args.device,
                      padding=1, padding_mode="reflect"),
            nn.ReLU()
        )

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(dim, 32, 5, device=self.args.device,
                      padding=2, padding_mode="reflect"),
            nn.ReLU()
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(dim, 20, 7, device=self.args.device,
                      padding=3, padding_mode="reflect"),
            nn.ReLU()
        )

        self.convw = nn.Sequential(
            nn.Conv2d(84, 1, 1, device=self.args.device),
            nn.ReLU()
        )

    def get_last_attention_mask(self):
        return self.last_reweight_mask

    def forward(self, x):
        downsampled_x = F.interpolate(x, (13, 13))

        a = self.conv3x3(downsampled_x)
        b = self.conv5x5(downsampled_x)
        c = self.conv7x7(downsampled_x)

        g_out = torch.cat((a, b, c), dim=1)

        w_out = self.convw(g_out)

        reweight_mask = F.interpolate(w_out, x.shape[2:])
        reweight_mask = torch.flatten(reweight_mask, 2)

        self.last_reweight_mask = reweight_mask

        return reweight_mask
