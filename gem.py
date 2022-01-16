import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        # From the official repository
        # return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        # The author call that function with 
        # return LF.gem(x, p=self.p.unsqueeze(-1).unsqueeze(-1), eps=self.eps)
        # So, I split the function and make p = self.p.unsqueeze(-1).unsqueeze(-1)
        x = torch.clamp(x, min=self.eps)
        x = torch.pow(x, self.p.unsqueeze(-1).unsqueeze(-1))
        x = F.avg_pool2d(x, x.size(-2), x.size(-1))
        x = torch.pow(x, 1./self.p.unsqueeze(-1).unsqueeze(-1))
        return x