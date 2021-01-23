import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))

class new(nn.Module):
    def __init__(self,inplace=True):
        super().__init__()

    def forward(self, x):
        return (F.softplus(2*x)-0.6931471805599453)/2  # ln2

def lncosh(x):
    ln2 = 0.6931471805599453
    return (2*x+torch.log(1+torch.exp(-2*x))-ln2)/2

def ori(x):
    return torch.log(torch.exp(x)*torch.cosh(x))/2