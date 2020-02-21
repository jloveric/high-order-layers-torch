import torch.nn as nn
from torch.autograd import Variable

class Polynomial(nn.Module):
    def __init__(self,basis):
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.c = nn.Parameter(torch.zeros(1))
        self.basis = basis

    def forward(self, x):
        # unfortunately we don't have automatic broadcasting yet
        a = self.a.expand_as(x)
        b = self.b.expand_as(x)
        c = self.c.expand_as(x)
        return a * torch.exp((x - b)^2 / c)

module = Gaussian()
x = Variable(torch.randn(20))
out = module(x)
loss = loss_fn(out)
loss.backward()