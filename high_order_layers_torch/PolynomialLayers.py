import torch.nn as nn
from torch.autograd import Variable
from LagrangePolynomial import *

class Polynomial(nn.Module):
    def __init__(self,n, in_features, out_features):
        self.poly = LagrangePoly(n)
        self.n = n
        
        #self.w = nn.Parameter(torch.zeros(n))
        self.w = torch.nn.Parameter(data=torch.Tensor(out_features, in_features*n), requires_grad=True)
        self.w.data.uniform_(-1, 1)
        #self.reset_parameters()
        
    def forward(self, x):
        #unfortunately we don't have automatic broadcasting yet
        #w = self.w.expand_as(x)
        fx = self.poly.interpolate(x, self.w)

        return fx