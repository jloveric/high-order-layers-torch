import torch.nn as nn
import torch
from torch.autograd import Variable
from .LagrangePolynomial import *


class Polynomial(nn.Module):
    def __init__(self, n, in_features, out_features):
        super().__init__()
        self.poly=LagrangePoly(n)
        self.n=n

        self.w=torch.nn.Parameter(data = torch.Tensor(
            out_features, in_features*n), requires_grad = True)
        self.w.data.uniform_(-1, 1)

    def forward(self, x):
        # unfortunately we don't have automatic broadcasting yet
        # w = self.w.expand_as(x)
        fx=self.poly.interpolate(x, self.w)

        return fx

class PiecewisePolynomial(nn.Module):
    def __init__(self, n, in_features, out_features, segments):
        super().__init__()
        self._poly=LagrangePoly(n)
        self._n=n
        self._segments=segments
        self.in_features=in_features
        self.w=torch.nn.Parameter(data = torch.Tensor(
            out_features, in_features*((n-1)*segments+1)), requires_grad = True)
        self.w.data.uniform_(-1, 1)

    def forward(self, x):
        # get the segment index
        id_min=(((x+1.0)/2.0)*self._segments).long()
        id_max=id_min+1

        # determine which weights are active
        wid_min=(id_min*(self._n-1)).long()
        wid_max=(id_max*(self._n-1)).long()+1

        # get the range of x in this segment

        x_min=self._eta(id_min)
        x_max=self._eta(id_max)

        # rescale to -1 to +1
        x_in=2.0*((x-x_min)/(x_max-x_min))-1.0

        w_list=[]
        for i in range(list(x_in.size())[0]):
            id_1=wid_min[i].numpy()[0]
            id_2=wid_max[i].numpy()[0]
            w=self.w[:, id_1:id_2]
            w_list.append(w)

        w_in=torch.stack(w_list)
        fx=self._poly.interpolate(x_in, w_in)
        return fx

    def _eta(self, index):
        """
        Arg:
            - index is the segment index
        """
        eta=index/float(self._segments)
        return eta*2-1


class PiecewiseDiscontinuousPolynomial(nn.Module):
    def __init__(self, n, in_features, out_features, segments):
        super().__init__()
        self._poly=LagrangePoly(n)
        self._n=n
        self._segments=segments
        self.in_features=in_features
        self.w=torch.nn.Parameter(data = torch.Tensor(
            out_features, in_features*n*segments), requires_grad = True)
        self.w.data.uniform_(-1, 1)

    def forward(self, x):
        # determine which segment it is in
        id_min=(((x+1.0)/2.0)*self._segments).long()
        id_max=id_min+1

        # determine which weights are active
        wid_min=(id_min*self._n).long()
        wid_max=(id_max*self._n).long()

        # get the range of x in this segment
        x_min=self._eta(id_min)
        x_max=self._eta(id_max)

        # rescale to -1 to +1
        x_in=2.0*((x-x_min)/(x_max-x_min))-1.0
        w_list=[]
        for i in range(list(x_in.size())[0]):
            id_1=wid_min[i].numpy()[0]
            id_2=wid_max[i].numpy()[0]
            w=self.w[:, id_1:id_2]
            w_list.append(w)


        w_in=torch.stack(w_list)
        fx=self._poly.interpolate(x_in, w_in)

        return fx

    def _eta(self, index):
        eta=index/float(self._segments)
        return eta*2-1
