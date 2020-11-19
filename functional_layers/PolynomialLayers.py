import torch.nn as nn
import torch
from torch.autograd import Variable
from .LagrangePolynomial import *


class Function(nn.Module):
    def __init__(self, n, in_features, out_features, basis):
        super().__init__()
        self.poly = basis
        self.n = n

        self.w = torch.nn.Parameter(data=torch.Tensor(
            out_features, in_features, n), requires_grad=True)
        self.w.data.uniform_(-1, 1)

        self.sum = torch.nn.Parameter(
            data=torch.Tensor(out_features), requires_grad=True)
        self.prod = torch.nn.Parameter(
            data=torch.Tensor(out_features), requires_grad=True)
        self.sum.data.uniform_(-1, 1)
        self.prod.data.uniform_(-1, 1)

    def forward(self, x):
        fsum, fprod = self.poly.interpolate(x, self.w)
        return fsum*self.sum+fprod*self.prod


class Polynomial(Function):
    def __init__(self, n, in_features, out_features):
        return super().__init__(n, in_features, out_features, LagrangePolyFlat(n))


class FourierSeries(Function):
    def __init__(self, n, in_features, out_features):
        return super().__init__(n, in_features, out_features, FourierSeriesFlat(n))


class PiecewisePolynomial(nn.Module):
    def __init__(self, n, in_features, out_features, segments):
        super().__init__()
        self._poly = LagrangePoly(n)
        self._n = n
        self._segments = segments
        self.in_features = in_features
        self.out_features = out_features
        self.w = torch.nn.Parameter(data=torch.Tensor(
            out_features, in_features, ((n-1)*segments+1)), requires_grad=True)
        self.w.data.uniform_(-1, 1)
        self.sum = torch.nn.Parameter(
            data=torch.Tensor(out_features), requires_grad=True)
        self.prod = torch.nn.Parameter(
            data=torch.Tensor(out_features), requires_grad=True)
        self.sum.data.uniform_(-1, 1)
        self.prod.data.uniform_(-1, 1)
        self.wrange = None

    """
    def build_wrange(self, wid_min_flat, wid_max_flat) :
        if self.wrange is None :
            self.wrange = []
            for i in range(wid_min_flat.shape[0]):
                self.wrange.append(torch.arange(
                    wid_min_flat[i], wid_max_flat[i], device=device))
        else :
            for i in range(wid_min_flat.shape[0]):
                self.wrange[i]=wid_min_flat[i]:wid_max_flat[i]
    """

    def forward(self, x):
        # get the segment index
        id_min = (((x+1.0)/2.0)*self._segments).long()
        device = id_min.device
        id_min = torch.where(id_min <= self._segments-1, id_min,
                             torch.tensor(self._segments-1, device=device))
        id_min = torch.where(id_min >= 0, id_min,
                             torch.tensor(0, device=device))
        id_max = id_min+1

        # determine which weights are active
        wid_min = (id_min*(self._n-1)).long()
        wid_max = (id_max*(self._n-1)).long()+1

        # Fill in the ranges
        wid_min_flat = wid_min.view(-1)
        wid_max_flat = wid_max.view(-1)
        #print('wid_min.shape', wid_min.shape)
        wrange = wid_min_flat.unsqueeze(-1)+torch.arange(self._n, device=device).view(-1)
        
        windex = (torch.arange(
            wrange.shape[0]*wrange.shape[1])//self._n) % (self.in_features)
        wrange = wrange.flatten()

        w = self.w[:, windex, wrange]
        
        # TODO: verify this is correct or that it actually matters how it's reshaped.
        w = w.view(self.out_features, -1, self.in_features, self._n)
        w = w.permute(1,2,0,3)

        # get the range of x in this segment
        x_min = self._eta(id_min)
        x_max = self._eta(id_max)

        # rescale to -1 to +1
        x_in = 2.0*((x-x_min)/(x_max-x_min))-1.0

        fsum, fprod = self._poly.interpolate(x_in, w)
        res = fsum*self.sum+fprod*self.prod
        return res

    def _eta(self, index):
        """
        Arg:
            - index is the segment index
        """
        eta = index/float(self._segments)
        return eta*2-1


class PiecewiseDiscontinuousPolynomial(nn.Module):
    def __init__(self, n, in_features, out_features, segments):
        super().__init__()
        self._poly = LagrangePoly(n)
        self._n = n
        self._segments = segments
        self.in_features = in_features
        self.out_features = out_features
        self.w = torch.nn.Parameter(data=torch.Tensor(
            out_features, in_features, n*segments), requires_grad=True)
        self.w.data.uniform_(-1, 1)

        self.sum = torch.nn.Parameter(
            data=torch.Tensor(out_features), requires_grad=True)
        self.prod = torch.nn.Parameter(
            data=torch.Tensor(out_features), requires_grad=True)
        self.sum.data.uniform_(-1, 1)
        self.prod.data.uniform_(-1, 1)

    def forward(self, x):
        # determine which segment it is in
        id_min = (((x+1.0)/2.0)*self._segments).long()
        device = id_min.device
        id_min = torch.where(id_min <= self._segments-1, id_min,
                             torch.tensor(self._segments-1, device=device))
        id_min = torch.where(id_min >= 0, id_min,
                             torch.tensor(0, device=device))
        id_max = id_min+1

        # determine which weights are active
        wid_min = (id_min*self._n).long()
        wid_max = (id_max*self._n).long()

        # Fill in the ranges
        wid_min_flat = wid_min.flatten()
        wid_max_flat = wid_max.flatten()

        # get the range of x in this segment
        x_min = self._eta(id_min)
        x_max = self._eta(id_max)

        # rescale to -1 to +1
        x_in = 2.0*((x-x_min)/(x_max-x_min))-1.0
        w_list = []

        wrange = wid_min_flat.unsqueeze(-1)+torch.arange(self._n, device=device).view(-1)

        # should be size batches*inputs*n
        windex = (torch.arange(
            wrange.shape[0]*wrange.shape[1])//self._n) % self.in_features
        wrange = wrange.flatten()

        w = self.w[:, windex, wrange]

        w = w.view(self.out_features, -1, self.in_features, self._n)
        w = w.permute(1,2,0,3)


        fsum, fprod = self._poly.interpolate(x_in, w)
        return fsum*self.sum+fprod*self.prod

    def _eta(self, index):
        eta = index/float(self._segments)
        return eta*2-1
