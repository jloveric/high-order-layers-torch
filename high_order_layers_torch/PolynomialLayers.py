import torch.nn as nn
import torch
from torch.autograd import Variable
from .LagrangePolynomial import *
from .utils import *


class Function(nn.Module):
    def __init__(self, n, in_features, out_features, basis, weight_magnitude: float = 1.0, periodicity: float = None, **kwargs):
        super().__init__()
        self.poly = basis
        self.n = n
        self.periodicity = periodicity
        self.w = torch.nn.Parameter(data=torch.Tensor(
            out_features, in_features, n), requires_grad=True)
        self.w.data.uniform_(-weight_magnitude/in_features,
                             weight_magnitude/in_features)

        self.result = torch.nn.Parameter(
            data=torch.Tensor(out_features), requires_grad=True)

    def forward(self, x):
        periodicity = self.periodicity
        if periodicity is not None:
            x = make_periodic(x, periodicity)

        result = self.poly.interpolate(x, self.w)

        return result


class Polynomial(Function):
    def __init__(self, n, in_features, out_features, length: float = 2.0, **kwargs):
        return super().__init__(n, in_features, out_features, LagrangePolyFlat(n, length=length), **kwargs)


class PolynomialProd(Function):
    def __init__(self, n, in_features, out_features, length: float = 2.0, **kwargs):
        return super().__init__(n, in_features, out_features, LagrangePolyFlatProd(n, length=length), **kwargs)


class FourierSeries(Function):
    def __init__(self, n: int, in_features: int, out_features: int, length: float = 2.0, **kwargs):
        return super().__init__(n, in_features, out_features, FourierSeriesFlat(n, length=length))


class Piecewise(nn.Module):
    def __init__(self, n, in_features, out_features, segments, length: int = 2.0, weight_magnitude=1.0, poly=None, periodicity=None, **kwargs):
        super().__init__()
        self._poly = poly(n)
        self._n = n
        self._segments = segments
        self.in_features = in_features
        self.out_features = out_features
        self.periodicity = periodicity
        self.w = torch.nn.Parameter(data=torch.Tensor(
            out_features, in_features, ((n-1)*segments+1)), requires_grad=True)
        self.w.data.uniform_(-weight_magnitude/in_features,
                             weight_magnitude/in_features)
        self.wrange = None
        self._length = length
        self._half = 0.5*length

    def forward(self, x):

        periodicity = self.periodicity
        if periodicity is not None:
            x = make_periodic(x, periodicity)

        # get the segment index
        id_min = (((x+self._half)/self._length)*self._segments).long()
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
        wrange = wid_min_flat.unsqueeze(-1) + \
            torch.arange(self._n, device=device).view(-1)

        windex = (torch.arange(
            wrange.shape[0]*wrange.shape[1])//self._n) % (self.in_features)
        wrange = wrange.flatten()

        w = self.w[:, windex, wrange]

        w = w.view(self.out_features, -1, self.in_features, self._n)
        w = w.permute(1, 2, 0, 3)

        # get the range of x in this segment
        x_min = self._eta(id_min)
        x_max = self._eta(id_max)

        # rescale to -1 to +1
        x_in = self._length*((x-x_min)/(x_max-x_min))-self._half

        result = self._poly.interpolate(x_in, w)
        return result

    def _eta(self, index):
        """
        Arg:
            - index is the segment index
        """
        eta = index/float(self._segments)
        return eta*2-1


class PiecewisePolynomial(Piecewise):
    def __init__(self, n, in_features, out_features, segments, length=2.0, weight_magnitude=1.0, periodicity: float = None, **kwargs):
        super().__init__(n, in_features, out_features, segments,
                         length, weight_magnitude, poly=LagrangePoly, periodicity=periodicity)


class PiecewisePolynomialProd(Piecewise):
    def __init__(self, n, in_features, out_features, segments, length=2.0, weight_magnitude=1.0, periodicity: float = None, **kwargs):
        super().__init__(n, in_features, out_features, segments,
                         length, weight_magnitude, poly=LagrangePolyProd, periodicity=periodicity)


class PiecewiseDiscontinuous(nn.Module):
    def __init__(self, n, in_features, out_features, segments, length=2.0, weight_magnitude=1.0, poly=None, periodicity: float = None, ** kwargs):
        super().__init__()
        self._poly = poly(n)
        self._n = n
        self._segments = segments
        self.in_features = in_features
        self.out_features = out_features
        self.periodicity = periodicity
        self.w = torch.nn.Parameter(data=torch.Tensor(
            out_features, in_features, n*segments), requires_grad=True)
        self.w.data.uniform_(-1/in_features, 1/in_features)

        self._length = length
        self._half = 0.5*length

    def forward(self, x):
        periodicity = self.periodicity
        if periodicity is not None:
            x = make_periodic(x, periodicity)

        # determine which segment it is in
        id_min = (((x+self._half)/self._length)*self._segments).long()
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
        x_in = self._length*((x-x_min)/(x_max-x_min))-self._half
        w_list = []

        wrange = wid_min_flat.unsqueeze(-1) + \
            torch.arange(self._n, device=device).view(-1)

        # should be size batches*inputs*n
        windex = (torch.arange(
            wrange.shape[0]*wrange.shape[1])//self._n) % self.in_features
        wrange = wrange.flatten()

        w = self.w[:, windex, wrange]

        w = w.view(self.out_features, -1, self.in_features, self._n)
        w = w.permute(1, 2, 0, 3)

        result = self._poly.interpolate(x_in, w)
        return result

    def _eta(self, index):
        eta = index/float(self._segments)
        return eta*2-1


class PiecewiseDiscontinuousPolynomial(PiecewiseDiscontinuous):
    def __init__(self, n, in_features, out_features, segments, length=2.0, weight_magnitude=1.0, periodicity: float = None, **kwargs):
        super().__init__(n, in_features, out_features, segments,
                         length, weight_magnitude, poly=LagrangePoly, periodicity=periodicity)


class PiecewiseDiscontinuousPolynomialProd(PiecewiseDiscontinuous):
    def __init__(self, n, in_features, out_features, segments, length=2.0, weight_magnitude=1.0, periodicity: float = None, **kwargs):
        super().__init__(n, in_features, out_features, segments,
                         length, weight_magnitude, poly=LagrangePolyProd, periodicity=periodicity)
