import torch.nn as nn
import torch
from torch.autograd import Variable
from .LagrangePolynomial import *


class Polynomial(nn.Module):
    def __init__(self, n, in_features, out_features):
        super(Polynomial, self).__init__()
        self.poly = LagrangePoly(n)
        self.n = n

        #self.w = nn.Parameter(torch.zeros(n))
        self.w = torch.nn.Parameter(data=torch.Tensor(
            out_features, in_features*n), requires_grad=True)
        self.w.data.uniform_(-1, 1)
        # self.reset_parameters()

    def forward(self, x):
        # unfortunately we don't have automatic broadcasting yet
        #w = self.w.expand_as(x)
        fx = self.poly.interpolate(x, self.w)

        return fx


class PiecewiseDiscontinuousPolynomial(nn.Module):
    def __init__(self, n, in_features, out_features, segments):
        super(PiecewiseDiscontinuousPolynomial, self).__init__()
        self._poly = LagrangePoly(n)
        self._n = n
        self._segments = segments
        self.in_features = in_features
        #self.w = nn.Parameter(torch.zeros(n))
        self.w = torch.nn.Parameter(data=torch.Tensor(
            in_features, in_features*n*segments), requires_grad=True)
        self.w.data.uniform_(-1, 1)
        # self.reset_parameters()

    def forward(self, x):
        # determine which segment it is in
        #print('x', x)

        id_min = (((x+1.0)/2.0)*self._segments).long()
        id_max = id_min+1
        # print('id_min',id_min,'id_max',id_max)

        # if x is out of bounds, interpolate based on the nearest bin

        '''
        id_min = torch.where(x > 1.0, self._segments-1+0*id_min, id_min)
        id_max = torch.where(x > 1.0, self._segments+0*id_max, id_max)
        id_min = torch.where(x < -1.0, 0*id_min, id_min)
        id_max = torch.where(x < -1.0, 1+0*id_max, id_max)
        '''

        #print('id_min', id_min, 'id_max', id_max)

        # determine which weights are active
        wid_min = (id_min*self._n).long()
        wid_max = (id_max*self._n).long()

        # get the range of x in this segment

        x_min = self._eta(id_min)
        x_max = self._eta(id_max)

        # rescale to -1 to +1
        x_in = 2.0*((x-x_min)/(x_max-x_min))-1.0
        #print('x_in', x_in)
        #x_in = x
        # print('x_min',x_min,'x_max',x_max,'x_in',x_in)

        # print('w',self.w.size())
        # print('wid_min',wid_min,'wid_max',wid_max)
        #print('x.size()',x.size())
        w_list = []
        for i in range(list(x_in.size())[0]):
            #print('i', i)
            #print('wid', wid_min[i], wid_max[i])
            id_1 = wid_min[i].numpy()[0]
            id_2 = wid_max[i].numpy()[0]
            w = self.w[:,id_1:id_2]
            #print('w.shape', w.size(), id_1, id_2)
            # TODO: clone doesn't actually seem necessary here.
            w_list.append(w)
            # print('w_in',self.w_in.size())
        #print('w_list', w_list)
        #print('xs', list(x.size())[-1])
        w_in = torch.stack(w_list)
        #print('w_in.size', w_in.size())
        # print('w_in.shape', w_in.size(), 'self.w.shape',
        #      self.w.size(), 'x_in.size', x_in.size())
        #print('x', x, 'x_in', x_in, 'w_in', w_in)
        fx = self._poly.interpolate(x_in, w_in)
        #print('fx', fx)
        return fx

    def _eta(self, index):
        eta = index/float(self._segments)
        return eta*2-1
