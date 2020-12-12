import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import init


class Product(Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, alpha=1.0, **kwargs) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.alpha = alpha
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        assemble = torch.einsum("ij,kj->ijk", x, self.weight)
        this_sum = torch.sum(assemble, dim=1)
        assemble = assemble+1.0
        assemble = torch.prod(assemble, dim=1)-(1-self.alpha)*this_sum
        assemble = assemble-1+self.bias
        return assemble
