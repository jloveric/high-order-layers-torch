[![Build Status](https://travis-ci.org/jloveric/high-order-layers-torch.svg?branch=master)](https://travis-ci.org/jloveric/high-order-layers-torch)

# Functional Layers in PyTorch
This is a PyTorch implementation of my tensorflow [repository](https://github.com/jloveric/high-order-layers) and is more complete due to the flexibility of PyTorch.

Lagrange Polynomial, Piecewise Lagrange Polynomial, Discontinuous Piecewise Lagrange Polynomial, Fourier Series, sum and product layers in PyTorch.  The sparsity of using piecewise polynomial layers means that by adding new segments the computational power of your network increases, but the time to complete a forward step remains constant.  Implementation includes simple fully connected layers and convolution layers using these models.  More details to come.  This is a PyTorch implementation of this [paper](https://www.researchgate.net/publication/276923198_Discontinuous_Piecewise_Polynomial_Neural_Networks) including extension to Fourier Series and convolutional neural networks.

The layers used here do not require additional activation functions and use a simple sum or product in place of the activation.  Product is performed in this manner...

# Fully Connected Layer Types
Lagrange Polynomial Chebyshev Points
Piecewise Continuous Lagrange Polynomial Chebyshev Points
Piecewise Discontinuous Lagrange Polynomial Chebyshev Points
Piecewise Discontinuous Lagrange Polynomial Chebyshev Points
Fourier Series

A helper function is provided in selecting and switching between these layers
```python
from high_order_layers_torch.layers import *
layer1 = high_order_fc_layers(
    layer_type=layer_type,
    n=n, 
    in_features=784,
    out_features=100,
    segments=segments,
    alpha=linear_part
)
```
where `layer_type` is on of 
```
"continuous" -> PiecewisePolynomial,
"continuous_prod" -> PiecewisePolynomialProd,
"discontinuous" -> PiecewiseDiscontinuousPolynomial,
"discontinuous_prod" -> PiecewiseDiscontinuousPolynomialProd,
"polynomial"-> Polynomial,
"polynomial_prod"-> PolynomialProd,
"product"-> Product,
"fourier"-> FourierSeries
```
`n` is the number of interpolation points per segment (if there are any), `segments` is the number of segments for piecewise polynomials, `alpha` is used in product layers and when set to 1 keeps the linear part of the product, when set to 0 it subtracts the linear part from the product.
## Product Layers
Product layers 

# Convolutional Layer Types
Lagrange Polynomial Chebyshev Points
Piecewise Continuous Lagrange Polynomial Chebyshev Points
Piecewise Discontinuous Lagrange Polynomial Chebyshev Points
Piecewise Discontinuous Lagrange Polynomial Chebyshev Points
Fourier Series

# Installing
## Installing locally
This repo uses poetry, so run
```
poetry install
```
and then
```
poetry shell
```

## Installing from pypi
```bash
pip install high-order-layers-torch
```
or
```
poetry add high-order-layers-torch
```

# Examples

## Simple function approximation
Approximating a simple function using a single input and single output (single layer) with no hidden layers
to approximate a function using continuous and discontinuous piecewise polynomials (with 5 pieces) and simple
polynomials and fourier series.  The standard approach using ReLU is non competitive.  To see more complex see
the implicit representation page [here](https://github.com/jloveric/high-order-implicit-representation).

![piecewise continuous polynomial](plots/piecewise_continuous.png)
![piecewise discontinuous polynomial](plots/piecewise_discontinuous.png)
![polynomial](plots/polynomial.png)
![fourier series](plots/fourier_series.png)



## mnist (convolutional)
```python
python mnist.py max_epochs=1 train_fraction=0.1 layer_type=continuous n=4 segments=2
```
## cifar100 (convolutional)
```
python cifar100.py -m max_epochs=20 train_fraction=1.0 layer_type=polynomial segments=2 n=7 nonlinearity=False rescale_output=False periodicity=2.0 lr=0.001 linear_output=False
```
## invariant mnist (fully connected)
```python
python invariant_mnist.py max_epochs=100 train_fraction=1 layer_type=polynomial n=5
```
Constructing the network
```
self.layer1 = high_order_fc_layers(
    layer_type=cfg.layer_type, n=cfg.n, in_features=784, out_features=100, segments=cfg.segments, alpha=cfg.linear_part)
self.layer2 = nn.LayerNorm(100)
self.layer3 = high_order_fc_layers(
    layer_type=cfg.layer_type, n=cfg.n, in_features=100, out_features=10, segments=cfg.segments, alpha=cfg.linear_part)
self.layer4 = nn.LayerNorm(10)
```
## Implicit Representation
An example of implicit representation can be found [here](https://github.com/jloveric/high-order-implicit-representation)
