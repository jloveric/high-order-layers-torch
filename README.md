[![Build Status](https://travis-ci.org/jloveric/functional-layers.svg?branch=master)](https://travis-ci.org/jloveric/functional-layers)

# Functional Layers in PyTorch
This is a PyTorch implementation of my tensorflow [repository](https://github.com/jloveric/high-order-layers) and is more complete due to the flexibility of PyTorch.

Lagrange Polynomial, Piecewise Lagrange Polynomial, Discontinuous Piecewise Lagrange Polynomial, and Fourier Series layers in PyTorch.  The sparsity of using piecewise polynomial layers means that by adding new segments the computational power of your network increases, but the time to complete a forward step remains constant.  Implementation includes simple fully connected layers and convolution layers using these models.  More details to come.  This is a PyTorch implementation of this [paper](https://www.researchgate.net/publication/276923198_Discontinuous_Piecewise_Polynomial_Neural_Networks) including extension to Fourier Series and convolutional neural networks.

# Installing
```bash
pipenv install -e git+git://github.com/jloveric/high-order-layers-torch@master#egg=high_order_layers_torch
```
# Examples

## mnist
```python
python mnist.py max_epochs=1 train_fraction=0.1 layer_type=piecewise segments=2
```
## cifar100
```
python cifar100.py max_epochs=1 train_fraction=0.1 layer_type=piecewise segments=2 n=3
```