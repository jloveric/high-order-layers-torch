![Build Status](https://github.com/jloveric/high-order-layers-torch/actions/workflows/python-app.yml/badge.svg)

# Piecewise Polynomial and Fourier Layers in PyTorch

This is a PyTorch implementation of my tensorflow [repository](https://github.com/jloveric/high-order-layers) and is more complete due to the flexibility of PyTorch.

Lagrange Polynomial, Piecewise Lagrange Polynomial, Discontinuous Piecewise Lagrange Polynomial, Fourier Series, sum and product layers in PyTorch.  The sparsity of using piecewise polynomial layers means that by adding new segments the representational power of your network increases, but the time to complete a forward step remains constant.  Implementation includes simple fully connected layers, convolution layers and deconvolutional layers using these models.  This is a PyTorch implementation of this [paper](https://www.researchgate.net/publication/276923198_Discontinuous_Piecewise_Polynomial_Neural_Networks) including extension to Fourier Series and convolutional neural networks.

## Idea

The idea is extremely simple - instead of a single weight at the synapse, use n-weights.  The n-weights describe a piecewise polynomial (or other complex function) and each of the n-weights can be updated independently.  A Lagrange polynomial and Gauss Lobatto points are used to minimize oscillations of the polynomial.  The same approach can be applied to any "functional" synapse, and I also have Fourier series synapses in this repo as well.  This can be implemented as construction of a polynomial or Fourier kernel followed by a standard pytorch layer where a linear activation is used.

In the image below each "link" instead of being a single weight, is a function of both x and a set of weights.  These functions can consist of an orthogonal basis functions for efficient approximation.

<img src="plots/NetworkZoom.png" width=50% height=50% style="display: block; margin: 0 auto">

## Why

Using higher order polynomial representations might allow networks with much fewer total weights. In physics, higher order methods
can be much more efficient. Spectral and discontinuous galerkin methods are examples of this.  Note that a standard neural network with relu activations is piecewise linear.  Here there are no bias weights and the "non-linearity" is in the synapse. 

In addition, it's well known that the dendrites are also computational units in neurons, for example [Dendritic action potentials and computation in human layer 2/3 cortical neurons](https://science.sciencemag.org/content/367/6473/83) and this is a simple way to add more computational power into the artificial neural network model. In addition it's been shown that a single pyramidal has the same computational capacity as a 5 to 8 layer convolutional NN, [Single cortical neurons as deep artificial neural networks](https://www.sciencedirect.com/science/article/abs/pii/S0896627321005018?dgcid=author)

## A note on the unit
The layers used here do not require additional activation functions and use a simple sum or product in place of the activation.  Product is performed in this manner

$$ product=-1+\prod_{i}(1 + f_{i})+(1-\alpha)\sum_{i}f_{i} $$

The 1 is added to each function output to as each of the sub products is also computed.  The linear part is controlled by
the alpha parameter.

# Fully Connected Layer Types
All polynomials are Lagrange polynomials with Chebyshev interpolation points.

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

where `layer_type` is one of
| layer_type          | representation
|--------------------|-------------------------|
|continuous         |  piecewise polynomial using sum at the neuron |
|continuous_prod    |  piecewise polynomial using products at the neuron |
|discontinuous      |  discontinuous piecewise polynomial with sum at the neuron|
|discontinuous_prod | discontinous piecewise polynomial with product at the neuron|
|polynomial | single polynomial (non piecewise) with sum at the neuron|
|polynomial_prod | single polynomial (non piecewise) with product at the neuron|
|product | Product |
|fourier | fourier series with sum at the neuron |



`n` is the number of interpolation points per segment for polynomials or the number of frequencies for fourier series, `segments` is the number of segments for piecewise polynomials, `alpha` is used in product layers and when set to 1 keeps the linear part of the product, when set to 0 it subtracts the linear part from the product.

## Product Layers

Product layers

# Convolutional Layer Types

```python
conv_layer = high_order_convolution_layers(layer_type=layer_type, n=n, in_channels=3, out_channels=6, kernel_size=5, segments=segments, rescale_output=rescale_output, periodicity=periodicity)
```         

All polynomials are Lagrange polynomials with Chebyshev interpolation points.
| layer_type   | representation       |
|--------------|----------------------|
|continuous(1d,2d)   | piecewise continuous polynomial
|discontinuous(1d,2d) | piecewise discontinuous polynomial
|polynomial(1d,2d) | single polynomial
|fourier(1d,2d) | fourier series convolution

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

```python
python examples/function_example.py 
```

## XOR : 0.5 for x*y > 0 else -0.5
Simple XOR problem using the standard network structure (2 inputs 2 hidden 1 output) this will also work with no hidden layers. The function is discontinuous along the axis and we try and fit that function. Using piecewise discontinuous layers the model can match the function exactly.  
![piecewise discontinuous polynomial](plots/xor_discontinuous.png)
With piecewise continuous it doesn't work quite as well.
![piecewise continuous polynomial](plots/xor_continuous.png)
Polynomial doesn't work well at all (expected).
![polynomial](plots/xor_polynomial.png)

## MNIST (convolutional)

```python
python examples/mnist.py max_epochs=1 train_fraction=0.1 layer_type=continuous n=4 segments=2
```

## CIFAR100 (convolutional)

```
python examples/cifar100.py -m max_epochs=20 train_fraction=1.0 layer_type=polynomial segments=2 n=7 nonlinearity=False rescale_output=False periodicity=2.0 lr=0.001 linear_output=False
```

## Variational Autoencoder
Still a WIP.  Does work, but needs improvement.
```
python examples/variational_autoencoder.py -m max_epochs=300 train_fraction=1.0
```
run with nevergrad for parameter tuning
```
python examples/variational_autoencoder.py -m
```
## Invariant MNIST (fully connected)
Without polynomial refinement
```python
python examples/invariant_mnist.py max_epochs=100 train_fraction=1 layer_type=polynomial n=5 p_refine=False
```
with polynomial refinement (p-refinement)
```
python examples/invariant_mnist.py max_epochs=100 train_fraction=1 layer_type=continuous n=2 p_refine=False target_n=5 p_refine=True
```
I've also added hp refinement, but it needs a lot of testing.
## Implicit Representation

An example of implicit representation for image compression, language generation can be found [here](https://github.com/jloveric/high-order-implicit-representation).  I intend to explore generative models in natural language further [here](https://github.com/jloveric/language-interpolation)

## PDEs in Fluid Dynamics

An example using implicit representation to solve hyperbolic (nonlinear) wave equations can be found [here](https://github.com/jloveric/neural-network-pdes)

## Natural Language Generation

Examples using these networks for natural language generation can be found
[here](https://github.com/jloveric/language-interpolation)

## Generative music

Work in progress
[here](https://github.com/jloveric/high-order-generative-music)


## Test and Coverage
After installing and running
```
poetry shell
```
run
```
pytest 
```
for coverage, run
```
coverage run -m pytest
```
and then
```
coverage report
```
## Reference
```
@misc{Loverich2020,
  author = {Loverich, John},
  title = {High Order Layers Torch},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jloveric/high-order-layers-torch}},
}
```