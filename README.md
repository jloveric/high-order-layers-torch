
<!---
[![CI](https://github.com/jloveric/high-order-layers-torch/actions/workflows/python-app.yml/badge.svg)](https://github.com/jloveric/high-order-layers-torch/actions/workflows/python-app.yml)
--->

# Piecewise Polynomial in PyTorch

This is a PyTorch implementation of my tensorflow [repository](https://github.com/jloveric/high-order-layers) and is more complete due to the flexibility of PyTorch.

Lagrange Polynomial, Piecewise Lagrange Polynomial, Discontinuous Piecewise Lagrange Polynomial, Fourier Series, sum and product layers in PyTorch.  The sparsity of using piecewise polynomial layers means that by adding new segments the representational power of your network increases, but the time to complete a forward step remains constant. Implementation includes simple fully connected layers, convolution layers and deconvolutional layers using these models. This is a PyTorch implementation of this [Discontinuous Piecewise Polynomial Neural Networks](https://www.researchgate.net/publication/276923198_Discontinuous_Piecewise_Polynomial_Neural_Networks) including huge number of extensions including continuous, Fourier series and convolutional neural networks... and many applications with varrying degrees of success.

## Collab Notebook
Using simple high order layers
[Simple function approximation](https://colab.research.google.com/drive/1kew0Kz4v5GB5D59-wP1rHZuCdhYknz4s?usp=sharing)

Using simple high order MLP
[2d function approximation](https://colab.research.google.com/drive/14wSNzBUFYk-1o6fuqiux_y33aV9VuwkF?usp=sharing)


## Idea

The idea is extremely simple, instead of a weight at the synapse we have a function F(x) that can be arbitrarily complex. As a practical matter I implement this by using multiple weights corresponding to each link, these weight are used as parameters of the function, and to make sure there is still some GPU efficiency, these weights are just coefficients of the basis functions.  In most of this work, the n-weights describe the value of a piecewise polynomial on a regular grid (in the case of a piecewise polynomial) each of the n-weights can be updated independently. A Lagrange polynomial and Gauss Lobatto points are used to minimize oscillations of the polynomial.  The same approach can be applied to any "functional" synapse, and I also have Fourier series synapses in this repo as well. Because the non-linearity is applied on the link, the node is simply a summation

In the image below each "link" instead of being a single weight, is a function of both x and a set of weights.  These functions can consist of an orthogonal basis functions for efficient approximation.

<img src="plots/NetworkZoom.png" width=50% height=50% style="display: block; margin: 0 auto">

A small layer then looks like this, the values at the nodes are just summed.

<img src="plots/PiecewisePolynomialLayer.svg" width=50% height=50% style="display: block; margin: 0 auto">

A single neuron input output pair with a piecewise function is shown below. In the case where we use polynomials, Lagrange polynomials are being used so the values of the weights are identical to the value of the function at that point. The spacing is determined by chebyshev lobatto points, so there are always weights at the edge of each segment. In the case of discontinuous polynomial, the weights there are 2 weights for each interior segment edge.

<img src="plots/NeuronDrawing.svg" width=50% height=50% style="display: block; margin: 0 auto">

The image below shows the function passing through the weights when using lagrange polynomials. Note that there is no derivative continuity at the boundaries.

<img src="plots/NeuronDrawingWeights.svg" width=50% height=50% style="display: block; margin: 0 auto">


## Why

Using higher order polynomial representations allow networks with much fewer total weights in certain cases. There is a
well known phenomena in numerical modeling known as exponential convergence using spectral methods when using hp refinement,
it's possible something like that can happen in neural networks as well.

## Is this a KAN?

Actually a single layer piecewise polynomial KAN (which is actually 2 layers) is a special case of a 2 layer
piecewise polynomial network, which is used in this repo. Therefore, a piecewise polynomial layer is actually "Half a KAN" so
it's actually simpler - Often all you need is a single polynomial layer at the input followed by a standard MLP so
having the piecewise polynomial layer is important. Other names that have been used in the past Deep FLANN
(functional link artificial neural network).

Lagrange polynomials are widely used in finite element analysis and have the advantage that the value of the weight
is actually the value of the function at that point in space. By limiting the weights you are limiting the maximum
value of the function (the function may be higher than the weights in between the nodes). Also, when you go beyond the
range of definition [-1,1] the polynomial is still defined using the last (or first) polynomial in the sequence, whether you want it defined that way at high polynomial order is another question. I mention a paper at the bottom where they do a linear
extension beyond the range [-1,1] so values do not rise too fast - but normalization works as well.

## Issues

What about instabilities due to steep gradients? Seems like you can get around those with various approaches, polynomial
refinement is one (start with piecewise linear and than increase the polynomial order after it converges), the lion
optimizer helps a lot as well.

The biggest issues I've experienced though are that it's slower than dense networks and certain operations can
take up more memory which can cause major issues with models that already push the limits of your gpu. Now that
KANs are popular, hopefully there will be enough people to address all these issues.

In general, with enough effort, it seems I can make them "work" for any place the classic ReLU network works and
in certain situations they clearly work much better. They also do a great job of overfitting, which just means,
I need more data. For problems where your inputs are positional, x and y..., they seem to be far better.

Finally, I believe these methods actually will benefit much more from (approximate) second order optimizers. I used those in
my original implementation. Although there are plenty of second order optimizers out there, to date, pytorch does
not have a standard one except LBFGS which has its own issues.

## Fully Connected Layer Types
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

## Convolutional Layer Types

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

## Initializing of layers
For non convolutional layers I've found that initializing the polynomials to continuous line across all segments, works better then a random wiggly polynomial. I don't have similar functions implemented for convolutional layers. Here is a function that does this initialization (it can be found in [networks.py](https://github.com/jloveric/high-order-layers-torch/blob/master/high_order_layers_torch/networks.py))
```
def initialize_network_polynomial_layers(
    network: nn.Module,
    max_slope: float,
    max_offset: float,
    scale_slope: Callable[[float], float] = lambda input_size: 1,
)
```

## h and p refinement
p refinement is taking an existing network and increasing the polynomial order of that network without changing the network output.  This allow the user to train a network at low polynomial order and then use that same network to initialize a network with higher polynomial order.  This is particularly useful since a high order polynomial network will often converge poorly without the right initialization, the lower order network provides a good initial solution.  The function for changing the order of a network is
```
from high_order_layers_torch.networks import interpolate_high_order_mlp
interpolate_high_order_mlp(
    network_in: HighOrderMLP, network_out: HighOrderMLP
```
current implementation only works with high order MLPs, not with convnets.  A similar function exists for h refinement.  h refinement is
refining the number of segments in a layer, and is used for similar reasoning.  Layers with lots of segments may be slow to converge
so the user starts with a small number of segments (1 or 2) and then increases the number of segments (h) using the lower initialization.  The following function currently only works for high order MLPs, not with convnets
```
from high_order_layers_torch.network import hp_refine_high_order_mlp
hp_refine_high_order_mlp(
    network_in: HighOrderMLP, network_out: HighOrderMLP
)
```
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
python examples/mnist.py max_epochs=1 train_fraction=0.1 layer_type=continuous2d n=4 segments=2
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
python examples/invariant_mnist.py max_epochs=100 train_fraction=1 mlp.layer_type=continuous mlp.n=5 mlp.p_refine=False mlp.hidden.layers=4
```
with polynomial refinement (p-refinement)
```
python examples/invariant_mnist.py max_epochs=100 train_fraction=1 layer_type=mlp.continuous mlp.n=2 mlp.target_n=5 mlp.p_refine=True
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

No real progress here
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
## A note on the product unit (I rarely use anymore)
The layers used here do not require additional activation functions and use a simple sum or product in place of the activation.
I almost always use sum units, but product units are performed in this manner

$$ product=-1+\prod_{i}(1 + f_{i})+(1-\alpha)\sum_{i}f_{i} $$

The 1 is added to each function output to as each of the sub products is also computed.  The linear part is controlled by
the alpha parameter.

## Notes on optimizer
The Lion optimizer seems to be the best choice since it performs better than Adam in general, but seems to work especially well
for the case of polynomials.

## Notes on normalization
Although you can use batchnorm, layernorm etc... work better, I've found that you can actually just use the infinity norm ("max_abs" norm) which has no parameters
for this formulation (same approach seems not to work very well for standard relu networks - but need to investigate this further).
The max_abs normalization is defined this way
```
normalized_x = x/(max(abs(x))+eps)
```
where the normalization is done per sample (as opposed to per batch).  The way the layers are formulated, we don't want the neuron
values to extend beyond [-1, 1] as the polynomial values grow rapidly beyond that range.  You can also use mirror periodicity to keep the
values within from growing rapidly. We want the values to cover the entire range [-1, 1] of the polynomials as the weights
are packed towards the edges of each segment (though using even number of segments means you'll have a lot of weights near the origin).


## Reference
You can refer to this repo here
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
and there is an old paper which was originally written in c++ and doesn't cover
nearly as much as this repo and all the dependent repos, which you can also refer
to
```
@article{loverich2015discontinuous,
  title={Discontinuous Piecewise Polynomial Neural Networks},
  author={Loverich, John},
  journal={arXiv preprint arXiv:1505.04211},
  year={2015}
}
```
## Notes

Recently the paper [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/pdf/2404.19756) was published (9 years after the original implementation of the technique in this repo), where B-splines were used on the grid. Looking at that repo, the real difference seems to be B-splines vs lagrange
polynomials.

This paper [Variations on the Chebyshev-Lagrange Activation Function](https://arxiv.org/abs/1906.10064) implements a linear extension 
to the values beyond [-1,1] so it might be interesting to implement