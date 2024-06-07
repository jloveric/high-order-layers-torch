#!/usr/bin/sh
python3 examples/invariant_mnist.py -m mlp.n=2,3,4,5,6 mlp.hidden.width=128 mlp.layer_type=polynomial optimizer=sophia,lion mlp.normalize=max_abs,layer_norm periodicity=null
python3 examples/invariant_mnist.py -m mlp.n=2,3,4,5,6 mlp.hidden.width=128 mlp.layer_type=continuous mlp.segments=2 optimizer=sophia,lion mlp.normalize=max_abs,layer_norm periodicity=null
python3 examples/invariant_mnist.py -m mlp.n=2,3,4,5,6 mlp.hidden.width=128 mlp.layer_type=discontinuous mlp.segments=2 optimizer=sophia,lion mlp.normalize=max_abs,layer_norm periodicity=null
python3 examples/invariant_mnist.py -m mlp.n=2,3,4,5,6 mlp.hidden.width=128 mlp.layer_type=fourier mlp.segments=2 optimizer=sophia,lion mlp.normalize=max_abs,layer_norm periodicity=null
