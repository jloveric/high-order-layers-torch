from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer

# Took this code from lucid rains version and updated for the sparse case


def exists(val):
    return val is not None


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2, update_indexes: torch.Tensor):
    """
    TODO: I can probably do this more efficiently
    :param update_indexes: True/False array indicating which indexes should be updated
    """

    p.data[update_indexes].mul_(1 - lr * wd)

    # weight update
    grad_update = grad[update_indexes]
    update = (
        exp_avg[update_indexes]
        .clone()
        .mul_(beta1)
        .add(grad_update, alpha=1 - beta1)
        .sign_()
    )
    p[update_indexes] = p[update_indexes].add_(update, alpha=-lr)

    # decay the momentum running average coefficient
    exp_avg[update_indexes] = (
        exp_avg[update_indexes].mul_(beta2).add_(grad_update, alpha=1 - beta2)
    )


class SparseLion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)

        super().__init__(params, defaults)

        self.update_fn = update_fn

        if use_triton:
            from lion_pytorch.triton import update_fn as triton_update_fn

            self.update_fn = triton_update_fn

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad, lr, wd, beta1, beta2, state = (
                    p.grad,
                    group["lr"],
                    group["weight_decay"],
                    *group["betas"],
                    self.state[p],
                )

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                # In each layer only a fraction of each path is actually updated. In my
                # case the gradients will be identically zero for those that should
                # not be updated.
                update_indexes = p.grad != 0.0

                exp_avg = state["exp_avg"]

                self.update_fn(p, grad, exp_avg, lr, wd, beta1, beta2, update_indexes)

        return loss
