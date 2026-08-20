"""Implementation of AdamW optimizer

Response to `adamw` problem of Assignment 1.

"""

from math import sqrt

from torch import optim, zeros_like
from torch.optim.optimizer import ParamsT


class AdamW(optim.Optimizer):

    def __init__(
        self,
        params: ParamsT,  # Params to be optimized
        alpha: float,  # Learning rate
        beta_1: float,  # EMA retention coefficient for gradient estimate
        beta_2: float,  # EMA retention coefficient for velocity-squared
        epsilon: float,  # numerical-stability coefficient
        lambda_: float,  # weight-decay coefficient
    ):
        c = "must be positive, got"
        if alpha <= 0:
            raise ValueError(f"learning rate alpha {c} {alpha}")
        if beta_1 <= 0:
            raise ValueError(f"gradient EMA coeff beta_1 {c} {beta_1}")
        if beta_2 <= 0:
            raise ValueError(f"v^2 EMA coeff beta_2 {c} {beta_2}")
        if epsilon <= 0:
            raise ValueError(f"numerical stability coeff epsilon {c} {epsilon}")
        if lambda_ <= 0:
            raise ValueError(f"weight-decay coeff lambda_ {c} {lambda_}")

        self.alpha, self.beta_1, self.beta_2 = alpha, beta_1, beta_2
        self.epsilon, self.lambda_ = epsilon, lambda_
        defaults = dict(
            alpha=alpha,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            lambda_=lambda_,
        )
        super().__init__(params, defaults)

    # Seems to be a bug in torch's typing of the base `step` method.
    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, closure=None
    ) -> None:
        if closure is not None:
            raise ValueError("Don't know how to handle the closure")
        for g in self.param_groups:
            alpha, beta_1, beta_2 = g["alpha"], g["beta_1"], g["beta_2"]
            epsilon, lambda_ = g["epsilon"], g["lambda_"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                s = self.state[p]
                t = s.get("t", 1)  # Training time-step, initial step 1
                m = s.get("m", zeros_like(p))  # "Momentum"
                v = s.get("v", zeros_like(p))  # "velocity-squared"
                g = p.grad.data  # Line 6 of Algorithm 1
                # Line 7
                alpha_t = alpha * sqrt(1 - beta_2**t) / (1 - beta_1**t)
                theta = p.data
                theta -= alpha * lambda_ * theta  # Line 8
                m = beta_1 * m + (1 - beta_1) * g  # Line 9
                v = beta_2 * v + (1 - beta_2) * (g * g)  # Line 10
                theta -= alpha_t * m / (v.sqrt() + epsilon)  # Line 11
                s["t"] = t + 1
                s["m"] = m
                s["v"] = v
