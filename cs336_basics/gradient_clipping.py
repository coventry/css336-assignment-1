from math import sqrt

EPS = 1e-6


def gradient_clip(parameters, max_l2_norm):
    "Clip gradients in `parameters`to `max_l2_norm`, as on p. 34"
    gradients = [p.grad for p in parameters if p.grad is not None]
    norm = sqrt(sum((g * g).sum() for g in gradients))
    if norm > max_l2_norm:
        for g in gradients:
            g *= max_l2_norm / (norm + EPS)
