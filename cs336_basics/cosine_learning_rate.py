from math import cos, pi


def cosine_learning_rate(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:  # Warm-up phase
        return alpha_max * t / T_w
    if T_w <= t <= T_c:
        return (
            alpha_min
            + (1 + cos(pi * (t - T_w) / (T_c - T_w)))
            * (alpha_max - alpha_min)
            / 2
        )
    return alpha_min
