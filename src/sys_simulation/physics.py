import numpy as np


def s_to_p(x_0: float, s: np.ndarray, dt: float) -> np.ndarray:
    """
    Fonction to convert discret speed measurement to position

    :param x_0: intial position
    :param s: speed vector
    :param dt: sampling time
    :return: position vector
    """
    p = np.zeros_like(s)
    p[0] = x_0
    for i in range(1, len(s)):
        p[i] = p[i-1] + s[i-1] * dt
    return p
