import math


def get_meler_nodes(degree):
    return [math.cos((2 * k - 1) / 2 / degree * math.pi) for k in range(1, degree + 1)]


def get_meler_coeffs(degree):
    coef = math.pi / degree
    return [coef for _ in range(degree)]
