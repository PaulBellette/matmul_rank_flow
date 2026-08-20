"""Tiny parsing helpers only; no matmul_rank_flow imports."""
from fractions import Fraction

def q(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x, 1)
    return Fraction(str(x))
