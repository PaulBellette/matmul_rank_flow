from __future__ import annotations

import argparse
import math


def z_from_t(t: float) -> float:
    """Companion diagonal-mixing parameter, stable on t in [0,1].

    It is the small real root of

        t^2 (z^2 - z + 1) = z.

    The rationalized quadratic formula avoids cancellation near t=0.
    """
    if t < -1.0e-14 or t > 1.0 + 1.0e-14:
        raise ValueError("t must lie in [0,1]")
    t = min(1.0, max(0.0, float(t)))
    if t == 0.0:
        return 0.0
    disc = math.sqrt(max(0.0, (1.0 - t * t) * (1.0 + 3.0 * t * t)))
    return 2.0 * t * t / (1.0 + t * t + disc)


def parameters(t: float) -> tuple[float, float]:
    z = z_from_t(t)
    lam = 1.0 / (1.0 + z**3)
    return z, lam


def multiply(A, B, t: float):
    """Exact one-parameter eight-product family for 2x2 matrices.

    A and B are nested length-2 sequences.  For t=0 this is schoolbook.
    For t=1 the first and last multiplication are identical and may be fused,
    leaving a seven-product Strassen algorithm.
    """
    z, lam = parameters(t)
    a, b = A[0]
    c, d = A[1]
    e, f = B[0]
    g, h = B[1]

    # Two diagonal channels.
    m0 = lam * (a + z * d) * (e + z * h)
    m7 = lam * (z * a + d) * (z * e + h)

    # Six channels whose amplitudes have been absorbed into their factor gauge.
    m1 = a * (f - t * h)
    m2 = (b - t * d) * (g + t * h)
    m3 = (t * a + b) * h
    m4 = (c + t * d) * e
    m5 = (-t * a + c) * (t * e + f)
    m6 = d * (-t * e + g)

    c11 = m0 + m2 - t * m3 + t * m6 + z * m7
    c12 = m1 + m3
    c21 = m4 + m6
    c22 = z * m0 + t * m1 - t * m4 + m5 + m7
    return [[c11, c12], [c21, c22]]


def formulas(t_symbol: str = "t", z_symbol: str = "z") -> str:
    t = t_symbol
    z = z_symbol
    return f"""Let A=[[a,b],[c,d]], B=[[e,f],[g,h]].
Choose 0 <= {t} <= 1 and {z} satisfying
    {t}^2 ({z}^2 - {z} + 1) = {z}.
Set lambda = 1/(1+{z}^3).

Eight scalar multiplications:
    m0 = lambda (a + {z} d)(e + {z} h)
    m1 = a(f - {t} h)
    m2 = (b - {t} d)(g + {t} h)
    m3 = ({t} a + b)h
    m4 = (c + {t} d)e
    m5 = (-{t} a + c)({t} e + f)
    m6 = d(-{t} e + g)
    m7 = lambda ({z} a + d)({z} e + h)

Recombine:
    C11 = m0 + m2 - {t} m3 + {t} m6 + {z} m7
    C12 = m1 + m3
    C21 = m4 + m6
    C22 = {z} m0 + {t} m1 - {t} m4 + m5 + m7

At {t}=0, {z}=0: schoolbook.
At {t}=1, {z}=1: m0 and m7 are half-copies of the same Strassen product and fuse.
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Simple exact schoolbook-to-Strassen algorithm family")
    p.add_argument("--t", type=float, default=0.5)
    args = p.parse_args()
    z, lam = parameters(args.t)
    print(formulas())
    print(f"For t={args.t:.16g}: z={z:.16g}, lambda={lam:.16g}")


if __name__ == "__main__":
    main()
