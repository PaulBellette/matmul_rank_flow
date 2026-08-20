"""SymPy certificate for the one-parameter schoolbook->fusion branch.

This script is intentionally independent of PyTorch.  It substitutes the
closed-form family into the nine reduced polynomial constraints and asks SymPy
to simplify each one to zero.
"""
from __future__ import annotations

import sympy as sp


def main() -> None:
    p = sp.symbols("p", nonnegative=True)
    plus = sp.sqrt(1 + 2 * p)
    minus = sp.sqrt(1 - 2 * p)

    A = (plus + minus) / 2
    F = (plus - minus) / 2
    B = sp.Integer(1)
    C = E = G = H = J = sp.Integer(0)
    D = sp.sqrt(p)
    I = sp.sqrt(1 - p)
    x = 1 / ((1 - p) * plus)
    y = 1 / (1 - p)

    e0 = A**3 * x - 3 * B * C**2 * y - 3 * D**2 * E * y + F**3 * x - 1
    e1 = -A**2 * G * x + B * C * I * y - B * C * J * y + C**2 * H * y - D**2 * H * y + D * E * I * y - D * E * J * y + F**2 * G * x
    e2 = A**2 * F * x + A * F**2 * x + 2 * B * C * D * y - B * D**2 * y - C**2 * E * y + 2 * C * D * E * y
    e3 = -A * G**2 * x + B * I**2 * y - 2 * C * H * J * y - 2 * D * H * I * y + E * J**2 * y - F * G**2 * x - 1
    e4 = A * G**2 * x + B * I * J * y - C * H * I * y + C * H * J * y + D * H * I * y - D * H * J * y + E * I * J * y + F * G**2 * x
    e5 = -A * G**2 * x + B * J**2 * y + 2 * C * H * I * y + 2 * D * H * J * y + E * I**2 * y - F * G**2 * x
    n0 = A**2 + 2 * G**2 + F**2 - 1
    n1 = B**2 + 2 * H**2 + E**2 - 1
    n2 = C**2 + D**2 + I**2 + J**2 - 1

    equations = [e0, e1, e2, e3, e4, e5, n0, n1, n2]
    simplified = [sp.simplify(e) for e in equations]

    print("A*F =", sp.simplify(A * F))
    print("A+F =", sp.simplify(A + F))
    for i, value in enumerate(simplified):
        print(f"constraint[{i}] = {value}")
    if any(v != 0 for v in simplified):
        raise SystemExit("symbolic verification failed")
    print("All nine reduced constraints simplify exactly to zero.")


if __name__ == "__main__":
    main()
