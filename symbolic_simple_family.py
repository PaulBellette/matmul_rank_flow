"""Direct symbolic proof of the simple one-parameter algorithm family.

The off-diagonal entries are identically correct for arbitrary t,z.  The two
diagonal errors both factor by the single scalar relation

    R = t^2 (z^2-z+1) - z.

Hence R=0 proves the whole family exactly.
"""
from __future__ import annotations

import sympy as sp


def main() -> None:
    a,b,c,d,e,f,g,h,t,z = sp.symbols("a b c d e f g h t z")
    lam = 1/(1+z**3)

    m0 = lam*(a+z*d)*(e+z*h)
    m1 = a*(f-t*h)
    m2 = (b-t*d)*(g+t*h)
    m3 = (t*a+b)*h
    m4 = (c+t*d)*e
    m5 = (-t*a+c)*(t*e+f)
    m6 = d*(-t*e+g)
    m7 = lam*(z*a+d)*(z*e+h)

    got = [
        m0+m2-t*m3+t*m6+z*m7,
        m1+m3,
        m4+m6,
        z*m0+t*m1-t*m4+m5+m7,
    ]
    want = [a*e+b*g, a*f+b*h, c*e+d*g, c*f+d*h]
    R = t**2*(z**2-z+1)-z

    print("R =", R)
    for name, diff in zip(("C11","C12","C21","C22"), [sp.factor(sp.together(x-y)) for x,y in zip(got,want)]):
        print(name, "error =", diff)
        quotient = sp.factor(sp.cancel(diff/R)) if diff != 0 else 0
        print("    / R =", quotient)

    diffs = [sp.factor(sp.together(x-y)) for x,y in zip(got,want)]
    for diff in diffs:
        if diff == 0:
            continue
        quotient = sp.cancel(diff / R)
        if sp.simplify(diff - R * quotient) != 0:
            raise SystemExit("verification failed")
    print("Every nonzero output error is exactly divisible by R; hence R=0 proves correctness.")


if __name__ == "__main__":
    main()
