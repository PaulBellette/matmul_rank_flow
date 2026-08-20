#!/usr/bin/env python3
"""Independent exact operational verifier using SymPy's AlgebraicField implementation."""
from __future__ import annotations
import argparse, json, random
from fractions import Fraction
from pathlib import Path
import sympy as sp
from sympy.polys.domains import QQ


def frac(s): return Fraction(str(s))
def q_to_sympy(q): q=frac(q); return sp.Rational(q.numerator,q.denominator)

def load(path):
    d=json.loads(Path(path).read_text())
    if all(k in d for k in ("U_power_basis","V_power_basis","W_power_basis","c_power_basis","number_field")):
        coeffs=[sp.Integer(x) for x in d["number_field"]["minimal_polynomial_coefficients_ascending"]]
        degree=len(coeffs)-1
        if degree==1:
            K=QQ
            def elt(v): return QQ.convert(q_to_sympy(v[0]))
        else:
            t=sp.Symbol('t'); p=sum(coeffs[i]*t**i for i in range(len(coeffs)))
            theta=sp.CRootOf(p,0)
            K=QQ.algebraic_field(theta)
            def elt(v):
                expr=sum(q_to_sympy(x)*theta**i for i,x in enumerate(v))
                return K.from_sympy(expr)
        U=[[elt(x) for x in row] for row in d["U_power_basis"]]
        V=[[elt(x) for x in row] for row in d["V_power_basis"]]
        W=[[elt(x) for x in row] for row in d["W_power_basis"]]
        c=[elt(x) for x in d["c_power_basis"]]
        return K,U,V,W,c
    if "radicand" in d:
        rad=sp.Integer(d["radicand"]); theta=sp.sqrt(rad); K=QQ.algebraic_field(theta)
        locals_={'sqrt':sp.sqrt}
        def elt_expr(s): return K.from_sympy(sp.sympify(s,locals=locals_))
        U=[[elt_expr(x) for x in row] for row in d["U"]]; V=[[elt_expr(x) for x in row] for row in d["V"]]
        W=[[elt_expr(x) for x in row] for row in d["W"]]; c=[elt_expr(x) for x in d["c"]]
        return K,U,V,W,c
    raise ValueError("unsupported certificate schema")

def z(K): return K.zero
def one(K): return K.one

def fast(A,B,K,U,V,W,c):
    out=[[z(K) for _ in range(3)] for _ in range(3)]
    for r in range(len(c)):
        L=z(K); R=z(K)
        for i in range(3):
            for j in range(3):
                q=3*i+j; L += U[q][r]*A[i][j]; R += V[q][r]*B[i][j]
        m=c[r]*L*R
        for i in range(3):
            for k in range(3): out[i][k] += W[3*i+k][r]*m
    return out

def naive(A,B,K):
    return [[sum((A[i][j]*B[j][k] for j in range(3)), z(K)) for k in range(3)] for i in range(3)]

def m2z(K): return [[z(K),z(K)],[z(K),z(K)]]
def m2add(A,B): return [[A[i][j]+B[i][j] for j in range(2)] for i in range(2)]
def m2scale(q,A): return [[q*A[i][j] for j in range(2)] for i in range(2)]
def m2mul(A,B,K): return [[sum((A[i][j]*B[j][k] for j in range(2)),z(K)) for k in range(2)] for i in range(2)]
def fast_blocks(A,B,K,U,V,W,c):
    out=[[m2z(K) for _ in range(3)] for _ in range(3)]
    for r in range(len(c)):
        L=m2z(K); R=m2z(K)
        for i in range(3):
            for j in range(3):
                q=3*i+j; L=m2add(L,m2scale(U[q][r],A[i][j])); R=m2add(R,m2scale(V[q][r],B[i][j]))
        m=m2scale(c[r],m2mul(L,R,K))
        for i in range(3):
            for k in range(3): out[i][k]=m2add(out[i][k],m2scale(W[3*i+k][r],m))
    return out
def naive_blocks(A,B,K):
    out=[[m2z(K) for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for k in range(3):
            for j in range(3): out[i][k]=m2add(out[i][k],m2mul(A[i][j],B[j][k],K))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("certificate",type=Path); ap.add_argument("--trials",type=int,default=50)
    ap.add_argument("--nc-trials",type=int,default=10); ap.add_argument("--seed",type=int,default=314159)
    a=ap.parse_args(); K,U,V,W,c=load(a.certificate); rng=random.Random(a.seed)
    for qa in range(9):
        for qb in range(9):
            A=[[z(K) for _ in range(3)] for _ in range(3)]; B=[[z(K) for _ in range(3)] for _ in range(3)]
            A[qa//3][qa%3]=one(K); B[qb//3][qb%3]=one(K)
            if fast(A,B,K,U,V,W,c)!=naive(A,B,K): raise SystemExit(f"FAIL matrix units {qa},{qb}")
    for _ in range(a.trials):
        A=[[K.convert(rng.randint(-4,4)) for _ in range(3)] for _ in range(3)]; B=[[K.convert(rng.randint(-4,4)) for _ in range(3)] for _ in range(3)]
        if fast(A,B,K,U,V,W,c)!=naive(A,B,K): raise SystemExit("FAIL random exact trial")
    for _ in range(a.nc_trials):
        A=[[[[K.convert(rng.randint(-2,2)) for _ in range(2)] for _ in range(2)] for _ in range(3)] for _ in range(3)]
        B=[[[[K.convert(rng.randint(-2,2)) for _ in range(2)] for _ in range(2)] for _ in range(3)] for _ in range(3)]
        if fast_blocks(A,B,K,U,V,W,c)!=naive_blocks(A,B,K): raise SystemExit("FAIL noncommutative exact trial")
    print("# SymPy independent exact operational verification")
    print(f"domain: {K}")
    print(f"rank: {len(c)}")
    print("81 matrix-unit pairs / 729 output coefficients: PASS")
    print(f"random exact trials: {a.trials} PASS")
    print(f"noncommutative M2(K) trials: {a.nc_trials} PASS")
    print("RESULT: PASS")
if __name__=='__main__': main()
