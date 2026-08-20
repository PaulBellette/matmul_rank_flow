#!/usr/bin/env python3
"""Self-contained exact verifier for rational certificates.

Uses only the Python standard library (Fraction, json, random). No NumPy, SymPy,
Torch, Sage, or matmul_rank_flow imports. It operationally evaluates the algorithm.
"""
from __future__ import annotations
import argparse, json, random
from fractions import Fraction
from pathlib import Path


def F(s): return Fraction(str(s))

def rational_from_basis(x):
    qs = [F(v) for v in x]
    if any(q != 0 for q in qs[1:]):
        raise ValueError("certificate is not rational")
    return qs[0]

def load(path):
    d = json.loads(Path(path).read_text())
    if not all(k in d for k in ("U_power_basis","V_power_basis","W_power_basis","c_power_basis")):
        raise ValueError("requires generic certificate with *_power_basis arrays")
    U = [[rational_from_basis(x) for x in row] for row in d["U_power_basis"]]
    V = [[rational_from_basis(x) for x in row] for row in d["V_power_basis"]]
    W = [[rational_from_basis(x) for x in row] for row in d["W_power_basis"]]
    c = [rational_from_basis(x) for x in d["c_power_basis"]]
    return U,V,W,c

def mm(A,B,U,V,W,c):
    af=[A[i][j] for i in range(3) for j in range(3)]
    bf=[B[i][j] for i in range(3) for j in range(3)]
    out=[[Fraction(0) for _ in range(3)] for _ in range(3)]
    for r in range(len(c)):
        L=sum((U[q][r]*af[q] for q in range(9)), Fraction(0))
        R=sum((V[q][r]*bf[q] for q in range(9)), Fraction(0))
        m=c[r]*L*R
        for i in range(3):
            for k in range(3): out[i][k] += W[3*i+k][r]*m
    return out

def naive(A,B):
    return [[sum((A[i][j]*B[j][k] for j in range(3)), Fraction(0)) for k in range(3)] for i in range(3)]

def mat2_zero(): return [[Fraction(0),Fraction(0)],[Fraction(0),Fraction(0)]]
def mat2_add(A,B): return [[A[i][j]+B[i][j] for j in range(2)] for i in range(2)]
def mat2_scale(q,A): return [[q*A[i][j] for j in range(2)] for i in range(2)]
def mat2_mul(A,B):
    return [[sum((A[i][j]*B[j][k] for j in range(2)), Fraction(0)) for k in range(2)] for i in range(2)]
def block_mm(A,B,U,V,W,c):
    out=[[mat2_zero() for _ in range(3)] for _ in range(3)]
    for r in range(len(c)):
        L=mat2_zero(); R=mat2_zero()
        for i in range(3):
            for j in range(3):
                q=3*i+j
                L=mat2_add(L,mat2_scale(U[q][r],A[i][j]))
                R=mat2_add(R,mat2_scale(V[q][r],B[i][j]))
        m=mat2_scale(c[r],mat2_mul(L,R))
        for i in range(3):
            for k in range(3): out[i][k]=mat2_add(out[i][k],mat2_scale(W[3*i+k][r],m))
    return out
def block_naive(A,B):
    out=[[mat2_zero() for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for k in range(3):
            for j in range(3): out[i][k]=mat2_add(out[i][k],mat2_mul(A[i][j],B[j][k]))
    return out

def modq(q,p): return (q.numerator%p)*pow(q.denominator%p,-1,p)%p
def verify_mod(U,V,W,c,p):
    Um=[[modq(x,p) for x in row] for row in U]; Vm=[[modq(x,p) for x in row] for row in V]
    Wm=[[modq(x,p) for x in row] for row in W]; cm=[modq(x,p) for x in c]
    for qa in range(9):
        for qb in range(9):
            out=[0]*9
            for r in range(len(c)):
                m=cm[r]*Um[qa][r]*Vm[qb][r]%p
                for qo in range(9): out[qo]=(out[qo]+Wm[qo][r]*m)%p
            ai,aj=divmod(qa,3); bi,bk=divmod(qb,3)
            target=[0]*9
            if aj==bi: target[3*ai+bk]=1
            if out!=target: return False,(qa,qb,out,target)
    return True,None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("certificate",type=Path)
    ap.add_argument("--trials",type=int,default=200); ap.add_argument("--nc-trials",type=int,default=50)
    ap.add_argument("--seed",type=int,default=20260821); ap.add_argument("--primes",default="2,3,5,7,11,101")
    a=ap.parse_args(); U,V,W,c=load(a.certificate); rng=random.Random(a.seed)
    for qa in range(9):
        for qb in range(9):
            A=[[Fraction(0) for _ in range(3)] for _ in range(3)]; B=[[Fraction(0) for _ in range(3)] for _ in range(3)]
            A[qa//3][qa%3]=1; B[qb//3][qb%3]=1
            if mm(A,B,U,V,W,c)!=naive(A,B): raise SystemExit(f"FAIL basis pair {qa},{qb}")
    for _ in range(a.trials):
        A=[[Fraction(rng.randint(-5,5)) for _ in range(3)] for _ in range(3)]
        B=[[Fraction(rng.randint(-5,5)) for _ in range(3)] for _ in range(3)]
        if mm(A,B,U,V,W,c)!=naive(A,B): raise SystemExit("FAIL random rational trial")
    for _ in range(a.nc_trials):
        A=[[[[Fraction(rng.randint(-3,3)) for _ in range(2)] for _ in range(2)] for _ in range(3)] for _ in range(3)]
        B=[[[[Fraction(rng.randint(-3,3)) for _ in range(2)] for _ in range(2)] for _ in range(3)] for _ in range(3)]
        if block_mm(A,B,U,V,W,c)!=block_naive(A,B): raise SystemExit("FAIL noncommutative trial")
    good=[]; skipped=[]
    for p in [int(x) for x in a.primes.split(',') if x.strip()]:
        try: ok,detail=verify_mod(U,V,W,c,p)
        except ValueError: skipped.append(p); continue
        if not ok: raise SystemExit(f"FAIL mod {p}: {detail}")
        good.append(p)
    print("# standalone exact rational verification")
    print(f"rank: {len(c)}")
    print("81 matrix-unit pairs: PASS")
    print(f"random exact rational trials: {a.trials} PASS")
    print(f"noncommutative M2(Q) trials: {a.nc_trials} PASS")
    print(f"finite fields checked: {good}" + (f"; skipped bad-denominator primes {skipped}" if skipped else ""))
    print("RESULT: PASS")

if __name__=='__main__': main()
