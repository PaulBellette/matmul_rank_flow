#!/usr/bin/env sage
"""Independent exact verifier for SageMath.

Run with: sage verify_sage.py path/to/rank23_exact.json
No code from matmul_rank_flow is imported. The certificate is evaluated as an
operational bilinear algorithm and compared with ordinary matrix multiplication.
"""
from sage.all import *
import argparse, json, random
from pathlib import Path


def rat(s): return QQ(str(s))

def load_cert(path):
    d=json.loads(Path(path).read_text())
    if all(k in d for k in ("U_power_basis","V_power_basis","W_power_basis","c_power_basis","number_field")):
        cc=[ZZ(x) for x in d['number_field']['minimal_polynomial_coefficients_ascending']]
        deg=len(cc)-1
        if deg==1:
            K=QQ; alpha=None
            def elt(v): return rat(v[0])
        else:
            Rt=PolynomialRing(QQ,'t'); t=Rt.gen(); p=sum(QQ(cc[i])*t**i for i in range(len(cc)))
            K=NumberField(p,'alpha'); alpha=K.gen()
            def elt(v): return K(sum(rat(x)*alpha**i for i,x in enumerate(v)))
        U=[[elt(x) for x in row] for row in d['U_power_basis']]
        V=[[elt(x) for x in row] for row in d['V_power_basis']]
        W=[[elt(x) for x in row] for row in d['W_power_basis']]
        c=[elt(x) for x in d['c_power_basis']]
        return d,K,U,V,W,c
    if 'radicand' in d:
        Rt=PolynomialRing(QQ,'t'); t=Rt.gen(); K=NumberField(t^2-ZZ(d['radicand']),'alpha'); alpha=K.gen()
        def pe(s):
            s=s.replace('**','^').replace('sqrt(%s)'%d['radicand'],'alpha')
            return K(sage_eval(s,locals={'alpha':alpha}))
        return d,K,[[pe(x) for x in row] for row in d['U']],[[pe(x) for x in row] for row in d['V']],[[pe(x) for x in row] for row in d['W']],[pe(x) for x in d['c']]
    raise ValueError('unsupported certificate schema')

def fast(A,B,K,U,V,W,c):
    C=[[K.zero() for _ in range(3)] for _ in range(3)]
    for r in range(len(c)):
        L=K.zero(); R=K.zero()
        for i in range(3):
            for j in range(3):
                q=3*i+j; L += U[q][r]*A[i][j]; R += V[q][r]*B[i][j]
        m=c[r]*L*R
        for i in range(3):
            for k in range(3): C[i][k]+=W[3*i+k][r]*m
    return C

def naive(A,B,K): return [[sum((A[i][j]*B[j][k] for j in range(3)),K.zero()) for k in range(3)] for i in range(3)]
def bz(K): return Matrix(K,2,2,[0]*4)
def fast_blocks(A,B,K,U,V,W,c):
    C=[[bz(K) for _ in range(3)] for _ in range(3)]
    for r in range(len(c)):
        L=bz(K); R=bz(K)
        for i in range(3):
            for j in range(3):
                q=3*i+j; L += U[q][r]*A[i][j]; R += V[q][r]*B[i][j]
        m=c[r]*(L*R)
        for i in range(3):
            for k in range(3): C[i][k]+=W[3*i+k][r]*m
    return C
def naive_blocks(A,B,K):
    C=[[bz(K) for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for k in range(3):
            for j in range(3): C[i][k]+=A[i][j]*B[j][k]
    return C

def finite_field_check(d,p):
    if not all(k in d for k in ('U_power_basis','V_power_basis','W_power_basis','c_power_basis','number_field')): return 'SKIP legacy schema'
    cc=[ZZ(x) for x in d['number_field']['minimal_polynomial_coefficients_ascending']]
    den=[]
    for key in ('U_power_basis','V_power_basis','W_power_basis'):
        for row in d[key]:
            for v in row:
                den.extend(rat(x).denominator() for x in v)
    for v in d['c_power_basis']: den.extend(rat(x).denominator() for x in v)
    if any(ZZ(q)%p==0 for q in den) or ZZ(cc[-1])%p==0: return 'SKIP bad prime'
    F=GF(p); Rt=PolynomialRing(F,'x'); x=Rt.gen(); poly=sum(F(cc[i])*x**i for i in range(len(cc)))
    S=Rt.quotient(poly,'a'); a=S.gen()
    def e(v): return sum(S(F(rat(q).numerator())/F(rat(q).denominator()))*a**i for i,q in enumerate(v))
    U=[[e(v) for v in row] for row in d['U_power_basis']]; V=[[e(v) for v in row] for row in d['V_power_basis']]
    W=[[e(v) for v in row] for row in d['W_power_basis']]; c=[e(v) for v in d['c_power_basis']]
    for qa in range(9):
        for qb in range(9):
            A=[[S.zero() for _ in range(3)] for _ in range(3)]; B=[[S.zero() for _ in range(3)] for _ in range(3)]
            A[qa//3][qa%3]=S.one(); B[qb//3][qb%3]=S.one()
            if fast(A,B,S,U,V,W,c)!=naive(A,B,S): return 'FAIL'
    return 'PASS'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('certificate',type=Path); ap.add_argument('--trials',type=int,default=100); ap.add_argument('--nc-trials',type=int,default=20)
    ap.add_argument('--primes',default='101,103,107'); ap.add_argument('--seed',type=int,default=8675309); args=ap.parse_args()
    d,K,U,V,W,c=load_cert(args.certificate); rng=random.Random(args.seed)
    for qa in range(9):
        for qb in range(9):
            A=[[K.zero() for _ in range(3)] for _ in range(3)]; B=[[K.zero() for _ in range(3)] for _ in range(3)]
            A[qa//3][qa%3]=K.one(); B[qb//3][qb%3]=K.one()
            assert fast(A,B,K,U,V,W,c)==naive(A,B,K)
    for _ in range(args.trials):
        A=[[K(rng.randint(-5,5)) for _ in range(3)] for _ in range(3)]; B=[[K(rng.randint(-5,5)) for _ in range(3)] for _ in range(3)]
        assert fast(A,B,K,U,V,W,c)==naive(A,B,K)
    for _ in range(args.nc_trials):
        A=[[Matrix(K,2,2,[rng.randint(-3,3) for _ in range(4)]) for _ in range(3)] for _ in range(3)]
        B=[[Matrix(K,2,2,[rng.randint(-3,3) for _ in range(4)]) for _ in range(3)] for _ in range(3)]
        assert fast_blocks(A,B,K,U,V,W,c)==naive_blocks(A,B,K)
    print('# Sage independent exact operational verification'); print('field:',K); print('rank:',len(c))
    print('81 matrix-unit pairs / 729 output coefficients: PASS'); print('random exact trials:',args.trials,'PASS'); print('noncommutative M2(K) trials:',args.nc_trials,'PASS')
    for p in [int(x) for x in args.primes.split(',') if x.strip()]: print('mod',p,finite_field_check(d,p))
    print('RESULT: PASS')
if __name__=='__main__': main()
