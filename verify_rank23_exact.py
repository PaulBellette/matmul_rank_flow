"""Standalone exact verifier for a rank-23 JSON certificate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sympy as sp
from exactify_rank23 import exact_verify

def parse(x):
    if isinstance(x, list): return [parse(y) for y in x]
    return sp.sympify(x)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('certificate',type=Path);args=ap.parse_args()
    obj=json.loads(args.certificate.read_text())
    U,V,W,c=(parse(obj[k]) for k in ('U','V','W','c'))
    ok,n,fail=exact_verify(U,V,W,c,3)
    print(f"exact={ok} nonzero_identities={n}")
    if not ok:
        for x in fail: print(x)
        raise SystemExit(1)
if __name__=='__main__':main()
