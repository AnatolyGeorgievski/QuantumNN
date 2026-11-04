#!/usr/bin/env python3
"""Контракт глубины 2 patch_emb без GPU: S=P·nout, формы M/K/E."""
from __future__ import annotations
import argparse
import sys


def ok(S: int, P: int, E: int) -> list[str]:
    bad: list[str] = []
    if P <= 0 or E <= 0:
        bad.append("P/E <= 0")
        return bad
    if S % P:
        bad.append(f"S={S} not multiple of P={P}")
        return bad
    nout = S // P
    M = nout * nout
    K = P * P * 3
    if M * K <= 0:
        bad.append("empty cols")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-S", type=int, default=224)
    ap.add_argument("-P", type=int, default=16)
    ap.add_argument("-E", type=int, default=128)
    args = ap.parse_args()
    bad = ok(args.S, args.P, args.E)
    if bad:
        print("UNSAT:", "; ".join(bad), file=sys.stderr)
        return 1
    n = args.S // args.P
    M = n * n
    K = args.P * args.P * 3
    print(f"sat  S={args.S} P={args.P} E={args.E} M={M} K={K}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
