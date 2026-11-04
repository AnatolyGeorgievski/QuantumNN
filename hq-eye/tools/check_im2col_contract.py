#!/usr/bin/env python3
"""Контракт глубины 2 im2col_tile без GPU."""
from __future__ import annotations
import argparse
import sys


def ok(S: int, P: int) -> list[str]:
    bad: list[str] = []
    if P <= 0:
        bad.append("P<=0")
        return bad
    if S % P:
        bad.append(f"S={S} not multiple of P={P}")
        return bad
    nout = S // P
    src_len = S * S
    dst_len = nout * nout * P * P
    if dst_len != S * S:
        bad.append(f"dst_len {dst_len} != src_len {src_len} (ожидание полной нарезки)")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-S", type=int, default=224)
    ap.add_argument("-P", type=int, default=16)
    args = ap.parse_args()
    bad = ok(args.S, args.P)
    if bad:
        print("UNSAT:", "; ".join(bad), file=sys.stderr)
        return 1
    n = args.S // args.P
    print(f"sat  S={args.S} P={args.P} GWS=({n},{n}) dst={n*n*args.P*args.P}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
