#!/usr/bin/env python3
"""Контракт глубины 2 pool_2D_avg без GPU: NDRange и покрытие s==k pad=0."""
from __future__ import annotations
import argparse
import sys

POOL_SG = 16


def ok(S: int, k: int, C: int, local0: int = POOL_SG) -> list[str]:
    bad: list[str] = []
    if local0 != POOL_SG:
        bad.append(f"local0={local0} != POOL_SG={POOL_SG}")
    if local0 == 64:
        bad.append("local0 is preferred-WG 64 — запрещено контрактом")
    if S % k != 0:
        bad.append(f"S={S} not multiple of k={k}")
        return bad
    ne0 = ne1 = S // k
    ne00 = ne01 = S
    kw = kh = k
    if ne00 < kw * ne0:
        bad.append("ne00 < kw*ne0")
    if ne01 < kh * ne1:
        bad.append("ne01 < kh*ne1")
    global0 = ne0 * POOL_SG
    if global0 // local0 != ne0:
        bad.append("groups0 != ne0")
    if C <= 0:
        bad.append("C<=0")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-S", type=int, default=56)
    ap.add_argument("-k", type=int, default=2)
    ap.add_argument("-C", type=int, default=3)
    ap.add_argument("--local", type=int, default=POOL_SG)
    args = ap.parse_args()
    bad = ok(args.S, args.k, args.C, args.local)
    if bad:
        print("UNSAT:", "; ".join(bad), file=sys.stderr)
        return 1
    print(f"sat  S={args.S} k={args.k} C={args.C} local={args.local} GWS=({args.S//args.k*POOL_SG},{args.S//args.k},{args.C})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
