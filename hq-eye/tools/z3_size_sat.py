#!/usr/bin/env python3
"""Глубина-2: выполнимость размерных равенств через Z3 QF_NIA.
Контракт раньше кода: сюда — формулы «кто/что совпадает», не ядро.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def check(asserts: list[str]) -> str:
    body = ["(set-logic QF_NIA)"]
    # declare ints appearing as bare names on LHS of declare lines — simple: user passes full asserts + decls
    body.extend(asserts)
    body.append("(check-sat)")
    text = "\n".join(body) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".smt2", delete=False) as f:
        f.write(text)
        path = f.name
    r = subprocess.run(["z3", path], capture_output=True, text=True)
    Path(path).unlink(missing_ok=True)
    line = (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else "unknown"
    return line


def main() -> int:
    ap = argparse.ArgumentParser(description="Z3 QF_NIA size sat (глубина 2)")
    ap.add_argument("preset", choices=["pool", "im2col", "patch", "warp"], help="готовый набор")
    ap.add_argument("--bad", action="store_true", help="сломать контракт (ожидай unsat)")
    args = ap.parse_args()

    if args.preset == "pool":
        decls = [
            "(declare-const POOL_SG Int)",
            "(declare-const local0 Int)",
            "(declare-const ne0 Int)",
            "(declare-const ne00 Int)",
            "(declare-const kw Int)",
        ]
        good = decls + [
            "(assert (= POOL_SG 16))",
            "(assert (= local0 POOL_SG))",
            "(assert (= ne0 28))",
            "(assert (= kw 2))",
            "(assert (= ne00 56))",
            "(assert (>= ne00 (* kw ne0)))",
            "(assert (= local0 16))",
        ]
        bad = decls + [
            "(assert (= POOL_SG 16))",
            "(assert (= local0 64))",
            "(assert (= local0 POOL_SG))",
        ]
        formula = bad if args.bad else good
    elif args.preset == "im2col":
        decls = ["(declare-const S Int)", "(declare-const P Int)", "(declare-const nout Int)"]
        good = decls + ["(assert (= S 224))", "(assert (= P 16))", "(assert (= nout 14))", "(assert (= S (* P nout)))"]
        bad = decls + ["(assert (= S 224))", "(assert (= P 15))", "(assert (= nout 14))", "(assert (= S (* P nout)))"]
        formula = bad if args.bad else good
    elif args.preset == "patch":
        decls = [
            "(declare-const S Int)", "(declare-const P Int)", "(declare-const nout Int)",
            "(declare-const M Int)", "(declare-const K Int)",
        ]
        good = decls + [
            "(assert (= S 224))", "(assert (= P 16))", "(assert (= nout 14))",
            "(assert (= S (* P nout)))", "(assert (= M (* nout nout)))",
            "(assert (= K (* (* P P) 3)))",
        ]
        bad = decls + [
            "(assert (= S 224))", "(assert (= P 15))", "(assert (= nout 14))",
            "(assert (= S (* P nout)))",
        ]
        formula = bad if args.bad else good
    else:  # warp
        decls = ["(declare-const len Int)", "(declare-const max_i Int)", "(declare-const max_ix Int)"]
        good = decls + [
            "(assert (= max_i 496))", "(assert (= max_ix 15))", "(assert (= len 512))",
            "(assert (< (+ max_i max_ix) len))",
        ]
        bad = decls + [
            "(assert (= max_i 496))", "(assert (= max_ix 15))", "(assert (= len 256))",
            "(assert (< (+ max_i max_ix) len))",
        ]
        formula = bad if args.bad else good

    ans = check(formula)
    print(ans)
    want = "unsat" if args.bad else "sat"
    return 0 if ans == want else 1


if __name__ == "__main__":
    raise SystemExit(main())
