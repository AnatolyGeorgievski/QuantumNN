#!/usr/bin/env python3
"""Шаг: POOL_2D avg (s==k, pad=0) vs CPU. Gemma3 mmproj ступень."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

KERNEL = (Path(__file__).resolve().parents[1] / "opencl" / "grain" / "pool_2d_tile.cl").read_text(encoding="utf-8")


def cpu_pool_avg(src: np.ndarray, k: int) -> np.ndarray:
    # src: [C,H,W] → [C,H/k,W/k]
    C, H, W = src.shape
    assert H % k == 0 and W % k == 0
    oh, ow = H // k, W // k
    out = np.empty((C, oh, ow), dtype=np.float32)
    for c in range(C):
        for oy in range(oh):
            for ox in range(ow):
                tile = src[c, oy * k : (oy + 1) * k, ox * k : (ox + 1) * k]
                out[c, oy, ox] = float(tile.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", type=int, default=0)
    ap.add_argument("-S", type=int, default=56, help="side (patches_per_side typical)")
    ap.add_argument("-k", type=int, default=2, help="kernel=stride")
    ap.add_argument("-C", type=int, default=3)
    args = ap.parse_args()
    import pyopencl as cl

    S, k, C = args.S, args.k, args.C
    if S % k:
        print("S must be multiple of k", file=sys.stderr)
        return 1
    oh = ow = S // k
    rng = np.random.default_rng(0)
    # layout for kernel: planes of H×W contiguous; we store [C,H,W]
    src = rng.standard_normal((C, S, S), dtype=np.float32)
    expect = cpu_pool_avg(src, k)

    # kernel expects [im][H][W] with ne00=W ne01=H — same as [C,H,W] row-major
    host = src.ravel()

    devs = [d for p in cl.get_platforms() for d in p.get_devices()]
    dev = devs[args.d]
    print(f"device[{args.d}]: {dev.name}")
    print(f"op: pool_2D_avg  S={S} k={k} C={C} -> {oh}x{ow}")

    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, KERNEL).build()

    buf_s = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host)
    buf_d = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=C * oh * ow * 4)

    # preferred subgroup
    try:
        sg = cl.Kernel(prg, "pool_2D_avg").get_work_group_info(
            cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, dev
        )
    except Exception:
        sg = 16
    if not sg:
        sg = 16

    prg.pool_2D_avg(
        queue,
        (ow * sg, oh, C),
        (sg, 1, 1),
        buf_s,
        buf_d,
        np.uint32(S),
        np.uint32(S),
        np.uint32(ow),
        np.uint32(oh),
        np.uint32(k),
        np.uint32(k),
    )
    got = np.empty((C, oh, ow), dtype=np.float32)
    cl.enqueue_copy(queue, got, buf_d)
    queue.finish()

    err = float(np.max(np.abs(got - expect)))
    ok = err < 1e-5
    print(f"max_abs_err={err:.3e}  {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
