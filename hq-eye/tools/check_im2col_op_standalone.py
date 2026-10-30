#!/usr/bin/env python3
"""Шаг 1: im2col_tile vs CPU. Самодостаточный (ядро внутри)."""
from __future__ import annotations
import argparse
import sys

import numpy as np

from ocl_dev import pick_device

KERNEL = r"""
kernel void im2col_tile(
        global float *src,
        global float *dst,
        uint W, uint H, uint P) {
    uint ox = get_global_id(0);
    uint oy = get_global_id(1);
    uint nout_x = W / P;
    uint nout_y = H / P;
    if (ox >= nout_x || oy >= nout_y) return;
    dst += (oy * nout_x + ox) * (P * P);
    for (uint ky = 0; ky < P; ky++)
        for (uint kx = 0; kx < P; kx++)
            dst[ky * P + kx] = src[(oy * P + ky) * W + (ox * P + kx)];
}
"""


def cpu_im2col_tile(src: np.ndarray, P: int) -> np.ndarray:
    H, W = src.shape
    nx, ny = W // P, H // P
    out = np.empty((ny, nx, P, P), dtype=np.float32)
    for oy in range(ny):
        for ox in range(nx):
            out[oy, ox] = src[oy * P : (oy + 1) * P, ox * P : (ox + 1) * P]
    return out.reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", type=int, default=0)
    ap.add_argument("-P", type=int, default=16)
    ap.add_argument("-S", type=int, default=224)
    args = ap.parse_args()
    import pyopencl as cl

    if args.S % args.P:
        print("S must be multiple of P", file=sys.stderr)
        return 1

    dev = pick_device(cl, args.d)
    print(f"device[{args.d}]: {dev.name}")
    print(f"op: im2col_tile  P={args.P}  S={args.S}")

    H = W = args.S
    P = args.P
    host_src = np.random.default_rng(0).standard_normal((H, W), dtype=np.float32)
    expect = cpu_im2col_tile(host_src, P)

    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, KERNEL).build()
    buf_s = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_src.ravel())
    nout = (H // P) * (W // P) * P * P
    buf_d = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=nout * 4)
    prg.im2col_tile(
        queue, (W // P, H // P), None, buf_s, buf_d, np.uint32(W), np.uint32(H), np.uint32(P)
    )
    got = np.empty(nout, dtype=np.float32)
    cl.enqueue_copy(queue, got, buf_d)
    queue.finish()
    err = float(np.max(np.abs(got - expect)))
    ok = err < 1e-5
    print(f"max_abs_err={err:.3e}  {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
