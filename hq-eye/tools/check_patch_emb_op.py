#!/usr/bin/env python3
"""Шаг 2a: геометрический patch embedding — im2col(P×P×C) + линейная проекция E.
Не полный mmproj / ViT. Сверка OpenCL vs CPU. Ассемблерный лист — см. opencl/отладка-тензорный-ассемблер.md."""
from __future__ import annotations
import argparse, sys
import numpy as np

KERNEL = r"""
kernel void im2col_rgb(
        global const float *src,   // H*W*3, HWC
        global float *dst,         // nout * (P*P*3)
        uint W, uint H, uint P) {
    uint ox = get_global_id(0);
    uint oy = get_global_id(1);
    uint nout_x = W / P;
    uint nout_y = H / P;
    if (ox >= nout_x || oy >= nout_y) return;
    global float *row = dst + (oy * nout_x + ox) * (P * P * 3u);
    uint t = 0;
    for (uint ky = 0; ky < P; ky++)
        for (uint kx = 0; kx < P; kx++) {
            uint iy = oy * P + ky;
            uint ix = ox * P + kx;
            uint base = (iy * W + ix) * 3u;
            row[t++] = src[base + 0];
            row[t++] = src[base + 1];
            row[t++] = src[base + 2];
        }
}

kernel void gemm_nt(
        global const float *A,  // M x K  (patches × P²C)
        global const float *B,  // E x K  (weights, row-major)
        global float *C,        // M x E
        uint M, uint E, uint K) {
    uint m = get_global_id(0);
    uint e = get_global_id(1);
    if (m >= M || e >= E) return;
    float acc = 0.f;
    for (uint k = 0; k < K; k++)
        acc += A[m * K + k] * B[e * K + k];
    C[m * E + e] = acc;
}
"""


def cpu_patch_emb(img: np.ndarray, W: np.ndarray, P: int) -> np.ndarray:
    """img H×W×3 → (nout, E). W is E×(P²·3)."""
    H, Ww, C = img.shape
    assert C == 3 and Ww == H
    nx, ny = Ww // P, H // P
    K = P * P * 3
    cols = np.empty((ny * nx, K), dtype=np.float32)
    t = 0
    for oy in range(ny):
        for ox in range(nx):
            patch = img[oy * P : (oy + 1) * P, ox * P : (ox + 1) * P, :].reshape(-1)
            cols[t] = patch
            t += 1
    # C = cols @ W.T  → (M, E)
    return cols @ W.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", type=int, default=0)
    ap.add_argument("-P", type=int, default=16)
    ap.add_argument("-S", type=int, default=224)
    ap.add_argument("-E", type=int, default=128, help="embedding dim (smoke; mmproj often 768/1152)")
    args = ap.parse_args()
    import pyopencl as cl

    if args.S % args.P:
        print("S must be multiple of P", file=sys.stderr)
        return 1

    P, S, E = args.P, args.S, args.E
    K = P * P * 3
    M = (S // P) * (S // P)
    rng = np.random.default_rng(0)
    img = rng.standard_normal((S, S, 3), dtype=np.float32)
    W = rng.standard_normal((E, K), dtype=np.float32)
    expect = cpu_patch_emb(img, W, P)

    print("=== assembler leaf (patch Conv2D) ===")
    print(f"IM2COL   inp_raw.f32({S}x{S}x3)  m0({P}x{P}x3)     -> {M}x{K}")
    print(f"MUL_MAT  m0.f32({E}x{K})         r1.f32({M}x{K})   -> {M}x{E}")
    print("(next mmproj: RESHAPE/TRANSPOSE/CONT · ADD bias · ADD pos · ViT)")
    print()

    devs = [d for p in cl.get_platforms() for d in p.get_devices()]
    dev = devs[args.d]
    print(f"device[{args.d}]: {dev.name}")
    print(f"op: patch_emb  P={P}  S={S}  E={E}  M={M}  K={K}")

    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, KERNEL).build()

    buf_img = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=img.ravel())
    buf_col = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=M * K * 4)
    buf_w = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=W.ravel())
    buf_out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=M * E * 4)

    prg.im2col_rgb(queue, (S // P, S // P), None, buf_img, buf_col, np.uint32(S), np.uint32(S), np.uint32(P))
    prg.gemm_nt(queue, (M, E), None, buf_col, buf_w, buf_out, np.uint32(M), np.uint32(E), np.uint32(K))
    got = np.empty((M, E), dtype=np.float32)
    cl.enqueue_copy(queue, got, buf_out)
    queue.finish()

    err = float(np.max(np.abs(got - expect)))
    # float32 GEMM: ~K fused mul-adds; 1e-3 ok for K~768 smoke (not bit-exact)
    ok = err < 1e-3
    print(f"max_abs_err={err:.3e}  {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
