/* qnn-encode: f32 -> HF8 (E4M3) or BF8 (E5M2), report round-trip error.
   Formulas match QuantumNN/qnn.h (HF8/BF8 / F8_E4M3).

   Stream (elementwise):
     qnn-encode hf8|bf8 [in.f32] [out.f8]

   Block FP8 like Qwen3.8-FP8:
     qnn-encode hf8-block N K [B] in.f32 out.hf8 out.scale.bf16
     qnn-encode hf8-block N K [B] in.f32 out.safetensors [tensor_name]
       -> F8_E4M3 `name` + BF16 `name_scale_inv` (default name=weight)
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define HF8_MAX_FIN 448.f

static uint8_t fp32_to_bf8(float s) {
    const int f32_bias = 127;
    const int bf8_bias = 15;
    union { float f; uint32_t bits; } v;
    v.f = s;
    uint8_t si = (v.bits & 0x80000000u) >> 24;
    uint8_t ex = (v.bits & 0x7F800000u) >> 23;
    uint32_t m;
    uint32_t ma = (v.bits & 0x007FFFFFu);
    if (ex == 0xFF) {
        ex = 0x1f;
        m = ma == 0 ? 0 : (ma >> 21) | 2;
    } else if (ex > (f32_bias + bf8_bias)) {
        ex = 0x1f; m = 0;
    } else if (ex < (f32_bias - bf8_bias) - 2) {
        ex = 0; m = 0;
    } else if (ex <= (f32_bias - bf8_bias)) {
        m = (ma | 0x00800000u) >> ((f32_bias - bf8_bias) + 1 - ex);
        m |= (((ma & 0x001fffffu) + 0x1fffffu) >> 21);
        ex = 0;
        uint32_t fixup = (m >> 21) & 1;
        m = (m + 0xfffffu + fixup) >> 21;
    } else {
        uint32_t fixup = (v.bits >> 21) & 1;
        v.bits = v.bits + 0xfffffu + fixup;
        ex = ((v.bits & 0x7F800000u) >> 23) - (f32_bias - bf8_bias);
        m = (v.bits & 0x007FFFFFu) >> 21;
    }
    return (uint8_t) (si | (ex << 2) | m);
}

static uint8_t fp32_to_hf8(float s) {
    const int f32_bias = 127;
    const int hf8_bias = 7;
    union { float f; uint32_t bits; } v;
    v.f = s;
    uint8_t si = (v.bits & 0x80000000u) >> 24;
    uint8_t ex = (v.bits & 0x7F800000u) >> 23;
    uint32_t m;
    uint32_t ma = (v.bits & 0x007FFFFFu);
    if (ex == 0xFF) {
        ex = 0xf;
        m = ma == 0 ? 0 : (ma >> 20) | 4;
    } else if (ex > (f32_bias + hf8_bias)) {
        ex = 0xf; m = 0;
    } else if (ex < (f32_bias - hf8_bias) - 3) {
        ex = 0; m = 0;
    } else if (ex <= (f32_bias - hf8_bias)) {
        m = (ma | 0x00800000u) >> ((f32_bias - hf8_bias) + 1 - ex);
        m |= (((ma & 0xfffffu) + 0xfffffu) >> 20);
        ex = 0;
        uint32_t fixup = (m >> 20) & 1;
        m = (m + 0x7ffffu + fixup) >> 20;
    } else {
        uint32_t fixup = (v.bits >> 20) & 1;
        v.bits = v.bits + 0x7ffffu + fixup;
        ex = ((v.bits & 0x7F800000u) >> 23) - (f32_bias - hf8_bias);
        m = (v.bits & 0x007FFFFFu) >> 20;
    }
    return (uint8_t) (si | (ex << 3) | m);
}

static float fp16_to_fp32(uint16_t h) {
    int si = h >> 15;
    int ex = (h >> 10) & 31;
    int ma = h & 1023;
    float mag;
    if (ex == 0) {
        mag = ldexpf((float) ma, -24);
    } else if (ex == 31) {
        mag = ma == 0 ? INFINITY : NAN;
    } else {
        mag = ldexpf(1.f + (float) ma / 1024.f, ex - 15);
    }
    return si ? -mag : mag;
}

static float bf8_to_fp32(uint8_t b) {
    return fp16_to_fp32((uint16_t) b << 8);
}

static float hf8_to_fp32(uint8_t b) {
    int si = b >> 7;
    int ex = (b >> 3) & 0xf;
    int ma = b & 7;
    float mag;
    if (ex == 0) {
        mag = ldexpf((float) ma, -9);
    } else if (ex == 0xf) {
        mag = ma == 0 ? INFINITY : NAN;
    } else {
        mag = ldexpf(1.f + (float) ma / 8.f, ex - 7);
    }
    return si ? -mag : mag;
}

static uint16_t fp32_to_bf16_bits(float s) {
    union { float f; uint32_t u; } v;
    v.f = s;
    return (uint16_t) (v.u >> 16);
}

static void acc_err(double *sum, double *max_abs, float x, float y) {
    if (!isfinite(x) || !isfinite(y)) return;
    double e = (double) y - (double) x;
    double a = fabs(e);
    if (a > *max_abs) *max_abs = a;
    *sum += e * e;
}

static int ends_with(const char *s, const char *suf) {
    size_t n = strlen(s), m = strlen(suf);
    return n >= m && strcmp(s + n - m, suf) == 0;
}

static int run_stream(const char * kind, FILE * in, FILE * out) {
    int hf = strcmp(kind, "hf8") == 0;
    int bf = strcmp(kind, "bf8") == 0;
    if (!hf && !bf) {
        fprintf(stderr, "type is hf8, bf8, or hf8-block\n");
        return 2;
    }
    double sum = 0;
    double max_abs = 0;
    uint64_t n = 0;
    float x;
    while (fread(&x, sizeof x, 1, in) == 1) {
        uint8_t q = hf ? fp32_to_hf8(x) : fp32_to_bf8(x);
        float y = hf ? hf8_to_fp32(q) : bf8_to_fp32(q);
        acc_err(&sum, &max_abs, x, y);
        if (out && fwrite(&q, 1, 1, out) != 1) {
            perror("write");
            return 1;
        }
        n++;
    }
    double rmse = n ? sqrt(sum / (double) n) : 0;
    printf("%s n=%llu max_abs=%.8g rmse=%.8g\n", kind, (unsigned long long) n, max_abs, rmse);
    return 0;
}

/*! \brief quantize in-place buffers; caller frees *q_out / *scales_out */
static int quant_hf8_block(size_t N, size_t K, size_t B, FILE * in,
                           uint8_t **q_out, uint16_t **scales_out,
                           size_t *nB_out, size_t *kB_out,
                           double *max_abs_out, double *rmse_out) {
    size_t n_elem = N * K;
    float *w = (float *) malloc(n_elem * sizeof(float));
    uint8_t *q = (uint8_t *) malloc(n_elem);
    if (!w || !q) {
        fprintf(stderr, "oom\n");
        free(w); free(q);
        return 1;
    }
    if (fread(w, sizeof(float), n_elem, in) != n_elem) {
        fprintf(stderr, "need %zu f32 values (N*K)\n", n_elem);
        free(w); free(q);
        return 1;
    }

    size_t nB = (N + B - 1) / B;
    size_t kB = (K + B - 1) / B;
    uint16_t *scales = (uint16_t *) malloc(nB * kB * sizeof(uint16_t));
    if (!scales) {
        fprintf(stderr, "oom scales\n");
        free(w); free(q);
        return 1;
    }

    double sum = 0;
    double max_abs = 0;
    uint64_t n = 0;

    for (size_t bi = 0; bi < nB; bi++) {
        for (size_t bj = 0; bj < kB; bj++) {
            size_t i0 = bi * B, j0 = bj * B;
            size_t i1 = i0 + B; if (i1 > N) i1 = N;
            size_t j1 = j0 + B; if (j1 > K) j1 = K;
            float amax = 0.f;
            for (size_t i = i0; i < i1; i++)
                for (size_t j = j0; j < j1; j++) {
                    float a = fabsf(w[i * K + j]);
                    if (a > amax) amax = a;
                }
            float s = (amax > 0.f) ? (amax / HF8_MAX_FIN) : 1.f;
            float inv_s = 1.f / s;
            scales[bi * kB + bj] = fp32_to_bf16_bits(s);
            for (size_t i = i0; i < i1; i++)
                for (size_t j = j0; j < j1; j++) {
                    size_t idx = i * K + j;
                    float x = w[idx];
                    uint8_t qq = fp32_to_hf8(x * inv_s);
                    q[idx] = qq;
                    acc_err(&sum, &max_abs, x, hf8_to_fp32(qq) * s);
                    n++;
                }
        }
    }

    free(w);
    *q_out = q;
    *scales_out = scales;
    *nB_out = nB;
    *kB_out = kB;
    *max_abs_out = max_abs;
    *rmse_out = n ? sqrt(sum / (double) n) : 0;
    return 0;
}

/*! \brief safetensors: scale_inv then weight (byte offsets) */
static int write_safetensors_fp8_pair(const char *path, const char *name,
                                      size_t N, size_t K, size_t nB, size_t kB,
                                      const uint8_t *q, const uint16_t *scales) {
    char scale_name[512];
    snprintf(scale_name, sizeof(scale_name), "%s_scale_inv", name);

    uint64_t scale_bytes = (uint64_t) nB * kB * 2;
    uint64_t weight_bytes = (uint64_t) N * K;
    uint64_t off0 = 0;
    uint64_t off1 = scale_bytes;
    uint64_t off2 = scale_bytes + weight_bytes;

    char header[2048];
    int hlen = snprintf(header, sizeof(header),
        "{\"%s\":{\"dtype\":\"BF16\",\"shape\":[%zu,%zu],\"data_offsets\":[%llu,%llu]},"
        "\"%s\":{\"dtype\":\"F8_E4M3\",\"shape\":[%zu,%zu],\"data_offsets\":[%llu,%llu]}}",
        scale_name, nB, kB,
        (unsigned long long) off0, (unsigned long long) off1,
        name, N, K,
        (unsigned long long) off1, (unsigned long long) off2);
    if (hlen < 0 || (size_t) hlen >= sizeof(header)) {
        fprintf(stderr, "header too long\n");
        return 1;
    }
    /* pad JSON to 8-byte multiple (safetensors convention) */
    while ((hlen % 8) != 0) {
        header[hlen++] = ' ';
        if ((size_t) hlen >= sizeof(header)) {
            fprintf(stderr, "header pad overflow\n");
            return 1;
        }
    }
    header[hlen] = 0;

    FILE *fp = fopen(path, "wb");
    if (!fp) { perror(path); return 1; }
    uint64_t hl = (uint64_t) hlen;
    if (fwrite(&hl, 8, 1, fp) != 1 ||
        fwrite(header, 1, hlen, fp) != (size_t) hlen ||
        fwrite(scales, 2, nB * kB, fp) != nB * kB ||
        fwrite(q, 1, N * K, fp) != N * K) {
        perror("write safetensors");
        fclose(fp);
        return 1;
    }
    fclose(fp);
    return 0;
}

static void usage(void) {
    fprintf(stderr,
        "qnn-encode hf8|bf8 [in.f32] [out.f8]\n"
        "qnn-encode hf8-block N K [B] in.f32 out.hf8 out.scale.bf16\n"
        "qnn-encode hf8-block N K [B] in.f32 out.safetensors [tensor_name]\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        float sample[] = { 0.f, 1.f, -1.f, 0.1f, -3.5f, 16.f, 1e-4f };
        FILE * tmp = tmpfile();
        if (!tmp) return 1;
        fwrite(sample, sizeof sample, 1, tmp);
        rewind(tmp);
        int rc = run_stream("hf8", tmp, NULL);
        rewind(tmp);
        rc |= run_stream("bf8", tmp, NULL);
        fclose(tmp);
        return rc;
    }

    if (strcmp(argv[1], "hf8-block") == 0) {
        size_t N = 0, K = 0, B = 128;
        const char *in_path = NULL;
        const char *out_a = NULL;
        const char *out_b = NULL;
        const char *tname = "weight";
        int st_mode = 0;

        /*
         * Raw:    hf8-block N K    in out.hf8 out.scale.bf16     (argc 7)
         *         hf8-block N K B  in out.hf8 out.scale.bf16     (argc 8)
         * ST:     hf8-block N K    in out.safetensors [name]     (argc 6|7)
         *         hf8-block N K B  in out.safetensors [name]     (argc 7|8)
         */
        if (argc < 6) { usage(); return 2; }

        N = (size_t) strtoull(argv[2], NULL, 10);
        K = (size_t) strtoull(argv[3], NULL, 10);

        /* detect whether argv[4] is B (digits-only path that isn't a filename with digits — use: if argc allows and next looks like path) */
        int has_B = 0;
        if (argc >= 7) {
            /* argv[4] is B if argv[5] exists as the input file path candidate:
               - ST argc7: N K in out.st name  -> argv[4]=in (not B)
               - ST argc7: N K B in out.st     -> argv[6] ends .safetensors
               - raw argc7: N K in ow os       -> argv[5] not .safetensors
             */
            if (ends_with(argv[argc - 1], ".safetensors") ||
                (argc >= 6 && ends_with(argv[5], ".safetensors")) ||
                (argc >= 7 && ends_with(argv[6], ".safetensors"))) {
                st_mode = 1;
            }
            if (st_mode) {
                if (ends_with(argv[5], ".safetensors")) {
                    /* N K in out.st [name] */
                    in_path = argv[4];
                    out_a = argv[5];
                    if (argc >= 7) tname = argv[6];
                } else if (ends_with(argv[6], ".safetensors")) {
                    /* N K B in out.st [name] */
                    has_B = 1;
                    B = (size_t) strtoull(argv[4], NULL, 10);
                    in_path = argv[5];
                    out_a = argv[6];
                    if (argc >= 8) tname = argv[7];
                } else {
                    usage();
                    return 2;
                }
            } else {
                /* raw two-file */
                if (argc == 7) {
                    in_path = argv[4];
                    out_a = argv[5];
                    out_b = argv[6];
                } else if (argc == 8) {
                    has_B = 1;
                    B = (size_t) strtoull(argv[4], NULL, 10);
                    in_path = argv[5];
                    out_a = argv[6];
                    out_b = argv[7];
                } else {
                    usage();
                    return 2;
                }
            }
        } else if (argc == 6) {
            /* N K in out.safetensors */
            if (!ends_with(argv[5], ".safetensors")) { usage(); return 2; }
            st_mode = 1;
            in_path = argv[4];
            out_a = argv[5];
        } else {
            usage();
            return 2;
        }

        (void) has_B;
        if (N == 0 || K == 0 || B == 0) {
            fprintf(stderr, "N, K, B must be > 0\n");
            return 2;
        }

        FILE * in = fopen(in_path, "rb");
        if (!in) { perror(in_path); return 1; }

        uint8_t *q = NULL;
        uint16_t *scales = NULL;
        size_t nB = 0, kB = 0;
        double max_abs = 0, rmse = 0;
        int rc = quant_hf8_block(N, K, B, in, &q, &scales, &nB, &kB, &max_abs, &rmse);
        fclose(in);
        if (rc) return rc;

        printf("hf8-block N=%zu K=%zu B=%zu scales=%zux%zu max_abs=%.8g rmse=%.8g\n",
               N, K, B, nB, kB, max_abs, rmse);

        if (st_mode) {
            rc = write_safetensors_fp8_pair(out_a, tname, N, K, nB, kB, q, scales);
            if (rc == 0)
                printf("wrote %s  (%s + %s_scale_inv)\n", out_a, tname, tname);
        } else {
            FILE * out_w = fopen(out_a, "wb");
            FILE * out_s = fopen(out_b, "wb");
            if (!out_w) { perror(out_a); rc = 1; }
            else if (!out_s) { perror(out_b); rc = 1; }
            else if (fwrite(q, 1, N * K, out_w) != N * K) {
                perror("write weight"); rc = 1;
            } else if (fwrite(scales, 2, nB * kB, out_s) != nB * kB) {
                perror("write scale"); rc = 1;
            }
            if (out_w) fclose(out_w);
            if (out_s) fclose(out_s);
        }

        free(q); free(scales);
        return rc;
    }

    const char * kind = argv[1];
    FILE * in = argc > 2 ? fopen(argv[2], "rb") : stdin;
    FILE * out = argc > 3 ? fopen(argv[3], "wb") : NULL;
    if (!in) { perror(argv[2]); return 1; }
    if (argc > 3 && !out) { perror(argv[3]); return 1; }
    int rc = run_stream(kind, in, out);
    if (in != stdin) fclose(in);
    if (out) fclose(out);
    return rc;
}
