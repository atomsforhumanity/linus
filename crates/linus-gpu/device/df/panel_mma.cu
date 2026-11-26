// linus-gpu DF-K: the shell panels' first GEMM on FP64 tensor cores, straight from the stored rows
// (src/df/mod.rs, PanelKernel::Mma and MmaSm90a). NVIDIA-only (inline PTX: mma.sync m16n8k16 f64,
// bulk copies, mbarriers' transaction counts; compute capability 9.0 and later), beside the
// portable path (the gathered panels and cuBLAS). Every output element has one writing thread and
// a fixed summation order (the k loop in steps of 16, each step one tensor-core product per tile),
// so the result is bitwise reproducible, and the same for every copy mode and target.
//
// For function f of the list (fn_tab[5 f ..]: panel row of k = 0, step d, partner columns w,
// first gathered-orbital column c0, the function m) and the chunk's orbital columns
// i < nc (cg column cc + i, cg rows of ldc doubles):
//   X[xo + q + nqb (i + nc m)] = sum_{k < w} B[off + src[r + d k] ld + q] cg[cc + i + ldc (c0 + k)]
// for q < nqb: a GEMM (q x i, inner k) whose first operand's rows are gathered stored rows.
//
// One block per (q tile of DFM_BM, i tile of DFM_BN, function), warp-specialized: the consumer
// warps (DFM_WM x DFM_WN, a multiple of 4, each a (DFM_BM / DFM_WM) x (DFM_BN / DFM_WN) tile of
// 16 x 8 products) only read fragments from shared memory and multiply; a producer warpgroup (4
// warps, one copying) copies each k-step's operands (16 rows of each) into one of DFM_ST stages.
// Per stage an mbarrier "full" (the operands have landed) and "empty" (every consumer has read
// them); no block-wide barrier in the k loop. When the consumers issued the copies themselves, a
// warp stalled at a copy issue held up its products, and the barrier of each k-step the whole
// block: on an H200 the copies and the products then took the sum of their times.
//   DFM_COPY 0: one bulk copy (cp.async.bulk, the TMA unit) per operand row of a stage, the
//               completion counted in bytes on the full barrier (rows 16-byte aligned and an even
//               number of doubles: the host checks);
//            1: cp.async of two doubles per lane (16-byte aligned rows), 2: one double per lane;
//               the full barrier counts the producer lanes' cp.async.mbarrier.arrive.
// Compiled for sm_90a (the architecture-specific target), the producer warpgroup hands its
// registers to the consumers (setmaxnreg: DFM_PREG per producer lane, DFM_CREG per consumer lane);
// for other targets the hand-over is left out (the same products, the consumers capped by the
// launch's register count). The function's stored-row indices (w of them) are read into shared
// memory first.
//
// The host (src/df/mod.rs, panel_mma_source) defines the tile and the register counts.

#if !defined(DFM_BM) || !defined(DFM_BN) || !defined(DFM_WM) || !defined(DFM_WN) || !defined(DFM_ST) || \
    !defined(DFM_COPY) || !defined(DFM_PREG) || !defined(DFM_CREG)
#error "df_panel_mma: the host defines DFM_BM, DFM_BN, DFM_WM, DFM_WN, DFM_ST, DFM_COPY, DFM_PREG and DFM_CREG"
#endif
#define DFM_BK 16
#define DFM_CW (DFM_WM * DFM_WN)
#define DFM_CT (32 * DFM_CW)
#define DFM_THREADS (DFM_CT + 128)
#define DFM_WTM (DFM_BM / DFM_WM)
#define DFM_WTN (DFM_BN / DFM_WN)
#define DFM_TM (DFM_WTM / 16)
#define DFM_TN (DFM_WTN / 8)
// shared rows padded by 4 doubles: the fragment loads of a half-warp hit distinct banks
#define DFM_AP (DFM_BM + 4)
#define DFM_BP (DFM_BN + 4)
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define DFM_SETMAXNREG(op, n) asm volatile("setmaxnreg." op ".sync.aligned.u32 %0;\n" ::"n"(n) : "memory")
#else
#define DFM_SETMAXNREG(op, n) ((void)0)
#endif

__device__ __forceinline__ unsigned dfm_sa(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void dfm_mma(double c[4], const double a[8], const double b[4])
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64 {%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15}, {%0,%1,%2,%3};\n"
                 : "+d"(c[0]), "+d"(c[1]), "+d"(c[2]), "+d"(c[3])
                 : "d"(a[0]), "d"(a[1]), "d"(a[2]), "d"(a[3]), "d"(a[4]), "d"(a[5]), "d"(a[6]),
                   "d"(a[7]), "d"(b[0]), "d"(b[1]), "d"(b[2]), "d"(b[3]));
}

__device__ __forceinline__ void dfm_bar_init(unsigned long long *bar, unsigned n)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(dfm_sa(bar)), "r"(n) : "memory");
}

__device__ __forceinline__ void dfm_bar_arrive(unsigned long long *bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(dfm_sa(bar)) : "memory");
}

// waits for the completion of the barrier's phase of parity `phase`
__device__ __forceinline__ void dfm_bar_wait(unsigned long long *bar, unsigned phase)
{
    unsigned ok = 0;
    while (!ok)
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     " selp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(ok)
                     : "r"(dfm_sa(bar)), "r"(phase)
                     : "memory");
}

#if DFM_COPY == 0
__device__ __forceinline__ void dfm_bar_expect(unsigned long long *bar, unsigned bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(dfm_sa(bar)), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void dfm_bulk(double *dst, const double *src, unsigned bytes,
    unsigned long long *bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 ::"r"(dfm_sa(dst)), "l"(src), "r"(bytes), "r"(dfm_sa(bar))
                 : "memory");
}
#else
// DFM_COPY == 1: two doubles, 2: one double; zero-filled when !ok (the source is not read then)
__device__ __forceinline__ void dfm_cp(double *dst, const double *src, bool ok)
{
#if DFM_COPY == 1
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dfm_sa(dst)), "l"(src), "r"(ok ? 16 : 0));
#else
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(dfm_sa(dst)), "l"(src), "r"(ok ? 8 : 0));
#endif
}

// the barrier counts this lane's arrival once its earlier cp.async copies have landed
__device__ __forceinline__ void dfm_cp_arrive(unsigned long long *bar)
{
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(dfm_sa(bar)) : "memory");
}
#endif

extern "C" __global__ void __launch_bounds__(DFM_THREADS, 1) df_panel_mma(int nqb, int nc, int ldc,
    const int *__restrict__ fn_tab, const int *__restrict__ src, const double *__restrict__ b,
    long long off, long long ld, const double *__restrict__ cg, long long cc,
    double *__restrict__ x, long long xo)
{
    extern __shared__ double dfm_smem[];
    double *as = dfm_smem;                                  // [DFM_ST][DFM_BK][DFM_AP]
    double *bs = dfm_smem + DFM_ST * DFM_BK * DFM_AP;       // [DFM_ST][DFM_BK][DFM_BP]
    unsigned long long *full = (unsigned long long *)(bs + DFM_ST * DFM_BK * DFM_BP); // [DFM_ST]
    unsigned long long *empty = full + DFM_ST;              // [DFM_ST]
    int *ridx = (int *)(empty + DFM_ST);                    // [w]
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int q0 = blockIdx.x * DFM_BM, n0 = blockIdx.y * DFM_BN;
    const int *fe = fn_tab + 5 * (long long)blockIdx.z;
    const int r0 = fe[0], d = fe[1], w = fe[2], c0 = fe[3], m = fe[4];
    const int nk = (w + DFM_BK - 1) / DFM_BK;
    for (int k = tid; k < w; k += DFM_THREADS) ridx[k] = src[r0 + d * k];
    if (tid == 0) {
#pragma unroll
        for (int st = 0; st < DFM_ST; ++st) {
            dfm_bar_init(&full[st], DFM_COPY == 0 ? 1 : 32);
            dfm_bar_init(&empty[st], DFM_CT);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (warp >= DFM_CW) {
        DFM_SETMAXNREG("dec", DFM_PREG);
        if (warp != DFM_CW) return;
        // ---- the producer: k-step s into stage s % DFM_ST once its previous use is read ----
        const double *bq = b + off + q0;
        const double *cgc = cg + cc + (long long)ldc * c0 + n0;
#if DFM_COPY == 0
        // bytes per row: A's nqb - q0 (even), B's nc - n0 rounded up to even (ldc and cc even:
        // the extra double is the row's padding or the next chunk's first column)
        const unsigned abytes = 8u * (unsigned)min(DFM_BM, nqb - q0);
        const unsigned bbytes = 8u * (unsigned)((min(DFM_BN, nc - n0) + 1) & ~1);
#endif
        for (int s = 0; s < nk; ++s) {
            const int st = s % DFM_ST;
            if (s >= DFM_ST) dfm_bar_wait(&empty[st], (unsigned)(s / DFM_ST + 1) & 1u);
            double *a = as + st * DFM_BK * DFM_AP;
            double *bb = bs + st * DFM_BK * DFM_BP;
            const int kb = s * DFM_BK;
#if DFM_COPY == 0
            const int rows = min(DFM_BK, w - kb);
            if (rows < DFM_BK) {
                // rows k >= w: zeros (the panel's last stage; nothing is copied into it later)
                for (int e = lane; e < (DFM_BK - rows) * DFM_AP; e += 32) a[rows * DFM_AP + e] = 0.0;
                for (int e = lane; e < (DFM_BK - rows) * DFM_BP; e += 32) bb[rows * DFM_BP + e] = 0.0;
            }
            __syncwarp();
            if (lane == 0) dfm_bar_expect(&full[st], (unsigned)rows * (abytes + bbytes));
            __syncwarp();
            // lanes < DFM_BK: A's rows; the others: B's rows
            if (lane < DFM_BK) {
                if (lane < rows)
                    dfm_bulk(a + lane * DFM_AP, bq + (long long)ridx[kb + lane] * ld, abytes, &full[st]);
            } else if (lane - DFM_BK < rows) {
                const int kk = lane - DFM_BK;
                dfm_bulk(bb + kk * DFM_BP, cgc + (long long)ldc * (kb + kk), bbytes, &full[st]);
            }
#else
            const int vw = DFM_COPY == 1 ? 2 : 1;
#pragma unroll 4
            for (int kk = 0; kk < DFM_BK; ++kk) {
                const int k = kb + kk;
                const bool kok = k < w;
                const double *ra = bq + (long long)ridx[kok ? k : 0] * ld;
#pragma unroll
                for (int q = vw * lane; q < DFM_BM; q += 32 * vw) {
                    const bool ok = kok && q0 + q < nqb;
                    dfm_cp(a + kk * DFM_AP + q, ok ? ra + q : b, ok);
                }
                const double *rb = cgc + (long long)ldc * k;
#pragma unroll
                for (int n = vw * lane; n < DFM_BN; n += 32 * vw) {
                    const bool ok = kok && n0 + n < nc;
                    dfm_cp(bb + kk * DFM_BP + n, ok ? rb + n : cg, ok);
                }
            }
            dfm_cp_arrive(&full[st]);
#endif
        }
#if DFM_COPY != 0
        asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
        return;
    }

    // ---- the consumers ----
    DFM_SETMAXNREG("inc", DFM_CREG);
    const int g = lane >> 2, t = lane & 3;
    const int wm = warp / DFM_WN, wn = warp % DFM_WN;
    double acc[DFM_TM][DFM_TN][4];
#pragma unroll
    for (int i = 0; i < DFM_TM; ++i)
#pragma unroll
        for (int j = 0; j < DFM_TN; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0;
    for (int s = 0; s < nk; ++s) {
        const int st = s % DFM_ST;
        dfm_bar_wait(&full[st], (unsigned)(s / DFM_ST) & 1u);
        // fragment element e of A: row g + 8 (e & 1), k t + 4 (e >> 1); of B: k t + 4 e,
        // column g. A's rows are permuted (row g + 8 h of the product is the tile's row 2 g + h),
        // so a lane's two rows are adjacent: one 16-byte load per k
        const double *a = as + st * DFM_BK * DFM_AP + wm * DFM_WTM + 2 * g;
        const double *bb = bs + st * DFM_BK * DFM_BP + wn * DFM_WTN + g;
        double bf[DFM_TN][4];
#pragma unroll
        for (int j = 0; j < DFM_TN; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) bf[j][e] = bb[(t + 4 * e) * DFM_BP + 8 * j];
#pragma unroll
        for (int i = 0; i < DFM_TM; ++i) {
            double af[8];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const double2 v = *(const double2 *)(a + (t + 4 * c) * DFM_AP + 16 * i);
                af[2 * c] = v.x;
                af[2 * c + 1] = v.y;
            }
#pragma unroll
            for (int j = 0; j < DFM_TN; ++j) dfm_mma(acc[i][j], af, bf[j]);
        }
        // every fragment of the stage is in registers: the producer may refill it
        dfm_bar_arrive(&empty[st]);
    }
    // X[xo + q + nqb (i + nc m)]; a lane's rows 2 g and 2 g + 1 are adjacent in X
    double *xm = x + xo + (long long)nqb * nc * m;
    const bool pair = ((xo | (long long)nqb) & 1) == 0;
#pragma unroll
    for (int i = 0; i < DFM_TM; ++i)
#pragma unroll
        for (int j = 0; j < DFM_TN; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                // accumulator e: row 2 g, column 2 t + e; e + 2: row 2 g + 1, the same column
                const int q = q0 + wm * DFM_WTM + 16 * i + 2 * g;
                const int n = n0 + wn * DFM_WTN + 8 * j + 2 * t + e;
                if (n >= nc) continue;
                double *p = xm + q + (long long)nqb * n;
                if (pair && q + 1 < nqb) {
                    *(double2 *)p = make_double2(acc[i][j][e], acc[i][j][2 + e]);
                } else {
                    if (q < nqb) p[0] = acc[i][j][e];
                    if (q + 1 < nqb) p[1] = acc[i][j][2 + e];
                }
            }
}
