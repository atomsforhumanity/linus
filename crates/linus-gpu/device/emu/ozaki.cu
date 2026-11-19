
// Ozaki scheme I on INT8 GEMMs (src/emu/mod.rs). A vector is a row of op(A) or a column of
// op(B); its digits go to out[(r s + pos) kp + kk], pos = p (ascending) or s - 1 - p.

// Per-vector max |x| as the bits of a double (order-free u64 atomicMax).
// Vector r's element kk at x[r ld + kk] (contiguous along k). Grid (rows, k / 2048), 256.
extern "C" __global__ void __launch_bounds__(256) emu_absmax_kc(long long rows, long long k,
    const double *__restrict__ x, long long ld, unsigned long long *__restrict__ mx)
{
    __shared__ double red[256];
    const long long r = blockIdx.x;
    const long long k0 = (long long)blockIdx.y * 2048;
    const long long k1 = k0 + 2048 < k ? k0 + 2048 : k;
    double m = 0.0;
    for (long long kk = k0 + threadIdx.x; kk < k1; kk += blockDim.x)
        m = fmax(m, fabs(x[r * ld + kk]));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h >>= 1) {
        if ((int)threadIdx.x < h) red[threadIdx.x] = fmax(red[threadIdx.x], red[threadIdx.x + h]);
        __syncthreads();
    }
    if (threadIdx.x == 0 && red[0] > 0.0)
        atomicMax(mx + r, (unsigned long long)__double_as_longlong(red[0]));
}

// As emu_absmax_kc with vector r's element kk at x[kk ld + r] (contiguous along the vectors).
// Block (32, 8): 32 vectors x 256 values of k. Grid (rows / 32, k / 256).
extern "C" __global__ void __launch_bounds__(256) emu_absmax_rc(long long rows, long long k,
    const double *__restrict__ x, long long ld, unsigned long long *__restrict__ mx)
{
    __shared__ double red[8][32];
    const long long r = (long long)blockIdx.x * 32 + threadIdx.x;
    const long long k0 = (long long)blockIdx.y * 256;
    const long long k1 = k0 + 256 < k ? k0 + 256 : k;
    double m = 0.0;
    if (r < rows)
        for (long long kk = k0 + threadIdx.y; kk < k1; kk += 8)
            m = fmax(m, fabs(x[kk * ld + r]));
    red[threadIdx.y][threadIdx.x] = m;
    __syncthreads();
    if (threadIdx.y == 0 && r < rows) {
        for (int i = 1; i < 8; ++i) m = fmax(m, red[i][threadIdx.x]);
        if (m > 0.0) atomicMax(mx + r, (unsigned long long)__double_as_longlong(m));
    }
}

// The scale exponent of each vector: max |x| <= 127/128 2^e (0 for a zero vector).
extern "C" __global__ void emu_exponents(long long rows, const unsigned long long *__restrict__ mx,
    int *__restrict__ e)
{
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    e[r] = emu_exponent(__longlong_as_double((long long)mx[r]));
}

// Digits of k-contiguous vectors (element kk of vector r at x[r ld + kk]); kk in [k, kp)
// written as 0. Grid (rows, kp / 256), 256.
extern "C" __global__ void __launch_bounds__(256) emu_slice_kc(long long rows, long long k,
    long long kp, int s, int desc, const double *__restrict__ x, long long ld,
    const int *__restrict__ e, signed char *__restrict__ out)
{
    const long long r = blockIdx.x;
    const long long kk = (long long)blockIdx.y * blockDim.x + threadIdx.x;
    if (kk >= kp) return;
    const double v = kk < k ? x[r * ld + kk] : 0.0;
    emu_digits(v, e[r], s, desc, out + r * s * kp + kk, kp);
}

// Digits of vectors contiguous along r (element kk of vector r at x[kk ld + r]), transposed
// through shared memory. Block (32, 8); grid (rows / 32, kp / 32).
extern "C" __global__ void __launch_bounds__(256) emu_slice_rc(long long rows, long long k,
    long long kp, int s, int desc, const double *__restrict__ x, long long ld,
    const int *__restrict__ e, signed char *__restrict__ out)
{
    __shared__ double t[32][33];
    const long long r0 = (long long)blockIdx.x * 32, k0 = (long long)blockIdx.y * 32;
    for (int i = threadIdx.y; i < 32; i += blockDim.y) {
        const long long kk = k0 + i, r = r0 + threadIdx.x;
        t[i][threadIdx.x] = (kk < k && r < rows) ? x[kk * ld + r] : 0.0;
    }
    __syncthreads();
    const long long kk = k0 + threadIdx.x;
    if (kk >= kp) return;
    for (int i = threadIdx.y; i < 32; i += blockDim.y) {
        const long long r = r0 + i;
        if (r >= rows) break;
        emu_digits(t[threadIdx.x][i], e[r], s, desc, out + r * s * kp + kk, kp);
    }
}

// The right operand of diagonal g in a GEMM over the left operand's digits 0..gmax: for each
// vector c < n, its descending digits g..0 (slots s-1-g.. of b) followed by (gmax - g) zero
// digit blocks: out[c kk + t], kk = (gmax + 1) kp, in 32-bit words. Grid (n, kk / 1024), 256.
extern "C" __global__ void emu_window(long long n, long long kp, int s, int g, int gmax,
    const unsigned int *__restrict__ b, unsigned int *__restrict__ out)
{
    const long long c = blockIdx.x;
    const long long kk = (long long)(gmax + 1) * kp / 4;
    const long long t = (long long)blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= kk) return;
    const long long valid = (long long)(g + 1) * kp / 4;
    out[c * kk + t] = t < valid ? b[(c * s + s - 1 - g) * kp / 4 + t] : 0u;
}

// c[r + col ldc] = alpha 2^(er[r] + ec[col]) sum_j 2^(-14-8 dg[j]) parts_j[r + col ldp]
// + beta c[..] for r < m, col < n: np INT32 partials of ldp x n (column-major, one after
// another) of the diagonals dg[j], in order of decreasing dg (smallest scales first). Partials
// of up to 4 neighbouring diagonals are summed exactly in int64 (shifts of at most 24 bits:
// below 2^58 for up to 8 partials), each such group converted once and the groups summed in
// FP64 in that fixed order. lower: only r >= col.
extern "C" __global__ void emu_combine(long long m, long long n, long long ldp, int np,
    const int *__restrict__ parts, const int *__restrict__ dg, const int *__restrict__ er,
    const int *__restrict__ ec, double alpha, double beta, double *__restrict__ c,
    long long ldc, int lower)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m * n) return;
    const long long r = t % m, col = t / m;
    if (lower && r < col) return;
    const long long at = col * ldp + r;
    double acc = 0.0;
    long long ia = 0;
    int gref = -1, cnt = 0;
    for (int j = 0; j < np; ++j) {
        const int g = dg[j];
        if (gref >= 0 && (gref - g > 3 || cnt == 8)) {
            acc += ldexp((double)ia, -14 - 8 * gref);
            ia = 0;
            gref = -1;
            cnt = 0;
        }
        if (gref < 0) gref = g;
        ia += (long long)parts[(long long)j * ldp * n + at] << (8 * (gref - g));
        ++cnt;
    }
    if (gref >= 0) acc += ldexp((double)ia, -14 - 8 * gref);
    const double v = alpha * ldexp(acc, er[r] + ec[col]);
    double *o = c + col * ldc + r;
    *o = beta == 0.0 ? v : v + beta * *o;
}
