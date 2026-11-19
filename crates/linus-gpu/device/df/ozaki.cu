
// linus-gpu DF exchange in Ozaki slices (src/df/mod.rs).

extern "C" __global__ void df_oz_fill(long long n, int v, int *__restrict__ a)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) a[t] = v;
}

// map[m nbf + n] = the stored row holding (m, n) in either orientation (-1 for dropped pairs,
// filled before): the dense planes A[q][m][n] = B[map[m nbf + n]][q], as df_unpack writes them.
extern "C" __global__ void df_oz_map(long long nx, long long nbf, const int *__restrict__ rm,
    const int *__restrict__ rn, const int *__restrict__ kind, int *__restrict__ map)
{
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const long long m = rm[x], n = rn[x];
    map[m * nbf + n] = (int)x;
    if (kind[x] == 0) map[n * nbf + m] = (int)x;
}

// e[e_off + q nbf + m] (q < nqb): the scale exponent of the vector (q, m) of GEMM1, the
// largest |A[q][m][n]| over n. b: the block's rows, B[x][q] at b[x ldb + q]. Block (32, 8),
// threads.x along q; grid (nbf, nqb / 32).
extern "C" __global__ void __launch_bounds__(256) df_oz_exponents(long long nbf, int nqb,
    const int *__restrict__ map, const double *__restrict__ b, long long ldb, long long e_off,
    int *__restrict__ e)
{
    __shared__ double red[8][32];
    const long long m = blockIdx.x;
    const int q = blockIdx.y * 32 + threadIdx.x;
    double mx = 0.0;
    if (q < nqb)
        for (long long n = threadIdx.y; n < nbf; n += 8) {
            const int x = map[m * nbf + n];
            if (x >= 0) mx = fmax(mx, fabs(b[(long long)x * ldb + q]));
        }
    red[threadIdx.y][threadIdx.x] = mx;
    __syncthreads();
    if (threadIdx.y == 0 && q < nqb) {
        for (int i = 1; i < 8; ++i) mx = fmax(mx, red[i][threadIdx.x]);
        e[e_off + (long long)q * nbf + m] = emu_exponent(mx);
    }
}

// The block's GEMM1 operand in slices: vector r = q nbf + m (q < nqb) holds A[q][m][n] over n,
// its digits at out[(r s + p) kp + n] (n in [nbf, kp) as 0) and its exponent e[e_off + r]
// copied to eo[r]. A tile of 32 q x 128 n of one m goes through shared memory: read along q
// (the stored rows, 256 bytes per row), written along n, four digits of a slice per 32-bit
// store. Block (32, 8); grid (nbf, kp / 128, nqb / 32).
extern "C" __global__ void __launch_bounds__(256) df_oz_slice(long long nbf, int nqb,
    long long kp, int s, const int *__restrict__ map, const double *__restrict__ b,
    long long ldb, const int *__restrict__ e, long long e_off, signed char *__restrict__ out,
    int *__restrict__ eo)
{
    __shared__ double t[128][33];
    const long long m = blockIdx.x;
    const long long n0 = (long long)blockIdx.y * 128;
    const int q0 = blockIdx.z * 32;
    for (int i = threadIdx.y; i < 128; i += blockDim.y) {
        const long long n = n0 + i;
        const int q = q0 + threadIdx.x;
        double v = 0.0;
        if (n < nbf && q < nqb) {
            const int x = map[m * nbf + n];
            if (x >= 0) v = b[(long long)x * ldb + q];
        }
        t[i][threadIdx.x] = v;
    }
    __syncthreads();
    const long long n = n0 + 4 * threadIdx.x;
    if (n >= kp) return;
    for (int i = threadIdx.y; i < 32; i += blockDim.y) {
        const int q = q0 + i;
        if (q >= nqb) break;
        const long long r = (long long)q * nbf + m;
        const int ex = e[e_off + r];
        long long f[4];
        for (int u = 0; u < 4; ++u) f[u] = emu_fixed(t[4 * threadIdx.x + u][i], ex, s);
        unsigned int *o = (unsigned int *)(out + r * s * kp + n);
        for (int p = 0; p < s; ++p) {
            unsigned int w = 0;
            for (int u = 0; u < 4; ++u)
                w |= ((unsigned int)emu_digit(f[u], s, p) & 0xffu) << (8 * u);
            o[(long long)p * kp / 4] = w;
        }
        if (n == 0) eo[r] = ex;
    }
}
