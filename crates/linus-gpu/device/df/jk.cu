
// linus-gpu DF-J/K helpers (src/df/mod.rs). Rows x of the pair-major tensor carry the
// functions (m, n) and a kind: 0 off-diagonal shell pair, 1 diagonal block m >= n, 2 m < n.
// Matrices are nbf x nbf column-major (symmetric ones are layout-agnostic).

// w[d * nx + x] = D_d[m, n] (+ D_d[n, m] off the diagonal shell blocks)
extern "C" __global__ void df_density_rows(long long nx, int nd, long long nbf,
    const int *__restrict__ rm, const int *__restrict__ rn, const int *__restrict__ kind,
    const double *__restrict__ dens, double *__restrict__ w)
{
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const long long m = rm[x], n = rn[x];
    const int k = kind[x];
    for (int d = 0; d < nd; ++d) {
        const double *D = dens + (long long)d * nbf * nbf;
        double v = D[n * nbf + m];
        if (k == 0) v = v + D[m * nbf + n];
        w[(long long)d * nx + x] = v;
    }
}

// J_d[m, n] = J_d[n, m] = jx[d * nx + x]; diagonal blocks from the rows with m >= n (the
// values the CPU DfJk keeps), so every element has one writer. J must be zeroed first.
extern "C" __global__ void df_rows_to_matrix(long long nx, int nd, long long nbf,
    const int *__restrict__ rm, const int *__restrict__ rn, const int *__restrict__ kind,
    const double *__restrict__ jx, double *__restrict__ j)
{
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx || kind[x] == 2) return;
    const long long m = rm[x], n = rn[x];
    for (int d = 0; d < nd; ++d) {
        const double v = jx[(long long)d * nx + x];
        double *J = j + (long long)d * nbf * nbf;
        J[n * nbf + m] = v;
        J[m * nbf + n] = v;
    }
}

// a[q][m][n] (q < nq, nbf x nbf per q) = b[x * ldb + q] for every row x; off-diagonal shell
// pairs also at a[q][n][m]. A 32 x 32 tile goes through shared memory so that the reads are
// contiguous along q and the writes along the rows. Block (32, 8).
#define DF_TILE 32
extern "C" __global__ void __launch_bounds__(256) df_unpack(long long nx, int nq, long long nbf,
    const int *__restrict__ rm, const int *__restrict__ rn, const int *__restrict__ kind,
    const double *__restrict__ b, long long ldb, double *__restrict__ a)
{
    __shared__ double t[DF_TILE][DF_TILE + 1];
    const long long xb = (long long)blockIdx.x * DF_TILE;
    const int qb = blockIdx.y * DF_TILE;
    for (int i = threadIdx.y; i < DF_TILE; i += blockDim.y) {
        const long long x = xb + i;
        const int q = qb + threadIdx.x;
        t[i][threadIdx.x] = (x < nx && q < nq) ? b[x * ldb + q] : 0.0;
    }
    __syncthreads();
    const long long x = xb + threadIdx.x;
    if (x >= nx) return;
    const long long m = rm[x], n = rn[x];
    const int k = kind[x];
    const long long plane = nbf * nbf;
    for (int i = threadIdx.y; i < DF_TILE; i += blockDim.y) {
        const int q = qb + i;
        if (q >= nq) break;
        const double v = t[threadIdx.x][i];
        double *aq = a + (long long)q * plane;
        aq[m * nbf + n] = v;
        if (k == 0) aq[n * nbf + m] = v;
    }
}

// The upper triangle of each nbf x nbf matrix (column-major) from its lower triangle.
extern "C" __global__ void df_mirror_lower(long long nbf, double *__restrict__ k)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nbf * nbf) return;
    const long long i = t % nbf, j = t / nbf;
    if (i >= j) return;
    double *K = k + (long long)blockIdx.y * nbf * nbf;
    K[j * nbf + i] = K[i * nbf + j];
}

// K = M + M^T for each nbf x nbf matrix (in place): the thread of (i, j), i > j, writes both
// elements, the diagonal's doubles it, so every element has one writer.
extern "C" __global__ void df_symmetrize(long long nbf, double *__restrict__ k)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nbf * nbf) return;
    const long long i = t % nbf, j = t / nbf;
    if (i < j) return;
    double *K = k + (long long)blockIdx.y * nbf * nbf;
    if (i == j) {
        K[i * nbf + i] = 2.0 * K[i * nbf + i];
        return;
    }
    const double v = K[j * nbf + i] + K[i * nbf + j];
    K[j * nbf + i] = v;
    K[i * nbf + j] = v;
}
