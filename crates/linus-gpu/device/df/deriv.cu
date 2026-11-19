// The DF derivative engines' shared kernels (src/df_deriv.rs): the gradient (src/grad/df.rs) and
// the Hessian (src/hess/df.rs) unpack fitted blocks and form Gamma alike. Rows x of the
// pair-major layout carry the functions (m, n) and a kind (0: an off-diagonal shell pair; else a
// diagonal shell block, both triangles stored); dense blocks are [m - m0][n][P] with n < m1.

// ad[((m - m0) m1 + n) naux + P] = h a[(x - x0) naux + P] for the rows x0.. of a block, h = 1/2 on
// diagonal shell blocks with `half`, else 1; with `mirror`, an off-diagonal shell pair with n
// inside the block also sets the transposed entry
extern "C" __global__ void dd_unpack(long long nrows, long long x0, int naux, long long m0,
    long long m1, int half, int mirror, const int *__restrict__ rm, const int *__restrict__ rn,
    const int *__restrict__ kind, const double *__restrict__ a, double *__restrict__ ad)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nrows * naux) return;
    const long long r = t / naux, P = t - r * naux;
    const long long x = x0 + r;
    const long long m = rm[x], n = rn[x];
    const int k = kind[x];
    const double v = half && k != 0 ? 0.5 * a[t] : a[t];
    ad[((m - m0) * m1 + n) * naux + P] = v;
    if (mirror && k == 0 && n >= m0) ad[((n - m0) * m1 + m) * naux + P] = v;
}

// Gamma of rows xs.. (nrows) of a sub-block: g[(x - xs) naux + P] = s f (c_P D_mn - Zd[m][n][P]),
// f = 2 off the diagonal shell blocks; the c_P D_mn term with `with_j`, D_mn = (D[n][m] +
// D[m][n]) / 2 with `sym`, else D[n][m] (a symmetric D, n x n); the Zd term with `with_k`, Zd the
// block's [m - m0][n][P] (n < m1)
extern "C" __global__ void dd_gamma(long long nrows, long long xs, int naux, double s, int sym,
    const int *__restrict__ rm, const int *__restrict__ rn, const int *__restrict__ kind,
    long long nbf, const double *__restrict__ dens, const double *__restrict__ cfit, int with_j,
    const double *__restrict__ zd, int with_k, long long m0, long long m1, double *__restrict__ g)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nrows * naux) return;
    const long long r = t / naux, P = t - r * naux;
    const long long x = xs + r;
    const long long m = rm[x], n = rn[x];
    double v = 0.0;
    if (with_j)
        v = cfit[P] * (sym ? 0.5 * (dens[n * nbf + m] + dens[m * nbf + n]) : dens[n * nbf + m]);
    if (with_k) v -= zd[((m - m0) * m1 + n) * naux + P];
    g[t] = s * (kind[x] == 0 ? 2.0 : 1.0) * v;
}
