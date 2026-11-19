
// linus-gpu DF gradient (src/grad/df.rs). Rows x of the pair-major layout carry the functions
// (m, n) and a kind (0 off-diagonal shell pair, 1/2 diagonal shell block); dense blocks are
// [m - m0][n][P] with n < m1; orbital-pair tensors [i][j][P] (P fastest). The unpack of dense
// blocks and Gamma are the DF derivative engines' common kernels (df/deriv.cu); the packed
// index is common.cuh's.

// xp[k naux + P] = (y_ij + y_ji) s_ij for the packed pairs k = (i >= j), s = 1 on the
// diagonal and sqrt 2 off it (so that dot products over k are the full sums over i, j)
extern "C" __global__ void gd_pack(int o, int naux, const double *__restrict__ y,
    double *__restrict__ xp)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long npair = (long long)o * (o + 1) / 2;
    if (t >= npair * naux) return;
    const long long k = t / naux, P = t - k * naux;
    long long i, j;
    linus_packed_ij(k, i, j);
    const double v = y[(i * o + j) * naux + P] + y[(j * o + i) * naux + P];
    xp[t] = i == j ? v : v * 1.4142135623730951;
}

// x[(i o + j) naux + P] = x[(j o + i) naux + P] = xp[k naux + P] / s_ij
extern "C" __global__ void gd_unpack_pairs(int o, int naux, const double *__restrict__ xp,
    double *__restrict__ x)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long npair = (long long)o * (o + 1) / 2;
    if (t >= npair * naux) return;
    const long long k = t / naux, P = t - k * naux;
    long long i, j;
    linus_packed_ij(k, i, j);
    const double v = i == j ? xp[t] : xp[t] * 0.7071067811865476;
    x[(i * o + j) * naux + P] = v;
    x[(j * o + i) * naux + P] = v;
}

// The three-center contraction of one sub-block (rows xs.., planes of ps elements): block
// (P tile, chunk); thread P walks the chunk's pairs and rows, G = f (c_P D_mn - Zd[m][n][P]);
// pair sums to the pair's atoms (block sums, fixed point), auxiliary sums -(dA + dB) G to P
// (fixed point). split: J (term 0) and K (term 1) kept apart.
extern "C" __global__ void __launch_bounds__(128) gd_contract(int naux, long long xs,
    long long ps, int ch0, const int *__restrict__ cp0, const int *__restrict__ cp1,
    const long long *__restrict__ px0, const int *__restrict__ pnab,
    const int *__restrict__ pat_a, const int *__restrict__ pat_b,
    const double *__restrict__ pf, const int *__restrict__ rm, const int *__restrict__ rn,
    long long nbf, const double *__restrict__ dens, const double *__restrict__ cfit,
    int with_k, const double *__restrict__ zd, long long m0, long long m1, int split, int natom,
    const double *__restrict__ d, unsigned long long *acc_atom, long long n_atom,
    unsigned long long *acc_aux, long long n_aux)
{
    const int ch = ch0 + blockIdx.y;
    const int P = blockIdx.x * blockDim.x + threadIdx.x;
    const int on = P < naux;
    const int nt = split ? 2 : 1;
    const double cP = on ? cfit[P] : 0.0;
    double ax[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int p = cp0[ch]; p < cp1[ch]; ++p) {
        double s[12];
        for (int k = 0; k < 12; ++k) s[k] = 0.0;
        if (on) {
            const double f = pf[p];
            const long long x0 = px0[p];
            const int nr = pnab[p];
            for (int r = 0; r < nr; ++r) {
                const long long x = x0 + r;
                const long long m = rm[x], n = rn[x];
                const double gj = f * cP * (0.5 * (dens[n * nbf + m] + dens[m * nbf + n]));
                const double gk = with_k ? -f * zd[((m - m0) * m1 + n) * naux + P] : 0.0;
                const long long o = (x - xs) * naux + P;
                for (int k = 0; k < 3; ++k) {
                    const double a = d[k * ps + o], b = d[(3 + k) * ps + o];
                    if (split) {
                        s[k] += a * gj;
                        s[3 + k] += b * gj;
                        s[6 + k] += a * gk;
                        s[9 + k] += b * gk;
                    } else {
                        const double g = gj + gk;
                        s[k] += a * g;
                        s[3 + k] += b * g;
                    }
                }
            }
        }
        const long long aa = pat_a[p], bb = pat_b[p];
        for (int t = 0; t < nt; ++t) {
            for (int k = 0; k < 3; ++k) {
                det_add_block(acc_atom, n_atom, (t * natom + aa) * 3 + k, s[t * 6 + k]);
                det_add_block(acc_atom, n_atom, (t * natom + bb) * 3 + k, s[t * 6 + 3 + k]);
                ax[t * 3 + k] -= s[t * 6 + k] + s[t * 6 + 3 + k];
            }
        }
    }
    if (on) {
        for (int t = 0; t < nt; ++t)
            for (int k = 0; k < 3; ++k)
                det_add(acc_aux, n_aux, ((long long)t * naux + P) * 3 + k, ax[t * 3 + k]);
    }
}
