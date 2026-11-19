
// h[(off + i) nh + r] = f sum_{j in row r} E_i[he[j]] D(o1 + i1, o2 + i2), mn = hm[j] = i1 n2 + i2,
// f = dscale (diagonal pairs) or 2 dscale: the Hermite densities of a class's primitives
extern "C" __global__ void pcm_herm_density(long long nprim, int nh, int nnz, int n2,
    const double *__restrict__ rec, const int *__restrict__ prim_pair, const int *__restrict__ pairs,
    const int *__restrict__ hp, const int *__restrict__ he, const int *__restrict__ hm,
    const double *__restrict__ dens, long long nbf, double dscale, double *__restrict__ h, long long off)
{
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nprim * nh) return;
    const long long i = x / nh;
    const int r = (int)(x - i * nh);
    const int *pi = pairs + 5 * prim_pair[i];
    const long long o1 = pi[0], o2 = pi[1];
    double s = 0.0;
    for (int j = hp[r]; j < hp[r + 1]; ++j) {
        const int mn = hm[j], i1 = mn / n2, i2 = mn - i1 * n2;
        s = fma(rec[i * nnz + he[j]], dens[(o2 + i2) * nbf + o1 + i1], s);
    }
    h[(off + i) * nh + r] = (pi[2] ? dscale : 2.0 * dscale) * s;
}

// out[e] = (base ? base[e] : 0) + sign sum_b part[b n + e], e < n (blocks in order)
extern "C" __global__ void pcm_reduce(long long n, long long nblk, const double *__restrict__ part,
    const double *__restrict__ base, double sign, double *__restrict__ out)
{
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    double s = 0.0;
    for (long long b = 0; b < nblk; ++b) s += part[b * n + e];
    out[e] = (base != 0 ? base[e] : 0.0) + sign * s;
}

// V(o1 + i1, o2 + i2) = V(o2 + i2, o1 + i1) = scale sum_{prims of the pair} sum_{j in row mn}
// E_i[ee[j]] A[(off + i) nh + er[j]]: the matrix block of a pair (a diagonal pair's lower
// triangle, mirrored)
extern "C" __global__ void pcm_matrix(long long npair, int nab, int n2, int nh, int nnz,
    const int *__restrict__ p0, const int *__restrict__ pairs, const int *__restrict__ ep,
    const int *__restrict__ ee, const int *__restrict__ er, const double *__restrict__ rec,
    const double *__restrict__ acc, long long off, long long nbf, double scale, double *__restrict__ v)
{
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= npair * nab) return;
    const long long pair = x / nab;
    const int mn = (int)(x - pair * nab), i1 = mn / n2, i2 = mn - i1 * n2;
    const int *pi = pairs + 5 * pair;
    if (pi[2] && i1 < i2) return;
    double s = 0.0;
    for (int i = p0[pair]; i < p0[pair + 1]; ++i)
        for (int j = ep[mn]; j < ep[mn + 1]; ++j)
            s = fma(rec[(long long)i * nnz + ee[j]], acc[(off + i) * nh + er[j]], s);
    s *= scale;
    const long long a = pi[0] + i1, b = pi[1] + i2;
    v[b * nbf + a] = s;
    v[a * nbf + b] = s;
}

// Per primitive: g_s = f sum over the nonzeros of derivative set s (d/dA x..z, d/dB x..z) of
// E_s D(mn) A1[r], added to atom pi[3] (A) or pi[4] (B), component s % 3, in fixed point
extern "C" __global__ void pcm_grad_atoms(long long nprim, int nnz, int nh1, int n2,
    const double *__restrict__ rec, const int *__restrict__ prim_pair, const int *__restrict__ pairs,
    const int *__restrict__ gs, const int *__restrict__ gr, const int *__restrict__ gm,
    const double *__restrict__ dens, long long nbf, double dscale, const double *__restrict__ a1,
    long long off, unsigned long long *__restrict__ gacc, long long natom3)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nprim) return;
    const int *pi = pairs + 5 * prim_pair[i];
    const long long o1 = pi[0], o2 = pi[1];
    const double f = pi[2] ? dscale : 2.0 * dscale;
    const double *ai = a1 + (off + i) * nh1;
    for (int s = 0; s < 6; ++s) {
        double g = 0.0;
        for (int j = gs[s]; j < gs[s + 1]; ++j) {
            const int mn = gm[j], i1 = mn / n2, i2 = mn - i1 * n2;
            g = fma(rec[i * nnz + j] * dens[(o2 + i2) * nbf + o1 + i1], ai[gr[j]], g);
        }
        det_add(gacc, natom3, 3 * (long long)(s < 3 ? pi[3] : pi[4]) + s % 3, f * g);
    }
}
