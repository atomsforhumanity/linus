// The support kernels of the density-fitted Coulomb terms in Hermite space (src/jengine.rs; used by
// src/grad/df_j.rs and src/jdirect_hermite.rs). One writer per output, or fixed-point sums (det_add).

// Over the eri3c builders' records: per primitive i of a class `rs` values [p, Px, Py, Pz, q, E
// nonzeros of the sets];
// per pair out[4] = (layout row of (0, 0), row strides of the kernel's first and second index,
// swap flag) and atom[2] (the atoms of the pair's shells sa, sb). w: the density per layout row
// (df_density_rows: D_mn + D_nm off the diagonal shell blocks).

// pr[4 i ..] = p, Px, Py, Pz of the class's primitive i
extern "C" __global__ void dfj_prims(long long nprim, int rs, const double *__restrict__ rec,
    double *__restrict__ pr)
{
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= 4 * nprim) return;
    const long long i = x / 4;
    pr[x] = rec[i * rs + (x - 4 * i)];
}

// h[i nh + r] = sum_{j in row r} rec[i rs + 5 + he[j]] w[row(hm[j])], row(mn) = out[0] +
// (mn / n2) out[1] + (mn % n2) out[2]: the Hermite density of the class's primitive i
extern "C" __global__ void dfj_pair_density(long long nprim, int nh, int rs, int n2,
    const double *__restrict__ rec, const int *__restrict__ prim_pair,
    const long long *__restrict__ out, const int *__restrict__ hp, const int *__restrict__ he,
    const int *__restrict__ hm, const double *__restrict__ w, double *__restrict__ h)
{
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nprim * nh) return;
    const long long i = x / nh;
    const int r = (int)(x - i * nh);
    const long long *po = out + 4 * (long long)prim_pair[i];
    double s = 0.0;
    for (int j = hp[r]; j < hp[r + 1]; ++j) {
        const int mn = hm[j], i1 = mn / n2, i2 = mn - i1 * n2;
        s = fma(rec[i * rs + 5 + he[j]], w[po[0] + i1 * po[1] + i2 * po[2]], s);
    }
    h[x] = s;
}

// The pair centres' part: per primitive i and derivative set s (A x, y, z, B x, y, z; A the
// kernel's first shell), g = sum_{j in set s} rec[i rs + 5 + j] w[row(gm[j])] a1[i nh1 + gr[j]]
// added to the atom of A or B (the record's sets back to back: gs[s] .. gs[s + 1])
extern "C" __global__ void dfj_pair_grad(long long nprim, int rs, int nh1, int n2,
    const double *__restrict__ rec, const int *__restrict__ prim_pair,
    const long long *__restrict__ out, const int *__restrict__ atom, const int *__restrict__ gs,
    const int *__restrict__ gr, const int *__restrict__ gm, const double *__restrict__ w,
    const double *__restrict__ a1, unsigned long long *__restrict__ gacc, long long natom3)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nprim) return;
    const int pair = prim_pair[i];
    const long long *po = out + 4 * (long long)pair;
    const int swp = (int)po[3];
    const int at1 = atom[2 * pair + swp], at2 = atom[2 * pair + 1 - swp];
    const double *ai = a1 + i * nh1;
    for (int s = 0; s < 6; ++s) {
        double g = 0.0;
        for (int j = gs[s]; j < gs[s + 1]; ++j) {
            const int mn = gm[j], i1 = mn / n2, i2 = mn - i1 * n2;
            g = fma(rec[i * rs + 5 + j] * w[po[0] + i1 * po[1] + i2 * po[2]], ai[gr[j]], g);
        }
        det_add(gacc, natom3, 3 * (long long)(s < 3 ? at1 : at2) + s % 3, g);
    }
}

// J's layout rows (the memory-lean DF-J): rows[row(pair, mn)] = sum_{i in the pair's primitives}
// sum_{j in mn's nonzeros} rec[i rs + 5 + me[j]] a[i nh + mr[j]], one thread per row, the
// primitives and the record's nonzeros in a fixed order (a: the pair primitives' accumulators
// against the fit)
extern "C" __global__ void dfj_pair_rows(long long npair, int nab, int nh, int rs, int n2,
    const int *__restrict__ p0, const double *__restrict__ rec,
    const long long *__restrict__ out, const int *__restrict__ mp, const int *__restrict__ me,
    const int *__restrict__ mr, const double *__restrict__ a, double *__restrict__ rows)
{
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= npair * nab) return;
    const long long pair = x / nab;
    const int mn = (int)(x - pair * nab), i1 = mn / n2, i2 = mn - i1 * n2;
    const long long *po = out + 4 * pair;
    double s = 0.0;
    for (int i = p0[pair]; i < p0[pair + 1]; ++i)
        for (int j = mp[mn]; j < mp[mn + 1]; ++j)
            s = fma(rec[(long long)i * rs + 5 + me[j]], a[(long long)i * nh + mr[j]], s);
    rows[po[0] + i1 * po[1] + i2 * po[2]] = s;
}

// Auxiliary primitives of degree l (nc = (l + 1)(l + 2) / 2 Hermite functions of degree l, nf
// functions per shell, S[m nc + u] the solid harmonics in Hermite order, w the primitive's
// c (2a)^-l): g[i nc + u] = w_i sum_m c[col + m] S[m nc + u]
extern "C" __global__ void dfj_aux_density(long long nprim, int nc, int nf,
    const int *__restrict__ pshell, const int *__restrict__ col, const double *__restrict__ w,
    const double *__restrict__ S, const double *__restrict__ c, double *__restrict__ g)
{
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nprim * nc) return;
    const long long i = x / nc;
    const int u = (int)(x - i * nc);
    const double *cs = c + col[pshell[i]];
    double s = 0.0;
    for (int m = 0; m < nf; ++m) s = fma(cs[m], S[m * nc + u], s);
    g[x] = w[i] * s;
}

// gamma[col_s + m] = sum_{prims i of shell s} w_i sum_u S[m nc + u] a[i nc + u]
extern "C" __global__ void dfj_aux_gamma(long long nshell, int nc, int nf,
    const int *__restrict__ p0, const int *__restrict__ col, const double *__restrict__ w,
    const double *__restrict__ S, const double *__restrict__ a, double *__restrict__ gamma)
{
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nshell * nf) return;
    const long long s = x / nf;
    const int m = (int)(x - s * nf);
    double t = 0.0;
    for (int i = p0[s]; i < p0[s + 1]; ++i) {
        double v = 0.0;
        for (int u = 0; u < nc; ++u) v = fma(S[m * nc + u], a[(long long)i * nc + u], v);
        t = fma(w[i], v, t);
    }
    gamma[col[s] + m] = t;
}

// The auxiliary centres' part: per primitive i, g_k = sum_u g[i nc + u] a1[i nc1 + up[k nc + u]]
// (a1: the accumulators of degree exactly l + 1, up the position of u + e_k among them) added
// to the shell's atom
extern "C" __global__ void dfj_aux_grad(long long nprim, int nc, int nc1,
    const int *__restrict__ pshell, const int *__restrict__ atom, const int *__restrict__ up,
    const double *__restrict__ g, const double *__restrict__ a1,
    unsigned long long *__restrict__ gacc, long long natom3)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nprim) return;
    const long long at = atom[pshell[i]];
    for (int k = 0; k < 3; ++k) {
        double s = 0.0;
        for (int u = 0; u < nc; ++u) s = fma(g[i * nc + u], a1[i * nc1 + up[k * nc + u]], s);
        det_add(gacc, natom3, 3 * at + k, s);
    }
}
