
// linus-gpu DF Hessian terms (src/hess/df.rs). Pair rows x carry (m, n) and a kind (0: off-diagonal
// shell pair; else a diagonal shell block, both triangles stored). Packed index pk = i (i + 1) / 2 + j,
// i >= j (common.cuh's linus_packed_ij). The MO pass's unpack and Gamma are the DF derivative
// engines' common kernels (df/deriv.cu), the vector operations blas1.cu's.

// Dense derivative block of nq auxiliary functions from the six planes d (rows x nq each):
// ib[((k nbf + m) nq + q) nbf + n] = (d_k m n|q), and for off-diagonal shell pairs
// ib[((k nbf + n) nq + q) nbf + m] = (m d_k n|q)
extern "C" __global__ void hd_unpack_der(long long nrows, int nq, long long nbf,
    const int *__restrict__ rm, const int *__restrict__ rn, const int *__restrict__ kind,
    const double *__restrict__ d, double *__restrict__ ib)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nrows * nq) return;
    const long long r = t / nq, q = t - r * nq;
    const long long m = rm[r], n = rn[r];
    const long long ps = nrows * nq;
    for (int k = 0; k < 3; ++k) ib[((k * nbf + m) * nq + q) * nbf + n] = d[k * ps + t];
    if (kind[r] == 0)
        for (int k = 0; k < 3; ++k) ib[((k * nbf + n) * nq + q) * nbf + m] = d[(3 + k) * ps + t];
}

// hq[(k nq + q) nbf + m] (+)= f sum_i hs[((k nbf + m) nq + q) o + i] c[m + i nbf]
extern "C" __global__ void hd_hq(int nq, long long nbf, int o, double f,
    const double *__restrict__ hs, const double *__restrict__ c, int acc, double *__restrict__ hq)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 3 * nq * nbf) return;
    const long long k = t / (nq * nbf), rem = t - k * nq * nbf;
    const long long q = rem / nbf, m = rem - q * nbf;
    const double *h = hs + ((k * nbf + m) * nq + q) * o;
    double v = 0.0;
    for (int i = 0; i < o; ++i) v = fma(h[i], c[m + i * nbf], v);
    v *= f;
    hq[t] = acc ? hq[t] + v : v;
}

// gamma^x: gx[(3 a + k) naux + p0 + q] = 2 sum_{m in a} hq - (a == ablk) 2 sum_m hq
extern "C" __global__ void hd_gx(int nq, long long nbf, int natom, int naux, int p0, int ablk,
    const int *__restrict__ ao0, const int *__restrict__ ao1, const double *__restrict__ hq,
    double *__restrict__ gx)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 3LL * nq * natom) return;
    const int a = (int)(t / (3 * nq));
    const int rem = (int)(t - (long long)a * 3 * nq);
    const int k = rem / nq, q = rem - k * nq;
    const double *row = hq + ((long long)k * nq + q) * nbf;
    double v = 0.0;
    for (int m = ao0[a]; m < ao1[a]; ++m) v += row[m];
    v *= 2.0;
    if (a == ablk) {
        double tot = 0.0;
        for (long long m = 0; m < nbf; ++m) tot += row[m];
        v -= 2.0 * tot;
    }
    gx[(3LL * a + k) * naux + p0 + q] = v;
}

// V^x c: vc[(3 a + k) naux + p] = (atom(p) == a) gc[k naux + p] + sum_{Q in a} g[(k naux + Q) naux + p] c[Q]
extern "C" __global__ void hd_vc(int naux, int natom, const int *__restrict__ xr0,
    const int *__restrict__ xr1, const int *__restrict__ aux_atom, const double *__restrict__ g,
    const double *__restrict__ gc, const double *__restrict__ c, double *__restrict__ vc)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 3LL * natom * naux) return;
    const long long x = t / naux, p = t - x * naux;
    const int a = (int)(x / 3), k = (int)(x - 3 * a);
    double v = aux_atom[p] == a ? gc[(long long)k * naux + p] : 0.0;
    for (int Q = xr0[a]; Q < xr1[a]; ++Q) v = fma(g[((long long)k * naux + Q) * naux + p], c[Q], v);
    vc[t] = v;
}

// X[(P o + i) o + j] = (T[(i o + j) naux + P] + T[(j o + i) naux + P]) / 2
extern "C" __global__ void hd_xsym(int o, int naux, const double *__restrict__ tt,
    double *__restrict__ x)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long oo = (long long)o * o;
    if (t >= oo * naux) return;
    const long long P = t / oo, ij = t - P * oo;
    const long long i = ij / o, j = ij - i * o;
    x[t] = 0.5 * (tt[(i * o + j) * naux + P] + tt[(j * o + i) * naux + P]);
}

// xp[P npk + pk] = X_ii, or sqrt 2 X_ij off the diagonal
extern "C" __global__ void hd_xpack(int o, int naux, const double *__restrict__ x,
    double *__restrict__ xp)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long npk = (long long)o * (o + 1) / 2;
    if (t >= npk * naux) return;
    const long long P = t / npk, pk = t - P * npk;
    long long i, j;
    linus_packed_ij(pk, i, j);
    const double v = x[(P * o + i) * o + j];
    xp[t] = i == j ? v : 1.4142135623730951 * v;
}

// zp[(m naux + P) o + i] = z[(m o + i) naux + P]
extern "C" __global__ void hd_ztrans(long long nbf, int o, int naux, const double *__restrict__ z,
    double *__restrict__ zp)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nbf * o * naux) return;
    const long long m = t / ((long long)naux * o), rem = t - m * naux * o;
    const long long P = rem / o, i = rem - P * o;
    zp[t] = z[(m * o + i) * naux + P];
}

// h[((k nbf + a) o + i) naux + p0 + q] = hs[((k nbf + a) nq + q) o + i]
extern "C" __global__ void hd_scatter_h(int nq, long long nbf, int o, int naux, int p0,
    const double *__restrict__ hs, double *__restrict__ h)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 3 * nbf * nq * o) return;
    const long long ka = t / ((long long)nq * o), rem = t - ka * nq * o;
    const long long q = rem / o, i = rem - q * o;
    h[(ka * o + i) * naux + p0 + q] = hs[t];
}

// hq[((k nbf + a) ns + q) o + i] = h[((k nbf + a) o + i) naux + p0 + q]
extern "C" __global__ void hd_gather_h(int ns, long long nbf, int o, int naux, int p0,
    const double *__restrict__ h, double *__restrict__ hq)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 3 * nbf * ns * o) return;
    const long long ka = t / ((long long)ns * o), rem = t - ka * ns * o;
    const long long q = rem / o, i = rem - q * o;
    hq[t] = h[(ka * o + i) * naux + p0 + q];
}

// Packed symmetric parts of C_A^T H^k[A rows] for outputs y (direction yk[y], AO rows
// ya0[y]..ya1[y]): r[(y ns + q) npk + pk] = f s_ij sum_a (c[a][i] h_a[q][j] + c[a][j] h_a[q][i]),
// s = 1 on the diagonal, 1/sqrt 2... i.e. sqrt 2 (S_ij + S_ji) off it and 2 S_ii on it
extern "C" __global__ void hd_symc(int ny, int ns, int o, long long nbf,
    const int *__restrict__ yk, const int *__restrict__ ya0, const int *__restrict__ ya1, double f,
    const double *__restrict__ c, const double *__restrict__ h, double *__restrict__ r,
    long long ystride)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long npk = (long long)o * (o + 1) / 2;
    if (t >= (long long)ny * ns * npk) return;
    const long long y = t / (ns * npk), rem = t - y * ns * npk;
    const long long q = rem / npk, pk = rem - q * npk;
    long long i, j;
    linus_packed_ij(pk, i, j);
    const long long k = yk[y];
    double v = 0.0;
    for (long long a = ya0[y]; a < ya1[y]; ++a) {
        const double *hb = h + ((k * nbf + a) * ns + q) * o;
        v = fma(c[a + i * nbf], hb[j], v);
        v = fma(c[a + j * nbf], hb[i], v);
    }
    r[y * ystride + q * npk + pk] = f * (i == j ? v : 1.4142135623730951 * v);
}

// The packed R^x_q unpacked into U ((ns o) x (n3 o), column-major):
// u[((x o + i) ns + q) o + j] = R^x_q,ij
extern "C" __global__ void hd_unpack_r(int n3, int ns, int o, const double *__restrict__ r,
    double *__restrict__ u)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long no = (long long)ns * o;
    if (t >= no * n3 * o) return;
    const long long col = t / no, row = t - col * no;
    const long long x = col / o, i = col - x * o;
    const long long q = row / o, j = row - q * o;
    const long long npk = (long long)o * (o + 1) / 2;
    const long long hi = i > j ? i : j, lo = i > j ? j : i;
    const double v = r[(x * ns + q) * npk + hi * (hi + 1) / 2 + lo];
    u[t] = i == j ? v : 0.7071067811865476 * v;
}

// K^x C's T^{A,k}: kacc[((3 A + k) o + i) n + b] += sum_{a in A} p[(a o + i) n + b], and for
// A = ablk also - sum_a p (the auxiliary block's atom); p[a] = I^k[a]^T Y[a] (n x o, column-major)
extern "C" __global__ void hd_tseg(int natom, long long nbf, int o, int k, int ablk,
    const int *__restrict__ ao0, const int *__restrict__ ao1, const double *__restrict__ p,
    double *__restrict__ kacc)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long no = nbf * o;
    if (t >= (long long)natom * no) return;
    const int A = (int)(t / no);
    const long long e = t - (long long)A * no;
    double v = 0.0;
    for (long long a = ao0[A]; a < ao1[A]; ++a) v += p[a * no + e];
    if (A == ablk) {
        double tot = 0.0;
        for (long long a = 0; a < nbf; ++a) tot += p[a * no + e];
        v -= tot;
    }
    kacc[(3LL * A + k) * no + e] += v;
}

// t[(i o + j) naux + P] = t[(j o + i) naux + P] = their mean, for i >= j (in place)
extern "C" __global__ void hd_tsym(int o, int naux, double *__restrict__ tt)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long npk = (long long)o * (o + 1) / 2;
    if (t >= npk * naux) return;
    const long long k = t / naux, P = t - k * naux;
    long long i, j;
    linus_packed_ij(k, i, j);
    if (i == j) return;
    const long long a = (i * o + j) * naux + P, b = (j * o + i) * naux + P;
    const double v = 0.5 * (tt[a] + tt[b]);
    tt[a] = v;
    tt[b] = v;
}

