
// erf(z r) / r and its first derivative in r (a series for z r < 0.02, as the CPU's coul_erf)
__device__ __forceinline__ void pcm_coul_erf(double z, double r, double &s, double &s1)
{
    const double x = z * r;
    const double c = 2.0 * z * 0.5641895835477563;   // 2 z / sqrt(pi)
    if (x < 0.02) {
        const double x2 = x * x;
        s = c * (1.0 - x2 / 3.0 + x2 * x2 / 10.0 - x2 * x2 * x2 / 42.0);
        s1 = c * z * x * (-2.0 / 3.0 + 0.4 * x2 - x2 * x2 / 7.0);
        return;
    }
    const double e = erf(x);
    const double g = c * exp(-x * x);
    s = e / r;
    s1 = g / r - e / (r * r);
}

// S (n x n, column-major), one thread per element
extern "C" __global__ void pcm_s_matrix(long long n, const double *__restrict__ r,
    const double *__restrict__ zeta, const double *__restrict__ sw, double *__restrict__ s)
{
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n * n) return;
    const long long j = x / n, i = x - j * n;
    if (i == j) {
        s[x] = zeta[i] * 0.7978845608028654 / sw[i];   // sqrt(2 / pi)
        return;
    }
    const double zi = zeta[i], zj = zeta[j];
    const double zij = zi * zj / sqrt(zi * zi + zj * zj);
    const double dx = r[3 * i] - r[3 * j], dy = r[3 * i + 1] - r[3 * j + 1], dz = r[3 * i + 2] - r[3 * j + 2];
    double v, v1;
    pcm_coul_erf(zij, sqrt(dx * dx + dy * dy + dz * dz), v, v1);
    s[x] = v;
}

// D (n x n, column-major) and D A (columns scaled by the areas), one thread per element
extern "C" __global__ void pcm_d_matrix(long long n, const double *__restrict__ r,
    const double *__restrict__ zeta, const double *__restrict__ nrm, const double *__restrict__ rad,
    const double *__restrict__ area, double *__restrict__ d, double *__restrict__ da)
{
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n * n) return;
    const long long j = x / n, i = x - j * n;
    double v;
    if (i == j) {
        v = -zeta[i] * 0.7978845608028654 / (2.0 * rad[i]);
    } else {
        const double zi = zeta[i], zj = zeta[j];
        const double zij = zi * zj / sqrt(zi * zi + zj * zj);
        const double ux = r[3 * i] - r[3 * j], uy = r[3 * i + 1] - r[3 * j + 1], uz = r[3 * i + 2] - r[3 * j + 2];
        const double rr = sqrt(ux * ux + uy * uy + uz * uz);
        double s, s1;
        pcm_coul_erf(zij, rr, s, s1);
        v = -s1 * (ux * nrm[3 * j] + uy * nrm[3 * j + 1] + uz * nrm[3 * j + 2]) / rr;
    }
    d[x] = v;
    da[x] = v * area[j];
}

// K = S - c (M + sym M^T), M = D A S (n x n, column-major, in place over m)
extern "C" __global__ void pcm_k_matrix(long long n, const double *__restrict__ s, double c, int sym,
    const double *__restrict__ m, double *__restrict__ k)
{
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n * n) return;
    const long long j = x / n, i = x - j * n;
    k[x] = s[x] - c * (sym ? m[x] + m[i * n + j] : m[x]);
}

// erf(z r) / r and its first two derivatives in r (as the CPU's coul_erf)
__device__ __forceinline__ void pcm_coul_erf2(double z, double r, double &s, double &s1, double &s2)
{
    const double x = z * r;
    const double c = 2.0 * z * 0.5641895835477563;
    if (x < 0.02) {
        const double x2 = x * x;
        s = c * (1.0 - x2 / 3.0 + x2 * x2 / 10.0 - x2 * x2 * x2 / 42.0);
        s1 = c * z * x * (-2.0 / 3.0 + 0.4 * x2 - x2 * x2 / 7.0);
        s2 = c * z * z * (-2.0 / 3.0 + 1.2 * x2 - 5.0 * x2 * x2 / 7.0);
        return;
    }
    const double e = erf(x);
    const double g = c * exp(-x * x);
    const double r2 = r * r;
    s = e / r;
    s1 = g / r - e / r2;
    s2 = -g * (2.0 * z * z + 2.0 / r2) + 2.0 * e / (r2 * r);
}

// part[(jc n + i) 3 + k]: sum over j != i of chunk jc of (a_i b_j + a_j b_i) S'(r_ij) u_ij[k] / r_ij
extern "C" __global__ void pcm_ds_points(long long n, long long jch, const double *__restrict__ r,
    const double *__restrict__ zeta, const double *__restrict__ a, const double *__restrict__ b,
    double *__restrict__ part)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long jc = blockIdx.y, j0 = jc * jch, j1 = n < j0 + jch ? n : j0 + jch;
    const double ri0 = r[3 * i], ri1 = r[3 * i + 1], ri2 = r[3 * i + 2], zi = zeta[i];
    double v0 = 0.0, v1 = 0.0, v2 = 0.0;
    for (long long j = j0; j < j1; ++j) {
        if (j == i) continue;
        const double c = a[i] * b[j] + a[j] * b[i];
        if (c == 0.0) continue;
        const double u0 = ri0 - r[3 * j], u1 = ri1 - r[3 * j + 1], u2 = ri2 - r[3 * j + 2];
        const double rr = sqrt(u0 * u0 + u1 * u1 + u2 * u2);
        const double zj = zeta[j];
        double sv, s1;
        pcm_coul_erf(zi * zj / sqrt(zi * zi + zj * zj), rr, sv, s1);
        const double f = c * s1 / rr;
        v0 += f * u0;
        v1 += f * u1;
        v2 += f * u2;
    }
    double *o = part + (jc * n + i) * 3;
    o[0] = v0;
    o[1] = v1;
    o[2] = v2;
}

// grad_u D_ij = g'(r) u_hat (u . n_j) + g(r) n_j, g = -S'(r)/r, u = r_i - r_j
__device__ __forceinline__ void pcm_grad_d(const double *r, const double *zeta, const double *nrm,
    long long i, long long j, double &g0, double &g1, double &g2)
{
    const double u0 = r[3 * i] - r[3 * j], u1 = r[3 * i + 1] - r[3 * j + 1], u2 = r[3 * i + 2] - r[3 * j + 2];
    const double rr = sqrt(u0 * u0 + u1 * u1 + u2 * u2);
    const double zi = zeta[i], zj = zeta[j];
    double s, s1, s2;
    pcm_coul_erf2(zi * zj / sqrt(zi * zi + zj * zj), rr, s, s1, s2);
    const double g = -s1 / rr;
    const double gp = -s2 / rr + s1 / (rr * rr);
    const double n0 = nrm[3 * j], n1 = nrm[3 * j + 1], n2 = nrm[3 * j + 2];
    const double un = u0 * n0 + u1 * n1 + u2 * n2;
    g0 = gp * u0 / rr * un + g * n0;
    g1 = gp * u1 / rr * un + g * n1;
    g2 = gp * u2 / rr * un + g * n2;
}

// part[(jc n + i) 3 + k]: sum over j != i of chunk jc of a_i b_j grad_u D_ij - a_j b_i grad_u D_ji
extern "C" __global__ void pcm_dd_points(long long n, long long jch, const double *__restrict__ r,
    const double *__restrict__ zeta, const double *__restrict__ nrm, const double *__restrict__ a,
    const double *__restrict__ b, double *__restrict__ part)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long jc = blockIdx.y, j0 = jc * jch, j1 = n < j0 + jch ? n : j0 + jch;
    double v0 = 0.0, v1 = 0.0, v2 = 0.0;
    for (long long j = j0; j < j1; ++j) {
        if (j == i) continue;
        double g0, g1, g2;
        const double c1 = a[i] * b[j];
        if (c1 != 0.0) {
            pcm_grad_d(r, zeta, nrm, i, j, g0, g1, g2);
            v0 += c1 * g0;
            v1 += c1 * g1;
            v2 += c1 * g2;
        }
        const double c2 = a[j] * b[i];
        if (c2 != 0.0) {
            pcm_grad_d(r, zeta, nrm, j, i, g0, g1, g2);
            v0 -= c2 * g0;
            v1 -= c2 * g1;
            v2 -= c2 * g2;
        }
    }
    double *o = part + (jc * n + i) * 3;
    o[0] = v0;
    o[1] = v1;
    o[2] = v2;
}

// out[e] = sum_b part[b len + e] in order, e < len
extern "C" __global__ void pcm_sum_parts(long long len, long long nblk, const double *__restrict__ part,
    double *__restrict__ out)
{
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= len) return;
    double s = 0.0;
    for (long long b = 0; b < nblk; ++b) s += part[b * len + e];
    out[e] = s;
}
