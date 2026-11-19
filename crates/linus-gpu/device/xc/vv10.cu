
// linus-gpu VV10 kernels (src/xc/vv10.rs). The active points a < n in ascending grid order
// (idx[a] = the grid point); their quantities in arrays of npad values (npad a multiple of
// VV_TILE), q[k npad + a] for k = 0..8: x, y, z (relative to the host's centre), omega, kappa,
// rho w, rho domega/drho, rho domega/dsigma, rho dkappa/drho. Padding points (n <= a < npad)
// sit at the centre with omega = kappa = 1 and rho w = 0: their pair terms are exactly zero.

#define VV_CB 1024

// counts[b] = the active points among [b VV_CB, (b + 1) VV_CB) (blockDim = VV_CB)
extern "C" __global__ void vv10_count(long long npts, double thresh,
    const double *__restrict__ rho, int *__restrict__ counts)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int a = p < npts && rho[p] >= thresh;
    const int c = __syncthreads_count(a);
    if (threadIdx.x == 0) counts[blockIdx.x] = c;
}

// off[b] = the active points before block b, off[nb] = all of them (one thread; nb is small)
extern "C" __global__ void vv10_scan(int nb, const int *__restrict__ counts,
    int *__restrict__ off)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int s = 0;
    for (int b = 0; b < nb; ++b) {
        off[b] = s;
        s += counts[b];
    }
    off[nb] = s;
}

// idx[off[b] + rank] = p, rank the point's place among its block's active points (blockDim =
// VV_CB; an inclusive scan of the flags in shared memory)
extern "C" __global__ void vv10_compact(long long npts, double thresh,
    const double *__restrict__ rho, const int *__restrict__ off, int *__restrict__ idx)
{
    __shared__ int s[VV_CB];
    const int t = threadIdx.x;
    const long long p = (long long)blockIdx.x * blockDim.x + t;
    const int a = p < npts && rho[p] >= thresh;
    s[t] = a;
    __syncthreads();
    for (int d = 1; d < (int)blockDim.x; d <<= 1) {
        const int v = t >= d ? s[t - d] : 0;
        __syncthreads();
        s[t] += v;
        __syncthreads();
    }
    if (a) idx[off[blockIdx.x] + s[t] - 1] = (int)p;
}

// the active points' quantities (FP64; the CPU's formulas), the padding up to npad
extern "C" __global__ void vv10_prep(int n, int npad, double cvv, double kvv, double pi43,
    double cx, double cy, double cz, const int *__restrict__ idx, const double *__restrict__ px,
    const double *__restrict__ py, const double *__restrict__ pz, const double *__restrict__ w,
    const double *__restrict__ rho, const double *__restrict__ grad, double *__restrict__ q)
{
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= npad) return;
    double v[9] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    if (a < n) {
        const long long p = idx[a];
        const double r = rho[p];
        const double g0 = grad[3 * p], g1 = grad[3 * p + 1], g2 = grad[3 * p + 2];
        const double s = g0 * g0 + g1 * g1 + g2 * g2;
        const double t = s / (r * r);
        const double wt = cvv * t * t;
        const double om = sqrt(wt + pi43 * r);
        const double ka = kvv * pow(r, 1.0 / 6.0);
        v[0] = px[p] - cx;
        v[1] = py[p] - cy;
        v[2] = pz[p] - cz;
        v[3] = om;
        v[4] = ka;
        v[5] = r * w[p];
        v[6] = (0.5 * pi43 * r - 2.0 * wt) / om;
        // C sigma / (rho^3 omega): finite at sigma = 0
        v[7] = cvv * s / (r * r * r * om);
        v[8] = ka / 6.0;
    }
    for (int k = 0; k < 9; ++k) q[(long long)k * npad + a] = v[k];
}

// the pair inputs (x, y, z, omega, kappa, rho w) in single precision: double-float (df = 1)
// qf[2 k npad + a] = hi = (float)v, qf[(2 k + 1) npad + a] = (float)(v - hi); FP32 (df = 0)
// qf[k npad + a] = (float)v
extern "C" __global__ void vv10_split(int npad, int df, const double *__restrict__ q,
    float *__restrict__ qf)
{
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= npad) return;
    for (int k = 0; k < 6; ++k) {
        const double v = q[(long long)k * npad + a];
        const float h = (float)v;
        if (df) {
            qf[2LL * k * npad + a] = h;
            qf[(2LL * k + 1) * npad + a] = (float)(v - (double)h);
        } else {
            qf[(long long)k * npad + a] = h;
        }
    }
}

// ---- the pair sums. Each point i is one thread; its j range is split into gridDim.y segments
// of whole tiles (segment g: tiles [nt g / nseg, nt (g + 1) / nseg)), each segment's sums
// written to s[(g ns + k) npad + i] (k = F, U, W and with FORCES the position sums x, y, z;
// ns = 3 or 6) and added in segment order by vv10_post. The host chooses nseg from the number
// of points alone, so the order of every sum is fixed by the input, never by the device.
#define VV_SEGMENT                                                                    \
    const int t = threadIdx.x;                                                        \
    const long long n = npad, i = (long long)blockIdx.x * VV_TILE + t;                \
    const long long nt = n / VV_TILE, sg = blockIdx.y, nsg = gridDim.y;               \
    const long long jb = nt * sg / nsg * VV_TILE, je = nt * (sg + 1) / nsg * VV_TILE; \
    double *o = s + sg * (FORCES ? 6 : 3) * n;

// FP64
template <bool FORCES>
__device__ __forceinline__ void vv10_pairs64(int npad, const double *__restrict__ q,
    double *__restrict__ s)
{
    __shared__ double tx[VV_TILE], ty[VV_TILE], tz[VV_TILE], to[VV_TILE], tk[VV_TILE],
        tr[VV_TILE];
    VV_SEGMENT
    const double xi = q[i], yi = q[n + i], zi = q[2 * n + i], oi = q[3 * n + i],
        ki = q[4 * n + i];
    double f = 0.0, u = 0.0, w = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
    for (long long j0 = jb; j0 < je; j0 += VV_TILE) {
        const long long j = j0 + t;
        tx[t] = q[j];
        ty[t] = q[n + j];
        tz[t] = q[2 * n + j];
        to[t] = q[3 * n + j];
        tk[t] = q[4 * n + j];
        tr[t] = q[5 * n + j];
        __syncthreads();
        for (int c = 0; c < VV_TILE; ++c) {
            const double dx = xi - tx[c], dy = yi - ty[c], dz = zi - tz[c];
            const double r2 = fma(dz, dz, fma(dy, dy, dx * dx));
            const double gp = fma(r2, to[c], tk[c]);
            const double g = fma(r2, oi, ki);
            const double gt = g + gp;
            const double p = g * gp * gt;
            const double qq = tr[c] / (p * p);
            f = fma(p, qq, f);
            const double qi = gp * (g + gt) * qq;
            u += qi;
            w = fma(qi, r2, w);
            if (FORCES) {
                const double qj = g * (gp + gt) * qq;
                const double cc = fma(oi, qi, to[c] * qj);
                fx = fma(cc, dx, fx);
                fy = fma(cc, dy, fy);
                fz = fma(cc, dz, fz);
            }
        }
        __syncthreads();
    }
    o[i] = f;
    o[n + i] = u;
    o[2 * n + i] = w;
    if (FORCES) {
        o[3 * n + i] = fx;
        o[4 * n + i] = fy;
        o[5 * n + i] = fz;
    }
}

extern "C" __global__ void __launch_bounds__(VV_TILE) vv10_pairs_f64(int npad,
    const double *__restrict__ q, double *__restrict__ s)
{
    vv10_pairs64<false>(npad, q, s);
}

extern "C" __global__ void __launch_bounds__(VV_TILE) vv10_pairs_f64_forces(int npad,
    const double *__restrict__ q, double *__restrict__ s)
{
    vv10_pairs64<true>(npad, q, s);
}

// ---- double-float arithmetic: a value is h + l. Normalized values have |l| <= ulp(h) / 2;
// the _u operations leave their result unnormalized (|l| a few ulp(h)), which every operation
// here accepts as input, and cost fewer instructions. The operations are written with explicit
// roundings (__fadd_rn, __fmul_rn, fmaf) so that no contraction changes their error-free
// transformations; the reciprocal starts from the IEEE-rounded __frcp_rn (the approximate
// hardware reciprocal may differ between GPU generations, which would break card-to-card
// reproducibility).
struct vvdf {
    float h, l;
};

// s + e = a + b exactly
__device__ __forceinline__ vvdf vv_two_sum(float a, float b)
{
    const float s = __fadd_rn(a, b);
    const float v = __fsub_rn(s, a);
    vvdf r;
    r.h = s;
    r.l = __fadd_rn(__fsub_rn(a, __fsub_rn(s, v)), __fsub_rn(b, v));
    return r;
}

// s + e = a + b exactly when |a| >= |b| (or a = 0)
__device__ __forceinline__ vvdf vv_fast_two_sum(float a, float b)
{
    const float s = __fadd_rn(a, b);
    vvdf r;
    r.h = s;
    r.l = __fsub_rn(b, __fsub_rn(s, a));
    return r;
}

__device__ __forceinline__ vvdf vv_norm(vvdf a)
{
    return vv_fast_two_sum(a.h, a.l);
}

// a + b (Dekker's add, normalized): relative error ~2^-46 for operands of one sign
__device__ __forceinline__ vvdf vv_add(vvdf a, vvdf b)
{
    const vvdf s = vv_two_sum(a.h, b.h);
    return vv_fast_two_sum(s.h, __fadd_rn(s.l, __fadd_rn(a.l, b.l)));
}

__device__ __forceinline__ vvdf vv_add_u(vvdf a, vvdf b)
{
    const vvdf s = vv_two_sum(a.h, b.h);
    vvdf r;
    r.h = s.h;
    r.l = __fadd_rn(s.l, __fadd_rn(a.l, b.l));
    return r;
}

__device__ __forceinline__ vvdf vv_sub_u(vvdf a, vvdf b)
{
    const vvdf s = vv_two_sum(a.h, -b.h);
    vvdf r;
    r.h = s.h;
    r.l = __fadd_rn(s.l, __fsub_rn(a.l, b.l));
    return r;
}

__device__ __forceinline__ vvdf vv_mul_u(vvdf a, vvdf b)
{
    const float p = __fmul_rn(a.h, b.h);
    vvdf r;
    r.h = p;
    r.l = fmaf(a.h, b.l, fmaf(a.l, b.h, fmaf(a.h, b.h, -p)));
    return r;
}

__device__ __forceinline__ vvdf vv_sqr_u(vvdf a)
{
    const float p = __fmul_rn(a.h, a.h);
    vvdf r;
    r.h = p;
    r.l = fmaf(__fadd_rn(a.h, a.h), a.l, fmaf(a.h, a.h, -p));
    return r;
}

// a b + c
__device__ __forceinline__ vvdf vv_fma_u(vvdf a, vvdf b, vvdf c)
{
    const float p = __fmul_rn(a.h, b.h);
    const float e = fmaf(a.h, b.l, fmaf(a.l, b.h, fmaf(a.h, b.h, -p)));
    const vvdf s = vv_two_sum(p, c.h);
    vvdf r;
    r.h = s.h;
    r.l = __fadd_rn(s.l, __fadd_rn(e, c.l));
    return r;
}

// 1 / b (normalized): r0 = 1 / b.h, one Newton step r0 + r0 (1 - b r0) (~2^-46 relative)
__device__ __forceinline__ vvdf vv_rcp(vvdf b)
{
    const float r0 = __frcp_rn(b.h);
    const float e = fmaf(-b.l, r0, fmaf(-b.h, r0, 1.0f));
    return vv_fast_two_sum(r0, __fmul_rn(r0, e));
}

__device__ __forceinline__ double vv_f64(vvdf a)
{
    return (double)a.h + (double)a.l;
}

// double-float pair sums (qf in double-float, vv10_split), each tile summed in double-float,
// the tiles in FP64
template <bool FORCES>
__device__ __forceinline__ void vv10_pairs_df(int npad, const float *__restrict__ qf,
    double *__restrict__ s)
{
    __shared__ float tq[12][VV_TILE];
    VV_SEGMENT
    vvdf xi, yi, zi, oi, ki;
    xi.h = qf[i];
    xi.l = qf[n + i];
    yi.h = qf[2 * n + i];
    yi.l = qf[3 * n + i];
    zi.h = qf[4 * n + i];
    zi.l = qf[5 * n + i];
    oi.h = qf[6 * n + i];
    oi.l = qf[7 * n + i];
    ki.h = qf[8 * n + i];
    ki.l = qf[9 * n + i];
    double F = 0.0, U = 0.0, W = 0.0, FX = 0.0, FY = 0.0, FZ = 0.0;
    for (long long j0 = jb; j0 < je; j0 += VV_TILE) {
        for (int k = 0; k < 12; ++k) tq[k][t] = qf[k * n + j0 + t];
        __syncthreads();
        vvdf f = {0.0f, 0.0f}, u = {0.0f, 0.0f}, w = {0.0f, 0.0f};
        vvdf fx = {0.0f, 0.0f}, fy = {0.0f, 0.0f}, fz = {0.0f, 0.0f};
        for (int c = 0; c < VV_TILE; ++c) {
            vvdf xj, yj, zj, oj, kj, rj;
            xj.h = tq[0][c];
            xj.l = tq[1][c];
            yj.h = tq[2][c];
            yj.l = tq[3][c];
            zj.h = tq[4][c];
            zj.l = tq[5][c];
            oj.h = tq[6][c];
            oj.l = tq[7][c];
            kj.h = tq[8][c];
            kj.l = tq[9][c];
            rj.h = tq[10][c];
            rj.l = tq[11][c];
            const vvdf dx = vv_sub_u(xi, xj), dy = vv_sub_u(yi, yj), dz = vv_sub_u(zi, zj);
            const vvdf r2 = vv_norm(vv_add_u(vv_add_u(vv_sqr_u(dx), vv_sqr_u(dy)), vv_sqr_u(dz)));
            const vvdf g = vv_fma_u(r2, oi, ki);
            const vvdf gp = vv_fma_u(r2, oj, kj);
            const vvdf gt = vv_add_u(g, gp);
            // 1/P, not 1/P^2: P^2 leaves the FP32 range for distant points of large molecules
            const vvdf rp = vv_rcp(vv_mul_u(vv_mul_u(g, gp), gt));
            const vvdf t1 = vv_mul_u(rj, rp);
            const vvdf qq = vv_mul_u(t1, rp);
            f = vv_add(f, t1);
            const vvdf qi = vv_mul_u(vv_mul_u(gp, vv_add_u(g, gt)), qq);
            u = vv_add(u, qi);
            w = vv_add(w, vv_mul_u(qi, r2));
            if (FORCES) {
                const vvdf qj = vv_mul_u(vv_mul_u(g, vv_add_u(gp, gt)), qq);
                const vvdf cc = vv_fma_u(oi, qi, vv_mul_u(oj, qj));
                fx = vv_add(fx, vv_mul_u(cc, dx));
                fy = vv_add(fy, vv_mul_u(cc, dy));
                fz = vv_add(fz, vv_mul_u(cc, dz));
            }
        }
        F += vv_f64(f);
        U += vv_f64(u);
        W += vv_f64(w);
        if (FORCES) {
            FX += vv_f64(fx);
            FY += vv_f64(fy);
            FZ += vv_f64(fz);
        }
        __syncthreads();
    }
    o[i] = F;
    o[n + i] = U;
    o[2 * n + i] = W;
    if (FORCES) {
        o[3 * n + i] = FX;
        o[4 * n + i] = FY;
        o[5 * n + i] = FZ;
    }
}

extern "C" __global__ void __launch_bounds__(VV_TILE) vv10_pairs_df64(int npad,
    const float *__restrict__ qf, double *__restrict__ s)
{
    vv10_pairs_df<false>(npad, qf, s);
}

extern "C" __global__ void __launch_bounds__(VV_TILE) vv10_pairs_df64_forces(int npad,
    const float *__restrict__ qf, double *__restrict__ s)
{
    vv10_pairs_df<true>(npad, qf, s);
}

// FP32 pair sums (qf in FP32), each tile summed in FP32, the tiles in FP64
template <bool FORCES>
__device__ __forceinline__ void vv10_pairs32(int npad, const float *__restrict__ qf,
    double *__restrict__ s)
{
    __shared__ float tq[6][VV_TILE];
    VV_SEGMENT
    const float xi = qf[i], yi = qf[n + i], zi = qf[2 * n + i], oi = qf[3 * n + i],
        ki = qf[4 * n + i];
    double F = 0.0, U = 0.0, W = 0.0, FX = 0.0, FY = 0.0, FZ = 0.0;
    for (long long j0 = jb; j0 < je; j0 += VV_TILE) {
        for (int k = 0; k < 6; ++k) tq[k][t] = qf[k * n + j0 + t];
        __syncthreads();
        float f = 0.0f, u = 0.0f, w = 0.0f, fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (int c = 0; c < VV_TILE; ++c) {
            const float dx = xi - tq[0][c], dy = yi - tq[1][c], dz = zi - tq[2][c];
            const float r2 = fmaf(dz, dz, fmaf(dy, dy, dx * dx));
            const float oj = tq[3][c];
            const float gp = fmaf(r2, oj, tq[4][c]);
            const float g = fmaf(r2, oi, ki);
            const float gt = g + gp;
            const float rp = __frcp_rn(g * gp * gt);
            const float t1 = tq[5][c] * rp;
            const float qq = t1 * rp;
            f += t1;
            const float qi = gp * (g + gt) * qq;
            u += qi;
            w = fmaf(qi, r2, w);
            if (FORCES) {
                const float qj = g * (gp + gt) * qq;
                const float cc = fmaf(oi, qi, oj * qj);
                fx = fmaf(cc, dx, fx);
                fy = fmaf(cc, dy, fy);
                fz = fmaf(cc, dz, fz);
            }
        }
        F += (double)f;
        U += (double)u;
        W += (double)w;
        if (FORCES) {
            FX += (double)fx;
            FY += (double)fy;
            FZ += (double)fz;
        }
        __syncthreads();
    }
    o[i] = F;
    o[n + i] = U;
    o[2 * n + i] = W;
    if (FORCES) {
        o[3 * n + i] = FX;
        o[4 * n + i] = FY;
        o[5 * n + i] = FZ;
    }
}

extern "C" __global__ void __launch_bounds__(VV_TILE) vv10_pairs_f32(int npad,
    const float *__restrict__ qf, double *__restrict__ s)
{
    vv10_pairs32<false>(npad, qf, s);
}

extern "C" __global__ void __launch_bounds__(VV_TILE) vv10_pairs_f32_forces(int npad,
    const float *__restrict__ qf, double *__restrict__ s)
{
    vv10_pairs32<true>(npad, qf, s);
}

// per active point a (grid point p = idx[a]), the segments' sums added in order, then with
// f = -3/2 F (the CPU's evaluate): zk[p] = sc (beta + zf f) (zf 1/2: E = sum w rho zk; zf 1:
// rho zk = dE/dw), vrho[p] = sc (beta + f + 3/2 (U rho dkappa/drho + W rho domega/drho)),
// vsigma[p] = sc 3/2 W rho domega/dsigma; with forces, force[3 p + k] = 3 sc rho w S_k
extern "C" __global__ void vv10_post(int n, int npad, int nseg, double beta, double sc,
    double zf, int forces, const int *__restrict__ idx, const double *__restrict__ q,
    const double *__restrict__ s, double *__restrict__ zk, double *__restrict__ vrho,
    double *__restrict__ vsigma, double *__restrict__ force)
{
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n) return;
    const long long p = idx[a], m = npad, ns = forces ? 6 : 3;
    double S[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int g = 0; g < nseg; ++g)
        for (int k = 0; k < ns; ++k) S[k] += s[(g * ns + k) * m + a];
    const double f = -1.5 * S[0], u = S[1], w = S[2];
    zk[p] = sc * (beta + zf * f);
    vrho[p] = sc * (beta + f + 1.5 * (u * q[8 * m + a] + w * q[6 * m + a]));
    vsigma[p] = sc * 1.5 * w * q[7 * m + a];
    if (forces) {
        const double c = 3.0 * sc * q[5 * m + a];
        force[3 * p] = c * S[3];
        force[3 * p + 1] = c * S[4];
        force[3 * p + 2] = c * S[5];
    }
}
