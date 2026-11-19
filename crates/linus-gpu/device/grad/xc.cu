
// linus-gpu XC gradient (src/grad/xc.rs). Block arrays are column-major with the block's
// points as rows: collocation phi[(f ncomp + d) np + g] (d: value, x, y, z, xx, xy, xz, yy,
// yz, zz), X and B [f np + g], Y [(f 3 + k) np + g]; per-point weights vg[k np + g], vt[g].
// Grid arrays as the XC engine's: rho[P nsp + s], sigma[P] or [3 P + {aa, ab, bb}],
// tau[P nsp + s], grad[(P nsp + s) 3 + k].

// wm = vr phi_0 + sum_k vg_k phi_k for spin s (vr = w vrho_s); vg and (meta-GGA) vt of the
// block's points
extern "C" __global__ void xg_weights(int np, long long p0, int nf, int ncomp, int nsp, int s,
    int gga, int mgga, const double *__restrict__ w, const double *__restrict__ vrho,
    const double *__restrict__ vsigma, const double *__restrict__ vtau,
    const double *__restrict__ grad, const double *__restrict__ phi, double *__restrict__ wm,
    double *__restrict__ vg, double *__restrict__ vt)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)np * nf) return;
    const long long g = t % np, f = t / np;
    const long long P = p0 + g;
    const double wg = w[P];
    const double vr = wg * vrho[P * nsp + s];
    const double *ph = phi + f * ncomp * np + g;
    if (gga) {
        double v[3];
        for (int k = 0; k < 3; ++k) {
            if (nsp == 1) {
                v[k] = wg * (2.0 * vsigma[P] * grad[P * 3 + k]);
            } else {
                v[k] = wg * (2.0 * vsigma[3 * P + 2 * s] * grad[(P * 2 + s) * 3 + k]
                    + vsigma[3 * P + 1] * grad[(P * 2 + 1 - s) * 3 + k]);
            }
        }
        wm[f * np + g] = vr * ph[0] + v[0] * ph[np] + v[1] * ph[2 * np] + v[2] * ph[3 * np];
        if (f == 0) {
            vg[g] = v[0];
            vg[np + g] = v[1];
            vg[2 * np + g] = v[2];
            if (mgga) vt[g] = wg * vtau[P * nsp + s];
        }
    } else {
        wm[f * np + g] = vr * ph[0];
    }
}

// c_i(f, g) of the module docs
__device__ __forceinline__ void xg_c(int np, int gga, int mgga, const double *ph,
    const double *x, const double *b, const double *y, const double *vg, const double *vt,
    long long f, int g, double c[3])
{
    const double bb = b[f * np + g];
    c[0] = ph[np + g] * bb;
    c[1] = ph[2 * np + g] * bb;
    c[2] = ph[3 * np + g] * bb;
    if (gga) {
        const double h0 = ph[4 * np + g], h1 = ph[5 * np + g], h2 = ph[6 * np + g];
        const double h3 = ph[7 * np + g], h4 = ph[8 * np + g], h5 = ph[9 * np + g];
        const double xx = x[f * np + g];
        const double v0 = vg[g], v1 = vg[np + g], v2 = vg[2 * np + g];
        c[0] += (h0 * v0 + h1 * v1 + h2 * v2) * xx;
        c[1] += (h1 * v0 + h3 * v1 + h4 * v2) * xx;
        c[2] += (h2 * v0 + h4 * v1 + h5 * v2) * xx;
        if (mgga) {
            const double *yf = y + f * 3 * np + g;
            const double y0 = yf[0], y1 = yf[np], y2 = yf[2 * np];
            const double ht = 0.5 * vt[g];
            c[0] += ht * (h0 * y0 + h1 * y1 + h2 * y2);
            c[1] += ht * (h1 * y0 + h3 * y1 + h4 * y2);
            c[2] += ht * (h2 * y0 + h4 * y1 + h5 * y2);
        }
    }
}

// acc[funcs[f] 3 + i] += -2 sum_g c_i(f, g): one warp per function, lanes over the points,
// a fixed shuffle tree, fixed point
extern "C" __global__ void xg_contract_funcs(int np, int nf, int ncomp, int gga, int mgga,
    const int *__restrict__ funcs, const double *__restrict__ phi, const double *__restrict__ x,
    const double *__restrict__ b, const double *__restrict__ y, const double *__restrict__ vg,
    const double *__restrict__ vt, unsigned long long *acc, long long nacc)
{
    const int lane = threadIdx.x % LINUS_WARP_SIZE;
    const long long f = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / LINUS_WARP_SIZE;
    if (f >= nf) return;
    const double *ph = phi + f * ncomp * np;
    double s[3] = {0.0, 0.0, 0.0};
    for (int g = lane; g < np; g += LINUS_WARP_SIZE) {
        double c[3];
        xg_c(np, gga, mgga, ph, x, b, y, vg, vt, f, g, c);
        s[0] += c[0];
        s[1] += c[1];
        s[2] += c[2];
    }
    for (int o = LINUS_WARP_SIZE / 2; o > 0; o >>= 1) {
        s[0] += linus_shfl_down(s[0], o);
        s[1] += linus_shfl_down(s[1], o);
        s[2] += linus_shfl_down(s[2], o);
    }
    if (lane == 0) {
        const long long m = funcs[f];
        for (int i = 0; i < 3; ++i) det_add(acc, nacc, m * 3 + i, -2.0 * s[i]);
    }
}

// tp[(p0 + g) 3 + i] += 2 sum_f c_i(f, g) (one thread per point; spins in turn)
extern "C" __global__ void xg_contract_points(int np, long long p0, int nf, int ncomp, int gga,
    int mgga, const double *__restrict__ phi, const double *__restrict__ x,
    const double *__restrict__ b, const double *__restrict__ y, const double *__restrict__ vg,
    const double *__restrict__ vt, double *__restrict__ tp)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np) return;
    double s[3] = {0.0, 0.0, 0.0};
    for (long long f = 0; f < nf; ++f) {
        double c[3];
        xg_c(np, gga, mgga, phi + f * ncomp * np, x, b, y, vg, vt, f, g, c);
        s[0] += c[0];
        s[1] += c[1];
        s[2] += c[2];
    }
    double *t = tp + (p0 + g) * 3;
    t[0] += 2.0 * s[0];
    t[1] += 2.0 * s[1];
    t[2] += 2.0 * s[2];
}

// Per-batch path (src/xc/batched.rs layout): W of spin s for batch j (vr = w vrho_s), and at
// l = 0 the point's vg (vgp[3 P + k]) and w vtau (vtp[P]); W in double or (w32) single
extern "C" __global__ void xgb_weights(int nc, int nsp, int s, int gga, int mgga,
    const long long *__restrict__ bd, int j0, const double *__restrict__ w,
    const double *__restrict__ vrho, const double *__restrict__ vsigma,
    const double *__restrict__ vtau, const double *__restrict__ grad,
    const double *__restrict__ phic, double *__restrict__ wc, float *__restrict__ wc32,
    double *__restrict__ vgp, double *__restrict__ vtp)
{
    const int j = j0 + blockIdx.y;
    const long long *d = bd + 9LL * j;
    const long long np = d[1], nf = d[2], po = d[5], xo = d[6], p0 = d[0];
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < np * nf;
         e += (long long)gridDim.x * blockDim.x) {
        const long long g = e % np, l = e / np, P = p0 + g;
        const double wg = w[P];
        const double vr = wg * vrho[P * nsp + s];
        const double *ph = phic + po + l * nc * np + g;
        double val;
        if (gga) {
            double v[3];
            for (int k = 0; k < 3; ++k) {
                if (nsp == 1) {
                    v[k] = wg * (2.0 * vsigma[P] * grad[P * 3 + k]);
                } else {
                    v[k] = wg * (2.0 * vsigma[3 * P + 2 * s] * grad[(P * 2 + s) * 3 + k]
                        + vsigma[3 * P + 1] * grad[(P * 2 + 1 - s) * 3 + k]);
                }
            }
            val = vr * ph[0] + v[0] * ph[np] + v[1] * ph[2 * np] + v[2] * ph[3 * np];
            if (l == 0) {
                vgp[3 * P] = v[0];
                vgp[3 * P + 1] = v[1];
                vgp[3 * P + 2] = v[2];
                if (mgga) vtp[P] = wg * vtau[P * nsp + s];
            }
        } else {
            val = vr * ph[0];
        }
        if (wc) wc[xo + e] = val;
        if (wc32) wc32[xo + e] = (float)val;
    }
}

// c_i(l, g) of the module docs for the per-batch layout: ph = the function's compressed values
// (nc np, with second derivatives), x and b its X and B at the point, y its three Y values at
// the point, vg and vt the point's weights
__device__ __forceinline__ void xgb_c(long long np, int gga, int mgga, const double *ph,
    double x, double b, const double *y, const double *vg, double vt, double c[3])
{
    c[0] = ph[np] * b;
    c[1] = ph[2 * np] * b;
    c[2] = ph[3 * np] * b;
    if (gga) {
        const double h0 = ph[4 * np], h1 = ph[5 * np], h2 = ph[6 * np];
        const double h3 = ph[7 * np], h4 = ph[8 * np], h5 = ph[9 * np];
        const double v0 = vg[0], v1 = vg[1], v2 = vg[2];
        c[0] += (h0 * v0 + h1 * v1 + h2 * v2) * x;
        c[1] += (h1 * v0 + h3 * v1 + h4 * v2) * x;
        c[2] += (h2 * v0 + h4 * v1 + h5 * v2) * x;
        if (mgga) {
            const double y0 = y[0], y1 = y[1], y2 = y[2];
            const double ht = 0.5 * vt;
            c[0] += ht * (h0 * y0 + h1 * y1 + h2 * y2);
            c[1] += ht * (h1 * y0 + h3 * y1 + h4 * y2);
            c[2] += ht * (h2 * y0 + h4 * y1 + h5 * y2);
        }
    }
}

// the arrays of point g of function l: X, B (double or single), Y
#define XGB_LOAD(l, g)                                                                        \
    const double *ph = phic + po + (l) * nc * np + (g);                                       \
    const long long xi = xo + (l) * np + (g);                                                 \
    const double xx = xs32 ? (double)xs32[xi] : xs[xi];                                       \
    const double bb = bw32 ? (double)bw32[xi] : bw[xi];                                       \
    double yv[3] = {0.0, 0.0, 0.0};                                                           \
    if (mgga) {                                                                               \
        const long long yi = 3 * xo + (l) * 3 * np + (g);                                     \
        for (int k = 0; k < 3; ++k) yv[k] = yw32 ? (double)yw32[yi + k * np] : yw[yi + k * np]; \
    }

// acc[gf[l] 3 + i] += -2 sum_g c_i(l, g): one warp per function of batch j0 + blockIdx.y,
// lanes over the points, a fixed shuffle tree, fixed point
extern "C" __global__ void xgb_contract_funcs(int nc, int gga, int mgga,
    const long long *__restrict__ bd, int j0, const int *__restrict__ gf,
    const double *__restrict__ phic, const double *__restrict__ xs, const double *__restrict__ bw,
    const double *__restrict__ yw, const float *__restrict__ xs32,
    const float *__restrict__ bw32, const float *__restrict__ yw32,
    const double *__restrict__ vgp, const double *__restrict__ vtp, unsigned long long *acc,
    long long nacc)
{
    const int j = j0 + blockIdx.y;
    const long long *d = bd + 9LL * j;
    const long long np = d[1], nf = d[2], fl = d[4], po = d[5], xo = d[6], p0 = d[0];
    const int lane = threadIdx.x % LINUS_WARP_SIZE;
    const long long l = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / LINUS_WARP_SIZE;
    if (l >= nf) return;
    double s[3] = {0.0, 0.0, 0.0};
    for (long long g = lane; g < np; g += LINUS_WARP_SIZE) {
        XGB_LOAD(l, g)
        double c[3];
        xgb_c(np, gga, mgga, ph, xx, bb, yv, vgp + 3 * (p0 + g), mgga ? vtp[p0 + g] : 0.0, c);
        s[0] += c[0];
        s[1] += c[1];
        s[2] += c[2];
    }
    for (int o = LINUS_WARP_SIZE / 2; o > 0; o >>= 1) {
        s[0] += linus_shfl_down(s[0], o);
        s[1] += linus_shfl_down(s[1], o);
        s[2] += linus_shfl_down(s[2], o);
    }
    if (lane == 0) {
        const long long m = gf[fl + l];
        for (int i = 0; i < 3; ++i) det_add(acc, nacc, m * 3 + i, -2.0 * s[i]);
    }
}

// tp[P 3 + i] += 2 sum_l c_i(l, g): one thread per point of batch j0 + blockIdx.y
extern "C" __global__ void xgb_contract_points(int nc, int gga, int mgga,
    const long long *__restrict__ bd, int j0, const double *__restrict__ phic,
    const double *__restrict__ xs, const double *__restrict__ bw, const double *__restrict__ yw,
    const float *__restrict__ xs32, const float *__restrict__ bw32,
    const float *__restrict__ yw32, const double *__restrict__ vgp,
    const double *__restrict__ vtp, double *__restrict__ tp)
{
    const int j = j0 + blockIdx.y;
    const long long *d = bd + 9LL * j;
    const long long np = d[1], nf = d[2], po = d[5], xo = d[6], p0 = d[0];
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np) return;
    double s[3] = {0.0, 0.0, 0.0};
    const double vt = mgga ? vtp[p0 + g] : 0.0;
    for (long long l = 0; l < nf; ++l) {
        XGB_LOAD(l, g)
        double c[3];
        xgb_c(np, gga, mgga, ph, xx, bb, yv, vgp + 3 * (p0 + g), vt, c);
        s[0] += c[0];
        s[1] += c[1];
        s[2] += c[2];
    }
    double *t = tp + (p0 + g) * 3;
    t[0] += 2.0 * s[0];
    t[1] += 2.0 * s[1];
    t[2] += 2.0 * s[2];
}

// eps[P] = rho_total[P] zk[P]
extern "C" __global__ void xg_energy_density(long long npts, int nsp,
    const double *__restrict__ rho, const double *__restrict__ zk, double *__restrict__ eps)
{
    const long long P = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (P >= npts) return;
    const double rt = nsp == 1 ? rho[P] : rho[2 * P] + rho[2 * P + 1];
    eps[P] = rt * zk[P];
}

// Becke weight derivatives sum_k f_k dw_k/dR (points moving with their owners) plus, with
// tp, the point-motion term tp_k for the owner. One block per chunk of points, nslot points
// at a time: thread t works on slot t / nat, atom t % nat (nslot > 1), or with one slot on
// atoms t, t + T, ... Per point the cell functions P_b = prod_c s(nu_bc), Z, the coefficients
// c_b P_b, then per atom the pair terms (each pair evaluated by both of its atoms), the u
// terms and the owner's minus their sum. Per slot and atom, sums over the slot's points in
// shared memory, then fixed point. With `screen`, the pair terms of two atoms that are both
// negligible at the point are left out (linus-grid's criterion, per point: atom b is active
// when |c_b P_b / min_c s_bc| > 1e-18 |f vol| / nat).
extern "C" __global__ void xg_becke(long long npts, int nat, int nslot, int chunk,
    const double *__restrict__ px, const double *__restrict__ py, const double *__restrict__ pz,
    const int *__restrict__ owner, const double *__restrict__ vol, const double *__restrict__ f,
    int with_tp, const double *__restrict__ tp, const double *__restrict__ rxyz,
    const double *__restrict__ inv_r, const double *__restrict__ aij, unsigned long long *acc,
    long long nacc, int screen)
{
    extern __shared__ double sh[];
    const int T = blockDim.x, tid = threadIdx.x;
    int slot, b0, bstep;
    if (nslot > 1) {
        slot = tid / nat;
        b0 = tid - slot * nat;
        bstep = nat;
        if (slot >= nslot) slot = -1;
    } else {
        slot = 0;
        b0 = tid;
        bstep = T;
    }
    const int per = 12 * nat + 2;
    double *d = sh + (slot < 0 ? 0 : slot) * per;
    double *u = d + nat;
    double *pb = u + 3 * nat;
    double *v = pb + nat;
    double *ga = v + 3 * nat;
    double *sm = ga + 3 * nat;
    double *cf = sm + nat;
    if (slot >= 0)
        for (int b = b0; b < nat; b += bstep) ga[3 * b] = ga[3 * b + 1] = ga[3 * b + 2] = 0.0;
    const long long k0 = (long long)blockIdx.x * chunk;
    const long long k1 = k0 + chunk < npts ? k0 + chunk : npts;
    __syncthreads();
    for (long long kb = k0; kb < k1; kb += nslot) {
        const long long k = kb + (slot < 0 ? 0 : slot);
        const int act = slot >= 0 && k < k1;
        double fk = 0.0;
        int own = 0;
        if (act) {
            fk = f[k];
            own = owner[k];
        }
        const int w = act && fk != 0.0;
        if (w) {
            const double qx = px[k], qy = py[k], qz = pz[k];
            for (int b = b0; b < nat; b += bstep) {
                const double dx = rxyz[3 * b] - qx, dy = rxyz[3 * b + 1] - qy,
                             dz = rxyz[3 * b + 2] - qz;
                const double dd = sqrt(dx * dx + dy * dy + dz * dz);
                const double inv = dd > 0.0 ? 1.0 / dd : 0.0;
                d[b] = dd;
                u[3 * b] = dx * inv;
                u[3 * b + 1] = dy * inv;
                u[3 * b + 2] = dz * inv;
            }
        }
        __syncthreads();
        if (w) {
            for (int b = b0; b < nat; b += bstep) {
                double p = 1.0, smin = 1.0;
                for (int c = 0; c < nat; ++c) {
                    if (c == b) continue;
                    const int i = b > c ? b : c, j = b > c ? c : b;
                    double g = (d[i] - d[j]) * inv_r[i * nat + j];
                    g += aij[i * nat + j] * (1.0 - g * g);
                    double t = g;
                    t = (3.0 - t * t) * t * 0.5;
                    t = (3.0 - t * t) * t * 0.5;
                    t = ((3.0 - t * t) * t * 0.5) * 0.5;
                    const double fac = b > c ? 0.5 - t : 0.5 + t;
                    p *= fac;
                    smin = fmin(smin, fac);
                }
                pb[b] = p;
                sm[b] = smin;
            }
        }
        __syncthreads();
        if (w && b0 == 0) {
            double z = 0.0;
            for (int b = 0; b < nat; ++b) z += pb[b];
            const double pa = pb[own];
            if (pa == 0.0 || z == 0.0) {
                cf[0] = 0.0;
                cf[1] = 0.0;
            } else {
                const double iz = 1.0 / z;
                cf[0] = -fk * vol[k] * pa * iz * iz;
                cf[1] = fk * vol[k] * iz;
            }
        }
        __syncthreads();
        if (w)
            for (int b = b0; b < nat; b += bstep) {
                pb[b] *= cf[0] + (b == own ? cf[1] : 0.0);
                // sm becomes the atom's activity at this point (1 or 0)
                sm[b] = !screen || (pb[b] != 0.0 && sm[b] > 0.0
                    && fabs(pb[b] / sm[b]) > 1e-18 * fabs(fk * vol[k]) / nat) ? 1.0 : 0.0;
            }
        __syncthreads();
        if (w) {
            for (int b = b0; b < nat; b += bstep) {
                double a = 0.0, e0 = 0.0, e1 = 0.0, e2 = 0.0;
                const int act_b = sm[b] != 0.0;
                for (int c = 0; c < nat; ++c) {
                    if (c == b || (!act_b && sm[c] == 0.0)) continue;
                    const int i = b > c ? b : c, j = b > c ? c : b;
                    const double fac = inv_r[i * nat + j], aa = aij[i * nat + j];
                    const double mu = (d[i] - d[j]) * fac;
                    const double nu = mu + aa * (1.0 - mu * mu);
                    const double dnu = 1.0 - 2.0 * aa * mu;
                    const double s0 = nu;
                    const double s1 = (3.0 - s0 * s0) * s0 * 0.5;
                    const double s2 = (3.0 - s1 * s1) * s1 * 0.5;
                    const double s3 = ((3.0 - s2 * s2) * s2 * 0.5) * 0.5;
                    const double ds = -0.5 * (1.5 * (1.0 - s2 * s2)) * (1.5 * (1.0 - s1 * s1))
                        * (1.5 * (1.0 - s0 * s0)) * dnu;
                    const double sij = 0.5 - s3, sji = 0.5 + s3;
                    const double qi = pb[i] != 0.0 ? pb[i] * ds / sij : 0.0;
                    const double qj = pb[j] != 0.0 ? pb[j] * ds / sji : 0.0;
                    const double q = qi - qj;
                    const double te = -q * (d[i] - d[j]) * fac * fac;
                    const double sg = b == i ? 1.0 : -1.0;
                    a += sg * q * fac;
                    e0 += sg * te * (rxyz[3 * i] - rxyz[3 * j]) * fac;
                    e1 += sg * te * (rxyz[3 * i + 1] - rxyz[3 * j + 1]) * fac;
                    e2 += sg * te * (rxyz[3 * i + 2] - rxyz[3 * j + 2]) * fac;
                }
                v[3 * b] = a * u[3 * b];
                v[3 * b + 1] = a * u[3 * b + 1];
                v[3 * b + 2] = a * u[3 * b + 2];
                ga[3 * b] += v[3 * b] + e0;
                ga[3 * b + 1] += v[3 * b + 1] + e1;
                ga[3 * b + 2] += v[3 * b + 2] + e2;
            }
        }
        __syncthreads();
        // the owner's thread: minus the u terms of all atoms, and the point motion
        if (act && (own - b0) % bstep == 0 && own >= b0) {
            if (w) {
                double s0 = 0.0, s1 = 0.0, s2 = 0.0;
                for (int b = 0; b < nat; ++b) {
                    s0 += v[3 * b];
                    s1 += v[3 * b + 1];
                    s2 += v[3 * b + 2];
                }
                ga[3 * own] -= s0;
                ga[3 * own + 1] -= s1;
                ga[3 * own + 2] -= s2;
            }
            if (with_tp) {
                ga[3 * own] += tp[3 * k];
                ga[3 * own + 1] += tp[3 * k + 1];
                ga[3 * own + 2] += tp[3 * k + 2];
            }
        }
        __syncthreads();
    }
    if (slot >= 0)
        for (int b = b0; b < nat; b += bstep)
            for (int x = 0; x < 3; ++x) det_add(acc, nacc, 3 * b + x, ga[3 * b + x]);
}
// The weight derivatives of xg_becke with every atom pair evaluated once per point (xg_becke
// evaluates each pair from both of its atoms): per slot, tps = blockDim / nslot threads; the
// pairs' switching values t_ij (i > j, pair index i (i - 1) / 2 + j) into shared memory, the
// cell functions as products of them, then per pair the derivative terms q_ij fac_ij and
// te_ij fac_ij (both atoms negligible: none), and per atom their sums in the order of the
// partner atoms. Shared per slot: 12 nat + 2 + 2 npair doubles. pidx: the pairs (i, j) in
// order.
extern "C" __global__ void xg_becke_pairs(long long npts, int nat, int nslot, int chunk,
    const double *__restrict__ px, const double *__restrict__ py, const double *__restrict__ pz,
    const int *__restrict__ owner, const double *__restrict__ vol, const double *__restrict__ f,
    int with_tp, const double *__restrict__ tp, const double *__restrict__ rxyz,
    const double *__restrict__ inv_r, const double *__restrict__ aij, unsigned long long *acc,
    long long nacc, int screen, const int *__restrict__ pidx)
{
    extern __shared__ double sh[];
    const int tps = blockDim.x / nslot;
    const int slot = threadIdx.x / tps, r = threadIdx.x - slot * tps;
    const int npair = nat * (nat - 1) / 2;
    const int per = 12 * nat + 2 + 2 * npair;
    const int live = slot < nslot;
    double *d = sh + (live ? slot : 0) * per;
    double *u = d + nat;
    double *pb = u + 3 * nat;
    double *v = pb + nat;
    double *ga = v + 3 * nat;
    double *sm = ga + 3 * nat;
    double *cf = sm + nat;
    double *tq = cf + 2;
    double *te = tq + npair;
    if (live)
        for (int b = r; b < nat; b += tps) ga[3 * b] = ga[3 * b + 1] = ga[3 * b + 2] = 0.0;
    const long long k0 = (long long)blockIdx.x * chunk;
    const long long k1 = k0 + chunk < npts ? k0 + chunk : npts;
    __syncthreads();
    for (long long kb = k0; kb < k1; kb += nslot) {
        const long long k = kb + (live ? slot : 0);
        const int act = live && k < k1;
        double fk = 0.0;
        int own = 0;
        if (act) {
            fk = f[k];
            own = owner[k];
        }
        const int w = act && fk != 0.0;
        if (w) {
            const double qx = px[k], qy = py[k], qz = pz[k];
            for (int b = r; b < nat; b += tps) {
                const double dx = rxyz[3 * b] - qx, dy = rxyz[3 * b + 1] - qy,
                             dz = rxyz[3 * b + 2] - qz;
                const double dd = sqrt(dx * dx + dy * dy + dz * dz);
                const double inv = dd > 0.0 ? 1.0 / dd : 0.0;
                d[b] = dd;
                u[3 * b] = dx * inv;
                u[3 * b + 1] = dy * inv;
                u[3 * b + 2] = dz * inv;
            }
        }
        __syncthreads();
        // the pairs' switching values
        if (w) {
            for (int p = r; p < npair; p += tps) {
                const int i = pidx[2 * p], j = pidx[2 * p + 1];
                double g = (d[i] - d[j]) * inv_r[i * nat + j];
                g += aij[i * nat + j] * (1.0 - g * g);
                double t = g;
                t = (3.0 - t * t) * t * 0.5;
                t = (3.0 - t * t) * t * 0.5;
                t = ((3.0 - t * t) * t * 0.5) * 0.5;
                tq[p] = t;
            }
        }
        __syncthreads();
        // cell functions and their smallest factors
        if (w) {
            for (int b = r; b < nat; b += tps) {
                double pr = 1.0, smin = 1.0;
                for (int c = 0; c < nat; ++c) {
                    if (c == b) continue;
                    const int i = b > c ? b : c, j = b > c ? c : b;
                    const double t = tq[i * (i - 1) / 2 + j];
                    const double fac = b > c ? 0.5 - t : 0.5 + t;
                    pr *= fac;
                    smin = fmin(smin, fac);
                }
                pb[b] = pr;
                sm[b] = smin;
            }
        }
        __syncthreads();
        if (w && r == 0) {
            double z = 0.0;
            for (int b = 0; b < nat; ++b) z += pb[b];
            const double pa = pb[own];
            if (pa == 0.0 || z == 0.0) {
                cf[0] = 0.0;
                cf[1] = 0.0;
            } else {
                const double iz = 1.0 / z;
                cf[0] = -fk * vol[k] * pa * iz * iz;
                cf[1] = fk * vol[k] * iz;
            }
        }
        __syncthreads();
        if (w)
            for (int b = r; b < nat; b += tps) {
                pb[b] *= cf[0] + (b == own ? cf[1] : 0.0);
                sm[b] = !screen || (pb[b] != 0.0 && sm[b] > 0.0
                    && fabs(pb[b] / sm[b]) > 1e-18 * fabs(fk * vol[k]) / nat) ? 1.0 : 0.0;
            }
        __syncthreads();
        // per pair: q fac and te fac (over the switching values just used)
        if (w) {
            for (int p = r; p < npair; p += tps) {
                {
                    const int i = pidx[2 * p], j = pidx[2 * p + 1];
                    if (sm[i] == 0.0 && sm[j] == 0.0) {
                        tq[p] = 0.0;
                        te[p] = 0.0;
                        continue;
                    }
                    const double fac = inv_r[i * nat + j], aa = aij[i * nat + j];
                    const double mu = (d[i] - d[j]) * fac;
                    const double nu = mu + aa * (1.0 - mu * mu);
                    const double dnu = 1.0 - 2.0 * aa * mu;
                    const double s0 = nu;
                    const double s1 = (3.0 - s0 * s0) * s0 * 0.5;
                    const double s2 = (3.0 - s1 * s1) * s1 * 0.5;
                    const double s3 = ((3.0 - s2 * s2) * s2 * 0.5) * 0.5;
                    const double ds = -0.5 * (1.5 * (1.0 - s2 * s2)) * (1.5 * (1.0 - s1 * s1))
                        * (1.5 * (1.0 - s0 * s0)) * dnu;
                    const double sij = 0.5 - s3, sji = 0.5 + s3;
                    const double qi = pb[i] != 0.0 ? pb[i] * ds / sij : 0.0;
                    const double qj = pb[j] != 0.0 ? pb[j] * ds / sji : 0.0;
                    const double q = qi - qj;
                    tq[p] = q * fac;
                    te[p] = -q * (d[i] - d[j]) * fac * fac * fac;
                }
            }
        }
        __syncthreads();
        if (w) {
            for (int b = r; b < nat; b += tps) {
                double a = 0.0, e0 = 0.0, e1 = 0.0, e2 = 0.0;
                for (int c = 0; c < nat; ++c) {
                    if (c == b) continue;
                    const int i = b > c ? b : c, j = b > c ? c : b;
                    const int p = i * (i - 1) / 2 + j;
                    const double sg = b == i ? 1.0 : -1.0;
                    a += sg * tq[p];
                    const double tt = sg * te[p];
                    e0 += tt * (rxyz[3 * i] - rxyz[3 * j]);
                    e1 += tt * (rxyz[3 * i + 1] - rxyz[3 * j + 1]);
                    e2 += tt * (rxyz[3 * i + 2] - rxyz[3 * j + 2]);
                }
                v[3 * b] = a * u[3 * b];
                v[3 * b + 1] = a * u[3 * b + 1];
                v[3 * b + 2] = a * u[3 * b + 2];
                ga[3 * b] += v[3 * b] + e0;
                ga[3 * b + 1] += v[3 * b + 1] + e1;
                ga[3 * b + 2] += v[3 * b + 2] + e2;
            }
        }
        __syncthreads();
        // the owner's thread: minus the u terms of all atoms, and the point motion
        if (act && own % tps == r) {
            if (w) {
                double s0 = 0.0, s1 = 0.0, s2 = 0.0;
                for (int b = 0; b < nat; ++b) {
                    s0 += v[3 * b];
                    s1 += v[3 * b + 1];
                    s2 += v[3 * b + 2];
                }
                ga[3 * own] -= s0;
                ga[3 * own + 1] -= s1;
                ga[3 * own + 2] -= s2;
            }
            if (with_tp) {
                ga[3 * own] += tp[3 * k];
                ga[3 * own + 1] += tp[3 * k + 1];
                ga[3 * own + 2] += tp[3 * k + 2];
            }
        }
        __syncthreads();
    }
    if (live)
        for (int b = r; b < nat; b += tps)
            for (int x = 0; x < 3; ++x) det_add(acc, nacc, 3 * b + x, ga[3 * b + x]);
}
