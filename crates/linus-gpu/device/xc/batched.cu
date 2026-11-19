
// linus-gpu per-batch XC kernels (src/xc/batched.rs). Batch descriptor j: bd[9 j + field],
// fields p0, np, nf, q0, fl, phi, x, d (BD_* in the host code). Launches cover the batches
// j0 + blockIdx.y. Compressed values of batch j: phic[phi + (l nc + c) np + g] for kept
// function l, component c, point g; point x function arrays X/W: [x + l np + g];
// function x function arrays D/V: [d + j' nf + i].

#define XB_P0 0
#define XB_NP 1
#define XB_NF 2
#define XB_Q0 3
#define XB_FL 4
#define XB_PHI 5
#define XB_X 6
#define XB_D 7

// flags[j fstride + f] = 1 when max over batch j's points and the nc components of
// |phi_B(f)| exceeds thresh (the CPU collocator's value test), for the block's functions
// f < nfB; one warp per function, lanes over the points; blockIdx.y = batch.
extern "C" __global__ void xcb_flags(int nfB, int nc, int npB, double thresh,
    const long long *__restrict__ bd, int j0, long long fstride, const double *__restrict__ phi,
    unsigned char *__restrict__ flags)
{
    const int lane = threadIdx.x % LINUS_WARP_SIZE;
    const long long f = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / LINUS_WARP_SIZE;
    if (f >= nfB) return;
    const int j = j0 + blockIdx.y;
    const long long *d = bd + 9LL * j;
    const int np = (int)d[XB_NP], q0 = (int)d[XB_Q0];
    double m = 0.0;
    for (int c = 0; c < nc; ++c) {
        const double *p = phi + (f * nc + c) * (long long)npB + q0;
        for (int g = lane; g < np; g += LINUS_WARP_SIZE) m = fmax(m, fabs(p[g]));
    }
    for (int o = LINUS_WARP_SIZE / 2; o > 0; o >>= 1) m = fmax(m, linus_shfl_down(m, o));
    if (lane == 0) flags[j * fstride + f] = m > thresh ? 1 : 0;
}

// the kept functions of every batch of a block into the compressed arrays (and, with
// phic32, their single-precision copies)
extern "C" __global__ void xcb_compress(int nc, int npB, const long long *__restrict__ bd,
    int j0, const int *__restrict__ lf, const double *__restrict__ phiB,
    double *__restrict__ phic, float *__restrict__ phic32)
{
    const int j = j0 + blockIdx.y;
    const long long *d = bd + 9LL * j;
    const long long np = d[XB_NP], nf = d[XB_NF], q0 = d[XB_Q0], fl = d[XB_FL], po = d[XB_PHI];
    const long long total = nf * nc * np;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total;
         e += (long long)gridDim.x * blockDim.x) {
        const long long g = e % np, r = e / np;
        const long long l = r / nc, c = r - l * nc;
        const double v = phiB[((long long)lf[fl + l] * nc + c) * npB + q0 + g];
        if (phic) phic[po + e] = v;
        if (phic32) phic32[po + e] = (float)v;
    }
}

// D sub-blocks of the batches: dsub[d + jj nf + i] = D[gf[jj] nbf + gf[i]] (and single copies)
extern "C" __global__ void xcb_gather_d(long long nbf, const long long *__restrict__ bd, int j0,
    const int *__restrict__ gf, const double *__restrict__ dm, double *__restrict__ dsub,
    float *__restrict__ dsub32)
{
    const int j = j0 + blockIdx.y;
    const long long *d = bd + 9LL * j;
    const long long nf = d[XB_NF], fl = d[XB_FL], dd = d[XB_D];
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < nf * nf;
         e += (long long)gridDim.x * blockDim.x) {
        const long long i = e % nf, jj = e / nf;
        const double v = dm[(long long)gf[fl + jj] * nbf + gf[fl + i]];
        if (dsub) dsub[dd + e] = v;
        if (dsub32) dsub32[dd + e] = (float)v;
    }
}

// densities of spin s at batch j's points from X = phi D (and Y = grad phi D for tau):
// rho = sum_l phi_l X_l, grad_k = 2 sum_l phi_{l,k} X_l, tau = 1/2 sum_l sum_k phi_{l,k} Y_{k,l}
// (times scale), in a fixed order over l; X, Y and phi in double or (x32, y32, phi32) single
// precision, summed in double
extern "C" __global__ void xcb_densities(int nc, int nsp, int s, double scale, int gga, int mgga,
    const long long *__restrict__ bd, int j0, const double *__restrict__ phic,
    const double *__restrict__ xc, const double *__restrict__ yc,
    const float *__restrict__ phic32, const float *__restrict__ xc32,
    const float *__restrict__ yc32, double *__restrict__ rho, double *__restrict__ grad,
    double *__restrict__ tau)
{
    const int j = j0 + blockIdx.y;
    const long long *d = bd + 9LL * j;
    const long long np = d[XB_NP];
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np) return;
    const long long nf = d[XB_NF], po = d[XB_PHI], xo = d[XB_X], P = d[XB_P0] + g;
    double r = 0.0, gx = 0.0, gy = 0.0, gz = 0.0, t = 0.0;
    for (long long l = 0; l < nf; ++l) {
        const long long pi = po + l * nc * np + g, xi = xo + l * np + g;
        double p0, p1 = 0.0, p2 = 0.0, p3 = 0.0, xv;
        if (phic32) {
            p0 = phic32[pi];
            if (gga || mgga) {
                p1 = phic32[pi + np];
                p2 = phic32[pi + 2 * np];
                p3 = phic32[pi + 3 * np];
            }
        } else {
            p0 = phic[pi];
            if (gga || mgga) {
                p1 = phic[pi + np];
                p2 = phic[pi + 2 * np];
                p3 = phic[pi + 3 * np];
            }
        }
        xv = xc32 ? (double)xc32[xi] : xc[xi];
        r += p0 * xv;
        if (gga) {
            gx += p1 * xv;
            gy += p2 * xv;
            gz += p3 * xv;
        }
        if (mgga) {
            const long long yi = 3 * xo + l * 3 * np + g;
            const double y0 = yc32 ? (double)yc32[yi] : yc[yi];
            const double y1 = yc32 ? (double)yc32[yi + np] : yc[yi + np];
            const double y2 = yc32 ? (double)yc32[yi + 2 * np] : yc[yi + 2 * np];
            t += p1 * y0 + p2 * y1 + p3 * y2;
        }
    }
    rho[P * nsp + s] = scale * r;
    if (gga) {
        grad[(P * nsp + s) * 3] = 2.0 * scale * gx;
        grad[(P * nsp + s) * 3 + 1] = 2.0 * scale * gy;
        grad[(P * nsp + s) * 3 + 2] = 2.0 * scale * gz;
    }
    if (mgga) tau[P * nsp + s] = 0.5 * scale * t;
}

// W of spin s for batch j: W[l np + g] = vr phi_l + sum_k vg_k phi_{l,k} (vr = 1/2 w vrho_s,
// vg_k = w (2 vsigma grad_k) or the spin form), in double or single precision
extern "C" __global__ void xcb_weights(int nc, int nsp, int s, int gga,
    const long long *__restrict__ bd, int j0, const double *__restrict__ w,
    const double *__restrict__ vrho, const double *__restrict__ vsigma,
    const double *__restrict__ grad, const double *__restrict__ phic, double *__restrict__ wc,
    float *__restrict__ wc32)
{
    const int j = j0 + blockIdx.y;
    const long long *d = bd + 9LL * j;
    const long long np = d[XB_NP], nf = d[XB_NF], po = d[XB_PHI], xo = d[XB_X], p0 = d[XB_P0];
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < np * nf;
         e += (long long)gridDim.x * blockDim.x) {
        const long long g = e % np, l = e / np, P = p0 + g;
        const double wg = w[P];
        const double vr = 0.5 * wg * vrho[P * nsp + s];
        const double *ph = phic + po + l * nc * np + g;
        double v;
        if (gga) {
            double vg[3];
            for (int k = 0; k < 3; ++k) {
                if (nsp == 1) {
                    vg[k] = wg * (2.0 * vsigma[P] * grad[P * 3 + k]);
                } else {
                    vg[k] = wg * (2.0 * vsigma[3 * P + 2 * s] * grad[(P * 2 + s) * 3 + k]
                        + vsigma[3 * P + 1] * grad[(P * 2 + 1 - s) * 3 + k]);
                }
            }
            v = vr * ph[0] + vg[0] * ph[np] + vg[1] * ph[2 * np] + vg[2] * ph[3 * np];
        } else {
            v = vr * ph[0];
        }
        if (wc) wc[xo + e] = v;
        if (wc32) wc32[xo + e] = (float)v;
    }
}

// Wt of spin s for batch j: Wt[(l 3 + k) np + g] = 1/4 w vtau_s phi_{l,k} (meta-GGA)
extern "C" __global__ void xcb_weights_tau(int nc, int nsp, int s,
    const long long *__restrict__ bd, int j0, const double *__restrict__ w,
    const double *__restrict__ vtau, const double *__restrict__ phic, double *__restrict__ wt,
    float *__restrict__ wt32)
{
    const int j = j0 + blockIdx.y;
    const long long *d = bd + 9LL * j;
    const long long np = d[XB_NP], nf = d[XB_NF], po = d[XB_PHI], xo = d[XB_X], p0 = d[XB_P0];
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < np * nf;
         e += (long long)gridDim.x * blockDim.x) {
        const long long g = e % np, l = e / np, P = p0 + g;
        const double vt = 0.25 * w[P] * vtau[P * nsp + s];
        for (int k = 0; k < 3; ++k) {
            const double v = vt * phic[po + (l * nc + 1 + k) * np + g];
            const long long o = 3 * xo + (l * 3 + k) * np + g;
            if (wt) wt[o] = v;
            if (wt32) wt32[o] = (float)v;
        }
    }
}

// the batch potentials into the fixed-point accumulator of the lower triangle:
// acc[gf[jj] nbf + gf[i]] += V[jj, i] + V[i, jj] for i >= jj (the global numbers ascend, so
// the element is in the lower triangle); V in double or single precision
extern "C" __global__ void xcb_vscatter(long long nbf, const long long *__restrict__ bd, int j0,
    const int *__restrict__ gf, const double *__restrict__ vc, const float *__restrict__ vc32,
    unsigned long long *acc, long long nacc)
{
    const int j = j0 + blockIdx.y;
    const long long *d = bd + 9LL * j;
    const long long nf = d[XB_NF], fl = d[XB_FL], dd = d[XB_D];
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < nf * nf;
         e += (long long)gridDim.x * blockDim.x) {
        const long long i = e % nf, jj = e / nf;
        if (i < jj) continue;
        const double v = vc32 ? (double)vc32[dd + jj * nf + i] + (double)vc32[dd + i * nf + jj]
                              : vc[dd + jj * nf + i] + vc[dd + i * nf + jj];
        det_add(acc, nacc, (long long)gf[fl + jj] * nbf + gf[fl + i], v);
    }
}

// v (nbf x nbf, both triangles) from the lower-triangle accumulator (NaN if it is poisoned)
extern "C" __global__ void xcb_vfinal(long long nbf, const unsigned long long *__restrict__ acc,
    double *__restrict__ v)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nbf * nbf) return;
    const long long r = t % nbf, c = t / nbf;
    const long long lo = r >= c ? c * nbf + r : r * nbf + c;
    v[t] = det_value(acc, nbf * nbf, lo);
}
