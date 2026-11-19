
// linus-gpu XC helpers (src/xc/mod.rs). Block arrays are column-major with the block's
// points as rows: collocation phi[(f * ncomp + d) * ld + g], orbitals on the grid
// psi[(i * ncomp + d) * ld + g]. Grid arrays (whole grid, point P) in libxc's layout:
// rho[P nsp + s], sigma[P] (closed shell) or sigma[3 P + {aa, ab, bb}], tau[P nsp + s];
// grad[(P nsp + s) 3 + k].

// csub[i nf + f] = c[i nbf + funcs[f]]
extern "C" __global__ void xc_gather_rows(int nf, int ntot, long long nbf,
    const int *__restrict__ funcs, const double *__restrict__ c, double *__restrict__ csub)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)nf * ntot) return;
    const long long f = t % nf, i = t / nf;
    csub[t] = c[i * nbf + funcs[f]];
}

// densities of the block's points from psi; channel 0 = columns [0, n0), channel 1 =
// [n0, ntot); occupations occ[i]; scale 2 for a closed shell's total density
extern "C" __global__ void xc_densities(int np, long long p0, int ncomp, long long ld, int nsp,
    int n0, int ntot, double scale, int gga, int mgga, const double *__restrict__ psi,
    const double *__restrict__ occ, double *__restrict__ rho, double *__restrict__ grad,
    double *__restrict__ tau)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np) return;
    const long long P = p0 + g;
    for (int s = 0; s < nsp; ++s) {
        const int i0 = s == 0 ? 0 : n0, i1 = s == 0 ? n0 : ntot;
        double r = 0.0, gx = 0.0, gy = 0.0, gz = 0.0, t = 0.0;
        for (int i = i0; i < i1; ++i) {
            const double *p = psi + (long long)i * ncomp * ld + g;
            const double n = occ[i], v = p[0];
            r += n * v * v;
            if (gga) {
                const double nv = 2.0 * n * v;
                gx += nv * p[ld];
                gy += nv * p[2 * ld];
                gz += nv * p[3 * ld];
            }
            if (mgga) t += n * (p[ld] * p[ld] + p[2 * ld] * p[2 * ld] + p[3 * ld] * p[3 * ld]);
        }
        rho[P * nsp + s] = scale * r;
        if (gga) {
            grad[(P * nsp + s) * 3] = scale * gx;
            grad[(P * nsp + s) * 3 + 1] = scale * gy;
            grad[(P * nsp + s) * 3 + 2] = scale * gz;
        }
        if (mgga) tau[P * nsp + s] = 0.5 * scale * t;
    }
}

// dsub[j nf + i] = d[funcs[j] nbf + funcs[i]]
extern "C" __global__ void xc_gather_sub(int nf, long long nbf, const int *__restrict__ funcs,
    const double *__restrict__ d, double *__restrict__ dsub)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)nf * nf) return;
    const long long i = t % nf, j = t / nf;
    dsub[t] = d[(long long)funcs[j] * nbf + funcs[i]];
}

// cs[i nbf + m] = c[i nbf + m] occ[i]
extern "C" __global__ void xc_scale_cols(long long nbf, int ntot, const double *__restrict__ c,
    const double *__restrict__ occ, double *__restrict__ cs)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nbf * ntot) return;
    cs[t] = c[t] * occ[t / nbf];
}

// densities of spin s at the block's points from X = phi_0 D_s (x[f np + g]) and, for tau,
// Y = (grad phi) D_s (y[f 3 np + k np + g]): rho = sum_f phi_0 X, grad_k = 2 sum_f phi_k X,
// tau = 1/2 sum_f sum_k phi_k Y_k (times scale)
extern "C" __global__ void xc_densities_dm(int np, long long p0, int ncomp, long long ld, int nf,
    int nsp, int s, double scale, int gga, int mgga, const double *__restrict__ phi,
    const double *__restrict__ x, const double *__restrict__ y, double *__restrict__ rho,
    double *__restrict__ grad, double *__restrict__ tau)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np) return;
    const long long P = p0 + g;
    double r = 0.0, gx = 0.0, gy = 0.0, gz = 0.0, t = 0.0;
    for (int f = 0; f < nf; ++f) {
        const double *p = phi + (long long)f * ncomp * ld + g;
        const double xv = x[(long long)f * np + g];
        r += p[0] * xv;
        if (gga) {
            gx += p[ld] * xv;
            gy += p[2 * ld] * xv;
            gz += p[3 * ld] * xv;
        }
        if (mgga) {
            const double *yf = y + (long long)f * 3 * np + g;
            t += p[ld] * yf[0] + p[2 * ld] * yf[np] + p[3 * ld] * yf[2 * np];
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

// sigma of every point from the density gradients
extern "C" __global__ void xc_sigma(long long npts, int nsp, const double *__restrict__ grad,
    double *__restrict__ sigma)
{
    const long long P = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (P >= npts) return;
    const double *a = grad + P * nsp * 3;
    if (nsp == 1) {
        sigma[P] = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
    } else {
        const double *b = a + 3;
        sigma[3 * P] = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
        sigma[3 * P + 1] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        sigma[3 * P + 2] = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
    }
}

// acc[0] += sum_P w[P] rho_total[P] zk[P] (fixed point; blockDim a multiple of the warp)
extern "C" __global__ void xc_energy(long long npts, int nsp, const double *__restrict__ w,
    const double *__restrict__ rho, const double *__restrict__ zk, unsigned long long *acc)
{
    const long long P = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if (P < npts) {
        const double rt = nsp == 1 ? rho[P] : rho[2 * P] + rho[2 * P + 1];
        e = w[P] * rt * zk[P];
    }
    det_add_block(acc, 1, 0, e);
}

// wm[f ld + g] = vr phi_0 + sum_k vg_k phi_k for spin s: vr = 1/2 w vrho_s,
// vg_k = w (2 vsigma grad_k) (closed shell) or w (2 vsigma_ss grad_s,k + vsigma_ab grad_s',k)
extern "C" __global__ void xc_weights(int np, long long p0, int nf, int ncomp, long long ld,
    int nsp, int s, int gga, const double *__restrict__ w, const double *__restrict__ vrho,
    const double *__restrict__ vsigma, const double *__restrict__ grad,
    const double *__restrict__ phi, double *__restrict__ wm)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)np * nf) return;
    const long long g = t % np, f = t / np;
    const long long P = p0 + g;
    const double wg = w[P];
    const double vr = 0.5 * wg * vrho[P * nsp + s];
    const double *ph = phi + f * ncomp * ld + g;
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
        wm[f * ld + g] = vr * ph[0] + vg[0] * ph[ld] + vg[1] * ph[2 * ld] + vg[2] * ph[3 * ld];
    } else {
        wm[f * ld + g] = vr * ph[0];
    }
}

// wt[(f 3 + k) ld + g] = 1/4 w vtau_s d_k phi_f (meta-GGA; 1/4 and the symmetrization give
// 1/2 vtau grad phi . grad phi)
extern "C" __global__ void xc_weights_tau(int np, long long p0, int nf, int ncomp,
    long long ld, int nsp, int s, const double *__restrict__ w, const double *__restrict__ vtau,
    const double *__restrict__ phi, double *__restrict__ wt)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)np * nf) return;
    const long long g = t % np, f = t / np;
    const long long P = p0 + g;
    const double vt = 0.25 * w[P] * vtau[P * nsp + s];
    for (int k = 0; k < 3; ++k) wt[(f * 3 + k) * ld + g] = vt * phi[(f * ncomp + 1 + k) * ld + g];
}

// v[funcs[j] nbf + funcs[i]] += m[j nf + i] + m[i nf + j] (one writer per element; exactly
// symmetric)
extern "C" __global__ void xc_scatter(int nf, long long nbf, const int *__restrict__ funcs,
    const double *__restrict__ m, double *__restrict__ v)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)nf * nf) return;
    const long long i = t % nf, j = t / nf;
    v[(long long)funcs[j] * nbf + funcs[i]] += m[j * nf + i] + m[i * nf + j];
}
