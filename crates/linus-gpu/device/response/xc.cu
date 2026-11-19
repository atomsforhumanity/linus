
// XCK_T: the element type of the collocation, orbitals on the grid, perturbed densities and
// products (double, or float for the single-precision module, whose entry points end in _f32);
// the kernel weights and the sums are double.
#ifndef XCK_T
#define XCK_T double
#define XCK_SFX(n) n
#endif
// linus-gpu XC kernel helpers (src/response/xc.rs). Block arrays are column-major with the
// block's points as rows: collocation F[(f nc + c) np + g], orbitals and factors on the grid
// X[(j nc + c) np + g] for column j. Grid arrays of the whole grid (point P) in libxc's layout:
// rho[P nsp + s], grad[(P nsp + s) 3 + k], tau[P nsp + s]; kernel weights wf[(c nu + e) npts + P].

// sub[j nf + f] = src[j nbf + funcs[f]], j < ncol
extern "C" __global__ void XCK_SFX(xck_gather_rows)(int nf, long long ncol, long long nbf,
    const int *__restrict__ funcs, const double *__restrict__ src, XCK_T *__restrict__ sub)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)nf * ncol) return;
    const long long f = t % nf, j = t / nf;
    sub[t] = src[j * nbf + funcs[f]];
}

// out[j nbf + funcs[f]] += v[j nf + f], j < ncol (one writer per element)
extern "C" __global__ void XCK_SFX(xck_scatter_rows)(int nf, long long ncol, long long nbf,
    const int *__restrict__ funcs, const XCK_T *__restrict__ v, double *__restrict__ out)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)nf * ncol) return;
    const long long f = t % nf, j = t / nf;
    out[j * nbf + funcs[f]] += v[t];
}

// ground-state densities of spin s from Phi = F C_s (o columns): rho = scale sum Phi^2,
// grad = 2 scale sum Phi grad Phi, tau = 1/2 scale sum |grad Phi|^2 (scale 2: a closed shell's
// total density)
extern "C" __global__ void xck_ground(int np, long long p0, int nc, int nsp, int s, int o,
    double scale, int gga, int mgga, const double *__restrict__ phi, double *__restrict__ rho,
    double *__restrict__ grad, double *__restrict__ tau)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= np) return;
    const long long ld = (long long)nc * np, P = p0 + g;
    double r = 0.0, gx = 0.0, gy = 0.0, gz = 0.0, t = 0.0;
    for (int i = 0; i < o; ++i) {
        const double *f = phi + i * ld + g;
        r += f[0] * f[0];
        if (gga) {
            gx += f[0] * f[np];
            gy += f[0] * f[2 * np];
            gz += f[0] * f[3 * np];
        }
        if (mgga) t += f[np] * f[np] + f[2 * np] * f[2 * np] + f[3 * np] * f[3 * np];
    }
    rho[P * nsp + s] = scale * r;
    if (gga) {
        grad[(P * nsp + s) * 3] = 2.0 * scale * gx;
        grad[(P * nsp + s) * 3 + 1] = 2.0 * scale * gy;
        grad[(P * nsp + s) * 3 + 2] = 2.0 * scale * gz;
    }
    if (mgga) tau[P * nsp + s] = 0.5 * scale * t;
}

#define XCK_MAXU 10

// GGA (no tau), the CPU's form: one thread per (point g, perturbation d of the group). Spin s:
// Phi_s = F C_s (4 np x o_s; only rows 0..np, the values, and the gradients are read),
// P_s = phi A_s (np x ndg o_s, values only), Q_s,d = Phi A_s,d^T (np x nf per perturbation).
// rho1 = 2 sum_i P_i Phi_i, grad rho1 = 2 sum_i P_i grad Phi_i + 2 sum_m grad phi_m Q_m (times
// scale); t = wF u1; P is overwritten by Z = t_rho Phi + t_k d_k Phi and Q by
// G = t_k d_k phi.
extern "C" __global__ void XCK_SFX(xck_pert_gga)(int np, long long p0, long long npts, int nsp,
    double scale, int ndg, int nf, const XCK_T *__restrict__ phi, int o0,
    const XCK_T *__restrict__ psi0, XCK_T *__restrict__ pz0, XCK_T *__restrict__ qg0, int o1,
    const XCK_T *__restrict__ psi1, XCK_T *__restrict__ pz1, XCK_T *__restrict__ qg1,
    const double *__restrict__ wf)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)np * ndg) return;
    const int g = (int)(t % np), d = (int)(t / np);
    const long long ld = 4LL * np, P = p0 + g;
    const int nu = 4 * nsp;
    double u[8], tt[8];
    for (int s = 0; s < nsp; ++s) {
        const int o = s == 0 ? o0 : o1;
        const XCK_T *psi = s == 0 ? psi0 : psi1;
        const XCK_T *pz = s == 0 ? pz0 : pz1;
        const XCK_T *qg = s == 0 ? qg0 : qg1;
        double r = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
        for (int i = 0; i < o; ++i) {
            const XCK_T *f = psi + i * ld + g;
            const double pv = pz[((long long)d * o + i) * np + g];
            r += pv * f[0];
            gx += pv * f[np];
            gy += pv * f[2 * np];
            gz += pv * f[3 * np];
        }
        if (o > 0) {
            const XCK_T *q = qg + (long long)d * nf * np + g;
            for (int m = 0; m < nf; ++m) {
                const double qm = q[(long long)m * np];
                const XCK_T *ph = phi + (long long)m * ld + g;
                gx += ph[np] * qm;
                gy += ph[2 * np] * qm;
                gz += ph[3 * np] * qm;
            }
        }
        u[4 * s] = 2.0 * scale * r;
        u[4 * s + 1] = 2.0 * scale * gx;
        u[4 * s + 2] = 2.0 * scale * gy;
        u[4 * s + 3] = 2.0 * scale * gz;
    }
    for (int c = 0; c < nu; ++c) {
        double v = 0.0;
        for (int e = 0; e < nu; ++e) v += wf[((long long)c * nu + e) * npts + P] * u[e];
        tt[c] = v;
    }
    for (int s = 0; s < nsp; ++s) {
        const int o = s == 0 ? o0 : o1;
        if (o == 0) continue;
        const XCK_T *psi = s == 0 ? psi0 : psi1;
        XCK_T *pz = s == 0 ? pz0 : pz1;
        XCK_T *qg = s == 0 ? qg0 : qg1;
        const double t0 = tt[4 * s], t1 = tt[4 * s + 1], t2 = tt[4 * s + 2], t3 = tt[4 * s + 3];
        for (int i = 0; i < o; ++i) {
            const XCK_T *f = psi + i * ld + g;
            pz[((long long)d * o + i) * np + g] = t0 * f[0] + t1 * f[np] + t2 * f[2 * np] + t3 * f[3 * np];
        }
        XCK_T *q = qg + (long long)d * nf * np + g;
        for (int m = 0; m < nf; ++m) {
            const XCK_T *ph = phi + (long long)m * ld + g;
            q[(long long)m * np] = t1 * ph[np] + t2 * ph[2 * np] + t3 * ph[3 * np];
        }
    }
}

// One thread per (point g, perturbation d of the group): u1 of every spin from P = F A and
// Phi = F C (rho1 = 2 sum P Phi, grad rho1 = 2 sum (grad P Phi + P grad Phi), tau1 = sum grad P
// . grad Phi; times scale), t = wF u1, and P overwritten by Z = [t_rho Phi + t_k d_k Phi;
// t_k Phi + 1/2 t_tau d_k Phi]. Spin s: Phi_s (nc np x o_s), P_s (nc np x ndg o_s).
extern "C" __global__ void XCK_SFX(xck_pert)(int np, long long p0, long long npts, int nc, int nsp,
    int ns, int gga, int mgga, double scale, int ndg, int o0, const XCK_T *__restrict__ phi0,
    XCK_T *__restrict__ pz0, int o1, const XCK_T *__restrict__ phi1, XCK_T *__restrict__ pz1,
    const double *__restrict__ wf)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)np * ndg) return;
    const int g = (int)(t % np), d = (int)(t / np);
    const long long ld = (long long)nc * np, P = p0 + g;
    const int nu = nsp * ns;
    double u[XCK_MAXU], tt[XCK_MAXU];
    for (int s = 0; s < nsp; ++s) {
        const int o = s == 0 ? o0 : o1;
        const XCK_T *phi = s == 0 ? phi0 : phi1;
        const XCK_T *pz = s == 0 ? pz0 : pz1;
        double r = 0.0, gx = 0.0, gy = 0.0, gz = 0.0, ta = 0.0;
        for (int i = 0; i < o; ++i) {
            const XCK_T *f = phi + i * ld + g;
            const XCK_T *p = pz + ((long long)d * o + i) * ld + g;
            r += p[0] * f[0];
            if (gga) {
                gx += p[np] * f[0] + p[0] * f[np];
                gy += p[2 * np] * f[0] + p[0] * f[2 * np];
                gz += p[3 * np] * f[0] + p[0] * f[3 * np];
            }
            if (mgga) ta += p[np] * f[np] + p[2 * np] * f[2 * np] + p[3 * np] * f[3 * np];
        }
        u[s * ns] = 2.0 * scale * r;
        if (gga) {
            u[s * ns + 1] = 2.0 * scale * gx;
            u[s * ns + 2] = 2.0 * scale * gy;
            u[s * ns + 3] = 2.0 * scale * gz;
        }
        if (mgga) u[s * ns + 4] = scale * ta;
    }
    for (int c = 0; c < nu; ++c) {
        double v = 0.0;
        for (int e = 0; e < nu; ++e) v += wf[((long long)c * nu + e) * npts + P] * u[e];
        tt[c] = v;
    }
    for (int s = 0; s < nsp; ++s) {
        const int o = s == 0 ? o0 : o1;
        const XCK_T *phi = s == 0 ? phi0 : phi1;
        XCK_T *pz = s == 0 ? pz0 : pz1;
        const double *ts = tt + s * ns;
        for (int i = 0; i < o; ++i) {
            const XCK_T *f = phi + i * ld + g;
            XCK_T *z = pz + ((long long)d * o + i) * ld + g;
            if (gga) {
                const double f0 = f[0], fx = f[np], fy = f[2 * np], fz = f[3 * np];
                const double ht = mgga ? 0.5 * ts[4] : 0.0;
                z[0] = ts[0] * f0 + ts[1] * fx + ts[2] * fy + ts[3] * fz;
                z[np] = ts[1] * f0 + ht * fx;
                z[2 * np] = ts[2] * f0 + ht * fy;
                z[3 * np] = ts[3] * f0 + ht * fz;
            } else {
                z[0] = ts[0] * f[0];
            }
        }
    }
}
