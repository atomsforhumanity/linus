
// linus-gpu XC Hessian terms (src/hess/xc.rs). Batch descriptor j: hb[XH_HB j + field]
// (H_* in the host code); compressed values phic[phi + (l nc + c) np + g] (XCP's layout);
// X_s[xstride s + x + (l nx + c) np + g] (components c of the collocation, c < nx);
// per point (point-major): w[P], v[P nu + c] (df/du, unweighted), wf[P nu^2 + c nu + e]
// (w d2f/du2); u components c = s ns + (0: rho, 1..3: grad rho).

#define XH_HB 32
#define H_P0 0
#define H_NP 1
#define H_NF 2
#define H_PHI 3
#define H_X 4
#define H_D 5
#define H_FL 6
#define H_NAT 7
#define H_AT 8
#define H_APTR 9
#define H_AFN 10
#define H_LA 11
#define H_U 12
#define H_S 13
#define H_R 14
#define H_TJ 15
#define H_EL 16
#define H_NG 17
#define H_GAT 18
#define H_DW 19
#define H_NO 20
#define H_OWN 21
#define H_AB 22
#define H_G 23
#define H_CC 24
#define H_LR 25
#define H_TG 26
#define H_GPT 27
#define H_OPT 28
#define H_OOF 29
#define H_OPTR 30

#define XH_LOOP(e, total) \
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < (total); \
         e += (long long)gridDim.x * blockDim.x)

// d_j u at each point (grid terms): du[j][s ns] = 2 sum_l d_j phi X_s0, du[j][s ns + 1 + k] =
// 2 sum_l (d_jk phi X_s0 + d_j phi X_sk), (meta-GGA) du[j][s ns + 4] = sum_l sum_k d_jk phi
// X_sk; t_j = wF d_j u -> tj[(j nu + c) np + g]
extern "C" __global__ void xh_point(int nc, int nx, int nsp, int ns, int gga, int mgga,
    const long long *__restrict__ hb, int j0, const double *__restrict__ phic,
    const double *__restrict__ xb, long long xstride, const double *__restrict__ wf,
    double *__restrict__ tj)
{
    const long long *d = hb + XH_HB * (long long)(j0 + blockIdx.y);
    const long long np = d[H_NP], nf = d[H_NF], po = d[H_PHI], xo = d[H_X], p0 = d[H_P0];
    const int nu = nsp * ns;
    XH_LOOP(g, np) {
        double du[3][10];
        for (int a = 0; a < 3; ++a)
            for (int c = 0; c < 10; ++c) du[a][c] = 0.0;
        for (int s = 0; s < nsp; ++s) {
            const double *xs = xb + s * xstride + xo;
            for (long long l = 0; l < nf; ++l) {
                const double *ph = phic + po + l * nc * np + g;
                const double *x = xs + l * nx * np + g;
                const double x0 = x[0];
                for (int j = 0; j < 3; ++j) {
                    const double pj = ph[XH_D1[j] * np];
                    du[j][s * ns] += 2.0 * pj * x0;
                    if (gga)
                        for (int k = 0; k < 3; ++k)
                            du[j][s * ns + 1 + k] +=
                                2.0 * (ph[XH_D2[3 * j + k] * np] * x0 + pj * x[XH_D1[k] * np]);
                    if (mgga)
                        for (int k = 0; k < 3; ++k)
                            du[j][s * ns + 4] += ph[XH_D2[3 * j + k] * np] * x[XH_D1[k] * np];
                }
            }
        }
        const double *f = wf + (p0 + g) * nu * nu;
        for (int j = 0; j < 3; ++j)
            for (int c = 0; c < nu; ++c) {
                double t = 0.0;
                for (int e = 0; e < nu; ++e) t += f[c * nu + e] * du[j][e];
                tj[d[H_TJ] + (j * nu + c) * np + g] = t;
            }
    }
}

// per (point, local atom a): u^{(a,i)}, t = wF u -> u, t [(3a + i) nu + c][g] (meta-GGA: tau^{ai}
// = -sum_k d_ki phi X_k); with grid terms s = v.u -> s[(3a + i)][g] and r_j = t_j.u + wv.d_j u
// -> r[((3a + i) 3 + j)][g]
extern "C" __global__ void xh_atom(int nc, int nx, int nsp, int ns, int gga, int mgga, int grid,
    const long long *__restrict__ hb, int j0, const int *__restrict__ ints,
    const double *__restrict__ phic, const double *__restrict__ xb, long long xstride,
    const double *__restrict__ w, const double *__restrict__ vu, const double *__restrict__ wf,
    const double *__restrict__ tjb, double *__restrict__ ub, double *__restrict__ tb,
    double *__restrict__ sb, double *__restrict__ rb)
{
    const long long *d = hb + XH_HB * (long long)(j0 + blockIdx.y);
    const long long np = d[H_NP], po = d[H_PHI], xo = d[H_X], p0 = d[H_P0];
    const int nat = (int)d[H_NAT];
    const int *aptr = ints + d[H_APTR], *afn = ints + d[H_AFN];
    const int nu = nsp * ns;
    XH_LOOP(e, np * nat) {
        const long long g = e % np;
        const int a = (int)(e / np);
        const long long P = p0 + g;
        const double wg = w[P];
        double u[3][10], r[3][3];
        for (int i = 0; i < 3; ++i) {
            for (int c = 0; c < 10; ++c) u[i][c] = 0.0;
            for (int j = 0; j < 3; ++j) r[i][j] = 0.0;
        }
        for (int s = 0; s < nsp; ++s) {
            const double *xs = xb + s * xstride + xo;
            const double *vv = vu + P * nu + s * ns;
            const double wv0 = wg * vv[0];
            double wvk[3] = {0.0, 0.0, 0.0};
            if (gga)
                for (int k = 0; k < 3; ++k) wvk[k] = wg * vv[1 + k];
            const double wvt = mgga ? wg * vv[4] : 0.0;
            for (int ff = aptr[a]; ff < aptr[a + 1]; ++ff) {
                const long long l = afn[ff];
                const double *ph = phic + po + l * nc * np + g;
                const double *x = xs + l * nx * np + g;
                const double x0 = x[0];
                double xk[3];
                for (int k = 0; k < 3; ++k) xk[k] = x[XH_D1[k] * np];
                for (int i = 0; i < 3; ++i) {
                    const double pi = ph[XH_D1[i] * np];
                    u[i][s * ns] += -2.0 * pi * x0;
                    if (gga)
                        for (int k = 0; k < 3; ++k)
                            u[i][s * ns + 1 + k] +=
                                -2.0 * (ph[XH_D2[3 * k + i] * np] * x0 + pi * xk[k]);
                    if (mgga)
                        for (int k = 0; k < 3; ++k)
                            u[i][s * ns + 4] += -1.0 * ph[XH_D2[3 * k + i] * np] * xk[k];
                }
                if (grid) {
                    double al = wv0 * x0, be[3], ga[3] = {0.0, 0.0, 0.0}, de[9];
                    for (int j = 0; j < 3; ++j) be[j] = wv0 * xk[j];
                    for (int q = 0; q < 9; ++q) de[q] = 0.0;
                    if (gga)
                        for (int k = 0; k < 3; ++k) {
                            al += wvk[k] * xk[k];
                            ga[k] = wvk[k] * x0;
                            for (int j = 0; j < 3; ++j) {
                                be[j] += wvk[k] * x[XH_D2[3 * j + k] * np];
                                de[3 * j + k] = wvk[k] * xk[j];
                            }
                        }
                    // meta-GGA: gamma_k += 1/2 wv_tau X_k, delta_jk += 1/2 wv_tau X_jk
                    if (mgga)
                        for (int k = 0; k < 3; ++k) {
                            ga[k] += 0.5 * wvt * xk[k];
                            for (int j = 0; j < 3; ++j)
                                de[3 * j + k] += 0.5 * wvt * x[XH_D2[3 * j + k] * np];
                        }
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j) {
                            double t = ph[XH_D2[3 * i + j] * np] * al + ph[XH_D1[i] * np] * be[j];
                            if (gga)
                                for (int k = 0; k < 3; ++k)
                                    t += ph[XH_D3[9 * i + 3 * j + k] * np] * ga[k]
                                        + ph[XH_D2[3 * i + k] * np] * de[3 * j + k];
                            r[i][j] += -2.0 * t;
                        }
                }
            }
        }
        const double *f = wf + P * nu * nu;
        for (int i = 0; i < 3; ++i) {
            const long long p = 3 * a + i;
            for (int c = 0; c < nu; ++c) {
                double t = 0.0;
                for (int q = 0; q < nu; ++q) t += f[c * nu + q] * u[i][q];
                ub[d[H_U] + (p * nu + c) * np + g] = u[i][c];
                tb[d[H_U] + (p * nu + c) * np + g] = t;
            }
            if (grid) {
                double sx = 0.0;
                for (int c = 0; c < nu; ++c) sx += vu[P * nu + c] * u[i][c];
                sb[d[H_S] + p * np + g] = sx;
                for (int j = 0; j < 3; ++j) {
                    double rr = r[i][j];
                    for (int c = 0; c < nu; ++c)
                        rr += tjb[d[H_TJ] + (j * nu + c) * np + g] * u[i][c];
                    rb[d[H_R] + (p * 3 + j) * np + g] = rr;
                }
            }
        }
    }
}

// the same-function second derivatives: one warp per (local atom a, pair i <= j), lanes over
// the points, a fixed shuffle tree; el[(3a + j) npl + 3a + i] += v (and the transpose)
// (meta-GGA: the d_kij phi factor 2 wv_k X + wv_tau X_k)
extern "C" __global__ void xh_same(int nc, int nx, int nsp, int ns, int gga, int mgga,
    const long long *__restrict__ hb, int j0, const int *__restrict__ ints,
    const double *__restrict__ phic, const double *__restrict__ xb, long long xstride,
    const double *__restrict__ w, const double *__restrict__ vu, double *__restrict__ el)
{
    const long long *d = hb + XH_HB * (long long)(j0 + blockIdx.y);
    const long long np = d[H_NP], po = d[H_PHI], xo = d[H_X], p0 = d[H_P0];
    const int nat = (int)d[H_NAT], npl = 3 * nat;
    const int *aptr = ints + d[H_APTR], *afn = ints + d[H_AFN];
    const int nu = nsp * ns;
    const int lane = threadIdx.x % LINUS_WARP_SIZE, wpb = blockDim.x / LINUS_WARP_SIZE;
    for (int wi = blockIdx.x * wpb + threadIdx.x / LINUS_WARP_SIZE; wi < 6 * nat;
         wi += gridDim.x * wpb) {
        const int a = wi / 6, q = wi % 6;
        const int i = q < 3 ? 0 : (q < 5 ? 1 : 2);
        const int j = q < 3 ? q : (q < 5 ? q - 2 : 2);
        double v = 0.0;
        for (long long g = lane; g < np; g += LINUS_WARP_SIZE) {
            const long long P = p0 + g;
            const double wg = w[P];
            for (int s = 0; s < nsp; ++s) {
                const double *vv = vu + P * nu + s * ns;
                const double wv0 = wg * vv[0];
                double wvk[3] = {0.0, 0.0, 0.0};
                if (gga)
                    for (int k = 0; k < 3; ++k) wvk[k] = wg * vv[1 + k];
                const double wvt = mgga ? wg * vv[4] : 0.0;
                const double *xs = xb + s * xstride + xo;
                for (int ff = aptr[a]; ff < aptr[a + 1]; ++ff) {
                    const long long l = afn[ff];
                    const double *ph = phic + po + l * nc * np + g;
                    const double *x = xs + l * nx * np + g;
                    const double x0 = x[0];
                    double s0 = 2.0 * wv0 * x0;
                    if (gga)
                        for (int k = 0; k < 3; ++k) s0 += 2.0 * wvk[k] * x[XH_D1[k] * np];
                    double t = ph[XH_D2[3 * i + j] * np] * s0;
                    if (gga)
                        for (int k = 0; k < 3; ++k) {
                            double sk = 2.0 * wvk[k] * x0;
                            if (mgga) sk += wvt * x[XH_D1[k] * np];
                            t += ph[XH_D3[9 * k + 3 * i + j] * np] * sk;
                        }
                    v += t;
                }
            }
        }
        for (int o = LINUS_WARP_SIZE / 2; o > 0; o >>= 1) v += linus_shfl_down(v, o);
        if (lane == 0) {
            el[d[H_EL] + (3 * a + j) * npl + 3 * a + i] += v;
            if (i != j) el[d[H_EL] + (3 * a + i) * npl + 3 * a + j] += v;
        }
    }
}

// the cross term's L and R of spin s (column i nf + l, rows g): L = d_i phi; R = 2 wv_rho
// d_i phi (LDA) or 2 Q (GGA), Q = wv_rho d_i phi + 2 sum_k wv_k d_ki phi. The CPU's GGA C^{ij} =
// P_i^T Q_j + Q_i^T P_j: the second part adds the transpose of the first's contribution to E,
// which is symmetrized at the end, so 2 P^T Q gives the same E. Meta-GGAs stack below them
// (columns of 4 np rows) the tau term's operands H_ki = d_ki phi and wv_tau H_ki (k = x, y, z):
// C^{ij} += sum_k H_ki^T wv_tau H_kj, as the CPU's
extern "C" __global__ void xh_lr(int nc, int nsp, int ns, int gga, int mgga, int s,
    const long long *__restrict__ hb, int j0, const double *__restrict__ phic,
    const double *__restrict__ w, const double *__restrict__ vu, double *__restrict__ lb,
    double *__restrict__ rb)
{
    const long long *d = hb + XH_HB * (long long)(j0 + blockIdx.y);
    const long long np = d[H_NP], nf = d[H_NF], po = d[H_PHI], p0 = d[H_P0];
    const int nu = nsp * ns;
    const long long nr = mgga ? 4 : 1;
    XH_LOOP(e, 3 * nf * np) {
        const long long g = e % np, rr = e / np;
        const long long l = rr % nf;
        const int i = (int)(rr / nf);
        const long long P = p0 + g;
        const double wg = w[P];
        const double *vv = vu + P * nu + s * ns;
        const double *ph = phic + po + l * nc * np + g;
        const double pi = ph[XH_D1[i] * np];
        const long long col = d[H_LR] + (i * nf + l) * nr * np;
        if (!gga) {
            lb[col + g] = pi;
            rb[col + g] = 2.0 * (wg * vv[0]) * pi;
        } else {
            double q = (wg * vv[0]) * pi;
            for (int k = 0; k < 3; ++k) q += 2.0 * (wg * vv[1 + k]) * ph[XH_D2[3 * k + i] * np];
            lb[col + g] = pi;
            rb[col + g] = 2.0 * q;
            if (mgga) {
                const double wvt = wg * vv[4];
                for (int k = 0; k < 3; ++k) {
                    const double h = ph[XH_D2[3 * k + i] * np];
                    lb[col + (1 + k) * np + g] = h;
                    rb[col + (1 + k) * np + g] = wvt * h;
                }
            }
        }
    }
}

// el[q npl + p] += sum_{mu in A} sum_{nu in B} D_s[nu nf + mu] C[(j nf + nu) 3nf + i nf + mu]
// (p = 3A + i, q = 3B + j); with scatter, the element (with everything before it) into the
// fixed-point E at (3 atom_A + i, 3 atom_B + j)
extern "C" __global__ void xh_cross(int scatter, long long n3, const long long *__restrict__ hb,
    int j0, const int *__restrict__ ints, const double *__restrict__ dsub,
    const double *__restrict__ cc, double *__restrict__ el, unsigned long long *acc,
    long long nacc)
{
    const long long *d = hb + XH_HB * (long long)(j0 + blockIdx.y);
    const long long nf = d[H_NF], dd = d[H_D], co = d[H_CC], eo = d[H_EL];
    const int nat = (int)d[H_NAT], npl = 3 * nat;
    const int *aptr = ints + d[H_APTR], *afn = ints + d[H_AFN], *atoms = ints + d[H_AT];
    const long long n3f = 3 * nf;
    XH_LOOP(e, (long long)npl * npl) {
        const int p = (int)(e % npl), q = (int)(e / npl);
        const int A = p / 3, i = p % 3, B = q / 3, j = q % 3;
        double v = 0.0;
        for (int fb = aptr[B]; fb < aptr[B + 1]; ++fb) {
            const long long nv = afn[fb];
            const double *ccol = cc + co + (j * nf + nv) * n3f + i * nf;
            const double *dr = dsub + dd + nv * nf;
            for (int fa = aptr[A]; fa < aptr[A + 1]; ++fa) {
                const long long m = afn[fa];
                v += dr[m] * ccol[m];
            }
        }
        const double tot = el[eo + (long long)q * npl + p] + v;
        if (scatter) {
            const long long gp = 3LL * atoms[A] + i, gq = 3LL * atoms[B] + j;
            det_add(acc, nacc, gp * n3 + gq, tot);
        } else {
            el[eo + (long long)q * npl + p] = tot;
        }
    }
}

// E1[3 atom_p + i][3 C + j] += sum_g s_p dw_C,j + sum over C's points of r_p,j
extern "C" __global__ void xh_e1(long long n3, const long long *__restrict__ hb, int j0,
    const int *__restrict__ ints, const int *__restrict__ owner, const double *__restrict__ sb,
    const double *__restrict__ rb, const double *__restrict__ dw, unsigned long long *acc,
    long long nacc)
{
    const long long *d = hb + XH_HB * (long long)(j0 + blockIdx.y);
    const long long np = d[H_NP], p0 = d[H_P0];
    const int nat = (int)d[H_NAT], ng = (int)d[H_NG];
    const int *atoms = ints + d[H_AT], *gat = ints + d[H_GAT];
    XH_LOOP(e, 9LL * nat * ng) {
        const int j = (int)(e % 3);
        const long long r = e / 3;
        const int ga = (int)(r % ng), p = (int)(r / ng);
        const double *sp = sb + d[H_S] + p * np;
        const double *wd = dw + d[H_DW] + ((long long)ga * 3 + j) * np;
        const double *rp = rb + d[H_R] + ((long long)p * 3 + j) * np;
        const int C = gat[ga];
        double v = 0.0;
        for (long long g = 0; g < np; ++g) {
            v += sp[g] * wd[g];
            if (owner[p0 + g] == C) v += rp[g];
        }
        const long long gp = 3LL * atoms[p / 3] + p % 3;
        det_add(acc, nacc, gp * n3 + 3LL * C + j, v);
    }
}

// The operands of the G matrices per owner o, over o's points only (the batch's points grouped
// by owner: position q of opts, owner oof[q], o's positions optr[o]..optr[o + 1]): A_si
// (comps 0..2), B_s (3), phi (4) and d_k phi (5..7), each nf x k_o (k_o = o's points):
// ab[(nsp nab + nh) nf optr[o] + ((s nab + comp) nf + l) k_o + (q - optr[o])] (nab = 8, nh = 0).
// Meta-GGAs (nab = 11, nh = 9) add the tau term's operands, stacked over k = x, y, z as 3 k_o
// rows per function: per spin T_s,k = 1/2 wv_tau,s d_k phi at comp 8 (rows k k_o + q - optr[o]
// of column l: ((s nab + 8) nf + 3 l) k_o + ...), and (spin-free) per i the stack H_ki = d_ki phi
// at ((nsp nab + 3 i) nf + 3 l) k_o + ...: G_s^i += H_i^T T_s (3 k_o deep)
extern "C" __global__ void xh_ab(int nc, int nsp, int ns, int gga, int mgga,
    const long long *__restrict__ hb, int j0, const int *__restrict__ ints,
    const double *__restrict__ phic, const double *__restrict__ w,
    const double *__restrict__ vu, double *__restrict__ ab)
{
    const long long *d = hb + XH_HB * (long long)(j0 + blockIdx.y);
    const long long np = d[H_NP], nf = d[H_NF], po = d[H_PHI], p0 = d[H_P0];
    const int *opts = ints + d[H_OPT], *oof = ints + d[H_OOF], *optr = ints + d[H_OPTR];
    const int nu = nsp * ns;
    const long long nab = mgga ? 11 : 8, nh = mgga ? 9 : 0;
    XH_LOOP(e, (long long)nsp * nf * np) {
        const long long q = e % np;
        const long long r = e / np;
        const long long l = r % nf;
        const int s = (int)(r / nf);
        const int o = oof[q];
        const long long q0 = optr[o], ko = optr[o + 1] - q0;
        const long long g = opts[q];
        const long long P = p0 + g;
        const double wg = w[P];
        const double *vv = vu + P * nu + s * ns;
        const double wv0 = wg * vv[0];
        const double *ph = phic + po + l * nc * np + g;
        const long long ob = d[H_AB] + (nsp * nab + nh) * nf * q0;
        const long long base = ob + ((long long)s * nab * nf + l) * ko + (q - q0);
        const long long cs = nf * ko;
        for (int i = 0; i < 3; ++i) {
            double a = wv0 * ph[XH_D1[i] * np];
            if (gga)
                for (int k = 0; k < 3; ++k) a += (wg * vv[1 + k]) * ph[XH_D2[3 * k + i] * np];
            ab[base + i * cs] = a;
            ab[base + (5 + i) * cs] = ph[XH_D1[i] * np];
        }
        double b = 0.0;
        if (gga)
            for (int k = 0; k < 3; ++k) b += (wg * vv[1 + k]) * ph[XH_D1[k] * np];
        ab[base + 3 * cs] = b;
        ab[base + 4 * cs] = ph[0];
        if (mgga) {
            const double ht = 0.5 * (wg * vv[4]);
            const long long tb = ob + ((long long)s * nab + 8) * cs + 3 * l * ko + (q - q0);
            for (int k = 0; k < 3; ++k) ab[tb + k * ko] = ht * ph[XH_D1[k] * np];
            if (s == 0)
                for (int i = 0; i < 3; ++i) {
                    const long long hb0 = ob + (nsp * nab + 3 * i) * cs + 3 * l * ko + (q - q0);
                    for (int k = 0; k < 3; ++k) ab[hb0 + k * ko] = ph[XH_D2[3 * k + i] * np];
                }
        }
    }
}

// the grid slots' first-order potentials: tg[((ga 3 + j) nu + c) np + g] = dw_ga,j v_c
// + (owner = gat[ga]) t_j,c
extern "C" __global__ void xh_ttg(int nu, const long long *__restrict__ hb, int j0,
    const int *__restrict__ ints, const int *__restrict__ owner, const double *__restrict__ vu,
    const double *__restrict__ tjb, const double *__restrict__ dw, double *__restrict__ tg)
{
    const long long *d = hb + XH_HB * (long long)(j0 + blockIdx.y);
    const long long np = d[H_NP], p0 = d[H_P0];
    const int ng = (int)d[H_NG];
    const int *gat = ints + d[H_GAT];
    XH_LOOP(e, (long long)ng * 3 * nu * np) {
        const long long g = e % np;
        long long r = e / np;
        const int c = (int)(r % nu);
        r /= nu;
        const int j = (int)(r % 3), ga = (int)(r / 3);
        const long long P = p0 + g;
        double v = dw[d[H_DW] + ((long long)ga * 3 + j) * np + g] * vu[P * nu + c];
        if (owner[P] == gat[ga]) v += tjb[d[H_TJ] + ((long long)j * nu + c) * np + g];
        tg[d[H_TG] + (((long long)ga * 3 + j) * nu + c) * np + g] = v;
    }
}

// count full matrices (nbf x nbf, column-major) from packed lower triangles at base + x npack
// (NaN if the accumulator is poisoned)
extern "C" __global__ void xh_unpack(long long nbf, long long npack, long long base, int count,
    const unsigned long long *__restrict__ acc, long long nacc, double *__restrict__ out)
{
    XH_LOOP(e, nbf * nbf * count) {
        const long long r = e % nbf, t = e / nbf;
        const long long c = t % nbf, x = t / nbf;
        const long long hi = r > c ? r : c, lo = r > c ? c : r;
        out[e] = det_value(acc, nacc, base + x * npack + lo * (2 * nbf - lo + 1) / 2 + hi - lo);
    }
}
