
// ---- the per-perturbation pair sums of VV10's second derivatives (linus's xc::vv10_resp, whose
// module docs give the formulas): for point i and perturbation p,
//   P1 = sum_j [a_j K0 + h K1],  P2 = sum_j [a_j K2 + h K3],  P3 = sum_j r2 (a_j K2 + h K3),
//   h = b_j + r2 c_j,  K0 = -T, K1 = T a_j, K2 = T a_i, K3 = -T (a_i a_j + 1/g_t^2)
// (without their factor 3/2), T = 1/(g_i g_j g_t), a_i = 1/g_i + 1/g_t. q: x, y, z, omega,
// kappa of the active points (npad each, padding at omega = kappa = 1); coef[(p 3 + c) ld + j]:
// a, b, c of perturbation p at point j < n (zero beyond); out[(c npp + p) npad + i]. One thread
// per point i over all points j in ascending order (tiles of VV_TILE through shared memory), VR_PC
// perturbations per block (grid: npad / VV_TILE x ceil(npp / VR_PC)): the order of every sum is
// fixed by the input, so the results are bitwise run to run and card to card.
#define VR_PC 4

extern "C" __global__ void __launch_bounds__(VV_TILE) vv10_resp_f64(int n, int npad, int ld,
    int npp, const double *__restrict__ q, const double *__restrict__ coef,
    double *__restrict__ out)
{
    __shared__ double tq[5][VV_TILE];
    __shared__ double tc[3 * VR_PC][VV_TILE];
    const int t = threadIdx.x;
    const long long m = npad, i = (long long)blockIdx.x * VV_TILE + t;
    const int p0 = blockIdx.y * VR_PC;
    const double xi = q[i], yi = q[m + i], zi = q[2 * m + i], oi = q[3 * m + i],
        ki = q[4 * m + i];
    double a1[VR_PC], a2[VR_PC], a3[VR_PC];
    for (int r = 0; r < VR_PC; ++r) {
        a1[r] = 0.0;
        a2[r] = 0.0;
        a3[r] = 0.0;
    }
    for (long long j0 = 0; j0 < m; j0 += VV_TILE) {
        const long long j = j0 + t;
        for (int k = 0; k < 5; ++k) tq[k][t] = q[k * m + j];
        for (int k = 0; k < 3 * VR_PC; ++k) {
            const int p = p0 + k / 3;
            tc[k][t] = (p < npp && j < n) ? coef[((long long)p * 3 + k % 3) * ld + j] : 0.0;
        }
        __syncthreads();
        for (int c = 0; c < VV_TILE; ++c) {
            const double dx = xi - tq[0][c], dy = yi - tq[1][c], dz = zi - tq[2][c];
            const double r2 = fma(dz, dz, fma(dy, dy, dx * dx));
            const double gi = fma(r2, oi, ki);
            const double gj = fma(r2, tq[3][c], tq[4][c]);
            const double gt = gi + gj;
            const double gg = gi * gj;
            const double tt = 1.0 / (gg * gt);
            const double ig = gj * gt * tt, igj = gi * gt * tt, igt = gg * tt;
            const double ai = ig + igt, aj = igj + igt;
            const double k0 = -tt, k1 = tt * aj, k2 = tt * ai;
            const double k3 = -(tt * fma(ai, aj, igt * igt));
#pragma unroll
            for (int r = 0; r < VR_PC; ++r) {
                const double al = tc[3 * r][c], b = tc[3 * r + 1][c], cc = tc[3 * r + 2][c];
                const double h = fma(r2, cc, b);
                a1[r] = fma(h, k1, fma(al, k0, a1[r]));
                const double qv = fma(h, k3, al * k2);
                a2[r] += qv;
                a3[r] = fma(r2, qv, a3[r]);
            }
        }
        __syncthreads();
    }
    for (int r = 0; r < VR_PC; ++r) {
        const long long p = p0 + r;
        if (p < npp) {
            out[p * m + i] = a1[r];
            out[(npp + p) * m + i] = a2[r];
            out[(2 * npp + p) * m + i] = a3[r];
        }
    }
}

// the same in double-float (qf: hi, lo of x, y, z, omega, kappa from vv10_split), each tile's sums
// in double-float, the tiles added in FP64
extern "C" __global__ void __launch_bounds__(VV_TILE) vv10_resp_df64(int n, int npad, int ld,
    int npp, const float *__restrict__ qf, const double *__restrict__ coef,
    double *__restrict__ out)
{
    __shared__ float tq[10][VV_TILE];
    __shared__ float tc[6 * VR_PC][VV_TILE];
    const int t = threadIdx.x;
    const long long m = npad, i = (long long)blockIdx.x * VV_TILE + t;
    const int p0 = blockIdx.y * VR_PC;
    vvdf xi, yi, zi, oi, ki;
    xi.h = qf[i];
    xi.l = qf[m + i];
    yi.h = qf[2 * m + i];
    yi.l = qf[3 * m + i];
    zi.h = qf[4 * m + i];
    zi.l = qf[5 * m + i];
    oi.h = qf[6 * m + i];
    oi.l = qf[7 * m + i];
    ki.h = qf[8 * m + i];
    ki.l = qf[9 * m + i];
    double A1[VR_PC], A2[VR_PC], A3[VR_PC];
    for (int r = 0; r < VR_PC; ++r) {
        A1[r] = 0.0;
        A2[r] = 0.0;
        A3[r] = 0.0;
    }
    for (long long j0 = 0; j0 < m; j0 += VV_TILE) {
        const long long j = j0 + t;
        for (int k = 0; k < 10; ++k) tq[k][t] = qf[k * m + j];
        for (int k = 0; k < 3 * VR_PC; ++k) {
            const int p = p0 + k / 3;
            const double v =
                (p < npp && j < n) ? coef[((long long)p * 3 + k % 3) * ld + j] : 0.0;
            const float h = (float)v;
            tc[2 * k][t] = h;
            tc[2 * k + 1][t] = (float)(v - (double)h);
        }
        __syncthreads();
        vvdf s1[VR_PC], s2[VR_PC], s3[VR_PC];
        for (int r = 0; r < VR_PC; ++r) {
            s1[r].h = 0.0f;
            s1[r].l = 0.0f;
            s2[r] = s1[r];
            s3[r] = s1[r];
        }
        for (int c = 0; c < VV_TILE; ++c) {
            vvdf xj, yj, zj, oj, kj;
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
            const vvdf dx = vv_sub_u(xi, xj), dy = vv_sub_u(yi, yj), dz = vv_sub_u(zi, zj);
            const vvdf r2 = vv_norm(vv_add_u(vv_add_u(vv_sqr_u(dx), vv_sqr_u(dy)), vv_sqr_u(dz)));
            const vvdf g = vv_fma_u(r2, oi, ki);
            const vvdf gp = vv_fma_u(r2, oj, kj);
            const vvdf gt = vv_add_u(g, gp);
            const vvdf gg = vv_mul_u(g, gp);
            const vvdf tt = vv_rcp(vv_mul_u(gg, gt));
            const vvdf ig = vv_mul_u(vv_mul_u(gp, gt), tt);
            const vvdf igj = vv_mul_u(vv_mul_u(g, gt), tt);
            const vvdf igt = vv_mul_u(gg, tt);
            const vvdf ai = vv_add_u(ig, igt), aj = vv_add_u(igj, igt);
            const vvdf k1 = vv_mul_u(tt, aj), k2 = vv_mul_u(tt, ai);
            // -K3 = T (a_i a_j + 1/g_t^2); -K0 = T
            const vvdf k3n = vv_mul_u(tt, vv_fma_u(ai, aj, vv_sqr_u(igt)));
#pragma unroll
            for (int r = 0; r < VR_PC; ++r) {
                vvdf al, b, cc;
                al.h = tc[6 * r][c];
                al.l = tc[6 * r + 1][c];
                b.h = tc[6 * r + 2][c];
                b.l = tc[6 * r + 3][c];
                cc.h = tc[6 * r + 4][c];
                cc.l = tc[6 * r + 5][c];
                const vvdf h = vv_fma_u(r2, cc, b);
                s1[r] = vv_add(s1[r], vv_sub_u(vv_mul_u(h, k1), vv_mul_u(al, tt)));
                const vvdf qv = vv_sub_u(vv_mul_u(al, k2), vv_mul_u(h, k3n));
                s2[r] = vv_add(s2[r], qv);
                s3[r] = vv_add(s3[r], vv_mul_u(r2, qv));
            }
        }
        for (int r = 0; r < VR_PC; ++r) {
            A1[r] += vv_f64(s1[r]);
            A2[r] += vv_f64(s2[r]);
            A3[r] += vv_f64(s3[r]);
        }
        __syncthreads();
    }
    for (int r = 0; r < VR_PC; ++r) {
        const long long p = p0 + r;
        if (p < npp) {
            out[p * m + i] = A1[r];
            out[(npp + p) * m + i] = A2[r];
            out[(2 * npp + p) * m + i] = A3[r];
        }
    }
}

// the same in FP32 (qs: x, y, z, omega, kappa in FP32 from vv10_split with df = 0; the
// coefficients rounded to FP32), each tile's sums in FP32, the tiles added in FP64: the CPKS
// kernel's products when the device's semilocal CPKS kernel runs FP32
extern "C" __global__ void __launch_bounds__(VV_TILE) vv10_resp_f32(int n, int npad, int ld,
    int npp, const float *__restrict__ qs, const double *__restrict__ coef,
    double *__restrict__ out)
{
    __shared__ float tq[5][VV_TILE];
    __shared__ float tc[3 * VR_PC][VV_TILE];
    const int t = threadIdx.x;
    const long long m = npad, i = (long long)blockIdx.x * VV_TILE + t;
    const int p0 = blockIdx.y * VR_PC;
    const float xi = qs[i], yi = qs[m + i], zi = qs[2 * m + i], oi = qs[3 * m + i],
        ki = qs[4 * m + i];
    double A1[VR_PC], A2[VR_PC], A3[VR_PC];
    for (int r = 0; r < VR_PC; ++r) {
        A1[r] = 0.0;
        A2[r] = 0.0;
        A3[r] = 0.0;
    }
    for (long long j0 = 0; j0 < m; j0 += VV_TILE) {
        const long long j = j0 + t;
        for (int k = 0; k < 5; ++k) tq[k][t] = qs[k * m + j];
        for (int k = 0; k < 3 * VR_PC; ++k) {
            const int p = p0 + k / 3;
            tc[k][t] = (p < npp && j < n) ? (float)coef[((long long)p * 3 + k % 3) * ld + j] : 0.0f;
        }
        __syncthreads();
        float s1[VR_PC], s2[VR_PC], s3[VR_PC];
        for (int r = 0; r < VR_PC; ++r) {
            s1[r] = 0.0f;
            s2[r] = 0.0f;
            s3[r] = 0.0f;
        }
        for (int c = 0; c < VV_TILE; ++c) {
            const float dx = xi - tq[0][c], dy = yi - tq[1][c], dz = zi - tq[2][c];
            const float r2 = fmaf(dz, dz, fmaf(dy, dy, dx * dx));
            const float gi = fmaf(r2, oi, ki);
            const float gj = fmaf(r2, tq[3][c], tq[4][c]);
            const float gt = gi + gj;
            const float gg = gi * gj;
            // 1/P, not 1/P^2: P^2 leaves the FP32 range for distant points of large molecules
            const float tt = __frcp_rn(gg * gt);
            const float ig = gj * gt * tt, igj = gi * gt * tt, igt = gg * tt;
            const float ai = ig + igt, aj = igj + igt;
            const float k0 = -tt, k1 = tt * aj, k2 = tt * ai;
            const float k3 = -(tt * fmaf(ai, aj, igt * igt));
#pragma unroll
            for (int r = 0; r < VR_PC; ++r) {
                const float al = tc[3 * r][c], b = tc[3 * r + 1][c], cc = tc[3 * r + 2][c];
                const float h = fmaf(r2, cc, b);
                s1[r] = fmaf(h, k1, fmaf(al, k0, s1[r]));
                const float qv = fmaf(h, k3, al * k2);
                s2[r] += qv;
                s3[r] = fmaf(r2, qv, s3[r]);
            }
        }
        for (int r = 0; r < VR_PC; ++r) {
            A1[r] += (double)s1[r];
            A2[r] += (double)s2[r];
            A3[r] += (double)s3[r];
        }
        __syncthreads();
    }
    for (int r = 0; r < VR_PC; ++r) {
        const long long p = p0 + r;
        if (p < npp) {
            out[p * m + i] = A1[r];
            out[(npp + p) * m + i] = A2[r];
            out[(2 * npp + p) * m + i] = A3[r];
        }
    }
}
