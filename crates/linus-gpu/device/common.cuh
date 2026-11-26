// Device functions shared by linus-gpu's kernel modules: pasted in front of a module's source by
// the engine that assembles it (the device layer prepends portable.h before it). Kernels shared
// by several engines are modules of their own instead (blas1.cu, df/deriv.cu).

#ifndef LINUS_COMMON_CUH
#define LINUS_COMMON_CUH

// (i, j) of the packed lower-triangle index k = i (i + 1) / 2 + j, i >= j
__device__ __forceinline__ void linus_packed_ij(long long k, long long &i, long long &j)
{
    i = (long long)((sqrt(8.0 * (double)k + 1.0) - 1.0) * 0.5);
    while (i * (i + 1) / 2 > k) --i;
    while ((i + 1) * (i + 2) / 2 <= k) ++i;
    j = k - i * (i + 1) / 2;
}

// Becke's switching function s(nu) = p(p(p(nu))) / 2, p(x) = (3 - x^2) x / 2: the cell factors of
// an atom pair are 0.5 - s and 0.5 + s (linus-grid's becke_s, operation for operation)
__device__ __forceinline__ double linus_becke_s(double nu)
{
    double t = nu;
    t = (3.0 - t * t) * t * 0.5;
    t = (3.0 - t * t) * t * 0.5;
    return ((3.0 - t * t) * t * 0.5) * 0.5;
}

#endif
