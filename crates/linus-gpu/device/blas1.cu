// linus-gpu's vector kernels (BLAS level 1, src/linalg/blas1.rs), shared by the engines:
// elementwise, one thread per element. Any operand may be the output itself (in place), so no
// pointer is __restrict__. Each kernel's rounding is fixed by its expression (fma where a term is
// fused), never left to the compiler's contraction: the engines' results depend on it bit for
// bit.

// out[i] = s_i x[i], with s_i = w[i] when w is given (a diagonal scaling), else a
extern "C" __global__ void blas1_scale(long long n, double a, const double *w, const double *x,
    double *out)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (w != 0 ? w[i] : a) * x[i];
}

// out[i] = a x[i] + b y[i] + c z[i] over the vectors given (y and z optional), summed left to
// right: the first product rounded, each further term fused into the sum (fma), so a caller fixes
// its rounding by the order of its terms
extern "C" __global__ void blas1_axpby(long long n, double a, const double *x, double b,
    const double *y, double c, const double *z, double *out)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = a * x[i];
    if (y != 0) v = fma(b, y[i], v);
    if (z != 0) v = fma(c, z[i], v);
    out[i] = v;
}
