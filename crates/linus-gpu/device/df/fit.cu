
// x[c ldx + r] = t[c ldt + r] for r < m, c < n (src/df/mod.rs, the Ozaki fit)
extern "C" __global__ void df_fit_store(long long m, long long n, const double *__restrict__ t,
    long long ldt, double *__restrict__ x, long long ldx)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * n) return;
    const long long r = i % m, c = i / m;
    x[c * ldx + r] = t[c * ldt + r];
}
