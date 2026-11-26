// The element-wise kernels of the block Krylov solver on the device (src/response/krylov.rs). One
// writer per output element; quotients are IEEE divisions, as the host solver's.

// z[:, j] = r[:, idx[j]] / d over len rows and nc columns (column-major, leading dimension len):
// the Jacobi-preconditioned residuals of the active right-hand sides
extern "C" __global__ void krylov_gather_div(long long len, long long nc, const long long *idx,
    const double *r, const double *d, double *z)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= len * nc) return;
    const long long i = t % len, j = t / len;
    z[t] = r[idx[j] * len + i] / d[i];
}

// v[i] = z[i] / s over len elements: a normalized direction
extern "C" __global__ void krylov_div(long long len, const double *z, double s, double *v)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) v[i] = z[i] / s;
}
