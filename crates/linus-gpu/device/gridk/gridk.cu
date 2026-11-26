// linus-gpu's grid-K kernels beside the generated potential-integral classes
// (src/kernels/gridk.rs, src/gridk/): element-wise kernels of the setup and the builds, and
// the screening factors. One block per chunk where a kernel reduces over a chunk's points, in a
// fixed shared-memory tree (blocks of 256 threads, the launch's).
extern "C" __global__ void gridk_rowscale(long long nrow, long long ncol, long long ld,
    const double *__restrict__ v, const double *__restrict__ x, double *__restrict__ out)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrow * ncol) return;
    const long long r = i % nrow, c = i / nrow;
    out[c * ld + r] = v[r] * x[c * ld + r];
}

extern "C" __global__ void gridk_colscale(long long nrow, long long ncol, long long ld,
    const double *__restrict__ v, const double *__restrict__ x, double *__restrict__ out)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrow * ncol) return;
    const long long r = i % nrow, c = i / nrow;
    out[c * ld + r] = v[c] * x[c * ld + r];
}

extern "C" __global__ void gridk_symadd(long long n, const double *__restrict__ k,
    const double *__restrict__ k0, double *__restrict__ out)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * n) return;
    const long long r = i % n, c = i / n;
    const double v = 0.5 * (k[r + c * n] + k[c + r * n]);
    out[i] = k0 != 0 ? k0[i] + v : v;
}

extern "C" __global__ void gridk_slices(long long n, int nslice, long long stride,
    double *__restrict__ z)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s = z[i];
    for (int k = 1; k < nslice; ++k) s += z[(long long)k * stride + i];
    z[i] = s;
}

extern "C" __global__ void __launch_bounds__(256) gridk_cxw(const double *__restrict__ x,
    long long ld, int nbf, const int *__restrict__ chunk, const double *__restrict__ w,
    double *__restrict__ out)
{
    __shared__ double sm[256];
    const int ch = blockIdx.x;
    const int g0 = chunk[2 * ch], g1 = chunk[2 * ch + 1];
    double m = 0.0;
    for (int p = threadIdx.x; p < nbf; p += 256) {
        const double *col = x + (long long)p * ld;
        double s = 0.0;
        for (int g = g0; g < g1; ++g) s += w[g] * fabs(col[g]);
        m = fmax(m, s);
    }
    sm[threadIdx.x] = m;
    __syncthreads();
    for (int h = 128; h > 0; h >>= 1) {
        if ((int)threadIdx.x < h) sm[threadIdx.x] = fmax(sm[threadIdx.x], sm[threadIdx.x + h]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[ch] = sm[0];
}

extern "C" __global__ void gridk_ymax(const double *__restrict__ y, long long ld,
    const int *__restrict__ chunk, const int *__restrict__ shoff, int nshell,
    double *__restrict__ ymax)
{
    const int ch = blockIdx.x;
    const int g0 = chunk[2 * ch], g1 = chunk[2 * ch + 1];
    for (int s = threadIdx.x; s < nshell; s += blockDim.x) {
        double m = 0.0;
        for (int c = shoff[s]; c < shoff[s + 1]; ++c) {
            const double *row = y + (long long)c * ld;
            for (int g = g0; g < g1; ++g) m = fmax(m, fabs(row[g]));
        }
        ymax[(long long)ch * nshell + s] = m;
    }
}
