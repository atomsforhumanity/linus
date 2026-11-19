
extern "C" __global__ void det_scatter_add(unsigned long long *acc, long long n, const long long *idx, const double *v, long long count) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) det_add(acc, n, idx[i], v[i]);
}

// acc[0] += sum of v[0..count] (one block-level partial sum per block)
extern "C" __global__ void det_sum(unsigned long long *acc, const double *v, long long count) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    det_add_block(acc, 1, 0, i < count ? v[i] : 0.0);
}
