// linus-gpu DF-K shell panels (src/df/mod.rs, PanelPlan): the stored rows of the pair-major
// tensor gathered per shell, and the occupied orbitals gathered per shell's partner functions.

// p[r * nq + q] = b[off + src[r] * ldb + q] for panel rows r < np and q < nq: a row copy per
// thread row (threadIdx.y), the threads of a row along q (reads and writes contiguous).
extern "C" __global__ void df_panel_gather(long long np, int nq, const int *__restrict__ src,
    const double *__restrict__ b, long long off, long long ldb, double *__restrict__ p)
{
    const long long r = (long long)blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= np) return;
    const double *row = b + off + (long long)src[r] * ldb;
    double *out = p + r * (long long)nq;
    for (int q = threadIdx.x; q < nq; q += blockDim.x) out[q] = row[q];
}

// cg[i + ldc * c] = cmat[cn[c] + nbf * i] for c < ncg and i < ntot (cmat nbf x ntot,
// column-major; rows of ldc >= ntot doubles, the padding left as it is): column c of the
// gathered orbitals is the row cn[c] of C, transposed.
extern "C" __global__ void df_panel_cg(long long ncg, int ntot, int ldc, long long nbf,
    const int *__restrict__ cn, const double *__restrict__ cmat, double *__restrict__ cg)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ncg * ntot) return;
    const long long i = t % ntot, c = t / ntot;
    cg[i + ldc * c] = cmat[(long long)cn[c] + nbf * i];
}
