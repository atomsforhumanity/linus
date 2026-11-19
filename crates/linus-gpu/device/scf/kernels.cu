
#define RED_THREADS 256
#define RED_BLOCKS 128

// (hi, lo) += v: TwoSum, the error term summed separately (Neumaier/Kahan-Babuska)
__device__ __forceinline__ void dd_add(double &hi, double &lo, double v) {
    double s = __dadd_rn(hi, v);
    double bp = __dsub_rn(s, hi);
    double e = __dadd_rn(__dsub_rn(hi, __dsub_rn(s, bp)), __dsub_rn(v, bp));
    hi = s;
    lo = __dadd_rn(lo, e);
}

// combine two partial results of a reduction mode
__device__ __forceinline__ void red_combine(int mode, double &hi, double &lo, double h2, double l2) {
    if (mode == 1) {
        hi = fmax(hi, h2);
    } else {
        dd_add(hi, lo, h2);
        lo = __dadd_rn(lo, l2);
    }
}

// Fixed shared-memory tree over the block's threads; thread 0 ends with the block's result.
__device__ __forceinline__ void red_block(int mode, double &hi, double &lo) {
    __shared__ double sh[RED_THREADS];
    __shared__ double sl[RED_THREADS];
    sh[threadIdx.x] = hi;
    sl[threadIdx.x] = lo;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            double h = sh[threadIdx.x], l = sl[threadIdx.x];
            red_combine(mode, h, l, sh[threadIdx.x + o], sl[threadIdx.x + o]);
            sh[threadIdx.x] = h;
            sl[threadIdx.x] = l;
        }
        __syncthreads();
    }
    hi = sh[0];
    lo = sl[0];
}

// Task t = blockIdx.y of the table (x, y, n, mode): MODE_DOT sum x_i y_i, MODE_MAXABS
// max |x_i|, MODE_NONFINITE the number of non-finite x_i. Partial results (hi, lo) of block b
// at part[2 (t RED_BLOCKS + b)].
extern "C" __global__ void __launch_bounds__(RED_THREADS) scf_reduce_partial(
    const unsigned long long *tab, double *part) {
    const unsigned long long *task = tab + 4 * blockIdx.y;
    const double *x = (const double *)task[0];
    const double *y = (const double *)task[1];
    const long long n = (long long)task[2];
    const int mode = (int)task[3];
    double hi = 0.0, lo = 0.0;
    const long long stride = (long long)RED_BLOCKS * RED_THREADS;
    for (long long i = (long long)blockIdx.x * RED_THREADS + threadIdx.x; i < n; i += stride) {
        if (mode == 0) {
            dd_add(hi, lo, __dmul_rn(x[i], y[i]));
        } else if (mode == 1) {
            hi = fmax(hi, fabs(x[i]));
        } else {
            if (!isfinite(x[i])) hi = __dadd_rn(hi, 1.0);
        }
    }
    red_block(mode, hi, lo);
    if (threadIdx.x == 0) {
        long long p = 2 * ((long long)blockIdx.y * RED_BLOCKS + blockIdx.x);
        part[p] = hi;
        part[p + 1] = lo;
    }
}

// One block (RED_BLOCKS threads) per task t = blockIdx.x: out[t] = the task's result, from its
// RED_BLOCKS partial results in a fixed tree.
extern "C" __global__ void __launch_bounds__(RED_BLOCKS) scf_reduce_finish(
    const unsigned long long *tab, const double *part, double *out) {
    const unsigned long long *task = tab + 4 * blockIdx.x;
    const int mode = (int)task[3];
    long long p = 2 * ((long long)blockIdx.x * RED_BLOCKS + threadIdx.x);
    double hi = part[p], lo = part[p + 1];
    __shared__ double sh[RED_BLOCKS];
    __shared__ double sl[RED_BLOCKS];
    sh[threadIdx.x] = hi;
    sl[threadIdx.x] = lo;
    __syncthreads();
    for (int o = RED_BLOCKS / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            double h = sh[threadIdx.x], l = sl[threadIdx.x];
            red_combine(mode, h, l, sh[threadIdx.x + o], sl[threadIdx.x + o]);
            sh[threadIdx.x] = h;
            sl[threadIdx.x] = l;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = mode == 1 ? sh[0] : __dadd_rn(sh[0], sl[0]);
}

// out = a + beta b (beta = +-1 gives a +- b with one rounding)
extern "C" __global__ void scf_add(long long n, const double *a, const double *b, double beta, double *out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __dadd_rn(a[i], __dmul_rn(beta, b[i]));
}

// Fock matrices of nsp spins back to back (ntot = nsp n2 elements):
// F_s = ((h + J) - K_s) + (Vxc_s + Venv), as the CPU's h + J - K_x,s + V_s with V_s = Vxc_s + Venv.
// h, J and Venv are one n2 matrix each; k, vxc and venv may be null (zero).
extern "C" __global__ void scf_fock(long long ntot, long long n2, const double *h, const double *j,
                                    const double *k, const double *vxc, const double *venv, double *f) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ntot) {
        long long e = i % n2;
        double t = __dadd_rn(h[e], j[e]);
        if (k) t = __dsub_rn(t, k[i]);
        if (vxc && venv) t = __dadd_rn(t, __dadd_rn(vxc[i], venv[e]));
        else if (vxc) t = __dadd_rn(t, vxc[i]);
        else if (venv) t = __dadd_rn(t, venv[e]);
        f[i] = t;
    }
}

// E = A - A^T (n x n, column-major)
extern "C" __global__ void scf_antisym(long long n, const double *a, double *e) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        long long i = idx % n, j = idx / n;
        e[idx] = __dsub_rn(a[idx], a[i * n + j]);
    }
}

// F_s + shift (S - S D_s S) for nsp spins back to back (S one n2 matrix)
extern "C" __global__ void scf_level_shift(long long ntot, long long n2, const double *f, const double *s,
                                           const double *sds, double shift, double *out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ntot) out[i] = __dadd_rn(f[i], __dmul_rn(__dsub_rn(s[i % n2], sds[i]), shift));
}

// out = sum_k c_k F_k in the order k = 0, 1, ... (tab: m pointers, then m coefficients as
// bits), as the CPU's f += F_k c_k
extern "C" __global__ void scf_combine(long long n, const unsigned long long *tab, int m, double *out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double acc = 0.0;
        for (int k = 0; k < m; k++) {
            const double *f = (const double *)tab[k];
            double c = __longlong_as_double((long long)tab[m + k]);
            acc = __dadd_rn(acc, __dmul_rn(f[i], c));
        }
        out[i] = acc;
    }
}

// X[:, c] = U[:, first + c] / sqrt(w[first + c]) (n rows, ncol columns)
extern "C" __global__ void scf_orth_columns(long long n, long long ncol, long long first, const double *u,
                                            const double *w, double *x) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * ncol) {
        long long i = idx % n, c = idx / n;
        x[idx] = u[(first + c) * n + i] / sqrt(w[first + c]);
    }
}

// X[:, c] = U[:, first + c] sqrt(sign w[first + c]) (n rows, ncol columns): the factors
// P P^T - N N^T of a symmetric matrix, as linus's factor_symmetric
extern "C" __global__ void scf_factor_columns(long long n, long long ncol, long long first, const double *u,
                                              const double *w, double sign, double *x) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * ncol) {
        long long i = idx % n, c = idx / n;
        x[idx] = __dmul_rn(u[(first + c) * n + i], sqrt(__dmul_rn(sign, w[first + c])));
    }
}
