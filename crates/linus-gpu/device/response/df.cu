
// linus-gpu response J/K helpers (src/response/df.rs).

// The block's Y in (i, q) order from the stored Q-major Yp:
// yb[(i nq + q) n + m] = yp[((q0 + q) o + i) n + m], i < o, q < nq.
extern "C" __global__ void resp_y_block(long long n, int o, int nq, long long q0,
    const double *__restrict__ yp, double *__restrict__ yb)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * o * nq) return;
    const long long m = t % n, r = t / n;
    const long long q = r % nq, i = r / nq;
    yb[t] = yp[((q0 + q) * o + i) * n + m];
}

// y[t] = (float)x[t] (single-precision exchange)
extern "C" __global__ void resp_to_f32(long long len, const double *__restrict__ x,
    float *__restrict__ y)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < len) y[t] = (float)x[t];
}

// One chunk of the stored exchange operator (src/response/df.rs, `OpChunk`), in place: the
// chunk p (rows (m, jj), m < n, jj < bj, of its panel; columns (l, i), i0 <= i < i0 + ni;
// leading dimension ld) holds G[(m, jj), (l, i)] = sum_Q Y_Q[m, j] Y_Q[l, i] from its GEMM;
// every n x n block (jj, i) is transposed and gets the pair part, P[(m, jj), (l, i)] <-
// G[(l, jj), (m, i)] + npk[map[m n + l] + nx (i + iend jj)] (npk: the panel's, i < iend; map
// -1: a dropped pair, 0). A thread block (32, 8) moves the pair of 32 x 32 tiles (ta, tb),
// (tb, ta) of one n x n block (ta >= tb) through shared memory: grid (tiles, tiles, bj ni).
// Every element has one writer.
extern "C" __global__ void __launch_bounds__(256) resp_op_fill(long long n, int iend, int i0,
    int ni, long long ld, double *__restrict__ p, const double *__restrict__ npk, long long nx,
    const int *__restrict__ map)
{
    __shared__ double t1[32][33];
    __shared__ double t2[32][33];
    const int ta = blockIdx.x, tb = blockIdx.y;
    if (ta < tb) return;
    const long long ii = blockIdx.z % ni, jj = blockIdx.z / ni;
    const long long i = i0 + ii;
    double *P = p + n * jj + n * ii * ld;
    const double *N = npk + nx * (i + (long long)iend * jj);
    const int tx = threadIdx.x;
    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        const long long m = ta * 32 + tx, l = tb * 32 + r;
        t1[r][tx] = (m < n && l < n) ? P[m + l * ld] : 0.0;
        const long long l2 = tb * 32 + tx, m2 = ta * 32 + r;
        t2[r][tx] = (l2 < n && m2 < n) ? P[l2 + m2 * ld] : 0.0;
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        const long long m = ta * 32 + tx, l = tb * 32 + r;
        if (m < n && l < n) {
            const int x = map[m * n + l];
            P[m + l * ld] = t2[tx][r] + (x >= 0 ? N[x] : 0.0);
        }
        const long long l2 = tb * 32 + tx, m2 = ta * 32 + r;
        if (ta != tb && l2 < n && m2 < n) {
            const int x = map[l2 * n + m2];
            P[l2 + m2 * ld] = t1[tx][r] + (x >= 0 ? N[x] : 0.0);
        }
    }
}

// y[t] += (double)x[t] (a block's single-precision exchange into the FP64 result)
extern "C" __global__ void resp_add_f32(long long len, const float *__restrict__ x,
    double *__restrict__ y)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < len) y[t] += (double)x[t];
}

// out[r + c ldo] += alpha (double)x[r + c ldx], r < rows, c < cols (an FP32 operator chunk's
// product into the FP64 result)
extern "C" __global__ void resp_acc_f32(long long rows, long long cols, double alpha,
    const float *__restrict__ x, long long ldx, double *__restrict__ out, long long ldo)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * cols) return;
    const long long r = t % rows, c = t / rows;
    out[r + c * ldo] += alpha * (double)x[r + c * ldx];
}

// One FP32 operator chunk from its FP64 sums (out of place): p[(m, jj), (l, i)] =
// (float)(g[(l, jj), (m, i)] + npk[map[m n + l] + nx (i + iend jj)]) for the chunk's columns
// i0 <= i < i0 + ni, g and p with leading dimension ld (both chunk-relative). A thread block
// (32, 8) writes one 32 x 32 tile of one n x n block, read transposed through shared memory:
// grid (tiles, tiles, bj ni).
extern "C" __global__ void __launch_bounds__(256) resp_op_fill_mixed(long long n, int iend,
    int i0, int ni, long long ld, const double *__restrict__ g, float *__restrict__ p,
    const double *__restrict__ npk, long long nx, const int *__restrict__ map)
{
    __shared__ double t[32][33];
    const int ta = blockIdx.x, tb = blockIdx.y;
    const long long ii = blockIdx.z % ni, jj = blockIdx.z / ni;
    const long long i = i0 + ii;
    const double *G = g + n * jj + n * ii * ld;
    float *P = p + n * jj + n * ii * ld;
    const double *N = npk + nx * (i + (long long)iend * jj);
    const int tx = threadIdx.x;
    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        const long long l = tb * 32 + tx, m = ta * 32 + r;
        t[r][tx] = (l < n && m < n) ? G[l + m * ld] : 0.0;
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        const long long m = ta * 32 + tx, l = tb * 32 + r;
        if (m < n && l < n) {
            const int x = map[m * n + l];
            P[m + l * ld] = (float)(t[tx][r] + (x >= 0 ? N[x] : 0.0));
        }
    }
}
