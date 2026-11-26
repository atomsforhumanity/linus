// linus-gpu grid partition (src/xc/partition.rs): the weights of an unscreened partition (Becke,
// LKO, SSF without screening) at every point, linus-grid's Partitioner::weights_block operation
// for operation (compiled without contraction): one thread per point, its atom distances d and
// cell functions P in the scratch arrays d[i ld + t], p[i ld + t] (t the point within the launch).
// scheme: 0 Becke, 1 LKO, 2 SSF (half = its a).

__device__ __forceinline__ double part_ssf_z(double m)
{
    const double m2 = m * m;
    return (1.0 / 16.0) * (m * (35.0 + m2 * (-35.0 + m2 * (21.0 - 5.0 * m2))));
}

extern "C" __global__ void grid_partition(long long np, long long k0, int n, int scheme,
    int adjust, double half, const double *__restrict__ x, const double *__restrict__ y,
    const double *__restrict__ z, const unsigned int *__restrict__ owner,
    const double *__restrict__ vol, const double *__restrict__ axyz,
    const double *__restrict__ inv_r, const double *__restrict__ a, double *d, double *p,
    long long ld, double *__restrict__ out)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= np) return;
    const long long k = k0 + t;
    const double px = x[k], py = y[k], pz = z[k];
    for (int i = 0; i < n; ++i) {
        const double dx = px - axyz[3 * i];
        const double dy = py - axyz[3 * i + 1];
        const double dz = pz - axyz[3 * i + 2];
        d[i * ld + t] = sqrt(dx * dx + dy * dy + dz * dz);
        p[i * ld + t] = 1.0;
    }
    for (int i = 1; i < n; ++i) {
        const double di = d[i * ld + t];
        double pi = p[i * ld + t];
        for (int j = 0; j < i; ++j) {
            const double dj = d[j * ld + t];
            const double fac = inv_r[i * n + j];
            const double aij = adjust ? a[i * n + j] : 0.0;
            if (scheme == 2) {
                double g = fac * (di - dj);
                if (adjust) g += aij * (1.0 - g * g);
                double zz = part_ssf_z(g / half);
                if (g <= -half) zz = -1.0;
                if (g >= half) zz = 1.0;
                pi *= 0.5 * (1.0 - zz);
                p[j * ld + t] *= 0.5 * (1.0 + zz);
            } else {
                double g = (di - dj) * fac;
                if (scheme == 1) g = fmin(fmax(g, -1.0), 1.0);
                if (adjust) g += aij * (1.0 - g * g);
                const double s = linus_becke_s(g);
                pi *= 0.5 - s;
                p[j * ld + t] *= 0.5 + s;
            }
        }
        p[i * ld + t] = pi;
    }
    double zs = 0.0;
    for (int i = 0; i < n; ++i) zs += p[i * ld + t];
    out[k] = vol[k] * p[(long long)owner[k] * ld + t] * (1.0 / zs);
}

// The cell factor of atom b against atom c at a point (distances db, dc) as grid_partition's
// pair loop computes it for the pair (max(b, c), min(b, c)): b's factor 0.5 - s (SSF: 0.5 (1 - z))
// when b is the larger index, 0.5 + s (0.5 (1 + z)) when it is the smaller. ms and as are the
// pair tables mirrored from their lower triangles: ms[b n + c] = inv_r[max n + min], the same
// for a (zero-size when there is no size adjustment).
__device__ __forceinline__ double part_factor(int b, int c, double db, double dc, int n,
    int scheme, int adjust, double half, const double *__restrict__ ms,
    const double *__restrict__ as)
{
    const bool hi = b > c;
    const double di = hi ? db : dc;
    const double dj = hi ? dc : db;
    const double fac = ms[(long long)b * n + c];
    const double aij = adjust ? as[(long long)b * n + c] : 0.0;
    if (scheme == 2) {
        double g = fac * (di - dj);
        if (adjust) g += aij * (1.0 - g * g);
        double zz = part_ssf_z(g / half);
        if (g <= -half) zz = -1.0;
        if (g >= half) zz = 1.0;
        return hi ? 0.5 * (1.0 - zz) : 0.5 * (1.0 + zz);
    }
    double g = (di - dj) * fac;
    if (scheme == 1) g = fmin(fmax(g, -1.0), 1.0);
    if (adjust) g += aij * (1.0 - g * g);
    const double s = linus_becke_s(g);
    return hi ? 0.5 - s : 0.5 + s;
}

// The weights of grid_partition with the atoms screened per point, in tiles of W points (one
// block per tile, W = LINUS_WARP_SIZE, lane p = point p of the tile throughout; the points in
// the order perm, spatially compact tiles). The cell function P_b of atom b is the product over
// every other atom c of its factor (each in [0, 1]), so its product over any subset bounds it
// from above. The warps share out the atoms: the distances dt[c W + p] (the block's scratch);
// each point's nearest atom m (the warps' minima, the first index on ties) and l, the product of
// m's factors (the warps' partial products, in any order: P_m to within a few ulps, and
// P_m <= sum_b P_b); the bound of every atom but the owner and m against the kn atoms nbr[m kn ..]
// nearest m (atoms whose bound falls below eps l dropped; kept atoms get bit p of mask[c]); the
// union U of the tile's kept atoms (index order); per atom u of U every point's P_u
// (grid_partition's product, every factor in index order, so the same bits) into pt[j W + p];
// then each point's sum over U in index order. U holds every atom kept for each of its points,
// so the terms left out are each below eps l of their point's sum; stats[0] += the atoms whose
// cell functions were computed, summed over the points (when stats is not null).
// Shared memory: n 64-bit masks, then per warp and point a double and an int (the minima, the
// partial products), per point P_owner, l, m and owner, the union's size, n shorts (U).
extern "C" __global__ void grid_partition_tiled(long long np, int n, int scheme, int adjust,
    double half, int kn, double eps, const unsigned int *__restrict__ perm,
    const double *__restrict__ x, const double *__restrict__ y, const double *__restrict__ z,
    const unsigned int *__restrict__ owner, const double *__restrict__ vol,
    const double *__restrict__ axyz, const double *__restrict__ ms,
    const double *__restrict__ as, const int *__restrict__ nbr, double *__restrict__ dsc,
    double *__restrict__ psc, double *__restrict__ out, unsigned long long *stats)
{
    extern __shared__ unsigned long long tile_smem[];
    const int W = LINUS_WARP_SIZE;
    const int lane = threadIdx.x % W;
    const int w = threadIdx.x / W;
    const int nw = blockDim.x / W;
    const unsigned long long below = (1ull << lane) - 1ull;
    unsigned long long *mask = tile_smem;
    double *wd = (double *)(mask + n);
    int *wi = (int *)(wd + nw * W);
    double *pown = (double *)(wi + nw * W + (nw * W) % 2);
    double *pl = pown + W;
    int *pm = (int *)(pl + W);
    int *pow_ = pm + W;
    int *ucnt = pow_ + W;
    unsigned short *ul = (unsigned short *)(ucnt + 1);
    double *dt = dsc + (long long)blockIdx.x * n * W;
    double *pt = psc + (long long)blockIdx.x * n * W;
    unsigned long long kept = 0;
    for (long long t0 = (long long)blockIdx.x * W; t0 < np; t0 += (long long)gridDim.x * W) {
        const int npt = np - t0 < W ? (int)(np - t0) : W;
        // the point of this lane (the last one repeated past the tile's end)
        const long long k = perm[t0 + (lane < npt ? lane : npt - 1)];
        const double px = x[k], py = y[k], pz = z[k];
        // distances and the warps' nearest atoms
        {
            double best = 1e300;
            int bi = n;
            for (int c = w; c < n; c += nw) {
                const double dx = px - axyz[3 * c];
                const double dy = py - axyz[3 * c + 1];
                const double dz = pz - axyz[3 * c + 2];
                const double dc = sqrt(dx * dx + dy * dy + dz * dz);
                dt[(long long)c * W + lane] = dc;
                if (dc < best) {
                    best = dc;
                    bi = c;
                }
            }
            wd[w * W + lane] = best;
            wi[w * W + lane] = bi;
        }
        for (int c = threadIdx.x; c < n; c += blockDim.x) mask[c] = 0ull;
        __syncthreads();
        if (w == 0) {
            double best = wd[lane];
            int bi = wi[lane];
            for (int v = 1; v < nw; ++v) {
                const double b2 = wd[v * W + lane];
                const int i2 = wi[v * W + lane];
                if (b2 < best || (b2 == best && i2 < bi)) {
                    best = b2;
                    bi = i2;
                }
            }
            pm[lane] = bi;
            pow_[lane] = (int)owner[k];
        }
        __syncthreads();
        const int m = pm[lane], own = pow_[lane];
        const double dm = dt[(long long)m * W + lane];
        // the warps' partial products of m's factors
        {
            double l = 1.0;
            for (int c = w; c < n; c += nw)
                if (c != m)
                    l *= part_factor(m, c, dm, dt[(long long)c * W + lane], n, scheme, adjust,
                        half, ms, as);
            wd[w * W + lane] = l;
        }
        __syncthreads();
        if (w == 0) {
            double l = wd[lane];
            for (int v = 1; v < nw; ++v) l *= wd[v * W + lane];
            pl[lane] = l;
        }
        __syncthreads();
        const double thr = eps * pl[lane];
        // the bounds, the warps sharing out the atoms
        for (int b = w; b < n; b += nw) {
            bool keep = lane < npt;
            if (keep && b != own && b != m) {
                const double db = dt[(long long)b * W + lane];
                double ub = 1.0;
                for (int q = 0; q < kn; ++q) {
                    const int c = nbr[m * kn + q];
                    if (c == b) continue;
                    ub *= part_factor(b, c, db, dt[(long long)c * W + lane], n, scheme, adjust,
                        half, ms, as);
                    if (ub < thr) {
                        keep = false;
                        break;
                    }
                }
            }
            const unsigned long long bal = linus_ballot(keep);
            if (lane == 0 && bal) mask[b] = bal;
        }
        __syncthreads();
        // the union in index order
        if (w == 0) {
            int cnt = 0;
            for (int base = 0; base < n; base += W) {
                const int c = base + lane;
                const bool in = c < n && mask[c] != 0ull;
                const unsigned long long bal = linus_ballot(in);
                if (in) ul[cnt + __popcll(bal & below)] = (unsigned short)c;
                cnt += __popcll(bal);
            }
            if (lane == 0) *ucnt = cnt;
        }
        __syncthreads();
        const int cnt = *ucnt;
        // per atom of the union: every point's cell function
        const double *__restrict__ dl = dt + lane;
        for (int j = w; j < cnt; j += nw) {
            const int u = ul[j];
            const double du = dl[(long long)u * W];
            double pb = 1.0;
            for (int c = 0; c < n; ++c)
                if (c != u)
                    pb *= part_factor(u, c, du, dl[(long long)c * W], n, scheme, adjust, half,
                        ms, as);
            pt[(long long)j * W + lane] = pb;
            if (u == own) pown[lane] = pb;
        }
        __syncthreads();
        if (w == 0 && lane < npt) {
            double zs = 0.0;
            for (int j = 0; j < cnt; ++j) zs += pt[(long long)j * W + lane];
            out[k] = vol[k] * pown[lane] * (1.0 / zs);
        }
        if (threadIdx.x == 0) kept += (unsigned long long)cnt * npt;
        __syncthreads();
    }
    if (stats && threadIdx.x == 0 && kept) atomicAdd(stats, kept);
}
