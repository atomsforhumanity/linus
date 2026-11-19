// Deterministic fixed-point accumulation on integer atomics (linus-gpu, src/detsum.rs).
// Needs portable.h (the host pastes it in front of every kernel).
//
// The split is linus's CPU accumulator's, operation for operation: a value v (clamped to
// +-2^32, NaN to -2^32) is h 2^-30 + r with h = trunc(v 2^30), and l = trunc(r 2^82). The CPU
// keeps two words per element (h and l) and carries l into h every 1024 additions; on the
// device additions arrive in any order, so l is split once more, l = m 2^26 + q with
// q in [0, 2^26), and an element holds three words H = sum h, M = sum m, Q = sum q, added with
// 64-bit integer atomics (exact and associative). The element's value is
// (H 2^52 + M 2^26 + Q) 2^-82, the same integer the CPU's (hi 2^52 + lo) holds, so the same
// bits. M and Q take 2^37 additions per element before they could overflow; H has the CPU's
// range.
//
// Invalid values (NaN, infinite, or beyond 2^32 in magnitude) poison the accumulator, as the
// CPU's records them: one flag word per accumulator, after its elements, is set (det_poison),
// and every element then reads NaN (det_value on the device, DetAcc::to_f64 on the host), so a
// bad input shows in the result instead of being clamped into a finite number. Finite inputs
// give the same bits as without the flag.
//
// Layout: structure of arrays, word w of element i at acc[w * n + i], the flag at acc[3 n].

#ifndef LINUS_DETSUM_H
#define LINUS_DETSUM_H

// The words of one addition of v; true if v is invalid (the words are then those of the
// clamped value, as the CPU's, and the caller poisons its accumulator).
__device__ __forceinline__ bool det_split(double v, long long &h, long long &m, long long &q) {
    const double s_hi = 1073741824.0;                           // 2^30
    const double inv_s_hi = 9.313225746154785e-10;              // 2^-30
    const double s_lo = 4835703278458516698824704.0;            // 2^82
    const double clamp = 4294967296.0;                          // 2^62 * 2^-30
    const bool bad = !(fabs(v) <= clamp);                       // NaN compares false
    v = fmin(fmax(v, -clamp), clamp);                           // fmax(NaN, x) = x, as f64::max
    h = (long long)(v * s_hi);                                  // truncation, as to_int_unchecked
    double r = v - (double)h * inv_s_hi;                        // exact: (double)h * 2^-30 is exact
    long long l = (long long)(r * s_lo);
    m = l >> 26;                                                // arithmetic shift: floor
    q = l & ((1LL << 26) - 1);
    return bad;
}

// Marks the accumulator of n elements invalid (its flag word acc[3 n]): every value reads NaN.
__device__ __forceinline__ void det_poison(unsigned long long *acc, long long n) {
    atomicOr(acc + 3 * n, 1ull);
}

// acc[i] += v
__device__ __forceinline__ void det_add(unsigned long long *acc, long long n, long long i, double v) {
    long long h, m, q;
    if (det_split(v, h, m, q)) det_poison(acc, n);
    atomicAdd(acc + i, (unsigned long long)h);
    atomicAdd(acc + n + i, (unsigned long long)m);
    atomicAdd(acc + 2 * n + i, (unsigned long long)q);
}

// Block-wide sum of one value per thread into acc[i]: the three words are summed exactly
// with shuffles within each warp and shared memory across warps, then added with one atomic
// per word per block. Every thread of the block must call it; blockDim.x a multiple of
// LINUS_WARP_SIZE, at most 1024.
__device__ __forceinline__ void det_add_block(unsigned long long *acc, long long n, long long i, double v) {
    __shared__ long long part[3][1024 / LINUS_WARP_SIZE];
    long long h, m, q;
    const bool bad = det_split(v, h, m, q);
    for (int o = LINUS_WARP_SIZE / 2; o > 0; o >>= 1) {
        h += linus_shfl_down(h, o);
        m += linus_shfl_down(m, o);
        q += linus_shfl_down(q, o);
    }
    const int lane = threadIdx.x % LINUS_WARP_SIZE, warp = threadIdx.x / LINUS_WARP_SIZE;
    if (lane == 0) {
        part[0][warp] = h;
        part[1][warp] = m;
        part[2][warp] = q;
    }
    // every thread reaches this barrier; it also tells whether any value was invalid
    if (__syncthreads_or(bad) && threadIdx.x == 0) det_poison(acc, n);
    if (warp == 0) {
        // at most 1024 / LINUS_WARP_SIZE <= LINUS_WARP_SIZE partial sums
        const int nw = blockDim.x / LINUS_WARP_SIZE;
        h = lane < nw ? part[0][lane] : 0;
        m = lane < nw ? part[1][lane] : 0;
        q = lane < nw ? part[2][lane] : 0;
        for (int o = LINUS_WARP_SIZE / 2; o > 0; o >>= 1) {
            h += linus_shfl_down(h, o);
            m += linus_shfl_down(m, o);
            q += linus_shfl_down(q, o);
        }
        if (lane == 0) {
            atomicAdd(acc + i, (unsigned long long)h);
            atomicAdd(acc + n + i, (unsigned long long)m);
            atomicAdd(acc + 2 * n + i, (unsigned long long)q);
        }
    }
    __syncthreads();
}

// The value of element i of an accumulator of n elements, on the device: the words carried
// into a normalized form, then one rounding (the host's detsum::value); NaN once the
// accumulator is poisoned.
__device__ __forceinline__ double det_value(const unsigned long long *acc, long long n, long long i) {
    if (acc[3 * n] != 0ull) return nan("");
    long long H = (long long)acc[i], M = (long long)acc[n + i], Q = (long long)acc[2 * n + i];
    const long long c1 = Q >> 26;
    Q -= c1 << 26;
    M += c1;
    const long long c2 = M >> 26;
    M -= c2 << 26;
    H += c2;
    return (double)H * 9.313225746154785e-10 + (double)((M << 26) + Q) * 2.0679515313825692e-25;
}

#endif
