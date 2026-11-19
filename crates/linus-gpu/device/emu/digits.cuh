
// Ozaki slicing (src/emu/mod.rs): the scale exponent of a vector with largest magnitude m,
// m <= 127/128 2^e (0 for a zero vector).
__device__ __forceinline__ int emu_exponent(double m)
{
    int ex = 0;
    if (m > 0.0) {
        const double f = frexp(m, &ex);
        if (f > 0.9921875) ex += 1;
    }
    return ex;
}

__device__ __forceinline__ void emu_cut(long long i, int s, int desc, signed char *o,
    long long kp);

// The s digits of v in a vector of exponent e: v 2^(8s-1-e) rounded to the integer I (half to
// even), cut into two's-complement bytes with borrows (low digits in [-128, 127], the leading
// one in [-127, 127]); digit p goes to o[pos(p) kp], pos = p (ascending) or s - 1 - p (desc).
// Integer operations on the bits of v only (FP64 is slow on some cards): v = +-m 2^(b-1075)
// with m = 2^52 + fraction (b the biased exponent; subnormals b = 1 without the hidden bit).
__device__ __forceinline__ void emu_digits(double v, int e, int s, int desc,
    signed char *o, long long kp)
{
    const long long bits = __double_as_longlong(v);
    long long m = bits & 0xfffffffffffffLL;
    int b = (int)((bits >> 52) & 0x7ff);
    if (b != 0) m |= 0x10000000000000LL;
    else b = 1;
    const int sh = b - 1075 + 8 * s - 1 - e;
    long long i = 0;
    if (sh >= 0) {
        i = m << sh;
    } else if (sh > -63) {
        const int r = -sh;
        const long long q = m >> r;
        const long long rem = m - (q << r);
        const long long half = 1LL << (r - 1);
        i = q + ((rem > half || (rem == half && (q & 1))) ? 1 : 0);
    }
    if (bits < 0) i = -i;
    emu_cut(i, s, desc, o, kp);
}

// v 2^(8s-1-e) rounded to an integer (half to even), as emu_digits does before cutting.
__device__ __forceinline__ long long emu_fixed(double v, int e, int s)
{
    const long long bits = __double_as_longlong(v);
    long long m = bits & 0xfffffffffffffLL;
    int b = (int)((bits >> 52) & 0x7ff);
    if (b != 0) m |= 0x10000000000000LL;
    else b = 1;
    const int sh = b - 1075 + 8 * s - 1 - e;
    long long i = 0;
    if (sh >= 0) {
        i = m << sh;
    } else if (sh > -63) {
        const int r = -sh;
        const long long q = m >> r;
        const long long rem = m - (q << r);
        const long long half = 1LL << (r - 1);
        i = q + ((rem > half || (rem == half && (q & 1))) ? 1 : 0);
    }
    return bits < 0 ? -i : i;
}

// The digit p (0: leading) of the fixed-point integer i in s digits.
__device__ __forceinline__ int emu_digit(long long i, int s, int p)
{
    for (int k = s - 1; k > p; --k) {
        int d = (int)(i & 0xff);
        if (d > 127) d -= 256;
        i = (i - d) >> 8;
    }
    if (p == 0) return (int)i;
    int d = (int)(i & 0xff);
    return d > 127 ? d - 256 : d;
}

// Cuts the fixed-point integer i into its s digits at o[pos(p) kp].
__device__ __forceinline__ void emu_cut(long long i, int s, int desc, signed char *o,
    long long kp)
{
    for (int p = s - 1; p > 0; --p) {
        int d = (int)(i & 0xff);
        if (d > 127) d -= 256;
        i = (i - d) >> 8;
        o[(long long)(desc ? s - 1 - p : p) * kp] = (signed char)d;
    }
    o[(long long)(desc ? s - 1 : 0) * kp] = (signed char)i;
}
