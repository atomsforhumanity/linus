//! Small dense kernels used by the derivative-integral passes (W-matrix times E/H rows), with
//! register blocking and runtime-dispatched AVX-512 / AVX2 variants, so no global
//! target-cpu flags are needed.

/// Column chunk widths: 8, 4 and 1 (scalar tail).
macro_rules! chunks {
    ($n:expr, |$c:ident, $w:ident| $body:block) => {{
        let mut $c = 0usize;
        while $c + 8 <= $n {
            const $w: usize = 8;
            $body
            $c += 8;
        }
        if $c + 4 <= $n {
            const $w: usize = 4;
            $body
            $c += 4;
        }
        while $c < $n {
            const $w: usize = 1;
            $body
            $c += 1;
        }
    }};
}

/// acc[i][c] += sum_j w[i * ldw + j] * src[j * n + c]. Rows of W are processed four at a
/// time (independent accumulation chains, each source chunk loaded once) with the
/// accumulator chunk kept in registers over j.
#[inline(always)]
fn wmul_acc_body(
    w: &[f64],
    ldw: usize,
    ni: usize,
    nj: usize,
    src: &[f64],
    n: usize,
    acc: &mut [f64],
) {
    assert!(
        src.len() >= nj * n && acc.len() >= ni * n && (ni == 0 || w.len() >= (ni - 1) * ldw + nj)
    );
    let mut i = 0;
    while i + 4 <= ni {
        let (w0, w1, w2, w3) = (
            &w[i * ldw..i * ldw + nj],
            &w[(i + 1) * ldw..(i + 1) * ldw + nj],
            &w[(i + 2) * ldw..(i + 2) * ldw + nj],
            &w[(i + 3) * ldw..(i + 3) * ldw + nj],
        );
        let (a0, rest) = acc[i * n..(i + 4) * n].split_at_mut(n);
        let (a1, rest) = rest.split_at_mut(n);
        let (a2, a3) = rest.split_at_mut(n);
        chunks!(n, |c, W| {
            let mut r0: [f64; W] = a0[c..c + W].try_into().unwrap();
            let mut r1: [f64; W] = a1[c..c + W].try_into().unwrap();
            let mut r2: [f64; W] = a2[c..c + W].try_into().unwrap();
            let mut r3: [f64; W] = a3[c..c + W].try_into().unwrap();
            for j in 0..nj {
                let sv: &[f64; W] = src[j * n + c..j * n + c + W].try_into().unwrap();
                let (x0, x1, x2, x3) = (w0[j], w1[j], w2[j], w3[j]);
                for t in 0..W {
                    r0[t] += x0 * sv[t];
                    r1[t] += x1 * sv[t];
                    r2[t] += x2 * sv[t];
                    r3[t] += x3 * sv[t];
                }
            }
            a0[c..c + W].copy_from_slice(&r0);
            a1[c..c + W].copy_from_slice(&r1);
            a2[c..c + W].copy_from_slice(&r2);
            a3[c..c + W].copy_from_slice(&r3);
        });
        i += 4;
    }
    while i < ni {
        let wr = &w[i * ldw..i * ldw + nj];
        let a = &mut acc[i * n..(i + 1) * n];
        chunks!(n, |c, W| {
            let mut r: [f64; W] = a[c..c + W].try_into().unwrap();
            for (j, &wv) in wr.iter().enumerate() {
                let sv: &[f64; W] = src[j * n + c..j * n + c + W].try_into().unwrap();
                for t in 0..W {
                    r[t] += wv * sv[t];
                }
            }
            a[c..c + W].copy_from_slice(&r);
        });
        i += 1;
    }
}

/// acc[j][c] += sum_i w[i * ldw + j] * src[i * n + c] (transposed W), four output rows at a
/// time.
#[inline(always)]
fn wmul_t_acc_body(
    w: &[f64],
    ldw: usize,
    ni: usize,
    nj: usize,
    src: &[f64],
    n: usize,
    acc: &mut [f64],
) {
    assert!(
        src.len() >= ni * n && acc.len() >= nj * n && (ni == 0 || w.len() >= (ni - 1) * ldw + nj)
    );
    let mut j = 0;
    while j + 4 <= nj {
        let (a0, rest) = acc[j * n..(j + 4) * n].split_at_mut(n);
        let (a1, rest) = rest.split_at_mut(n);
        let (a2, a3) = rest.split_at_mut(n);
        chunks!(n, |c, W| {
            let mut r0: [f64; W] = a0[c..c + W].try_into().unwrap();
            let mut r1: [f64; W] = a1[c..c + W].try_into().unwrap();
            let mut r2: [f64; W] = a2[c..c + W].try_into().unwrap();
            let mut r3: [f64; W] = a3[c..c + W].try_into().unwrap();
            for i in 0..ni {
                let sv: &[f64; W] = src[i * n + c..i * n + c + W].try_into().unwrap();
                let wr: &[f64; 4] = w[i * ldw + j..i * ldw + j + 4].try_into().unwrap();
                for t in 0..W {
                    r0[t] += wr[0] * sv[t];
                    r1[t] += wr[1] * sv[t];
                    r2[t] += wr[2] * sv[t];
                    r3[t] += wr[3] * sv[t];
                }
            }
            a0[c..c + W].copy_from_slice(&r0);
            a1[c..c + W].copy_from_slice(&r1);
            a2[c..c + W].copy_from_slice(&r2);
            a3[c..c + W].copy_from_slice(&r3);
        });
        j += 4;
    }
    while j < nj {
        let a = &mut acc[j * n..(j + 1) * n];
        chunks!(n, |c, W| {
            let mut r: [f64; W] = a[c..c + W].try_into().unwrap();
            for i in 0..ni {
                let wv = w[i * ldw + j];
                let sv: &[f64; W] = src[i * n + c..i * n + c + W].try_into().unwrap();
                for t in 0..W {
                    r[t] += wv * sv[t];
                }
            }
            a[c..c + W].copy_from_slice(&r);
        });
        j += 1;
    }
}

linus_num::multiversion!(
    /// acc[i][:] += sum_j w[i * ldw + j] * src[j][:] for i < ni, j < nj (rows of length n).
    fn wmul_acc(w: &[f64], ldw: usize, ni: usize, nj: usize, src: &[f64], n: usize, acc: &mut [f64]) -> () = wmul_acc_body);

linus_num::multiversion!(
    /// acc[j][:] += sum_i w[i * ldw + j] * src[i][:] for i < ni, j < nj (transposed W).
    fn wmul_t_acc(w: &[f64], ldw: usize, ni: usize, nj: usize, src: &[f64], n: usize, acc: &mut [f64]) -> () = wmul_t_acc_body);

/// g[k] = sum_{r < nr} sum_c e[r * n + c] * y[shift[r][k] * n + c] for k = 0, 1, 2.
#[inline(always)]
fn dot3_shift_body(e: &[f64], y: &[f64], shift: &[[u32; 3]], nr: usize, n: usize) -> [f64; 3] {
    assert!(e.len() >= nr * n && shift.len() >= nr);
    let mut acc = [[0.0f64; 8]; 3];
    let mut tail = [0.0f64; 3];
    for r in 0..nr {
        let er = &e[r * n..(r + 1) * n];
        for k in 0..3 {
            let rr = shift[r][k] as usize;
            let yr = &y[rr * n..(rr + 1) * n];
            let mut c = 0;
            while c + 8 <= n {
                let a: &[f64; 8] = er[c..c + 8].try_into().unwrap();
                let b: &[f64; 8] = yr[c..c + 8].try_into().unwrap();
                for t in 0..8 {
                    acc[k][t] += a[t] * b[t];
                }
                c += 8;
            }
            while c < n {
                tail[k] += er[c] * yr[c];
                c += 1;
            }
        }
    }
    let mut g = tail;
    for k in 0..3 {
        g[k] += acc[k].iter().sum::<f64>();
    }
    g
}

/// g[k] = sum_x e[k * len + x] * y[x] (three long contiguous dot products).
#[inline(always)]
#[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
fn dot3_long_body(e: &[f64], y: &[f64], len: usize) -> [f64; 3] {
    assert!(e.len() >= 3 * len && y.len() >= len);
    let mut g = [0.0; 3];
    let y = &y[..len];
    for k in 0..3 {
        let ek = &e[k * len..(k + 1) * len];
        let mut acc = [[0.0f64; 8]; 2];
        let mut c = 0;
        while c + 16 <= len {
            let a0: &[f64; 8] = ek[c..c + 8].try_into().unwrap();
            let b0: &[f64; 8] = y[c..c + 8].try_into().unwrap();
            let a1: &[f64; 8] = ek[c + 8..c + 16].try_into().unwrap();
            let b1: &[f64; 8] = y[c + 8..c + 16].try_into().unwrap();
            for t in 0..8 {
                acc[0][t] += a0[t] * b0[t];
                acc[1][t] += a1[t] * b1[t];
            }
            c += 16;
        }
        while c + 8 <= len {
            let a0: &[f64; 8] = ek[c..c + 8].try_into().unwrap();
            let b0: &[f64; 8] = y[c..c + 8].try_into().unwrap();
            for t in 0..8 {
                acc[0][t] += a0[t] * b0[t];
            }
            c += 8;
        }
        let mut s = 0.0;
        while c < len {
            s += ek[c] * y[c];
            c += 1;
        }
        for t in 0..8 {
            s += acc[0][t] + acc[1][t];
        }
        g[k] = s;
    }
    g
}

linus_num::multiversion!(
    /// Three long dot products against a component-major matrix: see [`dot3_long_body`].
    fn dot3_long(e: &[f64], y: &[f64], len: usize) -> [f64; 3] = dot3_long_body);

linus_num::multiversion!(
    /// Three dot products with Hermite-shifted rows of y: see [`dot3_shift_body`].
    fn dot3_shift(e: &[f64], y: &[f64], shift: &[[u32; 3]], nr: usize, n: usize) -> [f64; 3] = dot3_shift_body);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_match_naive() {
        let mut seed = 1u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        for &(ni, nj, ldw, n) in &[
            (1, 1, 1, 1),
            (4, 1, 4, 1),
            (10, 7, 12, 9),
            (20, 10, 20, 25),
            (3, 5, 6, 13),
            (7, 3, 3, 4),
        ] {
            let w: Vec<f64> = (0..ni * ldw).map(|_| rnd()).collect();
            let src: Vec<f64> = (0..nj.max(ni) * n).map(|_| rnd()).collect();
            let mut a1: Vec<f64> = (0..ni * n).map(|_| rnd()).collect();
            let mut a2 = a1.clone();
            wmul_acc(&w, ldw, ni, nj, &src, n, &mut a1);
            for i in 0..ni {
                for j in 0..nj {
                    for c in 0..n {
                        a2[i * n + c] += w[i * ldw + j] * src[j * n + c];
                    }
                }
            }
            assert!(a1.iter().zip(&a2).all(|(x, y)| (x - y).abs() < 1e-13));
            let mut b1: Vec<f64> = (0..nj * n).map(|_| rnd()).collect();
            let mut b2 = b1.clone();
            wmul_t_acc(&w, ldw, ni, nj, &src, n, &mut b1);
            for i in 0..ni {
                for j in 0..nj {
                    for c in 0..n {
                        b2[j * n + c] += w[i * ldw + j] * src[i * n + c];
                    }
                }
            }
            assert!(b1.iter().zip(&b2).all(|(x, y)| (x - y).abs() < 1e-13));
        }
    }
}
