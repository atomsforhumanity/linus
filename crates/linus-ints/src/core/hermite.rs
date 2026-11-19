//! McMurchie-Davidson Hermite machinery: index tables, Hermite expansion (E) coefficients,
//! Hermite Coulomb (R) integrals, and precomputed shell-pair data in the SHARK layout
//! (E-matrices already transformed to contracted spherical functions).

use crate::basis::{BasisSet, Shell};
use crate::core::boys;
use std::sync::OnceLock;

/// Number of Hermite functions Λ_tuv with t+u+v <= l.
#[inline]
pub const fn nherm(l: usize) -> usize {
    (l + 1) * (l + 2) * (l + 3) / 6
}

/// Position of (t,u,v) in the Hermite order: by total degree n, then t descending, then u
/// descending. Herm(l-1) is a prefix of Herm(l).
#[inline]
pub const fn herm_idx(t: usize, u: usize, v: usize) -> usize {
    let n = t + u + v;
    n * (n + 1) * (n + 2) / 6 + (n - t) * (n - t + 1) / 2 + (n - t - u)
}

pub const MAX_HERM_L: usize = 48;

/// (t,u,v) triples in Hermite order up to MAX_HERM_L.
pub fn herm_tuv() -> &'static [[u8; 3]] {
    static T: OnceLock<Vec<[u8; 3]>> = OnceLock::new();
    T.get_or_init(|| {
        let mut v = Vec::with_capacity(nherm(MAX_HERM_L));
        for n in 0..=MAX_HERM_L {
            for t in (0..=n).rev() {
                for u in (0..=n - t).rev() {
                    v.push([t as u8, u as u8, (n - t - u) as u8]);
                }
            }
        }
        for (i, tuv) in v.iter().enumerate() {
            debug_assert_eq!(
                herm_idx(tuv[0] as usize, tuv[1] as usize, tuv[2] as usize),
                i
            );
        }
        v
    })
}

const MAX_SUM_L: usize = 24;

/// For bra degree l1 and ket degree l2: table[r * nherm(l2) + s] = herm_idx(r + s), and the
/// ket sign (-1)^{t'+u'+v'} per s.
pub struct SumTable {
    pub idx: Vec<u32>,
    pub sign: Vec<f64>,
}

pub fn sum_table(l1: usize, l2: usize) -> &'static SumTable {
    static TABLES: OnceLock<Vec<OnceLock<SumTable>>> = OnceLock::new();
    let tabs = TABLES.get_or_init(|| {
        (0..(MAX_SUM_L + 1) * (MAX_SUM_L + 1))
            .map(|_| OnceLock::new())
            .collect()
    });
    assert!(l1 <= MAX_SUM_L && l2 <= MAX_SUM_L);
    tabs[l1 * (MAX_SUM_L + 1) + l2].get_or_init(|| {
        let tuv = herm_tuv();
        let (n1, n2) = (nherm(l1), nherm(l2));
        let mut idx = Vec::with_capacity(n1 * n2);
        for r in &tuv[..n1] {
            for s in &tuv[..n2] {
                idx.push(herm_idx(
                    (r[0] + s[0]) as usize,
                    (r[1] + s[1]) as usize,
                    (r[2] + s[2]) as usize,
                ) as u32);
            }
        }
        let sign = tuv[..n2]
            .iter()
            .map(|s| {
                if (s[0] + s[1] + s[2]) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        SumTable { idx, sign }
    })
}

/// 1D Hermite expansion coefficients E^{ij}_t, i <= la, j <= lb, t <= i + j, for a primitive
/// pair with exponents a, b and xab = A_x - B_x. Layout: out[(i * (lb+1) + j) * (la+lb+1) + t].
/// E^{00}_0 = exp(-mu xab^2) (the 1D Gaussian product prefactor).
pub fn hermite_e_1d(la: usize, lb: usize, a: f64, b: f64, xab: f64, out: &mut [f64]) {
    let p = a + b;
    let mu = a * b / p;
    let xpa = -b / p * xab;
    let xpb = a / p * xab;
    let oo2p = 0.5 / p;
    let nt = la + lb + 1;
    let stride_j = nt;
    let stride_i = (lb + 1) * nt;
    let len = (la + 1) * stride_i;
    out[..len].fill(0.0);
    out[0] = (-mu * xab * xab).exp();
    // i recursion at j = 0
    for i in 0..la {
        let (src, dst) = (i * stride_i, (i + 1) * stride_i);
        for t in 0..=i + 1 {
            let mut v = 0.0;
            if t > 0 {
                v += oo2p * out[src + t - 1];
            }
            if t <= i {
                v += xpa * out[src + t];
            }
            if t < i {
                v += (t + 1) as f64 * out[src + t + 1];
            }
            out[dst + t] = v;
        }
    }
    // j recursion
    for j in 0..lb {
        for i in 0..=la {
            let src = i * stride_i + j * stride_j;
            let dst = src + stride_j;
            for t in 0..=i + j + 1 {
                let mut v = 0.0;
                if t > 0 {
                    v += oo2p * out[src + t - 1];
                }
                if t <= i + j {
                    v += xpb * out[src + t];
                }
                if t < i + j {
                    v += (t + 1) as f64 * out[src + t + 1];
                }
                out[dst + t] = v;
            }
        }
    }
}

/// Hermite Coulomb integrals R_tuv(alpha, PQ) for t+u+v <= l, scaled by `pref`, in Hermite
/// order. `fm` holds F_0..F_l at alpha |PQ|^2. `work` needs 2 * nherm(l) entries.
#[inline]
pub fn r_tensor(
    l: usize,
    alpha: f64,
    pq: [f64; 3],
    fm: &[f64],
    pref: f64,
    out: &mut [f64],
    work: &mut [f64],
) {
    let tuv = herm_tuv();
    let m2a = -2.0 * alpha;
    // scaled R^{(n)}_{000} for n = 0..l
    let mut rn = [0.0f64; boys::MAX_M + 1];
    let mut f = pref;
    for n in 0..=l {
        rn[n] = f * fm[n];
        f *= m2a;
    }
    if l == 0 {
        out[0] = rn[0];
        return;
    }
    let nh = nherm(l);
    let (a, b) = work[..2 * nh].split_at_mut(nh);
    let mut prev: &mut [f64] = a;
    let mut cur: &mut [f64] = b;
    prev[0] = rn[l];
    let [x, y, z] = pq;
    for n in (0..l).rev() {
        let target: &mut [f64] = if n == 0 { &mut *out } else { &mut *cur };
        target[0] = rn[n];
        let cnt = nherm(l - n);
        for i in 1..cnt {
            let [t, u, v] = tuv[i];
            let (t, u, v) = (t as usize, u as usize, v as usize);
            let val = if t > 0 {
                let mut r = x * prev[herm_idx(t - 1, u, v)];
                if t > 1 {
                    r += (t - 1) as f64 * prev[herm_idx(t - 2, u, v)];
                }
                r
            } else if u > 0 {
                let mut r = y * prev[herm_idx(t, u - 1, v)];
                if u > 1 {
                    r += (u - 1) as f64 * prev[herm_idx(t, u - 2, v)];
                }
                r
            } else {
                let mut r = z * prev[herm_idx(t, u, v - 1)];
                if v > 1 {
                    r += (v - 1) as f64 * prev[herm_idx(t, u, v - 2)];
                }
                r
            };
            target[i] = val;
        }
        if n > 0 {
            std::mem::swap(&mut prev, &mut cur);
        }
    }
}

/// Precomputed data for one contracted shell pair (a, b) in the SHARK layout: for each
/// retained primitive pair, the exponent sum p, the center P and the E-matrix
/// E[r][mu nu] over Hermite functions r of degree <= la + lb and the contracted functions
/// mu of a and nu of b (spherical transform and contraction coefficients folded in).
#[derive(Clone, Debug)]
pub struct ShellPair {
    pub sa: usize,
    pub sb: usize,
    pub la: usize,
    pub lb: usize,
    pub na: usize,
    pub nb: usize,
    pub nprim: usize,
    pub p: Vec<f64>,
    pub pc: Vec<[f64; 3]>,
    /// nprim * nherm(la+lb) * na * nb
    pub e: Vec<f64>,
    /// Largest |E| element over primitives (for screening).
    pub emax: f64,
}

impl ShellPair {
    #[inline]
    pub fn lab(&self) -> usize {
        self.la + self.lb
    }
    #[inline]
    pub fn nab(&self) -> usize {
        self.na * self.nb
    }
    #[inline]
    pub fn e_prim(&self, ip: usize) -> &[f64] {
        let sz = nherm(self.la + self.lb) * self.na * self.nb;
        &self.e[ip * sz..(ip + 1) * sz]
    }
}

/// Description of a shell for pair construction (lets auxiliary single shells pair with a
/// unit s "shell" of exponent 0).
pub struct ShellView<'a> {
    pub l: usize,
    pub center: [f64; 3],
    pub exps: &'a [f64],
    pub coefs: &'a [f64],
    /// nfunc x ncart transformation (row-major)
    pub transform: Vec<f64>,
    pub nfunc: usize,
}

impl<'a> ShellView<'a> {
    pub fn from_shell(s: &'a Shell) -> Self {
        let transform = BasisSet::shell_transform(s);
        ShellView {
            l: s.l,
            center: s.center,
            exps: &s.exps,
            coefs: &s.coefs,
            transform,
            nfunc: s.nfunc(),
        }
    }
}

pub static UNIT_EXP: [f64; 1] = [0.0];
pub static UNIT_COEF: [f64; 1] = [1.0];

impl ShellView<'static> {
    /// The constant function 1 (exponent 0) used to turn 3- and 2-center integrals into
    /// 4-center ones.
    pub fn unit(center: [f64; 3]) -> Self {
        ShellView {
            l: 0,
            center,
            exps: &UNIT_EXP,
            coefs: &UNIT_COEF,
            transform: vec![1.0],
            nfunc: 1,
        }
    }
}

/// The pair data of shells `a` (index `sa`) and `b` (`sb`) without derivative expansions: the
/// order-zero case of the one primitive loop (`core::pair_data::build`).
pub fn build_pair(a: &ShellView, b: &ShellView, sa: usize, sb: usize) -> ShellPair {
    crate::core::pair_data::build(a, b, sa, sb, 0, crate::core::pair_data::PRIM_PAIR_CUT).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn herm_order_consistent() {
        let t = herm_tuv();
        for (i, x) in t.iter().take(nherm(10)).enumerate() {
            assert_eq!(herm_idx(x[0] as usize, x[1] as usize, x[2] as usize), i);
        }
    }

    #[test]
    fn r_tensor_against_direct() {
        // R_100 = X * R^{(1)}_000 = X * (-2a) F1 ; R_200 = R^{(1)}_000 + X^2 R^{(2)}_000
        let alpha = 0.7;
        let pq = [0.3, -0.4, 0.5];
        let t = alpha * (0.09 + 0.16 + 0.25);
        let mut fm = [0.0; 5];
        boys::boys(4, t, &mut fm);
        let mut out = vec![0.0; nherm(4)];
        let mut work = vec![0.0; 2 * nherm(4)];
        r_tensor(4, alpha, pq, &fm, 1.0, &mut out, &mut work);
        let r1 = -2.0 * alpha * fm[1];
        let r2 = 4.0 * alpha * alpha * fm[2];
        assert!((out[herm_idx(1, 0, 0)] - pq[0] * r1).abs() < 1e-15);
        assert!((out[herm_idx(2, 0, 0)] - (r1 + pq[0] * pq[0] * r2)).abs() < 1e-15);
        assert!((out[herm_idx(1, 1, 0)] - pq[0] * pq[1] * r2).abs() < 1e-15);
    }
}
