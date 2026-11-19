//! The shape of a generated three-center class: what the host must know to feed a kernel of
//! [`super::eri3c_gen`] (tools/intgen/cuda_eri3c.py), derived here from linus-ints' conventions
//! by the same rules as the generator, and checked against the hash the generator recorded.
//!
//! - The E patterns: the structural nonzeros (t', mn) of the Hermite expansion of a spherical
//!   orbital pair (l >= 2 spherical, p Cartesian x, y, z) and, for first derivatives, of its six
//!   derivatives d/dA_x..z, d/dB_x..z (one Hermite degree higher). A pair primitive's record is
//!   `[p, Px, Py, Pz, q, nonzeros of every set in pattern order]`.
//! - The R slots: the level-0 Hermite integrals R_{c + t'} the class needs (a contiguous range of
//!   Hermite indices), the H terms (S_k\[c\], slot of c + t') and the output rows (set, mn) with
//!   their nonzeros: the run-time tables of the "block" kernels.

use crate::cpu::{cart2sph, cart_components, herm_idx, herm_tuv, ncart, nherm};

/// Thread mapping of a generated kernel (see tools/intgen/ir.py, `CudaSchedule`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mapping {
    /// One thread per (orbital pair, auxiliary shell); grid over pairs x shells.
    Thread,
    /// One block per (orbital pair, tile of `ktile` auxiliary shells), dynamic shared memory.
    Block,
}

/// What a three-center kernel computes (the kind of a generated kernel and of an `Eri3c` builder).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Eri3cKind {
    /// (mn|P).
    Integrals,
    /// d/dA_x..z (mn|P) and d/dB_x..z (mn|P), six planes.
    Derivatives,
    /// Second derivatives, contracted with a Gamma into the Hessian's centre blocks
    /// (`Eri3c::hessian`): the 21 components X1_kl = d^2/dA_k dA_l, X2_kl = S_l d/dA_k,
    /// X3_kl = S_k S_l (S the derivative of the product's centre, linus-ints' `Eri3DerivPlan::hessian`) in the
    /// kernel's shell order.
    Hessian,
}

impl Eri3cKind {
    /// Output planes.
    pub fn planes(self) -> usize {
        match self {
            Eri3cKind::Integrals => 1,
            Eri3cKind::Derivatives => 6,
            Eri3cKind::Hessian => 21,
        }
    }
}

/// A generated kernel: source, entry point, schedule, launch data and record shape.
#[derive(Clone, Copy, Debug)]
pub struct GenKernel {
    pub kind: Eri3cKind,
    pub la: usize,
    pub lb: usize,
    pub lk: usize,
    pub entry: &'static str,
    /// The schedule tag (`CudaSchedule.tag()`).
    pub schedule: &'static str,
    /// Code the kernel source needs in front of it: the shared straight-line R recursion of
    /// the block kernels with R inline (an `r*.cuh` file), or empty.
    pub prelude: &'static str,
    pub source: &'static str,
    pub mapping: Mapping,
    /// Threads per block.
    pub threads: u32,
    /// Dynamic shared memory per block in bytes ("block" mapping).
    pub smem: u32,
    /// Auxiliary shells per block ("block" mapping).
    pub ktile: u32,
    /// f64 values per pair-primitive record.
    pub rec: usize,
    /// R slots, used t' indices, E nonzeros, output rows.
    pub nr: usize,
    pub nht: usize,
    pub nnz: usize,
    pub nrows: usize,
    /// FNV-1a hash of the patterns and R slots the source was generated for.
    pub hash: u64,
    /// FP64 operations per auxiliary primitive (Boys, prefactor) and per (pair primitive,
    /// auxiliary shell) (R, H, contraction), from the generator's IR.
    pub prim_flops: u32,
    pub pair_shell_flops: u32,
}

/// Functions of a shell in linus's convention (s and p Cartesian, l >= 2 spherical).
pub fn nfunc(l: usize) -> usize {
    match l {
        0 => 1,
        1 => 3,
        _ => 2 * l + 1,
    }
}

/// Per function of the shell, the Cartesian components with nonzero weight.
fn transform_pattern(l: usize) -> Vec<Vec<usize>> {
    let nc = ncart(l);
    if l <= 1 {
        return (0..nc).map(|i| vec![i]).collect();
    }
    let t = cart2sph(l);
    (0..2 * l + 1)
        .map(|m| (0..nc).filter(|&c| t[m * nc + c] != 0.0).collect())
        .collect()
}

/// Structural nonzeros (t', mn) of the Hermite expansion of a pair (la, lb), or of its derivative
/// d/dX_k (X the centre of a for `deriv = Some((0, k))`, of b for `Some((1, k))`), sorted.
pub fn e_pattern(la: usize, lb: usize, deriv: Option<(usize, usize)>) -> Vec<(usize, usize)> {
    let (ca, cb) = (cart_components(la), cart_components(lb));
    let (ta, tb) = (transform_pattern(la), transform_pattern(lb));
    let deg = la + lb + deriv.is_some() as usize;
    let tuv = &herm_tuv()[..nherm(deg)];
    let mut nz = std::collections::BTreeSet::new();
    let mut add = |mx: [usize; 3], mn: usize| {
        for (r, t) in tuv.iter().enumerate() {
            if (t[0] as usize) <= mx[0] && (t[1] as usize) <= mx[1] && (t[2] as usize) <= mx[2] {
                nz.insert((r, mn));
            }
        }
    };
    for (mu, ra) in ta.iter().enumerate() {
        for (nu, rb) in tb.iter().enumerate() {
            let mn = mu * tb.len() + nu;
            for &ia in ra {
                for &ib in rb {
                    let mut a = ca[ia];
                    let mut b = cb[ib];
                    match deriv {
                        None => add([a[0] + b[0], a[1] + b[1], a[2] + b[2]], mn),
                        Some((c, k)) => {
                            let d = if c == 0 { &mut a } else { &mut b };
                            d[k] += 1;
                            add([a[0] + b[0], a[1] + b[1], a[2] + b[2]], mn);
                            let d = if c == 0 { &mut a } else { &mut b };
                            d[k] -= 1;
                            if d[k] > 0 {
                                d[k] -= 1;
                                add([a[0] + b[0], a[1] + b[1], a[2] + b[2]], mn);
                            }
                        }
                    }
                }
            }
        }
    }
    nz.into_iter().collect()
}

/// Structural nonzeros (t', mn) of the Hermite expansion of d^2/dA_k dA_l of a pair (la, lb)
/// (both derivatives on the centre of a; degree la + lb + 2), sorted: the Cartesian components
/// i + s_k + s_l (s = +-1, the shifts that keep them non-negative) of a, as the generator's
/// `e_pattern_deriv2`.
pub fn e_pattern_deriv2(la: usize, lb: usize, k: usize, l: usize) -> Vec<(usize, usize)> {
    let (ca, cb) = (cart_components(la), cart_components(lb));
    let (ta, tb) = (transform_pattern(la), transform_pattern(lb));
    let tuv = &herm_tuv()[..nherm(la + lb + 2)];
    let mut nz = std::collections::BTreeSet::new();
    for (mu, ra) in ta.iter().enumerate() {
        for (nu, rb) in tb.iter().enumerate() {
            let mn = mu * tb.len() + nu;
            for &ia in ra {
                for &ib in rb {
                    let b = cb[ib];
                    for s1 in [1i64, -1] {
                        let mut a1 = ca[ia].map(|x| x as i64);
                        a1[k] += s1;
                        if a1[k] < 0 {
                            continue;
                        }
                        for s2 in [1i64, -1] {
                            let mut a2 = a1;
                            a2[l] += s2;
                            if a2[l] < 0 {
                                continue;
                            }
                            let mx = [0, 1, 2].map(|q| a2[q] as usize + b[q]);
                            for (r, t) in tuv.iter().enumerate() {
                                if (t[0] as usize) <= mx[0]
                                    && (t[1] as usize) <= mx[1]
                                    && (t[2] as usize) <= mx[2]
                                {
                                    nz.insert((r, mn));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    nz.into_iter().collect()
}

/// Second-derivative components (k, l), k <= l, in linus-ints' order (`D2_COMPS`).
pub const D2_COMPS: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

/// Head of a pair-primitive record: p, Px, Py, Pz, q.
pub const REC_HEAD: usize = 5;
/// Values per auxiliary primitive: a, c (2a)^-l (-1)^l, Ax, Ay, Az, q.
pub const AUX_REC: usize = 6;

/// The class (la lb | lk), la >= lb, of one kind, as the generator sees it.
#[derive(Clone, Debug)]
pub struct ClassShape {
    pub la: usize,
    pub lb: usize,
    pub lk: usize,
    pub kind: Eri3cKind,
    /// Hermite degree of the pair expansions (la + lb, + 1 for first derivatives, + 2 for
    /// second derivatives).
    pub hdeg: usize,
    pub nk: usize,
    pub nab: usize,
    pub na: usize,
    pub nb: usize,
    /// The value sets of a primitive's record (each over its own degree): E; for first
    /// derivatives A x, y, z, B x, y, z; for second derivatives E_AA,kl (`D2_COMPS`), E_A,k and
    /// E. Their record offsets are `soff`.
    pub vsets: Vec<Vec<(usize, usize)>>,
    /// The output sets: (value set, Hermite shift); the value sets themselves except for second
    /// derivatives (X1_kl = E_AA,kl, X2_kl = S_l E_A,k, X3_kl = S_k S_l E).
    pub osets: Vec<(usize, [usize; 3])>,
    /// The output sets' patterns (t' of the shifted expansion, mn), sorted.
    pub sets: Vec<Vec<(usize, usize)>>,
    pub soff: Vec<usize>,
    pub nnz: usize,
    pub rec: usize,
    /// Hermite indices t' used by any set, ascending.
    pub tp_used: Vec<usize>,
    /// Hermite indices of the R slots, ascending.
    pub r_targets: Vec<usize>,
}

impl ClassShape {
    pub fn new(la: usize, lb: usize, lk: usize, kind: Eri3cKind) -> ClassShape {
        assert!(la >= lb);
        let hdeg = la
            + lb
            + match kind {
                Eri3cKind::Integrals => 0,
                Eri3cKind::Derivatives => 1,
                Eri3cKind::Hessian => 2,
            };
        let vsets: Vec<Vec<(usize, usize)>> = match kind {
            Eri3cKind::Integrals => vec![e_pattern(la, lb, None)],
            Eri3cKind::Derivatives => (0..6)
                .map(|s| e_pattern(la, lb, Some((s / 3, s % 3))))
                .collect(),
            Eri3cKind::Hessian => D2_COMPS
                .iter()
                .map(|&(k, l)| e_pattern_deriv2(la, lb, k, l))
                .chain((0..3).map(|k| e_pattern(la, lb, Some((0, k)))))
                .chain(std::iter::once(e_pattern(la, lb, None)))
                .collect(),
        };
        let unit = |d: usize| -> [usize; 3] { [0, 1, 2].map(|q| (q == d) as usize) };
        let osets: Vec<(usize, [usize; 3])> = if kind == Eri3cKind::Hessian {
            (0..6)
                .map(|c| (c, [0; 3]))
                .chain((0..9).map(|kl| (6 + kl / 3, unit(kl % 3))))
                .chain(D2_COMPS.iter().map(|&(k, l)| {
                    let (a, b) = (unit(k), unit(l));
                    (9, [0, 1, 2].map(|q| a[q] + b[q]))
                }))
                .collect()
        } else {
            (0..vsets.len()).map(|s| (s, [0; 3])).collect()
        };
        let tuv_all = herm_tuv();
        let shifted = |r: usize, sh: [usize; 3]| -> usize {
            let t = tuv_all[r];
            herm_idx(
                t[0] as usize + sh[0],
                t[1] as usize + sh[1],
                t[2] as usize + sh[2],
            )
        };
        let sets: Vec<Vec<(usize, usize)>> = osets
            .iter()
            .map(|&(v, sh)| {
                let mut p: Vec<(usize, usize)> = vsets[v]
                    .iter()
                    .map(|&(r, mn)| (shifted(r, sh), mn))
                    .collect();
                p.sort();
                p
            })
            .collect();
        let mut soff = Vec::new();
        let mut o = 0;
        for p in &vsets {
            soff.push(o);
            o += p.len();
        }
        let tp: std::collections::BTreeSet<usize> = sets
            .iter()
            .flat_map(|p| p.iter().map(|&(r, _)| r))
            .collect();
        let tp_used: Vec<usize> = tp.into_iter().collect();
        let s = cart2sph(lk);
        let nc = ncart(lk);
        let nk = nfunc(lk);
        let cused: Vec<usize> = (0..nc)
            .filter(|&c| (0..nk).any(|k| s[k * nc + c] != 0.0))
            .collect();
        let carts = cart_components(lk);
        let carts = &carts;
        let tuv = herm_tuv();
        let rt: std::collections::BTreeSet<usize> = cused
            .iter()
            .flat_map(|&c| {
                tp_used.iter().map(move |&t| {
                    let (a, b) = (carts[c], tuv[t]);
                    herm_idx(
                        a[0] + b[0] as usize,
                        a[1] + b[1] as usize,
                        a[2] + b[2] as usize,
                    )
                })
            })
            .collect();
        ClassShape {
            la,
            lb,
            lk,
            kind,
            hdeg,
            nk,
            na: nfunc(la),
            nb: nfunc(lb),
            nab: nfunc(la) * nfunc(lb),
            vsets,
            osets,
            sets,
            soff,
            nnz: o,
            rec: REC_HEAD + o,
            tp_used,
            r_targets: rt.into_iter().collect(),
        }
    }

    pub fn nr(&self) -> usize {
        self.r_targets.len()
    }

    /// Output rows (set, mn).
    pub fn nrows(&self) -> usize {
        self.sets.len() * self.nab
    }

    /// Slot of R_{c + t'}.
    fn rslot(&self, c: usize, t: usize) -> usize {
        let (a, b) = (cart_components(self.lk)[c], herm_tuv()[t]);
        let h = herm_idx(
            a[0] + b[0] as usize,
            a[1] + b[1] as usize,
            a[2] + b[2] as usize,
        );
        self.r_targets.binary_search(&h).expect("R slot")
    }

    /// The "block" kernels' tables: (hp, hc, hr) the H terms per (k, t' position) as CSR with
    /// S_k\[c\] and the R slot of c + t'; (rp, re, rt) the rows (set, mn) with the record offsets
    /// of their E nonzeros and those nonzeros' t' positions, in t' order.
    pub fn tables(&self) -> Eri3cTables {
        let s = cart2sph(self.lk);
        let nc = ncart(self.lk);
        let mut t = Eri3cTables {
            hp: vec![0],
            rp: vec![0],
            ..Default::default()
        };
        for k in 0..self.nk {
            for &tp in &self.tp_used {
                for c in 0..nc {
                    let x = s[k * nc + c];
                    if x != 0.0 {
                        t.hc.push(x);
                        t.hr.push(self.rslot(c, tp) as i32);
                    }
                }
                t.hp.push(t.hc.len() as i32);
            }
        }
        let tpos = |r: usize| self.tp_used.binary_search(&r).unwrap() as i32;
        let tuv = herm_tuv();
        for &(v, sh) in &self.osets {
            let mut by: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.nab];
            for (j, &(r, mn)) in self.vsets[v].iter().enumerate() {
                let t = tuv[r];
                let rs = herm_idx(
                    t[0] as usize + sh[0],
                    t[1] as usize + sh[1],
                    t[2] as usize + sh[2],
                );
                by[mn].push((REC_HEAD + self.soff[v] + j, rs));
            }
            for mut row in by {
                row.sort_by_key(|&(_, r)| r);
                for (off, r) in row {
                    t.re.push(off as i32);
                    t.rt.push(tpos(r));
                }
                t.rp.push(t.re.len() as i32);
            }
        }
        t
    }

    /// The generator's FNV-1a hash of the sets and the R slots.
    pub fn pattern_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |x: u64| {
            h ^= x;
            h = h.wrapping_mul(0x100000001b3);
        };
        for p in &self.sets {
            mix(p.len() as u64);
            for &(r, mn) in p {
                mix((r * 65536 + mn) as u64);
            }
        }
        if self.kind == Eri3cKind::Hessian {
            // the record's value sets and the output sets' sources
            for p in &self.vsets {
                mix(p.len() as u64);
                for &(r, mn) in p {
                    mix((r * 65536 + mn) as u64);
                }
            }
            for &(v, sh) in &self.osets {
                mix((v * 4096 + sh[0] * 256 + sh[1] * 16 + sh[2]) as u64);
            }
        }
        for &x in &self.r_targets {
            mix(x as u64);
        }
        h
    }

    /// Checks this shape against the generated kernel's record.
    pub fn check(&self, k: &GenKernel) -> Result<(), String> {
        if (k.kind, k.la, k.lb, k.lk) != (self.kind, self.la, self.lb, self.lk)
            || k.rec != self.rec
            || k.nr != self.nr()
            || k.nht != self.tp_used.len()
            || k.nnz != self.nnz
            || k.nrows != self.nrows()
            || k.hash != self.pattern_hash()
        {
            return Err(format!(
                "generated kernel {} does not match the class shape derived from linus-ints \
                 (regenerate with tools/intgen/intgen.py cuda)",
                k.entry
            ));
        }
        Ok(())
    }
}

/// Run-time tables of a "block" kernel (see [`ClassShape::tables`]).
#[derive(Clone, Debug, Default)]
pub struct Eri3cTables {
    pub hp: Vec<i32>,
    pub hc: Vec<f64>,
    pub hr: Vec<i32>,
    pub rp: Vec<i32>,
    pub re: Vec<i32>,
    pub rt: Vec<i32>,
}

/// The generated kernel of a class and kind (the default schedule, tuned on sm_120).
pub fn gen_kernel(kind: Eri3cKind, la: usize, lb: usize, lk: usize) -> Option<&'static GenKernel> {
    super::eri3c_gen::KERNELS
        .iter()
        .find(|k| (k.kind, k.la, k.lb, k.lk) == (kind, la, lb, lk))
}

/// The generated kernel of a class and kind for a device of compute capability `cc`: the
/// architecture's own variant where its tuning chose another schedule
/// ([`super::eri3c_gen_arch::ARCH_KERNELS`]), else the default ([`gen_kernel`]).
pub fn gen_kernel_for(
    cc: (i32, i32),
    kind: Eri3cKind,
    la: usize,
    lb: usize,
    lk: usize,
) -> Option<&'static GenKernel> {
    let arch = (10 * cc.0 + cc.1) as u32;
    super::eri3c_gen_arch::ARCH_KERNELS
        .iter()
        .find(|(a, k)| *a == arch && (k.kind, k.la, k.lb, k.lk) == (kind, la, lb, lk))
        .map(|(_, k)| k)
        .or_else(|| gen_kernel(kind, la, lb, lk))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_generated_kernel_matches_the_shape_derived_from_linus_ints() {
        let mut n = 0;
        for k in super::super::eri3c_gen::KERNELS.iter() {
            ClassShape::new(k.la, k.lb, k.lk, k.kind).check(k).unwrap();
            n += 1;
        }
        // (la lb | lk) with lb <= la <= 4, lk <= 6, three kinds
        assert_eq!(n, 3 * 15 * 7);
        // the per-architecture variants: the same shapes, another schedule than the default's
        for (arch, k) in super::super::eri3c_gen_arch::ARCH_KERNELS.iter() {
            ClassShape::new(k.la, k.lb, k.lk, k.kind).check(k).unwrap();
            let d = gen_kernel(k.kind, k.la, k.lb, k.lk).unwrap();
            assert_ne!(d.schedule, k.schedule, "sm_{arch} {}", k.entry);
            let cc = ((*arch / 10) as i32, (*arch % 10) as i32);
            let got = gen_kernel_for(cc, k.kind, k.la, k.lb, k.lk).unwrap();
            assert_eq!(got.schedule, k.schedule);
        }
    }

    #[test]
    fn patterns_match_linus_ints_batched_kernels() {
        // the energy patterns are linus-ints' E_NZ tables (classes.rs) by construction; spot
        // check the counts recorded there
        for (la, lb, nnz) in [(1, 0, 6), (2, 2, 391), (3, 3, 2143), (4, 4, 7649)] {
            assert_eq!(e_pattern(la, lb, None).len(), nnz);
        }
    }

    #[test]
    fn tables_cover_every_nonzero_once() {
        let c = ClassShape::new(3, 2, 4, Eri3cKind::Derivatives);
        let t = c.tables();
        assert_eq!(t.re.len(), c.nnz);
        assert_eq!(t.rp.len(), c.nrows() + 1);
        assert_eq!(t.hp.len(), c.nk * c.tp_used.len() + 1);
        let mut seen = t.re.clone();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), c.nnz);
    }

    /// Second derivatives: the 21 outputs reference every value of the record (10 value sets)
    /// through shifts, and the shifted patterns stay inside the class's Hermite degree.
    #[test]
    fn second_derivative_tables_reference_the_value_sets() {
        for (la, lb, lk) in [(0, 0, 0), (1, 0, 2), (2, 1, 3), (3, 3, 1)] {
            let c = ClassShape::new(la, lb, lk, Eri3cKind::Hessian);
            assert_eq!((c.vsets.len(), c.osets.len()), (10, 21));
            let t = c.tables();
            let terms: usize = c.sets.iter().map(|p| p.len()).sum();
            assert_eq!(t.re.len(), terms);
            assert_eq!(t.rp.len(), c.nrows() + 1);
            let mut seen = t.re.clone();
            seen.sort();
            seen.dedup();
            assert_eq!(seen.len(), c.nnz);
            let nh = nherm(la + lb + 2);
            assert!(c.sets.iter().flatten().all(|&(r, _)| r < nh));
        }
    }
}
