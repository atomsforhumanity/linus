//! The raw path: the integrals of single shell pairs, triplets and quartets into a caller's
//! buffer, in the style of libint and libcint, for tests, checks and external users. The plans
//! ([`crate::JkPlan`] and the contraction drivers) are the fast path for whole matrices; the
//! raw path computes one block at a time with the same kernels.
//!
//! **Conventions** (the ones of docs/conventions.md, as everywhere in linus):
//! - functions within a shell: spherical functions m = -l..l (p shells x, y, z), or Cartesian
//!   components in lexicographic order (xx, xy, xz, yy, yz, zz); PySCF's (libcint's) order;
//! - normalization: unit-normalized spherical functions and x^l, y^l, z^l components;
//! - positions in bohr, integrals in Hartree atomic units, chemists' notation (ab|cd);
//! - layouts, row-major:
//!   - (ab|cd): `out[((a * nb + b) * nc + c) * nd + d]`;
//!   - first derivatives: `out[((X * 3 + k) * nabcd) + abcd]` for the centers X = A, B, C, D
//!     of the shells a, b, c, d and k = x, y, z (12 blocks);
//!   - second derivatives (not computed here yet; the layout callers can rely on): the full
//!     12 x 12 matrix of (center, direction) pairs, `out[((i * 12 + j) * nabcd) + abcd]` with
//!     i = 3 X + k and j = 3 Y + l (144 blocks, symmetric in i and j);
//!   - (P|ab): `out[(p * na + a) * nb + b]`; (P|Q): `out[p * nq + q]`;
//!   - one-electron: `out[a * nb + b]`.
//!
//! The first derivatives satisfy translational invariance exactly by construction: the
//! derivative with respect to D is minus the sum of the other three.

use crate::basis::Shell;
use crate::core::deriv::PairDeriv;
use crate::core::hermite::{ShellPair, ShellView};
use crate::core::pair_data::{self, PRIM_PAIR_CUT};
use crate::eri4c::jk_deriv::{eri_quartet_deriv_a, DerivAWork};
use crate::eri4c::{eri_quartet, EriWork};
use crate::error::IntsError;
use crate::operator::Operator;
use crate::pairs::DerivOrder;

/// The pair data of shells a and b (in this order), with first-derivative expansions when
/// `deriv` is set.
fn pair(a: &Shell, b: &Shell, deriv: bool) -> (ShellPair, Option<PairDeriv>) {
    let (va, vb) = (ShellView::from_shell(a), ShellView::from_shell(b));
    pair_data::build(&va, &vb, 0, 0, deriv as usize, PRIM_PAIR_CUT)
}

/// The pair (P, 1) of an auxiliary shell: three- and two-center integrals are four-center ones
/// with the constant function as the second shell.
fn aux_pair(p: &Shell) -> ShellPair {
    let (v, u) = (ShellView::from_shell(p), ShellView::unit(p.center));
    pair_data::build(&v, &u, 0, usize::MAX, 0, PRIM_PAIR_CUT).0
}

/// Two-electron integrals of one shell quartet, and their first derivatives, for one operator
/// (see the module docs for the layouts). One engine per thread: it keeps its scratch.
pub struct Eri4Raw {
    op: Operator,
    order: DerivOrder,
    w: EriWork,
    wd: DerivAWork,
    tmp: Vec<f64>,
}

impl Eri4Raw {
    /// An engine for `op` (1/r, erf(omega r)/r and their combinations) up to derivative order
    /// `order` (zero or one).
    pub fn new(op: Operator, order: DerivOrder) -> Result<Self, IntsError> {
        if order > DerivOrder::One {
            return Err(IntsError::Order {
                need: order,
                have: DerivOrder::One,
            });
        }
        Ok(Eri4Raw {
            op,
            order,
            w: EriWork::new(),
            wd: DerivAWork::with_op(op),
            tmp: Vec::new(),
        })
    }

    /// The length of the buffer [`Self::compute`] writes for these shells.
    pub fn buffer_len(&self, a: &Shell, b: &Shell, c: &Shell, d: &Shell) -> usize {
        let n = a.nfunc() * b.nfunc() * c.nfunc() * d.nfunc();
        if self.order == DerivOrder::Zero {
            n
        } else {
            12 * n
        }
    }

    /// (ab|cd), or its first derivatives with respect to the four centers, into `out`.
    ///
    /// # Panics
    /// When `out` is shorter than [`Self::buffer_len`].
    pub fn compute(&mut self, a: &Shell, b: &Shell, c: &Shell, d: &Shell, out: &mut [f64]) {
        let (na, nb, nc, nd) = (a.nfunc(), b.nfunc(), c.nfunc(), d.nfunc());
        let (nab, ncd) = (na * nb, nc * nd);
        let n = nab * ncd;
        if self.order == DerivOrder::Zero {
            let (bra, _) = pair(a, b, false);
            let (ket, _) = pair(c, d, false);
            eri_quartet(&bra, &ket, &self.op, &mut out[..n], &mut self.w);
            return;
        }
        let out = &mut out[..12 * n];
        let ket = pair(c, d, false).0;
        // d/dA: the derivative kernel of the first center of the bra
        let (bra, bd) = pair(a, b, true);
        eri_quartet_deriv_a(
            &bra,
            bd.as_ref().unwrap(),
            &ket,
            &mut out[..3 * n],
            &mut self.wd,
        );
        // d/dB: the same kernel on the bra (b, a), transposed back to (a, b)
        let (bra_ba, bd_ba) = pair(b, a, true);
        self.tmp.resize(3 * n, 0.0);
        eri_quartet_deriv_a(
            &bra_ba,
            bd_ba.as_ref().unwrap(),
            &ket,
            &mut self.tmp,
            &mut self.wd,
        );
        for k in 0..3 {
            for ia in 0..na {
                for ib in 0..nb {
                    let src = &self.tmp[(k * nab + ib * na + ia) * ncd..][..ncd];
                    out[(3 + k) * n + (ia * nb + ib) * ncd..][..ncd].copy_from_slice(src);
                }
            }
        }
        // d/dC: the kernel on the swapped quartet (cd|ab), transposed back to (ab|cd)
        let (ket_cd, kd_cd) = pair(c, d, true);
        let bra_ab = pair(a, b, false).0;
        eri_quartet_deriv_a(
            &ket_cd,
            kd_cd.as_ref().unwrap(),
            &bra_ab,
            &mut self.tmp,
            &mut self.wd,
        );
        for k in 0..3 {
            for cd in 0..ncd {
                for ab in 0..nab {
                    out[(6 + k) * n + ab * ncd + cd] = self.tmp[(k * ncd + cd) * nab + ab];
                }
            }
        }
        // d/dD from translational invariance
        for k in 0..3 {
            for x in 0..n {
                out[(9 + k) * n + x] =
                    -(out[k * n + x] + out[(3 + k) * n + x] + out[(6 + k) * n + x]);
            }
        }
    }
}

/// Three-center (P|ab) and two-center (P|Q) integrals of auxiliary shells P, Q with orbital
/// shells a, b, for one operator (see the module docs for the layouts).
pub struct Eri3Raw {
    op: Operator,
    w: EriWork,
}

impl Eri3Raw {
    /// An engine for `op`.
    pub fn new(op: Operator) -> Self {
        Eri3Raw {
            op,
            w: EriWork::new(),
        }
    }

    /// (P|ab) into `out` (np * na * nb values).
    pub fn three_center(&mut self, p: &Shell, a: &Shell, b: &Shell, out: &mut [f64]) {
        let n = p.nfunc() * a.nfunc() * b.nfunc();
        let ket = pair(a, b, false).0;
        eri_quartet(&aux_pair(p), &ket, &self.op, &mut out[..n], &mut self.w);
    }

    /// (P|Q) into `out` (np * nq values).
    pub fn two_center(&mut self, p: &Shell, q: &Shell, out: &mut [f64]) {
        let n = p.nfunc() * q.nfunc();
        eri_quartet(
            &aux_pair(p),
            &aux_pair(q),
            &self.op,
            &mut out[..n],
            &mut self.w,
        );
    }
}

/// One-electron integrals of one shell pair (layout `out[a * nb + b]`).
pub struct OneERaw;

impl OneERaw {
    /// The overlap <a|b> and kinetic energy <a| -1/2 nabla^2 |b> blocks.
    pub fn overlap_kinetic(a: &Shell, b: &Shell, s: &mut [f64], t: &mut [f64]) {
        crate::one_e::overlap_kinetic_block(a, b, s, t);
    }

    /// The potential of point charges, sum_C -q_C <a| 1/|r - C| |b>, for charges (q, C).
    pub fn potential(a: &Shell, b: &Shell, charges: &[(f64, [f64; 3])], out: &mut [f64]) {
        let src = crate::potential::PointCharges {
            charges,
            first_slot: 0,
        };
        crate::potential::matrix_block(&pair(a, b, false).0, &src, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{BasisDef, BasisSet};
    use crate::core::deriv::pair_with_deriv;
    use crate::eri4c::deriv::DerivWork;
    use crate::pairs::PairSet;

    fn basis() -> BasisSet {
        let def = BasisDef::load("def2-svp", None).unwrap();
        let positions = [[0.0, 0.0, 0.1], [1.4, 0.2, -0.9], [-1.5, -0.1, -0.8]];
        BasisSet::new(&def, &[8, 1, 1], &positions, true).unwrap()
    }

    /// The raw first derivatives against the reference kernel, their translational
    /// invariance, and the values against a plain quartet.
    #[test]
    fn eri4_raw_matches_reference() {
        let b = basis();
        let sh = &b.shells;
        let quartets = [(3, 0, 5, 1), (7, 3, 2, 9), (8, 8, 4, 0), (10, 6, 7, 2)];
        for op in [
            Operator::Coulomb,
            Operator::Attenuated {
                coulomb: 0.3,
                long_range: 0.5,
                omega: 0.4,
            },
        ] {
            let mut e0 = Eri4Raw::new(op, DerivOrder::Zero).unwrap();
            let mut e1 = Eri4Raw::new(op, DerivOrder::One).unwrap();
            let mut wr = DerivWork::with_op(op);
            for &(i, j, k, l) in &quartets {
                let (a, bb, c, d) = (&sh[i], &sh[j], &sh[k], &sh[l]);
                let mut v = vec![0.0; e0.buffer_len(a, bb, c, d)];
                e0.compute(a, bb, c, d, &mut v);
                let mut g = vec![0.0; e1.buffer_len(a, bb, c, d)];
                e1.compute(a, bb, c, d, &mut g);
                let n = v.len();
                let (bra, bd) = pair_with_deriv(&b, i, j, 1);
                let (ket, kd) = pair_with_deriv(&b, k, l, 1);
                let mut r = vec![0.0; 12 * n];
                crate::reference::eri_quartet_deriv1(&bra, &bd, &ket, &kd, &mut r, &mut wr);
                let scale = r.iter().fold(1e-30f64, |m, x| m.max(x.abs()));
                for (x, y) in g.iter().zip(&r) {
                    assert!(
                        (x - y).abs() < 1e-10 * scale.max(1.0),
                        "{op:?} {i}{j}{k}{l}: {x} {y}"
                    );
                }
                for kk in 0..3 {
                    for x in 0..n {
                        let s: f64 = (0..4).map(|c| g[(3 * c + kk) * n + x]).sum();
                        assert!(s.abs() < 1e-12 * scale.max(1.0));
                    }
                }
                let mut q = vec![0.0; n];
                eri_quartet(&bra, &ket, &op, &mut q, &mut EriWork::new());
                assert_eq!(q, v);
            }
        }
        assert!(Eri4Raw::new(Operator::Coulomb, DerivOrder::Two).is_err());
    }

    /// Three- and two-center integrals are the four-center ones with unit shells; the
    /// one-electron blocks are the matrices' blocks.
    #[test]
    fn eri3_and_one_e_raw() {
        let b = basis();
        let aux = BasisSet::new(
            &BasisDef::load("def2-universal-jkfit", None).unwrap(),
            &[8, 1, 1],
            &[[0.0, 0.0, 0.1], [1.4, 0.2, -0.9], [-1.5, -0.1, -0.8]],
            true,
        )
        .unwrap();
        let opts = crate::pairs::PairOptions::default();
        let (pairs, apairs) = (PairSet::build(&b, opts), PairSet::aux(&aux, opts));
        let r3 = crate::reference::eri_3c(&b, &pairs, &aux, &apairs);
        let r2 = crate::reference::eri_2c(&aux, &apairs);
        let (nbf, naux) = (b.nbf, aux.nbf);
        let mut e = Eri3Raw::new(Operator::Coulomb);
        let (mut big3, mut big2) = (0.0f64, 0.0f64);
        for (ip, iq, ia, ic) in [(4, 1, 3, 1), (7, 12, 5, 9), (2, 2, 8, 4)] {
            let (p, q) = (&aux.shells[ip], &aux.shells[iq]);
            let (a, c) = (&b.shells[ia], &b.shells[ic]);
            let (np, na, nc) = (p.nfunc(), a.nfunc(), c.nfunc());
            let mut v = vec![0.0; np * na * nc];
            e.three_center(p, a, c, &mut v);
            for x in 0..np {
                for i in 0..na {
                    for k in 0..nc {
                        let y = r3[(p.ao_start + x) * nbf * nbf
                            + (a.ao_start + i) * nbf
                            + c.ao_start
                            + k];
                        let z = v[(x * na + i) * nc + k];
                        assert!((z - y).abs() < 1e-12, "3c {ip}{ia}{ic} {x}{i}{k}: {z} {y}");
                        big3 = big3.max(y.abs());
                    }
                }
            }
            let mut v2 = vec![0.0; np * q.nfunc()];
            e.two_center(p, q, &mut v2);
            for x in 0..np {
                for y in 0..q.nfunc() {
                    let r = r2[(p.ao_start + x) * naux + q.ao_start + y];
                    let z = v2[x * q.nfunc() + y];
                    assert!((z - r).abs() < 1e-12, "2c {ip}{iq} {x}{y}: {z} {r}");
                    big2 = big2.max(r.abs());
                }
            }
        }
        assert!(big3 > 1e-2 && big2 > 1e-2, "{big3} {big2}");
        let (a, c) = (&b.shells[3], &b.shells[1]);
        let (s_full, t_full) = crate::reference::oneshot::overlap_kinetic(&b);
        let n = b.nbf;
        let (mut s, mut t) = (
            vec![0.0; a.nfunc() * c.nfunc()],
            vec![0.0; a.nfunc() * c.nfunc()],
        );
        OneERaw::overlap_kinetic(a, c, &mut s, &mut t);
        for ia in 0..a.nfunc() {
            for ic in 0..c.nfunc() {
                let x = (a.ao_start + ia) * n + c.ao_start + ic;
                assert!((s[ia * c.nfunc() + ic] - s_full[x]).abs() < 1e-14);
                assert!((t[ia * c.nfunc() + ic] - t_full[x]).abs() < 1e-14);
            }
        }
    }
}
