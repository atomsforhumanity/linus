//! The symmetry of a converged state's orbital rotations: the per-irrep searches of the stability
//! analyses (decision D45).
//!
//! At a point with symmetry the orbital Hessian is block-diagonal by irreducible
//! representation, so a Davidson search started in one block reaches another only through
//! whatever breaks the symmetry in its arithmetic: round-off, the one-sided products' O(h) term,
//! a grid. Whether a negative mode of another block is found then depends on the last bits
//! (triplet O3: a last-bit change of the SAD guess turned a caught saddle into a missed one).
//! The analyses therefore search each irrep of the state's abelian point group on its own, from
//! its lowest-gap pair, with every product projected onto the irrep: the search cannot drift,
//! and the Ritz value is a Rayleigh quotient of the full Hessian, an upper bound of its lowest
//! eigenvalue.
//!
//! The group is the largest D2h-family subgroup of the molecule's point group acting on its basis
//! ([`AoGroup`], linus-base) reduced to the elements the converged state keeps: an element whose
//! image of the occupied space leaves it by more than [`BROKEN`] is dropped (a symmetry-broken
//! state, a distorted geometry), and the largest subgroup of the rest is used. Its action on a
//! rotation kappa (virtual x occupied, per spin) is kappa -> U_v kappa U_o^T, with U = C^T S R C
//! the element's matrix over the orbitals; the projector onto irrep b is the product over the
//! generators k of (1 + chi_b(k) T_k) / 2. Nothing needs the orbitals to be symmetry-adapted: a
//! degenerate set that the eigensolver returned in an arbitrary basis projects the same.
//!
//! An irrep's start is its lowest-gap pair: the first pair, in the order of eps_a - eps_i (ties in
//! the vectors' order), with at least [`START_WEIGHT`] of its weight in the irrep, projected. A
//! pair of non-degenerate orbitals lies in one irrep and is its own start, signed by D29's phase
//! rule, so the first irrep's start is exactly the single start of a structure without symmetry.
//! A projected start (orbitals of a degenerate set) is signed by its AO form C_v kappa C_o^T: the
//! largest element positive, which is a property of the rotation, not of the eigensolver's basis.
//!
//! A search of [`super::WIDE_START_PRODUCTS`] or more products (the full analysis, the zero-mode
//! check of a Hessian SCF) starts from the irrep's wide start instead: the normalized sum of its
//! [`super::WIDE_PAIRS`] lowest-gap pairs, each signed as above. The orbital-energy gap can misrank
//! a pair's curvature by 0.2 Eh in a transition metal's d shell (FeH, B3LYP/def2-SVP: a dxy -> sigma
//! pair third in its irrep by gap, 0.19 Eh, carries a -1.1e-3 mode that a search from the lowest
//! pair alone never reaches); a two-product probe keeps the single pair, which a sum dilutes.

use super::phase;
use faer::{Mat, MatRef, Par};
use linus_base::get_or_build;
use linus_base::linalg::mm;
use linus_base::symmetry::abelian::AbelianGroup;
use linus_base::symmetry::ao::AoGroup;
use linus_base::symmetry::DEFAULT_TOLERANCE;
use linus_ints::BasisSet;
use std::sync::OnceLock;

/// An element of the group is broken at a converged point when it maps an occupied orbital onto
/// virtual ones by more than this (the largest |(C_v^T S R C_o)_ai| of either spin).
pub const BROKEN: f64 = 0.05;

/// A pair is an irrep's start when at least this fraction of it lies in the irrep.
pub const START_WEIGHT: f64 = 0.1;

/// A pair this close to lying wholly in one irrep is its own start (unprojected).
const PURE: f64 = 1.0 - 1e-6;

/// The structure and basis set an SCF's analyses read the molecule's symmetry from
/// (`ScfInput::geometry`); the group is built once, when an analysis first needs it.
pub struct Geometry<'a> {
    pub z: &'a [u32],
    pub positions: &'a [[f64; 3]],
    pub basis: &'a BasisSet,
    group: OnceLock<AoGroup>,
}

impl<'a> Geometry<'a> {
    pub fn new(z: &'a [u32], positions: &'a [[f64; 3]], basis: &'a BasisSet) -> Self {
        Geometry {
            z,
            positions,
            basis,
            group: OnceLock::new(),
        }
    }

    /// The molecule's abelian point group on its basis functions (built on first use).
    pub fn group(&self) -> &AoGroup {
        get_or_build(&self.group, || {
            AoGroup::new(self.z, self.positions, self.basis, DEFAULT_TOLERANCE)
        })
    }
}

/// One irrep's start: the irrep, its lowest gap eps_a - eps_i and the unit start vector
/// (flattened as the Davidson vectors).
#[derive(Clone, Debug)]
pub struct IrrepStart {
    pub irrep: usize,
    pub gap: f64,
    pub vector: Vec<f64>,
    /// The wide start: the normalized sum of the irrep's [`super::WIDE_PAIRS`] lowest pairs.
    pub wide: Vec<f64>,
}

/// The symmetry group of a converged state acting on its orbital rotations (see the module
/// docs).
pub struct PairSymmetry {
    group: AbelianGroup,
    /// Per spin and generator: U_o (nocc x nocc) and U_v (nvir x nvir).
    occ: Vec<Vec<Mat<f64>>>,
    vir: Vec<Vec<Mat<f64>>>,
    /// Per spin and element: the diagonals of U over the occupied and the virtual orbitals.
    occ_diag: Vec<Vec<Vec<f64>>>,
    vir_diag: Vec<Vec<Vec<f64>>>,
    nmo: usize,
    nocc: Vec<usize>,
}

impl PairSymmetry {
    /// The group `ao` at the orbitals `c` (per spin, nbf x nmo, `nocc` occupied; overlap `s`),
    /// reduced to the elements the state keeps.
    pub fn at(ao: &AoGroup, c: &[MatRef<'_, f64>], nocc: &[usize], s: MatRef<'_, f64>) -> Self {
        let par = Par::Seq;
        let nmo = c[0].ncols();
        let order = ao.group.order();
        // per spin: S C, and R_e C for every element e (R_e = the product of its generators)
        let sc: Vec<Mat<f64>> = c.iter().map(|cs| mm(s, *cs, par)).collect();
        let images: Vec<Vec<Mat<f64>>> = c
            .iter()
            .map(|cs| {
                let mut im: Vec<Mat<f64>> = vec![cs.to_owned()];
                for e in 1..order {
                    let low = e & e.wrapping_neg();
                    let k = low.trailing_zeros() as usize;
                    let r = ao.generators[k].apply(im[e ^ low].as_ref());
                    im.push(r);
                }
                im
            })
            .collect();
        // an element is kept when no occupied orbital's image leaves the occupied space
        let leak = |e: usize| -> f64 {
            let mut m: f64 = 0.0;
            for (sp, &no) in nocc.iter().enumerate() {
                let u = mm(
                    sc[sp].as_ref().subcols(no, nmo - no).transpose(),
                    images[sp][e].as_ref().subcols(0, no),
                    par,
                );
                for i in 0..no {
                    for a in 0..nmo - no {
                        m = m.max(u[(a, i)].abs());
                    }
                }
            }
            m
        };
        let kept: Vec<bool> = (0..order).map(|e| e == 0 || leak(e) < BROKEN).collect();
        let group = if kept.iter().all(|&k| k) {
            ao.group.clone()
        } else {
            ao.group.subgroup(|e| kept[e])
        };
        // the kept group's elements as elements of `ao`'s (matched by their matrices)
        let old_element = |m: &[[f64; 3]; 3]| -> usize {
            (0..order)
                .find(|&e| {
                    let x = ao.group.element(e).matrix;
                    (0..3).all(|i| (0..3).all(|j| (x[i][j] - m[i][j]).abs() < 1e-8))
                })
                .expect("a subgroup's element is an element of the group")
        };
        let elems: Vec<usize> = (0..group.order())
            .map(|e| old_element(&group.element(e).matrix))
            .collect();
        let gens: Vec<usize> = (0..group.generators.len()).map(|k| elems[1 << k]).collect();
        let block = |sp: usize, e: usize, off: usize, len: usize| -> Mat<f64> {
            mm(
                sc[sp].as_ref().subcols(off, len).transpose(),
                images[sp][e].as_ref().subcols(off, len),
                par,
            )
        };
        let diag = |sp: usize, e: usize, off: usize, len: usize| -> Vec<f64> {
            (off..off + len)
                .map(|p| {
                    (0..sc[sp].nrows())
                        .map(|mu| sc[sp][(mu, p)] * images[sp][e][(mu, p)])
                        .sum()
                })
                .collect()
        };
        let mut occ = Vec::new();
        let mut vir = Vec::new();
        let mut occ_diag = Vec::new();
        let mut vir_diag = Vec::new();
        for (sp, &no) in nocc.iter().enumerate() {
            occ.push(gens.iter().map(|&e| block(sp, e, 0, no)).collect());
            vir.push(gens.iter().map(|&e| block(sp, e, no, nmo - no)).collect());
            occ_diag.push(elems.iter().map(|&e| diag(sp, e, 0, no)).collect());
            vir_diag.push(elems.iter().map(|&e| diag(sp, e, no, nmo - no)).collect());
        }
        PairSymmetry {
            group,
            occ,
            vir,
            occ_diag,
            vir_diag,
            nmo,
            nocc: nocc.to_vec(),
        }
    }

    /// The converged state's group.
    pub fn group(&self) -> &AbelianGroup {
        &self.group
    }

    /// Project the rotation `v` (flattened per spin as (a, i) blocks) onto irrep `b`.
    pub fn project(&self, b: usize, v: &mut [f64]) {
        let mut off = 0;
        for (sp, &no) in self.nocc.iter().enumerate() {
            let nv = self.nmo - no;
            let len = nv * no;
            if len > 0 {
                let mut k = MatRef::from_column_major_slice(&v[off..off + len], nv, no).to_owned();
                for (j, (uo, uv)) in self.occ[sp].iter().zip(&self.vir[sp]).enumerate() {
                    let chi = AbelianGroup::character(b, 1 << j);
                    let t = mm(
                        mm(uv.as_ref(), k.as_ref(), Par::Seq).as_ref(),
                        uo.transpose(),
                        Par::Seq,
                    );
                    k = Mat::from_fn(nv, no, |a, i| 0.5 * (k[(a, i)] + chi * t[(a, i)]));
                }
                for i in 0..no {
                    for a in 0..nv {
                        v[off + i * nv + a] = k[(a, i)];
                    }
                }
            }
            off += len;
        }
    }

    /// The fraction of the pair (spin `sp`, occupied `i`, virtual `a` counted from the first
    /// virtual) that lies in irrep `b`: (1/|G|) sum_e chi_b(e) (U_v,e)_aa (U_o,e)_ii.
    fn weight(&self, b: usize, sp: usize, i: usize, a: usize) -> f64 {
        let n = self.group.order();
        (0..n)
            .map(|e| {
                AbelianGroup::character(b, e) * self.vir_diag[sp][e][a] * self.occ_diag[sp][e][i]
            })
            .sum::<f64>()
            / n as f64
    }

    /// One start per irrep that has a pair (see the module docs), in the order of their gaps
    /// (ties: the irrep found first), for the orbitals `c` (per spin) with energies `eps`.
    pub fn starts(&self, c: &[MatRef<'_, f64>], eps: &[&[f64]]) -> Vec<IrrepStart> {
        let mut pairs: Vec<(f64, usize, usize, usize, usize)> = Vec::new();
        let mut off = 0;
        for (sp, &no) in self.nocc.iter().enumerate() {
            let nv = self.nmo - no;
            for i in 0..no {
                for a in 0..nv {
                    pairs.push((eps[sp][no + a] - eps[sp][i], off + i * nv + a, sp, i, a));
                }
            }
            off += nv * no;
        }
        pairs.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        let n = self.group.order();
        let mut out: Vec<IrrepStart> = Vec::new();
        for &(gap, p, sp, i, a) in &pairs {
            if out.len() == n {
                break;
            }
            for b in 0..n {
                if out.iter().any(|s| s.irrep == b) {
                    continue;
                }
                let w = self.weight(b, sp, i, a);
                if w < START_WEIGHT {
                    continue;
                }
                let v = self.pair_start(c, b, w, (p, sp, i, a), off);
                out.push(IrrepStart {
                    irrep: b,
                    gap,
                    vector: v,
                    wide: Vec::new(),
                });
            }
        }
        // the wide starts: each irrep's lowest pairs, signed as its start is, summed
        for st in out.iter_mut() {
            let b = st.irrep;
            let mut wide = vec![0.0; off];
            let mut taken = 0;
            for &(_, p, sp, i, a) in &pairs {
                if taken == super::WIDE_PAIRS {
                    break;
                }
                let w = self.weight(b, sp, i, a);
                if w < START_WEIGHT {
                    continue;
                }
                let v = self.pair_start(c, b, w, (p, sp, i, a), off);
                wide.iter_mut().zip(&v).for_each(|(x, y)| *x += y);
                taken += 1;
            }
            let nrm = wide.iter().map(|x| x * x).sum::<f64>().sqrt();
            wide.iter_mut().for_each(|x| *x /= nrm);
            st.wide = wide;
        }
        out
    }

    /// The start vector over `n` rotations of the pair (index `p`, spin `sp`, occupied `i`,
    /// virtual `a`) with weight `w` in irrep `b`: a pure pair D29-signed, else projected onto the
    /// irrep and signed by its AO form.
    fn pair_start(
        &self,
        c: &[MatRef<'_, f64>],
        b: usize,
        w: f64,
        (p, sp, i, a): (usize, usize, usize, usize),
        n: usize,
    ) -> Vec<f64> {
        let no = self.nocc[sp];
        let mut v = vec![0.0; n];
        if w > PURE {
            // D29: the sign of the pair's two orbitals' phases
            let col = |q: usize| (0..c[sp].nrows()).map(move |mu| c[sp][(mu, q)]);
            v[p] = phase(col(i)) * phase(col(no + a));
        } else {
            v[p] = 1.0;
            self.project(b, &mut v);
            let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.iter_mut().for_each(|x| *x /= nrm);
            let sign = self.ao_sign(c, &v);
            v.iter_mut().for_each(|x| *x *= sign);
        }
        v
    }

    /// The sign that makes the largest element (the first of those within 1e-8 relative, in
    /// column-major order, per spin in turn) of the AO form C_v kappa C_o^T of the rotation `v`
    /// positive.
    fn ao_sign(&self, c: &[MatRef<'_, f64>], v: &[f64]) -> f64 {
        let mut best: Option<f64> = None;
        let mut big: f64 = 0.0;
        let mut forms = Vec::new();
        let mut off = 0;
        for (sp, &no) in self.nocc.iter().enumerate() {
            let nv = self.nmo - no;
            let k = MatRef::from_column_major_slice(&v[off..off + nv * no], nv, no);
            let f = mm(
                mm(c[sp].subcols(no, nv), k, Par::Seq).as_ref(),
                c[sp].subcols(0, no).transpose(),
                Par::Seq,
            );
            big = (0..f.ncols()).fold(big, |m, j| {
                (0..f.nrows()).fold(m, |m, i| m.max(f[(i, j)].abs()))
            });
            forms.push(f);
            off += nv * no;
        }
        'find: for f in &forms {
            for j in 0..f.ncols() {
                for i in 0..f.nrows() {
                    if f[(i, j)].abs() >= big * (1.0 - 1e-8) {
                        best = Some(f[(i, j)].signum());
                        break 'find;
                    }
                }
            }
        }
        best.unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{lowest_hessian_mode, RotatedGradient, Search, Starts};
    use super::*;
    use faer::Mat;
    use linus_base::linalg::{canonical_orthogonalizer, eigh, mat_from_rowmajor};
    use linus_ints::{AoMatrixMut, BasisDef, DerivOrder, OneEPlan, PairOptions, PairSet};

    /// A model orbital Hessian on the rotations of `c` (per spin, `nocc` occupied): A = sum_b P_b
    /// A0 P_b over the irreps of `sym` (A0: the gaps on the diagonal plus a seeded symmetric
    /// coupling), so A commutes with the group exactly; its "gradient" is linear, h A v.
    struct Model<'a> {
        c: Vec<Mat<f64>>,
        nocc: [usize; 2],
        sym: &'a PairSymmetry,
        a0: Mat<f64>,
    }

    impl Model<'_> {
        fn apply(&self, v: &[f64]) -> Vec<f64> {
            let mut out = vec![0.0; v.len()];
            for b in 0..self.sym.group().order() {
                let mut pv = v.to_vec();
                self.sym.project(b, &mut pv);
                let mut apv: Vec<f64> = (0..v.len())
                    .map(|i| (0..v.len()).map(|j| self.a0[(i, j)] * pv[j]).sum())
                    .collect();
                self.sym.project(b, &mut apv);
                out.iter_mut().zip(&apv).for_each(|(o, x)| *o += x);
            }
            out
        }
    }

    impl RotatedGradient for Model<'_> {
        type Error = ();
        fn nmo(&self) -> usize {
            self.c[0].ncols()
        }
        fn nocc(&self) -> [usize; 2] {
            self.nocc
        }
        fn orbital(&mut self, s: usize, p: usize) -> Vec<f64> {
            (0..self.c[s].nrows()).map(|r| self.c[s][(r, p)]).collect()
        }
        fn gradient(&mut self, v: &[f64], h: f64) -> Vec<f64> {
            self.apply(v).iter().map(|x| h * x).collect()
        }
        fn failure(&mut self) -> Option<()> {
            None
        }
        fn settle(&mut self) -> Result<bool, ()> {
            Ok(false)
        }
    }

    /// At an exactly symmetric point (water cation, C2v, the core Hamiltonian's orbitals) with an
    /// orbital Hessian that commutes with the group, the per-irrep search of the lowest pair's
    /// irrep (its products projected, the rest of the space at the shifted curvature) and the
    /// unprojected search from the same start (the irrep's wide start, which lies in the irrep)
    /// build the same Krylov space: the same eigenvalue to round-off. Both equal the block's dense
    /// lowest eigenvalue: the shift moves no real one.
    #[test]
    fn the_projected_search_equals_the_unprojected_one_at_exact_symmetry() {
        let z = [8, 1, 1];
        let pos = [[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]];
        let def = BasisDef::load("def2-svp", None).unwrap();
        let basis = BasisSet::new(&def, &z, &pos, true).unwrap();
        let n = basis.nbf;
        let pairs = PairSet::build(&basis, PairOptions::default());
        let one = OneEPlan::new(&basis, DerivOrder::Zero);
        let (mut s, mut t, mut v) = (vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]);
        one.overlap_kinetic(
            &mut AoMatrixMut::row_major(&mut s, n),
            &mut AoMatrixMut::row_major(&mut t, n),
        );
        let charges: Vec<(f64, [f64; 3])> =
            z.iter().zip(&pos).map(|(&q, &r)| (q as f64, r)).collect();
        one.with_charges(&pairs, &charges)
            .unwrap()
            .potential(&mut AoMatrixMut::row_major(&mut v, n));
        let s = mat_from_rowmajor(n, n, &s);
        let h = mat_from_rowmajor(n, n, &t) + mat_from_rowmajor(n, n, &v);
        let x = canonical_orthogonalizer(s.as_ref(), 1e-8, Par::Seq);
        let (eps, c) = crate::diagonalize(h.as_ref(), x.as_ref(), Par::Seq);
        let nocc = [5, 4];
        let ao = AoGroup::new(&z, &pos, &basis, DEFAULT_TOLERANCE);
        assert_eq!(ao.group.name, "C2v");
        let cr = [c.as_ref(), c.as_ref()];
        let sym = PairSymmetry::at(&ao, &cr, &nocc, s.as_ref());
        assert_eq!(sym.group().name, "C2v");
        let gaps: Vec<f64> = (0..2)
            .flat_map(|sp| {
                let (no, e) = (nocc[sp], &eps);
                (0..no).flat_map(move |i| (no..e.len()).map(move |a| e[a] - e[i]))
            })
            .collect();
        let dim = gaps.len();
        let mut seed = 12345u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 2000) as f64 / 1000.0 - 1.0
        };
        let mut a0 = Mat::from_fn(dim, dim, |i, j| if i == j { gaps[i] } else { 0.0 });
        for i in 0..dim {
            for j in 0..i {
                let x = 0.02 * rnd();
                a0[(i, j)] += x;
                a0[(j, i)] += x;
            }
        }
        let mut model = Model {
            c: vec![c.clone(), c.clone()],
            nocc,
            sym: &sym,
            a0,
        };
        let e: [&[f64]; 2] = [&eps, &eps];
        let with = Starts::new(&mut model, &e, Some(&ao), s.as_ref());
        let b0 = with.irreps[0].irrep;
        let start = with.irreps[0].wide.clone();
        let search = |irrep: Option<usize>| Search {
            products: 40,
            g0: None,
            guess: None,
            guess_first: false,
            irrep,
            stop_below: f64::NEG_INFINITY,
        };
        let projected = lowest_hessian_mode(&mut model, &with, search(Some(b0)));
        let nmo = model.nmo();
        let mut grad = |v: &[f64], h: f64| model.gradient(v, h);
        let plain =
            crate::stability::davidson(&mut grad, nmo, &e, &nocc, None, 40, None, 1.0, Some(start));
        assert!(
            (projected.theta - plain.0).abs() < 1e-12,
            "{} vs {}",
            projected.theta,
            plain.0
        );
        // the dense lowest eigenvalue of A in irrep b0
        let a = Mat::from_fn(dim, dim, |i, j| {
            let mut ej = vec![0.0; dim];
            ej[j] = 1.0;
            model.apply(&ej)[i]
        });
        let (w, u) = eigh(a.as_ref(), Par::Seq);
        let lowest_b0 = (0..dim)
            .filter(|&k| {
                let mut uk: Vec<f64> = (0..dim).map(|i| u[(i, k)]).collect();
                sym.project(b0, &mut uk);
                uk.iter().map(|x| x * x).sum::<f64>() > 0.5
            })
            .map(|k| w[k])
            .fold(f64::INFINITY, f64::min);
        assert!(
            (projected.theta - lowest_b0).abs() < 1e-5,
            "{} vs dense {lowest_b0}",
            projected.theta
        );
    }
}
