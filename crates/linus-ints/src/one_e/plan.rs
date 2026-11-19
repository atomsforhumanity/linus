//! The plan of the one-electron integrals of a basis: the overlap S, the kinetic energy T, the
//! dipole integrals, and, with point charges (the nuclei, external charges), their potential V;
//! with their nuclear first and second derivatives, as matrices or contracted with densities.
//!
//! [`OneEPlan::new`] fixes the basis and the highest derivative order; [`OneEPlan::with_charges`]
//! adds point charges and the pair set of their potential. Operations write into the caller's
//! buffers:
//! - matrices (order zero): [`OneEPlan::overlap_kinetic`], [`OneEPlan::overlap`],
//!   [`OneEPlan::dipole`], [`OneEPlan::potential`], as [`AoMatrixMut`] views;
//! - derivative matrices (order one): [`OneEPlan::overlap_kinetic_deriv`],
//!   [`OneEPlan::dipole_deriv`], [`OneEPlan::potential_deriv`], [`OneEPlan::hcore_deriv`];
//! - contractions with densities, over a caller-owned [`OneEWorkspace`]: gradients (order one,
//!   [`OneEPlan::overlap_kinetic_grad`], [`OneEPlan::dipole_grad`],
//!   [`OneEPlan::potential_grad`], [`OneEPlan::nuclear_attraction_grad`]) and Hessians (order
//!   two, [`OneEPlan::overlap_kinetic_hess`], [`OneEPlan::potential_hess`],
//!   [`OneEPlan::nuclear_attraction_hess`]).
//!
//! Conventions (as [`crate::one_e::deriv`]): densities are symmetric nbf x nbf row-major
//! slices; derivative matrices are (3 natom) x nbf x nbf row-major with perturbation index
//! x = 3 atom + k first; gradients are one [x, y, z] row per atom; Hessians (3 natom)^2
//! row-major. The potential's own coordinates (the charge positions) follow the atoms'
//! (slots natom + c); the nuclear-attraction operations fold charge c onto atom c.
//!
//! The contractions reduce over a fixed number of strided chunks of their shell pairs
//! (`linus_num::det::nchunk_for`, from the problem's size only), each chunk into its own
//! accumulator of the workspace's pool, added in chunk order: the results have the same bits
//! for any number of threads. The derivative matrices compute the pair blocks in parallel and
//! add them pair by pair (every element belongs to one pair).

use crate::basis::{BasisSet, Shell};
use crate::contract::AoMatrixMut;
use crate::engine::schedule::ordered_pool;
use crate::engine::WorkspaceSize;
use crate::error::IntsError;
use crate::one_e::deriv::{
    add_pair_block, dip_deriv1, pair_dot, shell_pairs, st_deriv1, st_deriv2, CartScratch,
};
use crate::one_e::{dipole_block, overlap_kinetic_block, scatter_sym_ao};
use crate::pairs::{DerivOrder, PairSet};
use crate::potential::{self, all_pairs, matrix_block, PointCharges, PotScratch};
use linus_num::det::{nchunk_for, DEFAULT_BUDGET};
use rayon::prelude::*;

/// Point charges (q, C) with the pair set of their potential.
#[derive(Clone, Copy)]
struct Charges<'a> {
    pairs: &'a PairSet,
    charges: &'a [(f64, [f64; 3])],
}

/// The one-electron integrals of a basis up to a derivative order (see the module docs).
pub struct OneEPlan<'a> {
    basis: &'a BasisSet,
    order: DerivOrder,
    /// every shell pair (i, j), i >= j, row by row (a pair set's order)
    all: Vec<(usize, usize)>,
    /// the pairs of shells on two atoms (the others' derivatives vanish)
    two: Vec<(usize, usize)>,
    charges: Option<Charges<'a>>,
    /// the largest angular momentum and function count of a shell
    lmax: usize,
    nf: usize,
}

/// The scratch of one chunk of an S, T or dipole contraction: two pair blocks of up to nine
/// components and the kernels' scratch, sized for the largest shell pair.
struct StScratch {
    s: Vec<f64>,
    t: Vec<f64>,
    cart: CartScratch,
}

/// The caller-owned memory of an [`OneEPlan`]'s contractions: the chunk accumulators of every
/// operation in one pool (sized for the largest), and per thread a scratch of the S, T and
/// dipole contractions and one of the potential's. Reusable across calls of its plan.
pub struct OneEWorkspace {
    pool: Vec<f64>,
    st: Vec<StScratch>,
    pot: Vec<PotScratch>,
    /// the scratches' sizes: functions of the largest shell, its angular momentum, the order
    nf: usize,
    lmax: usize,
    order: DerivOrder,
}

impl OneEWorkspace {
    /// A workspace with room for `values` accumulator values and the scratches of the current
    /// pool's threads.
    fn new(values: usize, nf: usize, lmax: usize, order: DerivOrder) -> Self {
        let mut ws = OneEWorkspace {
            pool: vec![0.0; values],
            st: Vec::new(),
            pot: Vec::new(),
            nf,
            lmax,
            order,
        };
        ws.grow();
        ws
    }

    /// A scratch of each kind for every thread of the current pool.
    fn grow(&mut self) {
        let nt = rayon::current_num_threads();
        let (nf, lmax, order) = (self.nf, self.lmax, self.order);
        let nb = 9 * nf * nf;
        if self.st.len() < nt {
            self.st.resize_with(nt, || StScratch {
                s: vec![0.0; nb],
                t: vec![0.0; nb],
                cart: CartScratch::reserved(lmax, nf),
            });
        }
        if self.pot.len() < nt {
            self.pot
                .resize_with(nt, || PotScratch::new(2 * lmax, order));
        }
    }

    fn carve_st(&mut self, nchunk: usize, len: usize) -> (&mut [f64], &mut [StScratch]) {
        assert!(
            self.pool.len() >= nchunk * len,
            "the workspace of another plan"
        );
        self.grow();
        (&mut self.pool[..nchunk * len], &mut self.st)
    }

    fn carve_pot(&mut self, nchunk: usize, len: usize) -> (&mut [f64], &mut [PotScratch]) {
        assert!(
            self.pool.len() >= nchunk * len,
            "the workspace of another plan"
        );
        self.grow();
        (&mut self.pool[..nchunk * len], &mut self.pot)
    }
}

/// The (chunk count, accumulator length) of each contraction of a kind.
type Chunks = Vec<(usize, usize)>;

/// The chunk count and accumulator length of a contraction over `n` items.
fn chunk(n: usize, len: usize) -> (usize, usize) {
    (nchunk_for(n, len, DEFAULT_BUDGET), len)
}

impl<'a> OneEPlan<'a> {
    /// The plan of the one-electron integrals of `basis` and their derivatives up to `order`.
    pub fn new(basis: &'a BasisSet, order: DerivOrder) -> Self {
        let sh = &basis.shells;
        let all = shell_pairs(sh.len());
        let two = all
            .iter()
            .copied()
            .filter(|&(i, j)| sh[i].atom != sh[j].atom)
            .collect();
        OneEPlan {
            basis,
            order,
            all,
            two,
            charges: None,
            lmax: basis.max_l(),
            nf: sh.iter().map(Shell::nfunc).max().unwrap_or(0),
        }
    }

    /// The plan with point charges (q, C) (the electrons see the potential -q / |r - C|) and the
    /// pair set of their potential (`pairs`, built from the plan's basis; expansions of the
    /// plan's order or higher). Refuses a pair set of a lower order.
    pub fn with_charges(
        mut self,
        pairs: &'a PairSet,
        charges: &'a [(f64, [f64; 3])],
    ) -> Result<Self, IntsError> {
        if pairs.deriv_order() < self.order {
            return Err(IntsError::Order {
                need: self.order,
                have: pairs.deriv_order(),
            });
        }
        self.charges = Some(Charges { pairs, charges });
        Ok(self)
    }

    /// The plan's derivative order.
    pub fn order(&self) -> DerivOrder {
        self.order
    }

    fn charges(&self) -> Charges<'a> {
        self.charges
            .expect("the potential needs point charges (OneEPlan::with_charges)")
    }

    fn need(&self, order: DerivOrder) {
        assert!(
            self.order >= order,
            "the operation needs a plan of order {order:?}, not {:?}",
            self.order
        );
    }

    /// The coordinate slots of the potential: the atoms, then the charges.
    fn nslots(&self) -> usize {
        self.basis.natom() + self.charges.map_or(0, |c| c.charges.len())
    }

    /// The contractions of the plan's order and below: (chunks, accumulator length) of the S,
    /// T and dipole ones and of the potential's.
    fn contractions(&self) -> (Chunks, Chunks) {
        let natom = self.basis.natom();
        let (mut st, mut pot) = (Vec::new(), Vec::new());
        if self.order >= DerivOrder::One {
            st.push(chunk(self.two.len(), 3 * natom));
            st.push(chunk(self.all.len(), 9 * natom));
            if let Some(c) = self.charges {
                pot.push(potential::chunks(
                    c.pairs.len(),
                    self.nslots(),
                    DerivOrder::One,
                ));
            }
        }
        if self.order == DerivOrder::Two {
            st.push(chunk(self.two.len(), 9 * natom * natom));
            if let Some(c) = self.charges {
                pot.push(potential::chunks(
                    c.pairs.len(),
                    self.nslots(),
                    DerivOrder::Two,
                ));
            }
        }
        (st, pot)
    }

    /// (pool values, S/T scratches, potential scratches) of a workspace.
    fn pool_size(&self) -> (usize, usize, usize) {
        let (st, pot) = self.contractions();
        let values = st.iter().chain(&pot).map(|(c, l)| c * l).max().unwrap_or(0);
        let nst = st.iter().map(|x| x.0).max().unwrap_or(0);
        let npot = pot.iter().map(|x| x.0).max().unwrap_or(0);
        (values, nst, npot)
    }

    /// The bytes of a workspace for the contractions of the plan's order and below: the
    /// scratches per thread, the chunk accumulators shared.
    pub fn workspace_size(&self) -> WorkspaceSize {
        let (values, _, _) = self.pool_size();
        let st = 8 * 18 * self.nf * self.nf + CartScratch::bytes(self.lmax, self.nf);
        WorkspaceSize {
            per_thread: st + PotScratch::bytes(2 * self.lmax, self.order),
            shared: 8 * values,
        }
    }

    /// A workspace for the contractions of the plan's order and below.
    pub fn workspace(&self) -> OneEWorkspace {
        let (values, _, _) = self.pool_size();
        OneEWorkspace::new(values, self.nf, self.lmax, self.order)
    }

    // ---------------------------------------------------------------------------------------
    // Matrices
    // ---------------------------------------------------------------------------------------

    /// The S and T blocks of every shell pair, computed in parallel, in pair order.
    fn st_blocks(&self) -> Vec<(usize, usize, Vec<f64>, Vec<f64>)> {
        let sh = &self.basis.shells;
        (0..sh.len())
            .into_par_iter()
            .flat_map_iter(|i| {
                (0..=i).map(move |j| {
                    let (a, b) = (&sh[i], &sh[j]);
                    let n = a.nfunc() * b.nfunc();
                    let mut s = vec![0.0; n];
                    let mut t = vec![0.0; n];
                    overlap_kinetic_block(a, b, &mut s, &mut t);
                    (i, j, s, t)
                })
            })
            .collect()
    }

    /// The overlap and kinetic-energy matrices into `s` and `t` (overwritten).
    pub fn overlap_kinetic(&self, s: &mut AoMatrixMut<'_>, t: &mut AoMatrixMut<'_>) {
        let sh = &self.basis.shells;
        for (i, j, sb, tb) in self.st_blocks() {
            let (a, b) = (&sh[i], &sh[j]);
            let (o, n) = ((a.ao_start, b.ao_start), (a.nfunc(), b.nfunc()));
            scatter_sym_ao(s, o, n, &sb);
            scatter_sym_ao(t, o, n, &tb);
        }
    }

    /// The overlap matrix into `s` (overwritten).
    pub fn overlap(&self, s: &mut AoMatrixMut<'_>) {
        let sh = &self.basis.shells;
        for (i, j, sb, _) in self.st_blocks() {
            let (a, b) = (&sh[i], &sh[j]);
            scatter_sym_ao(s, (a.ao_start, b.ao_start), (a.nfunc(), b.nfunc()), &sb);
        }
    }

    /// The dipole integrals <a| (r - origin)_k |b> for k = x, y, z into `out` (overwritten).
    pub fn dipole(&self, origin: [f64; 3], out: &mut [AoMatrixMut<'_>; 3]) {
        let sh = &self.basis.shells;
        let mut blk = Vec::new();
        for &(i, j) in &self.all {
            let (a, b) = (&sh[i], &sh[j]);
            let (o, n) = ((a.ao_start, b.ao_start), (a.nfunc(), b.nfunc()));
            blk.resize(3 * n.0 * n.1, 0.0);
            dipole_block(a, b, origin, &mut blk);
            for (d, m) in out.iter_mut().enumerate() {
                scatter_sym_ao(m, o, n, &blk[d * n.0 * n.1..(d + 1) * n.0 * n.1]);
            }
        }
    }

    /// The potential of the point charges, sum_C -q_C <a| 1/|r - C| |b>, into `v`
    /// (overwritten).
    pub fn potential(&self, v: &mut AoMatrixMut<'_>) {
        let Charges { pairs, charges } = self.charges();
        let sh = &self.basis.shells;
        let src = PointCharges {
            charges,
            first_slot: self.basis.natom(),
        };
        let blocks: Vec<(usize, Vec<f64>)> = pairs
            .pairs()
            .par_iter()
            .enumerate()
            .map(|(k, pr)| {
                let mut out = vec![0.0; pr.nab()];
                matrix_block(pr, &src, &mut out);
                (k, out)
            })
            .collect();
        for (k, blk) in blocks {
            let pr = pairs.pair(k);
            let (a, b) = (&sh[pr.sa], &sh[pr.sb]);
            scatter_sym_ao(v, (a.ao_start, b.ao_start), (pr.na, pr.nb), &blk);
        }
    }

    // ---------------------------------------------------------------------------------------
    // Derivative matrices
    // ---------------------------------------------------------------------------------------

    /// The overlap and kinetic derivative matrices dS/dX and dT/dX, each (3 natom) x nbf x nbf
    /// (overwritten; either may be left out). An order-One plan or higher.
    pub fn overlap_kinetic_deriv(&self, mut ds: Option<&mut [f64]>, mut dt: Option<&mut [f64]>) {
        self.need(DerivOrder::One);
        let nbf = self.basis.nbf;
        let n2 = nbf * nbf;
        let len = 3 * self.basis.natom() * n2;
        for m in [ds.as_deref_mut(), dt.as_deref_mut()].into_iter().flatten() {
            assert_eq!(m.len(), len, "a derivative matrix is (3 natom) x nbf x nbf");
            m.fill(0.0);
        }
        let sh = &self.basis.shells;
        let blocks: Vec<(usize, usize, Vec<f64>, Vec<f64>)> = self
            .two
            .par_iter()
            .map(|&(i, j)| {
                let n = sh[i].nfunc() * sh[j].nfunc();
                let mut s = vec![0.0; 3 * n];
                let mut t = vec![0.0; 3 * n];
                st_deriv1(&sh[i], &sh[j], &mut s, &mut t, &mut CartScratch::default());
                (i, j, s, t)
            })
            .collect();
        for (i, j, s, t) in blocks {
            let (a, b) = (&sh[i], &sh[j]);
            let (na, nb) = (a.nfunc(), b.nfunc());
            for k in 0..3 {
                for (atom, f) in [(a.atom, 1.0), (b.atom, -1.0)] {
                    let x = 3 * atom + k;
                    if let Some(m) = ds.as_deref_mut() {
                        add_pair_block(
                            &mut m[x * n2..(x + 1) * n2],
                            nbf,
                            a.ao_start,
                            b.ao_start,
                            na,
                            nb,
                            false,
                            &s[k * na * nb..(k + 1) * na * nb],
                            f,
                        );
                    }
                    if let Some(m) = dt.as_deref_mut() {
                        add_pair_block(
                            &mut m[x * n2..(x + 1) * n2],
                            nbf,
                            a.ao_start,
                            b.ao_start,
                            na,
                            nb,
                            false,
                            &t[k * na * nb..(k + 1) * na * nb],
                            f,
                        );
                    }
                }
            }
        }
    }

    /// The dipole derivative matrices into `out` (overwritten):
    /// out[((3 atom + k) * 3 + c) * nbf² + m * nbf + n] = d<m|(r - O)_c|n>/dX_{atom,k}. An
    /// order-One plan or higher.
    pub fn dipole_deriv(&self, origin: [f64; 3], out: &mut [f64]) {
        self.need(DerivOrder::One);
        let nbf = self.basis.nbf;
        let n2 = nbf * nbf;
        assert_eq!(out.len(), 9 * self.basis.natom() * n2);
        out.fill(0.0);
        let sh = &self.basis.shells;
        let blocks: Vec<(usize, usize, Vec<f64>, Vec<f64>)> = self
            .all
            .par_iter()
            .map(|&(i, j)| {
                let n = sh[i].nfunc() * sh[j].nfunc();
                let mut da = vec![0.0; 9 * n];
                let mut db = vec![0.0; 9 * n];
                let sc = &mut CartScratch::default();
                dip_deriv1(&sh[i], &sh[j], origin, (&mut da, &mut db), sc);
                (i, j, da, db)
            })
            .collect();
        for (i, j, da, db) in blocks {
            let (a, b) = (&sh[i], &sh[j]);
            let (na, nb) = (a.nfunc(), b.nfunc());
            let n = na * nb;
            for kc in 0..9 {
                for (atom, blk) in [(a.atom, &da), (b.atom, &db)] {
                    let x = 3 * atom * 3 + kc;
                    add_pair_block(
                        &mut out[x * n2..(x + 1) * n2],
                        nbf,
                        a.ao_start,
                        b.ao_start,
                        na,
                        nb,
                        i == j,
                        &blk[kc * n..(kc + 1) * n],
                        1.0,
                    );
                }
            }
        }
    }

    /// The derivative matrices of the point charges' potential, (3 (natom + ncharge)) x nbf²
    /// row-major (overwritten): the basis-function centers' coordinates (3 natom), then the
    /// charge positions' (3 ncharge). An order-One plan or higher.
    pub fn potential_deriv(&self, out: &mut [f64]) {
        self.need(DerivOrder::One);
        let Charges { pairs, charges } = self.charges();
        let natom = self.basis.natom();
        let src = PointCharges {
            charges,
            first_slot: natom,
        };
        let nbf = self.basis.nbf;
        assert_eq!(out.len(), 3 * self.nslots() * nbf * nbf);
        out.fill(0.0);
        potential::add_deriv_matrices(self.basis, &all_pairs(pairs), &src, out);
    }

    /// The core-Hamiltonian derivative matrices dh/dX = dT/dX + dV/dX into `out`
    /// ((3 natom) x nbf x nbf, overwritten), the point charges being the nuclei (charge i on
    /// atom i; the operator-center terms included). An order-One plan or higher.
    pub fn hcore_deriv(&self, out: &mut [f64]) {
        let natom = self.basis.natom();
        assert_eq!(
            self.charges().charges.len(),
            natom,
            "the core Hamiltonian's charges are the nuclei"
        );
        self.overlap_kinetic_deriv(None, Some(&mut *out));
        let nbf = self.basis.nbf;
        let mut dv = vec![0.0; 3 * self.nslots() * nbf * nbf];
        self.potential_deriv(&mut dv);
        let (dva, dvc) = dv.split_at(out.len());
        for ((x, a), b) in out.iter_mut().zip(dva).zip(dvc) {
            *x += a + b;
        }
    }

    // ---------------------------------------------------------------------------------------
    // Contractions
    // ---------------------------------------------------------------------------------------

    /// The gradient sum_ab Ws_ab dS_ab/dX + sum_ab Dt_ab dT_ab/dX into `out` (natom rows,
    /// overwritten; either density may be left out). An order-One plan or higher.
    pub fn overlap_kinetic_grad(
        &self,
        ws: Option<&[f64]>,
        dt: Option<&[f64]>,
        out: &mut [[f64; 3]],
        wk: &mut OneEWorkspace,
    ) {
        self.need(DerivOrder::One);
        let natom = self.basis.natom();
        assert_eq!(out.len(), natom, "the gradient has natom rows");
        let nbf = self.basis.nbf;
        let sh = &self.basis.shells;
        let two = &self.two[..];
        let (nchunk, len) = chunk(two.len(), 3 * natom);
        let (pool, scratch) = wk.carve_st(nchunk, len);
        ordered_pool(two.len(), pool, len, nchunk, scratch, |acc, sc, idx| {
            let g = acc.as_chunks_mut::<3>().0;
            let (i, j) = two[idx];
            let (a, b) = (&sh[i], &sh[j]);
            let (na, nb) = (a.nfunc(), b.nfunc());
            let n = na * nb;
            let StScratch { s, t, cart } = sc;
            let (s, t) = (&mut s[..3 * n], &mut t[..3 * n]);
            st_deriv1(a, b, s, t, cart);
            for k in 0..3 {
                let mut v = 0.0;
                if let Some(w) = ws {
                    v += pair_dot(
                        w,
                        nbf,
                        a.ao_start,
                        b.ao_start,
                        na,
                        nb,
                        &s[k * n..(k + 1) * n],
                    );
                }
                if let Some(d) = dt {
                    v += pair_dot(
                        d,
                        nbf,
                        a.ao_start,
                        b.ao_start,
                        na,
                        nb,
                        &t[k * n..(k + 1) * n],
                    );
                }
                // i != j always here (different atoms): factor 2 for the (b, a) block
                g[a.atom][k] += 2.0 * v;
                g[b.atom][k] -= 2.0 * v;
            }
        });
        out.as_flattened_mut().copy_from_slice(&pool[..len]);
    }

    /// The Hessian sum_ab Ws_ab d²S_ab/dXdY + sum_ab Dt_ab d²T_ab/dXdY into `out`
    /// ((3 natom)^2 row-major, overwritten). An order-Two plan.
    pub fn overlap_kinetic_hess(
        &self,
        ws: Option<&[f64]>,
        dt: Option<&[f64]>,
        out: &mut [f64],
        wk: &mut OneEWorkspace,
    ) {
        self.need(DerivOrder::Two);
        let n3 = 3 * self.basis.natom();
        assert_eq!(out.len(), n3 * n3, "the Hessian is (3 natom)^2");
        let nbf = self.basis.nbf;
        let sh = &self.basis.shells;
        let two = &self.two[..];
        let (nchunk, len) = chunk(two.len(), n3 * n3);
        let (pool, scratch) = wk.carve_st(nchunk, len);
        ordered_pool(two.len(), pool, len, nchunk, scratch, |h, sc, idx| {
            let (i, j) = two[idx];
            let (a, b) = (&sh[i], &sh[j]);
            let (na, nb) = (a.nfunc(), b.nfunc());
            let n = na * nb;
            let StScratch { s, t, cart } = sc;
            let (s, t) = (&mut s[..9 * n], &mut t[..9 * n]);
            st_deriv2(a, b, s, t, cart);
            let (x, y) = (a.atom, b.atom);
            for kl in 0..9 {
                let mut v = 0.0;
                if let Some(w) = ws {
                    v += pair_dot(
                        w,
                        nbf,
                        a.ao_start,
                        b.ao_start,
                        na,
                        nb,
                        &s[kl * n..(kl + 1) * n],
                    );
                }
                if let Some(d) = dt {
                    v += pair_dot(
                        d,
                        nbf,
                        a.ao_start,
                        b.ao_start,
                        na,
                        nb,
                        &t[kl * n..(kl + 1) * n],
                    );
                }
                v *= 2.0;
                let (k, l) = (kl / 3, kl % 3);
                // AA = BB = v, AB = BA = -v
                h[(3 * x + k) * n3 + 3 * x + l] += v;
                h[(3 * y + k) * n3 + 3 * y + l] += v;
                h[(3 * x + k) * n3 + 3 * y + l] -= v;
                h[(3 * y + l) * n3 + 3 * x + k] -= v;
            }
        });
        out.copy_from_slice(&pool[..len]);
    }

    /// sum_ab D_ab d<a|(r - O)_c|b>/dX_k per atom into `out` (out\[atom\]\[k\]\[c\],
    /// overwritten). An order-One plan or higher.
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    pub fn dipole_grad(
        &self,
        origin: [f64; 3],
        d: &[f64],
        out: &mut [[[f64; 3]; 3]],
        wk: &mut OneEWorkspace,
    ) {
        self.need(DerivOrder::One);
        let natom = self.basis.natom();
        assert_eq!(out.len(), natom, "the dipole gradient has natom rows");
        let nbf = self.basis.nbf;
        let sh = &self.basis.shells;
        let all = &self.all[..];
        let (nchunk, len) = chunk(all.len(), 9 * natom);
        let (pool, scratch) = wk.carve_st(nchunk, len);
        ordered_pool(all.len(), pool, len, nchunk, scratch, |acc, sc, idx| {
            let g = acc.as_chunks_mut::<3>().0.as_chunks_mut::<3>().0;
            let (i, j) = all[idx];
            let (a, b) = (&sh[i], &sh[j]);
            let (na, nb) = (a.nfunc(), b.nfunc());
            let n = na * nb;
            let StScratch { s: da, t: db, cart } = sc;
            let (da, db) = (&mut da[..9 * n], &mut db[..9 * n]);
            dip_deriv1(a, b, origin, (&mut *da, &mut *db), cart);
            let f = if i == j { 1.0 } else { 2.0 };
            for k in 0..3 {
                for c in 0..3 {
                    let kc = k * 3 + c;
                    let (o, blk) = ((a.ao_start, b.ao_start), kc * n..(kc + 1) * n);
                    g[a.atom][k][c] += f * pair_dot(d, nbf, o.0, o.1, na, nb, &da[blk.clone()]);
                    g[b.atom][k][c] += f * pair_dot(d, nbf, o.0, o.1, na, nb, &db[blk]);
                }
            }
        });
        out.as_flattened_mut()
            .as_flattened_mut()
            .copy_from_slice(&pool[..len]);
    }

    /// The gradient of sum_ab D_ab V_ab with V the point charges' potential into the
    /// workspace's pool (one row per slot: the atoms, then the charges).
    fn potential_grad_pool<'w>(&self, d: &[f64], wk: &'w mut OneEWorkspace) -> &'w [f64] {
        self.need(DerivOrder::One);
        let Charges { pairs, charges } = self.charges();
        let src = PointCharges {
            charges,
            first_slot: self.basis.natom(),
        };
        let (nchunk, len) = potential::chunks(pairs.len(), self.nslots(), DerivOrder::One);
        let (pool, scratch) = wk.carve_pot(nchunk, len);
        potential::gradient_fold(
            self.basis,
            &all_pairs(pairs),
            &src,
            d,
            pool,
            nchunk,
            scratch,
        );
        &pool[..len]
    }

    /// The gradient of sum_ab D_ab V_ab with V the point charges' potential into `out`
    /// (overwritten): natom rows for the basis-function centers, then one row per charge. An
    /// order-One plan or higher.
    pub fn potential_grad(&self, d: &[f64], out: &mut [[f64; 3]], wk: &mut OneEWorkspace) {
        assert_eq!(out.len(), self.nslots(), "one row per atom and per charge");
        out.as_flattened_mut()
            .copy_from_slice(self.potential_grad_pool(d, wk));
    }

    /// The nuclear-attraction gradient sum_ab D_ab dV_ab/dX into `out` (natom rows,
    /// overwritten), the point charges being the nuclei (charge i on atom i, moving with it).
    /// An order-One plan or higher.
    pub fn nuclear_attraction_grad(&self, d: &[f64], out: &mut [[f64; 3]], wk: &mut OneEWorkspace) {
        let natom = self.basis.natom();
        assert_eq!(
            self.charges().charges.len(),
            natom,
            "the charges are the nuclei"
        );
        assert_eq!(out.len(), natom, "the gradient has natom rows");
        let g = self.potential_grad_pool(d, wk).as_chunks::<3>().0;
        let (ga, gc) = g.split_at(natom);
        for ((o, a), c) in out.iter_mut().zip(ga).zip(gc) {
            for k in 0..3 {
                o[k] = a[k] + c[k];
            }
        }
    }

    /// The Hessian of sum_ab D_ab V_ab with V the point charges' potential into the
    /// workspace's pool, over the atoms' coordinates followed by the charges'.
    fn potential_hess_pool<'w>(&self, d: &[f64], wk: &'w mut OneEWorkspace) -> &'w [f64] {
        self.need(DerivOrder::Two);
        let Charges { pairs, charges } = self.charges();
        let src = PointCharges {
            charges,
            first_slot: self.basis.natom(),
        };
        let (nchunk, len) = potential::chunks(pairs.len(), self.nslots(), DerivOrder::Two);
        let (pool, scratch) = wk.carve_pot(nchunk, len);
        potential::hessian_fold(
            self.basis,
            &all_pairs(pairs),
            &src,
            d,
            pool,
            nchunk,
            scratch,
        );
        &pool[..len]
    }

    /// The Hessian of sum_ab D_ab V_ab with V the point charges' potential into `out`
    /// ((3 (natom + ncharge))^2 row-major, overwritten): the atoms' coordinates, then the
    /// charge positions'. An order-Two plan.
    pub fn potential_hess(&self, d: &[f64], out: &mut [f64], wk: &mut OneEWorkspace) {
        let n3 = 3 * self.nslots();
        assert_eq!(out.len(), n3 * n3, "the Hessian is (3 (natom + ncharge))^2");
        out.copy_from_slice(self.potential_hess_pool(d, wk));
    }

    /// The nuclear-attraction Hessian into `out` ((3 natom)^2 row-major, overwritten), the
    /// point charges being the nuclei (charge i on atom i). An order-Two plan.
    pub fn nuclear_attraction_hess(&self, d: &[f64], out: &mut [f64], wk: &mut OneEWorkspace) {
        let natom = self.basis.natom();
        assert_eq!(
            self.charges().charges.len(),
            natom,
            "the charges are the nuclei"
        );
        let m = 3 * natom;
        assert_eq!(out.len(), m * m, "the Hessian is (3 natom)^2");
        let h = self.potential_hess_pool(d, wk);
        let n3 = 2 * m;
        let map = |i: usize| if i < m { i } else { i - m };
        out.fill(0.0);
        for i in 0..n3 {
            for j in 0..n3 {
                out[map(i) * m + map(j)] += h[i * n3 + j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisDef;
    use crate::pairs::PairOptions;

    const POS: [[f64; 3]; 3] = [[0.0, 0.0, 0.2], [1.4, 0.3, -0.9], [-1.5, -0.1, -0.8]];

    fn basis() -> BasisSet {
        let def = BasisDef::load("def2-tzvp", None).unwrap();
        BasisSet::new(&def, &[8, 1, 1], &POS, true).unwrap()
    }

    fn density(n: usize, f: f64) -> Vec<f64> {
        let g = |i: usize, j: usize| f / (1.0 + i.abs_diff(j) as f64);
        (0..n * n).map(|x| g(x / n, x % n)).collect()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn caps(ws: &OneEWorkspace) -> Vec<usize> {
        let st = ws.st.iter().flat_map(|s| {
            let own = [s.s.capacity(), s.t.capacity()];
            own.into_iter().chain(s.cart.caps())
        });
        let pot = ws.pot.iter().flat_map(|p| p.caps());
        st.chain(pot).chain([ws.pool.capacity()]).collect()
    }

    /// Every contraction has the same bits at 1, 3 and 8 threads and on a reused workspace,
    /// which never grows (def2-TZVP, f shells); the contractions agree with the derivative
    /// matrices they contract; the nuclear-attraction gradient is translationally invariant.
    #[test]
    fn contractions_reuse_bits_and_agree_with_matrices() {
        let b = basis();
        let (n, natom) = (b.nbf, b.natom());
        let (d, w) = (density(n, 0.02), density(n, -0.013));
        let pairs = PairSet::build(&b, PairOptions::with_order(DerivOrder::Two));
        let charges: Vec<(f64, [f64; 3])> = [8.0, 1.0, 1.0].into_iter().zip(POS).collect();
        let plan = OneEPlan::new(&b, DerivOrder::Two)
            .with_charges(&pairs, &charges)
            .unwrap();
        // the workspace holds a scratch per thread of the pool it is made in: made in the largest
        // pool the runs use (8 threads), it must not grow, whatever the machine's CPU count (the
        // global pool has 4 threads on a 4-CPU CI runner)
        let pool = |nt| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap()
        };
        let mut ws = pool(8).install(|| plan.workspace());
        let before = caps(&ws);
        let n3 = 3 * natom;
        let run = |ws: &mut OneEWorkspace| -> Vec<f64> {
            let mut v = Vec::new();
            let mut g = vec![[0.0; 3]; natom];
            plan.overlap_kinetic_grad(Some(&w), Some(&d), &mut g, ws);
            v.extend(g.as_flattened());
            plan.nuclear_attraction_grad(&d, &mut g, ws);
            v.extend(g.as_flattened());
            let mut gp = vec![[0.0; 3]; 2 * natom];
            plan.potential_grad(&d, &mut gp, ws);
            v.extend(gp.as_flattened());
            let mut gd = vec![[[0.0; 3]; 3]; natom];
            plan.dipole_grad([0.1, -0.2, 0.3], &d, &mut gd, ws);
            v.extend(gd.as_flattened().as_flattened());
            let mut h = vec![0.0; n3 * n3];
            plan.overlap_kinetic_hess(Some(&w), Some(&d), &mut h, ws);
            v.extend(&h);
            plan.nuclear_attraction_hess(&d, &mut h, ws);
            v.extend(&h);
            let mut hp = vec![0.0; 4 * n3 * n3];
            plan.potential_hess(&d, &mut hp, ws);
            v.extend(&hp);
            v
        };
        let mut runs = Vec::new();
        for nt in [1, 3, 8, 8] {
            runs.push(pool(nt).install(|| run(&mut ws)));
        }
        assert_eq!(before, caps(&ws), "the workspace grew");
        for r in &runs[1..] {
            assert!(r
                .iter()
                .zip(&runs[0])
                .all(|(x, y)| x.to_bits() == y.to_bits()));
        }
        // the gradients against the derivative matrices
        let n2 = n * n;
        let (mut ds, mut dt, mut hx) = (vec![0.0; n3 * n2], vec![0.0; n3 * n2], vec![0.0; n3 * n2]);
        plan.overlap_kinetic_deriv(Some(&mut ds), Some(&mut dt));
        plan.hcore_deriv(&mut hx);
        let (gst, gna) = (&runs[0][..n3], &runs[0][n3..2 * n3]);
        for x in 0..n3 {
            let m = x * n2..(x + 1) * n2;
            let st = dot(&w, &ds[m.clone()]) + dot(&d, &dt[m.clone()]);
            let dv: Vec<f64> = hx[m.clone()]
                .iter()
                .zip(&dt[m])
                .map(|(h, t)| h - t)
                .collect();
            let na = dot(&d, &dv);
            assert!(
                (gst[x] - st).abs() < 1e-10 * st.abs().max(1.0),
                "{x}: {} {st}",
                gst[x]
            );
            assert!(
                (gna[x] - na).abs() < 1e-10 * na.abs().max(1.0),
                "{x}: {} {na}",
                gna[x]
            );
        }
        assert!(gst.iter().any(|x| x.abs() > 1e-3) && gna.iter().any(|x| x.abs() > 1e-3));
        for k in 0..3 {
            let s: f64 = (0..natom).map(|a| gna[3 * a + k]).sum();
            assert!(s.abs() < 1e-9, "translation {k}: {s}");
        }
    }
}
