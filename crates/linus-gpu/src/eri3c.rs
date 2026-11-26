//! Three-center integrals (mn|P) and their first derivatives built on the device, with the
//! generated fused kernels of [`crate::kernels::eri3c_gen`] (tools/intgen/cuda_eri3c.py), for
//! the Coulomb operator 1/r and the long-range operator erf(omega r)/r.
//!
//! The host side ([`Eri3cHost`], no GPU needed) takes an orbital basis, an auxiliary basis and a
//! list of orbital shell pairs with their output rows, and packs per class (l_a >= l_b) the pair
//! primitives (exponent, centre, Schwarz bound, the nonzeros of the Hermite expansions) and per
//! auxiliary angular momentum the primitives. The device side (`Eri3c`, feature `gpu`) uploads
//! that once, compiles the classes' kernels (NVRTC, disk cache) and fills, for any range of the
//! pairs, a row-major block `out[(row - row(first pair)) * naux + P]`:
//!
//! - [`Eri3cKind::Integrals`]: (mn|P), the rows of linus-ints' `pair_block` stacked over pairs
//!   (the layout of a density-fitting tensor with one row per function pair);
//! - [`Eri3cKind::Derivatives`]: six planes of that shape, plane `3 c + x` holding
//!   d/dX_x (mn|P) for X the centre of the pair's first shell (c = 0) or second shell (c = 1).
//!   The auxiliary centre's derivative is -(plane x + plane 3 + x) (translational invariance).
//!
//! The operator: every launch (`Eri3c::compute`, `Eri3c::compute_block`, `Eri3c::gradient`,
//! `Eri3c::hessian`) takes a linus [`Operator`]: 1/r, or erf(omega r)/r for any omega > 0
//! (a kernel argument; `Eri3c::prepare_operator` compiles those kernels once per builder).
//! Both use the same packed data, kernel sources and schedules: only the Boys stage differs
//! (the generator's `ERI3C_OPERATOR` step; linus-ints' `prim_hermite` with omega). A
//! combination a/r + b erf(omega r)/r is two builds of one builder (the integrals are linear in
//! the operator). Screening uses the Coulomb bounds, which bound the erf integrals too (as the
//! CPU's `pair_block` with omega).
//!
//! Every output element has one writing thread, so results are bitwise reproducible run to run.
//! Values agree with linus-ints' CPU kernels to round-off (different summation order; auxiliary
//! primitives of a contracted shell are summed before the recursion, which the CPU's lanes
//! cannot do).
//!
//! Screening: a (pair primitive, auxiliary primitive) term is skipped when the product of their
//! Schwarz bounds is below `prim_thresh` (0 computes everything). For derivatives the pair bound
//! is that of the derivative expansions (triangle inequality over Hermite functions, as
//! linus-ints' `dgrad`).

use crate::cpu::{
    boys, build_pair, eb_prim, herm_idx, herm_tuv, nherm, pair_data, r_tensor, Aux3c, BasisSet,
    Operator, Pair3c, Shell, ShellView, ERI_PREF, PRIM_PAIR_CUT,
};
use crate::error::GpuError;
use crate::kernels::eri3c_class::{nfunc, ClassShape, AUX_REC};
use rayon::prelude::*;

/// Shell pairs per parallel step of [`Eri3cHost`]'s packing (the records of a step are held
/// twice for a moment: per pair, then in their classes).
const PACK_CHUNK: usize = 4096;

/// The operators the generated kernels compute: None for 1/r, Some(omega) for erf(omega r)/r
/// (omega > 0). Anything else (a combination a/r + b erf(omega r)/r, a scaled operator, the
/// zero operator) is `GpuError::NotCovered`: build its parts and combine them.
pub fn kernel_operator(op: &Operator) -> Result<Option<f64>, GpuError> {
    if op.is_coulomb() {
        Ok(None)
    } else if op.coulomb() == 0.0
        && op.long_range() == 1.0
        && op.omega() > 0.0
        && op.omega().is_finite()
    {
        Ok(Some(op.omega()))
    } else {
        Err(GpuError::NotCovered(format!(
            "eri3c: operator {op:?} (the kernels compute 1/r and erf(omega r)/r with omega > 0)"
        )))
    }
}

/// What a builder computes (the generated kernels' kinds).
pub use crate::kernels::eri3c_class::Eri3cKind;

/// An orbital shell pair of the output and the output row of its function pair (0, 0); the
/// pair's rows are row + ia nb + ib (ia in shell `sa`, ib in shell `sb`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairRow {
    pub sa: usize,
    pub sb: usize,
    pub row: usize,
}

/// Pair data of one orbital class (la >= lb) in kernel order.
#[derive(Clone, Debug, Default)]
pub struct HostClass {
    pub la: usize,
    pub lb: usize,
    /// Indices of the class's pairs in the builder's pair list, ascending.
    pub pairs: Vec<usize>,
    /// Primitive offsets (pairs + 1) into the records.
    pub p0: Vec<i32>,
    /// Records of `rec` values per primitive: p, P, q, E nonzeros of every set.
    pub d: Vec<f64>,
    /// Per pair: output row of (0, 0), row strides of the kernel's first and second index,
    /// swap flag (the kernel's first shell is the pair's second).
    pub out: Vec<i64>,
    /// Per pair: the atoms of shells `sa` and `sb`.
    pub atom: Vec<i32>,
    pub rec: usize,
}

/// Auxiliary primitives of one angular momentum.
#[derive(Clone, Debug, Default)]
pub struct HostAux {
    pub l: usize,
    /// Primitive offsets (shells + 1).
    pub p0: Vec<i32>,
    /// Per primitive: a, c (2a)^-l (-1)^l / a (the kernels take 1/a from it), A, q.
    pub d: Vec<f64>,
    /// First auxiliary function of each shell.
    pub col: Vec<i32>,
    /// Atom of each shell.
    pub atom: Vec<i32>,
}

impl HostAux {
    pub fn nshell(&self) -> usize {
        self.col.len()
    }
}

/// The packed host data of a three-center build (see the module docs).
#[derive(Clone, Debug)]
pub struct Eri3cHost {
    pub kind: Eri3cKind,
    pub naux: usize,
    pub pairs: Vec<PairRow>,
    /// Output rows: pairs\[i\].row .. + nab(i); `row_end` is the end of the last pair.
    pub row_end: usize,
    pub classes: Vec<HostClass>,
    pub aux: Vec<HostAux>,
    pub prim_thresh: f64,
    /// Atoms (gradients have 3 natom components).
    pub natom: usize,
}

/// The basis set a shell belongs to: the generated classes cover l <= 4 on the orbital side and
/// l <= 6 on the auxiliary side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Orbital,
    Auxiliary,
}

impl Role {
    fn lmax(self) -> usize {
        match self {
            Role::Orbital => 4,
            Role::Auxiliary => 6,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Role::Orbital => "orbital basis",
            Role::Auxiliary => "auxiliary basis",
        }
    }
}

/// Whether the generated kernels cover the orbital basis `basis` and the auxiliary basis `aux`
/// (every shell spherical from l = 2 on, l within its role's classes), else why not: a caller
/// plans with it, so that the packing's contract errors are never reached.
pub fn covers(basis: &BasisSet, aux: &BasisSet) -> Result<(), String> {
    let why = |e: GpuError| match e {
        GpuError::Contract(w) => w,
        e => e.to_string(),
    };
    for s in &basis.shells {
        check_pure(s, Role::Orbital).map_err(why)?;
    }
    for s in &aux.shells {
        check_pure(s, Role::Auxiliary).map_err(why)?;
    }
    Ok(())
}

/// A shell the generated kernels cover: spherical from l = 2 on, l within its role's classes.
fn check_pure(s: &Shell, role: Role) -> Result<(), GpuError> {
    if s.l >= 2 && !s.pure {
        return Err(GpuError::Contract(format!(
            "{}: Cartesian l = {} shells are not supported by the generated kernels",
            role.name(),
            s.l
        )));
    }
    if s.l > role.lmax() {
        return Err(GpuError::Contract(format!(
            "{}: l = {} is beyond the generated classes (orbital l <= 4, auxiliary l <= 6)",
            role.name(),
            s.l
        )));
    }
    Ok(())
}

/// (|R_2t|)^1/2 at P = Q, alpha = p/2: the Coulomb norm of the Hermite function t of a
/// primitive pair, for t of degree <= l.
fn hermite_norms(p: f64, l: usize) -> Vec<f64> {
    let l2 = 2 * l;
    let mut fm = vec![0.0; l2 + 1];
    boys(l2, 0.0, &mut fm);
    let mut r = vec![0.0; nherm(l2)];
    let mut rw = vec![0.0; 2 * nherm(l2)];
    let pref = ERI_PREF / (p * p * (2.0 * p).sqrt());
    r_tensor(l2, 0.5 * p, [0.0; 3], &fm, pref, &mut r, &mut rw);
    herm_tuv()[..nherm(l)]
        .iter()
        .map(|t| {
            r[herm_idx(2 * t[0] as usize, 2 * t[1] as usize, 2 * t[2] as usize)]
                .abs()
                .sqrt()
        })
        .collect()
}

impl Eri3cHost {
    /// Packs `pairs` (shells of `basis`) against every shell of `aux`.
    pub fn new(
        basis: &BasisSet,
        aux: &BasisSet,
        pairs: &[PairRow],
        kind: Eri3cKind,
        prim_thresh: f64,
    ) -> Result<Eri3cHost, GpuError> {
        Self::build(basis, aux, pairs, kind, prim_thresh, None)
    }

    /// As [`Eri3cHost::new`] for integrals, from linus-ints' prepared pairs (e.g. a
    /// density-fitting layout's `Pair3c`s and their first rows): reuses their Hermite expansions
    /// and primitive bounds.
    pub fn from_pair3c(
        basis: &BasisSet,
        aux: &BasisSet,
        pairs: &[Pair3c],
        rows: &[usize],
        prim_thresh: f64,
    ) -> Result<Eri3cHost, GpuError> {
        let pr: Vec<PairRow> = pairs
            .iter()
            .zip(rows)
            .map(|(p, &row)| PairRow {
                sa: p.sa,
                sb: p.sb,
                row,
            })
            .collect();
        Self::build(
            basis,
            aux,
            &pr,
            Eri3cKind::Integrals,
            prim_thresh,
            Some(pairs),
        )
    }

    fn build(
        basis: &BasisSet,
        aux: &BasisSet,
        pairs: &[PairRow],
        kind: Eri3cKind,
        prim_thresh: f64,
        pre: Option<&[Pair3c]>,
    ) -> Result<Eri3cHost, GpuError> {
        for s in &basis.shells {
            check_pure(s, Role::Orbital)?;
        }
        for s in &aux.shells {
            check_pure(s, Role::Auxiliary)?;
        }
        let mut row_end = 0;
        for p in pairs {
            let (a, b) = (&basis.shells[p.sa], &basis.shells[p.sb]);
            row_end = row_end.max(p.row + a.nfunc() * b.nfunc());
        }
        // classes in (la, lb) order
        let mut classes: Vec<HostClass> = (0..=4)
            .flat_map(|la| (0..=la).map(move |lb| (la, lb)))
            .map(|(la, lb)| HostClass {
                la,
                lb,
                p0: vec![0],
                rec: ClassShape::new(la, lb, 0, kind).rec,
                ..Default::default()
            })
            .collect();
        let cidx = |la: usize, lb: usize| la * (la + 1) / 2 + lb;
        let shapes: Vec<ClassShape> = classes
            .iter()
            .map(|c| ClassShape::new(c.la, c.lb, 0, kind))
            .collect();
        // the records of pair i (every primitive's values, in the class's record layout) and
        // its class: independent per pair, so computed in parallel over chunks of pairs and
        // appended in pair order (the arrays are those of one serial pass, bit for bit)
        let records = |i: usize| -> (usize, Vec<f64>) {
            let pr = &pairs[i];
            let (la, lb) = (basis.shells[pr.sa].l, basis.shells[pr.sb].l);
            let swap = la < lb;
            let (s1, s2) = if swap { (pr.sb, pr.sa) } else { (pr.sa, pr.sb) };
            let (l1, l2) = (la.max(lb), la.min(lb));
            let ci = cidx(l1, l2);
            let shape = &shapes[ci];
            let (n1, n2) = (nfunc(l1), nfunc(l2));
            let nab = n1 * n2;
            let mut d: Vec<f64> = Vec::new();
            match kind {
                Eri3cKind::Integrals => {
                    let owned;
                    let p3: &Pair3c = match pre {
                        Some(pp) => &pp[i],
                        None => {
                            let sp = build_pair(
                                &ShellView::from_shell(&basis.shells[pr.sa]),
                                &ShellView::from_shell(&basis.shells[pr.sb]),
                                pr.sa,
                                pr.sb,
                            );
                            owned = Pair3c::from_shell_pair(&sp);
                            &owned
                        }
                    };
                    let nh = nherm(l1 + l2);
                    d.reserve(p3.nprim() * (5 + shape.sets[0].len()));
                    // Pair3c: et[ip][mn][t'] with mn = ia nb + ib in (sa, sb) order
                    for ip in 0..p3.nprim() {
                        d.push(p3.p[ip]);
                        d.extend_from_slice(&p3.pc[ip]);
                        d.push(p3.qprim[ip]);
                        let et = &p3.et[ip * nab * nh..(ip + 1) * nab * nh];
                        for &(r, mn) in &shape.sets[0] {
                            let (i1, i2) = (mn / n2, mn % n2);
                            let orig = if swap { i2 * n1 + i1 } else { mn };
                            d.push(et[orig * nh + r]);
                        }
                    }
                }
                Eri3cKind::Derivatives => {
                    let (va, vb) = (
                        ShellView::from_shell(&basis.shells[s1]),
                        ShellView::from_shell(&basis.shells[s2]),
                    );
                    let (sp, pd) = pair_data::build(&va, &vb, s1, s2, 1, PRIM_PAIR_CUT);
                    let pd = pd.expect("first-order pair data");
                    let nh1 = nherm(l1 + l2 + 1);
                    let mut eb = vec![0.0; nh1 * 3 * nab];
                    for ip in 0..sp.nprim {
                        let ea = pd.ea_prim(ip);
                        eb_prim(&sp, &pd, ip, &mut eb);
                        let start = d.len();
                        d.push(sp.p[ip]);
                        d.extend_from_slice(&sp.pc[ip]);
                        d.push(0.0);
                        // ea, eb: [r][k][mn] over Herm(lab + 1)
                        let norms = hermite_norms(sp.p[ip], l1 + l2 + 1);
                        let mut q: f64 = 0.0;
                        for (s, set) in shape.sets.iter().enumerate() {
                            let (src, k) = if s < 3 { (ea, s) } else { (&eb[..], s - 3) };
                            let mut rowsum = vec![0.0; nab];
                            for &(r, mn) in set {
                                let v = src[(r * 3 + k) * nab + mn];
                                d.push(v);
                                rowsum[mn] += v.abs() * norms[r];
                            }
                            q = rowsum.iter().cloned().fold(q, f64::max);
                        }
                        d[start + 4] = q;
                    }
                }
                Eri3cKind::Hessian => {
                    let (va, vb) = (
                        ShellView::from_shell(&basis.shells[s1]),
                        ShellView::from_shell(&basis.shells[s2]),
                    );
                    let (sp, pd) = pair_data::build(&va, &vb, s1, s2, 2, PRIM_PAIR_CUT);
                    let pd = pd.expect("second-order pair data");
                    let norms_of = |p: f64| hermite_norms(p, l1 + l2 + 2);
                    let tuv = herm_tuv();
                    for ip in 0..sp.nprim {
                        // E [r][mn] (lab), E_A [r][k][mn] (lab + 1), E_AA [r][kl][mn] (lab + 2)
                        let (e, ea, eaa) = (sp.e_prim(ip), pd.ea_prim(ip), pd.eaa_prim(ip));
                        let value = |v: usize, r: usize, mn: usize| -> f64 {
                            match v {
                                0..=5 => eaa[(r * 6 + v) * nab + mn],
                                6..=8 => ea[(r * 3 + v - 6) * nab + mn],
                                _ => e[r * nab + mn],
                            }
                        };
                        let start = d.len();
                        d.push(sp.p[ip]);
                        d.extend_from_slice(&sp.pc[ip]);
                        d.push(0.0);
                        for (v, set) in shape.vsets.iter().enumerate() {
                            for &(r, mn) in set {
                                d.push(value(v, r, mn));
                            }
                        }
                        // bound: per output set and row, sum |value| x the Coulomb norm of the
                        // (shifted) Hermite function it multiplies
                        let norms = norms_of(sp.p[ip]);
                        let mut q: f64 = 0.0;
                        for &(v, sh) in &shape.osets {
                            let mut rowsum = vec![0.0; nab];
                            for &(r, mn) in &shape.vsets[v] {
                                let t = tuv[r];
                                let rs = herm_idx(
                                    t[0] as usize + sh[0],
                                    t[1] as usize + sh[1],
                                    t[2] as usize + sh[2],
                                );
                                rowsum[mn] += value(v, r, mn).abs() * norms[rs];
                            }
                            q = rowsum.iter().cloned().fold(q, f64::max);
                        }
                        d[start + 4] = q;
                    }
                }
            }
            (ci, d)
        };
        let np = pairs.len();
        for c0 in (0..np).step_by(PACK_CHUNK) {
            let c1 = (c0 + PACK_CHUNK).min(np);
            let recs: Vec<(usize, Vec<f64>)> = (c0..c1).into_par_iter().map(records).collect();
            for (i, (ci, d)) in (c0..c1).zip(recs) {
                let pr = &pairs[i];
                let swap = basis.shells[pr.sa].l < basis.shells[pr.sb].l;
                let nb_orig = basis.shells[pr.sb].nfunc();
                let c = &mut classes[ci];
                c.pairs.push(i);
                // row of kernel function pair (i1, i2): unswapped (ia, ib) = (i1, i2), swapped
                // (ia, ib) = (i2, i1)
                let (s_a, s_b) = if swap {
                    (1, nb_orig as i64)
                } else {
                    (nb_orig as i64, 1)
                };
                c.out
                    .extend_from_slice(&[pr.row as i64, s_a, s_b, swap as i64]);
                c.atom.extend_from_slice(&[
                    basis.shells[pr.sa].atom as i32,
                    basis.shells[pr.sb].atom as i32,
                ]);
                c.d.extend_from_slice(&d);
                c.p0.push((c.d.len() / c.rec) as i32);
            }
        }
        // auxiliary primitives by l, with their Schwarz bounds (as linus-ints' Aux3c::chunks)
        let mut one = aux.clone();
        one.shells.clear();
        one.atom_shells.clear();
        let mut auxs: Vec<HostAux> = (0..=6)
            .map(|l| HostAux {
                l,
                p0: vec![0],
                ..Default::default()
            })
            .collect();
        for s in &aux.shells {
            let a = &mut auxs[s.l];
            for (&e, &cf) in s.exps.iter().zip(&s.coefs) {
                let sign = if s.l % 2 == 0 { 1.0 } else { -1.0 };
                let prim = Shell {
                    exps: vec![e],
                    coefs: vec![cf],
                    ao_start: 0,
                    ..s.clone()
                };
                one.nbf = prim.nfunc();
                one.shells = vec![prim];
                let q = Aux3c::new(&one).qshell[0];
                a.d.extend_from_slice(&[
                    e,
                    cf * (2.0 * e).powi(-(s.l as i32)) * sign / e,
                    s.center[0],
                    s.center[1],
                    s.center[2],
                    q,
                ]);
            }
            a.p0.push((a.d.len() / AUX_REC) as i32);
            a.col.push(s.ao_start as i32);
            a.atom.push(s.atom as i32);
        }
        let natom = basis
            .shells
            .iter()
            .chain(&aux.shells)
            .map(|s| s.atom + 1)
            .max()
            .unwrap_or(0)
            .max(basis.atom_shells.len());
        Ok(Eri3cHost {
            kind,
            naux: aux.nbf,
            pairs: pairs.to_vec(),
            row_end,
            classes,
            aux: auxs,
            prim_thresh,
            natom,
        })
    }

    /// Rows of pairs `range` (contiguous rows are assumed: each pair's rows follow the
    /// previous pair's).
    pub fn rows(&self, range: std::ops::Range<usize>) -> std::ops::Range<usize> {
        if range.is_empty() {
            return 0..0;
        }
        let r0 = self.pairs[range.start].row;
        let r1 = if range.end < self.pairs.len() {
            self.pairs[range.end].row
        } else {
            self.row_end
        };
        r0..r1
    }

    /// Elements of the output of pairs `range` (all planes).
    pub fn out_len(&self, range: std::ops::Range<usize>) -> usize {
        self.rows(range).len() * self.naux * self.kind.planes()
    }

    /// Model FP64 operations of pairs `range` (from the generator's IR counts): per (pair
    /// primitive, auxiliary primitive) the Boys function and prefactor, per (pair primitive,
    /// auxiliary shell) the R recursion, H and the contraction.
    pub fn flops(&self, range: std::ops::Range<usize>) -> f64 {
        self.flops_of(range, None)
    }

    /// As [`Eri3cHost::flops`] for class (la lb | lk) only, or all classes.
    pub fn flops_of(
        &self,
        range: std::ops::Range<usize>,
        class: Option<(usize, usize, usize)>,
    ) -> f64 {
        let mut f = 0.0;
        for c in &self.classes {
            let (c0, c1) = self.class_range(c, &range);
            let nprim: f64 = (c0..c1).map(|i| (c.p0[i + 1] - c.p0[i]) as f64).sum();
            for a in &self.aux {
                if a.nshell() == 0 || class.is_some_and(|k| k != (c.la, c.lb, a.l)) {
                    continue;
                }
                let k = crate::kernels::eri3c_class::gen_kernel(self.kind, c.la, c.lb, a.l)
                    .expect("generated kernel");
                let naprim = (a.d.len() / AUX_REC) as f64;
                f += nprim
                    * (naprim * k.prim_flops as f64
                        + a.nshell() as f64 * k.pair_shell_flops as f64);
            }
        }
        f
    }

    /// The class's pairs inside the builder's pair range.
    fn class_range(&self, c: &HostClass, range: &std::ops::Range<usize>) -> (usize, usize) {
        (
            c.pairs.partition_point(|&i| i < range.start),
            c.pairs.partition_point(|&i| i < range.end),
        )
    }

    /// The CPU reference of the operator `op` (linus-ints' `pair_block` for integrals,
    /// `Eri3DerivPlan::deriv_block` for derivatives, any a/r + b erf(omega r)/r, no screening)
    /// of pairs `range`, in the device layout, on the current rayon pool. For tests and
    /// benchmarks: the derivative reference forms a dense nbf x naux x nbf block per direction,
    /// so it is for small molecules only.
    pub fn cpu_reference(
        &self,
        basis: &BasisSet,
        aux: &BasisSet,
        range: std::ops::Range<usize>,
        op: &Operator,
    ) -> Vec<f64> {
        use crate::cpu::{
            pair_block, DerivOrder, Eri3Block, Eri3DerivOptions, Eri3DerivPlan, PairOptions,
            PairSet, Work3c,
        };
        use rayon::prelude::*;
        let rows = self.rows(range.clone());
        let naux = self.naux;
        let aux3 = Aux3c::new(aux);
        let pairs = &self.pairs[range];
        match self.kind {
            Eri3cKind::Integrals => {
                let chunks = aux3.chunks(0..aux.shells.len());
                let mut out = vec![0.0; rows.len() * naux];
                let blocks: Vec<(usize, Vec<f64>)> = pairs
                    .par_iter()
                    .map_init(Work3c::new, |w, pr| {
                        let sp = build_pair(
                            &ShellView::from_shell(&basis.shells[pr.sa]),
                            &ShellView::from_shell(&basis.shells[pr.sb]),
                            pr.sa,
                            pr.sb,
                        );
                        let p3 = Pair3c::from_shell_pair(&sp);
                        let mut blk = vec![0.0; p3.nab() * naux];
                        pair_block(&p3, &chunks, &mut blk, naux, 0.0, op, w);
                        (pr.row - rows.start, blk)
                    })
                    .collect();
                for (r, blk) in blocks {
                    out[r * naux..r * naux + blk.len()].copy_from_slice(&blk);
                }
                out
            }
            Eri3cKind::Derivatives => {
                let nbf = basis.nbf;
                let ao: Vec<(usize, usize)> = basis
                    .shells
                    .iter()
                    .map(|s| (s.ao_start, s.nfunc()))
                    .collect();
                // every pair with primitives (the reference reads the requested ones)
                let pl = PairSet::build(basis, PairOptions::with_order(DerivOrder::One));
                let ap = PairSet::aux(aux, PairOptions::default());
                let opts = Eri3DerivOptions {
                    thresholds: crate::cpu::DfThresholds {
                        deriv_pair: 0.0,
                        ..Default::default()
                    },
                };
                let plan = Eri3DerivPlan::new(basis, &pl, aux, &ap, DerivOrder::One, opts)
                    .expect("orbital pairs of order 1");
                let mut ws = plan.workspace(rayon::current_num_threads());
                let mut dense = vec![0.0; 3 * nbf * naux * nbf];
                let blk = Eri3Block {
                    shells: 0..aux.shells.len(),
                    op: *op,
                    thresh: 0.0,
                };
                plan.deriv_block(&blk, 0..nbf, &mut dense, &mut ws);
                let plane = rows.len() * naux;
                let mut out = vec![0.0; 6 * plane];
                let at = |x: usize, a: usize, p: usize, b: usize| {
                    dense[((x * nbf + a) * naux + p) * nbf + b]
                };
                for pr in pairs {
                    let (a0, na) = ao[pr.sa];
                    let (b0, nb) = ao[pr.sb];
                    for ia in 0..na {
                        for ib in 0..nb {
                            let row = pr.row - rows.start + ia * nb + ib;
                            for x in 0..3 {
                                for p in 0..naux {
                                    out[x * plane + row * naux + p] = at(x, a0 + ia, p, b0 + ib);
                                    out[(3 + x) * plane + row * naux + p] =
                                        at(x, b0 + ib, p, a0 + ia);
                                }
                            }
                        }
                    }
                }
                out
            }
            // second derivatives are checked contracted (linus-ints' Eri3DerivPlan::hessian against
            // Eri3c::hessian), not plane by plane
            Eri3cKind::Hessian => Vec::new(),
        }
    }
}

/// Every pair (a, b), a >= b, of `basis` that keeps at least one primitive pair, with rows in
/// that order (for tests and benchmarks; a density-fitting code passes its own screened list).
pub fn all_pairs(basis: &BasisSet) -> Vec<PairRow> {
    let mut out = Vec::new();
    let mut row = 0;
    for a in 0..basis.shells.len() {
        for b in 0..=a {
            let sp = build_pair(
                &ShellView::from_shell(&basis.shells[a]),
                &ShellView::from_shell(&basis.shells[b]),
                a,
                b,
            );
            if sp.nprim == 0 {
                continue;
            }
            out.push(PairRow { sa: a, sb: b, row });
            row += basis.shells[a].nfunc() * basis.shells[b].nfunc();
        }
    }
    out
}

#[cfg(feature = "gpu")]
pub use device::{Eri3c, Eri3cAuxBlocks, Eri3cStaged};

/// The extra compiler options of the generated kernels of `kind` on a device of compute
/// capability `cc` (besides the cache's base options and the architecture).
///
/// The second-derivative kernels ([`Eri3cKind::Hessian`]) on cc 9.x (H100/H200) take ptxas
/// -O1 instead of its default -O3. At -O3, ptxas of CUDA 13.0 (NVRTC 13.0.88; driver 580.178;
/// NVRTC 13.1.115 alike) builds sm_90 cubins of four 1/r classes that compute wrong values,
/// deterministically and independently of the blocks per SM: (ff|g), (gg|d), (gd|g), (gf|f),
/// relative errors 3.8e-3, 2.6e-3, 5.2e-6, 1.4e-6 of the class's Gamma-contracted Hessian
/// against the same source on sm_120 (test `eri3c_gamma_hessian_matches_hess_3c`, FeOH
/// def2-QZVP: 2.1e-2 absolute against linus-ints on an H200; the u1an39 B3LYP/def2-TZVP
/// Hessian 4.9e-4 off, from its C-F pairs' (ff|g)). At -O1 (and -O0) every class agrees with
/// sm_120 below 1e-12 relative and that Hessian with the CPU's to 7e-11; compute-sanitizer's
/// memcheck, initcheck, racecheck and synccheck find nothing in these kernels (sm_120), and a
/// standalone reproducer (the kernel's source and one launch's recorded inputs, no linus code)
/// shows the -O3 and -O1 cubins disagreeing on an H200 only. Cost on an H200: the u1an39
/// TZVP Hessian's second-derivative phase 0.79 to 1.04 s (of 44 s). sm_120 and the other kinds
/// keep -O3 (their results unchanged). Re-test (that test on an H200 at -O3) and remove when
/// the toolkit changes.
pub fn compile_options(cc: (i32, i32), kind: Eri3cKind) -> &'static [&'static str] {
    if cc.0 == 9 && kind == Eri3cKind::Hessian {
        &["--ptxas-options=-O1"]
    } else {
        &[]
    }
}

/// The module name and source of one operator's kernel of `gk` (or of `source` with gk's entry:
/// the tuner's candidates), as the builder compiles it: the operator header (1/r, or
/// erf(omega r)/r with `erf`), the fixed-point accumulator of the derivative kernels' Gamma
/// epilogue (and the second derivatives' Hessian blocks), the shared R recursion of block
/// kernels with R inline (the prelude), the kernel.
pub fn module_source(
    gk: &crate::kernels::eri3c_class::GenKernel,
    source: Option<&str>,
    erf: bool,
) -> (String, String) {
    use crate::kernels::eri3c_gen::{HESSIAN_H, OPERATOR_H};
    let detsum = match gk.kind {
        Eri3cKind::Derivatives => std::borrow::Cow::Borrowed(crate::detsum::DEVICE_SRC),
        Eri3cKind::Hessian => {
            std::borrow::Cow::Owned(format!("{}\n{HESSIAN_H}", crate::detsum::DEVICE_SRC))
        }
        Eri3cKind::Integrals => std::borrow::Cow::Borrowed(""),
    };
    let op = if erf { "#define ERI3C_ERF\n" } else { "" };
    let full = format!(
        "{op}{OPERATOR_H}\n{detsum}\n{}\n{}",
        gk.prelude,
        source.unwrap_or(gk.source)
    );
    let name = if erf {
        format!("{}_erf", gk.entry)
    } else {
        gk.entry.to_string()
    };
    (name, full)
}

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::detsum::DetAcc;
    use crate::kernels::boys::boys_table;
    use crate::kernels::eri3c_class::{gen_kernel_for, GenKernel, Mapping};
    use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};

    struct DevClass {
        p0: DeviceBuf<i32>,
        d: DeviceBuf<f64>,
        out: DeviceBuf<i64>,
        atom: DeviceBuf<i32>,
    }

    struct DevAux {
        p0: DeviceBuf<i32>,
        d: DeviceBuf<f64>,
        col: DeviceBuf<i32>,
        atom: DeviceBuf<i32>,
    }

    /// Blocks of auxiliary shells for [`Eri3c::compute_block`] ([`Eri3c::aux_blocks`]).
    pub struct Eri3cAuxBlocks {
        /// Per block: first auxiliary function and number of functions.
        pub blocks: Vec<(usize, usize)>,
        /// Per block and l: the range of the builder's shells of l in the block.
        ranges: Vec<[(usize, usize); 7]>,
        /// Per l: the first column of each shell relative to its block's first function.
        col: Vec<Option<DeviceBuf<i32>>>,
    }

    /// Where a launch puts its results: an output buffer from an element offset on, or a
    /// Gamma and the gradient accumulator.
    enum Target<'a> {
        Out(&'a mut DeviceBuf<f64>, usize),
        Gradient(&'a DeviceBuf<f64>, &'a mut DetAcc),
        /// A Gamma and the Hessian accumulator ((3 natom)^2 elements, M of
        /// [`Eri3c::hessian`]).
        Hessian(&'a DeviceBuf<f64>, &'a mut DetAcc),
    }

    struct DevTables {
        hp: DeviceBuf<i32>,
        hc: DeviceBuf<f64>,
        hr: DeviceBuf<i32>,
        rp: DeviceBuf<i32>,
        re: DeviceBuf<i32>,
        rt: DeviceBuf<i32>,
    }

    struct DevKernel {
        gk: &'static GenKernel,
        /// The 1/r kernel.
        kernel: Kernel,
        /// The erf(omega r)/r kernel of the same source ([`Eri3c::prepare_operator`]).
        kernel_erf: Option<Kernel>,
        /// The source the kernels were compiled from (the tuner's candidate, or None for the
        /// generated one).
        source: Option<String>,
        tables: Option<DevTables>,
    }

    /// The record arrays of an [`Eri3cHost`] (each class's `d`, the bulk of its data) copied
    /// into page-locked host memory, so that a builder's setup uploads them by DMA
    /// ([`Eri3c::with_staged`]) instead of through the driver's pageable path. Made ahead of
    /// the setup (e.g. while an SCF runs; [`crate::df::DfShared`] keeps them).
    pub struct Eri3cStaged {
        d: Vec<Option<crate::HostBuf<f64>>>,
    }

    impl Eri3cStaged {
        /// Page-locked copies of `host`'s record arrays (the copy on the current rayon pool).
        pub fn new(gpu: &Gpu, host: &Eri3cHost) -> Result<Eri3cStaged, GpuError> {
            use rayon::prelude::*;
            let mut d = Vec::with_capacity(host.classes.len());
            for c in &host.classes {
                if c.pairs.is_empty() || c.d.is_empty() {
                    d.push(None);
                    continue;
                }
                let mut hb = gpu.alloc_host::<f64>(c.d.len())?;
                hb.as_mut_slice()?
                    .par_chunks_mut(1 << 16)
                    .zip(c.d.par_chunks(1 << 16))
                    .for_each(|(a, b)| a.copy_from_slice(b));
                d.push(Some(hb));
            }
            Ok(Eri3cStaged { d })
        }

        /// Bytes of page-locked memory held.
        pub fn bytes(&self) -> usize {
            self.d.iter().flatten().map(|h| h.bytes()).sum()
        }
    }

    /// A three-center builder on the device (see the module docs).
    pub struct Eri3c {
        /// The packed host data (shared: [`crate::df::DfShared`] keeps it for other builders).
        host: std::sync::Arc<Eri3cHost>,
        /// Setup: seconds uploading the packed data (and its bytes), seconds getting the
        /// classes' kernels (compiled, or from the caches).
        pub setup_upload: f64,
        pub setup_upload_bytes: usize,
        pub setup_kernels: f64,
        btab: DeviceBuf<f64>,
        classes: Vec<Option<DevClass>>,
        aux: Vec<Option<DevAux>>,
        /// [class][l_aux]
        kernels: Vec<Vec<Option<DevKernel>>>,
    }

    impl Eri3c {
        /// Uploads `host` and compiles the kernels of its classes.
        pub fn new(
            gpu: &Gpu,
            host: impl Into<std::sync::Arc<Eri3cHost>>,
        ) -> Result<Eri3c, GpuError> {
            Self::with_staged(gpu, host, None)
        }

        /// As [`Eri3c::new`], the record arrays uploaded from `staged` (page-locked copies of
        /// `host`'s, [`Eri3cStaged`]) when given: the same builder.
        pub fn with_staged(
            gpu: &Gpu,
            host: impl Into<std::sync::Arc<Eri3cHost>>,
            staged: Option<&Eri3cStaged>,
        ) -> Result<Eri3c, GpuError> {
            let mut b = Self::records_only(gpu, host, staged)?;
            b.ensure_kernels(gpu)?;
            Ok(b)
        }

        /// As [`Eri3c::with_staged`] without the classes' kernels: the records on the device
        /// for engines that contract them themselves (`grad::df_j`), the kernels loaded by
        /// [`Eri3c::ensure_kernels`] before any launch of this builder's.
        pub fn records_only(
            gpu: &Gpu,
            host: impl Into<std::sync::Arc<Eri3cHost>>,
            staged: Option<&Eri3cStaged>,
        ) -> Result<Eri3c, GpuError> {
            let host = host.into();
            let t = std::time::Instant::now();
            let btab = gpu.upload(&boys_table())?;
            let mut classes = Vec::new();
            for (i, c) in host.classes.iter().enumerate() {
                classes.push(if c.pairs.is_empty() {
                    None
                } else {
                    let d = match staged.and_then(|s| s.d.get(i)).and_then(|h| h.as_ref()) {
                        Some(h) if h.len() == c.d.len() => {
                            let mut d = gpu.alloc_zeros::<f64>(h.len())?;
                            gpu.upload_from_host(h, 0, &mut d, 0, h.len())?;
                            d
                        }
                        _ => gpu.upload(&c.d)?,
                    };
                    Some(DevClass {
                        p0: gpu.upload(&c.p0)?,
                        d,
                        out: gpu.upload(&c.out)?,
                        atom: gpu.upload(&c.atom)?,
                    })
                });
            }
            let mut aux = Vec::new();
            for a in &host.aux {
                aux.push(if a.nshell() == 0 {
                    None
                } else {
                    Some(DevAux {
                        p0: gpu.upload(&a.p0)?,
                        d: gpu.upload(&a.d)?,
                        col: gpu.upload(&a.col)?,
                        atom: gpu.upload(&a.atom)?,
                    })
                });
            }
            let setup_upload = t.elapsed().as_secs_f64();
            let setup_upload_bytes = host
                .classes
                .iter()
                .map(|c| 8 * (c.d.len() + c.out.len()) + 4 * (c.p0.len() + c.atom.len()))
                .sum::<usize>()
                + host
                    .aux
                    .iter()
                    .map(|a| 8 * a.d.len() + 4 * (a.p0.len() + a.col.len() + a.atom.len()))
                    .sum::<usize>();
            Ok(Eri3c {
                host,
                setup_upload,
                setup_upload_bytes,
                setup_kernels: 0.0,
                btab,
                classes,
                aux,
                kernels: Vec::new(),
            })
        }

        /// Loads the classes' 1/r kernels (compiled, or from the caches) once; a builder made
        /// by [`Eri3c::records_only`] needs them before its first launch.
        pub fn ensure_kernels(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
            if self.kernels_loaded() {
                return Ok(());
            }
            let t = std::time::Instant::now();
            let host = &self.host;
            let mut kernels = Vec::new();
            for c in &host.classes {
                let mut row = Vec::new();
                for a in &host.aux {
                    if c.pairs.is_empty() || a.nshell() == 0 {
                        row.push(None);
                        continue;
                    }
                    let gk = gen_kernel_for(gpu.info().cc, host.kind, c.la, c.lb, a.l)
                        .ok_or_else(|| GpuError::Contract("no generated kernel".into()))?;
                    row.push(Some(Self::prepare(gpu, gk, None, false)?));
                }
                kernels.push(row);
            }
            self.kernels = kernels;
            self.setup_kernels += t.elapsed().as_secs_f64();
            Ok(())
        }

        /// The classes' kernels are loaded ([`Eri3c::ensure_kernels`]).
        pub fn kernels_loaded(&self) -> bool {
            !self.kernels.is_empty() || self.host.classes.is_empty()
        }

        /// Loads (compiles, or reads from the caches) the 1/r kernels a builder of `host` takes,
        /// so that its setup finds them in memory (e.g. ahead of it, while other device work
        /// runs); nothing is uploaded.
        pub fn preload(gpu: &Gpu, host: &Eri3cHost) -> Result<(), GpuError> {
            for c in host.classes.iter().filter(|c| !c.pairs.is_empty()) {
                for a in host.aux.iter().filter(|a| a.nshell() > 0) {
                    // the kernels the setup takes: this device's architecture's schedules
                    let gk = gen_kernel_for(gpu.info().cc, host.kind, c.la, c.lb, a.l)
                        .ok_or_else(|| GpuError::Contract("no generated kernel".into()))?;
                    Self::compile(gpu, gk, None, false)?;
                }
            }
            Ok(())
        }

        /// Compiles one operator's kernel of `gk` (or of `source` with gk's entry and launch
        /// data: the tuner's candidates): the operator header (1/r, or erf(omega r)/r with
        /// `erf`), the fixed-point accumulator of the derivative kernels' Gamma epilogue, the
        /// shared R recursion of block kernels with R inline (the prelude), the kernel.
        fn compile(
            gpu: &Gpu,
            gk: &'static GenKernel,
            source: Option<&str>,
            erf: bool,
        ) -> Result<Kernel, GpuError> {
            let (name, full) = module_source(gk, source, erf);
            let extra = compile_options(gpu.info().cc, gk.kind);
            let kernel = gpu.kernel_with(&name, &full, gk.entry, extra)?;
            if gk.mapping == Mapping::Block {
                // dynamic plus static shared memory may pass the default 48 KiB (the derivative
                // kernels' Gamma epilogue adds a static array): always opt in
                kernel.set_max_dynamic_shared(gk.smem)?;
            }
            Ok(kernel)
        }

        /// Compiles `gk` (or `source`, see [`Eri3c::compile`]) for 1/r and, with `erf`, for
        /// erf(omega r)/r, and uploads its tables.
        fn prepare(
            gpu: &Gpu,
            gk: &'static GenKernel,
            source: Option<&str>,
            erf: bool,
        ) -> Result<DevKernel, GpuError> {
            let shape = ClassShape::new(gk.la, gk.lb, gk.lk, gk.kind);
            shape.check(gk).map_err(GpuError::Contract)?;
            let kernel = Self::compile(gpu, gk, source, false)?;
            let kernel_erf = if erf {
                Some(Self::compile(gpu, gk, source, true)?)
            } else {
                None
            };
            let tables = if gk.mapping == Mapping::Block {
                let t = shape.tables();
                Some(DevTables {
                    hp: gpu.upload(&t.hp)?,
                    hc: gpu.upload(&t.hc)?,
                    hr: gpu.upload(&t.hr)?,
                    rp: gpu.upload(&t.rp)?,
                    re: gpu.upload(&t.re)?,
                    rt: gpu.upload(&t.rt)?,
                })
            } else {
                None
            };
            Ok(DevKernel {
                gk,
                kernel,
                kernel_erf,
                source: source.map(str::to_string),
                tables,
            })
        }

        pub fn host(&self) -> &Eri3cHost {
            &self.host
        }

        /// Class `ci`'s device records, for engines that contract the pair expansions
        /// themselves (`grad::df_j`): the records (`rec` values per primitive), per
        /// pair the output rows and swap flag (4 values) and the shells' atoms (2).
        pub(crate) fn class_records(
            &self,
            ci: usize,
        ) -> Option<(&DeviceBuf<f64>, &DeviceBuf<i64>, &DeviceBuf<i32>)> {
            self.classes
                .get(ci)
                .and_then(|c| c.as_ref())
                .map(|c| (&c.d, &c.out, &c.atom))
        }

        /// Makes the builder ready for the operator `op` ([`kernel_operator`]: 1/r, always
        /// ready, or erf(omega r)/r for any omega > 0): compiles the erf kernels of its classes
        /// once (the disk cache keeps them). Idempotent.
        pub fn prepare_operator(&mut self, gpu: &Gpu, op: &Operator) -> Result<(), GpuError> {
            if kernel_operator(op)?.is_none() {
                return Ok(());
            }
            self.ensure_kernels(gpu)?;
            for row in self.kernels.iter_mut() {
                for dk in row.iter_mut().flatten() {
                    if dk.kernel_erf.is_none() {
                        dk.kernel_erf =
                            Some(Self::compile(gpu, dk.gk, dk.source.as_deref(), true)?);
                    }
                }
            }
            Ok(())
        }

        /// The builder has its erf(omega r)/r kernels ([`Eri3c::prepare_operator`]).
        pub fn has_erf(&self) -> bool {
            self.kernels_loaded()
                && self
                    .kernels
                    .iter()
                    .flatten()
                    .flatten()
                    .all(|k| k.kernel_erf.is_some())
        }

        /// Swaps in a candidate kernel for class (la lb | lk): `gk` gives the entry point and
        /// launch data, `source` the code (the tuner).
        pub fn set_kernel(
            &mut self,
            gpu: &Gpu,
            gk: &'static GenKernel,
            source: &str,
        ) -> Result<(), GpuError> {
            // a records-only builder's kernels first
            self.ensure_kernels(gpu)?;
            let ci = gk.la * (gk.la + 1) / 2 + gk.lb;
            if let Some(Some(old)) = self.kernels.get(ci).and_then(|k| k.get(gk.lk)) {
                let erf = old.kernel_erf.is_some();
                self.kernels[ci][gk.lk] = Some(Self::prepare(gpu, gk, Some(source), erf)?);
            }
            Ok(())
        }

        /// Fills `out` from element `offset` on (at least [`Eri3cHost::out_len`] elements; e.g.
        /// the pair range's first row in a tensor of all rows: `offset` = row naux) with the
        /// output of pairs `range` for the operator `op` ([`kernel_operator`]: 1/r, or
        /// erf(omega r)/r after [`Eri3c::prepare_operator`]): rows relative to the first pair's
        /// row, `naux` columns, planes of rows x naux elements. Every element is written.
        pub fn compute(
            &self,
            gpu: &Gpu,
            range: std::ops::Range<usize>,
            op: &Operator,
            out: &mut DeviceBuf<f64>,
            offset: usize,
        ) -> Result<(), GpuError> {
            let omega = self.operator(op)?;
            self.compute_classes(gpu, range, Target::Out(out, offset), None, omega)
        }

        /// The erf parameter of `op` for a launch, checked against the compiled kernels.
        fn operator(&self, op: &Operator) -> Result<Option<f64>, GpuError> {
            let omega = kernel_operator(op)?;
            if omega.is_some() && !self.has_erf() {
                return Err(GpuError::Contract(
                    "eri3c: erf kernels not compiled (Eri3c::prepare_operator)".into(),
                ));
            }
            Ok(omega)
        }

        /// Error unless the builder computes `kind` (derivatives or second derivatives), `acc`
        /// holds at least `n_acc` accumulators and `gamma` one value per row of `range` and
        /// auxiliary function.
        fn check_contraction(
            &self,
            kind: Eri3cKind,
            range: &std::ops::Range<usize>,
            gamma: &DeviceBuf<f64>,
            (acc, n_acc): (&DetAcc, usize),
        ) -> Result<(), GpuError> {
            let h = &self.host;
            if h.kind != kind {
                return Err(GpuError::Contract(format!(
                    "eri3c: a {kind:?} contraction needs a {kind:?} builder, not {:?}",
                    h.kind
                )));
            }
            if acc.len() < n_acc {
                return Err(GpuError::Contract(format!(
                    "eri3c: {} accumulators, {n_acc} needed",
                    acc.len()
                )));
            }
            if gamma.len() < h.rows(range.clone()).len() * h.naux {
                return Err(GpuError::Contract("eri3c: Gamma buffer too small".into()));
            }
            Ok(())
        }

        /// Derivatives builder: adds g\[3 atom + x\] += sum Gamma\[row\]\[P\] d/dX_x (row|P) over
        /// the rows of pairs `range` and every auxiliary function, for X each atom (the pair
        /// centres, and the auxiliary centre by translational invariance), to `acc` (3 natom
        /// fixed-point accumulators: deterministic in any order), with the derivative integrals
        /// of the operator `op` ([`kernel_operator`]: 1/r, or erf(omega r)/r after
        /// [`Eri3c::prepare_operator`]). `gamma` has the layout of [`Eri3c::compute`]'s output
        /// for one plane (rows relative to the range's first row); each stored row (m, n)
        /// counts once, so a symmetric density's factor 2 for the off-diagonal shell pairs
        /// belongs in Gamma. No derivative integral is stored.
        pub fn gradient(
            &self,
            gpu: &Gpu,
            range: std::ops::Range<usize>,
            op: &Operator,
            gamma: &DeviceBuf<f64>,
            acc: &mut DetAcc,
        ) -> Result<(), GpuError> {
            let n_acc = 3 * self.host.natom;
            self.check_contraction(Eri3cKind::Derivatives, &range, gamma, (acc, n_acc))?;
            let omega = self.operator(op)?;
            self.compute_classes(gpu, range, Target::Gradient(gamma, acc), None, omega)
        }

        /// Second-derivatives builder: adds sum_{row, P} Gamma\[row\]\[P\] d^2/dX dY (row|P) over
        /// the rows of pairs `range` and every auxiliary function, for X, Y the atoms of the pair
        /// centres and of the auxiliary centre (translational invariance), to `acc`: (3 natom)^2
        /// fixed-point accumulators (row-major) holding M with the Hessian H = (M + M^T) / 2
        /// (deterministic in any order), with the integrals of the operator `op`
        /// ([`kernel_operator`]: 1/r, or erf(omega r)/r after [`Eri3c::prepare_operator`]).
        /// `gamma` has the layout of [`Eri3c::compute`]'s output for one plane (rows relative to
        /// the range's first row); each stored row counts once, so a symmetric Gamma's factor 2
        /// for the off-diagonal shell pairs belongs in it. No integral is stored.
        pub fn hessian(
            &self,
            gpu: &Gpu,
            range: std::ops::Range<usize>,
            op: &Operator,
            gamma: &DeviceBuf<f64>,
            acc: &mut DetAcc,
        ) -> Result<(), GpuError> {
            let n3 = 3 * self.host.natom;
            self.check_contraction(Eri3cKind::Hessian, &range, gamma, (acc, n3 * n3))?;
            let omega = self.operator(op)?;
            self.compute_classes(gpu, range, Target::Hessian(gamma, acc), None, omega)
        }

        /// As [`Eri3c::compute`] for one class (la lb | lk) only (the others are not written).
        pub fn compute_class(
            &self,
            gpu: &Gpu,
            range: std::ops::Range<usize>,
            out: &mut DeviceBuf<f64>,
            class: (usize, usize, usize),
        ) -> Result<(), GpuError> {
            self.compute_classes(gpu, range, Target::Out(out, 0), Some(class), None)
        }

        /// The blocks of auxiliary shells `blocks` (each (first function, functions), whole
        /// shells in the auxiliary basis's order) for [`Eri3c::compute_block`]: per block and
        /// angular momentum the range of the builder's shells of that l inside the block, and
        /// the shells' first columns relative to their block's first function.
        pub fn aux_blocks(
            &self,
            gpu: &Gpu,
            blocks: &[(usize, usize)],
        ) -> Result<Eri3cAuxBlocks, GpuError> {
            let h = &self.host;
            let mut ranges = Vec::with_capacity(blocks.len());
            let mut cols: Vec<Vec<i32>> = h.aux.iter().map(|a| a.col.clone()).collect();
            for &(p0, nq) in blocks {
                let mut r = [(0usize, 0usize); 7];
                for (l, a) in h.aux.iter().enumerate() {
                    // shells of one l are in the auxiliary basis's order: their columns ascend
                    let s0 = a.col.partition_point(|&c| (c as usize) < p0);
                    let s1 = a.col.partition_point(|&c| (c as usize) < p0 + nq);
                    for c in &mut cols[l][s0..s1] {
                        *c -= p0 as i32;
                    }
                    r[l] = (s0, s1);
                }
                ranges.push(r);
            }
            let nblk: usize = blocks.len();
            let covered: Vec<usize> = (0..h.aux.len())
                .map(|l| ranges.iter().map(|r| r[l].1 - r[l].0).sum())
                .collect();
            if covered.iter().zip(&h.aux).any(|(&c, a)| c > a.nshell()) || nblk == 0 {
                return Err(GpuError::Contract(
                    "eri3c: auxiliary blocks overlap or are empty".into(),
                ));
            }
            let col = cols
                .iter()
                .map(|c| {
                    if c.is_empty() {
                        Ok(None)
                    } else {
                        gpu.upload(c).map(Some)
                    }
                })
                .collect::<Result<_, _>>()?;
            Ok(Eri3cAuxBlocks {
                blocks: blocks.to_vec(),
                ranges,
                col,
            })
        }

        /// As [`Eri3c::compute`] for the auxiliary shells of block `b` of `blocks` only
        /// ([`Eri3c::aux_blocks`]): `out` holds rows x the block's functions (columns relative
        /// to its first function), planes of rows x block functions, from element `offset` on.
        /// Every element of that output is written.
        #[allow(clippy::too_many_arguments)]
        pub fn compute_block(
            &self,
            gpu: &Gpu,
            range: std::ops::Range<usize>,
            blocks: &Eri3cAuxBlocks,
            b: usize,
            op: &Operator,
            out: &mut DeviceBuf<f64>,
            offset: usize,
        ) -> Result<(), GpuError> {
            let omega = self.operator(op)?;
            if b >= blocks.blocks.len() {
                return Err(GpuError::Contract("eri3c: no such auxiliary block".into()));
            }
            self.launch_classes(
                gpu,
                range,
                Target::Out(out, offset),
                None,
                omega,
                Some((blocks, b)),
            )
        }

        /// Launches every class of pairs `range` (or `only`) for 1/r (`omega` None) or
        /// erf(omega r)/r.
        fn compute_classes(
            &self,
            gpu: &Gpu,
            range: std::ops::Range<usize>,
            target: Target,
            only: Option<(usize, usize, usize)>,
            omega: Option<f64>,
        ) -> Result<(), GpuError> {
            self.launch_classes(gpu, range, target, only, omega, None)
        }

        /// [`Eri3c::compute_classes`] against every auxiliary shell, or (`sel`) the shells of
        /// one block of [`Eri3cAuxBlocks`] with block-relative columns.
        fn launch_classes(
            &self,
            gpu: &Gpu,
            range: std::ops::Range<usize>,
            mut target: Target,
            only: Option<(usize, usize, usize)>,
            omega: Option<f64>,
            sel: Option<(&Eri3cAuxBlocks, usize)>,
        ) -> Result<(), GpuError> {
            if !self.kernels_loaded() {
                return Err(GpuError::Contract(
                    "eri3c: a records-only builder (ensure_kernels first)".into(),
                ));
            }
            let h = &self.host;
            // output columns: every auxiliary function, or the block's
            let ld = match sel {
                Some((bl, b)) => bl.blocks[b].1,
                None => h.naux,
            };
            if let Target::Out(out, off) = &target {
                let need = h.rows(range.clone()).len() * ld * h.kind.planes();
                if out.len() < off + need {
                    return Err(GpuError::Contract("eri3c: output buffer too small".into()));
                }
            }
            if sel.is_some() && !matches!(target, Target::Out(..)) {
                return Err(GpuError::Contract(
                    "eri3c: auxiliary blocks are for plain outputs".into(),
                ));
            }
            let rows = h.rows(range.clone());
            let pstride = (rows.len() * ld) as i64;
            for (ci, c) in h.classes.iter().enumerate() {
                let Some(dc) = &self.classes[ci] else {
                    continue;
                };
                let (c0, c1) = h.class_range(c, &range);
                if c0 == c1 {
                    continue;
                }
                for (l, a) in h.aux.iter().enumerate() {
                    if only.is_some_and(|o| o != (c.la, c.lb, l)) {
                        continue;
                    }
                    let (Some(da), Some(dk)) = (&self.aux[l], &self.kernels[ci][l]) else {
                        continue;
                    };
                    let kernel = match omega {
                        None => &dk.kernel,
                        Some(_) => dk.kernel_erf.as_ref().ok_or_else(|| {
                            GpuError::Contract("eri3c: erf kernels not compiled".into())
                        })?,
                    };
                    let npair = c1 - c0;
                    // the auxiliary shells of l: all, or the block's (offset pointers into the
                    // per-shell data, block-relative columns)
                    let (s0, nsh, col_arg) = match sel {
                        None => (0, a.nshell(), Arg::buf(&da.col)),
                        Some((bl, b)) => {
                            let (s0, s1) = bl.ranges[b][l];
                            let Some(col) = &bl.col[l] else {
                                continue;
                            };
                            (s0, s1 - s0, Arg::from(col.at(s0)))
                        }
                    };
                    if nsh == 0 {
                        continue;
                    }
                    let gk = dk.gk;
                    let ntile = nsh.div_ceil(gk.ktile.max(1) as usize);
                    let launch = match gk.mapping {
                        Mapping::Thread => Launch::linear(npair * nsh, gk.threads),
                        Mapping::Block => Launch {
                            grid: ((npair * ntile) as u32, 1, 1),
                            block: (gk.threads, 1, 1),
                            shared_bytes: gk.smem,
                        },
                    };
                    // thread-mapped kernels take no tables: any valid address
                    let tb = |f: fn(&DevTables) -> Arg| match &dk.tables {
                        Some(t) => f(t),
                        None => Arg::buf(&self.btab),
                    };
                    // plain builds pass null Gamma, atoms and accumulator (no epilogue)
                    let (out_arg, gam, pat, aat, gacc, n3) = match &mut target {
                        Target::Out(o, off) => (
                            Arg::from(o.at(*off)),
                            Arg::Ptr(0),
                            Arg::Ptr(0),
                            Arg::Ptr(0),
                            Arg::Ptr(0),
                            0,
                        ),
                        Target::Gradient(g, acc) => (
                            Arg::buf(g),
                            Arg::buf(g),
                            Arg::from(dc.atom.at(2 * c0)),
                            Arg::from(da.atom.at(s0)),
                            acc.arg(),
                            acc.len() as i64,
                        ),
                        // the Hessian epilogue takes 3 natom (its accumulator is the square)
                        Target::Hessian(g, acc) => (
                            Arg::buf(g),
                            Arg::buf(g),
                            Arg::from(dc.atom.at(2 * c0)),
                            Arg::from(da.atom.at(s0)),
                            acc.arg(),
                            3 * h.natom as i64,
                        ),
                    };
                    let mut args = vec![
                        Arg::I64(npair as i64),
                        Arg::I32(nsh as i32),
                        Arg::from(dc.p0.at(c0)),
                        Arg::buf(&dc.d),
                        Arg::from(dc.out.at(4 * c0)),
                        Arg::from(da.p0.at(s0)),
                        Arg::buf(&da.d),
                        col_arg,
                        Arg::buf(&self.btab),
                        Arg::I64(ld as i64),
                        Arg::I64(rows.start as i64),
                        Arg::I64(pstride),
                        Arg::F64(h.prim_thresh),
                        Arg::I32(ntile as i32),
                        tb(|t| Arg::buf(&t.hp)),
                        tb(|t| Arg::buf(&t.hc)),
                        tb(|t| Arg::buf(&t.hr)),
                        tb(|t| Arg::buf(&t.rp)),
                        tb(|t| Arg::buf(&t.re)),
                        tb(|t| Arg::buf(&t.rt)),
                        gam,
                        pat,
                        aat,
                        gacc,
                        Arg::I64(n3),
                        out_arg,
                    ];
                    // the erf kernels' operator argument (ERI3C_OPERATOR_ARGS): omega^2
                    if let Some(w) = omega {
                        args.push(Arg::F64(w * w));
                    }
                    // SAFETY: the arguments follow the generated signature (with omega^2 last
                    // for the erf kernels); the kernel reads the class's records of pairs
                    // c0..c1 (offset pointers), the auxiliary data of shells s0..s0 + nsh of l
                    // (offset pointers; a block's columns are relative to its first function,
                    // below its width ld) and the tables of its shape (checked against the
                    // generated hash). It writes rows rows.start.. of every plane (ld columns)
                    // inside `out` from the offset on (checked above), or, with a Gamma, reads
                    // those rows of Gamma (checked in `gradient`) and adds to the accumulator
                    // elements 3 atom + x below 3 natom <= its length.
                    unsafe { gpu.launch(kernel, launch, &args)? };
                }
            }
            Ok(())
        }

        /// The kernel of class (la lb | lk), if the builder has work for it.
        pub fn kernel(
            &self,
            la: usize,
            lb: usize,
            lk: usize,
        ) -> Option<(&'static GenKernel, &Kernel)> {
            self.kernels
                .get(la * (la + 1) / 2 + lb)
                .and_then(|row| row.get(lk))
                .and_then(|k| k.as_ref())
                .map(|k| (k.gk, &k.kernel))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness;

    /// ptxas -O1 for the second-derivative kernels on cc 9.x only ([`compile_options`]): the
    /// other kinds, and every kind on other architectures, keep the base options.
    #[test]
    fn second_derivatives_take_ptxas_o1_on_hopper_only() {
        for cc in [(9, 0), (9, 1)] {
            assert_eq!(
                compile_options(cc, Eri3cKind::Hessian),
                ["--ptxas-options=-O1"]
            );
            assert!(compile_options(cc, Eri3cKind::Integrals).is_empty());
            assert!(compile_options(cc, Eri3cKind::Derivatives).is_empty());
        }
        for cc in [(8, 0), (8, 9), (10, 0), (12, 0)] {
            for kind in [
                Eri3cKind::Integrals,
                Eri3cKind::Derivatives,
                Eri3cKind::Hessian,
            ] {
                assert!(compile_options(cc, kind).is_empty(), "{cc:?} {kind:?}");
            }
        }
    }

    /// The packing is the same, bit for bit, on one thread and on several (its pairs are
    /// packed in parallel and appended in pair order), for every kind, more pairs than one
    /// parallel step.
    #[test]
    fn packing_is_independent_of_the_threads() {
        let mut mol = harness::structure("u1an39").unwrap();
        mol.z.truncate(12);
        mol.positions.truncate(12);
        let basis = harness::basis(&mol, "def2-tzvp").unwrap();
        let aux = harness::basis(&mol, "def2-universal-jkfit").unwrap();
        let pairs = all_pairs(&basis);
        assert!(pairs.len() > PACK_CHUNK);
        let pack = |threads: usize, kind: Eri3cKind| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| Eri3cHost::new(&basis, &aux, &pairs, kind, 1e-16).unwrap())
        };
        for kind in [
            Eri3cKind::Integrals,
            Eri3cKind::Derivatives,
            Eri3cKind::Hessian,
        ] {
            let (a, b) = (pack(1, kind), pack(5, kind));
            assert_eq!(a.classes.len(), b.classes.len());
            for (x, y) in a.classes.iter().zip(&b.classes) {
                assert_eq!(
                    (&x.pairs, &x.p0, &x.out, &x.atom),
                    (&y.pairs, &y.p0, &y.out, &y.atom)
                );
                let bits = |v: &[f64]| v.iter().map(|z| z.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&x.d), bits(&y.d), "{kind:?} class {} {}", x.la, x.lb);
            }
        }
    }
}
