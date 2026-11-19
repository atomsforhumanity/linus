//! Contractions of first-derivative three-center integrals (P|mn)^x and two-center
//! integrals (P|Q)^x for all 3 natom nuclear perturbations, without storing derivative
//! tensors: the integral layer of density-fitted derivative Fock matrices (RI-J^x, RI-K^x)
//! and of density-fitted Hessians.
//!
//! For x = (atom A, k), with ∂_k P the derivative of an auxiliary function on A and ∂_k m
//! that of an AO function on A,
//!
//!   (P|mn)^x = δ(P∈A) (∂P|mn) + δ(m∈A) (∂m n|P) + δ(n∈A) (∂n m|P),
//!
//! so per atom two kinds of first-center derivative blocks suffice (the
//! [`crate::eri4c::jk_deriv::eri_quartet_deriv_a`] kernel): auxiliary-center blocks (∂P|MN) for P
//! on A and all AO shell pairs M >= N, and AO-center blocks (∂a b|P) for ordered AO pairs
//! (a on A, b any) and all auxiliary shells. Atoms are processed one at a time; within an
//! atom the work is split over auxiliary shells, each of which owns its rows of the
//! per-auxiliary-function outputs (γ^x, Λ^x), so only the nbf x nbf (or nbf x r) outputs
//! need per-thread copies.
//!
//! [`df3c_deriv_atoms`] contracts the blocks with any combination of
//! - auxiliary vectors c:       M^x_mn = sum_P (P|mn)^x c_P                 (nbf x nbf)
//! - symmetric AO densities D:  γ^x_P  = sum_mn (P|mn)^x D_mn               (naux)
//! - right factors Y_P (nbf x r): Y^x_mj = sum_P sum_l (P|ml)^x Y_P,lj      (nbf x r)
//! - orbitals C (nbf x o):      Λ^x_P,ij = sum_ml C_mi (P|ml)^x C_lj        (naux x o x o)
//!
//! and [`super::oneshot::metric_deriv_rows`] / [`apply_metric_deriv`] give
//! V^x X = sum_Q (P|Q)^x X_Q.

use crate::basis::BasisSet;
use crate::core::deriv::PairDeriv;
use crate::core::hermite::{ShellPair, ShellView};
use crate::core::kernels::{wmul_acc, wmul_t_acc};
use crate::core::pair_data::{self, PRIM_PAIR_CUT};
use crate::eri4c::deriv::schwarz_deriv;
use crate::eri4c::jk_deriv::{eri_quartet_deriv_a, DerivAWork};
use crate::pairs::PairSet;
use rayon::prelude::*;

/// Inputs of [`df3c_deriv_atoms`]; every field is optional (empty / None).
#[derive(Default, Clone)]
pub struct Df3cInput<'a> {
    /// Auxiliary vectors c (length naux) -> M^x = sum_P (P|mn)^x c_P.
    pub vecs: Vec<&'a [f64]>,
    /// Symmetric AO matrices D (nbf x nbf, row-major) -> γ^x_P = sum_mn (P|mn)^x D_mn.
    pub dens: Vec<&'a [f64]>,
    /// Right factors Y laid out \[P\]\[l\]\[j\] (naux x nbf x r) and r ->
    /// Y^x_mj = sum_P sum_l (P|ml)^x Y_P,lj.
    pub right: Option<(&'a [f64], usize)>,
    /// Orbitals C laid out \[m\]\[i\] (nbf x o, row-major) and o -> Λ^x_P,ij.
    pub orbitals: Option<(&'a [f64], usize)>,
}

/// Contractions for the three perturbations x = (atom, k) of one atom.
pub struct Df3cAtom {
    pub atom: usize,
    /// Per input vector: 3 x nbf x nbf (row-major, symmetric).
    pub m: Vec<Vec<f64>>,
    /// Per input density: 3 x naux.
    pub gamma: Vec<Vec<f64>>,
    /// 3 x nbf x r (empty without right factors).
    pub right: Vec<f64>,
    /// 3 x naux x o x o (empty without orbitals).
    pub lambda: Vec<f64>,
}

/// Per-thread accumulators of the nbf x nbf (nbf x r) outputs.
struct SharedAcc {
    /// per vector: 3 x n x n, auxiliary-center part (both triangles)
    maux: Vec<Vec<f64>>,
    /// per vector: 3 x n x n, AO first-center part (rows on the atom), symmetrized at the end
    mh: Vec<Vec<f64>>,
    /// 3 x n x r
    right: Vec<f64>,
}

impl SharedAcc {
    fn new(nv: usize, n: usize, r: usize) -> Self {
        SharedAcc {
            maux: vec![vec![0.0; 3 * n * n]; nv],
            mh: vec![vec![0.0; 3 * n * n]; nv],
            right: vec![0.0; 3 * n * r],
        }
    }
    fn add(mut self, o: SharedAcc) -> Self {
        for (a, b) in self
            .maux
            .iter_mut()
            .zip(&o.maux)
            .chain(self.mh.iter_mut().zip(&o.mh))
        {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        }
        self.right
            .iter_mut()
            .zip(&o.right)
            .for_each(|(x, y)| *x += y);
        self
    }
}

#[inline]
fn axpy(dst: &mut [f64], a: f64, src: &[f64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += a * s;
    }
}

/// Three-center derivative contractions for the atoms in `atoms` (see the module docs).
/// `apairs` are the auxiliary pairs ([`PairSet::aux`]) of derivative order >= 1
/// (order >= 1) of `aux`, which must be centered on the same atoms as `basis`. Blocks whose
/// Schwarz-type bound times the largest input magnitude is below `thresh` are skipped.
#[allow(clippy::too_many_arguments)]
pub fn df3c_deriv_atoms(
    basis: &BasisSet,
    pairs: &PairSet,
    aux: &BasisSet,
    apairs: &PairSet,
    input: &Df3cInput,
    atoms: &[usize],
    thresh: f64,
) -> Vec<Df3cAtom> {
    let n = basis.nbf;
    let n2 = n * n;
    let naux = aux.nbf;
    let sh = &basis.shells;
    let ash = &aux.shells;
    let ns = sh.len();
    let nv = input.vecs.len();
    let nd = input.dens.len();
    let (yv, r) = input.right.unwrap_or((&[], 0));
    let (cv, o) = input.orbitals.unwrap_or((&[], 0));
    assert!(yv.len() >= naux * n * r && cv.len() >= n * o);
    let maxabs = |v: &[f64]| v.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    let scale = input
        .vecs
        .iter()
        .chain(input.dens.iter())
        .map(|v| maxabs(v))
        .chain([maxabs(yv), maxabs(cv)])
        .fold(0.0f64, f64::max)
        .max(1e-300);
    let views: Vec<ShellView> = sh.iter().map(ShellView::from_shell).collect();
    let q_ao = pairs.schwarz();
    let q_aux = apairs.schwarz();
    let q1_aux = apairs.deriv_bounds().0;
    atoms
        .iter()
        .map(|&atom| {
            let aor = basis.atom_functions(atom);
            let na_at = aor.len();
            // ordered AO pairs (a on atom, b any) with first-center derivative data and bounds
            let bras: Vec<(usize, usize, ShellPair, PairDeriv, f64)> = basis
                .atom_shells
                .get(atom)
                .cloned()
                .unwrap_or(0..0)
                .flat_map(|a| (0..ns).map(move |b| (a, b)))
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(a, b)| {
                    let (pr, pd) = pair_data::build(&views[a], &views[b], a, b, 1, PRIM_PAIR_CUT);
                    let pd = pd.expect("first-order pair data");
                    let q =
                        schwarz_deriv(std::slice::from_ref(&pr), std::slice::from_ref(&pd)).0[0];
                    (a, b, pr, pd, q)
                })
                .collect();
            let aux_on: std::ops::Range<usize> = aux.atom_shells.get(atom).cloned().unwrap_or(0..0);
            // per auxiliary shell: (gamma rows [nd][3][np], lambda rows [3][np][o][o]) + shared
            // deterministic reduction over fixed chunks of auxiliary shells (independent of the
            // thread count)
            type St = (
                Vec<(usize, Vec<f64>, Vec<f64>)>,
                SharedAcc,
                DerivAWork,
                Vec<f64>,
                Vec<f64>,
                Vec<f64>,
            );
            let acc_len = nv * 6 * n2 + 3 * n * r;
            let nch = linus_num::det::nchunk_for(apairs.len(), acc_len, 1 << 24);
            let (rows, shared, ..): St = linus_num::det::par_fold_det(
                apairs.len(),
                nch,
                || {
                    (
                        Vec::new(),
                        SharedAcc::new(nv, n, r),
                        DerivAWork::new(),
                        Vec::<f64>::new(),
                        Vec::<f64>::new(),
                        Vec::<f64>::new(),
                    )
                },
                |st: &mut St, ps| {
                    let (rows, acc, w, buf, t, tmp) = st;
                    let ap = apairs.pair(ps);
                    let np = ap.na;
                    let p0 = ash[ps].ao_start;
                    if ap.nprim == 0 {
                        return;
                    }
                    let mut grow = vec![0.0; nd * 3 * np];
                    let mut lrow = vec![0.0; 3 * np * o * o];
                    // ---- auxiliary-center blocks (∂P|MN), P on the atom ----
                    if aux_on.contains(&ps) {
                        // T[k][p][m][j] = sum_l (∂p|ml) C_lj
                        t.clear();
                        t.resize(3 * np * n * o, 0.0);
                        for (kl, ket) in pairs.pairs().iter().enumerate() {
                            if ket.nprim == 0 || q1_aux[ps] * q_ao[kl] * scale < thresh {
                                continue;
                            }
                            let (sm, sn) = (ket.sa, ket.sb);
                            let (m0, n0) = (sh[sm].ao_start, sh[sn].ao_start);
                            let (nm, nn) = (ket.na, ket.nb);
                            let ncd = nm * nn;
                            let blk = 3 * np * ncd;
                            buf.resize(blk, 0.0);
                            eri_quartet_deriv_a(ap, apairs.deriv(ps), ket, buf, w);
                            let off = sm != sn;
                            let f = if off { 2.0 } else { 1.0 };
                            for k in 0..3 {
                                for p in 0..np {
                                    let b = &buf[(k * np + p) * ncd..(k * np + p + 1) * ncd];
                                    for (di, d) in input.dens.iter().enumerate() {
                                        let mut s = 0.0;
                                        for m in 0..nm {
                                            let drow =
                                                &d[(m0 + m) * n + n0..(m0 + m) * n + n0 + nn];
                                            for (x, y) in b[m * nn..(m + 1) * nn].iter().zip(drow) {
                                                s += x * y;
                                            }
                                        }
                                        grow[(di * 3 + k) * np + p] += f * s;
                                    }
                                    if r > 0 {
                                        // out[m] += sum_n b[m][n] Y_p[n] (and the (n, m) half)
                                        let y = &yv[(p0 + p) * n * r..(p0 + p + 1) * n * r];
                                        let out = &mut acc.right[k * n * r..(k + 1) * n * r];
                                        wmul_acc(
                                            b,
                                            nn,
                                            nm,
                                            nn,
                                            &y[n0 * r..],
                                            r,
                                            &mut out[m0 * r..],
                                        );
                                        if off {
                                            wmul_t_acc(
                                                b,
                                                nn,
                                                nm,
                                                nn,
                                                &y[m0 * r..],
                                                r,
                                                &mut out[n0 * r..],
                                            );
                                        }
                                    }
                                    if o > 0 {
                                        let tp =
                                            &mut t[(k * np + p) * n * o..(k * np + p + 1) * n * o];
                                        wmul_acc(
                                            b,
                                            nn,
                                            nm,
                                            nn,
                                            &cv[n0 * o..],
                                            o,
                                            &mut tp[m0 * o..],
                                        );
                                        if off {
                                            wmul_t_acc(
                                                b,
                                                nn,
                                                nm,
                                                nn,
                                                &cv[m0 * o..],
                                                o,
                                                &mut tp[n0 * o..],
                                            );
                                        }
                                    }
                                }
                                if nv > 0 {
                                    // tmp[m][n] = sum_p (∂p|mn) c_p
                                    for (vi, c) in input.vecs.iter().enumerate() {
                                        tmp.clear();
                                        tmp.resize(ncd, 0.0);
                                        for p in 0..np {
                                            axpy(
                                                &mut tmp[..ncd],
                                                c[p0 + p],
                                                &buf[(k * np + p) * ncd..(k * np + p + 1) * ncd],
                                            );
                                        }
                                        let mk = &mut acc.maux[vi][k * n2..(k + 1) * n2];
                                        for m in 0..nm {
                                            for nn_ in 0..nn {
                                                let v = tmp[m * nn + nn_];
                                                mk[(m0 + m) * n + n0 + nn_] += v;
                                                if off {
                                                    mk[(n0 + nn_) * n + m0 + m] += v;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if o > 0 {
                            // Λ[k][p] += C^T T[k][p]: lp[i][:] += sum_m C[m][i] T[m][:]
                            for kp in 0..3 * np {
                                let tp = &t[kp * n * o..(kp + 1) * n * o];
                                wmul_t_acc(
                                    cv,
                                    o,
                                    n,
                                    o,
                                    tp,
                                    o,
                                    &mut lrow[kp * o * o..(kp + 1) * o * o],
                                );
                            }
                        }
                    }
                    // ---- AO-center blocks (∂a b|P), a on the atom ----
                    if o > 0 {
                        // H[k][p][a][j] = sum_b (∂a b|p) C_bj (a relative to the atom)
                        t.clear();
                        t.resize(3 * np * na_at * o, 0.0);
                    }
                    for (sa, sb, bra, bd, qb) in &bras {
                        if bra.nprim == 0 || qb * q_aux[ps] * scale < thresh {
                            continue;
                        }
                        let (a0, b0) = (sh[*sa].ao_start, sh[*sb].ao_start);
                        let (nab_a, nab_b) = (bra.na, bra.nb);
                        let nab = nab_a * nab_b;
                        let blk = 3 * nab * np;
                        buf.resize(blk, 0.0);
                        eri_quartet_deriv_a(bra, bd, ap, buf, w);
                        for k in 0..3 {
                            let bk = &buf[k * nab * np..(k + 1) * nab * np];
                            for (di, d) in input.dens.iter().enumerate() {
                                for a in 0..nab_a {
                                    let drow = &d[(a0 + a) * n + b0..(a0 + a) * n + b0 + nab_b];
                                    for (b, &dv) in drow.iter().enumerate() {
                                        let row =
                                            &bk[(a * nab_b + b) * np..(a * nab_b + b + 1) * np];
                                        axpy(
                                            &mut grow[(di * 3 + k) * np..(di * 3 + k + 1) * np],
                                            2.0 * dv,
                                            row,
                                        );
                                    }
                                }
                            }
                            for (vi, c) in input.vecs.iter().enumerate() {
                                let cp = &c[p0..p0 + np];
                                let mk = &mut acc.mh[vi][k * n2..(k + 1) * n2];
                                for a in 0..nab_a {
                                    for b in 0..nab_b {
                                        let row =
                                            &bk[(a * nab_b + b) * np..(a * nab_b + b + 1) * np];
                                        mk[(a0 + a) * n + b0 + b] +=
                                            row.iter().zip(cp).map(|(x, y)| x * y).sum::<f64>();
                                    }
                                }
                            }
                            if r > 0 || o > 0 {
                                // bt[p][a][b] = (∂a b|p) for this k
                                tmp.clear();
                                tmp.resize(np * nab, 0.0);
                                for ab in 0..nab {
                                    for p in 0..np {
                                        tmp[p * nab + ab] = bk[ab * np + p];
                                    }
                                }
                                for p in 0..np {
                                    let bp = &tmp[p * nab..(p + 1) * nab];
                                    if r > 0 {
                                        // out[a] += sum_b bt[a][b] Y_p[b];  out[b] += sum_a bt[a][b] Y_p[a]
                                        let y = &yv[(p0 + p) * n * r..(p0 + p + 1) * n * r];
                                        let out = &mut acc.right[k * n * r..(k + 1) * n * r];
                                        wmul_acc(
                                            bp,
                                            nab_b,
                                            nab_a,
                                            nab_b,
                                            &y[b0 * r..],
                                            r,
                                            &mut out[a0 * r..],
                                        );
                                        wmul_t_acc(
                                            bp,
                                            nab_b,
                                            nab_a,
                                            nab_b,
                                            &y[a0 * r..],
                                            r,
                                            &mut out[b0 * r..],
                                        );
                                    }
                                    if o > 0 {
                                        let ar0 = a0 - aor.start;
                                        let hp = &mut t[(k * np + p) * na_at * o
                                            ..(k * np + p + 1) * na_at * o];
                                        wmul_acc(
                                            bp,
                                            nab_b,
                                            nab_a,
                                            nab_b,
                                            &cv[b0 * o..],
                                            o,
                                            &mut hp[ar0 * o..],
                                        );
                                    }
                                }
                            }
                        }
                    }
                    if o > 0 {
                        // Λ[k][p] += C_A^T H[k][p] + (C_A^T H[k][p])^T
                        tmp.clear();
                        tmp.resize(o * o, 0.0);
                        for kp in 0..3 * np {
                            tmp.fill(0.0);
                            let hp = &t[kp * na_at * o..(kp + 1) * na_at * o];
                            wmul_t_acc(&cv[aor.start * o..], o, na_at, o, hp, o, tmp);
                            let lp = &mut lrow[kp * o * o..(kp + 1) * o * o];
                            for i in 0..o {
                                for j in 0..o {
                                    lp[i * o + j] += tmp[i * o + j] + tmp[j * o + i];
                                }
                            }
                        }
                    }
                    rows.push((ps, grow, lrow));
                },
                |a, b| {
                    a.0.extend(b.0);
                    let acc = std::mem::replace(&mut a.1, SharedAcc::new(0, 0, 0));
                    a.1 = acc.add(b.1);
                },
            );
            // assemble
            let mut gamma = vec![vec![0.0; 3 * naux]; nd];
            let mut lambda = vec![0.0; 3 * naux * o * o];
            for (ps, grow, lrow) in rows {
                let np = apairs.pair(ps).na;
                let p0 = ash[ps].ao_start;
                for (di, g) in gamma.iter_mut().enumerate() {
                    for k in 0..3 {
                        g[k * naux + p0..k * naux + p0 + np]
                            .copy_from_slice(&grow[(di * 3 + k) * np..(di * 3 + k + 1) * np]);
                    }
                }
                for k in 0..3 {
                    for p in 0..np {
                        lambda[(k * naux + p0 + p) * o * o..(k * naux + p0 + p + 1) * o * o]
                            .copy_from_slice(&lrow[(k * np + p) * o * o..(k * np + p + 1) * o * o]);
                    }
                }
            }
            let m: Vec<Vec<f64>> = (0..nv)
                .map(|vi| {
                    let (ma, mh) = (&shared.maux[vi], &shared.mh[vi]);
                    let mut out = vec![0.0; 3 * n2];
                    for k in 0..3 {
                        for i in 0..n {
                            for j in 0..n {
                                out[k * n2 + i * n + j] = ma[k * n2 + i * n + j]
                                    + mh[k * n2 + i * n + j]
                                    + mh[k * n2 + j * n + i];
                            }
                        }
                    }
                    out
                })
                .collect();
            Df3cAtom {
                atom,
                m,
                gamma,
                right: shared.right,
                lambda,
            }
        })
        .collect()
}

/// V^{x} X for the three perturbations of one atom, from its
/// [`super::oneshot::metric_deriv_rows`] (plain
/// loops, for vectors or narrow X; use a GEMM for wide X):
/// X is naux x w (row-major); returns 3 x naux x w with
/// out\[k\]\[P\] = δ(P∈A) sum_Q (∂_k P|Q) X_Q + sum_{Q∈A} (∂_k Q|P) X_Q.
pub fn apply_metric_deriv(
    range: &std::ops::Range<usize>,
    rows: &[f64],
    naux: usize,
    x: &[f64],
    w: usize,
) -> Vec<f64> {
    let na = range.len();
    let mut out = vec![0.0; 3 * naux * w];
    out.par_chunks_mut(naux * w)
        .enumerate()
        .for_each(|(k, ok)| {
            // rows of the atom: sum_Q G[p][Q] X_Q
            for p in 0..na {
                let g = &rows[(k * na + p) * naux..(k * na + p + 1) * naux];
                let dst = &mut ok[(range.start + p) * w..(range.start + p + 1) * w];
                for (q, &gv) in g.iter().enumerate() {
                    if gv != 0.0 {
                        axpy(dst, gv, &x[q * w..(q + 1) * w]);
                    }
                }
            }
            // columns: out[P] += G[q][P] X_{q} for q on the atom
            for p in 0..na {
                let g = &rows[(k * na + p) * naux..(k * na + p + 1) * naux];
                let xq = &x[(range.start + p) * w..(range.start + p + 1) * w];
                for (q, &gv) in g.iter().enumerate() {
                    if gv != 0.0 {
                        axpy(&mut ok[q * w..(q + 1) * w], gv, xq);
                    }
                }
            }
        });
    out
}
