//! Pieces shared by the density-fitted (RI) code of the gradient ([`crate::df::gradient`]), the
//! Hessian terms ([`crate::df::hessian`]), the response J/K ([`crate::df::response`]) and the SCF
//! J/K ([`crate::df`]): blocks of whole auxiliary shells and the three-center integrals of
//! the significant orbital shell pairs ([`Mo3c`]).

use faer::{Accum, Mat, MatMut, MatRef, Par};
use linus_base::linalg::{matmul, solve_lower_in_place, solve_upper_in_place};
use linus_ints::basis::Shell;
use linus_ints::eri3c::{self, Aux3c, Pair3c, Work3c};
use linus_ints::operator::Operator;
use linus_ints::{BasisSet, PairSet};
use rayon::prelude::*;
use std::ops::Range;

/// A block of consecutive whole auxiliary shells.
#[derive(Clone, Debug)]
pub(crate) struct AuxBlock {
    /// First auxiliary function of the block.
    pub p0: usize,
    /// Number of auxiliary functions.
    pub nq: usize,
    pub shells: Range<usize>,
    /// The atom of all of its shells for blocks by atom, else usize::MAX.
    pub atom: usize,
}

/// Consecutive blocks of whole auxiliary shells with at most `maxf` functions each (at least one
/// shell per block), never spanning two atoms if `by_atom`.
pub(crate) fn aux_blocks(shells: &[Shell], maxf: usize, by_atom: bool) -> Vec<AuxBlock> {
    let mut out = vec![];
    let block = |r: Range<usize>| {
        let p0 = shells[r.start].ao_start;
        let last = &shells[r.end - 1];
        AuxBlock {
            p0,
            nq: last.ao_start + last.nfunc() - p0,
            atom: if by_atom {
                shells[r.start].atom
            } else {
                usize::MAX
            },
            shells: r,
        }
    };
    let mut s0 = 0;
    let mut nf = 0;
    for (s, sh) in shells.iter().enumerate() {
        let new_atom = by_atom && s > s0 && shells[s0].atom != sh.atom;
        if nf > 0 && (nf + sh.nfunc() > maxf || new_atom) {
            out.push(block(s0..s));
            s0 = s;
            nf = 0;
        }
        nf += sh.nfunc();
    }
    if s0 < shells.len() {
        out.push(block(s0..shells.len()));
    }
    out
}

/// The operator of a density-fitting tensor in the DF code's convention for `omega`: 1/r for
/// omega = 0, else the long-range erf(omega r)/r.
pub(crate) fn tensor_operator(omega: f64) -> Operator {
    if omega == 0.0 {
        Operator::Coulomb
    } else {
        Operator::LongRange { omega }
    }
}

/// The triangular solves of [`solve_metric_mmajor`] with the metric's Cholesky factor L.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MetricSolve {
    /// L Y = B: half-transformed three-center quantities L^-1 B.
    Lower,
    /// V X = L L^T X = B: fit coefficients V^-1 B.
    Full,
}

/// `solve` in place for every slab of an m-major [m][P][i] buffer `z` (each slab naux x o, naux
/// the order of `l`). Slabs go in chunks of max(256 / o, 1), each chunk transposed into one
/// right-hand side matrix and solved sequentially, so the result does not depend on the thread
/// count.
pub(crate) fn solve_metric_mmajor(l: MatRef<'_, f64>, z: &mut [f64], o: usize, solve: MetricSolve) {
    let naux = l.nrows();
    if o == 0 || z.is_empty() {
        return;
    }
    let mc = (256 / o).max(1);
    z.par_chunks_mut(mc * naux * o).for_each(|zc| {
        let nm = zc.len() / (naux * o);
        let mut t = Mat::<f64>::zeros(naux, nm * o);
        for mm in 0..nm {
            for p in 0..naux {
                for i in 0..o {
                    t[(p, mm * o + i)] = zc[mm * naux * o + p * o + i];
                }
            }
        }
        solve_lower_in_place(l, t.as_mut(), Par::Seq);
        if solve == MetricSolve::Full {
            solve_upper_in_place(l.transpose(), t.as_mut(), Par::Seq);
        }
        for mm in 0..nm {
            for p in 0..naux {
                for i in 0..o {
                    zc[mm * naux * o + p * o + i] = t[(p, mm * o + i)];
                }
            }
        }
    });
}

/// Raw pointer shared by tasks writing disjoint parts of a buffer.
#[derive(Clone, Copy)]
pub(crate) struct SendPtr(pub(crate) *mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    pub(crate) fn get(self) -> *mut f64 {
        self.0
    }
}

/// Exchange panel of one orbital shell: its significant partner shells and the stored rows.
struct Panel {
    f0: usize,
    nf: usize,
    /// (partner shell, pair index, stored transposed)
    partners: Vec<(usize, usize, bool)>,
    /// partner functions in partner order
    flist: Vec<usize>,
}

/// Three-center integrals of the significant shell pairs, one auxiliary block at a time.
pub(crate) struct Mo3c<'a> {
    nbf: usize,
    shells: Vec<(usize, usize)>,
    pairs: Vec<Pair3c>,
    x0: Vec<usize>,
    pub(crate) nx: usize,
    panels: Vec<Panel>,
    aux3c: std::borrow::Cow<'a, Aux3c>,
}

impl Mo3c<'static> {
    /// [`Mo3c::new`] owning its auxiliary data (for builders that outlive the caller's).
    pub(crate) fn new_owned(basis: &BasisSet, pairs: &PairSet, aux3c: Aux3c) -> Self {
        let mut m = Mo3c::new(basis, pairs, &aux3c);
        let parts = (
            m.nbf,
            std::mem::take(&mut m.shells),
            std::mem::take(&mut m.pairs),
            std::mem::take(&mut m.x0),
            m.nx,
            std::mem::take(&mut m.panels),
        );
        drop(m);
        Mo3c {
            nbf: parts.0,
            shells: parts.1,
            pairs: parts.2,
            x0: parts.3,
            nx: parts.4,
            panels: parts.5,
            aux3c: std::borrow::Cow::Owned(aux3c),
        }
    }
}

impl<'a> Mo3c<'a> {
    /// Dense copy of pair-major integrals (`buf`, nq auxiliary functions), m-major:
    /// out[(m * nq + P) * nbf + n] = (P|mn) for both triangles; pairs not stored are zero.
    pub(crate) fn dense_mmajor(&self, buf: &[f64], nq: usize, out: &mut [f64]) {
        let n = self.nbf;
        let out = &mut out[..n * nq * n];
        out.par_chunks_mut(1 << 16).for_each(|c| c.fill(0.0));
        let ptr = SendPtr(out.as_mut_ptr());
        self.pairs.par_iter().zip(&self.x0).for_each(|(p, &x0)| {
            let (a0, b0) = (self.shells[p.sa].0, self.shells[p.sb].0);
            // SAFETY: the (m, n) and (n, m) entries of a shell pair belong to that pair only
            let o = ptr.get();
            for ia in 0..p.na {
                for ib in 0..p.nb {
                    let row = &buf[(x0 + ia * p.nb + ib) * nq..(x0 + ia * p.nb + ib + 1) * nq];
                    let (m, nn) = (a0 + ia, b0 + ib);
                    for (q, &v) in row.iter().enumerate() {
                        unsafe {
                            *o.add((m * nq + q) * n + nn) = v;
                            *o.add((nn * nq + q) * n + m) = v;
                        }
                    }
                }
            }
        });
    }

    pub(crate) fn new(basis: &BasisSet, pairs: &PairSet, aux3c: &'a Aux3c) -> Self {
        let q = pairs.schwarz();
        let kept: Vec<usize> = (0..pairs.len())
            .filter(|&i| pairs.pair(i).nprim > 0 && q[i] * aux3c.qmax >= 1e-14)
            .collect();
        let p3: Vec<Pair3c> = kept
            .par_iter()
            .map(|&i| Pair3c::from_shell_pair(pairs.pair(i)))
            .collect();
        let mut x0 = Vec::with_capacity(p3.len());
        let mut nx = 0;
        for p in &p3 {
            x0.push(nx);
            nx += p.nab();
        }
        let shells: Vec<(usize, usize)> = basis
            .shells
            .iter()
            .map(|s| (s.ao_start, s.nfunc()))
            .collect();
        let ns = shells.len();
        let mut partners: Vec<Vec<(usize, usize, bool)>> = vec![vec![]; ns];
        for (i, p) in p3.iter().enumerate() {
            partners[p.sa].push((p.sb, i, false));
            if p.sa != p.sb {
                partners[p.sb].push((p.sa, i, true));
            }
        }
        let panels = (0..ns)
            .map(|a| {
                let mut ps = partners[a].clone();
                ps.sort_unstable_by_key(|x| x.0);
                let flist = ps
                    .iter()
                    .flat_map(|&(b, _, _)| shells[b].0..shells[b].0 + shells[b].1)
                    .collect();
                Panel {
                    f0: shells[a].0,
                    nf: shells[a].1,
                    partners: ps,
                    flist,
                }
            })
            .collect();
        Mo3c {
            nbf: basis.nbf,
            shells,
            pairs: p3,
            x0,
            nx,
            panels,
            aux3c: std::borrow::Cow::Borrowed(aux3c),
        }
    }

    /// Integrals (P|op|mn) for the auxiliary shells `ash` (functions relative to the first),
    /// pair-major: buf[row * nq + P] for the stored rows (op: Coulomb if omega = 0, else erf).
    pub(crate) fn ints(&self, ash: std::ops::Range<usize>, nq: usize, omega: f64, buf: &mut [f64]) {
        let chunks = self.aux3c.chunks(ash);
        let op = tensor_operator(omega);
        let buf = &mut buf[..self.nx * nq];
        buf.par_chunks_mut(1 << 16).for_each(|c| c.fill(0.0));
        let mut slabs: Vec<(&Pair3c, &mut [f64])> = Vec::with_capacity(self.pairs.len());
        let mut rest: &mut [f64] = buf;
        for p in &self.pairs {
            let (a, b) = rest.split_at_mut(p.nab() * nq);
            slabs.push((p, a));
            rest = b;
        }
        slabs
            .into_par_iter()
            .with_min_len(4)
            .for_each_init(Work3c::new, |w, (p, slab)| {
                eri3c::pair_block(p, &chunks, slab, nq, 1e-16, &op, w);
            });
    }

    /// D_mn per stored row, times 2 for rows of off-diagonal shell pairs (they stand for (m,n)
    /// and (n,m)); `d` is row-major nbf x nbf.
    pub(crate) fn density_rows(&self, d: &[f64]) -> Vec<f64> {
        let n = self.nbf;
        let mut w = vec![0.0; self.nx];
        for (p, &x0) in self.pairs.iter().zip(&self.x0) {
            let (a0, b0) = (self.shells[p.sa].0, self.shells[p.sb].0);
            let f = if p.sa == p.sb { 1.0 } else { 2.0 };
            for ia in 0..p.na {
                for ib in 0..p.nb {
                    w[x0 + ia * p.nb + ib] = f * d[(a0 + ia) * n + b0 + ib];
                }
            }
        }
        w
    }

    /// Half transformation Z[m][(P, i)] = sum_n (P|mn) C_ni from pair-major integrals; `c` is
    /// row-major nbf x o. Returns nbf x (nq o) row-major.
    pub(crate) fn half(&self, buf: &[f64], nq: usize, c: &[f64], o: usize) -> Vec<f64> {
        let mut z = vec![];
        self.half_into(buf, nq, c, o, &mut z);
        z
    }

    /// [`Mo3c::half`] into a reused buffer (resized to nbf x (nq o); every row is written).
    pub(crate) fn half_into(&self, buf: &[f64], nq: usize, c: &[f64], o: usize, z: &mut Vec<f64>) {
        let n = self.nbf;
        z.resize(n * nq * o, 0.0);
        let z = &mut z[..n * nq * o];
        if o == 0 {
            return;
        }
        let mut slabs: Vec<(&Panel, &mut [f64])> = Vec::with_capacity(self.panels.len());
        let mut rest: &mut [f64] = z;
        for pan in &self.panels {
            let (a, b) = rest.split_at_mut(pan.nf * nq * o);
            slabs.push((pan, a));
            rest = b;
        }
        // auxiliary sub-blocks whose gathered panel stays in L2 (whole panels stream the
        // gathered panel through memory)
        slabs.into_par_iter().for_each_init(
            || (Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()),
            |(tb, cb, zc), (pan, zs)| {
                let nl = pan.flist.len();
                if nl == 0 {
                    zs.fill(0.0);
                    return;
                }
                cb.resize(nl * o, 0.0);
                for (i, &f) in pan.flist.iter().enumerate() {
                    cb[i * o..(i + 1) * o].copy_from_slice(&c[f * o..(f + 1) * o]);
                }
                let cm = MatRef::from_row_major_slice(&cb[..nl * o], nl, o);
                let nfa = pan.nf;
                // ~96k doubles (768 KB) of gathered panel per sub-block
                let qc = (98304 / (nfa * nl).max(1)).clamp(8, nq.max(8)).min(nq);
                let mut q0 = 0;
                while q0 < nq {
                    let nqc = qc.min(nq - q0);
                    let mr = nfa * nqc;
                    // T[(fa, P'), n'] column-major mr x nl for P' in q0..q0+nqc
                    tb.resize(mr * nl, 0.0);
                    let mut noff = 0;
                    for &(bsh, pi, tr) in &pan.partners {
                        let nb = self.shells[bsh].1;
                        let x0 = self.x0[pi];
                        for ib in 0..nb {
                            let col = &mut tb[(noff + ib) * mr..(noff + ib + 1) * mr];
                            for ia in 0..nfa {
                                let x = if tr {
                                    x0 + ib * nfa + ia
                                } else {
                                    x0 + ia * nb + ib
                                };
                                col[ia * nqc..(ia + 1) * nqc]
                                    .copy_from_slice(&buf[x * nq + q0..x * nq + q0 + nqc]);
                            }
                        }
                        noff += nb;
                    }
                    zc.resize(mr * o, 0.0);
                    let tm = MatRef::from_column_major_slice(&tb[..mr * nl], mr, nl);
                    matmul(
                        MatMut::from_row_major_slice_mut(&mut zc[..mr * o], mr, o),
                        Accum::Replace,
                        tm,
                        cm,
                        1.0,
                        Par::Seq,
                    );
                    for ia in 0..nfa {
                        let dst = &mut zs[(ia * nq + q0) * o..(ia * nq + q0 + nqc) * o];
                        dst.copy_from_slice(&zc[ia * nqc * o..(ia + 1) * nqc * o]);
                    }
                    q0 += nqc;
                }
            },
        );
        let _ = self.panels.first().map(|p| p.f0);
    }
}
