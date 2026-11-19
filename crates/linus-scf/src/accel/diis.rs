//! DIIS extrapolation of Fock matrices (Pulay): the subspace of Fock and error matrices with its
//! drop rule ([`DiisDrop`], and the unpinning of [`crate::ScfAlgorithm::diis_unpin`]), the
//! extrapolation, and the subspace shift of a changed Fock operator.

use crate::settings::{DiisDrop, ScfAlgorithm};
use faer::{Mat, MatRef, Par};
use linus_base::linalg::{dot, eigh, mm};

/// When the DIIS subspace takes an iteration's Fock matrix and extrapolates, and when its drop
/// rule unpins (`diis_space`, `diis_start`, `diis_unpin`): one rule for both SCF loops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiisRule {
    space: usize,
    start: usize,
    unpin: f64,
}

impl DiisRule {
    pub fn new(a: &ScfAlgorithm) -> Self {
        DiisRule {
            space: a.diis_space,
            start: a.diis_start,
            unpin: a.diis_unpin,
        }
    }

    /// Iteration `iter`'s Fock matrix enters the subspace (`usable`: not a guess density's).
    pub fn admits(&self, iter: usize, usable: bool) -> bool {
        usable && iter >= self.start && self.space > 0
    }

    /// A subspace of `stored` entries is extrapolated at iteration `iter`.
    pub fn extrapolates(&self, iter: usize, usable: bool, stored: usize) -> bool {
        stored > 0 && usable && iter >= self.start
    }

    /// A new entry of squared error norm `norm` beside entries of squared norms `stored`: the drop
    /// rule `drop` switches from LargestError to Oldest, since the new error is `diis_unpin` times
    /// the smallest stored one (the SCF left the region of the best stored entries, which would pin
    /// the extrapolation there).
    pub fn unpins(&self, drop: DiisDrop, norm: f64, stored: &[f64]) -> bool {
        let emin = stored.iter().copied().fold(f64::INFINITY, f64::min);
        // the norms are squared
        drop == DiisDrop::LargestError
            && self.unpin > 0.0
            && !stored.is_empty()
            && norm > self.unpin * self.unpin * emin
    }
}

pub struct Diis {
    pub(crate) max: usize,
    pub(crate) drop: DiisDrop,
    pub(crate) focks: Vec<Vec<Mat<f64>>>,
    pub(crate) errs: Vec<Vec<Mat<f64>>>,
    /// Occupied orbitals of each entry's spin densities (D_s = C_s C_s^T; empty if not known),
    /// for [`Diis::shift`].
    pub(crate) occs: Vec<Vec<Mat<f64>>>,
}

impl Diis {
    pub(crate) fn new(max: usize) -> Self {
        Diis {
            max,
            drop: DiisDrop::LargestError,
            focks: vec![],
            errs: vec![],
            occs: vec![],
        }
    }
    /// An empty subspace of `max` entries with the drop rule `drop`.
    pub(crate) fn with_drop(max: usize, drop: DiisDrop) -> Self {
        Diis {
            drop,
            ..Diis::new(max)
        }
    }
    pub(crate) fn push(&mut self, f: Vec<Mat<f64>>, e: Vec<Mat<f64>>) {
        self.push_occ(f, e, vec![]);
    }
    /// The squared error norms of the entries, in order.
    pub(crate) fn norms(&self) -> Vec<f64> {
        self.errs.iter().map(|x| err_norm(x)).collect()
    }
    /// [`Diis::push`] with the occupied orbitals of the entry's densities.
    pub(crate) fn push_occ(&mut self, f: Vec<Mat<f64>>, e: Vec<Mat<f64>>, occ: Vec<Mat<f64>>) {
        if self.focks.len() == self.max {
            let out = self
                .drop
                .victim(&self.norms(), self.max)
                .expect("a full subspace");
            self.focks.remove(out);
            self.errs.remove(out);
            self.occs.remove(out);
        }
        self.focks.push(f);
        self.errs.push(e);
        self.occs.push(occ);
    }

    /// Carry the subspace over to a changed Fock operator F' = F + dF (per spin, dF taken at
    /// the current density): F_i += dF and e_i += X^T (dF D_i S - S D_i dF) X. Entries without
    /// orbitals are dropped.
    pub(crate) fn shift(
        &mut self,
        df: &[Mat<f64>],
        s: MatRef<'_, f64>,
        x: MatRef<'_, f64>,
        par: Par,
    ) {
        let keep: Vec<usize> = (0..self.focks.len())
            .filter(|&i| self.occs[i].len() == df.len())
            .collect();
        let take = |v: &mut Vec<Vec<Mat<f64>>>| -> Vec<Vec<Mat<f64>>> {
            keep.iter().map(|&i| std::mem::take(&mut v[i])).collect()
        };
        (self.focks, self.errs, self.occs) = (
            take(&mut self.focks),
            take(&mut self.errs),
            take(&mut self.occs),
        );
        for i in 0..self.focks.len() {
            for (sp, d) in df.iter().enumerate() {
                self.focks[i][sp] += d;
                // dF D S - S D dF = P Q^T - Q P^T with P = dF C, Q = S C
                let c = &self.occs[i][sp];
                let p = mm(x.transpose(), mm(d.as_ref(), c.as_ref(), par).as_ref(), par);
                let q = mm(x.transpose(), mm(s, c.as_ref(), par).as_ref(), par);
                let pq = mm(p.as_ref(), q.transpose(), par);
                self.errs[i][sp] += &pq - pq.transpose();
            }
        }
    }

    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    pub(crate) fn extrapolate(&self) -> Vec<Mat<f64>> {
        let n = self.focks.len();
        let mut b = Mat::<f64>::zeros(n + 1, n + 1);
        for i in 0..n {
            for j in 0..=i {
                let v: f64 = self.errs[i]
                    .iter()
                    .zip(&self.errs[j])
                    .map(|(a, c)| dot(a.as_ref(), c.as_ref()))
                    .sum();
                b[(i, j)] = v;
                b[(j, i)] = v;
            }
            b[(i, n)] = -1.0;
            b[(n, i)] = -1.0;
        }
        // scale for conditioning
        let s = (0..n).map(|i| b[(i, i)]).fold(0.0, f64::max).max(1e-300);
        for i in 0..n {
            for j in 0..n {
                b[(i, j)] /= s;
            }
        }
        let (w, u) = eigh(b.as_ref(), Par::Seq);
        let mut rhs = vec![0.0; n + 1];
        rhs[n] = -1.0;
        let mut x = vec![0.0; n + 1];
        let wmax = w.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        for k in 0..=n {
            if w[k].abs() < 1e-14 * wmax {
                continue;
            }
            let proj: f64 = (0..=n).map(|i| u[(i, k)] * rhs[i]).sum::<f64>() / w[k];
            for i in 0..=n {
                x[i] += proj * u[(i, k)];
            }
        }
        let nsp = self.focks[0].len();
        let nb = self.focks[0][0].nrows();
        (0..nsp)
            .map(|sp| {
                let mut f = Mat::<f64>::zeros(nb, nb);
                for i in 0..n {
                    f += &self.focks[i][sp] * x[i];
                }
                f
            })
            .collect()
    }
}

pub fn err_norm(e: &[Mat<f64>]) -> f64 {
    e.iter().map(|m| dot(m.as_ref(), m.as_ref())).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use linus_base::linalg::{canonical_orthogonalizer, max_abs};

    /// Shifting the DIIS subspace by dF equals pushing F + dF with recomputed errors.
    #[test]
    fn diis_shift_equals_rebuilt_subspace() {
        let (n, no) = (6, 2);
        let sym = |seed: f64| {
            let a = Mat::from_fn(n, n, |i, j| ((i * 5 + j * 3) as f64 * seed).sin());
            &a + a.transpose()
        };
        let s = Mat::from_fn(n, n, |i, j| {
            if i == j {
                1.0
            } else {
                0.05 / (1.0 + (i + j) as f64)
            }
        });
        let x = canonical_orthogonalizer(s.as_ref(), 1e-10, Par::Seq);
        let err = |f: &Mat<f64>, c: &Mat<f64>| {
            let d = mm(c.as_ref(), c.transpose(), Par::Seq);
            let fds = mm(
                mm(f.as_ref(), d.as_ref(), Par::Seq).as_ref(),
                s.as_ref(),
                Par::Seq,
            );
            let e = &fds - fds.transpose();
            mm(
                x.transpose(),
                mm(e.as_ref(), x.as_ref(), Par::Seq).as_ref(),
                Par::Seq,
            )
        };
        let df = sym(0.71) * 1e-2;
        let (mut a, mut b) = (Diis::new(4), Diis::new(4));
        for k in 0..3 {
            let f = sym(0.3 + 0.1 * k as f64);
            let c = Mat::from_fn(n, no, |i, j| ((i + 2 * j + k) as f64 * 0.37).cos());
            a.push_occ(vec![f.clone()], vec![err(&f, &c)], vec![c.clone()]);
            let f2 = &f + &df;
            b.push_occ(vec![f2.clone()], vec![err(&f2, &c)], vec![c]);
        }
        // an entry without orbitals is dropped by the shift
        a.push(vec![sym(0.9)], vec![sym(0.4)]);
        a.shift(&[df], s.as_ref(), x.as_ref(), Par::Seq);
        assert_eq!(a.focks.len(), 3);
        for i in 0..3 {
            for (p, q) in [
                (&a.focks[i][0], &b.focks[i][0]),
                (&a.errs[i][0], &b.errs[i][0]),
            ] {
                assert!(max_abs((p - q).as_ref()) < 1e-13);
            }
        }
        let (fa, fb) = (a.extrapolate(), b.extrapolate());
        assert!(max_abs((&fa[0] - &fb[0]).as_ref()) < 1e-11);
    }

    /// The unpinning rule (`diis_unpin`, the HBr+ pi-hole stall of tests/openshell_scf.rs): a new
    /// error more than `diis_unpin` times the smallest stored one (squared norms) switches the
    /// LargestError drop rule to Oldest; never with the rule at 0, never for Oldest, never on an
    /// empty subspace.
    #[test]
    fn a_new_error_far_above_the_best_unpins_the_subspace() {
        let rule = |unpin: f64| {
            DiisRule::new(&ScfAlgorithm {
                diis_unpin: unpin,
                ..Default::default()
            })
        };
        let stored = [4e-14, 1e-12, 3e-10];
        let (r10, r0) = (rule(10.0), rule(0.0));
        // squared norms: 100 x the smallest unpins, below it does not
        assert!(r10.unpins(DiisDrop::LargestError, 4.1e-12, &stored));
        assert!(!r10.unpins(DiisDrop::LargestError, 3.9e-12, &stored));
        assert!(!r10.unpins(DiisDrop::Oldest, 1.0, &stored));
        assert!(!r10.unpins(DiisDrop::LargestError, 1.0, &[]));
        // kept for good: the pinned subspace whatever the new error
        assert!(!r0.unpins(DiisDrop::LargestError, 1.0, &stored));
    }
}
