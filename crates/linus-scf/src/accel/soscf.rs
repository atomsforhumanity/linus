//! Second-order SCF steps: quasi-Newton (L-BFGS) orbital rotations (`Soscf`) and the exact
//! rotation C exp(K) of orbitals (`rotate`).

use faer::{Mat, Par};
use linus_base::linalg::{eigh, mm};

/// Quasi-Newton orbital optimizer (approximate second-order SCF): orbital rotations
/// kappa_ai from L-BFGS with the diagonal Hessian guess eps_a - eps_i (eps = diagonal of the
/// MO-basis Fock matrix), applied as C <- C exp(K). The step history lives in the moving MO
/// frame (as in ORCA's SOSCF), which is accurate near convergence where it is used.
pub(crate) struct Soscf {
    /// (step, gradient change, 1 / step.dy) pairs, newest last
    pub(crate) hist: Vec<(Vec<f64>, Vec<f64>, f64)>,
    pub(crate) last: Option<(Vec<f64>, Vec<f64>)>,
    mem: usize,
    max_step: f64,
    /// Keep a curvature pair when s.y > 1e-10 |s| |y| (relative) instead of s.y > 1e-14
    /// (absolute). Near a tight convergence threshold (max |F_ai| ~ 1e-9: steps ~ 1e-8) every
    /// pair falls below the absolute bound, L-BFGS degrades to the diagonal preconditioner and
    /// converges linearly along soft modes (triplet O3 B3LYP on the GPU, extreme preset: a
    /// factor 0.97 per iteration from 5e-9). The relative test runs in the max_iter escalation.
    relative: bool,
}

impl Soscf {
    pub(crate) fn new() -> Self {
        Soscf {
            hist: Vec::new(),
            last: None,
            mem: 10,
            max_step: 0.2,
            relative: false,
        }
    }

    /// [`Soscf::new`] with the relative curvature test (the `relative` field).
    pub(crate) fn with_relative_curvature() -> Self {
        Soscf {
            relative: true,
            ..Soscf::new()
        }
    }

    /// New orbitals for the Fock matrices `f` (per spin) built from the occupied orbitals of
    /// `c` (per spin, nbf x nmo, nocc[s] occupied).
    pub(crate) fn step(
        &mut self,
        c: &[Mat<f64>],
        f: &[&Mat<f64>],
        nocc: &[usize],
        par: Par,
    ) -> Vec<Mat<f64>> {
        // gradient and diagonal Hessian in the current MO frame
        let mut g = Vec::new();
        let mut h0 = Vec::new();
        for (s, cs) in c.iter().enumerate() {
            let fmo = mm(
                cs.transpose(),
                mm(f[s].as_ref(), cs.as_ref(), par).as_ref(),
                par,
            );
            let (no, nmo) = (nocc[s], cs.ncols());
            for i in 0..no {
                for a in no..nmo {
                    g.push(fmo[(a, i)]);
                    h0.push((fmo[(a, a)] - fmo[(i, i)]).max(0.05));
                }
            }
        }
        if let Some((s_prev, g_prev)) = self.last.take() {
            let y: Vec<f64> = g.iter().zip(&g_prev).map(|(a, b)| a - b).collect();
            let sy: f64 = s_prev.iter().zip(&y).map(|(a, b)| a * b).sum();
            let keep = if self.relative {
                let n2 = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
                sy > 1e-10 * n2(&s_prev) * n2(&y)
            } else {
                sy > 1e-14
            };
            if keep {
                if self.hist.len() == self.mem {
                    self.hist.remove(0);
                }
                self.hist.push((s_prev, y, 1.0 / sy));
            }
        }
        // L-BFGS two-loop recursion: r = H^-1 g
        let mut q = g.clone();
        let mut alpha = vec![0.0; self.hist.len()];
        for (k, (s, y, rho)) in self.hist.iter().enumerate().rev() {
            let a = rho * s.iter().zip(&q).map(|(u, v)| u * v).sum::<f64>();
            alpha[k] = a;
            q.iter_mut().zip(y).for_each(|(qi, yi)| *qi -= a * yi);
        }
        let mut r: Vec<f64> = q.iter().zip(&h0).map(|(qi, hi)| qi / hi).collect();
        for (k, (s, y, rho)) in self.hist.iter().enumerate() {
            let b = rho * y.iter().zip(&r).map(|(u, v)| u * v).sum::<f64>();
            r.iter_mut()
                .zip(s)
                .for_each(|(ri, si)| *ri += si * (alpha[k] - b));
        }
        let mut step: Vec<f64> = r.iter().map(|x| -x).collect();
        let smax = linus_base::linalg::max_abs_slice(&step);
        if smax > self.max_step {
            let sc = self.max_step / smax;
            step.iter_mut().for_each(|x| *x *= sc);
        }
        // rotate each spin's orbitals
        let mut out = Vec::new();
        let mut off = 0;
        for (s, cs) in c.iter().enumerate() {
            let (no, nmo) = (nocc[s], cs.ncols());
            let nv = nmo - no;
            let kappa = Mat::from_fn(nv, no, |a, i| step[off + i * nv + a]);
            off += no * nv;
            out.push(rotate(cs, &kappa, no, par));
        }
        self.last = Some((step, g));
        out
    }
}

/// C exp(K) for the antisymmetric K with virtual-occupied block kappa (nv x no):
/// with kappa^T kappa = V S^2 V^T,
///   C_o' = C_o V cos(S) V^T + C_v kappa V sinc(S) V^T
///   C_v' = C_v - C_o V sinc(S) V^T kappa^T + C_v kappa V ((cos(S) - 1) / S^2) V^T kappa^T.
pub(crate) fn rotate(c: &Mat<f64>, kappa: &Mat<f64>, no: usize, par: Par) -> Mat<f64> {
    let (nbf, nmo) = (c.nrows(), c.ncols());
    let co = c.subcols(0, no);
    let cv = c.subcols(no, nmo - no);
    let (cosm, sincm, cosm1) = rotation_functions(kappa, no, par);
    let cvk = mm(cv, kappa.as_ref(), par); // nbf x no
    let mut out = Mat::<f64>::zeros(nbf, nmo);
    let new_o = mm(co, cosm.as_ref(), par) + mm(cvk.as_ref(), sincm.as_ref(), par);
    let new_v = cv.to_owned() - mm(mm(co, sincm.as_ref(), par).as_ref(), kappa.transpose(), par)
        + mm(
            mm(cvk.as_ref(), cosm1.as_ref(), par).as_ref(),
            kappa.transpose(),
            par,
        );
    out.subcols_mut(0, no).copy_from(new_o.as_ref());
    out.subcols_mut(no, nmo - no).copy_from(new_v.as_ref());
    out
}

/// The functions of X = sqrt(kappa^T kappa) (no x no) that `rotate` applies: cos X,
/// sin X / X and (cos X - 1) / X^2.
pub fn rotation_functions(kappa: &Mat<f64>, no: usize, par: Par) -> (Mat<f64>, Mat<f64>, Mat<f64>) {
    let ktk = mm(kappa.transpose(), kappa.as_ref(), par);
    let (w, v) = eigh(ktk.as_ref(), Par::Seq);
    let fun = |f: &dyn Fn(f64) -> f64| -> Mat<f64> {
        let vf = Mat::from_fn(no, no, |i, k| v[(i, k)] * f(w[k].max(0.0).sqrt()));
        mm(vf.as_ref(), v.transpose(), par)
    };
    let cosm = fun(&|x| x.cos());
    let sincm = fun(&|x| {
        if x < 1e-8 {
            1.0 - x * x / 6.0
        } else {
            x.sin() / x
        }
    });
    let cosm1 = fun(&|x| {
        if x < 1e-4 {
            -0.5 + x * x / 24.0
        } else {
            (x.cos() - 1.0) / (x * x)
        }
    });
    (cosm, sincm, cosm1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_is_orthogonal_and_matches_first_order() {
        let (n, no) = (7, 3);
        let c = Mat::<f64>::identity(n, n);
        let kappa = Mat::from_fn(n - no, no, |a, i| 0.05 * ((a * 3 + i * 5) as f64).sin());
        let r = rotate(&c, &kappa, no, Par::Seq);
        let rtr = mm(r.transpose(), r.as_ref(), Par::Seq);
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((rtr[(i, j)] - want).abs() < 1e-14);
            }
        }
        // first order: C_v' ~ C_v - C_o kappa^T, C_o' ~ C_o + C_v kappa
        for a in 0..n - no {
            for i in 0..no {
                assert!((r[(no + a, i)] - kappa[(a, i)]).abs() < 1e-3);
                assert!((r[(i, no + a)] + kappa[(a, i)]).abs() < 1e-3);
            }
        }
    }
}
