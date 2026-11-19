//! Small helpers the integral engines share: the two-sided transformation of a shell pair's
//! Cartesian blocks to the shells' functions, the shell-pair maxima of a density, and per-atom
//! gradient sums.

use crate::basis::{ncart, BasisSet, Shell};
use crate::core::hermite::ShellView;

/// The transformations of a shell pair from Cartesian components to the shells' functions
/// (`ta`: na x nca, `tb`: nb x ncb, row-major; [`BasisSet::shell_transform`]).
pub(crate) struct PairTransform<'a> {
    ta: &'a [f64],
    tb: &'a [f64],
    na: usize,
    nb: usize,
    nca: usize,
    ncb: usize,
}

impl PairTransform<'static> {
    /// The cached transformations of the shells `a` and `b` ([`BasisSet::shell_transform_ref`]).
    pub(crate) fn of_shells(a: &Shell, b: &Shell) -> Self {
        PairTransform {
            ta: BasisSet::shell_transform_ref(a),
            tb: BasisSet::shell_transform_ref(b),
            na: a.nfunc(),
            nb: b.nfunc(),
            nca: ncart(a.l),
            ncb: ncart(b.l),
        }
    }
}

impl<'a> PairTransform<'a> {
    /// The transformations the shell views carry.
    pub(crate) fn of_views(a: &'a ShellView, b: &'a ShellView) -> Self {
        PairTransform {
            ta: &a.transform,
            tb: &b.transform,
            na: a.nfunc,
            nb: b.nfunc,
            nca: ncart(a.l),
            ncb: ncart(b.l),
        }
    }

    /// `nrows` Cartesian blocks (nca x ncb each) to function blocks (na x nb each):
    /// dst[row][mu][nu] = sum Ta[mu][ia] src[row][ia][ib] Tb[nu][ib], transforming the a side
    /// first (zero elements of Ta skipped). `tmp` is scratch.
    pub(crate) fn rows(&self, src: &[f64], nrows: usize, dst: &mut [f64], tmp: &mut Vec<f64>) {
        let (na, nb, nca, ncb) = (self.na, self.nb, self.nca, self.ncb);
        let (ta, tb) = (self.ta, self.tb);
        tmp.clear();
        tmp.resize(na * ncb, 0.0);
        for row in 0..nrows {
            let s = &src[row * nca * ncb..(row + 1) * nca * ncb];
            tmp.fill(0.0);
            for mu in 0..na {
                for ia in 0..nca {
                    let t = ta[mu * nca + ia];
                    if t == 0.0 {
                        continue;
                    }
                    for ib in 0..ncb {
                        tmp[mu * ncb + ib] += t * s[ia * ncb + ib];
                    }
                }
            }
            let d = &mut dst[row * na * nb..(row + 1) * na * nb];
            for mu in 0..na {
                for nu in 0..nb {
                    let mut v = 0.0;
                    for ib in 0..ncb {
                        v += tb[nu * ncb + ib] * tmp[mu * ncb + ib];
                    }
                    d[mu * nb + nu] = v;
                }
            }
        }
    }
}

/// Max |D| over each pair of shells (ns x ns) of the row-major nbf x nbf density `d`.
pub(crate) fn shell_pair_dmax(basis: &BasisSet, d: &[f64]) -> Vec<f64> {
    let sh = &basis.shells;
    let ns = sh.len();
    let nbf = basis.nbf;
    let mut m = vec![0.0f64; ns * ns];
    for a in 0..ns {
        for b in 0..ns {
            let mut x: f64 = 0.0;
            for i in sh[a].ao_start..sh[a].ao_start + sh[a].nfunc() {
                for j in sh[b].ao_start..sh[b].ao_start + sh[b].nfunc() {
                    x = x.max(d[i * nbf + j].abs());
                }
            }
            m[a * ns + b] = x;
        }
    }
    m
}

/// a += b, atom by atom.
pub(crate) fn add_grads_into(a: &mut [[f64; 3]], b: &[[f64; 3]]) {
    for (x, y) in a.iter_mut().zip(b) {
        for k in 0..3 {
            x[k] += y[k];
        }
    }
}
