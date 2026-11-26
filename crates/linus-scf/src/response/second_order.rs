//! Second-order properties assembled from two perturbations: the response part of E^{ba} from
//! perturbation b's first-order matrices and perturbation a's response, and the reductions
//! the assemblies use.
//!
//! With C1^a = C_v U^a - 1/2 C_o s1oo^a the first-order occupied orbitals of a's response and
//! e1^a the first-order occupied orbital energies ([`FirstOrderOrbitals`]), the response part of
//! a second-order energy is, per spin channel with occupation f_occ,
//!
//!   E^{ba} <- f_occ [2 (F^b C_o) . C1^a - 2 (S^b C_o) . (C1^a e_o) - s1oo^b . e1^a]
//!             - 2 f_occ U^b . r^a,
//!
//! the S^b terms for a perturbation b that moves the basis, the last term the stationary
//! correction ([`super::stationary`]) when b's response exists too ([`SecondOrder`]). The
//! Hessian is the case b = a = the nuclear displacements (the formulation of PySCF's RHF
//! Hessian, with the explicit second derivatives at fixed density added by
//! `linus::derivatives::hessian`); the response part of the dipole derivatives is the case
//! b = the electric field, a = the nuclear displacements (`linus::properties::dipole`, D35). A
//! new property (NMR shieldings: rows the nuclear magnetic moments, columns a London orbital
//! magnetic field) is a new perturbation and this assembly.
//!
//! Every second-order property reduces its contractions of two sets of per-component matrices,
//! A_k . B_l = sum_ij (A_k)_ij (B_l)_ij for every pair (k, l), in one way: [`stacked_gram`], all
//! pairs at once as one GEMM of the stacked (vectorized) matrices. faer's GEMM gives the same
//! bits at any thread count, for the Hessian's 3N x 3N as for the dipole derivatives' 3 x 3N
//! and the polarizability's 3 x 3 (tested below at the size of a 39-atom molecule).

use super::perturbation::Perturbation;
use super::stationary::Correction;
use faer::{Mat, Par};
use linus_base::error::LinusError;
use linus_base::linalg::mm;

/// The first-order occupied orbitals and orbital energies of a perturbation's response,
/// `[s][k]` (spin channel s, component k):
/// - C1^k = C_vs U^k - 1/2 C_os s1oo^k (nbf x nocc_s);
/// - e1^k_ij = (C_os^T F^k C_os)_ij + (C_os^T dF[D^k] C_os)_ij - 1/2 (e_i + e_j) s1oo^k_ij
///   (nocc_s x nocc_s), with dF[D^k] C_os = dF[D^k_oo] C_os + dF[U^k] C_os the induced Fock
///   matrix of the whole first-order density;
///
/// the s1oo terms (and D^k_oo) for a perturbation that moves the basis.
pub struct FirstOrderOrbitals {
    pub c1: Vec<Vec<Mat<f64>>>,
    pub e1: Vec<Vec<Mat<f64>>>,
}

impl FirstOrderOrbitals {
    /// The orbitals of the response `u` (`[s][k]`, nvir_s x nocc_s) of `p`, with the induced
    /// Fock matrices of its occupied block's fixed response, dF[D^k_oo] C_os (`induced_oo`, for
    /// a perturbation that moves the basis:
    /// [`crate::response::problem::ResponseProblem::induced_oo`]) and of the response,
    /// dF[U^k] C_os (`induced_vo`).
    pub fn new(
        p: &dyn Perturbation,
        u: &[Vec<Mat<f64>>],
        induced_oo: Option<&[Vec<Mat<f64>>]>,
        induced_vo: &[Vec<Mat<f64>>],
        par: Par,
    ) -> Self {
        let reference = p.reference();
        let s1oo = p.overlap().map(|o| &o.s1oo);
        let c1 = reference
            .channels
            .iter()
            .enumerate()
            .map(|(s, b)| {
                (0..p.components())
                    .map(|k| {
                        let cu = mm(b.c_vir.as_ref(), u[s][k].as_ref(), par);
                        match s1oo {
                            Some(s1) => cu + mm(b.c_occ.as_ref(), s1[s][k].as_ref(), par) * -0.5,
                            None => cu,
                        }
                    })
                    .collect()
            })
            .collect();
        let fxo = p.fock_occ();
        let e1 = reference
            .channels
            .iter()
            .enumerate()
            .map(|(s, b)| {
                let (cot, no, eo) = (b.c_occ.transpose(), b.nocc(), &b.eps_occ);
                (0..p.components())
                    .map(|k| {
                        let h1 = mm(cot, fxo[s][k].as_ref(), par);
                        let v1 = match induced_oo {
                            Some(voo) => mm(cot, (&voo[s][k] + &induced_vo[s][k]).as_ref(), par),
                            None => mm(cot, induced_vo[s][k].as_ref(), par),
                        };
                        match s1oo {
                            Some(s1) => Mat::from_fn(no, no, |i, j| {
                                h1[(i, j)] + v1[(i, j)] - 0.5 * (eo[i] + eo[j]) * s1[s][k][(i, j)]
                            }),
                            None => Mat::from_fn(no, no, |i, j| h1[(i, j)] + v1[(i, j)]),
                        }
                    })
                    .collect()
            })
            .collect();
        FirstOrderOrbitals { c1, e1 }
    }
}

/// The response part of a second-order property E^{ba} (module docs): the first-order matrices
/// of the rows' perturbation b, the first-order orbitals of the columns' perturbation a, and the
/// stationary correction U^b . r^a when both responses exist.
pub struct SecondOrder<'a> {
    pub rows: &'a dyn Perturbation,
    pub columns: &'a FirstOrderOrbitals,
    pub correction: Option<Correction<'a>>,
    /// The stacked GEMMs elsewhere (a device's; None: the host's [`stacked_gram`]).
    pub gram: Option<&'a dyn Gram>,
}

impl SecondOrder<'_> {
    /// Adds channel `s`'s part, with the channel's occupation `f_occ`, to `out` (rows x
    /// columns), as one GEMM of the stacked matrices per term. An error is the device's
    /// ([`Gram`]); on the host it cannot fail.
    pub fn add_channel(
        &self,
        out: &mut Mat<f64>,
        s: usize,
        f_occ: f64,
        par: Par,
    ) -> Result<(), LinusError> {
        let c1 = &self.columns.c1[s];
        stacked_gram_on(
            self.gram,
            out,
            2.0 * f_occ,
            &refs(&self.rows.fock_occ()[s]),
            &refs(c1),
            par,
        )?;
        if let Some(o) = self.rows.overlap() {
            let b = self.rows.reference().channels.spin(s);
            let (n, eo) = (b.c_occ.nrows(), &b.eps_occ);
            let c1e: Vec<Mat<f64>> = c1
                .iter()
                .map(|c| Mat::from_fn(n, b.nocc(), |i, k| c[(i, k)] * eo[k]))
                .collect();
            stacked_gram_on(
                self.gram,
                out,
                -2.0 * f_occ,
                &refs(&o.sxo[s]),
                &refs(&c1e),
                par,
            )?;
            stacked_gram_on(
                self.gram,
                out,
                -f_occ,
                &refs(&o.s1oo[s]),
                &refs(&self.columns.e1[s]),
                par,
            )?;
        }
        if let Some(c) = &self.correction {
            c.add_gram(self.gram, out, s, -2.0 * f_occ, par)?;
        }
        Ok(())
    }
}

/// The GEMM of [`stacked_gram`] on other hardware (a device's): hr += f A^T B of the stacked
/// matrices, its sums in that hardware's order (the result the host's to round-off). Ok(false):
/// nothing added, the hardware declined (out of its memory, recorded by it) and the host's GEMM
/// runs; an error is the calculation's.
pub trait Gram: Sync {
    fn gram(
        &self,
        hr: &mut Mat<f64>,
        f: f64,
        a: &[&Mat<f64>],
        b: &[&Mat<f64>],
    ) -> Result<bool, LinusError>;
}

/// [`stacked_gram`] by `gram` where it takes it, else on the host.
pub fn stacked_gram_on(
    gram: Option<&dyn Gram>,
    hr: &mut Mat<f64>,
    f: f64,
    a: &[&Mat<f64>],
    b: &[&Mat<f64>],
    par: Par,
) -> Result<(), LinusError> {
    if let Some(g) = gram {
        if g.gram(hr, f, a, b)? {
            return Ok(());
        }
    }
    stacked_gram(hr, f, a, b, par);
    Ok(())
}

/// hr += f A^T B with A, B the matrices of `a` and `b` vectorized as columns (all of the same
/// shape): hr\[x\]\[y\] += f sum_ij a\[x\]_ij b\[y\]_ij.
pub fn stacked_gram(hr: &mut Mat<f64>, f: f64, a: &[&Mat<f64>], b: &[&Mat<f64>], par: Par) {
    let len = a.first().map_or(0, |m| m.nrows() * m.ncols());
    if len == 0 {
        return;
    }
    // column c: the columns of v[c] one after another (its column-major vectorization), copied
    // column by column into a matrix made as Mat::from_fn makes it
    let stack = |v: &[&Mat<f64>]| {
        let mut m = Mat::from_fn(len, v.len(), |_, _| 0.0);
        for (c, src) in v.iter().enumerate() {
            let nr = src.nrows();
            let dst = m.col_as_slice_mut(c);
            for j in 0..src.ncols() {
                dst[j * nr..(j + 1) * nr].copy_from_slice(src.col_as_slice(j));
            }
        }
        m
    };
    let (am, bm) = (stack(a), stack(b));
    linus_base::linalg::matmul(
        hr.as_mut(),
        faer::Accum::Add,
        am.transpose(),
        bm.as_ref(),
        f,
        par,
    );
}

/// The matrices of a list, by reference (the layout [`stacked_gram`] takes).
pub fn refs(v: &[Mat<f64>]) -> Vec<&Mat<f64>> {
    v.iter().collect()
}
#[cfg(test)]
mod tests {
    use super::*;

    /// The assembly GEMMs (stacked_gram) give the same bits sequentially and on 3 and 8
    /// threads, for every shape the second-order properties use at the size of a 39-atom
    /// molecule (u1an39: 3N = 117): the Hessian's 3N x 3N, the dipole derivatives' 3 x 3N and
    /// the polarizability's 3 x 3, each over matrices of a small and a large basis (the
    /// contraction length nvir nocc up to 180,000).
    #[test]
    fn stacked_gram_is_thread_count_independent() {
        let gen = |m: usize, (nr, nc): (usize, usize), s: usize| -> Vec<Mat<f64>> {
            (0..m)
                .map(|p| {
                    Mat::from_fn(nr, nc, |i, j| {
                        ((i * 31 + j * 17 + p * 7 + s) % 101) as f64 / 101.0 - 0.5
                    })
                })
                .collect()
        };
        let pools: Vec<(usize, rayon::ThreadPool)> = [3, 8]
            .map(|t| {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build();
                (t, pool.unwrap())
            })
            .into();
        for shape in [(480, 120), (1200, 150)] {
            for (na, nb) in [(117, 117), (3, 117), (3, 3)] {
                let (ma, mb) = (gen(na, shape, 0), gen(nb, shape, 5));
                let (ra, rb) = (refs(&ma), refs(&mb));
                let mut seq = Mat::<f64>::zeros(na, nb);
                stacked_gram(&mut seq, -4.0, &ra, &rb, Par::Seq);
                // the stacks as they were built element by element (the same bits)
                let stack = |v: &[&Mat<f64>]| {
                    let len = shape.0 * shape.1;
                    Mat::from_fn(len, v.len(), |i, c| {
                        v[c][(i % v[c].nrows(), i / v[c].nrows())]
                    })
                };
                let mut old = Mat::<f64>::zeros(na, nb);
                linus_base::linalg::matmul(
                    old.as_mut(),
                    faer::Accum::Add,
                    stack(&ra).transpose(),
                    stack(&rb).as_ref(),
                    -4.0,
                    Par::Seq,
                );
                for x in 0..na {
                    for y in 0..nb {
                        assert_eq!(seq[(x, y)].to_bits(), old[(x, y)].to_bits());
                    }
                }
                for (t, pool) in &pools {
                    let mut par = Mat::<f64>::zeros(na, nb);
                    pool.install(|| stacked_gram(&mut par, -4.0, &ra, &rb, Par::rayon(*t)));
                    for x in 0..na {
                        for y in 0..nb {
                            assert_eq!(
                                seq[(x, y)].to_bits(),
                                par[(x, y)].to_bits(),
                                "{na} x {nb} over {shape:?}, {t} threads: ({x}, {y})"
                            );
                        }
                    }
                }
            }
        }
    }
}
