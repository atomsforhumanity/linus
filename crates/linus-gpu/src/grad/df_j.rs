//! The density-fitted J gradient in Hermite space: [`DfJHermite`], `GpuDfGrad`'s path without
//! exchange.
//!
//! With the total density D, gamma_P = (P|mn) D_mn, c = V^-1 gamma and the fitted density
//! rho_fit = sum_Q c_Q Q, the gradient of E_J = c^T gamma - 1/2 c^T V c is
//!
//!   sum_mn D_mn (d_{A,B}(mn) | rho_fit) + sum_P c_P (d_C P | rho_D - rho_fit)
//!
//! (d_{A,B}: the pair's centres; d_C: the auxiliary function's centre). Gamma is the rank-1
//! c_P D_mn, so no integral block, derivative plane or Gamma is formed: everything is a
//! J-engine contraction of Hermite Gaussians on the tables `crate::jengine` shares with the
//! memory-lean DF-J (its first four steps, `HermiteDf::fit`, are the gamma, c and g below),
//! - per orbital pair primitive its Hermite density h_t = sum_mn E^mn_t D_mn (from the eri3c
//!   integral builder's records and the density per layout row);
//! - per auxiliary primitive of degree l (exponent a, contraction coefficient c) its density
//!   g_u = c (2a)^-l sum_m c_m S_m[u] over the Hermite indices u of degree exactly l (a solid
//!   harmonic Gaussian is (2a)^-l sum_u S[u] Lambda_u);
//! - gamma from the auxiliary bra (degree l) against the pair kets h;
//! - the pair centres from the pair bra (degree la + lb + 1) against the auxiliary kets g,
//!   contracted with the derivative builder's expansions d/dA and d/dB and D;
//! - the auxiliary centres from the auxiliary bra (degree l + 1: d_C Lambda_u = Lambda_{u+e_C})
//!   against the pair kets h (the three-center term) and against the auxiliary kets g with
//!   weight -1 (the metric term, reported apart).
//!
//! Nothing is screened (every primitive pair the records hold against every auxiliary
//! primitive). Sums run in a fixed order (the ket degrees and classes in order, chunks of kets
//! in order) and the atom sums in fixed point ([`crate::detsum`]), so the gradient is bitwise
//! reproducible run to run; its summation order differs from the Gamma-contracted kernels'.
//!
//! Coverage: orbital l <= 3 and auxiliary l <= 4 (the generated kernels'); [`DfJHermite::new`]
//! declines (None) beyond.

use crate::cpu::{ncart, nherm};
use crate::detsum::DetAcc;
use crate::eri3c::Eri3c;
use crate::error::GpuError;
use crate::jengine::{HermiteDf, Passes};
use crate::kernels::jengine_gen::{JENGB, JENGBK, JENGK};
use crate::linalg::Gemm;
use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};

/// A gradient per atom and its metric part.
type GradParts = (Vec<[f64; 3]>, Vec<[f64; 3]>);

/// The J-only DF gradient's Hermite engine (see the module docs), set up over the eri3c
/// builders of a `GpuDfGrad`.
pub(crate) struct DfJHermite {
    core: HermiteDf,
    pair_grad: Kernel,
    aux_grad: Kernel,
}

impl DfJHermite {
    /// Sets up over the integral builder `ints` and the derivative builder `ders` (the same
    /// pairs and auxiliary basis); None when a class is beyond the generated kernels.
    pub(crate) fn new(gpu: &Gpu, ints: &Eri3c, ders: &Eri3c) -> Result<Option<Self>, GpuError> {
        let Some(core) = HermiteDf::new(gpu, ints, Passes::Gradient(ders))? else {
            return Ok(None);
        };
        let src = crate::jengine::MODULE.source();
        let k = |e: &str| gpu.kernel(crate::jengine::MODULE.name, &src, e);
        Ok(Some(DfJHermite {
            core,
            pair_grad: k("dfj_pair_grad")?,
            aux_grad: k("dfj_aux_grad")?,
        }))
    }

    /// The gradient per atom (natom) and its metric part for the density rows `w` (the layout
    /// rows of `ints`), the metric's Cholesky factor `chol` (naux x naux, lower) and the
    /// builders this engine was set up over.
    #[allow(
        clippy::too_many_arguments,
        reason = "the two builders, the operands and the metric's solver"
    )]
    pub(crate) fn gradient(
        &mut self,
        gpu: &Gpu,
        ints: &Eri3c,
        ders: &Eri3c,
        w: &DeviceBuf<f64>,
        chol: &DeviceBuf<f64>,
        gemm: &Gemm,
        naux: usize,
        natom: usize,
    ) -> Result<GradParts, GpuError> {
        // 1-4. the pairs' Hermite densities, gamma, c = V^-1 gamma, the auxiliary densities
        self.core.fit(gpu, ints, w, (chol, gemm, naux))?;
        let h = &self.core;
        let missing = || GpuError::Contract("DF J gradient: set up without its passes".into());
        // 5. the pair bras (degree l + 1) against the fit's kets, auxiliary degrees in order
        for p in &h.pairs {
            let pg = p.grad.as_ref().ok_or_else(missing)?;
            for (i, a) in h.aux.iter().enumerate() {
                h.accumulate(
                    gpu,
                    (&JENGK, p.l + 1),
                    (p.nprim, &pg.dprims),
                    (a.l, a.nprim, &a.prims, &a.g, &a.kidx),
                    nherm(p.l + 1),
                    &pg.a1,
                    i == 0,
                    1.0,
                )?;
            }
        }
        // 6. the auxiliary bras (degree l + 1) against the pairs, and against the fit (-1)
        for a in &h.aux {
            let ag = a.grad.as_ref().ok_or_else(missing)?;
            let nc1 = ncart(a.l + 1);
            for (i, p) in h.pairs.iter().enumerate() {
                h.accumulate(
                    gpu,
                    (&JENGB, a.l + 1),
                    (a.nprim, &a.prims),
                    (p.l, p.nprim, &p.prims, &p.h, &p.kidx),
                    nc1,
                    &ag.a1,
                    i == 0,
                    1.0,
                )?;
            }
            for (i, a2) in h.aux.iter().enumerate() {
                h.accumulate(
                    gpu,
                    (&JENGBK, a.l + 1),
                    (a.nprim, &a.prims),
                    (a2.l, a2.nprim, &a2.prims, &a2.g, &a2.kidx),
                    nc1,
                    &ag.a1m,
                    i == 0,
                    -1.0,
                )?;
            }
        }
        // 7. the contractions per atom: the three-center term, then the metric term apart
        let natom3 = 3 * natom;
        let mut g3 = DetAcc::new(gpu, natom3)?;
        let mut gm = DetAcc::new(gpu, natom3)?;
        for p in &h.pairs {
            let pg = p.grad.as_ref().ok_or_else(missing)?;
            let (drec, out, atom) = ders
                .class_records(p.ci)
                .ok_or_else(|| GpuError::Contract("DF J gradient: class records".into()))?;
            // SAFETY: matches dfj_pair_grad; nprim derivative records of rsd values, their
            // pairs' rows and atoms (below natom), the density rows, nherm(l + 1) accumulators
            // per primitive.
            unsafe {
                gpu.launch(
                    &self.pair_grad,
                    Launch::linear(p.nprim, 128),
                    &[
                        Arg::I64(p.nprim as i64),
                        Arg::I32(pg.rsd as i32),
                        Arg::I32(nherm(p.l + 1) as i32),
                        Arg::I32(p.n2 as i32),
                        Arg::buf(drec),
                        Arg::buf(&p.prim_pair),
                        Arg::buf(out),
                        Arg::buf(atom),
                        Arg::buf(&pg.gs),
                        Arg::buf(&pg.gr),
                        Arg::buf(&pg.gm),
                        Arg::buf(w),
                        Arg::buf(&pg.a1),
                        g3.arg(),
                        Arg::I64(natom3 as i64),
                    ],
                )
            }?;
        }
        for a in &h.aux {
            let ag = a.grad.as_ref().ok_or_else(missing)?;
            for (acc, sum) in [(&ag.a1, &mut g3), (&ag.a1m, &mut gm)] {
                // SAFETY: matches dfj_aux_grad; nprim primitives with nc densities and nc1
                // accumulators, their shells' atoms (below natom).
                unsafe {
                    gpu.launch(
                        &self.aux_grad,
                        Launch::linear(a.nprim, 128),
                        &[
                            Arg::I64(a.nprim as i64),
                            Arg::I32(ncart(a.l) as i32),
                            Arg::I32(ncart(a.l + 1) as i32),
                            Arg::buf(&a.pshell),
                            Arg::buf(&ag.atom),
                            Arg::buf(&ag.up),
                            Arg::buf(&a.g),
                            Arg::buf(acc),
                            sum.arg(),
                            Arg::I64(natom3 as i64),
                        ],
                    )
                }?;
            }
        }
        let (h3, hm) = (g3.to_f64(gpu)?, gm.to_f64(gpu)?);
        let per_atom = |v: &[f64]| -> Vec<[f64; 3]> {
            (0..natom)
                .map(|a| [v[3 * a], v[3 * a + 1], v[3 * a + 2]])
                .collect()
        };
        let metric = per_atom(&hm);
        let mut total = per_atom(&h3);
        for (t, m) in total.iter_mut().zip(&metric) {
            for k in 0..3 {
                t[k] += m[k];
            }
        }
        Ok((total, metric))
    }
}
