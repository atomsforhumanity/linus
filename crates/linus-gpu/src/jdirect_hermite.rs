//! The memory-lean DF-J in Hermite space ([`crate::jdirect`]'s engine where its classes are
//! covered): the J-engine contractions of the shared Hermite tables (`crate::jengine`) without
//! any three-center integral block.
//!
//! A build of one density (its layout rows w): the fit of `HermiteDf::fit` (the pairs' Hermite
//! densities h, gamma from the auxiliary bras against them, c = V^-1 gamma, the fit's auxiliary
//! densities g); then per pair primitive its accumulators A_t (degree <= la + lb) against the
//! auxiliary kets g (`jengk`), and J's rows (mn|rho_fit) = sum_{prims of the pair} sum_t E^mn_t A_t,
//! one thread per row, the primitives and the record's nonzeros in a fixed order
//! (`dfj_pair_rows`).
//!
//! Nothing is screened. Every output element has one writing thread and every sum a fixed order
//! (ket chunks summed in order), so J is bitwise reproducible run to run.
//!
//! Coverage: orbital l <= 3 and auxiliary l <= 4 (the generated kernels'); `HermiteJ::new`
//! declines (None) beyond.

#[cfg(feature = "gpu")]
pub(crate) use device::*;

#[cfg(feature = "gpu")]
mod device {
    use crate::cpu::nherm;
    use crate::eri3c::{Eri3c, Eri3cHost};
    use crate::jengine::{HermiteDf, Passes};
    use crate::kernels::eri3c_class::Eri3cKind;
    use crate::kernels::jengine_gen::JENGK;
    use crate::linalg::Gemm;
    use crate::{Arg, DeviceBuf, Gpu, GpuError, Kernel, Launch};

    /// The Hermite-space DF-J engine over an eri3c builder (module docs).
    pub(crate) struct HermiteJ {
        core: HermiteDf,
        rows: Kernel,
    }

    impl HermiteJ {
        /// Whether the generated kernels cover every class of `host` (integral records, orbital
        /// pairs of degree <= MAX_PAIR, auxiliary l <= MAX_AUX).
        pub(crate) fn covers(host: &Eri3cHost) -> bool {
            host.kind == Eri3cKind::Integrals && HermiteDf::covers(host)
        }

        /// Sets up over the integral builder `ints` (every layout pair, absolute rows); None
        /// when a class is beyond the generated kernels.
        pub(crate) fn new(gpu: &Gpu, ints: &Eri3c) -> Result<Option<Self>, GpuError> {
            let Some(core) = HermiteDf::new(gpu, ints, Passes::Rows)? else {
                return Ok(None);
            };
            let src = crate::jengine::MODULE.source();
            let rows = gpu.kernel(crate::jengine::MODULE.name, &src, "dfj_pair_rows")?;
            Ok(Some(HermiteJ { core, rows }))
        }

        /// Device bytes the engine holds (pair and auxiliary tables, densities, accumulators,
        /// the chunk partials).
        pub(crate) fn device_bytes(&self) -> usize {
            self.core.device_bytes()
        }

        /// J's layout rows `jr` (nx, overwritten where the classes have rows) of the density
        /// rows `w` (nx), with the metric's Cholesky factor `chol` (naux x naux, lower).
        #[allow(
            clippy::too_many_arguments,
            reason = "the builder, the operands and the metric's solver"
        )]
        pub(crate) fn rows(
            &mut self,
            gpu: &Gpu,
            ints: &Eri3c,
            w: &DeviceBuf<f64>,
            chol: &DeviceBuf<f64>,
            gemm: &Gemm,
            naux: usize,
            jr: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            self.core.fit(gpu, ints, w, (chol, gemm, naux))?;
            let h = &self.core;
            let missing = || GpuError::Contract("Hermite J: set up without its rows".into());
            // the pair bras (degree l) against the fit's kets, auxiliary degrees in order
            for p in &h.pairs {
                let pr = p.rows.as_ref().ok_or_else(missing)?;
                for (i, a) in h.aux.iter().enumerate() {
                    h.accumulate(
                        gpu,
                        (&JENGK, p.l),
                        (p.nprim, &p.prims),
                        (a.l, a.nprim, &a.prims, &a.g, &a.kidx),
                        nherm(p.l),
                        &pr.a,
                        i == 0,
                        1.0,
                    )?;
                }
            }
            // J's rows
            for p in &h.pairs {
                let pr = p.rows.as_ref().ok_or_else(missing)?;
                let (rec, out, _) = ints
                    .class_records(p.ci)
                    .ok_or_else(|| GpuError::Contract("Hermite J: class records".into()))?;
                // SAFETY: matches dfj_pair_rows; npair nab threads, each reading its pair's
                // primitives' records and accumulators and writing its own row (rows < nx: the
                // builder's absolute layout rows).
                unsafe {
                    gpu.launch(
                        &self.rows,
                        Launch::linear(pr.npair * pr.nab, 128),
                        &[
                            Arg::I64(pr.npair as i64),
                            Arg::I32(pr.nab as i32),
                            Arg::I32(nherm(p.l) as i32),
                            Arg::I32(p.rs() as i32),
                            Arg::I32(p.n2 as i32),
                            Arg::buf(&pr.p0),
                            Arg::buf(rec),
                            Arg::buf(out),
                            Arg::buf(&pr.mp),
                            Arg::buf(&pr.me),
                            Arg::buf(&pr.mr),
                            Arg::buf(&pr.a),
                            Arg::buf_mut(jr),
                        ],
                    )
                }?;
            }
            Ok(())
        }
    }
}
