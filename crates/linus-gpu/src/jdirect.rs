//! Density-fitted J without a stored three-index tensor: the memory-lean device J of large
//! molecules (where the DF engine's tensor, nbf^2/2 x naux, does not fit the device or the host),
//! for any method that needs J alone from density fitting (pure functionals, RI-J beside a K of
//! another kind).
//!
//! A build of nd densities on the DF layout's pairs and rows ([`crate::df::DfShared`]):
//! w = the densities' rows (D_mn + D_nm off the diagonal shell blocks; the DF engine's
//! `df_density_rows`), then per density gamma = (P|mn) w, c = V^-1 gamma (the metric's Cholesky
//! factor, two triangular solves) and J's rows (mn|P) c, then J (`df_rows_to_matrix`), by one of
//! two engines:
//!
//! - **Hermite** (where its classes are covered: orbital l <= 3, auxiliary l <= 4): the J-engine
//!   contractions of the pairs' Hermite densities and the fit's auxiliary densities, without any
//!   three-center integral block (`jdirect_hermite`);
//! - **eri3c** (otherwise): the integrals (mn|P) recomputed per batch of whole shell pairs
//!   (linus-gpu's three-center kernels, `Eri3c`), gamma as one GEMM per batch added in batch
//!   order and J's rows from a second pass; the device holds one batch (the workspace).
//!
//! The device also holds the factor (naux^2) and vectors of the rows. Deterministic run to run
//! (fixed batches and chunks, GEMMs and kernels with one writer per element). The caller picks
//! the engine (`JDirectEngine`; linus: the Hermite one on FP64-rich cards, where it is several
//! times faster, the eri3c one on FP64-poor cards).

#[cfg(feature = "gpu")]
pub use device::*;

#[cfg(feature = "gpu")]
mod device {
    use crate::blas::{GemmDims, MatMut, MatRef, Op};
    use crate::cpu::{BasisSet, DfThresholds, Operator};
    use crate::df::DfShared;
    use crate::df_deriv::{metric_factor, Metric, RowTables};
    use crate::engine::{Phases, Timeline};
    use crate::eri3c::Eri3c;
    use crate::linalg::Gemm;
    use crate::{Arg, DeviceBuf, Gpu, GpuError, Kernel, Launch};
    use linus_core::Arith;
    use std::sync::Arc;

    /// The device time of the builds so far by phase (seconds, from events), and the builds:
    /// the eri3c engine's two integral passes with their GEMMs (`gamma`, `rows`) and the fit;
    /// the Hermite engine's J-engine contractions with the fit (`gamma`) and J's matrix
    /// (`rows`).
    #[derive(Clone, Debug, Default)]
    pub struct JDirectTimes {
        pub gamma: f64,
        pub fit: f64,
        pub rows: f64,
        pub builds: usize,
    }

    impl Phases for JDirectTimes {
        fn phases(&self) -> impl IntoIterator<Item = (&'static str, f64)> {
            [
                ("gamma", self.gamma),
                ("fit", self.fit),
                ("rows", self.rows),
            ]
        }
    }

    /// The setup's phases (seconds) and the integral records' bytes on the device.
    #[derive(Clone, Debug, Default)]
    pub struct JDirectSetupTimes {
        /// The records and row tables in all (their upload and kernels included).
        pub records: f64,
        pub records_upload: f64,
        pub records_kernels: f64,
        /// The metric formed on the device (or uploaded from the CPU's), and its factorization.
        pub metric: f64,
        pub metric_potrf: f64,
        /// The Hermite engine's tables and kernels.
        pub hermite: f64,
        pub records_bytes: usize,
    }

    impl Phases for JDirectSetupTimes {
        fn phases(&self) -> impl IntoIterator<Item = (&'static str, f64)> {
            [
                ("records", self.records),
                ("records_upload", self.records_upload),
                ("records_kernels", self.records_kernels),
                ("metric", self.metric),
                ("metric_potrf", self.metric_potrf),
                ("hermite", self.hermite),
            ]
        }
    }

    /// Which engine builds J (module docs).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum JDirectEngine {
        /// The Hermite engine where its classes are covered, else eri3c.
        Hermite,
        /// The eri3c engine (recomputed integral blocks).
        Eri3c,
    }

    /// The memory-lean device DF-J (module docs).
    pub struct GpuDfJDirect {
        gpu: Arc<Gpu>,
        gemm: Gemm,
        shared: Arc<DfShared>,
        eri: Eri3c,
        rows: RowTables,
        chol: DeviceBuf<f64>,
        density_rows: Kernel,
        rows_to_matrix: Kernel,
        /// Batches of whole pairs: (first pair, end pair, first row, rows).
        batches: Vec<(usize, usize, usize, usize)>,
        ws: DeviceBuf<f64>,
        /// The Hermite engine (None: the eri3c engine's batches).
        hermite: Option<crate::jdirect_hermite::HermiteJ>,
        nbf: usize,
        naux: usize,
        nx: usize,
        times: JDirectTimes,
        setup: JDirectSetupTimes,
    }

    impl GpuDfJDirect {
        /// The engine for `basis` and `aux` (the DF layout and packed integrals of `shared` when
        /// given, else built here at the default thresholds), with a workspace of at most
        /// `workspace_bytes` for one batch of integrals (at least one shell pair's rows).
        pub fn new(
            gpu: Arc<Gpu>,
            basis: &BasisSet,
            aux: &BasisSet,
            shared: Option<Arc<DfShared>>,
            workspace_bytes: usize,
        ) -> Result<GpuDfJDirect, GpuError> {
            Self::with_engine(
                gpu,
                basis,
                aux,
                shared,
                workspace_bytes,
                JDirectEngine::Hermite,
            )
        }

        /// [`GpuDfJDirect::new`] with the engine `engine`.
        pub fn with_engine(
            gpu: Arc<Gpu>,
            basis: &BasisSet,
            aux: &BasisSet,
            shared: Option<Arc<DfShared>>,
            workspace_bytes: usize,
            engine: JDirectEngine,
        ) -> Result<GpuDfJDirect, GpuError> {
            let shared = match shared {
                Some(s) => s,
                None => Arc::new(DfShared::new(basis, aux, &DfThresholds::default())?),
            };
            let mut setup = JDirectSetupTimes::default();
            let t = std::time::Instant::now();
            let host = shared.ints.clone().map_err(GpuError::NotCovered)?;
            // the Hermite engine reads the records only: no integral kernels to load
            let hermite_covers =
                engine == JDirectEngine::Hermite && crate::jdirect_hermite::HermiteJ::covers(&host);
            let eri = if hermite_covers {
                Eri3c::records_only(&gpu, host, None)?
            } else {
                Eri3c::new(&gpu, host)?
            };
            setup.records_upload = eri.setup_upload;
            setup.records_kernels = eri.setup_kernels;
            setup.records_bytes = eri.setup_upload_bytes;
            let lay = shared.layout.clone();
            let rows = RowTables::upload(&gpu, &lay)?;
            setup.records = t.elapsed().as_secs_f64();
            // the metric formed on the device (the shared one, formed on the CPU, where the
            // device's three-center kernels do not cover the auxiliary basis), then factored
            let t = std::time::Instant::now();
            let (chol, form) = metric_factor(&gpu, Metric::Device(aux, &shared), lay.naux)?;
            setup.metric = form;
            setup.metric_potrf = t.elapsed().as_secs_f64() - form;
            let t = std::time::Instant::now();
            let hermite = if hermite_covers {
                crate::jdirect_hermite::HermiteJ::new(&gpu, &eri)?
            } else {
                None
            };
            setup.hermite = t.elapsed().as_secs_f64();
            let (batches, ws) = if hermite.is_some() {
                (Vec::new(), gpu.alloc_zeros::<f64>(1)?)
            } else {
                let max_rows = (workspace_bytes / (8 * lay.naux.max(1))).max(1);
                let batches = lay.row_batches(max_rows);
                let ws_rows = batches.iter().map(|b| b.3).max().unwrap_or(1);
                let ws = gpu.alloc_zeros::<f64>((ws_rows * lay.naux).max(1))?;
                (batches, ws)
            };
            let src = crate::df::MODULE.source();
            let k = |e: &str| gpu.kernel(crate::df::MODULE.name, &src, e);
            let mut gemm = Gemm::new(&gpu)?;
            gemm.prepare(&gpu, Arith::Fp64)?;
            Ok(GpuDfJDirect {
                density_rows: k("df_density_rows")?,
                rows_to_matrix: k("df_rows_to_matrix")?,
                nbf: lay.nbf,
                naux: lay.naux,
                nx: lay.nx,
                gemm,
                shared,
                eri,
                rows,
                chol,
                batches,
                ws,
                hermite,
                times: JDirectTimes::default(),
                setup,
                gpu,
            })
        }

        /// The layout and integrals (for other engines of the same problem).
        pub fn shared(&self) -> &Arc<DfShared> {
            &self.shared
        }

        /// Bytes the engine holds on the device: the factor, the workspace or the Hermite
        /// engine's tables, the row tables.
        pub fn device_bytes(&self) -> usize {
            8 * (self.naux * self.naux + self.ws.len())
                + 12 * self.nx
                + self.hermite.as_ref().map_or(0, |h| h.device_bytes())
        }

        /// The device time of the builds so far by phase.
        pub fn times(&self) -> &JDirectTimes {
            &self.times
        }

        /// The setup's phases and the records' bytes.
        pub fn setup_times(&self) -> &JDirectSetupTimes {
            &self.setup
        }

        /// Batches of integrals per pass (0 with the Hermite engine).
        pub fn nbatches(&self) -> usize {
            self.batches.len()
        }

        /// The engine: "hermite" or "eri3c" (module docs).
        pub fn engine(&self) -> &'static str {
            if self.hermite.is_some() {
                "hermite"
            } else {
                "eri3c"
            }
        }

        /// J of the `nd` densities at `dens` (nd x nbf^2, column-major, symmetric) into `j`
        /// (nd x nbf^2, overwritten).
        pub fn coulomb(
            &mut self,
            dens: &DeviceBuf<f64>,
            nd: usize,
            j: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            let gpu = self.gpu.clone();
            let (nbf, naux, nx) = (self.nbf, self.naux, self.nx);
            let n2 = nbf * nbf;
            if dens.len() < nd * n2 || j.len() < nd * n2 {
                return Err(GpuError::Contract(
                    "J-direct: density or J buffer too small".into(),
                ));
            }
            let fp64 = Arith::Fp64;
            let mut tl = Timeline::new(&gpu);
            let m0 = tl.mark()?;
            let mut w = gpu.alloc_zeros::<f64>((nx * nd).max(1))?;
            // SAFETY: matches df_density_rows; reads nx rows of the tables and nd nbf^2
            // densities, writes nd nx values.
            unsafe {
                gpu.launch(
                    &self.density_rows,
                    Launch::linear(nx, 256),
                    &[
                        Arg::I64(nx as i64),
                        Arg::I32(nd as i32),
                        Arg::I64(nbf as i64),
                        Arg::buf(&self.rows.m),
                        Arg::buf(&self.rows.n),
                        Arg::buf(&self.rows.kind),
                        Arg::buf(dens),
                        Arg::buf_mut(&mut w),
                    ],
                )
            }?;
            // the Hermite engine: per density, w's column through gamma, the fit and J's rows
            if let Some(h) = self.hermite.as_mut() {
                let mut jr = gpu.alloc_zeros::<f64>((nx * nd).max(1))?;
                let mut wd = gpu.alloc_zeros::<f64>(nx.max(1))?;
                let mut jd = gpu.alloc_zeros::<f64>(nx.max(1))?;
                for d in 0..nd {
                    gpu.copy_range(&w, d * nx, &mut wd, 0, nx)?;
                    h.rows(&gpu, &self.eri, &wd, &self.chol, &self.gemm, naux, &mut jd)?;
                    gpu.copy_range(&jd, 0, &mut jr, d * nx, nx)?;
                }
                let m1 = tl.close(|t: &mut JDirectTimes| &mut t.gamma, m0)?;
                self.to_matrix(&jr, nd, j)?;
                tl.close(|t: &mut JDirectTimes| &mut t.rows, m1)?;
                tl.add_to(&mut self.times)?;
                self.times.builds += 1;
                return Ok(());
            }
            // gamma = sum_batches (mn|P)^T w: naux x nd
            let mut gamma = gpu.alloc_zeros::<f64>((naux * nd).max(1))?;
            for (i, &(p0, p1, x0, nr)) in self.batches.iter().enumerate() {
                self.eri
                    .compute(&gpu, p0..p1, &Operator::Coulomb, &mut self.ws, 0)?;
                self.gemm.gemm(
                    fp64,
                    GemmDims::new(Op::N, Op::N, naux, nd, nr),
                    (1.0, if i == 0 { 0.0 } else { 1.0 }),
                    MatRef::new(&self.ws, 0, naux),
                    MatRef::new(&w, x0, nx),
                    MatMut::new(&mut gamma, 0, naux),
                )?;
            }
            let m1 = tl.close(|t: &mut JDirectTimes| &mut t.gamma, m0)?;
            // c = V^-1 gamma = L^-T L^-1 gamma
            self.gemm.trsm_lower_left(
                Op::N,
                (naux, nd),
                MatRef::new(&self.chol, 0, naux),
                MatMut::new(&mut gamma, 0, naux),
            )?;
            self.gemm.trsm_lower_left(
                Op::T,
                (naux, nd),
                MatRef::new(&self.chol, 0, naux),
                MatMut::new(&mut gamma, 0, naux),
            )?;
            let m2 = tl.close(|t: &mut JDirectTimes| &mut t.fit, m1)?;
            // J's rows = (mn|P) c per batch: nx x nd
            let mut jr = gpu.alloc_zeros::<f64>((nx * nd).max(1))?;
            for &(p0, p1, x0, nr) in &self.batches {
                self.eri
                    .compute(&gpu, p0..p1, &Operator::Coulomb, &mut self.ws, 0)?;
                self.gemm.gemm(
                    fp64,
                    GemmDims::new(Op::T, Op::N, nr, nd, naux),
                    (1.0, 0.0),
                    MatRef::new(&self.ws, 0, naux),
                    MatRef::new(&gamma, 0, naux),
                    MatMut::new(&mut jr, x0, nx),
                )?;
            }
            self.to_matrix(&jr, nd, j)?;
            tl.close(|t: &mut JDirectTimes| &mut t.rows, m2)?;
            tl.add_to(&mut self.times)?;
            self.times.builds += 1;
            Ok(())
        }

        /// J (nd x nbf^2, overwritten) of J's layout rows `jr` (nx x nd).
        fn to_matrix(
            &self,
            jr: &DeviceBuf<f64>,
            nd: usize,
            j: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            let gpu = &self.gpu;
            let (nbf, nx) = (self.nbf, self.nx);
            gpu.zero_range(j, 0, nd * nbf * nbf)?;
            // SAFETY: matches df_rows_to_matrix; reads nd nx row values, writes elements
            // (m, n), (n, m) < nbf^2 of nd matrices (j holds nd nbf^2, checked by coulomb).
            unsafe {
                gpu.launch(
                    &self.rows_to_matrix,
                    Launch::linear(nx, 256),
                    &[
                        Arg::I64(nx as i64),
                        Arg::I32(nd as i32),
                        Arg::I64(nbf as i64),
                        Arg::buf(&self.rows.m),
                        Arg::buf(&self.rows.n),
                        Arg::buf(&self.rows.kind),
                        Arg::buf(jr),
                        Arg::buf_mut(j),
                    ],
                )
            }
        }

        /// [`GpuDfJDirect::coulomb`] of host densities (nd x nbf^2, column-major), J on the host.
        pub fn coulomb_host(&mut self, dens: &[f64], nd: usize) -> Result<Vec<f64>, GpuError> {
            let n2 = self.nbf * self.nbf;
            let d = self.gpu.upload(dens)?;
            let mut j = self.gpu.alloc_zeros::<f64>((nd * n2).max(1))?;
            self.coulomb(&d, nd, &mut j)?;
            let mut v = self.gpu.download(&j)?;
            v.truncate(nd * n2);
            Ok(v)
        }
    }
}
