//! What the DF engines set up alike (feature `gpu`).
//!
//! The two-electron gradient (`crate::grad::df`) and the Hessian's two-electron terms
//! (`crate::hess::df`) both start from the basis sets' DF layout: the three-center integral
//! builder on its pairs, derivative builders (first, and for the Hessian second, derivatives),
//! the Coulomb metric's Cholesky factor L, and the row tables (each row's functions m, n and
//! its kind). With a [`DfShared`] that fits the basis sets and thresholds (an SCF setup's), the
//! packed data, derivative records and metric come from it, and their page-locked copies are
//! uploaded by DMA: the same builders, bit for bit, as built here. [`DerivSource`] is that one
//! setup; each engine adds its own plan and tables. The J/K engine (`crate::df::GpuDfJk`)
//! factors its metric ([`metric_factor`]) and keeps its row tables ([`RowTables`]) the same way.
//!
//! A call of each of these engines is a sequence of device phases on one stream, each a method
//! with the call's context (its inputs, sizes and timeline) and the device buffers earlier
//! phases left. The derivative engines take the spin channels' occupied orbitals side by side
//! on the device ([`upload_orbitals`] for host blocks). Every engine times its phases by
//! events on the stream (`engine::Timeline`), summed into the engine's times where the call waits.
//!
//! The derivative engines' common kernels are one module ([`MODULE`], `device/df/deriv.cu`,
//! launched through [`DerivKernels`]): a fitted block of rows unpacked into its dense
//! \[m - m0\]\[n\]\[P\] form, the gradient's way (diagonal shell blocks halved) or the Hessian's
//! (the transposed entries mirrored), and Gamma, s f (c_P D_mn - Zd\[m\]\[n\]\[P\]), with the
//! gradient's symmetrized density or the Hessian's symmetric one.

use crate::cpu::{metric, Aux3c, BasisSet, DfThresholds, Operator};
use crate::df::{DfLayout, DfShared};
use crate::eri3c::{Eri3c, Eri3cHost, Eri3cKind};
use crate::error::GpuError;
use crate::kernels::registry::KernelModule;
use crate::solver::Solver;
use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};
use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

const KERNELS: &str = include_str!("../device/df/deriv.cu");

/// The DF derivative engines' common kernels.
pub(crate) const MODULE: KernelModule = KernelModule::new(
    "dfderiv",
    || Cow::Borrowed(KERNELS),
    &["dd_unpack", "dd_gamma"],
);

/// Where a DF derivative engine's builders and metric come from: the basis sets and their
/// layout at the thresholds, and a fitting [`DfShared`] when given.
pub(crate) struct DerivSource<'a> {
    pub(crate) basis: &'a BasisSet,
    pub(crate) aux: &'a BasisSet,
    pub(crate) lay: &'a DfLayout,
    /// Already filtered: it fits the basis sets and thresholds ([`DfShared::fits`]).
    pub(crate) shared: Option<&'a DfShared>,
    pub(crate) thresholds: &'a DfThresholds,
}

impl DerivSource<'_> {
    /// The integral builder of the layout's pairs (the shared packed data, or packed here, at
    /// the primitive threshold) and the seconds of the packing (CPU).
    pub(crate) fn integrals(&self, gpu: &Gpu) -> Result<(Eri3c, f64), GpuError> {
        let (mut b, pack) = self.integral_records(gpu)?;
        b.ensure_kernels(gpu)?;
        Ok((b, pack))
    }

    /// As [`DerivSource::integrals`] without the kernels ([`Eri3c::records_only`]).
    pub(crate) fn integral_records(&self, gpu: &Gpu) -> Result<(Eri3c, f64), GpuError> {
        let t = Instant::now();
        let prim = self.thresholds.prim;
        let host: Arc<Eri3cHost> = match self.shared.map(|s| &s.ints) {
            Some(Ok(h)) => h.clone(),
            // the packing's reason: the error packing them here gives
            Some(Err(why)) => return Err(GpuError::Contract(why.clone())),
            None => {
                let lay = self.lay;
                Eri3cHost::from_pair3c(self.basis, self.aux, &lay.pairs, &lay.x0, prim)?.into()
            }
        };
        let pack = t.elapsed().as_secs_f64();
        let staged = self
            .shared
            .and_then(|s| s.staged_records(Eri3cKind::Integrals, prim));
        Ok((Eri3c::records_only(gpu, host, staged.as_deref())?, pack))
    }

    /// The `kind` derivative builder of the layout's pairs (the shared records, built there
    /// once, or built here, at the derivative primitive threshold) and the seconds of its
    /// records (CPU).
    pub(crate) fn derivatives(&self, gpu: &Gpu, kind: Eri3cKind) -> Result<(Eri3c, f64), GpuError> {
        let (mut b, pack) = self.derivative_records(gpu, kind)?;
        b.ensure_kernels(gpu)?;
        Ok((b, pack))
    }

    /// As [`DerivSource::derivatives`] without the kernels ([`Eri3c::records_only`]).
    pub(crate) fn derivative_records(
        &self,
        gpu: &Gpu,
        kind: Eri3cKind,
    ) -> Result<(Eri3c, f64), GpuError> {
        let t = Instant::now();
        let thresh = self.thresholds.deriv_prim;
        let host: Arc<Eri3cHost> = match self.shared {
            Some(s) => s.derivatives(self.basis, self.aux, kind, thresh)?,
            None => {
                let rows = self.lay.pair_rows();
                Eri3cHost::new(self.basis, self.aux, &rows, kind, thresh)?.into()
            }
        };
        let pack = t.elapsed().as_secs_f64();
        let staged = self.shared.and_then(|s| s.staged_records(kind, thresh));
        Ok((Eri3c::records_only(gpu, host, staged.as_deref())?, pack))
    }

    /// The Coulomb metric (P|Q) on the device (the shared one, from its page-locked copy when
    /// staged, or formed here on the CPU), factored in place into its lower Cholesky factor L,
    /// and the seconds of forming and uploading it ([`metric_factor`]).
    pub(crate) fn metric_factor(&self, gpu: &Gpu) -> Result<(DeviceBuf<f64>, f64), GpuError> {
        let aux3;
        let source = match self.shared {
            Some(s) => Metric::Staged(s),
            None => {
                aux3 = Aux3c::new(self.aux);
                Metric::Formed(&aux3)
            }
        };
        metric_factor(gpu, source, self.lay.naux)
    }
}

/// Where a DF engine's Coulomb metric (P|Q) (naux x naux) comes from.
pub(crate) enum Metric<'a> {
    /// A [`DfShared`]'s, from its page-locked copy (DMA) when staged
    /// ([`DfShared::upload_metric`]).
    Staged(&'a DfShared),
    /// Formed here on the CPU from the auxiliary basis.
    Formed(&'a Aux3c),
    /// Formed on the device from the auxiliary basis by the three-center kernels
    /// ([`device_metric`]); the [`DfShared`]'s where the kernels do not cover the basis.
    Device(&'a BasisSet, &'a DfShared),
}

/// The Coulomb metric (P|Q) of `aux` (naux x naux) on the device by the three-center kernels,
/// each auxiliary shell paired with a unit s function at its centre (the CPU's `metric` the
/// same way); None where the kernels do not cover the auxiliary basis (l > 4 or Cartesian d and
/// beyond).
fn device_metric(
    gpu: &Gpu,
    aux: &BasisSet,
    naux: usize,
) -> Result<Option<DeviceBuf<f64>>, GpuError> {
    use crate::cpu::{build_pair, Pair3c, Shell, ShellView};
    if aux.nbf != naux || aux.shells.iter().any(|s| s.l > 4 || (s.l >= 2 && !s.pure)) {
        return Ok(None);
    }
    // the auxiliary shells and one unit s function, as the pairs' basis
    let mut pb = aux.clone();
    let unit = pb.shells.len();
    pb.shells.push(Shell {
        l: 0,
        pure: true,
        atom: 0,
        center: [0.0; 3],
        exps: vec![0.0],
        coefs: vec![1.0],
        ao_start: aux.nbf,
    });
    pb.nbf += 1;
    let pairs: Vec<Pair3c> = aux
        .shells
        .iter()
        .enumerate()
        .map(|(i, s)| {
            Pair3c::from_shell_pair(&build_pair(
                &ShellView::from_shell(s),
                &ShellView::unit(s.center),
                i,
                unit,
            ))
        })
        .collect();
    let rows: Vec<usize> = aux.shells.iter().map(|s| s.ao_start).collect();
    let host = match Eri3cHost::from_pair3c(&pb, aux, &pairs, &rows, 0.0) {
        Ok(h) => h,
        Err(GpuError::Contract(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let eri = Eri3c::new(gpu, host)?;
    let mut v = gpu.alloc_zeros::<f64>((naux * naux).max(1))?;
    eri.compute(gpu, 0..pairs.len(), &Operator::Coulomb, &mut v, 0)?;
    Ok(Some(v))
}

/// The Coulomb metric of `source` on the device, factored in place into its lower Cholesky
/// factor L (cuSOLVER), and the seconds of forming (or uploading) it.
pub(crate) fn metric_factor(
    gpu: &Gpu,
    source: Metric<'_>,
    naux: usize,
) -> Result<(DeviceBuf<f64>, f64), GpuError> {
    let t = Instant::now();
    let mut chol = match source {
        Metric::Staged(s) => s.upload_metric(gpu)?,
        Metric::Formed(aux3) => gpu.upload(&metric(aux3, 0.0, &Operator::Coulomb))?,
        Metric::Device(aux, s) => match device_metric(gpu, aux, naux)? {
            Some(v) => v,
            None => s.upload_metric(gpu)?,
        },
    };
    let host = t.elapsed().as_secs_f64();
    Solver::new(gpu)?.potrf_lower(gpu, &mut chol, naux)?;
    Ok((chol, host))
}

/// The layout's row tables on the device: each row's functions m and n and its kind.
pub(crate) struct RowTables {
    pub(crate) m: DeviceBuf<i32>,
    pub(crate) n: DeviceBuf<i32>,
    pub(crate) kind: DeviceBuf<i32>,
}

impl RowTables {
    pub(crate) fn upload(gpu: &Gpu, lay: &DfLayout) -> Result<RowTables, GpuError> {
        Ok(RowTables {
            m: gpu.upload(&lay.row_m)?,
            n: gpu.upload(&lay.row_n)?,
            kind: gpu.upload(&lay.row_kind)?,
        })
    }
}

/// The occupied orbitals of the channels given on the host, per channel C (`nbf` x ncol,
/// column-major), side by side on the device (`nbf` x sum(ncols); one zero for none), and each
/// channel's columns. `err` makes the engine's error for a block of the wrong size.
pub(crate) fn upload_orbitals(
    gpu: &Gpu,
    nbf: usize,
    occ: &[(&[f64], usize)],
    err: fn(&str) -> GpuError,
) -> Result<(DeviceBuf<f64>, Vec<usize>), GpuError> {
    let mut flat = Vec::new();
    let mut ncols = Vec::new();
    for &(c, n) in occ {
        if c.len() != nbf * n {
            return Err(err("an orbital block has the wrong size"));
        }
        flat.extend_from_slice(c);
        ncols.push(n);
    }
    let cmo = gpu.upload(if flat.is_empty() { &[0.0] } else { &flat })?;
    Ok((cmo, ncols))
}

/// A sub-block of the layout's rows: rows `x0..x0 + nrows` of the row tables, their functions m
/// in `m0..m1` and n below `m1`, each with `naux` auxiliary functions.
#[derive(Clone, Copy)]
pub(crate) struct RowBlock<'a> {
    pub(crate) rows: &'a RowTables,
    pub(crate) x0: usize,
    pub(crate) nrows: usize,
    pub(crate) naux: usize,
    pub(crate) m0: usize,
    pub(crate) m1: usize,
}

/// How [`DerivKernels::unpack`] writes a dense block.
#[derive(Clone, Copy)]
pub(crate) enum Unpack {
    /// The gradient's: diagonal shell blocks (both triangles stored there) halved.
    Halved,
    /// The Hessian's MO pass: an off-diagonal shell pair with n inside the block also sets the
    /// transposed entry.
    Mirrored,
}

/// The Coulomb term c_P D_mn of [`DerivKernels::gamma`].
#[derive(Clone, Copy)]
pub(crate) struct GammaJ<'a> {
    /// c (naux).
    pub(crate) cfit: &'a DeviceBuf<f64>,
    /// D (nbf x nbf).
    pub(crate) dens: &'a DeviceBuf<f64>,
    pub(crate) nbf: usize,
    /// D_mn = (D\[n\]\[m\] + D\[m\]\[n\]) / 2 (the gradient's), else D\[n\]\[m\] (a symmetric
    /// D, the Hessian's).
    pub(crate) symmetrize: bool,
}

/// The terms of [`DerivKernels::gamma`], s f (c_P D_mn - Zd\[m\]\[n\]\[P\]): each when given.
#[derive(Clone, Copy)]
pub(crate) struct GammaTerms<'a> {
    pub(crate) s: f64,
    pub(crate) j: Option<GammaJ<'a>>,
    /// Zd, the block's \[m - m0\]\[n\]\[P\] (n < m1).
    pub(crate) zd: Option<&'a DeviceBuf<f64>>,
}

/// The derivative engines' common kernels ([`MODULE`]).
pub(crate) struct DerivKernels {
    unpack: Kernel,
    gamma: Kernel,
}

impl DerivKernels {
    pub(crate) fn new(gpu: &Gpu) -> Result<DerivKernels, GpuError> {
        let src = MODULE.source();
        Ok(DerivKernels {
            unpack: gpu.kernel(MODULE.name, &src, "dd_unpack")?,
            gamma: gpu.kernel(MODULE.name, &src, "dd_gamma")?,
        })
    }

    /// The block's fitted rows `a` (nrows x naux) into the dense block `ad`, the `form` way:
    /// ad\[((m - m0) m1 + n) naux + P\] = h a\[(x - x0) naux + P\].
    ///
    /// # Safety
    /// The row tables hold the block's rows, with m in \[m0, m1) and n < m1 for each; `a` holds
    /// nrows naux values and `ad` (m1 - m0) m1 naux.
    pub(crate) unsafe fn unpack(
        &self,
        gpu: &Gpu,
        blk: RowBlock<'_>,
        form: Unpack,
        a: &DeviceBuf<f64>,
        ad: &mut DeviceBuf<f64>,
    ) -> Result<(), GpuError> {
        let n = blk.nrows * blk.naux;
        if n == 0 {
            return Ok(());
        }
        let (half, mirror) = match form {
            Unpack::Halved => (1, 0),
            Unpack::Mirrored => (0, 1),
        };
        // SAFETY: forwarded to the caller; matches dd_unpack.
        unsafe {
            gpu.launch(
                &self.unpack,
                Launch::linear(n, 256),
                &[
                    Arg::I64(blk.nrows as i64),
                    Arg::I64(blk.x0 as i64),
                    Arg::I32(blk.naux as i32),
                    Arg::I64(blk.m0 as i64),
                    Arg::I64(blk.m1 as i64),
                    Arg::I32(half),
                    Arg::I32(mirror),
                    Arg::buf(&blk.rows.m),
                    Arg::buf(&blk.rows.n),
                    Arg::buf(&blk.rows.kind),
                    Arg::buf(a),
                    Arg::buf_mut(ad),
                ],
            )
        }
    }

    /// Gamma of the block's rows: g\[(x - x0) naux + P\] = s f (c_P D_mn - Zd\[m\]\[n\]\[P\]),
    /// f = 2 off the diagonal shell blocks.
    ///
    /// # Safety
    /// The row tables hold the block's rows, with m in \[m0, m1) and n < m1 (both below nbf)
    /// for each; `g` holds nrows naux values, c naux, D nbf^2 and Zd (m1 - m0) m1 naux.
    pub(crate) unsafe fn gamma(
        &self,
        gpu: &Gpu,
        blk: RowBlock<'_>,
        t: GammaTerms<'_>,
        g: &mut DeviceBuf<f64>,
    ) -> Result<(), GpuError> {
        let n = blk.nrows * blk.naux;
        if n == 0 {
            return Ok(());
        }
        let ptr = |b: Option<&DeviceBuf<f64>>| b.map_or(Arg::Ptr(0), Arg::buf);
        // SAFETY: forwarded to the caller; matches dd_gamma (the null pointers of absent terms
        // are not read).
        unsafe {
            gpu.launch(
                &self.gamma,
                Launch::linear(n, 256),
                &[
                    Arg::I64(blk.nrows as i64),
                    Arg::I64(blk.x0 as i64),
                    Arg::I32(blk.naux as i32),
                    Arg::F64(t.s),
                    Arg::I32(t.j.is_some_and(|j| j.symmetrize) as i32),
                    Arg::buf(&blk.rows.m),
                    Arg::buf(&blk.rows.n),
                    Arg::buf(&blk.rows.kind),
                    Arg::I64(t.j.map_or(0, |j| j.nbf) as i64),
                    ptr(t.j.map(|j| j.dens)),
                    ptr(t.j.map(|j| j.cfit)),
                    Arg::I32(t.j.is_some() as i32),
                    ptr(t.zd),
                    Arg::I32(t.zd.is_some() as i32),
                    Arg::I64(blk.m0 as i64),
                    Arg::I64(blk.m1 as i64),
                    Arg::buf_mut(g),
                ],
            )
        }
    }
}
