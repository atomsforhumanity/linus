//! The vector service: BLAS level-1 operations over the first `n` elements of device buffers
//! (`device/blas1.cu`), one definition for every engine.
//!
//! Each operation's rounding is part of its definition, because the engines' results depend on
//! it bit for bit: [`Blas1::axpby`] sums its terms left to right, the first product rounded and
//! each later term fused into the sum (one fma), so a caller fixes its rounding by the order of
//! its terms (`y = a y + b x` rounded as `fma(a, y, b x)` is the terms `[(b, x), (a, y)]`). An
//! operand may be the output itself ([`Operand::Out`], in place). Every launch is one thread per
//! element in blocks of 256.

use crate::kernels::registry::KernelModule;
use crate::{Arg, DeviceBuf, Gpu, GpuError, Kernel, Launch};
use std::borrow::Cow;

const KERNELS: &str = include_str!("../../device/blas1.cu");

/// The vector kernels' module.
pub(crate) const MODULE: KernelModule = KernelModule::new(
    "blas1",
    || Cow::Borrowed(KERNELS),
    &["blas1_scale", "blas1_axpby"],
);

/// Threads per block of every launch.
const THREADS: u32 = 256;

/// A vector operand: a buffer's first `n` elements, or the output's own (the operation in
/// place).
#[derive(Clone, Copy)]
pub enum Operand<'a> {
    Buf(&'a DeviceBuf<f64>),
    Out,
}

/// The factor of [`Blas1::scale`]: one scalar, or one per element (a diagonal).
#[derive(Clone, Copy)]
pub enum Scale<'a> {
    Scalar(f64),
    Diagonal(&'a DeviceBuf<f64>),
}

/// The vector kernels of an engine.
pub struct Blas1 {
    scale: Kernel,
    axpby: Kernel,
}

impl Blas1 {
    /// The service (its module compiled once per process, [`Gpu::kernel`]).
    pub fn new(gpu: &Gpu) -> Result<Blas1, GpuError> {
        let src = MODULE.source();
        Ok(Blas1 {
            scale: gpu.kernel(MODULE.name, &src, "blas1_scale")?,
            axpby: gpu.kernel(MODULE.name, &src, "blas1_axpby")?,
        })
    }

    /// `out = s x` over the first `n` elements (one rounded product each).
    pub fn scale(
        &self,
        gpu: &Gpu,
        n: usize,
        s: Scale<'_>,
        x: Operand<'_>,
        out: &mut DeviceBuf<f64>,
    ) -> Result<(), GpuError> {
        let (a, w) = match s {
            Scale::Scalar(a) => (a, Arg::Ptr(0)),
            Scale::Diagonal(w) => (0.0, operand(Operand::Buf(w), n, out)?),
        };
        let x = operand(x, n, out)?;
        if n == 0 {
            return Ok(());
        }
        // SAFETY: matches blas1_scale; n elements of w, x and out (checked by `operand`).
        unsafe {
            gpu.launch(
                &self.scale,
                Launch::linear(n, THREADS),
                &[Arg::I64(n as i64), Arg::F64(a), w, x, Arg::buf_mut(out)],
            )
        }
    }

    /// `out = c_1 v_1 + c_2 v_2 + c_3 v_3` over the first `n` elements for one to three terms
    /// `(c, v)`, summed left to right: `c_1 v_1` rounded, then each later term fused into the
    /// sum (`fma(c, v, sum)`). The order of the terms fixes the rounding.
    pub fn axpby(
        &self,
        gpu: &Gpu,
        n: usize,
        terms: &[(f64, Operand<'_>)],
        out: &mut DeviceBuf<f64>,
    ) -> Result<(), GpuError> {
        if terms.is_empty() || terms.len() > 3 {
            return Err(GpuError::Contract(format!(
                "axpby of {} terms (one to three)",
                terms.len()
            )));
        }
        let mut args = [(0.0, Arg::Ptr(0)); 3];
        for (slot, &(c, v)) in args.iter_mut().zip(terms) {
            *slot = (c, operand(v, n, out)?);
        }
        if n == 0 {
            return Ok(());
        }
        let [(a, x), (b, y), (c, z)] = args;
        // SAFETY: matches blas1_axpby; n elements of every operand given and of out (checked by
        // `operand`); a null y or z is not read.
        unsafe {
            gpu.launch(
                &self.axpby,
                Launch::linear(n, THREADS),
                &[
                    Arg::I64(n as i64),
                    Arg::F64(a),
                    x,
                    Arg::F64(b),
                    y,
                    Arg::F64(c),
                    z,
                    Arg::buf_mut(out),
                ],
            )
        }
    }
}

/// The address of `v` (the output's for [`Operand::Out`]), once `v` and the output both hold
/// `n` elements.
fn operand(v: Operand<'_>, n: usize, out: &DeviceBuf<f64>) -> Result<Arg, GpuError> {
    let b = match v {
        Operand::Buf(b) => b,
        Operand::Out => out,
    };
    if n > b.len() || n > out.len() {
        return Err(GpuError::Contract(format!(
            "vector operation over {n} elements of buffers of {} and {}",
            b.len(),
            out.len()
        )));
    }
    Ok(Arg::buf(b))
}
