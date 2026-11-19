//! FP64 GEMMs emulated on INT8 tensor cores: Ozaki scheme I with integer slices (feature
//! `gpu`). The vendor part, an INT8 GEMM with exact INT32 accumulation, is
//! `Blas::gemm_i8` (src/cuda/ozaki.rs); everything here is portable: the slicing and
//! recombination kernels are in the common CUDA/HIP subset.
//!
//! The scheme (Ozaki et al. 2012; on integer units Ootomo et al. 2024, with the fixed-shift
//! round-to-nearest splitting and the group-wise error-free accumulation of Uchino et al. 2025):
//! - A *vector* is a row of op(A) or a column of op(B): the values multiplied along the inner
//!   dimension k. Each vector r gets one power-of-two scale 2^e_r with max |x| <= 127/128 2^e_r,
//!   and its values are rounded to fixed point, x 2^-e_r ~= I 2^-(8s-1) with |I| < 2^(8s-1),
//!   then cut into s signed 8-bit digits, I = sum_p d_p 2^(8(s-1-p)) (two's-complement bytes
//!   with borrows, so |d_0| <= 127 and |d_p| <= 128): x ~= 2^e_r sum_p d_p 2^(-7-8p), 8s - 1
//!   bits per value relative to its vector's largest. The digits and the exponents are a
//!   `Sliced` operand; a stored operand is sliced once and reused.
//! - A B ~= 2^(e_r + f_c) sum_g 2^(-14-8g) S_g, S_g = sum_{p+q=g} D_p E_q^T, for the
//!   diagonals g < s (the s(s+1)/2 products of the "fast mode"; the dropped ones are below the
//!   rounding of the operands). Each S_g is exact in INT32 as long as at most 2^17 digit
//!   products are summed per element; within that bound the products of one diagonal run as
//!   one INT8 GEMM over the slices concatenated along k (the left operand's digits ascending,
//!   the right one's descending, `Order`), or as GEMMs accumulated in INT32.
//! - One recombination pass converts the diagonals to FP64 in a fixed order (smallest
//!   first) and applies the scales: C = alpha A B (+ beta C).
//!
//! Properties:
//! - Deterministic: integer sums are exact and the FP64 sums run in a fixed order, so results
//!   are bitwise reproducible (and independent of the GEMM algorithm cuBLAS picks).
//! - Smooth in the inputs at the rounding level: the slice count is fixed (no data-dependent
//!   choice as in cuBLAS's dynamic mantissa or OzBLAS's estimate), so a change of the inputs
//!   moves the result by at most the rounding of the 8s - 1-bit operands; the only
//!   discontinuities are that rounding and a vector's scale stepping by a factor 2.
//! - Accuracy: 8s - 1 bits per operand value relative to its vector's largest, the error
//!   bound of a fixed-point product; s = 7 (55 bits) is FP64 quality on density-fitted
//!   exchange data (docs/internal/gpu-spike.md, section KPS).

#[cfg(feature = "gpu")]
pub use device::*;

/// Digit products summed exactly in INT32: |d_p d_q| <= 2^14, so 2^17 of them stay below 2^31.
pub const INT32_TERMS: usize = 1 << 17;

/// Padding of a sliced vector's length (INT8 kernels want aligned leading dimensions).
pub const K_ALIGN: usize = 16;

/// Padding of the number of vectors: cuBLAS 13.2 on sm_120 refuses INT8 GEMMs
/// (`CUBLAS_STATUS_NOT_SUPPORTED`) whose output row count is not a multiple of 4, and for
/// inner dimensions below 128 also some multiples of 4 and 8 (36, 40, 104, 2152, 2156, ...);
/// multiples of 16 always ran (`ozaki_bench probe`). A sliced operand holds zero vectors up
/// to the next multiple of 16.
pub const ROW_ALIGN: usize = 16;

/// The products (p, q), digits p of the left operand and q of the right one, of diagonal g
/// (p + q = g) for operands of `sa` and `sb` slices, in order of p.
pub fn diagonal(g: usize, sa: usize, sb: usize) -> Vec<(usize, usize)> {
    (0..=g)
        .filter(|&p| p < sa && g - p < sb)
        .map(|p| (p, g - p))
        .collect()
}

/// Groups of products of a diagonal that one INT32 accumulator holds exactly for an inner
/// dimension `k`: at most `INT32_TERMS / k` products each (at least one; `k` itself must not
/// exceed `INT32_TERMS`).
pub fn groups(prods: &[(usize, usize)], k: usize) -> Vec<Vec<(usize, usize)>> {
    let per = (INT32_TERMS / k.max(1)).max(1);
    prods.chunks(per).map(|c| c.to_vec()).collect()
}

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::blas::{Blas, GemmDims, MatMut, MatRef, Op};
    use crate::error::GpuError;
    use crate::kernels::registry::KernelModule;
    use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};
    use std::borrow::Cow;

    /// The device functions of the slicing, for kernels elsewhere in the crate that slice
    /// while they gather (the DF exchange's unpack): `emu_exponent(max)` and
    /// `emu_digits(v, e, s, desc, out, kp)`.
    pub const DEVICE_FNS: &str = include_str!("../../device/emu/digits.cuh");

    const KERNELS: &str = include_str!("../../device/emu/ozaki.cu");

    /// The order of a sliced vector's digits in memory. A product's diagonal runs as one INT8
    /// GEMM over concatenated slices when the left operand is `Ascending` and the right one
    /// `Descending` (digit p of the left meets digit g - p of the right).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Order {
        Ascending,
        Descending,
    }

    /// Where the vectors of an FP64 operand are: (buffer, element offset, leading dimension),
    /// with vector r's element kk at off + r ld + kk (`contiguous`: along k) or at
    /// off + kk ld + r.
    #[derive(Clone, Copy)]
    pub struct Vectors<'a> {
        pub buf: &'a DeviceBuf<f64>,
        pub off: usize,
        pub ld: usize,
        pub contiguous: bool,
    }

    /// An operand in Ozaki slices: `rows` vectors of length `k` (padded to `kp`), `slices`
    /// INT8 digits each, digit p of vector r at ((r slices + pos(p)) kp) of `data`, and one
    /// scale exponent per vector; zero vectors pad the count to `rows_pad` ([`ROW_ALIGN`]).
    pub struct Sliced {
        pub rows: usize,
        pub rows_pad: usize,
        pub k: usize,
        pub kp: usize,
        pub slices: usize,
        pub order: Order,
        pub data: DeviceBuf<i8>,
        pub exp: DeviceBuf<i32>,
    }

    impl Sliced {
        /// Zeroed storage for `rows` vectors of length `k` in `slices` digits.
        pub fn new(
            gpu: &Gpu,
            rows: usize,
            k: usize,
            slices: usize,
            order: Order,
        ) -> Result<Sliced, GpuError> {
            if !(1..=8).contains(&slices) {
                return Err(GpuError::Contract(format!(
                    "Ozaki: {slices} slices (1 to 8 supported)"
                )));
            }
            let kp = k.div_ceil(K_ALIGN).max(1) * K_ALIGN;
            let rows_pad = rows.div_ceil(ROW_ALIGN).max(1) * ROW_ALIGN;
            Ok(Sliced {
                rows,
                rows_pad,
                k,
                kp,
                slices,
                order,
                data: gpu.alloc_zeros::<i8>(rows_pad * slices * kp)?,
                exp: gpu.alloc_zeros::<i32>(rows_pad)?,
            })
        }

        /// Device bytes held.
        pub fn bytes(&self) -> usize {
            self.data.len() + 4 * self.exp.len()
        }

        /// The whole operand as a view.
        pub fn view(&self) -> SlicedRef<'_> {
            SlicedRef {
                data: &self.data,
                data_off: 0,
                exp: &self.exp,
                exp_off: 0,
                rows: self.rows,
                rows_pad: self.rows_pad,
                k: self.k,
                kp: self.kp,
                slices: self.slices,
                order: self.order,
            }
        }

        /// Vectors r0..r0 + n as a view. As the right operand of a product only those
        /// vectors are read; as the left one the INT8 GEMM's rows are padded to [`ROW_ALIGN`]
        /// with the vectors that follow (computed and discarded), which must then be held
        /// (checked by the product).
        pub fn rows_view(&self, r0: usize, n: usize) -> Result<SlicedRef<'_>, GpuError> {
            let np = n.div_ceil(ROW_ALIGN).max(1) * ROW_ALIGN;
            if r0 + n > self.rows_pad {
                return Err(GpuError::Contract(format!(
                    "Ozaki: vectors {r0}..{} outside the {} held",
                    r0 + n,
                    self.rows_pad
                )));
            }
            Ok(SlicedRef {
                data: &self.data,
                data_off: r0 * self.slices * self.kp,
                exp: &self.exp,
                exp_off: r0,
                rows: n,
                rows_pad: np,
                k: self.k,
                kp: self.kp,
                slices: self.slices,
                order: self.order,
            })
        }

        /// Bytes of a sliced operand of these sizes.
        pub fn bytes_for(rows: usize, k: usize, slices: usize) -> usize {
            let rp = rows.div_ceil(ROW_ALIGN).max(1) * ROW_ALIGN;
            rp * (slices * (k.div_ceil(K_ALIGN).max(1) * K_ALIGN) + 4)
        }
    }

    /// A view of a sliced operand (all of a [`Sliced`] or a range of its vectors): the digits
    /// of vector r at `data_off + (r slices + pos(p)) kp`, its exponent at `exp_off + r`.
    #[derive(Clone, Copy)]
    pub struct SlicedRef<'a> {
        pub data: &'a DeviceBuf<i8>,
        pub data_off: usize,
        pub exp: &'a DeviceBuf<i32>,
        pub exp_off: usize,
        pub rows: usize,
        pub rows_pad: usize,
        pub k: usize,
        pub kp: usize,
        pub slices: usize,
        pub order: Order,
    }

    /// Reusable device workspace of the GEMMs (INT32 partial sums, the per-vector maxima,
    /// sliced copies of per-call operands).
    #[derive(Default)]
    pub struct OzakiWork {
        parts: Option<DeviceBuf<i32>>,
        win: Option<DeviceBuf<i8>>,
        shifts: Option<DeviceBuf<i32>>,
        mx: Option<DeviceBuf<u64>>,
        tmp_a: Option<Sliced>,
        tmp_b: Option<Sliced>,
    }

    impl OzakiWork {
        /// Device bytes held.
        pub fn bytes(&self) -> usize {
            self.parts.as_ref().map_or(0, |b| 4 * b.len())
                + self.mx.as_ref().map_or(0, |b| 8 * b.len())
                + self.win.as_ref().map_or(0, |b| b.len())
                + self.tmp_a.as_ref().map_or(0, |s| s.bytes())
                + self.tmp_b.as_ref().map_or(0, |s| s.bytes())
        }

        /// Frees everything.
        pub fn clear(&mut self) {
            *self = OzakiWork::default();
        }
    }

    /// A buffer of at least `n` elements in `slot` (grown when smaller; contents undefined).
    fn ensure<T: alloc::Zeroable>(
        gpu: &Gpu,
        slot: &mut Option<DeviceBuf<T>>,
        n: usize,
    ) -> Result<(), GpuError> {
        if slot.as_ref().is_none_or(|b| b.len() < n) {
            *slot = None;
            *slot = Some(T::alloc(gpu, n.max(1))?);
        }
        Ok(())
    }

    /// The element types the workspace holds (allocation without naming the vendor runtime).
    mod alloc {
        use crate::{DeviceBuf, Gpu, GpuError};
        pub trait Zeroable: Sized {
            fn alloc(gpu: &Gpu, n: usize) -> Result<DeviceBuf<Self>, GpuError>;
        }
        impl Zeroable for i8 {
            fn alloc(gpu: &Gpu, n: usize) -> Result<DeviceBuf<i8>, GpuError> {
                gpu.alloc_zeros::<i8>(n)
            }
        }
        impl Zeroable for i32 {
            fn alloc(gpu: &Gpu, n: usize) -> Result<DeviceBuf<i32>, GpuError> {
                gpu.alloc_zeros::<i32>(n)
            }
        }
        impl Zeroable for u64 {
            fn alloc(gpu: &Gpu, n: usize) -> Result<DeviceBuf<u64>, GpuError> {
                gpu.alloc_zeros::<u64>(n)
            }
        }
    }

    struct Kernels {
        window: Kernel,
        absmax_kc: Kernel,
        absmax_rc: Kernel,
        exponents: Kernel,
        slice_kc: Kernel,
        slice_rc: Kernel,
        combine: Kernel,
    }

    /// The Ozaki-scheme FP64 GEMM engine: a cuBLAS handle for the INT8 GEMMs and the slicing
    /// and recombination kernels (see the module docs).
    pub struct Ozaki {
        blas: Blas,
        kernels: Kernels,
        /// Products with at most this many right vectors (skinny GEMMs, limited by reading the
        /// left operand's digits) run neighbouring diagonals in pairs: one INT8 GEMM over the
        /// left digits 0..g+1 for diagonals g + 1 and g (the right operand zero-padded), which
        /// reads the left operand 16 times instead of 28 at 7 slices for 11 % more products.
        /// The integer sums are the same, so the result is too, bit for bit. 0: never. The
        /// default, 192, from sm_120: GEMM1 of ts61/TZVP's exchange blocks (397k x 153 x 1344)
        /// 1.18x faster, of ts113's (241k x 233 x 2160) 2 % slower.
        pub pair_max_n: usize,
    }

    /// The Ozaki scheme's module: the slicing's device functions, then its kernels.
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "emu",
        || Cow::Owned(format!("{DEVICE_FNS}{KERNELS}")),
        &[
            "emu_window",
            "emu_absmax_kc",
            "emu_absmax_rc",
            "emu_exponents",
            "emu_slice_kc",
            "emu_slice_rc",
            "emu_combine",
        ],
    );

    impl Ozaki {
        pub fn new(gpu: &Gpu) -> Result<Ozaki, GpuError> {
            let src = MODULE.source();
            let k = |e: &str| gpu.kernel(MODULE.name, &src, e);
            Ok(Ozaki {
                pair_max_n: 192,
                blas: Blas::new(gpu)?,
                kernels: Kernels {
                    window: k("emu_window")?,
                    absmax_kc: k("emu_absmax_kc")?,
                    absmax_rc: k("emu_absmax_rc")?,
                    exponents: k("emu_exponents")?,
                    slice_kc: k("emu_slice_kc")?,
                    slice_rc: k("emu_slice_rc")?,
                    combine: k("emu_combine")?,
                },
            })
        }

        /// The scale exponents of `out.rows` vectors of `src` (length `out.k`) into
        /// `out.exp`, using `mx` (at least `out.rows` u64) as scratch.
        fn exponents(
            &self,
            gpu: &Gpu,
            src: Vectors<'_>,
            out: &mut Sliced,
            mx: &mut DeviceBuf<u64>,
        ) -> Result<(), GpuError> {
            let (rows, k) = (out.rows, out.k);
            if rows == 0 {
                return Ok(());
            }
            check_src(&src, rows, k)?;
            gpu.zero_range(mx, 0, rows)?;
            let x = Arg::from(src.buf.at(src.off));
            let args = [
                Arg::I64(rows as i64),
                Arg::I64(k as i64),
                x,
                Arg::I64(src.ld as i64),
                Arg::buf_mut(mx),
            ];
            if k > 0 {
                let (kernel, cfg) = if src.contiguous {
                    (
                        &self.kernels.absmax_kc,
                        Launch {
                            grid: (rows as u32, k.div_ceil(2048) as u32, 1),
                            block: (256, 1, 1),
                            shared_bytes: 0,
                        },
                    )
                } else {
                    (
                        &self.kernels.absmax_rc,
                        Launch {
                            grid: (rows.div_ceil(32) as u32, k.div_ceil(256) as u32, 1),
                            block: (32, 8, 1),
                            shared_bytes: 0,
                        },
                    )
                };
                // SAFETY: matches emu_absmax_kc/rc; reads the rows x k values of src
                // (checked to lie in its buffer), writes rows u64 of mx.
                unsafe { gpu.launch(kernel, cfg, &args) }?;
            }
            // SAFETY: matches emu_exponents; rows u64 in, rows i32 out (out.exp holds rows).
            unsafe {
                gpu.launch(
                    &self.kernels.exponents,
                    Launch::linear(rows, 256),
                    &[
                        Arg::I64(rows as i64),
                        Arg::buf(mx),
                        Arg::buf_mut(&mut out.exp),
                    ],
                )
            }
        }

        /// The digits of the vectors of `src` with the exponents already in `out.exp`.
        pub fn slice_digits(
            &self,
            gpu: &Gpu,
            src: Vectors<'_>,
            out: &mut Sliced,
        ) -> Result<(), GpuError> {
            let (rows, k, kp, s) = (out.rows, out.k, out.kp, out.slices);
            if rows == 0 {
                return Ok(());
            }
            check_src(&src, rows, k)?;
            let args = [
                Arg::I64(rows as i64),
                Arg::I64(k as i64),
                Arg::I64(kp as i64),
                Arg::I32(s as i32),
                Arg::I32((out.order == Order::Descending) as i32),
                Arg::from(src.buf.at(src.off)),
                Arg::I64(src.ld as i64),
                Arg::buf(&out.exp),
                Arg::buf_mut(&mut out.data),
            ];
            let (kernel, cfg) = if src.contiguous {
                (
                    &self.kernels.slice_kc,
                    Launch {
                        grid: (rows as u32, kp.div_ceil(256) as u32, 1),
                        block: (256, 1, 1),
                        shared_bytes: 0,
                    },
                )
            } else {
                (
                    &self.kernels.slice_rc,
                    Launch {
                        grid: (rows.div_ceil(32) as u32, kp.div_ceil(32) as u32, 1),
                        block: (32, 8, 1),
                        shared_bytes: 0,
                    },
                )
            };
            // SAFETY: matches emu_slice_kc/rc; reads rows x k values of src (checked), writes
            // rows s kp digits of out.data (its size) with out.exp's rows exponents.
            unsafe { gpu.launch(kernel, cfg, &args) }
        }

        /// Slices the vectors of `src` into `out` (its rows, k, slice count and order):
        /// exponents, then digits. `work` holds the maxima.
        pub fn slice(
            &self,
            gpu: &Gpu,
            src: Vectors<'_>,
            out: &mut Sliced,
            work: &mut OzakiWork,
        ) -> Result<(), GpuError> {
            ensure(gpu, &mut work.mx, out.rows)?;
            let mx = work.mx.as_mut().unwrap();
            self.exponents(gpu, src, out, mx)?;
            self.slice_digits(gpu, src, out)
        }

        /// C = alpha A B^T + beta C in emulated FP64: the rows of C are `a`'s vectors (rows of
        /// op(A)), its columns `b`'s (columns of op(B)); C is a.rows x b.rows at (buffer,
        /// offset, ldc). The diagonals g < max(slices) of the digit products (module docs);
        /// each runs as concatenated GEMMs when `a` is ascending and `b` descending, else
        /// product by product.
        #[allow(clippy::too_many_arguments)]
        pub fn gemm(
            &self,
            gpu: &Gpu,
            a: SlicedRef<'_>,
            b: SlicedRef<'_>,
            alpha: f64,
            beta: f64,
            c: (&mut DeviceBuf<f64>, usize, usize),
            work: &mut OzakiWork,
        ) -> Result<(), GpuError> {
            self.run(gpu, a, b, alpha, beta, c, work, None)
        }

        /// The lower triangle (rows >= columns) of C = alpha A A^T + beta C, `a`'s vectors
        /// as both rows and columns, as integer GEMMs over `blocks` column blocks of the lower
        /// triangle (the upper triangle of C is not written). `b` holds the same vectors: `a`
        /// itself (a product per digit pair), or sliced again in descending digit order (each
        /// diagonal's products concatenated into one GEMM per column block; the same result
        /// bit for bit, fewer and larger GEMMs).
        #[allow(clippy::too_many_arguments)]
        pub fn syrk_lower(
            &self,
            gpu: &Gpu,
            a: SlicedRef<'_>,
            b: SlicedRef<'_>,
            alpha: f64,
            beta: f64,
            c: (&mut DeviceBuf<f64>, usize, usize),
            blocks: usize,
            work: &mut OzakiWork,
        ) -> Result<(), GpuError> {
            if b.rows != a.rows || b.k != a.k || b.slices != a.slices {
                return Err(GpuError::Contract(
                    "Ozaki SYRK: the right operand must hold the left one's vectors".into(),
                ));
            }
            self.run(gpu, a, b, alpha, beta, c, work, Some(blocks.max(1)))
        }

        /// [`Ozaki::gemm`] (`lower` None) or [`Ozaki::syrk_lower`] (`lower`: column blocks).
        #[allow(clippy::too_many_arguments)]
        fn run(
            &self,
            gpu: &Gpu,
            a: SlicedRef<'_>,
            b: SlicedRef<'_>,
            alpha: f64,
            beta: f64,
            c: (&mut DeviceBuf<f64>, usize, usize),
            work: &mut OzakiWork,
            lower: Option<usize>,
        ) -> Result<(), GpuError> {
            let (m, n) = (a.rows, b.rows);
            if a.k != b.k || a.kp != b.kp {
                return Err(GpuError::Contract(format!(
                    "Ozaki GEMM: inner dimensions {} and {} differ",
                    a.k, b.k
                )));
            }
            if lower.is_some() && m != n {
                return Err(GpuError::Contract("Ozaki SYRK: not square".into()));
            }
            let (cbuf, coff, ldc) = c;
            if m == 0 || n == 0 {
                return Ok(());
            }
            if ldc < m || coff + (n - 1) * ldc + m > cbuf.len() {
                return Err(GpuError::Contract(
                    "Ozaki GEMM: C outside its buffer".into(),
                ));
            }
            let (k, kp) = (a.k, a.kp);
            if k > INT32_TERMS {
                return Err(GpuError::Contract(format!(
                    "Ozaki GEMM: inner dimension {k} above {INT32_TERMS} (split it)"
                )));
            }
            let (sa, sb) = (a.slices, b.slices);
            let ng = sa.max(sb);
            // the partials: one per group of products, diagonals from the smallest scale
            let mut parts: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
            for g in (0..ng).rev() {
                for grp in groups(&diagonal(g, sa, sb), k) {
                    parts.push((g, grp));
                }
            }
            let np = parts.len();
            // the INT8 GEMMs' rows: a's padded vectors (zero digits beyond m)
            let mp = a.rows_pad;
            ensure(gpu, &mut work.parts, np * mp * n)?;
            // the partials' diagonals, decreasing (the recombination's order)
            let shifts: Vec<i32> = parts.iter().map(|(g, _)| *g as i32).collect();
            match work.shifts.as_mut() {
                Some(sb) if sb.len() >= np => gpu.upload_into(sb, 0, &shifts)?,
                _ => work.shifts = Some(gpu.upload(&shifts)?),
            }
            let pbuf = work.parts.as_mut().unwrap();
            // the left operand's padded rows are read, the right one's vectors only
            if a.data_off + a.rows_pad * sa * kp > a.data.len()
                || b.data_off + b.rows * sb * kp > b.data.len()
                || a.exp_off + a.rows > a.exp.len()
                || b.exp_off + b.rows > b.exp.len()
            {
                return Err(GpuError::Contract(
                    "Ozaki GEMM: a view outside its buffers".into(),
                ));
            }
            let concat = a.order == Order::Ascending && b.order == Order::Descending;
            let pos = |o: Order, s: usize, p: usize| match o {
                Order::Ascending => p,
                Order::Descending => s - 1 - p,
            };
            let (lda, ldb) = (sa * kp, sb * kp);
            // neighbouring diagonals in pairs (`pair_max_n`): every diagonal one group
            let paired = concat
                && lower.is_none()
                && n <= self.pair_max_n
                && parts.len() == ng
                && ng >= 2
                && sa == sb;
            if paired {
                let gmax0 = parts[0].0;
                ensure(gpu, &mut work.win, 2 * n * (gmax0 + 1) * kp)?;
                let win = work.win.as_mut().unwrap();
                let mut j = 0;
                while j < np {
                    let set = if j + 1 < np { 2 } else { 1 };
                    let gmax = parts[j].0;
                    let kk = (gmax + 1) * kp;
                    for d in 0..set {
                        let g = parts[j + d].0;
                        // SAFETY: matches emu_window; reads n vectors of b (slots s-1-g..s-1
                        // of each, inside b's view, checked above), writes n kk bytes of win at
                        // d n kk (win holds 2 n (gmax0 + 1) kp >= that).
                        unsafe {
                            gpu.launch(
                                &self.kernels.window,
                                Launch {
                                    grid: (n as u32, (kk / 4).div_ceil(256) as u32, 1),
                                    block: (256, 1, 1),
                                    shared_bytes: 0,
                                },
                                &[
                                    Arg::I64(n as i64),
                                    Arg::I64(kp as i64),
                                    Arg::I32(sb as i32),
                                    Arg::I32(g as i32),
                                    Arg::I32(gmax as i32),
                                    Arg::from(b.data.at(b.data_off)),
                                    Arg::from(win.at(d * n * kk)),
                                ],
                            )
                        }?;
                    }
                    // diagonals gmax (columns 0..n) and gmax - 1 (n..2n) into partials j, j+1
                    self.blas.gemm_i8(
                        GemmDims::new(Op::T, Op::N, mp, set * n, kk),
                        (1, 0),
                        MatRef::new(a.data, a.data_off, lda),
                        MatRef::new(&*win, 0, kk),
                        MatMut::new(&mut *pbuf, j * mp * n, mp),
                    )?;
                    j += set;
                }
            }
            for (j, (_, grp)) in parts.iter().enumerate() {
                if paired {
                    break;
                }
                let po = j * mp * n;
                // the group's INT8 GEMMs: (a's digits, b's digits, inner dimension, beta)
                let calls: Vec<(usize, usize, usize, i32)> = if concat {
                    // digits p0..p0+len of a meet g-p0.. of b: both contiguous
                    let (p0, q0) = grp[0];
                    vec![(
                        a.data_off + p0 * kp,
                        b.data_off + pos(b.order, sb, q0) * kp,
                        grp.len() * kp,
                        0,
                    )]
                } else {
                    grp.iter()
                        .enumerate()
                        .map(|(i, &(p, q))| {
                            (
                                a.data_off + pos(a.order, sa, p) * kp,
                                b.data_off + pos(b.order, sb, q) * kp,
                                kp,
                                (i > 0) as i32,
                            )
                        })
                        .collect()
                };
                for (ao, bo, kk, beta_i) in calls {
                    match lower {
                        None => self.blas.gemm_i8(
                            GemmDims::new(Op::T, Op::N, mp, n, kk),
                            (1, beta_i),
                            MatRef::new(a.data, ao, lda),
                            MatRef::new(b.data, bo, ldb),
                            MatMut::new(pbuf, po, mp),
                        )?,
                        Some(nb) => {
                            // column blocks starting at multiples of ROW_ALIGN, so that the
                            // rows j0..mp of each block stay a multiple of it
                            let bound = |blk: usize| (blk * m / nb) / ROW_ALIGN * ROW_ALIGN;
                            for blk in 0..nb {
                                let j0 = bound(blk);
                                let j1 = if blk + 1 == nb { m } else { bound(blk + 1) };
                                if j1 <= j0 {
                                    continue;
                                }
                                // rows j0.. of columns j0..j1
                                self.blas.gemm_i8(
                                    GemmDims::new(Op::T, Op::N, mp - j0, j1 - j0, kk),
                                    (1, beta_i),
                                    MatRef::new(a.data, j0 * lda + ao, lda),
                                    MatRef::new(b.data, j0 * ldb + bo, ldb),
                                    MatMut::new(pbuf, po + j0 * mp + j0, mp),
                                )?;
                            }
                        }
                    }
                }
            }
            let len = m * n;
            // SAFETY: matches emu_combine; np partials of mp n INT32 (ensured), np shifts, m and
            // n exponents, C m x n at coff with ldc (checked above).
            unsafe {
                gpu.launch(
                    &self.kernels.combine,
                    Launch::linear(len, 256),
                    &[
                        Arg::I64(m as i64),
                        Arg::I64(n as i64),
                        Arg::I64(mp as i64),
                        Arg::I32(np as i32),
                        Arg::buf(pbuf),
                        Arg::buf(work.shifts.as_ref().unwrap()),
                        Arg::from(a.exp.at(a.exp_off)),
                        Arg::from(b.exp.at(b.exp_off)),
                        Arg::F64(alpha),
                        Arg::F64(beta),
                        Arg::from(cbuf.at(coff)),
                        Arg::I64(ldc as i64),
                        Arg::I32(lower.is_some() as i32),
                    ],
                )
            }
        }

        /// C = alpha op(A) op(B) + beta C for FP64 operands, as [`Blas::gemm`], in
        /// emulated FP64 with `slices` slices: both operands are sliced on every call (into
        /// `work`), so this pays only where the GEMM is large against its operands; a stored
        /// operand should be sliced once ([`Ozaki::slice`]) and multiplied with
        /// [`Ozaki::gemm`].
        #[allow(clippy::too_many_arguments)]
        pub fn dgemm_at(
            &self,
            gpu: &Gpu,
            slices: usize,
            ta: Op,
            tb: Op,
            m: usize,
            n: usize,
            k: usize,
            alpha: f64,
            a: (&DeviceBuf<f64>, usize, usize),
            b: (&DeviceBuf<f64>, usize, usize),
            beta: f64,
            c: (&mut DeviceBuf<f64>, usize, usize),
            work: &mut OzakiWork,
        ) -> Result<(), GpuError> {
            let fits = |s: &Option<Sliced>, rows: usize| {
                s.as_ref()
                    .is_some_and(|s| s.rows == rows && s.k == k && s.slices == slices)
            };
            if !fits(&work.tmp_a, m) {
                work.tmp_a = None;
                work.tmp_a = Some(Sliced::new(gpu, m, k, slices, Order::Ascending)?);
            }
            if !fits(&work.tmp_b, n) {
                work.tmp_b = None;
                work.tmp_b = Some(Sliced::new(gpu, n, k, slices, Order::Descending)?);
            }
            let mut sa = work.tmp_a.take().unwrap();
            let mut sb = work.tmp_b.take().unwrap();
            let src_a = Vectors {
                buf: a.0,
                off: a.1,
                ld: a.2,
                contiguous: ta == Op::T,
            };
            let src_b = Vectors {
                buf: b.0,
                off: b.1,
                ld: b.2,
                contiguous: tb == Op::N,
            };
            let r = self
                .slice(gpu, src_a, &mut sa, work)
                .and_then(|_| self.slice(gpu, src_b, &mut sb, work))
                .and_then(|_| self.gemm(gpu, sa.view(), sb.view(), alpha, beta, c, work));
            work.tmp_a = Some(sa);
            work.tmp_b = Some(sb);
            r
        }
    }

    /// Checks that `rows` vectors of length `k` lie inside the source buffer.
    fn check_src(src: &Vectors<'_>, rows: usize, k: usize) -> Result<(), GpuError> {
        if rows == 0 || k == 0 {
            return Ok(());
        }
        let (inner, outer) = if src.contiguous { (k, rows) } else { (rows, k) };
        if src.ld < inner || src.off + (outer - 1) * src.ld + inner > src.buf.len() {
            return Err(GpuError::Contract(
                "Ozaki: operand outside its buffer".into(),
            ));
        }
        Ok(())
    }
}
