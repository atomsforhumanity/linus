//! The projected response J/K on the device: `GpuDfResponse` (feature `gpu`; the algorithm
//! is in the [module docs](super)).
//!
//! The engine owns the `GpuDfJk` whose fitted tensor it reads (device or host storage; a
//! range-separated exchange needs device storage), the occupied orbitals of each channel and
//! their half-transformed tensors Y_s (and Yw_s), all on the device. Inputs and outputs are
//! column-major host matrices with the perturbations side by side: per channel s the factors
//! A_{s,d} as one n x (nd o_s) matrix (perturbation d at columns d o_s..), and the products
//! G_{s,d} in the same layout.

#[cfg(feature = "gpu")]
pub use device::*;

#[cfg(feature = "gpu")]
mod device {
    use crate::blas::{GemmDims, GemmScalar, MatMut, MatRef, Op};
    use crate::cpu::{metric, Aux3c, BasisSet, Operator};
    use crate::df::{
        DfLayout, DfStorage, Factor, GpuDfJk, KMath, OzTaken, TensorBlock, ROW_OFFDIAG,
    };
    use crate::emu::{Order, Ozaki, OzakiWork, Sliced, Vectors};
    use crate::engine::Timeline;
    use crate::error::GpuError;
    use crate::kernels::registry::KernelModule;
    use crate::linalg::{Arith, Gemm};
    use crate::solver::Solver;
    use crate::{Arg, Copies, DeviceBuf, Gpu, Kernel, Launch, StreamedStore};
    use std::borrow::Cow;
    use std::time::Instant;

    const KERNELS: &str = include_str!("../../device/response/df.cu");

    /// The response J/K's module.
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "resp_df",
        || Cow::Borrowed(KERNELS),
        &[
            "resp_y_block",
            "resp_to_f32",
            "resp_add_f32",
            "resp_op_fill",
            "resp_op_fill_mixed",
            "resp_acc_f32",
        ],
    );

    /// Arithmetic of the response exchange GEMMs (the setup and J are native FP64).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum RespMath {
        /// Native FP64.
        Native,
        /// cuBLAS's fixed-point FP64 emulation with a fixed mantissa of `bits` bits.
        Emulated { bits: i32 },
        /// FP32 GEMMs (true FP32, no TF32), summed in FP32 over an auxiliary block and then in
        /// FP64: the counterpart of the CPU's single-precision response exchange, for the
        /// same uses (its Hessian errors are at or below the CPU single mode's).
        Single,
        /// The Ozaki scheme ([`crate::emu`], `slices` INT8 digits) for GEMM1, Z = B^T A, on the
        /// DF-J/K engine's pre-sliced planes, and with range separation on the operator
        /// tensor's planes too (required: an `Unsupported` error when they do not fit); the
        /// contractions in native FP64.
        Ozaki { slices: usize },
    }

    /// Settings of a [`GpuDfResponse`].
    #[derive(Clone, Debug)]
    pub struct GpuDfResponseOptions {
        /// Arithmetic of the exchange GEMMs (the setup and J are native FP64).
        pub k_math: RespMath,
        /// Device bytes of the exchange workspace per call (unpacked planes, Z and the
        /// stacked products): sets the perturbation groups and the block size.
        pub workspace: usize,
        /// Auxiliary functions per exchange block (0: from the workspace).
        pub k_block: usize,
        /// The exchange on a stored operator ([`KOperator`]).
        pub k_operator: KOperator,
        /// Device bytes the stored operator leaves free for other engines (the CPKS's XC
        /// kernel takes its workspace per call).
        pub k_operator_reserve: usize,
        /// The exchange arithmetic to switch to when [`KOperator::Auto`] would build the
        /// operator but it does not fit (None: keep `k_math`). linus's `precision.response:
        /// auto` on the GPU: the operator in FP64, else the FP32 exchange.
        pub k_operator_fallback: Option<RespMath>,
        /// Bytes of page-locked host memory the stored operator may take for the chunks that
        /// do not fit the device (streamed to it in every product; 0: device only).
        pub k_operator_host: usize,
        /// At most this many bytes of the stored operator on the device (None: as many as fit
        /// beside the reserve), the rest in host memory: for tests and measurements of the
        /// streamed operator on small molecules.
        pub k_operator_device_cap: Option<usize>,
        /// With [`RespMath::Single`]: the exchange on a stored operator in FP32 ([`KOperator`]
        /// as for FP64; its setup and products are SGEMMs, each chunk's product added to the
        /// FP64 result) instead of the FP32 exchange through the tensor.
        pub k_operator_fp32: bool,
        /// Bytes of page-locked host memory for the FP32 operator's chunks that do not fit the
        /// device (None: `k_operator_host`).
        pub k_operator_host32: Option<usize>,
        /// Device bytes the Ozaki exchange's pre-sliced planes of a range-separated operator
        /// tensor W leave free beside the exchange workspace (2 GiB).
        pub w_planes_reserve: usize,
    }

    impl Default for GpuDfResponseOptions {
        fn default() -> Self {
            GpuDfResponseOptions {
                k_math: RespMath::Native,
                workspace: 8 << 30,
                k_block: 0,
                k_operator: KOperator::Auto,
                k_operator_reserve: 6 << 30,
                k_operator_fallback: None,
                k_operator_host: 0,
                k_operator_device_cap: None,
                k_operator_fp32: false,
                k_operator_host32: None,
                w_planes_reserve: 2 << 30,
            }
        }
    }

    /// Whether the exchange runs on a stored operator: per channel the symmetric
    /// (n o) x (n o) matrix Op with K\[dD\] C = Op vec(A) for every factor A (the [module
    /// docs](super)), built once from the tensor and Y. A product then costs 2 (n o)^2 nd flops
    /// instead of about 2 n^2 o naux nd (1 + 3 o / n) through the tensor (20-40x fewer at
    /// def2-SVP sizes), at the price of (n o)^2 / 2 doubles of device memory and a setup of
    /// about (n o)^2 naux flops: for products of many perturbations (Hessians), not for a
    /// polarizability's three. Not with [`RespMath::Single`] (its FP32 exchange runs through
    /// the tensor).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum KOperator {
        /// The exchange through the tensor in every product.
        Off,
        /// Built at the first product whose perturbations make it pay (its setup and one
        /// product on it below two products through the tensor) when it fits the device beside
        /// `k_operator_reserve` bytes; else, and until then, through the tensor.
        Auto,
        /// Built at the first product (an error when it does not fit).
        On,
    }

    /// What a setup resolved to.
    #[derive(Clone, Debug)]
    pub struct GpuDfResponseResolved {
        pub nbf: usize,
        pub naux: usize,
        /// Occupied orbitals per channel.
        pub occ: Vec<usize>,
        /// The exchange operator of a range-separated hybrid (wK), if any.
        pub exchange_operator: Option<Operator>,
        pub storage: DfStorage,
        pub k_math: RespMath,
        /// Device bytes of the stored Y_s (and Yw_s) and of the operator tensor W.
        pub y_bytes: usize,
        pub w_bytes: usize,
        /// Bytes of the stored exchange operator ([`KOperator`]; 0: not built), and of them in
        /// page-locked host memory (streamed to the device in every product).
        pub k_operator_bytes: usize,
        pub k_operator_host_bytes: usize,
        /// Why [`KOperator::Auto`] left the exchange on the tensor when a product would have
        /// paid for the operator (it did not fit, or the tensor is not on the device).
        pub k_operator_note: Option<String>,
        /// The device bytes granted to the stored operator at the setup
        /// ([`GpuDfResponse::grant_k_operator`]; 0: none), and the build's draws beyond them.
        pub k_operator_granted: usize,
        pub k_operator_violations: Vec<crate::pool::PlanViolation>,
    }

    /// What the stored exchange operator would cost for products of a number of perturbations
    /// ([`GpuDfResponse::k_operator_plan`]).
    #[derive(Clone, Debug)]
    pub struct KOperatorPlan {
        /// Its setup and one product on it cost fewer flops than two products through the
        /// tensor ([`KOperator::Auto`] builds it).
        pub pays: bool,
        /// Bytes of the operators (all channels) and of the setup's temporaries.
        pub bytes: usize,
        pub setup_bytes: usize,
        /// Of `bytes`, in page-locked host memory (what does not fit the device beside the
        /// reserve), streamed through `staging_bytes` of device buffers.
        pub host_bytes: usize,
        pub staging_bytes: usize,
        /// Why it would not be built now (None: it would): the tensor is not on the device, or
        /// it does not fit beside the reserve.
        pub blocked: Option<String>,
        /// The estimated flops ([`KOperator`]'s model, all channels): of its setup, of one
        /// product of the `nd` perturbations on it, and of one such product through the tensor
        /// (the same GEMMs in every arithmetic). For cost models of the exchange path.
        pub setup_flops: f64,
        pub product_flops: f64,
        pub tensor_flops: f64,
    }

    /// Setup times (seconds, wall clock).
    #[derive(Clone, Debug, Default)]
    pub struct GpuDfResponseSetupTimes {
        /// The operator tensor W: integrals (CPU) and fit (device).
        pub w_tensor: f64,
        /// Y_s = B C_s (and Yw_s) for every auxiliary function.
        pub y: f64,
        pub total: f64,
    }

    /// Device times of the last product (seconds, from events).
    #[derive(Clone, Debug, Default)]
    pub struct GpuDfResponseCallTimes {
        pub total: f64,
        /// Factors to the device.
        pub upload: f64,
        pub j: f64,
        /// Host storage: tensor blocks to the device.
        pub stream: f64,
        pub k_unpack: f64,
        /// Z = B^T A (and Zw).
        pub k_gemm1: f64,
        /// The block's Y and X, the stacked products and the batched contractions (the sum of
        /// `k_xy`, `k_wst` and `k_batched`).
        pub k_contract: f64,
        /// Of `k_contract`: the block's Y (and Yw) in (i, q) order and X = Y^T C.
        pub k_xy: f64,
        /// Of `k_contract`: the stacked products Wst = Z^T C.
        pub k_wst: f64,
        /// Of `k_contract`: the two batched contractions over the perturbations.
        pub k_batched: f64,
        /// The stored exchange operator: its setup (in the product that built it) and the
        /// product on it.
        pub k_operator_setup: f64,
        pub k_operator: f64,
        /// Results to the host.
        pub download: f64,
        /// Perturbations per group and auxiliary functions per block of the exchange.
        pub k_group: usize,
        pub k_block: usize,
    }

    impl crate::engine::Phases for GpuDfResponseCallTimes {
        /// The exchange's parts as reported: the unpack, Z = B^T A, the contractions, and the
        /// stored operator's setup and product.
        fn phases(&self) -> impl IntoIterator<Item = (&'static str, f64)> {
            [
                ("upload", self.upload),
                ("j", self.j),
                ("stream", self.stream),
                ("k_unpack", self.k_unpack),
                ("k_gemm1", self.k_gemm1),
                ("k_contract", self.k_contract),
                ("k_operator_setup", self.k_operator_setup),
                ("k_operator", self.k_operator),
                ("download", self.download),
            ]
        }
    }

    struct Kernels {
        y_block: Kernel,
        to_f32: Kernel,
        add_f32: Kernel,
        op_fill: Kernel,
        op_fill_mixed: Kernel,
        acc_f32: Kernel,
    }

    struct Chan {
        o: usize,
        /// C_s, n x o column-major.
        c: DeviceBuf<f64>,
        /// Yp_s = [B_Q C_s]_Q, (n o) x naux column-major.
        yp: DeviceBuf<f64>,
        /// Yw_s = [W_Q C_s]_Q (range separation).
        ypw: Option<DeviceBuf<f64>>,
    }

    /// Elements of an operator chunk at most (1 GiB of FP64) unless one orbital's columns are
    /// more: chunks hold whole orbitals.
    const OP_CHUNK: usize = 1 << 27;
    /// Staging buffers of a streamed operator (copies in flight).
    const OP_STAGING: usize = 3;
    /// Chunk offsets are multiples of this many elements (512 bytes in FP64, 256 in FP32), so a
    /// chunk's GEMMs see the same alignment on the device and in a staging buffer (cuBLAS's kernel
    /// choice, and so the bits, may depend on it).
    const OP_ALIGN: usize = 64;

    /// Where a chunk of the stored operator is: element offsets in the device or host store.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Loc {
        Device(usize),
        Host(usize),
    }

    /// One chunk of a channel's stored operator: the columns of orbitals i0..i1 of panel
    /// (j0, bj) of the lower block triangle of the symmetric Op (rows (m, j), j0 <= j < j0 + bj;
    /// columns (l, i); r = m + n j), bj n x (i1 - i0) n column-major with leading dimension
    /// bj n.
    #[derive(Clone, Copy, Debug)]
    struct OpChunk {
        j0: usize,
        bj: usize,
        i0: usize,
        i1: usize,
        at: Loc,
    }

    impl OpChunk {
        fn rows(&self, n: usize) -> usize {
            self.bj * n
        }
        fn len(&self, n: usize) -> usize {
            self.bj * n * (self.i1 - self.i0) * n
        }
        /// Below its panel's diagonal block: its transpose acts on the panel's rows too.
        fn below(&self) -> bool {
            self.i1 <= self.j0
        }
    }

    /// The stored exchange operator of every channel ([`KOperator`]): its chunks on the device
    /// and, where they do not all fit, the rest in page-locked host memory, copied to staging
    /// buffers on a second stream in every product, overlapped with the products of the
    /// chunks before them.
    struct StoredOp<T> {
        /// Per channel its chunks in processing order (none without occupied orbitals).
        chunks: Vec<Vec<OpChunk>>,
        dev: Option<DeviceBuf<T>>,
        /// The chunks that do not fit on the device, in page-locked host memory streamed
        /// through [`OP_STAGING`] staging buffers on a copy stream.
        host: Option<StreamedStore<T>>,
        bytes: usize,
        host_bytes: usize,
    }

    /// Orbitals per operator panel: about 16 panels.
    fn op_block(o: usize) -> usize {
        o.div_ceil(16).max(1)
    }

    /// The chunk grid of one channel's operator (`n` functions, `o` orbitals) in processing
    /// order, as (j0, bj, i0, i1): panel by panel, the columns below the diagonal block, then
    /// the diagonal block, each in groups of whole orbitals of at most [`OP_CHUNK`] elements.
    /// It depends on n and o only, so the products' operations (and bits) do not depend on
    /// where the chunks are stored.
    fn op_grid(n: usize, o: usize) -> Vec<(usize, usize, usize, usize)> {
        let b = op_block(o);
        let mut v = Vec::new();
        for j0 in (0..o).step_by(b) {
            let bj = b.min(o - j0);
            let per = (OP_CHUNK / (bj * n * n).max(1)).max(1);
            for (s, e) in [(0, j0), (j0, j0 + bj)] {
                let mut i0 = s;
                while i0 < e {
                    let i1 = (i0 + per).min(e);
                    v.push((j0, bj, i0, i1));
                    i0 = i1;
                }
            }
        }
        v
    }

    /// Where the chunks of the channels with `occ` orbitals go for a device budget of
    /// `budget` elements (`cap`: at most this many on the device): all on the device when they
    /// fit, else the device keeps what fits beside [`OP_STAGING`] staging buffers of the
    /// largest chunk and the rest goes to host memory, spread evenly through the processing
    /// order so that the copies overlap the products of device chunks. Returns the chunks per
    /// channel, the device and host elements and the elements of one staging buffer (0: none).
    fn op_layout(
        n: usize,
        occ: &[usize],
        budget: usize,
        cap: Option<usize>,
    ) -> (Vec<Vec<OpChunk>>, usize, usize, usize) {
        let up = |x: usize| x.div_ceil(OP_ALIGN) * OP_ALIGN;
        let grids: Vec<Vec<(usize, usize, usize, usize)>> = occ
            .iter()
            .map(|&o| if o == 0 { Vec::new() } else { op_grid(n, o) })
            .collect();
        let lens = |g: &(usize, usize, usize, usize)| up(g.1 * n * (g.3 - g.2) * n);
        let total: usize = grids.iter().flatten().map(lens).sum();
        let maxlen = grids.iter().flatten().map(lens).max().unwrap_or(0);
        let budget = cap.map_or(budget, |c| budget.min(c));
        let (dmax, staging) = if total <= budget {
            (total, 0)
        } else {
            (budget.saturating_sub(OP_STAGING * maxlen), maxlen)
        };
        let host_total = total - dmax.min(total);
        let (mut dev, mut host, mut cum) = (0usize, 0usize, 0usize);
        let chunks = grids
            .iter()
            .map(|g| {
                g.iter()
                    .map(|c| {
                        let len = lens(c);
                        cum += len;
                        // host when the device is full, or while behind the even share of the
                        // host part and within it
                        let share = (host_total as f64 * cum as f64 / total as f64) as usize;
                        let at = if dev + len > dmax
                            || (host + len / 2 < share && host + len <= host_total)
                        {
                            host += len;
                            Loc::Host(host - len)
                        } else {
                            dev += len;
                            Loc::Device(dev - len)
                        };
                        OpChunk {
                            j0: c.0,
                            bj: c.1,
                            i0: c.2,
                            i1: c.3,
                            at,
                        }
                    })
                    .collect()
            })
            .collect();
        (chunks, dev, host, if host > 0 { staging } else { 0 })
    }

    /// Estimates for the stored operator of `nd` perturbations, in FP64 or (`fp32`) FP32:
    /// (flops of the exchange through the tensor, flops of the operator's setup, flops of a
    /// product on it, bytes of the operator, device bytes of its setup's temporaries).
    fn op_estimate(
        n: usize,
        naux: usize,
        nx: usize,
        occ: &[usize],
        rs: bool,
        nd: usize,
        fp32: bool,
    ) -> (f64, f64, f64, usize, usize) {
        let r = if rs { 2.0 } else { 1.0 };
        let (fnn, fa, fx, fd) = (n as f64, naux as f64, nx as f64, nd as f64);
        let (mut tensor, mut setup, mut call, mut bytes, mut temp) = (0.0, 0.0, 0.0, 0, 0);
        for &o in occ.iter().filter(|&&o| o > 0) {
            let fo = o as f64;
            let over = 1.0 + op_block(o) as f64 / fo;
            tensor += r * fd * (2.0 * fnn * fnn * fo * fa + 6.0 * fnn * fo * fo * fa);
            setup += r * ((fnn * fo).powi(2) * fa * over + fx * fa * fo * fo * over)
                + r * 2.0 * fo * fo * fnn * fa;
            call += 2.0 * (fnn * fo).powi(2) * fd;
            let ri = if rs { 2 } else { 1 };
            let t = if fp32 {
                // FP32 copies of the tensor (and W), Y, X; the pair part and a chunk's Y part in
                // FP32 block sums and FP64 totals; X in FP64 first
                bytes += 4 * op_layout(n, &[o], usize::MAX, None).1;
                4 * (ri * naux * nx + nx * op_block(o) * o + ri * n * o * naux + ri * o * o * naux)
                    + 8 * (o * o * naux + nx * op_block(o) * o)
                    + 12 * OP_CHUNK
                    + 4 * n * n
            } else {
                bytes += 8 * op_layout(n, &[o], usize::MAX, None).1;
                8 * (nx * op_block(o) * o + ri * o * o * naux) + 4 * n * n
            };
            temp = temp.max(t);
        }
        (tensor, setup, call, bytes, temp)
    }

    /// Projected response J/K of factored first-order densities (see the module docs).
    pub struct GpuDfResponse {
        jk: GpuDfJk,
        /// W = L^-1 (Q|w|mn), naux x nx column-major (range separation).
        w: Option<DeviceBuf<f64>>,
        /// W's Ozaki GEMM1 planes ([`RespMath::Ozaki`] with range separation).
        w_planes: Option<Sliced>,
        chans: Vec<Chan>,
        nbf: usize,
        naux: usize,
        /// The basis sets (the stored operator's row map).
        basis: BasisSet,
        aux: BasisSet,
        /// The stored exchange operator ([`KOperator`]).
        op: Option<StoredOp<f64>>,
        /// The stored exchange operator in FP32 ([`GpuDfResponseOptions::k_operator_fp32`]).
        op32: Option<StoredOp<f32>>,
        /// The stored exchange operator was built, or declined for good.
        op_decided: bool,
        /// The device memory granted to the stored operator and the device elements its chunks
        /// may take within it ([`GpuDfResponse::grant_k_operator`]).
        op_grant: Option<(crate::pool::Grant, usize)>,
        /// The GEMMs: FP64 and FP32, and the exchange's in `k_arith`.
        gemm: Gemm,
        /// The arithmetic of the exchange's FP64 GEMMs ([`cublas`] of the response's).
        k_arith: Arith,
        kernels: Kernels,
        opts: GpuDfResponseOptions,
        resolved: GpuDfResponseResolved,
        pub setup_times: GpuDfResponseSetupTimes,
        pub last: GpuDfResponseCallTimes,
    }

    /// The arithmetic of the exchange's FP64 GEMMs (single precision uses SGEMMs, the Ozaki
    /// scheme its own products).
    fn cublas(m: RespMath) -> Arith {
        match m {
            RespMath::Native | RespMath::Single | RespMath::Ozaki { .. } => Arith::Fp64,
            RespMath::Emulated { bits } => Arith::Fp64Emulated {
                bits: bits.clamp(0, u8::MAX as i32) as u8,
            },
        }
    }

    /// Converts `len` values of `x` (from element `xo`) to single precision into `y` (from
    /// `yo`).
    #[allow(clippy::too_many_arguments)]
    fn to_f32(
        gpu: &Gpu,
        k: &Kernel,
        x: &DeviceBuf<f64>,
        xo: usize,
        y: &mut DeviceBuf<f32>,
        yo: usize,
        len: usize,
    ) -> Result<(), GpuError> {
        if xo + len > x.len() || yo + len > y.len() {
            return Err(GpuError::Contract(
                "response J/K: conversion outside its buffers".into(),
            ));
        }
        // SAFETY: matches resp_to_f32; len values in and out (checked).
        unsafe {
            gpu.launch(
                k,
                Launch::linear(len, 256),
                &[
                    Arg::I64(len as i64),
                    Arg::from(x.at(xo)),
                    Arg::from(y.at(yo)),
                ],
            )
        }
    }

    /// Batched Y_Q = P_Q C (P_Q the nq unpacked planes of `a`) into Q-major `yp` at function q0.
    #[allow(clippy::too_many_arguments)]
    fn half_transform(
        gemm: &Gemm,
        n: usize,
        o: usize,
        nq: usize,
        a: &DeviceBuf<f64>,
        c: &DeviceBuf<f64>,
        yp: &mut DeviceBuf<f64>,
        q0: usize,
    ) -> Result<(), GpuError> {
        gemm.gemm_strided(
            Arith::Fp64,
            GemmDims::new(Op::N, Op::N, n, o, n).batch(nq),
            (1.0, 0.0),
            MatRef::new(a, 0, n).stride(n * n),
            MatRef::new(c, 0, n),
            MatMut::new(yp, q0 * n * o, n).stride(n * o),
        )
    }

    impl GpuDfResponse {
        /// The response J/K over the tensor of `jk` (set up for `basis` and `aux`) for the
        /// occupied orbitals `occ` (one channel: a closed shell; two: the spins of an open
        /// shell) and the exchange operator `xop` of a range-separated hybrid (None: the
        /// Coulomb exchange of HF and global hybrids, whose fraction is the `c_k` of
        /// [`GpuDfResponse::fock_occ`]). CPU work runs on the current rayon pool.
        pub fn new(
            gpu: &Gpu,
            mut jk: GpuDfJk,
            basis: &BasisSet,
            aux: &BasisSet,
            occ: &[Factor<'_>],
            xop: Option<Operator>,
            opts: &GpuDfResponseOptions,
        ) -> Result<GpuDfResponse, GpuError> {
            let t_all = Instant::now();
            let mut times = GpuDfResponseSetupTimes::default();
            let jr = jk.resolved().clone();
            let (n, naux, nx) = (jr.nbf, jr.naux, jr.nx);
            if basis.nbf != n || aux.nbf != naux {
                return Err(GpuError::Contract(
                    "response J/K: basis sets differ from the DF-J/K setup".into(),
                ));
            }
            if !(1..=2).contains(&occ.len()) {
                return Err(GpuError::Contract(format!(
                    "response J/K: {} channels (one or two)",
                    occ.len()
                )));
            }
            for f in occ {
                if f.c.len() != n * f.ncol {
                    return Err(GpuError::Contract(
                        "response J/K: occupied orbitals of the wrong size".into(),
                    ));
                }
            }
            let xop = xop.filter(|op| *op != Operator::Coulomb);
            let rs = xop.is_some();
            // memory: Y_s (Yw_s) and W, plus the setup's planes
            let ybytes: usize =
                occ.iter().map(|f| 8 * n * f.ncol * naux).sum::<usize>() * if rs { 2 } else { 1 };
            let wbytes = if rs { 8 * naux * nx } else { 0 };
            gpu.budget().check(
                "response J/K: the projected orbitals Y and W",
                ybytes + wbytes,
            )?;
            if rs && jr.storage != DfStorage::Device {
                return Err(GpuError::NotCovered(
                    "response J/K: range-separated exchange needs the tensor on the device".into(),
                ));
            }
            let mut gemm = Gemm::new(gpu)?;
            let k_arith = cublas(opts.k_math);
            gemm.prepare(gpu, k_arith)?;

            // the operator tensor W = L^-1 (Q|w|mn) on the rows of `jk` (Coulomb metric)
            let w = match &xop {
                None => None,
                Some(op) => {
                    let t = Instant::now();
                    let th = *jk.thresholds();
                    let aux3 = Aux3c::new(aux);
                    let chunks = aux3.chunks(0..aux.shells.len());
                    let lay = DfLayout::new(basis, &aux3, th.pair);
                    if lay.nx != nx {
                        return Err(GpuError::Contract(
                            "response J/K: the operator tensor's rows differ from the DF-J/K setup's"
                                .into(),
                        ));
                    }
                    let v = metric(&aux3, 0.0, &Operator::Coulomb);
                    let mut chol = gpu.upload(&v)?;
                    drop(v);
                    Solver::new(gpu)?.potrf_lower(gpu, &mut chol, naux)?;
                    let batch_rows = ((1usize << 30) / (8 * naux))
                        .max(lay.pairs.iter().map(|p| p.nab()).max().unwrap_or(1));
                    let mut hbuf = vec![0.0f64; batch_rows * naux];
                    let mut wt = gpu.alloc_zeros::<f64>(nx * naux)?;
                    for (p0, p1, r0, rows) in lay.row_batches(batch_rows) {
                        let hb = &mut hbuf[..rows * naux];
                        lay.integrals(&chunks, th.prim, op, p0, p1, hb);
                        gpu.upload_into(&mut wt, r0 * naux, hb)?;
                        gemm.trsm_lower_left(
                            Op::N,
                            (naux, rows),
                            MatRef::new(&chol, 0, naux),
                            MatMut::new(&mut wt, r0 * naux, naux),
                        )?;
                    }
                    gpu.synchronize()?;
                    times.w_tensor = t.elapsed().as_secs_f64();
                    Some(wt)
                }
            };

            // Y_s = [B_Q C_s] (and Yw_s), block by block of unpacked planes
            let t = Instant::now();
            let mut chans: Vec<Chan> = Vec::with_capacity(occ.len());
            for f in occ {
                let o = f.ncol;
                chans.push(Chan {
                    o,
                    c: gpu.upload(if o == 0 { &[0.0] } else { f.c })?,
                    yp: gpu.alloc_zeros::<f64>((n * o * naux).max(1))?,
                    ypw: if rs {
                        Some(gpu.alloc_zeros::<f64>((n * o * naux).max(1))?)
                    } else {
                        None
                    },
                });
            }
            let nq_y = ((opts.workspace / 2) / (8 * n * n).max(1)).clamp(1, 256);
            let mut a = gpu.alloc_zeros::<f64>(nq_y.min(naux) * n * n)?;
            {
                let chans = &mut chans;
                let (gemm, w) = (&gemm, &w);
                let a = &mut a;
                jk.for_each_block(gpu, |up, blk| {
                    for s0 in (0..blk.nq).step_by(nq_y) {
                        let nq = nq_y.min(blk.nq - s0);
                        up.unpack(gpu, blk, s0, nq, a)?;
                        for ch in chans.iter_mut().filter(|c| c.o > 0) {
                            half_transform(gemm, n, ch.o, nq, a, &ch.c, &mut ch.yp, blk.q0 + s0)?;
                        }
                    }
                    Ok(())
                })?;
                if let Some(wt) = w {
                    let up = jk.unpacker();
                    let wblk = TensorBlock {
                        q0: 0,
                        nq: naux,
                        buf: wt,
                        off: 0,
                        ld: naux,
                    };
                    for q0 in (0..naux).step_by(nq_y) {
                        let nq = nq_y.min(naux - q0);
                        up.unpack(gpu, &wblk, q0, nq, a)?;
                        for ch in chans.iter_mut().filter(|c| c.o > 0) {
                            let yw = ch.ypw.as_mut().unwrap();
                            half_transform(gemm, n, ch.o, nq, a, &ch.c, yw, q0)?;
                        }
                    }
                }
            }
            drop(a);
            gpu.synchronize()?;
            times.y = t.elapsed().as_secs_f64();
            let src = MODULE.source();
            let k = |e: &str| gpu.kernel(MODULE.name, &src, e);
            let kernels = Kernels {
                y_block: k("resp_y_block")?,
                to_f32: k("resp_to_f32")?,
                add_f32: k("resp_add_f32")?,
                op_fill: k("resp_op_fill")?,
                op_fill_mixed: k("resp_op_fill_mixed")?,
                acc_f32: k("resp_acc_f32")?,
            };
            times.total = t_all.elapsed().as_secs_f64();
            let mut eng = GpuDfResponse {
                resolved: GpuDfResponseResolved {
                    nbf: n,
                    naux,
                    occ: occ.iter().map(|f| f.ncol).collect(),
                    exchange_operator: xop,
                    storage: jr.storage,
                    k_math: opts.k_math,
                    y_bytes: ybytes,
                    w_bytes: wbytes,
                    k_operator_bytes: 0,
                    k_operator_host_bytes: 0,
                    k_operator_note: None,
                    k_operator_granted: 0,
                    k_operator_violations: Vec::new(),
                },
                jk,
                w,
                w_planes: None,
                op: None,
                op32: None,
                chans,
                nbf: n,
                naux,
                basis: basis.clone(),
                aux: aux.clone(),
                op_decided: false,
                op_grant: None,
                gemm,
                k_arith,
                kernels,
                opts: opts.clone(),
                setup_times: times,
                last: GpuDfResponseCallTimes::default(),
            };
            if let RespMath::Ozaki { .. } = opts.k_math {
                eng.set_k_math(gpu, opts.k_math)?;
            }
            Ok(eng)
        }

        pub fn resolved(&self) -> &GpuDfResponseResolved {
            &self.resolved
        }

        /// The DF-J/K engine (its tensor), back.
        pub fn into_jk(self) -> GpuDfJk {
            self.jk
        }

        /// Changes the arithmetic of the exchange GEMMs for the following products. Ozaki
        /// sets the DF-J/K engine's Ozaki exchange up (its tensor sliced once into planes) and
        /// is `Unsupported` when the planes do not fit (the arithmetic then stays as it was).
        pub fn set_k_math(&mut self, gpu: &Gpu, k_math: RespMath) -> Result<(), GpuError> {
            match k_math {
                RespMath::Ozaki { slices } => {
                    let before = self.jk.resolved().k_math;
                    self.jk.set_k_math(gpu, KMath::Ozaki { slices })?;
                    if self.jk.resolved().k_planes_bytes == 0 {
                        self.jk.set_k_math(gpu, before)?;
                        return Err(GpuError::NotCovered(
                            "response J/K: the Ozaki exchange multiplies the tensor's pre-sliced planes, which do not fit the device"
                                .into(),
                        ));
                    }
                    // range separation: the operator tensor's planes too, leaving the
                    // exchange workspace and a reserve free
                    self.w_planes = None;
                    if let Some(w) = self.w.take() {
                        let bytes = Sliced::bytes_for(self.naux * self.nbf + 16, self.nbf, slices);
                        let device = gpu.budget();
                        let need = bytes + self.opts.workspace;
                        let (w, r) = if !device.fits(need + self.opts.w_planes_reserve) {
                            (
                                w,
                                Err(device
                                    .over("response J/K: W's pre-sliced planes", need)
                                    .into()),
                            )
                        } else {
                            self.jk.oz_planes_for(gpu, w)
                        };
                        self.w = Some(w);
                        match r {
                            Ok(p) => self.w_planes = Some(p),
                            Err(e) => {
                                self.jk.set_k_math(gpu, before)?;
                                return Err(GpuError::NotCovered(format!(
                                    "response J/K: the Ozaki exchange's planes of the range-separated operator tensor: {e}"
                                )));
                            }
                        }
                    }
                }
                _ => {
                    self.w_planes = None;
                    if matches!(self.jk.resolved().k_math, KMath::Ozaki { .. }) {
                        self.jk.set_k_math(gpu, KMath::Native)?;
                    }
                }
            }
            self.k_arith = cublas(k_math);
            self.gemm.prepare(gpu, self.k_arith)?;
            self.opts.k_math = k_math;
            self.resolved.k_math = k_math;
            Ok(())
        }

        /// Changes the stored-operator policy for the following products ([`KOperator`]); `Off`
        /// frees an operator built before.
        pub fn set_k_operator(&mut self, k_operator: KOperator) {
            if k_operator == KOperator::Off {
                self.op = None;
                self.op32 = None;
                self.resolved.k_operator_bytes = 0;
                self.resolved.k_operator_host_bytes = 0;
            }
            if k_operator != self.opts.k_operator {
                self.op_decided = self.op.is_some() || self.op32.is_some();
                self.resolved.k_operator_note = None;
            }
            self.opts.k_operator = k_operator;
        }

        /// Caps the stored operator's device bytes (None: as many as fit), with `host` bytes of
        /// page-locked host memory for the rest, for an operator built after this call.
        pub fn set_k_operator_storage(&mut self, device_cap: Option<usize>, host: usize) {
            self.opts.k_operator_device_cap = device_cap;
            self.opts.k_operator_host = host;
        }

        /// With [`RespMath::Single`], the stored operator in FP32 (true) or the FP32 exchange
        /// through the tensor (false), for an operator built after this call.
        pub fn set_k_operator_fp32(&mut self, fp32: bool) {
            self.opts.k_operator_fp32 = fp32;
        }

        /// Whether the products run on the stored operator, building it first when the policy
        /// asks for it at `nd` perturbations (Auto: when it pays and fits; its setup time into
        /// `times`).
        fn use_operator(
            &mut self,
            gpu: &Gpu,
            nd: usize,
            tl: &mut Timeline<'_, GpuDfResponseCallTimes>,
        ) -> Result<bool, GpuError> {
            let (fp32, _) = self.op_fp32();
            if self.opts.k_math == RespMath::Single && !fp32 {
                return Ok(false);
            }
            if self.op_decided || self.opts.k_operator == KOperator::Off {
                return Ok(self.op.is_some() || self.op32.is_some());
            }
            let plan = self.k_operator_plan(gpu, nd);
            let auto = self.opts.k_operator == KOperator::Auto;
            if auto && !plan.pays {
                // not at this size of product; a later, larger one may
                return Ok(false);
            }
            // granted at the setup: its bytes are set aside, the free memory's check is moot
            let blocked = plan.blocked.filter(|_| self.op_grant.is_none());
            let res = match blocked {
                Some(w) => Err((
                    w,
                    GpuError::from(gpu.budget().over(
                        "response J/K: the stored exchange operator",
                        plan.bytes + plan.setup_bytes,
                    )),
                )),
                None => {
                    let m0 = tl.mark()?;
                    let r = if fp32 {
                        self.build_operator::<f32>(gpu)
                    } else {
                        self.build_operator::<f64>(gpu)
                    };
                    tl.close(|t| &mut t.k_operator_setup, m0)?;
                    r.map_err(|e| (e.to_string(), e))
                }
            };
            self.op_decided = true;
            match res {
                Ok(()) => {
                    let (bytes, host_bytes) = match (&self.op, &self.op32) {
                        (Some(op), _) => (op.bytes, op.host_bytes),
                        (_, Some(op)) => (op.bytes, op.host_bytes),
                        _ => (0, 0),
                    };
                    self.resolved.k_operator_bytes = bytes;
                    self.resolved.k_operator_host_bytes = host_bytes;
                    self.resolved.k_operator_note = None;
                    Ok(true)
                }
                Err((why, e)) => {
                    self.op = None;
                    self.op32 = None;
                    if auto {
                        let what = if fp32 { " in FP32" } else { "" };
                        let mut note = format!("the stored exchange operator{what}: {why}");
                        match self.opts.k_operator_fallback {
                            Some(f) if f != self.opts.k_math => {
                                self.set_k_math(gpu, f)?;
                                note.push_str(&format!("; the exchange in {f:?} instead"));
                                if self.op_fp32().0 {
                                    // the FP32 operator next (else the FP32 tensor exchange)
                                    self.op_decided = false;
                                    let on = self.use_operator(gpu, nd, tl)?;
                                    let then = self.resolved.k_operator_note.take();
                                    self.resolved.k_operator_note = Some(match then {
                                        Some(t) => format!("{note}; {t}"),
                                        None => format!("{note}, on the FP32 operator"),
                                    });
                                    return Ok(on);
                                }
                            }
                            _ => {}
                        }
                        self.resolved.k_operator_note = Some(note);
                        Ok(false)
                    } else {
                        Err(e)
                    }
                }
            }
        }

        /// The device elements the stored operator's chunks may take now: the free memory
        /// less its setup's temporaries and the reserve.
        fn op_budget(&self, gpu: &Gpu, temp: usize, esz: usize) -> usize {
            gpu.budget()
                .free()
                .saturating_sub(temp + self.opts.k_operator_reserve)
                / esz
        }

        /// Bytes of page-locked host memory the stored operator may take (FP64 or FP32).
        fn op_host(&self) -> usize {
            if self.op_fp32().0 {
                self.opts
                    .k_operator_host32
                    .unwrap_or(self.opts.k_operator_host)
            } else {
                self.opts.k_operator_host
            }
        }

        /// Whether the stored operator runs in FP32 ([`RespMath::Single`] with
        /// `k_operator_fp32`), and its element size.
        fn op_fp32(&self) -> (bool, usize) {
            let fp32 = self.opts.k_math == RespMath::Single && self.opts.k_operator_fp32;
            (fp32, if fp32 { 4 } else { 8 })
        }

        /// What the stored exchange operator would cost for products of `nd` perturbations,
        /// and whether it would be built now: its chunks on the device as far as they fit in
        /// the free memory less the setup's temporaries and the reserve, the rest in page-locked
        /// host memory up to `k_operator_host` bytes.
        /// Grants the stored exchange operator in FP64 for products of `nd` perturbations now,
        /// at the setup (the memory plan's grant, package PLAN): its device chunks, staging
        /// buffers and setup temporaries, laid out against the memory free now beside
        /// `k_operator_reserve`, are set aside in the device pool, and the operator is built at
        /// the first product within them ([`KOperator::On`]) whatever is allocated in between.
        /// Returns the bytes granted; an out-of-memory error when they do not fit, and an
        /// unsupported one when its host part exceeds `k_operator_host` or the tensor is not on
        /// the device (the operator stays off then).
        pub fn grant_k_operator(&mut self, gpu: &Gpu, nd: usize) -> Result<usize, GpuError> {
            let plan = self.k_operator_plan(gpu, nd);
            if let Some(why) = plan.blocked {
                return Err(GpuError::NotCovered(format!(
                    "response J/K: the stored exchange operator: {why}"
                )));
            }
            let (n, naux) = (self.nbf, self.naux);
            let nx = self.jk.resolved().nx;
            let occ: Vec<usize> = self.chans.iter().map(|c| c.o).collect();
            let (.., temp) = op_estimate(n, naux, nx, &occ, self.w.is_some(), 1, false);
            let budget = self.op_budget(gpu, temp, 8);
            let cap = self.opts.k_operator_device_cap.map(|c| c / 8);
            let (_, ndev, _, nstage) = op_layout(n, &occ, budget, cap);
            let bytes = 8 * (ndev + OP_STAGING * nstage) + temp;
            let grant = gpu.grant(bytes, "response J/K: the stored exchange operator")?;
            self.op_grant = Some((grant, budget));
            self.opts.k_operator = KOperator::On;
            self.resolved.k_operator_granted = bytes;
            Ok(bytes)
        }

        pub fn k_operator_plan(&self, gpu: &Gpu, nd: usize) -> KOperatorPlan {
            let occ: Vec<usize> = self.chans.iter().map(|c| c.o).collect();
            let nx = self.jk.resolved().nx;
            let (fp32, esz) = self.op_fp32();
            let (tensor, setup, call, bytes, temp) =
                op_estimate(self.nbf, self.naux, nx, &occ, self.w.is_some(), nd, fp32);
            let budget = self.op_budget(gpu, temp, esz);
            let cap = self.opts.k_operator_device_cap.map(|c| c / esz);
            let (_, _, host, staging) = op_layout(self.nbf, &occ, budget, cap);
            let free = gpu.budget().free();
            let host_bytes = esz * host;
            let host_ok = self.op_host();
            let blocked = if self.jk.resolved().storage != DfStorage::Device {
                Some("the tensor is not on the device".to_string())
            } else if host_bytes > host_ok {
                let fit = format!(
                    "{:.1} GiB (and {:.1} GiB during its setup) do not fit beside the {:.1} GiB reserve in the {:.1} GiB free",
                    bytes as f64 / GIB,
                    temp as f64 / GIB,
                    self.opts.k_operator_reserve as f64 / GIB,
                    free as f64 / GIB
                );
                Some(if host_ok == 0 {
                    fit
                } else {
                    format!(
                        "{fit}, and the {:.1} GiB beyond exceed the {:.1} GiB of host memory allowed",
                        host_bytes as f64 / GIB,
                        host_ok as f64 / GIB
                    )
                })
            } else {
                None
            };
            KOperatorPlan {
                pays: setup + call <= 2.0 * tensor,
                bytes,
                setup_bytes: temp,
                host_bytes,
                staging_bytes: esz * OP_STAGING * staging,
                blocked,
                setup_flops: setup,
                product_flops: call,
                tensor_flops: tensor,
            }
        }

        /// Builds every channel's stored operator ([`KOperator`]) in the element `T` (FP64, or
        /// FP32 with `k_operator_fp32`, [`OpElement`]) from the tensor (on the device), Y and X,
        /// chunk by chunk ([`build_channel`]); the chunks that do not fit the device are built
        /// in staging buffers and copied to page-locked host memory on a second stream,
        /// overlapped with the next chunks.
        fn build_operator<T: OpElement>(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
            let (n, naux) = (self.nbf, self.naux);
            let nx = self.jk.resolved().nx;
            let occ: Vec<usize> = self.chans.iter().map(|c| c.o).collect();
            let (.., temp) = op_estimate(n, naux, nx, &occ, self.w.is_some(), 1, T::FP32);
            // granted at the setup (FP64, `grant_k_operator`): the grant's layout, each
            // allocation drawn from it
            let grant = self.op_grant.take();
            let budget = match &grant {
                Some((_, b)) => *b,
                None => self.op_budget(gpu, temp, T::BYTES),
            };
            let draw = |bytes: usize| {
                if let Some((g, _)) = &grant {
                    g.draw(bytes);
                }
            };
            let cap = self.opts.k_operator_device_cap.map(|c| c / T::BYTES);
            let (chunks, ndev, nhost, nstage) = op_layout(n, &occ, budget, cap);
            if T::BYTES * nhost > self.op_host() {
                return Err(gpu.budget().over(T::WHAT, T::BYTES * (ndev + nhost)).into());
            }
            let map = self.pair_map(gpu)?;
            draw(T::BYTES * ndev);
            let mut op = StoredOp::<T> {
                chunks,
                dev: if ndev > 0 {
                    Some(T::alloc(gpu, ndev)?)
                } else {
                    None
                },
                host: None,
                bytes: T::BYTES * (ndev + nhost),
                host_bytes: T::BYTES * nhost,
            };
            if nhost > 0 {
                draw(T::BYTES * OP_STAGING * nstage);
                op.host = Some(T::streamed(gpu, nhost, nstage)?);
            }
            // the setup's temporaries
            draw(temp);
            if let Some((g, _)) = &grant {
                self.resolved.k_operator_violations = g.violations();
            }
            drop(grant);
            let Self {
                jk,
                w,
                chans,
                gemm,
                k_arith,
                kernels,
                ..
            } = self;
            let b = OpBuild {
                gpu,
                gemm,
                arith: *k_arith,
                kernels,
                dims: (n, naux, nx),
                map: &map,
            };
            let mut t = 0;
            let all = op.chunks.clone();
            jk.for_each_block(gpu, |_, blk| {
                // device storage: one block of every function, rows of naux
                if blk.nq != naux || blk.off != 0 || blk.ld != naux {
                    return Err(GpuError::Contract(
                        "response J/K: the operator needs the whole tensor in one block".into(),
                    ));
                }
                let s = T::view(gpu, &kernels.to_f32, blk.buf, naux * nx)?;
                let wt = match w.as_ref() {
                    Some(w) => Some(T::view(gpu, &kernels.to_f32, w, naux * nx)?),
                    None => None,
                };
                for (ch, chunks) in chans.iter().zip(&all) {
                    if ch.o > 0 {
                        build_channel(&b, ch, (&*s, wt.as_deref()), (chunks, &mut op), &mut t)?;
                    }
                }
                Ok(())
            })?;
            if let Some(h) = &op.host {
                h.join(gpu)?;
            }
            gpu.synchronize()?;
            *T::stored(&mut self.op, &mut self.op32) = Some(op);
            Ok(())
        }

        /// out_s -= c_k Op_s vec(A_s) on the stored operators in the element `T`: per chunk the
        /// product of its rows and, below the diagonal block, of its transpose ([`OpElement`]'s
        /// products), in a fixed order whatever the chunks' storage (so bitwise the same with
        /// any split between device and host). The host chunks are copied to the staging
        /// buffers on the copy stream, up to [`OP_STAGING`] ahead. FP64 products are always
        /// native (the setup follows the response's arithmetic): cuBLAS's emulation re-slices
        /// the operator in every product, which made it slower than FP64 below ~110
        /// perturbations (ts61/SVP: 1.57 against 0.46 s at 24, 1.61 against 2.69 s at 183), and
        /// a CPKS's later products have few.
        fn exchange_op<T: OpElement>(
            &mut self,
            gpu: &Gpu,
            acat: &[DeviceBuf<f64>],
            out: &mut [DeviceBuf<f64>],
            nd: usize,
            c_k: f64,
        ) -> Result<(), GpuError> {
            let Self {
                op,
                op32,
                chans,
                gemm,
                kernels,
                nbf,
                ..
            } = self;
            let n = *nbf;
            let op = T::stored(op, op32)
                .as_mut()
                .ok_or_else(|| GpuError::Contract(format!("{}: not built", T::WHAT)))?;
            // the factors in the element's type
            let a: Vec<Held<'_, T>> = acat
                .iter()
                .zip(chans.iter())
                .map(|(x, ch)| T::view(gpu, &kernels.to_f32, x, n * ch.o * nd))
                .collect::<Result<_, GpuError>>()?;
            let all = || op.chunks.iter().flatten();
            let rmax = all().map(|c| c.rows(n)).max().unwrap_or(1);
            let cmax = all().map(|c| (c.i1 - c.i0) * n).max().unwrap_or(1);
            // the FP32 products' results
            let mut t1 = T::FP32.then(|| T::alloc(gpu, rmax * nd)).transpose()?;
            let mut t2 = T::FP32.then(|| T::alloc(gpu, cmax * nd)).transpose()?;
            // the host chunks in processing order
            let hosted: Vec<&OpChunk> = all().filter(|c| matches!(c.at, Loc::Host(_))).collect();
            let ns = op.host.as_ref().map_or(1, |h| h.slots().max(1));
            // the first host chunks into the staging buffers (the next one after each product)
            let fetch = |host: &mut StreamedStore<_>, t: usize| -> Result<(), GpuError> {
                let c = hosted[t];
                let Loc::Host(off) = c.at else {
                    return Err(GpuError::Contract("response J/K: not a host chunk".into()));
                };
                host.load(gpu, t % ns, off, c.len(n))
            };
            if !hosted.is_empty() {
                let host = host_of(&mut op.host)?;
                for t in 0..hosted.len().min(ns) {
                    fetch(host, t)?;
                }
            }
            let p = OpProduct {
                gpu,
                gemm,
                acc: &kernels.acc_f32,
                nd,
                c_k,
            };
            let mut t = 0;
            for (s, chunks) in op.chunks.iter().enumerate() {
                let no = n * chans[s].o;
                for c in chunks {
                    let rows = c.rows(n);
                    let cols = (c.i1 - c.i0) * n;
                    let (buf, off) = match c.at {
                        Loc::Device(off) => (
                            op.dev.as_ref().ok_or_else(|| {
                                GpuError::Contract("response J/K: no device store".into())
                            })?,
                            off,
                        ),
                        Loc::Host(_) => (&*host_of(&mut op.host)?.slot(gpu, t % ns)?, 0),
                    };
                    T::product(
                        &p,
                        t1.as_mut(),
                        GemmDims::new(Op::N, Op::N, rows, nd, cols),
                        (
                            MatRef::new(buf, off, rows),
                            MatRef::new(&a[s], c.i0 * n, no),
                        ),
                        (&mut out[s], c.j0 * n, no),
                    )?;
                    if c.below() {
                        T::product(
                            &p,
                            t2.as_mut(),
                            GemmDims::new(Op::T, Op::N, cols, nd, rows),
                            (
                                MatRef::new(buf, off, rows),
                                MatRef::new(&a[s], c.j0 * n, no),
                            ),
                            (&mut out[s], c.i0 * n, no),
                        )?;
                    }
                    if let Loc::Host(_) = c.at {
                        let host = host_of(&mut op.host)?;
                        host.release(gpu, t % ns)?;
                        if t + ns < hosted.len() {
                            fetch(host, t + ns)?;
                        }
                        t += 1;
                    }
                }
            }
            Ok(())
        }

        /// The stored row of every function pair, either orientation (-1: dropped), as the
        /// tensor's unpack places them, on the device (n x n).
        fn pair_map(&self, gpu: &Gpu) -> Result<DeviceBuf<i32>, GpuError> {
            let n = self.nbf;
            let nx = self.jk.resolved().nx;
            let pair = self.jk.thresholds().pair;
            let lay = DfLayout::new(&self.basis, &Aux3c::new(&self.aux), pair);
            if lay.nx != nx {
                return Err(GpuError::Contract(
                    "response J/K: the operator's pair rows differ from the DF-J/K setup's".into(),
                ));
            }
            let mut map = vec![-1i32; n * n];
            for x in 0..nx {
                let (m, l) = (lay.row_m[x] as usize, lay.row_n[x] as usize);
                map[m * n + l] = x as i32;
                if lay.row_kind[x] == ROW_OFFDIAG {
                    map[l * n + m] = x as i32;
                }
            }
            gpu.upload(&map)
        }

        /// G_{s,d} = (c_j J\[sum_s' dD_{s',d}\] - c_k K\[dD_{s,d}\]) C_s for the factors `a[s]`
        /// (n x (nd o_s) column-major: A_{s,d} at columns d o_s..) of `nd` perturbations, in
        /// the same layout; K of the setup's exchange operator (wK) when it has one, and only
        /// when `c_k` is not 0.
        pub fn fock_occ(
            &mut self,
            gpu: &Gpu,
            a: &[&[f64]],
            nd: usize,
            c_j: f64,
            c_k: f64,
        ) -> Result<Vec<Vec<f64>>, GpuError> {
            let (n, naux) = (self.nbf, self.naux);
            let nch = self.chans.len();
            if a.len() != nch {
                return Err(GpuError::Contract(format!(
                    "response J/K: {} lists of factors for {nch} channels",
                    a.len()
                )));
            }
            for (s, ch) in self.chans.iter().enumerate() {
                if a[s].len() != n * nd * ch.o {
                    return Err(GpuError::Contract(format!(
                        "response J/K: channel {s} has {} factor values, not {n} x {nd} x {}",
                        a[s].len(),
                        ch.o
                    )));
                }
            }
            let mut times = GpuDfResponseCallTimes::default();
            if nd == 0 {
                self.last = times;
                return Ok(self.chans.iter().map(|_| Vec::new()).collect());
            }
            // the call's phases by events on the stream, added up once its results are downloaded
            let mut tl = Timeline::<GpuDfResponseCallTimes>::new(gpu);
            let m0 = tl.mark()?;
            let acat: Vec<DeviceBuf<f64>> = (0..nch)
                .map(|s| gpu.upload(if a[s].is_empty() { &[0.0] } else { a[s] }))
                .collect::<Result<_, _>>()?;
            let m_up = tl.close(|t| &mut t.upload, m0)?;

            // J: gamma = 2 sum_s Yp_s^T Acat_s (naux x nd), G_s = c_j Yp_s gamma
            let mut out: Vec<DeviceBuf<f64>> = self
                .chans
                .iter()
                .map(|ch| gpu.alloc_zeros::<f64>((n * ch.o * nd).max(1)))
                .collect::<Result<_, _>>()?;
            if c_j != 0.0 {
                let mut gamma = gpu.alloc_zeros::<f64>(naux * nd)?;
                let mut first = true;
                for (s, ch) in self.chans.iter().enumerate() {
                    if ch.o == 0 {
                        continue;
                    }
                    let no = n * ch.o;
                    self.gemm.gemm(
                        Arith::Fp64,
                        GemmDims::new(Op::T, Op::N, naux, nd, no),
                        (2.0, if first { 0.0 } else { 1.0 }),
                        MatRef::new(&ch.yp, 0, no),
                        MatRef::new(&acat[s], 0, no),
                        MatMut::new(&mut gamma, 0, naux),
                    )?;
                    first = false;
                }
                for (s, ch) in self.chans.iter().enumerate() {
                    if ch.o == 0 {
                        continue;
                    }
                    let no = n * ch.o;
                    self.gemm.gemm(
                        Arith::Fp64,
                        GemmDims::new(Op::N, Op::N, no, nd, naux),
                        (c_j, 0.0),
                        MatRef::new(&ch.yp, 0, no),
                        MatRef::new(&gamma, 0, naux),
                        MatMut::new(&mut out[s], 0, no),
                    )?;
                }
            }
            tl.close(|t| &mut t.j, m_up)?;

            let omax = self.chans.iter().map(|c| c.o).max().unwrap_or(0);
            if c_k != 0.0 && omax > 0 {
                if self.use_operator(gpu, nd, &mut tl)? {
                    let s = tl.mark()?;
                    if self.op32.is_some() {
                        self.exchange_op::<f32>(gpu, &acat, &mut out, nd, c_k)?;
                    } else {
                        self.exchange_op::<f64>(gpu, &acat, &mut out, nd, c_k)?;
                    }
                    tl.close(|t| &mut t.k_operator, s)?;
                } else {
                    (times.k_group, times.k_block) =
                        self.exchange(gpu, &acat, &mut out, nd, c_k, omax, &mut tl)?;
                }
            }

            let md = tl.mark()?;
            let res: Vec<Vec<f64>> = out
                .iter()
                .zip(&self.chans)
                .map(|(o, ch)| {
                    if ch.o == 0 {
                        Ok(Vec::new())
                    } else {
                        gpu.download(o)
                    }
                })
                .collect::<Result<_, _>>()?;
            let me = tl.close(|t| &mut t.download, md)?;
            tl.span(|t| &mut t.total, m0, me);
            tl.add_to(&mut times)?;
            times.k_contract = times.k_xy + times.k_wst + times.k_batched;
            self.last = times;
            Ok(res)
        }

        /// The exchange part of [`GpuDfResponse::fock_occ`]: out_s -= c_k K\[dD_s\] C_s, its
        /// phases on the call's timeline; the perturbations per group and auxiliary functions
        /// per block it ran with. Ozaki: the DF-J/K engine's Ozaki state is taken out for the
        /// call and each channel's factors sliced once.
        #[allow(clippy::too_many_arguments)]
        fn exchange(
            &mut self,
            gpu: &Gpu,
            acat: &[DeviceBuf<f64>],
            out: &mut [DeviceBuf<f64>],
            nd: usize,
            c_k: f64,
            omax: usize,
            tl: &mut Timeline<'_, GpuDfResponseCallTimes>,
        ) -> Result<(usize, usize), GpuError> {
            let mut work = OzakiWork::default();
            if !matches!(self.opts.k_math, RespMath::Ozaki { .. }) {
                return self.exchange_inner(gpu, acat, out, (nd, c_k, omax), tl, None, &mut work);
            }
            let t = self.jk.take_ozaki().ok_or_else(|| {
                GpuError::Contract("response J/K: the Ozaki exchange is not set up".into())
            })?;
            let wp = self.w_planes.take();
            let n = self.nbf;
            let res = (|| -> Result<(usize, usize), GpuError> {
                let mut acs = Vec::with_capacity(acat.len());
                for (x, ch) in acat.iter().zip(&self.chans) {
                    let rows = nd * ch.o;
                    let mut sl = Sliced::new(gpu, rows.max(1), n, t.slices(), Order::Descending)?;
                    if rows > 0 {
                        let src = Vectors {
                            buf: x,
                            off: 0,
                            ld: n,
                            contiguous: true,
                        };
                        t.ozaki().slice(gpu, src, &mut sl, &mut work)?;
                    }
                    acs.push(sl);
                }
                self.exchange_inner(
                    gpu,
                    acat,
                    out,
                    (nd, c_k, omax),
                    tl,
                    Some((&t, &acs, wp.as_ref())),
                    &mut work,
                )
            })();
            self.jk.put_ozaki(t);
            self.w_planes = wp;
            res
        }

        /// [`GpuDfResponse::exchange`] with the Ozaki state and the sliced factors, if any.
        #[allow(clippy::too_many_arguments)]
        fn exchange_inner(
            &mut self,
            gpu: &Gpu,
            acat: &[DeviceBuf<f64>],
            out: &mut [DeviceBuf<f64>],
            (nd, c_k, omax): (usize, f64, usize),
            tl: &mut Timeline<'_, GpuDfResponseCallTimes>,
            oz: Option<(&OzTaken, &[Sliced], Option<&Sliced>)>,
            work: &mut OzakiWork,
        ) -> Result<(usize, usize), GpuError> {
            let (n, naux) = (self.nbf, self.naux);
            let ozaki = oz.is_some();
            // the operator tensor's planes held: no FP64 planes of W either
            let oz_w = oz.is_some_and(|o| o.2.is_some());
            let rs = self.w.is_some();
            let single = self.opts.k_math == RespMath::Single;
            let (group, mut nq) =
                crate::response::exchange_plan(n, omax, nd, naux, rs, self.opts.workspace, 8, 512);
            if self.opts.k_block > 0 {
                nq = self.opts.k_block.min(naux);
            } else if single {
                // FP32 sums over at most 64 auxiliary functions before the FP64 accumulation
                nq = nq.min(64);
            }
            let wmax = group * omax;
            let r = if rs { 2 } else { 1 };
            // workspaces: planes (B and W), Z, the stacked A^T Y (Wst), the block's Y and X;
            // single precision: the f32 copies (planes, Y, factors, C) and a block accumulator
            // the tensor's FP64 planes (Ozaki: its pre-sliced planes instead)
            let mut a = gpu.alloc_zeros::<f64>(if ozaki { 1 } else { nq * n * n })?;
            // Ozaki GEMM1's columns per chunk: its INT32 partial sums within Z's bytes
            let oz_cols = (2 * wmax / 7).max(16);
            let mut aw = if rs && !oz_w {
                Some(gpu.alloc_zeros::<f64>(nq * n * n)?)
            } else {
                None
            };
            let mut yb = gpu.alloc_zeros::<f64>(r * n * omax * nq)?;
            // FP64 Z, Wst and X (single precision has its own)
            let f64_len = |len: usize| if single { 1 } else { len };
            let mut z = gpu.alloc_zeros::<f64>(f64_len(nq * n * wmax))?;
            let mut wst = gpu.alloc_zeros::<f64>(f64_len(wmax * nq * omax))?;
            let mut xs = gpu.alloc_zeros::<f64>(f64_len(r * omax * omax * nq))?;
            let mut s32 = if single {
                let conv = |x: &DeviceBuf<f64>, len: usize| -> Result<DeviceBuf<f32>, GpuError> {
                    let mut y = gpu.alloc_zeros::<f32>(len.max(1))?;
                    to_f32(gpu, &self.kernels.to_f32, x, 0, &mut y, 0, len)?;
                    Ok(y)
                };
                Some(Single {
                    a: gpu.alloc_zeros::<f32>(nq * n * n)?,
                    aw: gpu.alloc_zeros::<f32>(if rs { nq * n * n } else { 1 })?,
                    z: gpu.alloc_zeros::<f32>(nq * n * wmax)?,
                    wst: gpu.alloc_zeros::<f32>(wmax * nq * omax)?,
                    yb: gpu.alloc_zeros::<f32>(r * n * omax * nq)?,
                    xs: gpu.alloc_zeros::<f32>(r * omax * omax * nq)?,
                    acc: gpu.alloc_zeros::<f32>(n * wmax)?,
                    c: self
                        .chans
                        .iter()
                        .map(|ch| conv(&ch.c, n * ch.o))
                        .collect::<Result<_, _>>()?,
                    acat: acat
                        .iter()
                        .zip(&self.chans)
                        .map(|(x, ch)| conv(x, n * nd * ch.o))
                        .collect::<Result<_, _>>()?,
                })
            } else {
                None
            };
            // f: the coefficient of each exchange term (wK: one half of four terms)
            let f = if rs { -0.5 * c_k } else { -c_k };
            let groups: Vec<(usize, usize)> = (0..nd)
                .step_by(group)
                .map(|d0| (d0, group.min(nd - d0)))
                .collect();
            let host_storage = self.jk.resolved().storage == DfStorage::Host;
            let mut last_end: Option<usize> = None;
            // disjoint borrows of the engine for the block visitor
            let Self {
                jk,
                w,
                chans,
                gemm,
                k_arith,
                kernels,
                ..
            } = self;
            // the block's Y (or Yw) in (i, q) order into yb at `off`
            let this_y = |gpu: &Gpu,
                          yp: &DeviceBuf<f64>,
                          o: usize,
                          q0: usize,
                          nq: usize,
                          yb: &mut DeviceBuf<f64>,
                          off: usize|
             -> Result<(), GpuError> {
                let len = n * o * nq;
                if (q0 + nq) * n * o > yp.len() || off + len > yb.len() {
                    return Err(GpuError::Contract(
                        "response J/K: Y block outside its buffer".into(),
                    ));
                }
                // SAFETY: matches resp_y_block; reads functions q0..q0 + nq of yp, writes
                // n o nq values of yb at `off` (both checked above).
                unsafe {
                    gpu.launch(
                        &kernels.y_block,
                        Launch::linear(len, 256),
                        &[
                            Arg::I64(n as i64),
                            Arg::I32(o as i32),
                            Arg::I32(nq as i32),
                            Arg::I64(q0 as i64),
                            Arg::buf(yp),
                            Arg::from(yb.at(off)),
                        ],
                    )
                }
            };
            jk.for_each_block(gpu, |up, blk| {
                if host_storage {
                    // the stream copy was queued just before this call
                    let s = tl.mark()?;
                    if let Some(e) = last_end {
                        tl.span(|t| &mut t.stream, e, s);
                    }
                }
                for s0 in (0..blk.nq).step_by(nq) {
                    let nqb = nq.min(blk.nq - s0);
                    let q0 = blk.q0 + s0;
                    let s = tl.mark()?;
                    if !ozaki {
                        up.unpack(gpu, blk, s0, nqb, &mut a)?;
                    }
                    if let (Some(wt), Some(aw)) = (w.as_ref(), aw.as_mut()) {
                        let wblk = TensorBlock {
                            q0: 0,
                            nq: naux,
                            buf: wt,
                            off: 0,
                            ld: naux,
                        };
                        if !oz_w {
                            up.unpack(gpu, &wblk, q0, nqb, aw)?;
                        }
                    }
                    if let Some(sp) = s32.as_mut() {
                        to_f32(gpu, &kernels.to_f32, &a, 0, &mut sp.a, 0, nqb * n * n)?;
                        if let Some(aw) = aw.as_ref() {
                            to_f32(gpu, &kernels.to_f32, aw, 0, &mut sp.aw, 0, nqb * n * n)?;
                        }
                    }
                    tl.close(|t| &mut t.k_unpack, s)?;
                    for (ci, ch) in chans.iter().enumerate() {
                        let o = ch.o;
                        if o == 0 {
                            continue;
                        }
                        let ono = o * nqb;
                        // the block's Y (and Yw) in (i, q) order and X = Y^T C (Xw = Yw^T C)
                        let t0 = tl.mark()?;
                        this_y(gpu, &ch.yp, o, q0, nqb, &mut yb, 0)?;
                        if let Some(ypw) = ch.ypw.as_ref() {
                            this_y(gpu, ypw, o, q0, nqb, &mut yb, n * ono)?;
                        }
                        for t in 0..r {
                            match s32.as_mut() {
                                None => gemm.gemm(
                                    *k_arith,
                                    GemmDims::new(Op::T, Op::N, ono, o, n),
                                    (1.0, 0.0),
                                    MatRef::new(&yb, t * n * ono, n),
                                    MatRef::new(&ch.c, 0, n),
                                    MatMut::new(&mut xs, t * ono * o, ono),
                                )?,
                                Some(sp) => {
                                    to_f32(
                                        gpu,
                                        &kernels.to_f32,
                                        &yb,
                                        t * n * ono,
                                        &mut sp.yb,
                                        t * n * ono,
                                        n * ono,
                                    )?;
                                    gemm.gemm(
                                        Arith::Fp32,
                                        GemmDims::new(Op::T, Op::N, ono, o, n),
                                        (1.0, 0.0),
                                        MatRef::new(&sp.yb, t * n * ono, n),
                                        MatRef::new(&sp.c[ci], 0, n),
                                        MatMut::new(&mut sp.xs, t * ono * o, ono),
                                    )?
                                }
                            }
                        }
                        tl.close(|t| &mut t.k_xy, t0)?;
                        // offsets of Y, Yw, X, Xw in yb and xs
                        let (y_off, yw_off) = (0, if rs { n * ono } else { 0 });
                        let (x_off, xw_off) = (0, if rs { ono * o } else { 0 });
                        // passes: (planes: 0 B, 1 W; X of the other tensor, Y of the other
                        // tensor). B: Z X (Xw for wK) + Y(w) Wst; W: Zw X + Y Wst_w
                        let passes: Vec<(usize, usize, usize)> = if rs {
                            vec![(0, xw_off, yw_off), (1, x_off, y_off)]
                        } else {
                            vec![(0, x_off, y_off)]
                        };
                        for &(d0, ndg) in &groups {
                            let g = Group {
                                n,
                                o,
                                nqb,
                                ndg,
                                a_off: d0 * n * o,
                                f,
                            };
                            for (pi, &(planes, xo, yo)) in passes.iter().enumerate() {
                                let g0 = tl.mark()?;
                                match s32.as_mut() {
                                    None => {
                                        match (planes, oz) {
                                            (0, Some((t, acs, _))) => g.gemm1_ozaki(
                                                gpu,
                                                (t.ozaki(), t.planes()),
                                                &acs[ci],
                                                q0,
                                                &mut z,
                                                oz_cols,
                                                &mut *work,
                                            )?,
                                            (1, Some((t, acs, Some(wp)))) => g.gemm1_ozaki(
                                                gpu,
                                                (t.ozaki(), Some(wp)),
                                                &acs[ci],
                                                q0,
                                                &mut z,
                                                oz_cols,
                                                &mut *work,
                                            )?,
                                            _ => {
                                                let p = if planes == 0 {
                                                    &a
                                                } else {
                                                    aw.as_ref().unwrap()
                                                };
                                                g.gemm1_f64(
                                                    (gemm, *k_arith),
                                                    p,
                                                    &acat[ci],
                                                    &mut z,
                                                )?;
                                            }
                                        }
                                        let g1 = tl.close(|t| &mut t.k_gemm1, g0)?;
                                        g.wst_f64((gemm, *k_arith), &ch.c, &z, &mut wst)?;
                                        let g2 = tl.close(|t| &mut t.k_wst, g1)?;
                                        g.batched_f64(
                                            (gemm, *k_arith),
                                            (&z, &wst),
                                            (&xs, xo),
                                            (&yb, yo),
                                            &mut out[ci],
                                        )?;
                                        tl.close(|t| &mut t.k_batched, g2)?;
                                    }
                                    Some(sp) => {
                                        let Single {
                                            a: a32,
                                            aw: aw32,
                                            z: z32,
                                            wst: wst32,
                                            yb: yb32,
                                            xs: xs32,
                                            acc,
                                            c: c32,
                                            acat: acat32,
                                        } = sp;
                                        let p = if planes == 0 { &*a32 } else { &*aw32 };
                                        g.gemm1_f32(gemm, p, &acat32[ci], z32)?;
                                        let g1 = tl.close(|t| &mut t.k_gemm1, g0)?;
                                        g.contract_f32(
                                            gemm,
                                            &c32[ci],
                                            (&*z32, wst32),
                                            (&*xs32, xo),
                                            (&*yb32, yo),
                                            acc,
                                            pi > 0,
                                        )?;
                                        tl.close(|t| &mut t.k_batched, g1)?;
                                    }
                                }
                            }
                            if let Some(sp) = s32.as_ref() {
                                // the block's FP32 sum into the FP64 result
                                let len = n * ndg * o;
                                if d0 * n * o + len > out[ci].len() || len > sp.acc.len() {
                                    return Err(GpuError::Contract(
                                        "response J/K: accumulation outside".into(),
                                    ));
                                }
                                // SAFETY: matches resp_add_f32; len values in (acc) and out
                                // (the group's columns of out, checked).
                                unsafe {
                                    gpu.launch(
                                        &kernels.add_f32,
                                        Launch::linear(len, 256),
                                        &[
                                            Arg::I64(len as i64),
                                            Arg::buf(&sp.acc),
                                            Arg::from(out[ci].at(d0 * n * o)),
                                        ],
                                    )
                                }?;
                            }
                        }
                    }
                }
                last_end = Some(tl.mark()?);
                Ok(())
            })?;
            Ok((group, nq))
        }
    }

    const GIB: f64 = (1u64 << 30) as f64;

    /// A stored operator's host store (an error when the operator has none: a host chunk without
    /// it is a bug).
    fn host_of<T>(h: &mut Option<StreamedStore<T>>) -> Result<&mut StreamedStore<T>, GpuError> {
        h.as_mut()
            .ok_or_else(|| GpuError::Contract("response J/K: no host store".into()))
    }

    /// What the stored operator's build ([`build_channel`]) works with: the device, the GEMMs
    /// and the response's FP64 arithmetic, the kernels, (n, naux, nx) and the pair map.
    #[derive(Clone, Copy)]
    struct OpBuild<'a> {
        gpu: &'a Gpu,
        gemm: &'a Gemm,
        arith: Arith,
        kernels: &'a Kernels,
        dims: (usize, usize, usize),
        map: &'a DeviceBuf<i32>,
    }

    /// What the products on the stored operator work with ([`OpElement::product`]).
    struct OpProduct<'a> {
        gpu: &'a Gpu,
        gemm: &'a Gemm,
        acc: &'a Kernel,
        nd: usize,
        c_k: f64,
    }

    /// Where a sum over the auxiliary functions goes, from an element: FP64 sums (the panel's
    /// pair part, the FP32 build's chunk sums), or the FP64 build's chunk itself (its Y part,
    /// which `resp_op_fill` completes in place).
    enum Sums<'b, T> {
        Fp64(&'b mut DeviceBuf<f64>, usize),
        Chunk(&'b mut DeviceBuf<T>, usize),
    }

    /// An FP64 operand in the element's type: itself, or its FP32 copy.
    enum Held<'a, T> {
        Borrowed(&'a DeviceBuf<T>),
        Owned(DeviceBuf<T>),
    }

    impl<T> std::ops::Deref for Held<'_, T> {
        type Target = DeviceBuf<T>;
        fn deref(&self) -> &DeviceBuf<T> {
            match self {
                Held::Borrowed(b) => b,
                Held::Owned(b) => b,
            }
        }
    }

    /// The element of the stored exchange operator ([`StoredOp`]): FP64, or FP32 with
    /// [`GpuDfResponseOptions::k_operator_fp32`]. The operator's build
    /// ([`GpuDfResponse::build_operator`], [`build_channel`]) and its products
    /// ([`GpuDfResponse::exchange_op`]) are one implementation over it; the element decides
    /// the arithmetic of their sums, one to one the two builds' GEMMs and kernels:
    /// - f64: the response's GEMMs (in `k_arith`) straight into their targets, over all
    ///   auxiliary functions at once: X, the pair part, the Y part into the chunk itself (which
    ///   `resp_op_fill` completes in place); the products native DGEMMs into the result;
    /// - f32: SGEMMs on FP32 copies of the tensor (and W), Y and X (X formed in native FP64)
    ///   over blocks of [`OP32_KBLOCK`] auxiliary functions, each block's sum added in FP64
    ///   (`resp_acc_f32`); the chunk rounded to FP32 from its FP64 sums by
    ///   `resp_op_fill_mixed`; the products SGEMMs into FP32 buffers, each added to the FP64
    ///   result (so FP32 sums run over one chunk's columns or rows at most).
    trait OpElement: GemmScalar {
        /// Bytes per element.
        const BYTES: usize;
        /// The FP32 operator (its estimates, [`op_estimate`]).
        const FP32: bool;
        /// The operator in errors and records.
        const WHAT: &'static str;
        /// The arithmetic of the build's GEMMs on the element and of X (formed in FP64); None:
        /// the response's (`k_arith`).
        const ARITH: Option<Arith>;
        const X_ARITH: Option<Arith>;

        /// The engine's store of this element's operator.
        fn stored<'a>(
            op: &'a mut Option<StoredOp<f64>>,
            op32: &'a mut Option<StoredOp<f32>>,
        ) -> &'a mut Option<StoredOp<Self>>;

        /// `len` zeroed elements on the device.
        fn alloc(gpu: &Gpu, len: usize) -> Result<DeviceBuf<Self>, GpuError>;

        /// A host store of `len` elements streamed through [`OP_STAGING`] staging buffers of
        /// `slot` elements.
        fn streamed(gpu: &Gpu, len: usize, slot: usize) -> Result<StreamedStore<Self>, GpuError>;

        /// The first `len` elements of an FP64 operand on the element: the operand itself, or
        /// an FP32 copy (`resp_to_f32`).
        fn view<'a>(
            gpu: &Gpu,
            to32: &Kernel,
            x: &'a DeviceBuf<f64>,
            len: usize,
        ) -> Result<Held<'a, Self>, GpuError>;

        /// The same for an FP64 temporary the operand replaces (freed after its copy).
        fn own(
            gpu: &Gpu,
            to32: &Kernel,
            x: DeviceBuf<f64>,
            len: usize,
        ) -> Result<DeviceBuf<Self>, GpuError>;

        /// Adds sum_t alphas\[t\] P_t (`shape`: rows, cols with leading dimension rows, and the
        /// FP64 elements zeroed first) to `sums`, where `prod(t, q0, kb, scale, out, off)`
        /// queues term t's GEMM over the auxiliary functions q0..q0 + kb in the element's
        /// arithmetic, scaled by `scale`, into `out` from element `off`: f64 one GEMM per term
        /// straight into `sums` (alpha_t, beta 0 for the first term and 1 after); f32 per term
        /// and block of [`OP32_KBLOCK`] functions a GEMM into `tmp` (1, 0), added with alpha_t
        /// in FP64 (`resp_acc_f32`) to `sums`, zeroed first.
        fn ksum<P>(
            b: &OpBuild<'_>,
            shape: (usize, usize, usize),
            alphas: &[f64],
            sums: Sums<'_, Self>,
            tmp: Option<&mut DeviceBuf<Self>>,
            prod: P,
        ) -> Result<(), GpuError>
        where
            P: FnMut(
                usize,
                usize,
                usize,
                (Self, Self),
                &mut DeviceBuf<Self>,
                usize,
            ) -> Result<(), GpuError>;

        /// out\[off.., ld\] -= c_k op(chunk) A (`d`, nd columns; `ops` = (chunk, A)): f64 one
        /// native DGEMM into `out`; f32 an SGEMM into `tmp`, added to `out` in FP64.
        fn product(
            p: &OpProduct<'_>,
            tmp: Option<&mut DeviceBuf<Self>>,
            d: GemmDims,
            ops: (MatRef<'_, Self>, MatRef<'_, Self>),
            out: (&mut DeviceBuf<f64>, usize, usize),
        ) -> Result<(), GpuError>;
    }

    impl OpElement for f64 {
        const BYTES: usize = 8;
        const FP32: bool = false;
        const WHAT: &'static str = "response J/K: the stored exchange operator";
        const ARITH: Option<Arith> = None;
        const X_ARITH: Option<Arith> = None;

        fn stored<'a>(
            op: &'a mut Option<StoredOp<f64>>,
            _: &'a mut Option<StoredOp<f32>>,
        ) -> &'a mut Option<StoredOp<f64>> {
            op
        }

        fn alloc(gpu: &Gpu, len: usize) -> Result<DeviceBuf<f64>, GpuError> {
            gpu.alloc_zeros::<f64>(len)
        }

        fn streamed(gpu: &Gpu, len: usize, slot: usize) -> Result<StreamedStore<f64>, GpuError> {
            StreamedStore::new(gpu, len, (OP_STAGING, slot), Copies::Overlapped)
        }

        fn view<'a>(
            _: &Gpu,
            _: &Kernel,
            x: &'a DeviceBuf<f64>,
            _: usize,
        ) -> Result<Held<'a, f64>, GpuError> {
            Ok(Held::Borrowed(x))
        }

        fn own(
            _: &Gpu,
            _: &Kernel,
            x: DeviceBuf<f64>,
            _: usize,
        ) -> Result<DeviceBuf<f64>, GpuError> {
            Ok(x)
        }

        fn ksum<P>(
            b: &OpBuild<'_>,
            _: (usize, usize, usize),
            alphas: &[f64],
            sums: Sums<'_, f64>,
            _: Option<&mut DeviceBuf<f64>>,
            mut prod: P,
        ) -> Result<(), GpuError>
        where
            P: FnMut(
                usize,
                usize,
                usize,
                (f64, f64),
                &mut DeviceBuf<f64>,
                usize,
            ) -> Result<(), GpuError>,
        {
            let (out, off) = match sums {
                Sums::Fp64(b, o) | Sums::Chunk(b, o) => (b, o),
            };
            for (t, &alpha) in alphas.iter().enumerate() {
                let beta = if t == 0 { 0.0 } else { 1.0 };
                prod(t, 0, b.dims.1, (alpha, beta), &mut *out, off)?;
            }
            Ok(())
        }

        fn product(
            p: &OpProduct<'_>,
            _: Option<&mut DeviceBuf<f64>>,
            d: GemmDims,
            (chunk, a): (MatRef<'_, f64>, MatRef<'_, f64>),
            (out, off, ld): (&mut DeviceBuf<f64>, usize, usize),
        ) -> Result<(), GpuError> {
            p.gemm.gemm(
                Arith::Fp64,
                d,
                (-p.c_k, 1.0),
                chunk,
                a,
                MatMut::new(out, off, ld),
            )
        }
    }

    impl OpElement for f32 {
        const BYTES: usize = 4;
        const FP32: bool = true;
        const WHAT: &'static str = "response J/K: the FP32 exchange operator";
        const ARITH: Option<Arith> = Some(Arith::Fp32);
        const X_ARITH: Option<Arith> = Some(Arith::Fp64);

        fn stored<'a>(
            _: &'a mut Option<StoredOp<f64>>,
            op32: &'a mut Option<StoredOp<f32>>,
        ) -> &'a mut Option<StoredOp<f32>> {
            op32
        }

        fn alloc(gpu: &Gpu, len: usize) -> Result<DeviceBuf<f32>, GpuError> {
            gpu.alloc_zeros::<f32>(len)
        }

        fn streamed(gpu: &Gpu, len: usize, slot: usize) -> Result<StreamedStore<f32>, GpuError> {
            StreamedStore::new(gpu, len, (OP_STAGING, slot), Copies::Overlapped)
        }

        fn view<'a>(
            gpu: &Gpu,
            to32: &Kernel,
            x: &'a DeviceBuf<f64>,
            len: usize,
        ) -> Result<Held<'a, f32>, GpuError> {
            let mut y = gpu.alloc_zeros::<f32>(len.max(1))?;
            to_f32(gpu, to32, x, 0, &mut y, 0, len)?;
            Ok(Held::Owned(y))
        }

        fn own(
            gpu: &Gpu,
            to32: &Kernel,
            x: DeviceBuf<f64>,
            len: usize,
        ) -> Result<DeviceBuf<f32>, GpuError> {
            let mut y = gpu.alloc_zeros::<f32>(len.max(1))?;
            to_f32(gpu, to32, &x, 0, &mut y, 0, len)?;
            Ok(y)
        }

        fn ksum<P>(
            b: &OpBuild<'_>,
            (rows, cols, zero): (usize, usize, usize),
            alphas: &[f64],
            sums: Sums<'_, f32>,
            tmp: Option<&mut DeviceBuf<f32>>,
            mut prod: P,
        ) -> Result<(), GpuError>
        where
            P: FnMut(
                usize,
                usize,
                usize,
                (f32, f32),
                &mut DeviceBuf<f32>,
                usize,
            ) -> Result<(), GpuError>,
        {
            let (Sums::Fp64(out, off), Some(tmp)) = (sums, tmp) else {
                return Err(GpuError::Contract(
                    "response J/K: the FP32 operator's sums run in FP32 blocks into FP64".into(),
                ));
            };
            let (gpu, naux) = (b.gpu, b.dims.1);
            gpu.zero_range(out, off, zero)?;
            for (t, &alpha) in alphas.iter().enumerate() {
                for q0 in (0..naux).step_by(OP32_KBLOCK) {
                    let kb = OP32_KBLOCK.min(naux - q0);
                    prod(t, q0, kb, (1.0, 0.0), &mut *tmp, 0)?;
                    let acc = &b.kernels.acc_f32;
                    acc_f32(gpu, acc, (rows, cols), alpha, tmp, (&mut *out, off, rows))?;
                }
            }
            Ok(())
        }

        fn product(
            p: &OpProduct<'_>,
            tmp: Option<&mut DeviceBuf<f32>>,
            d: GemmDims,
            (chunk, a): (MatRef<'_, f32>, MatRef<'_, f32>),
            out: (&mut DeviceBuf<f64>, usize, usize),
        ) -> Result<(), GpuError> {
            let tmp = tmp
                .ok_or_else(|| GpuError::Contract("response J/K: no FP32 product buffer".into()))?;
            let m = d.m;
            p.gemm.gemm(
                Arith::Fp32,
                d,
                (1.0, 0.0),
                chunk,
                a,
                MatMut::new(&mut *tmp, 0, m),
            )?;
            acc_f32(p.gpu, p.acc, (m, p.nd), -p.c_k, tmp, out)
        }
    }

    /// y\[y0 + r + ldy c\] += alpha x\[r + rows c\] for the `rows` x `cols` values of x
    /// (`resp_acc_f32`: FP32 sums added in FP64).
    fn acc_f32(
        gpu: &Gpu,
        acc: &Kernel,
        (rows, cols): (usize, usize),
        alpha: f64,
        x: &DeviceBuf<f32>,
        (y, y0, ldy): (&mut DeviceBuf<f64>, usize, usize),
    ) -> Result<(), GpuError> {
        if rows * cols == 0 {
            return Ok(());
        }
        if rows * cols > x.len() || ldy < rows || y0 + rows + ldy * (cols - 1) > y.len() {
            return Err(GpuError::Contract("response J/K: FP32 sum outside".into()));
        }
        // SAFETY: matches resp_acc_f32; rows x cols values of x (ld rows) into y from element y0
        // with leading dimension ldy (all checked above).
        unsafe {
            gpu.launch(
                acc,
                Launch::linear(rows * cols, 256),
                &[
                    Arg::I64(rows as i64),
                    Arg::I64(cols as i64),
                    Arg::F64(alpha),
                    Arg::buf(x),
                    Arg::I64(rows as i64),
                    Arg::from(y.at(y0)),
                    Arg::I64(ldy as i64),
                ],
            )
        }
    }

    /// One channel's chunks of the stored operator ([`OpChunk`]) in the element `T`
    /// ([`OpElement`]) from the tensor `s` (naux x nx, the stored rows) and, with range
    /// separation, the operator tensor `w` on the same rows, both in the element's type:
    /// - X_Q = Y_Q^T C as X3 (o x o naux: element j + o i + o^2 Q), one GEMM in FP64 (Xw_Q
    ///   alike);
    /// - per panel (j0, bj), the pair part Npk_jj = S^T X3[rows o j..]^T (nx x iend per
    ///   orbital j of the panel; wK: (S^T Xw3 + W^T X3) / 2), batched over j;
    /// - per chunk (orbitals i0..i1 of the panel) the Y part G = Y[rows of j0..j0+bj]
    ///   Y[rows of i0..i1]^T (wK: (Yw Y^T + Y Yw^T) / 2), then `resp_op_fill`
    ///   (`resp_op_fill_mixed` from the FP32 build's FP64 sums).
    ///
    /// A host chunk (`t`: the host chunks built so far, over the channels) is built in staging
    /// buffer t mod k and copied to the host store on the copy stream, which first waits for
    /// the device's fill, while the device's stream goes on with the next chunks; it reuses a
    /// staging buffer only after its previous copy is done.
    fn build_channel<T: OpElement>(
        b: &OpBuild<'_>,
        ch: &Chan,
        (s, w): (&DeviceBuf<T>, Option<&DeviceBuf<T>>),
        (chunks, op): (&[OpChunk], &mut StoredOp<T>),
        t: &mut usize,
    ) -> Result<(), GpuError> {
        let OpBuild {
            gpu,
            gemm,
            arith,
            kernels,
            map,
            ..
        } = *b;
        let (n, naux, nx) = b.dims;
        let o = ch.o;
        let no = n * o;
        let x3_of = |y: &DeviceBuf<f64>| -> Result<DeviceBuf<T>, GpuError> {
            let mut x = gpu.alloc_zeros::<f64>(o * o * naux)?;
            gemm.gemm(
                T::X_ARITH.unwrap_or(arith),
                GemmDims::new(Op::T, Op::N, o, o * naux, n),
                (1.0, 0.0),
                MatRef::new(&ch.c, 0, n),
                MatRef::new(y, 0, n),
                MatMut::new(&mut x, 0, o),
            )?;
            T::own(gpu, &kernels.to_f32, x, o * o * naux)
        };
        let rs = matches!((&ch.ypw, w), (Some(_), Some(_)));
        let x3 = x3_of(&ch.yp)?;
        let x3w = match (&ch.ypw, rs) {
            (Some(y), true) => Some(x3_of(y)?),
            _ => None,
        };
        let y = T::view(gpu, &kernels.to_f32, &ch.yp, no * naux)?;
        let yw = match (&ch.ypw, rs) {
            (Some(y), true) => Some(T::view(gpu, &kernels.to_f32, y, no * naux)?),
            _ => None,
        };
        let bmax = chunks.iter().map(|c| c.bj).max().unwrap_or(1);
        let lmax = chunks.iter().map(|c| c.len(n)).max().unwrap_or(1);
        // the panel's pair part in FP64 and, for the FP32 build, its FP32 block sums and a
        // chunk's FP32 block sums and FP64 sums (allocated in this order)
        let mut npk32 = T::FP32.then(|| T::alloc(gpu, nx * bmax * o)).transpose()?;
        let mut npk = gpu.alloc_zeros::<f64>(nx * bmax * o)?;
        let mut chunk32 = T::FP32.then(|| T::alloc(gpu, lmax)).transpose()?;
        let mut chunk64 = T::FP32.then(|| gpu.alloc_zeros::<f64>(lmax)).transpose()?;
        // the pair part's terms (tensor, X, alpha) and the Y part's (left, right, alpha)
        let (pterms, np) = match (&x3w, w) {
            (Some(x3w), Some(w)) => ([(s, x3w, 0.5), (w, &x3, 0.5)], 2),
            _ => ([(s, &x3, 1.0); 2], 1),
        };
        let (yterms, ny) = match &yw {
            Some(yw) => ([(&**yw, &*y, 0.5), (&*y, &**yw, 0.5)], 2),
            None => ([(&*y, &*y, 1.0); 2], 1),
        };
        let (palpha, yalpha) = (pterms.map(|p| p.2), yterms.map(|p| p.2));
        let ns = op.host.as_ref().map_or(1, |h| h.slots().max(1));
        let mut panel: Option<usize> = None;
        for c in chunks {
            let (j0, bj) = (c.j0, c.bj);
            let (rows, iend) = (bj * n, j0 + bj);
            if nx * bj * iend > npk.len() || map.len() < n * n {
                return Err(GpuError::Contract(
                    "response J/K: operator panel outside its buffers".into(),
                ));
            }
            // the panel's pair part, at its first chunk
            if panel != Some(j0) {
                panel = Some(j0);
                let shape = (nx * iend * bj, 1, npk.len());
                T::ksum(
                    b,
                    shape,
                    &palpha[..np],
                    Sums::Fp64(&mut npk, 0),
                    npk32.as_mut(),
                    |k, q0, kb, scale, out, off| {
                        let (tm, x, _) = pterms[k];
                        gemm.gemm_strided(
                            T::ARITH.unwrap_or(arith),
                            GemmDims::new(Op::T, Op::T, nx, iend, kb).batch(bj),
                            scale,
                            MatRef::new(tm, q0, naux),
                            MatRef::new(x, o * j0 + o * o * q0, o * o).stride(o),
                            MatMut::new(out, off, nx).stride(nx * iend),
                        )
                    },
                )?;
            }
            let ni = c.i1 - c.i0;
            let len = c.len(n);
            // the Y part, into FP64 sums before the chunk is taken (FP32 build) or into the
            // chunk itself (FP64)
            let ypart = |sums: Sums<'_, T>, tmp: Option<&mut DeviceBuf<T>>| {
                T::ksum(
                    b,
                    (rows, ni * n, len),
                    &yalpha[..ny],
                    sums,
                    tmp,
                    |k, q0, kb, scale, out, off| {
                        let (l, r, _) = yterms[k];
                        gemm.gemm(
                            T::ARITH.unwrap_or(arith),
                            GemmDims::new(Op::N, Op::T, rows, ni * n, kb),
                            scale,
                            MatRef::new(l, j0 * n + q0 * no, no),
                            MatRef::new(r, c.i0 * n + q0 * no, no),
                            MatMut::new(out, off, rows),
                        )
                    },
                )
            };
            if let Some(g64) = chunk64.as_mut() {
                ypart(Sums::Fp64(g64, 0), chunk32.as_mut())?;
            }
            let (buf, off) = match c.at {
                Loc::Device(off) => (
                    op.dev.as_mut().ok_or_else(|| {
                        GpuError::Contract("response J/K: no device store".into())
                    })?,
                    off,
                ),
                Loc::Host(_) => (host_of(&mut op.host)?.slot(gpu, *t % ns)?, 0),
            };
            if off + len > buf.len() || chunk64.as_ref().is_some_and(|g| len > g.len()) {
                return Err(GpuError::Contract(
                    "response J/K: operator chunk outside its buffer".into(),
                ));
            }
            if chunk64.is_none() {
                ypart(Sums::Chunk(&mut *buf, off), None)?;
            }
            let tiles = n.div_ceil(32) as u32;
            let geometry = Launch {
                grid: (tiles, tiles, (bj * ni) as u32),
                block: (32, 8, 1),
                shared_bytes: 0,
            };
            // SAFETY: matches resp_op_fill (FP64: the chunk's rows x ni n elements at `off`, ld
            // rows, completed in place) and resp_op_fill_mixed (FP32: the chunk's FP64 sums in
            // chunk64, its FP32 elements at `off`); nx x bj iend values of npk and n^2 of map
            // (all checked above).
            unsafe {
                match &chunk64 {
                    None => gpu.launch(
                        &kernels.op_fill,
                        geometry,
                        &[
                            Arg::I64(n as i64),
                            Arg::I32(iend as i32),
                            Arg::I32(c.i0 as i32),
                            Arg::I32(ni as i32),
                            Arg::I64(rows as i64),
                            Arg::from(buf.at(off)),
                            Arg::buf(&npk),
                            Arg::I64(nx as i64),
                            Arg::buf(map),
                        ],
                    ),
                    Some(g64) => gpu.launch(
                        &kernels.op_fill_mixed,
                        geometry,
                        &[
                            Arg::I64(n as i64),
                            Arg::I32(iend as i32),
                            Arg::I32(c.i0 as i32),
                            Arg::I32(ni as i32),
                            Arg::I64(rows as i64),
                            Arg::buf(g64),
                            Arg::from(buf.at(off)),
                            Arg::buf(&npk),
                            Arg::I64(nx as i64),
                            Arg::buf(map),
                        ],
                    ),
                }
            }?;
            if let Loc::Host(hoff) = c.at {
                // the fill above is the slot's last use: its copy to the host waits for it
                let host = host_of(&mut op.host)?;
                host.release(gpu, *t % ns)?;
                host.store(gpu, *t % ns, hoff, c.len(n))?;
                *t += 1;
            }
        }
        Ok(())
    }

    /// Auxiliary functions per FP32 sum of the FP32 operator's setup; the blocks' sums are
    /// added in FP64 (one FP32 sum over all of them made the operator's products 5-8x less
    /// accurate than the FP32 exchange through the tensor, which sums 64 at a time).
    const OP32_KBLOCK: usize = 256;

    /// The single-precision buffers of an exchange call.
    struct Single {
        a: DeviceBuf<f32>,
        aw: DeviceBuf<f32>,
        z: DeviceBuf<f32>,
        wst: DeviceBuf<f32>,
        yb: DeviceBuf<f32>,
        xs: DeviceBuf<f32>,
        /// A group's exchange over one block (n x ndg o).
        acc: DeviceBuf<f32>,
        c: Vec<DeviceBuf<f32>>,
        acat: Vec<DeviceBuf<f32>>,
    }

    /// One group of perturbations in one auxiliary block: sizes, the group's first factor
    /// element (`a_off`, also its first output element) and the coefficient `f`.
    struct Group {
        n: usize,
        o: usize,
        nqb: usize,
        ndg: usize,
        a_off: usize,
        f: f64,
    }

    impl Group {
        /// Z = P^T [A_d] ((nqb n) x ndg o) for the planes P.
        fn gemm1_f64(
            &self,
            (gemm, arith): (&Gemm, Arith),
            planes: &DeviceBuf<f64>,
            acat: &DeviceBuf<f64>,
            z: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            let (n, nqb) = (self.n, self.nqb);
            gemm.gemm(
                arith,
                GemmDims::new(Op::T, Op::N, nqb * n, self.ndg * self.o, n),
                (1.0, 0.0),
                MatRef::new(planes, 0, n),
                MatRef::new(acat, self.a_off, n),
                MatMut::new(z, 0, nqb * n),
            )
        }

        /// [`Group::gemm1_f64`] in Ozaki slices: the pre-sliced planes of the tensor's
        /// functions q0.. times the group's factors (`acs`: every perturbation's factor
        /// columns, sliced once per product), in column chunks of `cols`.
        #[allow(clippy::too_many_arguments)]
        fn gemm1_ozaki(
            &self,
            gpu: &Gpu,
            (ozaki, planes): (&Ozaki, Option<&Sliced>),
            acs: &Sliced,
            q0: usize,
            z: &mut DeviceBuf<f64>,
            cols: usize,
            work: &mut OzakiWork,
        ) -> Result<(), GpuError> {
            let (n, nqb) = (self.n, self.nqb);
            let rows = nqb * n;
            let planes = planes.ok_or_else(|| {
                GpuError::Contract("response J/K: the Ozaki planes are not held".into())
            })?;
            let pv = planes.rows_view(q0 * n, rows)?;
            let wg = self.ndg * self.o;
            // the group's first factor column
            let v0 = self.a_off / n;
            let mut c0 = 0;
            while c0 < wg {
                let nc = cols.min(wg - c0);
                let av = acs.rows_view(v0 + c0, nc)?;
                ozaki.gemm(gpu, pv, av, 1.0, 0.0, (&mut *z, c0 * rows, rows), work)?;
                c0 += nc;
            }
            Ok(())
        }

        /// The stacked products Wst = Z^T C (Z as n x (wg nqb), rows (d, i, q)).
        fn wst_f64(
            &self,
            (gemm, arith): (&Gemm, Arith),
            c: &DeviceBuf<f64>,
            z: &DeviceBuf<f64>,
            wst: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            let (n, o, nqb) = (self.n, self.o, self.nqb);
            let wg = self.ndg * o;
            gemm.gemm(
                arith,
                GemmDims::new(Op::T, Op::N, wg * nqb, o, n),
                (1.0, 0.0),
                MatRef::new(z, 0, n),
                MatRef::new(c, 0, n),
                MatMut::new(wst, 0, wg * nqb),
            )
        }

        /// out_d += f Z_d X and out_d += f Y' Wst_d, batched over the group's perturbations
        /// (`x`, `y`: buffer and offset of X and of the other tensor's Y).
        fn batched_f64(
            &self,
            (gemm, arith): (&Gemm, Arith),
            (z, wst): (&DeviceBuf<f64>, &DeviceBuf<f64>),
            x: (&DeviceBuf<f64>, usize),
            y: (&DeviceBuf<f64>, usize),
            out: &mut DeviceBuf<f64>,
        ) -> Result<(), GpuError> {
            let (n, o, nqb, ndg) = (self.n, self.o, self.nqb, self.ndg);
            let (wg, ono) = (ndg * o, o * nqb);
            gemm.gemm_strided(
                arith,
                GemmDims::new(Op::N, Op::N, n, o, ono).batch(ndg),
                (self.f, 1.0),
                MatRef::new(z, 0, n).stride(ono * n),
                MatRef::new(x.0, x.1, ono),
                MatMut::new(out, self.a_off, n).stride(n * o),
            )?;
            gemm.gemm_strided(
                arith,
                GemmDims::new(Op::N, Op::N, n, o, ono).batch(ndg),
                (self.f, 1.0),
                MatRef::new(y.0, y.1, n),
                MatRef::new(wst, 0, wg * nqb).stride(ono),
                MatMut::new(out, self.a_off, n).stride(n * o),
            )
        }

        /// [`Group::gemm1_f64`] in single precision.
        fn gemm1_f32(
            &self,
            gemm: &Gemm,
            planes: &DeviceBuf<f32>,
            acat: &DeviceBuf<f32>,
            z: &mut DeviceBuf<f32>,
        ) -> Result<(), GpuError> {
            let (n, nqb) = (self.n, self.nqb);
            gemm.gemm(
                Arith::Fp32,
                GemmDims::new(Op::T, Op::N, nqb * n, self.ndg * self.o, n),
                (1.0, 0.0),
                MatRef::new(planes, 0, n),
                MatRef::new(acat, self.a_off, n),
                MatMut::new(z, 0, nqb * n),
            )
        }

        /// [`Group::wst_f64`] and [`Group::batched_f64`] in single precision into the block
        /// accumulator `acc` (n x ndg o; overwritten unless `add`).
        #[allow(clippy::too_many_arguments)]
        fn contract_f32(
            &self,
            gemm: &Gemm,
            c: &DeviceBuf<f32>,
            (z, wst): (&DeviceBuf<f32>, &mut DeviceBuf<f32>),
            x: (&DeviceBuf<f32>, usize),
            y: (&DeviceBuf<f32>, usize),
            acc: &mut DeviceBuf<f32>,
            add: bool,
        ) -> Result<(), GpuError> {
            let (n, o, nqb, ndg) = (self.n, self.o, self.nqb, self.ndg);
            let (wg, ono) = (ndg * o, o * nqb);
            let f = self.f as f32;
            gemm.gemm(
                Arith::Fp32,
                GemmDims::new(Op::T, Op::N, wg * nqb, o, n),
                (1.0, 0.0),
                MatRef::new(z, 0, n),
                MatRef::new(c, 0, n),
                MatMut::new(wst, 0, wg * nqb),
            )?;
            gemm.gemm_strided(
                Arith::Fp32,
                GemmDims::new(Op::N, Op::N, n, o, ono).batch(ndg),
                (f, if add { 1.0 } else { 0.0 }),
                MatRef::new(z, 0, n).stride(ono * n),
                MatRef::new(x.0, x.1, ono),
                MatMut::new(acc, 0, n).stride(n * o),
            )?;
            gemm.gemm_strided(
                Arith::Fp32,
                GemmDims::new(Op::N, Op::N, n, o, ono).batch(ndg),
                (f, 1.0),
                MatRef::new(y.0, y.1, n),
                MatRef::new(wst, 0, wg * nqb).stride(ono),
                MatMut::new(acc, 0, n).stride(n * o),
            )
        }
    }
}
