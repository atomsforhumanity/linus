//! The density-fitted two-electron terms of an analytic Hessian on the device: `GpuDfHess`
//! (feature `gpu`).
//!
//! The quantities of linus's CPU `df_hessian_2e_op` (`two_electron::df::hessian` in the
//! `linus` crate) for spin channels s with occupied orbitals C_s (n x o_s) and weights f_s
//! (closed shell: one channel, f = 2; open shell: alpha and beta, f = 1), the Coulomb metric
//! V = L L^T and the exchange energy -a_x/2 sum_s f_s sum_ij B_s,ij^T V^-1 B_s,ij of the
//! three-center integrals B_s,P,ij = (P|op|ij) of an operator op (1/r, or a combination
//! a/r + b erf(omega r)/r for the polarization terms of a range-separated hybrid's wK
//! exchange):
//!
//! - F^x C_s = (J^x\[D_t\] - a_x K^x\[D_s\]) C_s for every perturbation x = (atom A, k), the
//!   derivative Fock matrices of the CPHF right-hand sides at fixed density;
//! - the cross terms of the RI energy's second derivative at fixed orbitals: Coulomb
//!   r^x V^-1 r^y (r = gamma^x - V^x c) and exchange -a_x f_s R^x V^-1 R^y
//!   (R^x = C^T (P|..)^x C - V^x X);
//! - the explicit second-derivative integrals: three-center with
//!   Gamma_P = c_P D_t - a_x sum_s f_s C_s X_s,P C_s^T (the generated second-derivative kernels,
//!   [`crate::eri3c`]'s `Eri3c::hessian`), two-center with
//!   gamma_PQ = -c_P c_Q / 2 + a_x/2 sum_s f_s tr(X_s,P X_s,Q) (built on the device, contracted
//!   on the host by linus-ints' `Eri2DerivPlan`: naux^2 work against the three-center n^2 naux).
//!
//! Notation of the CPU module docs: I^k_P\[a\]\[b\] = (∂_k a b|P) (derivative of AO a's
//! centre), H^k_s,P = I^k_P C_s, Z_s = V^-1 (P|m i), X_s,P = C_s^T Z_s,P (symmetrized),
//! Y_s,P = C_s X_s,P, G^k_PQ = (∂_k P|Q), R^x = a^x + b^x + c^x with a^x_P = δ(P in A) E^k_P,
//! E^k_P = -(C^T H^k_P + h.c.) - sum_Q G^k_PQ X_Q, b^x_P = S^{A,k}_P + S^{A,k}_P^T
//! (S^{A,k}_P = C_A^T H^k_P\[A\]), c^x_P = -sum_{Q in A} G^k_QP X_Q.
//!
//! The exchange's cross term is taken as G = sum_P R~^x_P . R~^y_P with R~ = L^-1 R (V = L L^T),
//! where the CPU forms (a+b+c) V^-1 (b+c) + \[a V^-1 (b+c)\]^T + a V^-1 a with H-bar = V^-1 H:
//! L^-1 b^x is the b part of H~ = L^-1 H, L^-1 c^x the c part of L^-1 G^k^T, and
//! (L^-1 a^x)_P = sum_{Q in A} (L^-1)_PQ E^k_Q (L^-1 explicit, lower triangular). The exchange
//! part of F^x C, sum_P Z_P R^x_P, is sum_P W~_P R~^x_P with W~ = L^-1 W (Z = L^-T W~). One
//! triangular solve of H instead of two, and no V^-1 (b+c) or a V^-1 a.
//!
//! The device's work, all of it in FP64 on one stream:
//! 1. **MO pass** over blocks of whole first shells (the kept pairs of [`DfLayout`] as linus-gpu's
//!    DF gradient): (mn|op|P) by the generated kernels, gamma by a GEMV with the density rows,
//!    the dense block (both triangles inside the block) and W_s\[m\]\[i\]\[P\] = sum_n (mn|P) C_ni
//!    by strided-batched GEMMs (the rows m of the block, and the earlier functions n through
//!    the transposed block).
//! 2. **Fit:** c = V^-1 gamma, W~ = L^-1 W (kept as \[m\]\[P\]\[i\]) and Z = L^-T W~ by triangular
//!    solves; X_s by a GEMM and a symmetrizing kernel, packed (lower triangle, sqrt 2 off the
//!    diagonal).
//! 3. **Pass 1** over blocks of auxiliary functions of one atom (`Eri3c::compute_block`:
//!    every kept pair against the block's shells): the six derivative planes unpacked to the
//!    dense I^k\[a\]\[q\]\[b\] (the CPU's block), H^k_s by one GEMM, gamma^x, J1 = sum_P c_P I_P and
//!    its auxiliary-centre part, Y_s, sum_P H_P X_P, the T^{A,k} = I^k\[A\]^T Y\[A\] (per AO
//!    function by one batched GEMM, summed per atom by a kernel), and H^k_s stored for all P
//!    (\[k\]\[a\]\[i\]\[P\]).
//! 4. **J terms:** r = gamma^x - V^x c, sum_P Z_P r^x_P and J1 C by GEMMs, and the Coulomb
//!    cross term (triangular solve, GEMM).
//! 5. **E pre-pass** over sub-chunks of the auxiliary blocks: E^k for every P (a kernel over the
//!    rows, a GEMM with G), stored; then H~ = L^-1 H in place.
//! 6. **Pass 2** over the sub-chunks (no integrals): R~^x for every x (b part by the kernel over
//!    the atom's rows of H~, c part by GEMMs with L^-1 G^T, a part by GEMMs with L^-1 and E),
//!    the Gram GEMM, and sum_P W~_P R~^x_P (the exchange part of F^x C) by one GEMM.
//! 7. **Second derivatives** over the MO pass's blocks: Zd = a_x sum_s f_s C_s X_s C_s^T for the
//!    block's rows by GEMMs (as the DF gradient), Gamma in the pair-row layout by a kernel, the
//!    second-derivative kernels with their Gamma epilogue into a fixed-point (3N)^2 accumulator
//!    (per sub-block of pairs; the erf kernels too for an operator with an erf part); gamma_PQ
//!    by GEMMs, to the host for the two-center term.
//!
//! Then the downloads and the host assembly: F^x C from its J and K parts, the Gram matrices,
//! the fixed-point accumulator and the two-center term (linus-ints' `Eri2DerivPlan`).
//!
//! In the code each step is a method of `GpuDfHess` (`mo_pass`, `fit`, `pass1`,
//! `coulomb_terms`, `exchange` for steps 5 and 6, `second_derivatives`, `assemble`). A method
//! takes the call's context `HessCall` (the inputs, the sizes the steps share, the timeline of
//! device events behind [`GpuDfHessTimes`]) and the buffers earlier steps left (`MoPass`,
//! `Fit`, `Pass1`, `SecondDerivs`); `GpuDfHess::compute_device` runs them in order and frees
//! each buffer after the step that reads it last. The two longest steps are split further:
//! pass 1 into its buffers (`pass1_buffers`: the per-block workspace `Pass1Work` and the
//! `Pass1` it leaves) and one `pass1_block` per auxiliary block (`pass1_derivs`, `pass1_j1`,
//! `pass1_channel` with the exchange's `pass1_k_terms`, `pass1_gx`); the exchange into one
//! `exchange_channel` per channel (its workspace, the E pre-pass `exchange_e`, the solve, and
//! one `exchange_sub` per sub-chunk: `exchange_r`, the Gram GEMM, `exchange_wr`).
//!
//! Unlike the CPU, which keeps memory lean (H-bar in atom blocks, one integral pass per block),
//! the device stores H (then H~) for every auxiliary function (3 n o naux doubles per channel)
//! and computes the derivative integrals once.
//!
//! Deterministic: cuBLAS in a fixed configuration, one writer per element in every kernel,
//! fixed loop orders, blocks in a fixed order on one stream; the host assembly is sequential.

use crate::cpu::{BasisSet, DfThresholds};
use crate::df::DfLayout;
use crate::grad::df::{DfGradPlan, GpuDfGradOptions};

/// Settings of a `GpuDfHess` setup.
#[derive(Clone, Debug)]
pub struct GpuDfHessOptions {
    /// The screening thresholds: kept shell pairs, the three-center integrals' primitives and
    /// the derivative integrals'.
    pub thresholds: DfThresholds,
    /// Bytes of the MO pass's dense block ((m1 - m0) m1 naux doubles).
    pub block_bytes: usize,
    /// Bytes of an auxiliary block's dense derivative integrals (3 n^2 nq doubles); an atom's
    /// auxiliary functions are split when they exceed it.
    pub dblock_bytes: usize,
    /// Bytes of pass 2's per-sub-chunk buffers (R, V^-1 (b+c), their unpacked copy).
    pub rz_bytes: usize,
}

impl Default for GpuDfHessOptions {
    fn default() -> Self {
        GpuDfHessOptions {
            thresholds: DfThresholds::default(),
            block_bytes: 1 << 30,
            dblock_bytes: 1 << 30,
            rz_bytes: 2 << 30,
        }
    }
}

/// A block of auxiliary functions of one atom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuxAtomBlock {
    pub p0: usize,
    pub nq: usize,
    pub atom: usize,
}

/// The host plan of a `GpuDfHess` (no device needed).
pub struct DfHessPlan {
    /// The MO pass's blocks and the kept pairs (linus-gpu's DF gradient plan).
    pub grad: DfGradPlan,
    pub natom: usize,
    /// Per atom: its AO functions and its auxiliary functions.
    pub aor: Vec<std::ops::Range<usize>>,
    pub xr: Vec<std::ops::Range<usize>>,
    /// Atom of each auxiliary function.
    pub aux_atom: Vec<usize>,
    /// Pass 1 and 2's blocks of auxiliary functions (whole shells of one atom).
    pub aux_blocks: Vec<AuxAtomBlock>,
}

impl DfHessPlan {
    pub fn new(basis: &BasisSet, aux: &BasisSet, opts: &GpuDfHessOptions) -> DfHessPlan {
        Self::with_layout(None, basis, aux, opts)
    }

    /// The plan on the kept pairs of `layout` when given (built for `basis` and `aux` at
    /// `opts.thresholds.pair`, e.g. a [`crate::df::DfShared`]'s), else its own.
    pub fn with_layout(
        layout: Option<std::sync::Arc<DfLayout>>,
        basis: &BasisSet,
        aux: &BasisSet,
        opts: &GpuDfHessOptions,
    ) -> DfHessPlan {
        let gopts = GpuDfGradOptions {
            thresholds: opts.thresholds,
            block_bytes: opts.block_bytes,
            ..GpuDfGradOptions::default()
        };
        let grad = match layout {
            Some(l) => DfGradPlan::with_layout(l, basis, aux, &gopts),
            None => DfGradPlan::new(basis, aux, &gopts),
        };
        let natom = basis.atom_shells.len().max(aux.atom_shells.len());
        let range = |b: &BasisSet, a: usize| -> std::ops::Range<usize> {
            let r = b.atom_shells.get(a).cloned().unwrap_or(0..0);
            if r.is_empty() {
                return 0..0;
            }
            let (f, l) = (&b.shells[r.start], &b.shells[r.end - 1]);
            f.ao_start..l.ao_start + l.nfunc()
        };
        let aor = (0..natom).map(|a| range(basis, a)).collect();
        let xr = (0..natom).map(|a| range(aux, a)).collect();
        let mut aux_atom = vec![0; aux.nbf];
        for s in &aux.shells {
            aux_atom[s.ao_start..s.ao_start + s.nfunc()].fill(s.atom);
        }
        // whole shells of one atom, at most dblock_bytes of dense derivative integrals
        let n = basis.nbf.max(1);
        let maxq = (opts.dblock_bytes / (24 * n * n)).max(1);
        let mut aux_blocks = Vec::new();
        let sh = &aux.shells;
        let mut s0 = 0;
        while s0 < sh.len() {
            let mut s1 = s0 + 1;
            let mut nf = sh[s0].nfunc();
            while s1 < sh.len() && sh[s1].atom == sh[s0].atom && nf + sh[s1].nfunc() <= maxq {
                nf += sh[s1].nfunc();
                s1 += 1;
            }
            aux_blocks.push(AuxAtomBlock {
                p0: sh[s0].ao_start,
                nq: nf,
                atom: sh[s0].atom,
            });
            s0 = s1;
        }
        DfHessPlan {
            grad,
            natom,
            aor,
            xr,
            aux_atom,
            aux_blocks,
        }
    }

    pub fn layout(&self) -> &DfLayout {
        &self.grad.layout
    }
}

/// What `GpuDfHess` computes for one call (host matrices).
#[derive(Clone, Debug, Default)]
pub struct GpuDfHessParts {
    /// Per channel s: F^x C_s = (J^x\[D_t\] - a_x K^x\[D_s\]) C_s for x = 0..3 natom, each n x o_s
    /// column-major, x-major.
    pub fxo: Vec<Vec<f64>>,
    /// The explicit second derivative at fixed orbitals (cross terms and second-derivative
    /// integrals), (3 natom)^2 row-major.
    pub hess: Vec<f64>,
}

/// Seconds of the phases of the last call (device phases from events).
#[derive(Clone, Debug, Default)]
pub struct GpuDfHessTimes {
    pub total: f64,
    /// MO pass: integrals, gamma, the half-transformation.
    pub mo: f64,
    /// Fit: c, Z, X, the transposes.
    pub fit: f64,
    /// Pass 1: derivative integrals and their unpacking.
    pub deriv: f64,
    /// Pass 1: H, gamma^x, J1, Y, the K terms, the store of H.
    pub pass1: f64,
    /// sum Z r, the Coulomb cross term, H~ = L^-1 H.
    pub solve: f64,
    /// The E pre-pass and pass 2: R~, Gram.
    pub rz: f64,
    /// Pass 2: sum_P Z_P R^x_P.
    pub t2: f64,
    /// Second derivatives: Zd, Gamma and the three-center kernels; gamma_PQ.
    pub d2: f64,
    /// The two-center second derivatives on the host (wall clock).
    pub d2_2c: f64,
    /// Downloads and the host assembly (wall clock).
    pub host: f64,
}

#[cfg(feature = "gpu")]
impl crate::engine::Phases for GpuDfHessTimes {
    /// The second derivatives are `2nd` (device) and `2nd_2c` (the host's two-center term).
    fn phases(&self) -> impl IntoIterator<Item = (&'static str, f64)> {
        [
            ("mo", self.mo),
            ("fit", self.fit),
            ("deriv", self.deriv),
            ("pass1", self.pass1),
            ("solve", self.solve),
            ("rz", self.rz),
            ("t2", self.t2),
            ("2nd", self.d2),
            ("2nd_2c", self.d2_2c),
            ("host", self.host),
        ]
    }
}

#[cfg(feature = "gpu")]
pub use device::*;

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::blas::{GemmDims, MatMut, MatRef, Op};
    use crate::cpu::{
        AoMatrix, DerivOrder, Eri2DerivOptions, Eri2DerivPlan, Operator, PairOptions, PairSet,
    };
    use crate::detsum::DetAcc;
    use crate::df_deriv::{
        upload_orbitals, DerivKernels, DerivSource, GammaJ, GammaTerms, RowBlock, RowTables, Unpack,
    };
    use crate::engine::Timeline;
    use crate::eri3c::{Eri3c, Eri3cAuxBlocks, Eri3cKind};
    use crate::error::GpuError;
    use crate::kernels::registry::{KernelModule, COMMON_H};
    use crate::linalg::{Arith, Blas1, Gemm, Operand, Scale};
    use crate::pool::{Grant, PlanViolation};
    use crate::{Arg, DevPtr, DeviceBuf, Gpu, Kernel, Launch};
    use std::borrow::Cow;
    use std::time::Instant;

    const KERNELS: &str = include_str!("../../device/hess/df.cu");

    /// The DF Hessian's module: the common device functions, then its kernels.
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "dfhess",
        || Cow::Owned(format!("{COMMON_H}\n{KERNELS}")),
        &[
            "hd_unpack_der",
            "hd_hq",
            "hd_gx",
            "hd_vc",
            "hd_xsym",
            "hd_xpack",
            "hd_ztrans",
            "hd_scatter_h",
            "hd_gather_h",
            "hd_symc",
            "hd_unpack_r",
            "hd_tsym",
            "hd_tseg",
        ],
    );

    struct Kernels {
        density_rows: Kernel,
        /// The DF derivative engines' common kernels (the MO pass's unpack, Gamma).
        deriv: DerivKernels,
        blas1: Blas1,
        unpack_der: Kernel,
        hq: Kernel,
        gx: Kernel,
        vc: Kernel,
        xsym: Kernel,
        xpack: Kernel,
        ztrans: Kernel,
        scatter_h: Kernel,
        gather_h: Kernel,
        symc: Kernel,
        unpack_r: Kernel,
        tsym: Kernel,
        tseg: Kernel,
    }

    fn err(what: &str) -> GpuError {
        GpuError::Contract(format!("DF Hessian: {what}"))
    }

    /// The DF two-electron Hessian terms on the device (see the module docs).
    pub struct GpuDfHess {
        nbf: usize,
        naux: usize,
        plan: DfHessPlan,
        ints: Eri3c,
        ders: Eri3c,
        /// The second-derivative builder (Gamma epilogue into the Hessian).
        hder: Eri3c,
        auxb: Eri3cAuxBlocks,
        /// The auxiliary basis and its pair data for the host's two-center term.
        aux: BasisSet,
        /// the auxiliary pairs with second-order expansions
        ap: PairSet,
        chol: DeviceBuf<f64>,
        /// G^k_PQ = (∂_k P|Q), [k][P][Q].
        g: DeviceBuf<f64>,
        /// gt^k = L^-1 G^k^T, column-major ((P, Q) at P + naux Q) per k.
        gt: DeviceBuf<f64>,
        /// L^-1, column-major (lower triangular).
        linv: DeviceBuf<f64>,
        rows: RowTables,
        ao0: DeviceBuf<i32>,
        ao1: DeviceBuf<i32>,
        xr0: DeviceBuf<i32>,
        xr1: DeviceBuf<i32>,
        aux_atom: DeviceBuf<i32>,
        /// hd_symc's outputs: per perturbation x (k, AO rows of its atom), and the three
        /// directions over all rows.
        sym_atoms: SymRows,
        sym_all: SymRows,
        gemm: Gemm,
        kernels: Kernels,
        rz_bytes: usize,
        /// The exchange's sum_P W~_P R~^x_P in emulated FP64 (Ozaki scheme, slices), or native.
        emu: Option<(crate::emu::Ozaki, usize)>,
        emu_work: crate::emu::OzakiWork,
        /// The memory plan's grant of the calls' workspaces ([`GpuDfHess::grant_workspace`]),
        /// drawn by their allocations (freed, they give the bytes back).
        workspace: Option<Grant>,
        /// Setup seconds (wall clock).
        pub setup_seconds: f64,
        pub last: GpuDfHessTimes,
    }

    /// Launches one of this module's kernels.
    ///
    /// # Safety
    /// The arguments must match the kernel's signature and every index the kernel forms
    /// from them must lie inside the buffers passed.
    unsafe fn launch(gpu: &Gpu, k: &Kernel, n: usize, args: &[Arg]) -> Result<(), GpuError> {
        if n == 0 {
            return Ok(());
        }
        // SAFETY: forwarded to the caller.
        unsafe { gpu.launch(k, Launch::linear(n, 256), args) }
    }

    impl GpuDfHess {
        /// Sets up `basis` with the auxiliary basis `aux`: the plan, the integral and derivative
        /// builders (erf kernels added by [`GpuDfHess::prepare_operator`]), the metric's
        /// Cholesky factor L and L^-1, its derivative rows G (host, linus-ints) and L^-1 G^T.
        pub fn new(
            gpu: &Gpu,
            basis: &BasisSet,
            aux: &BasisSet,
            opts: &GpuDfHessOptions,
        ) -> Result<GpuDfHess, GpuError> {
            Self::with_shared(gpu, basis, aux, opts, None)
        }

        /// As [`GpuDfHess::new`], with the layout, the packed integrals, the metric and the
        /// derivative records of `shared` (an SCF setup's, [`crate::df::DfShared`]) when it
        /// fits the basis sets and thresholds: the same engine, bit for bit (records it builds
        /// stay there).
        pub fn with_shared(
            gpu: &Gpu,
            basis: &BasisSet,
            aux: &BasisSet,
            opts: &GpuDfHessOptions,
            shared: Option<&crate::df::DfShared>,
        ) -> Result<GpuDfHess, GpuError> {
            let t0 = Instant::now();
            let shared = shared.filter(|s| s.fits(basis, aux, &opts.thresholds));
            let plan = DfHessPlan::with_layout(shared.map(|s| s.layout.clone()), basis, aux, opts);
            let lay = plan.layout();
            let (nbf, naux, nx) = (lay.nbf, lay.naux, lay.nx);
            if nx == 0 || naux == 0 {
                return Err(err("no shell pair or no auxiliary function"));
            }
            let source = DerivSource {
                basis,
                aux,
                lay,
                shared,
                thresholds: &opts.thresholds,
            };
            let (ints, _) = source.integrals(gpu)?;
            let (ders, _) = source.derivatives(gpu, Eri3cKind::Derivatives)?;
            let (hder, _) = source.derivatives(gpu, Eri3cKind::Hessian)?;
            let blocks: Vec<(usize, usize)> =
                plan.aux_blocks.iter().map(|b| (b.p0, b.nq)).collect();
            let auxb = ders.aux_blocks(gpu, &blocks)?;
            let df = &crate::df::MODULE;
            let src = MODULE.source();
            let k = |e: &str| gpu.kernel(MODULE.name, &src, e);
            let kernels = Kernels {
                density_rows: gpu.kernel(df.name, &df.source(), "df_density_rows")?,
                deriv: DerivKernels::new(gpu)?,
                blas1: Blas1::new(gpu)?,
                unpack_der: k("hd_unpack_der")?,
                hq: k("hd_hq")?,
                gx: k("hd_gx")?,
                vc: k("hd_vc")?,
                xsym: k("hd_xsym")?,
                xpack: k("hd_xpack")?,
                ztrans: k("hd_ztrans")?,
                scatter_h: k("hd_scatter_h")?,
                gather_h: k("hd_gather_h")?,
                symc: k("hd_symc")?,
                unpack_r: k("hd_unpack_r")?,
                tsym: k("hd_tsym")?,
                tseg: k("hd_tseg")?,
            };
            let gemm = Gemm::new(gpu)?;
            // the metric, its factor L, L^-1
            let (chol, _) = source.metric_factor(gpu)?;
            let mut li = {
                let mut id = vec![0.0; naux * naux];
                for p in 0..naux {
                    id[p * naux + p] = 1.0;
                }
                gpu.upload(&id)?
            };
            gemm.trsm_lower_left(
                Op::N,
                (naux, naux),
                MatRef::new(&chol, 0, naux),
                MatMut::new(&mut li, 0, naux),
            )?;
            let linv = li;
            // G^k (host): g[k][P][Q] = (∂_k P|Q), the derivative with respect to P's center
            let ap = PairSet::aux(aux, PairOptions::with_order(DerivOrder::Two));
            let mut gh = vec![0.0; 3 * naux * naux];
            let metric = Eri2DerivOptions {
                thresh: DfThresholds::default().metric_deriv,
                op: Operator::Coulomb,
            };
            Eri2DerivPlan::new(aux, &ap, DerivOrder::One, metric)
                .expect("auxiliary pairs of order 2")
                .deriv_integrals(&mut gh);
            let g = gpu.upload(&gh)?;
            // gt^k = L^-1 G^k^T: G^k row-major read column-major is G^k^T
            let mut gt = gpu.upload(&gh)?;
            drop(gh);
            for k in 0..3 {
                gemm.trsm_lower_left(
                    Op::N,
                    (naux, naux),
                    MatRef::new(&chol, 0, naux),
                    MatMut::new(&mut gt, k * naux * naux, naux),
                )?;
            }
            let i32v = |v: Vec<usize>| -> Vec<i32> { v.into_iter().map(|x| x as i32).collect() };
            let natom = plan.natom;
            let ao0 = gpu.upload(&i32v(plan.aor.iter().map(|r| r.start).collect()))?;
            let ao1 = gpu.upload(&i32v(plan.aor.iter().map(|r| r.end).collect()))?;
            let xr0 = gpu.upload(&i32v(plan.xr.iter().map(|r| r.start).collect()))?;
            let xr1 = gpu.upload(&i32v(plan.xr.iter().map(|r| r.end).collect()))?;
            let aux_atom = gpu.upload(&i32v(plan.aux_atom.clone()))?;
            let sym_atoms = SymRows {
                k: gpu.upload(&(0..3 * natom).map(|x| (x % 3) as i32).collect::<Vec<_>>())?,
                a0: gpu.upload(&i32v(
                    (0..3 * natom).map(|x| plan.aor[x / 3].start).collect(),
                ))?,
                a1: gpu.upload(&i32v((0..3 * natom).map(|x| plan.aor[x / 3].end).collect()))?,
                len: 3 * natom,
            };
            let sym_all = SymRows {
                k: gpu.upload(&[0i32, 1, 2])?,
                a0: gpu.upload(&[0i32; 3])?,
                a1: gpu.upload(&[nbf as i32; 3])?,
                len: 3,
            };
            let rows = RowTables::upload(gpu, lay)?;
            gpu.synchronize()?;
            Ok(GpuDfHess {
                nbf,
                naux,
                ints,
                ders,
                hder,
                auxb,
                aux: aux.clone(),
                ap,
                chol,
                g,
                gt,
                linv,
                rows,
                ao0,
                ao1,
                xr0,
                xr1,
                aux_atom,
                sym_atoms,
                sym_all,
                gemm,
                kernels,
                rz_bytes: opts.rz_bytes,
                emu: None,
                emu_work: crate::emu::OzakiWork::default(),
                plan,
                workspace: None,
                setup_seconds: t0.elapsed().as_secs_f64(),
                last: GpuDfHessTimes::default(),
            })
        }

        /// Runs the exchange part of F^x C, sum_P W~_P R~^x_P (the largest GEMM), in emulated
        /// FP64 on INT8 tensor cores with `slices` slices (`crate::emu`'s Ozaki scheme; 7: 55
        /// bits), or natively with None.
        pub fn set_emulation(&mut self, gpu: &Gpu, slices: Option<usize>) -> Result<(), GpuError> {
            self.emu = match slices {
                Some(s) => Some((crate::emu::Ozaki::new(gpu)?, s)),
                None => None,
            };
            self.emu_work = crate::emu::OzakiWork::default();
            Ok(())
        }

        pub fn plan(&self) -> &DfHessPlan {
            &self.plan
        }

        /// The builders' erf kernels for the operator `op` (nothing without an erf part).
        pub fn prepare_operator(&mut self, gpu: &Gpu, op: &Operator) -> Result<(), GpuError> {
            if !op.has_erf() {
                return Ok(());
            }
            if op.omega() <= 0.0 || !op.omega().is_finite() {
                return Err(err("the range separation omega must be positive"));
            }
            let erf = Operator::LongRange { omega: op.omega() };
            self.ints.prepare_operator(gpu, &erf)?;
            self.ders.prepare_operator(gpu, &erf)?;
            self.hder.prepare_operator(gpu, &erf)
        }

        /// Device bytes of a call's workspaces with `ncols` occupied orbitals per channel
        /// (exchange: `with_k`), besides the setup's.
        pub fn workspace_bytes(&self, ncols: &[usize], with_k: bool, two_ops: bool) -> usize {
            let (n, naux) = (self.nbf, self.naux);
            let n3 = 3 * self.plan.natom;
            let nx = self.plan.layout().nx;
            let maxq = self.plan.aux_blocks.iter().map(|b| b.nq).max().unwrap_or(0);
            let sizes = self.plan.grad.max_block();
            let (dense, rows) = (sizes.dense, sizes.rows);
            let ops = if two_ops { 2 } else { 1 };
            let omax = ncols.iter().copied().max().unwrap_or(0);
            // MO pass, W/Z, the J terms
            let mut b = n * n + nx + 2 * naux + ops * rows * naux + dense * naux;
            b += 3 * naux * n3 + n3 * n3;
            for &o in ncols {
                b += n * o * naux + 3 * n3 * n * o + 2 * 3 * n * o;
            }
            // pass 1: planes, the dense block, H of the block, hq
            b += ops * 6 * nx * maxq + 3 * n * n * maxq + 3 * n * maxq * omax + 3 * maxq * n;
            if with_k {
                for &o in ncols {
                    let npk = o * (o + 1) / 2;
                    // X and its packed and [i][j][P] copies, W~p, H (then H~), E, the Gram
                    b += 2 * naux * o * o + naux * npk + n * naux * o + 3 * n * o * naux;
                    b += 3 * naux * npk + n3 * n3;
                }
                b += n * maxq * omax + n * n * omax + naux * naux + self.rz_bytes / 8;
            }
            8 * b
        }

        /// Sets aside `bytes` for the calls' workspaces (the largest [`GpuDfHess::workspace_bytes`]
        /// of the calls planned) as the memory plan's grant, which their allocations draw from;
        /// an out-of-memory error when they do not fit.
        pub fn grant_workspace(&mut self, gpu: &Gpu, bytes: usize) -> Result<(), GpuError> {
            self.workspace = None;
            self.workspace = Some(gpu.grant(bytes, "the Hessian's two-electron workspace")?);
            Ok(())
        }

        /// The calls' draws beyond the workspace's grant (plan violations).
        pub fn workspace_violations(&self) -> Vec<PlanViolation> {
            self.workspace
                .as_ref()
                .map(Grant::violations)
                .unwrap_or_default()
        }

        /// The terms for occupied orbitals given on the host: per channel C (nbf x ncol,
        /// column-major), weights `f` (closed shell \[2\], open shell \[1, 1\]), exchange
        /// fraction `a_x` of the operator `op`'s three-center integrals (1/r, or
        /// full/r + erf erf(omega r)/r), with the Coulomb terms when `want_j` (op = 1/r).
        pub fn compute(
            &mut self,
            gpu: &Gpu,
            occ: &[(&[f64], usize)],
            f: &[f64],
            a_x: f64,
            want_j: bool,
            op: &Operator,
        ) -> Result<GpuDfHessParts, GpuError> {
            let (cmo, ncols) = upload_orbitals(gpu, self.nbf, occ, err)?;
            self.compute_device(gpu, &cmo, &ncols, f, a_x, want_j, op)
        }

        /// As [`GpuDfHess::compute`] with the channels' orbitals on the device, side by side
        /// (nbf x sum(ncols), column-major). The device phases of the module docs run in
        /// order, each leaving the buffers later phases read; a buffer is freed after the
        /// phase that reads it last.
        #[allow(clippy::too_many_arguments)]
        pub fn compute_device(
            &mut self,
            gpu: &Gpu,
            cmo: &DeviceBuf<f64>,
            ncols: &[usize],
            f: &[f64],
            a_x: f64,
            want_j: bool,
            op: &Operator,
        ) -> Result<GpuDfHessParts, GpuError> {
            if want_j && !op.is_coulomb() {
                return Err(err("the Coulomb terms need the Coulomb operator"));
            }
            if f.len() != ncols.len() || ncols.is_empty() {
                return Err(err("one weight per channel expected"));
            }
            let n = self.nbf;
            let ot: usize = ncols.iter().sum();
            if cmo.len() < n * ot {
                return Err(err("orbital buffer too small"));
            }
            let with_k = a_x != 0.0 && ot > 0;
            // the operator of the three-center integrals: coul (mn|P) + erfc (mn|erf|P)
            let (coul, erfc) = (op.coulomb(), op.long_range());
            let erf_op = if op.has_erf() {
                self.prepare_operator(gpu, op)?;
                Some(Operator::LongRange { omega: op.omega() })
            } else {
                None
            };
            let c0s: Vec<usize> = ncols
                .iter()
                .scan(0, |a, &o| {
                    let v = *a;
                    *a += o;
                    Some(v * n)
                })
                .collect();
            let t_all = Instant::now();
            let lay = self.plan.layout();
            let (nx, npair) = (lay.nx, lay.pairs.len());
            let mut c = HessCall {
                gpu,
                cmo,
                ncols,
                f,
                c0s,
                a_x,
                want_j,
                with_k,
                coul,
                erfc,
                erf_op,
                n,
                naux: self.naux,
                natom: self.plan.natom,
                n3: 3 * self.plan.natom,
                nx,
                npair,
                maxq: self.plan.aux_blocks.iter().map(|b| b.nq).max().unwrap_or(0),
                omax: ncols.iter().copied().max().unwrap_or(0),
                // the Ozaki scheme's scratch (back in place at the end)
                emu_work: std::mem::take(&mut self.emu_work),
                tl: Timeline::new(gpu),
            };
            let MoPass { gamma, w, dens } = self.mo_pass(&mut c)?;
            let mut fit = self.fit(&mut c, gamma, w)?;
            let mut acc = self.pass1(&mut c, &mut fit)?;
            let hess_j = self.coulomb_terms(&mut c, &mut fit, &mut acc)?;
            let gram = self.exchange(&mut c, &fit, &mut acc)?;
            // H~ and W~ are the exchange's; X, its [i][j][P] copy and D the second derivatives'
            drop((std::mem::take(&mut acc.hst), std::mem::take(&mut fit.zp)));
            let d2 = self.second_derivatives(&mut c, &fit, dens.as_ref())?;
            drop((
                std::mem::take(&mut fit.x),
                std::mem::take(&mut fit.xt),
                dens,
            ));
            let (parts, mut times) = self.assemble(&c, &acc, hess_j, &gram, d2)?;
            self.emu_work = c.emu_work;
            times.total = t_all.elapsed().as_secs_f64();
            self.last = times;
            Ok(parts)
        }

        /// b = L^-1 b (`Op::N`) or L^-T b (`Op::T`) for `ncol` columns of naux.
        fn trsm(&self, op: Op, b: &mut DeviceBuf<f64>, ncol: usize) -> Result<(), GpuError> {
            let naux = self.naux;
            self.gemm.trsm_lower_left(
                op,
                (naux, ncol),
                MatRef::new(&self.chol, 0, naux),
                MatMut::new(b, 0, naux),
            )
        }

        /// The MO pass (step 1 of the module docs) over the blocks of whole first shells:
        /// (mn|op|P), gamma (with the Coulomb terms) and W_s.
        fn mo_pass(&self, c: &mut HessCall) -> Result<MoPass, GpuError> {
            let (gpu, cmo, ncols, f) = (c.gpu, c.cmo, c.ncols, c.f);
            let (n, naux, nx) = (c.n, c.naux, c.nx);
            let (want_j, coul, erfc) = (c.want_j, c.coul, c.erfc);
            let (c0s, erf_op) = (&c.c0s, &c.erf_op);
            let (n2, nch) = (n * n, ncols.len());
            let gemm = &self.gemm;
            let kr = &self.kernels;
            // ---- MO pass: gamma and W_s[m][i][P] = sum_n (mn|op|P) C_ni ----
            let s0 = c.tl.mark()?;
            let mut gamma = gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), naux)?;
            let mut wrow = gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), nx)?;
            // the total density (the Coulomb terms; kept for the second derivatives' Gamma)
            let mut dens_keep = None;
            if want_j {
                let mut dens = gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), n2)?;
                for s in 0..nch {
                    if ncols[s] > 0 {
                        gemm.gemm(
                            Arith::Fp64,
                            GemmDims::new(Op::N, Op::T, n, n, ncols[s]),
                            (f[s], 1.0),
                            MatRef::new(cmo, c0s[s], n),
                            MatRef::new(cmo, c0s[s], n),
                            MatMut::new(&mut dens, 0, n),
                        )?;
                    }
                }
                // SAFETY: df_density_rows: nx rows of the tables, one n^2 density, nx values.
                unsafe {
                    launch(
                        gpu,
                        &kr.density_rows,
                        nx,
                        &[
                            Arg::I64(nx as i64),
                            Arg::I32(1),
                            Arg::I64(n as i64),
                            Arg::buf(&self.rows.m),
                            Arg::buf(&self.rows.n),
                            Arg::buf(&self.rows.kind),
                            Arg::buf(&dens),
                            Arg::buf_mut(&mut wrow),
                        ],
                    )?
                };
                dens_keep = Some(dens);
            }
            let mut w: Vec<DeviceBuf<f64>> = ncols
                .iter()
                .map(|&o| {
                    gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), (n * o * naux).max(1))
                })
                .collect::<Result<_, _>>()?;
            {
                let sizes = self.plan.grad.max_block();
                let (max_rows, max_dense) = (sizes.rows, sizes.dense);
                let mut ablk =
                    gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), max_rows * naux)?;
                let mut abl2 = if erf_op.is_some() && coul != 0.0 {
                    Some(gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), max_rows * naux)?)
                } else {
                    None
                };
                let mut ad = gpu.alloc_zeros_granted::<f64>(
                    self.workspace.as_ref(),
                    (max_dense * naux).max(1),
                )?;
                for (bi, b) in self.plan.grad.blocks.iter().enumerate() {
                    let rows = b.x1 - b.x0;
                    // (mn|op|P) = coul (mn|P) + erfc (mn|erf|P)
                    match (erf_op, abl2.as_mut()) {
                        (None, _) => {
                            self.ints
                                .compute(gpu, b.p0..b.p1, &Operator::Coulomb, &mut ablk, 0)?
                        }
                        (Some(e), None) => {
                            self.ints.compute(gpu, b.p0..b.p1, e, &mut ablk, 0)?;
                            kr.blas1.scale(
                                gpu,
                                rows * naux,
                                Scale::Scalar(erfc),
                                Operand::Out,
                                &mut ablk,
                            )?;
                        }
                        (Some(e), Some(a2)) => {
                            self.ints
                                .compute(gpu, b.p0..b.p1, &Operator::Coulomb, &mut ablk, 0)?;
                            self.ints.compute(gpu, b.p0..b.p1, e, a2, 0)?;
                            // ablk = coul ablk + erfc a2, rounded as fma(coul, ablk, erfc a2)
                            kr.blas1.axpby(
                                gpu,
                                rows * naux,
                                &[(erfc, Operand::Buf(a2)), (coul, Operand::Out)],
                                &mut ablk,
                            )?;
                        }
                    }
                    if erf_op.is_none() && coul != 1.0 {
                        kr.blas1.scale(
                            gpu,
                            rows * naux,
                            Scale::Scalar(coul),
                            Operand::Out,
                            &mut ablk,
                        )?;
                    }
                    if want_j {
                        gemm.gemm(
                            Arith::Fp64,
                            GemmDims::new(Op::N, Op::N, naux, 1, rows),
                            (1.0, if bi == 0 { 0.0 } else { 1.0 }),
                            MatRef::new(&ablk, 0, naux),
                            MatRef::new(&wrow, b.x0, rows),
                            MatMut::new(&mut gamma, 0, naux),
                        )?;
                    }
                    let (m0, m1) = (b.m0, b.m1);
                    let nm = m1 - m0;
                    gpu.zero_range(&mut ad, 0, nm * m1 * naux)?;
                    let blk = RowBlock {
                        rows: &self.rows,
                        x0: b.x0,
                        nrows: rows,
                        naux,
                        m0,
                        m1,
                    };
                    // SAFETY: rows x0..x1 of the tables and of ablk; writes ((m - m0) m1 + n)
                    // naux + P < nm m1 naux (m in [m0, m1), n < m1 for the block's pairs, and
                    // the transposed entry only for n in [m0, m1)).
                    unsafe {
                        kr.deriv
                            .unpack(gpu, blk, Unpack::Mirrored, &ablk, &mut ad)?
                    };
                    for s in 0..nch {
                        let o = ncols[s];
                        if o == 0 {
                            continue;
                        }
                        // W[m] += Ad[m - m0] C[0..m1] for the block's rows m
                        gemm.gemm_strided(
                            Arith::Fp64,
                            GemmDims::new(Op::N, Op::N, naux, o, m1).batch(nm),
                            (1.0, 1.0),
                            MatRef::new(&ad, 0, naux).stride(m1 * naux),
                            MatRef::new(cmo, c0s[s], n),
                            MatMut::new(&mut w[s], m0 * o * naux, naux).stride(o * naux),
                        )?;
                        // W[n] += sum_{m in block} (mn|P) C_m for the earlier functions n
                        if m0 > 0 {
                            gemm.gemm_strided(
                                Arith::Fp64,
                                GemmDims::new(Op::N, Op::N, naux, o, nm).batch(m0),
                                (1.0, 1.0),
                                MatRef::new(&ad, 0, m1 * naux).stride(naux),
                                MatRef::new(cmo, c0s[s] + m0, n),
                                MatMut::new(&mut w[s], 0, naux).stride(o * naux),
                            )?;
                        }
                    }
                }
            }
            drop(wrow);
            c.tl.close(|ts| &mut ts.mo, s0)?;
            Ok(MoPass {
                gamma,
                w,
                dens: dens_keep,
            })
        }

        /// The fit (step 2): c and Z (with W~) by triangular solves, X_s, and V^x c with the
        /// Coulomb terms.
        fn fit(
            &self,
            c: &mut HessCall,
            mut gamma: DeviceBuf<f64>,
            mut w: Vec<DeviceBuf<f64>>,
        ) -> Result<Fit, GpuError> {
            let (gpu, cmo, ncols) = (c.gpu, c.cmo, c.ncols);
            let (n, naux, natom, n3) = (c.n, c.naux, c.natom, c.n3);
            let (want_j, with_k) = (c.want_j, c.with_k);
            let c0s = &c.c0s;
            let nch = ncols.len();
            let gemm = &self.gemm;
            let kr = &self.kernels;
            // ---- fit: c, Z = V^-1 W (with W~ = L^-1 W kept as [m][P][i]), X ----
            let s1 = c.tl.mark()?;
            if want_j {
                self.trsm(Op::N, &mut gamma, 1)?;
                self.trsm(Op::T, &mut gamma, 1)?;
            }
            let cfit = gamma;
            // W~ for the exchange's sum_P Z_P R^x_P = sum_P W~_P (L^-1 R^x)_P
            let mut zp: Vec<Option<DeviceBuf<f64>>> = Vec::new();
            for s in 0..nch {
                let o = ncols[s];
                if o == 0 {
                    zp.push(None);
                    continue;
                }
                self.trsm(Op::N, &mut w[s], n * o)?;
                if with_k {
                    let mut zps =
                        gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), n * naux * o)?;
                    // SAFETY: hd_ztrans reads and writes n o naux.
                    unsafe {
                        launch(
                            gpu,
                            &kr.ztrans,
                            n * o * naux,
                            &[
                                Arg::I64(n as i64),
                                Arg::I32(o as i32),
                                Arg::I32(naux as i32),
                                Arg::buf(&w[s]),
                                Arg::buf_mut(&mut zps),
                            ],
                        )?
                    };
                    zp.push(Some(zps));
                } else {
                    zp.push(None);
                }
                self.trsm(Op::T, &mut w[s], n * o)?;
            }
            let z = w;
            let mut x: Vec<Option<DeviceBuf<f64>>> = Vec::new();
            let mut xp: Vec<Option<DeviceBuf<f64>>> = Vec::new();
            // X symmetrized in the [i][j][P] layout (the second derivatives' Zd)
            let mut xt: Vec<Option<DeviceBuf<f64>>> = Vec::new();
            for s in 0..nch {
                let o = ncols[s];
                if !with_k || o == 0 {
                    x.push(None);
                    xp.push(None);
                    xt.push(None);
                    continue;
                }
                let npk = o * (o + 1) / 2;
                let mut tt =
                    gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), o * o * naux)?;
                // T[i][j][P] = sum_m C_mi Z[m][j][P]
                gemm.gemm(
                    Arith::Fp64,
                    GemmDims::new(Op::N, Op::N, o * naux, o, n),
                    (1.0, 0.0),
                    MatRef::new(&z[s], 0, o * naux),
                    MatRef::new(cmo, c0s[s], n),
                    MatMut::new(&mut tt, 0, o * naux),
                )?;
                let mut xs =
                    gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), naux * o * o)?;
                let mut xps =
                    gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), naux * npk)?;
                // SAFETY: hd_xsym reads o^2 naux of tt, writes naux o^2; hd_xpack reads naux o^2,
                // writes naux npk.
                unsafe {
                    launch(
                        gpu,
                        &kr.xsym,
                        naux * o * o,
                        &[
                            Arg::I32(o as i32),
                            Arg::I32(naux as i32),
                            Arg::buf(&tt),
                            Arg::buf_mut(&mut xs),
                        ],
                    )?;
                    launch(
                        gpu,
                        &kr.xpack,
                        naux * npk,
                        &[
                            Arg::I32(o as i32),
                            Arg::I32(naux as i32),
                            Arg::buf(&xs),
                            Arg::buf_mut(&mut xps),
                        ],
                    )?;
                }
                // SAFETY: hd_tsym: npk naux threads, each the pair (i, j), (j, i) of tt (o^2 naux).
                unsafe {
                    launch(
                        gpu,
                        &kr.tsym,
                        npk * naux,
                        &[
                            Arg::I32(o as i32),
                            Arg::I32(naux as i32),
                            Arg::buf_mut(&mut tt),
                        ],
                    )?
                };
                x.push(Some(xs));
                xp.push(Some(xps));
                xt.push(Some(tt));
            }
            // V^x c (J only)
            let (gx, vc) = if want_j {
                let mut gc = gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), 3 * naux)?;
                for k in 0..3 {
                    gemm.gemm(
                        Arith::Fp64,
                        GemmDims::new(Op::T, Op::N, naux, 1, naux),
                        (1.0, 0.0),
                        MatRef::new(&self.g, k * naux * naux, naux),
                        MatRef::new(&cfit, 0, naux),
                        MatMut::new(&mut gc, k * naux, naux),
                    )?;
                }
                let mut vc = gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), n3 * naux)?;
                // SAFETY: hd_vc: n3 naux outputs; g 3 naux^2, gc 3 naux, c naux, the atom tables.
                unsafe {
                    launch(
                        gpu,
                        &kr.vc,
                        n3 * naux,
                        &[
                            Arg::I32(naux as i32),
                            Arg::I32(natom as i32),
                            Arg::buf(&self.xr0),
                            Arg::buf(&self.xr1),
                            Arg::buf(&self.aux_atom),
                            Arg::buf(&self.g),
                            Arg::buf(&gc),
                            Arg::buf(&cfit),
                            Arg::buf_mut(&mut vc),
                        ],
                    )?
                };
                (
                    Some(gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), n3 * naux)?),
                    Some(vc),
                )
            } else {
                (None, None)
            };
            c.tl.close(|ts| &mut ts.fit, s1)?;
            Ok(Fit {
                cfit,
                z,
                zp,
                x,
                xp,
                xt,
                gx,
                vc,
            })
        }

        /// Pass 1 (step 3) over the auxiliary blocks: the derivative integrals unpacked, H^k_s,
        /// gamma^x, J1, Y_s, the K terms and T^{A,k}, and H^k_s stored for all P.
        fn pass1(&self, c: &mut HessCall, fit: &mut Fit) -> Result<Pass1, GpuError> {
            // ---- pass 1 over auxiliary blocks ----
            let (mut w, mut acc) = self.pass1_buffers(c)?;
            for bi in 0..self.plan.aux_blocks.len() {
                self.pass1_block(c, fit, &mut w, &mut acc, bi)?;
            }
            drop(w);
            Ok(acc)
        }

        /// Pass 1's buffers, in the order allocated: the per-block workspace (freed at the
        /// pass's end) and what the pass leaves (the F^x C accumulators, J1 and the stored H).
        fn pass1_buffers(&self, c: &HessCall) -> Result<(Pass1Work, Pass1), GpuError> {
            let (gpu, ncols, erf_op) = (c.gpu, c.ncols, &c.erf_op);
            let (n, naux, n3) = (c.n, c.naux, c.n3);
            let (nx, maxq, omax) = (c.nx, c.maxq, c.omax);
            let (want_j, with_k, coul) = (c.want_j, c.with_k, c.coul);
            let n2 = n * n;
            let grant = self.workspace.as_ref();
            let alloc = |len: usize| gpu.alloc_zeros_granted::<f64>(grant, len);
            let dbuf = alloc(6 * nx * maxq)?;
            let dbuf2 = if erf_op.is_some() && coul != 0.0 {
                Some(alloc(6 * nx * maxq)?)
            } else {
                None
            };
            let ib = alloc(3 * n2 * maxq)?;
            let hs = alloc((3 * n * maxq * omax).max(1))?;
            let hq = if want_j {
                Some(alloc(3 * maxq * n)?)
            } else {
                None
            };
            let yt = if with_k {
                Some(alloc((n * maxq * omax).max(1))?)
            } else {
                None
            };
            // P[a] = I^k[a]^T Y[a] for every AO function a (n x o each)
            let pbuf = if with_k {
                Some(alloc((n * n * omax).max(1))?)
            } else {
                None
            };
            let alloc_v = |len: usize| -> Result<Vec<DeviceBuf<f64>>, GpuError> {
                ncols.iter().map(|&o| alloc((len * o).max(1))).collect()
            };
            // F^x C accumulators per channel: J side and K side (n x o per x, column-major), the
            // Z r term ([x][m][i]); J1 C and the KY rows (3 n o)
            let jacc = alloc_v(n3 * n)?;
            let kacc = alloc_v(n3 * n)?;
            let fb = alloc_v(n3 * n)?;
            let ky1 = alloc_v(3 * n)?;
            let j1c = alloc_v(3 * n)?;
            let j1 = if want_j { Some(alloc(3 * n2)?) } else { None };
            let j1b = if want_j { Some(alloc(3 * n2)?) } else { None };
            let mut hst: Vec<Option<DeviceBuf<f64>>> = Vec::new();
            for &o in ncols {
                hst.push(if with_k && o > 0 {
                    Some(alloc(3 * n * o * naux)?)
                } else {
                    None
                });
            }
            let work = Pass1Work {
                dbuf,
                dbuf2,
                ib,
                hs,
                hq,
                yt,
                j1b,
                pbuf,
            };
            let acc = Pass1 {
                jacc,
                kacc,
                fb,
                ky1,
                j1c,
                j1,
                hst,
            };
            Ok((work, acc))
        }

        /// Pass 1 over auxiliary block `bi`: its derivative integrals (timed as `deriv`), then
        /// J1, each channel's terms and gamma^x (`pass1`).
        fn pass1_block(
            &self,
            c: &mut HessCall,
            fit: &mut Fit,
            w: &mut Pass1Work,
            acc: &mut Pass1,
            bi: usize,
        ) -> Result<(), GpuError> {
            let blk = self.plan.aux_blocks[bi];
            let s = c.tl.mark()?;
            self.pass1_derivs(c, w, bi, blk.nq)?;
            let s1 = c.tl.close(|ts| &mut ts.deriv, s)?;
            self.pass1_j1(c, fit, w, acc, blk)?;
            for sch in 0..c.ncols.len() {
                if c.ncols[sch] == 0 {
                    continue;
                }
                self.pass1_channel(c, fit, w, acc, blk, sch)?;
            }
            self.pass1_gx(c, fit, w, blk)?;
            c.tl.close(|ts| &mut ts.pass1, s1)?;
            Ok(())
        }

        /// The derivative integrals of auxiliary block `bi` (`nq` functions): the six planes of
        /// every kept pair against the block's shells, of coul (mn|P) + erfc (mn|erf|P),
        /// unpacked to the dense I^k\[a\]\[q\]\[b\].
        fn pass1_derivs(
            &self,
            c: &HessCall,
            w: &mut Pass1Work,
            bi: usize,
            nq: usize,
        ) -> Result<(), GpuError> {
            let (gpu, n, nx, npair) = (c.gpu, c.n, c.nx, c.npair);
            let (coul, erfc, erf_op) = (c.coul, c.erfc, &c.erf_op);
            let n2 = n * n;
            let kr = &self.kernels;
            match (erf_op, w.dbuf2.as_mut()) {
                (None, _) => self.ders.compute_block(
                    gpu,
                    0..npair,
                    &self.auxb,
                    bi,
                    &Operator::Coulomb,
                    &mut w.dbuf,
                    0,
                )?,
                (Some(e), None) => {
                    self.ders
                        .compute_block(gpu, 0..npair, &self.auxb, bi, e, &mut w.dbuf, 0)?;
                    kr.blas1.scale(
                        gpu,
                        6 * nx * nq,
                        Scale::Scalar(erfc),
                        Operand::Out,
                        &mut w.dbuf,
                    )?;
                }
                (Some(e), Some(d2)) => {
                    self.ders.compute_block(
                        gpu,
                        0..npair,
                        &self.auxb,
                        bi,
                        &Operator::Coulomb,
                        &mut w.dbuf,
                        0,
                    )?;
                    self.ders
                        .compute_block(gpu, 0..npair, &self.auxb, bi, e, d2, 0)?;
                    // dbuf = coul dbuf + erfc d2, rounded as fma(coul, dbuf, erfc d2)
                    kr.blas1.axpby(
                        gpu,
                        6 * nx * nq,
                        &[(erfc, Operand::Buf(d2)), (coul, Operand::Out)],
                        &mut w.dbuf,
                    )?;
                }
            }
            if erf_op.is_none() && coul != 1.0 {
                kr.blas1.scale(
                    gpu,
                    6 * nx * nq,
                    Scale::Scalar(coul),
                    Operand::Out,
                    &mut w.dbuf,
                )?;
            }
            gpu.zero_range(&mut w.ib, 0, 3 * n2 * nq)?;
            // SAFETY: hd_unpack_der: nx rows x nq of six planes; writes ((k n + a) nq + q) n
            // + b < 3 n^2 nq (a, b < n from the row tables, q < nq).
            unsafe {
                launch(
                    gpu,
                    &kr.unpack_der,
                    nx * nq,
                    &[
                        Arg::I64(nx as i64),
                        Arg::I32(nq as i32),
                        Arg::I64(n as i64),
                        Arg::buf(&self.rows.m),
                        Arg::buf(&self.rows.n),
                        Arg::buf(&self.rows.kind),
                        Arg::buf(&w.dbuf),
                        Arg::buf_mut(&mut w.ib),
                    ],
                )?
            };
            Ok(())
        }

        /// J1 (+)= sum_q c_q I_q of the block per (k, a) and J1b, the block's own (with the
        /// Coulomb terms): batched GEMVs.
        fn pass1_j1(
            &self,
            c: &HessCall,
            fit: &Fit,
            w: &mut Pass1Work,
            acc: &mut Pass1,
            blk: AuxAtomBlock,
        ) -> Result<(), GpuError> {
            let (n, p0, nq) = (c.n, blk.p0, blk.nq);
            if let (Some(j1), Some(j1b)) = (acc.j1.as_mut(), w.j1b.as_mut()) {
                for (beta, out) in [(1.0, &mut *j1), (0.0, &mut *j1b)] {
                    self.gemm.gemm_strided(
                        Arith::Fp64,
                        GemmDims::new(Op::N, Op::N, n, 1, nq).batch(3 * n),
                        (1.0, beta),
                        MatRef::new(&w.ib, 0, n).stride(nq * n),
                        MatRef::new(&fit.cfit, p0, nq),
                        MatMut::new(out, 0, n).stride(n),
                    )?;
                }
            }
            Ok(())
        }

        /// Channel `sch`'s terms of the block: H_s = I C_s, its share of hq (with the Coulomb
        /// terms), the auxiliary-centre part of M^x C, the exchange's terms
        /// ([`GpuDfHess::pass1_k_terms`]) and H_s stored for all P.
        fn pass1_channel(
            &self,
            c: &HessCall,
            fit: &Fit,
            w: &mut Pass1Work,
            acc: &mut Pass1,
            blk: AuxAtomBlock,
            sch: usize,
        ) -> Result<(), GpuError> {
            let (gpu, cmo, ncols, f) = (c.gpu, c.cmo, c.ncols, c.f);
            let (n, naux) = (c.n, c.naux);
            let (p0, nq, ablk) = (blk.p0, blk.nq, blk.atom);
            let n2 = n * n;
            let (o, cc) = (ncols[sch], c.c0s[sch]);
            let gemm = &self.gemm;
            let kr = &self.kernels;
            // H_s[k][a][q][i] = sum_b I[k][a][q][b] C_bi
            gemm.gemm(
                Arith::Fp64,
                GemmDims::new(Op::T, Op::N, o, 3 * n * nq, n),
                (1.0, 0.0),
                MatRef::new(cmo, cc, n),
                MatRef::new(&w.ib, 0, n),
                MatMut::new(&mut w.hs, 0, o),
            )?;
            if let Some(hq) = w.hq.as_mut() {
                // the block's first channel with orbitals sets hq, the others add to it
                let first = ncols[..sch].iter().all(|&o| o == 0);
                // SAFETY: hd_hq: 3 nq n outputs; hs 3 n nq o, c n o.
                unsafe {
                    launch(
                        gpu,
                        &kr.hq,
                        3 * nq * n,
                        &[
                            Arg::I32(nq as i32),
                            Arg::I64(n as i64),
                            Arg::I32(o as i32),
                            Arg::F64(f[sch]),
                            Arg::buf(&w.hs),
                            Arg::from(cmo.at(cc)),
                            Arg::I32((!first) as i32),
                            Arg::buf_mut(hq),
                        ],
                    )?
                };
            }
            if let Some(j1b) = w.j1b.as_ref() {
                // the auxiliary-centre part of M^x C: -(J1b + J1b^T) C for x = (ablk, k)
                for k in 0..3 {
                    let xo = (3 * ablk + k) * n * o;
                    for op in [Op::T, Op::N] {
                        gemm.gemm(
                            Arith::Fp64,
                            GemmDims::new(op, Op::N, n, o, n),
                            (-1.0, 1.0),
                            MatRef::new(j1b, k * n2, n),
                            MatRef::new(cmo, cc, n),
                            MatMut::new(&mut acc.jacc[sch], xo, n),
                        )?;
                    }
                }
            }
            if let Some(xs) = fit.x[sch].as_ref() {
                self.pass1_k_terms(c, xs, w, acc, blk, sch)?;
            }
            if let Some(h) = acc.hst[sch].as_mut() {
                // SAFETY: hd_scatter_h: reads 3 n nq o of hs, writes rows p0..p0 + nq of
                // the 3 n o columns of h (naux each).
                unsafe {
                    launch(
                        gpu,
                        &kr.scatter_h,
                        3 * n * nq * o,
                        &[
                            Arg::I32(nq as i32),
                            Arg::I64(n as i64),
                            Arg::I32(o as i32),
                            Arg::I32(naux as i32),
                            Arg::I32(p0 as i32),
                            Arg::buf(&w.hs),
                            Arg::buf_mut(h),
                        ],
                    )?
                };
            }
            Ok(())
        }

        /// The exchange's terms of channel `sch` in the block (its X_s given; with exchange):
        /// Y = C X per q, KY = sum_q H^k_q X_q into the KY rows and, negated, into
        /// x = (ablk, k), and T^{A,k}.
        fn pass1_k_terms(
            &self,
            c: &HessCall,
            xs: &DeviceBuf<f64>,
            w: &mut Pass1Work,
            acc: &mut Pass1,
            blk: AuxAtomBlock,
            sch: usize,
        ) -> Result<(), GpuError> {
            let Some(yt) = w.yt.as_mut() else {
                return Ok(());
            };
            let (gpu, cmo, n, natom) = (c.gpu, c.cmo, c.n, c.natom);
            let (p0, nq, ablk) = (blk.p0, blk.nq, blk.atom);
            let (o, cc) = (c.ncols[sch], c.c0s[sch]);
            let gemm = &self.gemm;
            // Y[a][q][i] = sum_j C_aj X_q,ji per q
            gemm.gemm_strided(
                Arith::Fp64,
                GemmDims::new(Op::N, Op::T, o, n, o).batch(nq),
                (1.0, 0.0),
                MatRef::new(xs, p0 * o * o, o).stride(o * o),
                MatRef::new(cmo, cc, n),
                MatMut::new(yt, 0, nq * o).stride(o),
            )?;
            for k in 0..3 {
                // KY = sum_q H^k_q X_q: into the KY rows (all) and, negated, into
                // x = (ablk, k)
                for (alpha, out, off) in [
                    (1.0, &mut acc.ky1[sch], k * n * o),
                    (-1.0, &mut acc.kacc[sch], (3 * ablk + k) * n * o),
                ] {
                    gemm.gemm(
                        Arith::Fp64,
                        GemmDims::new(Op::T, Op::T, n, o, nq * o),
                        (alpha, 1.0),
                        MatRef::new(&w.hs, k * n * nq * o, nq * o),
                        MatRef::new(xs, p0 * o * o, o),
                        MatMut::new(out, off, n),
                    )?;
                }
                // T^{A,k} = I^k[A rows]^T Y[A rows]: per AO function a the product
                // P[a] = I^k[a]^T Y[a] (one batched GEMM), summed over each atom's
                // functions into x = (A, k), the total subtracted for x = (ablk, k)
                let pb = w.pbuf.as_mut().ok_or_else(|| err("T buffer"))?;
                gemm.gemm_strided(
                    Arith::Fp64,
                    GemmDims::new(Op::N, Op::T, n, o, nq).batch(n),
                    (1.0, 0.0),
                    MatRef::new(&w.ib, k * n * nq * n, n).stride(nq * n),
                    MatRef::new(yt, 0, o).stride(nq * o),
                    MatMut::new(pb, 0, n).stride(n * o),
                )?;
                // SAFETY: hd_tseg: natom n o outputs at (3 A + k) n o + e < n3 n o
                // of kacc; reads n n o of P and the atom tables.
                unsafe {
                    launch(
                        gpu,
                        &self.kernels.tseg,
                        natom * n * o,
                        &[
                            Arg::I32(natom as i32),
                            Arg::I64(n as i64),
                            Arg::I32(o as i32),
                            Arg::I32(k as i32),
                            Arg::I32(ablk as i32),
                            Arg::buf(&self.ao0),
                            Arg::buf(&self.ao1),
                            Arg::buf(pb),
                            Arg::buf_mut(&mut acc.kacc[sch]),
                        ],
                    )?
                };
            }
            Ok(())
        }

        /// gamma^x of the block's auxiliary functions from hq (with the Coulomb terms).
        fn pass1_gx(
            &self,
            c: &HessCall,
            fit: &mut Fit,
            w: &Pass1Work,
            blk: AuxAtomBlock,
        ) -> Result<(), GpuError> {
            let (n, naux, natom) = (c.n, c.naux, c.natom);
            let (p0, nq, ablk) = (blk.p0, blk.nq, blk.atom);
            if let (Some(hq), Some(gx)) = (w.hq.as_ref(), fit.gx.as_mut()) {
                // SAFETY: hd_gx: 3 nq natom outputs at rows (3 a + k) naux + p0 + q < n3 naux;
                // hq 3 nq n.
                unsafe {
                    launch(
                        c.gpu,
                        &self.kernels.gx,
                        3 * nq * natom,
                        &[
                            Arg::I32(nq as i32),
                            Arg::I64(n as i64),
                            Arg::I32(natom as i32),
                            Arg::I32(naux as i32),
                            Arg::I32(p0 as i32),
                            Arg::I32(ablk as i32),
                            Arg::buf(&self.ao0),
                            Arg::buf(&self.ao1),
                            Arg::buf(hq),
                            Arg::buf_mut(gx),
                        ],
                    )?
                };
            }
            Ok(())
        }

        /// The J terms (step 4): r = gamma^x - V^x c, sum_P Z_P r_P, J1 C and the Coulomb
        /// cross term (as an n3 x n3 matrix); frees gamma^x (then r), V^x c and J1.
        fn coulomb_terms(
            &self,
            c: &mut HessCall,
            fit: &mut Fit,
            acc: &mut Pass1,
        ) -> Result<Option<DeviceBuf<f64>>, GpuError> {
            let (gpu, cmo, ncols) = (c.gpu, c.cmo, c.ncols);
            let (n, naux, natom, n3) = (c.n, c.naux, c.natom, c.n3);
            let c0s = &c.c0s;
            let (n2, nch) = (n * n, ncols.len());
            let gemm = &self.gemm;
            let kr = &self.kernels;
            // ---- J terms: r = gamma^x - V^x c, sum_P Z_P r_P, r V^-1 r ----
            let s = c.tl.mark()?;
            let mut hess_j = None;
            if let (Some(rv), Some(vc), Some(j1)) =
                (fit.gx.as_mut(), fit.vc.as_ref(), acc.j1.as_ref())
            {
                // r = fma(-1, V^x c, gamma^x)
                kr.blas1.axpby(
                    gpu,
                    n3 * naux,
                    &[(1.0, Operand::Out), (-1.0, Operand::Buf(vc))],
                    rv,
                )?;
                for sch in 0..nch {
                    let o = ncols[sch];
                    if o == 0 {
                        continue;
                    }
                    let cc = c0s[sch];
                    // Fb[x][m][i] = sum_P Z[m][i][P] r[x][P]
                    gemm.gemm(
                        Arith::Fp64,
                        GemmDims::new(Op::T, Op::N, n * o, n3, naux),
                        (1.0, 0.0),
                        MatRef::new(&fit.z[sch], 0, naux),
                        MatRef::new(rv, 0, naux),
                        MatMut::new(&mut acc.fb[sch], 0, n * o),
                    )?;
                    for k in 0..3 {
                        // J1^k C
                        gemm.gemm(
                            Arith::Fp64,
                            GemmDims::new(Op::T, Op::N, n, o, n),
                            (1.0, 0.0),
                            MatRef::new(j1, k * n2, n),
                            MatRef::new(cmo, cc, n),
                            MatMut::new(&mut acc.j1c[sch], k * n * o, n),
                        )?;
                        // J1^k[A rows]^T C[A rows] into x = (A, k)
                        for a in 0..natom {
                            let r = &self.plan.aor[a];
                            if r.is_empty() {
                                continue;
                            }
                            gemm.gemm(
                                Arith::Fp64,
                                GemmDims::new(Op::N, Op::N, n, o, r.len()),
                                (1.0, 1.0),
                                MatRef::new(j1, k * n2 + r.start * n, n),
                                MatRef::new(cmo, cc + r.start, n),
                                MatMut::new(&mut acc.jacc[sch], (3 * a + k) * n * o, n),
                            )?;
                        }
                    }
                }
                // Coulomb cross term (L^-1 r)^T (L^-1 r)
                gemm.trsm_lower_left(
                    Op::N,
                    (naux, n3),
                    MatRef::new(&self.chol, 0, naux),
                    MatMut::new(rv, 0, naux),
                )?;
                let mut hj = gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), n3 * n3)?;
                gemm.gemm(
                    Arith::Fp64,
                    GemmDims::new(Op::T, Op::N, n3, n3, naux),
                    (1.0, 0.0),
                    MatRef::new(rv, 0, naux),
                    MatRef::new(rv, 0, naux),
                    MatMut::new(&mut hj, 0, n3),
                )?;
                hess_j = Some(hj);
            }
            drop((fit.gx.take(), fit.vc.take(), acc.j1.take()));
            c.tl.close(|ts| &mut ts.solve, s)?;
            Ok(hess_j)
        }

        /// The exchange per channel (steps 5 and 6): the E pre-pass, H~ = L^-1 H in place, and
        /// pass 2 over the sub-chunks (R~^x, the Gram matrix, sum_P W~_P R~^x_P into the K
        /// side); the channels' Gram matrices sum_P R~^x . R~^y (n3 x n3).
        fn exchange(
            &self,
            c: &mut HessCall,
            fit: &Fit,
            acc: &mut Pass1,
        ) -> Result<Vec<Option<DeviceBuf<f64>>>, GpuError> {
            let (gpu, ncols, n3, with_k) = (c.gpu, c.ncols, c.n3, c.with_k);
            // ---- exchange: E^k for every P (from H), H~ = L^-1 H, then per sub-chunk
            // R~^x = L^-1 R^x for every x (b from H~, c from L^-1 G^T, a from L^-1 and E), the
            // Gram sum_P R~^x . R~^y = sum_P R^x V^-1 R^y and sum_P Z_P R^x_P = sum_P W~_P R~^x_P ----
            let mut gm: Vec<Option<DeviceBuf<f64>>> = Vec::new();
            for &o in ncols {
                gm.push(if with_k && o > 0 {
                    Some(gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), n3 * n3)?)
                } else {
                    None
                });
            }
            if with_k {
                for sch in 0..ncols.len() {
                    let o = ncols[sch];
                    if o == 0 {
                        continue;
                    }
                    let (Some(hs), Some(xps), Some(zps)) = (
                        acc.hst[sch].as_mut(),
                        fit.xp[sch].as_ref(),
                        fit.zp[sch].as_ref(),
                    ) else {
                        continue;
                    };
                    let ch = XChannel {
                        hs,
                        xps,
                        zps,
                        kacc: &mut acc.kacc[sch],
                        o,
                        cc: c.c0s[sch],
                    };
                    self.exchange_channel(c, ch, &mut gm[sch])?;
                }
            }
            Ok(gm)
        }

        /// One channel's exchange: its workspace, the E pre-pass from H
        /// ([`GpuDfHess::exchange_e`]), H~ = L^-1 H in place, then pass 2 over the sub-chunks of
        /// the auxiliary blocks ([`GpuDfHess::exchange_sub`]) into its Gram matrix `gm` and the K
        /// side of F^x C.
        fn exchange_channel(
            &self,
            c: &mut HessCall,
            mut ch: XChannel,
            gm: &mut Option<DeviceBuf<f64>>,
        ) -> Result<(), GpuError> {
            let (gpu, cmo) = (c.gpu, c.cmo);
            let (n, naux, n3, maxq) = (c.n, c.naux, c.n3, c.maxq);
            let o = ch.o;
            let npk = o * (o + 1) / 2;
            // sub-chunk size: R~ (n3 ns npk), U (ns o x n3 o), the gathered H (3 n ns o)
            let per_q = 8 * (n3 * npk + n3 * o * o + 3 * n * o);
            let ns_max = (self.rz_bytes / per_q.max(1)).clamp(1, maxq.max(1));
            let grant = self.workspace.as_ref();
            let rbuf = gpu.alloc_zeros_granted::<f64>(grant, n3 * ns_max * npk)?;
            let ubuf = gpu.alloc_zeros_granted::<f64>(grant, ns_max * o * n3 * o)?;
            let hg = gpu.alloc_zeros_granted::<f64>(grant, 3 * n * ns_max * o)?;
            // E^k[P][pk] for every P
            let ep = gpu.alloc_zeros_granted::<f64>(grant, 3 * naux * npk)?;
            let mut ws = XWork { ep, hg, ubuf, rbuf };
            let kx = XKernels {
                gpu,
                kr: &self.kernels,
                n,
                naux,
                o,
                npk,
                cptr: Arg::from(cmo.at(ch.cc)),
            };
            // the sub-chunks of the auxiliary blocks: (block's atom, first P, functions)
            let mut subs: Vec<(usize, usize, usize)> = Vec::new();
            for blk in &self.plan.aux_blocks {
                let mut q0 = 0;
                while q0 < blk.nq {
                    let ns = (blk.nq - q0).min(ns_max);
                    subs.push((blk.atom, blk.p0 + q0, ns));
                    q0 += ns;
                }
            }
            self.exchange_e(c, &kx, &ch, &mut ws, &subs)?;
            // H~ = L^-1 H in place
            let s = c.tl.mark()?;
            self.trsm(Op::N, ch.hs, 3 * n * o)?;
            c.tl.close(|ts| &mut ts.solve, s)?;
            let gms = gm.as_mut().ok_or_else(|| err("Gram buffer"))?;
            for &(_, pa, ns) in &subs {
                self.exchange_sub(c, &kx, &mut ch, &mut ws, gms, (pa, ns))?;
            }
            Ok(())
        }

        /// The E pre-pass (timed as `rz`): E^k = -(C^T H^k + h.c.) - G^k\[P rows\] Xp for every
        /// P, from H (before its solve).
        fn exchange_e(
            &self,
            c: &mut HessCall,
            kx: &XKernels,
            ch: &XChannel,
            ws: &mut XWork,
            subs: &[(usize, usize, usize)],
        ) -> Result<(), GpuError> {
            let (naux, npk) = (c.naux, kx.npk);
            let s = c.tl.mark()?;
            for &(_, pa, ns) in subs {
                kx.gather(ch.hs, &mut ws.hg, pa, ns)?;
                if ws.ep.len() < 3 * naux * npk || pa + ns > naux {
                    return Err(err("E buffer"));
                }
                kx.symc(
                    &self.sym_all,
                    -1.0,
                    &ws.hg,
                    ns,
                    ws.ep.at(pa * npk),
                    naux * npk,
                )?;
                for k in 0..3 {
                    self.gemm.gemm(
                        Arith::Fp64,
                        GemmDims::new(Op::N, Op::N, npk, ns, naux),
                        (-1.0, 1.0),
                        MatRef::new(ch.xps, 0, npk),
                        MatRef::new(&self.g, k * naux * naux + pa * naux, naux),
                        MatMut::new(&mut ws.ep, (k * naux + pa) * npk, npk),
                    )?;
                }
            }
            c.tl.close(|ts| &mut ts.rz, s)?;
            Ok(())
        }

        /// Pass 2 over the sub-chunk of `ns` functions from `pa`: R~^x for every x
        /// ([`GpuDfHess::exchange_r`]) and the Gram matrix sum_P R~^x . R~^y into `gms` (timed as
        /// `rz`), then sum_P W~_P R~^x_P into the K side ([`GpuDfHess::exchange_wr`], `t2`).
        fn exchange_sub(
            &self,
            c: &mut HessCall,
            kx: &XKernels,
            ch: &mut XChannel,
            ws: &mut XWork,
            gms: &mut DeviceBuf<f64>,
            (pa, ns): (usize, usize),
        ) -> Result<(), GpuError> {
            let len = ns * kx.npk;
            let s = c.tl.mark()?;
            self.exchange_r(c, kx, ch, ws, (pa, ns))?;
            // Gram: sum_P R~^x . R~^y
            self.gemm.gemm(
                Arith::Fp64,
                GemmDims::new(Op::T, Op::N, c.n3, c.n3, len),
                (1.0, 1.0),
                MatRef::new(&ws.rbuf, 0, len),
                MatRef::new(&ws.rbuf, 0, len),
                MatMut::new(gms, 0, c.n3),
            )?;
            let s1 = c.tl.mark()?;
            self.exchange_wr(c, ch, ws, (pa, ns))?;
            c.tl.span(|ts| &mut ts.rz, s, s1);
            c.tl.close(|ts| &mut ts.t2, s1)?;
            Ok(())
        }

        /// R~^x = L^-1 R^x of a sub-chunk for every x into the R buffer: the b part
        /// sym(C_A^T H~^k\[A rows\]) by the kernel, the c and a parts by GEMMs.
        fn exchange_r(
            &self,
            c: &HessCall,
            kx: &XKernels,
            ch: &XChannel,
            ws: &mut XWork,
            (pa, ns): (usize, usize),
        ) -> Result<(), GpuError> {
            let (naux, natom, npk) = (c.naux, c.natom, kx.npk);
            let len = ns * npk;
            kx.gather(ch.hs, &mut ws.hg, pa, ns)?;
            // b part: sym(C_A^T H~^k[A rows]) for every x
            kx.symc(&self.sym_atoms, 1.0, &ws.hg, ns, ws.rbuf.ptr(), len)?;
            for a in 0..natom {
                let r = &self.plan.xr[a];
                if r.is_empty() {
                    continue;
                }
                for k in 0..3 {
                    let xo = (3 * a + k) * len;
                    // c part: - sum_{Q in A} (L^-1 G^k^T)_PQ Xp_Q
                    self.gemm.gemm(
                        Arith::Fp64,
                        GemmDims::new(Op::N, Op::T, npk, ns, r.len()),
                        (-1.0, 1.0),
                        MatRef::new(ch.xps, r.start * npk, npk),
                        MatRef::new(&self.gt, k * naux * naux + pa + r.start * naux, naux),
                        MatMut::new(&mut ws.rbuf, xo, npk),
                    )?;
                    // a part: sum_{Q in A} (L^-1)_PQ E^k_Q (L^-1 is lower triangular: nothing
                    // for Q beyond the sub-chunk)
                    if r.start < pa + ns {
                        self.gemm.gemm(
                            Arith::Fp64,
                            GemmDims::new(Op::N, Op::T, npk, ns, r.len()),
                            (1.0, 1.0),
                            MatRef::new(&ws.ep, (k * naux + r.start) * npk, npk),
                            MatRef::new(&self.linv, pa + r.start * naux, naux),
                            MatMut::new(&mut ws.rbuf, xo, npk),
                        )?;
                    }
                }
            }
            Ok(())
        }

        /// sum_P W~_P R~^x_P of a sub-chunk into the K side: U = R~ unpacked, K^x C += W~p U
        /// (in emulated FP64 by the Ozaki scheme when set).
        fn exchange_wr(
            &self,
            c: &mut HessCall,
            ch: &mut XChannel,
            ws: &mut XWork,
            (pa, ns): (usize, usize),
        ) -> Result<(), GpuError> {
            let (gpu, n, naux, n3, o) = (c.gpu, c.n, c.naux, c.n3, ch.o);
            // SAFETY: hd_unpack_r: reads n3 ns npk of rbuf, writes ns o n3 o.
            unsafe {
                launch(
                    gpu,
                    &self.kernels.unpack_r,
                    ns * o * n3 * o,
                    &[
                        Arg::I32(n3 as i32),
                        Arg::I32(ns as i32),
                        Arg::I32(o as i32),
                        Arg::buf(&ws.rbuf),
                        Arg::buf_mut(&mut ws.ubuf),
                    ],
                )?
            };
            match &self.emu {
                Some((oz, slices)) => oz.dgemm_at(
                    gpu,
                    *slices,
                    Op::T,
                    Op::N,
                    n,
                    n3 * o,
                    ns * o,
                    1.0,
                    (ch.zps, pa * o, naux * o),
                    (&ws.ubuf, 0, ns * o),
                    1.0,
                    (&mut *ch.kacc, 0, n),
                    &mut c.emu_work,
                )?,
                None => self.gemm.gemm(
                    Arith::Fp64,
                    GemmDims::new(Op::T, Op::N, n, n3 * o, ns * o),
                    (1.0, 1.0),
                    MatRef::new(ch.zps, pa * o, naux * o),
                    MatRef::new(&ws.ubuf, 0, ns * o),
                    MatMut::new(ch.kacc, 0, n),
                )?,
            }
            Ok(())
        }

        /// The second derivatives (step 7) over the MO pass's blocks: Zd, Gamma in the pair-row
        /// layout and the second-derivative kernels into the fixed-point M; gamma_PQ of the
        /// two-center term.
        fn second_derivatives(
            &self,
            c: &mut HessCall,
            fit: &Fit,
            dens: Option<&DeviceBuf<f64>>,
        ) -> Result<SecondDerivs, GpuError> {
            let (gpu, cmo, ncols, f) = (c.gpu, c.cmo, c.ncols, c.f);
            let (n, naux, natom, n3, omax) = (c.n, c.naux, c.natom, c.n3, c.omax);
            let (a_x, want_j, with_k, coul, erfc) = (c.a_x, c.want_j, c.with_k, c.coul, c.erfc);
            let c0s = &c.c0s;
            let nch = ncols.len();
            let gemm = &self.gemm;
            let kr = &self.kernels;
            // ---- second derivatives: Gamma in the pair-row layout and the second-derivative
            // kernels (fixed-point M, H = (M + M^T) / 2); gamma_PQ of the two-center term ----
            let s = c.tl.mark()?;
            let natom_h = self.hder.host().natom;
            if natom_h != natom {
                return Err(err("the second-derivative builder counts other atoms"));
            }
            let mut hacc = DetAcc::new(gpu, n3 * n3)?;
            let mut gam2 = None;
            if want_j || with_k {
                let plan = &self.plan.grad;
                let sizes = plan.max_block();
                let (max_dense, max_mr, max_sub) = (sizes.dense, sizes.mrange, sizes.sub_rows);
                let mut zd = if with_k {
                    Some(gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), max_dense * naux)?)
                } else {
                    None
                };
                let mut tb = if with_k {
                    Some(gpu.alloc_zeros_granted::<f64>(
                        self.workspace.as_ref(),
                        (max_mr * omax * naux).max(1),
                    )?)
                } else {
                    None
                };
                let mut gbuf = gpu
                    .alloc_zeros_granted::<f64>(self.workspace.as_ref(), (max_sub * naux).max(1))?;
                // (factor, erf operator) of the launches: coul/r + erfc erf(omega r)/r
                let mut launches: Vec<(f64, Option<Operator>)> = Vec::new();
                match c.erf_op {
                    None => launches.push((coul, None)),
                    Some(e) => {
                        if coul != 0.0 {
                            launches.push((coul, None));
                        }
                        launches.push((erfc, Some(e)));
                    }
                }
                for b in &plan.blocks {
                    let (m0, m1) = (b.m0, b.m1);
                    let nm = m1 - m0;
                    if let (Some(zd), Some(t)) = (zd.as_mut(), tb.as_mut()) {
                        // Zd[m][n][P] = a_x sum_s f_s (C_s X_s,P C_s^T)_mn for the block's m, n < m1
                        let mut first = true;
                        for sch in 0..nch {
                            let o = ncols[sch];
                            let Some(xts) = fit.xt[sch].as_ref() else {
                                continue;
                            };
                            let cc = c0s[sch];
                            gemm.gemm(
                                Arith::Fp64,
                                GemmDims::new(Op::N, Op::T, o * naux, nm, o),
                                (a_x * f[sch], 0.0),
                                MatRef::new(xts, 0, o * naux),
                                MatRef::new(cmo, cc + m0, n),
                                MatMut::new(t, 0, o * naux),
                            )?;
                            gemm.gemm_strided(
                                Arith::Fp64,
                                GemmDims::new(Op::N, Op::T, naux, m1, o).batch(nm),
                                (1.0, if first { 0.0 } else { 1.0 }),
                                MatRef::new(t, 0, naux).stride(o * naux),
                                MatRef::new(cmo, cc, n),
                                MatMut::new(zd, 0, naux).stride(m1 * naux),
                            )?;
                            first = false;
                        }
                        if first {
                            gpu.zero_range(zd, 0, nm * m1 * naux)?;
                        }
                    }
                    for &(c0, c1) in &b.subs {
                        let rows = plan.chunk_rows(c0, c1);
                        let (p0, _) = plan.chunks[c0];
                        let (_, p1) = plan.chunks[c1 - 1];
                        let nr = rows.len();
                        let blk = RowBlock {
                            rows: &self.rows,
                            x0: rows.start,
                            nrows: nr,
                            naux,
                            m0,
                            m1,
                        };
                        for &(fct, ref eop) in &launches {
                            // the Coulomb term with the density (kept with want_j), the exchange
                            // term with Zd (with_k)
                            let terms = GammaTerms {
                                s: fct,
                                j: dens.map(|d| GammaJ {
                                    cfit: &fit.cfit,
                                    dens: d,
                                    nbf: n,
                                    symmetrize: false,
                                }),
                                zd: zd.as_ref(),
                            };
                            // SAFETY: nr naux outputs (<= gbuf); rows of the tables, D (n^2) and
                            // c (naux) with the Coulomb terms, Zd at ((m - m0) m1 + n) naux + P <
                            // nm m1 naux with exchange (m in [m0, m1), n < m1).
                            unsafe { kr.deriv.gamma(gpu, blk, terms, &mut gbuf)? };
                            let e = eop.as_ref().unwrap_or(&Operator::Coulomb);
                            self.hder.hessian(gpu, p0..p1, e, &gbuf, &mut hacc)?;
                        }
                    }
                }
                // gamma_PQ = -c_P c_Q / 2 + a_x/2 sum_s f_s <X_s,P, X_s,Q>
                let mut g2 =
                    gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), naux * naux)?;
                if want_j {
                    gemm.gemm(
                        Arith::Fp64,
                        GemmDims::new(Op::N, Op::T, naux, naux, 1),
                        (-0.5, 0.0),
                        MatRef::new(&fit.cfit, 0, naux),
                        MatRef::new(&fit.cfit, 0, naux),
                        MatMut::new(&mut g2, 0, naux),
                    )?;
                }
                for sch in 0..nch {
                    let o = ncols[sch];
                    if let Some(xs) = fit.x[sch].as_ref() {
                        gemm.gemm(
                            Arith::Fp64,
                            GemmDims::new(Op::T, Op::N, naux, naux, o * o),
                            (0.5 * a_x * f[sch], 1.0),
                            MatRef::new(xs, 0, o * o),
                            MatRef::new(xs, 0, o * o),
                            MatMut::new(&mut g2, 0, naux),
                        )?;
                    }
                }
                gam2 = Some(g2);
            }
            c.tl.close(|ts| &mut ts.d2, s)?;
            Ok(SecondDerivs { hacc, gam2 })
        }

        /// The downloads and the host assembly: F^x C per channel from its J and K sides, the
        /// explicit second derivative from the Coulomb cross term, the exchange's Gram
        /// matrices, the three-center M and the two-center term (host); the call's times
        /// (`total` left to the caller).
        fn assemble(
            &self,
            c: &HessCall,
            acc: &Pass1,
            hess_j: Option<DeviceBuf<f64>>,
            gram: &[Option<DeviceBuf<f64>>],
            d2: SecondDerivs,
        ) -> Result<(GpuDfHessParts, GpuDfHessTimes), GpuError> {
            let (gpu, ncols, f) = (c.gpu, c.ncols, c.f);
            let (n, n3, a_x, want_j, with_k) = (c.n, c.n3, c.a_x, c.want_j, c.with_k);
            let nch = ncols.len();
            // ---- downloads and the host assembly ----
            let th = Instant::now();
            let mut times = GpuDfHessTimes::default();
            c.tl.add_to(&mut times)?;
            let mut hess = match &hess_j {
                Some(hj) => gpu.download(hj)?,
                None => vec![0.0; n3 * n3],
            };
            // hj is symmetric (column-major = row-major)
            let mut fxo = Vec::with_capacity(nch);
            for sch in 0..nch {
                let o = ncols[sch];
                let mut out = vec![0.0; n3 * n * o];
                if o == 0 {
                    fxo.push(out);
                    continue;
                }
                let ja = gpu.download_range(&acc.jacc[sch], 0, n3 * n * o)?;
                let fbh = gpu.download_range(&acc.fb[sch], 0, n3 * n * o)?;
                let ka = gpu.download_range(&acc.kacc[sch], 0, n3 * n * o)?;
                let jc = gpu.download_range(&acc.j1c[sch], 0, 3 * n * o)?;
                let ky = gpu.download_range(&acc.ky1[sch], 0, 3 * n * o)?;
                for xi in 0..n3 {
                    let (a, k) = (xi / 3, xi % 3);
                    let rg = self.plan.aor[a].clone();
                    let base = xi * n * o;
                    for i in 0..o {
                        for m in 0..n {
                            let e = base + i * n + m;
                            let mut jv = ja[e];
                            let mut kv = ka[e];
                            if want_j {
                                jv += fbh[base + m * o + i];
                                if rg.contains(&m) {
                                    jv += jc[(k * o + i) * n + m];
                                }
                            }
                            if with_k && rg.contains(&m) {
                                kv += ky[(k * o + i) * n + m];
                            }
                            out[e] = if with_k { jv - a_x * kv } else { jv };
                        }
                    }
                }
                fxo.push(out);
                if with_k {
                    let Some(g1) = gram[sch].as_ref() else {
                        continue;
                    };
                    let gmh = gpu.download(g1)?;
                    // G = sum_P R~^x . R~^y (column-major), symmetrized
                    let gl = |i: usize, j: usize| gmh[i + j * n3];
                    for i in 0..n3 {
                        for j in 0..n3 {
                            hess[i * n3 + j] -= a_x * f[sch] * 0.5 * (gl(i, j) + gl(j, i));
                        }
                    }
                }
            }
            // the three-center second derivatives: H = (M + M^T) / 2
            let m = d2.hacc.to_f64(gpu)?;
            for i in 0..n3 {
                for j in 0..n3 {
                    hess[i * n3 + j] += 0.5 * (m[i * n3 + j] + m[j * n3 + i]);
                }
            }
            times.host = th.elapsed().as_secs_f64();
            // the two-center second derivatives (host, linus-ints' deterministic reduction)
            if let Some(g2) = d2.gam2 {
                let t2c = Instant::now();
                let g2h = gpu.download(&g2)?;
                let opts = Eri2DerivOptions {
                    thresh: DfThresholds::default().metric_deriv,
                    op: Operator::Coulomb,
                };
                let plan = Eri2DerivPlan::new(&self.aux, &self.ap, DerivOrder::Two, opts)
                    .expect("auxiliary pairs of order 2");
                let n3a = 3 * self.aux.natom();
                let mut h2 = vec![0.0; n3a * n3a];
                let gm = AoMatrix::row_major(&g2h, self.aux.nbf);
                plan.hessian(gm, &mut h2, &mut plan.workspace());
                hess.iter_mut().zip(&h2).for_each(|(a, b)| *a += b);
                times.d2_2c = t2c.elapsed().as_secs_f64();
            }
            Ok((GpuDfHessParts { fxo, hess }, times))
        }
    }

    /// hd_symc's outputs: per output y its direction k and its AO functions a0..a1.
    struct SymRows {
        k: DeviceBuf<i32>,
        a0: DeviceBuf<i32>,
        a1: DeviceBuf<i32>,
        /// The outputs.
        len: usize,
    }

    /// One call of [`GpuDfHess::compute_device`]: its inputs and the sizes its phases share,
    /// the engine's Ozaki scratch (back in place at the end) and the timeline of its phases.
    struct HessCall<'a> {
        gpu: &'a Gpu,
        /// The channels' occupied orbitals side by side (n x sum(ncols)), their columns, their
        /// weights f_s, and each channel's first element in `cmo`.
        cmo: &'a DeviceBuf<f64>,
        ncols: &'a [usize],
        f: &'a [f64],
        c0s: Vec<usize>,
        a_x: f64,
        /// The Coulomb terms; the exchange terms (a_x != 0 and an occupied orbital).
        want_j: bool,
        with_k: bool,
        /// The three-center integrals coul (mn|P) + erfc (mn|erf|P), and the erf operator when
        /// the operator has an erf part.
        coul: f64,
        erfc: f64,
        erf_op: Option<Operator>,
        n: usize,
        naux: usize,
        natom: usize,
        /// The perturbations x = (atom, k): 3 natom.
        n3: usize,
        /// The layout's rows and kept pairs.
        nx: usize,
        npair: usize,
        /// The most functions of an auxiliary block, the most orbitals of a channel.
        maxq: usize,
        omax: usize,
        emu_work: crate::emu::OzakiWork,
        tl: Timeline<'a, GpuDfHessTimes>,
    }

    /// What the MO pass leaves.
    struct MoPass {
        /// gamma_P = (P|mn) D_mn (with the Coulomb terms).
        gamma: DeviceBuf<f64>,
        /// Per channel W_s\[m\]\[i\]\[P\] = sum_n (mn|op|P) C_ni.
        w: Vec<DeviceBuf<f64>>,
        /// The total density (with the Coulomb terms), kept for the second derivatives' Gamma.
        dens: Option<DeviceBuf<f64>>,
    }

    /// What the fit leaves (per channel; the exchange's parts with exchange).
    struct Fit {
        /// c = V^-1 gamma.
        cfit: DeviceBuf<f64>,
        /// Z_s = V^-1 W_s (\[m\]\[i\]\[P\]); W~_s = L^-1 W_s as \[m\]\[P\]\[i\].
        z: Vec<DeviceBuf<f64>>,
        zp: Vec<Option<DeviceBuf<f64>>>,
        /// X_s (\[P\]\[i\]\[j\]), packed (\[P\]\[pk\]), and symmetrized as \[i\]\[j\]\[P\].
        x: Vec<Option<DeviceBuf<f64>>>,
        xp: Vec<Option<DeviceBuf<f64>>>,
        xt: Vec<Option<DeviceBuf<f64>>>,
        /// With the Coulomb terms: gamma^x (filled by pass 1; r = gamma^x - V^x c after the J
        /// terms) and V^x c.
        gx: Option<DeviceBuf<f64>>,
        vc: Option<DeviceBuf<f64>>,
    }

    /// What pass 1 leaves: the F^x C accumulators per channel, J1 and the stored H.
    struct Pass1 {
        /// n x o per perturbation x (column-major, x-major): the J side, the K side, and
        /// sum_P Z_P r^x_P (\[x\]\[m\]\[i\]).
        jacc: Vec<DeviceBuf<f64>>,
        kacc: Vec<DeviceBuf<f64>>,
        fb: Vec<DeviceBuf<f64>>,
        /// 3 n o: the KY rows and J1 C.
        ky1: Vec<DeviceBuf<f64>>,
        j1c: Vec<DeviceBuf<f64>>,
        /// J1^k = sum_P c_P I^k_P (with the Coulomb terms).
        j1: Option<DeviceBuf<f64>>,
        /// H^k_s for every P (\[k\]\[a\]\[i\]\[P\], with exchange); H~ after the exchange's solve.
        hst: Vec<Option<DeviceBuf<f64>>>,
    }

    /// Pass 1's per-block workspace, freed at the pass's end in the order listed.
    struct Pass1Work {
        /// The block's six derivative planes (and the erf operator's, for both parts).
        dbuf: DeviceBuf<f64>,
        dbuf2: Option<DeviceBuf<f64>>,
        /// The dense I^k\[a\]\[q\]\[b\], H_s of the block, and hq (with the Coulomb terms).
        ib: DeviceBuf<f64>,
        hs: DeviceBuf<f64>,
        hq: Option<DeviceBuf<f64>>,
        /// Y of the block (with exchange), J1b (the block's J1, with the Coulomb terms) and
        /// P\[a\] = I^k\[a\]^T Y\[a\] (with exchange).
        yt: Option<DeviceBuf<f64>>,
        j1b: Option<DeviceBuf<f64>>,
        pbuf: Option<DeviceBuf<f64>>,
    }

    /// One channel's part of the exchange: the stored H (solved in place to H~), X packed, W~
    /// (\[m\]\[P\]\[i\]), the K side of F^x C, its orbitals and its first element in `cmo`.
    struct XChannel<'b> {
        hs: &'b mut DeviceBuf<f64>,
        xps: &'b DeviceBuf<f64>,
        zps: &'b DeviceBuf<f64>,
        kacc: &'b mut DeviceBuf<f64>,
        o: usize,
        cc: usize,
    }

    /// One channel's exchange workspace: E^k for every P, a sub-chunk's gathered H, U and R~.
    /// Freed in the order listed, the reverse of their allocation.
    struct XWork {
        ep: DeviceBuf<f64>,
        hg: DeviceBuf<f64>,
        ubuf: DeviceBuf<f64>,
        rbuf: DeviceBuf<f64>,
    }

    /// The exchange's gather and symmetrized-product kernels for one channel: the device, the
    /// sizes and the channel's orbitals C (n x o, `cptr`).
    struct XKernels<'b> {
        gpu: &'b Gpu,
        kr: &'b Kernels,
        n: usize,
        naux: usize,
        o: usize,
        npk: usize,
        cptr: Arg,
    }

    impl XKernels<'_> {
        /// Rows pa..pa + ns of the 3 n o columns of the stored H (or H~) into `dst` (3 n ns o).
        fn gather(
            &self,
            src: &DeviceBuf<f64>,
            dst: &mut DeviceBuf<f64>,
            pa: usize,
            ns: usize,
        ) -> Result<(), GpuError> {
            let (n, o, naux) = (self.n, self.o, self.naux);
            // SAFETY: hd_gather_h: reads rows pa..pa + ns of the 3 n o columns
            // (naux each), writes 3 n ns o.
            unsafe {
                launch(
                    self.gpu,
                    &self.kr.gather_h,
                    3 * n * ns * o,
                    &[
                        Arg::I32(ns as i32),
                        Arg::I64(n as i64),
                        Arg::I32(o as i32),
                        Arg::I32(naux as i32),
                        Arg::I32(pa as i32),
                        Arg::buf(src),
                        Arg::buf_mut(dst),
                    ],
                )
            }
        }

        /// The packed symmetric parts of C_A^T h for the outputs `ys`, times `fct`, at `out`
        /// (output y at y `ystride`; hd_symc).
        fn symc(
            &self,
            ys: &SymRows,
            fct: f64,
            h: &DeviceBuf<f64>,
            ns: usize,
            out: DevPtr<f64>,
            ystride: usize,
        ) -> Result<(), GpuError> {
            let (n, o, npk, ny) = (self.n, self.o, self.npk, ys.len);
            // SAFETY: hd_symc: ny ns npk outputs at out + y ystride + q npk + pk (the
            // callers' buffers hold them); reads rows a < n of h (3 n ns o) and of C
            // (n o), directions < 3.
            unsafe {
                launch(
                    self.gpu,
                    &self.kr.symc,
                    ny * ns * npk,
                    &[
                        Arg::I32(ny as i32),
                        Arg::I32(ns as i32),
                        Arg::I32(o as i32),
                        Arg::I64(n as i64),
                        Arg::buf(&ys.k),
                        Arg::buf(&ys.a0),
                        Arg::buf(&ys.a1),
                        Arg::F64(fct),
                        self.cptr,
                        Arg::buf(h),
                        out.into(),
                        Arg::I64(ystride as i64),
                    ],
                )
            }
        }
    }

    /// What the second derivatives leave.
    struct SecondDerivs {
        /// The three-center term's M (H = (M + M^T) / 2), fixed point.
        hacc: DetAcc,
        /// gamma_PQ of the two-center term.
        gam2: Option<DeviceBuf<f64>>,
    }
}
