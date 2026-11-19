//! The explicit XC terms of an analytic Kohn-Sham Hessian on the device: `GpuXcHess` (feature
//! `gpu`).
//!
//! The quantities of linus's CPU `XcIntegrator::hessian_terms` and `grid_terms` (`xc::hessian`
//! in the `linus` crate), for spin channels s (one: a closed shell's total density; two: the
//! spin densities of an open shell), LDA, GGA and (tau-dependent) meta-GGA functionals:
//!
//! - **fixed grid:** the first derivatives V_s^x of the channels' XC potential matrices for
//!   every perturbation x = (atom A, i), and the explicit second derivative of E_xc;
//! - **grid response** (optional): what the grid moving with the atoms adds, each point with
//!   its owner and the partition weights' derivatives dw_g/dR_C (linus-grid's
//!   `point_weight_gradient`, so any partition: SSF, Becke, LKO): the potential derivatives
//!   dV_s^y and the derivative E1 of the fixed-grid XC gradient (not symmetric).
//!
//! Notation of the CPU module: per point the variables u = (rho_s, grad rho_s, tau_s) (nu =
//! nsp ns components, ns = 1 for LDA, 4 for GGA, 5 for meta-GGA), the weighted first and second
//! derivatives wv = w df/du and wF = w d2f/du2 (the sigmas folded in by the chain rule),
//! X_s = phi D_s, X_s,k = d_k phi D_s, u^{Ai} the change of u when atom A moves along i at
//! fixed density (tau^{Ai} = -sum_{mu in A} sum_k d_ki phi_mu X_k,mu), t^x = wF u^x. Then
//!
//!   V_s^x = N_s^x + N_s^x^T,  N_s^x = phi^T Y_s\[t^x\] (+ sum_k (d_k phi)^T 1/4 t_tau,s d_k phi)
//!           - (rows on A of G_s^i),
//!   Y_s\[t\] = 1/2 t_rho,s phi + sum_k t_k,s d_k phi,
//!   G_s^i = A_si^T phi + (d_i phi)^T B_s (+ sum_k (d_ki phi)^T 1/2 wv_tau,s d_k phi),
//!           A_si = wv_rho,s d_i phi + sum_k wv_k,s d_ki phi, B_s = sum_k wv_k,s d_k phi;
//!   E^xy = sum_g u^x . t^y (kernel) + same-function terms (A = B) + sum_s sum_{mu in A, nu in B}
//!          D_s,mu,nu C_s^{ij}_mu,nu (cross; C^{ij} = L_i^T R_j of the CPU), symmetrized;
//!   dV_s^{Cj} = sum over C's points of the motion terms (t_j = wF d_j u, and G_s^j with wv
//!          masked to C's points) + sum_g dw_g/dC_j Y_s\[v\] at g (v = df/du unweighted);
//!   E1[x][(C, j)] = sum_g dw_g/dC_j v.u^x + sum over C's points of (t_j . u^x + wv . d_j u^x).
//!
//! The device's work, over XCP's per-batch layout (`crate::xc::layout`: per grid batch the
//! functions the CPU's value test keeps, at the CPU Hessian's collocation order and
//! threshold), in FP64 on one stream:
//! 1. **Densities:** a pass with first-order collocation, X = phi D_s by the variable-size
//!    batched GEMM (and for meta-GGAs grad phi D_s), rho, grad rho and tau; the functional on
//!    the host (libxc, order 2), wv (as v) and wF per point back to the device.
//! 2. **Main pass** per chunk of batches: third-order collocation (GGA; second for LDA),
//!    X_s for the components 0..nx by GEMMs, then per batch
//!    - per point d_j u and t_j = wF d_j u (grid terms);
//!    - per (point, local atom) u^{Ai}, t^{Ai}, v.u^{Ai} and the r^{Ai}_j of E1;
//!    - E: u^T t by GEMM, the same-function terms (a warp per atom and pair), the cross term
//!      (for GGAs 2 P^T Q by GEMM instead of the CPU's P^T Q + Q^T P, whose second half only
//!      adds the transpose of the first's contribution to the symmetrized E; meta-GGAs stack
//!      the tau term's d_ki phi and wv_tau d_ki phi below P and 2 Q, a GEMM 4 np deep;
//!      contracted with D_s per atom pair), into a fixed-point (3N)^2 accumulator; E1 per
//!      (perturbation, atom, j) into another;
//!    - G_s^i per owner of the batch's points by GEMMs over that owner's points only (the
//!      batch's points grouped by owner: a k-d tree batch mixes several owners; meta-GGAs one
//!      more, 3 k_o deep, for the tau term), the grid slots' t (motion and weight parts), and
//!      one fused kernel for all the batch's potential slots (fixed (A, i, s) and grid
//!      (C, j, s)): phi^T Y\[t\] by 32 x 32 tiles with Y formed on the fly (meta-GGAs: the tau
//!      term's three d_k phi products as three more reduction rows per point, `xh_vslots_mg`),
//!      over the slot's points (all of the batch's for a fixed slot; for a grid slot
//!      those where its t can be nonzero: C's own points and those with a nonzero dw_g/dC,
//!      about 60 % on SSF grids), the G rows in the epilogue, each element added to the packed
//!      lower triangle of its accumulator in fixed point (N and N^T land on the same element:
//!      no N + N^T buffer). With `single` (linus's `precision.fp32: "on"`) the products of that
//!      kernel are FP32 (`xh_vslots32`: each element's sum over at most a batch's points), the
//!      G rows and the accumulation FP64.
//! 3. **V C:** each accumulated V_s^x (or dV_s^y) unpacked and multiplied by the channel's
//!    occupied orbitals on the device (strided-batched DGEMM): the CPHF right-hand sides only
//!    need V^x C_o.
//!
//! When the accumulators of all 3N perturbations do not fit `v_bytes` (as set, or what the
//! device has free beside the call's other workspaces, `GpuXcHess::fit_accumulators`), the main
//! pass runs once per group of atoms (perturbations of the group's atoms only; E in the first);
//! the fixed-point sums give the same bits in any grouping.
//!
//! Not covered: Laplacian-dependent meta-GGAs, and VV10 (no Hessian terms on either backend).

/// Settings of a `GpuXcHess` setup.
#[derive(Clone, Debug)]
pub struct GpuXcHessOptions {
    /// Functions whose values and derivatives (to the collocation order) stay below this over
    /// a grid batch are skipped (the CPU integrator's `func_thresh`).
    pub func_thresh: f64,
    /// Points per block of collocation (whole grid batches).
    pub block_points: usize,
    /// Device bytes for the per-batch workspaces of a chunk of batches.
    pub workspace: usize,
    /// Device bytes the fixed-point potential-derivative accumulators may take at most (more
    /// perturbations: one main pass per group of atoms; `GpuXcHess::fit_accumulators` lowers it
    /// to the device's free memory).
    pub v_bytes: usize,
    /// The potential slots' products in FP32 (their sums over at most a batch's points; the
    /// batches added in fixed point): linus's `precision.fp32: "on"`.
    pub single: bool,
    /// What runs the FP64 per-batch products.
    pub products: crate::xc::ProductEngine,
}

impl Default for GpuXcHessOptions {
    fn default() -> Self {
        GpuXcHessOptions {
            func_thresh: 1e-11,
            block_points: 2048,
            workspace: 1 << 31,
            v_bytes: 24 << 30,
            single: false,
            products: crate::xc::ProductEngine::Vgemm,
        }
    }
}

/// The XC Hessian terms of one call.
#[derive(Clone, Debug, Default)]
pub struct GpuXcHessResult {
    /// Per channel s: V_s^x C_s for x = 0..3N (x-major; each n x o_s, column-major).
    pub vfix_c: Vec<Vec<f64>>,
    /// Per channel: dV_s^y C_s (empty without the grid terms).
    pub vgrid_c: Vec<Vec<f64>>,
    /// The fixed-grid second derivative of E_xc, (3N)^2 row-major, symmetrized.
    pub hfix: Vec<f64>,
    /// E1 ((3N)^2 row-major; zeros without the grid terms).
    pub hgrid: Vec<f64>,
}

/// Wall seconds of the last call's phases.
#[derive(Clone, Debug, Default)]
pub struct GpuXcHessTimes {
    pub total: f64,
    pub densities: f64,
    pub functional: f64,
    pub main: f64,
    pub vc: f64,
    /// Main passes (groups of atoms).
    pub passes: usize,
}

#[cfg(feature = "gpu")]
impl crate::engine::Phases for GpuXcHessTimes {
    fn phases(&self) -> impl IntoIterator<Item = (&'static str, f64)> {
        [
            ("densities", self.densities),
            ("functional", self.functional),
            ("main", self.main),
            ("vc", self.vc),
        ]
    }
}

/// Setup statistics.
#[derive(Clone, Debug, Default)]
pub struct GpuXcHessStats {
    pub nbatches: usize,
    pub nchunks: usize,
    /// Potential slots (perturbation x spin) over all batches: fixed, grid.
    pub slots_fixed: usize,
    pub slots_grid: usize,
    /// Flops of the fused potential kernel (2 nf^2 times the slot's points).
    pub vslot_flops: f64,
    /// Points of the grid slots (per grid atom of each batch, its points with a nonzero
    /// first-order potential), and all their batches' points.
    pub grid_points: usize,
    pub grid_points_dense: usize,
    /// Seconds of the setup: the layout, and the weight derivatives on the host.
    pub layout_seconds: f64,
    pub weights_seconds: f64,
}

#[cfg(feature = "gpu")]
pub use device::*;

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use crate::blas::{GemmDims, MatMut, MatRef, Op};
    use crate::cpu::{deriv_indices, BasisSet, MolecularGrid};
    use crate::detsum::{self, DetAcc};
    use crate::error::GpuError;
    use crate::kernels::registry::KernelModule;
    use crate::kernels::vgemm::VgemmProduct;
    use crate::linalg::{Arith, Gemm};
    use crate::pool::{Grant, PlanViolation};
    use crate::response::xc::kernel_weights;
    use crate::xc::density::{DensityOps, DensityOut, DensityPass, DensityWork, PassTiming};
    use crate::xc::layout::{self, Chunk, Collocation, XcLayout, XcLayoutSpec};
    use crate::xc::{HostFunctional, Prod, ProdKind, XcKind};
    use crate::{Arg, DeviceBuf, Gpu, Kernel, Launch};
    use rayon::prelude::*;
    use std::borrow::Cow;
    use std::time::Instant;

    /// i64 fields per batch descriptor (`XH_HB`).
    const HB: usize = 32;
    const H_P0: usize = 0;
    const H_NP: usize = 1;
    const H_NF: usize = 2;
    const H_PHI: usize = 3;
    const H_X: usize = 4;
    const H_D: usize = 5;
    const H_FL: usize = 6;
    const H_NAT: usize = 7;
    const H_AT: usize = 8;
    const H_APTR: usize = 9;
    const H_AFN: usize = 10;
    const H_LA: usize = 11;
    const H_U: usize = 12;
    const H_S: usize = 13;
    const H_R: usize = 14;
    const H_TJ: usize = 15;
    const H_EL: usize = 16;
    const H_NG: usize = 17;
    const H_GAT: usize = 18;
    const H_DW: usize = 19;
    const H_NO: usize = 20;
    const H_OWN: usize = 21;
    const H_AB: usize = 22;
    const H_G: usize = 23;
    const H_CC: usize = 24;
    const H_LR: usize = 25;
    const H_TG: usize = 26;
    const H_GPT: usize = 27;
    const H_OPT: usize = 28;
    const H_OOF: usize = 29;
    const H_OPTR: usize = 30;

    /// Variables per spin channel: rho (LDA), grad rho (GGA), tau (meta-GGA).
    fn vars_per_spin(kind: XcKind) -> usize {
        crate::response::xc::vars_per_spin(kind.gga, kind.mgga)
    }

    /// i64 fields per potential slot (`xh_vslots`).
    const SD: usize = 10;
    /// Tile edge of `xh_vslots`.
    const VT: usize = 32;

    const KERNELS: &str = include_str!("../../device/hess/xc.cu");

    /// The fused potential-slot kernel (`xh_vslots`; `xh_vslots32` with FP32 products).
    const VSLOTS: &str = include_str!("../../device/hess/xc_vslots.cu.in");

    /// The XC Hessian's module ([`source`]).
    pub(crate) const MODULE: KernelModule = KernelModule::new(
        "xchess",
        || Cow::Owned(source()),
        &[
            "xh_point",
            "xh_atom",
            "xh_same",
            "xh_lr",
            "xh_cross",
            "xh_e1",
            "xh_ab",
            "xh_ttg",
            "xh_unpack",
            "xh_vslots",
            "xh_vslots32",
            "xh_vslots_mg",
            "xh_vslots32_mg",
        ],
    );

    /// The module source: detsum, the derivative tables, the kernels, the potential kernel in
    /// FP64 and FP32, each without and with the meta-GGA tau term.
    fn source() -> String {
        let vs = |name: &str, t: &str, mg: &str| {
            VSLOTS
                .replace("@NAME@", name)
                .replace("@T@", t)
                .replace("@MG@", mg)
        };
        format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}",
            crate::detsum::DEVICE_SRC,
            tables(),
            KERNELS,
            vs("xh_vslots", "double", "0"),
            vs("xh_vslots32", "float", "0"),
            vs("xh_vslots_mg", "double", "1"),
            vs("xh_vslots32_mg", "float", "1")
        )
    }

    /// The derivative index tables of the collocation order 3 (`ao::deriv_indices` order).
    fn tables() -> String {
        let di = deriv_indices(3);
        let find = |e: [usize; 3]| di.iter().position(|v| *v == e).expect("component");
        let (mut d1, mut d2, mut d3) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..3 {
            let mut e1 = [0; 3];
            e1[i] += 1;
            d1.push(find(e1));
            for j in 0..3 {
                let mut e2 = e1;
                e2[j] += 1;
                d2.push(find(e2));
                for k in 0..3 {
                    let mut e3 = e2;
                    e3[k] += 1;
                    d3.push(find(e3));
                }
            }
        }
        let list = |v: &[usize]| {
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "static __constant__ int XH_D1[3] = {{{}}};\nstatic __constant__ int XH_D2[9] = \
             {{{}}};\nstatic __constant__ int XH_D3[27] = {{{}}};\n",
            list(&d1),
            list(&d2),
            list(&d3)
        )
    }

    struct Kernels {
        point: Kernel,
        atom: Kernel,
        same: Kernel,
        lr: Kernel,
        cross: Kernel,
        e1: Kernel,
        ab: Kernel,
        ttg: Kernel,
        vslots: Kernel,
        vslots32: Kernel,
        vslots_mg: Kernel,
        vslots32_mg: Kernel,
        unpack: Kernel,
    }

    /// One grid batch on the host: local atoms of its kept functions (in order of first
    /// appearance, as the CPU), the functions grouped by atom, and (grid terms) its owners and
    /// weight-derivative atoms with the derivatives.
    struct BInfo {
        atoms: Vec<i32>,
        la: Vec<i32>,
        aptr: Vec<i32>,
        afn: Vec<i32>,
        gat: Vec<i32>,
        own: Vec<i32>,
        /// dw\[(ga 3 + j) np + g\]
        dw: Vec<f64>,
        /// Per grid atom the batch's points where its slots' first-order potentials can be
        /// nonzero (its own points, and those with a nonzero weight derivative):
        /// gpts\[gptr\[ga\]..gptr\[ga + 1\]\], ascending.
        gptr: Vec<i32>,
        gpts: Vec<i32>,
        /// The batch's points grouped by owner (one group without the grid terms): positions
        /// optr\[o\]..optr\[o + 1\] of opts are owner o's points, oof the owner of each position.
        optr: Vec<i32>,
        opts: Vec<i32>,
        oof: Vec<i32>,
    }

    /// Per-batch workspace sizes (doubles) and their offsets within a chunk.
    #[derive(Clone, Copy, Default)]
    struct Sizes {
        u: usize,
        s: usize,
        r: usize,
        tj: usize,
        el: usize,
        dw: usize,
        ab: usize,
        g: usize,
        cc: usize,
        lr: usize,
        tg: usize,
    }

    impl Sizes {
        fn add(&mut self, o: &Sizes) {
            self.u += o.u;
            self.s += o.s;
            self.r += o.r;
            self.tj += o.tj;
            self.el += o.el;
            self.dw += o.dw;
            self.ab += o.ab;
            self.g += o.g;
            self.cc += o.cc;
            self.lr += o.lr;
            self.tg += o.tg;
        }

        fn max(&mut self, o: &Sizes) {
            self.u = self.u.max(o.u);
            self.s = self.s.max(o.s);
            self.r = self.r.max(o.r);
            self.tj = self.tj.max(o.tj);
            self.el = self.el.max(o.el);
            self.dw = self.dw.max(o.dw);
            self.ab = self.ab.max(o.ab);
            self.g = self.g.max(o.g);
            self.cc = self.cc.max(o.cc);
            self.lr = self.lr.max(o.lr);
            self.tg = self.tg.max(o.tg);
        }

        fn total(&self) -> usize {
            // T has U's size
            2 * self.u
                + self.s
                + self.r
                + self.tj
                + self.el
                + self.dw
                + self.ab
                + self.g
                + self.cc
                + 2 * self.lr
                + self.tg
        }
    }

    /// The explicit XC Hessian terms on the device (see the module docs).
    pub struct GpuXcHess {
        nbf: usize,
        natom: usize,
        npts: usize,
        nsp: usize,
        grid_terms: bool,
        /// The functional on the host and its ingredients.
        functional: HostFunctional,
        /// Collocation components (order 3 GGA, 2 LDA), X components, of the density pass.
        nc: usize,
        nx: usize,
        nc1: usize,
        lay: XcLayout,
        colloc1: Vec<Kernel>,
        kern: Kernels,
        /// The FP64 products' kind ([`GpuXcHessOptions::products`]).
        fp64: ProdKind,
        gemm: Gemm,
        info: Vec<BInfo>,
        /// Per batch descriptors (HB fields), the int tables, per chunk the workspace sizes.
        hb: DeviceBuf<i64>,
        hb_host: Vec<i64>,
        ints: DeviceBuf<i32>,
        chunk_sizes: Vec<Sizes>,
        max_sizes: Sizes,
        xg: DeviceBuf<f64>,
        yg: DeviceBuf<f64>,
        zg: DeviceBuf<f64>,
        w: DeviceBuf<f64>,
        w_host: Vec<f64>,
        owner: DeviceBuf<i32>,
        v_bytes: usize,
        single: bool,
        /// The memory plan's grant of a call's workspaces ([`GpuXcHess::grant_workspace`]),
        /// drawn by their allocations (freed, they give the bytes back).
        workspace: Option<Grant>,
        stats: GpuXcHessStats,
        pub last: GpuXcHessTimes,
    }

    impl GpuXcHess {
        /// The functional `name` (a linus-xc name) on `grid` for `basis`, for `nsp` spin
        /// channels, with (`grid_terms`) or without the grid response.
        pub fn new(
            gpu: &Gpu,
            basis: &BasisSet,
            grid: &MolecularGrid,
            name: &str,
            nsp: usize,
            grid_terms: bool,
            opts: &GpuXcHessOptions,
        ) -> Result<GpuXcHess, GpuError> {
            let t0 = Instant::now();
            if !(1..=2).contains(&nsp) {
                return Err(GpuError::Contract(format!(
                    "XC Hessian: {nsp} spin channels"
                )));
            }
            let functional = HostFunctional::new(name, None)?;
            let kind = functional.kind();
            if functional.needs_lapl() {
                return Err(GpuError::NotCovered(format!(
                    "{name}: the XC Hessian terms of Laplacian-dependent meta-GGAs have no device \
                     kernels"
                )));
            }
            let (gga, mgga) = (kind.gga, kind.mgga);
            let order = if gga { 3 } else { 2 };
            let nc = crate::cpu::ncomp(order);
            let nx = if gga && grid_terms { 10 } else { 4 };
            let ns = vars_per_spin(kind);
            let nu = nsp * ns;
            // meta-GGAs: the G operands' comps per spin and spin-free (`xh_ab`), the cross
            // term's rows per point (`xh_lr`), the potential products per point (`xh_vslots`)
            let (nab, nh, nr) = if mgga { (11, 9, 4) } else { (8, 0, 1) };
            let natom = grid.atom_z.len();
            let mut lay = XcLayout::new(
                gpu,
                basis,
                grid,
                &XcLayoutSpec {
                    screen_order: order,
                    order,
                    thresh: opts.func_thresh,
                    block_points: opts.block_points,
                    budget: usize::MAX,
                },
            )?;
            let layout_seconds = t0.elapsed().as_secs_f64();
            // the batches' atoms and (grid terms) weight derivatives
            let t1 = Instant::now();
            let mut func_atom = vec![0i32; basis.nbf];
            for s in &basis.shells {
                func_atom[s.ao_start..s.ao_start + s.nfunc()].fill(s.atom as i32);
            }
            let info: Vec<BInfo> = lay
                .host
                .par_iter()
                .map_init(
                    || (grid.partitioner().scratch(), Vec::new()),
                    |(scr, out), h| {
                        let mut atoms: Vec<i32> = Vec::new();
                        let mut la = Vec::with_capacity(h.global.len());
                        for &f in &h.global {
                            let a = func_atom[f as usize];
                            let l = match atoms.iter().position(|&x| x == a) {
                                Some(l) => l,
                                None => {
                                    atoms.push(a);
                                    atoms.len() - 1
                                }
                            };
                            la.push(l as i32);
                        }
                        let mut aptr = vec![0i32; atoms.len() + 1];
                        for &l in &la {
                            aptr[l as usize + 1] += 1;
                        }
                        for a in 0..atoms.len() {
                            aptr[a + 1] += aptr[a];
                        }
                        let mut fill = aptr.clone();
                        let mut afn = vec![0i32; la.len()];
                        for (f, &l) in la.iter().enumerate() {
                            afn[fill[l as usize] as usize] = f as i32;
                            fill[l as usize] += 1;
                        }
                        let (mut gat, mut own, mut dw) = (Vec::new(), Vec::new(), Vec::new());
                        let (mut gptr, mut gpts) = (vec![0i32], Vec::new());
                        if grid_terms && h.np > 0 {
                            let np = h.np;
                            let mut entries: Vec<(usize, usize, [f64; 3])> = Vec::new();
                            for g in 0..np {
                                let p = h.p0 + g;
                                grid.point_weight_gradient(p, scr, out);
                                for &(c, d) in out.iter() {
                                    if d.iter().any(|&x| x != 0.0) {
                                        entries.push((g, c, d));
                                    }
                                }
                                own.push(grid.atom[p] as i32);
                            }
                            own.sort_unstable();
                            own.dedup();
                            gat = own.clone();
                            gat.extend(entries.iter().map(|e| e.1 as i32));
                            gat.sort_unstable();
                            gat.dedup();
                            dw = vec![0.0; gat.len() * 3 * np];
                            for (g, c, d) in entries {
                                let ga = gat.binary_search(&(c as i32)).expect("atom");
                                for j in 0..3 {
                                    dw[(ga * 3 + j) * np + g] += d[j];
                                }
                            }
                            for (ga, &c) in gat.iter().enumerate() {
                                for g in 0..np {
                                    let moves = grid.atom[h.p0 + g] as i32 == c;
                                    if moves || (0..3).any(|j| dw[(ga * 3 + j) * np + g] != 0.0) {
                                        gpts.push(g as i32);
                                    }
                                }
                                gptr.push(gpts.len() as i32);
                            }
                        }
                        let (mut optr, mut opts, mut oof) = (vec![0i32], Vec::new(), Vec::new());
                        if own.is_empty() {
                            opts.extend(0..h.np as i32);
                            oof.resize(h.np, 0);
                            optr.push(h.np as i32);
                        } else {
                            for (o, &c) in own.iter().enumerate() {
                                for g in 0..h.np {
                                    if grid.atom[h.p0 + g] as i32 == c {
                                        opts.push(g as i32);
                                        oof.push(o as i32);
                                    }
                                }
                                optr.push(opts.len() as i32);
                            }
                        }
                        BInfo {
                            atoms,
                            la,
                            aptr,
                            afn,
                            gat,
                            own,
                            dw,
                            gptr,
                            gpts,
                            optr,
                            opts,
                            oof,
                        }
                    },
                )
                .collect();
            let weights_seconds = t1.elapsed().as_secs_f64();
            // per-batch workspace sizes, chunks of whole blocks within the workspace
            let sizes: Vec<Sizes> = lay
                .host
                .iter()
                .zip(&info)
                .map(|(h, b)| {
                    let (np, nf) = (h.np, h.global.len());
                    let npl = 3 * b.atoms.len();
                    let ng = b.gat.len();
                    let no = if grid_terms { b.own.len() } else { 1 };
                    Sizes {
                        u: npl * nu * np,
                        s: if grid_terms { npl * np } else { 0 },
                        r: if grid_terms { 3 * npl * np } else { 0 },
                        tj: if grid_terms { 3 * nu * np } else { 0 },
                        el: npl * npl,
                        dw: ng * 3 * np,
                        ab: (nsp * nab + nh) * nf * np,
                        g: no * nsp * 3 * nf * nf,
                        cc: 9 * nf * nf,
                        lr: 3 * nf * nr * np,
                        tg: ng * 3 * nu * np,
                    }
                })
                .collect();
            let per_value = nc + nsp * nx;
            let budget = (opts.workspace / 8).max(1);
            let mut chunks: Vec<Chunk> = Vec::new();
            let mut b = 0;
            while b < lay.blocks.len() {
                let (mut cost, mut x_len, mut d_len, mut max_np) = (0usize, 0, 0, 0);
                let mut e = b;
                while e < lay.blocks.len() {
                    let lb = &lay.blocks[e];
                    let (mut c, mut bx, mut bd, mut bnp) = (0usize, 0, 0, 0);
                    for (h, sz) in lay.host[lb.j0..lb.j1].iter().zip(&sizes[lb.j0..lb.j1]) {
                        let nf = h.global.len();
                        c += sz.total() + per_value * nf * h.np + nsp * nf * nf;
                        bx += nf * h.np;
                        bd += nf * nf;
                        bnp = bnp.max(h.np);
                    }
                    if e > b && cost + c > budget {
                        break;
                    }
                    cost += c;
                    x_len += bx;
                    d_len += bd;
                    max_np = max_np.max(bnp);
                    e += 1;
                }
                chunks.push(Chunk {
                    b0: b,
                    b1: e,
                    j0: lay.blocks[b].j0,
                    j1: lay.blocks[e - 1].j1,
                    max_np,
                    x_len,
                    d_len,
                });
                b = e;
            }
            let (bd, lf, gf) = layout::descriptors(&lay.host, &chunks, nc);
            lay.bd = gpu.upload(if bd.is_empty() { &[0] } else { &bd })?;
            lay.lf = gpu.upload(if lf.is_empty() { &[0] } else { &lf })?;
            lay.gf = gpu.upload(if gf.is_empty() { &[0] } else { &gf })?;
            lay.max_x = chunks.iter().map(|c| c.x_len).max().unwrap_or(0);
            lay.max_d = chunks.iter().map(|c| c.d_len).max().unwrap_or(0);
            lay.chunks = chunks;
            // the descriptors and int tables
            let nb = lay.host.len();
            let mut hb = vec![0i64; HB * nb.max(1)];
            let mut ints: Vec<i32> = Vec::new();
            let mut chunk_sizes = Vec::with_capacity(lay.chunks.len());
            let mut max_sizes = Sizes::default();
            let mut stats = GpuXcHessStats {
                nbatches: nb,
                nchunks: lay.chunks.len(),
                layout_seconds,
                weights_seconds,
                ..Default::default()
            };
            for c in &lay.chunks {
                let mut off = Sizes::default();
                let (mut xo, mut dd) = (0usize, 0usize);
                for j in c.j0..c.j1 {
                    let h = &lay.host[j];
                    let bi = &info[j];
                    let nf = h.global.len();
                    let d = &mut hb[HB * j..HB * (j + 1)];
                    d[H_P0] = h.p0 as i64;
                    d[H_NP] = h.np as i64;
                    d[H_NF] = nf as i64;
                    d[H_PHI] = (nc * xo) as i64;
                    d[H_X] = (nx * xo) as i64;
                    d[H_D] = dd as i64;
                    d[H_FL] = bd[layout::BD * j + layout::BD_FL];
                    d[H_NAT] = bi.atoms.len() as i64;
                    let mut put = |v: &[i32]| {
                        let o = ints.len() as i64;
                        ints.extend_from_slice(v);
                        o
                    };
                    d[H_AT] = put(&bi.atoms);
                    d[H_APTR] = put(&bi.aptr);
                    d[H_AFN] = put(&bi.afn);
                    d[H_LA] = put(&bi.la);
                    d[H_NG] = bi.gat.len() as i64;
                    d[H_GAT] = put(&bi.gat);
                    d[H_GPT] = put(&bi.gpts);
                    d[H_OPT] = put(&bi.opts);
                    d[H_OOF] = put(&bi.oof);
                    d[H_OPTR] = put(&bi.optr);
                    d[H_NO] = if grid_terms { bi.own.len() as i64 } else { 1 };
                    d[H_OWN] = put(if grid_terms { &bi.own } else { &[-1] });
                    d[H_U] = off.u as i64;
                    d[H_S] = off.s as i64;
                    d[H_R] = off.r as i64;
                    d[H_TJ] = off.tj as i64;
                    d[H_EL] = off.el as i64;
                    d[H_DW] = off.dw as i64;
                    d[H_AB] = off.ab as i64;
                    d[H_G] = off.g as i64;
                    d[H_CC] = off.cc as i64;
                    d[H_LR] = off.lr as i64;
                    d[H_TG] = off.tg as i64;
                    off.add(&sizes[j]);
                    xo += nf * h.np;
                    dd += nf * nf;
                    let f2 = 2.0 * (nf * nf * h.np * nr) as f64;
                    stats.slots_fixed += nsp * 3 * bi.atoms.len();
                    stats.slots_grid += nsp * 3 * bi.gat.len();
                    stats.vslot_flops += f2 * (nsp * 3 * bi.atoms.len()) as f64;
                    let gp: usize = bi.gptr.windows(2).map(|w| (w[1] - w[0]) as usize).sum();
                    stats.vslot_flops += 2.0 * (nf * nf * gp * nr * nsp * 3) as f64;
                    stats.grid_points += gp;
                    stats.grid_points_dense += bi.gat.len() * h.np;
                }
                max_sizes.max(&off);
                chunk_sizes.push(off);
            }
            let tsrc = MODULE.source();
            let k = |e: &str| gpu.kernel(MODULE.name, &tsrc, e);
            let kern = Kernels {
                point: k("xh_point")?,
                atom: k("xh_atom")?,
                same: k("xh_same")?,
                lr: k("xh_lr")?,
                cross: k("xh_cross")?,
                e1: k("xh_e1")?,
                ab: k("xh_ab")?,
                ttg: k("xh_ttg")?,
                vslots: k("xh_vslots")?,
                vslots32: k("xh_vslots32")?,
                vslots_mg: k("xh_vslots_mg")?,
                vslots32_mg: k("xh_vslots32_mg")?,
                unpack: k("xh_unpack")?,
            };
            let nc1 = crate::cpu::ncomp(kind.order());
            let colloc1 = lay.kernels_of_order(gpu, basis, kind.order())?;
            let owner: Vec<i32> = grid.atom.iter().map(|&a| a as i32).collect();
            Ok(GpuXcHess {
                nbf: basis.nbf,
                natom,
                npts: grid.x.len(),
                nsp,
                grid_terms,
                functional,
                nc,
                nx,
                nc1,
                colloc1,
                kern,
                fp64: ProdKind::fp64(opts.products),
                gemm: Gemm::new(gpu)?,
                info,
                hb: gpu.upload(&hb)?,
                hb_host: hb,
                ints: gpu.upload(if ints.is_empty() { &[0] } else { &ints })?,
                chunk_sizes,
                max_sizes,
                xg: gpu.upload(&grid.x)?,
                yg: gpu.upload(&grid.y)?,
                zg: gpu.upload(&grid.z)?,
                w: gpu.upload(&grid.w)?,
                w_host: grid.w.clone(),
                owner: gpu.upload(if owner.is_empty() { &[0] } else { &owner })?,
                v_bytes: opts.v_bytes,
                single: opts.single,
                workspace: None,
                lay,
                stats,
                last: GpuXcHessTimes::default(),
            })
        }

        pub fn stats(&self) -> &GpuXcHessStats {
            &self.stats
        }

        /// Device bytes of the workspaces and accumulators a call allocates (an estimate, for
        /// the caller's check against free memory).
        pub fn workspace_bytes(&self, ncols: &[usize]) -> usize {
            self.fixed_bytes(ncols) + self.group_atoms() * self.per_atom_bytes()
        }

        /// Sets aside the workspaces of a call with `ncols` ([`Self::workspace_bytes`], the
        /// accumulators as fitted) as the memory plan's grant, which the call's allocations draw
        /// from; an out-of-memory error when they do not fit.
        pub fn grant_workspace(&mut self, gpu: &Gpu, ncols: &[usize]) -> Result<(), GpuError> {
            self.workspace = None;
            let need = self.workspace_bytes(ncols);
            self.workspace = Some(gpu.grant(need, "the Hessian's XC workspace")?);
            Ok(())
        }

        /// The calls' draws beyond the workspace's grant (plan violations).
        pub fn workspace_violations(&self) -> Vec<PlanViolation> {
            self.workspace
                .as_ref()
                .map(Grant::violations)
                .unwrap_or_default()
        }

        /// Size the potential-derivative accumulators to the `free` device bytes left beside
        /// the call's other workspaces (at least one atom's, at most `v_bytes` as set): under a
        /// memory cap more main passes, one per group of atoms, instead of no device run. The
        /// passes' fixed-point sums are the same bits in any grouping; with room for every atom
        /// (no cap) the grouping is unchanged. [`Self::workspace_bytes`] then tells whether the
        /// call fits.
        pub fn fit_accumulators(&mut self, ncols: &[usize], free: usize) {
            let room = free.saturating_sub(self.fixed_bytes(ncols));
            self.v_bytes = self.v_bytes.min(room.max(self.per_atom_bytes()));
        }

        /// Device bytes of one atom's accumulators: its 3 perturbations per kind (fixed grid,
        /// grid response) and channel, packed n (n + 1) / 2, three 8-byte words each.
        fn per_atom_bytes(&self) -> usize {
            let n = self.nbf;
            let nkind = if self.grid_terms { 2 } else { 1 };
            nkind * self.nsp * 3 * (n * (n + 1) / 2) * 24
        }

        /// The atoms of a main pass: as many as `v_bytes` holds, at least one.
        fn group_atoms(&self) -> usize {
            (self.v_bytes / self.per_atom_bytes().max(1)).clamp(1, self.natom.max(1))
        }

        /// Device bytes of a call's workspaces besides the accumulators.
        fn fixed_bytes(&self, ncols: &[usize]) -> usize {
            let (n, nsp) = (self.nbf, self.nsp);
            let lay = &self.lay;
            // per point v (nu) and wF (nu^2); the GGA's sizes at least
            let ns = vars_per_spin(self.functional.kind()).max(4);
            let doubles = lay.max_block_nf * self.nc * lay.max_block_np
                + (self.nc + nsp * self.nx) * lay.max_x
                + nsp * lay.max_d
                + self.max_sizes.total()
                + self.npts * (nsp * (ns + 1) + nsp * nsp * ns * ns + 8)
                // the FP32 copy of the compressed values (`single`)
                + if self.single { self.nc * lay.max_x / 2 } else { 0 };
            let omax = ncols.iter().copied().max().unwrap_or(0);
            8 * doubles + 8 * (2 * n * n + n * omax) * 8
        }

        /// The terms at the channels' densities `dens` (column-major n x n: one total density,
        /// or the alpha and beta densities) for the occupied orbitals `occ` (per channel,
        /// column-major n x o_s): see [`GpuXcHessResult`].
        pub fn compute(
            &mut self,
            gpu: &Gpu,
            dens: &[&[f64]],
            occ: &[(&[f64], usize)],
        ) -> Result<GpuXcHessResult, GpuError> {
            let t_all = Instant::now();
            let (n, nsp) = (self.nbf, self.nsp);
            if dens.len() != nsp || occ.len() != nsp {
                return Err(GpuError::Contract("XC Hessian: channels".into()));
            }
            for (d, (c, o)) in dens.iter().zip(occ) {
                if d.len() != n * n || c.len() < n * o {
                    return Err(GpuError::Contract("XC Hessian: matrix sizes".into()));
                }
            }
            let mut times = GpuXcHessTimes::default();
            let dm: Vec<f64> = dens.iter().flat_map(|d| d.iter().copied()).collect();
            let dm = gpu.upload_granted(self.workspace.as_ref(), &dm)?;
            // 1. densities and the functional
            let t = Instant::now();
            let (rho, grad, tau) = self.densities(gpu, &dm)?;
            times.densities = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let (vu, wf) = self.functional(&rho, &grad, &tau)?;
            let vu = gpu.upload_granted(self.workspace.as_ref(), &vu)?;
            let wf = gpu.upload_granted(self.workspace.as_ref(), &wf)?;
            times.functional = t.elapsed().as_secs_f64();
            // 2. the main pass(es)
            let t = Instant::now();
            let n3 = 3 * self.natom;
            let npack = n * (n + 1) / 2;
            let nkind = if self.grid_terms { 2 } else { 1 };
            let gsize = self.group_atoms();
            let mut res = GpuXcHessResult {
                vfix_c: occ.iter().map(|&(_, o)| vec![0.0; n3 * n * o]).collect(),
                vgrid_c: if self.grid_terms {
                    occ.iter().map(|&(_, o)| vec![0.0; n3 * n * o]).collect()
                } else {
                    Vec::new()
                },
                hfix: vec![0.0; n3 * n3],
                hgrid: vec![0.0; n3 * n3],
            };
            let mut efix = DetAcc::new(gpu, n3 * n3)?;
            let mut e1 = DetAcc::new(gpu, n3 * n3)?;
            let nacc = nkind * nsp * 3 * gsize * npack;
            let mut vacc =
                gpu.alloc_zeros_granted::<u64>(self.workspace.as_ref(), detsum::words(nacc))?;
            let cm: Vec<DeviceBuf<f64>> = occ
                .iter()
                .map(|&(c, o)| {
                    gpu.upload_granted(
                        self.workspace.as_ref(),
                        if o == 0 { &[0.0] } else { &c[..n * o] },
                    )
                })
                .collect::<Result<_, _>>()?;
            let mut g0 = 0;
            while g0 < self.natom {
                let g1 = (g0 + gsize).min(self.natom);
                if g0 > 0 {
                    gpu.zero(&mut vacc)?;
                }
                self.main_pass(
                    gpu,
                    &dm,
                    &vu,
                    &wf,
                    (g0, g1),
                    g0 == 0,
                    &mut efix,
                    &mut e1,
                    &mut vacc,
                    nacc,
                )?;
                times.passes += 1;
                let tv = Instant::now();
                self.v_times_c(gpu, &vacc, nacc, (g0, g1), &cm, occ, &mut res)?;
                times.vc += tv.elapsed().as_secs_f64();
                g0 = g1;
            }
            let mut h = efix.to_f64(gpu)?;
            for p in 0..n3 {
                for q in 0..p {
                    let s = 0.5 * (h[p * n3 + q] + h[q * n3 + p]);
                    h[p * n3 + q] = s;
                    h[q * n3 + p] = s;
                }
            }
            res.hfix = h;
            if self.grid_terms {
                res.hgrid = e1.to_f64(gpu)?;
            }
            times.main = t.elapsed().as_secs_f64() - times.vc;
            times.total = t_all.elapsed().as_secs_f64();
            self.last = times;
            Ok(res)
        }

        /// Pass 1: rho, grad rho and (meta-GGAs) tau of every point per spin (XCP's density
        /// kernels).
        #[allow(clippy::type_complexity)]
        fn densities(
            &self,
            gpu: &Gpu,
            dm: &DeviceBuf<f64>,
        ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), GpuError> {
            let (n, nsp, npts) = (self.nbf, self.nsp, self.npts);
            let (gga, mgga) = (self.functional.kind().gga, self.functional.kind().mgga);
            let lay = &self.lay;
            let nc1 = self.nc1;
            let mut phi = gpu.alloc_zeros_granted::<f64>(
                self.workspace.as_ref(),
                (lay.max_block_nf * nc1 * lay.max_block_np).max(1),
            )?;
            // compressed at the layout's offsets (nc values per kept value), nc1 used
            let mut phic = gpu.alloc_zeros_granted::<f64>(
                self.workspace.as_ref(),
                (self.nc * lay.max_x).max(1),
            )?;
            let xw = gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), lay.max_x.max(1))?;
            // meta-GGAs: Y = grad phi D (3 per kept value) for tau
            let yw = gpu.alloc_zeros_granted::<f64>(
                self.workspace.as_ref(),
                if mgga { 3 * lay.max_x } else { 1 }.max(1),
            )?;
            let mut dv =
                gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), lay.max_d.max(1))?;
            let mut rho = gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), npts * nsp)?;
            let mut grad = gpu.alloc_zeros_granted::<f64>(
                self.workspace.as_ref(),
                if gga { npts * nsp * 3 } else { 1 },
            )?;
            let mut tau = gpu.alloc_zeros_granted::<f64>(
                self.workspace.as_ref(),
                if mgga { npts * nsp } else { 1 },
            )?;
            // per chunk, X = phi D (and Y = grad phi D) of its batches
            let plans = lay
                .chunks
                .iter()
                .map(|c| -> Result<[Prod; 2], GpuError> {
                    let prods = lay.products(c, |h, [_, xo, dd]| {
                        let (np, nf) = (h.np, h.global.len());
                        Some(VgemmProduct {
                            m: np,
                            n: nf,
                            k: nf,
                            a: phic.at(self.nc * xo).addr(),
                            lda: nc1 * np,
                            b: dv.at(dd).addr(),
                            ldb: nf,
                            c: xw.at(xo).addr(),
                            ldc: np,
                        })
                    });
                    let px = Prod::new(gpu, &lay.vg, self.fp64, &prods)?;
                    let py = if mgga {
                        let prods = lay.products(c, |h, [_, xo, dd]| {
                            let (np, nf) = (h.np, h.global.len());
                            Some(VgemmProduct {
                                m: 3 * np,
                                n: nf,
                                k: nf,
                                a: phic.at(self.nc * xo + np).addr(),
                                lda: nc1 * np,
                                b: dv.at(dd).addr(),
                                ldb: nf,
                                c: yw.at(3 * xo).addr(),
                                ldc: 3 * np,
                            })
                        });
                        Prod::new(gpu, &lay.vg, self.fp64, &prods)?
                    } else {
                        Prod::None
                    };
                    Ok([px, py])
                })
                .collect::<Result<Vec<_>, _>>()?;
            let pass = DensityPass {
                lay,
                colloc: Collocation::With(&self.colloc1, nc1),
                xyz: [&self.xg, &self.yg, &self.zg],
                d: Some(dm),
                nbf: n,
                nsp,
                scale: 1.0,
                kind: self.functional.kind(),
                gemm: &self.gemm,
            };
            let mut work = DensityWork {
                phi: &mut phi,
                phic: &mut phic,
                phic32: None,
                ops: DensityOps::Fp64 {
                    dv: &mut dv,
                    xw: &xw,
                    yw: &yw,
                },
            };
            let out = DensityOut {
                rho: &mut rho,
                grad: &mut grad,
                tau: &mut tau,
            };
            pass.run(
                gpu,
                &mut work,
                out,
                None::<PassTiming<'_, '_, GpuXcHessTimes>>,
                |ci| [&plans[ci][0], &plans[ci][1]],
                |_, _| Ok(()),
            )?;
            let rho = gpu.download(&rho)?;
            let grad = if gga {
                gpu.download(&grad)?
            } else {
                Vec::new()
            };
            let tau = if mgga {
                gpu.download(&tau)?
            } else {
                Vec::new()
            };
            Ok((rho, grad, tau))
        }

        /// The functional on the host (order 2, [`HostFunctional::second`]): v = df/du
        /// (unweighted, point-major, nu per point) and wF (point-major, nu^2 per point).
        fn functional(
            &self,
            rho: &[f64],
            grad: &[f64],
            tau: &[f64],
        ) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
            let (nsp, npts) = (self.nsp, self.npts);
            let kind = self.functional.kind();
            let (gga, mgga) = (kind.gga, kind.mgga);
            let ns = vars_per_spin(kind);
            let nu = nsp * ns;
            let parts = self
                .functional
                .second(nsp, npts, rho, grad, tau, |pts, out, gr| {
                    let np = pts.len();
                    // v (unweighted) per point
                    let mut v = vec![0.0; np * nu];
                    let vrho = out.vrho();
                    for g in 0..np {
                        for s in 0..nsp {
                            v[g * nu + s * ns] = vrho[g * nsp + s];
                        }
                    }
                    if gga {
                        let vs = out.vsigma();
                        for g in 0..np {
                            let gk = |s: usize, k: usize| gr[(s * 3 + k) * np + g];
                            for k in 0..3 {
                                if nsp == 1 {
                                    v[g * nu + 1 + k] = 2.0 * vs[g] * gk(0, k);
                                } else {
                                    // dsigma_t / dg_{s,k}: aa 2 g_a, ab g_b (s = a) or g_a
                                    // (s = b), bb 2 g_b
                                    let (aa, ab, bb) = (vs[3 * g], vs[3 * g + 1], vs[3 * g + 2]);
                                    v[g * nu + 1 + k] = aa * 2.0 * gk(0, k) + ab * gk(1, k);
                                    v[g * nu + ns + 1 + k] = ab * gk(0, k) + bb * 2.0 * gk(1, k);
                                }
                            }
                        }
                    }
                    if mgga {
                        let vt = out.vtau();
                        for g in 0..np {
                            for s in 0..nsp {
                                v[g * nu + s * ns + 4] = vt[g * nsp + s];
                            }
                        }
                    }
                    // wF, component-major per chunk, then point-major
                    let wfc = kernel_weights(out, nsp, gga, mgga, &self.w_host[pts], gr, np);
                    let mut wf = vec![0.0; np * nu * nu];
                    for c in 0..nu * nu {
                        for g in 0..np {
                            wf[g * nu * nu + c] = wfc[c * np + g];
                        }
                    }
                    (v, wf)
                })?;
            let (mut v, mut wf) = (
                Vec::with_capacity(npts * nu),
                Vec::with_capacity(npts * nu * nu),
            );
            for (a, b) in parts {
                v.extend_from_slice(&a);
                wf.extend_from_slice(&b);
            }
            Ok((v, wf))
        }

        /// One main pass: perturbations of the atoms in `group` into the potential
        /// accumulators; with `with_e` also E and E1.
        #[allow(clippy::too_many_arguments)]
        fn main_pass(
            &self,
            gpu: &Gpu,
            dm: &DeviceBuf<f64>,
            vu: &DeviceBuf<f64>,
            wf: &DeviceBuf<f64>,
            group: (usize, usize),
            with_e: bool,
            efix: &mut DetAcc,
            e1: &mut DetAcc,
            vacc: &mut DeviceBuf<u64>,
            nacc: usize,
        ) -> Result<(), GpuError> {
            let (n, nsp) = (self.nbf, self.nsp);
            let n3 = 3 * self.natom;
            let (nc, nx) = (self.nc, self.nx);
            let (gga, mgga) = (self.functional.kind().gga, self.functional.kind().mgga);
            let ns = vars_per_spin(self.functional.kind());
            let nu = nsp * ns;
            // the G operands' comps per spin and spin-free, the cross term's rows per point
            let (nab, nh, nr) = if mgga { (11, 9, 4) } else { (8, 0, 1) };
            let grid = self.grid_terms;
            let npack = n * (n + 1) / 2;
            let nxg = 3 * (group.1 - group.0);
            let lay = &self.lay;
            let ms = self.max_sizes;
            let al = |k: usize| gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), k.max(1));
            let mut phi = al(lay.max_block_nf * nc * lay.max_block_np)?;
            let mut phic = al(nc * lay.max_x)?;
            let mut phic32 = gpu.alloc_zeros_granted::<f32>(
                self.workspace.as_ref(),
                if self.single { nc * lay.max_x } else { 1 }.max(1),
            )?;
            let xstride = nx * lay.max_x;
            let xb = al(nsp * xstride)?;
            let dsub = al(nsp * lay.max_d)?;
            let mut ub = al(ms.u)?;
            let mut tb = al(ms.u)?;
            let mut sb = al(ms.s)?;
            let mut rb = al(ms.r)?;
            let mut tjb = al(ms.tj)?;
            let mut el = al(ms.el)?;
            let mut dwb = al(ms.dw)?;
            let mut ab = al(ms.ab)?;
            let gb = al(ms.g)?;
            let cc = al(ms.cc)?;
            let mut lb = al(ms.lr)?;
            let mut rbm = al(ms.lr)?;
            let mut tg = al(ms.tg)?;
            let xyz = [&self.xg, &self.yg, &self.zg];
            let ws = gpu.warp_size() as usize;
            let addr = |b: &DeviceBuf<f64>, off: usize| b.at(off).addr();
            for (ci, c) in lay.chunks.clone().into_iter().enumerate() {
                let nbj = c.j1 - c.j0;
                if nbj == 0 {
                    continue;
                }
                let host = &lay.host[c.j0..c.j1];
                let hbd = |j: usize, f: usize| self.hb_host[HB * j + f] as usize;
                let max_np = c.max_np;
                let max_nf = host.iter().map(|h| h.global.len()).max().unwrap_or(0);
                let max_nat = self.info[c.j0..c.j1]
                    .iter()
                    .map(|b| b.atoms.len())
                    .max()
                    .unwrap_or(0);
                let cs = self.chunk_sizes[ci];
                // collocation and compression
                for bi in c.b0..c.b1 {
                    let b = &lay.blocks[bi];
                    if b.nf == 0 {
                        continue;
                    }
                    lay.collocate(gpu, b, xyz, &mut phi)?;
                    let p32 = if self.single { Some(&mut phic32) } else { None };
                    lay.compress(gpu, b, &phi, Some(&mut phic), p32)?;
                }
                // X_s = phi[0..nx] D_s
                let (maxd, _) = lay.chunk_max(&c);
                for sp in 0..nsp {
                    // SAFETY: as in the density pass, into spin sp's block of dsub.
                    unsafe {
                        gpu.launch(
                            &self.lay.k.gather_d,
                            layout::per_batch(maxd, nbj, 256),
                            &[
                                Arg::I64(n as i64),
                                Arg::buf(&lay.bd),
                                Arg::I32(c.j0 as i32),
                                Arg::buf(&lay.gf),
                                Arg::from(dm.at(sp * n * n)),
                                Arg::from(dsub.at(sp * lay.max_d)),
                                Arg::Ptr(0),
                            ],
                        )
                    }?;
                    let prods = lay.products(&c, |h, [po, xo, dd]| {
                        let (np, nf) = (h.np, h.global.len());
                        Some(VgemmProduct {
                            m: nx * np,
                            n: nf,
                            k: nf,
                            a: addr(&phic, po),
                            lda: nc * np,
                            b: addr(&dsub, sp * lay.max_d + dd),
                            ldb: nf,
                            c: addr(&xb, sp * xstride + nx * xo),
                            ldc: nx * np,
                        })
                    });
                    let p = Prod::new(gpu, &self.lay.vg, self.fp64, &prods)?;
                    // SAFETY: the products address phic, dsub and xb within their sizes.
                    unsafe { p.run(gpu, &self.lay.vg, &self.gemm, false, 0.0) }?;
                }
                let j0 = Arg::I32(c.j0 as i32);
                let hb = Arg::buf(&self.hb);
                let ints = Arg::buf(&self.ints);
                if grid {
                    // the chunk's weight derivatives
                    let mut dw = Vec::with_capacity(cs.dw);
                    for b in &self.info[c.j0..c.j1] {
                        dw.extend_from_slice(&b.dw);
                    }
                    if !dw.is_empty() {
                        gpu.upload_into(&mut dwb, 0, &dw)?;
                    }
                    // SAFETY: matches xh_point; reads the batches' compressed values, X and
                    // the points' wF, writes t_j at each batch's offset (< the chunk's size).
                    unsafe {
                        gpu.launch(
                            &self.kern.point,
                            layout::per_batch(max_np, nbj, 128),
                            &[
                                Arg::I32(nc as i32),
                                Arg::I32(nx as i32),
                                Arg::I32(nsp as i32),
                                Arg::I32(ns as i32),
                                Arg::I32(gga as i32),
                                Arg::I32(mgga as i32),
                                hb,
                                j0,
                                Arg::buf(&phic),
                                Arg::buf(&xb),
                                Arg::I64(xstride as i64),
                                Arg::buf(wf),
                                Arg::buf_mut(&mut tjb),
                            ],
                        )
                    }?;
                }
                // SAFETY: matches xh_atom; one thread per (point, local atom) of each batch
                // reads its functions' values and X and the point's arrays, writes the
                // batch's u, t (s, r) at its offsets.
                unsafe {
                    gpu.launch(
                        &self.kern.atom,
                        layout::per_batch(max_np * max_nat, nbj, 128),
                        &[
                            Arg::I32(nc as i32),
                            Arg::I32(nx as i32),
                            Arg::I32(nsp as i32),
                            Arg::I32(ns as i32),
                            Arg::I32(gga as i32),
                            Arg::I32(mgga as i32),
                            Arg::I32(grid as i32),
                            hb,
                            j0,
                            ints,
                            Arg::buf(&phic),
                            Arg::buf(&xb),
                            Arg::I64(xstride as i64),
                            Arg::buf(&self.w),
                            Arg::buf(vu),
                            Arg::buf(wf),
                            Arg::buf(&tjb),
                            Arg::buf_mut(&mut ub),
                            Arg::buf_mut(&mut tb),
                            Arg::buf_mut(&mut sb),
                            Arg::buf_mut(&mut rb),
                        ],
                    )
                }?;
                if with_e {
                    // E: u^T t, the same-function terms, the cross term per spin
                    let prods: Vec<VgemmProduct> = (c.j0..c.j1)
                        .filter_map(|j| {
                            let np = hbd(j, H_NP);
                            let npl = 3 * hbd(j, H_NAT);
                            (npl > 0).then(|| VgemmProduct {
                                m: npl,
                                n: npl,
                                k: nu * np,
                                a: addr(&ub, hbd(j, H_U)),
                                lda: nu * np,
                                b: addr(&tb, hbd(j, H_U)),
                                ldb: nu * np,
                                c: addr(&el, hbd(j, H_EL)),
                                ldc: npl,
                            })
                        })
                        .collect();
                    let p = Prod::new(gpu, &self.lay.vg, self.fp64, &prods)?;
                    // SAFETY: the products address ub, tb and el within the chunk's sizes.
                    unsafe { p.run(gpu, &self.lay.vg, &self.gemm, true, 0.0) }?;
                    // SAFETY: matches xh_same; a warp per (atom, pair) of each batch reads
                    // the batch's arrays and adds to its el block.
                    unsafe {
                        gpu.launch(
                            &self.kern.same,
                            layout::per_batch(6 * max_nat * ws, nbj, 256),
                            &[
                                Arg::I32(nc as i32),
                                Arg::I32(nx as i32),
                                Arg::I32(nsp as i32),
                                Arg::I32(ns as i32),
                                Arg::I32(gga as i32),
                                Arg::I32(mgga as i32),
                                hb,
                                j0,
                                ints,
                                Arg::buf(&phic),
                                Arg::buf(&xb),
                                Arg::I64(xstride as i64),
                                Arg::buf(&self.w),
                                Arg::buf(vu),
                                Arg::buf_mut(&mut el),
                            ],
                        )
                    }?;
                    let cprods: Vec<VgemmProduct> = (c.j0..c.j1)
                        .filter_map(|j| {
                            let (np, nf) = (hbd(j, H_NP), hbd(j, H_NF));
                            (nf > 0).then(|| VgemmProduct {
                                m: 3 * nf,
                                n: 3 * nf,
                                k: nr * np,
                                a: addr(&lb, hbd(j, H_LR)),
                                lda: nr * np,
                                b: addr(&rbm, hbd(j, H_LR)),
                                ldb: nr * np,
                                c: addr(&cc, hbd(j, H_CC)),
                                ldc: 3 * nf,
                            })
                        })
                        .collect();
                    let pc = Prod::new(gpu, &self.lay.vg, self.fp64, &cprods)?;
                    let max_npl = 3 * max_nat;
                    for sp in 0..nsp {
                        // SAFETY: matches xh_lr; writes each batch's L and R at its offset.
                        unsafe {
                            gpu.launch(
                                &self.kern.lr,
                                layout::per_batch(3 * max_nf * max_np, nbj, 256),
                                &[
                                    Arg::I32(nc as i32),
                                    Arg::I32(nsp as i32),
                                    Arg::I32(ns as i32),
                                    Arg::I32(gga as i32),
                                    Arg::I32(mgga as i32),
                                    Arg::I32(sp as i32),
                                    hb,
                                    j0,
                                    Arg::buf(&phic),
                                    Arg::buf(&self.w),
                                    Arg::buf(vu),
                                    Arg::buf_mut(&mut lb),
                                    Arg::buf_mut(&mut rbm),
                                ],
                            )
                        }?;
                        // SAFETY: the products address lb, rbm and cc within their sizes.
                        unsafe { pc.run(gpu, &self.lay.vg, &self.gemm, true, 0.0) }?;
                        // SAFETY: matches xh_cross; reads the batch's D_sp, C and el, writes
                        // el or the accumulator (indices below (3N)^2).
                        unsafe {
                            gpu.launch(
                                &self.kern.cross,
                                layout::per_batch(max_npl * max_npl, nbj, 128),
                                &[
                                    Arg::I32((sp + 1 == nsp) as i32),
                                    Arg::I64(n3 as i64),
                                    hb,
                                    j0,
                                    ints,
                                    Arg::from(dsub.at(sp * lay.max_d)),
                                    Arg::buf(&cc),
                                    Arg::buf_mut(&mut el),
                                    efix.arg(),
                                    Arg::I64((n3 * n3) as i64),
                                ],
                            )
                        }?;
                    }
                    if grid {
                        let max_ng = self.info[c.j0..c.j1]
                            .iter()
                            .map(|b| b.gat.len())
                            .max()
                            .unwrap_or(0);
                        // SAFETY: matches xh_e1; reads the batches' s, r and dw, adds to the
                        // (3N)^2 accumulator.
                        unsafe {
                            gpu.launch(
                                &self.kern.e1,
                                layout::per_batch(9 * max_nat * max_ng, nbj, 128),
                                &[
                                    Arg::I64(n3 as i64),
                                    hb,
                                    j0,
                                    ints,
                                    Arg::buf(&self.owner),
                                    Arg::buf(&sb),
                                    Arg::buf(&rb),
                                    Arg::buf(&dwb),
                                    e1.arg(),
                                    Arg::I64((n3 * n3) as i64),
                                ],
                            )
                        }?;
                    }
                }
                // V: G per owner, the grid slots' t, the slots
                // SAFETY: matches xh_ab; writes each batch's operands at its offset.
                unsafe {
                    gpu.launch(
                        &self.kern.ab,
                        layout::per_batch(nsp * max_nf * max_np, nbj, 256),
                        &[
                            Arg::I32(nc as i32),
                            Arg::I32(nsp as i32),
                            Arg::I32(ns as i32),
                            Arg::I32(gga as i32),
                            Arg::I32(mgga as i32),
                            hb,
                            j0,
                            ints,
                            Arg::buf(&phic),
                            Arg::buf(&self.w),
                            Arg::buf(vu),
                            Arg::buf_mut(&mut ab),
                        ],
                    )
                }?;
                let mut g1 = Vec::new();
                let mut g2 = Vec::new();
                let mut g3 = Vec::new();
                for (jj, h) in host.iter().enumerate() {
                    let j = c.j0 + jj;
                    let (np, nf) = (h.np, h.global.len());
                    if nf == 0 {
                        continue;
                    }
                    let no = hbd(j, H_NO);
                    let optr = &self.info[j].optr;
                    let _ = np;
                    for o in 0..no {
                        let (q0, ko) = (optr[o] as usize, (optr[o + 1] - optr[o]) as usize);
                        // the owner's block of operands
                        let ob = hbd(j, H_AB) + (nsp * nab + nh) * nf * q0;
                        for sp in 0..nsp {
                            // the owner's operands: A_i, B, phi, d_k phi (nf x ko each)
                            let abo = ob + sp * nab * nf * ko;
                            let cs = nf * ko;
                            for i in 0..3 {
                                let go = hbd(j, H_G) + ((o * nsp + sp) * 3 + i) * nf * nf;
                                g1.push(VgemmProduct {
                                    m: nf,
                                    n: nf,
                                    k: ko,
                                    a: addr(&ab, abo + i * cs),
                                    lda: ko,
                                    b: addr(&ab, abo + 4 * cs),
                                    ldb: ko,
                                    c: addr(&gb, go),
                                    ldc: nf,
                                });
                                if gga {
                                    // (d_i phi)^T B
                                    g2.push(VgemmProduct {
                                        m: nf,
                                        n: nf,
                                        k: ko,
                                        a: addr(&ab, abo + (5 + i) * cs),
                                        lda: ko,
                                        b: addr(&ab, abo + 3 * cs),
                                        ldb: ko,
                                        c: addr(&gb, go),
                                        ldc: nf,
                                    });
                                }
                                if mgga {
                                    // sum_k H_ki^T (1/2 wv_tau d_k phi), 3 ko deep
                                    g3.push(VgemmProduct {
                                        m: nf,
                                        n: nf,
                                        k: 3 * ko,
                                        a: addr(&ab, ob + (nsp * nab + 3 * i) * cs),
                                        lda: 3 * ko,
                                        b: addr(&ab, abo + 8 * cs),
                                        ldb: 3 * ko,
                                        c: addr(&gb, go),
                                        ldc: nf,
                                    });
                                }
                            }
                        }
                    }
                }
                let p1 = Prod::new(gpu, &self.lay.vg, self.fp64, &g1)?;
                let p2 = Prod::new(gpu, &self.lay.vg, self.fp64, &g2)?;
                // SAFETY: the products address ab, phic and gb within their sizes.
                unsafe {
                    p1.run(gpu, &self.lay.vg, &self.gemm, true, 0.0)?;
                    p2.run(gpu, &self.lay.vg, &self.gemm, true, 1.0)?;
                }
                if mgga {
                    let p3 = Prod::new(gpu, &self.lay.vg, self.fp64, &g3)?;
                    // SAFETY: as above.
                    unsafe { p3.run(gpu, &self.lay.vg, &self.gemm, true, 1.0) }?;
                }
                if grid {
                    let max_ng = self.info[c.j0..c.j1]
                        .iter()
                        .map(|b| b.gat.len())
                        .max()
                        .unwrap_or(0);
                    // SAFETY: matches xh_ttg; writes each batch's grid t at its offset.
                    unsafe {
                        gpu.launch(
                            &self.kern.ttg,
                            layout::per_batch(max_ng * 3 * nu * max_np, nbj, 256),
                            &[
                                Arg::I32(nu as i32),
                                hb,
                                j0,
                                ints,
                                Arg::buf(&self.owner),
                                Arg::buf(vu),
                                Arg::buf(&tjb),
                                Arg::buf(&dwb),
                                Arg::buf_mut(&mut tg),
                            ],
                        )
                    }?;
                }
                // the slots of the group's perturbations
                let mut sd: Vec<i64> = Vec::new();
                for (jj, h) in host.iter().enumerate() {
                    let j = c.j0 + jj;
                    if h.global.is_empty() {
                        continue;
                    }
                    let b = &self.info[j];
                    let np = h.np;
                    for (a, &at) in b.atoms.iter().enumerate() {
                        let at = at as usize;
                        if at < group.0 || at >= group.1 {
                            continue;
                        }
                        for i in 0..3 {
                            for sp in 0..nsp {
                                let to = hbd(j, H_U) + ((3 * a + i) * nu + sp * ns) * np;
                                let x = 3 * (at - group.0) + i;
                                sd.extend_from_slice(&[
                                    j as i64,
                                    addr(&tb, to) as i64,
                                    ((sp * nxg + x) * npack) as i64,
                                    0,
                                    a as i64,
                                    i as i64,
                                    sp as i64,
                                    -1,
                                    np as i64,
                                    0,
                                ]);
                            }
                        }
                    }
                    if grid {
                        for (ga, &at) in b.gat.iter().enumerate() {
                            let at = at as usize;
                            if at < group.0 || at >= group.1 {
                                continue;
                            }
                            let o = b.own.iter().position(|&x| x as usize == at);
                            let (p0, p1) = (b.gptr[ga] as usize, b.gptr[ga + 1] as usize);
                            if p1 == p0 {
                                continue;
                            }
                            for jc in 0..3 {
                                for sp in 0..nsp {
                                    let to = hbd(j, H_TG) + ((ga * 3 + jc) * nu + sp * ns) * np;
                                    let x = 3 * (at - group.0) + jc;
                                    sd.extend_from_slice(&[
                                        j as i64,
                                        addr(&tg, to) as i64,
                                        (((nsp + sp) * nxg + x) * npack) as i64,
                                        1,
                                        o.map_or(-1, |o| o as i64),
                                        jc as i64,
                                        sp as i64,
                                        hbd(j, H_GPT) as i64 + p0 as i64,
                                        (p1 - p0) as i64,
                                        0,
                                    ]);
                                }
                            }
                        }
                    }
                }
                let nslots = sd.len() / SD;
                if nslots > 0 {
                    let sdd = gpu.upload_granted(self.workspace.as_ref(), &sd)?;
                    let nt = max_nf.div_ceil(VT);
                    let mut s0 = 0;
                    while s0 < nslots {
                        let cnt = (nslots - s0).min(65535);
                        let cfg = Launch {
                            grid: ((nt * nt) as u32, cnt as u32, 1),
                            block: (128, 1, 1),
                            shared_bytes: 0,
                        };
                        // SAFETY: matches xh_vslots; each slot's batch arrays (compressed
                        // values, t, G) lie within the chunk's workspaces, its accumulator
                        // elements below nacc (base + packed index < nkind nsp nxg npack).
                        unsafe {
                            gpu.launch(
                                match (self.single, mgga) {
                                    (false, false) => &self.kern.vslots,
                                    (true, false) => &self.kern.vslots32,
                                    (false, true) => &self.kern.vslots_mg,
                                    (true, true) => &self.kern.vslots32_mg,
                                },
                                cfg,
                                &[
                                    Arg::I32(nc as i32),
                                    Arg::I32(nsp as i32),
                                    Arg::I32(gga as i32),
                                    Arg::I64(n as i64),
                                    hb,
                                    Arg::from(sdd.at(SD * s0)),
                                    ints,
                                    Arg::buf(&lay.gf),
                                    if self.single {
                                        Arg::buf(&phic32)
                                    } else {
                                        Arg::buf(&phic)
                                    },
                                    Arg::buf(&gb),
                                    Arg::buf_mut(vacc),
                                    Arg::I64(nacc as i64),
                                ],
                            )
                        }?;
                        s0 += cnt;
                    }
                }
            }
            Ok(())
        }

        /// V C for the group's accumulated potentials: unpacked in batches of matrices,
        /// multiplied by each channel's occupied orbitals, copied into the result.
        #[allow(clippy::too_many_arguments)]
        fn v_times_c(
            &self,
            gpu: &Gpu,
            vacc: &DeviceBuf<u64>,
            nacc: usize,
            group: (usize, usize),
            cm: &[DeviceBuf<f64>],
            occ: &[(&[f64], usize)],
            res: &mut GpuXcHessResult,
        ) -> Result<(), GpuError> {
            let (n, nsp) = (self.nbf, self.nsp);
            let npack = n * (n + 1) / 2;
            let nxg = 3 * (group.1 - group.0);
            let nkind = if self.grid_terms { 2 } else { 1 };
            let per = (128usize << 20) / (8 * n * n).max(1);
            let cnt_max = per.clamp(1, nxg.max(1));
            let omax = occ.iter().map(|o| o.1).max().unwrap_or(0);
            let mut full =
                gpu.alloc_zeros_granted::<f64>(self.workspace.as_ref(), (cnt_max * n * n).max(1))?;
            let mut out = gpu
                .alloc_zeros_granted::<f64>(self.workspace.as_ref(), (cnt_max * n * omax).max(1))?;
            for kind in 0..nkind {
                for sp in 0..nsp {
                    let o = occ[sp].1;
                    if o == 0 {
                        continue;
                    }
                    let mut x0 = 0;
                    while x0 < nxg {
                        let cnt = (nxg - x0).min(cnt_max);
                        let base = ((kind * nsp + sp) * nxg + x0) * npack;
                        // SAFETY: matches xh_unpack; reads cnt packed matrices below nacc,
                        // writes cnt n^2 values (<= full's size).
                        unsafe {
                            gpu.launch(
                                &self.kern.unpack,
                                Launch::linear((cnt * n * n).min(1 << 24), 256),
                                &[
                                    Arg::I64(n as i64),
                                    Arg::I64(npack as i64),
                                    Arg::I64(base as i64),
                                    Arg::I32(cnt as i32),
                                    Arg::buf(vacc),
                                    Arg::I64(nacc as i64),
                                    Arg::buf_mut(&mut full),
                                ],
                            )
                        }?;
                        self.gemm.gemm_strided(
                            Arith::Fp64,
                            GemmDims::new(Op::N, Op::N, n, o, n).batch(cnt),
                            (1.0, 0.0),
                            MatRef::new(&full, 0, n).stride(n * n),
                            MatRef::new(&cm[sp], 0, n),
                            MatMut::new(&mut out, 0, n).stride(n * o),
                        )?;
                        let h = gpu.download_range(&out, 0, cnt * n * o)?;
                        let dst = if kind == 0 {
                            &mut res.vfix_c[sp]
                        } else {
                            &mut res.vgrid_c[sp]
                        };
                        let xg0 = 3 * group.0 + x0;
                        dst[xg0 * n * o..(xg0 + cnt) * n * o].copy_from_slice(&h);
                        x0 += cnt;
                    }
                }
            }
            Ok(())
        }
    }
}
