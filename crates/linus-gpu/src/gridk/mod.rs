//! Grid-based exchange on the device: the per-build part of linus-jk's compensated grid-K
//! (`linus_jk::gridk`). The host hands over the dressed collocation Xbar (npts x nbf; the
//! overlap fit already applied) and the grid; a build of a difference density D_d is
//!
//! 1. Y = Xbar (D_d T) (one DGEMM; T the spherical-to-Cartesian transform, so the integral
//!    kernels work on Cartesian rows),
//! 2. per chunk and shell the max |Y| (the screening of the integral kernels),
//! 3. Z = w V(Y) by the potential-integral kernels, one per class of shell pairs
//!    ([`crate::kernels::gridk`]),
//! 4. G = Xbar^T Z (one DGEMM), downloaded; the host forms Kbar = sym(G T^T).
//!
//! The GEMMs are cuBLAS's FP64 (DMMA on data-center cards); everything is deterministic run to
//! run (the kernels write each Z element from one thread in a fixed order).

use crate::cpu::{BasisSet, Operator, PotPair};
use crate::kernels::gridk::{hermite_e, PAIR, PRIM};
#[cfg(feature = "gpu")]
use crate::{
    cpu::MolecularGrid,
    kernels::gridk::{generate, kernel_name, Style, THREADS},
};

/// The shell pairs of one class (bra l >= ket l), packed for the kernel.
#[derive(Clone, Debug, Default)]
pub struct ClassPairs {
    pub la: usize,
    pub lb: usize,
    pub info: Vec<i32>,
    pub est: Vec<f64>,
    pub prim: Vec<f64>,
    /// Each primitive pair's Hermite coefficients ([`crate::kernels::gridk::hermite_e`]).
    pub ecoef: Vec<f64>,
}

impl ClassPairs {
    pub fn npair(&self) -> usize {
        self.est.len()
    }
}

/// The significant shell pairs of `basis` by class ([`PotPair::significant`] at `prim_cut`, as
/// the CPU's grid-K), with first Cartesian rows `coff`.
pub fn pack_pairs(basis: &BasisSet, coff: &[usize], prim_cut: f64) -> Vec<ClassPairs> {
    let lmax = basis.shells.iter().map(|s| s.l).max().unwrap_or(0);
    let mut classes: Vec<ClassPairs> = Vec::new();
    for la in 0..=lmax {
        for lb in 0..=la {
            classes.push(ClassPairs {
                la,
                lb,
                ..Default::default()
            });
        }
    }
    let cls = |la: usize, lb: usize| la * (la + 1) / 2 + lb;
    for p in PotPair::significant(basis, prim_cut) {
        let (i, j) = (p.sa, p.sb);
        let est = p.est;
        // PotPair's bra is the shell with the larger l (the first on a tie)
        let (bra, ket) = if p.la < p.lb { (j, i) } else { (i, j) };
        let (sa, sb) = (&basis.shells[bra], &basis.shells[ket]);
        let c = &mut classes[cls(sa.l, sb.l)];
        let p0 = c.prim.len() / PRIM;
        for q in &p.prims {
            let pb = [
                q.pa[0] + sa.center[0] - sb.center[0],
                q.pa[1] + sa.center[1] - sb.center[1],
                q.pa[2] + sa.center[2] - sb.center[2],
            ];
            c.prim.extend_from_slice(&[
                q.p, q.coef, q.pc[0], q.pc[1], q.pc[2], q.pa[0], q.pa[1], q.pa[2], pb[0], pb[1],
                pb[2], q.est,
            ]);
            c.ecoef.extend(hermite_e(sa.l, sb.l, q.pa, pb, q.p));
        }
        let p1 = c.prim.len() / PRIM;
        c.info.extend_from_slice(&[
            bra as i32,
            ket as i32,
            coff[bra] as i32,
            coff[ket] as i32,
            p0 as i32,
            p1 as i32,
        ]);
        c.est.push(est);
    }
    debug_assert!(classes.iter().all(|c| c.info.len() == PAIR * c.npair()));
    classes.retain(|c| c.npair() > 0);
    classes
}

/// Bytes the extra copies of Z of a [`slices`] split may take on the device.
pub const SLICE_BYTES_MAX: usize = 4 << 30;
/// Thread blocks per SM a class kernel's launch aims at with its slices.
pub const BLOCKS_PER_SM: usize = 8;
/// Slices of a class's pairs at most.
pub const SLICES_MAX: usize = 16;

/// Slices of every class's pairs (blockIdx.y; each slice writes its own copy of Z, summed in
/// slice order after the kernels) for `nchunk` chunks of points on a device of `sm_count` SMs:
/// about [`BLOCKS_PER_SM`] blocks per SM (a small molecule's chunks alone leave the device
/// mostly idle), at most [`SLICES_MAX`], the extra copies of Z (`zlen` elements each) within
/// [`SLICE_BYTES_MAX`]. The slices order Z's sums, so they are part of the device's bits.
pub fn slices(sm_count: usize, nchunk: usize, zlen: usize) -> usize {
    let want = (BLOCKS_PER_SM * sm_count).div_ceil(nchunk.max(1));
    let room = (SLICE_BYTES_MAX / (8 * zlen.max(1))).max(1);
    want.clamp(1, SLICES_MAX).min(room)
}

/// Device bytes of a grid-K engine on `npts` points for `nbf` functions (`ncart` Cartesian
/// components) with `nslice` slices: the larger of its setup's peak (the collocation X with the
/// overlap fit's X A and W X A: 3 npts nbf) and what it keeps (Xbar, Y and Z's copies, the
/// points, a build's matrices). The pair tables are not counted (megabytes).
pub fn device_bytes(npts: usize, nbf: usize, ncart: usize, nslice: usize) -> usize {
    let setup = 3 * npts * nbf;
    let kept =
        npts * nbf + npts * ncart * (1 + nslice) + 4 * npts + 2 * nbf * nbf + 3 * nbf * ncart;
    8 * setup.max(kept)
}

/// The operator's (coulomb, long_range, omega) for the kernels.
pub fn operator_args(op: &Operator) -> (f64, f64, f64) {
    (op.coulomb(), op.long_range(), op.omega())
}

#[cfg(feature = "gpu")]
pub use device::*;

#[cfg(feature = "gpu")]
mod device;
