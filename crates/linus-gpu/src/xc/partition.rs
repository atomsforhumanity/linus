//! The grid's partition weights on the device (feature `gpu`): linus-grid's
//! [`DevicePartition`] for its unscreened partitions (Becke's, of PySCF's grids; LKO and SSF
//! when unscreened). The host evaluates every atom pair at every point (O(N^2) per point). The
//! device screens the atoms per point (`grid_partition_tiled`): a cell function P_b is a product
//! of factors in [0, 1], so its product over the atoms nearest the point's nearest atom m bounds
//! it; the atoms whose bound falls below 2^-100 times (an estimate of) P_m, itself at most the
//! sum of the cell functions, are dropped, and the kept atoms' cell functions are the host's
//! products, operation for operation (compiled without floating-point contraction). The weights
//! are the host's but for the dropped terms (below 2^-100 of the sum each: no weight's bits
//! changed in the tests) and the same from run to run; O(N) cell functions per point instead of
//! O(N^2) pairs. The points go in tiles of a warp's width, spatially compact (along a Morton
//! curve): each atom kept for one of the tile's points gets its cell function computed for all
//! of them, one read of its row of the pair tables per tile. `grid_partition` (every pair, the
//! host's bits exactly) serves molecules too large for the screened kernel's shared memory.

use crate::cpu::{DevicePartition, PairScheme, PairTables};
use crate::kernels::registry::{KernelModule, COMMON_H};
use crate::{Arg, Gpu, Launch};
use std::borrow::Cow;

const KERNELS: &str = include_str!("../../device/xc/partition.cu");

/// The partition's module (compiled without contraction: `--fmad=false`).
pub(crate) const MODULE: KernelModule = KernelModule::new(
    "xc_partition",
    || Cow::Owned(format!("{COMMON_H}\n{KERNELS}")),
    &["grid_partition", "grid_partition_tiled"],
);

/// The screened kernel's bound: the atoms nearest the point's nearest atom whose factors bound
/// each cell function.
const BOUND_ATOMS: usize = 16;

/// The screened kernel's threshold: a cell function whose bound falls below EPS times the
/// nearest atom's (at most the sum of them all) is dropped: 2^-100.
const EPS: f64 = 7.888609052210118e-31;

/// Shared memory a block of the screened kernel may use (the default limit).
const SHARED: usize = 48 << 10;

/// Bytes of the per-point scratch (distances and cell functions) a launch of the every-pair
/// kernel may hold.
const SCRATCH: usize = 256 << 20;

/// Bytes of the screened kernel's tile scratch (distances and cell functions of a tile's points
/// per atom, per block).
const TILE_SCRATCH: usize = 512 << 20;

/// Warps per block of the screened kernel (they share out the atoms of a tile).
const TILE_WARPS: usize = 8;

/// The grid builder's device partition on `gpu` ([`DevicePartition`]). Each evaluation takes
/// the device memory it needs as a grant first (`Gpu::grant`): out of memory, it fails before
/// any work with the device's error.
pub struct GpuPartition<'a> {
    pub gpu: &'a Gpu,
    /// The device bytes the partition may take (linus: its memory plan's `grid_partition`, the
    /// share the SCF's DF-J/K setup beside it leaves); a grant above it is the device's
    /// out-of-memory error before any work, whatever the memory free at the time. None: the
    /// device's free memory alone.
    pub allowance: Option<usize>,
}

impl GpuPartition<'_> {
    /// `bytes` of device memory for one evaluation, within the allowance and the device.
    fn grant(&self, bytes: usize) -> Result<crate::pool::Grant, crate::GpuError> {
        if let Some(a) = self.allowance.filter(|&a| bytes > a) {
            return Err(crate::GpuError::OutOfMemory {
                requested: bytes,
                in_use: 0,
                limit: a,
            });
        }
        self.gpu.grant(bytes, "grid partition")
    }

    /// The weights: screened per point where a tile's tables fit the shared memory, else from
    /// every atom pair; or the device's error.
    pub fn weights(
        &self,
        t: &PairTables,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        owner: &[u32],
        vol: &[f64],
    ) -> Result<Vec<f64>, crate::GpuError> {
        match self.screened(t, x, y, z, owner, vol)? {
            Some((w, _)) => Ok(w),
            None => self.every_pair(t, x, y, z, owner, vol),
        }
    }

    /// The weights with the atoms screened per point (`grid_partition_tiled`) and the average
    /// number of atoms whose cell functions are computed per point; None when a tile's tables
    /// exceed the shared memory (above about 4500 atoms).
    pub fn screened(
        &self,
        t: &PairTables,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        owner: &[u32],
        vol: &[f64],
    ) -> Result<Option<(Vec<f64>, f64)>, crate::GpuError> {
        let gpu = self.gpu;
        let n = t.xyz.len();
        let np = x.len();
        if [y.len(), z.len(), owner.len(), vol.len()] != [np; 4] || t.inv_r.len() != n * n {
            return Err(crate::GpuError::Contract("partition: array sizes".into()));
        }
        // the block's shared memory as the kernel lays it out: n masks and shorts, per warp and
        // point a double and an int, per point two doubles and two ints, the union's size
        let ws = gpu.warp_size() as usize;
        let tile_shared =
            10 * n + 12 * TILE_WARPS * ws + 4 * ((TILE_WARPS * ws) % 2) + 24 * ws + 4 + 16;
        if tile_shared > SHARED || n > u16::MAX as usize {
            return Ok(None);
        }
        if np == 0 {
            return Ok(Some((Vec::new(), 0.0)));
        }
        let (scheme, half) = scheme(t);
        let adjust = !t.a.is_empty();
        let kn = BOUND_ATOMS.min(n);
        // blocks within TILE_SCRATCH bytes of scratch (two n x W tiles of doubles each)
        let blocks = (TILE_SCRATCH / (16 * n * ws))
            .clamp(1, 4096)
            .min(np.div_ceil(ws));
        // the device memory of the buffers below: per point x, y, z, vol and out (f64), owner and
        // the order (u32); the atoms, the mirrored tables and neighbour lists, the count, the
        // tiles' scratch
        let bytes = 48 * np
            + 24 * n
            + 8 * n * n
            + 8 * if adjust { n * n } else { 1 }
            + 4 * n * kn
            + 8
            + 16 * blocks * n * ws;
        let grant = self.grant(bytes)?;
        let g = Some(&grant);
        // the tables mirrored from their lower triangles (the host's pair loop reads (i, j),
        // i > j): row b holds b's pairs, read along the row
        let mirror = |v: &[f64]| -> Vec<f64> {
            let mut m = vec![0.0; n * n];
            for b in 0..n {
                for c in 0..n {
                    if b != c {
                        m[b * n + c] = v[b.max(c) * n + b.min(c)];
                    }
                }
            }
            m
        };
        let ms = mirror(&t.inv_r);
        let am = if adjust { mirror(&t.a) } else { vec![0.0] };
        // per atom, the BOUND_ATOMS atoms nearest it (itself first; ties by index)
        let nbr: Vec<i32> = {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .flat_map_iter(|a| {
                    let r2 = |c: usize| {
                        let (p, q) = (t.xyz[a], t.xyz[c]);
                        (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)
                    };
                    let cmp = |b: &usize, c: &usize| r2(*b).total_cmp(&r2(*c)).then(b.cmp(c));
                    let mut ord: Vec<usize> = (0..n).collect();
                    if kn < n {
                        ord.select_nth_unstable_by(kn - 1, cmp);
                    }
                    ord.truncate(kn);
                    ord.sort_by(cmp);
                    ord.into_iter().map(|c| c as i32)
                })
                .collect()
        };
        let axyz: Vec<f64> = t.xyz.iter().flatten().copied().collect();
        let dx = gpu.upload_granted(g, x)?;
        let dy = gpu.upload_granted(g, y)?;
        let dz = gpu.upload_granted(g, z)?;
        let down = gpu.upload_granted(g, owner)?;
        let dvol = gpu.upload_granted(g, vol)?;
        let dxyz = gpu.upload_granted(g, &axyz)?;
        let dms = gpu.upload_granted(g, &ms)?;
        let dam = gpu.upload_granted(g, &am)?;
        let dnbr = gpu.upload_granted(g, &nbr)?;
        let mut out = gpu.alloc_zeros_granted::<f64>(g, np)?;
        let mut stats = gpu.alloc_zeros_granted::<u64>(g, 1)?;
        // the order: the points along a Morton curve of the atoms' box (spatially compact tiles
        // share most of their kept atoms), ties by index
        let perm = gpu.upload_granted(g, &morton_order(t, x, y, z))?;
        let kt = gpu.kernel_with(
            MODULE.name,
            &MODULE.source(),
            "grid_partition_tiled",
            &["--fmad=false"],
        )?;
        let mut dsc = gpu.alloc_zeros_granted::<f64>(g, blocks * n * ws)?;
        let mut psc = gpu.alloc_zeros_granted::<f64>(g, blocks * n * ws)?;
        within(&grant)?;
        // SAFETY: matches grid_partition_tiled: np points in the order perm (a permutation of
        // 0..np), out written at each; x, y, z, owner (< n) and vol of np points; ms and am
        // n x n (am one value without a size adjustment, not read), nbr n x kn atoms < n; each
        // block's scratch n W doubles in dsc and in psc; tile_shared bytes of shared memory as
        // the kernel lays them out for TILE_WARPS warps.
        unsafe {
            gpu.launch(
                &kt,
                Launch {
                    grid: (blocks as u32, 1, 1),
                    block: ((TILE_WARPS * ws) as u32, 1, 1),
                    shared_bytes: tile_shared as u32,
                },
                &[
                    Arg::I64(np as i64),
                    Arg::I32(n as i32),
                    Arg::I32(scheme),
                    Arg::I32(adjust as i32),
                    Arg::F64(half),
                    Arg::I32(kn as i32),
                    Arg::F64(EPS),
                    Arg::buf(&perm),
                    Arg::buf(&dx),
                    Arg::buf(&dy),
                    Arg::buf(&dz),
                    Arg::buf(&down),
                    Arg::buf(&dvol),
                    Arg::buf(&dxyz),
                    Arg::buf(&dms),
                    Arg::buf(&dam),
                    Arg::buf(&dnbr),
                    Arg::buf_mut(&mut dsc),
                    Arg::buf_mut(&mut psc),
                    Arg::buf_mut(&mut out),
                    Arg::buf_mut(&mut stats),
                ],
            )
        }?;
        let kept = gpu.download(&stats)?[0];
        Ok(Some((gpu.download(&out)?, kept as f64 / np as f64)))
    }

    /// The weights from every atom pair at every point (`grid_partition`: the host's bits).
    pub fn every_pair(
        &self,
        t: &PairTables,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        owner: &[u32],
        vol: &[f64],
    ) -> Result<Vec<f64>, crate::GpuError> {
        let gpu = self.gpu;
        let n = t.xyz.len();
        let np = x.len();
        if [y.len(), z.len(), owner.len(), vol.len()] != [np; 4] || t.inv_r.len() != n * n {
            return Err(crate::GpuError::Contract("partition: array sizes".into()));
        }
        if np == 0 {
            return Ok(Vec::new());
        }
        let k = gpu.kernel_with(
            MODULE.name,
            &MODULE.source(),
            "grid_partition",
            &["--fmad=false"],
        )?;
        let (scheme, half) = scheme(t);
        let adjust = !t.a.is_empty();
        // points per launch: the scratch within SCRATCH
        let ld = (SCRATCH / (16 * n)).clamp(1024, np.max(1024));
        // the device memory of the buffers below: per point x, y, z, vol and out (f64) and owner
        // (u32); the atoms and the tables; the scratch
        let bytes = 44 * np + 24 * n + 8 * n * n + 8 * if adjust { n * n } else { 1 } + 16 * n * ld;
        let grant = self.grant(bytes)?;
        let g = Some(&grant);
        let axyz: Vec<f64> = t.xyz.iter().flatten().copied().collect();
        let dx = gpu.upload_granted(g, x)?;
        let dy = gpu.upload_granted(g, y)?;
        let dz = gpu.upload_granted(g, z)?;
        let down = gpu.upload_granted(g, owner)?;
        let dvol = gpu.upload_granted(g, vol)?;
        let dxyz = gpu.upload_granted(g, &axyz)?;
        let dinv = gpu.upload_granted(g, &t.inv_r)?;
        let da = gpu.upload_granted(g, if adjust { &t.a } else { &[0.0][..] })?;
        let mut d = gpu.alloc_zeros_granted::<f64>(g, n * ld)?;
        let mut p = gpu.alloc_zeros_granted::<f64>(g, n * ld)?;
        let mut out = gpu.alloc_zeros_granted::<f64>(g, np)?;
        within(&grant)?;
        for k0 in (0..np).step_by(ld) {
            let m = ld.min(np - k0);
            // SAFETY: matches grid_partition: points k0..k0 + m of x, y, z, owner and vol in,
            // of out written; the scratch holds n ld values each; owners < n (the grid's atoms).
            unsafe {
                gpu.launch(
                    &k,
                    Launch::linear(m, 128),
                    &[
                        Arg::I64(m as i64),
                        Arg::I64(k0 as i64),
                        Arg::I32(n as i32),
                        Arg::I32(scheme),
                        Arg::I32(adjust as i32),
                        Arg::F64(half),
                        Arg::buf(&dx),
                        Arg::buf(&dy),
                        Arg::buf(&dz),
                        Arg::buf(&down),
                        Arg::buf(&dvol),
                        Arg::buf(&dxyz),
                        Arg::buf(&dinv),
                        Arg::buf(&da),
                        Arg::buf_mut(&mut d),
                        Arg::buf_mut(&mut p),
                        Arg::I64(ld as i64),
                        Arg::buf_mut(&mut out),
                    ],
                )
            }?;
        }
        gpu.download(&out)
    }
}

/// The points' indices along a 30-bit Morton curve of the atoms' bounding box (8 bohr of
/// margin; points beyond it clamped to its faces), ties by index: the tiled kernel's order.
fn morton_order(t: &PairTables, x: &[f64], y: &[f64], z: &[f64]) -> Vec<u32> {
    use rayon::prelude::*;
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in &t.xyz {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a] - 8.0);
            hi[a] = hi[a].max(p[a] + 8.0);
        }
    }
    let span = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max);
    let scale = 1024.0 / span;
    // 10 bits per axis, interleaved
    let spread = |v: f64| -> u64 {
        let mut q = (v * scale).clamp(0.0, 1023.0) as u64;
        q = (q | (q << 16)) & 0x0300_00ff;
        q = (q | (q << 8)) & 0x0300_f00f;
        q = (q | (q << 4)) & 0x030c_30c3;
        (q | (q << 2)) & 0x0924_9249
    };
    let mut keys: Vec<u64> = (0..x.len())
        .into_par_iter()
        .map(|k| {
            let c = spread(x[k] - lo[0]) << 2 | spread(y[k] - lo[1]) << 1 | spread(z[k] - lo[2]);
            c << 32 | k as u64
        })
        .collect();
    keys.par_sort_unstable();
    keys.into_par_iter().map(|key| key as u32).collect()
}

/// Ok when the evaluation's buffers stayed within its grant (the byte counts above are the
/// buffers'), else the broken count as an error of the device code.
fn within(grant: &crate::pool::Grant) -> Result<(), crate::GpuError> {
    match grant.violations().first() {
        None => Ok(()),
        Some(v) => Err(crate::GpuError::Contract(format!(
            "grid partition: {} bytes drawn with {} of the {} granted left",
            v.requested, v.remaining, v.granted
        ))),
    }
}

/// The kernels' scheme number and SSF's half-width.
fn scheme(t: &PairTables) -> (i32, f64) {
    match t.scheme {
        PairScheme::Becke => (0, 0.0),
        PairScheme::Lko => (1, 0.0),
        PairScheme::Ssf { half } => (2, half),
    }
}

/// The engine's weights, every device error the grid build's (a caller that falls back on some
/// wraps [`GpuPartition::weights`] in its own policy, as linus's GPU backend does).
impl DevicePartition for GpuPartition<'_> {
    fn weights(
        &self,
        t: &PairTables,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        owner: &[u32],
        vol: &[f64],
    ) -> Result<Option<Vec<f64>>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Some(GpuPartition::weights(self, t, x, y, z, owner, vol)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{GridSpec, MolecularGrid};
    use crate::harness;

    /// The device's weights are the host's bit for bit, screened (the default) and from every
    /// pair: PySCF's level-3 grid (Becke, Treutler's size adjustment).
    #[test]
    fn device_partition_is_the_hosts() {
        let Some(gpu) = crate::harness::gpu() else {
            return;
        };
        for name in ["la18", "u1an39"] {
            let mol = harness::structure(name).unwrap();
            let z = mol.z.clone();
            let spec = GridSpec::pyscf(3);
            let t = std::time::Instant::now();
            let host = MolecularGrid::build(&z, &mol.positions, &spec).unwrap();
            let th = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let dev = MolecularGrid::build_with(
                &z,
                &mol.positions,
                &spec,
                Some(&GpuPartition {
                    gpu: &gpu,
                    allowance: None,
                }),
            )
            .unwrap();
            let td = t.elapsed().as_secs_f64();
            eprintln!(
                "{name}: {} points, host {th:.3} s, device {td:.3} s",
                host.x.len()
            );
            let t = host.partitioner().pair_tables().unwrap();
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&host.w), bits(&dev.w), "{name}");
            assert_eq!(bits(&host.x), bits(&dev.x), "{name}");
            let part = GpuPartition {
                gpu: &gpu,
                allowance: None,
            };
            let g = &host;
            let (w, kept) = part
                .screened(&t, &g.x, &g.y, &g.z, &g.atom, &g.w_atomic)
                .unwrap()
                .unwrap();
            let all = part
                .every_pair(&t, &g.x, &g.y, &g.z, &g.atom, &g.w_atomic)
                .unwrap();
            eprintln!("{name}: {kept:.1} of {} atoms kept per point", t.xyz.len());
            assert_eq!(bits(&w), bits(&host.w), "{name} screened");
            assert_eq!(bits(&all), bits(&host.w), "{name} every pair");
            assert!(kept < t.xyz.len() as f64, "{name}: nothing screened");
        }
    }

    /// A grant above the allowance (linus: the memory plan's share for the partition) is the
    /// device's out-of-memory error before any work, which linus's grid builder turns into the
    /// host's weights with the reason; within it, the device's weights.
    #[test]
    fn a_grant_above_the_allowance_declines() {
        let Some(gpu) = crate::harness::gpu() else {
            return;
        };
        let mol = harness::structure("la18").unwrap();
        let g = MolecularGrid::build(&mol.z, &mol.positions, &GridSpec::pyscf(2)).unwrap();
        let t = g.partitioner().pair_tables().unwrap();
        let tight = GpuPartition {
            gpu: &gpu,
            allowance: Some(1 << 20),
        };
        match tight.weights(&t, &g.x, &g.y, &g.z, &g.atom, &g.w_atomic) {
            Err(e @ crate::GpuError::OutOfMemory { .. }) => assert!(e.falls_back_at_setup()),
            r => panic!("expected out of memory, got {:?}", r.map(|w| w.len())),
        }
        let roomy = GpuPartition {
            gpu: &gpu,
            allowance: Some(8 << 30),
        };
        let w = roomy
            .weights(&t, &g.x, &g.y, &g.z, &g.atom, &g.w_atomic)
            .unwrap();
        assert_eq!(w.len(), g.x.len());
    }

    /// The screened kernel's SSF and LKO factors (partitions the host screens exactly by
    /// itself; unscreened here), with and without the size adjustment: the host's
    /// `weights_block` bit for bit at la18's PySCF level-2 grid points.
    #[test]
    fn screened_kernel_repeats_ssf_and_lko() {
        use crate::cpu::{PartitionScheme, Partitioner, RadiiTable, SizeAdjust};
        let Some(gpu) = crate::harness::gpu() else {
            return;
        };
        let mol = harness::structure("la18").unwrap();
        let g = MolecularGrid::build(&mol.z, &mol.positions, &GridSpec::pyscf(2)).unwrap();
        let part = GpuPartition {
            gpu: &gpu,
            allowance: None,
        };
        for scheme in [PartitionScheme::SSF, PartitionScheme::LKO] {
            for adjust in [
                SizeAdjust::None,
                SizeAdjust::Treutler(RadiiTable::BraggPyscf),
            ] {
                let p =
                    Partitioner::new(&mol.z, &mol.positions, scheme, adjust).with_screening(false);
                let t = p.pair_tables().unwrap();
                let mut scr = p.scratch();
                let mut host = vec![0.0; g.x.len()];
                for k in 0..g.x.len() {
                    let r = k..k + 1;
                    p.weights_block(
                        &g.x[r.clone()],
                        &g.y[r.clone()],
                        &g.z[r.clone()],
                        g.atom[k] as usize,
                        &g.w_atomic[r.clone()],
                        &mut host[r],
                        &mut scr,
                    );
                }
                let (w, kept) = part
                    .screened(&t, &g.x, &g.y, &g.z, &g.atom, &g.w_atomic)
                    .unwrap()
                    .unwrap();
                eprintln!("{scheme:?} {adjust:?}: {kept:.1} atoms per point");
                let same = w.iter().zip(&host).all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "{scheme:?} {adjust:?}");
            }
        }
    }
}
