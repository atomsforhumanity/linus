//! `MolecularGrid::build_with` and its `DevicePartition` hook without a device: a partition that
//! declines leaves the weights to the host (the same grid as `build`, bit for bit), the weights
//! re-evaluated from `Partitioner::pair_tables` alone are the host's (the tables are exactly what
//! `weights_block` reads), and a device's failure or a wrong number of weights is the build's
//! error.

use linus_grid::*;
use std::sync::atomic::{AtomicUsize, Ordering};

const ANG: f64 = 1.0 / linus_core::units::BOHR_ANGSTROM_PYSCF;

/// Three waters (9 atoms), apart enough that far atoms' cell factors matter little.
fn waters() -> (Vec<u32>, Vec<[f64; 3]>) {
    let mut z = Vec::new();
    let mut xyz = Vec::new();
    for (k, o) in [[0.0, 0.0, 0.0], [2.9, 0.4, 0.1], [0.3, 3.0, -0.5]]
        .iter()
        .enumerate()
    {
        let t = 0.7 * k as f64;
        z.extend([8, 1, 1]);
        xyz.push([o[0] * ANG, o[1] * ANG, o[2] * ANG]);
        xyz.push([
            (o[0] + 0.76 * t.cos()) * ANG,
            (o[1] + 0.59) * ANG,
            o[2] * ANG,
        ]);
        xyz.push([
            (o[0] - 0.76) * ANG,
            (o[1] + 0.59 * t.sin()) * ANG,
            (o[2] + 0.2) * ANG,
        ]);
    }
    (z, xyz)
}

/// What a device partition does in each test.
enum Mode {
    Decline,
    Tables,
    Short,
    Fail,
}

struct Mock {
    mode: Mode,
    calls: AtomicUsize,
}

/// Becke's s(nu) (linus-grid's, operation for operation).
fn becke_s(nu: f64) -> f64 {
    let mut t = nu;
    t = (3.0 - t * t) * t * 0.5;
    t = (3.0 - t * t) * t * 0.5;
    ((3.0 - t * t) * t * 0.5) * 0.5
}

/// The weights of Becke's partition from the tables alone, in `weights_block`'s order: every
/// pair (i, j), i > j, its factors into P_i and P_j, then each point's sum in atom order.
fn from_tables(
    t: &PairTables,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    owner: &[u32],
    vol: &[f64],
) -> Vec<f64> {
    assert_eq!(t.scheme, PairScheme::Becke);
    let n = t.xyz.len();
    let adjust = !t.a.is_empty();
    (0..x.len())
        .map(|k| {
            let d: Vec<f64> = t
                .xyz
                .iter()
                .map(|c| {
                    let (dx, dy, dz) = (x[k] - c[0], y[k] - c[1], z[k] - c[2]);
                    (dx * dx + dy * dy + dz * dz).sqrt()
                })
                .collect();
            let mut p = vec![1.0; n];
            for i in 1..n {
                for j in 0..i {
                    let mut g = (d[i] - d[j]) * t.inv_r[i * n + j];
                    if adjust {
                        g += t.a[i * n + j] * (1.0 - g * g);
                    }
                    let s = becke_s(g);
                    p[i] *= 0.5 - s;
                    p[j] *= 0.5 + s;
                }
            }
            let mut zs = 0.0;
            for v in &p {
                zs += v;
            }
            vol[k] * p[owner[k] as usize] * (1.0 / zs)
        })
        .collect()
}

impl DevicePartition for Mock {
    fn weights(
        &self,
        t: &PairTables,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        owner: &[u32],
        vol: &[f64],
    ) -> Result<Option<Vec<f64>>, Box<dyn std::error::Error + Send + Sync>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        match self.mode {
            Mode::Decline => Ok(None),
            Mode::Tables => Ok(Some(from_tables(t, x, y, z, owner, vol))),
            Mode::Short => Ok(Some(vec![0.0; x.len() - 1])),
            Mode::Fail => Err("a device failure".into()),
        }
    }
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn device_partition_hook() {
    let (z, xyz) = waters();
    // PySCF's grids: Becke's partition with Treutler's size adjustment, unscreened
    let spec = GridSpec::pyscf(1);
    let host = MolecularGrid::build(&z, &xyz, &spec).unwrap();
    assert!(host.partitioner().pair_tables().is_some());
    for mode in [Mode::Decline, Mode::Tables] {
        let dev = Mock {
            mode,
            calls: AtomicUsize::new(0),
        };
        let (g, times) = MolecularGrid::build_timed(&z, &xyz, &spec, Some(&dev)).unwrap();
        assert_eq!(dev.calls.load(Ordering::Relaxed), 1);
        assert_eq!(bits(&g.w), bits(&host.w));
        assert_eq!(bits(&g.x), bits(&host.x));
        assert_eq!(g.batches.len(), host.batches.len());
        assert!(times.partition > 0.0);
    }
    for mode in [Mode::Short, Mode::Fail] {
        let dev = Mock {
            mode,
            calls: AtomicUsize::new(0),
        };
        let r = MolecularGrid::build_with(&z, &xyz, &spec, Some(&dev));
        assert!(matches!(r, Err(GridError::Device(_))));
    }
    // a screened partition (linus's grids) never asks the device
    let dev = Mock {
        mode: Mode::Fail,
        calls: AtomicUsize::new(0),
    };
    let lin = GridSpec::linus(1);
    let g = MolecularGrid::build_with(&z, &xyz, &lin, Some(&dev)).unwrap();
    assert_eq!(dev.calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        bits(&g.w),
        bits(&MolecularGrid::build(&z, &xyz, &lin).unwrap().w)
    );
}
