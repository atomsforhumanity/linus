//! Spatial batching of grid points (k-d tree median bisection) with bounding spheres.

use serde::{Deserialize, Serialize};

/// Batching options.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Batching {
    /// Maximum points per batch; bisection stops once a node has <= max_points, so batches
    /// hold between max_points/2 and max_points points (except tiny atoms/molecules).
    pub max_points: usize,
    /// Batch each atom's points separately (every batch has a single owning atom).
    pub per_atom: bool,
}

impl Default for Batching {
    fn default() -> Self {
        Batching {
            max_points: 256,
            per_atom: false,
        }
    }
}

/// A contiguous range of grid points with a bounding sphere.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Batch {
    pub start: usize,
    pub end: usize,
    /// Centre of the bounding sphere (centre of the axis-aligned bounding box).
    pub center: [f64; 3],
    /// Radius: max distance of a batch point from `center`.
    pub radius: f64,
}

impl Batch {
    pub fn len(&self) -> usize {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }

    /// The axis-aligned bounding box (lo, hi) of the batch's points in `grid`.
    pub fn bbox(&self, grid: &crate::MolecularGrid) -> ([f64; 3], [f64; 3]) {
        grid.bbox(self.start..self.end)
    }
}

/// The axis-aligned bounding box (lo, hi) of the points (x\[i\], y\[i\], z\[i\]); (+inf, -inf)
/// for no points.
pub fn bbox(x: &[f64], y: &[f64], z: &[f64]) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for ((&px, &py), &pz) in x.iter().zip(y).zip(z) {
        for (k, c) in [px, py, pz].into_iter().enumerate() {
            lo[k] = lo[k].min(c);
            hi[k] = hi[k].max(c);
        }
    }
    (lo, hi)
}

/// Points above which [`bbox_of`] runs in parallel, in chunks of [`BBOX_CHUNK`] (a box of fewer
/// points is quicker on one thread than the pool's hand-off).
const BBOX_PARALLEL: usize = 1 << 16;
const BBOX_CHUNK: usize = 1 << 14;

/// [`bbox`] of the points `idx`.
fn bbox_of(idx: &[u32], x: &[f64], y: &[f64], z: &[f64]) -> ([f64; 3], [f64; 3]) {
    // large index sets in parallel (min and max do not depend on the order)
    if idx.len() > BBOX_PARALLEL {
        use rayon::prelude::*;
        return idx
            .par_chunks(BBOX_CHUNK)
            .map(|c| bbox_of(c, x, y, z))
            .reduce(
                || ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]),
                |(l1, h1), (l2, h2)| {
                    (
                        [l1[0].min(l2[0]), l1[1].min(l2[1]), l1[2].min(l2[2])],
                        [h1[0].max(h2[0]), h1[1].max(h2[1]), h1[2].max(h2[2])],
                    )
                },
            );
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &i in idx {
        let i = i as usize;
        let p = [x[i], y[i], z[i]];
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

/// Recursively bisects `idx` in place; returns leaf lengths in order.
fn split(idx: &mut [u32], x: &[f64], y: &[f64], z: &[f64], max_points: usize) -> Vec<usize> {
    if idx.len() <= max_points {
        return vec![idx.len()];
    }
    let (lo, hi) = bbox_of(idx, x, y, z);
    let mut axis = 0;
    for k in 1..3 {
        if hi[k] - lo[k] > hi[axis] - lo[axis] {
            axis = k;
        }
    }
    let c = match axis {
        0 => x,
        1 => y,
        _ => z,
    };
    let mid = idx.len() / 2;
    idx.select_nth_unstable_by(mid, |&a, &b| {
        c[a as usize]
            .partial_cmp(&c[b as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let (l, r) = idx.split_at_mut(mid);
    let (mut a, b) = if l.len() > 4096 {
        rayon::join(
            || split(l, x, y, z, max_points),
            || split(r, x, y, z, max_points),
        )
    } else {
        (split(l, x, y, z, max_points), split(r, x, y, z, max_points))
    };
    a.extend(b);
    a
}

/// Computes a batch permutation. `groups` are contiguous ranges (e.g. per atom) that are
/// batched independently. Returns (permutation: new position -> old index, leaf lengths).
pub(crate) fn make_batches(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    groups: &[(usize, usize)],
    max_points: usize,
) -> (Vec<u32>, Vec<usize>) {
    let n = x.len();
    let mut perm: Vec<u32> = (0..n as u32).collect();
    let max_points = max_points.max(1);
    let mut lens = Vec::new();
    // split perm into group slices
    let mut rest: &mut [u32] = &mut perm;
    let mut slices = Vec::with_capacity(groups.len());
    let mut pos = 0;
    for &(s, e) in groups {
        assert_eq!(s, pos);
        let (a, b) = rest.split_at_mut(e - s);
        slices.push(a);
        rest = b;
        pos = e;
    }
    use rayon::prelude::*;
    let leaf_lists: Vec<Vec<usize>> = slices
        .into_par_iter()
        .map(|sl| {
            if sl.is_empty() {
                vec![]
            } else {
                split(sl, x, y, z, max_points)
            }
        })
        .collect();
    for l in leaf_lists {
        lens.extend(l);
    }
    (perm, lens)
}

impl crate::MolecularGrid {
    /// The grid's batches cut into consecutive pieces of at most `max_points` points, each with
    /// its bounding sphere (the blocks of seminumerical and grid-based exchange's integral
    /// passes).
    pub fn sub_batches(&self, max_points: usize) -> Vec<Batch> {
        let max = max_points.max(1);
        let mut out = Vec::new();
        for b in &self.batches {
            let mut s = b.start;
            while s < b.end {
                let e = (s + max).min(b.end);
                let (center, radius) = bounding_sphere(&self.x, &self.y, &self.z, s, e);
                out.push(Batch {
                    start: s,
                    end: e,
                    center,
                    radius,
                });
                s = e;
            }
        }
        out
    }
}

/// Bounding sphere (bbox centre, max distance) of points [s, e).
pub(crate) fn bounding_sphere(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    s: usize,
    e: usize,
) -> ([f64; 3], f64) {
    let (lo, hi) = bbox(&x[s..e], &y[s..e], &z[s..e]);
    let c = [
        0.5 * (lo[0] + hi[0]),
        0.5 * (lo[1] + hi[1]),
        0.5 * (lo[2] + hi[2]),
    ];
    let mut r2: f64 = 0.0;
    for i in s..e {
        let d = [x[i] - c[0], y[i] - c[1], z[i] - c[2]];
        r2 = r2.max(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    }
    (c, r2.sqrt())
}
