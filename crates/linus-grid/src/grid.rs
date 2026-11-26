//! Grid specification, atomic grids and molecular grid assembly.

use crate::batch::{bounding_sphere, make_batches, Batch, Batching};
use crate::elements::{period_index, row};
use crate::lebedev::{lebedev, lebedev_index, lebedev_npoints_for_degree};
use crate::orient::{rotation_matrix, Orientation};
use crate::partition::{PartitionScheme, Partitioner, Scratch, SizeAdjust};
use crate::prune::{angular_counts, orca_angular_max, Pruning};
use crate::radial::{radial_grid, RadialScheme};
use crate::GridError;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// PySCF `gen_grid.RAD_GRIDS[level][period]`.
pub const PYSCF_RAD_GRIDS: [[usize; 7]; 10] = [
    [10, 15, 20, 30, 35, 40, 50],
    [30, 40, 50, 60, 65, 70, 75],
    [40, 60, 65, 75, 80, 85, 90],
    [50, 75, 80, 90, 95, 100, 105],
    [60, 90, 95, 105, 110, 115, 120],
    [70, 105, 110, 120, 125, 130, 135],
    [80, 120, 125, 135, 140, 145, 150],
    [90, 135, 140, 150, 155, 160, 165],
    [100, 150, 155, 165, 170, 175, 180],
    [200, 200, 200, 200, 200, 200, 200],
];

/// PySCF `gen_grid.ANG_ORDER[level][period]` (Lebedev degrees).
pub const PYSCF_ANG_ORDER: [[usize; 7]; 10] = [
    [11, 15, 17, 17, 17, 17, 17],
    [17, 23, 23, 23, 23, 23, 23],
    [23, 29, 29, 29, 29, 29, 29],
    [29, 29, 35, 35, 35, 35, 35],
    [35, 41, 41, 41, 41, 41, 41],
    [41, 47, 47, 47, 47, 47, 47],
    [47, 53, 53, 53, 53, 53, 53],
    [53, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59],
    [65, 65, 65, 65, 65, 65, 65],
];

/// ORCA 6 optimised XC IntAcc per AngularGrid 1..=7 (manual, Table 7.2).
pub const ORCA_XC_INTACC: [f64; 7] = [4.004, 4.004, 4.159, 4.388, 4.629, 4.959, 4.959];

/// Rule giving (number of radial shells, nominal/max Lebedev points) for each element.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AtomSizes {
    /// PySCF `Grids.level` defaults (`RAD_GRIDS`, `ANG_ORDER` by period).
    PyscfLevel(u8),
    /// The same for every element.
    Uniform { n_rad: usize, n_ang: usize },
    /// ORCA-like: n_r = round(15 IntAcc - 40 + b ROW) and the largest Lebedev set of
    /// `angular_grid` (1..=7); H/He use angular_grid - 1 if `h_reduced` (ORCA's HGridReduced).
    OrcaLike {
        angular_grid: u8,
        int_acc: f64,
        b: f64,
        h_reduced: bool,
    },
    /// Per-row table: entry k is (n_rad, n_ang) for period k+1 (last entry reused beyond).
    PerRow(Vec<(usize, usize)>),
    /// ORCA 6.0's DefGrid `level` (1..=3: AngularGrid 3/4/6 at IntAcc 4.159/4.388/4.959): the
    /// radial shells ORCA uses for the element ([`crate::orca`]'s tables; unmeasured elements
    /// take their row's count) and the largest Lebedev set of the AngularGrid, one AngularGrid
    /// lower for H/He (HGridReduced).
    OrcaDefGrid { level: u8 },
    /// ORCA 6.0's NL (VV10) grid of DefGrid `defgrid` (1 or 2), approximated: the largest
    /// Lebedev set of AngularGrid `defgrid` for every element and the shell counts of
    /// [`crate::orca`]'s tables; see [`GridSpec::orca_nl`].
    OrcaNl { defgrid: u8 },
}

impl AtomSizes {
    pub fn sizes(&self, z: u32) -> Result<(usize, usize), GridError> {
        match self {
            AtomSizes::PyscfLevel(level) => {
                let l = *level as usize;
                if l > 9 {
                    return Err(GridError::InvalidSpec(format!("PySCF level {l} > 9")));
                }
                let p = period_index(z).min(6);
                let n_ang = lebedev_npoints_for_degree(PYSCF_ANG_ORDER[l][p]).expect("table");
                Ok((PYSCF_RAD_GRIDS[l][p], n_ang))
            }
            AtomSizes::Uniform { n_rad, n_ang } => Ok((*n_rad, *n_ang)),
            AtomSizes::OrcaLike {
                angular_grid,
                int_acc,
                b,
                h_reduced,
            } => {
                let mut g = *angular_grid as usize;
                if !(1..=7).contains(&g) {
                    return Err(GridError::InvalidSpec(format!(
                        "AngularGrid {g} not in 1..=7"
                    )));
                }
                if *h_reduced && z <= 2 && g > 1 {
                    g -= 1;
                }
                let nr = (15.0 * int_acc - 40.0 + b * row(z) as f64).round().max(1.0) as usize;
                Ok((nr, orca_angular_max(g)))
            }
            AtomSizes::OrcaDefGrid { level } => {
                let l = (*level).clamp(1, 3);
                let mut g = [3usize, 4, 6][l as usize - 1];
                if z <= 2 {
                    g -= 1;
                }
                let t = crate::orca::grids();
                Ok((t.defgrid_nr_or_row(z, l), orca_angular_max(g)))
            }
            AtomSizes::OrcaNl { defgrid } => {
                let d = (*defgrid).clamp(1, 2);
                let t = crate::orca::grids();
                Ok((t.nl_shells(z, d), orca_angular_max(d as usize)))
            }
            AtomSizes::PerRow(t) => {
                if t.is_empty() {
                    return Err(GridError::InvalidSpec("empty PerRow table".into()));
                }
                let p = period_index(z).min(t.len() - 1);
                Ok(t[p])
            }
        }
    }
}

/// Complete description of a molecular grid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GridSpec {
    pub radial: RadialScheme,
    pub sizes: AtomSizes,
    /// Element-specific overrides: (Z, n_rad, n_ang).
    pub overrides: Vec<(u32, usize, usize)>,
    pub pruning: Pruning,
    pub partition: PartitionScheme,
    pub size_adjust: SizeAdjust,
    /// Drop points with |w| <= cutoff after partitioning (None keeps all points).
    pub weight_cutoff: Option<f64>,
    /// Drop points whose partition weight w / w_atomic is <= cutoff (ORCA's `WeightCut`; an
    /// isolated atom keeps all its points). None (the default) keeps all points.
    #[serde(default)]
    pub partition_cutoff: Option<f64>,
    pub orientation: Orientation,
    pub batching: Batching,
}

/// Unpartitioned grid of one element: radial nodes/weights (weights include 4 pi r^2) and the
/// Lebedev point count of every shell. This is all a GPU backend (e.g. cuEST) needs, together
/// with the Lebedev tables and the partitioning scheme.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AtomicGrid {
    pub z: u32,
    pub r: Vec<f64>,
    pub w: Vec<f64>,
    pub n_ang: Vec<usize>,
}

impl AtomicGrid {
    pub fn npoints(&self) -> usize {
        self.n_ang.iter().sum()
    }
}

/// Builds the atomic grid of element `z` for a spec.
pub fn atomic_grid(spec: &GridSpec, z: u32) -> Result<AtomicGrid, GridError> {
    let (n_rad, n_ang) = match spec.overrides.iter().find(|o| o.0 == z) {
        Some(&(_, nr, na)) => (nr, na),
        None => spec.sizes.sizes(z)?,
    };
    lebedev_index(n_ang).ok_or(GridError::UnsupportedAngular(n_ang))?;
    let rg = radial_grid(&spec.radial, n_rad, z);
    let n_ang = angular_counts(&spec.pruning, z, &rg.r, n_ang)?;
    for &n in &n_ang {
        lebedev_index(n).ok_or(GridError::UnsupportedAngular(n))?;
    }
    Ok(AtomicGrid {
        z,
        r: rg.r,
        w: rg.w,
        n_ang,
    })
}

/// A molecular integration grid in structure-of-arrays layout, ordered by batch.
#[derive(Clone, Debug)]
pub struct MolecularGrid {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    /// Final (partitioned) quadrature weights.
    pub w: Vec<f64>,
    /// Unpartitioned atomic weights (radial x angular), needed for weight derivatives.
    pub w_atomic: Vec<f64>,
    /// Owning atom of each point.
    pub atom: Vec<u32>,
    /// Contiguous point ranges with bounding spheres.
    pub batches: Vec<Batch>,
    /// Distinct atomic grids; `atom_grid[i]` indexes this for atom i.
    pub atomic_grids: Vec<AtomicGrid>,
    pub atom_grid: Vec<usize>,
    /// Rotation applied to the Lebedev unit vectors (u' = M u).
    pub rotation: [[f64; 3]; 3],
    /// Number of points before the weight cutoff.
    pub npoints_before_cutoff: usize,
    pub atom_z: Vec<u32>,
    pub atom_xyz: Vec<[f64; 3]>,
    partitioner: Partitioner,
}

/// Points per partitioning block.
const BLOCK: usize = 128;

struct ShellOut<'a> {
    atom: usize,
    r: f64,
    w: f64,
    n: usize,
    x: &'a mut [f64],
    y: &'a mut [f64],
    z: &'a mut [f64],
    wa: &'a mut [f64],
}

/// A partition evaluated elsewhere (a device): the weights of an unscreened partition at every
/// point, as [`Partitioner::weights_block`] computes them (a device may leave out of a point's
/// sum of cell functions terms below 2^-100 of it).
pub trait DevicePartition: Sync {
    /// `vol[k] P_owner(k) / sum_B P_B` at the points (`x`, `y`, `z`, owners `owner`) for the
    /// partition `tables`, one per point; None: the device declines (the implementor records
    /// why) and the host computes them; Err: a failure of the calculation.
    fn weights(
        &self,
        tables: &crate::partition::PairTables,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        owner: &[u32],
        vol: &[f64],
    ) -> Result<Option<Vec<f64>>, Box<dyn std::error::Error + Send + Sync>>;
}

/// The partition weights `w` of the points (`x`, `y`, `z`, atomic weights `wa`) of the atoms'
/// consecutive runs `groups` (start, end; the owner is the run's index) on the host, in
/// parallel over blocks of up to [`BLOCK`] points of one run: the same bits for any blocking
/// (`weights_block` computes each point on its own).
#[allow(
    clippy::too_many_arguments,
    reason = "the points' coordinate and weight arrays"
)]
fn host_weights(
    partitioner: &Partitioner,
    groups: &[(usize, usize)],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    wa: &[f64],
    w: &mut [f64],
) {
    let mut blocks: Vec<(usize, usize, &mut [f64])> = Vec::new();
    let mut rest = w;
    let mut at = 0;
    for (atom, &(s, e)) in groups.iter().enumerate() {
        for k0 in (s..e).step_by(BLOCK) {
            let k1 = (k0 + BLOCK).min(e);
            let (skip, tail) = rest.split_at_mut(k0 - at);
            debug_assert!(skip.is_empty());
            let (out, tail) = tail.split_at_mut(k1 - k0);
            blocks.push((atom, k0, out));
            rest = tail;
            at = k1;
        }
    }
    blocks.into_par_iter().for_each_init(
        || partitioner.scratch(),
        |scr: &mut Scratch, (atom, k0, out)| {
            let k1 = k0 + out.len();
            partitioner.weights_block(
                &x[k0..k1],
                &y[k0..k1],
                &z[k0..k1],
                atom,
                &wa[k0..k1],
                out,
                scr,
            );
        },
    );
}

/// Wall seconds of a grid build's phases ([`MolecularGrid::build_timed`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GridTimes {
    /// The atomic grids and the points.
    pub points: f64,
    /// The partition weights (the device's, or the host's).
    pub partition: f64,
    /// The weight cutoff, the batching (the points' sort into batches) and the batches'
    /// bounding spheres.
    pub batches: f64,
}

impl MolecularGrid {
    /// Builds the grid for atoms with nuclear charges `z` at `xyz` (bohr).
    pub fn build(z: &[u32], xyz: &[[f64; 3]], spec: &GridSpec) -> Result<MolecularGrid, GridError> {
        Self::build_with(z, xyz, spec, None)
    }

    /// [`MolecularGrid::build`] with the partition weights from `device` where it gives them (an
    /// unscreened partition: Becke's, PySCF's grids), else from the host: the same grid when the
    /// device repeats the host's arithmetic (but for the terms it may leave out).
    pub fn build_with(
        z: &[u32],
        xyz: &[[f64; 3]],
        spec: &GridSpec,
        device: Option<&dyn DevicePartition>,
    ) -> Result<MolecularGrid, GridError> {
        Self::build_timed(z, xyz, spec, device).map(|(g, _)| g)
    }

    /// [`MolecularGrid::build_with`] and the wall time of its phases.
    pub fn build_timed(
        z: &[u32],
        xyz: &[[f64; 3]],
        spec: &GridSpec,
        device: Option<&dyn DevicePartition>,
    ) -> Result<(MolecularGrid, GridTimes), GridError> {
        let clock = std::time::Instant::now();
        let mut times = GridTimes::default();
        if z.len() != xyz.len() {
            return Err(GridError::InvalidSpec("z and xyz lengths differ".into()));
        }
        let natm = z.len();
        // distinct atomic grids by element
        let mut atomic_grids: Vec<AtomicGrid> = Vec::new();
        let mut atom_grid = Vec::with_capacity(natm);
        for &zi in z {
            match atomic_grids.iter().position(|g| g.z == zi) {
                Some(i) => atom_grid.push(i),
                None => {
                    atomic_grids.push(atomic_grid(spec, zi)?);
                    atom_grid.push(atomic_grids.len() - 1);
                }
            }
        }
        let rot = rotation_matrix(&spec.orientation, z, xyz);
        let identity = rot == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let partitioner = Partitioner::new(z, xyz, spec.partition, spec.size_adjust);
        let tables = device.and(partitioner.pair_tables());

        let npts: usize = atom_grid.iter().map(|&g| atomic_grids[g].npoints()).sum();
        let mut x = vec![0.0; npts];
        let mut y = vec![0.0; npts];
        let mut zz = vec![0.0; npts];
        let mut wa = vec![0.0; npts];
        let mut owner = vec![0u32; npts];
        let mut groups = Vec::with_capacity(natm);
        {
            // split output arrays into per-shell chunks
            let mut shells: Vec<ShellOut> = Vec::new();
            let (mut rx, mut ry, mut rz, mut rwa) =
                (&mut x[..], &mut y[..], &mut zz[..], &mut wa[..]);
            let mut pos = 0;
            for (ia, &gi) in atom_grid.iter().enumerate() {
                let g = &atomic_grids[gi];
                let start = pos;
                for (k, &n) in g.n_ang.iter().enumerate() {
                    let (a, b) = rx.split_at_mut(n);
                    rx = b;
                    let (c, d) = ry.split_at_mut(n);
                    ry = d;
                    let (e, f) = rz.split_at_mut(n);
                    rz = f;
                    let (j, l) = rwa.split_at_mut(n);
                    rwa = l;
                    shells.push(ShellOut {
                        atom: ia,
                        r: g.r[k],
                        w: g.w[k],
                        n,
                        x: a,
                        y: c,
                        z: e,
                        wa: j,
                    });
                    pos += n;
                }
                owner[start..pos].fill(ia as u32);
                groups.push((start, pos));
            }
            shells.par_iter_mut().for_each(|sh| {
                let leb = lebedev(sh.n).expect("validated");
                let c = xyz[sh.atom];
                for j in 0..sh.n {
                    let (ux, uy, uz) = if identity {
                        (leb.x[j], leb.y[j], leb.z[j])
                    } else {
                        let u = [leb.x[j], leb.y[j], leb.z[j]];
                        (
                            rot[0][0] * u[0] + rot[0][1] * u[1] + rot[0][2] * u[2],
                            rot[1][0] * u[0] + rot[1][1] * u[1] + rot[1][2] * u[2],
                            rot[2][0] * u[0] + rot[2][1] * u[1] + rot[2][2] * u[2],
                        )
                    };
                    // PySCF: coords = r * u + R_A ; vol = w_rad * w_ang
                    sh.x[j] = sh.r * ux + c[0];
                    sh.y[j] = sh.r * uy + c[1];
                    sh.z[j] = sh.r * uz + c[2];
                    sh.wa[j] = sh.w * leb.w[j];
                }
            });
        }
        times.points = clock.elapsed().as_secs_f64();
        // the partition weights: the device's where it gives them, else the host's
        let device_w = match (device, tables.as_ref()) {
            (Some(dev), Some(t)) => dev
                .weights(t, &x, &y, &zz, &owner, &wa)
                .map_err(GridError::Device)?,
            _ => None,
        };
        let mut w = match device_w {
            Some(v) if v.len() == npts => v,
            Some(v) => {
                return Err(GridError::Device(
                    format!(
                        "the device partition gave {} weights for {npts} points",
                        v.len()
                    )
                    .into(),
                ))
            }
            None => {
                let mut w = vec![0.0; npts];
                host_weights(&partitioner, &groups, &x, &y, &zz, &wa, &mut w);
                w
            }
        };
        times.partition = clock.elapsed().as_secs_f64() - times.points;
        let npoints_before_cutoff = npts;
        // weight cutoffs
        if spec.weight_cutoff.is_some() || spec.partition_cutoff.is_some() {
            let keep: Vec<bool> = w
                .iter()
                .zip(&wa)
                .map(|(&v, &a)| {
                    spec.weight_cutoff.is_none_or(|c| v.abs() > c)
                        && spec.partition_cutoff.is_none_or(|c| v > c * a)
                })
                .collect();
            let mut new_groups = Vec::with_capacity(natm);
            let mut k = 0;
            for &(s, e) in &groups {
                let s2 = k;
                for i in s..e {
                    if keep[i] {
                        x[k] = x[i];
                        y[k] = y[i];
                        zz[k] = zz[i];
                        w[k] = w[i];
                        wa[k] = wa[i];
                        owner[k] = owner[i];
                        k += 1;
                    }
                }
                new_groups.push((s2, k));
            }
            for v in [&mut x, &mut y, &mut zz, &mut w, &mut wa] {
                v.truncate(k);
            }
            owner.truncate(k);
            groups = new_groups;
        }
        // batching
        let n = x.len();
        let bgroups = if spec.batching.per_atom {
            groups.clone()
        } else {
            vec![(0, n)]
        };
        let (perm, lens) = make_batches(&x, &y, &zz, &bgroups, spec.batching.max_points);
        let apply =
            |v: &Vec<f64>| -> Vec<f64> { perm.par_iter().map(|&i| v[i as usize]).collect() };
        let x2 = apply(&x);
        let y2 = apply(&y);
        let z2 = apply(&zz);
        let w2 = apply(&w);
        let wa2 = apply(&wa);
        let o2: Vec<u32> = perm.par_iter().map(|&i| owner[i as usize]).collect();
        let starts: Vec<usize> = lens
            .iter()
            .scan(0, |s, &l| {
                *s += l;
                Some(*s - l)
            })
            .collect();
        let batches: Vec<Batch> = starts
            .par_iter()
            .zip(lens.par_iter())
            .map(|(&s, &l)| {
                let (c, r) = bounding_sphere(&x2, &y2, &z2, s, s + l);
                Batch {
                    start: s,
                    end: s + l,
                    center: c,
                    radius: r,
                }
            })
            .collect();
        times.batches = clock.elapsed().as_secs_f64() - times.points - times.partition;
        let grid = MolecularGrid {
            x: x2,
            y: y2,
            z: z2,
            w: w2,
            w_atomic: wa2,
            atom: o2,
            batches,
            atomic_grids,
            atom_grid,
            rotation: rot,
            npoints_before_cutoff,
            atom_z: z.to_vec(),
            atom_xyz: xyz.to_vec(),
            partitioner,
        };
        Ok((grid, times))
    }

    pub fn len(&self) -> usize {
        self.w.len()
    }

    pub fn is_empty(&self) -> bool {
        self.w.is_empty()
    }

    /// The axis-aligned bounding box (lo, hi) of the points `points` ([`crate::batch::bbox`]).
    pub fn bbox(&self, points: std::ops::Range<usize>) -> ([f64; 3], [f64; 3]) {
        crate::batch::bbox(
            &self.x[points.clone()],
            &self.y[points.clone()],
            &self.z[points],
        )
    }

    /// The partitioner used to build the grid.
    pub fn partitioner(&self) -> &Partitioner {
        &self.partitioner
    }

    /// Derivatives of point `i`'s weight with respect to the nuclear positions, the point moving
    /// with its owning atom: `out` receives (atom, dw_i/dR_atom). Returns w_i (recomputed).
    pub fn point_weight_gradient(
        &self,
        i: usize,
        scratch: &mut Scratch,
        out: &mut Vec<(usize, [f64; 3])>,
    ) -> f64 {
        self.partitioner.weight_gradient(
            [self.x[i], self.y[i], self.z[i]],
            self.atom[i] as usize,
            self.w_atomic[i],
            scratch,
            out,
        )
    }

    /// Grid-weight contribution to a nuclear gradient: G_D = sum_i f_i dw_i/dR_D, for per-point
    /// values `f` (e.g. the XC energy density e_xc(r_i)). Returns one 3-vector per atom.
    /// Unscreened partitions (Becke) use the vectorized block derivative; the others go point by
    /// point. Per-batch sums are added in batch order, so the result does not depend on the
    /// number of threads.
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    pub fn weight_gradient_contract(&self, f: &[f64]) -> Vec<[f64; 3]> {
        assert_eq!(f.len(), self.len());
        let natm = self.atom_z.len();
        let part = &self.partitioner;
        let block = part.has_block_gradient();
        let parts: Vec<Vec<[f64; 3]>> = self
            .batches
            .par_iter()
            .map_init(
                || (part.scratch(), Vec::new()),
                |(scr, out), b| {
                    let mut g = vec![[0.0; 3]; natm];
                    if block {
                        // sub-blocks keep the O(N m) scratch small
                        const SUB: usize = 64;
                        let mut s0 = b.start;
                        while s0 < b.end {
                            let s1 = (s0 + SUB).min(b.end);
                            if f[s0..s1].iter().any(|&v| v != 0.0) {
                                part.weight_gradient_block(
                                    &self.x[s0..s1],
                                    &self.y[s0..s1],
                                    &self.z[s0..s1],
                                    &self.atom[s0..s1],
                                    &self.w_atomic[s0..s1],
                                    &f[s0..s1],
                                    &mut g,
                                    scr,
                                );
                            }
                            s0 = s1;
                        }
                    } else {
                        for i in b.start..b.end {
                            if f[i] == 0.0 {
                                continue;
                            }
                            self.point_weight_gradient(i, scr, out);
                            for &(a, d) in out.iter() {
                                for k in 0..3 {
                                    g[a][k] += f[i] * d[k];
                                }
                            }
                        }
                    }
                    g
                },
            )
            .collect();
        let mut tot = vec![[0.0; 3]; natm];
        for g in &parts {
            for (t, v) in tot.iter_mut().zip(g) {
                for k in 0..3 {
                    t[k] += v[k];
                }
            }
        }
        tot
    }

    /// [`Self::weight_gradient_contract`] point by point through [`Self::point_weight_gradient`]
    /// (reference implementation; the reduction order depends on the thread schedule).
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    pub fn weight_gradient_contract_pointwise(&self, f: &[f64]) -> Vec<[f64; 3]> {
        assert_eq!(f.len(), self.len());
        let natm = self.atom_z.len();
        self.batches
            .par_iter()
            .fold(
                || (vec![[0.0; 3]; natm], self.partitioner.scratch(), Vec::new()),
                |(mut g, mut scr, mut out), b| {
                    for i in b.start..b.end {
                        if f[i] == 0.0 {
                            continue;
                        }
                        self.point_weight_gradient(i, &mut scr, &mut out);
                        for &(a, d) in &out {
                            for k in 0..3 {
                                g[a][k] += f[i] * d[k];
                            }
                        }
                    }
                    (g, scr, out)
                },
            )
            .map(|(g, _, _)| g)
            .reduce(
                || vec![[0.0; 3]; natm],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        for k in 0..3 {
                            x[k] += y[k];
                        }
                    }
                    a
                },
            )
    }

    /// Sum of weights times f(x, y, z) (Kahan-compensated, sequential).
    pub fn integrate(&self, f: impl Fn(f64, f64, f64) -> f64 + Sync) -> f64 {
        let parts: Vec<f64> = self
            .batches
            .par_iter()
            .map(|b| {
                let mut s = 0.0;
                let mut c = 0.0;
                for i in b.start..b.end {
                    let v = self.w[i] * f(self.x[i], self.y[i], self.z[i]) - c;
                    let t = s + v;
                    c = (t - s) - v;
                    s = t;
                }
                s
            })
            .collect();
        let mut s = 0.0;
        let mut c = 0.0;
        for v in parts {
            let v = v - c;
            let t = s + v;
            c = (t - s) - v;
            s = t;
        }
        s
    }
}
