//! Point groups and rotational symmetry numbers of molecules.
//!
//! The rotational symmetry number σ is the number of proper rotations (identity included) that
//! map the molecule onto itself, exchanging only equivalent atoms (same element and mass). It
//! is found by testing candidate axes (principal axes of inertia, directions to atoms, to
//! midpoints of equivalent-atom pairs, normals of planes through equivalent atoms) for C_n
//! rotations; every symmetry operation is recorded as the permutation of atoms it induces, and
//! σ is the order of the group these permutations generate (so an axis found twice, or one
//! missed but implied by others, cannot miscount it). The Schoenflies symbol comes from the
//! rotation axes plus inversion, mirror planes and S_2n axes, by the usual flow chart.
//!
//! Tolerance: an operation is accepted if it moves every atom to within `tol` (bohr) of an
//! equivalent atom. Optimized geometries keep their symmetry to ~1e-4 bohr or better; the
//! default 0.01 bohr also accepts slightly distorted ones.

pub mod abelian;
pub mod ao;
pub mod v3;

use std::collections::HashSet;
use v3::*;

/// Default tolerance (bohr) for symmetry operations.
pub const DEFAULT_TOLERANCE: f64 = 0.01;

/// A detected point group.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PointGroup {
    /// Schoenflies symbol: "C1", "Cs", "Ci", "C2v", "D3h", "Td", "Oh", "Ih", ..., "C*v" and "D*h"
    /// for linear molecules (C∞v, D∞h), "Kh" for an atom.
    pub name: String,
    /// Rotational symmetry number σ.
    pub symmetry_number: u32,
    /// Largest distance (bohr) between an atom moved by one of the found operations and the
    /// equivalent atom it lands on.
    pub max_deviation: f64,
    /// Tolerance used (bohr).
    pub tolerance: f64,
}

/// Atoms centred on the centre of mass, with equivalence classes.
#[derive(Clone, Debug)]
struct Frame {
    pos: Vec<V3>,
    kind: Vec<usize>,
    tol: f64,
}

impl Frame {
    /// The permutation an operation induces (atom a goes to `perm[a]`) and the largest
    /// displacement, or None if it is not a symmetry operation.
    #[allow(clippy::needless_range_loop, reason = "indexes several arrays in step")]
    fn perm(&self, op: &M3) -> Option<(Vec<usize>, f64)> {
        let n = self.pos.len();
        let mut perm = vec![usize::MAX; n];
        let mut used = vec![false; n];
        let mut dev: f64 = 0.0;
        for a in 0..n {
            let p = mv(op, self.pos[a]);
            let mut best = (f64::INFINITY, usize::MAX);
            for b in 0..n {
                if self.kind[b] == self.kind[a] {
                    let d = norm(sub(p, self.pos[b]));
                    if d < best.0 {
                        best = (d, b);
                    }
                }
            }
            if best.0 > self.tol || used[best.1] {
                return None;
            }
            used[best.1] = true;
            perm[a] = best.1;
            dev = dev.max(best.0);
        }
        // the operation was built from a candidate axis estimated from (noisy) positions; the
        // best orthogonal map with the same permutation and handedness says how far the
        // structure itself is from symmetric
        let det = op[0][0] * (op[1][1] * op[2][2] - op[1][2] * op[2][1])
            - op[0][1] * (op[1][0] * op[2][2] - op[1][2] * op[2][0])
            + op[0][2] * (op[1][0] * op[2][1] - op[1][1] * op[2][0]);
        let refined = self.best_fit_deviation(&perm, det > 0.0);
        Some((perm, dev.min(refined)))
    }

    /// Largest |R x_a - x_perm(a)| for the orthogonal R (det +1 if `proper`, else -1) that
    /// minimizes the summed squares (Kabsch).
    fn best_fit_deviation(&self, perm: &[usize], proper: bool) -> f64 {
        let mut a = faer::Mat::<f64>::zeros(3, 3);
        for (i, &j) in perm.iter().enumerate() {
            let (x, y) = (self.pos[i], self.pos[j]);
            for r in 0..3 {
                for c in 0..3 {
                    a[(r, c)] += y[r] * x[c];
                }
            }
        }
        let Some((u, _, v)) = crate::linalg::svd(a.as_ref(), faer::Par::Seq) else {
            return f64::INFINITY;
        };
        let uv = crate::linalg::mm(u.as_ref(), v.transpose(), faer::Par::Seq);
        let det = uv[(0, 0)] * (uv[(1, 1)] * uv[(2, 2)] - uv[(1, 2)] * uv[(2, 1)])
            - uv[(0, 1)] * (uv[(1, 0)] * uv[(2, 2)] - uv[(1, 2)] * uv[(2, 0)])
            + uv[(0, 2)] * (uv[(1, 0)] * uv[(2, 1)] - uv[(1, 1)] * uv[(2, 0)]);
        let want = if proper { 1.0 } else { -1.0 };
        let d = if det * want > 0.0 { 1.0 } else { -1.0 };
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = (0..3)
                    .map(|k| u[(i, k)] * if k == 2 { d } else { 1.0 } * v[(j, k)])
                    .sum();
            }
        }
        perm.iter()
            .enumerate()
            .map(|(i, &j)| norm(sub(mv(&r, self.pos[i]), self.pos[j])))
            .fold(0.0, f64::max)
    }
}

/// Rotation by `theta` about the unit vector `u` (Rodrigues).
fn rotation(u: V3, theta: f64) -> M3 {
    let (s, c) = theta.sin_cos();
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (1.0 - c) * u[i] * u[j] + if i == j { c } else { 0.0 };
        }
    }
    m[0][1] -= s * u[2];
    m[0][2] += s * u[1];
    m[1][0] += s * u[2];
    m[1][2] -= s * u[0];
    m[2][0] -= s * u[1];
    m[2][1] += s * u[0];
    m
}

/// Reflection through the plane with unit normal `u`.
fn reflection(u: V3) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = if i == j { 1.0 } else { 0.0 } - 2.0 * u[i] * u[j];
        }
    }
    m
}

fn matmul3(a: &M3, b: &M3) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}

/// Unit vectors, deduplicated up to sign.
#[derive(Default)]
struct Directions(Vec<V3>);

impl Directions {
    fn push(&mut self, v: V3, min_norm: f64) {
        let n = norm(v);
        if n <= min_norm {
            return;
        }
        let u = scale(v, 1.0 / n);
        if self.0.iter().all(|w| dot(*w, u).abs() < 1.0 - 1e-8) {
            self.0.push(u);
        }
    }
}

/// Close a set of permutations under composition (the group they generate).
fn closure(gens: &[Vec<usize>], n: usize) -> HashSet<Vec<usize>> {
    let mut set: HashSet<Vec<usize>> = HashSet::new();
    set.insert((0..n).collect());
    let mut frontier: Vec<Vec<usize>> = vec![(0..n).collect()];
    // the group is finite (at most 120 elements for molecules); stop at a generous bound
    while let Some(p) = frontier.pop() {
        for g in gens {
            let q: Vec<usize> = (0..n).map(|a| g[p[a]]).collect();
            if set.insert(q.clone()) {
                frontier.push(q);
            }
        }
        if set.len() > 1000 {
            break;
        }
    }
    set
}

/// The atoms `z` at `positions` (bohr) about their centre weighted by `weights`, atoms equivalent
/// when they have the same atomic number and weight.
fn centred(z: &[u32], weights: &[f64], positions: &[[f64; 3]], tol: f64) -> Frame {
    let n = z.len();
    let wtot: f64 = weights.iter().sum();
    let com = (0..3)
        .map(|k| (0..n).map(|a| weights[a] * positions[a][k]).sum::<f64>() / wtot)
        .collect::<Vec<_>>();
    let pos: Vec<V3> = positions
        .iter()
        .map(|r| [r[0] - com[0], r[1] - com[1], r[2] - com[2]])
        .collect();
    let mut kinds: Vec<(u32, f64)> = Vec::new();
    let kind: Vec<usize> = (0..n)
        .map(|a| {
            match kinds
                .iter()
                .position(|k| k.0 == z[a] && (k.1 - weights[a]).abs() < 1e-6)
            {
                Some(i) => i,
                None => {
                    kinds.push((z[a], weights[a]));
                    kinds.len() - 1
                }
            }
        })
        .collect();
    Frame { pos, kind, tol }
}

/// The inversion through the centre.
const INVERSION: M3 = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];

/// What the search for symmetry elements found ([`elements`]).
enum Elements {
    /// One atom.
    Atom,
    /// A linear structure along `axis`; `inversion`: the largest deviation of the inversion,
    /// None if it is not an operation; `off_axis`: the largest distance of an atom from the axis.
    Linear {
        axis: V3,
        inversion: Option<f64>,
        off_axis: f64,
    },
    /// Every other structure.
    General(Found),
}

/// The symmetry elements of a non-linear structure.
struct Found {
    /// Distinct rotation axes with the highest order found on each.
    rot_axes: Vec<(V3, usize)>,
    /// The order of the group of proper rotations (the rotational symmetry number).
    sigma: u32,
    inversion: bool,
    /// Mirror-plane normals.
    mirrors: Vec<V3>,
    /// The largest deviation of the operations found.
    dev: f64,
}

/// The symmetry elements of `frame` (atoms about the centre of `weights`).
fn elements(frame: &Frame, weights: &[f64]) -> Elements {
    let (pos, kind, tol) = (&frame.pos, &frame.kind, frame.tol);
    let n = pos.len();
    if n == 1 {
        return Elements::Atom;
    }
    let (_, axes) = inertia(weights, pos);
    // linear: every atom within tol of the axis of the smallest moment
    let lin_axis = axes[0];
    let off_axis = pos
        .iter()
        .map(|r| norm(cross(*r, lin_axis)))
        .fold(0.0, f64::max);
    if off_axis < tol {
        return Elements::Linear {
            axis: lin_axis,
            inversion: frame.perm(&INVERSION).map(|(_, d)| d),
            off_axis,
        };
    }
    // shells: equivalent atoms at the same distance from the centre of mass
    let mut shells: Vec<Vec<usize>> = Vec::new();
    for a in 0..n {
        let r = norm(pos[a]);
        match shells
            .iter_mut()
            .find(|s| kind[s[0]] == kind[a] && (norm(pos[s[0]]) - r).abs() < tol)
        {
            Some(s) => s.push(a),
            None => shells.push(vec![a]),
        }
    }
    let max_order = shells
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(1)
        .clamp(2, 24);
    // candidate rotation axes and mirror normals
    let mut cand = Directions::default();
    let mut normals = Directions::default();
    for ax in &axes {
        cand.push(*ax, 0.0);
        normals.push(*ax, 0.0);
    }
    for &r in pos {
        cand.push(r, tol);
    }
    for s in &shells {
        for (i, &a) in s.iter().enumerate() {
            for &b in &s[i + 1..] {
                cand.push(add(pos[a], pos[b]), tol);
                cand.push(cross(pos[a], pos[b]), tol * norm(pos[a]).max(1.0));
                normals.push(sub(pos[a], pos[b]), tol);
            }
        }
        // C_n (n >= 3) axes through no atom are normals of the regular polygons their orbits
        // form; adjacent vertices are near neighbours, so each atom with pairs of its four
        // nearest equivalent atoms covers them (all triples would cost O(m^3) for a big shell)
        if s.len() >= 3 {
            for &a in s {
                let mut near: Vec<usize> = s.iter().copied().filter(|&b| b != a).collect();
                near.sort_by(|&b, &c| {
                    norm(sub(pos[b], pos[a])).total_cmp(&norm(sub(pos[c], pos[a])))
                });
                near.truncate(4);
                for (i, &b) in near.iter().enumerate() {
                    for &c in &near[i + 1..] {
                        cand.push(
                            cross(sub(pos[b], pos[a]), sub(pos[c], pos[a])),
                            tol * norm(sub(pos[b], pos[a])).max(1.0),
                        );
                    }
                }
            }
        }
    }
    // proper rotations: the highest order that works on each candidate axis
    let mut dev: f64 = 0.0;
    let mut found: Vec<(V3, usize)> = Vec::new();
    let mut gens: Vec<Vec<usize>> = Vec::new();
    for &u in &cand.0 {
        for order in (2..=max_order).rev() {
            if let Some((p, d)) =
                frame.perm(&rotation(u, 2.0 * std::f64::consts::PI / order as f64))
            {
                dev = dev.max(d);
                gens.push(p);
                found.push((u, order));
                break;
            }
        }
    }
    let rot_group = closure(&gens, n);
    let sigma = rot_group.len() as u32;
    // distinct rotation axes (merge directions within ~1 degree), keeping the highest order
    let mut rot_axes: Vec<(V3, usize)> = Vec::new();
    for &(u, o) in &found {
        match rot_axes.iter_mut().find(|(w, _)| dot(*w, u).abs() > 0.9998) {
            Some(e) => e.1 = e.1.max(o),
            None => rot_axes.push((u, o)),
        }
    }
    // improper operations
    let inversion = frame
        .perm(&INVERSION)
        .inspect(|x| dev = dev.max(x.1))
        .is_some();
    for &(u, _) in &rot_axes {
        normals.push(u, 0.0);
    }
    let mut mirrors: Vec<V3> = Vec::new();
    for &u in &normals.0 {
        if let Some((_, d)) = frame.perm(&reflection(u)) {
            dev = dev.max(d);
            if mirrors.iter().all(|w| dot(*w, u).abs() < 0.9998) {
                mirrors.push(u);
            }
        }
    }
    Elements::General(Found {
        rot_axes,
        sigma,
        inversion,
        mirrors,
        dev,
    })
}

/// Point group and rotational symmetry number of atoms `z` with `masses` (u) at `positions`
/// (bohr). Atoms are equivalent if they have the same atomic number and mass.
pub fn detect(z: &[u32], masses: &[f64], positions: &[[f64; 3]], tol: f64) -> PointGroup {
    let frame = centred(z, masses, positions, tol);
    let pg = |name: &str, sigma: u32, dev: f64| PointGroup {
        name: name.into(),
        symmetry_number: sigma,
        max_deviation: dev,
        tolerance: tol,
    };
    match elements(&frame, masses) {
        Elements::Atom => pg("Kh", 1, 0.0),
        Elements::Linear {
            inversion: Some(d),
            off_axis,
            ..
        } => pg("D*h", 2, d.max(off_axis)),
        Elements::Linear {
            inversion: None,
            off_axis,
            ..
        } => pg("C*v", 1, off_axis),
        Elements::General(mut f) => {
            let name = classify(&frame, &f.rot_axes, f.inversion, &f.mirrors, &mut f.dev);
            pg(&name, f.sigma, f.dev)
        }
    }
}

/// Schoenflies symbol from the rotation axes (direction, highest order), inversion and mirror
/// planes (normals).
fn classify(
    frame: &Frame,
    axes: &[(V3, usize)],
    inversion: bool,
    mirrors: &[V3],
    dev: &mut f64,
) -> String {
    if axes.is_empty() {
        return if !mirrors.is_empty() {
            "Cs"
        } else if inversion {
            "Ci"
        } else {
            "C1"
        }
        .into();
    }
    let high: Vec<&(V3, usize)> = axes.iter().filter(|a| a.1 >= 3).collect();
    if high.len() >= 2 {
        let has = |o: usize| axes.iter().any(|a| a.1 == o);
        return if has(5) {
            if inversion {
                "Ih"
            } else {
                "I"
            }
        } else if has(4) {
            if inversion {
                "Oh"
            } else {
                "O"
            }
        } else if inversion {
            "Th"
        } else if !mirrors.is_empty() {
            "Td"
        } else {
            "T"
        }
        .into();
    }
    // principal axis: the highest order; with several (only C2 axes can tie: D2, D2d, D2h),
    // the choice that names the larger group (D2d's principal axis is its S4 axis)
    let nmax = axes.iter().map(|a| a.1).max().unwrap();
    let rank = |s: &str| {
        ["v", "d", "h"]
            .iter()
            .position(|x| s.ends_with(x))
            .map_or(0, |p| p + 1)
    };
    let mut best = String::new();
    for &(u, _) in axes.iter().filter(|a| a.1 == nmax) {
        let name = classify_axis(frame, axes, mirrors, u, nmax, dev);
        if best.is_empty() || rank(&name) > rank(&best) {
            best = name;
        }
    }
    best
}

/// The dihedral or cyclic group name with `u` (of order `nmax`) as the principal axis.
fn classify_axis(
    frame: &Frame,
    axes: &[(V3, usize)],
    mirrors: &[V3],
    u: V3,
    nmax: usize,
    dev: &mut f64,
) -> String {
    let perp_c2 = axes
        .iter()
        .filter(|a| a.1 == 2 && dot(a.0, u).abs() < 0.02)
        .count();
    let sigma_h = mirrors.iter().any(|m| dot(*m, u).abs() > 0.9998);
    let sigma_v = mirrors.iter().any(|m| dot(*m, u).abs() < 0.02);
    if perp_c2 >= nmax {
        let s = if sigma_h {
            "h"
        } else if sigma_v {
            "d"
        } else {
            ""
        };
        return format!("D{nmax}{s}");
    }
    if sigma_h {
        return format!("C{nmax}h");
    }
    if sigma_v {
        return format!("C{nmax}v");
    }
    // S_2n collinear with C_n
    let s2n = matmul3(
        &reflection(u),
        &rotation(u, std::f64::consts::PI / nmax as f64),
    );
    if let Some((_, d)) = frame.perm(&s2n) {
        *dev = dev.max(d);
        return format!("S{}", 2 * nmax);
    }
    format!("C{nmax}")
}

/// Principal moments of inertia (u bohr^2, ascending) and axes, for positions relative to the
/// centre of mass.
pub fn inertia(masses: &[f64], pos: &[V3]) -> ([f64; 3], [V3; 3]) {
    let mut t = faer::Mat::<f64>::zeros(3, 3);
    for (m, r) in masses.iter().zip(pos) {
        let r2 = dot(*r, *r);
        for i in 0..3 {
            for j in 0..3 {
                t[(i, j)] += m * (if i == j { r2 } else { 0.0 } - r[i] * r[j]);
            }
        }
    }
    let (w, v) = crate::linalg::eigh(t.as_ref(), faer::Par::Seq);
    let ax = |k: usize| [v[(0, k)], v[(1, k)], v[(2, k)]];
    (
        [w[0].max(0.0), w[1].max(0.0), w[2].max(0.0)],
        [ax(0), ax(1), ax(2)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The most abundant isotope's mass (u) of the elements below (linus-vib's table).
    fn mass(z: u32) -> f64 {
        match z {
            1 => 1.00782503223,
            5 => 11.00930536,
            6 => 12.0,
            7 => 14.00307400443,
            8 => 15.99491461957,
            9 => 18.99840316273,
            16 => 31.9720711744,
            _ => panic!("no test mass for Z = {z}"),
        }
    }

    fn pg(atoms: &[(u32, [f64; 3])]) -> PointGroup {
        let z: Vec<u32> = atoms.iter().map(|a| a.0).collect();
        let m: Vec<f64> = z.iter().map(|&zi| mass(zi)).collect();
        let c: Vec<[f64; 3]> = atoms.iter().map(|a| a.1).collect();
        detect(&z, &m, &c, DEFAULT_TOLERANCE)
    }

    /// Rotate and shift a structure so nothing lies along the coordinate axes.
    fn moved(atoms: &[(u32, [f64; 3])]) -> Vec<(u32, [f64; 3])> {
        let r = matmul3(
            &rotation(normalize([1.0, 2.0, 3.0]), 0.7),
            &rotation(normalize([-2.0, 0.5, 1.0]), 1.9),
        );
        atoms
            .iter()
            .map(|(z, p)| (*z, add(mv(&r, *p), [0.3, -1.2, 2.5])))
            .collect()
    }

    fn check(atoms: &[(u32, [f64; 3])], name: &str, sigma: u32) {
        for a in [atoms.to_vec(), moved(atoms)] {
            let g = pg(&a);
            assert_eq!((g.name.as_str(), g.symmetry_number), (name, sigma), "{g:?}");
            assert!(g.max_deviation < 1e-8);
        }
    }

    #[test]
    fn point_groups() {
        let s = 0.5_f64.sqrt();
        check(&[(8, [0.0, 0.0, 0.0])], "Kh", 1);
        check(
            &[
                (8, [0.0, 0.0, 0.2]),
                (1, [0.0, 1.4, -0.9]),
                (1, [0.0, -1.4, -0.9]),
            ],
            "C2v",
            2,
        );
        check(
            &[
                (8, [0.0, 0.0, 0.2]),
                (1, [0.0, 1.4, -0.9]),
                (1, [0.0, -1.5, -0.9]),
            ],
            "Cs",
            1,
        );
        check(
            &[
                (7, [0.0, 0.0, 0.2]),
                (1, [0.0, 1.8, -0.5]),
                (1, [1.8 * 0.75_f64.sqrt(), -0.9, -0.5]),
                (1, [-1.8 * 0.75_f64.sqrt(), -0.9, -0.5]),
            ],
            "C3v",
            3,
        );
        // linear
        check(
            &[
                (6, [0.0, 0.0, 0.0]),
                (8, [0.0, 0.0, 2.2]),
                (8, [0.0, 0.0, -2.2]),
            ],
            "D*h",
            2,
        );
        check(
            &[
                (1, [0.0, 0.0, -2.0]),
                (6, [0.0, 0.0, 0.0]),
                (7, [0.0, 0.0, 2.2]),
            ],
            "C*v",
            1,
        );
        // methane (Td), SF6 (Oh), ethylene (D2h), staggered ethane (D3d), benzene (D6h)
        let h = 1.2;
        check(
            &[
                (6, [0.0; 3]),
                (1, [h, h, h]),
                (1, [h, -h, -h]),
                (1, [-h, h, -h]),
                (1, [-h, -h, h]),
            ],
            "Td",
            12,
        );
        let f = 3.0;
        check(
            &[
                (16, [0.0; 3]),
                (9, [f, 0.0, 0.0]),
                (9, [-f, 0.0, 0.0]),
                (9, [0.0, f, 0.0]),
                (9, [0.0, -f, 0.0]),
                (9, [0.0, 0.0, f]),
                (9, [0.0, 0.0, -f]),
            ],
            "Oh",
            24,
        );
        check(
            &[
                (6, [1.26, 0.0, 0.0]),
                (6, [-1.26, 0.0, 0.0]),
                (1, [2.3, 1.7, 0.0]),
                (1, [2.3, -1.7, 0.0]),
                (1, [-2.3, 1.7, 0.0]),
                (1, [-2.3, -1.7, 0.0]),
            ],
            "D2h",
            4,
        );
        let mut ethane = vec![(6, [0.0, 0.0, 1.45]), (6, [0.0, 0.0, -1.45])];
        for k in 0..3 {
            let t = 2.0 * std::f64::consts::PI * k as f64 / 3.0;
            ethane.push((1, [1.9 * t.cos(), 1.9 * t.sin(), 2.1]));
            let t = t + std::f64::consts::PI / 3.0;
            ethane.push((1, [1.9 * t.cos(), 1.9 * t.sin(), -2.1]));
        }
        check(&ethane, "D3d", 6);
        let mut benzene = Vec::new();
        for k in 0..6 {
            let t = std::f64::consts::PI * k as f64 / 3.0;
            benzene.push((6, [2.64 * t.cos(), 2.64 * t.sin(), 0.0]));
            benzene.push((1, [4.69 * t.cos(), 4.69 * t.sin(), 0.0]));
        }
        check(&benzene, "D6h", 12);
        // allene (D2d), H2O2-like C2, trans-HONO-like... and an isotope breaking symmetry
        check(
            &[
                (6, [0.0; 3]),
                (6, [0.0, 0.0, 2.5]),
                (6, [0.0, 0.0, -2.5]),
                (1, [1.75, 0.0, 3.5]),
                (1, [-1.75, 0.0, 3.5]),
                (1, [0.0, 1.75, -3.5]),
                (1, [0.0, -1.75, -3.5]),
            ],
            "D2d",
            4,
        );
        check(
            &[
                (8, [s, 1.3, 0.0]),
                (8, [-s, -1.3, 0.0]),
                (1, [1.2, 1.7, 1.5]),
                (1, [-1.2, -1.7, 1.5]),
            ],
            "C2",
            2,
        );
        // icosahedral: B12H12 (axes through atoms, face centres, edge midpoints) and C60 (no
        // atom on any axis; 60 equivalent atoms)
        let phi = (1.0 + 5f64.sqrt()) / 2.0;
        let cyc = |v: [f64; 3]| [v, [v[1], v[2], v[0]], [v[2], v[0], v[1]]];
        let signs = |v: [f64; 3]| -> Vec<[f64; 3]> {
            let mut out: Vec<[f64; 3]> = Vec::new();
            for sx in [1.0, -1.0] {
                for sy in [1.0, -1.0] {
                    for sz in [1.0, -1.0] {
                        let w = [v[0] * sx, v[1] * sy, v[2] * sz];
                        if out.iter().all(|u| norm(sub(*u, w)) > 1e-9) {
                            out.push(w);
                        }
                    }
                }
            }
            out
        };
        let mut b12 = Vec::new();
        for v in cyc([0.0, 1.0, phi]) {
            for w in signs(v) {
                b12.push((5, scale(w, 3.4 / norm(w))));
                b12.push((1, scale(w, 5.6 / norm(w))));
            }
        }
        check(&b12, "Ih", 60);
        let mut c60 = Vec::new();
        for base in [
            [0.0, 1.0, 3.0 * phi],
            [1.0, 2.0 + phi, 2.0 * phi],
            [phi, 2.0, 2.0 * phi + 1.0],
        ] {
            for v in cyc(base) {
                for w in signs(v) {
                    c60.push((6, scale(w, 1.32)));
                }
            }
        }
        assert_eq!(c60.len(), 60);
        let t = std::time::Instant::now();
        check(&c60, "Ih", 60);
        assert!(
            t.elapsed().as_secs_f64() < 5.0,
            "C60 took {:?}",
            t.elapsed()
        );
        let z = [8, 1, 1];
        let m = [15.99491461957, 1.00782503223, 2.01410177812];
        let c = [[0.0, 0.0, 0.2], [0.0, 1.4, -0.9], [0.0, -1.4, -0.9]];
        let g = detect(&z, &m, &c, DEFAULT_TOLERANCE);
        assert_eq!((g.name.as_str(), g.symmetry_number), ("Cs", 1));
    }
}
