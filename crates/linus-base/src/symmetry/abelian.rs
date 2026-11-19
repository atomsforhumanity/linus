//! The largest abelian subgroup of a structure's point group in the D2h family (C1, Cs, Ci, C2,
//! C2v, C2h, D2, D2h): the group electronic-structure codes label orbitals with.
//!
//! Every element of these groups is an involution whose matrix is diagonal in one frame of three
//! orthogonal axes: the identity, the C2 rotations about the axes, the mirror planes normal to
//! them and the inversion. An element is written by the axes it reverses (bit k: axis k, so
//! C2(z) = x and y reversed), products are exclusive ors, and the irreducible representations
//! are the sign patterns over the generators: irrep `b` has the character -1 under generator `j`
//! when bit `j` of `b` is set, so the irrep of a product of functions is the exclusive or of
//! theirs and the character of irrep `b` under element `e` (a set of generators) is
//! (-1)^popcount(b & e).
//!
//! The frame is chosen among the structure's C2 axes and mirror normals (and the lab axes) to
//! give the largest group; among frames giving equally large groups, the one closest to the lab
//! axes wins (the lab-aligned Lebedev XC grids and PCM surfaces share its symmetry). The axes
//! are then named after Mulliken's conventions (J. Chem. Phys. 23, 1997 (1955)): z is the C2 axis
//! (in D2 and D2h the one through the most atoms) and x is perpendicular to the plane holding the
//! most atoms, so planar C2v molecules lie in the yz plane (water's HOMO is 1b1). Electronic
//! symmetry ignores masses: atoms are equivalent when they have the same atomic number (the
//! centre is the nuclear charge's), and a basis that differs between such atoms is the AO
//! representation's concern ([`super::ao`]).

use super::v3::*;
use super::{centred, elements, Elements, Frame};

/// An element of a D2h-family group in its frame: bit k set when it reverses axis k (x, y, z).
type Signs = u8;

const C2Y: Signs = 0b101;
const C2Z: Signs = 0b011;
const SIGMA_XY: Signs = 0b100;
const SIGMA_XZ: Signs = 0b010;
const INVERSION: Signs = 0b111;

/// One operation of a structure's point group: the orthogonal matrix about the structure's
/// centre (r -> matrix r) and the permutation of atoms it induces (atom a goes to `perm[a]`).
#[derive(Clone, Debug, PartialEq)]
pub struct Operation {
    pub matrix: M3,
    pub perm: Vec<usize>,
}

/// The largest D2h-family subgroup of a structure's point group (see the module docs).
#[derive(Clone, Debug)]
pub struct AbelianGroup {
    /// Schoenflies symbol: C1, Cs, Ci, C2, C2v, C2h, D2 or D2h.
    pub name: &'static str,
    /// The frame's x, y and z axes (orthonormal, right-handed).
    pub axes: [V3; 3],
    /// The generators (0 to 3), in the order the irrep names are written for.
    pub generators: Vec<Operation>,
    /// The generators in the frame.
    signs: Vec<Signs>,
    /// The atoms about the centre (to re-frame a subgroup).
    frame: Frame,
}

impl AbelianGroup {
    /// The number of elements (and of irreducible representations).
    pub fn order(&self) -> usize {
        1 << self.signs.len()
    }

    /// The trivial group (C1) of a structure.
    pub fn trivial(&self) -> bool {
        self.signs.is_empty()
    }

    /// The name of irrep `b` (Mulliken's symbols in the frame of [`AbelianGroup::axes`]).
    pub fn irrep_name(&self, b: usize) -> &'static str {
        const NAMES: [(&str, &[&str]); 8] = [
            ("C1", &["A"]),
            ("Cs", &["A'", "A\""]),
            ("Ci", &["Ag", "Au"]),
            ("C2", &["A", "B"]),
            // generators C2(z), sigma(xz)
            ("C2v", &["A1", "B1", "A2", "B2"]),
            // generators C2(z), i
            ("C2h", &["Ag", "Bg", "Au", "Bu"]),
            // generators C2(z), C2(y)
            ("D2", &["A", "B2", "B1", "B3"]),
            // generators C2(z), C2(y), i
            (
                "D2h",
                &["Ag", "B2g", "B1g", "B3g", "Au", "B2u", "B1u", "B3u"],
            ),
        ];
        NAMES
            .iter()
            .find(|(g, _)| *g == self.name)
            .map(|(_, n)| n[b])
            .expect("a D2h-family group")
    }

    /// The character of irrep `b` under the element `e` (the product of the generators whose
    /// bits are set).
    pub fn character(b: usize, e: usize) -> f64 {
        if (b & e).count_ones().is_multiple_of(2) {
            1.0
        } else {
            -1.0
        }
    }

    /// The element `e` (the product of the generators whose bits are set; 0: the identity).
    pub fn element(&self, e: usize) -> Operation {
        let n = self.frame.pos.len();
        let mut op = Operation {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            perm: (0..n).collect(),
        };
        for (j, g) in self.generators.iter().enumerate() {
            if e >> j & 1 == 1 {
                op.matrix = super::matmul3(&g.matrix, &op.matrix);
                op.perm = op.perm.iter().map(|&a| g.perm[a]).collect();
            }
        }
        op
    }

    /// The largest subgroup whose elements (as sets of generators, the identity 0 excluded) all
    /// pass `keep`, named and framed as a group of its own: the symmetry left when some elements
    /// are not symmetries of what the group acts on (a converged state that broke them).
    pub fn subgroup(&self, keep: impl Fn(usize) -> bool) -> AbelianGroup {
        let signs_of = |e: usize| -> Signs {
            (0..self.signs.len())
                .filter(|j| e >> j & 1 == 1)
                .fold(0, |s, j| s ^ self.signs[j])
        };
        let kept: Vec<Signs> = (1..self.order())
            .filter(|&e| keep(e))
            .map(signs_of)
            .collect();
        canonical(&self.frame, self.axes, &largest_subgroup(&kept))
    }
}

/// The span of `gens` under exclusive or (the identity included).
fn span(gens: &[Signs]) -> Vec<Signs> {
    let mut set = vec![0];
    for &g in gens {
        if !set.contains(&g) {
            let more: Vec<Signs> = set.iter().map(|&s| s ^ g).collect();
            set.extend(more);
        }
    }
    set
}

/// The largest group (closed under exclusive or) whose non-identity elements lie in `kept`;
/// among equally large ones, the first in a fixed order of generator choices. Its non-identity
/// elements.
fn largest_subgroup(kept: &[Signs]) -> Vec<Signs> {
    let mut best: Vec<Signs> = vec![0];
    for &a in kept {
        for &b in kept {
            for &c in kept {
                let s = span(&[a, b, c]);
                if s.len() > best.len() && s.iter().all(|&x| x == 0 || kept.contains(&x)) {
                    best = s;
                }
            }
        }
    }
    best.retain(|&x| x != 0);
    best
}

/// The matrix of the element `s` in the frame `axes`: A^T diag(+-1) A.
fn matrix(axes: &[V3; 3], s: Signs) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for (k, a) in axes.iter().enumerate() {
        let sign = if s >> k & 1 == 1 { -1.0 } else { 1.0 };
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += sign * a[i] * a[j];
            }
        }
    }
    m
}

/// The elements of the D2h group in the frame `axes` that are operations of `frame`.
fn operations_in(frame: &Frame, axes: &[V3; 3]) -> Vec<Signs> {
    (1..8u8)
        .filter(|&s| frame.perm(&matrix(axes, s)).is_some())
        .collect()
}

/// A right-handed orthonormal frame with z along `z` and x along the part of `x` orthogonal
/// to it.
fn frame_from(z: V3, x: V3) -> [V3; 3] {
    let z = normalize(z);
    let x = normalize(sub(x, scale(z, dot(x, z))));
    [x, cross(z, x), z]
}

/// The lab axis least aligned with `u` (the first of equal ones).
fn least_aligned(u: V3) -> V3 {
    let k = (0..3)
        .min_by(|&a, &b| u[a].abs().total_cmp(&u[b].abs()))
        .unwrap_or(0);
    let mut e = [0.0; 3];
    e[k] = 1.0;
    e
}

/// How close a frame is to the lab axes: the sum over its axes of the largest |component|.
fn alignment(axes: &[V3; 3]) -> f64 {
    axes.iter()
        .map(|a| a.iter().fold(0.0f64, |m, x| m.max(x.abs())))
        .sum()
}

/// The group `elems` (non-identity, closed) of `frame` in the frame `axes`, with its axes
/// renamed by the conventions of the module docs and its canonical generators.
fn canonical(frame: &Frame, axes: [V3; 3], elems: &[Signs]) -> AbelianGroup {
    let tol = frame.tol;
    let on_axis = |a: V3| {
        frame
            .pos
            .iter()
            .filter(|r| norm(cross(**r, a)) < tol)
            .count()
    };
    let in_plane = |n: V3| frame.pos.iter().filter(|r| dot(**r, n).abs() < tol).count();
    let has = |s: Signs| elems.contains(&s);
    // the frame axis reversed by none but this C2: (k: the axis whose C2 is in the group)
    let c2_axis = |k: usize| has(0b111 ^ (1 << k));
    let c2s: Vec<usize> = (0..3).filter(|&k| c2_axis(k)).collect();
    let mirrors: Vec<usize> = (0..3).filter(|&k| has(1 << k)).collect();
    // (name, the frame axes in x, y, z order, generators)
    let (name, order, gens): (&'static str, [usize; 3], Vec<Signs>) = match elems.len() {
        0 => ("C1", [0, 1, 2], vec![]),
        1 if has(INVERSION) => ("Ci", [0, 1, 2], vec![INVERSION]),
        1 if !c2s.is_empty() => ("C2", rest_then(c2s[0]), vec![C2Z]),
        1 => ("Cs", rest_then(mirrors[0]), vec![SIGMA_XY]),
        3 if c2s.len() == 3 => ("D2", dihedral(&axes, on_axis, in_plane), vec![C2Z, C2Y]),
        3 if has(INVERSION) => ("C2h", rest_then(c2s[0]), vec![C2Z, INVERSION]),
        3 => {
            // C2v: z the C2 axis, x normal to the mirror holding more atoms
            let z = c2s[0];
            let (a, b) = ((z + 1) % 3, (z + 2) % 3);
            let (a, b) = (a.min(b), a.max(b));
            let x = if in_plane(axes[b]) > in_plane(axes[a]) {
                b
            } else {
                a
            };
            ("C2v", [x, 3 - x - z, z], vec![C2Z, SIGMA_XZ])
        }
        _ => (
            "D2h",
            dihedral(&axes, on_axis, in_plane),
            vec![C2Z, C2Y, INVERSION],
        ),
    };
    let mut new = [axes[order[0]], axes[order[1]], axes[order[2]]];
    // right-handed (the sign of an axis changes none of the elements)
    if dot(cross(new[0], new[1]), new[2]) < 0.0 {
        new[1] = scale(new[1], -1.0);
    }
    let generators = gens
        .iter()
        .map(|&s| {
            let m = matrix(&new, s);
            let (perm, _) = frame
                .perm(&m)
                .expect("an element of the group is an operation of the structure");
            Operation { matrix: m, perm }
        })
        .collect();
    AbelianGroup {
        name,
        axes: new,
        generators,
        signs: gens,
        frame: frame.clone(),
    }
}

/// The frame axes in x, y, z order with axis `z` last (the other two in frame order).
fn rest_then(z: usize) -> [usize; 3] {
    let (a, b) = ((z + 1) % 3, (z + 2) % 3);
    [a.min(b), a.max(b), z]
}

/// The axes of D2 and D2h in x, y, z order: z the axis through the most atoms, x the one
/// normal to the plane (of the other two) holding the most atoms; ties to the frame order.
fn dihedral(
    axes: &[V3; 3],
    on_axis: impl Fn(V3) -> usize,
    in_plane: impl Fn(V3) -> usize,
) -> [usize; 3] {
    let z = (0..3)
        .rev()
        .max_by_key(|&k| on_axis(axes[k]))
        .expect("three axes");
    let [a, b, _] = rest_then(z);
    let x = if in_plane(axes[b]) > in_plane(axes[a]) {
        b
    } else {
        a
    };
    [x, 3 - x - z, z]
}

/// The largest abelian (D2h-family) subgroup of the point group of atoms `z` at `positions`
/// (bohr), with tolerance `tol` (bohr) as [`super::detect`]'s (see the module docs).
pub fn abelian_group(z: &[u32], positions: &[[f64; 3]], tol: f64) -> AbelianGroup {
    let weights: Vec<f64> = z.iter().map(|&zi| zi as f64).collect();
    let frame = centred(z, &weights, positions, tol);
    let lab: [V3; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut frames: Vec<[V3; 3]> = vec![lab];
    match elements(&frame, &weights) {
        Elements::Atom => {}
        Elements::Linear { axis, .. } => frames.push(frame_from(axis, least_aligned(axis))),
        Elements::General(found) => {
            // the directions of the elements: C2 axes (even-order rotations) and mirror normals
            let mut dirs: Vec<V3> = Vec::new();
            let c2 = found.rot_axes.iter().filter(|a| a.1 % 2 == 0).map(|a| a.0);
            for u in c2.chain(found.mirrors.iter().copied()) {
                if dirs.iter().all(|w| dot(*w, u).abs() < 0.9998) {
                    dirs.push(u);
                }
            }
            for (i, &d) in dirs.iter().enumerate() {
                frames.push(frame_from(d, least_aligned(d)));
                for &e in &dirs[i + 1..] {
                    if dot(d, e).abs() < 0.02 {
                        frames.push(frame_from(d, e));
                    }
                }
            }
        }
    }
    // the largest group, then the frame closest to the lab axes, then the first
    let mut best: Option<([V3; 3], Vec<Signs>, f64)> = None;
    for axes in frames {
        let elems = largest_subgroup(&operations_in(&frame, &axes));
        let score = alignment(&axes);
        let better = match &best {
            None => true,
            Some((_, e, s)) => {
                elems.len() > e.len() || (elems.len() == e.len() && score > s + 1e-9)
            }
        };
        if better {
            best = Some((axes, elems, score));
        }
    }
    let (axes, elems, _) = best.expect("the lab frame is a candidate");
    canonical(&frame, axes, &elems)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(atoms: &[(u32, [f64; 3])]) -> AbelianGroup {
        let z: Vec<u32> = atoms.iter().map(|a| a.0).collect();
        let c: Vec<[f64; 3]> = atoms.iter().map(|a| a.1).collect();
        abelian_group(&z, &c, super::super::DEFAULT_TOLERANCE)
    }

    /// Rotate and shift a structure so nothing lies along the coordinate axes.
    fn moved(atoms: &[(u32, [f64; 3])]) -> Vec<(u32, [f64; 3])> {
        let r = super::super::matmul3(
            &super::super::rotation(normalize([1.0, 2.0, 3.0]), 0.7),
            &super::super::rotation(normalize([-2.0, 0.5, 1.0]), 1.9),
        );
        atoms
            .iter()
            .map(|(z, p)| (*z, add(mv(&r, *p), [0.3, -1.2, 2.5])))
            .collect()
    }

    fn water() -> Vec<(u32, [f64; 3])> {
        vec![
            (8, [0.0, 0.0, 0.2]),
            (1, [0.0, 1.4, -0.9]),
            (1, [0.0, -1.4, -0.9]),
        ]
    }

    #[test]
    fn groups_and_frames() {
        let s = 0.5_f64.sqrt();
        let h = 1.2;
        let methane = vec![
            (6, [0.0; 3]),
            (1, [h, h, h]),
            (1, [h, -h, -h]),
            (1, [-h, h, -h]),
            (1, [-h, -h, h]),
        ];
        let ethylene = vec![
            (6, [1.26, 0.0, 0.0]),
            (6, [-1.26, 0.0, 0.0]),
            (1, [2.3, 1.7, 0.0]),
            (1, [2.3, -1.7, 0.0]),
            (1, [-2.3, 1.7, 0.0]),
            (1, [-2.3, -1.7, 0.0]),
        ];
        let nh3 = vec![
            (7, [0.0, 0.0, 0.2]),
            (1, [0.0, 1.8, -0.5]),
            (1, [1.8 * 0.75_f64.sqrt(), -0.9, -0.5]),
            (1, [-1.8 * 0.75_f64.sqrt(), -0.9, -0.5]),
        ];
        let h2o2 = vec![
            (8, [s, 1.3, 0.0]),
            (8, [-s, -1.3, 0.0]),
            (1, [1.2, 1.7, 1.5]),
            (1, [-1.2, -1.7, 1.5]),
        ];
        let trans = vec![
            (8, [1.3, 0.2, 0.0]),
            (8, [-1.3, -0.2, 0.0]),
            (1, [1.9, 1.8, 0.0]),
            (1, [-1.9, -1.8, 0.0]),
        ];
        let co2 = vec![(6, [0.0; 3]), (8, [0.0, 0.0, 2.2]), (8, [0.0, 0.0, -2.2])];
        let hcn = vec![
            (1, [0.0, 0.0, -2.0]),
            (6, [0.0, 0.0, 0.0]),
            (7, [0.0, 0.0, 2.2]),
        ];
        let hdo = water();
        for (atoms, name, order) in [
            (vec![(26, [0.0; 3])], "D2h", 8),
            (water(), "C2v", 4),
            // Td: D2 (lab-aligned C2 axes) and C2v are both of order 4
            (methane, "", 4),
            (ethylene, "D2h", 8),
            (nh3, "Cs", 2),
            (h2o2, "C2", 2),
            (trans, "C2h", 4),
            (co2, "D2h", 8),
            (hcn, "C2v", 4),
        ] {
            for a in [atoms.clone(), moved(&atoms)] {
                let g = group(&a);
                assert_eq!(g.order(), order, "{a:?}");
                assert!(name.is_empty() || g.name == name, "{} {a:?}", g.name);
                // every generator is an orthogonal involution that permutes the atoms
                for op in &g.generators {
                    let sq = super::super::matmul3(&op.matrix, &op.matrix);
                    for (i, row) in sq.iter().enumerate() {
                        for (j, x) in row.iter().enumerate() {
                            let want = if i == j { 1.0 } else { 0.0 };
                            assert!((x - want).abs() < 1e-12);
                        }
                    }
                }
            }
        }
        // isotopes do not break electronic symmetry: HDO is C2v (masses are not an input)
        assert_eq!(group(&hdo).name, "C2v");
        // planar C2v lies in the yz plane: x is normal to the molecular plane
        let g = group(&water());
        assert!((g.axes[0][0].abs() - 1.0).abs() < 1e-12, "{:?}", g.axes);
        assert!((g.axes[2][2].abs() - 1.0).abs() < 1e-12, "{:?}", g.axes);
        // ethylene: z along C=C, x normal to the plane
        let g = group(&[
            (6, [1.26, 0.0, 0.0]),
            (6, [-1.26, 0.0, 0.0]),
            (1, [2.3, 1.7, 0.0]),
            (1, [2.3, -1.7, 0.0]),
            (1, [-2.3, 1.7, 0.0]),
            (1, [-2.3, -1.7, 0.0]),
        ]);
        assert!((g.axes[2][0].abs() - 1.0).abs() < 1e-12, "{:?}", g.axes);
        assert!((g.axes[0][2].abs() - 1.0).abs() < 1e-12, "{:?}", g.axes);
    }

    #[test]
    fn irreps_and_subgroups() {
        let g = group(&water());
        assert_eq!(
            (0..4).map(|b| g.irrep_name(b)).collect::<Vec<_>>(),
            ["A1", "B1", "A2", "B2"]
        );
        // irrep B1 under C2 (element 1) and sigma(xz) (element 2) and sigma(yz) (element 3)
        assert_eq!(AbelianGroup::character(1, 1), -1.0);
        assert_eq!(AbelianGroup::character(1, 2), 1.0);
        assert_eq!(AbelianGroup::character(1, 3), -1.0);
        // without C2 (element 1) the largest subgroup is one of the mirrors: Cs
        let cs = g.subgroup(|e| e != 1 && e != 3);
        assert_eq!((cs.name, cs.order()), ("Cs", 2));
        // the mirror kept is sigma(xz): normal y, the Cs frame's z
        assert!((dot(cs.axes[2], g.axes[1]).abs() - 1.0).abs() < 1e-12);
        let c1 = g.subgroup(|_| false);
        assert!(c1.trivial() && c1.irrep_name(0) == "A");
        let d2h = group(&[(26, [0.0; 3])]);
        assert_eq!(d2h.subgroup(|_| true).name, "D2h");
        // D2h without the inversion and without the C2 about x and y: C2v about z?
        let sub = d2h.subgroup(|e| [1, 6, 7].contains(&e));
        assert_eq!(sub.name, "C2v");
    }
}
