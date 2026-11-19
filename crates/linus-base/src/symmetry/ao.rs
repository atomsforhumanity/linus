//! How a basis set's functions transform under an operation of the structure's point group:
//! g chi_mu = sum_nu chi_nu R_nu,mu, so an orbital with coefficients c goes to R c.
//!
//! An operation r -> O r maps atom A onto atom perm(A), and a shell on A onto the same shell on
//! perm(A) (the atoms are equivalent, so their basis functions are the same), with its angular
//! part rotated: a function f(r - R_A) goes to f(O^T (r - R_perm(A))). For the shell's angular
//! functions f_m (homogeneous polynomials of degree l: the Cartesian components x^a y^b z^c, all
//! with the normalization of x^l, or the real solid harmonics over them, linus-ints'
//! `cart2sph`), f_m(O^T u) = sum_m' f_m'(u) D_m'm. D is exact: the Cartesian components' images
//! are expanded as products of the linear forms (O^T u)_i (matrix M), and a spherical shell's
//! D = T G M T^T with T the solid harmonics over the components and G the components' metric, in
//! which T's rows are orthonormal.

use super::abelian::{abelian_group, AbelianGroup, Operation};
use super::v3::M3;
use faer::{Mat, MatRef};
use linus_ints::basis::{cart_components, ncart, BasisSet};
use linus_ints::core::solid_harmonics::{cart2sph, cart_metric};

/// One shell's image: its first function, the image shell's first function, and D (nfunc x
/// nfunc, row-major `D[m'][m]`).
#[derive(Clone, Debug)]
struct ShellImage {
    from: usize,
    to: usize,
    d: Vec<f64>,
}

/// The matrix R of an operation on a basis set's functions (see the module docs), stored per
/// shell.
#[derive(Clone, Debug)]
pub struct AoOperation {
    nbf: usize,
    shells: Vec<ShellImage>,
}

impl AoOperation {
    /// The operation `op` on the functions of `basis` (its atoms those `op` permutes); None when
    /// it maps an atom onto one whose shells differ (a basis that breaks the symmetry).
    pub fn new(basis: &BasisSet, op: &Operation) -> Option<AoOperation> {
        let mut shells = Vec::with_capacity(basis.shells.len());
        let mut cache: Vec<((usize, bool), Vec<f64>)> = Vec::new();
        for (a, range) in basis.atom_shells.iter().enumerate() {
            let image = basis.atom_shells.get(*op.perm.get(a)?)?;
            if image.len() != range.len() {
                return None;
            }
            for (s, t) in range.clone().zip(image.clone()) {
                let (sa, sb) = (&basis.shells[s], &basis.shells[t]);
                if sa.l != sb.l || sa.pure != sb.pure || sa.exps != sb.exps || sa.coefs != sb.coefs
                {
                    return None;
                }
                let key = (sa.l, sa.pure);
                let d = match cache.iter().find(|c| c.0 == key) {
                    Some(c) => c.1.clone(),
                    None => {
                        let d = shell_matrix(sa.l, sa.pure, &op.matrix);
                        cache.push((key, d.clone()));
                        d
                    }
                };
                shells.push(ShellImage {
                    from: sa.ao_start,
                    to: sb.ao_start,
                    d,
                });
            }
        }
        Some(AoOperation {
            nbf: basis.nbf,
            shells,
        })
    }

    /// R c: the coefficients of the images of the functions with coefficients `c` (nbf x k).
    pub fn apply(&self, c: MatRef<'_, f64>) -> Mat<f64> {
        let mut out = Mat::<f64>::zeros(self.nbf, c.ncols());
        for sh in &self.shells {
            let n = (sh.d.len() as f64).sqrt().round() as usize;
            for p in 0..c.ncols() {
                for i in 0..n {
                    let mut v = 0.0;
                    for j in 0..n {
                        v += sh.d[i * n + j] * c[(sh.from + j, p)];
                    }
                    out[(sh.to + i, p)] = v;
                }
            }
        }
        out
    }
}

/// A structure's abelian point group ([`super::abelian`]) acting on the functions of its basis
/// set: the largest D2h-family subgroup whose every element maps each atom's shells onto
/// identical ones, and its generators' [`AoOperation`]s.
#[derive(Clone, Debug)]
pub struct AoGroup {
    pub group: AbelianGroup,
    /// The generators on the basis functions, in the group's order.
    pub generators: Vec<AoOperation>,
}

impl AoGroup {
    /// The group of atoms `z` at `positions` (bohr; tolerance `tol`) on `basis`.
    pub fn new(z: &[u32], positions: &[[f64; 3]], basis: &BasisSet, tol: f64) -> AoGroup {
        let mut group = abelian_group(z, positions, tol);
        let fits = |g: &AbelianGroup, e: usize| AoOperation::new(basis, &g.element(e)).is_some();
        if (1..group.order()).any(|e| !fits(&group, e)) {
            let g0 = group.clone();
            group = g0.subgroup(|e| fits(&g0, e));
        }
        let generators = group
            .generators
            .iter()
            .map(|op| AoOperation::new(basis, op).expect("an element that maps the basis"))
            .collect();
        AoGroup { group, generators }
    }
}

/// D of a shell of angular momentum `l` (spherical if `pure`) under the orthogonal `o`: f_m(O^T
/// u) = sum_m' f_m'(u) `D[m'][m]`, row-major.
pub fn shell_matrix(l: usize, pure: bool, o: &M3) -> Vec<f64> {
    let m = cartesian_matrix(l, o);
    if !pure {
        return m;
    }
    let nc = ncart(l);
    let nf = 2 * l + 1;
    let t = cart2sph(l);
    let g = cart_metric(l);
    // D = T G M T^T
    let mut tg = vec![0.0; nf * nc];
    for a in 0..nf {
        for k in 0..nc {
            tg[a * nc + k] = (0..nc).map(|j| t[a * nc + j] * g[j * nc + k]).sum();
        }
    }
    let mut mt = vec![0.0; nc * nf];
    for k in 0..nc {
        for b in 0..nf {
            mt[k * nf + b] = (0..nc).map(|j| m[k * nc + j] * t[b * nc + j]).sum();
        }
    }
    let mut d = vec![0.0; nf * nf];
    for a in 0..nf {
        for b in 0..nf {
            d[a * nf + b] = (0..nc).map(|k| tg[a * nc + k] * mt[k * nf + b]).sum();
        }
    }
    d
}

/// M of the Cartesian components of degree `l` under `o`: x^a y^b z^c at O^T u, expanded as
/// sum_k' `M[k'][k]` u^(k'), row-major (component order of linus-ints' `cart_components`).
fn cartesian_matrix(l: usize, o: &M3) -> Vec<f64> {
    let comps = cart_components(l);
    let nc = comps.len();
    let index = |e: [usize; 3]| comps.iter().position(|c| *c == e).expect("a component");
    // a polynomial of degree d as coefficients over the exponents (a, b, c), dense in (l+1)^3
    let dim = l + 1;
    let at = |a: usize, b: usize, c: usize| (a * dim + b) * dim + c;
    let mut m = vec![0.0; nc * nc];
    for (k, e) in comps.iter().enumerate() {
        let mut poly = vec![0.0; dim * dim * dim];
        poly[at(0, 0, 0)] = 1.0;
        let mut degree = 0;
        // (O^T u)_i = sum_j O[j][i] u_j, once per power of component i
        for (i, &power) in e.iter().enumerate() {
            for _ in 0..power {
                let mut next = vec![0.0; dim * dim * dim];
                for a in 0..=degree {
                    for b in 0..=degree - a {
                        let c = degree - a - b;
                        let v = poly[at(a, b, c)];
                        if v == 0.0 {
                            continue;
                        }
                        next[at(a + 1, b, c)] += v * o[0][i];
                        next[at(a, b + 1, c)] += v * o[1][i];
                        next[at(a, b, c + 1)] += v * o[2][i];
                    }
                }
                poly = next;
                degree += 1;
            }
        }
        for a in 0..=l {
            for b in 0..=l - a {
                let c = l - a - b;
                m[index([a, b, c]) * nc + k] = poly[at(a, b, c)];
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::super::abelian::abelian_group;
    use super::super::v3::*;
    use super::*;
    use crate::linalg::{ao_view_mut, mm};
    use faer::Par;
    use linus_ints::{BasisDef, DerivOrder, OneEPlan};

    fn rot(u: V3, t: f64) -> M3 {
        super::super::rotation(normalize(u), t)
    }

    /// D is orthogonal for spherical shells and a representation: D(O1 O2) = D(O1) D(O2).
    #[test]
    fn shell_matrices_represent_rotations() {
        let (o1, o2) = (rot([1.0, 2.0, 3.0], 0.7), rot([-2.0, 0.5, 1.0], 1.9));
        let o12 = super::super::matmul3(&o1, &o2);
        for l in 0..=5 {
            for pure in [true, false] {
                let (d1, d2, d12) = (
                    shell_matrix(l, pure, &o1),
                    shell_matrix(l, pure, &o2),
                    shell_matrix(l, pure, &o12),
                );
                let n = (d1.len() as f64).sqrt() as usize;
                for i in 0..n {
                    for j in 0..n {
                        let p: f64 = (0..n).map(|k| d1[i * n + k] * d2[k * n + j]).sum();
                        assert!((p - d12[i * n + j]).abs() < 1e-12, "l {l} pure {pure}");
                        if pure {
                            let q: f64 = (0..n).map(|k| d1[k * n + i] * d1[k * n + j]).sum();
                            let want = if i == j { 1.0 } else { 0.0 };
                            assert!((q - want).abs() < 1e-12, "l {l}: D^T D");
                        }
                    }
                }
            }
        }
    }

    /// The overlap of a symmetric molecule in a general orientation (water, def2-TZVP: d and f
    /// shells) is invariant under every generator of its group: R^T S R = S.
    #[test]
    fn overlap_is_invariant() {
        let r = super::super::matmul3(&rot([1.0, 2.0, 3.0], 0.7), &rot([-2.0, 0.5, 1.0], 1.9));
        let pos: Vec<[f64; 3]> = [[0.0, 0.0, 0.2], [0.0, 1.4, -0.9], [0.0, -1.4, -0.9]]
            .iter()
            .map(|p| add(mv(&r, *p), [0.3, -1.2, 2.5]))
            .collect();
        let z = [8, 1, 1];
        let g = abelian_group(&z, &pos, super::super::DEFAULT_TOLERANCE);
        assert_eq!(g.name, "C2v");
        for pure in [true, false] {
            let def = BasisDef::load("def2-tzvp", None).unwrap();
            let basis = BasisSet::new(&def, &z, &pos, pure).unwrap();
            let n = basis.nbf;
            let mut s = Mat::<f64>::zeros(n, n);
            OneEPlan::new(&basis, DerivOrder::Zero).overlap(&mut ao_view_mut(s.as_mut()));
            for op in &g.generators {
                let ao = AoOperation::new(&basis, op).expect("the same basis on both H");
                let rm = ao.apply(Mat::<f64>::identity(n, n).as_ref());
                let srr = mm(
                    rm.transpose(),
                    mm(s.as_ref(), rm.as_ref(), Par::Seq).as_ref(),
                    Par::Seq,
                );
                for i in 0..n {
                    for j in 0..n {
                        assert!(
                            (srr[(i, j)] - s[(i, j)]).abs() < 1e-10,
                            "pure {pure} ({i}, {j}): {} vs {}",
                            srr[(i, j)],
                            s[(i, j)]
                        );
                    }
                }
            }
        }
    }
}
