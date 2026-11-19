//! Molden files of the SCF orbitals (`output.molden`), the format Molden, Jmol, Avogadro,
//! Multiwfn and PySCF read: atomic units, functions up to g, spherical (`[5D7F]`, `[9G]`) or
//! Cartesian. The CLI writes the file (`--molden FILE`); every front end finds it in the
//! result's `files` under [`MOLDEN_FILE`].
//!
//! The file is the one linus-py's `write_molden` writes from the result models, with the same
//! layout and number formats: shells by atom, coefficients of x^l-normalized primitives, the
//! functions in Molden's order (spherical m = 0, +1, -1, +2, -2, ...; Cartesian xx, yy, zz, xy,
//! xz, yz, ...) with unit-normalized Cartesian components. A `cartesian` file of a spherical
//! basis set expands each spherical function in its Cartesian components (the same orbitals:
//! the Cartesian set spans the spherical one). Restricted orbitals are written once with the
//! total occupations, unrestricted ones as alpha, then beta. Atoms with an ECP carry their
//! effective charge Z - n_core, as Molden readers expect, and the `[core]` section lists their
//! core electrons (`atom : n_core`, Molden2AIM's section, which PySCF reads and writes), so a
//! reader knows them for ECP atoms and not for all-electron ones of a smaller charge.

use crate::driver::options::MoldenForm;
use faer::Mat;
use linus_base::error::LinusError;
use linus_ints::basis::{ncart, BasisSet, Shell};
use std::fmt::Write as _;

/// The name of the Molden file in a result's `files`.
pub const MOLDEN_FILE: &str = "orbitals.molden";

/// The highest angular momentum a Molden file holds (g).
const LMAX: usize = 4;

const L_LETTERS: [char; LMAX + 1] = ['s', 'p', 'd', 'f', 'g'];

/// One set of orbitals: its spin label (`Alpha`, `Beta`), coefficients (nbf x nmo), energies
/// and occupations (one each per orbital).
pub(crate) struct OrbitalSet<'a> {
    pub spin: &'static str,
    pub c: &'a Mat<f64>,
    pub energies: &'a [f64],
    pub occupations: Vec<f64>,
}

/// A Molden file of `form` can hold the orbitals of `basis`: functions up to g, and a
/// `spherical` file needs spherical functions (Cartesian d functions hold an s-type
/// combination a spherical file cannot).
pub(crate) fn check(basis: &BasisSet, form: MoldenForm) -> Result<(), LinusError> {
    if form == MoldenForm::Off {
        return Ok(());
    }
    if let Some(l) = basis.shells.iter().map(|s| s.l).max().filter(|&l| l > LMAX) {
        return Err(LinusError::Setting {
            code: "unsupported",
            path: "output.molden".into(),
            message: format!(
                "Molden files stop at g functions (l = {LMAX}); this basis set has l = {l} \
                 (remove the setting, or use a basis set without {} functions)",
                letter(l)
            ),
        });
    }
    if form == MoldenForm::Spherical && basis.shells.iter().any(|s| s.l >= 2 && !s.pure) {
        return Err(LinusError::Setting {
            code: "unsupported",
            path: "output.molden".into(),
            message: "\"spherical\" needs spherical basis functions, and this calculation's are \
                      Cartesian (cartesian: true); use output.molden \"cartesian\""
                .into(),
        });
    }
    Ok(())
}

/// The Molden file of `sets` (orbitals in `basis`'s AO order) on the atoms `symbols` at
/// `positions` (bohr) with atomic numbers `z` and ECP core electrons `core` (zero for an
/// all-electron atom). `form` is `spherical` or `cartesian`, and [`check`] passed for it.
pub(crate) fn write(
    symbols: &[&str],
    z: &[u32],
    core: &[u32],
    positions: &[[f64; 3]],
    basis: &BasisSet,
    form: MoldenForm,
    sets: &[OrbitalSet],
) -> String {
    let cartesian = form == MoldenForm::Cartesian;
    // the shells by atom (a stable sort: the basis set's order within an atom)
    let mut shells: Vec<&Shell> = basis.shells.iter().collect();
    shells.sort_by_key(|s| s.atom);
    let mut out = String::new();
    let mut line = |s: &str| {
        out.push_str(s);
        out.push('\n');
    };
    line("[Molden Format]");
    line("[Title]");
    line(" written by linus");
    line("[Atoms] AU");
    let n_core = |i: usize| core.get(i).copied().unwrap_or(0);
    for (i, ((sym, &z), r)) in symbols.iter().zip(z).zip(positions).enumerate() {
        line(&format!(
            "{sym:<2} {:5} {:4} {:22.14} {:22.14} {:22.14}",
            i + 1,
            i64::from(z) - i64::from(n_core(i)),
            r[0],
            r[1],
            r[2]
        ));
    }
    line("[GTO]");
    for atom in 0..symbols.len() {
        line(&format!("{:5} 0", atom + 1));
        for s in shells.iter().filter(|s| s.atom == atom) {
            line(&format!(" {} {:4} 1.00", letter(s.l), s.exps.len()));
            for (&a, &c) in s.exps.iter().zip(&s.coefs) {
                line(&format!(" {} {}", e22(a), e22(c / xl_norm(s.l, a))));
            }
        }
        line("");
    }
    // the spherical flags of the shells with l >= 2 (Molden's default is Cartesian)
    let pure = |l: usize| {
        shells
            .iter()
            .any(|s| s.l == l)
            .then_some(!cartesian && shells.iter().any(|s| s.l == l && s.pure))
    };
    let (d, f, g) = (pure(2), pure(3), pure(4));
    if d == Some(true) && f != Some(false) {
        line("[5D7F]");
    } else if d == Some(true) {
        line("[5D10F]");
    } else if f == Some(true) {
        line("[7F]");
    }
    if g == Some(true) {
        line("[9G]");
    }
    if (0..symbols.len()).any(|i| n_core(i) > 0) {
        line("[core]");
        for i in (0..symbols.len()).filter(|&i| n_core(i) > 0) {
            line(&format!("{} : {}", i + 1, n_core(i)));
        }
    }
    line("[MO]");
    // each Molden function of each shell as terms (AO index, weight) of linus's functions
    let terms: Vec<Vec<(usize, f64)>> = shells
        .iter()
        .flat_map(|s| shell_terms(s, cartesian))
        .collect();
    for set in sets {
        for j in 0..set.c.ncols() {
            line(" Sym= A");
            line(&format!(" Ene= {:.12}", set.energies[j]));
            line(&format!(" Spin= {}", set.spin));
            line(&format!(" Occup= {:.8}", set.occupations[j]));
            for (i, t) in terms.iter().enumerate() {
                let v = match t.as_slice() {
                    // a function of linus's set, reordered and rescaled
                    [(k, w)] => set.c[(*k, j)] * w,
                    _ => t.iter().map(|&(k, w)| set.c[(k, j)] * w).sum(),
                };
                line(&format!(" {:5} {}", i + 1, e22(v)));
            }
        }
    }
    out
}

/// The Molden functions of shell `s` (Cartesian ones with `cartesian`, else the shell's own
/// kind) as terms (AO index, weight) of the shell's functions in linus's convention: spherical
/// m = -l..l (p as x, y, z), Cartesian components in lexicographic order normalized as x^l.
fn shell_terms(s: &Shell, cartesian: bool) -> Vec<Vec<(usize, f64)>> {
    let l = s.l;
    if s.pure && !cartesian {
        return molden_ms(l)
            .into_iter()
            .map(|m| vec![(s.ao_start + linus_m_index(l, m), 1.0)])
            .collect();
    }
    let lex = lexicographic(l);
    let nc = ncart(l);
    // the Cartesian component of the x^l norm in a unit-normalized one
    let norm = |(lx, ly, lz): (usize, usize, usize)| {
        (dfact(2 * lx as i64 - 1) * dfact(2 * ly as i64 - 1) * dfact(2 * lz as i64 - 1)
            / dfact(2 * l as i64 - 1))
        .sqrt()
    };
    molden_cartesian(l)
        .into_iter()
        .map(|c| {
            let i = lex
                .iter()
                .position(|&x| x == c)
                .expect("a component of the shell");
            if !s.pure {
                vec![(s.ao_start + i, norm(c))]
            } else {
                // a spherical function j is sum_i T[j, i] (Cartesian i): its coefficient adds
                // T[j, i] to Cartesian i's
                let t = BasisSet::shell_transform_ref(s);
                (0..s.nfunc())
                    .filter(|&j| t[j * nc + i] != 0.0)
                    .map(|j| (s.ao_start + j, t[j * nc + i] * norm(c)))
                    .collect()
            }
        })
        .collect()
}

/// Molden's order of the spherical functions: m = 0, +1, -1, +2, -2, ..., p as x, y, z.
fn molden_ms(l: usize) -> Vec<i64> {
    if l == 1 {
        return vec![1, -1, 0];
    }
    let mut ms = vec![0];
    for m in 1..=l as i64 {
        ms.extend([m, -m]);
    }
    ms
}

/// The index of spherical function m in linus's order (m = -l..l; p as x, y, z).
fn linus_m_index(l: usize, m: i64) -> usize {
    if l == 1 {
        return match m {
            1 => 0,
            -1 => 1,
            _ => 2,
        };
    }
    (m + l as i64) as usize
}

/// The Cartesian components (lx, ly, lz) in lexicographic order (lx, then ly descending).
fn lexicographic(l: usize) -> Vec<(usize, usize, usize)> {
    (0..=l)
        .rev()
        .flat_map(|lx| (0..=l - lx).rev().map(move |ly| (lx, ly, l - lx - ly)))
        .collect()
}

/// Molden's order of the Cartesian components (up to g).
fn molden_cartesian(l: usize) -> Vec<(usize, usize, usize)> {
    let table = match l {
        2 => "xx yy zz xy xz yz",
        3 => "xxx yyy zzz xyy xxy xxz xzz yzz yyz xyz",
        4 => "xxxx yyyy zzzz xxxy xxxz yyyx yyyz zzzx zzzy xxyy xxzz yyzz xxyz yyxz zzxy",
        _ => return lexicographic(l),
    };
    let count = |c: &str, a: char| c.chars().filter(|&x| x == a).count();
    table
        .split(' ')
        .map(|c| (count(c, 'x'), count(c, 'y'), count(c, 'z')))
        .collect()
}

fn letter(l: usize) -> String {
    L_LETTERS
        .get(l)
        .map_or_else(|| format!("l{l}"), |c| c.to_string())
}

/// (n)!! with (-1)!! = 1.
fn dfact(n: i64) -> f64 {
    let mut r = 1.0;
    let mut n = n;
    while n > 1 {
        r *= n as f64;
        n -= 2;
    }
    r
}

/// The normalization constant of the primitive x^l exp(-a r^2) (its powers as NumPy takes
/// them in linus-py's writer, so the two files agree to the digit).
fn xl_norm(l: usize, a: f64) -> f64 {
    let x = 4.0 * a;
    let xl = match l {
        0 => 1.0,
        1 => x,
        2 => x * x,
        _ => x.powf(l as f64),
    };
    ((2.0 * a / std::f64::consts::PI).powf(1.5) * xl / dfact(2 * l as i64 - 1)).sqrt()
}

/// `v` as Python's `f"{v:22.14e}"` (a two-digit exponent at least).
fn e22(v: f64) -> String {
    let s = format!("{v:.14e}");
    let t = match s.split_once('e') {
        Some((m, e)) => {
            let e: i32 = e.parse().expect("an exponent");
            let mut t = String::with_capacity(22);
            let _ = write!(t, "{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
            t
        }
        None => s,
    };
    format!("{t:>22}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_format() {
        assert_eq!(e22(0.0), "  0.00000000000000e+00");
        assert_eq!(e22(-1.5e-7), " -1.50000000000000e-07");
        assert_eq!(e22(12345.678), "  1.23456780000000e+04");
        assert_eq!(e22(1e-300), " 1.00000000000000e-300");
    }

    #[test]
    fn orders() {
        assert_eq!(molden_ms(2), [0, 1, -1, 2, -2]);
        assert_eq!(molden_ms(1), [1, -1, 0]);
        assert_eq!(linus_m_index(2, -2), 0);
        assert_eq!(linus_m_index(3, 0), 3);
        assert_eq!(
            lexicographic(2),
            [
                (2, 0, 0),
                (1, 1, 0),
                (1, 0, 1),
                (0, 2, 0),
                (0, 1, 1),
                (0, 0, 2)
            ]
        );
        for l in 0..=LMAX {
            let mut a = molden_cartesian(l);
            let mut b = lexicographic(l);
            a.sort();
            b.sort();
            assert_eq!(a, b, "l = {l}: Molden's components are the shell's");
        }
    }

    /// A spherical shell written Cartesian: the expansion of each spherical function in the
    /// unit-normalized components reproduces it (d: x^2 - y^2 and xy).
    #[test]
    fn spherical_as_cartesian() {
        let s = Shell {
            l: 2,
            pure: true,
            atom: 0,
            center: [0.0; 3],
            exps: vec![1.0],
            coefs: vec![1.0],
            ao_start: 0,
        };
        let terms = shell_terms(&s, true);
        assert_eq!(terms.len(), 6);
        // xy (Molden component 3) is linus's m = -2 alone: sqrt(3) x (1/sqrt(3))
        assert_eq!(terms[3].len(), 1);
        assert_eq!(terms[3][0].0, 0);
        assert!((terms[3][0].1 - 1.0).abs() < 1e-14);
        // xx (component 0) takes m = 0 (-1/2) and m = +2 (sqrt(3)/2)
        let w: Vec<(usize, f64)> = terms[0].clone();
        assert_eq!(w.iter().map(|t| t.0).collect::<Vec<_>>(), [2, 4]);
        assert!((w[0].1 + 0.5).abs() < 1e-14 && (w[1].1 - 3f64.sqrt() / 2.0).abs() < 1e-14);
    }
}
