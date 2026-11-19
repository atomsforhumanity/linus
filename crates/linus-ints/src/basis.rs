//! Basis sets: Basis Set Exchange JSON input, normalization, shell layout, and the basis set
//! library (loading by name with composites `a|b`, listing, did-you-mean hints, file names).
//!
//! Conventions (see docs/conventions.md):
//! - Cartesian components of a shell are ordered xx, xy, xz, yy, yz, zz (lx descending, then ly
//!   descending), as in libint, libcint and PySCF.
//! - Spherical (pure) functions are real solid harmonics ordered m = -l..l, except p shells,
//!   which keep the Cartesian order x, y, z (as in PySCF).
//! - Stored contraction coefficients include the primitive normalization for the x^l
//!   component and the contraction normalization, so that every spherical function and the
//!   x^l, y^l, z^l Cartesian components have unit norm.

use crate::core::solid_harmonics::cart2sph;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum BasisError {
    #[error("basis set '{basis}' has no functions for element Z={z}")]
    MissingElement { basis: String, z: u32 },
    /// No file of the set in any directory of the search path ([`search_path`]).
    #[error(
        "unknown basis set '{name}': no file {file} in {}",
        display_path(searched)
    )]
    NotFound {
        name: String,
        file: String,
        searched: Vec<PathBuf>,
    },
    #[error("cannot read basis set: {0}")]
    Io(#[from] std::io::Error),
    #[error("cannot parse basis set JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported basis set content: {0}")]
    Unsupported(String),
    /// Content no basis set can have (a non-positive or non-finite exponent, a non-finite
    /// coefficient, inconsistent lengths).
    #[error("invalid basis set content: {0}")]
    Invalid(String),
}

/// One contracted shell (segmented; general contractions are split into one shell per
/// contracted function, with zero coefficients removed).
#[derive(Clone, Debug)]
pub struct Shell {
    pub l: usize,
    pub pure: bool,
    pub atom: usize,
    pub center: [f64; 3],
    pub exps: Vec<f64>,
    /// Normalized contraction coefficients (primitive and contraction normalization included).
    pub coefs: Vec<f64>,
    /// Index of the first basis function of this shell.
    pub ao_start: usize,
}

impl Shell {
    #[inline]
    pub fn nfunc(&self) -> usize {
        if self.pure {
            2 * self.l + 1
        } else {
            ncart(self.l)
        }
    }
    #[inline]
    pub fn nprim(&self) -> usize {
        self.exps.len()
    }
}

#[inline]
pub const fn ncart(l: usize) -> usize {
    (l + 1) * (l + 2) / 2
}

/// Cartesian exponents of component `i` of a shell with angular momentum `l`.
pub fn cart_components(l: usize) -> Vec<[usize; 3]> {
    let mut v = Vec::with_capacity(ncart(l));
    for lx in (0..=l).rev() {
        for ly in (0..=l - lx).rev() {
            v.push([lx, ly, l - lx - ly]);
        }
    }
    v
}

/// A shell definition independent of position (what a basis set file holds for an element).
#[derive(Clone, Debug, PartialEq)]
pub struct ShellDef {
    pub l: usize,
    pub exps: Vec<f64>,
    /// Coefficients of normalized primitives, as in basis set files.
    pub coefs: Vec<f64>,
}

/// Element-resolved basis set definition (one entry per element).
#[derive(Clone, Debug, Default)]
pub struct BasisDef {
    pub name: String,
    pub elements: HashMap<u32, Vec<ShellDef>>,
    /// Number of core electrons replaced by an ECP, per element.
    pub ecp_electrons: HashMap<u32, u32>,
    /// Effective core potentials, per element (the elements of `ecp_electrons`).
    pub ecp: HashMap<u32, crate::one_e::ecp::EcpDef>,
}

#[derive(Deserialize)]
struct BseBasis {
    #[serde(default)]
    name: Option<String>,
    elements: HashMap<String, BseElement>,
}

#[derive(Deserialize)]
struct BseElement {
    #[serde(default)]
    electron_shells: Option<Vec<BseShell>>,
    #[serde(default)]
    ecp_electrons: Option<u32>,
    #[serde(default)]
    ecp_potentials: Option<Vec<BseEcp>>,
}

/// One angular-momentum channel of an ECP in the Basis Set Exchange format: terms
/// c r^(n-2) exp(-zeta r^2) with n = `r_exponents` (the Gaussian/NWChem convention).
#[derive(Deserialize)]
struct BseEcp {
    ecp_type: String,
    angular_momentum: Vec<usize>,
    r_exponents: Vec<i32>,
    gaussian_exponents: Vec<String>,
    coefficients: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct BseShell {
    #[allow(dead_code)]
    function_type: String,
    angular_momentum: Vec<usize>,
    exponents: Vec<String>,
    coefficients: Vec<Vec<String>>,
}

/// Exponents must be positive and finite (a zero or negative one makes the integrals
/// meaningless, and the Boys function arguments negative), coefficients finite, and every
/// coefficient row as long as the exponents.
fn check_primitives(z: u32, exps: &[f64], rows: &[Vec<f64>]) -> Result<(), BasisError> {
    if let Some(e) = exps.iter().find(|e| !(e.is_finite() && **e > 0.0)) {
        return Err(BasisError::Invalid(format!(
            "element Z={z}: exponent {e} (exponents must be positive and finite)"
        )));
    }
    for r in rows {
        if r.len() != exps.len() {
            return Err(BasisError::Invalid(format!(
                "element Z={z}: a shell with {} exponents and a row of {} coefficients",
                exps.len(),
                r.len()
            )));
        }
        if let Some(c) = r.iter().find(|c| !c.is_finite()) {
            return Err(BasisError::Invalid(format!(
                "element Z={z}: coefficient {c} (coefficients must be finite)"
            )));
        }
    }
    Ok(())
}

fn parse_f64(s: &str) -> Result<f64, BasisError> {
    s.trim()
        .replace(['D', 'd'], "E")
        .parse::<f64>()
        .map_err(|e| BasisError::Unsupported(format!("bad number '{s}': {e}")))
}

impl BasisDef {
    /// Parse the Basis Set Exchange JSON format (`basis_set_exchange.get_basis(name, fmt="json")`).
    pub fn from_bse_json(text: &str) -> Result<Self, BasisError> {
        let b: BseBasis = serde_json::from_str(text)?;
        let mut def = BasisDef {
            name: b.name.unwrap_or_default(),
            ..Default::default()
        };
        for (zs, el) in b.elements {
            let z: u32 = zs
                .parse()
                .map_err(|_| BasisError::Unsupported(format!("element key '{zs}'")))?;
            if let Some(n) = el.ecp_electrons {
                if n > 0 {
                    def.ecp_electrons.insert(z, n);
                }
            }
            if let Some(pots) = &el.ecp_potentials {
                let ecp = parse_ecp(z, el.ecp_electrons.unwrap_or(0), pots)?;
                def.ecp.insert(z, ecp);
            } else if el.ecp_electrons.unwrap_or(0) > 0 {
                return Err(BasisError::Unsupported(format!(
                    "element Z={z} replaces {} core electrons but has no ECP potentials",
                    el.ecp_electrons.unwrap_or(0)
                )));
            }
            let mut shells = Vec::new();
            for sh in el.electron_shells.unwrap_or_default() {
                let exps: Vec<f64> = sh
                    .exponents
                    .iter()
                    .map(|s| parse_f64(s))
                    .collect::<Result<_, _>>()?;
                let rows: Vec<Vec<f64>> = sh
                    .coefficients
                    .iter()
                    .map(|r| {
                        r.iter()
                            .map(|s| parse_f64(s))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<_, _>>()?;
                check_primitives(z, &exps, &rows)?;
                // SP-type shells: one row per angular momentum; general contraction: several
                // rows for a single angular momentum.
                let ls: Vec<usize> = if sh.angular_momentum.len() == 1 {
                    vec![sh.angular_momentum[0]; rows.len()]
                } else if sh.angular_momentum.len() == rows.len() {
                    sh.angular_momentum.clone()
                } else {
                    return Err(BasisError::Unsupported(format!(
                        "shell with angular momenta {:?} and {} coefficient rows",
                        sh.angular_momentum,
                        rows.len()
                    )));
                };
                for (l, row) in ls.into_iter().zip(rows) {
                    let (e, c): (Vec<f64>, Vec<f64>) = exps
                        .iter()
                        .zip(row.iter())
                        .filter(|(_, c)| **c != 0.0)
                        .map(|(e, c)| (*e, *c))
                        .unzip();
                    if !e.is_empty() {
                        shells.push(ShellDef {
                            l,
                            exps: e,
                            coefs: c,
                        });
                    }
                }
            }
            // Order shells by angular momentum within an element (stable), as PySCF, Psi4 and
            // ORCA do; basis set files do not always list them that way (cc-pVTZ Cl: s p f d).
            shells.sort_by_key(|s| s.l);
            def.elements.insert(z, shells);
        }
        Ok(def)
    }

    /// Load a basis set by name from a directory of BSE JSON files (`<dir>/<stem>.json`,
    /// [`basis_file_stem`]); None: the default search path ([`search_path`]).
    pub fn load(name: &str, dir: Option<&Path>) -> Result<Self, BasisError> {
        match dir {
            Some(d) => Self::load_from(name, &[d.to_path_buf()]),
            None => Self::load_from(name, &search_path(&[])),
        }
    }

    /// Load a basis set by name from the first directory of `path` that has its file.
    pub fn load_from(name: &str, path: &[PathBuf]) -> Result<Self, BasisError> {
        let file = find_basis_file(name, path)?;
        let text = std::fs::read_to_string(&file)?;
        let mut def = Self::from_bse_json(&text)?;
        if def.name.is_empty() {
            def.name = name.to_string();
        }
        Ok(def)
    }

    /// Load a basis set (orbital or auxiliary) by name from the search path `path`
    /// ([`BasisDef::load_from`]); `"a|b|..."` is a composite: each element (with its ECP) from
    /// the first set that has it.
    pub fn load_composite(name: &str, path: &[PathBuf]) -> Result<Self, BasisError> {
        let mut parts = name.split('|');
        let mut def = Self::load_from(parts.next().unwrap_or(name), path)?;
        for p in parts {
            let more = Self::load_from(p, path)?;
            for (z, shells) in more.elements {
                if !def.elements.contains_key(&z) {
                    if let Some(&n) = more.ecp_electrons.get(&z) {
                        def.ecp_electrons.insert(z, n);
                    }
                    if let Some(e) = more.ecp.get(&z) {
                        def.ecp.insert(z, e.clone());
                    }
                    def.elements.insert(z, shells);
                }
            }
        }
        if name.contains('|') {
            def.name = name.to_string();
        }
        Ok(def)
    }
}

/// The ECP of one element from its Basis Set Exchange channels: the channel with the highest
/// angular momentum is the local part U_L, the others hold U_l - U_L (NWChem's "ul" convention,
/// shared by PySCF, Psi4 and ORCA's def2 ECPs).
fn parse_ecp(z: u32, ncore: u32, pots: &[BseEcp]) -> Result<crate::one_e::ecp::EcpDef, BasisError> {
    use crate::one_e::ecp::{EcpDef, EcpTerm};
    if pots.is_empty() {
        return Err(BasisError::Unsupported(format!("element Z={z}: empty ECP")));
    }
    let mut lmax = 0;
    for p in pots {
        if p.ecp_type != "scalar_ecp" {
            return Err(BasisError::Unsupported(format!(
                "element Z={z}: ECP type {:?} (only scalar ECPs are supported)",
                p.ecp_type
            )));
        }
        if p.angular_momentum.len() != 1 || p.coefficients.len() != 1 {
            return Err(BasisError::Unsupported(format!(
                "element Z={z}: ECP channel with angular momenta {:?}",
                p.angular_momentum
            )));
        }
        lmax = lmax.max(p.angular_momentum[0]);
    }
    let mut def = EcpDef {
        ncore,
        local: Vec::new(),
        semilocal: vec![Vec::new(); lmax],
    };
    for p in pots {
        let l = p.angular_momentum[0];
        let (n, e, c) = (&p.r_exponents, &p.gaussian_exponents, &p.coefficients[0]);
        if n.len() != e.len() || e.len() != c.len() {
            return Err(BasisError::Unsupported(format!(
                "element Z={z}: ECP channel l={l} with inconsistent term counts"
            )));
        }
        for ((&n, e), c) in n.iter().zip(e).zip(c) {
            let t = EcpTerm {
                rpow: n - 2,
                zeta: parse_f64(e)?,
                coef: parse_f64(c)?,
            };
            if !(t.zeta.is_finite() && t.zeta > 0.0) || !t.coef.is_finite() {
                return Err(BasisError::Invalid(format!(
                    "element Z={z}: ECP channel l={l} term with exponent {} and coefficient {} \
                     (exponents must be positive and finite, coefficients finite)",
                    t.zeta, t.coef
                )));
            }
            if t.coef == 0.0 {
                continue;
            }
            if l == lmax {
                def.local.push(t);
            } else {
                def.semilocal[l].push(t);
            }
        }
    }
    Ok(def)
}

/// Characters of basis set names that file names avoid, and what they become: `*`, `+` and `/`
/// as the Basis Set Exchange writes them, `(`, `)` and `,` because a wheel's RECORD (CSV) and
/// some file systems and shells handle them badly (6-311+G(2d,p) is 6-311_pl_g_lp_2d_cm_p_rp_).
const FILE_ESCAPES: [(char, &str); 6] = [
    ('*', "_st_"),
    ('+', "_pl_"),
    ('/', "_sl_"),
    ('(', "_lp_"),
    (')', "_rp_"),
    (',', "_cm_"),
];

/// File name (without `.json`) of the basis set `name`: lower-cased, with `*`, `+`, `/`, `(`,
/// `)` and `,` written as `_st_`, `_pl_`, `_sl_`, `_lp_`, `_rp_` and `_cm_`.
pub fn basis_file_stem(name: &str) -> String {
    let mut s = name.to_lowercase();
    for (c, e) in FILE_ESCAPES {
        s = s.replace(c, e);
    }
    s
}

/// The basis set name of a file stem (the inverse of [`basis_file_stem`], lower-case).
pub fn basis_name_from_stem(stem: &str) -> String {
    let mut s = stem.to_string();
    for (c, e) in FILE_ESCAPES {
        s = s.replace(e, &c.to_string());
    }
    s
}

/// The file of basis set `name` in the first directory of `path` that has it.
pub fn find_basis_file(name: &str, path: &[PathBuf]) -> Result<PathBuf, BasisError> {
    let file = format!("{}.json", basis_file_stem(name));
    path.iter()
        .map(|d| d.join(&file))
        .find(|f| f.is_file())
        .ok_or_else(|| BasisError::NotFound {
            name: name.to_string(),
            file,
            searched: path.to_vec(),
        })
}

/// Names of the basis sets (orbital and auxiliary) in the directories of `path`, sorted, each
/// once.
pub fn available_basis_sets(path: &[PathBuf]) -> Vec<String> {
    let mut names: Vec<String> = path
        .iter()
        .filter_map(|d| std::fs::read_dir(d).ok())
        .flat_map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| {
                    e.file_name()
                        .to_str()
                        .and_then(|f| f.strip_suffix(".json"))
                        .map(basis_name_from_stem)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    names.sort();
    names.dedup();
    names
}

fn display_path(path: &[PathBuf]) -> String {
    if path.is_empty() {
        return "an empty search path".into();
    }
    path.iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Levenshtein distance of two strings (for did-you-mean hints).
pub fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + (ca != *cb) as usize)
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// The closest of `names` to `name` (by edit distance, at most a third of its length), as
/// " Did you mean 'X'?" (empty if none is close): for unknown basis set names
/// ([`available_basis_sets`]), and in linus for other names (solvents).
pub fn did_you_mean(name: &str, names: &[String]) -> String {
    let key = name.to_ascii_lowercase();
    let max = (key.len() / 3).max(2);
    let mut best: Vec<(usize, &String)> = names
        .iter()
        .map(|n| (edit_distance(&key, &n.to_ascii_lowercase()), n))
        .filter(|(d, _)| *d <= max)
        .collect();
    best.sort();
    match &best[..] {
        [] => String::new(),
        [(_, a)] => format!(" Did you mean '{a}'?"),
        [(_, a), (_, b), ..] => format!(" Did you mean '{a}' or '{b}'?"),
    }
}

/// The root of the source checkout in development builds (a compile-time path; its basis sets
/// are in `data/basis`); None in distributed builds, which are compiled with `LINUS_DIST` set
/// and so carry no paths of the machine that built them. (linus-grid has a twin for its data.)
pub fn dev_source_dir() -> Option<PathBuf> {
    DEV_SOURCE_DIR.map(|d| {
        let p = PathBuf::from(d);
        p.canonicalize().unwrap_or(p)
    })
}

const DEV_SOURCE_DIR: Option<&str> = match option_env!("LINUS_DIST") {
    Some(_) => None,
    None => Some(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")),
};

static PACKAGE_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Register the basis sets an installation ships (the Python package's `basis` directory; the
/// binding sets it once, at import). False if one was registered already.
pub fn set_package_dir(dir: PathBuf) -> bool {
    PACKAGE_DIR.set(dir).is_ok()
}

/// The basis sets of the installation: the registered package directory
/// ([`set_package_dir`]), else `<prefix>/share/linus/basis` for a program installed as
/// `<prefix>/bin/linus`.
pub fn package_dir() -> Option<PathBuf> {
    if let Some(d) = PACKAGE_DIR.get() {
        return Some(d.clone());
    }
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.parent()?.join("share/linus/basis"))
}

/// The basis set search path: `explicit` (the `basis_dir` setting), then `$LINUS_BASIS_DIR`
/// (one or more directories, separated as in `PATH`), the source tree of development builds
/// (`data/basis` of [`dev_source_dir`]) and the installation's sets ([`package_dir`]); each
/// directory once.
/// A set is read from the first directory that has its file.
pub fn search_path(explicit: &[PathBuf]) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = explicit.to_vec();
    if let Some(env) = std::env::var_os("LINUS_BASIS_DIR") {
        v.extend(std::env::split_paths(&env).filter(|p| !p.as_os_str().is_empty()));
    }
    v.extend(dev_source_dir().map(|d| d.join("data/basis")));
    v.extend(package_dir());
    let mut out: Vec<PathBuf> = Vec::with_capacity(v.len());
    for p in v {
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

/// Double factorial (n)!! with (-1)!! = 1.
pub fn dfact(n: i64) -> f64 {
    let mut r = 1.0;
    let mut k = n;
    while k > 1 {
        r *= k as f64;
        k -= 2;
    }
    r
}

/// Normalize a shell: returns coefficients for unnormalized primitives r^l exp(-a r^2)
/// such that the x^l component (and hence every spherical function) has unit norm.
pub fn normalize_shell(l: usize, exps: &[f64], coefs: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    let li = l as i64;
    let df = dfact(2 * li - 1);
    // primitive norms for x^l exp(-a r^2): N^2 = (2a/pi)^{3/2} (4a)^l / (2l-1)!!
    let c: Vec<f64> = exps
        .iter()
        .zip(coefs)
        .map(|(&a, &d)| {
            let n2 = (2.0 * a / PI).powf(1.5) * (4.0 * a).powi(l as i32) / df;
            d * n2.sqrt()
        })
        .collect();
    // contraction norm: <x^l|x^l> = sum_ij c_i c_j (pi/p)^{3/2} (2l-1)!! / (2p)^l
    let mut s = 0.0;
    for (i, &ai) in exps.iter().enumerate() {
        for (j, &aj) in exps.iter().enumerate() {
            let p = ai + aj;
            s += c[i] * c[j] * (PI / p).powf(1.5) * df / (2.0 * p).powi(l as i32);
        }
    }
    let f = 1.0 / s.sqrt();
    c.into_iter().map(|x| x * f).collect()
}

/// A basis set placed on a molecule.
#[derive(Clone, Debug)]
pub struct BasisSet {
    pub name: String,
    pub shells: Vec<Shell>,
    pub nbf: usize,
    /// Shell index range of each atom.
    pub atom_shells: Vec<std::ops::Range<usize>>,
    pub pure: bool,
}

impl BasisSet {
    /// Place `def` on atoms with atomic numbers `z` at `positions` (bohr).
    pub fn new(
        def: &BasisDef,
        z: &[u32],
        positions: &[[f64; 3]],
        pure: bool,
    ) -> Result<Self, BasisError> {
        let mut shells = Vec::new();
        let mut atom_shells = Vec::new();
        let mut ao = 0;
        for (atom, (&zi, &r)) in z.iter().zip(positions).enumerate() {
            let start = shells.len();
            if zi == 0 {
                atom_shells.push(start..start);
                continue; // ghost-free point charge or dummy
            }
            let defs = def
                .elements
                .get(&zi)
                .ok_or_else(|| BasisError::MissingElement {
                    basis: def.name.clone(),
                    z: zi,
                })?;
            for sd in defs {
                let sh = Shell {
                    l: sd.l,
                    pure: pure && sd.l >= 2,
                    atom,
                    center: r,
                    exps: sd.exps.clone(),
                    coefs: normalize_shell(sd.l, &sd.exps, &sd.coefs),
                    ao_start: ao,
                };
                ao += sh.nfunc();
                shells.push(sh);
            }
            atom_shells.push(start..shells.len());
        }
        Ok(BasisSet {
            name: def.name.clone(),
            shells,
            nbf: ao,
            atom_shells,
            pure,
        })
    }

    pub fn max_l(&self) -> usize {
        self.shells.iter().map(|s| s.l).max().unwrap_or(0)
    }

    /// The number of atoms the basis was placed on (atoms without functions included).
    pub fn natom(&self) -> usize {
        self.atom_shells.len()
    }

    /// The functions of the shells on `atom` (contiguous; empty for an atom without shells or
    /// beyond the basis' atoms).
    pub fn atom_functions(&self, atom: usize) -> std::ops::Range<usize> {
        let r = self.atom_shells.get(atom).cloned().unwrap_or(0..0);
        if r.is_empty() {
            return 0..0;
        }
        let (first, last) = (&self.shells[r.start], &self.shells[r.end - 1]);
        first.ao_start..last.ao_start + last.nfunc()
    }

    /// Move the basis to new coordinates (same atoms, same order).
    pub fn set_positions(&mut self, positions: &[[f64; 3]]) {
        for s in &mut self.shells {
            s.center = positions[s.atom];
        }
    }

    /// Transformation from Cartesian components (normalized as x^l) to the functions of a
    /// shell: returns a (nfunc x ncart) row-major matrix.
    /// Cached version of [`Self::shell_transform`] (no allocation).
    pub fn shell_transform_ref(sh: &Shell) -> &'static [f64] {
        use std::sync::OnceLock;
        static ID: OnceLock<Vec<Vec<f64>>> = OnceLock::new();
        if sh.pure {
            crate::core::solid_harmonics::cart2sph_cached(sh.l)
        } else {
            &ID.get_or_init(|| (0..=12).map(cart_norm_matrix).collect())[sh.l]
        }
    }

    pub fn shell_transform(sh: &Shell) -> Vec<f64> {
        if sh.pure {
            cart2sph(sh.l)
        } else {
            cart_norm_matrix(sh.l)
        }
    }
}

/// Diagonal matrix that scales x^lx y^ly z^lz (normalized as x^l) to... unchanged: we keep the
/// PySCF/libcint convention in which every Cartesian component of a shell carries the x^l
/// normalization. Identity (ncart x ncart).
pub fn cart_norm_matrix(l: usize) -> Vec<f64> {
    let n = ncart(l);
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_small_bse() {
        let json = r#"{"name":"test","elements":{"1":{"electron_shells":[
            {"function_type":"gto","angular_momentum":[0],"exponents":["3.0","0.5"],"coefficients":[["0.3","0.7"],["0.0","1.0"]]},
            {"function_type":"gto","angular_momentum":[0,1],"exponents":["1.0"],"coefficients":[["1.0"],["1.0"]]}
        ]}}}"#;
        let d = BasisDef::from_bse_json(json).unwrap();
        let h = &d.elements[&1];
        assert_eq!(h.len(), 4);
        assert_eq!(h[1].exps, vec![0.5]);
        assert_eq!(h[3].l, 1);
    }

    #[test]
    fn rejects_bad_primitives() {
        let shell = |e: &str, c: &str| {
            format!(
                r#"{{"elements":{{"1":{{"electron_shells":[{{"function_type":"gto","angular_momentum":[0],"exponents":[{e}],"coefficients":[[{c}]]}}]}}}}}}"#
            )
        };
        for (e, c) in [
            (r#""-1.0""#, r#""1.0""#),
            (r#""0.0""#, r#""1.0""#),
            (r#""NaN""#, r#""1.0""#),
            (r#""inf""#, r#""1.0""#),
            (r#""1.0""#, r#""NaN""#),
            (r#""1.0","2.0""#, r#""1.0""#),
        ] {
            let r = BasisDef::from_bse_json(&shell(e, c));
            assert!(matches!(r, Err(BasisError::Invalid(_))), "{e} {c}: {r:?}");
        }
        assert!(BasisDef::from_bse_json(&shell(r#""1.0""#, r#""1.0""#)).is_ok());
    }

    /// File names avoid `*`, `+`, `/`, `(`, `)` and `,`; listing maps them back.
    #[test]
    fn file_stems() {
        assert_eq!(
            basis_file_stem("6-311+G(2d,p)"),
            "6-311_pl_g_lp_2d_cm_p_rp_"
        );
        assert_eq!(basis_file_stem("6-31G**"), "6-31g_st__st_");
        for name in ["6-311+g(2d,p)", "6-31+g*", "def2-svp", "a/b"] {
            assert_eq!(basis_name_from_stem(&basis_file_stem(name)), name);
        }
        let d = BasisDef::load("6-311+G(2d,p)", None).unwrap();
        assert!(d.elements.contains_key(&8));
    }
}
